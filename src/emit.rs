//! JSON emission kernels: SIMD escape scanning and pinned-format number
//! writing. Pure computation over byte buffers: [`crate::Writer`] layers
//! grammar state on top; hosts with bespoke needs can call these directly.
//!
//! The byte-classification primitives (hit masks, class table, padded
//! tail staging) are shared with the parser and live in the
//! crate-internal `scan` module.

use crate::scan::{
    CLASS_STANDARD, ESCAPE_CLASS, MODE_ASCII_ONLY, MODE_SCRIPT_SAFE, MODE_STANDARD, class_bits,
};

/// String escaping variants.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EscapeMode {
    /// Escape `"`, `\` and control characters (the RFC 8259 minimum).
    #[default]
    Standard,
    /// Additionally escape `/` and U+2028/U+2029, making the output safe to
    /// embed in HTML `<script>` blocks and pre-ES2019 JavaScript source.
    ScriptSafe,
    /// Additionally escape all non-ASCII as `\uXXXX` (surrogate pairs for
    /// astral code points), producing 7-bit-clean output.
    AsciiOnly,
}

/// Escaping for hosts without SIMD kernels; SIMD targets handle entire
/// strings inside the fused loops and never come here.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod fallback {
    use super::{ESCAPE_CLASS, MAX_ESCAPE_EXPANSION, class_bits, push_short, write_escape_run_raw};

    /// SWAR-scan for the next escape, copy the clean span, write the
    /// escape run, repeat.
    pub(super) fn escape<const MODE: u8>(out: &mut Vec<u8>, s: &[u8]) {
        let mut i = 0;
        while i < s.len() {
            let next = find_escape::<MODE>(s, i);
            if next - i <= 16 {
                push_short(out, &s[i..next]);
            } else {
                out.extend_from_slice(&s[i..next]);
            }
            if next >= s.len() {
                return;
            }
            i = write_escape_run::<MODE>(out, s, next);
        }
    }

    /// Find the next byte at or after `from` that needs attention under
    /// `MODE`: SWAR 8 bytes per step, then the class table per byte.
    #[inline(always)]
    fn find_escape<const MODE: u8>(s: &[u8], mut from: usize) -> usize {
        while from + 8 <= s.len() {
            let w = u64::from_le_bytes(s[from..from + 8].try_into().unwrap());
            let m = crate::scan::swar_escape_mask::<MODE>(w);
            if m != 0 {
                return from + (m.trailing_zeros() >> 3) as usize;
            }
            from += 8;
        }
        while from < s.len() {
            if ESCAPE_CLASS[s[from] as usize] & class_bits::<MODE>() != 0 {
                return from;
            }
            from += 1;
        }
        from
    }

    /// Vec-front wrapper for [`write_escape_run_raw`]: reserves the run's
    /// worst case, commits with one `set_len`.
    fn write_escape_run<const MODE: u8>(out: &mut Vec<u8>, s: &[u8], i: usize) -> usize {
        out.reserve(MAX_ESCAPE_EXPANSION * (s.len() - i) + 8);
        // SAFETY: worst case reserved above; `set_len` covers only the
        // bytes the run wrote.
        unsafe {
            let dst = out.as_mut_ptr().add(out.len());
            let (next_i, wrote) = write_escape_run_raw::<MODE>(s, i, dst);
            out.set_len(out.len() + wrote);
            next_i
        }
    }
}

/// Append a short slice (≤ 16 bytes) with overlapping word stores instead of
/// a `memcpy` PLT call: tiny-copy call overhead showed up at 14% on
/// string-heavy generation profiles. Longer slices take a plain
/// `extend_from_slice` (hot callers prove the bound; the fallback keeps
/// this safe for everyone else).
#[inline(always)]
pub fn push_short(out: &mut Vec<u8>, s: &[u8]) {
    let n = s.len();
    if n > 16 {
        out.extend_from_slice(s);
        return;
    }
    out.reserve(16);
    // SAFETY: `n <= 16` readable bytes in `s`, 16 writable bytes reserved
    // above; `set_len` covers exactly the `n` written bytes.
    unsafe {
        crate::scan::copy_small(s.as_ptr(), out.as_mut_ptr().add(out.len()), n);
        out.set_len(out.len() + n);
    }
}

/// Shared body of every fused scan+store loop, regardless of vector width:
/// each chunk is speculatively stored into reserved capacity as it is
/// scanned, so clean data is touched once (scan-then-memcpy touches it
/// twice). The store always lands in reserved slack and `set_len` only
/// ever covers the validated prefix. The final partial chunk is staged
/// zero-padded on the stack ([`crate::scan::padded_tail`]) so the
/// full-width load never reads past the source; the mask is truncated to
/// the real remainder before use.
///
/// `$step(src, dst)` loads `$width` bytes from `src`, stores them to
/// `dst`, and returns the needs-escape mask with `$bits_per_lane` mask
/// bits per input byte. Consumes the entire buffer (returns from the
/// enclosing function on every completion path).
macro_rules! fused_scan_store {
    ($out:ident, $s:ident, $width:expr, $bits_per_lane:expr, $step:expr) => {{
        let mut i = 0usize;
        $out.reserve($s.len() + $width);
        let mut base = $out.as_mut_ptr();
        let mut w = $out.len();
        // Becomes true when the first escape triggers the worst-case
        // reservation; see `escape_run_slow_path` for the full story.
        let mut worst_case_reserved = false;
        macro_rules! handle_escape_run {
            () => {{
                let (next_i, next_w) = escape_run_slow_path::<MODE>(
                    $out,
                    $s,
                    i,
                    w,
                    $width,
                    &mut worst_case_reserved,
                    &mut base,
                );
                i = next_i;
                w = next_w;
                if i >= $s.len() {
                    $out.set_len(w);
                    return;
                }
            }};
        }
        loop {
            // Hot loop: whole chunks. One load, one speculative store, one
            // mask test per $width bytes of clean text.
            while i + $width <= $s.len() {
                let mask: u64 = $step($s.as_ptr().add(i), base.add(w));
                if mask == 0 {
                    i += $width;
                    w += $width;
                    continue;
                }
                // Lowest set bit = first byte needing an escape; bytes
                // before it were already stored and are kept.
                let k = mask.trailing_zeros() as usize / $bits_per_lane;
                i += k;
                w += k;
                // With the reservation in place, the always-escaped classes
                // write inline; escape-dense text cares about the few
                // nanoseconds a cold call would add per escape. The first
                // escape and the mode extras take the cold path.
                let b = *$s.get_unchecked(i);
                if worst_case_reserved && ESCAPE_CLASS[b as usize] & CLASS_STANDARD != 0 {
                    if b == b'"' || b == b'\\' {
                        base.add(w).write(b'\\');
                        base.add(w + 1).write(b);
                        w += 2;
                    } else {
                        std::ptr::copy_nonoverlapping(ESC_SEQ[b as usize].as_ptr(), base.add(w), 8);
                        w += ESC_LEN[b as usize] as usize;
                    }
                    i += 1;
                    if i >= $s.len() {
                        $out.set_len(w);
                        return;
                    }
                } else {
                    handle_escape_run!();
                }
            }

            let rem = $s.len() - i;
            if rem == 0 {
                $out.set_len(w);
                return;
            }
            // Stage the remainder zero-padded on the stack so the
            // full-width load never reads past `$s`; padding classifies
            // as controls, masked out by the remainder truncation below.
            // The full-width store lands in reserved slack (`$width`
            // extra reserved on both reservation paths).
            let chunk = crate::scan::padded_tail::<{ $width }>(&$s[i..]);
            let keep_low_rem_lanes: u64 = u64::MAX >> (64 - $bits_per_lane * rem);
            let mask: u64 = $step(chunk.as_ptr(), base.add(w)) & keep_low_rem_lanes;
            if mask == 0 {
                $out.set_len(w + rem);
                return;
            }
            let k = mask.trailing_zeros() as usize / $bits_per_lane;
            i += k;
            w += k;
            handle_escape_run!();
        }
    }};
}

/// AVX2 fused scan+store: 32 bytes per step, consuming the whole buffer.
/// Callers must have verified AVX2 at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn escape_fused_avx2<const MODE: u8>(out: &mut Vec<u8>, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes (see the macro docs); the step is the shared AVX2 kernel,
    // inlined here because this function carries the same feature.
    unsafe { fused_scan_store!(out, s, 32, 1, crate::scan::x86::avx2_copy_scan::<MODE>) }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Upper bound on output bytes per consumed source byte: a control
/// character becomes `\u00XX` (6 bytes), and an AsciiOnly astral scalar
/// becomes two 6-byte escapes for 4 source bytes (3x). One reservation of
/// `MAX_ESCAPE_EXPANSION * remaining` at the first escape therefore covers
/// every later store (chunk, escape, or masked tail) with no further
/// capacity checks.
const MAX_ESCAPE_EXPANSION: usize = 6;

/// Write `\uXXXX` at `dst`. Always exactly 6 bytes.
///
/// # Safety
///
/// `dst` must have 6 writable bytes.
#[inline]
unsafe fn write_u16_escape_raw(dst: *mut u8, cp: u16) -> usize {
    // SAFETY: writes at offsets 0..=5 within the 6 writable bytes the
    // caller provides (see # Safety).
    unsafe {
        dst.write(b'\\');
        dst.add(1).write(b'u');
        dst.add(2).write(HEX[(cp >> 12) as usize & 0xF]);
        dst.add(3).write(HEX[(cp >> 8) as usize & 0xF]);
        dst.add(4).write(HEX[(cp >> 4) as usize & 0xF]);
        dst.add(5).write(HEX[cp as usize & 0xF]);
    }
    6
}

/// Decode one UTF-8 sequence starting at `i` (assumed valid), returning
/// (code point, length).
#[inline]
fn decode_utf8(s: &[u8], i: usize) -> (u32, usize) {
    let b0 = u32::from(s[i]);
    if b0 < 0xE0 {
        (((b0 & 0x1F) << 6) | (u32::from(s[i + 1]) & 0x3F), 2)
    } else if b0 < 0xF0 {
        (
            ((b0 & 0x0F) << 12)
                | ((u32::from(s[i + 1]) & 0x3F) << 6)
                | (u32::from(s[i + 2]) & 0x3F),
            3,
        )
    } else {
        (
            ((b0 & 0x07) << 18)
                | ((u32::from(s[i + 1]) & 0x3F) << 12)
                | ((u32::from(s[i + 2]) & 0x3F) << 6)
                | (u32::from(s[i + 3]) & 0x3F),
            4,
        )
    }
}

/// Escape lengths for the always-escaped set (`"`, `\`, controls). Paired
/// with [`ESC_SEQ`]: an escape is one unconditional 8-byte copy from the
/// padded table, then the cursor advances by the true length.
static ESC_LEN: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 0x20 {
        t[i] = match i as u8 {
            0x08 | 0x0C | b'\n' | b'\r' | b'\t' => 2,
            _ => 6,
        };
        i += 1;
    }
    t[b'"' as usize] = 2;
    t[b'\\' as usize] = 2;
    t
};

static ESC_SEQ: [[u8; 8]; 256] = {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut t = [[0u8; 8]; 256];
    let mut i = 0;
    while i < 0x20 {
        let b = i as u8;
        t[i] = match b {
            0x08 => *b"\\b\0\0\0\0\0\0",
            0x0C => *b"\\f\0\0\0\0\0\0",
            b'\n' => *b"\\n\0\0\0\0\0\0",
            b'\r' => *b"\\r\0\0\0\0\0\0",
            b'\t' => *b"\\t\0\0\0\0\0\0",
            _ => [
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[(b >> 4) as usize],
                HEX[(b & 0xF) as usize],
                0,
                0,
            ],
        };
        i += 1;
    }
    t[b'"' as usize] = *b"\\\"\0\0\0\0\0\0";
    t[b'\\' as usize] = *b"\\\\\0\0\0\0\0\0";
    t
};

/// Write escapes for the run of consecutive special bytes starting at
/// `s[i]` through raw stores at `dst`. Returns (first source index past the
/// run, bytes written). Always-escaped bytes go through direct branches
/// (`"`, `\`, the dominant pair) or the padded table (controls: one
/// unconditional 8-byte copy). Consuming the whole run here avoids
/// re-entering the SIMD scan once per escape in escape-dense text.
///
/// # Safety
///
/// The caller must guarantee `MAX_ESCAPE_EXPANSION * (s.len() - i) + 8`
/// writable bytes at `dst` (the `+ 8` covers the padded table store's
/// overhang past a final short escape).
unsafe fn write_escape_run_raw<const MODE: u8>(
    s: &[u8],
    mut i: usize,
    dst: *mut u8,
) -> (usize, usize) {
    let mode_bits: u8 = class_bits::<MODE>();
    let mut w = 0usize;
    // SAFETY: the caller reserves `MAX_ESCAPE_EXPANSION * (s.len() - i) +
    // 8` bytes at `dst` (see # Safety). Every branch writes at most
    // MAX_ESCAPE_EXPANSION bytes per consumed source byte (two 6-byte
    // escapes for a 4-byte astral sequence), and the padded 8-byte table
    // store's overhang stays within the `+ 8` slack. Source reads are
    // bounds-checked (`s[i]`, `s.get`).
    unsafe {
        loop {
            let b = s[i];
            if b == b'"' {
                dst.add(w).write(b'\\');
                dst.add(w + 1).write(b'"');
                w += 2;
                i += 1;
            } else if b == b'\\' {
                dst.add(w).write(b'\\');
                dst.add(w + 1).write(b'\\');
                w += 2;
                i += 1;
            } else if b < 0x20 {
                std::ptr::copy_nonoverlapping(ESC_SEQ[b as usize].as_ptr(), dst.add(w), 8);
                w += ESC_LEN[b as usize] as usize;
                i += 1;
            } else if MODE == MODE_SCRIPT_SAFE && b == b'/' {
                dst.add(w).write(b'\\');
                dst.add(w + 1).write(b'/');
                w += 2;
                i += 1;
            } else if MODE == MODE_SCRIPT_SAFE && b == 0xE2 {
                // U+2028 / U+2029 are E2 80 A8 / E2 80 A9; any other
                // E2-led sequence passes through unescaped.
                if s.get(i + 1) == Some(&0x80) && matches!(s.get(i + 2), Some(&(0xA8 | 0xA9))) {
                    w += write_u16_escape_raw(dst.add(w), 0x2028 + u16::from(s[i + 2] - 0xA8));
                    i += 3;
                } else {
                    dst.add(w).write(0xE2);
                    w += 1;
                    i += 1;
                }
            } else if MODE == MODE_ASCII_ONLY && b >= 0x80 {
                let (cp, len) = decode_utf8(s, i);
                i += len;
                if cp >= 0x10000 {
                    let v = cp - 0x10000;
                    w += write_u16_escape_raw(dst.add(w), 0xD800 + (v >> 10) as u16);
                    w += write_u16_escape_raw(dst.add(w), 0xDC00 + (v & 0x3FF) as u16);
                } else {
                    w += write_u16_escape_raw(dst.add(w), cp as u16);
                }
            } else {
                // Unreachable by scan construction; keep bytes intact.
                dst.add(w).write(b);
                w += 1;
                i += 1;
            }
            // Only AsciiOnly loops here: its escapes come in runs (every
            // non-ASCII byte), so continuing saves one SIMD re-entry per
            // escaped character. Standard/ScriptSafe escapes are rarely
            // consecutive, and the extra check just slows them down.
            if MODE != MODE_ASCII_ONLY
                || i >= s.len()
                || ESCAPE_CLASS[s[i] as usize] & mode_bits == 0
            {
                return (i, w);
            }
        }
    }
}

/// Escape handling for the fused loops, kept out of line so the clean-text
/// loop stays small (inlining this costs clean short strings dearly). The
/// first call for a buffer makes the worst-case reservation (see
/// [`MAX_ESCAPE_EXPANSION`], plus `width` slack for speculative chunk
/// stores and 8 for the escape table's padded store), after which no store
/// needs a capacity check. Returns the advanced (source, write) cursors;
/// the caller owns `set_len`.
#[cold]
fn escape_run_slow_path<const MODE: u8>(
    out: &mut Vec<u8>,
    s: &[u8],
    i: usize,
    w: usize,
    width: usize,
    worst_case_reserved: &mut bool,
    base: &mut *mut u8,
) -> (usize, usize) {
    if !*worst_case_reserved {
        let needed = w + MAX_ESCAPE_EXPANSION * (s.len() - i) + width + 8;
        out.reserve(needed.saturating_sub(out.len()));
        *base = out.as_mut_ptr();
        *worst_case_reserved = true;
    }
    // SAFETY: the reservation above (sticky across calls for this buffer)
    // covers the run's worst case at offset `w`.
    let (next_i, wrote) = unsafe { write_escape_run_raw::<MODE>(s, i, base.add(w)) };
    (next_i, w + wrote)
}

fn escape_impl<const MODE: u8>(out: &mut Vec<u8>, s: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 presence confirmed at runtime.
        unsafe { escape_fused_avx2::<MODE>(out, s) }
    } else {
        escape_fused_16::<MODE>(out, s);
    }
    #[cfg(target_arch = "aarch64")]
    escape_fused_16::<MODE>(out, s);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fallback::escape::<MODE>(out, s);
}

/// NEON fused scan+store: 16 bytes per step (see [`fused_scan_store!`]).
/// The mask has 4 bits per lane (`vshrn` movemask substitute).
///
/// Both this function and the step kernel must be `inline(always)`: as
/// ordinary calls they miss inlining at the tail call site, which short
/// strings pay for on every call.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn escape_fused_16<const MODE: u8>(out: &mut Vec<u8>, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes; see the macro docs.
    unsafe { fused_scan_store!(out, s, 16, 4, crate::scan::neon::copy_scan::<MODE>) }
}

/// SSE2 fused scan+store: 16 bytes per step (see [`fused_scan_store!`]).
/// `inline(always)` for the same reason as the aarch64 variant.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn escape_fused_16<const MODE: u8>(out: &mut Vec<u8>, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes; see the macro docs.
    unsafe { fused_scan_store!(out, s, 16, 1, crate::scan::x86::sse2_copy_scan::<MODE>) }
}

/// Append `s` to `out` with JSON string escaping (no surrounding quotes).
/// `s` must be valid UTF-8 (the host guarantees it, e.g. via coderange).
pub fn escape_into(out: &mut Vec<u8>, s: &[u8], mode: EscapeMode) {
    match mode {
        EscapeMode::Standard => escape_impl::<MODE_STANDARD>(out, s),
        EscapeMode::ScriptSafe => escape_impl::<MODE_SCRIPT_SAFE>(out, s),
        EscapeMode::AsciiOnly => escape_impl::<MODE_ASCII_ONLY>(out, s),
    }
}

/// Append an integer.
#[inline]
pub fn write_i64(out: &mut Vec<u8>, v: i64) {
    let mut buf = itoa::Buffer::new();
    let s = buf.format(v).as_bytes();
    if s.len() <= 16 {
        push_short(out, s);
    } else {
        out.extend_from_slice(s);
    }
}

/// Append a finite f64 in the fpconv (Grisu2) format: decimal notation over
/// a wide exponent range, `.0` appended to integral values, `e±NN`
/// otherwise. This deliberately matches the widely deployed C `fpconv`
/// formatter byte-for-byte rather than Rust's shortest-round-trip
/// `Display`: Grisu2 without a fallback occasionally emits a non-shortest
/// digit sequence (still exactly round-trippable). See the `grisu2` module
/// source for provenance.
///
/// `f` must be finite (debug-asserted): this is the trusted kernel tier
/// for hosts that gate non-finite values themselves, and in release a
/// non-finite input produces digits of an unrelated finite number. The
/// [`crate::Writer`] tier enforces finiteness unconditionally.
pub fn write_f64(out: &mut Vec<u8>, f: f64) {
    debug_assert!(f.is_finite(), "write_f64 requires a finite value");
    crate::grisu2::dtoa(f, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn esc(s: &str, mode: EscapeMode) -> String {
        // Exact-size allocation: a tail overread would be out of the
        // allocation entirely, not just past the string's logical end.
        let exact: Box<[u8]> = s.as_bytes().to_vec().into_boxed_slice();
        let mut out = Vec::new();
        escape_into(&mut out, &exact, mode);
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn standard_escapes() {
        assert_eq!(esc("plain", EscapeMode::Standard), "plain");
        assert_eq!(esc("a\"b\\c", EscapeMode::Standard), "a\\\"b\\\\c");
        assert_eq!(
            esc("\n\t\r\u{8}\u{c}", EscapeMode::Standard),
            "\\n\\t\\r\\b\\f"
        );
        assert_eq!(esc("\u{1}", EscapeMode::Standard), "\\u0001");
        assert_eq!(esc("héllo", EscapeMode::Standard), "héllo");
        // Long clean strings stream through SIMD.
        let long = "x".repeat(1000);
        assert_eq!(esc(&long, EscapeMode::Standard), long);
        let mixed = format!("{long}\"{long}");
        assert_eq!(
            esc(&mixed, EscapeMode::Standard),
            format!("{long}\\\"{long}")
        );
    }

    /// Reference implementation for the boundary sweep: per-char escaping
    /// with no vector paths.
    fn esc_reference(s: &str, mode: EscapeMode) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                '/' if mode == EscapeMode::ScriptSafe => out.push_str("\\/"),
                '\u{2028}' | '\u{2029}' if mode == EscapeMode::ScriptSafe => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c if (c as u32) >= 0x80 && mode == EscapeMode::AsciiOnly => {
                    if (c as u32) >= 0x10000 {
                        let mut units = [0u16; 2];
                        c.encode_utf16(&mut units);
                        let _ = write!(out, "\\u{:04x}\\u{:04x}", units[0], units[1]);
                    } else {
                        let _ = write!(out, "\\u{:04x}", c as u32);
                    }
                }
                c => out.push(c),
            }
        }
        out
    }

    /// Every input length around the vector widths, with an escape (or
    /// none) at every position: exercises the padded tails, the short
    /// path, and chunk boundaries against the scalar reference. Inputs
    /// come from exact-size heap allocations so a tail overread would be
    /// out-of-allocation (caught under ASan/Miri-style tooling).
    #[test]
    fn escape_boundary_sweep() {
        for mode in [
            EscapeMode::Standard,
            EscapeMode::ScriptSafe,
            EscapeMode::AsciiOnly,
        ] {
            for len in 0..=40 {
                let clean: String = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"[..len].to_string();
                assert_eq!(esc(&clean, mode), esc_reference(&clean, mode), "len {len}");
                for pos in 0..len {
                    let mut bytes = clean.clone().into_bytes();
                    bytes[pos] = b'"';
                    let s = String::from_utf8(bytes).unwrap();
                    assert_eq!(
                        esc(&s, mode),
                        esc_reference(&s, mode),
                        "len {len} quote at {pos}"
                    );
                    let mut bytes = clean.clone().into_bytes();
                    bytes[pos] = b'\n';
                    let s = String::from_utf8(bytes).unwrap();
                    assert_eq!(
                        esc(&s, mode),
                        esc_reference(&s, mode),
                        "len {len} newline at {pos}"
                    );
                }
            }
            // Multi-byte content across the tail: lengths that split the
            // vector boundary mid-sequence.
            for pad in 0..=35 {
                let s = format!("{}é\u{2028}\u{1F600}", "y".repeat(pad));
                assert_eq!(esc(&s, mode), esc_reference(&s, mode), "pad {pad}");
            }
        }
    }

    /// The public short-copy helper must stay safe for any length.
    #[test]
    fn push_short_any_length() {
        for len in 0..=40 {
            let src: Vec<u8> = (0..len as u8).collect();
            let mut out = b"prefix".to_vec();
            push_short(&mut out, &src);
            assert_eq!(&out[..6], b"prefix");
            assert_eq!(&out[6..], &src[..], "len {len}");
        }
    }

    #[test]
    fn script_safe_escapes() {
        assert_eq!(esc("a/b", EscapeMode::ScriptSafe), "a\\/b");
        assert_eq!(
            esc("\u{2028}\u{2029}", EscapeMode::ScriptSafe),
            "\\u2028\\u2029"
        );
        // Other E2-prefixed sequences pass through.
        assert_eq!(esc("\u{20AC}", EscapeMode::ScriptSafe), "\u{20AC}");
    }

    #[test]
    fn ascii_only_escapes() {
        assert_eq!(esc("héllo", EscapeMode::AsciiOnly), "h\\u00e9llo");
        assert_eq!(esc("\u{1F600}", EscapeMode::AsciiOnly), "\\ud83d\\ude00");
        assert_eq!(
            esc("κόσμε", EscapeMode::AsciiOnly),
            "\\u03ba\\u03cc\\u03c3\\u03bc\\u03b5"
        );
    }

    #[test]
    fn float_fpconv_format() {
        // Expected bytes verified against the reference C fpconv output.
        // Note the deliberate non-shortest digits (Grisu2 without fallback).
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (5.0, "5.0"),
            (1.5, "1.5"),
            (0.0001, "0.0001"),
            (1e-5, "0.00001"),
            (1e14, "100000000000000.0"),
            (1e15, "1e+15"),
            (1e16, "1e+16"),
            (1.23456789e13, "12345678900000.0"),
            (1.5e-7, "0.00000015"),
            (1e100, "1e+100"),
            (-2.5, "-2.5"),
            (1.0 / 3.0, "0.3333333333333333"),
            (2.34387207031, "2.3438720703100002"),
            (-61.14917000000003, "-61.149170000000026"),
            (0.130293816489, "0.13029381648899999"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
        ];
        for &(f, want) in cases {
            let mut out = Vec::new();
            write_f64(&mut out, f);
            assert_eq!(std::str::from_utf8(&out).unwrap(), want, "for {f:e}");
        }
    }

    #[test]
    fn integers() {
        for v in [0i64, 1, -1, i64::MAX, i64::MIN, 42, -12345678901234] {
            let mut out = Vec::new();
            write_i64(&mut out, v);
            assert_eq!(String::from_utf8(out).unwrap(), v.to_string());
        }
    }
}
