//! JSON emission kernels: SIMD escape scanning and pinned-format number
//! writing. Pure computation over byte buffers: [`crate::Writer`] layers
//! grammar state on top; hosts with bespoke needs can call these directly.
//!
//! The byte-classification primitives (hit masks, class table, padded
//! tail staging) are shared with the parser and live in the
//! crate-internal `scan` module.

use crate::scan::{
    CLASS_STANDARD, ESCAPE_CLASS, JS_SEP_CONT, JS_SEP_LAST_2028, JS_SEP_LAST_2029, JS_SEP_LEAD,
    MODE_ASCII_ONLY, MODE_HTML_ENTITIES, MODE_HTML_SAFE, MODE_JS_SEPARATORS, MODE_SCRIPT_SAFE,
    MODE_STANDARD, class_bits,
};

/// Whether `MODE` escapes the JS line separators U+2028/U+2029.
const fn escapes_js_separators<const MODE: u8>() -> bool {
    matches!(MODE, MODE_SCRIPT_SAFE | MODE_HTML_SAFE | MODE_JS_SEPARATORS)
}

/// Whether `MODE` escapes the HTML-active characters `<`, `>`, `&`.
const fn escapes_html_entities<const MODE: u8>() -> bool {
    matches!(MODE, MODE_HTML_SAFE | MODE_HTML_ENTITIES)
}

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
    /// Additionally escape `<`, `>`, `&` (as their `\uXXXX` escapes) and
    /// U+2028/U+2029, making the output safe to interpolate into HTML
    /// documents (the profile HTML-embedding frameworks apply to JSON).
    HtmlSafe,
    /// Additionally escape only `<`, `>`, `&` (as their `\uXXXX`
    /// escapes): [`EscapeMode::HtmlSafe`] without the JS line
    /// separators.
    HtmlEntities,
    /// Additionally escape only U+2028/U+2029:
    /// [`EscapeMode::HtmlSafe`] without the HTML-active characters.
    JsSeparators,
}

/// Escaping for hosts without SIMD kernels; SIMD targets handle entire
/// strings inside the fused loops and never come here.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod fallback {
    use super::{
        ESCAPE_CLASS, EmitBuf, MAX_ESCAPE_EXPANSION, class_bits, push_short, write_escape_run_raw,
    };

    /// SWAR-scan for the next escape, copy the clean span, write the
    /// escape run, repeat.
    pub(super) fn escape<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8]) {
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

    /// Buffer-front wrapper for [`write_escape_run_raw`]: reserves the
    /// run's worst case, commits with one `set_len`.
    fn write_escape_run<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8], i: usize) -> usize {
        out.reserve(MAX_ESCAPE_EXPANSION * (s.len() - i) + 8);
        // SAFETY: worst case reserved above; `set_len` covers only the
        // bytes the run wrote.
        unsafe {
            let len = out.len();
            let dst = out.as_mut_ptr().add(len);
            let (next_i, wrote) = write_escape_run_raw::<MODE>(s, i, dst);
            out.set_len(len + wrote);
            next_i
        }
    }
}

/// Raw-capacity byte sink for the emission kernels: `Vec<u8>`'s
/// grow/write/publish contract as a trait, so hosts can point the
/// kernels at a foreign buffer (an interpreter's string type, an
/// arena) and skip the final copy out. Method names deliberately
/// mirror `Vec<u8>`; the kernels monomorphize, so a non-`Vec`
/// implementation costs nothing.
///
/// # Safety
///
/// Implementations must uphold what the kernels assume of `Vec<u8>`:
/// after `reserve(n)`, `as_mut_ptr() + len()` addresses at least `n`
/// writable bytes that stay valid until the next `reserve`; `len` only
/// changes through `set_len`; `set_len(l)` publishes exactly the `l`
/// bytes the caller initialized. Growth must preserve bytes up to
/// `len()`; bytes past `len()` may be discarded (the kernels publish
/// speculative prefixes before any mid-flight `reserve`).
pub unsafe trait EmitBuf {
    /// Guarantee `additional` writable bytes past `len()`.
    fn reserve(&mut self, additional: usize);
    /// Current published length in bytes.
    fn len(&self) -> usize;
    /// True when nothing has been published yet.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Base pointer of the buffer (re-derive after every `reserve`:
    /// growth may relocate storage).
    fn as_mut_ptr(&mut self) -> *mut u8;
    /// Publish `new_len` bytes.
    ///
    /// # Safety
    ///
    /// `new_len` must not exceed reserved capacity and every byte up
    /// to it must be initialized.
    unsafe fn set_len(&mut self, new_len: usize);

    /// Append one byte.
    #[inline(always)]
    fn push(&mut self, b: u8) {
        self.reserve(1);
        // SAFETY: one byte reserved above.
        unsafe {
            let len = self.len();
            self.as_mut_ptr().add(len).write(b);
            self.set_len(len + 1);
        }
    }

    /// Append a slice.
    #[inline(always)]
    fn extend_from_slice(&mut self, s: &[u8]) {
        self.reserve(s.len());
        // SAFETY: `s.len()` bytes reserved above; the borrow rules
        // keep `s` disjoint from this buffer's unpublished tail.
        unsafe {
            let len = self.len();
            core::ptr::copy_nonoverlapping(s.as_ptr(), self.as_mut_ptr().add(len), s.len());
            self.set_len(len + s.len());
        }
    }
}

// SAFETY: Vec<u8> is the reference implementation of the contract;
// every method delegates to the identically named inherent one.
unsafe impl EmitBuf for Vec<u8> {
    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }
    #[inline(always)]
    fn len(&self) -> usize {
        Vec::len(self)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        Vec::as_mut_ptr(self)
    }
    #[inline(always)]
    unsafe fn set_len(&mut self, new_len: usize) {
        // SAFETY: forwarded contract.
        unsafe { Vec::set_len(self, new_len) };
    }
    #[inline(always)]
    fn push(&mut self, b: u8) {
        Vec::push(self, b);
    }
    #[inline(always)]
    fn extend_from_slice(&mut self, s: &[u8]) {
        Vec::extend_from_slice(self, s);
    }
}

/// Append a short slice (≤ 16 bytes) with overlapping word stores instead of
/// a `memcpy` PLT call: tiny-copy call overhead showed up at 14% on
/// string-heavy generation profiles. Longer slices take a plain
/// `extend_from_slice` (hot callers prove the bound; the fallback keeps
/// this safe for everyone else).
#[inline(always)]
pub fn push_short<B: EmitBuf>(out: &mut B, s: &[u8]) {
    let n = s.len();
    if n > 16 {
        out.extend_from_slice(s);
        return;
    }
    out.reserve(16);
    // SAFETY: `n <= 16` readable bytes in `s`, 16 writable bytes reserved
    // above; `set_len` covers exactly the `n` written bytes.
    unsafe {
        let len = out.len();
        crate::scan::copy_small(s.as_ptr(), out.as_mut_ptr().add(len), n);
        out.set_len(len + n);
    }
}

/// The raw-pointer counterpart of [`push_short`], for hosts writing
/// into their own reservations: `n <= 32` copies with a pair of
/// overlapping word stores instead of a `memcpy` call (tiny-copy call
/// overhead is measurable on short-string workloads), longer copies
/// take `copy_nonoverlapping`.
///
/// # Safety
///
/// `n` readable bytes at `src`, `n` writable bytes at `dst`, and the
/// two ranges must not overlap.
#[inline(always)]
pub unsafe fn copy_short_raw(src: *const u8, dst: *mut u8, n: usize) {
    // SAFETY: forwarded contract; both branches stay within `n` bytes
    // at each pointer.
    unsafe {
        if n <= 32 {
            crate::scan::copy_small(src, dst, n);
        } else {
            std::ptr::copy_nonoverlapping(src, dst, n);
        }
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
                let (next_i, next_w) = escape_run_slow_path::<MODE, B>(
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
unsafe fn escape_fused_avx2<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes (see the macro docs); the step is the shared AVX2 kernel,
    // inlined here because this function carries the same feature.
    unsafe { fused_scan_store!(out, s, 32, 1, crate::scan::x86::avx2_copy_scan::<MODE>) }
}

/// AVX-512BW fused scan+store: 64 bytes per step, mask-register
/// classification. Callers must have verified AVX-512BW at runtime.
/// Not dispatched (see the rejection note at `escape_impl`): kept,
/// with its kernel_bench evidence, for scan-only consumers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[allow(dead_code)]
unsafe fn escape_fused_avx512<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes (see the macro docs); the step is the shared AVX-512
    // kernel, inlined here because this function carries the same
    // feature.
    unsafe { fused_scan_store!(out, s, 64, 1, crate::scan::x86::avx512_copy_scan::<MODE>) }
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
            } else if escapes_html_entities::<MODE>() && (b == b'<' || b == b'>' || b == b'&') {
                w += write_u16_escape_raw(dst.add(w), u16::from(b));
                i += 1;
            } else if escapes_js_separators::<MODE>() && b == JS_SEP_LEAD {
                // Any lead byte whose continuation is not a separator
                // passes through unescaped.
                if s.get(i + 1) == Some(&JS_SEP_CONT)
                    && matches!(s.get(i + 2), Some(&(JS_SEP_LAST_2028 | JS_SEP_LAST_2029)))
                {
                    w += write_u16_escape_raw(
                        dst.add(w),
                        0x2028 + u16::from(s[i + 2] - JS_SEP_LAST_2028),
                    );
                    i += 3;
                } else {
                    dst.add(w).write(JS_SEP_LEAD);
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
fn escape_run_slow_path<const MODE: u8, B: EmitBuf>(
    out: &mut B,
    s: &[u8],
    i: usize,
    w: usize,
    width: usize,
    worst_case_reserved: &mut bool,
    base: &mut *mut u8,
) -> (usize, usize) {
    if !*worst_case_reserved {
        let needed = w + MAX_ESCAPE_EXPANSION * (s.len() - i) + width + 8;
        // Publish the speculative prefix before growing: `reserve` is
        // only required to preserve bytes up to `len()`, and the fused
        // loop has initialized everything below `w`.
        // SAFETY: bytes `0..w` were written by the loop's stores.
        unsafe { out.set_len(w) };
        out.reserve(needed.saturating_sub(w));
        *base = out.as_mut_ptr();
        *worst_case_reserved = true;
    }
    // SAFETY: the reservation above (sticky across calls for this buffer)
    // covers the run's worst case at offset `w`.
    let (next_i, wrote) = unsafe { write_escape_run_raw::<MODE>(s, i, base.add(w)) };
    (next_i, w + wrote)
}

/// Below this length the AVX2 masked-tail entry costs more than the
/// 16-byte SSE2 loop: measured on Zen 4 (EPYC 9R14), 10-11B escapes run
/// 14.4ns through the masked tail vs 10.9ns through the 16B loop, while
/// at 16B+ the wide path wins or ties. Short strings (object keys
/// especially) dominate real documents, so they take the narrow loop.
#[cfg(target_arch = "x86_64")]
const AVX2_MIN_LEN: usize = 16;

// Tried and rejected (2026-07-15, Zen 4): dispatching escape emission
// to `escape_fused_avx512` for 64B+ input. Clean-string kernel shapes
// flew (long_clean 123->72ns, unicode 21->12ns), but every escape
// restarts the wide loop, and the corpus punished that: homebrew-llvm
// generation (long strings, isolated escapes) -15%, tolstoy flat,
// dense-escape text -20%. The kernel stays available for scan-only
// consumers where there is no store/restart cost.
fn escape_impl<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    if s.len() >= AVX2_MIN_LEN && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 presence confirmed at runtime.
        unsafe { escape_fused_avx2::<MODE, B>(out, s) }
    } else {
        escape_fused_16::<MODE, B>(out, s);
    }
    #[cfg(target_arch = "aarch64")]
    escape_fused_16::<MODE, B>(out, s);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fallback::escape::<MODE, B>(out, s);
}

/// NEON fused scan+store: 16 bytes per step (see [`fused_scan_store!`]).
/// The mask has 4 bits per lane (`vshrn` movemask substitute).
///
/// Both this function and the step kernel must be `inline(always)`: as
/// ordinary calls they miss inlining at the tail call site, which short
/// strings pay for on every call.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn escape_fused_16<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes; see the macro docs.
    unsafe { fused_scan_store!(out, s, 16, 4, crate::scan::neon::copy_scan::<MODE>) }
}

/// SSE2 fused scan+store: 16 bytes per step (see [`fused_scan_store!`]).
/// `inline(always)` for the same reason as the aarch64 variant.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn escape_fused_16<const MODE: u8, B: EmitBuf>(out: &mut B, s: &[u8]) {
    // SAFETY: the macro's loads/stores stay inside `s` / the reservation
    // it makes; see the macro docs.
    unsafe { fused_scan_store!(out, s, 16, 1, crate::scan::x86::sse2_copy_scan::<MODE>) }
}

/// Append `s` to `out` with JSON string escaping (no surrounding quotes).
/// `s` must be valid UTF-8 (the host guarantees it, e.g. via coderange).
pub fn escape_into<B: EmitBuf>(out: &mut B, s: &[u8], mode: EscapeMode) {
    match mode {
        EscapeMode::Standard => escape_impl::<MODE_STANDARD, B>(out, s),
        EscapeMode::ScriptSafe => escape_impl::<MODE_SCRIPT_SAFE, B>(out, s),
        EscapeMode::AsciiOnly => escape_impl::<MODE_ASCII_ONLY, B>(out, s),
        EscapeMode::HtmlSafe => escape_impl::<MODE_HTML_SAFE, B>(out, s),
        EscapeMode::HtmlEntities => escape_impl::<MODE_HTML_ENTITIES, B>(out, s),
        EscapeMode::JsSeparators => escape_impl::<MODE_JS_SEPARATORS, B>(out, s),
    }
}

/// Two-digit pairs "00".."99", the classic itoa table.
static DIGIT_PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[i * 2] = b'0' + (i / 10) as u8;
        t[i * 2 + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

/// Decimal digit count of `u` (1 for 0): floor(log2) scaled by the
/// classic 1233/4096 log10(2) approximation, corrected by one
/// power-of-ten compare.
#[inline(always)]
fn decimal_len_u64(u: u64) -> usize {
    static POW10: [u64; 19] = [
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        10_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
        10_000_000_000_000,
        100_000_000_000_000,
        1_000_000_000_000_000,
        10_000_000_000_000_000,
        100_000_000_000_000_000,
        1_000_000_000_000_000_000,
        10_000_000_000_000_000_000,
    ];
    let bits = 63 ^ (u | 1).leading_zeros() as usize;
    let approx = (bits * 1233) >> 12;
    approx + 1 + usize::from(u >= POW10[approx])
}

/// Append an integer.
#[inline]
pub fn write_i64<B: EmitBuf>(out: &mut B, v: i64) {
    out.reserve(I64_MAX_LEN);
    // SAFETY: I64_MAX_LEN reserved above covers the widest value.
    unsafe {
        let len = out.len();
        let n = write_i64_raw(out.as_mut_ptr().add(len), v);
        out.set_len(len + n);
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
pub fn write_f64<B: EmitBuf>(out: &mut B, f: f64) {
    debug_assert!(f.is_finite(), "write_f64 requires a finite value");
    crate::grisu2::dtoa(f, out);
}

/// Maximum bytes [`write_f64_raw`] writes.
pub const F64_MAX_LEN: usize = crate::grisu2::DTOA_MAX;

/// [`write_f64`] through a raw pointer, for hosts that batch many
/// numeric writes under one reservation and keep the cursor in a
/// register (the C fpconv calling convention). Returns the byte count.
///
/// # Safety
///
/// `dst` must have [`F64_MAX_LEN`] writable bytes. `f` must be finite
/// (debug-asserted, same contract as [`write_f64`]).
pub unsafe fn write_f64_raw(dst: *mut u8, f: f64) -> usize {
    // SAFETY: forwarded contract.
    unsafe { crate::grisu2::dtoa_raw(f, dst) }
}

/// Maximum bytes [`write_i64_raw`] writes ("-9223372036854775808").
pub const I64_MAX_LEN: usize = 20;

/// [`write_i64`] through a raw pointer; see [`write_f64_raw`] for the
/// use case. Returns the byte count.
///
/// Digits are written backward, in place, from a precomputed length:
/// no staging buffer and no copy out (routing through a stack itoa
/// buffer plus a short copy measured ~25% slower per integer on
/// int-dense generation).
///
/// # Safety
///
/// `dst` must have [`I64_MAX_LEN`] writable bytes.
pub unsafe fn write_i64_raw(dst: *mut u8, v: i64) -> usize {
    let neg = v < 0;
    let mut u = v.unsigned_abs();
    // SAFETY: the sign byte plus `decimal_len_u64(u)` digits fit in
    // the caller's I64_MAX_LEN reservation; `p` walks backward from
    // the exact end of that span to `digits_start`.
    unsafe {
        let mut w = 0usize;
        if neg {
            *dst = b'-';
            w = 1;
        }
        let len = decimal_len_u64(u);
        let digits_start = dst.add(w);
        let mut p = digits_start.add(len);
        while u >= 100 {
            let pair = ((u % 100) as usize) * 2;
            u /= 100;
            p = p.sub(2);
            p.copy_from_nonoverlapping(DIGIT_PAIRS.as_ptr().add(pair), 2);
        }
        if u >= 10 {
            p = p.sub(2);
            p.copy_from_nonoverlapping(DIGIT_PAIRS.as_ptr().add((u as usize) * 2), 2);
        } else {
            p = p.sub(1);
            *p = b'0' + u as u8;
        }
        debug_assert!(p == digits_start);
        w + len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_i64_matches_display_across_digit_boundaries() {
        let mut cases: Vec<i64> = vec![0, i64::MIN, i64::MAX];
        let mut p: i64 = 1;
        for _ in 0..18 {
            p *= 10;
            cases.extend([p - 1, p, p + 1, -(p - 1), -p, -(p + 1)]);
        }
        for v in cases {
            let mut out = Vec::new();
            write_i64(&mut out, v);
            assert_eq!(out, v.to_string().into_bytes(), "value {v}");
        }
    }

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
                '<' | '>' | '&'
                    if mode == EscapeMode::HtmlSafe || mode == EscapeMode::HtmlEntities =>
                {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                '\u{2028}' | '\u{2029}'
                    if matches!(
                        mode,
                        EscapeMode::ScriptSafe | EscapeMode::HtmlSafe | EscapeMode::JsSeparators
                    ) =>
                {
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
            EscapeMode::HtmlSafe,
            EscapeMode::HtmlEntities,
            EscapeMode::JsSeparators,
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
                    let mut bytes = clean.clone().into_bytes();
                    bytes[pos] = b'&';
                    let s = String::from_utf8(bytes).unwrap();
                    assert_eq!(
                        esc(&s, mode),
                        esc_reference(&s, mode),
                        "len {len} ampersand at {pos}"
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
    fn html_safe_escapes() {
        assert_eq!(
            esc("<script>a && b</script>", EscapeMode::HtmlSafe),
            "\\u003cscript\\u003ea \\u0026\\u0026 b\\u003c/script\\u003e"
        );
        assert_eq!(
            esc("\u{2028}\u{2029}", EscapeMode::HtmlSafe),
            "\\u2028\\u2029"
        );
        // `/` is NOT escaped in this mode, and other E2-prefixed
        // sequences pass through.
        assert_eq!(esc("a/b \u{20AC}", EscapeMode::HtmlSafe), "a/b \u{20AC}");
        assert_eq!(esc("héllo", EscapeMode::HtmlSafe), "héllo");
    }

    #[test]
    fn partial_html_profiles_escape_only_their_set() {
        assert_eq!(
            esc("a<b \u{2028}", EscapeMode::HtmlEntities),
            "a\\u003cb \u{2028}"
        );
        assert_eq!(esc("a<b \u{2028}", EscapeMode::JsSeparators), "a<b \\u2028");
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
