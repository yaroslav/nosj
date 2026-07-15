//! Scalar token parsing for the pull parser: strings (zero-copy fast path +
//! escape decoding), numbers (grammar-validating scan with i64/f64/bignum
//! split), and literals. Pure computation, no host types, unit-testable.

/// Failure while parsing a single string, number, or literal token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarError {
    /// Input ended before the closing `"`.
    UnterminatedString,
    /// A backslash escape other than `\" \\ \/ \b \f \n \r \t \uXXXX`.
    InvalidEscape,
    /// `\u` not followed by four hex digits.
    InvalidUnicodeEscape,
    /// A UTF-16 surrogate escape without its required pair.
    LoneSurrogate,
    /// A raw control character (< 0x20) inside a string.
    ControlCharacterInString,
    /// A token that started like a number but violates the JSON grammar.
    InvalidNumber,
    /// A token that started like `true`/`false`/`null` but is not one.
    InvalidLiteral,
    /// A scalar immediately followed by a byte that cannot end it.
    TrailingGarbageAfterScalar,
}

impl std::fmt::Display for ScalarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnterminatedString => "unterminated string",
            Self::InvalidEscape => "invalid escape sequence",
            Self::InvalidUnicodeEscape => "invalid \\u escape",
            Self::LoneSurrogate => "lone UTF-16 surrogate",
            Self::ControlCharacterInString => "raw control character in string",
            Self::InvalidNumber => "invalid number",
            Self::InvalidLiteral => "invalid literal",
            Self::TrailingGarbageAfterScalar => "unexpected characters after value",
        })
    }
}

impl std::error::Error for ScalarError {}

/// Result of scanning a string starting at its opening quote. `'j` is
/// the input's lifetime, `'s` the scratch buffer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrPart<'j, 's> {
    /// No escapes: borrows straight from the input.
    Borrowed(&'j str),
    /// Escapes decoded into the scratch buffer.
    Decoded(&'s str),
    /// Escapes decoded into the scratch buffer, and a lone low surrogate
    /// escape was emitted as its raw WTF-8 bytes (the leniency of widely
    /// deployed parsers): the content is NOT valid UTF-8 and must be
    /// handled as bytes.
    DecodedRaw(&'s [u8]),
}

/// Byte can legally follow a completed scalar token (or string close quote).
#[inline(always)]
#[must_use]
pub fn is_token_boundary(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t' | b'\n' | b'\r' | b'{' | b'}' | b'[' | b']' | b':' | b','
    )
}

/// Scalar reference loop for [`find_special`]: per-byte classification,
/// used for documents shorter than one vector and on non-SIMD hosts.
#[inline(always)]
fn find_special_scalar(input: &[u8], mut from: usize) -> Option<(usize, u8)> {
    while from < input.len() {
        let b = input[from];
        if b == b'"' || b == b'\\' || b < 0x20 {
            return Some((from, b));
        }
        from += 1;
    }
    None
}

/// Find the next `"`, `\`, or control character (< 0x20) at or after `from`.
/// SIMD-accelerated; folding the control-character check into this scan means
/// string content is only ever read once.
#[inline(always)]
fn find_special(input: &[u8], mut from: usize) -> Option<(usize, u8)> {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every 16-byte load starts at an offset with `offset + 16 <=
    // input.len()` (the probe and loop conditions guarantee it), and the
    // backwards tail loads at exactly `input.len() - 16`.
    unsafe {
        use crate::scan::{MODE_STANDARD, neon};
        use std::arch::aarch64::{vld1q_u8, vmaxvq_u8, vorrq_u8};

        // Most strings (keys, short values) end within 16 bytes: one probe
        // resolves them at half the vector work of the wide loop.
        if from + 16 <= input.len() {
            let mask = neon::hit_mask::<MODE_STANDARD>(vld1q_u8(input.as_ptr().add(from)));
            if mask != 0 {
                let i = from + (mask.trailing_zeros() as usize) / 4;
                return Some((i, input[i]));
            }
            from += 16;
        }

        // 32 bytes per iteration with a single-instruction early exit
        // (vmaxvq on the OR of both hit vectors), so long clean strings
        // stream through without extracting a mask. This shape suits the
        // parser, whose strings are mostly clean; text with frequent
        // escapes is better served by the emitter's 16-byte loop.
        while from + 32 <= input.len() {
            let h0 = neon::hit_vec::<MODE_STANDARD>(vld1q_u8(input.as_ptr().add(from)));
            let h1 = neon::hit_vec::<MODE_STANDARD>(vld1q_u8(input.as_ptr().add(from + 16)));
            if vmaxvq_u8(vorrq_u8(h0, h1)) == 0 {
                from += 32;
                continue;
            }
            let m0 = neon::nib_mask(h0);
            let i = if m0 != 0 {
                from + (m0.trailing_zeros() as usize) / 4
            } else {
                from + 16 + (neon::nib_mask(h1).trailing_zeros() as usize) / 4
            };
            return Some((i, input[i]));
        }
        while from + 16 <= input.len() {
            let mask = neon::hit_mask::<MODE_STANDARD>(vld1q_u8(input.as_ptr().add(from)));
            if mask != 0 {
                let i = from + (mask.trailing_zeros() as usize) / 4;
                return Some((i, input[i]));
            }
            from += 16;
        }

        // Backwards in-bounds tail; documents shorter than one vector
        // take the scalar loop instead.
        if from < input.len() && input.len() >= 16 {
            return neon::tail_find::<MODE_STANDARD>(input, from).map(|i| (i, input[i]));
        }
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: same bounds argument as the aarch64 block above.
    unsafe {
        use crate::scan::{MODE_STANDARD, x86};
        use std::arch::x86_64::_mm_loadu_si128;
        // AVX2 (32 bytes per step) is universal on production x86 since
        // Haswell/Zen 1; the detection result is cached by std, so this is
        // one predictable branch per call.
        if std::arch::is_x86_feature_detected!("avx2") {
            return find_special_avx2(input, from);
        }
        while from + 16 <= input.len() {
            let mask = x86::sse2_hit_mask::<MODE_STANDARD>(_mm_loadu_si128(
                input.as_ptr().add(from).cast(),
            ));
            if mask != 0 {
                let i = from + mask.trailing_zeros() as usize;
                return Some((i, input[i]));
            }
            from += 16;
        }
        // Backwards in-bounds tail; sub-vector documents go scalar.
        if from < input.len() && input.len() >= 16 {
            return x86::sse2_tail_find::<MODE_STANDARD>(input, from).map(|i| (i, input[i]));
        }
    }
    // Sub-vector documents and non-SIMD hosts.
    find_special_scalar(input, from)
}

/// AVX2 variant of [`find_special`]: 32 bytes per step, backwards
/// in-bounds tail. Same standard hit set, from [`crate::scan`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn find_special_avx2(input: &[u8], mut from: usize) -> Option<(usize, u8)> {
    // SAFETY: every 32-byte load starts at an offset with `offset + 32 <=
    // input.len()` (loop condition; the backwards tail loads at exactly
    // `input.len() - 32`). AVX2 availability is the caller's obligation.
    unsafe {
        use crate::scan::{MODE_STANDARD, x86};
        use std::arch::x86_64::_mm256_loadu_si256;
        while from + 32 <= input.len() {
            let mask = x86::avx2_hit_mask::<MODE_STANDARD>(_mm256_loadu_si256(
                input.as_ptr().add(from).cast(),
            ));
            if mask != 0 {
                let i = from + mask.trailing_zeros() as usize;
                return Some((i, input[i]));
            }
            from += 32;
        }
        // Backwards in-bounds tail; sub-vector documents go scalar (rare;
        // at most 31 bytes, once per call).
        if from < input.len() && input.len() >= 32 {
            return x86::avx2_tail_find::<MODE_STANDARD>(input, from).map(|i| (i, input[i]));
        }
        find_special_scalar(input, from)
    }
}

/// Hex digit values; `0xFF` marks a non-hex byte. Sentinel bits
/// accumulate under OR so [`hex4`] validates all four digits with one
/// branch instead of one per digit, measured on `\u`-dense documents
/// (activitypub is one `\uXXXX` per ~52 string bytes).
const HEX_TABLE: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = i as u8;
        i += 1;
    }
    let mut i = 0;
    while i < 6 {
        t[b'a' as usize + i] = 10 + i as u8;
        t[b'A' as usize + i] = 10 + i as u8;
        i += 1;
    }
    t
};

#[inline(always)]
fn hex4(input: &[u8], at: usize) -> Result<u32, ScalarError> {
    if at + 4 > input.len() {
        return Err(ScalarError::InvalidUnicodeEscape);
    }
    let mut v: u32 = 0;
    let mut bad: u8 = 0;
    for &b in &input[at..at + 4] {
        let d = HEX_TABLE[b as usize];
        bad |= d;
        v = v << 4 | (d & 0x0F) as u32;
    }
    if bad & 0x80 != 0 {
        return Err(ScalarError::InvalidUnicodeEscape);
    }
    Ok(v)
}

/// Encode `cp` as UTF-8 (or WTF-8 for lone surrogates) at `dst`, returning
/// the encoded length (1-4).
///
/// # Safety
///
/// `dst` must have 4 writable bytes.
#[inline(always)]
unsafe fn write_utf8_raw(dst: *mut u8, cp: u32) -> usize {
    // SAFETY: each arm writes exactly the count it returns (at most 4
    // bytes) within the caller-provided capacity (see # Safety).
    unsafe {
        match cp {
            0..=0x7F => {
                dst.write(cp as u8);
                1
            }
            0x80..=0x7FF => {
                dst.write(0xC0 | (cp >> 6) as u8);
                dst.add(1).write(0x80 | (cp & 0x3F) as u8);
                2
            }
            0x800..=0xFFFF => {
                dst.write(0xE0 | (cp >> 12) as u8);
                dst.add(1).write(0x80 | ((cp >> 6) & 0x3F) as u8);
                dst.add(2).write(0x80 | (cp & 0x3F) as u8);
                3
            }
            _ => {
                dst.write(0xF0 | (cp >> 18) as u8);
                dst.add(1).write(0x80 | ((cp >> 12) & 0x3F) as u8);
                dst.add(2).write(0x80 | ((cp >> 6) & 0x3F) as u8);
                dst.add(3).write(0x80 | (cp & 0x3F) as u8);
                4
            }
        }
    }
}

/// Parse the string whose opening quote is at `quote_idx`.
///
/// Returns the decoded content and the index one past the closing quote.
///
/// # Safety
///
/// `input` must be valid UTF-8 and `input[quote_idx]` must be the `"`
/// opening a string token (the tokenizer establishes both before calling
/// in). Raw copied spans are handed out as `&str` without re-validation,
/// so invalid UTF-8 here is undefined behavior, not an error.
#[inline]
pub unsafe fn parse_string<'j, 's>(
    input: &'j [u8],
    quote_idx: usize,
    scratch: &'s mut Vec<u8>,
) -> Result<(StrPart<'j, 's>, usize), ScalarError> {
    let start = quote_idx + 1;

    let (first_special, byte) =
        find_special(input, start).ok_or(ScalarError::UnterminatedString)?;

    if byte == b'"' {
        let span = &input[start..first_special];
        // SAFETY: input is valid UTF-8 and both boundaries are at ASCII
        // quotes, so the span is valid UTF-8.
        let s = unsafe { std::str::from_utf8_unchecked(span) };
        return Ok((StrPart::Borrowed(s), first_special + 1));
    }
    if byte != b'\\' {
        return Err(ScalarError::ControlCharacterInString);
    }

    // Escape path: decode into scratch with fused scan+store, where every
    // clean span is copied by the same vector step that scans it (scan-
    // then-copy touches the bytes twice). Decoded output never exceeds
    // the source (every escape shrinks: `\n` 2→1, `\uXXXX` 6→≤3, surrogate
    // pair 12→4), so one reservation makes the whole decode realloc-free
    // and every store below is within capacity by construction.
    scratch.clear();
    scratch.reserve((input.len() - start) + DECODE_SLACK);
    let base = scratch.as_mut_ptr();
    let mut scratch_raw = false;

    // Prefix before the first escape: already scanned by find_special, so
    // a plain copy is all that is owed.
    let mut w = first_special - start;
    // SAFETY: reserved above; the source range was bounds-checked by
    // find_special.
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr().add(start), base, w);
    }
    let mut i = first_special;

    loop {
        // input[i] == '\\'
        let esc = *input.get(i + 1).ok_or(ScalarError::UnterminatedString)?;
        i += 2;
        // SAFETY: all branches write within capacity per the shrink
        // invariant above.
        unsafe {
            match esc {
                b'"' => push_byte_raw(base, &mut w, b'"'),
                b'\\' => push_byte_raw(base, &mut w, b'\\'),
                b'/' => push_byte_raw(base, &mut w, b'/'),
                b'b' => push_byte_raw(base, &mut w, 0x08),
                b'f' => push_byte_raw(base, &mut w, 0x0C),
                b'n' => push_byte_raw(base, &mut w, b'\n'),
                b'r' => push_byte_raw(base, &mut w, b'\r'),
                b't' => push_byte_raw(base, &mut w, b'\t'),
                b'u' => {
                    let cp = hex4(input, i)?;
                    i += 4;
                    if (0xD800..=0xDBFF).contains(&cp) {
                        // High surrogate: require a low surrogate.
                        if input.get(i) == Some(&b'\\') && input.get(i + 1) == Some(&b'u') {
                            let lo = hex4(input, i + 2)?;
                            if !(0xDC00..=0xDFFF).contains(&lo) {
                                return Err(ScalarError::LoneSurrogate);
                            }
                            i += 6;
                            let combined = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            w += write_utf8_raw(base.add(w), combined);
                        } else {
                            return Err(ScalarError::LoneSurrogate);
                        }
                    } else if (0xDC00..=0xDFFF).contains(&cp) {
                        // Lone low surrogate: lenient parsers accept it and
                        // emit the code point's raw (WTF-8) bytes. Mirror
                        // that; the scratch content is no longer valid UTF-8.
                        w += write_utf8_raw(base.add(w), cp);
                        scratch_raw = true;
                    } else {
                        w += write_utf8_raw(base.add(w), cp);
                    }
                }
                _ => return Err(ScalarError::InvalidEscape),
            }
        }

        // Escape runs are common (76% of tolstoy's escapes are followed
        // by another, from `\r\n` text; 24% of activitypub's, from `<…`
        // HTML): one byte of lookahead skips a full-width vector scan
        // that would return at bit 0.
        if input.get(i) == Some(&b'\\') {
            continue;
        }

        // SAFETY: dst capacity per the shrink invariant.
        let (next, byte, next_w) =
            unsafe { decode_span(input, i, base, w) }.ok_or(ScalarError::UnterminatedString)?;
        w = next_w;
        i = next;
        if byte == b'"' {
            // SAFETY: exactly `w` initialized bytes were written above.
            unsafe {
                scratch.set_len(w);
            }
            let part = if scratch_raw {
                StrPart::DecodedRaw(scratch.as_slice())
            } else {
                // SAFETY: every decode path above writes valid UTF-8
                // (WTF-8 content takes the DecodedRaw arm instead).
                StrPart::Decoded(unsafe { std::str::from_utf8_unchecked(scratch) })
            };
            return Ok((part, i + 1));
        }
        if byte != b'\\' {
            return Err(ScalarError::ControlCharacterInString);
        }
    }
}

/// Slack beyond the source length reserved for the decode buffer: covers
/// the speculative full-width vector store of a partial final chunk.
const DECODE_SLACK: usize = 16;

/// Write one byte at `base + w` and advance the cursor.
///
/// # Safety
///
/// Capacity at `base + w` (callers hold the decode shrink invariant).
#[inline(always)]
unsafe fn push_byte_raw(base: *mut u8, w: &mut usize, b: u8) {
    // SAFETY: capacity at `base + w` per this function's contract.
    unsafe {
        base.add(*w).write(b);
    }
    *w += 1;
}

/// Copy clean bytes from `input[i..]` to `dst + w` while scanning for the
/// next special byte (`"`, `\`, control): one fused pass over
/// [`crate::scan`]'s per-ISA `copy_scan` step. Returns (special position,
/// special byte, updated write cursor), or `None` if the input ended
/// without one (unterminated string). The final partial chunk is staged
/// zero-padded on the stack ([`crate::scan::padded_tail`]); the padding
/// classifies as controls and is masked out by the remainder truncation.
///
/// # Safety
///
/// `dst` must have `w + (input.len() - i) + DECODE_SLACK` writable bytes:
/// clean copies advance byte-for-byte and the partial final chunk stores a
/// full vector width into the slack.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn decode_span(
    input: &[u8],
    mut i: usize,
    dst: *mut u8,
    mut w: usize,
) -> Option<(usize, u8, usize)> {
    use crate::scan::{MODE_STANDARD, neon, padded_tail};
    // SAFETY: source reads are whole 16-byte chunks bounded by the loop
    // condition, or the 16-byte stack staging buffer; every 16-byte store
    // lands within the caller's capacity contract.
    unsafe {
        while i + 16 <= input.len() {
            let mask = neon::copy_scan::<MODE_STANDARD>(input.as_ptr().add(i), dst.add(w));
            if mask != 0 {
                let k = (mask.trailing_zeros() as usize) / 4;
                return Some((i + k, input[i + k], w + k));
            }
            i += 16;
            w += 16;
        }
        let rem = input.len() - i;
        if rem == 0 {
            return None;
        }
        let chunk = padded_tail::<16>(&input[i..]);
        let mask = neon::copy_scan::<MODE_STANDARD>(chunk.as_ptr(), dst.add(w))
            & (u64::MAX >> (64 - 4 * rem));
        if mask != 0 {
            let k = (mask.trailing_zeros() as usize) / 4;
            Some((i + k, input[i + k], w + k))
        } else {
            None
        }
    }
}

/// x86 variant of [`decode_span`] (SSE2 step; same contract).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn decode_span(
    input: &[u8],
    mut i: usize,
    dst: *mut u8,
    mut w: usize,
) -> Option<(usize, u8, usize)> {
    use crate::scan::{MODE_STANDARD, padded_tail, x86};
    // SAFETY: same bounds and capacity argument as the aarch64 variant.
    unsafe {
        while i + 16 <= input.len() {
            let mask = x86::sse2_copy_scan::<MODE_STANDARD>(input.as_ptr().add(i), dst.add(w));
            if mask != 0 {
                let k = mask.trailing_zeros() as usize;
                return Some((i + k, input[i + k], w + k));
            }
            i += 16;
            w += 16;
        }
        let rem = input.len() - i;
        if rem == 0 {
            return None;
        }
        let chunk = padded_tail::<16>(&input[i..]);
        let mask =
            x86::sse2_copy_scan::<MODE_STANDARD>(chunk.as_ptr(), dst.add(w)) & ((1u64 << rem) - 1);
        if mask != 0 {
            let k = mask.trailing_zeros() as usize;
            Some((i + k, input[i + k], w + k))
        } else {
            None
        }
    }
}

/// Non-SIMD variant of [`decode_span`]: copy and classify per byte (same
/// contract).
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
unsafe fn decode_span(
    input: &[u8],
    mut i: usize,
    dst: *mut u8,
    mut w: usize,
) -> Option<(usize, u8, usize)> {
    while i < input.len() {
        let b = input[i];
        if b == b'"' || b == b'\\' || b < 0x20 {
            return Some((i, b, w));
        }
        // SAFETY: capacity per this function's contract.
        unsafe {
            dst.add(w).write(b);
        }
        i += 1;
        w += 1;
    }
    None
}

/// A parsed JSON number, split by representation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Number<'j> {
    /// Integer representable in `i64`.
    Int(i64),
    /// The integer literal `-0`, kept distinct so IEEE-sign-preserving
    /// hosts (JavaScript's `-0`) don't lose the sign; integer-preserving
    /// hosts fold it to plain 0 (the [`crate::Sink`] default does).
    NegativeZero,
    /// Floating-point number, exactly rounded.
    Float(f64),
    /// Integer too large for i64, as its validated ASCII digits (sign
    /// included).
    Big(&'j str),
}

/// Parse 8 ASCII digits loaded little-endian (first digit in the low byte),
/// using the `fast_float` / simdjson multiply-reduce trick.
#[inline(always)]
fn parse_8digits_le(mut v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064; // 100 + (10^6 << 32)
    const MUL2: u64 = 0x0000_2710_0000_0001; // 1 + (10^4 << 32)
    v = v.wrapping_sub(0x3030_3030_3030_3030);
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let v1 = (v & MASK).wrapping_mul(MUL1);
    let v2 = ((v >> 16) & MASK).wrapping_mul(MUL2);
    ((v1.wrapping_add(v2) >> 32) as u32) as u64
}

/// Parse 4 ASCII digits loaded little-endian.
#[inline(always)]
fn parse_4digits_le(mut v: u32) -> u64 {
    v = v.wrapping_sub(0x3030_3030);
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    (((v & 0xFF) * 100) + ((v >> 16) & 0xFF)) as u64
}

/// Fused digit-run scan: validates AND accumulates in a single pass, the
/// simdjson technique. When a chunk is only partially digits, the
/// match mask's trailing zeros say exactly how many, and they are consumed
/// from the already-loaded word (4 at a time, then bytes) with no reload.
/// The accumulator wraps on overlong runs; callers only use it when the
/// digit count fits u64.
#[inline(always)]
fn scan_digits_acc(input: &[u8], mut i: usize, acc: &mut u64) -> usize {
    const HI: u64 = 0xF0F0_F0F0_F0F0_F0F0;
    const ALL_DIGITS: u64 = 0x3333_3333_3333_3333;

    while i + 8 <= input.len() {
        let chunk = u64::from_le_bytes(input[i..i + 8].try_into().unwrap());
        let matched = (chunk & HI) | ((chunk.wrapping_add(0x0606_0606_0606_0606) & HI) >> 4);
        if matched == ALL_DIGITS {
            *acc = acc
                .wrapping_mul(100_000_000)
                .wrapping_add(parse_8digits_le(chunk));
            i += 8;
            continue;
        }

        let mut consecutive = ((matched ^ ALL_DIGITS).trailing_zeros() / 8) as usize;
        if consecutive >= 4 {
            *acc = acc
                .wrapping_mul(10_000)
                .wrapping_add(parse_4digits_le(chunk as u32));
            i += 4;
            consecutive -= 4;
        }
        while consecutive > 0 {
            *acc = acc.wrapping_mul(10).wrapping_add((input[i] - b'0') as u64);
            i += 1;
            consecutive -= 1;
        }
        return i;
    }

    while let Some(&b) = input.get(i) {
        if !b.is_ascii_digit() {
            break;
        }
        *acc = acc.wrapping_mul(10).wrapping_add((b - b'0') as u64);
        i += 1;
    }
    i
}

/// Parse a number starting at `start`. Returns the value and the exclusive
/// end index. Validates the strict JSON grammar in a single scan; floats are
/// built from the already-scanned digit parts (minimal-lexical, the
/// Eisel-Lemire implementation `serde_json` used) with no string re-parse.
///
/// # Panics
///
/// If `start` is out of bounds for `input` (the tokenizer always passes
/// the offset of a byte it just read).
#[inline]
pub fn parse_number(input: &[u8], start: usize) -> Result<(Number<'_>, usize), ScalarError> {
    let mut i = start;
    let negative = input[i] == b'-';
    if negative {
        i += 1;
    }

    // Mantissa is accumulated in the same pass that validates the digit runs
    // (fused, like simdjson). It wraps for overlong runs and
    // is only consumed when the digit count fits u64.
    let mut mantissa: u64 = 0;

    let int_start = i;
    match input.get(i) {
        Some(b'0') => i += 1, // no leading zeros: '0' must stand alone
        Some(b'1'..=b'9') => i = scan_digits_acc(input, i, &mut mantissa),
        _ => return Err(ScalarError::InvalidNumber),
    }
    let int_end = i;

    let mut frac_start = i;
    let mut frac_end = i;
    if input.get(i) == Some(&b'.') {
        i += 1;
        frac_start = i;
        i = scan_digits_acc(input, i, &mut mantissa);
        frac_end = i;
        if frac_end == frac_start {
            return Err(ScalarError::InvalidNumber);
        }
    }

    let mut explicit_exp: i64 = 0;
    let mut has_exp = false;
    if matches!(input.get(i), Some(b'e' | b'E')) {
        has_exp = true;
        i += 1;
        let exp_negative = match input.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let exp_start = i;
        let mut abs_exp: u64 = 0;
        i = scan_digits_acc(input, i, &mut abs_exp);
        if i == exp_start {
            return Err(ScalarError::InvalidNumber);
        }
        // Saturate huge exponents; they resolve to ±inf / 0 downstream.
        // Only significant digits count: the accumulator wraps on long
        // runs, but leading zeros ("e00…001") never contribute to it.
        // (Found by fuzzing: "1e00000000000000000000000" is 1e0, not inf.)
        let mut sig_start = exp_start;
        while input.get(sig_start) == Some(&b'0') {
            sig_start += 1;
        }
        explicit_exp = if i - sig_start >= 19 || abs_exp > i64::MAX as u64 {
            i64::MAX
        } else {
            abs_exp as i64
        };
        if exp_negative {
            explicit_exp = explicit_exp.wrapping_neg();
        }
    }

    if let Some(&b) = input.get(i)
        && !is_token_boundary(b)
    {
        return Err(ScalarError::TrailingGarbageAfterScalar);
    }

    if frac_end == frac_start && !has_exp {
        let digits = int_end - int_start;
        if digits <= 19 {
            if negative {
                if mantissa == 0 {
                    return Ok((Number::NegativeZero, i));
                }
                if mantissa <= i64::MAX as u64 + 1 {
                    return Ok((Number::Int((mantissa as i64).wrapping_neg()), i));
                }
            } else if i64::try_from(mantissa).is_ok() {
                return Ok((Number::Int(mantissa as i64), i));
            }
        }
        // SAFETY: the span was scanned above as an optional sign plus
        // ASCII digits, so it is valid UTF-8 regardless of the rest of
        // the input.
        let digits = unsafe { std::str::from_utf8_unchecked(&input[start..i]) };
        return Ok((Number::Big(digits), i));
    }

    // Float. Count significant digits (leading zeros don't count) to decide
    // whether the mantissa fits u64 untruncated.
    let int_len = int_end - int_start;
    let frac_len = frac_end - frac_start;
    let sig = if input[int_start] == b'0' && int_len == 1 {
        frac_len
            - input[frac_start..frac_end]
                .iter()
                .take_while(|&&b| b == b'0')
                .count()
    } else {
        int_len + frac_len
    };

    if sig <= 19 {
        let w = mantissa;
        let exp10 = explicit_exp.saturating_sub(frac_len as i64);

        let f = 'fast: {
            if w == 0 {
                break 'fast Some(0.0);
            }
            // Clinger: both operands exactly representable; one multiply.
            if w < (1 << 53) && (-22..=22).contains(&exp10) {
                let m = w as f64;
                break 'fast Some(if exp10 < 0 {
                    m / crate::float::POW10[(-exp10) as usize]
                } else {
                    m * crate::float::POW10[exp10 as usize]
                });
            }
            if exp10 < -342 {
                break 'fast Some(0.0);
            }
            if exp10 > 308 {
                break 'fast Some(f64::INFINITY);
            }
            crate::float::eisel_lemire(w, exp10)
        };
        if let Some(f) = f {
            return Ok((Number::Float(if negative { -f } else { f }), i));
        }
    }

    // Truncated mantissa or ambiguous rounding: string slow path.
    let f =
        fast_float2::parse::<f64, _>(&input[start..i]).map_err(|_| ScalarError::InvalidNumber)?;
    Ok((Number::Float(f), i))
}

/// Literal kinds recognized at scalar starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
    /// `true`
    True,
    /// `false`
    False,
    /// `null`
    Null,
}

/// Parse `true` / `false` / `null` starting at `start`. Returns the literal
/// and the exclusive end index.
///
/// # Panics
///
/// If `start` is out of bounds for `input` (the tokenizer always passes
/// the offset of a byte it just read).
#[inline]
pub fn parse_literal(input: &[u8], start: usize) -> Result<(Literal, usize), ScalarError> {
    let rest = &input[start..];
    let (lit, end) = if rest.starts_with(b"true") {
        (Literal::True, start + 4)
    } else if rest.starts_with(b"false") {
        (Literal::False, start + 5)
    } else if rest.starts_with(b"null") {
        (Literal::Null, start + 4)
    } else {
        return Err(ScalarError::InvalidLiteral);
    };
    if let Some(&b) = input.get(end)
        && !is_token_boundary(b)
    {
        return Err(ScalarError::TrailingGarbageAfterScalar);
    }
    Ok((lit, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str_owned(doc: &[u8]) -> Result<(String, usize), ScalarError> {
        let mut scratch = Vec::new();
        // SAFETY: test inputs are valid UTF-8 literals starting with `"`.
        let (part, end) = unsafe { parse_string(doc, 0, &mut scratch) }?;
        let s = match part {
            StrPart::Borrowed(s) | StrPart::Decoded(s) => s.to_string(),
            StrPart::DecodedRaw(b) => String::from_utf8_lossy(b).into_owned(),
        };
        Ok((s, end))
    }

    /// Raw byte access, for asserting WTF-8 output of lone surrogates.
    fn parse_str_raw_bytes(doc: &[u8]) -> Result<(Vec<u8>, bool), ScalarError> {
        let mut scratch = Vec::new();
        // SAFETY: test inputs are valid UTF-8 literals starting with `"`.
        let (part, _) = unsafe { parse_string(doc, 0, &mut scratch) }?;
        Ok(match part {
            StrPart::Borrowed(s) | StrPart::Decoded(s) => (s.as_bytes().to_vec(), false),
            StrPart::DecodedRaw(b) => (b.to_vec(), true),
        })
    }

    #[test]
    fn plain_string() {
        assert_eq!(
            parse_str_owned(br#""hello", tail"#).unwrap(),
            ("hello".into(), 7)
        );
    }

    #[test]
    fn escapes() {
        assert_eq!(
            parse_str_owned(br#""a\"b\\c\nd\t""#).unwrap().0,
            "a\"b\\c\nd\t"
        );
    }

    #[test]
    fn unicode_escape() {
        assert_eq!(parse_str_owned(b"\"A\\u00e9\"").unwrap().0, "A\u{e9}");
    }

    #[test]
    fn surrogate_pair() {
        assert_eq!(
            parse_str_owned(b"\"\\ud83d\\ude00\"").unwrap().0,
            "\u{1F600}"
        );
    }

    #[test]
    fn lone_high_surrogate_rejected() {
        // A high surrogate must be followed by a low one; an error even
        // in lenient parsers.
        assert_eq!(
            parse_str_owned(br#""\ud800xx""#).unwrap_err(),
            ScalarError::LoneSurrogate
        );
        assert_eq!(
            parse_str_owned(b"\"\\ud800\"").unwrap_err(),
            ScalarError::LoneSurrogate
        );
    }

    #[test]
    fn lone_low_surrogate_emitted_as_wtf8() {
        // Lenient-parser parity: a lone low surrogate is accepted and its
        // code point emitted as raw (invalid-UTF-8) bytes. \uDFAA -> ED BE AA.
        let (bytes, raw) = parse_str_raw_bytes(b"\"\\uDFAA\"").unwrap();
        assert!(raw);
        assert_eq!(bytes, [0xED, 0xBE, 0xAA]);
        // With trailing content, like the JSONTestSuite corpus case.
        let (bytes, raw) = parse_str_raw_bytes(b"\"\\uDd1ea\"").unwrap();
        assert!(raw);
        assert_eq!(bytes, [0xED, 0xB4, 0x9E, b'a']);
    }

    #[test]
    fn control_char_rejected() {
        assert_eq!(
            parse_str_owned(b"\"a\x01b\"").unwrap_err(),
            ScalarError::ControlCharacterInString
        );
    }

    #[test]
    fn unterminated() {
        assert_eq!(
            parse_str_owned(br#""abc"#).unwrap_err(),
            ScalarError::UnterminatedString
        );
    }

    #[test]
    fn long_string_simd_path() {
        let mut doc = vec![b'"'];
        doc.extend(std::iter::repeat_n(b'x', 1000));
        doc.push(b'"');
        let (s, end) = parse_str_owned(&doc).unwrap();
        assert_eq!(s.len(), 1000);
        assert_eq!(end, 1002);
    }

    #[test]
    fn numbers() {
        assert_eq!(parse_number(b"0,", 0).unwrap(), (Number::Int(0), 1));
        assert_eq!(parse_number(b"-1", 0).unwrap(), (Number::Int(-1), 2));
        // `-0` keeps its sign as an event; `-0.0` is already a float.
        assert_eq!(parse_number(b"-0,", 0).unwrap(), (Number::NegativeZero, 2));
        assert_eq!(parse_number(b"-0.0", 0).unwrap().0, Number::Float(-0.0));
        assert_eq!(
            parse_number(b"9223372036854775807", 0).unwrap().0,
            Number::Int(i64::MAX)
        );
        assert_eq!(
            parse_number(b"-9223372036854775808", 0).unwrap().0,
            Number::Int(i64::MIN)
        );
        assert_eq!(parse_number(b"1.5", 0).unwrap().0, Number::Float(1.5));
        assert_eq!(parse_number(b"1e3", 0).unwrap().0, Number::Float(1000.0));
        assert_eq!(
            parse_number(b"-2.5e-2", 0).unwrap().0,
            Number::Float(-0.025)
        );
        assert_eq!(
            parse_number(b"123456789012345678901", 0).unwrap().0,
            Number::Big("123456789012345678901")
        );
    }

    /// Every string length around the vector widths, at the very end of
    /// an exact-size allocation, with the closing quote (and an escape)
    /// at each boundary-relevant position. Exercises the backwards
    /// in-bounds tail of `find_special` and the padded tail of
    /// `decode_span`.
    #[test]
    fn string_boundary_sweep() {
        for len in 0..=40 {
            let content = &"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"[..len];
            // Clean string, document ends at the closing quote.
            let doc: Box<[u8]> = format!("\"{content}\"").into_bytes().into_boxed_slice();
            let (s, end) = parse_str_owned(&doc).unwrap();
            assert_eq!(s, content, "len {len}");
            assert_eq!(end, doc.len(), "len {len}");
            // Escape at every position: decode goes through the scratch
            // (fused decode_span) path.
            for pos in 0..len {
                let mut inner = content.to_string().into_bytes();
                inner[pos] = b'\\';
                inner.insert(pos + 1, b'n');
                let doc: Box<[u8]> = format!("\"{}\"", String::from_utf8(inner).unwrap())
                    .into_bytes()
                    .into_boxed_slice();
                let (s, _) = parse_str_owned(&doc).unwrap();
                let mut expected = content.to_string();
                expected.replace_range(pos..=pos, "\n");
                assert_eq!(s, expected, "len {len} escape at {pos}");
            }
            // Unterminated: the tail must report None, not a phantom hit.
            let doc: Box<[u8]> = format!("\"{content}").into_bytes().into_boxed_slice();
            assert_eq!(
                parse_str_owned(&doc).unwrap_err(),
                ScalarError::UnterminatedString,
                "unterminated len {len}"
            );
        }
    }

    /// Leading zeros in an exponent are not significant digits: a long
    /// run of them must not trip the overflow saturation (fuzz finding).
    #[test]
    fn exponent_leading_zeros() {
        let long_zero = b"1e00000000000000000000000";
        assert_eq!(parse_number(long_zero, 0).unwrap().0, Number::Float(1.0));
        assert_eq!(
            parse_number(b"1e-00000000000000000000001", 0).unwrap().0,
            Number::Float(0.1)
        );
        assert_eq!(
            parse_number(b"9999999999999999991e00000000000000000000000", 0)
                .unwrap()
                .0,
            Number::Float(9999999999999999991.0)
        );
        // Genuinely huge exponents still saturate to infinity / zero.
        assert_eq!(
            parse_number(b"1e0000000000000000000009999", 0).unwrap().0,
            Number::Float(f64::INFINITY)
        );
        assert_eq!(
            parse_number(b"1e-0000000000000000000009999", 0).unwrap().0,
            Number::Float(0.0)
        );
    }

    #[test]
    fn bad_numbers() {
        for bad in [&b"01"[..], b"-", b"1.", b".5", b"1e", b"1e+", b"+1"] {
            assert!(parse_number(bad, 0).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn literals() {
        assert_eq!(parse_literal(b"true,", 0).unwrap(), (Literal::True, 4));
        assert_eq!(parse_literal(b"false]", 0).unwrap(), (Literal::False, 5));
        assert_eq!(parse_literal(b"null}", 0).unwrap(), (Literal::Null, 4));
        assert!(parse_literal(b"nul", 0).is_err());
        assert!(parse_literal(b"tru", 0).is_err());
    }
}
