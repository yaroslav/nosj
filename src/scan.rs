//! Shared byte-scanning primitives: the "does this byte need attention"
//! predicate in every form the crate needs: per-architecture SIMD hit
//! masks, a SWAR mask for 8-byte steps, a per-byte class table for scalar
//! tails, and the page-boundary guard that makes overreading vector tails
//! safe.
//!
//! Both the parser's string scan ([`crate::scalars`]) and the emitter's
//! escape scan ([`crate::emit`]) classify against the same base set
//! (`"`, `\`, and control characters below 0x20), so the predicate lives
//! here exactly once per architecture. The emitter adds two optional sets
//! via the `MODE` const parameter; the parser always scans
//! `MODE_STANDARD`.

// Const-generic discriminants for [`crate::emit::EscapeMode`]: stable Rust
// only allows primitive const-generic parameters, so the public enum is
// lowered to these at the dispatch point in `emit::escape_into`. Replace
// with `const MODE: EscapeMode` once `adt_const_params` stabilizes.
pub(crate) const MODE_STANDARD: u8 = 0;
pub(crate) const MODE_SCRIPT_SAFE: u8 = 1;
pub(crate) const MODE_ASCII_ONLY: u8 = 2;

// Per-byte class bits in [`ESCAPE_CLASS`].
pub(crate) const CLASS_STANDARD: u8 = 1 << 0;
pub(crate) const CLASS_SCRIPT_SAFE_EXTRA: u8 = 1 << 1;
pub(crate) const CLASS_ASCII_ONLY_EXTRA: u8 = 1 << 2;

/// The [`ESCAPE_CLASS`] bits that need attention under `MODE`.
pub(crate) const fn class_bits<const MODE: u8>() -> u8 {
    match MODE {
        MODE_SCRIPT_SAFE => CLASS_STANDARD | CLASS_SCRIPT_SAFE_EXTRA,
        MODE_ASCII_ONLY => CLASS_STANDARD | CLASS_ASCII_ONLY_EXTRA,
        _ => CLASS_STANDARD,
    }
}

/// Per-byte `CLASS_*` membership bits. One table load replaces five
/// compares on the scalar path.
pub(crate) static ESCAPE_CLASS: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        if b == b'"' || b == b'\\' || b < 0x20 {
            t[i] |= CLASS_STANDARD;
        }
        if b == b'/' || b == 0xE2 {
            t[i] |= CLASS_SCRIPT_SAFE_EXTRA;
        }
        if b >= 0x80 {
            t[i] |= CLASS_ASCII_ONLY_EXTRA;
        }
        i += 1;
    }
    t
};

/// Copy `n <= 32` bytes with a pair of overlapping word moves instead of a
/// `memcpy` call: tiny-copy call overhead is measurable on short-string
/// workloads (14% of a string-heavy generation profile; see
/// [`crate::emit::push_short`]).
///
/// # Safety
///
/// `n` readable bytes at `src`, `n` writable bytes at `dst`, and the two
/// ranges must not overlap.
#[inline(always)]
pub(crate) unsafe fn copy_small(src: *const u8, dst: *mut u8, n: usize) {
    debug_assert!(n <= 32);
    /// One overlapping pair: word-size bytes at offset 0 and at `n - size`
    /// cover `size..=2*size` bytes; the overlapped middle is written twice
    /// with identical data.
    macro_rules! word_pair {
        ($t:ty) => {{
            const SIZE: usize = size_of::<$t>();
            dst.cast::<$t>()
                .write_unaligned(src.cast::<$t>().read_unaligned());
            dst.add(n - SIZE)
                .cast::<$t>()
                .write_unaligned(src.add(n - SIZE).cast::<$t>().read_unaligned());
        }};
    }
    // SAFETY: each taken arm's word size was just tested `<= n`, so every
    // access lands within the caller-guaranteed `n` bytes at each pointer.
    // All accesses are explicitly unaligned.
    unsafe {
        if n >= 16 {
            word_pair!(u128);
        } else if n >= 8 {
            word_pair!(u64);
        } else if n >= 4 {
            word_pair!(u32);
        } else if n >= 2 {
            word_pair!(u16);
        } else if n == 1 {
            dst.write(src.read());
        }
    }
}

/// Stage a partial final chunk (`tail.len() < W`) in a zero-padded stack
/// buffer so full-width vector loads never read past the source
/// allocation. Reading past a slice's end is undefined behavior in Rust's
/// allocation model even when it cannot fault at the hardware level (the
/// page-guard trick this replaced), so tails load from this copy instead.
///
/// Zero padding classifies as a control character in every scan mode, so
/// callers must truncate result masks to the real remainder, which they
/// already do.
#[inline(always)]
pub(crate) fn padded_tail<const W: usize>(tail: &[u8]) -> [u8; W] {
    debug_assert!(tail.len() < W);
    let mut buf = [0u8; W];
    // SAFETY: `tail.len()` readable bytes at the slice's pointer,
    // `tail.len() < W` writable bytes in `buf`; distinct allocations.
    unsafe { copy_small(tail.as_ptr(), buf.as_mut_ptr(), tail.len()) };
    buf
}

/// Byte-parallel needs-attention test over 8 bytes: the escape scan for
/// hosts without SIMD, and the tests' cross-check reference. Both
/// detectors are the exact bit-hack forms (no false positives):
/// `hasvalue` for equality, `hasless(n<=128)` for the control range. One
/// bit set in the high bit of each matching byte; position is
/// `trailing_zeros() >> 3`.
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
#[inline(always)]
pub(crate) fn swar_escape_mask<const MODE: u8>(w: u64) -> u64 {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    #[inline(always)]
    fn eq(w: u64, b: u8) -> u64 {
        let v = w ^ (LO * u64::from(b));
        v.wrapping_sub(LO) & !v & HI
    }
    let mut m = eq(w, b'"') | eq(w, b'\\') | (w.wrapping_sub(LO * 0x20) & !w & HI);
    if MODE == MODE_SCRIPT_SAFE {
        m |= eq(w, b'/') | eq(w, 0xE2);
    }
    if MODE == MODE_ASCII_ONLY {
        m |= w & HI;
    }
    m
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use super::{MODE_ASCII_ONLY, MODE_SCRIPT_SAFE};
    use std::arch::aarch64::{
        uint8x16_t, vceqq_u8, vcgeq_u8, vcltq_u8, vdupq_n_u8, vget_lane_u64, vld1q_u8, vorrq_u8,
        vreinterpret_u64_u8, vreinterpretq_u16_u8, vshrn_n_u16, vst1q_u8,
    };

    /// Per-lane needs-attention flags (all-ones lanes) for 16 bytes.
    /// Kept separate from [`nib_mask`] so callers scanning wide can OR two
    /// hit vectors and early-exit on `vmaxvq_u8 == 0` (one instruction)
    /// before paying for any mask extraction.
    #[inline(always)]
    pub(crate) unsafe fn hit_vec<const MODE: u8>(v: uint8x16_t) -> uint8x16_t {
        // SAFETY: register-only NEON operations on an already-loaded
        // vector; NEON is baseline on aarch64.
        unsafe {
            let mut hit = vorrq_u8(
                vorrq_u8(
                    vceqq_u8(v, vdupq_n_u8(b'"')),
                    vceqq_u8(v, vdupq_n_u8(b'\\')),
                ),
                vcltq_u8(v, vdupq_n_u8(0x20)),
            );
            if MODE == MODE_SCRIPT_SAFE {
                hit = vorrq_u8(
                    hit,
                    vorrq_u8(vceqq_u8(v, vdupq_n_u8(b'/')), vceqq_u8(v, vdupq_n_u8(0xE2))),
                );
            }
            if MODE == MODE_ASCII_ONLY {
                hit = vorrq_u8(hit, vcgeq_u8(v, vdupq_n_u8(0x80)));
            }
            hit
        }
    }

    /// NEON has no movemask; the standard substitute narrows each 16-bit
    /// pair to its middle 4 bits (`vshrn` by 4), yielding a u64 with 4 bits
    /// per lane. Position of the first hit: `trailing_zeros() / 4`.
    #[inline(always)]
    pub(crate) unsafe fn nib_mask(hit: uint8x16_t) -> u64 {
        // SAFETY: register-only NEON operations; NEON is baseline on
        // aarch64.
        unsafe {
            vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
                hit,
            ))))
        }
    }

    /// [`hit_vec`] + [`nib_mask`] in one call, for 16-byte-at-a-time loops.
    #[inline(always)]
    pub(crate) unsafe fn hit_mask<const MODE: u8>(v: uint8x16_t) -> u64 {
        // SAFETY: register-only composition of the two functions above.
        unsafe { nib_mask(hit_vec::<MODE>(v)) }
    }

    /// The fused scan+store step: load 16 bytes from `src`, store them to
    /// `dst`, return the needs-attention mask (4 bits per lane). Shared by
    /// the emitter's fused escape loop and the parser's fused decode.
    ///
    /// # Safety
    ///
    /// 16 readable bytes at `src`, 16 writable bytes at `dst`.
    #[inline(always)]
    pub(crate) unsafe fn copy_scan<const MODE: u8>(src: *const u8, dst: *mut u8) -> u64 {
        // SAFETY: per this function's contract.
        unsafe {
            let v = vld1q_u8(src);
            vst1q_u8(dst, v);
            hit_mask::<MODE>(v)
        }
    }

    /// First needs-attention byte in the tail `[from, input.len())`, or
    /// `None`. Loads the buffer's LAST 16 bytes (fully in bounds,
    /// overlapping bytes before `from` the caller already scanned clean)
    /// and drops mask lanes before `from`, so a stale hit in the overlap
    /// can never surface.
    #[inline(always)]
    pub(crate) fn tail_find<const MODE: u8>(input: &[u8], from: usize) -> Option<usize> {
        let back = input.len().checked_sub(16)?;
        // Whole-vector remainders belong in the caller's main loop; the
        // lane shift below relies on `0 < from - back <= 15`.
        debug_assert!(from > back && from < input.len());
        // SAFETY: `back + 16 == input.len()`, so the load is entirely
        // inside `input`.
        let mask = unsafe { hit_mask::<MODE>(vld1q_u8(input.as_ptr().add(back))) }
            & (u64::MAX << (4 * (from - back)));
        if mask == 0 {
            None
        } else {
            Some(back + (mask.trailing_zeros() as usize) / 4)
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::{MODE_ASCII_ONLY, MODE_SCRIPT_SAFE};
    use std::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_min_epu8,
        _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8, _mm_storeu_si128, _mm256_cmpeq_epi8,
        _mm256_loadu_si256, _mm256_min_epu8, _mm256_movemask_epi8, _mm256_or_si256,
        _mm256_set1_epi8, _mm256_storeu_si256, _mm512_cmpeq_epi8_mask, _mm512_cmplt_epu8_mask,
        _mm512_loadu_si512, _mm512_movepi8_mask, _mm512_set1_epi8, _mm512_storeu_si512,
    };

    /// Needs-attention movemask for 16 bytes (1 bit per byte).
    #[inline(always)]
    pub(crate) unsafe fn sse2_hit_mask<const MODE: u8>(v: __m128i) -> u32 {
        // SAFETY: register-only SSE2 operations on an already-loaded
        // vector; SSE2 is baseline on x86_64.
        unsafe {
            // Unsigned v <= 0x1F via min(v, 0x1F) == v.
            let is_ctrl = _mm_cmpeq_epi8(_mm_min_epu8(v, _mm_set1_epi8(0x1F)), v);
            let mut hit = _mm_or_si128(
                _mm_or_si128(
                    _mm_cmpeq_epi8(v, _mm_set1_epi8(b'"' as i8)),
                    _mm_cmpeq_epi8(v, _mm_set1_epi8(b'\\' as i8)),
                ),
                is_ctrl,
            );
            if MODE == MODE_SCRIPT_SAFE {
                hit = _mm_or_si128(
                    hit,
                    _mm_or_si128(
                        _mm_cmpeq_epi8(v, _mm_set1_epi8(b'/' as i8)),
                        _mm_cmpeq_epi8(v, _mm_set1_epi8(0xE2u8 as i8)),
                    ),
                );
            }
            let mut mask = _mm_movemask_epi8(hit) as u32;
            if MODE == MODE_ASCII_ONLY {
                // The sign bit doubles as the >= 0x80 test.
                mask |= _mm_movemask_epi8(v) as u32;
            }
            mask
        }
    }

    /// Needs-attention movemask for 32 bytes. Callers must have verified
    /// AVX2 at runtime (universal on production x86 since Haswell/Zen 1).
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn avx2_hit_mask<const MODE: u8>(v: __m256i) -> u32 {
        let is_ctrl = _mm256_cmpeq_epi8(_mm256_min_epu8(v, _mm256_set1_epi8(0x1F)), v);
        let mut hit = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'"' as i8)),
                _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'\\' as i8)),
            ),
            is_ctrl,
        );
        if MODE == MODE_SCRIPT_SAFE {
            hit = _mm256_or_si256(
                hit,
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'/' as i8)),
                    _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0xE2u8 as i8)),
                ),
            );
        }
        let mut mask = _mm256_movemask_epi8(hit) as u32;
        if MODE == MODE_ASCII_ONLY {
            mask |= _mm256_movemask_epi8(v) as u32;
        }
        mask
    }

    /// The fused scan+store step: load 16 bytes from `src`, store them to
    /// `dst`, return the needs-attention mask (1 bit per byte). Shared by
    /// the emitter's fused escape loop and the parser's fused decode.
    ///
    /// # Safety
    ///
    /// 16 readable bytes at `src`, 16 writable bytes at `dst`.
    #[inline(always)]
    pub(crate) unsafe fn sse2_copy_scan<const MODE: u8>(src: *const u8, dst: *mut u8) -> u64 {
        // SAFETY: per this function's contract.
        unsafe {
            let v = _mm_loadu_si128(src.cast());
            _mm_storeu_si128(dst.cast(), v);
            u64::from(sse2_hit_mask::<MODE>(v))
        }
    }

    /// AVX2 [`sse2_copy_scan`]: 32 bytes per step.
    ///
    /// # Safety
    ///
    /// 32 readable bytes at `src`, 32 writable bytes at `dst`; AVX2
    /// verified at runtime by the dispatch site.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn avx2_copy_scan<const MODE: u8>(src: *const u8, dst: *mut u8) -> u64 {
        // SAFETY: per this function's contract.
        unsafe {
            let v = _mm256_loadu_si256(src.cast());
            _mm256_storeu_si256(dst.cast(), v);
            u64::from(avx2_hit_mask::<MODE>(v))
        }
    }

    /// Needs-attention mask for 64 bytes, straight into a `__mmask64`:
    /// AVX-512BW compares produce mask registers natively, so there is
    /// no movemask step and the unsigned control-byte compare exists
    /// directly. Callers must have verified AVX-512BW at runtime.
    /// Currently undispatched (emission rejected it; see emit.rs);
    /// kept for scan-only consumers.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    #[allow(dead_code)]
    pub(crate) unsafe fn avx512_hit_mask<const MODE: u8>(v: __m512i) -> u64 {
        let mut mask = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'"' as i8))
            | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\\' as i8))
            | _mm512_cmplt_epu8_mask(v, _mm512_set1_epi8(0x20));
        if MODE == MODE_SCRIPT_SAFE {
            mask |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'/' as i8))
                | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(0xE2u8 as i8));
        }
        if MODE == MODE_ASCII_ONLY {
            // The sign bit doubles as the >= 0x80 test.
            mask |= _mm512_movepi8_mask(v) as u64;
        }
        mask
    }

    /// AVX-512 [`sse2_copy_scan`]: 64 bytes per step.
    ///
    /// # Safety
    ///
    /// 64 readable bytes at `src`, 64 writable bytes at `dst`;
    /// AVX-512BW verified at runtime by the dispatch site.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    #[allow(dead_code)]
    pub(crate) unsafe fn avx512_copy_scan<const MODE: u8>(src: *const u8, dst: *mut u8) -> u64 {
        // SAFETY: per this function's contract.
        unsafe {
            let v = _mm512_loadu_si512(src.cast());
            _mm512_storeu_si512(dst.cast(), v);
            avx512_hit_mask::<MODE>(v)
        }
    }

    /// First needs-attention byte in the tail `[from, input.len())`, or
    /// `None`. Loads the buffer's LAST 16 bytes (fully in bounds,
    /// overlapping bytes before `from` the caller already scanned clean)
    /// and drops mask lanes before `from`, so a stale hit in the overlap
    /// can never surface.
    #[inline(always)]
    pub(crate) fn sse2_tail_find<const MODE: u8>(input: &[u8], from: usize) -> Option<usize> {
        let back = input.len().checked_sub(16)?;
        // Whole-vector remainders belong in the caller's main loop; the
        // lane shift below relies on `0 < from - back <= 15`.
        debug_assert!(from > back && from < input.len());
        // SAFETY: `back + 16 == input.len()`, so the load is entirely
        // inside `input`.
        let mask =
            unsafe { sse2_hit_mask::<MODE>(_mm_loadu_si128(input.as_ptr().add(back).cast())) }
                & (u32::MAX << (from - back));
        if mask == 0 {
            None
        } else {
            Some(back + mask.trailing_zeros() as usize)
        }
    }

    /// AVX2 [`sse2_tail_find`]: a 32-byte backwards window. AVX2 verified
    /// at runtime by the dispatch site.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn avx2_tail_find<const MODE: u8>(input: &[u8], from: usize) -> Option<usize> {
        let back = input.len().checked_sub(32)?;
        debug_assert!(from > back && from < input.len());
        // SAFETY: `back + 32 == input.len()`, so the load is entirely
        // inside `input`.
        let mask =
            unsafe { avx2_hit_mask::<MODE>(_mm256_loadu_si256(input.as_ptr().add(back).cast())) }
                & (u32::MAX << (from - back));
        if mask == 0 {
            None
        } else {
            Some(back + mask.trailing_zeros() as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar table is the reference; SWAR must agree byte-for-byte.
    #[test]
    fn swar_agrees_with_class_table() {
        fn check<const MODE: u8>() {
            for b in 0..=255u8 {
                let word = u64::from_le_bytes([b, b'a', b, 0x00, b'z', b, b'a', b]);
                let mask = swar_escape_mask::<MODE>(word);
                for (lane, &byte) in [b, b'a', b, 0x00, b'z', b, b'a', b].iter().enumerate() {
                    let expected = ESCAPE_CLASS[byte as usize] & class_bits::<MODE>() != 0;
                    let got = mask & (0x80 << (8 * lane)) != 0;
                    assert_eq!(got, expected, "mode {MODE} byte {byte:#x} lane {lane}");
                }
            }
        }
        check::<MODE_STANDARD>();
        check::<MODE_SCRIPT_SAFE>();
        check::<MODE_ASCII_ONLY>();
    }
}
