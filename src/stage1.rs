//! Stage 1: SIMD structural indexing (simdjson architecture, first-party).
//!
//! Produces the offsets of every token start in the input: structural
//! characters (`{}[]:,`) outside strings, the opening quote of every string,
//! and the first byte of every scalar run (numbers, literals). String
//! interiors and continuation bytes of scalars are never indexed, so stage 2
//! can treat consecutive index entries as consecutive tokens.
//!
//! Two implementations: a byte-exact scalar model (reference + non-aarch64
//! fallback) and a NEON 64-bytes-per-block implementation. Unit tests verify
//! they agree bit-for-bit.
//!
//! UTF-8 validation is intentionally absent: `&str` entry points already
//! guarantee validity, and `*_utf8_unchecked` entry points exist for hosts
//! whose runtime tracks string validity itself (cached validity flags are
//! common in language runtimes).

/// Carry state threaded across 64-byte blocks.
#[derive(Default)]
#[allow(clippy::struct_field_names)] // the `prev_` prefix is the point: these are carries
struct Carries {
    /// All-ones if the previous block ended inside a string.
    prev_in_string: u64,
    /// Bit 0 set if the first byte of the next block is escaped.
    prev_escaped: u64,
    /// Bit 63 of the previous block's nonquote-scalar mask.
    prev_scalar: u64,
}

/// One 64-byte block's per-class bit masks, produced by each ISA's
/// `classify_block`.
struct BlockMasks {
    backslash: u64,
    quote: u64,
    op: u64,
    ws: u64,
}

impl Carries {
    /// simdjson's odd/even backslash-run resolution: returns the mask of
    /// escaped characters (characters preceded by an unescaped backslash).
    #[inline(always)]
    fn find_escaped(&mut self, backslash: u64) -> u64 {
        const EVEN: u64 = 0x5555_5555_5555_5555;
        if backslash == 0 {
            let escaped = self.prev_escaped;
            self.prev_escaped = 0;
            return escaped;
        }
        let backslash = backslash & !self.prev_escaped;
        let follows_escape = backslash << 1 | self.prev_escaped;
        let odd_starts = backslash & !EVEN & !follows_escape;
        let (seq_start_on_even, carry) = odd_starts.overflowing_add(backslash);
        self.prev_escaped = u64::from(carry);
        let invert = seq_start_on_even << 1;
        (EVEN ^ invert) & follows_escape
    }

    /// Combine the per-class masks of one 64-byte block into the final
    /// structural mask, updating carries. Shared by both implementations.
    #[inline(always)]
    fn block_structurals(&mut self, backslash: u64, quote_raw: u64, op: u64, ws: u64) -> u64 {
        let escaped = self.find_escaped(backslash);
        let quote = quote_raw & !escaped;

        // Prefix XOR: bit i = parity of quote bits at positions <= i.
        let mut string_mask = prefix_xor(quote);
        string_mask ^= self.prev_in_string;
        // Interior + closing quote, excluding the opening quote.
        let string_tail = string_mask ^ quote;
        self.prev_in_string = ((string_mask as i64) >> 63) as u64;

        let scalar = !(op | ws);
        let nonquote_scalar = scalar & !quote;
        let follows_nonquote_scalar = nonquote_scalar << 1 | self.prev_scalar;
        self.prev_scalar = nonquote_scalar >> 63;

        (op | (scalar & !follows_nonquote_scalar)) & !string_tail
    }

    /// Bracket/separator mask for one block: `op` filtered to outside
    /// strings. The container-skip scanner consumes only these bits;
    /// no pseudo-scalar machinery, no index writes.
    #[inline(always)]
    fn block_ops(&mut self, backslash: u64, quote_raw: u64, op: u64) -> u64 {
        let escaped = self.find_escaped(backslash);
        let quote = quote_raw & !escaped;
        let mut string_mask = prefix_xor(quote);
        string_mask ^= self.prev_in_string;
        let string_tail = string_mask ^ quote;
        self.prev_in_string = ((string_mask as i64) >> 63) as u64;
        op & !string_tail
    }
}

/// Carry-less multiply by all-ones = prefix XOR. PMULL where the target has
/// it (Apple Silicon always does); six shift-xors otherwise.
#[inline(always)]
fn prefix_xor(x: u64) -> u64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: register-only PMULL; the `aes` target feature is
    // compile-time guaranteed by the cfg.
    unsafe {
        return std::arch::aarch64::vmull_p64(x, u64::MAX) as u64;
    }
    #[allow(unreachable_code)]
    {
        let mut x = x;
        x ^= x << 1;
        x ^= x << 2;
        x ^= x << 4;
        x ^= x << 8;
        x ^= x << 16;
        x ^= x << 32;
        x
    }
}

/// Write the set-bit offsets of `bits` into `out`, 8 slots per iteration with
/// unconditional writes (simdjson's flatten trick). The caller must have
/// reserved capacity for the worst case plus 8 slots of slack; garbage
/// written past the real count stays beyond `len()`.
///
/// With `PACK`, each entry is `((base + tz) << 8) | byte`, where the byte is
/// read from `bytes[tz]`, the just-classified block, hot in L1. Consumers
/// then get offset and token byte from a single load.
#[inline(always)]
fn flatten_bits<const PACK: bool>(out: &mut Vec<u32>, base: u32, mut bits: u64, bytes: *const u8) {
    if bits == 0 {
        return;
    }
    let cnt = bits.count_ones() as usize;
    let start = out.len();
    debug_assert!(out.capacity() >= start + cnt + 8);
    // SAFETY: callers reserve `input.len() + 8` entries up front; total
    // structural count never exceeds the input length, and each 8-entry
    // block write below stays within that `+ 8` slack (`set_len` covers
    // only the `cnt` real entries).
    unsafe {
        let ptr = out.as_mut_ptr().add(start);
        let mut written = 0usize;
        while written < cnt {
            for k in 0..8 {
                let tz = bits.trailing_zeros() & 63;
                let entry = if PACK {
                    ((base + tz) << 8) | *bytes.add(tz as usize) as u32
                } else {
                    base + tz
                };
                ptr.add(written + k).write(entry);
                bits &= bits.wrapping_sub(1);
            }
            written += 8;
        }
        out.set_len(start + cnt);
    }
}

/// simdjson's shufti classification tables: `LO[b & 0xF] & HI[b >> 4]` yields
/// class bits (bits 0-2 structural, bits 3-4 whitespace). Shared verbatim by
/// the scalar model and every SIMD kernel so all implementations agree
/// bit-for-bit (including the benign classification of some >0x7F bytes,
/// which only occur outside strings in invalid JSON).
pub(crate) const LO_NIBBLE: [u8; 16] = [16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0];
pub(crate) const HI_NIBBLE: [u8; 16] = [8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0];

#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
#[inline(always)]
fn classify_byte(b: u8) -> (bool, bool, bool, bool) {
    let backslash = b == b'\\';
    let quote = b == b'"';
    let class = LO_NIBBLE[(b & 0x0F) as usize] & HI_NIBBLE[(b >> 4) as usize];
    let op = class & 0x07 != 0;
    let ws = class & 0x18 != 0;
    (backslash, quote, op, ws)
}

/// Byte-exact scalar model of the SIMD algorithm (the tests'
/// cross-check reference, and the whole implementation on non-SIMD
/// hosts).
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
pub fn index_scalar(input: &[u8], out: &mut Vec<u32>) {
    index_scalar_impl::<false>(input, out);
}

#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn index_scalar_impl<const PACK: bool>(input: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(input.len() + 8);
    let mut carries = Carries::default();

    let mut chunks = input.chunks_exact(64);
    let mut base: u32 = 0;
    for chunk in &mut chunks {
        let (mut bs, mut quote, mut op, mut ws) = (0u64, 0u64, 0u64, 0u64);
        for (i, &b) in chunk.iter().enumerate() {
            let (cb, cq, co, cw) = classify_byte(b);
            bs |= (cb as u64) << i;
            quote |= (cq as u64) << i;
            op |= (co as u64) << i;
            ws |= (cw as u64) << i;
        }
        let structurals = carries.block_structurals(bs, quote, op, ws);
        flatten_bits::<PACK>(out, base, structurals, chunk.as_ptr());
        base += 64;
    }

    let rem = chunks.remainder();
    if !rem.is_empty() {
        // 64-byte buffer so packed-byte reads stay in bounds.
        let mut buf = [0u8; 64];
        buf[..rem.len()].copy_from_slice(rem);
        let (mut bs, mut quote, mut op, mut ws) = (0u64, 0u64, 0u64, 0u64);
        for (i, &b) in buf[..rem.len()].iter().enumerate() {
            let (cb, cq, co, cw) = classify_byte(b);
            bs |= (cb as u64) << i;
            quote |= (cq as u64) << i;
            op |= (co as u64) << i;
            ws |= (cw as u64) << i;
        }
        // Zero padding classifies as scalar; mask to the real length.
        let structurals =
            carries.block_structurals(bs, quote, op, ws) & (u64::MAX >> (64 - rem.len()));
        flatten_bits::<PACK>(out, base, structurals, buf.as_ptr());
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{
        uint8x16_t, vandq_u8, vceqq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u8, vpaddq_u8,
        vqtbl1q_u8, vreinterpretq_u64_u8, vshrq_n_u8, vtstq_u8,
    };

    /// Combine four 16-lane compare results into a 64-bit mask.
    #[inline(always)]
    unsafe fn movemask64(m0: uint8x16_t, m1: uint8x16_t, m2: uint8x16_t, m3: uint8x16_t) -> u64 {
        // SAFETY: register-only NEON operations; the transmute reinterprets
        // a 16-byte array as a 16-lane vector (same size, no invalid bit
        // patterns).
        unsafe {
            let bit_mask: uint8x16_t = std::mem::transmute([
                0x01u8, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10,
                0x20, 0x40, 0x80,
            ]);
            let t0 = vandq_u8(m0, bit_mask);
            let t1 = vandq_u8(m1, bit_mask);
            let t2 = vandq_u8(m2, bit_mask);
            let t3 = vandq_u8(m3, bit_mask);
            let sum0 = vpaddq_u8(t0, t1);
            let sum1 = vpaddq_u8(t2, t3);
            let sum0 = vpaddq_u8(sum0, sum1);
            let sum0 = vpaddq_u8(sum0, sum0);
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(sum0))
        }
    }

    use super::BlockMasks;

    /// simdjson's nibble-table classification: `lo_table[b & 0xF] & hi_table[b >> 4]`
    /// yields class bits (bits 0-2 mark structural characters, bits 3-4 mark
    /// JSON whitespace). Two table lookups replace ten byte compares.
    #[inline(always)]
    pub(super) unsafe fn classify_block(ptr: *const u8) -> BlockMasks {
        // SAFETY: the caller provides 64 readable bytes at `ptr` (whole
        // blocks of the input, or the zero-padded stack staging buffer);
        // everything else is register-only.
        unsafe {
            const LO: [u8; 16] = super::LO_NIBBLE;
            const HI: [u8; 16] = super::HI_NIBBLE;

            let v0 = vld1q_u8(ptr);
            let v1 = vld1q_u8(ptr.add(16));
            let v2 = vld1q_u8(ptr.add(32));
            let v3 = vld1q_u8(ptr.add(48));

            let lo_tbl = vld1q_u8(LO.as_ptr());
            let hi_tbl = vld1q_u8(HI.as_ptr());
            let nib = vdupq_n_u8(0x0F);

            macro_rules! classify {
                ($v:expr) => {{
                    let lo = vqtbl1q_u8(lo_tbl, vandq_u8($v, nib));
                    let hi = vqtbl1q_u8(hi_tbl, vshrq_n_u8::<4>($v));
                    vandq_u8(lo, hi)
                }};
            }

            let c0 = classify!(v0);
            let c1 = classify!(v1);
            let c2 = classify!(v2);
            let c3 = classify!(v3);

            let op_bits = vdupq_n_u8(0x07);
            let ws_bits = vdupq_n_u8(0x18);
            let op = movemask64(
                vtstq_u8(c0, op_bits),
                vtstq_u8(c1, op_bits),
                vtstq_u8(c2, op_bits),
                vtstq_u8(c3, op_bits),
            );
            let ws = movemask64(
                vtstq_u8(c0, ws_bits),
                vtstq_u8(c1, ws_bits),
                vtstq_u8(c2, ws_bits),
                vtstq_u8(c3, ws_bits),
            );

            macro_rules! eq_mask {
                ($byte:expr) => {{
                    let b = vdupq_n_u8($byte);
                    movemask64(
                        vceqq_u8(v0, b),
                        vceqq_u8(v1, b),
                        vceqq_u8(v2, b),
                        vceqq_u8(v3, b),
                    )
                }};
            }
            let backslash = eq_mask!(b'\\');
            let quote = eq_mask!(b'"');

            BlockMasks {
                backslash,
                quote,
                op,
                ws,
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use std::arch::x86_64::*;

    use super::BlockMasks;

    /// SSE2 is baseline on x86_64; no runtime detection required.
    #[inline(always)]
    pub(super) unsafe fn classify_block(ptr: *const u8) -> BlockMasks {
        // SAFETY: the caller provides 64 readable bytes at `ptr` (whole
        // blocks of the input, or the zero-padded stack staging buffer);
        // everything else is register-only SSE2.
        unsafe {
            let v0 = _mm_loadu_si128(ptr.cast());
            let v1 = _mm_loadu_si128(ptr.add(16).cast());
            let v2 = _mm_loadu_si128(ptr.add(32).cast());
            let v3 = _mm_loadu_si128(ptr.add(48).cast());

            macro_rules! eq_mask {
                ($byte:expr) => {{
                    let b = _mm_set1_epi8($byte as i8);
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(v0, b)) as u64)
                        | (_mm_movemask_epi8(_mm_cmpeq_epi8(v1, b)) as u64) << 16
                        | (_mm_movemask_epi8(_mm_cmpeq_epi8(v2, b)) as u64) << 32
                        | (_mm_movemask_epi8(_mm_cmpeq_epi8(v3, b)) as u64) << 48
                }};
            }

            let backslash = eq_mask!(b'\\');
            let quote = eq_mask!(b'"');
            let op = eq_mask!(b'{')
                | eq_mask!(b'}')
                | eq_mask!(b'[')
                | eq_mask!(b']')
                | eq_mask!(b':')
                | eq_mask!(b',');
            let ws = eq_mask!(b' ') | eq_mask!(b'\t') | eq_mask!(b'\n') | eq_mask!(b'\r');

            BlockMasks {
                backslash,
                quote,
                op,
                ws,
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{BlockMasks, Carries, flatten_bits};
    use std::arch::x86_64::*;

    /// One 64-byte block as [`BlockMasks`] for the container-end skip
    /// driver: two 32-byte vectors, the same nibble class tables as
    /// indexing. Whitespace is left zero; the skip consumes only
    /// backslash/quote/op bits. The table broadcasts hoist out of the
    /// block loop once this inlines into [`container_end_avx2`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn classify_block(ptr: *const u8) -> BlockMasks {
        // SAFETY: the caller provides 64 readable bytes at `ptr`;
        // everything else is register-only AVX2 (enabled on this fn and
        // verified at the dispatch site).
        unsafe {
            let lo128 = _mm_loadu_si128(super::LO_NIBBLE.as_ptr().cast());
            let hi128 = _mm_loadu_si128(super::HI_NIBBLE.as_ptr().cast());
            let lo_tbl = _mm256_broadcastsi128_si256(lo128);
            let hi_tbl = _mm256_broadcastsi128_si256(hi128);
            let nib = _mm256_set1_epi8(0x0F);
            let op_bits = _mm256_set1_epi8(0x07);
            let zero = _mm256_setzero_si256();
            let quote_v = _mm256_set1_epi8(b'"' as i8);
            let bs_v = _mm256_set1_epi8(b'\\' as i8);

            let v0 = _mm256_loadu_si256(ptr.cast());
            let v1 = _mm256_loadu_si256(ptr.add(32).cast());

            macro_rules! classify {
                ($v:expr) => {{
                    let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256($v, nib));
                    let hi = _mm256_shuffle_epi8(
                        hi_tbl,
                        _mm256_and_si256(_mm256_srli_epi16($v, 4), nib),
                    );
                    _mm256_and_si256(lo, hi)
                }};
            }
            let c0 = classify!(v0);
            let c1 = classify!(v1);

            macro_rules! nonzero_mask {
                ($c:expr, $bits:expr) => {{
                    !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256($c, $bits), zero))
                        as u32)
                }};
            }
            let op = nonzero_mask!(c0, op_bits) as u64 | (nonzero_mask!(c1, op_bits) as u64) << 32;

            macro_rules! eq_mask {
                ($b:expr) => {{
                    (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, $b)) as u32) as u64
                        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, $b)) as u32) as u64) << 32
                }};
            }
            let quote = eq_mask!(quote_v);
            let backslash = eq_mask!(bs_v);

            BlockMasks {
                backslash,
                quote,
                op,
                ws: 0,
            }
        }
    }

    /// [`super::container_end`] monomorphized under AVX2, so the
    /// classify calls inline and the table broadcasts hoist out of the
    /// block loop.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn container_end_avx2(input: &[u8], start: usize) -> Option<usize> {
        super::container_end_driver(input, start, |ptr| {
            // SAFETY: the driver passes pointers with 64 readable bytes
            // (whole in-bounds blocks or its zero-padded staging
            // buffer); AVX2 is inherited from the enclosing fn.
            unsafe { classify_block(ptr) }
        })
    }

    /// 32-byte-vector variant: half the loads per 64-byte block, native
    /// movemask, nibble tables broadcast across both 128-bit lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn index_avx2<const PACK: bool>(input: &[u8], out: &mut Vec<u32>) {
        // SAFETY: AVX2 availability is the caller's obligation (runtime
        // detection at the dispatch site). All loads cover whole 64-byte
        // blocks bounded by `full_blocks`, or the zero-padded staging
        // buffer; `flatten_bits` capacity comes from the reserve above.
        unsafe {
            out.clear();
            out.reserve(input.len() + 8);
            let mut carries = Carries::default();

            let lo128 = _mm_loadu_si128(super::LO_NIBBLE.as_ptr().cast());
            let hi128 = _mm_loadu_si128(super::HI_NIBBLE.as_ptr().cast());
            let lo_tbl = _mm256_broadcastsi128_si256(lo128);
            let hi_tbl = _mm256_broadcastsi128_si256(hi128);
            let nib = _mm256_set1_epi8(0x0F);
            let op_bits = _mm256_set1_epi8(0x07);
            let ws_bits = _mm256_set1_epi8(0x18);
            let zero = _mm256_setzero_si256();
            let quote_v = _mm256_set1_epi8(b'"' as i8);
            let bs_v = _mm256_set1_epi8(b'\\' as i8);

            // Nine parameters by design: the broadcast vector constants are
            // hoisted out of the block loop and passed in registers.
            #[allow(clippy::too_many_arguments, clippy::items_after_statements)]
            #[inline(always)]
            unsafe fn block_masks(
                ptr: *const u8,
                lo_tbl: __m256i,
                hi_tbl: __m256i,
                nib: __m256i,
                op_bits: __m256i,
                ws_bits: __m256i,
                zero: __m256i,
                quote_v: __m256i,
                bs_v: __m256i,
            ) -> (u64, u64, u64, u64) {
                // SAFETY: the caller provides 64 readable bytes at `ptr`;
                // everything else is register-only AVX2 (enabled on this
                // fn and verified at the dispatch site).
                unsafe {
                    let v0 = _mm256_loadu_si256(ptr.cast());
                    let v1 = _mm256_loadu_si256(ptr.add(32).cast());

                    macro_rules! classify {
                        ($v:expr) => {{
                            let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256($v, nib));
                            let hi = _mm256_shuffle_epi8(
                                hi_tbl,
                                _mm256_and_si256(_mm256_srli_epi16($v, 4), nib),
                            );
                            _mm256_and_si256(lo, hi)
                        }};
                    }
                    let c0 = classify!(v0);
                    let c1 = classify!(v1);

                    macro_rules! nonzero_mask {
                        ($c:expr, $bits:expr) => {{
                            !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                                _mm256_and_si256($c, $bits),
                                zero,
                            )) as u32)
                        }};
                    }
                    let op = nonzero_mask!(c0, op_bits) as u64
                        | (nonzero_mask!(c1, op_bits) as u64) << 32;
                    let ws = nonzero_mask!(c0, ws_bits) as u64
                        | (nonzero_mask!(c1, ws_bits) as u64) << 32;

                    macro_rules! eq_mask {
                        ($b:expr) => {{
                            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, $b)) as u32) as u64
                                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, $b)) as u32) as u64)
                                    << 32
                        }};
                    }
                    let quote = eq_mask!(quote_v);
                    let backslash = eq_mask!(bs_v);

                    (backslash, quote, op, ws)
                }
            }

            let len = input.len();
            let full_blocks = len / 64;
            let ptr = input.as_ptr();

            for block in 0..full_blocks {
                let (bs, quote, op, ws) = block_masks(
                    ptr.add(block * 64),
                    lo_tbl,
                    hi_tbl,
                    nib,
                    op_bits,
                    ws_bits,
                    zero,
                    quote_v,
                    bs_v,
                );
                let structurals = carries.block_structurals(bs, quote, op, ws);
                flatten_bits::<PACK>(out, (block * 64) as u32, structurals, ptr.add(block * 64));
            }

            let rem = len % 64;
            if rem != 0 {
                let mut buf = [0u8; 64];
                buf[..rem].copy_from_slice(&input[len - rem..]);
                let (bs, quote, op, ws) = block_masks(
                    buf.as_ptr(),
                    lo_tbl,
                    hi_tbl,
                    nib,
                    op_bits,
                    ws_bits,
                    zero,
                    quote_v,
                    bs_v,
                );
                let structurals =
                    carries.block_structurals(bs, quote, op, ws) & (u64::MAX >> (64 - rem));
                flatten_bits::<PACK>(out, (len - rem) as u32, structurals, buf.as_ptr());
            }
        }
    }
}

/// Inputs below this size can use the packed index encoding
/// (`(offset << 8) | byte` in each u32 entry).
pub const PACKED_INDEX_MAX_LEN: usize = 1 << 24;

/// Largest input the ordinary `u32`-offset index can address. Larger
/// documents must use the fused cursor, whose positions are `usize`.
pub const INDEX_MAX_LEN: usize = u32::MAX as usize;

/// True if a `len`-byte document fits the chosen index encoding: offsets
/// are `u32` (24 bits in the packed encoding, which shares each entry
/// with the token byte).
#[inline]
const fn index_len_ok(len: usize, packed: bool) -> bool {
    if packed {
        len < PACKED_INDEX_MAX_LEN
    } else {
        len <= INDEX_MAX_LEN
    }
}

#[inline(always)]
fn index_dispatch<const PACK: bool>(input: &[u8], out: &mut Vec<u32>) {
    #[cfg(target_arch = "aarch64")]
    {
        index_driver::<PACK>(input, out, |ptr| {
            // SAFETY: the driver passes pointers with 64 readable bytes
            // (whole in-bounds blocks or its zero-padded staging buffer).
            unsafe { neon::classify_block(ptr) }
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature checked at runtime.
            unsafe { avx2::index_avx2::<PACK>(input, out) };
        } else {
            index_driver::<PACK>(input, out, |ptr| {
                // SAFETY: the driver passes pointers with 64 readable
                // bytes (whole in-bounds blocks or its zero-padded
                // staging buffer).
                unsafe { sse2::classify_block(ptr) }
            });
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        index_scalar_impl::<PACK>(input, out);
    }
}

/// Shared body of the 16-byte-vector indexers: classify 64-byte blocks,
/// resolve structurals, flatten offsets. NEON and SSE2 differ only in
/// `classify`; AVX2 keeps its own paired-block shape.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn index_driver<const PACK: bool>(
    input: &[u8],
    out: &mut Vec<u32>,
    classify: impl Fn(*const u8) -> BlockMasks,
) {
    out.clear();
    out.reserve(input.len() + 8);
    let mut carries = Carries::default();

    let len = input.len();
    let full_blocks = len / 64;
    let ptr = input.as_ptr();

    for block in 0..full_blocks {
        // `block * 64 + 64 <= len`, so the slice-derived pointer has 64
        // readable bytes, the classify closure's requirement.
        let m = classify(input[block * 64..].as_ptr());
        let structurals = carries.block_structurals(m.backslash, m.quote, m.op, m.ws);
        // SAFETY: same in-bounds offset as the classify above.
        flatten_bits::<PACK>(out, (block * 64) as u32, structurals, unsafe {
            ptr.add(block * 64)
        });
    }

    let rem = len % 64;
    if rem != 0 {
        let mut buf = [0u8; 64];
        buf[..rem].copy_from_slice(&input[len - rem..]);
        let m = classify(buf.as_ptr());
        let structurals =
            carries.block_structurals(m.backslash, m.quote, m.op, m.ws) & (u64::MAX >> (64 - rem));
        flatten_bits::<PACK>(out, (len - rem) as u32, structurals, buf.as_ptr());
    }
}

/// Index `input`, writing token-start offsets into `out`.
///
/// # Panics
///
/// If `input` is longer than [`INDEX_MAX_LEN`] (offsets would truncate).
/// Larger documents parse through the fused cursor, whose positions are
/// `usize`.
pub fn index(input: &[u8], out: &mut Vec<u32>) {
    assert!(
        index_len_ok(input.len(), false),
        "input too large for u32 index offsets; use the fused cursor"
    );
    index_dispatch::<false>(input, out);
}

/// Index `input` with the packed encoding: each entry is
/// `(offset << 8) | token_byte`, so consumers read offset and byte from one
/// load. Measured neutral on Apple Silicon and not wired into any entry
/// point, but the machinery is kept and tested for hardware where the
/// trade goes the other way.
///
/// # Panics
///
/// If `input.len() >= PACKED_INDEX_MAX_LEN` (the packed offset field is
/// 24 bits).
#[cfg(test)]
pub fn index_packed(input: &[u8], out: &mut Vec<u32>) {
    assert!(
        index_len_ok(input.len(), true),
        "input too large for the packed index encoding (24-bit offsets)"
    );
    index_dispatch::<true>(input, out);
}

/// Offset one past the closer matching the container opener at `start`
/// (`input[start]` must be `{` or `[`), or `None` if the input ends
/// first. This is the cursor-mode container skip: 64-byte blocks are
/// classified exactly like indexing, but only bracket bits outside
/// strings are consumed: no index is built, and scanning stops at the
/// matching closer, so cost is proportional to the container's size.
///
/// Bracket *kinds* are not matched and non-bracket content is not
/// validated (the documented structural-skip semantics).
pub(crate) fn container_end(input: &[u8], start: usize) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        container_end_driver(input, start, |ptr| {
            // SAFETY: the driver passes pointers with 64 readable bytes
            // (whole in-bounds blocks or its zero-padded staging buffer).
            unsafe { neon::classify_block(ptr) }
        })
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature checked at runtime.
            unsafe { avx2::container_end_avx2(input, start) }
        } else {
            container_end_driver(input, start, |ptr| {
                // SAFETY: the driver passes pointers with 64 readable
                // bytes (whole in-bounds blocks or its zero-padded
                // staging buffer).
                unsafe { sse2::classify_block(ptr) }
            })
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        container_end_scalar(input, start)
    }
}

/// Shared body of [`container_end`]: walk 64-byte blocks from the one
/// containing `start`, tracking bracket depth over the string-filtered
/// `op` mask.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn container_end_driver(
    input: &[u8],
    start: usize,
    classify: impl Fn(*const u8) -> BlockMasks,
) -> Option<usize> {
    debug_assert!(matches!(input.get(start), Some(b'{' | b'[')));
    let mut carries = Carries::default();
    let mut depth = 0usize;
    let mut block = start & !63;
    // Bits below `start` in the first block belong to bytes the caller
    // already consumed; trimming them from every class mask keeps both
    // the depth count and the in-string parity honest.
    let mut live = u64::MAX << (start - block);
    while block < input.len() {
        let masks = if block + 64 <= input.len() {
            classify(input[block..].as_ptr())
        } else {
            let mut buf = [0u8; 64];
            buf[..input.len() - block].copy_from_slice(&input[block..]);
            // Zero padding classifies as neither quote nor op.
            classify(buf.as_ptr())
        };
        let mut ops =
            carries.block_ops(masks.backslash & live, masks.quote & live, masks.op & live);
        live = u64::MAX;
        while ops != 0 {
            let bit = ops.trailing_zeros() as usize;
            ops &= ops - 1;
            match input[block + bit] {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(block + bit + 1);
                    }
                }
                _ => {} // ':' and ',' ride along in the op class
            }
        }
        block += 64;
    }
    None
}

/// Byte-at-a-time [`container_end`] for hosts without SIMD kernels.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn container_end_scalar(input: &[u8], start: usize) -> Option<usize> {
    debug_assert!(matches!(input.get(start), Some(b'{' | b'[')));
    let mut depth = 0usize;
    let mut i = start;
    while i < input.len() {
        match input[i] {
            b'"' => {
                i += 1;
                loop {
                    match input.get(i)? {
                        b'\\' => i += 2,
                        b'"' => break,
                        _ => i += 1,
                    }
                }
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The length predicate, exercised directly; allocating multi-GiB
    /// documents to hit the real asserts is not worth a test's memory.
    #[test]
    fn index_length_limits() {
        assert!(index_len_ok(0, false));
        assert!(index_len_ok(0, true));
        assert!(index_len_ok(INDEX_MAX_LEN, false));
        assert!(!index_len_ok(INDEX_MAX_LEN + 1, false));
        assert!(index_len_ok(PACKED_INDEX_MAX_LEN - 1, true));
        assert!(!index_len_ok(PACKED_INDEX_MAX_LEN, true));
    }

    fn both(input: &[u8]) -> (Vec<u32>, Vec<u32>) {
        let mut s = Vec::new();
        let mut v = Vec::new();
        index_scalar(input, &mut s);
        index(input, &mut v);
        (s, v)
    }

    fn assert_match(input: &[u8]) {
        let (s, v) = both(input);
        assert_eq!(
            s,
            v,
            "scalar vs simd mismatch for {:?}",
            String::from_utf8_lossy(input)
        );
    }

    fn offsets(input: &[u8]) -> Vec<u32> {
        let (s, v) = both(input);
        assert_eq!(s, v);
        s
    }

    #[test]
    fn simple_object() {
        assert_eq!(offsets(br#"{"a":1}"#), vec![0, 1, 4, 5, 6]);
    }

    #[test]
    fn strings_hide_structurals() {
        // Braces and colons inside the string must not be indexed.
        assert_eq!(offsets(br#"["{a:1},[]"]"#), vec![0, 1, 11]);
    }

    #[test]
    fn escaped_quote_stays_in_string() {
        // "a\"b" is one string token.
        assert_eq!(offsets(br#"["a\"b"]"#), vec![0, 1, 7]);
    }

    #[test]
    fn escaped_backslash_then_quote_ends_string() {
        // "a\\" ends at the final quote; following comma is structural.
        assert_eq!(offsets(br#"["a\\",1]"#), vec![0, 1, 6, 7, 8]);
    }

    #[test]
    fn scalar_run_indexed_once() {
        assert_eq!(offsets(b"[true,null,123.45]"), vec![0, 1, 5, 6, 10, 11, 17]);
    }

    #[test]
    fn whitespace_between_tokens() {
        assert_eq!(offsets(b" { \"k\" : 42 } "), vec![1, 3, 7, 9, 12]);
    }

    #[test]
    fn cross_block_string() {
        // String spanning the 64-byte block boundary.
        let mut doc = Vec::from(&br#"{"key":""#[..]);
        doc.extend(std::iter::repeat_n(b'x', 100));
        doc.extend(br#"","z":1}"#);
        assert_match(&doc);
    }

    #[test]
    fn cross_block_escapes() {
        // Backslash run straddling a block boundary.
        for pad in 55..70 {
            let mut doc = Vec::from(&b"[\""[..]);
            doc.extend(std::iter::repeat_n(b'a', pad));
            doc.extend(br#"\\\""#);
            doc.extend(br#"tail"]"#);
            assert_match(&doc);
        }
    }

    #[test]
    fn benchmark_files_match() {
        for name in ["twitter", "canada", "citm_catalog", "tolstoy", "numbers"] {
            let path = format!(
                "{}/../../benchmark/{}.json",
                env!("CARGO_MANIFEST_DIR"),
                name
            );
            if let Ok(data) = std::fs::read(&path) {
                assert_match(&data);
            }
        }
    }

    #[test]
    fn packed_matches_unpacked() {
        let mut docs: Vec<Vec<u8>> = vec![
            br#"{"a":[1,"b\\n",true],"c":null}"#.to_vec(),
            b" [1,2,3] ".to_vec(),
        ];
        for name in ["twitter", "canada", "tolstoy"] {
            let path = format!(
                "{}/../../benchmark/{}.json",
                env!("CARGO_MANIFEST_DIR"),
                name
            );
            if let Ok(data) = std::fs::read(&path) {
                docs.push(data);
            }
        }
        for doc in &docs {
            let (mut plain, mut packed) = (Vec::new(), Vec::new());
            index(doc, &mut plain);
            index_packed(doc, &mut packed);
            assert_eq!(plain.len(), packed.len());
            for (&p, &q) in plain.iter().zip(packed.iter()) {
                assert_eq!(q >> 8, p, "offset mismatch");
                assert_eq!((q & 0xFF) as u8, doc[p as usize], "byte mismatch at {p}");
            }
        }
    }

    #[test]
    fn block_boundary_sweep() {
        // Structural characters at every position around block boundaries.
        for pos in 0..200 {
            let mut doc = vec![b' '; pos];
            doc.extend(br#"{"a":[1,"b\\n",true]}"#);
            assert_match(&doc);
        }
    }
}
