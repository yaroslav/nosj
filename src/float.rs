//! Exact f64 construction from (decimal mantissa, power of ten), via the
//! Eisel-Lemire algorithm plus the Clinger exact-arithmetic fast path.
//! Ported from `fast_float` / `fast_double_parser` (Apache-2.0/MIT, Daniel
//! Lemire et al.); verified differentially against fast-float2 in tests.

use crate::el_table::EL_POWERS;

/// Exact powers of ten representable in f64.
pub(crate) const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

const SMALLEST_POWER_OF_TEN: i64 = -342;
const LARGEST_POWER_OF_TEN: i64 = 308;
const MANTISSA_EXPLICIT_BITS: i32 = 52;
const MINIMUM_EXPONENT: i32 = -1023;
const INFINITE_POWER: i32 = 0x7FF;
const MIN_EXPONENT_ROUND_TO_EVEN: i64 = -4;
const MAX_EXPONENT_ROUND_TO_EVEN: i64 = 23;

#[inline(always)]
fn power(q: i64) -> i32 {
    (((152_170 + 65_536) * q) >> 16) as i32 + 63
}

/// Compute `w × 10^q` exactly rounded to f64, for a non-truncated 19-digit-max
/// mantissa `w != 0` and `q` within the table range. Returns `None` in the
/// (astronomically rare) ambiguous case; the caller falls back to a string
/// parser.
#[inline]
pub(crate) fn eisel_lemire(w: u64, q: i64) -> Option<f64> {
    debug_assert!(w != 0);
    debug_assert!((SMALLEST_POWER_OF_TEN..=LARGEST_POWER_OF_TEN).contains(&q));

    let lz = w.leading_zeros() as i32;
    let w_norm = w << lz;

    let idx = (q - SMALLEST_POWER_OF_TEN) as usize;
    let (hi5, lo5) = EL_POWERS[idx];

    // 55-bit precision product (mantissa bits + 3).
    let first = (w_norm as u128) * (hi5 as u128);
    let mut product_hi = (first >> 64) as u64;
    let mut product_lo = first as u64;
    if product_hi & 0x1FF == 0x1FF {
        let second = (w_norm as u128) * (lo5 as u128);
        let second_hi = (second >> 64) as u64;
        let (sum, carry) = product_lo.overflowing_add(second_hi);
        product_lo = sum;
        product_hi = product_hi.wrapping_add(carry as u64);
        if product_hi & 0x1FF == 0x1FF && product_lo.wrapping_add(1) == 0 {
            return None;
        }
    }

    let upperbit = (product_hi >> 63) as i32;
    let mut mantissa = product_hi >> (upperbit + 64 - MANTISSA_EXPLICIT_BITS - 3);
    let mut power2 = power(q) + upperbit - lz - MINIMUM_EXPONENT;

    if power2 <= 0 {
        // Subnormal (or underflow to zero).
        if -power2 + 1 >= 64 {
            return Some(0.0);
        }
        mantissa >>= -power2 + 1;
        mantissa += mantissa & 1;
        mantissa >>= 1;
        let exp_field = u64::from(mantissa >= (1 << MANTISSA_EXPLICIT_BITS));
        return Some(f64::from_bits(
            (exp_field << MANTISSA_EXPLICIT_BITS)
                | (mantissa & ((1 << MANTISSA_EXPLICIT_BITS) - 1)),
        ));
    }

    // Round-ties-to-even edge: we may sit exactly between two floats.
    if product_lo <= 1
        && (MIN_EXPONENT_ROUND_TO_EVEN..=MAX_EXPONENT_ROUND_TO_EVEN).contains(&q)
        && mantissa & 3 == 1
        && (mantissa << (upperbit + 64 - MANTISSA_EXPLICIT_BITS - 3)) == product_hi
    {
        mantissa &= !1u64;
    }

    mantissa += mantissa & 1;
    mantissa >>= 1;
    if mantissa >= (2 << MANTISSA_EXPLICIT_BITS) {
        mantissa = 1 << MANTISSA_EXPLICIT_BITS;
        power2 += 1;
    }
    mantissa &= !(1u64 << MANTISSA_EXPLICIT_BITS);

    if power2 >= INFINITE_POWER {
        return Some(f64::INFINITY);
    }
    Some(f64::from_bits(
        ((power2 as u64) << MANTISSA_EXPLICIT_BITS) | mantissa,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: render the number and parse with fast-float2.
    fn reference(w: u64, q: i64) -> f64 {
        let s = format!("{w}e{q}");
        fast_float2::parse::<f64, _>(s.as_bytes()).unwrap()
    }

    fn check(w: u64, q: i64) {
        if w == 0 || !(SMALLEST_POWER_OF_TEN..=LARGEST_POWER_OF_TEN).contains(&q) {
            return;
        }
        if let Some(got) = eisel_lemire(w, q) {
            let want = reference(w, q);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "w={w} q={q}: el={got:e} ref={want:e}"
            );
        }
    }

    #[test]
    fn known_hard_cases() {
        // (mantissa, q) pairs covering min-normal, subnormal, max, 2^53 edges.
        check(22250738585072014, -324); // ~min normal
        check(22250738585072011, -324);
        check(17976931348623157, 292); // max double
        check(5, -324); // min subnormal
        check(9007199254740993, 0); // 2^53 + 1 (round to even)
        check(9007199254740995, 0);
        check(1, 308);
        check(1, -307);
        check(1, -342);
        check(99999999999999999, -200);
        check(65613616943359375, -15); // canada-style coordinate
        check(123456789012345679, -20);
    }

    #[test]
    fn differential_sweep() {
        // Deterministic xorshift; a few hundred thousand random cases across
        // magnitudes and exponents.
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..300_000 {
            let w = rng() >> (rng() % 45); // spread digit counts
            let q = (rng() % 640) as i64 - 330;
            check(w, q);
        }
    }

    #[test]
    fn exponent_grid() {
        for q in -342..=308 {
            for w in [
                1u64,
                9,
                10,
                99,
                12345,
                10_000_000_000_000_000,
                u64::MAX >> 11,
            ] {
                check(w, q);
            }
        }
    }
}
