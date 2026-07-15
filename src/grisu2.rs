//! Grisu2 double-to-string in the fpconv format: a line-faithful port of
//! `fpconv.c` (night-shift/fpconv, Boost Software License), including the
//! ".0"-suffix-for-integral-values modification from ruby/json's vendored
//! copy.
//!
//! Byte-for-byte compatibility with that reference C implementation is a
//! crate guarantee, pinned by tests. Grisu2 without a fallback is *not*
//! guaranteed shortest (e.g. it emits `2.3438720703100002` where the
//! shortest round-trip is `2.34387207031`), so a shortest formatter (ryu)
//! cannot reproduce these bytes. Every arithmetic step below mirrors the C
//! source, including its unsigned wrapping in `round_digit` comparisons.

#[derive(Clone, Copy)]
struct Fp {
    frac: u64,
    exp: i32,
}

const NPOWERS: i32 = 87;
const STEPPOWERS: i32 = 8;
const FIRSTPOWER: i32 = -348; // 10^-348
const EXPMAX: i32 = -32;
const EXPMIN: i32 = -60;

const FRACMASK: u64 = 0x000F_FFFF_FFFF_FFFF;
const EXPMASK: u64 = 0x7FF0_0000_0000_0000;
const HIDDENBIT: u64 = 0x0010_0000_0000_0000;
const SIGNMASK: u64 = 0x8000_0000_0000_0000;
const EXPBIAS: i32 = 1023 + 52;

#[rustfmt::skip]
const POWERS_TEN: [Fp; 87] = [
    Fp { frac: 18054884314459144840, exp: -1220 }, Fp { frac: 13451937075301367670, exp: -1193 },
    Fp { frac: 10022474136428063862, exp: -1166 }, Fp { frac: 14934650266808366570, exp: -1140 },
    Fp { frac: 11127181549972568877, exp: -1113 }, Fp { frac: 16580792590934885855, exp: -1087 },
    Fp { frac: 12353653155963782858, exp: -1060 }, Fp { frac: 18408377700990114895, exp: -1034 },
    Fp { frac: 13715310171984221708, exp: -1007 }, Fp { frac: 10218702384817765436, exp: -980 },
    Fp { frac: 15227053142812498563, exp: -954 },  Fp { frac: 11345038669416679861, exp: -927 },
    Fp { frac: 16905424996341287883, exp: -901 },  Fp { frac: 12595523146049147757, exp: -874 },
    Fp { frac: 9384396036005875287,  exp: -847 },  Fp { frac: 13983839803942852151, exp: -821 },
    Fp { frac: 10418772551374772303, exp: -794 },  Fp { frac: 15525180923007089351, exp: -768 },
    Fp { frac: 11567161174868858868, exp: -741 },  Fp { frac: 17236413322193710309, exp: -715 },
    Fp { frac: 12842128665889583758, exp: -688 },  Fp { frac: 9568131466127621947,  exp: -661 },
    Fp { frac: 14257626930069360058, exp: -635 },  Fp { frac: 10622759856335341974, exp: -608 },
    Fp { frac: 15829145694278690180, exp: -582 },  Fp { frac: 11793632577567316726, exp: -555 },
    Fp { frac: 17573882009934360870, exp: -529 },  Fp { frac: 13093562431584567480, exp: -502 },
    Fp { frac: 9755464219737475723,  exp: -475 },  Fp { frac: 14536774485912137811, exp: -449 },
    Fp { frac: 10830740992659433045, exp: -422 },  Fp { frac: 16139061738043178685, exp: -396 },
    Fp { frac: 12024538023802026127, exp: -369 },  Fp { frac: 17917957937422433684, exp: -343 },
    Fp { frac: 13349918974505688015, exp: -316 },  Fp { frac: 9946464728195732843,  exp: -289 },
    Fp { frac: 14821387422376473014, exp: -263 },  Fp { frac: 11042794154864902060, exp: -236 },
    Fp { frac: 16455045573212060422, exp: -210 },  Fp { frac: 12259964326927110867, exp: -183 },
    Fp { frac: 18268770466636286478, exp: -157 },  Fp { frac: 13611294676837538539, exp: -130 },
    Fp { frac: 10141204801825835212, exp: -103 },  Fp { frac: 15111572745182864684, exp: -77 },
    Fp { frac: 11258999068426240000, exp: -50 },   Fp { frac: 16777216000000000000, exp: -24 },
    Fp { frac: 12500000000000000000, exp: 3 },     Fp { frac: 9313225746154785156,  exp: 30 },
    Fp { frac: 13877787807814456755, exp: 56 },    Fp { frac: 10339757656912845936, exp: 83 },
    Fp { frac: 15407439555097886824, exp: 109 },   Fp { frac: 11479437019748901445, exp: 136 },
    Fp { frac: 17105694144590052135, exp: 162 },   Fp { frac: 12744735289059618216, exp: 189 },
    Fp { frac: 9495567745759798747,  exp: 216 },   Fp { frac: 14149498560666738074, exp: 242 },
    Fp { frac: 10542197943230523224, exp: 269 },   Fp { frac: 15709099088952724970, exp: 295 },
    Fp { frac: 11704190886730495818, exp: 322 },   Fp { frac: 17440603504673385349, exp: 348 },
    Fp { frac: 12994262207056124023, exp: 375 },   Fp { frac: 9681479787123295682,  exp: 402 },
    Fp { frac: 14426529090290212157, exp: 428 },   Fp { frac: 10748601772107342003, exp: 455 },
    Fp { frac: 16016664761464807395, exp: 481 },   Fp { frac: 11933345169920330789, exp: 508 },
    Fp { frac: 17782069995880619868, exp: 534 },   Fp { frac: 13248674568444952270, exp: 561 },
    Fp { frac: 9871031767461413346,  exp: 588 },   Fp { frac: 14708983551653345445, exp: 614 },
    Fp { frac: 10959046745042015199, exp: 641 },   Fp { frac: 16330252207878254650, exp: 667 },
    Fp { frac: 12166986024289022870, exp: 694 },   Fp { frac: 18130221999122236476, exp: 720 },
    Fp { frac: 13508068024458167312, exp: 747 },   Fp { frac: 10064294952495520794, exp: 774 },
    Fp { frac: 14996968138956309548, exp: 800 },   Fp { frac: 11173611982879273257, exp: 827 },
    Fp { frac: 16649979327439178909, exp: 853 },   Fp { frac: 12405201291620119593, exp: 880 },
    Fp { frac: 9242595204427927429,  exp: 907 },   Fp { frac: 13772540099066387757, exp: 933 },
    Fp { frac: 10261342003245940623, exp: 960 },   Fp { frac: 15290591125556738113, exp: 986 },
    Fp { frac: 11392378155556871081, exp: 1013 },  Fp { frac: 16975966327722178521, exp: 1039 },
    Fp { frac: 12648080533535911531, exp: 1066 },
];

const TENS: [u64; 20] = [
    10_000_000_000_000_000_000,
    1_000_000_000_000_000_000,
    100_000_000_000_000_000,
    10_000_000_000_000_000,
    1_000_000_000_000_000,
    100_000_000_000_000,
    10_000_000_000_000,
    1_000_000_000_000,
    100_000_000_000,
    10_000_000_000,
    1_000_000_000,
    100_000_000,
    10_000_000,
    1_000_000,
    100_000,
    10_000,
    1_000,
    100,
    10,
    1,
];

fn find_cachedpow10(exp: i32, k: &mut i32) -> Fp {
    const ONE_LOG_TEN: f64 = 0.301_029_995_663_981_14;

    let approx = (-f64::from(exp + NPOWERS) * ONE_LOG_TEN) as i32;
    let mut idx = (approx - FIRSTPOWER) / STEPPOWERS;

    loop {
        let current = exp + POWERS_TEN[idx as usize].exp + 64;
        if current < EXPMIN {
            idx += 1;
            continue;
        }
        if current > EXPMAX {
            idx -= 1;
            continue;
        }
        *k = FIRSTPOWER + idx * STEPPOWERS;
        return POWERS_TEN[idx as usize];
    }
}

fn build_fp(d: f64) -> Fp {
    let bits = d.to_bits();
    let mut fp = Fp {
        frac: bits & FRACMASK,
        exp: ((bits & EXPMASK) >> 52) as i32,
    };
    if fp.exp != 0 {
        fp.frac += HIDDENBIT;
        fp.exp -= EXPBIAS;
    } else {
        fp.exp = -EXPBIAS + 1;
    }
    fp
}

fn normalize(fp: &mut Fp) {
    while (fp.frac & HIDDENBIT) == 0 {
        fp.frac <<= 1;
        fp.exp -= 1;
    }
    let shift = 64 - 52 - 1;
    fp.frac <<= shift;
    fp.exp -= shift;
}

fn get_normalized_boundaries(fp: &Fp, lower: &mut Fp, upper: &mut Fp) {
    upper.frac = (fp.frac << 1) + 1;
    upper.exp = fp.exp - 1;
    while (upper.frac & (HIDDENBIT << 1)) == 0 {
        upper.frac <<= 1;
        upper.exp -= 1;
    }
    let u_shift = 64 - 52 - 2;
    upper.frac <<= u_shift;
    upper.exp -= u_shift;

    let l_shift: i32 = if fp.frac == HIDDENBIT { 2 } else { 1 };
    lower.frac = (fp.frac << l_shift) - 1;
    lower.exp = fp.exp - l_shift;

    lower.frac <<= lower.exp - upper.exp;
    lower.exp = upper.exp;
}

fn multiply(a: &Fp, b: &Fp) -> Fp {
    const LOMASK: u64 = 0x0000_0000_FFFF_FFFF;

    let ah_bl = (a.frac >> 32) * (b.frac & LOMASK);
    let al_bh = (a.frac & LOMASK) * (b.frac >> 32);
    let al_bl = (a.frac & LOMASK) * (b.frac & LOMASK);
    let ah_bh = (a.frac >> 32) * (b.frac >> 32);

    let mut tmp = (ah_bl & LOMASK) + (al_bh & LOMASK) + (al_bl >> 32);
    // round up
    tmp += 1 << 31;

    Fp {
        frac: ah_bh + (ah_bl >> 32) + (al_bh >> 32) + (tmp >> 32),
        exp: a.exp + b.exp + 64,
    }
}

fn round_digit(digits: &mut [u8], ndigits: usize, delta: u64, mut rem: u64, kappa: u64, frac: u64) {
    // The C source relies on unsigned wrapping in these comparisons when
    // `kappa` has overflowed u64 (large shifts); mirror it exactly.
    while rem < frac
        && delta.wrapping_sub(rem) >= kappa
        && (rem.wrapping_add(kappa) < frac
            || frac - rem > rem.wrapping_add(kappa).wrapping_sub(frac))
    {
        digits[ndigits - 1] -= 1;
        rem = rem.wrapping_add(kappa);
    }
}

fn generate_digits(fp: &Fp, upper: &Fp, lower: &Fp, digits: &mut [u8; 18], k: &mut i32) -> usize {
    let wfrac = upper.frac - fp.frac;
    let mut delta = upper.frac - lower.frac;

    let one = Fp {
        frac: 1u64 << -upper.exp,
        exp: upper.exp,
    };

    let mut part1 = upper.frac >> -one.exp;
    let mut part2 = upper.frac & (one.frac - 1);

    let mut idx = 0usize;
    let mut kappa: i32 = 10;

    // 1000000000
    let mut div_i = 10usize;
    while kappa > 0 {
        let div = TENS[div_i];
        let digit = part1 / div;
        if digit != 0 || idx != 0 {
            digits[idx] = b'0' + digit as u8;
            idx += 1;
        }
        part1 -= digit * div;
        kappa -= 1;

        let tmp = (part1 << -one.exp) + part2;
        if tmp <= delta {
            *k += kappa;
            round_digit(
                digits,
                idx,
                delta,
                tmp,
                div.wrapping_shl(-one.exp as u32),
                wfrac,
            );
            return idx;
        }
        div_i += 1;
    }

    // 10
    let mut unit_i = 18usize;
    loop {
        part2 *= 10;
        delta *= 10;
        kappa -= 1;

        let digit = part2 >> -one.exp;
        if digit != 0 || idx != 0 {
            digits[idx] = b'0' + digit as u8;
            idx += 1;
        }

        part2 &= one.frac - 1;
        if part2 < delta {
            *k += kappa;
            round_digit(
                digits,
                idx,
                delta,
                part2,
                one.frac,
                wfrac.wrapping_mul(TENS[unit_i]),
            );
            return idx;
        }
        unit_i -= 1;
    }
}

fn grisu2(d: f64, digits: &mut [u8; 18], k: &mut i32) -> usize {
    let mut w = build_fp(d);

    let mut lower = Fp { frac: 0, exp: 0 };
    let mut upper = Fp { frac: 0, exp: 0 };
    get_normalized_boundaries(&w, &mut lower, &mut upper);

    normalize(&mut w);

    let mut cached_k = 0i32;
    let cp = find_cachedpow10(upper.exp, &mut cached_k);

    w = multiply(&w, &cp);
    let mut upper = multiply(&upper, &cp);
    let mut lower = multiply(&lower, &cp);

    lower.frac += 1;
    upper.frac -= 1;

    *k = -cached_k;

    generate_digits(&w, &upper, &lower, digits, k)
}

fn emit_digits(digits: &[u8], k: i32, neg: bool, dest: &mut Vec<u8>) {
    let ndigits = digits.len() as i32;
    let mut exp = (k + ndigits - 1).abs();

    // plain integer, with ".0" appended (the fpconv modification)
    if k >= 0 && exp < 15 {
        dest.extend_from_slice(digits);
        for _ in 0..k {
            dest.push(b'0');
        }
        dest.extend_from_slice(b".0");
        return;
    }

    // decimal without scientific notation
    if k < 0 && (k > -7 || exp < 10) {
        let offset = ndigits - k.abs();
        if offset <= 0 {
            dest.extend_from_slice(b"0.");
            for _ in 0..-offset {
                dest.push(b'0');
            }
            dest.extend_from_slice(digits);
        } else {
            dest.extend_from_slice(&digits[..offset as usize]);
            dest.push(b'.');
            dest.extend_from_slice(&digits[offset as usize..]);
        }
        return;
    }

    // scientific notation; the truncation and its use in the sign mirror C
    let ndigits = ndigits.min(18 - i32::from(neg)) as usize;
    dest.push(digits[0]);
    if ndigits > 1 {
        dest.push(b'.');
        dest.extend_from_slice(&digits[1..ndigits]);
    }
    dest.push(b'e');
    dest.push(if k + ndigits as i32 - 1 < 0 {
        b'-'
    } else {
        b'+'
    });

    let mut cent = 0;
    if exp > 99 {
        cent = exp / 100;
        dest.push(b'0' + cent as u8);
        exp -= cent * 100;
    }
    if exp > 9 {
        let dec = exp / 10;
        dest.push(b'0' + dec as u8);
        exp -= dec * 10;
    } else if cent != 0 {
        dest.push(b'0');
    }
    dest.push(b'0' + (exp % 10) as u8);
}

/// Append `d` in the fpconv format. `d` must be finite; non-finite values
/// are the caller's policy decision.
pub(crate) fn dtoa(d: f64, dest: &mut Vec<u8>) {
    debug_assert!(d.is_finite());
    if d.to_bits() & SIGNMASK != 0 {
        dest.push(b'-');
    }
    if d == 0.0 {
        dest.extend_from_slice(b"0.0");
        return;
    }
    let mut digits = [0u8; 18];
    let mut k = 0i32;
    let ndigits = grisu2(d, &mut digits, &mut k);
    emit_digits(&digits[..ndigits], k, d < 0.0, dest);
}
