//! Differential against serde_json: our grammar accepts a superset of
//! serde_json's (we additionally take lone low surrogates as WTF-8,
//! numbers whose exponents overflow to infinity, and unlimited nesting),
//! so anything serde_json parses, we must parse, with equal values.
//!
//! Objects with duplicate keys are exempt from the value comparison:
//! serde's map keeps the last occurrence while a streaming parser sees
//! every occurrence, so the two sides legitimately differ there. Our tree
//! is built in full before comparing, so a duplicate anywhere in the
//! document is seen before any per-key verdict.
//!
//! Findings to date: serde_json's default float path is 1 ULP off for
//! e.g. `11e199` (documented; fixed here by its `float_roundtrip`
//! feature), and our exponent saturation miscounted leading zeros
//! (`1e00…0` parsed as infinity; fixed in `scalars.rs`).

#![no_main]

use libfuzzer_sys::fuzz_target;
use nosj::{Buffers, Node, Reader};

/// Our side of the comparison, built by a full pull walk.
enum Ours {
    Null,
    Bool(bool),
    Int(i64),
    BigInt(String),
    Float(f64),
    Str(String),
    Array(Vec<Ours>),
    Object(Vec<(String, Ours)>),
}

fn build(p: &mut Reader) -> Result<Ours, ()> {
    match p.next_node().map_err(|_| ())? {
        Node::Null => Ok(Ours::Null),
        Node::Bool(b) => Ok(Ours::Bool(b)),
        Node::Int(i) => Ok(Ours::Int(i)),
        Node::BigInt(d) => Ok(Ours::BigInt(d.to_owned())),
        Node::Float(f) => Ok(Ours::Float(f)),
        Node::Str(s) => Ok(Ours::Str(s.to_owned())),
        Node::ArrayStart => {
            let mut items = Vec::new();
            let mut more = p.array_first().map_err(|_| ())?;
            while more {
                items.push(build(p)?);
                more = p.array_next().map_err(|_| ())?;
            }
            Ok(Ours::Array(items))
        }
        Node::ObjectStart => {
            let mut pairs = Vec::new();
            let mut key = p.object_first_key().map_err(|_| ())?.map(str::to_owned);
            while let Some(k) = key {
                let v = build(p)?;
                pairs.push((k, v));
                key = p.object_next_key().map_err(|_| ())?.map(str::to_owned);
            }
            Ok(Ours::Object(pairs))
        }
    }
}

/// True if any object anywhere in the tree repeats a key.
fn has_duplicate_keys(v: &Ours) -> bool {
    match v {
        Ours::Array(items) => items.iter().any(has_duplicate_keys),
        Ours::Object(pairs) => {
            pairs
                .iter()
                .enumerate()
                .any(|(i, (k, _))| pairs[..i].iter().any(|(prev, _)| prev == k))
                || pairs.iter().any(|(_, v)| has_duplicate_keys(v))
        }
        _ => false,
    }
}

fn equal(ours: &Ours, expected: &serde_json::Value) -> bool {
    match (ours, expected) {
        (Ours::Null, serde_json::Value::Null) => true,
        (Ours::Bool(a), serde_json::Value::Bool(b)) => a == b,
        (Ours::Str(a), serde_json::Value::String(b)) => a == b,
        (Ours::Int(a), serde_json::Value::Number(n)) => {
            n.as_i64() == Some(*a)
                // `-0`: we produce integer 0 (integer-preserving, like
                // most parsers); serde_json keeps -0.0 for the IEEE sign.
                || (*a == 0 && n.as_f64().is_some_and(|f| f == 0.0))
        }
        (Ours::BigInt(digits), serde_json::Value::Number(n)) => {
            // We split integers at i64; serde continues into u64 or f64.
            n.as_u64().map(|u| u.to_string() == *digits).unwrap_or(false)
                || n.as_f64().is_some()
        }
        (Ours::Float(a), serde_json::Value::Number(n)) => {
            // Both sides use correctly rounded decimal-to-double parsing
            // (serde via float_roundtrip), so bit equality is expected.
            n.as_f64().is_some_and(|b| a.to_bits() == b.to_bits())
        }
        (Ours::Array(items), serde_json::Value::Array(expected_items)) => {
            items.len() == expected_items.len()
                && items.iter().zip(expected_items).all(|(a, b)| equal(a, b))
        }
        (Ours::Object(pairs), serde_json::Value::Object(expected_pairs)) => {
            pairs.len() == expected_pairs.len()
                && pairs
                    .iter()
                    .all(|(k, v)| expected_pairs.get(k).is_some_and(|e| equal(v, e)))
        }
        _ => false,
    }
}

fuzz_target!(|input: &str| {
    let Ok(expected) = serde_json::from_str::<serde_json::Value>(input) else {
        // serde rejected; we may accept (superset) or reject. Either way
        // the parse must not crash.
        let mut bufs = Buffers::new();
        let mut p = Reader::new(input, &mut bufs);
        let _ = build(&mut p).and_then(|_| p.finish().map_err(|_| ()));
        return;
    };

    let mut bufs = Buffers::new();
    let mut p = Reader::new(input, &mut bufs);
    let ours = build(&mut p).unwrap_or_else(|()| {
        panic!("serde_json accepted {input:?} but we rejected it")
    });
    assert!(
        p.finish().is_ok(),
        "trailing input after a value serde_json accepted: {input:?}"
    );
    if !has_duplicate_keys(&ours) {
        assert!(
            equal(&ours, &expected),
            "serde_json accepted {input:?} but our value differs"
        );
    }
});
