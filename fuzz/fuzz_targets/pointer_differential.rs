//! Differential for JSON Pointer resolution against
//! `serde_json::Value::pointer`: for any document serde accepts and any
//! pointer, both sides must agree on hit-vs-miss, and a hit's raw slice
//! must reparse to serde's resolved value.
//!
//! Exemptions, both documented semantics: pointer-syntax errors (we
//! return an error where serde returns `None`), and duplicate keys
//! (streaming first-match vs serde's last-wins map; documents with a
//! duplicate key anywhere are exempt from the verdict).

#![no_main]

use libfuzzer_sys::fuzz_target;
use nosj::{Buffers, ErrorKind, Node, Reader};

/// Full pull walk returning whether any object repeats a key. Errors
/// count as "has duplicates" so a walk failure can never turn a
/// legitimate exemption into a false positive.
fn has_duplicate_keys(p: &mut Reader) -> Result<bool, ()> {
    match p.next_node().map_err(|_| ())? {
        Node::ObjectStart => {
            let mut seen: Vec<String> = Vec::new();
            let mut dup = false;
            let mut key = p.object_first_key().map_err(|_| ())?.map(str::to_owned);
            while let Some(k) = key {
                dup |= seen.contains(&k);
                seen.push(k);
                dup |= has_duplicate_keys(p)?;
                key = p.object_next_key().map_err(|_| ())?.map(str::to_owned);
            }
            Ok(dup)
        }
        Node::ArrayStart => {
            let mut dup = false;
            let mut more = p.array_first().map_err(|_| ())?;
            while more {
                dup |= has_duplicate_keys(p)?;
                more = p.array_next().map_err(|_| ())?;
            }
            Ok(dup)
        }
        _ => Ok(false),
    }
}

fn doc_is_dup_exempt(doc: &str, bufs: &mut Buffers) -> bool {
    let mut p = Reader::new(doc, bufs);
    has_duplicate_keys(&mut p).unwrap_or(true)
}

/// Full pull walk touching every value: validates all scalars, structure,
/// and trailing bytes. The strictest acceptance nosj offers.
fn walk_all(p: &mut Reader) -> Result<(), ()> {
    match p.next_node().map_err(|_| ())? {
        Node::ObjectStart => {
            let mut more = p.object_first_key().map_err(|_| ())?.is_some();
            while more {
                walk_all(p)?;
                more = p.object_next_key().map_err(|_| ())?.is_some();
            }
        }
        Node::ArrayStart => {
            let mut more = p.array_first().map_err(|_| ())?;
            while more {
                walk_all(p)?;
                more = p.array_next().map_err(|_| ())?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn doc_fully_valid(doc: &str, bufs: &mut Buffers) -> bool {
    let mut p = Reader::new(doc, bufs);
    walk_all(&mut p).is_ok() && p.finish().is_ok()
}

fuzz_target!(|input: (&str, &str)| {
    let (doc, ptr) = input;
    let mut bufs = Buffers::new();
    let ours = nosj::pointer(doc, ptr, &mut bufs);

    // Internal invariant: the batch resolver (a separate walker) must
    // match single resolution on any document both accept, with duplicated
    // pointers getting identical aligned answers. Documented asymmetry:
    // batch scans every byte *some* pointer needs, so on malformed
    // documents it may error where a lone query (which aborts at its
    // first missing key) returns `Ok(None)`. An error on a fully valid
    // document is therefore still a bug.
    let batch = nosj::pointers(doc, &[ptr, ptr], &mut bufs);
    match &batch {
        Ok(many) => {
            let single = ours
                .as_ref()
                .expect("batch resolved a document single resolution rejects");
            assert_eq!(
                (many[0], many[1]),
                (*single, *single),
                "batch/single split: doc {doc:?} ptr {ptr:?}"
            );
        }
        Err(_) => {
            assert!(
                ours.is_err() || !doc_fully_valid(doc, &mut bufs),
                "batch errored on a valid document: doc {doc:?} ptr {ptr:?} {batch:?}"
            );
        }
    }

    let Ok(serde_doc) = serde_json::from_str::<serde_json::Value>(doc) else {
        // serde rejected the document; resolution above must simply not
        // crash (we accept a superset: huge exponents, deep nesting).
        return;
    };
    let expected = serde_doc.pointer(ptr);

    match ours {
        Err(e) if matches!(e.kind, ErrorKind::InvalidPointer) => {
            // Documented split: bad pointer syntax is an error for us,
            // None for serde.
            assert!(
                expected.is_none(),
                "serde resolved a pointer we reject as malformed: {ptr:?}"
            );
        }
        Err(e) => panic!("serde accepted the document but resolution failed: {e} (ptr {ptr:?})"),
        Ok(None) => {
            if expected.is_some() && !doc_is_dup_exempt(doc, &mut bufs) {
                panic!("we missed what serde found: doc {doc:?} ptr {ptr:?}");
            }
        }
        Ok(Some(slice)) => {
            let reparsed: serde_json::Value =
                serde_json::from_str(slice).expect("resolved slice must reparse");
            let agrees = expected.is_some_and(|v| *v == reparsed);
            if !agrees && !doc_is_dup_exempt(doc, &mut bufs) {
                panic!(
                    "resolution mismatch: doc {doc:?} ptr {ptr:?} ours {slice:?} serde {expected:?}"
                );
            }
        }
    }
});
