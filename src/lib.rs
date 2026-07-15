#![cfg_attr(docsrs, feature(doc_cfg))]
//! # nosj
//!
//! Pull, push and fused-cursor JSON parsers plus a streaming writer, for
//! hosts that build their own values (language runtimes, database engines,
//! columnar loaders): any consumer that wants JSON events without a DOM, a
//! tape, serde, or intermediate copies of the input. SIMD-accelerated
//! throughout (NEON on aarch64; SSE2 baseline and runtime-detected AVX2 on
//! x86-64; SWAR + table fallbacks elsewhere).
//!
//! ## Interfaces
//!
//! Four interfaces, one per driving pattern:
//!
//! | Direction | Host-driven | Library-driven |
//! |-----------|-------------|----------------|
//! | Parse     | **pull** ([`Reader`]) | **push** ([`Sink`] via [`parse`] / [`parse_indexed`]) |
//! | Generate  | **write** ([`Writer`]) | (none) |
//!
//! - [`parse`] (push): a fused single-pass byte cursor in the
//!   architectural lineage of yyjson, and the fastest path for building a
//!   full value tree. Flat containers are consumed in tight scalar loops
//!   with no frame-stack traffic.
//! - [`parse_indexed`] (push): SIMD stage-1 structural indexing (simdjson lineage),
//!   then one dispatch per token. The indexing phase is exposed separately
//!   ([`Buffers::preindex`] + [`parse_preindexed_utf8_unchecked`]) so hosts
//!   with a global interpreter lock (a GVL/GIL) can run it outside the lock.
//! - [`Reader`] (pull): grammar-enforcing `next_node()` navigation for
//!   selective or partial consumption.
//! - [`Writer`] (write): a streaming, infallible push writer; the host emits
//!   events, the writer owns separators, depth, layout, and escaping.
//!
//! ## Parsing
//!
//! ```
//! use nosj::{Buffers, Node, Reader};
//!
//! let mut bufs = Buffers::new();
//! let mut p = Reader::new(r#"{"a": [1, true]}"#, &mut bufs);
//!
//! assert!(matches!(p.next_node().unwrap(), Node::ObjectStart));
//! assert_eq!(p.object_first_key().unwrap(), Some("a"));
//! assert!(matches!(p.next_node().unwrap(), Node::ArrayStart));
//! assert!(p.array_first().unwrap());
//! assert!(matches!(p.next_node().unwrap(), Node::Int(1)));
//! assert!(p.array_next().unwrap());
//! assert!(matches!(p.next_node().unwrap(), Node::Bool(true)));
//! assert!(!p.array_next().unwrap());
//! assert_eq!(p.object_next_key().unwrap(), None);
//! p.finish().unwrap();
//! ```
//!
//! ## Generating
//!
//! ```
//! use nosj::{WriteOptions, Writer};
//!
//! let mut out = Vec::new();
//! let pretty = WriteOptions::pretty();
//! let mut w = Writer::new(&mut out, &pretty);
//! w.begin_object();
//! w.key("size");
//! w.int(3);
//! w.end_object();
//! assert_eq!(out, b"{\n  \"size\": 3\n}");
//! ```
//!
//! Number formatting is deliberately pinned: integers as plain decimal,
//! floats in the fpconv (Grisu2) format; see [`emit::write_f64`] for why
//! that differs from Rust's shortest-round-trip `Display`.
//!
//! ## Grammar extensions
//!
//! Strict RFC 8259 by default. [`ParseOptions`] opts into extensions many
//! deployed parsers accept: `NaN` / `Infinity` / `-Infinity` literals and
//! trailing commas. All option checks sit on cold paths; the defaults cost
//! nothing per token.
//!
//! ## UTF-8 contract
//!
//! `&str` entry points are safe. The `*_utf8_unchecked` entry points skip
//! whole-input validation for hosts whose runtime already tracks string
//! validity; string content is still validated structurally during the
//! parse. One deliberate leniency, matching widely deployed parsers: a lone
//! *low* surrogate escape decodes to its raw WTF-8 bytes and is delivered
//! via [`Sink::str_bytes`] (lossy-converted by default); a lone *high*
//! surrogate is an error.

pub mod emit;
pub mod scalars;

mod cursor;
mod driver;
mod el_table;
mod float;
mod grisu2;
mod pointer;
mod reader;
mod scan;
mod stage1;
mod writer;

// Parse entry points follow one naming grammar with three axes: engine
// (`parse` = fused cursor, the default recommendation; `parse_indexed`
// = two-pass structural indexing), input trust (`_utf8_unchecked` = the
// host's runtime vouches for UTF-8), and grammar extensions (`_with`
// takes `ParseOptions`). `parse_preindexed_utf8_unchecked` is the
// second half of the split `Buffers::preindex` phase.
pub use cursor::{
    ParseOptions, parse, parse_utf8_unchecked, parse_utf8_unchecked_with, parse_with,
};
pub use driver::{
    DriveError, Sink, parse_indexed, parse_indexed_utf8_unchecked, parse_preindexed_utf8_unchecked,
};
pub use pointer::{pointer, pointer_utf8_unchecked, pointers, pointers_utf8_unchecked};
pub use reader::{Buffers, ErrorKind, Node, ParseError, Reader};
pub use writer::{FloatFormat, WriteOptions, Writer};

/// The public types stay thread-friendly: a regression here (an
/// accidental `Rc`, raw-pointer field, or non-`Sync` cache) fails to
/// compile.
#[cfg(test)]
mod thread_assertions {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn public_types_are_send_and_sync() {
        assert_send::<crate::Buffers>();
        assert_sync::<crate::Buffers>();
        assert_send::<crate::Reader<'_, '_>>();
        assert_send::<crate::Writer<'_>>();
        assert_send::<crate::WriteOptions>();
        assert_sync::<crate::WriteOptions>();
        assert_send::<crate::ParseError>();
        assert_sync::<crate::ParseError>();
        assert_send::<crate::ParseOptions>();
        assert_sync::<crate::ParseOptions>();
        assert_send::<crate::Node<'_>>();
        assert_sync::<crate::Node<'_>>();
    }
}
