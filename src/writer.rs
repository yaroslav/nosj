//! Streaming push writer, the generation dual of the parse-side [`Sink`]:
//! there the parser drives events into the host, here the host drives events
//! into the writer, which owns all grammar state (separators, nesting depth,
//! layout, escaping).
//!
//! The writer is infallible (it appends to a `Vec<u8>`) and non-validating:
//! it emits exactly the event sequence it is given. Feeding it an invalid
//! sequence (a key outside an object, unbalanced containers) produces
//! invalid JSON rather than an error; hosts that need validation typically
//! already walk a well-formed value tree. The one enforced precondition is
//! float finiteness ([`Writer::float`] panics on NaN/infinity, which JSON
//! cannot represent).
//!
//! ```
//! use nosj::Writer;
//!
//! let mut out = Vec::new();
//! let mut w = Writer::compact(&mut out);
//! w.begin_object();
//! w.key("id");
//! w.int(7);
//! w.key("tags");
//! w.begin_array();
//! w.str("a\nb");
//! w.boolean(true);
//! w.end_array();
//! w.end_object();
//! assert_eq!(out, br#"{"id":7,"tags":["a\nb",true]}"#);
//! ```
//!
//! [`Sink`]: crate::Sink

use crate::emit::{self, EscapeMode};

/// Layout and escaping configuration for a [`Writer`].
///
/// All layout fields are raw byte strings so hosts can pass through
/// arbitrary user-supplied separators. Empty fields (the default) produce
/// compact output; [`WriteOptions::pretty`] produces the conventional
/// two-space-indented form.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WriteOptions {
    /// Repeated once per nesting level before each array element and object
    /// key.
    pub indent: Vec<u8>,
    /// Written after the `:` of each key/value pair.
    pub space: Vec<u8>,
    /// Written before the `:` of each key/value pair.
    pub space_before: Vec<u8>,
    /// Written after `{`, after each pair's `,`, and before the closing `}`
    /// of a non-empty object.
    pub object_nl: Vec<u8>,
    /// Written after `[`, after each element's `,`, and before the closing
    /// `]` of a non-empty array.
    pub array_nl: Vec<u8>,
    /// String escaping variant (see [`EscapeMode`]).
    pub escape: EscapeMode,
    /// Float formatting (see [`FloatFormat`]).
    pub float: FloatFormat,
}

/// Float formatting for [`Writer::float`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FloatFormat {
    /// The fpconv (Grisu2) format, byte-compatible with the widely
    /// deployed C reference; the crate's pinned-format guarantee.
    #[default]
    Fpconv,
    /// Shortest round-trip via zmij (Schubfach). Roughly 2x faster on
    /// float-dominated documents; the bytes differ from fpconv (both parse
    /// back to identical values). Requires the `shortest-floats` feature.
    #[cfg(feature = "shortest-floats")]
    #[cfg_attr(docsrs, doc(cfg(feature = "shortest-floats")))]
    Shortest,
}

impl WriteOptions {
    /// Compact output: no whitespace anywhere, standard escaping.
    pub const COMPACT: WriteOptions = WriteOptions {
        indent: Vec::new(),
        space: Vec::new(),
        space_before: Vec::new(),
        object_nl: Vec::new(),
        array_nl: Vec::new(),
        escape: EscapeMode::Standard,
        float: FloatFormat::Fpconv,
    };

    /// Conventional pretty-printing: two-space indent, newline-separated
    /// members, a space after each `:`.
    #[must_use]
    pub fn pretty() -> WriteOptions {
        WriteOptions {
            indent: b"  ".to_vec(),
            space: b" ".to_vec(),
            space_before: Vec::new(),
            object_nl: b"\n".to_vec(),
            array_nl: b"\n".to_vec(),
            escape: EscapeMode::Standard,
            float: FloatFormat::Fpconv,
        }
    }

    fn is_compact_layout(&self) -> bool {
        self.indent.is_empty()
            && self.space.is_empty()
            && self.space_before.is_empty()
            && self.object_nl.is_empty()
            && self.array_nl.is_empty()
    }
}

/// Streaming JSON writer over a byte buffer.
///
/// Event methods mirror the parse-side [`Sink`](crate::Sink): scalars
/// ([`null`](Writer::null), [`boolean`](Writer::boolean),
/// [`int`](Writer::int), [`float`](Writer::float), [`str`](Writer::str)),
/// containers (`begin_*` / `end_*`), and [`key`](Writer::key) inside
/// objects. [`value_raw`](Writer::value_raw) splices pre-serialized bytes
/// (big integers, non-finite-number policies, embedded fragments) into
/// value position.
#[doc(alias = "serialize")]
#[doc(alias = "stringify")]
#[doc(alias = "generate")]
pub struct Writer<'a> {
    out: &'a mut Vec<u8>,
    cfg: &'a WriteOptions,
    /// Layout fields all empty; lets compact output skip the layout writes
    /// with one predictable branch.
    compact: bool,
    depth: usize,
    /// True while the current container has no members yet (and at the
    /// root before the first value).
    first: bool,
    /// True between a `key()` and its value; suppresses the value prefix.
    after_key: bool,
}

impl std::fmt::Debug for Writer<'_> {
    /// Grammar state and output length, not the buffer contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("depth", &self.depth)
            .field("first", &self.first)
            .field("after_key", &self.after_key)
            .field("compact", &self.compact)
            .field("written", &self.out.len())
            .finish_non_exhaustive()
    }
}

/// Borrowable default configuration ([`WriteOptions::COMPACT`] is a `const`,
/// which cannot be borrowed past the enclosing statement).
static COMPACT: WriteOptions = WriteOptions::COMPACT;

impl<'a> Writer<'a> {
    /// A writer appending to `out`. Existing content is preserved.
    #[must_use]
    pub fn new(out: &'a mut Vec<u8>, cfg: &'a WriteOptions) -> Writer<'a> {
        Writer {
            compact: cfg.is_compact_layout(),
            out,
            cfg,
            depth: 0,
            first: true,
            after_key: false,
        }
    }

    /// A compact writer appending to `out`: no whitespace, standard
    /// escaping.
    #[must_use]
    pub fn compact(out: &'a mut Vec<u8>) -> Writer<'a> {
        Writer::new(out, &COMPACT)
    }

    /// Current container nesting depth (0 at the root). Hosts enforcing a
    /// nesting limit check this at `begin_*` time.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The buffer being written to.
    #[must_use]
    pub fn buffer(&mut self) -> &mut Vec<u8> {
        self.out
    }

    #[inline(always)]
    fn push_indent(&mut self) {
        for _ in 0..self.depth {
            self.out.extend_from_slice(&self.cfg.indent);
        }
    }

    /// Comma/newline/indent owed before a value in array or root position.
    #[inline(always)]
    fn value_prefix(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
        if self.depth > 0 {
            if !self.first {
                self.out.push(b',');
            }
            if !self.compact {
                self.out.extend_from_slice(&self.cfg.array_nl);
                self.push_indent();
            }
        }
        self.first = false;
    }

    /// `null` in value position.
    #[inline]
    pub fn null(&mut self) {
        self.value_prefix();
        self.out.extend_from_slice(b"null");
    }

    /// `true` or `false` in value position.
    #[inline]
    pub fn boolean(&mut self, value: bool) {
        self.value_prefix();
        self.out
            .extend_from_slice(if value { b"true" } else { b"false" });
    }

    /// Integer in value position.
    #[inline]
    pub fn int(&mut self, value: i64) {
        self.value_prefix();
        emit::write_i64(self.out, value);
    }

    /// Finite float in value position, formatted per the configured
    /// [`FloatFormat`]. Non-finite values are the host's policy decision
    /// (error out, or splice a literal via [`Writer::value_raw`]): JSON
    /// has no representation for them, so passing one here panics rather
    /// than corrupting the output (the digit generator would emit an
    /// unrelated finite number).
    ///
    /// # Panics
    ///
    /// If `value` is NaN or infinite.
    #[inline]
    pub fn float(&mut self, value: f64) {
        assert!(
            value.is_finite(),
            "Writer::float requires a finite value; encode a non-finite \
             policy via Writer::value_raw"
        );
        self.value_prefix();
        match self.cfg.float {
            FloatFormat::Fpconv => emit::write_f64(self.out, value),
            #[cfg(feature = "shortest-floats")]
            FloatFormat::Shortest => {
                let mut buf = zmij::Buffer::new();
                self.out
                    .extend_from_slice(buf.format_finite(value).as_bytes());
            }
        }
    }

    /// String in value position, quoted and escaped per the configured
    /// [`EscapeMode`].
    #[inline]
    pub fn str(&mut self, value: &str) {
        self.value_prefix();
        self.write_quoted(value.as_bytes());
    }

    /// Like [`Writer::str`], for hosts whose runtime vouches for UTF-8.
    ///
    /// # Safety
    ///
    /// `value` must be valid UTF-8 ([`EscapeMode::AsciiOnly`] decodes it).
    #[inline]
    pub unsafe fn str_utf8_unchecked(&mut self, value: &[u8]) {
        self.value_prefix();
        self.write_quoted(value);
    }

    /// Pre-serialized bytes in value position: big-integer digits, a
    /// non-finite-number literal, an embedded JSON fragment. Spliced
    /// verbatim, with no quoting, no escaping, no validation.
    #[inline]
    pub fn value_raw(&mut self, bytes: &[u8]) {
        self.value_prefix();
        self.out.extend_from_slice(bytes);
    }

    /// Object key: separator and layout for the pair, the quoted key, and
    /// the `:`. The next event supplies the pair's value.
    #[inline]
    pub fn key(&mut self, key: &str) {
        self.key_prefix();
        self.write_quoted(key.as_bytes());
        self.key_suffix();
    }

    /// Like [`Writer::key`], for hosts whose runtime vouches for UTF-8.
    ///
    /// # Safety
    ///
    /// `key` must be valid UTF-8 ([`EscapeMode::AsciiOnly`] decodes it).
    #[inline]
    pub unsafe fn key_utf8_unchecked(&mut self, key: &[u8]) {
        self.key_prefix();
        self.write_quoted(key);
        self.key_suffix();
    }

    #[inline(always)]
    fn key_prefix(&mut self) {
        if !self.first {
            self.out.push(b',');
        }
        if !self.compact {
            self.out.extend_from_slice(&self.cfg.object_nl);
            self.push_indent();
        }
        self.first = false;
    }

    #[inline(always)]
    fn key_suffix(&mut self) {
        if !self.compact {
            self.out.extend_from_slice(&self.cfg.space_before);
        }
        self.out.push(b':');
        if !self.compact {
            self.out.extend_from_slice(&self.cfg.space);
        }
        self.after_key = true;
    }

    /// Open an array in value position.
    #[inline]
    pub fn begin_array(&mut self) {
        self.value_prefix();
        self.out.push(b'[');
        self.depth += 1;
        self.first = true;
    }

    /// Close the current array. An unmatched closer at the root keeps the
    /// non-validating contract: invalid JSON comes out, nothing worse
    /// (the depth saturates rather than underflowing into the indent
    /// loop).
    #[inline]
    pub fn end_array(&mut self) {
        self.end_container(b']');
    }

    /// Open an object in value position.
    #[inline]
    pub fn begin_object(&mut self) {
        self.value_prefix();
        self.out.push(b'{');
        self.depth += 1;
        self.first = true;
    }

    /// Close the current object. Root-level misuse saturates like
    /// [`Writer::end_array`].
    #[inline]
    pub fn end_object(&mut self) {
        self.end_container(b'}');
    }

    #[inline(always)]
    fn end_container(&mut self, closer: u8) {
        self.depth = self.depth.saturating_sub(1);
        // Layout before the closer only for non-empty containers with a
        // configured newline; per-element indent above is unconditional.
        let nl = if closer == b']' {
            &self.cfg.array_nl
        } else {
            &self.cfg.object_nl
        };
        if !self.first && !self.compact && !nl.is_empty() {
            self.out.extend_from_slice(nl);
            self.push_indent();
        }
        self.out.push(closer);
        self.first = false;
    }

    #[inline(always)]
    fn write_quoted(&mut self, bytes: &[u8]) {
        self.out.reserve(bytes.len() + 2);
        self.out.push(b'"');
        emit::escape_into(self.out, bytes, self.cfg.escape);
        self.out.push(b'"');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-finite floats must fail loudly in every build, never emit
    /// digits of some unrelated finite number.
    #[test]
    #[should_panic(expected = "finite")]
    fn float_rejects_nan() {
        let mut out = Vec::new();
        Writer::compact(&mut out).float(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn float_rejects_infinity() {
        let mut out = Vec::new();
        Writer::compact(&mut out).float(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn float_rejects_neg_infinity() {
        let mut out = Vec::new();
        Writer::compact(&mut out).float(f64::NEG_INFINITY);
    }

    /// Root-level closers are protocol misuse; the non-validating
    /// contract says invalid JSON out, not a depth underflow that turns
    /// the pretty indent loop pathological.
    #[test]
    fn root_closers_saturate() {
        let mut out = Vec::new();
        {
            let mut w = Writer::compact(&mut out);
            w.end_array();
            w.end_object();
            w.int(1);
        }
        assert_eq!(out, b"]}1");

        let mut out = Vec::new();
        {
            let pretty = WriteOptions::pretty();
            let mut w = Writer::new(&mut out, &pretty);
            w.end_object();
            w.begin_array();
            w.int(1);
            w.end_array();
        }
        // Still terminates and emits balanced-after-the-fact layout.
        assert_eq!(String::from_utf8(out).unwrap(), "}[\n  1\n]");
    }

    fn doc(cfg: &WriteOptions, f: impl FnOnce(&mut Writer)) -> String {
        let mut out = Vec::new();
        let mut w = Writer::new(&mut out, cfg);
        f(&mut w);
        assert_eq!(w.depth(), 0, "unbalanced containers in test doc");
        String::from_utf8(out).unwrap()
    }

    fn sample(w: &mut Writer) {
        w.begin_object();
        w.key("a");
        w.begin_array();
        w.int(1);
        w.begin_object();
        w.key("x");
        w.int(2);
        w.end_object();
        w.str("s");
        w.end_array();
        w.key("e");
        w.begin_object();
        w.end_object();
        w.key("ea");
        w.begin_array();
        w.end_array();
        w.key("n");
        w.null();
        w.end_object();
    }

    #[test]
    fn compact() {
        assert_eq!(
            doc(&WriteOptions::COMPACT, sample),
            r#"{"a":[1,{"x":2},"s"],"e":{},"ea":[],"n":null}"#
        );
    }

    #[test]
    fn pretty() {
        let expected = "{\n  \"a\": [\n    1,\n    {\n      \"x\": 2\n    },\n    \"s\"\n  ],\n  \"e\": {},\n  \"ea\": [],\n  \"n\": null\n}";
        assert_eq!(doc(&WriteOptions::pretty(), sample), expected);
    }

    #[test]
    fn custom_layout_partial_newlines() {
        // object_nl set but array_nl empty: array elements still get the
        // per-element indent, but the array closer gets no layout.
        let cfg = WriteOptions {
            indent: b"..".to_vec(),
            object_nl: b"|".to_vec(),
            ..WriteOptions::default()
        };
        assert_eq!(
            doc(&cfg, sample),
            "{|..\"a\":[....1,....{|......\"x\":2|....},....\"s\"],|..\"e\":{},|..\"ea\":[],|..\"n\":null|}"
        );
    }

    #[test]
    fn root_scalars_and_raw() {
        assert_eq!(doc(&WriteOptions::COMPACT, |w| w.str("top")), "\"top\"");
        // Not `Writer::null`: the lifetime parameter defeats HRTB inference.
        #[allow(clippy::redundant_closure_for_method_calls)]
        let null = |w: &mut Writer| w.null();
        assert_eq!(doc(&WriteOptions::COMPACT, null), "null");
        assert_eq!(doc(&WriteOptions::COMPACT, |w| w.float(1.5)), "1.5");
        assert_eq!(
            doc(&WriteOptions::COMPACT, |w| {
                w.begin_array();
                w.value_raw(b"1208925819614629174706176");
                w.value_raw(b"NaN");
                w.end_array();
            }),
            "[1208925819614629174706176,NaN]"
        );
    }

    #[test]
    fn escape_modes() {
        let script_safe = WriteOptions {
            escape: EscapeMode::ScriptSafe,
            ..WriteOptions::default()
        };
        assert_eq!(
            doc(&script_safe, |w| w.str("a/b\u{2028}")),
            "\"a\\/b\\u2028\""
        );
        let ascii = WriteOptions {
            escape: EscapeMode::AsciiOnly,
            ..WriteOptions::default()
        };
        assert_eq!(doc(&ascii, |w| w.str("héllo")), "\"h\\u00e9llo\"");
    }

    #[test]
    fn pretty_root_scalar_has_no_layout() {
        assert_eq!(doc(&WriteOptions::pretty(), |w| w.int(42)), "42");
    }

    #[test]
    fn appends_after_existing_content() {
        let mut out = b"data: ".to_vec();
        let cfg = WriteOptions::COMPACT;
        let mut w = Writer::new(&mut out, &cfg);
        w.begin_array();
        w.int(1);
        w.end_array();
        assert_eq!(out, b"data: [1]");
    }
}
