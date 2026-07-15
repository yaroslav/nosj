//! The public pull API: a grammar-enforcing cursor over the stage-1 index.

use crate::scalars::{self, Literal, Number, ScalarError, StrPart};
use crate::stage1;

/// Reusable allocations. Keep one per thread and hand it to each [`Reader`]
/// or [`crate::parse`] call to amortize allocations across parses.
#[derive(Default)]
pub struct Buffers {
    indexes: Vec<u32>,
    pub(crate) scratch: Vec<u8>,
    frames: Vec<crate::driver::Frame>,
    /// Length of the input the index was last built for: a best-effort
    /// staleness check for the preindexed drive path (the `# Safety`
    /// contract remains authoritative).
    indexed_len: usize,
}

impl std::fmt::Debug for Buffers {
    /// Capacities only; the contents are scratch state, not data.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffers")
            .field("index_capacity", &self.indexes.capacity())
            .field("scratch_capacity", &self.scratch.capacity())
            .field("frame_capacity", &self.frames.capacity())
            .finish_non_exhaustive()
    }
}

impl Buffers {
    /// Empty buffers; allocations grow on first use and are reused after.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Run stage 1 (structural indexing) only: pure computation, no
    /// callbacks, no host-runtime interaction. Callers embedded in runtimes
    /// with a global lock (a GVL/GIL) can run this phase outside the lock,
    /// then finish with [`crate::parse_preindexed_utf8_unchecked`].
    pub fn preindex(&mut self, input: &[u8]) {
        crate::stage1::index(input, &mut self.indexes);
        self.indexed_len = input.len();
    }

    /// Best-effort guard for [`crate::parse_preindexed_utf8_unchecked`]:
    /// the last-indexed input length must match.
    pub(crate) fn assert_preindexed(&self, input: &[u8]) {
        debug_assert!(
            self.indexed_len == input.len(),
            "drive_preindexed input does not match the preindexed input \
             ({} vs {} bytes)",
            input.len(),
            self.indexed_len,
        );
    }

    pub(crate) fn split_for_driver(
        &mut self,
    ) -> (&mut Vec<u32>, &mut Vec<u8>, &mut Vec<crate::driver::Frame>) {
        (&mut self.indexes, &mut self.scratch, &mut self.frames)
    }
}

/// A single value-position token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Node<'a> {
    /// `{`; iterate with [`Reader::object_first_key`] / [`Reader::object_next_key`].
    ObjectStart,
    /// `[`; iterate with [`Reader::array_first`] / [`Reader::array_next`].
    ArrayStart,
    /// String value, decoded. Borrows the parser; consume before advancing.
    Str(&'a str),
    /// Integer representable in `i64`.
    Int(i64),
    /// Floating-point number, exactly rounded.
    Float(f64),
    /// Integer too large for i64, as its ASCII digits.
    BigInt(&'a str),
    /// `true` or `false`.
    Bool(bool),
    /// `null`.
    Null,
}

/// What went wrong, without position information (see [`ParseError`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A byte that cannot start any JSON token.
    UnexpectedCharacter,
    /// Input ended inside a value or container.
    UnexpectedEnd,
    /// Bytes remained after the top-level value.
    TrailingCharacters,
    /// The grammar required the described token here.
    Expected(&'static str),
    /// A string, number, or literal failed to parse.
    Scalar(ScalarError),
    /// A non-empty JSON Pointer that does not start with `/` (see
    /// [`crate::pointer`]).
    InvalidPointer,
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedCharacter => f.write_str("unexpected character"),
            Self::UnexpectedEnd => f.write_str("unexpected end of input"),
            Self::TrailingCharacters => f.write_str("trailing characters after the document"),
            Self::Expected(what) => write!(f, "expected {what}"),
            Self::Scalar(e) => write!(f, "{e}"),
            Self::InvalidPointer => f.write_str("JSON Pointer must be empty or start with '/'"),
        }
    }
}

/// A parse failure with the byte offset where it was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset into the input.
    pub offset: usize,
    /// What went wrong.
    pub kind: ErrorKind,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.kind, self.offset)
    }
}

impl std::error::Error for ParseError {}

/// Crate-internal constructors: every parse path builds errors through
/// these (all `#[cold]`; error construction is never the hot path).
impl ParseError {
    #[cold]
    pub(crate) fn expected(what: &'static str, offset: usize) -> ParseError {
        ParseError {
            offset,
            kind: ErrorKind::Expected(what),
        }
    }

    #[cold]
    pub(crate) fn unexpected_end(offset: usize) -> ParseError {
        ParseError {
            offset,
            kind: ErrorKind::UnexpectedEnd,
        }
    }

    #[cold]
    pub(crate) fn unexpected_character(offset: usize) -> ParseError {
        ParseError {
            offset,
            kind: ErrorKind::UnexpectedCharacter,
        }
    }

    #[cold]
    pub(crate) fn trailing_characters(offset: usize) -> ParseError {
        ParseError {
            offset,
            kind: ErrorKind::TrailingCharacters,
        }
    }

    #[cold]
    pub(crate) fn scalar(kind: ScalarError, offset: usize) -> ParseError {
        ParseError {
            offset,
            kind: ErrorKind::Scalar(kind),
        }
    }
}

/// Pull cursor over one JSON document.
pub struct Reader<'j, 'b> {
    input: &'j [u8],
    bufs: &'b mut Buffers,
    pos: usize,
}

impl std::fmt::Debug for Reader<'_, '_> {
    /// Position only; dumping the document would drown the output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("token_pos", &self.pos)
            .field("input_len", &self.input.len())
            .finish_non_exhaustive()
    }
}

impl<'j, 'b> Reader<'j, 'b> {
    /// Parse a `&str`; UTF-8 validity is guaranteed by the type.
    #[must_use]
    pub fn new(input: &'j str, bufs: &'b mut Buffers) -> Self {
        // SAFETY: &str is valid UTF-8.
        unsafe { Self::from_utf8_unchecked(input.as_bytes(), bufs) }
    }

    /// Parse raw bytes the caller guarantees are valid UTF-8 (e.g. vouched
    /// for by a host runtime's cached validity state).
    ///
    /// # Safety
    ///
    /// `input` must be valid UTF-8; string tokens are handed out as `&str`
    /// without re-validation.
    pub unsafe fn from_utf8_unchecked(input: &'j [u8], bufs: &'b mut Buffers) -> Self {
        stage1::index(input, &mut bufs.indexes);
        Self {
            input,
            bufs,
            pos: 0,
        }
    }

    #[inline(always)]
    fn peek_tok(&self) -> Option<(usize, u8)> {
        self.bufs.indexes.get(self.pos).map(|&off| {
            let off = off as usize;
            (off, self.input[off])
        })
    }

    #[inline(always)]
    fn take_tok(&mut self) -> Result<(usize, u8), ParseError> {
        match self.peek_tok() {
            Some(t) => {
                self.pos += 1;
                Ok(t)
            }
            None => Err(ParseError::unexpected_end(self.input.len())),
        }
    }

    /// Parse the next value-position token.
    ///
    /// Named `next_node` (not `next`) deliberately: the parser is not an
    /// [`Iterator`], because returned [`Node`]s borrow the parser and must
    /// be consumed before advancing.
    #[inline]
    pub fn next_node(&mut self) -> Result<Node<'_>, ParseError> {
        let (off, byte) = self.take_tok()?;
        match byte {
            b'{' => Ok(Node::ObjectStart),
            b'[' => Ok(Node::ArrayStart),
            b'"' => {
                // SAFETY: `self.input` came from `&str` (or the caller's
                // UTF-8 vouch), and `off` holds the `"` just tokenized.
                let part =
                    unsafe { scalars::parse_string(self.input, off, &mut self.bufs.scratch) }
                        .map_err(|e| ParseError::scalar(e, off))?;
                Ok(Node::Str(match part.0 {
                    StrPart::Borrowed(s) | StrPart::Decoded(s) => s,
                    // The pull API's Node::Str is &str-typed; lone-surrogate
                    // content (WTF-8) is rejected here. Byte-preserving
                    // consumers should use the Sink drivers.
                    StrPart::DecodedRaw(_) => {
                        return Err(ParseError::scalar(ScalarError::LoneSurrogate, off));
                    }
                }))
            }
            b'-' | b'0'..=b'9' => {
                let (num, _) = scalars::parse_number(self.input, off)
                    .map_err(|e| ParseError::scalar(e, off))?;
                Ok(match num {
                    Number::Int(i) => Node::Int(i),
                    // The pull API folds `-0` to integer 0; hosts that
                    // keep the IEEE sign use the push API and override
                    // `Sink::negative_zero`.
                    Number::NegativeZero => Node::Int(0),
                    Number::Float(f) => Node::Float(f),
                    Number::Big(digits) => Node::BigInt(digits),
                })
            }
            b't' | b'f' | b'n' => {
                let (lit, _) = scalars::parse_literal(self.input, off)
                    .map_err(|e| ParseError::scalar(e, off))?;
                Ok(match lit {
                    Literal::True => Node::Bool(true),
                    Literal::False => Node::Bool(false),
                    Literal::Null => Node::Null,
                })
            }
            _ => Err(ParseError::unexpected_character(off)),
        }
    }

    /// Decode the key string at `off` and consume the following `:`.
    #[inline(always)]
    fn read_key(&mut self, off: usize) -> Result<Option<&str>, ParseError> {
        // The colon is the next index token regardless of the key's
        // bytes; consuming it first lets the decoded key borrow the
        // scratch buffer for the whole return.
        match self.take_tok()? {
            (_, b':') => {}
            (o, _) => return Err(ParseError::expected("':'", o)),
        }
        // SAFETY: `self.input` came from `&str` (or the caller's UTF-8
        // vouch), and `off` holds the `"` just tokenized.
        let part = unsafe { scalars::parse_string(self.input, off, &mut self.bufs.scratch) }
            .map_err(|e| ParseError::scalar(e, off))?;
        Ok(Some(match part.0 {
            StrPart::Borrowed(s) | StrPart::Decoded(s) => s,
            // See next_node: the pull API rejects WTF-8 content.
            StrPart::DecodedRaw(_) => {
                return Err(ParseError::scalar(ScalarError::LoneSurrogate, off));
            }
        }))
    }

    /// After [`Node::ObjectStart`]: the first key, or `None` for `{}`.
    /// The `:` after the key is consumed; call [`Reader::next_node`] for the value.
    #[inline]
    pub fn object_first_key(&mut self) -> Result<Option<&str>, ParseError> {
        match self.take_tok()? {
            (_, b'}') => Ok(None),
            (off, b'"') => self.read_key(off),
            (o, _) => Err(ParseError::expected("'\"' or '}'", o)),
        }
    }

    /// After an object value: the next key, or `None` at `}`.
    #[inline]
    pub fn object_next_key(&mut self) -> Result<Option<&str>, ParseError> {
        match self.take_tok()? {
            (_, b'}') => Ok(None),
            (_, b',') => match self.take_tok()? {
                (off, b'"') => self.read_key(off),
                (o, _) => Err(ParseError::expected("'\"'", o)),
            },
            (o, _) => Err(ParseError::expected("',' or '}'", o)),
        }
    }

    /// After [`Node::ArrayStart`]: `true` if the array has a first element
    /// (call [`Reader::next_node`] to get it), `false` for `[]`.
    #[inline]
    pub fn array_first(&mut self) -> Result<bool, ParseError> {
        match self.peek_tok() {
            Some((_, b']')) => {
                self.pos += 1;
                Ok(false)
            }
            Some(_) => Ok(true),
            None => Err(ParseError::unexpected_end(self.input.len())),
        }
    }

    /// After an array element: `true` if another element follows.
    #[inline]
    pub fn array_next(&mut self) -> Result<bool, ParseError> {
        match self.take_tok()? {
            (_, b']') => Ok(false),
            (_, b',') => Ok(true),
            (o, _) => Err(ParseError::expected("',' or ']'", o)),
        }
    }

    /// Consume the value at value position without parsing it, returning
    /// its raw text. This is the enabler for partial parsing. Call it
    /// wherever [`Reader::next_node`] would be legal; hold the returned
    /// slice as a lazy value or feed it to any nosj entry point later.
    ///
    /// Scalar values are fully validated (their tokenizer runs to find
    /// the end). A skipped **container** is only structurally validated:
    /// brackets must balance (kinds are not matched), and the scalar
    /// tokens inside are not parsed, since the skip is a pure walk over
    /// the stage-1 index, touching no document bytes. A later full parse
    /// may therefore reject a document whose subtree this method skipped.
    #[inline]
    pub fn skip_value(&mut self) -> Result<&'j str, ParseError> {
        let (start, byte) = self.take_tok()?;
        let end = match byte {
            b'"' => {
                // SAFETY: `self.input` came from `&str` (or the caller's
                // UTF-8 vouch), and `start` holds the `"` just tokenized.
                unsafe { scalars::parse_string(self.input, start, &mut self.bufs.scratch) }
                    .map_err(|e| ParseError::scalar(e, start))?
                    .1
            }
            b'-' | b'0'..=b'9' => {
                scalars::parse_number(self.input, start)
                    .map_err(|e| ParseError::scalar(e, start))?
                    .1
            }
            b't' | b'f' | b'n' => {
                scalars::parse_literal(self.input, start)
                    .map_err(|e| ParseError::scalar(e, start))?
                    .1
            }
            b'{' | b'[' => self.skip_container()?,
            _ => {
                return Err(ParseError::unexpected_character(start));
            }
        };
        // SAFETY: the constructor guarantees `input` is valid UTF-8, and
        // both boundaries fall on token edges (ASCII).
        Ok(unsafe { std::str::from_utf8_unchecked(&self.input[start..end]) })
    }

    /// Walk the index past the container whose opener was just consumed,
    /// returning the offset one past its closer. Pure index walk: depth
    /// counting on brackets, every other token entry stepped over.
    fn skip_container(&mut self) -> Result<usize, ParseError> {
        let mut depth = 1usize;
        loop {
            let (off, byte) = self.take_tok()?;
            match byte {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(off + 1);
                    }
                }
                b'"' | b'-' | b'0'..=b'9' | b't' | b'f' | b'n' | b',' | b':' => {}
                _ => {
                    return Err(ParseError {
                        offset: off,
                        kind: ErrorKind::UnexpectedCharacter,
                    });
                }
            }
        }
    }

    /// Assert the document is complete (no trailing tokens).
    #[inline]
    pub fn finish(&mut self) -> Result<(), ParseError> {
        match self.peek_tok() {
            None => Ok(()),
            Some((o, _)) => Err(ParseError {
                offset: o,
                kind: ErrorKind::TrailingCharacters,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recursively walk one value, returning a debug rendering.
    fn walk(p: &mut Reader<'_, '_>) -> Result<String, ParseError> {
        Ok(match p.next_node()? {
            Node::ObjectStart => {
                let mut s = String::from("{");
                let mut key = p.object_first_key()?.map(str::to_string);
                while let Some(k) = key {
                    s.push_str(&k);
                    s.push('=');
                    s.push_str(&walk(p)?);
                    s.push(';');
                    key = p.object_next_key()?.map(str::to_string);
                }
                s.push('}');
                s
            }
            Node::ArrayStart => {
                let mut s = String::from("[");
                let mut more = p.array_first()?;
                while more {
                    s.push_str(&walk(p)?);
                    s.push(';');
                    more = p.array_next()?;
                }
                s.push(']');
                s
            }
            Node::Str(v) => format!("str({v})"),
            Node::Int(v) => format!("int({v})"),
            Node::Float(v) => format!("f({v})"),
            Node::BigInt(v) => format!("big({v})"),
            Node::Bool(v) => format!("b({v})"),
            Node::Null => "null".into(),
        })
    }

    fn parse(doc: &str) -> Result<String, ParseError> {
        let mut bufs = Buffers::new();
        let mut p = Reader::new(doc, &mut bufs);
        let out = walk(&mut p)?;
        p.finish()?;
        Ok(out)
    }

    #[test]
    fn round_trips() {
        assert_eq!(
            parse(r#"{"a":1,"b":[true,null]}"#).unwrap(),
            "{a=int(1);b=[b(true);null;];}"
        );
        assert_eq!(parse("[]").unwrap(), "[]");
        assert_eq!(parse("{}").unwrap(), "{}");
        assert_eq!(parse("  42  ").unwrap(), "int(42)");
        assert_eq!(parse(r#""hi""#).unwrap(), "str(hi)");
        assert_eq!(parse("[[[1]]]").unwrap(), "[[[int(1);];];]");
    }

    #[test]
    fn errors() {
        assert!(parse("").is_err());
        assert!(parse("{").is_err());
        assert!(parse(r#"{"a"}"#).is_err());
        assert!(parse(r#"{"a":}"#).is_err());
        assert!(parse("[1,]").is_err());
        assert!(parse("[1 2]").is_err());
        assert!(parse("{} {}").is_err());
        assert!(parse("truex").is_err());
        assert!(parse("01").is_err());
    }

    /// Skip every kind of value and check the returned raw slice and the
    /// parser position afterwards.
    #[test]
    fn skip_value_kinds() {
        let doc = r#"{"a":[1,2.5,"x\n",{"k":[true,null]},-0],"b":"tail"}"#;
        // Skip the whole document at the root.
        let mut bufs = Buffers::new();
        let mut p = Reader::new(doc, &mut bufs);
        assert_eq!(p.skip_value().unwrap(), doc);
        p.finish().unwrap();

        // Skip "a"'s array value, then parse "b" normally.
        let mut p = Reader::new(doc, &mut bufs);
        assert_eq!(p.next_node().unwrap(), Node::ObjectStart);
        assert_eq!(p.object_first_key().unwrap(), Some("a"));
        assert_eq!(
            p.skip_value().unwrap(),
            r#"[1,2.5,"x\n",{"k":[true,null]},-0]"#
        );
        assert_eq!(p.object_next_key().unwrap(), Some("b"));
        assert_eq!(p.next_node().unwrap(), Node::Str("tail"));
        assert_eq!(p.object_next_key().unwrap(), None);
        p.finish().unwrap();

        // Scalar skips return the exact token text (and validate it).
        for (doc, raw) in [
            ("  1234  ", "1234"),
            (" -2.5e3 ", "-2.5e3"),
            (r#" "es\tc" "#, r#""es\tc""#),
            (" true ", "true"),
            (" null ", "null"),
        ] {
            let mut p = Reader::new(doc, &mut bufs);
            assert_eq!(p.skip_value().unwrap(), raw, "{doc:?}");
            p.finish().unwrap();
        }
    }

    /// Skipping and walking must consume exactly the same tokens: skip a
    /// value, then re-parse its returned slice and get the same rendering
    /// as walking it in place.
    #[test]
    fn skip_matches_walk() {
        let doc = r#"[[1,[2,[3]]],{"a":{"b":[{"c":"d"}]}},"s",7.5,null]"#;
        let mut bufs = Buffers::new();

        // Walk each element of the root array normally...
        let mut p = Reader::new(doc, &mut bufs);
        assert_eq!(p.next_node().unwrap(), Node::ArrayStart);
        let mut walked = Vec::new();
        let mut more = p.array_first().unwrap();
        while more {
            walked.push(walk(&mut p).unwrap());
            more = p.array_next().unwrap();
        }
        p.finish().unwrap();

        // ...then skip each element and re-parse its slice.
        let mut p = Reader::new(doc, &mut bufs);
        assert_eq!(p.next_node().unwrap(), Node::ArrayStart);
        let mut slices = Vec::new();
        let mut more = p.array_first().unwrap();
        while more {
            slices.push(p.skip_value().unwrap());
            more = p.array_next().unwrap();
        }
        p.finish().unwrap();

        assert_eq!(walked.len(), slices.len());
        for (rendered, slice) in walked.iter().zip(&slices) {
            let mut bufs = Buffers::new();
            let mut p = Reader::new(slice, &mut bufs);
            assert_eq!(&walk(&mut p).unwrap(), rendered, "slice {slice:?}");
            p.finish().unwrap();
        }
    }

    #[test]
    fn skip_errors() {
        let mut bufs = Buffers::new();
        // Unterminated container.
        let mut p = Reader::new("[1,2", &mut bufs);
        assert!(p.skip_value().is_err());
        // Invalid scalar at skip position is still validated.
        let mut p = Reader::new("01", &mut bufs);
        assert!(p.skip_value().is_err());
        let mut p = Reader::new("\"unterminated", &mut bufs);
        assert!(p.skip_value().is_err());
    }

    #[test]
    fn benchmark_files_walk_clean() {
        for name in [
            "twitter",
            "canada",
            "citm_catalog",
            "tolstoy",
            "numbers",
            "mesh",
        ] {
            let path = format!(
                "{}/../../benchmark/{}.json",
                env!("CARGO_MANIFEST_DIR"),
                name
            );
            if let Ok(data) = std::fs::read_to_string(&path) {
                let mut bufs = Buffers::new();
                let mut p = Reader::new(&data, &mut bufs);
                walk(&mut p).unwrap_or_else(|e| panic!("{name}: {e}"));
                p.finish().unwrap_or_else(|e| panic!("{name}: {e}"));

                // Skipping the whole document consumes exactly the same
                // tokens and returns the trimmed source.
                let mut p = Reader::new(&data, &mut bufs);
                let slice = p.skip_value().unwrap_or_else(|e| panic!("{name}: {e}"));
                let json_ws: &[char] = &[' ', '\t', '\n', '\r'];
                assert_eq!(slice, data.trim_matches(json_ws), "{name}: root skip slice");
                p.finish().unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    }
}
