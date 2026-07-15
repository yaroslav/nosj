//! Push-mode driver: parse a whole document in one pass, delivering events
//! to a [`Sink`]. One dispatch per token and no intermediate token values:
//! the fastest interface for consumers that build a full value tree (FFI
//! object construction, DOM building). The pull [`crate::Reader`] remains the
//! right tool for partial/selective consumption.
//!
//! The goto-style state machine follows the shape of simdjson's stage 2
//! (also used by the simd-json Rust port, Apache-2.0/MIT).

use crate::reader::{Buffers, ParseError};
use crate::scalars::{self, Literal, Number, StrPart};
use crate::stage1;

/// Receives parse events in document order.
///
/// `mark()` is called at every container start; the matching `end_*` receives
/// that mark back along with the element/pair count, so a sink keeping a flat
/// value stack can slice off the container's children directly.
///
/// # Example
///
/// The flat-value-stack pattern most hosts use: every value pushes one
/// entry, and `end_*` splits off exactly the container's children:
///
/// ```
/// use nosj::{Buffers, Sink, parse};
///
/// #[derive(Debug, PartialEq)]
/// enum Value {
///     Null,
///     Bool(bool),
///     Int(i64),
///     Float(f64),
///     Str(String),
///     Array(Vec<Value>),
///     Object(Vec<(String, Value)>),
/// }
///
/// #[derive(Default)]
/// struct Tree {
///     stack: Vec<Value>,
///     keys: Vec<String>,
/// }
///
/// impl Sink for Tree {
///     type Error = std::convert::Infallible;
///
///     fn null(&mut self) -> Result<(), Self::Error> {
///         self.stack.push(Value::Null);
///         Ok(())
///     }
///     fn boolean(&mut self, v: bool) -> Result<(), Self::Error> {
///         self.stack.push(Value::Bool(v));
///         Ok(())
///     }
///     fn int(&mut self, v: i64) -> Result<(), Self::Error> {
///         self.stack.push(Value::Int(v));
///         Ok(())
///     }
///     fn float(&mut self, v: f64) -> Result<(), Self::Error> {
///         self.stack.push(Value::Float(v));
///         Ok(())
///     }
///     fn big_int(&mut self, digits: &str) -> Result<(), Self::Error> {
///         self.stack.push(Value::Str(digits.to_owned())); // host's choice
///         Ok(())
///     }
///     fn str(&mut self, v: &str) -> Result<(), Self::Error> {
///         self.stack.push(Value::Str(v.to_owned()));
///         Ok(())
///     }
///     fn key(&mut self, k: &str) -> Result<(), Self::Error> {
///         self.keys.push(k.to_owned());
///         Ok(())
///     }
///     fn mark(&self) -> usize {
///         self.stack.len()
///     }
///     fn end_array(&mut self, mark: usize, _len: usize) -> Result<(), Self::Error> {
///         let items = self.stack.split_off(mark);
///         self.stack.push(Value::Array(items));
///         Ok(())
///     }
///     fn end_object(&mut self, mark: usize, pairs: usize) -> Result<(), Self::Error> {
///         let values = self.stack.split_off(mark);
///         let keys = self.keys.split_off(self.keys.len() - pairs);
///         self.stack.push(Value::Object(keys.into_iter().zip(values).collect()));
///         Ok(())
///     }
/// }
///
/// let mut bufs = Buffers::new();
/// let mut sink = Tree::default();
/// parse(r#"{"a": [1, true]}"#, &mut bufs, &mut sink).unwrap();
/// assert_eq!(
///     sink.stack.pop(),
///     Some(Value::Object(vec![(
///         "a".into(),
///         Value::Array(vec![Value::Int(1), Value::Bool(true)]),
///     )]))
/// );
/// ```
pub trait Sink {
    /// Error type used by the sink to abort the parse.
    type Error;

    /// `null` in value position.
    fn null(&mut self) -> Result<(), Self::Error>;
    /// `true` or `false` in value position.
    fn boolean(&mut self, value: bool) -> Result<(), Self::Error>;
    /// Integer representable in `i64`.
    fn int(&mut self, value: i64) -> Result<(), Self::Error>;
    /// Floating-point number, exactly rounded.
    fn float(&mut self, value: f64) -> Result<(), Self::Error>;
    /// Integer too large for i64, as validated ASCII digits.
    fn big_int(&mut self, digits: &str) -> Result<(), Self::Error>;
    /// String value; only valid during this call.
    fn str(&mut self, value: &str) -> Result<(), Self::Error>;
    /// Object key; only valid during this call.
    fn key(&mut self, key: &str) -> Result<(), Self::Error>;
    /// Position token recorded at container start.
    fn mark(&self) -> usize;
    /// Close an array holding the `len` values accepted since `mark`.
    fn end_array(&mut self, mark: usize, len: usize) -> Result<(), Self::Error>;
    /// Close an object holding the `pairs` key/value pairs accepted since `mark`.
    fn end_object(&mut self, mark: usize, pairs: usize) -> Result<(), Self::Error>;

    /// Called before `mark()` when an array opens. Sinks building eagerly can
    /// allocate the container here and bound their pending state.
    #[inline(always)]
    fn begin_array(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Called before `mark()` when an object opens.
    #[inline(always)]
    fn begin_object(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Called periodically (every 256 elements) inside long arrays; sinks may
    /// spill accepted elements into the container to bound pending state.
    /// This keeps per-GC marking of pending values O(1) instead of O(array).
    #[inline(always)]
    fn array_checkpoint(&mut self, _mark: usize) -> Result<(), Self::Error> {
        Ok(())
    }

    /// The integer literal `-0`. The default folds it to integer 0
    /// (integer-preserving hosts); IEEE-sign-preserving hosts (e.g.
    /// JavaScript, where `JSON.parse("-0")` is `-0`) override this to a
    /// negative-zero float.
    #[inline(always)]
    fn negative_zero(&mut self) -> Result<(), Self::Error> {
        self.int(0)
    }

    /// String value whose decoded content is NOT valid UTF-8 (a lone low
    /// surrogate escape, which lenient parsers accept as raw WTF-8 bytes).
    /// The default lossy-converts; byte-oriented hosts should override to
    /// preserve the exact bytes.
    #[inline(always)]
    fn str_bytes(&mut self, value: &[u8]) -> Result<(), Self::Error> {
        self.str(&String::from_utf8_lossy(value))
    }
    /// Object key whose decoded content is NOT valid UTF-8 (see
    /// [`Sink::str_bytes`]).
    #[inline(always)]
    fn key_bytes(&mut self, key: &[u8]) -> Result<(), Self::Error> {
        self.key(&String::from_utf8_lossy(key))
    }
}

/// Failure while driving a sink: either the document didn't parse, or the
/// sink aborted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriveError<E> {
    /// The input is not valid JSON.
    Parse(ParseError),
    /// The sink returned an error; parsing stopped immediately.
    Sink(E),
}

impl<E: std::fmt::Display> std::fmt::Display for DriveError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "{e}"),
            Self::Sink(e) => write!(f, "sink error: {e}"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for DriveError<E> {}

impl<E> From<ParseError> for DriveError<E> {
    fn from(e: ParseError) -> Self {
        DriveError::Parse(e)
    }
}

pub(crate) enum Frame {
    Root,
    Object { mark: usize, cnt: usize },
    Array { mark: usize, cnt: usize },
}

// The scalar-emission steps shared by both push engines (indexed driver
// and fused cursor): tokenize one scalar at `off`, deliver it to the
// sink, return one past its final byte. `#[inline(always)]` because
// these are the hot loop's bodies, split out only so they exist once.

/// Emit the string value whose `"` is at `off`.
///
/// # Safety
///
/// `input` must be valid UTF-8.
#[inline(always)]
pub(crate) unsafe fn sink_str<S: Sink>(
    input: &[u8],
    off: usize,
    scratch: &mut Vec<u8>,
    sink: &mut S,
) -> Result<usize, DriveError<S::Error>> {
    // SAFETY: forwarded contract; `off` holds the `"` the caller matched.
    let (part, end) = unsafe { scalars::parse_string(input, off, scratch) }
        .map_err(|e| ParseError::scalar(e, off))?;
    match part {
        StrPart::Borrowed(s) | StrPart::Decoded(s) => sink.str(s).map_err(DriveError::Sink)?,
        StrPart::DecodedRaw(b) => sink.str_bytes(b).map_err(DriveError::Sink)?,
    }
    Ok(end)
}

/// Emit the object key whose `"` is at `off` (the caller consumes the
/// following `:`).
///
/// # Safety
///
/// `input` must be valid UTF-8.
#[inline(always)]
pub(crate) unsafe fn sink_key<S: Sink>(
    input: &[u8],
    off: usize,
    scratch: &mut Vec<u8>,
    sink: &mut S,
) -> Result<usize, DriveError<S::Error>> {
    // SAFETY: forwarded contract; `off` holds the `"` the caller matched.
    let (part, end) = unsafe { scalars::parse_string(input, off, scratch) }
        .map_err(|e| ParseError::scalar(e, off))?;
    match part {
        StrPart::Borrowed(s) | StrPart::Decoded(s) => sink.key(s).map_err(DriveError::Sink)?,
        StrPart::DecodedRaw(b) => sink.key_bytes(b).map_err(DriveError::Sink)?,
    }
    Ok(end)
}

/// Emit the number starting at `off`.
#[inline(always)]
pub(crate) fn sink_number<S: Sink>(
    input: &[u8],
    off: usize,
    sink: &mut S,
) -> Result<usize, DriveError<S::Error>> {
    let (num, end) = scalars::parse_number(input, off).map_err(|e| ParseError::scalar(e, off))?;
    match num {
        Number::Int(v) => sink.int(v),
        Number::NegativeZero => sink.negative_zero(),
        Number::Float(v) => sink.float(v),
        Number::Big(digits) => sink.big_int(digits),
    }
    .map_err(DriveError::Sink)?;
    Ok(end)
}

/// Emit the `true`/`false`/`null` literal starting at `off`.
#[inline(always)]
pub(crate) fn sink_literal<S: Sink>(
    input: &[u8],
    off: usize,
    sink: &mut S,
) -> Result<usize, DriveError<S::Error>> {
    let (lit, end) = scalars::parse_literal(input, off).map_err(|e| ParseError::scalar(e, off))?;
    match lit {
        Literal::True => sink.boolean(true),
        Literal::False => sink.boolean(false),
        Literal::Null => sink.null(),
    }
    .map_err(DriveError::Sink)?;
    Ok(end)
}

/// Parse `input`, pushing events into `sink`.
pub fn parse_indexed<S: Sink>(
    input: &str,
    bufs: &mut Buffers,
    sink: &mut S,
) -> Result<(), DriveError<S::Error>> {
    // SAFETY: &str is valid UTF-8.
    unsafe { parse_indexed_utf8_unchecked(input.as_bytes(), bufs, sink) }
}

/// Like [`parse_indexed`], for callers whose runtime vouches for UTF-8
/// validity.
///
/// # Safety
///
/// `input` must be valid UTF-8.
pub unsafe fn parse_indexed_utf8_unchecked<S: Sink>(
    input: &[u8],
    bufs: &mut Buffers,
    sink: &mut S,
) -> Result<(), DriveError<S::Error>> {
    let (indexes, scratch, stack) = bufs.split_for_driver();
    // Measured on Apple Silicon: the packed index encoding
    // (`stage1::index_packed` + `drive_inner::<_, true>`) costs more in
    // stage 1's flatten loop than it saves here: the driver's byte loads
    // are L1-resident. Kept available for hardware where that trades
    // differently.
    stage1::index(input, indexes);
    drive_inner::<S, false>(input, indexes, scratch, stack, sink)
}

/// Drive a sink over buffers already indexed by [`Buffers::preindex`],
/// skipping the indexing phase.
///
/// # Safety
///
/// `input` must be valid UTF-8 **and** byte-identical to the slice passed to
/// the matching [`Buffers::preindex`] call: the stored index offsets are read
/// back against it. A shorter or different input produces parse errors or
/// panics (never undefined behavior; all reads are bounds-checked).
pub unsafe fn parse_preindexed_utf8_unchecked<S: Sink>(
    input: &[u8],
    bufs: &mut Buffers,
    sink: &mut S,
) -> Result<(), DriveError<S::Error>> {
    bufs.assert_preindexed(input);
    let (indexes, scratch, stack) = bufs.split_for_driver();
    drive_inner::<S, false>(input, indexes, scratch, stack, sink)
}

fn drive_inner<S: Sink, const PACKED: bool>(
    input: &[u8],
    idx: &[u32],
    scratch: &mut Vec<u8>,
    stack: &mut Vec<Frame>,
    sink: &mut S,
) -> Result<(), DriveError<S::Error>> {
    stack.clear();
    let mut pos: usize = 0;

    macro_rules! take {
        () => {{
            if pos < idx.len() {
                let e = idx[pos];
                pos += 1;
                if PACKED {
                    ((e >> 8) as usize, e as u8)
                } else {
                    let off = e as usize;
                    (off, input[off])
                }
            } else {
                return Err(ParseError::unexpected_end(input.len()).into());
            }
        }};
    }

    macro_rules! fail {
        ($off:expr, $what:expr) => {
            return Err(ParseError::expected($what, $off).into())
        };
    }

    macro_rules! sink_try {
        ($e:expr) => {
            $e.map_err(DriveError::Sink)?
        };
    }

    // The scalar bodies live in the shared `sink_*` helpers; the driver
    // ignores the returned end offset (the index owns positioning).
    macro_rules! emit_scalar {
        ($off:expr, $byte:expr) => {{
            match $byte {
                b'"' => {
                    // SAFETY: `input` is valid UTF-8 per this function's
                    // contract, and `$off` holds the `"` found by stage 1.
                    unsafe { sink_str(input, $off, scratch, sink) }?;
                }
                b'-' | b'0'..=b'9' => {
                    sink_number(input, $off, sink)?;
                }
                b't' | b'f' | b'n' => {
                    sink_literal(input, $off, sink)?;
                }
                _ => return Err(ParseError::unexpected_character($off).into()),
            }
        }};
    }

    macro_rules! emit_key {
        ($off:expr) => {{
            // SAFETY: `input` is valid UTF-8 per this function's contract,
            // and `$off` holds the `"` found by stage 1.
            unsafe { sink_key(input, $off, scratch, sink) }?;
            let (coff, cbyte) = take!();
            if cbyte != b':' {
                fail!(coff, "':'");
            }
        }};
    }

    macro_rules! finish {
        () => {{
            if pos < idx.len() {
                let off = if PACKED {
                    (idx[pos] >> 8) as usize
                } else {
                    idx[pos] as usize
                };
                return Err(ParseError::trailing_characters(off).into());
            }
            return Ok(());
        }};
    }

    // Current container registers; parents on the frame stack.
    let mut mark: usize;
    let mut cnt: usize;
    let mut is_object: bool;

    // Root value.
    let (off, byte) = take!();
    match byte {
        b'{' => {
            stack.push(Frame::Root);
            sink_try!(sink.begin_object());
            mark = sink.mark();
            cnt = 0;
            is_object = true;
        }
        b'[' => {
            stack.push(Frame::Root);
            sink_try!(sink.begin_array());
            mark = sink.mark();
            cnt = 0;
            is_object = false;
        }
        _ => {
            emit_scalar!(off, byte);
            finish!();
        }
    }

    // `true` when a container was just entered (must accept `}`/`]` or first
    // element); `false` right after a value (must accept `,` or the closer).
    let mut fresh = true;

    loop {
        if fresh {
            fresh = false;
            // First key/element or immediate close.
            let (off, byte) = take!();
            let closer = if is_object { b'}' } else { b']' };
            if byte == closer {
                // Empty container: fall through to scope end.
            } else if is_object {
                if byte != b'"' {
                    fail!(off, "'\"' or '}'");
                }
                emit_key!(off);
                cnt += 1;
                let (voff, vbyte) = take!();
                match vbyte {
                    b'{' => {
                        stack.push(Frame::Object { mark, cnt });
                        sink_try!(sink.begin_object());
                        mark = sink.mark();
                        cnt = 0;
                        is_object = true;
                        fresh = true;
                    }
                    b'[' => {
                        stack.push(Frame::Object { mark, cnt });
                        sink_try!(sink.begin_array());
                        mark = sink.mark();
                        cnt = 0;
                        is_object = false;
                        fresh = true;
                    }
                    _ => emit_scalar!(voff, vbyte),
                }
                continue;
            } else {
                cnt += 1;
                match byte {
                    b'{' => {
                        stack.push(Frame::Array { mark, cnt });
                        sink_try!(sink.begin_object());
                        mark = sink.mark();
                        cnt = 0;
                        is_object = true;
                        fresh = true;
                    }
                    b'[' => {
                        stack.push(Frame::Array { mark, cnt });
                        sink_try!(sink.begin_array());
                        mark = sink.mark();
                        cnt = 0;
                        is_object = false;
                        fresh = true;
                    }
                    _ => emit_scalar!(off, byte),
                }
                continue;
            }
        } else {
            // After a value: separator or closer.
            let (off, byte) = take!();
            match byte {
                b',' => {
                    if is_object {
                        let (koff, kbyte) = take!();
                        if kbyte != b'"' {
                            fail!(koff, "'\"'");
                        }
                        emit_key!(koff);
                    }
                    cnt += 1;
                    if !is_object && cnt & 255 == 0 {
                        sink_try!(sink.array_checkpoint(mark));
                    }
                    let (voff, vbyte) = take!();
                    match vbyte {
                        b'{' => {
                            let frame = if is_object {
                                Frame::Object { mark, cnt }
                            } else {
                                Frame::Array { mark, cnt }
                            };
                            stack.push(frame);
                            sink_try!(sink.begin_object());
                            mark = sink.mark();
                            cnt = 0;
                            is_object = true;
                            fresh = true;
                        }
                        b'[' => {
                            let frame = if is_object {
                                Frame::Object { mark, cnt }
                            } else {
                                Frame::Array { mark, cnt }
                            };
                            stack.push(frame);
                            sink_try!(sink.begin_array());
                            mark = sink.mark();
                            cnt = 0;
                            is_object = false;
                            fresh = true;
                        }
                        _ => emit_scalar!(voff, vbyte),
                    }
                    continue;
                }
                b'}' if is_object => {}
                b']' if !is_object => {}
                _ => fail!(off, "',' or closer"),
            }
        }

        // Scope end: close the current container, then keep closing while
        // parents are also at their closer.
        loop {
            if is_object {
                sink_try!(sink.end_object(mark, cnt));
            } else {
                sink_try!(sink.end_array(mark, cnt));
            }
            match stack.pop().expect("frame stack underflow") {
                Frame::Root => finish!(),
                Frame::Object {
                    mark: pmark,
                    cnt: pcnt,
                } => {
                    mark = pmark;
                    cnt = pcnt;
                    is_object = true;
                }
                Frame::Array {
                    mark: pmark,
                    cnt: pcnt,
                } => {
                    mark = pmark;
                    cnt = pcnt;
                    is_object = false;
                }
            }
            // After the closed child (a value in the parent): ',' or closer.
            let (off, byte) = take!();
            match byte {
                b',' => {
                    if is_object {
                        let (koff, kbyte) = take!();
                        if kbyte != b'"' {
                            fail!(koff, "'\"'");
                        }
                        emit_key!(koff);
                    }
                    cnt += 1;
                    if !is_object && cnt & 255 == 0 {
                        sink_try!(sink.array_checkpoint(mark));
                    }
                    let (voff, vbyte) = take!();
                    match vbyte {
                        b'{' => {
                            let frame = if is_object {
                                Frame::Object { mark, cnt }
                            } else {
                                Frame::Array { mark, cnt }
                            };
                            stack.push(frame);
                            sink_try!(sink.begin_object());
                            mark = sink.mark();
                            cnt = 0;
                            is_object = true;
                            fresh = true;
                        }
                        b'[' => {
                            let frame = if is_object {
                                Frame::Object { mark, cnt }
                            } else {
                                Frame::Array { mark, cnt }
                            };
                            stack.push(frame);
                            sink_try!(sink.begin_array());
                            mark = sink.mark();
                            cnt = 0;
                            is_object = false;
                            fresh = true;
                        }
                        _ => emit_scalar!(voff, vbyte),
                    }
                    break;
                }
                b'}' if is_object => {}
                b']' if !is_object => {}
                _ => fail!(off, "',' or closer"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    #[derive(Default)]
    struct TraceSink {
        out: String,
        depth: usize,
    }

    impl Sink for TraceSink {
        type Error = ();

        fn null(&mut self) -> Result<(), ()> {
            self.out.push_str("n;");
            Ok(())
        }
        fn boolean(&mut self, v: bool) -> Result<(), ()> {
            self.out.push_str(if v { "T;" } else { "F;" });
            Ok(())
        }
        fn int(&mut self, v: i64) -> Result<(), ()> {
            let _ = write!(self.out, "i{v};");
            Ok(())
        }
        fn float(&mut self, v: f64) -> Result<(), ()> {
            let _ = write!(self.out, "f{v};");
            Ok(())
        }
        fn big_int(&mut self, d: &str) -> Result<(), ()> {
            let _ = write!(self.out, "B{d};");
            Ok(())
        }
        fn str(&mut self, s: &str) -> Result<(), ()> {
            let _ = write!(self.out, "s{s};");
            Ok(())
        }
        fn key(&mut self, k: &str) -> Result<(), ()> {
            let _ = write!(self.out, "k{k};");
            Ok(())
        }
        fn mark(&self) -> usize {
            self.depth
        }
        fn end_array(&mut self, _m: usize, len: usize) -> Result<(), ()> {
            let _ = write!(self.out, "A{len};");
            Ok(())
        }
        fn end_object(&mut self, _m: usize, pairs: usize) -> Result<(), ()> {
            let _ = write!(self.out, "O{pairs};");
            Ok(())
        }
    }

    fn trace(doc: &str) -> Result<String, ()> {
        let mut bufs = Buffers::new();
        let mut sink = TraceSink::default();
        parse_indexed(doc, &mut bufs, &mut sink).map_err(|_| ())?;
        Ok(sink.out)
    }

    #[test]
    fn shapes() {
        assert_eq!(trace("{}").unwrap(), "O0;");
        assert_eq!(trace("[]").unwrap(), "A0;");
        assert_eq!(trace("[1,2]").unwrap(), "i1;i2;A2;");
        assert_eq!(trace(r#"{"a":1}"#).unwrap(), "ka;i1;O1;");
        assert_eq!(
            trace(r#"{"a":[true,null],"b":{"c":"x"}}"#).unwrap(),
            "ka;T;n;A2;kb;kc;sx;O1;O2;"
        );
        assert_eq!(trace("[[],[[]]]").unwrap(), "A0;A0;A1;A2;");
        assert_eq!(trace("42").unwrap(), "i42;");
    }

    #[test]
    fn drive_errors() {
        for bad in ["", "{", "[1,]", "[1 2]", r#"{"a"}"#, "{} {}", "]"] {
            assert!(trace(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn unpacked_driver_path_works() {
        // The >16MB unpacked encoding, exercised directly.
        let doc = br#"{"a":[1,"x",true,null]}"#;
        let mut bufs = Buffers::new();
        let (indexes, scratch, stack) = bufs.split_for_driver();
        crate::stage1::index(doc, indexes);
        let mut sink = TraceSink::default();
        super::drive_inner::<_, false>(doc, indexes, scratch, stack, &mut sink).unwrap();
        assert_eq!(sink.out, "ka;i1;sx;T;n;A4;O1;");
    }

    #[test]
    fn benchmark_files_drive_clean() {
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
                let mut sink = TraceSink::default();
                parse_indexed(&data, &mut bufs, &mut sink)
                    .map_err(|e| match e {
                        DriveError::Parse(p) => p.to_string(),
                        DriveError::Sink(()) => "sink".into(),
                    })
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    }
}
