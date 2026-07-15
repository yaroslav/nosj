//! Fused cursor mode: a byte-cursor single-pass parser (the architecture of
//! yyjson and other fused parsers) composed from this crate's SIMD/SWAR
//! kernels.
//! No structural index: whitespace is skipped per token and every input
//! byte is visited exactly once.
//!
//! Container entry is a *flat attempt*: `scalar (, scalar)* closer` is
//! consumed in a tight loop with no frame-stack traffic. Only when an element
//! is itself a container does the parser descend (one frame push, the same
//! cost the generic path always paid) and re-attempt the nested container.
//! Flat leaves (coordinate pairs, small option objects) never touch the
//! frame stack.

use crate::driver::{DriveError, Frame, Sink, sink_key, sink_literal, sink_number, sink_str};
use crate::reader::{Buffers, ParseError};
use crate::scalars;

/// Sentinel `mark` marking the root pseudo-container.
const ROOT_MARK: usize = usize::MAX;

/// Grammar extensions beyond RFC 8259 that many deployed parsers accept.
/// All checks sit on cold paths; the defaults cost nothing per token.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParseOptions {
    /// Accept `NaN`, `Infinity` and `-Infinity` in value position.
    pub allow_nan: bool,
    /// Accept a trailing comma before `]` / `}`.
    pub allow_trailing_comma: bool,
}

/// Skip JSON whitespace. After a newline, runs of indentation spaces are
/// skipped 8 bytes at a time, a fast path for pretty-printed input.
#[inline(always)]
fn skip_ws(input: &[u8], mut i: usize) -> usize {
    while i < input.len() {
        match input[i] {
            b' ' | b'\t' | b'\r' => i += 1,
            b'\n' => {
                i += 1;
                while i + 8 <= input.len() {
                    let chunk = u64::from_le_bytes(input[i..i + 8].try_into().unwrap());
                    if chunk == 0x2020_2020_2020_2020 {
                        i += 8;
                        continue;
                    }
                    i += ((chunk ^ 0x2020_2020_2020_2020).trailing_zeros() / 8) as usize;
                    break;
                }
            }
            _ => break,
        }
    }
    i
}

/// Parse `input` in a single fused pass, pushing events into `sink`.
#[doc(alias = "deserialize")]
#[doc(alias = "from_str")]
pub fn parse<S: Sink>(
    input: &str,
    bufs: &mut Buffers,
    sink: &mut S,
) -> Result<(), DriveError<S::Error>> {
    // SAFETY: &str is valid UTF-8.
    unsafe { parse_utf8_unchecked(input.as_bytes(), bufs, sink) }
}

/// Like [`parse`], with grammar extensions from [`ParseOptions`].
pub fn parse_with<S: Sink>(
    input: &str,
    bufs: &mut Buffers,
    sink: &mut S,
    opts: ParseOptions,
) -> Result<(), DriveError<S::Error>> {
    // SAFETY: &str is valid UTF-8.
    unsafe { parse_utf8_unchecked_with(input.as_bytes(), bufs, sink, opts) }
}

/// Like [`parse`], for callers whose runtime vouches for UTF-8.
///
/// # Safety
///
/// `input` must be valid UTF-8.
pub unsafe fn parse_utf8_unchecked<S: Sink>(
    input: &[u8],
    bufs: &mut Buffers,
    sink: &mut S,
) -> Result<(), DriveError<S::Error>> {
    // SAFETY: forwarded contract; the caller vouches `input` is UTF-8.
    unsafe { parse_utf8_unchecked_with(input, bufs, sink, ParseOptions::default()) }
}

/// Like [`parse_utf8_unchecked`], with grammar extensions.
///
/// # Safety
///
/// `input` must be valid UTF-8.
pub unsafe fn parse_utf8_unchecked_with<S: Sink>(
    input: &[u8],
    bufs: &mut Buffers,
    sink: &mut S,
    opts: ParseOptions,
) -> Result<(), DriveError<S::Error>> {
    let (_indexes, scratch, stack) = bufs.split_for_driver();
    stack.clear();
    let mut pos: usize = 0;

    macro_rules! next_significant {
        () => {{
            pos = skip_ws(input, pos);
            if pos < input.len() {
                (pos, input[pos])
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

    // The scalar bodies live in the shared `sink_*` helpers; the cursor
    // continues from the end offset they return. Only the `allow_nan`
    // keywords are cursor-specific.
    macro_rules! emit_scalar {
        ($off:expr, $byte:expr) => {{
            match $byte {
                b'"' => {
                    // SAFETY: `input` is valid UTF-8 per this function's
                    // contract, and `$off` holds the `"` the tokenizer
                    // just matched.
                    pos = unsafe { sink_str(input, $off, scratch, sink) }?;
                }
                b'-' | b'0'..=b'9' => {
                    if $byte == b'-' && opts.allow_nan && input.get($off + 1) == Some(&b'I') {
                        nan_keyword!($off, b"-Infinity", f64::NEG_INFINITY);
                    } else {
                        pos = sink_number(input, $off, sink)?;
                    }
                }
                b't' | b'f' | b'n' => pos = sink_literal(input, $off, sink)?,
                b'N' if opts.allow_nan => nan_keyword!($off, b"NaN", f64::NAN),
                b'I' if opts.allow_nan => nan_keyword!($off, b"Infinity", f64::INFINITY),
                _ => return Err(ParseError::unexpected_character($off).into()),
            }
        }};
    }

    /// `allow_nan` keywords: match, boundary-check, emit as float.
    macro_rules! nan_keyword {
        ($off:expr, $kw:expr, $val:expr) => {{
            let end = $off + $kw.len();
            let matched = input.len() >= end
                && &input[$off..end] == $kw
                && input
                    .get(end)
                    .is_none_or(|&b| scalars::is_token_boundary(b));
            if !matched {
                return Err(ParseError::unexpected_character($off).into());
            }
            pos = end;
            sink_try!(sink.float($val));
        }};
    }

    macro_rules! emit_key {
        ($off:expr) => {{
            // SAFETY: `input` is valid UTF-8 per this function's contract,
            // and `$off` holds the `"` the tokenizer just matched.
            pos = unsafe { sink_key(input, $off, scratch, sink) }?;
            // The colon almost always follows the key's closing quote
            // directly ("key": value); skip the whitespace machinery then.
            if input.get(pos) == Some(&b':') {
                pos += 1;
            } else {
                let (coff, cbyte) = next_significant!();
                if cbyte != b':' {
                    fail!(coff, "':'");
                }
                pos += 1;
            }
        }};
    }

    macro_rules! finish {
        () => {{
            pos = skip_ws(input, pos);
            if pos < input.len() {
                return Err(ParseError::trailing_characters(pos).into());
            }
            return Ok(());
        }};
    }

    // Current container registers; parents on the frame stack.
    let mut mark: usize = ROOT_MARK;
    let mut cnt: usize = 0;
    let mut is_object: bool = false;

    macro_rules! frame_from_current {
        () => {{
            if mark == ROOT_MARK {
                Frame::Root
            } else if is_object {
                Frame::Object { mark, cnt }
            } else {
                Frame::Array { mark, cnt }
            }
        }};
    }

    /// Consume a container starting at its (already consumed) opener.
    /// Flat runs of scalars are consumed with no frame traffic; a nested
    /// container pushes the current registers and re-attempts one level in.
    /// On exit, a complete container value has been emitted into the
    /// (possibly descended) current container.
    macro_rules! enter_container {
        ($opener:expr) => {{
            let mut opener = $opener;
            pos += 1;
            'attempt: loop {
                if opener == b'{' {
                    sink_try!(sink.begin_object());
                    let omark = sink.mark();
                    let mut ocnt: usize = 0;
                    loop {
                        let (koff, kbyte) = next_significant!();
                        // ocnt > 0 here means we are right after a comma.
                        if kbyte == b'}' && (ocnt == 0 || opts.allow_trailing_comma) {
                            pos += 1;
                            sink_try!(sink.end_object(omark, ocnt));
                            break 'attempt;
                        }
                        if kbyte != b'"' {
                            fail!(koff, "'\"' or '}'");
                        }
                        emit_key!(koff);
                        ocnt += 1;
                        let (voff, vbyte) = next_significant!();
                        match vbyte {
                            b'{' | b'[' => {
                                stack.push(frame_from_current!());
                                mark = omark;
                                cnt = ocnt;
                                is_object = true;
                                opener = vbyte;
                                pos += 1;
                                continue 'attempt;
                            }
                            _ => emit_scalar!(voff, vbyte),
                        }
                        let (soff, sbyte) = next_significant!();
                        match sbyte {
                            b',' => pos += 1,
                            b'}' => {
                                pos += 1;
                                sink_try!(sink.end_object(omark, ocnt));
                                break 'attempt;
                            }
                            _ => fail!(soff, "',' or '}'"),
                        }
                    }
                } else {
                    sink_try!(sink.begin_array());
                    let amark = sink.mark();
                    let mut acnt: usize = 0;
                    loop {
                        let (eoff, ebyte) = next_significant!();
                        // acnt > 0 here means we are right after a comma.
                        if ebyte == b']' && (acnt == 0 || opts.allow_trailing_comma) {
                            pos += 1;
                            sink_try!(sink.end_array(amark, acnt));
                            break 'attempt;
                        }
                        acnt += 1;
                        match ebyte {
                            b'{' | b'[' => {
                                stack.push(frame_from_current!());
                                mark = amark;
                                cnt = acnt;
                                is_object = false;
                                opener = ebyte;
                                pos += 1;
                                continue 'attempt;
                            }
                            _ => emit_scalar!(eoff, ebyte),
                        }
                        // Dense-array fast path: in minified numeric data the
                        // comma follows the value with no whitespace.
                        if input.get(pos) == Some(&b',') {
                            pos += 1;
                            if acnt & 255 == 0 {
                                sink_try!(sink.array_checkpoint(amark));
                            }
                            continue;
                        }
                        let (soff, sbyte) = next_significant!();
                        match sbyte {
                            b',' => {
                                pos += 1;
                                if acnt & 255 == 0 {
                                    sink_try!(sink.array_checkpoint(amark));
                                }
                            }
                            b']' => {
                                pos += 1;
                                sink_try!(sink.end_array(amark, acnt));
                                break 'attempt;
                            }
                            _ => fail!(soff, "',' or ']'"),
                        }
                    }
                }
            }
        }};
    }

    // Root value.
    let (off, byte) = next_significant!();
    match byte {
        b'{' | b'[' => {
            stack.push(Frame::Root);
            enter_container!(byte);
        }
        _ => {
            emit_scalar!(off, byte);
            finish!();
        }
    }

    // Main loop: a value just completed inside the current container (or at
    // the root). Handle the separator or closer, then the next value.
    loop {
        if mark == ROOT_MARK {
            finish!();
        }
        let (off, byte) = next_significant!();
        match byte {
            b',' => {
                pos += 1;
                if is_object {
                    let (koff, kbyte) = next_significant!();
                    if kbyte == b'}' && opts.allow_trailing_comma {
                        continue; // main loop re-reads and closes the object
                    }
                    if kbyte != b'"' {
                        fail!(koff, "'\"'");
                    }
                    emit_key!(koff);
                } else {
                    if opts.allow_trailing_comma {
                        let (_, peeked) = next_significant!();
                        if peeked == b']' {
                            continue; // main loop re-reads and closes the array
                        }
                    }
                    if cnt & 255 == 0 {
                        sink_try!(sink.array_checkpoint(mark));
                    }
                }
                cnt += 1;
                let (voff, vbyte) = next_significant!();
                match vbyte {
                    b'{' | b'[' => enter_container!(vbyte),
                    _ => emit_scalar!(voff, vbyte),
                }
            }
            b'}' if is_object => {
                pos += 1;
                sink_try!(sink.end_object(mark, cnt));
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
            }
            b']' if !is_object => {
                pos += 1;
                sink_try!(sink.end_array(mark, cnt));
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
            }
            _ => fail!(off, "',' or closer"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver;
    use std::fmt::Write;

    /// Sink that records the event stream as a string (same as driver tests).
    #[derive(Default)]
    struct TraceSink {
        out: String,
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
            0
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

    fn cursor_trace(doc: &str) -> Result<String, ()> {
        let mut bufs = Buffers::new();
        let mut sink = TraceSink::default();
        parse(doc, &mut bufs, &mut sink).map_err(|_| ())?;
        Ok(sink.out)
    }

    #[test]
    fn shapes() {
        assert_eq!(cursor_trace("{}").unwrap(), "O0;");
        assert_eq!(cursor_trace(" [ 1 , 2 ] ").unwrap(), "i1;i2;A2;");
        assert_eq!(
            cursor_trace("{\n  \"a\": [true, null],\n  \"b\": {\"c\": \"x\"}\n}").unwrap(),
            "ka;T;n;A2;kb;kc;sx;O1;O2;"
        );
        assert_eq!(cursor_trace("42").unwrap(), "i42;");
        // Flat-attempt descend paths.
        assert_eq!(
            cursor_trace("[[1,2],[3,4]]").unwrap(),
            "i1;i2;A2;i3;i4;A2;A2;"
        );
        assert_eq!(cursor_trace("[[],[[]]]").unwrap(), "A0;A0;A1;A2;");
        assert_eq!(
            cursor_trace(r#"{"a":{"b":{"c":1}},"d":2}"#).unwrap(),
            "ka;kb;kc;i1;O1;O1;kd;i2;O2;"
        );
    }

    #[test]
    fn cursor_errors() {
        for bad in [
            "", "{", "[1,]", "[1 2]", r#"{"a"}"#, "{} {}", "]", "01", "truex", "{,}",
        ] {
            assert!(cursor_trace(bad).is_err(), "accepted {bad:?}");
        }
    }

    fn cursor_trace_with(doc: &str, opts: ParseOptions) -> Result<String, ()> {
        let mut bufs = Buffers::new();
        let mut sink = TraceSink::default();
        // The safe options entry point: exercising it here covers the
        // whole option matrix through the safe API.
        parse_with(doc, &mut bufs, &mut sink, opts).map_err(|_| ())?;
        Ok(sink.out)
    }

    #[test]
    fn allow_nan_option() {
        let opts = ParseOptions {
            allow_nan: true,
            ..Default::default()
        };
        assert_eq!(
            cursor_trace_with("[NaN, Infinity, -Infinity]", opts).unwrap(),
            "fNaN;finf;f-inf;A3;"
        );
        assert_eq!(cursor_trace_with("NaN", opts).unwrap(), "fNaN;");
        // Boundary and default-mode rejection.
        assert!(cursor_trace_with("[NaNx]", opts).is_err());
        assert!(cursor_trace_with("[Inf]", opts).is_err());
        assert!(cursor_trace("[NaN]").is_err());
    }

    #[test]
    fn allow_trailing_comma_option() {
        let opts = ParseOptions {
            allow_trailing_comma: true,
            ..Default::default()
        };
        assert_eq!(cursor_trace_with("[1,2,]", opts).unwrap(), "i1;i2;A2;");
        assert_eq!(cursor_trace_with(r#"{"a":1,}"#, opts).unwrap(), "ka;i1;O1;");
        // Through the main loop (containers with nested children).
        assert_eq!(
            cursor_trace_with("[[1],[2],]", opts).unwrap(),
            "i1;A1;i2;A1;A2;"
        );
        assert_eq!(
            cursor_trace_with(r#"{"a":{"b":1},}"#, opts).unwrap(),
            "ka;kb;i1;O1;O1;"
        );
        // A comma alone is still invalid.
        assert!(cursor_trace_with("[,]", opts).is_err());
        assert!(cursor_trace_with("[1,,]", opts).is_err());
        assert!(cursor_trace("[1,]").is_err());
    }

    /// The decisive test: cursor mode and index mode must produce identical
    /// event streams on every benchmark file.
    #[test]
    fn cursor_matches_index_driver() {
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
                let mut a = TraceSink::default();
                driver::parse_indexed(&data, &mut bufs, &mut a)
                    .map_err(|_| ())
                    .unwrap();
                let mut b = TraceSink::default();
                parse(&data, &mut bufs, &mut b).map_err(|_| ()).unwrap();
                assert_eq!(a.out, b.out, "event stream mismatch on {name}");
            }
        }
    }
}
