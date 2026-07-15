//! RFC 6901 JSON Pointer resolution: partial parsing's front door.
//!
//! Resolution is a forward byte cursor: it tokenizes only the navigation
//! levels it walks through and steps over everything else. Sibling
//! containers are skipped by [`crate::stage1::container_end`] (64-byte
//! blocks classified exactly like indexing, but consuming only bracket
//! bits, stopping at the matching closer, and building no index), so the
//! cost of a query is proportional to how far into the document the
//! target sits, not to the document's size. The matched value comes back
//! as its raw text, ready to feed to any nosj entry point, or to keep
//! as a lazy value.
//!
//! Miss semantics match `serde_json::Value::pointer`: a missing key, an
//! out-of-range index, or a token that is not valid array-index syntax
//! (`-`, `01`, non-digits) resolves to `Ok(None)`. Only malformed pointer
//! *syntax* (a non-empty pointer without a leading `/`) is an error.
//!
//! Duplicate keys resolve to the **first** occurrence (streaming
//! semantics; a last-wins map like serde's sees the last).

use crate::reader::{Buffers, ErrorKind, ParseError};
use crate::scalars::{self, StrPart};
use crate::stage1;

/// Resolve `pointer` against `input`, returning the raw text of the
/// matched value (`None` if the pointer doesn't match anything).
///
/// ```
/// let mut bufs = nosj::Buffers::new();
/// let doc = r#"{"users":[{"name":"ada"},{"name":"grace"}]}"#;
/// let raw = nosj::pointer(doc, "/users/1/name", &mut bufs).unwrap();
/// assert_eq!(raw, Some("\"grace\""));
/// ```
#[doc(alias = "get")]
#[doc(alias = "dig")]
#[doc(alias = "json_pointer")]
pub fn pointer<'j>(
    input: &'j str,
    pointer: &str,
    bufs: &mut Buffers,
) -> Result<Option<&'j str>, ParseError> {
    // SAFETY: &str is valid UTF-8.
    unsafe { pointer_utf8_unchecked(input.as_bytes(), pointer, bufs) }
}

/// Like [`pointer()`], for callers whose runtime vouches for UTF-8.
///
/// # Safety
///
/// `input` must be valid UTF-8.
pub unsafe fn pointer_utf8_unchecked<'j>(
    input: &'j [u8],
    pointer: &str,
    bufs: &mut Buffers,
) -> Result<Option<&'j str>, ParseError> {
    if !pointer.is_empty() && !pointer.starts_with('/') {
        return Err(ParseError {
            offset: 0,
            kind: ErrorKind::InvalidPointer,
        });
    }
    // SAFETY: forwarded contract; the caller vouches `input` is UTF-8.
    let range = unsafe { resolve(input, pointer, &mut bufs.scratch) }?;
    // SAFETY: `input` is valid UTF-8 and the range falls on token edges
    // (ASCII boundaries).
    Ok(range.map(|(start, end)| unsafe { std::str::from_utf8_unchecked(&input[start..end]) }))
}

#[inline(always)]
fn skip_ws(input: &[u8], mut i: usize) -> usize {
    while let Some(&b) = input.get(i) {
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        } else {
            break;
        }
    }
    i
}

/// End offset of the value starting at `i` (whitespace already skipped).
/// Scalars run their tokenizer (fully validated); containers are skipped
/// structurally by [`stage1::container_end`].
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn value_end(input: &[u8], i: usize, scratch: &mut Vec<u8>) -> Result<usize, ParseError> {
    let eof = || ParseError::unexpected_end(input.len());
    match *input.get(i).ok_or_else(eof)? {
        // SAFETY: UTF-8 per this function's contract; `i` holds the `"`.
        b'"' => unsafe { scalars::parse_string(input, i, scratch) }
            .map(|(_, end)| end)
            .map_err(|e| ParseError::scalar(e, i)),
        b'-' | b'0'..=b'9' => scalars::parse_number(input, i)
            .map(|(_, end)| end)
            .map_err(|e| ParseError::scalar(e, i)),
        b't' | b'f' | b'n' => scalars::parse_literal(input, i)
            .map(|(_, end)| end)
            .map_err(|e| ParseError::scalar(e, i)),
        b'{' | b'[' => stage1::container_end(input, i).ok_or_else(eof),
        _ => Err(ParseError::unexpected_character(i)),
    }
}

/// Walk the pointer tokens, returning the byte range of the match.
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn resolve(
    input: &[u8],
    pointer: &str,
    scratch: &mut Vec<u8>,
) -> Result<Option<(usize, usize)>, ParseError> {
    let mut i = skip_ws(input, 0);

    if pointer.is_empty() {
        // The whole document is the match; enforce a complete document
        // like a full parse would.
        // SAFETY: forwarded UTF-8 contract.
        let end = unsafe { value_end(input, i, scratch) }?;
        let trail = skip_ws(input, end);
        if trail != input.len() {
            return Err(ParseError::trailing_characters(trail));
        }
        return Ok(Some((i, end)));
    }

    let mut tokens = pointer.split('/').skip(1).peekable();
    while let Some(token) = tokens.next() {
        let is_last = tokens.peek().is_none();
        let value_pos = match input.get(i).copied() {
            None => return Err(ParseError::unexpected_end(input.len())),
            Some(b'{') => {
                // SAFETY: forwarded UTF-8 contract.
                unsafe { descend_object(input, i, token, scratch) }?
            }
            Some(b'[') => {
                // SAFETY: forwarded UTF-8 contract.
                unsafe { descend_array(input, i, token, scratch) }?
            }
            // A scalar cannot be descended into.
            Some(_) => None,
        };
        let Some(value_pos) = value_pos else {
            return Ok(None);
        };
        i = value_pos;
        if is_last {
            // SAFETY: forwarded UTF-8 contract.
            let end = unsafe { value_end(input, i, scratch) }?;
            return Ok(Some((i, end)));
        }
    }
    unreachable!("the token loop returns on its last iteration");
}

/// Resolve many pointers in **one forward pass**, returning raw slices
/// positionally aligned with `pointers` (`None` where a pointer doesn't
/// match). Sequential [`pointer()`] calls re-scan the document prefix per
/// query; this walks it once, descending only into members some pointer
/// still needs and skipping everything else at block speed.
///
/// Duplicate keys resolve first-match, like [`pointer()`]. The walk stops
/// as soon as every pointer is resolved.
///
/// On **valid** JSON the result is exactly what resolving each pointer
/// individually would produce. On *malformed* documents the batch may
/// report an error where an individual [`pointer()`] call would return
/// `Ok(None)`: one pass scans every byte *some* pointer needs, so it can
/// reach malformed content that a lone query (which aborts at its first
/// missing key) never would.
///
/// ```
/// let mut bufs = nosj::Buffers::new();
/// let doc = r#"{"a":1,"b":{"c":2},"d":[3,4]}"#;
/// let got = nosj::pointers(doc, &["/d/1", "/a", "/x"], &mut bufs).unwrap();
/// assert_eq!(got, [Some("4"), Some("1"), None]);
/// ```
#[doc(alias = "get_many")]
pub fn pointers<'j>(
    input: &'j str,
    pointers: &[&str],
    bufs: &mut Buffers,
) -> Result<Vec<Option<&'j str>>, ParseError> {
    // SAFETY: &str is valid UTF-8.
    unsafe { pointers_utf8_unchecked(input.as_bytes(), pointers, bufs) }
}

/// Like [`pointers()`], for callers whose runtime vouches for UTF-8.
///
/// # Safety
///
/// `input` must be valid UTF-8.
pub unsafe fn pointers_utf8_unchecked<'j>(
    input: &'j [u8],
    pointers: &[&str],
    bufs: &mut Buffers,
) -> Result<Vec<Option<&'j str>>, ParseError> {
    // Tokenize (and unescape) every pointer up front: once per token,
    // not once per level visit. The arena owns the tokens; queries borrow.
    let arena: Vec<Vec<Token<'_>>> = pointers
        .iter()
        .map(|ptr| {
            if ptr.is_empty() {
                Vec::new()
            } else {
                ptr.split('/').skip(1).map(Token::parse).collect()
            }
        })
        .collect();
    let mut queries: Vec<Query<'_>> = Vec::with_capacity(pointers.len());
    for (id, ptr) in pointers.iter().enumerate() {
        if !ptr.is_empty() && !ptr.starts_with('/') {
            return Err(ParseError {
                offset: 0,
                kind: ErrorKind::InvalidPointer,
            });
        }
        queries.push(Query {
            id,
            tokens: &arena[id],
        });
    }

    let mut results: Vec<Option<(usize, usize)>> = vec![None; pointers.len()];
    let mut remaining = queries.len();
    let start = skip_ws(input, 0);
    // SAFETY: forwarded UTF-8 contract.
    unsafe {
        resolve_set(
            input,
            start,
            &queries,
            &mut results,
            &mut remaining,
            &mut bufs.scratch,
        )
    }?;
    // Root pointers keep the single-pointer contract of validating a
    // complete document; a root query always resolves, so its recorded
    // end is always available.
    if let Some(root) = pointers.iter().position(|p| p.is_empty()) {
        let (_, end) = results[root].expect("the root pointer always matches");
        let trail = skip_ws(input, end);
        if trail != input.len() {
            return Err(ParseError::trailing_characters(trail));
        }
    }
    Ok(results
        .into_iter()
        // SAFETY: `input` is valid UTF-8 and the ranges fall on token
        // edges (ASCII boundaries).
        .map(|r| r.map(|(s, e)| unsafe { std::str::from_utf8_unchecked(&input[s..e]) }))
        .collect())
}

/// One pre-tokenized pointer still being resolved: its slot in the
/// results and the tokens below the current level.
#[derive(Clone, Copy)]
struct Query<'p> {
    id: usize,
    tokens: &'p [Token<'p>],
}

/// One pointer token, unescaped once, with its array-index reading
/// precomputed.
struct Token<'p> {
    text: std::borrow::Cow<'p, str>,
    /// `Some` when the raw token is valid index syntax (digits, no
    /// leading zeros).
    index: Option<usize>,
}

impl<'p> Token<'p> {
    fn parse(raw: &'p str) -> Token<'p> {
        let valid_index = !raw.is_empty()
            && raw.bytes().all(|b| b.is_ascii_digit())
            && (raw == "0" || !raw.starts_with('0'));
        Token {
            // `~1` -> `/`, then `~0` -> `~` (RFC 6901 order).
            text: if raw.contains('~') {
                raw.replace("~1", "/").replace("~0", "~").into()
            } else {
                raw.into()
            },
            index: if valid_index { raw.parse().ok() } else { None },
        }
    }
}

/// Resolve `queries` against the value at `i`. Returns `Some(end)` of the
/// value, or `None` when every pointer in the whole batch has been
/// resolved (the abort signal; no caller needs positions after that).
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn resolve_set(
    input: &[u8],
    i: usize,
    queries: &[Query<'_>],
    results: &mut Vec<Option<(usize, usize)>>,
    remaining: &mut usize,
    scratch: &mut Vec<u8>,
) -> Result<Option<usize>, ParseError> {
    let deeper: Vec<Query<'_>> = queries
        .iter()
        .copied()
        .filter(|q| !q.tokens.is_empty())
        .collect();

    let end = match input.get(i).copied() {
        None => return Err(ParseError::unexpected_end(input.len())),
        Some(b'{') if !deeper.is_empty() => {
            // SAFETY: forwarded UTF-8 contract.
            let walked = unsafe { walk_object(input, i, &deeper, results, remaining, scratch) }?;
            match walked {
                Some(end) => end,
                None => return Ok(None),
            }
        }
        Some(b'[') if !deeper.is_empty() => {
            // SAFETY: forwarded UTF-8 contract.
            let walked = unsafe { walk_array(input, i, &deeper, results, remaining, scratch) }?;
            match walked {
                Some(end) => end,
                None => return Ok(None),
            }
        }
        // Scalar, or a container no pointer descends into: one skip.
        // SAFETY: forwarded UTF-8 contract.
        _ => unsafe { value_end(input, i, scratch) }?,
    };

    for q in queries {
        if q.tokens.is_empty() {
            results[q.id] = Some((i, end));
            *remaining -= 1;
        }
    }
    if *remaining == 0 {
        return Ok(None);
    }
    Ok(Some(end))
}

/// Walk one object once, recursing into members some query names.
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn walk_object(
    input: &[u8],
    mut i: usize,
    deeper: &[Query<'_>],
    results: &mut Vec<Option<(usize, usize)>>,
    remaining: &mut usize,
    scratch: &mut Vec<u8>,
) -> Result<Option<usize>, ParseError> {
    i = skip_ws(input, i + 1);
    if input.get(i) == Some(&b'}') {
        return Ok(Some(i + 1)); // empty object
    }
    // First-match-wins, hit or miss: the first member with a matching key
    // consumes the query, so a later duplicate key can never re-resolve
    // it (parity with single [`pointer()`] resolution).
    let mut live: Vec<Query<'_>> = deeper.to_vec();
    loop {
        if input.get(i) != Some(&b'"') {
            return Err(ParseError::expected("'\"'", i));
        }
        // SAFETY: forwarded UTF-8 contract; `i` holds a `"`.
        let (part, key_end) = unsafe { scalars::parse_string(input, i, scratch) }
            .map_err(|e| ParseError::scalar(e, i))?;
        // WTF-8 keys cannot equal a &str pointer token.
        let mut advanced: Vec<Query<'_>> = Vec::new();
        if let StrPart::Borrowed(k) | StrPart::Decoded(k) = part {
            live.retain(|q| {
                let hit = q.tokens[0].text == k;
                if hit {
                    advanced.push(Query {
                        id: q.id,
                        tokens: &q.tokens[1..],
                    });
                }
                !hit
            });
        }
        i = skip_ws(input, key_end);
        if input.get(i) != Some(&b':') {
            return Err(ParseError::expected("':'", i));
        }
        i = skip_ws(input, i + 1);
        let value_past = if advanced.is_empty() {
            // SAFETY: forwarded UTF-8 contract.
            unsafe { value_end(input, i, scratch) }?
        } else {
            // SAFETY: forwarded UTF-8 contract.
            match unsafe { resolve_set(input, i, &advanced, results, remaining, scratch) }? {
                Some(end) => end,
                None => return Ok(None),
            }
        };
        i = skip_ws(input, value_past);
        match input.get(i) {
            Some(b',') => i = skip_ws(input, i + 1),
            Some(b'}') => return Ok(Some(i + 1)),
            Some(_) => return Err(ParseError::expected("',' or '}'", i)),
            None => return Err(ParseError::unexpected_end(input.len())),
        }
    }
}

/// Walk one array once, recursing at indices some query names.
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn walk_array(
    input: &[u8],
    mut i: usize,
    deeper: &[Query<'_>],
    results: &mut Vec<Option<(usize, usize)>>,
    remaining: &mut usize,
    scratch: &mut Vec<u8>,
) -> Result<Option<usize>, ParseError> {
    i = skip_ws(input, i + 1);
    if input.get(i) == Some(&b']') {
        return Ok(Some(i + 1)); // empty array
    }
    // Elements past the largest queried index are skipped, not walked.
    let max_wanted = deeper.iter().filter_map(|q| q.tokens[0].index).max();
    let mut element = 0usize;
    loop {
        let advanced: Vec<Query<'_>> = deeper
            .iter()
            .filter(|q| q.tokens[0].index == Some(element))
            .map(|q| Query {
                id: q.id,
                tokens: &q.tokens[1..],
            })
            .collect();
        let value_past = if advanced.is_empty() {
            // SAFETY: forwarded UTF-8 contract.
            unsafe { value_end(input, i, scratch) }?
        } else {
            // SAFETY: forwarded UTF-8 contract.
            match unsafe { resolve_set(input, i, &advanced, results, remaining, scratch) }? {
                Some(end) => end,
                None => return Ok(None),
            }
        };
        i = skip_ws(input, value_past);
        match input.get(i) {
            Some(b',') => i = skip_ws(input, i + 1),
            Some(b']') => return Ok(Some(i + 1)),
            Some(_) => return Err(ParseError::expected("',' or ']'", i)),
            None => return Err(ParseError::unexpected_end(input.len())),
        }
        element += 1;
        // Nothing deeper wants later elements; no caller needs this
        // array's end if the whole batch resolves inside it, but the
        // parent still might, so skip the remainder structurally.
        if max_wanted.is_some_and(|m| element > m) {
            // Rewind to the opener is impossible; finish by skipping
            // element-by-element (cheap: container_end per element).
            loop {
                // SAFETY: forwarded UTF-8 contract.
                let past = unsafe { value_end(input, i, scratch) }?;
                i = skip_ws(input, past);
                match input.get(i) {
                    Some(b',') => i = skip_ws(input, i + 1),
                    Some(b']') => return Ok(Some(i + 1)),
                    Some(_) => return Err(ParseError::expected("',' or ']'", i)),
                    None => return Err(ParseError::unexpected_end(input.len())),
                }
            }
        }
    }
}

/// One object level: find `token`'s value, skipping non-matching members.
/// `i` is at the `{`; returns the matched value's position, or `None`
/// when the key is absent.
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn descend_object(
    input: &[u8],
    mut i: usize,
    token: &str,
    scratch: &mut Vec<u8>,
) -> Result<Option<usize>, ParseError> {
    // Unescape once per level: `~1` -> `/`, then `~0` -> `~` (RFC 6901
    // order). Clean tokens compare borrowed.
    let unescaped: std::borrow::Cow<'_, str> = if token.contains('~') {
        token.replace("~1", "/").replace("~0", "~").into()
    } else {
        token.into()
    };
    i = skip_ws(input, i + 1);
    if input.get(i) == Some(&b'}') {
        return Ok(None); // empty object
    }
    loop {
        if input.get(i) != Some(&b'"') {
            return Err(ParseError::expected("'\"'", i));
        }
        // SAFETY: forwarded UTF-8 contract; `i` holds a `"`.
        let (part, key_end) = unsafe { scalars::parse_string(input, i, scratch) }
            .map_err(|e| ParseError::scalar(e, i))?;
        let matched = match part {
            StrPart::Borrowed(k) | StrPart::Decoded(k) => k == unescaped,
            // WTF-8 keys cannot equal a &str pointer token.
            StrPart::DecodedRaw(_) => false,
        };
        i = skip_ws(input, key_end);
        if input.get(i) != Some(&b':') {
            return Err(ParseError::expected("':'", i));
        }
        i = skip_ws(input, i + 1);
        if matched {
            return Ok(Some(i)); // the value at `i` is this level's match
        }
        // SAFETY: forwarded UTF-8 contract.
        let value_past = unsafe { value_end(input, i, scratch) }?;
        i = skip_ws(input, value_past);
        match input.get(i) {
            Some(b',') => i = skip_ws(input, i + 1),
            Some(b'}') => return Ok(None), // key not present
            Some(_) => return Err(ParseError::expected("',' or '}'", i)),
            None => return Err(ParseError::unexpected_end(input.len())),
        }
    }
}

/// One array level: find element `token`'s position, skipping earlier
/// elements. `i` is at the `[`; returns `None` for out-of-range indices
/// or tokens that are not valid index syntax.
///
/// # Safety
///
/// `input` must be valid UTF-8.
unsafe fn descend_array(
    input: &[u8],
    mut i: usize,
    token: &str,
    scratch: &mut Vec<u8>,
) -> Result<Option<usize>, ParseError> {
    // Array-index tokens are digits without leading zeros; anything else
    // ("-", "01", "1x", "") cannot match.
    let valid = !token.is_empty()
        && token.bytes().all(|b| b.is_ascii_digit())
        && (token == "0" || !token.starts_with('0'));
    let target = if valid {
        token.parse::<usize>().ok()
    } else {
        None
    };
    let Some(target) = target else {
        return Ok(None);
    };
    i = skip_ws(input, i + 1);
    if input.get(i) == Some(&b']') {
        return Ok(None); // empty array
    }
    for _ in 0..target {
        // SAFETY: forwarded UTF-8 contract.
        let value_past = unsafe { value_end(input, i, scratch) }?;
        i = skip_ws(input, value_past);
        match input.get(i) {
            Some(b',') => i = skip_ws(input, i + 1),
            Some(b']') => return Ok(None), // out of range
            Some(_) => return Err(ParseError::expected("',' or ']'", i)),
            None => return Err(ParseError::unexpected_end(input.len())),
        }
    }
    Ok(Some(i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<'j>(doc: &'j str, ptr: &str) -> Option<&'j str> {
        let mut bufs = Buffers::new();
        pointer(doc, ptr, &mut bufs).unwrap()
    }

    /// The RFC 6901 §5 example document and pointer set.
    #[test]
    fn rfc6901_examples() {
        let doc = r#"{
            "foo": ["bar", "baz"],
            "": 0,
            "a/b": 1,
            "c%d": 2,
            "e^f": 3,
            "g|h": 4,
            "i\\j": 5,
            "k\"l": 6,
            " ": 7,
            "m~n": 8
        }"#;
        assert_eq!(get(doc, ""), Some(doc.trim()));
        assert_eq!(get(doc, "/foo"), Some(r#"["bar", "baz"]"#));
        assert_eq!(get(doc, "/foo/0"), Some("\"bar\""));
        assert_eq!(get(doc, "/"), Some("0"));
        assert_eq!(get(doc, "/a~1b"), Some("1"));
        assert_eq!(get(doc, "/c%d"), Some("2"));
        assert_eq!(get(doc, "/e^f"), Some("3"));
        assert_eq!(get(doc, "/g|h"), Some("4"));
        assert_eq!(get(doc, "/i\\j"), Some("5"));
        assert_eq!(get(doc, "/k\"l"), Some("6"));
        assert_eq!(get(doc, "/ "), Some("7"));
        assert_eq!(get(doc, "/m~0n"), Some("8"));
    }

    #[test]
    fn misses_are_none() {
        let doc = r#"{"a":[10,20],"b":{"c":true}}"#;
        assert_eq!(get(doc, "/missing"), None);
        assert_eq!(get(doc, "/a/2"), None); // out of range
        assert_eq!(get(doc, "/a/-"), None); // past-the-end token
        assert_eq!(get(doc, "/a/01"), None); // leading zero
        assert_eq!(get(doc, "/a/x"), None); // not an index
        assert_eq!(get(doc, "/a/0/deeper"), None); // scalar descent
        assert_eq!(get(doc, "/b/c/d"), None);
        assert_eq!(get("[]", "/0"), None); // empty array
    }

    #[test]
    fn nested_and_escaped() {
        let doc = r#"{"a":{"b":[{"deep":[1,{"x":"found"}]}]},"z":9}"#;
        assert_eq!(get(doc, "/a/b/0/deep/1/x"), Some("\"found\""));
        assert_eq!(get(doc, "/z"), Some("9"));
        // Keys with escapes in the document decode before comparing.
        let doc = r#"{"k\ney": 42}"#;
        assert_eq!(get(doc, "/k\ney"), Some("42"));
        // Duplicate keys: first occurrence wins.
        let doc = r#"{"d":1,"d":2}"#;
        assert_eq!(get(doc, "/d"), Some("1"));
        // Brackets and quotes inside skipped strings don't confuse the
        // structural skip.
        let doc = r#"{"skip":["}{][", "a\"b", {"x":"]"}],"hit":1}"#;
        assert_eq!(get(doc, "/hit"), Some("1"));
    }

    #[test]
    fn errors() {
        let mut bufs = Buffers::new();
        // Malformed pointer syntax.
        assert_eq!(
            pointer("{}", "no-slash", &mut bufs).unwrap_err().kind,
            ErrorKind::InvalidPointer
        );
        // Malformed document still errors even when skipping.
        assert!(pointer("{\"a\":", "/a", &mut bufs).is_err());
        assert!(pointer("[1,2", "/5", &mut bufs).is_err());
    }

    /// The batch resolver must agree with sequential single resolution
    /// on every pointer, in every combination.
    #[test]
    fn batch_matches_sequential() {
        let doc = r#"{
            "users": [{"name":"ada","tags":[1,2]},{"name":"grace"}],
            "a/b": 5, "m~n": {"deep": true}, "empty": {}, "arr": []
        }"#;
        let ptrs = [
            "/users/1/name",
            "/users/0/tags/1",
            "/users/0/name",
            "/missing",
            "/users/5",
            "/a~1b",
            "/m~0n/deep",
            "/empty/x",
            "/arr/0",
            "/users/0/tags",
            "",
        ];
        let mut bufs = Buffers::new();
        let batch = pointers(doc, &ptrs, &mut bufs).unwrap();
        for (ptr, got) in ptrs.iter().zip(&batch) {
            let single = pointer(doc, ptr, &mut bufs).unwrap();
            assert_eq!(*got, single, "pointer {ptr:?}");
        }
        // Duplicate pointers each get their own aligned answer.
        let batch = pointers(doc, &["/a~1b", "/a~1b"], &mut bufs).unwrap();
        assert_eq!(batch, [Some("5"), Some("5")]);
        // Empty batch is a no-op.
        assert_eq!(
            pointers(doc, &[], &mut bufs).unwrap(),
            Vec::<Option<&str>>::new()
        );
        // Malformed pointer syntax errors the whole call.
        assert!(pointers(doc, &["/ok", "bad"], &mut bufs).is_err());
    }

    /// A duplicate key must not re-resolve a query its first match
    /// already consumed, even when the first match was a dead end.
    #[test]
    fn batch_duplicate_keys_first_match_wins() {
        let mut bufs = Buffers::new();
        let doc = r#"{"a":8,"a":{"b":9}}"#;
        assert_eq!(pointer(doc, "/a/b", &mut bufs), Ok(None));
        assert_eq!(
            pointers(doc, &["/a/b", "/a"], &mut bufs).unwrap(),
            [None, Some("8")]
        );
    }

    /// Documented asymmetry: on malformed documents the batch resolver
    /// may error where single resolution misses, because one pass scans
    /// every byte some pointer needs.
    #[test]
    fn batch_validates_what_it_scans() {
        let mut bufs = Buffers::new();
        assert_eq!(pointer("-", "/", &mut bufs), Ok(None));
        assert!(pointers("-", &["/"], &mut bufs).is_err());
    }

    /// The matched slice re-parses to the same thing a full walk sees.
    #[test]
    fn slice_reparses() {
        let doc = r#"{"list":[{"n":1.5},{"n":"two"},{"n":[3]}]}"#;
        for (ptr, expected) in [
            ("/list/0", r#"{"n":1.5}"#),
            ("/list/1/n", "\"two\""),
            ("/list/2/n", "[3]"),
        ] {
            let raw = get(doc, ptr).unwrap();
            assert_eq!(raw, expected);
            let mut bufs = Buffers::new();
            let mut p = crate::Reader::new(raw, &mut bufs);
            p.skip_value().unwrap();
            p.finish().unwrap();
        }
    }
}
