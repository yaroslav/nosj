# nosj

nosj is a SIMD-accelerated JSON parser and writer that never builds its
own values. It hands you **events**—object begins, keys, strings,
numbers—and you build your structure directly: a runtime's heap
objects, an arena, an Arrow column. Strings arrive as borrowed slices
straight from the input buffer whenever no unescaping is needed.

**Why:** if the values you need aren't Rust values, every DOM parser
makes you pay twice—once to build its tree, once to convert every node
into yours. That second pass re-touches every string, allocates
everything again, and routinely costs more than the fast parse saved.
nosj deletes the intermediate tree in both directions: the streaming
`Writer` takes events from your own traversal, so generation never
materializes a Rust-side copy either.

**And when you only need a few fields**, don't parse at all: the JSON
Pointer resolver takes a path like `/users/1/name` and walks the raw
bytes straight to it, skipping every sibling at 64-byte-block speed and
returning the value's raw text. A query costs what it skips, not what
the document weighs—a field deep in a 600 KB document resolves in
~100µs where parse-then-navigate costs milliseconds—and a whole set
of pointers resolves in one pass for about the price of the deepest
one.

**How it differs:** serde_json, simd-json, and sonic-rs produce *their*
representation (a `Value`, a tape, a lazy document). nosj produces
none—you get a push `Sink`, a pull `Reader`, a streaming `Writer`, and
raw-text JSON Pointer resolution. If you want a `Value` tree in Rust,
use serde_json; if you want your own values at SIMD speed, that is the
entire point of nosj.

> [!WARNING]
> This crate was produced entirely by **Claude Fable 5**.
> It is extensively verified: differential fuzzing against serde_json,
> sanitizer-instrumented kernel tests, byte-exact cross-checks of every
> SIMD path against a scalar model, and continuous byte-for-byte
> output-parity gates in its host application.

## Show me

```rust
use nosj::{Buffers, Node, Reader, WriteOptions, Writer};

// Parse (pull): ask for events, build whatever you want.
let mut bufs = Buffers::new();
let mut p = Reader::new(r#"{"a": [1, true]}"#, &mut bufs);
assert!(matches!(p.next_node().unwrap(), Node::ObjectStart));
assert_eq!(p.object_first_key().unwrap(), Some("a"));

// Generate (write): emit events, the writer owns the syntax.
let mut out = Vec::new();
let mut w = Writer::new(&mut out, &WriteOptions::COMPACT);
w.begin_object();
w.key("a");
w.begin_array();
w.int(1);
w.boolean(true);
w.end_array();
w.end_object();
assert_eq!(out, br#"{"a":[1,true]}"#);
```

Grab one field out of a big document without parsing the rest:

```rust
let mut bufs = nosj::Buffers::new();
let doc = r#"{"users":[{"name":"ada"},{"name":"grace"}]}"#;
let raw = nosj::pointer(doc, "/users/1/name", &mut bufs).unwrap();
assert_eq!(raw, Some("\"grace\""));
```

## What's in the box

| Direction | Host-driven | Library-driven |
|-----------|-------------|----------------|
| Parse     | **pull**—`Reader` | **push**—`Sink` via `parse` / `parse_indexed` |
| Generate  | **write**—`Writer` |—|
| Partial   | **pointer**—`pointer` / `pointers` |—|

**The parser**: strict RFC 8259 (opt-in `NaN`/`Infinity` and trailing
commas on cold paths), one set of SIMD tokenizers, three ways to drive
it:

- `parse`—fused single-pass byte cursor (the yyjson architecture),
  pushing events into your `Sink`. The default and fastest way to build
  a complete structure; flat containers run in tight scalar loops with
  zero frame-stack traffic.
- `parse_indexed`—SIMD stage-1 pass indexes every token boundary, then
  one dispatch per token into the same `Sink`. The indexing phase is
  pure computation, exposed separately (`Buffers::preindex`) so hosts
  with a runtime lock (a GVL/GIL) can run it outside the lock.
- `Reader`—pull: ask for the next event, navigate objects key by key,
  `skip_value()` over subtrees you don't care about, grammar enforced
  as you walk.

**The writer** (`Writer`): streaming push writer—you emit events, it
owns separators, nesting, layout (compact through fully custom pretty),
and escaping, and it cannot fail (no `Result` on the hot path). Number
bytes are pinned (itoa integers, fpconv floats) with an opt-in
shortest-round-trip float mode.

**The partial parser** (`pointer`, `pointers`): resolves JSON Pointers
(`/users/1/name`, with the standard `~0`/`~1` escapes) against the raw
document and returns the matched value's raw text without parsing
anything else. Siblings are skipped at 64-byte block speed, so a query
costs what it skips, not what the document weighs; `pointers` resolves
a whole set of paths in one forward pass for about the price of its
deepest member. Misses are `Ok(None)` (`serde_json::Value::pointer`
semantics); skipped interiors are bracket-balance-checked, the resolved
target fully validated.

µs per query, Apple M-series (`examples/pointer_bench`):

| shape | nosj pointer | sonic-rs get | serde parse+pointer |
|---|---:|---:|---:|
| twitter early (`/statuses/0/id`) | 0.2 | 0.2 | 1262.5 |
| twitter late (`/statuses/95/user/screen_name`) | **104.9** | 427.8 | 1263.2 |
| citm deep (`/performances/40/.../areaId`) | **69.8** | 228.1 | 2481.0 |

| nosj pointers (batch of 5) | nosj pointer ×5 (sequential) | sonic-rs get_many |
|---:|---:|---:|
| **109.4** | 347.8 | 435.9 |

## Benchmarks

`cargo run --release --features shortest-floats --example compare`:
alternating round-robin blocks over `benchmark/` (the classic simdjson
corpus plus real-world payloads), MB/s, higher is better, whole-binary
PGO with every contender trained equally. Apple M-series, rustc 1.97.0,
2026-07-15; x86 numbers pending real hardware.

nosj appears more than once because it has no DOM: **events** drives
the fused cursor into a non-allocating sink (the design point);
**tree** builds a naive owned `Vec`/`String` tree so the comparison
with DOM libraries is apples-to-apples; **shortest floats** is the same
tree with the opt-in `FloatFormat::Shortest` instead of pinned fpconv
bytes. Parse events mode leads on 11 of 13 files, generation on 11 of
13 counting shortest-floats where floats dominate.

### Parse (MB/s)

| file | nosj (events) | nosj (tree) | serde_json | sonic-rs | simd-json | simd-json (borrowed) | jiter | json-rust | json-steroids | asmjson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| activitypub | 2183 | 901 | 645 | **2282** | 695 | 1039 | 1016 | 679 | 612 |—|
| canada | **1276** | 773 | 534 | 1173 | 419 | 420 | 401 | 658 | 488 | 940 |
| citm_catalog | **3173** | 979 | 765 | 2293 | 856 | 1120 | 993 | 910 | 852 | 1544 |
| gsoc-2018 | **5144** | 1920 | 1297 | 4517 | 1300 | 2038 | 2503 | 1012 | 891 |—|
| homebrew-formula | **2412** | 596 | 413 | 2123 | 445 | 709 | 1023 | 690 | 534 |—|
| homebrew-llvm | 2320 | 1721 | 1389 | **3741** | 1118 | 1292 | 1905 | 1362 | 1377 | 811 |
| mesh | **1116** | 917 | 402 | 914 | 684 | 684 | 686 | 816 | 583 | 768 |
| numbers | **1139** | 997 | 590 | 1010 | 947 | 953 | 901 | 840 | 701 | 859 |
| ohai | **1627** | 570 | 353 | 1453 | 556 | 1029 | 1028 | 564 | 540 | 1184 |
| simple | **1212** | 363 | 293 | 788 | 275 | 372 | 559 | 529 | 451 | 600 |
| small_mixed | **534** | 151 | 200 | 271 | 107 | 120 | 195 | 189 | 227 | 213 |
| tolstoy | **13670** | 11058 | 4505 | 9600 | 2084 | 2163 | 314 | 1128 | 443 | 488 |
| twitter | **2422** | 853 | 518 | 2036 | 661 | 1196 | 993 | 830 | 632 |—|

### Generate (MB/s)

| file | nosj (tree) | nosj (shortest floats) | serde_json | sonic-rs | simd-json | json-rust | json-steroids |
|---|---:|---:|---:|---:|---:|---:|---:|
| activitypub | 3712 | **3722** | 1570 | 3164 | 1904 | 1587 | 1843 |
| canada | 352 | **845** | 801 | 816 | 792 | 1616\* | 777 |
| citm_catalog | 1570 | **1574** | 1167 | 1390 | 1149 | 1269 | 1271 |
| gsoc-2018 | 6900 | **6972** | 1877 | 6565 | 2963 | 1854 | 2563 |
| homebrew-formula | 2635 | **2967** | 1017 | 2084 | 1017 | 1269 | 1384 |
| homebrew-llvm | 2590 | **2619** | 1579 | 2378 | 1433 | 1590 | 1200 |
| mesh | 394 | 597 | **598** | 562 | 499 | 933\* | 496 |
| numbers | 328 | **662** | 630 | 629 | 518 | 1218\* | 497 |
| ohai | 2033 | **2044** | 1034 | 1680 | 1100 | 1118 | 1201 |
| simple | 1114 | **1118** | 768 | 706 | 769 | 832 | 842 |
| small_mixed | 475 | **478** | 430 | 402 | 354 | 340 | 355 |
| tolstoy | 7828 | 7797 | 2402 | **13762** | 4012 | 2125 | 2877 |
| twitter | 3066 | **3073** | 1483 | 2734 | 1357 | 1576 | 1797 |

\* json-rust's float output is not byte-faithful. Blank cells: asmjson
(AVX-512-specialized, running its fallback here) failed those files.

## How it is fast

**Parsing**

- Stage 1 in the simdjson lineage: shufti nibble-table classification,
  odd/even escape carries, PMULL/CLMUL prefix-XOR quote masks, unrolled
  index flatten. NEON on aarch64; SSE2 baseline and runtime-detected AVX2 on
  x86_64; a byte-exact scalar model cross-checked bit-for-bit in tests.
- Strings: zero-copy for the escape-free case; 32-byte SIMD scans with the
  control-character check folded in, so string bytes are read once.
- Numbers: fused SWAR validation + accumulation (eight digits per step),
  Eisel-Lemire float construction verified differentially against
  fast-float2 over hundreds of thousands of cases.

**Generation**

- Fused scan+store escaping: each 16-byte chunk is speculatively stored
  into reserved output capacity as it is scanned, so clean string data is
  touched once. SWAR tail and table-driven scalar path for short strings;
  overlapping-word copies instead of `memcpy` calls for tiny spans.
- Three escape modes: standard (RFC 8259 minimum), script-safe (`/`,
  U+2028/U+2029—safe for HTML `<script>` embedding), and ASCII-only
  (`\uXXXX` with surrogate pairs).
- Integers via `itoa`; floats in the **fpconv (Grisu2) format**—a
  line-faithful port of the widely deployed `fpconv.c`, byte-for-byte
  compatible with the C reference and pinned by differential tests.
  Deliberately not Rust's shortest-round-trip `Display`: reproducing the
  reference bytes exactly is the guarantee hosts need, with
  `FloatFormat::Shortest` as the opt-in alternative.

## Where it came from

nosj was born inside a language-runtime binding that set out to beat
that runtime's built-in, heavily optimized C JSON library. Parsing
speed was never the blocker—the post-parse conversion pass was, so
this crate became the parser that skips it. Exercised end to end by
its first host, it benchmarks faster than that C implementation across
the full classic suite in both directions, with byte-for-byte output
parity verified continuously.

How everything works—architecture, pipeline internals, the memory
safety model, and why the unusual decisions are deliberate—is written
up top to bottom in [doc/DESIGN.md](doc/DESIGN.md).

## UTF-8 contract

`&str` entry points are safe. `*_utf8_unchecked` entry points skip
whole-input validation for hosts whose runtime already tracks string
validity; string content is still validated structurally during the parse.
One deliberate leniency, matching widely deployed parsers: a lone *low*
surrogate escape decodes to raw WTF-8 bytes, delivered via
`Sink::str_bytes` (lossy-converted unless overridden); a lone *high*
surrogate is an error.

## Testing and fuzzing

Unit tests cross-check every SIMD kernel against its scalar reference and
pin the two push drivers to identical event streams. Three cargo-fuzz
targets extend that continuously: driver-vs-cursor trace equality,
writer→parser round-trips across all layout configurations, and a
value-for-value differential against serde_json (with its
`float_roundtrip` feature, so both sides are correctly rounded). CI runs
each target briefly on every push.

## Installation

```sh
cargo add nosj
```

## License

MIT, with two derived components under their upstream licenses (SPDX:
`MIT AND BSL-1.0 AND Apache-2.0`):

- `src/grisu2.rs` is a line-faithful port of
  [fpconv](https://github.com/night-shift/fpconv)—Boost Software
  License 1.0 (`LICENSE-BSL-1.0`).
- The correctly-rounded float parser (`src/float.rs`,
  `src/el_table.rs`) is ported from
  [fast_float](https://github.com/fastfloat/fast_float) /
  fast_double_parser, and `src/stage1.rs` uses structural-indexing
  techniques from [simdjson](https://github.com/simdjson/simdjson)—
  Apache License 2.0 (`LICENSE-APACHE-2.0`).

Every derivative, plus architectural lineage acknowledgments, is
itemized in the `NOTICE` file.
