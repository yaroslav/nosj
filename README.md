# nosj

**The fastest JSON parser on 11 of 13 classic-corpus files: 2-19x
serde_json, ahead of sonic-rs and simd-json, up to 12 GB/s—[see
Benchmarks](#benchmarks).**

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

µs per query, AWS EC2 c7a.2xlarge (AMD EPYC 9R14, Zen 4;
`examples/pointer_bench`):

| shape | nosj pointer | sonic-rs get | serde parse+pointer |
|---|---:|---:|---:|
| twitter early (`/statuses/0/id`) | 0.1 | 0.2 | 1723.9 |
| twitter late (`/statuses/95/user/screen_name`) | **58.5** | 220.1 | 1715.7 |
| citm deep (`/performances/40/.../areaId`) | **45.9** | 128.4 | 3875.2 |

| nosj pointers (batch of 5) | nosj pointer ×5 (sequential) | sonic-rs get_many |
|---:|---:|---:|
| **66.3** | 217.3 | 238.4 |

## Benchmarks

`cargo run --release --features shortest-floats --example compare`:
alternating round-robin blocks over `benchmark/` (the classic simdjson
corpus plus real-world payloads), MB/s, higher is better, whole-binary
PGO with every contender trained equally. AWS EC2 c7a.2xlarge
(AMD EPYC 9R14, Zen 4; AVX2 paths active), rustc 1.97.0, 2026-07-16.

nosj appears more than once because it has no DOM: **events** drives
the fused cursor into a non-allocating sink (the design point);
**tree** builds a naive owned `Vec`/`String` tree so the comparison
with DOM libraries is apples-to-apples; **shortest floats** is the same
tree with the opt-in `FloatFormat::Shortest` instead of pinned fpconv
bytes. Parse events mode is the fastest parser on 11 of 13 files
(2-19x serde_json, up to 12 GB/s on tolstoy). Generation beats
serde_json on all 13 (counting shortest-floats where floats
dominate); sonic-rs's x86 serializer takes most string-heavy files
end-to-end.

### Parse (MB/s)

| file | nosj (events) | nosj (tree) | serde_json | sonic-rs | simd-json | simd-json (borrowed) | jiter | json-rust | json-steroids | asmjson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| activitypub | **3080** | 672 | 512 | 2123 | 430 | 608 | 708 | 583 | 441 |—|
| canada | **1110** | 592 | 348 | 1014 | 348 | 350 | 346 | 566 | 364 | 959 |
| citm_catalog | **3317** | 822 | 567 | 2162 | 653 | 815 | 862 | 786 | 664 | 1974 |
| gsoc-2018 | **6666** | 1615 | 1212 | 5278 | 1183 | 2102 | 1315 | 875 | 879 |—|
| homebrew-formula | **3366** | 313 | 177 | 1654 | 166 | 289 | 422 | 382 | 246 |—|
| homebrew-llvm | 3475 | 1834 | 1350 | **4123** | 1543 | 2040 | 1427 | 1246 | 1121 | 957 |
| mesh | **1044** | 673 | 408 | 867 | 479 | 479 | 607 | 615 | 319 | 771 |
| numbers | 1294 | 978 | 595 | **1321** | 727 | 729 | 938 | 948 | 566 | 956 |
| ohai | **2229** | 433 | 408 | 1827 | 387 | 779 | 718 | 590 | 356 | 1356 |
| simple | **1246** | 377 | 367 | 726 | 253 | 357 | 508 | 498 | 414 | 670 |
| small_mixed | **455** | 153 | 145 | 198 | 105 | 111 | 171 | 183 | 153 | 213 |
| tolstoy | **12101** | 9584 | 4461 | 11202 | 3496 | 3799 | 368 | 895 | 399 | 485 |
| twitter | **3166** | 667 | 408 | 2452 | 530 | 955 | 811 | 687 | 484 |—|

### Generate (MB/s)

| file | nosj (tree) | nosj (shortest floats) | serde_json | sonic-rs | simd-json | json-rust | json-steroids |
|---|---:|---:|---:|---:|---:|---:|---:|
| activitypub | 2372 | 2353 | 1277 | **3317** | 2283 | 1350 | 2531 |
| canada | 302 | **805** | 766 | 776 | 612 | 1289\* | 613 |
| citm_catalog | 1077 | 1088 | 876 | **1138** | 1003 | 1068 | 1068 |
| gsoc-2018 | 4709 | 4745 | 1451 | **7307** | 4110 | 1528 | 4608 |
| homebrew-formula | **1288** | 1285 | 391 | 1082 | 592 | 661 | 780 |
| homebrew-llvm | 2870 | 2835 | 1265 | **2934** | 2466 | 1289 | 2595 |
| mesh | 381 | **681** | 615 | 617 | 440 | 871\* | 427 |
| numbers | 320 | **715** | 693 | 690 | 465 | 1230\* | 481 |
| ohai | 1042 | 1048 | 984 | **1574** | 1078 | 1154 | 1200 |
| simple | 668 | 671 | 667 | 658 | 756 | **888** | 722 |
| small_mixed | **345** | 341 | 339 | 319 | 330 | 334 | 264 |
| tolstoy | 9973 | 10085 | 1624 | **18947** | 7275 | 1631 | 907 |
| twitter | 1626 | 1633 | 1107 | **2374** | 1185 | 1284 | 1190 |

\* json-rust's float output is not byte-faithful. Blank cells: asmjson
(here running its native AVX-512 path) failed those files.

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
