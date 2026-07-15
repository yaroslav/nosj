# nosj—design document

This document explains how the crate works, top to bottom: what it is
for, how the pieces fit, how each pipeline stage operates at the byte
level, why the unusual decisions are deliberate, and how correctness and
performance are enforced. It assumes general systems-programming
knowledge but no prior exposure to SIMD JSON parsers.

Reading order for newcomers: sections 1–4 give the architecture; 5–9
walk the parse pipeline; 10–12 the generation pipeline; 13–16 the
safety model, contracts, and verification.

> [!WARNING]
> This crate was produced entirely by **Claude Fable 5**.
> It is extensively verified: differential fuzzing against serde_json,
> sanitizer-instrumented kernel tests, byte-exact cross-checks of every
> SIMD path against a scalar model, and continuous byte-for-byte
> output-parity gates in its host application.

---

## 1. Purpose and thesis

nosj is a JSON **parsing and generation** library for *hosts that build
their own values*: language runtimes, database engines, columnar
loaders—any consumer with a native value representation of its own.

The thesis: for such consumers, the classic library-provided document
tree (a DOM, a tape, a `serde` model) is pure overhead. The document
gets materialized once in the library's representation and then
converted—touched twice, allocated twice. The fastest tree is the one
built **directly in the host's representation, during the parse**. So
nosj's job is narrow and deep:

- **Parse:** validate JSON and deliver *events and scalars* (already
  decoded: `i64`, `f64`, `&str` slices) to the host as fast as
  possible, without intermediate copies of the input.
- **Generate:** accept *events* from the host and produce JSON bytes,
  again without intermediate representations.

Everything else follows from that: there is no `Value` type in the
public API, no allocation the host didn't ask for, and the SIMD
machinery exists to make the per-event overhead small enough that the
host's own value construction dominates.

Non-goals: schema validation, serde integration, incremental/streaming
input (a document is one contiguous buffer), and any host-specific
functionality—the crate is language-agnostic by policy.

## 2. The four interfaces

One interface per driving pattern, all sharing the same underlying
tokenizers:

| Direction | Host-driven | Library-driven |
|-----------|-------------|----------------|
| Parse     | **pull**—`Reader` | **push**—`Sink` via `parse` / `parse_indexed` |
| Generate  | **write**—`Writer` |—|

- **`parse` (push, fused):** a single-pass byte cursor in the
  architectural lineage of yyjson. The fastest path for consuming a
  whole document. This is the default recommendation.
- **`parse_indexed` (push):** SIMD structural indexing first (the
  simdjson lineage), then one dispatch per token. Slightly slower than
  the cursor on most documents, but the indexing phase is exposed
  separately (`Buffers::preindex` +
  `parse_preindexed_utf8_unchecked`), so hosts whose runtime has a
  global lock can run the pure-computation phase *outside* the lock
  and only hold it while values are constructed.
- **`Reader` (pull):** grammar-enforcing `next_node()` /
  `object_next_key()` navigation, for selective or partial
  consumption where building everything would be wasteful.
  `skip_value()` steps over whole subtrees without parsing them.
  `nosj::pointer` resolves RFC 6901 JSON Pointers with a forward byte
  cursor that never indexes at all (§8).
- **`Writer` (write):** a streaming, infallible push writer. The host
  emits events (`begin_object`, `key`, `int`, …); the writer owns all
  grammar state—separators, nesting depth, layout, escaping.

Each parse entry point comes in tiers: a safe `&str` version
(`parse`), a `_with` variant taking `ParseOptions` (grammar
extensions: `NaN`/`Infinity` literals, trailing commas—all checked on
cold paths only), and `_utf8_unchecked` variants for hosts whose
runtime already tracks string validity and should not pay a second
UTF-8 validation.

## 3. Data flow

```
PARSE (push, fused)                    PARSE (push, indexed)
===================                    =====================
input bytes                            input bytes
     |                                      |
     v                                      v
 parse ──────────┐              stage1::index          (pass 1)
     |                  |                   |  64B blocks -> Vec<u32> offsets
     | tokenizes as it  |                   v
     | goes, one pass   |              driver::drive_inner    (pass 2)
     v                  |                   |  one dispatch per token
 scalars::{parse_string,|                   v
  parse_number, ...}    |              same scalar tokenizers
     |                  |                   |
     v                  v                   v
          Sink events (null/bool/int/float/str/key/
          begin_*/end_*(mark,count)/negative_zero/str_bytes)
                          |
                          v
              host builds its native values

GENERATE
========
host walks its values
     |
     v
 Writer events (key/int/float/str/begin_*/end_*)
     |            grammar state: depth, separators, layout
     v
 emit kernels: escape_into (fused SIMD scan+store),
               write_i64 (itoa), write_f64 (grisu2/fpconv)
     |
     v
 output Vec<u8>
```

`Buffers` is the reusable scratch state a host keeps between parses
(three vectors, grown once and reused): `indexes: Vec<u32>` (stage-1
output), `scratch: Vec<u8>` (decoded strings that contain escapes),
`frames: Vec<Frame>` (container stack).

## 4. Module map

Dependency arrows point downward; nothing depends upward.

| Module (lines) | Responsibility |
|---|---|
| `lib.rs` (107) | Docs, re-exports. Public surface: `Reader`, `Buffers`, `Node`, `Sink`, `parse*`, `pointer`, `Writer`, `WriteOptions`, `EscapeMode`, `FloatFormat`, errors, plus `scalars`/`emit` as public modules for bespoke hosts. `stage1` is deliberately private: its public value (indexing outside a runtime lock) is fully covered by `Buffers::preindex`, and the raw `u32` index encoding stays evolvable. |
| `reader.rs` (427) | `Buffers` + the pull `Reader` (grammar-enforcing state machine over the same tokenizers). |
| `cursor.rs` (679) | The fused single-pass push driver (`parse*`), flat-container fast loops, `ParseOptions`. |
| `driver.rs` (670) | The `Sink` trait; the indexed push driver (`parse*`)—a goto-style stage-2 state machine over the stage-1 index. |
| `stage1.rs` (779) | Structural indexing: 64-byte-block SIMD classification producing token-start offsets. Per-ISA modules (NEON / SSE2 / AVX2) + scalar reference. |
| `scalars.rs` (1008) | Token-level parsing: strings (zero-copy fast path + escape decoding), numbers (fused validate+accumulate, `i64`/`-0`/`f64`/bignum split), literals. |
| `float.rs` (176) | Decimal→double: Eisel-Lemire with the Clinger fast path. |
| `el_table.rs` (657) | Generated 128-bit truncated powers of five for Eisel-Lemire (`examples/gen_el_table.rs` regenerates it byte-identically). |
| `scan.rs` (413) | The shared byte-scanning vocabulary: needs-attention predicates per ISA, fused copy+scan step, backwards tail finder, small-copy/padded-staging helpers, class tables, SWAR fallback. |
| `emit.rs` (706) | Emission kernels: fused SIMD escaping, escape tables, integer writing, float writing entry. |
| `writer.rs` (547) | The `Writer`: grammar state and layout over the emit kernels. |
| `grisu2.rs` (371) | Float formatting: line-faithful port of fpconv (Grisu2), the crate's pinned output format. |

The layering rule that keeps this understandable: **`scan.rs` owns
every byte-level primitive, defined once per architecture**; `scalars`
and `emit` compose those primitives into token-level operations;
`cursor`/`driver`/`parser`/`writer` compose *those* into document-level
state machines. When parser and emitter need the same trick (they scan
for the same `"`, `\`, control-character set), it exists exactly once.

## 5. Stage 1: structural indexing (`stage1.rs`)

Used by the indexed push path only. Input is processed in 64-byte
blocks; each block yields four 64-bit masks (one bit per byte):
backslash, quote, structural operator (`{}[]:,`), whitespace.

- **Classification** is simdjson's nibble-table trick: two 16-entry
  tables indexed by a byte's low and high nibble, ANDed —
  `lo[b & 0xF] & hi[b >> 4]`—give per-byte class bits with two
  shuffles instead of ten compares. NEON classifies four 16-byte
  vectors per block (`movemask64` builds the 64-bit mask with `vpaddq`
  chains, since NEON has no movemask); SSE2 does the same with native
  movemasks; AVX2 halves the loads with 32-byte vectors.
- **String interiors** must be excluded from structural detection.
  Escaped quotes are removed from the quote mask by the classic
  odd-length-backslash-run computation; then the *in-string* regions
  are derived with a carry-less multiply: `prefix_xor(quote_mask)`
  turns quote boundaries into a fill (PMULL with an all-ones operand
  on aarch64—one instruction; six shift-XORs elsewhere). Carries
  (`Carries`) thread backslash/in-string state across blocks.
- **`flatten_bits`** converts the surviving structural mask into
  offsets appended to `Vec<u32>`, writing 8 candidate entries per
  iteration into reserved slack and advancing by `count_ones()` —
  branch-free with respect to the bit pattern. Callers reserve
  `input.len() + 8` entries once.
- The **ragged final block** is staged in a zero-padded 64-byte stack
  buffer—the same no-overread idiom used everywhere (§13).
- **Encodings:** ordinary entries are plain `u32` offsets (documents up
  to `INDEX_MAX_LEN` = 4 GiB − 1, enforced by a real assert);
  `index_packed` stores `(offset << 8) | token_byte` so stage 2 reads
  the offset and the byte in one load (24-bit offsets,
  `PACKED_INDEX_MAX_LEN`). The packed variant is kept and tested but
  not default-wired: on Apple Silicon the pack step costs more in
  stage 1 than the fused load saves in stage 2 (the driver's byte
  loads are L1-resident). Documents beyond the limits belong to the
  fused cursor, whose positions are `usize`.

## 6. Stage 2: the indexed driver (`driver.rs`)

A goto-style state machine (the shape of simdjson's stage 2) that walks
the offset index: `take!` reads the next token offset and byte, and the
machine dispatches once per token—object/array open/close, key,
colon, comma, or a scalar handed to `scalars::*`. Container bookkeeping
lives in explicit heap `Frame`s (`Root` / `Object{mark,cnt}` /
`Array{mark,cnt}`), so nesting depth is bounded by memory, not the call
stack; hosts wanting a nesting limit enforce it in their `Sink`.

## 7. The fused cursor (`cursor.rs`)

The single-pass driver: no index, one left-to-right scan where
tokenization and dispatch are the same loop. Two properties make it the
fastest tree-building path:

- **Flat-container fast loops.** Once inside an array or object, the
  cursor stays in a tight loop consuming `element, separator` pairs
  with the current container's registers (`mark`, `cnt`) in locals —
  no frame traffic at all. Only when an element is itself a container
  does the current state spill to the frame stack. Arrays of scalars —
  the bulk of real documents—never touch it.
- **One-pass scalars.** The cursor calls the same `scalars` tokenizers
  at the byte where a value starts; there is no separate "find the
  token boundaries" pass to re-synchronize with.

Long arrays additionally call `Sink::array_checkpoint` every 256
elements so eager hosts can spill accepted elements and keep their
pending state O(1).

`ParseOptions` extensions are checked only on paths that are already
cold (a token that failed the standard match), so the default grammar
pays nothing for their existence.

## 8. The pull parser and partial access (`reader.rs`, `pointer.rs`)

`Reader` wraps the same tokenizers in a grammar-enforcing pull API:
`next_node()` returns `Node` (`ObjectStart`, `Int(i64)`, `Str(&str)`,
…), with `object_first_key`/`object_next_key`/`array_first`/
`array_next` navigating structure. It exists for selective consumption;
it is not the fast path for whole documents (the borrow discipline of
handing out `&str` per call costs it the fused loop). The pull API
folds the integer literal `-0` to `Int(0)`; hosts that must preserve
the IEEE sign use the push API (§9, `negative_zero`).

The parser is index-based (`Reader::new` runs stage 1 and walks
`Buffers::indexes`), which makes **`skip_value`** cheap in a way a byte
cursor could not be: skipping a container is bracket counting over
`u32` index entries—the bytes of the skipped subtree are never
re-read (strings were already quote-resolved by stage 1; numbers are
just stepped-over entries). The skipped value's raw text comes back as
a slice for lazy use. The semantics are deliberate and documented:
scalar values at the *target* are fully validated (their tokenizer runs
to find the token end), while the interior of a skipped container is
only structurally validated (bracket balance—not bracket kinds, not
scalar contents), so a later full parse may reject what a skip stepped
over.

**`nosj::pointer`** (RFC 6901 JSON Pointer, `pointer.rs`) is a forward
*byte cursor*, deliberately not built on the index: it tokenizes only
the navigation levels it walks (keys compare zero-copy when unescaped)
and steps over sibling values. Sibling containers are skipped by
`stage1::container_end`: 64-byte blocks classified with the same
nibble-table + quote-parity machinery as indexing
(`Carries::block_ops`—the string-filtered bracket mask), consuming
only bracket bits, with
no index writes, stopping at the matching closer. A query therefore
costs what it skips, not what the document weighs: an early target
resolves in fractions of a microsecond, and long skips run at block
speed—measured 4.1x faster than sonic-rs's byte-level skip-scan on
twitter's last status (104.9 vs 427.8µs) and 3.3x on a citm mid-file
target, with early targets tied at 0.2µs. The skip classifier
dispatches like indexing—NEON on aarch64, SSE2 baseline with
runtime-detected AVX2 on x86-64 (two 32-byte loads and lane-broadcast
nibble tables per block; whitespace bits skipped since `block_ops`
never reads them)—QEMU-TCG directional: AVX2 skip 1.7-1.8x over
SSE2, pending real-silicon validation.

Miss semantics track `serde_json::Value::pointer` (missing key,
out-of-range or malformed index token → `Ok(None)`); only malformed
pointer *syntax* is an error (`ErrorKind::InvalidPointer`); duplicate
keys resolve to the first occurrence (streaming order); the resolved
target is fully validated while skipped container interiors are
bracket-balance-checked only.

**`nosj::pointers`** batches N pointers into **one** forward pass. Each
pointer is pre-tokenized (unescaped once, array-index syntax
precomputed) into a `Query`; at each container the resolver partitions
the live queries—matches here are recorded, deeper matches advance by
one token and recurse, everything no query names is skipped by
`value_end`. Array walks stop tokenizing past the largest queried
index; the whole walk aborts as soon as every query is resolved. A
batch therefore costs about what its single deepest query costs
(measured: 5 twitter fields batch 109µs vs 348µs sequential vs 436µs
sonic `get_many`). Two semantics pinned by fuzzing
(`pointer_differential` asserts batch ≡ single on valid documents):
a query is *consumed* by the first member whose key matches—hit or
dead-end—so a later duplicate key can never re-resolve it; and on
malformed documents the batch may error where a lone query would miss,
because one pass scans every byte *some* pointer needs.

## 9. Scalar tokenizers (`scalars.rs`, `float.rs`)

### Strings

`parse_string(input, quote_idx, scratch)` returns `StrPart`:

- **`Borrowed(&str)`—the zero-copy fast path.** `find_special` scans
  for the first `"`, `\`, or control byte. If it is the closing quote,
  the string's bytes are handed out as a slice of the input—no copy,
  no allocation. Most strings in real documents take this path.
  `find_special`'s shape is tuned for it: a single 16-byte probe first
  (most keys and short values resolve in one vector), then a 32-byte
  loop whose early-exit is a single instruction (`vmaxvq` on the OR of
  two hit vectors—no mask extraction until something hits), then the
  backwards in-bounds tail (§13).
- **`Decoded(&str)`—the escape path.** On the first `\`, decoding
  switches to the scratch buffer. The *shrink invariant* makes this
  allocation-free after one `reserve`: every escape sequence decodes
  to fewer bytes than its source (`\n` 2→1, `\uXXXX` 6→≤3, surrogate
  pairs 12→4), so `scratch.reserve(remaining + DECODE_SLACK)` once is
  enough for the whole string; all subsequent writes are raw stores
  against that capacity. Clean spans between escapes are moved by
  `decode_span`—a fused copy+scan (§13's `copy_scan` step) that
  copies 16 bytes per iteration *while* checking them, so clean data
  is touched exactly once. `DECODE_SLACK` (16) absorbs the full-width
  final store.
- **`DecodedRaw(&[u8])`—the WTF-8 case.** A lone *low* surrogate escape
  (`"\udc00"`) decodes to its raw WTF-8 bytes, delivered through
  `Sink::str_bytes` because the result is not valid UTF-8. This
  matches widely deployed lenient parsers. A lone *high* surrogate is
  an error. Hosts that don't care get a lossy-converted `&str` from
  the default trait method.

### Numbers

`parse_number` is one fused pass: the same loop that validates the
digit grammar accumulates the mantissa (`scan_digits_acc`, an
8-bytes-at-a-time SWAR digit check with wrapping accumulation—the
value is only *used* when the digit count proves it didn't wrap).
The result is split by representation:

- `Number::Int(i64)`—integers that fit, the common case.
- `Number::NegativeZero`—the literal `-0`, kept distinct because hosts
  disagree about it: integer-preserving hosts want `0`,
  IEEE-preserving hosts want `-0.0`. The `Sink::negative_zero` event
  defaults to `int(0)`, so only hosts that care override it.
- `Number::Big(&str)`—integers past `i64`, delivered as their
  validated ASCII digits (`Sink::big_int`); the host picks its bignum.
- `Number::Float(f64)`—everything else, correctly rounded, built
  from the already-scanned parts (no re-parse of the text):
  1. **Clinger fast path:** if the mantissa fits 2⁵³ and the exponent
     is small, one float multiply/divide is exactly rounded.
  2. **Eisel-Lemire** (`float.rs`): the 128-bit
     mantissa × truncated-power-of-five multiply-reduce; the table
     (`el_table.rs`) is generated by `examples/gen_el_table.rs` (a
     self-contained bignum in ~150 lines, so regeneration needs
     nothing but this crate's toolchain).
  3. The rare inputs Eisel-Lemire cannot decide fall back to
     `fast-float2`'s slow path.

  Exponent parsing saturates *huge* exponents to ±∞/0—counting only
  significant digits, because a fuzzer proved leading zeros
  (`1e0000000000000000000001`) must not trip the saturation.

### The Sink contract

Events arrive in document order. `mark()` is called at container start
and handed back to the matching `end_array(mark, len)` /
`end_object(mark, pairs)`—a sink keeping one flat value stack can
slice off exactly the container's children with no per-container
allocation of its own. All string/key payloads are valid only during
the call; the host copies (or interns) what it keeps.

## 10. Emission kernels (`emit.rs`)

### Escape modes

`EscapeMode`: `Standard` (RFC 8259 minimum: `"`, `\`, controls),
`ScriptSafe` (adds `/` and U+2028/U+2029 for `<script>` embedding),
`AsciiOnly` (adds all non-ASCII as `\uXXXX`, surrogate pairs for astral
code points). Internally the enum is lowered to `MODE_*` `u8`
const-generics—stable Rust cannot use enums as const parameters—and
one exhaustive `match` in `escape_into` keeps them in lockstep.

### The fused scan+store loop (`fused_scan_store!`)

The central emission idea: while scanning a string for bytes that need
escaping, each just-scanned chunk is **speculatively stored** into the
output's reserved capacity. Clean text—the overwhelming majority—is
therefore touched once (scan-then-memcpy would touch it twice). The
single macro is instantiated per width: NEON 16B (4 mask bits/lane via
the `vshrn` movemask substitute), SSE2 16B, AVX2 32B (runtime-detected;
never compile-flagged). The steps themselves are `scan::copy_scan`.

Capacity discipline makes every store check-free:

- Entry reserves `len + width`—enough for all speculative chunk
  stores of a clean string.
- The **first escape** triggers one worst-case reservation:
  `MAX_ESCAPE_EXPANSION (6) × remaining + width + 8`. After that, no
  store on any path needs a capacity check: 6 covers `\u00XX`, and an
  AsciiOnly astral pair (12 bytes from 4) averages 3.
- `set_len` only ever covers validated bytes.

Escape handling is split by temperature: the *first* escape (and all
mode-specific ones) take a `#[cold]` out-of-line function
(`escape_run_slow_path`)—inlining it into the hot loop measurably
cost clean short strings 4×. Once the worst-case reservation exists,
the always-escaped classes (`"`, `\`, controls) are handled *inline* at
the hit site: `"`/`\` by two direct byte stores (a lookup table lost to
these branches), controls by one unconditional 8-byte copy from the
padded `ESC_SEQ` table with the cursor advanced by the true length
(`ESC_LEN`)—that padding is why the reservation carries `+ 8`.
Escape *runs* loop without re-entering SIMD only in AsciiOnly mode,
where escapes really do arrive in runs (+45% there, −5% elsewhere —
so the other modes don't pay for it).

`push_short` appends ≤16-byte slices as a pair of overlapping word
stores (`scan::copy_small`) instead of a `memcpy` call—tiny-copy call
overhead was 14% of a string-heavy generation profile. Longer input
falls back to `extend_from_slice`, keeping the public function total.

Inlining is load-bearing here: `escape_fused_16` and the step kernels
are `#[inline(always)]` because as ordinary calls they inlined in the
hot loop but *not* at the tail call site (AVX2 variants use plain
`#[inline]`—rustc rejects `always` with `target_feature`, and
same-feature callers still inline).

### Numbers out

`write_i64` = itoa into a stack buffer + `push_short`. `write_f64` is
the pinned float format (§12).

## 11. The Writer (`writer.rs`)

A thin grammar-state layer over the emit kernels: `depth`, `first`
(current container has no members yet), `after_key` (suppress the next
value prefix). `WriteOptions` holds layout byte-strings (`indent`,
`space`, `space_before`, `object_nl`, `array_nl`) plus the escape mode
and float format; a single `compact` flag (all layout fields empty)
lets compact output skip every layout branch predictably.

The writer is **infallible and non-validating**: it emits exactly the
event sequence it is given; misuse produces invalid JSON, not UB or a
panic—unmatched root closers saturate the depth rather than
underflowing it into the indent loop. The one enforced precondition is
float finiteness: `Writer::float` panics on NaN/∞ in every build,
because JSON cannot represent them and the digit generator would
otherwise silently emit an unrelated finite number. Hosts with a policy
(e.g. an allow-NaN mode) splice literals through `value_raw`, which is
also the escape hatch for bignum digits and embedded fragments.
`emit::write_f64` remains the documented *trusted* tier for hosts that
gate non-finite values themselves.

## 12. Float output: the fpconv contract (`grisu2.rs`)

Float formatting is a **line-faithful port of fpconv** (Grisu2,
night-shift/fpconv, BSL-1.0—see `LICENSE-BSL-1.0`), including the
`.0`-suffix-for-integral-values behavior. This is a compatibility
contract, not an implementation detail: Grisu2 *without a fallback* is
not guaranteed shortest—it emits e.g. `2.3438720703100002` where the
shortest round-trip is `2.34387207031`—and downstream ecosystems have
those exact bytes baked into tests and stored data. A shortest
formatter (ryu, zmij) **cannot reproduce them**, which is why one is
not used. The port mirrors the C source arithmetic step for step,
including its unsigned wrapping comparisons, and is pinned by byte
tests.

For hosts without that legacy, the opt-in `shortest-floats` feature
adds `FloatFormat::Shortest` (zmij/Schubfach)—roughly 2× faster on
float-dominated documents, different bytes. The default never changes.

## 13. Memory-safety architecture

The crate contains ~70 `unsafe` blocks, every one carrying a `SAFETY:`
proof (enforced by `clippy::undocumented_unsafe_blocks`). They fall
into three tiers:

1. **Public contract tier**—`unsafe fn`s whose obligation is the
   *caller's*: the `*_utf8_unchecked` entry points (host runtime
   vouches for UTF-8), `scalars::parse_string` (valid UTF-8 + a real
   quote index), `Writer::str_utf8_unchecked`. Each documents its
   contract under `# Safety`; internal callers discharge it with a
   proof comment at the call site.
2. **Capacity contract tier**—raw stores into `Vec` spare capacity,
   made check-free by named, reserve-once invariants:
   `MAX_ESCAPE_EXPANSION` (§10), `DECODE_SLACK` + the shrink invariant
   (§9), stage 1's `len + 8` index reservation. The pattern is always:
   one `reserve` whose arithmetic is stated where it happens, raw
   writes against it, one `set_len` covering only validated bytes.
3. **Register tier**—SIMD intrinsics on already-loaded vectors;
   trivially sound, still documented (NEON/SSE2 are baseline on their
   targets; AVX2 is runtime-detected at every dispatch site).

**No overreads, by construction.** Reading past a slice's end is
undefined behavior in Rust's allocation model even when it cannot fault
at the hardware level, so the classic page-guarded overreading tail is
banned. Two sound tail shapes replace it, chosen by call shape:

- **Scan-only tails** (`scan::tail_find`): re-load the buffer's *last*
  full vector—entirely in bounds, overlapping bytes already scanned
  clean—and drop mask lanes before the cursor, so a stale hit in the
  overlap can never surface.
- **Fused scan+store tails** (`scan::padded_tail`): stage the
  remainder in a zero-padded stack chunk and run the normal full-width
  step against it. Padding zeros classify as control characters in
  every mode and are masked out by the remainder truncation the tails
  already perform. A backwards load is *not* usable here: after an
  escape, source and output offsets diverge, and a backwards store
  would overwrite expanded output.

Because there is no overread anywhere, sanitizer-instrumented fuzzing
exercises exactly the code that ships—there is no `cfg(fuzzing)`
substitution.

## 14. Deliberate contracts (do not "fix")

Distilled; each is documented at its definition site:

- fpconv float bytes, not shortest (§12).
- Lone low surrogates → WTF-8 via `str_bytes`; lone high → error (§9).
- `-0` is its own event with an integer-folding default (§9).
- Huge exponents saturate to ±∞/0 by *significant* digit count (§9).
- Nesting is unlimited in the library (heap frames); limits are host
  policy, enforceable in `Sink` callbacks or via `Writer::depth`.
- The writer validates nothing except float finiteness (§11).
- `EscapeMode` lowers to `MODE_*` `u8` const-generics until
  `adt_const_params` stabilizes (§10).
- Indexed-path documents are capped at `u32`/24-bit offsets with real
  asserts; the cursor has no such limit (§5).

## 15. Performance principles

- **x86 (Intel and AMD) is the production target**; AVX2 is assumed
  present in production but always runtime-detected, never
  compile-flagged. aarch64/NEON is fully supported; a SWAR + class
  table implementation covers everything else.
- **Touch data once.** The fused scan+store escape loop, the fused
  decode copy+scan, and the fused digit validate+accumulate all exist
  to collapse "check it, then move it" into one pass.
- **Reserve once, then write raw.** Capacity checks in per-byte loops
  are the enemy; every hot loop's stores are pre-authorized by a
  single reservation with stated arithmetic (§13 tier 2).
- **Shape for short inputs.** Most JSON strings are short keys: the
  16-byte probe before the wide loop, `push_short`, and the
  `#[inline(always)]` tail-call discipline all serve strings under ~16
  bytes, which dominate real documents.
- **Split hot from cold explicitly.** First-escape/option/error paths
  are `#[cold]` or cold-by-construction; the clean-text loop carries
  one branch per chunk.
- **Measure, never infer.** Benchmark claims come from the alternating
  round-robin harness (`examples/compare`) over a real-document
  corpus, on a quiet machine; kernel microbenches
  (`examples/kernel_bench`) guide but never decide—sub-30ns kernels
  swing with code alignment. PGO (profile-generate → train on the
  corpus → profile-use) is part of the release gate and worth 4–30%
  per file; comparisons train *all* contenders for fairness.

## 16. Verification

Layered so each level catches what the previous one can't:

- **Unit cross-checks** pin every SIMD kernel to a scalar reference:
  SWAR vs the class table, stage 1 vector vs scalar indexing (bit
  exact), driver-vs-cursor event-trace equality, escape/string
  boundary sweeps at every length around the vector widths—run
  against *exact-size* allocations, so a reintroduced overread lands
  outside the allocation. fpconv output is pinned byte-for-byte,
  including known non-shortest cases.
- **Fuzzing** (`fuzz/`, three targets): `parse_differential` (the two
  push drivers must produce identical event streams or the same
  rejection), `roundtrip` (any tree the writer serializes, in every
  layout, reparses to the identical tree, floats bit-exact), and
  `vs_serde` (anything serde_json accepts, nosj must accept with equal
  values; serde's `float_roundtrip` feature makes bit-equality a valid
  oracle). The targets have already paid for themselves: they found
  the exponent-leading-zeros bug and forced precise `-0` and
  duplicate-key semantics.
- **CI** runs the full matrix (x86-64 with AVX2, aarch64/NEON), an
  SSE2-only job (x86 tests under Rosetta on ARM runners—Rosetta
  cannot execute AVX2, so runtime detection falls back), MSRV, docs
  with denied warnings, packaging, and short fuzz smoke runs; releases
  are manual and additionally gate on the full test suite under a
  PGO-optimized build, because that is the build mode hosts ship.

## 17. Known limitations

- Input must be one contiguous buffer; there is no incremental feed.
- The indexed path caps documents at 4 GiB − 1 (use the cursor above).
- `Reader` (pull) does not expose `-0` or options; the push API does.
- Non-x86/ARM targets run the portable SWAR/scalar paths—correct,
  tested, not fast.
