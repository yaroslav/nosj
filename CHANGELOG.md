## [0.2.0] - 2026-07-17

- New escape mode `EscapeMode::HtmlSafe`: additionally escapes `<`,
  `>`, `&` (as their `\uXXXX` escapes) and U+2028/U+2029, producing
  output safe to interpolate into HTML documents—the profile
  HTML-embedding frameworks apply to JSON. Fully fused into the escape
  kernels (SWAR, NEON, SSE2/AVX2, and the undispatched AVX-512
  primitives), so HTML-safe emission costs the same single pass as any
  other mode. `/` is not escaped in this mode; other `0xE2`-led
  sequences (em-dashes, curly quotes) pass through untouched.
- The two partial profiles ship as modes too: `EscapeMode::HtmlEntities`
  (only `<`, `>`, `&`) and `EscapeMode::JsSeparators` (only
  U+2028/U+2029), so hosts toggling the two halves independently never
  need a post-pass.
- New `emit::copy_short_raw`: the raw-pointer counterpart of
  `emit::push_short` (overlapping word stores for `n <= 32`, `memcpy`
  beyond), for hosts writing into their own reservations.

## [0.1.1] - 2026-07-16

Performance, measured on real x86 silicon (AWS EC2 c7a.2xlarge, AMD
EPYC 9R14, Zen 4)—the first hardware validation of the runtime-detected
AVX2 paths (all tests and fuzz targets pass unchanged):

- Escape emission now routes strings shorter than 16 bytes to the
  16-byte SSE2 loop instead of the AVX2 masked-tail path: the wide
  path's entry overhead made 10-11 byte escapes (object keys
  especially) ~32% slower. Long and unicode-heavy strings keep the
  AVX2 kernel and its 2.2-2.5x advantage.
- The Grisu2 power-of-ten multiply uses a single 128-bit widening
  multiply instead of the C source's four 32-bit partial products
  (bit-identical rounding, pinned by the fpconv byte tests):
  `write_f64` ~10% faster.
- Float formatting writes through a raw cursor over one 32-byte
  reservation (the C sources' `char*` writer shape) instead of
  per-write `Vec` operations: the small variable-length slice copies
  compiled to libc memmove calls that measured 42% of float-heavy
  generation. Identical bytes; `write_f64` 219→188ns per 5 mixed
  floats end to end.
- AVX-512BW scan primitives (64-byte hit-mask and copy-scan) land
  undispatched: escape *emission* rejected them on corpus evidence
  (every escape restarts the wide loop), but scan-only consumers can
  pick them up.
- New `emit::EmitBuf` trait: the emission kernels (`escape_into`,
  `write_i64`, `write_f64`, `push_short`) are now generic over a
  raw-capacity byte sink with `Vec<u8>`'s contract, so hosts can point
  them at foreign buffers. `Vec<u8>` implements it; existing callers
  compile unchanged. The escape slow path now publishes its
  speculative prefix before any mid-kernel reserve, making the
  growth-preservation contract explicit.
- New `emit::write_f64_raw` / `emit::write_i64_raw` (with
  `F64_MAX_LEN` / `I64_MAX_LEN`): the number writers through a raw
  pointer, the C calling convention, for hosts that batch numeric runs
  under one reservation and keep the write cursor in a register.
  Measured in the Ruby host on Zen 4: flipped every float-dominated
  generation benchmark from behind to ahead of the C reference.
- Integers are written backward in place from a precomputed digit
  count (two-digit table), replacing the itoa dependency: the staging
  buffer plus copy-out measured ~25% slower per integer on int-dense
  generation. One dependency fewer; bytes identical, pinned by a
  digit-boundary test.

## [0.1.0] - 2026-07-15

Initial release.

- **Parse**—three interfaces over one set of SIMD tokenizers:
  - `parse` (push, fused single pass)—the fastest path for
    building a full value tree; `ParseOptions` grammar extensions.
  - `parse_indexed` (push)—SIMD stage-1 structural indexing exposed
    separately (`Buffers::preindex`) so hosts with a runtime lock can
    index outside it.
  - `Reader` (pull)—grammar-enforcing navigation with `skip_value`
    subtree skipping.
  - `nosj::pointer`—RFC 6901 JSON Pointer resolution as a forward
    cursor; query cost is proportional to skip distance, not document
    size. `nosj::pointers` resolves a pointer set in one forward pass
    (a batch costs about its single deepest query). Container skipping
    dispatches NEON / SSE2 / runtime-detected AVX2 like indexing.
- **Generate**—`Writer`, a streaming infallible push writer over
  SIMD escape kernels (`emit`), with compact-through-custom-pretty
  layout, three escape modes, and pinned fpconv float bytes
  (opt-in shortest-round-trip via the `shortest-floats` feature).
- SIMD: NEON on aarch64; SSE2 baseline plus runtime-detected AVX2 on
  x86-64; SWAR and table fallbacks elsewhere. No overreads anywhere —
  sanitizer-instrumented fuzzing exercises the shipped kernels.
- Correctness: correctly rounded float parsing (Clinger fast path,
  Eisel-Lemire, fast-float2 fallback); WTF-8 delivery for lone low
  surrogates; `-0` preserved as a distinct event; four differential
  fuzz targets in CI.
