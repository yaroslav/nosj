## [0.1.0]

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
