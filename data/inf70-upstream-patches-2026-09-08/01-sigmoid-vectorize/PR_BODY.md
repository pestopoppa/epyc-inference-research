## ggml-cpu: vectorise sigmoid (`ggml_vec_sigmoid_f32` was dead code)

### Summary

`ggml_vec_sigmoid_f32` has existed in `ggml/src/ggml-cpu/vec.h` as a scalar
`for (i) y[i] = 1.f/(1.f+expf(-x[i]));` loop **with no callers**.
`ggml_compute_forward_sigmoid` never used it — it goes through `unary_op<op_sigmoid>`,
which applies `1.f/(1.f+expf(-x))` **per element via libm**, one call at a time.

The result is that `GGML_UNARY_OP_SIGMOID` is the only common activation on the CPU
backend that never got vectorised, while its immediate neighbour `ggml_vec_silu_f32`
has used the SIMD `ggml_v_expf` all along. On identically shaped `[10240,1,1]` f32
tensors, in the same graph, on the same thread, we measured (on our own fork, not
upstream) **sigmoid 50.0 us vs silu 4.0 us — a 12.5x gap**, explained entirely by 10240
scalar `expf` calls at ~4.6 ns each. This is a missing **vectorisation**, not a missing
parallelisation (see the separate (row, column-chunk) split PR for that axis).

### The change

`sigmoid(x) = 1/(1+exp(-x))` and `silu(x) = x/(1+exp(-x))` differ only in the numerator,
so `ggml_v_sigmoid` is `ggml_v_silu` with `1` in place of `x` — the identical
`ggml_v_expf` kernel. Added for every arch that already has `ggml_v_silu`: **AVX-512,
AVX2, SSE2, SVE, NEON, RVV**, each immediately adjacent to its `ggml_v_silu` sibling.

`ggml_vec_sigmoid_f32` becomes a real SIMD kernel in `vec.cpp`, with the same
scalar-tail structure as `ggml_vec_silu_f32`; its `vec.h` definition becomes a
declaration. `ggml_compute_forward_sigmoid` (`unary-ops.cpp`) routes the f32->f32 case
to a new `ggml_compute_forward_sigmoid_f32`, row-split the same way
`ggml_compute_forward_silu_f32` (`ops.cpp`) already is; every other type combination
keeps the existing generic `unary_op<op_sigmoid>` path unchanged.

No new numerics are introduced by this PR: this is the same `ggml_v_expf` the tree
documents as *"maximum error 1.45358 plus 0.5 ulps"* and already ships in
`ggml_vec_silu_f32`, `ggml_vec_soft_max_f32` and friends. Shipped **unconditionally**,
the same way `silu` is — no new build or runtime flag.

### Accuracy — measured here against current upstream (`master` @ `e71b805`)

4,194,304 inputs spanning ±40 linearly, plus the saturation and signed-zero edge cases
(`0, -0, ±1e-30, 88, -104, ±inf`), `ggml_vec_sigmoid_f32` (SIMD) vs `1/(1+expf(-x))`
computed in double and rounded to float:

| metric | value |
|---|---|
| bitwise exact vs libm | 60.29 % |
| within 1 ulp | 94.29 % |
| max absolute error | 9.207e-08 |
| max relative error | 2.554e-07 (worst at x ≈ -16.78) |
| results outside [0,1] | 0 |
| `sigmoid(0)` / `sigmoid(-0)` | exactly 0.5 |
| `sigmoid(±1e-30)` | 0.5 |
| `sigmoid(88)` / `sigmoid(-104)` | 1 / 0 |
| `sigmoid(±inf)` | 1 / 0, no NaNs |

(An earlier measurement on a downstream fork, different compiler/libc, reported 74.2%
exact / 92.1% within 1 ulp / max rel error 2.75e-07 at x ≈ -6.38 — same order of
magnitude and identical qualitative conclusion; both are reported here rather than
picking one, since the exact percentages are sensitive to the host's libm.)

### Testing

- `test-backend-ops test -b CPU -o SIGMOID`: 8/8 passed (this suite's tolerance already
  accepts the same ulp class of change that `SILU` ships at).
- Broader regression sweep, `test-backend-ops test -b CPU -o SIGMOID,SILU,ABS,NEG,SGN,STEP,TANH,ELU,RELU,HARDSIGMOID,HARDSWISH,EXP`:
  96/96 passed.
- All of the above run at `OMP_NUM_THREADS=48`, `-b CPU` (the harness is vacuous on
  this backend without `-b CPU`).

### Note for reviewers

Sigmoid is used as a gate in several recent architectures (hyper-connection gates, MoE
routers, GLU variants) where it can sit on the critical path at batch 1, so the benefit
is not confined to large batches.

**A downstream evaluation is worth flagging honestly.** On one specific decode stack
(a custom architecture with a wide, already-thread-split `hc_gate` sigmoid), this
vectorisation measured **+0.0 pp** end-to-end once that stack's existing 48-way thread
split was already in place — the split was already hiding the scalar cost by giving
each thread only ~1 µs of the original 50 µs sigmoid — and it produced a measured
**top-1 token flip after ~19 generated tokens** in one of three test prompts (greedy,
256 tokens), consistent with the ulp-level numeric change above. That downstream stack
therefore did not adopt this change. **The situation this PR targets is different**: any
CPU build/config that does *not* already split identical-shape elementwise ops across
threads at batch 1 (i.e. does not carry something equivalent to the separate
(row, column-chunk) split PR) still pays the full scalar `expf` cost per `SIGMOID` node,
and this SIMD kernel removes exactly that cost, the same way it already does for `SILU`.
Numerically it is drop-in with `silu`'s existing precedent, so we believe it is correct
to ship unconditionally; we call out the flip only so a reviewer who cares about
byte-identical output across a refactor sees it named rather than discovered.
