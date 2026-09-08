## ggml-cpu: parallelise elementwise kernels over (row, column-chunk) pairs at batch 1

### Summary

`get_thread_range()` (`ggml/src/ggml-cpu/common.h`) splits work over **rows only**:

```c
const int64_t nr = ggml_nrows(src0);
const int64_t dr = (nr + nth - 1)/nth;
const int64_t ir0 = dr*ith;
const int64_t ir1 = MIN(ir0 + dr, nr);
```

At batch 1 (single-token decode), every elementwise tensor in a typical graph is
`[nc,1,1,1]`, i.e. `nr == 1`. With `nr < nth`, `dr == 1`, so only `ith == 0` gets a
non-empty range `[0,1)` — every other thread computes `ir0 >= nr`, gets an empty range,
and falls straight through to the barrier having done no work. The parallelism the
caller asked for by setting `n_threads > 1` is simply unavailable to these ops at batch
1, no matter how many threads are configured.

This affects every caller of `apply_unary_op` / `apply_unary_op_functor`
(`unary-ops.cpp`, all 20-odd scalar `GGML_OP_UNARY` cases including `SIGMOID`, `TANH`,
`RELU`, `EXP`, `SQR`, `SQRT`, ...), `apply_binary_op` (`binary-ops.cpp`, `SUB` and any
other op sharing that template), and `ggml_compute_forward_repeat_f32`/`_f16`
(`ops.cpp`), which was **single-threaded outright** (`if (ith != 0) return;`) even
though every destination copy is disjoint.

### The fix

Elementwise ops compute each output element from the input element(s) at the same
index, so cutting a row into column chunks and dealing the `(row, chunk)` pairs out to
the threads is **bit-identical to the row-only result by construction**: every output
element is still produced by exactly one thread from the same input bytes, just not
necessarily the same thread as before. This mirrors the existing `GET_ROWS`
`(row, column-chunk)` split (`ggml_get_rows_split_init()`), applied to the generic
elementwise templates.

`get_rowcol_split()` (new, `common.h`) computes a `(row, chunk)` enumeration:

- When `nr >= nth` (enough rows to occupy every thread already), or the row is shorter
  than `ggml_rowcol_min_elems()` (512 elements — chunking overhead isn't worth it below
  that), it degenerates to exactly the old single-chunk-per-row behaviour (`ncc == 1`,
  `cstep == nc`). **Zero-cost, zero-behaviour-change** when the split isn't needed.
- Otherwise it picks enough column chunks per row to occupy every thread, aligned to a
  cache line of the widest element type touched, so two threads never write the same
  line.

`apply_unary_op`, `apply_unary_op_functor`, and `apply_binary_op` are rewritten to
iterate `(row, chunk)` tasks instead of rows; `ggml_compute_forward_repeat_f32`/`_f16`
drop their `if (ith != 0) return;` and use the same split.

Shipped **unconditionally** — no new build or runtime flag. It is bit-identical to the
existing behaviour when the split doesn't fire, and a strict occupancy improvement
(same output, more threads doing useful work) when it does.

### Correctness

`test-backend-ops -b CPU`, every elementwise op the patch touches plus `REPEAT`, at
`OMP_NUM_THREADS=48`:

```
SIGMOID, SILU, ABS, NEG, SGN, STEP, TANH, ELU, RELU, HARDSIGMOID, HARDSWISH, EXP,
GELU, GELU_QUICK, SQR, SQRT, SIN, COS, LOG, EXPM1, SOFTPLUS, FLOOR, CEIL, ROUND,
TRUNC, MUL, ADD, REPEAT
```

**424/424 passed, 0 failures.** (`-b CPU` is required — the harness is vacuous on this
backend without it.)

### Performance — measured here against current upstream (`master` @ `e71b805`)

Standalone micro-benchmark (not `test-backend-ops`; included in this PR's supporting
material): a single `SIGMOID(f32, [10240,1,1,1])` node — the batch-1 shape this patch
targets — plus an unrelated padding `ADD1` node (needed only so `ggml_graph_plan()`
doesn't cap the whole graph's thread count to 1, which it does for a graph made
entirely of `n_tasks == 1` ops; a real decode graph always has a `MUL_MAT`/etc. node
alongside, so this is representative, not contrived). 20,000 timed calls after 200
calls of warmup, no sigmoid vectorisation applied in either arm — this isolates the
split's effect alone:

| `n_threads` | before (us/call) | after (us/call) | delta |
|---:|---:|---:|---:|
| 1  | 14.735 | 15.130 | +2.7% (noise — `nr(1) < nth(1)` is false, so the split does not fire; expected to match) |
| 8  | 22.640 | 11.118 | **-50.9%** |
| 48 | 30.708 | 17.723 | **-42.3%** |
| 96 | 48.616 | 29.878 | **-38.5%** |

Note the "before" column *increases* with thread count even though no useful work is
being added — that's barrier/launch overhead scaling with idle threads, which is
exactly what this patch removes.

This is a synthetic, single-node isolation, not a serving-throughput number; the
mechanism it demonstrates (occupancy at batch 1) is the same one a full decode graph
exercises across hundreds of such nodes per token.

### Scope note

`ARGSORT` is deliberately **not** touched: it is `std::sort` over one row (`nr == 1`),
a sort is not elementwise (output element `i` depends on the whole row, not input
element `i`), and a parallel merge over many threads would cost more barrier overhead
than the sequential sort it would replace at the row sizes this fires at.
