# PC-4m CONCAT dim0 Row-Partition Hardening

Date: 2026-07-20

Scope: post-candidate `llama.cpp-experimental` source hardening for the default-off
`GGML_CPU_CONCAT_DIM0_ROWS=1` CPU path introduced by PC-4k and validated by
PC-4l. The frozen v7 promotion candidate remains `6ad45fa3ff`; this is
post-candidate research and is not a promotion-candidate update.

## Source Review

The hardening pass kept the optimization generic but opt-in:

- `ggml/src/ggml-cpu/ops.cpp`
  - added a support predicate for the fast path instead of relying on a naked
    env+dim check;
  - requires dim0 concat and matching block sizes across `src0`, `src1`, and
    `dst`;
  - requires dim0 lengths to be divisible by the type block size;
  - leaves all unsupported shapes on the existing concat kernels.
- `tests/test-backend-ops.cpp`
  - broadened dim0 transpose coverage from only `src1` transposed to `src0`
    transposed, `src1` transposed, and both transposed;
  - covers F32/F16/BF16 at `n_seq=1` and `n_seq=2`.

The row-partition implementation continues to copy one physical type block at a
time, matching the existing `concat_any` block-copy model for quantized dim0
concat. No default behavior changes unless `GGML_CPU_CONCAT_DIM0_ROWS=1` is set.

## Validation

All validation used the experimental DSO path explicitly:

```bash
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin
```

Commands and results:

- `git diff --check -- ggml/src/ggml-cpu/ops.cpp tests/test-backend-ops.cpp`
  passed.
- `cmake --build build-k24-cpu --target test-backend-ops -j 16` passed.
- `test-backend-ops -o CONCAT -b CPU -j 1` passed: `210/210`.
- `GGML_CPU_CONCAT_DIM0_ROWS=1 test-backend-ops -o CONCAT -b CPU -j 1`
  passed: `210/210`.
- `cmake --build build-k24-cpu --target test-recurrent-state-rollback -j 16`
  passed.
- `test-recurrent-state-rollback --model build-k24-cpu/tests/test-models/qwen35moe-moe.gguf`
  passed with env off.
- `GGML_CPU_CONCAT_DIM0_ROWS=1 test-recurrent-state-rollback --model build-k24-cpu/tests/test-models/qwen35moe-moe.gguf`
  passed with env on.

An unpinned `test-recurrent-state-rollback --help` probe segfaulted because the
ambient loader resolved the executable against production-v6 shared libraries.
That is an ABI/loader guardrail, not a PC-4m model result; all valid test runs
pin the experimental library path.

## Decision

PC-4m closes as source-hardened and correctness-expanded. The candidate remains
default-off and experimental only. Next step is an operator-approved
`llama.cpp-experimental` commit/package decision; do not commit or promote this
kernel patch silently, and do not turn it default-on without a separate
decision-grade promotion gate.
