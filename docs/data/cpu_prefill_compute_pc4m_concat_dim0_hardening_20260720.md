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

PC-4m closed as source-hardened and correctness-expanded. At the PC-4m
checkpoint, the candidate remained default-off and experimental only pending an
operator-approved `llama.cpp-experimental` commit/package decision; it was not
authorized for silent promotion or default-on behavior.

## PC-4n Addendum

PC-4n completed later on 2026-07-20 after explicit operator approval.

- Commit: `/mnt/raid0/llm/llama.cpp-experimental` `93d945885`
  (`Add default-off CPU CONCAT dim0 row partition`), pushed to
  `fork/experimental-v7-refresh-20260716`.
- Scope: only `ggml/src/ggml-cpu/ops.cpp` and
  `tests/test-backend-ops.cpp`.
- Guardrail: the path remains `GGML_CPU_CONCAT_DIM0_ROWS=1` default-off and
  now requires exact matching tensor types before using the row-partition
  kernel.
- Frozen-v7 status: unchanged; `6ad45fa3ff` remains the promotion candidate.

Post-commit validation repeated the focused PC-4m correctness gate with
experimental shared libraries pinned through `LD_LIBRARY_PATH`:

- `git diff --check -- ggml/src/ggml-cpu/ops.cpp tests/test-backend-ops.cpp`
- `cmake --build build-k24-cpu --target test-backend-ops -j 16`
- env-off CPU `CONCAT`: `210/210`
- env-on CPU `CONCAT`: `210/210`
- env-on `test-recurrent-state-rollback --model
  build-k24-cpu/tests/test-models/qwen35moe-moe.gguf`: restored successfully

PC-4o is the remaining admission/default-policy gate for the committed
post-candidate path. Do not treat PC-4n as default-on or as part of frozen v7.
