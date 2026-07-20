# PC-4g routed-MoE compact aggregation prototype

Date: 2026-07-20

## Scope

Test the first default-off PC-4g prototype suggested by PC-4f: skip the eager
`ggml_build_forward_expand` calls for routed MoE expert views and aggregation
adds, letting the final output expansion pull the same dependencies. The
prototype was gated by `LLAMA_QWEN35_MOE_COMPACT_AGGREGATION=1`, then removed
after the measured prefill regression below. The v7 promotion candidate remains
frozen at `6ad45fa3ff` / binary `10098`; this was post-candidate research only.

## Artifacts

- Exact-output smoke:
  `data/cpu_prefill_compute/pc4g-qwen35moe-compact-aggregation-simple-smoke-20260720T010239Z/`
- `p8192/n1` paired bench:
  `data/cpu_prefill_compute/pc4g-qwen35moe-compact-aggregation-p8192-20260720T010334Z/`
- Parsed paired summary:
  `data/cpu_prefill_compute/pc4g-qwen35moe-compact-aggregation-p8192-20260720T010334Z/summary.json`

## Validation

Build/test before and after the reverted prototype:

- `git diff --check -- src/llama-graph.h src/llama-graph.cpp src/models/models.h src/models/qwen35moe.cpp src/models/qwen35.cpp`
- `cmake --build build-k24-cpu --target llama-bench llama-simple -j 16`
- `ctest --test-dir build-k24-cpu -R '^test-llama-archs$' --output-on-failure`

All passed. The failed prototype code was reverted, leaving only the PC-4f
diagnostic instrumentation in the experimental worktree.

## Exact-output smoke

Harness: `llama-simple`, CPU, Qwen3.6-35B-A3B Q8, greedy, `24` generated
tokens.

Result: default and compact outputs were byte-identical.

| Arm | Wall | Prompt eval | Decode |
|---|---:|---:|---:|
| Default | `0:03.71` | `36.91 t/s` | `12.32 t/s` |
| Compact aggregation | `0:03.64` | `37.05 t/s` | `12.42 t/s` |

The smoke establishes that the graph schedule change can preserve output on a
small qwen35moe-family shape. It is not a performance gate.

## `p8192/n1` result

Harness: `llama-bench`, CPU-only, Qwen3.5-122B-A10B UD-Q4_K_M, `-t 96`,
`-p 8192`, `-n 1`, `-r 2`, `GGML_IQK=1`, `-fa 1`, `-mmp 0`, `-dev none`,
`-ngl 0`, `-nopo 1`, `-nkvo 1`.

| Arm | `pp8192` | `tg1` | Wall | Max RSS |
|---|---:|---:|---:|---:|
| Default | `141.588462 t/s` | `5.242545 t/s` | `3:41.92` | `76970508 KiB` |
| Compact aggregation | `100.069829 t/s` | `4.840255 t/s` | `4:17.40` | `77042104 KiB` |

The prototype regressed prompt throughput by about `29.3%`, decode by about
`7.7%`, and wall time by about `16.0%`.

## Decision

PC-4g compact aggregation is rejected. Do not re-propose "skip eager routed MoE
view/add expansion" as a keep-candidate without a new mechanism or a different
profile. The view/aggregation nodes are graph-large but the eager expansion
appears beneficial for the scheduler on the real `p8192/n1` workload.

The next safe PC-4 step is not a `mul_mat_id` rewrite. Use a profile-first
router/weights investigation or a narrower scheduling change that preserves the
beneficial view/add expansion behavior.

## Side finding

`llama-cli` initially aborted at `common/arg.cpp:2551` because
`libllama-cli-impl.so` / `libllama-server-impl.so` were stale relative to a
newer `libllama-common.so.0` after `common_params` layout changed. Rebuilding
`llama-cli` and `llama-server` in `build-k24-cpu` restored matching
experimental linkage; `llama-cli --version` now reports build `10099`
(`12a292f0c`).
