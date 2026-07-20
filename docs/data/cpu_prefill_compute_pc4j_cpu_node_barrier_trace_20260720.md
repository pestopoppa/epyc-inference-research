# PC-4j qwen35moe CPU node/barrier trace

Date: 2026-07-20

## Scope

Attribute the PC-4h/PC-4i OpenMP barrier-heavy qwen35moe prefill profile inside
the single CPU scheduler split. PC-4i proved the `p8192/n1` workload is not
cross-split-bound: every graph ran as one CPU split with `4471` nodes and `0`
cross-backend inputs. PC-4j therefore instruments CPU graph execution around
`ggml_graph_compute_thread`, not the scheduler split planner.

This is attribution evidence only. The trace adds timing overhead and must not
be used as a clean throughput benchmark.

## Artifacts

- Run root:
  `data/cpu_prefill_compute/pc4j-qwen35moe-cpu-node-barrier-trace-20260720T015248Z/`
- Parsed summary:
  `data/cpu_prefill_compute/pc4j-qwen35moe-cpu-node-barrier-trace-20260720T015248Z/summary.json`
- CPU trace stderr:
  `data/cpu_prefill_compute/pc4j-qwen35moe-cpu-node-barrier-trace-20260720T015248Z/bench_stderr.txt`
- Strict postflight:
  `data/cpu_prefill_compute/pc4j-qwen35moe-cpu-node-barrier-trace-20260720T015248Z/post_process_check_strict.txt`

`bench.json` is intentionally excluded from versioned evidence because the
repository PII hook flags the long `model_n_params` field. The sanitized
`summary.json` preserves the benchmark rows used below.

## Command shape

Model:
`/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Binary:
`/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench`

Key flags/env:

- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin`
- `GGML_IQK=1`
- `GGML_CPU_TRACE_GRAPH=1`
- `GGML_CPU_TRACE_GRAPH_TOPK=16`
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`
- `taskset -c 0-95 numactl --interleave=all`
- `-t 96 -p 8192 -n 1 -r 1 -fa 1 -mmp 0 -dev none -ngl 0 -nopo 1 -nkvo 1 -o json`

The explicit `LD_LIBRARY_PATH` is required because the ambient shell path
contains production-v6 libraries. The run root records `ldd.txt`, which shows
`libllama*` and `libggml*` resolved from the experimental build directory.

## Result

The run exited `0`.

| Shape | Result |
|---|---:|
| `pp8192/n0` | `99.266829 t/s` |
| `tg1` | `5.173757 t/s` |

Trace metadata:

| Field | Value |
|---|---:|
| Graph summaries | `34` |
| Unique graph nodes | `4471` |
| Unique traced nodes | `2668` |
| Threads | `96` |
| Median compute time | `288785618 us` |
| Median barrier time | `202412198.5 us` |
| Median barrier/compute ratio | `0.7009` |

Top barrier-attributed operator classes:

| Rank | Op | Barrier time | Share of top-16 barrier time | Representative node |
|---:|---|---:|---:|---|
| 1 | `CONCAT` | `2399051163 us` | `36.9%` | `conv_input-0` |
| 2 | `MUL_MAT_ID` | `1174958024 us` | `18.1%` | `ffn_moe_gate-0` |
| 3 | `GATED_DELTA_NET` | `554312238 us` | `8.5%` | `node_44` |
| 4 | `MUL_MAT` | `538605575 us` | `8.3%` | `node_13` |
| 5 | `ADD` | `408310503 us` | `6.3%` | `node_36` |
| 6 | `RMS_NORM` | `256106183 us` | `3.9%` | `norm-0` |
| 7 | `FLASH_ATTN_EXT` | `230435030 us` | `3.5%` | `node_324` |
| 8 | `MUL` | `172958529 us` | `2.7%` | `gate-0` |

Top compute-attributed operator classes:

| Rank | Op | Compute time | Share of top compute time | Representative node |
|---:|---|---:|---:|---|
| 1 | `MUL_MAT_ID` | `3692188731 us` | `40.1%` | `ffn_moe_gate-0` |
| 2 | `MUL_MAT` | `2881268302 us` | `31.3%` | `node_13` |
| 3 | `FLASH_ATTN_EXT` | `809476594 us` | `8.8%` | `node_324` |
| 4 | `GATED_DELTA_NET` | `511441338 us` | `5.6%` | `node_44` |

IQK activation was present:

- `[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=12 n_as=256)`
- `[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=14 ne00=3072)`

## Source mapping

The dominant barrier-attributed node, `conv_input-*`, maps to the shared
delta-net recurrent state builder:

- `src/models/delta-net-base.cpp`: `build_conv_state()`
- Callers: `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`,
  `src/models/qwen3next.cpp`
- Current graph sequence:
  `build_rs(conv_states_all)` -> reshape conv state -> transpose `qkv_mixed` ->
  `ggml_concat(conv_states, qkv_mixed, dim=0)` -> `ggml_ssm_conv()` ->
  conv-state update views/copies.

This means the current PC-4 target is not MTP concat, scheduler splits, router
top-k, or a low-level `mul_mat_id` rewrite. It is the recurrent conv-input/state
graph boundary feeding SSM convolution, with `MUL_MAT_ID` remaining the main
compute sink rather than the first barrier-count target.

## Decision

PC-4j closes the CPU-backend attribution gate. The next implementable target is
a default-off PC-4k probe around `build_conv_state()` / `conv_input` in the
shared delta-net base. The probe should reduce or fuse the
`conv_states + qkv_mixed` concat/state-update graph boundary only if it can pass
an exact-output smoke and then show lower barrier-attributed time plus lower
wall time on repeated qwen35moe `p8192/n1`.

Do not re-open router/top-k, routed view/add aggregation, scheduler split/copy,
or `mul_mat_id` math prototypes from the current evidence set.
