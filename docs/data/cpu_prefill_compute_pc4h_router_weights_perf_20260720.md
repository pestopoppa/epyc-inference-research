# PC-4h qwen35moe router/weights perf profile

Date: 2026-07-20

## Scope

Profile the routed `ffn_moe_router_weights` island before attempting another
PC-4 scheduling prototype. PC-4f showed router/weights as the largest
routed-MoE graph-node subphase (`11` nodes), and PC-4g rejected skipping eager
view/add expansion. This run asks whether router/top-k/weights symbols are hot
enough to justify a targeted prototype.

## Artifacts

- Run root:
  `data/cpu_prefill_compute/pc4h-qwen35moe-router-weights-perf-20260720T011655Z/`
- Parsed summary:
  `data/cpu_prefill_compute/pc4h-qwen35moe-router-weights-perf-20260720T011655Z/summary.json`
- Filtered symbol lines:
  `data/cpu_prefill_compute/pc4h-qwen35moe-router-weights-perf-20260720T011655Z/router_weight_symbols.txt`
- Raw `perf.data` and full reports remain local scratch in the run root and are
  not intended for git.

## Command shape

Model:
`/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Binary:
`/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench`

Key flags/env:

- `perf record -F 99 -g --call-graph fp`
- `GGML_IQK=1`
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`
- `taskset -c 0-95 numactl --interleave=all`
- `-t 96 -p 8192 -n 1 -r 1 -fa 1 -mmp 0 -dev none -ngl 0 -nopo 1 -nkvo 1 -o json`

Kernel symbols were restricted by `/proc/sys/kernel/kptr_restrict`; user-space
symbols were still resolved. Cleanup found no residual `llama-bench`,
`llama-server`, `llama-cli`, `llama-simple`, `perf`, `rocprof`, AutoPilot, or
KFD GPU PIDs.

## Result

The run exited `0` and captured `1,563,013` samples.

| Shape | Result |
|---|---:|
| `pp8192/n0` | `100.278682 t/s` |
| `tg1` | `5.110189 t/s` |

Top children profile:

| Symbol/path | Children | Self | Interpretation |
|---|---:|---:|---|
| `ggml_graph_compute_thread.isra.0` | `59.19%` | `0.01%` | worker execution envelope |
| `GOMP_barrier` / `__kmpc_barrier` | `43.95%` | included in worker path | dominant spin/barrier cost |
| `ggml_compute_forward_flash_attn_ext` | `4.68%` | low | largest resolved compute island |
| `ggml_compute_forward_gated_delta_net` | `2.08%` | low | recurrent attention, not router |
| `ggml_compute_forward_mul` | `1.71%` | low | generic elementwise |
| `ggml_compute_forward_rms_norm_mul_fused` | `1.35%` | low | fused norm |
| `ggml_compute_forward_add_non_quantized` | `1.35%` | low | add path |
| `ggml_compute_forward_glu` | `1.19%` | low | activation path |
| `ggml_compute_forward_ssm_conv` | `1.08%` | low | recurrent conv |

Router/top-k/weights symbols were tiny in the no-children profile:

| Symbol | Self |
|---|---:|
| `ggml_vec_soft_max_f32` | `0.05%` |
| `std::__introsort_loop<cmp_argsort...>` | `0.02%` |
| `ggml_compute_forward_get_rows` | `0.02%` |
| `ggml_compute_forward_soft_max` | `0.02%` |
| `ggml_compute_forward_argsort` | `0.01%` |
| `ggml_compute_forward_div` | `0.00%` |
| `ggml_compute_forward_clamp` | `0.00%` |
| `ggml_compute_forward_set_rows` | `0.00%` |

## Decision

PC-4h closes as a router/weights no-go. Do not start a top-k/router/weights
prototype from the current evidence: the graph-node count was high, but the
resolved symbol cost is too small. The live bottleneck is still scheduler /
barrier behavior.

The next PC-4 step should attribute OpenMP barriers to graph scheduling
boundaries before changing math or routing operators. Candidate direction:
collect graph/scheduler execution telemetry that maps barrier-heavy regions to
specific graph segments or split boundaries, then test one default-off
scheduler-level prototype only if that attribution identifies a target.
