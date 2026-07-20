# PC-4f qwen35moe routed-MoE helper trace

Date: 2026-07-20

## Scope

Run a default-off routed-MoE helper diagnostic after PC-4e localized the FFN
island to routed `ffn_moe`. This is still target selection and prototype
preparation, not a production kernel change. The v7 promotion candidate remains
frozen at `6ad45fa3ff` / binary `10098`.

## Artifacts

- Run root:
  `data/cpu_prefill_compute/pc4f-qwen35-routed-moe-subtrace-20260720T004730Z/`
- Compact parsed summary:
  `data/cpu_prefill_compute/pc4f-qwen35-routed-moe-subtrace-20260720T004730Z/reports/trace_summary.json`
- Raw stderr/time trace:
  `data/cpu_prefill_compute/pc4f-qwen35-routed-moe-subtrace-20260720T004730Z/trace/architect_p8192_n1.trace_stderr_time.txt`

## Command shape

Model:
`/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Binary:
`/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench`

Key flags/env:

- `LLAMA_QWEN35_PREFILL_TRACE=2`
- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin:/usr/lib/llvm-20/lib`
- `GGML_IQK=1`
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`
- `taskset -c 0-95 numactl --interleave=all`
- `-t 96 -fa 1 -mmp 0 -p 8192 -n 1 -r 1 -dev none -ngl 0 -nopo 1 -nkvo 1 -o json -v`

Preflight `ldd` resolved `libllama.so.0` and `libggml-cpu.so.0` from the
experimental build directory. The run was CPU-only; cleanup found no residual
`llama-bench`, `llama-server`, `perf`, `rocprof`, AutoPilot, or KFD GPU PIDs.

## Result

The run exited `0`.

| Shape | Result |
|---|---:|
| `pp8192/n0` | `110.411171 t/s` |
| `tg1` | `5.162607 t/s` |
| Max RSS | `77044864 KiB` |
| Wall time | `2:38.75` |

The extra diagnostic expands several additional graph boundaries, so this
throughput row is trace-overhead evidence only, not a performance comparison.

Trace coverage:

| Item | Result |
|---|---:|
| Graph builds traced | `45` |
| Final graph-node count | `4471` every build |
| Phase lines | `2160` |
| Subphase lines | `50220` |

Routed-MoE deltas, median per traced subphase line:

| Subphase | Count | Median delta | Unique deltas |
|---|---:|---:|---|
| `ffn_moe_router_weights` | `2160` | `11` | `11` |
| `ffn_moe_gate_up` | `2160` | `2` | `2` |
| `ffn_moe_activation` | `2160` | `2` | `2` |
| `ffn_moe_down` | `2160` | `1` | `1` |
| `ffn_moe_weighted` | `2160` | `1` | `1` |
| `ffn_moe_expert_views` | `2160` | `8` | `8` |
| `ffn_moe_aggregate` | `2160` | `7` | `7` |
| `ffn_moe` | `2160` | `32` | `32` |
| `ffn_total` | `2160` | `40` | `40` |

## Interpretation

The routed MoE helper is not dominated by the actual gate/up/down math nodes in
the graph-node trace. The largest MoE-local graph-node contributors are:

- router/weights: `11` nodes;
- expert view expansion: `8` nodes;
- expert aggregation: `7` nodes.

Gate-up, activation, down projection, and expert weighting together account for
only `6` nodes. This aligns with the PC-3 timing profile: MoE `mul_mat_id`
math is still important, but the OpenMP spin/pause target likely needs fewer
helper-level scheduling boundaries around routing, views, and aggregation
rather than a blind new dot kernel.

## Decision

PC-4f closes as a routed-MoE diagnostic checkpoint. The next safe implementation
step is PC-4g:

1. Prototype exactly one default-off change around routed MoE view/aggregation
   scheduling, or around router/weights scheduling if a follow-up profile says
   top-k routing is the actual spin source.
2. Do not rewrite `mul_mat_id` math kernels in PC-4g; that remains a separate
   follow-up if barrier-count reduction fails.
3. Acceptance remains exact-output smoke plus repeated `p8192/n1` profile
   showing lower libomp spin/pause share and lower wall time.
