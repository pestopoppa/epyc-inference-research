# PC-4e qwen35moe FFN boundary trace

Date: 2026-07-20

## Scope

Add and run a default-off qwen35moe-local FFN boundary trace so PC-4 can
choose a precise MoE/FFN implementation boundary before any prototype. This is
post-candidate research only: the v7 promotion candidate remains frozen at
`6ad45fa3ff` / binary `10098`, and the trace instrumentation is not part of
the promotion candidate.

## Artifacts

- Run root:
  `data/cpu_prefill_compute/pc4e-qwen35-ffn-subtrace-20260720T003822Z/`
- Compact parsed summary:
  `data/cpu_prefill_compute/pc4e-qwen35-ffn-subtrace-20260720T003822Z/reports/trace_summary.json`
- Raw stderr/time trace:
  `data/cpu_prefill_compute/pc4e-qwen35-ffn-subtrace-20260720T003822Z/trace/architect_p8192_n1.trace_stderr_time.txt`

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
| `pp8192/n0` | `115.842650 t/s` |
| `tg1` | `5.266122 t/s` |
| Max RSS | `77043940 KiB` |
| Wall time | `2:31.91` |

Trace coverage:

| Item | Result |
|---|---:|
| Graph builds traced | `45` |
| Final graph-node count | `4471` every build |
| Phase lines | `2160` |
| Subphase lines | `35100` |

FFN boundary deltas, median per traced subphase line:

| Subphase | Count | Median delta | Unique deltas |
|---|---:|---:|---|
| `ffn_moe` | `2160` | `32` | `32` |
| `ffn_shared` | `2160` | `4` | `4` |
| `ffn_shared_gate` | `2160` | `2` | `2` |
| `ffn_shared_gated` | `2160` | `1` | `1` |
| `ffn_moe_shared_add` | `2160` | `1` | `1` |
| `ffn_total` | `2160` | `40` | `40` |

Context deltas stayed consistent with PC-4c:

| Subphase | Count | Median delta | Unique deltas |
|---|---:|---:|---|
| `linear_attn_total` | `1620` | `53` | `53,55` |
| `full_attn_total` | `540` | `29` | `29` |

## Interpretation

The FFN island is now localized: routed MoE accounts for `32` of the stable
`40` FFN graph nodes on every layer. Shared expert, shared gate, shared gating
multiply, and the final routed+shared add account for only `8` nodes combined.

This keeps PC-4 aligned with the PC-3 timing profile: OpenMP spin/barrier and
MoE `mul_mat_id` remain the dominant implementation direction, while recurrent
GDN/SSM remains too small in timing evidence for first-patch status.

## Decision

PC-4e closes as a diagnostic checkpoint. The next safe implementation step is
PC-4f:

1. Add a narrow, default-off diagnostic inside the routed `build_moe_ffn` helper
   or a qwen35moe-only wrapper that separates router/top-k, gate-up, down
   projection, weighting, per-expert view expansion, and expert aggregation.
2. Prototype only after that diagnostic identifies the boundary tied to the
   OpenMP spin/pause bucket.
3. Preserve the PC-4 acceptance rule: exact-output smoke plus repeated
   `p8192/n1` profile showing lower libomp spin/pause and lower wall time.

Do not spend PC-4 implementation effort on shared-expert fusion first; its
graph-node budget is small relative to routed MoE.
