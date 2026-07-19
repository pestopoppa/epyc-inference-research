# PC-3 CPU Prefill-Compute Target Selection - 2026-07-19

Status: zero-inference follow-up to PC-0. This pass reuses the local OP-2
`perf.data` artifact and does not run inference, start servers, restart
AutoPilot, or touch production-v6 files.

Input profile:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/pc0-op2-20260719T225343Z/perf-record/architect_p8192_n1.perf.data`

Related PC-0 summary:
`docs/data/cpu_prefill_compute_pc0_op2_20260719.md`

## What Was Resolved

The large `(deleted)` DSO in the PC-0 report is not an unknown llama.cpp binary.
`perf buildid-list` maps build-id `597017da07b7fbe219d04036e9ca30d46654951b`
to `/usr/lib/llvm-20/lib/libomp.so.5`. The mmap event records that image as
`/ (deleted)`, which is why the initial `perf report` printed raw offsets.

Direct offset checks against `/usr/lib/llvm-20/lib/libomp.so.5`:

| Offset | Meaning |
|---|---|
| `0x7fea0` | LLVM OpenMP worker spin/pause loop (`pause` instruction in the worker wait path) |
| `0x7fad0` | same OpenMP worker wait/spin region |
| `0x7ff66` | transition from the worker loop to `__kmp_invoke_microtask` |
| `0xe5df0` | `rdtsc` helper adjacent to `__kmp_invoke_microtask` |

The raw symbol label `__kmpc_threadprivate_register_vec+...` is not a useful
source-level target; it is a nearest exported-symbol artifact for a stripped
libomp image. The disassembly identifies the actual hot region as OpenMP worker
spin/poll and microtask dispatch overhead.

## Target Ranking

Children report from the same profile:

| Area | Children |
|---|---:|
| `ggml_graph_compute_thread` | `48.30%` |
| OpenMP spin/pause hot offset (`libomp.so.5` `0x7fea0`) | `38.36%` self |
| `ggml_iqk_try_mul_mat_id` | `22.67%` |
| `iqk_mul_mat_moe` | `22.51%` |
| `ggml_compute_forward_mul_mat` | `10.37%` |
| `ggml_compute_forward_flash_attn_ext` | `5.59%` |
| `ggml_compute_forward_gated_delta_net` | `1.62%` |
| `ggml_compute_forward_rms_norm_mul_fused` | `1.08%` |
| `ggml_compute_forward_ssm_conv` | `1.03%` |

## Verdict

PC-3 closes positive for target selection. The first implementation target is
not a new low-level dot-product kernel. It is barrier-count / graph-fusion work
that reduces OpenMP worker spin and scheduling boundaries in the qwen35 prefill
graph.

Concrete first target:

1. In the qwen35/qwen35moe prefill graph, identify adjacent same-input compute
   islands around MoE feed-forward and SSM/GDN projections.
2. Prototype one default-off experimental fusion or graph grouping that reduces
   the number of OpenMP-dispatched graph nodes while preserving exact output.
3. Re-profile `p8192/n1` and require the libomp spin/pause bucket plus total wall
   time to drop before expanding scope.

Math hot paths remain important but are second in order:

- `ggml_iqk_try_mul_mat_id` / `iqk_mul_mat_moe` are the largest resolved math
  subtree, so MoE matmul packing/conversion can be a follow-up after barrier
  reduction.
- `ggml_compute_forward_mul_mat` and CPU flash-attn are visible, but smaller
  than the OpenMP wait bucket on this shape.
- `gated_delta_net`, fused RMS norm, and SSM conv are present but too small for
  first-kernel status in this profile.

## Local Raw Artifacts

Generated local reports under the PC-0 run root:

- `reports/architect_p8192_n1.perf_report.pc3_dso_symbol.txt`
- `reports/architect_p8192_n1.perf_report.pc3_children.txt`

These are local raw artifacts only. They contain long hardware counter values
and should not be committed unless a sanitized artifact format or narrow
benchmark-counter allow-list is added for the repository PII hook.
