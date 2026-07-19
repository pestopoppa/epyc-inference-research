# P-GPU-1 Artifact Completeness Audit

- Schema: `epyc.pgpu1_artifact_completeness_audit.v1`
- Created: `2026-07-19T09:33:07.707266+00:00`
- Scope: artifact-only; no inference, server, benchmark, build, or ROCm command executed
- Overall status: `incomplete`
- Recommendation: `rerun_required_for_incomplete_artifacts`

| Artifact | Status | Recommendation | Missing required fields | Near misses |
|---|---|---|---|---|
| `data/k35_stack_context_matrix/frontdoor_pgpu1_candidate_20260718Tquiet` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `cpu_interference_policy`, `post_cleanup_vram_sample` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `cpu_interference_policy`, `post_cleanup_vram_sample` |
| `data/k35_stack_context_matrix/frontdoor_context_edges_20260718Tcodex/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `rep_count`, `cpu_interference_policy`, `post_cleanup_vram_sample` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `cpu_interference_policy`, `post_cleanup_vram_sample` |
| `/mnt/raid0/llm/tmp/k35-memory-backfill-20260717T1400Z/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `rep_count`, `cpu_interference_policy`, `post_cleanup_vram_sample` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `cpu_interference_policy`, `post_cleanup_vram_sample` |
| `/mnt/raid0/llm/tmp/k35-minicpm-service-matrix-20260717T2045Z/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `rep_count`, `cpu_interference_policy`, `post_cleanup_vram_sample` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `cpu_interference_policy`, `post_cleanup_vram_sample` |
| `/mnt/raid0/llm/tmp/k35-frontdoor-operational-1024-20260717T201842Z/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `warmup_discard_policy`, `rep_count`, `cpu_interference_policy`, `post_cleanup_vram_sample` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `cpu_interference_policy`, `post_cleanup_vram_sample` |
| `data/gpu-mi210/axa2_32k_prefill_qwen35_122b_v1_q4kv_b1024_ub256_20260719T071051Z/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `vram_pid_util_samples`, `warmup_discard_policy`, `rep_count`, `cpu_interference_policy` | - |
| `data/gpu-mi210/axa2_fa_all_quants_mixed_kv_validation_20260719T073906Z/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `vram_pid_util_samples`, `rep_count`, `cpu_interference_policy` | - |
| `data/gpu-mi210/axa2_current_build_no_warmup_homogeneous_controls_20260719T074757Z/summary.json` | `incomplete` | `rerun_required` | `rocm_clocks_before_after`, `rocm_power_before_after`, `rocm_temp_before_after`, `vram_pid_util_samples`, `rep_count`, `cpu_interference_policy` | - |

## Field Semantics

A near miss means related evidence exists but does not satisfy the explicit P-GPU-1 field.
For example, `process_blockers: []` is not the same as an explicit CPU-stack interference policy.

No inference, server, benchmark, build, or ROCm command is run by this audit.
