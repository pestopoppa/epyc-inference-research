# v7 Final Cutover Smoke - 2026-07-19

## Scope

Final pre-promotion coherence / garbage smoke for experimental v7:
`/mnt/raid0/llm/llama.cpp-experimental`, branch `experimental-v7-refresh-20260716`,
source tip `6ad45fa3ff`.

Preflight found the HIP server binary was stale (`llama-server --version` reported
`6a8dd5ea6` while source was `6ad45fa3ff`). The experimental HIP `llama-server`
target was rebuilt in `llama.cpp-experimental` only. Production v6 was not built,
edited, or touched. Rebuilt binary reports `version: 10098 (6ad45fa3f)`.

AutoPilot stayed stopped. Final postflight showed no `llama-server`, K35 runner,
AutoPilot, or KFD process and `0%` VRAM.

## Harness Fixes

- `k35_stack_context_matrix_runner.py`: updated the `P-GPU-1` caveat from stale
  amendment-prep wording to the ratified production-named-kernel rule.
- `k35_stack_context_matrix_runner.py`: fixed cleanup proof. `ps -p <dead_pid>`
  exits `1` when the PID is absent, which is valid proof of cleanup.
- `k35_vision_matrix_runner.py`: added `vision_escalation_cpu_qwen25vl_alias`
  to match the current orchestrator safety alias.
- `k35_vision_matrix_runner.py`: moved the known-bad Qwen3-VL-30B MoE4
  escalation row out of the default set.
- `k35_vision_matrix_runner.py`: CPU-only rows now hide ROCm devices, while
  MI210 rows keep `ROCm0`.
- `k35_vision_matrix_runner.py`: cleanup failures now fail the summary instead
  of silently relying on the final blocker scan.

Validation: `python3 -m py_compile ...` and
`PYTHONPATH=scripts/benchmark python3 -m unittest scripts/benchmark/test_k35_vision_matrix_runner.py scripts/benchmark/test_k35_stack_context_matrix_runner.py`
passed (`39` tests).

## Non-Vision Result

Artifact:
`/mnt/raid0/llm/epyc-inference-research/data/v7_final_cutover_smoke/nonvision_20260719T183723Z/summary.json`

Status: `ok`; cleanup failures `0`; cleanup process blockers `0`.

| Scenario | Status | Prompt t/s | Decode t/s | Draft acceptance |
|---|---:|---:|---:|---:|
| `frontdoor_gpu_native_mtp` | ok | 1472.02 | 122.81 | 1.000 |
| `worker_general_cpu_composed_spec` | ok | 252.19 | 63.48 | 0.580 |
| `architect_general_cpu_native_mtp` | ok | 122.93 | 23.53 | 0.990 |
| `ingest_long_context_cpu_default_experts` | ok | 199.06 | 20.60 | n/a |

The non-vision fixture intentionally asks for repeated `benchmark` tokens; the
repeated output is expected for this throughput/garbage smoke and not a content
quality defect.

## Vision Result

Artifact:
`/mnt/raid0/llm/epyc-inference-research/data/v7_final_cutover_smoke/vision_20260719T184344Z/summary.json`

Status: `ok`; cleanup failures `0`; cleanup process blockers `0`.

| Scenario | Role | Fixture pass | Prompt t/s range | Decode t/s range |
|---|---|---:|---:|---:|
| `worker_vision_cpu_qwen25vl` | `worker_vision` | 4/4 | 140.14-171.43 | 31.80-41.83 |
| `vision_escalation_cpu_qwen25vl_alias` | `vision_escalation` | 4/4 | 140.09-173.03 | 31.99-41.66 |
| `vision_candidate_mi210_minicpm_o45_q4` | `vision_escalation_candidate` | 4/4 | 735.72-884.52 | 114.82-126.91 |

The stale default Qwen3-VL-30B escalation row was not run; that lane is known to
fail the chart fixture and is no longer the current registry-safe
`vision_escalation` default.

## Decision Boundary

This closes the pre-promotion final cutover coherence / garbage smoke for the
current experimental-v7 candidate. These rows are observation-grade because they
use `llama.cpp-experimental`; `P-GPU-1` production-number certification still
requires post-promotion reruns on `production-consolidated-v7`.

Remaining v7 readiness question: native GLM-MTP alpha / throughput only if the
operator keeps GLM acceleration coupled to the v7 release after the reviewer
route-away verdict.
