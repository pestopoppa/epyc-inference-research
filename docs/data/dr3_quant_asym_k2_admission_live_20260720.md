# DR-3 Quant-Asymmetric K2 Live Admission Runner

Date: 2026-07-20

Admission script: `scripts/benchmark/dr3_quant_asym_k2_admission_runner.py`

Opportunity-cost script: `scripts/benchmark/dr3_frontdoor_opportunity_cost_gate.py`

Passing live artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071200Z_live_smoke_ctx8192_r1_v2/`

Passing default admission artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071816Z_dr3c_default_ctx8192_16384_r1/`

Passing opportunity-cost artifact:
`data/dr3_frontdoor_opportunity_cost/dr3_frontdoor_opportunity_cost_20260720T074853Z_live_ctx8192_r1/`

Dry-run artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071100Z_live_runner_dryrun_v2/`

Opportunity-cost dry-run artifact:
`data/dr3_frontdoor_opportunity_cost/dr3_frontdoor_opportunity_cost_20260720T075235Z_dryrun_v2/`

## Scope

DR-3b turns the DR-3a dry-run package into a default-off live admission runner.
The runner is still observation-grade and does not add a serving route,
NumericSwarm surface, or production-stack config.

Execution shape:

- CPU baseline: Qwen3.5-122B `UD-Q4_K_M`, `--device none`, `--spec-type none`.
- Combined K2: CPU Qwen3.5-122B `UD-Q4_K_M` verifier plus MI210 Qwen3.5-122B
  `UD-IQ2_M` drafter, `--spec-type draft-mtp`, `--spec-draft-n-max 2`.
- Fresh sequential `llama-server` instances per arm.
- Context band: `8192`.
- Rows: one row per admission task class.

## Runner Changes

- Materializes six broader admission rows: structured JSON, strict formatting,
  code-review no-bug control, architect JSON decision, exact-copy repetitive
  output, and long-context tail recall.
- Adds per-row quality checks and per-row `max_tokens`.
- Scores output stability according to each row's declared equivalence rule:
  exact hash where required; semantic rows require both baseline and combined K2
  to pass the row checker.
- Preserves explicit non-serving gates: `serving_route_allowed=false`,
  `numeric_swarm_surface_allowed=false`, `decision_grade=false`.
- Records cleanup proof and refuses contaminated preflight by default.

DR-3d adds a separate dry-run-first opportunity-cost gate. It runs the resident
frontdoor shape, tears it down, runs the combined K2 CPU-verifier + MI210-drafter
lane, tears that down, then reloads frontdoor to measure whether the lease causes
a reload/decode regression. It keeps `serving_route_allowed=false` and
`numeric_swarm_surface_allowed=false`.

## Validation

```bash
python3 -m py_compile scripts/benchmark/dr3_quant_asym_k2_admission_runner.py scripts/benchmark/test_dr3_quant_asym_k2_admission_runner.py
uv run --with pytest pytest -q scripts/benchmark/test_dr3_quant_asym_k2_admission_runner.py
python3 scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --output-dir data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071100Z_live_runner_dryrun_v2 --context-band 8192 --rows-per-class 1
python3 scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --execute --output-dir data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071200Z_live_smoke_ctx8192_r1_v2 --context-band 8192 --rows-per-class 1 --startup-timeout 900 --request-timeout 900 --max-tokens 512
python3 -m py_compile scripts/benchmark/dr3_frontdoor_opportunity_cost_gate.py scripts/benchmark/test_dr3_frontdoor_opportunity_cost_gate.py
uv run --with pytest pytest -q scripts/benchmark/test_dr3_frontdoor_opportunity_cost_gate.py scripts/benchmark/test_dr3_quant_asym_k2_admission_prep.py scripts/benchmark/test_dr3_quant_asym_k2_admission_runner.py
python3 scripts/benchmark/dr3_frontdoor_opportunity_cost_gate.py --output-dir data/dr3_frontdoor_opportunity_cost/dr3_frontdoor_opportunity_cost_20260720T075235Z_dryrun_v2 --context 8192
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin python3 scripts/benchmark/dr3_frontdoor_opportunity_cost_gate.py --execute --output-dir data/dr3_frontdoor_opportunity_cost/dr3_frontdoor_opportunity_cost_20260720T074853Z_live_ctx8192_r1 --context 8192 --reps 1 --frontdoor-max-tokens 512 --min-completion-tokens 128 --dr3-max-tokens 512 --request-timeout 1200 --startup-timeout 1200
```

Focused test result after the DR-3d runner landed: `17 passed`.

## Live Smoke Result

Summary:

- `quality_gate.status=pass` (`12/12`).
- `output_stability_gate.status=pass`.
- `context_coverage_gate.status=pass` for the requested 8K band.
- `cleanup_proof.status=pass`; post-run check showed no llama-family process and
  no KFD PID leak.
- `observation_grade=true`.
- `decision_grade=false`.
- `serving_route_allowed=false`.

Speed:

| Arm | Decode t/s | Ratio vs CPU baseline | Alpha |
|---|---:|---:|---:|
| CPU baseline 8K | 7.185 | 1.000x | n/a |
| Combined K2 8K | 11.104 | 1.545x | 0.876 |

The combined K2 row accepted `408/466` draft tokens. Spec telemetry was observed.

## DR-3c Default Admission Result

Default 8K+16K package:

```bash
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin \
python3 scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --execute \
  --output-dir data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071816Z_dr3c_default_ctx8192_16384_r1 \
  --context-band 8192 --context-band 16384 --rows-per-class 1 \
  --startup-timeout 900 --request-timeout 1200 --max-tokens 1024
```

Summary:

- `quality_gate.status=pass` (`24/24`).
- `output_stability_gate.status=pass`.
- `context_coverage_gate.status=pass` for `8192` and `16384`.
- `cleanup_proof.status=pass`.
- `observation_grade=true`.
- `decision_grade=false`.
- `serving_route_allowed=false`.
- `numeric_swarm_surface_allowed=false`.

Speed:

| Context | CPU baseline decode t/s | Combined K2 decode t/s | Ratio | Alpha | Draft accepted/generated |
|---:|---:|---:|---:|---:|---:|
| 8192 | 6.980 | 10.535 | 1.509x | 0.876 | 408/466 |
| 16384 | 6.979 | 10.429 | 1.494x | 0.879 | 420/478 |

The package keeps the lane non-serving. `frontdoor_opportunity_cost_gate` remains
`not_run`, and `p_gpu_1_gate.status=not_applicable_to_experimental_observation`.

## DR-3d Frontdoor Opportunity-Cost Result

The frontdoor opportunity-cost gate passed as an experimental-v7 observation:

- Artifact:
  `data/dr3_frontdoor_opportunity_cost/dr3_frontdoor_opportunity_cost_20260720T074853Z_live_ctx8192_r1/`.
- `frontdoor_opportunity_cost_gate.status=pass`.
- `cleanup_proof.status=pass`; post-run verification showed no llama-family
  process leak and no KFD PID.
- `observation_grade=true`; `decision_grade=false`.
- `serving_route_allowed=false`; `numeric_swarm_surface_allowed=false`.
- `p_gpu_1_gate.status=not_applicable_to_experimental_observation`.

| Measurement | Value |
|---|---:|
| Frontdoor before lease decode | `93.690 t/s` |
| Frontdoor before lease load wall | `7.439 s` |
| Frontdoor after eviction/reload decode | `94.157 t/s` |
| Frontdoor after eviction/reload load wall | `7.461 s` |
| After/before decode ratio | `1.005x` |
| DR-3 K2 active decode | `11.701 t/s` |
| DR-3 K2 active alpha | `1.000` |
| DR-3 K2 draft accepted/generated | `128/128` |

This removes the frontdoor opportunity-cost gate as an experimental observation
blocker for the K2 lane. It does not authorize serving, because production GPU
claims and production routing still require the production-named `P-GPU-1` rerun
path after the operator promotes v7.

## Interpretation

DR-3b closed the live-runner implementation gap, DR-3c closed the default
8K+16K admission-package execution gap, and DR-3d closed the experimental
frontdoor opportunity-cost gate for the quant-asymmetric K2 lane. These results
do not admit a production route yet.

Remaining work:

- Rerun required GPU claims under production-named `P-GPU-1` if the result is
  used for decision-grade production routing.
