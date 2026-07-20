# DR-3 Quant-Asymmetric K2 Live Admission Runner

Date: 2026-07-20

Script: `scripts/benchmark/dr3_quant_asym_k2_admission_runner.py`

Passing live artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071200Z_live_smoke_ctx8192_r1_v2/`

Dry-run artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071100Z_live_runner_dryrun_v2/`

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

## Validation

```bash
python3 -m py_compile scripts/benchmark/dr3_quant_asym_k2_admission_runner.py scripts/benchmark/test_dr3_quant_asym_k2_admission_runner.py
uv run --with pytest pytest -q scripts/benchmark/test_dr3_quant_asym_k2_admission_runner.py
python3 scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --output-dir data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071100Z_live_runner_dryrun_v2 --context-band 8192 --rows-per-class 1
python3 scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --execute --output-dir data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071200Z_live_smoke_ctx8192_r1_v2 --context-band 8192 --rows-per-class 1 --startup-timeout 900 --request-timeout 900 --max-tokens 512
```

Focused test result: `8 passed`.

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

## Interpretation

DR-3b closes the live-runner implementation gap and gives a clean 8K
observation-grade admission smoke for the quant-asymmetric K2 lane. It does not
admit a production route yet.

Remaining work:

- Run the default 8K+16K admission package, not just the 8K smoke.
- Run the frontdoor opportunity-cost gate before any routing policy.
- Rerun required GPU claims under production-named `P-GPU-1` if the result is
  used for decision-grade production routing.
