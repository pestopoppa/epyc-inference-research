# DR-3 Quant-Asymmetric K2 Admission Prep

Date: 2026-07-20
Artifact: `data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T063100Z_codex_dryrun/`
Script: `scripts/benchmark/dr3_quant_asym_k2_admission_prep.py`
Status: dry-run package complete; live-runner follow-up added later the same day

## Purpose

DR-0e.2 proved the quant-asymmetric Qwen3.5-122B CPU verifier plus MI210 IQ2
drafter design is viable on a bounded slice. DR-2 selected K2 as the first
default-off serving candidate. DR-3 turns that decision into an executable package
shape before any live serving route exists.

## Package Contents

The prep script writes:

- `manifest.json`: fixed K2 decision, model/binary identity, admission task classes,
  required gates, and launch templates.
- `task_packet.jsonl`: broader admission task-class definitions.
- `commands.sh`: CPU baseline and combined-K2 launch templates for 8K and 16K
  context bands.
- `operator_run.sh`: explicit non-execution guard; live executor is not implemented.
- `summary.json`: dry-run summary and blockers.

Focused validation:

```bash
python3 -m py_compile scripts/benchmark/dr3_quant_asym_k2_admission_prep.py scripts/benchmark/test_dr3_quant_asym_k2_admission_prep.py
uv run --with pytest pytest -q scripts/benchmark/test_dr3_quant_asym_k2_admission_prep.py
```

Result: `5 passed`.

## Current Plan

Fixed K: `2`.

Context bands: `8192`, `16384`.

Admission task classes:

- `structured_json_long`
- `strict_formatting`
- `code_review_no_bug_controls`
- `architect_json_decisions`
- `long_repetitive_output`
- `long_context_tail`

Required gates:

- CPU-target equivalence.
- Quality non-regression.
- 8K and 16K coverage before routing.
- MI210 lease and cleanup proof.
- Frontdoor opportunity-cost measurement.
- Production-named `P-GPU-1` certification for any decision-grade GPU claim.

## Live-Runner Follow-Up

Follow-up report:
`docs/data/dr3_quant_asym_k2_admission_live_20260720.md`.

Live runner:
`scripts/benchmark/dr3_quant_asym_k2_admission_runner.py`.

Corrected 8K smoke artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T071200Z_live_smoke_ctx8192_r1_v2/`.

The live runner passed a one-row-per-class 8K smoke: quality `12/12`, output
stability pass, cleanup pass, `observation_grade=true`, combined K2 `11.104 t/s`
vs CPU baseline `7.185 t/s` (`1.545x`, alpha `0.876`). It remains non-serving
and decision-grade false pending default 8K+16K admission, frontdoor opportunity
cost, and production-named `P-GPU-1` gates.

## Verdict

DR-3a is scaffolded and DR-3b now has a live executor plus an 8K observation
smoke. The next code/execution step is the default 8K+16K admission package; do
not add a serving route or NumericSwarm K tunable until that broader package and
the frontdoor opportunity-cost gate pass.
