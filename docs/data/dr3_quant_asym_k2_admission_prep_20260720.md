# DR-3 Quant-Asymmetric K2 Admission Prep

Date: 2026-07-20
Artifact: `data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T063100Z_codex_dryrun/`
Script: `scripts/benchmark/dr3_quant_asym_k2_admission_prep.py`
Status: dry-run package only; no inference executed

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

## Verdict

DR-3 is scaffolded, not executable. The next code step is a live admission executor
that materializes task rows, runs CPU baseline vs combined K2, scores equivalence,
and keeps serving/NumericSwarm integration disabled until the package passes.
