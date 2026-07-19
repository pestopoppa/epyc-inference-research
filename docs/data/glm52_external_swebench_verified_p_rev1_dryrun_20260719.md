# GLM-5.2 SWE-Bench-Verified P-REV-1 Dry-Run (2026-07-19)

## Scope

- Rows: `docs/data/glm52_external_swebench_verified_n24_rows_20260719.jsonl`
- Plan artifact: `data/glm52_external_ground_truth_direct/glm52-external-swebench-verified-n24-p-rev1-dryrun-20260719/plan.json`
- Protocol: `p_rev1`
- Attestation: `MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`
- Era: `p_rev1_attested`

## Result

The external direct runner now accepts homogeneous `patch_review_oracle` rows without `candidate_b`, uses the existing ReviewDecision JSON schema (`approve|reject`, numeric confidence, `blocking.tripwire`, and `evidence`), and includes the SWE-Bench-Verified `FAIL_TO_PASS` / `PASS_TO_PASS` oracle in the prompt.

The dry-run is execution-ready:

- `execution_allowed=true`
- `refusal_reasons=[]`
- `24` accept-control rows
- source mix: `astropy/astropy=1`, `django/django=11`, `pydata/xarray=1`, `pylint-dev/pylint=1`, `sphinx-doc/sphinx=2`, `sympy/sympy=8`
- first prompt: `571` estimated tokens against `15744` max
- response schema: ReviewDecision, not pairwise `A|B`

This is no-inference prep only. The live CPU-only GLM run remains the open gate that reports SWE false-reject rate on known-good patches.

## Validation

```bash
uv run --with pytest pytest -q scripts/benchmark/test_glm52_external_ground_truth_adapter.py scripts/benchmark/test_glm52_external_ground_truth_direct_runner.py scripts/benchmark/test_glm52_swebench_verified_adapter.py
python3 -m py_compile scripts/benchmark/glm52_external_ground_truth_adapter.py scripts/benchmark/glm52_external_ground_truth_direct_runner.py scripts/benchmark/glm52_swebench_verified_adapter.py
```

Result: `24 passed`; compile checks passed.
