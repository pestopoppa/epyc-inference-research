# CJ-15 minimal frozen local judge-cascade fixture

This is an offline conformance fixture for the local judge-cascade response
contract. It makes no model or hosted API calls and does not reproduce or
validate the separate 510-pair Jev simulation.

## Contents and provenance

- `source_records.jsonl` contains six frozen rows projected from the EPYC
  typed-versus-LLM judge redundancy receipt captured on 2026-09-18. It retains
  the source case IDs and hashes, answer hashes, rubric criteria and labels,
  and both readers' recorded outputs. The upstream receipt did not contain
  answer text; the source module hash in `manifest.json` binds those answer
  hashes to the source case definitions. The six selected IDs and source file
  digests are fixed in the manifest.
- `stress_rows.jsonl` contains seven small, hand-authored scenarios in both
  forward and reverse option order (14 rows). Scenarios cover valid typed
  output, primary abstention with fallback, omitted choice with fallback,
  invalid choice with fallback abstention, rubric/display-name swaps,
  misleading style, and a direct order control. Scripted response strings are
  parser fixtures, not observed model outputs.
- `manifest.json` freezes the source provenance, authored generation recipe,
  reference/evaluation split, scorer, expected output for each stress row,
  file digests, and row counts. `manifest.sha256` freezes the exact manifest
  bytes.

The historical source rows are calibration/reference material only. They are
not used to tune the parser or scripted responses. The 14 stress rows form the
evaluation split. Paired option orders remain in the same split. The scorer
uses all 14 rows as its denominator: final abstentions, invalid outputs, and
wrong selections count as incorrect, while invalid and abstention counts are
also reported separately.

## Validation

From the repository root:

```bash
python3 scripts/typed_decisions/cj15_fixture.py
pytest -q scripts/typed_decisions/test_cj15_fixture.py
```

This is a small parser/fallback conformance screen, not a semantic-quality or
calibration benchmark. It does not establish live Jev latency, price, or
accuracy. Those require the separately gated inference tasks.
