# GLM-5.2 C-CRAB Accept-Control Filter - 2026-07-18

Purpose: deterministic full-candidate accept-control selection for the next GLM patch-review
confirmation run. This is a row-selection artifact only; it does not make the rows
decision-grade by itself.

## Why This Exists

The targeted n=12 GLM C-CRAB confirmation rejected all six prior hard negatives, but
false-rejected `nearmiss-v1:c-crab:1a06ca93c57f49ac`. Audit found that row is only an
observation-grade merged-patch accept, with no executable oracle and no patch-internal test
evidence. It should not be used as a hard accepted control.

## Recommended Filter

- `source_benchmark == "c-crab"`
- `source_suite == "python"`
- `gold_label == "accept"`
- `gold_source == "merged_pr_accepted"`
- `provenance.clean_control == true`
- `defect_origin == "natural"`
- `ambiguous_tail == false`
- candidate includes at least one test-like path
- candidate includes added test/assert evidence: `def test_`, `class Test`, `assert`,
  `self.assert*`, `pytest.*`, or `with pytest.raises`
- candidate length `<15000` chars
- sort lexicographically by `row_id`, take first `n`

## Primary N=24 Row Ids

```text
nearmiss-v1:c-crab:00710c9c18cd10fb
nearmiss-v1:c-crab:0110087826d99378
nearmiss-v1:c-crab:04d390e945dd768e
nearmiss-v1:c-crab:060c7e12cfd0cb06
nearmiss-v1:c-crab:0735945503ef9330
nearmiss-v1:c-crab:08cafcb6483d8389
nearmiss-v1:c-crab:09584d0209952576
nearmiss-v1:c-crab:0b5adcf2e8a30f49
nearmiss-v1:c-crab:0c6318021a8a500b
nearmiss-v1:c-crab:0e31e881b1af8ab5
nearmiss-v1:c-crab:0f20280ccef865cb
nearmiss-v1:c-crab:10070430d41b73e9
nearmiss-v1:c-crab:12dedf2f36029e2c
nearmiss-v1:c-crab:1600ca8239e2f6e0
nearmiss-v1:c-crab:19fd2cb501691488
nearmiss-v1:c-crab:1a3868334fafed91
nearmiss-v1:c-crab:1a64c956e9fceeda
nearmiss-v1:c-crab:1af8c54719aff460
nearmiss-v1:c-crab:200003ca11cb7699
nearmiss-v1:c-crab:20e3c97d771762dd
nearmiss-v1:c-crab:24fbdf7de1c6fa44
nearmiss-v1:c-crab:28cdbe123c730a18
nearmiss-v1:c-crab:29de0af708959323
nearmiss-v1:c-crab:2b6c409775aad431
```

## Caveat

The corpus lacks hard accept confidence for C-CRAB/Python accepts: all available accepts are
`gold_confidence=observation` and `executable_oracle=null`. The filter above hardens the
selection with patch-internal test evidence, but it is still not a substitute for an
executable oracle or manual label sign-off.
