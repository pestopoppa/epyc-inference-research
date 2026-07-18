# GLM-5.2 C-CRAB Row-ID Patch-Review Screen - 2026-07-18

Purpose: test the repaired GLM-5.2 patch-review prompt on a pinned, balanced
C-CRAB control slice instead of a seed-dependent pool sample.

Runner changes: `scripts/benchmark/glm52_reviewer_corpus_direct_runner.py` now
supports repeated `--row-id` and `--row-ids-file` selection so manually audited
or provisional controls can be rerun exactly.

Command shape:

```bash
python3 scripts/benchmark/glm52_reviewer_corpus_direct_runner.py \
  --execute \
  --domain code \
  --gold-confidence multi_oracle,observation \
  --source-benchmark c-crab \
  --source-suite python \
  --row-id nearmiss-v1:c-crab:6b710a8f003b6d7f \
  --row-id nearmiss-v1:c-crab:28cdbe123c730a18 \
  --row-id nearmiss-v1:c-crab:34224c9c98534550 \
  --row-id nearmiss-v1:c-crab:c7b60daa9aa5eadd \
  --row-id nearmiss-v1:c-crab:74d37c7877c3c037 \
  --row-id nearmiss-v1:c-crab:0e49d06ddc8f2635 \
  --band p12000_tk16384 \
  --max-field-chars 24000 \
  --max-tokens 320 \
  --temperature 0 \
  --seed 42 \
  --trace-logs \
  --output-dir data/glm52_reviewer_corpus_direct/glm52-ccrab-patch-review-rowid-n6-20260718Tcodex
```

Result:

| Metric | Value |
|---|---:|
| Rows | 6 |
| Accept controls | 3 |
| Reject controls | 3 |
| Parse failures | 0 |
| False accepts | 1 |
| False rejects | 0 |
| FA rate | 33.3% |
| FR rate | 0.0% |
| Elapsed | 912.505 s |

Row outcomes:

| Row | Gold | GLM | Evidence summary |
|---|---|---|---|
| `6b710a8f003b6d7f` | accept | approve | Lazy DB connection moved behind `ensure_connection()` with tests. |
| `28cdbe123c730a18` | accept | approve | Bool `qcut` path coerces before binning and has tests. |
| `34224c9c98534550` | accept | approve | Digit-starting identifiers are quoted where required. |
| `c7b60daa9aa5eadd` | reject | reject | Logistic regression class-weight dtype mismatch risk. |
| `74d37c7877c3c037` | reject | reject | Patch tests subtraction while task requests addition. |
| `0e49d06ddc8f2635` | reject | approve | False accept; raw-slice SQLFluff L009 patch still needs harder negative-evidence prompt or label review. |

Interpretation:

- The repaired prompt is a real improvement over the matched C-CRAB n=24 run
  that over-approved (`FA=91.7%`), and it avoids the previous small-slice
  false-reject failure on the three selected accept controls.
- GLM-5.2 is still not patch-review role-ready. The remaining failure is an
  unsafe false accept on a multi-oracle reject, so P-REV-1 should not run yet.
- Next useful work is either a targeted hard-negative prompt repair around
  superficially plausible raw-slice/style patches, or independent label review
  of `0e49d06ddc8f2635` before expanding the pinned slice.

Cleanup:

- Runner stopped the temporary GLM server after row 6.
- Post-run process check found no `llama-server`, GLM runner, AutoPilot, or KFD
  process.
