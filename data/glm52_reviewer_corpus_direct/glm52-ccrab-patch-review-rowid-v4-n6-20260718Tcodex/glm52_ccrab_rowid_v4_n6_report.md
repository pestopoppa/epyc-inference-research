# GLM-5.2 C-CRAB Row-ID v4 Prompt Screen - 2026-07-18

Purpose: test whether the v4 patch-review prompt, which explicitly requires
task/test alignment before approval, fixes the pinned SQLFluff L009 false accept
from the previous C-CRAB row-id screen.

Runner change:

- `DEFAULT_RUBRIC_VERSION` changed to
  `glm52_direct_nearmiss_review_v4+binary_schema+task_test_alignment`.
- The patch-diff prompt now tells the reviewer to identify both the exact task
  behavior fixed and the changed test/assertion that would fail without the fix.
- It also tells the reviewer to reject nearby/pass-only tests and helper/API
  changes not tied to the rule path that caused the reported failure.

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
| Elapsed | 922.310 s |

Row outcomes:

| Row | Gold | GLM | Disposition |
|---|---|---|---|
| `6b710a8f003b6d7f` | accept | approve | Correct. |
| `28cdbe123c730a18` | accept | approve | Correct. |
| `34224c9c98534550` | accept | approve | Correct. |
| `c7b60daa9aa5eadd` | reject | reject | Correct. |
| `74d37c7877c3c037` | reject | reject | Correct. |
| `0e49d06ddc8f2635` | reject | approve | Still false-accepted. |

Interpretation:

- v4 preserved the three accept controls and two existing reject controls.
- v4 did not fix the SQLFluff L009 false accept. The model still treated a
  templated-newline pass fixture and raw-slice helper API change as sufficient
  evidence for a dbt/raw-space failure.
- GLM-5.2 remains patch-review quality-blocked. Do not run P-REV-1 from this
  prompt state.
- The next repair should add a scorer/prompt feature that makes "pass-only
  neighboring test does not reproduce the reported failure" a first-class
  rejection reason, or it should include oracle-provided review notes for this
  hard-negative class.

Cleanup:

- Runner stopped the temporary GLM server after row 6.
- Post-run checks found no `llama-server`, GLM runner, AutoPilot, or KFD
  process.
