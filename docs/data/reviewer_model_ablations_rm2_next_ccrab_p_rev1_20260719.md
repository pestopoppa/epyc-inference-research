# RM-2.next Reviewer Anchors on C-CRAB P-REV-1

Date: 2026-07-19

Corpus slice: `docs/data/rm2_reviewer_slate_ccrab_p_rev1_matched_row_ids_20260719.txt`
(`24` hard accept controls + `24` hard negatives; `GC-shadow-repair4b.2c`).

Protocol: `P-REV-1`, attestation `MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`.

| Arm | Artifact | FA | FR | AUC | ECE | Brier | Parse | Median row wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 objective-verifier floor | `data/reviewer_model_ablations/rm2-next-a0-objective-floor-ccrab-p-rev1-20260719T205208Z` | 0.0% | 0.0% | undefined | 0.000 | 0.000 | 0.0% | 0.0s |
| A3 same-family GPU heavyweight, Qwen3.5-122B UD-IQ2_M | `data/reviewer_model_ablations/rm2-next-a3-qwen35-122b-iq2-ccrab-p-rev1-20260719T204845Z` | 12.5% | 58.3% | 0.513 | 0.302 | 0.319 | 0.0% | 5.5s |
| A1 status-quo architect self-review, Qwen3.5-122B UD-Q4_K_M CPU | `data/reviewer_model_ablations/rm2-next-a1-architect-statusquo-ccrab-p-rev1-20260719T210513Z` | 45.8% | 41.7% | 0.463 | 0.385 | 0.397 | 0.0% | 41.4s |

Comparator context from the earlier same-slice slate:

| Arm | FA | FR | AUC | ECE | Median row wall |
|---|---:|---:|---:|---:|---:|
| GLM-5.2 UD-IQ2_M CPU baseline | 41.7% | 25.0% | 0.509 | 0.239 | 121.7s |
| Qwen3.6-27B dense Q8 | 54.2% | 16.7% | 0.503 | 0.316 | 6.2s |
| Qwable-v1 IQ4_XS standalone | 54.2% | 45.8% | 0.438 | 0.441 | 2.1s |
| Qwen3.6-27B + Qwable scaffold | 33.3% | 41.7% | 0.659 | 0.315 | 6.5s |

## Interpretation

- A0 is a no-inference objective floor/ceiling for the calibration plumbing, not a deployable reviewer.
- A3 materially reduces false accepts versus GLM (`12.5%` vs `41.7%`) but false-rejects most good patches (`58.3%`), so it is not a production reviewer replacement.
- A3's AUC remains near random (`0.513`), matching the broader finding that this prompt/schema family is not separating the C-CRAB accept/reject boundary cleanly.
- A1 status-quo self-review is parse-clean but worse than GLM on the matched C-CRAB slice (`FA 45.8%`, `FR 41.7%`, `AUC 0.463`, `ECE 0.385`) and is too slow for ordinary reviewer use on CPU (`2164.234s` total wall, median row wall `41.4s`).

Decision-grade conclusion: RM-2.next is complete and does not produce a clean reviewer route. The reviewer/control-plane choice remains a policy decision or requires a new repair hypothesis / RM-3 screening path.
