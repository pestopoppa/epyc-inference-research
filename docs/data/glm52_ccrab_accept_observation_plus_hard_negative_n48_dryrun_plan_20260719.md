# GLM-5.2 C-CRAB Accept+Hard-Negative Dry-Run Plan

- Schema: `glm52_reviewer_corpus_direct.v1`
- Date: `2026-07-19`
- Runner: `scripts/benchmark/glm52_reviewer_corpus_direct_runner.py`
- Plan path: `data/glm52_reviewer_corpus_direct/glm52-gc4b-acceptobs-plus-hardneg-dryrun-20260719Tcodex/plan.json`
- Row ids: `docs/data/glm52_ccrab_accept_observation_plus_hard_negative_n48_dryrun_row_ids_20260719.txt`
- Hard-negative filter: `docs/data/glm52_ccrab_hard_negative_n24_filter_20260719.json`
- Mode: `dry-run`
- Live inference launched: `false`

## Plan Shape

- `execution_allowed`: `true`
- `inventory.status`: `ready`
- `n_selected`: `48`
- `selected_label_counts`: `accept=24`, `reject=24`
- `representation_counts`: `c-crab|python|no_scoring_method=48`
- `candidate_payload_scope_counts`: `full_candidate=48`
- `candidate_chars`: min `764`, p50 `4686`, max `14934`
- `refusal_reasons`: `[]`

## Gate Status

This plan proves the GLM C-CRAB patch-review runner can accept a homogeneous 24-accept / 24-reject
full-candidate slice once signed accept controls exist. It is not live-run permission: the accept
side in this dry-run uses observation-only `merged_pr_accepted` row ids from
`glm52_ccrab_accept_control_n24_row_ids_20260718.txt`.

Do not execute this row file live until `glm52_ccrab_accept_control_signoff_status_*.json` reports
`decision_grade=true` and a regenerated row-id file uses signed hard-accept controls.
