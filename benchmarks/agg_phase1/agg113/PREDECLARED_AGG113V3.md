# AGG-1.16 predeclared: `agg_pipeline.v3` out-of-sample on the AGG-1.13 set (10 docs / 76 cells / 30 questions)

Written 2026-11-16 by autopilot-backlog-session, **before any v3 data**. No v3 store, v3 cell vector, v3 canonical table or v3 answer exists yet.
Operator decision 2026-11-16: "Bundle v3, verify". Bundle both AGG-1.15 fixes as `agg_pipeline.v3`, then check them out of sample on the AGG-1.13
10 documents before v3 becomes the default. `PREDECLARED_AGG113.md` (sha256 `a8b71525...`) and amendments 1 and 2 are **not edited**. Questions, golds,
cells, `m1_hits`, the scorer, `paired_113` and the corpus are frozen and reused unchanged. This is a new file, not amendment 3, because it adds a new arm
(v3) and does not change how any AGG-1.13 arm is run or scored.

## The version under test

Orchestrator `92139534` (`agg_pipeline.v3`): v3 = v2 + `keep_slot.harvest.v1` (`fill_doc(harvest=True)`) + `canonical.v3` (`v2 + long_term_debt_label`,
with v2's `attributable_net_income` guard). L3, the window plan, the prompts, the typed L4 call, `drop_unquoted`, `reconcile` and value-in-quote are all
v2's. The tests in `tests/unit/test_aggregation_pipeline_v3.py` assert that v1 and v2 are unchanged.

## What is known before any data (design facts, not outcomes)

* **The harvest changes no model call.** `doc_call_plan` does not read `harvest`, so the windows called and their replies are v2's. The harvest only
  keeps reply rows of a slot the document did not plan. Under v2/v3 the planned set is `SCHEMA_FIELDS` (8 fields), so the only unplanned slot is
  `long_term_debt`. One count was read from the saved v2 r2 `t_window` records while this design was being written (counts only, no values, no cell
  outcome): 20 reply rows carry `long_term_debt`.
* **`canonical.v3` differs from v2 only on the field `long_term_debt`** (`guard_long_term_debt_label` returns its input unchanged for every other field).
* **The AGG-1.13 frozen inputs contain no `long_term_debt`.** The 76 cells are revenue, net_income, capex, total_assets, operating_cash_flow,
  total_current_assets, total_current_liabilities and dividends_paid (`facts_sheet.csv`). The 30 questions use none of it (`aggregation_questions.json`
  `expr`). The arm C SQL catalog is `pivot_catalog(fields=agg_phase0.METRICS)`, which has no `long_term_debt` column.
* **Prediction (stated before data):** M1 v3 = v2 on every one of the 76 cells (71/76 raw, 67 canonical), with 0 regressions and 0 fixes. Arm C's model
  inputs are byte-identical to v2's on all 30 questions, so the v3 arm C answers are v2's (25/30).
* **What this set cannot do (honest scope):** it contains no `long_term_debt` cell or question, so it **cannot measure the gain** of either fix. It is
  an out-of-sample **non-regression** check only. A gain claim needs `long_term_debt` cells that were not used in AGG-1.15. Listing that need is part of
  the verdict.

## Question

Does `agg_pipeline.v3` hold out of sample on the AGG-1.13 set, with no loss vs `agg_pipeline.v2` on the 76 cells and the 30 questions?

## Method: offline replay first (zero compute); a model run only if it is required

**Step 1, OFFLINE (no model, no server).** Script `agg113v3.py --phase replay`. Inputs, each sha256-recorded:
`/mnt/raid0/llm/tmp/agg113-ingest-v2-cpu-20261114-r2/out/{results.jsonl,store.sqlite,cells.json,result.cpu.json}` and
`/mnt/raid0/llm/tmp/agg113-arms-gpu-20261114-r2/out/{results.jsonl,store.v2.sqlite}`. All are read-only; only copies are written.

1. **Parity (validity gate for the replay).** Replay the v2 typed finish (`keep_slot` -> `drop_unquoted` -> `reconcile`) from every final `t_window`
   record. It must reproduce the stored L4 fact ids of the v2 r2 store exactly (stored-not-replayed = replayed-not-stored = 0). Re-canonicalising a copy
   under v2 must reproduce the stored v2 cell vector (`cells.json`: 71 raw / 67 canonical, every cell equal). If either check fails, the replay is
   **invalid**. Go to step 2 with a real ingest; nothing from the replay is reported as a result.
2. **v3 store (CPU M1).** Replay with `keep_slot_harvest`. Write the facts the harvest adds (it must remove none: base-not-in-harvest = 0) into a COPY
   of the v2 r2 CPU store. Re-canonicalise with `pipeline_spec("agg_pipeline.v3")`. Read the raw and canonical cell vectors with the AGG-1.13 reader
   (`agg112.readings`).
3. **v3 store (arm C).** Insert the same delta into a COPY of the GPU row's post-audit `store.v2.sqlite` (the store arm C answered from). Re-canonicalise
   with v3. The delta facts stay `unaudited` (they were not in the AGG-1.1 audit sample when it ran; re-running the audit is not part of this check, and
   `long_term_debt` is outside every question).
4. **Arm C input identity.** For each of the 30 arm C records in the GPU r2 `results.jsonl`, compare the v2 store and the v3 store on everything the model
   saw: the planner decision inputs (`sql_present`, built layers, docs meta); the pivot catalog (entities, fields) and therefore the SQL prompt and
   grammar; and the rows returned by the recorded `sql_executed` statement (therefore the answer prompt). If all 30 are identical, the v3 arm C answer
   **is** the recorded v2 answer, and it is scored as such and labelled `by_input_identity`. If any question differs, the identity claim fails for that
   question, and the arm C row (step 2) is required.

**Step 2, MODEL (only if step 1 requires it).** Prepared, dry-run validated, never published unless a trigger fires:
* `agg113v3-arms-gpu-20261116`: arm C only (v3 store from step 1.3 + `agg110.v1`), 30 questions, the AGG-1.13 r2 np1 196k recipe unchanged. Trigger:
  the step 1.4 identity fails for at least 1 question.
* A CPU v3 ingest row is required only if step 1.1 parity fails. It would be the AGG-1.13 v2 ingest recipe with `agg_pipeline.v3`, and it needs runner
  support that is not built yet. If parity fails, a dated amendment to this file comes first.

## Metrics and readings (mechanical)

* **Primary, M1.** v3 vs v2 raw on the 76 cells, paired (`paired_113(v3, v2)`). Reported: v3 count, v2 count (71), b, c, regressions (v2 found, v3 not)
  and fixes. **Pass = 0 regressions vs v2.** Canonical v3 vs v2 (67): the same, as a co-primary reading with 0 regressions required.
* **Secondary, P-large.** Arm C with the v3 store vs arm C on v2 (25/30), paired over the 30 questions. Pass = 0 regressions.
* **Pilot / futility.** The pilot cells are the 22 cells of the AGG-1.13 pilot documents (WALMART, ULTABEAUTY_2023, BLOCK). **Stop-on-futility:** v3
  raw M1 on the pilot cells strictly below v2 on the same cells. The verdict is then `agg113v3_futility`: v3 does not become the default, and the rest is
  reported as counts only. Offline, the pilot look is read first and the full vector after it.
* **Stop-on-answer.** The question is answered when step 1 ends with valid parity and an identity verdict for all 30 (or the GPU row's 30 arm C
  answers). Nothing else runs.

## Verdicts

* `agg113v3_holds` (0 regressions on M1 raw, M1 canonical and P-large arm C) **plus** `gain_untested_on_this_set`. v3 is cleared for default **as a
  non-regression**. The gain of the two fixes is still measured only in-sample (AGG-1.15, 6 cells).
* `agg113v3_regresses`: at least 1 regression. v3 is not the default; the regressions are listed.
* `agg113v3_futility`: the pilot look tripped.
* `replay_invalid`: parity failed. Nothing is reported; a CPU ingest follows a dated amendment.

## Estimates (ESTIMATE, not measurements)

| step | estimate | basis |
|---|---|---|
| step 1 offline replay | under 120 s CPU, no model | agg115.py over 305 window records plus store copies of about 43 MB |
| arm C GPU row (only if triggered) | about 900 s | AGG-1.13 r2: 30 answers x about 14 s, plus one cold 27B np1 196k load (about 300 s), plus smoke |
