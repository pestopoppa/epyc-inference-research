# AGG-1.6 predeclared - union-window retrieval and typed multi-field output, A/B on the AGG-1.11 target cells

Written 2026-11-12 by the AGG-1.6 preparation session, before any AGG-1.6 model call and before any C1 or T datum exists.
**This is post-hoc and in-sample, and says so.** The 38 target cells are the L4 misses that the AGG-1.11 step-1 triage
(`agg111_triage.json`, research `2a9e7fb47`, sha256 `f8a2f0ff560ed61d8d97acda193a7a4eb0c607be47e63d435261f05ab4dd4291`, pinned in the runner as `agg16.FROZEN`) found on the very AGG-1.1 / AGG-1.3 / AGG-1.4
data the arms are run on, and the field-specific retrieval queries were written after reading those misses. The result
estimates how many KNOWN misses the two changes recover; it is not evidence that they generalise to unseen cells, documents
or models, and it is not a gate on AGG-1.1 (the AGG-1.1 gates, golds, scorer and L4 stage are not edited: `PREDECLARED.md`,
sha256 `2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af`, stays the identity of that run).

## Question

Box AGG-1.6 (`handoffs/active/knowledge-pipelines.md`), re-scoped by the AGG-1.11 triage: of the 39 L4-missed cells, the
dominant causes were FORMAT (the L4b reply is cut at `n_predict` 256; about 0.097 of fresh L4b replies) and RETRIEVAL (the
gold window is outside the top-6). (1) Does retrieving the union of the current top-6 windows and a field-specific top-3
fill the RETRIEVAL cells? (2) Does a larger `n_predict` (512), or a typed multi-field output with capped rows and bounded quote
length, remove the truncation and fill the FORMAT cells? (3) What does each cost in calls and prefill per unit, and does either
change any cell that is not a target?

Scope: the AGG-1.1 and AGG-1.3 r3 stores (read-only seeds), Qwen3.6-35B-A3B Q8_0 L4 on a CPU quarter, the unchanged L4 stage
(`facts_llm.py` sha256 `c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451`: prompt prefix, `ROW_GBNF`, quote-in-chunk
check, L4d reconcile and unit cross-check). It concludes nothing about other corpora, models or the 27B planner.

## Targets (frozen input)

From `agg111_triage.json`: every cell whose class by the L4 pipeline alone (`l4_class`, whatever its primary class) is FORMAT (21 cells) or RETRIEVAL (17 cells),
deduplicated on (document, field, fiscal year): 38 cells (5+5 on the AGG-1.3 r3 store at FY2021, 3 on the AGG-1.4 r3 backfill field
FY2022, 25 on the AGG-1.1 store at FY2022). The same (document, field) can appear at FY2021 and FY2022 (MICROSOFT capex and
dividends_paid, JOHNSON_JOHNSON capex): those are different cells with different golds and different prompts. The **14 primary
FORMAT cells** are the targets whose triage `primary` class is FORMAT (5 on AGG-1.3, 9 on AGG-1.1). Cells whose primary class is L3
(11 of the 38, recoverable with no model by `l3_period.v2`) stay in the target set: the question is what L4 can do for them.
Gold values are the verified `agg111_triage.json` golds; no gold is edited.

## Arms

* **C0 (reference, never re-run).** The AGG-1.1 / AGG-1.3 r3 / AGG-1.4 r3 L4 unit of each target cell as saved in that run's
  `results.jsonl` (top-6 windows, `n_predict` 256, one extraction call per admitted window). Its calls, prefill, parse failures and
  cut replies come from the saved receipts; its target-cell fill is 0 of 38 by construction of the triage (the L4 pipeline alone
  produced no right fact), recomputed from the triage cell records.
* **C1.** The same prompt, grammar, admission (L4a) and L4b/L4d stage as C0 (`LazyFieldBackfill`, `hi` 0.6, `lo` 0.2), with
  (i) the candidate windows replaced by the **union** of the current top-6 (query `<field words> <description>`) and a
  field-specific top-3 (`agg111_triage.FIELD_QUERY[field]`, sims normalised to that query's best window; a window in both keeps the
  current sim), and (ii) `n_predict` 512. One unit per target cell.
* **T.** The union windows of C1, no L4a gate (the union IS the admission), and ONE call per distinct window: a single typed
  object with one slot per schema field (the 8 `METRICS` fields plus `long_term_debt`), each slot an array of AT MOST 2 rows
  `{period, value, unit, quote, confidence}`, the quote bounded to 100 characters and verbatim from the window, the period to 20
  characters, whitespace between tokens to one space; `n_predict` 1024. Every row goes through the unchanged
  `facts_llm.extract` (quote-in-chunk check, declared-scale cross-check) and `facts_llm.reconcile`. A window shared by several
  target cells of one document is called once; its calls and prefill are charged to the first cell that needed it. Technique A of
  `SCHEMA_NOTE.md`; techniques B and C are not part of this row.

Same model, same server flags, `temperature 0`, `top_k 1`, `seed 0`, thinking off, as the AGG-1.1 CPU row.

## Metrics (per arm, over the 38 target cells unless stated)

1. **Truncated-or-unparseable rate** = extraction calls whose reply was cut (`finish_reason == length`) or did not parse as the
   expected JSON, over all extraction calls of the arm. C0: the saved `unparseable` receipt notes over the saved L4b calls of the
   target units (the saved replies are not available; a cut reply is exactly an unparseable one under the constrained grammar).
2. **Target-cell fill** = cells with a reconciled fact for (document, field) whose period names the cell's fiscal year and whose
   value is within 0.5 percent of the gold in USD millions (the AGG-1.1 M1 hit rule, per cell and fiscal year); reported for all 38,
   for the 14 primary FORMAT cells, and for the 13 primary RETRIEVAL cells.
3. **Calls per unit and prefill per unit** = all model calls (gate and extraction) and `prompt_n` tokens of the arm over the 38
   cells, plus the same split per primary class.
4. **Fixes and breaks on non-target cells touched.** A non-target cell is a verified gold cell of the store (the 211 FY2022 cells of the
   AGG-1.1 store; the 24 filled FY2021 cells of `agg13_cells.csv` for the three documents the AGG-1.3 r3 store replaced)
   that is not one of the 38 and for which the arm produced at least one fact for (document, field, fiscal year).
   "Touched" cells are scored under `canonical.v1` on (the store's saved facts) versus (the saved facts plus the arm's facts); a fix
   is not-ok to ok, a break is ok to not-ok. Reported per arm; any break is listed by cell.

## Pilot and stops (fixed now)

* **Pilot cells (exactly 5):** FY2022 `MICROSOFT_2022_10K` `capex`, `dividends_paid`; FY2022 `COSTCO_2022_10K` `net_income`,
  `total_assets`, `total_current_liabilities` (all five are L4-class FORMAT cells; two documents, so the typed arm's shared windows are
  exercised; chosen to keep the pilot inside its wall bound: the offline plan gives 34 C1 windows plus 24 distinct T windows, about
  58 calls). Run C1 then T on those 5 cells, wall-clocked as the sum of the unit walls.
* **Pilot futility:** an arm that fills fewer than 2 of the 5 pilot cells does not continue; if neither arm fills 2, the row stops
  (`phase1_gate_tripped`, futility, gate-class futility-only). The pilot is not a conclusion about the arms.
* **Budget stop (re-book, not a gate):** pilot wall above 600 s, or the projected full wall (pilot wall x 38 / 5) above twice the
  booked estimate. The booked figure is **4,200 s, an ESTIMATE, not a measurement**: the offline plan counts 288 C1 calls (280
  union windows plus 8 gate calls) and 201 distinct T windows; at the measured 8.2 s per call (about 6 calls per unit in the
  AGG-1.1 and AGG-1.4 receipts) that is 4,010 s, plus about 120 s for the server start. The run replaces it.
* **Stop-on-answer:** the job ends when `result.cpu.json` is written (all continuing arms over all 38 cells, the metrics above, the
  verdict). Resumable per unit; a re-grant repeats no finished unit. `FILLER_DEADLINE` / stopfile give exit 75.

## DONE WHEN (the AGG-1.6 sub-bullet, fixed now)

The winning arm is the arm with the most primary-FORMAT cells filled (ties: lower truncated-or-unparseable rate, then fewer calls per
unit). The row is **MET** if the winning arm has a truncated-or-unparseable rate of at most 0.01 over all its extraction calls AND at
least 10 of the 14 primary FORMAT cells filled. Otherwise it is **NOT MET** and the numbers are filed as they are. Neither verdict
adopts anything: adoption of union windows or of the typed output into the ingest path is a separate, later decision that needs a
held-out check. The verdict is never used to edit AGG-1.1.

## Expectations, stated before data (so a result can contradict them)

C1 should fill most of the 14 primary FORMAT cells (the replies were cut at exactly 256 tokens) and most of the RETRIEVAL cells
whose gold window is in the union (the in-sample probe found the gold window in the union for 206 of 211 AGG-1.1 cells). T's
truncation rate is the open question (nine slots can run long); its calls per cell should fall below C1's where several target
cells share a document. The typed arm may fill fewer cells than C1 because a window answers several fields at once.
