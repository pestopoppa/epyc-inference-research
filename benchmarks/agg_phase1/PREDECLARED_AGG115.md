# PREDECLARED_AGG115 — recover the `agg_pipeline.v2` long_term_debt regression vs AGG-1.4 r3 (step 1, offline)

Written 2026-11-16 by autopilot-backlog-session BEFORE any fix result was computed. Box: root
`handoffs/active/knowledge-pipelines.md` AGG-1.15. No model, no compute row in step 1.

## Question

AGG-1.11 step 2's gap diagnosis (`/mnt/raid0/llm/tmp/agg111s2-cpu-20261116/out/result.cpu.json`, `gap_diagnosis`)
found 6 FY2022 long_term_debt cells HIT on the AGG-1.4 r3 store and not HIT on the AGG-1.12 r2 (`agg_pipeline.v2`)
store: RETRIEVAL 3 (LOCKHEEDMARTIN, AMD, MGMRESORTS), CANONICAL 3 (PEPSICO, COSTCO, JOHNSON_JOHNSON). Which
offline-checkable, opt-in change recovers or explains each, with zero regressions?

## Pre-fix mechanism observation (read before any fix was scored; it shapes the candidates)

* The AGG-1.12 ingest planned `facts_typed.SCHEMA_FIELDS` (8 fields) per document (`agg112.doc_plans`,
  `finish_doc(..., ft.SCHEMA_FIELDS)`). `long_term_debt` is a typed SLOT (every reply carries it) but never a
  planned field, so `keep_slot` drops every long_term_debt row of every window. The r2 store holds 0 L4
  long_term_debt facts for all 6 documents (only L3 rows).
* The "CANONICAL" label is therefore a gap-diagnosis misnomer: its branch fires when the gold row is in a saved
  reply and not in the r2 facts table; nothing reached `select_canonical`. Whether canonical.v2 would pick the
  right row once the row is kept is measured, not assumed.
* MGMRESORTS gold 7432.817 USD_millions is printed as `7,432,817` (USD thousands). `agg111_triage.num_re(7432.817)`
  may not match that form, so "no gold window in the doc" may be a regex artefact; checked by a thousands-form
  search.

## Fix candidates (all opt-in; every default stays byte-identical)

* **H1 `keep_slot` harvest of unplanned slots** (`facts_typed`, opt-in): a window's rows for a SLOT that is not a
  planned field of the document are kept (no extra call: the slot is already in every reply). Planned fields keep
  today's `keep_slot`. Replayed offline from the saved `t_window` records of the AGG-1.12 ingest (903 windows) into a
  COPY of the r2 store, and from the AGG-1.11 step 2 records into copies of its work stores.
* **H2 plan long_term_debt** (ingest fields = SLOTS): adds long_term_debt union windows; windows not already called
  need new L4 calls (step 2 only, never run here). Offline: the count of new windows per document.
* **R retrieval variants**, rank of the gold window per RETRIEVAL cell under (a) the current union windows
  (top-6 desc query + top-3 `FIELD_QUERY`), (b) union + a long_term_debt-specific query (statement-label wording:
  "long-term debt net of current portion excluding current portion total long-term debt"), k_field 3 and 5,
  (c) `l1_passage_bm25.v2` numeric re-rank (orchestrator `69f30b17`) over the same windows and query.
  Windows per unit (distinct windows per document) reported for each.
* **C canonical.v3** (opt-in, v2 byte-identical): only if, after H1, a cell's gold row is kept and
  canonical.v2 still picks another candidate.

## Measures and the no-regression rule

* The 25 AGG-1.4 backfill cells (M1, `agg1314_zero_cell_diag` rule) on the r2 store copy: before / after.
* The 211 FY2022 cells (`agg112.readings`: raw M1 and canonical) on the r2 store copy, canonical re-run under
  `agg_pipeline.v2`'s rule: **0 of the 211 may regress** on either measure vs the unmodified r2 store.
* The 49 AGG-1.11 cells (24 agg13 + 25 agg14; M1 and canonical) on copies of
  `/mnt/raid0/llm/tmp/agg111s2-cpu-20261116/out/store_agg1{3,4}.sqlite`: **0 of the 49 may regress**.
* A candidate with any regression is rejected, not tuned on the regressed cell.

## Stop rules (from the box)

Gate-class futility-only. Stop-on-answer when the per-cell before/after table and (if needed) the step-2 re-run
job are written. Stop-on-futility if the offline fixes recover 0 of 6 and the pilot fills 0 of 3.
IN-SAMPLE, POST-HOC: the 6 cells are the cells the design came from; any gain on them is labelled so.
