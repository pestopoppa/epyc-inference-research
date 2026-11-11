# AGG-1.4 predeclared - amendment 1 (2026-11-11, written BEFORE any AGG-1.4 data and before the compute row was published)

Amends `PREDECLARED_AGG14.md` (sha256 `e27b4dcb212dd2692f69e9247a4ae923a1d43a8741eb724ae287df4bd63c038b`, committed and
pushed as research `81642408c`, **not edited**). Clarifications found while implementing the runner, with no model call and
no AGG-1.4 data; no gate threshold moves. The amendment's own sha256 is recorded in the compute row and the run spec.

1. **G3 snapshot columns.** The sha256 over the earlier L4 facts covers (`fact_id`, `doc_hash`, `field`, `value`, `unit`,
   `quote`, `audit_status`, `period`, `page`), a superset of the original list. G3 additionally requires the 224 AGG-1.1
   units to be present as seeded cache hits (8 fields x 28 documents) and the counts of `documents`, `chunks`, L3 facts and
   earlier-field L4 facts to be identical before and after.
2. **G4 receipt checks.** Besides `path == "sql"`, `llm_called == false`, `backfill.trigger == true`, `fields ==
   ["long_term_debt"]` and the 12 documents, the receipt must have `path_label == "sql+backfill"` and `schema ==
   "decision_receipt.v1"`. A receipt that fails stops the row at once, before any model call.
3. **Where the trigger lives.** `planner.question_fields`, `planner.question_period`, `planner.backfill_plan` and the
   `plan(..., backfill=)` receipt block are orchestrator commit `6bf51c12e606` (the same commit as AGG-1.3's supersession
   ledger). The job's `orch/` is that commit's `src/` tree, sealed by `SOURCE.sha256`. The L4 stage files are byte-identical
   to the AGG-1.1 pin (checked by the reuse test in preflight).
4. **Field description is injected, not edited.** `phase1_arms.l4_unit` reads `A.METRICS[field]`; the runner adds the key for
   the duration of one unit and restores the table (tested), so the AGG-1.1 `l4_unit` function and the Phase 0 files stay
   byte-identical (their sha256 pins are asserted in preflight).
5. **Informational deterministic check.** After the backfill the canonical table is refreshed and the highest FY2022
   `long_term_debt` row is read with a plain SQL statement (no model) and compared with the gold argmax; reported, not gated.
6. **Smoke run.** `FILLER_SMOKE=1` runs preflight, the receipt and ONE real L4 unit (`l4:long_term_debt:COSTCO_2022_10K`),
   then exits 0; the main run resumes from there.
