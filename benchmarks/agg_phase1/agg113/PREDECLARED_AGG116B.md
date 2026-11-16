# PREDECLARED AGG-1.16b — agg_pipeline.v3 out-of-sample long_term_debt gain (pre-data)

Written and committed 2026-11-16 together with `ltd_golds_agg116b.csv`, BEFORE any v2/v3 store value was read.

**Question.** On the 10 AGG-1.13 documents, does `agg_pipeline.v3` (keep_slot.harvest.v1 + canonical.v3) recover
long_term_debt cells that `agg_pipeline.v2` misses, without breaking any cell?

**Golds.** `ltd_golds_agg116b.csv`: long_term_debt FY2022 and FY2021 per document, derived from the filing text only
(pypdfium2 page text; page, line, verbatim quote). AGG-1.4 convention (PREDECLARED_AGG14.md): FILLED iff the consolidated
balance sheet prints a line labeled "long-term debt" (admitted qualifiers); otherwise EXCLUDED with the reason. USD millions.
7 documents x 2 years = 14 filled cells; Intel, Oracle, Ulta Beauty = 6 EXCLUDED cells.
Limitation: pypdf (the second extractor) is not installed in any env on the host; only pypdfium2 was used.

**Measures (M1 raw and canonical, v3 vs v2, paired on the same cells).**
- Raw: the frozen `agg1314_zero_cell_diag.diagnose_cell` HIT rule over the raw `facts` table (value within TOL,
  period contains the cell's year).
- Canonical: the canonical row for (doc, long_term_debt, period == year) within the same tolerance; v2 store
  re-canonicalised under the v2 rule, v3 store under canonical.v3.
- Stores: the AGG-1.16 offline replay outputs `/mnt/raid0/llm/tmp/agg113v3-replay-20261116/store.v2check.sqlite`
  and `store.v3.sqlite`; no model call.
- Plus the 76-cell AGG-1.13 regression re-check (must be unchanged: raw 71 / canonical 67 for both, 0 regressions).

**Decision rule (report only).** Adoption-ready iff v3 >= v2 on raw and canonical AND 0 regressions (a cell v2 gets
right and v3 does not), on the long_term_debt cells and on the 76 cells. No other threshold; no significance test gates.
