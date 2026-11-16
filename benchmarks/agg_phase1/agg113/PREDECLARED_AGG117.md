# PREDECLARED AGG-1.17 — residual long_term_debt misses under agg_pipeline.v3 (pre-fix)

Written and committed 2026-11-16 BEFORE any candidate fix was scored. Box: root `handoffs/active/knowledge-pipelines.md` AGG-1.17.
Offline: no model call, no server. Golds frozen: `ltd_golds_agg116b.csv` (research 4916f38bd).

**Question.** For each of the 8 held-out long_term_debt cells `agg_pipeline.v3` (orchestrator d7efa440) misses —
BLOCK FY2022/FY2021 (NO_FACT), FOOTLOCKER FY2022 (NEAR_ROUNDED 452 vs 446), FOOTLOCKER FY2021 (PERIOD_LABEL),
AMCOR FY2021, NIKE FY2021, PG&E FY2022/FY2021 (WRONG_VALUE) — which pipeline step first loses it, and does a
minimal, zero-new-call, opt-in rule recover any of them without breaking any cell?

**Step attribution.** First matching step, agg111s2 order: RETRIEVAL/window (no called window holds the gold line)
-> L3 period -> L4 reply (saved reply text lacks the gold value or mis-labels it) -> value-in-quote -> harvest
(keep_slot / drop_unquoted) -> canonical (a correct raw fact exists but the canonical row is another) -> gold (gold wrong).

**Candidate fix classes (opt-in only; the v3 default is not changed).**
1. Unit: a value printed in thousands inside a quote whose statement says "in thousands" -> scale to millions (MGM class, AGG-1.15).
2. Period: fiscal-year-end in January -> the column headed year N+1 is fiscal N (period re-label from fy_end).
3. Canonical: prefer the "long-term debt" balance-sheet line over current-portion / subsidiary / note lines; prefer the
   value whose quote matches the balance-sheet line over a rounded or near value from prose.
4. Window: add an already-stored but uncalled window to a slot (only if it needs no new call -> otherwise step 2 job).

**Measures.** Per cell, before (v3) and after each fix: raw HIT (frozen `agg1314_zero_cell_diag.diagnose_cell`) and
canonical OK (canonical row for (doc, long_term_debt, fy) within TOL), as in agg116b.py.

**0-regression rule.** A fix is admissible only with 0 regressions (a cell correct before and not after) over all four
sets: the 211 FY2022 cells, the 49 AGG-1.11 cells, the 76 AGG-1.13 cells and the 14 AGG-1.16b long_term_debt cells
(raw and canonical). Any regression rejects the fix; no threshold, no significance test.

**Stop rules (from the box).** Stop-on-answer: all 8 causes filed and, if a fix ran, its re-score filed.
Stop-on-futility: no cause isolable for at least 5 of 8 cells offline, or the 3-cell pilot fixes 0 cells or regresses
any hit. Gate class: futility-only. Label of every fix result: IN-SAMPLE, POST-HOC (the 14 cells are now seen).
