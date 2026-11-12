# AGG-1.9 predeclared - targeted L4 fill of the 26 never-extracted AGG-1.1 cells plus a canonicalisation guard

Written 2026-11-12 by the AGG-1.9 / AGG-1.10 preparation session, before any AGG-1.9 model call and before any
canonicalisation-guard measurement. **This is post-hoc work on AGG-1.1 data and says so:** the AGG-1.1 store and results are
already known, and the AGG-1.8 miss table (research `073c2488f`, `agg18_miss_table.json` sha256
`97ebad89a70e2802d4d1c12a5418389ee1471ae24709c8e747c42ba05dafa193`, produced by `agg18_miss_table.py`) is where the 26 cells,
the 8 canonicaliser-passed-over cells and the 14 store-gap questions were named. The guard acceptance rule below is fixed
before the guard is measured, but its probe set (the 211 verified cells of the AGG-1.1 store) is the same set that
motivated it: any guard accepted here is **in-sample**, with no held-out corpus, and is no evidence that it generalises. It
sits beside `PREDECLARED.md` (AGG-1.1, sha256 `2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af`, not edited)
and imports its frozen scorer, golds, cells, L4 stage and M1 definition unchanged. Its own sha256 is recorded in the compute
row, the job's `precond.sh` and the runner (`run_agg19.py`).

## Question

Box AGG-1.9 (`handoffs/active/knowledge-pipelines.md`): of the 211 verified FY2022 cells the AGG-1.1 aggregation questions
read, 26 were never extracted correctly (20 missing, 5 wrong, 1 unit x1000) and 8 had a correct candidate in `facts` that
the canonicaliser passed over. (1) Does re-running L4 for exactly those 26 (field, document) pairs, now with the AGG-1.3 /
AGG-1.4 fiscal-year and unit cross-check fixes, fill cells? (2) Do two narrow canonicalisation guards (attributable net
income, thousands-unit) repair canonical choices without breaking any retained cell? (3) What does that do to the 14 saved
store-gap SQLs when they are re-run on the filled store?

Scope: the AGG-1.1 28-document store, Qwen3.6-35B-A3B L4 on a CPU quarter, the unchanged L4 stage (`facts_llm.py` sha256
`c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451`, the amended stage of AGG-1.3 / AGG-1.4 amendment 2).
It concludes nothing about other corpora, models or scales, and it does not measure the 27B planner (that is AGG-1.10).

## The 26 cells (frozen input)

`agg19_cells.csv`, derived by `agg19_cells.py` from `agg18_miss_table.json`: `bad_cells` where `correct_candidate_in_facts`
is false and the state is not `spurious` (the Boeing `dividends_paid` spurious cell is a deletion, not a fill). One L4 unit
per distinct (field, document); the AGG-1.1 L4 units for exactly those pairs are NOT seeded as cache hits and run again, the
other 198 of the 224 AGG-1.1 units are seeded as cache hits and are never rerun. The L4 stage is `phase1_arms.l4_unit_fy`
(per-document fiscal year from the corpus manifest, FY2022 here, so the prompt is byte-identical to AGG-1.1's; the L4d
unit-label cross-check of orchestrator `1a408ce4` is what differs from AGG-1.1). **Expectation, stated now:** because the
prompt and the retrieved windows are AGG-1.1's, a unit whose AGG-1.1 failure was a retrieval or reading failure will most
likely fail again; the cross-check can heal only the unit-label failures (the 1 unit x1000 cell and some of the 5 wrong
cells). A fill rate near the AGG-1.4 r3 backfill rate (6 of 9) is NOT expected.

## Canonicalisation guards (deterministic, no model)

`canonical.v1` (AGG-1.1; `facts_canonical` as the AGG-1.1 store holds it) stays the default and is not edited: the new rule
is `canonical.v2`, selected by an explicit argument, so Phase 0 and AGG-1.1 stay reproducible. `canonical.v2` = `canonical.v1`
applied after two candidate filters, each applied per (document, field, fiscal year) group and only when it leaves at least
one candidate:

* **attributable net income** (`field == net_income` only): if any candidate's quote carries "attributable to" (net income /
  net earnings / net loss attributable), only those candidates are kept; among the kept, a value agreed by more of them
  (within the canonical tolerance 0.005) wins before the `canonical.v1` order;
* **thousands-unit**: a candidate whose normalised value is at least 100x the median of the other candidates' normalised
  values (at least two others) is a thousands-unit candidate and is dropped; and a candidate whose normalised value is 1/1000
  of a value that at least two other candidates agree on (within 0.005 of 1000x) is dropped.

**Acceptance rule (fixed before measurement):** each guard is measured separately and together on the 211 cells with the
`canonical.v1` ordering as the control. A guard is ADOPTED only if, on the AGG-1.1 store (before any fill), it fixes at least
one cell and breaks none (a cell ok under `canonical.v1` that is not ok under `canonical.v2`), per field. A guard that breaks
any cell is rejected and reported with its fixed and broken cell lists. Variants tried and their counts are filed in
`agg19_guard_probe.json`, including the rejected ones; the AGG-1.8 tier-flip (prefer `l4_llm` over `l3_textrow`, 7 fixed, 19
broken) is NOT proposed. Cells fixed by neither guard (for example a revenue line chosen from a segment table) are reported as
not addressed, not chased with a third guard in this row.

## Pilot, stops and gate class

* **Pilot:** 5 cells, the sorted cell list (by document, field) at indices 0, 5, 10, 15, 20, at most 300 s. A cell counts as
  FILLED when, after its unit, the raw `facts` table holds a FY2022 candidate that matches the gold within `rel_tol` after the
  unit scale (the AGG-1.1 M1 definition, `phase1_arms.m1_hits`, `M1R_UNIT_SCALE`).
* **Stop on futility:** the pilot fills fewer than 2 of 5 cells. The run files `phase1_gate_tripped` naming the pilot; the
  remaining units are not run. A pilot wall above 300 s or a projected total above 2x the booked estimate is a budget stop
  (re-book), not a gate.
* **Stop on answer:** `result.cpu.json` written with the re-score.
* **Gate class:** futility-only. There is no early GO.

## DONE WHEN (the box, verbatim in content)

1. **M1 over 211 reported before vs after.** Before: the AGG-1.1 store (185/211 = 0.877). After: the filled store. Also
   reported, informational: the per-cell canonical correctness (`canonical.v1` and `canonical.v2`) over the 211.
2. **The 14 saved store-gap SQLs are re-run** offline (no model) on the filled store's `facts_canonical` under `canonical.v2`
   and scored with the AGG-1.8 SQL scorer (`agg18_miss_table.py` `sql_scores`, tolerance 0.005); reported per question, with the
   gold-corrected-cells upper bound (14 of 14 flips, from AGG-1.8) and the AGG-1.1 result (0 of 14) beside it. No model call.
3. **No retained cell regresses.** A retained cell is one whose canonical value is correct under `canonical.v1` on the AGG-1.1
   store (177 of 211) or whose raw facts already hit the gold (185 of 211 by M1). The gate: zero retained cells are correct
   before and not correct after, on both measures, on the filled store under `canonical.v2`. A regression trips the gate and is
   reported cell by cell; the fill is not silently kept.

## Caveats (recorded now)

* **Post-hoc and in-sample** (above). The 26 cells, the guards and the 14 SQLs all come from the same AGG-1.1 data; success
  here shows the AGG-1.8 diagnosis was mechanically right, not that the guards generalise.
* **Fill is not canonical.** A cell can be FILLED (candidate present) and still lose the canonical choice; both are reported.
* **Spurious cell** (Boeing `dividends_paid`, gold NA) is not repaired here; it is listed as not addressed.
* The M1 definition counts any matching candidate in raw `facts` (including a 1000x-mislabelled one only if it matches after
  the scale), exactly as AGG-1.1; it does not say the canonical row is right.
* Estimate: 600 s on a CPU quarter is the box's ESTIMATE, not a measurement. AGG-1.4 r3 measured about 6 L4 calls per unit at
  about 8.2 s per call (about 50 s per unit); 26 units at that rate would be about 1,300 s, so the box's 26 to 40 calls
  assumption may be low. The row is resumable (rc 75), and a deadline stop is a re-grant, not a gate.
