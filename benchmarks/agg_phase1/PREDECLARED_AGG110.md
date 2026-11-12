# AGG-1.10 predeclared - planner plan/SQL fixes for derived questions, GPU re-score of the 26 affected questions

Written 2026-11-12 by the AGG-1.9 / AGG-1.10 preparation session, before any AGG-1.10 model call. **This is post-hoc work on
AGG-1.1 data and says so:** the 26 questions, their misses and the defect classes were read from the AGG-1.8 miss table
(research `073c2488f`, `agg18_miss_table.json` sha256 `97ebad89a70e2802d4d1c12a5418389ee1471ae24709c8e747c42ba05dafa193`)
and the saved AGG-1.1 SQL strings; the fixes are designed against those defects, so the 26 are a fit set, not a held-out
test. The result says whether the specific defects are removed on the questions that showed them; it is no estimate of
accuracy on unseen derived questions. It sits beside `PREDECLARED.md` (AGG-1.1, sha256
`2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af`, not edited) and imports its frozen questions, golds,
deterministic scorer and gates unchanged. Its own sha256 is recorded in the compute row, the job's `precond.sh` and the runner
(`run_agg110.py`).

## Question

Box AGG-1.10: 17 AGG-1.1 P-large misses were wrong plans (a ratio, difference or filter answered with one single-field
SELECT) and 9 were broken SQL (3 runaway statements that hit the 160-token SQL cap, and 6 that aggregated a text column, `SUM(field)` or
`SUM(quote)`; the box's "7 truncated at n_predict" counts answers whose finish reason was `length`, a separate call). With
the plan grammar fixed (a pivot view, an entity-conjunct guard) and the SQL `n_predict` sized to the truncations seen, how many
of those 26 questions does P-large answer correctly against the unchanged AGG-1.1 store, at what M4 cost?

Scope: Qwen3.8-27B Q8_0 on the MI210 (v10 GPU kernel store), the AGG-1.1 28-document store (a copy, never mutated; the AGG-1.9
fill is NOT used, the two rows are independent), exactly the 26 questions listed in `agg18_miss_table.json` as
`wrong_plan_derived_or_filtered` (17) or `sql_malformed_or_degenerate` (9). It concludes nothing about the other 32 questions,
other models or corpora.

## Planner versions (reproducibility of Phase 0 and AGG-1.1)

Every answer record carries `planner_version`. `agg11.v0` is the Phase 0 / AGG-1.1 behaviour, untouched and the default of the
library: `phase0_arms.sql_answer`, table `facts_canonical`, `sql_n_predict` 160. `agg110.v1` is the plan this row tests. A
zero-model test asserts that `agg11.v0` still builds the byte-identical SQL prompt and grammar of AGG-1.1 and that the frozen
Phase 0 files keep their pinned sha256. Nothing under `agg_phase0/` is edited.

## The `agg110.v1` plan (deterministic parts have tests; only the SQL text is the model's)

1. **Pivot view.** `facts_pivot(entity, doc_id, period, <one numeric column per canonical field>)`, a TEMP VIEW over
   `facts_canonical` of the corpus (never persisted; the store file is not altered), NULL where a document has no canonical row
   for that field and year. A ratio or a difference is then one expression over one row: `net_income / revenue`.
2. **Plan grammar** (GBNF, structurally bounded): at most 3 select items (`entity`, a numeric expression, or an aggregate
   `SUM|AVG|MIN|MAX(expr)` / `COUNT(*)` over a NUMERIC expression only, never a text column); an expression is a field column
   or `field op field` with `op` in `/`, `-`; at most 5 `WHERE` conjuncts of the forms `period = '<year>'`, `expr cmp number`
   (digits bounded: at most 12 integer and 6 fraction digits) and `entity != '<name>'` (names from the corpus entity list); the
   only `ORDER BY` is an expression with `ASC|DESC`; `LIMIT` at most 2 digits. There is no `entity =` form.
3. **Entity-conjunct guard.** Any statement with 3 or more `entity =` conjuncts is rejected before execution and re-asked ONCE
   with the reason; a second failure is recorded as `sql_invalid` and answered from no rows (counted, never hidden). The
   grammar already makes the form unreachable; the guard is the deterministic mirror and also guards the `agg11.v0` grammar.
4. **Null guard.** Every field column the statement references gets an `IS NOT NULL` conjunct before execution (a company
   lacking a needed field is excluded, which is the question text's own "excluded" rule); without it SQLite ranks NULL first
   under `ORDER BY ... ASC`.
5. **SQL `n_predict`.** `agg110.v1` uses 256 (AGG-1.1: 160). The AGG-1.1 SQL calls that hit the 160 cap were runaway
   repetitions (three statements, `agg15`, `agg16`, `p1_03`, of 541-566 characters of repeated `entity = ...` or `value, ...`;
   the longest other AGG-1.1 statement was 259 characters), not long legitimate statements; the bounded grammar removes the runaway, and 256 leaves headroom over the longest
   legitimate `agg110.v1` reference statement of the 26, which must be at most 60% of the cap (checked offline by
   `agg110_ceiling.py`; a longer one fails preflight). The answer call (1,536 tokens) and the answer prompt are unchanged.
6. **Routing record.** `planner.sql_shape(question)` returns `pivot` when the question needs two or more fields or carries a
   derived/filter cue, else `single`; recorded per row and reported over all 58 questions (zero model). The 26 re-scored
   questions all run `agg110.v1` regardless of route (the planner version is the factor under test); the routing report says
   how many of the currently-correct questions would also take the pivot plan, which is the exposure a later regression run
   would have to cover.

## Frozen derived files (computed before any model call)

`agg110_ceiling.json` (by `agg110_ceiling.py`, no model): for each of the 26, a reference `agg110.v1` statement compiled
from the golds' `op / expr / where / exclusions` and executed on a read-only copy of the AGG-1.1 store: the ceiling (is the
answer correct when the plan is right, given the store's bad cells?), the longest statement, and the AGG-1.1 reference M4
(median 2 calls / 644.5 prefill over the same 26). The scorer is the frozen `phase0_arms.score_agg`. The ceiling is reported
beside the accuracy; it is not a gate.

## Pilot, stops and gate class

* **Pilot:** 6 questions, at most 600 s: the first 6 questions of the ceiling-true set in the fixed stratum order
  `continuity-malformed, continuity-plan, margin/top3, where-filter sum, current-ratio count, dividends where` (frozen in
  `agg110_pilot.json` before the run; if a stratum has no ceiling-true question the next stratum's second is used). A pilot
  question counts as FIXED when its answer is correct under the frozen scorer (all 26 were wrong in AGG-1.1).
* **Stop on futility:** the pilot fixes 0 of 6. The run files `phase1_gate_tripped`; the other 20 are not run. A pilot wall
  above 600 s is a budget stop (re-book), not a gate.
* **Stop on answer:** all 26 scored, `result.gpu.json` written.
* **Gate class:** futility-only; no early GO.

## DONE WHEN (the box)

A predeclared re-score on the 26 is filed with:

1. **Before/after accuracy** on the 26 (before: 0 of 26 by construction; after: correct / 26 under the frozen scorer), split
   by class (wrong plan 17, malformed SQL 9) and by ceiling (ceiling-true vs store-limited), with the SQL each question
   produced and the planner version on every record.
2. **M4 unchanged within bounds:** median calls and median prefill per P-large question over every question that ran,
   against the AGG-1.1 bounds (median calls <= 3, median prefill <= 2,510; kill median calls > 8) and beside the AGG-1.1
   reference medians over the same 26. M4 is reported over whatever ran; if only the pilot ran, it says so.
3. The AGG-1.1 `agg11.v0` path reproduces (the zero-model identity test in the row's preflight).

No accuracy threshold is predeclared: any outcome is filed. The verdict is `agg_1_10_done` when all 26 are scored and M4 is
within bounds; an M4 trip files `phase1_gate_tripped` naming M4.

## Caveats (recorded now)

* Post-hoc and in-sample (above); the AGG-1.1 store still has bad cells, so some of the 26 cannot be right whatever the plan
  (the ceiling file says which); the AGG-1.9 fill is the fix for those and is a separate row.
* The answer call has finish reason `length` on 13 of the 17 wrong-plan misses in AGG-1.1 because the rows it was shown did
  not answer the question; the answer prompt is unchanged here, so a residual answer truncation is possible and is reported.
* Re-scoring the other 32 questions is not required; M4 and accuracy are over the 26 (plus any extra questions run), not the
  58, and AGG-1.1's 18 correct answers are not re-verified under `agg110.v1`.
* Estimate: 1,800 s on the GPU window is the box's ESTIMATE, not a measurement (26 questions x 2 to 3 calls; AGG-1.1 P-large
  answers took about 4 to 60 s each).
