# AGG-1.12 predeclared - AMENDMENT 1 (2026-11-13): re-run the GPU answer leg on the AGG-1.14 corrected store

Written 2026-11-13 by the AGG-1.12 r2 preparation session, **before any model call of the re-run** (the job dir, its row and its out/ do not
exist yet when this file is written). `PREDECLARED_AGG112.md` (sha256
`22479709afbe5110115f6b2d1a951e799b8bb8752fabc5c51681c4aa30f59c1c`) and `agg112_reference.json` (sha256
`f23bcbb33bb7d71718d4e0d68ce57aad8813226a5668500b5a8ff1104ead36ae`) are NOT edited and stay in force; this amendment changes only the
store the GPU leg reads, the orchestrator pin, and where the no-regression rule reads its cells. Its own sha256 is recorded in the r2 row's
`precond.sh`. It remains **post-hoc and IN-SAMPLE**, as the parent says.

## The regression and its cause

The first AGG-1.12 GPU leg (`/mnt/raid0/llm/tmp/agg112-arms-gpu-20261112`, orchestrator `4d0265cd`) filed `agg_1_12_not_met`: gates held (M4 median 2
calls / 790.5 prefill for `agg110.v1`, 2 / 606.5 for `agg11.v0`), P-large `agg110.v1` 39 of 58 = 0.672 (AGG-1.1 18 of 58; 22 gained, 1 lost, paired
z 4.38), `agg11.v0` on the v2 store 26 of 58, and **one canonical cell broke against AGG-1.1: JOHNSON_JOHNSON dividends_paid**. Cause (AGG-1.14):
the typed L4 stage stored value 13,895.3 with the quote "Cash dividends paid were $4.45 per share in fiscal year 2022", a value that does not
appear in its own quote, and `canonical.v2` chose it over the correct 11,682 that was also in the store. The typed stage lacks the AGG-1.1
`facts_llm` rule that a value must be printed in (or be a declared-scale restatement of) its verbatim quote. The AGG-1.14 fix is orchestrator
`737bf0535d1abecba154a2a22fbba0d910821ed8` (`facts_typed.drop_unquoted`, reusing `facts_llm`'s printed-number check; v1 untouched), applied
inside the typed stage's `finish_doc`.

AGG-1.14 (research `1a6ad950b6e3254c9958fdc2dc264eb2b22220c5`, `agg114_rescore.py`) replayed the saved `t_window` records of the CPU row through
the corrected stage OFFLINE. A rule-OFF baseline replay reproduced the saved store's 345 L4 facts and its 208 raw / 202 canonical readings first.
Of the 445 rows entering the stage, 21 violate the rule and are dropped (74 of the 693 rows the model returned violate it); L4 facts 345 -> 324.

## The operator decision (2026-11-13)

"Rerun the GPU answers first": re-run the AGG-1.12 GPU answer leg on the corrected store so every answer comes from the model, then judge
AGG-1.12 on that run. The AGG-1.14 offline P-large re-score (40 of 58 for `agg110.v1`, 27 of 58 for `agg11.v0`) is a MECHANICAL reading of
changed SQL rows by a deterministic renderer, not a model answer, and is not the AGG-1.12 result; it is recorded only as the expectation the
re-run can be compared to (not a gate).

## The corrected store, built offline

`/mnt/raid0/llm/tmp/agg112-r2-store-20261113/store.corrected.sqlite` (read-only file; sqlite backup of `fixed_replay.sqlite` from
`agg114_rescore.py`'s rule-on replay of `/mnt/raid0/llm/tmp/agg112-ingest-cpu-20261112/out/{results.jsonl,store.sqlite}` under orchestrator
`737bf0535`, journal mode DELETE; the saved stores were only ever copied):

* sha256 `d0cfb07634c699534bb956b9669e0e7e4a2b98b654335956de11d144f2435c5c`;
* 324 `l4_llm.v1` facts, 17,520 canonical rows; **raw M1 208 of 211, canonical 203, 0 raw and 0 canonical cells broken against AGG-1.1
  (185 raw / 177 canonical), 158 of 158 held-out cells right on both measures**; JOHNSON_JOHNSON dividends_paid canonical = 11,682, right;
* re-measured from the final file (a copy of it, never the file itself) with the runner's own readings: 208 / 203 / 0 / 0.

`result.cpu.corrected.json` (same directory; sha256 `ad4b578c51a5da19df81f175de6eaace08f36b027a8268d1da41455ff539af69`) is the saved CPU row's
`result.cpu.json` with every reading of the fact table (M1, canonical, regressions, strata, the AGG-1.9 report) re-read from the corrected store by
`agg112_r2_cpu_result.py` (this commit; deterministic, two runs byte-identical, no model). **M3 and every ingest-cost field are the saved CPU row's,
unchanged**: the quote rule drops rows after the calls were made; it adds and removes no model call. The GPU runner takes M1 / M3 / regressions
from this file (`agg112.verdict`), so the no-regression rule below is evaluated over the corrected store's cells and not over the saved store's.

## Rerun scope (unchanged from the parent)

* The same 58 verified questions (frozen `aggregation_questions.json` / `golds.jsonl`), the same two planners (`agg110.v1` primary, then `agg11.v0`), on
  ONE store (the corrected one, `canonical.v2` attributable net income, rebuilt once at run start), the same frozen scorer (`phase0_arms.score_agg`),
  the same model and recipes (Qwen3.8-27B-Q8_0 + DFlash2 on the v10 GPU kernel store: np4 recipe for the L4c audit sample, then the np1 196k recipe
  for P-large). The audit sample is recomputed from the corrected store. EVERY answer is a fresh model answer: out/ is new, nothing is reused from the
  first leg.
* The same gates, any trip -> `phase1_gate_tripped` naming it: M1 and M3 from the CPU result (M1 found >= 185: 208; M3 inside the Phase 0 kill
  bounds: the saved row's measured M3); **M4 median calls <= 3 and median prefill <= 2,510 over the 58 for BOTH planners (kill 8 calls)**.
* The same mechanical verdict: `phase1_gate_tripped` | `agg_1_12_not_met` (a cell right in AGG-1.1 is not right now, raw or canonical, **evaluated on
  the corrected store**; listed with its held-out / touched stratum) | `agg_1_12_done`. P-large accuracy and the planner and pipeline effects (paired
  z = (b - c) / sqrt(b + c)) are FILED against AGG-1.1 with their direction, not gated. The rule is unchanged: only the store whose cells it reads is.
* Determinism: temperature 0, top_k 1, seed 0, thinking off, `-np 1`; the first leg and the re-run can still differ by one or two questions on the same
  bytes of store (batching, speculative drafting); per-question differences to the first leg are listed with the run, and a difference on a question
  whose rows did not change between the two stores is reported as run-to-run variation, not as an effect of the fix.

## The first leg stays on record

The first GPU leg's `result.gpu-answers.json`, `results.jsonl`, `verdict.json` (filed `fail`, `agg_1_12_not_met`) and its store stay where they are and
are never edited, overwritten or reclaimed for this purpose. The AGG-1.12 report cites BOTH legs: the first leg (uncorrected store: the defect, the
1 broken cell, 39 of 58 and 26 of 58) and the r2 leg (corrected store). The operator's decision makes the r2 run the one AGG-1.12 is judged on; the
first leg is the evidence that the defect existed and that the correction changed the outcome.

## Still IN-SAMPLE; the fix is post-data

The value-in-quote rule was added AFTER the first leg's data showed the JOHNSON_JOHNSON break, on the very 211 cells and 58 questions the design came
from. A zero-regression result on the corrected store is therefore a regression check on THIS store (the 158 held-out cells are a regression test
only: AGG-1.1 was 158 of 158 there), not evidence that the rule generalises, and P-large accuracy remains IN-SAMPLE for any gain (0 of 58 questions
read only untouched cells). All parent caveats stand. **The AGG-1.13 out-of-sample check must use orchestrator `737bf0535` or newer**; a
check that runs the typed stage without the rule (orchestrator `4d0265cd`) measures the defective stage and does not count.

## Pins of the r2 row

Orchestrator `737bf0535d1abecba154a2a22fbba0d910821ed8` (`src/` as committed; `facts_llm.py` unchanged, sha256 recorded by the runner's pin check);
research: the commit that contains this file (recorded in the row's `res/.pin`); corrected store and corrected CPU result by the sha256 values above;
`precond.sh` fails closed (rc 12) on any mismatch of the pins, this amendment's sha256, the store's sha256, the corrected CPU result's sha256, the
parent files' sha256, the v10 GPU binary, or `SOURCE.sha256`.
