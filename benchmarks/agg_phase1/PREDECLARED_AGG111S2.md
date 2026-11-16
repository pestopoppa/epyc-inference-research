# PREDECLARED -- AGG-1.11 step 2: typed-L4 + union-window re-run on the 13 AGG-1.3/1.4 r3 misses

Written 2026-11-16, BEFORE any step-2 data: no model call of this design has been made, and the per-cell outcome of the
backfill-gap diagnosis below has not been read. POST-HOC and IN-SAMPLE: the 13 targets are the AGG-1.11 step 1 triage misses
(`agg111_triage.json`, groups a and b) of the same two stores, and the typed-L4 stage was designed on AGG-1.6 cells that include
them. Gate class: futility-only. Nothing here is adopted into the pipeline; the result closes (or not) the AGG-1.11 step 2 box.

## Question

On copies of the AGG-1.3 r3 store (`agg13-reingest-cpu-20261111-r3`, replacement filings for MICROSOFT, JOHNSON_JOHNSON and 3M)
and the AGG-1.4 r3 store (`agg14-backfill-cpu-20261111-r3`), does the `agg_pipeline.v2` typed L4 stage (orchestrator
`facts_typed`, pin `cc613c22`: one typed multi-field call per distinct union window, current top-6 plus field top-3, value-in-quote
rule, `reconcile`) fill the 13 missed cells, and what is M1 before versus after over the 24 AGG-1.3 changed FY2021 slots and the 25
AGG-1.4 backfill `long_term_debt` cells, with no retained cell regressing?

## Targets, units and stores

* Targets: the 13 cells of `agg111_triage.json` with group a (10, FY2021, AGG-1.3 r3 store) or b (3, FY2022 `long_term_debt`,
  AGG-1.4 r3 store). The runner refuses any other count.
* Units: per (store, document), the DISTINCT union windows of that document's TARGET fields only
  (`facts_typed.doc_call_plan(windows, target_fields, descs)`, descriptions rendered for the cell's fiscal year as in AGG-1.3
  amendment 2). One typed call per window; a window record is the resumable unit. The offline plan is 69 calls
  (3M 10, JOHNSON_JOHNSON 19, MICROSOFT 18, ACTIVISIONBLIZZARD 8, BOEING 8, COCACOLA 6).
* Per document finish: `keep_slot` to the fields that chose the window (target fields only), `drop_unquoted`, `reconcile`, written
  to the WORKING COPY of its store (`out/store_agg13.sqlite`, `out/store_agg14.sqlite`). The sealed seeds in the job's `seed/` and
  the original row stores are never written. L3 facts and every non-target field are untouched.
* Model: worker_general Qwen3.6-35B-A3B-MTP Q8_0, the AGG-1.12/1.13 CPU recipe (v10 CPU kernel store, draft-mtp n-max 4, KV q8_0,
  -np 2, -c 32768, -fa on, -ngl 0), temperature 0.

## Metrics

* **M1 (primary)**: the frozen AGG-1.3/1.4 cell rule, `agg1314_zero_cell_diag.diagnose_cell` verdict HIT (value within rel_tol 0.005
  after the stored unit scale, period names the cell's fiscal year), over the 24 `agg13_cells.csv` cells on the AGG-1.3 store and the
  25 `agg14_cells.csv` cells on the AGG-1.4 store. BEFORE = the sealed seed copy; AFTER = the working copy after the re-run.
  Reported as 24, 25 and 49 totals, plus target fill (of 13, of 10, of 3) and every per-cell transition.
* **Regression (DONE-WHEN guard)**: a cell of the 49 that is HIT before and not HIT after. Any regression makes the verdict
  `agg111s2_regression` (step 2 not met) whatever the fill. Canonical readings (`select_canonical`, default rule, before and after)
  are reported beside raw, not gated.
* Cost: calls, prefill, wall, bad (cut or unparseable) replies.

## Pilot, gates and stops

* Pilot: 5 fixed cells, their union windows only, before anything else:
  COCACOLA_2022_10K long_term_debt (b), 3M_2022_10K total_assets and total_current_liabilities (a),
  MICROSOFT_2022_10K capex (a), JOHNSON_JOHNSON_2022_10K total_assets (a). Planned about 31 calls, about 300 s ESTIMATE.
  A pilot cell is scored in memory from its own windows (keep_slot to its field, drop_unquoted, reconcile, the M1 hit rule).
* **Stop-on-futility**: the pilot fills 0 of 5 -> stop, `result.cpu.json` with `futility_stop: pilot`, rc 3. Nothing else runs.
* Budget stops (re-book, not a gate, rc 5, no result file): pilot wall > 600 s (2x the box's 300 s); projected total wall > 2x
  the booked estimate.
* **Stop-on-answer**: the job ends when `result.cpu.json` carries the before/after M1 over the 24 + 25 cells and the backfill-gap
  diagnosis. No further run, no tuning, no second arm on these cells.
* Infrastructure stops (STOP file, grant deadline) exit 75 and resume from the window records.

## The 25-backfill gap diagnosis (zero model, part of the same result)

The 2026-11-16 check found the 25 backfill cells HIT 15/25 on the AGG-1.12 r2 store
(`agg112-arms-gpu-20261112-r2/out/store.sqlite`, sha256 `59adb2a8...d93`) versus 20/25 on the AGG-1.4 r3 store; the cause was not
isolated. For EVERY cell HIT on AGG-1.4 r3 and not HIT on the AGG-1.12 r2 store (expected 5; a different count is reported as is),
the runner logs which step dropped it, first match in this order, with the evidence row:

1. **PERIOD**: the r2 store holds a fact of the gold value whose period does not name FY2022 (L3 `l3_period.v2` or L4).
2. **VALUE_IN_QUOTE (post-hoc)**: no gold-valued fact in the r2 store, but one with the right period in the AGG-1.12 ingest store
   (`agg112-ingest-cpu-20261112/out/store.sqlite`, before the AGG-1.14 correction).
3. **RETRIEVAL**: no `long_term_debt` union window of the AGG-1.12 plan (`doc_call_plan` over the 9 schema fields, FY2022
   descriptions) contains the gold number as printed.
4. **EXTRACTION**: a union window contains it, but no `long_term_debt` row of that window's saved typed reply (AGG-1.12 ingest
   `results.jsonl` `t_window` records) carries the gold value.
5. **PERIOD (L4 row)**: such rows exist but none has a period naming FY2022.
6. **VALUE_IN_QUOTE**: such rows with the period exist and every one fails `facts_typed.value_in_quote`.
7. **CANONICAL**: a quoted, period-correct gold row exists and the stored facts lack it (the L4d `reconcile` / canonical step dropped it).
8. **OTHER**: none of the above (reported with its evidence).

Each lost cell also logs how AGG-1.4 r3 got it (the winning fact's extractor, period and quote: L3 versus the lazy L4 backfill).

## Honesty

In-sample and post-hoc; the gain is not an out-of-sample claim (AGG-1.13 holds that). A defect found after data needs a dated
amendment before any rerun, and results after it are labelled POST-DATA. The pilot verdict is a count; per-cell outputs are read only
from the final `result.cpu.json`.
