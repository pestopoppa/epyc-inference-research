# AGG-1.13 predeclared - AMENDMENT 1 (2026-11-14): the fiscal-year-end column rule for the four January-end filers

Written 2026-11-14 by the AGG-1.13 amendment session, **before any data**: no model call, no server, no v1 or v2 ingest, no planner call and no answer
exists for these documents, and no pipeline output on them has been read by anyone. `PREDECLARED_AGG113.md` (sha256
`a8b715254051062d6bfe184066916a01ae4c9f7a7b7a7db67203e26a1fff7ecf`) and every frozen input in its table (questions, golds, cells, `m1_hits`, the scorer, the metric
definitions, the Phase 0 kill bounds, the M4 bounds, the gates, the corpus, the arms and the order) are **NOT edited and stay in force**. This file adds one
pre-processing rule, one orchestrator pin and one runner flag, and changes nothing else. Its own sha256 is recorded in the runner (`agg113.py`
`AMENDMENT_SHA`), in the runner's `pin_problems` preflight check and in every compute row's `precond.sh`; a file cannot carry its own hash.

## The preflight failure

The step B runner (research `fe87e9041`, `run_agg113.py --phase preflight`) runs PREDECLARED checkpoint 0, the hard fiscal-year contract: the January-end
filers must be reachable as fiscal `2022` by the L3 period logic and by the frozen `m1_hits` (period must contain `2022`). It is zero-compute and reads no
output: one fixture table per (document, statement) whose header is the one the filing PRINTS (`agg113/design113.py` `PRINTED_HEADERS`, from the PDF text of the
gold pages read for gold building), pushed through the pinned orchestrator's L3 table logic, the resulting first-column fact read by `m1_hits`. Result on the
real corpus: Best Buy and Walmart pass; **Ulta (all 7 gold cells) and Foot Locker's balance sheet (3 cells) fail** for both L3 versions: their printed first
column is the calendar date `January 28, 2023` (the filing's fiscal 2022), the year regex labels it `2023`, and `m1_hits` cannot see it. Of the 31
January-end cells 21 pass and 10 fail (20 problem lines = 10 cells x 2 L3 versions). Foot Locker's income and cash-flow statements print bare years
(`2022 2021 2020`) and pass. Left alone, M1 for these 10 cells would have been decided by the L3 period label in the runner's reading, not by the pipeline under test
(rule 2 of *Honesty rules*: needs a dated amendment BEFORE any data).

## The operator decision (2026-11-14)

AGG-1.13 box item "Fix the table reader's year logic": implement a fiscal-year-end rule **so that NEITHER arm gains an edge**, then build the rows and publish the
l3 row. This amendment is that implementation's declaration.

## The rule (orchestrator `facts_det`, opt-in `fy_end`, orchestrator `8ac97157557ca45a4b86c8c9d2874c4b13977b6b`)

A date column header is relabelled to the fiscal year when ALL of the following hold; otherwise the label is exactly what the unamended code gives:

1. every year-bearing cell of the header row (table path) or every year on the header line (text-row path) is a pure printed date (optionally prefixed `As of`,
   `Year ended`, `Fiscal year ended`, ...), never a bare year, `Fiscal 2023` or a label that merely contains a date;
2. every such date's month is in `fy_end` (default January, February, March; configurable);
3. the filing's manifest fiscal year (corpus `fiscal_year`, the same per-document value `agg_pipeline.v2` already uses for the first-column rule; 2022 for every
   document here) is known and the FIRST such column's calendar year is the manifest fiscal year + 1.

Then every date column of that header is shifted back one year (`January 28, 2023 | January 29, 2022` becomes fiscal `2022 | 2021`). The manifest fiscal
year is the authority for the first column; a filing whose first column already carries the manifest year (Best Buy `January 29, 2022`, fiscal 2022) is **not**
shifted, which a naive "January means the prior year" rule would have broken. Bare-year headers (Walmart; Foot Locker's income and cash-flow statements),
calendar-year filers (`December 31, 2022`) and a missing manifest fiscal year are untouched. Both extractors that read headers apply it: the table extractor
(`l3_table.v1`) and the text-row extractor (`l3_textrow.v1`); the optional `date_header` / `inline_header` components are not part of either arm and are not touched.

**One flag, both versions, defaults unchanged.** `fy_end` is independent of `period_version`: `agg_pipeline.v1` (legacy period) and `agg_pipeline.v2` take it the same
way (`PipelineSpec.l3_kwargs(fy_end=)`, `IndexRegistry(fy_end=)`, `facts_det.extract_corpus_facts(fy_end=)`). With it unset every call is the unamended call;
the orchestrator tests assert byte-identical facts for both versions at table, text-row, corpus and registry level (`tests/unit/test_aggregation_fy_end.py`, 31 tests;
the existing 163 aggregation tests pass unchanged), so Phase 0, AGG-1.1 and AGG-1.12 reproduce. The orchestrator diff against `737bf0535` is exactly
`src/aggregation/{facts_det,pipeline,registry}.py` plus that test file; `facts_llm.py` (sha256 `c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451`),
`facts_typed.py` and the value-in-quote rule are unchanged, and `8ac97157` is a descendant of `737bf0535` as PREDECLARED *Arms* requires.

## Applied identically to arms A, B and C

* The runner flag `--fy-end` is passed by every row. The shared L0-L3 seed (`l3` row) is built with it on BOTH L3 paths of `run_l3` (the per-document table/regex facts and the
  corpus-level `IndexRegistry` rebuild with the text-row facts), so no fact is written twice under two period labels. The seed records `fy_end: true` and this file's sha256
  in `result.l3.json`; the ingest rows refuse a seed without both.
* Arm A (v1 ingest, planner `agg11.v0`) keeps the seed's L3 facts, which therefore carry the rule. Arms B and C (v2 ingest, one store, planners `agg11.v0` / `agg110.v1`) wipe
  their COPY and re-extract L3 under `l3_period.v2` **with the same `fy_end`**. **Both L3 versions now see these 10 cells** (the preflight contract passes for 31 of 31 January-end cells
  for v1 and v2). The `m1_hits` reading, the L4 stages (v1 `facts_llm`, v2 `facts_typed`), `canonical.v1` / `canonical.v2`, the planners, the audit and the scorer are unchanged.
* The preflight contract with the rule on is the declared pass condition for checkpoint 0; the l3 row additionally records `store_fy_check` (counts only, from the real parse). A real-parse
  shape the rule does not cover is reported, not tuned: any further rule change is a new dated amendment before the next data and its results are labelled POST-DATA (rule 2).

## Where the rule came from (honesty)

The rule was designed **only from the printed column headers** (`PRINTED_HEADERS`, the printed first columns of the gold pages) and the manifest fiscal year. It was not designed,
tuned, checked or selected from any v1 or v2 output, store row, fact, reply, SQL or answer on these documents, and none exists. Rule 1 of *Honesty rules* is intact. The same rule is
fixed for both arms before either arm runs; it favours neither. It is not an L3-only advantage for v2: the unamended code mislabelled the column in both versions equally.

**L4 label path (reported, not changed).** The L4 stages do not map column headers in code: `facts_llm` and `facts_typed` store the period string the model returns
(`str(r.get("period"))`), and the prompt renders each field description for the document's manifest fiscal year (`FY2022` for every document, via
`phase1_arms.l4_description`). L4 therefore labels the cells as the model labels them under a `FY2022` instruction; L4 is unchanged by this amendment.

## Unchanged

Questions (30), golds (80 gold cells, 76 filled), scorer, `m1_hits`, metrics, strata, M3 and M4 bounds, pilot documents and order, futility rules, gate classes, the verdict script, every frozen input
sha256 in `PREDECLARED_AGG113.md`, the model, kernels and recipes, and the AGG-1.14 value-in-quote rule.

## Power note

n is unchanged: 76 M1 cells, 30 questions; **this amendment adds no power**. Its direction is toward fewer discordant pairs: before it, for the 10 cells, no L3 version could be credited by `m1_hits`, so only L4 could
find them in either arm; with the rule on, L3 credits both arms equally, so a cell L3 now finds in both arms can no longer be a v2-only gain at M1. The planning prior in PREDECLARED *Expected effect* (b about 8 of 76, in-sample
and already discounted) therefore overstates the reachable b by up to the number of those 10 cells that v1's L4 would have missed; that number cannot be measured without reading outputs, which rule 1 forbids. Read M1 and
P-large exactly as the parent says: a null is "not shown at this n", never "no effect"; the box's z >= 2.5 level stays unreachable and is not restored by this amendment. The 10 affected cells are 13% of 76; the
strata and the thousands-unit and unlabelled-subtotal counts are unchanged.
