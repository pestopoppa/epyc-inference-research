# AGG-1.13 predeclared - AMENDMENT 2 (2026-11-15): the fiscal-year-end rule on the SPLIT headers the real parse produces

Written 2026-11-15 by the AGG-1.13 amendment session, **before any data**: no model call, no server, no v1 or v2 ingest, no L4 stage, no planner call and no answer exists for these
documents, and no output of any arm has been produced, let alone read. `PREDECLARED_AGG113.md` (sha256
`a8b715254051062d6bfe184066916a01ae4c9f7a7b7a7db67203e26a1fff7ecf`) and `PREDECLARED_AGG113_AMENDMENT1_20261114.md` (sha256
`279645ad90b74f479a3decd397c8ffc8b26750491332a478985750d9aae2daaa`) are **NOT edited and stay in force**; every frozen input (questions, golds, cells, `m1_hits`, the scorer, the metrics, the
Phase 0 kill bounds, the M4 bounds, the gates, the corpus, the arms, the order) is unchanged. This file adds one orchestrator pin, one extension of the amendment 1 rule to the header shape the
real parse actually produces, one stricter preflight and one stricter l3 completion check, and changes nothing else. Its own sha256 is recorded in the runner (`agg113.py` `AMENDMENT2_SHA`), in
the runner's `pin_problems` preflight check, in `seed_fy_end_problem` and in every compute row's `precond.sh`; a file cannot carry its own hash.

## What amendment 1 achieved on the real parse: nothing

Amendment 1 (orchestrator `8ac97157`) relabels a date column header to the fiscal year when the header carries WHOLE printed dates (`January 28, 2023`). The real parse of the four January-end
filings does not produce whole dates on the statements that matter. The documents are flattened paragraphs read by the text-row extractor (`l3_textrow.v1`; the local OpenDataLoader parse detects
no table on them: the seed holds 96 tables in all; Foot Locker has none and Ulta's five are exhibit forms, so none carries a date header), and Ulta and Foot Locker print the date in two fragments:

* a month-day line `Fiscal year ended January 28, January 29, January 30,`, a blank line, then the year line `(In thousands, except per share data) 2023 2022 2021` (Ulta income statement p55,
  cash-flow statement p57); or
* the month-days leading the year line itself: `January 28, January 29, (In thousands, except per share data) 2023 2022` (Ulta balance sheet p54; Foot Locker balance sheet p37, with `ASSETS` after the
  years; Ulta note p77, after a separate `Fiscal year ended` line).

`facts_det._fy_relabel_years` matched only whole `Month D, YYYY` dates, so on these lines it found no date, returned the printed years unchanged, and the first column stayed `2023`. Amendment 1's rule
was a **no-op on the real parse**. The l3 row `agg113-l3-cpu-20261114` (the shared seed, no model) recorded it in its own `store_fy_check` (counts only): Ulta 0 of 7 gold cells under fiscal 2022 (all 7 only
under another period), Foot Locker 5 of 8 under 2022, 0 under another period, 3 with no L3 value; Best Buy 6/0/2 and Walmart 6/0/2 (Foot Locker's 3 no-value cells are its balance-sheet totals, which print
without a caption). The ingest rows `agg113-ingest-v1-cpu-20261114` and `agg113-ingest-v2-cpu-20261114` were stopped by STOP file before any L4 output existed, and the GPU row was never published.

## Why the amendment 1 preflight passed anyway (the vacuous pass)

The preflight contract (`design113.py` `PRINTED_HEADERS`, `agg113.py` `fy_contract`) used **intact** dates (`["January 28, 2023", "January 29, 2022"]`) and pushed them through
`facts_from_table` only. That is a shape the real parse never produces for these filings, so the check exercised a code path the run does not take and could not fail. It also never read the seed's
own page text. Amendment 2 fixes both (below).

## The fix (orchestrator `facts_det`, opt-in `fy_end`, orchestrator `cc613c22e39c6e605ca2f717107b8e3fc1498093`)

Designed **only from the printed headers** (the header lines of the gold pages, quoted in `design113.py` `PRINTED_HEADERS`) and the manifest fiscal year. Every condition of amendment 1 is kept: a date
column header is relabelled only if every date's month is in `fy_end` (default January, February, March), the manifest fiscal year is known, and the FIRST column's calendar year is the manifest
fiscal year + 1; then every column is shifted back one year, through the unchanged `_fy_shift`. What changes is how a date is READ from a split header:

1. **Text rows.** A header line is a *month-day-only line* when, after its `Month D,` tokens are removed, only the filler words `fiscal`, `year(s)`, `ended`, `ending`, `as`, `of`, `at`, `period(s)`, `week(s)`
   remain. The months of the most recent such line are remembered **across blank lines** and **expire at the next non-blank line** (consumed by it if it is a year header, dropped otherwise), and
   never cross a page. A year header line takes its months from the month-day tokens that **lead** it (before its first year), else from the remembered line. The i-th month is paired with the
   i-th year **only when the counts are equal**; the pairs go through `_fy_shift`. One month-day for three years (Walmart `Fiscal Years Ended January 31, (Amounts in millions) 2022 2021 2020`) is
   not paired and is left alone, as are whole-date headers (Best Buy), which keep amendment 1's reading.
2. **Tables.** The same shape can in principle reach the table reader (a header row of month-day-only cells over a row of bare years). It is read the same way: the cells of the row directly above the
   year header that are month-day-only, paired with the year columns when the counts match, then `_fy_shift`. The real seed contains no such table; the path is covered by a unit test only.

`fy_end` stays opt-in (default off); with it unset every call is the unamended call and the facts are byte-identical for `agg_pipeline.v1`, `agg_pipeline.v2` and every earlier store. With it set, the
facts of the six non-January-end documents are byte-identical to amendment 1's, and so are Best Buy's and Walmart's (measured on the cached real parse of all ten documents, both L3 versions: zero
changed facts for those eight; only Ulta's and Foot Locker's split headers change). The orchestrator tests (`tests/unit/test_aggregation_fy_end_split_header.py`, 20 tests) quote the real printed page text verbatim: Ulta p54, p55, p57, p77, Foot Locker p37,
Best Buy p41 and Walmart p53; the 194 earlier aggregation and text-row tests pass unchanged. The orchestrator diff against `8ac97157` is `src/aggregation/facts_det.py` plus that test file.

## Applied identically to arms A, B and C

The runner flag `--fy-end` is unchanged and is passed by every row; the new reading is inside the orchestrator function every row calls, on both L3 paths of `run_l3`, on the corpus-level registry rebuild, and in
the v2 arms' re-extract. The shared seed is rebuilt (`-r2` rows, same settings as `-r1`) under this orchestrator pin, and the ingest and GPU rows read only that seed; the seed records `fy_end: true` and BOTH
amendment sha256s, and an arm refuses a seed without them. `agg_pipeline.v1` (legacy period) and `agg_pipeline.v2` take the rule the same way, so neither arm gains an edge: the unamended code mislabelled
the same columns in both versions and the amended code relabels the same columns in both. L4 (`facts_llm`, `facts_typed`), `canonical.v1` / `canonical.v2`, the planners, the audit, the scorer and `m1_hits` are unchanged.

## The stricter checks (the vacuous pass cannot recur)

* **Preflight, fixtures.** `PRINTED_HEADERS` now holds the header lines **as the real parse flattens them**, split headers kept split, and the contract runs them through `facts_from_page_text` (the real path)
  as well as the intact-date table fixtures (`TABLE_HEADERS`, amendment 1's, kept). 31 January-end gold cells, both L3 versions, both paths.
* **Preflight, real page text.** `--parse-cache` points the preflight at the seed's own parse cache (the l3 row's `out/cache`, copied into every `-r2` row). It runs `facts_from_page_text` over the
  **actual page text** of the four January-end documents for both L3 versions and counts, per document, the gold cells whose value is held under fiscal 2022 / only under another period / not at all
  (the same counts `store_fy_check` records). The floor: **no gold cell holds its value only under another period, and all 7 Ulta gold cells hold it under fiscal 2022** (Foot Locker's 3 uncaptioned
  balance-sheet totals are the declared unlabelled-subtotal trap and are reported, not part of the floor). A missing cache file for a January-end document is a **problem**, not a skip: a check that reads
  nothing must not pass.
* **l3 completion.** `run_l3` applies the same floor to the finished store's real L3 facts (`store_fy_check`) and returns `complete: false`, rc 2, without writing a complete seed if it fails; the
  ingest rows refuse an incomplete seed.

## Where the fix came from (honesty)

The fix was designed from the **printed headers** of the gold pages (the PDF text of the pages named in `facts_sheet.csv`, the same pages amendment 1 cited, read now from the l3 row's parse cache) and from the
l3 row's `store_fy_check` counts. Read in this step: those page texts, and counts (changed facts per page, gold cells by period class) of the **deterministic L3 text-row extractor**, which both arms share and
which involves no model. Not read, because they do not exist: any L4 output, any v1 or v2 ingest row, planner, SQL, reply or answer. PREDECLARED rule 1 is stated about "a v1 or v2 output": none exists, and
the L3 seed is the shared pre-model layer. Rule 1 is not weakened for anything downstream, and no per-cell hit vector of any arm exists to be seen. The rule was not tuned on any arm's gold-cell result; it was
confirmed against the real printed headers, and it changes only headers, never which rows are read.

Per PREDECLARED rule 2 (and the last sentence of amendment 1's *Applied identically*), this is a dated amendment before the rerun: the r1 seed (built under amendment 1 alone) **stays on record** as the pre-fix
result, and every result produced from the `-r2` rows is labelled **POST-DATA** and shown beside it.

## Unchanged

Questions (30), golds (80 gold cells, 76 filled), scorer, `m1_hits`, metrics, strata, M3 and M4 bounds, pilot documents and order, futility rules, gate classes, the verdict script, every frozen input
sha256 in `PREDECLARED_AGG113.md`, the model, kernels and recipes, the AGG-1.14 value-in-quote rule, the L4 stages and the amended v1 `facts_llm.py` (sha256
`c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451`).

## Power note

n is unchanged: 76 M1 cells, 30 questions; **this amendment adds no power**. Its direction is the one amendment 1 stated: with the split headers read, the L3 layer credits the same cells (Ulta's 7; Foot Locker's 3 balance-sheet
gold cells are uncaptioned totals no L3 reads) in both arms, so a cell L3 now finds in both can no longer be a v2-only gain at M1. Read M1 and P-large exactly as the parent says: a null is "not shown at this n", never "no
effect"; the box's z >= 2.5 level stays unreachable and is not restored by this amendment.
