# AGG-1.13 predeclared - out-of-sample validation of `agg_pipeline.v2` on 10 new FinanceBench filings

Written 2026-11-13 by the AGG-1.13 step A session, **before any model call on these documents, and before any pipeline output
for them exists** (no v1 or v2 ingest, no planner call, no answer: the only thing done with these PDFs is reading their printed
statements to build gold cells, and a deterministic extraction check). It sits beside `../PREDECLARED.md` (AGG-1.1, sha256
`2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af`), `../PREDECLARED_AGG112.md` (sha256
`22479709afbe5110115f6b2d1a951e799b8bb8752fabc5c51681c4aa30f59c1c`) and its Amendment 1 (sha256
`b5dec670f649b50af48ec4971e7503018d84c0f831c6bb859613b53b2348c584`), none of which is edited, and imports their frozen scorer
(`agg_phase0/phase0_arms.score_agg`), metric definitions, the Phase 0 kill bounds and the M4 bounds unchanged. **This is the
honest out-of-sample test: nothing on these documents may be inspected beyond gold building** (rule 1 under *Honesty rules*).
Its own sha256 is recorded in `PREDECLARED_AGG113.sha256` (same commit), in the AGG-1.13 box, the compute rows' `precond.sh` and
the runner; a file cannot carry its own hash.

## Question

Box AGG-1.13 (`handoffs/active/knowledge-pipelines.md`, root `3ef139d0f`): AGG-1.12 could only show in-sample gains (every M1 cell that
changed and all 58 questions were read while v2 was designed in AGG-1.8 / 1.11 / 1.6). On FinanceBench fiscal-2022 10-Ks that were never
part of any AGG-1.x document set, with gold cells and aggregation questions built before any system ran on them, **does
`agg_pipeline.v2` (typed L4 over union windows, `l3_period.v2`, `canonical.v2` attributable-net-income guard, value-in-quote rule) beat
`agg_pipeline.v1` on M1 (table recall) and on P-large aggregation accuracy, with what paired uncertainty, and are M3 and M4 inside the
Phase 0 bounds?**

Scope: Qwen3.6-35B-A3B-MTP Q8_0 on the v10 production CPU kernel store (CPU quarter) for ingest; Qwen3.8-27B Q8_0 + DFlash2 on the MI210 (v10 GPU kernel
store) for the L4c audit and the answers. It concludes nothing about other corpora, models, question types or fiscal years.

## The corpus: 10 documents, not 15 (a shortfall, with its cause)

The box asks for "about 15" filings. **Only 10 exist.** The selection rule (`selection.json`, produced by `build_agg113.py select`, which
fails on any candidate without an audit row):

1. universe: FinanceBench `financebench_document_information.jsonl` (sha256 below) rows with `doc_type` 10k;
2. candidates: `doc_period` 2022 or 2023 (a January-end filer's fiscal 2022 can be filed under either period) that are not in the AGG-1.1 28
   (`../corpus.json`) nor in `../corpus_agg13.json`;
3. **fiscal 2022 as the filing itself labels it** (AGG-1.1's selection text): the cover "fiscal year ended" date and the label the filing
   prints decide, not FinanceBench's period; the January-end retailers (the only filers whose label can lag the calendar) were downloaded and
   read for this audit (14 of the 20 candidates were downloaded and read; the other 6 end their fiscal year in mid-2023 and cannot be fiscal 2022 by a label convention that
   names the year by its end year; each of the 20 has its evidence row in `selection.json`);
4. never used: a document name appearing in any `benchmarks/agg_phase0`, `benchmarks/agg_phase1` (outside this directory) or `tests/test_agg*`
   file is excluded (the script scans; beyond 27 of the 28 it found nine test-fixture placeholders such as `AAA_2022_10K`, and `JOHNSON_JOHNSON_2021_10K`, `MICROSOFT_2021_10K`
   and `MICROSOFT_2023_10K`, none of which is a fiscal-2022 candidate);
5. take ALL eligible documents when fewer than 15 are eligible (no sampling is possible); with 15 or more, the 15 smallest
   `sha256("agg113-20261113|doc_name")`. The same salted order fixes the pilot (first 3, below).

FinanceBench's whole public `pdfs/` directory holds 37 fiscal-2022-by-period 10-Ks; 28 are the AGG-1.1 corpus; of the remaining nine
`FOOTLOCKER_2022_10K` is fiscal 2021 by its own label (every page prints "2021 Form 10-K", year ended 2022-01-29) and is **excluded**, while two
January-end filings FinanceBench files under 2023 are fiscal 2022 by their own label and are **included**: `FOOTLOCKER_2023_10K` (printed "2022
Form 10-K", 52 weeks ended 2023-01-28) and `ULTABEAUTY_2023_10K` ("fiscal 2022", year ended 2023-01-28). Result: **10 documents** (not a
choice: every eligible document is used). Getting to 15 needs documents of another fiscal year, which changes the `FY2022` contract of the
questions, the L4 description and the per-document fiscal-year path; that would be a separate, declared design, not done here.

| doc | company | GICS sector | fiscal year end | pages | traps (declared strata) |
|---|---|---|---|---|---|
| AMCOR_2022_10K | Amcor | Materials | 2022-06-30 | 144 | capex line combines PP&E and other intangibles |
| BESTBUY_2022_10K | Best Buy | Consumer Discretionary | 2022-01-29 | 111 | January year end labelled fiscal 2022; "Net earnings" |
| BLOCK_2022_10K | Block | Information Technology | 2022-12-31 | 196 | **USD thousands**; net loss; no dividends |
| FOOTLOCKER_2023_10K | Foot Locker | Consumer Discretionary | 2023-01-28 | 96 | file named 2023 is fiscal 2022; **unlabelled subtotals** (total assets, current assets, current liabilities) |
| INTEL_2022_10K | Intel | Information Technology | 2022-12-31 | 134 | 24.7 MB PDF; NCI line before attributable net income |
| NIKE_2022_10K | Nike | Consumer Discretionary | 2022-05-31 | 108 | cash-flow dividend line combines common and preferred (NA by the AGG-1.1 rule) |
| ORACLE_2022_10K | Oracle | Information Technology | 2022-05-31 | 132 | ligature glyphs (pypdf prints "ti" as NUL) |
| PG_E_2022_10K | PG&E Corporation | Utilities | 2022-12-31 | 352 | **combined registrant filing** (Corporation and Utility statements); no common dividend; small-caps "TOTALASSETS" |
| ULTABEAUTY_2023_10K | Ulta Beauty | Consumer Discretionary | 2023-01-28 | 105 | **USD thousands**; file named 2023 is fiscal 2022; no dividend |
| WALMART_2022_10K | Walmart | Consumer Staples | 2022-01-31 | 173 | total revenues vs net sales (two revenue lines); January year end labelled fiscal 2022 |

Downloaded once each, sequentially, from `https://github.com/patronus-ai/financebench/raw/main/pdfs/<doc>.pdf` into
`/mnt/raid0/llm/datasets/financebench-20261113-agg113/pdfs/` (the audit-only PDFs of excluded candidates are under `audit_pdfs/` of the same
directory and are never ingested). sha256, bytes and pages per document are in `corpus.json`; its manifest digest is below. No document
of this corpus was used in any AGG-1.x design artifact or run (rule 4 above, checked by script).

## Gold cells

80 cells (10 documents x the 8 AGG-1.1 metrics: revenue, net_income, capex, total_assets, operating_cash_flow, total_current_assets,
total_current_liabilities, dividends_paid; same definitions, conventions and USD-millions units as `agg_phase0.METRICS`: net income
attributable to shareholders, capex and dividends as positive outflow magnitudes, total revenue lines, dividends to common shareholders only).
**76 filled and verified, 4 NA** (dividends: Block and Ulta Beauty never pay or declare cash dividends; PG&E Corporation may not pay common dividends
and has no dividend line; Nike's cash-flow line combines common and preferred, NA by the AGG-1.1 precedent, operator decision 2026-11-10 #3). No cell was
left unpinned: every filled cell has a page and a verbatim quote and passed the cross-extractor check; there are no unverified cells.

Standard (the AGG-1.1 step A standard for its 144 non-Phase-0 cells, **weaker than Phase 0's two blind readers**): one reader (the step A agent,
reading pypdfium2 page text) plus the deterministic check `../verify_phase1_cells.py` (pypdfium2 5.3.0 and pypdf 6.16.1): the quote must be found,
whitespace and `$` normalised, on the stated PDF page in BOTH extractors, and the stated value must be a number printed in the quote
(`facts_sheet_xcheck.json`: 80 cells, 80 checked, **0 fail**, 30 cells print their figure on a single page only). Six cells carry `confidence=medium`
(Amcor capex, whose line combines PP&E and other intangibles; Foot Locker total assets, current assets and current liabilities, whose subtotals are
unlabelled on the balance sheet face and are confirmed by summing the printed components; Oracle operating cash flow and current liabilities, whose
quote is the figure run because pypdf prints the "ti" ligature of that PDF as NUL, so the label can only match in pdfium). No LLM call produced or
scored any value. A cell in a thousands-unit filing is stored as USD millions (Block and Ulta Beauty, 14 filled cells: a pipeline value 1000x off is a
miss, as in AGG-1.1).

## Questions

**30 aggregation questions**, `aggregation_questions.json`, generated by `agg113_questions.py` from the cell sheet: the AGG-1.1 `p1_*` templates (same wording, ops
inside the frozen scorer's set: argmax, argmin, top3, sum, count_gt, count_lt) re-pointed at the 10 documents; every threshold (`t`, `where.t`) is picked
by `pick_threshold` (the round number with 30-70% of the values on the strict side and the largest minimum relative gap to any value); a template
is kept only if its margin audit passes (minimum relative gap, including the gap of its `where` filter, >= 1.5%, no near-tie; stricter than AGG-1.1,
which did not audit `where` thresholds); per-family quota margin 4, net_income 4, revenue 3, cash_flow 3, free_cash_flow 3, capex 3, balance_sheet 5, dividends 5.
**30 of 30 verified, 0 excluded, 0 dropped by the audit; smallest margin 1.65% (n13_13, lowest operating cash flow: Foot Locker 173 vs Block 175.9), 1.8%
(n13_03), 2.5% (n13_26, n13_28).** Every one of the 76 filled cells feeds at least one question. The gold, per-company values and per-cell evidence
(doc, metric, value, page) are in `golds.jsonl`. These golds are new numbers: no AGG-1.1 or Phase 0 answer applies, and no continuity subset exists.
Conventions inherited unchanged: an NA cell drops the company from that question; Nike is excluded from the dividend questions and the text says so.

## Arms (the same model, kernel and recipes in every arm; one run each)

All three arms run the orchestrator at **`737bf0535d1abecba154a2a22fbba0d910821ed8` or a newer commit** (origin/main HEAD at writing is exactly
`737bf0535`; a newer pin must be an ancestor-checked descendant and the diff of `src/aggregation/` and `facts_*` must be listed in the run spec; a run
without the AGG-1.14 value-in-quote rule, orchestrator `4d0265cd`, does not count: Amendment 1 of AGG-1.12). The pipeline version is the orchestrator flag
`ORCHESTRATOR_AGG_PIPELINE` (v1 default, byte-identical to the legacy path by the orchestrator's tests).

| arm | ingest | planner | role |
|---|---|---|---|
| **A** `agg_pipeline.v1` + `agg11.v0` | v1 exactly as AGG-1.1 (legacy L3 period, `facts_llm` top-6 windows per (field, document) with the admission gate, one call per admitted window and field, `n_predict` 256, `canonical.v1`) | `agg11.v0` | the baseline: the pipeline AGG-1.1 was judged with |
| **C** `agg_pipeline.v2` + `agg110.v1` | v2 (`facts_typed` over union windows, `l3_period.v2`, `canonical.v2`, `drop_unquoted`) | `agg110.v1` | the candidate: the combination AGG-1.12 judged |
| **B** `agg_pipeline.v2` + `agg11.v0` | v2, same store as C | `agg11.v0` | separates the pipeline from the planner |

Pipeline effect = B vs A (same planner, different store); planner effect = C vs B (same store, different planner); the total, the primary P-large
comparison, is C vs A. M1 is a property of the ingest only (no planner): v2 store vs v1 store. A fresh v1 run is made (not the AGG-1.1 saved numbers),
so run-to-run variation of v1 is inside the comparison; v1 and v2 are ingested from the SAME L0/L1/L2 parse of the 10 documents (one shared L3 load, no
model, copied for each arm, never mutated).

## Metrics

* **M1 table recall (primary, ingest only)**: the AGG-1.1 definition (`phase1_arms.m1_hits`): a cell is found if the raw `facts` table (L3 union L4, after L4d)
  holds a fact for the same document and field with a period containing 2022 and a value within 0.5% relative; no unit leniency. Denominator: **76 filled cells**
  (22 of them in the 3 pilot documents), reported per document, per stratum (thousands-unit filers, unlabelled-subtotal filer, combined-registrant filer,
  January-end filers, May-year-end filers) and also on the canonical table (`agg18_miss_table.cell_state`, secondary). **Paired**: for each cell, found by
  v1 only (c), by v2 only (b), by both, by neither.
* **P-large accuracy (primary, answers)**: the frozen deterministic scorer over the **30** questions, arm C vs arm A (total), B vs A (pipeline), C vs B
  (planner); paired counts b (only the first-named arm right), c (only the second), both, neither.
* **Paired statistic (both metrics)**: z = (b - c) / sqrt(b + c) and the exact two-sided binomial p of b of (b + c) at 0.5; the paired difference
  d = (b - c) / n with SE sqrt(b + c - (b - c)^2 / n) / n and a 95% interval d +/- 1.96 SE; when b + c = 0 the statistic is undefined and said so.
  Reading, mechanical and **not a gate**: `v2_better` if b > c and z >= 2.0; `v2_worse` if c > b and z <= -2.0; otherwise `inconclusive` (direction and
  numbers still filed); `meets_planned_z` records z >= 2.5 (the box's planning level). The box's DONE WHEN is "filed with a paired SE", whatever the sign.
* **M3 ingest cost** (per arm; the Phase 0 kill bounds): prefill tokens / corpus tokens <= 2.0 and LLM calls / 100 chunks <= 150. v1 counts one call per
  admitted window and field and v2 one per distinct window; the bounds are the same numbers and the units of count differ (calls per unit is a derived quantity).
* **M4 per-question cost** (per arm): median calls <= 3 and median prefill <= 2,510 over the 30 (the AGG-1.1 / AGG-1.12 bounds; kill at 8 calls).
* Reported without a gate: SQL latency, wall per question, per-document M1 lists and, **only after both ingests are scored**, the listed regressions (cells found by v1
  and not by v2) and fixes with their strata. All of the above is filed per arm, never averaged across arms.

## Expected effect versus the sampling error (stated before the data; in-sample prior, honestly discounted)

The only prior is AGG-1.12 / AGG-1.14, which is **in-sample**: v2 was designed on the very 26 never-extracted cells and 58 questions it was then scored on, so
every figure below overstates what a new filing gives, by an amount that cannot be measured from those data. Planning prior (AGG-1.12 r1 and the AGG-1.14
offline re-score; 211 cells, 58 questions): raw M1 185 -> 208 (v1 0.877, v2 0.986; 23 fixed, 0 broken, i.e. discordant 10.9% of cells, all in v2's favour);
canonical 177 -> 203 (26 / 0); P-large with `agg110.v1` 18 -> 39 of 58 on the model leg (22 gained, 1 lost, z 4.4) and 40 of 58 on the offline re-score (23 / 1, z 4.5);
the pipeline effect alone 8 / 0 (z 2.8); the planner effect alone 14 / 1 (z 3.4).

* **M1, n = 76.** If the in-sample rates held, v1 misses about 9 cells (12.3% of 76) and v2 fixes about 88% of them with no breaks: b about 8, c about 0, z about 2.8,
  d about +0.10 (95% interval about +0.04 to +0.17). Discounted by half (the likelier planning case): b about 4, c 0 to 1, z about 1.3 to 2.0, d about +0.04. **The box's
  planned level, z >= 2.5 for a +7-point M1 gain, is NOT reachable with this corpus**: +7 points is about 5 cells (z 2.2 at c = 0, 1.9 at c = 1); z >= 2.5
  needs b - c of at least 7 with c = 0 (b = 7, c = 0: z 2.65; b = 6, c = 0: 2.45), a +9-point gain, because the box sized the corpus at about 115 cells and
  76 exist. The unpaired binomial SE of an M1 near 0.95 over 76 cells is 0.025. The pilot (22 cells) has an SE near 0.08 and can only screen for a defect.
* **P-large, n = 30.** In-sample b about 0.40 and c about 0.02 of the questions: b about 12, c about 0.5, z about 3.3; halved: b about 6, c about 1, z about 1.9. The smallest
  paired effect that clears z = 2 is b = 4 with c = 0 (z 2.0), b = 7 with c = 1 (z 2.1), b = 9 with c = 2 (z 2.1); b = 6, c = 1 (z 1.9) is **indistinguishable from noise** at this n
  and will be reported as that. The binomial SE of an accuracy near 0.5 over 30 questions is 0.09, so a single-arm accuracy is a rough number. The pipeline effect alone (B vs A) has an
  in-sample prior of 8 / 0 over 58 and, at n = 30, about 4 / 0 (z 2.0) before discounting: expect it inconclusive. The 30 questions share the same 76 cells (each question reads
  5 to 10 companies), so the questions are positively correlated and the effective n is smaller than 30.
* **No regression, with its real strength.** Observing c = 0 over the cells v1 gets right (about 67 of 76) bounds the per-cell break rate near 4.5% (one-sided 95%, 3/n); it is a
  point check on these cells, not a certificate of a rate.
* **What a null means.** This design can detect a large effect or a defect; a small true gain (under about +5 points of M1 or +20% of questions) will not reach z = 2 here. A result inside
  the noise is "not shown at this n", never "no effect" and never "v2 is not better".
* **Determinism.** temperature 0, top_k 1, seed 0, thinking off, `-np 2` on the CPU ingest and `-np 1` for the answers: reproducible per call on the same binary, but batching can
  change the bytes of a reply; a difference of one or two cells or questions between an identical re-run and this run is within that.

## Honesty rules (the out-of-sample contract)

1. **Nothing on these documents may be inspected beyond gold building.** The only reading done so far is the extraction of the 80 gold cells and the deterministic check. No person or
   agent reads a v1 or v2 output on these documents (facts, store rows, replies, SQL, answers) before the analysis script has produced the final result; the runner's pilot and interim
   outputs are counts only (hits, calls, tokens, seconds), never per-cell or per-question lists.
2. **No change after data.** v1, v2, both planners, the scorer, the corpus, the cells, the questions and every threshold above are frozen. A defect found in any of them while the rows run
   (including a gold error) needs a dated amendment to this file BEFORE the rerun, and every result produced under or after the fix is labelled **POST-DATA** and shown beside the pre-fix
   result, which stays on record; nothing is tuned on these documents. v2 or a planner changed because of anything seen here is no longer under test and the claim is withdrawn, not rescued.
3. **The reader.** The gold cells were read by the step A agent session (an LLM session in the same lineage that designed v2), not blind, from pypdfium2 text, with a deterministic cross-extractor
   check; they were built before any system ran and none was chosen by looking at a system's output. A disputed cell is listed and handled under rule 2.
4. **Selection.** The corpus is every eligible document, not a draw; the deliberate traps (thousands-unit filers, unlabelled subtotals, a combined registrant, January fiscal years) were
   chosen by availability, not tuned to v2's known misses, and are declared strata. A pass supports "v2 generalises to filings of this kind"; it does not support other question types,
   other fiscal years, scale beyond 10 filings or other models.
5. **One run per arm.** A re-run of v1 or v2 on the same data is a replicate and does not replace the first; the first complete run is the result.

## Pilot, futility and checkpoints (from the box; gate class futility-only)

0. **Preflight (zero compute)**: this file's sha256, the orchestrator pin and the research commit recorded in the spec; the 10 PDF sha256 equal `corpus.json`; `build_agg113.py check` exits 0;
   30 of 30 verified; the stage-equals-arm-T code check of AGG-1.12; an offline per-document fiscal-year contract check (`agg1314_zero_cell_diag.fy_contract_problems`: January-end filers
   whose fiscal 2022 ends in 2022-01 (Best Buy, Walmart) or 2023-01 (Foot Locker, Ulta Beauty) must be reachable as `2022` by the L3 period logic and by `m1_hits`; a hard failure here is a
   runner defect fixed under rule 2 BEFORE any model call, because the runner's own period reading, not the pipeline, would otherwise decide M1).
1. **Shared L3 load** (CPU, no model, the Phase 0 / AGG-1.1 path): futility if >= 2 of 10 fail to parse (infrastructure defect: stop, fix under rule 2, rerun); fewer are `doc_fail` and their cells count as
   misses, never silently dropped.
2. **Pilot, bounded <= 1,800 s per arm**: ingest of **3 documents, the first 3 of the salted order: WALMART_2022_10K, ULTABEAUTY_2023_10K, BLOCK_2022_10K** (24 units, **22 filled cells**), v1 and
   v2, each on its own CPU quarter or one after the other, each <= 1,800 s wall (ESTIMATE v1 about 1,300 s, v2 about 1,050 s); the pilot units are the first units of the full ingest (resumable, nothing
   discarded). **Futility (stop and file `agg113_futility`; the GPU row is never requested)**: the v2 pilot's raw M1 on the 22 cells is below v1's on the SAME cells (strictly fewer hits), or a
   projected M3 beyond a kill bound for either arm. Budget stops (re-book, NOT gates, exit 5): pilot wall > 1,800 s; projected L4 wall > 1.5x the booked figure.
3. **Interim look after 6 of the 10 documents** (pilot included): futility only, the same two tests on the completed documents' cells; never an early GO.
4. **Final ingest look** (all 10 documents, both arms): M1 and M3 per arm and the paired M1; the CPU result is written; the GPU row's precondition `cpu_ready` is "both ingests complete, no futility stop,
   no M3 trip". **Stop-on-answer** when both versions are scored (the GPU result file is written).
5. **GPU row**: the L4c audit sample (the AGG-1.1 recipe, unmodified) on each store, then P-large on the 30 under A, then B, then C (order fixed), on one np1 196k server per the AGG-1.12 r2 recipe;
   M4 per arm. A failed M4 bound is filed `phase1_gate_tripped` naming it; P-large accuracy is not a gate.
6. Reuse valid evidence first: golds, cells, parse and the shared L3 load are built once; the v1 ingest is NOT taken from any earlier run (different documents).

## Verdict (mechanical, from committed files by script)

`agg_1_13_filed` always, with M1 and P-large for A, B and C, the paired statistics, the strata, M3 and M4 per arm, the regressions and fixes (after both are scored), and the labels of
the *Paired statistic* reading; plus `agg113_futility` or `phase1_gate_tripped` (naming M3:v1, M3:v2, M4:A, M4:B or M4:C) when those stops fire. The result carries no
`adopt_agg_pipeline_v2_as_default` recommendation computed from these data alone; the owner reads it beside AGG-1.12 (in-sample) and rule 4.

## Offline plan and estimates (ESTIMATE, from measured rates; the pilot replaces them)

Units: 10 documents x 8 fields = 80 (field, document) units. Measured rates: v1 6.07 calls per unit at about 8.2 s per call (AGG-1.1 and Phase 0 r2: 1,359 calls / 224 units; 483 calls /
4,147 s for 80 units); v2 4.03 calls per unit at 9.68 s per call (AGG-1.12: 903 calls / 224 units; AGG-1.6 arm T 201 calls in 1,946 s). The calls per unit depend on the documents' windows and are
re-estimated by `--dry-run` once the runner exists and by the pilot.

| row | ESTIMATE | basis |
|---|---|---|
| shared L3 parse and load, 10 documents, no model | about 1,700 s CPU | Phase 0 r3: 1,658 s for 10 documents; these 10 are about 1.5 M tokens (1,551 pages) vs Phase 0's 1.96 M |
| v1 L4 ingest (arm A) | about 4,100 s CPU quarter (486 calls x 8.2 s = 3,985 s + 120 s server load) | 80 units x 6.07 |
| v2 L4 ingest (arms B, C) | about 3,300 s CPU quarter (322 calls x 9.68 s = 3,117 s + 120 s load + 60 s L3) | 80 units x 4.03 |
| GPU: audit x2 stores, P-large 30 x 3 arms, two cold 27B loads | about 2,400 s | AGG-1.1 P-large 804 s / 58 questions (13.9 s each) x 90 answers = 1,250 s plus the AGG-1.12 audit and load overhead (about 800 s) |

Total about 9,100 s ESTIMATE CPU (the v1 and v2 ingests can run on two quarters at once) plus about 2,400 s GPU (the box's figures, 12,000 s and 4,000 s, were for 15 documents).

## Frozen inputs (recorded before any data)

| file (this directory) | sha256 |
|---|---|
| `financebench_document_information.jsonl` (361 rows; fetched from the FinanceBench repo `data/` on 2026-11-13) | `1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575` |
| `selection.json` | `52e11d3b93b321cb1feac4dbbaa429bdfda0328dcb6ef1c1cbe85c02dd80191d` |
| `corpus.json` (10 docs: path, sha256, bytes, pages, source URL, fiscal year end) | `f45ee6a4080f0eddf8f410184295cad8154ca78b84ae09be778427cbd42846ec` |
| manifest digest (canonical JSON of the `corpus.json` docs array) | `94c8497392bd304685b1b48aac553af0b18bbb97ea29a41286b3257a5ca5dc97` |
| `facts_sheet.csv` (10 x 8 = 80 cells; 76 filled, 4 NA; page and verbatim quote per cell) | `fd37e254ff6d6d2766b3681884348d9feabe7fd0144f4aca9b58a1ba72021328` |
| `facts_sheet_xcheck.json` | `321c95bd2e33b95354f3b5df6f286dbaab0b1cf8703b89815aa2732ce7a3d5d7` |
| `aggregation_questions.json` (30 questions) | `c73986a4ed89333484c5156a3b10a4adddac2f98e67c5ffe43b184a2813d4d18` |
| `golds.jsonl` | `c5c2325c9d915a58f13051b63d398ffb76ca7ff02ea333c33d10740af5203ca3` |
| `verification_record.json` | `e55ae142c8dc05b95162a6de6112e7f72e6d3deb9575e8535d51df8b9005bf79` |
| `../verify_phase1_cells.py` (reused unchanged) | `ce74aa5ab65d460009768274fa3d2e4d569ec511ee4700dee693c5b5056fad78` |

Pins: orchestrator `737bf0535d1abecba154a2a22fbba0d910821ed8` or newer (above); research: the commit that contains this file (recorded in each row's `res/.pin`); v10 kernel store
binaries as in AGG-1.12 (CPU `4483bcfc92c5c3bb0005df31e1c668cb565d49f7b40ba94f4df5a126cbbb648c`, GPU `2b49713f0e3c022132393bf302dab954a1e2c1974b91c28d01ce2d192abdfa7b`, `--version` 10303,
ggml linkage verified by `verify_ggml_linkage.sh`); the step B runner (`run_agg113.py`) lands after this file as a code-only commit that must not change questions, golds, scorer, metrics, arms or
gates, with its sha recorded in the run spec.

## Post-hoc caveats (all of them stand in the result)

1. 10 documents and 76 cells, not 15 and about 115: power is below the box's planning level (see *Expected effect*); a null is "not shown at this n".
2. One reader plus the cross-extractor check (not two blind readers); 6 medium-confidence cells; 30 of 80 cells print their figure on only one page.
3. The corpus is the whole eligible set, so it contains the traps listed above by availability; the strata are small (thousands-unit filers 14 cells, the unlabelled-subtotal filer 8 cells) and a stratum is
   reported, not tested.
4. v1 is run fresh at the v2 orchestrator pin (flag v1), not at the AGG-1.1 commit; it is byte-identical to the legacy path by the orchestrator's tests, but the model, kernel build and `facts_llm` unit/fiscal-year fixes
   (`1a408ce4`, ON for both versions) are the AGG-1.12 ones, not AGG-1.1's, so A is not a replay of AGG-1.1.
5. The question set is template-derived over 8 metrics and 10 companies; each question reads 5 to 10 companies; questions are correlated through shared cells.
6. M3 counts differ in unit between v1 and v2 (caveat 6 of AGG-1.12).
7. Gate classes stay futility-only at interim looks until OP-ABSEQ / OP-PQP / OP-ALPHAWIN are decided; no early GO is possible.
8. All times and call counts above are ESTIMATES from measured rates, not measurements.
