# PREDECLARED_AGG118: held-out verification of `agg_pipeline.v4` against `agg_pipeline.v3`

Written 2026-11-16, before any ingest of these documents. No model output on any of the 10 documents exists or was read. Box:
root `handoffs/active/knowledge-pipelines.md`, AGG-1.18. Plan: the AGG-1.17 follow-up, research `7e7dd4339` (`agg118.py`).

## Question

`agg_pipeline.v4` (orchestrator `88bba4fd`, opt-in, default stays v3) = v3 + the four AGG-1.17 L3 text-row components
(`pipeline.V4_L3_TEXTROW` = ltd_qualifier, note_ref, tail_label, pipe_cells_ltd) + the L4 quote rule `value_in_quote.rounded.v1`.
Its offline gain (14 LTD cells 6 -> 14; canonical +1 on the 76, 211 and agg14 sets; 0 regressions) is IN-SAMPLE: the 14 cells tuned
AGG-1.17 and BLOCK tuned the quote rule. On 10 filings never used in any AGG-1.x design, does v4 gain over v3, and does it regress
any cell? A v4 regression on any held-out cell blocks the default flip.

## Corpus (frozen)

`corpus.json`, corpus_id `agg118_heldout10`, manifest digest `3c8d2a7ec78abd6547187fde4bb557e0cc6b1a11e186b2286f21e5e6daaa3e41`.
10 FinanceBench 10-Ks from the same source as AGG-1.13 (`https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf`,
the `build_agg113.py` URL form), downloaded one at a time to `/mnt/raid0/llm/datasets/financebench-agg118-heldout/pdfs/`, sha256
in `corpus.json` and `SHA256SUMS` beside the PDFs. SALESFORCE_2023's bytes equal the AGG-1.13 audit copy (same sha256).

* FEDEX_2023_10K, SALESFORCE_2023_10K: issuers never in any AGG-1.x corpus.
* 8 FY2019 10-Ks of AGG-1.1-only issuers (not the AGG-1.13 10, not the replacement 3M / JOHNSON_JOHNSON / MICROSOFT 2021 docs):
  MGMRESORTS_2019 and NETFLIX_2019 (statements in USD thousands, the quote-rule trap), AMERICANWATERWORKS_2019, BOEING_2019,
  CORNING_2019, CVSHEALTH_2019, PEPSICO_2019, VERIZON_2019 (each prints a balance-sheet line labelled "long-term debt").
  FY2019 and FY2018 columns overlap no fiscal-year column of any used filing (AGG-1.1 FY2022/2021/2020; the replacement 2021 docs
  2021/2020/2019 income columns are a different filing of a different year's statements, and the scored periods here are 2019/2018).

## Golds (frozen, `facts_sheet.csv`, written from the filing text only)

The 8 SCHEMA_FIELDS + long_term_debt, for FY and FY-1: 180 cells = **174 filled + 4 NA + 2 EXCLUDED**. Each row: page, 1-based
line and the verbatim quote in the pypdfium2 page text (whitespace collapsed); `build_agg118.py check` proves every quote is on its
line and every gold number is printed in its quote (180/180). **pypdfium2 5.3.0 is the only extractor**: there is no second-extractor
cross-check in this step (AGG-1.13 used pypdf as a second reader). Conventions (AGG-1.1 / AGG-1.4 / AGG-1.13): USD millions
(thousands divided by 1000); net income = attributable to the parent where printed; capex and dividends as outflow magnitudes;
operating cash flow signed (Netflix and Boeing 2019 negative); fiscal year = the fy_end rule (a year end in months 1-3 belongs to
the prior year: SALESFORCE_2023, year ended January 31 2023, is scored as 2022 / 2021); long_term_debt only from a line whose label
carries "long-term debt", else EXCLUDED with the reason. NA: dividends_paid for NETFLIX_2019 and SALESFORCE_2023 (none paid).
EXCLUDED: SALESFORCE_2023 long_term_debt 2022 and 2021 (label "Noncurrent debt"). No P-large questions (the box asks for none).

## Arms and the one ingest

ONE ingest under `agg_pipeline.v3` (orchestrator `88bba4fd`) of the 10 documents: local ODL parse, store, L3 (v3 kwargs with fy_end),
typed L4 (one call per union window, `keep_slot.harvest.v1`, `value_in_quote.v1`), every typed reply saved. v4 is DERIVED OFFLINE
from the same saved replies by the `agg118.py` method: on a copy of the v3 store, (1) the L3 delta from `pipeline_spec(v4).l3_kwargs`
vs `pipeline_spec(v3).l3_kwargs` (text-row components), (2) the saved replies re-finished under `value_in_quote.rounded.v1`
(`agg118.apply_quote`). Both copies re-canonicalised (canonical.v3, same for both). Parity checks: the v3 kwargs reproduce exactly the
stored L3 rows, and the v1-rule replay reproduces only stored fact ids (`replayed_not_stored` = 0); a failure is a gate stop, not a result.
Model: worker_general Qwen3.6-35B-A3B-MTP Q8_0 on the v10 CPU kernel store, temperature 0, top_k 1, seed 0, thinking off; CPU quarter.
Runner: `run_agg118.py` (research), scorer `agg116b.score` (unchanged: raw = any stored fact HITs the gold; canonical = the
canonical row for (doc, field, fy) equals the gold within the frozen tolerance).

## Metrics

Primary: v3 vs v4 over the 174 filled cells, paired, raw and canonical: counts, fixed (v4 hit, v3 miss), regressed (v3 hit, v4 miss),
exact two-sided sign test on fixed vs regressed. Secondary: the long_term_debt subset (18 filled cells). Cost: LLM calls, prefill
tokens, L4 wall (identical for both arms by construction).

## Rules

* **0-regression rule**: any held-out cell (raw or canonical) right under v3 and wrong under v4 is a regression and blocks the
  default flip, whatever the gains.
* **Pilot**: the first 3 documents NETFLIX_2019_10K, SALESFORCE_2023_10K, BOEING_2019_10K (thousands trap, January year end and a new
  issuer, a small filing with a long-term debt line; 48 filled cells), about 800 s, bound 1,800 s. After the pilot the derivation and
  both scorings run on the pilot cells; COUNTS ONLY are written for the pilot (no per-cell reading before the final look).
* **Stop-on-futility** (gate class futility-only, no early GO): the pilot shows v4 = v3 on every pilot cell (no cell fixed or
  regressed, raw and canonical), or any v4 regression in the pilot. Either stops the row (rc 3) with `result.cpu.json` filed.
* **Stop-on-answer**: the v3 vs v4 table (raw, canonical, long_term_debt subset) and the regression list are filed.
* Budget stops (re-book, not gates, rc 5): pilot L4 wall > 1,800 s; pilot-projected L4 wall > 2x the booked 3,300 s; after the pilot,
  L4 wall of the other 7 documents > 1.5x the pilot projection. Two or more documents failing to parse is an infrastructure stop (rc 6).
* Power: at n = 174 a null on gains is "not shown at this n"; the decision rule is the regression count, which needs no power.
* Any defect found after data needs a dated amendment before a rerun; results after it are POST-DATA.

## Expected effect

In-sample v4 fixed LTD cells whose balance-sheet line carries a qualifier or note reference after "Long-term debt" (here CORNING
"(Note 11)", MGM ", net", FEDEX ", LESS CURRENT PORTION", PEPSICO "Obligations") and thousand-scaled replies the model rounds
(MGM, NETFLIX). Prior: small positive (a few cells), 0 regressions expected.
