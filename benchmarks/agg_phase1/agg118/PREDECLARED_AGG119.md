# PREDECLARED AGG-1.19 — held-out L3 recall: fy_end for a lone month-day header, and FY-1 caption synonyms

Written 2026-11-16, BEFORE any fix is scored. Box: root `handoffs/active/knowledge-pipelines.md` → AGG-1.19.
Diagnosis it rests on: `agg118_pilot_diag.py` / `agg118_pilot_diag.json` (research `a60561946`), zero model.

## Question

On the AGG-1.18 out-of-sample pilot (NETFLIX_2019, SALESFORCE_2023, BOEING_2019; 48 cells; v3 raw 36/48), do two opt-in L3
components recover the 12 raw misses the diagnosis attributes to L3, without changing any cell on any existing set?

## The two components (opt-in, orchestrator `facts_det`; every existing pipeline byte-identical)

1. `fy_end_lone_monthday` — under `fy_end` only: ONE month-day header (``Fiscal Year Ended January 31,``, on the year line before its
   first year or on the preceding header line) applies to EVERY year column that follows it. The pairs then go through the UNCHANGED
   manifest guard `_fy_shift` (all months in `fy_end`, first calendar year == manifest FY + 1). Walmart and the other AGG-1.13 January
   filers whose first year equals the manifest FY are not shifted (tested explicitly).
2. `fy1_captions` — four normalised text-row captions mapped to their schema field: `net cash used in operating activities` and
   `net cash used provided by operating activities` → `operating_cash_flow`; `net loss earnings` → `net_income`;
   `property plant and equipment additions` → `capex`. Exact normalised match only; no current-portion or other-statement caption admitted.

Variants scored: each component alone, and both together (the proposed bundle; `agg_pipeline.v5` = v4 + both if it passes).

## Method (offline, no model, no compute row)

The agg117 method: for every store, a COPY brought to the v3 state; L3 re-extracted from its stored pages with the kwargs that
reproduce the stored L3 facts (parity, else the store is not scored); then with those kwargs plus the variant's components; the
fact-signature delta applied to a fresh copy, re-canonicalised under canonical.v3 and scored. Raw and canonical are both scored.

## Cell sets and the 0-regression rule

Every existing cell set, raw AND canonical: **211** (AGG-1.12 r2), **49×2** (AGG-1.11 s2 agg13 / agg14), **76** (AGG-1.13),
**14** LTD (AGG-1.16b), and the **48** AGG-1.18 pilot cells. Any cell right BEFORE and wrong AFTER, on any set, is a regression.
The 7 unrun held-out docs (MGM, American Water, Corning, CVS, PepsiCo, Verizon, FedEx) are NOT scored here; they stay reserved for step 2.
Also reported (not gated): Boeing net_income 2019 canonical (v3 picks -677 from the equity statement), whether it changes.

## Stop rules (gate class: futility-only)

- STOP-ON-FUTILITY: the bundle fixes **fewer than 6 of the 12** pilot raw misses, OR **any regression** on any set (raw or canonical).
  Then no v5, no step 2.
- STOP-ON-ANSWER: the offline table (fixed per variant, per set) and the regression list are filed.
- Step 2 (conditional on passing): ONE v3 ingest of the 7 unseen held-out docs, v4 and v5 derived offline from the same replies;
  2-doc pilot first; futility if the pilot shows 0 fixed for v5 vs v3, or any v5 regression vs v3.

Label: the 12 target cells are POST-HOC (the diagnosis saw them); only the step-2 docs are out of sample.
