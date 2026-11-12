# AGG-1.3 predeclared - amendment 2 (2026-11-12, written BEFORE any AGG-1.3 r3 data and before the r3 compute row was published)

Amends `PREDECLARED_AGG13.md` (sha256 `fde0cb0d4bea0fc816e1393b38533c82c766853330d1218d2b0b37c3c5607183`) as already amended by
`PREDECLARED_AGG13_AMENDMENT1_20261111.md` (sha256 `07831066b7549c1bb2b97b92e97ec960ec415ee794724189d120be537031d84c`).
Neither file is edited. The amendment's own sha256 is recorded in the r3 compute row and the run spec and asserted by the
runner's preflight (`agg13.AMENDMENT2_SHA`).

## 1. Diagnosis (no model; research `101e6671d`, `benchmarks/agg_phase1/agg1314_zero_cell_diag.py` and `agg13_r2_zero_cell_diag.json`)

AGG-1.3 r2 (`agg13-reingest-cpu-20261111-r2`) passed G1-G4 as frozen (3 documents re-parsed, 25 skipped by hash, supersession
ledger and SQL absence ok, 146 calls / 189,422 prefill tokens, M1 0.782 = 165/211) but the 24 cells of the three replaced
FY2021 documents (MICROSOFT, JOHNSON_JOHNSON, 3M) scored 0/24. Cause: `agg_phase0.METRICS` bakes `FY2022` into every L4 field
description ("..., FY2022 (USD millions)"). The three replacement filings are FY2021 10-Ks with no FY2022 column, so the model
returned `[]` for 19 of the 24 units and labelled the facts it did return (3 carry the exact FY2021 gold value) `FY2022`;
`agg13.m1_hits_fy` (the period must contain `2021`) can never accept those. The scorer is correct; the stage that fed it was
configured for the wrong year. The period-agnostic diagnostic is 3/24, which is not a gate number.

A second, independent defect in the same shared stage (found by the AGG-1.4 r2 diagnosis): the model's unit label was trusted
by `facts_llm.reconcile`. It applies to any L4 unit, so it also applies to the 24 rerun units here.

## 2. Operator decision (2026-11-12)

"Fix both, amend, rerun." (AGG-1.3 and AGG-1.4 r2 zero-cell diagnoses.) r3 is a rerun of the AGG-1.3 CPU row on fresh outputs
under the fixed stage; the first mover of this amendment is that decision, not a result.

## 3. What changes (exactly)

**(a) Fiscal year of the L4 field description is a parameter, taken from the document's own fiscal year.**
`phase1_arms.l4_description(desc, fy)` returns `desc` unchanged for `fy` None or 2022 and otherwise replaces the literal
`FY2022` by `FY<fy>`; `phase1_arms.doc_fiscal_year(entry, corpus)` reads the fiscal year from the corpus manifest (the entry's
`fiscal_year`, else the manifest's `fiscal_year`, else 2022). The three replaced slots carry `fiscal_year: 2021` in
`corpus_agg13.json` (frozen, unchanged: sha256 `933b4d2f47f313cb903127a27f12f6f348b6887bdf813d8bc2fc8fe424926e43`), so their
8 units per document are extracted and retrieved with descriptions that say `FY2021` (and "Total assets at FY2021 year end",
etc.), and the model reports the period it was asked for. `phase1_arms.l4_unit_fy` is the new unit function used for the rerun
units: it is `l4_unit` with the description rendered by `l4_description`, plus the record fields `fy`, `n_unit_overrides` and
`unit_overrides`. The AGG-1.1 `l4_unit` and the frozen `agg_phase0.METRICS` text are NOT edited (their sha256 pins and the
`l4_unit` AST equality with `phase0_arms.run_cpu`'s L4 loop are still asserted). For an FY2022 document `l4_unit_fy` renders the
byte-identical description, BM25 query, extraction prompt and record as `l4_unit` (regression test
`tests/test_agg_phase1_runner.py::FyRenderingTests`).

**(b) Unit-label cross-check in the L4 stage (`src/aggregation/facts_llm.py`, orchestrator).** `extract` records the filing's
declared scale for each fact: the quoted span itself (`$6.5 billion`, or a unit header inside the quote), else the nearest
preceding `(in millions|thousands|billions)` header in the chunk window (the header `facts_det` already reads for L3), else the
same lookup in the stored parent chunk. `reconcile` no longer trusts the label blindly: when the label is a money unit that
contradicts the declared scale AND the value traces to the quote (the printed number, or the printed number already restated
to millions by the declared scale) AND the restated size is plausible (< 5 trillion USD), the declared scale is used and the
fact carries a `unit_override` note (`label`, `declared`, `to`, `value_is`, `basis`, `raw_value`; surfaced in the unit record
as `unit_overrides`). In every other case the label is kept, exactly as before; a fact with no declared scale takes the previous
path unchanged. The extraction prompt, the row grammar, the quote check and the confidence set are unchanged.
New `facts_llm.py` sha256: `c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451` (orchestrator
`tests/unit/test_aggregation_unit_override.py` includes the six AGG-1.4 r2 facts reconciling to the gold).
**The scorer's rule is not touched:** `agg13.m1_hits_fy` / `phase1_arms.m1_hits` keep "period contains the cell's year, value
within 0.5 % after the stored unit's scale, no leniency for a 1000x unit error".

**(c) Preflight is stricter, as a hard check.** `agg13.fy_problems` runs `agg1314_zero_cell_diag.fy_contract_problems` over
every document's gold cells with the descriptions rendered for that document's own fiscal year; any description that names a
fiscal year other than the cells' fails preflight (rc 2) before any model call. `agg13.pin_problems` asserts this amendment's
sha256. The shell `precond.sh` of the r3 job asserts it too.

**(d) The GPU row's L4c audit uses the same rendering (the row is prepared, not published by this amendment's authors).**
`phase0_arms.run_audit` (frozen, sha256-pinned, not edited) re-extracts every audited fact with `A.METRICS[field]`, i.e. with
`FY2022`, which on a FY2021 slot would dispute every correct fact. `agg13.run_audit_fy` is that audit with the description
rendered for the audited fact's document's own fiscal year (`phase1_arms.l4_description`) and the re-extracted rows passed
through `facts_llm.reconcile` (so the unit cross-check applies as it did to the stored fact); `run_agg13.py --phase gpu-audit`
uses it. For an FY2022 document with correct unit labels its verdicts are those of `run_audit`. The audit result is
informational (`audit_status`, `audit_done`); no gate reads it. The P-large answer prompts (`phase0_arms`) are untouched.

## 4. What does NOT change

* All gates and thresholds: G1 (exactly 3 re-parsed and re-extracted, 25 skipped by hash), G2 (supersession), G3 (fresh calls
  <= 160 and prefill <= 201,887 tokens, 3/28 x AGG-1.1 x 1.10), G4 (M1 >= 0.75 over the same 211 cells, 187 retained + 24
  changed, and the retained-document regression check), G5 on the GPU row; stop rules, budget stops, the gold cells
  (`agg13_cells.csv`), the golds, the questions, the scorers, the corpus manifest, the Phase 0 and AGG-1.1 pins.
* The 200 retained L4 units remain the AGG-1.1 cache hits (seeded from the AGG-1.1 results, FY2022 documents). They were
  produced by the AGG-1.1 stage and are not recomputed; the amended stage applies to the 24 rerun units only.

## 5. r2 stays on record

`agg13-reingest-cpu-20261111-r2` (verdict pass, native `agg_1_3_cpu_pass`, with the 0/24 caveat in its `verdict.json`), its
`out/` and the diagnosis files remain exactly as they are. r3 is a new row (`agg13-reingest-cpu-20261111-r3`, supersedes r2)
with a fresh `out/`; its result is reported next to r2's, never in place of it. The r2 "pass" is not retracted: it is the
honest outcome of the frozen gates on a mis-configured stage and says nothing about re-extraction quality on new documents.

## 6. L4-reuse check (preflight checkpoint 0)

`agg13.l4_reuse_check13` keeps the AGG-1.1 rule against the AGG-1.1 job's own pin for `llm_backend.py` and `phase0_arms.py` and
for the `l4_unit` AST, and now expects the changed stage for the rerun units: `facts_llm.py` must equal the sha256 above
(`phase1_arms.AMENDED_L4`; the AGG-1.1 pin is still printed) and `phase1_arms.l4_fy_stage_problems()` must be empty (FY2022
rendering is the frozen description for every field; FY2021 rendering is the same text with the year swapped and names no other
year). A different `facts_llm.py` fails closed. The retained units' cache-hit status is unaffected.

## 7. Limits stated in advance

* The cross-check needs a declared scale visible in the quote or in the stored chunk; a fact whose table header is not in the
  chunk keeps its label (and is not corrected).
* The r3 M1 on the 24 changed cells is a new measurement under the fixed stage, one run, and is reported as such; a pass of
  G4 does not by itself mean the changed-document extraction quality is good (report the per-document split beside it).
