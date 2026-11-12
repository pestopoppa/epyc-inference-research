# AGG-1.4 predeclared - amendment 2 (2026-11-12, written BEFORE any AGG-1.4 r3 data and before the r3 compute row was published)

Amends `PREDECLARED_AGG14.md` (sha256 `e27b4dcb212dd2692f69e9247a4ae923a1d43a8741eb724ae287df4bd63c038b`) as already amended by
`PREDECLARED_AGG14_AMENDMENT1_20261111.md` (sha256 `8a632c67748acadbe7a525fd1ff5a5f0b094da585e97e37915f3987b8ab46b49`).
Neither file is edited. The amendment's own sha256 is recorded in the r3 compute row and the run spec and asserted by the
runner's preflight (`agg14.AMENDMENT2_SHA`).

## 1. Diagnosis (no model; research `101e6671d`, `benchmarks/agg_phase1/agg1314_zero_cell_diag.py` and `agg14_r2_zero_cell_diag.json`)

AGG-1.4 r2 (`agg14-backfill-cpu-20261111-r2`) tripped G2: M1 0.64 (16/25), backfilled documents 0/9 (G1 ok: 72 calls <= 100,
12/12 units; G3 ok; G4 ok). Cause: all 13 `long_term_debt` L4 facts (9 documents) were labelled `unit: "USD"` by the model while their `value` is the
number in the filing's own scale: the number as printed in a USD-millions table (8 documents) or, for MGM Resorts, whose table
is in USD thousands, the printed 7,432,817 already restated to 7432.817 millions. `facts_llm.reconcile` trusted the label and
multiplied by 1e-6, so 2,467 was stored as 0.002467 USD_millions. The frozen scorer has no unit leniency (a 1000x unit error is a miss) and correctly rejected all of them. Six of
the nine backfilled gold cells (LOCKHEEDMARTIN, PEPSICO, AMD, COSTCO, JOHNSON_JOHNSON, MGMRESORTS) are in that class; the
diagnostic unit-normalised number (22/25 = 0.88) is not a gate number. Three misses are genuine and outside this defect
(ACTIVISIONBLIZZARD: no fact; BOEING: no fact; COCACOLA: wrong L3 column), and they stay misses.

## 2. Operator decision (2026-11-12)

"Fix both, amend, rerun." (AGG-1.3 and AGG-1.4 r2 zero-cell diagnoses.) r3 reruns the AGG-1.4 CPU row on a fresh `out/` under the
fixed stage.

## 3. What changes (exactly)

**(a) Unit-label cross-check in the L4 stage (`src/aggregation/facts_llm.py`, orchestrator).** `extract` records the filing's
declared scale for each fact: the quoted span itself (`$6.5 billion`, or a unit header inside the quote), else the nearest
preceding `(in millions|thousands|billions)` header in the chunk window (the header `facts_det` already reads for L3), else the
same lookup in the stored parent chunk. `reconcile` no longer trusts the label blindly: when the label is a money unit that
contradicts the declared scale AND the value traces to the quote (the printed number, or the printed number already restated
to millions by the declared scale) AND the restated size is plausible (< 5 trillion USD), the declared scale is used and the
fact carries a `unit_override` note (`label`, `declared`, `to`, `value_is`, `basis`, `raw_value`); the unit record carries
`n_unit_overrides` and `unit_overrides`. In every other case the label is kept, exactly as before; a fact with no declared
scale takes the previous path unchanged. The extraction prompt, the row grammar, the quote check and the confidence set are
unchanged. New `facts_llm.py` sha256: `c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451`. Orchestrator
`tests/unit/test_aggregation_unit_override.py` includes the six r2 facts (label USD, printed in a millions table, MGM in a
thousands table) reconciling to the gold values in `agg14_cells.csv`. **The scorer's no-leniency rule is not touched**
(`phase1_arms.m1_hits` and its unit table are unchanged).

**(b) Fiscal year of the field description is a parameter taken from the manifest, and a hard preflight.** The AGG-1.4 field
description `FIELD_DESC` ("..., FY2022 year end (USD millions, positive)") is now rendered through
`phase1_arms.l4_description(desc, fy)` with `fy` = the document's own fiscal year from `corpus.json` (2022 for all 28
documents), via `phase1_arms.l4_unit_fy`. For these FY2022 documents the rendered description, BM25 query, prompt and unit
record are byte-identical to the r2 stage (`tests/test_agg_phase1_runner.py::FyRenderingTests`); the AGG-1.1 `l4_unit` function stays unedited and AST-equal to
`phase0_arms.run_cpu`'s L4 loop. `agg14.fy_problems` runs `agg1314_zero_cell_diag.fy_contract_problems` as a hard preflight
check (rc 2 on any description naming a fiscal year other than its gold cell's) and `agg14.pin_problems` asserts this
amendment's sha256; `precond.sh` of the r3 job asserts it too.

**(c) Informational output.** The result gains `scale_signature` (`agg1314_zero_cell_diag.scale_signature`: L4
`long_term_debt` facts stored with |value| < 1 USD_millions, the r2 mis-scale signature; expected 0 abs_lt_1 on r3),
`n_unit_overrides` and `unit_overrides`. They are not gated.

## 4. What does NOT change

All gates and thresholds: G1 (fresh calls <= 100, pilot COSTCO, 12 documents), G2 (field M1 >= 0.75 over the 25 filled gold
cells, the 3 NA documents excluded and counted), G3 (the 224 earlier L4 units untouched: snapshot sha256, 0 reruns, 0 calls),
G4 (planner receipt: sql path, no model, backfill trigger, field, 12 documents); stop rules, budget stops, the gold cells
(`agg14_cells.csv`), the field pick, the scorer, the corpus, the Phase 0 and AGG-1.1 pins. The 224 earlier units remain the
AGG-1.1 cache hits and are not recomputed.

## 5. r2 stays on record

`agg14-backfill-cpu-20261111-r2` (verdict fail, native `phase1_gate_tripped`, G2 M1 0.64), its `out/` and the diagnosis files
remain exactly as they are. r3 is a new row (`agg14-backfill-cpu-20261111-r3`, supersedes r2) with a fresh `out/`; its result is
reported next to r2's, never in place of it. The gate that tripped in r2 is not relaxed: r3 is judged by the same G2.

## 6. L4-reuse check (preflight checkpoint 0)

`agg14.l4_reuse_check14` keeps the AGG-1.1 rule against the AGG-1.1 job's own pin for `llm_backend.py` and `phase0_arms.py` and
for the `l4_unit` AST, and now expects the changed stage for the backfill units: `facts_llm.py` must equal the sha256 above
(`phase1_arms.AMENDED_L4`; the AGG-1.1 pin is still printed) and `phase1_arms.l4_fy_stage_problems()` must be empty. A different
`facts_llm.py` fails closed. The 224 earlier units' cache-hit status is unaffected.

## 7. Limits stated in advance

* The cross-check needs a declared scale visible in the quote or in the stored chunk; a fact whose table header is not in the
  chunk keeps its label. The r2 PepsiCo page-97 fact ("Total $ 35,657 $ 36,026", a pension table with no unit header in the
  preceding chunks) is such a case and stays mis-scaled; it is a wrong-line fact either way.
* The three genuine r2 misses (ACTIVISIONBLIZZARD, BOEING, COCACOLA) are not addressed here; G2 is judged on the same 25
  cells, and r3 may still fall short of 0.75 for those reasons.
