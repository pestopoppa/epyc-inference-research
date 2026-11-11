# AGG-1.4 predeclared - one lazy backfill field

Written 2026-11-11 by the AGG-1.3 / AGG-1.4 preparation session, **before any AGG-1.4 gold cell exists and before any
AGG-1.4 model call**. The only data that existed when this file was written is the AGG-1.1 result (store,
`results.jsonl`, `verdict.json`) and the field-pick table `agg14_field_pick.json` computed from it by
`agg14_field_pick.py` (no model, no gold). It sits beside `PREDECLARED.md` (AGG-1.1, sha256
`2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af`, **not edited**) and imports its frozen scorer, kill
constants and L4 stage unchanged. This file's sha256 is recorded in the AGG-1.4 compute row, the run spec and the runner
(`run_agg14.py`).

## Question

AGG-1.4 box: pick the field with the lowest L3 recall on the 28-document corpus that no L4 unit has yet extracted; ask a
question that needs it so the planner takes the `sql(+backfill)` path and builds only that field's L4 units over the
admitted chunks. Does lazy backfill build a missing field cheaply (<= 100 calls), with recall that clears M1 >= 0.75,
without touching anything already built, and does the planner receipt show why?

Scope: the AGG-1.1 28-document store, Qwen3.6-35B-A3B L4 (CPU quarter), the unchanged L4 stage. It concludes nothing about
other fields, models, corpora or scales, and it does not measure answer accuracy of the 27B (see "What is not in this row").

## The field and why (deterministic rule, `agg14_field_pick.py`)

"L3 recall" needs gold, and the AGG-1.1 gold covers only the 8 metrics, all of which have 28 L4 units. The 8 are therefore
not eligible (the box's own "no L4 unit has yet extracted"), so the pick is made over the other L3 canonical field names
(`facts_det.FIELD_ALIASES`) with a gold-free proxy:

1. candidates = the 9 field names L3 can emit that are neither an L4 field nor an alias of one (`net_income_attributable`
   is `net_income`);
2. applicable documents = those whose PDF text (pypdfium2) has a statement-style line starting with one of the field's
   labels and carrying a figure of >= 3 digits within 40 characters;
3. proxy = applicable documents with >= 1 L3 FY2022 fact for the field / applicable documents (per document, so an
   upper bound of true recall: a wrong value counts as a hit);
4. eligible = >= 20 applicable documents (below that, M1 >= 0.75 cannot be tested with a binomial SE under about 0.1);
5. pick = lowest proxy among the eligible, ties by field name.

Result (`agg14_field_pick.json`; applicable / L3 hit / proxy): `long_term_debt` 25 / 16 / **0.640** (picked),
`depreciation_amortization` 23 / 16 / 0.696, `total_liabilities` 27 / 19 / 0.704, `total_equity` 27 / 21 / 0.778,
`accounts_payable` 25 / 20 / 0.800, `cash_and_equivalents` 22 / 20 / 0.909; ineligible for fewer than 20 applicable
documents: `cogs` 15 / 8 / 0.533, `inventories` 17 / 14 / 0.824, `operating_income` 19 / 15 / 0.789. **Judgment recorded:**
`cogs` has the lowest raw proxy but only 15 applicable documents, and its backfill set would be 19 documents x 6.08 calls
= about 116 calls, over the box's 100-call gate by construction; the eligibility floor of 20 is the rule that excludes
it and was fixed here before any AGG-1.4 data. A reader who prefers `cogs` should say so before the row runs.

**Backfill set (the trigger).** The documents with no canonical row for (`long_term_debt`, 2022) after L3 + the existing L4
(`select_canonical`, in memory, nothing written): 12 documents: `ACTIVISIONBLIZZARD_2022_10K`, `ADOBE_2022_10K`,
`AES_2022_10K`, `AMD_2022_10K`, `APPLE_2022_10K`, `BOEING_2022_10K`, `COCACOLA_2022_10K`, `COSTCO_2022_10K`,
`JOHNSON_JOHNSON_2022_10K`, `LOCKHEEDMARTIN_2022_10K`, `MGMRESORTS_2022_10K`, `PEPSICO_2022_10K`. The runner recomputes it and
fails preflight if it differs from this list.

## The question and the planner receipt

Question (fixed): "Which company in the corpus reported the highest FY2022 long-term debt (non-current portion), in USD
millions?" It contains a corpus cue ("Which company", "in the corpus"), so `scoped_facts` keeps the 28-document corpus and
the planner takes `sql` (L3 facts are present, corpus >= 1M tokens: `size_rule`, no model call). The field the question
needs is resolved by a deterministic phrase table (`planner.question_fields`, from `facts_det.FIELD_ALIASES`; no model);
`planner.backfill_plan` compares it with the canonical catalog and returns, per needed field, the documents without a
canonical row. The planner receipt (`decision_receipt.v1`) carries `backfill = {trigger, fields, missing_docs, per_field}`.
**Gate:** the receipt shows `path == "sql"`, `llm_called == false`, `backfill.trigger == true`, `backfill.fields ==
["long_term_debt"]` and the 12 documents above.

## Extraction (Jev-style, unchanged L4)

Each missing (field, document) pair is one L4 unit run by the AGG-1.1 `phase1_arms.l4_unit` itself (not a copy): the
document's windows (10,000 characters) ranked by BM25 for the field's description, top 6 per document, L4a admission
(relative score >= 0.6 admit, <= 0.2 reject, in between a one-token `yes`/`no` GBNF gate), L4b one GBNF-constrained row call
per admitted window (`ROW_GBNF`: `period`, `value`, `unit` from a closed set, **verbatim `quote` required**, `confidence`
from a closed set; `n_predict` 256; quotes not found in the window are rejected deterministically), L4d unit/scale
reconcile. The only new input is the field description (the prompt's field line): "Long-term debt (non-current portion on
the balance sheet, excluding current maturities), FY2022 year end (USD millions, positive)". Grammar, quote check and
`n_predict` cap are therefore the box's Jev-style typed output, with no new grammar to validate.

## Gold for the field (deterministic, no LLM)

One cell per document (28), page + verbatim quote, USD millions, same standard as AGG-1.1 step A: one reader (the
preparing agent, reading extracted page text) + the cross-extractor check (`verify_phase1_cells.py` rule: quote found on
its page in both pypdfium2 and pypdf, value printed in the quote). **Convention fixed here:** a cell is FILLED iff the
consolidated balance sheet prints a line labeled "long-term debt" (any qualifier: "net of current portion", "excluding
current portion", "less current portion", "and finance lease obligations") for the FY2022 year end; the value is that
line's figure (thousands -> millions). Otherwise NA with a note naming the label the filing uses instead (for example a
bare "Term debt" or "Debt"): never mapped by judgment. A cell that fails the cross-extractor check is excluded and counted.
Files `agg14_cells.csv`, `agg14_xcheck.json`, pinned in the runner by sha256 before the compute row is published; the
reader did not use the L3/L4 facts of the store to find values.

## Metrics and gates (exactly the box's DONE WHEN)

- **G1 backfill calls <= 100 (gate).** Fresh LLM calls (L4a gate calls + L4b extract calls) of the backfill units.
  Expected 12 units x 6.08 calls/unit (the AGG-1.1 measured mean, range 5-8) = about 73 calls (ESTIMATE); a 100-call
  cap leaves 27 calls of headroom, so a trip needs a structural change in calls per unit (about 8.3 per unit).
- **G2 field recall, M1 >= 0.75 for that field (gate, and reported).** Recall against the hand-verified gold cells
  (filled cells only; NA cells excluded and counted): a cell is found if the raw `facts` table (L3 U L4, after L4d)
  holds a fact for the same document and field with a period containing 2022 and a value within 0.5% after unit scale (the
  AGG-1.1 rule, no unit leniency). Also reported without a gate: L3-only recall on the same cells before the backfill,
  the lift, and the recall on the 12 backfilled documents alone.
- **G3 untouched (gate).** Every previously built L4 unit is a cache hit: the AGG-1.1 results' 224 units are seeded as
  reused and none re-runs; zero LLM calls are spent on any of the 8 earlier fields; the sha256 of the sorted
  (`fact_id`, `doc_hash`, `field`, `value`, `unit`, `quote`, `audit_status`) rows of every `l4_llm.v1` fact of the 8
  earlier fields is identical before and after; `documents`, `chunks` and L3 facts counts are unchanged.
- **G4 planner receipt (gate).** As above.

A gate trip files `phase1_gate_tripped` naming the gate.

## Pilot, stopping rules

**Bounded pilot:** the field's extraction on ONE document, the smallest by extracted size among the 12 (`COSTCO_2022_10K`,
239,294 characters), <= 600 s of unit wall. **Futility (stop, file G1):** pilot calls x 12 > 100 (projection) or the running
total of fresh calls passes 100 after any unit. **Budget stop (rc 5, re-book, not a gate):** the pilot unit wall > 600 s,
or the projected total wall (pilot wall x 12) > 2x the booked 1,000 s. Formal-gate cells stay futility-only at interim
looks; no early GO (GO needs all 12 units). Reuse valid evidence first: the AGG-1.1 store and units are reused;
the AGG-1.1 stores are never mutated (own copy; `SOURCE.sha256` seals the code).

## Cost of the REPL-retrieval alternative for the same field (ESTIMATE, recorded as the box asks)

For "highest long-term debt" the REPL arm (B1: local 27B REPL + BM25 hybrid retrieval, Phase 0/1 recipe, no persistent
fact layer) has to find and read the balance sheet of every document in the question's scope each time the question is asked.
AGG-1.1 B1 measured on 30 questions: mean 23.2 calls, median 30 (the cap) with 16 of 30 hitting the 30-call cap
(`budget_exhausted`), median prefill 76,760 tokens, mean wall 134.7 s (5.8 s per call), on the 27B GPU. For this field:
at least one search and one read per document = **>= 56 calls for the 28 documents (about 2x the 30-call cap, so the
arm cannot finish in its own budget)** and about **24-25 calls for just the 12 documents the backfill covers**, at about
5.8 s per call = **about 145 s and 40-60k prefill tokens per question on the GPU for the 12 documents, about 325 s and
about 140k tokens for all 28** (read size 12,000 characters, about 3k tokens per document, plus the growing REPL history).
The backfill costs about 73 calls (about 650 s on a CPU quarter, about 12 x 7.6k = 92k prefill tokens) **once**, after which
SQL answers cost the AGG-1.1 P-large median of 2 calls and about 600 prefill tokens per question; break-even is about 3
questions about this field. These are estimates from B1's measured rates, not measurements; the REPL arm is not run here.

## Budget (ESTIMATE)

About 1,000 s on one CPU quarter: 12 units x 54 s (AGG-1.1 measured 52-63 s per unit) = 650 s + server start and page-cache
load of the 37.8 GB model (the first AGG-1.1 unit took 370 s) + margin; the box's provisional 1,000 s (100 calls x 8.2 s)
is kept. No GPU row (below).

## What is not in this row (and why there is no GPU row)

DONE WHEN needs no 27B step: the receipt is deterministic, the backfill is on the CPU quarter, recall is table recall
against gold. The L4c audit of the new facts and an end-to-end 27B answer to the question are **not** part of this row;
the question's answer correctness is not claimed. If the operator wants the answer measured, that is a separate GPU row
gated on this row's `result.cpu.json`.

## Honest caveats

1. **The pick rests on a proxy** (document-level L3 coverage among documents whose text states the field), not on gold
   recall; the eligibility floor was needed to exclude `cogs`. The gold-recall ranking of the other candidates is unknown.
2. **One reader + cross-extractor** gold, 28 cells, NA cells excluded; Apple ("Term debt") and Adobe ("Debt") are expected
   NA by the fixed convention, which removes two backfill documents from M1 by construction.
3. **Existing L3 values for the 16 covered documents are not re-checked by the backfill**; a wrong L3 value counts as found
   if any fact within 0.5% exists, so M1 can read above the SQL answer's quality (AGG-1.1 M1 has the same property).
4. **One field, one question, one corpus.** A pass supports "lazy backfill of a mid-recall field stays inside 100 calls
   here", not a general backfill economy.
5. **Orchestrator additions** (`planner.question_fields`, `planner.backfill_plan`, the receipt's `backfill` block) are new
   code added for this box; the L4 stage files are byte-identical to the AGG-1.1 pin.

## Pins

Frozen by the runner (`run_agg14.py` preflight): `agg14_field_pick.json`, `agg14_cells.csv`, `agg14_xcheck.json`,
`corpus.json` (AGG-1.1, unchanged), the AGG-1.1 Phase 0 code pins, and the AGG-1.1 `results.jsonl`. Orchestrator and
research commits are recorded in the run spec. Any mid-run fix needs a dated amendment to this file BEFORE the rerun and
says POST-DATA in its heading if data exists.
