# AGG-1.3 predeclared - incremental re-ingest of 3 changed documents

Written 2026-11-11 by the AGG-1.3 / AGG-1.4 preparation session, **before the 3 replacement filings were downloaded,
before any gold cell for them exists and before any AGG-1.3 model call**. The only data that existed when this file was
written is the AGG-1.1 result (store, `results.jsonl`, `verdict.json`), which this box reuses. It sits beside
`PREDECLARED.md` (AGG-1.1, sha256 `2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af`, **not edited**) and
imports its frozen scorer, kill constants and L4 stage unchanged. A file cannot carry its own hash: this file's sha256 is
recorded in the AGG-1.3 compute row, the run spec and the runner (`run_agg13.py`).

## Question

AGG-1.3 box (`handoffs/active/knowledge-pipelines.md`): replace 3 documents of the 28-document store with their
other-fiscal-year 10-Ks (new content hashes) and re-run ingest through the content-hash path. Does the store re-do
exactly the changed work, supersede the dead facts cleanly, and stay inside the incremental cost bound, with
aggregation quality (M1) and per-question cost (M4) intact?

Scope: the AGG-1.1 store, Qwen3.6-35B-A3B L4 ingest, Qwen3.8-27B for the P-large measurement. It concludes nothing
about other models, corpora, scales, or about documents that change in place (same PDF bytes, new parse), or about a
filing that *adds* facts to an entity.

## The 3 replaced documents and the replacement rule (deterministic)

Rule: the three AGG-1.1 **pilot documents**, `MICROSOFT_2022_10K`, `JOHNSON_JOHNSON_2022_10K`, `3M_2022_10K`
(PREDECLARED.md checkpoint 2: ranks 5, 11, 17 of the 18 new documents by extracted size, fixed before any AGG-1.1 data),
because the box's stop rule is stated against "its AGG-1.1 pilot rate" and only the pilot documents have a measured
per-document L4 rate. Replacement filing: the FinanceBench 10-K of the **same company for fiscal year 2021**, the nearest
other fiscal year for which a FinanceBench 10-K exists for all three (`MICROSOFT_2021_10K`, `JOHNSON_JOHNSON_2021_10K`,
`3M_2021_10K`; a FY2023 10-K exists only for Microsoft; the repository listing is `gh api repos/patronus-ai/financebench/contents/pdfs`).
Each PDF is downloaded once, sequentially, from `https://github.com/patronus-ai/financebench/raw/main/pdfs/<doc>.pdf` into
`/mnt/raid0/llm/datasets/financebench-20261111-agg1/pdfs/` and sha256-pinned in `corpus_agg13.json`.
A FY2021 filing prints FY2021, FY2020 and FY2019 columns and **no FY2022 column**, so the superseded FY2022 facts of these
three companies cannot be re-created by the new documents: "absent from SQL answers" is a real test.

The store `doc_id` of a replaced slot stays the old `doc_name` (`MICROSOFT_2022_10K`, ...). `doc_id` is the opaque
identity of the slot in the corpus; the content hash is the identity of what it holds (this is how
`CorpusStore.ingest` already defines "replace": same `doc_id`, new content hash). The new PDF file name is recorded
beside it in `corpus_agg13.json`.

## Gold for the 3 new filings (deterministic, no LLM)

Same procedure and conventions as AGG-1.1 step A: the 8 metrics (`revenue`, `net_income`, `capex`, `total_assets`,
`operating_cash_flow`, `total_current_assets`, `total_current_liabilities`, `dividends_paid`) at the filing's own fiscal
year (2021), one cell per metric with PDF page and a verbatim quote, USD millions; capex and dividends are positive
magnitudes of outflows; net income is attributable to the company's shareholders; an absent line is NA, never assumed.
Verification: one reader (the preparing agent reading extracted page text) plus the deterministic cross-extractor check of
`verify_phase1_cells.py`: every quote must be found, whitespace/`$`-normalised, on its stated PDF page in BOTH pypdfium2
and pypdf, and the value must equal a number printed in the quote. A cell that fails is excluded and counted; **no
LLM produces or scores any gold value**. This is the same one-reader standard the operator accepted for AGG-1.1
(PREDECLARED.md caveat 3). Files: `agg13_cells.csv`, `agg13_xcheck.json` (written after this file, pinned in the runner
by sha256 before the compute row is published).

## Work and metrics

Phases of `run_agg13.py` (CPU row, `--phase reingest`): copy the AGG-1.1 store; seed the 200 L4 units of the 25 retained
documents as cache hits from the AGG-1.1 `results.jsonl` after the L4 reuse test (the L4 stage sources are identical to
the AGG-1.1 pin: `facts_llm.py`, `llm_backend.py`, `phase0_arms.py` and the `l4_unit` statements); send **all 28 documents**
through `CorpusStore.ingest` (parse result from the content-keyed parse cache when the PDF sha256 matches a cached parse,
a real local-ODL parse otherwise); L3-extract and L4-extract (8 units each) only the documents the store reports
`replaced`; refresh the indexes; compute M1; write `result.cpu.json`.

- **G1 manifest (gate).** The run manifest (`reingest_manifest.json`, one entry per document: PDF sha256, parse
  cached/real, store receipt, L3 facts written, L4 units run) shows **exactly 3 documents re-parsed and re-extracted and
  25 skipped by hash**: for the 25, the store receipt is `unchanged` (same content hash), no real parse, no L3 write, no
  L4 unit, zero LLM calls; for the 3, receipt `replaced`, one real parse, L3 facts written, 8 L4 units run. Any other split,
  any parse failure of the 3 (`doc_fail`), or any retained document with a real parse trips G1.
- **G2 supersession (gate).** The old content hashes of the 3 slots (read from the base store before ingest): each has
  0 rows left in `documents`, `chunks`, `tables_`, `facts` and `facts_canonical`; each has its facts recorded in the
  store's `facts_superseded` ledger with `superseded_by` = the new hash, and one `documents_superseded` row; the count
  of ledger facts equals the count the base store held. After `refresh_canonical`: no canonical row carries an old hash
  or the `fact_id` of a superseded fact; deterministic probe SQL over `facts_canonical` for each replaced slot returns only
  rows whose `doc_hash` is the new hash. **GPU row (same gate, behavioural half):** every P-large SQL statement of the
  58 questions is re-executed read-only; no returned row equals a superseded fact by `(doc_id, field, period, value, quote)`
  unless the new document independently carries the identical row. Any hit trips G2.
- **G3 cost (gate).** Fresh LLM calls and fresh prefill tokens of the re-ingest (the 24 new L4 units only; seeded units
  are not counted) are each **<= 3/28 x the full AGG-1.1 ingest x 1.10**. Full AGG-1.1 ingest (the recorded `ingest`
  record, Phase 0 reused units included): 1,359 calls, 1,712,985 prefill tokens. Bounds: **160 calls** (160.16) and
  **201,892 prefill tokens** (201,892.6); the plain 3/28 shares are 145.6 calls and 183,534 tokens. The same three documents
  cost 146 calls and 185,976 prefill tokens in the AGG-1.1 pilot, so the expected value sits **about 9% under the call
  bound and 8% under the prefill bound**: a trip is a live possibility (a FY2021 filing could be 10% longer), not a
  formality, and is reported as a G3 trip, never rescued by a rerun with different windows.
- **G4 aggregation (gate).** M1 >= 0.75 on the changed corpus: cells = the AGG-1.1 verified cells of the 25 retained
  documents (187 = 211 - 24, found by the AGG-1.1 rule: a fact in the raw `facts` table, same doc and field, a period
  containing 2022, value within 0.5% after unit scale, no unit leniency) + every filled cell of `agg13_cells.csv` (the
  same rule with period containing 2021). Reported separately for the 25 retained documents (must equal the AGG-1.1 hits
  for those documents, a regression check on "untouched") and for the 3 new filings.
- **G5 per-question cost (gate, GPU row).** M4 unchanged: median P-large calls per question <= 3 and median prefill
  <= 2,510 over the 58 questions on the changed store (the AGG-1.1 M4 bounds; AGG-1.1 measured 2 calls and 606.5
  prefill; unchanged = inside the same Phase 1 bounds, +1 call / +2,000 tokens over Phase 0). The questions are the
  frozen 58 of AGG-1.1; answers are **not scored** (their golds describe the pre-replacement corpus, so a score would
  mix the replacement into M2); the number of questions whose gold depends on a replaced company is reported as
  context. The GPU row also runs the Phase 0 L4c audit over the changed store first, exactly as AGG-1.1 did.

A gate trip files `phase1_gate_tripped` naming the gate. Reported without a gate: wall seconds (parse, L3, L4) per
document, calls and prefill per unit and per document against the AGG-1.1 pilot, SQL latency.

## Stopping rules and pilot

The 3-document ingest is the pilot (it is the whole job). Order: `MICROSOFT_2022_10K`, `JOHNSON_JOHNSON_2022_10K`,
`3M_2022_10K`. **Budget stop (rc 5, re-book, not a gate):** the first changed document costs more than 2x its AGG-1.1
pilot rate (MICROSOFT: 48 calls, 57,618 prefill tokens, 416.2 s of L4 unit wall for 8 units): more than 96 calls, or
115,236 prefill tokens, or 832.4 s of unit wall. **Futility (gate stop):** running fresh calls or prefill over the G3
bound after any unit (the bound can only grow); a parse failure of any of the 3. Formal-gate cells stay futility-only
at interim looks; there is no early GO. Reuse valid evidence first: the AGG-1.1 store, units and facts are reused, never
rebuilt; the AGG-1.1 stores are never mutated (the job holds its own copy; `SOURCE.sha256` seals the code).

## Budget (ESTIMATE; the run replaces it)

CPU row: about 1,500 s on one CPU quarter, derived two ways: 3/28 of the measured AGG-1.1 full ingest wall (L3 of Phase 0
1,658 s + L3 of 18 documents 112 s + L4 7,842 s new + 4,147 s Phase 0 = about 13,760 s; 3/28 = 1,474 s) and bottom-up from
the AGG-1.1 pilot documents (24 units, 1,287 s of L4 wall, parse+L3 about 20 s, plus server start). Booked 1,800 s. The
box's provisional 7,200 s is not supported by the measured rates and is replaced. GPU row (27B audit + 58 P-large
answers, no B1): about 900 s (P-large 6.4 s median and 13.9 s mean per question measured in AGG-1.1, audit about 10 units)
ESTIMATE.

## Honest caveats (read before any verdict)

1. **Three documents, one corpus family, one direction.** FY2022 -> FY2021 replacement of three large-cap 10-Ks. A pass
   supports "the store's content-hash path re-does only changed work here"; it says nothing about a corpus of
   frequently changing documents.
2. **The slot identity is a design choice.** `doc_id` stays the old name so the store's existing replace path is
   exercised; a deployment that keys documents by file name would see 3 new documents and 3 stale ones instead.
3. **G3's margin is about 9%.** The bound is the box's; the pilot documents' own AGG-1.1 cost is already 91% of it.
4. **Gold standard is one reader + the cross-extractor check** for the 3 new filings (as AGG-1.1 caveat 3).
5. **Supersession is a store feature added for this box** (orchestrator commit recorded in the run spec): before
   this, `CorpusStore.ingest` deleted the old rows without a trace, which satisfies "absent" but not "marked superseded".
6. **M4 is measured on a different question-to-corpus mapping** than AGG-1.1 (three companies have no FY2022 rows): the
   SQL answers legitimately differ; only the cost medians are gated.
7. **Parse cache.** The 25 retained documents' parse results come from the AGG-1.1 and Phase 0 content-keyed parse caches,
   so "skipped by hash" is proven at the store (`unchanged` receipt) and at the cache key (PDF sha256 equal to the
   pinned AGG-1.1 manifest), not by re-parsing 25 filings.

## Pins

Frozen by the runner (`run_agg13.py` preflight): `corpus_agg13.json`, `agg13_cells.csv`, `agg13_xcheck.json`,
the AGG-1.1 `facts_sheet.csv`, `golds.jsonl`, `verification_record.json`, `aggregation_questions.json` (sha256 in the
AGG-1.1 table, unchanged), the Phase 0 code pins of AGG-1.1, and the AGG-1.1 `result.cpu.json` / `results.jsonl`
numbers above. Orchestrator and research commits are recorded in the run spec. Any mid-run fix needs a dated amendment
to this file BEFORE the rerun and says POST-DATA in its heading if data exists.
