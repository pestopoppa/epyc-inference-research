# AGG Phase 1 (AGG-1.1) - predeclared question, metrics, n, arms, gates, stopping rules and budget

Written 2026-11-11 by the AGG-1.1 step A session, **before any Phase 1 model call or data exists**. It mirrors the
structure of `benchmarks/agg_phase0/PREDECLARED.md` (sha256 `976a5fe87e3db33d6a5e4547ee4bb6cae324f44d107980342b62f7b6212b8bc9`,
rerun amendment included, **unchanged by this commit**: a test pins it). Phase 1 is a separate file rather than an
amendment to that one, so the Phase 0 verdict keeps citing the exact bytes it ran under. The runner records this file's
sha256 in every Phase 1 results file; a run whose sha256 differs from the one recorded in the AGG-1.1 box is not a Phase 1
result. A file cannot carry its own hash: the sha256 is recorded in the AGG-1.1 box, the compute row and the run spec.
Kill-gate thresholds are the Phase 0 `KILL` constants in `agg_phase0.py` (imported, not copied).

## Question

Does the Phase 0 result (an integrated planner + lazy fact layer, L3 deterministic + L4 LLM, answers cross-document
aggregation better than REPL + hybrid retrieval at bounded ingest cost) survive 2.8x more documents and 2.3x more
tokens, on filings and question families that were not in Phase 0, inside the Phase 0 kill bounds? Scope is exactly:
integrated planner + lazy fact layer on Qwen3.6-35B-A3B (L4 ingest) / Qwen3.8-27B (audit, answers), 28 FinanceBench FY2022
10-Ks, about 4.5M tokens (estimate: 17,898,545 extracted characters / 3.98 characters per token, the Phase 0 ratio).
It concludes nothing about other models, corpora, scales beyond 28 documents, or SLIDERS' viability.

## Honest caveats (read before any verdict)

1. **The Phase 0 GO was reached after post-data fixes.** The first Phase 0 run was filed FAIL (`superseded_v2_not_viable`,
   M2 and M5). The three defects were fixed after the data existed (orchestrator `9a0ab00d88ffdccab3dd9f25d7f5db26c93b7069`
   canonical facts table + single-document planner scope; research `13afe97c` canonical SQL path, B-today overflow
   loop, call cap), and the rerun passed (`go_phase_1`: M1 0.92, M2 P-large 0.55 vs B1 0.20, M3 0.31 / 19.4, M4 2 calls /
   510 prefill). So Phase 0 shows the technique works **with the defects removed**, on n = 20 aggregation questions over one
   10-filing corpus, about 3 SE of effect. Phase 1 inherits that implementation as the system under test and is the
   first measurement on code and data not tuned against the failures: it is not a replication of the Phase 0 verdict,
   and a Phase 1 pass does not retroactively make the post-data fixes blind.
2. **Phase 1 is not blind to Phase 0's failure modes.** The question families and the NA/exclusion conventions were
   written by people who had read the Phase 0 failures (duplicates, NA handling, dividends). Mitigations: questions,
   golds and thresholds are frozen below before any model output exists; thresholds were chosen by a margin audit on the
   gold values (min relative gap >= 1.5%, no near-tie in the 58), never by looking at any system's answers; the scorer is the Phase 0
   deterministic scorer unchanged.
3. **Verification standard is weaker for the 144 new cells than Phase 0's.** Phase 0's 80 cells had two independent blind
   readers. The 144 new cells (18 documents x 8 metrics) have one reader (the step A agent reading the extracted page text)
   plus a deterministic cross-extractor check: each quote must be found, whitespace/`$`-normalised, on its stated PDF page in
   BOTH pypdfium2 and pypdf text, and its value must equal a number printed in the quote (`verify_phase1_cells.py`,
   `facts_sheet_xcheck.json`: 224 cells, 220 checked, 0 fail; the 4 unchecked are Phase 0 NA cells with no quote).
   82 of 224 cells print their figure on only one page, so there is no second page to corroborate. 10 new cells are
   medium confidence and listed in `facts_sheet.csv` (`confidence=medium`: four no-dividend or ambiguous-dividend NAs (Adobe, Amazon, AMD, AmEx),
   AmEx capex net of $1M sales, Apple dividends incl. dividend equivalents, eBay OCF incl. discontinued, Kraft Heinz
   dividends line, MGM unlabeled total-assets line, Netflix OCF merged line). No LLM call produced or scored any value.
   If the operator wants the Phase 0 two-reader standard for the new cells, a blind second sheet must be committed
   BEFORE Step B; it changes no value unless a cell is disputed.
4. **One corpus family.** All 28 documents are FinanceBench FY2022 10-Ks and the questions are template-derived over 8
   metrics. A pass supports "the integrated stack scales to 28 such filings"; it does not support unseen question types.
5. **n.** 58 questions: P-large vs B1 is paired, SE of the difference about 0.08 at n = 58 (see Expected effect).

## Corpus and questions (frozen inputs)

| File | sha256 |
|---|---|
| `corpus.json` (28 docs: path, sha256, bytes, pages, source URL; `manifest_digest_sha256` over the docs array) | `bdeb7a978f036a8e2599bc6db752899d5d18bd57f5fbb8ac52a6dd3a1da41f84` |
| manifest digest (canonical JSON of the docs array) | `5c834d5e05c224365405a4a9d99e5b3184d55447991df91840c54af5562b3705` |
| `facts_sheet.csv` (28 docs x 8 metrics = 224 cells; 211 filled, 13 NA; page + verbatim quote per cell) | `9f1bbfe34857532306da65ccbad6a6fceed3ab7db9c9fd3fe24da239cfe074aa` |
| `facts_sheet_xcheck.json` | `49b747d0d08abf0d29a99ac1bc552ac29fa35b7084c5b7dcd45a4f20a840df1c` |
| `aggregation_questions.json` (58 questions) | `92e9d389f65fb6d5576ba276095076e9edcfedb36916d01a890e4b01cb30939f` |
| `golds.jsonl` (golds derived by script from the sheet; per-cell evidence doc/metric/value/page) | `fbe97354a73d23e32f84224d2eb629be1813ed208839a0a82c43a120e1b35e1f` |
| `verification_record.json` | `704352675d2e5fa60b247099367cba6290acfca786c231eaaabde5db5e6b386b` |

- **Corpus**: the 10 Phase 0 documents (same PDFs, same sha256 pins, checked against `agg_phase0/corpus.json` at build) +
  18 more FinanceBench FY2022 10-Ks: 3M, Activision Blizzard, Adobe, Amazon, AMD, American Express, Apple, Boeing,
  Costco, CVS Health, eBay, Johnson & Johnson, Kraft Heinz, McDonald's, MGM Resorts, Microsoft, Netflix, Pfizer
  (downloaded once, sequentially, from `https://github.com/patronus-ai/financebench/raw/main/pdfs/`; local copies under
  `/mnt/raid0/llm/datasets/financebench-20261111-agg1/pdfs/`). "FY2022" is each filing's own fiscal 2022 (Adobe
  2022-12-02, Apple 2022-09-24, Costco 2022-08-28, Microsoft 2022-06-30, J&J fiscal year ended 2023-01-01).
  Deliberate traps: USD thousands (MGM Resorts, Netflix), reversed column order (Amazon: 2020 2021 2022), net losses
  (Amazon, Boeing, eBay, AES), no-dividend filers (Adobe, Amazon, AMD, Boeing, Netflix, PayPal), an unclassified balance sheet
  (American Express, JPMorgan).
- **Questions: 58 aggregation questions.** 20 continuity (`agg01`-`agg20`: the Phase 0 question set, same field, op, threshold
  and text, **re-evaluated over the 28-document corpus**, so their golds are new numbers; `agg18` text and exclusion list
  changed from "other than JPMorgan" to "other than JPMorgan and American Express" because AmEx's cash-flow dividend line also
  combines preferred and common, and the 28-document gold would otherwise silently omit it) + 38 new (`p1_01`-`p1_38`:
  net margin, net income, revenue, cash flow, free cash flow, capex, balance sheet, current ratio, dividends and payout; ops
  argmax / argmin / top3 / sum / count_gt / count_lt only, with `where` filters and `exclude_docs` stated in the question text).
  No single-fact questions in AGG-1.1 (single-document parity is AGG-1.5). The 10 Phase 0 documents' gold cells are the Phase 0 verified cells,
  copied byte-for-byte; the continuity set is therefore comparable to Phase 0 in wording, not in gold values or n of companies.
- **Only questions with `verified: true` run; unverified = excluded, never assumed; the excluded count is reported next to every
  metric.** Currently 58 of 58 verified, 0 excluded. A question is verified only if every contributing cell is a verified cell.
- Conventions inherited from Phase 0: an NA cell drops the company from that question (JPMorgan capex and current
  assets/liabilities; AmEx current assets/liabilities; no-dividend filers' dividends); capex and dividends are positive
  magnitudes of outflows; net income is attributable to the company's shareholders; revenue is the statement's total revenue
  line (Costco: total revenue incl. membership fees; AmEx: revenues net of interest expense); money in USD millions.

## Arms

| Arm | Questions | Path |
|---|---|---|
| P-large | 58 aggregation | planner -> `sql` over the canonical facts table of the 28-document store (L3 deterministic + lazy L4 on Qwen3.6-35B-A3B), answers on the Qwen3.8-27B np1 196k server; Phase 0 recipes and rerun-amendment fixes unchanged |
| B1 | 58 aggregation | REPL agent + lexical BM25 hybrid retrieval over the same chunks, 27B GPU, <= 30 calls (Phase 0 B1 unchanged) |

No B2, P-small or B-today in AGG-1.1. Local only (no cloud baseline). One integrated stack (every arm runs through
`src/aggregation/` plus the current document flow, no side system).

## Metrics and gates

- **M1 table recall (gate)**: the Phase 0 rule, over every filled verified cell that feeds a verified question: **211 cells**
  (75 Phase 0 + 136 new). Found = a fact in the raw `facts` table (L3 union L4, after L4d; M1 input untouched by the
  canonical table) for the same doc and field with a period containing 2022 and value within 0.5% relative.
  Reported per document, for the 10 Phase 0 documents (75 cells) and for the 18 new (136) separately. No unit leniency: a fact
  stored 1000x off because a filing is in thousands is a miss (MGM Resorts and Netflix, 16 cells, are the designed test of this).
- **M3 ingest cost (gate)**: prefill tokens / corpus tokens and LLM calls / 100 chunks over the 28-document ingest.
  If Phase 0 L4 units are reused (below), the gated value is (Phase 0 recorded 613,361 prefill, 483 calls, 2,484 chunks,
  1,961,062 corpus tokens) + (Phase 1 incremental), and the incremental-only value is reported beside it.
- **M4 per-question cost (gate)**: median calls and median prefill tokens per P-large question over the 58.
- **SQL latency (reported, no gate)**: median, p95, max seconds of the generated-SQL execution on the 28-document store, and
  median/max wall per P-large question (Phase 0 comparator: mean 7 s, max 26 s over 20 questions, LLM calls included).
- **M2 answer accuracy (informational, any outcome reported, no gate in AGG-1.1)**: deterministic scorer
  (`phase0_arms.score_agg`, unchanged), P-large vs B1 on the 58; also reported for the 20 continuity and the 38 new
  separately, with paired counts (both right, only P-large, only B1, neither).

### Gates (any trip -> `phase1_gate_tripped` naming the gate; no trip -> AGG-1.1 DONE)

| Gate | Trips when |
|---|---|
| M1 | M1 < 0.75 after L4d on the 28-document corpus (Phase 0 kill was 0.70; the Phase 1 gate is the box's 0.75) |
| M3 | prefill / corpus tokens > 2.0, or calls / 100 chunks > 150 (Phase 0 kill bounds) |
| M4 | median P-large calls per question > 8 (kill), or median calls > 3, or median prefill > 2,510 (Phase 0 medians 2 calls / 510 prefill, +1 call / +2,000 tokens) |

Phase 0 comparators: M1 0.92 (69/75), L3-only 0.827; M3 0.31 and 19.4; M4 2 calls and 510 prefill over 30 questions
(aggregation only: 2 calls and 498 prefill; the gate reads the 58 aggregation questions against the 2 / 510 pair).

## Expected effect versus sampling error

- **M1**: n = 211 cells. Expected 0.85-0.92 (Phase 0 0.92 after L4d; 0.83 L3-only); binomial SE at 0.90 is 0.021, the gate
  0.75 is about 7 SE below; at a true 0.80 the SE is 0.027 and the gate is about 1.9 SE away, which is the region where a
  verdict is genuinely uncertain. Risk factors named in advance: USD-thousands filings, AmEx/JPMorgan unclassified statements
  (NA cells excluded, not counted), reversed-column Amazon.
- **M2 (informational)**: n = 58 paired questions. If Phase 0 replicates (P-large 0.55, B1 0.20), the SE of the difference is
  about sqrt(0.55 x 0.45/58 + 0.20 x 0.80/58) = 0.084, so +0.35 is 4.2 SE; if the effect halves (+0.175) it is 2.1 SE.
  B1 degrades with corpus size (13 of 20 Phase 0 runs hit the 30-call cap), so a larger gap is plausible.
- **M3, M4**: deterministic accounting, not sampling-limited. Phase 0 slack against the kill bounds is 6x (prefill ratio) and
  7.7x (calls/100 chunks), so a trip needs a structural change in cost, not noise. Predicted from linear scaling in
  (field, doc) units (80 -> 224 units): about 1.35k calls and 1.7M prefill tokens in total, prefill ratio about 0.38, calls /
  100 chunks about 24 (calls scale with units, 2.8x; chunks with tokens, 2.3x).

## Checkpoints and stopping rules

Formal-gate cells (M1, M3, M4) stay **futility-only at interim looks** (no early GO; GO needs all 28 documents ingested and
the GPU arms complete). Reuse valid evidence first; save incremental results (fsynced per unit, resumable); stop when the
scoped question is answered.

0. **Preflight (zero compute)**: PREDECLARED sha256, orchestrator pin and research commit recorded in the spec; the 28 PDF
   sha256 equal `corpus.json`; `build_phase1.py check` exits 0; 58 of 58 verified. **L4 reuse test**: the L4 stage sources
   (`src/aggregation/facts_llm.py`, `llm_backend.py`, and the L4 loop of `phase0_arms.run_cpu`) are diffed between the Phase 0 CPU
   job's SOURCE.sha256 pin and the Phase 1 pin; the 80 Phase 0 L4 units (10 documents x 8 fields) are reused as cache hits
   only if that diff is empty, otherwise all 28 documents are ingested fresh and M3 is measured fresh.
1. **L3 load of the 18 new documents** (CPU, no model, Phase 0 `l3_load.py` path). Futility: >= 3 of 18 fail to parse
   (infrastructure defect: stop, fix, dated amendment, rerun). Fewer failures are recorded `doc_fail`, their cells count as
   M1 misses (never silently dropped).
2. **Pilot (bounded <= 1,800 s wall)**: L4 ingest of 3 new documents fixed in advance as ranks 5, 11 and 17 of the 18 new
   documents by extracted size: **MICROSOFT_2022_10K, JOHNSON_JOHNSON_2022_10K, 3M_2022_10K** (24 units, 24 gold cells).
   It replaces every provisional estimate: calls per unit, prefill per unit, seconds per unit, calls / 100 chunks,
   prefill ratio, pilot M1. **Futility (stop, file the gate)**: pilot projection of M3 beyond a kill bound -> M3 tripped;
   pilot M1 < 0.60 on its 24 cells (expected >= 0.85, about 3.4 SE below) -> M1 tripped; projected ingest wall > 2x the booked
   estimate -> stop and re-book (a budget stop, not a gate).
3. **Interim look after 10 of the 18 new documents** (pilot included): futility only: the gate is mathematically
   unreachable (`found + not_yet_evaluated < 0.75 x 211`) -> M1 tripped; running M3 over a kill bound -> M3 tripped.
4. **Final ingest look** (all 28 documents): compute M1 and M3. M1 < 0.75 or M3 over a bound stops here: **the GPU window is
   never requested** (Phase 0 futility 2).
5. **GPU phase**: audit sample (Phase 0 L4c recipe), then P-large on the 58, then B1 on the 58 in order continuity first,
   then new by id. **Stop-on-answer for B1 only** (M2 is informational): after B1 has answered 30 questions, if P-large minus
   B1 on those 30, paired, is >= 3 SE (SE from the discordant pairs), B1 stops and M2 is reported on the answered subset with
   its n (paired z = (b - c) / sqrt(b + c), b = only P-large right, c = only B1 right). Otherwise B1 runs to 58. M4 is computed on all 58 P-large answers.
6. **Zero-compute reuse**: golds, verified cells, the Phase 0 store (for the 10 documents' L1/L3/L4 units) and the L3 load of
   Phase 0 are reused, never rebuilt.

## Budget caps (estimates are estimates; the pilot replaces them)

| Item | Estimate (basis) | Booked cap |
|---|---|---|
| L3 parse + load, 18 new docs | about 3,000 s CPU, no model (Phase 0 r3: 1,658 s for 10 docs; ESTIMATE) | 1.5x |
| L4 ingest, Phase 0 units reused (144 new units) | about 7,500 s on a CPU quarter (Phase 0 measured 4,147-5,066 s for 80 units, 52-63 s/unit, 483 calls; ESTIMATE) | 1.25x of the pilot-based projection; hard stop at 1.5x |
| L4 ingest, if no reuse (224 units) | about 11,600-14,200 s (ESTIMATE; the handoff's provisional 54,000 s scaled a figure that the Phase 0 evidence does not support) | same |
| GPU window | about 5,000 s if B1 stops at 30 questions, about 8,000 s if B1 runs to 58 (Phase 0: B1 109 s per question, P-large 7 s, audit 3 units; ESTIMATE) | 9,000 s |
| Calls per unit | Phase 0 5.8 | stop and re-estimate above 12 |

No new downloads, no cloud model, no experimental-branch kernel change; production kernel v10 served from the kernel store.

## Pins and amendments

- Orchestrator: `9a0ab00d88ffdccab3dd9f25d7f5db26c93b7069` (facts_canonical, single-document planner scope).
- Research: the commit that adds this file (recorded in the AGG-1.1 box and the run spec; a file cannot carry its own commit).
- The Step B runner wiring (the Phase 0 runner hardcodes the 10-document set, `single_fact` files and `A.DOCS`) lands after this file as
  a code-only commit; it must not change questions, golds, scorer, metrics or gates, and its sha is recorded in the run spec.
- **Any mid-run fix needs a dated amendment to this file BEFORE the rerun; a fix made after Phase 1 data exists is POST-DATA and
  must say so in its heading** (as the Phase 0 rerun amendment does). Results from before and after stay on record.
- Phase 1 closes AGG-1.1 only: AGG-1.2 to AGG-1.5 have their own gates and do not inherit a pass.
