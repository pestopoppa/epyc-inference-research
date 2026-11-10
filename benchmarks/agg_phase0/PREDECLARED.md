# AGG Phase 0 — predeclared metrics and kill gate

Frozen before any run. Source: `aggregation-plan-fable.md` rev 2 §(c) (2026-11-10). The runner records this
file's sha256 in every results file; a run whose sha256 differs from the one recorded in the AGG-0.1 box is
not a Phase-0 result. Thresholds are mirrored in `agg_phase0.py` `KILL` (a test asserts they match).

## Question

Does the integrated planner + lazy fact layer (L3 deterministic + L4 LLM) answer cross-document aggregation
over a 10-filing FY2022 corpus materially better than REPL + hybrid retrieval, at bounded ingest cost, while
costing nothing on single-document questions?

## Corpus and questions

- 10 FinanceBench FY2022 10-Ks (`corpus.json`, sha256-pinned): FB5 + AES, Coca-Cola, General Mills,
  Lockheed Martin, PepsiCo.
- 20 constructed cross-document aggregation questions (`aggregation_questions.json`, golds `golds.jsonl`
  derived from the hand-filled `facts_sheet.csv`).
- 10 original FinanceBench single-fact questions (`single_fact.jsonl`), used in P-large (whole corpus) and,
  the same 10, in P-small and B-today (each against its own 10-K only).
- **Only questions with `verified: true` in `verification_record.json` run. Unverified = excluded, never
  assumed.** The excluded count is reported next to every metric.

## Arms

| Arm | Questions | Path |
|---|---|---|
| P-large | 20 agg + 10 single | planner -> `sql` (L3 + lazy L4) over the 10-doc corpus |
| P-small | 10 single | planner on the one 10-K |
| B1 | 20 agg + 10 single | REPL agent + hybrid retrieval over the same chunks, 27B GPU, <= 30 calls |
| B2 | 5 agg (labelled, first 5 by id) | `ingest_long_context` whole-document read per doc, then aggregate |
| B-today | 10 single | current flow unchanged (routing_decision token rule + whole-doc / REPL) |

## Metrics

- **M1 table recall** — share of the 20 agg questions' gold evidence cells (`facts_sheet.csv`, status
  `filled`) present in `facts` after L4d (value within 0.5% relative, same doc/field/period). Isolates ingest.
- **M2 answer accuracy** — CJ-8 three-valued local judge on agg answers; exact string/number check on the
  single-fact 10 (numbers within 1% or the FinanceBench rounding).
- **M3 ingest cost** — prefill tokens / corpus tokens, and LLM calls / 100 chunks, for P-large ingest.
- **M4 per-question cost** — median calls and median prefill tokens per P-large question.
- **M5 small-corpus parity** — P-small vs B-today on the same 10: correct counts within 1 question; per
  question at most +1 call and +2,000 prefill tokens; **no L2/L3/L4 index in the store manifest** for P-small.

## Kill gate (any trip -> `superseded_v2_not_viable`, name the gate)

| Gate | Trips when |
|---|---|
| M1 | M1 < 0.70 |
| M3 | prefill / corpus tokens > 2.0, or calls / 100 chunks > 150 |
| M2 | P-large agg accuracy - B1 agg accuracy < 25 pp (n = 20, binomial SE ~ 0.10, ~2.5 SE) |
| M4 | median P-large calls per question > 8 |
| M5 | any M5 parity condition fails |

No trip -> `go_phase_1`.

## Stopping rules

- **Stop-on-futility 1:** M5's manifest check is read first (zero compute); a built L2-L4 index on P-small stops
  the run.
- **Stop-on-futility 2:** M1 is computed after CPU ingest and **before** the MI210 window is requested; M1 < 0.70
  stops the run (M1 gate tripped, GPU never requested).
- Formal-gate cells stay futility-only at interim looks (no early GO).
- Every record is appended to `results.jsonl` as produced (numeric per-stage record early).

## Amended pre-data 2026-11-10 (before any Phase-0 model call; no results exist)

Operator/coordinator decisions of 2026-11-10, recorded before any data. Kill-gate thresholds and the
`KILL` mirror in `agg_phase0.py` are unchanged.

1. **Scoring: deterministic scorer replaces CJ-8.** Reason: every gold is numeric or a closed-set company
   name, so a deterministic scorer removes judge variance. Implementation: `phase0_arms.score_agg` /
   `score_single`, applied to the answer's `FINAL:` line (else its last line; no line = `correct: null`,
   counted as not correct).
   - Numeric answers match when |answer - gold| <= max(0.5% x |gold|, half a unit of the gold's last stated
     decimal). Unit and scale words (thousand / million / billion, $, M, bn) normalise to USD millions
     before comparing; `%` values compare in percent units. Why 0.5%: it is the M1 cell tolerance. It absorbs
     rounding to the nearest $1M on 10-company sums (~5e5) and $0.1B restatements of a ~$10B line, and it
     still rejects a wrong line item. Those typically differ by >= 1%. The half-unit term honours
     FinanceBench's stated rounding (e.g. 0.4% with one decimal).
   - Counts must be the exact integer. Company names (argmax/argmin/top3) use an exact normalised match: lowercase,
     punctuation and legal suffixes (inc, corp, corporation, company, co, plc, ltd, llc, the) removed. top3 needs
     all three in order. Prompts list the corpus company names verbatim.
   - Single-fact golds are encoded as `single_fact_keys.json` (yes/no where the gold states it, plus the
     gold's numbers; money compared by absolute value when the yes/no carries the sign, signed for AES ROA).
2. **Planner typed call stays a stub for Phase 0.** P-small and P-large take the deterministic size rules;
   inside the ambiguous band the receipt records `decided_by: deterministic_default_stub` (`stub: true`). No
   planner LLM call is made or counted.
3. **Hardware and servers.** L4a/L4b/L4d run on worker_general (Qwen3.6-35B-A3B-MTP-Q8_0, private CPU-quarter
   server, `-np 2`). The 27B (Qwen3.8-27B-Q8_0 + DFlash2) runs in a private `--gpu-quiet exclusive` MI210 window.
   It runs as two sequential servers, one at a time. The **np4** server
   (`qwen3.8-27b-q8-gpu-dflash2-np4.json`, 4096 tokens per slot) serves only the L4c audit sample. **All answer arms**
   (P-large answers incl. the SQL step, P-small, B1, B2, B-today) run on the **np1 196k** server
   (`eval/qwen3.8-27b-q8-gpu-dflash2-longctx196k-np1.json`). Any request that does not fit the window is recorded
   `status: out_of_window` (`correct: null`), never truncated.
4. **Arm details fixed before data.**
   - L4 schema seed = the 8 metric fields above, with their descriptions. Candidates are BM25 top-6 windows
     (<= 10,000 chars) per (field, doc). sim = score / best score in that doc. Admit at >= 0.6, reject at <= 0.2, and
     between them a logprob yes/no gate (p(yes) >= 0.5).
   - M1 counts a gold cell found when a fact for the same doc and field, with a period containing 2022, matches
     |value| within 0.5%.
   - L4d residual conflicts stay `disputed` (no native-catalogue call in Phase 0).
   - B1 retrieval is **lexical BM25 only** (no L2 dense index exists in Phase 0), SEARCH/READ/FINAL, <= 30 calls.
   - B-today is the whole-document read that routing_decision's long-context rule selects for a 10-K-size request
     (ingest_long_context), served on the np1 27B.
   - P-large SQL returning no rows is recorded `backfill_needed` (lazy L4 for a new field is a CPU step, not run
     inside the GPU window).
   - P-small and B-today whole-document reads are sent with `cache_prompt: false`, so each pays a full prefill
     and M5's prefill parity cannot depend on which arm warmed the slot. Two questions share a filing (Corning,
     Verizon). B2 reads per (doc, question) **doc-outer**, with the filing first and the question last, so the
     filing prefix is reused across its 5 questions. B2's cost counts every per-doc note call plus the
     aggregate call.
   - `prefill_tokens` everywhere = tokens actually prefilled by the server (`timings.prompt_n`); slot-cache
     reuse is recorded separately as `cache_n`.
