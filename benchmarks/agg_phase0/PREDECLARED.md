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
