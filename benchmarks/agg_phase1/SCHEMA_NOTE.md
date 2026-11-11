# AGG Phase 1: Jev-techniques note (typed decisions, constrained outputs)

Operator rule: schema-shaped work must weigh Jev techniques and the REPL-retrieval alternative. Non-binding for
the run (the arms are fixed in `PREDECLARED.md`); this note says what could cut calls and tokens and whether it is
an arm or a follow-up. Numbers below are measured from the Phase 0 evidence with no model call.

## What the Phase 0 ingest actually spends

Source: `/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2/out/results.jsonl` (80 `l4_unit` records).

- 483 calls, 613,361 prefill tokens over 80 (field, doc) units. **L4b extraction is 465 calls (96%) and 594,684
  prefill tokens (97%)**; L4a admission is 18 calls; the typed planner is a stub (0 calls).
- Each unit sends the BM25 top-6 windows (<= 10,000 chars) for ONE field, one call per window, about 1,280 prefill
  tokens per call. Prefix reuse is negligible (`cache_n` 6,078 of 613,361) because the loop is field-outer,
  doc-inner, so a window that serves two fields is never resident for the second.
- **Windows are shared across fields**: recomputing the same BM25 candidates on the Phase 0 store gives 480
  (field, window) pairs but only **311 distinct windows** (1.54x by count, 1.68x by characters, 2,074,799 vs
  1,233,762 chars). Income-statement, balance-sheet and cash-flow windows each serve 2-3 of the 8 fields.
- Answer path: P-large aggregation questions take 2 calls (grammar-constrained SQL, then an answer call that
  restates the SQL result; most Phase 0 answer texts just echo the returned value, but agg03, agg15, agg16 and
  agg17 show the model reasoning over returned rows). Wall 7 s mean per question, so the answer arm is not where time goes; the ingest is.

## Techniques that would cut calls and tokens

| # | Technique | Mechanism | Expected effect (estimate) | Risk |
|---|---|---|---|---|
| A | Window-keyed typed multi-field extraction | one L4b call per DISTINCT window, GBNF object with the 8 field slots, each `{value, unit, quote} or null`, quote check unchanged | L4b calls -35% (311 vs 480 per 10 docs), prefill -41% (1.68x by chars); decode grows a little | recall can move either way (a window now answers fields it did not rank for): must be re-measured by M1 |
| B | Prompt layout + loop order only | window text first (cached prefix), field description last; doc-outer, field-inner, `id_slot` pinned | same calls, prefill up to -40% via slot-cache hits, zero schema change | cache eviction under `-np 2`; verify with `cache_n` |
| C | Deterministic answer slot | for single-row aggregates the SQL result is formatted to `FINAL:` by code, no answer call | median calls 2 -> 1, median prefill about 500 -> 250 | only valid when SQL does the whole aggregate (agg03/15/16/17 show the LLM sometimes reasons over returned rows itself); keep the LLM answer as fallback |
| D | Typed planner Choice (+Score) | single constrained token over {whole_doc, repl, retrieval, sql}; already designed in AGG-0.3 | 1 call of a few tokens inside the ambiguous band only | stays a stub in Phase 1: no ambiguous-band question exists in the 58 |
| E | Typed B1 actions | SEARCH/READ/FINAL as a grammar-constrained action token | fewer wasted B1 calls (13 of 20 Phase 0 B1 aggregation runs hit the 30-call cap) | it improves the BASELINE; useful only to separate "grammar helps" from "architecture helps" |

REPL-retrieval alternative (weighed against the schema route, as the standing constraint requires): B1 is that
alternative. Phase 0: B1 aggregation 0.20 at median 30 calls and 78,933 prefill tokens per question, versus
P-large 0.55 at 2 calls and 498 tokens. Per-question, the typed route is 15x cheaper in calls once the one-time
ingest is paid; the ingest (483 calls) equals roughly 16 B1 questions at 30 calls, so the break-even is a corpus queried more
than about 17 times (483 calls / 28 calls saved per question). That is a per-corpus fact about Phase 0, not a Phase 1 result.

## Arm or follow-up?

**Follow-up, not an arm of AGG-1.1.** Reasons: (1) the AGG-1.1 gates are "M3 and M4 unchanged versus Phase 0" and
runs are on pinned commits; any of A-C changes the code under test and confounds exactly those gates. (2) Reusing
the Phase 0 L4 units (the default, see PREDECLARED) requires the L4 stage code to be unchanged. (3) Arms multiply
GPU and CPU windows for a question (M1, M3, M4 at 28 docs) that does not need them.

Proposed follow-up (one row, after AGG-1.1 Step B): `AGG-1.6 typed-ingest A/B` on the AGG-1.1 pilot documents
(MICROSOFT, JOHNSON_JOHNSON, 3M): technique B alone, technique A alone, A+B, against the AGG-1.1 pilot's own
calls, prefill and M1 for the same 3 documents (so the baseline costs nothing extra). Question: calls and prefill
per unit down by >= 30% at M1 within 0.03 of baseline. Bounded pilot <= 1,800 s per variant. Technique C is a
separate zero-risk code change measured on the existing 58 questions' SQL results.
