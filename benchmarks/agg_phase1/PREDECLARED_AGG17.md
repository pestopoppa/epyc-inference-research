# PREDECLARED — AGG-1.7 `context.search` lexical recall (written 2026-11-16, before any candidate result)

Box: `handoffs/active/knowledge-pipelines.md` AGG-1.7 (epyc-root). Runner: `agg17_recall.py` (this directory).

## Question
On the 20 AGG-1.5 label queries the gold quote is in the top 8 passage hits for 16/20 (misses p15_09, p15_10,
p15_11, p15_16), for `context.search` and the `l1_passage_bm25` P arm alike. Why, from the engine's own scores and
tokenization, and does a cheap ranking change raise recall@8 on the labels without losing any recorded r1/r2 SEARCH
string (`agg15_trace_queries_r1_r2.json`: 45 records, 16 unique (qid, query) pairs)?

## Method (zero model, no server, no GPU)
- Corpus text: the stored chunks of the r3 parity store (`agg15-parity-gpu-20261111-r3/out/parity_store.sqlite`, copied
  once, source read-only) joined by blank lines, per document; engine = pinned orchestrator `cc613c22`
  (`chunk_sections`, `tokenize`, `BM25Index`, `ContextSearchIndex`, dense off).
- Gold = the question's verified quote (whitespace/$-folded, the AGG-1.5 `sq` test) inside a top-8 passage.
- Baseline must reproduce 16/20 with exactly the 4 misses, else infrastructure stop (rc 4), no candidate is run.
  (A zero-candidate baseline-only smoke was run while building the runner to test the plumbing: 16/20, same 4 misses,
  trace 12/16 unique. No candidate was run and no miss was diagnosed before this file.)
- Diagnosis per miss: gold passage rank and BM25 score, rank-8 cutoff, per-term tf/idf in gold vs top 1, passage
  lengths, number of passages holding every query term, whether the gold passage carries a line-label table row.
- Candidates, in this order (research re-rankers over the v1 engine's own BM25 top-64; builder labels
  `l1_passage_bm25.v2-cand-*`; `l1_passage_bm25.v1` is never edited and the AGG-1.5 parity claim stays on v1):
  1. `c1_label_norm` — statement-boilerplate query normalisation (fixed stoplist in the runner);
  2. `c2_line_label` — passages with a line that STARTS with the query phrase followed by a number/`$`/`(` first;
  3. `c3_norm_line_label` — 1 then 2;
  4. `c4_numeric_row` — BM25 x (1 + numeric-token fraction).
  Not available, recorded not assumed: L3 row pointers (the r3 store's `tables_` is checked; expected empty) and dense L2
  (no embedder: the box forbids a model or server).

## Metrics
recall@8 on the 20 labels; on the 16 unique trace pairs and the 45 trace records; misses fixed; labels/trace lost vs
baseline; ms per query (cost). No other metric decides.

## Stop rules
- STOP-ON-ANSWER: after the first candidate (in the order above) at 20/20 labels with zero trace pairs lost; the rest
  are recorded `not_run`.
- STOP-ON-FUTILITY: all four run and none fixes any of the 4 misses -> verdict `futility`.
- Otherwise -> `partial_gain` (each candidate's before/after filed).
- Budget: the whole run is <= 600 s (bounded pilot of the box); a STOP file ends it between phases (resumable: each
  phase is a file and is skipped on rerun).

## Honesty
No candidate is promoted to a builder by this run; a winner is a proposal for `l1_passage_bm25.v2` with its own
parity check. Changes to the candidate list after any candidate result need a dated amendment and are POST-DATA.
