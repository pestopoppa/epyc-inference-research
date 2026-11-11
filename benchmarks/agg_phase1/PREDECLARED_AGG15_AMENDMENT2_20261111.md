# AGG-1.5 - amendment 2, 2026-11-11 (POST-DATA): the P arm indexes passages, not stored chunks; repeated-SEARCH guard; third attempt

Written 2026-11-11, **after AGG-1.5 data exists from two runs**. This is a POST-DATA amendment to `PREDECLARED_AGG15.md` (sha256
`74c219d6cd8200c20b011fd381afe4d4df74f626b5244f80d7350fcb2ad885fc`, byte-for-byte unchanged, still the frozen rule set for
questions, golds, scorer and gates) and a successor to `PREDECLARED_AGG15_AMENDMENT_20261111.md` (amendment 1, sha256
`5b121cbf40240824787bf4cc2ec4ba4796814d50078ab00105a805b94c4697c0`, unchanged, still in force for the 800-character read span and the
repeated-READ guard). A file cannot carry its own hash: this file's sha256 is recorded in the commit message of the commit that adds it,
in the r3 run spec and in the AGG-1.5 handoff note.

**This is the third attempt.** The operator decided on 2026-11-11 (AGG-1.5): "Third try now". The fix below is **post-data**: it was
chosen after reading the r1 and r2 traces and after a zero-model offline replay of their query strings, not predeclared. A pass in r3
supports the M5 claim only for the amended P arm; it does not erase r1 or r2.

## Both earlier runs stay on record

- **r1** `agg15-parity-gpu-20261111` (research pin `00d8c98a3c20a7239a005258c0fb92ae804fe400`): pilot only, filed `fail` /
  `phase1_gate_tripped`, component `prefill`. P-retrieval looped on PEPSICO p15_02 (11 SEARCH, 1 READ, 12-call cap). Evidence
  `/mnt/raid0/llm/tmp/agg15-parity-gpu-20261111/out/results.jsonl`.
- **r2** `agg15-parity-gpu-20261111-r2` (research pin `0522f3fc52b046936b4e45acb84a678438babee4`, amendment 1): pilot only, filed
  `fail` / `phase1_gate_tripped`, component `prefill` (`max_extra_prefill` 24,284, `max_extra_calls` 9). Pilot accuracy P 3/5 vs B 4/5.
  P hit the 12-call cap on PEPSICO p15_02 (26,591 prefill tokens; 5 repeated SEARCH strings, no gold read) and on JPMORGAN p15_05
  (22,665 prefill; 4 repeated READs of the same chunk). B (`context.search`) answered every pilot question in 3-4 calls. Evidence
  `/mnt/raid0/llm/tmp/agg15-parity-gpu-20261111-r2/out/results.jsonl`.

r3 writes to its own `out/`; r1 and r2 results are not reused, edited or counted as r3 data. r3 supersedes r2 as the row, not as evidence.

## Diagnosis (zero model, offline replay of the actual SEARCH strings)

Method: `benchmarks/agg_phase1/parity_replay.py`, committed with its output `agg15_replay_r1_r2.json` and the frozen query list
`agg15_trace_queries_r1_r2.json` (the 45 SEARCH strings of both runs' traces, both arms, plus the 20 question label queries). Three
backends over the SAME document text (the store's chunks joined by a blank line, as B's section text is in the runner), no model, no
server: `old_P` (the r1/r2 P arm: registry `l1_bm25` over the stored chunks, a hit shown as `title[:80] | text[:200]`), `B`
(`ContextBundle.search`, lexical) and `new_P` (the r3 arm). "Gold" is the question's verified quote (whitespace-folded) inside the hit's
full text.

Ruled out, with evidence:

- **Tokenizer, BM25 parameters, dense, scope filter: identical.** Both arms call the same `context_search.BM25Index` / `tokenize`
  (k1 1.2, b 0.75); dense is off in both; the planner scoped each question to its one document and B holds that document as one
  section, so there is no scope filter to differ.
- **Ranking recall: not the cause.** old_P put the gold chunk in the top 8 for 18 of 20 label queries (B: 16 of 20) and for 45 of 45
  trace queries (B: 30 of 45). On the failing questions the gold chunk ranked 1st (p15_02 `Total Current Assets`, 1st on four of the
  first six queries) and 3rd (p15_05).

The cause is **chunk granularity, which breaks hit presentation and the read window**:

| measure, gold hit (20 label queries / 45 trace queries) | old_P (r1/r2) | B context.search |
|---|---|---|
| median hit size, characters | 3,147 / 9,789 | 737 / 760 |
| median offset of the gold quote inside the hit | 1,002 / 6,008 | 264 / 322 |
| gold quote inside the 200 characters shown to the model | 1 of 20 / 0 of 45 | 5 of 20 / 2 of 33 found |
| r2's 800-character READ window contained the gold figure | 19 of 20 / 26 of 45 | n/a (hit span is the read) |

1. The P arm indexed the store's L0 chunks whole. They run 3-10k characters (mean 3.1-4.7k per document, max about 10k) against
   `context.search`'s 800-character line-aware passages. A hit was then the HEAD of the chunk, `title[:80] | text[:200]`, while the matching
   text sat thousands of characters in. p15_02: the PepsiCo balance sheet is at offset 6,008 of the 9,789-character chunk
   `c23f592eb1ea5048:00092`, titled "Funding" (section segmentation put the statement under that heading). The query `Total Current
   Assets` ranked it first, but the model was shown "Funding ... We make contributions to pension trusts", could not tell it was the hit,
   never read it in 12 calls, and searched again with new wording. B's first hit for the same query is the 800-character balance-sheet
   passage starting `Cash and cash equivalents $ 4,954 ...`.
2. When the model did read a hit, r2's heuristic window (800 characters around the best query-word occurrence in the chunk) often missed the
   figure: for p15_05 the window held the gold figure on 0 of the 4 queries (the quote is at offset 871 of a 3,147-character chunk; the window
   centred on another occurrence of the phrase), so the model re-read the same chunk four times. Amendment 1's span cap fixed the size of the
   read and hid the content; amendment 1's duplicate-read note only counted the loop.
3. r1 and r2 were therefore the same failure through two read granularities: the index granularity, not the read granularity, differed
   between the arms.

## What changes

1. **P arm index: `l1_passage_bm25`** (epyc-orchestrator `src/aggregation/registry.py`, commit `66889eee54a14fb23ab07aa30f4181a8f8435d40`, on
   `origin/main`). A new registry builder, layer L1, that runs `ContextSearchIndex` (the engine behind `context.search`: line-aware
   800-character passages, BM25, lexical mode) over the corpus text rebuilt from the store's chunks. It is keyed
   `(corpus, content hash, builder, model)` and built ONCE per corpus, then reused: the store-backed, no-per-request-rebuild property that is
   the point of AGG-1.5 is kept, and the manifest layer is L1 (no L2/L3/L4). `l1_bm25` is unchanged. Over the same text, ranking, spans, line
   numbers and scores equal a per-request `context.search`; the tests assert it.
2. **Same pointers and hit presentation in both arms** (`parity_m5.py`): a pointer is `<start>-<end>` into the corpus text, a hit is shown as
   `line N | span[:200]`, a read is at most 800 characters at the pointer (amendment 1's cap, one parser `parse_span` for both arms).
   P keeps the stored chunk each pointer starts in as provenance (`PArm.ptr_chunk`, not shown). r2's `hit_window` heuristic is removed.
3. **Repeated-SEARCH guard, both arms, one shared code path** (`retrieval_loop`): a `SEARCH` whose query equals an earlier one in the same
   question (whitespace-folded and case-folded, because the tokenizer lower-cases and the hits are identical) returns the note
   `(already searched: <query>; its hits are earlier in this conversation. Search different keywords, read a pointer or answer.)` instead of
   hits. It still counts as a call. An empty query is not marked asked. Each answer record gains `dup_searches`; the meta record gains
   `p_builder` and `search_guard`.

Honest scope notes:

- **Parity is now by construction for ranking and hit text.** The replay shows `new_P` identical to B on 20 of 20 label queries and 45 of 45
  trace queries (same pointers, same order, same spans). The accuracy, calls and prefill gates therefore measure model variance and the
  guard, not retrieval quality; what AGG-1.5 still tests that was not true before is the engineering claim: a store-backed index built once
  per document reproduces `context.search` over that document, with one build per document instead of one per request, no L2/L3/L4 row, and the
  same answers. The saving (index builds and build seconds) is still reported as before.
- **The offline numbers do not predict r3's accuracy.** They show P can no longer be worse than B on retrieval; B's own recall is not high
  (label query: gold in the top 8 for 16 of 20 questions, in both arms), and r3 inherits that for both arms equally.
- The repeated-SEARCH guard would have fired on r1/r2 p15_02 (5 repeated queries) but would not have ended that loop: the loop's cause is
  items 1 and 2 above, which the index change removes. The guard is symmetric hygiene.
- B in AGG-1.5 searches the store's chunk text, as in r1 and r2 (`"\n\n".join(chunk texts)`), so both arms see the same bytes. That is a
  property of the experiment, not of production `context.search` over a raw document.

## What does not change (gates unchanged)

`PREDECLARED_AGG15.md` and amendment 1, apart from what is stated above: questions (20, golds 20/20 verified), scorer
(`phase0_arms.score_single`), gates (accuracy |P-B| <= 1 question; calls max(P-B) <= +1; prefill max(P-B) <= +2,000 tokens; no L2/L3/L4 in the
parity store manifest; P-side index builds <= 1 per document), tolerances, the 12-call cap, search k = 8, the 200-character hit snippets, the
prompt and grammar, the pilot rules (<= 900 s; futility: accuracy difference > 2 questions or any question's P-B prefill > 2,000), the
model, server recipe and kernel, the Phase 0 store, the verdict function `parity_m5.score`. No cap changes. No gate is loosened, none is
dropped.

## Rerun

Row `agg15-parity-gpu-20261111-r3` (supersedes `-r2`): the pilot, then the full 20-question phase under the unchanged gates. A pilot trip files
`phase1_gate_tripped` again; no early GO at the pilot; the full phase runs only if the pilot does not trip. Pins: epyc-orchestrator
`66889eee54a14fb23ab07aa30f4181a8f8435d40`; research = the commit that adds this file, the amended runner and the replay evidence (in the r3
spec). ESTIMATE 1,800 s (r2's pilot took 121 s for 5 questions x 2 arms; not a measurement of the full phase).
