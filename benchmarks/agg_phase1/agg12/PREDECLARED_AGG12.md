# AGG-1.2 predeclared - Oolong-real pilot, P-large vs B1 (local 27B), DRAFT written 2026-11-16 before any model call

Sits beside `PREDECLARED.md` (AGG-1.1) and imports its typed-schema answer slot, frozen scorer style and M4 definition. Not edited
after its sha256 is written into the row's `spec.json` and `precond.sh`.

## Question
Box AGG-1.2 (knowledge-pipelines.md): on the REAL Oolong split (oolongbench/oolong-real, config `dnd`, split `test`, MIT, the
synthetic Oolong subset excluded), does the agg_pipeline P-large path (planner + L4/SQL over typed facts) beat B1 (local
Qwen3.8-27B REPL agent, same chunks, BM25 hybrid retrieval, <= 30 calls) on aggregation counting questions, and is M4 (median
calls / prefill per question) inside the Phase 0 bounds (<= 3 calls, <= 2510 prefill)? No cloud model anywhere.

## Frozen inputs
* Dataset: /mnt/raid0/llm/datasets/oolong-real-3ae6ab58 (HF revision 3ae6ab58ffcc4253ed52daeb58e02ed4cfea5784); sha256 in
  `SHA256SUMS` there (recorded by precond.sh; test.jsonl is 9,251,930,544 bytes).
* Corpus: ONE context window of `dnd/test.jsonl`, `singledoc_*` types only (one D&D episode transcript, 122k-207k chars, fits the
  196k-ctx np1 recipe). Multidoc windows (299k-1.19M chars, up to about 300k tokens) are OUT OF SCOPE: they exceed the 196k recipe.
  Window = the median-length singledoc window (ties: smaller context_window_id). Selection is `driver/select_sample.py`, no
  `answer` field read before selection.
* Sample n = 10: from that window, 5 `singledoc_rolls` + 5 `singledoc_spells` questions, integer-gold questions only (the dataset also has list-valued golds, e.g. 'last three spells'; they are excluded by a gold-FORMAT filter, no model output involved), ids sorted, indices 0,2,4,6,8 per type.
  Sample file sha256 is sealed in SOURCE.sha256 by the publisher.
* Gold = the dataset `answer` (integer count). Scorer: exact integer match after parse (Phase 0 `score_agg`); no partial credit,
  no tolerance. Unparseable = wrong, counted separately.
* Typed schema: answer slot unchanged from Phase 0 (typed integer). Retrieval-only comparator recorded: BM25 top-k chunks fed to
  the 27B with a one-shot count prompt, no planner, no REPL (1 call per question; informational, not gated).
* Arms: P-large (agg_pipeline.v2 planner agg110.v1 over a store built by L4 extraction of the episode into typed roll/spell
  events), B1 (Phase 0 B1 unchanged). Both on Qwen3.8-27B-Q8_0 + DFlash2 from the v10 GPU kernel store, temperature 0, seed 0.
  Ingest of the one window uses the CPU quarter row first (Qwen3.6-35B-A3B L4), then the GPU row answers.

## Measures
* M2 (decision measure): paired P-large vs B1 accuracy on n = 10 with counts (both right, only P, only B1, neither).
* M4: median calls and prefill per P-large answer; unchanged-vs-Phase-0 means <= 3 calls and <= 2510 prefill.
* M1: `null` with reason "Oolong-real ships no gold-evidence cells (answer only)"; never 0.0.
* Latency and cleanup: server pid proof, KFD residency, stop verified dead (job.sh pattern of agg113).

## Expected effect vs sample-size error
At n = 10 the paired SE of the accuracy difference is about 0.15-0.19 (discordant-pair based). Only a gap of >= 0.5 is resolvable; a
smaller one is reported as UNRESOLVED, never as a tie win. Phase 0/1 priors: P-large 0.31-0.55 vs B1 0.13-0.20 on financial
aggregation; Oolong counting over a 50k-token transcript is harder for extraction (events are dialogue, not tables), so a P-large
collapse is a live outcome.

## Stops and gate class
* Checkpoints after question 5 and 10 (counts only, no per-question output read before the final verdict script).
* STOP-ON-ANSWER (decision grade): after all 10, if P-large and B1 differ by >= 3 SE (paired z = (b-c)/sqrt(b+c) >= 3, needs
  b+c >= 9) OR are tied within 1 question, the box is answered; no extension.
* If separated by < 3 SE and > 1 question: UNRESOLVED; extension to a second window is a NEW predeclared row, not this one.
* STOP-ON-FUTILITY: ingest produces zero parsed events for the window (store empty), or the GPU window ends / a drain; wall cap
  1800 s for the pilot (handoff bound) with the 14400 s provisional window NOT consumed.
* B1 hits the 30-call cap on a question: record as wrong with reason `cap`, not redrawn. Unknowns never redrawn.
* Gate class: futility-only at interim looks (no early GO).

## Done-when mapping
(a) split on disk with sha256 (done 2026-11-16) and sample result filed with M4 vs Phase 0, M1 null+reason, P-large vs B1 reported.
(b) not needed: the split was obtainable.
