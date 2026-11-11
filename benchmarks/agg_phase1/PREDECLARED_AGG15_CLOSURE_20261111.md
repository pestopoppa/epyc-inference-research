# AGG-1.5 - closure basis, 2026-11-11 (POST-DATA, operator decision)

Written 2026-11-11, **after AGG-1.5 data exists from three runs**. This file states what AGG-1.5 is closed on, and what it is not. It
changes no frozen rule: `PREDECLARED_AGG15.md` (sha256 `74c219d6cd8200c20b011fd381afe4d4df74f626b5244f80d7350fcb2ad885fc`), amendment 1
(`5b121cbf40240824787bf4cc2ec4ba4796814d50078ab00105a805b94c4697c0`) and amendment 2
(`d017c000adf20bc932f4db308cd2e4616bac0f8b7ab86bd60bfc5936bb90bcfa`) stay byte-for-byte as they were. A file cannot carry its own hash:
this file's sha256 is recorded in the commit message of the commit that adds it and in the AGG-1.5 handoff note.

## Decision

Operator, 2026-11-11, AGG-1.5: **"Persist index, re-check, no GPU."** Close AGG-1.5 on the r3 GPU run plus a zero-model persistence
re-check; do not spend a fourth GPU run on a component that is a software property of the registry.

## What the closure rests on (and nothing else)

1. **r3 GPU parity on accuracy, calls and prefill.** Run `agg15-parity-gpu-20261111-r3`, all 20 pairs, v10 GPU kernel store, amended P arm
   (passage-level L1 index, repeated-SEARCH guard). Accuracy P 18/20 = B 18/20 (difference 0, tolerance 1); median calls 3 / 3
   (max extra calls 0, tolerance 1); median prefill 2,143.5 / 2,143.5 (max extra prefill 462, tolerance 2,000); no L2/L3/L4 built.
   Evidence: `/mnt/raid0/llm/tmp/agg15-parity-gpu-20261111-r3/out/{result.parity.json,results.jsonl}`, `verdict.json`.
   **Caveat, stated plainly:** this parity holds *by construction*. Both arms now run the same engine (`ContextSearchIndex`, the
   `context.search` engine) over the same text; the zero-model re-check below measures that the two arms' top-8 pointers (chunk span,
   line) are identical on all 20 label queries. r3 therefore confirms that the surrounding loop (planner scope, SEARCH/READ/FINAL, guards,
   read span) costs nothing extra on a GPU run; it is not independent evidence that a different retrieval backend would match
   `context.search`.
2. **The persistence re-check** (`benchmarks/agg_phase1/parity_persist_check.py`, output `parity_persist_check_20261111.json`, no model,
   no server, no GPU). The orchestrator persists the `l1_passage_bm25` index next to the store, keyed by (corpus, content hash,
   builder version), atomic write with an integrity header, loaded by a fresh process instead of rebuilt (epyc-orchestrator
   `4299454597bb78d7ad6b3ce6c642c8f46f2c727e`, tests in `tests/unit/test_aggregation_index_persist.py`). The check replays the 20 label
   queries through P in two separate processes over a copy of the r3 parity store and asserts: process 2 has 0 builds; the hits of the
   two processes are identical (chunk, span, line, bit-identical score, in order); at most 1 build per document across both processes.
   Result: **passed, 9 of 9 assertions** (see the JSON): process 1 built 4 (one per document) and process 2 built 0 and loaded 4; the
   per-document build total across both processes is 1 for each of the 4 documents; hits identical for all 20 queries; the manifest of
   the copy records `origin='loaded'`, `n_built=1`, `n_loaded=1` per document; the r3 source store's sha256 is unchanged
   (`a0167cea67523b554f15a425ebd8ea1125d71330dfcbb43533bbaf8bdab17501`).

## How the failed component is satisfied

r3's only failing component was `no_rebuild_per_request`: P index builds 5 (JPMORGAN 2, the others 1) against B 20. The second JPMORGAN
build was a process boundary (the smoke process and the resumed main run), not a per-request rebuild: within each process every document
was built once. The registry cached in memory only. Under the operator's decision the component is **satisfied by the zero-model test,
not by a GPU run**: the property "a process restart does not rebuild an unchanged index" is a property of the registry and the store, it
does not depend on the model or the GPU, and the two-process test exercises exactly the failure r3 hit.

What this does **not** establish: no GPU run has measured the post-persistence P arm. `parity_m5.py`'s `PArm.open` counts an index as a
build when it is not in the registry's in-memory table (`key not in registry._live`), so it would still count a *loaded* index as one build
per process; that runner is pinned by r3 and was not amended, and the closure does not need it. A future GPU run that wants the persisted
counters should read `registry.built` / `registry.loaded` (or the manifest `origin`), as the re-check does.

## What stays on record

- **r1** `agg15-parity-gpu-20261111`: filed `fail`, `phase1_gate_tripped`, component `prefill`.
- **r2** `agg15-parity-gpu-20261111-r2`: filed `fail`, `phase1_gate_tripped`, component `prefill`.
- **r3** `agg15-parity-gpu-20261111-r3`: filed `fail`, `phase1_gate_tripped`, component `no_rebuild_per_request`.

None of the three verdicts is edited, re-filed or counted as a pass. The closure is an operator decision layered on three filed fails, with
the basis above; it is not a fourth run and not a mechanical `PASS`.

## Known weakness carried forward (not closed here): lexical recall

On the 20 label queries the gold quote is in the top 8 hits for **16 of 20** questions, for B (`context.search`) and for P alike (P's
ranking equals B's on 20 of 20 label queries; both arms miss `p15_09`, `p15_10`, `p15_11`, `p15_16`). That is a property of the shared
lexical engine (800-character line-aware passages, BM25, dense off), not of persistence and not of the parity claim, which is about P
matching B. It bounds what either arm can answer from retrieval alone and is tracked as its own task, AGG-1.7 in
`handoffs/active/knowledge-pipelines.md`.

## Claim grammar

M5, scoped: *the L1 passage-index P arm matches `context.search` on a 20-question, 4-document mid-size set on accuracy, calls and prefill
(r3, one GPU run, identical engine), and a persisted index is loaded rather than rebuilt across a process boundary (zero-model, 2
processes, 4 documents).* It does not claim: a pass of the r3 mechanical gate, any GPU measurement of the persisted arm, retrieval
recall above 16 of 20, or anything for corpora outside the 4 documents (200k to 1M tokens).
