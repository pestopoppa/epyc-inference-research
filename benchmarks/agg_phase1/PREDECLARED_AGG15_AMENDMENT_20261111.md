# AGG-1.5 - amendment 2026-11-11 (POST-DATA): read-span cap and repeated-read guard, then rerun

Written 2026-11-11, **after AGG-1.5 data exists**. This is a POST-DATA amendment to `PREDECLARED_AGG15.md` (sha256
`74c219d6cd8200c20b011fd381afe4d4df74f626b5244f80d7350fcb2ad885fc`, which stays byte-for-byte unchanged and stays the frozen
rule set for questions, golds, scorer and gates). A file cannot carry its own hash: this file's sha256 is recorded in the
commit message of the commit that adds it, in the r2 run spec and in the AGG-1.5 handoff note.

## The trip (r1, kept on record)

Row `agg15-parity-gpu-20261111` (research pin `00d8c98a3c20a7239a005258c0fb92ae804fe400`) ran the pilot (questions 1-5, both
arms, 89 s wall, 10 answers) and filed `fail` / `phase1_gate_tripped`, failing component `prefill` (pilot futility:
max (P - B) prefill per question > 2,000). Evidence: `/mnt/raid0/llm/tmp/agg15-parity-gpu-20261111/out/`
(`results.jsonl`, `result.parity.json`) and `verdict.json`.

- Accuracy 4/4 on each side (P missed PEPSICO p15_02, which is a call-cap exhaustion with no answer; B missed AES p15_04).
- Median calls 3 / 3; median prefill P 3,194 vs B 2,278 (+916, inside the 2,000 bound).
- The trip is ONE question. On PEPSICO p15_02 the P arm used the whole 12-call cap (`budget_exhausted`): 27,086 prefill
  tokens against B's 2,307, so +9 calls and +24,779 prefill. Its trace is 11 `SEARCH` actions and 1 `READ`
  (`c23f592eb1ea5048:00083`, a stored chunk read whole): after the first 5 searches it read that chunk, then issued 6 more
  searches (`Consolidated Balance Sheets` four times, `Total Current Assets 2022` once, ...) and never converged. There was
  no repeated READ in r1; the loop was a repeated-SEARCH loop that began after one whole-chunk read. The other four P questions
  prefilled between -1,284 and +1,188 tokens against B (p15_01 -1,284; p15_03 +437; p15_04 +947; p15_05 +1,188) and each used
  exactly 1 SEARCH + 1 READ.
- Cause class: a read-granularity difference between the arms. `PREDECLARED_AGG15.md` states P reads a stored chunk whole (up to
  12,000 characters, average 3.1k) while B reads an 800-character span (the `ContextSearchIndex` chunk, `CHUNK_MAX_CHARS = 800`),
  and that a `prefill` trip would be "a finding about the planner's read granularity". r1 is that finding.

## Decision and standing of the fixes

The operator decided on 2026-11-11 (AGG-1.5): **fix, amend, rerun**. The fixes below are therefore post-data and made with the
operator's decision; they were chosen after reading the r1 trace, not predeclared. Anything r2 shows is read with that in view:
a pass supports the claim only for the amended arms, and r1's trip is not erased.

## What changes (`benchmarks/agg_phase1/parity_m5.py`, research repo; nothing else)

1. **Read span cap, both arms: 800 characters** (`READ_SPAN_CHARS`, replacing the 12,000-character `READ_CHARS`). B already reads
   the hit span (at most 800 characters); a B read of a wider pointer is now clamped to 800 as well. P reads at most 800
   characters of the chunk its pointer names: the window around the hit for the SEARCH query that returned that pointer
   (`hit_window`: candidate windows centred on query words, ranked by whole-query phrase present, then distinct query words, then
   occurrences, ties to the earliest; the start of the chunk when no query word is present). Deterministic, no model. A test checks
   that, on the 20 frozen questions, the window holds the gold figure whenever the top-ranked chunk holds it.
2. **Repeated-read guard, both arms, one shared code path** (`retrieval_loop`): a `READ` of a pointer (P: chunk id; B: character
   span after the 800 cap) the model has already read in this question returns the short note
   `(already read: <ptr>; its text is earlier in this conversation. Read a different pointer or answer.)` instead of the text. The
   call still counts as a call. B had no such guard, so it is applied to both and the comparison stays the retrieval backend only.
   Unknown pointers are not marked read. Each answer record gains `dup_reads`; the meta record gains `read_span_chars`.

   Honest scope note: in r1 no pointer was read twice, so this guard would not have changed the p15_02 trace; it is the
   operator-decided hygiene fix, applied symmetrically. A repeated-SEARCH guard is NOT part of this amendment.

## What does not change

Questions, golds, scorer, gates (accuracy <= 1 question; calls <= +1; prefill <= +2,000 tokens; no L2/L3/L4; P index builds <= 1 per
document), tolerances, the 12-call cap, search k = 8, the 200-character hit snippets, the prompt and grammar, the pilot rules
(<= 900 s; futility accuracy > 2 questions or prefill > 2,000), the model, server recipe and kernel, the Phase 0 store and the
orchestrator pin `9a0ab00d88ffdccab3dd9f25d7f5db26c93b7069`. No budget cap changes.

## Both runs stay on record

r1 (`agg15-parity-gpu-20261111`, `phase1_gate_tripped` at the pilot, `prefill`, reading chunks whole) and r2
(`agg15-parity-gpu-20261111-r2`, the amended arms) are both reported, side by side, in any write-up. r2 writes to its own
`out/` directory; r1's results are not reused, not edited and not counted as r2 data. r2 supersedes r1 as the row, not as evidence.

## Rerun

The pilot, then the full 20-question phase, rerun under the UNCHANGED gates above and the unchanged pilot futility rules: a
pilot trip in r2 files `phase1_gate_tripped` again; no early GO at the pilot; the full phase runs only when the pilot does not
trip; the verdict is `parity_m5.score` as before. Row: `agg15-parity-gpu-20261111-r2`, research pin = the commit that adds
this file and the amended runner (in the r2 spec), rank 10, ESTIMATE 1,800 s (r1's pilot took 89 s for 5 questions; not a
measurement of the full phase).
