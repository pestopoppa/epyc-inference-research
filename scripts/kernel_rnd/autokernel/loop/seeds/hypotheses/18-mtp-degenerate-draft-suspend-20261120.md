# Seed from the Pipeline session's taskflow r4 sweep (MTP acceptance at np=1), 2026-11-20

Source: coordinator relay of the Pipeline r4 sweep (Qwen3.6-35B-A3B-MTP Q8_0, C14 GPU, 64k per slot, id_slot pinning).
Pooled MTP acceptance was 0.45 at np=1 vs 0.82/0.80 at np=2/4. The cause was NOT a server bug: 2 of 208 requests degenerated
into repetitive output that ran to the 4096-token cap with ~0 accepted drafts and carried ~40 % of all drafted tokens; the
per-request mean acceptance was 0.78, the same as np=2/4. The investigation box PFXC-MTP-NP1 was cancelled. Grade:
`design_prior`. Gate class: futility-only. Optional: the harness proxy can abort degenerate requests itself.

## AK-H-MTP-DEGEN-1 (proposed) -- suspend drafting / flag degenerate output on sustained low acceptance
When a slot's rolling draft acceptance stays below 0.2 for >= 512 generated tokens, the server either (a) suspends drafting for
that slot (decode continues target-only, with no verify rows wasted on rejected drafts), re-probing every N tokens, or (b) marks
the response as degenerate (a field in timings / finish metadata) so the client can abort or retry.
Expected: on degenerate requests, verify cost falls to plain decode (MTP D=2 verify of 3 rows vs 1 row; on the A3B ~10-20 % of
per-token time); pooled throughput on such traces rises by the drafted-token share of the degenerate tail (~40 % of drafted
tokens in r4). Effect on healthy requests must be 0 (threshold never fires at acceptance ~0.78).
**Falsifier:** replaying the r4 degenerate pair on the A3B with the knob on gives < 3 % wall-time reduction for those requests, OR
any healthy request (acceptance >= 0.5) trips the threshold, OR greedy output changes (drafting must not change tokens). Effort S
(server-context.cpp speculative loop: rolling counter per slot plus a default-off knob).
