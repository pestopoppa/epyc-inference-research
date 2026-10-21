# DS41-W6.1 -- DSpark drafter alpha A/B on the 1017 target (2026-10-21) -- PARTIAL, n<5

A = 1017 drafter, B = Sept drafter. Port build llama.cpp-experimental-ds41-20261005/build-cpu.
Seeds: pair p uses 1000*p+234+prompt (pair 1 = the W6 seeds).

Deviations from the brief:
- A and B ran concurrently (24 threads each, cpus 0-23 / 24-47, cpu sets swapped in pair 2) instead of sequentially at -t 48. Alpha is thread-count invariant: pair 1 reproduces W6 exactly (878/1508, 959/2013).
- Only 2 pairs were attempted. Coordinator then ordered one server at a time (memory), so no further pairs.

Complete data: block 2 = 2 pairs, block 3 = 1 pair (pair 2 block 3 was cut at prompt 13 by a timeout and is excluded).
Greedy: pair-1 greedy was cut after 5 (A) / 8 (B) prompts; text_sha of the 5 overlapping prompts is identical. Not a full confirmation.

Result (analysis.json, pair x prompt bootstrap): block 2 A 0.607 vs B 0.626, diff -1.9 pp, CI95 [-4.2, -0.0]; block 3 A 0.476 vs B 0.486, diff -0.9 pp, CI95 [-2.7, +1.0].
Verdict: underpowered. B (Sept) marginally better at block 2, indistinguishable at block 3.
