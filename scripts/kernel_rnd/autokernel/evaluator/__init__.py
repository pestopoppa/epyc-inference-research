"""autokernel.evaluator — the AK3 trusted tiered evaluator (T0 / T1 / T2).

WHY THIS PACKAGE EXISTS
-----------------------
It replaces `scripts/kernel_rnd/kernel_eval.sh`, which is fenced off and exits 2.
That script emitted `"status":"OK"` unconditionally, set `COH="coherent"` for any
non-empty generation, and ran its baseline comparison only when `--baseline-env`
happened to be passed — so every anchor-less run entered `kernel_store.py`'s
CORRECT-ONLY Pareto view as if it had been verified. The replacement is not a
tidier script; it is a different shape:

  * a verdict is **computed** from gate results and cannot be stamped
    (`api.compute_verdict` is the only constructor of `api.Verdict`, and
    `Verdict.__post_init__` re-derives the status from its own evidence);
  * an **explicit anchor** — source commit, binary SHA-256, linkage SHA-256 — is
    a precondition of every performance, coherence, correctness, capacity or
    determinism comparison, and its absence is `INVALID`, never "coherent";
  * **correctness precedence is lexicographic**: `Verdict.rank_key()` raises
    rather than returning a penalised rank; and
  * every run either satisfies the **search-grade conjunction** or is `INVALID`
    with the failing conjuncts named, and every **void condition** is a checked
    precondition whose reason is journaled.

Governing instrument: `epyc-root/measurement/protocols/kernel-research.md`
(Annex K, **P-AK-SEARCH-1**, RATIFIED 2026-08-03). It emits a **search record,
not a claim**. Owning design: `handoffs/active/autokernel-research-loop.md`,
phase AK3 (§5.4, §6.4, §8.5.1, §8.6, §9, §15.2).

SCOPE
-----
T0, T1 (T1a/T1b/T1c) and T2 only. **T3 and T4 are not owned here** — they are
release instruments, explicitly outside P-AK-SEARCH-1's scope and owned by AK5.
`api.admit_tier()` refuses them by name; `api.ReleaseTierEvaluator` is the seam
AK5 implements.

This package runs no inference, no benchmark and no build; it starts, stops and
signals no process; and it writes no file. `api.audit_no_write_or_process_paths()`
proves the last of those from the module's own AST rather than asserting it in
prose (design §5.4: the trusted runner "has no authority to modify candidate
source or production state").
"""
