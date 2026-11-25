# DTREE-HEADROOM-REPLAY — tree headroom on 27B DFlash2, offline replay — PREDECLARATION

Sealed 2026-11-25 before any replay output on this capture exists (this file and `headroom_replay.py` are committed before
`replay` runs; the commit order is the seal). Box: `handoffs/active/speculative-decoding.md` DTREE-HEADROOM-REPLAY.
Row: `ak-dtree-headroom-replay-cpu-20261125` (CPU, python replay only, no model load, no server). Thresholds below are copied
from the box's `done when`; nothing is tuned after the replay.

## Question
How much tokens-per-pass headroom is left in a tree for 27B DFlash2 at sampled temperatures: (a) the speed-of-light ceiling at
verifier budgets N in {8, 12, 16}; (b) what a Pando-style best-first tree (k children per level, pruned to N nodes by cumulative
draft probability, intake-2963) recovers of it, against the served chain V0 and the current builder.

## Input (frozen)
Capture `/mnt/raid0/llm/tmp/ak-dice-capture-27b-dflash2-gpu-20261123-r2/out/capture/cap-L01.jsonl` sha256
82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845; unit index `.../out/results.jsonl` sha256
38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1 (script refuses on mismatch). 192 units = 64 prompts x
T {0.1, 0.3, 1.0}. Stochastic targets only: T0.3 and T1.0 gate, T0.1 reported.

## Arms (tokens per pass = 1 + expected accepted draft tokens, every round counted, served depth L as in dice_replay_27b)
- **V0** served chain: the realized chain, accept min(1, p/q) per depth (`dice_replay_27b.v0_rej`, exact). Must reproduce the
  capture's realized acceptance rate within 2% at both gating T, else **FUTILITY_BASELINE** and nothing is filed.
- **ceiling** oracle drafter: best N-node tree chosen with the target's own post-sampler rows, nodes in order of cumulative
  target probability (optimal, because reach never increases along a path). A node can be expanded only where the capture holds
  the target row of its token path (anchor, realized chain, captured 32-column tree), so the ceiling is a ceiling over captured
  information and not over all trees. This is stated, not corrected.
- **pando** children per node = top-k by the drafter's q (DFlash2 lattice conditioned on the parent's lattice index), nodes in
  order of cumulative draft probability, N nodes, same expandability rule. k in {1, 2, 3, 4, 6, 8}; per (T, N) the k with the
  best mean tokens per pass on calibration prompts 0-31 is frozen (ties: smaller k) and reported on holdout 32-63.
- **cur** the current builder, `dice_replay.blind_shape` (spine with sibling leaves) at W = N.
- Two verifies per tree. **V1t (gating)**: lossless verify of a deterministic tree, the target's sample walks the tree and a
  child is accepted iff the sample equals it, so reach(child) = reach(parent) * p_parent(child), exact. **V1q (secondary,
  never gating)**: ordered recursive rejection with the q pool (`rrs_q` convention), exact analytic; with deterministic
  children it is not lossless and can exceed the ceiling, so it is reported with the cells where it does.
- Known approximations (stated): tree shape is evaluated per round on the realized round's captured rows; deterministic top-k
  children differ from the served sampled chain, so Pando k=1 over V0 is reported (`greedy_chain_V1t/V0`) to separate the
  draft-selection effect from the tree-shape effect.

## Statistics
Ratios are ratio-of-means over holdout prompts 32-63; 95% CI = prompt-clustered bootstrap over holdout prompts (B = 2000,
seed 20261125). Gating metric for the Pando clause is `pando_V1t/V0` at the frozen k.

## Verdict (box thresholds, gate class futility-only; holdout)
- **CLOSE_TREE_SHAPE**: ceiling/V0 point < 1.10 at N=16 at both gating T, OR Pando V1t/V0 CI-upper < 1.02 at every gating
  (T, N) cell.
- **FILE_GPU_ROW**: Pando V1t/V0 point >= 1.03 with CI-lower > 1.0 at some gating (T, N). Checked before CLOSE.
- **INCONCLUSIVE**: otherwise.
Stop on answer: CI half-width <= 1% for the Pando ratios (`ci_halfwidth_le_1pct`); the replay is one deterministic analytic
pass over the frozen capture, so there are no interim looks. Stop on futility: V0 check fails.

## Validation before the capture is replayed
`selftest` (SELFTEST_HEADROOM.json, replay refuses unless PASS): ordered recursive rejection hand numbers; target-walk hand
numbers; two-level tree; single-child chain under V1q equals `v0_rej`; ceiling order and sum; Pando expands only under a
captured row; on the first capture unit (no replay output written) the ceiling dominates every V1t tree, is monotone in N.
