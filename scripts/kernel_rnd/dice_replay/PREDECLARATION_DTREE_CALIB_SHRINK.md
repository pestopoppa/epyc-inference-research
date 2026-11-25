# DTREE-CALIB-SHRINK-REPLAY — Platt-calibrated tree stop and load-adaptive shrink on 27B DFlash2, offline replay — PREDECLARATION

Sealed 2026-11-25 before any replay output for this question exists (this file and `calib_shrink_replay.py` are committed before
`fit` / `replay` run; the commit order is the seal). Origin: TreeSpark (intake-2975; arXiv 2609.22098) as applied to the
C21 V2 RRS tree KEEP (`ak-c21-tree-official-gpu-20261124`). Box: `handoffs/active/speculative-decoding.md`
DTREE-HEADROOM-REPLAY (follow-up note). Row: `ak-dtree-calib-shrink-replay-cpu-20261125` (CPU quarter filler, python only, no model, no server).

## Question
(1) Does an edge-acceptance Platt calibration, p_hat = sigmoid(a*logit(q) + b), used as the stop threshold for tree growth,
beat the same growth rule on the drafter's raw q? (2) Does a load-adaptive shrink to the chain (price ladder over load) beat
the best load-blind width rule when verify columns cost more?

## Input (frozen)
Capture `/mnt/raid0/llm/tmp/ak-dice-capture-27b-dflash2-gpu-20261123-r2/out/capture/cap-L01.jsonl` sha256
82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845, unit index `.../out/results.jsonl` sha256
38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1 (script refuses on mismatch). The C21 tree row logs only
per-request aggregates (no per-edge dump), so this capture is the only per-edge data (drafter lattice q, target rows on a
captured 32-column tree). Gating T: 0.3 and 1.0; T0.1 reported.

## Fit and judge
Platt (a, b) is fitted per T on calibration prompts 0-31 by damped Newton with a 1e-3 L2 pull to (1, 0). Edge set: the top-4
drafter candidates under every captured node whose ancestors all equalled the target argmax; label 1 iff the candidate equals
the target argmax at that node (greedy match). The paper's (0.67, -0.46) is reported, not used. Everything is judged on
holdout prompts 32-63; the best load-blind arm per (T, m) is frozen on calibration prompts.

## Arms (tokens per pass = 1 + exact expected accepted tokens; V1t lossless target-sample walk; target rows only where captured)
chain (served V0, cost 1); fixed10 / fixed16 (current builder at W = 10 / 16); auto (PROXY for the C21 AUTO width rule, whose
rule text was not found: best-first on raw q with the value stop rule); platt (same on p_hat); shrink (platt grown under the
load-priced stop rule, then the chain wins when the tree's predicted tokens per cost does not beat the chain's).
Value stop rule, no tuned threshold: add the next node iff its predicted reach r > R * (cost(n+1) - cost(n)), R = current
predicted (1 + sum reach) / cost(n); W <= 16, k = 4 children per node, same expandability rule as headroom_replay.py.

## Metric and price ladder
Modelled accepted tokens per verify-cost unit = sum tokens / sum cost over holdout rounds. cost_m(W) = 1 + m*(f(W) - 1) with f
the measured 27B round-cost ratio (4 -> 1.0, 8 -> 1.1013, 12 -> 1.2557, 16 -> 1.3396; `dice_replay_27b.BREAK_EVEN`),
m in {1, 2, 3}. m = 1 is the measured np1 table. **m = 2, 3 are sealed hypotheses for np2 / np4 verify, not measurements**:
the shrink verdict is therefore conditional on those prices and says so. Arms other than shrink choose their tree at m = 1
(load-blind); every arm is scored at the scenario's m.

## Statistics
Ratio of sums over holdout prompts; 95% CI = prompt-clustered bootstrap over holdout prompts (B = 2000, seed 20261126).

## Verdict (gate class futility-only; holdout, gating T)
- **FUTILITY_BASELINE**: V0 analytic acceptance not within 2% of realized at T0.3 or T1.0. **FUTILITY_FIT**: a Platt fit that is non-finite or has a <= 0 at a gating T.
- **calib win**: platt / auto >= 1.01 with CI-lower > 1.0 at m = 1 at some gating T.
- **shrink win**: shrink / best_blind >= 1.01 with CI-lower > 1.0 at some gating T with m in {2, 3}.
- Verdict FILE_CALIB_AND_SHRINK, FILE_CALIB, FILE_SHRINK (by which wins hold), CLOSE if every platt/auto (m=1) and shrink/best_blind (m>=2) CI-upper < 1.01 at all gating cells, else INCONCLUSIVE.
Stop on answer: all 192 units replayed and analysed (one deterministic pass, no interim looks). Stop on futility: baseline or fit failure.

## Validation before the capture is replayed
`selftest` (replay refuses unless PASS): cost table and ladder; p_hat hand value; Platt recovers a planted (0.67, -0.46) from 20000
synthetic edges; growth is nested and eligible-only; the stop rule shrinks as price rises; on the first real capture unit
the arms stay within 0..16 nodes, tokens >= 1 and cost >= 1, edges exist and a unit-0 fit is finite.
