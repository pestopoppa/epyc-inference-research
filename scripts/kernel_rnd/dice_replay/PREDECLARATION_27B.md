# SPEC-DTREE-DICE step 2b — offline replay, 27B DFlash2 half — PREDECLARATION

Sealed 2026-11-23 before any replay output on this capture exists (this file and `dice_replay_27b.py` are committed
before `replay` runs; the commit order is the seal). Box: `handoffs/active/speculative-decoding.md` SPEC-DTREE-DICE and
SPEC-DTREE-ACCEPT-GAP (authoritative). Structure follows `PREDECLARATION.md` (A3B half); differences are stated.

## Input (frozen)
- Capture `/mnt/raid0/llm/tmp/ak-dice-capture-27b-dflash2-gpu-20261123-r2/out/capture/cap-L01.jsonl`,
  sha256 82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845; unit index `.../out/results.jsonl`
  sha256 38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1 (script refuses on mismatch).
  192 units = 64 prompts x T {0.1, 0.3, 1.0}, 7499 verify rounds, all `verify: rejection` (store 2a7602113 with
  SPEC-DISTS-TRUNC; sample_compare 0), 138 capped rounds (n_draft < 7).
- Schema dice.v1 @ 2a7602113: DFlash2 lattice D = 7 x K = 16 (ids + predecessor-conditioned selector scores), chain and
  its candidate indices ck, a forced 32-column superset tree (chain U probability-greedy U DICE order, ranks pr/dr) with
  the target's post-sampler candidates (id, p) at every verified node row. Sampler top_k 20, top_p 0.95, min_p 0.05.

## Lessons applied (SPEC-DTREE-ACCEPT-GAP, root 78c415e19)
1. V0 must reproduce the SERVED 27B DFlash2 T > 0 accept, which is rejection sampling with the drafter's distributions
   (common/sampling.cpp dists overload), not sample-and-compare, at the served n-max 7 (registry argv
   `--spec-draft-n-max 7`; capped rounds use their n_draft).
2. The target verifier is V1 (tree recursive rejection), because DFlash2 serves rejection; V2/V3 reported.

## Arms (W = draft nodes, anchor excluded; W in {4, 8, 12, 16})
- **V0 served**: q_j = softmax(score(j, ck[j-1]) / T) over the 16 lattice candidates; accept chain_j w.p.
  min(1, p_j(c)/q_j(c)); analytic sum_j prod_{i<=j}. W-independent (the served chain always drafts 7). A keyed MC of the
  same rule (V0_mc) must agree.
- **V1** (target): spine-with-leaves shape = the frozen `blind_shape` under q; per depth the chain token is the FIRST
  draw from q and m_j further siblings are drawn without replacement from q minus the chain (keyed Gumbel-top-k); recursive
  rejection (accept s w.p. min(1, r(s)/q_cur(s)); else r <- norm(max(r - q_cur, 0)), q_cur <- q_cur without s); continue
  only through the chain. With m_j = 0 it equals V0. Siblings are leaves, so V1 is a LOWER bound on full-subtree RRS.
- **V2**: frozen Gumbel-max walk on the same shape (analytic + MC check). **V3**: frozen DICE selection, rho fitted on
  calibration prompts 0-31 per (T, W), frozen for holdout.
- **V2t** (informative): exact target-sample walk on the CAPTURED probability-greedy tree prefix (pr ranks 1..W) with full
  subtrees; rounds lacking ranks 1..W are excluded from V2t and its own V0 denominator (V0t).

## Metric and analysis
Tokens per target pass = accepted draft tokens + 1, every round counted. R = 64 keyed MC draws (V1, V2mc, V3, V0_mc).
Holdout prompts 32-63; ratios are ratio-of-means; 95% CI = prompt-clustered bootstrap (B = 2000, seed 20261123).

## V0 validation (reported, gates trust in the verdict)
(a) Internal: V0 analytic acceptance rate (sum E[acc] / sum drafted) vs the capture's REALIZED acceptance, per T,
    same sampler and prompts: must agree within 2%. (b) External: vs measured served 27B acceptance, A arms of
    ak-dtree-sampled-gpu-20261123 (0.47 / 0.43 at T0.3 / T1.0) and ak-specdist-trunc-ab-gpu-20261123 C19 L256
    (0.465 / 0.437); different prompts and sampler (server default top_k 40), so reported with that caveat.
    If (a) fails, the verdict is INVALID_V0 and nothing is published.

## Break-even (verify cost) and verdict
Break-even = measured 27B tree round-cost ratio vs chain (implied_round_cost_ratio, ak-dtree-batch-27b-gpu-20261120-r4:
T8 1.0067, T10 1.1960, T12 1.2346, T16 1.3186; T = columns incl. anchor) at W + 1 columns, linear interpolation,
extrapolated past T16 with the T12->T16 slope: W4 1.0 (fewer rows than the chain), W8 1.1013, W12 1.2557, W16 1.3396.
The A3B replay's host-gap table (1.0/1.079/1.079/1.141) is reported as an optimistic alternative only.
- **PUBLISH_V2_ROW** (tree win): holdout V1/V0 point >= break-even(W) and CI-lower > 1.0 at some served T in
  {0.3, 1.0} and some W. Gates publishing ak-dtree-sampled-v2-gpu-20261123.
- **PARITY**: else, if break-even(W) lies inside the V1/V0 CI at some served (T, W).
- **PROJECTED_LOSS**: otherwise. T0.1 reported, not gating. Gate class futility-only.

## Validation before the capture is touched
Synthetic suite (SELFTEST_27B.json): (a) V1 emitted token distribution equals p (chi-square p > 0.001, 200k draws) with
the chain drawn from q, m = 0 and m = 3, q != p and q = p; (b) V1 with m = 0 acceptance equals V0 = 1 - TV(p, q) within
3 SE; (c) V2t analytic on a hand tree equals the hand value. Replay refuses unless PASS.
