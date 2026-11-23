# SPEC-DTREE-DICE step 2b — offline replay, A3B MTP half — PREDECLARATION

Sealed 2026-11-23 before any replay output on the capture exists. Box: `handoffs/active/speculative-decoding.md`
SPEC-DTREE-DICE (authoritative). Citations: intake-2831 (DICE), intake-2846#02, intake-2715#00, intake-2717#01,
intake-2845#record.

## Input (frozen)
- Capture `/mnt/raid0/llm/tmp/ak-dice-capture-a3b-mtp-cpu-20261123-r2/out/capture/cap-L01.jsonl`,
  sha256 00d810a82ba86172ef5860a3ed4055a3817cf66004b18ca88808beddddd74bb7 (re-verified; script refuses on mismatch).
- Unit index `.../out/results.jsonl` sha256 ff058e74f82e5b4f4c75c4562d5dba95299b7c052af1e6b1ad56f6ff80fbd6ea
  (byte offsets off0/off1 map each unit = prompt_idx x T to its rounds).
- Schema dice.v1 @ fork 347014966: per round, MTP chain (D = 4), MTP top-10 (id, logit, p) per drafted depth, and per
  verified row (anchor + chain, <= 5 rows) the target top-64 logits and the post-sampler candidate list (id, p) of the
  served sampler (top_k 40, top_p 0.95, min_p 0.05, T in {0.1, 0.3, 1.0}).

## Capture limits (stated up front, bound every conclusion)
1. The A3B MTP capture is a CHAIN capture (tree = false). Target rows exist only along the chain prefix, and MTP
   proposals exist only conditioned on the chain prefix. The replayable tree family is therefore the
   **chain spine with leaf siblings**: at depth j the chain token (the only node that may have children) plus up to 9
   sibling leaves from the MTP top-10 at depth j. Siblings' subtrees are not replayable. Every tree arm is a LOWER
   bound on what the same arm would reach with full subtrees; V0 is exact.
2. Max depth 4 (D = 4), max 4 x 10 = 40 candidate nodes; "max 32 verified nodes" capture cap is not binding for MTP.
3. Counterfactual rounds are replayed on the served (V0) trajectory's prefixes (standard offline-replay approximation).
4. MTP-only; the 27B DFlash2 half is a separate GPU capture row and is NOT covered by this verdict.

## Arms (node budget W = number of draft nodes, anchor excluded; W in {1, 4, 8, 12, 16})
- **V0 served path**: sample-and-compare on the MTP argmax chain truncated to min(W, D). Analytic expectation
  sum_j prod_{i<=j} p_i(chain_i). Reported also as V0s = served chain at full D (W-independent).
- Draw-blind tree shape (used by V1, V2): probability-greedy best-first over the spine-with-leaves family under the MTP
  q (q = MTP top-10 p renormalised), node value = prod_{i<j} q_i(chain_i) * q_j(c), parent must be selected; W nodes.
- **V1 recursive rejection without replacement**: same shape (siblings per depth m_j); at depth j the chain token is a
  point-mass draft verified first (accept w.p. r(c), residual r <- r without c); the m_j siblings are DRAWN without
  replacement from q restricted to non-chain ids (drafter-own keyed Gumbel-top-k, independent of target noise) and
  verified by recursive rejection (accept s w.p. min(1, r(s)/q_cur(s)); else r <- norm(max(r - q_cur, 0)),
  q_cur <- q_cur without s). Continue only through the chain token.
- **V2 Gumbel-max walk, draw-blind selection**: target token at row j = argmax_c log p_j(c) + g_j(c) with pre-sampled
  keyed Gumbel g; tree = the draw-blind shape with the top-q sibling ids; walk accepts while the target token is a
  tree node. Analytic expectation sum_j prod_{i<j} p_i(chain_i) * p_j(S_j); MC must agree.
- **V3 DICE draw-informed selection**: same target Gumbel-max rule (same g), but the W nodes are chosen best-first by
  cumulative score sum over path of [alpha log q(c) + sigma g(c) - max_siblings(alpha log q + sigma g)] using the SAME
  g the target uses. Order is invariant to positive scaling, so alpha = 1 and rho = sigma/alpha is fitted on a grid
  {0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5} per (T, W) on CALIBRATION prompts 0-31 only, maximising mean tokens per
  pass; the fitted rho is then frozen for holdout.
- V4 exponential-race verifier (k >= 3 siblings): optional, NOT run in this replay (declared now).

## Metric
Expected tokens per target pass = accepted draft tokens + 1 (bonus), averaged over every verify round (every round
counted, none dropped; rounds with fewer than 5 rows use depth <= rows - 1). Expectation over target randomness via
analytic formulas (V0, V2) and R = 64 keyed Monte-Carlo draws per round (V1, V2-check, V3); keys = sha256(task, round,
draw, role). Per-round per-arm results in `out/results.jsonl`.

## Analysis
- Holdout = prompts 32-63. Ratios are ratio-of-means over holdout rounds; 95% CI = prompt-clustered paired bootstrap
  (B = 2000, resample prompts, seed 20261123).
- Break-even for V3/V0 (verify-cost proxy, from DTREE GPU measurements in the box, no A3B CPU tree cost exists yet):
  W <= 4: 1.000; W = 8, 12: 1.079 (T12 host-gap round ratio 41.36/38.32); W = 16: 1.141 (T16 reached only e2e parity
  0.9966 at tau ratio 0.58/0.51 = 1.137, so break-even = 1.137/0.9966).
- **Futility (CLOSE_FUTILITY)**: holdout V3/V1 < 1.02 at every T and every W <= 16, AND V3/V0 below break-even at
  every (T, W).
- **FILE_GPU_ROW**: V3/V0 holdout point estimate >= break-even(W) with CI-lower > 1.0 at some (T, W) AND V3/V1 >= 1.02
  there.
- **FILE_RRS_ROW**: otherwise, if V1/V0 >= break-even(W) with CI-lower > 1.0 at some (T, W).
- Else CLOSE_FUTILITY (tree arms do not clear verify cost; reported with which clause held).
- Stop-on-answer: prompt-clustered 95% CI half-width <= 1% on V3/V1 at the highest-share T. Step-1 temperature share
  is not filed yet, so the half-width is reported at every T and the claim uses T = 0.3 provisionally (role default
  band 0.1-0.3); if not met, the verdict is still issued (gate-class futility-only) with the half-width stated.

## Validation before the capture is touched
Synthetic distributions with known answers: (a) V0/V1/V2/V3 emitted depth-0 token distribution equals target p,
chi-square p > 0.001 on >= 200k draws per arm (q != p and q = p cases); (b) V2 MC acceptance within 3 SE of analytic
p(S); (c) q = p, rho = 1: V3 W = 1 single-depth acceptance == 1. Replay refuses to run on the capture unless the
synthetic suite PASSes.
