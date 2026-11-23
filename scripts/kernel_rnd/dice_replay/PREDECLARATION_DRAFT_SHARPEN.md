# SPEC-DRAFT-SHARPEN — truncation-matched and sharpened DFlash2 draft distributions, offline replay — PREDECLARATION

Sealed 2026-11-23 before any replay output on this capture exists (this file and `draft_sharpen_replay.py` are committed
before `replay` runs; the commit order is the seal). Box: `handoffs/active/speculative-decoding.md` SPEC-DRAFT-SHARPEN.
Row: `ak-draft-sharpen-replay-cpu-20261123` (CPU, python replay only, no model inference).

## Question
Does either lossless drafting lever raise 27B DFlash2 tokens per target pass at served temperatures?
- (a) **trunc**: the served draft distribution q is untruncated (common/speculative.cpp:1806-1830 @ C20 `af7ad26dd`)
  while the target p is post-sampler (top_k 20, top_p 0.95, min_p 0.05). q' applies the target's sampler to the
  drafter's candidates, renormalizes, then samples. The rejection accept (common/sampling.cpp:1035-1125) stays exact for
  any q', so the emitted distribution is unchanged.
- (b) **scale**: q = softmax(scale * score / T) over the K = 16 lattice candidates, one scale per T.

## Input (frozen)
Capture `/mnt/raid0/llm/tmp/ak-dice-capture-27b-dflash2-gpu-20261123-r2/out/capture/cap-L01.jsonl` sha256
82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845; unit index `.../out/results.jsonl` sha256
38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1 (script refuses on mismatch). 192 units = 64 prompts x
T {0.1, 0.3, 1.0}, all rounds `verify: rejection`. Capture-content check (done before sealing, schema only): each round
holds the DFlash2 lattice (D 7 x K 16 token ids `lat_tok` and predecessor-conditioned RAW selector scores `lat_sc`,
from which the served q is softmax(score / T), exactly as `dice_replay_27b.qdist`), the chain and its candidate indices
`ck`, and the target's post-sampler candidates (id, p) on every verified node row including every chain depth. So
scores are available directly (no T * log q recovery needed) and p is the final target distribution.

## Estimator
Per round, at the served depth L = min(n_draft, D) (n-max 7; capped rounds use n_draft):
a(scale, trunc) = sum_{j<L} prod_{i<=j} alpha_i, alpha_i = sum_c min(p_i(c), q'_i(c)) (= 1 - TV, the single-draft
rejection acceptance), along the REALIZED chain prefix: p_i is the captured target row at chain depth i and q'_i is
conditioned on the realized predecessor ck[i-1]. Tokens per pass V = 1 + a, every round counted.
Truncation rule: llama.cpp default chain order top_k -> top_p -> min_p -> temperature, so the kept set is computed on
softmax(scale * score) (pre-temperature; top_p keeps the minimal prefix whose cumulative mass reaches 0.95, crossing
token included; min_p keeps p >= 0.05 * p_max), then q' = softmax(scale * score / T) restricted to it, renormalized.
Duplicated lattice tokens are merged (as served qdist). V0 = (scale 1.0, trunc off) = served q.
Known approximation (stated, not corrected): conditioning on the realized prefix ignores that a different q' changes
which chain is drafted; this is the standard per-position 1 - TV projection.

## Grid, fit, holdout
scale in {0.5, 0.6, ..., 1.5} x trunc in {off, on} x T in {0.3, 1.0} gating (T0.1 reported, not gating). Per T the
(scale, trunc) maximizing calibration-prompt (0-31) mean V is V* (ties: scale nearest 1.0, then trunc off), frozen and
reported on holdout prompts 32-63. Ratios are ratio-of-means V*/V0; 95% CI = prompt-clustered bootstrap over holdout
prompts (B = 2000, seed 20261123). Attribution arms on holdout: trunc only (1.0, on), scale only (calibration-best scale,
trunc off), trunc at its calibration-best scale.

## V0 validation (gates the verdict)
V0 analytic acceptance rate (sum a / sum L) vs the capture's REALIZED acceptance rate per gating T must agree within 2%
(|ratio - 1| <= 0.02), else the verdict is **FUTILITY_BASELINE** and nothing is filed.

## Verdict (gate class futility-only; holdout)
- **KEEP**: V*/V0 >= 1.03 with CI-lower > 1.0 at T0.3 or T1.0 -> file a C21 fork row; report the carrier per T: `trunc`
  (trunc-only CI-lower > 1.0 and point >= 1.01), `scale` (same test on scale-only), both, or `both_jointly`.
- **CLOSE**: V*/V0 < 1.01 and CI-upper < 1.02 at both T.
- **INCONCLUSIVE**: otherwise.
Stop on answer: CI half-width <= 1% at both gating T (reported as `ci_halfwidth_le_1pct`); the replay is a single
deterministic analytic pass over the frozen capture, so there are no interim looks and no extension.

## Validation before the capture is touched
`selftest` (SELFTEST_SHARPEN.json, replay refuses unless PASS): truncation hand cases (top_k, top_p, min_p, off);
scale 1 / trunc off equals the served qdist; scale s at T equals qdist at T/s; alpha = 1 - TV, and a 200k-draw MC of the
rejection rule with a truncated q' accepts at alpha within 4 SE and emits p (chi-square p > 0.001) — the lever is lossless.
