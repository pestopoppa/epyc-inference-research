# Dynamic Expert Selection Phase 0 entropy probe (analytical proxy)

**Date**: 2026-04-29
**Source**: `moe-dynamic-expert-selection.md` Phase 0 step 0.1 (entropy histogram)
**Method**: Use existing MoE-Spec PPL drift data (`decision_v5_FINAL.md`, full 32-chunk PPL measurements 2026-04-28) as analytical proxy for routing-distribution shape.

## Hypothesis (from handoff)

If routing distribution is bimodal (some experts critical, most negligible), Entropy-gated K could be a real knob. If unimodal (roughly uniform), entropy-gated K won't help.

## Method

The MoE-Spec budget B masks the bottom (n_expert - B) experts to -INFINITY. PPL drift as a function of B reveals routing concentration:

- **High-entropy routing** (uniform): PPL drift should scale ~`-log(B/n_expert)` per layer × n_layers — every dropped expert costs equally.
- **Low-entropy routing** (concentrated): PPL drift should be **near-zero until B drops below the "critical" expert count**, then catastrophic.

## Data (from existing measurements)

### Coder-30B Q4_K_M (n_expert=128, n_expert_used=8)

Source: `data/cpu_optimization/2026-04-28-moe-spec-phase-1/coder30b_ppl32_B*.log`

| B | B as % of n_expert | PPL chunk-3 | drift vs B=0 |
|---|---|---|---|
| 0 (off) | 100% | 9.86 | reference |
| 128 (gate-skip) | 100% | 9.86 | bit-exact ✓ |
| 96 | 75% | 9.75 | -1.1% (noise; *better*) |
| 64 | 50% | 10.52 | **+6.7%** |
| 32 | 25% | not measured but documented severe | catastrophic |

### REAP-246B Q4_K_M (n_expert=80, n_expert_used=8)

Source: `data/cpu_optimization/2026-04-28-moe-spec-phase-1/reap246b_ppl32_B*.log`

| B | B as % of n_expert | PPL chunk-3 (normalized) | drift |
|---|---|---|---|
| 0 (off) | 100% | 9.30 | reference |
| 80 (gate-skip) | 100% | 9.30 | bit-exact ✓ |
| 60 | 75% | 9.36 | +0.6% (essentially preserved) |
| 40 | 50% | 11.44 | **+23%** |
| 20 | 25% | 15.79 | **+70% (catastrophic)** |

## Pattern interpretation

Both Coder and REAP show **bimodal routing distribution**:

1. **Top ~75% of experts (by aggregated routing prob) carry 99%+ of contribution**: at B=96/Coder and B=60/REAP, PPL is essentially preserved (within noise).
2. **The next ~25% drop is catastrophic**: at B=64/Coder and B=40/REAP, PPL drift is meaningful (+6-23%).
3. **Below 50% budget**: dramatic quality collapse on REAP (+70% drift at B=20).

This is **NOT a uniform distribution** (which would show smooth PPL increase from B=128→0). It's clearly bimodal: experts split into "important" and "long-tail" classes.

## Phase 0 entropy probe verdict: GO with caveats

The routing distribution is **structurally bimodal** on tested Coder/REAP. An Entropy-gated K mechanism that detects high-entropy (uniform) routing distributions and uses lower K would be ineffective on these models because **routing is rarely high-entropy on production decode** — the top-K experts almost always dominate.

**Refined verdict**: classical "Entropy-gated K" (use lower K when entropy is high) is **NOT a productive direction** for these models. Routing is consistently concentrated; there's no uniform-distribution regime to exploit by reducing K.

**Alternative direction** (more promising): **Adaptive top-B per-batch** — i.e., learn B per-batch from the routing distribution shape. The MoE-Spec mechanism already does this via fixed B; making B adaptive to the actual distribution shape per batch (e.g., B = num experts above a threshold prob) would be the orthogonal axis.

## Phase 1 deferred / re-scoped

Phase 1 of `moe-dynamic-expert-selection.md` proposed implementing one of:
1. Dynamic Skipping (per-token threshold) — TESTABLE, but per-token mask requires custom variable-K argsort; ~100 LOC, MEDIUM risk
2. OD-MoE single-layer lookahead (84-91% accuracy) — saves routing compute (cheap layer); marginal benefit
3. MoE Pathfinder — DEPRIORITIZED (infrastructure too heavy)
4. Entropy-gated K — **THIS PROBE NEGATIVE** for the regime tested

Re-scope: **deprioritize Phase 1 implementation indefinitely**. The structural finding (routing is bimodal, not high-entropy) limits the upside of entropy-based mechanisms on Coder/REAP/Q8.

## Closure scope

> "Routing distribution on tested production MoE models (Coder-30B, REAP-246B; inferred from MoE-Spec PPL drift data) is structurally bimodal — top ~75% of experts carry 99%+ of contribution; bottom ~25% are noise-level. Entropy-gated K mechanisms that exploit high-entropy regimes will not deliver on greedy-temp inference for these models. Does NOT generalize to: (a) higher-temperature sampling regimes (entropy may be higher there); (b) different model classes (e.g., DeepSeek V3 with 256 experts and aux-loss-free routing — routing dynamics may differ); (c) Dynamic Skipping or OD-MoE lookahead mechanisms which probe a different distribution property."
