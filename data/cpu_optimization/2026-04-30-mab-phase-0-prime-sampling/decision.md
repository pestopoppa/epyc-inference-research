# MAB Phase 0' Sampling Regime — DECISION (INCONCLUSIVE, signal worth follow-up)

## Verdict

**INCONCLUSIVE — POTENTIAL SIGNAL**.

- **Fixed-seed temp=0.7 NO-GO**: 18/18 reps produced byte-identical
  output between linear and tree (deterministic verifier). t/s within
  0.1-0.6% across both Coder and REAP. Same conclusion as Phase 0
  greedy regime. Probe-design caveat: fixed seed makes the comparison
  uninformative.
- **Random-seed temp=0.7 POTENTIAL SIGNAL**: tree +9.6% over linear
  on Coder (n=9, p≈0.23 — not significant). Per-prompt: p0 +18%, p1
  +8%, p2 +1%. Per-rep variance is high (tree wins p1_r2 by +52%, loses
  p1_r0 by -25%) — pattern consistent with tree helping when drafter
  is weak (low accept rate) and hurting when drafter is strong.

Phase 0's closure-inflation language explicitly opened the door for
sampling-regime testing. This probe finds a real but noisy positive
signal — not enough for a clean GO, not enough for a clean NO-GO.

## Recommended next action — DO NOT LAUNCH MAB PHASE 1 YET

A higher-rep replication probe (~2-4 hours) is the cheapest cut to
decide between GO and NO-GO. n=9 is too few for the variance observed.

Specifically:
1. **Coder random-seed at n≥30**: replicate the +9.6% signal. If t-stat
   clears p<0.05, the signal is real.
2. **REAP random-seed at n≥30**: extend coverage to the second target.
3. **Drafter-quality predictor sketch**: identify a per-decode-round
   feature that distinguishes "drafter weak" from "drafter strong"
   rounds. Without this, a context-free MAB selector cannot exploit
   the per-rep variance pattern.

Phase 1 implementation (~245 LOC) is justified ONLY if (1) and (3) both
pass.

## What this changes vs Phase 0 NO-GO closure

The Phase 0 closure (2026-04-29) wrote:

> "DySpec heap-spec tree at `--draft-p-split=0.05` with greedy decoding
> (temperature=0) produces BIT-IDENTICAL outputs to `--draft-p-split=0`
> linear baseline ... The MAB selector over the paper's arm pool ...
> cannot recover headroom that is structurally absent at temp=0 ...
> Does NOT generalize to: Higher-temperature sampling, different arm
> pool, sampling-decoding configurations."

This Phase 0' empirically confirms the door was correctly left open
for the sampling regime, AND finds a real (if noisy) signal there.
The Phase 0 closure stands; this is an extension of scope, not a
retraction.

## Cross-references

- Parent: `2026-04-29-mab-tree-selector-phase-0/decision.md` (Phase 0 NO-GO at greedy temp)
- Handoff: `handoffs/active/mab-tree-shape-selector.md`
