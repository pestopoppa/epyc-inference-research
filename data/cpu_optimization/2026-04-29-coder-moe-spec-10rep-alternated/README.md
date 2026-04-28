# Coder-30B MoE-Spec 10-rep alternated — Phase 3 #2

**Date**: 2026-04-29
**Hypothesis**: 10-rep alternated B=0/B=64 should settle whether earlier inconsistent Coder MoE-Spec results (-43% / parity / +84% across builds and cache states) are signal or noise.
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`

## Result: NOT DEPLOYABLE (confirms earlier verdict)

5 trial-pairs × alternated B=0/B=64 × 5-rep each:

| Trial | B=0 (mean ± std) | B=64 (mean ± std) | Δ |
|---|---|---|---|
| 1 | 182.51 ± 7.18 | 179.69 ± 26.09 | -1.5% |
| 2 | 169.54 ± 6.54 | 199.87 ± 9.60 | +17.9% |
| 3 | 215.72 ± 6.85 | 164.78 ± 2.33 | -23.6% |
| 4 | 153.98 ± 1.52 | 183.12 ± 19.61 | +18.9% |
| 5 | 179.06 ± 15.47 | 91.61 ± 18.08 | **-48.8% (outlier)** |
| **mean** | **180.16** | **163.81** | **-9% mean, ±27% trial-to-trial std** |

Pattern: trial-to-trial std on the SAME config (B=0 ranges 154-216 → 40% spread) exceeds B=0 vs B=64 difference. Inconclusive signal. Trial 5 B=64=91 is a major outlier (~half of all other measurements at same config).

## Confirms decision_v5_FINAL.md verdict

> "Coder-30B B=64 NOT deployable — Result varied wildly across builds + cache states + system noise."

The 10-rep alternated method added another 5 trials of evidence: still no consistent signal direction. Coder MoE-Spec mask overhead/savings ratio is too marginal on this smaller model under realistic system conditions.

## Why Coder is fundamentally noise-prone (vs REAP)

Per `decision_v5_FINAL.md` analysis:
- **Coder-30B** (17 GB model): mask-construction adds ~6 ggml ops per MoE layer × 48 layers = ~288 extra ops. Per-token compute is light (high t/s baseline ~180-220). Mask overhead is ~5-10% of total compute. MoE-Spec savings (DRAM expert-weight bandwidth reduction) competes with this overhead.
- **REAP-246B** (138 GB model): same fixed mask overhead, but per-token compute is ~5x heavier. Mask/compute ratio ~1-2%. MoE-Spec savings dominate cleanly. (REAP B=40 = +13.5% pp32 robust across all builds.)

The structural conclusion is **deployable on REAP, not on Coder**, regardless of measurement methodology.

## Closure

Coder MoE-Spec is closed via test (this run + earlier evidence). REAP MoE-Spec at B=40 remains the only deployable spec-dec lever from the MoE-Spec mechanism.

Bundle: 10 raw bench logs + summary + this decision file.
