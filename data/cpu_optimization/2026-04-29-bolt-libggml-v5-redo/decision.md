# BOLT-libggml v5 redo — NOT DEPLOYABLE on v5 PGO

## Headline

| Metric | PGO baseline | BOLT | Δ |
|---|---|---|---|
| Coder pp32 (earlier alternated) | 213 / 152 (high variance) | 390 / 172 | inconsistent (+82.8%, +12.9%) |
| Coder pp64 trial 1 | 283.48 ± 8.93 | 245.67 ± 20.85 | **-13.3%** |
| Coder pp64 trial 2 | 263.54 ± 17.51 | 238.16 ± 22.47 | **-9.6%** |
| Coder pp64 trial 3 | 225.09 ± 15.27 | 285.33 ± 5.78 | **+26.8%** |
| Coder pp64 across-trial PGO range | 225 to 283 (+26%) | — | **System noise floor exceeds BOLT signal** |
| REAP pp32 PGO | 53.19 ± 1.13 | — | reference |
| REAP pp32 BOLT | — | 53.29 ± 1.15 | **+0.2% (parity, expected)** |

## Verdict: NOT DEPLOYABLE

BOLT-libggml on v5 PGO produces noise-band results on Coder. Within-trial-pair comparisons (Coder pp64) show BOLT inconsistent (-13%, -10%, +27% across 3 sequentially-alternated trials). The system noise floor (PGO baseline ranging 225-283 across same-config trials) is larger than any consistent BOLT signal.

REAP shows expected parity (+0.2%) — BOLT-libggml profile was Coder-only via the per-model fdata (merge-fdata-20 produces legacy-format output rejected by llvm-bolt; same bug as morning's session).

## Why morning's BOLT-libggml +2.1% on Coder doesn't transfer to v5 PGO

1. **Morning's measurement was on pre-v5 build** (different toolchain, before clang+libomp+znver5+PGO compounded gains). PGO already optimizes hot-path block layout; BOLT's marginal hot-block-reorder improvement on top of PGO is in the noise band.
2. **Function coverage limit**: BOLT-INFO consistently reports 4-5% function coverage even with 60s × 4 model perf records. Hot-path is structurally narrow (mul_mat / GEMM kernels); 95% of binary is cold. Adding more samples does not increase coverage past this structural limit.
3. **Quantitative budget**: morning's +2.1% was likely closer to BOLT's true ceiling on this workload. v5 PGO's +18-20% codegen win covers most of the same ground; BOLT after PGO has marginal additional headroom.

## Closure scope (per closure-inflation policy)

> "BOLT-libggml on v5 PGO build (clang+libomp+znver5+PGO mixed-B) does NOT produce a deployable Coder-30B pp32/pp64 forward-pass gain. Within-trial alternated measurements show no consistent directional signal (-13% to +27% across 3 trials with same config baseline ranging 225-283 t/s). REAP-246B is at parity (+0.2%, expected per morning's workload-sensitivity finding). Does NOT generalize to 'BOLT is dead on CPU': morning's pre-v5 BOLT delivered +2.1% on a different toolchain; the structural finding here is that BOLT after PGO adds marginal layout improvement that is dominated by within-system noise on this hardware."

## Reopen criteria (NOT current Phase 1, documented for future)

- Newer BOLT version that supports merge-fdata legacy-to-modern conversion
- Different perf-record workload that exercises more functions (e.g., long-context prefill that triggers attention-side codepaths)
- Different llvm-bolt options (e.g., `-frame-opt`, `-jump-tables=move`)
- Quieter measurement environment (no megasync, controlled thermal state)

## Phase 3 #1 status: documented + closed via test

PGO-only is the v5 production binary recommendation. BOLT-Coder per-role variant was a candidate gain path; under v5 PGO it is not measurable above noise.
