# Phase C — MAB tree-shape selector re-test under canonical

**Date**: 2026-04-29
**Verdict**: **CLOSURE STANDS — gate NOT flipped**, but framing materially revised. Original "definitive negative -3.97% p=0.012" → "Coder neutral (NS), REAP significantly worse (p<0.001)" under canonical recipe.

## Phase C result (n=90 paired Coder)

| Arm | n=90 mean t/s |
|---|---|
| linear | 45.54 |
| tree | 46.78 |
| Δ tree-linear | **+2.72% (p=0.237 NS)** |

Original Phase 0'' (broken OMP env): tree -3.97% (p=0.012). **Direction flipped** under canonical recipe, but n=90 not enough to confirm new direction.

## Phase C.1 extension result (cumulative)

Phase C n=90 was inconclusive on direction; extended to n=180 paired on Coder + n=90 paired on REAP to settle the question.

| Model | n paired | linear t/s | tree t/s | Δ | t | p |
|---|---|---|---|---|---|---|
| Coder | 180 | 45.73 | 45.12 | **-1.34%** | -0.911 | 0.36 (NS) |
| REAP  |  90 |  8.27 |  7.60 | **-8.20%** | -3.591 | **0.0003** |

The Phase C +2.72% on Coder was sampling variance — doubling n showed actual direction is slight negative, not significant. REAP shows tree is clearly and significantly worse.

## Why the original framing was wrong

Original Phase 0'' run script (`run_probe.sh` in `2026-04-29-mab-phase-0-prime-prime-replication/`) was missing the OMP env stack (`OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`). It DID have `numactl --interleave=all`, so the regression wasn't as bad as e.g. CPU4 phase 1, but barrier wait-policy was still passive.

Tree-shape paths involve more candidate branches → more aux-context decode passes → more OMP barrier traffic. Under broken-OMP regime, this asymmetric barrier traffic showed up as -3.97% (statistically clean signal because sustained vs the linear baseline). Under canonical OMP, the barrier overhead drops materially in both arms, but the SIGN-of-difference depends on workload-shape interaction.

## Closure framing (REVISED)

> MAB tree-shape selector under canonical recipe (OMP_PROC_BIND=spread + OMP_PLACES=cores + OMP_WAIT_POLICY=active + numactl --interleave=all + --mmap 0) does NOT deliver a positive throughput signal on the test workload (Coder-30B Q4_K_M and REAP-246B-A35B Q4_K_M with Qwen3-Coder-DRAFT-0.75B drafter, temp=0.7, top_k=40, top_p=0.95, random per-request seed, 64-token decode):
>
> - Coder-30B: tree -1.34% vs linear (n=180 paired, p=0.36, NOT significant)
> - REAP-246B: tree -8.20% vs linear (n=90 paired, p=0.0003, HIGHLY significant)
>
> The original Phase 0'' framing of "tree hurts at -3.97% (p=0.012) — DEFINITIVE NEGATIVE on Coder" was overstated due to broken-OMP-env baseline contamination. Under canonical OMP recipe, Coder is neutral (within noise); REAP is reliably negative. NEITHER model justifies Phase 1 implementation.
>
> Does NOT generalize beyond Coder/REAP at these specific (drafter, temp, top_k, top_p, seed-mode, p_split) settings. Other (workload, drafter, temperature, p_split, K) configurations remain unevaluated.

## Files

- `run_phaseC.sh` — initial n=90 measurement
- `run_phaseC1_extension.sh` — extension to n=180 Coder + n=90 REAP
- `phaseC_master.log`, `phaseC1_master.log` — master logs
- `srv_*.log` — per-config server logs
- `comp_coder_linear_p*_r*.json` (n=180), `comp_coder_tree_p*_r*.json` (n=180) — Coder responses
- `comp_reap_linear_p*_r*.json` (n=90), `comp_reap_tree_p*_r*.json` (n=90) — REAP responses
- `decision.md` — this document
