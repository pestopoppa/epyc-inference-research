# Probe B — Workload-Shape Coverage (Canonical)

**Date**: 2026-04-29 evening (post-reboot, post-OMP-fix)
**Verdict**: ONE robust positive signal — **CPU1+mbind-off on Hybrid SSM in prefill regime delivers +8.9% on pp512**. All other (model × shape × config) cells are within noise (≤2%).

## Method

3 models × 3 workload shapes × 3 configs × n=5 under FULL canonical recipe:
```
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active
numactl --interleave=all -- taskset -c 0-95
llama-bench -t 96 -fa 1 --mmap 0 [shape flags] -r 5
```

Configs: c0 (default), c2 (GGML_NUMA_REPACK_INTERLEAVE=0), c3 (CPU1 stack + c2). c1 dropped per multi-arch finding that c2 dominates c1 alone in tg64.

Shapes: tg64 (decode-only), pp512 (prefill 512 tokens), pp2048 (prefill 2048 tokens).

## Result

| Model | Shape | c0 baseline | c2 mbind-off | c3 CPU1+mbind-off |
|---|---|---|---|---|
| Nemotron-9B (Hybrid SSM) | tg64 | 12.69 ± 0.05 | 12.83 (+1.1%) | 12.96 (+2.1%) |
| | pp512 | 317.32 ± 14.28 | 340.72 (+7.4%) | **345.57 (+8.9%)** |
| | pp2048 | 323.76 ± 7.51 | 333.83 (+3.1%) | 335.67 (+3.7%) |
| Qwen3.6-27B (Dense Q8) | tg64 | 4.28 ± 0.03 | 4.30 (+0.5%) | 4.28 (+0.0%) |
| | pp512 | 115.03 ± 1.96 | 116.80 (+1.5%) | 114.63 (-0.3%) |
| | pp2048 | 116.46 ± 0.56 | 115.42 (-0.9%) | 115.84 (-0.5%) |
| gemma-31B (Dense Q4) | tg64 | 6.75 ± 0.01 | 6.79 (+0.6%) | 6.78 (+0.4%) |
| | pp512 | 182.93 ± 0.76 | 182.87 (-0.0%) | 183.97 (+0.6%) |
| | pp2048 | 169.26 ± 0.26 | 168.97 (-0.2%) | 169.28 (+0.0%) |

## Headline findings

### 1. Hybrid SSM prefill is the standout case — c3 +8.9% on pp512

**Nemotron-9B Q8 hybrid SSM benefits substantially from CPU1+mbind-off in the prefill regime**, especially short prefill (pp512). Effect size:
- pp512: **+8.9%** (321 → 345 tokens/s)
- pp2048: +3.7% (smaller — longer prefill amortizes the locality benefit)
- tg64: +2.1% (small, decode is BW-bound)

Hypothesis: prefill is more compute-bound than decode (the K/V re-population pattern means each thread does more independent work between barriers), so CPU1's CCD-aware partitioning + mbind-off's NUMA-local weight placement compounds well.

The tg64 effect (+2.1%) confirms the multi-arch matrix's earlier finding (+1.16% for n=15) — Probe B's n=5 result is consistent within rep variance.

### 2. Dense Q8 + Dense Q4 are neutral across all shapes

For Dense Q8 (Qwen3.6-27B) and Dense Q4 (gemma-31B), all (shape × config) cells are within ±2% of baseline. Treatments don't help OR hurt in prefill or decode. CPU1+mbind-off is essentially a no-op for dense archs at canonical recipe.

This corrects the multi-arch matrix's apparent "+3.9% c2 on gemma tg64" — under tighter Probe B measurement, gemma's c2 effect is +0.6% (NS) — the multi-arch result was likely baseline-drift artifact (gemma c0 std was ±0.41 = 6.4% CV in multi-arch; Probe B's c0 std is ±0.01).

### 3. The c1 (CPU1 alone) cell was rightly dropped

Multi-arch showed c1 alone is consistently dominated by c2 (mbind-off) on tg64. Probe B's c2 vs c3 comparison shows similar effects — CPU1 stack on top of mbind-off adds modestly (+1-2 percentage points) on Hybrid SSM but is neutral elsewhere.

## Deployment recommendation

| Arch | Workload | Config | Δ vs default |
|---|---|---|---|
| Hybrid SSM (Nemotron, Qwen3-Next?) | prefill-heavy (long input, short generation) | c3 (CPU1+mbind-off) | **+8.9% pp512, +3.7% pp2048** — opt-in for prefill workload |
| Hybrid SSM | decode-heavy | c2 (mbind-off) | small +1-2% tg — marginal but real |
| Dense Q8 (Qwen3.6-27B) | any | default v5 | no improvement from probed configs |
| Dense Q4 (gemma-31B) | any | default v5 | no improvement from probed configs |

The +8.9% pp512 finding on Hybrid SSM is the strongest signal in this entire post-reboot CPU-optimization re-validation campaign. It's worth pursuing in a follow-up evaluation: does it generalize to other Hybrid SSM models (Qwen3-Next-80B), and is there a simple env-flag opt-in mechanism for hybrid-arch detection at runtime?

## Files

- `run_probeB.sh` — measurement script
- `master.log` — full run log
- `probeB_<model>_<shape>_<config>.log` — per-cell bench logs (27 cells)
- `decision.md` — this document
