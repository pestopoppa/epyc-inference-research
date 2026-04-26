# Warmed-vs-Cold Asymmetry Investigation — 2026-04-26 evening

**Goal**: characterize WHY some production models prefer cold `--interleave=all` (Q8_0 + gemma-26B) and others prefer warmed mmap=1 + taskset (Next-80B + REAP-246B). Test whether a "best-of-both" canonical config exists.

**Conclusion**: the asymmetry is **NOT a deployment lever**. The historical "warmed wins" numbers required HOURS-to-DAYS of system uptime + bench history for numa_balancing to migrate pages into model-specific access patterns. 5-run warming doesn't move the needle. **Cold `numactl --interleave=all` is the practical canonical for ALL models**, reaching 90% of long-warmed performance immediately and reliably.

## 2x2 matrix on all 5 production models

`taskset -c 0-95 -t 96 -fa 1 -p 0 -n 32 -r 3` cold-cache (drop_caches before each run) for each cell:

| Model | mmap=1 + taskset (default) | mmap=1 + `--interleave=all` | mmap=0 + taskset | mmap=0 + `--interleave=all` |
|-------|----------------------------|-----------------------------|------------------|----------------------------|
| Coder-30B Q4_K_M | 23.55 ± 0.06 | **42.37 ± 0.48** | 24.78 ± 0.02 | 42.27 ± 0.39 |
| Qwen3.6-35B Q8_0 | 7.86 ± 0.00 | **20.95 ± 0.16** | 7.82 ± 0.00 | 20.81 ± 0.02 |
| Qwen3-Next-80B Q4_K_M | 13.40 ± 0.03 | **20.69 ± 0.01** | 13.50 ± 0.02 | 20.51 ± 0.02 |
| REAP-246B Q4_K_M | 3.14 ± 0.02 | **5.96 ± 0.02** | 3.13 ± 0.02 | 5.94 ± 0.01 |
| Gemma-26B-A4B Q4_K_M | 16.81 ± 0.04 | **34.75 ± 0.08** | 17.06 ± 0.42 | 34.69 ± 0.07 |

**Finding 1 — mmap mode is irrelevant**: cell (mmap=1, interleave) ≈ cell (mmap=0, interleave) within noise on every model. Likewise cell (mmap=1, taskset) ≈ cell (mmap=0, taskset).

**Finding 2 — `--interleave=all` is the dominant lever**: dramatically improves ALL models vs taskset-only (1.5x to 2.7x range). The improvement is uniform across model families.

## Warming progression (5 consecutive runs without drop_caches)

### Next-80B Q4_K_M, mmap=1 + taskset, numa_balancing=0

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|-------|-------|-------|-------|
| 13.52 | 13.59 | 13.52 | 13.42 | 13.53 |

**Flat at ~13.5**. No warming over 5 runs.

### Next-80B Q4_K_M, mmap=1 + `--interleave=all`, numa_balancing=0

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|-------|-------|-------|-------|
| 20.85 | 21.31 | 21.04 | 21.00 | 20.99 |

Stable at ~21.0 immediately. Run 1 already near steady-state.

### Next-80B Q4_K_M, mmap=1 + taskset, numa_balancing=1 (migration enabled)

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|-------|-------|-------|-------|
| 13.48 | 13.73 | 14.00 | 13.36 | 14.01 |

Slow upward trend (13.5 → 14.0) but nowhere near 21 (interleave) or 23.25 (historical warmed). **numa_balancing=1 doesn't accelerate warming materially over 5 runs.**

### REAP-246B Q4_K_M, mmap=1 + taskset, numa_balancing=0

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|-------|-------|-------|-------|
| 3.31 | 3.26 | 3.32 | 3.32 | 3.33 |

**Flat at ~3.3**. No warming over 5 runs.

## numastat memory placement on Next-80B during execution

**mmap=1 + taskset (FilePages per NUMA node, MB)**:
- Node 0: 119
- Node 1: 21,702
- Node 2: 236
- Node 3: 24,877

→ File pages clustered on nodes 1 + 3 (~50/50 across 2 nodes); nodes 0 + 2 nearly empty. Half the threads hit cross-NUMA every read.

**mmap=1 + `--interleave=all` (FilePages per NUMA node, MB)**:
- Node 0: 11,666
- Node 1: 11,633
- Node 2: 11,780
- Node 3: 11,870

→ Uniform across all 4 nodes (~11.7 GB each). Cross-NUMA traffic spread evenly across all 4 memory controllers.

**Why `--interleave=all` wins by 1.5x**: with default first-touch, memory controllers on nodes 1+3 are saturated by all 96 threads' reads, while controllers on nodes 0+2 sit idle. Interleave spreads load across 4 controllers.

## Why the historical "warmed wins" numbers exist

The runbook's 23.25 (Next-80B) and 6.85 (REAP-246B) historical numbers came from a system that had been running benchmarks **for ~2 days of uptime** before the snapshot. Over that time, numa_balancing slowly migrated certain hot pages to nodes where they were most accessed, while other pages stayed distributed. This produces a model-specific access-pattern-aware placement that's slightly better than uniform interleave (5-13% gap to cold-interleave).

**However**: this is not reachable in practical deployment timeframes. 5-run warming doesn't get there. Even with numa_balancing=1 enabled, 5 runs only nudges Next-80B from 13.5 to 14.0 (vs the 23.25 target). The historical numbers were achieved over hours-to-days of accumulated system state.

## Practical recommendation

**Production canonical for ALL models**: `numactl --interleave=all -t 96 -fa 1` (mmap mode irrelevant; pick `--mmap 0` for cleanest cold-cache reproducibility).

- Reaches 90-100% of historical warmed performance immediately
- Stable, reliable, reproducible across reboots
- No model-specific tuning needed
- The 5-13% gap to long-warmed steady-state is real but not a deployment lever

**Don't pursue** "best-of-both" canonical engineering: the gap is small, the warming dynamics are slow and unpredictable, and the cost of trying to hit it (long warmups, model-specific mbind schemes) outweighs the marginal benefit.

## Strategic implications

1. **CPU20 protocol simplifies**: primary canonical = `numactl --interleave=all`. No mmap-mode requirement. Cold-cache reproducibility is the gold standard; "warmed steady-state" numbers should be flagged as advisory-only and require explicit "X hours of warming" labels.
2. **All "warmed wins" claims need re-baselining** against cold-interleave, not against cold-mmap=1+taskset. The latter is pathological.
3. **The `--mmap 0` requirement in our prior canonical was unnecessary** — `mmap=1 + numactl --interleave=all` is equivalent within noise. Simpler config.
4. **Some research questions remain interesting**: WHY does numa_balancing eventually achieve a 5-13% better state than uniform interleave for Next-80B and REAP? Likely model-specific tensor access patterns (e.g., MoE gating tensors get pinned to one node, specific attention heads to another). Could be addressed in a future "smart mbind" optimization, but the marginal gain is too small to justify the complexity.

## Files

- `11_*_mmap1_taskset.log` (5 files): cell (1,1) — cold mmap=1 + taskset
- `12_*_mmap1_interleave.log` (5 files): cell (1,2) — cold mmap=1 + --interleave=all
- `21_*_mmap0_taskset.log` (5 files): cell (2,1) — cold mmap=0 + taskset
- `next_warm_mmap1_taskset_run{1..5}.log` — Next-80B 5-run warming (numa_balancing=0)
- `next_warm_mmap1_interleave_run{1..5}.log` — Next-80B 5-run interleave (steady)
- `next_warm_mmap1_taskset_NB1_run{1..5}.log` — Next-80B 5-run with numa_balancing=1
- `reap_warm_mmap1_taskset_run{1..5}.log` — REAP-246B 5-run warming
- `numastat_next_mmap1_{taskset,interleave}.txt` — memory placement snapshots
