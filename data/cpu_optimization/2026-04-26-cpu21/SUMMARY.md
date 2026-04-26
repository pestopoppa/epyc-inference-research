# CPU21 OpenMP Runtime/Scheduling Matrix — 2026-04-26 evening

**Goal**: identify whether OpenMP runtime tuning (libgomp affinity, schedule policy, wait policy) recovers any of the 96% parallelism loss attributed to sync overhead in CPU24-narrow.

**Method**: sweep on Coder-30B Q4_K_M (sync-bound class proxy). All runs use the proper canonical wrapper (`numactl --interleave=all --physcpubind=0-95 -t 96 -fa 1 -p 0 -n 32 -r 3`); only the OpenMP env vars vary. `drop_caches` between runs.

## Phase A — Affinity matrix (default schedule, default wait policy)

| Config | t/s | Δ vs baseline |
|--------|-----|---------------|
| A_baseline_no_omp (default) | 43.82 ± 0.04 | reference |
| OMP_PROC_BIND=close,  OMP_PLACES=cores | 45.91 ± 0.19 | +4.8% |
| OMP_PROC_BIND=close,  OMP_PLACES=threads | 46.15 ± 0.05 | +5.3% |
| **OMP_PROC_BIND=spread, OMP_PLACES=cores** | **46.52 ± 0.12** | **+6.2%** ← best in Phase A |
| OMP_PROC_BIND=spread, OMP_PLACES=threads | 45.98 ± 0.06 | +4.9% |
| OMP_PROC_BIND=master, OMP_PLACES=cores | HUNG (killed after 6 min) | n/a — pathological |
| OMP_PROC_BIND=false | 43.90 ± 0.11 | +0.2% (~baseline) |

**Finding**: explicit affinity binding (`close` or `spread`) consistently delivers **+5-6% over default**. `spread + cores` is the best single-knob change. `master` mode binds all 96 threads to the master's CPU — pathological at 96-thread scale. `false` is equivalent to default.

## Phase B — Schedule × chunk matrix (default affinity)

| Config | t/s | Δ vs baseline |
|--------|-----|---------------|
| OMP_SCHEDULE=static,1 | 43.78 ± 0.03 | −0.1% |
| OMP_SCHEDULE=static,4 | 44.11 ± 0.04 | +0.7% |
| OMP_SCHEDULE=dynamic,1 | 43.84 ± 0.11 | 0.0% |
| OMP_SCHEDULE=dynamic,4 | 43.83 ± 0.07 | 0.0% |
| OMP_SCHEDULE=guided,1 | 43.81 ± 0.08 | 0.0% |
| OMP_SCHEDULE=guided,4 | 43.96 ± 0.04 | +0.3% |

**Finding**: schedule policy is **within noise**. libgomp's defaults (and llama.cpp's internal scheduling logic) are already near-optimal for this workload — each thread is assigned a deterministic slice of the matmul rows; explicit OMP_SCHEDULE doesn't change the partitioning materially.

## Phase C — Wait policy

| Config | t/s | Δ vs baseline |
|--------|-----|---------------|
| OMP_WAIT_POLICY=active | 44.03 ± 0.01 | +0.5% |
| **OMP_WAIT_POLICY=passive** | **8.04 ± 0.05** | **−81.6% catastrophic** |

**Finding**: `OMP_WAIT_POLICY=passive` is a deployment trap. It puts threads to sleep on barrier wait (vs. spin-waiting). At 96 threads with hundreds of barriers per token, the wake-up latency dominates — 5.5× regression. The default behavior is already close to active; explicit `active` gives a small +0.5% bump.

## Best stack identified (Coder-30B Q4_K_M)

**`OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`** — combination of best Phase A + best Phase C. Schedule policy doesn't matter, leave default.

Expected combined gain: ~+6.5% over baseline (additive of +6.2% affinity + +0.5% active wait, capped by some overlap).

## Cross-model verification — COMPLETE

Follow-up sweep on REAP-246B Q4_K_M (sync-bound large MoE), Qwen3.6-35B Q8_0 (BW-bound), and Coder-30B Q4_K_M with combined stack (additivity check):

### REAP-246B Q4_K_M

| Config | t/s | Δ vs baseline |
|---|---|---|
| REAP_baseline | 6.14 ± 0.01 | reference |
| REAP_spread_cores | 6.33 ± 0.01 | +3.1% |
| REAP_close_threads | 6.34 ± 0.00 | +3.3% |
| REAP_combined_stack (spread+cores + active) | 6.33 ± 0.00 | +3.1% |

**Finding**: affinity helps but smaller gain (~3%) than on Coder-30B (~6%). REAP is dominated by structural sync overhead at the BARRIER level (96 sync points/token × 96 layers) — affinity reduces inter-thread coordination latency but can't address the fundamental sync count. Active wait policy is neutral on REAP (combined ≈ spread alone).

### Qwen3.6-35B-A3B Q8_0

| Config | t/s | Δ vs baseline |
|---|---|---|
| Q8_baseline | 21.36 ± 0.04 | reference |
| Q8_spread_cores | 22.96 ± 0.05 | +7.5% |
| Q8_close_threads | 23.02 ± 0.04 | +7.8% |
| Q8_combined_stack (spread+cores + active) | **23.04 ± 0.01** | **+7.9%** |

**Finding**: BIGGEST gain on the BW-bound class (+7.9%). Affinity tuning interacts well with the CPU2 auto-mbind on the CPU_REPACK buffer — pinned threads access their assigned buffer regions with consistent NUMA distance.

### Coder-30B Q4_K_M (combined stack additivity check)

| Config | t/s | Δ vs default |
|---|---|---|
| baseline (from main sweep) | 43.82 ± 0.04 | reference |
| spread_cores alone | 46.52 ± 0.12 | +6.2% |
| active wait alone | 44.03 ± 0.01 | +0.5% |
| **combined_stack** | **47.08 ± 0.15** | **+7.4%** |

Combined ≈ spread alone (+6.2%) + active alone (+0.5%) = +6.7% sum vs +7.4% measured → near-additive (slightly super-additive within noise). Confirms the levers compose.

### Cross-class summary (FULL 5-model picture, 2026-04-26 evening)

| Model | Class | Baseline (no OMP) | Combined stack | Δ |
|-------|-------|-------------------|----------------|---|
| Qwen3-Coder-30B-A3B Q4_K_M | sync-small | 43.82 | **47.08** | **+7.4%** |
| Qwen3.6-35B-A3B Q8_0 | BW-bound | 21.36 | **23.04** | **+7.9%** |
| Qwen3-Next-80B-A3B Q4_K_M | sync-small/hybrid | 21.37 | **22.15** | **+3.7%** |
| Qwen3-Coder-REAP-246B-A35B Q4_K_M | sync-large | 6.14 | **6.33** | **+3.1%** |
| gemma-4-26B-A4B-it Q4_K_M | sync-small/mixed | 36.45 | **38.59** | **+5.9%** |

Affinity tuning is **the first universal-positive lever** identified in the 2026-04 CPU optimization work. Every prior "win" had asymmetric/regressive cases on some model. CPU21 combined stack is positive on every class — modest on REAP (+3%, capped by structural sync), strong on BW-bound and small sync-bound (+7-8%).

## Implications

1. **Affinity tuning is a real, free, robust lever** (~+5-6% on Coder-30B). Should be added to the canonical config alongside `numactl --interleave=all`.
2. **Sync-class is recoverable at the runtime layer.** The CPU24-narrow finding (96% parallelism loss to sync overhead) is partially mitigated by better thread placement — explicit affinity reduces inter-thread coordination latency by keeping threads on cores with predictable cache topology.
3. **Schedule policy has no headroom on this workload.** libgomp's defaults are good; CPU22 (dynamic load balancing) would need to operate at a HIGHER level than `OMP_SCHEDULE` to be useful (e.g. work-stealing inside `mul_mat_id`, not at OpenMP partitioning level).
4. **OMP_WAIT_POLICY=passive must be explicitly avoided** in any deployment script. Worth a guard in `cpu-benchmark-rigor-and-revalidation.md` (CPU20 protocol) to validate `OMP_WAIT_POLICY != passive` at session start.

## Updated canonical baseline (proposed)

The current canonical (`numactl --interleave=all --physcpubind=0-95 -t 96 -fa 1`) should be extended to:

```
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
  numactl --interleave=all --physcpubind=0-95 \
  llama-bench -t 96 -fa 1 -p 0 -n 32
```

This becomes the new "proper canonical" reference for any cross-session optimization comparison.

## Files

- `A_baseline_no_omp.log`, `A_proc_*.log` (6 affinity variants; master_cores is partial/killed)
- `B_static_*.log`, `B_dynamic_*.log`, `B_guided_*.log` (6 schedule variants)
- `C_active.log`, `C_passive.log` (2 wait policies)
- `cpu21_followup.sh` (post-sweep cross-model verification)
- `followup/` (REAP-246B + Qwen3.6-35B Q8_0 + Coder-30B combined stack)
