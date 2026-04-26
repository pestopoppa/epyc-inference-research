# CPU21 OpenMP Runtime/Scheduling Matrix — 2026-04-26 evening

**Goal**: identify whether OpenMP runtime tuning (libgomp affinity, schedule policy, wait policy) recovers any of the 96% parallelism loss attributed to sync overhead in CPU24-narrow.

**Method**: sweep on Coder-30B Q4_K_M (sync-bound class proxy). All runs use the proper canonical wrapper (`numactl --interleave=all --physcpubind=0-95 -t 96 -fa 1 -p 0 -n 32 -r 3 --mmap 0`); only the OpenMP env vars vary.

## Phase A — Affinity matrix

(default schedule, default wait policy)

| Config | t/s | Δ vs baseline |
|--------|-----|---------------|
| A_baseline_no_omp (default) | TBD | reference |
| OMP_PROC_BIND=close,  OMP_PLACES=cores | TBD | TBD |
| OMP_PROC_BIND=close,  OMP_PLACES=threads | TBD | TBD |
| OMP_PROC_BIND=spread, OMP_PLACES=cores | TBD | TBD |
| OMP_PROC_BIND=spread, OMP_PLACES=threads | TBD | TBD |
| OMP_PROC_BIND=master, OMP_PLACES=cores | TBD | TBD |
| OMP_PROC_BIND=false | TBD | TBD |

## Phase B — Schedule × chunk matrix

(default affinity)

| Config | t/s | Δ vs baseline |
|--------|-----|---------------|
| OMP_SCHEDULE=static,1 | TBD | TBD |
| OMP_SCHEDULE=static,4 | TBD | TBD |
| OMP_SCHEDULE=dynamic,1 | TBD | TBD |
| OMP_SCHEDULE=dynamic,4 | TBD | TBD |
| OMP_SCHEDULE=guided,1 | TBD | TBD |
| OMP_SCHEDULE=guided,4 | TBD | TBD |

## Phase C — Wait policy

| Config | t/s | Δ vs baseline |
|--------|-----|---------------|
| OMP_WAIT_POLICY=active | TBD | TBD |
| OMP_WAIT_POLICY=passive | TBD | TBD |

## Best stack (TBD)

Top 3 configurations:
1. TBD
2. TBD
3. TBD

## Discussion

(Fill in after data collection)

## Implications

- If best Phase A config delivers >3% over baseline → affinity tuning is a real lever; promote to default for production canonical
- If best Phase B config delivers >3% over baseline → schedule policy is a real lever
- If both are within noise → libgomp's defaults are already near-optimal; sync overhead is structural and CPU22 (dynamic balancing) is the right next track
- If best stack delivers >10% over baseline → significantly recovers the 96% parallelism loss, validates runtime-tuning hypothesis

## Files

(per-config logs, see `data/cpu_optimization/2026-04-26-cpu21/`)
