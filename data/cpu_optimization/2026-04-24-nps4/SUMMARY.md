# Post-NPS4 Reboot Re-benchmark (2026-04-24, later session)

System rebooted into NPS4 per `handoffs/active/nps-reboot-runbook.md`. All measurements on `llama.cpp-experimental` branch `cpu-optimization/backlog-2026-04-23`, HEAD `a64d27dee` (Phase 1.0+1.1 commit on top of `9e048fbc1`). THP=always, numa_balancing=0, perf_event_paranoid=1 re-applied after reboot. Governor: performance, boost active (verified 4.5 GHz under load).

## NUMA topology

```
available: 4 nodes (0-3)
node 0 cpus: 0-23, 96-119    size: 289860 MB
node 1 cpus: 24-47, 120-143  size: 290287 MB
node 2 cpus: 48-71, 144-167  size: 290287 MB
node 3 cpus: 72-95, 168-191  size: 290182 MB
distances: 10 / 12
```

Each NUMA node now owns 3 DDR5 channels (12 total / 4 nodes).

## Single-instance thread sweep — Qwen3-Coder-30B-A3B Q4_K_M

`-p 0 -n 64 -r 3` via llama-bench, quiet host.

| Config | t/s |
|---|---|
| 24t node0 phys (`taskset 0-23`) | 15.32 |
| 24t `cpunodebind=0 membind=0` | 15.37 |
| 48t node0 phys+SMT | 14.50 |
| 48t nodes 0,1 phys | 18.81 |
| 48t `interleave=0,1` | 21.66 |
| 48t `membind=0,1` | 19.18 |
| 96t all-phys `taskset 0-95` | 21.58 |
| 96t `membind=all` | 22.29 |
| 96t `--numa distribute -mmp 1` | 22.64 |
| 96t `interleave=all` (OMP build) | **25.35** |
| 144t interleave | 24.69 |
| 192t `--numa distribute` | 18.25 |

## CPU1 Phase 1.0+1.1 (noOMP build, 96t)

| Config | t/s | Δ vs OMP+interleave |
|---|---|---|
| noOMP flat | 14.01 | — |
| noOMP + CCD pools (`GGML_CCD_POOLS=1`) | 15.07 | — |
| noOMP flat + `interleave=all` | 24.77 | baseline |
| **noOMP + CCD pools + `interleave=all`** | **27.86** | **+12.5%** |

## Concurrent-split (30B-A3B Q4, `-p 0 -n 32 -r 2`, NPS4-native per-node membind)

| Layout | Aggregate t/s |
|---|---|
| 4×48t (1 inst per node, 48 logical each) | 36.17 |
| 4×24t phys (1 inst per node, 24 phys each) | 37.12 |
| **48×4t** (12 inst per node, 2 phys+2 SMT per inst) | **104.35** |

Frontdoor 35B-A3B Q4 4×48t NPS4 = **37.74** t/s aggregate (vs NPS2 registry ~50.8).

## NPS2 vs NPS4 head-to-head

| Metric | NPS2 (freeze) | NPS4 (best) | Δ |
|---|---|---|---|
| Single-inst 96t OMP flat | 47.17 | 25.35 | −46% |
| Single-inst CPU1 P1.0+1.1 | 44.85 | 27.86 | −38% |
| 4×48t frontdoor agg | 50.8 | 37.74 | −26% |
| 24t single-node | 43.55 | 15.37 | −65% |

## Why interleave does NOT fully recover NPS2 performance

1. **Remote-access ratio grows.** NPS4+interleave → 75% remote per thread (3 of 4 are non-local). NPS2+interleave → 50%. 1.5× hop-latency tax.
2. **Within-node channel stripe halves.** Sequential burst inside one node only hits 3 channels instead of 6.
3. **Directory coherency overhead scales with domain count.** 4 domains have more snoop/lookup paths than 2.

`numa_balancing=1` auto-migration was tested (20.42 t/s) — slightly WORSE than `=0` (21.03 t/s). Migration during active decode costs more than it saves.

## Decision gates

- ✅ **+12.5% CPU1 gain** under NPS4 meets the ">10%" proceed-with-Phase-1.2/1.3 gate.
- ❌ **−26% multi-instance 4×48t** regression meets the "rollback" gate.

Verdict: conflicting gates; user decision needed. Recommendation: **Option 2 — stay on NPS4 and implement Phase 1.3** (NUMA-bound weight mbind, 2-3 days). If Phase 1.3 closes the single-instance gap to within 10% of NPS2, we keep NPS4 and gain the CPU1 uplift + 48×4t concurrent peak. Rollback remains a safe fallback if Phase 1.3 disappoints.

## Artifacts

- `thread-sweep/*.json` — single-instance thread sweep
- `noomp-*.json` — CPU1 Phase 1.0+1.1 measurements
- `concurrent/*.json` — 4×48, 4×24phys, 48×4 per-instance JSON
- `thread-sweep-nps4.sh`, `concurrent-4x48-nps4.sh`, `concurrent-48x4-nps4.sh` — sweep scripts
