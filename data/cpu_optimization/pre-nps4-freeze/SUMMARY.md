# Pre-NPS4-reboot Baseline Freeze (2026-04-24)

Captured just before scheduled NPS4 BIOS reboot. All numbers on `llama.cpp-experimental` at `cpu-optimization/backlog-2026-04-23` branch (HEAD `9e048fbc1`).

## System state

- **NPS mode**: NPS2 (2 NUMA nodes × 6 channels each; distance 10/12)
- THP: `always`, defrag: `always`
- `numa_balancing`: 0 (off)
- `perf_event_paranoid`: 1
- 1× 1GB hugepage on node 1
- Governor: `performance`
- SMT: enabled (192 logical on 96 physical)

## Canonical single-instance thread sweep

**Model**: Qwen3-Coder-30B-A3B Q4_K_M, `-p 0 -n 64 -r 3`, quiet host, OMP build.

| Config | avg t/s | stddev |
|---|---|---|
| 24t (taskset 0-23) | **43.55** | 0.008 |
| 48t (taskset 0-47) | **45.53** | 0.28 |
| 96t (taskset 0-95, all physical both nodes) | **47.17** | 0.22 |
| 192t (--numa distribute -mmp 1) | 14.90 (bimodal 7.99/21.81) | 9.77 |

## CPU1 Phase 1.0 + 1.1 (noOMP build, `GGML_CCD_POOLS=1`)

**Model**: same as above. Compared to noOMP flat baseline and OMP production.

| Build | Config | t/s | Δ vs noOMP flat |
|---|---|---|---|
| OMP flat (production) | 96t | 47.17 | +21% |
| **noOMP + Phase 1.0+1.1 (CCD barrier + pinning)** | 96t (GGML_CCD_POOLS=1) | **44.85** | **+15%** |
| noOMP flat | 96t | 38.87 | — |

**Note**: this freeze measurement shows CCD +15% over noOMP flat, different from earlier-in-session measurement (−2% cold cache, then neutral). Page cache state and warm-up matter. The fair "apples-to-apples" comparison is OMP vs noOMP+CCD: **CCD recovers ~75% of the noOMP → OMP gap**. Still ~5% behind OMP.

## Concurrent-split sweep (copied from 2026-04-24 run)

**-p 0 -n 32 -r 2, SMT-paired cpusets per instance**:

| Model | 4×48t | 8×24t | 16×12t | 32×6t | 48×4t | Peak |
|---|---|---|---|---|---|---|
| Qwen3.6-27B Q8 | 6.62 | 7.91 | 8.55 | 10.47 | **15.39** | 48×4t |
| Qwen3.6-35B-A3B Q8 | 64.26 | 76.35 | 85.89 | 92.75 | **135.08** | 48×4t |
| Qwen2.5-Coder-32B Q4 | 13.64 | 15.08 | 16.01 | **20.03** | 17.34 ↓ | 32×6t |

35B-A3B Q8 at 48×4t = ~100% of 460 GB/s BW roofline.

## 2-way NUMA microbench (`tp_gemv_numa_bench`, 7.6 GB GEMV)

| Mode | GB/s |
|---|---|
| Flat 96t (no NUMA awareness) | 246.3 |
| 2×48t NUMA-local (mbind + first-touch) | 250.0 (+1.5%, noise) |

Verified page placement via `move_pages`.

## Artifacts

- `system-state.txt` — numactl, cpuinfo, knob settings
- `thread-{24,48,96,192}t.json` — single-instance thread sweep
- `cpu1-phase1a-b-96t.json` — CPU1 Phase 1.0+1.1 freeze
- `noomp-flat-96t.json` — noOMP baseline
- `concurrent-sweep-{16x12,32x6,48x4}.log` — concurrent aggregate sweep

## Post-reboot comparison protocol

See `handoffs/active/nps-reboot-runbook.md` § Step 5 for full re-benchmark protocol. Key configs to re-measure and compare:

1. Same thread sweep (24t/48t/96t/192t) — should be ~equal (compute unchanged)
2. Same CPU1 Phase 1.0+1.1 — **hope: +10-25% over NPS2's 44.85**
3. Same concurrent-split sweep — cpusets may need re-pinning to match new NUMA node boundaries
4. Extended NUMA microbench — 4-way instead of 2-way; measure if gain scales
