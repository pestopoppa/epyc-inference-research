# Multi-Arch Coverage Probe — CPU1 + CPU2 mbind on Dense / Hybrid SSM / Dense Q4

> **⚠️ DATA INTEGRITY WARNING (added 2026-04-29 post-hoc, REVISED)**: this bundle's measurements (both first-pass n=5 and n=30 replication) are unreliable due to TWO compounding causes discovered post-hoc:
>
> 1. **Concurrent-agent contention**: 3 other claude sessions active during measurement window may have run llama-bench concurrently. The 21% baseline drift on gemma-31B between first-pass and replication is consistent with intermittent CPU contention.
>
> 2. **Host-level throttle (ROOT CAUSE)**: the EPYC 9655 was in a degraded power state (38 of 96 cores stuck at 1998 MHz under load instead of expected 2800-3000 MHz all-core boost). Coder-30B Q4_K_M tg32 reproduced at **11-20 t/s vs production canonical 58.65 t/s** — 3-5× regression. This is hardware-level, not just contention. User rebooted the host to restore.
>
> **All absolute numbers in this bundle are unreliable.** Relative comparisons (CPU1 vs no-CPU1, mbind on vs off) may still be approximately correct since both arms ran under the same throttle. The clean re-run with per-cell pgrep guards (`../2026-04-29-multi-arch-coverage-rerun/`) is also affected by the host throttle and should be re-run from scratch on the post-reboot host.

**Date**: 2026-04-29
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/` after CPU4 Phase 1 patch (commit `9f6191581`).
**Source**: User direction "Multi-arch coverage — test the existing v5 PGO + CPU1 stack + CPU2 mbind on dense, hybrid SSM, attention-only models" 2026-04-29.
**Coverage gap (per cpu-kernel-env-flags-inventory (in epyc-root handoffs/active/))**: CPU1 stack tested only on Coder-30B Q4_K_M (+1.8%) and Qwen3.6-35B Q8_0 (parity); CPU2 mbind tested only on Q8_0 MoE (+6%) and Q4_K_M MoE (-0.9%). Dense Q8, hybrid SSM, dense Q4 untested.

## Method

5-rep canonical (`taskset -c 0-95 -t 96 -fa 1`, no other env vars except the test config). llama-bench `-p 0 -n 64 -r 5` (tg64). v5 PGO build.

Tested 3 architecture classes × 4 flag configs:

| Tag | Model | Arch class |
|---|---|---|
| qwen36_27b | Qwen3.6-27B-Q8_0 | Dense Q8 (BW-bound, 28 GB) |
| nemotron_9b | NVIDIA-Nemotron-Nano-9B-v2-Q8_0 | Hybrid SSM (Mamba2 + attention) |
| gemma_31b | gemma-4-31B-it-Q4_K_M | Dense Q4_K_M |

| Config | Env flags |
|---|---|
| c0 | (none — default `GGML_NUMA_REPACK_INTERLEAVE=1` per CPU2 mbind kill-switch) |
| c1 | `GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1` (CPU1 stack) |
| c2 | `GGML_NUMA_REPACK_INTERLEAVE=0` (CPU2 mbind OFF) |
| c3 | CPU1 stack + CPU2 mbind OFF |

## First-pass results (n=5 reps each)

| Model | Config | tg64 t/s | Δ vs baseline |
|---|---|---|---|
| Qwen3.6-27B Q8 | baseline | 1.68 ± 0.03 | — |
| Qwen3.6-27B Q8 | CPU1 | 1.71 ± 0.03 | **+1.79%** |
| Qwen3.6-27B Q8 | CPU2-off | 1.68 ± 0.05 | +0.00% |
| Qwen3.6-27B Q8 | CPU1 + CPU2-off | 1.70 ± 0.04 | +1.19% |
| Nemotron-9B Q8 | baseline | 6.48 ± 0.64 | — |
| Nemotron-9B Q8 | CPU1 | 6.67 ± 0.71 | +2.93% (high CV) |
| Nemotron-9B Q8 | CPU2-off | 6.44 ± 0.90 | -0.62% |
| **Nemotron-9B Q8** | **CPU1 + CPU2-off** | **7.35 ± 0.15** | **+13.43%** |
| gemma-4-31B Q4 | baseline | 4.45 ± 0.29 | — |
| gemma-4-31B Q4 | CPU1 | 3.90 ± 0.66 | **-12.36%** |
| gemma-4-31B Q4 | CPU2-off | 3.96 ± 0.31 | -11.01% |
| gemma-4-31B Q4 | CPU1 + CPU2-off | 3.93 ± 0.10 | -11.69% |

## Headline findings

### 1. Hybrid SSM (Nemotron-9B Q8): CPU1 + mbind=off = +13.43% (TENTATIVE — verifying at n=30)

The combination of CPU1 stack + CPU2 mbind OFF on Nemotron hybrid SSM produced **+13.43% with low std (±0.15)**. CPU1 alone gave +2.93% (high CV); mbind-off alone was -0.62% (margin). The combination is super-additive.

Hypothesis: with mbind ON, weights interleaved across NUMA nodes (1/4 local for any thread under NPS4); with mbind OFF, weights concentrate on whichever node first-touched at load. CPU1's CCD-aware partitioning might pair more efficiently with concentrated weight placement than with interleaved placement on this specific architecture.

The c0 baseline std (0.64) was high (~10% CV) at n=5 — replicating at n=30 to confirm before claiming the +13.43% as production-pushable.

### 2. Dense Q8 (Qwen3.6-27B): CPU1 +1.79%, CPU2 mbind no effect

CPU1 stack delivers a small positive (+1.79%) consistent with the prior Coder-30B Q4_K_M finding (+1.8% at P3). CPU2 mbind has no effect on dense Q8 (the mbind only helps when CPU_REPACK buffer matters; dense Q8 is direct-read). 

### 3. Dense Q4 (gemma-4-31B): consistent -11 to -12% across all configs (TENTATIVE — verifying at n=30)

Both CPU1 alone, CPU2-off alone, and combined ALL hurt by ~11-12%. This is the first model class where CPU1 stack is NEGATIVE. Possible causes:
- gemma-4 op chain has different shape (sliding-window attention?) that doesn't fit CCD-aware partitioning
- gemma-4 quantization layout interacts poorly with CCD work-distribution
- High variance baseline (±0.29) — n=5 with bursty noise floor — could be noise, but the SAME direction across 3 different configs suggests it's real

Replicating at n=30 to confirm.

## n=30 high-rep replication (in progress / pending)

[FILLED IN AFTER REPLICATION COMPLETES]

## Implications (preliminary, pending replication)

If +13.43% on Nemotron and -12% on gemma are confirmed at n=30:

1. **CPU1 stack + mbind-off** becomes a **per-arch deployment recommendation**, not a universal toggle. Specifically:
   - Hybrid SSM (Nemotron, possibly Qwen3-Next): enable both
   - Dense Q8 (Qwen3.6-27B, similar): enable CPU1 stack alone (small +)
   - MoE Q4_K_M (Coder-30B, REAP-246B, Next-80B): existing recommendation (CPU1 default-off opt-in)
   - Dense Q4 (gemma family): DO NOT ENABLE CPU1 stack (regression)

2. **The whitelist needs per-arch tagging**, not just per-flag. The current inventory treats CPU1 as universally default-off opt-in; this probe shows it has clear winners and clear losers across architectures.

3. **CPU2 mbind = `GGML_NUMA_REPACK_INTERLEAVE=1` default ON** may need revisiting for hybrid SSM — the +0.62% penalty on Nemotron c2 (mbind off) suggests mbind helps slightly there alone, but interferes with CPU1 stack when combined.

## Files

- `run_probe.sh` — first-pass measurement script (3 models × 4 configs × n=5)
- `run_replicate.sh` — high-rep replication (Nemotron c0/c3 + gemma c0/c1 × n=30)
- `probe_master.log` / `replicate_master.log` — master logs with timing + aggregates
- `bench_*.log` — llama-bench logs for each cell
- `rep30_*.log` — replication logs at n=30
- decision (markdown, pending) — final verdict + closure scope (after replication)
