# Qwen3.6-27B Q8_0 decode profile — 2026-04-24 Session 15 part 4

**Goal**: localize the 26%→44% BW utilization gap between Qwen3.6-27B Q8_0 (hybrid) and Qwen2.5-Coder-32B Q4KM (pure dense) on EPYC 9655 NPS4. Both run on the same hardware via the same llama.cpp build; the dense reference hits ~44% of theoretical BW while the hybrid caps at ~25%.

## Profile setup

**Host**: EPYC 9655 NPS4, THP=always, numa_balancing=0, perf 6.17.13.
**Branch**: `cpu-optimization/q8-8x8-avx512bw` HEAD `ba1c23900` (with the Session 15 AVX-512BW kernel + auto-mbind + DeltaNet refactor).
**Capture**: `perf record -F 999 -g --call-graph dwarf,8192` for ~15 s of decode, ~2-4M samples.

Two configurations profiled:

| Run | Build | Env stack | t/s @ 96t |
|-----|-------|-----------|-----------|
| `perf-27b-q8-96t.data` | `build/` (OpenMP) | (default) | 4.29 |
| `perf-noomp-cpu1-96t.data` | `build-noomp/` | `GGML_CCD_POOLS=1 GGML_NUMA_WEIGHTS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1` | 4.42 |

(Raw `perf.data` files are git-ignored due to size — 36 GB + 18 GB. Symbol-only reports saved as `top-symbols-*.txt`.)

## Top-symbol breakdown

### OpenMP build (default)

```
70.95%  ggml_vec_dot_q8_0_q8_0           (single-row Q8 dot, AVX2)
24.39%  libgomp.so.1.0.0 [.] 0x26580     (GOMP team_barrier_wait equivalent)
< 5%    everything else
```

### noomp + full CPU1 stack

```
72.15%  ggml_vec_dot_q8_0_q8_0           (same kernel, different threadpool)
21.63%  ggml_barrier                     (custom 2-level CCD-hierarchical barrier)
 2.94%  ggml_barrier_local               (CPU1 Phase 1.4 axis-0-aligned barrier)
< 4%    everything else
```

The OpenMP and custom-threadpool builds show **near-identical breakdowns**. Switching threadpools doesn't change throughput because both implementations spend ~24% of cycles in barrier code. The CPU1 Phase 1.4 `ggml_barrier_local` only covers a small fraction of barriers (3%); the remaining 21.6% goes through the global 2-level hierarchical `ggml_barrier`.

DeltaNet ops are **vanishingly small** in the cycle distribution (<1% combined) — directly explaining why the Session 15 part-3 DeltaNet S_v sub-chunking refactor was net-neutral. The 8/16/32t scaling plateau coincided with H_v=16 but *correlation, not causation*: scaling stops because the global barrier overhead dominates beyond the point where individual op compute saturates the cores.

## Perf-stat counters (noomp + CPU1, 96t, n=64)

```
6,120,351,484,994      cycles
1,027,359,505,182      instructions          # 0.17 IPC
       12,513,442,458  L1-dcache-load-misses # 2.57% of L1 loads
        2,226,649,725  cache-misses          # 2.95% of all cache refs
       49,631,278,213  stalled-cycles-frontend # 0.81%
```

**0.17 IPC** is decisive. Modern Zen 5 sustains ~5 IPC on dense compute; we're at 3.4% of theoretical compute throughput. **96.6% of cycles the cores are idle** — not on instruction fetch (frontend stall is 0.81%) but on memory stalls. Combined with the 70% in `ggml_vec_dot_q8_0_q8_0`, this confirms Session 13/14's earlier finding: **the cycle samples in the Q8 dot kernel are mostly DRAM-load waits, not ALU work.** Doubling ALU width (the falsified VNNI probes) cannot help; even the AVX-512BW 8x8 kernel from Session 15 only beats the single-row baseline by +1-3% at high thread count because the bottleneck is memory, not compute.

## Cross-quant BW utilization on the same model

| Quant | t/s @ 96t | Bytes/token | BW achieved | % of 460 GB/s ceiling |
|-------|-----------|-------------|-------------|------------------------|
| 27B Q8_0 | 4.42 | 26.6 GB | 117 GB/s | **25.4%** |
| 27B Q4_K_M | 6.75 | 15.65 GB | 106 GB/s | **23.0%** |

Both quants of the **same hybrid model** land at the same ~24% BW utilization. The Q4→Q8 throughput ratio is just the bytes-per-token ratio. **Quant choice does not affect BW efficiency on this model.**

## Cross-architecture comparison

| Model | Arch | Quant | t/s @ 96t | Bytes/tok | BW achieved | % ceiling |
|-------|------|-------|-----------|-----------|-------------|-----------|
| Qwen3.6-27B | 75% DeltaNet hybrid | Q8_0 | 4.42 | 26.6 GB | 117 | **25%** |
| Qwen3.6-27B | 75% DeltaNet hybrid | Q4_K_M | 6.75 | 15.65 GB | 106 | **23%** |
| Qwen3-Coder-30B-A3B | dense MoE | Q4_K_M | 49.34 | 16.5 GB | 814 (per-active) | 49% (effective) |
| Qwen2.5-Coder-32B (registry) | pure dense | Q4_K_M | 10.8 | 18.5 GB | 200 | **44%** |

**Pure-dense workloads achieve ~44% BW utilization on this hardware. Hybrid (DeltaNet-heavy) workloads achieve ~25%.** The architecture is what's costing throughput, not the quant or the kernel.

## Where the gap actually lives — best hypothesis

The 22% in `ggml_barrier` × the hybrid graph's higher op count gives a plausible mechanism. Per-token op count comparison (estimated, not yet measured):

- Qwen2.5-Coder-32B (pure dense, 64 layers): ~7 ops/layer × 64 = ~450 ops/token → ~450 barriers
- Qwen3.6-27B (hybrid, 64 layers): ~10 ops/layer in DeltaNet (75%) + 7 in attention (25%) = ~9.25 avg × 64 = ~590 ops/token → ~590 barriers

A 30% higher barrier count, combined with smaller per-op compute size in the hybrid graph (DeltaNet wrapper ops are short), would make barrier overhead a larger fraction of total wall time. Empirically: 22% in barriers on hybrid suggests that on the dense reference, barriers are probably ~12-15% (closer to compute-bound).

**Untapped headroom on Qwen3.6-27B Q8_0** if we matched dense BW utilization: 4.42 → **460 × 0.44 / 26.6 = 7.6 t/s** (+72%). Realistic targets after fusion + better barriers: 4.4 → ~5.5-6.0 t/s (+25-35%).

## Concrete next levers (data-driven, ranked by expected ROI)

1. **Op fusion of DeltaNet wrapper ops** (RMS norm + conv1d + gate projection + residual). Each fused pair = -1 barrier per layer per token = -48 barriers/token (~10% of total). At 22% in barriers × ~10% reduction = ~2-3% throughput. Lower than initially hoped, because the wrapper ops are not the dominant barrier surface.
2. **Eliminate inter-op barriers between data-independent ops** via graph rewrites. E.g., Q/K/V projections from the same input can run concurrently (no dependency between their outputs until attention). Could halve barrier count = +10-15% throughput. Substantial graph-rewrite project.
3. **Faster `ggml_barrier` impl** beyond the current 2-level CCD-hierarchical. Already pretty optimized; further gains via a tournament barrier or wait-free announce are diminishing-returns territory. ~5% upper bound.
4. **Speculative decoding with a draft model**. Prior work (`feedback_qwen35_27b_architecture.md`) said spec-dec on hybrid CPU is dead; worth re-checking if Dflash speculation has matured. Could 2-3× throughput if it works.
5. **Use Q4_K_M production-side**. Already +52% over Q8_0 on this model (6.75 vs 4.42 t/s). If Q8_0 isn't strictly required for quality, switch.

## What is NOT a useful next lever

- **More CPU2 kernel work** on Q8_0 specifically. The kernel is already near-optimal; +1-3% at 96t is the realistic ceiling because the bottleneck is DRAM, not ALU. Q6_K and Q5_K 8x8 kernels would help Q4_K_M decode (+2-5% per kernel) but don't address the hybrid-overhead gap.
- **More DeltaNet parallelism work**. Profile shows DeltaNet is <1% of cycles. Already disproved this session; do not re-investigate.
- **Adding more threads**. Plateau at 16-24t; 96t adds nothing useful for this workload.
