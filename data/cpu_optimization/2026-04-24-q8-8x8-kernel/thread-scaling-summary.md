# Q8_0 AVX-512BW 8x8 GEMV — Thread Scaling + NUMA Placement

**Date**: 2026-04-24
**Model**: Qwen3.6-27B-Q8_0 (26.6 GB)
**Host**: EPYC 9655 NPS4, THP=always, numa_balancing=0
**Build**: `cpu-optimization/q8-8x8-avx512bw` branch
**Flags**: `-fa 1 --numa distribute -p 0 -n ≥32`

## The NUMA-placement finding

First measurements showed the 8x8 repack path regressing 56–73% vs baseline at 4–96 threads. Root cause was NOT the kernel and NOT activation quantization — it was that **`ggml_aligned_malloc` for the CPU_REPACK buffer first-touches all pages on NUMA node 0**, so the 26.6 GB of repacked weights live on one node and 96 threads across 4 NPS4 nodes all read through that single node's memory controllers.

The baseline non-repacked path uses `mmap` on the GGUF file, and the CPU1 Phase 1.3 infrastructure (`GGML_NUMA_WEIGHTS=1` → `set_mempolicy(MPOL_INTERLEAVE)`) is applied before mmap so its pages distribute across all nodes. `GGML_NUMA_WEIGHTS=1` sets a *process-wide* memory policy which also affects subsequent allocations — so merely exporting that env var before model load interleaves the CPU_REPACK buffer too.

## Thread scaling results (with `GGML_NUMA_WEIGHTS=1` for both paths)

| Threads | Baseline (non-repacked) | Repack 8x8 + AVX-512BW | Δ |
|---------|-------------------------|-------------------------|---|
| 1 | 0.84 t/s | **1.12 t/s** | **+33.3%** |
| 12 | 4.41 | 4.54 | +2.9% |
| 24 | 4.50 | 4.54 | +0.9% |
| 48 | 4.51 | 4.56 | +1.1% |
| 96 | 4.42 | 4.45 | +0.7% |

At production thread counts (96t, NPS4) the two paths converge to the same ~4.5 t/s — this is the **memory-bandwidth ceiling** of Q8_0 decode on this hardware (~26% of the 17 t/s theoretical roofline at 460 GB/s). Both paths saturate the ceiling; the 8x8 kernel has a small but consistent +1–3% edge.

At 1 thread the AVX-512BW 8x8 kernel beats the single-row AVX2 baseline by +33% — the 8-row-amortization win is visible when DRAM bandwidth isn't saturated.

## Before-vs-after table (why the NUMA fix matters)

| Threads | Baseline | Repack+BW w/o NUMA_WEIGHTS | Repack+BW w/ NUMA_WEIGHTS |
|---------|----------|----------------------------|---------------------------|
| 1 | 0.84 | 1.05 | 1.12 |
| 24 | 4.46 | 1.60 | 4.54 |
| 96 | 4.38 | 1.58 | 4.45 |

Without `GGML_NUMA_WEIGHTS=1` the repack path caps at ~1.6 t/s regardless of thread count — the one-node BW ceiling, not the aggregate-node BW ceiling.

## Correctness

- PPL on Wikitext-2 (3 chunks, ctx=512, `-fa on --numa distribute`) with AVX-512BW path active: **6.6985 ± 0.708**. Sensible baseline, no NaN, no divergence.
- Disassembly of `ggml_gemv_q8_0_8x8_q8_0`: hot loop emits `vpabsb` + `vpmaddubsw` + `vpmaddwd` + `vpaddd` — the intended AVX-512BW path, NOT `vpdpbusd` (VNNI was falsified twice on Zen 5 in Sessions 13/14).

## Ancillary fix (kept in-branch)

`forward_mul_mat` in `tensor_traits` previously serialized the "remainder row" activation quantization: `for (i11 = i11_processed + ith; i11 < ne11; i11 += nth)` meant only thread 0 ran `from_float` when ne11=1, all others hit the barrier. Replaced with the K-parallel pattern from `ggml_compute_forward_mul_mat` (ggml-cpu.c:1466-1475), partitioning K-blocks across threads per row. Effect measured as neutral (NUMA placement dominated), but it's still a correct parallelization that matches the standard-path convention and will compose with future fixes.

## Decision

- **Kernel + repack landed on `cpu-optimization/q8-8x8-avx512bw`**, default OFF behind `GGML_Q8_0_8X8=1` + `GGML_Q8_0_8X8_AVX=1`.
- **Safe to flip default ON for NUMA hosts running with `GGML_NUMA_WEIGHTS=1`** — matches baseline at BW ceiling, +33% at 1t, no correctness regression.
- **Production NUMA hosts must set `GGML_NUMA_WEIGHTS=1`** or the repack buffer hotspots one node. Proposed follow-up: auto-mbind the CPU_REPACK buffer to MPOL_INTERLEAVE when `ggml_is_numa()` is true, so the fix is default-on without needing the env var.

## Recommended follow-ups

1. **Auto-mbind the CPU_REPACK buffer** on multi-NUMA systems (inside `ggml_backend_cpu_repack_buffer_type_alloc_buffer`) so the NUMA fix is default-on without requiring `GGML_NUMA_WEIGHTS=1`. Narrow-scope change, pays off for every repacked quant, not just Q8_0.
2. **Follow the same 8x8 pattern for Q6_K and Q5_K** (Session 14 left these dispatcher-NEON-only too; same 77.4%-in-vec_dot profile as Q8_0 before this session, though with more complex kernels due to 4+2 bit-split unpacking).
3. **Cross-check Q4_K_M at 1t** to see if the +33% single-thread win applies there too — if Q4_K_M hits 1t gains as well, the tensor_traits path is demonstrably correct at any thread count when NUMA is handled.
