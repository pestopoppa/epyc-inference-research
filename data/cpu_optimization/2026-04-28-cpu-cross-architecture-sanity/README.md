# Cross-Architecture Sanity Coverage — CPU2/CPU21/CPU25/libomp on Dense/Hybrid (artifact bundle)

**Track**: Phase 2.6 of closure-inflation remediation plan ([handoff cluster](../../../../../workspace/handoffs/active/cpu-inference-optimization-index.md))
**Run date**: 2026-04-28
**Purpose**: Close peer-review finding #11 (the CPU optimization track has been almost entirely MoE-focused; most "exhausted" findings stated in MoE terms when underlying mechanism may be architecture-independent). This bundle measures four already-validated MoE optimizations on the dense/hybrid Qwen3.6-27B Q8_0 (hybrid SSM-Dense per memory `feedback_qwen35_27b_architecture`; metadata reports `qwen35 27B Q8_0`, 26.62 GiB, 26.90 B params).

## Verdict

**All four tested CPU optimizations are NEUTRAL on dense/hybrid Qwen3.6-27B Q8_0** (within run-to-run noise, ±2% combined std).

| Optimization | dense Qwen3.6-27B Q8 | Coder-30B Q4_K_M MoE (reference) | Generalization |
|---|---|---|---|
| CPU21 affinity stack (`spread+cores+active`) | 4.65 → 4.66 (+0.2%) | 43.82 → 46.52 (+6.2% to +8%) | **NEGATIVE** — MoE-only |
| CPU25 NUMA_MIRROR=4 | 4.77 → 4.71 (-1.3%, within noise) | 48.16 → 47.66 (-1.0%, within noise) | **CONFIRMED NEGATIVE both classes** |
| CPU2 Q8_0 8x8 AVX-512BW SIMD | 4.78 → 4.73 (-1.0%, within noise) | +1.6% @ 96t (BW-saturated; bigger at low thread count) | **NEGATIVE** — kernel is BW-bound at 96t both classes |
| libomp runtime (clang+libomp+znver5) | 4.65 → 4.69 (-1.7% vs gcc-no-march; within noise) | 50.06 → 53.28 (+6.4% vs libgomp+znver5) | **MoE-specific gain confirmed** |

## Why dense doesn't benefit

The dense/hybrid class is **uniformly BW-bound**:
- Dense + Q8 = 26 GB total weights / 96 threads = ~0.27 GB/thread × 4.79 GB/s/thread BW share = pure DRAM streaming.
- All threads do uniform work (no MoE routing variation), so static partitioning is already optimal.
- No barrier/sync overhead amplification: with uniform compute, threads finish their row-shards together; libomp's lower-overhead barriers don't help.
- The Q8_0 8x8 SIMD kernel's compute throughput exceeds the per-thread DRAM BW share, so faster compute doesn't reduce wall-time.

This is a clean confirmation that **the recent CPU optimizations exploit MoE-specific architectural properties** (per-token expert-routing variation that benefits from finer scheduling, sync overhead at MoE expert dispatch barriers, smaller per-thread row-shard tiles in A3B-class active params). The dense/hybrid class doesn't have these properties.

## Closure scope updates

This bundle closes peer-review finding #11 with explicit per-class scope:

- **CPU21 affinity stack (`spread+cores+active`)**: closed scope = "MoE-Q4_K_M class +3-8% deployable; dense/hybrid neutral; NOT a universal win."
- **CPU25 NUMA_MIRROR**: closure unchanged ("DECISIVE NEGATIVE on single-socket NPS4") — now confirmed across both MoE proxies AND dense/hybrid. Hardware is DRAM-channel-bound for ALL architectures tested.
- **CPU2 Q8_0 SIMD**: closure unchanged ("production-ready opt-in, +1-3% at 96t on MoE Q4_K_M") — confirmed neutral on dense/hybrid at 96t (BW-bound both ways).
- **libomp runtime**: closure narrowed to "+6.4% on Coder-30B Q4_K_M MoE specifically; neutral on Qwen3.6-35B Q8 frontdoor MoE, REAP-246B large MoE, and dense/hybrid Qwen3.6-27B Q8_0."

The v5 cherry-pick implication for libomp is unchanged: ship a libomp-built llama-server (universal binary). +6.4% on Coder-30B; neutral on others (including dense). Single-binary audit story remains preferred.

## Commands run

Wrapper: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active taskset -c 0-95 numactl --interleave=all -t 96 -fa 1 -mmp 0 -r 3`.

Builds:
- `build_znver5/`: gcc + libgomp + `-march=znver5` (HEAD `90a17af62`)
- `build_mirror/`: gcc + libgomp + `-march=znver5` + `-DGGML_NUMA_MIRROR=4` (HEAD `90a17af62`)
- `build_libomp/`: clang-20 + libomp + `-march=znver5` (HEAD `29a69599a`)

Model: `/mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf` (qwen35 27B Q8_0, 26.62 GiB, 26.90 B params; hybrid SSM-Dense).

## Files

| File | Purpose |
|---|---|
| `dense_cpu21_baseline.log` | dense baseline (no OMP env) |
| `dense_cpu21_best_stack.log` | dense + CPU21 affinity stack |
| `dense_cpu25_baseline_znver5.log` | dense baseline (CPU21 stack, build_znver5) |
| `dense_cpu25_mirror4.log` | dense + NUMA_MIRROR=4 (CPU21 stack) |
| `dense_cpu2_q8_env0.log` | dense + Q8_0 SIMD env=0 |
| `dense_cpu2_q8_env1.log` | dense + Q8_0 SIMD env=1 |
| `dense_libomp.log` | dense + clang+libomp build |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated cross-architecture deltas |
| `decision.md` | verdict + closure scope updates per track |
