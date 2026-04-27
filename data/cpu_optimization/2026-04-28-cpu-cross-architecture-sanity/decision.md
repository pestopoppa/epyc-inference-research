# Cross-Architecture Sanity Coverage — Decision

**Verdict**: **CLOSED** — peer-review finding #11 (MoE-only test gap) addressed.

## What was decided

All four already-validated CPU optimizations were measured on dense/hybrid Qwen3.6-27B Q8_0 (the only non-MoE production model in the lineup). All four are **NEUTRAL** on dense (within run-to-run noise):

| Optimization | dense Δ | MoE Δ (reference) |
|---|---|---|
| CPU21 affinity stack (`spread+cores+active`) | +0.2% | +6.2% on Coder-30B Q4_K_M, +3-8% across MoE class |
| CPU25 NUMA_MIRROR=4 | -1.3% | -1.0% on Coder-30B Q4_K_M (DECISIVE NEGATIVE on this hardware) |
| CPU2 Q8_0 8x8 AVX-512BW SIMD | -1.0% | +1-3% @ 96t on Q8_0 MoE (BW-saturated ceiling) |
| libomp runtime (clang+libomp+znver5) | -1.7% | **+6.4%** on Coder-30B Q4_K_M (apples-to-apples vs gcc+libgomp+znver5) |

## Why

The dense/hybrid class is **uniformly DRAM-BW-bound at 96 threads**:
- 26 GB Q8 weights / 96 threads ≈ 0.27 GB/thread; per-thread BW share = 4.79 GB/s/thread → decode is pure DRAM streaming.
- All threads do uniform work (no MoE expert-routing variation), so static partitioning is already optimal. No scheduler/runtime overhead to reduce.
- No barrier/sync amplification: with uniform compute, threads complete row-shards in lock-step.

The CPU optimizations exploit MoE-specific architectural properties:
- CPU21 affinity stack: helps when there's per-token routing variation that benefits from cache-warm spread.
- CPU25 NUMA_MIRROR: would help if fabric were the binding constraint (it isn't on single-socket NPS4 — confirmed across both architectures).
- CPU2 Q8_0 SIMD: helps at low thread counts where compute is the binding constraint, not BW.
- libomp: helps when barrier overhead and dynamic task balancing matter — i.e., on smaller-active-param MoE classes (Coder-30B-A3B's thinner per-thread tiles).

## Closure scope updates per track

- **CPU21**: closure scope = "MoE-Q4_K_M class +3-8% deployable; dense/hybrid neutral; **NOT a universal win**". Still production-ship-worthy because dense is still on the canonical baseline; the MoE-class wins are positive net to the system.
- **CPU25 NUMA_MIRROR**: closure unchanged ("DECISIVE NEGATIVE on single-socket NPS4") — now confirmed across both MoE proxies AND dense/hybrid. Hardware is DRAM-channel-bound for ALL architectures tested. Reopen still requires 2-socket configuration.
- **CPU2 Q8_0 SIMD**: closure unchanged (kernel correctness via PPL bit-exact; throughput +1-3% @ 96t MoE Q8_0; neutral on dense at 96t). Production-ready opt-in regardless.
- **libomp runtime**: closure narrowed to "+6.4% on Coder-30B Q4_K_M MoE specifically; neutral on Qwen3.6-35B Q8 frontdoor MoE, REAP-246B large MoE, and Qwen3.6-27B Q8 dense/hybrid". v5 cherry-pick implication: ship libomp-built llama-server (universal binary). +6.4% on Coder-30B; neutral elsewhere.

## What this does NOT close

- **CPU22 work-stealing** (Phase 3 of remediation): still pending. Note that dense doesn't trigger `mul_mat_id`, so dense can't benefit from work-stealing; this is correctly scoped to MoE in the handoff.

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.6 (this bundle, COMPLETE).

## Closure-inflation lesson

This Phase 2.6 confirmation prevents a future "all software CPU optimizations are exhausted on dense" claim from being inflated to "all software CPU optimizations are exhausted on this hardware". The dense/hybrid result IS that CPU21 + CPU2 + libomp + NUMA_MIRROR don't help dense — but that's because dense is BW-bound and the optimizations exploit MoE-specific properties. The closures stand for the MoE classes; dense gets nothing from these specific levers but isn't necessarily exhausted of all CPU optimization potential (e.g., a different lever like per-thread BW reservation, NUMA-aware tensor splitting, or a smaller working set could conceivably help). Future dense-specific optimization is left explicitly open.
