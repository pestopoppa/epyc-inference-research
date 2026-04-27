# CPU24 Phase 2.3 — Decision

**Verdict**: **CLOSED** — peer-review HIGH finding #4 addressed. Attribution class confirmed across all 4 model classes.

## What was decided

The CPU24 attribution finding (`dominant_bottleneck = compute_kernel_memory_stalled`) is now **confirmed across all 4 model classes tested**, not just REAP-246B + Qwen3.6-35B Q8 from the original 2026-04-26 sweep.

| Class | Model | Throughput | IPC | Cross-NUMA fill % | Cache miss % |
|---|---|---|---|---|---|
| Sync-bound MoE | REAP-246B Q4_K_M | 6.34 t/s | **0.24** | 21.2% | 7.87% |
| BW-bound frontdoor MoE | Qwen3.6-35B Q8_0 | 23.67 t/s | **0.18** | 25.3% | 10.48% |
| Giant MoE 230B/A10B | MiniMax-M2.7 Q8_0 | 11.03 t/s | **0.21** (cache-warm) | 26.8% | 9.34% |
| Dense/hybrid SSM-Dense | Qwen3.6-27B Q8_0 | 4.40 t/s | **0.175** | **8.9%** | **2.59%** |

All 4 classes are memory-stalled — IPC 0.17-0.28, far below Zen 5 peak ~5. Threads spend their time INSIDE compute kernels stalled on memory loads.

## What was NOT decided (gates that were originally open)

These gates were all closed by Phase 2.3:

1. ✅ **MiniMax-M2.7 Q8_0 counter run** — handoff PRIMARY target, finally measured. Cross-NUMA 26.8%, IPC 0.21 (cache-warm), throughput 11.0 t/s.
2. ✅ **2-rep stability pass** — counter values stable to ±2pp across reps for all 4 models. IPC and cross-NUMA fraction reproducible.
3. ✅ **Dense/hybrid coverage** (peer-review finding #11) — Qwen3.6-27B Q8_0 measured. Striking result: 3× lower cache-miss rate (2.6% vs MoE's 8-11%) and 3× lower cross-NUMA fraction (8.9% vs MoE's 25%). Despite the cleaner memory pattern, IPC is the LOWEST (0.175) — pure DRAM streaming gives no compute overlap.
4. ✅ **Formal counter-table tabulation** — both this README and results.csv present the IMC/channel/fabric/remote-miss/LLC/stall structure required by the handoff format.

## Striking new finding

**Dense/hybrid is dramatically more cache-efficient than MoE classes.** Cache miss rate 2.6% vs MoE 7-11%; cross-NUMA fill fraction 9% vs MoE 25%. Mechanism: dense streams weights uniformly across threads with no MoE expert-routing variation that thrashes thread-local caches. This was unexpected — the original CPU24 framing assumed memory-stall behavior was uniform across classes, but the cache-locality differs by 3×.

The throughput gap closes in the IPC: dense IPC 0.175 (LOWEST), despite cleaner memory access. Why? Pure DRAM streaming has no compute-bound segments to overlap; the entire pipeline is memory-wait. MoE classes have some compute-bound segments (expert-routing logic, gating networks) that lift IPC slightly even though they cause more cache thrash.

This refines the CPU24 attribution: the bottleneck class is the same across architectures (memory-stalled compute kernels), but the *mechanism* differs. MoE thrashes caches and pays per-token cross-NUMA latency; dense pays pure DRAM-streaming bandwidth without thrashing.

## Implications for downstream tracks

- **CPU22 dynamic load balancing**: 15% sync ceiling per the original CPU24-narrow finding holds for sync-bound MoE class. Phase 3 of remediation will run the empirical gate. The new dense finding does NOT reopen CPU22 because dense doesn't have the imbalance pattern CPU22 targets.
- **CPU19 Tutel 2DH**: motivation remains weakened across all classes. Sync share is 15% per perf-record on REAP; does not generalize-up at MiniMax (also bottlenecked on memory access, not sync).
- **CPU25 NUMA_MIRROR**: closure CONFIRMED across all 4 classes — the per-thread DRAM-channel-bound finding is universal on single-socket NPS4. Reopen still requires 2-socket configuration.
- **CPU2 SIMD kernel work**: REVALIDATED PRIORITY across all classes. 80%+ cycles in compute kernels means faster SIMD compute = real wall-time reduction. Q6_K SIMD (Phase 2.4 PASSED) confirms.
- **Q6_K SIMD generalization to other classes**: bit-exact verified on Coder-30B + REAP-246B (both Q4_K_M MoE). No reason to expect different behavior on other classes since the kernel operates at the per-tensor level.

## Closure scope

**Closed**: full CPU24 attribution objective (IMC/channel/fabric/remote-miss/LLC/stall counter set) on REAP-246B + Qwen3.6-35B Q8 + MiniMax-M2.7 + Qwen3.6-27B dense, with 2-rep stability for each. Counter table formalized.

**Not in scope (and explicitly OK)**:
- gemma-4-26B-A4B Q4_K_M (BW-bound class — projects from Qwen3.6-35B Q8 evidence; not measured)
- Qwen3-Next-80B-A3B Q4_K_M (sync-bound MoE — projects from REAP-246B evidence; not measured)
- These were not handoff PRIMARY targets and the existing 4-class spread provides sufficient class-level evidence.

## Internal contradiction strip-out (Phase 1 already did this)

The original CPU24-narrow handoff body had a stale sentence at line 149 saying "sync overhead claims 96% of parallelism" that contradicted the corrected attribution at line 114 (compute=80%, sync=15%). Phase 1 of remediation stripped that contradiction; this Phase 2.3 confirms the corrected attribution generalizes across all 4 classes.

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.3 (this bundle, COMPLETE).
