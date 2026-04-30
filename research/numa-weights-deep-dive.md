# NUMA-Aware Weight Placement on EPYC 9655 NPS4 — Research Deep-Dive

**Status**: ARCHIVED LEARNING (post-strip from llama.cpp v5 branch)
**Created**: 2026-04-30
**Triggering decision**: v5 cleanup audit Q3 — STRIP deprecated GGML_NUMA_WEIGHTS family; preserve as 1-page research record before deletion.

This note compresses ~3 weeks of CPU1 Phase 1.x work on weight placement into a reusable lesson, so the design history isn't lost when the implementation is git-history-only.

## Problem statement

Under canonical EPYC 9655 NPS4 inference (96 threads, `numactl --interleave=all`, `--mmap 0`), MoE expert weights are interleaved across 4 NUMA nodes at 4 KB-page granularity. For any single dense matmul, ~75% of weight reads cross at least one Infinity Fabric hop. Decode is bandwidth-bound; cross-node reads pay both higher latency AND fewer concurrent in-flight bytes per node.

Hypothesis: explicit NUMA-aware placement (matching weight pages to the CCD that will read them) should outperform random page-interleave by reducing fabric traffic.

## Mechanisms tried

| Mechanism | Approach | Outcome |
|---|---|---|
| **GGML_NUMA_WEIGHTS=interleave (Phase 1.3 v1)** | Process-wide `set_mempolicy(MPOL_INTERLEAVE)` BEFORE mmap | DEPRECATED. Process-wide policy leaked into KV cache, threadpool stacks, intermediate buffers — wildly variable performance (±13-22 t/s std on Coder-30B Q4_K_M; -38% mean on Qwen3.6-35B Q8_0). Process-wide leak was the killer. |
| **GGML_NUMA_WEIGHTS=interleave (Phase 1.3 fix 2026-04-26)** | Per-region `mbind(MPOL_INTERLEAVE)` AFTER mmap | UNSTABLE. Scopes the policy correctly but kernel readahead bypasses mbind (placement uses readahead-thread's node), and on shared-file-cache hosts the page-cache may already hold pages from earlier loads under conflicting policies. Variance reduced but mean still degrades on Q8_0. |
| **GGML_NUMA_WEIGHTS=local (Phase 1.3 v2)** | No global policy + per-CCD warmup pass with first-touch placement | NET-NEUTRAL alone. Required pairing with GGML_CCD_WORK_DIST=1 to actually exploit the placement. Pairing showed marginal win (+1-2%) on some workloads, but the warmup adds ~30-60 sec to model load and the runtime gain didn't justify it for production. |
| **GGML_NUMA_REPLICATE (Lever A')** | 4× per-node anonymous-mmap replicas of the full model | NET-NEGATIVE on shared-budget hosts. 4× memory blowup means a 17 GB Q4_K_M model becomes 68 GB extra (in addition to file mmap). On a 1.1 TB machine running multiple model instances simultaneously, this evicts page cache for OTHER models, hurting cross-role throughput. Only viable for single-model deployments. |
| **GGML_EXPERT_ANON_COPIES (CPU15 Phase 2)** | Per-MoE-expert per-node anon-mmap, copying expert bytes after first-touch | NET-NEGATIVE in interaction. Conceptually elegant — ~131 GiB expert weights × 1.0 vs replication's 4× — but mul_mat_id dispatch overhead increases (per-expert lookup in tensor info registry) and cross-CCD coordination at the expert dispatch site hurts when expert affinity doesn't match the active token's chosen experts. |
| **GGML_EXPERT_CCD_LAYOUT (CPU15 Phase 1b)** | Per-expert mbind on file mmap pages, no copy | NEAR-NULL. mbind() can't reliably move file-backed cached pages without `CAP_SYS_NICE`. Even with the capability, the kernel's readahead path operates outside the per-region policy, so placement drifts. |

## Why all mechanisms failed (root cause)

**Three converging factors:**

1. **Default `numactl --interleave=all` already gets ~92% of the achievable BW.** EPYC 9655 NPS4 has 4 nodes × 3 DDR5 channels × ~38 GB/s = ~460 GB/s aggregate. Page-granular interleave hits ~420 GB/s in measurement, and the remaining ~40 GB/s gap is mostly dot-product ALU stalls, not fabric latency. The "75% cross-node reads pay" framing was technically correct but the magnitude was small relative to what the CPU was actually waiting on (DRAM access latency).

2. **Per-token expert affinity is non-stationary.** MoE routing selects different experts for different tokens. Pinning expert E to CCD K helps when K's threads are evaluating E, hurts otherwise. Static affinity averages worse than uniform interleave when token expert distribution is uniform.

3. **Mechanism cost is not zero.** Every approach added either model-load time (warmup pass), memory (replication), runtime overhead (per-expert lookup), or correctness risk (mbind on cached file pages). On a hardware platform that's already DRAM-channel-bound at NPS4, even a 1-2% mechanism overhead can erase any locality gain.

## Lessons for future hardware

| Hypothesis | When it might pay |
|---|---|
| Multi-socket (2× EPYC, NPS8) | Cross-socket fabric latency dominates, locality matters more, NUMA-aware placement may hit ≥10% gains |
| L3-as-NUMA BIOS setting | More NUMA nodes (CCD-level) means smaller per-node share; locality concept shifts to CCD-level (what CPU1 stack already does) |
| Future HBM EPYC variants | If memory BW headroom exists, locality games matter less; CPU1 stack may suffice |
| Mass-MoE workloads (256+ experts) | Sparser expert affinity makes static pinning more valid; warmup cost amortizes over many tokens per token |

## What WAS preserved into v5

- **CPU1 Phase 1.0 + 1.1**: per-CCD 2-level barrier + CCD-aware cpumask. Validated +1.8% on Coder-30B Q4_K_M (2026-04-26 P3 isolation). Stable. Default-OFF in v5; orchestrator wires on for `worker` role only.
- **CPU1 Phase 1.2 + 1.4 (`GGML_CCD_WORK_DIST` + `GGML_BARRIER_LOCAL_BETWEEN_OPS`)**: per-CCD work distribution + CCD-local between-op barrier. Stable.
- **CPU2 mbind kill-switch (`GGML_NUMA_REPACK_INTERLEAVE`, default-on)**: NOT a NUMA_WEIGHTS mechanism — operates on CPU_REPACK heap buffers (allocator-time analog to mmap-time NUMA hints). +6% AND stabilizing on Q8_0 MoE; safe with explicit kill-switch.

## Code references (git history pre-v5 cleanup audit)

For readers tracing the original implementation:
- Original `set_mempolicy` approach: commit `e249ed5f1` (CPU1 Phase 1.3 v1)
- Per-region `mbind` fix: commit `ed77d5220` (CPU1 P1.3 fix per-region)
- Phase 1.3 v2 'local' mode: commits `88d3d6dc5` + `6efe765f9`
- Lever A' replication: commit `1fcc16d39`
- CPU15 Phase 1b expert-ccd-layout: in `llama-model-loader.cpp` 1497-end (now soft-stripped)
- CPU15 Phase 2 anon-copies: in `llama-model-loader.cpp` 49-91 + 1737-1912 (now hard-stripped)

Bundle paths for measurement evidence:
- `data/cpu_optimization/2026-04-26-cpu1-p3-isolation/` — CPU1 NUMA_WEIGHTS instability isolation
- `data/cpu_optimization/2026-04-26-asymmetry/` — Q8 frontdoor finding (motivated CPU15 Phase 3.2 inter-process EP as the deployable answer)
- `data/cpu_optimization/2026-04-{19,20,21}-cpu15-phase{1,2}*/` — CPU15 Phase 1+2 superseded measurements

## Cross-references

- v5 cleanup audit Q3 decision: `handoffs/active/v5-push-cleanup-audit.md` §14 row "GGML_NUMA_WEIGHTS"
- Closure-via-test memory: `feedback_canonical_baseline_protocol.md`
- Generalization (don't extrapolate from one falsified mechanism): `feedback_closure_inflation.md`
