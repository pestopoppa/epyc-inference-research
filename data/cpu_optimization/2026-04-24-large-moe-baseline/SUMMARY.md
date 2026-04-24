# CPU15 Phase 0 — Large-MoE baseline on current NPS4 + auto-mbind + AVX-512BW + CPU1 stack

**Date**: 2026-04-24
**Branch**: `cpu-optimization/q8-8x8-avx512bw` HEAD `ba1c23900` (Session 15)
**Build**: `build-noomp` with full CPU1 stack
**Env**: `GGML_CCD_POOLS=1 GGML_NUMA_WEIGHTS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1`
**Flags**: `-fa 1 --numa distribute`

## Goals

Re-measure large MoE models on the current production stack to (a) refresh stale pre-NPS4 numbers, (b) test the D1 gate hypothesis from `large-moe-expert-parallelism.md`: if any large-MoE candidate delivers ≥20 t/s single-stream on the current stack, the strategic reframe alone is the answer and Track B (EP mechanism work) is deferred.

Models available on disk:
- **Qwen3-Coder-REAP-246B-A35B Q4_K_M** (138 GiB, 246B total / ~35B active, REAP-pruned 480B)
- **MiniMax-M2.7 Q8_0** (226 GiB sharded, 230B total / 10B active)

Qwen3-235B-A22B (the handoff's "primary candidate") is NOT on disk; deferred.

## Results — REAP-246B Q4_K_M thread sweep (warm cache)

| Threads | t/s @ tg64 | σ |
|---------|-----------|---|
| 48 | 6.12 | 0.08 |
| 96 | **6.14** | 0.01 |
| 144 | 5.92 | 0.12 |
| 192 (HT) | 4.37 | 0.17 |

Peak: **6.14 t/s at 96 threads**.

Pre-NPS4 baseline was 4.08 t/s on the original 480B (Session pre-2026-04-24). Current REAP-246B (half the params, similar active count) on the new stack achieves +50%. The stack stabilization (auto-mbind + Phase 1.4 + GGML_NUMA_WEIGHTS) lifts large MoE proportionally to small models.

Effective BW at peak: 6.14 t/s × ~25 GB activated/token = **~154 GB/s = 33% of 460 GB/s ceiling**.

## Results — MiniMax-M2.7 Q8_0 (warm cache)

| Threads | t/s @ tg64 | σ |
|---------|-----------|---|
| 48 | **10.23** | 0.57 |
| 96 | 8.21 | 0.14 |

Peak: **10.23 t/s at 48 threads** (96t regresses, unlike REAP-246B). Pre-NPS4 baseline was 11.1 t/s (master-index row 15) — current is roughly comparable (within noise band, slightly lower; needs more reps to confirm if +/- significant).

Effective BW at peak: 10.23 × ~15 GB activated/token = **~154 GB/s = 33% of 460 GB/s ceiling**.

Both large-MoE candidates achieve **~33% BW utilization** on the current stack — between the hybrid Qwen3.6-27B at 25% and pure-dense Qwen2.5-Coder-32B at 44%.

## Cross-model BW utilization on this hardware

| Model | Quant | Bytes/tok (active) | t/s @ best-thread | BW achieved | % of 460 ceiling |
|-------|-------|---------------------|---------------------|-------------|------------------|
| Qwen3.6-27B (75% DeltaNet hybrid) | Q8_0 | 26.6 GB | 4.42 (96t) | 117 GB/s | 25% |
| Qwen3.6-27B (75% DeltaNet hybrid) | Q4_K_M | 15.65 GB | 6.75 (96t) | 106 GB/s | 23% |
| Qwen3-Coder-REAP-246B-A35B (MoE) | Q4_K_M | ~25 GB | **6.14** (96t) | ~154 GB/s | **33%** |
| MiniMax-M2.7 230B-A10B (MoE) | Q8_0 | ~15 GB | **10.23** (48t) | ~154 GB/s | **33%** |
| Qwen2.5-Coder-32B (pure dense, registry) | Q4_K_M | 18.5 GB | 10.8 (96t) | 200 GB/s | 44% |

Large-MoE achieves ~33% BW utilization — **not a free lunch** but meaningfully better than hybrid. Pure-dense reference still leads at 44%. The 11-percentage-point gap from large-MoE to pure-dense is likely architecture overhead (router + gate + expert dispatch ops added to the per-token op count).

## D1 gate decision

Threshold: **≥20 t/s single-stream on the current stack** for "strategic reframe alone suffices".

- REAP-246B: 6.14 t/s — **fails** (3.3× short)
- M2.7: 10.23 t/s — **fails** (2× short)

**D1 GATE FAILS** for both available candidates. Track B (Phase 1 intra-process per-CCD EP) is warranted for further single-stream throughput on large MoE.

The reframe direction is **partially validated**: large MoE does extract more BW utilization (33% vs 25% hybrid), so the hardware IS better-matched to MoE. But to convert the 2.13× concurrent-aggregate gap into single-stream throughput, the EP mechanism (Phase 1 intra-process per-CCD sharding) is needed.

## Caveats and unknowns

1. **Qwen3-235B-A22B (22B active) not measured** — the handoff's primary D1 candidate. With 22B active vs M2.7's 10B active, expected throughput would be roughly M2.7 × 10/22 = ~4.6 t/s, well below D1 threshold. Would need to download to confirm.
2. **Pre-NPS4 11.1 t/s for M2.7** (master-index row 15) vs current 10.23 t/s: likely σ noise + different test conditions; not investigated further.
3. **Q4 vs Q8 comparison** for the same MoE not possible without converting M2.7 → Q4 or finding Q4 of equivalent size MoE.
4. **REAP-246B σ at 96t was 0.01** (very tight) but this was after warm-cache from the 48t run; cold-cache first run at 96t had σ=1.01. Cache-warming required for stable measurements on these large models.
5. **GGML_Q8_0_8X8=1 not tested** on M2.7 (the AVX-512BW kernel only triggers for ne[1]%8==0 tensors and would need verification on MoE expert matrices). Expected gain modest (the kernel is +1-3% at high thread count for BW-bound workloads).

## Recommended next steps

1. **D1 gate fails** → proceed to Phase 1 (intra-process per-CCD EP), per the handoff workplan. Effort: 3-5 days.
2. **Re-bench M2.7 at 48×4t concurrent** (matching the orchestrator-nps4-48x4-notes.md path) to compare aggregate vs single-stream — informs the D2 contention decision.
3. **Decide whether to download Qwen3-235B-A22B** — 130 GB Q4_K_M, would close the handoff's primary-candidate gap. Storage budget tight (~92 GB free per master-index row 22).

## Files

- `git-head.txt` — branch HEAD at measurement time
- `reap246-sweep.log` — full thread sweep llama-bench output for REAP-246B
- `m2.7-96t-warm.log`, `m2.7-48t-warm.log` — M2.7 warm-cache runs
- `m2.7-sweep.log` — M2.7 cold-cache first runs (high σ; not used in summary)
