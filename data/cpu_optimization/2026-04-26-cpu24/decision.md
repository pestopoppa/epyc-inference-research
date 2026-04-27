# CPU24 — Decision

**Verdict**: **PARTIAL** (REAP + Qwen3.6-35B Q8 attribution corrected; MiniMax-M2.7 + dense/hybrid + 2-rep stability + formal counter table format remain).

## What was decided on 2026-04-26 evening

**Attribution class for REAP-246B Q4_K_M @ 96t (proper canonical)**:

> `dominant_bottleneck = compute_kernel (memory-stalled INSIDE compute path)`

Evidence (from raw logs in this bundle):
1. **80% of cycles in compute kernels** (perf-record): `gemv_q4_K_8x8_q8_K` 64.37%, `vec_dot_q6_K_q8_K` 15.64%.
2. **15% of cycles in libgomp internal sync** (perf-record offset 0x26580). Real sync overhead but secondary.
3. **IPC = 0.39** (perf-stat) — kernels memory-stalled, not compute-saturated.
4. **20% cross-NUMA fills** (`(dram_io_far + remote_cache) / total_fills` = 904M / 4636M; with `numactl --interleave=all` this is the expected uniform-distribution rate, NOT abnormal cross-NUMA traffic).
5. **26% aggregate DRAM bandwidth used** (~118 GB/s of 460 GB/s) — NOT system-saturated; per-thread BW share is the bottleneck (4.79 GB/s/thread).
6. **Scaling efficiency at 96t**: 4.27× vs ideal 96× (single-thread = 1.41 t/s, 96-thread = 6.01 t/s).

**Original "sync_imbalance" attribution was WRONG.** It was based on the 4.27× scaling efficiency interpreted as 96% sync overhead — a misreading because perf-record reveals threads spend their time INSIDE the compute kernels (memory-stalled), not waiting at barriers.

## What was NOT decided (gates that remain open)

The CPU24 handoff Objective (line 11-12, 27-29) lists these as primary targets and mandatory protocol pieces:

- **MiniMax-M2.7 Q8_0 counter run**: NOT RUN. The 2026-04-26 evening session ran Qwen3.6-35B Q8_0 instead (stated as "Q8 comparison" in handoff body); MiniMax was never executed.
- **2-rep stability pass on REAP + Qwen3.6-35B**: NOT RUN. Single rep each.
- **Formal counter table per the handoff format** (IMC/channel, fabric, remote miss, LLC, stall class as discrete columns): NOT TABULATED. The raw perf-stat logs contain the data; this bundle's `results.csv` formalizes it for REAP + Q8 only.
- **Dense/hybrid (Qwen3.5/3.6-27B) counter run**: NOT RUN. The IPC=0.39 / compute-kernel-memory-stalled finding is stated in MoE-only terms; underlying mechanism (per-thread BW contention) is architecture-independent.

## Internal contradiction in handoff body (2026-04-27 evening fix)

The corrected attribution at line 114 ("compute_kernel memory-stalled INSIDE compute path") and the supporting line 125 ("80% of cycles are in compute kernels; only 15% in OpenMP sync") were directly contradicted by line 149 stating "sync overhead claims 96% of parallelism, not bandwidth saturation, not memory placement". Line 149 was leftover framing from the pre-perf-record draft (when the sync-imbalance hypothesis was active). The remediation Phase 1 stripped that contradictory sentence.

## Closure scope

**Closed**: REAP-246B Q4_K_M attribution at proper canonical = `dominant_bottleneck = compute_kernel_memory_stalled`. This finding is the basis for:
- CPU22 ceiling estimate (sync share 15% → dynamic balancing best-case ~7%).
- CPU19 deprioritization (Tutel 2DH addresses sync, sync is only 15%).
- CPU25 NUMA_MIRROR motivation (per-thread BW = 4.79 GB/s; later refuted at Phase 2 gate because hardware is DRAM-channel-bound, not fabric-bound — see `numa-mirror-integration.md`).

**NOT closed**: MiniMax-M2.7 attribution (handoff primary target). Dense/hybrid generalization (finding #11). 2-rep stability pass (handoff protocol requirement). Formal counter-table extraction.

## Implications for downstream tracks

These derive from the closed REAP attribution and remain valid pending MiniMax + dense confirmation:

- **CPU19 Tutel 2DH**: motivation weakened. Sync share is 15%; even halving it is at most ~7-8%. Keep stub for archival; do NOT pursue without new evidence (e.g., a workload where MiniMax counter reveals different sync share).
- **CPU22 dynamic load balancing**: re-scoped. Gain ceiling ≈ sync share (15%). Phase 3 of remediation runs the empirical gate (≥10% on 2 sync-bound models). If positive, balancing redirects sync time into productive compute. If null/negative, track honestly closes via test.
- **CPU21 OpenMP runtime matrix**: COMPLETE for libgomp affinity submatrix; +3-8% deployable. Consistent with sync being secondary (15%) vs compute-stall (80%).
- **CPU2 SIMD kernel work**: REVALIDATED PRIORITY. 80% of cycles ARE in compute kernels (gemv + vec_dot), faster SIMD = real wall-time reduction. Q6_K + Q5_K extensions directly attack the dominant cycle consumer.
- **CPU25 NUMA_MIRROR**: Phase 2 throughput gate FAILED (-1.0% Coder-30B, +0.6% Qwen3.6-35B Q8) because hardware is DRAM-channel-bound, not fabric-bound. CPU24 perf-record could not distinguish fabric-stall from DRAM-channel-stall; CPU25 cleanly ruled out the fabric-stall hypothesis.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.3:
- MiniMax-M2.7 Q8_0 perf-stat counter run at proper canonical.
- Qwen3.5/3.6-27B Q8_0 dense/hybrid counter run.
- 2-rep stability pass on REAP + Qwen3.6-35B + MiniMax + dense.
- Formal counter table (IMC/channel, fabric, remote miss, LLC, stall class) for all four models.
- decision.md per model class.

Output dir: `2026-04-28-cpu24-minimax-and-dense/`.
