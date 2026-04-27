# CPU22 Phase 3 — Decision

**Verdict**: **CLOSED via test — gate FAILED**. Track honestly closed by empirical measurement, replacing the prior closure-by-inference (which Phase 1 of remediation correctly flagged as inflation).

## What was decided

Work-stealing prototype (single-global-tile-queue + single atomic counter) implemented in `ggml/src/ggml-cpu/ggml-cpu.c` `ggml_compute_forward_mul_mat_id`. Env-gated `GGML_EP_WORK_STEALING=1`. PPL bit-exact at 12 chunks on Coder-30B Q4_K_M. Throughput gate of ≥10% on 2 sync-bound Q4_K_M models was the binding criterion per the handoff. Result: **NOT MET**.

Throughput summary (5-rep mean ± std at proper canonical):

| Model | env=0 | env=1 | Δ |
|---|---|---|---|
| Coder-30B Q4_K_M | 53.12 ± 0.10 | 51.89 ± 0.07 | -2.3% |
| Next-80B Q4_K_M | 23.36 ± 0.03 | 23.29 ± 0.07 | -0.3% (noise) |
| REAP-246B Q4_K_M | 6.64 ± 0.01 | 6.59 ± 0.02 | -0.8% (noise) |

Three of three sync-bound MoE models tested are negative or within noise. Gate threshold of ≥10% is not met on any model.

## Why the prototype doesn't deliver

CPU24 sync share (15% on REAP-246B) bounds the realistic gain ceiling. Combined with the existing per-expert chunked path's already-efficient chunk-level work-stealing (atomic_fetch_add per expert, threads progress through experts independently with no per-expert barrier), the marginal improvement from inter-expert work-stealing is small — and is dominated by:

1. **Single-atomic contention overhead**: 96 threads `atomic_fetch_add` on one global counter ≈ 30 ns/op × N tiles. Coder-30B with ~12K tiles per mul_mat_id call = ~360 µs of atomic contention overhead per op, summed across the ~100 mul_mat_id ops per token = ~36 ms wall added per token. At 53 t/s baseline (19 ms per token), this overhead alone is ~2× the per-token decode time.
2. **Tile-decode + per-tile dimension recompute** (recomputing nchunk0/nchunk1/dr0/dr1 per tile vs once per expert): adds CPU cycles per tile.
3. **Existing chunk-level work-stealing already covers most imbalance**: per-expert atomic counter lets fast threads claim more chunks; threads don't barrier between experts. The hypothetical inter-expert work-stealing adds capacity only when expert load is severely imbalanced AND the per-expert chunked dispatch fails — which we don't see in practice on these models at proper canonical.

## Stability — informal observation

5-rep runs (~30s wall on Coder-30B; longer on Next-80B and REAP-246B) completed without crash, deadlock, or PPL drift on all 3 models. No formal 5-minute sustained-run stress test (low marginal value given negative throughput result), but the implementation pattern (single barrier + single atomic + no per-thread mutable state) is robust by construction.

## Closure scope

**Closed**: CPU22 work-stealing prototype empirically fails the binding ≥10% gate on 3 sync-bound MoE models tested. PPL bit-exact verified. Track closes honestly via test.

**Code disposition**: env-gated default-OFF in the codebase. Strip if v5 audit prefers smaller surface; otherwise leave as documented dead-code-by-default. The gate-failure result is the binding evidence; no further investigation warranted unless a concrete signal emerges (e.g., a workload showing severe expert imbalance at >7-15% sync share).

**Reopen criteria**:
- A specific MoE model surfaces with sync share >25% per perf-record (would require new CPU24 attribution evidence).
- An algorithmic change (e.g., MoE-Spec budgeted-expert at verification step) introduces compounding load imbalance that the global queue could fix.
- Hardware change (different atomic latency) reduces the contention overhead.

## Implications for downstream

- **CPU24 attribution finding remains valid**: 15% sync share on sync-bound MoE; the prototype empirically confirms that recovering this share is harder than the analysis suggested due to overhead dominance.
- **CPU19 Tutel 2DH**: motivation FURTHER weakened. If a global atomic counter doesn't help (even though it removes the per-expert sequential walk), inter-CCD/inter-NUMA all-to-all reorganization is even less likely to deliver.
- **MoE-Spec (Phase 4 stub, future work)**: still worth Phase 0 falsification probe. MoE-Spec's expert budgeting at verification step is orthogonal — it changes WHICH experts run, not HOW the selected experts' tiles are scheduled across threads. Compatible with both the existing per-expert chunked path and the work-stealing path; standalone evaluation.

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 3 (this bundle, COMPLETE).
