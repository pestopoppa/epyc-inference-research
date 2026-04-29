# CPU4 Phase 1 — DECISION: NO-GO via test

## Verdict

**Phase 1 GATE NOT MET.** All 3 sync-bound Q4_K_M models show null-or-negative direction at COALESCE=1:

| Model | n_total | Δ_pct | Note |
|---|---|---|---|
| Coder-30B Q4_K_M | 20 reps | -10 to -20% | High variance (CV ~30%) but consistent direction across 3 alternated trials |
| Next-80B Q4_K_M  | 5 reps | -6.2% | Noisy (CV 30%) |
| REAP-246B Q4_K_M | 5 reps | -2.3% | Clean signal (CV <1%) |

Phase 0 had estimated +24-29% reduction in barrier count → expected ≥+5% throughput. Phase 1 measurement shows ZERO models hitting the +5% gate.

## Why Phase 0 was wrong

Phase 0 manual analysis included MUL_MAT/MUL_MAT_ID in the coalescable allowlist. Smoke test at COALESCE=1 with MUL_MAT included produced GARBLED OUTPUT — discovered the shared `params->wdata` buffer hazard (MUL_MAT writes src1 quantization to wdata before its internal barrier; coalescing lets op N+1 clobber wdata while op N's chunk-loop still reads it). This was missed in Phase 0 static analysis.

After excluding MUL_MAT/MUL_MAT_ID for safety, the achievable per-token barrier-count reduction dropped from 24-29% to ~5% (1 skippable barrier per layer in attention block: ROPE-Q → RMS_NORM-K is the only IND pair under safe allowlist; MoE FFN block has 0 skippable). Gain ceiling is structurally below the +5% gate threshold.

## Correctness gate — PASSED

PPL chunk-3 bit-exact on Coder + REAP. The op-coalescing path produces bit-identical output (no silent reordering bugs).

## Operational disposition

- Patch stays in tree disabled-by-default (`GGML_BARRIER_COALESCE=1` env required; default off). Same treatment as slot-promotion dispatcher v1.
- ~80 LOC ggml-cpu.c addition, ENV-gated, costs nothing at default.
- Re-evaluate ONLY if:
  - A different barrier implementation (lower overhead) makes the per-iteration check cost-free
  - A wdata-aware MUL_MAT coalescing variant is designed (would unlock the high-value Q/K/V chain coalescing)
  - A different model architecture with different op-chain shape is benchmarked

## Closure scope (per closure-inflation policy)

> "CPU4 Phase 1 op-coalesced barriers (`GGML_BARRIER_COALESCE=1`, env-gated default-off, ~80 LOC in ggml-cpu.c) at HEAD `d45126db5` on `feature/cpu-ep-inter-process` is structurally net-negative or null on the 3 sync-bound Q4_K_M models under v5 PGO build (Coder-30B-A3B -10 to -20%, Next-80B -6.2% noisy, REAP-246B -2.3% clean). PPL bit-exact verified.
>
> Phase 0's 24-29% estimated reduction was empirically WRONG: it included MUL_MAT/MUL_MAT_ID in the coalescable allowlist without checking the shared `params->wdata` buffer hazard. With the wdata-aware safe allowlist (excludes MUL_MAT/MUL_MAT_ID), the achievable per-token barrier-count reduction drops to ~5% (1 skippable barrier per layer in attention block under conservative independence rule). The mechanism's gain ceiling is structurally bounded below the gate threshold.
>
> Does NOT generalize to: (a) architectures with different op chains (dense, hybrid SSM, attention-only); (b) future ggml graph rewrites that fuse Q/K/V into non-wdata-sharing paths; (c) prefill workloads with different op-chain shape; (d) different barrier implementations where the skip-savings might exceed per-iteration check overhead.
>
> Operational disposition: code in tree disabled-by-default. The other 5 untested CPU4 deferred avenues remain open in the design note (token-to-expert rebalance, hybrid static+dynamic spillover, cross-CCD work migration, MoE quant layout rebalance, AND a new variant: wdata-aware MUL_MAT coalescing where each op gets its own wdata segment)."

## Lesson for future Phase 0 analyses

**Phase 0 manual op-chain analyses MUST check buffer-sharing constraints**, not just dependency-graph (src/dst) independence. Specifically:

- Does cur op write to a shared mutable buffer (params->wdata, params->wsize, threadpool state)?
- Does next op read from the same buffer?
- If yes, coalescing is unsafe even if direct src/dst dependency is absent.

This was missed in the original Phase 0 design note. Future analyses should add this gate.

## Cross-references

- Phase 0 (now superseded): [`../2026-04-29-cpu4-op-coalesced-barriers-phase0/decision.md`](../2026-04-29-cpu4-op-coalesced-barriers-phase0/decision.md) — GO based on incomplete analysis
- Design note: [`cpu4-deferred-avenues-design-note.md`](../../../../../workspace/handoffs/active/cpu4-deferred-avenues-design-note.md) — design #7 now TESTED-AND-FAILED
- Parent CPU4: [`cpu-hierarchical-barrier.md`](../../../../../workspace/handoffs/active/cpu-hierarchical-barrier.md) — original 2-level barrier (also closed negative 2026-04-26)
