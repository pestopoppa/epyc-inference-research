# CPU4 Op-Coalesced Barriers — Phase 0 DECISION: GO

## Verdict

**Phase 0 GATE PASSED** — 24-29% estimated per-token barrier-count reduction on Qwen3 MoE Coder-30B-A3B and REAP-246B-A35B targets, well above the 10% gate threshold.

## Summary of evidence

Per-layer op-chain analysis identifies **5 skippable barriers per layer** in the attention block (Q/K/V projections + Q-norm/K-norm + RoPE-Q/RoPE-K — all pairs where adjacent ops read the same input but write disjoint outputs). **0-1 skippable barriers per layer** in the MoE FFN block (mostly serialized).

| Block | Total | Skippable | Reduction |
|---|---|---|---|
| Attention | 11 | 5 | 45% |
| MoE FFN | 10 | 0-1 | 0-10% |
| Per-layer | 21 | 5-6 | **24-29%** |

Per-token (Coder-30B 48 layers + final): ~1012 total barriers, 240-288 skippable.

## Why this is sound

1. **Independence is structural**: in `build_qkv` (src/llama-graph.cpp:1069-1143), Q, K, V projections all read the SAME normed input but write to disjoint output tensors (`Qcur`, `Kcur`, `Vcur`). No data race when their barriers are coalesced.
2. **MUL_MAT internal barriers preserved**: each MUL_MAT op has its own internal barrier at `ggml/src/ggml-cpu/ggml-cpu.c:1487` for thread coordination. Op-coalescing skips only the BETWEEN-OP barrier (line 3770/3773/etc), not the internal one. So thread coordination is preserved.
3. **No in-place mutation**: ggml's MUL_MAT writes to a fresh dst tensor — input `cur` is read-only across Q/K/V projections.
4. **Existing Phase 1.4 (`GGML_BARRIER_LOCAL_BETWEEN_OPS`) does NOT cover this case**: Phase 1.4 downgrades to CCD-local barrier when `cur_op` is partitioned and `next_op` is elementwise. Q→K (both MUL_MAT) is NOT in Phase 1.4's downgrade rule, so falls through to the full global barrier. Op-coalescing CATCHES this case — the gain is incremental over Phase 1.4.

## Recommended Phase 1 scope

**~150 LOC, 1-2 days implementation + 1 day measurement** at `/mnt/raid0/llm/llama.cpp-experimental/ggml/src/ggml-cpu/ggml-cpu.c`:

1. **Graph-setup-time dependency pass** (~80 LOC): walk `cgraph->nodes[]` once at compute setup, mark each node's `coalesce_with_next` based on the conservative rule (consecutive MUL_MAT/MUL_MAT_ID/RMS_NORM/ROPE ops where N+1's input doesn't include N's output, AND no further consumer of N's output exists before the next required barrier).
2. **Compute-loop modification** (~30 LOC): in the per-op iteration at `ggml-cpu.c:3709-3782`, skip the between-op barrier when `cur_node->coalesce_with_next` is true.
3. **Env gate** (~10 LOC): `GGML_BARRIER_COALESCE=1` (default off), with one-time INFO log on enable.
4. **Tests + integration** (~30 LOC): unit-test the dependency pass on a synthetic graph; wire env into existing CPU1-stack tests.

## Phase 1 gates (binding)

- PPL bit-exact 32-chunk WikiText-2 on Coder-30B-A3B Q4_K_M AND REAP-246B-A35B Q4_K_M.
- 5-rep canonical t/s ≥ +5% on at least 2 of 3 sync-bound Q4_K_M models (Coder-30B, Next-80B, REAP-246B). If +10% on all 3, default-on candidate. If neutral, revert.

## Phase 1 risk areas

- Memory-ordering glitches (PPL bit-exact gate is the catcher).
- Thread-coordination drift if optimistic threads outpace by multiple ops (mitigated by preserved MUL_MAT internal barriers as hard sync points).
- Interaction with existing CPU1 stack (CCD_POOLS, CCD_WORK_DIST, BARRIER_LOCAL_BETWEEN_OPS) — must test all combinations.

## Pre-Phase-1 todo

- Verify whether `build_moe_ffn` fuses gate+up into single MUL_MAT_ID (current Qwen3MoE path) or keeps them separate. Adjusts MoE block skippable count by ±1.
- Confirm predicted op chain matches actual graph order via `LLAMA_LOG_DEBUG=1` dump on a single decode token (low risk — code inspection and src/models/qwen3moe.cpp + src/llama-graph.cpp give us the build order definitively, but a runtime check is cheap insurance).

## Closure-inflation discipline

If Phase 1 prototype regresses or is neutral:

> "Op-coalesced-barrier prototype at HEAD <commit> on `feature/cpu-ep-inter-process` measured no t/s gain (or PPL drift) on the production Q4_K_M lineup despite 24-29% predicted barrier-count reduction. Cause: <specific overhead — possibly the per-graph dependency-pass cost, or the savings being absorbed by other bottlenecks like CPU24's compute_kernel_memory_stalled attribution>. Different coalescing rules (more aggressive, e.g. without per-pair dependency check) MAY behave differently but were not tested."

If Phase 1 graduates to default-on:

> "Op-coalesced-barrier `GGML_BARRIER_COALESCE=1` adds +X% on the Q4_K_M sync-bound MoE class with PPL bit-exact. v5+1 cherry-pick candidate."

## Cross-references

- Parent design note: [`cpu4-deferred-avenues-design-note.md`](../../../../../workspace/handoffs/active/cpu4-deferred-avenues-design-note.md)
- Original CPU4 closure: [`cpu-hierarchical-barrier.md`](../../../../../workspace/handoffs/active/cpu-hierarchical-barrier.md) (closed 2026-04-26 for the 2-level CCD-aware barrier variant)
- Sibling track: [`cpu-dynamic-moe-load-balancing.md`](../../../../../workspace/handoffs/active/cpu-dynamic-moe-load-balancing.md) (closed 2026-04-28 for global tile-queue work-stealing variant)
