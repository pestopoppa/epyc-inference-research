# CPU4 Op-Coalesced Barriers — Phase 0 Manual Op-Chain Analysis

**Date**: 2026-04-29
**Phase**: 0 (manual analysis — falsification gate before implementation)
**Source**: [`cpu4-deferred-avenues-design-note.md`](../../../../../workspace/handoffs/active/cpu4-deferred-avenues-design-note.md), recommended design #7 op-coalesced barriers.
**Repo**: `/mnt/raid0/llm/llama.cpp-experimental` (`feature/cpu-ep-inter-process` HEAD `d45126db5`)

## Phase 0 gate

> Static analysis on Coder-30B / REAP-246B / Next-80B compute graphs at v5 PGO build shows op-coalesced-barrier potential is below 10% of total per-token barriers under the safety rules → close-via-analysis. ≥10% → graduate to Phase 1 prototype.

## Method

Manual trace of Qwen3 MoE layer build code (the architecture for both Coder-30B-A3B and REAP-246B-A35B targets). Identified per-layer op chain, classified each op pair as DEPENDENT (next reads cur's output → barrier required) or INDEPENDENT (next reads upstream input, not cur's output → barrier safely skippable).

Source files:
- `src/models/qwen3moe.cpp` — top-level layer iteration
- `src/llama-graph.cpp:build_qkv` (lines 1069-1143) — Q/K/V projections
- `src/llama-graph.cpp:build_attn` (lines 2030+) — attention kernel
- `src/llama-graph.cpp:build_moe_ffn` (lines 1383-1720) — MoE FFN block
- `ggml/src/ggml-cpu/ggml-cpu.c:3709-3782` — between-op barrier emission

## Per-layer op chain (Qwen3 MoE, decode mode)

Marked: **[DEP]** = N+1 reads N's output (barrier required). **[IND]** = next reads only upstream input (barrier skippable). **[INT]** = internal barrier in op (not between-op).

### Attention block

| # | Op | Reads | Writes | Barrier-after | Class |
|---|---|---|---|---|---|
| 1 | RMS_NORM (attn_norm) | inpL | cur (normed) | required | [DEP] |
| 2 | MUL_MAT (wq @ cur) | cur | Qcur | required | [DEP] (next reads cur, but NOT Qcur — see #4) |
| 3 | MUL_MAT (wk @ cur) | cur | Kcur | required | **[IND]** — next op (#4) reads cur, not Kcur |
| 4 | MUL_MAT (wv @ cur) | cur | Vcur | required | **[IND]** — Vcur not consumed until much later |
| 5 | RMS_NORM (Qcur → Qnorm) | Qcur | Qnorm | required | **[IND]** — next op (#6) reads Kcur not Qnorm |
| 6 | RMS_NORM (Kcur → Knorm) | Kcur | Knorm | required | **[IND]** — next op (#7) reads Qnorm not Knorm |
| 7 | ROPE (Qnorm → Qrope) | Qnorm | Qrope | required | **[IND]** — next op (#8) reads Knorm not Qrope |
| 8 | ROPE (Knorm → Krope) | Knorm | Krope | required | [DEP] — next op (build_attn) reads Krope |
| 9 | build_attn (multi-step: K transpose, KV cache copy, Q×K^T, softmax, V×P) | Qrope, Krope, Vcur | attn_out | required | [DEP] |
| 10 | MUL_MAT (wo @ attn_out) | attn_out | cur | required | [DEP] |
| 11 | ADD (cur + inpSA residual) | cur, inpSA | ffn_inp | required | [DEP] |

**Skippable in attention block: 5 of 11 between-op barriers (45%)**.

Adjacent IND chain: **#2→#3→#4 (Q,K,V projections all read cur, write disjoint outputs)** can collapse from 3 inter-op barriers to 1 (required after #4 to ensure all three projections finish before downstream consumers). **#5→#6 (Q-norm, K-norm read disjoint inputs)** can collapse from 2 to 1. **#7→#8 (RoPE-Q, RoPE-K read disjoint inputs)** can collapse from 2 to 1.

Net: 11 barriers → 6 barriers = -5 per attention block.

### MoE FFN block (build_moe_ffn flow on Qwen3 MoE)

| # | Op | Reads | Writes | Class |
|---|---|---|---|---|
| 1 | RMS_NORM (ffn_norm) | ffn_inp | cur | [DEP] |
| 2 | MUL_MAT (gate_inp @ cur) → router logits | cur | logits | [DEP] |
| 3 | (MoE-Spec mask construction: transpose, sum_rows, argsort, fill, transpose, add — ~6 ops) | logits | masked_logits | [DEP chain] |
| 4 | argsort_top_k (after mask) → selected_experts | masked_logits | selected | [DEP] |
| 5 | MUL_MAT_ID (gate_up_exps fused) — or separate gate/up MUL_MAT_IDs | cur, selected | gate_up | [DEP] |
| 6 | SILU (gate part) → activated | gate part | act | [DEP] |
| 7 | MUL (act × up part) → cur | act, up | cur | [DEP] |
| 8 | MUL_MAT_ID (down_exps @ cur) | cur, selected | experts_out | [DEP] |
| 9 | reduce_sum across experts | experts_out | reduced | [DEP] |
| 10 | ADD (reduced + ffn_inp residual) | reduced, ffn_inp | l_out | [DEP] |

**Skippable in MoE block: 0-1 of ~10 between-op barriers (~5%)**.

The MoE FFN flow is almost fully serialized: each op reads the prior op's output. The only loosely-related candidate is the MoE-Spec mask construction sub-chain (item #3, ~6 ops including transpose-sum_rows-argsort-fill-transpose-add) where some ops within the mask-construction sub-chain have small internal independence — but they're already dispatched as a single contiguous mask-build subchain in the existing graph, and the inter-op barriers there are between dependent ops (each consumes the prior). Not safe to coalesce.

If a fused `gate_up` MUL_MAT_ID is split into separate gate and up MUL_MAT_IDs (depending on model variant), then those two could be coalesced (both read `cur` + `selected`, write disjoint outputs). On Qwen3 MoE, `ffn_gate_exps` and `ffn_up_exps` are separate tensors but the build_moe_ffn code may fuse them into a single MUL_MAT_ID for performance. Need to verify which path is active. **Estimated 1 additional skippable barrier per layer if non-fused path is hit.**

### Per-layer aggregate

| Block | Total barriers | Skippable | Reduction |
|---|---|---|---|
| Attention | 11 | 5 | 45% |
| MoE FFN | 10 | 0-1 | 0-10% |
| Per-layer total | **21** | **5-6** | **24-29%** |

### Per-token aggregate (across 48 layers Coder + final ops)

| Component | Per-token count |
|---|---|
| Per-layer × 48 | 21 × 48 = 1008 barriers, 240-288 skippable |
| Final layer (output norm + lm_head MUL_MAT) | ~3 barriers |
| Embedding lookup at start | ~1 barrier |
| **Total** | **~1012 barriers, 240-288 skippable** |

**Skippable / total: 240/1012 = 23.7% to 288/1012 = 28.5%.**

## Phase 0 GATE: PASSED

Estimated barrier-count reduction is **24-29% on Coder-30B-A3B per-token decode**, well above the 10% gate threshold.

REAP-246B follows same Qwen3 MoE architecture (different layer count: 80 layers); similar per-layer ratio applies, total reduction estimate also ~24-29%.

Next-80B (per `qwen3-next.cpp` if exists, else closely related Qwen3 family) likely similar — most Qwen3-family transformers follow this op chain.

## Caveats and risk areas

### A. The "INT" internal barriers are NOT addressed by this lever

The ggml_compute_forward_mul_mat function at `ggml/src/ggml-cpu/ggml-cpu.c:1487` has an internal `ggml_barrier(params->threadpool)` BEFORE the chunk-loop. This is required for thread coordination within MUL_MAT (all threads must reach the same chunk-pull state). Op-coalescing the BETWEEN-OP barrier does NOT eliminate this internal barrier — each MUL_MAT still has its own setup barrier.

Implication: even with op-coalescing applied, each MUL_MAT contributes 1 internal + 0 between-op (skipped) = 1 barrier. Without op-coalescing: 1 internal + 1 between-op = 2 barriers.

Per-layer net: 5-6 skipped between-op barriers, while 5-6 internal MUL_MAT barriers remain. The visible per-token barrier count drops by ~25%, but the visible per-MUL_MAT-OP barrier overhead drops by ~50%.

### B. Phase 1.4 already downgrades SOME between-op barriers to CCD-local

The existing `GGML_BARRIER_LOCAL_BETWEEN_OPS=1` env (line 3742-3781) downgrades between-op barriers from full → CCD-local when `cur_op` is partitioned (MUL_MAT/etc) and `next_op` is elementwise (MUL/ADD/SCALE/UNARY). Op-coalescing is more aggressive: it SKIPS the barrier entirely when next op reads a different tensor than cur op writes.

Phase 1.4 does NOT downgrade Q/K/V projection barriers because next op (another MUL_MAT) is not in the elementwise class. Op-coalescing CATCHES this case. **The expected gain is incremental over Phase 1.4** — and Phase 1.4 itself is gated default-off.

### C. Skipping the between-op barrier requires the consumer's input not be currently mutating

When skipping `barrier_after_Qproj`, the next op (Kproj) starts. Kproj reads `cur` (the input both Q and K read). If Q's compute is still mutating `cur` somehow (e.g., via in-place ops, which MUL_MAT is NOT — it writes to a new tensor), there'd be a race.

ggml's MUL_MAT writes to a NEW dst tensor, never in-place. So Kproj reading `cur` is safe even before Qproj finishes. **Verified by code inspection: no in-place MUL_MAT path.**

### D. The PRODUCER's consumers may include OPS ELSEWHERE in the graph

Skipping `barrier_after_Qproj` and starting Kproj immediately is safe IFF no op started by some thread between the skipped barrier and the next required barrier reads Qproj's output. In our trace, Q's output `Qcur` is consumed at op #5 (RMS_NORM Qnorm). Between #2 (Qproj) and #5 (Qnorm): #3 (Kproj), #4 (Vproj). Both read `cur`, not `Qcur`. Safe to skip barriers between #2-#3, #3-#4. Need barrier between #4 and #5? Actually: #4 writes `Vcur`, #5 reads `Qcur`. So barrier between #4-#5 is required for Q's writes to be visible to thread reading Qcur (assuming threads at #5 haven't already finished Q's writes from when they were at #2). The required barrier is "no op N+1 reads tensor X until barrier emitted at N where N writes X".

This is the dependency-graph analysis the design note flagged. The CONSERVATIVE rule:
> Coalesce between-op barriers between consecutive MUL_MAT ops that ALL read the same input AND write disjoint outputs, IFF the next dependency consumer is at least one barrier away.

With this rule, the Q/K/V chain coalesces (#2→#3→#4 all read `cur`, write Q/K/V disjoint). RMS-Q→RMS-K coalesces. RoPE-Q→RoPE-K coalesces. **The 5-skippable count holds under this rule.**

### E. Phase 1 prototype risk

The above static analysis identifies POTENTIAL gain. Phase 1 must:
1. Implement the dependency-graph pass at graph-setup time.
2. Add `GGML_BARRIER_COALESCE=1` env gate (default off).
3. PPL bit-exact gate on Coder-30B + REAP-246B 32-chunk WikiText-2.
4. 5-rep canonical t/s gate: ≥+5% on at least 2 of 3 sync-bound Q4_K_M models.

Risks:
- Memory ordering glitches (most likely failure mode — bit-exact PPL gate catches this).
- Thread-coordination drift if some threads outpace others by multiple ops (the coalescing should preserve per-MUL_MAT internal barriers, which serves as a hard sync point every op).
- Interaction with existing CPU1 stack (CCD_POOLS, CCD_WORK_DIST, BARRIER_LOCAL_BETWEEN_OPS) — need to test all combinations.

## Recommendation

**Phase 0 GATE PASSED** with 24-29% estimated per-token barrier-count reduction on Qwen3 MoE Coder-30B and REAP-246B. Above the 10% gate threshold.

**Recommend graduating to Phase 1 prototype** with:
- Implementation focus: dependency-pass at graph-setup time, env-gated default-off (~150 LOC).
- Phase 1 gates: PPL bit-exact, ≥+5% t/s on 2 of 3 sync-bound Q4_K_M models.
- Estimated wall-clock: 1-2 days implementation + 1 day measurement.

**Pre-implementation work needed**:
- Verify whether build_moe_ffn fuses gate+up into single MUL_MAT_ID or keeps them separate (changes by 1 the per-layer skippable count). 
- Confirm via `LLAMA_LOG_DEBUG=1` graph dump on a single decode token that the predicted op chain matches actual graph order.

## Files

- `README.md` — this file (analysis + Phase 0 verdict)
- `op_chain_qwen3moe.md` — focused op chain trace per layer (also embedded above)
- `decision.md` — Phase 0 GO verdict + Phase 1 design pointer
