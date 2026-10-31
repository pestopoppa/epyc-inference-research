# GDN-2 b_proj / w_proj — Low-Rank Factorization Static Analysis

**Task:** Z12 (zero-compute) — can GDN-2's `b_proj` / `w_proj` be low-rank factorized?
**Date:** 2026-08-25 · **Era:** E9 (claim grammar per `agents/shared/MEASUREMENT_POLICY.md`)
**Owner of record:** `handoffs/active/multiscreen-attention-evaluation.md` (Z12), informs `B9` in
`handoffs/active/log-linear-gated-deltanet-readiness.md`.
**Compute:** none (static analysis; no inference, no GPU, no model load).
**Primary source citations:** `intake-1281#record` (record-level discussion — the record carries the
verified Appendix C.1 projection-shape claim and the +12.5 % derived figure quoted below). Gated by
`scripts/vidya/cli.py cite-check --as-of 2026-08-25` (no blocking verdicts).

---

## 1. Verdict (one paragraph)

**Yes — both projections can be low-rank factorized, and the +12.5 % active-parameter objection to
GDN-2 does NOT survive factorization.** The two projections the paper adds are
`d_model → H·d_k` (`b_proj`, erase gate) and `d_model → H_v·d_v` (`w_proj`, write gate)
(Appendix C.1, verified in `intake-1281#record`). Factorizing each as `A: d_model → r` followed by `B: r → H·d`
(dropping to rank `r`) cuts the added always-active parameter cost by `~9.6×` at
`r = 128` on our production geometry — from **~377.5 M (~+12.6 % of the A3B active budget) to
~39.3 M (~+1.3 %)** — bringing the gate overhead into the same class as the layer's own
already-factorized `f_proj`/`g_proj`. Precedent is both in-tree (Kimi Linear's KDA forget-gate
factorization `LLM_TENSOR_SSM_F_A` / `SSM_F_B`) and in the reference implementation itself (fla's
`GatedDeltaNet2` already ships `f_proj` and `g_proj` as low-rank bottlenecks; only `b_proj`/`w_proj`
remain full-rank, with no justification in the paper). Whether a rank-128 gate *retains the reported
gain* is a training-time question that stays compute-gated with whoever runs `B9(b)` — this note
does not claim it.

## 2. Geometry (verified, production artifact)

Source: GGUF header of `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` (read 2026-08-25) +
frozen-tree loader `llama.cpp/src/models/qwen35moe.cpp` @ `0db32c06e3e5` (read-only).

| Quantity | Value | Source |
|---|---|---|
| `d_model` (embedding_length) | 2048 | GGUF `qwen35moe.embedding_length` |
| Blocks | 41 | GGUF `qwen35moe.block_count` |
| GDN layers | **30** | `ssm_*` tensors in blocks {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38}; remaining 11 are full attention |
| `H_k` (ssm K heads) | 16 | loader: `n_k_heads = ssm_n_group` = 16 |
| `H_v` (ssm V heads) | 32 | loader: `n_v_heads = ssm_dt_rank` = 32 |
| `d_k` (= `d_v`) | 128 | loader: `head_k_dim = head_v_dim = ssm_d_state` = 128 |
| `key_dim = H_k·d_k` | 2048 | 16 × 128 |
| `value_dim = H_v·d_v` | 4096 | 32 × 128 |
| A3B active budget | ≈ 3.0 B | model class (35 B total, A3B active) |

The GDN block in this model is a **GVA (Grouped Value Attention)** configuration: 16 QK heads
broadcast to 32 value heads — exactly the `num_v_heads > num_heads` case the fla reference layer
handles (`g = num_v_heads // num_heads`).

## 3. Parameter arithmetic — full-rank (the +12.5 % objection, quantified)

GDN-2's two new projections (paper Appendix C.1, verified in `intake-1281#record`):
erase gate `b_proj: d_model → H·d_k`; write gate `w_proj: d_model → H_v·d_v`.

| Tensor | Shape | Params |
|---|---|---|
| `b_proj` | 2048 → 2048 | 4,194,304 |
| `w_proj` | 2048 → 4096 | 8,388,608 |
| **per GDN layer** | | **12,582,912 (~12.58 M)** |
| **× 30 layers** | | **377,487,360 (~377.5 M)** |

Against the ≈ 3.0 B active budget: **~+12.6 % always-active dense parameters** — this is the
"+12.5 %" of `intake-1281#record` (derived there as +12.52 M/layer, +375.5 M total; this note's
exact arithmetic reconciles at rounding). At decode, weights dominate the bandwidth stream, so the
first-order cost is a ~11 % decode-throughput loss on a bandwidth-bound CPU path
(`intake-1281#record`, derived).

Reference-default geometry (the paper's own H=16, d=128, `expand_v=1` — no GVA): both projections
are 2048 → 2048, i.e. 8,388,608 per layer for both gates combined — the same per-gate picture,
smaller per-layer total.

## 4. Rank budget — the factorization

Factorize each gate as `gate(x) = B · A · x` with `A: d_model → r`, `B: r → H·d`.

Cost per projection at rank `r`: `r·(d_model + out_dim)`.

| Rank `r` | b_proj cost | w_proj cost | per layer (both) | × 30 layers | vs full-rank | % of 3.0 B active |
|---|---|---|---|---|---|---|
| full | 4.194 M | 8.389 M | 12.583 M | 377.5 M | 1.00× | ~12.6 % |
| 32 | 131 k | 197 k | 0.328 M | 9.83 M | 38.4× less | ~0.33 % |
| 64 | 262 k | 393 k | 0.655 M | 19.66 M | 19.2× less | ~0.66 % |
| **128** (fla f/g rank; Kimi head dim) | 524 k | 786 k | **1.311 M** | **39.32 M** | **9.6× less** | **~1.3 %** |
| 256 | 1.049 M | 1.573 M | 2.622 M | 78.65 M | 4.8× less | ~2.6 % |
| 512 | 2.097 M | 3.146 M | 5.243 M | 157.3 M | 2.4× less | ~5.2 % |
| 1024 | 4.194 M | 6.291 M | 10.486 M | 314.6 M | 1.2× less | ~10.5 % |

Break-even rank: `b_proj` at `r = 1024` (2048·2048 = 4096·1024 exactly), `w_proj` at `r ≈ 1365`.
Any `r < 1024` strictly reduces both.

**Rank-128 rationale:** it is the rank the GDN-2 reference implementation already uses for the
layer's other two projections (`head_v_dim`), and the same rank class Kimi Linear uses for its KDA
forget-gate bottleneck (the KDA head dim). It is not a new hyperparameter class — it is the
reference's own design language applied to the remaining two projections.

## 5. Precedent

### 5.1 In-tree: Kimi Linear (KDA) — the production kernel tree already factorizes a decay projection

Verified read-only at `0db32c06e3e5`, `llama.cpp/src/models/kimi-linear.cpp:80-81`:

```
layer.ssm_f_a = create_tensor(..., {n_embd, n_embd_head_k_kda});                    // d_model → r
layer.ssm_f_b = create_tensor(..., {n_embd_head_k_kda, n_embd_head_k_kda * n_head}); // r → H·d
```

Consumed in the graph (`kimi-linear.cpp:302-304`) as the **decay/forget gate**:

```
g1 = -exp(A_log) * softplus(f_b(f_a(cur)) + dt_bias)
```

- Tensors registered as `LLM_TENSOR_SSM_F_A` / `LLM_TENSOR_SSM_F_B` at `src/llama-arch.h:473-474`
  ("kimi: forget gate projection A / B"), mapped to `blk.{bid}.ssm_f_a` / `ssm_f_b` at
  `gguf-py/gguf/constants.py:1239-1240` — exactly as the handoff states.
- This is the direct structural analog of what GDN-2 does with `b_proj`/`w_proj`: a channel-wise
  gate on the recurrence, computed as a projection of the input, factorized through a rank-`r`
  bottleneck instead of a full-rank map.
- GDN-2 recovers KDA exactly when `b_t = w_t = β_t·1` (Eq. 10 recovery conditions,
  `intake-1281#record`), so KDA is the *degenerate-rank* member of the same family — its scalar-β decay
  is rank-1 per head. The channel-wise `b` generalizes that scalar to a per-channel mask; nothing
  in the paper or the reference requires the mask to be generated by a full-rank map.

### 5.2 In the reference implementation itself: fla's GDN-2 already factorizes two of its four gates

Verified 2026-08-25 by fetching `fla-org/flash-linear-attention` @ `bc3b101d`
(`fla/layers/gdn2.py`), the exact revision the intake read:

```
# Decay-gate projection: low-rank bottleneck through head_v_dim to key_dim.
self.f_proj = nn.Sequential(
    nn.Linear(hidden_size, self.head_v_dim, bias=False),
    nn.Linear(self.head_v_dim, self.key_dim, bias=False),
)
...
self.g_proj = nn.Sequential(
    nn.Linear(hidden_size, self.head_v_dim, bias=False),
    nn.Linear(self.head_v_dim, self.value_dim, bias=True),
)
# GDN-2 channel-wise gates: b on the K axis, w on the V axis.
self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)      # FULL RANK
self.w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)    # FULL RANK
```

So the layer's own authors shipped the low-rank-bottleneck pattern for `f_proj` (decay) and
`g_proj` (output gate) — `d_model → head_v_dim → key_dim/value_dim`, rank 128 in reference
defaults — and left `b_proj`/`w_proj` full-rank. The paper gives **no justification** for that
asymmetry (`intake-1281#record`: "The method adds two full-rank per-layer projections … the paper
reports no parameter-overhead figure"; zero mentions of gate-rank ablation).

### 5.3 KDA lineage (context)

`b` and `w` are sigmoid-squashed per-channel masks in `[0,1]^{d}`. The paper's own w-only ablation
(w channel-wise, b scalar) *degrades* the baseline (16.55 ppl vs unmodified GDN 16.40 —
`intake-1281#record`), i.e. the write gate is the more fragile of the two; if a rank budget ever
had to be split asymmetrically, `b` (which carries the MK-NIAH-1 recall gains, `intake-1281#record`)
is the one that would justify higher rank. This is a training-time hypothesis, noted here only to
inform the B9(b)-owner's validation design.

## 6. What this analysis can and cannot claim

**Can claim (static, verified):**
- The full-rank choice is what produces the +12.6 % active-parameter cost on our geometry
  (exact arithmetic above).
- A rank-`r` factorization (`r < 1024`) strictly reduces that cost, monotonically; at `r = 128` —
  the reference's own gate rank — the cost drops ~9.6× to ~1.3 % of the active budget.
- The pattern has direct precedent in our production kernel tree (Kimi `SSM_F_A`/`SSM_F_B`) and in
  the GDN-2 reference implementation's own `f_proj`/`g_proj`.
- Therefore the "+12.5 % active parameters" objection to GDN-2 **does not survive factorization**;
  it survives only the "both gates stay full-rank" reading, which the reference itself does not
  extend to its other gates and the paper never justifies.

**Cannot claim (compute-gated, belongs to the B9(b) owner):**
- That a rank-128 gate *retains the reported gain* (the MK-NIAH-1 recall gains, the RULER
  S-NIAH-3 behavior, the 53.11/53.97 averages). Rank sufficiency is a training-time property; the
  gate's output stays channel-wise `[0,1]^{d}`, but the input-side linear span shrinks to rank `r`.
  The handoff names this explicitly: "validating that a low-rank gate retains the reported gain is
  compute-gated and belongs to whoever runs B9(b), not here."
- Anything about the erase-range (β ∈ [0,2]) control — untouched by factorization, still the
  omitted-control question `B9(b)` owns.

## 7. Bottom line for the handoff row

`Z12`: **factorizable — yes.** Rank budget `r ≤ 128` (preferred: `r = 128`, the reference's own
gate rank) brings GDN-2's added active parameters from ~377.5 M (~+12.6 %) to ~39.3 M (~+1.3 %),
a ~9.6× reduction, with in-tree and in-reference precedent. The +12.5 % objection **does not
survive** factorization; the rank-sufficiency question does, and stays with `B9(b)`.

---

*Era E9 claim grammar: all quantitative statements above are either (a) verified reads of primary
sources (GGUF header, frozen-tree source, pinned fla revision — all named above), or (b)
derivations with full arithmetic shown, or (c) citations of `intake-1281#record`
claim anchors. Nothing here asserts a measured model behavior.*
