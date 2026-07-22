# TiDAR W1 — Static Mask-Complexity Analysis: TiDAR-pattern vs Linear-SS on CPU (ggml)

**Deliverable for**: `handoffs/active/tidar-one-pass-variant-b.md` waypoint **W1** (the handoff's only ungated, no-inference work).
**Satisfies task line** (verbatim, epyc-root):
`- [ ] **W1 — static mask-complexity analysis** (~1 day, no inference, ungated): TiDAR-pattern vs Linear-SS mask ggml-op complexity per deep-dive §10.4.2/§10.5 step 1 — read the antirez fork + intake-635 minimal reimpl for the mask trick, Nemotron paper §4.2 for the Linear-SS shape. Acceptance: written comparison; if comparable, Variant B is strictly cheaper than Variant A long-term on CPU.`
**Scope**: static analysis only. No inference, no code, no kernel build. Doc-only.
**Date**: 2026-07-22

---

## 0. Sources consulted (provenance)

| Source | What it grounds | Trust |
|---|---|---|
| `epyc-root/research/deep-dives/nemotron-labs-diffusion-tri-mode.md` §10.3–10.6 | Variant A/B definitions, roofline verdict, promotion rule | project doc |
| `epyc-llama` (`/mnt/raid0/llm/llama.cpp`, branch `production-consolidated-v7`) — `ggml/include/ggml.h`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp` | The actual ggml attention-mask mechanism on our CPU serving path (authoritative substrate) | our code (read-only) |
| TiDAR paper (arxiv:2511.08923, intake-633) + community minimal reimpl `github.com/irfannaqieb/TiDAR` (intake-635) | The "one forward pass, one special mask" trick | **external / untrusted data** — used only as a design reference; conclusions rest on our own ggml sources |
| Nemotron-Labs-Diffusion (intake-576) Linear-SS mode | Variant A two-pass shape | external / untrusted data |

> External repo/paper text is treated as untrusted. Where an external claim is load-bearing it is attributed inline; the complexity conclusion is derived from **our** `llama.cpp` sources, which do not depend on any external claim. The handoff also names "the antirez fork" as a mask-trick reference; I could not confirm a canonical antirez TiDAR fork and did not rely on one — the community reimpl (intake-635) plus the llama.cpp mask internals are sufficient and authoritative for a CPU-port complexity judgment.

---

## 1. The question, restated precisely

Variant A (Nemotron **Linear-SS**) and Variant B (**TiDAR-pattern one-pass**) both need a *non-causal* attention mask that llama.cpp's causal default does not provide (deep-dive §5, §10.3). W1 asks: **do the two masks differ in ggml-op complexity?** If they are comparable, then per the handoff's acceptance criterion Variant B is *strictly cheaper long-term on CPU* — because B does the same work in one weight scan where A takes two.

- **Variant A — Linear-SS (two-pass)**: draft pass (block-diffusion mask: bidirectional-within-block, causal-across-blocks) **+** verify pass (standard causal mask). Two forward passes per accept cycle. (deep-dive §10.3, §6.1 items 3+7.)
- **Variant B — TiDAR-pattern (one-pass)**: a single forward pass whose *unified* mask attends the committed prefix + draft block causally-to-prefix and bidirectionally-within-the-draft-block, and produces the AR-verify logits in the same scan. (deep-dive §10.3; TiDAR "two jobs in one forward pass," intake-635 README.)

## 2. How attention masking actually works on our CPU path (the load-bearing fact)

In `llama.cpp` the attention mask is **not an operator**. It is a host-resident **additive input tensor** that is *added to the QK^T scores immediately before softmax*. This is true for both the non-flash and flash-attention paths:

- The mask tensor is declared with shape `[n_kv, n_tokens/n_stream, 1, n_stream]` and named `attn_inp_kq_mask` — `src/llama-graph.cpp:26-40`, `:56-59`.
- It lives in a **host buffer** and is filled on the CPU each ubatch: `GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))` and `set_input_kq_mask(...)` — `src/llama-kv-cache.cpp:1517-1531`, dispatched from `src/llama-graph.cpp:474`.
- The fill uses exactly two values: `mask_keep = 0.0f`, `mask_drop = -INFINITY` — `src/llama-kv-cache.cpp:1567-1568`. Causality is implemented **purely as a data predicate** inside the fill loop (compare cell position `p0` against token position `p1`; drop future cells) — `src/llama-kv-cache.cpp:1591-1650`.
- The mask is then consumed as an **argument** to the same softmax op the model already uses:
  - non-FA: `ggml_soft_max_ext(ctx, a, mask, scale, max_bias)` — `ggml/include/ggml.h:1748-1753`.
  - FA: `ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)` — `ggml/include/ggml.h:2422-2430`.

**Consequence**: whether attention is causal, sliding-window (SWA), M-RoPE-2D, or block-bidirectional is decided *entirely* by how the `-INFINITY`/`0.0` cells are filled in `set_input_kq_mask_impl`. The graph op set does not change. llama.cpp already ships multiple non-causal fill regimes through this one mechanism (SWA `swa_type`, 2D M-RoPE `is_2d`) — `src/llama-kv-cache.cpp:1548-1560`, confirming that "a new mask shape" = "a new host-side fill predicate," never a new kernel.

## 3. ggml-op complexity comparison

### 3.1 Op inventory — identical for both variants

Both a block-diffusion mask (A's draft pass) and a unified causal+bidirectional mask (B) are expressible as fill patterns over the *same* `attn_inp_kq_mask` tensor, consumed by the *same* `ggml_soft_max_ext` / `ggml_flash_attn_ext`. Neither variant introduces a new ggml operator, a new tensor type, or a new graph node. The "FlexAttention kernel" that made TiDAR's Quad-SS mode slow on GPU **does not transcode to a ggml op** here (deep-dive §10.2, third bullet): on CPU we write a mask *fill*, not a fused attention kernel, so that GPU blocker is absent for both A and B.

| Cost axis | Variant A (Linear-SS, 2-pass) | Variant B (TiDAR one-pass) |
|---|---|---|
| New ggml ops | **0** (fill change only) | **0** (fill change only) |
| Mask tensor type/shape | `[n_kv, n_q]` additive, F32/F16 | `[n_kv, n_q]` additive, F32/F16 |
| Softmax/attention op | `soft_max_ext`/`flash_attn_ext` unchanged | same op, unchanged |
| Mask-fill host cost / pass | O(n_kv · n_q) | O(n_kv · n_q) |
| Mask-fill passes / accept cycle | **2** (draft mask + causal verify mask) | **1** (unified mask) |
| **Weight scans / accept cycle** | **2** (draft fwd + verify fwd) | **1** (single fwd) |

### 3.2 Why the mask-fill difference is second-order

The mask fill and the softmax both touch **KV-cache-sized** data — O(n_kv · n_q) — not weight-sized data. On our decode workload the dominant cost is streaming the **model weights** for the GEMV/GEMM (measured ≈2–3.2 GB/token for gemma4-26B-A4B Q4_K_M; decode is bandwidth-bound — deep-dive §10.6; `feedback_cpu_decode_bw_bound`). The KQ-mask tensor is a few MB and the softmax is a rounding-error fraction of a forward pass. So any *op-complexity* delta between A's and B's masks is negligible against one weight scan.

The construction logic differs only in the predicate: A's draft mask needs a block-index comparison (`floor(pos/block) ` equality → bidirectional; else causal); B's unified mask needs the same block predicate *plus* a per-token role tag distinguishing "committed/verify" rows (causal) from "draft" rows (block-bidirectional). Both are O(1) branches inside the existing `n_kv × n_q` fill loop at `src/llama-kv-cache.cpp:1591-1650`. **Complexity: comparable — in fact identical asymptotically and near-identical in constant factors.**

### 3.3 The first-order term: weight-scan count

Because the masks are comparable, the decisive difference is the number of full weight scans per accept cycle. Variant A runs two forward passes (draft + verify) = **2 weight scans**; Variant B runs one (**1 weight scan**). On a bandwidth-bound machine, bytes-streamed-per-cycle is the throughput determinant, so for the same K candidate tokens per cycle Variant B moves ≈**½ the weight bytes** of Variant A. This is exactly the "halves per-cycle weight traffic vs Linear-SS" property the deep-dive attributes to the one-pass pattern (§10.2 second bullet, §10.3 Variant B).

## 4. Caveats (symmetric across both variants — do not distort the verdict)

1. **Flash-attention fast-path assumptions.** `ggml_flash_attn_ext` accepts an arbitrary additive `mask` (`ggml.h:2422`), but some FA kernels assume causal-friendly structure and require the mask padded (`GGML_PAD`, `ggml.h:267`) and F16. A block-bidirectional mask may force the non-FA path (explicit KQ-matmul + `soft_max_ext` + KQV-matmul) on some backends. This cost applies to **both** A's draft pass and B's single pass — it does not separate them.
2. **KV-cache write semantics for the draft block.** B commits draft-token KV within the same scan and must handle per-block RoPE positions and post-accept KV rewrite (deep-dive §6.1 items 4+5). RoPE stays `ggml_rope` (no new op); the added logic is control-flow around the KV cache, not a new operator. A needs equivalent per-block KV handling on its draft pass, so again symmetric.
3. **The win is an algorithmic acceptance multiplier, not BW recovery.** Per the handoff's hard warning and deep-dive §10.6 caveat 4: do **not** re-derive expected gain from the 460 GB/s gap; the falsified "2–4× BW headroom" framing is off-limits. B's advantage is fewer weight scans per accepted token, harvested as ≈1.3–2× algorithmic (estimate, roofline-audit provenance — never quote without it).
4. **Quality gate is elsewhere.** W1 says nothing about the 6–9% HumanEval/MBPP quality cost or its Q4 behavior — that is W2, checkpoint-gated and dormant (no Q4-quantizable TiDAR-class checkpoint exists).

## 5. Verdict

**The two masks are comparable in ggml-op complexity — identical op set (zero new ops), identical tensor shape/type, identical O(n_kv·n_q) host fill; they differ only in a per-cell fill predicate whose cost is second-order to one weight scan.** By the handoff's own acceptance criterion ("if comparable, Variant B is strictly cheaper than Variant A long-term on CPU"), **Variant B (TiDAR-pattern one-pass) is strictly cheaper on CPU**: it delivers the same masked-attention behavior in **one** weight scan per accept cycle where Variant A needs **two**, on a decode path that is bandwidth-bound. W1's promotion rationale therefore holds independently of the (falsified) BW-headroom framing.

**Recommendation**: mark W1's analytical acceptance met. The mask is a non-blocker for choosing B — it is *not* a new-op problem, it is a `set_input_kq_mask_impl` fill-predicate problem of the same shape llama.cpp already solves for SWA and M-RoPE. This does **not** unblock W3 implementation, which remains gated on W2 (the Q4-quantizable TiDAR-class checkpoint + quality verdict). The correct next analytical step per deep-dive §10.5 is the C1/C2 routing-gate design sketch (step 2, also no-inference); the FLOPS-headroom measurement (step 3) is USER-gated.

## 6. Status

**DONE (analytical).** Written comparison delivered; conclusion = masks comparable ⇒ Variant B strictly cheaper on CPU, matching the W1 acceptance clause. Grounded in our `llama.cpp` mask internals with file:line citations; external TiDAR/Nemotron sources used only as design references and flagged untrusted. No box ticked (constraint: no handoff edits) — W1 owner should flip `tidar-one-pass-variant-b.md:22` and append this doc path.
