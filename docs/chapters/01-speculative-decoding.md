# Chapter 01: Speculative Decoding (Track 1)

> **Current Status (May 2026)**: This chapter documents the 2026-02 baseline foundations. On EPYC 9655 with the current production stack, **NUMA 4-way parallel serving is the primary acceleration lever (6.7x aggregate)**; speculative decoding contributes an incremental +17–21% on top of that. The "11x" headline below is a 2025 Qwen2.5-Coder-32B + 0.5B-draft measurement; current production worker (Gemma4-26B-A4B MTP, swapped in 2026-05-08) achieves only 1.06x via MTP on MoE batch=1 (2.98x on dense Gemma4-31B). For 2026-04+ findings see [Chapter 10](10-advanced-speculative-decoding.md) and the production wiki on [speculative decoding](../../wiki/speculative-decoding.md).

## Introduction

Speculative decoding is a foundational optimization technique that achieved **11x speedup** on code generation in our 2025 baselines. The approach uses a small "draft" model to propose multiple tokens, which are then verified in parallel by the larger "target" model. Since the memory-bound generation step reads the entire model for each token, verifying N tokens in one pass costs only slightly more than generating one token.

## How It Works

The trick is simple: instead of reading the entire model once per token, a small draft model sprints ahead and proposes K tokens at once. The big model then verifies all K in a single forward pass — if the draft guessed right, you get K tokens for the cost of one.

<details>
<summary>Verification mechanics and key insight</summary>

```
Standard Generation:
  Token 1 → Read full model → Token 2 → Read full model → Token 3 → ...

Speculative Decoding:
  Draft K tokens → Read full model once → Verify all K → Accept N of K
```

**Key Insight**: If the draft model proposes K tokens and N are accepted, we generate N tokens for approximately the cost of 1, achieving up to Kx speedup (limited by acceptance rate).

</details>

## Best Results

The headline numbers speak for themselves — an 11x speedup on code generation with a 0.5B draft model driving a 32B target. The sweet spot depends on the model pair and content type, but every model we've tested benefits significantly from speculation.

> **Note**: Table below reflects 2026-02 baseline measurements on Qwen2.5-family targets. For 2026-04+ production results on EPYC (Gemma4 MTP, DFlash NO-GO, REAP models, NUMA-saturated incremental gains), see [Chapter 10: Advanced Speculative Decoding](10-advanced-speculative-decoding.md) and the production wiki entry on [speculative decoding](../../wiki/speculative-decoding.md).

<details>
<summary>Performance measurements by model pair (2026-02 baseline)</summary>

| Target Model | Draft Model | K | Acceptance | Speedup |
|--------------|-------------|---|------------|---------|
| **Qwen2.5-Coder-32B** | Qwen2.5-Coder-0.5B | **24** | 70.8% | **11x** |
| Qwen2.5-Coder-32B | Qwen2.5-Coder-0.5B | 16 | 75% | 5.9x |
| Qwen2.5-VL-7B | Qwen2.5-Coder-0.5B | 8 (temp=0.7) | 74.2% | **3.7x** |
| Qwen2.5-Math-7B | Qwen2.5-Coder-0.5B | 8 | 65.6% | **3.9x** |
| Qwen2.5-Math-72B | Qwen2.5-Coder-0.5B | 16 (temp=0.5) | 60.3% | **7.3x** |
| Meta-Llama-70B | PARD-Llama-3.2-1B | 8 | 79% | **5.0x** |

**Scope caveat**: Qwen2.5-Coder-32B was replaced in production by **Gemma4-26B-A4B-Q4_K_M** (worker_general) on 2026-05-08. The current production speculative-decoding mechanism is MTP (Multi-Token Prediction) integrated into the worker model itself, not an external 0.5B draft — and on Gemma4's MoE architecture the measured speedup is only **1.06x** at batch=1 due to MoE-batch-cancellation effects (dense Gemma4-31B sees 2.98x). The 11x figure is therefore historical context, not a current production benchmark.

</details>

## K-Value Optimization

The number of draft tokens (K) is the single most impactful tuning parameter. Bigger models can absorb higher K because their per-token verification cost is so large that even moderate acceptance rates pay off. Content type matters just as much — code is highly predictable with 80-90% acceptance, while creative prose drops to 30-50%.

> **Revised 2026-04+**: The K-tuning landscape changes fundamentally under tree speculation and NUMA parallelism. Empirical validation on EPYC ([Chapter 10 §10.3](10-advanced-speculative-decoding.md#103-large-k-linear-speculation-phase-3)) shows that linear speculation **saturates at K~16** on this hardware: throughput is flat from K=16 → K=256 across all measured target/draft pairs (acceptance decays geometrically). Tree branching can lift the per-cycle accepted-token count to ~8–12 (vs ~4–6 linear), but only on f16 targets or large MoE; medium dense Q4/Q6 verification is too fast to amortize the ~41ms tree-construction overhead. See Chapter 10 for the full empirical profile.

<details>
<summary>K-value tuning by model size and content type</summary>

### Discovery: Larger K for Larger Models

| Model Size | Optimal K | Reasoning |
|------------|-----------|-----------|
| 7B targets | K=8 | High baseline speed, K>8 reduces acceptance too much |
| 32B targets | K=16-24 | Lower baseline speed, more tokens per verification worthwhile |
| 72B targets | K=16 | Balance of acceptance vs verification cost |

**Process for New Models**:
1. Start with K=8, measure speed and acceptance
2. If acceptance >60%, try K=12, K=16, K=24
3. Plot speed vs K - optimal is where curve flattens
4. Higher K = more draft tokens but lower acceptance per token

### Context-Dependent Performance

| Context Type | K | Speed (t/s) | Acceptance | Speedup |
|--------------|---|-------------|------------|---------|
| **Code** | 24 | **28.79** | 83.33% | **10.0x** |
| Code | 8 | 17.32 | 86.84% | 6.0x |
| Prose | 8 | **7.85** | 32.44% | 2.7x |
| Prose | 24 | 6.22 | 14.76% | 2.2x |

### Recommended K by Content Type

| Content Type | Optimal K | Expected Acceptance | Expected Speedup |
|--------------|-----------|---------------------|------------------|
| Code/structured | 20-24 | 80-90% | 8-10x |
| JSON/schemas | 8-12 | 60-80% | 5-7x |
| General/mixed | 8-12 | 50-70% | 4-6x |
| Creative/prose | 4-8 | 30-50% | 2-4x |

</details>

## Temperature Tuning Discovery

Here's a counterintuitive finding: non-zero temperature can actually *improve* speculative decoding for some model pairs. The hypothesis is that temp=0 produces overly deterministic drafts that diverge from the target's probability distribution. A little randomness makes the draft more "target-like."

> **Caveat (2026-05)**: The findings below are 2025 baseline measurements on the Qwen2.5 family. Recent experiments (May 2026) on Gemma4 and REAP-pruned models indicate **temperature tuning is largely orthogonal to MTP/draft selection** and interacts non-trivially with MoE expert routing. The model-family-specific direction reported here does not generalize cleanly to current production targets — re-measure per pair rather than transferring constants.

<details>
<summary>Temperature effect measurements</summary>

| Model | temp=0 | temp=0.5 | temp=0.7 | Best |
|-------|--------|----------|----------|------|
| Qwen2.5-VL-7B | 28.3 t/s | 37.4 t/s | **57.1 t/s** | temp=0.7 |
| Qwen2.5-Math-72B | 6.0 t/s | **7.5 t/s** | N/A | temp=0.5 |
| Qwen2.5-Coder-32B | **26.6 t/s** | 19.0 t/s | 19.4 t/s | temp=0 |

**Recommendation**: If acceptance is <50% at temp=0, try temp=0.3-0.7.

</details>

## Compatibility Matrix

Speculative decoding requires **exact tokenizer compatibility** between draft and target models. Same vocabulary size, identical special tokens, same tokenizer type. Same model family is NOT enough — DeepSeek-R1-Distill-Qwen-32B can't use DeepSeek-R1-Distill-Qwen-1.5B despite similar names because they have different vocab sizes.

<details>
<summary>Compatibility table and failure modes</summary>

| Target Family | Compatible Drafts | Incompatible |
|---------------|-------------------|--------------|
| Qwen2.5-* | Qwen2.5-0.5B, Qwen2.5-1.5B | Qwen3-*, DeepSeek-*, PARD variants |
| Qwen3-* | Qwen3-0.6B | Qwen2.5-* |
| Meta-Llama-3.* | PARD-Llama-3.2-1B | Other families |
| DeepSeek-R1-Distill-* | **None found** | All tested |

**Critical Failure Mode**: DeepSeek-R1-Distill-Qwen-32B cannot use DeepSeek-R1-Distill-Qwen-1.5B — they have different vocab sizes (152,064 vs 151,936).

</details>

## Quick Start Command

The standard launch pattern for speculative decoding on this hardware.

> **Legacy CLI example** — `llama-speculative` is the 2025 standalone CLI binary. Current production launches go through `llama-server` (event-based API) driven by the orchestrator registry; see `/workspace/repos/epyc-orchestrator/orchestration/model_registry.yaml` and `epyc-orchestrator/scripts/server/orchestrator_stack.py` for the actual production launch params (slot IDs, NUMA pinning, MTP-specific flags).

<details>
<summary>Code: legacy 2025 CLI launch command</summary>

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "Your prompt"
```

**Flags**:
- `-m`: Target (large) model
- `-md`: Draft (small) model
- `--draft-max`: K value (max tokens to draft)
- `-t 96`: Use all physical cores

</details>

## SSM Architecture Incompatibility (Updated 2026-04)

Hybrid SSM architectures (Qwen3-Next, Qwen3.5) face a **verification wall**: multi-token verification costs approximately N× single-token decode because ~75% of layers are sequential recurrent (Delta Net / Mamba2). External speculative decoding with a separate draft model is not viable — the MTP-1 closed handoff measured **0.56x throughput at batch=2** on Qwen3.5, and naive draft-then-verify corrupts the recurrent state on rejection.

However, two paths are now viable on this architectural class:

1. **Prompt lookup via auto freeze-recurrent** (validated 2026-03-10): The server auto-activates `--freeze-recurrent-draft` on hybrid models, freezing SSM state writes during speculation. Acceptance drops ~13pp vs dense targets, but net throughput is positive (+5–10% on Qwen3.5-9B with a fast Qwen2.5-Coder-0.5B drafter; see [Chapter 10 §11.5](10-advanced-speculative-decoding.md#115-freeze-recurrent-speculation--breakthrough)). On Qwen3-Next-80B summarization, this combines with prompt-lookup n-gram drafting (Chapter 03) for large gains.
2. **Expert reduction (Track 2)**: Independent of speculation, see Chapter 02.

External speculative decoding with a separate draft model **remains incompatible** on hybrid SSM without freeze-recurrent. Slot-promotion speculation (April 2026) was tested as an alternative mechanism and found net-negative (see Chapter 05).

<details>
<summary>References</summary>

### Foundational Papers

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. Proceedings of the 40th International Conference on Machine Learning (ICML). https://arxiv.org/abs/2211.17192

2. Chen, C., Borgeaud, S., Irving, G., Lespiau, J. B., Sifre, L., & Jumper, J. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv preprint. https://arxiv.org/abs/2302.01318

3. Xia, H., Ge, T., Wang, P., Chen, S., Wei, F., & Sui, Z. (2024). *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding*. ACL 2024. https://arxiv.org/abs/2401.07851

### K-Value and Temperature Optimization

4. Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M. W., Gholami, A., & Keutzer, K. (2024). *Speculative Decoding with Big Little Decoder*. NeurIPS 2023. https://arxiv.org/abs/2302.07863

5. Sun, Z., Suresh, A. T., Ro, J. H., Beirami, A., Jain, H., & Yu, F. (2024). *SpecTr: Fast Speculative Decoding via Optimal Transport*. NeurIPS 2023. https://arxiv.org/abs/2310.15141

### PARD (Parallel Aligned Draft)

6. AMD Research. (2025). *PARD: Permutation-Aligned Residual Draft for Ultra-Fast Speculative Decoding*. https://github.com/AMD-AIG-AIMA/AMD-PACE

### Implementation Resources

7. Gerganov, G., et al. (2024). *llama.cpp Speculative Decoding*. GitHub. https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative

8. vLLM Team. (2024). *Speculative Decoding in vLLM*. vLLM Blog. https://blog.vllm.ai/2024/10/17/spec-decode.html

### Block Diffusion Drafting

10. *DFlash: O(1) Block-Diffusion Speculative Drafting* (Feb 2026). See `handoffs/completed/dflash-block-diffusion-speculation.md` for the EPYC port and the Q4_K_M NO-GO finding. The technique remains a foundational reference for unified-model self-speculation directions (Nemotron-Labs-Diffusion, May 2026).

### MTP Drafting (Production, May 2026)

11. *Gemma4 MTP via ik_llama.cpp PR #1744* (2026-05-08). Production speculative-decoding mechanism for `worker_general`; see `progress/2026-05/2026-05-08.md` and `wiki/speculative-decoding.md` § "Gemma 4 MTP Drafter."

### Curated Literature

9. Zhang, H., et al. (2024). *SpeculativeDecodingPapers: A Curated List*. GitHub. https://github.com/hemingkx/SpeculativeDecodingPapers

</details>

## Architect Model Spec Decode Results (2026-02-13)

Large architect models historically used full experts + spec decode (quality over speed). Key findings from the 2026-02 baseline:

**Qwen3-Coder-480B-A35B** (BOS = comma, token 11):
- Standard Qwen3 drafts: 0% acceptance (BOS mismatch)
- jukofyork vocab-transplant draft (`Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf`): 74-82% acceptance on code refactoring, 57% on novel generation. **Status (2026-05)**: experimental / not in current production registry — included here as historical context, not an active configuration.
- Historical config: Full experts + spec (K=16) = 9.00 t/s (1.38x). MoE3+spec was 12.74 t/s but sacrifices quality.

**Qwen3-235B-A22B**:
- 0.6B Q8_0 draft dramatically outperforms 1.7B: 55% vs 21% acceptance. Smaller draft wins on CPU due to faster proposal generation.
- Historical config: Full experts + 0.6B spec (K=16) = 6.08 t/s (1.15x). MoE4+spec was 8.21 t/s but sacrifices quality.

**Historical policy**: Architect roles prioritized quality. Full experts + spec decode was the production config. Frontdoor/coder roles used MoE + spec + lookup (speed matters more). This policy has been superseded by the May 2026 stack consolidation — see Production Update below.

---

## Production Update (May 2026) — Gemma4 MTP, REAP, DFlash

The 2025-baseline configurations above no longer reflect production. Headline changes:

- **Gemma4-26B-A4B MTP (DEPLOYED, 2026-05-08)**: Replaced Qwen2.5-Coder-32B as `worker_general`. MTP (Multi-Token Prediction) drafting is integrated into the model itself via ik_llama.cpp PR #1744 — no external 0.5B draft. Measured **1.06x** on the 26B-A4B MoE at batch=1 (MoE expert-routing cancels most of the speculation gain), **2.98x on dense Gemma4-31B**, +18pp tool_compliance, +36% tps end-to-end. Production launch requires `KMP_BLOCKTIME=10` and the 8 MTP-specific flags wired into `orchestrator_stack.py`; see `progress/2026-05/2026-05-08.md`.
- **REAP-pruned pure-MoE targets (DEPLOYED)**: REAP-25B (15GB, pure MoE — no SSM) achieves 39.62 t/s at `dm=24` (+101% vs baseline), and REAP-246B (pure MoE from Qwen3-Next-235B) re-enables standard `--draft` speculation on what would otherwise be a hybrid-SSM target. REAP pruning is the most impactful single intervention enabling speculation on the Qwen3-Next family. See `handoffs/completed/reap-moe-expert-pruning.md`.
- **DFlash block diffusion (NO-GO on Q4_K_M, 2026-02)**: Frontier O(1)-draft technique. On GPU/f16 hits 6.49 accepted tokens per round; on EPYC Q4_K_M, quantization noise in hidden-state extraction degrades acceptance to 27% per token (13.0 t/s vs 36.5 t/s autoregressive). Documented in Chapter 05 as deprecated for CPU Q4_K_M; remains a foundational paper worth citing in the references section.
- **MoE-Spec verification-budget mechanism (DEPLOYABLE)**: Independent expert union reduction during verification batches; +15.2% on REAP-246B forward-pass, +3% e2e. Orthogonal to MTP/draft selection.
- **Nemotron-Labs-Diffusion (2026-05-19)**: New unified self-speculation architecture with dense Ministral3 backbone (no SSM); 5.46 accepted tokens per cycle on some tasks (better than EAGLE-3/MTP). CPU portability assessment in progress — see wiki § "Unified-model self-speculation."

For deeper analysis of all five, see [Chapter 10 §10.5](10-advanced-speculative-decoding.md) and `/workspace/wiki/speculative-decoding.md` § "Updates — 2026-04-28".

---

*Next: [Chapter 02: MoE Expert Reduction](02-moe-optimization.md)*
