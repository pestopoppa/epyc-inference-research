# Chapter 05: Speculative Decoding (Track 1)

## Introduction

Speculative decoding is our primary optimization technique, achieving **11x speedup** on code generation. The approach uses a small "draft" model to propose multiple tokens, which are then verified in parallel by the larger "target" model. Since the memory-bound generation step reads the entire model for each token, verifying N tokens in one pass costs only slightly more than generating one token.

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

<details>
<summary>Performance measurements by model pair</summary>

| Target Model | Draft Model | K | Acceptance | Speedup |
|--------------|-------------|---|------------|---------|
| **Qwen2.5-Coder-32B** | Qwen2.5-Coder-0.5B | **24** | 70.8% | **11x** |
| Qwen2.5-Coder-32B | Qwen2.5-Coder-0.5B | 16 | 75% | 5.9x |
| Qwen2.5-VL-7B | Qwen2.5-Coder-0.5B | 8 (temp=0.7) | 74.2% | **3.7x** |
| Qwen2.5-Math-7B | Qwen2.5-Coder-0.5B | 8 | 65.6% | **3.9x** |
| Qwen2.5-Math-72B | Qwen2.5-Coder-0.5B | 16 (temp=0.5) | 60.3% | **7.3x** |
| Meta-Llama-70B | PARD-Llama-3.2-1B | 8 | 79% | **5.0x** |

</details>

## K-Value Optimization

The number of draft tokens (K) is the single most impactful tuning parameter. Bigger models can absorb higher K because their per-token verification cost is so large that even moderate acceptance rates pay off. Content type matters just as much — code is highly predictable with 80-90% acceptance, while creative prose drops to 30-50%.

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

<details>
<summary>Code: launch command and flags</summary>

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

## SSM Architecture Incompatibility

Hybrid SSM models (Qwen3-Next series) are fundamentally incompatible with speculative decoding. SSM architectures maintain recurrent state that can't be rolled back like KV cache — when draft tokens are rejected, the state is permanently corrupted. Use expert reduction only for these models.

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

### Curated Literature

9. Zhang, H., et al. (2024). *SpeculativeDecodingPapers: A Curated List*. GitHub. https://github.com/hemingkx/SpeculativeDecodingPapers

</details>

## Architect Model Spec Decode Results (2026-02-13)

Large architect models use full experts + spec decode (quality over speed). Key findings:

**Qwen3-Coder-480B-A35B** (BOS = comma, token 11):
- Standard Qwen3 drafts: 0% acceptance (BOS mismatch)
- jukofyork vocab-transplant draft (`Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf`): 74-82% acceptance on code refactoring, 57% on novel generation
- Production config: Full experts + spec (K=16) = 9.00 t/s (1.38x). MoE3+spec was 12.74 t/s but sacrifices quality.

**Qwen3-235B-A22B**:
- 0.6B Q8_0 draft dramatically outperforms 1.7B: 55% vs 21% acceptance. Smaller draft wins on CPU due to faster proposal generation.
- Production config: Full experts + 0.6B spec (K=16) = 6.08 t/s (1.15x). MoE4+spec was 8.21 t/s but sacrifices quality.

**Policy**: Architect roles prioritize quality. Full experts + spec decode is the production config. Frontdoor/coder roles use MoE + spec + lookup (speed matters more).

---

*Previous: [Chapter 04: Storage & Safety](04-storage-and-safety.md)* | *Next: [Chapter 06: MoE Optimization](06-moe-optimization.md)*
