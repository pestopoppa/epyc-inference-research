# Chapter 09: Deprecated Approaches

## Introduction

Not all optimization attempts succeed. This chapter documents approaches we tested and abandoned, preserving the lessons learned. Understanding *why* these failed prevents future researchers from rediscovering the same dead ends.

## Track 3: EAGLE-1 (Self-Speculative)

EAGLE-1 uses a trained autoregression head to predict future tokens from hidden states, enabling self-speculative decoding without a separate draft model. We tried it because it eliminates the need for a compatible draft model and published results showed 2-3x speedups. It produced a **0% acceptance rate** — EAGLE checkpoints are trained on specific model versions, and GGUF quantization breaks that compatibility completely.

<details>
<summary>Details and attempted fixes</summary>

### Why We Tried It

- Eliminates need for compatible draft model
- Potentially higher acceptance rates (same model family)
- Published results showed 2-3x speedup

### What Happened

**Result**: 0% acceptance rate

**Root Cause**: EAGLE checkpoints are trained on specific model versions. Our GGUF-quantized models have different internal representations. The autoregression head produces garbage predictions.

**Attempted Fixes**:
- Different EAGLE checkpoints
- Various quantization levels
- Temperature adjustments

None worked. EAGLE requires exact checkpoint compatibility that GGUF conversion breaks.

### Lesson Learned

Self-speculative methods requiring trained components are fragile to quantization and format changes. Prefer methods that work with any model.

</details>

---

## Track 7: CAS-Spec (Layer Skipping)

Cascade Speculative Drafting generates draft tokens by skipping early layers, using the remaining layers as a cheaper "draft model" within the same network. The idea is elegant — same weights, different depth. In practice: **0.446% acceptance rate**. Without trained exit classifiers, layer-skipped outputs diverge too far from the full model's predictions.

<details>
<summary>Details and analysis</summary>

### Why We Tried It

- No external draft model needed
- Theoretically elegant (same weights, different depth)
- Paper reported 2.3x speedup

### What Happened

**Result**: 0.446% acceptance rate

**Root Cause**: Without trained exit classifiers, the layer-skipped outputs diverge too much from full-model outputs. The "draft" tokens are essentially random relative to what the full model would produce.

**Analysis**: CAS-Spec requires:
1. Trained exit classifiers per layer (we don't have these)
2. Calibrated confidence thresholds (model-specific)
3. Architecture that supports clean layer boundaries

Our GGUF models lack the necessary trained components.

### Lesson Learned

Layer-skipping methods need trained classifiers. Raw layer output without proper exit prediction is useless for speculation.

</details>

---

## Track 5: SSM Speculation

We tried applying speculative decoding and prompt lookup to SSM-hybrid models (Qwen3-Next series). The result was corrupted output and invalid model state. SSM architectures maintain recurrent state that can't be rolled back like KV cache — when draft tokens are rejected, the state is permanently broken.

<details>
<summary>Technical details and state rollback illustration</summary>

```
Dense model rollback:
  Token 1 → KV[1] → Token 2 → KV[2] → Reject Token 2 → Restore KV[1] ✅

SSM model rollback:
  Token 1 → KV[1] + State[1] → Token 2 → KV[2] + State[2]
  → Reject Token 2 → Restore KV[1] but State still = State[2] ❌
```

**Lesson Learned**: **NEVER use speculation with SSM models.** This is a fundamental architectural incompatibility. Use expert reduction (Track 2) only.

</details>

---

## Track 4: Medusa

Medusa adds multiple parallel prediction heads to the model, each predicting a different future token position. We skipped it — requires training heads per model, training data and compute are significant, and heads don't transfer between model versions. External draft models (Track 1) achieve similar speedups without per-model training.

---

## Track 9: CLaSp/SWIFT

Similar to CAS-Spec — uses layer outputs for self-drafting with trained classifiers. Same fundamental issue: requires trained exit classifiers we don't have.

---

## Track 10: Kangaroo

Trains a small adapter network that predicts when the draft model will be accepted. Skipped — requires adapter training per model pair, overhead doesn't justify marginal gains over baseline Track 1, and adds another component to maintain.

---

## Summary: What Works vs What Doesn't

The pattern is clear: methods that work use separate, complete models or exploit structural properties (MoE sparsity, n-gram overlap) with no per-model training. Methods that fail require trained components we don't have, assume checkpoint compatibility that GGUF breaks, or can't handle state rollback.

<details>
<summary>Full comparison tables</summary>

### Works (Production)

| Track | Method | Speedup | Key Requirement |
|-------|--------|---------|-----------------|
| 1 | External Draft | 5.9-11x | Compatible tokenizer |
| 2 | MoE Reduction | 21-52% | MoE architecture |
| 8 | Prompt Lookup | 2-12.7x | Grounded task (overlap) |

### Doesn't Work (Deprecated)

| Track | Method | Failure Mode | Alternative |
|-------|--------|--------------|-------------|
| 3 | EAGLE-1 | Checkpoint incompatibility | Use Track 1 |
| 7 | CAS-Spec | No trained classifiers | Use Track 1 |
| 5 | SSM Speculation | State corruption | Use Track 2 only |
| 4 | Medusa | Requires head training | Use Track 1 |
| 9 | CLaSp/SWIFT | Same as CAS-Spec | Use Track 1 |
| 10 | Kangaroo | Requires adapter training | Use Track 1 |

</details>

<details>
<summary>References</summary>

### EAGLE Series

1. Li, Y., Cai, T., Zhang, Y., Chen, D., & He, D. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. ICML 2024. https://arxiv.org/abs/2401.15077

2. Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. EMNLP 2024. https://arxiv.org/abs/2406.16858

3. SafeAI Lab. (2024). *EAGLE: Extrapolation Algorithm for Greater Language-model Efficiency*. GitHub Repository. https://github.com/SafeAILab/EAGLE

### CAS-Spec and Layer Skipping

4. Chen, Z., Yang, X., Lin, J., Sun, C., Huang, J., & Chang, K. C. C. (2024). *Cascade Speculative Drafting for Even Faster LLM Inference*. NeurIPS 2025. https://arxiv.org/abs/2510.26843

5. Elhoushi, M., Shrivastava, A., Liskovich, D., Hosmer, B., Wasti, B., Lai, L., ... & Khabsa, M. (2024). *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*. ACL 2024. https://arxiv.org/abs/2404.16710

6. Zhang, B., Bai, H., Lin, X., Zhao, J., Hou, L., & Quan, C. (2025). *SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration*. ICLR 2025. https://openreview.net/forum?id=EKJhH5D5wA

7. Du, J., Wei, Y., & Ji, Z. (2025). *CLaSp: In-Context Learning of Adaptive Speculative Decoding via Dynamic Programming*. ACL 2025. https://arxiv.org/abs/2505.24196

### Medusa

8. Cai, T., Li, Y., Geng, Z., Peng, H., & Dao, T. (2024). *Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads*. ICML 2024. https://arxiv.org/abs/2401.10774

9. FasterDecoding. (2024). *Medusa GitHub Repository*. https://github.com/FasterDecoding/Medusa

### Kangaroo

10. Liu, F., Tang, Y., Liu, Z., Ni, Y., Han, K., & Wang, Y. (2024). *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting*. NeurIPS 2024. https://github.com/Equationliu/Kangaroo

### SSM Architectures

11. Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM 2024. https://arxiv.org/abs/2312.00752

12. Anthony, Q., Tokpanov, Y., Glorioso, P., & Berner, D. (2024). *BlackMamba: Mixture of Experts for State-Space Models*. arXiv preprint. https://arxiv.org/abs/2402.01771

### Cascade Methods

13. Chen, Z., Yang, X., Lin, J., Sun, C., Huang, J., & Chang, K. C. C. (2024). *Cascade Speculative Drafting for Even Faster LLM Inference*. arXiv preprint. https://arxiv.org/pdf/2312.11462

14. Spector, B., & Re, C. (2024). *Accelerating LLM Inference with Staged Speculative Decoding*. ICML 2024. https://arxiv.org/abs/2405.19261

### Curated Literature

15. Zhang, H., et al. (2024). *SpeculativeDecodingPapers: A Curated List of Speculative Decoding Research*. GitHub Repository. https://github.com/hemingkx/SpeculativeDecodingPapers

</details>

---

*Previous: [Chapter 08: RadixAttention](08-radix-attention.md)* | *Next: [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md)*
