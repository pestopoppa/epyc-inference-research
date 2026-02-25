# Early Failure Prediction for LLM Inference

**Project**: AMD EPYC 9655 Inference Optimization - Orchestrator Enhancement
**Date**: 2026-01-05
**Status**: Research Complete, Heuristic Implementation In Progress

---

## Executive Summary

This document reviews techniques for predicting LLM task failure *during* generation, enabling early abort to save compute. Three approaches are analyzed:

1. **ZIP-RC** — Zero-overhead logit repurposing (requires training, no open source)
2. **Gnosis** — 5M param failure head (best candidate, transfers across model sizes)
3. **Heuristics** — Training-free entropy/repetition monitoring (implementable now)

**Recommendation**: Implement training-free heuristics immediately. Monitor Gnosis for open source release — it offers 10-50× cheaper training than ZIP-RC with zero-shot transfer.

---

## Method Comparison

| Method | Predictor Size | Scales with Model? | Zero-Shot Transfer? | Training Required |
|--------|---------------|-------------------|---------------------|-------------------|
| ZIP-RC | 0 extra params | No (reuses logits) | **No** (per-model) | Yes, ~100k rollouts |
| Gnosis | 5M params fixed | **No** | **Yes** | Yes, once per family |
| Heuristics | 0 extra params | No | N/A | **No** |

---

## 1. ZIP-RC: Zero-Overhead Introspection

### Citation

> Manvi, R., Hong, J., Seyde, T., Labonne, M., Lechner, M., & Levine, S. (2025).
> *Zero-Overhead Introspection for Adaptive Test-Time Compute*.
> arXiv:2512.01457. https://arxiv.org/abs/2512.01457

### Method

ZIP-RC repurposes 64 unused vocabulary logits during each forward pass to predict:
- **Expected final reward** (probability of task success)
- **Remaining token count** (compute cost prediction)

At every token, the model outputs a joint distribution over (reward, remaining_length) using logits that are masked before sampling.

### Key Innovation

> "ZIP dedicates a fixed set of these logits (typically 64 tokens) to parameterize auxiliary predictions while masking them before sampling to prevent interference with the primary language modeling."

### Results

| Metric | Value |
|--------|-------|
| Accuracy improvement | +12% over Best-of-N voting |
| Compute efficiency | Equal or lower average cost |
| Pareto frontier | Smooth quality-compute-latency tradeoff |

### Training Requirements

| Model Size | Training Data | Estimated GPU Hours | Notes |
|------------|---------------|---------------------|-------|
| 350M-500M | 100k rollouts | ~8-16 H100 hours | Draft model |
| 1.7B-2B | 100k rollouts | ~24-48 H100 hours | Small target |
| 7B-8B | 100k rollouts | ~96-192 H100 hours | Typical worker |
| 32B | 100k rollouts | ~384-768 H100 hours | Large coder |

*Training: 2 epochs, αKL=10-100, DeepScaleR/MATH/GSM8K data*

### Limitations

- **No open source release** — code and models not publicly available
- **Per-model training** — no transfer across model sizes
- **High compute cost** — 100k+ rollouts per model

### Status for Our Project

**Future consideration** — requires training infrastructure we don't currently have.

---

## 2. Gnosis: Self-Awareness via Internal Circuits

### Citation

> Ghasemabadi, A., & Niu, D. (2024).
> *Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits*.
> arXiv:2512.20578. https://arxiv.org/abs/2512.20578

### Method

Gnosis adds a ~5M parameter "failure head" that observes hidden states and attention patterns to predict correctness. The backbone remains frozen.

**Architecture breakdown:**
- Hidden Circuit Encoder: 2.6M params
- Attention Circuit Encoder: 1.4M params
- Gated fusion head: ~1M params

### Key Innovation

> "Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost."

### Results

| Metric | Gnosis | Skywork-8B | Gemini 2.5 Pro | MLP-Prob |
|--------|--------|------------|----------------|----------|
| Math AUROC | **0.95** | 0.90 | 0.91 | 0.77 |
| TriviaQA AUROC | **0.87** | — | — | 0.80 |
| MMLU-Pro AUROC | **0.80** | — | — | 0.73 |

**Early Detection Performance:**
- At **40% generation**: Matches full-solution accuracy of other methods
- Can abort before model finishes reasoning

**Latency:**
- ~25ms constant overhead (sequence length independent)
- 37× faster than Skywork-8B at 12k tokens
- 99× faster than Skywork-8B at 24k tokens

### Key Advantages

1. **Fixed size**: 5M params works for 1.7B → 20B models
2. **Zero-shot transfer**: Train on 1.7B, works on 4B, 8B, 20B
3. **Early detection**: 40% of generation sufficient
4. **Beats billion-param judges**: Outperforms 8B reward models

### Training Requirements

| Backbone Size | Training Data | GPU Hours | Cloud Cost |
|---------------|---------------|-----------|------------|
| 1.7B | 54k examples | ~4-6 A100 hours | ~$5-10 |
| 4B-8B | 54k examples | ~8-12 A100 hours | ~$10-15 |
| 20B MoE | 54k examples | ~24 A100 hours | ~$25 |

*Training: 2 epochs, Adam, lr=1e-4, 54k examples (14k math + 40k QA)*

### vs. ZIP-RC

| Aspect | ZIP-RC | Gnosis |
|--------|--------|--------|
| Extra params | 0 | 5M (fixed) |
| Training cost | ~100-700 H100 hours | ~4-24 A100 hours |
| Transfer | No | Yes (zero-shot) |
| Early detection | Every token | 40% sufficient |

**Gnosis is 10-50× cheaper** with better transfer properties.

### Status for Our Project

**Best candidate for future implementation** — monitor for open source release.

---

## 3. Training-Free Heuristics (Implementable Now)

These methods require no training and work with existing GGUF models.

### 3.1 Token Entropy

**Source**: Standard information theory; used in [KnowLoop](https://arxiv.org/abs/2406.00430)

**Method**: Compute Shannon entropy from token logits at each step.

```python
def token_entropy(logits: np.ndarray) -> float:
    probs = softmax(logits)
    return -np.sum(probs * np.log(probs + 1e-10))
```

**Signal**: High sustained entropy (>4.0) indicates model uncertainty.

### 3.2 Entropy Spikes

**Method**: Detect sudden jumps in entropy between tokens.

**Signal**: Spike >2.0 may indicate model lost coherent reasoning path.

### 3.3 Perplexity Trend

**Method**: Track rolling average perplexity over generation window.

**Signal**: Monotonically rising perplexity suggests degrading output quality.

### 3.4 Repetition Detection

**Method**: Count n-gram frequencies in generated output.

**Signal**: >30% repeated 3-grams indicates degeneration loop.

### 3.5 Output Length Anomaly

**Method**: Compare current token count to task-specific median.

**Signal**: >2× median length may indicate runaway generation.

### Threshold Recommendations

| Signal | Default | Code | Reasoning | High-Tier |
|--------|---------|------|-----------|-----------|
| Entropy | >4.0 | >4.0 | >4.5 | >6.0 |
| Entropy spike | >2.0 | >2.0 | >2.5 | >4.0 |
| Repetition | >0.3 | >0.2 | >0.3 | >0.4 |
| Min tokens | 50 | 100 | 30 | 200 |

---

## 4. Open Source Resources

### llm-uncertainty (ICLR 2024)

> Xiong, M., et al. (2024). *Can LLMs Express Their Uncertainty?*
> GitHub: https://github.com/MiaoXiong2320/llm-uncertainty

Methods for uncertainty estimation in LLMs without internal access.

### collaborative-calibration

> GitHub: https://github.com/minnesotanlp/collaborative-calibration

Training-free post-hoc calibration via multi-agent deliberation.

### KnowLoop

> *Evaluating Uncertainty-based Failure Detection for Closed-Loop LLM Planners*.
> arXiv:2406.00430. https://arxiv.org/abs/2406.00430

Evaluates token probability, entropy, and self-explained confidence for failure detection.

---

## 5. Integration with Orchestrator

### Current State

`src/failure_router.py` handles escalation **after** complete task failure:
```
worker → coder → architect → FAIL
```

### Enhancement: Early Abort

Add `GenerationMonitor` to detect failure **during** generation:

```
Generation starts
     │
     ├─[token 50]─→ Check entropy/repetition
     │              └─ Abort? → EARLY_ABORT → Escalate immediately
     │
     ├─[token 100]─→ Check again
     │              └─ Abort? → EARLY_ABORT → Escalate
     │
     └─[complete]─→ Run gates → Pass/Fail → Normal routing
```

### Benefits

1. **Compute savings**: Don't waste tokens on failing generation
2. **Faster feedback**: Escalate sooner, get correct answer faster
3. **Resource efficiency**: Free up model for next task

### Configuration

Progressive threshold relaxation by escalation tier:

```yaml
tier_overrides:
  worker:    { entropy: 4.5, spike: 2.5 }   # Strictest
  coder:     { entropy: 5.0, spike: 3.0 }   # More relaxed
  architect: { entropy: 6.0, spike: 4.0 }   # Most tolerant
```

Rationale: Higher-tier models handle more complex tasks with natural uncertainty.

---

## 6. Future Work

### Short-Term (Heuristics)

1. Implement `GenerationMonitor` with entropy/repetition tracking
2. Integrate with `failure_router.py` for early abort
3. Collect baseline metrics during production runs
4. Tune thresholds based on observed distributions

### Medium-Term (Gnosis)

1. Monitor for open source release of Gnosis
2. Evaluate feasibility of training 5M param head
3. Test zero-shot transfer on our Qwen model family
4. Compare to heuristic baseline

### Long-Term (ZIP-RC)

1. If training infrastructure available, consider ZIP-RC
2. Would require per-model training (~100k rollouts each)
3. Only worthwhile if heuristics/Gnosis insufficient

---

## 7. References

1. Manvi, R., Hong, J., Seyde, T., Labonne, M., Lechner, M., & Levine, S. (2025).
   Zero-Overhead Introspection for Adaptive Test-Time Compute.
   *arXiv preprint arXiv:2512.01457*.

2. Ghasemabadi, A., & Niu, D. (2024).
   Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits.
   *arXiv preprint arXiv:2512.20578*.

3. Xiong, M., et al. (2024).
   Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.
   *ICLR 2024*.

4. Ren, J., et al. (2024).
   Evaluating Uncertainty-based Failure Detection for Closed-Loop LLM Planners.
   *arXiv preprint arXiv:2406.00430*.

5. Chen, Z., et al. (2024).
   Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation.
   *arXiv preprint arXiv:2404.09127*.

---

## Appendix: Compute Cost Summary

### Training a Complete Failure Prediction System

**Option A: ZIP-RC (Per-Model Training)**
```
Draft (0.5B):     ~12 H100 hours
Worker (8B):      ~144 H100 hours
Coder (32B):      ~576 H100 hours
Architect (235B): ~2000+ H100 hours (estimated)
────────────────────────────────────
Total:            ~2700+ H100 hours (~$8000+ cloud)
```

**Option B: Gnosis (One Head, Transfers)**
```
Train on Qwen3-1.7B: ~5 A100 hours (~$10)
Transfers to: 4B, 8B, 32B, 235B
────────────────────────────────────
Total:               ~5 A100 hours (~$10)
```

**Option C: Heuristics (No Training)**
```
Training cost: $0
Implementation: ~1-2 days engineering
────────────────────────────────────
Total:         $0 + engineering time
```

**Recommendation**: Start with Option C, upgrade to Option B when Gnosis is available.
