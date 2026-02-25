# LLM Inference Optimization on AMD EPYC 9655 Turin
## A Comprehensive Research Report

**Author:** [Your Name]
**Date:** December 2025
**System:** AMD EPYC 9655 "Turin" (96 cores, 1.13TB DDR5)

---

## Abstract

[2-3 paragraphs summarizing: the problem (LLM inference is slow/expensive), the approach (speculative decoding + other optimizations), key results (X speedup achieved), and significance (CPU-only inference becomes viable)]

---

## 1. Introduction

### 1.1 Motivation
- LLM inference is memory-bandwidth bound on CPUs
- High-core-count server CPUs like EPYC have massive memory bandwidth
- Can speculative decoding unlock this potential?

### 1.2 Research Questions
1. Which speculative decoding techniques work best on CPU?
2. How do MoE models behave differently from dense models?
3. What parameter tuning yields optimal performance?
4. Can multiple techniques be combined for compounding gains?

### 1.3 Contributions
- [List 3-5 key contributions/findings]

---

## 2. System Configuration

### 2.1 Hardware
| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" (96 cores, 192 threads, Zen 5) |
| RAM | 1.13 TB DDR5-5600 ECC (12 channels, ~460 GB/s theoretical) |
| Storage | 2x Solidigm P44 Pro 2TB NVMe RAID0 |
| TDP | 500W |

### 2.2 Software Stack
| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 |
| Inference Engine | llama.cpp (commit: [hash]) |
| BLAS | AMD AOCL BLIS 5.0 |
| Compiler | GCC 14.x with `-march=znver5` |

### 2.3 Runtime Configuration
```bash
# Critical settings for reproducibility
export OMP_NUM_THREADS=1        # Prevent nested parallelism
numactl --interleave=all        # Saturate all memory channels
-t 96                            # Physical cores only (no SMT)
```

**Why these settings:**
- `OMP_NUM_THREADS=1`: llama.cpp uses its own threading; OMP nesting causes contention
- `--interleave=all`: Distributes memory across all 12 channels for maximum bandwidth
- 96 threads: SMT (192) degrades performance due to cache contention

---

## 3. Research Methodology

### 3.1 Track-Based Approach

We evaluated speculative decoding techniques in parallel "tracks":

| Track | Technique | Type | Training Required |
|-------|-----------|------|-------------------|
| 1 | External Draft Model | Draft-based | No |
| 2 | MoE Expert Reduction | Model optimization | No |
| 3 | EAGLE-1 | Learned draft head | Yes (pretrained) |
| 6 | SuffixDecoding | Retrieval-based | No |
| 7 | CAS-Spec (Layer Skip) | Self-draft | No |
| 8 | Prompt Lookup | Retrieval-based | No |

### 3.2 Evaluation Metrics
- **Tokens/second (t/s)**: Primary throughput metric
- **Speedup**: Ratio vs baseline (no speculation)
- **Acceptance rate**: % of draft tokens accepted by target model
- **Quality**: Verified output matches greedy decoding (temp=0)

### 3.3 Benchmark Protocol
1. **Warmup**: 1 generation discarded (cache effects)
2. **Iterations**: 3+ runs per configuration
3. **Prompt types**: Code generation, summarization, math
4. **Token count**: 128 tokens generated per run

---

## 4. Results

### 4.1 Baseline Performance

[Table of baseline t/s for all tested models, grouped by size]

### 4.2 Track 1: External Draft Model

**Methodology:**
- Small model (0.5B-1.7B) generates K draft tokens
- Target model verifies all K tokens in single forward pass
- Accepted tokens output; rejected tokens regenerated

**Key Finding: K-Value Tuning**
| Model Size | Optimal K | Reasoning |
|------------|-----------|-----------|
| 7B | K=8 | High baseline speed, diminishing returns beyond K=8 |
| 32B | K=16-24 | Verification cost amortized over more tokens |
| 72B | K=16 | Balance of acceptance vs verification overhead |

**Results:**
| Target | Draft | K | Acceptance | Speedup |
|--------|-------|---|------------|---------|
| Qwen2.5-Coder-32B | Qwen2.5-0.5B | 24 | 70.8% | **11x** |
| ... | ... | ... | ... | ... |

**Unexpected Discovery: Temperature Tuning**
Non-zero temperature can improve speculative decoding performance:
- Qwen2.5-VL-7B: temp=0.7 gives 57.1 t/s vs 28.3 t/s at temp=0 (2x improvement)
- Hypothesis: Softened distributions improve draft-target alignment

### 4.3 Track 2: MoE Expert Reduction

**Methodology:**
- Reduce active experts via `--override-kv ARCH.expert_used_count=int:N`
- Tests: 8 → 6 → 4 → 3 experts on 30B MoE models

**Results:**
[Table of expert count vs speed vs quality]

**Key Finding:** MoE models are already fast (3B active params run at 24 t/s). Traditional speculative decoding provides NO benefit—draft overhead exceeds savings.

### 4.4 Track 8: Prompt Lookup

**Methodology:**
- N-gram matching in prompt for draft candidates
- Zero computational cost (pure lookup)
- Falls back to Track 1 when no matches

**Results:**
| Task Type | Model | Speedup |
|-----------|-------|---------|
| Summarization | Qwen3-Next-80B | **12.7x** |
| Code editing | Qwen2.5-Coder-32B | 8.6x |

### 4.5 Failed Tracks: Lessons Learned

#### Track 3: EAGLE-1 (0% Acceptance)
**Problem:** Despite using official EAGLE checkpoints, acceptance rate was exactly 0%.
**Investigation:** 20+ hours of debugging revealed architecture/checkpoint incompatibility.
**Lesson:** "Zero-shot" EAGLE requires exact model-checkpoint matching; not generalizable.

#### Track 7: CAS-Spec (0.446% Acceptance)
**Problem:** Layer-skip self-drafting produced nearly random tokens.
**Root cause:** Knowledge in transformer layers is NOT evenly distributed; skipping layers loses critical information. Would require trained exit classifiers.
**Lesson:** Self-drafting without training is unreliable.

---

## 5. Key Insights

### 5.1 CPU Inference is Memory-Bound, Not Compute-Bound
Small draft models (0.5B at 85 t/s) vastly outperform larger drafts (7B at 8 t/s) despite lower acceptance rates. On CPU, more speculation rounds beat higher-quality drafts.

### 5.2 MoE Models Break Speculative Decoding Assumptions
MoE models with 3B active parameters already run at "draft model" speeds. Adding speculation overhead hurts rather than helps. Different optimization strategies needed.

### 5.3 Retrieval-Based Methods Excel on Grounded Tasks
Prompt Lookup (12.7x) dramatically outperforms model-based drafting (5.9x) on tasks with high prompt-output overlap. Zero-cost to implement.

### 5.4 Temperature Affects More Than Quality
Non-zero temperature can improve speculative decoding by softening distribution alignment. Optimal temp varies by model family.

---

## 6. Recommended Configuration

### For Dense Models (7B-72B)
```bash
llama-speculative \
  -m TARGET.gguf \
  -md DRAFT_0.5B.gguf \
  --draft-max [8 for 7B, 16-24 for 32B+] \
  -t 96 --temp 0.5
```

### For MoE Models (30B-480B)
```bash
llama-cli \
  -m MOE_MODEL.gguf \
  --override-kv ARCH.expert_used_count=int:4 \
  -t 96
```

### For Grounded Tasks (Summarization, Code Editing)
Use Prompt Lookup first, fall back to external draft.

---

## 7. Future Work

1. **SuffixDecoding (Track 6)**: Test retrieval from session history for agentic workloads
2. **Hybrid approaches**: Combine Prompt Lookup → SuffixDecoding → External Draft cascade
3. **Dynamic K selection**: Adjust speculation depth based on content type
4. **MoE-specific optimizations**: Expert prefetching, routing prediction

---

## 8. Conclusions

[2-3 paragraphs summarizing the research journey, key findings, and practical implications for CPU-based LLM inference]

---

## References

### Speculative Decoding
1. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2022)
2. [SuffixDecoding Paper](https://suffix-decoding.github.io/) - NeurIPS 2025
3. [CAS-Spec](https://arxiv.org/abs/2510.26843) - Cascade Speculative Drafting

### MoE Models
4. Fedus et al., "Switch Transformers" (2021)
5. Jiang et al., "Mixtral of Experts" (2024)

### Implementation
6. [llama.cpp](https://github.com/ggerganov/llama.cpp)
7. [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding)

### Hardware
8. [Zen 5 AVX-512 Analysis](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/)

---

## Appendix A: Full Benchmark Data

[Link to CSV or embed full tables]

## Appendix B: Reproducibility Checklist

- [ ] llama.cpp version: [commit hash]
- [ ] Model quantization: Q4_K_M
- [ ] NUMA configuration: `numactl --interleave=all`
- [ ] Thread count: 96 (physical cores)
- [ ] Temperature: As specified per test
- [ ] Prompt: [standard prompts used]

---

*This research was conducted in December 2025.*
