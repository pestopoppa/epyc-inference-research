# Research Results Summary

**Last Updated:** 2026-02-03 (Added hard benchmark suites for MemRL mode-advantage)
**System:** AMD EPYC 9655 (96 cores, 1.13TB DDR5), llama.cpp

---

## Hard Benchmark Suites (2026-02-03)

New HuggingFace benchmark adapters added for stronger MemRL mode-advantage signal:

| Suite | Source | Questions | Scoring | Expected Frontdoor |
|-------|--------|-----------|---------|-------------------|
| `gpqa` | ankner/gpqa (Diamond) | 448 | multiple_choice | ~30% (graduate science) |
| `simpleqa` | MAISAAI/openai_simple_qa_test_set | 4,326 | exact_match | ~35% (factual lookup) |
| `hotpotqa` | hotpotqa/hotpot_qa (hard) | 7,405 | f1 (≥0.5) | ~40% (multi-hop QA) |
| `livecodebench` | greengerong/leetcode | 2,360 | code_execution | ~20% (competition code) |
| `debugbench` | Rtian/DebugBench | 4,253 | code_execution | ~40% (bug fixing) |
| `usaco` | codegenning/usacobench_formatted | 520 | code_execution | ~8% (olympiad) |
| `mode_advantage_hard` | Hand-written YAML | 60 | mixed | <50% by design |

**Total:** 19,372 questions across 7 hard suites for overnight MemRL seeding.

**Expected Impact:** Frontdoor pass rate <50%, specialist pass rate >70%, +1.0 comparative rewards in 25-35% of episodes.

See [Chapter 24](../../chapters/24-benchmark-suite-construction.md) for suite construction details.

---

## Best Results

| Configuration | Speed | Speedup | Quality | Use Case |
|---------------|-------|---------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | 12.7x | — | Document QA with source |
| **Qwen2.5-7B + spec (K=24)** | **46.6 t/s** | **2.5x** | 90% | Fast general tasks |
| **Qwen3-Coder-30B-A3B + MoE6 + spec + lookup** | **47.11 t/s** | **1.61x** | — | **Code gen (NEW 2026-02-13)** |
| **Qwen3-VL-4B Q4_K_M** | **18.0 t/s** | — | **94%** | Vision tasks (best quality) |
| Qwen3-VL-30B-A3B + MoE4 | 27.6 t/s | +111% | 75% | Vision tasks (faster, lower quality) |
| Prompt Lookup (code editing) | 25.82 t/s | 8.6x | — | Refactoring, code review |
| **Qwen3-4B-Thinking + spec (K=4)** | **24.2 t/s** | **2.1x** | 88% | Fast thinking |
| Qwen3-Coder-30B-A3B + MoE4 | 22.0 t/s | +83% | **100%** | Code (no spec) |
| **Qwen3-Coder-480B + full + spec (K=16)** | **9.00 t/s** | **1.38x** | — | **480B architect (NEW 2026-02-13)** |
| **Qwen3-235B-A22B + full + spec (K=16)** | **6.08 t/s** | **1.15x** | — | **235B architect (NEW 2026-02-13)** |
| **Qwen2.5-Coder-32B + spec (K=24)** | **21.3 t/s** | **6.3x** | 93% | Code generation |
| **gemma-3-27B + spec (K=16)** | **19.6 t/s** | **8.9x** | 95% | General tasks |
| **gemma-3-12b + spec (K=16)** | **14.8 t/s** | **1.6x** | 97% | General tasks |
| **Qwen3-32B + spec (K=8)** | **12.2 t/s** | **7.6x** | 95% | General tasks |
| MoE Expert Reduction (4-6 experts) | +21-120% | — | — | MoE models |

---

## 🆕 Concurrent Inference Sweep (2026-02-19)

Benchmarked per-role optimal `-np` (parallel slots) using `scripts/benchmark/concurrent_inference_sweep.py`.

| Role | Port | Recommended `-np` | Aggregate TPS Change | p95 Multiplier |
|------|------|--------------------|----------------------|----------------|
| frontdoor (30B MoE) | 8080 | **2** (was 1) | **+121.74%** | 1.33 |
| coder (32B dense) | 8081 | 1 (keep) | — | 1.98 (rejected) |
| worker (7B) | 8082 | 1 (keep) | — | ≥1.505 (rejected) |

**Action**: Frontdoor removed from `SERIAL_ROLES`, now starts with `-np 2`.

---

## 🆕 Parallel Tensor Repacking (2025-12-21)

**Problem:** Model loading on 96-core EPYC was bottlenecked by single-threaded tensor repacking for AVX-512 optimization.

**Solution:** OpenMP parallelization of repack functions in llama.cpp.

| Model Size | Before | After | Speedup |
|------------|--------|-------|---------|
| 6.8GB Q4_K | 5.0s | 3.3s | **1.5x** |
| 19GB Q4_K | 11.9s | 5.3s | **2.2x** |
| 271GB Q4_K | ~150s | ~60s | **~2.5x** |

**Additional finding:** Removing `OMP_NUM_THREADS=1` also improved prompt processing 2.4x (49 → 119 t/s).

**Status:**
- PR submitted: https://github.com/ggml-org/llama.cpp/pull/18239
- Patch applied locally: `patches/llama-cpp-parallel-repack.patch`
- Benchmark scripts updated to use parallel repack

**Affected quant types:** Q4_0, Q4_K, Q2_K, IQ4_NL (Q6_K and Q8_0 don't use repacking)

---

## 🆕 Coder Model Selection (2025-12-16)

**Quality Evaluation Task:** Binary search with docstring and empty array handling.

| Model | Baseline | Optimized | Method | Quality |
|-------|----------|-----------|--------|---------|
| **Qwen3-Coder-30B-A3B-Instruct** | 29.28 t/s | **47.11 t/s** | MoE6 + spec + lookup | ⭐⭐⭐⭐⭐ |
| Qwen3-Coder-53B-A3B-TOTAL-RECALL | 10.3 t/s | 14.0 t/s | MoE 4 experts | ⭐⭐⭐⭐⭐ |
| Qwen2.5-Coder-32B-Instruct | 3.4 t/s | 39.44 t/s | Spec decode K=24 + lookup | ⭐⭐⭐⭐⭐ |
| **Qwen3-Coder-480B-A35B-Instruct** | 6.53 t/s | **9.00 t/s** | Full experts + spec K=16 | ⭐⭐⭐⭐⭐ |

**Finding:** All models produce equivalent quality code. Speed is the only differentiator. jukofyork vocab transplant draft verified for all Qwen3-Coder models (2026-02-13). Architect roles use full experts (no MoE) for maximum quality.

**Decision (updated 2026-02-13):**
- `frontdoor` = Qwen3-Coder-30B-A3B-Instruct (47.11 t/s MoE6+spec+lookup) - fastest
- `coder_escalation` = Qwen2.5-Coder-32B-Instruct (39.44 t/s spec+lookup) - dense fallback
- `architect_coding` = Qwen3-Coder-480B (9.00 t/s full+spec) - ultimate escalation (full quality)

**Coding Escalation Hierarchy:**
```
frontdoor (47 t/s) → coder_escalation (39 t/s) → architect_coding (9 t/s)
```

---

## Summarization Model Comparison (2026-01-26)

**Task:** Executive summary of Twyne V1 Whitepaper (~10K chars context, 20 pages)

| Model | Acceleration | Elapsed | Tokens | t/s | Summary Len | Quality |
|-------|--------------|---------|--------|-----|-------------|---------|
| Qwen2.5-Coder-32B | Spec K=24 | 63.1s | 355 | 5.63 | 1945 chars | Good |
| Qwen3-32B | Spec K=8 | 62.3s | 349 | 5.60 | 2035 chars | Good |
| **Qwen3-Next-80B-A3B** | **MoE4** | 73.8s | 464 | **6.29** | **2560 chars** | **Best** |

**Key Findings:**
- All three non-thinking models produce clean summaries (no `<think>` artifacts)
- **Qwen3-Next-80B-A3B (MoE4) recommended for summarization tasks**:
  - Most comprehensive output (32% more content)
  - Highest detail coverage
  - No speculation needed (SSM incompatible anyway)
- Thinking models (Qwen3-*-Thinking) unusable: output reasoning even with `--reasoning-budget 0`

**Summarization Role Assignment:**
- `ingest_long_context` = Qwen3-Next-80B-A3B + MoE4 (best quality)
- Fallback: Qwen3-32B + spec K=8 (dense, good quality)
- Avoid: Thinking-variant models, coder-specialized models

---

## Very Large Models (100B+)

### Baseline Performance
| Model | Size | Quant | Active Params | Baseline | Notes |
|-------|------|-------|---------------|----------|-------|
| Qwen3-235B-A22B | 133GB | Q4_K_M | ~22B | **3.6 t/s** | MoE, fits in RAM |
| Qwen3-VL-235B-A22B-Thinking | 124GB | Q4_K_S | ~22B | **3.23 t/s** | VL+MoE, thinking variant |
| Qwen3-Coder-480B-A35B | 271GB | Q4_K_M | ~35B | **6.53 t/s** | MoE, largest tested (updated 2025-12-21) |
| GLM-4.6-355B-A32B | 189GB | Q4_K_S | ~32B | **2.24 t/s** | MoE (glm4moe) |
| Qwen3-Next-80B-A3B | 45GB | Q4_K_M | ~3B | **8.43-10.12 t/s** | SSM+MoE hybrid |

### Optimization Results
| Model | Baseline | +Expert Reduction | +Lookup | Best |
|-------|----------|-------------------|---------|------|
| **Qwen3-Coder-480B** | 6.53 t/s | 10.30 t/s (+58% MOE3) | Garbage | **+58%** |
| **Qwen3-VL-235B-Thinking** | 3.23 t/s | 7.12 t/s (+120%) | 3.82 t/s | **+120%** |
| **Qwen3-235B** | 3.6 t/s | 6.75 t/s (+87%) | 6.35 t/s | **+87%** |
| **GLM-4.6-355B** | 2.24 t/s | 3.97 t/s (+77%) | 3.37-3.65 t/s | **+77%** |

**Note (2025-12-21):** Qwen3-Coder-480B results updated after parallel repack fix:
- Architecture: 160 total experts, **8 experts used by default** (35B active params)
- Old baseline (with OMP_NUM_THREADS=1): 2.25 t/s
- New baseline (with parallel repack): 6.53 t/s (+190%)
- **MOE3 (3 experts): 10.30 t/s (+58%) - OPTIMAL** ✓
- MOE4 (4 experts): 9.25 t/s (+42%) - good fallback
- MOE5 (5 experts): 8.50 t/s (+30%)
- MOE2 (2 experts): 11.51 t/s - **GARBAGE output, unusable** ✗

### MoE + Lookup Combination (Detailed)

**Key Finding: SSM models (like Qwen3-Next) are incompatible with speculation-based methods.**

| Model | Hard Mask Alone | Lookup + Hard Mask | Combination Benefit |
|-------|-----------------|--------------------|--------------------|
| **Qwen3-Next-80B-A3B** | 9.8 t/s | ❌ FAILS | SSM incompatible |
| Qwen3-Coder-30B-A3B | 17.7 t/s | 11.6 t/s | 0.66x ❌ |
| Qwen3-VL-30B-A3B | 27.6 t/s | ~20 t/s | 0.72x ❌ |
| Qwen3-235B-A22B | 7.2 t/s | ~6.5 t/s | 0.90x ❌ |

**When to combine vs use standalone:**

| Model Type | Best Approach | Reasoning |
|------------|---------------|-----------|
| **SSM/Hybrid (Qwen3-Next)** | Expert reduction only | Speculation incompatible (consecutive positions) |
| **30B MoE** | MoE6 + spec + lookup | **47.11 t/s** (jukofyork draft, 2026-02-13) |
| **480B MoE** | Full experts + spec (no MoE) | **9.00 t/s** (quality over speed for architect) |
| **235B MoE** | Full experts + spec (no MoE) | **6.08 t/s** (0.6B Q8_0 draft, 53% accept) |

**Commands:**
```bash
# SSM models (Qwen3-Next): Expert reduction only
llama-cli -m Qwen3-Next-80B-A3B.gguf --override-kv qwen3next.expert_used_count=int:4 -t 96

# 30B MoE: Expert reduction + spec + lookup (fastest, 47.11 t/s)
llama-server -m Qwen3-Coder-30B-A3B.gguf -md jukofyork-0.75B-Q8_0.gguf --draft-max 16 --override-kv qwen3moe.expert_used_count=int:6 --lookup -t 96
```

### Qwen3-Next-80B (SSM+MoE Hybrid)

**Architecture:** SSM + MoE hybrid with 512 experts, 10 active by default (~3B active params)

| Configuration | Speed | vs Baseline | Quality |
|---------------|-------|-------------|---------|
| Baseline (10 experts) | 10.12 t/s | — | ✅ |
| **4 experts** | **6.29 t/s** | — | **✅ Best for summarization** |
| 2 experts | 11.55 t/s | +14% | ✅ Good (speed priority) |
| Speculative decoding | ❌ FAILS | — | SSM incompatible |
| Prompt lookup | ❌ FAILS | — | SSM incompatible |

**Performance/Quality trade-off:**
- MoE4: 6.29 t/s - **Recommended for summarization** (most comprehensive output)
- MoE2: 11.55 t/s - Use when speed critical, quality acceptable

**Key insight:** Unlike Qwen3-235B (which produces garbage at 2 experts), Qwen3-Next-80B maintains quality even at 2 experts. This is likely because:
- 512 experts with 2 active still provides reasonable routing options
- SSM component provides additional sequence modeling capacity

### Key Finding: Largest Models Benefit Most
- 480B model: **+48-80% speedup** from expert reduction + lookup
- Expert reduction more effective than speculative decoding on MoE
- All 100B+ models run entirely in RAM (no GPU needed)
- **SSM models:** Expert reduction only - speculation/lookup incompatible

---

## Dense Models (32B-72B)

### Baselines
| Model | Size | Quant | Baseline | Notes |
|-------|------|-------|----------|-------|
| DeepSeek-R1-32B | 18.5GB | Q4_K_M | **6.01 t/s** | Fastest 32B |
| Qwen2.5-Coder-32B | 18.5GB | Q4_K_M | **5.79 t/s** | Code specialist |
| Gemma-3-27B-QAT | 14.5GB | Q4_0 | **4.72 t/s** | QAT quantized |
| Qwen3-32B | 18.4GB | Q4_K_M | **3.67 t/s** | Slower than R1 |
| Meta-Llama-3.1-70B | 40GB | Q4_K_M | **1.96 t/s** | Dense 70B |
| Hermes-4-70B | 40GB | Q4_K_M | **1.73 t/s** | Llama-based |
| DeepSeek-R1-Llama-70B | 40GB | Q4_K_M | **1.73 t/s** | R1 distilled |
| Meta-Llama-3-70B | 40GB | Q4_K_M | **1.72 t/s** | Original Llama 3 |
| Qwen2.5-72B-Instruct | 41GB | Q4_K_M | **1.70 t/s** | Qwen 72B |
| Qwen2.5-Math-72B | 41GB | Q4_K_M | **1.41 t/s** | Math specialist |
| Qwen2.5-72B | 41GB | Q4_K_M | **0.85 t/s** | Base (slow) |

### Speculative Decoding Results (Dense) - Updated 2026-01-16

**Corrected K-sweep benchmarks** (server timing bug fixed):

| Model + Draft | Quality | Baseline | Optimized | Speedup | Best K |
|---------------|---------|----------|-----------|---------|--------|
| **Qwen2.5-7B + Coder-0.5B** | 90% | 18.5 t/s | **46.6 t/s** | **2.5x** | K=24 |
| **Qwen2.5-Coder-32B + qwen25_coder** | 93% | 3.4 t/s | **21.3 t/s** | **6.3x** | K=24 |
| **gemma-3-27B + gemma3** | 95% | 2.2 t/s | **19.6 t/s** | **8.9x** | K=16 |
| **gemma-3-12b + gemma3** | 97% | 9.3 t/s | **14.8 t/s** | **1.6x** | K=16 |
| **DeepSeek-R1-32B + 1.5B** | 94% | 2.0 t/s | **10.1 t/s** | **5.1x** | K=8 |
| **Meta-Llama-3.1-70B + PARD-1B** | 93% | 2.1 t/s | **9.0 t/s** | **4.3x** | K=24 |
| **DeepSeek-R1-7B + PARD-1.5B** | 88% | 10.6 t/s | **9.9 t/s** | **0.9x** | K=4 |
| **DeepSeek-R1-14B-Q6KL + PARD-1.5B** | 98% | 5.1 t/s | **8.5 t/s** | **1.7x** | K=8 |
| **Qwen2.5-Math-72B + 1.5B** | 77% | 2.0 t/s | **7.0 t/s** | **3.5x** | K=24 |
| **Qwen2.5-72B + qwen25** | 91% | 1.9 t/s | **3.0 t/s** | **1.6x** | K=4 |

**Key findings:**
- Large models (70B+): Higher K is better (K=24+), performance still increasing
- Small/medium models: K=8 or K=16 optimal, higher K hurts performance
- MoE models: K=8 is optimal for all tested combinations
- Quality preserved: spec decode is mathematically equivalent to baseline
- Best draft models: qwen2.5-coder-0.5b (Qwen2.5), PARD-Llama-3.2-1B (Llama), PARD-DeepSeek-R1-1.5B (R1)

### Prompt Lookup Results (Dense)
| Model | Summarize | Code | Edit |
|-------|-----------|------|------|
| Qwen2.5-Coder-32B | 6.50 t/s | 4.78 t/s | 4.94 t/s |
| Qwen3-32B | 5.09 t/s | 4.51 t/s | 3.99 t/s |
| DeepSeek-R1-32B | 4.78 t/s | 4.74 t/s | 4.17 t/s |
| Gemma-3-27B | 8.03 t/s | 6.52 t/s | 6.42 t/s |
| Meta-Llama-3.1-70B | 3.15 t/s | 1.67 t/s | 1.76 t/s |
| Hermes-4-70B | 3.72 t/s | 2.54 t/s | 2.76 t/s |
| DeepSeek-R1-Llama-70B | 3.02 t/s | 2.38 t/s | 2.23 t/s |
| Qwen2.5-72B-Instruct | 3.46 t/s | 1.97 t/s | 2.02 t/s |
| Qwen2.5-Math-72B | 2.04 t/s | 0.88 t/s | 0.85 t/s |

---

## MoE Models (30B-A3B Class)

### Baselines (Fastest MoE)
| Model | Quant | Active Params | Baseline | Notes |
|-------|-------|---------------|----------|-------|
| Qwen3-Coder-30B-A3B | Q4_K_M | ~3B | **12.0 t/s** | Code specialist |
| Qwen3-VL-30B-A3B | Q4_K_M | ~3B | **13.1 t/s** | Vision-Language |
| **Qwen3-Coder-53B-A3B** | Q4_K_M | ~3B | **10.3 t/s** | TOTAL-RECALL-v2 finetune (30GB) |
| Qwen3-1.7B (draft) | Q4_K_M | 1.7B | **43.3 t/s** | Draft model |
| Qwen3-VL-2B (draft) | Q4_K_M | 2B | **46.6 t/s** | VL draft |

### Expert Reduction (Hard Mask)
| Model | Baseline | 4 experts | 6 experts |
|-------|----------|-----------|-----------|
| Qwen3-Coder-30B-A3B | 12.0 t/s | **17.7 t/s** | 16.5 t/s |
| Qwen3-VL-30B-A3B | 13.1 t/s | **27.6 t/s** | 25.3 t/s |
| **Qwen3-Coder-53B-A3B** | 10.3 t/s | **14.0 t/s (+36%)** | 12.7 t/s |

### Prompt Lookup (MoE)
| Model | Best Lookup t/s |
|-------|-----------------|
| Qwen3-Coder-30B-A3B-MoE4 | 11.6 t/s |
| Qwen3-30B-A3B-Thinking-MoE4 | 19.0 t/s |

---

## Small Models (7B-14B)

### Baselines
| Model | Size | Quant | Baseline | Notes |
|-------|------|-------|----------|-------|
| Meta-Llama-3-8B | 4.7GB | Q4_K_M | **17.52 t/s** | Fastest 8B |
| Qwen2.5-VL-7B | 4.4GB | Q4_K_M | **15.28 t/s** | VL model |
| DeepSeek-R1-Llama-8B | 4.6GB | Q4_K_M | **13.42 t/s** | R1 distilled |
| DeepSeek-R1-Qwen-7B | 4.4GB | Q4_K_M | **13.15 t/s** | R1 distilled |
| Qwen2.5-Math-7B | 4.4GB | Q4_K_M | **12.44 t/s** | Math specialist |
| Gemma-3-12B | 6.8GB | Q4_K_M | **10.42 t/s** | Medium |
| DeepSeek-R1-Qwen-14B | 8.4GB | Q4_K_M | **6.44 t/s** | Larger R1 |

### Speculative Decoding (Small Models: 4B-14B)
| Model + Draft | Speed | Speedup | Accept | Notes |
|---------------|-------|---------|--------|-------|
| **Qwen3-4B-Thinking + 0.6B** | **24.2 t/s** | **2.1x** | — | K=4 optimal |
| **DeepSeek-R1-14B-Q6_K_L + PARD-1.5B** | **8.5 t/s** | **1.7x** | — | K=8 optimal |
| **Qwen2.5-Math-7B + Coder-0.5B** | **23.5 t/s** | **2.3x** | — | K=16 optimal |
| **gemma-3-12b + gemma3** | **14.8 t/s** | **1.6x** | — | K=16 optimal |
| **DeepSeek-R1-7B + PARD-1.5B** | **9.9 t/s** | **0.9x** | — | K=4, marginal |

### Prompt Lookup (Small)
| Model | Best Lookup t/s |
|-------|-----------------|
| DeepSeek-R1-Qwen-7B | 18.4 t/s |
| Qwen2.5-Math-7B | 18.4 t/s |
| DeepSeek-R1-Llama-8B | 17.3 t/s |
| Meta-Llama-3-8B | 17.1 t/s |
| gemma-3-12b | 10.9 t/s |
| DeepSeek-R1-Qwen-14B | 9.9 t/s |

---

## Key Insights

### 1. Small Drafts Win on CPU
- 0.5B draft at 85 t/s vs 7B draft at 8 t/s
- More speculation rounds beat higher acceptance rates
- **Rule:** Use smallest compatible draft model

### 2. MoE + Speculative Decoding: It Depends
- **Qwen3-VL-30B-A3B MoE4**: 27.6 t/s; with standard spec: ~20 t/s (0.72x slower) -- spec HURTS
- **Qwen3-Coder-30B-A3B MoE6**: 30.84 t/s; with jukofyork spec+lookup: **47.11 t/s** (1.53x faster) -- spec HELPS
- **Qwen3-Coder-480B full experts**: 6.53 t/s; with jukofyork spec: **9.00 t/s** (1.38x faster) -- spec HELPS
- **Qwen3-235B-A22B full experts**: 5.30 t/s; with 0.6B spec: **6.08 t/s** (1.15x faster) -- spec HELPS
- **Key:** jukofyork vocab transplant draft (BOS=comma match) enables high acceptance (70-82%) on Coder family. Standard 0.6B draft gets 53% on 235B. Architect roles use full experts for quality.
- **Policy:** Frontdoor/coder: MoE + spec + lookup (speed). Architects: full experts + spec (quality).

### 3. K-Value Tuning
| Model Size | Optimal K | Reason |
|------------|-----------|--------|
| 7B | K=8 | High baseline, diminishing returns |
| 32B | K=16-24 | Verification cost amortized |
| 72B | K=16 | Balance acceptance vs overhead |

### 4. Temperature Tuning
- Non-zero temperature can improve speculative decoding acceptance rates
- **Rule:** Try temp=0.5-0.7 if acceptance rate is low

---

## Track Status

| Track | Method | Status | Result |
|-------|--------|--------|--------|
| 1 | External Draft | **Production** | 5.9-11x |
| 2 | MoE Expert Reduction | **Production** | +21-48% |
| 8 | Prompt Lookup | **Production** | 8.6-12.7x |
| A | System (Hugepages/NUMA) | **Tested - Already Optimal** | interleave=all best |
| 6 | SuffixDecoding | **= Track 8** | Same as Prompt Lookup |
| C | Draft Quantization | **Tested - No Benefit** | Q8_0 optimal |
| 3 | EAGLE-1 | Deprecated | 0% acceptance |
| 7 | CAS-Spec | Blocked | 0.446% acceptance |

## New Draft Models Available

| Model | Quantization | Size | Path |
|-------|--------------|------|------|
| Qwen2-0.5B | Q2_K | 323MB | `QuantFactory/Qwen2-0.5B-GGUF/Qwen2-0.5B.Q2_K.gguf` |
| Qwen2.5-Coder-1.5B | Q2_K | 645MB | `QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q2_K.gguf` |
| Qwen3-0.6B | Q2_K | 283MB | `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q2_K.gguf` |
| Qwen3-Embedding-0.6B | Q8_0 | — | `Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf` |

**Benchmark Results (2025-12-15):**

| Model | Q2_K Speed | vs Q8_0 |
|-------|------------|---------|
| Qwen3-0.6B | **221 t/s** | 3.4x faster |
| Qwen2-0.5B | **208 t/s** | 2.4x faster |
| Qwen2.5-Coder-1.5B | **98 t/s** | (no Q8_0 baseline) |

**Speculative Decoding Results:**

| Draft Model | Accept | Spec Speed | Verdict |
|-------------|--------|------------|---------|
| Qwen2.5-Coder-0.5B Q8_0 | 58% | **22.5 t/s** | Best (smaller = faster) |
| Qwen2.5-Coder-1.5B Q4_K_M | 58% | 12.5 t/s | Works but slower than 0.5B |
| Qwen2.5-Coder-1.5B Q2_K | 57% | 13.1 t/s | Slower despite faster raw speed |
| Qwen2-0.5B Q2_K | FAIL | — | Wrong vocab family |
| Qwen3-0.6B Q2_K | N/A | — | Wrong model family |

**Conclusion:** Q2_K raw speed gains don't translate to speculative decoding — smaller models still win on CPU.

---

## Quick Commands

```bash
# Track 1: External Draft (5.9-11x)
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative -m TARGET.gguf -md DRAFT.gguf \
  --draft-max 16 -t 96

# Track 2: MoE Expert Reduction (+21-48%)
llama-cli -m MOE.gguf \
  --override-kv ARCH.expert_used_count=int:4 -t 96

# Track 8: Prompt Lookup (8.6-12.7x)
# Use --lookup-ngram-min 3 with prompt containing repeated patterns
```

---

## Failed Approaches (Lessons)

### EAGLE-1 (0% Acceptance)
- Problem: Architecture/checkpoint incompatibility
- Lesson: "Zero-shot" EAGLE requires exact model-checkpoint matching

### CAS-Spec Layer Skip (0.446% Acceptance)
- Problem: Knowledge not evenly distributed across layers
- Lesson: Self-drafting without training produces garbage

### MoE + Speculative Decoding
- Problem: Slower than baseline (0.26-0.84x)
- Lesson: Don't add speculation overhead to already-fast MoE

### SSM/Hybrid Models + Speculation
- Problem: "inconsistent sequence positions" error
- Models affected: Qwen3-Next (SSM+MoE hybrid)
- Lesson: SSM requires consecutive positions - incompatible with ALL speculation methods (speculative decoding, prompt lookup, EAGLE, etc.)

### Qwen3-Coder-480B + Speculative Decoding
- Problem: "draft model special tokens must match target model" error
- BOS token mismatch: Qwen3-Coder-480B has BOS=',' (token 11) vs standard BOS='<|endoftext|>' (token 151643)
- Tested drafts: Qwen3-0.6B, Qwen2.5-Coder-0.5B - both fail
- Lesson: Verify tokenizer compatibility before attempting speculation; unusual BOS tokens block all compatible draft models
- **Workaround:** Use 4-expert reduction (9.25 t/s, +42% vs baseline)
- **Warning:** 2-expert reduction produces GARBAGE output despite higher speed (11.51 t/s)

### Qwen3-Coder-53B-A3B + Speculative Decoding
- Problem: Token mismatch with Qwen2.5 drafts, low acceptance (8.96%) with Qwen3-0.6B
- Tested drafts: Qwen2.5-Coder-0.5B (fails - token mismatch), Qwen3-0.6B (works but 8.96% accept)
- Lesson: MoE models with different distributions don't benefit from small dense drafts
- **Workaround:** Use expert reduction instead (+50% with 4 experts)

### Vanilla Test-Time Training (2026-01-13)
- **Paper**: [TTT-E2E](https://arxiv.org/abs/2512.23675) - End-to-End Test-Time Training for Long Context
- **Goal**: Adapt model weights on context during inference for improved long-context handling
- **Problem**: FFN-only training (required due to missing attention backward pass) failed to improve factual retrieval
- **Benchmark**: Needle-in-haystack at 3.5K tokens
  - Baseline: **Correctly retrieved** secret key
  - TTT (90% training accuracy): **Failed** - hallucinated "not specified in documentation"
- **Additional constraints**:
  - Requires FP32 models (4.6GB for 1B, ~100GB for 7B)
  - llama.cpp training crashes fixed but attention gradients still unavailable
  - TTT-E2E paper used meta-trained checkpoints (not available)
- **Lesson**: Vanilla TTT on off-the-shelf models doesn't improve retrieval; likely needs attention-inclusive training + custom TTT-fine-tuned checkpoints
- **Bug report**: https://github.com/ggml-org/llama.cpp/issues/18805
- **Status**: Research paused - NO-GO

---

## Evaluated But Not Applicable (2026-01-05)

Techniques researched but determined incompatible with our CPU-only llama.cpp stack.

### SpecDiff-2: Discrete Diffusion Drafters
- **Paper**: [arXiv:2511.00606](https://arxiv.org/abs/2511.00606) (November 2025)
- **Claim**: 55% improvement over EAGLE-2, 5.5x speedup using diffusion models as drafters
- **How it works**: Uses DiffuLLaMA/DiffuCoder (discrete diffusion LLMs) as non-autoregressive drafters that generate entire sequences in parallel
- **Why not applicable**:
  - Requires GPU (CUDA, Flash Attention 2)
  - Diffusion LLMs are PyTorch-only, no GGUF/ggml support
  - Drafter models are 7B+ parameters — on CPU, running a 7B diffusion model would be slower than running the target directly
  - Fundamentally different architecture with no conversion path
- **Verdict**: GPU-only technique, cannot be ported to CPU/llama.cpp

### Jacobi Decoding / Lookahead Decoding
- **Source**: [Hao AI Lab](https://hao-ai-lab.github.io/blogs/jacobi-forcing/)
- **Claim**: Parallel token generation without draft model, 1.5-2.3x speedup
- **How it works**: Treat AR decoding as solving nonlinear equations; initialize future positions with guesses, refine in parallel until convergence
- **Why not applicable**:
  - Each Jacobi step processes N tokens simultaneously → requires GPU parallelism
  - On CPU, processing N tokens = N× sequential compute (no speedup)
  - Related techniques (Consistency LLMs, Jacobi Forcing) require model retraining
  - All implementations are GPU-only (vLLM, custom CUDA)
- **Verdict**: Parallel verification only benefits GPU; no CPU implementation exists

### Consistency LLMs (CLLMs)
- **Paper**: [CLLMs](https://arxiv.org/html/2403.00835v1)
- **Claim**: 2.4-3.4x speedup via parallel n-token decoding
- **Why not applicable**:
  - Requires training/finetuning the model (we use pretrained GGUFs)
  - Relies on GPU parallel compute for the parallel decoding benefit
- **Verdict**: Training-dependent + GPU-only

### Mixture-of-Recursions (MoR)
- **Paper**: [arXiv:2507.10524](https://arxiv.org/abs/2507.10524) (NeurIPS 2025)
- **Claim**: 2× inference throughput via recursive transformers with per-token adaptive depth
- **How it works**: Single weight-tied block reused across recursion steps; lightweight routers assign different recursion depths per token
- **Why not applicable**:
  - Requires models trained from scratch with MoR architecture (cannot convert existing models)
  - No GGUF support — official implementation is PyTorch + Flash Attention 2 only
  - Only 360M parameter research models available (no production-scale models)
  - GPU-only (training: 4× H100/A100, inference assumes CUDA)
  - Our existing methods already provide 5.5-12.7× speedups vs MoR's 2×
- **Verdict**: Training-time architecture, no conversion path, inferior gains

### Summary: CPU Speculation Limits

All high-performing modern speculation techniques rely on GPU parallelism:

| Technique | Parallelism Required | CPU Viable |
|-----------|---------------------|------------|
| External Draft (our Track 1) | None — sequential | ✅ **YES** |
| Prompt Lookup (our Track 8) | None — n-gram matching | ✅ **YES** |
| MoE Reduction (our Track 2) | None — fewer experts | ✅ **YES** |
| SpecDiff-2 (diffusion drafter) | GPU matrix ops | ❌ No |
| Jacobi/Lookahead | GPU parallel forward | ❌ No |
| EAGLE-2 | GPU + training | ❌ No |
| Medusa | GPU + training | ❌ No |
| CLLMs | GPU + training | ❌ No |
| MoR (recursive transformer) | Training-time arch | ❌ No |

**Conclusion**: For CPU inference, our current production tracks (external draft, prompt lookup, MoE reduction) represent the practical ceiling. Novel GPU-parallel techniques cannot be ported without fundamental architectural changes.

---

## 🆕 CPU Optimization R&D (2026-01-05)

Following the question "could we develop the architectural changes ourselves?", we've identified and evaluated potential CPU-specific optimizations.

### Research Tracks

| Track | Priority | Status | Expected Gain | Notes |
|-------|----------|--------|---------------|-------|
| **B: Tree Speculation** | HIGH | Ready to test | 10-30% | Already in llama.cpp, K-sweep in progress |
| **A: T-MAC** | MEDIUM | Cloned | 2-4× (uncertain) | x86 gains uncertain per README |
| **D: AVX-512 Kernels** | LOW | Devc handoff ready | 20-50% | Autonomous agent development |
| **C: Multi-Draft Parallel** | BLOCKED | Needs BIOS | 30-50% | Requires NPS4 (only 2 NUMA nodes) |

### Key Findings

1. **T-MAC (Lookup Table Inference)**
   - Location: `/mnt/raid0/llm/T-MAC/`
   - Problem: Requires model reconversion (existing Q4_K_M GGUFs not compatible)
   - Warning: "Cannot guarantee significant speedup on x86 platforms" (README)
   - Best gains at 1-2 bit (quality tradeoff)

2. **NUMA Topology**
   - Actual: 2 nodes (NPS1), not 8 as assumed
   - Each node: 48 cores + ~567GB RAM
   - Multi-draft parallel blocked without BIOS change to NPS4

3. **Tree Speculation**
   - Already in llama.cpp via `--draft-max` with tree sampling
   - Current benchmark already sweeping K=4,8,16,24,32

### Documents

- Full R&D Plan: `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`
- Findings: `/mnt/raid0/llm/claude/research/cpu_optimization_findings.md`
- Kernel Dev Handoff: `/mnt/raid0/llm/claude/research/kernel_dev_handoff.md`

---

## Benchmark Framework (2025-12-16)

### 8 Quality Benchmark Suites

| Suite | Purpose | Auto-Scoring |
|-------|---------|--------------|
| **Thinking** | Chain-of-thought, multi-step reasoning | Manual |
| **Coder** | Code generation, debugging, refactoring | Manual |
| **VL** | Vision-language (OCR, image understanding) | Manual |
| **General** | Instruction following, summarization | Manual |
| **Agentic** | Tool calling, function extraction | Partial |
| **Math** | Mathematical reasoning, step verification | Partial |
| **Long Context** | Information retrieval (4K-50K tokens) | Auto |
| **Instruction Precision** | Exact format compliance | **Full Auto** |

### Permanent Storage

```
benchmarks/
├── prompts/v1/          # Versioned YAML prompt files
│   ├── thinking.yaml
│   ├── coder.yaml
│   ├── vl.yaml
│   ├── general.yaml
│   ├── agentic.yaml
│   ├── math.yaml
│   ├── long_context.yaml
│   └── instruction_precision.yaml
├── results/
│   ├── runs/            # Raw outputs per run with metadata
│   └── index.jsonl      # Structured index for querying
└── baselines/           # Reference checkpoints
```

### Commands

```bash
# Run all 8 suites
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all

# Run specific suite
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision

# Compare runs
./scripts/benchmark/compare_results.sh --baseline ID --current ID

# List all runs
./scripts/benchmark/compare_results.sh --list-runs
```

### Why Instruction Precision Matters for Orchestration

Models that fail instruction precision tests will break orchestration:
- "Output only JSON" → model adds "Here's the JSON:" → **parsing failure**
- "Exactly 3 items" → model gives 4 → **schema validation failure**
- "Do not mention X" → model mentions X → **context pollution**

**Orchestration readiness thresholds:**
- Workers: T1 100%, T2 75%+
- Orchestrators: T1 100%, T2 100%, T3 75%+

---

## Claude-as-Judge Quality Review (2025-12-18, Updated 2026-01-19)

### The Scoring Problem: Ceiling Effects

Our original scoring methodology used absolute scores (0-3 per question → binary pass/fail). This caused **ceiling effects** where radically different models scored identically:

| Model Size | Old Score | Reality |
|------------|-----------|---------|
| Qwen2.5-0.5B | 10/10 (100%) | Basic, often wrong |
| Qwen3-235B | 10/10 (100%) | Expert-level reasoning |

A 0.5B model and a 235B model both scoring "100%" meant we couldn't differentiate quality. The benchmark questions were too easy, and the scoring was too generous.

### New Methodology: Relative Scoring (2026-01-19)

**Solution:** Score each response on a **0-100 scale per suite**, comparing against:
1. Reference answers from benchmark YAML files
2. Other model responses for the same question

| Score | Meaning |
|-------|---------|
| **90-100** | Matches or exceeds reference answer quality |
| **70-89** | Correct with good reasoning, minor gaps |
| **50-69** | Partially correct, significant omissions |
| **30-49** | Wrong approach but reasonable attempt |
| **0-29** | Empty, garbage, or completely wrong |

**Result:** Clear differentiation between model tiers. A 4.5+ point gap now separates model size classes (0.5B vs 7B vs 32B vs 70B+).

### Key Findings from Rescore

**72 models evaluated across 8 benchmark suites:**

1. **Size matters again:** Larger models consistently outscore smaller ones when relative scoring is applied
2. **MoE models excel:** Qwen3-235B-A22B achieves 94% — the best large model result
3. **Small thinking models punch above weight:** MathSmith-8B (93.7%) nearly matches 70B+ models
4. **Instruction precision is hard:** No model exceeds 78/110 on strict format compliance
5. **Smaller VL models outperform larger:** Qwen3-VL-4B (94%) beats 30B (75%) and 235B (56%) on figure analysis

### Master Benchmark Scores (Relative Scoring)

**Top 25 Models by Overall Score:**

| Model | Think | Gen | Math | Agent | Code | Inst | Long | VL | Total | Pct | t/s |
|-------|-------|-----|------|-------|------|------|------|-----|-------|-----|-----|
| Qwen3-235B-A22B | - | 94 | - | - | 94 | - | - | - | 188/200 | **94.0%** | 5.8 |
| MathSmith-Qwen3-8B.Q4 | 93 | 95 | 93 | - | - | - | - | - | 281/300 | **93.7%** | 16.2 |
| GLM-4.6 | - | - | 95 | - | 35/40 | - | - | - | 130/140 | **92.9%** | 3.1 |
| DeepSeek-R1-Llama-8B | 86 | 91 | 96 | 93 | - | - | - | - | 366/400 | **91.5%** | 9.4 |
| MathSmith-Qwen3-8B.Q8 | 93 | - | 88 | - | - | - | - | - | 181/200 | **90.5%** | 11.5 |
| DeepSeek-R1-Qwen-7B | 87 | 89 | 96 | 89 | - | - | - | - | 361/400 | **90.2%** | 16.9 |
| Qwen3-Coder-480B | - | 90 | - | - | 87 | - | - | - | 177/200 | **88.5%** | 6.0 |
| Qwen3-4B-Thinking | 80 | 79 | 95 | 98 | - | - | - | - | 352/400 | **88.0%** | 11.5 |
| Qwen3-32B | 93 | 86 | 196/200 | 100 | 95 | 54/110 | - | - | 624/710 | **87.9%** | 3.2 |
| Hermes-4-70B | 95 | 86 | 190/200 | 98 | 91 | 55/110 | - | - | 615/710 | **86.6%** | 2.9 |
| gemma-3-27B-QAT | 90 | 91 | 93 | 100 | 85 | 58/110 | 27/30 | - | 544/640 | **85.0%** | 2.2 |
| DeepSeek-R1-Qwen-32B | 92 | 95 | 95 | 100 | 92 | 43/110 | - | - | 517/610 | **84.8%** | 2.0 |
| Qwen3-30B-A3B-Think | 80 | 90 | 180/200 | 98 | 91 | 60/110 | - | - | 599/710 | **84.4%** | 24.2 |
| gemma-3-12b | 92 | 94 | 94 | 95 | 81 | 54/110 | - | - | 510/610 | **83.6%** | 9.4 |
| DeepSeek-R1-0528-Qwen3-8B | 93 | 88 | 96 | 93 | 86 | 52/110 | - | - | 508/610 | **83.3%** | 13.3 |
| Qwen2.5-Coder-32B | 90 | 93 | 170/200 | 100 | 90 | 47/110 | - | - | 590/710 | **83.1%** | 8.4 |
| Qwen2.5-72B-Instruct | 90 | 83 | 90 | 94 | 83 | 76/110 | 32/50 | - | 548/660 | **83.0%** | 3.3 |
| xLAM-2-1B-fc-r | - | 83 | - | - | - | - | - | - | 83/100 | **83.0%** | 50.4 |
| Meta-Llama-3.1-70B | 81 | 95 | 190/200 | 100 | 73 | 49/110 | - | - | 588/710 | **82.8%** | 8.9 |
| Qwen3-VL-235B-A22B-Think | - | - | - | - | - | - | - | 56 | 56/100 | **56%** | 4.6 |

**Draft Models (Pareto Frontier: Quality vs Speed):**

| Model | Score | Speed | Notes |
|-------|-------|-------|-------|
| Qwen3-1.7B-Q8_0 | **82%** | 36 t/s | Best quality small draft |
| gemma-3-1b-Q8_0 | **76.5%** | 114 t/s | Best quality/speed balance |
| Qwen2.5-0.5B.Q8_0 | **65%** | 157 t/s | Fastest reasonable draft |
| PARD-Qwen3-0.6B | **63.5%** | 82 t/s | PARD optimization |
| Qwen2.5-Coder-0.5B | **59%** | 142 t/s | Coder family draft |

### Top Performers by Category

| Category | Model | Score | Speed | Notes |
|----------|-------|-------|-------|-------|
| **Architect** | Qwen3-235B-A22B | 94.0% | 5.8 t/s | Best overall quality |
| **Thinking** | MathSmith-Qwen3-8B | 93.7% | 16.2 t/s | Best reasoning |
| **Math** | DeepSeek-R1-Llama-8B | 96/100 | 9.4 t/s | Best math reasoning |
| **Agentic** | Qwen3-4B-Thinking | 98/100 | 11.5 t/s | Best tool calling |
| **Coder** | Qwen3-235B-A22B | 94/100 | 5.8 t/s | Best code generation |
| **General** | MathSmith-Qwen3-8B | 95/100 | 16.2 t/s | Best general tasks |
| **Vision** | Qwen3-VL-4B | 94% (34/36) | 18 t/s | Best VL model (2026-01-27 valid benchmark) |
| **Fast Draft** | Qwen2.5-0.5B.Q8_0 | 65% | 157 t/s | Speed optimized |
| **Quality Draft** | Qwen3-1.7B-Q8_0 | 82% | 36 t/s | Quality optimized |

### Critical Issues Discovered

1. **Meta-Llama-3.1-8B.Q4_K_S: 49.7%** - Repetitive degeneration on hard questions. Avoid for complex reasoning.
2. **Phi-4-reasoning-plus: 49.3-49.5%** - Despite "reasoning" name, scores below 50%. Poor quality.
3. **Qwen3-VL models: 0% agentic** - All Qwen3-VL models return empty tool calls. Use Qwen2.5-VL-7B for tool-using vision.
4. **Instruction precision ceiling: ~70%** - No model reliably follows strict format constraints.

### Production Model Configuration (2026-01-20)

Complete production model lineup with relative scoring validation.

#### HOT Tier (~45GB) - Always Resident

| Role | Model | Score | Speed | Configuration | Size |
|------|-------|-------|-------|---------------|------|
| **frontdoor** | Qwen3-Coder-30B-A3B-Instruct | 89.5% | 47.11 t/s | MoE6 + spec + lookup | 20GB |
| **coder_escalation** | Qwen2.5-Coder-32B | 91.5% | 39.44 t/s | spec K=24 + lookup | 18.5GB |
| **worker** | Qwen2.5-7B-Instruct | 74.5% | 50 t/s | spec K=16 + draft | 4.4GB |
| **voice_server** | faster-whisper large-v3-turbo | — | 2.8x RT | CPU int8, port 9000 | 4GB |

**HOT Draft Models:** Qwen2.5-Coder-0.5B-Q8_0 (76%, 142 t/s), Qwen2.5-0.5B-Q8_0 (80%, 157 t/s)

#### WARM Tier (~470GB) - Load on Demand

| Role | Model | Score | Speed | Configuration | Size |
|------|-------|-------|-------|---------------|------|
| **architect_general** | Qwen3-235B-A22B | 94.0% | 6.75 t/s | MoE4 experts | 133GB |
| **architect_coding** | Qwen3-Coder-480B | 88.5% | 10.3 t/s | MoE3 experts | 271GB |
| **ingest_long_context** | Qwen3-Next-80B-A3B | 77.0% | 8 t/s | MoE2 (SSM, NO SPEC!) | 46GB |
| **vision_qwen3_vl_4b** | Qwen3-VL-4B-Instruct | **94% VL** | 18 t/s | mmproj required | 2.3GB |

#### Vision Models (Valid 2026-01-27 Benchmark)

| Role | Model | VL Score | Speed | Tier | Use Case |
|------|-------|----------|-------|------|----------|
| **vision_qwen3_vl_4b** | Qwen3-VL-4B | **94% (34/36)** | 18 t/s | **HOT** | Document figures (RECOMMENDED) |
| worker_vision_agentic | Qwen2.5-VL-7B | 81% (29/36) | 17 t/s | WARM | Tool-using vision (only agentic VL) |
| vision_qwen3_vl_8b | Qwen3-VL-8B | 86% (31/36) | 15 t/s | WARM | Alternative escalation |
| **vision_escalation** | Qwen3-VL-30B-A3B | 75% (27/36) | 27 t/s | COLD | Manual request only |

**Key Findings:**
- 4B outperforms 30B/235B due to no timeout truncation and accurate OCR
- Qwen3-VL = 0% agentic (no tool calls) - use Qwen2.5-VL-7B for agentic vision
- Use document summary context (~8K chars) for optimal figure analysis

#### Formalizers

| Role | Model | Score | Speed | Purpose |
|------|-------|-------|-------|---------|
| **formalizer** | MathSmith-Qwen3-8B.Q8_0 | 95.0% | 14 t/s | Problem formalization |
| **tool_formalizer** | xLAM-2-1B-fc-r | 83.0% | 50.4 t/s | Function calling / tools |
| **document_formalizer** | LightOnOCR-2-1B-bbox-Q4_K_M | N/A | **0.17 pg/s** (8×12t) | PDF/document OCR with bbox |

#### Memory Budget

| Tier | RAM | Status |
|------|-----|--------|
| HOT | ~45GB | Always resident |
| WARM | ~470GB | Load 2-3 at a time |
| Headroom | ~615GB | Context/KV cache |
| **Total** | **1.13TB** | ✅ Fits |

**Full benchmark data:** `benchmarks/results/reviews/summary_relative.csv` (148 configs, 72+ models)

### Global Role Recommendations (Updated 2026-01-06)

> **See also:** [ESCALATION_FLOW.md](ESCALATION_FLOW.md) for comprehensive escalation diagrams, trigger mechanisms, and deprecation list.

**Memory Budget:** 1.13 TB available | Hot Pool: ~35 GB | Warm Pool: ~460 GB | Headroom: 634 GB

---

#### Memory Pool Configuration

**HOT POOL (~35 GB) - Always Resident:**
| Model | Size | Speed | Purpose |
|-------|------|-------|---------|
| frontdoor (Qwen3-Coder-30B + MoE6+spec+lookup) | 20 GB | 47.11 t/s | Orchestrator (always on) |
| draft_qwen25_coder (0.5B Q4_K_M) | 0.4 GB | 142 t/s | Draft for Qwen2.5-Coder family |
| draft_qwen25 (0.5B Q8_0) | 0.5 GB | 157 t/s | Draft for Qwen2.5 family |
| draft_pard_llama (1B Q4_0) | 0.9 GB | 76 t/s | Draft for Llama family |
| draft_r1_distill (1.5B Q8_0) | 1.8 GB | 59 t/s | Draft for DeepSeek R1 family |
| worker_general (Llama-3-8B) | 4.7 GB | 14.7 t/s | Boilerplate, rewrites |
| worker_math (Qwen2.5-Math-7B) | 4.4 GB | 23.5 t/s | Edge cases (with spec) |
| toolrunner (Llama-3-8B) | 4.7 GB | 14.7 t/s | Log triage, tool output |

**WARM POOL (~460 GB) - Load 2-3 Based on Task:**
| Model | Size | Quality | Speed | Acceleration | Best For |
|-------|------|---------|-------|--------------|----------|
| **architect_qwen2_5_72b** | 44 GB | **91%** | 3.0 t/s | spec K=4 | General architecture |
| **architect_meta_llama_3_1_70b** | 40 GB | **93%** | 9.0 t/s | spec K=24 | Highest quality design |
| **math_qwen2_5_math_72b** | 44 GB | **92%** | 7.0 t/s | spec K=24 | Math reasoning |
| **worker_summarize** | 18 GB | **93%** | 21.3 t/s | spec K=24 | Document summarization |
| **ingest_qwen2_5_coder_32b** | 18 GB | 93% | 21.3 t/s | spec K=24 | Fast ingest/code |
| **thinking_deepseek_r1_14b** | 8 GB | **98%** | 8.5 t/s | spec K=8 | Chain-of-thought |
| **thinking_reasoning** (Next-80B) | 45 GB | **99%** | 9.8 t/s | MoE4 | Deep reasoning |
| **ingest_long_context** (Next-80B) | 45 GB | **99%** | 9.8 t/s | MoE4 | Very long docs |
| **architect_general** (235B) | 134 GB | 91% | 6.08 t/s | Full+spec | System design (full quality) |
| vision_escalation (30B-A3B) | 18 GB | - | 27.6 t/s | MoE4 | Complex vision |
| architect_coding (480B) | 272 GB | **94%** | 9.00 t/s | Full+spec | Ultimate escalation (full quality) |

---

#### Production Role Assignments

> **Note:** Quality percentages from master benchmark table (2026-01-11). Speeds for spec decode configs from separate K-sweep testing.

**Tier A: Frontdoor (Interactive, Low Latency)**

| Priority | Model | Quality | Speed | When to Use |
|----------|-------|---------|-------|-------------|
| **PRIMARY** | **Qwen3-Coder-30B-A3B + MoE6+spec+lookup** | **90%** | 47.11 t/s | Default for all routing ⭐ |
| FAST | Qwen3-Coder-30B-A3B + MoE4 | 89% | 17.7 t/s | When speed > quality (no draft) |

**Note (2026-02-13):** MoE6+spec+lookup is now the production config (2.58x over baseline). jukofyork vocab transplant draft verified. MoE2 is BROKEN (0%).

**Tier B: Architects (Quality > Speed)**

| Role | Priority | Model | Quality | Speed | When to Use |
|------|----------|-------|---------|-------|-------------|
| **architect_coding** | 1 | Qwen3-Coder-480B + MoE4 | 94% | 6.6 t/s | Final escalation for complex code |
| | 2 | Qwen3-235B + MoE4 | 91% | 7.2 t/s | General architecture fallback |
| **architect_general** | 1 | **Qwen2.5-72B + spec K=4** | **91%** | **3.0 t/s** | **High-quality design** |
| | 2 | Meta-Llama-3.1-70B + spec K=24 | **93%** | 9.0 t/s | Highest quality |
| | 3 | Qwen3-235B + MoE4 | 91% | 7.2 t/s | Large MoE (no spec) |

**Tier B: Thinking/Reasoning**

| Priority | Model | Quality | Speed | When to Use |
|----------|-------|---------|-------|-------------|
| 1 | **Qwen3-Next-80B-Thinking + MoE4** | **99%** | 9.8 t/s | **Deep multi-step reasoning** |
| 2 | DeepSeek-R1-Distill-Qwen-14B-Q6_K_L + spec K=8 | 98% | 8.5 t/s | Fast reasoning with spec decode |
| 3 | DeepSeek-R1-Distill-Qwen-32B | 94% | 2.0 t/s | Large reasoner (no spec) |
| 4 | DeepSeek-R1-Distill-Llama-8B | 88% | 9.4 t/s | Small fast reasoner |
| 5 | Qwen3-4B-Thinking-2507 + spec K=4 | 88% | 24.2 t/s | Tiny fast reasoner |

**Tier B: Math Specialist**

| Priority | Model | Quality | Speed | When to Use |
|----------|-------|---------|-------|-------------|
| 1 | **Qwen2.5-Math-72B + spec** | 92% | **7.0 t/s** | **Production math** |
| 2 | Qwen2.5-Math-7B + spec | 90% | 23.5 t/s | Fast math with spec |

**Tier B: Ingest/Long Context**

| Priority | Model | Quality | Speed | When to Use |
|----------|-------|---------|-------|-------------|
| 1 | **Qwen2.5-Coder-32B + spec** | 93% | **21.3 t/s** | **Fast bulk ingest** ⭐ |
| 2 | Meta-Llama-3.1-70B + spec K=24 | 93% | 9.0 t/s | Higher quality ingest |
| 3 | Qwen3-Next-80B + MoE4 (SSM) | 99% | 9.8 t/s | Very long context (128K+) |

**Tier C: Workers (Speed > Quality)**

| Role | Model | Quality | Speed | Acceleration |
|------|-------|---------|-------|--------------|
| **worker_summarize** | Qwen2.5-Coder-32B | **93%** | **21.3 t/s** | spec K=24 ⭐ |
| worker_general | Llama-3-8B | 90% | 14.7 t/s | baseline |
| worker_math | Qwen2.5-Math-7B | 90% | 23.5 t/s | spec K=16 |

**Tier D: Draft Models (Spec Decode Targets)**

| Family | Draft Model | Quality | Speed | Compatible With |
|--------|-------------|---------|-------|-----------------|
| **Qwen2.5** | qwen2.5-coder-0.5B Q4_K_M | 55% | 142 t/s | Qwen2.5-Coder-32B, Qwen2.5-72B |
| **Qwen2.5** | qwen2.5-0.5B Q8_0 | 40% | 157 t/s | Qwen2.5-72B, Qwen2.5-Math-72B |
| **Llama** | PARD-Llama-3.2-1B Q4_0 | 95% | 76 t/s | Meta-Llama-3.1-70B, Llama-3-8B |
| **DeepSeek** | R1-Distill-Qwen-1.5B Q8_0 | 95% | 59 t/s | DeepSeek-R1-Distill-Qwen-32B |
| **Qwen3** | PARD-Qwen3-0.6B Q4_0 | 95% | 82 t/s | Qwen3-4B-Thinking |

---

#### Quick Reference: Best Config Per Task

| Task Type | Model + Config | Quality | Speed |
|-----------|----------------|---------|-------|
| **Code generation** | Qwen2.5-Coder-32B + spec K=24 | 93% | 21.3 t/s |
| **Document summary** | Qwen2.5-Coder-32B + spec K=24 | **93%** | 21.3 t/s |
| **Math reasoning** | Qwen2.5-Math-72B + spec K=24 | **92%** | 7.0 t/s |
| **Architecture design** | Qwen2.5-72B + spec K=4 | 91% | 3.0 t/s |
| **High-quality design** | Llama-3.1-70B + spec K=24 | **93%** | 9.0 t/s |
| **Fast reasoning** | DeepSeek-R1-14B-Q6_K_L + spec K=8 | **98%** | 8.5 t/s |
| **Complex reasoning** | Qwen3-Next-80B + MoE4 | **99%** | 9.8 t/s |
| **Long context** | Qwen3-Next-80B + MoE4 | **99%** | 9.8 t/s |
| **Ultimate escalation** | Qwen3-Coder-480B + MoE4 | 94% | 6.6 t/s |

---

#### Deprecated/Avoid

| Model | Reason |
|-------|--------|
| coder_escalation (Qwen3-53B baseline) | 38% quality, repetition loops |
| architect_meta_llama_3_70b | 33% quality (use 3.1 instead) |
| GLM-4.6-355B | 59% quality, slow |
| **GLM-4.7-Flash** | **43% quality, SEVERE repetition/degeneration loops** |
| Qwen3-VL-* (all sizes) | 0% agentic - empty tool calls |
| MathSmith-Hard-Problem-Synthesizer | 5x slower than expected |

⚠️ **VL BENCHMARK INVALIDATION (2025-01-06):** Previous VL scores were INVALID. **FIXED 2026-01-27:** VL benchmark re-run with proper image passing using `llama-mtmd-cli`. See valid results below.

---

## ✅ VL Benchmark Results (2026-01-27) - VALID

**Benchmark:** Hardened VL suite with 12 questions (2x T1, 5x T2, 5x T3) using OCRBench, DocVQA, ChartQA, and Twyne whitepaper images.

**Key Findings:**
1. All models 4B-8B got basic OCR correct ("Centre")
2. 235B models had OCR regression ("Centie") AND timeout truncation
3. Score differences between 4B/7B/8B are marginal and may reflect chart interpretation rather than capability

### Raw Scores (Timeout-Penalized)

| Model | VL Score | Pct | Avg t/s | Notes |
|-------|----------|-----|---------|-------|
| **Qwen3-VL-4B Q4_K_M** | **34/36** | **94%** | 18.0 | ✅ Best raw score, accurate bar identification |
| **Qwen3-VL-4B Q8_0** | **34/36** | **94%** | 14.8 | ✅ Same quality as Q4_K_M |
| Qwen3-VL-8B Q4_K_M | 31/36 | 86% | 15.4 | ✅ Good quality, minor chart ID differences |
| Qwen3-VL-8B Q8_0 | 31/36 | 86% | 9.5 | ✅ Same quality, slower |
| Qwen2.5-VL-7B | 29/36 | 81% | 17.2 | ✅ Good general VL performance |
| Qwen3-VL-30B-A3B | 27/36 | 75% | 19.0 | ⚠️ "Centric" OCR error |
| Qwen3-VL-30B-A3B MoE4 | 27/36 | 75% | 27.6 | ⚠️ Same quality, +45% faster |
| Qwen3-VL-235B-A22B | 20/36 | 56% | 4.6 | ⚠️ "Centie" OCR + timeout truncation |
| Qwen3-VL-235B-A22B MoE4 | 19/36 | 53% | 6.7 | ⚠️ Timeout truncation |

### Quality-Adjusted Scores (Partial Output Assessment)

The 235B models were truncated by timeout, but their **partial output quality was HIGH**:
- t3_q1: "0.57" correct + excellent chart critique methodology
- t3_q3: Comprehensive DeFi security audit with 5/10 rating justification
- t3_q4: Detailed Nash equilibrium analysis, game theory, bank run scenarios

| Model | Quality-Adj Score | Pct | Notes |
|-------|-------------------|-----|-------|
| **Qwen3-VL-4B** | 34/36 | 94% | No truncation, accurate readings |
| **Qwen3-VL-8B** | 31/36 | 86% | No truncation, minor chart ID diff |
| **Qwen3-VL-235B** | ~30/36 | ~83% | Partial outputs HIGH quality, penalized by timeout |
| Qwen3-VL-30B-A3B | 27/36 | 75% | OCR error on basic text ("Centric") |

**VL Role Recommendations:**
- `worker_vision` = **Qwen3-VL-4B Q4_K_M** (94%, 18 t/s) - Best quality/speed ratio
- `vision_escalation` = **Qwen3-VL-8B Q4_K_M** (86%, 15.4 t/s) - Good balance
- For complex analysis: **Qwen3-VL-235B** with extended timeouts (excellent reasoning quality)

**Analysis Notes:**
1. **4B vs 7B/8B margin is narrow**: Difference is in chart legend interpretation, not core VL capability
2. **235B timeout issue**: With longer timeouts, 235B would likely score 83%+ (quality of partial outputs is excellent)
3. **30B OCR regression**: "Centric" instead of "Centre" - genuine model issue, not timeout
4. **Basic OCR**: All 4B/7B/8B models read "Centre" correctly; 30B+ had errors

---

⚠️ **Qwen3-VL Warning:** All Qwen3-VL models (2B/4B/8B) score 0% on agentic tasks - all tool-call prompts return empty. Use Qwen2.5-VL for vision tasks requiring tool coordination.

⚠️ **DeepSeek-R1-0528-Qwen3-8B Warning:** 100% thinking accuracy but only 24% instruction precision. Model echoes prompts and outputs verbose reasoning instead of clean structured output. Unsuitable for orchestration tasks requiring exact format compliance.

⚠️ **MathSmith-Hard-Problem-Synthesizer-Qwen3-8B Warning:** Runs at 3.5 t/s instead of expected 15+ t/s for an 8B model (5x slower). Uses only 6% of memory bandwidth - severely compute-bound. Likely mradermacher GGUF conversion issue or hidden architecture differences. Use Qwen2.5-Math-7B-Instruct (11.3 t/s) instead.

`*` = scored on old (easier) questions before hardening
`†` = incomplete benchmark (partial test run)
Scores without markers = tested on complete 60-120 question suites (2025-12-19)

### Complete Model Database with Claude Scores

All models from registry (~70 unique models). Empty cells = not yet benchmarked. `Baseline t/s` = raw benchmark speed.

**Note:** Scores marked with `*` were evaluated on the OLD (easier) benchmark questions before hardening. Scores without `*` are on the new post-doctoral level T3 questions.

#### Tier A-B: Production & Specialist Models (MoE/Hybrid Architecture)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | VL | Pct | Baseline t/s | Optimized t/s (config) |
|-------|------|----------|---------|------|---------|-------|-----------|-----|-----|--------------|------------------------|
| **Qwen3-Coder-480B-A35B** | architect_coding | 28/30 | - | - | 28/30 | 27/27 | - | - | **95%**† | 6.53 | (8 experts default) |
| Qwen3-Coder-480B-A35B (MoE 2) | architect_coding | 3/30 | - | - | 1/30 | 10/30 | - | - | **14%** | 7.6 | GARBAGE output ❌ |
| **Qwen3-Coder-480B-A35B (MoE 3)** | architect_coding | - | - | - | - | - | - | - | - | 10.30 | **+58% OPTIMAL ✓** |
| Qwen3-Coder-480B-A35B (MoE 4) | architect_coding | 27/30 | - | - | 30/30 | 30/30 | - | - | **88%** | 7.1 | +9% |
| Qwen3-Coder-480B-A35B (MoE 5) | architect_coding | - | - | - | - | - | - | - | - | 8.50 | +30% |
| Qwen3-Coder-480B-A35B (MoE 6) | architect_coding | - | 29/30 | - | 30/30 | - | - | - | **95%**‡ | 6.2 | -5% |
| **GLM-4.6-355B-A32B** | general | 24/30 | - | - | 12/30 | 5/6 | 18/33 | - | **59%** | 3.4 | Lower quality than Qwen |
| GLM-4.6-355B-A32B (MoE 2) | general | - | - | - | - | - | - | - | - | - | - |
| GLM-4.6-355B-A32B (MoE 4) | general | - | - | - | - | - | - | - | - | 3.97 | 3.5 (+lookup) ❌ |
| GLM-4.6-355B-A32B | ingest | 29/30 | - | - | 13/30 | - | 19/33 | - | **60%** | 3.7 | ~60% across roles |
| ~~**GLM-4.7-Flash**~~ | general | 3/10 | 6/10 | 7/10 | 4/10 | 5/10 | 1/11 | - | **43%** ⚠️ | 19.14 | AVOID - severe repetition/degeneration |
| **MiniMax-M2.1-Q4_K_M** | general | 8/10 | 8/10 | 9/10 | 9/10 | 8/10 | 4/11 | - | **75%** | 8.87 | Good reasoning, prompt repetition |
| MiniMax-M2.1-Q6_K | general | 8/10 | 8/10 | 9/10 | 8/10 | 8/10 | 4/11 | - | **74%** | 8.20 | Similar to Q4, slightly slower |
| Qwen3-VL-235B-A22B | vision | - | - | - | - | - | - | 56% | **56%** | 4.6 | Valid 2026-01-27, timeout truncation |
| Qwen3-VL-235B-A22B (MoE 4) | vision | - | - | - | - | - | - | 53% | **53%** | 6.7 | Valid 2026-01-27 |
| **Qwen3-235B-A22B** | architect_general | 28/30 | 23/30 | 29/30 | 28/30 | 30/30 | 25/33 | - | **88%** | 5.9 | - |
| **Qwen3-235B-A22B (MoE 2)** | architect_general | 28/30 | 28/30 | 26/30 | 25/30 | 27/30 | 28/33 | - | **88%** | 8.2 | **+39% OPTIMAL ✓** |
| Qwen3-235B-A22B (MoE 4) | architect_general | 28/30 | 25/30 | 28/30 | 27/30 | 25/30 | 27/33 | - | **87%** | 7.3 | +24% |
| Qwen3-235B-A22B (MoE 6) | architect_general | 23/30 | 24/30 | 27/30 | 26/30 | 28/30 | 28/33 | - | **86%** | 6.7 | +14% |
| **Qwen3-Next-80B-A3B** | ingest | 29/30 | - | - | 25/30 | - | 12/33 | - | **74%** | 9.7 | SSM (no spec) |
| Qwen3-Next-80B-A3B (MoE 2) | ingest | 28/30 | 21/30 | 30/30 | 22/30 | 23/30 | 24/33 | - | **80%** | 9.8 | +1% (SSM) |
| Qwen3-Next-80B-A3B (MoE 4) | ingest | 30/30 | 29/30 | 30/30 | 30/30 | 27/30 | 15/33 | - | **89%** | 9.9 | +2% (SSM) |
| Qwen3-Next-80B-A3B (MoE 6) | ingest | 30/30 | - | - | 30/30 | - | 16/33 | - | **84%**‡ | 9.8 | +1% (SSM) |
| **Qwen3-Next-80B-A3B-Thinking** | thinking | 30/30 | 30/30 | 30/30 | 27/30 | 21/30 | 28/33 | - | **92%** | 9.2 | SSM (no spec) |
| **Qwen3-Next-80B-A3B-Thinking (MoE 2)** | thinking | 30/30 | - | 30/30 | - | 27/30 | 33/33 | - | **98%**† | 10.3 | **+12% ✓** |
| Qwen3-Next-80B-A3B-Thinking (MoE 4) | thinking | 30/30 | 29/30 | 30/30 | 29/30 | 30/30 | 31/33 | - | **98%** | 9.8 | +7% (SSM) |
| Qwen3-Next-80B-A3B-Thinking (MoE 6) | thinking | 30/30 | 30/30 | 30/30 | 29/30 | 24/30 | 30/33 | - | **95%** | 9.7 | +5% (SSM) |
| **Qwen3-Coder-53B-A3B** | coder_escalation | 19/30 | 11/27 | 12/30 | 11/30 | 9/30 | 7/33 | - | **38%** ⚠️ | 9.2 | Repetition loops |
| Qwen3-Coder-53B-A3B (MoE 2) | coder_escalation | 0/30 | 0/30 | 0/27 | 0/30 | 0/30 | 0/33 | - | **0%** ⚠️ | 14.8 | DEAD |
| Qwen3-Coder-53B-A3B (MoE 4) | coder_escalation | 8/30 | 22/30 | 26/30 | 24/30 | 16/30 | 15/33 | - | **61%** | 14.0 | 21% looping |
| Qwen3-Coder-53B-A3B (MoE 6) | coder_escalation | 28/30 | 25/30 | 28/30 | 22/30 | 29/30 | 24/33 | - | **85%** | 12.7 | ✓ |
| **Qwen3-Coder-30B-A3B** | frontdoor/coder | 27/30 | 20/30 | 20/30 | 27/30 | 30/30 | 22/33 | - | **80%** | 17.1 | - |
| Qwen3-Coder-30B-A3B (MoE 2) | frontdoor/coder | 0/24 | - | - | - | - | - | - | **0%** ⚠️ | 12.1 | DEAD |
| Qwen3-Coder-30B-A3B (MoE 4) | frontdoor/coder | 24/30 | 29/30 | 30/30 | 18/30 | 22/30 | 26/33 | - | **81%** | 23.6 | 29.92 (+lookup) ❌ |
| **Qwen3-Coder-30B-A3B (MoE 6)** | frontdoor/coder | 27/30 | 29/30 | 30/30 | 30/30 | 26/30 | 23/33 | - | **90%** ⭐ | 18.3 | **+48% quality vs MoE4** |
| **Qwen3-30B-A3B-Thinking-2507 (Q8_0)** | thinking | 24/30 | 22/30 | 20/30 | 20/30 | 21/30 | 11/33 | - | **64%** | 17.4 | - |
| Qwen3-30B-A3B-Thinking-2507 (Q8_0, MoE 2) | thinking | 0/30 | 0/27 | 0/18 | 0/30 | 1/24 | 0/33 | - | **1%** ⚠️ | 24.2 | GARBAGE |
| Qwen3-30B-A3B-Thinking-2507 (Q8_0, MoE 4) | thinking | 21/30 | 16/27 | - | 11/24 | 12/30 | - | - | **54%** | 19.6 | - |
| Qwen3-30B-A3B-Thinking-2507 (Q8_0, MoE 6) | thinking | 26/30 | 29/30 | 29/30 | 30/30 | 30/30 | 21/33 | - | **90%** | 19.0 | ✓ |
| **Qwen3-30B-A3B-Thinking-2507 (ingest)** | ingest | 26/30 | 25/30 | 25/30 | 24/30 | 21/30 | 22/33 | - | **78%** | 17.6 | - |
| Qwen3-30B-A3B-Thinking-2507 (ingest, MoE 2) | ingest | - | - | - | - | - | - | - | **11%** ⚠️ | 24.3 | DEAD |
| Qwen3-30B-A3B-Thinking-2507 (ingest, MoE 4) | ingest | 27/30 | 30/30 | 30/30 | 30/30 | 28/30 | 21/33 | - | **91%** | 21.4 | ✓ |
| Qwen3-30B-A3B-Thinking-2507 (ingest, MoE 6) | ingest | 29/30 | 29/30 | 30/30 | 23/30 | 29/30 | 22/33 | - | **89%** | 19.5 | ✓ |
| **Qwen3-30B-A3B-Thinking-2507 (Q4_K_S)** | thinking | - | - | - | - | - | - | - | - | - | - |
| Qwen3-30B-A3B-Thinking-2507 (Q4_K_S, MoE 2) | thinking | - | - | - | - | - | - | - | - | - | - |
| Qwen3-30B-A3B-Thinking-2507 (Q4_K_S, MoE 4) | thinking | - | - | - | - | - | - | - | - | - | - |
| Qwen3-VL-30B-A3B | vision_escalation | - | - | - | - | - | - | 75% | **75%** | 19.0 | Valid 2026-01-27 |
| Qwen3-VL-30B-A3B (MoE 4) | vision_escalation | - | - | - | - | - | - | 75% | **75%** | 27.6 | Valid 2026-01-27, +45% speed |

**Notes:**
- † Qwen3-Coder-480B score excludes long_context suite (4/18) due to timeout issues at 40K+ token contexts. Score of 83/87 = 95% on thinking+agentic+coder only.
- ‡ MOE6 partial run (22 questions: agentic+general+2 long_context). Full benchmark pending.
- ⚠️ Qwen3-Coder-480B has a tokenizer quirk: occasionally outputs Chinese characters (e.g., "6日消息1" instead of "60") in numerical contexts. Does not affect reasoning quality - correct answer usually follows.
- **MOE Quality Summary (2025-12-24):** MOE2=14% (garbage), MOE4=88% (good), MOE6=95% (partial). MOE3 untested but expected similar to MOE4/6.
- **MOE8 is redundant (2025-12-29):** Qwen3-Coder-480B uses 8 experts by default. MOE8 test confirmed: 96% at 4.4 t/s = baseline.
- **SSM Model Finding (2025-12-30):** SSM+MoE hybrids hit a **ceiling effect** where moe2/moe4 produce identical speeds (~10.2 t/s). The ~12% speedup from baseline→moeX is real, but further expert reduction doesn't help. Instruct variant: +1-2% (SSM bottleneck from start). Thinking variant: +12% (baseline slower at 9.2 t/s, moeX hits same 10.2 t/s ceiling).
- **Benchmark Variance (2026-01-07):** Same model tested under different roles (frontdoor_moe6 vs coder_escalation) showed 90% vs 76% scores. Variance likely due to model non-determinism. Use role-specific scores for role-specific decisions.

#### Tier A-B: Production & Specialist Models (Dense 70B+)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| **Meta-Llama-3.1-70B** | architect | 29/30 | 29/30 | 25/30 | 30/30 | 27/30 | 25/33 | **90%** | 2.1 | 84.31 | PARD-Llama-3.2-1B Q4_0 | 24 | - |
| **Hermes-4-70B** | architect | 29/30 | 29/30 | 22/30 | 28/30 | 30/30 | 15/33 | **84%** | 2.7 | - | - | - | - |
| Hermes-4-70B | ingest | 30/30 | 29/30 | 30/30 | 24/30 | 25/30 | 15/33 | **84%** | 2.9 | - | - | - | - |
| Meta-Llama-3.1-70B | ingest | 28/30 | 28/30 | 23/30 | 26/30 | 23/30 | 21/33 | **81%** | 2.0 | 85.75 | PARD-Llama-3.2-1B Q4_0 | 24 | - |
| Qwen2.5-Math-72B (2nd run) | math | 22/30 | 30/30 | 26/30 | 20/30 | 21/30 | 22/33 | **77%** | 2.0 | 158.85 | Qwen2.5-0.5B | 24 | - |
| Qwen2.5-72B | ingest | 28/30 | 26/30 | 27/30 | 20/30 | 19/30 | 18/33 | **75%** | 2.2 | - | - | - | - |
| DeepSeek-R1-Distill-Llama-70B | thinking | 20/30 | 20/30 | 20/30 | 21/30 | 20/30 | 13/33 | **62%** | 1.0 | - | - | - | - |
| Qwen2.5-Math-72B (Q4_K_M) | math | 21/30 | 18/30 | 21/30 | 19/30 | 21/30 | 11/33 | **61%** | ~2.0 | 7.55 | Qwen2.5-0.5B | 12 | 0.5 |
| Meta-Llama-3-70B | architect | 0/30 | 13/30 | 5/30 | 12/30 | 15/30 | 16/33 | **33%** ⚠️ | 14.9 | 6.42 | PARD-Llama-3.2-1B | 8 | - |
| Qwen2.5-72B (base) | architect | - | - | - | - | - | - | - | 0.85 | - | - | - | - |
| Qwen2.5-72B-Instruct | architect | - | - | - | - | - | - | - | 1.70 | 8.53 | Qwen2.5-0.5B | 16 | - |
| Qwen2.5-Math-72B (Q6_K) | math | - | - | - | - | - | - | - | - | - | - | - | - |
| GLM-4.6 (dense) | general | - | - | - | - | - | - | - | - | - | - | - | - |

#### Tier A-B: Production & Specialist Models (Dense 27B-32B)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| DeepSeek-R1-Distill-Qwen-32B (Q6_K) | thinking | 26/30 | - | - | - | 13/18 | - | **81%**‡ | 1.78 | - | - | - | - |
| Qwen3-32B | general | 28/30 | - | - | - | 30/30 | - | **97%**‡ | 1.64 | 5.87 | Qwen3-0.6B | 8 | - |
| Qwen3-32B | ingest | 30/30 | 27/30 | 30/30 | 21/30 | 29/30 | 22/33 | **87%** | 1.65 | - | - | - | - |
| Gemma-3-27B-QAT | general | - | - | - | - | - | - | - | 4.72 | - | - | - | - |
| Qwen2.5-Coder-32B | coder/summarize | 28/30 | - | - | 8/9 | 30/30 | - | **96%**‡ | 2.99 | 33.0 | Qwen2.5-Coder-0.5B | 24 | - |
| Qwen2.5-Coder-32B | ingest | 22/30 | 20/30 | 20/30 | 22/30 | 29/30 | 19/33 | **72%** | 3.43 | - | - | - | - |

#### Tier B: Thinking & Reasoning Models (7B-14B)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| DeepSeek-R1-Distill-Qwen-14B (Q4_K_M) | thinking | 29/30 | 22/30 | 26/30 | 27/30 | 23/30 | 22/33 | **81.4%** | 2.8 | - | - | - | - |
| DeepSeek-R1-Distill-Qwen-14B (Q6_K_L) | thinking | 29/30 | 29/30 | 28/30 | 29/30 | 30/30 | 29/33 | **95.1%** | 4.4 | - | - | - | - |
| DeepSeek-R1-Distill-Llama-8B | thinking | 28/30 | 24/30 | 30/30 | 30/30 | - | - | **93%*** | 7.2 | - | - | - | - |
| DeepSeek-R1-Distill-Qwen-7B | thinking | 29/30 | 19/30 | 30/30 | 24/30 | - | - | 85%* | 7.7 | - | - | - | - |
| DeepSeek-R1-0528-Qwen3-8B | thinking | 21/30 | 18/30 | 24/30 | 19/30 | 23/30 | 12/33 | **63.9%** ⚠️ | 8.1 | - | - | - | - |
| Qwen3-4B-Thinking-2507-Q8_0 | thinking | 22/30 | 19/30 | 8/9 | 5/9 | - | - | **69%** | 18.0 | - | - | - | - |

#### Tier C: Workers & General Models

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| **Qwen2.5-7B-Q4_K_S** | general | 40/60 | 42/60 | 42/60 | 40/60 | 46/60 | 44/66 | **69.4%** | 16.0 | - | - | - | - |
| Meta-Llama-3-8B-Instruct | worker_general | 37/60 | 36/60 | 19/30 | 40/60 | 22/30 | 22/33 | **64.5%** | 12.6 | - | - | - | - |
| Meta-Llama-3-8B-Instruct | toolrunner | 19/30 | 37/60 | 20/30 | 20/30 | 22/30 | 22/33 | **65.7%** | 13.8 | - | - | - | - |
| Meta-Llama-3.1-8B (Q4_K_S) | general | 4/30 | 6/30 | 2/30 | 7/30 | 4/30 | 4/33 | **15%** | 63.5 | - | - | - | - |
| **Qwen2.5-Math-7B-Instruct** | worker_math | 43/60 | 30/60 | 27/30 | 26/60 | - | - | **60.0%** | 11.3 | - | - | - | - |
| Qwen2.5-VL-7B-Instruct | worker_vision_agentic | - | - | - | - | - | - | **81% VL** | 17.2 | - | - | - | - |
| Qwen2.5-Coder-32B | worker_summarize | - | - | - | - | - | - | - | 5.79 | 95.18 | (lookup) | - | - |
| **Gemma-3-12B-IT** | general | 29/30 | 30/30 | 27/30 | 28/30 | 26/30 | 28/33 | **91.8%** | 9.3 | - | - | - | - |
| Gemma-3-27B-IT-QAT | general | 12/12 | 29/30 | 27/30 | 30/30 | 29/30 | 27/33 | **93.3%** | 2.0 | - | - | - | - |
| MathSmith-Qwen3-8B ⚠️ | math | 30/30 | 28/30 | 28/30 | - | - | - | **95.6%** | **3.4** ⚠️ | - | - | - | - |
| MathSmith-Hard-Problem-Synthesizer-Qwen3-8B | formalizer | - | 20/30 | 10/15 | - | - | - | **66.7%** | 11.1 | - | - | - | - |

#### Vision Models (Valid 2026-01-27 Benchmark)

> **✅ VALID VL BENCHMARK (2026-01-27)**: Images properly passed via `llama-mtmd-cli`. See "VL Benchmark Results" section for full analysis.

| Model | Role | VL Score | Speed | Notes |
|-------|------|----------|-------|-------|
| **Qwen3-VL-4B** | vision_primary | **94% (34/36)** | 18.0 t/s | Best quality, HOT tier |
| Qwen3-VL-8B | vision_alt | 86% (31/36) | 15.4 t/s | Good alternative |
| Qwen2.5-VL-7B | vision_agentic | 81% (29/36) | 17.2 t/s | Only VL with tool calls |
| Qwen3-VL-30B-A3B | vision_escalation | 75% (27/36) | 19.0 t/s | Manual escalation |
| Qwen3-VL-30B-A3B MoE4 | vision_escalation | 75% (27/36) | 27.6 t/s | Faster with expert reduction |
| Qwen3-VL-235B-A22B | vision_complex | 56% (20/36) | 4.6 t/s | Timeout truncation issues |
| Qwen3-VL-235B-A22B MoE4 | vision_complex | 53% (19/36) | 6.7 t/s | Still has truncation |

**Key Finding**: Smaller models outperform larger on VL tasks (4B=94% > 30B=75% > 235B=56%)

#### Tier D: Draft Models (Qwen2.5 Family)

**⚠️ RESCORED 2026-01-07:** Previous scores from deprecated suites were inflated. New hardened suite scores below.

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| **Qwen2.5-Coder-1.5B** (Q4_K_M) | draft_primary | 11/30 | 19/30 | - | - | - | - | **50%** | 99.7 | Qwen2.5-* (best quality/speed) |
| Qwen2.5-Math-1.5B (Q4_K_M) | draft | 20/30 | 11/30 | - | - | - | - | **52%** | 54.3 | Qwen2.5-Math-* |
| Qwen2.5-Math-1.5B (Q6_K) | draft | 17/27 | 10/30 | - | - | - | - | **47%** | 57.4 | Qwen2.5-Math-* |
| Qwen2.5-0.5B (Q8_0) | draft | 14/30 | 12/30 | - | - | - | - | **43%** | 156.8 | Qwen2.5-* |
| Qwen2.5-Coder-0.5B (Q8_0) | draft | 16/30 | 8/30 | - | - | - | - | **40%** | 142.2 | Qwen2.5-Coder-* |
| Qwen2-0.5B (Q2_K) | draft ⚠️ | 5/30 | 6/30 | - | - | - | - | **18%** | 155.9 | Qwen2-* (Q2_K degrades quality) |
| Qwen2.5-Coder-1.5B (Q2_K) | draft ⚠️ | 2/30 | 2/30 | - | - | - | - | **7%** | 87.9 | UNUSABLE (Q2_K severely degrades) |

#### Tier D: Draft Models (Qwen3 Family)

**⚠️ RESCORED 2026-01-07:** Previous scores from deprecated suites were inflated. Qwen3-0.6B is UNUSABLE.

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| **Qwen3-1.7B** (Q4_K_M) | draft_primary | 21/30 | 15/30 | - | - | - | - | **60%** | 43.3 | Qwen3-* (best quality for Qwen3) |
| Qwen3-0.6B (Q8_0) | draft ⛔ | 0/30 | 2/30 | - | - | - | - | **3%** | 95.3 | UNUSABLE (repetitive garbage) |
| PARD-Qwen3-0.6B (Q4_0) | draft | 18/30 | 21/30 | - | - | - | - | **65%** | 78.4 | Qwen3-* (PARD version works better) |
| Co-Rewarding-II-Qwen3-1.7B-Math | draft (⚠️) | 21/30 | 21/30 | - | - | - | - | 70%* | 22.0 | incompatible |
| Qwen3-Embedding-0.6B | draft (⚠️) | - | - | - | - | - | - | - | - | embedding only |
| Qwen3-VL-1B-Merged (Q8_0) | draft (⚠️) | 4/6 | - | - | - | - | - | 67%* | 67.7 | corrupted output |

#### Tier D: Draft Models (DeepSeek-R1-Distill Family)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| **DeepSeek-R1-Distill-Qwen-1.5B** (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 29.5 | DeepSeek-R1-Distill-* |
| DeepSeek-R1-Distill-Qwen-1.5B (lmstudio Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 48.0 | DeepSeek-R1-Distill-* |
| PARD-DeepSeek-R1-1.5B (Q5_K_S) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 58.0 | DeepSeek-R1-Distill-* |
| PARD-DeepSeek-R1-1.5B (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 40.9 | DeepSeek-R1-Distill-* |

#### Tier D: Draft Models (Llama Family)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| PARD-Llama-3.2-1B (Q4_0) | draft | 19/30 | 19/30 | - | - | - | - | **63%** | 82.3 | Llama-3.2/3.3 |
| PARD-Llama-3.2-1B (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 71.4 | Llama-3.2/3.3 |

### Review Files

- Location: `benchmarks/results/reviews/`
- Per-model: `{model}_baseline.csv`
- Summary: `summary.csv`

---

## Retrieval Stack (NextPLAID ColBERT)

### Current Models

| Index | Model | Params | Dim | Port | Quantization | Index Size | Benchmark |
|-------|-------|--------|-----|------|-------------|------------|-----------|
| Code | LateOn-Code | 130M | 128 | :8088 | ONNX INT8 | 336MB | MTEB-Code 74.12 |
| Docs | answerai-colbert-small-v1 | 33M | 96 | :8089 | ONNX INT8 | 31MB | unscored |

### Index Configuration

- PQ compression: `nbits=4` (IVF+PQ hybrid)
- AST chunking for Python (tree-sitter), fallback for other languages
- IVF files: `ivf.npy`, `centroids.npy`, `cluster_threshold.npy`, `bucket_cutoffs.npy`

### Candidate Upgrade

| Model | Params | Dim | BEIR avg | LongEmbed | Notes |
|-------|--------|-----|----------|-----------|-------|
| GTE-ModernColBERT-v1 | 149M | 128 | **54.67** | **88.39** | Needs ONNX conversion |

**Rationale**: Docs model upgrade — 128-dim (matches code index), ModernBERT backbone (same family as LateOn-Code), SOTA on long-context retrieval. See `handoffs/active/colbert-zero-research-integration.md`.

---

## Full Data

- Detailed results: `logs/research_report.md`
- Methodology: `research/speculative_decoding_research.md`
- Blog template: `research/research_report_template.md`
- Benchmark prompts: `benchmarks/prompts/v1/`
- Benchmark results: `benchmarks/results/`
- Claude-as-Judge reviews: `benchmarks/results/reviews/`
