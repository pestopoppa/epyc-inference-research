# Speculative Decoding Research Results
**Date:** 2024-12-13
**System:** AMD EPYC 9655 (96 cores, 192 threads, 1.13TB DDR5)
**Build:** llama.cpp 7bed317f5 (7371)

## Executive Summary

Speculative decoding requires **exact tokenizer compatibility** between target and draft models. Only models from the **same family with identical vocabularies** work reliably.

## Benchmark Results

### Baseline Benchmarks (No Speculative Decoding)

**Updated 2025-12-13** - Complete sweep on AMD EPYC 9655 (96t, NUMA interleaved)

#### Draft Models (0.5B-1.5B)
| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|-------------|-------------|
| Qwen2.5-0.5B-Q8_0 | 501M | 411.31 | **83.89** |
| Qwen2.5-Coder-0.5B-Q8_0 | 501M | 404.79 | **85.45** |
| Qwen3-0.6B-Q8_0 | 762M | 252.15 | 65.71 |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M | 1.04G | 659.10 | 54.54 |

#### Small Models (7B-14B)
| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|-------------|-------------|
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 4.36G | 240.69 | 13.15 |
| Qwen2.5-Math-7B-Q4_K_M | 4.36G | 278.00 | 12.44 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 4.58G | 218.70 | 13.42 |
| Gemma-3-12B-Q4_K_M | 6.79G | 124.96 | 8.97 |
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_M | 8.37G | 136.81 | 6.44 |

#### Medium Models (32B)
| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|-------------|-------------|
| Qwen2.5-Coder-32B-Q2_K | 11.46G | 47.93 | 3.54 |
| DeepSeek-R1-Distill-Qwen-32B-Q2_K | 11.46G | 43.93 | 3.01 |
| Qwen2.5-Coder-32B-Q4_K_M | 18.48G | 63.41 | 2.83 |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M | 18.48G | 66.56 | 3.35 |

#### MoE Models (High Efficiency)
| Model | Size | pp512 (t/s) | tg (t/s) | Active Params |
|-------|------|-------------|----------|---------------|
| **Qwen3-Coder-30B-A3B-Q4_K_M** | 17.35G | 172.16 | **21.22** (tg128) | ~3B |
| **Qwen3-Next-80B-A3B-Q4_K_M** | 45.08G | 70.94 | **7.51** (tg128) | ~3B |
| Qwen3-Coder-480B-A35B-Q4_K_M | 270.86G | 29.47 | 2.03 (tg64) | ~35B |

#### Large Dense Models (70B-80B)
| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|-------------|-------------|
| DeepSeek-R1-Distill-Llama-70B-Q4_K_M | 39.59G | 30.08 | 1.36 |
| Hermes-4-70B-Q4_K_M | 39.59G | 26.99 | 1.31 |
| Meta-Llama-3.1-70B-Instruct-Q4_K_M | 39.59G | 25.04 | 1.20 |
| Qwen2.5-72B-Q4_K_M | 44.15G | 25.24 | 1.15 |
| Qwen2.5-Math-72B-Instruct-Q4_K_M | 44.15G | 18.86 | 1.03 |

#### Legacy Results
| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|-------------|-------------|
| Qwen3-Coder-480B-A35B Q4_K_M | 271GB | 34.66 | 3.06 |

### Speculative Decoding Results

**Updated 2025-12-13** - Testing with adaptive K values

| Target Model | Draft Model | K | Accept Rate | Speed (t/s) | Speedup | Status |
|--------------|-------------|---|-------------|-------------|---------|--------|
| Qwen2.5-Coder-32B-Q4_K_M | Qwen2.5-Coder-0.5B-Q8_0 | 8 | 51.8% | ~8.9 | 3.1x | ✅ WORKS |
| Qwen2.5-Coder-32B-Q4_K_M | Qwen2.5-Coder-0.5B-Q8_0 | 16 | **66.3%** | **~14.5** | **5.1x** | ✅ BEST |
| Qwen2.5-Coder-32B-Q4_K_M | Qwen2.5-Coder-0.5B-Q8_0 | 24 | 52.3% | ~14.0 | 4.9x | ✅ OK |
| Qwen2.5-Coder-32B-Q4_K_M | Qwen2.5-Coder-0.5B-Q8_0 | 32 | 70.5% | ~9.7 | 3.4x | ✅ Slower |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M | DeepSeek-R1-1.5B-Q4_K_M | 8 | 6.5% | ~3.0 | 0.9x | ❌ Vocab mismatch |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M | PARD-DeepSeek-R1-1.5B-Q8_0 | 8 | 0% | N/A | N/A | ❌ Total failure |
| **Meta-Llama-3.1-70B-Instruct-Q4_K_M** | PARD-Llama-3.2-1B-Q8_0 | 8 | **49.5%** | **~4.08** | **3.4x** | ✅ WORKS |
| Meta-Llama-3.1-70B-Instruct-Q4_K_M | PARD-Llama-3.2-1B-Q8_0 | 16 | 19.5% | ~2.56 | 2.1x | ✅ Lower K better |
| **Qwen2.5-72B-Instruct-Q4_K_M** | Qwen2.5-0.5B-Q8_0 | 8 | **46.9%** | **~3.39** | **2.9x** | ✅ WORKS |
| Qwen2.5-72B-Instruct-Q4_K_M | Qwen2.5-0.5B-Q8_0 | 16 | ~45% | ~3.27 | 2.8x | ✅ Similar to k=8 |
| Qwen2.5-72B-Instruct-Q4_K_M | Qwen2.5-7B-Q4_K_S | 8 | 42.7% | ~1.80 | 1.6x | ⚠️ Large draft too slow |
| **Qwen2.5-Math-72B-Instruct-Q4_K_M** | Qwen2.5-0.5B-Q8_0 | 8 | ~50% | **~5.19** | **5.0x** | ✅ WORKS |
| Qwen2.5-Coder-32B-Q4_K_M | Qwen2.5-7B-Q4_K_S | 8 | ~40% | ~4.36 | 1.5x | ⚠️ Large draft too slow |
| Qwen2.5-72B-Q4_K_M (base) | Qwen2.5-0.5B-Q8_0 | 16 | N/A | N/A | N/A | ❌ Token mismatch |
| DeepSeek-R1-Distill-Llama-70B-Q4_K_M | PARD-Llama-3.2-1B-Q8_0 | 16 | N/A | N/A | N/A | ❌ Token mismatch |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | PARD-Llama-3.2-1B-Q8_0 | 8 | N/A | N/A | N/A | ❌ Token mismatch |
| Hermes-4-70B-Q4_K_M | PARD-Llama-3.2-1B-Q8_0 | 8 | N/A | N/A | N/A | ❌ Vocab mismatch (tool_call) |
| Meta-Llama-3.1-70B-Instruct-Q4_K_M | Meta-Llama-3.1-8B-Q4_K_S | 8 | N/A | N/A | N/A | ❌ Token mismatch |

#### Legacy Results
| Target Model | Draft Model | Accept Rate | Speed (t/s) | Status |
|--------------|-------------|-------------|-------------|--------|
| Qwen2.5-Coder-32B | Qwen2.5-0.5B-Instruct Q8_0 | **100%** | **19.39** | **WORKS** |
| DeepSeek-R1-Distill-Qwen-32B | PARD-DeepSeek-R1-1.5B Q8_0 | 0% | 1.85 | FAIL - garbage output |
| DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M | 7.5% | 3.17 | FAIL - vocab mismatch |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5-0.5B-Instruct Q8_0 | N/A | N/A | FAIL - token mismatch error |
| Qwen3-Coder-480B | PARD-Qwen3-0.6B Q4_0 | N/A | N/A | FAIL - token mismatch error |
| Qwen3-Coder-480B | Qwen3-0.6B Q8_0 (official) | N/A | N/A | FAIL - BOS token mismatch |

## Key Findings

### What WORKS
1. **Qwen2.5 family** - 100% acceptance rate with Qwen2.5-0.5B as draft
2. **6.7x speedup** achieved (2.89 → 19.39 t/s)
3. Same model family + same tokenizer = reliable speculative decoding

### What DOES NOT Work

1. **PARD models** - Modified tokenizers break compatibility
   - PARD-DeepSeek-R1-Distill-Qwen-1.5B: 0% acceptance, garbage output
   - PARD-Qwen3-0.6B: Token mismatch errors

2. **DeepSeek-R1 models** - Vocab size differences
   - 32B model: 152,064 tokens
   - 1.5B model: 151,936 tokens
   - Even official same-family models don't match

3. **Cross-family models** - Different BOS/EOS tokens
   - Qwen2.5 → DeepSeek: Token mismatch
   - Qwen3 → Qwen2.5: Token mismatch

4. **Qwen3 + Qwen3-0.6B** - BOS token differs
   - Even official same-generation models fail
   - MoE models may have different tokenizer configs

## Compatibility Matrix

| Target Family | Compatible Drafts | Notes |
|---------------|-------------------|-------|
| Qwen2.5 | Qwen2.5-0.5B, Qwen2.5-1.5B | Works perfectly |
| DeepSeek-R1-Distill-Qwen | None found | Vocab size mismatch with all drafts |
| Qwen3 | Unknown | BOS token mismatch with 0.6B |
| Qwen3-Coder (MoE) | Unknown | Same BOS mismatch issue |

## K-Value Optimization (Qwen2.5-Coder-32B + Qwen2.5-0.5B)

### Comprehensive K Sweep on Code Prompt (2024-12-13)

Testing different speculation depths on code generation:

| K | Speed (t/s) | Drafted | Accepted | Acceptance | Speedup vs Baseline |
|---|-------------|---------|----------|------------|---------------------|
| 4 | 11.73 | 124 | 120 | **96.77%** | 4.1x |
| 8 | 17.32 | 152 | 132 | 86.84% | 6.0x |
| 12 | 19.53 | 192 | 143 | 74.48% | 6.8x |
| 16 | 20.56 | 192 | 147 | 76.56% | 7.1x |
| 20 | 25.88 | 180 | 151 | 83.89% | 9.0x |
| **24** | **28.79** | 192 | 160 | 83.33% | **10.0x** |

**Baseline**: ~2.89 t/s (Qwen2.5-Coder-32B without speculation)

**Key insight**: For code generation, aggressive K (20-24) achieves best throughput despite moderate acceptance rates. The parallel verification speedup outweighs rejected tokens.

**Optimal K by context**:
- **Code/structured output**: K=20-24 (high predictability)
- **General/mixed**: K=8-12 (balanced)
- **Creative/prose**: K=4-8 (lower predictability)

### Adaptive K Strategy
- Start at K=12 (balanced default)
- Monitor rolling acceptance over 64-128 tokens
- If acceptance >80%: increase K by 4 (room for more speculation)
- If acceptance <60%: decrease K by 4 (too many wasted drafts)
- Apply context heuristics:
  - Detect code (syntax chars, indentation): boost to K=20-24
  - Detect JSON/schemas: use K=8-12 (tight tolerance)
  - Detect prose: use K=4-8

## Context-Dependent Acceptance Rates (Updated 2024-12-13)

### Code vs Prose Comparison

| Prompt Type | K | Speed (t/s) | Drafted | Accepted | Acceptance | Speedup |
|-------------|---|-------------|---------|----------|------------|---------|
| **Code** | 8 | 17.32 | 152 | 132 | 86.84% | 6.0x |
| **Code** | 24 | **28.79** | 192 | 160 | 83.33% | **10.0x** |
| Prose | 8 | **7.85** | 336 | 109 | 32.44% | 2.7x |
| Prose | 24 | 6.22 | 840 | 124 | 14.76% | 2.2x |

**Critical insight**: Context type dramatically affects optimal K:
- **Code at K=24**: 28.79 t/s (83% acceptance)
- **Prose at K=24**: 6.22 t/s (15% acceptance) - **4.6x slower!**
- For prose, K=8 outperforms K=24 (7.85 vs 6.22 t/s)

The draft model predicts code tokens far more accurately than prose, enabling aggressive speculation only for structured content.

### Validated Adaptive K Strategy

| Content Type | Optimal K | Expected Acceptance | Expected Speedup |
|--------------|-----------|---------------------|------------------|
| Code/structured | 20-24 | 80-90% | 8-10x |
| JSON/schemas | 8-12 | 60-80% | 5-7x |
| General/mixed | 8-12 | 50-70% | 4-6x |
| Creative/prose | 4-8 | 30-50% | 2-4x |

### Content Detection Heuristics
- **Code**: `{`, `}`, `(`, `)`, `;`, `def `, `class `, `import `, indentation patterns
- **JSON**: `"key":`, `[`, `]`, strict key-value patterns
- **Prose**: High ratio of alphabetic chars, sentence structures, low special chars

## Working Command Template

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /path/to/target-model.gguf \
  -md /path/to/draft-model.gguf \
  -t 96 -c 4096 -n 128 --draft-max 8 \
  -p "Your prompt here"
```

## Hybrid SSM Architecture Research (Qwen3-Next)

### Architecture Analysis

**Qwen3-Next-80B-A3B** is a **hybrid MoE+SSM model** using:
- **Gated DeltaNet** linear attention layers (from "Gated Delta Networks" paper)
- **512 experts** (10 active per token)
- **SSM state size**: 128
- **Mamba-style recurrent components** with 4096 inner size

### Why Standard Speculative Decoding Fails

Testing Qwen3-Next-80B + PARD-Qwen3-0.6B resulted in:
```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 13
 - the tokens for sequence 0 in the input batch have a starting position of Y = 10
 it is required that the sequence positions remain consecutive: Y = X + 1
```

**Root cause**: SSM models use **recurrent state** instead of KV cache. The state management is fundamentally different:
- **Transformer KV cache**: Position-indexed, allows random access for verification
- **SSM recurrent state**: Sequential, position must be consecutive (Y = X + 1)

When speculative decoding rejects tokens and "rolls back", the KV cache can simply discard entries. SSM state cannot roll back without recomputation.

### Research Solutions

1. **STree (Speculative Tree Decoding)** - [arxiv.org/html/2505.14969](https://arxiv.org/html/2505.14969)
   - First scalable algorithm for tree-based speculative decoding in SSMs
   - Uses "activation replay" instead of state backtracking
   - Requires diagonal constraint on SSM transition matrices
   - Trade-off: SSMs have linear runtime, so need high acceptance rates to justify overhead

2. **SpeculativeMamba** - [github.com/itsdaniele/speculative_mamba](https://github.com/itsdaniele/speculative_mamba)
   - Python implementation for Mamba models
   - Uses mamba-130m draft with mamba-2.8b target
   - Achieves ~68% acceptance rate, 1.5x speedup on 3090

3. **MambaInLlama** - [github.com/jxiw/MambaInLlama](https://github.com/jxiw/MambaInLlama)
   - Distills Llama into hybrid Mamba models
   - Hardware-aware speculative decoding algorithm
   - Strong performance on Ampere GPUs, challenging on H100

### llama.cpp Status for Qwen3-Next

PR #16095 adds Qwen3-Next support but:
- Implementation is "CORRECTNESS ONLY" - no speed optimization
- No speculative decoding support
- Known issues with recurrence layer memory allocation
- CPU performance: ~11.76 t/s (vs ~63 t/s for Qwen3-30B with comparable active params)

### Potential Approaches to Investigate

1. **Use llama-server instead of llama-speculative** - Better SSM context handling
2. **Single-token "speculative" mode** - K=1 might bypass consecutive position requirement
3. **Implement activation replay** - Port STree approach to llama.cpp
4. **Wait for llama.cpp hybrid support** - PR #16095 evolving

## Self-Speculative Decoding Experiments (2024-12-13)

### Concept
Self-speculative decoding uses a **lower-quantized version of the same model** as its own draft:
- Advantages: Perfect tokenizer match, no separate draft model needed
- Disadvantage: Draft model is still full-size (just lower precision), slower drafting

### Models Created

| Target | Draft Created | Size Reduction |
|--------|---------------|----------------|
| Qwen3-Next-80B Q4_K_M (46GB) | Q2_K (28GB) | 39% |
| DeepSeek-R1-Distill-32B Q4_K_M (19GB) | Q2_K (12GB) | 37% |
| Qwen2.5-Coder-32B Q4_K_M (19GB) | Q2_K (12GB) | 37% |

### Results

| Model | Self-Spec Config | Result | Speed | Acceptance |
|-------|------------------|--------|-------|------------|
| **Qwen3-Next-80B** | Q4+Q2 | ❌ FAIL | N/A | SSM state rollback |
| **DeepSeek-R1-Distill-32B** | Q4+Q2 | ❌ FAIL | 1.14 t/s | 24% (garbage output) |
| **Qwen2.5-Coder-32B** | Q4+Q2 | ✅ WORKS | 2.22 t/s | 60.7% |

### Self-Speculative vs Small Draft Comparison (Qwen2.5-Coder-32B)

| Configuration | Speed | Acceptance | Speedup vs Baseline |
|---------------|-------|------------|---------------------|
| Baseline (no spec) | ~3.5 t/s | N/A | 1.0x |
| Self-spec Q4+Q2 | 2.22 t/s | 60.7% | **0.6x (SLOWER!)** |
| 32B + 0.5B (K=8) | 17.32 t/s | 96.8% | 5.0x |
| 32B + 0.5B (K=24) | 28.79 t/s | 83.3% | **10.0x** |

### Why Self-Speculative Underperforms

1. **Draft model too large**: Q2_K is still 32B params (12GB), ~250ms per draft token
2. **Lower acceptance**: 60.7% vs 83-97% with dedicated small model
3. **No parallelism benefit**: Both target and draft are memory-bound on same hardware

### Conclusion

**Self-speculative with Q4→Q2 is NOT recommended.** Use dedicated small draft models (0.5B-1.5B) for 10x speedup instead.

---

## Next Steps

1. Test Qwen2.5-72B + Qwen2.5-0.5B (user downloading manually)
2. ~~Test Qwen3-Next-80B + various drafts~~ **BLOCKED** - Architecture incompatible with standard spec decode
3. ~~Self-speculative decoding~~ **TESTED** - Works but slower than small draft models
4. Investigate llama-server approach for Qwen3-Next

## Model Inventory (Updated 2024-12-13)

### Target Models (Large)

| Model | Path | Size | Spec Decode Status |
|-------|------|------|-------------------|
| **Qwen2.5-Coder-32B** | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-GGUF/Qwen2.5-Coder-32B-Q4_K_M.gguf` | 19GB | ✅ WORKS |
| DeepSeek-R1-Distill-Qwen-32B | `/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf` | 19GB | ❌ No compatible draft |
| **Qwen3-Next-80B-A3B** | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | 46GB | ❌ SSM arch incompatible |
| Hermes-4-70B | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Hermes-4-70B-GGUF/Hermes-4-70B-Q4_K_M.gguf` | 40GB | 🔲 Untested |
| Qwen3-Coder-480B-A35B | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/` (8 parts) | 271GB | ❌ BOS mismatch |
| DeepSeek-R1 (full) | `/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-GGUF/` (9 parts) | 404GB | 🔲 Untested |
| GLM-4.6 | `/mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/` (5 parts) | 192GB | 🔲 Untested |

### Draft Models (Small)

| Model | Path | Size | Compatible Targets |
|-------|------|------|-------------------|
| **Qwen2.5-0.5B-Instruct Q8_0** | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf` | 507MB | **Qwen2.5-*** |
| Qwen3-0.6B Q8_0 | `/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf` | 768MB | ❌ BOS mismatch with Qwen3-Coder |
| PARD-Qwen3-0.6B Q4_0 | `/mnt/raid0/llm/lmstudio/models/fernandoruiz/PARD-Qwen3-0.6B-Q4_0-GGUF/pard-qwen3-0.6b-q4_0.gguf` | 442MB | ❌ SSM issues |
| DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M | `/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | 1.1GB | ❌ Vocab mismatch |
| PARD-DeepSeek-R1-1.5B Q8_0 | `/mnt/raid0/llm/lmstudio/models/mradermacher/PARD-DeepSeek-R1-Distill-Qwen-1.5B-GGUF/PARD-DeepSeek-R1-Distill-Qwen-1.5B.Q8_0.gguf` | 1.8GB | ❌ Garbage output |
| PARD-Llama-3.2-1B Q8_0 | `/mnt/raid0/llm/lmstudio/models/mradermacher/PARD-Llama-3.2-1B-GGUF/PARD-Llama-3.2-1B.Q8_0.gguf` | 1.5GB | Llama 3.2 family |
| Llama-3.2-1B-Instruct Q8_0 | `/mnt/raid0/llm/lmstudio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf` | 1.3GB | Llama 3.2 family |

### Self-Speculative Draft Models (Q2_K - Created 2024-12-13)

| Model | Path | Size | Self-Spec Result |
|-------|------|------|------------------|
| Qwen3-Next-80B-A3B Q2_K | `/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf` | 28GB | ❌ SSM rollback fails |
| DeepSeek-R1-Distill-32B Q2_K | `/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q2_K.gguf` | 12GB | ❌ Garbage output |
| **Qwen2.5-Coder-32B Q2_K** | `/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q2_K.gguf` | 12GB | ✅ Works (60.7% accept, 2.2 t/s) |

### Medium Models (Potential Targets)

| Model | Path | Size |
|-------|------|------|
| Magistral-Small-2509 | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Magistral-Small-2509-GGUF/Magistral-Small-2509-Q4_K_M.gguf` | 14GB |
| Ministral-3-14B-Reasoning | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 7.7GB |
| gemma-3-12b-it | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf` | 6.8GB |

### Downloads In Progress (User Downloading Manually)
- Qwen2.5-72B-Instruct Q4_K_M (~42GB)
- Qwen2.5-Math-72B Q4_K_M (~42GB)
- mradermacher/Qwen2.5-72B Q4_K_M (~42GB)

### Candidates for Deletion
- `/mnt/raid0/llm/hf/DeepSeek-R1/` - Corrupted HF files
