# Multi-Token Prediction (MTP) Investigation for GLM-4.6

**Created:** 2026-01-08
**Updated:** 2026-01-08 (Major corrections - architecture is MoE, not dense)
**Status:** Research In Progress - llama.cpp PR has architectural issues

---

## Executive Summary

Multi-Token Prediction (MTP) is a **self-speculative decoding** technique built into GLM-4 series models. Unlike traditional speculative decoding which requires a separate draft model, MTP uses auxiliary prediction heads embedded within the model itself to predict multiple future tokens simultaneously.

**Key Findings (CORRECTED):**
- GLM-4.6-355B is **MoE** (160 experts, 8 active = ~32B active), NOT dense as previously stated
- MoE expert reduction **already works**: 2.24 → 3.97 t/s (+77%)
- GLM-4.6 has MTP heads (`nextn_predict_layers = 1`) but they are **UNTESTED**
- llama.cpp PR #15225 has **fundamental architectural issues** - sequential processing defeats MTP benefit
- vLLM reports >90% acceptance rates, but llama.cpp PR author says "not expected to outperform baseline"
- **Potential opportunity**: MTP could stack with MoE reduction for additional 20-50% gains (unverified)
- **Quality concern**: 59% benchmark score, lower than Qwen alternatives

---

## What is Multi-Token Prediction?

### Traditional Speculative Decoding
```
[Draft Model] --> predicts N tokens --> [Target Model] --> verifies --> accepts/rejects
```
Requires two separate models loaded in memory.

### MTP (Self-Speculative Decoding)
```
[Main Model + Built-in MTP Heads] --> predicts N tokens --> [Main Model] --> verifies
```
Uses auxiliary prediction heads that share the model's trunk, eliminating the need for a separate draft model.

### Architecture

MTP-enabled models have additional layers (NextN layers) that predict future tokens:
- **NextN_1**: Predicts token at position t+1
- **NextN_2**: Predicts token at position t+2
- **NextN_N**: Predicts token at position t+N

These heads share the main model's embeddings and hidden states, making them highly accurate drafters with minimal memory overhead.

### Theoretical Advantages

| Aspect | Traditional Spec Decode | MTP |
|--------|------------------------|-----|
| Memory | 2 models (target + draft) | 1 model + small heads |
| Draft Accuracy | Depends on draft model | High (shares trunk) |
| Tokenizer Match | Must match exactly | Guaranteed (same model) |
| VRAM Overhead | Draft model size | ~5-10% for heads |

---

## Why MTP is Relevant to GLM-4.6

### GLM-4 Series MTP Support

| Model | MTP Layers | Status |
|-------|-----------|--------|
| GLM-4.5 | Layer 92 (NextN) | Primary target for llama.cpp PR |
| GLM-4.6 | Layer 92 (NextN) | Supported via weight-tying fallback |
| GLM-4.7 | MTP layers | Fully supported in vLLM |

### Our GLM-4.6 Configuration (CORRECTED)

**Model Path:** `/mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/GLM-4.6-Q4_K_S-00001-of-00005.gguf`

**Architecture Details (from model metadata):**
```
general.architecture: glm4moe
general.size_label: 160x19B
glm4moe.block_count: 93
glm4moe.expert_count: 160
glm4moe.expert_used_count: 8 (default)
glm4moe.nextn_predict_layers: 1  # MTP HEAD CONFIRMED!
tokenizer.ggml.model: gpt2 (151,552 vocab)
```

**Total Size:** 189GB (5 split GGUF files)
**Active Parameters:** ~32B (8 of 160 experts active)

### Benchmark Performance

| Role | Thinking | Agentic | Inst.Prec | Total | Speed |
|------|----------|---------|-----------|-------|-------|
| general_glm_4_6 | 24/30 (80%) | 12/30 (40%) | 18/33 (55%) | 59% | 3.4 t/s |
| ingest_glm_4_6 | 29/30 (97%) | 13/30 (43%) | 19/33 (58%) | 60% | 3.7 t/s |

**Key Characteristics:**
- Strong at reasoning/thinking tasks
- Weak at agentic/tool-calling tasks
- Verbose outputs often truncated at 2048 tokens
- Quality below Qwen alternatives (59% vs 65-70% for comparable Qwen models)

### Current Acceleration Status

| Method | Baseline | Optimized | Speedup | Status |
|--------|----------|-----------|---------|--------|
| MoE Expert Reduction (8→4) | 2.24 t/s | 3.97 t/s | **+77%** | ✅ Working |
| MTP (1 spec token) | 3.97 t/s | ~5.2 t/s? | +30%? | ❓ Untested |

### Why MTP Could Help Further

GLM-4.6 is an **MoE model with MTP heads**, so:
- MoE expert reduction already works (+77%)
- MTP could potentially **stack** with MoE reduction for additional gains
- No external draft model exists with matching vocab (151k custom tokens)
- MTP is the ONLY path to further acceleration beyond MoE reduction

**Potential Combined Speedup:**
- Baseline: 2.24 t/s
- MoE 4: 3.97 t/s (+77%)
- MoE 4 + MTP (theoretical): 5.2-6.0 t/s (+130-170% total)

This would make GLM-4.6 competitive with Qwen3-235B at 6.75 t/s.

---

## Why We Haven't Considered MTP Yet

### 1. llama.cpp PR Not Merged

[PR #15225](https://github.com/ggml-org/llama.cpp/pull/15225) implements GLM-style MTP but:
- Status: **Open** (ready for review as of December 21, 2025)
- Not yet merged into main branch
- Our llama.cpp build does not have the `--mtp` flag

### 2. MTP Tensors Skipped by Default

From llama.cpp loader behavior:
- MTP layers (NextN) are skipped by default to save VRAM
- Requires explicit `--mtp` or similar flag to load
- No visual indication during model load that MTP is available

### 3. Not Automatic

Unlike some optimizations, MTP does not activate automatically:
- vLLM: Requires `--speculative-config.method mtp`
- llama.cpp: Will require `n_mtp` parameter in context params
- We never knew to enable it

### 4. Focus on Other Acceleration Methods

Our research prioritized:
- Track 1: External draft models (11x achieved)
- Track 2: MoE expert reduction (87% speedup on Qwen3-235B, **77% on GLM-4.6**)
- Track 8: Prompt lookup (12.7x)

**Note (CORRECTED):** GLM-4.6 is NOT dense - it's MoE and already benefits from expert reduction. MTP was overlooked because it requires PR #15225 which isn't merged.

---

## Expected Outcomes

### Performance Benchmarks from Other Implementations

| Implementation | Acceptance Rate | Speedup | Configuration |
|----------------|-----------------|---------|---------------|
| vLLM (GLM-4.7) | >90% | Up to 60% | `num_speculative_tokens=1` |
| llama.cpp (dev) | ~70% | 20-25% | Standard samplers |
| llama.cpp (creative) | 7-11% | Negative | Quantized + high temp |

### Conservative Estimates for GLM-4.6

| Scenario | Current | Expected | Speedup |
|----------|---------|----------|---------|
| Baseline | 2.24 t/s | - | - |
| MTP (n=1) | - | 2.7-2.9 t/s | +20-30% |
| MTP (optimized) | - | 3.0-3.5 t/s | +35-55% |

### Caveats

1. **Quantization Impact**: MTP acceptance drops significantly with quantized models + creative sampling
2. **Implementation Maturity**: llama.cpp PR is development preview, not production-ready
3. **Hardware Dependence**: Performance varies with hardware; MoE vs dense affects overhead

---

## Recommendations (REVISED 2026-01-08)

### Critical Finding: llama.cpp PR #15225 Has Architectural Issues

The PR author explicitly states the implementation is "not expected to outperform baseline" due to:

1. **Sequential token-by-token processing** - defeats MTP purpose
   ```cpp
   // Current PR - broken pattern
   for (int i = 0; i < n_draft; ++i) {
       llama_decode(ctx, mtp_batch);  // ONE token per decode = GEMV
   }
   ```

2. **Alternating pass pattern** - limits to 1.5 tokens/pass vs theoretical 2

3. **Low acceptance rates** - 7-11% reported with quantization + creative sampling

### Path Forward

#### Option A: Test PR #15225 As-Is (LOW EFFORT)
1. Fetch and build PR branch
2. Test on GLM-4.6 with MoE 4 to measure actual performance
3. Determine if the sequential bottleneck is as bad as claimed on EPYC

#### Option B: Optimize PR #15225 (HIGH EFFORT)
If testing shows MTP fundamentally works but is slow:
1. Replace sequential draft loop with batched processing
2. Implement parallel verification (like vLLM/SGLang)
3. Contribute optimizations to upstream or maintain in fork

#### Option C: Use vLLM for MTP (MEDIUM EFFORT)
If llama.cpp MTP is unworkable:
1. Install vLLM with CPU support
2. Benchmark GLM-4 MTP in vLLM
3. Use vLLM for GLM models if speedup is significant

### Registry Updates Needed

1. ~~Correct GLM-4.6 from "dense" to "MoE"~~ DONE - it was already correct in registry
2. Add note that MTP exists but is untested
3. Update model_registry.yaml with MTP potential if testing succeeds

---

## MTP Refactoring Plan (2026-01-08)

### Investigation Summary

Deep analysis of PR #15225, existing speculative patterns, and vLLM/SGLang implementations confirms the sequential bottleneck and identifies concrete refactoring approach.

### Critical Finding: Sequential Bottleneck Confirmed

PR #15225's core problem in `examples/speculative/speculative.cpp`:

```cpp
// BOTTLENECK: Sequential loop - each draft = full forward pass
for (int i = 0; i < n_draft; ++i) {
    mtp_batch.n_tokens = 0;
    common_batch_add(mtp_batch, current_input_id, current_n_past, {seq_id}, true);
    llama_decode(ctx, mtp_batch);  // ONE token per decode = GEMV (memory-bound)
    llama_token id_next = common_sampler_sample_speculative(smpl, ctx, 0);
    current_input_id = id_next;    // DATA DEPENDENCY prevents batching here!
    current_n_past++;
}
```

**Why this is slow on CPU:**
- Each `llama_decode()` with batch=1 becomes GEMV (matrix-vector multiply)
- GEMV is memory-bound: reads entire MTP head weights for ONE token
- N draft tokens = N full weight reads = N x memory latency
- vLLM reports 1.5 tokens/pass vs 2.0 theoretical with this pattern

### vLLM/SGLang's Solution: Parallel Verification

```
Standard (PR #15225):
  Draft[0] -> decode -> sample -> Draft[1] -> decode -> sample -> ... (sequential)

Parallel Verification (vLLM/SGLang):
  1. Draft ALL tokens in ONE batched call (or use cached logits)
  2. Verify ALL drafts against main model in ONE batched forward pass
  3. Accept longest matching prefix
  4. Repeat from rejection point
```

### Concrete Refactoring Approach

**Phase A: Fix the Draft Loop (MEDIUM effort)**

Replace sequential drafting with logit caching:

```cpp
// NEW: Use main model's last-token logits to get first draft
llama_token draft_tokens[n_draft];
draft_tokens[0] = sample_from_logits(main_model_logits);

// For subsequent drafts: batch process MTP head
common_batch mtp_batch;
for (int i = 0; i < n_draft - 1; ++i) {
    common_batch_add(mtp_batch, draft_tokens[i], pos + i, {seq_id}, true);
}
llama_decode(mtp_ctx, mtp_batch);  // ONE decode for all drafts

// Sample all draft positions from batched logits
for (int i = 1; i < n_draft; ++i) {
    draft_tokens[i] = sample_from_logits(logits_at_position(i-1));
}
```

**Phase B: Parallel Verification (MEDIUM effort)**

```cpp
// Instead of alternating main/MTP, verify ALL drafts at once
common_batch verify_batch;
for (int i = 0; i < n_draft; ++i) {
    common_batch_add(verify_batch, draft_tokens[i], pos + i, {seq_id}, true);
}
llama_decode(main_ctx, verify_batch);  // ONE decode verifies all

// Compare logits: accept longest matching prefix
int accepted = 0;
for (int i = 0; i < n_draft; ++i) {
    llama_token main_token = sample_from_logits(main_logits[i]);
    if (main_token == draft_tokens[i]) {
        accepted++;
    } else {
        break;  // Autoregressive: must stop at first mismatch
    }
}
```

**Phase C: Unified Forward Pass (HIGH effort, future)**

Long-term: modify GGML graph to run main trunk + MTP head in single pass.

### Files to Modify (Priority Order)

| File | Change | Effort |
|------|--------|--------|
| `examples/speculative/speculative.cpp` | Replace sequential loop | MEDIUM |
| `common/speculative.cpp` | Add batched MTP functions | MEDIUM |
| `src/llama-batch.cpp` | Handle MTP position offsets | LOW |
| `include/llama.h` | Add `llama_mtp_batch_*` API | LOW |

### Expected Performance

| Scenario | Current (PR #15225) | Refactored | Improvement |
|----------|---------------------|------------|-------------|
| Draft 2 tokens | 2 forward passes | 1 batched pass | 50% fewer passes |
| Draft 4 tokens | 4 forward passes | 1-2 batched passes | 60-75% fewer passes |
| Verification | Sequential | 1 batched pass | Same as draft count |

**Conservative estimate:** 30-40% speedup over PR #15225
**Optimistic estimate:** 50-60% speedup (matching vLLM)

### Blockers & Next Steps

1. **Wait for PR #15225 to merge** - Cannot test without MTP loader support
2. **Cherry-pick loader code** - Alternative: cherry-pick just the tensor loading into fork
3. **Test baseline** - Measure PR #15225's actual performance before optimizing
4. **Implement refactoring** - Apply batched drafting and parallel verification
5. **Contribute upstream** - If successful, offer to help fix #15225 or submit new PR

### Related Research

- AVX-512 VNNI Q8_0 optimization tested: 8% speedup on small models, 0% on larger (not submitting PR - bottleneck is elsewhere)
- vec_dot IS called for single-token inference (confirmed via objdump - 541 VNNI instructions in binary)
- Memory bandwidth is NOT the bottleneck (only using 17-23% of theoretical)

---

## References

- [llama.cpp PR #15225: Implement GLM-style MTP](https://github.com/ggml-org/llama.cpp/pull/15225)
- [vLLM GLM-4.X Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html)
- [llama.cpp Discussion #12130: Speculative Decoding / MTP](https://github.com/ggml-org/llama.cpp/discussions/12130)
- [Meta Research: Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [SGLang MTP Integration](https://lmsys.org/blog/2025-07-17-mtp/)
- [FastMTP: Enhanced Multi-Token Prediction](https://arxiv.org/html/2509.18362)

---

## Appendix: MTP vs Traditional Speculative Decoding

### When to Use MTP
- Model has built-in MTP heads (GLM-4 series, DeepSeek V3)
- No compatible external draft model available
- VRAM constrained (cannot load draft model)

### When to Use Traditional Spec Decode
- High-quality draft model available (jukofyork, Qwen-0.5B)
- Target model lacks MTP heads
- Higher speedups possible (11x vs 1.6x)

### Hybrid Approaches
Some research explores combining MTP with external drafts for even higher acceptance rates. Not yet implemented in llama.cpp.
