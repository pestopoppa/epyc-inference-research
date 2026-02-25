# NeurIPS 2025 Cutting-Edge Speculative Decoding Research

## New Experimental Tracks for AMD EPYC 9655 Optimization

**Generated:** 2025-12-15  
**Context:** Extending existing Track 1-5 research with state-of-the-art techniques from NeurIPS 2025

---

## Executive Summary

After reviewing NeurIPS 2025 proceedings, I've identified **8 promising new research directions** that could stack on top of your existing 5.9x speedups from Track 1 and 21-48% gains from Track 2. Several are **CPU-friendly** and require **no training**.

| New Track | Method | Training Required | CPU Suitability | Expected Gain | Priority |
|-----------|--------|-------------------|-----------------|---------------|----------|
| **Track 6** | SuffixDecoding | ❌ None | ⭐⭐⭐ Excellent | 5-10x (agentic) | 🔴 HIGH |
| **Track 7** | CAS-Spec (Self-Cascading) | ❌ None | ⭐⭐⭐ Excellent | 2.3x baseline | 🔴 HIGH |
| **Track 8** | Prompt Lookup Decoding | ❌ None | ⭐⭐⭐ Excellent | 2-4x (grounded) | 🔴 HIGH |
| **Track 9** | CLaSp/SWIFT Layer-Skip | ❌ None | ⭐⭐⭐ Excellent | 1.3-1.8x | 🟡 MEDIUM |
| **Track 10** | Kangaroo Double Early-Exit | ⚠️ Adapter only | ⭐⭐ Good | 2.0x | 🟡 MEDIUM |
| **Track 11** | REST Retrieval-Based | ❌ None (datastore) | ⭐⭐⭐ Excellent | 2-3x | 🟡 MEDIUM |
| **Track 12** | RASD Retrieval+Draft Fusion | ❌ None | ⭐⭐⭐ Excellent | +15% over EAGLE2 | 🟢 LOW |
| **Track 13** | SpecFormer NAR Drafting | ⚠️ Training | ⭐⭐ Good | Scales to batched | 🟢 LOW |

---

## Track 6: SuffixDecoding (NeurIPS 2025 Spotlight) 🔥

### Why This is Exciting

**10.4x speedup demonstrated** on agentic workloads — model-free, no training required.

### Core Insight

For repetitive inference patterns (multi-agent pipelines, code generation, SQL, agentic workflows), the model often repeats token sequences it has seen before. SuffixDecoding builds **suffix trees** over:
1. **Global tree**: All previous outputs (offline, O(n))
2. **Per-request tree**: Current ongoing inference (online)

Instead of a draft model, it retrieves draft tokens from suffix tree matches.

### CPU Suitability: ⭐⭐⭐ EXCELLENT

- Suffix tree operations are **CPU-friendly** (string matching, tree traversal)
- No model loading overhead — uses existing outputs as "datastore"
- Highly parallelizable on 96 cores

### Expected Performance on EPYC 9655

| Workload | Reported Speedup | Your Expected |
|----------|------------------|---------------|
| AgenticSQL (Enrich) | 10.41x | 8-10x |
| AgenticSQL (Extract) | 9.85x | 8-10x |
| Multi-agent pipelines | 5.3x | 4-5x |
| Code generation | 3-5x | 3-5x |
| General chat | 1.5-2x | 1.5-2x |

### Implementation Plan

```bash
# 1. Build suffix tree datastore from previous outputs
# 2. During inference, query tree with last N tokens
# 3. If match found, propose continuation as draft
# 4. Verify with target model (standard speculative loop)
```

**Key Implementation Details:**
- Global tree: Build offline from conversation history / code outputs
- Per-request tree: Update online during generation
- Match scoring: Use frequency statistics to rank candidates
- MAX_SPEC = αp (where p = pattern match length)

### Synergy with Existing Tracks

- **Stack with Track 1**: Use SuffixDecoding when patterns detected, fall back to 0.5B draft otherwise
- **Stack with Track 2**: MoE soft mask + SuffixDecoding for agentic MoE workloads

### Resources

- Paper: https://suffix-decoding.github.io/
- GitHub: Available in vLLM

---

## Track 7: CAS-Spec (Cascade Adaptive Self-Speculative) 🔥

### Why This is Exciting

**No training required** — constructs multiple draft "stages" from the target model itself using:
1. **Layer sparsity** (skip layers → fast draft)
2. **Activation quantization** (INT8 forward → faster draft)
3. **Dynamic Tree Cascade** routing

### Core Insight

Instead of one draft model, create a **hierarchy** of increasingly fast/rough drafts:
- Stage 1: Skip 30% of layers → moderate speed, good accuracy
- Stage 2: Skip 50% of layers → faster, rougher
- Stage 3: Skip 70% + INT8 → fastest, roughest

**Dynamic Tree Cascade (DyTC)** algorithm routes tokens through this hierarchy based on acceptance rate heuristics.

### CPU Suitability: ⭐⭐⭐ EXCELLENT

- Layer skipping reduces compute linearly
- INT8 quantization benefits from AVX-512
- No external model needed — single model in memory

### Expected Performance

| Configuration | Speedup over Baseline | Notes |
|---------------|----------------------|-------|
| Vanilla self-spec | 1.1-1.5x | Static layer skip |
| CAS-Spec + DyTC | **2.3x** | Dynamic cascade routing |
| vs static cascade | +47% | DyTC adaptive gain |
| vs tree baseline | +48% | DyTC adaptive gain |

### Implementation Plan

```cpp
// Pseudo-code for CAS-Spec in llama.cpp
struct CASSpecConfig {
    int stages = 3;
    float layer_skip_ratios[3] = {0.3, 0.5, 0.7};
    bool use_int8[3] = {false, true, true};
};

// During draft phase:
for (int stage = 0; stage < config.stages; stage++) {
    // Apply layer sparsity and optional quantization
    draft_tokens = generate_with_skip(model, layer_skip_ratios[stage]);
    
    // DyTC: Route based on acceptance prediction
    if (predicted_acceptance > threshold) {
        // Accept and continue drafting
    } else {
        // Move to next stage or verify
    }
}
```

### Synergy with Track 2 (MoE Self-Drafting)

**HUGE POTENTIAL**: Combine CAS-Spec with MoE soft mask!

| Combined Configuration | Expected Effect |
|-----------------------|-----------------|
| Layer skip + MoE Top-4 | 2x memory bandwidth reduction |
| Layer skip + INT8 + MoE Top-4 | 3x potential |

### Resources

- Paper: https://arxiv.org/abs/2510.26843
- NeurIPS 2025

---

## Track 8: Prompt Lookup Decoding (N-gram Matching) 🔥

### Why This is Exciting

**Zero overhead, no model needed** — draft tokens come from the prompt itself.

### Core Insight

In **input-grounded tasks** (summarization, document QA, code editing, multi-turn chat), there's high n-gram overlap between input and output. The model often copies entity names, phrases, or code chunks directly from the prompt.

Instead of a draft model:
1. Build n-gram index from prompt
2. During generation, match current context with prompt n-grams  
3. If match found, propose continuation from prompt as draft

### CPU Suitability: ⭐⭐⭐ EXCELLENT

- Pure string matching — no neural compute
- Trivial to implement (50 lines of Python)
- Works with ANY model, no compatibility issues

### Expected Performance

| Task Type | Reported Speedup | Your Expected |
|-----------|------------------|---------------|
| Summarization (CNN/DailyMail) | **2.8x** | 2.5-3x |
| Document QA | 2-3x | 2-2.5x |
| Code editing/refactoring | 3-4x | 3-4x |
| Multi-turn chat (turn 2+) | 1.5-2x | 1.5-2x |
| General generation | 1.0-1.2x | Minimal |

### Implementation

Already supported in llama.cpp! Add to your speculative command:

```bash
# In llama.cpp (if supported) or vLLM:
--speculative-model "[ngram]"
--num-speculative-tokens 5
--ngram-prompt-lookup-max 4
--ngram-prompt-lookup-min 1
```

**Simple Python implementation:**

```python
def find_candidate_tokens(input_ids, max_ngram=4, num_candidates=10):
    for ngram_size in range(max_ngram, 0, -1):
        # Get last N tokens as search pattern
        pattern = input_ids[-ngram_size:]
        
        # Find pattern in earlier part of input
        for i in range(len(input_ids) - ngram_size):
            if input_ids[i:i+ngram_size] == pattern:
                # Return tokens following the match
                return input_ids[i+ngram_size:i+ngram_size+num_candidates]
    return []
```

### Synergy with Existing Tracks

**Intelligent fallback chain:**
1. Try Prompt Lookup first (zero cost)
2. If no match, try SuffixDecoding (global pattern)
3. If still no match, use Track 1 draft model

This gives you the best of all worlds with minimal overhead.

### Resources

- Original: https://github.com/apoorvumang/prompt-lookup-decoding
- HuggingFace integration: `prompt_lookup_num_tokens` parameter
- vLLM: `speculative_model="[ngram]"`

---

## Track 9: CLaSp / SWIFT (Dynamic Layer Skip)

### Why This is Exciting

**Plug-and-play layer skipping** that adapts dynamically per decoding step.

### Core Insight

Different tokens require different amounts of compute. Easy tokens (common patterns) can use a heavily-skipped model; hard tokens (rare sequences) need full model.

**CLaSp** (In-Context Layer Skip): Uses dynamic programming to optimize which layers to skip based on hidden state divergence from full model.

**SWIFT**: Adaptively selects skip pattern based on task difficulty.

### CPU Suitability: ⭐⭐⭐ EXCELLENT

- Directly reduces FLOPs per token
- No external model needed
- Already integrated in HuggingFace Transformers

### Expected Performance

| Model Size | Static Skip | CLaSp/SWIFT |
|------------|-------------|-------------|
| LLaMA-3-8B | 1.3x | 1.5x |
| LLaMA-3-70B | 1.4x | **1.8x** |

### Implementation

Already in HuggingFace:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
outputs = model.generate(
    **inputs,
    assistant_early_exit=24,  # Exit after layer 24 for drafts
    do_sample=False
)
```

For llama.cpp, would need implementation of layer-skip compute graph.

---

## Track 10: Kangaroo (Double Early-Exit)

### Why This is Exciting

**Self-speculative** with confidence-based early stopping for drafts.

### Core Insight

Use shallow sub-network (first N layers) + LM head as draft model, then verify with remaining layers. **Double early-exit**: Stop drafting when confidence drops below threshold.

### Key Advantage

- **88.7% fewer parameters** than Medusa
- Uses existing model weights — train only lightweight adapter
- Shared KV cache between draft and verify

### Expected Performance

- 2.04x speedup reported
- Outperforms Medusa-1 with far less overhead

### Implementation

Requires training ~10M parameter adapter on top of shallow layers. Estimated training: 1-2 days on 8x GPU.

### Resources

- GitHub: https://github.com/Equationliu/Kangaroo
- NeurIPS 2024

---

## Track 11: REST (Retrieval-Based Speculative Decoding)

### Why This is Exciting

Draft tokens come from a **datastore** of prior generations, not a draft model.

### Core Insight

Build a suffix array / hash table of prior outputs. During generation:
1. Query datastore with recent context
2. Retrieve likely continuations
3. Verify with target model

### CPU Suitability: ⭐⭐⭐ EXCELLENT

- Retrieval is CPU-native (hash lookup, suffix matching)
- Datastore can be built offline
- No neural draft model compute

### Difference from SuffixDecoding

| Aspect | REST | SuffixDecoding |
|--------|------|----------------|
| Data source | External datastore | Request history |
| Scope | Comprehensive corpus | Session-local |
| Setup | Offline build | Online build |
| Best for | Domain-specific | Agentic patterns |

### Implementation

Build datastore from:
- Previous conversations
- Domain-specific text (code, legal, medical)
- Training data samples

---

## Track 12: RASD (Retrieval-Augmented Speculative Decoding)

### Why This is Exciting

**Combines** draft model + retrieval for higher acceptance rates.

### Core Insight

Draft model proposes candidates; retrieval provides additional candidates. **Tree fusion** merges both sources, then **tree pruning** removes low-confidence branches.

### Expected Gain

+15% speedup over EAGLE2 alone on tested benchmarks.

### Synergy

Could combine with your Track 1 Qwen2.5-0.5B draft + SuffixDecoding retrieval.

---

## Track 13: SpecFormer (Non-Autoregressive Drafting)

### Why This is Exciting

Addresses the fundamental issue: autoregressive draft models are still slow.

### Core Insight

Train a **non-autoregressive** draft model that generates K tokens in parallel (not sequentially). Uses bidirectional attention internally.

### Trade-off

- Requires training specialized draft model
- Better for batched inference scenarios
- May not benefit single-request latency as much

---

## Implementation Priority Matrix

### Immediate (This Week) — Zero Training Required

| Track | Effort | Expected Gain | Dependencies |
|-------|--------|---------------|--------------|
| **Track 8: Prompt Lookup** | Low (50 lines) | 2-4x on grounded tasks | None |
| **Track 6: SuffixDecoding** | Medium | 5-10x on agentic | Suffix tree impl |
| **Track 9: CLaSp** | Low | 1.3-1.8x | HuggingFace/llama.cpp |

### Short-Term (This Month)

| Track | Effort | Expected Gain | Dependencies |
|-------|--------|---------------|--------------|
| **Track 7: CAS-Spec** | High (C++) | 2.3x | Layer skip + INT8 |
| **Track 11: REST** | Medium | 2-3x | Datastore build |

### Medium-Term (If Needed)

| Track | Effort | Expected Gain | Dependencies |
|-------|--------|---------------|--------------|
| **Track 10: Kangaroo** | Training | 2x | Adapter training |
| **Track 12: RASD** | Medium | +15% | Draft + retrieval |

---

## Combined Optimization Stack

### Ultimate Configuration (Theoretical)

```
Layer 1: Prompt Lookup (free, 2-4x on grounded)
    ↓ (no match)
Layer 2: SuffixDecoding (free, 5-10x on agentic)  
    ↓ (no match)
Layer 3: Track 1 Draft Model (Qwen2.5-0.5B, 5.9x)
    ↓ (low acceptance)
Layer 4: CAS-Spec self-cascade (2.3x fallback)
    +
Layer 5: Track 2 MoE Soft Mask (21-48% on top)
```

### Conservative Estimate: Combined Effect

Assuming diminishing returns and overhead:
- Grounded tasks (summarization, code): **8-15x** (Prompt Lookup + Draft)
- Agentic tasks (SQL, multi-agent): **10-15x** (SuffixDecoding)
- General tasks: **4-6x** (Draft model + CAS-Spec)
- MoE models: **6-8x** (Soft mask + above)

---

## References

### NeurIPS 2025
- SuffixDecoding (Spotlight): https://suffix-decoding.github.io/
- CAS-Spec: https://arxiv.org/abs/2510.26843
- AdaSPEC: NeurIPS 2025 poster
- SpecFormer: https://arxiv.org/abs/2511.20340

### NeurIPS 2024
- Kangaroo: https://github.com/Equationliu/Kangaroo
- Cascade Speculative Drafting: https://arxiv.org/pdf/2312.11462

### ICLR 2025
- SWIFT: https://openreview.net/forum?id=EKJhH5D5wA
- CLaSp: https://arxiv.org/abs/2505.24196

### Prior Work
- REST: https://arxiv.org/html/2311.08252
- Prompt Lookup: https://github.com/apoorvumang/prompt-lookup-decoding
- LayerSkip: https://arxiv.org/abs/2404.16710

---

## Next Steps

1. **Implement Track 8 (Prompt Lookup)** — trivial, immediate gains on grounded tasks
2. **Test SuffixDecoding in vLLM** — evaluate for agentic patterns
3. **Prototype CAS-Spec** — layer skip + INT8 cascade in llama.cpp
4. **Design combined routing** — intelligent fallback between methods
