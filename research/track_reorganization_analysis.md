# Track Reorganization: Compounding Gains vs Mutual Exclusivity

## Analysis Framework

The key question: **Which techniques can stack together vs. which compete for the same speedup?**

---

## Mutual Exclusivity Analysis

### The Core Insight: Draft Token Sources

Every speculative decoding method answers: **"Where do draft tokens come from?"**

| Source Type | Methods | Can Combine? |
|-------------|---------|--------------|
| **External Small Model** | Track 1 (Qwen2.5-0.5B) | ❌ Mutually exclusive with other draft sources |
| **Self-Model (Layer Skip)** | Track 9 (CLaSp/SWIFT), Track 7 (CAS-Spec) | ❌ Mutually exclusive with external model |
| **Self-Model (EAGLE/Medusa)** | Track 3 (EAGLE), Track 4 (Medusa) | ❌ Mutually exclusive with external model |
| **Retrieval (Prompt)** | Track 8 (Prompt Lookup) | ✅ Can augment any draft method |
| **Retrieval (Datastore)** | Track 6 (SuffixDecoding), Track 11 (REST) | ✅ Can augment any draft method |
| **MoE Gating Reduction** | Track 2 (Soft/Hard Mask) | ✅ Orthogonal - applies to target model |

### Key Relationships

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DRAFT TOKEN SOURCE (Pick ONE)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Option A: External Draft Model (Track 1)                            │
│            └── Qwen2.5-0.5B → 5.9x proven                           │
│                                                                      │
│  Option B: Self-Draft via Layer Skip (Track 7/9)                     │
│            └── CAS-Spec / CLaSp → 1.5-2.3x expected                 │
│                                                                      │
│  Option C: Self-Draft via Trained Head (Track 3/10)                  │
│            └── EAGLE / Kangaroo → 2-3x (requires training/debug)    │
└─────────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────────┐
│              RETRIEVAL AUGMENTATION (Add to ANY above)               │
├─────────────────────────────────────────────────────────────────────┤
│  Track 8: Prompt Lookup (zero cost)                                  │
│           └── +2-4x on grounded tasks, STACKS with draft model      │
│                                                                      │
│  Track 6: SuffixDecoding (pattern matching)                          │
│           └── +5-10x on agentic, STACKS with draft model            │
│                                                                      │
│  Track 11: REST Datastore (offline corpus)                           │
│            └── +2-3x domain-specific, STACKS with draft model       │
└─────────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────────┐
│              TARGET MODEL OPTIMIZATION (Orthogonal)                  │
├─────────────────────────────────────────────────────────────────────┤
│  Track 2: MoE Soft/Hard Mask                                         │
│           └── 21-48% speedup on MoE models, INDEPENDENT of above    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Compounding Relationships

### ✅ STACK: Retrieval + Draft Model

**How it works:** Try retrieval first (zero cost), fall back to draft model if no match.

```python
def get_draft_tokens(context, prompt, draft_model):
    # Layer 1: Prompt Lookup (FREE)
    candidates = prompt_lookup(context, prompt, ngram_size=4)
    if candidates and len(candidates) >= 3:
        return candidates, "prompt_lookup"
    
    # Layer 2: SuffixDecoding (near-free for agentic)
    candidates = suffix_tree_lookup(context, global_tree)
    if candidates and len(candidates) >= 3:
        return candidates, "suffix"
    
    # Layer 3: Draft model (costs compute)
    candidates = draft_model.generate(context, k=8)
    return candidates, "draft_model"
```

**Expected compounding:**
- Grounded tasks: 60% prompt lookup (4x) + 40% draft (5.9x) = **~4.8x average**
- Agentic tasks: 70% suffix (8x) + 30% draft (5.9x) = **~7.4x average**

### ✅ STACK: Draft Model + MoE Mask (Track 1 + Track 2)

**How it works:** MoE soft mask speeds up the TARGET model's verification pass.

| Configuration | Draft Speed | Target Speed | Combined |
|---------------|-------------|--------------|----------|
| Track 1 alone | 85 t/s | 24.8 t/s | 5.9x |
| Track 1 + Track 2 | 85 t/s | 30.0 t/s (+21%) | **~7.1x** |

**Why it stacks:** Draft model speed unchanged, but verification is faster → more tokens/sec.

### ✅ STACK: Retrieval + MoE Mask (Track 6/8 + Track 2)

**How it works:** Retrieval provides drafts, MoE mask speeds verification.

Even better synergy because retrieval has ~zero draft cost.

### ❌ EXCLUSIVE: External Draft vs Self-Draft

You cannot use both Qwen2.5-0.5B (Track 1) AND CAS-Spec layer-skip (Track 7) simultaneously. Pick one.

**Decision criteria:**
- Track 1 wins when: You have a well-matched draft model (same tokenizer, high acceptance)
- Track 7/9 wins when: No compatible draft available, or want zero-dependency solution

### ❌ EXCLUSIVE: EAGLE vs Medusa vs Kangaroo

All three are trained internal prediction heads. Pick one (if any).

---

## Reorganized Track Priority

### Tier 1: Implement Immediately (Compounds with Track 1)

| New Track | Effort | Compounds With | Expected Boost |
|-----------|--------|----------------|----------------|
| **Track 8: Prompt Lookup** | 1 hour | Track 1, Track 2 | +50-100% on grounded |
| **Track 6: SuffixDecoding** | 1 day | Track 1, Track 2 | +100-200% on agentic |

**Why Tier 1:** Zero/low cost, stacks multiplicatively with your existing 5.9x.

### Tier 2: Implement This Week (Compounds with Track 2)

| Track | Effort | Compounds With | Expected Boost |
|-------|--------|----------------|----------------|
| **Track 2 + Track 1 Integration** | 2 hours | Already have both | 21% on MoE targets |

**Why Tier 2:** You've already implemented Track 2 soft mask. Just need to test it combined with Track 1 speculative.

### Tier 3: Implement If Needed (Alternative to Track 1)

| Track | Effort | When to Use | Expected Speedup |
|-------|--------|-------------|------------------|
| **Track 7: CAS-Spec** | 1 week (C++) | No compatible draft model | 2.3x baseline |
| **Track 9: CLaSp/SWIFT** | 2 days | Quick self-spec fallback | 1.5-1.8x baseline |

**Why Tier 3:** Only needed if Track 1 fails for a specific model family (e.g., DeepSeek-R1 with no compatible draft).

### Tier 4: Deprioritize (Training Required / Blocked)

| Track | Status | Reason |
|-------|--------|--------|
| **Track 3: EAGLE-1** | ⛔ BLOCKED | 0% acceptance, deep debugging needed |
| **Track 10: Kangaroo** | ⚠️ Training | Requires adapter training |
| **Track 4: Medusa** | ⚠️ Training | Requires head training |
| **Track 13: SpecFormer** | ⚠️ Training | NAR draft requires training |

---

## EAGLE Deprecation Assessment

### Current EAGLE Status

| Metric | Value |
|--------|-------|
| Time invested | ~20+ hours debugging |
| Acceptance rate | 0% (blocked) |
| Root causes identified | 5 falsified, 4+ remaining |
| Estimated time to fix | Unknown (days to weeks) |
| Alternative speedup available | 5.9x already working |

### Cost-Benefit Analysis

**Option A: Continue EAGLE Debugging**
```
Investment: 10-40 more hours
Expected outcome: 2-3x IF it works
Risk: May never work (checkpoint incompatibility)
Opportunity cost: Not implementing Tier 1-2 tracks
```

**Option B: Pivot to Tier 1-2 Tracks**
```
Investment: 4-8 hours
Expected outcome: 
  - Prompt Lookup: +50-100% on grounded tasks
  - SuffixDecoding: +100-200% on agentic tasks  
  - Track 2 integration: +21% on MoE
Risk: Low (well-documented techniques)
Opportunity cost: EAGLE stays blocked
```

### Recommendation: **Deprecate EAGLE, Pivot to Tier 1-2**

**Rationale:**

1. **EAGLE is blocked on fundamentals** — The 0% acceptance with correct attention output suggests checkpoint incompatibility or layer extraction bugs that may require PyTorch reference comparison or retraining.

2. **You already have 5.9x** — Track 1 is working. EAGLE's 2-3x potential doesn't exceed what you have.

3. **Tier 1 tracks compound** — Prompt Lookup + SuffixDecoding on TOP of Track 1 could yield 8-12x on specific workloads with 1-2 days work.

4. **EAGLE requires exact model match** — Even if fixed, you'd need EAGLE checkpoints for each model family. Tier 1-2 tracks are model-agnostic.

### EAGLE Resurrection Criteria

Revisit EAGLE only if:
1. llama.cpp adds official EAGLE support (PR #13908)
2. Pre-trained EAGLE checkpoints become available for Qwen2.5/Qwen3 families
3. You need to squeeze beyond 10x and all other tracks are exhausted

---

## Final Reorganized Track List

```
PRODUCTION (Working Now)
├── Track 1: External Draft Model ────────────────── 5.9x ✅
│   └── Qwen2.5-Coder-32B + Qwen2.5-0.5B
└── Track 2: MoE Soft Mask ───────────────────────── +21-48% ✅
    └── --override-kv expert_used_count=4

IMPLEMENT THIS WEEK (Compounds with above)
├── Track 8: Prompt Lookup ───────────────────────── +50-100% (grounded)
│   └── Zero cost, try before draft model
├── Track 6: SuffixDecoding ──────────────────────── +100-200% (agentic)
│   └── Build suffix tree from session history
└── Track 1+2 Integration ────────────────────────── Combined 7x+
    └── Test soft mask on MoE targets with spec decode

IMPLEMENT IF NEEDED (Alternatives)
├── Track 7: CAS-Spec ────────────────────────────── 2.3x (no draft model)
│   └── Layer skip cascade, for DeepSeek-R1 etc
└── Track 9: CLaSp/SWIFT ─────────────────────────── 1.5-1.8x (quick fallback)
    └── Dynamic layer skip per token

DEPRECATED (Blocked/Training Required)
├── Track 3: EAGLE-1 ─────────────────────────────── ⛔ 0% acceptance
├── Track 4: Medusa ──────────────────────────────── ⚠️ Training required
├── Track 10: Kangaroo ───────────────────────────── ⚠️ Adapter training
└── Track 5: SSM Speculation ─────────────────────── ⛔ Architecture incompatible
```

---

## Implementation Plan

### Day 1: Prompt Lookup Integration
```bash
# Test if llama.cpp supports it natively
./llama-speculative --help | grep -i ngram

# If not, implement Python wrapper:
# 1. Tokenize prompt
# 2. During generation, match last N tokens against prompt
# 3. If match, propose continuation from prompt
# 4. Feed to target model for verification
```

### Day 2: SuffixDecoding Prototype
```python
# Build suffix tree from session outputs
from suffix_trees import STree

class SuffixDraftProvider:
    def __init__(self):
        self.global_tree = STree("")
        self.session_outputs = []
    
    def add_output(self, text):
        self.session_outputs.append(text)
        # Rebuild tree periodically
        if len(self.session_outputs) % 10 == 0:
            self.global_tree = STree("".join(self.session_outputs))
    
    def get_drafts(self, context, k=8):
        # Find longest match in tree
        match = self.global_tree.find_longest_match(context[-32:])
        if match and len(match.continuation) >= k:
            return match.continuation[:k]
        return None
```

### Day 3: Track 1+2 Integration Testing
```bash
# Test MoE model with both spec decode AND soft mask
numactl --interleave=all ./llama-speculative \
  -m Qwen3-VL-30B-A3B-Q4_K_M.gguf \
  -md Qwen3_VL_2B-Q4_K_M.gguf \
  --draft 8 \
  --override-kv qwen3vlmoe.expert_used_count=int:4 \
  -t 96 -p "test prompt"
```

---

## Expected Outcome

| Workload Type | Current | After Tier 1-2 | Improvement |
|---------------|---------|----------------|-------------|
| Code generation | 5.9x | 6-7x | +15% (prompt lookup helps) |
| Summarization | 5.9x | 8-10x | +50% (prompt lookup dominates) |
| Agentic/SQL | 5.9x | 10-15x | +100% (suffix tree dominates) |
| MoE models | 1.2x | 2-3x | +100% (soft mask + spec) |
| General chat | 5.9x | 6-7x | +15% (slight prompt gains) |

**Bottom line:** Pivot away from EAGLE. The Tier 1-2 tracks offer higher ROI with lower risk.
