# Chapter 07: Prompt Lookup (Track 8)

## Introduction

Prompt Lookup is our highest-performing technique for grounded tasks, achieving **12.7x speedup** on summarization. Unlike speculative decoding which requires a draft model, prompt lookup extracts candidate tokens directly from the input prompt using n-gram matching.

## How It Works

When the model generates a token, the system checks whether the next few tokens already appear in the input prompt. For tasks like summarization, code editing, and document QA, the output frequently quotes the input verbatim — prompt lookup exploits this for essentially free speedup since no draft model is needed.

<details>
<summary>Mechanism and example</summary>

```
Input: "The quick brown fox jumps over the lazy dog. Summarize:"
Generated: "The quick brown"
→ N-gram match found! Draft: "fox jumps over"
→ Verify against model → Accept all 3 tokens
```

**Key Insight**: Summarization, code editing, and QA tasks frequently generate text that appears verbatim in the input. Prompt lookup exploits this for free speedup (no draft model needed).

</details>

## Best Results

The speedups range from transformative (12.7x for summarization) to negligible (1x for novel generation). The pattern is clear: the more the output overlaps with the input, the bigger the win. Pure generation tasks with no source material see almost no benefit — unless you use corpus augmentation (see below).

<details>
<summary>Speed measurements by task type</summary>

| Task Type | Model | Baseline | With Lookup | Speedup |
|-----------|-------|----------|-------------|---------|
| Summarization | Qwen3-Next-80B | 7.5 t/s | 95.18 t/s | **12.7x** |
| Code editing | Qwen2.5-Coder-32B | 3.0 t/s | 25.82 t/s | **8.6x** |
| Document QA | Qwen2.5-72B | ~4 t/s | ~8 t/s | **2x** |
| Code generation | Any | - | - | 1.0-1.2x |
| Code generation (w/ V3 corpus) | Coder-32B | 7.3 t/s | 12.6 t/s | **1.72x** |
| Code generation (w/ V3 corpus) | Coder-30B | - | - | 1.16x |

</details>

<details>
<summary>When to use prompt lookup</summary>

| Task Type | Expected Speedup | Reasoning |
|-----------|------------------|-----------|
| Summarization | 8-13x | Output is subset of input |
| Code refactoring | 5-9x | Most code preserved |
| Document QA | 2-4x | Answers often quote source |
| Translation | 1.5-3x | Some terms preserved |
| Code generation | ~1x | Novel output, no overlap |
| Creative writing | ~1x | Novel output, no overlap |

</details>

## Configuration

The minimum n-gram size controls how aggressively the system matches. Lower values (3) catch more matches but risk false positives; higher values (5+) are more conservative. Start with 3 for grounded tasks.

<details>
<summary>N-gram size tuning</summary>

| Setting | Behavior | Best For |
|---------|----------|----------|
| `3` | Aggressive matching | Summarization, high overlap |
| `4` | Balanced | General use |
| `5+` | Conservative | Reduce false matches |

<details>
<summary>Code: quick start command</summary>

```bash
# For summarization/QA tasks with source material
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  --lookup-ngram-min 3 \
  -t 96 -f prompt_with_source_material.txt
```

</details>
</details>

## Combining with Other Techniques

Prompt lookup stacks beautifully with both MoE reduction and speculative decoding. In llama-server, spec decode takes priority — the draft model proposes tokens first, and prompt lookup fills gaps when the draft model has low confidence. The best combined result is 47.11 t/s on Qwen3-Coder-30B.

<details>
<summary>Compatibility and combination stack</summary>

| Combination | Compatible | Result |
|-------------|------------|--------|
| Lookup + MoE Reduction | ✅ Yes | **47.11 t/s** on Qwen3-Coder-30B (MoE6+spec+lookup) |
| Lookup + Speculative | ✅ Yes | **39.44 t/s** on Qwen2.5-Coder-32B (spec-first, lookup fallback) |
| Lookup + SSM | ❌ No | SSM state corruption (consecutive position requirement) |

<details>
<summary>Code: optimal draft token selection stack</summary>

```python
def get_draft_tokens(context, prompt):
    # Layer 1: Draft model (higher acceptance on novel tokens)
    drafts = draft_model.generate(context, k=8)
    if drafts.confidence > threshold:
        return drafts

    # Layer 2: Prompt Lookup fallback (FREE - zero compute)
    candidates = prompt_lookup(context, prompt, ngram_size=3)
    if candidates and len(candidates) >= 3:
        return candidates

    return drafts  # Fall through to draft regardless
```

</details>
</details>

## SSM Warning

**CRITICAL**: Do not use prompt lookup with Qwen3-Next (SSM) models. The SSM architecture requires consecutive token positions — draft token rejection corrupts the recurrent state. See [Chapter 06: MoE Optimization](06-moe-optimization.md) for SSM-safe alternatives.

## Implementation Notes

Prompt lookup requires no additional models or training. It's pure algorithmic optimization built into llama.cpp — an n-gram index of the input prompt, sub-millisecond matching, and standard verification against the main model.

<details>
<summary>Implementation details and measurement</summary>

How it works internally:
1. Building an n-gram index of the input prompt at inference start
2. After each generated token, searching for matching n-grams
3. If match found, proposing those tokens as drafts
4. Verifying with the main model (same as speculative decoding)

The overhead is minimal - index building is O(n) in prompt length.

<details>
<summary>Code: measuring effectiveness</summary>

```bash
# Run with and without lookup, compare speeds
# With lookup:
llama-cli -m MODEL.gguf --lookup-ngram-min 3 -f task.txt -n 500

# Without lookup (baseline):
llama-cli -m MODEL.gguf -f task.txt -n 500
```

</details>

If speedup is <1.3x, prompt lookup isn't worth enabling for that task type.

</details>

## Corpus-Augmented Prompt Lookup (Phase 2A)

Standard prompt lookup only matches against the user's input. For novel code generation, there's nothing to match — acceptance is ~0%. Corpus-augmented prompt stuffing solves this by injecting retrieved code snippets into the prompt before inference, expanding the n-gram search space. The result: Coder-family models gain 6-17% on novel generation, with the 480B model seeing the biggest benefit (+17% speed from +15.6pp acceptance).

<details>
<summary>Architecture and retrieval pipeline</summary>

```
User: "implement async retry with exponential backoff"
  │
  ▼
CorpusRetriever (sub-ms query)
  │  SQLite n-gram index → top-3 matching snippets
  ▼
Prompt Assembly
  │  <reference_code> [retrieved snippets, ~750 tokens] </reference_code>
  │  <user> [original request] </user>
  ▼
llama-server (--lookup)
  │  n-gram matches now hit retrieved snippets
  │  + original prompt + spec decode drafts
  ▼
Output (higher acceptance rate on novel generation)
```

</details>

<details>
<summary>Implementation and configuration</summary>

- **Index**: V3 sharded SQLite — 16 shards, 76.6M snippets, 5.4B word-level 4-grams, 651GB. Built from The Stack v1 via `scripts/corpus/build_index_v3.py`. MD5-based shard routing for deterministic query distribution.
- **Retriever**: `src/services/corpus_retrieval.py` — singleton `CorpusRetriever`, auto-detects JSON (v1) / SQLite (v2) / sharded SQLite (v3) index. Per-shard mmap for O(1) startup.
- **Prompt injection**: `build_corpus_context()` in `src/prompt_builders/builder.py`. Runs on turn 0 for lookup-enabled roles. Injects as `## Reference Code` section.
- **Telemetry**: `src/backends/llama_server.py` extracts `draft_n` / `draft_n_accepted` from llama-server timings.
- **Token Normalization**: Both index builder and retriever strip non-alphanumeric characters (except underscore) before n-gram extraction. `class Foo(Bar):` and `class foo bar` produce the same n-grams.
- **Keyword Fallback**: JSON format only. V3 sharded index uses n-gram matching exclusively — NL keyword queries hit sparsely via code comments (e.g. `"binary search tree in"` → 57 matches).

<details>
<summary>Config: model registry YAML</summary>

```yaml
runtime_defaults:
  corpus_retrieval:
    enabled: true            # Per-role: only Coder-family
    index_path: /mnt/raid0/llm/cache/corpus/v3_sharded  # V3 sharded, 76.6M snippets
    max_snippets: 3
    max_chars: 3000          # ~750 tokens budget
```

</details>
</details>

<details>
<summary>A/B results (MVP Corpus: 73K snippets, 338MB)</summary>

| Model | Task | Acceptance Δ | Speed Δ | Verdict |
|-------|------|-------------|---------|---------|
| Qwen3-Coder-480B | BST | +15.6pp (74.9→90.5%) | +17% (8.3→9.7 t/s) | **Best** |
| Qwen2.5-Coder-32B | BST | +8.7pp (84.6→93.3%) | +6% (30.8→32.7 t/s) | **Good** |
| Qwen3-Coder-480B | HTTP | +3.4pp | +9% | Positive |
| Qwen3-235B-A22B | HTTP | +6.6pp | +2% | Marginal |
| Qwen2.5-7B | HTTP | +5.3pp | +1% | Saturated |
| Qwen3-Coder-30B | BST | +2.1pp | -12% | Negative |
| Qwen3-235B-A22B | BST | -12.1pp | -17% | Negative |

**Finding**: Coder-family models benefit most. Enabled for 32B and 480B only.

</details>

<details>
<summary>V3 Full Corpus results (76.6M snippets, 651GB, 16 shards)</summary>

V3 sharded index dramatically improved on MVP results. Quality gate with 6 code-gen prompts (Claude-as-Judge):

| Model | Avg Speed Δ | Avg Acceptance Δ | Quality Δ |
|-------|------------|-------------------|-----------|
| Qwen3-Coder-30B | **+16.3%** | +5.6pp | +0.38 (neutral) |
| Qwen2.5-Coder-32B | **+72.3%** | +22.3pp | +0.04 (neutral) |
| Qwen3-Coder-480B | +1.4% | +0.8pp | +1.25 (positive) |

Per-prompt variance is high: `graph_shortest` on 32B sees +277% (Dijkstra/A* in The Stack perfectly matches model output), while `bst_iterator` sees 0% (model's BST differs from corpus patterns).

**Key finding**: V3 reversed the MVP index's -12% regression on 30B → +16% gain. The 32B model benefits most due to high base acceptance rates amplified by longer matching sequences. 480B gains are marginal since the model's own knowledge already produces high acceptance.

**Production**: 30B corpus enabled. 480B not enabled (marginal, risk/reward too low).

</details>

<details>
<summary>Phase 2B-Sidecar: CLOSED (2026-02-19)</summary>

Implemented corpus sidecar as pluggable speculative decoding source in `llama.cpp-experimental` (branch `feature/corpus-sidecar`). Instead of prompt-level injection (Phase 2A), feeds token-level n-grams directly into the speculation loop's `nc_static` cache. Acceptance rates (55-66% on 30B) capped well below Phase 2A — externally-tokenized n-grams don't align well with what the draft model proposes. Phase 2A prompt injection remains the production approach.

</details>

<details>
<summary>Phase 2B-Quality RAG: ABANDONED (2026-02-15)</summary>

Attempted to improve code quality (not just speed) by instructing the model to "study and adapt" retrieved patterns. Tested on 7B (delta -0.96) and 32B (delta -1.38) — prompt-level RAG actively hurts quality. Models either ignore the instruction or get confused by reference code. Only works with models fine-tuned for RAG (e.g., SWE-Dev-7B/32B). Phase 2A (speed-only, silent injection) remains the production approach.

</details>

## Closed Investigations

<details>
<summary>Q3: First-20-token re-query — CLOSED (2026-02-19)</summary>

**Question**: Does re-querying the corpus with the model's first ~20 generated tokens improve over keyword-only NL retrieval?

**Ablation** (6 quality gate prompts, 32B outputs, V3 index):
- Keyword NL extraction (production): 21 gram hits
- First-20-token re-query: 53 gram hits (+152%)
- Full output n-grams (ceiling): 981 gram hits (+4571%)

**Decision**: Not worth implementing. The model's own generated code provides 47x more n-gram material than any re-query (981 vs 53 hits), and this is free via prompt lookup's self-matching against prior output. Re-query latency (185ms, 2.3 tokens at 12.6 t/s) costs more than the marginal benefit.

</details>

<details>
<summary>Q5: SoftMatcha v2 soft matching — CLOSED (2026-02-19)</summary>

**Question**: Can GloVe/FastText word embeddings enable fuzzy corpus matching ("calculate" ≈ "compute") via SoftMatcha v2?

**Findings**: Coverage was higher than expected (GloVe 79%, FastText 86%), but dominated by trivially matchable tokens (operators, English keywords). Code-specific compound identifiers (`self.assertEqual`, `camelCase`) had <3% FastText coverage. SoftMatcha requires consecutive token matching — NL query phrases never appear consecutively in code, returning 0 matches at all thresholds (1.0 to 0.5). Soft matches on code tokens are noise (`for` ≈ `return` at 0.53 in GloVe — meaningless).

**Decision**: SoftMatcha architecturally unsuitable for code retrieval. Exact n-gram matching via V3 SQLite remains the correct approach.

</details>

<details>
<summary>References</summary>

### Prompt Lookup and N-gram Methods

1. Saxena, A. (2023). *Prompt Lookup Decoding*. GitHub Repository. https://github.com/apoorvumang/prompt-lookup-decoding

2. Yang, N., Ge, T., Wang, L., Jiao, B., Jiang, D., Yang, L., ... & Wei, F. (2023). *Inference with Reference: Lossless Acceleration of Large Language Models*. arXiv preprint. https://arxiv.org/abs/2304.04487

### Retrieval-Based Speculative Decoding

3. He, Z., Zhong, Z., Cai, T., Lee, J., & He, D. (2023). *REST: Retrieval-Based Speculative Decoding*. NAACL 2024. https://arxiv.org/abs/2311.08252

4. Zhang, A., Deng, C., Oguz, B., Ott, M., & Çelikyilmaz, A. (2024). *RASD: Retrieval-Augmented Speculative Decoding*. arXiv preprint. https://arxiv.org/abs/2503.03434

### Suffix Tree Methods

5. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). *SuffixDecoding: A Model-Free Approach to Speeding Up Large Language Model Inference*. NeurIPS 2025 Spotlight. https://suffix-decoding.github.io/

6. Cai, T. (2024). *Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads*. GitHub Repository. https://github.com/FasterDecoding/Medusa

### Implementation Resources

7. HuggingFace. (2024). *Generation Strategies: Speculative Decoding*. HuggingFace Transformers Documentation. https://huggingface.co/docs/transformers/generation_strategies

8. vLLM Team. (2024). *N-gram Prompt Lookup in vLLM*. vLLM Documentation. https://docs.vllm.ai/en/latest/features/spec_decode.html

9. Gerganov, G., et al. (2024). *llama-lookup: Prompt Lookup Decoding in llama.cpp*. GitHub. https://github.com/ggml-org/llama.cpp/tree/master/examples/lookup

</details>

---

*Previous: [Chapter 06: MoE Optimization](06-moe-optimization.md)* | *Next: [Chapter 08: RadixAttention](08-radix-attention.md)*
