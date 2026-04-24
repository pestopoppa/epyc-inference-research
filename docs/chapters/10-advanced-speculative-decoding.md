# Chapter 10: Advanced Speculative Decoding Techniques

## Motivation

Our EPYC 9655 system is **memory-bandwidth-bound** during autoregressive decoding: each token requires reading the entire model from DRAM. With 8-channel DDR5 (~300+ GB/s aggregate bandwidth), we have substantial bandwidth but still hit this wall with large quantized models. This memory-bound nature has two implications:

1. **CPU cores are underutilized** — arithmetic units idle while waiting for memory reads
2. **Parallel model execution is feasible** — running multiple small models costs mainly bandwidth, and we may have compute headroom to spare

This chapter surveys advanced speculative decoding techniques from recent literature (2024–2026) that exploit these properties. Unlike Chapter 01's basic two-model speculation and Chapter 03's prompt-lookup, these techniques use tree structures, hierarchical verification, self-speculation, heterogeneous scheduling, and meta-speculation to push acceptance rates and throughput further.

---

## 1. Tree Speculative Decoding

### 1.1 SpecInfer (ASPLOS '24)

**Paper**: Miao et al., "SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification" [[arXiv:2305.09781](https://arxiv.org/abs/2305.09781)]

SpecInfer replaces single-sequence draft-then-verify with a **token tree**. Multiple candidate continuations are organized as a tree where each root-to-leaf path is a complete candidate sequence. The target model verifies all paths in a single forward pass via a **topology-aware causal mask**.

**Tree construction**: Two approaches:
- *Expansion-based*: Preset config vector `<k₁, k₂, ..., kₘ>` where kᵢ is top-k branching at level i. Using top-5 tokens raises per-token acceptance from 70% to 89% (greedy), 57% to 97% (stochastic).
- *Multi-SSM boosting*: Multiple small draft models trained via unsupervised boosting — each fine-tuned on prompts where prior SSMs failed. Outputs merge into a single tree.

**Verification**: Topology-aware causal mask batches all tree attention into a single kernel. DFS-ordered KV cache avoids per-sequence duplication.

**Results**: 1.5–3.5x speedup on A10 GPUs. Offloading scenarios (model on CPU, draft on GPU) yield 2.6–3.5x.

**Limitations**: Static tree topology. GPU-only (FlexFlow + CUDA). Degrades at high batch sizes.

### 1.2 Sequoia (ICML '24)

**Paper**: Chen et al., "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding" [[arXiv:2402.12374](https://arxiv.org/abs/2402.12374)]
**Code**: [github.com/Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia)

Sequoia solves three problems SpecInfer left open: optimal tree topology, temperature robustness, and hardware-aware tree sizing.

**DP-optimized topology**: Tree construction formulated as dynamic programming — maximize expected accepted tokens `F(T) = Σ f(v)` where f(v) is the probability of reaching node v. Budget allocates more descendants to high-acceptance early layers. Theoretical scaling: Ω(b · log(n) / log(log(n))) expected tokens with tree size n.

**Sampling without replacement**: After rejection, the draft token's probability is zeroed — preventing "same mistake twice." Achieves optimal transport acceptance rate `1 - ||P-Q||₁/2`. Robust across all temperatures, unlike SpecInfer's top-k.

**Hardware-aware optimizer**: Grid search over (tree_size, depth) pairs using empirically measured `t(n)` verification latency per hardware platform.

**Results**: 4.04x on Llama2-7B (A100), 9.96x on Llama2-70B with offloading (L40). Consistently 33%+ more tokens/step than SpecInfer at equivalent tree sizes.

**Limitations**: GPU-only (CUDA 12.1). LLaMA family only. No quantization support.

### 1.3 OPT-Tree (TACL 2025)

**Paper**: Wang et al., "OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure" [[arXiv:2406.17276](https://arxiv.org/abs/2406.17276)]
**Code**: [github.com/Jikai0Wang/OPT-Tree](https://github.com/Jikai0Wang/OPT-Tree)

OPT-Tree constructs a **fully dynamic tree per decoding step** by maximizing E(A) = Σ p̂ⱼⁱ (sum of cumulative path probabilities). The tree shape changes at every step based on draft model's current probability distributions.

**Algorithm**: Greedy top-k expansion at each layer, then select N nodes with largest cumulative path probabilities. Early stopping when layer increment < δ (threshold must satisfy μ < δ < 1 where μ is draft/target latency ratio).

**Results**: Up to 3.05x (Llama2-70B + EAGLE). With 500+ node budgets, >10 tokens accepted per step.

**Key finding**: Dynamic trees consistently outperform static/offline-calibrated trees (including Sequoia) at all tree sizes, with the gap widening above ~150 nodes.

### 1.4 DySpec (WWW 2025)

**Paper**: Xiong et al., "DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure" [[arXiv:2410.11744](https://arxiv.org/abs/2410.11744)]

DySpec uses **heap-based greedy expansion** with draft probabilities as acceptance proxy. Proves optimality under KL-divergence constraint between draft/target distributions.

**Algorithm**: Max-heap of expandable nodes keyed by estimated acceptance probability. Pop → sample → push children. Threshold variant for large budgets: include all nodes with estimated rate > 1/n, reducing draft calls from O(N) to O(depth).

**Results**: **9.1x** on Llama2-70B (A100 + CPU offloading, greedy), 6.21x at temp=0.6. Up to 16.04 tokens accepted per step with 768-node trees. Tree construction overhead <2% (C++ implementation).

**Key finding**: No offline calibration needed (unlike Sequoia). Provably optimal greedy strategy. The 9.1x number comes from CPU-offloading where target inference is ~5s/step, making each accepted token extremely valuable — directly relevant to our CPU scenario.

### 1.5 SpecExec — Massively Parallel Verification (Jun 2024)

**Paper**: "SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices" [[arXiv:2406.02532](https://arxiv.org/abs/2406.02532)]

SpecExec contains what may be the single most important observation for CPU inference:

> When inference is bandwidth-bound, **hundreds of tokens can be processed in the same time as one**.

The target model's cost is dominated by **weight loading**, not by how many tokens are being verified. So instead of verifying one candidate sequence, SpecExec verifies **up to 2048 tokens in a single pass** — and the marginal cost per additional token is near-zero.

**Algorithm**: Deterministic tree construction via modified Dijkstra/SSSP over draft probability landscape. Selects K tokens with highest cumulative probability up to depth D. The entire tree is verified in one batched forward pass.

**Phase separation**: The execution is strictly sequential — draft model builds the tree, then stops, then target model verifies. **No concurrent execution required.** This demolishes the common objection that speculative decoding requires running two models simultaneously.

```
Phase 1: Draft model builds speculation tree (fast, small model)
Phase 2: Target model verifies entire tree in one pass (expensive, but amortized)
```

**Results**:
- Llama2-70B (4-bit) on A100 with offloading: 6.02 tok/s (8.9x speedup)
- 16-bit on RTX 3090 with offloading: 3.12 tok/s (18.7x speedup)
- Generates **10–20 accepted tokens per target model iteration** at large budgets
- RTX 2080Ti (1.3B draft → 70B target): 1.86 tok/s (6.1x)

**Why this changes the CPU calculus**: The original "CPU Relevance of Tree Methods" table below assumed tree verification is expensive on CPU. SpecExec shows this assumption is wrong when bandwidth-bound. On our EPYC with ~300 GB/s reading a Q4 32B model (~18GB), one forward pass takes ~60ms regardless of whether we verify 1 token or 1000 tokens — the weight-loading dominates. This means:

- **Optimal tree size on CPU is NOT 8–32 nodes** — it could be **hundreds to thousands**
- The draft model (e.g., Qwen 2B at ~1.5GB) costs only ~5ms per forward pass
- A draft tree of 1024 tokens costs maybe 50–100ms to build but saves dozens of 60ms verification passes

### 1.6 Dynamic Delayed Tree Expansion (Feb 2026)

**Paper**: "Dynamic Delayed Tree Expansion" [[arXiv:2602.16994](https://arxiv.org/abs/2602.16994)]

Standard tree methods branch immediately from the root. Delayed expansion instead drafts a **single path for L₁ tokens** before branching into K paths of length L₂. The insight: near the root, draft and target distributions align well (high acceptance), so branching wastes budget. Deeper in the tree, where distributions diverge, multiple paths provide the most benefit.

**Neural selector**: Lightweight MLP predicts optimal (K, L₁, L₂) per decoding step from draft/target hidden states, entropy, and KL divergence. Chooses from 225 configurations.

**Results**: ~5% higher throughput vs best existing methods. Block efficiency improvements of 10–25%.

**CPU relevance**: Delayed branching produces smaller trees for the same effective acceptance rate. The draft phase generates fewer tokens (single path to L₁, then K paths for L₂), reducing total draft model weight loads.

### CPU Relevance of Tree Methods (Revised)

The SpecExec observation fundamentally changes the analysis. On bandwidth-bound CPU systems:

| Factor | GPU Context | CPU Context (Revised) |
|--------|------------|------------------------|
| Tree verification cost | Cheap (batched) | **Also cheap** (bandwidth-bound, not token-count-bound) |
| Optimal tree size | 64–768 nodes | **Hundreds to thousands** (weight-load amortization) |
| Optimal depth | 7–22 | **Similar or deeper** (each accepted token saves a full weight read) |
| Draft model cost | Negligible | Moderate but amortizable (small model, fast per-token) |
| Key constraint | Compute budget | **Memory bandwidth** (already saturated by weight reads) |

**Actionable**: The DP topology optimization and dynamic tree construction algorithms are portable to llama.cpp. DySpec's <2% overhead C++ tree builder is closest to integration-ready. The key insight from SpecExec is that on CPU, **larger trees are cheaper than assumed** because verification cost is dominated by weight loading, not token count. llama.cpp's existing `--draft` uses single-sequence speculation; tree verification requires implementing topology-aware causal masks in ggml.

---

## 2. Hierarchical Speculative Decoding

### 2.1 HiSpec — Early-Exit Hierarchy (Oct 2025)

**Paper**: Kumar et al., "HiSpec: Hierarchical Speculative Decoding for LLMs" [[arXiv:2510.01336](https://arxiv.org/abs/2510.01336)]

HiSpec addresses **verification latency** (not draft speed) by using early-exit layers within a single model as intermediate verifiers.

**Three-level hierarchy** within one model:
- **Draft layer (L_d)**: ~1/8 of model depth (e.g., layer 3 of 32)
- **Intermediate verifier (L_i)**: ~1/4 of model depth (e.g., layer 8 of 32)
- **Full model (L_f)**: All layers

**Key insight**: Early layers (up to 1/4 depth) correctly generate up to 69% of response tokens, making intermediate verification a highly effective cheap filter.

**Mechanism**: Draft → intermediate verify → buffer tentatively accepted tokens (N_i=4) → full verify only for tokens passing intermediate check. KV caches shared across all three stages.

**Results**: Up to 2.08x (Llama3-8B on CNN/DM). Average 1.7x across models. Token acceptance rate improves from 39.7% to 58.1% (+46%). Outperforms LayerSkip (1.14x), Lookahead (1.20x), AdaDecode (1.00x).

**Hardware**: 4x H100 with AMD EPYC 9354 CPU.

**Limitations**: Requires early-exit capable models (trained with EE heads). GPU-only evaluation.

### 2.2 HSD — Lossless Hierarchical Verification (Jan 2026)

**Paper**: Zhou et al., "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding" [[arXiv:2601.05724](https://arxiv.org/abs/2601.05724)]
**Code**: [github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding)

HSD addresses the **verification algorithm itself**. Standard token-wise verification checks each draft token independently. Sequence-level (blockwise) verification considers joint probabilities but faces exponential complexity. HSD decomposes the joint distribution hierarchically across prefix-conditioned "branches."

**Mathematical framework**: Backward scan for longest accepted prefix → branch divergence quantifies local probability deficit → capped branch resampling enables one-shot resampling. Provably lossless (output identical to target model).

**Results**: 3–7% block efficiency improvement on standard pairs. **+12.4% decoding speed when combined with EAGLE-3** (71.59 → 80.49 tok/s). Multi-draft (11 candidates): +5.9% further.

**SpecBundle models** (HuggingFace: [lmsys/specbundle](https://huggingface.co/collections/lmsys/specbundle)): 17 production EAGLE-3 draft models (0.2B–1B) for targets from 8B to 480B. Designed for SGLang integration.

### CPU Relevance

**HiSpec** is the most promising hierarchical technique for CPU. Exiting at 1/8 depth during drafting reduces compute proportionally — on CPU where every FLOP matters (no GPU parallelism to hide latency), filtering 69% of tokens at 1/4 depth before running the full model is extremely valuable. The KV cache reuse maps well to CPU cache locality. **Update (April 2026)**: The `n_layer_exit` API now provides per-decode adaptive layer exit in our llama.cpp fork (see Section 13.5), lifting the infrastructure blocker. Remaining challenge: training per-layer confidence routers from TIDE calibration data.

**HSD's** verification algorithm improvement is "free" — verification overhead is <1% of total time. Implementing capped branch resampling in llama.cpp's verification path would yield 3–7% speedup on top of existing `--draft` with no memory cost. Combined with EAGLE-3 draft models from SpecBundle, the 12.4% speedup is meaningful.

---

## 3. Self-Speculative Decoding

### 3.1 SparseSpec — Sparse Attention Self-Speculation (Dec 2025)

**Paper**: Zhao et al., "Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding" [[arXiv:2512.01278](https://arxiv.org/abs/2512.01278)]

SparseSpec uses the **same model as both draft and target** — no separate draft model needed. The draft phase runs all layers but attends to only **5% of the KV-cache** ("pillar" tokens with highest attention mass). The key insight: reasoning models with long outputs (10K–13K tokens) shift from compute-bound to memory-bandwidth-bound during decode, making sparse attention highly effective.

**PillarAttn mechanism**: Pillar token identification is zero-overhead — reuses attention scores from the full-attention verification step. Top-K selection determines which KV-cache entries to load. Sparsity pattern is dynamic (re-identified every 8 steps as critical tokens shift during long generation).

**Results**: 2.13x vs vLLM baseline (Qwen3-8B on AIME). 77% per-token acceptance (6.16/8 tokens). Attention latency reduced 3.29x.

**Hardware**: DGX-H100-SXM5. GPU batched serving only.

### 3.2 LayerSkip (Meta, ACL 2024) — The Foundational Self-Spec Paper

**Paper**: Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" [[arXiv:2404.16710](https://arxiv.org/abs/2404.16710)]

Draft generated by exiting at an earlier layer (e.g., layer 16 of 32), using the shared LM head. KV-cache and activations shared between draft and verify — no redundant computation.

**Results**: Up to 2.16x (CNN/DM summarization). **Requires training modification** (graduated layer dropout).

### 3.3 SWIFT (ICLR 2025) — Training-Free Self-Speculation

**Paper**: "SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration" [[arXiv:2410.06916](https://arxiv.org/abs/2410.06916)]

**Plug-and-play**: No training required. Adaptively selects which layers to skip at inference time based on input context. Works on any off-the-shelf model.

**Results**: 1.3–1.6x speedup. Lower than LayerSkip but requires zero training.

### 3.4 EAGLE Family (ICML '24 / EMNLP '24 / NeurIPS '25)

Lightweight autoregressive prediction head attached to target model's hidden states. Not strictly self-speculative but represents SOTA in draft-head approaches.

| Version | Speedup | Acceptance Length |
|---------|---------|-------------------|
| EAGLE-1 | ~2.5–3x | ~3.5 tokens/cycle |
| EAGLE-2 | ~3–5x | ~4.0 tokens/cycle |
| EAGLE-3 | **3.0–6.5x** | ~4.5–5.0 tokens/cycle |

### CPU Relevance

| Method | Training Required | Memory Overhead | CPU Applicability |
|--------|-------------------|-----------------|-------------------|
| SparseSpec | No | ~0.5% | Low — GPU batched serving design |
| LayerSkip | Yes (layer dropout) | None (shared) | Medium — if models trained with recipe |
| SWIFT | No (plug-and-play) | None | **High** — works on any GGUF model |
| EAGLE-3 | Yes (head training) | Small (head params) | Medium — needs GGUF head support |

**SWIFT** is the most directly applicable: no training, works on any model, layer-skipping maps to reduced compute per draft token. Active llama.cpp discussion exists ([#10787](https://github.com/ggml-org/llama.cpp/discussions/10787)).

**Complementarity with lookup speculation**: Our existing `spec_lookup` configs have zero draft cost but lower acceptance on complex reasoning. Self-speculative approaches have non-zero draft cost but higher acceptance on unpredictable content. A hybrid could use lookup when n-gram patterns are strong, falling back to self-speculation when they aren't.

---

## 4. Heterogeneous Processor Partitioning

### 4.1 Ghidorah (Jun 2025)

**Paper**: Wei et al., "Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism" [[arXiv:2505.23219](https://arxiv.org/abs/2505.23219)]

Ghidorah partitions speculative decoding across CPU and GPU on unified-memory SoCs (Jetson NX).

**Hetero-Core Model Parallelism (HCMP)**:
- **Linear layers**: Column-split across processors. Both receive identical input activations, write to non-overlapping output regions. **No all-reduce synchronization needed.**
- **Attention**: Dense component (KV cache for existing context) → GPU. Sparse component (new draft tokens) → CPU. Online softmax for independent partial computation.

**ARCA scheduler**: Offline profiling finds optimal verification width. Key finding: **W=16 beats W=64** — wider trees increase compute faster than acceptance grows.

**Results**: Up to 7.6x total (3.27x from speculation × 2.31x from CPU+GPU parallelism). ARM sparse attention optimization: 3.49x over naive.

**ARM sparse attention kernels**: Row-wise data access, NEON 128-bit vector FMA, register-resident accumulation. Dense → sparse improvement: 1.90x.

### CPU Relevance

**Directly transferable insights**:

1. **Sparse attention optimization**: The ARM NEON techniques translate to x86 AVX-512 on EPYC. As KV cache grows, attention matrices become increasingly sparse for draft tokens — exploiting this yields 1.9–3.5x attention speedups.

2. **Verification width sweet spot**: Even on GPU, W=16 beats W=64. On CPU-only systems, optimal width is likely W=4 to W=8.

3. **NUMA-aware weight partitioning**: Dual-socket EPYC with Infinity Fabric is analogous to unified memory on SoCs. Column-split linear layers across NUMA nodes with no-allreduce output regions could replicate Ghidorah's parallelism benefit without a GPU.

4. **Memory bandwidth is the bottleneck, confirmed**: Single-sample decoding is memory-bandwidth-bound. Speculation increases arithmetic intensity — the right strategy for our bandwidth-constrained CPU.

---

## 5. Parallel and Multi-Draft Speculation

### 5.1 ParallelSpec — Single-Pass Parallel Drafting (Oct 2024)

**Paper**: "ParallelSpec: Parallel Drafter for Efficient Speculative Decoding" [[arXiv:2410.05589](https://arxiv.org/abs/2410.05589)]

ParallelSpec replaces the autoregressive draft model with a **parallel drafter** that generates k tokens in a single forward pass. Architecture: a single Transformer layer (202M params for 7B target) with k trainable `[MASK]` tokens. The mask positions predict k future tokens simultaneously.

**Key benefit for CPU**: If the draft model costs T_draft per autoregressive step and you need k=4 tokens, traditional approaches pay 4 × T_draft. ParallelSpec pays ~1 × T_draft. On CPU where T_draft might be 50–200ms, this saves 150–600ms per draft round. Weight loading drops from k× to 1×.

**Results**: Up to 2.84x (Llama2-13B). 62.7% improvement over baseline Medusa. Average acceptance length 2.39 → 3.31 tokens. Requires training the parallel drafter head.

### 5.2 PEARL — Concurrent Draft + Verify (ICLR 2025)

**Paper**: "PEARL: Parallel Speculative Decoding with Adaptive Draft Length" [[arXiv:2408.11850](https://arxiv.org/abs/2408.11850)]

PEARL eliminates the mutual waiting problem between draft and target models:
- **Pre-verify**: Run target model verification concurrently with draft generation. First draft token verified "for free."
- **Post-verify**: When all drafts accepted, draft model continues generating during verification rather than waiting idle.
- **Adaptive draft length**: γ = round(T_target / T_draft), automatically matching hardware speed differential.

**Results**: Up to 4.43x (CodeLlama 7B → 70B). Mean accepted tokens: up to 39.9 per step (vs 5.69 vanilla SD).

**CPU relevance**: On EPYC with many cores, dedicate core subsets to draft and target models running truly in parallel across NUMA domains. The speed ratio c between target and draft on CPU would be large (10–50x), meaning γ would be high (10–50 draft tokens), and post-verify would generate many additional tokens during each long verification pass. **Critical caveat**: concurrent execution doubles peak bandwidth demand. On dual-socket EPYC, pin draft model to socket 0 and target to socket 1 for independent bandwidth pools.

### 5.3 SpecHub — Multi-Draft Verification (EMNLP 2024)

**Paper**: "SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding" [[arXiv:2411.05289](https://arxiv.org/abs/2411.05289)]

When K draft models (or K samples from one draft model) each propose a token, SpecHub optimizes the acceptance decision. Uses a hub-and-spoke sparsification of the optimal transport joint distribution, reducing the LP to O(|V|) variables solvable in linear time.

**Key finding**: Second-draft acceptance rate improves 63% over baseline (0.1021 → 0.1660). But k>2 drafts show diminishing returns (curse of dimensionality in OT).

**CPU relevance**: With 96 cores, running K=2 independent draft samples in parallel threads is trivial. Each draft explores a different branch of the token distribution. SpecHub's linear-time verification adds negligible overhead. The practical sweet spot is K=2.

### 5.4 The Multi-Worker Architecture

Your notes correctly identify that our EPYC system is unusually well-suited for **parallel draft workers**:

```
Draft worker 1 (cores 0-15)  ─┐
Draft worker 2 (cores 16-31) ─┤── merge into speculation tree
Draft worker 3 (cores 32-47) ─┤
Draft worker 4 (cores 48-63) ─┘
                                    ↓
                        Target model verification (all cores)
                                    ↓
                        Accept longest matching path
```

This is mathematically equivalent to SpecInfer's multi-SSM boosting but exploiting CPU core parallelism instead of multi-GPU parallelism. Each worker explores different token paths. The target model then verifies a large combined tree. Papers implementing this concept:
- SpecInfer: Multi-SSM boosting (different draft models per GPU)
- SpecHub: K=2 drafts from same model with optimal acceptance
- ParallelSpec: Single-pass k-token parallel drafting

---

## 6. KV-Cache Tree Architecture

### 6.1 The Shared-Prefix Insight

All tree-based methods share a critical implementation detail that dramatically reduces memory bandwidth: **branches in the speculation tree share KV-cache prefix states**.

Without sharing:
```
branch 1: full KV copy (prefix + A + B + C)
branch 2: full KV copy (prefix + A + B + D)
branch 3: full KV copy (prefix + A + E)
```

With KV-cache tree:
```
KV(prefix)          ← computed once, shared
  └─ KV(A)          ← shared by branches 1, 2, 3
      ├─ KV(B)      ← shared by branches 1, 2
      │   ├─ KV(C)  ← branch 1 only
      │   └─ KV(D)  ← branch 2 only
      └─ KV(E)      ← branch 3 only
```

The verification pass processes nodes in **topological order**, computing attention for each node once and reusing it for all descendants. Compute complexity becomes O(nodes in tree) instead of O(branches × sequence_length).

This is explicitly implemented in:
- **SpecInfer**: DFS-ordered KV cache with topology-aware causal mask
- **Sequoia**: Tree attention with shared prefix computation
- **SpecExec**: "Cache tree" structure with explicit prefix reuse

### 6.2 Why This Matters More on CPU Than GPU

On GPU, compute is relatively cheap so KV duplication across branches is tolerable. On CPU, **every byte read from memory costs real time**. For a 32-layer model with hidden_dim=4096 and a 1000-token context:

```
Full KV per branch:  32 layers × 2 (K+V) × 1000 tokens × 4096 dim × 2 bytes = ~500 MB
Delta KV per branch: 32 layers × 2 (K+V) × 5 new tokens × 4096 dim × 2 bytes = ~2.5 MB
```

With 10 branches, the difference is **5GB vs 25MB** of KV cache reads per verification step. On CPU where memory bandwidth is the bottleneck, this is the difference between feasible and infeasible tree speculation.

### 6.3 NUMA-Aware KV Sharding

The KV-cache tree architecture maps naturally to NUMA topology on EPYC:

```
NUMA Node 0                    NUMA Node 1
├─ KV shard (layers 0-7)      ├─ KV shard (layers 8-15)
├─ Draft workers (branch A)   ├─ Draft workers (branch B)
└─ Local L3 cache             └─ Local L3 cache

NUMA Node 2                    NUMA Node 3
├─ KV shard (layers 16-23)    ├─ KV shard (layers 24-31)
├─ Draft workers (branch C)   ├─ Draft workers (branch D)
└─ Local L3 cache             └─ Local L3 cache
```

**Execution pipeline**:
1. **Broadcast prefix KV**: Replicate shared prefix KV state to all NUMA nodes (one-time cost per prefix update)
2. **Parallel draft expansion**: Each NUMA node builds its own subtree using local memory (no cross-socket traffic)
3. **Assemble verification batch**: Gather candidate sequences into combined tree
4. **Verify**: Target model evaluates tree. Layer-sharded KV means each NUMA node handles its layer range from local memory

**Why this avoids the NUMA penalty**: The naive approach — single global KV cache — forces threads on NUMA nodes 1–3 to read across Infinity Fabric from node 0, at 1.5–2x latency penalty. With sharding, each node reads its own KV shard locally. The only cross-node traffic is activations between layers (small: hidden_dim × batch × 2 bytes ≈ 8KB per token per layer boundary), not the entire KV cache.

**Alternative: branch-per-NUMA**: Instead of layer-sharding the KV cache, assign entire speculation branches to NUMA nodes:
```
NUMA 0 → branch A (full KV for branch A, all layers)
NUMA 1 → branch B (full KV for branch B, all layers)
NUMA 2 → branch C
NUMA 3 → branch D
```
Each node builds its subtree entirely from local memory. Prefix KV is replicated (affordable with 1.5TB RAM). This trades memory for locality and avoids all cross-node traffic during drafting.

---

## 7. Speculative Speculative Decoding (SSD)

### 5.1 Saguaro (ICLR 2026)

**Paper**: Kumar, Dao & May, "Speculative Speculative Decoding" [[arXiv:2603.03251](https://arxiv.org/abs/2603.03251)]
**Code**: [github.com/tanishqkumar/ssd](https://github.com/tanishqkumar/ssd)

SSD adds a **meta-speculation layer**: while the target model verifies current draft tokens, the draft model **predicts what the verification outcome will be** and pre-computes the next speculation for those predicted outcomes. On a cache hit (correct prediction), the next speculation returns with zero drafting latency.

**Architecture**: Two asynchronous processes on **separate hardware**:
- Verifier (target model, e.g., 4x H100): Standard SD verification
- Speculator (draft model, e.g., 1x H100): Predicts outcomes, populates speculation cache

**Saguaro algorithm** addresses three challenges:
1. **Outcome prediction**: Draft model's own logits guess top-F_k bonus tokens at each position. Up to 90% prediction accuracy. Geometric fan-out allocation optimally distributes cache budget.
2. **Acceptance/prediction tradeoff**: Novel sampling scheme downweights cached tokens in draft distribution, concentrating residual distribution mass on cached entries. Cache hit rate monotonically increases.
3. **Cache miss fallback**: Neural speculator at low batch sizes, fast (random) speculator at high batch sizes. Threshold strategy proven optimal.

**Results**: Up to **2x faster than optimized SD baselines** (SGLang, vLLM). Up to **5x faster than autoregressive**. Lossless (identical output distribution).

**Limitations**: Requires separate hardware for draft model. Not beneficial for throughput-focused workloads. Early implementation (March 2026, single contributor).

### CPU Relevance

SSD's core insight — predicting verification outcomes to overlap drafting with verification — maps to **multi-threaded CPU inference**. On EPYC with 128+ threads:
- Target model verification on one set of cores
- Draft model + outcome prediction on another set of cores
- Shared memory (no PCIe transfer) makes the cache mechanism essentially free

The speculation cache itself stores token sequences (not weights or KV caches), so memory overhead is negligible. This is one of the few techniques where CPU's abundant thread count and shared memory is a clear advantage over GPU's separate-device constraint.

---

## 8. Comparative Analysis

### Technique Comparison Matrix

| Technique | Speedup (Reported) | Draft Overhead | Memory Overhead | Training Required | CPU Feasibility |
|-----------|-------------------|----------------|-----------------|-------------------|-----------------|
| SpecInfer | 1.5–3.5x | Separate model | SSM params (<1%) | SSM training | Medium |
| Sequoia | 4.04–9.96x | Separate model | Tree KV cache | No | **High** (algorithm portable) |
| OPT-Tree | Up to 3.05x | Separate model | Dynamic tree | No | Medium (algorithm portable) |
| DySpec | Up to 9.1x | Separate model | Dynamic tree | No | **High** (C++ builder) |
| **SpecExec** | **Up to 18.7x** | Separate model | Cache tree | No | **Highest** (designed for bandwidth-bound) |
| Delayed Tree | +5–25% block eff. | Separate model | Smaller tree | MLP selector | Medium-High |
| HiSpec | Up to 2.08x | None (early exit) | None (shared) | EE heads | Medium |
| HSD | +3–12.4% | Separate model | Negligible | No | **High** (verification algo) |
| SparseSpec | Up to 2.13x | None (sparse attn) | ~0.5% | No | Low (GPU batched) |
| LayerSkip | Up to 2.16x | None (layer skip) | None (shared) | Yes (layer dropout) | Medium |
| SWIFT | 1.3–1.6x | None (layer skip) | None | No | **High** (plug-and-play) |
| **ParallelSpec** | Up to 2.84x | Parallel drafter | Single layer | Yes (drafter) | **High** (reduces draft BW k→1) |
| **PEARL** | Up to 4.43x | Separate model | None | No | **Highest** (concurrent + adaptive) |
| **SpecHub** | +63% 2nd-draft acc. | Multi-draft | Negligible | No | **High** (K=2 parallel drafts) |
| Ghidorah | Up to 7.6x | Medusa heads | Head params | Yes (Medusa) | **High** (sparse attn) |
| SSD/Saguaro | Up to 5x | Separate + cache | Cache (negligible) | No | **High** (multi-thread) |

### Prioritized Research Directions for EPYC

**Tier 0 — Paradigm shift** (changes how we think about CPU speculation):

The SpecExec/Sequoia insight that **verification of N tokens costs the same as 1 token when bandwidth-bound** means our EPYC system should target **very large speculation trees** (hundreds to thousands of nodes), not the small trees (8–32) we initially assumed. This reframes every other technique.

**Tier 1 — Implement now** (low risk, clear benefit):
1. **SpecExec-style large tree verification**: Profile actual verification cost vs tree size on EPYC. Confirm that verifying 100–1000 tokens costs ≈ the same as 1. If confirmed, increase `--draft` depth dramatically.
2. **HSD verification algorithm**: Replace token-wise verification in llama.cpp's `--draft` with capped branch resampling. Free 3–7% speedup, no memory cost, no training needed.
3. **Phase-separated scheduling**: Ensure draft model runs to completion before target model starts (no concurrency needed). This simplifies implementation enormously.

**Tier 2 — Prototype and measure** (moderate effort, high potential):
4. **DySpec dynamic tree construction**: Port the C++ heap-based tree builder to llama.cpp. Needs topology-aware causal mask in ggml. Would replace fixed-width speculation with adaptive trees tuned per-step.
5. **PEARL concurrent draft+verify**: Pin draft model to NUMA node 0, target to NUMA node 1. Overlap draft generation with verification. Adaptive γ based on measured speed ratio.
6. **Multi-worker parallel drafting**: Run 2–4 draft workers on separate core groups, each exploring different token paths. Merge into combined speculation tree. Verify in one pass.
7. **KV-cache tree with shared prefixes**: Implement branch-aware KV cache in llama.cpp to avoid duplicating prefix KV across speculation branches. Critical for large trees.
8. **SWIFT self-speculation**: Layer-skipping for draft tokens without training. Infrastructure now available via `n_layer_exit` API (Section 13.5). Active community discussion ([#10787](https://github.com/ggml-org/llama.cpp/discussions/10787)).

**Tier 3 — Long-term research** (high effort, transformative potential):
9. **NUMA-aware KV sharding**: Branch-per-NUMA speculation — each NUMA node builds and verifies a subtree from local memory. Prefix KV replicated (affordable with 1.5TB RAM). Zero cross-node traffic during drafting.
10. **SSD/Saguaro meta-speculation**: Predict verification outcomes to pre-compute next draft. Speculation cache in shared memory across EPYC thread groups.
11. **Sparse attention kernels**: AVX-512 sparse Q×K^T and A×V for speculative verification. 1.9–3.5x attention speedup on growing contexts.
12. **Hybrid lookup + self-spec + tree**: Use prompt-lookup when n-gram patterns are strong, self-speculation when they aren't, tree expansion when uncertainty is high. Adaptive switching based on recent acceptance rates and entropy.

---

## 9. The Memory-Bound Thesis

All surveyed papers converge on a common finding: **single-sample LLM decoding is memory-bandwidth-bound**. This has specific implications for our EPYC setup — and the SpecExec paper crystallizes the key insight.

### What Being Memory-Bound Means

During autoregressive decoding, each token generation requires reading the entire model weights from DRAM. For a Q4-quantized 32B model (~18GB), generating one token reads ~18GB from memory. With 8-channel DDR5 at ~300 GB/s, that's ~60ms per token — and the arithmetic to process those weights takes far less time than reading them.

### The SpecExec Insight: Verification is Nearly Free

SpecExec's central observation changes the entire calculus:

```
Without speculation:  target forward pass → 1 token
With tree speculation: target forward pass → 3–20 tokens accepted
```

Because the forward pass cost is **weight-loading time** (not compute time), verifying a tree of N candidate tokens costs approximately the same as verifying 1 token. The model weights must be read from DRAM regardless — the additional computation of checking N tokens against those weights adds negligible time on bandwidth-bound hardware.

**Concrete numbers for our system** (Q4 32B model, ~18GB weights, ~300 GB/s bandwidth):
- Baseline: 1 forward pass (~60ms) → 1 token → **~17 tok/s**
- With 1024-token speculation tree accepting ~10 tokens: 1 pass (~62ms) + draft time → **~100+ effective tok/s**
- Draft cost (Qwen 2B, ~1.5GB): ~5ms per forward pass, trivial vs target

### The Phase Separation Principle

A common misconception is that speculative decoding requires concurrent model execution. The surveyed papers show this is **not required**:

```
Phase 1: Draft model runs ALONE → builds speculation tree → stops
Phase 2: Target model runs ALONE → verifies entire tree → accepts tokens
```

The models **never need to execute simultaneously**. This simplifies implementation enormously for CPU systems where thread contention and bandwidth sharing between concurrent models would be problematic.

However, PEARL shows that concurrent execution (when feasible via NUMA separation) provides an additional speedup by eliminating idle time between phases.

### Why This Favors Speculation on CPU

Speculative decoding increases **arithmetic intensity** (compute per byte read). Verifying K draft tokens in one pass reads the model weights once but performs K× the computation. On GPU, this benefit is bounded because GPUs are already compute-efficient. On CPU, where compute units sit idle during memory reads, there's significant headroom to do more work per memory read — speculation fills that gap.

### Verification as Batch Inference

Tree speculation converts verification into a **batch inference problem**:

```
N candidate continuations (from speculation tree)
            ↓
1 batched forward pass (single weight-loading event)
            ↓
Choose accepted prefix (longest matching path)
```

This is identical to standard prefill-phase batching, which transformers already handle efficiently. The target model is most efficient when evaluating many sequences simultaneously — tree speculation exploits exactly this.

### Why Parallel Models Are Feasible

If we're bandwidth-bound, running a second small model (draft) consumes mostly bandwidth, not compute. But the compute cores *are available*. With careful thread pinning and NUMA-aware placement:
- Target model on cores 0–63 (socket 0)
- Draft model on cores 64–127 (socket 1)
- Each socket has independent memory channels

This is essentially Ghidorah's HCMP on a CPU-only system, with NUMA replacing unified GPU/CPU memory.

### Quantitative Framework (Revised)

The traditional speculation speedup formula:

```
Speedup ≈ E[accepted_tokens] / (1 + μ × depth)
```

On GPU: μ is tiny (draft model is fast) → deep trees with many nodes are optimal.
On CPU: μ is larger (draft model also bandwidth-bound) → shallower trees, fewer nodes.

**But** the SpecExec insight adds a correction: on bandwidth-bound hardware, the **verification cost scales sub-linearly with tree size**. The revised framework:

```
Speedup ≈ E[accepted_tokens] / (T_draft(tree_size) / T_target + 1)
```

Where T_target is approximately constant (weight-loading time) regardless of how many tokens are being verified. This means the denominator barely grows with tree size, and the numerator (accepted tokens) grows logarithmically with tree size (Sequoia's Ω(b·log(n)/log(log(n))) bound).

For our system: T_target ≈ 60ms (18GB at 300 GB/s), T_draft ≈ 5ms per step (1.5GB at 300 GB/s). Building a depth-8 tree costs ~40ms of drafting. Total cycle: ~100ms. If 8 tokens accepted: 80 tok/s effective. If 15 tokens accepted: 150 tok/s effective. Compared to baseline ~17 tok/s, this is a **5–9x speedup** — and we haven't yet applied NUMA parallelism, concurrent execution, or multi-worker drafting.

### The Combined Architecture Vision

Combining the insights from all surveyed papers into a single architecture for our EPYC system:

```
┌──────────────── EPYC 9655 (96 cores, 8-ch DDR5, 1.5TB RAM) ────────────────┐
│                                                                              │
│  ┌─── NUMA 0 ──────────┐  ┌─── NUMA 1 ──────────┐                          │
│  │ Draft Worker A       │  │ Draft Worker B       │                          │
│  │ (Qwen 2B, cores 0-7)│  │ (Qwen 2B, cores 8-15)│                         │
│  │ KV shard (local)     │  │ KV shard (local)     │                          │
│  │ Builds subtree A     │  │ Builds subtree B     │                          │
│  └──────────────────────┘  └──────────────────────┘                          │
│                                                                              │
│  ┌─── NUMA 2 ──────────┐  ┌─── NUMA 3 ──────────┐                          │
│  │ Draft Worker C       │  │ Draft Worker D       │                          │
│  │ (Qwen 2B, cores 16-23│  │ (Qwen 2B, cores 24-31│                        │
│  │ KV shard (local)     │  │ KV shard (local)     │                          │
│  │ Builds subtree C     │  │ Builds subtree D     │                          │
│  └──────────────────────┘  └──────────────────────┘                          │
│                     │                                                        │
│                     ▼                                                        │
│           ┌─── Merge Tree ───┐                                               │
│           │ Combine subtrees │                                               │
│           │ into single tree │                                               │
│           │ (1000+ nodes)    │                                               │
│           └────────┬─────────┘                                               │
│                    ▼                                                         │
│    ┌──── Target Model Verification ────┐                                     │
│    │ DeepSeek-R1-Q4 (all 96 cores)     │                                     │
│    │ Single forward pass               │                                     │
│    │ Tree attention with shared KV     │                                     │
│    │ Cost ≈ 1 token (bandwidth-bound)  │                                     │
│    │ Accepts 5–20 tokens per pass      │                                     │
│    └───────────────────────────────────┘                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

This combines: SpecExec (large tree + batch verify), multi-worker drafting (SpecInfer/SpecHub), KV-cache tree sharing (Section 6), NUMA-aware partitioning (Ghidorah/Section 6.3), phase separation (no concurrency needed), and dynamic tree topology (DySpec/OPT-Tree).

---

## 10. Empirical Validation — EPYC 9655 Verification Profile

> Full experiment details: `docs/experiments/specexec-verification-profile.md`
> Date: 2026-03-10. Hardware: EPYC 9655 (96 cores, 768 GB DDR5-5600). llama.cpp build 8208.

### 10.1 Batch Verification Latency (Phase 1)

Measured prompt-processing time (batch verification analog) using `llama-bench -p <N> -n 0` across batch sizes 1-512 for five target models, NUMA distribute mode.

**Key result — verification cost ratio (time at N=64 / time at N=1)**:

| Model | Size | Quant | 64/1 Ratio | Assessment |
|-------|------|-------|-----------|------------|
| Qwen2.5-7B-f16 | 15 GB | f16 | 1.69x | Near-flat — bandwidth-bound |
| Qwen3.5-0.8B-Q8_0 | 775 MB | Q8_0 | 0.97x | Compute-bound (batch amortizes overhead) |
| Qwen3.5-9B-Q4_K_M | 5.3 GB | Q4_K_M | 4.39x | Linear scaling — dequant-bound |
| Qwen3.5-27B-Q4_K_M | 16 GB | Q4_K_M | 4.05x | Linear scaling — dequant-bound |
| Qwen2.5-Coder-32B-Q4_K_M | 20 GB | Q4_K_M | 4.96x | Linear scaling — dequant-bound |

**The SpecExec thesis (near-flat verification) holds only for f16 models on this hardware.** Q4_K_M models show ~4-5x cost growth at batch 64 due to dequantization compute overhead. The CPU DDR5 bandwidth regime differs from the GPU HBM bandwidth regime SpecExec targets.

NUMA `distribute` is 75-94% faster than `isolate` for single-token verification on large models.

### 10.2 Draft Model Cost Profiling (Phase 2)

Per-token generation speed for 9 draft candidates (`llama-bench -p 0 -n 128`). Fastest drafters: Qwen2.5-Coder-0.5B-Q8_0 (185 t/s, 5.4 ms/tok) and Qwen3-Coder-0.75B-Q4_0 (181 t/s). Slowest: Qwen3.5-0.8B-Q8_0 (44 t/s, 22.6 ms/tok) — the Qwen3.5 architecture has unexpectedly high per-token overhead for its size.

Critical ratio (draft cost / target verify-1-token): ranges from 0.038 (Coder-0.5B → 32B, can draft 26 tokens per verification) to 0.275 (Qwen3.5-0.8B → 9B, only 3 tokens per verification).

### 10.3 Large-K Linear Speculation (Phase 3)

End-to-end throughput with `--draft-max K` from 16 to 256 for four target+draft pairs, 20 prompts each.

| Pair | K=16 t/s | K=64 t/s | K=256 t/s | Trend |
|------|----------|----------|-----------|-------|
| Qwen2.5-7B + 0.5B | 42.0 | 43.0 | 43.3 | Flat |
| Qwen2.5-Coder-32B + 0.5B | 16.8 | 16.8 | 17.1 | Flat |
| Qwen3.5-9B + 0.8B | 11.9 | 11.8 | 11.7 | Flat |
| Qwen3.5-27B + 0.8B | 6.9 | 6.7 | 5.2 | Degrading |

**Throughput is flat from K=16 to K=256.** Linear speculation saturates at K~16 because acceptance decays geometrically — the probability of accepting >20 tokens in a sequence is negligible. Extra draft tokens beyond K=16 are generated but almost never accepted.

### 10.4 Implications for Tree Speculation

The Phase 3 result is the strongest argument **for** tree speculation: linear K is saturated, but the verification budget (Phase 1) allows processing more candidates. A tree with branching factor 2-4 and depth 4-5 would produce ~16-64 candidates, each a short high-acceptance-probability path, yielding ~8-12 accepted tokens per cycle vs ~4-6 for linear.

However, Phase 1 shows verification cost is **not** near-free for Q4_K_M models (4-5x at N=64). Net expected gain from tree speculation: **1.5-2.5x over linear K=16** for Q4_K_M targets, potentially higher for f16 targets where verification is genuinely near-flat.

The verification function `common_sampler_sample_and_accept_n()` (`common/sampling.cpp:521-548`) implements linear-only verification. Tree extension estimated at ~260-370 LOC. Blocked on upstream tree attention support in llama.cpp.

---

## 11. Empirical Results — HSD, Self-Speculation, and Hybrid SSM (March 2026)

> Hardware: EPYC 9655 (96 cores, 768 GB DDR5-5600). llama.cpp build 8208, branch `feature/ssm-checkpoint-opt`.
> Full handoff: `epyc-root/handoffs/active/hsd-hierarchical-self-speculation.md`

This section documents experiments with four advanced speculation techniques on two model architectures: **Qwen3.5** (hybrid SSM — Mamba2 + attention every 4th layer) and **Qwen3** (dense — pure attention). The results reveal sharp architectural boundaries for speculation viability.

### 11.1 Self-Speculation on Hybrid SSM (Qwen3.5)

Using `--n-layer-exit-draft` for self-speculation, where the same model serves as both target and draft by exiting early during draft generation.

| Config | 9B t/s | Delta | Accept Rate | 27B t/s | Delta |
|--------|--------|-------|-------------|---------|-------|
| baseline | 15.91 | — | — | 4.51 | — |
| external 0.8B | 10.59 | -33% | 62.5% | 3.51 | -22% |
| self-spec exit=8 | 8.83 | -44% | 77.1% | — | — |
| self-spec exit=16 | 7.76 | -51% | 69.9% | 2.85 | -37% |
| prompt lookup | SEGFAULT | — | — | SEGFAULT | — |

**All speculation configs slower than baseline.** Acceptance rates are high (69-77%) — the early-exit logits are reasonable because Qwen3.5's attention layers (every 4th) carry most of the predictive signal. But SSM checkpoint/restore overhead dominates: each speculation round requires copying ~2 GB of recurrent state (GPU→CPU→GPU roundtrip for save, CPU→GPU for restore), plus a re-advance decode after restoring.

Prompt lookup previously segfaulted because the stateless n-gram cache generates drafts without knowledge of recurrent state — verification decodes on mismatched SSM state. Fixed by auto-activating freeze-recurrent for all speculation on hybrid models (not just lookup). Lookup now works on hybrid models with frozen SSM state, same acceptance trade-off as external draft with freeze-recurrent.

### 11.2 Self-Speculation on Dense (Qwen3-32B)

| Config | 32B t/s | Delta | Accept Rate |
|--------|---------|-------|-------------|
| baseline | 7.85 | — | — |
| self-spec exit=16 | ~7.7 | -2% | 1.5% |
| self-spec exit=32 | ~7.7 | -2% | 0.5% |

**Near-zero acceptance rates.** Intermediate layer logits don't produce useful next-token predictions without early-exit training (SWIFT/LayerSkip). The ~2% slowdown comes from the overhead of running the draft decode loop with no accepted tokens. Self-speculation is architecturally blocked on dense models without fine-tuning.

### 11.3 HiSpec Intermediate Verification

Hierarchical speculative decoding (`--hierarchical-spec`): two-pass verification where an intermediate decode at partial depth pre-filters draft tokens before full verification.

| Config | Qwen3-32B t/s | vs Baseline | Accept Rate |
|--------|---------------|-------------|-------------|
| baseline | 8.44 | — | — |
| external Qwen2.5-Coder-0.5B | **13.07** | **+54.9%** | 52.5% |
| external Qwen3-0.6B | **13.06** | **+54.7%** | 50.8% |
| HiSpec intermediate=16 | 7.57 | -10.3% | 28.0% |
| HiSpec intermediate=32 | 7.50 | -11.1% | 26.8% |

**HiSpec is a bust on dense models.** The intermediate decode at partial depth rejects good draft tokens — acceptance halves compared to standard verification. Same root cause as self-spec: untrained intermediate layers can't predict next tokens accurately enough to pre-filter. The extra decode pass costs more than the savings from filtering.

### 11.4 SSM Checkpoint Optimization

Double-buffer pointer swap: pre-allocate shadow r_l/s_l tensors, `ggml_backend_tensor_copy()` for checkpoint, `std::swap()` for O(1) restore. Also partial cell metadata save (1-8 active cells vs 1024).

| Config | Pre-opt t/s | Post-opt t/s | Delta | Accept Rate |
|--------|-------------|--------------|-------|-------------|
| baseline | 15.91 | 15.76 | -1% | — |
| external Qwen3.5-0.8B | 10.59 | 11.60 | +9.5% | 62.8% |
| external Qwen2.5-Coder-0.5B | — | 14.58 | -7.5% vs base | 60.7% |
| self-spec exit=8 | 8.83 | 8.27 | -6% | 75.0% |

Checkpoint optimization shows measurable improvement (+9.5% on external draft), but still not enough to overcome the fundamental overhead. The Qwen2.5-Coder-0.5B drafter reaches 14.58 t/s — the closest speculation has come to baseline on hybrid, but still 7.5% slower.

### 11.5 Freeze-Recurrent Speculation — BREAKTHROUGH

`--freeze-recurrent-draft`: disable SSM state writes during speculation, eliminating checkpoint/save/restore/re-advance entirely. The SSM state becomes stale during speculation (draft tokens don't update it), but the attention layers still provide useful context.

| Config | t/s | vs Baseline | Accept Rate |
|--------|-----|-------------|-------------|
| baseline | 15.14 | — | — |
| external Qwen2.5-Coder-0.5B | 15.15 | +0.1% | 62.7% |
| **freeze + ext Qwen2.5-Coder** | **15.96** | **+5.4%** | 47.9% |
| freeze + ext Qwen3.5-0.8B | 12.06 | -20.3% | 48.6% |

**First time speculation beats baseline on hybrid SSM.** Acceptance drops ~13 percentage points (stale SSM state), but the fast Qwen2.5-Coder-0.5B drafter (185 t/s) compensates. The key insight: freeze-recurrent trades acceptance quality for zero overhead — and on a fast-enough drafter, this trade is net positive.

Note: without freeze-recurrent, external draft barely breaks even (15.15 t/s, +0.1%) — the checkpoint/restore overhead is exactly cancelled by speculation gains. This validates freeze-recurrent as essential for hybrid SSM speculation.

### 11.6 External Draft on Dense (Qwen3-32B) — Production Config

| Config | t/s | vs Baseline | Accept Rate |
|--------|-----|-------------|-------------|
| baseline | 8.44 | — | — |
| **external Qwen2.5-Coder-0.5B** | **13.07** | **+54.9%** | 52.5% |
| **external Qwen3-0.6B** | **13.06** | **+54.7%** | 50.8% |

External draft with a fast drafter yields the strongest result across all experiments. Both drafters achieve near-identical throughput despite different architectures. This is the validated production configuration for dense models.

### 11.7 HSD Capped Branch Resampling — A/B Validation

A/B benchmark isolating HSD's marginal contribution on dense Qwen3-32B with external draft (Qwen2.5-Coder-0.5B), 20 prompts, `--draft-max 16`, temp=0.7.

| Config | t/s | Draft Accepted | Draft Total | Acceptance Rate |
|--------|-----|----------------|-------------|-----------------|
| WITH HSD (default) | 13.03 | 2319 | 4362 | 53.16% |
| WITHOUT HSD (`--no-hsd`) | 12.93 | 2290 | 4388 | 52.18% |

**HSD adds +29 accepted tokens (+1.3%), +0.98pp acceptance, +0.8% throughput.** The effect is small because the `p_draft > 0.3` threshold is conservative and at most one recovery is allowed per speculation sequence. The improvement is essentially free — no measurable overhead from the probability lookup. Kept enabled by default.

### 11.8 Cross-Architecture Drafting

Dense drafters (Qwen2.5-Coder-0.5B) work for hybrid targets (Qwen3.5-9B): 48-63% acceptance rates across all configs. The drafter doesn't need to match the target architecture — it only needs to produce reasonable candidate tokens. The target model's full forward pass handles quality verification.

### 11.9 Decision Matrix

Which speculation config to use per architecture:

| Target Architecture | Recommended Config | Expected Gain | Notes |
|----|----|----|-----|
| Dense (Qwen3, Qwen2.5, Llama) | External draft (Qwen2.5-Coder-0.5B) | **+50-55%** | Production validated. HSD capped branch resampling adds marginal improvement. |
| Dense f16 (high-quality targets) | External draft + tree speculation (`--draft-p-split 0.1`) | **+4.4%** (measured, 7B) | Near-flat verification makes tree profitable. Overhead ~41ms/round is hard floor — gains are modest, not the +100-200% theoretical. See §12. |
| Hybrid SSM (Qwen3.5, Qwen3.5-MoE) | Freeze-recurrent + external draft | **+5-10%** | Only viable config. Must use fast drafter (>150 t/s). Slower drafters net negative. |
| Hybrid SSM (without freeze-recurrent) | No speculation | — | Checkpoint/restore overhead neutralizes all speculation gains. |
| Any (self-speculation) | Not recommended | Negative | Requires SWIFT/LayerSkip fine-tuning. Without it: near-zero acceptance (dense) or checkpoint-dominated (hybrid). |
| Any (HiSpec intermediate) | Not recommended | Negative | Intermediate logits untrained for next-token prediction. Rejects good drafts. |

---

## 12. Empirical Results — Tree Speculation (DySpec) on CPU (March 2026)

> Hardware: EPYC 9655 (96 cores, 768 GB DDR5-5600). llama.cpp build on `feature/tree-speculation` branch off `production-consolidated-v2`.

### 12.1 DySpec Implementation Summary

Heap-based dynamic tree construction in `common/speculative.cpp`. Branching factor: 7 at root → 5 (depth ≤ 2) → 3 (depth ≤ 4) → 2 (deeper). Cap 32 seq_ids. Primary child inherits parent seq_id/sampler (zero clone cost); alternatives clone KV via `llama_memory_seq_cp` + `common_sampler_clone`. Multi-path target verification for dense models (`--kv-unified`).

### 12.2 Baseline Results (Phases 1-3)

| Pair | Target + Drafter | Quant | Arch | Linear t/s | Tree Best t/s | +Tokens | Delta | Multi-path |
|------|------------------|-------|------|-----------|---------------|---------|-------|------------|
| 1 | Qwen2.5-7B + 0.5B | f16 | Dense | 29.45 | 30.76 | +25 | **+4.4%** | Yes |
| 8 | Qwen3-235B-A22B + 0.6B | Q4_KM | MoE | 10.97 | 11.19 | +35 | **+2.0%** | Yes |
| 5 | Qwen2.5-Coder-32B + 0.5B | Q4_KM | Dense | 19.50 | 18.84 | +20 | -3.4% | Yes |
| 7 | DS-R1-Distill-32B + 0.5B | Q6_K | Dense | 11.57 | 10.60 | +44 | -8.4% | Yes |
| 6 | Qwen3.5-122B-A10B + 0.8B | Q4_KM | Hybrid | 5.75 | 4.36 | — | -24% | No (SSM) |

Tree wins when target verification latency per round exceeds construction overhead (~41ms). Two regimes: f16 targets (near-flat verification, 1.69x at N=64) and large MoE (slow verification per round). Medium dense Q4/Q6 (32B) verifies too fast — overhead dominates.

### 12.3 Phase 5a — Branching Reduction Experiment (REVERTED)

Hypothesis: Reducing branching factor cuts KV copies proportionally, lowering overhead from ~41ms to ~20-25ms.

Changes tested: MAX_BRANCHES 7→4, branching 4/3/2 (was 7/5/3/2), seq_id cap 32→16.

| Pair | Target | Pre-5a Best Tree | Post-5a Best Tree | Verdict |
|------|--------|-----------------|-------------------|---------|
| 1 | Qwen2.5-7B f16 | 33.85 (+2.7% vs 32.97 linear) | was +4.4% | **Regressed** — fewer branches → fewer alt paths accepted |
| 5 | Qwen2.5-Coder-32B Q4 | 19.37 (-4.1% vs 20.20 linear) | was -5.2% | **Improved 1pp** — but still net negative |
| 8 | Qwen3-235B MoE | 8.18 (-6.6% vs 8.76 linear) | was +2.0% | **Regressed badly** — MoE benefited from wider branching |

**Key finding**: Branching reduction is a false economy. It cuts both overhead AND acceptance quality. The wider tree was finding genuinely useful alternative paths on targets where verification is cheap (f16) or inherently slow (MoE). The Q4 target improved marginally because its fast verification couldn't exploit alt paths anyway.

All changes reverted. Tree construction overhead (~41ms/round) is a hard floor given llama.cpp's full-range `seq_cp` requirement.

### 12.4 Phase 5 — Overhead Reduction Summary (All Paths Exhausted)

| Approach | Status | Reason |
|----------|--------|--------|
| 5a. Reduce branching | ❌ Tested, reverted | Hurts acceptance more than it helps overhead |
| 5b. Lazy KV copies | ❌ Not viable | `llama_kv_cache.cpp` asserts full-range for cross-stream seq_cp; can't defer (decode needs per-alt seq_id) |
| 5c. Sampler pooling | ❌ Not viable | Mirostat state prevents revert; clone cost (~1ms) is irreducible; total sampler overhead (~50-73ms) is secondary to KV copies |

**Implication**: The ~41ms/round construction overhead is architectural in llama.cpp. The only viable path is Phase 6 (adaptive tree sizing — bypass overhead entirely on easy/hard prompts where tree adds no value).

### 12.5 Viability Assessment

Tree speculation is production-viable for:
- **f16 targets** (any size): Near-flat verification cost. +4.4% demonstrated on 7B, likely higher on larger f16 models.
- **Large MoE Q4 (235B+)**: Slow per-round verification makes extra accepted tokens valuable. +2.0% on 235B.

Tree speculation is NOT viable for:
- **Medium dense Q4/Q6 (32B)**: Verification too fast (~50ms/round), overhead dominates. Consistently -3% to -8%.
- **Hybrid/SSM models**: Multi-path verification fundamentally incompatible (recurrent state can't fork). See Section 13 for Delta Net-specific approaches.

Production recommendation: Enable tree (`--draft-p-split 0.1`) only for `architect_general` (235B MoE) and any f16 targets. Keep linear for 32B Q4 workers.

Data: `epyc-inference-research/data/tree_speculation/server_sweep_2026031{1,4,5}_*.csv`

---

## 13. Tree Speculation for Delta Net Hybrid Models — Architecture Analysis (March 2026)

### 13.1 The Architecture Misconception

Our hybrid models (Qwen3.5-35B-A3B, Qwen3-Next-80B-A3B) were labeled "75% Mamba layers" in early documentation. This is **incorrect**. They use **Delta Net** (gated linear attention), implemented in `delta-net-base.cpp`, NOT Mamba2's `ggml_ssm_scan`.

The Delta Net recurrence:
```
s_t = exp(g_t) * s_{t-1} + k_t ⊗ (beta_t * (v_t - s_{t-1}^T k_t))
```

Key differences from Mamba2:
- **State shape**: Outer-product matrix `{S_v, S_v, H_v, n_seqs}` (not Mamba2's `{d_state, head_dim, n_head}`)
- **Decay**: `g = softplus(alpha + bias) * ssm_a` — multiplicative, analogous to Mamba2's `exp(dA)`
- **Nonlinearity**: The `s_old^T k` term in the delta creates a state-dependent update that Mamba2 lacks
- **Two compute paths**: Autoregressive (single token, lines 291-376) and chunked (multi-token, chunk_size=64, lines 15-289)

True Mamba2 models in llama.cpp (Jamba, Falcon-H1, Granite-Hybrid, PLaMo2/3) use `ggml_ssm_scan` and have the diagonal `A` matrix that enables STree's tree-masked scan. None of these are in our production stack.

### 13.2 Why STree Doesn't Apply to Delta Net

STree ([arXiv:2505.14969](https://arxiv.org/abs/2505.14969)) converts Mamba2's sequential scan into a tree-masked matrix multiply by exploiting the **linear** state recurrence:
```
Mamba2: s_new = dA * s_old + B * u    (linear in s — log-space trick works)
```

Delta Net's recurrence is **nonlinear** in state:
```
Delta Net: s_new = exp(g) * s_old + k ⊗ beta * (v - s_old^T k)
```

The `s_old^T k` term makes the update depend on the current state value, not just a multiplicative decay. This prevents the cumulative log-sum factorization that STree relies on. The tree-masked cumulative sum `dA_tree[s] = dA_tree[parent(s)] + dA[s]` has no analogue for the Delta Net update.

### 13.3 Viable Approaches

#### Approach A: Sequential State Replay (Practical)

Replay Delta Net states sequentially for each tree path:
1. Snapshot recurrent state before speculation
2. For each root-to-leaf path: restore snapshot, run autoregressive Delta Net per token
3. Attention layers batch across paths via `seq_id` (existing Phase 3 infrastructure)
4. Accept longest matching prefix across all paths

**Cost**: N_paths × path_length autoregressive passes through recurrent layers only. For 8 paths of depth 5: ~40 recurrent-layer passes at ~1ms/token/layer ≈ 40-60ms. Attention layers add ~0ms marginal (already batched).

**Correctness**: Exact — identical to sequential decoding. No approximation.

**Expected throughput**: +20-35% on Qwen3.5-35B vs baseline, recovering the ~13pp acceptance loss from freeze-recurrent.

#### Approach B: Linearized Delta Net (Research)

Freeze the `s_old^T k` product at the base state value:
```
s_t ≈ exp(g_t) * s_{t-1} + k_t ⊗ beta_t * (v_t - s_base^T k_t)
```

This makes the recurrence linear, enabling tree-masked cumulative sum (adapted from STree). The existing `build_delta_net_chunking` code (lines 88-106 in `delta-net-base.cpp`) already computes `g_cs = cumsum(g)` and builds `exp(g_cs_j - g_cs_i)` — a tree version would replace `cumsum` with tree-masked accumulation.

**Trade-off**: Approximation quality degrades with path length. Viability risk ~40%. If KL divergence exceeds 0.5 at depth 3, linearization is too lossy.

**Advantage**: Single forward pass for all tree paths — eliminates per-path replay overhead entirely.

### 13.4 STree Reference (for Mamba2 Models)

STree's approach works directly on true Mamba2 models. Key results (MambaInLlama-8B, 50% hybrid, RTX 3090):
- Greedy decoding: 1.74x speedup (vs 1.69x vanilla speculative decoding)
- Temperature=1: 1.36x (vs 1.23x vanilla)
- Memory: 0.57-0.91x of unrolled baseline

Core algorithm: tree-structured mask `L` on cumulative state transitions:
```
dA_tree[0] = dA[0]
for s in 1..N-1:
    dA_tree[s] = dA_tree[parent(s)] + dA[s]
```

**Repository**: [github.com/wyc1997/stree](https://github.com/wyc1997/stree). **Limitation**: Asserts `seqlen < chunk_size` — tree must fit in one chunk.

### 13.5 TIDE Implementation Status

The self-speculation experiments in Sections 11.1–11.2 identified the core blocker: llama.cpp had no per-decode adaptive layer exit. As of April 2026, this has been addressed with the **TIDE (Token-level Intelligent Dynamic Exit)** infrastructure in our llama.cpp fork.

#### n_layer_exit API

A new public API allows callers to change the effective model depth between decode calls:

```c
// include/llama.h
LLAMA_API void llama_set_n_layer_exit(struct llama_context * ctx, int32_t n_layer_exit);
```

When `n_layer_exit > 0 && n_layer_exit < n_layer`, the forward pass stops after layer `n_layer_exit` and projects directly to the vocabulary via the LM head. Setting `n_layer_exit = 0` restores full-depth evaluation. This is implemented in the `build_graph` function of each supported architecture:

- `src/models/qwen35moe.cpp` — Qwen3.5 MoE (hybrid SSM)
- `src/models/gemma4-iswa.cpp` — Gemma4 ISWA
- `src/models/minimax-m2.cpp` — MiniMax-M2

The per-decode granularity means a router can decide the exit layer for each token independently, enabling the self-speculation and hierarchical verification patterns described in Sections 2 and 3 without requiring a separate draft model.

#### Calibration Tooling

Two complementary calibration tools exist:

1. **C++ calibration tool** (`tools/tide-calibrate/tide-calibrate.cpp` in epyc-llama): Uses `cb_eval` callback to capture `l_out-N` tensors at checkpoint layers during a single forward pass. Computes per-layer cosine similarity between intermediate and final hidden states. **Single-pass callback method is 36x faster than pairwise** (1 forward pass per sample vs 2*(N-1) in pairwise mode). Falls back to pairwise automatically if the callback fails to capture all checkpoints.

2. **Python calibration script** (`scripts/benchmark/calibrate_tide_router.py` in epyc-inference-research): Records full hidden states at checkpoint layers (every 4th layer by default) for flexible experimentation with router architectures. Includes router MLP training from stored hidden states.

#### Relationship to Earlier Findings

The n_layer_exit API directly addresses the limitations identified in:
- **Section 11.2** (self-spec on dense models): Near-zero acceptance without early-exit training. TIDE provides the infrastructure — a trained router at each checkpoint layer could predict when intermediate logits are sufficient.
- **Section 11.3** (HiSpec intermediate verification): Failed because untrained intermediate layers reject good drafts. TIDE's calibration data enables training per-layer confidence predictors to make informed exit decisions.
- **Section 2.1** (HiSpec): The "EE-capable models and llama.cpp does not support early-exit inference" blocker is now partially lifted — the API exists, but production use requires calibrated routers.

**Projection Results (2026-04-23):**
- Speed: 1.76x confirmed (8.4 vs 4.8 t/s) at 50% layer exit on Qwen3.6-27B
- Linear projection (n_embd x n_embd): perfect on calibration data, fails on unseen prompts
- Element-wise RMSNorm: fails (cos=0.45, can't rotate hidden states)
- Open problem: need adapter that generalizes — bottleneck MLP or LoRA-style fine-tuning

### 13.6 Research References

1. Wu et al., "STree: Speculative Tree Decoding for Hybrid State-Space Models," NeurIPS 2025. [arXiv:2505.14969](https://arxiv.org/abs/2505.14969)
2. Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule," NeurIPS 2024. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) (GDN/Delta Net architecture)
3. Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length," NeurIPS 2024. [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) (Chunked Delta Net algorithm)
4. Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers," ICML 2021. [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) (Delta Net foundation)
5. Chen et al., "RAD: Redundancy-Aware Distillation for Hybrid Models," May 2025. [arXiv:2505.22135](https://arxiv.org/abs/2505.22135)

---

## References

### Tree Speculative Decoding
1. Miao et al., "SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification," ASPLOS '24. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
2. Chen et al., "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding," ICML '24. [arXiv:2402.12374](https://arxiv.org/abs/2402.12374). [Code](https://github.com/Infini-AI-Lab/Sequoia)
3. Wang et al., "OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure," TACL 2025. [arXiv:2406.17276](https://arxiv.org/abs/2406.17276). [Code](https://github.com/Jikai0Wang/OPT-Tree)
4. Xiong et al., "DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure," WWW 2025. [arXiv:2410.11744](https://arxiv.org/abs/2410.11744)
5. "SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices," Jun 2024. [arXiv:2406.02532](https://arxiv.org/abs/2406.02532)
6. "Dynamic Delayed Tree Expansion," Feb 2026. [arXiv:2602.16994](https://arxiv.org/abs/2602.16994)

### Hierarchical Speculative Decoding
7. Kumar et al., "HiSpec: Hierarchical Speculative Decoding for LLMs," Oct 2025. [arXiv:2510.01336](https://arxiv.org/abs/2510.01336)
8. Zhou et al., "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding," Jan 2026. [arXiv:2601.05724](https://arxiv.org/abs/2601.05724). [Code](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding)
9. SpecBundle EAGLE-3 Draft Models. [HuggingFace](https://huggingface.co/collections/lmsys/specbundle)

### Self-Speculative Decoding
10. Zhao et al., "Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding," Dec 2025. [arXiv:2512.01278](https://arxiv.org/abs/2512.01278)
11. Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding," Meta, ACL 2024. [arXiv:2404.16710](https://arxiv.org/abs/2404.16710). [Code](https://github.com/facebookresearch/LayerSkip)
12. "SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration," ICLR 2025. [arXiv:2410.06916](https://arxiv.org/abs/2410.06916)
13. EAGLE-3: "Scaling up Inference Acceleration." [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)

### Parallel and Multi-Draft Speculation
14. "ParallelSpec: Parallel Drafter for Efficient Speculative Decoding," Oct 2024. [arXiv:2410.05589](https://arxiv.org/abs/2410.05589)
15. "PEARL: Parallel Speculative Decoding with Adaptive Draft Length," ICLR 2025. [arXiv:2408.11850](https://arxiv.org/abs/2408.11850)
16. "SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding," EMNLP 2024. [arXiv:2411.05289](https://arxiv.org/abs/2411.05289)

### Heterogeneous Processor Partitioning
17. Wei et al., "Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism," Jun 2025. [arXiv:2505.23219](https://arxiv.org/abs/2505.23219)
18. "Dovetail: CPU/GPU Heterogeneous Speculative Decoding," EMNLP 2025. [arXiv:2412.18934](https://arxiv.org/abs/2412.18934)

### Speculative Speculative Decoding
19. Kumar, Dao & May, "Speculative Speculative Decoding," ICLR 2026. [arXiv:2603.03251](https://arxiv.org/abs/2603.03251). [Code](https://github.com/tanishqkumar/ssd)

### SSM / Hybrid Model Speculation
20. Wu et al., "STree: Speculative Tree Decoding for Hybrid State-Space Models," NeurIPS 2025. [arXiv:2505.14969](https://arxiv.org/abs/2505.14969). [Code](https://github.com/wyc1997/stree)
21. Yang et al., "SpecMamba: Accelerating Mamba Inference on FPGA with Speculative Decoding," ICCAD 2025. [arXiv:2509.19873](https://arxiv.org/abs/2509.19873)
22. Chen et al., "RAD: Redundancy-Aware Distillation for Hybrid Models via Self-Speculative Decoding," May 2025. [arXiv:2505.22135](https://arxiv.org/abs/2505.22135)
23. "The Mamba in the Llama: Distilling and Accelerating Hybrid Models," NeurIPS 2024. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/723933067ad315269b620bc0d2c05cba-Paper-Conference.pdf)

### Delta Net / Gated Linear Attention
24. Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule," NeurIPS 2024. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
25. Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length," NeurIPS 2024. [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
26. Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers," ICML 2021. [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)

### Related / Background
27. "CLaSp: In-Context Layer Skip for Self-Speculative Decoding." [arXiv:2505.24196](https://arxiv.org/abs/2505.24196)
28. Speculative Decoding Papers Collection. [GitHub](https://github.com/hemingkx/SpeculativeDecodingPapers)
29. llama.cpp LayerSkip Discussion. [#10787](https://github.com/ggml-org/llama.cpp/discussions/10787)
30. llama.cpp Speculative Decoding Docs. [speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
