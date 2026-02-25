# AMD EPYC 9655 Inference Optimization - Research Progress

**Project Start:** December 2025
**Last Updated:** 2026-01-13
**Hardware:** AMD EPYC 9655 "Turin" (96 cores, 192 threads), 1.13 TB DDR5-5600, 2x 2TB NVMe RAID0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Timeline](#2-research-timeline)
3. [Speculative Decoding Investigation](#3-speculative-decoding-investigation)
4. [MoE Expert Reduction Research](#4-moe-expert-reduction-research)
5. [Prompt Lookup Decoding](#5-prompt-lookup-decoding)
6. [Orchestration System Development](#6-orchestration-system-development)
7. [Benchmarking Framework Evolution](#7-benchmarking-framework-evolution)
8. [AMD PACE Investigation](#8-amd-pace-investigation)
9. [Infrastructure Contributions](#9-infrastructure-contributions)
10. [Reproduction Results Summary](#10-reproduction-results-summary)
11. [Deprecated Approaches](#11-deprecated-approaches)
12. [Future Directions](#12-future-directions)
13. [Literature References](#13-literature-references)

---

## 1. Project Overview

### Objective

Optimize local LLM inference on AMD EPYC 9655 CPU for a hierarchical multi-agent orchestration system. The goal is to achieve production-ready inference speeds for interactive and batch workloads without GPU acceleration.

### Key Constraints

- **CPU-only inference**: No GPU available; must leverage AVX-512, high core count, and large RAM
- **Memory bandwidth bound**: ~460 GB/s theoretical, actual inference limited by memory access patterns
- **Mixed model architectures**: Dense, MoE (Mixture of Experts), and SSM-hybrid models require different optimization strategies
- **Quality preservation**: Speedups must not degrade output quality below acceptable thresholds

### Best Results Achieved

| Method | Speedup | Use Case | Status |
|--------|---------|----------|--------|
| Prompt Lookup (summarization) | **12.7x** | Document QA, summarization | Production |
| External Draft (Qwen2.5-Coder-32B + 0.5B, K=24) | **11x** | Code generation | Production |
| Prompt Lookup (code editing) | **8.6x** | Refactoring, code review | Production |
| MoE Expert Reduction (4 experts) | **+21-52%** | MoE models | Production |

---

## 2. Research Timeline

### December 2025

| Date | Milestone | Details |
|------|-----------|---------|
| Dec 11-14 | Initial speculative decoding exploration | Tested external draft models, EAGLE-1, baseline measurements |
| Dec 15 | NeurIPS 2025 literature review | Identified 8 new research tracks from recent papers |
| Dec 15 | Strategic action plan created | Prioritized tracks, deprecated EAGLE |
| Dec 16 | Orchestration Phase 1 complete | TaskIR, dispatcher, executor, 184 unit tests |
| Dec 18 | Claude-as-Judge framework | Independent quality evaluation of 32 models |
| Dec 18 | Benchmark hardening | Removed trivial questions, added post-doctoral T3 |
| Dec 21 | Parallel tensor repack | 2.2x faster model loading, PR submitted upstream |
| Dec 21-24 | MoE expert sweep | Comprehensive quality/speed testing for 2-8 experts |

### January 2026

| Date | Milestone | Details |
|------|-----------|---------|
| Jan 1 | Benchmark error reporting improvements | Better error extraction from llama.cpp output |
| Jan 1 | llama.cpp PR #18239 CI fix | OpenMP guard fix for upstream contribution |
| Jan 2 | AMD PACE setup complete | Dependencies installed, benchmark script ready |
| Jan 4 | RLM integration planning | Recursive Language Models architecture design |
| Jan 4 | Research consolidation | This document created |
| Jan 13 | TTT research completed | Fixed llama.cpp training, benchmarked vs baseline, NO-GO |

---

## 3. Speculative Decoding Investigation

### 3.1 Track 1: External Draft Model (Production)

**Status:** ✅ Production

**Method:** Use a small draft model (0.5B-1B parameters) to propose tokens, verify with large target model.

**Implementation:**
```bash
llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96
```

**Results:**

| Target Model | Draft Model | K | Acceptance | Speed | Speedup |
|--------------|-------------|---|------------|-------|---------|
| Qwen2.5-Coder-32B Q4_K_M | Qwen2.5-Coder-0.5B Q8_0 | 24 | ~50% | 33.0 t/s | **11x** |
| Qwen2.5-72B Q4_K_M | Qwen2.5-0.5B Q8_0 | 16 | ~45% | 8.53 t/s | **5.8x** |
| Meta-Llama-70B | PARD-Llama-1B Q8_0 | 8 | ~60% | 84 t/s | ~5x |

**Key Findings:**
- Optimal K depends on acceptance rate; K=24 works well for code generation
- Draft model must share tokenizer with target (or be compatible)
- Q8_0 quantization for draft preserves acceptance rate better than Q4
- Very small drafts (0.5B) outperform larger drafts (1B-3B) on CPU due to memory bandwidth

**Files:**
- `research/speculative_decoding_research.md` - Initial exploration
- `research/SPECULATIVE_DECODING_REPORT.md` - Detailed methodology
- `research/speculative_decoding_results.md` - Raw benchmark data

---

### 3.2 Track 3: EAGLE-1 (Deprecated)

**Status:** ❌ Deprecated after 20+ hours debugging

**Method:** Train autoregressive head on top of target model's hidden states to predict future tokens.

**What We Tried:**
1. Downloaded EAGLE-1 checkpoints for Vicuna-7B and LLaMA2-Chat-7B
2. Converted to GGUF format
3. Integrated with llama.cpp EAGLE implementation
4. Extensive debugging of attention mask, layer extraction, quantization

**Results:**
- **0% acceptance rate** across all configurations
- Attention output was correct (verified against reference)
- Hidden state extraction appeared correct
- Root cause: Quantization mismatch between EAGLE checkpoint (FP16) and GGUF target (Q4_K_M)

**Root Cause Analysis:**
1. EAGLE was trained on FP16 hidden states
2. Quantized model produces different hidden state distributions
3. No SpecMQuant checkpoints available for tested models
4. Would require retraining EAGLE on quantized model outputs

**Lessons Learned:**
- EAGLE requires exact model match (same weights, same precision)
- Quantization fundamentally changes hidden state space
- Without training infrastructure, EAGLE is not viable for quantized models

**Time Investment:** ~20+ hours
**Recommendation:** Do not revisit unless SpecMQuant checkpoints become available

---

### 3.3 Track 7: CAS-Spec / Self-Speculative (Deprecated)

**Status:** ❌ Deprecated

**Method:** Use layer-skipped version of target model as draft (no external model needed).

**What We Tried:**
1. Implemented layer-skip inference in llama.cpp
2. Tested various skip ratios (30%, 50%, 70%)
3. Added INT8 quantization for skipped forward pass

**Results:**
- **0.446% acceptance rate** (essentially random)
- Layer-skipped model produces incompatible hidden states
- Without trained exit classifiers, layer skip is not viable

**Why It Failed:**
- CAS-Spec paper relies on trained confidence predictors
- Raw layer skip without calibration produces garbage predictions
- INT8 forward pass further degrades hidden state quality

**Related Approaches Also Blocked:**
- CLaSp (ICLR 2025) - Same layer-skip dependency
- SWIFT (ICLR 2025) - Same layer-skip dependency

---

### 3.4 Track 4: Medusa (Skipped)

**Status:** ⏸️ Skipped - Training required

**Method:** Train multiple prediction heads for parallel token speculation.

**Why Skipped:**
- Requires training heads for each target model
- No pre-trained Medusa heads available for Qwen/LLaMA models
- Training infrastructure not available

---

### 3.5 Track 5: SSM Speculation (Blocked)

**Status:** ⛔ Architecturally incompatible

**Method:** Apply speculative decoding to SSM-hybrid models (Qwen3-Next).

**Why Blocked:**
- SSM (State Space Model) architecture requires consecutive token positions
- Speculative decoding proposes multiple tokens in parallel
- Fundamental architectural mismatch - cannot be resolved without model changes

**Affected Models:**
- Qwen3-Next-80B-A3B (SSM-hybrid)
- Any future SSM or Mamba-based models

**Workaround:** Use MoE expert reduction only for SSM models

---

### 3.6 Track 6: SuffixDecoding (High Priority - Untested)

**Status:** 🔬 Untested - High priority for future work

**Method:** Use suffix trees over previous outputs as draft source (model-free).

**Expected Benefits:**
- **10x+ speedup** on agentic/repetitive workloads (reported in paper)
- Zero model overhead - pure string matching
- Compounds with external draft (try suffix first, fall back to draft)

**Implementation Plan:**
1. Build global suffix tree from session history
2. During generation, query tree with recent context
3. If match found, propose continuation as draft
4. Verify with target model

**Why Not Yet Implemented:**
- Requires C++ implementation in llama.cpp
- Need to design efficient suffix tree data structure
- Integration with existing speculation pipeline

**References:**
- Paper: https://suffix-decoding.github.io/ (NeurIPS 2025 Spotlight)

---

## 4. MoE Expert Reduction Research

### Overview

MoE (Mixture of Experts) models activate only a subset of parameters per token. By overriding the expert count, we can reduce memory bandwidth at the cost of quality.

**Implementation:**
```bash
llama-cli -m model.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96
```

### Expert Sweep Results (Qwen3-Coder-480B-A35B)

| Config | Experts | Speed | Speedup | Quality (Claude-as-Judge) |
|--------|---------|-------|---------|---------------------------|
| Baseline | 8 | 6.53 t/s | — | 95% |
| MOE2 | 2 | 12.10 t/s | +85% | **14%** (garbage) |
| **MOE3** | **3** | **10.30 t/s** | **+58%** | **~88%** (good) |
| MOE4 | 4 | 9.25 t/s | +42% | 88% |
| MOE5 | 5 | 8.50 t/s | +30% | ~90% |
| MOE6 | 6 | 6.20 t/s | -5% | 95% |

### Key Findings

1. **MOE3 is optimal** for Qwen3-Coder-480B: Best speed/quality tradeoff
2. **MOE2 produces garbage**: Repetitive text, broken JSON, incoherent reasoning
3. **Quality cliff exists**: Below 3 experts, quality degrades catastrophically
4. **Model-specific tuning required**: Optimal expert count varies by model size/architecture

### MoE Override Keys by Architecture

| Architecture | Override Key | Default Experts |
|--------------|--------------|-----------------|
| Qwen3-MoE | `qwen3moe.expert_used_count` | 8 |
| Qwen3-Next (SSM) | `qwen3next.expert_used_count` | 3 |
| DeepSeek-V2 | `deepseekmoe.expert_used_count` | 6 |
| GLM-4-MoE | `glm4moe.expert_used_count` | 2 |
| Mixtral | `llama.expert_used_count` | 2 |

### Production Configuration

| Role | Model | Experts | Speed | Notes |
|------|-------|---------|-------|-------|
| frontdoor | Qwen3-Coder-30B-A3B | 4 | 45.3 t/s | Always hot |
| coder_primary | Qwen3-Coder-30B-A3B | 4 | 45.3 t/s | Primary code gen |
| ingest_long_context | Qwen3-Next-80B-A3B | 2 | 11.6 t/s | NO speculation (SSM) |
| architect_coding | Qwen3-Coder-480B-A35B | 3 | 10.3 t/s | Escalation only |
| architect_general | Qwen3-235B-A22B | 4 | 6.75 t/s | System design |

---

## 5. Prompt Lookup Decoding

### Overview

**Status:** ✅ Production

Use n-gram matching against input prompt as draft source. When generating output that references input content (summarization, code editing, document QA), the model often copies phrases directly.

**Implementation:**
```bash
llama-cli -m model.gguf \
  --lookup-ngram-min 3 \
  -f prompt_with_source.txt \
  -t 96
```

### Results

| Task Type | Speedup | Notes |
|-----------|---------|-------|
| Summarization (CNN/DailyMail style) | **12.7x** | 95.18 t/s achieved |
| Code editing/refactoring | **8.6x** | 25.82 t/s |
| Document QA | **2-3x** | Entity/phrase copying |
| General generation | **1.0-1.2x** | Minimal gain |

### When to Use

- ✅ Document summarization
- ✅ Code refactoring (input contains source code)
- ✅ Multi-turn chat (previous turns in context)
- ❌ Creative writing (no input overlap)
- ❌ Math/reasoning (output diverges from input)

### Compounding with External Draft

Prompt lookup can be combined with external draft in a fallback chain:
1. Try prompt lookup (zero cost)
2. If no match, use draft model
3. Verify with target

**Not yet implemented** - requires llama.cpp modification

---

## 6. Orchestration System Development

### Architecture

Hierarchical multi-agent system with four tiers:

| Tier | Role | Purpose | Model |
|------|------|---------|-------|
| A | Front Door | Intent classification, task routing | Qwen3-Coder-30B-A3B |
| B | Specialists | Code gen, long context, architecture | Various 30B-480B |
| C | Workers | File-level implementation, tests | 7B-8B models |
| D | Draft | Speculative decoding | 0.5B models |

### Implementation Status

| Component | Status | Files |
|-----------|--------|-------|
| TaskIR Schema | Complete | `orchestration/task_ir.schema.json` |
| ArchitectureIR Schema | Complete | `orchestration/architecture_ir.schema.json` |
| Model Registry | Complete | `orchestration/model_registry.yaml` |
| Registry Loader | Complete | `src/registry_loader.py` |
| Dispatcher | Complete | `src/dispatcher.py` |
| Executor | Complete | `src/executor.py` |
| Context Manager | Complete | `src/context_manager.py` |
| Model Server | Complete | `src/model_server.py` |
| CLI Entry Point | Complete | `src/cli.py` |
| Gate Runner | Planned | — |
| HTTP API | Planned | — |

### Test Coverage

- **184 unit tests** passing
- **9 integration tests** passing
- **2 E2E tests** with real inference

### Key Design Decisions

1. **Hierarchical, not swarm**: Strong models set trajectory, cheap workers execute
2. **Contracts first**: APIs/schemas constrain expansion
3. **Gates decide correctness**: No debate-driven convergence
4. **Mixed acceleration**: Spec decode for dense, MoE reduction for MoE, nothing for SSM

### RLM Integration (Planned)

Integration with Recursive Language Models pattern for long-context handling:
- Root LM operates on context as Python variable (never sees full context)
- Sub-LMs receive narrowed context slices via `llm_call()` / `llm_batch()`
- Expected 60-80% reduction in root LM token usage

**Reference:** Zhang, Kraska, Khattab - "Recursive Language Models" (arXiv:2512.24601)

---

## 7. Benchmarking Framework Evolution

### Initial Approach (Dec 2025)

- Algorithmic rubric-based scoring
- Single-file benchmark scripts
- Results stored in temporary directories

### Problems Identified

- Algorithmic scoring severely underscored models (38% vs 89% Claude-as-Judge)
- Results lost after model deletion
- No structured comparison across runs
- Ceiling effects in questions (top models hit 93%)

### Current Framework

**8 Benchmark Suites:**

| Suite | Purpose | Questions |
|-------|---------|-----------|
| thinking | Chain-of-thought reasoning | 10 |
| coder | Code generation, debugging | 10 |
| vl | Vision-language (OCR, images) | 10 |
| general | Instruction following | 10 |
| agentic | Tool calling, function extraction | 10 |
| math | Mathematical reasoning | 10 |
| long_context | 4K-50K token retrieval | 9 |
| instruction_precision | Exact format compliance | 10 |

**Storage Structure:**
```
benchmarks/
├── prompts/v1/           # Versioned prompt files
│   ├── thinking.yaml
│   ├── coder.yaml
│   └── ...
├── results/
│   ├── runs/{run_id}/    # Per-run results
│   ├── index.jsonl       # Structured index
│   └── reviews/          # Claude-as-Judge scores
```

### Claude-as-Judge Methodology

Independent quality evaluation scoring (0-3):
- **3**: Correct answer with good reasoning
- **2**: Partially correct or truncated
- **1**: Wrong but reasonable attempt
- **0**: Completely wrong, empty, or garbage

**32 models reviewed** as of Dec 2025

### Benchmark Hardening (Dec 18, 2025)

Removed 3 trivial T1 questions per suite, added post-doctoral T3 questions to address ceiling effects.

---

## 8. AMD PACE Investigation

### Status

**Ready to test** as of Jan 2, 2026

### Overview

AMD PACE (Performance Acceleration for CPU Execution) provides native PyTorch inference with PARD speculative decoding, optimized for AMD EPYC.

**Claimed Performance:** 380 t/s on Llama 3.1 8B with PARD

### Setup

- **Repository:** `/mnt/raid0/llm/AMD-PACE/`
- **Environment:** `conda activate pace-env`
- **Benchmark script:** `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py`

### Models Available

| Target | Draft | Purpose |
|--------|-------|---------|
| Qwen2.5-7B-Instruct | PARD-Qwen2.5-0.5B | Baseline comparison |
| Llama-3.1-8B-Instruct | PARD-Llama-3.2-1B | Speed test |
| DeepSeek-R1-Distill-Qwen-32B | PARD-DeepSeek-1.5B | Large model |

### Comparison Points

| Metric | AMD PACE | llama.cpp |
|--------|----------|-----------|
| Format | BF16 safetensors | GGUF Q4_K_M/Q8_0 |
| Memory | ~2x more | Quantized |
| Speed | TBD | 20-85 t/s |

### Decision Criteria

Adopt AMD PACE if:
- ≥2x faster than llama.cpp for equivalent quality
- Stable under extended testing
- Quality matches on benchmark suite

---

## 9. Infrastructure Contributions

### Parallel Tensor Repacking (PR #18239)

**Status:** Merged upstream

**Problem:** Model loading bottlenecked by single-threaded tensor repacking for AVX-512.

**Solution:** Added OpenMP parallelization to 6 repack functions in `ggml-cpu/repack.cpp`.

**Results:**

| Model Size | Before | After | Speedup |
|------------|--------|-------|---------|
| 6.8GB Q4_K | 5.0s | 3.3s | **1.5x** |
| 19GB Q4_K | 11.9s | 5.3s | **2.2x** |
| 271GB Q4_K | ~150s | ~60s | **~2.5x** |

**Files:**
- `patches/llama-cpp-parallel-repack.patch`
- PR: https://github.com/ggml-org/llama.cpp/pull/18239

### Benchmark Infrastructure Fixes

1. **MoE server restart logic**: Fixed benchmark to restart server when expert count changes
2. **Error message extraction**: Improved error reporting to show meaningful errors instead of build banners
3. **Streaming inference**: Implemented partial output collection on timeout
4. **Flash attention integration**: Added `-fa on` for faster long-context prompt processing

---

## 10. Reproduction Results Summary

### Successful Reproductions

| Paper/Claim | Our Result | Notes |
|-------------|------------|-------|
| External draft speedup (llama.cpp) | **5.9-11x** ✅ | Matches documented behavior |
| MoE expert reduction | **+21-52%** ✅ | Qwen3 models respond well |
| Prompt lookup on grounded tasks | **8.6-12.7x** ✅ | Excellent on summarization |
| Parallel tensor repack speedup | **2.2x** ✅ | 96-core scaling confirmed |

### Failed Reproductions

| Paper/Claim | Our Result | Root Cause |
|-------------|------------|------------|
| EAGLE-1 (2-3x speedup) | **0% acceptance** ❌ | Quantization mismatch |
| CAS-Spec (2.3x speedup) | **0.446% acceptance** ❌ | Missing trained classifiers |
| Layer-skip self-speculation | **Not viable** ❌ | Same as CAS-Spec |
| Vanilla TTT (long context) | **Worse than baseline** ❌ | FFN-only training insufficient, needs meta-trained checkpoints |

### Partially Verified

| Claim | Status | Notes |
|-------|--------|-------|
| SuffixDecoding (10x on agentic) | 🔬 Untested | High priority, needs implementation |
| AMD PACE (380 t/s) | 🔬 Untested | Ready to test |
| RLM token efficiency (60-80%) | 🔬 Untested | Architecture designed, needs implementation |

---

## 11. Deprecated Approaches

### Definitively Deprecated

| Approach | Time Invested | Why Deprecated |
|----------|---------------|----------------|
| EAGLE-1 | 20+ hours | 0% acceptance, quantization incompatibility |
| CAS-Spec / Layer-skip | 8 hours | Needs trained classifiers we don't have |
| SSM speculation | 2 hours | Fundamental architecture incompatibility |
| Medusa | 0 hours | Training required, no checkpoints |
| Kangaroo | 0 hours | Adapter training required |
| Vanilla TTT | 8 hours | Worse than baseline on retrieval, requires FP32 + custom checkpoints |

### Conditions for Revival

**EAGLE-1:**
- SpecMQuant checkpoints released for Qwen/LLaMA
- OR llama.cpp adds official EAGLE support with training scripts

**CAS-Spec/CLaSp:**
- Pre-trained exit classifiers released
- OR we develop training infrastructure

**Vanilla TTT:**
- TTT-E2E checkpoints released (meta-trained models)
- OR attention-inclusive training support in llama.cpp
- AND quantization support (currently requires FP32)

---

## 12. Future Directions

### High Priority

1. **SuffixDecoding implementation** - Expected 5-10x on agentic workloads
2. **AMD PACE benchmarking** - May provide significant speedup for BF16 inference
3. **RLM execution engine** - Enable 1M+ token context handling

### Medium Priority

1. **REST datastore** - Retrieval-based speculation from domain corpus
2. **Hybrid drafting** - Prompt lookup → Suffix → External draft fallback chain
3. **Prefix caching** - Shared KV cache for repeated context patterns

### Low Priority

1. **Dynamic speculative depth** - Adaptive K based on acceptance rate
2. **Hugepage optimization** - 1GB pages for reduced TLB misses
3. **NUMA pinning refinement** - Explicit binding vs interleave

---

## 13. Literature References

### Core Speculative Decoding

| Paper | Venue | Link | Notes |
|-------|-------|------|-------|
| Speculative Decoding (Leviathan et al.) | ICML 2023 | Original spec decode paper | Foundation |
| EAGLE: Speculative Sampling | ICML 2024 | https://arxiv.org/abs/2401.15077 | Autoregressive head |
| EAGLE-2 | 2024 | https://arxiv.org/abs/2406.16858 | Improved EAGLE |
| Medusa | 2024 | https://arxiv.org/abs/2401.10774 | Multiple heads |

### NeurIPS 2025

| Paper | Link | Status |
|-------|------|--------|
| SuffixDecoding (Spotlight) | https://suffix-decoding.github.io/ | High priority |
| CAS-Spec | https://arxiv.org/abs/2510.26843 | Tested, failed |
| SpecFormer | https://arxiv.org/abs/2511.20340 | Training required |
| AdaSPEC | NeurIPS 2025 poster | Not evaluated |

### NeurIPS 2024

| Paper | Link | Status |
|-------|------|--------|
| Kangaroo | https://github.com/Equationliu/Kangaroo | Training required |
| Cascade Speculative Drafting | https://arxiv.org/pdf/2312.11462 | Reference |

### ICLR 2025

| Paper | Link | Status |
|-------|------|--------|
| SWIFT | https://openreview.net/forum?id=EKJhH5D5wA | Same issues as CAS-Spec |
| CLaSp | https://arxiv.org/abs/2505.24196 | Same issues as CAS-Spec |

### Other Speculation Methods

| Paper | Link | Notes |
|-------|------|-------|
| REST (Retrieval-Based) | https://arxiv.org/html/2311.08252 | Datastore approach |
| Prompt Lookup | https://github.com/apoorvumang/prompt-lookup-decoding | N-gram matching |
| LayerSkip | https://arxiv.org/abs/2404.16710 | Layer-skip speculation |
| DISCO (Dynamic) | 2024 | Adaptive K |
| SpecInfer | 2024 | LLM serving optimization |

### Recursive Language Models

| Resource | Link | Notes |
|----------|------|-------|
| arXiv Paper | https://arxiv.org/abs/2512.24601 | Zhang, Kraska, Khattab |
| Author's Blog | https://alexzhang13.github.io/blog/2025/rlm/ | Implementation details |
| Prime Intellect RLMEnv | https://www.primeintellect.ai/blog/rlm | Benchmark results |
| GitHub | https://github.com/ysz/recursive-llm | Reference implementation |

### AMD/CPU Optimization

| Resource | Link | Notes |
|----------|------|-------|
| AMD PACE | Internal: `/mnt/raid0/llm/AMD-PACE/` | Native PyTorch |
| PARD Draft Models | HuggingFace | Various PARD-* models |
| llama.cpp | https://github.com/ggml-org/llama.cpp | Main inference engine |
| llama.cpp Fork | https://github.com/pestopoppa/llama.cpp | Local optimizations |
| Parallel Repack PR | https://github.com/ggml-org/llama.cpp/pull/18239 | Our contribution |

### MoE Optimization

| Paper | Link | Notes |
|-------|------|-------|
| Mixture of Experts (Original) | Shazeer et al. 2017 | Foundation |
| DeepSeek-MoE | DeepSeek technical report | Expert routing |
| Qwen3-MoE | Qwen technical report | 3B active params |

### Test-Time Training (Deprecated)

| Resource | Link | Notes |
|----------|------|-------|
| TTT-E2E Paper | https://arxiv.org/abs/2512.23675 | Meta-trained models required |
| TTT-E2E Repo | https://github.com/test-time-training/e2e | JAX, no checkpoints |
| Bug Report | https://github.com/ggml-org/llama.cpp/issues/18805 | llama.cpp training fixes |
| Handoff | `handoffs/archived/vanilla-ttt-feasibility.md` | Full analysis |

---

## Appendix A: Key File Locations

### Research Documents

| Document | Path |
|----------|------|
| This file | `/mnt/raid0/llm/claude/research/RESEARCH_PROGRESS.md` |
| Orchestration Methodology | `/mnt/raid0/llm/claude/research/Hierarchical_Orchestration_Methodology.md` |
| Implementation Plan | `/mnt/raid0/llm/claude/research/Orchestrator_Implementation_Plan.md` |
| Results Summary | `/mnt/raid0/llm/claude/research/RESULTS_SUMMARY.md` |
| NeurIPS Tracks | `/mnt/raid0/llm/claude/research/neurips_2025_spec_decoding_tracks.md` |

### Progress Reports

| Report | Path |
|--------|------|
| Dec 16, 2025 | `/mnt/raid0/llm/claude/orchestration/progress/PROGRESS_2025-12-16.md` |
| Dec 21, 2025 | `/mnt/raid0/llm/claude/orchestration/progress/PROGRESS_2025-12-21.md` |
| Dec 22, 2025 | `/mnt/raid0/llm/claude/orchestration/progress/PROGRESS_2025-12-22.md` |
| Dec 24, 2025 | `/mnt/raid0/llm/claude/orchestration/progress/PROGRESS_2025-12-24.md` |
| Jan 1, 2026 | `/mnt/raid0/llm/claude/orchestration/progress/PROGRESS_2026-01-01.md` |

### Benchmark Infrastructure

| Component | Path |
|-----------|------|
| Prompt files | `/mnt/raid0/llm/claude/benchmarks/prompts/v1/` |
| Results | `/mnt/raid0/llm/claude/benchmarks/results/` |
| Reviews | `/mnt/raid0/llm/claude/benchmarks/results/reviews/` |
| Benchmark runner | `/mnt/raid0/llm/claude/scripts/benchmark/run_benchmark.py` |
| AMD PACE benchmark | `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py` |

### Orchestration

| Component | Path |
|-----------|------|
| TaskIR Schema | `/mnt/raid0/llm/claude/orchestration/task_ir.schema.json` |
| Model Registry | `/mnt/raid0/llm/claude/orchestration/model_registry.yaml` |
| Dispatcher | `/mnt/raid0/llm/claude/src/dispatcher.py` |
| Executor | `/mnt/raid0/llm/claude/src/executor.py` |

---

## Appendix B: Command Quick Reference

### Speculative Decoding (External Draft)
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative \
  -m TARGET.gguf \
  -md DRAFT.gguf \
  --draft-max 24 -t 96 -p "prompt"
```

### MoE Expert Reduction
```bash
numactl --interleave=all \
  llama-cli \
  -m MOE_MODEL.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

### Prompt Lookup
```bash
numactl --interleave=all \
  llama-cli \
  -m MODEL.gguf \
  --lookup-ngram-min 3 \
  -t 96 -f prompt_with_context.txt
```

### Run Benchmark Suite
```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all
```

### Run Gates
```bash
cd /mnt/raid0/llm/claude && make gates
```

---

*Document generated: 2026-01-04*
*Last comprehensive update: 2026-01-04*
