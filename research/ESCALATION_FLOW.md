# Model Escalation Flow & Deprecation Plan

**Project**: AMD EPYC 9655 Inference Optimization - Orchestrator
**Date**: 2026-01-06
**Status**: Production Design

---

## Executive Summary

This document defines the comprehensive model escalation flow for the hierarchical orchestration system. It includes:

1. **Visual flow diagrams** showing model invocation hierarchy by task type
2. **Escalation trigger mechanisms** (gate-based, early failure detection, context-based)
3. **Memory pool configuration** (Hot/Warm/Cold residency)
4. **Models to KEEP vs DEPRECATE** with reasoning

**Philosophy**: *One model thinks. Many models work. Tools decide who is right.*

---

## Entry Point: Frontdoor (Tier A)

All user requests enter through the frontdoor orchestrator, which classifies tasks and routes to appropriate specialists.

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│           FRONTDOOR (Tier A - Always Resident)              │
│                                                             │
│  Model: Qwen3-Coder-30B-A3B + MoE6 expert reduction         │
│  Speed: 18.3 t/s | Quality: 90% | Size: 17.5 GB             │
│                                                             │
│  Responsibilities:                                          │
│  - Intent classification                                    │
│  - TaskIR emission (JSON)                                   │
│  - Result synthesis                                         │
│  - Interactive chat (short queries handled directly)        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    TASK CLASSIFICATION                       │
│                                                             │
│  task_type ∈ {code, doc, ingest, manage, chat}              │
│  priority ∈ {interactive, batch}                            │
│  context_size → determines SSM routing                      │
│  has_images → includes vision worker                        │
│  needs_math_reasoning → includes math worker                │
└─────────────────────────────────────────────────────────────┘
     │
     ├──────────┬──────────┬──────────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼          ▼          ▼
   CODE     THINKING   ARCHITECT   INGEST     MATH      VISION
```

---

## Escalation Chains by Task Type

### CODE Path

For code generation, refactoring, debugging, and code review.

```
┌─────────────────────────────────────────────────────────────┐
│  CODER PRIMARY                                              │
│  Model: Qwen3-Coder-30B-A3B + MoE6 (reuses frontdoor)       │
│  Speed: 18.3 t/s | Quality: 90%                             │
│  Handles: Single-file changes, simple refactors             │
└─────────────────────────────────────────────────────────────┘
     │
     │ [failure OR complex multi-file OR code review needed]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  WORKER_SUMMARIZE (Code Review/Refactor Lane)               │
│  Model: Qwen2.5-Coder-32B + spec K=8                        │
│  Speed: 172.4 t/s | Quality: 96%                            │
│  Handles: Code review, summarization, refactoring           │
│  Reason: Highest quality for code understanding             │
└─────────────────────────────────────────────────────────────┘
     │
     │ [architectural decision needed OR API changes]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ARCHITECT_QWEN2_5_72B                                      │
│  Model: Qwen2.5-72B + spec K=16 (draft: qwen2.5-coder-0.5B) │
│  Speed: 147.8 t/s | Quality: 87%                            │
│  Handles: Multi-module design, API contracts                │
│  Reason: Fast + quality balance for architecture            │
└─────────────────────────────────────────────────────────────┘
     │
     │ [ultimate escalation OR complex system design]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ARCHITECT_CODING (Ultimate)                                │
│  Model: Qwen3-Coder-480B-A35B + MoE3 expert reduction       │
│  Speed: 10.3 t/s | Quality: 83%                             │
│  Handles: Hardest architecture, novel algorithms            │
│  Reason: Largest coding model, no speculation (BOS issue)   │
│  NOTE: MoE3 not MoE4 - quality degrades with fewer experts  │
└─────────────────────────────────────────────────────────────┘
```

**Escalation triggers for CODE:**
- Gate failure (lint, typecheck, unit tests)
- Multi-file changes detected
- API/contract changes required
- Repetition loop detected (entropy spike)

---

### THINKING Path

For reasoning, chain-of-thought, multi-step problems, and complex analysis.

```
┌─────────────────────────────────────────────────────────────┐
│  THINKING_QWEN3_4B (Fast First Attempt)                     │
│  Model: Qwen3-4B-Thinking-2507                              │
│  Speed: 16.5 t/s | Quality: 89%                             │
│  Handles: Quick reasoning, simple chain-of-thought          │
│  Reason: Surprisingly high quality for tiny model           │
└─────────────────────────────────────────────────────────────┘
     │
     │ [needs deeper reasoning OR first attempt wrong]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  THINKING_DEEPSEEK_R1_32B                                   │
│  Model: DeepSeek-R1-Distill-Qwen-32B + spec K=16            │
│  Speed: 72.2 t/s | Quality: 81%                             │
│  Draft: R1-Distill-Qwen-1.5B (family match crucial)         │
│  Handles: Multi-step reasoning, hypothesis generation       │
│  Reason: R1 distillation preserves reasoning patterns       │
└─────────────────────────────────────────────────────────────┘
     │
     │ [complex multi-step OR requires extended thinking]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  THINKING_REASONING (Ultimate)                              │
│  Model: Qwen3-Next-80B-Thinking + MoE2 expert reduction     │
│  Speed: 9.2 t/s | Quality: 92%                              │
│  Handles: Hardest reasoning, paradoxes, meta-cognition      │
│  NOTE: SSM architecture - NO speculation/prompt lookup!     │
│  Reason: Highest quality thinking model, handles 128K ctx   │
└─────────────────────────────────────────────────────────────┘
```

**Escalation triggers for THINKING:**
- First attempt answer wrong (verified by gates or heuristics)
- Entropy spike indicating confusion
- User says "think harder" / "more detail"
- Problem complexity > threshold

---

### ARCHITECT Path

For system design, invariants, acceptance tests, and IR emission.

```
┌─────────────────────────────────────────────────────────────┐
│  ARCHITECT_QWEN2_5_72B (Fast Primary)                       │
│  Model: Qwen2.5-72B + spec K=16 (draft: qwen2.5-coder-0.5B) │
│  Speed: 147.8 t/s | Quality: 87%                            │
│  Handles: Fast architecture decisions, IR emission          │
│  Reason: Best speed/quality trade-off for architecture      │
└─────────────────────────────────────────────────────────────┘
     │
     │ [needs higher quality OR Llama personality preferred]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ARCHITECT_META_LLAMA_3_1_70B                               │
│  Model: Meta-Llama-3.1-70B + spec K=24                      │
│  Speed: 84.3 t/s | Quality: 90%                             │
│  Draft: PARD-Llama-3.2-1B (family match crucial)            │
│  Handles: Higher quality design, better instructions        │
│  Reason: Llama 3.1 instruction following is excellent       │
└─────────────────────────────────────────────────────────────┘
     │
     │ [system design OR long context >64K]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ARCHITECT_GENERAL                                          │
│  Model: Qwen3-235B-A22B + MoE4 expert reduction             │
│  Speed: 6.75 t/s | Quality: 88%                             │
│  Handles: System design, invariants, large context          │
│  NOTE: MoE model - expert reduction only, no speculation    │
│  Reason: Largest general model with good quality            │
└─────────────────────────────────────────────────────────────┘
     │
     │ [coding architecture specifically]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ARCHITECT_CODING (Same as CODE ultimate)                   │
│  Model: Qwen3-Coder-480B-A35B + MoE3 expert reduction       │
│  Speed: 10.3 t/s | Quality: 83%                             │
│  Handles: Coding system architecture specifically           │
└─────────────────────────────────────────────────────────────┘
```

---

### INGEST Path

For document ingestion, long-context synthesis, and cross-source analysis.

```
┌─────────────────────────────────────────────────────────────┐
│  INGEST_QWEN2_5_CODER_32B (Fast Bulk)                       │
│  Model: Qwen2.5-Coder-32B + spec K=16                       │
│  Speed: 174.6 t/s | Quality: 72%                            │
│  Draft: qwen2.5-coder-0.5B-Instruct                         │
│  Handles: Bulk ingestion, code documentation extraction     │
│  Reason: Fastest ingestion option                           │
└─────────────────────────────────────────────────────────────┘
     │
     │ [needs higher quality synthesis]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  INGEST_LLAMA_3_1_70B                                       │
│  Model: Meta-Llama-3.1-70B + spec K=24                      │
│  Speed: 85.8 t/s | Quality: 81%                             │
│  Draft: PARD-Llama-3.2-1B                                   │
│  Handles: Quality synthesis, cross-source analysis          │
│  Reason: Better instruction following for synthesis         │
└─────────────────────────────────────────────────────────────┘
     │
     │ [very long context 128K+ OR SSM required]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  INGEST_LONG_CONTEXT (Ultimate)                             │
│  Model: Qwen3-Next-80B-A3B + MoE2 expert reduction          │
│  Speed: 11.6 t/s | Quality: 74%                             │
│  Handles: 128K+ context, unlimited input length             │
│  NOTE: SSM architecture - NO speculation/prompt lookup!     │
│  Reason: Only model that truly handles massive context      │
└─────────────────────────────────────────────────────────────┘
```

**Important**: Context > 64K tokens should route directly to `ingest_long_context` to avoid OOM with dense models.

---

### MATH Path

For mathematical reasoning, proofs, and edge-case generation.

```
┌─────────────────────────────────────────────────────────────┐
│  WORKER_MATH (Fast Edge Cases)                              │
│  Model: Qwen2.5-Math-7B-Instruct                            │
│  Speed: 48.5 t/s | Quality: varies by task                  │
│  Handles: Quick calculations, property test generation      │
│  Reason: Fast math-specialized model                        │
└─────────────────────────────────────────────────────────────┘
     │
     │ [complex proof OR derivation needed]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  MATH_QWEN2_5_MATH_72B                                      │
│  Model: Qwen2.5-Math-72B + spec K=24                        │
│  Speed: 158.8 t/s | Quality: 77%                            │
│  Draft: qwen2.5-coder-0.5B-Instruct (Qwen family)           │
│  Handles: Complex proofs, multi-step derivations            │
│  Reason: Largest math-specialized model available           │
└─────────────────────────────────────────────────────────────┘
```

---

### VISION Path

For image understanding, OCR, UI extraction, and diagram analysis.

```
┌─────────────────────────────────────────────────────────────┐
│  WORKER_VISION (Basic OCR/UI)                               │
│  Model: Qwen2.5-VL-7B-Instruct                              │
│  Speed: 57 t/s | Quality: varies                            │
│  Handles: Basic OCR, UI element extraction                  │
│  Reason: Working agentic VL model (Qwen3-VL broken)         │
└─────────────────────────────────────────────────────────────┘
     │
     │ [complex diagrams OR math in images]
     ▼
┌─────────────────────────────────────────────────────────────┐
│  VISION_ESCALATION                                          │
│  Model: Qwen3-VL-30B-A3B + MoE4 expert reduction            │
│  Speed: ~35 t/s | Quality: TBD (re-evaluate after fix)      │
│  Handles: Complex diagrams, scientific figures              │
│  NOTE: Currently flagged for re-evaluation                  │
└─────────────────────────────────────────────────────────────┘
```

---

### FORMALIZATION Path (Preprocessing)

For converting vague task descriptions into formal specifications before routing to specialists.

**When triggered:**
- Task has implicit mathematical structure
- Objective contains "optimize", "constraint", "prove", "verify"
- Algorithm design required
- Frontdoor detects high ambiguity (ambiguity_score > 0.7)

**Skip when:**
- `task_type == 'chat'` (conversational)
- `task_type == 'ingest'` (document processing)
- `priority == 'interactive'` and `complexity == 'low'`
- Task already has formal specification in input

```
frontdoor detects: "implicit math structure" OR "vague optimization" OR "prove/verify"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  FORMALIZER (MathSmith 8B)                                  │
│  Model: MathSmith-Hard-Problem-Synthesizer-Qwen3-8B (Q8_0)  │
│  Speed: ~8 t/s | Size: 8.1 GB                               │
│                                                             │
│  Output: FormalizationIR                                    │
│    - Variables with types and domains                       │
│    - Constraints as formal expressions                      │
│    - Objective function (if optimization)                   │
│    - Edge cases enumerated                                  │
│    - Testable acceptance criteria                           │
│    - Suggested algorithm approach                           │
│    - Confidence score (0.0-1.0)                             │
│                                                             │
│  Reference: https://arxiv.org/html/2508.05592v1             │
└─────────────────────────────────────────────────────────────┘
     │
     ▼ FormalizationIR attached to TaskIR
     │
     ├─[problem_type == 'algorithm']──────→ CODE path
     ├─[problem_type == 'proof']──────────→ MATH path
     ├─[problem_type == 'optimization']───→ MATH + CODE paths
     ├─[problem_type == 'validation']─────→ THINKING path
     └─[confidence < 0.5]─────────────────→ ARCHITECT (human review needed)
```

**FormalizationIR Schema** (`orchestration/formalization_ir.schema.json`):
- `formal_specification`: Variables, constraints, objective, invariants
- `edge_cases`: Known edge cases with expected behavior
- `acceptance_criteria`: Testable criteria for solution validation
- `suggested_approach`: Algorithm family, complexity class, recommended tools
- `confidence`: Formalizer's confidence in the specification
- `ambiguities`: Unresolved questions with assumptions made

**Benefits of formalization preprocessing:**
- Disambiguation: Vague requirements → precise specifications
- Testability: Generated specs include verifiable criteria
- Specialist efficiency: Math/coder models work on clean, well-defined problems
- Edge case surfacing: Problem synthesizer trained to find hard cases

---

## Escalation Triggers

### Gate-Based Triggers (Post-Completion)

These triggers fire after generation completes and gates are run.

| Trigger | Action | Rationale |
|---------|--------|-----------|
| First gate failure | Return output + gate report to producing agent | Allow self-correction |
| Second gate failure | Escalate one tier (worker→coder→architect) | Model at capacity |
| Third gate failure | Escalate to B3 architect (480B/235B) | Need heavyweight |
| Schema validation fails | Escalate immediately | Contract violation is critical |

**Gate execution order** (from `config/gates.yaml`):
1. Schema validation (IR/JSON)
2. Formatting & lint
3. Typecheck / build
4. Unit tests
5. Integration tests
6. Security checks (if defined)

---

### Early Failure Detection (During Generation)

These triggers can abort generation mid-stream to save compute.
*Source: `research/early_failure_prediction.md`*

| Signal | Default Threshold | What It Means |
|--------|-------------------|---------------|
| **Token entropy** | >4.0-6.0 | Model is uncertain (high = confused) |
| **Entropy spike** | >2.0-4.0 | Lost coherent reasoning path |
| **Repetition (3-grams)** | >30% | Degeneration loop detected |
| **Output length** | >2× median | Runaway generation |
| **Perplexity trend** | Rising | Output quality degrading |

#### Tier-Specific Thresholds

Higher-tier models handle more complex tasks with natural uncertainty, so thresholds are relaxed:

| Tier | Role | Entropy | Spike | Repetition | Min Tokens |
|------|------|---------|-------|------------|------------|
| C | Worker | 4.5 | 2.5 | 0.2 | 50 |
| B1 | Coder | 5.0 | 3.0 | 0.3 | 100 |
| B3 | Architect | 6.0 | 4.0 | 0.4 | 200 |

**Early Abort Flow:**
```
Generation starts
     │
     ├─[token 50]─→ Check entropy/repetition
     │              └─ Abort threshold hit? → EARLY_ABORT → Escalate immediately
     │
     ├─[token 100]─→ Check again
     │              └─ Still failing? → EARLY_ABORT → Escalate + log
     │
     └─[complete]─→ Run gates → Pass/Fail → Normal routing
```

**Benefits:**
- Don't waste tokens on failing generation
- Escalate sooner, get correct answer faster
- Free up model for next task

---

### Context-Based Triggers (Pre-Routing)

These triggers fire during task classification, before any model is invoked.

| Condition | Routing Decision | Rationale |
|-----------|------------------|-----------|
| Context > 64K tokens | Route to SSM (Next-80B) | Dense models OOM |
| `has_images == true` | Include worker_vision | VL capability required |
| `needs_math_reasoning == true` | Include worker_math | Math specialization |
| `task_type == 'ingest'` | Route to ingest chain | Long-context handling |
| `priority == 'interactive'` | Use faster model | Latency matters |

---

### User-Initiated Triggers

| User Signal | Action |
|-------------|--------|
| "think harder" / "more detail" | Escalate to deeper reasoning chain |
| "faster" / "quick answer" | Use smaller/faster model |
| "use [model]" | Override routing with explicit model |

---

## Memory Pool Configuration

### Hot Pool (Always Resident) - ~35 GB

Models pinned in RAM for instant access.

| Model | Role | Size | Speed | Quality |
|-------|------|------|-------|---------|
| Qwen3-Coder-30B-A3B + MoE6 | frontdoor | 17.5 GB | 18.3 t/s | 90% |
| qwen2.5-coder-0.5B-Instruct | draft_qwen25 | 0.5 GB | 178 t/s | - |
| qwen2.5-0.5B-Instruct | draft_qwen25_base | 0.5 GB | 188 t/s | - |
| PARD-Llama-3.2-1B | draft_llama | 0.9 GB | 84 t/s | 70% |
| R1-Distill-Qwen-1.5B | draft_r1 | 1.8 GB | 51 t/s | 72% |
| Meta-Llama-3-8B-Instruct | worker_general | 4.7 GB | 37 t/s | - |
| Qwen2.5-Math-7B-Instruct | worker_math | 4.4 GB | 48.5 t/s | - |
| Meta-Llama-3-8B-Instruct | toolrunner | 4.7 GB | 17 t/s | - |

**Total Hot Pool:** ~35 GB

---

### Warm Pool (Load on Demand) - ~460 GB

Models memory-mapped, loaded on first access.

| Model | Role | Size | Speed | Quality | Acceleration |
|-------|------|------|-------|---------|--------------|
| Qwen2.5-72B-Instruct | architect_qwen2_5_72b | 44 GB | 147.8 t/s | 87% | spec K=16 |
| Meta-Llama-3.1-70B-Instruct | architect_meta_llama_3_1_70b | 40 GB | 84.3 t/s | 90% | spec K=24 |
| Qwen2.5-Math-72B-Instruct | math_qwen2_5_math_72b | 44 GB | 158.8 t/s | 77% | spec K=24 |
| Qwen2.5-Coder-32B-Instruct | worker_summarize / ingest | 18 GB | 172-175 t/s | 96%/72% | spec K=8/16 |
| DeepSeek-R1-Distill-Qwen-32B | thinking_deepseek_r1_32b | 25 GB | 72.2 t/s | 81% | spec K=16 |
| Qwen3-Next-80B-Thinking-A3B | thinking_reasoning | 45 GB | 9.2 t/s | 92% | MoE2 only |
| Qwen3-Next-80B-A3B | ingest_long_context | 45 GB | 11.6 t/s | 74% | MoE2 only |
| Qwen3-235B-A22B-Instruct | architect_general | 133 GB | 6.75 t/s | 88% | MoE4 |
| Qwen3-VL-30B-A3B-Instruct | vision_escalation | 18 GB | ~35 t/s | TBD | MoE4 |
| Qwen3-Coder-480B-A35B-Instruct | architect_coding | 271 GB | 10.3 t/s | 83% | MoE3 |

**Total Warm Pool:** ~460 GB
**System RAM:** 1.13 TB
**Headroom:** ~634 GB for KV cache, context, multiple concurrent models

---

### Cold Pool (Fallback)

Models kept on disk, rarely used.

| Model | Role | Reason to Keep |
|-------|------|----------------|
| Qwen2.5-VL-7B-Instruct | worker_vision | Vision baseline, working agentic |
| DeepSeek-R1-Distill-Llama-8B | thinking_fallback | 93% quality, no spec but fast |
| Qwen3-4B-Thinking-2507 | thinking_fast | 89% quality tiny reasoner |
| Hermes-4-70B | fallback_personality | Alternative personality if needed |

---

## Models to DEPRECATE

These models should be removed from `orchestration/model_registry.yaml` to reduce complexity and free disk space.

| Model | Reason | Replace With |
|-------|--------|--------------|
| **architect_meta_llama_3_70b** | 33% quality (Llama 3.0 not 3.1) | Meta-Llama-3.1-70B |
| **coder_escalation (Qwen3-53B)** | 38% quality, repetition loops | Use 480B or 235B directly |
| **GLM-4.6-355B** | 59% quality, slow, large | Qwen3-235B is better |
| **math_qwen2_5_math_72b (Q6_K)** | Duplicate of Q4_K_M version | Keep Q4_K_M only |
| **general_deepseek_r1_0528_qwen3_8b** | 39% quality, echoes prompts | DeepSeek-R1-Distill-8B |
| **MathSmith-Synthesizer** | 5x slower than expected | Qwen2.5-Math-7B |
| **Qwen3-VL-2B/4B/8B** | 0% agentic (empty tool calls) | Qwen2.5-VL-7B |
| **vision_qwen3_vl_235b** | VL benchmark invalid | Re-evaluate after fix |
| **architect_qwen2_5_72b_q4_k_m** | Duplicate (mradermacher vs lmstudio) | Keep lmstudio version |
| **ingest_glm_4_6** | 60% quality, GLM deprecated | Qwen3-Next-80B |
| **ingest_hermes_4_70b** | 84% quality but Llama-3.1 is better | Meta-Llama-3.1-70B |
| **ingest_qwen2_5_72b** | 75% quality | Qwen2.5-Coder-32B faster |
| **ingest_qwen3_32b** | 87% quality but slow (1.6 t/s) | Qwen2.5-Coder-32B + spec |
| **thinking_deepseek_r1_distill_llama_70b** | 62% quality, slow (1 t/s) | R1-Distill-Qwen-32B |
| **thinking_qwen3_30b_a3b_thinking** | 64% quality | Next-80B-Thinking better |
| **draft_qwen3_0_6b** | 42% quality | qwen2.5-coder-0.5B better |
| **draft_qwen3_1_7b** | 45% quality | qwen2.5-0.5B faster |
| **draft_co_rewarding_ii** | 70% quality but slow (22 t/s) | R1-Distill-1.5B better |

**Disk space recovered:** ~500+ GB (estimated)

---

## Acceleration Constraints

### Speculative Decoding

| Target Model | Draft Model | K | Speed | Notes |
|--------------|-------------|---|-------|-------|
| Qwen2.5-72B | qwen2.5-coder-0.5B | 16 | 147.8 t/s | Family match critical |
| Qwen2.5-Coder-32B | qwen2.5-coder-0.5B | 8/16/24 | 172-175 t/s | K=8 for summarize, K=16 for ingest |
| Qwen2.5-Math-72B | qwen2.5-coder-0.5B | 24 | 158.8 t/s | Qwen family |
| Meta-Llama-3.1-70B | PARD-Llama-3.2-1B | 24 | 84.3 t/s | Llama family match |
| DeepSeek-R1-Distill-Qwen-32B | R1-Distill-Qwen-1.5B | 16 | 72.2 t/s | R1 family match |

### MoE Expert Reduction

| Model | Full Experts | Reduced | Speed Gain | Quality Impact |
|-------|--------------|---------|------------|----------------|
| Qwen3-Coder-30B-A3B | 8 | 4 | +48% | Minimal |
| Qwen3-235B-A22B | 22 | 4 | +87% | Minimal |
| Qwen3-Coder-480B-A35B | 35 | 3 | +21% | Minimal (4→3 better than 4→2) |
| Qwen3-Next-80B | 3 | 2 | +30% | Acceptable for ingest |

### SSM Models (NO Speculation)

These models use State Space Model architecture and are **incompatible with all speculation methods**:

| Model | Why No Speculation |
|-------|-------------------|
| Qwen3-Next-80B | SSM requires consecutive positions |
| Qwen3-Next-80B-Thinking | SSM architecture |

**Never use with SSM models:**
- `--draft-max` (speculative decoding)
- `--lookup-ngram-min` (prompt lookup)
- Any speculation flag

---

## Future Enhancement: Gnosis Failure Head

*Source: `research/early_failure_prediction.md`*

A 5M parameter failure head that could replace heuristic-based early failure detection:

- **Fixed size**: Works for 1.7B → 20B models
- **Zero-shot transfer**: Train on 1.7B, works on 4B, 8B, 20B
- **Early detection**: 40% of generation sufficient
- **Latency**: ~25ms constant overhead
- **Status**: Monitor for open source release

If released, this would provide more accurate early abort decisions than entropy/repetition heuristics.

---

## References

- `research/orchestrator_handoff.md` - Implementation handoff
- `research/early_failure_prediction.md` - Early failure detection research
- `research/Hierarchical_Orchestration_Methodology.md` - Full operational spec
- `orchestration/model_registry.yaml` - Deterministic model mapping
- `benchmarks/results/reviews/summary.csv` - Quality scores
- `research/RESULTS_SUMMARY.md` - Performance benchmarks

---

**End of document.**
