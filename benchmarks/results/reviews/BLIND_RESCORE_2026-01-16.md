# Claude-as-Judge Blind Rescore - 2026-01-16

## Overview

This document contains the results of a comprehensive blind rescore of ALL model benchmark responses from `/workspace/benchmarks/results/runs/20251220_214317/`. Scoring was performed against reference answers from the YAML files in `/workspace/benchmarks/prompts/v1/`.

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| **3** | Correct answer with good reasoning |
| **2** | Partially correct or correct but truncated |
| **1** | Wrong answer but reasonable attempt |
| **0** | Completely wrong, empty, or no answer |

---

## Summary Tables

### Thinking Models (12 models scored)

| Model | Score | Pct | Notes |
|-------|-------|-----|-------|
| Qwen3-Next-80B-A3B-Thinking-Q4_K_S | 30/33 | **91%** | Best overall - excellent math (24/27) |
| DeepSeek-R1-Distill-Qwen-32B-Q6_K | 23/27 | **85%** | Excellent thinking quality, very slow |
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_M | 29/36 | **81%** | Strong general task performance |
| Qwen3-30B-A3B-Thinking-2507-Q8_0 | 19/24 | **79%** | Good with truncation issues |
| Phi-4-reasoning-plus-Q4_K_M | 9/12 | **75%** | Good thinking quality |
| Qwen3-4B-Thinking-2507-Q8_0 | 17/24 | **71%** | Very verbose |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 21/30 | **70%** | Verbose but functional |
| DeepSeek-R1-Distill-Llama-70B-Q4_K_M | 24/36 | **67%** | Many truncations |
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 17/30 | **57%** | Format compliance issues |

### Draft Models (22 models scored)

| Model | Score | Pct | Avg TPS | Notes |
|-------|-------|-----|---------|-------|
| gemma-3-1b-it-Q8_0 | 57/60 | **95%** | 114.1 | Best quality with good speed |
| Qwen3-1.7B-Q8_0 | 57/60 | **95%** | 36.3 | Highest quality, slower |
| PARD-DeepSeek-R1-Distill-Qwen-1.5B.Q5_K_S | 56/60 | **93%** | 45.6 | Excellent with good speed |
| Qwen3-1.7B-Q4_K_M | 54/60 | **90%** | 43.3 | Strong performer |
| PARD-DeepSeek-R1-Distill-Qwen-1.5B.Q8_0 | 52/60 | **87%** | 47.2 | Good quality |
| DeepSeek-R1-Distill-Qwen-1.5B-Q8_0 | 50/60 | **83%** | 59.1 | Solid performance |
| Qwen_Qwen3-0.6B-Q8_0 | 49/60 | **82%** | 67.8 | Good balance |
| PARD-Llama-3.2-1B.Q8_0 | 48/60 | **80%** | 41.7 | Good quality |
| pard-qwen3-0.6b-q4_0 | 47/60 | **78%** | 81.6 | Speed optimized |
| Qwen2.5-0.5B.Q8_0 | 45/60 | **75%** | 156.8 | Fastest, reasonable quality |
| Qwen2.5-Coder-0.5B-Q4_K_M | 42/60 | **70%** | 142.2 | High speed |
| Qwen3-0.6B-Q2_K | 40/60 | **67%** | 95.3 | Lower quality |

### Architect/Ingest Models (15 models scored)

| Model | Score | Pct | Notes |
|-------|-------|-----|-------|
| Qwen3-30B-A3B-Thinking-2507-Q8_0 (ingest) | 161/183 | **88.0%** | Best ingest model |
| Qwen3-235B-A22B-Q4_K_M | 175/201 | **87.1%** | Strong architect |
| Meta-Llama-3.1-70B-Instruct-Q4_K_M | 158/183 | **86.3%** | Good performance |
| Qwen3-32B-Q4_K_M (ingest) | 158/183 | **86.3%** | Solid ingest |
| Qwen3-Next-80B-A3B-Instruct-Q4_K_M | 157/183 | **85.8%** | Long context capable |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (ingest) | 180/210 | **85.7%** | Code-focused ingest |
| Hermes-4-70B-Q4_K_M | 156/183 | **85.2%** | Hermes variant |
| Qwen2.5-Coder-32B-Instruct-Q4_K_M (ingest) | 156/183 | **85.2%** | Code ingest |
| Qwen2.5-72B-Instruct-Q4_K_M | 154/198 | **77.8%** | Some issues |
| Qwen3-Coder-480B-A35B-Instruct-Q4_K_M | 162/210 | **77.1%** | Coding architect |

### Coder/Tool/Vision Models (19 models scored)

| Model | Score | Pct | Notes |
|-------|-------|-----|-------|
| MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M | 90/90 | **100.0%** | Perfect on tested suites |
| xLAM-2-1B-fc-r-Q4_K_M | 30/30 | **100.0%** | Perfect on general |
| MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q8_0 | 88/90 | **97.8%** | Near-perfect |
| xLAM-1b-fc-r.Q4_K_M | 29/30 | **96.7%** | Excellent |
| Qwen3-Coder-53B-A3B-TOTAL-RECALL-v2-Q4_K_M | 166/183 | **90.7%** | 100% on coder suite |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (frontdoor) | 188/210 | **89.5%** | Strong routing model |
| Qwen2.5-Math-72B-Instruct-Q4_K_M | 163/183 | **89.1%** | Good math |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (coder_primary) | 187/210 | **89.0%** | Primary coder |
| Qwen3_VL_2B.Q4_K_M | 73/90 | **81.1%** | Best vision |
| Qwen3-VL-30B-A3B-Instruct-Q4_K_M | 41/90 | **45.6%** | Severe issues |

### General/Worker Models (9 models scored)

| Model | Score | Pct | Notes |
|-------|-------|-----|-------|
| gemma-3-12b-it-Q4_K_M | ~80% | — | Strong general model |
| gemma-3-27B-it-QAT-Q4_0 | ~77% | — | Some algorithmic issues |
| Meta-Llama-3-8B-Instruct-fp16 | ~75% | — | Verbose but good |
| Qwen2.5-Coder-32B-Instruct-Q4_K_M (worker_summarize) | ~100% | — | Excellent on tested suites |
| Meta-Llama-3-8B-Instruct-Q4_K_M (worker_general) | ~70% | — | Decent with repetition |
| Qwen2.5-7B.Q4_K_S | ~55% | — | Degenerates on hard questions |
| Qwen2.5-VL-7B-Instruct-Q4_K_M (worker_vision) | ~52% | — | Vision-only (as expected) |
| Qwen2.5-Math-7B-Instruct-Q4_K_M (worker_math) | ~33% | — | Struggles with non-math |
| Meta-Llama-3.1-8B.Q4_K_S | **~6%** | — | **SEVERE**: Repetitive degeneration |

---

## Drift Analysis

### Comparison with Previous Scores (summary.csv)

| Model | Previous | New Blind Rescore | Delta | Notes |
|-------|----------|-------------------|-------|-------|
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_M | 87% | 81% | -6% | More conservative scoring |
| Qwen3-30B-A3B-Thinking-2507-Q8_0 | 93%+ | 79% | -14% | Truncation penalty |
| gemma-3-12b-it-Q4_K_M | 97% | ~80% | -17% | Stricter on general suite |
| Meta-Llama-3.1-70B-Instruct-Q4_K_M | 93% | 86% | -7% | Slightly stricter |
| Vision models | Variable | 45-81% | — | Now more consistent |

### Key Drift Observations

1. **Thinking Models Scored Lower**: Previous scores likely didn't penalize token truncation as heavily. Many thinking models hit 2047 token limit during `<think>` reasoning.

2. **Instruction Precision Consistently Low**: All models scored 48-67% on instruction_precision suite. This wasn't always captured in previous reviews.

3. **Vision Model Agentic Failures**: Vision models consistently score 33% on agentic suite (echoing prompts instead of JSON). This is now explicitly tracked.

4. **Meta-Llama-3.1-8B.Q4_K_S Critical Issue**: This model has severe generation issues (repetitive patterns, prompt echoing). Previous scores may have been inflated.

---

## Critical Issues Discovered

### 1. Meta-Llama-3.1-8B.Q4_K_S (~6%)
**Severity: CRITICAL**
- Model frequently echoes prompts without answering
- Degenerates into repetitive patterns ("No assertions. No assertions...")
- Should NOT be used in production

### 2. Token Truncation (Affects Many Thinking Models)
**Severity: MODERATE**
- 2047 token limit cuts off reasoning mid-thought
- Good reasoning often truncated before conclusion
- Consider increasing token limits for thinking models

### 3. Vision Model Agentic Failures (33%)
**Severity: EXPECTED**
- Vision models don't generate tool call JSON
- Output `/image` interface prompts for text-only tasks
- Not a bug - specialized behavior

### 4. Instruction Precision Suite (54-67% typical)
**Severity: MODERATE**
- All models struggle with strict format constraints
- Resisting elaboration and word limits are hardest
- Self-referential constraints almost never solved

---

## Scoring Methodology

### Reference Files Used
```
/workspace/benchmarks/prompts/v1/
├── thinking.yaml        (10 questions)
├── general.yaml         (10 questions)
├── math.yaml            (10 questions)
├── agentic.yaml         (10 questions)
├── coder.yaml           (10 questions)
├── instruction_precision.yaml (11 questions)
├── vl.yaml              (10 questions)
└── long_context.yaml    (9 questions)
```

### Scoring Agents Used
- **Agent a611c09**: Thinking model baselines (12 models)
- **Agent af520cd**: Draft model baselines (22 models)
- **Agent a38f1f6**: General/worker model baselines (9 models)
- **Agent ab5f718**: Architect/ingest model baselines (15 models)
- **Agent a5efe3b**: Coder/tool/vision model baselines (19 models)

### Output Files Created
```
/workspace/benchmarks/results/reviews/
├── thinking_models_comprehensive_scores.csv
├── thinking_models_summary.csv
├── general_worker_scores_jan2026.csv
├── architect_ingest_baseline_scores.csv (944 rows)
└── BLIND_RESCORE_2026-01-16.md (this file)
```

---

## Top Performers by Category

| Category | Model | Score |
|----------|-------|-------|
| **Thinking** | Qwen3-Next-80B-A3B-Thinking-Q4_K_S | 91% |
| **Draft (Quality)** | gemma-3-1b-it-Q8_0 | 95% |
| **Draft (Speed)** | Qwen2.5-0.5B.Q8_0 | 75% @ 157 t/s |
| **Architect** | Qwen3-235B-A22B-Q4_K_M | 87% |
| **Ingest** | Qwen3-30B-A3B-Thinking-2507-Q8_0 | 88% |
| **Coder** | Qwen3-Coder-53B-A3B-TOTAL-RECALL-v2-Q4_K_M | 91% |
| **Formalizer** | MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M | 100% |
| **Tool Calling** | xLAM-2-1B-fc-r-Q4_K_M | 100% |
| **Vision** | Qwen3_VL_2B.Q4_K_M | 81% |

### Models to Avoid

| Model | Reason |
|-------|--------|
| Meta-Llama-3.1-8B.Q4_K_S | Severe degeneration (~6%) |
| Qwen2.5-Coder-1.5B.Q2_K | 67% score |
| Qwen3-0.6B-Q2_K | 67% score |
| Qwen3-VL-30B-A3B-Instruct-Q4_K_M | 46% score |

---

## Appendix: Full CSV Locations

1. **Thinking Models**: `/workspace/benchmarks/results/reviews/thinking_models_comprehensive_scores.csv`
2. **Draft Models**: Output in agent af520cd task output
3. **General/Worker**: `/workspace/benchmarks/results/reviews/general_worker_scores_jan2026.csv`
4. **Architect/Ingest**: `/workspace/benchmarks/results/reviews/architect_ingest_baseline_scores.csv`
5. **Coder/Tool/Vision**: Output in agent a5efe3b task output

---

*Generated: 2026-01-18*
*Total Models Scored: 77 baseline configurations*
*Total Questions Evaluated: ~2,000+ individual responses*
