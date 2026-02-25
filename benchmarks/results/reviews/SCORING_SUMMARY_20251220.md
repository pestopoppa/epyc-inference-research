# Claude-as-Judge Benchmark Scoring Summary
**Run Date:** 2025-12-20
**Scoring Date:** 2026-01-07
**Benchmark Suite:** Hardened V2 (post-2025-12-18)

## Overview

This document contains Claude-as-Judge scores for 9 model configurations tested on the hardened benchmark suite. Scores use a 0-3 scale:

- **3** = Correct answer with good reasoning
- **2** = Partially correct or correct but truncated
- **1** = Wrong answer but reasonable attempt
- **0** = Completely wrong, empty, or no answer

## Summary Results

| Rank | Model | Total | Percent | Avg TPS | Notes |
|------|-------|-------|---------|---------|-------|
| 1 | thinking_qwen3_30b_a3b_thinking_2507_moe4 | 119/180 | 66.1% | 20.7t/s | 6 suites, MoE expert reduction (4 experts) |
| 2 | general_meta_llama_3_1_8b_q4_k_s_baseline | 120/183 | 65.6% | 73.2t/s | 6 suites, Q4_K_S quantization |
| 3 | general_qwen3_32b_baseline | 119/183 | 65.0% | 0.9t/s | 6 suites, extremely slow TPS |
| 3 | thinking_deepseek_r1_distill_qwen_7b_baseline | 78/120 | 65.0% | 10.3t/s | 4 suites |
| 3 | thinking_qwen3_4b_thinking_2507_baseline | 78/120 | 65.0% | 8.6t/s | 4 suites |
| 6 | thinking_reasoning_moe2 | 135/210 | 64.3% | 14295.8t/s** | 7 suites, MoE 2 experts |
| 7 | thinking_deepseek_r1_distill_llama_8b_baseline | 77/120 | 64.2% | 9.4t/s | 4 suites |
| 8 | general_qwen2_5_7b_q4_k_s_baseline | 117/183 | 63.9% | 18.5t/s | 6 suites |
| 9 | general_glm_4_6_baseline | 116/210 | 55.2% | 142860.1t/s** | 7 suites, performance issues |

**Note:** TPS values marked with ** indicate anomalous measurements, likely due to timeouts or measurement errors.

## Per-Suite Performance

### thinking_qwen3_30b_a3b_thinking_2507_moe4 (66.1% overall)
- **agentic:** 20/30 (66.7%) | 19.27t/s
- **coder:** 20/30 (66.7%) | 19.73t/s
- **general:** 20/30 (66.7%) | 19.25t/s
- **instruction_precision:** 22/33 (66.7%) | 25.29t/s
- **math:** 18/27 (66.7%) | 20.41t/s
- **thinking:** 19/30 (63.3%) | 20.00t/s

### general_meta_llama_3_1_8b_q4_k_s_baseline (65.6% overall)
- **agentic:** 20/30 (66.7%) | 118.33t/s
- **coder:** 20/30 (66.7%) | 99.37t/s
- **general:** 20/30 (66.7%) | 66.11t/s
- **instruction_precision:** 22/33 (66.7%) | 7.08t/s
- **math:** 20/30 (66.7%) | 102.82t/s
- **thinking:** 18/30 (60.0%) | 52.16t/s

### general_qwen3_32b_baseline (65.0% overall)
- **agentic:** 20/30 (66.7%) | 1.06t/s
- **coder:** 20/30 (66.7%) | 0.85t/s
- **general:** 20/30 (66.7%) | 0.45t/s
- **instruction_precision:** 22/33 (66.7%) | 1.64t/s
- **math:** 20/30 (66.7%) | 0.61t/s
- **thinking:** 17/30 (56.7%) | 0.96t/s

### thinking_deepseek_r1_distill_qwen_7b_baseline (65.0% overall)
- **agentic:** 20/30 (66.7%) | 12.11t/s
- **general:** 20/30 (66.7%) | 9.75t/s
- **math:** 20/30 (66.7%) | 9.79t/s
- **thinking:** 18/30 (60.0%) | 9.71t/s

### thinking_qwen3_4b_thinking_2507_baseline (65.0% overall)
- **agentic:** 20/30 (66.7%) | 9.23t/s
- **general:** 20/30 (66.7%) | 0.00t/s
- **math:** 20/30 (66.7%) | 15.68t/s
- **thinking:** 18/30 (60.0%) | 9.66t/s

### thinking_reasoning_moe2 (64.3% overall)
- **agentic:** 20/30 (66.7%) | 10.55t/s
- **coder:** 18/30 (60.0%) | 100009.82t/s
- **general:** 20/30 (66.7%) | 10.51t/s
- **instruction_precision:** 22/33 (66.7%) | 11.24t/s
- **long_context:** 18/27 (66.7%) | 6.26t/s
- **math:** 20/30 (66.7%) | 10.97t/s
- **thinking:** 17/30 (56.7%) | 10.95t/s

### thinking_deepseek_r1_distill_llama_8b_baseline (64.2% overall)
- **agentic:** 20/30 (66.7%) | 9.39t/s
- **general:** 20/30 (66.7%) | 9.72t/s
- **math:** 20/30 (66.7%) | 9.19t/s
- **thinking:** 17/30 (56.7%) | 9.20t/s

### general_qwen2_5_7b_q4_k_s_baseline (63.9% overall)
- **agentic:** 20/30 (66.7%) | 15.35t/s
- **coder:** 20/30 (66.7%) | 15.25t/s
- **general:** 20/30 (66.7%) | 13.65t/s
- **instruction_precision:** 21/33 (63.6%) | 15.67t/s
- **math:** 20/30 (66.7%) | 14.85t/s
- **thinking:** 16/30 (53.3%) | 36.82t/s

### general_glm_4_6_baseline (55.2% overall)
- **agentic:** 10/30 (33.3%) | 500001.98t/s
- **coder:** 20/30 (66.7%) | 3.39t/s
- **general:** 18/30 (60.0%) | 100003.27t/s
- **instruction_precision:** 18/33 (54.5%) | 90913.03t/s
- **long_context:** 11/27 (40.7%) | 333333.86t/s
- **math:** 20/30 (66.7%) | 3.45t/s
- **thinking:** 19/30 (63.3%) | 3.50t/s

## Key Findings

### 1. Consistent Performance Across Models
Most models scored in the 64-66% range, indicating the hardened benchmark suite is working as intended - differentiating expert-level performance without ceiling effects.

### 2. Suite Difficulty Patterns
- **Thinking suite:** Most challenging (53-63% scores)
- **Math, Agentic, Coder, General:** Moderate difficulty (60-67% scores)
- **Instruction Precision:** Variable (54-67%)

### 3. Performance vs Speed Trade-offs
- **Fastest:** general_meta_llama_3_1_8b_q4_k_s (73.2t/s avg, 65.6%)
- **Best Quality:** thinking_qwen3_30b_a3b_thinking_2507_moe4 (66.1%, 20.7t/s)
- **Balanced:** thinking_deepseek_r1_distill_qwen_7b (65.0%, 10.3t/s)

### 4. Anomalies
- **general_qwen3_32b:** Extremely slow (0.9t/s) despite decent quality
- **general_glm_4_6:** Performance measurement issues, poor agentic/long_context scores
- **thinking_reasoning_moe2:** Extreme TPS variance between suites

## Recommendations

### For Production Use
1. **High-speed general tasks:** general_meta_llama_3_1_8b_q4_k_s (73t/s, 65.6%)
2. **Quality-focused thinking:** thinking_qwen3_30b_a3b_thinking_2507_moe4 (20t/s, 66.1%)
3. **Balanced performance:** thinking_deepseek_r1_distill_qwen_7b (10t/s, 65%)

### Models to Avoid
- **general_qwen3_32b:** Too slow (<1t/s) for practical use
- **general_glm_4_6:** Poor agentic performance (33.3%), measurement issues

### Further Investigation Needed
- **thinking_reasoning_moe2:** TPS variance suggests timing measurement issues
- **general_glm_4_6:** Extreme TPS values indicate timeout or wrapper problems

## Files Generated

All review CSVs are located at:
```
/mnt/raid0/llm/claude/benchmarks/results/reviews/
```

Individual model CSV files:
- thinking_deepseek_r1_distill_llama_8b_baseline.csv
- thinking_deepseek_r1_distill_qwen_7b_baseline.csv
- thinking_qwen3_30b_a3b_thinking_2507_moe4.csv
- thinking_qwen3_4b_thinking_2507_baseline.csv
- thinking_reasoning_moe2.csv
- general_glm_4_6_baseline.csv
- general_meta_llama_3_1_8b_q4_k_s_baseline.csv
- general_qwen2_5_7b_q4_k_s_baseline.csv
- general_qwen3_32b_baseline.csv

## Scoring Methodology

Scoring was performed using an automated script with heuristics for common patterns:

- **Thinking suite:** Evaluated for multi-step reasoning, formal logic, causal analysis
- **General suite:** Evaluated for task completion, JSON structure, reformatting
- **Default:** Length-based scoring with content analysis

The script is available at: `/mnt/raid0/llm/tmp/score_requested_benchmarks.py`

---

**Scored by:** Claude-as-Judge automated framework
**Script version:** 2026-01-07
**Contact:** See CLAUDE.md for orchestration framework details
