# Claude-as-Judge Scoring Summary
## Date: 2026-01-06
## Scorer: Claude Sonnet 4.5

### Files Scored

This summary covers 11 benchmark result files from the `20251217_160429` run:

1. `draft_qwen25_baseline.json`
2. `draft_qwen2_5_coder_0_5b_baseline.json`
3. `draft_qwen2_5_coder_1_5b_q2_k_baseline.json`
4. `draft_qwen2_5_coder_1_5b_q4_k_m_baseline.json`
5. `draft_qwen25_coder_baseline.json`
6. `draft_qwen2_5_math_1_5b_baseline.json`
7. `draft_qwen2_5_math_1_5b_q6k_baseline.json`
8. `draft_qwen3_0_6b_baseline.json`
9. `draft_qwen3_1_7b_baseline.json`
10. `draft_qwen3_vl_1b_merged_q8_0_baseline.json`
11. `general_deepseek_r1_0528_qwen3_8b_baseline.json`

### Scoring Results

| Model | Questions Scored | Total Score | Max Score | Percentage |
|-------|------------------|-------------|-----------|------------|
| draft_qwen25 | 40 | 54 | 120 | 45.0% |
| draft_qwen2_5_coder_0_5b | 40 | 53 | 120 | 44.2% |
| draft_qwen2_5_coder_1_5b_q2_k | 40 | 52 | 120 | 43.3% |
| draft_qwen2_5_coder_1_5b_q4_k_m | 40 | 54 | 120 | 45.0% |
| draft_qwen25_coder | 40 | 53 | 120 | 44.2% |
| draft_qwen2_5_math_1_5b | 40 | 52 | 120 | 43.3% |
| draft_qwen2_5_math_1_5b_q6k | 40 | 53 | 120 | 44.2% |
| draft_qwen3_0_6b | 40 | 50 | 120 | 41.7% |
| draft_qwen3_1_7b | 40 | 54 | 120 | 45.0% |
| draft_qwen3_vl_1b_merged_q8_0 | 2 | 2 | 6 | 33.3% |
| general_deepseek_r1_0528_qwen3_8b | 61 | 72 | 183 | 39.3% |

### Key Observations

#### Draft Models (0.5B - 1.7B)
- **Average Score**: 43.5% across draft models (excluding VL model)
- **Performance**: All draft models scored between 41.7% and 45.0%
- **Quality Issues**:
  - Many responses just echo the prompt without answering
  - Frequent repetitive/garbage text generation
  - Some correct responses on simple tasks (JSON parsing, email extraction)
  - T3 (post-doctoral) questions mostly scored 0-1 (attempted but incorrect)

#### Vision Model
- `draft_qwen3_vl_1b_merged_q8_0` only had 2 questions in the benchmark
- Scored 2/6 (33.3%) - lowest score
- Likely not tested on full benchmark suite

#### General Model (8B)
- `general_deepseek_r1_0528_qwen3_8b` tested on 61 questions (more suites than draft models)
- Scored 39.3% - surprisingly lower than draft models
- This may indicate:
  - Different question distribution (more difficult questions)
  - Model-specific weaknesses
  - Different test conditions

### Scoring Methodology

**Scoring Scale (0-3):**
- **3**: Correct answer with good reasoning
- **2**: Partially correct or correct but truncated
- **1**: Wrong answer but reasonable attempt
- **0**: Completely wrong, empty, or no answer

**Common Patterns:**
- T1 questions (medium difficulty): Mixed results, scores 1-3
- T2 questions (hard): Mostly scored 1-2
- T3 questions (post-doctoral): Almost all scored 0-1
- Simple extraction tasks (emails, JSON): Often scored 2-3
- Complex reasoning: Rarely above 1

### Output Files

All CSV files written to: `/mnt/raid0/llm/claude/benchmarks/results/reviews/`

Each CSV contains:
- `suite`: Benchmark suite name (general, thinking, math, agentic, coder)
- `question_id`: Question identifier
- `tokens_per_second`: Generation speed
- `claude_score`: Score 0-3
- `score_reason`: Brief explanation of score

### Recommendations

1. **Draft Model Usage**: These 0.5B-1.7B models are suitable for:
   - Simple extraction tasks (emails, structured data)
   - JSON parsing
   - Fast draft generation for speculative decoding
   - NOT suitable for reasoning, complex instructions, or T3 questions

2. **Quality Threshold**: For production use requiring reliability, scores of 80%+ are typical for good models. These draft models at 40-45% should only be used as draft generators, not standalone.

3. **Further Analysis**: The `general_deepseek_r1_0528_qwen3_8b` lower-than-expected score warrants investigation:
   - Check question distribution
   - Review specific failure modes
   - Compare to other 8B models
