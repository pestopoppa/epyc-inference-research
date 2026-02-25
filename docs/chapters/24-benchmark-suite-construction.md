# Chapter 24: Benchmark Suite Construction

## Introduction

Our 8-suite benchmark framework (Chapter 21) relies on deterministically scorable questions derived from public benchmark datasets. This chapter documents the construction methodology, scoring contracts, and reproduction instructions so anyone can independently rebuild the question pools.

The benchmark prompts themselves are gitignored — they are reconstructible artifacts, not source-of-truth data. This chapter is the source of truth.

## Design Principles

Every question in the suite must be machine-verifiable with no human judgment needed. We only use public benchmark sources, stratify into three difficulty tiers, and require at least 40 questions per suite for statistical significance in the MemRL learning loop.

<details>
<summary>Full design principles</summary>

1. **Deterministic scoring only.** Every question has a machine-verifiable answer. No Claude-as-Judge, no rubrics, no subjective evaluation. This enables automated regression testing and MemRL reward injection without API costs or evaluator variance.

2. **Public provenance.** All questions are derived from or inspired by established public benchmarks. No proprietary datasets. Anyone with access to the source benchmarks can reconstruct equivalent pools.

3. **Tier stratification.** Each suite has three difficulty tiers (T1/T2/T3) to differentiate model capability. T1 tests basic competence, T2 tests working knowledge, T3 tests expert-level reasoning.

4. **Minimum pool size: 40 questions per suite.** This is the floor needed for statistically meaningful sampling in the MemRL learning loop (10-prompt samples across 3-5 iterations).

</details>

## Suite Specifications

Each suite lives in a YAML file under `benchmarks/prompts/debug/`. The format is straightforward: a suite header with scoring defaults, followed by a list of questions with IDs, tiers, prompts, and expected answers.

<details>
<summary>File format and required fields</summary>

### File Format

Each suite is a YAML file at `benchmarks/prompts/debug/{suite}.yaml`:

<details>
<summary>Code: YAML file structure example</summary>

```yaml
suite: thinking
version: "1.0"
scoring_default:
  method: multiple_choice   # default if question omits scoring_method

questions:
  - id: arc_001             # unique within suite
    tier: 1                 # 1, 2, or 3
    prompt: |
      Which of the following is true about sound waves?
      A) They can travel through a vacuum
      B) They travel faster in solids than in gases
      C) They travel at the speed of light
      D) They cannot travel through liquids
    expected: "B"
    scoring_method: multiple_choice
    scoring_config: {}      # optional, method-specific
```

</details>

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier within the suite |
| `tier` | int | Difficulty: 1 (basic), 2 (working), 3 (expert) |
| `prompt` | string | The question text sent to the model |
| `expected` | string | Ground-truth answer (or empty for code_execution/programmatic) |
| `scoring_method` | string | One of: `multiple_choice`, `exact_match`, `code_execution`, `substring`, `programmatic` |
| `scoring_config` | dict | Method-specific parameters (optional) |

</details>

## Scoring Methods

All scoring is implemented in `scripts/benchmark/debug_scorer.py`. There are five methods, each targeting a different class of benchmark question. The scorer tries multiple extraction strategies before failing, so model output format is somewhat flexible.

<details>
<summary>Scoring method details and configuration</summary>

### 1. `multiple_choice`

**Source benchmarks:** ARC-Challenge, MMLU, HellaSwag

Extracts a letter (A/B/C/D) from model output and compares to `expected`. Uses multiple extraction strategies: "Answer: X", "(X)", standalone letter on last line, letter frequency analysis.

<details>
<summary>Code: multiple_choice YAML config</summary>

```yaml
scoring_method: multiple_choice
expected: "B"
```

</details>

### 2. `exact_match`

**Source benchmarks:** GSM8K, MATH

Extracts a value via regex, compares to `expected`. Supports numeric comparison (float tolerance 1e-6) and string normalization.

<details>
<summary>Code: exact_match YAML config</summary>

```yaml
scoring_method: exact_match
expected: "42"
scoring_config:
  extract_pattern: "####\\s*(\\d+)"   # GSM8K convention
  normalize: true                      # strip, lowercase, remove trailing period
```

</details>

### 3. `code_execution`

**Source benchmarks:** HumanEval, MBPP

Extracts a Python function from model output, appends test code, executes in a subprocess with a 10-second timeout. Passes if all assertions succeed.

<details>
<summary>Code: code_execution YAML config</summary>

```yaml
scoring_method: code_execution
expected: ""
scoring_config:
  test_code: |
    assert fibonacci(10) == 55
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
```

</details>

### 4. `substring`

**Source benchmarks:** Needle-in-a-Haystack, BFCL (function name presence)

Checks whether a specific substring appears in the model output. Optional case sensitivity.

<details>
<summary>Code: substring YAML config</summary>

```yaml
scoring_method: substring
scoring_config:
  substring: "AquaSense Pro-X7"
  case_sensitive: false
```

</details>

### 5. `programmatic`

**Source benchmark:** IFEval

Runs format-specific verifiers (word count, JSON validity, keyword presence, case constraints, list format, etc.). Each verifier is a Python function in the scorer.

<details>
<summary>Code: programmatic YAML config</summary>

```yaml
scoring_method: programmatic
scoring_config:
  verifier: word_count_range
  min_val: 40
  max_val: 50
```

</details>

Available verifiers: `word_count_range`, `word_count_max`, `word_count_min`, `contains_keyword`, `no_keyword`, `numbered_list`, `bullet_list`, `json_valid`, `starts_with`, `ends_with`, `all_uppercase`, `all_lowercase`, `comma_separated`, `paragraph_count`, `sentence_count_min`.

</details>

## Suite-by-Suite Construction

Each of the nine suites draws from different public benchmarks and uses different scoring methods. Below you will find the source breakdown, tier distribution, and reconstruction notes for every suite.

<details>
<summary>Thinking suite (40 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| ARC-Challenge | `arc_*` | 1-2 | ~17 | multiple_choice |
| Logic puzzles (constructed) | `logic_*` | 2-3 | ~23 | exact_match / multiple_choice |

**Tier 1 (13):** Basic science literacy — state of matter, food chains, plant biology, weather, optics. Direct ARC-Challenge style 4-option MCQ.

**Tier 2 (14):** Applied reasoning — inherited vs learned traits, erosion, symbiosis, circuit logic, river crossing, deductive puzzles (truth-tellers/liars, age constraints).

**Tier 3 (13):** Expert logic — scheduling optimization, combinatorial deduction, probability, causal inference DAGs, knight/knave paradoxes, constraint satisfaction.

**Reconstruction:** Take 15-20 ARC-Challenge Easy/Challenge questions from the public Hugging Face dataset. Add 20 constructed logic puzzles with unambiguous numerical or letter answers. Verify each answer manually.

</details>

<details>
<summary>Math suite (40 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| GSM8K | `gsm8k_*` | 1-2 | ~25 | exact_match (####) |
| MATH | `math_*` | 3 | ~15 | exact_match |

**Tier 1 (12):** Single-step arithmetic word problems — total cost, simple averages, basic fractions.

**Tier 2 (13):** Multi-step word problems — compound interest, rate problems, work problems, mixtures.

**Tier 3 (15):** Competition-level — modular arithmetic, number theory, combinatorics, series convergence, prime sums, last digits of powers.

**Reconstruction:** Sample 20 GSM8K problems from the public test split. Add 20 MATH-competition-style problems at Levels 3-5. All answers must be integers or simple expressions extractable via `####\s*(\d+)` or `####\s*(\S+)`.

**Validation note:** Always verify math answers by hand. Agent-generated questions had 2 answer errors in the original batch (corrected: prime sum 23+29+31+37=120, not 129; average distances adjusted for integer result).

</details>

<details>
<summary>General suite (42 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| MMLU (miscellaneous) | `mmlu_misc_*` | 1 | ~8 | multiple_choice |
| MMLU (history) | `mmlu_hist_*` | 2 | ~7 | multiple_choice |
| MMLU (science) | `mmlu_sci_*` | 1-2 | ~7 | multiple_choice |
| MMLU (CS) | `mmlu_cs_*` | 2-3 | ~7 | multiple_choice |
| MMLU (philosophy) | `mmlu_phil_*` | 2-3 | ~6 | multiple_choice |
| MMLU (economics) | `mmlu_econ_*` | 2-3 | ~7 | multiple_choice |

**Reconstruction:** Sample ~7 questions per MMLU subject (miscellaneous, world history, natural science, computer science, philosophy, economics). Use the public validation/test splits. All 4-option MCQ with single correct letter.

**Validation note:** Check for near-duplicates across subjects (e.g., "Red Planet" can appear in both miscellaneous and science).

</details>

<details>
<summary>Coder suite (40 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| HumanEval | `humaneval_*` | 1 | ~14 | code_execution |
| MBPP | `mbpp_*` | 1-2 | ~13 | code_execution |
| Constructed (harder) | `code_hard_*` | 2-3 | ~13 | code_execution |

**Tier 1 (16):** One-liner functions — max of list, is_even, reverse string, count vowels, celsius conversion, sum/min/product of list, remove duplicates.

**Tier 2 (15):** Multi-function problems — fibonacci, prime checking, prime factorization, longest common prefix, merge sorted lists, rotate list, find missing number, balanced brackets, two sum, group anagrams.

**Tier 3 (9):** Algorithm design — longest palindromic substring, permutation generation, quicksort, binary search, Dijkstra's shortest path, matrix multiplication.

**Reconstruction:** Take 10 HumanEval functions (easiest) and 10 MBPP problems. Write 20 additional problems covering standard algorithms. Each question needs 3-6 assertion test cases in `scoring_config.test_code`.

**Key constraint:** All functions must be self-contained Python (no imports required for the function itself, though test code may use stdlib). Function name is specified in the prompt.

</details>

<details>
<summary>VL -- Vision-Language suite (40 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| Text-described visuals | `vl_text_*` | 1 | ~5 | exact_match / multiple_choice |
| Charts/graphs | `vl_chart_*` | 1-3 | ~14 | exact_match |
| Science diagrams | `vl_diagram_*` | 2-3 | ~7 | exact_match / multiple_choice |
| Statistics visuals | `vl_stat_*` | 2-3 | ~6 | exact_match / substring |
| Infographics | `vl_info_*` | 2-3 | ~8 | exact_match / multiple_choice |

**Design decision:** We use text descriptions of visual content rather than actual images. This enables deterministic scoring without vision model dependencies and works with text-only models. The test measures whether the model can reason about spatial/quantitative information presented textually — a proxy for visual reasoning that still differentiates models.

**Tier 1 (8):** Simple chart reading — bar chart tallest bar, line chart peak, table lookup.

**Tier 2 (18):** Multi-step visual reasoning — stacked bar totals, Venn diagram intersections, trend identification, Gantt chart duration, heatmap extremes.

**Tier 3 (14):** Statistical interpretation — box plot comparisons, correlation matrices, ROC curves, confusion matrix metrics, Q-Q plot interpretation, network topology analysis.

**Reconstruction:** Write 40 text descriptions of common visualization types. Each must have a single unambiguous numerical or categorical answer. Cover: bar, line, pie, scatter, box plot, histogram, heatmap, Venn diagram, network topology, Gantt chart, confusion matrix, correlation matrix.

</details>

<details>
<summary>Agentic suite (40 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| BFCL-inspired | `bfcl_*` | 1-3 | 40 | substring (function name) |

**Tier 1 (11):** Single available function, simple parameter extraction — `get_weather("London")`, `send_email(to=..., subject=..., body=...)`.

**Tier 2 (15):** Multiple available functions, must select correct one — `compress_file` vs `decompress_file`, `http_post` vs `http_get`, `create_user` vs `update_user` vs `delete_user`.

**Tier 3 (14):** Complex parameter extraction — nested objects, lists, multiple optional parameters. E.g., `deploy_container(image=..., ports=[...], env_vars={...}, restart_policy=...)`.

**Scoring contract:** `substring` check for the correct function name in the output. This is deliberately lenient — we care that the model selects the right tool, not that JSON is perfectly formatted (instruction_precision suite covers format compliance).

**Reconstruction:** Define 40 function signatures with docstrings. Write natural language requests that map to exactly one function. Ensure tier 2 questions have plausible distractors (similar functions where only one is correct). Use Berkeley Function Calling Leaderboard (BFCL) function formats as reference.

</details>

<details>
<summary>Instruction Precision suite (43 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| IFEval-inspired | `ifeval_*` | 1-3 | 43 | programmatic (verifiers) |

**Tier 1 (13):** Single constraint — word count exact, include keyword, numbered list, starts with specific word.

**Tier 2 (17):** Moderate constraint — all uppercase, all lowercase, JSON object, bullet list, ends with phrase, word count range.

**Tier 3 (13):** Compound constraints — comma-separated format, paragraph count, sentence count minimum, JSON array, combined keyword + word count + case.

<details>
<summary>Data: verifier coverage breakdown</summary>

| Verifier | What It Checks | Count |
|----------|----------------|-------|
| `word_count_range` | Exact word count or range | ~5 |
| `word_count_max` / `_min` | Upper/lower bound on words | ~4 |
| `contains_keyword` / `no_keyword` | Keyword presence/absence | ~8 |
| `numbered_list` / `bullet_list` | List formatting | ~4 |
| `json_valid` | Parseable JSON output | ~4 |
| `starts_with` / `ends_with` | First/last text match | ~4 |
| `all_uppercase` / `all_lowercase` | Case compliance | ~5 |
| `paragraph_count` / `sentence_count_min` | Structure compliance | ~5 |
| `comma_separated` | Single-line CSV format | ~2 |

</details>

**Reconstruction:** Write 40+ prompts with explicit format constraints. Each prompt must specify exactly one verifiable constraint (or a compound constraint where one verifier covers the critical aspect). Use IFEval's constraint taxonomy as reference.

**Validation note:** Each question must map to exactly one verifier. Prompts that require multiple verifiers (e.g., "all lowercase AND contains keyword") should be scored on the harder-to-satisfy constraint, since the scorer applies one verifier per question.

</details>

<details>
<summary>Long Context suite (40 questions)</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| Needle-in-a-Haystack | `needle_*` | 1-3 | 40 | exact_match / substring |

**Tier 1 (8):** Short passages (200-400 words), single fact retrieval — date, measurement, ID number.

**Tier 2 (14):** Medium passages (400-800 words), fact buried in technical context — specific model numbers, personnel names, project codes.

**Tier 3 (18):** Long passages (800-1500 words), multi-document scenarios — cross-referencing 3-4 documents to find a specific detail, temporal reasoning across reports, extracting anomalies from dense technical text.

**Domain coverage:** Biotech, cybersecurity, energy, pharma, telecom, satellite imaging, materials science, fusion research, autonomous vehicles, smart cities, agriculture, wildlife conservation, digital preservation, quantum computing, environmental monitoring.

**Reconstruction:** Write 40 synthetic passages embedding specific "needle" facts (names, dates, model numbers, measurements). Each passage should contain plausible distractor information. The needle should not be in the first or last sentence. Use `substring` scoring for names/phrases, `exact_match` with numeric extraction for numbers.

</details>

<details>
<summary>Mode Advantage suite (90 questions) -- February 2026</summary>

| Source | ID Prefix | Tier | Count | Scoring |
|--------|-----------|------|-------|---------|
| Computation-gated | `ma_comp_*` | 2-3 | 15 | exact_match |
| Iterative-fix | `ma_iter_*` | 2-3 | 15 | code_execution / exact_match |
| Multi-step composition | `ma_multi_*` | 2-3 | 15 | exact_match |
| Escalation-gated | `ma_esc_*` | 3 | 15 | code_execution |
| Mini-SWE | `ma_swe_*` | 2-3 | 30 | code_execution |

**Purpose**: Produce strong comparative rewards for MemRL routing. Unlike other suites where direct inference often suffices, these tasks structurally require specific execution modes (react/repl/delegation).

**Tier 2 (45):** Computation that models hallucinate (modular arithmetic, statistics, string encoding), iterative debugging (bug fix with test suite, regex construction), and multi-step chained calculations.

**Tier 3 (45):** Expert algorithms (A*, trie, Union-Find, KMP, Bloom filter, Aho-Corasick), mini-SWE tasks (broken class + failing tests + known fix), and complex constraint satisfaction.

**Scoring breakdown**: 31 exact_match, 57 code_execution, 2 substring. All expected answers verified by Python computation. SWE tasks use code_execution with inline test assertions.

**Reconstruction:** Write tasks where the answer is structurally uncomputable by direct inference: large-number modular arithmetic, iterative code debugging, multi-step data processing, graduate-level algorithm implementation. Verify every expected answer by running the computation. Use `mode_advantage: true` tag on exemplars added to other suites.

**Validation note:** Agent-generated mode-advantage questions had 17 answer errors across 76 tasks in the original batch. Always verify expected answers by computation, especially for: modular arithmetic (rounding), financial calculations (compound interest), combinatorial counting, and CSV data aggregation.

**Exemplars in other suites**: 16 additional `mode_advantage: true` tagged questions across math (+4), coder (+4), agentic (+4), and long_context (+4).

</details>

## Current Pool Statistics

Across all nine suites we have 431 questions total, skewing toward T2 and T3 difficulty since that is where models actually diverge in capability.

<details>
<summary>Pool statistics by suite</summary>

| Suite | Count | T1 | T2 | T3 | Primary Scoring |
|-------|-------|----|----|----|--------------------|
| thinking | 40 | 13 | 14 | 13 | multiple_choice |
| math | 44 | 12 | 13 | 19 | exact_match |
| general | 42 | 10 | 17 | 15 | multiple_choice |
| coder | 44 | 16 | 15 | 13 | code_execution |
| vl | 40 | 8 | 18 | 14 | exact_match |
| agentic | 44 | 11 | 15 | 18 | substring |
| instruction_precision | 43 | 13 | 17 | 13 | programmatic |
| long_context | 44 | 8 | 14 | 22 | substring / exact_match |
| mode_advantage | 90 | 0 | 45 | 45 | code_execution / exact_match |
| **TOTAL** | **431** | 91 | 168 | 172 | -- |

</details>

## Reconstruction Procedure

You can rebuild the full question pool from scratch if the YAML files are lost or if you want to create a fresh independent set. The steps below walk through stub creation, population from public sources, and validation.

<details>
<summary>Step-by-step reconstruction instructions</summary>

1. **Install dependencies:**

<details>
<summary>Code: install pyyaml</summary>

```bash
pip install pyyaml
```

</details>

2. **Create suite stubs:**

<details>
<summary>Code: generate empty YAML stubs for all suites</summary>

```bash
mkdir -p benchmarks/prompts/debug
for suite in thinking math general coder vl agentic instruction_precision long_context mode_advantage; do
  cat > benchmarks/prompts/debug/${suite}.yaml << EOF
suite: ${suite}
version: "1.0"
scoring_default:
  method: multiple_choice

questions: []
EOF
done
```

</details>

3. **Populate from public sources:**
   - **thinking:** 15-20 ARC-Challenge questions (HF: `allenai/ai2_arc`, Challenge split) + 20 logic puzzles
   - **math:** 20 GSM8K questions (HF: `openai/gsm8k`, test split) + 20 MATH questions (HF: `hendrycks/math`, Levels 3-5)
   - **general:** 7 per subject from MMLU (HF: `cais/mmlu`, validation split) across 6 subjects
   - **coder:** 10 HumanEval (GitHub: `openai/human-eval`) + 10 MBPP (HF: `google-research-datasets/mbpp`) + 20 constructed
   - **vl:** 40 text-described visuals (constructed -- no image dependency)
   - **agentic:** 40 function-calling scenarios (inspired by BFCL: `ShishirPatil/gorilla`)
   - **instruction_precision:** 40 format-constraint prompts (inspired by IFEval: `google/IFEval`)
   - **long_context:** 40 needle-in-haystack passages (constructed)

4. **Validate:**

<details>
<summary>Code: Python validation script for all suites</summary>

```python
import yaml
for suite in ['thinking', 'math', 'general', 'coder', 'vl',
              'agentic', 'instruction_precision', 'long_context',
              'mode_advantage']:
    with open(f'benchmarks/prompts/debug/{suite}.yaml') as f:
        data = yaml.safe_load(f)
    ids = [q['id'] for q in data['questions']]
    assert len(ids) == len(set(ids)), f"Duplicate IDs in {suite}"
    assert len(ids) >= 40, f"{suite} has only {len(ids)} questions"
    for q in data['questions']:
        assert all(k in q for k in ['id', 'tier', 'prompt', 'scoring_method'])
```

</details>

5. **Verify math answers manually.** Agent-generated math questions are the most error-prone. Compute every expected answer by hand before committing.

</details>

## Quality Gates for New Questions

When adding questions to any suite, follow these six checks. They are quick but they prevent the most common errors we have hit in practice (duplicate IDs, ambiguous answers, wrong tier assignment).

<details>
<summary>Quality gate checklist</summary>

1. **No duplicate IDs** within a suite
2. **No near-duplicate prompts** (same concept with trivially different wording)
3. **Single unambiguous answer** -- if reasonable people could disagree, skip it
4. **Scoring method matches** -- the verifier must be able to check the answer without human judgment
5. **Manual answer verification** -- especially for math (compute by hand) and logic (trace the reasoning)
6. **Tier assignment** -- T1: one-step, T2: multi-step, T3: requires domain expertise or creative insight

</details>

## Relationship to Other Benchmark Layers

The project has two parallel benchmark tracks. The `v1/` suite uses Claude-as-Judge with rubric scoring for open-ended quality assessment. The `debug/` suite documented here uses machine verifiers for automated regression testing and MemRL rewards. Both cover the same eight categories but serve different purposes.

<details>
<summary>Comparison of rubric vs deterministic scoring</summary>

```
benchmarks/prompts/v1/          # Rubric-scored (Claude-as-Judge) — open-ended quality
benchmarks/prompts/debug/       # Deterministically scored — automated regression (THIS CHAPTER)
```

| Aspect | v1/ (Rubric) | debug/ (Deterministic) |
|--------|-------------|----------------------|
| Scoring | Claude-as-Judge (0-3 scale) | Machine verifier (pass/fail) |
| Cost | ~$0.02/question (API call) | Free (local execution) |
| Use case | Quality assessment, model ranking | Regression testing, MemRL rewards |
| Question style | Open-ended, subjective | Closed-form, single correct answer |
| Pool size | 10-15 per suite | 40+ per suite |

The debug suite is the foundation for Phase 3 MemRL validation: seeding specialist Q-values, running learning loops, and gating regressions -- all without API costs.

</details>

## References

<details>
<summary>Academic references and source datasets</summary>

1. Clark, P., et al. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.* https://arxiv.org/abs/1803.05457
2. Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K).* https://arxiv.org/abs/2110.14168
3. Hendrycks, D., et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset.* https://arxiv.org/abs/2103.03874
4. Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding (MMLU).* https://arxiv.org/abs/2009.03300
5. Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code (HumanEval).* https://arxiv.org/abs/2107.03374
6. Austin, J., et al. (2021). *Program Synthesis with Large Language Models (MBPP).* https://arxiv.org/abs/2108.07732
7. Zhou, J., et al. (2023). *Instruction-Following Evaluation for Large Language Models (IFEval).* https://arxiv.org/abs/2311.07911
8. Patil, S. G., et al. (2023). *Gorilla: Large Language Model Connected with Massive APIs (BFCL).* https://arxiv.org/abs/2305.15334
9. Kamradt, G. (2023). *Needle in a Haystack: Pressure Testing LLMs.* https://github.com/gkamradt/LLMTest_NeedleInAHaystack

</details>

---

*Previous: [Chapter 23: Security & Monitoring](23-security-and-monitoring.md)* | *Next: TBD*
