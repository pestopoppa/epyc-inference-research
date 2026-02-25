# Math Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Active

---

## Purpose

Benchmark mathematical reasoning capability for models that may handle:
- Calculation tasks in orchestration
- Data analysis and statistics
- Optimization problems
- Proof verification

---

## Proposed Math Roles

| Role | Purpose | Speed Target |
|------|---------|--------------|
| math_worker | Fast arithmetic, simple algebra | >40 t/s |
| math_specialist | Complex multi-step problems | >20 t/s |
| math_oracle | Proofs, advanced calculus, verification | >10 t/s |

---

## Quality Rubric (Tiered Difficulty)

**Scoring:**
- 1-2: Wrong answer, fundamental errors
- 3: Partial solution, calculation errors
- 4: Correct answer, minor notation issues
- 5: Correct, well-explained, efficient method

---

### TIER 1: Baseline (math_worker should score 4-5)

#### M-T1-Q1: Basic Arithmetic
```
Prompt: "Calculate: 847 × 23 + 156 ÷ 4 - 89

Show your work step by step, then give the final answer."

Expected:
847 × 23 = 19,481
156 ÷ 4 = 39
19,481 + 39 - 89 = 19,431
```
| Criterion | Weight |
|-----------|--------|
| Correct final answer | 50% |
| Correct intermediate steps | 30% |
| Clear presentation | 20% |

#### M-T1-Q2: Simple Algebra
```
Prompt: "Solve for x: 3x + 7 = 22

Show each step."

Expected:
3x + 7 = 22
3x = 22 - 7
3x = 15
x = 5
```
| Criterion | Weight |
|-----------|--------|
| Correct answer (x = 5) | 50% |
| Valid algebraic steps | 30% |
| Clear notation | 20% |

#### M-T1-Q3: Unit Conversion
```
Prompt: "Convert 2.5 kilometers to:
1. meters
2. centimeters
3. miles (use 1 mile = 1.609 km)"

Expected:
1. 2,500 meters
2. 250,000 centimeters
3. ~1.554 miles
```
| Criterion | Weight |
|-----------|--------|
| All conversions correct | 60% |
| Shows conversion factors | 25% |
| Appropriate precision | 15% |

---

### TIER 2: Medium-Hard (worker: 3-4, specialist: 4-5)

#### M-T2-Q1: Multi-Step Word Problem
```
Prompt: "A store offers a 20% discount on all items. After the discount,
a 8% sales tax is applied. If an item originally costs $150:

1. What is the price after discount?
2. What is the final price after tax?
3. What percentage of the original price is the final price?"

Expected:
1. $150 × 0.80 = $120
2. $120 × 1.08 = $129.60
3. $129.60 / $150 = 86.4%
```
| Criterion | Weight |
|-----------|--------|
| All three answers correct | 50% |
| Clear step-by-step work | 30% |
| Correct order of operations | 20% |

#### M-T2-Q2: System of Equations
```
Prompt: "Solve the system of equations:
2x + 3y = 13
4x - y = 5

Find the values of x and y."

Expected:
From equation 2: y = 4x - 5
Substitute into equation 1: 2x + 3(4x - 5) = 13
2x + 12x - 15 = 13
14x = 28
x = 2, y = 3
```
| Criterion | Weight |
|-----------|--------|
| Correct x value | 25% |
| Correct y value | 25% |
| Valid method | 30% |
| Verification | 20% |

#### M-T2-Q3: Probability
```
Prompt: "A bag contains 5 red balls, 3 blue balls, and 2 green balls.
If you draw 2 balls without replacement:

1. What is the probability both are red?
2. What is the probability of getting one red and one blue?"

Expected:
1. P(both red) = (5/10) × (4/9) = 20/90 = 2/9 ≈ 0.222
2. P(red then blue) + P(blue then red) = (5/10)(3/9) + (3/10)(5/9) = 30/90 = 1/3 ≈ 0.333
```
| Criterion | Weight |
|-----------|--------|
| Correct probability 1 | 30% |
| Correct probability 2 | 30% |
| Shows combinatorial reasoning | 25% |
| Simplified fractions | 15% |

---

### TIER 3: Hard (worker: 2-3, specialist: 3-4, oracle: 4-5)

#### M-T3-Q1: Optimization
```
Prompt: "A farmer has 200 meters of fencing to enclose a rectangular field
that borders a river (no fencing needed on the river side).

What dimensions maximize the enclosed area? What is the maximum area?"

Expected:
Let width = x, length = y
Constraint: 2x + y = 200, so y = 200 - 2x
Area = x × y = x(200 - 2x) = 200x - 2x²
dA/dx = 200 - 4x = 0
x = 50 meters, y = 100 meters
Maximum area = 5,000 square meters
```
| Criterion | Weight |
|-----------|--------|
| Correct dimensions | 30% |
| Correct maximum area | 25% |
| Valid calculus/algebra method | 30% |
| Clear constraint setup | 15% |

#### M-T3-Q2: Proof/Logical Reasoning
```
Prompt: "Prove that the sum of the first n positive integers is n(n+1)/2.

Use mathematical induction."

Expected:
Base case: n=1: 1 = 1(2)/2 = 1 ✓
Inductive hypothesis: Assume true for n=k: 1+2+...+k = k(k+1)/2
Inductive step: Prove for n=k+1:
1+2+...+k+(k+1) = k(k+1)/2 + (k+1)
= (k+1)(k/2 + 1)
= (k+1)(k+2)/2
This matches the formula for n=k+1. QED.
```
| Criterion | Weight |
|-----------|--------|
| Valid base case | 20% |
| Clear inductive hypothesis | 20% |
| Correct inductive step | 40% |
| Proper conclusion | 20% |

#### M-T3-Q3: Calculus Integration
```
Prompt: "Evaluate the definite integral:
∫₀² (3x² + 2x - 1) dx

Show your work."

Expected:
∫(3x² + 2x - 1)dx = x³ + x² - x + C
[x³ + x² - x]₀² = (8 + 4 - 2) - (0 + 0 - 0) = 10
```
| Criterion | Weight |
|-----------|--------|
| Correct antiderivative | 30% |
| Correct evaluation at bounds | 30% |
| Correct final answer (10) | 30% |
| Clear presentation | 10% |

#### M-T3-Q4: Statistical Analysis
```
Prompt: "Given the dataset: 12, 15, 18, 22, 25, 28, 30, 35

Calculate:
1. Mean
2. Median
3. Standard deviation (population)
4. Is there any outlier using the 1.5×IQR rule?"

Expected:
1. Mean = (12+15+18+22+25+28+30+35)/8 = 185/8 = 23.125
2. Median = (22+25)/2 = 23.5
3. σ = √[Σ(x-μ)²/n] ≈ 7.35
4. Q1=16.5, Q3=29, IQR=12.5, bounds=[16.5-18.75, 29+18.75]=[-2.25, 47.75]
   No outliers (all values within bounds)
```
| Criterion | Weight |
|-----------|--------|
| Correct mean | 20% |
| Correct median | 20% |
| Correct std dev (within 0.5) | 30% |
| Correct outlier analysis | 30% |

---

## Rubric Scoring Summary

| Tier | math_worker Target | math_specialist Target | math_oracle Target |
|------|-------------------|----------------------|-------------------|
| T1 (Baseline) | 4-5 | 5 | 5 |
| T2 (Medium) | 3-4 | 4-5 | 5 |
| T3 (Hard) | 2-3 | 3-4 | 4-5 |

**Role assignment based on scores:**
- math_worker: Must score 4+ on T1, 3+ on T2
- math_specialist: Must score 5 on T1, 4+ on T2, 3+ on T3
- math_oracle: Must score 5 on T1-T2, 4+ on T3

---

## Models to Benchmark

### Specialized Math Models
| Model | Size | Type | Notes |
|-------|------|------|-------|
| Qwen2.5-Math-7B-Instruct | ~4GB | Dense | Fast math specialist |
| Qwen2.5-Math-72B-Instruct | ~40GB | Dense | Heavy math oracle |

### Reasoning Models (Strong Math)
| Model | Size | Type | Notes |
|-------|------|------|-------|
| DeepSeek-R1-Distill-Qwen-7B | ~4GB | Dense | Reasoning distillation |
| DeepSeek-R1-Distill-Qwen-14B | ~8GB | Dense | Reasoning distillation |
| DeepSeek-R1-Distill-Llama-8B | ~4GB | Dense | Reasoning distillation |
| DeepSeek-R1-Distill-Llama-70B | ~40GB | Dense | Heavy reasoning |

### General Models (Math Capable)
| Model | Size | Type | Notes |
|-------|------|------|-------|
| Qwen2.5-72B-Instruct | ~40GB | Dense | Strong general math |
| Qwen3-Next-80B-A3B | ~18GB | MoE | Test with MoE reduction |

---

## Benchmark Scripts

### Single Model Testing
```bash
# Dense model (baseline only)
./scripts/benchmark/run_math_rubric.sh /path/to/model.gguf ModelName dense

# MoE model (baseline + moe4 configurations)
./scripts/benchmark/run_math_rubric.sh /path/to/model.gguf ModelName qwen3moe
```

### Full Suite
```bash
# Run all math models with appropriate configs
./scripts/benchmark/run_all_math_benchmarks.sh
```

### Architecture Types
| Arch | Configurations Tested |
|------|----------------------|
| `dense` | baseline |
| `qwen3moe` | baseline, moe4 |

Results are saved with config prefix: `{model}_{config}_{test}.txt`
Output directory: `/tmp/claude/math_rubric_results/`

### Overnight Suite
```bash
# Run all benchmarks (thinking, coder, vl, general, agentic, math) + optimization tests
./scripts/benchmark/run_overnight_benchmark_suite.sh

# Run only math benchmark suite
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite math

# Dry run to see what would execute
./scripts/benchmark/run_overnight_benchmark_suite.sh --dry-run
```

---

## Results (fill in after benchmarking)

| Model | Quant | Config | Speed | T1 | T2 | T3 | Role |
|-------|-------|--------|-------|----|----|-----|------|
| TBD | | | | | | | |

---

## Notes

- Arithmetic precision is critical - small errors cascade
- For complex problems, prefer models that show work
- Specialized math models (Qwen2.5-Math) should outperform general models on T2-T3
- Reasoning models (DeepSeek-R1) may be slower but more accurate on proofs
