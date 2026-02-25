# Coder Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Active

---

## Current Coder Stack

| Role | Model | Speed | Acceleration |
|------|-------|-------|--------------|
| coder_primary | Qwen3-Coder-30B-A3B-Instruct | 45.3 t/s | MoE (4 experts) |
| coder_escalation | Qwen3-Coder-53B-A3B-TOTAL-RECALL-v2 | 30.4 t/s | MoE (4 experts) |
| architect_coding | Qwen3-Coder-480B-A35B-Instruct | 5.23 t/s | MoE (2 experts) |

**Coder escalation ladder:**
```
coder_primary (30B MoE, fast, hot)
    ↓
coder_escalation (53B MoE, balanced)
    ↓
architect_coding (480B MoE, heavy)
```

---

## Models to Benchmark

### Downloaded (Ready)
- [x] Qwen3-Coder-30B-A3B-Instruct (coder_primary)
- [x] Qwen3-Coder-53B-A3B-TOTAL-RECALL-v2 (coder_escalation)
- [ ] Qwen2.5-Coder-32B-Instruct (deprecated - too slow)
- [ ] Qwen2.5-Coder-7B-Instruct (fast worker candidate?)

### Candidates
- Qwen3-Coder-14B (smaller, faster?)
- DeepSeek-Coder-V2 variants
- CodeLlama variants

---

## Quality Rubric (Tiered Difficulty)

**Design principle:** Questions should differentiate models. Easy questions (T1) verify basic competence. Hard questions (T2-3) separate coder_primary from coder_escalation.

**Scoring:**
- 1-2: Wrong or broken code
- 3: Works but has issues (inefficient, missing edge cases)
- 4: Correct with minor style issues
- 5: Correct, clean, handles edge cases

---

### TIER 1: Baseline (coder_primary should score 4-5)

#### C-T1-Q1: Simple Function
```
Prompt: "Write a Python function that returns the factorial of a number n. Include type hints."
Expected: Correct recursive or iterative implementation with type hints.
```
| Criterion | Weight |
|-----------|--------|
| Correct logic | 50% |
| Type hints | 20% |
| Handles edge cases (n=0, n=1) | 20% |
| Clean code | 10% |

#### C-T1-Q2: String Manipulation
```
Prompt: "Write a function that reverses words in a string but keeps word order.
Example: 'hello world' -> 'olleh dlrow'"
Expected: Split, reverse each, join.
```
| Criterion | Weight |
|-----------|--------|
| Correct output | 60% |
| Handles spaces correctly | 20% |
| Clean implementation | 20% |

#### C-T1-Q3: Basic Data Structure
```
Prompt: "Write a Python class for a Stack with push, pop, peek, and is_empty methods."
Expected: Correct stack implementation.
```
| Criterion | Weight |
|-----------|--------|
| All methods work | 50% |
| Handles empty stack | 30% |
| Clean code | 20% |

---

### TIER 2: Medium-Hard (coder_primary: 3-4, escalation: 4-5)

#### C-T2-Q1: Algorithm with Edge Cases
```
Prompt: "Write a function to find the longest palindromic substring in a string.
Return the first one if there are multiple of the same length."
Expected: O(n²) or better solution, handles empty string and single char.
```
| Criterion | Weight |
|-----------|--------|
| Correct algorithm | 40% |
| Handles edge cases | 30% |
| Efficiency (not brute force O(n³)) | 20% |
| Clean code | 10% |

#### C-T2-Q2: Concurrent Code
```
Prompt: "Write a Python function that fetches multiple URLs concurrently using asyncio.
Return a dict mapping URL to response status code. Handle timeouts (5s per request)."
Expected: Uses aiohttp or similar, proper async/await, timeout handling.
```
| Criterion | Weight |
|-----------|--------|
| Correct async pattern | 40% |
| Timeout handling | 30% |
| Error handling | 20% |
| Clean code | 10% |

#### C-T2-Q3: Design Pattern
```
Prompt: "Implement an LRU cache class with get(key) and put(key, value) methods.
Both should be O(1). Use a capacity of n items."
Expected: OrderedDict or dict + doubly linked list implementation.
```
| Criterion | Weight |
|-----------|--------|
| O(1) operations | 40% |
| Correct LRU eviction | 30% |
| Clean implementation | 20% |
| Handles capacity | 10% |

---

### TIER 3: Hard (coder_primary: 2-3, escalation: 3-4, architect: 4-5)

#### C-T3-Q1: Complex Algorithm
```
Prompt: "Implement a function that finds all valid IP addresses that can be formed
by inserting dots into a string of digits. Example: '25525511135' ->
['255.255.11.135', '255.255.111.35']"
Expected: Backtracking solution with proper validation.
```
| Criterion | Weight |
|-----------|--------|
| Correct algorithm | 40% |
| All valid IPs found | 30% |
| Efficient (not generating invalid candidates) | 20% |
| Clean code | 10% |

#### C-T3-Q2: System Design Code
```
Prompt: "Write a rate limiter class that allows N requests per minute using the
token bucket algorithm. Include methods: allow_request() -> bool, and
reset(). Make it thread-safe."
Expected: Token bucket with thread-safe implementation.
```
| Criterion | Weight |
|-----------|--------|
| Correct token bucket logic | 40% |
| Thread safety | 30% |
| Time handling | 20% |
| Clean code | 10% |

#### C-T3-Q3: Bug Fix in Complex Code
```
Prompt: "Fix the bug in this code:
```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= merged[-1][1]:
            merged[-1][1] = intervals[i][1]
        else:
            merged.append(intervals[i])
    return merged
```
What's wrong? Fix it."
Expected: Bug is merged[-1][1] = intervals[i][1] should be max() of both.
```
| Criterion | Weight |
|-----------|--------|
| Identifies the bug | 50% |
| Correct fix | 30% |
| Explains why | 20% |

#### C-T3-Q4: Code Generation from Spec
```
Prompt: "Write a Python decorator @retry(max_attempts=3, delay=1.0, backoff=2.0)
that retries a function on exception with exponential backoff.
Should work with both sync and async functions."
Expected: Decorator handling both sync/async, proper backoff logic.
```
| Criterion | Weight |
|-----------|--------|
| Works for sync functions | 30% |
| Works for async functions | 30% |
| Correct backoff | 25% |
| Clean implementation | 15% |

---

## Rubric Scoring Summary

| Tier | coder_primary Target | coder_escalation Target | architect_coding Target |
|------|---------------------|------------------------|------------------------|
| T1 (Baseline) | 4-5 | 5 | 5 |
| T2 (Medium) | 3-4 | 4-5 | 5 |
| T3 (Hard) | 2-3 | 3-4 | 4-5 |

**Role assignment based on scores:**
- coder_primary: Must score 4+ on T1, 3+ on T2
- coder_escalation: Must score 5 on T1, 4+ on T2, 3+ on T3
- architect_coding: Must score 5 on T1-T2, 4+ on T3

---

## Acceleration Compatibility

| Model | MoE Reduction | Spec Decode | Notes |
|-------|---------------|-------------|-------|
| Qwen3-Coder-30B-A3B | Yes (4 experts) | No | MoE primary |
| Qwen3-Coder-53B-A3B | Yes (4 experts) | No | MoE escalation |
| Qwen2.5-Coder-32B | No (dense) | Yes | Too slow even with spec |
| Qwen2.5-Coder-7B | No (dense) | Yes | Fast but quality? |

---

## Benchmark Scripts

### Single Model Testing
```bash
# Dense model (baseline only)
./scripts/benchmark/run_coder_rubric.sh /path/to/model.gguf ModelName dense

# MoE model (baseline + moe4 configurations)
./scripts/benchmark/run_coder_rubric.sh /path/to/model.gguf ModelName qwen3moe
```

### Full Suite
```bash
# Run all coder models with appropriate configs
./scripts/benchmark/run_all_coder_benchmarks.sh
```

### Architecture Types
| Arch | Configurations Tested |
|------|----------------------|
| `dense` | baseline |
| `qwen3moe` | baseline, moe4 |

Results are saved with config prefix: `{model}_{config}_{test}.txt`

---

## Results (fill in after benchmarking)

| Model | Quant | Config | Speed | T1 | T2 | T3 | Role |
|-------|-------|--------|-------|----|----|-----|------|
| Qwen3-Coder-30B-A3B | Q4_K_M | baseline | TBD | TBD | TBD | TBD | - |
| Qwen3-Coder-30B-A3B | Q4_K_M | moe4 | 45.3 t/s | TBD | TBD | TBD | coder_primary? |
| Qwen3-Coder-53B-A3B | Q4_K_M | baseline | TBD | TBD | TBD | TBD | - |
| Qwen3-Coder-53B-A3B | Q4_K_M | moe4 | 30.4 t/s | TBD | TBD | TBD | coder_escalation? |
| Qwen2.5-Coder-32B | Q4_K_M | baseline | 9.7 t/s | TBD | TBD | TBD | deprecated? |
