# Thinking Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Pending (awaiting model downloads)

---

## Concept: Reasoning Oracle

Workers and generals can consult a small Thinking model for precise reasoning questions without escalating the entire task. This provides a cheap "think harder" capability.

**Use cases:**
- Decision points: "Which approach is better: A or B?"
- Trade-off analysis: "What are the implications of X vs Y?"
- Correctness checks: "Is this logic sound given constraints Z?"
- Disambiguation: "What does the user likely mean by X?"

---

## Proposed Thinking Roles

| Role | Purpose | Model Candidate | Speed Target |
|------|---------|-----------------|--------------|
| oracle_reasoning | Fast reasoning for precise questions | Qwen3-4B-Thinking-2507 | >40 t/s |
| oracle_general | Quality reasoning for complex questions | Qwen3-30B-A3B-Thinking-2507 | >25 t/s (MoE) |
| oracle_architect | Heavy reasoning, multi-step analysis | Qwen3-Next-80B-A3B-Thinking | >10 t/s (MoE) |

**Thinking escalation ladder:**
```
oracle_reasoning (4B, fast, hot)
    ↓ (if reasoning insufficient)
oracle_general (30B MoE, balanced)
    ↓ (if still insufficient)
oracle_architect (80B MoE, heavy reasoning)
    ↓ (if task needs full decomposition)
architect_* (Instruct, task planning)
```

**Note:** Oracle roles answer questions. Architect roles decompose tasks.
Oracles are consulted BY workers. Architects manage workers.

**Design decision:** Text oracle ladder uses text-only models (no VL overhead).
Vision has its own ladder: worker_vision → vision_escalation → vision_architect (VL+Thinking).

---

## Models to Benchmark

### Downloaded (Ready)
- [ ] (add models as they complete)

### Downloading
- [ ] Qwen3-4B-Thinking-2507-Q8_0 (~4GB) - oracle_reasoning candidate
- [ ] Qwen3-30B-A3B-Thinking-2507-Q4_K_S (~18GB) - oracle_general candidate
- [ ] Qwen3-30B-A3B-Thinking-2507-Q8_0 (~30GB) - quality comparison
- [ ] Qwen3-Next-80B-A3B-Thinking-Q4_K_S (~45GB) - oracle_architect / heavy reasoning

### Candidates

| Model | Size | Type | Notes |
|-------|------|------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | ~1GB | Distilled | Fastest, quality TBD |
| DeepSeek-R1-Distill-Qwen-7B | ~4GB | Distilled | Good balance |
| DeepSeek-R1-Distill-Qwen-14B | ~8GB | Distilled | Higher quality |
| DeepSeek-R1-Distill-Qwen-32B | ~18GB | Distilled | oracle_general candidate |
| Qwen3-4B (thinking mode) | ~2.5GB | Native | Fast, enable thinking |
| Qwen3-8B (thinking mode) | ~5GB | Native | Quality/speed balance |
| Qwen3-14B (thinking mode) | ~8GB | Native | oracle_general candidate |
| Qwen3-30B-A3B (thinking mode) | ~18GB | MoE+Thinking | MoE speedup + thinking |
| QwQ-32B | ~18GB | Native | Qwen's dedicated reasoning model |

---

## Benchmark Metrics

### Performance (measure on our system)

| Metric | Command/Method | Target |
|--------|----------------|--------|
| Baseline t/s | `llama-bench -m MODEL -t 96 -p 512 -n 256` | Record speed |
| Thinking t/s | Same with thinking enabled | May be slower |
| Time to answer | Wall clock for standard questions | <5s for oracle_reasoning |
| Token overhead | Thinking tokens vs output tokens | Track ratio |

### Quality Rubric (Tiered Difficulty)

**Design principle:** Questions should differentiate models. Easy questions (Tier 1) verify basic competence. Hard questions (Tier 2-3) separate oracle tiers.

**Scoring:**
- 1-2: Wrong or confused
- 3: Partially correct, missing key insight
- 4: Correct with minor issues
- 5: Correct with nuanced understanding

---

## TIER 1: Baseline (4B should score 4-5)

#### T1-Q1: Simple Algorithm Choice
```
Question: "Sort 10 mostly-sorted items. Quicksort or insertion sort? One sentence."
Expected: Insertion sort - O(n) for nearly sorted.
```

#### T1-Q2: Basic Thread Safety
```
Question: "Is `self.count += 1` thread-safe in Python? One sentence."
Expected: No - read-modify-write race condition (GIL doesn't help here).
```

---

## TIER 2: Medium-Hard (4B: 2-3, 30B: 4-5)

#### T2-Q1: Counterintuitive Performance
```
Question: "Python function called 1000x/sec creates a 1KB dict each call.
Better to: A) Pre-allocate global dict and clear() each time, or B) Create new dict each time?
Explain memory and performance implications in 2-3 sentences."

Expected answer: B is usually better because:
- clear() doesn't release memory, just marks slots empty
- Modern allocators (pymalloc) are optimized for small allocations
- Creating new dicts lets GC work properly, avoids memory bloat
- Global mutable state adds complexity

Common mistakes (score 3):
- Assumes A is faster without understanding clear() semantics
- Ignores memory implications
- Doesn't consider GC behavior
```

#### T2-Q2: Subtle Concurrency Bug
```
Question: "Find the bug in this cache:
```python
cache = {}
lock = threading.Lock()

def get_cached(key, compute_fn):
    if key in cache:
        return cache[key]
    with lock:
        result = compute_fn()
        cache[key] = result
        return result
```
One paragraph explanation."

Expected answer: Double-checked locking bug (check-then-act race):
- Two threads both see key not in cache
- Both enter lock sequentially
- Both compute and write (wasted work)
- Fix: Check again inside lock, or use setdefault/get pattern

Common mistakes (score 3):
- Says it's thread-safe (wrong)
- Identifies race but not the specific pattern
- Suggests wrong fix (e.g., RLock doesn't help)
```

#### T2-Q3: Non-obvious API Design
```
Question: "For a library function that might fail, which return style is best?
A) return (success: bool, data, error_msg)
B) return {"success": bool, "data": ..., "error": ...}
C) raise Exception on error, return data on success
Brief reasoning for library consumed by external users."

Expected answer: C is generally best for Python libraries because:
- Pythonic - exceptions are idiomatic
- Type-safe - return type is just the data type
- Composable - callers can use normal try/except
- Clear control flow - success path is clean

When B might be better: REST APIs, cross-language serialization
When A might be better: Performance-critical code avoiding exception overhead

Common mistakes (score 3):
- Picks A or B without understanding Python idioms
- Doesn't consider composability
- Ignores type safety implications
```

---

## TIER 3: Very Hard (4B: 1-2, 30B: 3-4, 80B: 4-5)

#### T3-Q1: Multi-constraint Dependency Resolution
```
Question: "Service startup constraints:
- A must start before B
- B must start before C or D (either one)
- C and D cannot start simultaneously (resource conflict)
- E requires both C AND D to be running

What is the minimum number of sequential startup phases?
List the phases."

Expected answer: 4 phases
Phase 1: A
Phase 2: B
Phase 3: C (or D)
Phase 4: D (or C), then E can start once both are up

Wait - E requires both C AND D running simultaneously. So:
Phase 1: A
Phase 2: B
Phase 3: C
Phase 4: D
Phase 5: E (needs C and D both up)

Actually 5 phases if E needs both running, 4 if E just needs both completed.

Key insight: The C/D mutual exclusion constraint forces serialization.

Common mistakes (score 2-3):
- Tries to parallelize C and D (violates constraint)
- Miscounts phases
- Doesn't realize E needs both C AND D
```

#### T3-Q2: Distributed Systems - Vector Clocks
```
Question: "Vector clock merge:
P1 has clock [2,0,1], P2 has clock [1,2,0], P3 has clock [0,0,2].
1) P1 sends message to P2. What is P2's clock after receiving?
2) P2 then sends to P3. What is P3's clock after receiving?
Show work."

Expected answer:
1) P2 receives from P1:
   - P2 increments own position: [1,3,0]
   - Merge with P1's clock: max([2,0,1], [1,3,0]) = [2,3,1]
   - P2's new clock: [2,3,1]

2) P3 receives from P2:
   - P3 increments own position: [0,0,3]
   - Merge with P2's clock: max([2,3,1], [0,0,3]) = [2,3,3]
   - P3's new clock: [2,3,3]

Common mistakes (score 2-3):
- Forgets to increment receiver's own position before merge
- Uses wrong merge operation (addition instead of max)
- Confuses sender/receiver positions
```

#### T3-Q3: Subtle Type System Implications
```
Question: "Consider this function:
```python
from typing import Sequence

def first_or_default(items: Sequence[T], default: T) -> T:
    return items[0] if items else default
```
What happens with: first_or_default([], None)?
Is the type signature correct? What's the subtle issue?"

Expected answer:
The call works at runtime but has type issues:
1) If T is inferred from [], T is unknown/Any
2) None as default means T could be Optional[X]
3) The signature doesn't express that None is a valid default

Subtle issue: Variance and None handling
- Should default be Optional[T]? Then return type should be Optional[T]
- Or should we use overloads for None vs non-None defaults?
- TypeVar bound consideration: T should probably be bound

Better signatures:
```python
@overload
def first_or_default(items: Sequence[T], default: T) -> T: ...
@overload
def first_or_default(items: Sequence[T], default: None) -> T | None: ...
```

Common mistakes (score 2-3):
- Says "it works fine" (misses type issues)
- Doesn't understand TypeVar inference from empty sequence
- Doesn't consider None as special case
```

#### T3-Q4: Probabilistic Reasoning
```
Question: "Load balancer randomly routes to 3 servers with equal probability.
Server latencies: S1=10ms, S2=50ms, S3=100ms.
What is the expected MEDIAN latency over many requests?"

Expected answer:
Each request has 1/3 chance of each latency.
The distribution of single-request latencies is:
- P(10ms) = 1/3
- P(50ms) = 1/3
- P(100ms) = 1/3

For the median over many requests:
The median of this distribution is 50ms (the middle value).

Key insight: Don't confuse mean (53.3ms) with median.
The question asks for median, not mean or p99.

Common mistakes (score 2-3):
- Calculates mean instead of median
- Confuses "expected median" with complex statistics
- Overcomplicates with sampling distribution reasoning
```

---

## Rubric Scoring Summary

| Tier | 4B Target | 30B Target | 80B Target |
|------|-----------|------------|------------|
| T1 (Baseline) | 4-5 | 5 | 5 |
| T2 (Medium-Hard) | 2-3 | 4-5 | 5 |
| T3 (Very Hard) | 1-2 | 3-4 | 4-5 |

**Role assignment based on scores:**
- oracle_reasoning: Must score 4+ on T1, 3+ on T2
- oracle_general: Must score 5 on T1, 4+ on T2, 3+ on T3
- oracle_architect: Must score 5 on T1-T2, 4+ on T3

---

## Thinking Mode Configuration

### DeepSeek R1 Distill
- Native thinking, no special config needed
- Outputs `<think>...</think>` blocks

### Qwen3 Thinking Mode
```bash
# Enable via system prompt or temperature
--temp 0.6  # Recommended for thinking
# Or include "Think step by step" in prompt
```

### Thinking Token Handling
- Track `<think>` tokens separately from output
- May want to strip thinking from final oracle response
- Or pass thinking to worker for context

---

## Acceleration Compatibility

| Model Type | MoE Reduction | Spec Decode | Notes |
|------------|---------------|-------------|-------|
| R1-Distill (dense) | No | Yes | Use small draft |
| Qwen3-xB (dense) | No | Yes | Use Qwen2.5-0.5B draft |
| Qwen3-xB-A3B (MoE) | Yes (4 experts) | No | MoE + thinking |

---

## Decision Criteria

### For oracle_reasoning (fast, frequent) - Qwen3-4B-Thinking
1. Speed > 40 t/s (will be called often)
2. Correct on simple/medium questions (rubric score > 3.5)
3. Concise answers (not verbose chains)
4. Memory < 5GB (stays hot alongside workers)

### For oracle_general (balanced) - Qwen3-30B-A3B-Thinking
1. Speed > 25 t/s with MoE reduction (4 experts)
2. Correct on medium/hard questions (rubric score > 4.0)
3. Good reasoning chains without excessive verbosity
4. Memory < 20GB (can stay warm or cold-load fast)

### For oracle_architect (heavy reasoning) - Qwen3-Next-80B-A3B-Thinking
1. Speed > 10 t/s with MoE reduction (2-4 experts)
2. Correct on hard questions (rubric score > 4.5)
3. Can handle complex multi-step reasoning
4. Memory < 50GB (cold-loadable)

---

## Integration Notes

### How workers invoke oracle
```python
# Option 1: Explicit tool
{"tool": "consult_oracle", "question": "Which is better: A or B?"}

# Option 2: Orchestrator detects uncertainty markers
"I'm not sure whether..." → auto-routes to oracle

# Option 3: Two-phase
Step 1: Worker generates draft + questions
Step 2: Oracle answers questions
Step 3: Worker incorporates answers
```

### Response format
```json
{
  "answer": "Use insertion sort",
  "reasoning": "Nearly sorted data is O(n) with insertion sort vs O(n log n) for quicksort",
  "confidence": "high",
  "thinking_tokens": 45,
  "output_tokens": 28
}
```

---

## Benchmark Scripts

### Single Model Testing
```bash
# Dense model (baseline only)
./scripts/benchmark/run_thinking_rubric.sh /path/to/model.gguf ModelName dense

# MoE model (baseline + moe4 configurations)
./scripts/benchmark/run_thinking_rubric.sh /path/to/model.gguf ModelName qwen3moe
```

### Full Suite
```bash
# Run all thinking models with appropriate configs
./scripts/benchmark/run_all_thinking_benchmarks.sh
```

### Architecture Types
| Arch | Configurations Tested |
|------|----------------------|
| `dense` | baseline |
| `qwen3moe` | baseline, moe4 |

Results are saved with config prefix: `{model}_{config}_{test}.txt`
Output directory: `/tmp/claude/thinking_rubric_results/`

---

## Results (fill in after benchmarking)

| Model | Quant | MoE | Speed | T1 | T2 | T3 | Role |
|-------|-------|-----|-------|----|----|-----|------|
| Qwen3-4B-Thinking-2507 | Q8_0 | N/A | 22 t/s | **5.0** | **2.3** | **2.5** | ❌ T1-only (below oracle_reasoning threshold) |
| Qwen3-30B-A3B-Thinking-2507 | Q4_K_S | 4 | TBD | | | | oracle_general? |
| Qwen3-30B-A3B-Thinking-2507 | Q8_0 | 4 | TBD | | | | (quality ref) |
| Qwen3-Next-80B-A3B-Thinking | Q4_K_S | 2-4 | TBD | | | | oracle_architect? |

### Qwen3-4B-Thinking-2507 Detailed Results

**Speed:** 22.7 t/s generation (Q8_0, 4GB, dense model)

**Thinking Format:** Uses `<think>...</think>` blocks with detailed internal reasoning.

**Quality Rubric (New Tiered System):**
| Tier | Question | Score | Speed | Notes |
|------|----------|-------|-------|-------|
| T1 | Algorithm choice | 5/5 | 22.6 t/s | Correct - insertion sort for mostly-sorted |
| T1 | Thread safety | 5/5 | 22.1 t/s | Correct - race condition explained |
| T2 | Dict reuse | **2/5** | 22.1 t/s | **WRONG** - said pre-allocate better (new dict is correct) |
| T2 | Cache bug | **2/5** | 22.2 t/s | **MISSED** - focused on compute_fn, not double-checked locking |
| T2 | API design | 3/5 | 22.3 t/s | Picked dict (expected exceptions), reasoning OK |
| T3 | Dependency | 2/5 | 22.3 t/s | Got stuck parsing constraints |
| T3 | Vector clocks | **1/5** | 21.9 t/s | **FAILED** - couldn't recall rules, didn't solve |
| T3 | Type system | 3/5 | 21.9 t/s | Partial - None issue identified, missed TypeVar |
| T3 | Probabilistic | 4/5 | 22.0 t/s | Correct insight (median=50ms) |

**Tier Averages:**
- T1: **5.0/5** ✓ (baseline competence confirmed)
- T2: **2.3/5** ✗ (below 3+ threshold for oracle_reasoning)
- T3: **2.5/5** ✗ (struggles with complex reasoning)

**Speed:** Consistent 21.9-22.6 t/s across all tests (below 40 t/s target)

**Verdict:** 4B model suitable for **T1-level questions ONLY**. For T2+ difficulty, MUST escalate to oracle_general (30B). Model does NOT meet oracle_reasoning criteria (needs 3+ on T2).

### Speed Notes
- Record baseline (all experts) and optimized (reduced experts) for MoE models
- Record thinking overhead (thinking tokens vs output tokens)

---

## Notes

- Thinking models output reasoning chains - need to decide if workers see this or just the answer
- Token overhead from thinking may be 2-5x output tokens
- Small distilled models may have degraded reasoning vs large originals
- Test both "think step by step" prompting AND native thinking modes
