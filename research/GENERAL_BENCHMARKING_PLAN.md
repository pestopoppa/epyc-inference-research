# General/Instruct Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Active

---

## Purpose

Benchmark general instruction-following capability for `worker_general` and `architect_general` roles. These models handle non-specialized tasks: summarization, reformatting, synthesis, and general Q&A.

---

## Proposed General Roles

| Role | Purpose | Speed Target |
|------|---------|--------------|
| worker_general | Fast general tasks, reformatting, simple synthesis | >40 t/s |
| general_escalation | Complex instructions, longer outputs | >25 t/s |
| architect_general | Heavy synthesis, multi-document analysis | >10 t/s |

---

## Quality Rubric (Tiered Difficulty)

**Scoring:**
- 1-2: Wrong, incomplete, or doesn't follow instructions
- 3: Partially correct, missing key requirements
- 4: Correct with minor issues
- 5: Correct, complete, well-formatted

---

### TIER 1: Baseline (worker_general should score 4-5)

#### G-T1-Q1: Simple Reformatting
```
Prompt: "Convert this to a bullet list:
The meeting covered three topics. First, we discussed the Q3 budget which is $50K over.
Second, we reviewed hiring plans for two engineers. Third, we set the launch date for March 15."

Expected:
- Q3 budget: $50K over
- Hiring plans: 2 engineers
- Launch date: March 15
```
| Criterion | Weight |
|-----------|--------|
| All 3 items captured | 50% |
| Bullet format correct | 30% |
| Concise (not verbose) | 20% |

#### G-T1-Q2: Basic Summarization
```
Prompt: "Summarize in one sentence:
The new caching layer reduced API latency from 200ms to 45ms, improving user experience
significantly. However, it increased memory usage by 2GB per server, requiring us to
upgrade our instance types from m5.large to m5.xlarge, adding $500/month to our AWS bill."

Expected: ~1 sentence capturing latency improvement, memory cost, and AWS cost increase.
```
| Criterion | Weight |
|-----------|--------|
| Captures key trade-off | 50% |
| Single sentence | 30% |
| No hallucinated details | 20% |

#### G-T1-Q3: Extraction
```
Prompt: "Extract all email addresses from this text:
Contact John at john.doe@company.com for sales inquiries. For support, reach out to
support@company.com or help-desk@company.com. Press inquiries: press@external.org"

Expected: john.doe@company.com, support@company.com, help-desk@company.com, press@external.org
```
| Criterion | Weight |
|-----------|--------|
| All 4 emails found | 60% |
| No false positives | 30% |
| Clean format | 10% |

---

### TIER 2: Medium-Hard (worker: 3-4, escalation: 4-5)

#### G-T2-Q1: Structured Output (JSON)
```
Prompt: "Parse this into JSON with fields: name, role, department, start_date
'Sarah Chen joined as Senior Engineer in the Platform team on 2024-03-15.'"

Expected:
{
  "name": "Sarah Chen",
  "role": "Senior Engineer",
  "department": "Platform",
  "start_date": "2024-03-15"
}
```
| Criterion | Weight |
|-----------|--------|
| Valid JSON | 30% |
| All fields correct | 40% |
| Correct data types | 20% |
| No extra fields | 10% |

#### G-T2-Q2: Multi-Step Instructions
```
Prompt: "Process this list:
1. Remove duplicates
2. Sort alphabetically
3. Number each item
4. Add a count at the end

Items: banana, Apple, cherry, BANANA, apple, Date, cherry"

Expected:
1. Apple
2. Banana
3. Cherry
4. Date
Total: 4 items
```
| Criterion | Weight |
|-----------|--------|
| Duplicates removed (case-insensitive) | 30% |
| Alphabetically sorted | 25% |
| Numbered correctly | 25% |
| Count included | 20% |

#### G-T2-Q3: Comparative Summary
```
Prompt: "Compare these two approaches in 2-3 sentences:
Approach A: Microservices - Each feature is a separate service with its own database.
Pros: Independent scaling, isolated failures. Cons: Network overhead, data consistency challenges.

Approach B: Monolith - Single application with shared database.
Pros: Simple deployment, easy data joins. Cons: Scaling limitations, coupled codebase."

Expected: Balanced comparison mentioning trade-offs of both, appropriate use cases.
```
| Criterion | Weight |
|-----------|--------|
| Mentions both approaches | 30% |
| Captures key trade-offs | 40% |
| Suggests appropriate use cases | 20% |
| Concise (2-3 sentences) | 10% |

---

### TIER 3: Hard (worker: 2-3, escalation: 3-4, architect: 4-5)

#### G-T3-Q1: Complex Synthesis
```
Prompt: "Synthesize these three perspectives into a unified recommendation:

Engineering: 'We need 3 months to refactor the auth system properly. Rushing will create tech debt.'
Product: 'Customers are churning due to login issues. We need a fix in 2 weeks.'
Finance: 'Q4 budget is tight. Any solution over $20K needs board approval.'

Provide a concrete recommendation in 3-4 sentences."

Expected: Balanced recommendation acknowledging all constraints, possibly phased approach.
```
| Criterion | Weight |
|-----------|--------|
| Addresses all 3 perspectives | 30% |
| Concrete actionable recommendation | 30% |
| Realistic given constraints | 25% |
| Appropriate length | 15% |

#### G-T3-Q2: Schema Transformation
```
Prompt: "Transform this flat data into nested YAML grouped by department:

employees:
- name: Alice, dept: Engineering, level: Senior
- name: Bob, dept: Sales, level: Junior
- name: Carol, dept: Engineering, level: Junior
- name: Dave, dept: Sales, level: Senior"

Expected:
departments:
  Engineering:
    - name: Alice
      level: Senior
    - name: Carol
      level: Junior
  Sales:
    - name: Bob
      level: Junior
    - name: Dave
      level: Senior
```
| Criterion | Weight |
|-----------|--------|
| Valid YAML | 25% |
| Correct grouping | 35% |
| All data preserved | 25% |
| Clean structure | 15% |

#### G-T3-Q3: Constraint Satisfaction
```
Prompt: "Schedule these meetings given constraints:
- Team sync (60min): Must include Alice, Bob, Carol
- 1:1 Alice-Dave (30min)
- 1:1 Bob-Dave (30min)
- Dave only available 9-11am and 2-4pm
- Alice unavailable 10-11am
- No back-to-back meetings for anyone

Available slots: 9am, 9:30am, 10am, 10:30am, 11am, 2pm, 2:30pm, 3pm, 3:30pm

Output a valid schedule or explain why impossible."

Expected: Valid schedule respecting all constraints, or correct identification of conflict.
```
| Criterion | Weight |
|-----------|--------|
| All constraints respected | 40% |
| All meetings scheduled (if possible) | 30% |
| Clear format | 15% |
| Correct reasoning if impossible | 15% |

#### G-T3-Q4: Multi-Document Consistency
```
Prompt: "These 3 documents describe the same system. Find inconsistencies:

Doc A: 'The API accepts POST requests with JSON body. Rate limit is 100 req/min.'
Doc B: 'Send data via POST with form-encoded body. Rate limit is 100 requests per minute.'
Doc C: 'API endpoint accepts POST. JSON payload required. Rate limited to 1000 req/hour.'

List all inconsistencies found."

Expected:
1. Body format: JSON vs form-encoded (A/C say JSON, B says form-encoded)
2. Rate limit: 100/min = 6000/hour, but C says 1000/hour (inconsistent)
```
| Criterion | Weight |
|-----------|--------|
| Identifies body format inconsistency | 35% |
| Identifies rate limit inconsistency | 35% |
| Clear explanation | 20% |
| No false positives | 10% |

---

## Rubric Scoring Summary

| Tier | worker_general Target | general_escalation Target | architect_general Target |
|------|----------------------|--------------------------|-------------------------|
| T1 (Baseline) | 4-5 | 5 | 5 |
| T2 (Medium) | 3-4 | 4-5 | 5 |
| T3 (Hard) | 2-3 | 3-4 | 4-5 |

**Role assignment based on scores:**
- worker_general: Must score 4+ on T1, 3+ on T2
- general_escalation: Must score 5 on T1, 4+ on T2, 3+ on T3
- architect_general: Must score 5 on T1-T2, 4+ on T3

---

## Models to Benchmark

### Primary Candidates
| Model | Size | Type | Notes |
|-------|------|------|-------|
| Qwen2.5-7B-Instruct | ~4GB | Dense | Fast worker candidate |
| Qwen2.5-14B-Instruct | ~8GB | Dense | Escalation candidate |
| Qwen2.5-32B-Instruct | ~18GB | Dense | Architect candidate |
| Qwen2.5-72B-Instruct | ~40GB | Dense | Heavy architect |
| Qwen3-30B-A3B-Instruct | ~18GB | MoE | Fast escalation (MoE speedup) |

### Acceleration Compatibility
| Model | MoE Reduction | Spec Decode | Notes |
|-------|---------------|-------------|-------|
| Qwen2.5-xB (dense) | No | Yes | Use small draft |
| Qwen3-xB-A3B (MoE) | Yes (4 experts) | No | MoE primary |

---

## Benchmark Scripts

### Single Model Testing
```bash
# Dense model (baseline only)
./scripts/benchmark/run_general_rubric.sh /path/to/model.gguf ModelName dense

# MoE model (baseline + moe4 configurations)
./scripts/benchmark/run_general_rubric.sh /path/to/model.gguf ModelName qwen3moe
```

### Full Suite
```bash
# Run all general models with appropriate configs
./scripts/benchmark/run_all_general_benchmarks.sh
```

### Architecture Types
| Arch | Configurations Tested |
|------|----------------------|
| `dense` | baseline |
| `qwen3moe` | baseline, moe4 |

Results are saved with config prefix: `{model}_{config}_{test}.txt`
Output directory: `/tmp/claude/general_rubric_results/`

### Overnight Suite
```bash
# Run all benchmarks (thinking, coder, vl, general, agentic) + optimization tests
./scripts/benchmark/run_overnight_benchmark_suite.sh

# Run only general benchmark suite
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite general

# Dry run to see what would execute
./scripts/benchmark/run_overnight_benchmark_suite.sh --dry-run
```

---

## Results (fill in after benchmarking)

| Model | Quant | Config | Speed | T1 | T2 | T3 | Role |
|-------|-------|--------|-------|----|----|-----|------|
| TBD | | | | | | | |
