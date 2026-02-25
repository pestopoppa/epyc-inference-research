# Long Context Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Active

---

## Purpose

Benchmark long context handling capability critical for:
- Large codebase analysis
- Multi-file understanding
- Document summarization
- Needle-in-haystack retrieval

---

## Proposed Long Context Roles

| Role | Purpose | Context Target |
|------|---------|----------------|
| context_worker | Handle 4-8K contexts reliably | 8K tokens |
| context_specialist | Handle 16-32K contexts | 32K tokens |
| context_oracle | Handle 64K+ contexts | 64K+ tokens |

---

## Quality Rubric (Tiered by Context Length)

**Scoring:**
- 1-2: Lost key information, hallucinated, or failed
- 3: Partial recall, missed some details
- 4: Correct with minor omissions
- 5: Complete, accurate, well-organized

---

### TIER 1: Short Context (4K tokens)

#### LC-T1-Q1: Information Retrieval
```
Context: ~3000 tokens of technical documentation
Hidden fact: "The API rate limit is 847 requests per minute"

Prompt: "What is the exact API rate limit mentioned in the documentation?"

Expected: "847 requests per minute"
```
| Criterion | Weight |
|-----------|--------|
| Exact number correct | 60% |
| Correct units | 30% |
| No hallucination | 10% |

#### LC-T1-Q2: Summary with Details
```
Context: ~3500 tokens meeting notes

Prompt: "Summarize the key decisions made. Include all action items with assignees."

Expected: All decisions and action items captured
```
| Criterion | Weight |
|-----------|--------|
| All decisions captured | 40% |
| All action items found | 40% |
| Correct assignees | 20% |

#### LC-T1-Q3: Multi-Section Reference
```
Context: ~4000 tokens with 5 sections

Prompt: "What does Section 3 say about error handling, and how does it relate to Section 5's logging requirements?"

Expected: Accurate cross-reference between sections
```
| Criterion | Weight |
|-----------|--------|
| Section 3 content correct | 35% |
| Section 5 content correct | 35% |
| Relationship identified | 30% |

---

### TIER 2: Medium Context (8-16K tokens)

#### LC-T2-Q1: Needle in Haystack
```
Context: ~12000 tokens of code files
Hidden: "// SECRET: The password is 'correct-horse-battery-staple'"

Prompt: "Find any hardcoded secrets or passwords in the codebase."

Expected: Finds the exact password
```
| Criterion | Weight |
|-----------|--------|
| Secret found | 50% |
| Exact value correct | 30% |
| Location identified | 20% |

#### LC-T2-Q2: Multi-File Analysis
```
Context: 5 Python files (~10K tokens total)
- main.py imports from utils.py
- utils.py has a bug in parse_config()
- config.py defines the schema

Prompt: "Trace the data flow from config loading to main execution. Identify any bugs."

Expected: Correct flow + bug identification
```
| Criterion | Weight |
|-----------|--------|
| Correct data flow | 40% |
| Bug identified | 40% |
| Clear explanation | 20% |

#### LC-T2-Q3: Comprehensive Extraction
```
Context: ~15000 tokens legal document

Prompt: "Extract all dates, monetary amounts, and party names mentioned."

Expected: Complete extraction with no misses
```
| Criterion | Weight |
|-----------|--------|
| All dates found | 30% |
| All amounts found | 30% |
| All parties found | 30% |
| No false positives | 10% |

---

### TIER 3: Long Context (32K+ tokens)

#### LC-T3-Q1: Deep Needle
```
Context: ~40000 tokens of logs/text
Hidden fact buried at ~30K position: "CRITICAL: Server 7 failed at 03:47:22"

Prompt: "What critical server failure is mentioned and at what time?"

Expected: Server 7, 03:47:22
```
| Criterion | Weight |
|-----------|--------|
| Correct server | 40% |
| Correct time | 40% |
| Found despite depth | 20% |

#### LC-T3-Q2: Multi-Document Synthesis
```
Context: 4 separate documents (~35K total)
- Doc A: Q1 financial report
- Doc B: Q2 financial report
- Doc C: Market analysis
- Doc D: Competitor report

Prompt: "Compare Q1 vs Q2 performance and correlate with market/competitor factors from the other documents."

Expected: Integrated analysis across all 4 docs
```
| Criterion | Weight |
|-----------|--------|
| Q1/Q2 comparison accurate | 30% |
| Market factors included | 25% |
| Competitor factors included | 25% |
| Coherent synthesis | 20% |

#### LC-T3-Q3: Codebase Understanding
```
Context: Full small project (~50K tokens)
- 15 Python files
- Complex class hierarchy
- Multiple design patterns

Prompt: "Explain the overall architecture. What design patterns are used? Draw the class hierarchy."

Expected: Accurate architecture description
```
| Criterion | Weight |
|-----------|--------|
| Architecture correct | 35% |
| Design patterns identified | 35% |
| Class hierarchy accurate | 30% |

---

## Rubric Scoring Summary

| Tier | context_worker Target | context_specialist Target | context_oracle Target |
|------|----------------------|--------------------------|----------------------|
| T1 (4K) | 4-5 | 5 | 5 |
| T2 (8-16K) | 3-4 | 4-5 | 5 |
| T3 (32K+) | 2-3 | 3-4 | 4-5 |

---

## Context Length Notes

- T1 tests: ~4000 tokens context
- T2 tests: ~8000-16000 tokens context
- T3 tests: ~32000-50000 tokens context

Models with smaller context windows will fail T3 tests entirely.

---

## Benchmark Scripts

### Single Model Testing
```bash
./scripts/benchmark/run_long_context_rubric.sh /path/to/model.gguf ModelName dense
```

### Full Suite
```bash
./scripts/benchmark/run_all_long_context_benchmarks.sh
```

Results saved to: `/tmp/claude/long_context_rubric_results/`

---

## Results (fill in after benchmarking)

| Model | Context Window | T1 | T2 | T3 | Role |
|-------|---------------|----|----|-----|------|
| TBD | | | | | |
