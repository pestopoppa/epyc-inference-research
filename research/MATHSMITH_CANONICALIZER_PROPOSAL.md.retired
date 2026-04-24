# MathSmith as Problem Canonicalizer

**Status:** Proposal
**Date:** 2025-12-20
**Model:** MathSmith-Hard-Problem-Synthesizer-Qwen3-8B (Q8_0)

---

## Key Insight

MathSmith-Hard is **not optimized for solving** — it's optimized for **understanding problem structure**. This makes it ideal as a **semantic normalizer** in a two-stage pipeline.

## The Problem with Solver Models

Solver-trained models (DeepSeek-R1, Qwen-Math, etc.) are dangerous for formalization:

- They **silently assume** missing constraints
- They **choose convenient interpretations**
- They **collapse ambiguity** to proceed
- They often "helpfully" solve along the way

## What MathSmith-Hard Does Instead

- **Surfaces ambiguity** rather than resolving it
- **Names missing assumptions** explicitly
- Prefers **explicit quantifiers**
- Comfortable stopping **before** a solution exists

This is exactly what "predigestion" requires.

---

## Proposed Two-Stage Pipeline

```
[ MathSmith-Hard (3.4 t/s) ]
  informal → formal problem statement
          ↓
[ Solver model (Qwen2.5-Math-7B @ 48 t/s) ]
  formal → solution
```

### Benefits

- Solver sees clean structure
- Ambiguity surfaced early
- Reasoning tokens spent on math, not interpretation
- Higher success rate on hard problems

### Speed Consideration

MathSmith runs at 3.4 t/s (5x slower than expected 8B). However:
- Canonicalization is a **one-time cost** per problem
- Output is typically short (structured spec, not solution)
- Total pipeline time likely still favorable vs. solver struggling with ambiguous input

---

## Prompting Pattern

```
You are given an informal mathematical problem.

Your task is to reformulate it into a precise, formal problem statement
that a symbolic or neural math solver could operate on.

Requirements:
- Make all variables explicit
- Specify domains and quantifiers
- Make all assumptions explicit
- Do NOT solve the problem
- Do NOT simplify or optimize
- Do NOT add new constraints
- If multiple formalizations are possible, list them separately

Output format:
1. Formalized problem statement(s)
2. Explicit list of assumptions made explicit
3. Notes on remaining ambiguity (if any)
```

---

## Use Cases

Particularly valuable for:

- Handwritten or natural-language problems
- Research notes with implicit assumptions
- StackExchange-style questions
- Contest problems with narrative framing
- Geometry problems with implicit constraints
- Asymptotic or "for large n" statements

Also useful when:
- A solver model is "almost right but keeps failing"
- Ambiguity is suspected as the blocker

---

## Safeguards

Because the *Hard* variant deprioritizes consistency:

1. **Explicit constraint**: "Do not add assumptions not already implicit"
2. **Verification pass**: Run a verifier to check formalization preserves intent

---

## Future Work

### Priority: Investigate Slow Speed

MathSmith runs at 3.4 t/s — an 8B Q8_0 model should run at 12-15+ t/s.

**Investigation tasks:**
- [ ] Profile with `perf` to identify bottleneck (compute vs memory vs conversion)
- [ ] Compare GGUF metadata with standard Qwen3-8B
- [ ] Check if mradermacher conversion introduced architecture quirks
- [ ] Test with different thread counts (current: 96 threads may be overkill for 8B)

**Speculative decoding opportunity:**
- [ ] Test Qwen3-0.6B as draft (should be tokenizer-compatible with Qwen3 base)
- [ ] Test Qwen3-1.7B as draft (higher quality, still fast)
- [ ] Current registry forbids spec decode due to "PARD-Qwen3 token mismatch" — but vanilla Qwen3 drafts may work

If spec decode works with Qwen3-0.6B (89 t/s draft):
- Expected: 3-4x speedup → ~10-14 t/s
- Would make canonicalization pipeline much more practical

### Other Items

- [ ] Design formal-spec DSL for structured output
- [ ] Benchmark canonicalization → solve vs. direct solve on hard problems
- [ ] Integrate into orchestration as `math_canonicalizer` role
- [ ] Test with Qwen2.5-Math-72B as solver (7.55 t/s with spec decode)

---

## Registry Entry (When Ready)

```yaml
math_canonicalizer:
  description: Problem formalization/canonicalization (not solving)
  model:
    name: MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q8_0
    path: mradermacher/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B-GGUF/...
  role_type: preprocessor
  output_format: structured_problem_spec
  downstream: [worker_math, math_72b]
  constraints:
    forbid:
      - speculative_decoding  # Token mismatch
    performance_note: 3.4 t/s (acceptable for one-time preprocessing)
```
