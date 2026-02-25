# Instruction Precision Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Active

---

## Purpose

Benchmark exact instruction following - critical for orchestration where:
- Output must be parseable (JSON only, no explanation)
- Format must match specification exactly
- Length/count constraints must be respected
- Negative instructions ("do NOT") must be followed

---

## Why This Matters for Orchestration

Models that "mostly" follow instructions cause parsing failures:
- "Output only JSON" but model adds "Here's the JSON:"
- "Exactly 3 items" but model gives 4
- "Under 50 words" but model writes 100
- "Do not include X" but model includes X anyway

---

## Quality Rubric

**Scoring (Binary for most tests):**
- 0: Failed to follow instruction
- 1: Followed instruction exactly

**Aggregate scoring:**
- T1: Should be 100% (all models)
- T2: Should be 90%+ (orchestration participants)
- T3: Should be 80%+ (orchestrators/architects)

---

### TIER 1: Basic Format Compliance

#### IP-T1-Q1: JSON Only
```
Prompt: "Convert this to JSON: Name is Alice, age is 30, city is Boston.
Output ONLY the JSON. No explanation, no markdown, no extra text."

Expected: {"name": "Alice", "age": 30, "city": "Boston"}
Fail if: Any text outside the JSON object
```

#### IP-T1-Q2: Exact Count
```
Prompt: "List exactly 3 benefits of exercise. No more, no less."

Expected: Exactly 3 items
Fail if: 2 items, 4+ items, or unnumbered prose
```

#### IP-T1-Q3: Single Word
```
Prompt: "Is 17 a prime number? Answer with only 'yes' or 'no', nothing else."

Expected: "yes" (or "Yes")
Fail if: "Yes, 17 is prime" or any additional text
```

---

### TIER 2: Complex Constraints

#### IP-T2-Q1: Negative Instruction
```
Prompt: "Explain photosynthesis in 2 sentences. Do NOT mention sunlight or sun."

Expected: Valid explanation without "sun" or "sunlight"
Fail if: Contains "sun" or "sunlight"
```

#### IP-T2-Q2: Word Limit
```
Prompt: "Describe machine learning in exactly 20-25 words. Count carefully."

Expected: 20-25 words exactly
Fail if: <20 or >25 words
```

#### IP-T2-Q3: Structured Format
```
Prompt: "List 3 programming languages with their release year.
Format EXACTLY as: LANGUAGE (YEAR)
One per line, no bullets, no numbers."

Expected:
Python (1991)
Java (1995)
JavaScript (1995)

Fail if: Bullets, numbers, different format
```

#### IP-T2-Q4: Multiple Constraints
```
Prompt: "Name 5 countries in Europe.
- Exactly 5 countries
- Alphabetical order
- One per line
- No additional text"

Expected: 5 European countries, A-Z order, one per line
Fail if: Any constraint violated
```

---

### TIER 3: Adversarial Compliance

#### IP-T3-Q1: Resist Elaboration
```
Prompt: "What is 2+2? Reply with only the number."

Expected: "4"
Fail if: "4." or "The answer is 4" or "2+2=4"
```

#### IP-T3-Q2: Maintain Format Under Complexity
```
Prompt: "Analyze this error and respond in EXACTLY this format:
ERROR_TYPE: <type>
ROOT_CAUSE: <one sentence>
FIX: <one sentence>

Error: TypeError: Cannot read property 'map' of undefined"

Expected: Exact format with three labeled lines
Fail if: Different format, extra sections, prose explanation
```

#### IP-T3-Q3: Empty/Null Handling
```
Prompt: "Extract all email addresses from this text: 'Hello world'
If none found, output exactly: NONE
Do not explain or apologize."

Expected: "NONE"
Fail if: "No email addresses found" or any variation
```

#### IP-T3-Q4: Conflicting Preferences
```
Prompt: "Summarize this in one sentence: [long technical paragraph]
The sentence must:
- Be under 15 words
- Start with 'The'
- End with a period
- Not use the word 'system'"

Expected: Sentence meeting all 4 criteria
Fail if: Any criterion violated
```

---

## Scoring Summary

| Tier | Pass Threshold | Critical For |
|------|---------------|--------------|
| T1 | 100% | All orchestration models |
| T2 | 90%+ | Workers and above |
| T3 | 80%+ | Orchestrators/architects |

---

## Automated Scoring

Unlike other rubrics, Instruction Precision can be scored automatically:

```python
def score_json_only(response):
    try:
        json.loads(response.strip())
        return 1 if response.strip().startswith('{') else 0
    except:
        return 0

def score_exact_count(response, target_count):
    items = [l for l in response.strip().split('\n') if l.strip()]
    return 1 if len(items) == target_count else 0

def score_word_limit(response, min_words, max_words):
    count = len(response.split())
    return 1 if min_words <= count <= max_words else 0
```

---

## Benchmark Scripts

```bash
# Single model
./scripts/benchmark/run_instruction_precision_rubric.sh /path/to/model.gguf ModelName dense

# All models
./scripts/benchmark/run_all_instruction_precision_benchmarks.sh
```

Results saved to: `/tmp/claude/instruction_precision_rubric_results/`

---

## Results (fill in after benchmarking)

| Model | T1 (3 tests) | T2 (4 tests) | T3 (4 tests) | Total | Orchestration Ready? |
|-------|--------------|--------------|--------------|-------|---------------------|
| TBD | /3 | /4 | /4 | /11 | |
