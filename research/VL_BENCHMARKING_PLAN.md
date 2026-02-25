# Vision-Language Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Pending (awaiting model downloads)

---

## Current Vision Stack

| Role | Model | Speed | Acceleration |
|------|-------|-------|--------------|
| worker_vision | Qwen2.5-VL-7B-Instruct | 57.1 t/s | Spec Decode (K=8) |
| vision_escalation | Qwen3-VL-30B-A3B-Instruct | ~35 t/s | MoE (4 experts) |
| vision_architect | Qwen3-VL-235B-A22B-Thinking | 7.12 t/s | MoE (2 experts) |

**Vision escalation ladder:**
```
worker_vision (7B, fast, hot)
    ↓
vision_escalation (30B MoE, balanced)
    ↓
vision_architect (235B VL+Thinking, heavy reasoning)
```

**Design decision:** VL models have their own ladder separate from text.
The 235B VL+Thinking is reserved for vision tasks, not used for text-only reasoning.

---

## Models to Benchmark

### Downloaded (Ready)
- [ ] Qwen2.5-VL-7B-Instruct (baseline reference)
- [ ] Qwen3-VL-30B-A3B-Instruct (current escalation)

### Downloading
- [ ] (add models as they complete)

### Candidates to Consider
- Qwen2.5-VL-72B-Instruct (dense, ~45GB)
- Qwen3-VL-2B-Instruct (edge/draft candidate?)
- Qwen3-VL-4B-Instruct (fast worker candidate?)
- Other VL models as available

---

## Benchmark Metrics

### Quality Benchmarks (if available via prompt testing)

| Benchmark | What It Tests | Priority |
|-----------|---------------|----------|
| MMMU | Complex multi-step reasoning on images | HIGH |
| DocVQA | Document text extraction and QA | HIGH |
| MathVista | Mathematical reasoning in visual context | HIGH |
| ChartQA | Chart/graph understanding | MEDIUM |
| VideoMME | Long video comprehension | MEDIUM |
| ScreenSpot | GUI element identification | MEDIUM |
| OCRBench | Text recognition accuracy | LOW |

### Performance Benchmarks (measure on our system)

| Metric | Command/Method | Target |
|--------|----------------|--------|
| Baseline t/s | `llama-bench -m MODEL -t 96 -p 512 -n 128` | Record raw speed |
| MoE optimized t/s | `--override-kv MODEL.expert_used_count=int:4` | For Qwen3 VL only |
| Spec decode t/s | `llama-speculative -md DRAFT --draft-max 16` | For Qwen2.5 VL |
| Spec decode acceptance | Check output for acceptance rate | >20% useful |
| Cold load time | Time from start to first token | <30s for escalation |
| Memory usage | `nvidia-smi` or process memory | Fits in RAM tier |

---

## Quality Rubric (Tiered Difficulty)

**Design principle:** Questions should differentiate models. Easy questions (T1) verify basic competence. Hard questions (T2-3) separate worker from escalation tiers.

**Scoring:**
- 1-2: Wrong or confused
- 3: Partially correct, missing key details
- 4: Correct with minor issues
- 5: Correct with full understanding

**Test Images:** Located in `/mnt/raid0/llm/claude/test_images/vl_rubric/`

---

### TIER 1: Baseline (worker_vision should score 4-5)

#### VL-T1-Q1: Simple OCR
```
Image: text_simple.png (black text on white: "Hello World 123")
Prompt: "What text is shown in this image?"
Expected: "Hello World 123"
```
| Criterion | Weight |
|-----------|--------|
| Exact text match | 70% |
| Correct capitalization | 20% |
| No hallucinations | 10% |

#### VL-T1-Q2: Basic Object Count
```
Image: shapes_basic.png (3 red circles, 2 blue squares)
Prompt: "How many shapes are in this image? List them."
Expected: "5 shapes: 3 red circles, 2 blue squares"
```
| Criterion | Weight |
|-----------|--------|
| Correct total count | 40% |
| Correct per-shape count | 40% |
| Correct colors | 20% |

#### VL-T1-Q3: Simple Scene Description
```
Image: icon_folder.png (standard folder icon)
Prompt: "What does this icon represent?"
Expected: "A folder / file folder / directory"
```
| Criterion | Weight |
|-----------|--------|
| Correct identification | 80% |
| Reasonable description | 20% |

---

### TIER 2: Medium-Hard (worker_vision: 3-4, escalation: 4-5)

#### VL-T2-Q1: Chart Reading
```
Image: chart_bar.png (bar chart with 4 bars: A=10, B=25, C=15, D=20)
Prompt: "What is the value of bar B? Which bar has the highest value?"
Expected: "B = 25. Bar B has the highest value."
```
| Criterion | Weight |
|-----------|--------|
| Correct value for B | 40% |
| Correct highest identification | 40% |
| No value confusion | 20% |

#### VL-T2-Q2: Document Structure
```
Image: doc_invoice.png (simple invoice with: Date, Item, Qty, Price, Total)
Prompt: "Extract the total amount from this invoice."
Expected: Correct total value
```
| Criterion | Weight |
|-----------|--------|
| Correct total extraction | 60% |
| Correct field identification | 30% |
| No hallucinated data | 10% |

#### VL-T2-Q3: Code Screenshot
```
Image: code_python.png (Python function with obvious bug)
Prompt: "What does this code do? Is there a bug?"
Expected: Describes function purpose, identifies bug
```
| Criterion | Weight |
|-----------|--------|
| Correct code reading | 40% |
| Bug identification | 40% |
| Clear explanation | 20% |

---

### TIER 3: Hard (worker_vision: 2-3, escalation: 3-4, architect: 4-5)

#### VL-T3-Q1: Math in Image
```
Image: math_equation.png (handwritten: "2x + 5 = 13, solve for x")
Prompt: "Solve the equation shown in the image."
Expected: "x = 4" with working
```
| Criterion | Weight |
|-----------|--------|
| Correct answer | 50% |
| Shows working | 30% |
| Reads equation correctly | 20% |

#### VL-T3-Q2: Complex Diagram
```
Image: diagram_flowchart.png (decision flowchart with 5 nodes)
Prompt: "Trace the path if input > 10 and flag = true"
Expected: Correct path through flowchart
```
| Criterion | Weight |
|-----------|--------|
| Correct path | 50% |
| Understands conditions | 30% |
| Complete trace | 20% |

#### VL-T3-Q3: Subtle Visual Detail
```
Image: diff_images.png (two similar images with 3 differences)
Prompt: "Find all differences between these two images."
Expected: Lists all 3 differences
```
| Criterion | Weight |
|-----------|--------|
| Finds all differences | 50% |
| Correct descriptions | 30% |
| No false positives | 20% |

#### VL-T3-Q4: Multi-step Visual Reasoning
```
Image: puzzle_grid.png (3x3 grid with pattern, one cell missing)
Prompt: "What shape should go in the empty cell? Explain the pattern."
Expected: Correct shape with pattern explanation
```
| Criterion | Weight |
|-----------|--------|
| Correct answer | 40% |
| Pattern identified | 40% |
| Clear reasoning | 20% |

---

## Rubric Scoring Summary

| Tier | worker_vision Target | vision_escalation Target | vision_architect Target |
|------|---------------------|-------------------------|------------------------|
| T1 (Baseline) | 4-5 | 5 | 5 |
| T2 (Medium) | 3-4 | 4-5 | 5 |
| T3 (Hard) | 2-3 | 3-4 | 4-5 |

**Role assignment based on scores:**
- worker_vision: Must score 4+ on T1, 3+ on T2
- vision_escalation: Must score 5 on T1, 4+ on T2, 3+ on T3
- vision_architect: Must score 5 on T1-T2, 4+ on T3

---

## Acceleration Compatibility Matrix

| Model | MoE Reduction | Spec Decode | Prompt Lookup |
|-------|---------------|-------------|---------------|
| Qwen2.5-VL-7B | No (dense) | Yes (draft_qwen25) | Untested |
| Qwen2.5-VL-72B | No (dense) | Maybe (K=16?) | Untested |
| Qwen3-VL-30B-A3B | Yes (4 experts) | No (MoE) | Untested |
| Qwen3-VL-2B | Check arch | Check | Untested |
| Qwen3-VL-4B | Check arch | Check | Untested |

---

## Decision Criteria

### For worker_vision (fast, interactive)
1. Speed > 40 t/s with acceleration
2. Good DocVQA / OCR performance
3. Handles screenshots and simple images well
4. Memory < 8GB preferred

### For vision_escalation (quality, batch)
1. Speed > 20 t/s acceptable (cold load OK)
2. Strong MMMU / MathVista scores
3. Long video support preferred
4. Memory < 50GB (cold loadable)

### For potential vision_architect (if needed)
1. Best-in-class reasoning
2. Speed secondary concern
3. Memory can be large (split files OK)
4. Qwen3-VL-235B-A22B-Thinking candidate?

---

## Notes

- Qwen3-VL uses MoE architecture → benefits from expert reduction
- Qwen2.5-VL is dense → benefits from speculative decoding
- mmproj files required for all VL models
- Temperature 0.7 recommended for VL spec decode (per quirks)

---

## Benchmark Scripts

### Single Model Testing
```bash
# Dense model (baseline only)
./scripts/benchmark/run_vl_rubric.sh /path/to/model.gguf /path/to/mmproj.gguf ModelName dense

# MoE model (baseline + moe4 configurations)
./scripts/benchmark/run_vl_rubric.sh /path/to/model.gguf /path/to/mmproj.gguf ModelName qwen3vlmoe
```

### Full Suite
```bash
# Run all VL models with appropriate configs
./scripts/benchmark/run_all_vl_benchmarks.sh
```

### Architecture Types
| Arch | Configurations Tested |
|------|----------------------|
| `dense` | baseline |
| `qwen3vlmoe` | baseline, moe4 |

Results are saved with config prefix: `{model}_{config}_{test}.txt`
Output directory: `/tmp/claude/vl_rubric_results/`

---

## Results (fill in after benchmarking)

### Quality Rubric Results

| Model | Quant | Config | Speed | T1 | T2 | T3 | Role |
|-------|-------|--------|-------|----|----|-----|------|
| Qwen3-VL-4B | Q4_K_M | baseline | ~33 t/s | TBD | TBD | TBD | - |
| Qwen2.5-VL-7B | Q4_K_M | baseline | ~20 t/s | TBD | TBD | TBD | - |
| Qwen3-VL-8B | Q4_K_M | baseline | ~21 t/s | TBD | TBD | TBD | - |
| Qwen3-VL-30B-A3B | Q4_K_M | baseline | TBD | TBD | TBD | TBD | - |
| Qwen3-VL-30B-A3B | Q4_K_M | moe4 | TBD | TBD | TBD | TBD | vision_escalation? |

### Speed Benchmarks (Previous)

| Model | Quant | Size | Baseline | Optimized | Best Acceleration | Notes |
|-------|-------|------|----------|-----------|-------------------|-------|
| Qwen2.5-VL-7B | Q4_K_M | ~4GB | 15.28 t/s | **57.1 t/s** | Spec Decode K=8 | **KEEP** - fastest VL |
| Qwen3-VL-8B | Q4_K_M | 4.68 GiB | **24.2 t/s** | N/A | Dense (no MoE) | Slower than 2.5-VL |
| Qwen3-VL-8B | Q8_0 | 8.11 GiB | **10.8 t/s** | N/A | Dense (no MoE) | Too slow |
| Qwen3-VL-30B-A3B | Q4_K_M | TBD | TBD | TBD | MoE 4 experts | TBD |

### Qwen3-VL-8B-Instruct Analysis

**Speed:** Q4_K_M is 2.2x faster than Q8_0 (memory bandwidth limited)

**Comparison with Qwen2.5-VL-7B:**
- Qwen2.5-VL-7B + Spec Decode: 57.1 t/s
- Qwen3-VL-8B Q4_K_M: 24.2 t/s (2.4x slower)

**Recommendation:** Keep Qwen2.5-VL-7B as worker_vision. Qwen3-VL-8B is slower and offers no acceleration (dense model).

**Quirks documented:**
- Dense model (n_expert=0) - no MoE reduction
- Auto-enables interactive mode (has chat template)
- For VL inference, need mmproj file: `mmproj-Qwen3-VL-8B-Instruct-F16.gguf`
