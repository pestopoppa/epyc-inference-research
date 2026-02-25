# AMD PACE Native PyTorch Testing - Handoff Document

**Goal**: Test AMD PACE native PyTorch inference (BF16) vs llama.cpp GGUF.

**Status**: READY TO EXECUTE (all dependencies installed, scripts created)

**Last Updated**: 2026-01-04

---

## Quick Start (Copy-Paste Ready)

```bash
# 1. Activate environment
conda activate pace-env

# 2. Run quick speed test (3 prompts, ~5 minutes)
cd /mnt/raid0/llm/claude/scripts/benchmark
python bench_amd_pace.py --quick --model qwen7b

# 3. Run full benchmark (all models, baseline + PARD)
python bench_amd_pace.py

# 4. Run existing legacy test script (single model)
cd /mnt/raid0/llm/claude/scripts/legacy
python run_pard_test.py
```

---

## Clarification: PARD-in-llama.cpp vs AMD PACE Native

### What You ALREADY Tested ✅

**PARD models as draft models in llama.cpp** (GGUF format):
- `PARD-Llama-3.2-1B` Q4_0/Q8_0 → 82-85 t/s as draft
- `PARD-DeepSeek-R1-1.5B` Q5_K_S/Q8_0 → 58-71 t/s as draft
- Used with `llama-speculative` for speculative decoding
- Results: **Meta-Llama-70B + PARD-Llama-1B = 84 t/s**

### What Was NEVER Tested ❌

**AMD PACE native PyTorch** with HuggingFace models:
- Uses BF16 (not quantized)
- Native PARD implementation in PyTorch
- Claims **380 t/s** on Llama 3.1 8B (vs our ~85 t/s with llama.cpp)
- Uses AMD EPYC AVX-512 optimizations

---

## Current Infrastructure (December 11, 2025)

| Component | Status | Location |
|-----------|--------|----------|
| AMD PACE repo | Built | `/mnt/raid0/llm/AMD-PACE/` |
| conda environment | Ready | `pace-env` |
| Test script | Exists | `/mnt/raid0/llm/claude/scripts/legacy/run_pard_test.py` |

### HuggingFace Models (for AMD PACE native)

| Model | Path | Purpose |
|-------|------|---------|
| Qwen2.5-7B-Instruct | `/mnt/raid0/llm/hf/Qwen2.5-7B-Instruct` | Target model |
| Llama-3.1-8B-Instruct | `/mnt/raid0/llm/hf/Llama-3.1-8B-Instruct` | Target model |
| DeepSeek-R1-Distill-Qwen-32B | `/mnt/raid0/llm/hf/DeepSeek-R1-Distill-Qwen-32B` | Target model |
| PARD-Qwen2.5-0.5B | `/mnt/raid0/llm/hf/PARD-Qwen2.5-0.5B` | Draft model |
| PARD-Qwen3-0.6B | `/mnt/raid0/llm/hf/PARD-Qwen3-0.6B` | Draft model |
| PARD-Llama-3.2-1B | `/mnt/raid0/llm/hf/PARD-Llama-3.2-1B` | Draft model |
| PARD-DeepSeek-R1-Distill-Qwen-1.5B | `/mnt/raid0/llm/hf/PARD-DeepSeek-R1-Distill-Qwen-1.5B` | Draft model |

### Never Tested

- AMD PACE native PyTorch execution never ran
- No logs or results from AMD PACE
- Test script exists but was never successfully executed

---

## Updated Implementation Plan

### Phase 1: Verify AMD PACE Works (After Current Benchmark)

**Goal**: Run existing test script once to confirm installation is functional.

```bash
# Activate existing environment
conda activate pace-env

# Run existing test script
cd /mnt/raid0/llm/claude/scripts/legacy
python run_pard_test.py 2>&1 | tee /mnt/raid0/llm/claude/logs/amd_pace_first_test.log
```

**Expected output**: Tokens/sec measurement for Qwen2.5-7B + PARD-Qwen2.5-0.5B

### Phase 2: Create Benchmark Integration Script

**Goal**: Create a script that matches our existing benchmark framework patterns.

**File**: `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py`

Key features:
1. Use same prompts as `run_benchmark.py` for quality comparison
2. Test multiple model configurations (with/without PARD)
3. Output JSON results compatible with `results.py` format
4. Measure: tokens/sec, time-to-first-token, memory usage

**Test configurations**:
| Config | Target | Draft | Notes |
|--------|--------|-------|-------|
| qwen7b_baseline | Qwen2.5-7B | None | No speculation |
| qwen7b_pard | Qwen2.5-7B | PARD-Qwen2.5-0.5B | PARD speculation |
| llama8b_baseline | Llama-3.1-8B | None | No speculation |
| llama8b_pard | Llama-3.1-8B | PARD-Llama-3.2-1B | PARD speculation |
| deepseek32b_pard | DeepSeek-R1-32B | PARD-DeepSeek-1.5B | Large model |

### Phase 3: Comparison Testing

**Goal**: Direct comparison of AMD PACE vs llama.cpp on equivalent models.

| Test | AMD PACE | llama.cpp | Notes |
|------|----------|-----------|-------|
| Qwen2.5-7B | BF16 | Q8_0 GGUF | Closest quality match |
| Qwen2.5-7B + PARD | BF16 + 0.5B | Q8_0 + 0.5B Q8_0 | Spec decode comparison |
| DeepSeek-32B | BF16 + 1.5B | Q4_K_M + 1.5B Q8_0 | Large model |

### Phase 4: Results Documentation

**File**: `/mnt/raid0/llm/claude/research/amd_pace_results.md`

Template:
```markdown
# AMD PACE Benchmark Results

## Test Date: YYYY-MM-DD

## Configuration
- CPU: AMD EPYC 9655 (96 cores)
- RAM: 1.13 TB DDR5-5600
- AMD PACE version: (from git log)
- Torch threads: 96

## Speed Results

| Model | AMD PACE (t/s) | llama.cpp (t/s) | Ratio |
|-------|----------------|-----------------|-------|
| Qwen2.5-7B baseline | TBD | TBD | TBD |
| Qwen2.5-7B + PARD | TBD | TBD | TBD |

## Quality Comparison

Run same benchmark prompts through both backends.

## Decision

[ ] AMD PACE is significantly faster (≥2x) → Consider for production
[ ] llama.cpp remains faster → Continue using llama.cpp
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py` | Create | Benchmark script |
| `/mnt/raid0/llm/claude/research/amd_pace_results.md` | Create | Results document |
| `/mnt/raid0/llm/claude/logs/amd_pace_first_test.log` | Create | First test output |

---

## Setup Status (All Complete ✅)

| Component | Status | Path |
|-----------|--------|------|
| AMD PACE repo | Built | `/mnt/raid0/llm/AMD-PACE/` |
| conda environment | Ready | `pace-env` |
| Dependencies | Installed | transformers 4.51.3, regex, safetensors, etc. |
| Benchmark script | Created | `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py` |

### HuggingFace Models (BF16 Safetensors) - All Verified ✅

| Model | Size | Path |
|-------|------|------|
| Qwen2.5-7B-Instruct | ~15GB | `/mnt/raid0/llm/hf/Qwen2.5-7B-Instruct` |
| PARD-Qwen2.5-0.5B | ~1.3GB | `/mnt/raid0/llm/hf/PARD-Qwen2.5-0.5B` |
| Llama-3.1-8B-Instruct | ~16GB | `/mnt/raid0/llm/hf/Llama-3.1-8B-Instruct` |
| PARD-Llama-3.2-1B | ~2GB | `/mnt/raid0/llm/hf/PARD-Llama-3.2-1B` |
| DeepSeek-R1-32B | ~64GB | `/mnt/raid0/llm/hf/DeepSeek-R1-Distill-Qwen-32B` |
| PARD-DeepSeek-1.5B | ~3GB | `/mnt/raid0/llm/hf/PARD-DeepSeek-R1-Distill-Qwen-1.5B` |

---

## Dependencies Installed (2026-01-02)

All dependencies have been installed. The following were added to `pace-env`:
- `regex` (transformers dependency)
- `safetensors`, `accelerate`, `psutil`, `dill`, `tabulate`
- Downgraded `transformers` to 4.51.3 (required by AMD PACE)

**Verification command (should print "AMD PACE import successful"):**
```bash
conda run -n pace-env python -c "from pace.llm import LLMModel; print('AMD PACE import successful')"
```

---

## Scripts Available

### 1. New Benchmark Script (Recommended)
**Path**: `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py`

Features:
- Tests multiple models (Qwen 7B, Llama 8B, DeepSeek 32B)
- Runs baseline (no speculation) AND PARD speculation
- Uses same prompts as llama.cpp benchmark for fair comparison
- Saves JSON results to `/mnt/raid0/llm/claude/benchmarks/results/`

```bash
# All models, both configs
python bench_amd_pace.py

# Quick test (3 prompts only)
python bench_amd_pace.py --quick

# Specific model
python bench_amd_pace.py --model qwen7b

# Baseline only (no PARD)
python bench_amd_pace.py --baseline-only

# PARD only
python bench_amd_pace.py --pard-only

# Dry run (show what would be tested)
python bench_amd_pace.py --dry-run
```

### 2. Legacy Test Script
**Path**: `/mnt/raid0/llm/claude/scripts/legacy/run_pard_test.py`

Simple single-model test. Good for verifying installation works.

```bash
cd /mnt/raid0/llm/claude/scripts/legacy
python run_pard_test.py
```

---

## Execution Checklist

1. [x] Fix dependencies in pace-env (COMPLETED 2026-01-02)
2. [x] Create `bench_amd_pace.py` with benchmark integration (COMPLETED 2026-01-02)
3. [ ] Run quick test to verify installation: `python bench_amd_pace.py --quick --model qwen7b`
4. [ ] Run full benchmark: `python bench_amd_pace.py`
5. [ ] Compare results with llama.cpp benchmarks
6. [ ] Document results in `/mnt/raid0/llm/claude/research/amd_pace_results.md`
7. [ ] Update model_registry.yaml if AMD PACE is faster

---

## Expected Results & Comparison Points

### AMD PACE Claims (from their benchmarks)
- Llama 3.1 8B: **380 t/s** with PARD speculation
- Significant speedup over baseline HuggingFace inference

### llama.cpp Reference (our measured values)
| Model | Config | Speed | Notes |
|-------|--------|-------|-------|
| Qwen2.5-7B Q8_0 | baseline | ~20-25 t/s | No speculation |
| Qwen2.5-7B Q8_0 | speculative + 0.5B | ~80-100 t/s | With draft model |
| Meta-Llama-70B | PARD-Llama-1B draft | 84 t/s | Large model |

### What to Compare
1. **Baseline speed**: AMD PACE BF16 vs llama.cpp Q8_0 (same model, no speculation)
2. **PARD speed**: AMD PACE PARD vs llama.cpp speculative (same draft model)
3. **Memory usage**: BF16 uses more RAM than Q8, acceptable on this system (1.13 TB)

---

## Decision Criteria

**Adopt AMD PACE for production if:**
- ≥2x faster than llama.cpp Q8 for equivalent quality
- Stable (no crashes during extended testing)
- Quality matches llama.cpp on benchmark suite

**Continue with llama.cpp if:**
- AMD PACE is <2x faster
- Stability issues
- Quality degradation detected

---

## Troubleshooting

### Import Errors
```bash
# If you see "ModuleNotFoundError", reinstall dependencies:
conda activate pace-env
pip install regex safetensors accelerate psutil dill tabulate
pip install transformers==4.51.3
```

### Model Not Found
```bash
# Check if models exist:
ls -la /mnt/raid0/llm/hf/Qwen2.5-7B-Instruct/
ls -la /mnt/raid0/llm/hf/PARD-Qwen2.5-0.5B/
```

### Out of Memory
- Start with smaller models (Qwen 7B before DeepSeek 32B)
- BF16 uses ~2x memory of Q4 quantization
- 7B model needs ~14GB RAM, 32B needs ~64GB

### Slow First Run
- First run compiles kernels, takes several minutes
- Subsequent runs are faster

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| AMD PACE crashes | Test with small model first |
| BF16 uses too much RAM | Start with 7B, scale up gradually |
| Quality differs | Use exact same prompts as llama.cpp benchmark |
| PARD doesn't accelerate | Test baseline first, add PARD incrementally |

---

## Key Finding

**AMD PACE is now READY TO TEST.** As of 2026-01-02:
- ✅ All dependencies installed and verified
- ✅ Benchmark script created (`bench_amd_pace.py`)
- ✅ ALL 6 models downloaded and verified (3 targets + 3 PARD drafts)
- ✅ Import test passes: `from pace.llm import LLMModel`

**Next step**: Run `python bench_amd_pace.py --quick --model qwen7b` to get first speed measurement.

---

## File Locations Summary

| Purpose | Path |
|---------|------|
| Benchmark script | `/mnt/raid0/llm/claude/scripts/benchmark/bench_amd_pace.py` |
| Legacy test script | `/mnt/raid0/llm/claude/scripts/legacy/run_pard_test.py` |
| Results directory | `/mnt/raid0/llm/claude/benchmarks/results/runs/` |
| HuggingFace models | `/mnt/raid0/llm/hf/` |
| AMD PACE repo | `/mnt/raid0/llm/AMD-PACE/` |
| This handoff doc | `/mnt/raid0/llm/claude/research/amd_pace_testing.md` |
| Orchestrator handoff | `/mnt/raid0/llm/claude/research/orchestrator_handoff.md` |

---

## Related: Orchestrator Implementation (2026-01-04)

While AMD PACE testing is pending, orchestrator components were built:

| Component | Status | Tests |
|-----------|--------|-------|
| REPL Environment | Complete | 49 |
| LLM Primitives | Complete (mock mode) | 31 |
| Gate Runner | Complete | 22 |
| Failure Router | Complete | 51 |
| FastAPI API | Complete | 26 |

See `/mnt/raid0/llm/claude/research/orchestrator_handoff.md` for details.

These components work in mock mode without loading models, enabling parallel development.
