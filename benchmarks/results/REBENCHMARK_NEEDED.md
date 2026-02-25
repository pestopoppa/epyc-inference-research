# Models Needing Re-Benchmark (Corrupted Runs)

Date: 2026-01-15

## CRITICAL: Speculative Decoding Speed Bug (Fixed 2026-01-15)

**All spec_draft speed test results were deleted** due to llama-server timing bug.

### Bug Details
- llama-server reports `predicted_per_second` ~28x inflated for spec decode
- Manual test: Qwen2.5-72B + 0.5B showed 7.055 t/s actual vs ~198 t/s reported
- Root cause: Server's timing calculation is broken for speculative decoding mode

### Fix Applied
- `run_benchmark.py` now uses subprocess mode (llama-speculative CLI) for all spec decode speed tests
- CLI uses correct timing calculation from output_parser.py

### Files Deleted
- All `*spec_draft*.json` files from all runs (439 files)
- These contained inflated speeds that were ~28x too high

### Action Required
Re-run benchmark with spec decode configs to get correct speeds:
```bash
./scripts/benchmark/run_benchmark.py --server-mode
```

---

## Corrupted Runs (Contains llama.cpp loader output instead of model responses)

- thinking_deepseek_r1_distill_qwen_7b_baseline
- thinking_deepseek_r1_distill_llama_8b_baseline

## Action Required
Re-run benchmark script for these models.
