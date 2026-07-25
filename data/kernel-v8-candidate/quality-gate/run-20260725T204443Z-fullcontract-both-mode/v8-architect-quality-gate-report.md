# V7 Kernel Quality-Gate Report

**Verdict**: PASS: all 2/2 suites within regression threshold (-5.0%).

## Inputs

- Baseline kernel: `production-consolidated-v7` (`/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server`)
- Candidate kernel: `production-consolidated-v8` (`/mnt/raid0/llm/llama.cpp/build/bin/llama-server`)
- Model(s): `architect_general Qwen3.5-122B-A10B-UD-Q4_K_M MTP q4/f16`
- Regression threshold: -5.0%
- Min questions per suite: 50

## Gates

| Suite | Baseline Acc | Candidate Acc | Delta | Verdict |
|---|---:|---:|---:|---|
| gpqa | 56.9% | 56.9% | +0.0% | ✓ OK: 56.9% vs baseline 56.9% (delta +0.0%) |
| mmlu_pro | 63.5% | 63.5% | +0.0% | ✓ OK: 63.5% vs baseline 63.5% (delta +0.0%) |

## Summary

- Suites evaluated: 2
- Passed: 2
- Failed: 0
- Missing: 0
