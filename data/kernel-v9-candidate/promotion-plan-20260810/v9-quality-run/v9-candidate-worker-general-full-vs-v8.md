# V7 Kernel Quality-Gate Report

**Verdict**: PASS: all 2/2 suites within regression threshold (-5.0%).

## Inputs

- Baseline kernel: `production-consolidated-v8` (`/mnt/raid0/llm/llama.cpp/build/bin/llama-server`)
- Candidate kernel: `experimental-v9-dspark-promotion` (`/mnt/raid0/llm/llama.cpp-experimental/build-v9-cpu/bin/llama-server`)
- Model(s): `worker_general gemma q4 + drafter q8`
- Regression threshold: -5.0%
- Min questions per suite: 195

## Gates

| Suite | Baseline Acc | Candidate Acc | Delta | Verdict |
|---|---:|---:|---:|---|
| gpqa | 24.6% | 25.1% | +0.5% | ✓ OK: 25.1% vs baseline 24.6% (delta +0.5%) |
| mmlu_pro | 36.5% | 36.0% | -0.5% | ✓ OK: 36.0% vs baseline 36.5% (delta -0.5%) |

## Summary

- Suites evaluated: 2
- Passed: 2
- Failed: 0
- Missing: 0
