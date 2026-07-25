# V7 Kernel Quality-Gate Report

**Verdict**: PASS: all 2/2 suites within regression threshold (-5.0%).

## Inputs

- Baseline kernel: `production-consolidated-v7` (`/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server`)
- Candidate kernel: `production-consolidated-v8` (`/mnt/raid0/llm/llama.cpp/build/bin/llama-server`)
- Model(s): `worker_general gemma q4 + drafter q8`
- Regression threshold: -5.0%
- Min questions per suite: 50

## Gates

| Suite | Baseline Acc | Candidate Acc | Delta | Verdict |
|---|---:|---:|---:|---|
| gpqa | 24.6% | 24.6% | +0.0% | ✓ OK: 24.6% vs baseline 24.6% (delta +0.0%) |
| mmlu_pro | 36.5% | 36.5% | +0.0% | ✓ OK: 36.5% vs baseline 36.5% (delta +0.0%) |

## Summary

- Suites evaluated: 2
- Passed: 2
- Failed: 0
- Missing: 0
