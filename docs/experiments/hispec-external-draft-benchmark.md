# HiSpec + External Draft Benchmark Results

**Date**: 2026-03-10
**Branch**: feature/ssm-checkpoint-opt (from production-consolidated-v2)
**Optimization**: Double-buffer pointer swap for SSM checkpoint/restore

## Configuration

- Threads: 96
- Tokens predicted: 256 per prompt
- Prompts: 10 (mix of code + reasoning)
- Draft max: 16

## What's being tested

### Dense (Qwen3-32B)
HiSpec uses intermediate verification to filter bad drafts before full verification.
Intermediate logits at layer N/4 or N/2 evaluate draft tokens cheaply.

### SSM Hybrid (Qwen3.5-9B) — Checkpoint Optimization Validation
Double-buffer optimization eliminates restore memcpy (~144MB) via O(1) pointer swap.
Comparing against previous benchmark to measure improvement.

## Results

```
model          config                  hispec_depth  tps    tokens_generated  generation_sec  draft_accepted  draft_total  acceptance_rate  n_prompts
qwen35-9b-ssm  baseline                0             15.14  2560              169.018         0               0            0                10
qwen35-9b-ssm  external_qwen35_08b     0             10.41  2560              245.811         1385            2439         .5678            10
qwen35-9b-ssm  self_spec_exit8         8             8.61   2560              296.992         1848            2365         .7813            10
qwen35-9b-ssm  external_coder05b       0             15.15  2123              140.106         1225            1955         .6265            10
qwen35-9b-ssm  freeze_ext_qwen35_08b   0             12.06  2075              171.965         1063            2185         .4864            10
qwen35-9b-ssm  freeze_ext_coder05b     0             15.96  2205              138.107         1127            2351         .4793            10
qwen35-9b-ssm  freeze_self_spec_exit8  8             7.69   1988              258.451         1145            2164         .5291            10
```

## Previous SSM results (pre-optimization, 2026-03-10)

| Config | 9B t/s | Delta | Accept Rate |
|--------|--------|-------|-------------|
| baseline | 15.91 | — | — |
| external 0.8B | 10.59 | -33% | 62.5% |
| self-spec exit=8 | 8.83 | -44% | 77.1% |

## Raw data

`data/hsd/hispec_external_20260310_211210.csv`
