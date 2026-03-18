# Self-Speculation Benchmark Results (HSD Phase 2b)

**Date**: 2026-03-10
**Branch**: production-consolidated-v2 (HSD Phases 1-3)
**Architecture**: Qwen3.5 = hybrid SSM (Mamba2 + attention, full_attention_interval=4)

## Configuration

- Threads: 96 (EPYC, numactl --interleave=all)
- Tokens predicted: 128 per prompt
- Prompts: 5 (mix of code + thinking)
- Draft max: 16
- Temperature: 0.7

## Qwen3.5-9B (32 layers, Q4_K_M, 5.4GB)

```
┌───────────────────┬──────┬──────────┬──────┬──────────┬────────────────┐
│ Config            │ t/s  │ delta    │ acc  │ gen      │ accept_rate    │
├───────────────────┼──────┼──────────┼──────┼──────────┼────────────────┤
│ baseline (no spec)│15.91 │ —        │ —    │ —        │ —              │
│ external 0.8B     │10.59 │ -33%     │ 346  │ 554      │ 62.5%          │
│ self-spec exit=8  │ 8.83 │ -44%     │ 374  │ 485      │ 77.1%          │
│ self-spec exit=11 │ 7.69 │ -52%     │ 388  │ 589      │ 65.9%          │
│ self-spec exit=16 │ 7.76 │ -51%     │ 334  │ 478      │ 69.9%          │
│ prompt lookup     │  ——  │ SEGFAULT │ —    │ —        │ —              │
└───────────────────┴──────┴──────────┴──────┴──────────┴────────────────┘
```

## Qwen3.5-27B (64 layers, Q4_K_M, 16.7GB)

```
┌───────────────────┬──────┬──────────┬──────┬──────────┬────────────────┐
│ Config            │ t/s  │ delta    │ acc  │ gen      │ accept_rate    │
├───────────────────┼──────┼──────────┼──────┼──────────┼────────────────┤
│ baseline (no spec)│ 4.51 │ —        │ —    │ —        │ —              │
│ external 0.8B     │ 3.51 │ -22%     │ 272  │ (n/a*)   │ (n/a*)         │
│ self-spec exit=16 │ 2.85 │ -37%     │ 454  │ (n/a*)   │ (n/a*)         │
│ self-spec exit=21 │ 2.58 │ -43%     │ 412  │ (n/a*)   │ (n/a*)         │
│ self-spec exit=32 │ 2.71 │ -40%     │ 439  │ (n/a*)   │ (n/a*)         │
│ prompt lookup     │  ——  │ SEGFAULT │ —    │ —        │ —              │
└───────────────────┴──────┴──────────┴──────┴──────────┴────────────────┘
```

*27B run used old draft_n parsing (draft_total=0), acceptance rate not captured.

## Comparison with Existing Best Configs (from Qwen3.5 benchmark handoff)

| Model | Baseline | Best Known Accel | Self-Spec Best | Self-Spec vs Best |
|-------|----------|------------------|----------------|-------------------|
| 9B Q4_K_M | 15.91 t/s | lookup_n5: 25.1 t/s | exit=8: 8.83 t/s | -65% |
| 27B Q4_K_M | 4.51 t/s | spec k32: 13.4 t/s | exit=16: 2.85 t/s | -79% |

## Key Findings

### 1. Self-speculation is a net negative for Qwen3.5 hybrid models

All self-speculation configs are **slower than baseline** by 33-52%. The SSM checkpoint/restore overhead completely negates the accepted token gains, even with acceptance rates of 62-77%.

### 2. The SSM checkpoint overhead is the bottleneck

Qwen3.5 uses a hybrid architecture (Mamba2 SSM + attention every 4th layer). Speculative decoding on hybrid models requires:
- Checkpoint save of recurrent state before speculation
- Checkpoint restore on any rejection
- Non-consecutive position warnings from recurrent memory

This overhead is more expensive than simply decoding tokens sequentially.

### 3. Self-spec acceptance rates are surprisingly high

Self-speculation at exit=8 (25% of layers) achieves 77.1% acceptance rate — higher than external draft at 62.5%. Early exit from the same model produces higher-quality draft tokens than a separate small model. But the speed penalty from loading the same large model as both target and draft negates this advantage.

### 4. Prompt lookup segfaults on Qwen3.5

The `--lookup` flag causes crashes with Qwen3.5 models (both 9B and 27B). This appears to be a bug in the prompt lookup + recurrent memory interaction. Filed for investigation.

### 5. External draft is also slower than baseline

External draft (Qwen3.5-0.8B) at 10.59 t/s is 33% slower than baseline (15.91 t/s) for the 9B model. This confirms the benchmark handoff finding: "Spec decode is a bust for [Qwen3.5 hybrid models] due to SSM checkpoint overhead."

## Implications for HSD

- **Phase 1 (HSD capped branch resampling)**: Still valid — improves acceptance rate regardless of model architecture. But the throughput benefit is only realized on non-hybrid models.
- **Phase 3 (HiSpec hierarchical verification)**: Not beneficial for Qwen3.5 — the intermediate verification adds *more* decode passes, worsening the SSM overhead.
- **Target models for self-speculation**: Dense models only (Qwen3, Qwen2.5, Llama). Hybrid SSM models should use lookup or no acceleration.

## Recommended Next Steps

1. Re-run self-speculation benchmarks on **dense Qwen3-32B** (pure attention, no SSM overhead)
2. Re-run on **Qwen2.5-Coder-32B** (dense, existing production model)
3. Investigate prompt lookup segfault on Qwen3.5
4. Consider gating self-speculation behind architecture detection (skip for hybrid models)

## Raw Data

- 9B: `data/hsd/self_speculation_20260310_125258.csv`
- 27B: `data/hsd/self_speculation_20260310_123437.csv`
