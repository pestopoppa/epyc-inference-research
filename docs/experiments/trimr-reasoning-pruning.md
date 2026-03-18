# TrimR Reasoning Pruning Evaluation

**Status**: Planned
**Date**: 2026-03-14
**Paper**: TrimR (arxiv:2505.17155) — training-free verifier-based reasoning token pruning
**Related**: OPSDC (arxiv:2603.05433) — conciseness prompting yields 37% token reduction

## Hypothesis

TrimR-style verifier-based pruning can reduce reasoning token count by 30-50% without accuracy loss on our benchmark suites. The verifier cost (extra inference calls) is offset by reduced output token generation.

## Method

1. **Baseline**: Generate full reasoning traces from Qwen3 models on math/reasoning suites
2. **Think-strip**: Remove `<think>...</think>` blocks entirely (already done in scorer)
3. **TrimR-lite**: Score answer quality with/without individual reasoning paragraphs
4. **Metrics**: Token count, accuracy (via `score_answer()`), latency, verifier overhead

## Suites Under Test

| Suite | #Questions | Scoring | Why |
|-------|-----------|---------|-----|
| math | 100 | exact_match | Clean numeric answers, easy to verify |
| gsm8k | 100 | exact_match | Multi-step arithmetic, reasoning-heavy |
| gpqa | 100 | multiple_choice | Graduate-level QA, needs full reasoning |

## Feasibility Questions

1. **Streaming intercept**: Can we intercept `<think>` blocks in llama.cpp streaming before client delivery?
   - Answer: Likely post-hoc only — llama-server streams token-by-token, no semantic block awareness
   - Alternative: Buffer until `</think>` seen, then decide whether to forward

2. **Verifier model**: Which model judges step quality?
   - Option A: Worker model (Qwen2.5-7B on port 8082) — fast, cheap
   - Option B: Target model itself — more accurate, expensive
   - Recommendation: Start with worker model

3. **Granularity**: Prune at what level?
   - Sentence level: Too fine-grained, high overhead
   - Paragraph level: Good balance (3-8 reasoning blocks per trace)
   - `<think>` block level: Coarsest, lowest overhead

## Results

### Baseline (Full Reasoning)

| Suite | Accuracy | Avg Tokens | Avg Think Tokens |
|-------|----------|-----------|-----------------|
| math | TBD | TBD | TBD |
| gsm8k | TBD | TBD | TBD |
| gpqa | TBD | TBD | TBD |

### Think-Strip (Remove All Reasoning)

| Suite | Accuracy | Avg Tokens | Delta vs Baseline |
|-------|----------|-----------|-------------------|
| math | TBD | TBD | TBD |
| gsm8k | TBD | TBD | TBD |
| gpqa | TBD | TBD | TBD |

### TrimR-Lite (Selective Pruning)

| Suite | Accuracy | Avg Tokens | Pruning Ratio | Verifier Overhead |
|-------|----------|-----------|--------------|-------------------|
| math | TBD | TBD | TBD | TBD |
| gsm8k | TBD | TBD | TBD | TBD |
| gpqa | TBD | TBD | TBD | TBD |

## Verdict

TBD — pending evaluation run.

## Next Steps

- If pruning ratio > 30% with < 2% accuracy drop: integrate into orchestrator post-processing
- If verifier overhead > 50% of generation time: not worth it, stick with conciseness prompting
- If streaming intercept feasible: implement `<think>` buffering in chat pipeline
