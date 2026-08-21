# SEAL Control Vector Experiment -- Final Results

**Status**: PARKED -- findings archived, not deploying
**Date**: 2026-04-09 to 2026-04-13
**Companion doc**: [seal-concise-reasoning.md](seal-concise-reasoning.md) (original experiment design)

## Summary

SEAL (linear activation steering) control vectors were trained and evaluated across three production model architectures to reduce reasoning verbosity. Results are architecture-dependent: modest token savings on standard/MoE attention models, catastrophic failure on SSM-hybrid. Parking in favor of AM KV compaction which delivers 5x gains with zero degradation and no model-specific constraints.

---

## Key Results by Model

### Worker: Qwen3-Coder-30B-A3B (MoE, Q4_K_M)

| Metric | Value |
|--------|-------|
| Architecture | MoE + standard attention |
| Accuracy | 7/7 baseline -> 7/7 with cvector (no degradation) |
| Token reduction (average) | -7.5% |
| Token reduction (reasoning-heavy) | -12% to -18% |
| Token reduction (code/concise) | 0% (no effect) |

Best results of the three models. Token savings are real but concentrated on reasoning-heavy problems -- code generation and already-concise outputs see no benefit.

**Scale factor sweep (30B)**:

| Scale | Token Reduction | Notes |
|-------|----------------|-------|
| 0.3 | -10.3% | Safe, consistent |
| 0.5 | -28.4% | Saturates here |
| 0.7 | -28.5% | No additional gain over 0.5 |

Saturation at scale=0.5 indicates the intervention has a ceiling effect -- pushing harder does not produce more conciseness.

**Focused accuracy check (30B, 5 math problems)**:

| Condition | Score |
|-----------|-------|
| With cvector | 5/5 |
| Baseline | 4/5 |

Accuracy actually improved on the focused task set, suggesting the cvector may help the model avoid unnecessary reasoning detours on math.

### Coder: Qwen2.5-Coder-32B (Dense, Q4_K_M)

| Metric | Value |
|--------|-------|
| Architecture | Standard dense attention |
| Accuracy | 7/7 baseline -> 7/7 with cvector (no degradation) |
| Token change (average) | +2.2% (neutral) |

Effectively no gain. The coder model's outputs are already concise, so the conciseness vector has nothing to compress.

### Frontdoor: Qwen3.5-35B-A3B (SSM-Hybrid, Gated Delta Net)

| Metric | Value |
|--------|-------|
| Architecture | SSM-hybrid with Gated Delta Net layers |
| Result | **CATASTROPHIC FAILURE** |
| Behavior | cvector at scale=0.5 suppresses generation to 1 token per response |

Gated Delta Net (linear attention) layers respond destructively to activation steering. The control vector does not modulate verbosity -- it collapses the generation entirely. This model architecture is fundamentally incompatible with SEAL-style linear control vectors.

---

## Architecture Compatibility Matrix

| Architecture | Example Model | SEAL Compatible | Notes |
|--------------|---------------|-----------------|-------|
| Standard dense attention | Qwen2.5-Coder-32B | YES | Works, modest gains on reasoning |
| MoE + standard attention | Qwen3-Coder-30B-A3B | YES | Best results (-7.5% to -28% depending on task) |
| SSM-hybrid / Gated Delta Net | Qwen3.5-35B-A3B | **NO** | Linear attention layers respond destructively |

Key insight: SEAL requires standard softmax attention in the residual stream. Any model using linear attention variants (Gated Delta Net, Mamba, RWKV, etc.) should be assumed incompatible until proven otherwise.

---

## Why We Are Parking This

1. **Modest average gains**: -7.5% token reduction on the worker is real but not transformative. Compare to AM KV compaction delivering 5x cache efficiency with zero quality degradation.

2. **Task-dependent**: Savings only appear on reasoning-heavy problems. Code generation and already-concise outputs see 0% improvement. This means conditional routing is required to avoid wasted work.

3. **Conditional routing adds complexity**: To deploy effectively, the orchestrator would need to apply cvector only for reasoning tasks, adding task-type detection logic and per-request server configuration.

4. **SSM incompatibility blocks universal deployment**: The frontdoor model (Qwen3.5 SSM-hybrid) is catastrophically broken with cvectors. Any stack including SSM-hybrid models cannot use a uniform cvector policy.

5. **Better alternatives exist**: AM KV compaction (attention-matching) provides 5x gains with zero degradation and works across all architectures. This is a strictly superior investment.

---

## Artifacts

### Control Vector GGUFs

All stored at `/mnt/raid0/llm/models/`:

| File | Model | Status |
|------|-------|--------|
| `qwen3-coder-30b-seal-concise.gguf` | Worker (30B MoE) | Tested, modest gains |
| `qwen2.5-coder-32b-q4km-seal-concise.gguf` | Coder (32B dense) | Tested, neutral |
| `qwen3.5-35b-a3b-seal-concise.gguf` | Frontdoor (35B SSM) | Tested, BROKEN |
| `qwen2.5-7b-seal-concise.gguf` | 7B (small, for dev) | Generated, not systematically evaluated |
| `qwen2.5-coder-32b-seal-concise.gguf` | Coder (32B, non-Q4KM) | Earlier variant |

### Training Data

| File | Description |
|------|-------------|
| `scripts/seal/positive.txt` | 80 concise-style completions |
| `scripts/seal/negative.txt` | 80 verbose-style completions |
| `scripts/seal/generate_pairs.py` | Pair generation script |

### Evaluation Scripts

| File | Description |
|------|-------------|
| `scripts/seal/eval_cvectors.py` | Per-model cvector evaluation |
| `scripts/benchmark/seal_multi_role_regression_check.py` | Multi-role comparison test (exit 1 on regression) |

### Result Files

All at `benchmarks/results/runs/`:

| File | Description |
|------|-------------|
| `seal-30b-sweep.json` | Scale factor sweep (0.3, 0.5, 0.7) on 30B |
| `seal-30b-accuracy.json` | Focused accuracy test (5 math problems) |
| `seal-30b-ab.json` | A/B comparison runs |
| `seal-coder-32b-q4km.json` | Coder model evaluation |
| `seal-frontdoor-35b.json` | Frontdoor model evaluation (catastrophic) |

---

## Technical Notes

### v3 GGML_OP_GLU Stale Library Issue

During this work, the `cvector-generator` binary crashed on Qwen3/3.5 models due to a stale `libggml-cpu.so.0` from a pre-`ALL_VARIANTS` build. The GLU operation required by Qwen3's architecture was missing. Fix: rebuild cvector-generator against `libggml-cpu-zen4.so` from the current build.

### MTP GGUF Variant Incompatibility

The `Qwen3.5-35B-A3B-MTP` GGUF variant cannot be loaded by cvector-generator. The extra multi-token prediction block at `blk.40` is misclassified as an SSM block, causing a load failure. Use the non-MTP GGUF variant for any future cvector work with this model.

### cvector-generator Build

The generator was rebuilt against the zen4 CPU backend to resolve the GLU issue:
```
libggml-cpu-zen4.so  (not libggml-cpu.so.0)
```

---

## Conditions for Revisiting

This experiment should be revisited if and when:

1. **Reasoning-only worker model**: A new worker model is deployed that uses standard attention (no SSM), ideally MoE with reasoning specialization. The 30B results suggest this would be the highest-value target.

2. **Conditional cvector routing in orchestrator**: The orchestrator gains the ability to conditionally apply `--control-vector-scaled` per request based on `task_type`. This would allow applying cvector only to reasoning tasks where it helps.

3. **Per-category scale tuning**: The ability to tune scale factor per problem category (e.g., math=0.5, code=0, general=0.3) would maximize gains while avoiding the +2.2% penalty on code tasks.

4. **Nonlinear alternatives mature**: FlowSteer (intake-126, arXiv:2602.05559) offers 5.4x better distributional alignment than SEAL but requires ODE solver infrastructure. If llama.cpp gains plugin support for custom inference hooks, FlowSteer becomes viable.

5. **SSM-compatible steering**: Research into activation steering compatible with linear attention / Gated Delta Net architectures. This is an open problem as of April 2026.

---

## Related Work

| Reference | Relevance |
|-----------|-----------|
| intake-126 (FlowSteer) | Nonlinear alternative, 5.4x better alignment, deferred (no ODE solver in llama.cpp) |
| intake-127 (TrimR) | Complementary: post-generation token pruning |
| intake-129 (short-m@k) | Complementary: parallel generation selection |
| [AM KV compaction](../../handoffs/active/attention-matching-kv-compaction.md) | Superior alternative: 5x cache efficiency, zero degradation, architecture-agnostic |
| reasoning-compression.md Actions 12-13 | Prompt-level brevity directives (already deployed) |
