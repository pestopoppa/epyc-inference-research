# SEAL Control Vectors for Concise Reasoning — Experiment Design

**Status**: prep-complete (scripts ready, awaiting model servers)
**Model**: Qwen3-32B (dense attention, `qwen3.cpp` has `build_cvec()`)
**Created**: 2026-04-09

## Hypothesis

Linear control vectors (SEAL baseline) can reduce reasoning verbosity by 15-50% on Qwen3-32B with <2pp accuracy loss on MATH-500 and GPQA-Diamond benchmarks.

## Background

- **SEAL** (linear activation steering): Add a fixed vector to the residual stream at inference time. Deployable today via llama.cpp `--control-vector` / `--control-vector-scaled` flags.
- **FlowSteer** (nonlinear, MLP-based): 5.4x better distributional alignment than SEAL, but requires ODE solver infrastructure not available in llama.cpp. Deferred.
- **Source**: intake-126 (FlowSteer, arXiv:2602.05559), research/deep-dives/flowsteer-concise-reasoning.md

## Method

### 1. Generate Contrastive Pairs
```bash
cd /mnt/raid0/llm/epyc-inference-research/scripts/seal
python generate_pairs.py --output-dir /tmp/seal-pairs --n-pairs 100
```

### 2. Generate Control Vector
```bash
cd /mnt/raid0/llm/llama.cpp
./build/bin/cvector-generator \
    -m /path/to/qwen3-32b.gguf \
    --positive-file /tmp/seal-pairs/positive.txt \
    --negative-file /tmp/seal-pairs/negative.txt \
    --pca-iter 1000 \
    --pca-batch 100 \
    --completions-count 64 \
    --control-vector-layer-start 40 \
    --control-vector-layer-end 60 \
    -o /tmp/seal-concise-qwen3-32b.gguf \
    -ngl 99
```

**Key parameters**:
- Intervention layers: 40-60 (mid-to-late layers, optimal for 32B models)
- PCA: 1000 iterations, batch 100
- Completions: 64 per prompt pair
- Output: F32 control vector in GGUF format

### 3. Evaluate at Multiple Scaling Factors
```bash
# Baseline (no vector)
python eval_cvectors.py --model-port 8080 --baseline-only --output results_baseline.json

# For each scaling factor, restart server with:
# llama-server -m model.gguf --control-vector-scaled /tmp/seal-concise-qwen3-32b.gguf,0.3
python eval_cvectors.py --model-port 8080 --cvector /tmp/seal-concise-qwen3-32b.gguf --scaling 0.3 --output results_0.3.json

# Repeat for 0.5, 0.7
```

### 4. Scaling Factor Selection

| Factor | Expected Effect |
|--------|----------------|
| 0.3 | Mild conciseness — safest, smallest token reduction |
| 0.5 | Moderate — target sweet spot |
| 0.7 | Aggressive — may harm accuracy on hard problems |

## Success Criteria

- **Primary**: >=15% token reduction with <2pp accuracy loss on MATH-500
- **Secondary**: >=30% token reduction with <2pp on GPQA-Diamond
- **Stretch**: >=40% token reduction maintaining accuracy

## Model Compatibility

| Model | `build_cvec()` | Compatible |
|-------|----------------|------------|
| Qwen3-32B (dense) | Yes (qwen3.cpp) | YES |
| Qwen2.5 | Yes (qwen2.cpp) | YES |
| Qwen3.5 (hybrid SSM) | No (qwen35.cpp) | NO |

## Risks

1. **Quality**: Control vectors may reduce accuracy on hard problems disproportionately
2. **GGUF quantization interaction**: Control vectors are F32 but model is quantized — interaction untested
3. **Pair quality**: Low-diversity pairs produce weak/noisy vectors

## Timeline

- Day 1: Generate pairs, run cvector-generator (~2-4 hours for 32B model)
- Day 2: Evaluate at 3 scaling factors, analyze results, document findings

## Related Work

| Reference | Relevance |
|-----------|-----------|
| intake-126 (FlowSteer) | Nonlinear alternative, deferred |
| intake-127 (TrimR) | Complementary: post-generation pruning |
| intake-129 (short-m@k) | Complementary: parallel generation selection |
| reasoning-compression.md Actions 12-13 | Prompt-level brevity (already deployed) |
