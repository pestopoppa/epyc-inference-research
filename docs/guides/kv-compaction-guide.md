# KV Cache Compaction — Practical Guide

## What It Does

KV compaction reduces the memory footprint of a model's key-value cache by keeping only the most important token positions. A 5x compaction means a 10,000-token context uses the memory of a 2,000-token context — while preserving answer quality.

## When To Use It

- **Long-context inference** — contexts >4K tokens where KV memory is the bottleneck
- **Multi-slot serving** — free KV memory to serve more concurrent requests
- **After prefill** — compact the prompt's KV before starting generation

## When NOT To Use It

- **Short contexts** (<500 tokens) — overhead not worth it, marginal memory savings
- **During generation** — compact between turns, not mid-generation (the endpoint pauses the slot)

## Quick Start

```bash
# Start server normally (no special flags needed)
llama-server -m model.gguf -c 8192 --port 8080

# Send a long prompt
curl -X POST http://localhost:8080/completion \
  -d '{"prompt": "...", "n_predict": 1, "cache_prompt": true}'

# Compact to 30% of original KV (3.3x compression)
curl -X POST "http://localhost:8080/slots/0?action=compact" \
  -d '{"keep_ratio": 0.3}'

# Generate — uses compacted cache, no quality loss
curl -X POST http://localhost:8080/completion \
  -d '{"prompt": "...", "n_predict": 200, "cache_prompt": true}'
```

## Tuning Parameters

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| `keep_ratio` | 0.5 (2x) | 0.3 (3.3x) | 0.2 (5x) |
| `keep_first` | 10 | 8 | 5 |
| `keep_last` | 20 | 15 | 10 |
| `beta` | 0.05 | 0.1 | 0.1 |

Start with **balanced** settings. Our tests show zero degradation at 5x on factual retrieval and coding tasks. For safety-critical applications, use conservative.

## Model Compatibility

| Architecture | Supported | Notes |
|-------------|-----------|-------|
| Standard attention (Qwen2.5, Llama, etc.) | Yes | Full support, tested at 5x |
| MoE + attention (Qwen3-Coder) | Yes | Full support |
| SSM-hybrid (Qwen3.5) | Yes | Compacts attention layers only; recurrent state preserved |
| ALiBi models | Partial | Beta injection works but not tested extensively |

## How It Works (Simplified)

1. **Score** — Each KV position gets an importance score based on key-vector magnitude (positions with larger key norms attract more attention)
2. **Select** — Keep the highest-scoring positions plus anchors (first/last tokens)
3. **Bias** — Set a small positive beta on kept positions to compensate for removed ones
4. **Restore** — The server adopts the compacted cache as its new state

The entire operation runs in ~1ms for a few hundred cells. It's a metadata operation, not a computation.

## Memory Savings

For Qwen2.5-Coder-32B at Q4_K_M:

| Context | Full KV | 3x Compact | 5x Compact |
|---------|---------|------------|------------|
| 4K tokens | 224 MB | 75 MB | 45 MB |
| 16K tokens | 896 MB | 299 MB | 179 MB |
| 32K tokens | 1.8 GB | 597 MB | 358 MB |

Combined with Hadamard q4_0 quantization (already deployed): **20x total compression** (4x quant x 5x compaction).
