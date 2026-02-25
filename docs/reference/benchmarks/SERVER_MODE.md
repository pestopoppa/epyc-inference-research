# Benchmark Server Mode Optimization

Quick reference for how the benchmark system handles model loading efficiently.

**Summary:** Server mode enables model pre-loading for efficient batch benchmarking.

---

## Key Insight: K is Per-Request

llama-server supports speculative decoding with **per-request K values**:

| Parameter | Scope | Server Restart? |
|-----------|-------|-----------------|
| Target model (`-m`) | Startup | Yes |
| Draft model (`-md`) | Startup | Yes |
| MoE experts (`--override-kv`) | Startup | Yes |
| **K value (`speculative.n_max`)** | **Per-request** | **No** |

This means: start server once with draft model, test all K values (k=8, k=16, k=24) via HTTP without restarting.

---

## Benchmark Flow Example

For a large MoE model (e.g., Qwen3-Coder-480B, 271GB) with spec decode:

| Step | Config | Server State | Restart? |
|------|--------|--------------|----------|
| 1 | baseline | target only | Initial start |
| 2 | moe4 | target + 4 experts | Yes (MoE change) |
| 3 | moe4_spec_k8 | target + draft | Yes (add draft) |
| 4 | moe4_spec_k16 | target + draft | **No** |
| 5 | moe4_spec_k24 | target + draft | **No** |

**Result:** 271GB model stays loaded. Only ~1GB draft added once. K values tested via `speculative.n_max` in HTTP payload.

---

## Implementation Files

| File | What It Does |
|------|--------------|
| `scripts/lib/executor.py` | `ServerManager.start()` accepts `draft_model_path`, `draft_max` |
| `scripts/lib/executor.py` | `ServerManager.run_inference()` accepts `speculative_n_max` |
| `scripts/benchmark/run_benchmark.py` | Tracks `current_server_draft`, restarts only when draft changes |

---

## Fallback: Subprocess Mode

When server mode can't be used (VL models, or server unavailable):
- Uses `llama-speculative` binary via subprocess
- Dynamic timeout: `max(180, int(size_gb * 3) + 120)` seconds
- For 271GB model: ~15 min timeout (accounts for model load time)

---

## When to Use What

| Scenario | Mode | Notes |
|----------|------|-------|
| Baseline/MoE quality tests | Server | Model stays in RAM |
| Spec decode (any K) | Server | K is per-request param |
| VL models | Subprocess | Server doesn't support mmproj |
| Prompt lookup | Subprocess | Different binary (llama-lookup) |

---

*Added: 2026-01-10 after implementing server-side spec decode support*
