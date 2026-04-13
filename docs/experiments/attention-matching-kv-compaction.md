# Attention Matching KV Cache Compaction

**Status**: Production-ready (merged to `production-consolidated-v3` 2026-04-13)
**Paper**: arxiv:2602.16284 (Zweiger, Fu, Guo, Yoon Kim — MIT)
**Handoff**: `epyc-root/handoffs/active/attention-matching-kv-compaction.md`

## Summary

Attention Matching (AM) compresses the KV cache by selecting the most important token positions and fitting per-position attention biases (beta) to preserve attention output quality. Our implementation adds a native `compact` endpoint to llama-server that achieves 2-5x KV compression with zero quality degradation on factual, coding, and reasoning tasks.

## Key Results

### Compression vs Quality (save/compact/restore pipeline)

| Model | Prompts | 2x | 3x | 5x |
|-------|---------|----|----|-----|
| Qwen2.5-7B f16 | 4 factual/science | 4/4 PASS | 4/4 PASS | 4/4 PASS |
| Coder-32B Q4KM | 5 coding prompts | 5/5 PASS | 5/5 PASS | 5/5 PASS |
| Qwen3.5-35B SSM-hybrid | 1 factual | PASS | — | — |

### Long Context (7B, save/compact/restore)

| Context Length | Tokens | 3x | 5x |
|---------------|--------|----|----|
| Short (~250 words) | 356 | PASS | PASS |
| Medium (~500 words) | 694 | PASS | PASS |
| Long (~1K words) | 1370 | PASS | PASS |
| XLong (~2K words) | 2722 | PASS | PASS |

### Per-Layer Analysis (7B, attention weights via transformers)

| Layer | 2x | 5x | 10x |
|-------|----|----|-----|
| L0 (early) | 1.000 | 1.000 | 0.994 |
| L14 (middle) | 1.000 | 0.878 | 0.768 |
| L27 (deep) | 1.000 | 0.840 | 0.658 |
| **Average** | **1.000** | **0.906** | **0.807** |

Early layers compress well at high ratios. Deep layers need conservative ratios. The production `compact` endpoint uses K-norm importance scoring to select the best positions per layer.

## Architecture

### L1: Beta Bias Kernel
Per-token additive attention bias stored in `llama_kv_cell_ext.beta`. Injected into the `kq_mask` during attention computation — works with both Flash Attention and standard softmax paths. Zero overhead when beta=0 (default for non-compacted entries).

### L4: Native Compact Endpoint
`POST /slots/{id}?action=compact` — self-contained C++ implementation in llama-server:
1. Saves current slot state to memory buffer
2. Parses cell metadata + K/V data
3. Scores positions by K-vector L2 norm (sum across layers)
4. Selects top-k by importance (always keeps first N + last M as anchors)
5. Rebuilds compact state with fewer cells + beta values
6. Restores compact state (server adopts it as ground truth)

Handles SSM-hybrid models by preserving recurrent state tail bytes unchanged.

### State Format
Versioned with per-stream flags field. Bit 0 = ext (beta + 2D positions) present. Backward compatible — old state files without flags are detected and read correctly.

## Usage

### Server Endpoint

```bash
# Prefill a prompt
curl -X POST http://localhost:8080/completion \
  -d '{"prompt": "...", "n_predict": 1, "cache_prompt": true, "id_slot": 0}'

# Compact the KV cache (3.3x compression)
curl -X POST "http://localhost:8080/slots/0?action=compact" \
  -d '{"keep_ratio": 0.3, "beta": 0.1, "keep_first": 8, "keep_last": 15}'

# Continue generation from compacted cache
curl -X POST http://localhost:8080/completion \
  -d '{"prompt": "...", "n_predict": 200, "cache_prompt": true, "id_slot": 0}'
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keep_ratio` | 0.5 | Fraction of KV entries to keep (0.2 = 5x compression) |
| `beta` | 0.1 | Attention bias value set on kept entries |
| `keep_first` | 8 | Always keep first N positions (system prompt anchors) |
| `keep_last` | 15 | Always keep last N positions (recent context anchors) |

### Additional Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /slots/{id}?action=set-beta` | Set beta on individual positions: `{"betas": [{"pos": N, "beta": F}, ...]}` |
| `POST /slots/{id}?action=seq-rm` | Remove position ranges: `{"ranges": [{"p0": N, "p1": M}, ...]}` |

### C API

```c
// Set per-token attention bias
bool llama_memory_set_beta(llama_memory_t mem, llama_seq_id seq_id, llama_pos pos, float beta);
```

## Python State Compactor

For offline compaction (not requiring a running server):

```bash
# Inspect a saved state
python scripts/benchmark/state_compactor.py state.bin --info

# Compact to 50%
python scripts/benchmark/state_compactor.py state.bin compact.bin --keep-ratio 0.5 --beta 0.1

# Compact specific positions
python scripts/benchmark/state_compactor.py state.bin compact.bin --keep-positions 0,1,5,10,20,30
```

## Files

| File | Repo | Purpose |
|------|------|---------|
| `scripts/benchmark/attention_matching.py` | inference-research | AM algorithm port (Python, for analysis) |
| `scripts/benchmark/state_compactor.py` | inference-research | State file parser/compactor (Python) |
| `scripts/benchmark/eval_expected_attention.py` | inference-research | EA benchmark scaffold |
| `src/llama-kv-cells.h` | llama.cpp | `float beta` in cell ext |
| `src/llama-kv-cache.cpp` | llama.cpp | Beta injection + state serialization |
| `tools/server/server-context.cpp` | llama.cpp | `compact`, `set-beta`, `seq-rm` endpoints |
| `tests/test-am-beta-injection.cpp` | llama.cpp | E2E beta injection test |

## Deferred Work

- **L4c: True NNLS scoring** — Current L4b uses K-norm as importance proxy. Full NNLS (attention-weight-based scoring from the paper) requires retaining attention weights during inference (graph modification). Would improve selection at high compression ratios.
- **Online compaction** — Automatically compact during generation when KV exceeds a threshold. The paper shows 6 consecutive 50% compactions preserve reasoning on AIME.
- **Fitted values (C2)** — Current implementation keeps original V values. The paper fits new V values via OLS to better approximate attention output. Would improve quality at high compression.
