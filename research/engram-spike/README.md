# engram-spike

Reference reimplementation of DeepSeek's Engram module (paper arXiv:2601.07372) refactored from `deepseek-ai/Engram`'s 422-line demo for use as a research kit.

Scope: Track B Phase 0 of `handoffs/active/engram-conditional-memory.md` (epyc-root). The goal is to be able to:

1. Load a frozen pretrained Transformer backbone (HF transformers).
2. Splice a paper-faithful Engram layer at chosen depths.
3. Train only the Engram parameters with a two-group AdamW (Engram LR×5, WD=0; backbone frozen).
4. Compare frozen-backbone-Engram against co-trained-Engram on a small proxy to derisk the larger Qwen3.6 retrofit.

Changes from upstream `engram_demo_v1.py`:

- Removed global `engram_cfg` / `backbone_config` — config is passed via dataclass into `Engram(engram_config, backbone_config, layer_id)`.
- Added `engram.init.apply_identity_init(module)` that zero-inits `value_proj.weight`, `value_proj.bias`, and `short_conv.conv.weight`. The paper specifies zero-init for identity preservation at step 0; the upstream demo omits both lines.
- Tokenizer / CompressedTokenizer moved to a separate optional module (no network dependency for unit tests).
- Dropped mocked attention/MoE — this package is the Engram layer only; integration with a real backbone happens via `hooks.py` (forthcoming).
- Added a clean `numpy`-backed hash with a torch-tensor wrapper (avoids the PR-#15 device-mismatch bug while staying numerically identical to the paper).

## Layout

```
engram/
  __init__.py
  config.py        — EngramConfig, BackboneConfig dataclasses
  hash.py          — NgramHashMapping (numpy backend, deterministic)
  modules.py       — ShortConv, MultiHeadEmbedding, Engram nn.Modules
  init.py          — apply_identity_init() + helpers
  tokenizer.py     — CompressedTokenizer (optional; needs HF transformers)
tests/
  test_shapes.py
  test_identity_at_step_zero.py   ← load-bearing invariant
  test_hash_determinism.py
  test_hash_uniformity.py
  test_gradient_flow_frozen_backbone.py
```

## Run tests

```
cd /mnt/raid0/llm/epyc-inference-research/research/engram-spike
python3 -m pytest tests/ -v
```

No GPU required, no network required. All tests use tiny configs (vocab=100, hidden=32, hc_mult=2) and run in seconds on CPU.

## Status

Phase 0a (vendoring + tests) — in progress (2026-05-24).
