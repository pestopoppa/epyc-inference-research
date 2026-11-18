# DFlash2 drafter fine-tune pipeline (DF2-FT-1 / DF2-BLOCK16)

Target: Qwen3.8-27B. Drafter: z-lab DFlash2 (block 8 now; block 16 planned). Champion: C14 `eee3da838`.
Pilot plan: `/mnt/raid0/llm/tmp/df2-block16-train-20261118/spec.draft.json` (not published).

| file | role |
|---|---|
| `df2_feature_dump.cpp`, `CMakeLists.txt` | on-policy target feature dump; out-of-tree build against the champion store |
| `df2_dump_format.py` | df2-feat-v1 shard contract: writer, reader, `seal`, `check`, `prep-requests` |
| `df2_model.py` | PyTorch DFlash2 drafter with llama.cpp C14 semantics + CE/KL/selector losses |
| `train_df2.py`, `train_block16.yaml` | offline-feature trainer (reference and Automodel fallback), block-16 config |
| `gguf_to_hf.py` | BF16 drafter GGUF -> HF dir; warm-start without downloading the z-lab repo |
| `export_gguf.sh`, `verify_drafter_gguf.py` | HF -> `convert_hf_to_gguf.py` (DFlash2DraftModel) -> Q8_0, then metadata/tensor gate |
| `smoke_test.py` | CPU smoke: dump format, conv parity, block agnosticism, training steps, resume, export round-trip |

## Facts these tools rely on (C14 `eee3da838`)

- **No llama.cpp change is needed for the dump.** `llama_set_embeddings_layer_inp` and
  `llama_get_embeddings_layer_inp` (`src/llama-ext.h:178,182`) give layer-input hidden states for every token
  of a decode. The DFlash speculative path already uses them (`common/speculative.cpp:918,1003`), and the
  store's `libllama.so` exports them. The tap ids are layer-input ids, i.e. HF `target_layer_ids + 1`
  (`conversion/qwen.py:706-709`). For the 27B that is `[6,20,34,48,62]`. qwen35 records
  `t_layer_inp[il]` only for `il < n_layer` (`src/models/qwen35.cpp:156`), so the final hidden comes from
  `llama_get_embeddings_ith` with pooling NONE.
- **Block 16 adds no parameters.** The drafter has no per-position weights: the MASK embedding is shared, the
  model uses NeoX RoPE, the conv is a kernel-2 in-block shift, and the selector is token-indexed. A
  block-8 checkpoint therefore loads into a block-16 model unchanged. Only `dflash_config.block_size`
  changes. `smoke_test.py` asserts both points: the parameter count is identical, and slots 0-7 still shift
  when 8 MASK rows join the non-causal block. That shift is why the config uses a {8,16} curriculum.
- **Runtime.** C14 reads `dflash.block_size` from the GGUF (default 16 if the key is absent, so
  `verify_drafter_gguf.py` treats absence as a defect). It clamps n_max to `block_size - 1`
  (`common/speculative.cpp:1457-1462`). Graph sizes follow `dflash_block_size`.

## Dependencies

- Available now: torch 2.6 CPU, numpy, safetensors, transformers, gguf and pyyaml in
  `/mnt/raid0/llm/venvs/llama-gguf-convert`. The smoke test runs there.
- Not installed: NeMo Automodel, and any ROCm torch. System `python3` has no torch or numpy. The
  `ml-training` venv is empty.
