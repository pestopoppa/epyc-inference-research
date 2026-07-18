# GLM-MTP and Sparse Final-Attention Prep - 2026-07-18

Scope: no-inference source prep for the v7 experimental kernel. Production v6 remains frozen.

## Native GLM-MTP

Current state: GLM GGUF metadata/tensors preserve `NEXTN`, but native MTP is scaffold-only.

Observed source shape in `/mnt/raid0/llm/llama.cpp-experimental`:

- `src/models/glm-dsa.cpp`: main loop marks `i >= n_layer` tail layers with `TENSOR_SKIP | TENSOR_NOT_REQUIRED`, then creates `layer.nextn.*` tensors under those flags.
- `src/models/glm4-moe.cpp`: same preserved-but-unused `nextn` tensor pattern.
- `src/models/qwen35.cpp`: useful reference. It loads tail MTP blocks without skip flags and dispatches `LLM_GRAPH_TYPE_DECODER_MTP` to `graph_mtp`.
- `common/speculative.cpp`: existing native MTP driver path to use once GLM can build the decoder-MTP graph.

Minimum implementation gate:

- Tail `NEXTN` tensors load as required for GLM candidate models.
- `glm-dsa` and/or `glm4-moe` dispatch `LLM_GRAPH_TYPE_DECODER_MTP`.
- `--spec-type draft-mtp` reports nonzero draft and accepted-token counters.
- MTP on/off A/B preserves reviewer-quality pass criteria and improves throughput on representative GLM review prompts.

Prep before inference:

- Map GLM tail tensor names/shapes against Qwen35 `load_block_mtp`.
- Decide whether `glm-dsa` needs a DSA-aware MTP graph or whether `glm4-moe` can validate a dense/non-DSA MTP path first.
- Prepare an experimental-only build manifest with commit, `LD_LIBRARY_PATH`, model path, and exact MTP A/B commands.

## Real Sparse Final-Attention

Current state: DSA is still dense-mask attention. The lightning indexer selects top-k positions, but final attention still materializes a dense mask and calls full K/V attention.

Observed source shape:

- `src/models/deepseek32.cpp`: constructs/passes lightning-indexer `top_k`.
- `src/llama-graph.cpp`: `llm_graph_context::build_attn(...)` fills a dense `kq_mask_all`, unsets top-k rows with `ggml_set_rows`, then calls `build_attn_mha`.
- `src/llama-kv-cache-dsa.cpp`: owns the main MLA KV cache plus indexer KV cache.

Minimum implementation gate:

- Add a true indexed-attention path that gathers/attends only selected KV rows, with dense-mask fallback.
- CPU/reference checks pass against the current dense-mask output for fixed prompts/seeds.
- 32K/64K/128K context runs show latency scaling with selected `indexer_top_k`, not full context length.
- Quality/reviewer acceptance remains unchanged relative to the dense-mask baseline.

Prep before inference:

- Draft API boundary for indexed DSA attention: selected-row gather, CPU reference, GPU fused path, and fallback guard.
- Add small shape/unit tests before full GLM long-context runs.
- Keep promotion downstream of GLM reviewer quality recovery; do not spend hours on GLM long-context performance if the reviewer lane is still quality-blocked.
