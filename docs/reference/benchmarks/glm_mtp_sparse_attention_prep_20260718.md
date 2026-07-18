# GLM-MTP and Sparse Final-Attention Prep - 2026-07-18

Scope: no-inference source prep for the v7 experimental kernel. Production v6 remains frozen.
This note is a prep artifact, not a claim that either acceleration path is implemented.

## Upstream / Current-Tree Verdict

No current official upstream path was found that already supports GLM/GLM-DSA native
MTP or real indexed sparse DSA final attention. Upstream/current tree has generic MTP
metadata and APIs, and Qwen-specific MTP graph implementations, but GLM tail tensors are
still preserved/loaded as unused tensors and GLM graph construction does not dispatch a
decoder-MTP graph.

## Native GLM-MTP

Current state: GLM GGUF metadata/tensors preserve `NEXTN`, but native MTP is scaffold-only.

Observed source shape in `/mnt/raid0/llm/llama.cpp-experimental`:

- `src/models/glm-dsa.cpp:35-37`: reads `LLM_KV_NEXTN_PREDICT_LAYERS` into
  `hparams.n_layer_nextn`, so metadata is present.
- `src/models/glm-dsa.cpp:76-82`: marks every `i >= n_layer` tail block with
  `TENSOR_SKIP | TENSOR_NOT_REQUIRED`.
- `src/models/glm-dsa.cpp:135-145`: creates `layer.nextn.*` tensors under those skipped
  flags.
- `src/models/glm-dsa.cpp:149-150`: always returns the normal `graph`; no
  `LLM_GRAPH_TYPE_DECODER_MTP` branch exists.
- `src/models/glm4-moe.cpp:52-59`: also skips appended NextN blocks.
- `src/models/glm4-moe.cpp:115-124`: creates preserved-but-unused `nextn` tensors.
- `src/models/glm4-moe.cpp:129-130`: always returns the normal `graph`.
- `src/models/qwen35.cpp:96-126`: reference loader pattern that loads tail MTP blocks
  without skip flags.
- `src/models/qwen35.cpp:129-134`: reference `LLM_GRAPH_TYPE_DECODER_MTP` dispatch.
- `src/models/qwen35.cpp:510-644`: reference `graph_mtp` skeleton: token + hidden inputs,
  `hnorm`/`enorm`/concat/`eh_proj`, attention/FFN, `h_nextn`, shared head logits.
- `common/speculative.cpp` plus `src/llama-ext.h`: existing native-MTP driver path should be
  reusable after GLM target graph emits `t_h_nextn` and the GLM MTP context consumes token
  plus hidden inputs.

Minimum implementation gate:

- Tail `NEXTN` tensors load as required for GLM candidate models.
- `glm-dsa` and/or `glm4-moe` dispatch `LLM_GRAPH_TYPE_DECODER_MTP`.
- `--spec-type draft-mtp` reports nonzero draft and accepted-token counters.
- MTP on/off A/B preserves reviewer-quality pass criteria and improves throughput on representative GLM review prompts.

Prep before inference:

- Map GLM tail tensor names/shapes against Qwen35 `load_block_mtp`.
- Decide whether `glm-dsa` needs a DSA-aware MTP graph or whether `glm4-moe` can validate a dense/non-DSA MTP path first.
- Prepare an experimental-only build manifest with commit, `LD_LIBRARY_PATH`, model path, and exact MTP A/B commands.

Implementation sequence:

1. Tensor-contract audit: dump/record GLM tail tensor names and shapes from the actual
   GLM-5.2 GGUF, then compare them to the Qwen35 MTP block fields. This is required before
   changing loader flags because GLM-DSA may not have the same dense-attention tail shape as
   Qwen.
2. Loader split: separate GLM trunk loading from MTP-tail loading as Qwen35 does. Tail tensors
   needed by the candidate MTP graph must not be skipped; optional shared-head/embed tensors
   can remain optional.
3. Graph dispatch: add `LLM_GRAPH_TYPE_DECODER_MTP` dispatch for the GLM architecture under
   test. Do this first for the smallest GLM family that has known-good tail tensor coverage.
4. Graph implementation: start from Qwen35 `graph_mtp`, then replace the attention and FFN
   internals with the matching GLM/DeepSeek32-style block. For GLM-DSA, this likely means an
   MLA/DSA-aware MTP graph, not a direct dense Qwen clone.
5. Driver verification: use `--spec-type draft-mtp` only after the graph exists, and require
   nonzero draft plus accepted-token counters before any speed claim.
6. Quality gate: MTP on/off must preserve the GLM reviewer accept/reject gate. Speed without
   reviewer quality does not advance GLM role admission.

### GLM-5.2 NextN Tensor Contract

Artifact: `docs/data/glm52_nextn_tensor_contract_20260718.json`, generated with
`scripts/benchmark/gguf_tensor_contract.py --contract glm-nextn` against
`/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/`. The contract helper is fail-closed:
future GLM-MTP loader/graph changes should run this preflight first and refuse the port if
required `nextn` tensors or the physical tail-layer invariant are missing.

Result: the GLM-5.2 GGUF is `general.architecture=glm-dsa`, `glm-dsa.block_count=79`,
and `glm-dsa.nextn_predict_layers=1`. The validator passes with physical tail layer
`[78]` and tail group counts `attention=9`, `ffn=9`, `indexer=5`, `nextn=4`,
`other=0`. The appended `blk.78.*` tail is entirely in shard
`GLM-5.2-UD-IQ2_M-00006-of-00006.gguf` and contains 27 tensors.

Required NextN tensors and shapes:

- `nextn.eh_proj.weight [12288, 6144]`
- `nextn.enorm.weight [6144]`
- `nextn.hnorm.weight [6144]`

Optional NextN tensor present:

- `nextn.shared_head_norm.weight [6144]`

Other tail tensor groups:

- Full MLA/DSA attention fields: `attn_q_a`, `attn_q_a_norm`, `attn_q_b`,
  `attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_k_b`, `attn_v_b`, `attn_output`,
  and `attn_norm`.
- Full MoE FFN fields: expert gate/up/down tensors, shared expert tensors,
  `ffn_gate_inp`, `ffn_norm`, and `exp_probs_b`.
- DSA indexer fields: `indexer.attn_q_b`, `indexer.attn_k`, `indexer.k_norm.*`,
  and `indexer.proj`.

Port implication: GLM-MTP is not a trivial dense Qwen35 clone. The tail is a full
GLM-DSA/MLA/MoE block plus NextN projection/norm tensors. The contract does **not**
show separate `nextn.embed_tokens` or `nextn.shared_head_head` tensors, so the first
graph should plan to reuse the main token embedding and output head, as Qwen35 can do
when its optional shared-head tensors are absent. The loader split still must stop
skipping `blk.78.*`, but the graph body should combine the Qwen35 NextN `eh_proj` /
`hnorm` / `enorm` / `h_nextn` pattern with the GLM/DeepSeek32 DSA block internals.

## Real Sparse Final-Attention

Current state: DSA is still dense-mask attention. The lightning indexer selects top-k positions, but final attention still materializes a dense mask and calls full K/V attention.

Observed source shape:

- `src/models/glm-dsa.cpp:24-27` and `src/models/glm-dsa.cpp:103-108`: GLM-DSA loads
  DSA/indexer hparams and tensors.
- `src/models/deepseek32.cpp:227-360`: computes lightning-indexer scores and `top_k`.
- `src/models/deepseek32.cpp:436-438`: passes `top_k` into the DSA attention helper.
- `src/llama-graph.cpp:2823-2847`: fills a full dense `kq_mask_all`, then unsets selected
  top-k rows with `ggml_set_rows`.
- `src/llama-graph.cpp:2850-2853`: retrieves full cached K/V and calls `build_attn_mha`.
- `src/llama-graph.cpp:2384-2517`: `build_attn_mha` still runs normal dense attention or
  flash attention over the full K/V extent; the mask affects softmax eligibility, not the
  amount of K/V traversed.
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

Implementation sequence:

1. CPU oracle: add a tiny fixed-shape test comparing dense masked DSA attention against a
   sparse/indexed reference. Suggested shape: `n_kv=16`, `n_q=2`, `n_head=2`,
   `head_dim=8`, `top_k=4`, fixed top-k indices per query.
2. Graph/API boundary: add a helper that accepts Q, cached K/V, and `top_k`, with a dense
   fallback path. The first implementation may gather/compact selected K/V rows before
   calling existing attention, but the test must prove numerical equivalence to the current
   dense-mask path.
3. Backend work: add the real MI210/HIP path only after the CPU oracle is green. The useful
   backend must avoid materializing full dense masks and avoid traversing full K/V when
   `top_k << n_ctx`.
4. Scaling gate: 32K/64K/128K GLM runs must show latency dominated by selected `top_k`, not
   full prompt length, before calling this real sparse final attention.
5. Quality gate: outputs must match the dense-mask baseline on fixed prompts/seeds before any
   reviewer admission or v7 promotion claim consumes the speedup.
