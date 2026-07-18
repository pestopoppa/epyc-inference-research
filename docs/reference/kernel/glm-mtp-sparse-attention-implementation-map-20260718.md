# GLM Native MTP and Sparse Final-Attention Implementation Map

Date: 2026-07-18
Scope: zero-inference source audit of `/mnt/raid0/llm/llama.cpp-experimental` only.
Production v6, orchestrator, registries, AutoPilot, and the experimental files already
being edited by the main agent were left untouched.

## Current Verdict

The experimental-v7 tree already contains a buildable GLM-5.2 `glm-dsa` single-NextN
MTP scaffold. The remaining native-MTP work is extension and claim-grade validation:
the graph is explicitly limited to one NextN layer and GLM-4-moe still has no MTP graph
dispatch. No throughput, acceptance, or quality claim should be made before the GLM
reviewer gate `GC-shadow-repair4b -> P-REV-1` closes.

The DSA indexer is real, but final attention is still dense-mask. `top_k` contains
selected positions; it does not compact the MLA KV tensors. The next sparse-attention
slice should therefore introduce an indexed-attention API with a CPU oracle and a
dense fallback before any HIP implementation.

## Files Read

- Active handoffs: `glm52-reviewer-capability-gates.md`,
  `tree-draft-forward-port-plan.md`, `gemma-challenge-kernel-techniques-v7.md`,
  and `cpu-prefill-compute-large-models.md`.
- Experimental kernel: `src/models/glm-dsa.cpp`, `src/models/glm4-moe.cpp`,
  `src/models/deepseek32.cpp`, `src/llama-graph.cpp`, `src/llama-graph.h`,
  `src/llama-context.cpp`, `src/llama-kv-cache-dsa.cpp`,
  `src/llama-kv-cache-dsa.h`, `common/speculative.cpp`, and
  `tools/server/server-context.cpp`.
- Research reference: `docs/reference/benchmarks/glm_mtp_sparse_attention_prep_20260718.md`.

## Native GLM-MTP Anchors

1. **Architecture dispatch**: `src/models/glm-dsa.cpp:155-160` selects
   `graph_mtp` for `LLM_GRAPH_TYPE_DECODER_MTP`; the normal graph remains the fallback.
2. **Single-NextN contract**: `src/models/glm-dsa.cpp:162-189` asserts MLA, at least
   one NextN layer, exactly one NextN layer, and the required tail tensors.
3. **Input contract**: `src/models/glm-dsa.cpp:211-234` accepts token and hidden-state
   inputs. It reuses `model.tok_embd` when no separate NextN embedding exists.
4. **GLM-DSA tail**: `src/models/glm-dsa.cpp:241-447` performs NextN norms/projection,
   DSA indexer scoring/top-k, MLA Q/K/V construction, and calls the shared DSA
   attention builder.
5. **MTP output**: `src/models/glm-dsa.cpp:449-515` runs the dense/MoE FFN tail,
   exports `res->t_h_nextn` at line 503, and reuses the main output head when the
   optional NextN head is absent.
6. **Driver lifecycle**: `common/speculative.cpp:1621-1678` configures the MTP
   context and NextN embedding extraction; `common/speculative.cpp:1764-1851` verifies
   hidden rows; `:1853-2025` drafts and carries hidden state across steps.
7. **Server handoff**: `tools/server/server-context.cpp:3525-3528` requests per-token
   outputs so MTP can mirror `t_h_nextn` into the draft context.
8. **Unfinished family**: `src/models/glm4-moe.cpp:52-59`, `:115-124`, and
   `:129-130` still skip the NextN tail and always dispatch the normal graph. This is a
   separate family task, not required to validate the GLM-5.2 `glm-dsa` scaffold.

### MTP implementation order after the quality gate

- Re-run the tensor-contract preflight against the exact candidate GGUF before loader
  changes. The GLM-5.2 contract is one physical tail block (`blk.78`) with three
  required NextN tensors and an optional shared-head norm; the main embed/output head
  fallback in `glm-dsa.cpp` is therefore intentional.
- Keep `glm-dsa` single-layer support isolated and add multi-layer support only after
  explicit tensor contracts for another model. Do not broaden the existing assertion
  opportunistically.
- For claim-grade A/B, require nonzero draft/accepted counters, coherent output, and
  reviewer-quality parity. The existing eight-token/bounded smokes are build/scaffold
  evidence only.

## Real Sparse Final-Attention Anchors

1. **Indexer selection**: `src/models/deepseek32.cpp:227-360` computes the indexer
   score for every current ubatch and produces `top_k`; the same shape is used by the
   GLM MTP graph at `src/models/glm-dsa.cpp:264-379`.
2. **DSA attention entry point**: `src/llama-graph.h:1127-1140` accepts `top_k` in the
   DSA overload of `build_attn`.
3. **Dense-mask materialization**: `src/llama-graph.cpp:2791-2847` fills a full KV
   mask with `-INFINITY`, uses `ggml_set_rows` to unmask selected positions, and adds
   the original causal mask.
4. **Full-KV attention**: `src/llama-graph.cpp:2849-2853` obtains the entire cached
   K/V tensors and calls `build_attn_mha`.
5. **Dense traversal**: `src/llama-graph.cpp:2384-2517` uses flash attention or
   `ggml_mul_mat(k, q)` plus softmax over the full K/V extent. The mask changes
   eligibility, not K/V traversal.
6. **Cache ownership**: `src/llama-kv-cache-dsa.cpp:30-52` maintains separate MLA and
   Lightning Indexer caches. `src/llama-kv-cache-dsa.h:68-81` exposes those caches;
   there is currently no selected-row MLA gather API.

### Next safe code slice

- Add a new indexed DSA attention helper at the graph/API boundary, preserving the
  current dense-mask path as the default fallback.
- First implementation may compact selected KV rows with existing graph operations,
  but must explicitly handle MLA's cached KV layout and the current top-k shape
  `[n_indexer_top_k, n_tokens]`.
- Add a fixed-shape CPU oracle test in a new disjoint test file: compare dense-mask
  attention with indexed attention for `n_kv=16`, `n_q=2`, `n_head=2`, `head_dim=8`,
  `top_k=4`, including causal masking and duplicate/unsorted indices.
- Only after the oracle is green should the backend path avoid full mask/KV traversal.
  The HIP path should be separately guarded and retain dense fallback for unsupported
  shapes.
- The acceptance criterion is scaling with selected `top_k`, not only numerical
  equality. This requires a later operator-approved long-context bench and is outside
  this zero-inference slice.

## Ready for Build/Bench

- **Build-ready now**: existing experimental-v7 GLM-5.2 single-NextN scaffold, using
  the already modified experimental worktree and matching shared libraries. This is
  source/build readiness only; no build or inference was run here because the request
  explicitly forbids inference and the main agent owns the dirty kernel files.
- **Bench-ready after quality authorization**: native `--spec-type draft-mtp` A/B on
  GLM-5.2 with live draft/accepted counters and reviewer-quality checks.
- **Not bench-ready as an acceleration claim**: DSA `top_k` versus dense-mask. A real
  sparse final-attention implementation and CPU oracle are still required.

## Validation Run

- Read-only `rg`, `nl`, `git status`, and `git log` inspections completed.
- `gitnexus status` was stale initially; `scripts/gitnexus-analyze.sh` was run, and the
  root index now reports up-to-date at commit `c0bb3f7`.
- GitNexus impact queries for the experimental symbols returned `UNKNOWN/not found`
  because the experimental kernel is not one of the indexed repositories; no high or
  critical blast-radius result was available. The source audit above is the direct
  fallback.
- No inference, benchmark, build, AutoPilot restart, production-v6 edit, orchestrator
  edit, registry edit, or existing benchmark-runner edit was performed.
