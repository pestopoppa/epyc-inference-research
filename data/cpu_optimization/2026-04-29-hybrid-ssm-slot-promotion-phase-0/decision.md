# Hybrid SSM Slot-Promotion — Phase 0 Decision

**Verdict**: **GO with REVISED scope** (Phase 1 LOC ~50-100 instead of projected 360-635; risk LOW-MEDIUM; wall-clock 1 week)

## Summary

Phase 0 falsification probe found that the slot-promotion mechanism cited by intake-490 (PyTorch SGLang blog Dec 2025) is **already structurally implemented in our llama.cpp fork**. Per-sequence Delta Net state allocation, lazy COW seq_cp, and tree-branching state forks are all present. The existing DySpec heap-spec already uses these primitives.

What's actually new in intake-490 vs our existing fork:
1. **MambaRadixCache** (cross-request state-snapshot prefix tree) — out of scope for single-user CPU regime
2. **DFlash-style NUMA-parallel verification** (each candidate on a different NUMA quarter as a single-token decode) — IN scope, ~50-100 LOC server-side scheduling change

## Phase 1 LOC + risk + wall-clock estimate

| File | LOC | Risk | Wall-clock |
|---|---|---|---|
| `src/llama-context.cpp` | 0 | LOW | n/a (per-seq state already allocated by `n_seq_max`) |
| `src/llama-cparams.h` | 0 | LOW | n/a |
| `src/models/delta-net-base.cpp` | 0 | LOW | n/a (build_rs already loads per-seq state) |
| `src/models/qwen35moe.cpp` | 0 | LOW | n/a (line 285 already passes per-seq state) |
| `common/speculative.cpp` | 0 | LOW | n/a (DySpec heap-spec already uses seq_cp for tree) |
| `common/arg.cpp` + `common.{h,cpp}` | ~10-20 | LOW | minor (only if adding `--spec-numa-pin` flag for opt-in) |
| `tools/server/server-context.cpp` | **+50 to +100** | **MEDIUM** | 2-3 days (the actual NEW work: NUMA-pin per-candidate verify pass to a different NUMA quarter) |
| **Total** | **~50-120** | **LOW-MEDIUM** | **~1 week** (Phase 1.0 measurement: 0.5d; Phase 1.1 NUMA-parallel impl: 2-3d; Phase 2 testing: 2d) |

## Critical findings (file:line citations)

1. `src/models/qwen35moe.cpp:254` — `conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs)` — convolution state per-sequence
2. `src/models/qwen35moe.cpp:285` — `state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs)` — Delta Net state per-sequence
3. `src/llama-memory-recurrent.cpp:214-249` — `seq_cp` is metadata-only (lazy COW), NOT a memcpy
4. `common/speculative.cpp:1271` — DySpec heap-spec already calls `llama_memory_seq_cp(mem_dft, fn.seq_id, child_seq, 0, -1)` for tree branching
5. `common/speculative.cpp:1294-1297` — heap-spec already cleans up via `llama_memory_seq_rm` after each round (the equivalent of slot promotion + discard)
6. Qwen3.6-35B-A3B GGUF: `general.architecture = qwen35moe` → same architecture handler as Qwen3.5-35B-A3B → IS hybrid Delta Net (Workstream B applies)

## Phase 1 plan (revised)

1. **Phase 1.0** (~half a day): empirical confirmation. Measure DySpec heap-spec on Qwen3.5-35B-A3B Q4_K_M at v5 PGO with `--draft-p-split=0.05 --draft-max=N` for N in {16, 24, 32}. Acceptance rate + end-to-end throughput vs `--draft-p-split=0` linear baseline. CPU20 bundle.

2. **Phase 1.1** (~2-3 days): if Phase 1.0 shows ≥30% acceptance + ≥0% end-to-end, implement NUMA-parallel candidate verification in `tools/server/server-context.cpp`. ~50-100 LOC.

3. **Phase 2** (existing in handoff): production decision based on Phase 1 results.

## Reopener gates (closure-inflation policy compliance)

The 6 closed SSM-hybrid handoffs (`ssm-hybrid-acceleration.md` et al.) closed under the assumption that **"K-token batched verification = N × single-token cost"**. Our DySpec heap-spec on hybrid Delta Net does NOT use K-token batched verify — each tree node is verified via separate `llama_decode` call with `seq_cp`'d state. **This is precisely the mechanism intake-490 advocates.**

The closure of those 6 handoffs may already be partially superseded by the existing DySpec heap-spec — but this has not been empirically tested on hybrid Delta Net models in our fork. **Phase 1.0 is the first such test.**

If Phase 1.0 shows the existing heap-spec on Qwen3.5/3.6-35B-A3B does NOT achieve ≥30% acceptance / ≥0% end-to-end gain, the closure is scoped to:

> "DySpec heap-spec on Qwen3.5/3.6-35B-A3B-Q4_K_M at v5 PGO build under our current NUMA single-instance regime fails to achieve ≥30% acceptance / ≥0% end-to-end gain. The slot-promotion mechanism (seq_cp + heap-spec) IS structurally implemented but does not deliver on this specific model class at this build. Does NOT generalize to 'all hybrid spec-dec on CPU is dead'. DFlash-style NUMA-parallel verification (Phase 1.1) is gated on Phase 1.0 success and was not tested."

## What this Phase 0 verdict does NOT do

- Does NOT test acceptance rate or throughput on Qwen3.5/3.6-35B-A3B (Phase 1.0 deliverable)
- Does NOT implement NUMA-parallel verify (Phase 1.1 deliverable)
- Does NOT modify any source files
- Does NOT touch `model_registry.yaml` (production state — separate user-confirm gate)
