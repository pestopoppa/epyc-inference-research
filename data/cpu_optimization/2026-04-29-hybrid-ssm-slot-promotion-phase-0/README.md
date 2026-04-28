# Hybrid SSM Slot-Promotion — Phase 0 (research-only)

**Date**: 2026-04-29
**Phase**: 0 (falsification probe — no benchmark, research + code-trace only)
**Branch**: `feature/cpu-ep-inter-process` HEAD `0c8d05597`
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`
**Source**: intake-490 (PyTorch SGLang blog Dec 2025) + `hybrid-ssm-slot-promotion-spec-dec.md`

## Hypothesis

Per-candidate slot allocation (S_new = S_parent + Δ) + DFlash-style NUMA-parallel verify breaks the K-token-batched-verification serialization wall on Delta Net hybrids. The 6 closed handoffs (`ssm-hybrid-acceleration.md` et al.) closed under "K-token batched verify = N × single-token cost" — slot-promotion may overturn that.

## Phase 0 GO/NO-GO

GO iff: LOW or MEDIUM risk + ≤800 LOC + ≤2 weeks wall-clock for Phase 1 prototype.

## Methodology (research only)

Step 0.1 — Read intake-490 SGLang blog end-to-end via WebFetch
Step 0.2 — Trace Delta Net state in `src/models/delta-net-base.cpp`, `qwen35moe.cpp`, `src/llama-context.cpp`, `src/llama-memory-recurrent.{h,cpp}`, `common/speculative.cpp`
Step 0.3 — Verify Qwen3.6-35B-A3B-Q8_0 architecture via gguf metadata strings
Step 0.4 — LOC + risk + wall-clock per file

## Verdict

**GO with REVISED scope**: existing fork infrastructure (`llama_memory_seq_cp`, `build_rs`, DySpec heap-spec) ALREADY implements the slot-promotion mechanism semantically. Phase 1 LOC drops from projected 360-635 to ~50-100. The actually-new work is server-side NUMA-pinning of per-candidate verify passes.

See `decision.md` for full LOC breakdown + reopener-gate framing.

## Files

- `decision.md` — full GO verdict with revised LOC + risk + wall-clock
- `system-state.txt` — environment snapshot
- `process-pre.txt`, `process-post.txt` — placeholders (no benchmark)
- `ld_debug.txt` — placeholder
- `results.csv` — placeholder (no benchmark; see Phase 1.0 for measurements)
