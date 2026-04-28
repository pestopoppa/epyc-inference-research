# Slot-Promotion Phase 1.1 — Foundation v1→v3 + crash discoveries

**Date**: 2026-04-29
**Target**: Qwen3.6-35B-A3B-Q8_0 (qwen35moe arch = hybrid Delta Net)
**Drafter**: Qwen3-1.7B-Q8_0
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`
**Branch HEAD progression**: `0c8d05597` → `a5c48050c` (foundation v1) → `d056c1f20` (foundation v3 rollback)

## Session purpose

Implement Phase 1.1 NUMA-parallel candidate verify scheduler per `hybrid-ssm-slot-promotion-spec-dec.md`. Phase 1.0 (heap-spec works on hybrid Delta Net) was confirmed end-to-end at 6.80 t/s. Phase 1.1 was scoped at "~50-100 LOC, 2-3 days" in the original handoff Phase 0 estimate.

## What actually happened

Empirical survey of `common/speculative.cpp:1240-1314` falsified the original LOC estimate — heap-spec only branches on the DRAFT side via `seq_cp`. The target verify is single-path. K-way parallel target verify has to be built from scratch (~360-510 LOC realistic).

### Foundation v1 (commit `a5c48050c`, +135 LOC)

Created K=4 auxiliary `llama_context` instances at server load, each pinned to one NUMA quarter via dedicated `ggml_threadpool` with quarter-restricted cpumask. Smoke-tested OK on Qwen2.5-0.5B-Instruct dense.

### Foundation v2 (rolled back, ephemeral)

Reduced to: K=4 parses, but only PRIMARY ctx gets a quarter-pinned threadpool (no aux ctxs). Smoke-tested OK on Qwen2.5-0.5B-Instruct.

### CRASH discovery 1: hybrid Delta Net + threadpool attachment

Both v1 and v2 crashed (segfault) on Qwen3.6-35B-A3B Q8_0 (hybrid Delta Net) during/after slot init, with 17+ repeated:
```
warn: failed to set affinity mask 0x... : Invalid argument (22)
```
from threadpool worker threads' `sched_setaffinity` calls.

Same exact code paths run cleanly on dense Qwen2.5-0.5B. The interaction between ggml_threadpool affinity and the hybrid model's recurrent-state path is not yet understood. Needs investigation in the dispatcher session.

### Foundation v3 rollback (commit `d056c1f20`, -77 LOC)

Reverted to CLI-surface-only:
- `--spec-numa-quarters K` flag parses + env var works (preserved for registry/launcher staging)
- K>=2 takes NO effect — no threadpool/ctx changes
- K=1 default path unchanged from pre-Phase-1.1 binary

### CRASH discovery 2: pre-existing speculative.cpp:1066 vocab assertion

Attempted K=1 baseline reconfirm. Server aborted mid-completion with:
```
slot update_batch: id  2 | task 14 | draft size 10 exceeds max 9, truncating
/mnt/raid0/llm/llama.cpp-experimental/common/speculative.cpp:1066:
  GGML_ASSERT(n_chars < 0) failed
```

This is a PRE-EXISTING bug (HEAD `0c8d05597`, before Phase 1.1 work). The assertion is in the `vocab_cmpt = false` branch of `common_speculative_state_tree::draft` — the probe call `llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false)` returned non-negative (likely because `id_last` decoded to an empty piece). The Phase 1.0 README's "vocab-compatible drafter" claim is likely incorrect; Phase 1.0's prompts presumably never happened to invoke `id_last` on an empty-piece token.

## Net session deliverable

| Artifact | Status |
|---|---|
| `--spec-numa-quarters K` CLI surface | LANDED at `d056c1f20` (no-op for K>=2) |
| K-context + threadpool foundation | NEEDS REDESIGN (crashes on hybrid) |
| Phase 1.1 measurement on Qwen3.6-35B Q8 | NOT POSSIBLE (assertion blocks spec-dec on this drafter pair) |
| Phase 1.0 GATE result | UNAFFECTED (still standing as 6.80 t/s baseline) |

## Next session prerequisites

1. Investigate ggml_threadpool sched_setaffinity EINVAL on hybrid Delta Net + 35B Q8.
2. Fix `speculative.cpp:1066` assertion (relax to `GGML_ASSERT(n_chars <= 0)` or guard `id_last` empty-piece edge case) OR find a truly vocab-compatible drafter.
3. After both unblocks: implement K-parallel dispatcher (~360-510 LOC, ~1-2 weeks).

## Files in this bundle

| File | Description |
|---|---|
| `README.md` | This file |
| `decision.md` | Explicit verdict: FOUNDATION ONLY; redesign required; no measurement |
| `system-state.txt` | numactl --hardware + nproc + branch HEAD |
| `process-pre.txt` | Process snapshot at session start |
| `srv_k4_linear.log` | Server log showing foundation v1 hybrid crash |
| `srv_k4_safe_smoke.log` | Server log showing foundation v3 dense success |
| `srv_k1_baseline.log` | Server log showing speculative.cpp:1066 crash |
| `comp_k1_baseline_p*_r*.json` | 9 completions captured before assertion fired |
| `run_phase11a.sh` | Bench script for foundation v1/v2 attempt |
| `run_baseline_reconfirm.sh` | Bench script for foundation v3 baseline reconfirm attempt |
| `bench_master.log`, `baseline_reconfirm.log` | Bench wrapper logs |
