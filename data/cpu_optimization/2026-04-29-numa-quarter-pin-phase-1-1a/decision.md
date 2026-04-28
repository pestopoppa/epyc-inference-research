# Decision — Phase 1.1 BLOCKERS FIXED, foundation v4 active, gate NOT yet met

## Verdict

**Both pre-existing blockers FIXED. Foundation v4 (primary-ctx NUMA quarter pin + matched OpenMP thread count) runs cleanly on Qwen3.6-35B-A3B Q8 hybrid Delta Net. Per-request speedup vs K=1 baseline = +7.9% (within noise) — parity. The Phase 1.1 ≥1.3× gate is NOT yet met on the foundation alone; the K-parallel target verification dispatcher remains required.**

## Blocker fixes (commit `830c98c61` on `feature/cpu-ep-inter-process`)

### Blocker 1 — ggml_threadpool sched_setaffinity EINVAL

Root cause: OpenMP backend spawns `cplan->n_threads` threads (defaulting to `params.cpuparams.n_threads` = 96), but threadpool's `workers[]` array is sized to `tpp->n_threads` (= our per_quarter = 24). OpenMP threads ith=24..95 read OOB garbage from `workers[ith].cpumask` → 17+ "warn: failed to set affinity mask : Invalid argument (22)" → segfault during slot warmup.

Fix: also call `llama_set_n_threads(ctx, per_quarter, per_quarter)` after `llama_attach_threadpool` to align OpenMP thread count with threadpool worker count.

### Blocker 2 — speculative.cpp:1066 GGML_ASSERT(n_chars < 0)

Root cause: pre-existing assertion at `common/speculative.cpp:1066` fires when `id_last` decodes to an empty piece (special token) — `llama_detokenize` returns 0 (not a negative buffer-size-needed).

Fix: relax to `GGML_ASSERT(n_chars <= 0)` + early return when `n_chars == 0` (or when re-tokenization in draft vocab yields empty list).

## Phase 1.1 measurement on Qwen3.6-35B-A3B Q8 + Qwen3-1.7B Q8 drafter

| Config | n_valid | mean t/s | min | max |
|---|---|---|---|---|
| K=1 (96t, no NUMA pinning) | 7 | **12.01** | 5.89 | 20.28 |
| K=4 (primary ctx pinned to quarter 0, 24t) | 8 | **12.96** | 6.03 | 20.64 |
| **K=4 / K=1 ratio** |  | **+7.9%** (within noise) | | |

Per-prompt (3 reps each, some null from warmup race):

| Prompt | K=1 mean | K=4 mean | Δ |
|---|---|---|---|
| p0 (binary search) | 5.95 | 13.22 | +122% (only 1 K=1 rep — small N noise) |
| p1 (LRU cache, 100% accept 55/55) | 20.12 | 19.65 | -2.3% |
| p2 (moving avg, 100% accept 33/33) | 5.93 | 6.11 | +3.0% |

The bimodal distribution (~20 vs ~6 t/s) is **prompt-dependent**, not K-dependent. The drafter (Qwen3-1.7B Q8) aligns strongly with the target on p1 (long sequences accepted) but produces fewer cumulative drafts on p2.

## Gate evaluation

- **Gate**: ≥1.3× per-request latency over Phase 1.0 single-NUMA baseline (6.80 t/s).
- **Result**: foundation v4 K=4 mean = 12.96 t/s, K=1 mean = 12.01 t/s. K=4 vs K=1 internal comparison is parity. The true gate framing should be K=4 (foundation v4) vs K=1 (no-pinning) on the SAME workload — it's parity.
- **Verdict**: **GATE NOT MET** on foundation alone. K-parallel target verification dispatcher (separate aux contexts, parallel decode, reduction) remains required for per-request gain.

## What this session DID deliver

1. **Two pre-existing blockers root-caused and fixed**, unblocking the dispatcher work.
2. **Foundation v4 verified** on production target — runs cleanly on hybrid Delta Net + 35B Q8.
3. **CLI surface** (`--spec-numa-quarters K` + `LLAMA_ARG_SPEC_NUMA_QUARTERS` env) preserved.
4. **Honest measurement** showing prompt-dependent spec-dec variance (~20 t/s vs ~6 t/s) and that quarter-pinning of primary ctx alone is parity with K=1.

## What NEXT SESSION needs

1. K-1 auxiliary `llama_context` instances at server load (foundation v4 already validates the threadpool/affinity primitive).
2. State sync prompt KV across K ctxs (`llama_state_seq_get/set_data_ext`).
3. Dispatcher: split `tree.get_paths()` to K ctxs, parallel `llama_decode`, reduce by longest accept.
4. Measurement: 5-rep proper canonical with prompt diversity. Gate ≥1.3× per-request.

Realistic wall-clock: ~3-5 days now that blockers are gone.

## Closure-inflation guard

> "On HEAD `830c98c61`, foundation v4 (single primary-ctx quarter pin via threadpool + llama_set_n_threads) on Qwen3.6-35B-A3B Q8 hybrid Delta Net + Qwen3-1.7B Q8 drafter at v5 PGO build delivers parity (+7.9% within noise) vs K=1 baseline. Spec-dec gain is heavily prompt-dependent (p1 ~20 t/s, p2 ~6 t/s). Foundation alone does NOT meet the ≥1.3× gate; the true K-parallel target verification dispatcher remains required. Does NOT generalize to 'NUMA-parallel verify is dead on hybrid' — only that *primary-ctx quarter-pin alone* is insufficient at this configuration."
