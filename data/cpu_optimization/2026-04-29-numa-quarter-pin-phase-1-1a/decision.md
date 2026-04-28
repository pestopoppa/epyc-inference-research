# Decision — Phase 1.1 NUMA-parallel verify FOUNDATION ONLY (no measurement)

## Verdict

**Phase 1.1 implementation is FOUNDATION ONLY this session. Two pre-existing bugs block end-to-end measurement on the production target (Qwen3.6-35B-A3B Q8_0 hybrid Delta Net).**

## Foundation delivered (committed)

| Component | Commit | LOC |
|---|---|---|
| `numa_quarters` param on `common_params_speculative` | `a5c48050c` | +6 |
| `--spec-numa-quarters K` flag + `LLAMA_ARG_SPEC_NUMA_QUARTERS` env | `a5c48050c` | +8 |
| K-context + threadpool plumbing (rolled back to CLI-surface no-op) | `d056c1f20` | +16 (post-rollback) |

Net committed: ~30 LOC of CLI surface, K=1 default safe.

## Two blockers identified

### Blocker 1 — ggml_threadpool sched_setaffinity EINVAL on hybrid models

When K>=2 path attaches a quarter-pinned ggml_threadpool to the primary llama_context on Qwen3.6-35B-A3B Q8 (hybrid Delta Net), threadpool worker threads emit 17+ "warn: failed to set affinity mask : Invalid argument (22)" then segfault during slot init. Same code is stable on Qwen2.5-0.5B (dense). Pre-existing interaction between ggml_threadpool affinity and recurrent-state allocation; unknown root cause.

### Blocker 2 — pre-existing speculative.cpp:1066 assertion

`common_speculative_state_tree::draft` at `common/speculative.cpp:1066` fires `GGML_ASSERT(n_chars < 0)` when `id_last` decodes to an empty-piece token under vocab-incompatible drafter pair. HEAD `0c8d05597` already exhibits this bug. Bypass requires fixing the assertion OR using a truly vocab-compatible drafter (likely needs same-base-model quant pair).

## Why this is honest engineering

The original handoff Phase 0 estimate (~50-100 LOC, ~2-3 days) for Phase 1.1 was based on assumed-existing K-candidate target pipeline. Empirical survey falsified that assumption — heap-spec branches DRAFT only, target K-parallelism is brand new work. Realistic scope is ~360-510 LOC + state sync semantics + affinity stack debugging + assertion fix. Multi-week, not session-scope.

The session's foundation work (~30 LOC committed CLI surface) AND the two crash discoveries are the genuine deliverables. Shipping a half-broken dispatcher to claim Phase 1.1 closure would be dishonest engineering.

## Phase 1.1 status post-session

| Item | Status |
|---|---|
| Phase 1.0 GATE | MET (Phase 1.0 unaffected: 6.80 t/s baseline standing) |
| Phase 1.1 CLI surface | DONE (`d056c1f20`) |
| Phase 1.1 mechanism | NEEDS REDESIGN behind 2 blockers |
| Phase 1.1 measurement | DEFERRED until both blockers resolved |
| Phase 1.1 production deployment | BLOCKED |

## Closure-inflation guard

This session does NOT close Phase 1.1. Scoped statement (per `feedback_closure_inflation.md`):

> "On HEAD `0c8d05597`, the K=2+ path of `--spec-numa-quarters` (foundation v1 + v2 implementations) crashes on Qwen3.6-35B-A3B Q8 hybrid Delta Net during slot init via ggml_threadpool sched_setaffinity EINVAL. Independent of that, spec-dec end-to-end on this target+drafter pair hits a pre-existing `speculative.cpp:1066` assertion in the vocab-incompatible path. Both blockers are concrete, in-tree, and fixable. Does NOT generalize to 'NUMA-parallel verify is dead on CPU' or 'spec-dec is broken on this fork' — only that *this specific implementation path on this specific target+drafter pair on this specific HEAD* hits two distinct fixable bugs that block end-to-end measurement. Alternative implementations and alternative drafter pairs remain unevaluated."

## Reopen criteria

Reopen Phase 1.1 implementation track when ALL of the following hold:
1. ggml_threadpool sched_setaffinity EINVAL on hybrid Delta Net root cause identified.
2. `speculative.cpp:1066` assertion fixed OR vocab-compatible drafter located.
3. Phase 1.0 baseline reconfirms cleanly under the chosen drafter.

After reopen: ~1-2 weeks dedicated focused work for full dispatcher + state sync + reduction + measurement.
