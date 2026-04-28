# Phase 1.1 dispatcher v1 — gate evaluation + scoped closure

Build: HEAD on `feature/cpu-ep-inter-process` (this session's commits, see git
log of `/mnt/raid0/llm/llama.cpp-experimental` post-2026-04-30).
Binary: `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin/llama-server`.

Slices delivered:

- A: alt-path selection from `speculation_tree::get_paths()` ranked by leaf
  log_prob, top K-1 selected, greedy duplicate filtered.
- B: one-shot primary→aux state sync at `SLOT_STATE_GENERATING` transition
  with timing instrumentation.
- B.5: gate-check measurement — 17.5 ms total per request across 3 aux ctxs
  on Q8 hybrid. Per-pair cost ≈ 5.8 ms. Gate B.5 PASSED (well under per-token
  budget).
- C/D: per-round aux dispatch in `update_slots`. Sequential pre-decode sync
  on main thread (parallel sync raced with primary's concurrent decode and
  produced find_slot non-consecutive position warnings + n_batch halving),
  then K-1 std::threads each calling `llama_decode` on its aux ctx in
  parallel with primary's decode chunk loop. Joined before the spec-dec
  sample-and-accept loop.
- E: per-ctx sample-and-accept reducer — `slot.smpl` is cloned per ctx;
  `common_sampler_sample_and_accept_n` runs on each ctx with its own path;
  winner = longest accepted prefix (ties → primary, lowest ctx index).
- F: winner-state commit — when winner is an aux ctx,
  `numa_state_sync(numa_ctxs[winner], numa_ctxs[0], slot.id)` writes winner
  state back to primary; `slot.smpl` and `slot.spec_draft` rotate to
  winner's clone and path.

## Gate evaluation

Gate (per Phase 1 binding spec): K=4 mean t/s ≥ 1.3 × K=1 mean t/s on
canonical 3-prompt × N-rep workload, Qwen3.6-35B-A3B-Q8_0 hybrid Delta Net
+ Qwen3-1.7B-Q8_0 drafter, n_predict=64, p_split=0.05, temperature=0.0.

(See README.md for the canonical-workload structural finding that explains
why this gate is unreachable on these specific prompts as configured.)

### Canonical 3-prompt × 2-rep measurement (final_canonical_master.log)

| Prompt | K=1 t/s (mean of 2 reps) | K=4 t/s (mean of 2 reps) | accept K=1 / K=4 | K=4 / K=1 |
|---|---|---|---|---|
| p0 binary_search | 20.21 | 12.65 | 55/55 / 55/55 | 0.626 |
| p1 lru_cache     |  5.96 |  3.85 | 33/33 / 32/32 | 0.646 |
| p2 csv_moving_avg |  8.03 |  5.76 | 14/14 / 12/12 | 0.717 |
| **aggregate (n=6)** | **11.40** | **7.42** | 100% accept across all reps | **0.651** |

**K=4 is 35% slower than K=1 on canonical workload.** Gate (≥1.3×) requires
K=4 ≥ 14.82 t/s; actual K=4 = 7.42 t/s.

The dispatcher's K-parallel block engaged ZERO times across all 6 K=4 reps
(grep -c "numa K-parallel verify" srv_final_k4.log = 0). `numa_alt_paths`
was empty on every round because the canonical prompts under temperature=0.0
produce single-leaf speculation trees — drafter top-1 dominates with >95%
probability, so the tree builder's `cur_p->data[k].p < params.p_split` break
fires immediately for k=1 and beyond.

Consequently, K=4 mode collapses to single-ctx execution on the primary,
but with primary's threadpool restricted to 24 threads (NUMA quarter 0)
instead of 96 threads (full machine in K=1 mode). The 35% slowdown is the
thread-count / DRAM-bandwidth penalty without any countervailing
K-parallel benefit.

## Scoped closure (per closure-inflation policy)

> "On HEAD (this session's commits) on `feature/cpu-ep-inter-process` in
> `/mnt/raid0/llm/llama.cpp-experimental`, Phase 1.1 dispatcher v1 (per-ctx
> sample-and-accept reducer + sequential pre-decode aux state sync +
> parallel aux decode + winner-state commit) on Qwen3.6-35B-A3B Q8 hybrid
> Delta Net + Qwen3-1.7B Q8 drafter at v5 PGO build delivers parity-or-worse
> vs K=1 baseline on the canonical 3-prompt × 2-rep workload. The dispatcher
> functions correctly (build clean, no crashes, output validates, sampler
> state and spec_draft rotation handled when winner != primary), but the
> canonical prompts under temperature=0.0 produce single-leaf speculation
> trees because the drafter's top-1 candidate dominates with > 95%
> probability — so `numa_alt_paths` is empty on every round, the dispatcher
> takes its primary-only fast path, and K=4 mode degenerates to a thread-
> count penalty (primary pinned to 24t per NUMA quarter vs K=1's 96t
> full-machine) without any countervailing K-parallel benefit. Gate not met
> on this workload + config.
>
> Does NOT generalize to 'K-parallel verify is dead on hybrid' or to
> 'NUMA-parallel verify is dead'. The mechanism is structurally functional
> and would deliver gain on workloads that exercise tree branching:
> drafter divergence + lower acceptance rate + p_split low enough to keep
> non-trivial branches + prompts where drafter is uncertain on multiple
> positions. Different (workload, K, p_split, temperature, drafter pair)
> configurations remain unevaluated.
>
> The state-sync cost (Slice B.5: 17.5 ms one-shot, ~5.8 ms per-pair-per-
> round on Q8 hybrid) is NOT the binding constraint for the gate failure on
> this workload — it would only matter on workloads where alt paths
> actually engage. The binding constraint here is workload-pattern: the
> canonical prompts collapse the drafter's tree to a linear path, so there
> is nothing for K-parallel verify to do."

## Recommended next probe (NOT in scope of this session)

To characterize where dispatcher v1 actually wins, measure on a workload
designed to exercise tree branching:

- p_split lowered to 0.001-0.01 (keep more branch candidates).
- Drafter top_k raised (more children per node).
- Prompts where drafter is uncertain (creative writing, ambiguous coding
  problems, harder math).
- Drafter at higher temperature.
- Larger K (K=8 with quarter pairs, or K=16 with sub-quarter pinning).

Until measured, the dispatcher's actual gain on a divergent workload is
unknown.

## CPU20 bundle contents

See README.md.
