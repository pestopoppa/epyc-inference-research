# Phase 1.1 dispatcher v1 — state-sync cost probe + canonical measurement

## Purpose

Drive the K-parallel candidate verify dispatcher (Phase 1.1) from foundation v5
+ dispatcher v0 (pass-through, HEAD `64df7284b`) to a working v1 with all
sub-slices wired:

1. Slice A — alt-path selection from `speculation_tree::get_paths()`.
2. Slice B — one-shot state sync at `SLOT_STATE_GENERATING` transition.
3. Slice B.5 — measure state-sync cost in isolation (gate-check before
   committing to dispatcher build-out).
4. Slice C/D — K-parallel aux decode dispatch (build K-1 alt-path
   `llama_batch`es; spawn K-1 std::threads after a sequential pre-decode sync;
   join after primary decode).
5. Slice E — per-ctx sample-and-accept reducer (clone sampler per ctx; pick
   winner = longest accepted prefix; ties → primary).
6. Slice F — winner-state commit (sync winner ctx → primary; rotate
   `slot.smpl` and `slot.spec_draft` to winner's path).
7. Slice G — final canonical 3-prompt × 2-rep measurement on
   Qwen3.6-35B-A3B-Q8_0 + Qwen3-1.7B-Q8_0 drafter at v5 PGO build.

## State-sync cost (Slice B.5 gate check)

Probe (run_probe.sh): boot K=4, hit 6 /completion calls, capture per-request
one-shot sync cost.

Result: **62.81 MiB/aux ctx, 17.5 ms total per request across 3 aux ctxs**
on 4096-ctx 32-token n_predict workload.

| Brief estimate | Measured |
|---|---|
| ~330 MiB/ctx state | 62.81 MiB/ctx (5.3× smaller) |
| ~20-25 ms per round | ~17.5 ms one-shot per request |
| Per-round sync cost | ~5.8 ms per primary→aux pair |

Sync cost is well within the per-token gate budget (~64 ms/token at 1.3× of
12 t/s). **Gate B.5 PASSED.**

## Dispatcher v1 architecture

- `numa_select_top_k_alt_paths` (server-context.cpp): walks
  `speculation_tree::get_paths()`, ranks leaves by `log_probs`, returns up to
  K-1 alt paths excluding greedy.
- `slot.numa_alt_paths`: per-slot vector of alt-path token sequences,
  populated in `update_batch` when K>1.
- Pre-decode block in `update_slots` (between batch packing and primary
  decode loop): selects ONE spec-dec slot with non-empty `numa_alt_paths`,
  builds K-1 aux `llama_batch`es, runs **sequential** pre-decode
  primary→aux state sync (parallel sync was racing with primary's decode →
  find_slot non-consecutive position warnings + n_batch halving), then
  spawns K-1 aux decode threads.
- Aux decode runs in parallel with primary decode.
- After chunk loop ends: aux threads joined; spec-dec sample-and-accept
  loop calls `dispatch_numa_parallel_verify`.
- Dispatcher v1: clones `slot.smpl` per ctx, runs sample_and_accept on each
  path, picks winner = longest accepted prefix, syncs winner→primary if
  needed, rotates `slot.smpl` and `slot.spec_draft`.

## Critical structural finding (canonical workload limitation)

The canonical 3-prompt workload (binary_search, lru_cache, csv_moving_avg)
under temperature=0.0 + p_split=0.05 produces **single-leaf trees** because
the drafter's top-1 candidate dominates with > 95% probability on these
simple coding prompts. With single-leaf trees:

- `tree.get_paths()` returns 1 path (greedy).
- `numa_select_top_k_alt_paths` returns empty.
- `slot.numa_alt_paths` is empty.
- The dispatcher's K-parallel block is skipped (no aux work).

Consequently, on canonical workload, K=4 mode degenerates to **single-ctx
decode on the primary, but with primary pinned to 24 threads (NUMA quarter
0) instead of 96 threads (full machine)**. This costs ~25-40% on per-token
decode wall-clock vs K=1.

K-parallel verify cannot help on prompts where the drafter has effectively
100% greedy confidence — all K paths verify identically and converge.
The mechanism would deliver gain on workloads where:

- drafter divergence is realistic (lower acceptance rate),
- p_split is set low enough to expand the tree with non-trivial branches,
- prompts are challenging enough that drafter is uncertain on multiple
  positions.

## Phase 1.1 GATE: NOT MET on canonical workload

(See decision.md for scoped wording per closure-inflation policy.)

## Files

- `run_probe.sh` / `srv_probe.log` — Slice B.5 cost probe
- `run_smoke_v1.sh` / `srv_smoke_v1.log` — dispatcher v1 smoke
- `run_smoke_k1.sh` / `srv_smoke_k1.log` — K=1 baseline reconfirm
- `run_final_canonical.sh` / `final_canonical_master.log` — final 3×2 measurement
- `comp_final_k{1,4}_p{0,1,2}_r{0,1}.json` — per-request completion JSON
- `srv_final_k1.log` / `srv_final_k4.log` — server logs
- `decision.md` — gate evaluation + scoped closure language
