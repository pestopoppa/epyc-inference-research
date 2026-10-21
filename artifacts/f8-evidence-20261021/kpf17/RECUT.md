# KPF-01 re-cut, 2026-10-20

- Champion tip resolved live: `git -C /mnt/raid0/llm/llama.cpp rev-parse ak/champion/llama-cpp-ffc1bac82eec`
  = `e33229348635731cad66dba8d121854024b6f5d2` (unchanged from the dispatch's e33229348).
- Old P1 base `90c12df42` is an ancestor of the tip; the champion is 34 commits past it.
- Worktree `/mnt/raid0/llm/worktrees/kv-prefix-fork-p1r-20261020`, branch `experimental/kv-prefix-fork-p1r-20261020`.
- Old worktree `/mnt/raid0/llm/worktrees/kv-prefix-fork-p1` and branch `experimental/kv-prefix-fork-p1-20261004` untouched.

## Cherry-picks (in order)

| Old sha | New sha | Subject | Result |
|---|---|---|---|
| eb59174a1 | 528b76f99 | llama: attention-only zero-copy seq share and cell accounting (KPF-12, KPF-16 API) | 1 conflict, `common/common.h` |
| 993b73765 | 95b2d32da | server: cross-slot prefix fork, checkpoint_at, fork telemetry | clean auto-merge (server-context.cpp, server-task.*, server-schema.cpp) |
| a17ebe094 | 0b450aac8 | tests: shared-cell KV tests and server slot-fork tests (KPF-17) | 1 conflict, `tools/server/tests/utils.py` |
| 8279261a5 | 778beea19 | server: P2 interface (epoch, hash ladder, ids, /props, message checkpoint_at) | clean auto-merge |
| 857d7ec1f | ad6124c79 | tests: byte-level purge check, natural-text prompts, ... (KPF-17) | clean auto-merge |

## Conflicts

### 1. `common/common.h`, `struct common_params` (eb59174a1)
- Champion side (`7052e26df`, decode-aware prefill budget): added `prefill_budget_decoding` and
  `prefill_budget_target_ms` after `checkpoint_min_step`.
- P1 side: added `slot_fork_min_tokens` at the same anchor.
- Resolution: keep both (budget fields first, then `slot_fork_min_tokens`). Independent POD fields with
  independent argv (`--prefill-budget-*` vs `--slot-fork-min-tokens`); `common/arg.cpp` auto-merged with both
  options present. No semantic interaction.

### 2. `tools/server/tests/utils.py`, `ServerProcess` (a17ebe094)
- Champion side (`7052e26df`): `prefill_budget_decoding` / `prefill_budget_target_ms` fields and their argv emission.
- P1 side: `slot_fork_min_tokens` / `ctx_checkpoints` fields and argv emission, same anchor.
- Resolution: keep both, champion's argv block first. The dataclass fields had already auto-merged.

## Semantic review of the clean auto-merges (server-context.cpp)

The only champion commit touching `tools/server/` since 90c12df42 is `7052e26df` (prefill budget). It changes the
prompt-fill loop bound (`batch.size() < n_prompt_limit` instead of `< n_batch`) and, with a budget active,
narrows `near_prompt_end` to the two canonical near-end offsets. The P1 `checkpoint_at` (KPF-15) code:
- breaks the fill loop *after* pushing the token that ends at a requested position, independent of the bound,
  so a budget-shortened chunk that stops early simply resumes in the next iteration and still breaks at the
  junction;
- creates the pinned checkpoint when a chunk *starts* at a requested position (`ckpt_pinned`), which bypasses
  both `near_prompt_end` and the min-step spacing.
So the two compose: the budget can only split chunks further, never move or drop a junction. No ambiguity;
nothing re-written by hand. The fork itself is taken in `launch_slot_with_task`, which the budget does not touch.
Other champion commits touching files the P1 commits edit (`common/speculative.cpp` DSpark, `src/llama-context.cpp`
profiling/DS41, `src/llama-kv-cache.cpp` 3-line DS41 change) do not overlap the P1 hunks;
`common_speculative_{get,set}_state` signatures are unchanged.

## Integrity of the re-cut (coordinator constraints 1 and 2)

- Added/removed lines of every re-cut commit are identical to the original (`diff` of the `^[+-]` lines of
  `git show old` vs `git show new` = 0 lines for all five; patch-id identical for 8279261a5→778beea19 and
  857d7ec1f→ad6124c79, differing only in context for the other three). So the P2 interface consumed by the
  orchestrator (`/props.slot_fork`, `/slots` `content_epoch`, `prefix_hash`/`fnv1a64_tokens`, `id_slot`/`id_task`
  in OAI timings, `checkpoint_at`, fork timings fields) is byte-for-byte the 8279261a5 code: no name, shape or
  hash change.
- Upstream checkpoint-placement commits 73618f27a (#24176), 0c4fa7a98 (#25472), db94854ff (#24411), 6f3a9f3de
  (#24110), 7c158fbb4 (#24108) are all ancestors of the OLD base 90c12df42 (`git merge-base --is-ancestor`), so
  the original P1 commits were already written on top of them and no new collision arises in the re-cut. The
  only new server-side champion commit is 7052e26df (prefill budget), reviewed above.
- #25472 min-step eviction vs pinned: in the re-cut `create_checkpoint`, the min-step eviction predicate is
  `!it->pinned && it->id_task != id_task && ...`, and the n_ctx_checkpoints cap evicts only the oldest
  UNPINNED one. Pinned `checkpoint_at` checkpoints survive both (code inherited unchanged from 993b73765).
