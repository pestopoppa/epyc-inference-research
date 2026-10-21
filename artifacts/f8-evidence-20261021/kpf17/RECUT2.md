# KPF re-cut 2 onto dd3d9721c (slotdraft #26004 + #30574), 2026-10-20

Base: `experimental/slotdraft-30574-20261020` @ dd3d9721c (champion e33229348 + 1c999e15b + deb387b8b + dd3d9721c).
New branch `experimental/kv-prefix-fork-p1r2-20261020` (worktree /mnt/raid0/llm/worktrees/kv-prefix-fork-p1r2-20261020).
`experimental/kv-prefix-fork-p1r-20261020` (tip a1f0c8cf9) untouched.

## Cherry-picks (in order)
| Old | New | Result |
|---|---|---|
| 528b76f99 | efd92d46a | clean |
| 95b2d32da | 8860d1e91 | clean (server-context.cpp auto-merged) |
| 0b450aac8 | 43e33c885 | clean |
| 778beea19 | ebc00c4f2 | clean (server-context.cpp auto-merged) |
| ad6124c79 | d0557661a | clean |
| a1f0c8cf9 | f95052587 | clean (server-context.cpp auto-merged) |

## Conflicts
Zero textual conflicts. The common.h / utils.py conflicts of the first re-cut (RECUT.md) do not recur (their champion side is already in the base).

## Semantic review of the auto-merges (server-context.cpp)
- Slot file layout unchanged vs dd3d9721c: llama payload | SCKP (v1) | SDFT (v1). `save_slot_checkpoints` writes
  only n_tokens, pos_min, pos_max and the three blobs; `load_slot_checkpoints` default-constructs
  `common_prompt_checkpoint` (`pinned = false`). The fork added NO field to the file format.
- SLOT_RESTORE: the fork's `content_changed()` (P2 epoch bump) and `kv_altered = true` (restored cells have unknown
  provenance, so the slot is never a fork source) sit between `prompt.tokens.insert` and `load_slot_checkpoints`;
  upstream's checkpoint/draft appendix loads run after them. Compatible.
- `create_checkpoint(..., pinned)` and the a1f0c8cf9 "hold each junction once" logic operate on in-memory checkpoints
  only; a restored (unpinned) junction checkpoint is simply "pinned existing" if a later `checkpoint_at` lands on the same state.
- P2 fields (`/props.slot_fork`, `/slots` `content_epoch`, `prefix_hash`/`fnv1a64_tokens`, `id_slot`/`id_task` in timings,
  `checkpoint_at` incl. `{"message":k}`, `pinned` in `/slots` checkpoints) are the picked commits' code, unchanged.

## Known limitation: pinned junctions in slot files
The slot file stores no `pinned` flag (kept byte-compatible with upstream). A slot holding a pinned fork junction,
saved and restored, comes back with that checkpoint UNPINNED: it can then be evicted by the min-step rule or the
`n_ctx_checkpoints` cap. The data and the restore path are correct; only the eviction protection is lost.
Proposed follow-up (not done): carry pinned in a new appendix version (SCKP v2 with a per-checkpoint flag byte, or a
separate optional block after SDFT), loader accepting v1 and v2, so upstream-written files keep loading. Diverges from
upstream layout, so it needs the operator's call.

## New test
`tools/server/tests/unit/test_slot_fork.py::test_pinned_junction_slot_file_roundtrip`: pinned junction inherited by a
fork child, saved to a slot file, restored into another slot; checkpoint count preserved, `pinned` false after restore;
a request on the restored slot and a later request both match fresh. Skips on pure-attention models (no checkpoints).

## Pre-existing, not from the fork
`test-llama-archs -o` aborts at "deepseek41 Dense ... key not found expert_feed_forward_length" (rc 255); it still
writes qwen35-dense.gguf, which the tests use. The cherry-picks touch nothing in that test.

## Build/tests
Claim q2 (cpus 48-71), region-lock role build tag kpf-p1r2-rebase; logs in
/mnt/raid0/llm/tmp/kv-prefix-fork/p1r2-20261020/{logs,smoke-latest}. Champion for off==champion:
/mnt/raid0/llm/tmp/slotdraft-30574/bin-port (built from deb387b8b; dd3d9721c differs from it by one test-file import line).
