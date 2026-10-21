# KPF P1 re-cut: CPU gate report (2026-10-20)

Root for all relative paths here: `/mnt/raid0/llm/tmp/kv-prefix-fork/p1r-20261020/`. The KPF-18 runs are in
`kpf18cpu/run-20261020T163128Z/` (shortened below to `R/`).

The run was stopped early at 19:31Z on the coordinator's instruction, because the GPU window needed q3. Every own
PID was verified dead and the q3 claim was released.

## Build under test

- Branch `experimental/kv-prefix-fork-p1r-20261020`, worktree `/mnt/raid0/llm/worktrees/kv-prefix-fork-p1r-20261020`.
- HEAD `a1f0c8cf9`, which is champion `e33229348` + the 5 re-cut P1 commits + 1 fix. Re-cut details are in `RECUT.md`.
- CPU build `build-cpu`: gcc-15, Release, `GGML_NATIVE=ON`, shared, OpenMP. This is the store recipe of
  `cpu-20261006-4348de400`. The binary reports `version 10348 (a1f0c8cf9)`. Build logs: `logs/build-fork*.log`.
- Champion control build: e33229348 built with the same recipe in a temporary worktree, which has since been removed.
  Its server sha prefix is `826dde956a372eee`. Log: `logs/build-champ.log`.
- Linkage: `verify_ggml_linkage.sh` PASS. Every runner record carries its own copy, in `R/logs/*.linkage.txt`.

## Tests (Qwen3.5-0.8B hybrid, Qwen2.5-0.5B attention, gemma-3-1b SWA, generated qwen35; `-t 8`, q3)

Final run: `smoke-20261020T162449Z/summary.txt`, on the HEAD binary.

| Test | Result | Log |
|---|---|---|
| test-kv-seq-share, 4 memory types (includes the PARTIAL_ONLY restore into another seq while src keeps decoding) | rc 0 ×4 | `smoke-*/kv-seq-share-*.log` |
| test-save-load-state (qwen25 + qwen35-0.8b) | PASS ×2 | `smoke-*/save-load-state-*.log` |
| test-state-restore-fragmented (qwen25 + qwen35-0.8b) | SUCCESS ×2 | same dir |
| test-recurrent-state-rollback (generated qwen35) | OK | same dir |
| test_slot_fork.py, Qwen3.5-0.8B | 11 passed, including the new duplicate-checkpoint regression | `smoke-20261020T162449Z/pytest-qwen35-0.8b.log`, `logs/pytest-qwen35-postfix.log` |
| test_slot_fork.py, Qwen2.5-0.5B | 8 passed, 3 skipped (kv mode) | `smoke-20261020T162449Z/pytest-qwen25-0.5b.log` |
| off == champion e33229348 | 7/7 identical on each model | `smoke-*/smoke-*.log` |
| gen-models (test-llama-archs) | rc 255 | see below |

The gen-models rc 255 fails at the champion's own `deepseek41` arch ("key not found
deepseek41.expert_feed_forward_length"). No P1 commit touches it. `qwen35-dense.gguf` was still generated and its
tests ran.

## Open divergence finding from 10-04: a numerical artefact, not a fork bug

It reproduces on the re-cut: the hybrid cache-idle child 2 diverges from fresh at token 57/64. The script is
`scripts/divergence.py`. It runs these arms on the same binary:

| Arm | What it is |
|---|---|
| F | fresh |
| Ffrag | fresh on a deliberately fragmented pool |
| S | slot 0's own continuation of a trunk it prefilled itself, with no fork |
| C | fork children, cache-idle and no-cache-idle |

Results:

| Model | Comparison | Result |
|---|---|---|
| Qwen3.5-0.8B (`div2-qwen35-0.8b/run.log`) | C-cache-idle vs S | **bit-identical** (tokens and top-5 probs) on 3/3 |
| | F vs S (no fork at all) | diverges at 57 on child 2 |
| | F vs Ffrag (fresh, different layout) | diverges at 57 on child 2 |
| Qwen2.5-0.5B (`div2-qwen25-0.5b/run.log`) | F = S = C-cache-idle | bit-identical 3/3 |
| | C-no-cache-idle | diverges at 46, which is exactly where Ffrag diverges (46) |

So the fork restores exactly the state a slot reaches by prefilling the trunk itself. The gap to "fresh" comes from
two sources, and both reproduce with no fork:
- trunk-vs-whole-prompt chunk boundaries (the GDN chunked scan);
- KV cell layout.

There is no drafter in this model, so drafter restore is not involved. No semantics were changed.

## KPF-18 CPU subset

Setup:
- Model: Qwen3.6-35B-A3B-MTP-Q8_0, frontdoor argv (MTP n-max 4).
- Argv deltas: `-t 24`, `-c 98304`, `--kv-unified`, `--no-cache-idle-slots`, no `--mlock`. See
  `kpf18cpu/argv.frontdoor-cpu-gate.txt`.
- Run under q3, greedy, `n_probs` 20, 128 tokens.
- Runner: the inherited `kpf18_gate.py` with a `--cpu` mode added, at `kpf18cpu/kpf18_gate.py`.
- Prompt set: trunks 4k/8k/12k/16k × suffixes 256/512/768/1024 = 16 pairs. sha256 `fe8c8c92…`, file
  `R/prompt_set.jsonl`.

| Item | Result (counts) | Record |
|---|---|---|
| 6 off = champion | **8/8 bit-identical** (tokens and top-20 probs ×128). timings/slots/metrics key sets equal. failed-memory-slot 0 | `R/records/kpf18-20261020T164034Z-2.json` |
| 2 checkpoint fork vs source's own continuation | tokens identical 16/16, bit-identical (probs) 15/16. **Defective:** runner bug A below, so the fork used the near-end checkpoint (T−1536), not T. MTP draft counts differ on 4/16 | `R/records/kpf18-20261020T165157Z-1.json` |
| 5 seq-bit safety, cache-idle on and off | Pool invariant 22/22 ×2; metrics = slots 22/22 ×2; continuation with the source intact vs reused bit-identical 2/2 ×2; failed-memory-slot 0. `pool_used_eq_expected` 12/14 ×2: both misses are bug A (expected 16384 on the 4096 group; the observed 4096 is correct) | `R/records/kpf18-20261020T171335Z-1.json`, `…173438Z-2.json` |
| 1 equivalence vs fresh (partial, 10/16 pairs) | vs_fresh: top-1 10/10, greedy prefix 128/128 on 10/10, bit-identical 10/10. Control floor (fresh A vs B): 10/10 identical. Same for MTP off (`speculative.n_max 0`). **Defective:** bug B below, so these 10 children forked fresh A/B's L−1 checkpoints, not the trunk (`n_fork = L−1`; check `fork_taken_n_fork_eq_T` 0/10). That makes this a checkpoint-fork exactness result, not trunk-fork equivalence | `R/records/kpf18-20261020T175207Z-{1,2}.json` |
| 3 source invariance (busy source, fork mid-decode) | 2/2 bit-identical; control 2/2; child forked from the busy source 2/2 | `R/records/kpf18-20261020T182506Z-1.json` |
| 4 mid-prefill junction under prefill budget (`--prefill-budget-decoding 256 -b 512 -ub 512`) | vs_fresh 2/2 bit-identical; control 2/2; fork from the busy checkpoint 2/2; 610 mixed iterations | `R/records/kpf18-20261020T185012Z-1.json` |
| 7 fan-out speed | **Not measured.** Stopped in window A1 (fork_off) during the trunk; the partial record has no child rows | `R/item7.log` |

Bug A and bug B:
- **Bug A, closure late-binding.** Items 2, 3, 4 and 5 bound `T` late, so every group used the last group's T.
  Fixed in my copy before items 3 and 4 ran. **The same bug is in the GPU runner**
  `/mnt/raid0/llm/tmp/kpf-p1-20261004/kpf18/kpf18_gate.py`.
- **Bug B, item-1 ordering.** The child was sent after fresh A/B, so it forks their full-prompt checkpoint. Fixed in
  my copy: the child is now sent first. This bug is also in the GPU runner.

The re-runs of items 1, 2 and 5 with the fixes were queued, then cancelled by the coordinator stop.

### Fan-out prefill (the only fan-out numbers measured)

Qwen3.5-0.8B, 1560-token trunk, 3 children, from `smoke-20261020T162449Z/smoke-qwen35-0.8b.log`:

| Quantity | Value |
|---|---|
| Children's `n_fork_tokens` | 1560 each |
| Children's prompt tokens processed | 15 / 9 / 11, vs 1575 / 1569 / 1571 fresh |
| Unique pool (no-cache-idle) | 1772 cells, vs 6452 logical |

The 35B 16k fan-out (item 7) is still open. Timings are informational only.

## Code changes (commit `a1f0c8cf9`)

A trunk with `checkpoint_at:[-1]` followed by chained forks left each child holding **3 copies** of the junction
state. Each copy is ~150 MiB per slot on the 27B. The fix:
- the `end` fork snapshot is not pushed when a pinned twin was cloned;
- `create_checkpoint` skips an unpinned duplicate of a pinned one.

The champion path is unchanged (item 6 8/8; smoke off==champion 7/7). Regression test:
`test_fork_chain_holds_each_junction_checkpoint_once`, which failed before the fix
(`logs/pytest-dupckpt-prefix.log`) and passes after.
