# 2026-08-27 — AutoKernel audit, stop, and crash-fix

**Session:** operator audit session (no lane at start; created
`lane/autokernel-restructure-20260827`). Self-contained close-out.

## Mandate

Operator: the deepseek-v4-flash opencode session (owns GPU) had spent weeks trying to get
AutoKernel running, held the GPU, and produced nothing. Review/audit, remove dead weight, get it
running.

## Findings

- **0 scientific attempts across every campaign v3→v27**, 0 champion promotions. The one real result
  in the period (§22 64-VGPR occupancy cliff) came from a human running two `llama-bench` commands;
  the loop never reproduced it.
- **Root cause = self-inflicted restart loop, not a kernel problem.** (1) Planner shells out to
  `codex exec`; a codex 401 outage produced 284 failures in 23 min because transients retried with
  zero backoff. (2) The supervisor forced `max_restarts=0` for deployments, so every crash was a
  permanent exit and the OPERATOR became the restart loop (≥9 hand relaunches in 48h), and recovery
  mints a fresh sealed deployment so the iteration counter reset to 0 each time.
- **Structural:** ~15-40 LOC of real measurement inside ~278K LOC of custody scaffolding ("receipt"
  ×2735, "authority" ×824 in source); ~49:1 governance-to-science commit ratio.
- Full v27 crash forensics: 11 crashes mapped to raise sites and classified (transient / self-inflicted
  refusal / bug / operator). 4 of 11 were the KFD sampler; 2 were the codex outage.

## Actions

- **Stopped v27** — supervisor/factory/build/watcher/kfd-watchdog terminated by captured PID, verified
  dead (KFD proc count 0). `OPERATOR-STOP-20260827.md` left in the deployment dir.
- **Reclaimed worktrees** — 144/146 removed (2 handoff-referenced kept), dirty state backed up to
  `/mnt/raid0/llm/autokernel/reclaim-20260827/`. Free space 371 G → 589 G.
- **Fixed all four deterministic crash causes** on the lane, unit-tested (776 controller/execution
  tests pass; 3 failures are a pre-existing `claude/versions` env artifact identical on main):
  - `79e9ef1c` — planner exponential backoff + actor timeout + transport→transient reclassification;
    supervisor `max_restarts=0` clamp lifted (a restart is a resume from durable state).
  - `3a02c54f` — KFD sampler `owner_root_pid` (own subtree ≠ foreign) + `wait_until_clear` preflight
    gate; `create_campaign_worktree(prune_orphan_branch=True)` cleans a dead orphan ref.
- **Verified dead-weight list** — the "40K LOC dead" figure was wrong (grep missed `campaign.py`'s
  parenthesized import + `scripts/benchmark/` runners; 51/82 candidates are live). 19 modules /
  10.5K LOC are provably dead — safe to strip in a separate FOOTPRINT-regen commit.

## State / next

- Branch `lane/autokernel-restructure-20260827` (`3a02c54f`), NOT merged. Needs a GPU launch window to
  confirm residency-on-real-hardware end-to-end, then merge.
- Rider: `handoffs/active/autokernel-restart-and-strip.md`. Remaining: disk-expiry follow-up (not a
  crash), the dead-weight strip, and the long-term re-scope (custody at the promotion boundary, thin
  screening loop) — a design session.
