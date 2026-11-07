# AutoKernel custody review — 2026-11-07

Fence: `scripts/kernel_rnd/autokernel/loop/test_discard_invariant.py` (last pins c17d0d0b7).
Question per change: can it discard, skip, double-retain or orphan a candidate artifact / claim?

## loop.py — PINNED
Commits: 9ddf6d38e (AK-FAILRULE-1), 3ab4ffbf1 (JEV-4).
- `_RULE_GATES` gains `oracle_suite_empty`: a build refused because the anchor's oracle suite is
  empty is now a RULE gate (no authoring attempt spent; build checkpoint resumable once gate rules
  change) instead of a patch verdict. **Custody-changing-but-correct**: an empty suite is a harness
  property, not the patch's; keeps the checkpoint instead of retiring it. Mirrored in resume.RULE_GATES.
  Covered by test_prebuild_rules.py.
- `Outcome.critic_call` field + row emission; critic1 transient path computes `used` outside the
  `if critic1` (value identical when critic1 set) and attaches `critic_call` metadata. Checkpoint
  writes (`progress["inflight"]`, `_critic1_spent`) unchanged. **Custody-neutral**.
- critic pass 2: `ActorTransient` from `critic.review_patch` is now caught, annotates
  `progress["critic2"]["last_transient"]` (checkpoint retained, not cleared) and re-raises; previously
  it propagated identically. `ActorStopped` re-raised untouched. **Custody-neutral** (test_jev4_typed_reasons.py).

## pool.py — PINNED
Commit: 5afd1ae08. `promote_anchor` provenance `built_at` becomes a timestamp (or None) instead of the
dir path. Build-failure raise unchanged, no artifact disposition touched. **Custody-neutral**.

## run.py — PINNED
Commits: 9ddf6d38e, 1dae4bac8, 7810106af, 5afd1ae08.
- `suite_preflight.refusal` before the build returns `(False, [Verdict(SUITE_EMPTY_GATE)])` — the
  same return shape as the existing `op_scope` refusal immediately above, so the candidate follows the
  existing rule-gate retention path (see loop.py). **Custody-changing-but-correct**.
- census `read_split_census` / `quant_label`: metadata on scope records only. **Neutral**.
- validation target build wrapped in `target_build()` to stamp completion time; same `local_compiles`
  call and `gates.run_all` sequencing; provenance fields only. **Neutral**.

## resume.py — NOT PINNED (VIOLATION)
Commits: 3ab4ffbf1 (typed Ineligible — str subclass, same truthiness/text: neutral), 9ddf6d38e
(RULE_GATES mirror: correct), 2d3247e49 (`CONSUMED_PREFIX`: consumed-elsewhere checkpoints listed, not
claimed/rejected — in ResumeQueue.refresh and prepare; correct, it only stops re-rejecting settled
work), 3d2467a49 (fault re-emission skip in `prepare`).

**VIOLATION — 3d2467a49, `prepare()` (resume.py:1923 `faulted` set, resume.py:1930-1933 skip):**
the skip keys on a ledger row of the source with `result_status == "lane_error"`. That status is only
present while the source is RELEASED or exhausted; `ClaimLedger.claim` re-take sets
`result_status=NULL`, and the retry then settles to its own status. Scenario (reproduced):
1. Resumed checkpoint S faults -> `lane_error` row carries copy C (`resumed_from=S`, same stage), S released.
2. Next launch: C skipped (`fault_reemissions=1`), S re-claimed and run; gate refuses -> S settled `resume_rejected`.
3. Next launch: S no longer `lane_error` -> C queued as fresh work with a fresh retry budget: the same
   retained patch is re-run after its verdict (double custody of one candidate).
Also hit if a launch dies while S is re-claimed (status NULL). Pre-existing before 3d2467a49 (copies were
always queued), so a narrowed, not new, defect — but the fence protects exactly this.
Failure-injection test added: `test_resume_infra.py::test_a_fault_reemission_stays_unqueued_after_its_source_settles`
(`expectedFailure` until fixed). Suggested fix: skip C whenever ANY claim row exists for
`(resumed_from, stage)` (or consult claim_events for a `released` event), not only `lane_error` ones.

## actors.py — NOT PINNED (owned by Pipeline Management, being changed)
dd3ab6326: opencode v2 argv drops `--dir`; workspace binding now rides process `cwd`. Both call sites
(`_run_stoppable_in` Popen `cwd=str(cwd)`, and the `cwd=str(workspace)` / `cwd=workspace` runs at
actors.py ~991/995) pass the workspace, so the author still edits the candidate worktree.
**Custody-neutral as of dd3ab6326**, with one latent risk: any future caller that omits `cwd` would let
an opencode-v2 author write into the process cwd (edits escape the candidate tree -> empty diff /
foreign-tree edits). Recommend an assert `cwd is not None` for opencode v2 in the next port.

## resume.py — FIXED and PINNED (follow-up, same day)
`prepare()` now treats a `lane_error` re-emission C as owned by its source when ANY claim row exists
for `(C.resumed_from, C.stage)`, whatever its status (released, exhausted, settled, or in-flight NULL).
A copy at a different stage is still queued as new authored work, because the key includes the stage.
The re-review found no remaining path where a settled or in-flight source's copy gets a fresh id or
budget. Tests: `..._after_its_source_settles` (expectedFailure removed) and the new
`..._while_its_source_retry_is_in_flight` (mid-retry death) both pass, and the bounded-retry test stays green.
