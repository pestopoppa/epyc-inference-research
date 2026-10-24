# C1-05 AutoKernel rollback integrity

Completed on `refactor/autokernel-rollback-reviewed-01a16fc3`, based on the
unreviewed C1-05 WIP `5e0442b928b2d5284da779b9ad5f23a063c730c8`.
Implementation commit: `62c68ad99ace4c5cba92b736497fd1507fcce561`.

The interaction-regression rollback now turns git timeouts and spawn errors into
`RunAborted`, alongside the WIP's failed-exit and wrong-HEAD checks. The error
propagates through the accumulator wrapper and `loop.iterate`; the candidate
cannot be returned as kept, rebuilt or remeasured after an unverified rollback.
The shared `_git` helper was not changed beyond the inherited WIP.

Behavioral tests bind main's actual compiled accumulator closures with injected
git, build and measurement operations. They cover reset and verification exit,
timeout and spawn failures, wrong post-reset HEAD, successful restore/rebuild/
recheck, keep propagation and the inherited `_git` exit-check contract.

Validation:

- Final 13 tests against pre-fix WIP: **5 failed, 8 passed**, 0.66 s. The failures
  reproduce the swallowed timeout/spawn errors and candidate returned as kept.
- Final regression file after the fix: **13 passed**, 0.65 s.
- Targeted loop tests (`test_rollback_integrity`, `test_loop`, `test_accumulate`,
  `test_bundle_recovery`, `test_pipeline`, `test_candidate_dispositions`,
  `test_keep_overhead`, `test_anchor`, `test_production`,
  `test_serial_anchor_epoch`): **401 passed, 23 subtests passed**, 16.04 s.
- `git diff --check`: passed. Implementation delta: +152/-1, net +151 lines
  (151 test lines and one replaced source line).

All tests used the existing research `.venv/bin/python -m pytest -q`, with
`--basetemp` below `/mnt/raid0/llm/tmp/refactor-01a16fc3/rollback`, wrapped in
`/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 0-95
--role build --tag refactor-rollback-01a16fc3 -- bash -c 'cd
/mnt/raid0/llm/worktrees/refactor-01a16fc3-rollback && ...'`.
The explicit `cd` is required because region-lock changes the child directory.

Fresh own-worktree GitNexus analysis reported `_git` at HIGH risk (42 upstream
symbols), so its proposed edit was stopped. The approved narrow edit is inside
`_accumulate_after_keep`: LOW risk, two upstream symbols. The wrapper is LOW,
one upstream symbol. All own-worktree indexing used the repository wrapper.
An initial relative wrapper command indexed orchestrator because of region-lock's
child directory; no orchestrator source was changed.

Inherited call sites were reviewed: required source/cleanliness/identity reads
fail closed; diagnostic sidecars retain unavailable fallbacks; optional epoch,
baseline, status and revert-abort calls retain `check=False`. PPL eviction retains
its existing best-effort abort and reports conflict without rebuilding after a
failed revert. No additional helper behavior was introduced.

The main reviewed the narrow diff and authorized this closeout and a push of
the explicit review branch after passing tests. **DO NOT LAUNCH AUTOKERNEL**
remains in force. No production merge, promotion, service, kernel, or inference
operation occurred. Main integration still requires the separate hold release
and deliberate source re-pin; this code completion does not grant either.
