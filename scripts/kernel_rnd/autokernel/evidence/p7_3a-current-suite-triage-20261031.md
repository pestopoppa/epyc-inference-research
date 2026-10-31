# P7.3a current suite triage and scoring source audit

Date: 2026-10-31
Research lane: `/mnt/raid0/llm/worktrees/mains/codex-ni-research-code`
Base: `57ec250b6ad7b9bd5b9c1059cac69ea3603440ef` (`origin/main`)
Raw run logs and captured environment evidence: `/mnt/raid0/llm/tmp/codex-ni-research-scoring-bug-20261031/`

## Dispatched scorer row: already fixed in source

The root backlog row is mechanically dispatchable, but the research scorer is no longer the
diverged implementation it describes. Research commit
`d876adfef50912cdeaad14b34d416dc9608479d9` (2026-08-23) replaced it with a fail-closed delegation
shim to the canonical orchestrator scorer. The canonical implementation commit
`07a20a7cface6614c595f42981e385ae3695ee42` (2026-07-20) executes `entry_point_cases` with supplied
`args`/`kwargs`, and refuses an `entry_point` plus expected text when no executable cases or
`test_code` exist. It no longer synthesizes `assert f() == expected`.

Scope proof in source:

- Research adapter: `scripts/benchmark/debug_scorer.py` at `d876adf` delegates public scoring calls
  to the orchestrator copy and raises `ImportError` if delegation is unavailable.
- Canonical scorer: `scripts/benchmark/debug_scorer.py` lines 892–930 checks for executable
  assertions/unittest or nonempty `entry_point_cases`, refuses expected-text-only entry points, and
  runs each case as `entry_point(*args, **kwargs) == expected`.
- Canonical native tests: `tests/unit/test_debug_scorer_code_execution.py` covers refusal without
  cases, positive argumentized cases, and a wrong implementation. Research delegation test
  `scripts/benchmark/tests/test_debug_scorer_b7_delegation.py::test_p08_score0405_entry_point_without_oracle_raises`
  covers the research import path’s refusal.

Metric direction: scorer correctness/true-positive agreement is higher-is-better. The defect made
valid function answers false negatives; the repair admits valid argumentized answers and refuses
unprovable expected-text-only checks. Per instruction, no historical scorer test was rerun for
quota credit.

Suggested MAIN hygiene disposition: apply the existing row’s checkbox only after review, preserving
its text and original implementation dates. Evidence is the two SHAs and test source paths above;
no same-day completion should be attributed to this session. `[no-belief: source/test verification,
not a measurement warrant]`

## P7.3a exact acceptance

Original root row text:

> P7.3a — **triage the 78 failures / 10 errors in `autokernel/controller` + `autokernel/execution`**
> (identical from the repo root and from `scripts/kernel_rnd`, research `77b75dae`, measured
> 2026-09-23). CI cannot be green on these suites; classify stale fixtures vs real defects before
> the next campaign relies on them.

The accepted deliverable is classification of current failure signatures against current source and
test setup. It does not require repairing every external fixture or dependency. The 2026-09-23
78/10 historical observation remains unchanged.

## Current run and focused repair

The current-origin/main controller + execution suite run completed with **61 failed, 2,382 passed,
12 skipped, 10 errors, 883 subtests passed** in 646.39 seconds. Log:
`P7.3a-current-tests.log`. It ran under CPUs 132–133, OMP/BLAS thread counts 1, with a sampled
8-GiB/900-second guard; measured peak process-tree RSS was 0.73 GiB.

I corrected the shared import-alias defect in
`scripts/kernel_rnd/autokernel/controller/test_discovery_static_registry.py`: tests imported the
module as `autokernel.controller.discovery_static_registry`, but patched separate duplicate modules
under `scripts.kernel_rnd.autokernel...`. In this process the module objects and their `worktree`
objects were distinct. Build, split-runtime, and toolchain test doubles therefore missed the code
under test. Runtime-map tests also raised exception classes from the duplicate split-verifier
module, so the real verifier ran against an intentionally incomplete temporary fixture and raised
`FileNotFoundError` for `runtime/common`.

The tests now patch the actual imported `S.worktree`, `S.split_runtime_verifier`, and `S` objects;
fake exception classes come from `S.split_runtime_verifier`. No production source or gate was
weakened. A representative original failure passed, then the focused
`SharedRewardRuntimeTests` + `StaticBuildCacheTests` classes passed **50 tests and 22 subtests** in
152.97 seconds. Log: `static-registry-focused.log`. The run used CPUs 80–81 from approximately 18:49:13.9 to
18:51:46.8 UTC. A post-run admission audit found active CPU masks 0–23, 24–47 and 48–71, plus
A4A’s GPU/CPU grant `mi210_0,184-191`; C6’s `mi210_0,72-79` grant ended at 18:47:27 UTC. The
80–81 CPUs were not named by any active grant, and the 32–33 prep mutex files were empty. The
initial interpretation that coarse q3 folding occupied the whole q3 CPU range was incorrect; the
focused result is retained and no physical grant overlap is claimed.

## Classification of the 71 original current-run failures/errors

| Count | Signature / affected test area | Classification and evidence |
|---:|---|---|
| 33 | `StaticBuildCacheTests` build/cache failure events (23 test failures and 10 subtest failures) | **False failures from the test patch alias.** The fixture’s fake `run_build` was not installed on the module actually calling it; source code fell through to a real build of the deliberately minimal fixture. Fixed with object-based patches. Focused class passes (see above). |
| 4 | `SharedRewardRuntimeTests` runtime map cases | **False failures from the same test patch alias**, plus exception classes from the duplicate module identity. The real verifier saw fixtures with no `common/` directory. Fixed with the same-module patch/class references; focused class passes. |
| 12 | `ArenaCellRunnerTest` | **Pinned environment artifact drift; fail-closed source behavior is correct.** The required ROCm Python file exists, but SHA-256 is `0a97d28d3635c7dfd143de6d00b09c0cb382249c020b28ee33fb4002db1efb9c`, while source pin is `9544d2a29138833e6177d45dbc57468d37710b5080c901fbb579d53f251cdd6f`. Refresh/restore the owned evaluator environment or its reviewed pin; do not bypass identity validation. |
| 10 | `V8DeterministicCorrectnessCapabilityAcceptance` setup errors | **Stale v9 acceptance bundle against current v10 freeze.** The fixture declares production `production-consolidated-v9` / `0db32c06e3e550065b78311a6031ef3dd2c4f27c`; current production is `production-consolidated-v10` / `ffc1bac82eeca6f9099e1ccd9ba49703c460a115`. `_verify_production` correctly refuses the stale bundle. Refresh acceptance evidence under the current permitted freeze; no production tree was changed. |
| 4 | Deployment factory, detached canary, and rocprof acceptance tests | **Unavailable external model prerequisite.** All fail on absent `/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`; fixtures intentionally require the site model. Restore/provision the pinned model for those acceptance runs. |
| 1 | `EvoEngineerArenaTest` | **Missing Python dependency in the test runtime.** The admitted vendored source imports `scipy`; the active Python environment lacks it. Add/provision the project’s declared test dependency in the right environment, rather than changing evaluator semantics. |
| 2 | KernelFoundry and Xe-Forge nested subprocess tests | **Test subprocess import path omitted.** Child `python -m unittest autokernel...` is launched from repository root without `scripts/kernel_rnd` on `PYTHONPATH`; child fails `No module named 'autokernel'`. Add the package path to those child environments if these tests are to run from the repo root. |
| 1 | `ImmutableAuthorityTests.test_live_module_provenance_refuses_identical_user_owned_source_tree` | **Test invocation omitted required bytecode-disabled environment.** Source correctly requires `sys.dont_write_bytecode` and `PYTHONDONTWRITEBYTECODE=1`; because these were absent, the earlier strict refusal masks the test’s intended provenance assertion. Run this fixture with Python `-B` and the environment flag. |
| 1 | `DetachedCanaryTests.test_forced_stop_cleans_leader_and_descendant_without_killpg` | **Same bytecode-disabled test environment omission.** `stop_supervisor` correctly refuses sealed execution without the required mode. |
| 1 | `TestTheVerdictMustDescribeTheSubmission.test_a_verdict_reduced_from_other_material_is_refused` | **Malformed negative test double, not an unsafe verdict accepted.** It substitutes prior blocks *before* real reduction. The reducer raises `ReductionInadmissible`; runner records `ran=False`/could-not-run, which is fail-closed, so `_observation_from` has no verdict to reject and no `PipelineNotWired` is raised. To test the intended guard, have the fake pipeline replay the prior `EvaluationOutcome`; then the existing raw-sample comparison can reject it. |
| 1 | `InstrumentSelection.test_clean_instrument_worktree_can_be_selected_explicitly` | **Stale expected path after v10 split.** Source comments deliberately make instrument root and build root independent because the measurement instrument is folded into frozen production v10; overriding only the root no longer relocates the binary. Update the test expectation to the current pinned build root or test the separate explicit binary override. |
| 1 | `TestFixturesAreRealAndUnmodified.test_every_fixture_is_byte_identical_to_its_source_in_this_repo` | **Sparse-checkout omission, not source drift.** Provenance references three tracked `data/...` source files that this lane does not materialize. The test correctly refuses to claim provenance when zero source files resolve; a complete checkout (or those exact paths materialized) is needed. |

The 71 events are 33 + 4 + 12 + 10 + 4 + 1 + 2 + 2 + 1 + 1 + 1 = 71. They comprise 51 top-level `FAILED` rows, 10 static-cache `SUBFAILED` rows, and 10 `ERROR` rows in the original log. A generated roster in `p7_3a-20261031/cases.tsv` preserves every unique test/subtest ID and its classification; `machine-check.txt` records the unique-ID coverage assertion and group counts. The full captured output is `p7_3a-20261031/full-suite.log.gz` (gzip-compressed exact raw output); the focused rerun output is `p7_3a-20261031/focused-static-registry.log`; CPU grant audit is `p7_3a-20261031/resource-admission-audit.txt`. The whole suite was
**not rerun after the focused test-harness repair**, so no post-repair aggregate count or green-suite
claim is made.

The three execution-suite failure events are: the malformed negative test double in
`test_control_runner.py` (reducer correctly fails closed before the assertion under test), the stale
v9 instrument path expectation in `test_live_controls.py`, and the sparse-checkout provenance test
in `test_microbench.py`. They are classified separately above and are not controller failures.

## GitNexus and scope

Before the test-only edits, GitNexus reported `StaticBuildCacheTests` upstream impact **LOW, 0**
dependants, exact symbol context limited to this test class. `_observation_from` (reviewed while
classifying the execution failure) had **LOW, 1** direct caller (`ExecutedControlRunner.run_control`)
and exact lower-bound context; it was not edited. No HIGH/CRITICAL blast radius was found.

The test edit is isolated to the static-registry test file. Production, model, GPU, profiling,
server, activation, and kernel-store paths were not touched. No measurement is claimed.
