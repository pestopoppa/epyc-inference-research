# OAB-31 budget decision package — PENDING, NOT APPLIED

Research source revision `94b1dd64c163d915eb1347b118f57e44e981bf78`. The source and tests are uncommitted and unpublished. Parent review of the new custody owner and final production/test deltas precedes an operator decision. No proposal below changes files or conveys approval.

The guard requires each exception to have `ceiling` and a nonempty `approval` reference. An operator must authorize an actual reference before any entries are applied. This proposal gives the exact smallest integer ceiling sufficient for each changed, positively over-budget file in the reviewed snapshot. Five proposed ceilings include already-existing debt; the exception mechanism cannot distinguish that debt from this feature. The data below retains both numbers, so approval does not assert that baseline debt is fixed. No other ceiling, grading, timing or model default is proposed to change.

Recommendation: after source review, approve these six feature-specific snapshot ceilings with an explicit operator approval reference; retain the other existing failures as debt. Alternative: reject the exceptions and keep OAB-31 open while implementing a behavior-preserving reduction. The decision concerns budget authorization and does not reclassify existing debt.

## Exact source and review artifacts

- Production patch: `production-decision-review.patch`, SHA-256 `a02fe2f5b8e9c869ec5115046f57237360a4ac29c6524eaf0727170563a98a4e`.
- Test patch: `tests-decision-review.patch`, SHA-256 `4b27c02e3f5b2155696665a9c2021acf4ba28e8cbd3e0994f91487d37e516540`.
- Per-file SHA-256, complete machine-readable baseline/current guard findings and measurements: `budget-decision-evidence.json`. Both patch files include untracked new files. No staging occurred.
- Artifact lifecycle and controls are documented below; full test logs and original baseline-source reproduction remain in this task scratch.

## Changed loop modules and proposed exact ceilings

| File | Frozen ceiling | Base 94 count | Current count | OAB delta | Base overage | Current overage | Proposed exception ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| actor_metrics.py | 381 | 487 | 486 | -1 | 106 | 105 | none |
| actors.py | 2909 | 2925 | 3037 | +112 | 16 | 128 | 3037 |
| bestof.py | 1056 | 1056 | 1135 | +79 | 0 | 79 | 1135 |
| loop.py | 1394 | 1396 | 1427 | +31 | 2 | 33 | 1427 |
| pipeline.py | 325 | 335 | 355 | +20 | 10 | 30 | 355 |
| pool.py | 441 | 443 | 488 | +45 | 2 | 47 | 488 |
| resume.py | 2873 | 2873 | 2872 | -1 | 0 | 0 | none |
| run.py | 6127 | 6452 | 6588 | +136 | 325 | 461 | 6588 |

Pending exception entries, shown as intended values rather than valid manifest JSON because the operator approval reference does not exist yet:

- `actors.py`: ceiling `3037`; approval reference **PENDING**.
- `bestof.py`: ceiling `1135`; approval reference **PENDING**.
- `loop.py`: ceiling `1427`; approval reference **PENDING**.
- `pipeline.py`: ceiling `355`; approval reference **PENDING**.
- `pool.py`: ceiling `488`; approval reference **PENDING**.
- `run.py`: ceiling `6588`; approval reference **PENDING**.

The unchanged checker covers loop-only source; the new shared owner `autokernel/custody.py` has **147 code lines outside that scope**. It is included in the AST custody review fence and scratch creator inventory. Aggregate loop source: **106026 → 106447** code lines; including this new owner makes the task aggregate change **+568**. No claim is made that the loop-only guard limits the new owner.

## Existing baseline guard debt, separately recorded

Exact baseline source reproduced **36** guard errors; the current source has **37**. The only newly failing category is `bestof.py`, which was exactly at its ceiling and is now 79 over. Approving the six proposed entries would suppress five pre-existing overage findings as well as the new bestof finding, leaving 31 pre-existing categories; it would not repair the underlying debt. Baseline failures:

- `new in-scope source has no baseline entry: floor_units.py`
- `new in-scope source has no baseline entry: gpu_counter.py`
- `new in-scope source has no baseline entry: keep_screen.py`
- `new in-scope source has no baseline entry: lane_watchdog.py`
- `new in-scope source has no baseline entry: lane_watchdog_child.py`
- `new in-scope source has no baseline entry: provider_factories.py`
- `new in-scope source has no baseline entry: same_code_tripwire.py`
- `new in-scope source has no baseline entry: validate_keep_screen_contract.py`
- `accumulate.py: 763 code lines exceeds ceiling 472 by 291`
- `actor_metrics.py: 487 code lines exceeds ceiling 381 by 106`
- `actors.py: 2925 code lines exceeds ceiling 2909 by 16`
- `archive.py: 500 code lines exceeds ceiling 469 by 31`
- `belief_context.py: 591 code lines exceeds ceiling 580 by 11`
- `bench.py: 386 code lines exceeds ceiling 385 by 1`
- `campaign_service.py: 714 code lines exceeds ceiling 703 by 11`
- `champion_reanchor.py: 525 code lines exceeds ceiling 465 by 60`
- `gates.py: 2993 code lines exceeds ceiling 2988 by 5`
- `gpu_serving_profile.py: 699 code lines exceeds ceiling 697 by 2`
- `lane_targets.py: 325 code lines exceeds ceiling 267 by 58`
- `loop.py: 1396 code lines exceeds ceiling 1394 by 2`
- `new_epoch.py: 91 code lines exceeds ceiling 68 by 23`
- `pipeline.py: 335 code lines exceeds ceiling 325 by 10`
- `pool.py: 443 code lines exceeds ceiling 441 by 2`
- `production.py: 1312 code lines exceeds ceiling 1311 by 1`
- `residency.py: 537 code lines exceeds ceiling 477 by 60`
- `resolved_recipe.py: 1374 code lines exceeds ceiling 1364 by 10`
- `roofline_coverage.py: 227 code lines exceeds ceiling 213 by 14`
- `run.py: 6452 code lines exceeds ceiling 6127 by 325`
- `seed_bundle.py: 139 code lines exceeds ceiling 100 by 39`
- `serial_run.py: 2404 code lines exceeds ceiling 2340 by 64`
- `serving.py: 1587 code lines exceeds ceiling 1566 by 21`
- `serving_beliefs.py: 203 code lines exceeds ceiling 181 by 22`
- `serving_gate.py: 153 code lines exceeds ceiling 137 by 16`
- `status.py: 407 code lines exceeds ceiling 262 by 145`
- `surface_fold.py: 283 code lines exceeds ceiling 225 by 58`
- `unified_driver.py: 1293 code lines exceeds ceiling 1231 by 62`

The scratch creator inventory separately has its baseline 17 unlisted and 8 stale categories; changed creators, including the shared owner, are accounted for.

## Artifact lifecycle and crash behavior

- **Hypothesis:** a pre-author pool hold binds the actual HEAD and accepted hypothesis, then an inflight record is fsynced. Return or failure writes a durable disposition before the hold clears. Pooled rejection records its hypothesis. A missing patch permits progress only after actual base and full-clean proof.
- **Patch:** capture combines a binary diff from pre-author base to current worktree with accepted untracked source and rechecks HEAD and capture for races. The patch and metadata are durably written, then the disposition, before clearing the hold or resetting. Panel scratch uses original-base capture including author commits; storage failure retains the original fenced tree.
- **Full raw reply:** raw bytes are captured under a scratch fence and persisted with a call record before release. HTTP error bodies and status are included. Panel reply bundles are copied and fsynced before per-member disposition and release. Call-log append acquires a durable transaction fence before reading the original offset; successful append fsyncs the destination and parent before source truncation. Failed appends roll back to the original offset; a failed rollback leaves the transaction fence, even for an accidentally parseable tail. Existing corrupt tails refuse new records. The primary actor, panel transfer and sibling evidence writers all use the shared append primitive.
- **Crash and storage failure:** marker files and parents are synced before author threads spawn; custody failure raises `RunAborted` through broad panel handlers and leaves originals held. AST fingerprints and mutation tests are conservative review fences, not formal all-program proof.

## Validation and limits

Latest guarded transaction/actor-metrics/full-panel/AST suite: 119 passed. Actor and actor-budget suite: 174 passed, 1 skipped, 6 subtests. Earlier guarded bestof suite: 46 passed. Split loop/pipeline/candidate: 138 passed plus 13 subtests; resume: 102 passed plus 2 subtests; campaign continuation: 6 passed. Focused tests include real committed authors, no-edit proof, partial call-log writes followed by a real actor writer, rollback failures with corrupt and parseable tails, concurrent writers, non-UTF8 error bodies and call logs, persistence faults, and mutation fences. `shared-owner-final-controls.log` records 110 passed before later narrow changes; `shared-owner-final4.log` records 24 passing focused checks afterward. A combined suite was refused by the supervisor on a parent-change monitoring gap and drained; it is not counted. Other exact baseline failure reproductions and raw logs are retained alongside this package. No new inference run or measurement claim was made.
