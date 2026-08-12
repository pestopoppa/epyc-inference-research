# AutoKernel footprint — what campaign #1 actually imports

**Generated from the import graph, and asserted against it** by
`test_campaign_footprint.TestFootprintDocumentMatchesTheTree`, on each run:
every `yes`/`no` against the walked graph, every module against a row, every row
against a module that exists, every row against a stated reason, and the three
totals against the tree. If this file drifts from the tree in any of those, the
suite goes red. It is an assertion, not a description.

The per-row LINE COUNT is the one column that is regenerated but not asserted
to the line, and that is deliberate: five sessions share this clone, asserting
it exactly produced five red suites in forty minutes on 2026-08-04 and not one
of them was a boundary violation. The totals carry a 1,000-line tolerance for
the same reason — enough to ignore a plane being edited, not enough to ignore a
plane MOVING. `--refresh` restores every mechanical column to the line.

Reachability is computed by walking the AST of every module reachable from the
declared campaign-#1 roots — following relative imports at any level,
function-level imports, parent-package `__init__` side effects and dynamic
`import_module(…)` strings, absolute and relative. It is not a grep. A dynamic
import whose module name is not a literal (an f-string, a concatenation) cannot
be followed by any static walk, so it is REPORTED rather than skipped —
`test_no_dynamic_import_on_the_campaign_path_is_unresolvable` is red while one
exists, because a walk that silently steps over it reports a clean boundary over
a module it never looked at.

---

## The number

| | non-test lines |
|---|---:|
| **ON THE CAMPAIGN PATH** | **70,045** |
| **DEFERRED** (provably unreachable) | **53,420** |
| **TOTAL** | **123,465** |

**The deferred plane is explicit.** The compact modules deliberately off the
mutation/build path are offline analysis or pre-campaign planning surfaces: the
observe-only proposal diagnostic, prior-art compiler, substrate and lane
registries, turn-productivity reducer, and operator-invoked live-control
producer. The running campaign does import the compile-artifact veto because
that check must fire before behavioral T0. Reachability remains enforced module
by module; a new strategy or release plane is still forbidden by default.

The three figures above are regenerated from the tree and asserted row by row; no
percentage or line count is repeated anywhere else in this file, because a number
stated twice is a number that can drift in one of the two places.

### What the operator deleted, 2026-08-04

Until 2026-08-04 this table read *roughly half of this package is not on the path
from "an idea for a kernel" to "a measured number"*, over a total near 101k. The
operator acted on it: `release/`, `adapters/`, `surface/` and the AK4 strategy
plane under `controller/` were removed — about 79,600 lines including their
tests, recoverable from the tag `autokernel-preserve-20260804`. On 2026-08-12,
the selected release compiler and speech adapters were restored for AK9 while
remaining provably unreachable from campaign #1; `surface/` remains deleted. The rationale is
`epyc-root/artifacts/operator/autokernel-simplification-review.md`; the
prediction this document made — that removing them could not change what campaign
#1 does, because the walked graph reached none of them — held, and no reachability
assertion in `test_campaign_footprint.py` changed.

### The last four modules to cross, 2026-08-04

What survived the deletion under `controller/` was its MEMORY half:
`hypotheses.py` (the operator's hypothesis drop-in), `do_not_repeat.py` (the
§19.2 ledger) and `shared.py` (the six lines the two of them needed from the
plane that was removed). This document argued they were deferred *"read by
whoever PROPOSES a candidate and by nothing that measures one"*.

That was right about STRATEGY and wrong about CLAIMS. `hypotheses.py` also holds
`claim_for_hypothesis`, which documents itself as **"The ONLY route from a
hypothesis to a resource claim"** and enforces the rule that a falsifier is
optional when a question is written and mandatory before a claim is spent on it.
While it sat on the far side of this boundary it had **zero non-test callers**:
`campaign.py` called `acquire_cpu_region_claim` directly, so the gate enforced
nothing. The driver is what SPENDS the claim, so the gate belongs in the driver,
and `campaign.py --hypothesis` now acquires the region claim through it.

`do_not_repeat.py` came with it, and the check was made before admitting it
rather than after: `check_do_not_repeat(*, regime, matches)` is pure and the
driver never calls it — but `HypothesisTracker.authorize_claim(ledger=…)` has no
default and `claim_for_hypothesis` refuses a token carrying no verdict, so no
spendable token exists without a real ledger. `compile_for_tracker` builds one
from the tracker's own record, and it lives there. `controller/__init__.py`
binds both modules, so reaching one reaches both in any case.

**The prefix is still banned.** `test_campaign_footprint.CONTROLLER_ALLOWED` is a
LIST of four module names, not a prefix allowance: a fifth module dropped into
`controller/` is a boundary finding, and
`test_a_direct_import_of_controller_is_caught` plants exactly that module in a
copy of the tree and asserts it is caught.

### One correction to the earlier estimate

The deferred half was previously put at ~54k on the assumption that
`evaluator/integrity.py` and `evaluator/surface.py` were also unreachable.
**They are not.** `worktree.py` builds its build-identity receipt
from `integrity.check_clean_build_from_snapshot` and `integrity.hash_source_tree`,
`microbench.py` hashes the benchmarked binary with `integrity.sha256_file`, and
`chain.py` reads `surface.AffectedSurface` change classes. Those two modules
cannot be cut without cutting the receipt that proves a candidate was built
clean.

What *is* deferred inside them is their **gate derivation** —
`integrity.SourceIntegrityGateRunner` / `SourceIntegrityFirstRunner` and
`surface.SurfaceGateRunner`, the second of the two coexisting §8.5.1
derivations, the one that consumes evidence nothing produces (T0 today produces
nine of seventeen surfaces). Those get a **symbol fence** in
`TestTheProvenanceFence` — an allowlist frozen from the tree on 2026-08-04, so
any new name bound from either module fails the suite — rather than a
reachability ban that could only be satisfied by hollowing out `chain.py`.

A boundary you can only pass by deleting what it inspects measures the test
author, not the code.

---

## Every module

`yes` = reachable from the declared campaign-#1 roots. Reasons are a real
incident or a measured fact; "reduced rigour" is not a reason.

| module | lines | campaign #1 | reason |
|---|---:|:---:|---|
| `campaign.py` | 4,742 | yes | THE ENTRYPOINT. Before it landed, `grep -rl "__main__|argparse|def main("` over every non-test module returned nothing: 94k lines, 5,695 passing tests, and no way to start it — which is the whole reason this package has produced no result |
| `dashboard.py` | 201 | yes | the terminal result was fsynced but the only dashboard exporter had been deleted, so active AutoKernel work remained permanently absent from the operator surface; this compact projection dates itself from the journal entry and cannot make an old campaign fresh |
| `__init__.py` | 14 | yes | package docstring; `schemas` is declared here as the single source of record shape |
| `schemas.py` | 3,783 | yes | one record shape — every module is written against it and none invents its own |
| `journal.py` | 2,289 | yes | AutoPilot lost 232 trials and ~16 days when a restart came up empty and nothing objected |
| `fault_rehearsal.py` | 1,080 | no | governed process-only acceptance producer; real crash/restart, advisory revocation and tamper evidence must remain operator-invoked and cannot gain campaign mutation, inference, benchmark, kernel, stack or release authority |
| `offline_least_commitment.py` | 465 | no | AP-WM-1 observe-only archive analysis; importing an offline hypothesis diagnostic into the mutation/build path would give it accidental live authority |
| `least_commitment_archive_builder.py` | 429 | no | AK-WM-2a strict real-record projection; it reads hash-bound completed campaign evidence into an observe-only archive and cannot launch, rank, mutate, or promote |
| `least_commitment_receipts.py` | 836 | no | AK-WM-2a governed offline producer; it copies only explicitly SHA-pinned journal fields into receipts, rejects torn completed journals, and cannot launch, rank, mutate, or promote |
| `least_commitment_capture.py` | 487 | yes | prospective hash-bound IQK diagnostic/control contract; live campaigns only reduce declared outcome functions and it exposes no selector, champion, release, process, or inference API |
| `least_commitment_heldout.py` | 488 | yes | prospective held-out trust boundary; projects the effect only from a clean distinct completed campaign journal, derives the cross-regime candidate frame, and replays that projection before a target campaign can claim resources |
| `prepare_iqk_matched_pair.py` | 388 | no | deterministic pre-claim intervention/control publisher; derives shared seeds and the complete one-factor frame, validates both capture plans, and explicitly cannot infer, build, claim, execute, or mutate a journal |
| `evidence_path_rehearsal.py` | 279 | no | architecture-only CLI proving proposal/capture/control/AP-WM/champion/readiness/T3/package producer coverage; emits no empirical evidence or live authority |
| `turn_productivity.py` | 481 | no | AK-PT-1/AK-X-6 archive reducer; it consumes completed refine-turn records and may only withhold future search advancement, so campaign #1 must not give it live rank or mutation authority |
| `prior_art.py` | 597 | no | deterministic proposal-input compiler; it classifies findings before a proposal exists and has no place in the mutation/build process |
| `profile_report.py` | 587 | no | RVP-1–7 deterministic offline C4 report; it consumes completed paired traces and has no mutation, build, profiler-launch or ranking authority |
| `profile_context.py` | 287 | no | C4 hash-bound discovery/evaluator projection; it exposes diagnostic context to authoring without gaining verdict or ranking authority |
| `placement_context.py` | 216 | no | P2-5j hash-bound placement-belief projection; it exposes observation-only topology context to authoring without gaining selection, ranking, launch, or promotion authority |
| `hipkittens_lds.py` | 430 | no | INF-03 deterministic offline gfx90a LDS topology reducer; it consumes completed counter captures and gives authoring diagnostic context without profiler-launch or ranking authority |
| `c5_seed_corpus.py` | 309 | no | C5 static HyRA task registry; it contributes hash-bound, non-numeric gfx90a authoring context and must never gain campaign mutation, scoring, or NVIDIA-to-MI210 claim-transfer authority |
| `datatype_targets.py` | 132 | no | INF-03 static FP8/NVFP4 authoring contracts; they expose no cross-vendor performance numbers and have no campaign mutation, execution, scoring, or capability-attestation authority |
| `substrate.py` | 340 | no | validated planning facts; it reads checked-in measured/datasheet receipts before proposal construction and never joins the mutation/build path |
| `lanes.py` | 314 | no | screening declarations and rank-transfer calibration; without measured calibration campaign #1 stays on the full verified path |
| `artifact_diff.py` | 200 | yes | AK-TR-6 must veto an unconfirmed GPU claim before the behavioral T0 provider can launch |
| `candidate_record.py` | 289 | yes | every executed candidate is fsynced from the exact built snapshot and evaluation event identities before terminal STOP |
| `source_candidate.py` | 443 | yes | source-changing proposals consume one immutable embedded patch bundle through the guarded worktree mutation boundary |
| `source_prerequisite_package.py` | 534 | yes | source candidates may rank only after archived raw sensitivity, hostile and checker CSV bytes are re-reduced and rebound to the exact live build identities |
| `source_prerequisite_producer.py` | 352 | yes | a source candidate with no prior archive must produce sensitivity, hostile and checker receipts under the campaign's already-held claims before behavioral T0 |
| `storage.py` | 1,859 | yes | the 2026-07-04 async-prefetch win was written to `/mnt/raid0/llm/tmp/` and that directory no longer exists |
| `adapters/__init__.py` | 21 | no | AK9 adapter namespace only; campaign #1 remains llama_cpu-only and imports no speech release surface |
| `adapters/whisper_stt.py` | 1,843 | no | AK9 pure whisper.cpp tree, metric, linkage, protocol-prerequisite and release-binding declarations; no inference, build, mutation or freeze authority |
| `adapters/qwentts_tts.py` | 1,981 | no | AK9 pure qwentts.cpp declarations; pins the STT intelligibility instrument and requires ggml-submodule closure traversal without inference, build, mutation or freeze authority |
| `release/__init__.py` | 36 | no | AK9 release namespace; binds only the read-only plan compiler while readiness, T3, packager and preflight require explicit operator-side imports, all outside campaign #1 |
| `release/plan.py` | 2,539 | no | AK9 read-only release-plan compiler; derives exact per-tree cells and fails closed on missing evidence or single-backend no-op candidates, but cannot run or promote one |
| `release/preflight.py` | 184 | no | AK5 release-local pure preflight decisions over caller-supplied host, resource-claim and storage receipts; restores no controller guard or autonomous state machine and performs no observation or action |
| `release/readiness.py` | 4,312 | no | AK5/AK6 operator-facing readiness reducer; preserves per-backend and per-phase outcomes without a cross-device scalar, and has no mutation, execution, freeze or cutover authority |
| `release/t3.py` | 6,661 | no | AK5 dry-run release gate and sealed evidence receipt compiler; release mode refuses while P-KERNEL-FREEZE-1 is unratified, and the module cannot write, build, launch, signal, freeze or cut over |
| `release/packager.py` | 4,276 | no | AK6 operator package renderer over completed T3 evidence; emits only in-memory drafts and terminal `RELEASE_PACKAGE_READY`, never freeze eligibility or an executed production transaction |
| `release/closeout.py` | 482 | no | AK6 operator-triggered integration seam: joins the lean journaled champion to readiness, dry-run T3 and a validated package with crash/preemption/tamper terminal records, but has no build, inference, process or production-write capability |
| `release/live_material.py` | 280 | no | actual-journal release-material compiler; hash-binds composed champion evidence to an exact externally measured full-build seal for operator-triggered dry run only, with no build, inference, process, transport, freeze, cutover or production-write authority |
| `evaluator/__init__.py` | 41 | yes | docstring only — it binds no submodule, so importing `evaluator.api` does not drag the plane in |
| `evaluator/api.py` | 3,320 | yes | a `Verdict` is constructible only via `compute_verdict()`; `kernel_eval.sh` stamped `"status":"OK"` unconditionally |
| `evaluator/correctness.py` | 3,952 | yes | throughput is reward-hackable: deleting the computation is the fastest kernel there is |
| `evaluator/recipes.py` | 2,433 | yes | argv from a hashed constructor — production drifted off NUMA interleave 2026-05-24 and the front door ended up at 46% of canonical |
| `evaluator/devices.py` | 813 | yes | a GPU cell must not be satisfied by `Device 0: CPU` |
| `evaluator/controls.py` | 2,406 | yes | the A/A control plane — 2026-08-04 measured 1.62% / 1.88% between-run CV over four identical runs |
| `evaluator/baseline_honesty.py` | 186 | no | AK-BH-4 exact-surface strongest-provider selector; it rejects AUTO and cross-surface transfer before a campaign can claim an honest floor |
| `evaluator/c3_epyc_suite.py` | 754 | no | INF-48 offline C3/C5 contract reducer; it binds EPYC op cases to exact vendor evidence and a captured-workload whole-model exit without launch, mutation, release, or promotion authority |
| `evaluator/c3_epyc_compiler.py` | 519 | no | INF-48 offline JSON plan/receipt compiler; it is the callable controller/backend seam but cannot launch captures, benchmarks, builds, patches, inference, or promotion |
| `evaluator/c3_epyc_tensor_capture.py` | 599 | no | INF-48 governed real-model tensor-capture producer; only its explicitly authorized execution seam may invoke a workload, while its receipts carry tensor identity and no correctness, timing, ranking, or promotion authority |
| `evaluator/c3_apex_runner.py` | 1,219 | no | INF-48 exact Apex mapping seam: binds one reviewed gfx90a k228 trace or the ordered branch-aware k175 composite plus governed tensor/source/model/toolchain identities, and remains outside campaign mutation and candidate authoring |
| `evaluator/sensitivity.py` | 333 | yes | RVP-C2-7/C2-11/C5-2 offline two-axis reducer; its report has no live authority until `source_candidate_authority.py` binds exact candidate/evaluator/evidence provenance, and missing or insensitive populations remain unscoreable |
| `evaluator/oracle_integrity.py` | 145 | yes | RVP-C2-8/C2-9 offline reducers; hostile/checker results gain T0 authority only through the exact provenance bridge and never independently rank a candidate |
| `evaluator/source_candidate_authority.py` | 87 | yes | fail-closed bridge from offline sensitivity/hostile/checker outputs to live T0: binds source, evaluator bundle, suite, capture mode and evidence hashes without launching or mutating anything |
| `evaluator/historical_tasks.py` | 194 | no | RVP-C5-R/C3-2 sealed historical-task descriptor and expert-ceiling reducer; no terminal candidate means `COULD_NOT_CHECK` |
| `evaluator/rebench_scoring.py` | 120 | no | AK-RB-1 offline reference-normalized scoring and matched-budget curve reducer; it consumes completed behavior checks and timings but has no campaign mutation, execution, or verdict authority |
| `evaluator/statistics.py` | 3,669 | yes | **calibration constants and `median` only.** Its e-process made the gate unpassable: threshold 10 against a sign-martingale that tops out at 5.5687, at every effect size. Fenced by `TestNoOptionalStopping` |
| `evaluator/integrity.py` | 3,661 | yes | **provenance primitives only** — `sha256_file`, `hash_source_tree`, `EMPTY_TREE_SHA256`, the clean-build snapshot check. Its §8.5.1 gate runner is fenced off |
| `evaluator/surface.py` | 3,195 | yes | **change-class constants only** — `AffectedSurface`, the core/shared-header fanout classes. `SurfaceGateRunner` is fenced off |
| `execution/__init__.py` | 24 | yes | docstring only; states the deny-8 limits every executor inherits |
| `execution/worktree.py` | 2,883 | yes | no candidate exists without it: production-tip anchoring, campaign worktree, build, build-identity receipt |
| `execution/provider.py` | 71 | yes | provider-backed proposals must resolve to isolated prefixes outside shared ROCm/system locations and frozen production trees before candidate recording |
| `execution/microbench.py` | 4,345 | yes | paired ALTERNATING blocks plus C6-10 ranked hard cases — each hostile unit changes the receipted recipe and contributes blocks to the same rank instead of living only in a correctness gate |
| `execution/device_sampler.py` | 410 | yes | RVP-C3-4 numeric 250 ms ROCm state producer; it brackets the exact captured benchmark-process lifetime and fails closed on missing fields, failed probes, empty traces, or cadence gaps |
| `execution/instrument_integrity.py` | 111 | yes | RVP-C6-1: a candidate binary is built from candidate-controlled source, so every live T1 invocation must re-pin its reward-bearing translation unit to the named anchor before it can emit a number |
| `execution/physical_bounds.py` | 197 | yes | RVP-C6-4 physical impossibility screen: per-shape conservative work floors and hardware peak ceilings are bound to the exact delivered unit and recipe/model/parameter frame, then every live sample is checked before ranking |
| `execution/powercap_broker.py` | 248 | yes | the v9 CPU preflight could not read root-owned 0400 package counters, while running the campaign as root correctly failed the non-root candidate sandbox; a captured networkless read-only container now exposes only exact package-energy integers |
| `execution/reward_hack_scan.py` | 186 | yes | RVP-C6-6/C6-9 plus static C6-2/C6-3 detectors: protected-frame, pointer-memo, structured-shortcut, environment/timing and stream/thread findings; the named 10 planted/15 clean corpus states sensitivity/specificity/FPR, not arbitrary-program coverage |
| `execution/reward_hack_corpus.py` | 370 | no | operator-invoked instrument producer that compiles and runs the named 10 planted/15 clean HIP corpus on gfx90a under the shared device claim, with normal and anti-short-circuit units both timed in the ranked stream |
| `execution/sandbox.py` | 1,076 | yes | C6/INF-03 process boundary: default candidate profile retains Landlock write confinement; broker-only controller profile default-denies networking and read/exec outside exact roots/files/executables while inheriting one peer-bound broker UDS; direct model profile binds fixed Claude/Bun self/kernel reads to the actual model PID, permits outbound INET clients, and denies broker/GPU/new AF_UNIX/server authority; evaluator profile exposes only exact ROCm devices and pinned runtime inputs while denying networking, broker inheritance, cross-process memory and io_uring; all retain signal/ptrace/namespace denials, finite limits, owned cgroup and policy-hashed receipts |
| `execution/t0_provider.py` | 3,697 | yes | the predecessor harness tested MUL_MAT only, so a kernel that broke MUL_MAT_ID — MoE dispatch, every token in production — passed it cleanly |
| `execution/control_runner.py` | 1,805 | yes | runs the neutral / A-A controls that the measured drift makes mandatory rather than optional |
| `execution/live_controls.py` | 1,413 | no | standalone, operator-invoked calibration producer for the fixed five controls; it prepares the instrument before campaign #1 and is deliberately not imported by the mutation/build entrypoint |
| `execution/cpu_region_claim.py` | 2,408 | yes | 2026-08-04: two A/A runs were destroyed by a legitimate co-tenant because the loop held no claim. Before this module a claim could be READ but never acquired |
| `execution/chain.py` | 1,928 | yes | holds the seams — four mismatches between executors and evaluator, one of them a field whose meaning INVERTS across the seam |
| `resource/__init__.py` | 28 | yes | docstring only; names the `resource`-shadows-stdlib hazard the loop must not trip |
| `resource/device_claim.py` | 1,826 | yes | §2.6's first row of substrate that exists nowhere in the project: a cross-process GPU device claim someone actually holds |
| `resource/preflight.py` | 1,788 | yes | INC-20260731: a name-pattern kill took out another agent's `llama-server` twice, and `earlyoom`, whose argv names what it guards |
| `resource/claim_witness.py` | 325 | yes | invariant 9 — idle sensing is never a claim, and the witness is what tells the two apart |
| `controller/__init__.py` | 89 | yes | binds every surviving controller module, so importing one reaches both — which is why `controller.do_not_repeat` is on the path whether or not the driver names it, and why `CONTROLLER_ALLOWED` lists this file rather than leaving the edge unexplained |
| `controller/champion.py` | 1,461 | no | AK4 lean lifecycle projection and composition transaction: consumes validated journal evidence, requires a real combined rebuild/evaluation through an injected runner, and carries no source-mutation, build, benchmark, process, release, or production-write capability |
| `controller/sequencer.py` | 468 | no | AK4 deterministic outer ordering seam: consumes supplied proposals and injected execution capabilities, journals bounded stop states, and remains unbound from `controller.__init__` so campaign #1 cannot acquire a second live loop |
| `controller/completed_campaign_adapter.py` | 105 | no | strict live-journal-to-sequencer join; banks only event-bound dispatch/mechanism-confirmed wins above floor and MDE and launches nothing |
| `controller/arena_adapter.py` | 546 | no | INF-03 paper-pin GEAK/AgentKernelArena integration: validates clean vendor sources and physical gfx90a identity, binds C4 into a hygienic priced prompt, and launches registered whole-agent task adapters without entering campaign mutation or scoring; callback failures reap the exact captured process group |
| `controller/arena_controller_sandbox.py` | 687 | no | INF-03 controller/model isolation adapter: exact copied workspace plus pinned Python/Codex/Claude/Node/source/CA runtime discovery and fixed Bun volatile reads, broad/device/production/campaign/symlink refusal, broker-only controller prefix, per-model-PID outbound-client prefix, exact activation verification, and descendant-draining cgroup teardown; no direct GPU or evaluator profile |
| `controller/claude_codex_actor_critic.py` | 799 | no | INF-03 bounded Claude-planner/critic + Codex-actor controller: exact three-argument Arena launcher, explicit model/effort pins, brokered model and candidate-evaluation calls, measured starting-state fallback and best-candidate materialization, eight-item measurement/critic revision memory projected into each next planner and fully receipt-bound, semantic task-manifest confinement with four exact receipt-bound/scrubbed control-plane roots excluded and refused as candidates, launcher-cgroup-owned timeout teardown, and transcript/outcome/artifact hashing; its preflight invokes neither CLI |
| `controller/codex_container_actor.py` | 219 | no | INF-03/AK-LE-3 host-compatible Codex actor boundary: pins a read-only container image, exposes exactly one writable host bind for the copied Arena workspace, stages auth only in an automatically erased temporary directory, and admits exactly the reviewed gpt-5.6-sol/gpt-5.6-terra high-effort model cells without granting host filesystem writes |
| `controller/arena_campaign.py` | 1,073 | no | INF-03 matched-panel audit and injected-runner seam: binds the baseline plus seven controller arms to equal tasks and checkpoints, admits each governed arm only with exact adapter/upstream/CLI/model pins, refuses incomplete panels before execution, and remains outside live campaign mutation and scoring |
| `controller/arena_cell_runner.py` | 3,505 | no | INF-03 fail-closed runner: parent-owned authenticated broker serializes model and evaluation frames; direct model CLIs receive fresh per-PID outbound-client sandboxes, writable Codex actors retain an attested digest-pinned single-bind container, and every candidate evaluation enters a fresh GPU evaluator sandbox under exact claims; persistent model/child/request/result/lifecycle evidence is rehashed on restore |
| `controller/arena_evaluator_child.py` | 207 | no | INF-03 claim-blind evaluator child: strictly reconstructs the hash-bound starting baseline, verifies pinned Python/config/vendor identities, invokes the vendor evaluator only inside the dedicated deny-network GPU sandbox, and emits one strict self-hashed JSON result |
| `controller/arena_upstream_common.py` | 730 | no | INF-03 licensed-controller client: bounded read-only text-model calls are GPU-blind only after an OS-level device-open proof; the controller accepts the exact parent-declared source tuple without importing PyYAML/vendor evaluator code, and complete candidate bytes receive feedback solely through interrupt-safe writes to the one preconnected authenticated parent-worker broker descriptor—never destination-bearing send or new socket authority |
| `controller/evoengineer_arena.py` | 487 | no | INF-03 exact-source EvoEngineer-Full seam: pins the paper-era MIT source, preserves its named operators/parameters, and runs its executable CLI only through the parent-worker evaluation broker and device-isolating hash-bound launcher |
| `controller/k_search_arena.py` | 283 | no | INF-03 licensed K-Search world-model/tree port: injects the centralized Arena evaluator into the exact pinned upstream Task seam without entering campaign mutation or scoring |
| `controller/geak_v1_arena.py` | 344 | no | INF-03 licensed GEAK-v1 OptimAgent port: maps its ROCm dataset and reflection loop onto centralized Arena evidence without entering campaign mutation or scoring |
| `controller/xe_forge_arena.py` | 593 | no | INF-03 licensed Xe-Forge gfx90a port: retains the pinned DSPyEngine linear-CoVeR analysis/planning loop, replaces process-global device prompts with scoped AMD MI210 guidance, uses a no-shape executor gate, and routes initialized KernelBench execution through Arena |
| `controller/kernelfoundry_arena.py` | 481 | no | INF-03 licensed KernelFoundry gfx90a port: retains inherited Controller.run_single MAP-Elites/island branching, strictly activates pinned Triton feature patterns, binds measured parent transitions to upstream QD tracking, and routes evaluation through Arena |
| `controller/arena_roundtrip.py` | 216 | no | EVL-47 SC20/SC21 prospective GEAK/Arena receipt producer; emits only observed correctness/timing rates and retains preflight as non-ordinal dependency evidence, with no launch, mutation, or verdict authority |
| `controller/hip_authoring_arm.py` | 627 | no | governed raw-HIP compatibility seam: pins and hashes a true Torch2HIP task/candidate/toolchain, compiles GPU-blind for gfx90a, and scopes distinct MI210 claims to baseline/final evaluation; emits observation-only correctness/timing-validity rows with no ranking, campaign, production-tree, or promotion authority |
| `controller/hip_decision_grade.py` | 516 | no | task-local raw-HIP decision evaluator: source-before-suite sealing, unseen hostile inputs, independent host-double reduction, static C6 scan, exact Torch-ROCm-compile C3 provider, per-arm gfx90a duration admission, paired e-process reduction, and no release/promotion authority |
| `controller/hip_decision_grade_worker.py` | 232 | no | C6-contained raw-HIP child: receives inputs but never expected outputs, performs double-poison/determinism checks, and captures exact candidate/provider timings plus Torch-Inductor implementation identity |
| `controller/authoring_contract.py` | 468 | no | AK-PL-1/AK-LE-4/AK-LE-5 pre-proposal adapter: fully rendered prompt leak refusal, priced never-bulk-read context, reversible compaction, and structured external numbers; it calls no model and must not gain mutation/build authority |
| `controller/loop_experiments.py` | 680 | no | AK-LE-1/2/3 observation-only experiment contract: predeclares matched planner/scaffold cells, renders target values only into planner context, reduces complete observations, and emits a distinct planner-only AK-LE-1/2 receipt rather than fabricating missing scaffold evidence; no model, campaign, ranking, champion, or release authority |
| `controller/loop_experiment_runner.py` | 730 | no | AK-LE-1/2 governed planner bridge and AK-LE-3 router: compiles exact model/effort cells, runs read-only captured Claude/Codex processes, seals strict observations, and delegates scaffold work to the separate governed writer seam; no campaign, ranking, champion, or release authority |
| `controller/loop_scaffold_runner.py` | 1,216 | no | AK-LE-3 governed SAME-MODEL scaffold seam: exact selected task/context/champion/source/container-actor/evaluator pins, reviewed Sol/Terra × direct/split cells, matched wall time, fresh disposable baseline/candidate worktrees, captured PID/process groups, audited write scope, sealed role checkpoints, exclusive MI210 claim, verified worktree removal, and centralized Arena evaluation without campaign/ranking/champion/release authority |
| `controller/arena_scaffold_evaluator.py` | 151 | no | AK-LE-3 isolated AgentKernelArena worker: independently revalidates pinned disposable worktrees and the clean evaluator/task source, measures baseline then candidate through the centralized vendor evaluator, rejects actor-reported scores, and returns diagnostic-only compile/correctness/timing facts |
| `controller/loop_experiment_prefilter.py` | 613 | no | AK-LE-1/2 deterministic external reducer: requires an independently persisted/hash-pinned experiment-only structural prefilter contract before manifest compilation, verifies the panel and sealed observation bytes, binds exact normalized fingerprints plus prior/duplicate decisions, and emits either a partial receipt or durable refusal without invoking/replacing the campaign do-not-repeat gate or gaining model/operator labels or campaign/ranking/champion/release authority |
| `controller/loop_experiment_beliefs.py` | 371 | no | EVL-47 SC29 prospective AK-LE projection: wraps the exact source-pinned planner reducer, emits self-hashed per-cell search-persistence measurements bound to manifest/panel/prefilter/evidence/producer identities, and cannot gain campaign, ranking, champion, release, inference, build, or mutation authority |
| `controller/reward_monitor.py` | 453 | no | C6 monitor adapter: binds campaign/candidate traces and the whole journal tree to a predeclared monitor panel, requires awareness plus reasoning visibility, and reports sensitivity/specificity/FPR without calling a model |
| `controller/hypotheses.py` | 4,493 | yes | `claim_for_hypothesis` — the falsifier-before-compute gate `campaign.py --hypothesis` acquires its region claim through. It calls itself the ONLY route from a hypothesis to a resource claim and had ZERO non-test callers until 2026-08-04, because this boundary put it on the far side of the line: the driver is what SPENDS the claim |
| `controller/do_not_repeat.py` | 2,205 | yes | the §19.2 ledger a loop needs to tell "tried and failed" from "never tried". On the path because `authorize_claim(ledger=…)` has no default and `claim_for_hypothesis` refuses a token with no verdict, so no spendable token exists without a real one — `compile_for_tracker` is it |
| `controller/shared.py` | 166 | yes | the six lines `hypotheses` and `do_not_repeat` reached into the removed plane for — `ControllerError`, `selection_block()`, `LEDGER_DIMENSIONS` and the fingerprint pair. Twenty thousand lines were pinned by six, because a concern shared by two modules had nowhere to live; reached only through them, and the campaign path names nothing in it |

---

## What the boundary test enforces

`test_campaign_footprint.py`, beside the entrypoint:

1. **`TestCampaignFootprint.test_campaign_path_does_not_reach_the_deferred_half`**
   — the walked graph reaches nothing under `controller/`, `release/`,
   `adapters/` or `surface/`. `release/` and `adapters/` are live deferred AK9
   prefixes, while `surface/` remains deleted; none may come onto the campaign path.
   `controller/` is also a live prefix, and
   it is the DEFERRED figure in the table above.
2. **`test_the_deferred_half_is_still_on_disk`**,
   **`test_the_deferred_half_is_a_real_share_of_the_tree`** and
   **`test_the_entrypoint_exists`** — anti-vacuity. The boundary must not start
   passing because its targets were renamed, because the deferred modules migrated
   onto the campaign path, or because the entrypoint it is drawn around was
   deleted. The last of those was a `skipUnless`, which meant `rm campaign.py`
   turned this whole file into a green no-op. The line floor in the second was
   RE-PINNED on 2026-08-04: 40,000 was calibrated against a 46k deferred half that
   the operator then deleted, and a floor no surviving file can clear is not a
   check, it is a permanent red. The derivation of the replacement is in the
   constant's own comment.
2b. **`test_no_dynamic_import_on_the_campaign_path_is_unresolvable`** — every
   check here is a statement about a graph that was WALKED, and an
   `import_module(f"{__package__}.…")` is a hole in that graph. It is REPORTED,
   never stepped over: a walk that skips what it cannot follow returns a clean
   boundary over a module nobody looked at.
3. **`TestTheProvenanceFence`** — `integrity.py` and `surface.py` are reachable
   only through a frozen allowlist of hashing / change-class names, and neither
   deferred gate runner is wired. Both the allowlist and the denylist are
   checked against the modules' own ASTs, so a typo cannot silently forbid or
   permit nothing. The fence reads four idioms, and the last two defeated its
   first version: a direct name import; `integrity.X` through a one-hop alias;
   `evaluator.integrity.X` through a CHAINED alias, which runs because
   `worktree.py` has already made `integrity` an attribute of the package; and
   `getattr(integrity, "X")`.
4. **`TestNoOptionalStopping`** — the campaign path binds no e-process name, and
   reaches no interim-look method through an object it legitimately holds
   (`CampaignStatistics.sequential_evaluation()` is exactly the hole a
   name-level check leaves open). The accept rule, `api._resolve_effect`, cannot
   import `statistics` at all, so it can READ a recorded e-value but never run
   one.
   **Scope limit, stated so it is not mistaken for cover:** this is a check on
   the import graph, so it forbids optional stopping *inside a process*. It says
   nothing by itself about `execution/README.md` §6.5 — a declared round re-run
   until it crosses — because that is optional stopping ACROSS processes. The
   durable completed-run ledger closes that separate seam; this static assertion
   remains responsible only for the in-process path. §6.5 closed 2026-08-05.
5. **`TestTheWalkerItself`** — synthetic graphs that bite-verify the walker
   (transitive edge, function-level import, `try:`-guarded import,
   parent-`__init__` side effect, two-level relative import, dynamic import by
   absolute string, by relative string, and by f-string, chained alias, and
   `getattr`), with compliant-path controls beside them that must produce no
   finding.
5b. **`TestTheTotalsCheckBites`** — the totals check against text it must reject:
   a swapped pair, a plane moving, a total restated in the prose, a missing row.
   Its own first fixture put the two halves one tolerance apart, which made a
   swap unobservable and the bite test FAIL — so it also records, as an
   assertion, the bound it really has.
6. **`TestTheBoundaryCatchesRealTreeViolations`** — the same four checks run
   against a **copy of this package with a violation planted in it**: every
   mutation caught, and compliant-path controls beside each, all silent. Toy
   fixtures verify that the walker resolves each import *form*; this verifies
   the checks fire on the real closure. The copy is why: the shared
   clone (`/workspace/repos/…` and `/mnt/raid0/llm/…` are one checkout) is never
   written to.
   **Re-pointed 2026-08-04.** Five of these plants named modules under the deleted
   planes. A probe that imports a module which does not exist plants NOTHING — the
   walk drops a name with no file, the check reports clean, and the test fails
   claiming the walker did not bite when what was missing was the target. They now
   plant against `controller/`, the one deferred prefix still on disk, which keeps
   each mechanism (direct import, function-level import, parent-`__init__` side
   effect, dynamic import by absolute and by relative string) exercised against the
   real tree. The deleted prefixes keep the one assertion still meaningful about
   them — `test_re_adding_a_deleted_plane_and_importing_it_is_caught`, which writes
   an empty stub plane into the temp COPY and proves the ban fires the moment
   something depends on it again.

## Three findings this exercise produced

**1. The improvement verdict is currently unreachable on the campaign path.**
`api._resolve_effect` returns `EFFECT_EVIDENCE_BELOW_THRESHOLD` whenever
`effect.e_value < effect.threshold`, and `EffectEstimate` requires both fields at
construction. For the CPU decode cell the calibration solves `threshold = 10`
while the sign-martingale tops out at 5.5687 over five same-sign blocks — at four
different effect magnitudes, because the construction is sign-based. So a
candidate that is genuinely faster still resolves to "evidence below threshold".
Campaign #1 must either produce its T1 result outside `_resolve_effect`, or
declare a construction whose e-value can cross. The 2026-08-04 A/A data
(between-run CV 1.62% / 1.88%) says the honest answer is the first one: a median
over paired deltas covers this noise, and the e-process was never warranted here.
`TestNoOptionalStopping` fences the mechanism; it does not decide this question.

**2. `campaign.py` does not reach the A/A control plane.** The entrypoint imports
`journal`, `schemas`, `storage`, `evaluator.{api, correctness, devices, recipes}`,
`execution.{chain, cpu_region_claim, microbench, t0_provider, worktree}` and
`resource.{claim_witness, device_claim, preflight}` — and neither
`evaluator/controls.py` nor `execution/control_runner.py`. Those
two are declared campaign-#1 roots here, and they are declared for a measured
reason: on 2026-08-04 four A/A runs of identical code showed decode declining
MONOTONICALLY (52.76 → 52.31 → 51.62 → 50.52), which is drift, not scatter, and
no number of repetitions removes it. Interleaved paired blocks handle the drift
WITHIN a comparison; the neutral control is what tells you the drift is there at
all. A campaign that never runs one cannot distinguish "this kernel is 4% faster"
from "this kernel ran first". This is a gap in the entrypoint, not in the list.

**3. `evaluator/` is reachable in full — all nine modules.** There is no thinner
evaluator slice available without breaking a real seam. If the operator wants the
campaign path smaller, `evaluator/` is where the next boundary has to be drawn,
and it is a refactor, not a deletion.

## Acting on this

To delete a deferred plane: move its prefix into `DELETED_BY_OPERATOR` in
`test_campaign_footprint.py`, remove the directory, and delete its rows here.
The reachability assertion still holds — an absent module is unreachable — and
that edit is why this was written as a test rather than as a recommendation.

**What that edit does NOT cover, measured on 2026-08-04 by doing it.** "Nothing
else changes" was too strong, and the correction generalises past this package:

* Every OTHER test that named a module in the plane is now a test with no
  subject. Three kinds turned up — a boundary bite-test whose planted violation
  can no longer be planted, a composition test importing the plane's driver, and
  an anti-vacuity control that proved a rule was not so narrow it rejected its own
  real consumers. The first is re-pointed at a surviving plane, the second is
  excised, and the third has no replacement and was removed with a note saying
  what was lost. Fixing the first two by asserting less would have been the exact
  failure this document exists to catch.
* Any calibrated constant measured against the deleted lines — here the deferred
  line floor — must be re-derived from the tree that is left, not lowered until
  the suite goes quiet.
* A plane rarely leaves cleanly. `controller/` could not: two modules the operator
  kept reached into the removed plane for six lines, which now live in
  `controller/shared.py`. Budget for a survivors' module before starting.
