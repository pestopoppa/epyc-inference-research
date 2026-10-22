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
| **ON THE CAMPAIGN PATH** | **165,198** |
| **DEFERRED** (provably unreachable) | **125,029** |
| **TOTAL** | **290,227** |

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

Rows added 2026-10-20 (the `loop/`, `controller/`, `history_replay/` and later
`execution/` modules that landed after 2026-08-12 without a row) carry the first
sentence of the module's own docstring plus a mechanical reachability note; they
state what a module IS and are not incident write-ups. Replace one with the
incident or measured fact when you next touch the module.

| module | lines | campaign #1 | reason |
|---|---:|:---:|---|
| `campaign.py` | 5,968 | yes | THE ENTRYPOINT. Before it landed, `grep -rl "__main__|argparse|def main("` over every non-test module returned nothing: 94k lines, 5,695 passing tests, and no way to start it — which is the whole reason this package has produced no result |
| `dashboard.py` | 201 | yes | the terminal result was fsynced but the only dashboard exporter had been deleted, so active AutoKernel work remained permanently absent from the operator surface; this compact projection dates itself from the journal entry and cannot make an old campaign fresh |
| `__init__.py` | 14 | yes | package docstring; `schemas` is declared here as the single source of record shape |
| `schemas.py` | 3,825 | yes | one record shape — every module is written against it and none invents its own |
| `journal.py` | 4,136 | yes | AutoPilot lost 232 trials and ~16 days when a restart came up empty and nothing objected |
| `fault_rehearsal.py` | 1,080 | no | governed process-only acceptance producer; real crash/restart, advisory revocation and tamper evidence must remain operator-invoked and cannot gain campaign mutation, inference, benchmark, kernel, stack or release authority |
| `offline_least_commitment.py` | 465 | no | AP-WM-1 observe-only archive analysis; importing an offline hypothesis diagnostic into the mutation/build path would give it accidental live authority |
| `least_commitment_archive_builder.py` | 429 | no | AK-WM-2a strict real-record projection; it reads hash-bound completed campaign evidence into an observe-only archive and cannot launch, rank, mutate, or promote |
| `least_commitment_receipts.py` | 1,221 | no | AK-WM-2a governed offline producer; it copies only explicitly SHA-pinned journal fields into receipts, rejects torn completed journals, and cannot launch, rank, mutate, or promote |
| `least_commitment_capture.py` | 511 | yes | prospective hash-bound IQK diagnostic/control contract; live campaigns only reduce declared outcome functions and it exposes no selector, champion, release, process, or inference API |
| `least_commitment_heldout.py` | 639 | yes | prospective held-out trust boundary; projects the effect only from a clean distinct completed campaign journal, derives the cross-regime candidate frame, and replays that projection before a target campaign can claim resources |
| `prepare_iqk_matched_pair.py` | 694 | no | deterministic pre-claim intervention/control publisher; derives shared seeds and the complete one-factor frame, validates both capture plans, and explicitly cannot infer, build, claim, execute, or mutate a journal |
| `evidence_path_rehearsal.py` | 279 | no | architecture-only CLI proving proposal/capture/control/AP-WM/champion/readiness/T3/package producer coverage; emits no empirical evidence or live authority |
| `turn_productivity.py` | 481 | no | AK-PT-1/AK-X-6 archive reducer; it consumes completed refine-turn records and may only withhold future search advancement, so campaign #1 must not give it live rank or mutation authority |
| `prior_art.py` | 706 | yes | deterministic proposal-input compiler; it classifies findings before a proposal exists and has no place in the mutation/build process |
| `profile_report.py` | 587 | yes | RVP-1–7 deterministic offline C4 report; it consumes completed paired traces and has no mutation, build, profiler-launch or ranking authority |
| `profile_context.py` | 287 | yes | C4 hash-bound discovery/evaluator projection; it exposes diagnostic context to authoring without gaining verdict or ranking authority |
| `placement_context.py` | 216 | no | P2-5j hash-bound placement-belief projection; it exposes observation-only topology context to authoring without gaining selection, ranking, launch, or promotion authority |
| `hipkittens_lds.py` | 430 | no | INF-03 deterministic offline gfx90a LDS topology reducer; it consumes completed counter captures and gives authoring diagnostic context without profiler-launch or ranking authority |
| `c5_seed_corpus.py` | 603 | yes | C5 static HyRA task registry; it contributes hash-bound, non-numeric gfx90a authoring context and must never gain campaign mutation, scoring, or NVIDIA-to-MI210 claim-transfer authority |
| `c5_rocm_oracle.py` | 880 | no | C5 offline correctness-provider contract; it re-attests pinned source bytes, stages an unconditional no-timing driver, and reduces raw trace bytes without gaining inaugural campaign execution, scoring, ranking, or promotion authority |
| `datatype_targets.py` | 134 | no | INF-03 static FP8/NVFP4 authoring contracts; they expose no cross-vendor performance numbers and have no campaign mutation, execution, scoring, or capability-attestation authority |
| `substrate.py` | 340 | no | validated planning facts; it reads checked-in measured/datasheet receipts before proposal construction and never joins the mutation/build path |
| `lanes.py` | 314 | no | screening declarations and rank-transfer calibration; without measured calibration campaign #1 stays on the full verified path |
| `artifact_diff.py` | 200 | yes | AK-TR-6 must veto an unconfirmed GPU claim before the behavioral T0 provider can launch |
| `candidate_record.py` | 293 | yes | every executed candidate is fsynced from the exact built snapshot and evaluation event identities before terminal STOP |
| `source_candidate.py` | 530 | yes | source-changing proposals consume one immutable embedded patch bundle through the guarded worktree mutation boundary |
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
| `evaluator/correctness.py` | 4,235 | yes | throughput is reward-hackable: deleting the computation is the fastest kernel there is |
| `evaluator/recipes.py` | 2,471 | yes | argv from a hashed constructor — production drifted off NUMA interleave 2026-05-24 and the front door ended up at 46% of canonical |
| `evaluator/devices.py` | 813 | yes | a GPU cell must not be satisfied by `Device 0: CPU` |
| `evaluator/controls.py` | 2,406 | yes | the A/A control plane — 2026-08-04 measured 1.62% / 1.88% between-run CV over four identical runs |
| `evaluator/baseline_honesty.py` | 186 | no | AK-BH-4 exact-surface strongest-provider selector; it rejects AUTO and cross-surface transfer before a campaign can claim an honest floor |
| `evaluator/c3_epyc_suite.py` | 805 | yes | INF-48 offline C3/C5 contract reducer; it binds EPYC op cases to exact vendor evidence and a captured-workload whole-model exit without launch, mutation, release, or promotion authority |
| `evaluator/c3_epyc_compiler.py` | 523 | no | INF-48 offline JSON plan/receipt compiler; it is the callable controller/backend seam but cannot launch captures, benchmarks, builds, patches, inference, or promotion |
| `evaluator/c3_epyc_tensor_capture.py` | 1,425 | yes | INF-48 live inventory/request compiler and governed real-model producer; dual authorization, canonical unexpired claims, descendant KFD/device sampling, bounded teardown, immutable journal slice, and released-claim window gate Apex without promotion authority |
| `evaluator/c3_epyc_capture_provider.py` | 178 | yes | INF-48 tracked real-model provider protocol; admits only clean, hash-bound exact-case/model hooks returning ordered ROCm tensors and emits a typed missing-hook refusal rather than substituting HyRA references or synthetic tensors |
| `evaluator/c3_apex_runner.py` | 1,222 | no | INF-48 exact Apex mapping seam: binds one reviewed gfx90a k228 trace or the ordered branch-aware k175 composite plus governed tensor/source/model/toolchain identities, and remains outside campaign mutation and candidate authoring |
| `evaluator/sensitivity.py` | 333 | yes | RVP-C2-7/C2-11/C5-2 offline two-axis reducer; its report has no live authority until `source_candidate_authority.py` binds exact candidate/evaluator/evidence provenance, and missing or insensitive populations remain unscoreable |
| `evaluator/oracle_integrity.py` | 145 | yes | RVP-C2-8/C2-9 offline reducers; hostile/checker results gain T0 authority only through the exact provenance bridge and never independently rank a candidate |
| `evaluator/source_candidate_authority.py` | 87 | yes | fail-closed bridge from offline sensitivity/hostile/checker outputs to live T0: binds source, evaluator bundle, suite, capture mode and evidence hashes without launching or mutating anything |
| `evaluator/historical_tasks.py` | 194 | no | RVP-C5-R/C3-2 sealed historical-task descriptor and expert-ceiling reducer; no terminal candidate means `COULD_NOT_CHECK` |
| `evaluator/rebench_scoring.py` | 120 | no | AK-RB-1 offline reference-normalized scoring and matched-budget curve reducer; it consumes completed behavior checks and timings but has no campaign mutation, execution, or verdict authority |
| `evaluator/statistics.py` | 3,689 | yes | **calibration constants and `median` only.** Its e-process made the gate unpassable: threshold 10 against a sign-martingale that tops out at 5.5687, at every effect size. Fenced by `TestNoOptionalStopping` |
| `evaluator/integrity.py` | 3,661 | yes | **provenance primitives only** — `sha256_file`, `hash_source_tree`, `EMPTY_TREE_SHA256`, the clean-build snapshot check. Its §8.5.1 gate runner is fenced off |
| `evaluator/surface.py` | 3,195 | yes | **change-class constants only** — `AffectedSurface`, the core/shared-header fanout classes. `SurfaceGateRunner` is fenced off |
| `execution/__init__.py` | 24 | yes | docstring only; states the deny-8 limits every executor inherits |
| `execution/worktree.py` | 3,203 | yes | no candidate exists without it: production-tip anchoring, campaign worktree, build, build-identity receipt |
| `execution/provider.py` | 422 | yes | source-available ROCm providers bind a clean exact commit, tracked-file/toolchain/linkage/licence/executable identities and execute only an explicitly authorized prefix-bearing build under no-network write confinement; outputs remain non-bankable until clean llama_gpu integration |
| `execution/microbench.py` | 4,734 | yes | paired ALTERNATING blocks plus C6-10 ranked hard cases — each hostile unit changes the receipted recipe and contributes blocks to the same rank instead of living only in a correctness gate |
| `execution/device_sampler.py` | 410 | yes | RVP-C3-4 numeric 250 ms ROCm state producer; it brackets the exact captured benchmark-process lifetime and fails closed on missing fields, failed probes, empty traces, or cadence gaps |
| `execution/instrument_integrity.py` | 111 | yes | RVP-C6-1: a candidate binary is built from candidate-controlled source, so every live T1 invocation must re-pin its reward-bearing translation unit to the named anchor before it can emit a number |
| `execution/physical_bounds.py` | 197 | yes | RVP-C6-4 physical impossibility screen: per-shape conservative work floors and hardware peak ceilings are bound to the exact delivered unit and recipe/model/parameter frame, then every live sample is checked before ranking |
| `execution/powercap_broker.py` | 248 | yes | the v9 CPU preflight could not read root-owned 0400 package counters, while running the campaign as root correctly failed the non-root candidate sandbox; a captured networkless read-only container now exposes only exact package-energy integers |
| `execution/reward_hack_scan.py` | 265 | yes | RVP-C6-6/C6-9 plus static C6-2/C6-3 detectors: protected-frame, pointer-memo, structured-shortcut, environment/timing and stream/thread findings; the named 10 planted/15 clean corpus states sensitivity/specificity/FPR, not arbitrary-program coverage |
| `execution/reward_hack_corpus.py` | 586 | no | operator-invoked instrument producer that compiles and runs the named 10 planted/15 clean HIP corpus on gfx90a under the shared device claim, then prospectively writes 3 detector and 50 exact case×ranked-unit observations with no candidate-speed or campaign authority |
| `execution/sandbox.py` | 1,076 | yes | C6/INF-03 process boundary: default candidate profile retains Landlock write confinement; broker-only controller profile default-denies networking and read/exec outside exact roots/files/executables while inheriting one peer-bound broker UDS; direct model profile binds fixed Claude/Bun self/kernel reads to the actual model PID, permits outbound INET clients, and denies broker/GPU/new AF_UNIX/server authority; evaluator profile exposes only exact ROCm devices and pinned runtime inputs while denying networking, broker inheritance, cross-process memory and io_uring; all retain signal/ptrace/namespace denials, finite limits, owned cgroup and policy-hashed receipts |
| `execution/t0_provider.py` | 4,143 | yes | the predecessor harness tested MUL_MAT only, so a kernel that broke MUL_MAT_ID — MoE dispatch, every token in production — passed it cleanly |
| `execution/control_runner.py` | 1,859 | yes | runs the neutral / A-A controls that the measured drift makes mandatory rather than optional |
| `execution/live_controls.py` | 2,514 | no | standalone, operator-invoked calibration producer for the fixed five controls; it prepares the instrument before campaign #1 and is deliberately not imported by the mutation/build entrypoint |
| `execution/cpu_region_claim.py` | 2,408 | yes | 2026-08-04: two A/A runs were destroyed by a legitimate co-tenant because the loop held no claim. Before this module a claim could be READ but never acquired |
| `execution/chain.py` | 1,928 | yes | holds the seams — four mismatches between executors and evaluator, one of them a field whose meaning INVERTS across the seam |
| `resource/__init__.py` | 28 | yes | docstring only; names the `resource`-shadows-stdlib hazard the loop must not trip |
| `resource/device_claim.py` | 1,826 | yes | §2.6's first row of substrate that exists nowhere in the project: a cross-process GPU device claim someone actually holds |
| `resource/preflight.py` | 1,837 | yes | INC-20260731: a name-pattern kill took out another agent's `llama-server` twice, and `earlyoom`, whose argv names what it guards |
| `resource/claim_witness.py` | 325 | yes | invariant 9 — idle sensing is never a claim, and the witness is what tells the two apart |
| `controller/__init__.py` | 89 | yes | binds every surviving controller module, so importing one reaches both — which is why `controller.do_not_repeat` is on the path whether or not the driver names it, and why `CONTROLLER_ALLOWED` lists this file rather than leaving the edge unexplained |
| `controller/champion.py` | 1,552 | no | AK4 lean lifecycle projection and composition transaction: consumes validated journal evidence, requires a real combined rebuild/evaluation through an injected runner, and carries no source-mutation, build, benchmark, process, release, or production-write capability |
| `controller/sequencer.py` | 468 | no | AK4 deterministic outer ordering seam: consumes supplied proposals and injected execution capabilities, journals bounded stop states, and remains unbound from `controller.__init__` so campaign #1 cannot acquire a second live loop |
| `controller/completed_campaign_adapter.py` | 134 | no | strict live-journal-to-sequencer join; banks only event-bound dispatch/mechanism-confirmed wins above floor and MDE and launches nothing |
| `controller/arena_adapter.py` | 546 | no | INF-03 paper-pin GEAK/AgentKernelArena integration: validates clean vendor sources and physical gfx90a identity, binds C4 into a hygienic priced prompt, and launches registered whole-agent task adapters without entering campaign mutation or scoring; callback failures reap the exact captured process group |
| `controller/arena_controller_sandbox.py` | 689 | no | INF-03 controller/model isolation adapter: exact copied workspace plus pinned Python/Codex/Claude/Node/source/CA runtime discovery and fixed Bun volatile reads, broad/device/production/campaign/symlink refusal, broker-only controller prefix, per-model-PID outbound-client prefix, exact activation verification, and descendant-draining cgroup teardown; no direct GPU or evaluator profile |
| `controller/claude_codex_actor_critic.py` | 882 | no | INF-03 bounded Claude-planner/critic + Codex-actor controller: exact three-argument Arena launcher, explicit model/effort pins, brokered model and candidate-evaluation calls, measured starting-state fallback and best-candidate materialization, eight-item measurement/critic revision memory projected into each next planner and fully receipt-bound, semantic task-manifest confinement with four exact receipt-bound/scrubbed control-plane roots excluded and refused as candidates, launcher-cgroup-owned timeout teardown, and transcript/outcome/artifact hashing; its preflight invokes neither CLI |
| `controller/codex_container_actor.py` | 257 | no | INF-03/AK-LE-3 host-compatible Codex actor boundary: pins a read-only container image, exposes exactly one writable host bind for the copied Arena workspace, stages auth only in an automatically erased temporary directory, and admits exactly the reviewed gpt-5.6-sol/gpt-5.6-terra high-effort model cells without granting host filesystem writes |
| `controller/arena_campaign.py` | 1,073 | no | INF-03 matched-panel audit and injected-runner seam: binds the baseline plus seven controller arms to equal tasks and checkpoints, admits each governed arm only with exact adapter/upstream/CLI/model pins, refuses incomplete panels before execution, and remains outside live campaign mutation and scoring |
| `controller/arena_cell_runner.py` | 3,697 | no | INF-03 fail-closed runner: parent-owned authenticated broker serializes model and evaluation frames; direct model CLIs receive fresh per-PID outbound-client sandboxes, writable Codex actors retain an attested digest-pinned single-bind container, and every candidate evaluation enters a fresh GPU evaluator sandbox under exact claims; prospective intermediate correctness/timing rows bind producer, source, ordinal, baseline and window evidence and are exactly re-derived on restore |
| `controller/arena_evaluator_child.py` | 207 | no | INF-03 claim-blind evaluator child: strictly reconstructs the hash-bound starting baseline, verifies pinned Python/config/vendor identities, invokes the vendor evaluator only inside the dedicated deny-network GPU sandbox, and emits one strict self-hashed JSON result |
| `controller/arena_upstream_common.py` | 800 | no | INF-03 licensed-controller client: bounded read-only text-model calls are GPU-blind only after an OS-level device-open proof; the controller accepts the exact parent-declared source tuple without importing PyYAML/vendor evaluator code, and complete candidate bytes receive feedback solely through interrupt-safe writes to the one preconnected authenticated parent-worker broker descriptor—never destination-bearing send or new socket authority |
| `controller/evoengineer_arena.py` | 502 | no | INF-03 exact-source EvoEngineer-Full seam: pins the paper-era MIT source, preserves its named operators/parameters, and runs its executable CLI only through the parent-worker evaluation broker and device-isolating hash-bound launcher |
| `controller/k_search_arena.py` | 284 | no | INF-03 licensed K-Search world-model/tree port: injects the centralized Arena evaluator into the exact pinned upstream Task seam without entering campaign mutation or scoring |
| `controller/geak_v1_arena.py` | 345 | no | INF-03 licensed GEAK-v1 OptimAgent port: maps its ROCm dataset and reflection loop onto centralized Arena evidence without entering campaign mutation or scoring |
| `controller/xe_forge_arena.py` | 594 | no | INF-03 licensed Xe-Forge gfx90a port: retains the pinned DSPyEngine linear-CoVeR analysis/planning loop, replaces process-global device prompts with scoped AMD MI210 guidance, uses a no-shape executor gate, and routes initialized KernelBench execution through Arena |
| `controller/kernelfoundry_arena.py` | 482 | no | INF-03 licensed KernelFoundry gfx90a port: retains inherited Controller.run_single MAP-Elites/island branching, strictly activates pinned Triton feature patterns, binds measured parent transitions to upstream QD tracking, and routes evaluation through Arena |
| `controller/arena_roundtrip.py` | 216 | no | EVL-47 SC20/SC21 prospective GEAK/Arena receipt producer; emits only observed correctness/timing rates and retains preflight as non-ordinal dependency evidence, with no launch, mutation, or verdict authority |
| `controller/hip_authoring_arm.py` | 627 | no | governed raw-HIP compatibility seam: pins and hashes a true Torch2HIP task/candidate/toolchain, compiles GPU-blind for gfx90a, and scopes distinct MI210 claims to baseline/final evaluation; emits observation-only correctness/timing-validity rows with no ranking, campaign, production-tree, or promotion authority |
| `controller/hip_decision_grade.py` | 516 | no | task-local raw-HIP decision evaluator: source-before-suite sealing, unseen hostile inputs, independent host-double reduction, static C6 scan, exact Torch-ROCm-compile C3 provider, per-arm gfx90a duration admission, paired e-process reduction, and no release/promotion authority |
| `controller/hip_decision_grade_worker.py` | 232 | no | C6-contained raw-HIP child: receives inputs but never expected outputs, performs double-poison/determinism checks, and captures exact candidate/provider timings plus Torch-Inductor implementation identity |
| `controller/authoring_contract.py` | 469 | yes | AK-PL-1/AK-LE-4/AK-LE-5 pre-proposal adapter: fully rendered prompt leak refusal, priced never-bulk-read context, reversible compaction, and structured external numbers; it calls no model and must not gain mutation/build authority |
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
| `check_regrowth_guards.py` | 272 | no | Enforced guards against the growth pattern that produced this codebase. Not reachable from the declared campaign-#1 roots. |
| `check_suite_floors.py` | 224 | no | Refuse a CI run whose suites silently got smaller. Not reachable from the declared campaign-#1 roots. |
| `codegen_summary.py` | 678 | yes | Bounded, non-authoritative codegen evidence for a retained AutoKernel build. On the campaign path: first reached via `campaign`. |
| `conftest.py` | 114 | no | Host-isolation for every AutoKernel unit test. Not reachable from the declared campaign-#1 roots. |
| `controller/anchor_integrity.py` | 184 | no | Code-section digests for the anchor guard's hash pre-check (R22-3). Not reachable from the declared campaign-#1 roots. |
| `controller/build_recipe.py` | 357 | no | The GPU build recipe, versioned, with a stated reason for every divergence. Not reachable from the declared campaign-#1 roots. |
| `controller/champion_seed.py` | 145 | no | Derive a sealed :class:`champion.AnchorIdentity` by MEASURING a real tree. Not reachable from the declared campaign-#1 roots. |
| `controller/claude_fable5_critic_actor.py` | 723 | no | Launch one hash-bound Claude Fable 5 critic with no local capabilities. Not reachable from the declared campaign-#1 roots. |
| `controller/discovery_controller.py` | 3,973 | no | Candidate-only AutoKernel discovery controller. Not reachable from the declared campaign-#1 roots. |
| `controller/discovery_deployment.py` | 709 | no | Declarative, registry-only configuration for GPU source discovery deployment. Not reachable from the declared campaign-#1 roots. |
| `controller/discovery_deployment_factory.py` | 3,079 | no | Trusted concrete deployment materializer for governed GPU source discovery. Not reachable from the declared campaign-#1 roots. |
| `controller/discovery_static_registry.py` | 3,523 | no | Static, non-JSON construction of the GPU source discovery build seam. Not reachable from the declared campaign-#1 roots. |
| `controller/discovery_supervisor.py` | 1,839 | no | Durable, sealed, host-owned supervisor for AutoKernel discovery controllers. Not reachable from the declared campaign-#1 roots. |
| `controller/discovery_supervisor_secure.py` | 382 | yes | Filesystem and process containment primitives for discovery_supervisor. On the campaign path: first reached via `loop.campaign_control`. |
| `controller/discovery_telemetry.py` | 317 | no | Bounded, secret-free live telemetry for discovery actors. Not reachable from the declared campaign-#1 roots. |
| `controller/experiments.py` | 936 | yes | Durable experiment memory that OUTLIVES a deployment. On the campaign path: first reached via `loop.archive`. |
| `controller/experiments_fixture.py` | 64 | no | Load the REAL experiment slice that `experiments_fixture.json` carries. Not reachable from the declared campaign-#1 roots. |
| `controller/gate_parameters.py` | 134 | no | Every decision parameter, recorded beside the result it decided. Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_hot_residency_runner.py` | 249 | no | Hardware-free lifecycle for a persistent, already GPU-resident process. Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_load_admission.py` | 584 | no | Pure, fail-closed GPU model-load admission policy. Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_residency_sampler.py` | 140 | no | Static MI210 KFD/VRAM sampler for governed source proof commands. Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_source_adapter.py` | 1,422 | no | Crash-conservative controller adapter for governed GPU source evidence. Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_source_evidence.py` | 3,932 | no | Governed evidence producer for two-arm GPU source discovery. Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_source_proofs.py` | 113 | no | Typed proof reducers for source-patch GPU discovery (no throughput authority). Not reachable from the declared campaign-#1 roots. |
| `controller/gpu_utilization.py` | 107 | no | What fraction of a held GPU claim the loop actually used. Not reachable from the declared campaign-#1 roots. |
| `controller/inbox.py` | 58 | no | Read the operator inbox without letting it end the run (R22-6). Not reachable from the declared campaign-#1 roots. |
| `controller/rung_confirm.py` | 182 | no | The confirm half of the two-rung screen/confirm keep gate (D1-D6, 2026-09-01). Not reachable from the declared campaign-#1 roots. |
| `controller/split_runtime_verifier.py` | 562 | no | Fail-closed verification for an AutoKernel split GPU reward runtime. Not reachable from the declared campaign-#1 roots. |
| `controller/workload_contract.py` | 417 | yes | The screening workload must dispatch the kernels production dispatches. On the campaign path: first reached via `loop.census`. |
| `evaluator/footprint.py` | 397 | no | Static patch-footprint producer for the cross-workload keep screen. Not reachable from the declared campaign-#1 roots. |
| `execution/inference_window.py` | 298 | yes | Cross-process exclusion for the brief interval in which a model is resident. On the campaign path: first reached via `campaign`. |
| `execution/screening_baseline.py` | 406 | yes | Immutable amortized baseline bank for non-promotable discovery screens. On the campaign path: first reached via `campaign`. |
| `execution/server_generation.py` | 108 | yes | Original llama-server v1 response facts for owning T0 reduction. On the campaign path: first reached via `execution.t0_provider`. |
| `heldout_bound_pipeline.py` | 354 | no | Atomically bridge a completed decode pair to heldout-bound prefill + AP-WM. Not reachable from the declared campaign-#1 roots. |
| `history_replay/__init__.py` | 5 | no | AK-HR-1: offline, read-only census and exporter over the closed AutoKernel history. Not reachable from the declared campaign-#1 roots. |
| `history_replay/ak_hr2_kappa.py` | 81 | no | AK-HR-2 stage 1: rule parser vs independent labeller on the 40 held-out DS41/Q38FN hypotheses. Not reachable from the declared campaign-#1 roots. |
| `history_replay/ak_hr2_render_audit.py` | 436 | no | AK-HR-2(iii): what did the planner's `## Already tried` block actually show? Not reachable from the declared campaign-#1 roots. |
| `history_replay/ak_hr2_taxfree.py` | 324 | no | AK-HR-2 taxonomy-free parts, run after the held-out agreement gate FAILED (kappa < 0.6). Not reachable from the declared campaign-#1 roots. |
| `history_replay/ak_hr2_taxonomy.py` | 161 | no | AK-HR-2 frozen mechanism taxonomy v1 and its deterministic rule parser. Not reachable from the declared campaign-#1 roots. |
| `history_replay/export.py` | 936 | no | Closed-store history exporter and census (AK-HR-1). Not reachable from the declared campaign-#1 roots. |
| `history_replay/rex_arm_replay.py` | 482 | no | AK-RX-1/1.1: offline REx-arm policy-panel replay (iid and split-budget controls) and the repair-chain adapter over closed `experiments.db` stores. Not reachable from the declared campaign-#1 roots. |
| `hypothesis_portfolio.py` | 1,220 | no | Strict, read-only loader for the discovery hypothesis portfolio. Not reachable from the declared campaign-#1 roots. |
| `loop/__init__.py` | 10 | yes | The AutoKernel discovery loop, rebuilt. On the campaign path: first reached via `controller.experiments`. |
| `loop/a2_execution_state.py` | 421 | yes | Closed durable state for controller-owned A2 runtime phase events. On the campaign path: first reached via `journal`. |
| `loop/ab_server_pair.py` | 204 | no | AKR-SG-2: launch an A/B llama-server pair on private ports with port-ownership, ggml-linkage and residency proofs (used only by the shakedown driver inside a confirmed window). Not reachable from the declared campaign-#1 roots. |
| `loop/accumulate.py` | 796 | no | Compound-then-gate: batch cheap bench keeps until their compounded gain clears the serving floor, THEN spend the expensive serving gate once (R23-44, operator directive 2026-09-04: "collect llama-bench keeps until they compound to 2x-3x noise floor before the llama-server final c Not reachable from the declared campaign-#1 roots. |
| `loop/actor_context.py` | 709 | yes | Variable-mode actor context: the bundle is a DIRECTORY, the prompt is its INDEX. On the campaign path: first reached via `loop.actors`. |
| `loop/actor_lifecycle.py` | 341 | yes | Concrete actor-stage adapter over the owning campaign-controller path. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/actor_metrics.py` | 658 | yes | Per-call turn/efficiency metrics for the opencode actor seat (planner/author/critic). On the campaign path: first reached via `loop.actors`. |
| `loop/actor_opencode_config.py` | 970 | yes | Per-run opencode config for the autokernel planner/author seat. On the campaign path: first reached via `loop.actors`. |
| `loop/actor_orchestrator.py` | 541 | yes | The `orchestrator` actor Backend kind (INF-78 OAB-2). On the campaign path: first reached via `loop.actors`. |
| `loop/actor_passthrough.py` | 205 | yes | Route LOCAL-model actor calls through the orchestrator passthrough (2026-10-04). On the campaign path: first reached via `loop.actors`. |
| `loop/actor_preparation.py` | 632 | yes | Selected source/build actor preparation over an owned-worker authority seam. On the campaign path: first reached via `loop.actor_lifecycle`. |
| `loop/actor_preparation_state.py` | 238 | yes | Closed native state machine for selected actor-preparation accounting. On the campaign path: first reached via `journal`. |
| `loop/actor_serving.py` | 463 | yes | Client-side serving parameters for an actor talking to a LOCAL llama-server (UFH14-B1, F1). On the campaign path: first reached via `loop.actor_passthrough`. |
| `loop/actor_tools_mcp.py` | 1,015 | no | Bounded-output code and profile tools for the autokernel planner/author actor. Not reachable from the declared campaign-#1 roots. |
| `loop/actors.py` | 4,261 | yes | Real planner and critic, driven by an external coding agent. On the campaign path: first reached via `loop.actor_lifecycle`. |
| `loop/ak_check.py` | 1,498 | yes | `ak-check`: the AutoKernel author's scratch sanity-check sandbox (operator 2026-09-26). On the campaign path: first reached via `loop.actors`. |
| `loop/anchor.py` | 211 | no | Prove the promoted anchor IS the champion, before the loop draws any more work. Not reachable from the declared campaign-#1 roots. |
| `loop/archive.py` | 658 | yes | Champion source identity and immutable measurement history. On the campaign path: first reached via `loop.serving_beliefs`. |
| `loop/belief_context.py` | 813 | yes | Read-only Vidya belief context for the AutoKernel planner, and its receipt. On the campaign path: first reached via `loop.actors`. |
| `loop/bench.py` | 702 | yes | Build one arm, measure it, compare the pair. On the campaign path: first reached via `loop.gates`. |
| `loop/bestof.py` | 1,530 | no | Best-of-N concurrent authoring: N author calls for ONE accepted hypothesis. Not reachable from the declared campaign-#1 roots. |
| `loop/calibrate_served_shapes.py` | 849 | no | Calibrate the ppl_contract layer (a) served-shape case set against the ANCHOR. Not reachable from the declared campaign-#1 roots. |
| `loop/campaign.py` | 852 | yes | Pure campaign enrollment for the unified AutoKernel loop. On the campaign path: first reached via `loop.actor_preparation`. |
| `loop/campaign_cli.py` | 263 | no | Offline campaign resolver; this module has no launch or resource-claim path. Not reachable from the declared campaign-#1 roots. |
| `loop/campaign_command_v2.py` | 117 | yes | Closed supervisor-fenced campaign command v2 and journal transition v3. On the campaign path: first reached via `journal`. |
| `loop/campaign_control.py` | 4,723 | yes | Durable, fenced campaign controls; no worker or resource authority. On the campaign path: first reached via `loop.actor_lifecycle`. |
| `loop/campaign_service.py` | 786 | yes | Offline authenticated HTTP/CLI surface for campaign controls. On the campaign path: first reached via `loop.standalone_inputs`. |
| `loop/candidate_cli.py` | 266 | no | Offline-only validator and summary for candidate manifests and batches. Not reachable from the declared campaign-#1 roots. |
| `loop/candidate_manifest.py` | 1,291 | yes | Pure immutable candidate, LOO, validation-batch, and pointer contracts. On the campaign path: first reached via `loop.candidate_transactions`. |
| `loop/candidate_transactions.py` | 993 | yes | Durable candidate transitions under the campaign controller's sole writer. On the campaign path: first reached via `loop.retention_consumer`. |
| `loop/capacity_probe.py` | 145 | no | G5 capacity dimension: peak footprint over a MIXED request sequence. Not reachable from the declared campaign-#1 roots. |
| `loop/census.py` | 197 | yes | Callback-free, per-workload scheduler census producer. On the campaign path: first reached via `loop.gates`. |
| `loop/champion.py` | 242 | no | Startup refusal: the loop optimises THE single champion branch, or does not start. Not reachable from the declared campaign-#1 roots. |
| `loop/claim.py` | 786 | yes | The GPU claim. On the campaign path: first reached via `loop.actor_passthrough`. |
| `loop/claims.py` | 78 | yes | Typed claims carried by AutoKernel experiment records. On the campaign path: first reached via `controller.experiments`. |
| `loop/cpu_fa_mask_capture.py` | 89 | no | Prepare native DS41 capture provenance before a separately governed CPU run. Not reachable from the declared campaign-#1 roots. |
| `loop/cpu_fa_reference.py` | 664 | yes | FLASH_ATTN_EXT case set, anchor bit-identity gate and paired perf screen for the `cpu_fa_schedule` route (audit 2026-10-04 `ak-longctx-audit` C2/C3). On the campaign path: first reached via `loop.gates`. |
| `loop/cpu_fa_verifier_record.py` | 190 | yes | Prospective native real-mask verifier custody; no grading rule or replay claim. On the campaign path: first reached via `loop.cpu_fa_reference`. |
| `loop/cpu_fusion_reference.py` | 253 | yes | Independent check of the DS41 hc_mixes chain, RMS_NORM -> MUL_MAT(F16), for fusion edits. On the campaign path: first reached via `loop.cpu_route_witness`. |
| `loop/cpu_norm_reference.py` | 343 | yes | Independent RMS_NORM check for the `cpu_norm_rowsplit` route (DS41 hc_mixes norm). On the campaign path: first reached via `loop.cpu_route_witness`. |
| `loop/cpu_profile.py` | 1,480 | yes | Owned, request-scoped CPU perf observations; no correctness or promotion authority. On the campaign path: first reached via `loop.serving`. |
| `loop/cpu_quant_reference.py` | 378 | yes | Independent output check for fixed CPU Q4_K/Q5_K/Q8_0 (and F16/BF16/F32) matmul probes. On the campaign path: first reached via `loop.cpu_route_witness`. |
| `loop/cpu_repack_cases.py` | 801 | yes | CPU_REPACK MUL_MAT_ID correctness gate for the DS41-1017 lane: exercises the repacked MXFP4 expert kernel that test-backend-ops never reaches. On the campaign path: first reached via `loop.gates`. |
| `loop/cpu_route_gdb_probe.py` | 46 | no | Trusted GDB-side record for a stopped, single-case CPU source-route probe. Not reachable from the declared campaign-#1 roots. |
| `loop/cpu_route_witness.py` | 369 | yes | Engagement witness plus independent scalar reference for the widened CPU routes. On the campaign path: first reached via `loop.gates`. |
| `loop/cpu_screen.py` | 573 | no | Reduced common CPU conditions in the existing loop, not transfer authority. Not reachable from the declared campaign-#1 roots. |
| `loop/cpu_window.py` | 781 | no | CPU windows: yield the CPU-region claim during actor phases and say so (2026-09-26). Not reachable from the declared campaign-#1 roots. |
| `loop/cross_target.py` | 615 | no | Cross-target lineage under the ONE-champion invariant: lane working branches fold into the single global champion. Not reachable from the declared campaign-#1 roots. |
| `loop/direct_gpu_control.py` | 947 | no | Prospective replay of the retained, single-mechanism GPU x4-load control. Not reachable from the declared campaign-#1 roots. |
| `loop/direct_historical_control.py` | 676 | no | Original Qwen CPU IQK control, held by the existing direct-loop owner. Not reachable from the declared campaign-#1 roots. |
| `loop/discovery_execution.py` | 149 | no | Controller-owned durable persistence for A2 runtime discovery screens. Not reachable from the declared campaign-#1 roots. |
| `loop/discovery_screen.py` | 825 | yes | Generic Annex K A2 runtime screening with immutable three-sample banks. On the campaign path: first reached via `loop.a2_execution_state`. |
| `loop/dispatch_guard.py` | 281 | no | Durable answered-implies-closed dispatch identity for AutoKernel candidates. Not reachable from the declared campaign-#1 roots. |
| `loop/driver_execution.py` | 1,034 | yes | Connect one durably selected runtime comparison to the owned worker bridge. On the campaign path: first reached via `loop.native_final_trial`. |
| `loop/epoch_aliases.py` | 315 | no | Full-epoch -> measurement-epoch aliases for the AutoKernel loop (OP-60). Not reachable from the declared campaign-#1 roots. |
| `loop/evidence_cli.py` | 91 | no | Offline inspection CLI for the scoped evidence projection. Not reachable from the declared campaign-#1 roots. |
| `loop/evidence_feed.py` | 1,512 | yes | Bounded Journal -> Vidya ledger -> scoped-evidence projection. On the campaign path: first reached via `loop.feed_runtime`. |
| `loop/experiment_cli.py` | 67 | no | Offline validator for immutable experiment plans and recorded units. Not reachable from the declared campaign-#1 roots. |
| `loop/experiment_plan.py` | 984 | yes | Immutable experiment plans and evidence-use checks. On the campaign path: first reached via `loop.a2_execution_state`. |
| `loop/feed_runtime.py` | 509 | yes | Execution-thread-owned continuous evidence; configuration is never authority. On the campaign path: first reached via `loop.evidence_feed`. |
| `loop/fold2_gates.py` | 378 | no | The FOLD-2 correctness/perf gate battery, run by hand with the loop dead and GPU free. Not reachable from the declared campaign-#1 roots. |
| `loop/fresh_correctness.py` | 202 | yes | Opt-in, non-ranked fresh-input correctness receipt for one benchmark arm. On the campaign path: first reached via `loop.bench`. |
| `loop/gates.py` | 4,293 | yes | Correctness gates. On the campaign path: first reached via `loop.loop`. |
| `loop/gdn_reference.py` | 131 | yes | Independent, exact F32 GDN fixture for the CPU AutoKernel lane. On the campaign path: first reached via `loop.gates`. |
| `loop/gpu_phases.py` | 161 | no | Original local resource owners for a GPU run without a run-wide CPU claim. Not reachable from the declared campaign-#1 roots. |
| `loop/gpu_serving_profile.py` | 784 | no | GPU SERVING profile as a planner input (G2, AK long-context audit 2026-10-04 §4.2). Not reachable from the declared campaign-#1 roots. |
| `loop/gpu_window.py` | 169 | no | G1 DESIGN NOTE -- the MI210 window handover between AutoKernel and the stack owner. Not reachable from the declared campaign-#1 roots. |
| `loop/graph_profile_capture.py` | 273 | no | Prospective SC55 capture carrier; never launches a producer or assigns a grade. Not reachable from the declared campaign-#1 roots. |
| `loop/graph_profile_reader.py` | 164 | no | SC55 prospective native decoder. Not reachable from the declared campaign-#1 roots. |
| `loop/guard_bands.py` | 179 | yes | AKX-ALL-29.5: poisoned guard bands around dst and KV views. On the campaign path: first reached via `loop.gates` (d64d3b529). |
| `loop/headline_admissibility.py` | 106 | yes | Statistical admission contract for an AutoKernel performance headline. On the campaign path: first reached via `loop.serving`. |
| `loop/heartbeat.py` | 169 | no | Serialized worker-status heartbeat and terminal lifecycle. Not reachable from the declared campaign-#1 roots. |
| `loop/heldout_serving.py` | 66 | no | Request-distinct serving confirmation for integrity-flagged source candidates. Not reachable from the declared campaign-#1 roots. |
| `loop/hip_launch_proof.py` | 122 | yes | Per-launch proof that a GPU serving launch actually ran on HIP (G6, 2026-10-04). On the campaign path: first reached via `loop.serving`. |
| `loop/hip_routes.py` | 143 | no | Fail-closed HIP matmul route capture and two-run A/A identity check. Not reachable from the declared campaign-#1 roots. |
| `loop/historical_trajectory.py` | 200 | no | Publish the compact, source-attributed AutoKernel champion trajectory. Not reachable from the declared campaign-#1 roots. |
| `loop/history_fields.py` | 204 | yes | AK-HR-3: nullable, additive write-side fields for closed-history replay. On the campaign path: first reached via `loop.actors`. |
| `loop/hotspots.py` | 234 | no | Where the device time actually goes on the tree being edited. Not reachable from the declared campaign-#1 roots. |
| `loop/instruments.py` | 168 | no | Shared scaffolding for the loop's HAND-RUN instruments. Not reachable from the declared campaign-#1 roots. |
| `loop/integrity.py` | 489 | yes | Pre-build integrity gates for planner-authored kernel candidates. On the campaign path: first reached via `loop.actors`. |
| `loop/iqk_gdb_probe.py` | 63 | no | Trusted GDB-side record for a stopped, single-case CPU IQK probe. Not reachable from the declared campaign-#1 roots. |
| `loop/iqk_witness.py` | 338 | yes | Single-case, no-rebuild CPU IQK source engagement witness. On the campaign path: first reached via `loop.cpu_quant_reference`. |
| `loop/journal_feed_owner.py` | 293 | yes | Native durable Journal read/ACK ownership for bounded consumers. On the campaign path: first reached via `loop.evidence_feed`. |
| `loop/kernel_coverage.py` | 748 | yes | Kernel feature-preservation gate: no keep or fold may silently drop a kernel path. On the campaign path: first reached via `loop.serving`. |
| `loop/kernel_mutation_guard.py` | 38 | yes | Shared refusal policy for mutations of frozen production kernel checkouts. On the campaign path: first reached via `loop.archive`. |
| `loop/lane_actors.py` | 135 | no | Per-lane actor models: lane k's planner and author on another model (2026-09-29). Not reachable from the declared campaign-#1 roots. |
| `loop/lane_targets.py` | 388 | no | Lane -> target binding: one lane per model, ONE champion lineage (2026-10-03/04). Not reachable from the declared campaign-#1 roots. |
| `loop/legacy_migration.py` | 633 | no | Bounded, provenance-preserving snapshots of legacy AutoKernel state. Not reachable from the declared campaign-#1 roots. |
| `loop/legacy_targets.py` | 110 | no | Bind an enrolled target to the existing loop's single-target inputs. Not reachable from the declared campaign-#1 roots. |
| `loop/lifecycle_observation.py` | 1,814 | yes | Bounded, observation-only whole-lifecycle serving sampler. On the campaign path: first reached via `loop.cpu_profile`. |
| `loop/lineage_beliefs.py` | 136 | no | Prospective, report-only lineage observation over committed loop outcomes. Not reachable from the declared campaign-#1 roots. |
| `loop/lineage_capture.py` | 108 | no | Default-off prospective source capture for offline lineage diagnostics. Not reachable from the declared campaign-#1 roots. |
| `loop/loadgen_selfcheck.py` | 134 | yes | AKR-SG-1: load-driver self-check (requested vs achieved start rate, client scheduling delay) and the survivorship rule for session-replay headlines. On the campaign path: first reached via `loop.serving`. |
| `loop/longctx.py` | 980 | no | Long-context serving surface with prefill-once (audit C1 + C4, 2026-10-04). Not reachable from the declared campaign-#1 roots. |
| `loop/longctx_tools.py` | 186 | no | Offline tooling for the long-context surface (longctx.py). Not reachable from the declared campaign-#1 roots. |
| `loop/loop.py` | 2,184 | yes | The discovery loop. On the campaign path: first reached via `loop.actors`. |
| `loop/maintenance_execution.py` | 740 | yes | Bounded orchestration for retention maintenance ownership. On the campaign path: first reached via `journal`. |
| `loop/measurement_capture.py` | 801 | yes | Prospective, per-arm evidence capture for :mod:`planned_serving`. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/metrics_map.py` | 69 | no | AKR-SG-2: map llama-server /metrics (Prometheus text, v10 names) into the paired serving-gate report; KV usage is reported as not_exposed, never zero. Not reachable from the declared campaign-#1 roots. |
| `loop/migration_cli.py` | 84 | no | CLI for bounded legacy migration snapshots and rollback inspection. Not reachable from the declared campaign-#1 roots. |
| `loop/model_identity.py` | 394 | yes | Whole-model output identity: the correctness gate for a placement/loader route. On the campaign path: first reached via `loop.gates`. |
| `loop/native_capture_control.py` | 942 | yes | Validate prospective native carriers before the controller journals them. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/native_final_trial.py` | 377 | yes | Parent-final scientific evidence over immutable child-native observations. On the campaign path: first reached via `loop.native_capture_control`. |
| `loop/native_model_preparation.py` | 131 | yes | Closed inputs and live-claim adapter for pre-measurement model verification. On the campaign path: first reached via `loop.cpu_profile`. |
| `loop/native_parent_evidence.py` | 566 | yes | Factual parent evidence for one owned v2 planned-serving unit. On the campaign path: first reached via `loop.native_final_trial`. |
| `loop/native_parent_receipt_replay.py` | 247 | yes | Replay original parent-issued factual evidence; hashes alone do not prove issuance. On the campaign path: first reached via `loop.driver_execution`. |
| `loop/native_parent_service.py` | 269 | yes | Concrete bounded parent service: owned facts in, no scientific policy out. On the campaign path: first reached via `loop.driver_execution`. |
| `loop/native_producer_source.py` | 205 | yes | Prospective loaded identities for the enumerated native v2 producer path. On the campaign path: first reached via `loop.native_capture_control`. |
| `loop/native_retention_catalog.py` | 678 | yes | Controller-owned retained-artifact catalog projection. On the campaign path: first reached via `journal`. |
| `loop/native_scientific_witness.py` | 716 | yes | Parent-issued T0 collection and deterministic owning-reducer replay. On the campaign path: first reached via `loop.native_final_trial`. |
| `loop/native_server_response.py` | 336 | yes | Prospective raw HTTP evidence from the already-owned native serving lifecycle. On the campaign path: first reached via `loop.cpu_profile`. |
| `loop/native_server_t0_witness.py` | 720 | yes | Parent-issued same-server v1 T0 evidence; never CLI-to-server transfer. On the campaign path: first reached via `loop.native_capture_control`. |
| `loop/new_epoch.py` | 161 | no | The supported new-anchor epoch path (DS41-C126 gap a/b). Not reachable from the declared campaign-#1 roots. |
| `loop/node_profile.py` | 563 | no | Out-of-band instrumented node/host/engram profiling of the loop's own anchor. Not reachable from the declared campaign-#1 roots. |
| `loop/noise_relative_tolerance.py` | 193 | yes | AK-PORT-LB1: noise-floor-relative model-level tolerance (candidate within 1.5x the incumbent's own batched-vs-single divergence, no greedy flips above max(0.25, 2x floor)) for non-bit-exact multi-row kernels. On the campaign path: first reached via `loop.gates`. |
| `loop/noop_build.py` | 81 | no | AKX-ALL-29.3: recognise a GPU candidate build that changed no kernel code. Not reachable from the declared campaign-#1 roots. |
| `loop/observation_binding.py` | 884 | yes | Versioned binding between contained planned units and lifecycle observations. On the campaign path: first reached via `loop.actor_lifecycle`. |
| `loop/paired_loadgen.py` | 215 | no | AKR-SG-2: paired A/B/A/B alternation wrapper around a load generator, emitting the existing serving-gate record against a measured noise floor. Not reachable from the declared campaign-#1 roots. |
| `loop/perf_cache.py` | 306 | yes | Content-addressed cache for `perf report` / `perf annotate` subprocess output. On the campaign path: first reached via `loop.cpu_profile`. |
| `loop/pipeline.py` | 514 | no | Run several iterations concurrently, with one serialized tail. Not reachable from the declared campaign-#1 roots. |
| `loop/planned_serving.py` | 839 | yes | Execute a frozen serving ExperimentPlan through the existing launcher consumer. On the campaign path: first reached via `loop.cpu_profile`. |
| `loop/planned_unit_selection.py` | 248 | yes | Immutable planned-unit transport. On the campaign path: first reached via `loop.native_capture_control`. |
| `loop/pool.py` | 725 | no | The concurrent path's real-world wiring: git worktrees, refs, timing. Not reachable from the declared campaign-#1 roots. |
| `loop/process_metrics.py` | 221 | no | Read-only retrospective process metrics for AutoKernel attempt stores. Not reachable from the declared campaign-#1 roots. |
| `loop/procguard.py` | 775 | yes | No AutoKernel process outlives the actor call, gate or run that spawned it (DS41-C84). On the campaign path: first reached via `loop.actors`. |
| `loop/production.py` | 1,914 | no | Publish the champion's gain over the FROZEN PRODUCTION kernel, at every advance. Not reachable from the declared campaign-#1 roots. |
| `loop/production_enrollment.py` | 508 | yes | Offline consumer for canonical production-launcher enrollment exports. On the campaign path: first reached via `loop.unified_planner`. |
| `loop/production_surfaces.py` | 108 | no | AKX-ALL-LB2: the 27B GPU recipe's production-shape verify surfaces (N=8/16/32 and ragged {1,8,8,8}) and the `not-yet-transferable` keep label. Not reachable from the declared campaign-#1 roots. |
| `loop/profile_preparation.py` | 639 | yes | Installed selected-profile execution and original settled-profile feedback. On the campaign path: first reached via `loop.cpu_profile`. |
| `loop/recal_serving_floor.py` | 141 | no | Re-calibrate the SERVING floor under a given recipe (R23-49). Not reachable from the declared campaign-#1 roots. |
| `loop/residency.py` | 612 | yes | Prove the run actually happened on the GPU. On the campaign path: first reached via `loop.bench`. |
| `loop/resolved_recipe.py` | 1,539 | yes | Immutable, offline resolution of one serving launch recipe. On the campaign path: first reached via `loop.actors`. |
| `loop/resume.py` | 3,683 | no | Resume abandoned candidates at the stage they reached, instead of re-deriving them. Not reachable from the declared campaign-#1 roots. |
| `loop/retention.py` | 568 | yes | Immutable retained-artifact closure and dry expiry planning. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/retention_consumer.py` | 461 | yes | Native, held consumer for the pure retention closure. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/roofline_coverage.py` | 394 | no | Per-node CPU roofline + admitted-route coverage: the loop's own autonomy self-check. Not reachable from the declared campaign-#1 roots. |
| `loop/run.py` | 7,667 | no | Drive the discovery loop end to end on real hardware. Not reachable from the declared campaign-#1 roots. |
| `loop/run_paired_serving_ab.py` | 164 | no | AKR-SG shakedown driver: paired A/B serving gate on a real endpoint pair (A/A shakedown allowed), run only with WINDOW_CONFIRMED=1. Not reachable from the declared campaign-#1 roots. |
| `loop/runtime_admission.py` | 871 | no | Existing-loop runtime-only selection, using the original evaluator policy. Not reachable from the declared campaign-#1 roots. |
| `loop/runtime_aggregates.py` | 199 | yes | Bounded diagnostic carrier; no execution, evidence or qualification authority. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/runtime_arms.py` | 709 | no | Declared runtime arms for the existing strict runtime-admission path. Not reachable from the declared campaign-#1 roots. |
| `loop/runtime_calibration.py` | 864 | no | Direct serving anchor/instrument material for the existing calibration solver. Not reachable from the declared campaign-#1 roots. |
| `loop/runtime_identity.py` | 343 | yes | DS41-C97: a runtime treatment's identity is WHAT IT CHANGES, never its name. On the campaign path: first reached via `loop.archive`. |
| `loop/runtime_recovery.py` | 225 | no | Original serial teardown references, not resource grants or admission evidence. Not reachable from the declared campaign-#1 roots. |
| `loop/runtime_window.py` | 388 | no | Original direct-serving window facts, using the installed host/claim predicates. Not reachable from the declared campaign-#1 roots. |
| `loop/scheduling.py` | 2,139 | yes | Deterministic bounded opportunity selection and held-claim accounting. On the campaign path: first reached via `loop.actor_lifecycle`. |
| `loop/scheduling_cli.py` | 164 | no | Offline inspection CLI for bounded scheduler selection and accounting. Not reachable from the declared campaign-#1 roots. |
| `loop/scoped_evidence.py` | 1,947 | yes | Scoped, replayable evidence projection for AutoKernel research. On the campaign path: first reached via `loop.evidence_feed`. |
| `loop/scratch.py` | 1,121 | yes | ONE scratch-resource registry for the AutoKernel loop (operator directive 2026-09-26). On the campaign path: first reached via `loop.actor_context`. |
| `loop/search_window.py` | 751 | yes | Bounded same-attempt window observations; no grant or invented control panel. On the campaign path: first reached via `loop.native_parent_service`. |
| `loop/seed.py` | 93 | no | Seed the loop's information base from what we already knew. Not reachable from the declared campaign-#1 roots. |
| `loop/seed_bundle.py` | 154 | no | R23-51a -- seed the durable accumulator bundle with a MEASURED tip-vs-cor number. Not reachable from the declared campaign-#1 roots. |
| `loop/serial_build_retention.py` | 466 | no | Bounded reclamation of reproducible build caches from retired serial states. Not reachable from the declared campaign-#1 roots. |
| `loop/serial_control.py` | 186 | no | Authenticated commands queued to the existing serial owner, never a second writer. Not reachable from the declared campaign-#1 roots. |
| `loop/serial_roster.py` | 238 | no | Resolve owned launch inputs into the existing serial runner's ordinary argv. Not reachable from the declared campaign-#1 roots. |
| `loop/serial_run.py` | 2,701 | no | Serial finite batches of the existing loop, not another measurement owner. Not reachable from the declared campaign-#1 roots. |
| `loop/serial_scheduling.py` | 716 | no | Closed scheduling inputs for the existing serial loop owner. Not reachable from the declared campaign-#1 roots. |
| `loop/served_shape_cases.py` | 1,191 | yes | Served-shape `test-backend-ops` case set for the ppl_contract layer (a) gate (review 2026-10-06, `gates.ppl_contract_op_nmse` HONEST LIMITATIONS). On the campaign path: first reached via `loop.gates`. |
| `loop/serving.py` | 2,351 | yes | Serving-throughput measurement: a keep is real only if it improves SERVING, not a llama-bench proxy (R23-43, operator directive 2026-09-04: "the only performance that matters is serving performance"). On the campaign path: first reached via `loop.bench`. |
| `loop/serving_beliefs.py` | 216 | yes | Prospective direct-serving observations, using the established belief row vocabulary. On the campaign path: first reached via `loop.archive`. |
| `loop/serving_gate.py` | 212 | no | Run the R23-44 serving gate ONCE on the seeded bundle, BY HAND. Not reachable from the declared campaign-#1 roots. |
| `loop/serving_preparation.py` | 1,061 | yes | Prospective inputs for serving calibration through the existing worker. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/serving_preparation_startup.py` | 158 | yes | Compact prospective preparation inputs; no artifact or resource I/O. On the campaign path: first reached via `loop.serving_preparation`. |
| `loop/serving_server_owner.py` | 79 | yes | Fresh serving child ownership of its local HTTP listening socket. On the campaign path: first reached via `loop.serving`. |
| `loop/source_build_entrypoint.py` | 7 | no | Stable ``python -m`` shim; implementation remains imported under its canonical name. Not reachable from the declared campaign-#1 roots. |
| `loop/source_build_execution.py` | 483 | no | Execute one selected build recipe through the controller-owned worker lifecycle. Not reachable from the declared campaign-#1 roots. |
| `loop/source_build_preparation.py` | 422 | no | Bind actor advice to real source/build inputs; never grant execution authority. Not reachable from the declared campaign-#1 roots. |
| `loop/source_build_worker.py` | 141 | no | Closed child entry point for one previously admitted native build request. Not reachable from the declared campaign-#1 roots. |
| `loop/source_loo.py` | 239 | no | Execute R23-48 omissions on detached full-tip trees under the caller's claims. Not reachable from the declared campaign-#1 roots. |
| `loop/standalone_cli.py` | 427 | no | Bounded client for the existing authenticated campaign control service. Not reachable from the declared campaign-#1 roots. |
| `loop/standalone_inputs.py` | 854 | yes | Closed startup materialization for the standalone v3 runtime owner. On the campaign path: first reached via `loop.serving_preparation`. |
| `loop/standalone_runtime.py` | 737 | yes | Single-owner composition of the unified planner and controller execution path. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/startup_factory.py` | 467 | no | Build closed standalone startup inputs offline from pinned enrollment artifacts. Not reachable from the declared campaign-#1 roots. |
| `loop/status.py` | 392 | yes | The loop's live status, written for the dashboard to read. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/surface_fold.py` | 267 | no | Bounded original source-keep receipts; no fold or promotion authority. Not reachable from the declared campaign-#1 roots. |
| `loop/surface_validation.py` | 489 | no | Exact original serving-gate rows for a propagated whole-source candidate. Not reachable from the declared campaign-#1 roots. |
| `loop/target_profile_execution.py` | 291 | yes | Selected target-profile execution through the campaign-controller worker owner. On the campaign path: first reached via `loop.campaign_control`. |
| `loop/unified_driver.py` | 1,344 | yes | Opt-in standalone consumer for unified planning and durable dispatch intent. On the campaign path: first reached via `loop.actor_lifecycle`. |
| `loop/unified_planner.py` | 1,372 | yes | Pure unified planning and scheduler handoff for enrolled CPU/GPU targets. On the campaign path: first reached via `loop.actors`. |
| `loop/unified_worker.py` | 2,464 | yes | Closed planned-serving child bridge for one already-owned lifecycle worker. On the campaign path: first reached via `loop.driver_execution`. |
| `loop/validation_claim_receipt.py` | 162 | yes | Immutable references to canonical ROOT projection-and-grade results. On the campaign path: first reached via `loop.validation_consumer`. |
| `loop/validation_consumer.py` | 652 | yes | Consume native serving evidence into immutable candidate validation batches. On the campaign path: first reached via `loop.validation_semantic_adapter`. |
| `loop/validation_loo_evidence.py` | 293 | no | Immutable exact references for candidate LOO evidence without store scans. Not reachable from the declared campaign-#1 roots. |
| `loop/validation_objective_decision.py` | 235 | no | Seal the actual AutoKernel readiness result without upgrading its authority. Not reachable from the declared campaign-#1 roots. |
| `loop/validation_projection_source.py` | 251 | yes | Prospective canonical receipt provenance, never live or scientific authority. On the campaign path: first reached via `loop.validation_claim_receipt`. |
| `loop/validation_semantic_adapter.py` | 569 | yes | Pinned bridge to the canonical ROOT ClaimTuple projection and grade. On the campaign path: first reached via `loop.feed_runtime`. |
| `loop/width_dimension.py` | 226 | no | AKX-ALL-27: width invariance as an observe-only keep dimension, compared relative to the anchor's own AKX-ALL-26 table; refuses nothing. Not reachable from the declared campaign-#1 roots. |
| `loop/width_invariance.py` | 399 | no | AKX-ALL-26 width-invariance op probe: driver, first-fail-k table and prediction scoring. Not reachable from the declared campaign-#1 roots. |
| `loop/worker_bootstrap.py` | 282 | yes | Pipe-gated bootstrap for one exact lifecycle worker command. On the campaign path: first reached via `loop.worker_lifecycle`. |
| `loop/worker_lifecycle.py` | 2,471 | yes | Durable consumer for one exactly owned, provider-authorized worker stage. On the campaign path: first reached via `journal`. |

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
