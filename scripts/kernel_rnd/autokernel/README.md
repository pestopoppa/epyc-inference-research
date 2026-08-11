# `autokernel` — a search for faster llama.cpp kernels

**What it is.** A harness that takes one experimental kernel change, builds it in
a throwaway worktree off the *current* frozen production tip, proves it still
computes the right answer, measures it against production in interleaved paired
blocks, and keeps or reverts it. Production is never touched.

**The one command** — from `/mnt/raid0/llm/epyc-inference-research`:

```bash
python3 -m scripts.kernel_rnd.autokernel.campaign --model /path/to/model.gguf
```

**Dry run is the default.** It composes and prints every applicable step, including
the exact argv and env both arms would be spawned with, and executes nothing.
`--execute` additionally requires `--i-hold-the-host`, a validated
`--proposal-manifest`, an exact-unit physical envelope and a current
identity-bound calibration bundle; proposal-v3 is fsynced before any host work.

**Status: no candidate has ever been built, and no benchmark has ever been run
by this package.** Not one. The loop above has been composed end to end and
never executed. Until 2026-08-04 there was no entrypoint at all — `grep -rln
'__main__|argparse|def main('` over every non-test module returned nothing —
and that, not a missing gate and not a missing statistic, is why 100k lines and
6,000 passing tests have produced no results. What still stands between here and
a first number is [§ *The honest list*](execution/README.md#6-before-a-first-campaign-can-start--the-honest-list).

**Essential vs deferred.** Until 2026-08-04 roughly half of this package — the
controller's research strategy, the release plane, the adapters, the dashboard
surface — was provably unreachable from the path between "an idea for a kernel"
and "a measured number". That split was computed by walking the import graph, not
asserted, and the operator acted on it: those planes were removed, about 79,600
lines including their tests, recoverable from the tag
`autokernel-preserve-20260804`. Later the same day the surviving controller
memory was opened *module by module* — see **A question, or an exploration**
below. Three small pure analysis modules are deliberately off the campaign import
path: prior-art classification, screening-lane planning, and offline
least-commitment evaluation. They cannot launch or mutate anything, and become
reachable only when their owning workflow supplies evidence. The campaign-path
boundary still guards every new import. [`FOOTPRINT.md`](FOOTPRINT.md) carries the current totals and every
module's reachability row by row, and `test_campaign_footprint.py` turns the
suite red if the document and the tree disagree. **The figures live there and
only there** — a number stated in two documents drifts in exactly one of them.

### A question, or an exploration

```bash
python3 -m scripts.kernel_rnd.autokernel.campaign --model /path/to/model.gguf \
    --journal-root /some/durable/root \
    --hypothesis akh-fuse-norm-cluster --hypothesis-store /path/to/operator_hypotheses.json
```

With `--hypothesis`, the region claim is acquired through
`controller/hypotheses.py::claim_for_hypothesis`, so a question with **no
falsifier** — or a placeholder one (`"tbd"`) — cannot reach a claim, and the
refusal lands before anything is acquired and before the worktree exists. The
falsifier travels into the claim's own receipt, so the resource record and the
question record say the same thing. Without `--hypothesis` the campaign is
**exploratory** and the record says so in as many words: an unexplained absence
and a declared exploratory run must not read the same afterwards.

That gate had been written, documented as *"The ONLY route from a hypothesis to a
resource claim"*, and never called by anything but its own tests — the driver
acquired the claim directly. Writing your own question:
[`HYPOTHESES.md`](HYPOTHESES.md).

### The two things the harness is built around

1. **T0 before any speed number at all.** Throughput is reward-hackable —
   deleting the computation is the fastest kernel there is. The predecessor
   harness tested `MUL_MAT` only, so a kernel that broke `MUL_MAT_ID` (MoE
   dispatch, on *every* token in production) passed it cleanly. If T0 fails,
   `run_campaign` returns before a single block is planned.
2. **Interleaved paired blocks, because drift was measured.** Four A/A runs of
   *identical* code on a quiet host declined monotonically on decode (52.76 →
   52.31 → 51.62 → 50.52). A candidate-then-anchor A/B therefore charges the
   second arm a systematic ~4%, and *more repetitions do not remove it.* The
   accept rule is `min(delta) > 0` over N pre-committed pairs AND
   `median(relative) > contribution_floor`, with the floor and B_min supplied by
   a current identity-bound calibration bundle. The historical A/A still
   motivates pairing and the separate anchor-movement control; it does not
   authorize v9 ranking.

### Quant-specific roofline targets

`substrate.compare_per_quant()` keeps Q4, Q8 and BF16 as separate cells because
their `bytes_per_token` values are different denominators. Local headroom uses
the MI210's measured-achievable bandwidth; cross-vendor comparison is
datasheet-to-datasheet only. The result is permanently labelled
`diagnostic_and_routing_only` and is not accepted by a promotion gate.

The checked-in CUDA registry currently has one absolute primary-source anchor:
Hazy Research's reported 78% H100 memory-bandwidth utilization for batch-1,
single-sequence BF16 Llama-3.2-1B decode. It is usable only by the exact BF16
regime. Q4_K and Q8_0 return `COULD_NOT_CHECK`: Marlin reports a near-ideal
3.87x *relative* INT4-g128 speedup, not absolute spec-basis bandwidth use, and
INT4-g128 is not GGUF Q4_K. The implementation records that evidence gap rather
than borrowing BF16's target or converting a relative speedup into a roofline.

### Where to read next

| You want | Go to |
|---|---|
| To run a campaign | [`campaign.py`](campaign.py)'s module docstring — the loop order, the accept rule, and the import boundary |
| What blocks a first result | [`execution/README.md` §6](execution/README.md) — the honest list |
| The research programme | [`program.md`](program.md) |
| What is on the path and what is not | [`FOOTPRINT.md`](FOOTPRINT.md) |
| GEAK / AgentKernelArena on MI210 | [`controller/ARENA_INTEGRATION.md`](controller/ARENA_INTEGRATION.md) |

### Representation-bound proposal comparisons (AK-WM-1)

New proposal records use `epyc.autokernel.proposal.v3`. They bind the vocabulary,
source receipts, alternatives, empirical-demand weights, abstraction cost,
canonical encoding, and semantics-preserving recoding fixtures into one derived
`frame_sha256`. Proposal v2 remains readable, but new candidate orderings are
comparable only when `check_representation_comparable()` returns `PASS`.

`offline_least_commitment.py` implements the AP-WM-1 shadow protocol over matched,
completed proposal archives. It reports direction, conditional predictive value,
robust sign error, effective pairs/noise exclusions, and recoding stability. Its
only authority label is `observe_only`; it is intentionally outside the campaign
import path and exposes no selector, champion, T2, or T3 mutation API.

For the CPU known-win diagnostic, a `change_class: "parameter"` proposal must
declare `change.parameter_surface` with exact `candidate` and `anchor` mappings.
The only arm-local key currently licensed by the recipe registry is
`ggml_iqk` (`"0"` or `"1"`). `campaign.py` projects that declaration into the
existing `MicrobenchPlan` arm overrides, and its dry run prints the two distinct
environments; identical arms or any unregistered key are refused before a claim.

---

## Scope and authority

Owning design: [`/workspace/handoffs/active/autokernel-research-loop.md`](/workspace/handoffs/active/autokernel-research-loop.md)
(the `epyc-root` handoff). Section references below (`§5.8`, `invariant 7`, …)
are to that document.

**AutoKernel's release-side job ends at a *release package*.** A human executes
every freeze and cutover (§1.3, invariant 5); nothing in this package carries
freeze or cutover authority, and `schemas.find_authority_flavoured_keys()`
refuses any auth-flavoured key in a machine-authored record so the absence is
*checked*, not merely intended.

Nothing here writes to a production kernel tree. Nothing outside `execution/`
starts a process, runs a benchmark, or runs inference; `execution/`'s five
executors do, under a held claim, and only when the driver is run with
`--execute`. `--execute` is refused at argv time — before any claim — while
`HostOps.unimplemented_seams()` is non-empty. The first IQK parameter campaign
has built-in no-source mutation, measured per-tool anchor, and registered T0
evidence adapters; it requires only the operator's healthy all-core
`--nominal-khz` at its remaining adapter seam after the current live-control
calibration exists. Source-changing campaigns additionally require their own
`apply_candidate` and proposal-specific `t0_evidence` adapters.

---

## What is implemented on the lean campaign path

| Module | What it owns |
|---|---|
| `schemas.py` | The §7 record contracts (`v2`–`v4` remain readable; evaluation-event `v5` adds parsed device state to v4's transfer surface), canonical JSON/content hashing, the `PASS`/`FAIL`/`COULD_NOT_CHECK` `Check` type, and record-level checkers. Proposal v3 requires a structured `external_numbers` list with source revision and independently re-derived same-quant/same-basis roofline utilization. **Single source of truth: every other module is written against these and must not invent a record shape.** |
| `journal.py` | The append-only, fsynced, **sharded** primary record. Shard ordering, torn-tail repair, cursors, archiving, supersession (record-scope and retrieval-scope), tombstones, preflight attestations, derived views, and `check_view_consistency`. |
| `storage.py` | §3.7 durability classes, §5.8 retention classes and rule-bound tombstoned expiry, per-campaign quota and `DISK_PRESSURE`, the `data/<campaign>/` evidence root with `SHA256SUMS` + README, and `verify_durability`. |
| `resource/device_claim.py` | The cross-process **exclusive GPU device claim** (§2.6) — `flock(LOCK_EX)` on a never-unlinked lock file, PID+start-time+boot-id liveness, journaled crash reclamation, quiesce-and-drain revocation, and the claim receipt id that lands in every evaluation event. |
| `resource/preflight.py` | The **one** audited read-only inference preflight (§3.5): claim witness as the target instrument, an opt-in name-pattern enumerator as the labelled interim one, and an AST self-audit proving the module cannot deliver a signal. |
| `resource/claim_witness.py` | The seam between the three above: a conforming `GpuClaimReader` over the device claim, and resolution of an evaluation event's opaque `resource_claim_receipt` back to the claim that produced it. |
| `evaluator/api.py` | **AK3.** The typed evaluator interface and the only place a verdict is COMPUTED. Owns the protocol's eight preconditions, twelve void conditions, fourteen search-grade conjuncts, the record grammar, the tier state machine, and the local-A/A-derived `MinimumMeasurableDuration`: a T1a cell below that floor is inconclusive and has no speed rank. |
| `evaluator/integrity.py` | **AK3.** §8.5.1 source integrity: an ELF `.dynsym`/`.symtab` reader, an Itanium-ABI demangler, symbol/arity/registration/dispatch-predicate diffs, clean-build-from-snapshot provenance, unified-diff parsing against the change-class envelope, the mechanically derived `core_header` risk tier, the §10.6 complexity ceiling, repair-from-clean-parent, and the gates that bind all of it to the request's anchor, artifact and derived surface. |
| `evaluator/surface.py` | **AK3.** §6.4 affected-surface derivation from the build system's OWN dependency information (make/ninja depfiles, cmake link lines), the dispatch-trace stage, three-stage reconciliation (`derived ⊇ traced`), the actor's declaration as a SCORED prediction rather than a scope input, and §3.2 normalized-binary comparison. |
| `evaluator/correctness.py` | **AK3.** The seventeen T0 gates: backend-op units, exact-reference comparison, unseen boundary shapes, surface reconciliation, the no-fallback proof, state/rollback/teardown, sanitizers (ASAN/UBSAN, no core dumps), output coherence vs the anchor, determinism class, binary+linkage identity, and anti-reward-hacking. `CoherenceVerdict` is computed and cannot be stamped. |
| `evaluator/statistics.py` | **AK3.** The calibration block in its normative solve order — `φ`, `B_min`, the α budgets and their thresholds, the anchor-gate band — the anytime-valid e-process (two bundle-fixed constructions), the pre-committed stopping rule with bounded extension, the MDE, order control, the selection/confirmation split, and the reducer that produces a conforming `api.EffectEstimate`. |
| `evaluator/controls.py` | **AK3.** The five controls as hashed data (definitions AND predicates), the A/A cadence scheduler, the historical-win-replay declared contract with its normative unavailable branch and operator escalation, and the projection into `api.WindowAttestations`. |
| `evaluator/rebench_scoring.py` | **AK-RB-1.** Reference-normalized log-time scoring for behavior-preserving optimization: 0 is the measured starting state, 1 is the measured strong reference, incorrect/unverified work receives no score, and best-so-far curves are emitted at matched 2h/8h/32h budgets. The score is deliberately unclipped so regressions and reference-beating candidates remain visible. |
| `evaluator/oracle_integrity.py` | **RVP-C2-8/C2-9.** Fail-closed reducers for same-shape hostile input distributions and checker isolation. They bind suite seed/version, require four distinct materialized populations, and refuse property/reference evidence whose checker translation unit reaches device code, forced MMQ, CUDA flash attention, or rocWMMA paths. The live runner refuses dirty or frozen source identities and retains one device claim plus numeric sampling across both probes. |
| `evaluator/historical_tasks.py` | **RVP-C5-R/C3-2.** Validates sealed local historical-task descriptors that expose only the pre-optimization parent to the actor, preserve whether the original argv was recovered, and pin the model, full benchmark surface, metric direction, repeats, parent, and human expert. The expert-ceiling reducer reports candidate-to-parent and candidate-to-human deltas only when a terminal candidate exists; an archive with no candidate is `COULD_NOT_CHECK`, never a stamped win. |
| `evaluator/recipes.py` | **AK3.** The codified recipe constructors — every measurement argv is emitted by one, carrying its constructor id and content hash — for `test-backend-ops`, `test-quantize-perf` and `llama-bench`, CPU and GPU. Every T1a recipe requires a typed local A/A timing floor; a bare numeric or foreign floor is refused. |
| `execution/worktree.py` | **AK2.** Fresh worktrees from the reviewed one-commit measurement overlay on current production v9, pathspec commits, clean candidate-local builds, and build identities. Candidate-controlled CMake configure/build is C6-sandboxed and returns evaluator-owned activation plus verified-cgroup-teardown receipts. |
| `execution/sandbox.py` | **C6.** Native Landlock write confinement, seccomp signal/network/namespace denial, non-root identity, finite rlimits, per-invocation cgroup-v2 containment, candidate-proof activation receipts, and descendant-draining teardown. Startup is fail-closed. |
| `execution/t0_provider.py` | **AK3 T0.** Real correctness invocations. The live campaign constructs it only with a C6 policy; recorded replay starts no process. |
| `execution/microbench.py` | **AK3 T1 / RVP-C6-2/C6-3/C6-10.** Fresh process and empty invocation-only writable state for every arm, interleaved paired blocks, mandatory `--autokernel-harden` receipts, per-CPU frequency observations, and exact-window package-energy deltas. Each repetition uses unique content and context/input addresses in an ordinary/full-device-sync hybrid pair; only the synchronized twin is ranked, both brackets must preserve the exact thread set, and a >20% GPU median divergence is an integrity failure rather than a corrected speed result. Anti-short-circuit units must change the real recipe, appear beside a normal control, and receive blocks in the same ranked stream. Package values are explicitly shared-package rather than lane-exclusive. |
| `execution/instrument_integrity.py` | **RVP-C6-1.** Re-hashes the complete three-file reward-instrument manifest in explicit candidate and named-anchor source roots before live T0/T1 work and immediately before every T1 invocation; missing, unreadable, or changed source is a hard refusal. Build roots are never accepted as source identity. |
| `execution/physical_bounds.py` | **RVP-C6-4.** Immutable per-shape work lower bounds plus hardware peak upper bounds. The time floor is `max(compute, memory)` and the equivalent throughput ceiling is `min(compute, memory)`; every sample above it is refused before ranking. The envelope binds the registered delivered unit and an exact recipe/model/parameter-frame digest; serialized derived values are re-derived rather than trusted. |
| `execution/reward_hack_scan.py` | **RVP-C6-6/C6-9 plus static parts of C6-2/C6-3.** Versioned added-line detectors over the committed candidate diff for protected-frame edits, pointer memoization, structured-input/known-shape short circuits, environment/timing probes, stream creation, and thread/async creation. Missing detector ids make empty findings `COULD_NOT_CHECK`. The checked-in broad corpus states 10/10 planted sensitivity, 15/15 clean specificity, and 0/15 FPR for this named source taxonomy—never arbitrary-program coverage. Runtime stream synchronization and thread-set assertions live in `microbench.py`. |
| `execution/live_controls.py` | **AK3 live calibration.** Dry-run-default CPU producer that predeclares a fresh v9/hardened campaign and an exact RVP-C6-4 envelope for every control leg, copies the reviewed measurement binary and its build-local DSOs byte-for-byte into the evidence bundle, acquires q0–q3, measures fresh A/A/neutral/control legs, solves calibration, and sends all five controls through the candidate dispatcher. The prior v8 result remains historical evidence in `data/autokernel_controls_3pct_20260805/`; it is not silently reused as v9 calibration. |
| `controller/hypotheses.py` | The operator hypothesis store and falsifier-before-claim gate used by `campaign.py --hypothesis`. |
| `controller/do_not_repeat.py` | The receipt-bearing negative/constraint ledger required before a hypothesis can spend a claim. |
| `controller/shared.py` | The small shared vocabulary needed by the two surviving controller-memory modules. |
| `controller/authoring_contract.py` | **AK-PL-1 / AK-LE-4 / AK-LE-5 (off campaign path).** The only reviewed fully-rendered authoring-prompt seam; sealed-evaluator leak scan with compliant control, priced never-bulk-read context, reversible compaction headers with exact git recovery, and typed external numeric priors. It calls no model and selects no work. |
| `controller/reward_monitor.py` | **C6 sabotage/sandbagging monitor contract (off campaign path).** Compiles the named threat model with monitor-awareness and reasoning visibility mandatory, binds every reasoning trace to its campaign/candidate plus the whole journal tree, reduces a declaration-bound complete model×sample panel as mean-of-mean@k, and reports sensitivity, specificity, and false-positive rate against a predeclared budget. It calls no model and cannot stamp correctness. |
| `prior_art.py` | The deterministic four-way prior-art gate, explicit any/all trace-keyword matching, expected-absence override, v9-pinned source catalogue, cumulative wall-share pruning, and hash-bound offline `rocprofv2` reducer used by AK-DEL-1. |
| `profile_report.py` | **RVP-1–7 offline C4.** Consumes paired hash-bound mapping/formal `rocprofv2` captures, enforces stage separation and the 10-warm-up/5-active window, renders kernel/overlap/fuse plus architecture-shape tables above the 1% family floor, records the complete profiler candidate registry including RPD/gfx90a reachability, and makes host-catalogue exclusions explicit. Source matching is exact and reviewed; the optional judgment pass can attach only `low`/`medium`/`high` similarity plus a catalogue note. It launches nothing. |
| `../../benchmark/run_autokernel_omniperf_fallback.py` | **INF-37 profiler fallback.** Captures seeded, repeated fixed-shape ROCm op families through Omniperf 2.0.1 plus `rocprof` v1 when `rocprofv2` crashes. It requires a clean exact source commit, hashes the binary/profiler/Python environment, holds the MI210 claim across correctness preflight and SQ/TCC collection, samples device state, and emits diagnostic-only receipts with device timestamps and counter totals. |
| `../../benchmark/run_autokernel_rocprofv1_attribution.py` | **K28 whole-model fallback.** Captures timestamp-only `rocprof` v1 attribution at predeclared prompt lengths when `rocprofv2` cannot trace the model. It binds a clean source, binary, model, profiler, and exact build-local linkage; holds and samples the MI210; requires the real GDN kernel; and emits diagnostic-only active-kernel shares rather than a throughput verdict. |
| `hipkittens_lds.py` | **INF-03 offline gfx90a adapter.** Solves LDS bank count from the complete four-bank overlap pattern, validates the all-pairs phase relation, reads hash-bound rocprofv2/v3 counter CSVs, and projects a diagnostic-only authoring-context item. It neither launches a profiler nor assumes the CDNA3 answer transfers. |
| `c5_seed_corpus.py` | **C5 static seed registry.** Pins the eight intake-884 HyRA SOL-ExecBench artifacts and their NVIDIA/Hopper-only attestations, separates direct Triton references from CUDA-bound re-authoring targets, and emits non-numeric gfx90a task context. No upstream latency or SOL score is admitted as an MI210 target. |
| `substrate.py` | Validates the checked-in MI210 compute/bandwidth/PCIe/NUMA facts, preserves measured and datasheet bases separately, re-derives both roofline ridges, and builds exact-quant diagnostic surfaces. Cross-vendor cells are spec-basis; a missing exact CUDA anchor is `COULD_NOT_CHECK`, never a pooled or borrowed target. |
| `lanes.py` | The lane registry, historical 4/8/16/32/48-way CPU shapes, isolation checks, change-class-specific rank calibration, and full-instance verification rule. |
| `artifact_diff.py` | The compile-only VGPR/SGPR/scratch/instruction-mix comparison that vetoes an unconfirmed GPU claim before behavioral T0 can launch. |
| `offline_least_commitment.py` | The observe-only AK-WM-2/AP-WM-1 diagnostic over matched completed-proposal archives; it has no live selection authority. |
| `turn_productivity.py` | **AK-PT-1 / AK-X-6.** Immutable per-refine-turn `(turn, task, correct?, speedup)` records, mechanically derived rescued/persistent classes, and a campaign-calibration-derived e-process rule. It may label a turn repair-only and withhold search advancement; it has no ranking, retention, promotion, or deployment authority. |
| `dashboard.py` | **AK6.** The compact `/kernel` contract-v2 producer retained by the campaign path after the old `surface/` plane was deleted. It projects only the already-fsynced terminal `STOP_STATE`: campaign and backend standing are observed; champion, headroom and release package are explicitly `not_reported`; journal time drives freshness; and the atomic export is refused under scratch, a production tree, or any checkout. |

AK-DEL-1 is replayable from
`data/autokernel/prior_art/ak-del-1-k25-q8-mmvq-n1-20260717/`. The checked-in
raw MI210 profile is content-hashed before parsing; start/end timestamps are
aggregated by stable kernel family, the 1% floor is applied, and both count and
duration bucket splits are emitted. The bounded K25 corpus resolves 3/3
admitted families to existing paths and therefore recommends catalogue work
before a novel-kernel proposal generator. That is a scope decision over this
corpus, not a claim about unprofiled workloads.

The C4 profiler report is a separate offline consumer of completed traces. Its
mapping trace must use graphs-disabled or lower-fusion attribution; its formal
trace must use production optimizations, and both must bind the same stage and
source commit. The deterministic pass owns every measured value and source
locator. A later model sees only emitted pattern ids and may add one bounded
similarity label plus a catalogue comparison; it cannot rewrite timing,
attribution, scope, or architecture-shape facts.

### gfx90a LDS topology (HipKittens method, not framework)

`scripts/benchmark/run_autokernel_lds_solver.py` adapts HipKittens's
`ds_read_b128` bank/phase experiment to the installed MI210. It builds a
self-contained HIP probe for exactly `gfx90a`, holds the normal AutoKernel device
claim, samples device state, and captures `SQ_INSTS_LDS` plus
`SQ_LDS_BANK_CONFLICT` with the side-loaded ROCm 6.2 profiler. The probe contains
no HipKittens headers or Torch dependency.

The bank sweep starts with the method's lanes 0/1 and fits every measured offset
against the exact repeating starting-bank alias pattern for 16/32/64/128-bank
candidates. A unique exact fit both solves the stride and empirically establishes
that the pair can conflict. The runner then uses distinct addresses separated by
that stride to measure all 2,016 lane pairs; identical addresses are deliberately
not used because gfx90a coalesces/broadcasts them. The same-phase relation must
form cliques, and the initial bank pair must validate inside one clique. Repeated
samples use a median conflict predicate.

Run from a clean, committed research checkout and place evidence outside scratch:

```bash
python3 scripts/benchmark/run_autokernel_lds_solver.py \
  --campaign-id inf03-lds-gfx90a-YYYYMMDD \
  --source-commit "$(git rev-parse HEAD)" \
  --output-dir /mnt/raid0/llm/autokernel/probes/inf03-lds-gfx90a-YYYYMMDD
```

The passing receipt is `diagnostic_only`. A `topology_matches_cdna3` result says
the CDNA3 swizzle constants are a plausible authoring prior; it does not establish
their correctness or performance in a llama.cpp kernel. Controllers consume the
receipt only through `hipkittens_lds.topology_context_item(path,
expected_sha256=...)`, which preserves the evidence hash and exposes no verdict or
ranking authority.

### C5 HyRA seed corpus

`c5_seed_corpus.json` pins Hyra-results commit
`26ebfbe7d491e6521d8bb5fc21fe88bb31460825` and exact artifact hashes for
k138, k145, k154, k175, k215, k225, k227, and k228. The handoff-designated
direct set is k138, k145, k154, k175, and k228; k215, k225, and k227
are explicitly CUDA-bound behavior/target references. A direct reference can
still contain a supplemental NVIDIA path, so the registry records observed
cuBLAS, FlashInfer, CuPy, WMMA, and `mma.sync` bindings instead of implying
source-level portability.

Every row has the same terminal intake disposition: re-author for MI210/gfx90a
with wavefront-64, MFMA, and LDS assumptions made explicit, then re-attest from
correctness through timing on the physical card. The upstream completed/correct
results remain NVIDIA/Hopper-only provenance. `seed_context_item()` deliberately
omits their SOL scores and latencies from authoring context; an arena task may
select exact rows through `ArenaTask.c5_seed_ids`.

### AK6 — the operator surface

`dashboard.py` is the PRODUCER half. `HostOps.journal()` calls it only after the
terminal record is durable; an export failure warns but cannot relabel that record as lost. Its consumer is the handoff
hub in **epyc-root** (`dashboard/panels.py`, `dashboard/server.py`,
`dashboard/static/kernel.html`), which owns the panel→producer registry, the
per-panel freshness envelope, the transport watchdog and the `/api/health` fold.
The hub never imports this package. It also reports committed implementation
history and durable A/A/work bundles under `_activity`, but that context never
enters `_freshness` or health. The halves are bound by two suites:

| Suite | What only it can fail on |
|---|---|
| `test_dashboard.py` | the producer: old evidence cannot be re-dated, unowned planes are explicit, and the writer cannot reach scratch, a checkout, or production |
| `epyc-root/tests/test_dashboard_panels.py` | the consumer and seam: absent ≠ empty, activity cannot resurrect runtime health, the registry is total, and restart chaos never goes green over a dead producer |

The seam suite exists because both halves were green while disagreeing: the hub
overwrote `contract_version` (a producer-owned key, integer `2`) with the string
`"v2"`, so what `/api/kernel` served no longer validated under
`schemas.validate_kernel_dashboard_v2`; and `static/kernel.html` drew only a v1
run log, so a **fully reported** v2 contract rendered as an empty page under the
sentence *"the kernel-R&D loop has not exported any results"*. Both are fixed in
epyc-root (hub-derived fields are now `_contract_version` / `_freshness` /
`_render`, and the empty-state sentence is derived on the wire from the document
that was read). The two suites fail if either behavior returns.

`RestartChaosTest` reproduces the incident this surface exists for: a producer
alive and reporting → the producer dies → time passes → the board goes from green
to naming it, and stays naming it **across a hub restart**. Death is simulated by
not exporting and by injecting the clock; **no process is started, signalled or
killed**, and the suite audits its own source to keep it that way.

### AK3 — what the evaluator delivers

It replaces `scripts/kernel_rnd/kernel_eval.sh`, which is fenced off and exits 2.
That script's three defects are the three structural properties here:

1. **It stamped a literal** (`"status":"OK"` in a `printf`). A `Verdict` is
   constructible only by `compute_verdict()`, and `Verdict.__post_init__`
   re-derives `status`, `effect_resolution`, `speed_rank_admissible`,
   `integrity_flags` and `derivation` from the evidence on the same object,
   raising `VerdictTampering` on any disagreement.
2. **It reported coherence with no anchor** (`COH="coherent"` for any non-empty
   generation). An absent, incomplete or mutated anchor is a `VoidFinding`, the
   verdict is `INVALID`, and every gate declaring `requires_anchor=True` has its
   PASS demoted to `COULD_NOT_CHECK` before the status is derived.
3. **It let speed be reported beside a correctness problem.** `rank_key()` raises
   `SpeedRankUnavailable` rather than returning a penalised rank, and
   `rank_candidates()` returns `(ranked, unrankable)` so a withheld rank is a
   journaled reason, never a shorter list.

Governing instrument:
[`/workspace/measurement/protocols/kernel-research.md`](/workspace/measurement/protocols/kernel-research.md)
(Annex K, **P-AK-SEARCH-1**, RATIFIED 2026-08-03). It emits a **search record,
not a claim**, and T3/T4 are refused by name — they are release instruments
outside the protocol's scope, owned by AK5 through the `ReleaseTierEvaluator`
seam.

`evaluator/test_conformance.py` walks that protocol's own sections and asserts
one test per obligation. Its coverage test fails if a registered obligation has
no claiming test, if a test claims an unregistered obligation, or if a claiming
test asserts nothing — so deleting a test is not a way to make it pass.
Obligations whose subject is a real measurement are registered in `SEAM_ONLY`
with the reason they cannot be asserted directly: a **declared** state, never a
silent skip.

### AK4 — what the controller delivers

AK1–AK3 built the substrate; **AK4 is the half that WALKS the loop.** It is also
the half where the project's two most expensive AutoPilot scars live, and each is
answered structurally rather than by a check somebody remembers to call:

- **A control that was requested but never verified.** AutoPilot's pause was a
  silent no-op for months because the run state was cached in memory and written
  back over the operator's change. Here the latch is a FILE, re-read from disk at
  the top of every iteration under the journal write lock; no object holds a
  `ControlLatch` as attribute state; `ControllerStateMachine.__slots__` has no
  slot for one, so a future edit that tries to cache it fails at runtime; and
  `audit_no_cached_control_state()` proves it from the object rather than from a
  comment. A halt survives restart, and an ack without its latch — the
  crash-between-ack-and-latch window — is a HARD failure rather than "no control
  pending".
- **A restart that came up empty with nothing objecting.** 232 trials and ~16
  days of compute vanished because a rebuilt derived view disagreed with the
  record and nothing refused to start. §8.2 step 10 refuses, and the deliberate
  rebase is an explicit escape that lands its reason in the journal, so an
  intentional wipe is never indistinguishable from the loss.

The authority rule the whole plane rests on: **the LLM proposes and interprets; a
deterministic controller disposes every gate and stop condition.** `check_stop_
evidence` takes no `origin` parameter and `check_do_not_repeat` takes no author,
so there is no argument along which trust could be extended; a critic disposition
can only make a proposal's fate WORSE; and an operator's stop request meets the
same evidence table as an internally generated one. `test_ak4_conformance.py`
asserts this per obligation — the five things P-AK-SEARCH-1 authorizes, the nine
it denies, its eight preconditions, and §4's twenty invariants, one test each.
Obligations that cannot be checked without taking a measurement are registered in
that file's `SEAM_ONLY` table with the reason and the owner, never skipped: a
skipped test is silent, a registered seam is a list an auditor can count.

### Guarantees the package as a whole makes

1. **Nothing is lost by crashing.** Every append is fsynced before it returns; a
   torn tail is discarded loudly with its own `TORN_APPEND_DISCARDED` event;
   `read_all()` raises on any unreadable line rather than returning a partial
   history that looks exactly like a complete one. `test_integration` rebuilds
   the whole campaign from the journal alone and asserts the digest matches.
2. **Inability to evaluate is a THIRD outcome.** Every checker returns
   `PASS` / `FAIL` / `COULD_NOT_CHECK`. `PreflightResult.__bool__` raises, so
   `if preflight(...)` is a stack trace rather than a silent misreading.
3. **Evidence is never evicted; only the `expirable` class expires**, and only
   with a tombstone in the journal *before* the bytes go.
4. **A live claim holder is never stolen from.** `unknown` liveness raises and
   journals a defect; revocation is quiesce-and-drain, never a signal.
5. **The record and the retrieval view are different objects.** `read_all()`
   keeps a withdrawn belief and its prose forever; `retrieve()` withholds it, and
   citing it back in raises (§5.5 items 6/7, invariant 20).
6. **Explicit failure over silent fallback, everywhere.** There is no no-op
   journal, no default sink, no fail-open reader; a missing dependency or an
   unreadable file raises.

---

### AK5/AK6/AK8/AK9 — what the release plane delivers

**The cardinal rule: AutoKernel never freezes and never cuts over.** It produces a
release PACKAGE that a human executes. A kernel freeze crosses four human-only
trust boundaries (`MEASUREMENT.md:140-142` — the freeze, the era-registry rows,
the AutoPilot baseline apply, and the pinned human-only path list), so there is no
such authority here to hold, delegate, or flag. Every one of the twelve §11.2 "may
not" capabilities is a function whose entire body is a `raise`, and
`packager.audit_refusal_doors_raise_unconditionally()` walks the list and FAILs on
a door that grew an `if`.

That rule is *proved*, not documented, in four independent ways:

- each module's own AST self-audit, anchored to its own module identity so the
  guarantee is not obtainable by handing the auditor a different string;
- `release/test_release_integration.py::TestNoProductionWritePathsAnywhere`,
  which parses **every** module under `release/` and `adapters/` — enumerated from
  the filesystem, so a module added tomorrow is covered by default — and proves
  no write/spawn/signal call exists in any non-test module and that no call
  anywhere names a production branch, a stable kernel path, `instrument_eras.yaml`
  or `autopilot_baseline.yaml`. It parses calls rather than grepping text,
  because both directories name all four targets in nearly every docstring;
- `t3.TransactionPlan` refuses construction with `executed=True`, and the §10.2
  phase-8 transaction is a DRY RUN by type;
- `schemas.find_authority_flavoured_keys()` over the sealed bundle and over the
  package's own record, enforced at construction.

The release path is not the same for every backend, and that asymmetry is
load-bearing (AK-D9/AK-D23): `llama_cpu`/`llama_gpu`/`whisper_stt`/`qwentts_tts`
release through a kernel freeze; `serving_runtime` travels the §11.6 three-gate
stack-change path and is refused **by name at all four release-plane doors** —
`plan.ReleaseTarget`, `t3.ReleasePlanView`, `packager.OperatorFreezeRequest` and
`readiness.ObjectiveSpec`.

§10.4's calibration is wired and passing, and was **re-run against the real
artifacts after the 2026-08-03 hardening pass** to prove the hardening did not
break the compiler. All three outcomes still hold:

| input | verdict |
|---|---|
| no waiver | **FAIL**, naming `qwen36_q8-tg128-iqk1` and `qwen36_q8-pp2048-iqk1` |
| waiver, no N-1 archive | **FAIL** — no archived incumbent, so no rollback target |
| waiver + reconstructed N-1 archive | **PASS_WITH_WAIVER**, exactly those two claims suppressed by name, forfeit recorded |

Two things about the calibration changed in the truth-up, both because the old
form could pass without inspecting anything:

- **Both documents are now read off disk.** The ratification always was; the
  *waiver* came from `test_t3.V8_WAIVER`, so the claim "against the record rather
  than a fixture of it" was true of the ratification and false of the half that
  decides PASS_WITH_WAIVER. `waive_q8_cpu_prefill_v8_20260725.json` is now read
  too, and its sha256 is checked against `evidence_sha256.waive_q8` **on the
  ratification** — the attestation hashes its own waiver. That is a CONSISTENCY
  fact about two operator-owned documents, and it is worth having; it is not
  authenticity, and this README used to claim it was. The ratification is itself an
  unsigned JSON file in the same directory, so anyone who can rewrite the waiver can
  rewrite the digest that pins it. See *Closed by the 2026-08-03 hardening pass* for
  what reading the file does and does not buy.
- **Absence is a FAILURE, never a skip.** `TestPreservedV8Calibration` used to call
  `self.skipTest()` when the artifact was missing. Removing the file turned the one
  check that says the compiler is right into a silent `ran=0, failures=0` — a
  guarantee obtainable by deleting what it inspects.

## Integration seams reconciled — AK5/AK6/AK8/AK9 release plane (2026-08-03)

Same pattern as AK3 and AK4, and the same lesson: `plan.py`, `readiness.py`,
`t3.py`, `packager.py` and the three adapters were each written and red-teamed
alone, each suite was green, and seven disagreements lived in the gaps. Every one
of them passed both of the involved modules' own suites first, because each module
was right *about itself*. `release/test_release_integration.py` is where they stay
fixed.

1. **A §3.2 drop verdict `plan.py` refuses and `t3.py` accepted.**
   `plan.drop_verdict_contradictions()` re-derives `may_drop_cells` from stage 1,
   stage 2, agreement, transfer scope and findings — because the boolean is a
   plain FIELD and it deletes a backend's whole matrix. `t3.UnchangedView` READ
   it, and guarded only `unchanged_outcome`. A view with `agreement=FAIL` and no
   stage 2 was accepted and dropped the backend. `unchanged_view()` accepts a
   hand-built view and a bare mapping by design, so the compiler is not the only
   door; the same conditions now hold at both, as `UnchangedView.drop_contradictions()`.
2. **§1.6's conjunction was adjudicated over standings nothing joined to the
   matrix.** `phase_performance_matrix` measures cells; `phase_capacity_utility`
   decides the whole per-phase objective from caller-supplied `PhaseStanding`
   records. Nothing connected them: a standing with `cell_ids=()`, or citing a
   correctness cell, or a diagnostic cell, or a cell of another phase, satisfied
   it alone — and deleting every prefill CELL while keeping the prefill STANDING
   left the release gate at **PASS**. A standing must now name at least one
   planned, gating, production-optimal `performance_matrix` cell of its own
   (backend, phase) that this run recorded a result for.
3. **A conjunct could be deleted rather than failed.** `readiness.py`'s red-team
   closed exactly this on the advisory signal (an objective naming only `decode`
   reached `objective_met`); the release gate had the identical hole in
   `phase_protocols`, where it decides a freeze. Both now read
   `schemas.PHASES_BY_BACKEND`, the SSOT `plan.BackendBinding` already held itself
   to.
4. **`serving_runtime` was refused at three doors and admitted at the fourth.**
   It is in `schemas.BACKENDS` but absent from both `SOURCE_TREE_BY_BACKEND` and
   `PHASES_BY_BACKEND`, so in `readiness.py` the champion-lineage check and the
   §1.6 conjunction check each degraded to a **no-op** on it while the signal
   still rendered as a kernel backend's — and the readiness line is what a freeze
   request cites.
5. **A FAILing run licensed claims.** `compute_verdict` built
   `ReleaseReceipt.claims` from every passing gating cell regardless of verdict,
   and AK6 renders `len(receipt.claims)` onto the operator's first page as *"claims
   licensed: N"* and copies the receipt into the durable package record. A release
   that did not pass licenses nothing; the cells that did pass are kept, in
   `withheld_claims`.
6. **A `CellResult` could relabel the plan's cell.** `T3Request` cross-checked
   only that the result's `cell_id` was in the plan, so every other facet was the
   measured party's to restate. Flipping `co_resident` True satisfies the §10.2
   phase 4 `llama_cpu` co-residency requirement with a run that was never
   co-resident; flipping it False deletes the only cell class that measures the
   machine the way production runs it. §12 derives scope MECHANICALLY, so the
   scope facets must now agree. `reps` is deliberately exempt — the compiler runs
   before anything is measured.
7. **The machine-actor vocabulary and the waiver verifier were in different
   modules.** AK6 guards five human-authority identity fields with
   `MACHINE_ACTOR_TOKENS`; `t3.verify_waiver` accepted any non-empty
   `authorized_by`, so a waiver attributed to `autokernel` verified as
   human-attested and turned FAIL into PASS_WITH_WAIVER. The import direction is
   packager → t3, so the check lands in the packager rather than forking the token
   set downward — a self-granted waiver cannot reach a package a human executes.

Two more, found in the same pass and outside the release plane proper:

8. **`release/test_plan.py` and `test_readiness.py` loaded a SECOND copy of
   `schemas` and `surface`.** The flat `sys.path.insert` + `from autokernel import …`
   idiom binds to a different module object than the rest of the package uses
   under `unittest discover -t .`, so every `isinstance` guard across that boundary
   fails silently — `compile_release_plan` would refuse a genuine
   `surface.BackendUnchangedResult` for being the other copy's. The README already
   forbade the idiom; both release-plane suites now use relative imports.
9. **`adapters/serving_runtime.py`'s self-audit was the one of three not anchored
   to its own module.** Its two siblings bind the audited AST to their own
   `BACKEND` and checker names; this one returned PASS for any clean text,
   including a sibling adapter's source.

One asymmetry was **pinned rather than "fixed"**: `t3.Cell` admits a `None`
`workload_phase` while `readiness.T2Cell` requires one, because correctness,
quality, stability and capacity cells are not throughput cells and have no phase
to be non-inferior in. The performance-matrix cells that DO carry one are checked
against `PHASES_BY_BACKEND` at the compiler and at the objective.

## Integration seams reconciled — AK4 controller (2026-08-03)

Six AK4 modules were built in parallel against one state machine. Each was green
on its own; a suite of individually-green modules is precisely the shape in which
a seam defect survives, because every module is consistent with itself and the
disagreement lives in the gap. The integration pass found six, and
`controller/test_loop_integration.py` is where they stay fixed.

1. **Two `proposal_fingerprint` implementations writing one journal field.** The
   planner adapter hashed `change.conceptual_change` — free prose — and the
   screener did not. Both wrote `PROPOSAL_SKIPPED.payload["fingerprint"]`, and
   `read_skip_history()` counts them in ONE dict against a threshold of two, so
   two skips of one concept counted 1 + 1: §8.4's auto-blacklist never fired and
   §8.10's degradation run was computed over a key the record did not use. Now
   one algorithm in `controller/fingerprint.py`, and it is the PROSE-FREE one —
   the planner-side test that asserted rewording minted a new fingerprint was
   asserting attempt 119 looking novel, and has been inverted.
2. **Two §6.5 oracle registries sharing ONE id out of nineteen.** The compiler
   rendered `upstream llama.cpp / ggml` into the planner brief; the critic gated
   on `llama.cpp_upstream` and rejected the citation as *"not in the declared
   registry"* — a refusal that blamed the planner for the controller's own
   disagreement. `controller/oracles.py` now holds the table once, at both
   granularities (the §6.5 row and the tree a port names), and both consumers
   derive.
3. **Two harvest-class vocabularies.** The critic had three classes and the
   compiler four, so §6.5's own FlashAttention/FlashInfer row (`conditional`)
   was inexpressible in the plane that gates it.
4. **Two hypothesis-origin vocabularies.** `hypotheses` opens at `controller` and
   `import`; `context` accepted neither, and offered a `record` origin the store
   cannot produce. A controller-opened hypothesis raised `ContextInputError` on
   its way into the very brief §8.4.0 requires it to appear in.
5. **A reserved closure word the DISPOSER did not know.** `guards` refused
   "exhausted"; `state_machine.check_stop_evidence` did not — and `stop()` and
   `dispose_stop_request()` are both public, so a stop that never met a guard
   reached the record on the word §8.10 names first. The machine now owns the
   vocabulary AND the matcher, `guards` compiles nothing of its own, and the scan
   covers the enumeration as well as the reason.
6. **Three budget units and no converter.** §7.1's manifest declares HOURS, a
   §7.2 proposal declares MINUTES, and `context.reduce_budget_ledger()`
   accumulates SECONDS. The obvious wiring makes the budget gate 60x too
   permissive, in the direction that overspends.
   `selection.budget_remaining_from_caps()` is now the only sanctioned crossing,
   and a missing cap raises rather than defaulting.

One apparent disagreement was **deliberate and has been pinned rather than
"fixed"**: `SUPERSEDED_FACT` rejects in `selection` and is advisory in
`hypotheses`, because §19.2 says *"do not execute the stale PROPOSAL; regenerate
from current source"* — it closes the proposal, not the question. Making the two
equal would close research the design keeps open, so a test asserts the asymmetry
and says why.

## Integration seams reconciled — AK3 evaluator (2026-08-03)

The seven evaluator modules were written in parallel against one interface. Each
passed its own suite and its own red-team pass; every defect below lived
*between* two of them, where each module was individually correct and the two
descriptions of the same object did not match.
`evaluator/test_integration.py` is the regression barrier for these; each row
was verified non-vacuous by reverting the fix in an isolated copy and confirming
the suite fails.

| Seam | Disagreement | Resolution |
|---|---|---|
| statistics → api → schemas | `PairedBlockReducer` set `EffectEstimate.raw_samples` to nested **tuples** (the estimate is a frozen, hashable dataclass); `schemas.canonical_json` **refuses tuples**. Every reduction the reducer actually produced raised `TypeError` out of `content_hash(event)` — the record could not be hashed, journaled, or emitted. Neither unit suite saw it: each fixture used the shape its own module wanted. | `api._canonicalizable()` converts tuple→list recursively for the event's `performance.raw_samples`, and RAISES (naming the path) on anything else `schemas` cannot represent rather than stringifying it. |
| controls → api | `ControlPanelResult.definitions_check` — the control **definitions and predicate** tamper digest — had no field in `api.WindowAttestations`, so *"any post-hoc change to … the control definitions"* (What voids a run) could not reach `check_void_conditions` at all. A campaign could rebind a control predicate and every record in the window still read as clean. | New required `WindowAttestations.control_definitions_immutable`, folded into `VOID_POST_HOC_RULE_CHANGE` and into search-grade conjunct 8, plus `controls.window_control_attestations()` — the projection that fills it, mirroring `statistics.BlockReduction.window_checks`. |
| api ↔ statistics | `CalibrationOutputs.e_process_construction_id` accepted any non-empty string while `statistics.CONSTRUCTIONS` is the bundle's registry. Three suites' fixtures named `paired_betting_supermartingale/v1`, which no reducer implements — a recorded selection nothing can reproduce. | `api.E_PROCESS_CONSTRUCTION_IDS` is the registry-of-record and `CalibrationOutputs` refuses an id outside it; `statistics.py` asserts at **import time** that its own registry has exactly those ids, so drift is an `ImportError` rather than an unreproducible record. |
| surface → integrity | `SourceIntegrityInputs.declared_surface_scope` is a caller-supplied `"full_tree"`/`"partial"` string and the core-header tier FAILs on an under-declaration — while `surface.AffectedSurface.full_tree` is the **derived** answer to the same question, and nothing compared them. A caller who derived `full_tree=True` and typed `"partial"` got a PASS. | `integrity.surface_scope_for()` projects the derived manifest into the scope string, `check_declared_surface_scope()` compares them (COULD_NOT_CHECK when unbound), and `SourceIntegrityGateRunner(derived_surfaces=…)` emits `integrity.declared_surface_scope_binding`. The binding is a declared capability — `surface_binding` says whether the runner has it, and a registered-but-missing manifest raises rather than falling back. |
| api ↔ any runner | A gate runner returning **zero** gates derived to `status: pass`: `_derive` walks the gate list and an empty list contributes nothing to worsen. `TierDispatcher` guarded the *unregistered*-tier case for exactly this reason and left the empty-return case open. | `TierDispatcher.dispatch` raises `EvaluatorNotWired` on an empty gate sequence: a tier that produced no findings because it ran nothing is not a tier that found nothing. |
| api ↔ statistics (record) | `render_search_record_grammar` prints `request.metric` and `request.metric_direction` beside `effect.value`, and nothing compared them. An estimate of a *different* quantity rendered under this cell's metric name and was fully search-grade. | `check_record_grammar_complete` FAILs on a metric or direction mismatch, citing `MEASUREMENT.md:25-26` — the grammar's head is one triple. |
| api ↔ calibration | `EffectEstimate.noise_floor` need not be the cell's calibrated `φ`, and `_resolve_effect` reads the floor **on the record**. Zeroing it turned a sub-floor estimate into a rankable `improvement`, defeating *"an estimate whose magnitude does not exceed φ MUST NOT be ranked"*. | `evaluate_search_grade` FAILs `calibration_block_accepted` when the record's floor is not the cell's calibrated `φ`. |
| api ↔ schemas ↔ journal (**a voided run could not be journaled**) | *"A voided run is journaled as `INVALID` with its reason, and is **never silently discarded**."* `evaluation_event.v2` required `anchor.binary_sha256`/`linkage_sha256` unconditionally and `Journal.append` validates before it writes — so the ANCHOR-MISSING void, the one case where there is no digest to record, could produce **no valid record at all**. `build_evaluation_event` was right to raise rather than invent a digest, and the run survived only as `durable_payload`, which is not the primary record. | `evaluation_event.v3` permits `anchor` to be **structurally absent**, and only when `status == "invalid"` **and** `integrity_flags` names an anchor void reason (`schemas.ANCHOR_VOID_REASONS`, checked against `api.VOID_REASONS` in `test_conformance`). `schemas.is_placeholder_digest` refuses `0`*64 and friends in an anchor, so the exemption cannot be taken by faking one instead. `AnchorMissing` now fires only for an anchor **claimed** and unreadable/fabricated. |
| schemas ↔ precondition 4 | Precondition 4 names the anchor by source commit **and** binary SHA-256 **and** linkage SHA-256. `evaluation_event.v2.anchor` carried two of the three, so `source_commit` rode along as an unvalidated extra key in a free-form block — present in practice, checked by nothing. | `v3.anchor.source_commit` is REQUIRED and validated by the same `_need_commit` helper as `candidate.worktree.source_commit`. |

## Integration seams reconciled — AK1 + AK2 substrate (2026-08-03)

The five substrate modules were built in parallel against the same design. Each
passed its own suite; the defects were all *between* them.
`autokernel/test_integration.py` is the regression barrier for these.

| Seam | Disagreement | Resolution |
|---|---|---|
| journal ↔ storage | `storage.tombstone_id` identifies a reclamation by (campaign, path, sha256, kind, rule); the journal's receipt view was keyed by **the content hash alone**. Two byte-identical build trees at two paths were two reclamations there and **one** slot here, and `check_view_consistency` said PASS because it recounted by the same key. | `journal.tombstone_view_key()` keys on (hash, path); `path` is now REQUIRED by the native TOMBSTONE validator; the checker adds a second, independently keyed recount over distinct `tombstone_id`s. |
| storage → journal | The sink handed the journal a record carrying `epyc.autokernel.artifact_tombstone.v1` that `validate_artifact_tombstone` had never seen — `plan_expiry` validates only the `intent` record, and the `reclaimed`/`failed` variants are built afterwards. | `JournalTombstoneSink.append` validates every record before it goes. |
| preflight → journal | `require_no_concurrent_inference` hands the attestation back on the exception "so the caller can journal it" (invariant 7) — and the journal's **closed** kind vocabulary had no kind to journal it as. The instruction was unfollowable. | New native kind `PREFLIGHT_ATTESTATION` + `Journal.append_preflight_attestation()`. Its validator mirrors `PreflightResult.__post_init__`, so the durable record cannot say less than the object it came from. |
| device_claim → preflight | `preflight` needs a `GpuClaimReader`; `device_claim` produces `ClaimReceipt`s; nothing bridged them, so **every GPU-scoped preflight was `COULD_NOT_CHECK` even with the claim held**, and preflight's refusal text still said the substrate "does not exist yet". The obvious hand-rolled bridge also passed `ClaimReceipt.holder_label: Optional[str]` into `GpuClaimWitness.holder_label: str`, rendering a FAIL finding's `whose` as the literal string `"None"`. | `claim_witness.device_claim_witness_reader()` / `gpu_claim_sources()`; `GpuClaimWitness.__post_init__` now refuses an unattributable witness. |
| device_claim ↔ schemas | `evaluation_event.resource_claim_receipt` is an opaque string, and a non-empty string is exactly what an *invented* receipt also is. `check_device_claim_held` needs `{claim_id, device_id}`; the event carries no device. Nothing downstream could resolve the binding. | `claim_witness.resolve_claim_receipt()` / `check_event_claim_receipt()` resolve the id against the claim journal, which already records the full receipt on every `claim_acquired`. |
| storage ↔ everything | `storage.py` did an unconditional import-time `sys.path.insert(0, <autokernel dir>)` and imported a **flat** `schemas`, so `storage.Check is schemas.Check` was `False` and every `isinstance(v, schemas.Check)` across the seam silently said no — and `import resource` anywhere later in the process resolved to `autokernel/resource/__init__.py` instead of the **stdlib** `resource`, the shadowing that package's own docstring forbids. | Package-relative import with a file-identity assertion, `sys.path` touched only on the genuine flat-import fallback. Asserted structurally from each module's AST, since the test files legitimately insert paths themselves. |

---

## The scar-tissue refactor — Steps 0–2 (2026-08-04)

The refactor plan
([`/workspace/artifacts/operator/autokernel-refactor-plan.md`](/workspace/artifacts/operator/autokernel-refactor-plan.md))
found that ~21,000 of the ~22,700 duplicated lines lived inside the code the
simplification review had already condemned, so **the volume prize was collected
by the deletion, not by refactoring.** What was left in the keep set was four
live defects and two small hoists. Those are Steps 0–2 and they are done. Steps
3–5 (`capability.py`, the `Derived` mixin) were **not** done and are not
scheduled: Step 3 is *run the loop*.

**Read the line accounting before the prose.** Steps 0–2 deleted **196** lines of
production code and added 470, for a **net +274**; with tests the net is
**+1,112**. The plan's headline "~1,300 lines collapsed in surviving code" was
never Steps 0–2's to deliver — its own table attributes ~430 of that to Step 4
alone, and Step 2's own text calls the validator prize "57 redundant lines … a
trivial line prize". **This refactor made the package larger.** What it bought is
below, and it is a property, not a line count.

### What is now impossible

| Was | Is |
|---|---|
| `correctness.BuildProvenance.output_binary_sha256` — the identity of the built candidate — accepted `"0" * 64` | Refused. `schemas.require.sha256` is the one body and it calls `schemas.is_placeholder_digest`. |
| `worktree._req_sha256` accepted `"0" * 64` | Refused. The one legitimate filler-shaped digest, `integrity.EMPTY_TREE_SHA256` (= `sha256("")`, the reading that proves a build dir was clean), goes through `worktree._req_tree_digest`, which admits that value and nothing else — verified by execution against `"0"*64` and `"f"*64`. |
| `BuildProvenance` and `DiffPolicyEvidence` carried no `produced_by`, so a candidate-supplied record was indistinguishable from a measured one | Both carry it, `_req_producer` validates it, and `check_clean_build_from_snapshot`, `check_semantic_diff_conformance` and `check_schema_and_diff_policy` each **FAIL** a record whose producer is not the evaluator. The field is not inert: same record, producer flipped, outcome flips PASS↔FAIL. |
| `api.audit_no_write_or_process_paths("")` returned **PASS** — the guarantee obtained by deleting the thing under inspection | `COULD_NOT_CHECK`, for `""`, whitespace, a comment, a lone docstring and `x = 1`. The no-argument path is additionally bound to `MODULE_ID` by `_defines_this_module`, so the self-audit proves it read its own file instead of assuming it. |
| Nine of eleven `Check` reducers derived **PASS from zero sub-checks** | `schemas.Check.worst_of` is the one lattice; an empty vector is `COULD_NOT_CHECK` with a stated reason. All four keep-set reducers delegate. Verified: no input — `()`, `[]`, `iter([])`, an exhausted generator, `""`, `{}`, `set()` — yields PASS, and a non-`Check` element raises rather than being skipped. |

The **outcome** of every reducer is unchanged on non-empty vectors: 168
one-to-three-element vectors over PASS / PASS-with-prose / COULD_NOT_CHECK /
FAIL, differentially compared against the pre-refactor bodies, agree in every
case. **Reasons did change** — non-PASS reasons are now prefixed `[FAIL]` /
`[COULD_NOT_CHECK]`, and reasons attached to a PASS sub-check are dropped. The
only consumer of a reduced `.reasons` is `campaign.py:1593`, and it reads them
only on the non-PASS path, where they are preserved.

### What is NOT closed, and must not be read as closed

- **The identity-audit hole (S4) is still live.** `evaluator/integrity.py:3526`
  returns **PASS** for `""`, for a comment-only module and for `X = 1` — verified
  by execution, today, on the current tree. It has its own AST walker and does
  not delegate, so it did not inherit the `api.py` fix. It is untouched on
  purpose (`integrity.py` is condemned; the plan forbids refactoring code about
  to be removed). `evaluator/controls.py:2389` still has the same
  `source: Optional[str] = None` signature hole but **delegates** to `api`, so it
  now answers `COULD_NOT_CHECK` — closed by inheritance, not at source. The other
  two sites the plan named, `controller/guards.py:3556` and `release/t3.py:6124`,
  were **deleted** rather than fixed. So the class stands at: one fixed, one
  fixed by delegation, two deleted, **one live**. The full closure is Step 4's
  required `module_id` kwarg on `capability.prove`, which is deferred until after
  campaign #1.
- **`evaluator/api._require_sha256` is shape-only and accepts `"0" * 64`.**
  `AnchorIdentity.binary_sha256` / `linkage_sha256` — the fields precondition 4
  exists for — are validated by it. Migrating it errors 152 tests, 8 of them in
  `evaluator/test_surface.py` (condemned) and 3 in `test_program_md.py`, so it
  unblocks the day those land. It is enumerated in
  `test_schemas_require.KNOWN_WEAKER_DIGEST_VALIDATORS`, the list fails if it
  grows, and `test_the_known_weaker_entry_is_still_earned` fails the day the debt
  is paid and the licence is left behind.
- **A validator is not the same thing as a validated field.** Of the 30 keep-set
  records carrying a digest field, 11 validate it strictly; 5 (all in
  `evaluator/api.py`) use the weak validator above; and **6 have no
  `__post_init__` at all** — `journal.TornTail`, `journal.Views`,
  `storage.ExpirableArtifact`, `statistics.CalibrationSolve` and
  `worktree.BuildResult` accept `"0" * 64` in a digest field today. This is
  pre-existing (verified identical before the refactor) and outside what
  `test_schemas_require.py` can see: that guard proves no module *re-derives* a
  validator, never that a record *uses* one.

### What makes the hoist durable

Not the hoist. `test_schemas_require.py` (32 tests, in `PYTEST_SMOKE`). Writing a
fourth `_req_sha256` needs the shape and the predicate; both live only in
`schemas`, so there are two routes and the quiet one is closed — no keep-set
module may contain `^[0-9a-f]{64}$` as a whole-string pattern (deliberately not a
substring rule: `storage._SHA256SUMS_LINE_RE` and `t0_provider._BUILD_LOG_REF_RE`
legitimately *parse* a line containing a digest), no second `*placeholder*`
predicate may exist, every promoted name must resolve to `schemas.require.<family>`
by AST **and** by `assertIs` identity. The remaining route is
`import schemas; schemas.SHA256_RE`, which is a line on the diff.

---

## The shared lock root, and the follow-up that belongs to `epyc-orchestrator`

`device_claim.py` writes `gpu_device.<device_id>.lock` into **the same on-disk
root** as the orchestrator's CPU region locks:

```
/mnt/raid0/llm/tmp/
├── cpu_region.frontdoor.q0.lock      # epyc-orchestrator/src/runtime/cpu_region_lock.py
├── cpu_region.GLOBAL.all.lock
├── gpu_device.mi210_0.lock           # autokernel/resource/device_claim.py
└── gpu_device.mi210_0.revoke.json
```

Resolution order is copied **exactly** from `cpu_region_lock._tmp_dir()`:
`ORCHESTRATOR_TMP_DIR`, then `ORCHESTRATOR_PATHS_TMP_DIR`, then the hard-coded
path. **There is deliberately no AutoKernel-specific override env var**: a
research-repo-only variable would let the two repositories resolve different
roots and silently stop excluding each other, which is the single failure the
whole module exists to prevent. Tests pass `lock_root=` explicitly instead.

Sharing the root is what makes an orchestrator process and a research process
exclude each other **without either importing the other's code** — the exclusion
fact is a kernel `flock`, not an agreement between two Python packages.

### Follow-up owned by whoever holds `epyc-orchestrator`

AK2's checklist requires *"extend `region_lock_cli.py` with a device verb, or add
a sibling CLI sharing its lock root; **do not fork the lock semantics**"*. That
is a change in the orchestrator repository and is **not** in scope here — this
package is the research repo's half. Three items are waiting on that side:

1. **A `--device` verb** on `region_lock_cli.py` (or a sibling CLI on the same
   root) that acquires/inspects/revokes through *these* semantics rather than
   reimplementing them.
2. **Scope or retire `src/gpu_lease.py` for cross-process use.** It is a
   `threading.Condition` lease: correct for intra-process ownership inside one
   orchestrator (`axa2_live_cutover_bundle.py:535`), and structurally unable to
   exclude a second process. Either label it intra-process-only or migrate that
   call site.
3. **An explicit mode/ownership statement for the lock root.** It is currently
   `0777` with lock files at `0o666 & ~umask`. Cross-repo exclusion depends on
   every participant being able to `O_RDWR` the file; a different uid with
   `umask 022` fails *closed* (good) but is then silently unable to participate
   (not good).

A fourth, smaller item lives on the research side and is recorded here so it is
not lost: `preflight.py` mirrors `cpu_region_lock`'s payload key set, and the
test that "pins" it is circular (it asserts a constant defined in the same test
file). Making it non-circular needs a cross-repo read and is a design call.

---

## Running the tests

Plain `unittest`; no pytest dependency, no network, no GPU, no inference. Every
suite is safe to run on the shared host at any time.

```bash
cd /mnt/raid0/llm/epyc-inference-research

# The whole package (this is the gate).
python3 -m unittest discover -s scripts/kernel_rnd/autokernel -t . -p 'test_*.py'

# Same, with the resource-leak screen the repo just fixed a class of bugs for.
python3 -W error::ResourceWarning -m unittest discover \
    -s scripts/kernel_rnd/autokernel -t . -p 'test_*.py'

# One suite.
python3 -m unittest scripts.kernel_rnd.autokernel.test_integration
python3 -m unittest scripts.kernel_rnd.autokernel.resource.test_claim_witness

# The cross-module suites — the ones that fail when two modules disagree.
python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_conformance.py
python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_integration.py
python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_ak4_conformance.py
python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_loop_integration.py
python3 -m unittest scripts.kernel_rnd.autokernel.release.test_release_integration

# As a plain script.
python3 scripts/kernel_rnd/autokernel/test_integration.py
```

Expected: **3824 tests, OK (expected failures=1)** as of the 2026-08-11 offline
C4 report closure. The one `expectedFailure` is
`test_preflight.RealKernelLockEncodingTest.test_KNOWN_HOLE_unlinking_a_held_lock_
file_hides_its_live_holder` — a real, documented hole (unlinking a held lock file
hides its live holder from the `/proc/locks` witness), deliberately left visible
so the suite flips to a failure the moment someone closes it.

The package is also in `Makefile`'s `PYTEST_SMOKE`, so `make lint` and
`make test` cover it:

```bash
make lint     # ruff over PYTHON_SMOKE + PYTEST_SMOKE
make test     # uv run pytest -q $(PYTEST_SMOKE)
```

### Import convention

Import **through the package** (`from autokernel import journal`), not by putting
`autokernel/` or `autokernel/resource/` on `sys.path`. Both shortcuts load a
second copy of `schemas.py`, and the second one shadows the stdlib `resource`
module for the rest of the process. The existing per-module test files still use
the flat idiom for historical reasons; new code should not.

This is not stylistic. Under `unittest discover -t .` the package is already
imported as `scripts.kernel_rnd.autokernel`, so `from autokernel.evaluator import
surface` creates a **second module object with different classes**, and every
`isinstance` guard across that boundary returns False for an object that is
genuinely of that type. The release plane's two flat suites were converted for
exactly this reason (seam 8 above); the remaining ~23 are a live hazard for any
future cross-suite fixture reuse.

---

## Current completion boundary (2026-08-11)

The lean campaign path is implemented through a terminal bank/refuse result. The
current C6 tranche adds a full-device-synchronized ranked twin, exact thread-set
escape checks, per-shape physical ceilings, a stated 10-planted/15-clean detector
corpus, ranked anti-short-circuit units, and an observe-only sabotage/sandbagging
monitor contract. A terminal journal entry now exports the durable dashboard
contract; implementation activity is presentation context and cannot make an absent
runtime producer look live.

What remains is not another hidden static plane:

- A real v9 CPU candidate campaign has not run. It requires an explicitly authorized
  inference window. Until it runs there is no real proposal archive, banked candidate,
  champion, or empirical input for AK-WM-2/AP-WM-1b.
- The independent C2 property/reference oracle is owned by the ROCm verify/profile
  backend. AutoKernel consumes its evidence but does not manufacture a sibling
  implementation and call that independent. The consumption seam is now complete:
  `test-backend-ops` `AK_PROP_V1` receipts are parsed separately from reference and
  diagnostic text, the residual verdict is re-derived, suite-seed mismatches refuse,
  and each per-op/backend/shape residual travels as a structured
  `epyc.autokernel.property_measurement.v1` row inside the evaluation event's
  `t0.backend_op_units` vector. Real rows still await a sealed experimental instrument
  and an authorized backend-op run.
- The layout axis is a separate T0 pass rather than an implicit side effect of the value
  suite. `OpSuitePlan.layout_probe` emits `--autokernel-layouts`; the tool selects only
  cases with transpose, stride-gap, or offset inputs and emits a suite-seed-bound
  `AK_LAYOUT_V1` receipt. The gate requires all three families and at least one case,
  while an unsupported layout is a hard failure rather than `not_supported`. The
  instrument compiles; no layout case has been executed in this session.
- The value axis is independently selected by `OpSuitePlan.value_transform_probe` and
  refuses to coexist with the layout flag. Each packed floating case runs identity, x3,
  x0.01, and negate against the same shape with fail-any semantics; a passing receipt must
  show all four completed. Property residuals carry the input-transform coordinate through
  the evaluation event and Vidya projection. The experimental target compiles, but no
  transform case has run in this session.
- The recurrent-state axis is independently selected by `OpSuitePlan.stateful_probe`.
  Each emitted `SSM_SCAN`, `SSM_CONV`, cache-backed `FLASH_ATTN_EXT`, or
  `GATED_DELTA_NET` case must carry `AK_STATE_V1` proof that its explicit state inputs
  began byte-identical across candidate/reference runs, remained byte-identical to their
  initial contents, and exposed at least one final-state tensor in the compared output
  set. Missing cases, receipts, target ops, seed bindings, or any triad leg refuse. The
  experimental target and no-backend contract self-test compile; no stateful case has run
  in this session.
- The release and speech-adapter planes were deliberately removed on 2026-08-04 and
  remain recoverable from `autokernel-preserve-20260804`. Restore the narrow release
  slice only for a real champion/freeze request, and restore the speech slice only
  when a speech campaign is scheduled.
- Real restart/crash/resource-preemption/tamper campaign rehearsal and every freeze,
  re-anchor, and watch-window row remain empirical work; fixture and fault-injection
  tests do not substitute for those runs.

## Historical pre-pruning audit (superseded as current status)

The sections below preserve the 2026-08-03 red-team inventory. They are useful as
design history, but many named planes were deleted or narrowed by the operator's
2026-08-04 lean-loop decision. The current boundary above and `FOOTPRINT.md` are
authoritative for what is live now.

### Remaining in AK1

- **Bootstrap knowledge corpus (§19).** The event *kinds* exist and are
  validated (`LEGACY_SOURCE_DISCOVERED`, `PRIOR_ATOMIZED`, `SEED_COMPILED`, …)
  but nothing enumerates or content-hashes the historical source manifest,
  atomizes the source draft, or compiles the three derived memory products. No
  regime-matched retrieval proof for fixed planner/critic fixtures.
- **`kernel_store.py` quarantine.** Existing rows are not yet marked
  `legacy_unverified`, and `kernel_eval.sh` is not yet marked
  deprecated-and-unrunnable. Contaminated legacy rows can still reach a reader.
- **§19.3 suppression receipts** are validated on `RETRIEVAL_SUPERSEDED`, but
  nothing re-verifies them on an anchor move.
- **SQLite as a rebuildable view.** The journal is the source of truth and
  `rebuild_views()` folds it in memory; there is no SQLite projection and no
  rebuild command.
- **Failure / mechanism / do-not-repeat / context views.** `Views` covers
  campaigns, proposals, candidates, evaluations, champions, release packages,
  waivers, tombstones, the frontier and stop states. The planner-facing
  *failure*, *mechanism* and *do-not-repeat* views do not exist.
- **`Views` are record-scope, not retrieval-scope.** Slot payloads carry raw
  `narrative`, including for a `RETRIEVAL_SUPERSEDED` belief. The invariant-20
  boundary lives only on `Journal.retrieve()`; a consumer rendering views into a
  planner brief must apply `strip_narrative()` and
  `retrieval_superseded_event_ids` itself. **This is the highest-value remaining
  item in the journal** and needs a retrieval-scope view API.
- **`PREFLIGHT_ATTESTATION` is folded into no view.** It is in the record and the
  consistency checker ignores it, exactly like `OPERATOR_CONTROL_ACK` and
  `VIEW_REBASED`. AK3's evaluator is the consumer that will want the fold.
- **The §3.7 durability exposures are not cleared** — the np_context decision
  surface is still under `/mnt/raid0/llm/tmp/`, the two np_context study bundles
  are untracked, and the P2-5j protocol is not restored to git.
- **`check_evidence_durability.py` is not extended** to cover AutoKernel
  citations.
- **Validators not yet written:** stale production anchor, undeclared change
  class on a proposal (the field is validated; the *policy* is not), unbounded
  resource/storage request, missing fallback.
- **`kernel_store.py:88` file-handle warning** is not fixed, and pytest is not
  pinned in `pyproject.toml` / `uv.lock` (the Makefile still injects it with
  `--with pytest`).

### Remaining in AK2

- **CLOSED 2026-08-03/04, hardened 2026-08-11 — worktree managers and build layout.**
  `execution/worktree.py` creates `llama.cpp-ak-<campaign_id>` worktrees off the
  re-resolved production tip, namespaces `ak/<campaign_id>/…` branches, does
  pathspec-limited commits in the shared clone, configures and builds with
  `GGML_CCACHE=OFF` forced and a load-average cap that is a precondition rather
  than a note, and emits a build-identity receipt. Production-path denial is now
  a TYPE (`SandboxPath`) rather than a check, and `GitRepo` — the only class that
  may address a frozen tree — carries no content-mutating git verb. Candidate-
  controlled configure and build processes now run under the C6 sandbox too;
  tiny real CMake projects exercise that exact path, while no kernel candidate
  build has yet been run.
- **CLOSED 2026-08-03/04 — CPU region claim *acquisition*.**
  `execution/cpu_region_claim.py` acquires it: per-region `flock`s with a payload,
  a journal, partial-acquisition unwind, and a `verify_held()` that re-reads the
  lock rather than returning a cached flag. `check_precondition_1` is the
  conjunction of "the locks are held" and "the footprint is covered".
  Cross-repo integration with the orchestrator's `cpu_region_lock` remains open
  (see the three items above).
- **Co-residency policy integration.** `evaluation_event.co_residency` is
  validated; nothing decides it.
- **Owned cgroup / PID-receipt process scope with verified teardown.** Delivered
  for candidate execution by `execution/sandbox.py`: one cgroup-v2 leaf per
  invocation, descendant kill through that leaf only, empty-membership proof,
  then removal.
- **Host-health / reboot-required / cache-preparation states**, the one-week
  uptime ceiling, and the §10.7 reboot decision package.
- **Session-bus registration** — roster id, heartbeat at every task boundary,
  outbox, lane declaration, revoke handling, C19/C20 visibility, and the
  re-read-instructions checkpoint.
- **`scripts/utils/agent_log.sh` wiring** and rollback-command logging.
- **CLOSED 2026-08-11 — C6 candidate execution is native and fail-closed on this host.** Landlock
  ABI 6 confines writes to the invocation-owned campaign directory; seccomp
  denies host signalling, networking and namespace escape; uid 0 is refused;
  finite rlimits and the owned cgroup apply before `execve`. Candidate-controlled
  CMake configure/build, live T0, and paired-block runners all require this path.
  Activation receipts and stdout/stderr live in an evaluator-owned sibling the
  candidate cannot rewrite; teardown drains descendants, proves empty membership,
  and removes the cgroup. Recorded/replay runners spawn nothing.
- **`kernel_eval.sh`'s `gpu_idle()` is not yet deleted.** AK2's acceptance
  criterion is "deleted, not wrapped"; the replacement now exists, the deletion
  has not happened.
- **Resource starvation / drain / resume tests and campaign checkpointing.**

### Remaining in AK3

The evaluator is the machinery, not the run. Everything below is a real gap and
none of it is closed by the suites above.

- **PARTLY CLOSED 2026-08-03/04 — the execution layer is WRITTEN, and it has
  still NEVER BEEN RUN.** `execution/` now holds the five executors the seams
  were declared for: `cpu_region_claim.py` (CPU region claim acquisition, real
  `flock`s), `worktree.py` (production-tip anchoring, campaign worktrees, cmake
  build, build-identity receipt), `t0_provider.py` (the real
  `correctness.T0EvidenceProvider` — it launches `test-backend-ops`,
  `verify_ggml_linkage.sh`, generations and the sanitizers), `microbench.py` (the
  T1 paired-block `llama-bench` runner) and `control_runner.py` (the real
  `controls.ControlRunner`). `execution/chain.py` holds the projections between
  them and the evaluator, and `execution/test_execution_chain.py` walks the whole
  composition — claim, worktree, build, T0, T1, controls, verdict, bank,
  teardown — and asserts the frozen trees are byte-identical afterwards.

  **What has NOT happened: any of it running.** No candidate has been built by
  this code, no op suite has been launched, no benchmark has been taken, no
  calibration block has been solved on real A/A material. Every number in every
  test is still synthetic or a recorded fixture replayed through
  `RecordedProcessRunner`/`RecordedSpawner`. AK3's acceptance criterion — *"the
  first phase that consumes inference"* — is therefore **still not met**: the
  machinery now exists end to end, the campaign does not. The host was at load
  ~67 with six resident `llama-server` instances when this was written, and a
  bench under that is garbage data and theft besides.

  `execution/README.md` is the runbook for the session that owns compute: cold
  start to a first candidate, what to check before starting, what "it is working"
  looks like, what should abort it, and the honest list of what still blocks a
  first campaign. Three items on that list matter before anything is run:

  1. **No candidate can cross the evidence threshold.** For the CPU decode cell
     the calibration solves `B_min = 5` and `threshold = 10`, and the
     sign-martingale e-value over five same-sign blocks tops out at 5.57 —
     verified at four different effect magnitudes, all returning the same
     e-value, because the construction is sign-based. Crossing needs the declared
     extension round, and `MicrobenchPlan` has no `segment`/`extension_round`
     field, so `MicrobenchRunner` emits `SEGMENT_BASE` and nothing else.
     `statistics._check_extension_structure` and `microbench.plan_blocks` are
     both already ready for it. Pinned by
     `test_execution_chain.TestTheExtensionRoundHasNoProducer`.
  2. **T0 produces nine of seventeen surfaces.** A clean candidate today reads
     8 PASS / 9 COULD_NOT_CHECK / 0 FAIL. Symbol tables, the parsed diff, the
     sanitizer and boundary-shape gates and the state-safety gate all need
     producers or a `ChangeSurface` derivation the plan does not currently carry.
  3. **The anchor triple cannot name two tools.** `api.AnchorIdentity.binary_sha256`
     is single-valued; T0 hashes the anchor `llama-cli` and `microbench` compares
     against the anchor `llama-bench`. `chain.bind_anchor(..., tool=…)` plus
     `chain.check_anchor_build_is_one_build` is the workaround; the api.py change
     is a required follow-up.
- **Four seams between the executors and the evaluator were mismatched and are
  now projections with tests** (`execution/chain.py`): two different
  `BuildProvenance` records with an INVERTING field
  (`production_tree_paths` is a denylist, `production_tree_paths_touched` is a
  violation list); artifact digests that must be MEASURED rather than copied off
  the receipt, or the clean-build gate's equality becomes `x == x`; the anchor,
  per tool; and one claim object that two Protocols want in two different shapes.
  Each has a mutation-verified test and a compliant-path control.
- **The remaining runner seams.** `api.TierGateRunner` still has three
  evidence-consuming implementations (`correctness.T0CorrectnessRunner`,
  `integrity.SourceIntegrityGateRunner`, `surface.SurfaceGateRunner`).
  `correctness.T0EvidenceProvider` now has two implementations —
  `StaticEvidenceProvider` (a dict, for replay) and
  `execution.t0_provider.ExecutedT0EvidenceProvider` (the real one) — and
  `controls.ControlRunner` has `execution.control_runner.ExecutedControlRunner`.
  `integrity.SourceIntegrityGateRunner` and `surface.SurfaceGateRunner` still
  consume evidence nothing produces.
- **Two derivations of the §8.5.1 gates coexist.** `integrity.py` implements them
  over ELF tables, parsed diffs and build provenance; `correctness.py` carries
  three shallower `t0.source_integrity.*` gates over self-declared evidence
  objects. Composing both runners emits both (the ids do not collide, and
  `test_conformance` asserts that), and both are `integrity`-class so either
  failing blocks ranking — but a caller wiring ONLY the correctness runner gets
  the shallow three and nothing says the deep set never ran. Converging them is
  an AK4 decision, not a local edit.
- **`derived ⊇ traced` is unsatisfiable against a full-suite trace.** `surface.py`
  compares the whole traced op set against the candidate's derived surface, so a
  T0 trace that also dispatched unrelated ops reads as an escape. Either the
  trace must be pre-filtered to the derived surface or the symbol axes must
  compare only *changed* dispatch; nothing states or enforces either today.
  `TracedSurface` also records no denominator of what the trace covered, which
  the record grammar's `scope=` requires.
- **Precondition 6 reaches one argv surface of about six.** Only
  `correctness.SanitizerInvocation` carries an `api.RecipeReceipt`;
  `OpSuiteEvidence`, `CoherenceEvidence`, `DeterminismEvidence`,
  `BoundaryShapeEvidence`, `LinkageEvidence` and `StateSafetyEvidence` carry an
  opaque `receipt_ref: str`. **A hand-typed `test-backend-ops` argv is
  undetectable at T0.** `recipes.py` also exposes no `verify_receipt()`, so
  `api.check_preconditions` passes precondition 6 on the mere *presence* of a
  `RecipeReceipt` — a frozen dataclass of three well-shaped strings.
- **Three §8.5.1 gates cannot detect a self-report.** `BuildProvenance` and
  `DiffPolicyEvidence` have no `produced_by` field, so `build_dir_was_fresh`,
  `incremental_objects_present` and `commit_was_pathspec_limited` are taken on
  faith. Every other evidence type has the field.
- **CLOSED 2026-08-03 — precondition 4 is enforced on all three components in
  `correctness`, and every piece of evidence carrying anchor-derived material now
  names the anchor it was captured against.** Five evidence types —
  `LinkageEvidence`, `CoherenceEvidence`, `DeterminismEvidence`,
  `StaticAnalysisEvidence`, `AntiRewardHackingEvidence` — each carry
  `anchor_source_commit` + `anchor_binary_sha256` + `anchor_linkage_sha256` under
  ONE rule (`_validate_anchor_triple`): all three or none — a partially named
  anchor resolves to more than one artifact — and no placeholders
  (`schemas.is_placeholder_digest`), which is the same rule
  `evaluation_event.v3` applies to the record these feed. The linkage gate
  compares the commit as well as the two digests, so an anchor rebuilt from a
  different source at an identical digest is visible. Every consumer compares the
  capture's recorded anchor against the anchor it is handed, using
  `api.AnchorIdentity.identity_matches()` (which therefore now has callers in
  `correctness.py`), and RAISES an `EvidenceAnchorMismatch` subclass —
  `CoherenceAnchorMismatch`, `DeterminismAnchorMismatch`,
  `StaticAnalysisAnchorMismatch`, `AntiRewardHackingAnchorMismatch` — rather than
  downgrading: invariant 11 makes re-scoring saved outputs the normal path, so
  that is exactly where a capture taken against anchor A gets scored against
  anchor B. Evidence that recorded NO identity is COULD_NOT_CHECK on every
  surface, never an implicit match, and never a downgraded FAIL either.
  The live consequence is closed with it: `check_output_coherence`'s
  reconciliation of the anchor's determinism class now requires the coherence
  capture and `DeterminismEvidence` to name the SAME anchor before their
  agreement counts as corroboration — a mismatch raises, and an unrecorded
  determinism identity is COULD_NOT_CHECK, because two records agreeing does not
  establish they describe one anchor.
  **What remains here:** the recorded identity is still the producer's
  declaration — `produced_by` is checked, but nothing re-derives that a capture
  really ran against the anchor it names, so this binds an honest producer's
  replay and not a lying one's capture. `LinkageEvidence` reads its triple
  field-by-field (per-component reason texts) rather than through
  `recorded_anchor()`, so the linkage gate FAILs on a drifted anchor where the
  other four raise — the outcome differs by design, the binding does not.
  `integrity.check_evidence_binding` does bind the ELF tables, and registration
  tables carry no provenance digest at all.
- **The §8.5.1 (5) repair cap is enforced by nothing.** No gate consults a
  `RepairLedger` or `RepairPolicy`; `check_repair_from_clean_parent` accepts
  `attempt_index=99`, and `check_repair_from_clean_parent(None)` PASSes, which is
  exactly how a repair evades the clean-parent rule.
- **Control seed rotation is declared, hashed, and never enforced on the run
  path.** `derive_control_seed`, `ControlBundle.seed_for` and
  `SeedRotationSchedule.check_rotation` have no caller; `run_all` hands all five
  controls the same seed.
- **The calibration outputs are re-typable at the campaign boundary.**
  `CampaignStatistics` takes an `api.CalibrationOutputs`, not the
  `CalibrationSolve` that produced it, so a hand-built outputs object drives the
  reducer. The MDE resists (it is recomputed from the retained A/A pool); `φ`,
  `α_sel` and `α_conf` do not. `CalibrationAttempt.solve_order_recorded` is a
  dataclass *default*, so *"a conforming implementation MUST record that it did"*
  is satisfied by a literal.
- **`ControlPanelResult` has no mint token.** `api.Verdict` re-derives its own
  status and refuses a stamped one; a `ControlPanelResult` built by hand with an
  all-PASS panel yields `may_rank=True` with no control ever run.
- **Calibration recomputation at a campaign boundary and the A/A cadence driver
  are still nobody's.** `release/readiness.py` now computes the advisory signal
  and `controller/composition.py` composes a champion lineage, but nothing
  recomputes the calibration block at a boundary or drives the cadence.
- **T3 is refused HERE and implemented in AK5.** `api.admit_tier("T3")` still
  raises by name — that is the point — and `release/t3.T3Runner` is the
  `ReleaseTierEvaluator` that fills the seam. T4 remains unimplemented.

### Remaining in AK4

- **No runner.** AK4 is the plane that DECIDES; nothing yet drives a real
  campaign end to end against the real evaluator, holds a real device claim for
  a window, or calls a real model. `test_loop_integration.py` walks the whole
  loop against fakes, which proves the seams fit and proves nothing about a
  live host.
- **`stop_policy.max_consecutive_proposal_skips` has no declared home.**
  `selection.planner_health_stop_request()` requires it and rightly refuses to
  invent one, but `schemas.validate_campaign` does not name it, so a
  §7.1-conforming manifest can omit the single input `PLANNER_DEGRADED` needs
  and the loop discovers it by raising. The schema is AK1's;
  `test_ak4_conformance.TestDeclaredCampaignControls` is the record of the gap
  and FAILS the day it is closed.
- **`mechanism_class` on `composition.admit_to_frontier` is a caller parameter**
  bound to nothing but the planner-authored `change_class`, so the §8.9 diversity
  floor is decided by a label a model wrote. Closing it needs a signature change
  (resolve it from the proposal via `views`), which is a design call.
- **Several unbounded reads.** `ProposalScreener.screen` folds the whole journal
  per proposal (O(journal) per call, O(N²) per campaign) and
  `hypotheses.planner_round_block` re-surfaces every attempt of every hypothesis
  forever. `journal.read_since(reader_id)` exists and is unused. Bounding either
  needs a counted, receipted truncation — a silent slice would be exactly the
  discard §8.4 forbids.
- **The prose fields that are still primary keys.** `selection.mechanism` is
  free text and is both the ledger's key and part of the blacklist fingerprint;
  `hierarchy_layer` is self-declared, so the actor chooses how many §8.3 receipts
  it owes. Both need a campaign-declared vocabulary.
- **`context` and `planner` section vocabularies do not match** (nine shared, six
  and ten unshared). Harmless today because the compiler's bundle binds to the
  adapter directly by `manifest_sha256` rather than round-tripping through
  `ContextManifest` — asserted in `test_loop_integration` — but a caller that
  hand-builds a `ContextManifest` from compiler sections has no total mapping.

### Closed by the 2026-08-03 hardening pass (was carried-forward, now done)

Each of these was on the list above until this pass. They are recorded because a
carried-forward list nobody ever *removes* from stops being read.

- **Per-phase protocol ratification** — `t3.PhaseProtocolBinding` carries a hashed,
  ratified binding per `(backend, phase)`. A bare id is still accepted, but as the
  UNBOUND state reported COULD_NOT_CHECK, so the gap is journaled rather than
  passing silently. `declared_ratified_protocol_ids()` derives the adapters' set
  from the request's own bindings — no constant, no flag.
- **`t3.verify_waiver` accepting any non-empty `authorized_by`** — the vocabulary
  moved to `schemas.MACHINE_ACTOR_TOKENS` and both planes read it. Separator
  spellings (`auto-kernel`, `auto_kernel`, `auto.kernel`) are refused too, and
  `machine_attributions` scans five attribution fields. Authorship is now
  *(named non-machine actor)* OR *(no attribution at all AND provenance PASS)*,
  because the ratified v8 waiver names no author and its legacy schema cannot be
  tightened without invalidating a genuine record.
- **The operator-attestation root was manufacturable with `mkdir -p`** — the
  repo-name strip is now taken only for names in `schemas.REPO_CHECKOUT_NAMES`, so
  `/mnt/raid0/llm/tmp/artifacts/operator/…` (the loop's own scratch root) no longer
  reduces to an operator citation.
- **Per-phase standings were deduplicated last-wins** — a duplicate
  `(backend, phase)` is a blocking contradiction naming both standings and both
  evidence refs. It made the verdict a function of the caller's list order, and the
  caller is the party being gated.
- **`expected_gain` never compared to a standing**; **`sealed_fingerprint` blind to
  waiver coverage** (`FINGERPRINT_FACETS` is now 18, including
  `active_waiver_coverage`); **`MechanismConfirmation` carried no timestamp**
  (`measured_at` is required); **`check_matrix_coverage` accepted foreign-backend
  cells *and* foreign-backend capacity deltas** (one `_require_one_backend` door
  over both); **`ArchivedBuild.libraries` carried no backend attribution** (now
  `((backends, path, sha256), …)`).
- **`serving_runtime`'s `kernels/production` substring pattern** is gone, so a §11.6
  package can name its own serving binary; `..` traversal out of the stable path and
  `a//b` slash runs are both refused; gate 2 and gate 3 are tied through typed
  `subjects`/`tied_to` validated in `ThreeGateResult.__post_init__`, so a caller
  assembling a verdict from journalled outcomes cannot skip the tie.
- **The packager's four self-audits** are now alias-, dispatch- and rebinding-aware,
  and package assembly checks that the sealed candidate **is** the graded one and
  that era-row kinds are **traced from `rows[]`** rather than declared.
- **Sub-floor estimates could be selected as the weakest or the best protected
  cell** — the readiness figure ranked precisely the cells the evaluator refused
  to rank. `evaluator/api.py`'s `_RANKABLE_RESOLUTIONS` is `(improvement,
  regression)` and its comment is the governing text: *"Below the noise floor is
  not a small win; it is not a win… a result you cannot order."*
  `release/readiness.py::_phase_figure` filtered only on `_rank_admissible`,
  which is about VERDICT VALIDITY (INVALID, prior gate failed, INCONCLUSIVE) and
  says nothing about effect resolution — and selecting a cell as *weakest* or
  *best* IS a rank.

  **This was an operator decision, not a red-team call**, because the obvious fix
  is wrong on its own: under §1.6's non-inferiority half a HEALTHY backend
  produces cells at `no_detectable_difference`, so simply excluding them makes
  the most common healthy outcome render as `None` — "no protected-cell figure",
  an absence. Absences read as coverage gaps, and a coverage gap is what a later
  session closes by loosening the gate. Three options were put to the operator
  and **option B was chosen**: make the states structurally distinct rather than
  overloading one representation. `_phase_figure` now returns exactly one of
  three things and `PhaseStanding.__post_init__` refuses a fourth
  (`PHASE_FIGURE_TYPES`):

  | State | Representation | Rendered line |
  |---|---|---|
  | nothing measured | `None` | `no protected-cell figure` |
  | all at parity | `ParityFigure` — counts, ids, census, and the **binding** sensitivity (`max(mde, floor)`) of the blindest cell, with that cell and its event named. `value`/`best_value` RAISE rather than being absent, because `getattr(fig, "value", None)` is how a caller restores the absence | `2/2 protected cells at parity, nothing above +/-0.018 distinguishable — measured, no detectable difference at any of them` |
  | orderable | `ReadinessFigure`, selected over the orderable subset only, disclosing `N/M protected cells, P at parity` | `weakest orderable protected cell …` |

  **What a parity standing claims, and what it does not.** It claims: every
  protected cell in the phase was measured, and none produced an effect this
  campaign's own sensitivity can separate from nothing — at a stated bound, from
  a named cell and a named event. It does **not** claim the effect is zero:
  sub-floor means the sign and the size are both unknown, which is why there is
  no number to read off it. It does **not** claim the objective was met — §1.6
  requires an improvement, so a wholly-parity backend is `non_inferior=PASS`,
  `improved=FAIL`, `standing=objective_not_met`. And it does not claim to be
  informative on its own: when the phase could not have resolved the campaign's
  own advisory reference gain, the line says **UNDERPOWERED FOR THIS CAMPAIGN**
  and names the parity result as a statement about the measurement rather than
  about the candidate. The reference comparison stays `COULD_NOT_CHECK` either
  way — AK-D3 made that figure advisory and labelling it cannot make it a gate.

  **The stop rule consumes this, and the seam is typed.**
  `ParityFigure.observation_fields()` carries no `readiness` key, so
  `guards.ReadinessObservation(**fields)` is a `TypeError`; the controller's
  series entries are two TYPES under `ReadinessSeriesEntry`
  (`ReadinessObservation` has a magnitude, `ParityObservation`'s `readiness`
  property raises), and `observation_from_fields()` is the one seam reader,
  branching on which keys exist and defaulting nothing. `guard_plateau` counts
  parity rounds as rounds — dropping them would trend a self-chosen
  subsequence — while they contribute nothing to `best`, and it emits
  `PLATEAU_STOP` on one of two named bases (`plateau_basis`): a measured
  improvement below the floor, or **no detectable effect in any round**, the
  latter reported with no `improvement`/`opening_readiness`/`best_readiness` at
  all, because a substituted `0.0` is a trend in a quantity nobody measured. The
  all-parity basis is admissible only when every round could have seen the
  campaign's target; otherwise, and when no target was declared, the guard
  answers `COULD_NOT_EVALUATE` — conditions a campaign can fix, unlike the
  categorical refusal they replaced, which never terminated a converged
  non-inferiority campaign.
- **The cardinal-rule audit corpus was 18 of 42 modules** — found by this pass, not
  reported by any red team. `TestNoProductionWritePathsAnywhere` globbed `release/`
  and `adapters/` non-recursively. Rule 1 ("writes nothing at all") is correctly
  scoped to those two planes, because `journal.py`, `storage.py`,
  `resource/device_claim.py` and `controller/state_machine.py` write by design. But
  **rule 2** ("no call names a human-only target") is not a property of a plane, and
  it was never applied to the twenty-four modules that actually hold the
  primitives: a `shutil.rmtree('/mnt/raid0/llm/kernels/production/cpu')` appended to
  the real `storage.py` produces a finding the moment the auditor is handed the
  source, and the suite never handed it over. Rule 2 now runs over the whole
  package, discovery is recursive, and the corpus is asserted to contain the
  write-capable modules so it cannot pass by inspecting nothing.
- **§10.4 waivers are READ, not quoted — the largest residual in the release
  plane, and the last one on this list.** `t3.WaiverBinding` carried `document`,
  `document_path` and `observed_sha256` as three independent caller assertions with
  **nothing reading the file**, so a document the caller invented, pinned to its own
  digest, at a path that does not exist verified — and took its *authorship* from
  `attribution_source="operator_owned_path"`, borrowing the standing of a directory
  it was not in. §10.4 turns a FAIL into PASS_WITH_WAIVER, so that was the authority
  path of the entire freeze gate resting on the honesty of the party being gated.

  `t3.waiver_binding_from_path()` is now the only constructor of a trusted binding.
  It checks the citation shape before any I/O, `lstat`s (refusing a symlink, a
  non-regular file, a hardlink, an oversized one), reads **once**, re-`lstat`s for
  `(dev, ino, size, mtime, ctime)`, refuses a resolved path in scratch / a
  production tree / outside the declared `DEFAULT_ATTESTATION_ROOTS`, refuses a BOM
  or a duplicate JSON key (one file that means two things), hashes the **raw bytes**
  (`schemas.raw_bytes_digest` — *not* `content_hash`, which matches no ratified
  `evidence_sha256`), and optionally requires the digest a preserved ratification
  pins. It returns a `ReadWaiver` carrying a `WaiverReadReceipt` that only this
  function can mint, and the receipt's document is asserted to be the parse of the
  very `bytes` object that was hashed, **by object identity** — there is one `bytes`
  object in the function, so "the document returned is the one whose bytes were
  hashed" is a fact rather than a hope. The gate asks
  `t3.waiver_read_violations(binding)`, deliberately **not**
  `isinstance(b, ReadWaiver)`: a three-line subclass that declines to run
  `__post_init__` satisfies the isinstance test with no receipt at all, so the
  capability is the mint token and only something that inspects the token tests for
  it.

  `WaiverBinding` survives as the QUOTATION type, because *"I did not read the
  file"* must stay expressible — a design that made it inexpressible would push
  callers into asserting digests they never computed, which is worse than the
  defect. It fails **closed**: COULD_NOT_CHECK → a blocking identity-phase reason,
  so an unread waiver does not merely suppress nothing, it stops the run. A REFUSAL
  to read is a third state and raises `WaiverNotReadable`; it is never downgraded to
  "nobody looked".

  Three AST audits keep it true after this session:
  `audit_waiver_reader_is_the_only_reader()` (one read call, no second door, the
  mint token named exactly three times), `audit_reader_narrowing_is_never_widened()`
  (no shipping module passes `attestation_roots` / `max_bytes` / `boundary`), and
  `audit_waiver_binding_is_constructed_only_by_the_reader()` — an allowlist of
  `(module, enclosing function)` covering the **three** construction sites that
  exist, so a fourth is a test failure rather than a code review someone has to
  catch. Test modules are out of scope on purpose: proving a quotation suppresses
  nothing is most of the evidence the fix works.

  **What reading the file does NOT give us — read this before citing the above.**
  Reading buys **provenance** (these bytes came from an operator-owned path this
  process opened, not from the caller's argument list) and **integrity** (they hash
  to what was pinned, and did not change under the read). It does **not** buy
  **authenticity**. There is still **no signature anywhere in this system**. Nothing
  proves a human wrote the waiver; the evidence is a filesystem location and a
  digest, and any process that can write `/workspace/artifacts/operator/` can
  produce a document that passes every check in this paragraph — including the
  ratification cross-check, since the ratification is an unsigned JSON file in the
  same directory. `attribution_source="operator_owned_path"` is *"it was where the
  operator keeps things"*, which is a much weaker claim than *"the operator signed
  it"* and must not be quoted as the stronger one. The honest summary is that a
  waiver is now as trustworthy as the write permissions on one directory. Closing
  the remaining gap needs a signature contract (detached signature or a
  human-only-writable ratification chain) and is a measurement-constitution
  amendment, not a code change — the trust boundary is human-amendment-only.

  Residual, named: the fd-based `os.open(O_NOFOLLOW|O_NONBLOCK)` + `fstat(fd)`
  design is **not implementable inside `t3.py`** — `audit_no_write_or_process_paths`
  and rule 1 forbid the release plane from importing `os` or calling `open` in any
  spelling. The reader uses lstat-before / read / lstat-after instead, so a race
  that swaps a regular file for a symlink *between* the pre-`lstat` and the read is
  caught after the fact rather than in the syscall, and the FIFO case is not caught
  at all — swapping in a FIFO at that instant **blocks the gate indefinitely**
  (availability, not authenticity: it needs an attacker who can already write the
  waiver). The trade is deliberate: the capability audit holds against every future
  edit of the module, and rule 1 was **not** relaxed for the reader — reading is not
  writing, `release/t3.py` takes zero rule-1 findings unexempted, and
  `test_the_10_4_waiver_reader_needed_no_exemption` asserts exactly that, so a
  future edit cannot buy itself an exemption it does not need.

### Remaining in AK5/AK6 (the release plane)

- **No runner, and no live release has been rehearsed.** Everything below the
  gate is fixtures. Nothing has compiled a plan from the real
  `orchestration/derived/stack_priors.yaml`, sealed a real candidate, held a real
  compute window, or produced a package an operator has read. The chain is proved
  to FIT; nothing here is evidence about a real freeze.
- **`P-KERNEL-FREEZE-1` is a DRAFT.** `t3.phase_identity_preflight` raises
  `ReleaseProtocolNotRatified` for `mode="release"` under an unratified protocol,
  so **every** run this package can currently perform is a dry run. Ratification
  is human-only and is the gate on AK5 being usable at all.
- **§1.6's `regression_band` is compared to nothing measured.** The pass made
  `expected_gain` realisable against the measured standing; the *band* stays
  structurally uncheckable at T3 because `PhaseStanding` carries a vocabulary
  (`improved`/`non_inferior`/`regressed`), not an effect magnitude — `readiness`
  has the magnitude. Seam: a numeric oriented effect on `PhaseStanding`.
- **`fnmatchcase` `*` crosses `/` in the operator manifest matcher**, so a glob
  written as `measurement/protocols/*` + `.md` also matches documents nested
  several directories deeper, which then read as operator-owned. Broader is *more*
  permissive for this consumer. **Deliberately not fixed:** it changes how a
  human-authored manifest is interpreted, and making `autokernel` disagree with
  `session_bus.py validate`'s matcher would be worse than the widening. Operator
  call.
- **CLOSED 2026-08-03 — `WaiverVerification.covered_cell_ids` is `()` unless the
  check PASSed.** It used to be populated regardless, so a refused waiver landed in
  the durable bundle carrying a waived-looking coverage list — the same defect (a
  record asserting coverage nobody verified) one layer out from the reader above.
- **Identity re-joining can false-positive on a two-part human name** whose parts
  concatenate to a machine token (`"Bo T"` → `bot`). Fail-closed, and documented in
  `schemas.identity_candidates`; introduced by the hardening pass itself.
- **`OperatorCommand.validation_receipt` is any non-empty string.** "Pre-validated
  end to end" is a caller declaration, and `validated=True` also decides which
  commands count for transaction coverage. A structural fix needs a receipt
  vocabulary/digest contract belonging in `schemas.py` and the evidence plane, not
  in the packager.
- **`_transaction_elements` coverage is satisfied by a command that NAMES an
  element**, not by proving it acts on it — a comment mentioning
  `instrument_eras.yaml` covers the era-registry element. Tightening it needs a
  verb vocabulary per element kind.
- **Era-row *content* (era_id prefix, duplicates) is validated only inside
  `draft_era_registry_row`**, not at package assembly, which accepts a hand-built
  mapping. The READY-reaching part of that hole (declared-vs-traced `kinds`) is
  fixed; the content part is not.
- **The mint-alias residual**: a refused class passed through a data structure and
  called later is still unauditable. The assignment-alias route is closed; a general
  rule would forbid the `isinstance` table this module legitimately uses.
- **`_within_surface`'s denominator mixes refs with paths**, so a `target_paths`
  entry equal to a branch or tag name reads as "contained".
### Remaining in AK6 (the operator surface)

Built: the v2 contract and its producer, the panel→producer registry, the
per-panel freshness envelope, the transport watchdog, the `/api/health` fold, the
restart chaos test. Still open — every item below was found by a red-team pass or
by the seam integration and is reported rather than quietly carried:

- **A caller can still date the contract from fabricated inputs.**
  `ControllerObservation`, the champion record and `ReadinessReport.computed_at`
  are caller-supplied value objects: a caller that fabricates a `Transition`, or
  re-runs the reducer over month-old evaluations, stamps a fresh time on stale
  evidence. Not closable at the producer boundary (it must accept the owners'
  objects). Mitigated, not fixed: `observe_controller` reads
  `machine.ledger`, and `producer.run.{campaign_id, controller_seq}` lets a
  consumer see the loop did not advance whatever the timestamps say — which the
  hub's watermark arm already does.
- **`export_contract(path=…)` still accepts any durable, non-scratch,
  non-production, non-checkout `.json` path.** The default is the only path
  anything reads; a caller that passes another one writes a file nobody consumes.
- **`refused` and `not_reported` render identically on the consumer side.** The
  producer distinguishes them (a champion record that failed validation is
  `refused`); the hub folds both into `unreported`, and its verdict sentence says
  "has no producer" for a section whose producer explicitly refused.
- **Nothing schedules the export.** There is no exporter loop, cron or controller
  hook calling `export_contract`, so the durable path stays empty until AK7 wires
  one — which the hub renders honestly (`NOBODY IS REPORTING`, fold `absent`) and
  is the reason `/api/health` reads `absent` on this host today.
- **The autopilot outcome contract has no exporter and its default path is
  ephemeral.** `AUTOPILOT_OUTCOME_JSON` defaults to
  `/mnt/raid0/llm/tmp/autopilot/outcome_contract.json` — the same sweep root the
  kernel export was moved OFF in this phase. It belongs to epyc-orchestrator, so
  it is flagged, not changed. The `outcome` panel is watched anyway: it is the
  panel the trial-1302 outage happened on.
- **§9.7's `llama_cpu` co-residency requirement is dischargeable by a co-resident
  PREFILL cell.** The pass narrowed it by role and admissibility, but not by phase —
  and `_CO_RESIDENT_WHY`'s entire stated mechanism is that *"CPU decode is
  bandwidth-bound"*, which a compute-bound prefill measurement cannot show.
  **Deliberately not fixed:** §9.7's ratified text says only *"at least one
  co-resident cell for `llama_cpu`"* and names no phase, so narrowing it is an
  operator amendment, not a red-team call. Either narrow it (a one-line predicate
  plus its control) or amend `_CO_RESIDENT_WHY` to stop asserting a mechanism the
  check does not enforce.
- **Coverage/repetition checks still count inadmissible cells.** A cell whose
  verdict is INVALID, or which failed a prior gate, still counts toward the §9.7
  coverage and repetition tallies. (The *other* half of this item — sub-floor
  estimates being selected as weakest or best — was decided by the operator and
  is closed; see the readiness-figure entry in the 2026-08-03 list above.)

### Remaining in AK8/AK9 (the adapters)

- **Both speech release-protocol families are drafts**, so `whisper_stt` and
  `qwentts_tts` are search-legal and release-blocked by design. Their phase
  vocabularies are absent from `schemas.PHASES_BY_BACKEND` for the same reason.
- **The gate-2/gate-3 tie forbids gate 3 from verifying any service gate 2 did not
  observe.** Two intended services with one restarted now yields
  `FAIL: service 'router' has no gate-2 start observation`, so a stack change that
  restarts 1 of 4 services cannot have gate 3 verify the other 3 against config —
  and the way to make it pass is to verify *less*. **Not fixed:** it is a genuine
  semantic choice about §11.6's gate 3 (whole stack vs. restarted subset), not a bug
  with one right answer. Recommendation: keep the tie mandatory for restarted
  services and admit an unrestarted one as fully compared but tied-COULD_NOT_CHECK,
  forcing COULD_NOT_CHECK rather than PASS.
- **`subjects`/`tied_to` are forgeable by a caller who hand-builds both outcomes.**
  The binding proves the two outcomes are *about* one process set; it cannot prove
  the tokens were ever observed. Unavoidable in any design that takes observations
  as data — the same limit applied to the string `evidence_ref` it replaced.
- **`classify_stable_kernel_path_use` has zero production consumers.** The real
  guard path is `_stable_kernel_path_findings` inside `_scan_node`; the exported
  classifier is exercised only by tests, so its COULD_NOT_CHECK cannot be misread
  today — but nothing stops a future consumer from treating it as a green receipt.
- **A `..` below the stable path is refused as undecidable, never resolved.**
  Resolving it would mean following the very symlink that is the trust boundary.
  The consequence is that a legitimate command using `..` *below* the link is
  refused rather than admitted — fail-closed, and deliberate.
- **`gate_live_equals_config`'s local `started = _parse_instant(...)` shadows the
  `started` parameter** inside the per-service loop. Harmless today
  (`started_by_id` is built before the loop); a trap for the next editor.
- **`check_device_evidence(expected_lane="gpu")` accepts `Device 0: CPU`** in both
  speech adapters: it requires a `Device N: <name>` line and never checks the name
  denotes a GPU. A correct fix needs a device-name vocabulary that belongs in the
  evaluator bundle.
- **`check_exclusion_rate(aa_dispersion=…)` has a unit hazard** (fraction vs
  percentage-point, no suffix) and **`check_stage_attribution(tolerance_ms=…)` is
  unbounded** — a tolerance larger than the measurement passes.
- **One `EXPECTED_SHARED_LIBRARIES` set for the whole inventory**, so the §10.2
  phase-2 gate is unrunnable for inventory members that link a subset.

### Escalations raised by this work, for the operator

- **The 2026-07-31 ggml-linkage remediation is not live in this container.**
  `/etc/environment`, `.devcontainer/devcontainer.json` and `.devc/overrides.json`
  were all cleaned, but the running container still exports the pre-remediation
  `LD_LIBRARY_PATH`, under which all four frozen speech binaries resolve
  `libggml*.so.0` from `/mnt/raid0/llm/llama.cpp/build/bin` (ggml 0.16.0) while
  loading `libggml-hip.so.0` from their own tree. Any speech measurement taken
  from an agent shell here without an explicit per-launcher `LD_LIBRARY_PATH` is
  attributable to the wrong build. Operator action: container restart.
- **`ratify_speech_kernel_freeze_20260731.json:27-28` anchors "whisper
  large-v3-turbo f16" to `wer_pct: 2.35`**, which in
  `/mnt/raid0/llm/tmp/stt_wer_results.json` belongs to *faster-whisper
  large-v3-turbo int8 CPU 48t* (44/1870). The `whisper.cpp large-v3-turbo f16
  MI210 GPU` arm is **3.37 % (63/1870)**. A ratified receipt is corrected by a
  superseding receipt, which is human-only.
- **§10.4 waivers have provenance and integrity but no AUTHENTICITY, and closing
  that is not a code change.** After the reader (above), a waiver is trusted because
  it was read from `/workspace/artifacts/operator/` and hashes to what a preserved
  ratification pins. Neither fact distinguishes a document an operator wrote from
  one any process with write access to that directory produced — the ratification
  that pins the digest is an unsigned file in the same directory. Every mechanism
  available *inside* this package is now spent; the next step is a signature
  contract (a detached signature, or a ratification chain rooted somewhere the
  agents cannot write), and the measurement trust boundary is human-amendment-only.
  **Operator decision**, not deferred work: whether the freeze gate needs
  cryptographic waiver authenticity, or whether directory permissions plus the
  digest chain are the accepted bar. The gate is safe under either answer — it just
  should not be described as authenticated under the second.

### Not started at all

**AK7** beyond its entry point: `packager.OperatorFreezeRequest` is the door and
`audit_no_clock_or_self_trigger()` proves the packager has no clock and never
constructs one of its own, but the cadence policy AK7 describes has no
implementation (deliberately — AK-D25 keeps cadence an operator policy).

### Known, documented holes in what *is* implemented

- **`evaluator/integrity.py:3526` `audit_no_write_or_process_paths` still returns
  PASS for source that defines nothing** (`""`, a comment, `X = 1`). The same
  defect was fixed at `evaluator/api.py` on 2026-08-04 and deleted with
  `controller/guards.py` and `release/t3.py`; `integrity.py` has its own walker,
  does not delegate, and is on the deletion list, so it was deliberately left.
  **Do not read the S4 class as closed.** See
  [§ *The scar-tissue refactor*](#the-scar-tissue-refactor--steps-02-2026-08-04).
- **`evaluator/api._require_sha256` accepts `"0" * 64`**, so
  `AnchorIdentity.binary_sha256` and `ArtifactIdentity`'s digests can carry a
  fabricated identity. Named in `KNOWN_WEAKER_DIGEST_VALIDATORS`; unblocks when
  `test_surface.py` and `test_program_md.py` fixtures land.
- **Six keep-set records validate no digest at all** (`journal.TornTail`,
  `journal.Views`, `storage.ExpirableArtifact`, `statistics.CalibrationSolve`,
  `worktree.BuildResult`) — they have no `__post_init__`. Pre-existing; the
  `schemas.require` conformance guard cannot see it, because it proves no module
  re-derives a validator, not that a record uses one.
- Unlinking a held lock file hides its live holder from the `/proc/locks`
  witness — a tmp reaper over `/mnt/raid0/llm/tmp` does it, no attacker needed.
  Tracked as an `expectedFailure` in `test_preflight.py`.
- `Journal.write_lock()` re-entrancy is not thread-safe (cross-process is
  correct; two threads sharing one `Journal` both believe they hold a lock only
  one acquired).
- Liveness identity has no PID-namespace component, so two containers sharing a
  hostname with private PID namespaces would classify a *live* foreign holder as
  DEAD. Not triggerable on this host today (`/proc/self/ns/pid` is the init ns).
- Neither `storage.expire_artifact` nor the tombstone path takes any lock; two
  processes can both pass `plan_expiry` on one artifact.
- `_parse_line` rejects any unknown `kind`, so a journal written by a newer
  version is entirely unreadable by an older reader — fail-closed by design, and
  a forward-compatibility wall.
