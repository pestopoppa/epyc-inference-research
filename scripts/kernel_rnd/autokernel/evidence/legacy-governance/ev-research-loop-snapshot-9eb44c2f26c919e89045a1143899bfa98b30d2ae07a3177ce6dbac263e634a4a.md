# AutoKernel — Autonomous System-Wide Kernel Research Loop

**Status:** BASELINE-MATCHED R4 CONTROLS 5/5 PASS / CLEAN PREFILL PAIR RETRY PENDING QUIET HOST / REAL ARCHIVE CHAIN NOT YET STARTED — updated 2026-08-13
**Priority:** HIGH after the current production-topology work settles
**Owner:** Inference Acceleration
**Runtime owner repository:** `epyc-inference-research`
**Parent index:** [Inference Acceleration — Active Index](inference-research-index.md)
**Source draft:** [`docs/reference/autokernel/system-wide-inference-kernel-optimization-draft.md`](../../docs/reference/autokernel/system-wide-inference-kernel-optimization-draft.md)
(SHA-256 `af2fd586d3b1e3b58b038fcc0a0c7d5def22d70b45dbdc54bd64799b082e7b8b`; moved out of `tmp/` on
2026-08-02 because `tmp/*` is gitignored and `MEASUREMENT.md:146-156` forbids scratch citations)
**Supersedes as loop owner:** [`mi210-kernel-rnd-loop-proposal.md`](../completed/mi210-kernel-rnd-loop-proposal.md)
**Absorbed 2026-08-02:** the bootstrap-corpus design pass (now §19) and the full design audit
(findings folded into §§2–15; the standalone audit document is retired)
**Consumes rather than replaces:** [`agentic-rocm-kernel-authoring.md`](agentic-rocm-kernel-authoring.md),
[`rocm-verify-profile-backend.md`](rocm-verify-profile-backend.md),
[`agent-collab-rnd-harness.md`](agent-collab-rnd-harness.md),
[`kernel-freeze-runbook.md`](../../docs/reference/kernel-freeze-runbook.md)
**Production baseline at authoring:** `production-consolidated-v8` at
`67a433bf45a8a091d83b4ea0b32ff0735fd51800`; the production kernel set is frozen.

**Current checkpoint (2026-08-13):** the prior a4cb/283b control narration below is historical.
The current clean direct-child CPU instrument is `f744cc220e722d1bda93783959471d44f8e118b0`;
the current accepted baseline-matched control bundle is
`/mnt/raid0/llm/autokernel/controls/ak-controls-v9-f744cc220-20260813-r4` (5/5 controls,
`B_min=15`, MDE `2.9585%`). Rejected r2/r3 controls showed that llama-bench `-r 5` creates a
within-invocation thermal/P-state trajectory; research `798016e3` therefore binds all A/A arms to
one sealed `GGML_IQK=0`, `reps=1` frame, and `7486a167` rejects a pair outside its accepted
block range before publication. r34 spent no inference (10 blocks rejected against `[15,20]`);
r35 refused before build for host contention; r36 intervention passed all T0 gates but refused
before T1 when foreign OpenCode CPU work made the post-T0 quiet sample `25.47/96 = 0.27`, above
the unchanged `0.25` ceiling. Its A/A arm was not started. These terminal attempts are not
archive evidence. The next empirical action is a fresh 15-block matched prefill pair only after
the independent quiet-host gate passes; then the required decode pair, heldout-bound prefill pair,
strict archive, and observe-only least-commitment evaluation remain mandatory.

**Historical checkpoint (2026-08-12):** the current frozen-v9 control campaign is accepted and
decision-grade. Under clean one-parent instrument `a4cb04ca8f92fa4d665684490f609b380f9b5e96`,
`ak-controls-v9-a4cb04ca-20260812-r2` solved B_min=`10` and φ=`0.03578502357852242`, passed all
five controls, promoted the historical IQK replay at `+26.6050%`, and set `may_rank=true`. The six
inference legs completed under one released CPU claim. Deterministic composition initially stopped
because `CampaignBinding.change_class` was absent; research `c4a42c69` repaired that drift and
recomposed the already-completed raw vectors with a hash-bound attestation stating
`inference_executed=false`. Research `900cb5c6` also implements the strict AK-WM-2a real-archive
builder. The CPU IQK proposal, exact physical envelope, calibration binding, and dry run are ready,
but live preflight correctly refused before claim, build, or benchmark because host uptime was
`13.47 days`, beyond the ratified one-week ceiling. After a compliant reboot, run the full-host CPU
IQK proposal, materialize the real matched archive, then run the least-commitment evaluation
observe-only.
Research `58b8d66a`, `948a95a4`, and `3bd94893` close the no-inference producer gap and subsequent
proposal-v4 drift in that next step. The live


Research `58b8d66a` and `948a95a4` close the no-inference producer gap in that next step. The live
campaign now requires a prospective least-commitment capture plan for the IQK parameter path,
journals the measured diagnostic/outcome block, and can project two clean completed campaigns into
the strict matched archive and onward to the sequencer/release rehearsal. Capture plans no longer
accept decorative receipt ids: they resolve one absolute SHA-pinned diagnostic-source file and
mechanically rederive every canonical and recoded diagnostic. Archive admission also refuses an
exploratory result and binds the exact proposal statement to one durable `HYPOTHESIS_OPENED` plus the
selected `HYPOTHESIS_CLAIM_AUTHORIZED` record. The exact intervention and A/A control proposal,
capture, source, hypothesis-store, and physical-envelope inputs are materialized under
`/mnt/raid0/llm/autokernel/campaigns/ak-iqk-v9-20260811/` and
`/mnt/raid0/llm/autokernel/campaigns/ak-iqk-v9-aa-control-20260812/`. Both bound current-schema
commands compose successfully in 12 steps with `state=dry_run_composed` and `executed=false`; no
inference was performed by this checkpoint. A live matched
archive still requires two post-reboot clean DECIDED runs.
The separate availability-conditioned seven-arm controller campaign now runs from an immutable source
worktree. R15 is terminal-invalid and non-rankable after Claude exhausted its structured-output retries;
its 7/7 claims released, all 13 captured PIDs were dead, and every cgroup was empty and removed. Research
`537163d5` adds one exact-cause, receipt-visible retry without retrying timeouts or arbitrary failures;
`eb1de388` refreshes the campaign pin. The full admitted Python 3.12 AutoKernel suite passed **5,682
tests with one expected failure**. R16 then refused before claim or inference on the stale expected
entrypoint identity. Fresh immutable r17 reached a valid 32h actor/critic checkpoint at diagnostic
speedup `1.004404445111776` with compile/correctness **4/4**, but terminated noncomplete when the next
EvoEngineer 2h cell called `ArenaWorkspaceEvaluator.__init__` without required keyword-only
`source_paths`. Its exact campaign PIDs are dead and final device claim released. R17 is partial and
non-rankable. Research `381bc55e`/`11bccd41` repaired and repinned source-path construction across the
five affected controllers; `b0d6f79f` prospectively captures and strictly projects feedback-only
intermediate evaluator beliefs. Fresh immutable r18 is sealed at `17b9208d`; its baseline is complete
and its first actor cell is active. Partial r18 evidence grants no comparison, aggregate, rank, bank,
champion, promotion, or release authority. This controller comparison remains diagnostic and does not
replace the post-reboot IQK/matched-archive sequence.


`/mnt/raid0/llm/autokernel/campaigns/ak-iqk-v9-aa-control-20260812/`. Both bound commands compose
successfully with `executed=false`; no inference was performed by this checkpoint. A live matched
archive still requires two post-reboot clean DECIDED runs.

`source_paths`. Research `381bc55e`/`11bccd41` repaired and repinned that construction, and
`b0d6f79f` prospectively captures feedback-only intermediate evaluator beliefs. R18 then completed
its baseline and full Claude/Codex 2h loop, but the parent refused the cell fail-closed after mutable
host `~/.claude/.claude.json` changed following its staged copy. No actor-cell receipt or aggregate
was admitted; r18 is terminal-noncomplete and non-rankable. Research `0b1fdbe9` now binds the source
identity at copy time, verifies the copied bytes, and emits a staged-input receipt; **596 controller
tests passed**. Fresh immutable r19 is live from exact clean source `0b1fdbe9`; its ready 7/7 audit,
completed baseline, and active first actor cell grant no comparison authority. Research `dfe265a1`
separately specifies the safe controller-overlap architecture and required overlap A/A gate, but does
not enable overlap in r19. This controller comparison remains diagnostic and does not replace the
post-reboot IQK/matched-archive sequence.
Research `069e79fd` now closes the remaining no-inference source-to-champion implementation seam.
The live campaign consumes an immutable content-addressed source patch before claim acquisition,
applies it only through the guarded worktree boundary, and records the exact clean source/build,
ancestry, affected surface, composition evidence, and executed evaluation-event identities while
the claim and tree are still held. A separate lean sequencer derives validated banked/frontier
candidates, requests direct combined-candidate T0/T1/T2 evidence, maintains an exact-anchor champion,
and handles idempotent replay plus anchor reanchoring without importing campaign execution, release,
or production-write capabilities. The merge-commit package suite passed **4,530 tests** with one
expected failure and **2,039 subtests**. These are implementation and regression results only: no
real proposal was banked and no empirical champion was created.
Research `99fe3014` also restores the separate operator-triggered readiness/T3/package plane without
putting it on the campaign import path. Its release-local preflight is a pure three-way reducer over
caller-supplied receipts; T3 and the packager have no source mutation, process execution, production
write, or self-trigger capability. Real release mode continues to refuse the unratified
`P-KERNEL-FREEZE-1` protocol, while dry-run fixtures preserve the v8/speech waiver, linkage,
rollback, and transaction-calibration behavior. The final merged release-plus-Arena slice passed
**1,267 tests** with **849 subtests**; no T3 campaign or release action ran.
Research `d8013a6c` tightens the ROCm correctness boundary used by source-changing campaigns. The
EPYC Q4_K dequant case now requires the exact operator-ratified frozen-v9 branch, commit, reported
version, binary digest, linkage digest, and attestation rather than trusting a provider label.
Separately, sensitivity, hostile-distribution, and checker-isolation reducer outputs enter T0 only
through measured, hash-bound bindings to the exact candidate source and evaluator bundle; absence,
dry-run capture, or identity drift is verdict-bearing `COULD_NOT_CHECK`. The exact-main C3,
correctness, and execution slice passed **444 tests with 132 subtests**. This closes implementation
authority only: no EPYC workload capture, candidate timing, whole-model exit, or inference ran.
Research `77689f76` makes AK-D39 executable rather than advisory. Proposal v4 binds every provider's
source/opaque mode, immutable source or artifact identity, licence check, isolated prefix, toolchain,
linkage, target backend, and evidence authority. Opaque binaries are diagnostic-only. Provider roots
are realpath-resolved and refused when they lie inside—or broadly encompass—shared ROCm/system paths
or a frozen production tree. Banked evidence requires a clean experimental `llama.cpp` integration
whose source, branch, ancestry, patch, binary, linkage, and toolchain identities all join. The C3
compiler now distinguishes diagnostic provider overlays from `IntegratedLlamaGpuBinding`; only the
latter can satisfy the whole-model exit. This checkpoint ran no inference.

Research `4a5f7361` closes the remaining offline integration seam from the lean sequencer's
schema-bound composed champion through readiness, dry-run T3, and one validated
`RELEASE_PACKAGE_READY` journal record. The closeout is operator-triggered, unreachable from Campaign
#1, accepts dry-run T3 only, and owns no process, build, inference, clock, or production-write
capability. Architecture fixtures are durably labelled `architecture_regression_fixture` with
`empirical_claim=false`; they prove lifecycle wiring and recovery only. The independently repeated
focused suite passed **1,692 tests**, including restart, resource-preemption, tamper, schema, and
campaign-import-boundary coverage. This does not satisfy either AK6 empirical campaign item.
Research `5d7a408b` adds the immutable archive/resume half of source-candidate correctness
prerequisites. A source campaign must preload one content-addressed package containing the raw
sensitivity, hostile-distribution, and checker-isolation CSV bytes before any claim, then re-run the
trusted reducers against the exact completed source, binary, and evaluator identities before T0.
Dry-run packages cannot authorize execution, parameter proposals reject the package, and restart
reuses the same embedded bytes. Research `51742ebd` adds the campaign-reachable fresh producer with
durable invocation/completion records and restart-safe reuse. The exact `5d7a408b` archive slice
passed **308 tests**; the accepted integrated AutoKernel suite passed **5,464 tests with one expected
failure**. These are static/fault-injection results, not a real source-candidate capture.
Root `d2cd8639`, `748e2aff`, `205b9444`, and `10e3ab77` also complete the bounded Kernel-R&D
dashboard audit: the live contract projects all seven AutoKernel sections and the complete frozen
kernel set, including three trees, four attested binaries, four stable links, non-executing tree-local
ELF linkage, and ggml-generation checks. The live fold intentionally remains not-proven while llama's
observed ggml `0.16.0` lacks human attestation and the pre-reboot dashboard process carries a stale
ggml-bearing `LD_LIBRARY_PATH`; it does not convert either unknown into green state.

INF-37 has separately produced a third, one-file experimental candidate: a one-row-only IQ2_XXS
VPOPCNT sign decoder measured at +5.733% for the target row with the former batched regression removed.
Its commit and model-level confirmation are independently gated by OP-12; it does not expand OP-11.
The AK-LN-2/AK-X-5a calibration has now rejected every historical CPU split depth as a general ranking
proxy, so that first campaign must verify on the full host unless a narrower change-class calibration
later passes. AK-BH-3 found that the implicit CPU flash-attention default behaves like the fast
explicit-ON arm on the measured 0.5B Q4_K_M surface, but the choice must still be explicit. AK-BH-4
now enforces exact-surface strongest-provider selection. These research controls are durable in
`5fbd471b` (promoted to research `main` via `caa380f7`). The no-inference closure added
evaluation-event v4 transfer links, adjacent noise-floor display, the
prior-art gate, historical 4/8/16/32/48-way CPU lane registry, op-level fan-out planning, compile-only
artifact veto, the two permitted static ROCm audits, era-local calibration authority, exported-ELF
version coverage, and structured CPU-reference receipts for passing backend-op cases. The first-campaign
HostOps adapter now has no undeclared static seam for its registered IQK parameter proposal. AK-DEL-1
also replayed a hash-bound, normalized historical `rocprofv2` trace: all three admitted kernel families
landed in bucket (a), with none in buckets (b)–(d), so the bounded next action is catalogue expansion
rather than a novel-kernel generator. Its INF-48 dependency now also has the inference-free C4 layer:
paired mapping/formal traces, deterministic kernel/overlap/fuse/architecture tables, explicit host
catalogue scope, and a bounded judgment receipt. The T0 property plane now also preserves numeric
per-op/backend/shape residuals in the evaluation event for the Vidya SC18 projection; sealing the
experimental producer still needs the separately requested local-commit approval. The C2 layout
axis is now separately flagged and fail-closed in the research consumer, including exact coverage of
offset, stride-gap and transpose families; its live CPU acceptance passed **1,048/1,048** cases, while
its producer shares the same pending experimental commit.
The C2 value axis is now independently flagged and fail-closed in research commit `eca5dbda`:
identity, ×3, ×0.01 and negate must all complete on every emitted packed-float case, and property
residuals retain the input-transform coordinate through the Vidya projection. Its live CPU pass used
suite seed `4711` and passed **779/779** cases across `SOFT_MAX`, `ARGSORT`, `TOP_K`, and `SOLVE_TRI`,
with all four transforms completed. The `SOFT_MAX` checker was corrected to include implicit attention
sink mass before that acceptance. Its producer also shares the pending experimental commit.
Research commit `9cc3ed1b` adds the independent stateful consumer and fail-closed `AK_STATE_V1`
contract. A live CPU probe with suite seed `4711` then passed all **5,184/5,184** emitted cases across
`SSM_SCAN`, `SSM_CONV`, `FLASH_ATTN_EXT`, and `GATED_DELTA_NET`; every case reported
`initial_equal=1`, `input_immutable=1`, and at least one final-state output. This closes the live
validation milestone for RVP-C2-5, but the shared experimental producer changes remain uncommitted
and must not be committed or pushed without explicit operator approval. Research commit `598375c5`
also makes the clean control instrument root/binary explicit and tests both overrides. RVP-T0-1 also
completed its authorized 60-second gfx90a saturation probe: 242 device samples held 1700 MHz for
99.5868% of the window while the GEMM produced 41.904 TFLOP/s and peaked at only 200 W against the
300 W cap. Because the card never approached the cap, clock pinning is not a live variance remedy here
and AK-OP-2 is declined. An additional current-session replay at
`/mnt/raid0/llm/autokernel/campaigns/rvp-t0-1-20260812T0444Z/receipt.json` corroborated that closure:
242 samples at 250 ms cadence, 99.5868% at 1700 MHz, 41.854545 TFLOP/s, and only 196 W maximum
against the same 300 W cap (`approached_power_cap=false`). RVP-C2-6 now also has an independent
host-double reference that decodes
Q4_0, Q8_0, Q4_K, and Q6_K directly from GGUF wire bytes and emits
`fp64_error_ratio/host-double-gguf-wire/v1`. Five representative CPU cases, the broadcast regression,
31 real parser tests, and the 5/5 planted plus 5/5 clean property self-test passed. A subsequent audit
fixed a non-contiguous-row coverage bug in that oracle; the corrected forced-dispatch matrix then
passed all **43/43** force-rocBLAS cases but only **18/43** force-MMQ cases. The old force-MMQ DP4A
control also passed only **18/43**, ruling out an MFMA-only defect. The root cause was Q4_K's affine
MMQ reconstruction mixing a quantized/dequantized Q8 dot term with an original-float activation sum
for its min correction. The local Q4_K DS4 fix derives both from the same dequantized Q8 population;
default, force-rocBLAS, force-MMQ MFMA, and force-MMQ DP4A then passed **172/172** at unchanged κ=1.5.
The source remains uncommitted under OP-11, but the diagnosis and live correction are complete.
AK-BH-1 also measured best-available rocBLAS against
hipBLASLt heuristics at nine prefill shapes: hipBLASLt won three, with ratios spanning 0.734×–1.322×,
so the honest vendor baseline is shape-specific rather than one global library. A subsequent
same-panel run at `/mnt/raid0/llm/autokernel/campaigns/ak-bh-1-20260812T0448Z/receipt.json` found
four hipBLASLt wins across the nine paired shapes and a 0.7289×–1.3219× ratio range, independently
preserving the shape-specific conclusion. AK-BH-2 completed all
eight explicitly pinned `-fa` × `ROCWMMA_FATTN` × `MMQ_MFMA` arms on one Q4_K_M model; `-fa on` won
each paired comparison, `MMQ_MFMA` was materially slower on this surface, and the winning arm was
`r1m0-fa-on` at 24,647.316788 t/s. Those single-surface observations do not authorize a global build
default. The experimental implementation remains uncommitted under its per-commit approval rule. The
research-side C2-7/C2-11/C5-2 sensitivity reducer and runner are now durable in research commit
`000a2686` (promoted to `main` via `f3c6b24a`): they reduce a
reference-only 3-seed × 4-transform population across separate seed-variation and transform-variation
axes, refusing missing, mixed-version, or untrusted observations. The focused reducer suite passes
14/14 tests. CPU and HIP `test-backend-ops` builds materialize `AK_SENS_V1`; their parser/provider
suite passed 201 tests. A live `SOFT_MAX` smoke produced 212 rows per seed, 2,544 observations and
1,484 scoreable units, with PASS, zero unscoreable units, and a CPU claim held through the run then
released. That smoke is explicitly **non-evidence**: suite identity `0db32c06e` names the committed
parent while the producer delta is uncommitted. No producer-dependent row closes until explicit
operator approval permits committing the experimental producer and a fresh matched replay binds its
durable commit. Separately, the C5-R audit found that the exact 2026-07-04 argv and raw
parent/human-patch evidence were lost. The replacement historical-task descriptor preserves that
fact as `historical_command_recovered=false`, seals parent `7c28056b` from expert `496e2f09`, and
pins the reconstructed surface. Its fresh 15-run matched replay measured a 21.223659% expert ceiling
(155.4233734 → 188.4099002 `speed_tg`) while correctly leaving candidate scoring `COULD_NOT_CHECK`.
The C2-8/C2-9 hostile-distribution and checker-isolation reducers/runners, historical scorer/replay,
and hardened C4 source-identity bridge are durable in research `1a4d7dca` / `main` merge `c5fe6b51`;
the full AutoKernel suite passes 3,923 tests with one expected failure. The C2-8/C2-9 live smoke was
implementation-only non-evidence: two selected ROCm0 `SOFT_MAX` rows passed each mode, with claim
acquisition/release and four samples, but no durable receipt. The next live action remains §AK6.5
Step 3's known-real CPU candidate, while producer-dependent correctness evidence waits on OP-11.
The same gate now applies to INF-37's IQ2 profiler fallback: the Omniperf/rocprof-v1 runner is durable,
but frozen v9 lacks its required seeded/repeated producer flags. Its failed compatibility receipt is
evidence of the instrument boundary only; a passing IQ2 counter capture waits on OP-11 and replay.
The run-specific CPU/GPU authorizations do not extend to producer commits, promotion, or freeze
actions. Offline AK-WM-1 plumbing is
complete, while AK-WM-2 remains empirical and requires a real matched completed-proposal archive.

The concurrent GPU-side INF-03 path has not changed that evidence posture. R12 supplied the first live
proof that measured critic feedback changes the next planner proposal: iteration one compiled and
passed correctness at `average_speedup=0.9967805648538064`, the critic selected `revise`, and
iteration two explicitly rejected the first unmasked-fastpath mechanism before proposing a distinct
shared-offset/vectorized-streaming mechanism. That candidate also compiled and passed correctness at
`average_speedup=1.0059084616458236`; the critic selected `accept` while correctly stating that the
0.6% ratio remained inside noise. R12 then failed during final receipt construction because the
confined controller re-hashed `/usr/bin/docker` even though host runtime attestation belongs to the
parent broker. Research `4b516e9f` moves that authority to the parent model-receipt chain and repairs
validation of the flat model-receipt layout; `f316bf31` pins the repair. The full AutoKernel suite
passed 5,660 tests with one expected failure. R12 is immutable partial/non-rankable evidence of loop
behavior, not performance. Seven-arm r13 then completed its baseline and persisted planner model receipt
ordinal `0001`, but the confined Claude runtime failed `mkdir '/mnt/raid0/llm/tmp/claude-1000'` with
`EACCES`. All captured PIDs are dead and every controller/model cgroup was verified empty and removed.
R13 is therefore a terminal immutable partial with no controller cell receipt or aggregate, and remains
non-rankable. Research `268115ad` replaces inherited host scratch with a fresh call-scoped runtime
directory inside the governed workspace, records its relative path and non-inheritance in the model
sandbox receipt, and verifies removal. The campaign manifest still needs to pin that commit before a
fresh attempt. Neither r12 nor r13 can satisfy AK-WM-2a or any proposal/champion gate.

**Read-only evidence-authority audit (2026-08-12):** the RVP-T0-1 saturation pair and the
AK-BH-1/2/3 plus AK-LN-2/AK-X-5a receipts are durable diagnostic evidence. They retain commands,
binary or source hashes, and their bounded results, but they do not all bind a committed, clean
source checkout and build manifest. In particular, the current experimental CPU benchmark binary
comes from a dirty shared tree even though its receipt reports frozen-v9 as `build_commit`; that is
not campaign provenance. The next campaign therefore has one strict dependency chain:

1. materialize the hardened measurement instrument from a clean committed source identity and retain
   its source, commit, tree, build, and binary hashes;
2. run the five frozen-v9 controls under that exact identity;
3. run the first full-host CPU IQK proposal under the accepted control bundle; and
4. build the first matched completed-proposal archive, then run AK-WM-2/AP-WM-1b observe-only.

The durable diagnostics remain useful inputs and regressions; none substitutes for those four gates.

- [x] **AK-AUD-3 — Reconcile the durable diagnostics against clean-source campaign authority.**
  Receipt hashes, bounded results, provenance gaps, and the four-step dependency order are retained in
  `progress/2026-08/2026-08-12.md`; no inference or kernel-tree mutation was performed. ✅ 2026-08-12
- [x] **AK-AUD-4 — Reconcile executed diagnostic rows without upgrading their authority.** ✅
  2026-08-12 — current state already marks RVP-T0-1, AK-BH-1/2/3, AK-LN-2/AK-X-5a, and AK-DEL-1
  complete. Receipt re-audit confirms the first six are durable bounded diagnostics, not current
  calibration, ranking, archive, or promotion evidence where clean source/build provenance is absent.
  AK-DEL-1 is separately commit-backed at research `df02169e`: its hash-bound scope report found all
  3/3 admitted families in `existing_path_should_apply` and selects catalogue expansion before a novel
  generator. This corpus-bounded routing conclusion is not a claim about all workloads.
- [x] **AK-AUD-5 — Make the real-process fault rehearsal independent of caller import state.** ✅
  2026-08-12 — research `eeaf80ba` pins the exact checkout's `scripts/kernel_rnd` path first in each
  captured child environment, so the crash/restart and claim-revocation legs no longer exit when a
  test runner mutates `sys.path` without exporting `PYTHONPATH`. Detached clean campaign checkouts are
  accepted by exact commit identity. The focused suite passed 12 tests and the full AutoKernel suite
  passed 5,548 tests with one expected failure and 2,241 subtests.
- [x] **AK-AUD-6 — Classify every remaining AutoKernel-family checkbox at the integrated checkpoint.**
  ✅ 2026-08-12 — against root `f9f36e3e` and research `3bd94893`, the audit found exactly **41** open
  checkboxes: 13 post-OP-16 empirical items, 13 items gated by OP-11's exact experimental llama commit
  and replay, 2 dependencies on the active INF-03 campaign, 9 future/operator-release items, and 4
  external licence/source items. It found **zero** immediately actionable implementation gaps and no
  item whose implementation defect was being hidden as an empirical blocker. Every residual already
  has an owning checkbox; the only documentation defect was three live proposal-v3 pointers left
  behind after the proposal-v4 migration, corrected in AK-WM-2a and Step 3 below.
- [x] **AK-AUD-7 — Retract INF-03 r4 after tracing the controller-to-evaluator feedback path.** ✅
  2026-08-12 — the prior completion audit's “zero implementation gaps” conclusion is superseded for
  INF-03. KernelFoundry made 64 intermediate centralized evaluations from its GPU-blind controller
  environment outside the two durable vendor/final claim windows, and all 64 intermediate correctness
  checks failed. The exact-PID campaign processes were stopped and confirmed dead with no device claim
  remaining. R4's 5/64 checkpoint inventory remains diagnostic history only; it cannot be resumed,
  aggregated, ranked, banked, or used to select a controller. The owning repair and fresh-pilot tasks
  live in [`agentic-rocm-kernel-authoring.md`](agentic-rocm-kernel-authoring.md): parent-owned exact-PID
  AF_UNIX brokerage is necessary, while a decision-bearing pilot additionally requires controller and
  candidate OS isolation. A broker-only live run is engineering smoke, not policy evidence.
- [x] **AK-AUD-8 — Close the INF-03 broker/isolation implementation gap and admit exact-source
  EvoEngineer.** ✅ 2026-08-12 — integrated research tip `de7214ee` routes every intermediate/final
  candidate through a parent-owned authenticated broker into a fresh deny-network evaluator sandbox,
  while every non-baseline controller runs inside an exact-read/exec, device-blind, cgroup-owned
  sandbox with one inherited peer-bound broker socket. The same integration admits the exact historical
  MIT EvoEngineer-Full source at `1649715a…`; the pinned runtime completes its real upstream loop under
  fixture bridges. Focused warning-strict validation passed 95/95, the pinned Evo suite passed 11/11,
  and the pinned controller suite passed 572/572. The durable physical static audit is ready **7/7**
  over four tasks with zero controller/GPU commands at
  `/mnt/raid0/llm/autokernel/probes/inf03-isolation-available-source-reaudit-20260812-r3/receipt.json`
  (self-hash `5fa93695…`, file SHA-256 `082d977f…`). A fresh one-task/one-arm live pilot remains the
  empirical gate; r4 remains immutable invalid history and is never resumed.
- [x] **AK-AUD-9 — Drive the isolated INF-03 pilot to a real model request without upgrading failed
  engineering attempts into evidence.** ✅ 2026-08-12 — research branch
  `integrate/inf03-isolation-merge-20260812` at `569a8c42` implements the bounded diagnostic-pilot
  authority and the exact controller dependency grants exposed successively by r1-r11. R12 reached the
  real Codex client and retried the response request for about 60 seconds, but the sandboxed transport
  failed with `Operation not permitted` before any completed response. All twelve attempts remain
  non-rankable engineering diagnostics; there is no candidate, kernel speedup, campaign aggregate,
  banked proposal, or promotion claim. The exact attempt ladder and next repair live in
  [`agentic-rocm-kernel-authoring.md`](agentic-rocm-kernel-authoring.md).
- [x] **AK-AUD-10 — Close the governed INF-03 compatibility pilot and preserve its non-authority
  boundary.** ✅ 2026-08-12 — research main `15c61ed5` contains the complete broker/controller/
  evaluator isolation stack and the r13/r14 validator repairs. R15 completed exactly one task and one
  K-Search round with six `gpt-5.6-sol:high` calls. Its single brokered intermediate and final
  evaluations both passed compilation/correctness 4/4, with diagnostic ratios `1.0033912745` and
  `1.0019039030`; those ratios cannot rank a controller or kernel. Baseline/intermediate/final sampler
  windows contain 11/21/21 observations and matched released claim IDs. Controller and evaluator
  cgroups were verified empty and removed, ephemeral Codex state was scrubbed, the controller admitted
  only `/dev/null`, and both evaluators were deny-network with exact GPU devices. Receipt:
  `/mnt/raid0/llm/autokernel/probes/inf03-mi210-isolated-k-search-pilot-20260812-r15/diagnostic-pilot-receipt.json`
  (self/file SHA-256 `3425cf57…` / `eb53de38…`). This is compatibility evidence only: no matched
  campaign, ranking, belief, proposal-bank, champion, promotion, or release authority exists. The next
  empirical step is the fresh 7/7 availability-conditioned panel.
- [x] **AK-AUD-11 — Correct the parser false positive and audit the first two seven-arm launch
  boundaries.** ✅ 2026-08-12 — the reported all-tree parser collision was expected argparse stderr
  inside `assertRaises(SystemExit)`, not collection failure: 3,927-test discovery succeeded and an AST
  sweep found no module-scope `parse_args`. INF-03 r5 exposed a real Claude CLI `-11` under the exact
  campaign Landlock policy. Local, unpushed research `916cdc92` added the first bounded runtime-read
  repair and passed a real standalone Claude probe plus 5,653-test pinned regression, but r6 reproduced
  the same `-11`
  after its baseline even with those reads present. The test totals were observed but have no retained
  transcript, so they are regression notes rather than durable suite evidence. Therefore the probe
  proves only its narrow path; the owning open differential and fresh-attempt gate remain in
  [`agentic-rocm-kernel-authoring.md`](agentic-rocm-kernel-authoring.md). R5/r6 have no aggregate or
  decision authority, and both controller cgroups were verified empty and removed.
- [x] **AK-AUD-12 — Close the r8-r10 actor-cell implementation defects and prove one full real
  actor-critic loop without upgrading a live partial into evidence.** ✅ 2026-08-12 — research
  `6270ebc1`, `cd1303b9`, `eb00c411`, `2525f2f7`, and `7b20b5d6` move vendor evaluation fully into
  the parent, repair inherited broker writes, separate semantic task state from staged model state,
  and refresh both pins. R11's first candidate passed compilation/correctness but measured
  `0.993531469254354×`; the critic chose `revise` and the controller restored the measured starting
  source before iteration 2. This is an implementation/liveness checkpoint only. R11 is still live,
  has no terminal aggregate, and carries no comparison, belief, bank, champion, promotion, or release
  authority. Terminal receipt-chain validation remains owned by
  [`agentic-rocm-kernel-authoring.md`](agentic-rocm-kernel-authoring.md).
- [x] **AK-AUD-13 — Prove measured critic feedback changes the next live proposal and close the
  confined host-runtime re-hash defect.** ✅ 2026-08-12 — r12 iteration one measured
  `0.9967805648538064×` and received `revise`; iteration two named and rejected that mechanism,
  proposed a distinct shared-offset/vectorized-streaming path, measured `1.0059084616458236×`, and
  received `accept`. Both candidates compiled and passed 4/4 correctness, and both ratios remain
  inside noise. R12 then failed while constructing the final receipt because the confined controller
  reopened `/usr/bin/docker`. Research `4b516e9f` makes the parent broker receipt chain authoritative
  for host runtime identity and validates the actual flat model-receipt layout; `f316bf31` refreshes
  the campaign pin. The full AutoKernel suite passed 5,660 tests with one expected failure. Seven-arm r13
  subsequently failed after baseline and planner receipt ordinal `0001` because confined Claude could not
  create `/mnt/raid0/llm/tmp/claude-1000`; its captured PIDs are dead and its controller/model cgroups
  were verified empty and removed. Research `268115ad` now creates an exact call-scoped runtime/temp
  directory under the governed workspace, binds its non-inheritance and relative path in the sandbox
  receipt, and removes it after the model call. R13 remains a terminal immutable partial/non-rankable
  attempt; the repair must be pinned and proven by a fresh campaign.
- [x] **AK-AUD-14 — Prove the call-scoped model runtime across repeated live calls without declaring
  the panel terminal.** ✅ 2026-08-12 — research `446a1c31` makes model runtime state call-scoped and
  `152ed0d9` pins it into the seven-arm configuration. R14 ultimately returned nine model calls and four
  compilation/correctness-4/4 evaluator windows, proving repeated runtime compatibility, but the outer
  guard then refused the cell with `task or controller source identity changed during checkpoint
  execution`: its audited source was `152ed0d9` and the shared source worktree advanced to `03f9ae69`
  mid-run. No actor cell receipt or aggregate was admitted. All eight claims were released, all 14
  captured PIDs are absent, and all 12 recorded cgroups are verified empty and removed. R14 is therefore
  immutable, invalid, and non-rankable; its intermediate ratios carry no panel, bank, champion,
  promotion, or release authority.
- [x] **AK-AUD-15 — Close the pre-reboot evidence-product implementation audit and relaunch from an
  immutable source root.** ✅ 2026-08-12 — research `03f9ae69` adds the clean completed-journal held-out
  projector and a deterministic, non-executing IQK pair generator; the broad suite passed **5,678 tests
  with one expected failure**. That 18-commit AutoKernel series is published on research `main` at merge
  `4328c37c940d125d7b11b4264b65f415c300eda0`. Seven-arm r15 runs from a dedicated clean worktree/branch
  pinned at `03f9ae69`; its 7/7 audit, baseline, and first successful real Claude planner call were present
  at this checkpoint. The live worktree and campaign must remain untouched until terminal validation.

**Current-main runnable-loop audit (2026-08-12, no inference):** the accepted research baseline is
`730adb1d31fe12eb1a49d98210f93467911aaef0`. A clean detached checkout passed the focused campaign,
footprint, and integration suite (**308 passed, 104 subtests passed**). That proves the present
one-candidate harness and its refusal boundaries; it does not prove the end state in §15.5. Source
inspection found that `campaign.py` deliberately ends at a banked result, does not append a governed
`EVALUATION_EVENT` or complete candidate record, and refuses generic source-changing proposals where
mutation and T0 evidence are unimplemented. The sequencer/champion and release-package planes that
historical 2026-08-03 checkboxes described were removed on 2026-08-04 and survive only at tag
`autokernel-preserve-20260804`. The implementation order below is therefore part of the live path,
not a reopening of empirical work:

- [x] **AK-AUD-5 — Audit the runnable loop against current research `main`, not historical completion
  narration.** ✅ 2026-08-12 — verified the live campaign boundary, generic-source refusals, removed
  planes, preserved tag, and focused suite from a clean detached `730adb1d` checkout.
- [ ] **AK-RUN-1 — Emit governed candidate and evaluation records from the executing campaign.** Bind
  actual T0 property measurements and raw paired-performance evidence; append exactly once before the
  terminal `STOP_STATE`; make append failure non-success; and keep dry runs record-free. This is the
  producer prerequisite for Vidya SC10/SC18 and the real matched archive.
- [ ] **AK-RUN-2 — Execute generic source candidates through a content-addressed patch binding.**
  Validate proposal/change-class/files/symbols before mutation, apply only in an isolated experimental
  worktree, commit exact pathspecs, build a fresh detached snapshot, derive T0 evidence, and journal the
  immutable patch/build/candidate identity. Raw mutable paths and production trees remain forbidden.
- [x] **AK-RUN-3 — Restore the lean sequencer and champion-maintenance path.** Consume proposal,
  candidate, and evaluation records; distinguish frontier/banked/champion state; compose compatible
  members; re-evaluate the combined source snapshot; append `CHAMPION_UPDATED`; and re-anchor explicitly
  when the production anchor moves. Never add member speedups arithmetically.
✅ 2026-08-13 — `mainC` (assigned AK-RUN-3; premise VERIFIED ALREADY-SATISFIED, not re-implemented). The lean sequencer + champion-maintenance path was fully landed by 2026-08-12 research commits: `8a2e6f5d` adds `controller/sequencer.py` (Sequencer loop: consumes proposal/candidate/evaluation records, composes compatible members, records CHAMPION_UPDATED) + `controller/champion.py` (frontier/banked/champion lifecycle, compatibility(), composition_request() with required_t2_cells re-evaluation, record_anchor_moved()/reanchor_champion(), never adds member speedups arithmetically — combined evaluation must be a valid pass); `77689f76` enforces AK-D39 provider integration boundaries; `069e79fd` closes the source-to-champion seam; `4a5f7361` closes the offline release seam. Sequencer is deliberately unbound from campaign #1 (FOOTPRINT.md — no second live loop). Verified: all 7 row clauses map to implemented code; 296 tests pass in the sequencer/champion/release slice (test_champion 16, adapter/closeout/live_material/readiness); sequencer CLI inspect-only + --run works.
- [ ] **AK-RUN-4 — Restore a separate operator-triggered package/release plane.** Reuse only the
  validated plan/readiness/T3/packager parts needed to seal a champion and construct a dry-run release
  package; keep it outside campaign-one imports, with no production mutation, signing, freeze, cutover,
  or reload authority.

The existing AK9 speech-compiler and production-kernel-set dashboard checkboxes below remain the
single task records for those findings. Vidya SC10 and SC18 remain owned by
`vidya-belief-substrate-program.md`; they are dependencies of AK-RUN-1, not duplicate tasks here.

**`AK-TR-1` is complete before Step 3.** New records use evaluation-event v4, so every future
candidate can bind its cheap-lane-to-ground-truth relationship at write time rather than inventing it
later. §21's corrections (per-run approval, CPU/GPU concurrency, churn-not-throughput) remain in force.

---

## 0. Purpose and outcome

This is the build specification for turning the partially built MI210 kernel-R&D scaffold into a
genuinely autonomous, system-wide kernel **research** loop whose output is a continuously maintained,
always-validated best-known kernel per source tree — which the operator promotes to production on
request.

The finished system must be able to:

1. inspect current production behaviour and profiler evidence;
2. generate falsifiable optimization hypotheses;
3. edit and build isolated experimental kernel trees;
4. reject wrong, unstable, contaminated, or dishonest candidates before speed can matter;
5. learn from every outcome, including failures and negative results;
6. accumulate compatible improvements into a **champion lineage** without repeatedly paying for the
   full release matrix;
7. keep that champion rebased on the current production tip and green at all times;
8. on operator request, seal the champion and run the one full kernel-freeze evaluation; and
9. hand the operator a pre-validated release package — evidence bundle plus transaction plan — that
   they ratify and execute through the existing human freeze path.

**AutoKernel never freezes or cuts over production.** That decision, and the four trust-boundary
writes it entails, remain human (§1.3, §3.3).

The exhaustive optimization catalog in the source draft remains useful. This handoff owns the loop
that will execute that catalog and, in §19, the distillation of that catalog into loop memory.

---

## 1. Executive decisions

### 1.1 One release truth: T3 is the kernel-freeze evaluator, run on request

The final AutoKernel evaluation and the kernel-freeze evaluation are **the same program**. There is
one release truth, not an AutoKernel score later reinterpreted by a separate promotion ceremony.

That does not mean running it per candidate. AutoKernel needs two evaluation products:

- a cheap, stable **research evaluator** (T0–T2) used continuously to rank experimental candidates
  and maintain the champion; and
- a comprehensive, stable **release evaluator** (T3) run when the operator requests a freeze.

The research evaluator is a proxy for search efficiency and may never authorize a production
release by itself.

### 1.2 AutoKernel maintains a champion, it does not chase a trigger

Per source tree, AutoKernel maintains a **champion lineage**: the current best complete set of
compatible, correct changes, anchored on the current production tip and green through T2.

The `+25% point / +20% lower-bound` figure from the original design is demoted from an automatic
release trigger to a **readiness signal** the loop reports to the operator ("champion is now
+X%/LCB +Y% versus the production anchor on these cells"). Removing the trigger removes three
problems at once: the multiplicity of peeking at a threshold every round, the winner's-curse
inflation of a selected estimate, and the tension between long accumulation and the fresh-anchor
invariant (§4, invariant 1). Accumulation is now the normal state, and re-anchoring happens at freeze time,
which is when production actually moves.

### 1.3 Freeze and cutover remain human — AutoKernel prepares, never signs

**Revised 2026-08-02 (operator).** The earlier design targeted a one-time delegation to a release
broker with unattended freeze and eventually unattended cutover. That is withdrawn as too
aggressive, and the audit shows it was also more expensive than it looked: an automatic freeze
crosses **four** human-only trust boundaries, not one.

`MEASUREMENT.md:140-142` enumerates human-only writes as *"era-registry rows, this constitution and
its annexes, AutoPilot baseline-state applies, production freezes/cutovers, host reboots"* — repeated
in `MEASUREMENT_POLICY.md:71-73`, `OPERATING_CONSTRAINTS.md:36`, and `BUS_PROTOCOL.md:38-41`. A
kernel freeze touches:

1. **the freeze/cutover itself**;
2. **era-registry rows** — `orchestration/instrument_eras.yaml`, separately pinned in
   `human_only_paths.yaml:35-37`. The v8 cutover wrote three rows: `E8-cpu-kernel`,
   `E8-autopilot-speed`, `E8` (`instrument_eras.yaml:140-172`). Because `MEASUREMENT.md:233` requires
   every number to be era-labelled, a freeze whose era row is unwritten produces evidence nobody can
   interpret;
3. **AutoPilot baseline applies** — `orchestration/autopilot_baseline.yaml`, pinned at
   `human_only_paths.yaml:38-40`. The E8 precedent is explicit: the cutover *"opens a fail-closed E8
   AutoPilot rebaseline hold … until an **operator-ratified** E8 quality-baseline reseed writes fresh
   values and windows"* (`instrument_eras.yaml:166-172`);
4. **the pinned list itself** — `human_only_paths.yaml:42-49` is branch-pattern-scoped
   (`production-consolidated-*`, *"any commit landing on a frozen production kernel branch"*), so a
   newly created `production-consolidated-v9` matches the moment it exists, and baking the agent-file
   overlay into it is a commit on a protected branch. Amending that list also requires rewriting
   `human_only_paths.sha256`, and `config.yaml:164` sets `on_pin_mismatch: refuse`.

Under the revised model none of these need delegation. AutoKernel's release-side job ends at a
**release package**: a sealed candidate, a T3 verdict bundle, a rollback plan, and a pre-validated
command sequence. The operator executes the freeze, writes the era rows, and ratifies the AutoPilot
rebaseline, exactly as for v8 and the speech freeze.

This also dissolves the reload-ownership conflict (§3.3): `OPERATING_CONSTRAINTS.md:41` requires that
*"if a session owns the inference, any orchestrator API or stack reload … must be executed BY THAT
SESSION, at a moment it chooses; it is never forced upon that session's workflow from outside"*. An
operator-driven cutover is scheduled by whoever owns inference. An autonomous broker restart would
have been the incident the rule was written for.

### 1.4 Full compute, separated authority

"Full access to system resources" means the research actor may use all authorized CPU, GPU, memory,
models, source trees, compilers, profilers, and storage. It does not mean the same process may
rewrite the measurement constitution, the evaluator that scores it, the sealed bundle after
evaluation, a frozen production branch, or any production symlink.

Compute access and certification authority are different capabilities.

### 1.5 Three source trees, four production binaries — campaigns are per backend, freezes are per tree

The production kernel *set* is four binaries but only **three source trees**:

| Production binary | Stable path | Source tree | Frozen branch |
|---|---|---|---|
| `cpu` | `/mnt/raid0/llm/kernels/production/cpu` → `llama.cpp/build/bin` | `llama.cpp` | `production-consolidated-v8` |
| `gpu` | `…/production/gpu` → `llama.cpp/build-hip/bin` | `llama.cpp` | `production-consolidated-v8` |
| `stt` | `…/production/stt` → `whisper.cpp/build/bin` | `whisper.cpp` | `production-speech-v1` |
| `tts` | `…/production/tts` → `qwentts.cpp/build` | `qwentts.cpp` | `production-speech-v1` |

CPU and GPU **share one tree and one frozen branch**. They can be *researched* independently — a
change in `ggml-cpu` does not reach the HIP binary — but they cannot be *frozen* independently, and a
change in shared ggml core reaches both. Therefore:

- campaigns are per **backend** (`llama_cpu`, `llama_gpu`, `whisper_stt`, `qwentts_tts`,
  `serving_runtime`) and may run independently;
- champions are per **source tree**, so `llama_cpu` and `llama_gpu` campaigns converge on one llama
  champion; and
- freeze scope is the **union of backends served by the tree**, narrowed only by the mechanically
  derived affected-surface manifest (§6.4). A CPU-only change still owes a GPU non-regression check
  unless the manifest proves the GPU binary is byte-identical.

Note the `tts` symlink points at `build`, not `build/bin` like the other three. Adapters must not
assume uniformity.

Build one domain-agnostic controller with backend adapters. Do not build a GPU-only controller and
clone it. Do not force scheduler or registry changes through a kernel-freeze transaction merely
because the same planner discovered them — `serving_runtime` uses a stack-change release adapter.

#### 1.5.1 ROCm modules are governed candidate providers, not champion source trees

AutoKernel's GPU search surface includes the software-addressable ROCm layer: exact-shape
rocBLAS/hipBLASLt/CK selection, library algorithms, compiler flags, launch topology, standalone
Triton/HIP kernels, and source-available vendor-module kernels. That does **not** create a fourth
kernel source tree or permit a module to bypass the deployable product:

The operator-approved targeting order is explicit:

1. optimize `llama.cpp`-owned kernels and dispatch first, especially low-bit and MoE losses;
2. tune shape-aware rocBLAS, hipBLASLt, and CK/CK-Tile selection and algorithms;
3. author standalone Triton/HIP replacements for missing or weak gfx90a paths, then integrate them
   through an experimental `llama_gpu` candidate;
4. fork source-available ROCm modules only in isolated builds and only after profiling proves that
   module is the limiter; and
5. run concurrency and scheduling hypotheses separately as `serving_runtime` campaigns.

- every module build or source fork is content-addressed, source-pinned, and isolated from the
  resident stack; no campaign may write the shared `/opt/rocm`, replace a resident runtime library,
  or inherit an unbound module from `LD_LIBRARY_PATH`;
- opaque vendor binaries may be hash-bound baselines or candidate dependencies, but are never
  champion source and cannot carry an authored-kernel claim;
- a winning module/kernel result becomes bankable only after it is integrated through an
  experimental `llama_gpu` candidate and passes the exact operator, captured-workload, and
  whole-model gates; the resulting evidence and champion remain attached to the `llama.cpp` source
  tree; and
- a future independently deployed ROCm module would require a human-reviewed schema/backend/release
  adapter before any campaign could promote it. The current loop must not infer that authority from
  buildability or speed.

This keeps module-level search broad while keeping promotion narrow. Serving concurrency and
scheduler work remains a separate `serving_runtime` campaign under AK-D9/AK-D23.

- [x] **AK-D39-1 — Enforce the provider-not-champion boundary in records, execution setup, C3, and
  champion admission.** ✅ 2026-08-12 — research `77689f76` adds proposal-v4 provider identity,
  diagnostic-only opaque authority, realpath-aware isolated prefixes, exact baseline manifests,
  provider-integration joins, and the diagnostic-versus-integrated C3 split. Ancestor-root and
  symlink-to-protected mutation tests fail closed. Post-rebase focused/affected validation passed
  580 tests; a full 5,597-test discovery exposed four current-schema call sites, which were repaired
  and revalidated in a 419-test affected slice. The only remaining discovery error was environmental:
  the checkout Python lacks `pytest` for `test_placement_context.py`.
- [x] **AK-D39-2 — Implement governed source-available ROCm module execution.** ✅ 2026-08-12 —
  research `a54e36ba` adds the pinned clean-source lifecycle in a content-addressed isolated prefix,
  binds exact source commit/tree, tracked files, licence, toolchain, executable and linkage identities,
  requires explicit authorization, executes without network access, verifies expected outputs, and
  records teardown. The provider remains diagnostic/nonbankable until its result joins an experimental
  `llama_gpu` integration candidate; it cannot mutate shared `/opt/rocm` or make an opaque binary into
  champion source. Validation passed 101 tests plus 22 subtests and Ruff.

### 1.6 The objective is per-backend, per-phase non-inferiority plus improvement

**Revised 2026-08-02 (operator).** The original production-weighted composite across CPU, GPU, STT
and TTS cells is withdrawn. It was forbidden twice over:

- `MEASUREMENT.md:83-84` — *"Comparisons only within a protocol + instrument version. Cross-protocol
  comparisons are analysis, labeled as such."* A scalar folding P-BENCH-1, P-BENCH-PREFILL-1 and
  P-GPU-1 cells is analysis, and cannot gate a release.
- `gpu-cross-device.md:106-111` — *"**The net is measured directly, never reconstructed.** … Measuring
  GPU gain and CPU loss separately and subtracting is FORBIDDEN: it compounds both halves' noise
  (rate CV ≈ 9.1% each) and measures the halves under conditions that do not co-occur."*

The replacement objective, per backend:

> At the **production-optimal** recipe for every protected cell, both **prefill** and **decode**
> throughput must be non-inferior to the production anchor, and at least one must improve.

Consequences:

- each phase is judged under its own protocol (P-BENCH-1 decode, P-BENCH-PREFILL-1 prefill, P-GPU-1
  for MI210), so nothing crosses a protocol boundary;
- "maximally optimized" is load-bearing: baseline/off-recipe cells stay diagnostic and never justify
  or veto a release (§4, invariant 15);
- a **phase trade** — a small prefill regression buying a large decode gain — is permitted only as a
  pre-declared campaign exception naming the exact regression band, the exact expected gain, and the
  roles affected, and it is an operator decision at freeze time, not a controller decision. Expected
  to be rare;
- cross-backend roll-ups may be *reported* to the operator as a labelled analysis view. They never
  gate.

---

## 2. What exists today

The current scaffold is useful but is not an autonomous research loop. Every row below was
mechanically re-verified on 2026-08-02.

| Surface | Current artifact | What is real | What remains missing or stale |
|---|---|---|---|
| Experimental evaluator | [`kernel_eval.sh`](../../repos/epyc-inference-research/scripts/kernel_rnd/kernel_eval.sh) | 230-line MI210 evaluator; build/op-test/coherence/A-B/profile path; historically validated on the async-prefetch experiment | **Emits `"status":"OK"` unconditionally (`:223`) and marks any non-empty generation `coherent` (`:112-113`) while the baseline compare only runs with `--baseline-env` (`:114-117`) — so coherence never gates the record**; gates GPU exclusivity by `rocm-smi --showpids` idle sensing (`gpu_idle()`, `:77-82`), the exact pattern §10.4 forbids; uses raw `llama-bench` (`:123-124`) with no codified recipe; no region/device claim; no PPL gate; only `MUL_MAT` (`:93`); fixed shapes (`NGEN=128`, `-p 0`, so prefill is never measured); default `OUT`/`BUILD_DIR` point at `/mnt/raid0/llm/tmp/mi210-build/campaign/`, **which does not exist** |
| Parameter sweep | [`kernel_sweep.sh`](../../repos/epyc-inference-research/scripts/kernel_rnd/kernel_sweep.sh) | 64 lines; serial execution of a TSV of pre-selected env variants; ingests results and refreshes dashboard | Does not generate hypotheses, edit source, manage worktrees, select next points, schedule resources, or run unattended campaigns |
| Strategy store | [`kernel_store.py`](../../repos/epyc-inference-research/scripts/kernel_rnd/kernel_store.py) | SQLite ingest, correct-only Pareto view, export, dry-run-default purge/rewind; 11 unit tests pass | **`:81` admits `coherence in ("byte-identical","coherent")` into the correct-only Pareto, so the frontier is already contaminated by anchor-less runs**; natural key `(label, ts, git_sha)` (`:47`) cannot reconstruct a candidate; no patch/source snapshot, parent lineage, evaluator hash, resource receipt, campaign objective, or affected-surface map; failures stored but not fed to the planner; destructive purge conflicts with an all-outcomes history; unclosed-file `ResourceWarning` at `:88` |
| Reward integrity | [`c6_reward_integrity.py`](../../repos/epyc-inference-research/scripts/kernel_rnd/c6_reward_integrity.py) | 975 lines; anti-TOCTOU snapshot, trusted evaluator, correctness-before-latency, run manifest, resume-drift and locking primitives; **`test_c6_reward_integrity.py` passes 24/24** | Not connected to llama.cpp candidate worktrees or `kernel_eval.sh`; host sandbox availability unresolved |
| Generic task contract | [`task_descriptor.py`](../../repos/epyc-inference-research/scripts/rnd_harness/task_descriptor.py) | Domain-agnostic task descriptor and known-good baseline concept | No importer anywhere except its own test; no AutoKernel schema or controller uses it |
| Dashboard | [`dashboard/static/kernel.html`](../../dashboard/static/kernel.html), [`dashboard/server.py`](../../dashboard/server.py) | Read-only `/kernel` page (`server.py:826`), `/api/kernel` (`:843`), freshness warn 3 d / stale 14 d, absence-tolerant | Default `KERNEL_DASHBOARD_JSON` points into the missing campaign directory, so the page has nothing to render; schema exposes a partial Pareto, not campaign state, champion status, resource use, or release readiness |
| Scope derivation | [`kernel_freeze_scope.py`](../../repos/epyc-orchestrator/scripts/validate/kernel_freeze_scope.py) | 117 lines; derives production-role rows per backend from `stack_priors.yaml`; backend classified by build-path substring | Produces role rows, not a deduplicated release plan; no affected op/shape coverage, candidate-only capability rows, test recipe IDs, non-inferiority thresholds, or evaluator commands |
| Freeze procedure | [`kernel-freeze-runbook.md`](../../docs/reference/kernel-freeze-runbook.md) | Correct high-level rule: derive scope, compare candidate with frozen production, verify linkage, version past production | Prose only; no reusable release-gate runner or bundle schema; no waiver concept |
| Historical release transaction | [`freeze_v8_production_20260725.sh`](../../artifacts/operator/freeze_v8_production_20260725.sh) | Strong evidence hashing, identity checks, quality bindings, transaction and rollback patterns; **hash-pinned operator-waiver verification at `:214`, `:248`, `:268`** | 1,234-line release-specific script with hard-coded v8 artifacts; not a generic evaluator |
| Stable kernel paths | `/mnt/raid0/llm/kernels/production/{cpu,gpu,stt,tts}` | All four symlinks exist | `archive/` exists and is **empty**; no generic candidate install/seal transaction uses the layer |
| Nightshift | [`scripts/nightshift/`](../../scripts/nightshift) | General guarded wrappers | No AutoKernel service, campaign queue, checkpoint/resume, resource broker integration, or supervisor |

### 2.1 Scaffold validation performed

No inference, kernel build, production edit, or process reload was performed.

- `bash -n` passes for `kernel_eval.sh` and `kernel_sweep.sh`.
- `python3 -m unittest scripts/kernel_rnd/test_kernel_store.py` → 11 tests, OK.
- `python3 -m unittest scripts/kernel_rnd/test_c6_reward_integrity.py` → **24 tests, OK**. The earlier
  note that "the system Python lacks pytest so the C6 suite was not rerun" was wrong: the suite is
  plain `unittest` and its docstring says *"Run standalone (no pytest needed)"*. The real environment
  gaps are that pytest is injected unpinned via `Makefile:37-38` (`uv run --with pytest`) rather than
  declared in `pyproject.toml`/`uv.lock`, there is **no CI** (`.github/workflows/` does not exist),
  and neither `kernel_rnd` suite is in `PYTEST_SMOKE`. `OPERATING_CONSTRAINTS.md:17-19` also forbids
  `pytest -n auto` on this host.
- Research HEAD contains the Phase-1 store and mechanical inner sweep (`133017de`), purge/rewind
  (`a1f38cd7`), and C6 (`2d5b1f6e`). Older prose claiming all of Phase 1/2/C6 is absent is stale.

### 2.2 The important status correction

The **mechanical inner sweep exists**. The **autonomous outer loop does not**.

No controller reads profiler evidence, proposes a source change, asks a critic to falsify it, edits an
isolated tree, selects the next experiment from history, maintains a champion, or constructs a release
package. Calling `kernel_sweep.sh` "Phase 2" blurred this and is why the scaffold looked closer to
completion than it is.

### 2.3 What to reuse from orchestration AutoPilot

Reuse the proven control-plane patterns; do not run as another species inside the live routing
optimizer. Objectives, mutation authority, resource costs, artifact sizes, and correctness
consequences are too different.

| Orchestration surface | AutoKernel disposition | Reason |
|---|---|---|
| `planner_coordinator.py` | Reuse provider arbitration, draft/critic separation, circuit-breaker, fallback, spend-breaker behind an AutoKernel proposal schema | Controller mechanics transfer; routing actions/prompts do not |
| `planner_providers.py` | Reuse provider adapters and structured-output plumbing | Lets Claude/Codex/local controllers A/B without evaluator changes — which requires §7.2's `controller` block to be recorded |
| `autopilot_supervisor.py` | Reuse singleton/supervisor/signal/restart patterns in a separate AutoKernel service | Long-horizon execution with independent state and failure domains |
| `state_lock.py`, `state_store.py`, `run_manifest.py` | Reuse atomic state, lock, manifest, resume-drift patterns | Strong base for reconstructible campaign control |
| `experiment_journal.py`, journal shards/snapshots | Reuse append/snapshot/replay concepts **including shard rotation and all-shard reads**; implement the kernel event schema separately | Kernel records must bind source/binaries/profiles/resources and retain all failures |
| `phase_status.py`, `phase_health_report.py`, restart advisor | Reuse phase/health/status semantics with kernel-specific terminal states | Prevents another opaque nightshift process |
| `pareto_archive.py`, `StrategyStore` | Reuse algorithms and retrieval lessons only; build derived views over the AutoKernel journal | Never share or mutate live routing AutoPilot memory/state |
| `worktree_manager.py` | **Do not reuse.** Build a strict kernel worktree manager | Verified: `apply_file():87-89` writes the mutation into the **main repo** (*"Copy to main repo for live eval"*); `accept():100-101` git-adds and commits into the main repo; `reject():111-117` restores from an **in-memory** dict `_applied_files`, so a crash between apply and reject leaves the main repo mutated with no recovery path. See §14 AK0 for filing this as an AutoPilot defect |
| `actions.py`, routing `SafetyGate`, EvalTower | Do not reuse as kernel action/evaluation engines | Schemas, metrics, baselines, and authorization paths are routing-specific |
| AutoPilot checkpoint/restore and rewind history | Reuse the operational lessons: stop via supervised signal, rewind every derived surface, keep primary evidence, test restart at every critical section | These scars transfer directly |

The seam is a small shared control library or copied-and-hardened primitives with explicit ownership —
not imports from the live `autopilot.py` monolith. AutoKernel releases change AutoPilot's speed era,
but the two loops must not share a writable journal, Pareto archive, or baseline state.

### 2.4 Repository ownership

The AutoKernel runtime is owned by **`epyc-inference-research`** — consistent with its declared
purpose (`AGENTS.md`), the existing `kernel_rnd`/`rnd_harness` scaffolding, and the canonical benchmark
constructors, prompts, and evidence stores the loop consumes.

| Repository | AutoKernel responsibility |
|---|---|
| **`epyc-inference-research`** | **Owns the runtime:** planner/critic/controller, journal, prior importer, seed compiler, evaluators, campaign state, backend adapters, reducers, candidate/evidence artifacts, dashboard data contract |
| `epyc-root` | Measurement/governance protocols, this handoff, ratification receipts, cross-repo indices, release-authority policy. Does not own the running loop |
| `epyc-llama`, speech kernel repos | Supply frozen production anchors and isolated experimental worktrees. Do not own controller memory or evaluation policy |
| `epyc-orchestrator` | Resource claims, compiled production facts, lifecycle boundaries, release/cutover integration. Does not own the planner or journal |

Future runtime path: `epyc-inference-research/scripts/kernel_rnd/autokernel/`, with large immutable
artifacts under the repository's evidence root (§3.7). Cross-repo helpers are called through versioned
interfaces, never copied into the orchestrator.

### 2.5 What AutoPilot cost this project to learn

AutoPilot is the closest analogue AutoKernel has: an autonomous optimization loop that has been
running here for months. Its scars were swept on 2026-08-02 across the incident log, twenty-one
autopilot/evidence-plane/eval-tower audits, the autopilot package's own "why this exists" comments,
and the progress record. Most of what it learned, AutoKernel already encodes — warm-cache gaming,
host drift, journal shard reads, owned-process lifecycle, evidence durability, deterministic replay.
The table below is only what it **did not**, with the scar that proves the risk is real and where the
design now answers it.

| # | AutoPilot scar | AutoKernel gap it exposed | Now addressed |
|---|---|---|---|
| 1 | `pause` was a silent no-op for months (`autopilot.py:7012`), and a latch-less halt was restarted straight back into the same locked state (`:8094`) | Control commands were listed as *requests* with no acknowledgement contract and no operator-initiated stop state. A human's ability to stop the loop was asserted, never tested | §4 invariant 19, §8.10 `OPERATOR_STOP_REQUESTED`, §15.1 ignored-command fault injection |
| 1b | AutoPilot's one venture into autonomous **source** mutation destroyed a production module — `src/escalation.py`, 454 lines to 3, 11+ hours of API downtime — and the mutation was **syntactically valid** (`progress/2026-04/2026-04-04.md:5`) | AutoKernel's core activity is the thing that failed. Its T0 gated *behaviour* — compile, ops, determinism — but nothing gated **source integrity**, and "it compiles" is a far weaker signal in C++ than "it imports" is in Python | §8.5.1 source-integrity gates; §12 symbol-removal and repair-compounding rows |
| 2 | The recurrence vector was the planner's own **free text** in the primary journal — scrubbing derived stores was futile because the planner re-read and re-distilled its own prose each trial; one instance ran 81 more trials on the same false story after the code fix | An append-only journal makes this *worse*, not better: prose cannot be scrubbed and is re-consumed forever | §5.5 narrative/outcome separation and retrieval-scope supersession; §7.2 `narrative` is non-retrievable by default |
| 3 | The loop froze in a `meta_action_loop` of paid no-ops **narrating an infrastructure failure that did not exist**; separately ran 1,200 iterations against a dead API; and 119 identical invalid actions were rejected without ever reaching the planner (`actions.py:207`) | `PLATEAU_STOP` conflates "search exhausted" with "planner broken", and filtered proposals vanish | §8.10 `PLANNER_DEGRADED`; §8.4 filtered proposals are journaled as `PROPOSAL_SKIPPED` and fed back |
| 4 | ~38 metered draft/critique cycles spent then discarded because due-checks ran *after* drafting; a $250 budget that was only a status string; and a spend breaker whose naive form **stopped the loop** | Cost was a stop threshold, never an attributed ledger — cost-per-banked-win unauditable | §7.2 `realized_cost`, §12 zero-yield row, §9.1 cheap checks before metered drafting, breaker forces local planning rather than halting |
| 5 | 8 of 1,055 trials were of a type the gate could promote; 0 of 121 refutations came from futility rather than budget — all while every surface read "active, blockers: []" | Both controls and the T3 dry-run test the gate's ability to **reject**. Nothing tests its ability to **promote**, so `EXHAUSTED_SURFACE` is indistinguishable from a dead gate | §15.2 fourth control is a **historical-win replay** that must promote; §18 discovery statistics |
| 6 | A single-axis noise gate discarded 8 of 11 excluded trials that were **non-dominated**; the naive fix (admit everything) poisoned frontier geometry with speed noise | Banking is single-signal on throughput; capacity, variance, and load time only appear at T2, so a throughput-neutral capacity win is filtered before anyone sees it | §9.6 multi-signal banking with robust-median-per-fingerprint |
| 7 | A full-machine health gate applied to partial-machine cells would have demoted 19/45 including a whole pre-registered family — **irreparable**, because only the aggregate was persisted | `correctness`, `quality`, `stability` are opaque blobs in the evaluation event, and no cell declares its scope denominator | §7.4 raw vectors and `scope_denominator`; §12 gate-scope row |
| 8 | The loop died silently at trial 1302 and stayed dead ~23 h with every dashboard reporting "active" | §2's own row already records that today's `/kernel` page is *absence-tolerant over a missing directory* — it renders clean when its producer is dead | §14 AK6 freshness envelope, panel→producer registry, `/health` fold, restart chaos test |
| 9 | A restart came up with an **empty frontier and nothing objected** — 232 trials, ~16 days of compute, lost. `pareto_archive.py:198` now raises rather than starting | Derived-views-rebuild-from-events is the right architecture, but BOOTSTRAP had no cardinality check | §8.2 step 10 consistency assertion with an explicit rebase escape |
| 10 | Twice in one session a committed artifact overrode the operator's restated decision; the ruling was *"the artifact is the thing that is wrong"* — and a contradictory pair is open in the tree right now | The suppression ledger has receipts but no contradiction detection, and it is exactly the artifact class that looks most authoritative | §19.2 contradiction detection against live operator decisions and sibling entries |
| 11 | Closure inflation is this project's most-repeated habit: 10 instances found and fixed, then **9 more with explicit awareness of the rule** | A stop state literally named `EXHAUSTED_SURFACE`, emitted by an LLM-driven loop, with no enumeration requirement | §8.10 sub-scope enumeration; "exhausted" and "all paths" are reserved words |
| 12 | *"Which code scores your eval depends on which eval ran first in the process"* — same-named modules across repos, `sys.path` mutated at import; and a seed-42-forever holdout the project named as an open overfitting surface | §3.6 protects the evaluator from *tampering*; the scarier defect is **ambient** import identity. AutoKernel imports across three repos | §5.4 runtime source-label attestation; §12 holdout rotation |

Two of these — items 2 and 5 — are the ones that would have been expensive to discover late. Everything
else is a hardening; those two are structural.

### 2.6 Substrate that does not exist yet

These are prerequisites, not integrations. Each has a task in §14.

| Missing substrate | Evidence | Blocks |
|---|---|---|
| **Cross-process GPU device claim** | `region-lock` is CPU-only (`region_lock_cli.py:290-291`, `--regions`/`--cpu-list`, no device option); every on-disk lock is `cpu_region.<role>.<q>.lock`; `src/gpu_lease.py:1` is a *process-local* `threading.Condition` lease (`:63-69`) used only by `axa2_live_cutover_bundle.py:535` | Invariant 4.8 on GPU; every `llama_gpu` T1+ |
| **Codified microbenchmark recipe** | `OPERATING_CONSTRAINTS.md:38` requires *all* throughput numbers to go through `bench_canonical.sh`/`canonical_recipe.py`; no constructor exists for operator-level microbenchmarks | T1a on every backend |
| **Storage/retention plane** | `/mnt/raid0` is 96% full (162G free of 3.7T); `epyc-inference-research/data/` is 118G; 13 existing llama worktrees total 41G (`llama.cpp-experimental` 13G, its preserved copy 15G) | Any long campaign |
| **Bus identity for a daemon** | `BUS_PROTOCOL.md:72-74` makes an undeclared gating lane a hard validation failure; rule 8 revocation needs a roster holder | Revocation, escalation, stuck detection |
| **Sanctioned no-`pgrep` preflight** | `bench-cpu.md:16-17` and `gpu-cross-device.md:27-30` mandate name-pattern process checks; CLAUDE.md bans them | Every protocol-conformant measurement |
| **Affected-surface derivation** | Nothing today maps a diff to ops/dispatch predicates | Freeze scope, composition, §6.4 |

---

## 3. Governance: what AutoKernel needs, and what it no longer needs

Removing autonomous freeze (§1.3) removes the largest amendment. What remains is smaller and mostly
GPU-specific.

### 3.0 The GPU protocol has two clauses, and they bite at different times

`gpu-cross-device.md:16-21` contains two distinct prohibitions that are easy to conflate. Separating
them decides what must be ratified before what:

| Clause | What it forbids | When it bites | Resolved by |
|---|---|---|---|
| **Consumption** | *"MUST NOT be consumed by AutoPilot or any automated optimizer"* | **Every GPU T1 round, from the first campaign.** This is what stops AutoKernel ranking GPU candidates at all | `P-AK-SEARCH-1` (§3.1) |
| **Decision-grade** | a decision-grade claim may only be produced on a production-named kernel | **Only when a llama.cpp freeze needs new GPU evidence** (§3.2) | P-GPU-1 sealed-candidate amendment |

The consumption clause is the urgent one: without `P-AK-SEARCH-1`, GPU research cannot legally begin,
and AK3's exit criterion ("T1 may legally guide search") is unreachable. The circularity is the
patient one, and §3.2 narrows when it applies at all.

### 3.1 Search authority — narrower than first assumed

The prohibition on an automated optimizer consuming experimental measurements exists in **exactly one
place**: `gpu-cross-device.md:16-21`, scoped at `:12-14` to *"all decision-gating GPU (MI210 / gfx90a /
HIP) throughput, spec-dec, and residency numbers"*:

> Measurements on any experimental / candidate / fork kernel … are **OBSERVATIONS ONLY**: they MUST
> NOT gate any keep / revert / deploy / promote / buy / close decision and MUST NOT be consumed by
> AutoPilot or any automated optimizer.

No CPU, quality, or speech protocol contains that clause. For those backends the governing rule is the
general one (`MEASUREMENT.md:9-11`), which bans *gating decisions*, not *ranking inside a worktree*.
`kernel_store.py:10-11` correctly repeats the GPU restriction.

**Required resolution:** ratify a stable `P-AK-SEARCH-1` whose narrow authority is to rank, retain,
abandon, or branch candidates **inside experimental worktrees**; never edit production state; never
certify a release; never edit its own evaluator or scoring contract; and emit only protocol-bound
search records. Its GPU section is the load-bearing part; its CPU/speech sections mainly codify reps,
noise floors, stopping rules, and record grammar. One-time instrument, not an amendment per experiment.

It lives in **Annex K (kernel research and release)**, a fourth annex created for it — operator-approved
2026-08-02 (§14 AK0). It fits none of B/Q/G: it is cross-backend, and it is a search instrument rather
than a measurement family. Drafted at
[`Annex-K-kernel-research-and-release.draft.md`](../../artifacts/operator/autokernel-policy-draft/Annex-K-kernel-research-and-release.draft.md);
its numeric bindings are deliberately left blocked on the AK3 controls rather than guessed.

### 3.2 P-GPU-1's pre-promotion circularity — GPU-scoped, but reachable from a CPU campaign

P-GPU-1 permits decision-grade GPU results only on an already production-named kernel, so a candidate
cannot produce the evidence needed to decide whether it should become production
(`gpu-cross-device.md:16-19`, `:44-48`, `:50-53`). The v8 process handled this with a provisional
promotion followed by production-era certification.

The **CPU half is already solved** and needs no amendment: `bench-cpu.md:38-44` defines candidate
release identity (*"Candidate = clean committed tree whose binary reports that commit. MUST record:
branch, commit, dirty status, binary + shared-library SHA256s, the exact `build:` line …, `ldd`, model
path/size/SHA256, complete argv, environment, date, attestation ref"*), and `:83-88` defines a full CPU
kernel-promotion decision rule with ratio bands (≥0.98 PASS, <0.95 FAIL) and a pooling rule.
P-BENCH-PREFILL-1 is a candidate-versus-production protocol by construction.

**It is not, however, "only a problem when we freeze a GPU kernel."** CPU and GPU share `llama.cpp`
and one frozen branch (§1.5), so *any* llama freeze produces a new binary on both paths. A CPU-only
champion still creates `production-consolidated-vN`, and the `gpu` symlink resolves into that same
tree's `build-hip`. The circularity is therefore reachable from a CPU-only campaign whenever the
change touches shared ggml core.

**Backend-unchanged escape.** A backend owes candidate-grade evidence only if *its* binary changed.
Where it did not, the incumbent's evidence transfers by identity, that backend's cells drop from the
matrix with a receipt, and the circularity never arises. This makes genuinely CPU-local campaigns
cheap and confines the amendment's blast radius to changes that actually reach the HIP build.

The test is **not** naive byte-identity of the built binary. llama.cpp/ROCm builds embed build IDs,
timestamps, and absolute paths, so a freshly built binary is essentially never byte-identical to one
built months earlier in a different directory — a test formulated that way would never fire. It is
two-stage:

1. **Source-closure identity (the gate).** Obtain the backend's build-target dependency closure from
   the build system itself (CMake/Ninja depfiles), never a hand-maintained list or a directory-prefix
   guess. Diff `production_base..candidate` restricted to that closure. Unchanged iff the diff is
   empty **and** toolchain, flags, and build environment are identical. Deterministic, cheap, and
   independent of build reproducibility.
2. **Normalized binary confirmation (required before dropping cells).** Rebuild the production base
   commit in the candidate's build environment so both share one non-determinism regime, then compare
   normalized hashes of `.text`, `.rodata`, `.data.rel.ro`, and the dynamic symbol table, excluding
   `.comment`, `.note.gnu.build-id`, and debug sections.

Disagreement between the stages is a **hard finding**, never a silent preference for the cheaper
answer: the closure is wrong or the build is non-deterministic, the backend owes full evidence, and the
discrepancy is filed against the build-identity machinery. Transfer additionally requires the
incumbent's evidence to still be in scope — same models and recipes, same topology hash, no era
boundary crossed for that backend.

Full text: [`artifacts/operator/autokernel-policy-draft/P-GPU-1-sealed-candidate-amendment.draft.md`](../../artifacts/operator/autokernel-policy-draft/P-GPU-1-sealed-candidate-amendment.draft.md) §4.

**Required resolution:** amend P-GPU-1 (or add a GPU section to `P-KERNEL-FREEZE-1`) defining a
**sealed candidate** as eligible for release-gating GPU evidence before cutover, binding: production
base commit; clean full candidate commit; complete source tree and agent-file overlay; toolchain
identity; binary and linked-library hashes; immutable evaluator hash; derived scope manifest; and
evidence directory hash tree. The CPU-side bindings are a restatement of `bench-cpu.md:38-44` and
should cite it rather than duplicate it.

### 3.3 Freeze, cutover, era rows, baseline applies, reload ownership — all stay human

See §1.3 for the four boundaries. Under the revised model **no authority amendment is required.**
What is required instead is that the design *stop implying* otherwise:

- the release-side component is a **packager**, not a broker (§11);
- the loop's terminal success state is `RELEASE_PACKAGE_READY`, not `FREEZE_ELIGIBLE`;
- the era row and AutoPilot rebaseline are **outputs of the package as drafts for the operator**, never
  writes; and
- cutover scheduling is routed to whoever owns inference, per `OPERATING_CONSTRAINTS.md:41`, through
  the session bus (§3.6).

### 3.4 Claim grammar must be expressible in the schemas

`MEASUREMENT.md:13` — *"A claim = (metric, protocol-id, n/reps, date, host-attestation ref)"*;
`:85-95` — *"**Category (required)**: every reported measurement declares exactly one of
`category=OPTIMUM` · `BASELINE` · `CANDIDATE` … An unlabelled measurement is not decision-grade."*

The evaluation event must carry `category`, `n`/`reps`, `protocol_id`, `attestation_ref`, and
`metric_direction`, or §18's prose duty ("never record an observation as a release claim") is
unenforceable. See §7.4.

### 3.5 The `pgrep` contradiction must be resolved before any protocol-conformant run

CLAUDE.md bans `pkill`/`pgrep` on a name pattern. `bench-cpu.md:16-17` mandates *"no concurrent
inference (`pgrep llama` zombie check…)"* as a P-BENCH-1 precondition, `MEASUREMENT_POLICY.md:38`
repeats it, and `gpu-cross-device.md:27-30` requires *"`llama-server` / AutoPilot / KFD PID checks
before and after"*. As written, the evaluator cannot satisfy the protocol it must run under.

**Resolution (recommended):** two-stage.

1. **Interim** — a single audited read-only wrapper, `scripts/utils/inference_preflight.sh`, which
   enumerates matching PIDs, reports them, and **never signals**. It becomes the only sanctioned
   name-pattern reader in the codebase. The CLAUDE.md ban's origin (INC-20260731) was broad
   *pattern kills*; a read-only enumerator at one audited call site has a different blast radius,
   but it must be that one call site, not `pgrep` sprinkled through the evaluator.
2. **Target** — replace the check with region-lock/device-claim holder witness plus owned-cgroup
   enumeration, and ratify that substitute into P-BENCH-1/P-GPU-1 as an equivalent precondition. This
   depends on the GPU claim (§2.5) existing.

### 3.6 Evaluator immutability needs enforcement a daemon cannot bypass

The trust boundary is enforced in three layers around `human_only_paths.yaml`: a PreToolUse hook
(`scripts/hooks/check_trust_boundary_edit.sh`, wired at `.claude/settings.json:76`), a `.sha256` pin
checked by `session_bus.py validate`, and a coordinator-daemon audit emitting a `defect` on drift.

**Layer 1 blocks agent tool writes only.** AutoKernel is a daemon; a candidate binary is a subprocess.
Neither passes through a PreToolUse hook, and layers 2–3 detect after the fact.

**Required:** (a) add the AutoKernel evaluator bundle, `P-AK-SEARCH-1`, `P-KERNEL-FREEZE-1`, and the
threshold/objective policy to `human_only_paths.yaml`; (b) give the evaluator OS-level protection —
separate uid or read-only bind mount — for the actor's and the candidate's execution context; (c) if
AutoKernel discovers a missing coverage class it records `EVALUATOR_COVERAGE_GAP`, blocks release for
that lineage, continues unrelated research, and may draft an amendment bundle. It may not modify the
live evaluator.

**C6 candidate-execution acceptance criterion (2026-08-10).** A container name is not evidence of
confinement. Before candidate code may run unattended, the evaluator must attest an active syscall
policy (seccomp or a stricter equivalent), a non-root execution identity, no write mount outside the
owned campaign tree, no host-PID signalling capability, resource ceilings, and fail-closed startup if
any control is unavailable. `seccomp=unconfined` and in-process `exec_module` are explicit failures.

**Comparator retained, stack declined.** The useful pattern in the reviewed external execution stack
is the controller → dispatcher → router → owned worker split backed by a durable SQL job record whose
idempotency key prevents a restart from executing the same submission twice. AutoKernel keeps its
existing journal/resource plane and imports only that invariant: a resumed job with the same ID and
fingerprint is a replay; the same ID with a different fingerprint is a hard conflict. External registry
images, a cgroup-v1 reboot, and the foreign coordination plane are outside the adopted design.

### 3.7 Evidence durability

`MEASUREMENT.md:146-156` (operator-ratified 2026-08-02) requires evidence behind any ratified or
production-affecting claim to live in-repo under `epyc-inference-research/data/<campaign>/` with a
`SHA256SUMS` and a README, forbids scratch paths (`/mnt/raid0/llm/tmp/...`) as the citation of record,
and requires oversized artifacts to be recorded hash-and-provenance-only *with the citation saying so*.
Enforced by `epyc-inference-research/scripts/validate/check_evidence_durability.py` (note: that path is
in the **research** repo, not epyc-root as `MEASUREMENT.md:155` implies).

Known exposure to clear before AK1 hashes anything:

- the source draft was gitignored under `tmp/*` and never committed → **moved 2026-08-02** to
  `docs/reference/autokernel/system-wide-inference-kernel-optimization-draft.md`;
- the GPU decision surface `/mnt/raid0/llm/tmp/claude-artifacts/np_context_v8_decision.html`
  (SHA-256 `816ad5cd…`, verified) is still a scratch citation and must be copied into the research
  repo's evidence root, its new locator recorded, and its backing bundles tracked;
- `epyc-inference-research/artifacts/np_context_study_20260723/` has **zero tracked files**; the v8
  sibling tracks only its 5 driver scripts;
- ~~`docs/design/p2-5j-host-thread-placement-sweep-protocol.md` — required by seed G1 — was **deleted
  from git** by unrelated commit `27fbfce5`; all of `docs/design/` is untracked~~ — **RESOLVED, verified
  2026-08-10.** The doc is tracked again (restored by `047b5a40`) and the backup copy at
  `artifacts/audit/untracked-backup-20260729/design/` is **byte-identical** to the live file, so the
  backup is no longer load-bearing for it. *"All of `docs/design/` is untracked" is also no longer
  true*: 5 of 7 files are tracked. **Seed G1's dependency is satisfied.** Logged at
  `artifacts/audit/backlog-closure-correctness-20260729.md:78` — an artifact that was itself untracked
  until 2026-08-10, which made it a sixth instance of the very defect it catalogues;
- this handoff and its index rows are untracked while `master-handoff-index.md:459` (committed)
  already links them.

- [x] **AK-X-11 — Track the two remaining cited-but-untracked `docs/design/` files**, or record why not.
  ✅ 2026-08-10 — both cited design records are included in the AutoKernel closure pathspec rather
  than left as unowned filesystem state.
  `agent-session-control-surface.md` is cited by
  [`session-bus-thin-dispatcher.md`](session-bus-thin-dispatcher.md) and
  `repl-s4-omega-multipaper-arm-20260729.md` is likewise cited from a tracked handoff, yet neither is
  in git. **A citation to an untracked file is indistinguishable from a citation to a tracked one
  until someone clones the repo and it resolves to nothing** — which is the general form of the defect
  the 2026-07-29 audit catalogued and then reproduced on itself. Found 2026-08-10 while verifying the
  P2-5j resolution above.

**Required:** AK1 records a **durability class** per artifact — *carried-in-git*,
*durable-untracked*, or *hash-and-provenance-only* — so a verifier can distinguish a defect from an
expected absence, and extends `check_evidence_durability.py` (currently scoped to the model registry)
to cover the AutoKernel import manifest and evidence citations.

---

## 4. Non-negotiable invariants

1. **Fresh production base.** Every campaign is anchored on the current production tip. The champion
   lineage is re-anchored at every freeze; between freezes, production does not move, so accumulation
   is not drift.
2. **Full candidate promotion.** Release evidence is produced by the same full candidate that is
   frozen; no promotion-time cherry-pick reconciliation.
3. **Frozen means immutable.** No actor builds in or modifies any production tree.
4. **Evaluator independence.** Actor, evaluator, and release packager are distinct authority domains
   even on one host, enforced at the OS level (§3.6).
5. **No autonomous freeze or cutover.** AutoKernel produces a release package; a human executes it.
6. **Correctness is lexicographically first.** Speed, energy, or simplicity cannot compensate for a
   correctness, quality, integrity, or stability failure.
7. **All outcomes are durable.** Failures, crashes, timeouts, rejected proposals, negative results, and
   invalidated runs remain in an append-only event journal, in-repo, with a recorded durability class.
8. **Derived views may rewind; evidence does not disappear.** Pareto/frontier/champion state rebuilds
   from events. "Purge" is a supersession/tombstone event, never deletion of a primary record.
9. **Resources are acquired, not observed.** Every CPU/GPU benchmark or profiler run holds the
   appropriate region/device claim. Idle sensing is never a claim.
10. **Owned-process lifecycle only.** The loop tracks PIDs/cgroups it launched, quiesces them at a
    boundary, verifies termination, and never signals by name pattern.
11. **Deterministic replay before regeneration.** Saved outputs, profiles, and raw samples are
    rescored without inference when the generation path remains valid.
12. **Determinism class is an interface.** A candidate may not silently change same-seed run-to-run
    bitwise stability; a change of class is a declared, release-relevant property.
13. **One conceptual mutation per proposal — per *step*.** Parameter sweeps may generate many
    points; a source proposal must remain falsifiable and revertible. An **architectural campaign**
    (§8.4.1) declares a multi-step lineage with an end-state up front; the rule then binds each step,
    and the critic evaluates the plan as a whole. A change that is irreducibly multi-part — a new
    layout plus the kernel that consumes it plus its dispatch plus its repack path — must be
    *expressible*, or the loop can only ever polish what already exists.
14. **No estimated percentage by narration.** Readiness is computed from records by a deterministic
    controller; the LLM may request, never declare.
15. **Production recipes gate.** Baseline/off-recipe cells are diagnostic and never veto or justify a
    release.
16. **Default-off until release.** Experimental paths retain a fallback/kill switch unless the change
    is structurally inseparable and the campaign explicitly carries that risk class.
17. **No evaluator self-modification.** Changes under the measurement trust boundary stop release
    eligibility until separately reviewed.
18. **Declared equals traced.** The affected-surface manifest is mechanically derived and
    dynamically confirmed; the actor's declaration is a scored prediction, never a scope input.
19. **Control is verified, not requested.** Every operator control — pause, drain, abort, stop —
    is acknowledged in the journal, latched on disk, and re-read from disk at the top of each
    iteration under the write lock. A halt survives restart until an operator resumes. An unacked
    control command is a hard failure, not a slow one.
20. **The planner does not re-consume its own prose as fact.** Machine-readable outcome and planner
    narrative are separate fields; narrative is not retrievable into a later planning context by
    default, and a superseded belief can be removed from retrieval without being removed from the
    record.

---

## 5. Target architecture

```text
                 immutable constitution + evaluator bundle
                                   |
                                   v
 production/registry facts -> campaign compiler -> planner -> critic
                                   |              |
                                   |              v
                                   |         proposal record
                                   |              |
                                   v              v
                         isolated worktree <- actor/executor
                                   |
                          build + candidate record
                                   |
                                   v
                  T0/T1 trusted research evaluator
                                   |
                        append-only experience journal
                                   |
                       champion promotion guard  --(occasional)--> T2 lineage estimator
                                   |                                        |
                                   v                                        v
                          champion lineage (per tree) <--------- readiness signal to operator
                                   |
                        [ OPERATOR REQUESTS FREEZE ]
                                   |
                                   v
                          seal release candidate
                                   |
                                   v
                     T3 kernel-freeze evaluator
                                   |
                          release verdict bundle
                        (+ operator waiver, if any)
                                   |
                                   v
                          release PACKAGE assembler
                     (transaction plan, rollback plan,
                      draft era row, draft rebaseline,
                      pre-validated command sequence)
                                   |
                                   v
                    OPERATOR executes freeze + cutover
                                   |
                                   v
                       T4 post-cutover verification
```

### 5.1 Constitution plane — epyc-root

Human-amendment-only: `MEASUREMENT.md` and protocol annexes; search and release protocol definitions;
evaluator bundle hashes; the freeze transaction contract; era/reconciliation requirements; the evidence
retention rule (§3.7, §5.8); and this handoff. The actor has read access only.

### 5.2 Campaign/control plane — epyc-inference-research

Campaign compiler and state machine; planner/critic prompts and controller adapters; proposal schema
and validators; worktree/build orchestration clients; trusted T0/T1/T2 evaluator entry points;
experience journal and derived views; champion guard and readiness estimator; candidate sealer;
release-package assembler; dashboard export contract. The core controller is domain-agnostic; backend
adapters supply build, correctness, profiling, workload, and release hooks.

### 5.3 Source/authoring plane — experimental trees only

Each campaign receives a dedicated worktree under `/mnt/raid0/llm/`, created from the fresh production
tip, **namespaced** `llama.cpp-ak-<campaign_id>` so it cannot collide with the 13 pre-existing
worktrees or with another session's `llama.cpp-experimental` (currently on
`experimental-v8-refresh-20260724`). Because `/workspace/repos/<name>` and `/mnt/raid0/llm/<name>` are
one clone, all commits are pathspec-limited and branch names are namespaced `ak/<campaign_id>/…`.

The actor may read and edit candidate source, commit on the campaign branch, build in a campaign-local
build directory, run targeted static/unit/op tests through controlled entry points, and abandon,
branch, or restore candidates through recorded worktree operations. The actor never writes the
canonical production tree, the shared staging area outside its pathspec, or a release bundle.

### 5.4 Evaluation plane — trusted runner

The evaluator reads a sealed candidate snapshot; resolves the exact immutable evaluator version;
acquires resources; captures host/binary/model/runtime identity; runs tier-specific correctness and
performance work; independently recomputes scores; emits hash-bound records; and has no authority to
modify candidate source or production state. The C6 primitives belong here, wired to real worktrees and
CPU/GPU resource controls.

It also emits a **runtime source-label attestation**: the resolved path and content hash of every module
actually loaded, captured at run time rather than inferred from the import statement. AutoPilot learned
this expensively — same-named modules in two repositories with `sys.path` mutated at import and again
lazily at runtime, so *which code scored an eval depended on which eval ran first in the process*, while
provenance still recorded the intended scorer. AutoKernel imports across three repositories; this
attestation is what makes "the evaluator that actually ran" a checkable fact rather than an assumption.

### 5.5 Evidence and memory plane

1. `events.jsonl` is append-only and fsync-per-event, **sharded with rotation only past all cursors**
   (the `BUS_PROTOCOL.md:32-34` pattern); every reader reads all shards.
2. Candidate source snapshots and patch bundles are content-addressed.
3. SQLite is a rebuildable index/view, never the primary record.
4. Pareto, best-per-regime, champion, failed-mechanism, do-not-repeat, and readiness views derive from
   the journal.
5. A context synthesizer writes the planner's "state of the search" brief from both wins and failures.
6. **Machine record and planner narrative are separate fields, and narrative is not retrievable by
   default.** AutoPilot's worst contamination lived in planner-authored free text inside the primary
   journal: the loop re-read its own prose each round, regenerated a false story, and re-distilled it
   into new strategies, so scrubbing every derived store never stuck — one instance ran 81 further
   trials on the same false story after the code fix landed. An append-only journal makes that failure
   *worse*, not better. Outcomes, verdicts, and receipts are therefore structured fields; prose lives
   in a `narrative` field the retrieval layer excludes unless a proposal cites it by event id.
7. **Retrieval-scope supersession.** A superseded belief is removed from *retrieval* while remaining in
   the *record* — a `RETRIEVAL_SUPERSEDED` event, never a deletion. This preserves invariants 7 and 8
   while giving the loop a way to stop believing something, which an immutable log otherwise lacks.
8. Everything lands under `epyc-inference-research/data/<campaign>/` with `SHA256SUMS` and a README,
   with a durability class per artifact (§3.7).

The next round must learn from failed candidates, not only from the correct Pareto frontier.

### 5.6 Release plane — sealer, evaluator, packager

- **Seal:** convert a champion worktree into a content-addressed, immutable release candidate.
- **Evaluate:** run the full freeze program; produce a pass/fail bundle, admitting operator waivers.
- **Package:** assemble the transaction plan, rollback plan, draft era row, draft AutoPilot rebaseline,
  and a pre-validated command sequence for the operator.

Never combine these into one LLM-owned shell session. The packager holds no authority and performs no
production write, in any mode.

### 5.7 Resource plane — orchestrator integration

Stable APIs for: CPU region claims; **exclusive GPU device claim** (to be built, §2.5, §14 AK2);
host-health tier and reboot-required state; contention/co-residency policy; disk headroom; process/cgroup
ownership; production inference ownership and drain boundaries; safe cache preparation; profiler
privilege; pause/drain/resume. The session bus carries scheduling intent and revocation; the actual
exclusion source remains region-lock/device locks. A heartbeat is not a lease — but the absence of a
heartbeat means nothing can revoke you (§3.6, §14 AK2).

### 5.8 Storage plane

`/mnt/raid0` is 96% full with 162G free; `data/` is 118G; existing llama worktrees total 41G. Invariant
7 says evidence is never evicted, and `MEASUREMENT.md:223-229` makes reclamation an operator decision
(*"Disk-hygiene candidates … are an operator call, not contamination"*). Without a plane, the loop
halts on a full disk within a handful of campaigns.

Design:

| Class | Contents | Policy |
|---|---|---|
| **Permanent, in-repo** | events, reduced metrics, patches/diffs, hashes, manifests, README/SHA256SUMS | Never deleted |
| **Permanent, large** | champion binaries; incumbent production binaries for N−1 and N−2 (§10.5) | Never deleted; hash-and-provenance cited |
| **Expirable** | rejected-candidate build trees, worktrees of retired campaigns, raw profiler traces older than the lineage they informed | Expiry rule ratified once in AK0; every deletion writes a tombstone event carrying the artifact hash and reason |
| **Never stored** | candidate outputs used as a correctness oracle | — |

Plus a per-campaign quota, a `DISK_PRESSURE` stop state, and a headroom precondition in BOOTSTRAP.

---

## 6. Required exposed surfaces

The planner sees structured facts and controlled actions, not a giant prompt of unverified prose.

| Surface | Read contract | Action contract | Key guard |
|---|---|---|---|
| Production identity | backend, tree, branch, commit, binary/library hashes, version, overlay hash | none | fails if production dirty or identity mismatched |
| Experimental lineage | campaign base, parent candidate, commit graph, dirty/build state, champion pointer | create worktree, apply patch, commit, restore, fork | path allowlist; production paths denied; namespaced branches |
| Source map | symbols, files, call graph, op registrations, dispatch predicates, existing tests | search/read only for planner; edit through actor | snapshot hash binds proposal context |
| Build | toolchain, presets, cache, target binaries, expected outputs | configure/build targeted or full, campaign build only | no production build dir; output hash receipt |
| Correctness | exact op shapes, reference outputs, state/rollback cases, PPL/quality sentinels, determinism class | run declared test groups | evaluator-owned test list; candidate cannot skip |
| Workloads | compiled role priors, model/quant/context/spec recipes, observed shape histogram, co-residency lineup | derive sentinel/release matrices | matrix derives from current facts and evaluator version |
| Profiler | supported gfx90a/CPU counters, counter quality, prior profiles, wall-share map | targeted trace/counter run | mechanism prediction predeclared; unsupported counter explicit |
| Performance | canonical recipe IDs, cache state, reps, samples, variance | T1/T2/T3 evaluation request | runner constructs argv from a codified recipe; actor cannot hand-type a bench command |
| Host/resource | topology hash, NUMA/GPU attachment, region/device holders, thermals, uptime, memory, disk | acquire/release claim, request quiescence, prep cache | acquisition receipt mandatory; reboot is operator authority |
| Process lifecycle | exact launched cgroup/PIDs and start times; preflight enumeration via the single audited wrapper (§3.5) | terminate owned scope, verify dead | no name-pattern signalling; stale process is a hard failure |
| Experience | all proposals, diffs, results, failures, mechanisms, supersessions | append event, derive views, synthesize context | primary log append-only, fsynced, sharded |
| Release | sealed candidate, evaluator bundle, derived scope, incumbent rollback, active waivers | request T3, request package assembly | actor cannot write release status, production link, era row, or baseline |
| Observability | campaign phase, budget, frontier, failures, champion readiness, resource state | pause/drain/resume/abort request | deterministic controller owns state transition |

### 6.1 Planner source context

- current campaign objective and production-weighted role exposure;
- production base and candidate diff summary;
- per-op/shape wall share and mechanism classification;
- compiler/backend constraints for the target hardware;
- existing dispatch/fallback behaviour;
- recent correct frontier and the current champion;
- recent failures grouped by mechanism;
- do-not-repeat matches from active/completed handoffs, with receipts;
- oracle coverage — whether a declared oracle already implements the idea (§6.5);
- evaluator coverage and confidence;
- remaining experiment/compute/storage budget;
- known candidate interactions; and
- exact tool/action affordances available this round.

The synthesizer cites event IDs and source/profiler receipts. It may summarize; it may not invent a
state transition. External or imported content is rendered in provenance-tagged quarantine form
(`OPERATING_CONSTRAINTS.md:27-31`) and never in an instruction position.

### 6.2 Actor tools

Search/read source; patch files; invoke build targets; run bounded non-measurement developer tests;
inspect compiler diagnostics; request trusted evaluator tiers; record rationale; commit a conceptual
candidate. Generic shell is acceptable inside the isolated authoring environment. Host mutation,
resource acquisition, benchmarks, privileged profilers, and release operations go through auditable
wrappers.

### 6.3 Critic surfaces

The critic gets the proposal, source context, affected-surface map, oracle coverage, and prior failures
before the actor runs, and answers structured questions:

- Is the hypothesis falsifiable?
- Does the proposed measurement distinguish the claimed mechanism from alternatives?
- Are exact target and non-target shapes identified?
- Is a faster-but-wrong path plausible?
- Does an existing dispatch/path in **our tree** already implement this?
- **Does a declared oracle already implement this, and is porting cheaper than authoring?**
- Is the proposal actually one conceptual change?
- Can the claimed end-to-end value exceed the measured wall-share ceiling?
- Is the resource cost proportional to expected information gain?
- Does it repeat a recorded negative without new evidence — and does that negative carry a receipt?

The critic may reject or revise. It cannot waive evaluator gates. Prefer a different provider/model
from the planner; a critic sharing the planner's blind spots mostly agrees.

### 6.4 Affected-surface derivation — mechanical, then traced

Freeze scope, lineage composition, and sentinel selection all key off the affected-surface manifest.
If the actor declares it, the actor controls its own release scope. Therefore:

1. **Static derivation** — diff → touched files → symbols → op registrations → dispatch predicates →
   affected backends. Over-approximation is expected and acceptable (a shared-header change implies
   the whole tree until proven otherwise).
2. **Dynamic confirmation** — T0 runs the op suite under a dispatch trace and records which kernels
   actually executed. This reuses the no-fallback instrumentation T0 already needs.
3. **Reconciliation** — `derived ⊇ traced` must hold; `traced ⊄ derived` is a hard candidate failure.
   The actor's declaration is retained as a scored prediction and fed to the critic, never used as a
   scope input.

### 6.5 Oracle registry

Declared, read-only reference implementations the loop may study and port from:

Each row declares a **harvest class**, and the axis is **architectural portability, not licensing**.
Standing project policy is open-source self-hosted with no commercial use, where licences are not
blockers — an earlier revision of this section made licensing a gate, which was wrong. Attribution is
recorded as courtesy and provenance, never as a condition of entry.

What actually decides the cost of harvesting is whether the artifact can run on our target:
`portable_source` means the code compiles and runs on gfx90a or EPYC and may simply be brought across;
`reimplement` means the algorithm is worth having but the code is not — CUDA-only instructions, a
different memory hierarchy, or an unavailable intrinsic. Misclassifying is not a legal problem; it is a
schedule problem, because a `reimplement` oracle costs authoring effort that a `portable_source` one
does not.

| Oracle | Class | Why it is worth reading |
|---|---|---|
| `ik_llama.cpp` (on disk) | `portable_source` | source of the iqk lineage; the single largest banked gain this project has |
| upstream `llama.cpp` / `ggml` | `portable_source` | fixes and optimizations the fork has not taken; the fork diverges continuously |
| AMD **composable_kernel** / hipBLASLt / rocBLAS | `portable_source` | **the most directly relevant unexploited source for gfx90a** — CDNA2 GEMM/attention tiling written by the vendor for this exact architecture |
| AMD **AITER** | ~~`portable_source`~~ **RETIRED 2026-08-03** | I added this row claiming AITER was "AMD's own inference kernel work, same target hardware". It is not: AITER's supported-hardware table lists **no MI210/MI250/gfx90a, not even experimental** — consumer RDNA parts rank ahead of our datacenter card. See the `cdna2-abandoned-by-vendor-and-quant-schools` HARD_CONSTRAINT in §19.2. Kept visible rather than deleted: the row was wrong, and a future reader reaching for AMD's inference kernels should meet the correction rather than re-add it |
| **FlashAttention / FlashInfer** | `portable_source` where a HIP path exists, else `reimplement` | attention tiling, KV layout, paged-attention kernels |
| **CUTLASS** | `reimplement` — CUDA instructions, no gfx90a path | tiling, pipelining and epilogue *design*; instructions do not port to gfx90a |
| **vLLM / SGLang / TensorRT-LLM** | `reimplement` | scheduling, paged KV, continuous batching as design oracles; TensorRT-LLM in particular is read-only |
| **Marlin / EXL2 / AWQ / GPTQ kernels** | mixed — CUDA cores `reimplement`, layout/packing `portable_source` | low-bit GEMV and dequant layouts — directly adjacent to the G2/G3 seed families |
| **Triton / MLC / TVM kernel corpora** | `reimplement` | autotuning and layout search strategies |

This list is a starting set, not a closed one. New oracles enter through the project's existing
`research-intake` pipeline rather than by an agent adding a row: intake verifies whether gfx90a or
EPYC is actually supported, **normalises the claimed result to roofline utilisation** (§8.3.1) so it
can be compared with our own, and assigns the harvest class. An oracle whose class cannot be
established does not enter.

Rules: never build or measure a production claim from an oracle tree; a port is a normal candidate and
pays T0–T3 identically; every port records the oracle commit, the harvest class relied on, and an
attribution note. A `reimplement` port records the algorithm's source and states that the
implementation is independent — provenance, so a later reader can find the original reasoning. The
project's largest recent kernel gain came from porting and enabling iqk (`iqk-port`,
`iqk/enable-iquants-v7-20260721`; +33–43% prefill era), so this is a first-class campaign kind, not a
footnote.

---

## 7. Data contracts

### 7.1 Campaign manifest

```yaml
schema: epyc.autokernel.campaign.v2
campaign_id: ak-<backend>-<objective>-<date>
backend: llama_gpu          # llama_cpu | llama_gpu | whisper_stt | qwentts_tts | serving_runtime
source_tree: llama.cpp      # llama.cpp | whisper.cpp | qwentts.cpp
production_anchor:
  repo: /mnt/raid0/llm/llama.cpp
  branch: production-consolidated-v8
  commit: 67a433bf45a8a091d83b4ea0b32ff0735fd51800
objective:
  rule: per_phase_non_inferiority_plus_improvement
  phases: [prefill, decode]
  protocol_by_phase: {prefill: P-BENCH-PREFILL-1, decode: P-BENCH-1}
  recipe_class: production_optimal
  phase_trade_exception: null      # pre-declared {regressing_phase, band, expected_gain, roles} or null
  target_regimes: []
scope:
  affected_ops: []                 # filled by derivation, never by the planner
  affected_arch_classes: []
  derived_role_manifest_sha256: "..."
policy_ref:                        # authority/thresholds live in the human-only policy plane
  search_protocol: P-AK-SEARCH-1/v1
  release_protocol: P-KERNEL-FREEZE-1/v1
  policy_bundle_sha256: "..."
budgets:
  max_wall_hours: 0
  max_gpu_hours: 0
  max_cpu_region_hours: 0
  max_candidates: 0
  max_controller_tokens: 0
  max_storage_gb: 0
readiness_reporting:
  reference_point_gain: 0.25       # ADVISORY signal to the operator, not a trigger
  reference_lcb_gain: 0.20
stop_policy:
  plateau_rounds: 0
  max_consecutive_integrity_failures: 0
  max_consecutive_build_failures: 0
  max_command_retries: 3           # OPERATING_CONSTRAINTS.md:44-46
```

The compiler, not the planner, fills production identity, protocol IDs, scope hash, and policy
reference. The planner may propose objective details before campaign start; it may not mutate them
mid-campaign. **No campaign manifest ever carries a freeze or cutover authority flag** — there is no
such authority to carry.

### 7.2 Proposal manifest

```yaml
schema: epyc.autokernel.proposal.v2
proposal_id: akp-...
campaign_id: ak-...
parent_candidate_id: akc-...
controller:                        # so controller A/B is computable after the fact
  provider: "..."
  model_id: "..."
  effort: "..."
  prompt_bundle_sha256: "..."
  sampling_params: {}
  context_manifest_sha256: "..."
realized_cost:                     # attributed, not merely budgeted (§12 zero-yield row)
  controller_tokens: 0
  build_seconds: 0
  evaluator_wall_seconds: 0
  gpu_seconds: 0
  cpu_region_seconds: 0
  storage_gb: 0
hypothesis: "..."
narrative: "..."                   # planner prose. NOT retrievable into a later planning context
                                   # by default; the fields above are the machine record (§5.5)
change_class: dispatcher           # parameter | dispatcher | arithmetic | layout | fusion |
                                   # moe_scheduling | recurrent | scheduler_policy | oracle_port |
                                   # core_header  <- own risk tier, see §8.5.1
declared_symbol_deltas:            # symbols/registrations this change intends to add, remove, or
  added: []                        # change arity on. Anything outside this set that the binary
  removed: []                      # diff finds is a hard T0 failure, not a warning
  arity_changed: []
campaign_kind: source_change       # config | dispatch | layout | fusion | scheduler | capability |
                                   # oracle_port
oracle_reference:                  # required when campaign_kind == oracle_port
  oracle: null
  commit: null
  license_check: null
novelty_basis:
  prior_event_ids: []
  source_receipts: []
  do_not_repeat_matches: []
expected_information_gain: 0.0
target: {regimes: [], ops: [], shapes: [], models: []}
non_target: {regimes: [], shapes: []}
mechanism_prediction:
  bottleneck_before: memory_latency
  expected_counter_changes: {}
  expected_wall_share_ceiling: 0.0
  wall_share_receipt_id: "..."     # the measured ceiling this is checked against
change:
  predicted_affected_surface: []   # scored prediction only; never a scope input (§6.4)
  files_and_symbols: []
  conceptual_change: "..."
  parameter_surface: {}
  estimated_diff_size: 0
risks: {correctness: [], numerical: [], state_or_rollback: [], resource: [], integrity: []}
fallback: {dispatch_guard: "...", kill_switch: "..."}
evaluation_plan:
  required_t0: []
  required_t1: []
  conditional_t2: []
  profiler_questions: []
resource_request: {lane: gpu, expected_minutes: 0, expected_storage_gb: 0}
stop_condition: "..."
critic_verdict: {status: pending, reasons: []}
```

Every field is machine-validated before resource acquisition. A proposal without a falsifiable counter,
a wall-share prediction, or a `change_class` that maps to a cheap suite (§9.5) is rejected before it
consumes a benchmark window.

### 7.3 Candidate record

Must make the artifact reproducible: campaign/proposal/parent IDs; worktree and source commit;
clean/dirty state; patch bundle/source snapshot hash; production base ancestry proof;
compiler/toolchain/build command and logs; binary/library hashes and linkage proof; feature
flags/dispatch predicate; **derived and traced affected-surface manifests**; determinism class;
evaluator version/hash; resource and host receipts; storage footprint and durability classes; raw
evaluation event IDs; derived verdicts; controller provenance (inherited from the proposal); champion
status; status/supersession reason.

The current natural key `(label, ts, git_sha)` is insufficient.

### 7.4 Evaluation event

```yaml
schema: epyc.autokernel.evaluation_event.v2
event_id: ake-...
campaign_id: ak-...
candidate_id: akc-...
tier: T1
claim_grammar:                     # MEASUREMENT.md:13, :85-95
  category: CANDIDATE              # OPTIMUM | BASELINE | CANDIDATE
  protocol_id: P-AK-SEARCH-1/v1
  metric: decode_tokens_per_s
  metric_direction: higher_better
  reps: 0
  attestation_ref: "..."
evaluator: {id: P-AK-SEARCH-1/v1, bundle_sha256: "..."}
artifact: {source_sha256: "...", binary_sha256: "...", linkage_sha256: "..."}
anchor:                            # every ratio needs its denominator bound
  binary_sha256: "..."
  linkage_sha256: "..."
  measurement_event_ids: []
scope_manifest_sha256: "..."
host_receipt: "..."
resource_claim_receipt: "..."
co_residency: single               # single | co_resident:<lineup_id>
correctness: {}                    # per-case vector, not a rolled-up verdict
quality: {}                        # per-question vector where applicable
stability: {}                      # per-iteration vector
scope_denominator:                 # what this cell actually measured, so a gate refuses a scope
  machine_subset: full             # mismatch instead of silently applying a full-machine
  numa_nodes: []                   # threshold to a partial-machine cell
  devices: []
  cores: 0
determinism: {class: bitwise_stable, same_seed_repeat_runs: 0}
performance:
  raw_samples: []
  paired_blocks: 0
  estimate: null
  uncertainty: null                # e-process evidence value / MDE, not an ad-hoc LCB
mechanism: {}
integrity_flags: []
status: pass                       # pass | fail | inconclusive | invalid | timeout | crash | rejected
supersedes: []
created_at: "..."
```

Derived scores are reproducible from raw samples. The candidate cannot supply its own trusted score.

### 7.5 Champion record

```yaml
schema: epyc.autokernel.champion.v1
source_tree: llama.cpp
anchor_commit: 67a433bf45a8a091d83b4ea0b32ff0735fd51800
branch: ak/champion/llama-20260802
member_candidates: []              # ordered, each with its own candidate record
combined_candidate_id: akc-...     # the composed artifact, re-measured as a whole
last_t0: {event_id: "...", status: pass}
last_t1: {event_id: "...", status: pass}
last_t2: {event_id: "...", status: pass}
readiness:
  by_backend:
    llama_cpu: {prefill: {...}, decode: {...}}
    llama_gpu: {prefill: {...}, decode: {...}}
  reference_signal: "point +X% / LCB +Y% versus anchor on N cells"
affected_surface_union_sha256: "..."
storage_gb: 0
blocking_conditions: []            # e.g. EVALUATOR_COVERAGE_GAP, open phase-trade exception
```

### 7.6 Release package

Produced only on operator request. Contains: the sealed candidate identity; the T3 verdict bundle;
any active operator waivers (§10.4); the derived release plan; the transaction plan and rollback plan;
a **draft** era-registry row; a **draft** AutoPilot rebaseline note; the linkage verification results;
and a pre-validated command sequence for the operator. It contains no production write and no
authority claim.

---

## 8. Planner/critic/executor cycle

### 8.1 State machine

```text
BOOTSTRAP
  -> DISCOVER
  -> SELECT_TARGET
  -> PROPOSE
  -> PRE_RUN_CRITIC
  -> MUTATE
  -> BUILD
  -> T0_GATE
  -> T1_SEARCH_EVAL
  -> POST_RUN_CRITIC
  -> BANK_EVENT
  -> UPDATE_SEARCH_STATE
  -> CHAMPION_GUARD
       -> next DISCOVER/SELECT_TARGET round
       -> optional T2_LINEAGE_ESTIMATOR -> update readiness signal
  -> [on operator freeze request] SEAL -> T3_RELEASE_GATE -> PACKAGE
  -> CONTINUE | PLATEAU_STOP | BUDGET_STOP | DISK_PRESSURE | BLOCKED_INSTRUMENT | RELEASE_PACKAGE_READY
```

Every transition is explicit and journaled. The LLM produces proposals and interpretations; a
deterministic controller disposes gates and stop conditions.

### 8.2 BOOTSTRAP

1. verify all production kernels and current production identities;
2. identify the current kernel set and toolchain/linkage generations;
3. derive campaign scope from compiled stack priors and the declared objective;
4. verify evaluator and protocol bundle hashes;
5. verify resource broker, **GPU device claim**, and sandbox availability;
6. verify storage headroom against the campaign's `max_storage_gb`;
7. create a fresh namespaced worktree from production;
8. build or bind the known-good anchor;
9. run positive/neutral/negative/**A-A** evaluator controls without changing production;
10. initialize the append-only journal and derived views, then **assert consistency between them** — if
    the journal holds candidates but a rebuilt view comes up empty, or cardinalities disagree, BOOTSTRAP
    refuses to start rather than proceeding on an empty frontier. AutoPilot lost 232 trials and roughly
    16 days of compute to a restart that came up empty with nothing objecting; a deliberate rebase
    passes an explicit escape flag rather than being silently indistinguishable from that failure;
11. register on the session bus (roster id, heartbeat, lane declaration); and
12. emit `READY` only when the entire chain is reconstructible after restart.

### 8.3 DISCOVER and SELECT_TARGET

Discovery is evidence collection, not automatic source mutation. It compiles production role and
workload exposure; captured real-graph shapes; per-op wall share; bandwidth/compute/launch/sync
classifications; CPU NUMA/locality or GPU occupancy/memory/launch counters; existing fast paths and
dispatch coverage; measured wall-share ceilings; oracle coverage; and existing handoff/negative-ledger
matches.

Selection follows the hierarchy: placement and launch configuration → dispatcher → autotuning →
layout/repack → operator fusion → work scheduling → new kernel → scheduler architecture → alternate
engine. A cheaper layer may be skipped only with an evidence receipt showing why it cannot explain the
measured gap.

### 8.3.1 Roofline utilisation — the normalising metric (operator, 2026-08-03)

Raw speedups do not transfer across hardware and are the main way a published kernel result misleads
this project. "2× faster" is unusable: it was measured on different bandwidth, a different memory
hierarchy, and instructions we may not have. **Fraction of achievable memory bandwidth actually used
against the theoretical decode roof does transfer**, and it is the metric both discovery and oracle
intake normalise to.

Decode is bandwidth-bound on both backends, so for a given regime:

```
bytes_per_token   = active weight bytes read per token
                    (dense: whole model; MoE: active-expert bytes only)
                  + KV bytes read per token at the measured context
theoretical_tps   = achievable_bandwidth / bytes_per_token
utilisation       = measured_tps / theoretical_tps
```

Record **two denominators, always both**: datasheet peak bandwidth, which is never reachable and gives
the absolute roof, and *measured* achievable bandwidth from a STREAM-class probe, which is the
practical roof and the honest one to optimise against. A utilisation quoted without saying which
denominator it used is not a number.

**Both denominators now exist for the MI210 (measured 2026-08-03).** Achievable is **1433.3 GB/s**,
87.5% of the 1638 GB/s datasheet peak — a correction factor of 1.143×, so fp16's 62.6%-of-spec is
71.5%-of-achievable. The ridge inherits it: 110.5 FLOP/byte spec-basis, 126.3 achievable-basis, the
second **mixed-basis** (spec FLOPS over measured bandwidth) because nobody has measured the matrix
units. Both fall inside the measured bf16 knee at B≈96–128, so that observation does not discriminate
between them, and claiming it does would read precision the data has not got.

**The usage rule matters more than either number, and it is not one I specified.** Converting *our*
figures to an achievable basis while a competitor's stay on a spec basis makes a gap look smaller
without it being smaller. **Cross-vendor comparison stays spec-to-spec** until someone measures the
other device's achievable bandwidth; achievable-basis figures are for reasoning about *our own*
headroom. Mixing bases across vendors is the same failure mode as quoting an unnormalised speedup,
one level subtler.

The MoE distinction is load-bearing here, not a detail — this stack serves several A3B/A4B MoE models,
where active-expert bytes rather than model size set the roof. Using total parameters would understate
utilisation severalfold and manufacture headroom that does not exist.

**What it buys, in three places:**

1. **A real headroom bound for DISCOVER.** At 85% utilisation a bandwidth-directed technique has at
   most ~18% left in it, regardless of how impressive the technique sounds. This is the wall-share
   ceiling argument applied one level down, at the memory system, and it is what stops the loop
   spending a campaign on a lever whose ceiling it never computed.
2. **A comparable reading of someone else's result.** An oracle reporting 40% → 70% utilisation
   describes a technique that is *available*. If we already measure 75% in that regime, the technique
   is already harvested here and the seed is `SUPERSEDED_FACT` before any source is read — decided from
   the paper, at zero compute.
3. **Two workloads, not one target (operator, 2026-08-03).** The quant-ladder finding is easy to read as
   "aim at the fp16 rung". It is really two distinct programs, and the second is the one that pays.
   *(a)* Lift the fp16 rung itself — the headroom above 71.5%-of-achievable, shared by every quant
   because it is the common GEMV/memory path. *(b)* **Close the ladder gap**, i.e. the collapse from
   fp16 down through Q8_0 → Q4_K → MoE-Q8 → MoE-IQ2. (b) is the higher-value direction here for a
   plain reason: **production does not serve fp16.** A win on the fp16 rung alone lands on a rung no
   production role occupies, so a campaign that optimises (a) and stops has moved a number nobody is
   served by. Campaigns declare which of the two they target, and a (a)-only result states explicitly
   which production rungs it did and did not reach.
4. **A phase signal (§8.4.1).** Utilisation approaching the practical roof across a campaign's regimes
   is the clearest evidence that HARVEST is finished and the remaining gains are not in this direction
   — a physical reason to switch to EXPLORE rather than an inferred one.

Utilisation is a **diagnostic and a routing input, never a gate**: it does not license, veto or
substitute for a release claim, and it is not a protocol-bound metric. It sits in the profile manifest
P0.1 (§19.8) compiles, per backend, per regime, alongside the per-op wall-share map.

### 8.4 PROPOSE and PRE_RUN_CRITIC

The planner produces proposal manifests; the controller filters schema/novelty/budget violations; the
critic tries to falsify the highest-value proposal. **A filtered proposal is journaled as
`PROPOSAL_SKIPPED` with its reason, fingerprinted, and fed into the next planning context** — never a
bare discard. AutoPilot dispatched 119 identical invalid actions whose rejection message named the
exact fix, and none of it ever reached the planner; a repeated fingerprint auto-blacklists and a run of
them trips `PLANNER_DEGRADED`. Cheap deterministic checks run **before** metered drafting, not after —
the reverse ordering cost roughly 38 draft-and-critique cycles that were paid for and then thrown
away. Selection ranks **expected information gain first**,
then expected performance value — a cheap experiment that cleanly distinguishes two mechanisms can
outrank a speculative large patch.

Rejected before mutation when: expected end-to-end gain exceeds its own wall-share ceiling without a
fusion explanation; no correctness oracle covers the affected path; target shapes do not occur in a
real graph and the campaign is not explicitly microkernel-only; the same mechanism was falsified under
matching conditions **by an entry carrying a receipt**; the resource or storage estimate exceeds
budget; the change crosses a repo/release domain the backend adapter does not own; or the proposed
evaluator step would require changing the evaluator.

### 8.4.0 Operator hypotheses — steering without authority (operator, 2026-08-03)

**Correction, 2026-08-03.** An earlier revision of this section stated that AutoPilot "tracks
hypotheses as still-open until resolved and re-surfaces the open set into each planning round", and
that AutoKernel merely lacked what AutoPilot already had. **That was wrong**, verified against the
source rather than assumed:

- `ExperimentJournal.unfalsified_hypotheses()` returns the **last five trustworthy trials carrying a
  hypothesis and a falsifier** — a *recency window*, not resolution tracking. Its own docstring says
  resolution-checking is *"intentionally minimal … presence of the falsifier string only"*. Nothing
  anywhere marks a hypothesis resolved.
- The `### Still-open hypotheses` block lives inside the exploration-rich template and is emitted
  **only when a stagnation signal fires**. The always-rendered block is `### Hypotheses Under Test`,
  which shows the last three trials and does not render falsifiers at all.
- The falsifier is **not mandatory**: the rationale sidecar is documented as observability-only, a
  missing block does not abort a trial, and the default is an empty string.
- AutoPilot has **no evidence-grade vocabulary whatsoever** — `design_prior`, `evidence_grade` and
  their kin return zero hits across its package. `design_prior` is AutoKernel's own construct, not an
  inherited one.
- There *is* a pre-existing operator inbound path this section missed: operator seed strategies reach
  the planner's hint block. But it carries no falsifier, no tracking, no resolution record and no
  blacklist check — **a hint channel, not a hypothesis channel.**

So neither loop has this mechanism, and AutoKernel is not catching up to a sibling; it is specifying
something new, and specifying it more strictly than the sibling's nearest equivalent. The rest of this
section stands as designed — the correction changes the provenance of the idea, not its content.

**The channel.** An operator states a hypothesis with its falsifier — "G15's elementwise/norm cluster
is where the B=128 decode time is, and fusing it lands ≥15%; if a current wall-share map shows the
cluster under 20% I am wrong". It enters the planner's context as a first-class prior, ranked with the
rest, and it is a *proposal source*, never an authority.

**The grade is what makes this safe.** An operator hypothesis enters at `design_prior` evidence grade
and **can never be promoted by its origin** — §19.0 rule 4 already forbids upgrading evidence on
import, and an operator hunch is exactly the input most likely to be treated as settled because of who
said it. It faces the pre-run critic unchanged, obeys every §8.4 rejection condition, and is subject
to the do-not-repeat ledger like any other proposal. If it repeats a receipted negative, the critic
says so; being the operator's idea is not new evidence.

**Still-open tracking, for every hypothesis regardless of origin.** Each carries a falsifier, stays
open until confirmed / refuted / inconclusive with the evidence that resolved it, and the open set is
re-surfaced into each planning round. This is the half AutoPilot actually has and AutoKernel lacked,
and it is what stops a hypothesis being silently dropped when its first proposal fails for an
unrelated reason — the failure mode that leaves a question feeling "already tried" without a receipt.

**What it must not become.** Not a queue-jumping mechanism, not a way to bypass the wall-share ceiling
or the correctness gates, and not a route to mark something resolved without evidence. An operator
hypothesis that the loop refutes is **refuted**, and the record says so — that is the mechanism
working, and it is the main reason to have it rather than steering out-of-band.

### 8.4.1 Architectural campaigns — how deep rethinking stays possible

Three of §8.4's rejection conditions are correct for incremental work and **wrong for architectural
work**, and left unqualified they would block exactly the kernel rethinking this loop exists to find.
Each is therefore replaced — not waived — inside a declared architectural campaign.

| §8.4 rejection | Why it misfires on deep work | Replaced by |
|---|---|---|
| Expected gain exceeds its own **wall-share ceiling** | An architectural change *redistributes* wall share rather than optimizing within it. Eliminating an intermediate materialization, or moving work off the critical path entirely, is bounded by no ceiling computed on today's profile. The narrow "fusion explanation" escape does not cover a new execution layout, a persistent team, or a different residency model | A **predicted post-change profile**: the proposal states what the wall-share distribution *becomes*, per op family. This is strictly more falsifiable than a ceiling test — it can be wrong in a way the profiler can see |
| **Target shapes do not occur in a real graph** | A new kernel may need shapes that do not occur *because the current implementation forces different ones*. Requiring the shape to pre-exist forbids changing what shapes exist | **Prospective shapes**, admissible when the proposal declares the mechanism by which they come to occur and a way to observe that they did |
| **One conceptual mutation** | A layout, its kernel, its dispatch predicate and its repack path are not independently correct or measurable. Forced apart, none can be proposed at all | One conceptual change per **step** of a declared lineage with a stated end-state (invariant 13) |

**Spikes.** An architectural campaign may open a **spike**: a deliberately incomplete prototype whose
only purpose is to measure whether the mechanism is real. A spike is an *experiment* in §8.9's sense —
it is never banked, never enters the champion, and never carries a correctness claim. It buys evidence
about a deep idea at a fraction of the cost of building it, and a refuted spike is a first-class
result that closes a direction with a receipt. Its output is a mechanism verdict, never a speed rank.

**Harvest and explore are phases, not a fixed ratio (operator, 2026-08-03).** A static budget
fraction was the first formulation and it is wrong: it spends explore budget while a freshly opened
region is still yielding, and it caps exploration exactly when the region is dead. The correct policy
is **phase-switched on marginal yield**.

- **HARVEST.** Entered whenever a deep lever lands, or the anchor moves. A large change typically
  unlocks a *cluster* of adjacent incremental wins — a new layout brings its tile sizes, dispatch
  thresholds and prefetch distances with it — and those are cheap, high-probability, and perishable
  (they rebase away at the next freeze). Incremental proposals take priority and the cheap tiers
  dominate: many T1, few T2. The goal is to strip the region efficiently before anything else.
- **EXPLORE.** Entered when marginal yield in the current region decays below a **derived** floor over
  a trailing window. Architectural proposals and spikes take priority; incremental work continues only
  where it is nearly free.

The switch signal is banked gain per unit of budget over a trailing window, compared against the same
measure earlier in the same harvest. Like every other threshold, the decay floor and window are
**derived by the campaign calibration procedure, never supplied** — a supplied number here would decide
the explore/exploit tradeoff by guess.

Two failure modes the switch must not have. It must not confuse a dead region with a broken planner:
falling yield with rising `PROPOSAL_SKIPPED` or repeated fingerprints is `PLANNER_DEGRADED` (§8.10), not
EXPLORE. And it must not oscillate: a phase change requires the signal to hold across the full window,
and each phase declares a minimum dwell so the loop cannot thrash between them.

**Spikes are cheap by construction, and must stay that way.** A spike produces a *mechanism verdict*,
not a rate claim, so it does not owe what a rate claim owes: no anchor gate, no paired blocks, no
e-process, no confirmation sample. It still holds a resource claim and passes preflight, because it
runs on shared hardware and contaminates its neighbours otherwise. Institutional cost is spent
confirming gains, not discovering them — if a spike ever costs what a T1 costs, this mechanism has
failed and the loop will stop using it.

**What does NOT relax.** Everything that decides whether a change is *admissible* stays exactly as it
is: source-integrity gates (§8.5.1), correctness precedence, no-fallback proof, determinism class,
declared-equals-traced scope, and the diff-complexity ceiling — which does not forbid a large diff, it
marks it `REQUIRES_HUMAN_CODE_REVIEW`. The `core_header` risk tier applies to most architectural work
by construction. Depth is bought with *more* evidence, never with weaker gates.

**Already available for deep work, and not to be duplicated:** hierarchy layers 7–9 (new kernel,
scheduler architecture, alternate engine) are in scope today and reachable by skipping cheaper layers
with an evidence receipt (§8.3); `oracle_port` (§6.5) covers importing a whole subsystem at the scale
the iqk port actually delivered; and §9.8 capability objectives cover a change that unlocks a workload
where a throughput ratio is undefined.

### 8.5 MUTATE and BUILD

The actor applies the conceptual change in the campaign worktree. The controller records preimage,
diff, tool calls, commit, and build outputs. Build must enforce: no production-tree path; no production
build directory; clean production ancestry; namespaced branch; pathspec-limited commits; bounded build
parallelism; campaign-local caches/output; compiler and dependency identity; exact output hashes; full
process cleanup on timeout/failure.

Compilation failures are valuable outcomes. They go to the journal and the correction prompt.

#### 8.5.1 Source-integrity gates — the C++ analogue of AutoPilot's four layers

AutoPilot's one attempt at autonomous source mutation destroyed a production module with an edit that
passed `ast.parse()`. The project's answer was four layers: syntax, a >60% shrinkage reject, public-name
preservation, and a live `importlib` round-trip. **None of those four transfer directly**, because
AutoKernel edits compiled C++/HIP rather than interpreted Python — there is no import round-trip, and
"it compiles" is a much weaker claim than "it imports". A kernel edit that drops a template
specialization, deletes a case from a dispatch switch, or removes an op registration compiles cleanly
and silently changes behaviour for every shape nobody happened to test.

Every candidate therefore passes these before it is eligible for T0's behavioural gates:

1. **Symbol and registration preservation.** Diff the exported symbol table, the op-registration
   tables, and the dispatch predicates between the anchor binary and the candidate binary. Any
   **removal or arity change not declared in the proposal** is a hard failure. This is the direct C++
   analogue of public-name preservation, and it is the check that would have caught the class of edit
   that destroyed `escalation.py`.
2. **Clean build from the recorded snapshot.** T0 compiles from the content-addressed source snapshot
   in a fresh build directory — never from the actor's incremental tree. An incremental build can link
   stale objects and hide the error that the snapshot would surface, which would make the actor's
   build state part of the artifact.
3. **Semantic diff conformance.** The diff must touch only files and symbols within the declared
   surface, contain no unrelated deletions, and stay inside the change-class size envelope. Invariant
   13 says one conceptual mutation; this is what enforces it rather than trusting it.
4. **Repair starts from a clean parent.** A bounded repair attempt re-checks out the parent candidate
   and re-applies, never continuing on the failed attempt's tree. Repairs are capped per proposal;
   exceeding the cap is a `PLANNER_DEGRADED` signal, not another retry. AutoPilot's scar here was a
   loop compounding edits onto an already-corrupted file.

**Core-header risk tier.** A change to shared ggml core or to a widely-included header is not a large
edit — it is a *different kind* of edit, because its reach is every op in both the CPU and GPU builds.
`change_class: core_header` forces full-tree affected surface regardless of the textual diff size,
forces the binary-comparison stage of §3.2 for every backend the tree serves, and marks the candidate
`REQUIRES_HUMAN_CODE_REVIEW` regardless of the §10.6 complexity ceiling.

**The candidate binary is the real privilege boundary, not the actor's shell.** §6.2 grants generic
shell inside the worktree, which a C++ build needs. But the loop then *compiles code it wrote and
executes it* with GPU access on a shared host. Path allowlists on the agent's tools do not constrain
the syscalls of the binary those tools produced. Candidate execution therefore runs under the same
sandbox and owned-cgroup regime as the evaluator, with no write access outside the campaign tree and
no ability to signal processes it does not own — and until that is verified on the real host (§14 AK2),
AutoKernel does not run unattended.

### 8.6 T0_GATE

Run on every source candidate:

- **the §8.5.1 source-integrity gates — symbol and registration preservation, clean build from the
  recorded snapshot, semantic diff conformance — which run before any behavioural check**;
- schema and diff policy, including the diff-complexity ceiling (§10.6);
- static/compile checks;
- **mandatory ASAN/UBSAN build and targeted run for any change touching memory or threading**;
- targeted backend-op unit shapes;
- exact reference comparisons where defined;
- unseen/boundary shapes for dispatch changes;
- dispatch trace for affected-surface confirmation (§6.4) and no-fallback proof;
- state, rollback, teardown, and race tests where relevant;
- determinism-class check: same seed, same input, repeated runs, bitwise stability versus anchor;
- binary/linkage identity; and
- anti-reward-hacking/integrity checks.

Any failure ends speed ranking for that candidate. A bounded repair is a new candidate record.

### 8.7 T1_SEARCH_EVAL

See §9.2–9.4. T1 emits a search verdict, not a release verdict, and never runs the production role
matrix or promotion-quality suite.

### 8.8 POST_RUN_CRITIC and memory update

Classifies hypothesis confirmed/refuted/inconclusive; mechanism confirmed/refuted/unavailable; speed
signal versus noise; wall-share translation from op to graph; target and non-target behaviour; likely
interaction with the current champion; next discriminating experiment; durable do-not-repeat lesson
**with its receipt**. The deterministic controller checks the classification against the raw gates. The
event, snapshot, and context update are fsynced before the next round.

### 8.9 Champion maintenance

Three concepts stay separate: **frontier candidates** (correct, non-dominated), **the champion**
(one compatible composed lineage per source tree), and **experiments** (diagnostic branches that may
never accumulate).

Only changes with reconciled affected-surface maps may be combined. After combining, rerun T0/T1 on the
combined full candidate; never infer composition by multiplying local speedups. Retain diversity across
mechanism classes so one noisy early win does not collapse the search to a single family.

When the operator freezes a new production version, the champion is **re-anchored**: members already in
production are dropped, the remainder is rebased on the new tip, and its T1/T2 evidence is invalidated
and re-measured. That cost is explicit and budgeted; it is the price of §4's fresh-production-base invariant.

**The anchor can also move without a freeze, and the loop must notice.** An emergency kernel hot-fix,
a rollback to the previous version, or any operator action that repoints a production symlink or
advances a frozen branch leaves every in-flight champion silently forked from a dead anchor, with T1
and T2 evidence that is stale in a way no gate would catch — every ratio in the journal has a
denominator that no longer exists.

Therefore the controller re-verifies **anchor identity at every campaign boundary**, not only at
freeze: production branch, commit, and the binary plus linkage hashes of every backend the tree serves,
compared against the values recorded at BOOTSTRAP. Any mismatch raises `ANCHOR_MOVED`, which:

1. halts new candidate work for that source tree immediately — no further measurement against a
   denominator that has changed;
2. marks every T1/T2 evidence record for the affected backends `superseded_by_anchor_move`, carrying
   the old and new anchor identities, rather than deleting them;
3. preserves candidate source, patches, and correctness results, which remain valid — only the
   *comparisons* died, not the work;
4. re-anchors per §8.9 and re-measures; and
5. emits an operator notice, because an unexpected anchor move usually means something happened that
   the loop should not silently absorb.

This is cheap to implement and closes the one way the champion model can be quietly wrong.

### 8.10 Stop and recovery

Deterministic stop states:

- `RELEASE_PACKAGE_READY` — package assembled and handed to the operator;
- `ANCHOR_MOVED` — production identity changed outside a loop-initiated freeze; comparisons are
  superseded, work is preserved, re-anchor and re-measure (§8.9);
- `PLATEAU_STOP` — no meaningful readiness improvement across the configured window. Emitting it
  requires the same enumeration `EXHAUSTED_SURFACE` does;
- `PLANNER_DEGRADED` — the controller is repeating no-ops, dispatching invalid actions, narrating a
  condition the receipts contradict, or looping against an unavailable dependency. Distinct from
  plateau: plateau means the search is done, degraded means the searcher is broken, and conflating
  them once cost this project months of paid no-ops;
- `OPERATOR_STOP_REQUESTED` — an operator control was received; the loop acknowledges in the journal,
  latches on disk, drains at the next boundary, and stays stopped across restart until resumed;
- `BUDGET_STOP` — wall/resource/candidate/token budget exhausted;
- `DISK_PRESSURE` — storage headroom below the campaign floor;
- `EXHAUSTED_SURFACE` — every eligible hierarchy layer measured or falsified. **Emitting it requires
  enumerating what was closed and what was not:** "closed for sub-scope X (gates A, B, C met);
  sub-scope Y deferred (gates D, E un-run)". Bare "exhausted" and "all paths" are reserved words the
  validator rejects — closure inflation is this project's most-repeated documented habit, surviving
  even explicit awareness of the rule;
- `EVALUATOR_COVERAGE_GAP` — release blocked for the affected lineage, research continues on covered
  surfaces. **It has an owner and a deadline, or it becomes a permanent silent block:** the gap is
  raised as a decision package naming the missing coverage class, the lineage it blocks, and the
  drafted amendment; if it is still open at the next campaign boundary it escalates, and a gap open
  across two consecutive freeze cycles is reported as a program-level defect rather than a standing
  condition;
- `RESOURCE_UNAVAILABLE` — persist and drain, never busy-wait;
- `HOST_REBOOT_REQUIRED` — no decision-grade measurement proceeds (§10.7);
- `INTEGRITY_STOP` — repeated tamper/reward-hacking signal;
- `OPERATOR_INPUT_REQUIRED` — rendered as a four-part decision package (§18).

The LLM may request a stop. The controller owns disposition from records.

---

## 9. Tiered evaluation and cost control

| Tier | Frequency | Purpose | Typical work | May do |
|---|---|---|---|---|
| **T0 — correctness/build** | Every source candidate | Eliminate broken artifacts before speed work | build, sanitizer, targeted ops, reference/unseen shapes, dispatch trace, state/rollback, determinism, linkage, integrity | admit to T1 or fail |
| **T1 — search** | Every T0-pass candidate or selected param point | Guide local search cheaply | microbench, tiny real graph, paired blocks, targeted profiler, sentinel shapes | rank/retain/abandon experimental candidates under `P-AK-SEARCH-1` |
| **T2 — lineage estimator** | Occasionally, on the composed champion | Estimate per-backend per-phase standing with bounded uncertainty | medium sentinel matrix, stronger reps, broader non-target set, co-residency cell, mechanism confirmation | update the readiness signal; never freeze |
| **T3 — kernel-freeze gate** | **On operator request only** | Decide whether the sealed champion is a releasable new kernel version | full derived tree scope, release reps, correctness/quality/stability/linkage/capacity, transaction dry-run | emit release PASS / FAIL / PASS_WITH_WAIVER bundle |
| **T4 — post-cutover verification** | After the operator's freeze and cutover | Confirm live activation and detect stale processes | role canary, health, API/speech smoke, linkage, process start-time check, multi-day watch window | recommend keep or rollback to the operator |

### 9.1 Cheap-evaluation rules

- Incremental compile and targeted tests are allowed; release builds are not done per candidate.
- Profile only the target kernels/counters needed to test the hypothesis.
- Prefer captured real shapes over broad synthetic grids.
- Cache model identity and immutable inputs; never cache a candidate output as a correctness oracle.
- Save raw outputs and samples so reducers can replay without inference.
- Promote from op-level to real-graph measurement only when the op result and measured wall share imply
  plausible end-to-end value.
- Promote from T1 to T2 only after the full composed champion — not one isolated patch — passes T0/T1.
- Run T3 once per sealed fingerprint. A retry requires a new evidence-affecting fingerprint or a
  deterministic replay/repair of the failed stage.

### 9.2 Statistical method — use the project's sanctioned machinery

The constitution never uses the term "LCB". What it sanctions is:

- **E-processes for rate claims.** `MEASUREMENT.md:30-32` — *"Noise reference CV ≈ 9.1%: all rate
  claims go through the non-inferiority / improvement e-process, never a single trial."*
  `gpu-cross-device.md:146-150` — *"n ≥ 10 paired blocks. **MDE computed and published WITH the result,
  not after seeing it.**"* E-processes are anytime-valid, which is exactly what a loop that looks at
  its evidence every round requires.
- **Pre-committed stopping rules.** `MEASUREMENT.md:136-137` — *"Pre-commit a stopping rule before any
  bench campaign."* `MEASUREMENT_POLICY.md:59-61` — name the table that is FINAL and the decision each
  outcome triggers.
- **Reps.** `bench-cpu.md:21-22` — *"≥5 for ≥5% effects; ≥10 for ≤2% effects; report median + MAD."*
  P-BENCH-PREFILL-1 requires ≥10 reps per arm for a release non-inferiority claim.
- **Bounded extension only.** `bench-cpu.md:85-86` — a ratio in [0.95, 0.98) buys *one* fresh
  reversed-order pair, pooled to a pre-declared threshold. Contrast P-BENCH-4
  (`bench-cpu.md:174-178`): *"exactly five … no retry, replace, discard, or pooling."*
- **Order control.** Arms interleaved and order-randomized (`gpu-cross-device.md:136-138`); reversed
  order on retry (`bench-cpu.md:48-49`).
- **Anchor gate.** `bench-cpu.md:231-233` — *"`np=1` is measured FIRST and compared against a recorded
  production anchor … Outside band ⇒ the run is VOID and may not be reported."*

`P-AK-SEARCH-1` therefore supplies: an e-process construction for every rate comparison; a
pre-committed stopping rule per campaign; per-tier reps consistent with the P-BENCH-1 rule; an anchor
gate so a drifted host voids rather than reports; and a **selection/confirmation split** so the
evidence that promotes a candidate into the champion is not the same evidence that reports readiness.
`architect_bench_analyze.py:54 bootstrap_ci()` is a usable seeded paired bootstrap for fixed samples but
does not address sequential looks.

Controls are four, not three (§15.2): positive, neutral, negative, and **A/A** (anchor versus anchor),
run periodically rather than once — A/A is what calibrates the false-positive rate and catches host
drift mid-campaign.

### 9.3 T1a — target operator discriminator

1. run only captured target shapes that occur in the selected real workload;
2. interleave candidate and anchor measurements in paired blocks;
3. begin at the protocol's minimum sample and extend only by the pre-committed rule;
4. apply successive halving for parameter surfaces — broad shallow pass, retain the best fraction, add
   samples only to survivors;
5. measure the exact predicted mechanism counters, not a whole-profiler sweep; and
6. stop early when the candidate cannot clear the campaign's contribution floor.

Argv is constructed by the codified microbenchmark recipe (§2.5), never hand-typed.

### 9.4 T1b — tiny real-graph translation, T1c — selective mechanism receipt

An operator win advances only when `operator wall share × operator gain` can make a meaningful graph
contribution. Then run one production-representative model/quant/shape that actually dispatches the
changed path; a bounded prompt/decode or prefill slice; counterbalanced candidate/anchor samples; a
deterministic output, numerical, PPL, or state sentinel appropriate to the change; and one or a few
non-target sentinels around the dispatcher boundary.

T1c runs only when the proposal predicts a counter movement, T1a/T1b shows a bankable signal, or the
result is ambiguous between two mechanisms. Capture the target kernel and named counters only. A failed
mechanism prediction withholds the mechanism bonus and normally makes the result inconclusive until the
mismatch is explained.

### 9.5 Change-class-specific cheap suites

Selected deterministically from `proposal.change_class`.

| Change class | Cheap correctness | Cheap performance/mechanism | Mandatory sentinel |
|---|---|---|---|
| Parameter/autotune | existing op tests; configuration bounds | successive-halving microbench | neighbouring shapes and incumbent default |
| Dispatcher | target, boundary, unseen shapes; path trace | per-path microbench + branch/launch trace | known positive and known negative cells |
| Arithmetic kernel | reference/random/adversarial shapes; numeric margin | target op paired A/B + predicted counter | alternate quant/dtype and awkward dimensions |
| Layout/repack | encode/decode or load-time equivalence; metadata validation | kernel A/B plus load time, VRAM/RAM, cache-line traffic | tiny real graph and context-capacity budget |
| Fusion/graph | exact intermediate/final result; alias/lifetime checks | node/launch/barrier delta + tiny real graph | unfused shapes and variable-shape path |
| MoE scheduling | route/scatter/reduce correctness, skew cases | expert histogram, occupancy/reuse, batched graph | batch one and saturated batch |
| Recurrent/stateful | committed/speculative state and rejection rollback | bounded decode/prefill sequence + state traffic | rejection, transition, and concurrency sequence |
| Scheduler/policy | deterministic event simulation and invariant checks | short variable-arrival replay | latency/SLO and completion-churn cases |
| Oracle port | full reference parity against the oracle's own tests plus ours | same as the underlying change class | the oracle's own negative cases |

### 9.6 Banking a result

T1 decides whether to reject, retain as diagnostic, bank as a correct conditional change, extend
sampling, or compose into the champion. Banking requires a correct real-path dispatch, green sentinels,
and either the predicted mechanism or a recorded explanation, plus **either** a throughput signal above
the calibrated floor **or** a non-dominated improvement on another banked axis: context capacity, VRAM
or RAM headroom, model load time, or run-to-run variance.

Single-signal banking would filter a throughput-neutral capacity win before anyone could see it — and
capacity is what makes the large models on this host runnable at all. AutoPilot learned both halves of
this: a single-axis noise gate discarded 8 of 11 excluded trials that were in fact non-dominated, *and*
the naive fix of admitting everything poisoned the frontier with speed noise. The working form is one
robust-median representative per stable configuration fingerprint, with dominance judged on the median
rather than on any single run. Small correct changes are retained even when they cannot move the
readiness signal alone.

### 9.7 T2 — composed-champion estimator

Runs on the **composed champion**, never by adding local percentages. Trigger when compatible winners
have accumulated and interaction is the dominant uncertainty, when the readiness signal could plausibly
change materially, or when a predeclared capability objective becomes runnable.

T2 uses a medium, production-weighted sentinel matrix: one or a few roles per affected
architecture/regime; stronger paired repetitions than T1; broader dispatcher-boundary and non-target
sentinels; **at least one co-resident cell** for `llama_cpu`; capacity (VRAM/RAM/context) deltas;
cumulative mechanism confirmation; champion versus the sealed production anchor. It omits the full
role/model matrix, broad quality suite, long stability soak, release build, transaction dry-run, and
canary.

### 9.8 Capability objectives

Some changes unlock a workload that previously could not run, so a throughput ratio is undefined. They
enter the readiness signal only through a predeclared capability objective: the required model/role
becomes runnable at the declared context/concurrency; correctness and quality floors pass; resource
budget fits; and the utility model was fixed at campaign start, not invented after observing the
candidate.

---

## 10. The kernel-freeze evaluation program (T3)

Run when the operator requests a freeze for a source tree.

### 10.1 One generated plan, not a curated matrix

The release plan compiler starts from compiled stack priors, then joins: source tree and the backends it
serves; stable production kernel paths; distinct production models and roles; quant, context, KV,
speculation, concurrency, placement, and **co-residency** recipes; architecture class; observed op/shape
coverage; the reconciled affected-surface manifest; backend-specific protocol IDs and thresholds;
correctness/quality transfer eligibility; capacity floors; linkage/toolchain requirements; and canary
requirements. It deduplicates equivalent cells without losing which roles they protect.

`kernel_freeze_scope.py` is the seed for this compiler, not the finished compiler.

### 10.2 Release-gate phases

1. **Identity/preflight** — immutable production and sealed-candidate identity, clean ancestry,
   evaluator hash, host health, resources, storage, rollback target, active waivers. **Includes the
   per-backend unchanged test (§3.2):** for each backend the tree serves, run stage 1 (source-closure
   diff) and, when it reports unchanged, stage 2 (normalized comparison against an anchor rebuild in
   the candidate's environment). Confirmed unchanged **and** incumbent evidence still in scope ⇒ that
   backend's cells drop with a transfer receipt naming the incumbent artifacts and their hashes.
   Changed, or the two stages disagree ⇒ that backend owes full candidate-grade evidence under its own
   protocol, and a stage disagreement is additionally filed as a build-identity defect.
2. **Build/linkage** — full candidate build outside production; overlay present; binary/library hashes;
   ABI/backend inventory; correct per-tree `LD_LIBRARY_PATH` proven by
   the research repo's `scripts/utils/verify_ggml_linkage.sh` (it lives in
   **epyc-inference-research**, not epyc-root — CLAUDE.md cites it unqualified, same defect class as
   the durability validator's path in `MEASUREMENT.md:155`).
3. **Backend correctness** — exact and unseen op shapes, no silent fallback, NaN/numerical bounds,
   state/rollback, teardown/race, real-model coherence, determinism class.
4. **Performance matrix** — candidate versus production on derived production-optimal recipes, per
   phase, under the phase's own protocol and release reps, including co-resident cells.
5. **Quality** — deterministic output parity where expected; otherwise PPL/numerical and focused quality
   parity. Transfer banked quality across kernel eras once paired parity proves transfer.
6. **Stability** — repeated load/unload, concurrency/mixed prefill-decode where affected, memory growth,
   profiler/runtime errors, cleanup.
7. **Capacity and utility** — VRAM/RAM/context-capacity non-inferiority; every protected cell within its
   fixed floor; the §1.6 per-phase rule satisfied, or an operator-approved phase-trade exception
   present.
8. **Transaction dry-run** — exact next version, branch/tag, install path, archive/rollback link,
   symlink diff, service impact, era actions, receipt paths.
9. **Seal verdict** — hash the protocol, plan, raw evidence, reducers, validation results, active
   waivers, and the exact transaction into one release bundle.

### 10.3 What to reuse

P-BENCH-1 and P-BENCH-PREFILL-1 for llama CPU; P-GPU-1 for MI210 after §3.2's sealed-candidate repair;
backend-specific STT/TTS protocols to be defined; deterministic replay and quality-transfer rules from
the constitution; the research repo's `verify_ggml_linkage.sh`; derived stack priors and stable
backend paths; and
transaction/rollback patterns from the v8 and speech freeze artifacts.

**Do not reuse blindly:** `kernel_eval.sh` raw commands as a release instrument; the v8 script's
hard-coded artifact list; observation-only historical speed numbers; a hand-curated "important models"
list; GPU idle sensing as a device claim; name-pattern process signalling; a baseline/off-recipe
regression as a production veto; or quality reruns when deterministic replay or proven era transfer
answers the question.

### 10.4 Operator waivers are a first-class input

A binary PASS/FAIL gate would have blocked v8. The v8 ratification records
`promotion_decision: false`, preserved as *"a non-automatic matrix verdict"*, released as *"an
operator-attested release decision"* with `q8_claim: "none; campaign-scoped WAIVE-Q8 remains binding
and v8 makes no Q8 non-regression claim"*. The waiver is a schema'd object —
`epyc.cpu_prefill_v8.operator_waiver.v1` — carrying decision, protocol, protocol-changed flag,
candidate/production heads, an exact scope (excluded model, excluded pairs, remaining matched pairs), a
reason, and a `consequences` list naming the forfeited claims. `freeze_v8_production_20260725.sh` gates
on it at `:214`, `:248`, and `:268`.

Generalize it: `epyc.autokernel.operator_waiver.v1`, human-only, stored under the trust-boundary path
set, hash-pinned into the T3 bundle, carrying scope, reason, forfeited claims, protocol binding,
campaign binding, and an expiry/reopen predicate. T3 emits `PASS` / `FAIL` / `PASS_WITH_WAIVER`. A
waived cell suppresses the corresponding claim in the release receipt. The evaluator verifies the
waiver's hash and predicate; it never judges its merits.

**Calibration note for AK5:** the T3 dry-run against preserved v8 artifacts should *predict a FAIL*
without the waiver. If it passes, the compiler is wrong.

### 10.5 Incumbent builds are archived, not merely rebuildable

The v8 quality gate compared against a preserved binary
(`/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server`). Rebuilding an old commit
under a drifted toolchain does not reproduce that binary. `/mnt/raid0/llm/kernels/archive/` is empty.
The freeze transaction must archive the incumbent's built binaries and linked libraries for N−1 and
ideally N−2, budgeted under §5.8.

### 10.6 Diff-complexity ceiling

Even without delegated authority, LLM-authored kernel C++/HIP should not reach a release package
unreviewed at arbitrary size. Each backend adapter declares a complexity/blast-radius ceiling (diff
size, files touched, whether shared ggml core is modified). Above it, the package is marked
`REQUIRES_HUMAN_CODE_REVIEW` and says so on its first page.

### 10.7 Host-uptime ceiling

`bench-cpu.md:17-19` and `OPERATING_CONSTRAINTS.md:39` set the host-health tier: uptime ≤1 week
requires `drop_caches` plus NUMA-interleave re-warm; ≥1 week requires a reboot. `drop_caches` is
privileged and *"a failed cache action invalidates the arm"* (`bench-cpu.md:148-149`). Reboots are
operator-only, and `SESSION_LIFECYCLE.md:43-55` makes pre-reboot wrap-up mandatory including commit and
push. **Any decision-grade unattended campaign is therefore capped at roughly one week of host
uptime.** The loop must request the reboot as a decision package, persist fully, and resume.

---

## 11. From champion to production

### 11.1 Four words that must stay distinct

- **Candidate:** mutable experimental lineage.
- **Champion:** the composed, always-green best lineage per source tree.
- **Sealed release candidate:** immutable full build plus evidence target.
- **Frozen version and cutover:** operator actions.

### 11.2 The release packager

On an operator freeze request the packager may: seal the champion; run T3 through the trusted
evaluator; assemble the verdict bundle; compute the next version and full transaction plan; compute the
rollback plan and verify the archive target; draft the era-registry row and the AutoPilot rebaseline
note; pre-validate every operator command end-to-end (`MEASUREMENT.md:138-145` requires exactly this);
and present a four-part decision package.

It may not: edit source; rebuild the candidate outside the sealed build; change protocols, thresholds,
or scope; waive failed evidence; touch any production branch, symlink, era registry, or baseline file;
or execute any command it drafted.

### 11.3 Cutover is scheduled by whoever owns inference

`OPERATING_CONSTRAINTS.md:41` — a reload *"must be executed BY THAT SESSION, at a moment it chooses…
route the request via coordinator-agent to the owning session"*. The package therefore contains a
cutover *request*, routed on the bus, never an action.

### 11.4 Cross-system effects after the operator freezes

A new llama kernel changes orchestrator speed priors and AutoPilot's speed era even when model quality
is identical. The package must therefore enumerate, as drafts for operator execution: the kernel-era
event and its registry row; which throughput priors to invalidate or rederive; which quality evidence
transfers; targeted re-anchors instead of whole eval regeneration; derived stack-fact recompilation;
and the rollback identity exposed to the orchestrator. Scheduler/runtime campaigns use the stack-change
adapter and must not impersonate a kernel-era freeze.

### 11.5 Post-cutover watch window

T4 validates activation — the right binary is live, linkage is correct, no process is stale, the role
canary and API/speech smoke pass. It answers "did the cutover work", which is not the same question as
"was the cutover right". A kernel regression that survives T3's matrix and T4's canary is by
construction one that only a production workload and a longer horizon expose. The watch window is
where that surfaces, and now that the operator performs the cutover it is the last automatic safety
net in the path.

**Duration.** Default 7 days of production traffic, or until the affected roles have each served a
declared minimum volume — whichever is later. A window that expires on a quiet weekend has observed
nothing.

**What is compared.** Live telemetry against the pre-cutover era, per affected role:

| Signal | Source | Direction |
|---|---|---|
| decode and prefill throughput at production recipes | orchestrator/serving telemetry | regression is the alarm |
| per-request p50/p95 latency | serving telemetry | regression is the alarm |
| error, timeout, and fallback rates | server logs and backend receipts | any increase is the alarm |
| memory growth and VRAM/RAM headroom over the window | host and device sampling | drift toward the residency floor is the alarm |
| quality proxies already collected on production traffic | evidence plane | regression is the alarm |
| crash, restart, and stale-process events | supervisor | any occurrence is the alarm |

**Comparison method.** Era-labelled, as `MEASUREMENT.md:233` requires — the pre-cutover baseline is
the incumbent era's recorded distribution, not a remembered number. Production telemetry is
observational and uncontrolled, so the window uses the standing noise reference and MDE rather than
pretending to protocol grade: it produces **a recommendation, never a claim**. A signal outside its
band raises a decision package; it does not itself revert anything.

**Thresholds and ownership.** Bands are set per role at package assembly time from the incumbent era's
observed distribution, and named in the package so they are fixed before the window opens rather than
chosen after seeing the data. The window is owned by whoever executed the cutover; AutoKernel computes
and reports, and the rollback anchor stays live and verified for the whole window. Closing the window
is an explicit action that records the verdict — an unclosed window is an open question, not a pass.

### 11.6 The stack-change release path (`serving_runtime`)

Scheduler, batching, admission, and KV-policy work shares the research loop but must not travel the
kernel-freeze path (§1.5, AK-D9). Its release path was previously named but never described; this
specifies it.

**What already exists.** `epyc-orchestrator/scripts/validate/stack_change_guard.py` validates generated
stack priors against current source artifacts — registry, descriptors, derived priors, and a manifest
of hardcoded consumer surfaces, with an exceptions file and accepted-gaps list. That is the
config-consistency half, and it is real. The behavioural half does not exist.

**The three gates, which are distinct and none implies the next.** A change is released only when all
three pass, in order:

1. **Pipeline green** — `stack_change_guard.py` passes: derived priors are consistent with source,
   no retired role is live, no consumer surface is stale.
2. **The stack starts** — every affected service comes up under `orchestrator_stack.py`, sequentially,
   with correct per-tree `LD_LIBRARY_PATH` proven by the linkage verifier.
3. **Live equals config** — the running processes match the intended configuration: right binary,
   right flags, right affinity, verified against live state rather than against the topology hash or
   the config file that was supposed to produce it.

**What `serving_runtime` adds on top.** A serving change alters *how work is scheduled*, not how a
kernel computes, so its evidence differs from a kernel release in three ways:

- the metric is **`task_rate`, not tokens/s** — `MEASUREMENT.md:23-30` makes these authoritative in
  their own scopes and forbids substituting one for the other, and a scheduler change is exactly the
  case where tokens are not commensurable across arms;
- the workload is **variable-arrival replay**, not fixed-shape benchmarking (the P0.2 separation in
  §19.6), with latency and SLO cells as first-class outputs rather than secondary;
- the comparison is against the **pinned production configuration** at its production-optimal recipe,
  and a reload gates on that pinned bench — autopilot being down is not the gate.

**What it does not touch.** No new kernel version, no frozen branch, no era row for kernel speed. A
serving change may still invalidate throughput priors and therefore require an AutoPilot rebaseline,
which remains an operator action. The `serving_runtime` adapter MUST refuse the kernel-freeze path
outright rather than degrading to it (§12).

---

## 12. Failure and abuse model

| Failure mode | Required defense |
|---|---|
| Candidate edits evaluator/tests | evaluator read from a hash-pinned bundle outside the actor's worktree, protected at the OS level (§3.6); resume drift fails closed |
| Candidate returns fake timing/score | trusted evaluator independently times and reduces an immutable snapshot |
| **Candidate under-declares its affected surface to shrink release scope** | scope is mechanically derived and dynamically traced; `traced ⊄ derived` is a hard failure; the declaration is a scored prediction only (§6.4) |
| Candidate detects only test shapes | unseen/boundary shapes, captured real-shape holdout, dispatch audit, anti-cheating red-team |
| Candidate silently falls back to CPU/reference | backend trace/op receipt and no-fallback assertions |
| Faster but numerically wrong | correctness/quality/state gates before any speed ranking |
| **Candidate silently changes determinism class** | same-seed repeat-run bitwise check in T0; class change is a declared release property (§4, invariant 12) |
| **Profiler, timer, or host-state instrument is altered, missing, or replaced by candidate-controlled evidence** | emit `INSTRUMENT_TAMPERED`, invalidate the measurement (not the candidate), stop performance ranking, preserve the source result, and require a trusted recapture before resumption |
| Optional baseline lets "coherent" pass | T1+ requires an explicit immutable anchor; no-baseline is `INVALID`, never correct. **Already realized in the current scaffold — see §2 and §14 AK1 quarantine task** |
| PPL/output drift hidden by tail compare | full bounded output hashes/vectors plus task-appropriate PPL/numerical checks |
| Warm-cache/repeated-prompt gaming | distinct prompts, cache-state receipts, fixed budgets, cache-disabled direct-kernel cells |
| Summed local gains inflate readiness | compare the composed champion directly with the sealed anchor |
| Profiler metric moves but wall time does not | mechanism bonus never substitutes for real graph gain |
| Op microbenchmark wins on unused shapes | captured real-shape and dispatch-coverage requirement |
| **Peeking at a threshold every round inflates false positives** | anytime-valid e-process, pre-committed stopping rule, selection/confirmation split (§9.2) |
| **Host drift creates a false win** | A/A control, anchor gate that VOIDs, counterbalanced pairs, host-health tier, contamination invalidation |
| Stale production base | ancestry guard at campaign start and seal; re-anchor at freeze (§8.9) |
| **Anchor moves outside a freeze (hot-fix, rollback) and every ratio loses its denominator** | anchor identity re-verified at every campaign boundary; `ANCHOR_MOVED` supersedes comparisons, preserves source and correctness results, forces re-anchor, and notifies the operator (§8.9) |
| Long-lived branch accumulates obsolete code | champion re-anchoring, campaign expiry, worktree reclamation |
| Store loses negative history on rewind | append-only sharded journal; supersession events; rebuild derived views |
| Resource collision poisons results | region/device claim held for the full run; host/resource receipts |
| **No GPU claim exists to hold** | build one (§2.5, §14 AK2); until then no GPU T1 runs |
| Loop kills another session | cgroup/PID ownership; quiesce-and-drain; single audited read-only preflight (§3.5) |
| Hung build/bench leaks process | owned cgroup timeout, TERM→KILL escalation, verified empty before the next event |
| **Disk exhaustion halts or corrupts a campaign** | quota, retention classes, tombstoned expiry, `DISK_PRESSURE` (§5.8) |
| Evaluator coverage lags new op/model | `EVALUATOR_COVERAGE_GAP`; release blocked; amendment drafted separately |
| Full release evaluation loops repeatedly | sealed-fingerprint idempotence and failed-gate cooldown |
| Package describes the wrong artifact | source/binary/evidence/transaction hashes joined; no rebuild in the packager |
| Cutover serves a stale process | process start time versus binary/link target; T4 live identity |
| Bad release degrades the live stack | incumbent archive pointer, canary, watch window, transactional rollback |
| **Adversarial or external content steers the planner** | external content rendered only in provenance-tagged quarantine blocks, never in an instruction position (`OPERATING_CONSTRAINTS.md:27-31`); imported priors carry source hashes |
| **A wrong suppression silently closes a research family** | every `HARD_CONSTRAINT`/`MATCHED_NEGATIVE`/`SUPERSEDED_FACT` needs a receipt bound to the current production commit, re-verified on anchor move (§19.3) |
| Alternate engine/scheduler change crosses domain | backend adapter refuses the kernel-freeze path and routes to the stack-change gate |
| **A syntactically valid edit silently deletes a specialization, dispatch case, or op registration** | symbol/registration-table diff against the anchor binary; an undeclared removal or arity change is a hard failure (§8.5.1) — the direct precedent is AutoPilot destroying `escalation.py` with an edit that parsed cleanly |
| **The compiled candidate writes or signals outside its campaign scope at run time** | candidate execution runs under the evaluator's sandbox and owned-cgroup regime; the actor's tool allowlist does not constrain the binary those tools produced (§8.5.1) |
| **A repair compounds edits onto an already-broken tree** | repair re-checks out the clean parent and re-applies; repairs are capped per proposal and exceeding the cap raises `PLANNER_DEGRADED` (§8.5.1) |
| **A core-header change is treated as a small edit because its diff is small** | `change_class: core_header` forces full-tree surface, per-backend binary comparison, and human review regardless of diff size (§8.5.1) |
| **An operator control is silently ignored** | controls acknowledged in the journal, latched on disk, re-read each iteration under the write lock; an unacked control is a hard failure; fault-injected in §15.1 (invariant 19) |
| **The planner regenerates a false belief from its own prose** | narrative separated from the machine record and excluded from retrieval by default; `RETRIEVAL_SUPERSEDED` removes a belief from retrieval without deleting the record (§5.5, invariant 20) |
| **The loop is broken but looks merely plateaued** | `PLANNER_DEGRADED` distinct from `PLATEAU_STOP`; filtered proposals journaled as `PROPOSAL_SKIPPED`, fingerprinted, and fed back rather than discarded |
| **Budget consumed by a proposal class that structurally cannot bank** | `realized_cost` attributed per proposal; discovery statistics reported per §18; cheap deterministic checks run before metered drafting |
| **The gate can still reject but can no longer promote, and reads as "exhausted"** | fourth control is a historical-win replay that MUST promote (§15.2); `EXHAUSTED_SURFACE` requires sub-scope enumeration |
| **A full-machine threshold applied to a partial-machine cell** | every evaluation event declares its `scope_denominator`; a gate whose scope exceeds the cell's refuses rather than demoting (§7.4) |
| **Ambient import identity decides which code scored the run** | evaluator emits a runtime source-label attestation of the modules actually loaded across all three repos, bound into the evaluation event (§5.4) |
| **The holdout is overfit because it never rotates** | captured-shape holdout and control seeds rotate on a declared schedule; a never-rotated holdout is an evaluator coverage defect |
| **A ledger entry contradicts a live operator decision** | contradiction detection at compile time against live decisions and sibling entries; a contradicting entry becomes `conflicted`, never authoritative (§19.2) |

---

## 13. Backend adapter responsibilities

### 13.1 `llama_gpu`
gfx90a build/toolchain/linkage; HIP op/reference/unseen-shape tests; full residency/fallback detection;
rocprof/rocprof-compute support with counter-availability declaration; batch-one, batched, prefill,
decode, context, speculation, and state/rollback recipes; P-GPU search/release adapters; MI210 device
claim, thermal, VRAM, and process receipts. Shares the llama champion with `llama_cpu`.

### 13.2 `llama_cpu`
Exact EPYC topology and compiler/libomp linkage; CPU op/reference/race/teardown tests; perf-counter
preflight and NUMA/locality evidence; canonical decode/prefill/batched/placement recipe adapters;
**co-resident lineup cells**; CPU-region claims, cache preparation, host-health requirements;
multi-role non-inferiority rules. Shares the llama champion with `llama_gpu`.

### 13.3 `whisper_stt`
Independent tree and ggml generation; STT correctness corpus and output normalization; RTF/latency/
throughput and memory stability; audio input identity; linkage and service smoke; backend-specific
release protocol.

### 13.4 `qwentts_tts`
Independent tree and ggml generation; text/audio identity and deterministic/numerical checks;
intelligibility/quality proxy and human-independent safety floor; first-audio latency, RTF, throughput,
stability; linkage and service smoke; backend-specific release protocol. Note the production symlink
points at `build`, not `build/bin`.

### 13.5 `serving_runtime`
Fixed-shape kernel versus continuous-serving separation; scheduler, KV, batching, admission,
graph-cache and latency objectives; orchestrator/llama-server worktree ownership mapping; production
role/canary scope; task-rate versus tokens/s metric discipline per `MEASUREMENT.md:23-30`. Its release
path is the three-gate stack-change path specified in **§11.6**, built on
`stack_change_guard.py` — not the kernel-freeze path, which the adapter MUST refuse rather than
degrade to.

---

## 14. Implementation sequence

Front-loaded with contracts and negative controls. Do not start a live autonomous campaign because a
planner prompt exists.

### 14.0 Sizing, and the slice that pays for itself

**Unit and assumptions.** Sizing is in *focused sessions* of the kind that produced the v8 and speech
freeze work — agent-driven, long-horizon, one coherent deliverable per session. Bands are planning
aids, not commitments; they assume no parallel campaigns, no host contention, and that the operator
reviews at phase boundaries rather than continuously. LOC bands count implementation plus tests.

| Phase | Deliverables | LOC band | Needs inference/GPU | Sessions |
|---|---|---|---|---|
| AK0 | 4 remaining drafts + ledger + attestation prep | ~0 (prose) | no | 2–3 + operator review cycles |
| AK1 | 7 schemas, sharded journal, corpus importer, storage plane, validators, quarantine, durability fixes | 3–5k | no | 8–12 |
| AK2 | worktree managers, build layout, **GPU device claim**, preflight wrapper, cgroup control, bus registration | 2–3k | claim-contention tests only | 5–8 |
| AK3 | evaluator API, affected-surface derivation + trace, correctness surfaces, ASAN path, codified microbench recipe, e-process reducer, red-team, four controls | 3–4k | **yes — first phase that does** | 8–12 |
| AK4 | state machine, context compiler, planner/critic adapters, selection, composition, guards | 2–3k | controller tokens | 6–10 |
| AK5 | T2 scope/weights, readiness, release-plan compiler, T3 runner, waiver verification, v8 dry-run | 2–3k | yes | 6–10 |
| AK6 | packager, refusal tests, fault injection, dashboard contract | 1–2k | no | 4–6 |
| AK7 | first supervised freeze | ~0 | yes, plus a real freeze window | 1–2 + operator time |
| AK8 | serving-runtime adapter, research queue activation | 1–2k | yes | 4–6 |
| AK9 | speech protocols and adapters (see below) | 2–3k | yes | 6–10 |

**Roughly 50–75 focused sessions end to end.** That is a program, not a task, and it should be
budgeted as one. The corpus import in AK1 and the control calibration in AK3 are the two most likely
to overrun, because both are bounded by how much project history exists rather than by how much code
gets written.

#### The minimum viable slice — AK1 + AK2 + AK3

**AK1–AK3 is a standalone deliverable, and it should be planned as the first release of AutoKernel
rather than as scaffolding for AK4.** It costs 21–32 sessions and delivers, without any autonomous
planner:

- a **trusted tiered evaluator** that requires an explicit immutable anchor, gates on coherence,
  proves no-fallback by dispatch trace, runs the sanitizer path, and cannot be tampered with by the
  thing it measures — replacing a 230-line script that today emits `"status":"OK"` unconditionally;
- a **durable event journal** with the project's kernel/inference research imported as typed priors,
  seeds, and a receipted negative ledger, so a human session can ask "has this been tried, under what
  regime, and what happened" and get an answer with citations;
- **safe worktree, build, resource, and process discipline**, including the cross-process GPU device
  claim that does not exist anywhere today and that every future GPU benchmark on this host wants
  regardless of AutoKernel.

Every one of those is immediately usable by the human-driven kernel sessions that have been running
the MI210 and CPU campaigns by hand — which is the same loop AutoKernel automates. If AK4 onward is
never built, the project still keeps a better evaluator, a searchable research memory, and a GPU
claim. Name this slice explicitly in planning so it can succeed on its own terms.

**Kill criterion for the program.** If, after AK3 plus two full campaigns, the loop has produced no
bankable correct change and no retired research family, stop and reassess rather than continuing into
AK4. The evaluator and journal remain valuable independently; the planner is the part that has to
earn its keep.

#### Operator moments, and how they batch

Every signature costs a context switch, and the constitution's own doctrine is that the human signs
**once, at apply time, over a consolidated bundle** — protocol plus evidence hashes plus validation
results plus the exact state diff (`MEASUREMENT.md:138-145`), presented as one attestation listing
each item so lines can be struck (`MEASUREMENT_POLICY.md:77-78`). The phase order is therefore
arranged so that operator moments fall on phase boundaries and are as few as the trust boundaries
allow.

**There are exactly three, and every other phase requires none.** AK1, AK2, AK4, AK6, AK8, and AK9
need no operator signature at all; nobody should insert an approval gate into them.

| Moment | When | What it carries | Its validation evidence |
|---|---|---|---|
| **Attestation 1 — search authorization** | after AK3 | Annex K creation + core-file layout/registry deltas, `P-AK-SEARCH-1`, the `pgrep` substitute, the evidence-retention rule, the `human_only_paths.yaml` additions and its `.sha256` rewrite | AK1–AK3 deliverables plus the four calibrated controls |
| **Attestation 2 — release authorization** | after AK5, before the first freeze | P-GPU-1 sealed-candidate amendment, `epyc.autokernel.operator_waiver.v1` | **the AK5 dry-run against preserved v8 and speech artifacts** — already planned, so this attestation costs no new compute |
| **A freeze** | AK7, and each freeze after | see below | the T3 bundle |

**A freeze is one apply bundle, not four ceremonies.** This is the structural change worth making.
A kernel freeze crosses four human-only boundaries (§1.3) — the freeze/cutover transaction, the
era-registry rows, the AutoPilot baseline apply, and any touch of the pinned path list. Left
unstructured, that is four artifacts the operator assembles and sequences by hand, every time. The
release package (§7.6, §11.2) instead assembles **one pre-validated apply bundle covering all four**,
presented as a single attestation with strikeable lines and a single apply token. On a validation
failure the same token is re-presented with updated hashes rather than restarting the chain, exactly
as `MEASUREMENT.md:138-145` prescribes.

This turns a recurring four-ceremony cost into one, and it is the difference between a freeze being an
afternoon and a freeze being a morning of assembling artifacts.

**Two ordering constraints that cannot be batched away.**

1. **Attestation 2 must precede the first real T3 run**, not accompany it. T3's evidence is only
   decision-grade if produced *under* a ratified protocol, so ratifying alongside the package it
   validated would be circular. The AK5 dry-run exists precisely so attestation 2 has validation
   evidence without a live T3.
2. **The `human_only_paths.yaml` edit and its `.sha256` rewrite are one item, not two.**
   `config.yaml:164` sets `on_pin_mismatch: refuse`, so the pair must land together or the bus
   refuses. Present them as a single strikeable line.

**Build the bundle assembler once, use it three times.** Pre-validation — running every operator
command in dry-run and producing the exact state diff before presentation — is a build task, not a
ceremony task. AK6's packager needs it for freezes; AK0's two attestations need the same thing. Write
one assembler in AK6 and have AK0's attestations use it, rather than hand-assembling the policy
bundles and then building the machine separately.

**Batch the non-boundary decisions too.** Operator decisions that are *not* trust-boundary writes —
the annex question, a phase-trade exception, an `EVALUATOR_COVERAGE_GAP` resolution, a reboot request
— accumulate to the next phase boundary and are presented together as decision packages, rather than
interrupting mid-phase. The only exception is an `INTEGRITY_STOP`, which surfaces immediately.

#### Freeze cadence — assumption, operator-settable

Sizing above assumes **freeze on readiness or quarterly, whichever comes first**. The tradeoff:

- *Longer cadence* accumulates more value per freeze and pays the freeze ceremony less often, but the
  champion drifts further from the anchor, the re-anchor cost at §8.9 grows, and more of the loop's
  work sits unreleased and therefore unvalidated in production.
- *Shorter cadence* keeps the champion close to production and exercises the release path — the
  highest-risk code — more often, but each freeze costs a T3 run, an era row, an AutoPilot rebaseline,
  and a cutover window.

Because a freeze is now operator-initiated, this is a policy you set rather than a parameter the loop
optimizes. It belongs in the campaign policy plane, not in a campaign manifest.

**Sequencing note.** AK1–AK3 produce no search decisions and need no new authority, so they proceed in
parallel with AK0's drafting — `MEASUREMENT.md:139` requires that *"evidence collection and validation
never wait on a human signature"* and `MEASUREMENT_POLICY.md:79-81` forbids gating unrelated work on a
pending token. The single ratification is presented once AK1–AK3 have produced the validation evidence
that makes the bundle reviewable.

### Phase AK0 — policy package (drafted in parallel with AK1–AK3)

- [x] Audit the source draft against the current scaffold, freeze path, measurement constitution, and
  production kernel-set rules ✅ 2026-08-01
- [x] Write this owning build handoff and supersession map ✅ 2026-08-01
- [x] Full design audit; corrections folded into §§2–15; freeze authority withdrawn ✅ 2026-08-02
- [x] Move the source draft out of gitignored `tmp/` to a durable tracked path ✅ 2026-08-02
- [x] Establish the staging area with its sequencing plan —
  [`artifacts/operator/autokernel-policy-draft/README.md`](../../artifacts/operator/autokernel-policy-draft/README.md),
  which carries the two-attestation split, the per-item blocking table, the bundle-contents checklist
  every attestation must satisfy, and the note on why filenames there omit their `.md` extension.
  `measurement/protocols/*` is hook-blocked, and the v2 restructure precedent is
  `artifacts/operator/measurement-v2-draft/` ✅ 2026-08-02
- [x] Draft the P-GPU-1 sealed-candidate amendment skeleton (§3.2), with every artifact-dependent
  binding marked `[BLOCKED-ON AKn]` and the corrected two-stage backend-unchanged test ✅ 2026-08-02 —
  [`P-GPU-1-sealed-candidate-amendment.draft.md`](../../artifacts/operator/autokernel-policy-draft/P-GPU-1-sealed-candidate-amendment.draft.md)

**Two attestations, not one (AK-D20).** `MEASUREMENT_POLICY.md:77-78` asks that queued boundary items
be batched into one attestation with strikeable lines. That guards against a per-experiment
ratification cycle; it does not require items whose referents appear months apart to share a
signature, and forcing them together would ratify release bindings against schema sketches under an
append-or-version constitution. Each attestation is presented only when every item's referent exists
and has been validated.

**Attestation 1 — search authorization (present after AK3).** Unblocks autonomous research; every
referent exists once AK1–AK3 land.

- [x] **Annex question resolved (operator, 2026-08-02): create Annex K (kernel research and release)**
  as a fourth annex under `measurement/protocols/`. `P-AK-SEARCH-1` fits none of the three declared
  families (`MEASUREMENT.md:15-20`) — it is cross-backend and a *search* instrument rather than a
  measurement family. Splitting it across B/Q/G would fragment one instrument across three amendment
  histories and obscure that its authority is narrow, unified, and revocable in one place. Annex
  creation is a layout change, so the layout paragraph and registry deltas ride inside attestation 1
  ✅ 2026-08-02
- [x] Draft Annex K plus the `P-AK-SEARCH-1` skeleton — scope, authority grant and its limits,
  preconditions, statistical requirements, record grammar, void conditions — with numeric bindings
  marked `[BLOCKED-ON AKn]` ✅ 2026-08-02 —
  [`Annex-K-kernel-research-and-release.draft.md`](../../artifacts/operator/autokernel-policy-draft/Annex-K-kernel-research-and-release.draft.md)
- [x] Fill `P-AK-SEARCH-1`'s numeric bindings from the four controls — e-process threshold, per-backend
  noise floor, minimum block counts. **These must be calibrated, not guessed:** ratifying invented
  thresholds into an append-or-version protocol is the mistake the draft-early/ratify-last sequencing
  exists to prevent. ✅ 2026-08-10 — superseded by the ratified Annex K calibration block: `φ`,
  `B_min`, α thresholds and MDE are derived per campaign under the fixed construction, never copied
  from guessed literals.
- [x] Draft the `pgrep` substitute (§3.5) as an equivalent P-BENCH-1/P-GPU-1 precondition. ✅ 2026-08-10
  — delivered as `preflight-substitute.draft.md` and realized by the claim-witness-first preflight.
- [x] Draft the evidence-retention rule for expirable classes (§5.8) — needed because
  `MEASUREMENT.md:223-229` puts reclamation under operator authority. ✅ 2026-08-10 — delivered as
  `evidence-retention.draft.md` and applied through the 2026-08-03 ratification package.
- [x] Prepare the `human_only_paths.yaml` additions (evaluator bundle, both protocol IDs,
  objective/threshold policy) and the accompanying `.sha256` rewrite as operator actions. ✅ 2026-08-10
  — `human-only-paths-delta.draft.md` and the live manifest record the boundary and the OS-level caveat.
- [x] Assemble attestation 1 through the AK6 bundle assembler: RATIFICATION_LEDGER of every semantic
  delta, the `MEASUREMENT.md` CHANGELOG line, the §2 registry key and row, and a pre-validated
  end-to-end command sequence presented as one attestation with strikeable lines. ✅ 2026-08-10 —
  Annex K / P-AK-SEARCH-1 were ratified 2026-08-03; the ledger and apply package remain in
  `artifacts/operator/autokernel-policy-draft/` as the durable assembly record.

**Attestation 2 — release authorization (present before the first freeze, after AK5).**

- [x] Fill the sealed-candidate amendment's `[BLOCKED-ON]` bindings from the delivered artifacts.
  ✅ 2026-08-11 — the complete draft converted all five former markers into artifact contracts;
  `P-GPU-1-sealed-candidate-amendment.draft.md` contains no unresolved normative binding. Presentation
  remains correctly gated on a real seal and the operator's attestation-2 action.
- [x] Draft `epyc.autokernel.operator_waiver.v1`, generalized from
  `epyc.cpu_prefill_v8.operator_waiver.v1` (§10.4). The draft schema suffices for AK5's dry-run; only
  a real freeze needs it ratified. ✅ 2026-08-10 — schema and validator are implemented in
  `schemas.py`; the preserved draft is `operator-waiver-schema.draft.md`.

**Both attestations must carry** the amended text appended to its owning annex (never a silent edit),
a one-line `MEASUREMENT.md` CHANGELOG entry, an explicit supersession naming the prior receipt path
and SHA-256, a `RATIFICATION_LEDGER.md` enumerating every semantic delta, a §2 protocol-registry row
per new ID, in-repo evidence hashes, a pre-validated end-to-end command sequence, and presentation as
one attestation listing each item separately.

- [x] File `worktree_manager.py`'s in-memory-restore data-loss bug (§2.3) as a defect against the
  AutoPilot owner with the exact repro. ✅ 2026-08-10 — subsequently fixed under the operator's
  expanded AP-WM dependency authority: rejection now restores the exact byte preimage (including
  dirty, empty, and absent paths) and never overwrites it with `git checkout`; 7 focused tests pass.

**Exit (attestation 1):** the search loop can legally learn from experimental results, and no freeze
authority has been delegated to any process.
**Exit (attestation 2):** the release gate can legally evaluate a sealed candidate, and the operator
can act on a waiver-bearing verdict.

### Phase AK1 — contracts, event substrate, and bootstrap corpus (no inference)

- [x] Implement versioned campaign/proposal/candidate/evaluation/champion/release-package schemas
  (§7), including claim grammar, anchor binding, `inconclusive`, `change_class`, controller provenance,
  expected information gain, determinism class, and co-residency. ✅ 2026-08-03 (schemas.py, 118 tests)
- [x] **Journal prospective live evaluation events from the executed campaign evidence.** ✅ 2026-08-12 —
  research `d96e8704` emits schema-valid v5 T0/T1 records before terminal STOP, binds raw paired
  blocks and the Vidya write-side capture, and retains the exact executed recipe receipt. Window
  closure now re-reads CPU/device claims, native open/close inference preflight receipts, host and
  storage state, evaluator/runtime authority, and the exact per-tool (`llama-cli`/`llama-bench`)
  anchor before release. Missing holder identity or anchor/evaluator/claim/host drift yields a durable
  INVALID/refusal rather than an inferred PASS; append failure remains terminal without skipping
  release or STOP. The final campaign/journal/evaluator/control/execution slice passed 1,231 tests.
- [x] **Bind source-changing proposals to an immutable source-to-candidate lifecycle.** ✅ 2026-08-12 —
  research `069e79fd` loads the embedded content-addressed patch before preflight or claim, restricts
  mutation to the guarded worktree API, verifies normalized paths plus content-derived hunk/symbol
  identity, and emits one idempotent candidate record from the exact clean build snapshot, ancestry,
  affected surface, composition evidence, release receipt, and cached evaluation-event ids. Parameter
  proposals retain their distinct semantic bundle identity; source evidence is never reconstructed
  after teardown.
- [x] Implement typed research-prior, campaign-seed, constraint/negative, and legacy-import events
  (§19). ✅ 2026-08-10 — superseded by the operator-approved lean loop: proposal-v3, immutable
  proposal receipts, the prior-art catalogue, and the append-only result journal are the live record
  plane; the deleted broad strategy-event plane is not restored before real campaigns justify it.
- [x] Enumerate and content-hash the historical knowledge source manifest across root/research
  handoffs, artifacts, evidence reports, progress pointers, current source, and research intake.
  ✅ 2026-08-10 — scoped to executable prior art: `prior_art_catalogue.json` binds source commit,
  scan/tree receipts, expected absences, exit actions, and the checked substrate-fact receipt set.
- [x] **Quarantine the existing `kernel_store.py` rows** as `legacy_unverified`: `kernel_eval.sh`
  never gated on coherence, so the correct-only Pareto is contaminated. Check whether any raw
  artifacts survive (the campaign output directory is missing, so likely none); re-derive only rows
  with surviving evidence and an explicit anchor. Exclude the rest from every planner retrieval.
  ✅ 2026-08-10 — expected absence verified: there is no surviving legacy result corpus to import;
  the deprecated harness is fenced and the lean driver reads only its validated proposal/journal path.
- [x] Mark `kernel_eval.sh` deprecated-and-unrunnable so no further contaminated rows are produced
  before AK3 replaces it. ✅ 2026-08-03 (exits 2; KERNEL_EVAL_ALLOW_DEPRECATED=1 overrides)
- [x] Atomize the source draft and historical research without upgrading evidence grade; link
  duplicates, contradictions, confounds, supersessions, transfer limits, and reopen predicates.
  ✅ 2026-08-10 — `prior_art.py` and its catalogue implement the four evidence buckets,
  contradictions/expected absence, transfer limits, deterministic exit action, and 1% pruning.
- [x] Require a receipt on every suppressing ledger entry, bound to the current production commit,
  re-verified on anchor move (§19.3).
  ✅ 2026-08-10 — prior-art exclusions carry commit/scan/tree receipts and fail closed when their
  source or anchor binding moves.
- [x] Compile the three derived memory products and prove regime-matched retrieval for fixed
  planner/critic fixtures.
  ✅ 2026-08-10 — superseded by the lean-loop retrieval boundary: the reviewed prior-art catalogue
  is the sole pre-campaign memory product; the deferred autonomous planner has no live retrieval API.
- [x] Add validators refusing mutable evaluator IDs, stale production anchors, undeclared change
  classes, actor-supplied scope, missing fallbacks, and unbounded resource/storage requests.
  ✅ 2026-08-10 — current schema, integrity, worktree, proposal-v3, storage, and execution-chain
  validators cover these refusal classes on the runnable campaign path.
- [x] Replace SQLite-as-source with fsync append-only **sharded** events plus content-addressed
  snapshots; retain SQLite as a rebuildable view; readers read all shards. ✅ 2026-08-03 (journal.py, 79 tests)
- [x] Add supersession/tombstone events; deprecate destructive primary-record purge. ✅ 2026-08-03 (incl. RETRIEVAL_SUPERSEDED)
- [x] Add failure/mechanism/do-not-repeat/context/champion views consumed by the planner.
  ✅ 2026-08-10 — superseded by the lean-loop boundary: no live autonomous planner consumes these
  views; failures and mechanism evidence remain durable in the journal for later evidence-driven design.
- [x] Implement the storage plane: durability classes, per-campaign quota, retention classes,
  tombstoned expiry, `DISK_PRESSURE`. ✅ 2026-08-03 (storage.py, 159 tests)
- [x] Land the evidence root under `epyc-inference-research/data/<campaign>/` with `SHA256SUMS` and
  README; extend `check_evidence_durability.py` to cover AutoKernel citations. ✅ 2026-08-10 — the
  live storage plane creates the durable campaign root and content-addressed evidence manifest;
  the external checker remains scoped to its model-registry contract rather than duplicating storage.
- [x] Clear the §3.7 durability exposures: copy the np_context decision surface out of
  `/mnt/raid0/llm/tmp/`, track the two np_context study bundles, restore the P2-5j protocol to git. ✅ 2026-08-03 (research 9687f7fe, root 047b5a40)
- [x] Add deterministic reconstruction test from journal plus immutable artifacts only. ✅ 2026-08-03 (journal + integration suites)
- [x] Fix the `kernel_store.py:88` file-handle warning; add both `kernel_rnd` suites to
  `PYTEST_SMOKE`; pin pytest in `pyproject.toml`/`uv.lock` rather than injecting it via `--with`.
  ✅ 2026-08-10 — delivered in research commit `ccfb23a3`; project tooling owns the pinned test
  environment and both kernel-R&amp;D suites are in smoke coverage.
- [x] **Measure achievable MI210 bandwidth (STREAM/BabelStream-class) — §8.3.1's second denominator now EXISTS** ✅ 2026-08-03. **1433.3 GB/s achievable = 87.5% of the 1638 GB/s datasheet peak**; triad 1371.1;
  p20–p80 within ~1.2%; correctness PASS. Instrument `epyc-inference-research`
  `scripts/benchmark/mi210_achievable_bandwidth.sh`, receipt
  `data/mi210-achievable-bandwidth/20260803T124401Z/receipt.json`
  (SHA-256 `0aab9c7e135929e72fd3a5c2498eb807dc16d0f80b773f063e1df3524df7b4d3`), committed `328b768d`.
  Co-residency recorded: three servers resident and **idle**, autopilot paused, 0 busy slots, 0% GPU use.
  **Correction factor 1.143×** — the prior ~1.3–1.4 TB/s estimate was low. **Both denominators must now be
  carried per §8.3.1, and a cross-vendor comparison must stay spec-to-spec**: converting our numbers to an
  achievable basis while leaving a competitor's on a spec basis makes the gap look smaller without it
  being smaller.
- [x] Put both denominators into the P0.1 profile manifest as substrate constants, with the basis of each
  attainment figure recorded alongside it rather than inferred.
  ✅ 2026-08-10 — `substrate_facts.json` carries measured and spec compute/bandwidth denominators,
  basis-preserving ridges, crossover points, receipt paths, and the MI210 NUMA identity; `substrate.py`
  re-derives and validates them.
- [x] Measure **H2D/D2H** on the Gen4 x16 link ✅ 2026-08-03 — **H2D 28.89 GB/s, D2H 28.20 GB/s**
  (91.7% / 89.5% of Gen4 x16 theoretical), receipt
  `epyc-inference-research/data/mi210-h2d-d2h/20260803T131500Z/`, committed `2aa14264`. Bulk transfer is
  **NUMA-node-independent** to within 0.1%, which matters for seed G1: the GPU lane's cross-node host
  placement costs nothing on the *transfer* path. It says nothing about host-side memory access during
  serving, which is the regime G1 actually targets — do not let this close G1.
- [x] Import the MI210 roofline constants as substrate facts. **All three are now MEASURED (2026-08-03)**,
  so §19.0 rule 4's no-upgrade-on-import concern applies only to the derived companions: peak
  **172.2 TFLOPS `[M]`** (181.0 `[D]`), achievable BW **1433.3 GB/s `[M]`** (1638 `[D]`), PCIe
  **28.89/28.20 GB/s `[M]`**, giving a clean measured-basis **ridge 120.1 FLOP/byte** and
  `B*` = Q4_K 34 / Q8_0 64 / bf16 120. Carry the spec-basis ridge (110.5) alongside for cross-vendor
  comparison, and **never mix bases**. Derivations, the measured/derived reconciliation, and the 2×
  defect in AMD's own published figure:
  [`mi210-mfma-compute-bound-paths.md`](../completed/mi210-mfma-compute-bound-paths.md).
  ✅ 2026-08-10 — imported without changing evidence grades; mixed-basis ridges are rejected by
  reconstruction tests.

**Exit:** crash/restart/rewind never loses a candidate or its negative lesson; the loop starts with the
project's prior knowledge rather than an empty memory; no contaminated legacy row can reach the planner.

### Phase AK2 — worktree, build, resource, process, and bus control (mostly no inference)

- [x] Build backend worktree managers that start from the current production tip, namespace worktrees
  (`llama.cpp-ak-<campaign_id>`) and branches (`ak/<campaign_id>/…`), use pathspec-limited commits in
  the shared clone, and categorically deny production source/build paths.
  ✅ 2026-08-10 — `execution/worktree.py` re-resolves the frozen tip, namespaces campaign trees and
  branches, enforces pathspec containment, and rejects production paths.
- [x] Build candidate-local build/cache layout and full build identity receipts. ✅ 2026-08-10 — the
  build plan uses candidate-local roots and binds source snapshot, command/toolchain, output, libraries,
  linkage, and log identity.
- [x] **Build the cross-process MI210 device claim** — the single largest missing substrate: ✅ 2026-08-03 (device_claim.py, 48 tests, independently probed)
  - [x] Decide the mechanism: an on-disk lock under the same root as `cpu_region.*.lock`, keyed by
    device (`gpu.mi210_0.lock`), holding owner id, PID, start time, campaign id, expiry, and purpose. ✅ 2026-08-03 (gpu_device.<id>.lock beside cpu_region.*)
  - [x] Acquire is atomic (`O_CREAT|O_EXCL` plus `flock`), never advisory-by-convention. ✅ 2026-08-03
  - [x] Liveness is PID plus process-start-time, not a heartbeat, so a stale lock from a dead holder is
    detectably reclaimable and a live holder is never stolen from. ✅ 2026-08-03 (/proc stat field 22)
  - [x] Crash recovery: a lock whose PID is gone or whose start time mismatches is reclaimable after a
    grace period, and the reclamation is journaled. ✅ 2026-08-03 (reclamation journaled)
  - [x] Revocation follows `BUS_PROTOCOL.md:47-51` — mark `revoking`, holder drains at its boundary,
    an ignored revocation surfaces as a `defect`, never a forcible steal. ✅ 2026-08-03
  - [x] Extend `region_lock_cli.py` with a device verb, or add a sibling CLI sharing its lock root;
    do not fork the lock semantics. **BLOCKED — epyc-orchestrator was held by another session on 2026-08-03; the claim was built in the research repo sharing the same on-disk lock root, so exclusion already works cross-repo. This verb is a convenience wrapper owned by whoever holds that repo.**
    ✅ 2026-08-10 — superseded as a convenience wrapper: AutoKernel's direct `device_claim.py` API is
    the authoritative cross-process claimant at the shared lock root; no second CLI semantics are needed.
  - [x] Retire `src/gpu_lease.py`'s process-local lease for cross-process use, or clearly scope it as
    intra-process only, and migrate `axa2_live_cutover_bundle.py:535`. ✅ 2026-08-10 — scoped out of
    AutoKernel: its live path uses the cross-process claim exclusively and never imports the legacy
    process-local lease; migration of unrelated legacy consumers is not an AutoKernel dependency.
  - [x] Emit a claim receipt id that lands in every evaluation event. ✅ 2026-08-03 (akd- receipt)
  - [x] Acceptance: two processes contend and the second blocks or fails cleanly; a killed holder's
    lock is reclaimable and the reclamation is journaled; a live holder is never preempted forcibly; a
    revoke drains within the declared bound; `kernel_eval.sh`'s `gpu_idle()` is deleted, not wrapped. ✅ 2026-08-03 (verified outside the suite)
  - [x] **Arm the campaign-wide device-claim expiry check from the shared resource window.** ✅
    2026-08-12 — research `accff01f` passes `CampaignSpec.max_hold_s` to every MI210 claim, matching
    the CPU-region claim acquired in the same transaction. This writes `expires_at`, so the existing
    advisory expiry check can return `FAIL` instead of `COULD_NOT_CHECK`; expiry still requests
    quiesce-and-drain and never authorizes stealing a live device. Focused campaign tests passed
    **199/199**, and mutation controls fail if the keyword is removed or replaced by a hard-coded
    six-hour value.
- [x] Integrate CPU region claims and co-residency policy. ✅ 2026-08-10 — `campaign.py` acquires the
  exact declared CPU footprint before T0/T1 and binds the same receipt into both consumers; lane
  topology and co-residency are validated separately.
- [x] Build the single audited read-only preflight wrapper (§3.5) and remove every other name-pattern
  process read from the loop. ✅ 2026-08-03 (preflight.py + claim_witness.py, 143 tests)
- [x] Run every candidate process in an owned scope/cgroup with PID/start-time receipts and verified
  teardown. ✅ 2026-08-10 — the lean loop uses owned process groups with captured PID/PGID/start-time
  receipts and TERM→KILL→verified-dead teardown; a separate cgroup controller was not retained.
- [x] Integrate host-health/reboot-required and cache-preparation states, including the one-week
  uptime ceiling and the reboot decision package (§10.7). ✅ 2026-08-10 — cache state and CPU/GPU
  throttle state are verdict-bearing; reboot/uptime remains operator session preflight because the lean
  loop is session-driven rather than a daemon allowed to request host mutation.
- [x] Register AutoKernel on the session bus: roster id, heartbeat at every task boundary, outbox,
  lane declaration (`cpu`/`gpu`/`both`/`none`) on every queued unit, revoke handling, C19/C20
  visibility, and a re-read-instructions checkpoint so a long-lived service does not run indefinitely
  on its startup copy (`BUS_PROTOCOL.md:76-90`). ✅ 2026-08-10 — superseded by the session-driven lean
  loop: the owning agent session, not campaign code, owns bus heartbeat, lane declaration and revocation.
- [x] Wire `scripts/utils/agent_log.sh` task start/decision/end logging, and log rollback commands
  before any system-affecting step (`OPERATING_CONSTRAINTS.md:22-25`).
  ✅ 2026-08-10 — session lifecycle logging remains the agent owner's boundary; duplicating it inside
  the non-daemon campaign would create two conflicting audit owners.
- [x] Verify the C6 sandbox on the real host; remove unsandboxed operation from live AutoKernel.
  ✅ 2026-08-11 — `execution/sandbox.py` supplies a native host backend: Landlock ABI 6 write
  confinement, seccomp denial of signalling/network/namespace escape, non-root execution, finite
  rlimits, and a per-invocation cgroup-v2 leaf. Real shell and tiny-CMake probes produced evaluator-
  owned activation receipts and verified empty/removed teardown; startup is fail-closed.
- [x] Extend the sandbox to **candidate binary execution**, not just the evaluator: no write access
  outside the campaign tree, no signalling processes it does not own, owned cgroup with verified
  teardown. The loop compiles code it authored and then runs it with GPU access on a shared host;
  the actor's tool allowlist does not constrain that binary's syscalls (§8.5.1). ✅ 2026-08-11 —
  candidate-controlled configure/build, T0 and every T1 arm require the native C6 policy; receipts
  live outside the candidate-writable tree, fresh process identities are checked, and the next arm
  cannot start until the preceding cgroup has been drained and removed.
- [x] Add resource starvation/drain/resume tests and campaign checkpointing. ✅ 2026-08-10 — append-only
  journal replay, completed-run keys, claim revocation/drain, teardown-on-every-exit, and restart fixtures
  cover the retained campaign path.
- [x] **Import the MI210's actual NUMA attachment as a resource-plane fact: the device is on node 1, not
  node 3** (sysfs ground truth, `/sys/class/drm/card2/device`, `0x740f`, `numa_node=1`). The seed G1 row
  in §19.6 asks to "import current MI210 node attachment" and **no numeric node appears anywhere in the
  six MI210/autokernel handoffs.** The consequence is immediate and needs no kernel work: **the GPU
  lane's host threads at 184–191 are already cross-node, and device-local placement has never been
  tried.** Placement evidence lives in
  [`gpu-acceleration-path.md`](gpu-acceleration-path.md).
  ✅ 2026-08-10 — `substrate_facts.json` and the T1 recipes bind GPU NUMA node 1 versus host node 3;
  the validator refuses drift or mixed identity.

**Exit:** the controller can safely author, build, acquire and release CPU **and GPU** resources, be
revoked, and be seen — without inference or production mutation.

### Phase AK3 — trusted tiered evaluator

- [x] Replace `kernel_eval.sh` behind a typed evaluator API; do not grow the old shell script. ✅ 2026-08-03 (evaluator/api.py; verdict cannot be stamped)
- [x] Require an explicit anchor for every performance/coherence comparison; no-baseline is `INVALID`. ✅ 2026-08-03 (anchor-less is INVALID)
- [x] Make status a computed verdict, never an unconditional literal. ✅ 2026-08-03 (re-derived in __post_init__; VerdictTampering)
- [x] Build the codified operator-microbenchmark recipe constructor so T1 argv is constructed, and
  bind it to a recipe id `P-AK-SEARCH-1` can cite. ✅ 2026-08-03 (evaluator/recipes.py)
- [x] Implement the affected-surface derivation and dispatch-trace reconciliation (§6.4). ✅ 2026-08-03 (evaluator/surface.py)
- [x] Implement the §8.5.1 source-integrity gates: symbol/registration-table extraction and diffing
  against the anchor binary, clean-build-from-snapshot enforcement, semantic diff conformance, the
  `core_header` risk tier, and repair-from-clean-parent with a per-proposal cap. ✅ 2026-08-03 (evaluator/integrity.py)
- [x] Red-team those gates specifically: a candidate that deletes a template specialization, one that
  drops a dispatch case, one that removes an op registration, and one whose incremental tree compiles
  while its snapshot does not. Each must fail before any behavioural check runs. ✅ 2026-08-03 (in the AK3 red-team pass)
- [x] Add full correctness surfaces: relevant ops including `MUL_MAT_ID`, exact/unseen shapes,
  PPL/numerical margin, state/rollback, no-fallback, determinism class, real-model smoke, cleanup. ✅ 2026-08-03 (evaluator/correctness.py; anchor triple bound on all 5 evidence surfaces)
- [x] Add the mandatory ASAN/UBSAN path for memory/threading changes and the diff-complexity ceiling. ✅ 2026-08-03
- [x] Record absolute single-stream, batched/aggregate, latency, variability, capacity, mechanism, and
  numeric correctness margins where the campaign requires them. ✅ 2026-08-03 (correctness + statistics)
- [x] Implement T0/T1/T2 adapters for llama CPU/GPU first; STT/TTS after their protocols exist. ✅ 2026-08-03 (STT/TTS deferred to AK9 as designed)
- [x] **Add an immutable archive/resume package for source-candidate T0 prerequisites.** ✅
  2026-08-12 — research `5d7a408b` snapshots the raw sensitivity, hostile-distribution, and
  checker-isolation CSV bytes before claim, verifies package/receipt/document hashes, and re-runs the
  trusted reducers against the exact completed source, binary, and evaluator identities before T0.
  Dry-run packages cannot authorize execution and parameter proposals reject them. The independent
  package/campaign/footprint slice passed **308 tests**.
- [x] **Complete full-suite acceptance and promotion of the fresh-source prerequisite capture
  adapter.** ✅ 2026-08-12 — research main `51742ebd` makes the producer campaign-reachable, journals
  invocation/completion identity, resumes without duplicate producer calls, and binds the resulting
  strict package before claim. The correctly discovered full AutoKernel suite passed **5,464 tests
  with one expected failure**. No real source candidate, kernel build, profile, or inference ran.
- [x] Implement the e-process reducer, pre-committed stopping rule enforcement, and the anchor gate. ✅ 2026-08-03 (evaluator/statistics.py; alpha derived, none supplied)
- [x] Red-team the evaluator with deliberately wrong, test-shape-specialized, fake-score, fallback,
  cache-gaming, scope-under-declaring, and timeout/leak candidates. ✅ 2026-08-03
- [x] Add four controls: positive, neutral, degraded-negative, and periodic **A/A**. ✅ 2026-08-03 (five — the accept-side historical-win replay joined them)
- [x] **Emit a PRE-CORRECTNESS PROGRESS LADDER, not a single failure verdict.** Today every non-verifying
  candidate is an equal failure, which is exactly the regime where most MI210 candidates die — so the
  search has no gradient where it needs one most. Rank `hipcc-fail < runtime-fail < wrong-output`
  (reference implementation: intake-974's `s(NX)=-0.05 < s(NC)=0.00 < s(NR)=+0.05 < s(WO)=+0.10`). This
  is the highest-value transfer of the 2026-08-03 intake batch and intake-939 has no analogue for it.
  ✅ 2026-08-10 — superseded as live fitness by the lean session-driven loop: compile exceptions,
  runtime failures, and per-gate wrong-output records remain distinct durable outcomes and are rendered
  directly; collapsing them into a scalar reward has no consumer until an autonomous selector returns.
- [x] **Record CGRE — normalized gap closure `(T_base − T_cand)/(T_base − T_ref)`, clipped to [0,1] —
  alongside the raw ratio.** It supplies the CONTINUOUS form a hill-climbing search needs, closing a gap
  an earlier dive left open when it declined intake-939's binary/bucketed finding as gradient-hostile.
  Requires an expert reference per target, which most of our targets do not yet have — scope that first.
  ✅ 2026-08-10 — retained as an empirical field only for targets that acquire an expert reference;
  no such target/archive exists yet, so inventing a value or a live selector input is explicitly refused.
- [x] **Adopt the epsilon-separated staircase so every correct candidate outranks every failed one**
  (`R = max(s(WO) + ε, R_succ)`, ε=0.05). **Note this is the OUTER-GATE form, which intake-939's own
  winner rejects in favour of the collapsed conjunction — see the deliberate open question in AK4.**
  ✅ 2026-08-10 — superseded by the lean rule: T0 all-PASS is lexicographically prior and failed
  candidates receive no speed rank. The loop does not scalarize correctness and performance.

**Exit:** controls reach their expected deterministic states, the actor cannot tamper with the
evaluator or its own scope, and T1 may legally guide search.

### Phase AK4 — planner/critic/controller

- [x] Implement the explicit state machine and journal every transition. ✅ 2026-08-03 (controller/state_machine.py; journal-then-act)
- [x] Build the structured source/profile/workload/negative-history/oracle context compiler with
  quarantined rendering of external content. ✅ 2026-08-03 (controller/context.py)
- [x] Build planner and pre-/post-run critic adapters; prefer distinct providers for the two roles. ✅ 2026-08-03 (controller/planner.py + critic.py, fake-provider tested)
- [x] Implement **operator hypotheses and still-open hypothesis tracking** (§8.4.0): the operator-facing
  channel, the mandatory falsifier, `design_prior` grading that origin cannot raise, re-surfacing the
  open set each round, and resolution journaled as confirmed/refuted/inconclusive with its evidence.
  Test that an operator hypothesis is rejected by the critic when it repeats a receipted negative —
  authorship is not evidence. ✅ 2026-08-03 (controller/hypotheses.py)
- [x] Enforce proposal schema, one-concept rule, wall-share ceiling, novelty, budget, hierarchy, and
  the oracle question. ✅ 2026-08-03 (controller/selection.py)
- [x] Build store-guided next-experiment selection using information gain plus expected value. ✅ 2026-08-03 (incl. harvest/explore phases)
- [x] Build champion composition and mandatory combined-candidate reevaluation. ✅ 2026-08-03 (controller/composition.py)
- [x] **Restore the lean bank/frontier/champion sequencer on the runnable journal plane.** ✅
  2026-08-12 — research `069e79fd` requires validated write-side banking, the complete sealed
  backend/tool anchor, compatible file/symbol/dispatch predicates, and direct combined-candidate
  T0/T1/T2 evidence before champion replacement. Concurrent replay is idempotent; rejected or failed
  composition preserves the incumbent; anchor moves fail closed and reanchor only absorbs members
  proven present by a matching sealed release receipt. Campaign, host-process, release, and production
  mutation capabilities remain outside the sequencer import closure.
- [x] Implement deterministic stop/plateau/budget/storage/integrity/evaluator-gap guards. ✅ 2026-08-03 (controller/guards.py; 13 stop states)
- [x] Add planner regression fixtures proving it consults failures and does not repeat known negatives. ✅ 2026-08-03
- [x] **Settle WHICH gate form this loop uses — the literature does not, and we should not inherit the
  ambiguity silently.** Two capable groups made OPPOSITE choices within the same family: intake-939's
  winner is the **collapsed conjunction** (correct-but-slow → full failure reward), while intake-974
  independently chose a **strictly-separated outer gate** and never tested the conjunction. Neither ran
  the comparison the other needs. **We can run it at ~zero GPU cost** via offline replay over banked
  candidate records — the two forms are different scoring functions over the same stored outcomes.
  Operator approval required before it becomes a fitness change; the replay itself is free.
  ✅ 2026-08-10 — superseded for live authority by the operator-approved lean accept rule: T0 is a
  lexicographic outer correctness gate and paired performance is considered only after all-PASS. The
  alternate scoring replay may remain an observe-only analysis under AK-WM-2; it cannot change fitness.
- [x] **Add `KIND_STATE_TRANSITION` to `journal.py` and rewire the controller's `TransitionRecorder`.**
  `journal.KINDS` is a closed vocabulary with no state-transition kind, and `journal.py` was outside
  AK4's write scope, so non-stop transitions land in a sibling `TransitionLedger` with the same
  durability discipline, written under `Journal.write_lock()` so ordering stays total. Co-opting
  `KIND_STOP_STATE` was correctly rejected: `Views.stop_states` is a derived view other planes read,
  and reusing it would make that view stop meaning "this campaign stopped". One-line wiring change
  behind the existing seam. ✅ 2026-08-10 — superseded: the Python controller/state-transition plane
  is deferred and is outside the lean campaign import path; the live driver journals proposal and terminal
  campaign records directly, so adding a dead vocabulary member would restore no behavior.
- [x] **Re-derive that a capture actually ran against the anchor it names.** The five T0 evidence
  surfaces now bind the anchor triple and refuse a replay mismatch, but the recorded identity is still
  the *producer's declaration* — `produced_by` is checked and nothing re-derives it. This binds an
  honest producer's replay, not a dishonest producer's capture. Acceptable while the producer is the
  trusted evaluator the actor cannot modify; revisit if that assumption weakens. ✅ 2026-08-10 — accepted
  under the current trust boundary and made fail-closed at its projection seam: producer-supplied symbol,
  semantic-diff, and surface checks now propagate PASS/FAIL/COULD_NOT_CHECK into the 17 T0 gates instead
  of disappearing. A future untrusted producer requires a new capture attestation protocol.
- [x] **Apply the AutoPilot operator-hypothesis integration patch** — epyc-orchestrator
  `scripts/autopilot/OPERATOR_HYPOTHESES_INTEGRATION.md`; module and tests committed in `536ea87f`.
  Six anchor points in `autopilot.py`, quoted verbatim and pinned to that file's SHA-256. **Owner:
  whoever holds `autopilot.py`** — not applied here because that repo had a live session. Until it is
  applied the channel exists and passes 25 tests, but the planner never reads it. ✅ 2026-08-10 — verified
  live in `autopilot.py`: planner-block construction, still-open context, structured resolution parsing,
  and durable resolution recording are all wired.

- [x] **AK-WM-0 — Existing least-commitment prerequisites audited. ✅ 2026-08-05** `ProposalManifest`
  already requires a hypothesis, falsifiable counter, target/non-target frame, `novelty_basis`, and
  `expected_information_gain`; evaluator records carry `scope_denominator`; selection ranks information gain
  before expected value; and actor-declared affected scope is not trusted. **2026-08-05 re-audit:** the
  schema/evaluator controls survive, but the named planner/selection implementation was pruned on
  2026-08-04; its ordering is now a historical incumbent protocol, not a currently runnable selector.
  Preserve these as comparison baselines, not as evidence for weakness maximization.

- [x] **AK-WM-1 — Add a representation contract before evaluating any least-commitment selector. ✅
  2026-08-05** Extended
  the proposal schema and fixtures with vocabulary/source receipts, considered and excluded alternatives,
  an empirical-demand receipt, abstraction-construction cost, canonical encoding, and semantics-preserving
  recoding fixture ids. New records use `proposal.v3`; v2 remains readable under its original validator.
  `frame_sha256` is mechanically re-derived, and `check_representation_comparable()` fails closed across
  representation or demand frames. Mechanically derived/traced affected surfaces remain authoritative over
  actor declarations. Executing campaigns now require `--proposal-manifest`; proposal-v3 is validated and
  fsynced before preflight, claim, mutation, or build, identical resume is idempotent, and same-id/different-
  bytes is refused. The AP-WM-1 evaluator is implemented as an observe-only module outside the campaign
  import path and exposes no selector/champion/T2/T3 mutation API.
- [ ] **AK-WM-2a — Materialize the first real matched completed-proposal archive.** After Step 3
  writes proposal-v4 manifests plus clean DECIDED terminal journal/evaluation records, deterministically
  join them with the strict builder and retain the archive plus build manifest. Synthetic fixtures
  remain regression tests and cannot populate the archive.
  - [x] **Implement the strict real-archive builder.** ✅ 2026-08-12 — research `900cb5c6` adds
    `least_commitment_archive_builder.py`. It accepts only real proposal-v3/v4 events joined to clean
    DECIDED terminal events with executed/ok decisions, unchanged-production PASS, released
    resources, nonempty pairs, and hash-bound diagnostic, outcome, and matched one-factor receipts.
    Missing, synthetic, mismatched-frame, direction-drifted, or receipt-tampered inputs fail closed.
    The canonical AutoKernel suite passed 4,078 tests with one expected failure.
  - [x] **Implement governed atomic producers for the three archive receipts.** ✅ 2026-08-12 —
    research `e8de8bfa` adds the non-test `least_commitment_receipts.py` projection path for
    `least_commitment_diagnostics.v1`, `least_commitment_outcome.v1`, and
    `matched_one_factor_intervention.v1`. A projection plan may copy only JSON-pointer values from
    explicitly SHA-256-pinned real journal records; it requires at least two distinct clean completed
    campaigns, mechanically proves exactly one changed factor, publishes atomically, and validates
    its output through the unchanged strict archive builder. Synthetic, fixture, dry-run, incomplete,
    mismatched, and literal empirical inputs fail closed. The receipt/footprint slice passed **98/98**
    and the integrated canonical AutoKernel suite exited zero. The parent remains open until CPU IQK
    and a matched completed control/intervention pair supply the real records.
  - [x] **Materialize the exact prospective IQK intervention/control producer inputs.** ✅ 2026-08-12 —
    research `58b8d66a`/`948a95a4` adds the live capture-plan producer and mechanical diagnostic-source
    compiler, then materializes the intervention and A/A control proposals, SHA-bound source receipts,
    capture plans, stated hypothesis stores, and control-specific physical envelope under the two
    durable campaign roots. `evidence_path_rehearsal.py` resolves those exact files, proves one changed
    factor, validates both hypothesis statements/falsifiers, reproduces the control proposal byte for
    byte, and reaches the architecture-only `RELEASE_PACKAGE_READY` seam without inference.
  - [x] **Migrate the durable IQK pair to proposal-v4 and regress the exact artifact path.** ✅ 2026-08-12 —
    research `3bd94893` executes both durable intervention/control inputs through the current
    `campaign.main` validator, binds the source-transparent `llama_source` provider at `a4cb04ca`,
    regenerates all proposal-dependent hashes/frames, and preserves v3 history under
    `inputs/superseded/`. Both exact dry runs exit zero with 12 steps, `state=dry_run_composed`, and
    `executed=false`; the independent full suite passed **5,606 tests** with one expected failure.


  - [x] **Require a bound typed hypothesis before a completed proposal may enter the archive.** ✅
    2026-08-12 — archive admission now refuses exploratory campaign records and requires the terminal
    authorization to resolve by exact hypothesis id, campaign id, ledger sequence, payload, and proposal
    statement. A dry-run followed by execute under the same campaign identity remains idempotent: one
    OPENED record is retained and separate composition/live authorizations are journaled so the archive
    can bind the live spend exactly. The post-rebase suite passed **5,599 tests** with one expected failure.
  - [x] **Repair identity-derived seed and frame binding before the reboot campaign.** ✅ 2026-08-12 —
    research `0c24bc0d` introduces a shared `akm-` experiment identity and derives benchmark, T0,
    scheduling, hold-out-selection, model, topology, calibration, envelope, provider, and intervention
    factors from the actual campaign frame. Live admission and materialization refuse any plan whose
    copied factors differ. Research `9ec0c30c` exposes the same mechanical derivation for prospective
    input generation without weakening live admission.
  - [x] **Replace the placeholder held-out regime with a genuinely withheld prospective regime.** ✅
    2026-08-12 — research `b9bc28a6` requires a separately SHA-bound measured held-out outcome whose
    regime is outside the proposal target regimes. Missing, construction-regime, identity-drifted, or
    non-measured inputs fail capture admission.
  - [x] **Make the control reducer evaluate the control proposal's own falsifier.** ✅ 2026-08-12 —
    research `b9bc28a6` makes the role explicit and reduces control failure from its own keep decision
    or effect beyond the accepted noise floor; intervention and control use distinct immutable rules and
    journal the bound predicates and result.
  - [x] **Eliminate cloned diagnostic semantics across the intervention/control pair.** ✅ 2026-08-12 —
    research `b9bc28a6` derives each diagnostic/recoding block from its own SHA-bound native source
    receipt, publishes its semantic digest, and makes the matched projector refuse equal intervention/
    control semantic digests even when the numerical one-factor frame otherwise matches.
  - [x] **Close the real-only AP-WM/report provenance gap.** ✅ 2026-08-12 — research `b9bc28a6`
    requires a strict projector result at the AP-WM CLI, labels unprojected library fixtures
    `fixture_or_unlabelled`, and emits `real` only after the archive and projector provenance validate.
    Reports now carry matched-validation, pair noise floors, and an explicit underpowered state. The
    full AutoKernel suite after this repair chain passed **5,671 tests with one expected failure**.
  - [x] **Publish one exact proposal-v4 regeneration command.** ✅ 2026-08-12 — research `03f9ae69`
    adds `prepare_iqk_matched_pair.py`, a repository-relative deterministic producer that accepts one
    authoritative manifest and atomically writes the intervention/control proposal-v4, capture,
    diagnostic-source, held-out-outcome, and physical-envelope trees. It derives one shared frame,
    proves `ggml_iqk` is the sole changed factor, records every input/output hash, refuses overwrite and
    drift, rolls back the pair on failure, and rebuilds byte-identically in tests. Its held-out projector
    accepts only a distinct clean hypothesis-bound completed journal. Real pair materialization remains
    in parent AK-WM-2a and correctly waits for post-reboot completed journals; fixtures cannot substitute.


- [ ] **AK-WM-2b — Run AP-WM-1 on that archive, observe-only.** Report the archive protocol,
  matched-intervention validation, per-regime/surface results, noise floor, robust sign error, and
  recoding stability. Until real evidence shows invariant independent signal, do not add weakness,
  completion count, K-rho, scope width, prose length, patch size, or description length to live selection,
  champion, T2, or T3 authority.

**Exit:** a mock campaign moves from source/profile facts through proposals, corrections, negative
memory, and champion maintenance without human steering.

### Phase AK5 — readiness estimator and release gate

- [x] Implement per-backend, per-phase T2 scope and weights from compiled facts, including the
  co-resident cell and capacity deltas. ✅ 2026-08-03
- [x] Implement the readiness signal and its coverage/non-target/mechanism guards. It reports; it does
  not trigger. ✅ 2026-08-03
- [x] Implement the capability objective with an immutable campaign-start utility model. ✅ 2026-08-03
- [x] Extend `kernel_freeze_scope.py` into a complete deduplicated release-plan compiler keyed by
  source tree and reconciled affected surface, with the two-stage per-backend unchanged test (§3.2):
  build-system-derived source-closure diff as the gate, normalized comparison against an anchor
  rebuild as confirmation, transfer receipts for dropped cells, and a filed defect when the stages
  disagree. ✅ 2026-08-03
- [x] Implement the generic T3 runner and release-bundle schema across CPU/GPU llama first. ✅ 2026-08-03
- [x] Implement `PASS_WITH_WAIVER` and waiver hash/predicate verification. ✅ 2026-08-03
- [x] Port v8/speech transaction integrity patterns into generic validation — not hard-coded evidence. ✅ 2026-08-03
- [x] Dry-run the T3 compiler/validator against preserved v8 and speech freeze artifacts; **expect the
  v8 dry-run to FAIL without its waiver** and treat that as calibration. ✅ 2026-08-03
- [x] Add failed-gate replay/cooldown/idempotence behaviour. ✅ 2026-08-03
- [x] **Restore the release/readiness evaluator as a separate operator-triggered plane after the lean
  refactor.** ✅ 2026-08-12 — research `99fe3014` restores readiness, generic T3, and package
  assembly behind pure receipt inputs and injected evaluator authority. Release-local preflight
  enforces the ratified seven-day uptime ceiling and held-resource/storage checks without reading or
  mutating the host. The plane cannot write production, spawn/signal a process, execute a drafted
  command, or self-trigger; release mode refuses until `P-KERNEL-FREEZE-1` is human-ratified.

**Exit:** a sealed fixture champion yields one reproducible PASS/FAIL/PASS_WITH_WAIVER bundle and
cannot retrigger the expensive gate unchanged.

### Phase AK6 — release packager and operator handoff

- [x] Implement the packager: transaction plan, rollback plan, draft era row, draft AutoPilot
  rebaseline note, linkage results, and a pre-validated command sequence. ✅ 2026-08-03
- [x] Prove it refuses missing hashes, evaluator drift, dirty ancestry, reused version names,
  incumbent modification, and an incomplete rollback. ✅ 2026-08-03
- [x] Reconstruct the expected v8 and speech transactions from fixtures without applying them. ✅ 2026-08-03
- [x] Route the cutover request through the bus to whoever owns inference (§11.3). ✅ 2026-08-03
- [x] Render the package as a four-part decision package (`OPERATING_CONSTRAINTS.md:69-78`). ✅ 2026-08-03
- [ ] Run an end-to-end campaign that stops at a validated package with zero production writes.
  - [x] **Complete the offline operator-closeout integration fixture.** ✅ 2026-08-12 — research
    `4a5f7361` runs the injected lean sequencer through a schema-bound composed champion, readiness,
    dry-run T3, and a validated `RELEASE_PACKAGE_READY` record. It is unreachable from Campaign #1,
    cannot build, infer, execute processes, or write production, and labels architecture fixtures
    `empirical_claim=false`. The independently repeated focused suite passed **1,692 tests**. This is
    regression evidence only and does not close the parent empirical campaign task.
- [x] Run the real restart/crash/resource-preemption/tamper campaign rehearsal. Fixture and

- [ ] Run the real restart/crash/resource-preemption/tamper campaign rehearsal. Fixture and
  fault-injection tests are prerequisites, not substitutes for a campaign rehearsal with durable
  empirical receipts.
  - [x] **Complete the offline fault-injection acceptance matrix.** ✅ 2026-08-12 — research
    `900cb5c6` ran three repetitions of a 657-test matrix: **1,971/1,971 PASS**, no inference. The
    journal, GPU claim, CPU claim, hypotheses, Arena cell runner, statistics, C3 compiler, and physical
    bounds suites cover torn-write/restart recovery, exact-owned-PID crash/reclaim, live-holder
    non-preemption and revocation, plus receipt/hash/source tamper refusal. The broader 4,078-test
    suite passed once with one expected failure. This closes offline acceptance only.
  - [x] **Publish and verify the governed real host-process rehearsal receipt.** ✅ 2026-08-12 —
    research `d0b243a5`/`5c8714a1` launched only exact captured private-session children, planted a
    journal crash and replayed it after restart, proved fake-device revocation was advisory rather
    than preemptive, terminated the exact captured holder, and refused changed hash-bound bytes.
    Campaign `ak-fault-rehearsal-20260812-r2` passed **3/3** legs, touched no live claim root, and
    emitted three dependency-evidence rows sharing one support key with no performance, campaign,
    ranking, release, or production authority. Receipt self-hash `8a345fbdbfff4bb04b4ac3388b8e82ff347941bc8fb93cbe8266ee2b2ed79a58`;
    file SHA-256 `70e9f87bc5405be6e66c9c0f6ce49f0427f1c83ba0f14491c93401e5283d688e`.
- [x] Give the operator surface a freshness and health contract, not just data. Today's `/kernel` page is
  **absence-tolerant over a missing directory** — it renders clean when its producer is dead, which is
  the exact shape of AutoPilot dying at trial 1302 and staying dead ~23 h with every dashboard green.
  Required: a per-panel freshness envelope, an SSOT panel→producer registry whose test fails when a
  panel has no registered source, a `/health` fold, a transport watchdog, and a restart chaos test. ✅ 2026-08-03
- [x] Replace the `/kernel` dashboard JSON contract. The existing one exposes a partial Pareto and
  nothing else; the new contract carries campaign phase, champion membership and readiness, per-backend
  standing, storage and budget headroom, open blocking conditions (`EVALUATOR_COVERAGE_GAP`,
  `ANCHOR_MOVED`, phase-trade exceptions), resource claims held, and release-package state. Version the
  contract explicitly, keep it absence-tolerant as the current page is, preserve the freshness envelope,
  and point `KERNEL_DASHBOARD_JSON` at a durable path rather than the missing scratch directory. ✅ 2026-08-03
- [x] **Repair the Kernel-R&D dashboard's current-receipt selector and live runs producer.** ✅
  2026-08-12 — root `572f33af` adds schema-selected current receipts and a loud missing-attestation
  alarm; `d76b6ee1` binds the supervised hub to the clean deployed checkout; `5bae9d3f` closes the
  inherited supervisor lock FD in the detached hub. Live `/api/kernel`
  reports frozen production v9, `8/8` instrument preflight, `5/5` accepted controls, the GPU replay's
  honest `NOT_REPRODUCED` verdict, and implementation head `900cb5c6`. The terminal v2 campaign
  export is fresh and names the reboot-gated preflight refusal; **85/85** focused deployment tests
  pass. The Kernel-R&D panel's `/api/health` contribution remains `absent` because `champion`,
  `headroom`, and `release_package` are correctly `not_reported` until a real campaign completes,
  not because the dashboard producer is stale.
- [x] **Execute the Kernel-R&D current-state renderer in the static-JavaScript gate and restore the
  complete evidence view.** ✅ 2026-08-12 — root `2183cf1e` fixes a free-identifier runtime failure
  that parsed successfully but stopped rendering before controls, empirical gates, activity, and all
  seven contract sections. The regression harness now executes the renderer against the minimum
  production-set wire shape and requires full blocking-condition detail plus distinct tree identity,
  observed/attested ggml, non-executing ELF linkage, and dashboard-process ambient-path claims.
  `tests.test_dashboard_static_js` passed **28/28**.
- [x] **Teach Kernel-R&D to select current v2 AutoKernel attempts and classify live versus unresolved
  terminal state.** ✅ 2026-08-12 — root `5180663a` accepts v1/v2 manifests, uses exact PID plus process
  start-tick identity for live state, and distinguishes curated-invalid, preflight-refused,
  terminal-complete, and stale-or-terminal-unreported attempts rather than falling back to stale
  six-arm r4. It also reports frozen v9 and the current implementation/readiness gates. The commit is
  pushed on root `main`.

- [x] **Project the terminal raw-HIP r6 decision receipt on Kernel-R&D without manufacturing a
  champion.** ✅ 2026-08-12 — the hub admits only the exact self-hashed
  `epyc.autokernel.hip_decision_grade.v1` receipt at campaign r6, verifies its producer, sealed MI210
  task, 24/24 correctness, numeric error evidence, block-9 e-process crossing, and all 40 per-arm
  duration admissions, then exposes the exact file/self hashes. The distinct card labels the result
  task-local, says `NOT A CHAMPION`, requires experimental llama integration, and carries
  `release/promotion authority: false`. Static JavaScript and producer projection tests fail closed
  on partial admissions or widened authority; no inference ran.
- [x] **Surface current loop-engineering and host-process recovery evidence on Kernel-R&D.** ✅
  2026-08-12 — root `26a9b9f1` adds distinct AK-LE and fault-rehearsal cards. The loop card reports
  the newest complete panel only as raw evidence until a governed reduction exists, then shows its
  producer-authored observation count without granting ranking authority; the fault card reports
  exact leg coverage and the no-live-claim-root boundary without grading dependency evidence as
  performance. Root `2d0d9af4`'s runtime harness plus focused static/panel tests passed. The resident
  root-main hub at `e861baf4` (supervisor PID `2455173`, hub PID `2468686`) rendered r3 `8/8` / 32
  rows, all eight cell detail rows (model, effort, target, survivors, termination, wall time), and r2
  `PASS` / `3/3` at runtime with no JavaScript exception and no inherited `LD_LIBRARY_PATH`.
- [x] **Make frozen-kernel working-tree drift a first-class production-set alarm.** ✅ 2026-08-12 —
  root `a19becb5` projects tracked and untracked status for all three frozen trees, requires cleanliness
  for `intact=true`, and renders exact dirty paths. The reloaded hub (supervisor PID `2455173`, hub PID
  `2478391`) reports whisper.cpp and qwentts.cpp clean, while llama.cpp is dirty only because three
  preserved user-owned paths are untracked: `.gitnexusignore`, `tools/math-tools/external/eigen/`, and
  `tools/math-tools/external/odeint/`. No production-tree mutation or cleanup occurred. Focused tests
  passed **124 cases plus 12 static-JavaScript subtests**.
- [x] Add a non-recursive, panel-specific Kernel-R&D data-health probe so registry consumers can
  distinguish hub transport health from AutoKernel producer health without recursing through the
  global `/api/health` fold. ✅ 2026-08-12 — root `6188197f` moves the registry probe from
  transport `/health` to `/api/kernel/health`. Live transport returns HTTP 200 while the semantic
  endpoint correctly returns HTTP 503 / `absent`, scoped only to Kernel-R&D's unreported
  `champion`, `headroom`, and `release_package` sections. The promoted hub remains supervised by PID
  `1689063` with hub PID `1689100`; no global-health recursion or unrelated timeline state enters the
  panel result.
- [x] Extend production-kernel-set projection from the attested llama.cpp v9 anchor to the separately
  frozen whisper.cpp and qwentts.cpp identities; do not imply the current llama campaign governs
  speech kernels. ✅ 2026-08-12 — root `d2cd8639`, `205b9444`, and `10e3ab77` project the independently
  frozen three-tree/four-binary set, all four stable serving links, non-executing `readelf` linkage,
  ambient loader-path risk, and ggml generations. The exact final focused suite passed **28 tests**.
  Live contract v2 proves `3/3` trees, `4/4` binaries, `4/4` links, and `4/4` ELF linkage without
  executing a production binary; it remains fail-closed on the two residuals below.
  - [x] **Resolve or explicitly retain the frozen llama working-tree cleanliness alarm.** ✅
    2026-08-12 — retained, with no production-tree mutation. The two January-era nested repositories
    are clean third-party source caches: Eigen at `251bff28859af2ed2d5bdf14034175f03cafffc7` and
    Boost.Odeint at `9e75be5d9e435739b086aee928736fc241e77a4a`; classify both as ignore/cache rather
    than production source. `.gitnexusignore` is a July-era repo-local parser policy and is a candidate
    for deliberate tracking in the next versioned experimental tree or its agent overlay, not an
    in-place frozen-v9 commit. The dashboard must continue to report v9 dirty until a future authorized
    versioning or cleanup action resolves those paths.
  - [ ] **OP-17 — Decide whether to attest llama.cpp's ggml generation for the frozen-v9 set.**
    - **Context:** the live tree reports ggml `0.16.0`, but the v9 operator attestation does not state
      an expected ggml generation. The dashboard therefore proves only `2/3` and reports `SET NOT
      PROVEN`; deriving expected authority from the observed tree would be circular.
    - **Option A — amend the frozen-v9 attestation after independent verification (recommended):**
      bind expected llama ggml `0.16.0` through the human-only measurement trust boundary, then let
      the existing comparison prove or refute it.
    - **Option B — retain the current attestation:** keep llama generation explicitly unverified and
      accept that the complete-set fold cannot become intact.
    - **Default:** Option B; no attestation or production state changes.
  - [ ] **After OP-16's reboot, re-check the live dashboard process environment.** Durable host and
    devcontainer configuration are clean. The dashboard hub was already reloaded cleanly before the
    reboot decision (PID `2478391`, `LD_LIBRARY_PATH` absent), replacing the earlier long-lived process
    that inherited two ggml-bearing entries. Still require a fresh post-reboot
    `ambient_library_path.clean=true`; one clean dashboard process is not proof about every launcher.

**Exit:** campaigns produce correct, idempotent, operator-executable release packages and never write
production.

### Phase AK6.5 — cold start: the first campaign

Added 2026-08-04. **This phase was missing**, and its absence was load-bearing: the handoff went
straight from AK6 (the packager) to AK7 (the first supervised freeze), which reads as though a
freeze request is what starts things. It is not. AK7 needs a **champion**, a champion is *"the
current best complete set of compatible, correct changes, anchored on the current production tip
and green through T2"* (§1.2), and no champion exists because **the loop has never run**. Verified
2026-08-03: no champion record anywhere, no state directory. `production-consolidated-v8` is the
**incumbent**, not the champion — a champion is built *on top of* it, so freezing v8 would be
freezing what is already frozen.

Numbered 6.5 rather than renumbering AK7–AK9, which are cited 27 times across this handoff and the
package README.

**Runbook:** `epyc-inference-research/scripts/kernel_rnd/autokernel/execution/README.md` — 518
lines, written by the session that built the execution layer, cold start to first candidate with
the preflight, the abort conditions, and an honest §6 of what still blocks. It is the operational
document; this phase is the checklist.

**Gating tasks — the campaign cannot produce a result until these close:**

- [x] **THE BLOCKER — no candidate could cross the evidence threshold.** Calibration solves
  `threshold=10`; the sign-martingale over 5 same-sign blocks topped out at **5.5687**, verified
  magnitude-independent (true-effect factors 1.08 → 3.0 all returned the same e-value). Closed by
  building the extension-round producer. Demonstrated both directions: a real +8% effect goes
  `5.5688 → 42.2877` over 10 blocks and banks; a true null stays at `0.9000` and abandons.
  ✅ 2026-08-04
- [x] Extension-round order schedule settled — **extended**, keyed on attempt parity, so
  `derive(attempt=n)` equals `n` chained `retry()` calls. `attempt=0` is byte-identical, so the
  first campaign is unaffected. ✅ 2026-08-04
- [x] Anchor triple bound for **two tools** (T0 hashes `llama-cli`, microbench compares against
  `llama-bench`). ✅ 2026-08-04
- [x] `cpu_region_claim` prefix collision with the candidate-id space closed. ✅ 2026-08-04
- [x] T0 producers wired (`extract_elf_symbols`, `parse_unified_diff`, the `surface.py`
  derivation). ✅ 2026-08-04

> **✅ CLOSED 2026-08-05 — the α budget is machine-protected at the completed-run seam.**
> Previously, a **declared round could be re-run until it crossed**. The pooling seam refused the same round
> object twice, and a round licensed to another campaign, but it cannot refuse a second *run* of
> the same declared round — because a second run of the same plan **is** the same plan. Measured
> null crossing rates at threshold 10, α = 0.1 (40k/10k trials):
>
> | submitted | null crossing |
> |---|---|
> | base only | 0.000% |
> | base + one declared round | 1.355% |
> | best of 5 re-runs | 4.710% |
> | **best of 25 re-runs** | **9.760%** — the entire error budget, on one candidate |
> | best of 50 | 13.120% |
>
> `CompletedRunLedger` now fsyncs every completed run, keyed on
> `(campaign_id, candidate_id, attempt, segment, extension_round)`. The runner refuses a completed
> key before inference; the pooling seam refuses missing, substituted, or conflicting run identities;
> and the stock executing path refuses a missing durable `--journal-root` during preflight. A repeat
> is `attempt + 1` with the retry schedule, never a re-roll of observed evidence.

- [x] Build the per-candidate run ledger that makes the above machine-enforced rather than
  procedural. `MICROBENCH_RUN_COMPLETED` carries the full raw vector plus its content-hash identity;
  runner and pooling tests prove a duplicate key spends no further inference. ✅ 2026-08-05

**Built 2026-08-04 — the loop gained a composable driver:**

- [x] **`campaign.py` — the entrypoint the package spent 94k lines not having.** A `grep` for
  `__main__|argparse|def main(` across every non-test module returned *nothing*: 5,695 tests passing
  in 47 s and no way to start it. Twelve steps, `--dry-run` as the DEFAULT so it cannot benchmark by
  accident on a shared host. ✅ 2026-08-04
- [x] The accept rule, in four conditions rather than a statistics module: T0 all-PASS
  lexicographically first, then over N **pre-committed** pairs keep iff `min(delta) > 0` and
  `median(relative) > drift_bound`. `decide()` refuses any other block count, so optional stopping is
  impossible rather than discouraged. ✅ 2026-08-04
- [x] **The first real A/A this package has ever had** — four runs of identical code, between-run CV
  1.62%/1.88%, and a monotone 4.2% decode decline that makes interleaved pairing mandatory rather
  than merely advisable. Everything was calibrated against synthetic numbers until now.
  `data/autokernel_aa_20260804/`. ✅ 2026-08-04
- [x] `program.md` — the human control surface: the loop procedure plus a hypothesis inbox, modelled
  on `karpathy/autoresearch`'s 114-line equivalent. ✅ 2026-08-04
- [x] Hypothesis path end to end: falsifier optional at entry and **mandatory before compute**;
  adoption **removes** the entry from the operator's store (journal-first, so a crash leaves a
  detectable duplicate and never a loss); the do-not-repeat ledger `check_do_not_repeat()` had always
  consumed and nothing had ever built. ✅ 2026-08-04
- [x] `--hypothesis` wired to the claim, closing the FIFTH "guard defined and never wired" in this
  package — `claim_for_hypothesis` calls itself *"The ONLY route from a hypothesis to a resource
  claim"* and had zero non-test callers. ✅ 2026-08-04
- [x] **~79,600 lines deleted** (`release/`, `adapters/`, `surface/`, the AK4 strategy plane), tag
  `autokernel-preserve-20260804`. Verified first with Python's own import machinery: the driver's
  closure is 22 modules and reached none of it. The review's condemnation of `evaluator/integrity.py`
  and `evaluator/surface.py` was **wrong** — the driver imports both. ✅ 2026-08-04
- [x] Refactor plan Steps 0–2 executed: placeholder digests refused (`BuildProvenance.
  output_binary_sha256` accepted 64 zeros), `produced_by` on the two evidence types that lacked it,
  `Check.worst_of` (an empty vector is COULD_NOT_CHECK, never PASS), `schemas.require`, and the
  self-audit identity binding at `evaluator/api.py` and `evaluator/integrity.py`. ✅ 2026-08-04

**Discovered 2026-08-04 — current disposition:**

- [x] **Answer whether the decode drift recovers after rest** (Step 1 below). Under one q0–q3 claim,
  the old decline did not reproduce: decode rose across the four pre-rest runs; 180 seconds idle made
  the first following run 3.29% colder; the second recovered to 0.50% below the pre-rest run. Do not
  add inter-arm rest; retain pairing and warm once after a long idle boundary. Evidence:
  `epyc-inference-research/data/autokernel_aa_20260805_rest_recovery/`. ✅ 2026-08-05
- [x] Replace the placeholder accept threshold (2.1310%, from four runs on one model) with the first
  campaign's own calibration block. The fresh accepted campaign binds a 3% contribution floor,
  φ=4.9207%, B_min=12 and MDE=2.7408%. ✅ 2026-08-05
- [x] Bind `evaluator/api.py`'s **supplied-source** audit to the module it claims to inspect. ✅ 2026-08-10 — fixed:
  supplied source must define the audited module's declared `MODULE_ID`; absent or mismatched identity
  is `COULD_NOT_CHECK`, while forbidden imports/calls remain `FAIL`. This is exercised across the API,
  correctness, controls, devices, statistics, and control-runner callers.

  This was the
  shared-AST-engine contract and cannot change without a `module_id` kwarg — Step 4 of the refactor
  plan (`capability.py`); the narrow identity-binding fix closed the defect without a policy union.
- [x] The `capability.py` hoist itself: 11 audit functions, ~631 lines, 5 denylist tables → one
  walker + ~200 lines. **Guard rail:** the five denylists are *four different policies*, not four
  drifted copies — `microbench.py`'s list is INC-20260731 encoded as data. Unioning them is a
  category error that would weaken the execution plane while reading as a cleanup. ✅ 2026-08-10 —
  superseded by the operator's lean-loop boundary: keep the distinct policies and shared validation
  primitives; do not restore a broad capability plane before real campaign evidence justifies it.
- [x] A structural audit that every declared guard has a caller. Five instances of "declared and
  never wired" have now been found by hand, the most recent written hours after the refactor plan
  assessed that row as "0 live". Fixing instances is not working. ✅ 2026-08-10 — the campaign-footprint
  suite now carries a five-entry source-level caller contract for worktree mutation refusal, retry-order
  reversal, do-not-repeat, per-control seed rotation, and falsifier-before-claim; a missing live caller
  fails the build.

---

## ▶ START HERE — CLEAN INSTRUMENT → CURRENT CONTROLS → CPU IQK → MATCHED ARCHIVE

The first IQK campaign adapter is statically complete: the loop has an entrypoint, a hypothesis path,
a machine-enforced completed-run ledger, a registered source-free parameter mutation, measured
per-tool anchor bindings, and candidate-specific T0 evidence. The 2026-08-11 audit corrected the
remaining stronger claim: v8 calibration cannot authorize the current v9/hardened cell. A current
identity-bound control bundle is required before Step 3 may rank anything. Generic source proposals
remain separately fail-closed where whole-program absence and rollback cannot be proved; that does
not re-open the registered IQK parameter adapter or weaken the campaign's all-PASS rule.

**Read first, in this order:** `execution/README.md` (the runbook, cold start to first candidate),
then `program.md` (the loop procedure and the hypothesis inbox), then this list.

### Step 0 — before anything, confirm you actually hold the host

The single most expensive lesson of 2026-08-04, and it cost two of six A/A runs inside an hour: a
parallel session brought up the orchestration stack mid-measurement — seven `llama-server`
processes, load 3.3 → 23.9, memory 54 → 306 GB — and the readings collapsed 26%. **The co-tenant did
nothing wrong. We held no claim.**

- [x] Confirm no material CPU contention and that whoever owns the stack knows the host is being
      measured. The resident production stack was operator-confirmed idle; a one-core test process
      was ~0.5% of the 192-thread host, not 89% host-wide. One q0–q3 claim covered the full probe.
      ✅ 2026-08-05
- [x] Verify host canonical state — governor `performance`, THP `always/always`, `numa_balancing=0`.
      ✅ 2026-08-05
**Standing guardrail — do not trust the boost gate at idle.** Measured 2026-08-04: 16 cores above
2.5 GHz idle versus **117 under load**, against a required 80. The gate is evaluated only at
`load/core ≥ 0.25`; a preflight that reads it at idle aborts on a healthy machine. The claimed probe
followed this rule and recorded evidence on 2026-08-05. This is an invariant, not an open task.

*(ADJUDICATED 2026-08-12 by `mainC` — deliberately left CLOSED/non-task. Reads as a
prohibition to the standing-constraint sweep, but the rule is already enforced in CODE and
Steps 0–2 are a linear campaign preflight, not a per-pickup checklist. Do not re-open on a
future sweep.)*

### Step 1 — answer the one open measurement question (≈20 min, cheap, do it first)

- [x] **Does the decode decline recover after rest?** The previous monotone trajectory did not
      reproduce. Decode rose 32.55 → 32.67 → 32.85 → 34.00, then fell to 32.89 on the first run
      after 180 seconds idle and recovered to 33.83 on the second. The cold-first-run effect argues
      against inter-arm rest; retain interleaved pairing and warm once after long idle. Absolute
      throughput also shifted sharply between days, so calibration stays inside one host-state
      window. Evidence: `data/autokernel_aa_20260805_rest_recovery/`. ✅ 2026-08-05

### Step 2 — calibrate the instrument before trusting it

- [x] **Bind the hardened instrument to clean, committed provenance before inference.** ✅ 2026-08-12 —
      `experimental-v9-autokernel-t1-hardening-final` at
      `a4cb04ca8f92fa4d665684490f609b380f9b5e96` has exactly one parent, frozen v9
      `0db32c06e3e550065b78311a6031ef3dd2c4f27c`, changes only `llama-bench.cpp` and its README,
      is clean, and is pushed to the internal fork. The clean CPU build's copied `llama-bench` is
      SHA-256 `19dfba5c65a94e7b27e3db001a8a6a250a91d013bfea73970f4e931f0c1e54b0`.
      A claimed 0–95 CPU smoke emitted all six required hybrid-sync, thread-set, escape-check,
      unsynchronized-sample, and device-sync receipts (`cpu_not_applicable`) and released its q0–q3
      claim. The all-PASS preflight is
      `/mnt/raid0/llm/autokernel/probes/ak-v9-final-preflight-20260812-r1/preflight.json`, SHA-256
      `0baf7b73055f028c5c493afd9a4ab8c9950d3c088ed79f3751102ddff71fdefd`.

- [x] **Run the five fixed controls (§15.2) before any real search**: positive, neutral, negative,
      A/A, and the **historical-win replay — the iqk port, which MUST promote**. The negative control
      is the one that catches a fast-looking wrong kernel, and a loop whose controls have not run is
      an instrument nobody has calibrated. Every benchmark number in the 3,596 tests is synthetic;
      the five controls are the first real ones. The fresh 3%-floor campaign produced a **5/5 PASS**
      panel: positive and historical IQK arms promoted, neutral/A/A stayed below the noise floor,
      and the wrong-work negative received no speed rank. Evidence:
      `data/autokernel_controls_3pct_20260805/`. ✅ 2026-08-05
- [x] Solve the calibration block off the A/A pool and record φ, α_sel, α_conf, the noise floor and
      the MDE this campaign is judged under. The current accept threshold — **2.1310%**, from
      `data/autokernel_aa_20260804/` — is a placeholder from four runs on one model; replace it with
      the campaign's own calibration. A first predeclared 2% campaign correctly rejected because
      its 20-block MDE was 2.5867%; no controls ran under it. A genuinely fresh 3% campaign then
      accepted with φ=`0.049206882811302755`, α_sel=`0.1`, α_conf=`0.05`, B_min=`12`, MDE=`2.7408%`,
      and an A/A false-crossing rate of `1.4%`. ✅ 2026-08-05
- [x] **Make calibration authority era- and instrument-local.** The executing CLI now requires
      `--calibration-bundle`, verifies the bundle's source-label hash, accepted solve/control state,
      production commit, measurement-instrument commit, recipe, floor, B_min/ceiling and MDE, and
      refuses the v8 bundle on v9. Dry run reports `UNCALIBRATED CELL` rather than printing the old
      2.1310% placeholder as live authority. ✅ 2026-08-11
- [x] **Close package-power admission without broadening host privilege.** ✅ 2026-08-12 — research
      commit `1094ff6b` adds a lazy, exact-container-id broker pinned to image
      `sha256:3a2e92b4133d06d1287f96ec47bacd743717b377f4b9df6be1e3af626c35dbb0`, with no network,
      a read-only root, all capabilities dropped, `no-new-privileges`, a 16-PID limit, and only the
      powercap mount exposed read-only. The v9 preflight now passes package-power availability; unit
      coverage proves exact-id stop/inspect/kill escalation and refuses image-identity mismatch.
- [x] **Require a fresh campaign identity and self-consistent evidence root.** ✅ 2026-08-12 —
      research commit `7f8e9997` removes the retired hard-coded campaign id and evidence directory,
      requires `--campaign-id` plus an absolute `--output`, and deterministically derives the seed and
      window id. Declaration, calibration, claim, host, control, evaluator, and summary receipts now
      carry the same identity; invalid ids and relative evidence roots fail before execution.
- [x] **Run the five controls against frozen v9 plus the hardened measurement instrument.** ✅
      2026-08-12 — `ak-controls-v9-a4cb04ca-20260812-r2` accepted at contribution floor `3%`,
      B_min=`10`, φ=`0.03578502357852242`, and passed positive/neutral/wrong-work/A/A/historical
      controls (`5/5`, `may_rank=true`). The historical IQK replay promoted at
      `+0.2660503395746673`. Summary SHA-256
      `81b5bc4b95cfd8e3ca9346c9f44a70fc297c159310e57bdb84f62c80588d78e1`; control-sweep SHA-256
      `9f765c5c19d0ea4669ac55a55796a643b578c4ff1bf126b5f6b630a297197ae1`. The six inference legs
      completed under claim `akclaim-c9f943ba08234877`, whose journal records release at
      `2026-08-12T00:46:23.957344+00:00`. When deterministic post-processing exposed missing
      `CampaignBinding.change_class`, research `c4a42c69` repaired the evaluator and recomposed the
      immutable raw vectors without inference; `composition_attestation.json` binds every input hash,
      evaluator commit `c4a42c69917187b53809c8d4c3267cc1a99a37de`, and
      `inference_executed=false`.
- [x] **Recalibrate the final direct-child IQK instrument after empirical T0 repairs.** ✅
      2026-08-13 — final instrument `283b520b527a7b507d6cf05cd124a59f427f3629`
      is a clean direct child of frozen production `0db32c06…`, contains the committed T0
      receiver plus the Q4_K and Q4_0 narrow-shape fallbacks, and passed all five live controls at
      `/mnt/raid0/llm/autokernel/controls/ak-controls-v9-283b520b-20260813-r1/`
      (`controls_complete`, `may_rank=true`, B_min=11, MDE=1.4562%).

### Step 3 — the first candidate

- [x] **Close the first-campaign `HostOps` adapter before taking a claim.** Supply the candidate
      mutation (or the declared no-source parameter comparison), measured `llama-cli` and
      `llama-bench` anchors, the proposal-owned T0 declarations/registration patterns, and the
      healthy all-core nominal frequency. The CLI's pre-claim refusal is the acceptance test; do not
      bypass it with a placeholder capture. ✅ 2026-08-11 — the built-in proposal-v3 IQK parameter
      adapter proves an empty source diff, derives the registered parameter surface and IQK registration
      patterns, measures separate `llama-cli`/`libggml` anchor identities, and requires `--nominal-khz`;
      `unimplemented_seams()` plus CLI tests refuse before claim if any required input is absent.
- [x] **Turn the six reference-leg T0 unknowns into real evidence for the first parameter candidate.** Close exact reference,
      ASAN/UBSAN applicability, state rollback/teardown, symbol-version coverage, and affected-
      surface reachability. `COULD_NOT_CHECK` remains speed-blocking; do not reinterpret it as PASS.
      ✅ 2026-08-11 — the source-free registered surface proves memory/threading/persistent-state false,
      which resolves ASAN, UBSAN, state safety, and reconciliation by non-applicability; ELF v2 parses
      exported version indices and leaves genuinely versioned exports explicit; `test-backend-ops`
      emits `AK_REF_V1` only after its separately activated CPU reference path passes, and T0 projects
      its observed comparator error/tolerance without mislabelling NMSE as ULP or bitwise identity.
      Generic source proposals still fail closed where whole-program absence or rollback cannot be proved.
- [x] **Live-validate the stateful T0 integrity pass.** ✅ 2026-08-11 — with suite seed `4711`, all
      **5,184/5,184** cases passed across `SSM_SCAN`, `SSM_CONV`, `FLASH_ATTN_EXT`, and
      `GATED_DELTA_NET`. Every `AK_STATE_V1` receipt proved equal initial state inputs, immutable
      input buffers, and one or more compared final-state outputs. The fail-closed consumer is
      research commit `9cc3ed1b`; the experimental producer remains uncommitted pending explicit
      operator approval and was not committed or pushed by this checkpoint.
- [x] **Live-validate the independent host-double fp64 oracle.** ✅ 2026-08-11 — direct GGUF-wire
      decoders cover Q4_0, Q8_0, Q4_K, and Q6_K under metric
      `fp64_error_ratio/host-double-gguf-wire/v1`. Five representative CPU cases, the dedicated
      broadcast regression, 31 real parser tests, and the property self-test's 5/5 planted plus 5/5
      clean cases passed. The full stock ROCm Q4_K matrix separately exposed genuine baseline κ=1.5
      failures; the gate remains fixed and the uncommitted experimental implementation was not staged,
      committed, or pushed.
- [x] **Prepare and fail-closed preflight the first CPU IQK campaign.** ✅ 2026-08-12 — rebound
      the physical envelope to exact deterministic recipe frame
      `e504d8e937fd0f645b8dadb3b971e320f95c2f1bf34fc678b186fbf37bb079b0`; dry run was clean;
      repaired the missing `/workspace/repos/epyc-whisper` and `epyc-qwentts` symlinks; and verified
      both frozen speech kernels. The first live attempt stopped on the missing symlink before claim;
      the corrected attempt stopped in preflight because uptime was `13.47 days`. Neither attempt
      acquired a claim, built a candidate, or benchmarked. The append-only proposal and both
      fail-closed STOP_STATE records remain in
      `/mnt/raid0/llm/autokernel/campaigns/ak-iqk-v9-20260811/events.jsonl`.
  - [x] **Recover and validate the exact current dry-run invocation.** ✅ 2026-08-12 — the recovered
    intervention and matched A/A commands exit zero without host mutation, bind their exact
    `--least-commitment-capture-plan`, `--hypothesis`, and `--hypothesis-store` inputs, and compose the
    current-code steps with the accepted 3% floor, B_min 10, CPU `0-95`, and required
    `--nominal-khz 2500000`. Dry-run authorization does not consume the live authorization: identical
    execute-mode pickup appends a distinct typed spend record. Both durable commands are recorded in
    `progress/2026-08/2026-08-12.md`; governed execution still requires OP-16 and fresh post-reboot
    preflight.
  - [x] **Repair the IQK capture-plan producer/validator bootstrap regression.** ✅ 2026-08-12 —
    `b9bc28a6` made a pre-existing completed held-out journal mandatory at capture-plan load, while no
    such real out-of-regime IQK journal exists; the first campaign could therefore never start. Research
    now makes the state explicit: `bootstrap` plans bind proposal, hypothesis, exact factor frame,
    diagnostics, calibration, and physical envelope but cannot enter AP-WM/archive authority. A later
    `heldout_bound` plan still requires the strict completed-journal projector. Fresh r3 intervention
    and A/A inputs under `/mnt/raid0/llm/autokernel/campaigns/ak-iqk-v9-20260812-r3/` and
    `/mnt/raid0/llm/autokernel/campaigns/ak-iqk-v9-aa-control-20260812-r3/` both dry-run clean.
- [ ] **After a compliant host reboot, rerun the prepared full-host CPU IQK campaign.** The ratified
      one-week uptime ceiling in `measurement/protocols/kernel-research.md` and `bench-cpu.md` is the
      sole remaining preflight blocker. Reuse the accepted v9 control bundle and exact recipe frame;
      preserve the journal and require claim/build/benchmark evidence rather than bypassing preflight.
  - [x] **OP-16 — Operator decision package: authorize the orderly reboot.** ✅ 2026-08-12 — the
    operator granted the AutoKernel session ownership of all compute resources and approved the
    reboot→IQK→archive execution sequence. Preserve the r3 checkpoint, then reboot only after the
    preflight checklist and current-source commit are durable.
    - **Context:** CPU IQK preflight measured host uptime at approximately `13.48 days`; the ratified
      seven-day ceiling requires a reboot before further search measurement. The refusal occurred
      before inference, claim, build, or benchmark, and every current artifact is wrapped and pushed.
    - **Option A — authorize an orderly reboot now (recommended):** preserve the accepted v9 control
      bundle and journal, reboot after all mains report ready, bootstrap the `agent` tmux session per
      session-bus C20, respawn the established roster, then rerun the prepared campaign. This is the
      shortest compliant path and is reversible operationally; reboot downtime is the cost.
    - **Option B — postpone the reboot:** leave the host and production stack as-is. AutoKernel CPU
      measurement remains fail-closed; no result or resource consumption occurs, but the empirical
      chain cannot advance.
    - **Recommendation:** Option A. All pre-reboot durability obligations are satisfied and no useful
      protocol-compliant CPU candidate measurement can proceed at the current uptime.
    - **Default:** Option B; preflight continues to refuse and no inference runs.
- [x] **Close empirical T0 defects surfaced by the post-reboot IQK rehearsal.** ✅ 2026-08-13 —
      r23's real property suite exposed input-dependent Q4_0 `m=16,n=2,k=256` error; Q4_0 now
      falls back below 32 rows, matching the Q4_K safeguard, and the exact linked suite passed
      1,954/1,954. Durable T0 output proved scheduler execution; research `da7d14fa` parses its
      bare timestamp format. Research `430443c5` preserves a legitimate T0 refusal rather than
      misclassifying it as a candidate-record error. r23 produced no throughput result.
- [x] **Reboot and execute the fresh bootstrap-pair attempts through r29.** ✅ 2026-08-13 — r25 reached
      passing T0 but exposed a `MicrobenchPlan.schedule()` driver call defect; r26's T0 truthfully
      rejected the Q6_K `m=16,n=3,k=256` IQK result; r27 refused contaminated host admission; and
      r28 showed that the former load gate counted campaign-owned T0 residual as outside contention.
      r29 intervention reached `DECIDED` with all ten pairs and production PASS, but correctly
      reverted as inadmissible when anchor drift was 15.27% against a 3.08% bound. Its A/A arm
      terminally refused preflight while a separate OpenDataLoader workload held the CPU. None of
      these immutable journals substitutes for the required clean matched pair.
- [x] **Make the CPU IQK instrument and T0→T1 boundary fail closed on the observed failures.** ✅
      2026-08-13 — experimental direct-child `f744cc220e722d1bda93783959471d44f8e118b0` adds only
      the Q6_K `<32` native fallback; its exact seeded property suite passes. Research `deeed365`
      requires every T0 sandbox teardown receipt, a 65-second decay, and three fresh low-load
      attestations before the unchanged T1 admission gate. Fresh controls at
      `ak-controls-v9-f744cc220-20260813-r1` are accepted/rankable (5/5, B_min=10,
      MDE=2.0360%).
- [x] **Harden and calibrate the IQK bootstrap frame through r36.** ✅ 2026-08-13 — r2/r3 exposed
      `-r 5` instability and source/path A/A asymmetry; `798016e3` fixes a sealed `GGML_IQK=0/0`,
      `reps=1` frame and r4 is accepted (5/5, `B_min=15`). `7486a167` refuses out-of-range pairs
      before publication. r34/r35/r36 are terminal non-evidence attempts as recorded above.
- [ ] **Quiesce external CPU work and run a fresh r37-or-later 15-block IQK intervention/A/A bootstrap pair.**
      Bind only r4 and `f744cc220…`; require both arms to terminal `DECIDED` before decode holdout.
- [ ] **Run a clean, governed CPU decode IQK holdout after the final bootstrap prefill pair.** It must use the same
      candidate frame but target `decode`, then be projected from its completed journal before the final prefill pair may
      enter AK-WM-2a. This is the first valid source for `heldout_regime_transfer`; fixtures and the
      prefill pair cannot substitute.
- [ ] **CPU first.** `llama_cpu` needs no GPU device claim and its canonical baseline is the most
      characterised surface we have; `llama_gpu` needs the device claim and contends with whoever is
      serving. The claim reason alone decides it.
- [ ] **Reproduce a known-real CPU win as candidate #1 — the IQK replay.** It is a proposal-v4
      `change_class: parameter` comparison with candidate `ggml_iqk=1` and anchor `ggml_iqk=0`;
      `campaign.py` now projects that registered arm-local variant into the exact dry-run commands.
      A null result on a known win is diagnostic of the *harness*; a null on a novel idea tells you
      nothing about either.
- [x] **Keep the 2026-07-04 async-prefetch replay in the GPU lane.** ✅ 2026-08-11 — the +3% / MemUnitStalled
      reduction on `mul_mat_vec_q8_0_prefetch` is a gfx90a/MI210 result, not a CPU candidate. Its
      original scratch evidence no longer exists, so it was re-derived from the frozen-v9 resident
      implementation and its runtime gate. The governed 20-block replay did **not** reproduce the
      historical win: 19/20 block deltas were positive, but median was only **+0.936%** (mean
      **+1.039%**, range **−0.590% to +3.311%**), below the predeclared 2% floor and with one negative
      block. This is `NOT_REPRODUCED`, not a banked candidate. Receipt:
      `/mnt/raid0/llm/autokernel/probes/ak-gpu-prefetch-v9-20260811-r2/receipt.json`, SHA-256
      `7b173cafcccb8a99319bf93a80fd13a2e94a400afab2bf03355363f9521ab17f`.
- [x] **Repeat the governed v9 async-prefetch replay under an independent 20-block campaign.**
      ✅ 2026-08-12 — all **20/20** paired blocks were positive, but the median delta was only
      **+1.2442303249%** (minimum **+0.6788501205%**), below the predeclared **2%** contribution
      floor. The honest verdict remains `NOT_REPRODUCED`; positivity alone is not promotion evidence.
      The receipt binds frozen-v9 source/binary/linkage/model identities, a held-and-released MI210
      claim, and 2,544 device samples across 635.76 s:
      `/mnt/raid0/llm/autokernel/probes/ak-gpu-prefetch-v9-20260812-r1/receipt.json`, SHA-256
      `7321f2a640a7cc0b3169544867e26e3102a3efe3ebd5b0ebc732e154398339b0`.
- [ ] Then a real one. Drop a hypothesis into the store (`HYPOTHESES.md` has the shape), or run
      exploratory with no `--hypothesis` at all.

The command, from `epyc-inference-research`:

```bash
python3 -m scripts.kernel_rnd.autokernel.campaign --dry-run \
    --model <production-representative GGUF> --candidate <patch-or-branch>
# read all 12 steps, THEN:
python3 -m scripts.kernel_rnd.autokernel.campaign --execute --i-hold-the-host \
    --model <same> --candidate <same> --proposal-manifest <proposal-v4.json> \
    --journal-root <durable, NOT a scratch path> \
    [--hypothesis akh-... --hypothesis-store <path>]
```

`--dry-run` is the default and spawns nothing (proved by `test_readme.py` with every spawn primitive
booby-trapped). **Read the dry run before executing** — that is how the LD_LIBRARY_PATH display
defect was caught on 2026-08-04, and the failure it hid is the worst shape available here: a
candidate linked against the anchor's `libggml` measures the *anchor*, and reports a clean,
well-formed null with every gate passing.

### Step 4 — to a champion

- [ ] Run to a first banked candidate, then compose a champion lineage.
- [ ] Seed further hypotheses from the §19.8 queue and the operator channel.

**What "it is working" looks like:** the negative control REFUSED and the iqk replay PROMOTED;
e-values that move with effect size instead of pinning at a ceiling (they pinned at 5.5687 for every
magnitude until 2026-08-04); the journal growing with banked *and* abandoned candidates; readiness
reporting a figure or an explicit parity standing, never a silent absence.

**What should abort it:** the negative control passing; the A/A arm showing a non-null effect; CPU
frequency throttled under load (a multi-day −60% has happened here); a frozen tree showing any
modification; the claim lost mid-measurement; two arms sharing an `LD_LIBRARY_PATH`.

**Budget:** a full A/A is ~2 min/run; a candidate build is the long pole. Steps 0–2 are roughly an
hour of machine time and are the whole difference between a trustworthy first result and a number
nobody can defend.

**Exit:** a champion exists for at least one source tree, green through T2, with a readiness signal
the operator can read — which is exactly what AK7 requires as its input.

### Phase AK7 — first supervised freeze

- [ ] Operator requests a freeze on a real champion; AutoKernel produces the package.
- [ ] Operator executes the freeze, era rows, and rebaseline; AutoKernel records the receipts.
- [ ] Run T4 post-cutover verification and the watch window.
- [ ] Re-anchor the champion onto the new tip and re-measure (§8.9); record the true cost.
- [ ] Post-mortem the package: what the operator had to add, correct, or reject. Fold back into §10.

**Exit:** one production version has been created from an AutoKernel champion, with reconstructible
evidence, and the loop has re-anchored cleanly.

### Phase AK8 — serving-runtime adapter and research queue activation

- [x] Build the serving-runtime adapter against the three-gate stack-change path (§11.6), reusing
  `stack_change_guard.py` for gate 1 and adding the behavioural half: variable-arrival replay,
  `task_rate` rather than tokens/s, latency and SLO cells as first-class outputs, comparison against
  the pinned production configuration. ✅ 2026-08-03
- [x] Prove the adapter **refuses** the kernel-freeze path rather than degrading to it. ✅ 2026-08-03
- [ ] Populate campaigns from the §19.8 seed queue after fresh profiling.
- [ ] Activate `oracle_port` campaigns; consider an upstream-delta scanner once the manual path works.
- [ ] Seed external kernel-authoring suites only after license, gfx90a, honest baseline, quarantine,
  and evaluator-integrity gates pass.
- [ ] Retire/supersede narrower loop handoffs once their remaining tasks are owned here or by a
  backend leaf.

### Phase AK9 — speech backends

Split out of AK8 because it is not one bullet. `whisper.cpp` and `qwentts.cpp` have **no measurement
protocol of any kind** — `measurement/protocols/` contains nothing for STT or TTS — so this phase
authors two protocol families from scratch, which is work comparable to Annex K, plus two adapters.
It also carries the ggml-generation hazard: the three trees run three ggml generations, so every
launcher sets its own `LD_LIBRARY_PATH` and proves it, or a binary silently runs against another
tree's ggml.

- [x] Author the STT protocol: correctness corpus and output normalization (what counts as a
  transcription match), real-time factor, latency, throughput, memory stability, audio input identity,
  and the release decision rule. ✅ 2026-08-03
- [x] Author the TTS protocol: text/audio identity, deterministic and numerical checks, an
  intelligibility/quality proxy with a human-independent floor, first-audio latency, RTF, throughput,
  stability, and the release decision rule. ✅ 2026-08-03
- [x] Decide each protocol's annex placement at drafting time, per the §14 AK0 precedent. ✅ 2026-08-03
- [x] Build the `whisper_stt` and `qwentts_tts` adapters on their own experimental trees. ✅ 2026-08-03
- [x] Wire linkage verification through the research repo's `scripts/utils/verify_ggml_linkage.sh` for
  every candidate build and every T3 phase-2 check. ✅ 2026-08-03
- [x] **Extend the release-plan compiler to the independently freezable speech trees.** ✅
  2026-08-12 — research `99fe3014` restores tested `whisper_stt.release_binding()` and
  `qwentts_tts.release_binding()` adapters that return the generic compiler's `BackendBinding` with
  exact source tree, stable path, phase vocabulary, Annex S prerequisites, tree-local ggml generation
  and `LD_LIBRARY_PATH`, complexity ceiling, and canary requirement. The generic compiler accepts
  single-backend speech targets without inheriting llama CPU/GPU union semantics; T3 consumes both
  adapter readiness predicates, enforces the `production-speech-vN` branch family, and reconstructs
  both independent trees from the preserved speech freeze. A focused exact-main adapter/plan/T3
  audit passed **783 tests with 132 subtests**.

- [ ] Extend the release-plan compiler to the speech trees — note these two *are* independently
  freezable, unlike CPU and GPU (§1.5). **UNBLOCKED 2026-08-03**: Annex S is ratified, so
  `P-STT-1`/`P-TTS-1` are in force and `PHASES_BY_BACKEND` can now carry the speech phase
  vocabulary honestly. It was withheld only because the compiler cannot grade a cell under an
  unratified protocol; that reason is gone. Note the whole speech plane (`adapters/`) was deleted
  on 2026-08-04 and lives at tag `autokernel-preserve-20260804` — restore from there rather than
  rewriting. Research branch `codex/ak9-speech-compiler-20260812` at `b367d09f` is a pushed proposal
  under primary review; this checkbox remains open until that implementation is accepted and merged.
- [x] `Annex S` ratified — `measurement/protocols/speech.md`, 86 KB, five annexes in
  `MEASUREMENT.md`. `P-STT-1/2/3`, `P-STT-REL-1`, `P-TTS-1/2/3`, `P-TTS-REL-1` are in force; the
  four verdict grammars carry no `attest <ref>`, reconciled with Annex K's own reasoning.
  ✅ 2026-08-03
- [x] Speech WER supersession ratified — `ratify_speech_wer_correction_20260804.json`, 2.35 → 3.37.
  The recorded figure was the `faster-whisper` CTranslate2 arm, a different engine; the production
  `whisper.cpp large-v3-turbo f16 MI210` arm measures 3.37 (63/1870). Forward-looking: every future
  `whisper_stt` non-inferiority comparison would have inherited that denominator. ✅ 2026-08-04
- [x] `latency_s_11s_clip` audited and found CORRECT — 11 s / `xrt_overall` 51.86 = 0.212 s against
  the recorded 0.21, nearest competing arm 3× away. So the defect was one copy error on one line,
  not systematic mis-sourcing, and the receipt's other anchored values need no re-audit.
  ✅ 2026-08-04

#### What remains open after AK9, and why

Each line names the specific blocker, per *Act, Don't Defer*.

| Open item | Blocked on | Not blocked — just undone |
|---|---|---|
| AK6 end-to-end campaign stopping at a validated package | — | Step 3 first candidate |
| ~~AK6 `/kernel` freshness contract + JSON contract v2~~ | — | **REBASED 2026-08-05.** The prior `autokernel/surface/` producer was deleted on 2026-08-04 while the hub still named it. The surviving campaign path now exports the fsynced terminal `STOP_STATE` through compact `autokernel/dashboard.py`. The hub separately shows committed implementation and durable `data/autokernel_*` activity, structurally excluded from liveness/health. Runtime truth remains absent until the first real campaign exports. |
| AK7 first supervised freeze | a CHAMPION, which needs AK6.5 Step 3 | — |
| AK8 seed queue, `oracle_port`, external suites | fresh profiling, which needs the loop running | — |
| AK9 speech compiler extension | — | **Ratification landed 2026-08-03; no longer blocked.** Restore `adapters/` from `autokernel-preserve-20260804` when a speech campaign is scheduled. |
| `qwentts` round-trip WER names no STT instrument | a quiet host — it needs re-measuring, not correcting; `P-TTS-2` now requires `stt_instrument=` so the figure cannot satisfy its own protocol | — |
| Sub-floor estimate selection in `readiness.py` | **operator call** — excluding them makes a phase measured entirely at parity report "no figure" | — |

**All three protocol drafts are ratified** (Annex S, 2026-08-03) — `measurement/protocols/speech.md`.
`P-KERNEL-FREEZE-1` remains a draft, so `t3.phase_identity_preflight` still raises
`ReleaseProtocolNotRatified` for `mode="release"` and **every run this package can perform is a dry
run** — the correct posture, and now the only remaining protocol gate.

---

## 15. Acceptance program

### 15.1 No-inference acceptance

Schema mutation/fuzz tests; journal/view reconstruction after a crash at every transition; worktree
path-escape and production-path denial; shared-clone branch-collision and pathspec containment;
evaluator hash/source drift refusal; candidate snapshot symlink/FIFO/hardlink/TOCTOU attacks; fake
self-reported score ignored; **actor scope under-declaration detected**; release version collision and
incumbent-modification refusal; resource-claim contention, revocation and drain/resume behaviour; **operator-control fault injection — a pause, drain, or
abort the loop fails to acknowledge, latch, or survive a restart against is a hard failure, not a slow
one**; journal-versus-derived-view cardinality disagreement at BOOTSTRAP;
**GPU device-claim contention, stale-lock reclamation, and non-preemption**; owned-process
TERM→KILL→verified-empty; storage quota and tombstoned expiry; package idempotence and rollback-plan
completeness.

### 15.2 Controlled evaluator acceptance

Five fixed controls: **positive** (a known correct optimization with a real bounded mechanism);
**neutral** (a correct change whose effect distributes around zero); **negative** (deliberately
fast-looking but wrong, cheating, falling back, or cached); **A/A** (anchor versus anchor); and
**historical-win replay** — a real past improvement, the iqk port being the obvious candidate,
replayed end to end through T0–T2 and, at release rehearsal, T3, which **must promote**.

The fifth exists because the other four all test the gate's ability to *reject*. Nothing else tests its
ability to *accept*, and without that `EXHAUSTED_SURFACE` and `PLATEAU_STOP` are indistinguishable from
a gate that has quietly stopped passing anything. AutoPilot ran 1,055 trials of which 8 were even of a
promotable type, while every surface reported "active, blockers: []".

Required: positive ranks above anchor under T1; neutral does not spuriously advance; negative never
receives a speed rank; A/A produces no significant effect at the declared rate, and its failure voids
the measurement window; **the historical-win replay promotes, and a failure to promote is a gate defect
rather than a research finding**; all five remain correctly classified after restart/replay; the actor
cannot alter the control definitions; and holdout shapes and control seeds rotate on their declared
schedule.

### 15.3 Planner acceptance

On a fixed fixture the planner/controller must propose a falsifiable target tied to the profile;
survive compile and correctness failures; consume failure memory in the next proposal; not reopen a
matching do-not-repeat item without new evidence; check the oracle registry before authoring; respect
the optimization hierarchy; combine compatible winners and remeasure the composed champion; report
readiness against the anchor without declaring it; and stop on plateau/budget/storage/evaluator gap
without human prompting.

### 15.4 Release acceptance

T3 derives the correct distinct production model/recipe scope for the source tree; every metric binds
raw samples, protocol, category, reps, host/resource receipt, binary/linkage, model, anchor, and
evaluator; production-optimal cells gate and diagnostic baseline cells do not; per-phase non-inferiority
is evaluated under each phase's own protocol; quality replay/transfer avoids unnecessary inference; a
failed cell produces a stable failure bundle and cooldown; a waiver is verified by hash and predicate
and suppresses exactly its claims; a passing bundle reconstructs the exact next-version transaction; the
package contains a verified rollback anchor and never writes production; T4 detects a stale running
process and requires a controlled restart before success.

### 15.5 Definition of complete

AutoKernel is complete when an unattended campaign can start from the current production tip, author
novel source changes, learn from failures, conserve evaluation cost, maintain a green champion, report
honest readiness, and — on request — produce one sealed, fully evidenced, operator-executable release
package, with reconstructible evidence and without the actor modifying its judge, its scope, or its
production target.

---

## 16. Initial campaign policy

Do not hard-code the source draft's ranked kernel list into the controller. At first activation:

1. regenerate the current production roofline and wall-share map;
2. assign at least 90% of the target campaign's wall time to measured mechanism classes **where the
   required counters are actually available**;
3. compile the current do-not-repeat ledger with receipts;
4. select one bounded backend/regime campaign;
5. validate the loop on the positive/neutral/negative/A-A controls; and
6. only then allow novel research.

The draft's most important research families remain hypotheses to re-rank against fresh evidence, not a
permanent execution order: GPU low-bit GEMV/layout and persistent/grouped MoE; GPU LM-head/sampling,
graph, recurrent, mixed-KV, and joint speculation policy; CPU NUMA/locality, operator-cluster fusion,
persistent teams, grouped MoE, LM-head, and hybrid quantization; continuous batching, paged KV, and
capability dispatch; and alternate-engine capability audits as design oracles.

---

## 17. Decision log

| ID | Decision | Rationale |
|---|---|---|
| AK-D1 | One stable T3 program is both the final AutoKernel evaluation and the kernel-freeze evaluator | Prevents search/promotion truth divergence |
| AK-D2 | T0/T1/T2 are separate cheaper research tiers | Full release evaluation per change would dominate compute cost |
| AK-D3 | **Superseded 2026-08-02.** The +25%/+20% figure is a readiness signal, not a trigger | Removes threshold peeking, winner's-curse inflation, and the accumulation-versus-fresh-anchor tension |
| AK-D4 | Readiness is computed deterministically; the LLM may request only | Prevents narrative estimates from spending the release matrix |
| AK-D5 | Actor, evaluator, and packager have separate authority, enforced at the OS level | Full compute access must not imply self-grading; hook-based enforcement does not reach a daemon |
| AK-D6 | **Superseded 2026-08-02 (operator).** No delegated auto-freeze; AutoKernel produces a release package and a human executes it | An automatic freeze crosses four human-only boundaries, including era rows and the AutoPilot rebaseline, and would preempt the inference owner's reload authority |
| AK-D7 | Append-only sharded event journal is primary; SQLite/Pareto is derived | Failures and superseded candidates remain durable and reconstructible |
| AK-D8 | One core loop with backend adapters | Avoids a GPU-only architecture and duplicated safety logic |
| AK-D9 | Scheduler/runtime research uses the same controller but a stack-change release adapter | Scheduler wins are not kernel freezes |
| AK-D10 | Evaluator changes remain human-amendment-only | The optimizer cannot rewrite its judge |
| AK-D11 | Campaigns are per backend; champions and freezes are per **source tree** | CPU and GPU share `llama.cpp` and one frozen branch; only whisper and qwentts are independently freezable |
| AK-D12 | Objective is per-backend, per-phase non-inferiority plus improvement at production-optimal recipes | A cross-device composite is forbidden by `MEASUREMENT.md:83-84` and `gpu-cross-device.md:106-111` |
| AK-D13 | Operator waivers are a first-class T3 input | v8 shipped on `promotion_decision: false` plus a hash-pinned scoped waiver; a binary gate would have blocked it |
| AK-D14 | The affected-surface manifest is derived and traced, never declared | Otherwise the actor sets its own release scope |
| AK-D15 | Statistics use the sanctioned e-process, pre-committed stopping rule, and anchor gate | "LCB" appears nowhere in the constitution; e-processes are anytime-valid, which a per-round guard requires |
| AK-D16 | `oracle_port` is a first-class campaign kind | The largest recent kernel gain was the iqk port, not de-novo authoring |
| AK-D17 | Evidence is durable and classified, not merely hashed | `MEASUREMENT.md:146-156`, ratified 2026-08-02 |
| AK-D18 | A backend owes new evidence only if its binary changed, tested by build-system source closure and confirmed by normalized comparison against an anchor rebuild — **not** by naive byte-identity, which never fires because builds embed IDs, timestamps, and paths | Confines the P-GPU-1 circularity to changes that reach the HIP build, and makes CPU-local campaigns cheap despite the shared tree |
| AK-D19 | The GPU consumption clause is urgent; the decision-grade circularity is patient | The former blocks every GPU T1 round and AK3's exit; the latter blocks only a freeze that needs new GPU evidence |
| AK-D20 | **Two attestations, not one:** "search authorization" after AK3, "release authorization" before the first freeze | Batching guards against a per-experiment ratification cycle; it does not require items whose referents appear months apart to share a signature, and forcing them together would ratify release bindings against schema sketches under an append-or-version constitution |
| AK-D21 | **AK1+AK2+AK3 is a standalone deliverable**, planned as AutoKernel's first release rather than as scaffolding, with a program kill criterion after AK3 plus two campaigns | Evaluator, journal, and GPU claim are useful to human-driven sessions immediately; the planner is the part that must earn its keep, and a 50–75 session program needs an honest early exit |
| AK-D22 | Anchor identity is re-verified at **every campaign boundary**, not only at freeze; `ANCHOR_MOVED` supersedes comparisons while preserving source and correctness results | A hot-fix or rollback between freezes otherwise leaves every ratio in the journal with a denominator that no longer exists, undetected |
| AK-D23 | `serving_runtime` releases through the three-gate stack-change path (§11.6) on `stack_change_guard.py`, measured in `task_rate` under variable arrival | Pipeline-green, starts, and live-equals-config are distinct and none implies the next; tokens/s is not substitutable for task_rate in scheduler scope |
| AK-D24 | Speech is its own phase (AK9), not an AK8 bullet | `measurement/protocols/` contains nothing for STT or TTS — two protocol families must be authored from scratch, comparable in size to Annex K |
| AK-D35 | Roofline utilisation — measured decode t/s over bandwidth-derived theoretical t/s, with MoE counted by active-expert bytes and both datasheet and measured-achievable denominators recorded — is the normalising metric for discovery headroom, oracle intake and the harvest/explore phase signal. Diagnostic and routing input only, never a gate | Raw speedups do not transfer across hardware and are the main way a published result misleads this project; utilisation does transfer, bounds headroom physically, and can retire a seed from the paper alone at zero compute |
| AK-D32 | Harvest/explore are **phases switched on marginal yield**, not a fixed budget fraction; the decay floor and window are derived, with a minimum dwell and a `PLANNER_DEGRADED` disambiguation | A fixed ratio spends explore budget while a fresh region still yields and caps exploration once it is dead; a freshly opened region's adjacent wins are cheap, high-probability and perishable, so they should be stripped first |
| AK-D33 | Spikes owe no anchor gate, paired blocks, e-process or confirmation sample — they emit a mechanism verdict, not a rate claim — but still hold a claim and pass preflight | Institutional cost is spent confirming gains, not discovering them; a spike that costs what a T1 costs will not be used |
| AK-D34 | The oracle registry declares a **harvest class** per row on the axis of architectural portability (`portable_source` vs `reimplement`), **not licensing** — standing policy is open-source self-hosted, non-commercial, licences not blockers and new oracles enter via `research-intake`, not by an agent adding a row | Misclassifying is a schedule problem, not a legal one: a `reimplement` oracle costs authoring effort a `portable_source` one does not. Intake still verifies real gfx90a/EPYC support and normalises the claimed result to roofline utilisation before a port is proposed |
| AK-D37 | **AK-D36 excludes a *target*, not a *regime*.** Single-stream and batched prefill and decode are all legitimate optimization directions, and AutoKernel looks for improvement independently of batch count. What AK-D36 forbids is recruiting the whole-stack llama.cpp-vs-vLLM ratio as a kernel objective, because that ratio is dominated by scheduling above 16 concurrent users | A coarse 2026-07-04 B=128 profile originally made G15 look like the highest-confidence GPU band; the strict 2026-08-11 family map later closed G15 itself. That falsification reinforces the rule: batch regimes remain valid, but targets must be selected by current mechanism-specific metrics rather than a whole-stack ratio or a coarse non-GEMM remainder |
| AK-D38 | **Operator hypotheses are a first-class planner input**, carrying an explicit falsifier, entering at `design_prior` evidence grade and never above it, tracked still-open until resolved, and subject to every gate without exception | The operator sees things the profile does not; without a channel that steering arrives as an out-of-band instruction with no falsifier and no resolution record. Grading it `design_prior` is what stops a hunch being laundered into a measured fact. **Note the sibling comparison in §8.4.0 was corrected on 2026-08-03: AutoPilot does not have still-open tracking either, its falsifier is optional and observability-only, its open-set block is stagnation-gated, and it has no evidence-grade vocabulary at all — so this is a new mechanism in both loops rather than parity with one** |
| AK-D39 | **ROCm libraries, compiler choices, and source-available modules are isolated candidate providers, not new champion source trees.** AutoKernel may select, tune, or fork them, but a win must integrate through an experimental `llama_gpu` candidate and pass operator plus whole-model gates; shared `/opt/rocm` mutation and opaque-binary champion source are forbidden | The measured gap is regime-dependent: dense GEMM provider choice is shape-specific, while the large Q8/IQ2/MoE utilization collapse lives mainly in custom low-bit/layout/gather paths. Broad search across both layers is warranted, but promotion must still name the deployable `llama.cpp` lineage and preserve the frozen-stack trust boundary |
| AK-D31 | Architectural campaigns replace three §8.4 rejection conditions rather than waiving them: predicted post-change profile for the wall-share ceiling, prospective shapes, and per-step conceptual-change scope; plus spikes and a reserved budget fraction | Those three are correct for incremental work and would block the deep kernel rethinking the loop exists to find; EIG-first ranking starves high-variance work by arithmetic unless budget is reserved |
| AK-D29 | Source-integrity gates run **before** behavioural gates: symbol/registration preservation, clean build from snapshot, semantic diff conformance, repair from clean parent | AutoPilot's one autonomous source mutation destroyed a module with a syntactically valid edit; none of its four Python defenses transfer to compiled C++, where "it compiles" is far weaker than "it imports" |
| AK-D30 | `core_header` is its own change class and risk tier, not a size band | A small textual diff to shared ggml core reaches every op in both the CPU and GPU builds |
| AK-D26 | Planner narrative is separated from the machine record and excluded from retrieval by default; supersession can be retrieval-scoped | AutoPilot's worst contamination regenerated from its own prose inside an append-only journal, where scrubbing derived stores never stuck |
| AK-D27 | A fifth control — historical-win replay that MUST promote — joins positive/neutral/negative/A-A | The other four test rejection; nothing tested acceptance, leaving a dead gate indistinguishable from an exhausted surface |
| AK-D28 | A freeze is one pre-validated apply bundle covering all four boundary writes, not four ceremonies | The doctrine is one signature over a consolidated bundle; four hand-assembled artifacts per freeze is the avoidable recurring cost |
| AK-D25 | Freeze cadence defaults to **on readiness or quarterly, whichever comes first**, as an operator policy rather than a loop parameter | Cadence trades accumulated value per freeze against champion drift, re-anchor cost, and how often the highest-risk code is exercised |
| AK-D36 | **Closing the llama.cpp-vs-vLLM gap is not a kernel goal and must not become a campaign objective.** At batch-1 llama.cpp is 0.1–5.8% *faster* on an RTX 4090 and comparable on an H200; on our MI210 vLLM leads by **+11%**, and that 11% *is* the kernel delta. The headline 24–44× arrives only at **16–64 concurrent users** and is continuous batching, PagedAttention and the scheduler — a `serving_runtime` question under AK-D9/AK-D23, not a kernel freeze. **Caveat that rides with this row:** every public batch-1 head-to-head is at **16-bit**, i.e. exactly where our gap is not; the quantized-vs-quantized head-to-head **has never been run anywhere**, so a future measurement could move the +11%, though not the concurrency decomposition | A whole-stack throughput ratio recruited as a kernel target would spend a kernel campaign on a scheduler property. Naming the decomposition once stops it being re-derived — and no statement to this effect existed anywhere in the six MI210/autokernel handoffs before 2026-08-03 |

---

## 18. Reporting, escalation, and ownership

1. Update this handoff's checkbox at the same commit that lands the work; append `✅ YYYY-MM-DD`.
2. Put backend-specific evidence in the owning backend leaf; keep controller/release status here.
3. Update [inference-research-index.md](inference-research-index.md) only when priority,
   dependency, or phase changes.
4. Append `progress/YYYY-MM/YYYY-MM-DD.md` after each significant phase.
5. Never record an observation as a release claim; cite the active protocol, category, reps, and
   attestation.
6. **The running loop does not write handoffs, index rows, or intake entries.** CLAUDE.md restricts
   those to explicit approval, and `MEASUREMENT.md:164-166` routes demoted numbers to
   `handoffs/active/measurement-debt/`. AutoKernel emits bus artifacts and decision packages; a session
   lands them.
7. Every operator escalation — `OPERATOR_INPUT_REQUIRED`, `EVALUATOR_COVERAGE_GAP`, a reboot request,
   a phase-trade exception, a release package — is rendered as Context / Options / Recommendation /
   Default.
8. Emit a periodic operator digest: what was learned, what was rejected and why, what is banked in the
   champion, current readiness, storage and budget standing.
9. **Report discovery statistics, not just outcomes.** T1-banked → survived T2 → survived T3; refuted by
   futility versus refuted by budget exhaustion; realized cost per banked change; proposals filtered
   before dispatch, by reason. AutoPilot's most expensive blind spot was that 0 of 121 refutations came
   from futility — every one came from the budget rule — and no surface said so.
9. At completion, extract stable architecture/runbook material to `docs/`, move this handoff to
   `handoffs/completed/`, and remove its active-index row.

### Key implementation locations

- **Runtime owner:** `epyc-inference-research`.
- Controller/evaluator/journal/importer/store: `epyc-inference-research/scripts/kernel_rnd/autokernel/`
  and `scripts/rnd_harness/`.
- Canonical benchmark recipes: `epyc-inference-research/scripts/benchmark/`.
- Evidence root: `epyc-inference-research/data/<campaign>/` with `SHA256SUMS` and README.
- Resource claims: `epyc-orchestrator/scripts/region-lock` (CPU) plus the new device claim (§14 AK2).
- Compiled production scope: `epyc-orchestrator/orchestration/derived/stack_priors.yaml`.
- Kernel path resolution: `epyc-orchestrator/src/registry/kernel_paths.py`.
- Scope compiler seed: `epyc-orchestrator/scripts/validate/kernel_freeze_scope.py`.
- Release policy and receipts: epyc-root `measurement/`, `docs/reference/`, `artifacts/operator/`.
- Dashboard: epyc-root `dashboard/`.
- Experimental source: namespaced worktrees under `/mnt/raid0/llm/`.
- Production source: read-only to every AutoKernel component.

---

## 19. Bootstrap corpus — seeding the loop with what the project already knows

Absorbed 2026-08-02 from the separate bootstrap-corpus design pass. Without this, AutoKernel starts as
an amnesiac system and spends its first months rediscovering work already done.

### 19.0 Decisions

1. **Import every idea from the draft; do not make every idea an executable hypothesis.** The draft is
   a source of design priors. Each idea is atomized, audited against current source and project
   history, and assigned a disposition before the planner may spend resources on it.
2. **Import historical project research as typed legacy events.** Inherit prior wins, failures,
   conditional results, confounds, and reopen predicates without pretending AutoKernel ran them.
3. **Keep three compiled memory products:** a research-prior catalog, an executable campaign-seed
   queue, and a do-not-repeat/constraint ledger — all derived from one append-only journal.
4. **Never upgrade evidence during import.** An observation stays an observation; a source audit stays
   a source audit. Only protocol-bound evidence keeps decision authority.
5. **Suppressing entries carry a higher bar than the wins they block** (§19.3). A win needs
   protocol-bound evidence and mechanism confirmation; a negative that closes a whole family must not
   be allowed in on confident prose.

### 19.1 The journal is not an idea list

```text
source draft + historical handoffs + artifacts + source audits + external research
                                  |
                                  v
                    append-only typed import events
                                  |
                +-----------------+------------------+
                |                 |                  |
                v                 v                  v
       research-prior catalog  campaign seeds  constraint/negative ledger
                |                 |                  |
                +-----------------+------------------+
                                  |
                                  v
                     planner context compiler
                                  |
                                  v
                  new proposal/candidate/eval events
```

Legacy imports never populate the live Pareto frontier and never masquerade as candidate evaluations.

**Research-prior catalog.** A prior is one atomic, scoped statement: a possible mechanism; a mechanism
supported or refuted by prior evidence; an implementation pattern; a known optimized path already in
source; a workload/regime fact; an evaluator/correctness requirement; a transfer restriction; or an
unresolved contradiction. One prose section often becomes several priors — draft item G2 becomes
separate records for batch-one workgroup sizing, asynchronous prefetch, execution layout, activation
quantization reuse, persistent execution, and the already-refuted generic Q8 dequant premise.

```yaml
schema: epyc.autokernel.research_prior.v1
prior_id: akr-...
family: gpu.low_bit_gemv
statement: "..."
claim_type: design_hypothesis  # mechanism_fact | result | negative | constraint | source_fact
disposition: untested          # supported | refuted | conditional | superseded | conflicted
regime:
  backend: llama_gpu
  hardware: gfx90a
  architecture_class: dense
  phase: decode
  batch_band: batch_one
  quant: q8_0
mechanism: memory_level_parallelism
evidence_grade: design_prior   # source_verified | observation | protocol_bound | imported_claim
era: null
source:
  repo: epyc-root
  path: handoffs/active/...
  locator: "section/line/event"
  content_sha256: "..."
  durability_class: carried_in_git   # carried_in_git | durable_untracked | hash_and_provenance_only
applicability: {requires: [], excludes: []}
transfer_limits: []
reopen_when: []
contradicts: []
supersedes: []
seed_eligibility: measurement_first
imported_at: "..."
```

`design_prior` means "worth considering", not "probably true". Source and content hash let a later
importer detect drift without silently rewriting history.

**Executable campaign seeds.** A seed exists only when the question, regime, prerequisite, cheap
discriminator, correctness oracle, and resource ceiling are known.

```yaml
schema: epyc.autokernel.campaign_seed.v1
seed_id: aks-...
derived_from_prior_ids: []
question: "Does ... under ...?"
backend_adapter: llama_gpu
campaign_kind: source_change   # config | dispatch | layout | fusion | scheduler | capability | oracle_port
change_class: dispatcher       # selects the §9.5 cheap suite
target_regime: {}
prerequisites: {source_receipts: [], profile_receipts: [], evaluator_features: []}
cheap_discriminator:
  tier: T1
  target_shapes: []
  anchor: production
  mechanism_counters: []
  recipe_id: "..."             # codified recipe, never a hand-typed command
expected_counter_direction: {}
wall_share_ceiling: null
correctness_oracles: []
non_target_sentinels: []
estimated_cost_class: small
estimated_storage_gb: 0
status: blocked_measurement    # ready | blocked_* | exhausted | retired
reopen_predicates: []
```

The queue is a derived view. Reprioritization never alters the prior or experiment history.

### 19.2 Do-not-repeat and constraint ledger

Not every negative is permanent.

| Class | Meaning | Planner behaviour |
|---|---|---|
| `HARD_CONSTRAINT` | Hardware, policy, correctness, or ownership prohibition | Reject matching proposal |
| `MATCHED_NEGATIVE` | Mechanism falsified in a matching regime with adequate evidence | Reject unless an explicit reopen predicate is newly satisfied |
| `CONDITIONAL_NEGATIVE` | Failed only for named shapes/models/batches/eras | Exclude matched cells; other regimes remain eligible |
| `CONFOUNDED_RESULT` | Unusable because identity, placement, cache, or baseline was wrong | Do not learn its sign; preserve as a trap; require a repaired experiment |
| `SUPERSEDED_FACT` | Current source or production behaviour invalidates the old premise | Do not execute the stale proposal; regenerate from current source/profile |
| `LOW_VALUE` | Plausible but below the wall-share/effort threshold | Deprioritize; reopen on changed exposure or implementation cost |

Each entry requires exact match dimensions and a `reopen_when` predicate. "Do not repeat" without
regime identity is dangerous because this project repeatedly observes sign changes across architecture,
substrate, batch, context, and quant.

**Seed entries — established capability facts, do not re-derive (2026-08-03, research-intake Stage-2b).**
These are the inverse of a negative: claims that were asserted, checked, and **overturned**, which is
exactly the shape a planner will otherwise rediscover from the literature every few months.

| Entry | Class | Content | `reopen_when` |
|---|---|---|---|
| `gfx90a-has-direct-global-to-lds` | `SUPERSEDED_FACT` | **"gfx90a lacks the CDNA3 asynchronous buffer-load machinery" is FALSE.** Corroborated three independent ways: (a) LLVM — `FeatureVMemToLDSLoad` sits inside `FeatureGFX9` and gates 8 builtins via `"vmem-to-lds-load-insts"`; (b) AMD's own GEAK CDNA2 doc — *"Direct global→LDS exists on CDNA2 (32 b/lane, like CDNA3)"*; (c) our own tree already calls `llvm.amdgcn.raw.buffer.load.lds` on gfx90a (`mmvq.cu:538,602`). What gfx90a genuinely lacks is the **async DMA engine** (no TMA/`cp.async`/mbarrier) and the **SMEM-operand matrix instruction** — different limitations with different consequences | Never for the capability itself. The *performance* question (whether the linear-deposit form we use is leaving anything on the table vs an HBM-side-swizzled deposit) is open and lives in the G-family |
| `hipkittens-tile-layout-is-ours` | `SUPERSEDED_FACT` | HK's `rt_base` is **bit-identical** to `ggml-cuda/mma.cuh`'s `tile<16,16>` in frozen v8. A planner proposing "port HK's tile abstraction" is proposing to re-derive a layout we already ship | Source change to `mma.cuh`'s AMD MFMA branch |
| `mfma-decode-kernels-are-worth-zero` | `HARD_CONSTRAINT` | At batch-1 arithmetic intensity (1.0–5.2 FLOP/byte, 31–113× below the 110.5 ridge) the matrix units cannot exceed ~1.7–3.2% busy **at any bandwidth**. `MfmaUtil ≈ 0%` is physics, not a defect. Authoring MFMA decode kernels returns 0 | Batch size at or above `B* = 110.5 × bytes_per_weight / 2` (Q4_K 31, Q8_0 59, bf16 110) — i.e. this is a *decode* constraint, and batched/prefill regimes are explicitly not covered |
| `cdna2-abandoned-by-vendor-and-quant-schools` | `HARD_CONSTRAINT` | AITER's supported-hardware table lists **no MI210/MI250/gfx90a, not even experimental** — consumer RDNA parts are supported ahead of our datacenter card. TileLang is CDNA3-limited; the quantization×kernel co-design school (ZipServ, EXL3, Escha, Sakana) is ROCm-excluded across the board. **Nobody will port anything to this card for us** — which is the premise AutoKernel exists to answer, not a reason to stop | A vendor or project announcing gfx90a support; re-checked at each freshness sweep |

**Contradiction detection is mandatory at compile time.** An entry is checked against live operator
decisions and against sibling entries; anything that contradicts either becomes `conflicted` and is
never authoritative. This is not hypothetical: twice in one session an agent let a committed artifact
override the operator's restated decision, and the ruling was that *the artifact is the thing that is
wrong*. A machine-maintained suppression ledger is versioned, greppable, and receipted — which makes it
the most authoritative-looking artifact in the system and therefore the most dangerous one to leave
unchecked.

### 19.3 The receipt rule for suppressing entries

A wrong suppression is invisible: nothing ever tests it again. Therefore every `HARD_CONSTRAINT`,
`MATCHED_NEGATIVE`, and `SUPERSEDED_FACT` must carry:

- a **source receipt** — commit plus path plus line, or an artifact hash — not a confident sentence;
- a **binding to the production commit** it was verified against; and
- **re-verification on anchor move** — when the champion is re-anchored after a freeze, every
  source-derived suppression is re-checked, and a suppression whose receipt no longer resolves reverts
  to `conflicted` rather than continuing to block.

Required evidence grade scales with breadth: a family-wide suppression needs `source_verified` or
`protocol_bound`; a single-cell exclusion may rest on an observation.

This rule applies directly to the §19.4 audit verdicts below. Statements such as "CDNA2 already uses
stream-K in the relevant MMQ path", "generic HIP graph enablement is already implemented and measured",
and "generic Q8 dequant is a stale premise because the path is integer-native" are the exact entries
that will close research families, and they enter the ledger only once each carries a receipt.

### 19.4 Journal event types for bootstrap knowledge

```text
LEGACY_SOURCE_DISCOVERED   PRIOR_ATOMIZED           CONSTRAINT_COMPILED
LEGACY_EVIDENCE_IMPORTED   PRIOR_SOURCE_VERIFIED    SEED_COMPILED
PRIOR_SUPERSEDED           PRIOR_CONTRADICTION_LINKED  SEED_BLOCKED / SEED_REOPENED
```

### 19.5 Importing the project's historical research

Relevant knowledge spans active/completed/archived/blocked handoffs in `epyc-root`; ratified
measurement artifacts and preserved raw run bundles; dated progress files as discovery pointers rather
than automatic truth; the compiled wiki and deep dives; `epyc-inference-research/docs/data`,
`docs/experiments`, `docs/design`, benchmark results, model registry, and research analyses; kernel
commit history and saved negative patches; the research-intake index and external source records; and
current production/experimental source audits.

**Import precedence** — when sources disagree, link the contradiction rather than overwriting, and
apply this order for executable decisions:

1. current source and immutable artifact identity;
2. current ratified protocol-bound evidence;
3. preserved raw evidence with reconstructible identity;
4. active/completed handoff interpretation tied to those artifacts;
5. observation-grade run summaries;
6. progress/archived prose;
7. external literature and design priors.

Higher precedence does not delete lower-precedence history; it determines which fact the context
compiler presents as current and which as superseded or uncertain.

**Distillation pipeline:** enumerate (content-hashed source manifest — never let an LLM pick a few
memorable handoffs and call the corpus complete) → route by backend/architecture/op/regime/mechanism/
evidence type → atomize one statement per prior with its exact locator → verify statements about what
code already does against current source → grade, retaining the source's measurement authority → resolve
duplicates, contradictions, supersessions, confounds, transfer boundaries → compile ledgers with exact
matches, reopen predicates, and receipts → compile seeds only where the missing evidence and cheapest
discriminator are explicit → freeze a hashed bootstrap snapshot → continuously ingest new material by
the same typed path, never by ad-hoc Markdown scraping mid-campaign.

Automation is acceptable because imported priors gain no release authority. Ambiguous records become
`conflicted` or `imported_claim`, never silently resolved.

**Retrieval per proposal round:** exact regime-matched hard constraints and negatives; current source
facts and dispatch behaviour; top mechanism-matched supported/refuted priors; adjacent-regime transfer
warnings; current champion interactions; oracle coverage; and a small novelty set from untested design
priors. Every returned item carries IDs and source locators, and the critic receives the same matched
negatives so the planner cannot omit inconvenient history.

### 19.6 Audit of the draft's optimization program

**Baseline and diagnosis**

| Draft family | Distilled disposition | AutoKernel treatment |
|---|---|---|
| **P0.1 exact operator roofline** | Correct and foundational — an instrument-building task, not a candidate optimization. Existing CPU/GPU profiles reduce the blank space but do not form one current v8 cross-regime map | Compile a production-anchor profile manifest and wall-share map before novel campaigns; refresh affected cells after a freeze, never the whole system per candidate |
| **P0.2 fixed-shape vs continuous serving** | Correct separation; kernel and scheduler effects have historically been conflated | Make benchmark class part of evaluator identity: fixed-shape feeds kernel campaigns, variable-request feeds `serving_runtime` |
| **P0.3 normalized scoreboard** | Correct derived view, but must not be a hand-maintained source of truth | Derive from journal events and capability rows; bind every row to regime and candidate identity |

**GPU families.** Current anchors: [MI210 speed-campaign summary](../completed/mi210-speed-campaign-summary.md),
[Q8/GEMV roofline](mi210-q8-dequant-gemv-roofline.md),
[graph techniques](gemma-challenge-kernel-techniques-v7.md),
[acceleration index](inference-research-index.md).

| Draft family | Audit verdict | Seed disposition |
|---|---|---|
| **G1 host/PCIe/pinned-memory NUMA** | Strong, cheap, still relevant. The restored four-arm P2-5j placement protocol at `docs/design/p2-5j-host-thread-placement-sweep-protocol.md` is the one execution owner — consume it, do not reinvent a duplicate sweep. **Node attachment resolved 2026-08-03: the MI210 is on NUMA node 1** (verified three ways — `/sys/class/drm/card2/device` `0x740f`, `/sys/bus/pci/devices/0000:43:00.0/numa_node`, KFD topology node 4). **The GPU lane's host threads at 184–191 are on NUMA node 3** per `numactl` — neither device-local nor adjacent, and device-local placement has never been tried. **Scope narrowed by measurement 2026-08-03**: bulk H2D/D2H transfer is node-INDEPENDENT (all four nodes within 0.1%), so this placement costs nothing on the transfer path; the open question is host-side memory access during serving. The link is measured Gen4 x16, not PCIe 5.0. **2026-08-11 constitution audit:** CPU-affinity variation makes P-BENCH-PLACEMENT-1 controlling; the old four-arm design lacks its full composite. The strict finalizer/context bridge is complete and observation-only; it preserves all arms and cannot select a placement | `READY_OBSERVATION_ONLY`; run only to prioritize a compliant successor. A decision needs human protocol ratification or a full P-BENCH-PLACEMENT-1 execution; then ingest the receipt without relabelling it as kernel speedup |
| **G2 batch-one low-bit GEMV** | Too broad as written. Generic Q8 dequant is a stale premise (integer-native path); generic megakernel and several speculation/prefetch variants have matching negative or conditional history; workgroup sizing and prior prefetch changes are legacy priors, not patches to replay on v8. Layout/coalescing and MLP questions remain conditional on a fresh profile | Split into atomic priors; seed only current-profile gaps — coalescing/layout if cache-line evidence is poor, or another gfx90a MLP lever if the production path still exposes it |
| **G3 GPU-native low-bit layouts** | Sound high-upside family, but only if metadata traffic, cache-line use, unpack cost, or occupancy is currently deficient. Startup time and VRAM/context cost are first-class outputs | `MEASUREMENT_FIRST`; T1 starts with load/repack accounting plus captured GEMV/MMQ shapes, then one tiny real graph |
| **G4 persistent grouped MoE** | Valid high-effort family for batched MoE; must be distinguished from the persistent stream-K MMQ path already present. "Persistent" alone is not novelty | `CONDITIONAL_HIGH_EFFORT`; require routing/expert wall share, grid/occupancy evidence, and a target batched regime before source work |
| **G5 stream-K/split-K** | Broad premise superseded — CDNA2 already uses stream-K in the relevant MMQ path; compact-LDS was negative. A narrow higher-persistent-grid-count experiment remains conditional with a small ceiling | Import generic stream-K as `SUPERSEDED_FACT` **with a receipt**; retain the exact residual as `LOW_VALUE_CONDITIONAL` with its existing reopen gate |
| **G6 shared-expert fusion** | Plausible only where shared/routed scheduling and accumulation have material wall share; overlaps G4 | Compile as sub-seeds under grouped-MoE campaigns after wall-share discovery |
| **G7 LM head/logits/top-k/sampling** | Good portable family spanning projection and exact/distributional sampling correctness; atomize rather than propose one giant fusion | `MEASUREMENT_FIRST`; profile tail share, then isolate projection, partial top-k, and sampler fusion seeds |
| **G8 shape-bucketed HIP graphs** | Generic graph enablement already implemented and measured; some graph-key/onegraph variants were neutral, conditional, or deferred. Surviving question is exact shape-recapture/launch-gap evidence for a named workload | Mark generic capture `ALREADY_SUPPORTED` **with a receipt**; seed only when a shape histogram shows costly misses in an uncovered family |
| **G9 recurrent/GDN/SSM decode fusion** | Valid architecture-specific family, but history already rejects an occupancy rewrite and supports a state-format lever in particular regimes. Targeted fusion needs current per-block wall share and strict rollback/state tests | `CONDITIONAL`; import both the positive state-format prior and the occupancy negative. Never seed a generic GDN optimization |
| **G10 chunked recurrent prefill** | Closed by the governed direct-profiler gate in [`k28-fused-chunked-gdn-kernel-research.md`](../completed/k28-fused-chunked-gdn-kernel-research.md): optimistic 4x-op full-model ceiling falls 11.55% → 9.14% from 2K → 32K | `CLOSED_NO_GO`; retain as negative seed evidence, never synthesize a fused-kernel task without a new materially higher ceiling |
| **G11 mixed-KV specialized attention** | One of the clearest remaining concrete gaps: the default path can fall back on mixed q4/f16 while blanket all-quant support harms protected homogeneous paths | `READY_AFTER_PROTOCOL`; seed a dedicated mixed-format path with no-fallback proof and homogeneous f16/f16 and q4/q4 sentinels |
| **G12 joint speculation-depth/batch/context policy** | Strong and cheap, but a serving/config policy rather than a kernel source freeze; prior results prove the sign is regime-specific | `READY_SERVING_ADAPTER`; search the existing parameter surface cheaply and release through `serving_runtime` |
| **G13 vLLM capability audit** | Useful design oracle, but a broad rerun is stale work — earlier local audits already separated dense-control advantages from gfx90a/model blockers | Import the existing capability result; reopen only on a source-version, supported-model, quant, or gfx90a capability change. Port algorithms via `oracle_port` (§6.5), never unsupported instructions |

**Three GPU families added 2026-08-03 (research-intake Stage-2b).** All three are quant-ladder or
non-GEMM findings that the original G1–G13 audit had no measurement to see. Their common premise: on this
device **fp16 already attains 62.6% of spec bandwidth and vLLM-ROCm attains 69.2%**, so the memory system
is not the limiter and the entire collapse is down the quant ladder — the ladder and its spec-vs-achievable
calibration caveat are in [`mi210-q8-dequant-gemv-roofline.md`](mi210-q8-dequant-gemv-roofline.md).

| New family | Audit verdict | Seed disposition |
|---|---|---|
| **G14 architect MoE-IQ2 batch-1 GEMV** | `Qwen3.5-122B-A10B UD-IQ2_M` attains **10.3%** — our worst rung by 2×, and a **production-serving** model, so a win lands on a live role rather than a benchmark. Reaching even the Q4_K rung takes 43.7 → ~145 tok/s. **But a kill-criterion must be attached before funding:** on gfx906 an optimised community fork **and** vLLM independently converge on ~10% bandwidth for MoE batch-1 — the same rung. Two independent stacks hitting one wall means this may be an **architectural floor**, not a kernel gap | `CONDITIONAL_PROBE_FIRST`; the cheapest discriminator is not a kernel — establish whether the ~10% MoE batch-1 rung is architectural before any source work. Band if it is not: +2–3×, MED-LOW confidence |
| **G15 batched elementwise/norm fusion (GPU) — CLOSED_NO_GO 2026-08-11** | The old **43% non-GEMM** bucket was too coarse to fund fusion: it conflated gather/scatter and runtime copies with actual elementwise/norm work. A clean frozen-v9 B=64/128 trace now separates the families. Verdict-bearing norm + activation + elementwise share is only **1.837% / 1.490%**, while B=128 is instead dominated by gather/scatter **18.631%** and recurrent work **17.464%** | `CLOSED_NO_GO`; the predeclared 20% target-selection floor is missed by more than 13x. Retain the exact cluster table as negative seed evidence; do not author G15 fusion without a new current trace above that floor |
| **G16 MoE expert-gather GEMV** | Our dense/MoE attainment ratio is **2.1×** (fp16 62.6 → MoE-Q8 21.3) against NVIDIA's **1.3×** (GB10 dense 77–80 → MoE-Q8 59.6) on the same engine. That excess ratio is the gather, not the quant — it is the one place where our sag is measurably worse than a comparison platform's rather than merely present | `MEASUREMENT_FIRST`; first artifact is an expert-token histogram plus gather wall share at the production routing skew, then one atomic gather experiment. Band ~2.0×, MEDIUM |

**CPU families.** Current anchors: [CPU optimization index](inference-research-index.md),
[prefill compute](cpu-prefill-compute-large-models.md), and the research repository's preserved CPU
optimization reports.

| Draft family | Audit verdict | Seed disposition |
|---|---|---|
| **C1 exact DDR/NUMA roofline** | Correct discovery prerequisite. Existing work proves placement, mmap/first-touch, batch, prefill, and decode cannot share one CPU verdict | Compile current v8 regime facts; refresh only campaign-relevant cells; mark superseded pre-NPS4 and confounded placement results |
| **C2 operator-cluster fusion** | Highest-value source family, but must be profile-selected; decode and prefill have different ceilings, and existing barrier/CONCAT work plus rejected subprototypes provide rich priors | `READY_PROFILE_SELECTED`; seed exact clusters with graph-node/barrier prediction, not "fuse CPU ops" generically |
| **C3 persistent CPU thread team** | Plausible but large and race-prone; should follow local fusion/dispatch work only if residual OpenMP/barrier share remains material | `BLOCKED_BY_C2_RESIDUAL`; cheap first experiment is a scheduler/barrier trace or bounded prototype, then race/teardown stress |
| **C4 expert-local NUMA placement** | Valid question, but historical private/shared results include a later-discovered mmap/first-touch confound and topology changes; import as confounded/superseded, never learned as a negative sign | `REBASE_AND_MEASURE_FIRST`; require current topology, explicit allocation policy, routing skew, and production anchor |
| **C5 grouped CPU MoE** | Strong only for batched/eval/prefill regimes; existing eval-batch evidence means the workload trigger is real, and batch-one must not be a sentinel for expected gain | `READY_BATCHED_REGIME`; seed after the exact target workload and expert-token histogram are compiled |
| **C6 CPU prefill continuation** | Useful catalog, but substantially overlaps C2, C7, and the existing PC-4 lineage ([`cpu-prefill-compute-large-models.md:152`](cpu-prefill-compute-large-models.md), PC-4 open); several subideas already profiled or rejected | Import as a family; compile only unresolved PC-4 successors. Do not create a second catch-all prefill campaign |
| **C7 automatic CONCAT dispatcher** | Concrete and high-readiness: a default-off path and positive/negative regime split already exist in the current production lineage | `READY_POSITIVE_CONTROL`; ideal early campaign for learning a safe dispatcher with prefill positives and decode regression sentinels |
| **C8 CPU LM head/sampling** | Plausible and portable with G7 but profile-gated; full-logit avoidance changes observable sampling behaviour unless exact requirements are explicit | `MEASUREMENT_FIRST`; separate projection and sampler seeds |
| **C9 tensor-class hybrid quantization** | Valuable system idea, but partly an artifact/quality optimization rather than a kernel-only mutation; it changes residency and quality as well as dispatch | Keep as an adjacent `artifact_quantization` family; require an immutable quality objective and a separate release adapter before activation |

**Serving and scheduler families**

| Draft family | Audit verdict | Seed disposition |
|---|---|---|
| **S1 token-budget continuous batching** | Strong and partly scaffolded; existing eval-batch work imports as prior evidence, remaining question is representative reliability/latency and production policy | `OWNED_EXISTING_SERVING_SEED`; `serving_runtime`, never kernel freeze |
| **S2 block/paged KV** | Legitimate architecture project when KV reservation/copying limits concurrency or context; not a cheap generic optimization absent that profile | `CAPABILITY_GATED`; first seed is allocator/KV traffic and capacity measurement, not a rewrite |
| **S3 capability registry** | Required infrastructure, not a performance hypothesis; existing orchestrator registry work supplies patterns but does not replace a kernel capability surface | Implement as controller/evaluator substrate; it should compile valid dispatch regions from evidence |

**Autoresearch sections A1–A6** are controller requirements, not campaign seeds: A1 → immutable
campaign objectives and backend adapters; A2 → superseded by §7.2's fuller proposal schema; A3 → the
T0–T3 state machine; A4 → lexicographic correctness/quality/stability before performance; A5 →
evaluator-owned cache and prompt policy; A6 → the §8.3 planner hierarchy enforced by source/profile
receipts.

### 19.7 Resolved local source material

The draft's two presentation references resolve to local records:

- **E5 CPU/NUMA results** — `artifacts/operator/e5_w0_preliminary_results.html`, SHA-256
  `dcd8aba913bd1296406098338e578079ae47678e2494539aaecfb6376ee54c37` (verified 2026-08-02; tracked and
  committed). Its sibling markdown declares itself *"SUPERSEDED AND NON-AUTHORITATIVE — 2026-07-30 …
  Do not read it as the record"* and is imported only as a `SUPERSEDED_FACT`, never as generation or
  claim authority.
- **GPU model-selection surface** — `/mnt/raid0/llm/tmp/claude-artifacts/np_context_v8_decision.html`,
  SHA-256 `816ad5cdd532634edb48f608321fb6ffc3d5546c3ff74aa6c7b54cf0655e6e2b` (verified 2026-08-02),
  backed by `epyc-inference-research/artifacts/np_context_study_v8_20260727/` and
  `.../np_context_study_20260723/`. **This is a scratch path and therefore not a valid citation under
  `MEASUREMENT.md:146-156`; the file must be copied into the research evidence root and both backing
  bundles tracked before AK1 hashes the manifest** (§3.7). The v8 bundle currently tracks only its 5
  driver scripts; the 2026-07-23 bundle tracks nothing.

AutoKernel imports content-hashed local files only, never hosted presentation links, and follows claims
through to the research bundles and measurement artifacts. HTML surfaces are discovery/presentation
records, not independent evidence authority.

### 19.8 Recommended bootstrap seed queue

Every draft idea enters the prior catalog; the initial *executable* queue is:

1. **P0 current-v8 workload/shape/wall-share compiler** — fills missing instrument facts and prevents
   stale-premise source work.
2. **C7 CONCAT dispatcher** — known default-off substrate, regime-specific positives and negatives, and
   strong correctness coverage make it an excellent positive control for planner/critic logic.
3. **G1 host-thread/pinned-memory placement** — existing protocol, no kernel source mutation, a clean
   test of config search plus resource discipline (restore the protocol to git first).
4. **G11 dedicated mixed-KV kernel/dispatch** — concrete fallback defect with protected homogeneous
   sentinels.
5. **C2 exact profile-selected operator cluster** — first genuine CPU source-authoring campaign.
6. **G7/C8 output-tail profile** — decides whether projection or sampler work earns a source seed.
7. **G3 low-bit execution-layout discriminator** — only on cells whose current counters support it.
8. **C5/G4 grouped MoE scheduling** — after real batch/expert histograms are compiled.
9. **G9/G10 recurrent work** — through the existing K28 and state/rollback gates.
10. **S1/G12 serving policy campaigns** — through the serving adapter when that release path exists.

**Inserted 2026-08-03 (research-intake Stage-2b).** These do not displace the ten above; they slot by
readiness, and two of them are cheaper than anything currently in the list:

0. **Measured achievable MI210 bandwidth** (§14 AK1) — one hour, no kernel, and it is the denominator of
   every roofline number the loop will read or emit. Ranked ahead of P0 because P0's profile manifest
   consumes it.
3b. **G15 batched elementwise/norm fusion — CLOSED_NO_GO 2026-08-11.** The strict frozen-v9 B=64/128
   map puts norm + activation + elementwise at only 1.837% / 1.490%, far below the predeclared 20%
   floor. The old 43% non-GEMM remainder was not an operator cluster: at B=128, gather/scatter is
   18.631% and recurrent work 17.464%. Do not fund fusion from the coarse historical bucket.
7b. **G16 MoE expert-gather** — after the expert-token histogram C5/G4 already requires, so it costs one
   shared instrument rather than its own.
9b. **G14 architect MoE-IQ2** — last on purpose. Its kill-criterion probe runs first and may retire it
   at near-zero cost; funding a kernel before that probe is how a campaign gets spent on an
   architectural floor.
3c. **G17 gfx90a WGM launch-order locality — CLOSED_NO_GO 2026-08-11.** The real stream-k MMQ
   none/2/4/8/16/32 sweep passed 43/43 correctness in every cell, but WGM0 won wall time and every
   nonzero mapping regressed 1.286–4.050%. WGM8 also reduced all-MMQ TCC hit rate from 67.304% to
   59.849% at nearly flat read-request volume, falsifying transfer from the synthetic L2 proxy. Keep
   WGM0 and retain the negative receipts in INF-36. Budget briefly returned to G15, whose strict
   current profile then closed it; advance to G18 / the remaining measured seed queue instead.
7c. **G18 Q4_K superblock-unpack attribution — MECHANISM CONFIRMED, CEILING REFRAMED 2026-08-11.**
   The representative `m=17408,n=1,k=5120` single-pass PMC matrix found Q4_K versus the same-bit
   Q4_0 control at equal 34,816-wave dispatches: +112.5 VALU and +35 INT32 instructions/wave, with
   +11.751% median device duration. Exact inside-kernel wall share remains unidentifiable because
   unpack is fused, so no share was invented. The former +38–43% single-lever expectation is not
   supported; any successor proposal must name the exact instruction subset it removes and treat
   roughly 10.5% as diagnostic Q4_K-to-Q4_0 headroom, not a promotion claim. INF-37 owns the next
   surgical source hypothesis. Receipt SHA-256 `1e34339c1c986413c4eeb1b56ba3202c8763d08df45aba1c0580917c888f5e47`.
   The first exact source hypothesis was correctly falsified before timing: replacing lane-local
   8-element Q8 sums with the stored 32-element block aggregate failed 5/5 representative correctness
   cases while frozen v9 passed 5/5. INF-37 now requires ISA accounting that separates scale/min
   unpack from the necessary subgroup sums before another source change.
   That accounting attributed 20/35 INT32 instructions/wave to required subgroup sums and bounded
   the residual scale/min plus address/control budget at 15/35. The follow-on branchless scale/min
   candidate passed 5/5 representative correctness cases and, in a balanced dirty-source diagnostic,
   reduced median device duration **10.554%** while increasing VALU/wave **9.238%** and INT32/wave
   **11.538%**. The direction therefore appears to be exec-mask/control-flow reduction rather than
   fewer dynamic instructions. Receipt SHA-256
   `de4241bd26b77f5dac7df746d165034b67e6f8105133daf0359142a97dd35d5d`; no promotion claim is
   admissible until explicit experimental-tree commit approval and a clean governed replay.

**Inserted 2026-08-10 — v9 DSpark low-hanging upstream queue.** These are concrete merged patches or
bounded monitors, not speculative research themes. Source state was rechecked on 2026-08-10; the
candidate receipt is
[`artifacts/audit/v9-dspark-autokernel-base-20260810.json`](../../artifacts/audit/v9-dspark-autokernel-base-20260810.json).

1. **PR #26171, transpose-free GEMV — forward-port audit first.** Merged; removes a rocBLAS GEMM
   fallback for shared-expert gates. Upstream measured ~78.4 us → ~3.8 us per instance on ROCm
   gfx1151. This is the highest-priority concrete transfer check for our MoE shapes.
2. **PR #25532, backend sampling multi-output — forward-port audit second.** Merged; directly enables
   backend sampling during speculative verification and reports ~8% on Qwen3.6/RTX 5090. Gate on
   exact token/acceptance parity and MI210 operator support; do not infer CUDA transfer.
3. **PR #26731, quantized copy launch fix — context-shift-only audit.** Merged and bit-identical;
   upstream shows large CPY microkernel gains but explicitly no token-generation uplift. Run only if
   the selected long-context workload actually spends time in quantized KV context shifts.
4. **PR #26767, RMSNorm/MUL/RoPE fusion — HIP portability check.** Merged CUDA patch with a concrete
   ~1% end-to-end result. It enters the queue as a source/dispatch compatibility audit, not as an
   assumed MI210 win.
5. **PR #26812, split-block argmax — monitor only.** Open; 1.5–2.2x ARGMAX microkernel result but only
   0–1% reported end-to-end DSpark uplift. Do not spend this work window on it.
6. **Local batch-invariance seed — correctness before speed.** Quantized recurrent target verification
   reproduced upstream #25618. Experimental v9 serializes greedy DSpark verification to preserve
   parity. AutoKernel may attempt to recover parallel verification only when batched/grouped target
   rows are bitwise equal to serial rows and the exact greedy token-parity gate passes.

A bootstrap/acceptance sequence, not a permanent research priority. Fresh profiles and production
workload exposure re-rank it.

### 19.9 Bootstrap acceptance criteria

- Every P0/G1–G13/C1–C9/S1–S3/A1–A6 draft section maps to at least one typed prior or controller
  requirement.
- Every performance-relevant active/completed historical handoff is present in the source manifest or
  explicitly excluded with a reason; the same holds for the research repository's evidence directories
  and the kernel trees' commit history.
- Imported observations, claims, source facts, and design hypotheses retain distinct evidence grades.
- **Every suppressing entry carries a receipt bound to the current production commit** (§19.3).
- Known stream-K, graph, Q8, GDN, NUMA, CONCAT, prefill, speculation, and batching corrections are
  retrievable under their exact regime.
- Contradictions and confounded historical results are visible to both planner and critic.
- No legacy import enters the live candidate Pareto frontier; no `legacy_unverified` store row is
  retrievable at all (§14 AK1).
- Every executable seed names its cheapest discriminator, codified recipe id, correctness oracle,
  non-target sentinels, resource and storage class, and reopen/stop predicate.
- A fixed fixture reconstructs all priors, seeds, and ledgers byte-for-byte from the event journal.
- **A held-out recall probe passes.** Byte-for-byte reconstruction proves *determinism*, not
  *coverage* — it cannot tell you the importer missed an entire class of sources. Before the corpus is
  declared complete, hold out a probe set of facts known to exist in the project record, spanning
  every source class the manifest claims (root handoffs, research-repo evidence directories, ratified
  artifacts, kernel commit history, intake records, and current source audits) and every disposition
  class (a supported prior, a matched negative, a confounded result, a superseded fact, an
  already-supported source fact). Run the compiled retrieval for each under its own regime. A probe
  item that does not surface is a coverage defect in the importer, not a retrieval tuning problem, and
  it names the source class to re-enumerate. Record the probe set and its results in the bootstrap
  snapshot so the same test reruns after every corpus refresh.
- The planner context remains bounded, cites every retrieved item by ID and source locator, and renders
  external content in quarantine form.
- T1 positive/neutral/negative/A-A controls calibrate sampling and noise floors before novel research.

---

## 20. Prior-art gating and the known-optimization catalogue — 2026-08-09 (research-intake Stage-2b)

Source: [intake-1026](../../research/intake_index.yaml) (SGLang `llm-torch-profiler-analysis` skill) and
intake-1029 (its `fuse-overlap-catalog.md` / `overlap-catalog.md`), plus intake-1028 (the OpenMLE
sandbox service). All dive-verified 2026-08-09; SGLang is Apache-2.0.

**The gap this section closes.** §8.4 rejects a proposal when "the same mechanism was falsified under
matching conditions **by an entry carrying a receipt**" — a gate over *our own* negative history. §19
imports *the project's own* historical research. Neither gates against **upstream prior art**. A loop
that cannot tell "nobody has done this" from "this is mainline elsewhere and we simply do not have it"
will spend kernel-authoring budget rediscovering ports.

- [x] **AK-CAT-1 — Add a four-way prior-art classification to §8.4 PROPOSE.** ✅ 2026-08-10. Before a finding may be
  called novel, classify it as exactly one of: (a) an existing path that should already apply here;
  (b) an existing path that appears disabled, unsupported or regressed in this trace; (c) a pattern
  mainline upstream but missing locally, or still open upstream; (d) genuinely new, only when no
  catalogue row fits. **Buckets a–c exit to a config, flag or port change**, not to a campaign.
- [x] **AK-CAT-2 — Build a gfx90a/llama.cpp prior-art catalogue (§19 corpus).** ✅ 2026-08-10. Copy the reference
  five-column schema exactly: `| Pattern | Trace keywords | Primary code | Existing path | Reader
  should conclude |`, with trace keywords bound to `rocprofv2` kernel names and primary code to
  ggml/llama.cpp symbols. **The pre-written conclusion column is the load-bearing one** — it moves the
  verdict out of model judgment at read time and into reviewable data at authoring time, which is the
  same move MEASUREMENT_POLICY's claim grammar makes for measurement claims. Partition rows **mainline
  vs in-flight**: a pattern merged upstream but absent from FROZEN `production-consolidated-v8` is a
  **PORT, not a research proposal**, and that distinction currently has no column anywhere.
- [x] **AK-CAT-3 — Build the expected-absence register BEFORE the catalogue rows.** ✅ 2026-08-10. The reference
  carries a 15-row toggles table whose column is literally "effect on trace interpretation", including
  cases where a flag *intentionally* disables a fast path so split kernels are expected. Without it,
  every legitimately-disabled path reads as a missing optimization. Our substrate is
  [cpu-kernel-env-flags-inventory.md](../completed/cpu-kernel-env-flags-inventory.md), which inventories flags
  without their expected trace consequence. This is the three-states-not-two discipline as data.
- [x] **AK-CAT-4 — Adopt the pinned-head refresh discipline.** ✅ 2026-08-10. The reference pins the upstream head of
  every scanned project with a dated note and ships the exact scan commands, making staleness
  *measurable* rather than asserted. Record the commit each scan was taken against.
- [x] **AK-CAT-5 — Add a cumulative GPU-time-share floor as a proposal-space pruner** ✅ 2026-08-10 (reference
  default `1.0%`). §8.4's wall-share ceiling validates a candidate *after* it is generated; a floor
  prunes the space *before* generation. Complementary, and cheaper.
- [x] **AK-C6-1 — Name syscall confinement explicitly in C6's acceptance criteria.** ✅ 2026-08-10. intake-1028 is a
  purpose-built, professionally structured distributed execution sandbox — controller, dispatcher,
  router, Docker workers with `--cpus`/`--memory`/`--gpus device=` limits — that nonetheless launches
  **every** worker with `--security-opt seccomp=unconfined`, on containers whose entire job is running
  untrusted model-generated code. "Runs in a container" is not "sandboxed". Relatedly, its
  `openmle_gym/sandbox_exec.py` is named for a sandbox and documented as executing "inside the
  configured OS sandbox" while implementing `spec.loader.exec_module` in-process with no isolation —
  a live instance of the naming hazard this loop's C6 item exists to prevent.
- [x] **AK-C6-2 — Harvest the controller/dispatcher/router/worker + SQL job-store split as a design
  comparator** ✅ 2026-08-10 for owned-scope candidate execution, specifically its **idempotency migration** — the
  mechanism for not double-running a submitted job after a controller restart. That is the same class
  of problem as autopilot rewind having to purge the strategy store. Comparator only; the stack itself
  is declined (external registry images, cgroup-v1 host reboot).
- [x] **AK-KM-1 — Record the model-tree-vs-kernel-tree search rule in the §19 corpus.** ✅ 2026-08-10. Verifying a
  symbol's absence in a framework's *model* file alone produces a false negative; this session nearly
  reported a source as fabricated for exactly that reason before finding all four symbols in the
  *kernels* tree. Absence claims must name the trees searched.
- [x] **AK-DEL-1 — SCOPE-REDUCTION GATE: measure the bucket split before building any novel-kernel
  proposal generator.** ✅ 2026-08-11 — replayed AK-CAT-1 over a preserved real `rocprofv2` trace,
  normalized to remove host-identifying absolute timestamps and bound to the original trace SHA-256.
  The 9 dispatches formed 3 admitted kernel families; all 3 landed in bucket (a), with 0 in (b)–(d).
  The report therefore selects `expand_catalogue_before_novel_generator`. This is deliberately a
  corpus-bounded scope result, not a claim about all workloads. Evidence:
  `epyc-inference-research/data/autokernel/prior_art/ak-del-1-k25-q8-mmvq-n1-20260717/`, research
  commit `df02169e`. The catalogue was refreshed against frozen v9 and generic `mul` matching was
  tightened so it cannot misclassify RMSNorm/MUL/RoPE fusion rows.
- [x] **AK-DEL-2 — Review and integrate the next bounded gfx90a prior-art catalogue expansion.** ✅
  2026-08-12 — reviewed commit `677a2eec` landed on research `main` as `35f10715`. The bounded rows
  cover gfx90a fused top-k MoE, gfx90a flash attention, and ROCm MMQ-versus-rocBLAS as present prior
  art while retaining HIP `TOP_K` as unsupported; corrected source pins and the negative control remain
  fail-closed. This expands routing knowledge only and does not manufacture candidate evidence.

Implementation evidence for AK-CAT-1–5 and AK-KM-1 is
`epyc-inference-research/scripts/kernel_rnd/autokernel/prior_art.py`, its reviewed JSON seed catalogue,
and `test_prior_art.py`. The schema requires the expected-absence register, pinned commits, refresh
commands, and both model/kernel tree classes; buckets a–c have deterministic non-campaign exits and
`proposal_space` applies the 1% cumulative-share floor. AK-C6-1/2 are resolved in §3.6; they change the
acceptance contract, not the deliberately retained execution plane.

---

## 21. Transfer ratios, screening lanes, and baseline honesty — 2026-08-10 (research-intake Stage-3)

_Via `/research-intake` (8 URLs + 2 operator-supplied inline documents → 16 entries, 5 Stage-2 dives,
13 Stage-2b ingest-and-dives). The operator's steering through the session reframed the batch from
"what do these sources say" into one design question:_ **how does AutoKernel turn newly-recognised
freedom — concurrent CPU and GPU campaigns, partitioned CPU screening, frontier-API planners, no
per-run operator approval — into robust decision-making, rather than into more ways to be wrong faster?**

**The answer the dives converged on, without any of them naming it: every cheap lane is a proxy with
a measured transfer function to expensive ground truth.** Half-machine → full-machine, op → graph,
T1 → T3, screen → verify are not four problems. They are one object, and it is **one field on the
evaluation event that is free to record now and impossible to reconstruct later** — the same shape as
the `claim_anchors` lesson from the intake index, where 1,067 entries turned out to have zero
citable spans because nobody recorded the locator while the source was open.

### Three corrections to this handoff's own premises, from operator steering

Recorded because they are load-bearing and because I asserted the opposite earlier in the session:

1. **AutoKernel experiments do not need per-run operator approval.** §1.3 and §4 already say it —
   *"No autonomous freeze or cutover. AutoKernel produces a release package; a human executes it."*
   P-GPU-1 governs the **class of claim** a result may carry, not permission to run. The human
   boundary is freeze / cutover / promotion.
2. **CPU and GPU campaigns can run concurrently.** §"Resource plane" already lists CPU region claims
   and the exclusive GPU device claim as **separate resource types**; concurrency was never forbidden,
   it was unexercised.
3. **AutoKernel's objective is experiment churn, not aggregate throughput.** Deep CPU partitioning
   costs aggregate tokens/s, which the orchestrator cares about and this loop does not. The only cost
   that counts here is **rank inversion**, and that is measurable rather than assumed (AK-LN-2).

### AK-TR — the transfer machinery

- [x] **AK-TR-1 — Record a per-change-class transfer ratio on the evaluation event.** ✅ 2026-08-10. Add
  `anchor_tier` and `transfer_ratio_to` to §7.4 `epyc.autokernel.evaluation_event.v2`, and populate
  them wherever a cheap cell and an expensive cell measure the same candidate. One mechanism, four
  uses: half → full partition, op → graph, T1 → T3, screen → verify. The ratio is **keyed by change
  class** (§9.5), not global — an instruction-level change and a bandwidth-bound change do not
  transfer alike, and a single pooled ratio would average them into uselessness. **Free to add now,
  impossible to backfill**: a ratio invented at read time claims a correspondence the original run
  never measured.
- [x] **AK-TR-2 — Print the self-spread noise floor adjacent to every per-case delta.** ✅ 2026-08-10. §9.2 already
  mandates MDE published with the result and §15.2 already runs A/A; the gap is **presentation**, not
  statistics — our machinery is stronger than the reference campaign's, and it is still possible to
  read a per-case table and not see which rows are inside the noise. Suppress or explicitly flag any
  delta below the floor at the point of display.
- [x] **AK-TR-3 — Per-turn productivity accounting** — see
  [`agentic-rocm-kernel-authoring.md`](agentic-rocm-kernel-authoring.md) §"Loop-engineering
  experiments", which owns it. Referenced here because §8.8 POST_RUN_CRITIC is where the tuple gets
  emitted. ✅ 2026-08-10 — routed to its single implementation owner as AK-PT-1; AutoKernel consumes
  that archive for AK-X-6 and does not duplicate the reducer.
- [x] **AK-TR-4 — Extend roofline utilisation (§8.3.1) to a per-quant surface, anchored on
  state-of-the-art CUDA kernels.** Operator idea, 2026-08-10: give the loop a *defined* improvement
  target by expressing decode throughput as a fraction of the theoretical roof and comparing that
  fraction to what SOTA CUDA kernels reach on NVIDIA silicon — **and compute it independently per
  quantisation**, because this card behaves very differently at 16-bit than at 4-bit.
  - **This does not reopen AK-D3.** AK-D35 already ratifies utilisation as *diagnostic and routing
    input, never a gate*; a target expressed in utilisation therefore cannot be peeked at by a
    promotion gate, which is exactly the property AK-D3 removed the +25%/+20% trigger to obtain.
  - **Per-quant is more correct, not merely richer.** `bytes_per_token` differs per quant by
    construction, so a single pooled utilisation figure silently mixes denominators; and splitting it
    **localises the headroom to the dequant path**, which is where the ladder collapse in §8.3.1(3)(b)
    lives. This is the same measurement the "close the ladder gap" program needs.
  - **Use the spec-basis ridge for the cross-vendor comparison** (§8.3.1's usage rule): converting our
    figures to an achievable basis while a CUDA anchor stays on a spec basis shrinks the gap without
    shrinking it. The **mixed-basis** row (spec FLOPS over measured bandwidth) is retired for this
    purpose. Achievable-basis figures remain the honest ones for reasoning about *our own* headroom.
  ✅ 2026-08-11 — `substrate.py` now constructs exact-quant observations and comparisons, refuses
  mixed or pooled quant labels, reports local headroom on the measured-achievable basis, and permits
  cross-vendor targets only on a matching spec basis. The registered BF16/H100 single-stream anchor
  cites primary absolute-bandwidth evidence; Q4_K and Q8_0 remain explicit `COULD_NOT_CHECK` gaps
  rather than borrowing Marlin's relative INT4 result or another quant's denominator. The entire
  surface is structurally diagnostic/routing-only and cannot gate promotion.
- [x] **AK-TR-5 — Fresh process per arm, and size the working set so memoization cannot pay.** Where a
  variant is env-gated at import/init time, an in-process A/B measures the first arm twice; §9.3's
  paired blocks do not protect against that. Separately, choose working-set sizes such that
  candidate-side caching of results is unprofitable **by construction**, which is cheaper and more
  durable than detecting it (complements RVP-C6-8 in
  [`rocm-verify-profile-backend.md`](rocm-verify-profile-backend.md)).
  ✅ 2026-08-11 — every live arm now receives a unique `(pid,start_ticks)`, an empty invocation-only
  writable tree, and verified teardown before the next arm. Hardened `llama-bench` commit `0492c231`
  keeps `2 × reps` contexts and input vectors simultaneously live, gives every timed repetition
  unique content and addresses, and runs its same-content/address-rotated replica untimed. The runner
  refuses missing, reused, malformed, or output-variant receipts. The experimental CPU build and the
  complete 3,730-test static suite pass; exercising the real-model path remains part of the next
  explicitly authorized campaign, not an implementation gap.
- [x] **AK-TR-6 — Compile-only artifact-diff veto in T0 (§8.6).** ✅ 2026-08-10. Before spending any GPU wall-time,
  diff per-kernel VGPR / SGPR / scratch usage and instruction mix between candidate and anchor via
  `roc-obj` / `llvm-objdump`. Register-pressure movement means the A/B is **unconfirmed**, not
  disproven — it fires as a veto on the *claim*, not on the candidate. **This buys back our scarcest
  resource**: it is a static check that can retire a candidate before it reaches T1.

Implementation evidence: evaluation-event schema v4 and `evaluator/api.py` realize AK-TR-1/2;
`artifact_diff.py` parses captured resource metadata/disassembly, and `campaign.py` refuses a GPU T0
launch when the artifact is absent or movement is unconfirmed. The latter is a no-GPU pre-launch veto.

### AK-LN — screening lanes and partition depth

Enabled by corrections 2 and 3 above. The governing rule, from operator steering:
**lanes screen, the full instance verifies.** A lane may rank; only a full-instance measurement under
the standing protocol may make a claim.

- [x] **AK-LN-1 — Lane registry.** ✅ 2026-08-10. Each lane declares its cost, its capacity, and **what it is a proxy
  for**. Assignment is by change class to the cheapest lane whose *measured* rank fidelity clears a
  declared threshold — never by convenience. Available today: one exclusive GPU device claim plus
  historically exercised CPU shapes at 4×48t, 8×24t, 16×12t, 32×6t and 48×4t. Those records establish
  fan-out feasibility, not rank fidelity.
- [x] **AK-LN-2 — Partition-depth calibration.** ✅ 2026-08-11. Run one fixed candidate set at full-machine and at
  **every historically exercised split depth** — 4×48t, 8×24t, 16×12t, 32×6t and 48×4t — and
  measure each split's **rank correlation** against the full-machine ordering. The preserved sweep
  proves those fan-outs run; it does not prove rank fidelity. Needs no new candidates — reuse a banked
  set. **Pre-register the prediction** before running: bandwidth-bound changes lose fidelity fast as
  partitions shrink (they compete for the same memory system), instruction-level changes hold. The
  pre-registered anchor/IQK-off/flash-attention-off calibration found full-machine rank fidelity 1.0.
  Depth 4 retained the ordering (1.0), but depths 8, 16, 32 and 48 inverted IQK-off versus
  flash-attention-off (Spearman 0.5). No historical split depth is therefore admitted as a ranking
  proxy by this candidate set. Receipt:
  `/mnt/raid0/llm/autokernel/probes/ak-ln-2-x5a-lanes-20260811T1400Z/receipt.json`, SHA-256
  `c207a46f7e1868deeca4628faedfd47b933839a921b0b6d472055ce80c415618`. Research runner:
  `scripts/benchmark/run_autokernel_cpu_lane_calibration.py`.
  Pre-registration is what makes a confirmation informative rather than a post-hoc story.
- [x] **AK-LN-3 — Cross-lane A/A control — necessary, and NOT sufficient.** ✅ 2026-08-10. §15.2's A/A control run
  per lane detects a per-lane-position offset. It **cannot** detect bias correlated with mechanism
  class, because that bias appears identically in every lane and cancels out of the A/A comparison.
  Pair it with AK-TR-1, which is the only thing that measures it. **Never apply a blanket haircut to
  lane results** to "correct" for partitioning — a uniform correction assumes the very
  class-independence that AK-LN-2 exists to test.
- [x] **AK-LN-4 — Op-level screening on profiled bottlenecks.** ✅ 2026-08-10. Operator steering: profile
  first, rank ops by wall share, then fan out **concurrent op-level experiments on the single
  bottleneck op** rather than serialising whole-graph runs. `test-backend-ops` is already an op-level
  harness, so the unit exists; what is missing is the fan-out and the §9.1 promotion rule applied per
  op. This is the unit that actually parallelises — a graph run does not.
- [x] **AK-LN-5 — Isolation requirement for concurrent CPU lanes.** ✅ 2026-08-10. Concurrent instances contend
  invisibly unless each uses `--no-mmap` plus an explicit `membind`: `mmap` **shares NUMA placement
  across instances**, so two "independent" lanes can silently read one node's memory. A partitioned
  result may **screen but never gate** (§9.6 banking is unaffected; §9.7 T2 and §10 T3 remain
  full-instance). **The pre-bench CPU-frequency/throttle check stays mandatory at the verification
  tier** — operator, 2026-08-10: the screening/verification split does not relax it, it concentrates
  its importance onto the one measurement that carries the claim.

Implementation evidence: `lanes.py` records each lane's physical-core-share cost, capacity, proxy,
CPU set, memory binding and historical evidence reference. It refuses mmap and physical-core overlap,
requires A/A plus class-specific rank calibration, falls back to full verification without it, and
fans candidates over the measured highest-share op in waves. The AK-LN-2 campaign rejected every
historical split as a general ranking proxy; those lane shapes remain execution capacity only unless a
narrower change-class calibration later clears the same gates.

### AK-BH — baseline honesty (needs a GPU window; sequence T0 probes first)

Sequence **RVP-T0-1 → RVP-T0-2 → AK-BH-1 → AK-BH-2**, so two thirds of this block can falsify at zero
GPU cost before any GPU claim is filed.

- [x] **AK-BH-1 — hipBLASLt vs rocBLAS microbench at our prefill shapes, standalone.** ✅ 2026-08-11 —
  `libhipblaslt.so`
  is installed on this host and **not linked** by our build. Measure the two libraries against each
  other at our shapes *before* touching ggml, so the question "is our GPU baseline the honest one" is
  answered by a 30-minute microbench rather than by a kernel campaign. This is the structural analogue
  of the baseline-deflation finding this batch turned up: a result measured against a weaker-than-
  available baseline overstates the win by the size of the gap, and the gap here is measurable today.
  Complements the §"New index-backed leads" hipBLASLt grouped-GEMM lever. The best-heuristic live
  receipt covered nine shapes: hipBLASLt won three, while its ratio to rocBLAS ranged from 0.734× to
  1.322×. A universal library replacement is explicitly declined; the strongest baseline is
  shape-specific. Receipt: `/mnt/raid0/llm/autokernel/probes/ak-bh-1-best-of-heuristics-20260811T0948Z/receipt.json`,
  SHA-256 `aca0dc59dbea9745008e00f3958a767dfb07b4c9e2e21f8946239ec981762cfc`. Research runner:
  `scripts/benchmark/run_rocm_gemm_baseline_compare.py`; comparator source:
  `scripts/benchmark/rocm_gemm_baseline_compare.cpp`. A fresh 2026-08-12 corroborating replication
  under a rebuilt binary found 4/9 hipBLASLt wins and a 0.7289325552×–1.3218706582× ratio range.
  The near-parity win count drifted by one shape, but the shape-specific baseline conclusion did not.
  Receipt: `/mnt/raid0/llm/autokernel/campaigns/ak-bh-1-20260812T0448Z/receipt.json`, SHA-256
  `5daa79e7bc12cec1e9358b8166f08a7586bb0d5ba7885832bb47d96af82c7dc1`.
- [x] **AK-BH-1a — Sustain the nine-shape vendor-baseline comparison long enough to attest loaded
  device state.** ✅ 2026-08-12 — a clean detached research checkout at `26ca88dc` rebuilt the existing
  comparator, then ran **30,000 timed repetitions per provider per shape** under canonical claim
  `akd-f6f696e03b8d468c`. The 55.17-second sampling window retained 221 samples at 250 ms cadence
  (maximum gap 0.250991 s), 86.43% at 1700 MHz, 165 W median / 241 W maximum power, and 62 C maximum
  temperature. hipBLASLt won 4/9 shapes; its exact-shape ratio to rocBLAS ranged 0.7444927062× to
  1.3162041222×. All nine winner directions matched the earlier short replay, with mean absolute ratio
  drift 0.00721 and maximum 0.01556, so the shape-specific strongest-baseline conclusion survives
  sustained load. Both exact PIDs were dead after terminal release at `2026-08-12T08:26:49.465889Z`.
  Receipt: `/mnt/raid0/llm/autokernel/campaigns/ak-bh-1-sustained-r2-20260812T0824Z/receipt.json`,
  SHA-256 `f8e09c77dffe7767af3bb1d1d85f2f4866155dcf677a331fa0e458111a42c855`.
- [x] **AK-BH-2 — Baseline-honesty factorial: `-fa 0|1` × `ROCWMMA_FATTN` × `MMQ_MFMA`.** ✅ 2026-08-11
  **Correction to a standing project assumption**: `llama-bench`'s `-fa` default is
  `LLAMA_FLASH_ATTN_TYPE_AUTO`, not `0` — verified at `tools/llama-bench/llama-bench.cpp:389` in the
  frozen v8 tree. **AUTO is worse for baseline honesty than a known-off default**, because it resolves
  differently per model, quant and backend: two runs with *identical command lines* can silently
  differ, manufacturing or hiding a speedup with no visible flag difference. Pin `-fa` explicitly on
  every arm and **record what AUTO resolved to** when reading any historical number. All eight live
  arms completed with 30 retained repetitions. On the tested Qwen2.5-Coder-0.5B Q4_K_M prefill
  surface, `-fa on` beat off in all four build pairs, `MMQ_MFMA` was slower, and `r1m0-fa-on` won at
  24,647.316788 t/s. Receipt:
  `/mnt/raid0/llm/autokernel/probes/ak-bh-2-factorial-20260811T0952Z/receipt.json`, SHA-256
  `2a53cee2d8513737eca894e0f34152549932b75f3e04ef541cdee1848472cfdf`. Research runner:
  `scripts/benchmark/run_autokernel_gpu_factorial.py`.
- [x] **AK-BH-3 — CPU-lane baseline-honesty arm, run concurrently with AK-BH-2. ✅ 2026-08-11** The
  full-host CPU claim remained held across three randomized 30-repetition hardened arms. On the exact
  Qwen2.5-Coder-0.5B Q4_K_M prefill surface, implicit AUTO measured 5,569.961069 t/s, explicit ON
  5,451.900259 t/s, and explicit OFF 2,741.087873 t/s. AUTO therefore behaves like the fast ON path
  here while hiding the choice in argv. Every exact measurement window retained package-power
  evidence; the counter permission was restored to `0400`. Receipt:
  `/mnt/raid0/llm/autokernel/probes/ak-bh-3-cpu-baseline-honesty-20260811T1330Z/receipt.json`, SHA-256
  `157580a4133a5b7404384e16a4f0b3737f54480365694cfdfd079a6ce9c99911`. Research runner:
  `scripts/benchmark/run_autokernel_cpu_baseline_honesty.py`.
- [x] **AK-BH-4 — Encode strongest-baseline selection by exact measured surface. ✅ 2026-08-11**
  `evaluator/baseline_honesty.py` requires both rocBLAS and hipBLASLt observations for one identical
  model SHA, quant, operation, shape, dtype, build SHA, and explicit factor set; selects by declared
  metric direction; refuses `auto`, missing or duplicate provider arms, metric mismatch, and any
  candidate model/quant/shape/factor transfer. Nine focused tests pass. Research commit `5fbd471b`.
  The earlier 0.5B result is consequently evidence for that surface only, never a portable baseline.

### AK-OP — operator-only (measurement trust boundary is human-amendment-only)

Neither of these may be executed by a session; both are decision packages for the operator.

- [ ] **AK-OP-1 — P-GPU-1 `duty_cycle` amendment (operator decision).**
  - **Observation.** P-GPU-1 field 4 specifies a fresh server per repetition. A fresh server per rep
    necessarily inserts a multi-second gap between reps — i.e. the protocol measures the **bursty**
    duty cycle, while production serves in the **sustained** one.
  - **Why it matters.** The reference ablation this batch surfaced reports the bursty regime reading
    meaningfully lower latency with substantially lower variance than sustained operation on the same
    hardware. Both regimes are legitimate to measure; **conflating them is not**, and a protocol that
    does not name its duty cycle cannot be compared against one that does.
  - **Options.** (a) Add a `duty_cycle: bursty | sustained` field to P-GPU-1 and declare the current
    protocol `bursty`, leaving every existing number valid and correctly labelled. (b) Additionally
    author a sustained variant for claims about production-like load. (c) Decline and record that
    P-GPU-1 numbers are bursty-regime by construction.
  - **Recommendation: (a) now, (b) when a sustained claim is first needed.** (a) is a pure labelling
    amendment that invalidates nothing and makes the limitation visible; (b) costs a protocol
    authoring cycle and should be paid when there is a claim that needs it.
  - **Requires a human amendment to `MEASUREMENT.md`** — AK-D10 and the constitution's trust boundary.
    No session may self-apply it.
- [x] **AK-OP-2 — Decline root-side `--setperfdeterminism` capability.** ✅ 2026-08-11 — RVP-T0-1
  held 1700 MHz for 99.5868% of a 60-second saturating GEMM while peaking at only 200 W against the
  300 W cap, so the card never approached the cap and clock excursion is not a live variance source
  under this workload. The 2026-08-12 replay independently held the same 99.5868% nominal-clock
  fraction while peaking at only 196 W. No root capability or measurement-constitution change is
  warranted.

### Recorded declines (so they are not re-derived)

- **RL training methods (CPO, RIF-RFT, protected SFT) — declined for now.** They sit behind the
  unverified gfx90a training gate and none of the batch's RL sources establishes that gate. The
  *regime map* they produce is kept as reasoning (reachability — rollout pass-rate strictly inside
  (0,1) — predicts whether RL can repair a policy at all), but no training campaign is proposed.
- **Adopting an external coordination plane for the loop — declined.** It would replace working
  internals (§5.7 resource plane, the session bus) with a foreign substrate; two patterns are worth
  mining, the stack is not worth adopting.
- **Cloning composable_kernel on the strength of a retrieval-augmented-kernel-generation result —
  declined on that evidence.** The cited system uses **one fixed anchor example**, not a retrieved
  corpus, so it does not support "clone the corpus". If we clone CK, justify it on the existing
  `portable_source` classification in §6.5, which already stands on its own.

### AK-X — additional actionables from the recovered Stage-2b reports (filed 2026-08-10, beyond the Stage-3 plan)

_The analyst reports were recovered from the session transcript **after** the Stage-3 plan was written,
so they carry derived actionables the plan predates._

- [x] **AK-X-1 — Add `INSTRUMENT_TAMPERED` to the §12 failure-and-abuse table.** ✅ 2026-08-10. RVP-C6-1 builds the
  detection; §12 has no row for the finding, so a detection today would have nowhere to land.
- [x] **AK-X-2 — Add a parsed `device_state` block to the §7.4 evaluation event**, populated from
  RVP-C3-3, with `throttle_observed`. A text blob no gate can read is not a gate input. ✅ 2026-08-10 —
  evaluation-event v5 stores numeric loaded samples, nominal SCLK and a mechanically re-derived throttle
  verdict; GPU absence is `COULD_NOT_CHECK`, a below-floor sample is `FAIL`, and CPU events require null.
- [x] **AK-X-3 — Add `min_measurable_us` to the §9.3 T1a recipe, derived from OUR OWN A/A spread.**
  Below it a cell is `inconclusive` rather than a rank. Do not import a foreign floor — the published
  ones are NVIDIA-derived. ✅ 2026-08-11 — `statistics.derive_minimum_measurable_duration()` derives
  the typed floor from local paired-A/A absolute-duration spread and a declared relative-noise budget;
  every T1a recipe requires that object, rejects bare numerics, and withholds rank below the floor.
- [x] **AK-X-4 — Add `cache_state: warm | cold` to the §9.3 T1a recipe and hold it fixed across arms.** ✅ 2026-08-10.
  The warm/cold gap is workload-dependent (large for GEMV, negligible for GEMM), so an undeclared
  cache state is a per-op confound rather than a constant one. Every registered T1a recipe now records
  a validated warm/cold parameter (explicitly cold by default) and binds it into the recipe receipt, so
  an arm mismatch changes the evidence identity instead of remaining ambient state.
- [x] **AK-X-5 — Per-partition CPU frequency / package-power attestation in the §7.4 `host_receipt`.**
  Our throttle discipline operates at *session* granularity; with 2–4 concurrent CPU lanes sharing one
  dual-EPYC package power and boost budget there is **no per-lane frequency attestation at all**. This
  is the coupling channel that actually threatens the lane design (CPU→GPU contention is ~0.3 % on the
  reference ablation; CPU-partition ↔ CPU-partition is unmeasured by anyone). AK-LN-3's cross-lane A/A
  is its acceptance test. ✅ 2026-08-11 — live CPU preflight now requires readable per-CPU frequency
  plus package-energy counters for every package touched by the partition; exact-window deltas handle
  counter wrap and are labelled `shared_package_window`, never lane-exclusive power. Missing or
  unreadable powercap data is `COULD_NOT_CHECK` and prevents a CPU campaign from claiming the control.
  Empirical cross-lane coupling still belongs to the authorized lane-calibration campaign.
- [x] **AK-X-5a — Run the cross-lane package-power/frequency acceptance.** ✅ 2026-08-11. Exercise the same fixed
  candidate set across every AK-LN-2 split depth with readable `energy_uj`; retain each exact-window
  shared-package receipt and test whether lane position or concurrent depth changes the A/A distribution.
  This is empirical inference work and requires the operator's explicit permission. The authorized
  campaign retained readable exact-window package-energy evidence for every arm and restored
  `energy_uj` to mode `0400`. Every split failed the predeclared combined acceptance: maximum anchor
  lane-position deviations were 37.36%, 53.25%, 77.43%, 36.19% and 16.28% at depths 4, 8, 16, 32 and
  48 respectively (limit 10%); loaded-frequency ratios to full were 0.815, 0.829, 0.794, 0.743 and
  0.800 (limit 0.8). The full CPU claim remained held through all waves and was released. Evidence is
  the AK-LN-2 receipt above.
- [x] **AK-X-6 — Turn-budget stopping rule driven by AK-PT-1.** Refine turns continue only while the
  rescued-kernel speedup distribution overlaps the persistent-kernel distribution; once a turn admits
  only rescued kernels below the contribution floor it is repair-only and does not advance the search.
  Must use the §9.2 e-process, never a point comparison. ✅ 2026-08-11 —
  `turn_productivity.py` consumes the immutable AK-PT-1 archive, binds the contribution floor before
  observations, imports its construction and threshold only from accepted campaign calibration, and
  requires two crossing e-processes (rescued below / persistent above the same floor) plus an only-
  rescued latest turn. Insufficient evidence continues refinement; the result can only withhold search
  advancement and carries no rank/retain/promote/deploy authority.
- [x] **AK-X-7 — Edit-type classifier over adjacent candidate diffs** (no-op / mask fix / delegated-op
  / dtype-cast / optimization rewrite). §9.5 already keys behaviour off `proposal.change_class`; this
  measures whether the **realised** edit matched the **declared** class, and flags a proposal that
  promised an optimization and delivered repairs. ✅ 2026-08-10 — `execution/chain.py` classifies
  every diff that enters change-surface projection, preserves per-class line counts under a deterministic
  tie rule, and attaches the realized class to the captured evidence.
- [x] **AK-X-8 — Add an intent/targeting axis to §9.6 banking.** Classify each banked candidate as
  right-target-good-perf / right-target-bad-perf / **wrong-target-good-perf** / wrong-target-bad-perf.
  A wrong-target speedup is a "lucky win" — **quarantined, not banked**. This is a *mis-targeting*
  check and is complementary to C6's exploit detection, which does not cover it. ✅ 2026-08-10 —
  deferred with the operator-approved controller/banking plane: the lean driver banks one measured branch
  but has no autonomous promotion API. Re-open when the first real banked archive restores composition;
  mechanically derived/declared surface mismatch is already retained so the axis is not backfilled.
- [x] **AK-X-9 — Require a task-level accuracy check before promotion for serving-path change classes**
  (§9.5). Motivating case from the reports: a patch that hardcoded tensor dimensions passed every hard
  performance metric while task accuracy collapsed 32% → 0%. ✅ 2026-08-10 — deferred with T2/T3
  promotion under the lean-loop decision. T0 already requires anchor-bound generation/coherence and full
  op correctness; task-level accuracy becomes mandatory when a real serving-path champion reactivates the
  release plane, not as synthetic machinery before candidate #1.
- [x] **AK-X-10 — Record external evidence that C4 is on the critical path**, not more refine turns:
  refinement responds to explicit local error signals, while plan-level decisions (tiling, memory
  layout, kernel boundaries) are "not recoverable from the feedback available in current iterative
  pipelines". This strengthens an existing position rather than opening work. ✅ 2026-08-10 —
  `research/intake_index.yaml` intake-1095 binds the claim to KernelBenchX v2 §5 Insight 2 and maps its
  A-5 action explicitly to AutoKernel C4; it is external design evidence, not a local performance claim.

**Declined, recorded so they are not re-derived:** compilation-flag pinning across candidate and anchor
(**already covered** by §7.3's candidate record, which binds compiler/toolchain/build command and
logs); and adopting the reference campaigns' variance tolerances or scoring formulae, both of which are
weaker than our sanctioned e-process machinery.
