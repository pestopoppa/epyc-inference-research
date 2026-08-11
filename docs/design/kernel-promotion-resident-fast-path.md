# Resident fast path for kernel-promotion qualification

**Status:** prospective implementation design; no existing promotion evidence is regraded
**Written:** 2026-08-10
**First intended use:** the promotion cycle after the v9 qualification, unless the complete
instrument and required human measurement amendments are ratified before an earlier campaign starts

## 1. Objective and invariants

Reduce end-to-end kernel-promotion wall time by keeping model servers resident across independent
requests and by running non-conflicting work in parallel. The fast path must preserve every property
that makes the current procedure trustworthy:

- exact frozen-production and sealed-candidate Git, binary, model, recipe, and shared-library identity;
- candidate-local CPU/HIP `ggml` linkage proof;
- deterministic request pairing and exact output parity;
- live all-TID affinity, NUMA placement, region-lock, contention-matrix, and GPU-device ownership;
- explicit warmup/discard policy, complete raw rows, median/MAD, and paired comparison;
- captured-PID-only teardown, process-death/VRAM-settle proof, sealed immutable publication, and a
  tested rollback to the current fresh-server instrument;
- no pre-promotion candidate GPU observation being represented as decision-grade P-GPU evidence.

This is not authority to change the in-flight v9 qualification. Its artifacts retain the grade of
the instruments that created them. A fast-path run becomes promotion evidence only after the exact
runner, schema, packs, thresholds, and protocol amendments named below are prospectively pinned.

## 2. Measured current cost

### 2.1 CPU production-role matrix

The completed v8/v9 K35 ABBA sequence under
`data/kernel-v9-candidate/promotion-plan-20260810/{a1-v8,b1-v9,b2-v9,a2-v8}-run/summary.json`
contains 80 successful cells. The four blocks consumed approximately 1,560 seconds wall time, while
the request intervals sum to approximately 719 seconds. About **841 seconds (54%)** were therefore
outside inference.

Median non-request time per repetition was:

| Scenario | Median overhead |
|---|---:|
| `v9_frontdoor_cpu_native_mtp` | 10.27 s |
| `v9_worker_general_cpu_native_mtp` | 6.68 s |
| `v9_architect_critic_cpu_native_mtp` | 14.79 s |
| `v9_ingest_long_context_cpu_no_spec` | 10.44 s |

First repetitions paid 15.8–32.8 seconds outside the request. This is structural, not an incidental
slow run: [`DEFAULT_WARMUP_DISCARD_POLICY`](../../scripts/benchmark/k35_stack_context_matrix_runner.py)
declares a fresh server per repetition; `run_cell()` launches at line 1208 and always terminates at
line 1256; `execute_plan()` serially list-comprehends every cell at line 1382. Server command
construction itself should be retained from `build_server_argv()` at line 606.

### 2.2 Quality and DFlash

The current quality wrapper
[`run_v9_quality_gate.sh`](../../data/kernel-v9-candidate/promotion-plan-20260810/run_v9_quality_gate.sh)
already uses one resident server per role, but regenerates the full stratified `n=200` suite. The
v9 worker run recorded 422 seconds; the equivalent banked v8 architect run recorded 1,361 seconds.
The two-role candidate phase is therefore approximately 30 minutes even though v8 outputs already
exist.

Historical production DFlash evidence at
`data/gpu-mi210/laguna-iq2-dflash-pgpu1-v8-rerun1/run-20260725T184624Z/summary.json` took about seven
minutes for ten fresh-server replicates, of which approximately 157 seconds were prompt plus decode.
[`laguna_pgpu1_dflash_runner.py`](../../scripts/benchmark/laguna_pgpu1_dflash_runner.py) hard-codes
fresh-server policy at line 58, emits a new-port base/DFlash cell for every replicate in
`build_plan()` at line 1389, and launches/settles a server in `run_replicate()` at line 1428.

## 3. Execution architecture

The fast path has four phases. Only a genuine resource conflict serializes work.

1. **Build and static preparation.** Build clean CPU and HIP trees in independent build directories,
   pinned to disjoint host halves. In parallel: hash binaries/models, produce linkage reports, stage
   the rollback bundle, validate the candidate overlay, generate request packs, and run non-inference
   source/schema tests.
2. **Resident CPU cohorts.** For one model/recipe/context cohort at a time, load paired production and
   candidate servers sequentially and execute request-level ABBA while both remain resident.
3. **One isolated candidate hot stack.** Load the production-shaped candidate stack once, then run
   functional, topology, orchestration, and parity checks without repeated teardown.
4. **Exclusive performance windows.** Serialize full-CPU and MI210 decision windows where their
   physical CPU claims overlap. After cutover, perform production-named resident P-GPU/DFlash
   certification; candidate GPU rows remain bounded observations.

Large `mlock` loads remain sequential. The production launcher documents concurrent page-fault plus
lock crashes at
`epyc-orchestrator/scripts/server/stack_commands.py::cmd_start` lines 1726–1728. Residency removes
repeated loads; it does not make simultaneous loading safe.

## 4. Paired resident CPU ABBA runner

### 4.1 Lifecycle

Implement `scripts/benchmark/kernel_promotion_resident_runner.py`. Reuse, rather than duplicate:

- `k35_stack_context_matrix_runner.py::build_server_argv()` for the exact recipe;
- its request-body, response summarization, identity, telemetry, and parity helpers;
- `scripts/utils/verify_ggml_linkage.sh` for process-local library proof;
- root `scripts/coordination/region-lock` and the current contention-matrix freshness check;
- the existing safe terminate helper, limited to PIDs created by this runner.

For each `(scenario, nominal_context)` cohort:

1. Acquire the complete declared region set and write the lock receipt and topology hash.
2. Resolve and attest both arms before launching: Git head/branch/cleanliness, binary path/SHA/version,
   build manifest, backend list, `LD_LIBRARY_PATH`, linkage report, model/draft hashes, and full argv/env.
3. Prewarm unique GGUF inodes once under `numactl --interleave=all`. Launch production, wait healthy,
   prove affinity/linkage; then launch candidate and repeat. Do not overlap the load intervals.
4. Run a 64-token same-shape warmup on each server. Exclude it and record it explicitly.
5. Execute five quartets `A,B,B,A`, producing ten independent requests per arm. Alternate the quartet's
   first arm across cohorts to expose position bias. Use `cache_prompt:false`, the same immutable body,
   deterministic sampling, and a fresh artifact directory per request.
6. Before and after each request, sample the inactive peer's process CPU counters and RSS. It may remain
   resident but must not perform meaningful work. Capture active all-TID affinity during the request.
7. Compute exact paired token/content parity and paired throughput ratios. No failed, short, retried,
   replaced, or discarded measured request is allowed.
8. Terminate the two captured process groups once; prove both dead, ports closed, lock released, and
   host state restored.

The comparator key becomes
`(scenario, nominal_context, pair_index, schedule_position)`, replacing the current block-oriented
`(scenario, nominal_context, block, rep)` key. It retains exact response/token comparison and reports
median, MAD, per-pair ratios, arm-order bias, and failure classifications.

### 4.2 Evidence schema

Publish a sealed `epyc.kernel_promotion_resident_window.v1` object with these required fields. A
missing or malformed required field makes the window an observation.

```text
schema, created_at, completed_at, status, lifecycle_mode = "paired_resident"
instrument:
  repo, commit, tree_sha256, source_sha256, invocation, protocol, attestation_ref
host:
  boot_id, uptime_s, cpu, numa_nodes, mem_total_gib, os_reserve_gib
  topology_hash, contention_matrix_hash, interference_policy
resource_claim:
  region_lock_receipt, regions, allowed_cpus, mem_policy
  gpu_lock_receipt|null, visible_device|null
cohort:
  scenario, role, nominal_context, request_pack_sha256
  warmup_policy, discard_policy, schedule_sha256, reps_per_arm
arms.{production,candidate}:
  branch, commit, dirty, binary_path, binary_sha256, server_version
  backend_list, runtime_stage, ld_library_path, linkage_report_sha256
  model_path, model_size, model_sha256, quant
  draft_model_path|null, draft_model_size|null, draft_model_sha256|null
servers[]:
  arm, pid, pgid, port, argv, argv_sha256, env, env_sha256
  launch_started_at, healthy_at, health_response_sha256
  all_tid_affinity_samples, numa_policy, resident_memory_samples
warmups[]:
  arm, request_sha256, response_sha256, token_count, timing, excluded = true
requests[]:
  pair_index, schedule_position, arm, started_at, completed_at
  request_path, request_sha256, response_path, response_sha256
  prompt_sha256, input_tokens, output_tokens, output_token_ids
  content_sha256, finish_reason, server_timings, spec_counters
  effective_speculative_settings, active_affinity
  idle_peer_cpu_before, idle_peer_cpu_after, idle_peer_cpu_s, idle_peer_rss_delta
comparisons[]:
  pair_index, parity_partner, exact_token_parity, exact_content_parity
  production_tps, candidate_tps, candidate_over_production
summary:
  parity_pass, median_by_arm, mad_by_arm, median_paired_ratio, mad_paired_ratio
  arm_order_bias, idle_peer_pass, throughput_guard_pass, complete
telemetry:
  before_window, after_both_healthy, after_window, after_cleanup
cleanup:
  captured_pids, term_attempts, death_proof, ports_closed, vram_settled|null
publication:
  staged_manifest, file_sha256s, complete_marker, atomic_rename
```

Raw response files remain authoritative; summary values are reproducible projections. Every request
also emits the project `ClaimTuple` write-side carrier once a benchmark-results adapter exists; the
runner does not invent a second grading ladder.

### 4.3 Resident validity and fallback triggers

Before decision use, run an incumbent-vs-itself calibration with two copies of the exact production
binary. Resident mode is accepted only if all hold:

- no inactive peer exceeds `max(0.25 CPU-s, 1% of request wall time)` on any measured request;
- median A/B ratio is within `[0.99, 1.01]`, with no schedule-position effect above 1%;
- every affinity, identity, linkage, parity, completeness, and cleanup assertion passes;
- no OOM, swap growth, region conflict, server reset, or graph-shape carryover is observed.

Any violation automatically re-emits the identical plan with `lifecycle_mode=fresh_server`, using the
current one-server-per-cell runner. It does not relax a threshold or partially retain resident rows.
`--server-lifecycle fresh` remains a first-class operator-selectable mode indefinitely.

## 5. Isolated sealed-candidate hot stack

Implement `scripts/benchmark/kernel_promotion_hot_stack.py` as a promotion-only launcher. It must not
repoint `/mnt/raid0/llm/kernels/production/{cpu,gpu}` or mutate the production registry.

Inputs are:

- a sealed candidate CPU and HIP runtime-stage root;
- the exact production stack manifest and topology hash;
- an isolated port map, PID/state directory, and log root;
- a per-backend binary and library override whose linkage is verified before readiness;
- an explicit test selection and resource-claim plan.

Reuse `epyc-orchestrator/scripts/server/stack_prewarm.py::collect_targets()` and `prewarm_all()` to
deduplicate GGUFs by inode and warm them once. Reuse stack command generation and health/affinity
checks, but inject the sealed candidate runtime stage after command generation. The current
`--stack-profile` is validation-only (`stack_commands.py` lines 1375–1377), while
`epyc-orchestrator/src/registry/kernel_paths.py::backend_dir()` intentionally resolves only the
production runtime. Neither is currently a safe candidate launcher.

Once resident, the hot stack runs:

1. live binary/linkage/backend and architecture identity on every process;
2. architecture, loader-reshape, backend-op, recurrent-state, and request-schema smokes;
3. all-TID affinity and declared NUMA-policy checks;
4. topology allowed-pair concurrency tests only for disjoint physical claims;
5. forced-role orchestration requests through an isolated API, if that API can consume the isolated
   endpoint manifest; otherwise endpoint-level tests remain pre-cutover and the production API is
   certified after cutover;
6. the broad exact-parity pack below;
7. captured-PID teardown, port closure, lock release, and sealed result publication.

Add `--endpoint-manifest` or `--base-url-map` to `quarter_stack_smoke.py`; it must not infer canonical
production ports when testing an isolated candidate stack.

## 6. Broad exact-parity pack

The measurement policy says quality scores transfer across kernel eras once parity is proven and
requires deterministic replay before regeneration. Operationalize that rule with a versioned,
SHA-pinned pack rather than interpreting it ad hoc.

### 6.1 Coverage

The first pack contains at least 32 distinct requests. Every deployed kernel-sensitive serving shape
gets at least four items, with tags permitted to satisfy multiple dimensions. It must cover:

- frontdoor CPU MoE + native MTP;
- worker-general CPU MoE + native MTP;
- architect-critic CPU dense/MoE + native MTP;
- ingest long-context recurrent/SSM with speculation disabled;
- architect-general HIP and worker-vision HIP/mmproj;
- short and long prompt shapes, structured JSON, prose, code, reasoning, stop and length finishes;
- DSpark/DFlash baseline and enabled requests on the same `-np 1` server: at least four request pairs
  with effective `n_max=0` and four with the validated positive cap;
- loader-reshape and recurrent-state-sensitive prompts selected from the relevant regression fixtures.

Each item pins request JSON, prompt hash, model/recipe identity, seed, expected finish condition, banked
production response bytes, output token IDs where exposed, content hash, and semantic validator. Pack
construction is deterministic and stratified; there is no resampling after candidate output is seen.

### 6.2 Stopping rule

- **Pass:** every item completes, all structural/semantic validators pass, and candidate output token
  IDs and content are exactly equal to banked production. Transfer the unchanged full-suite quality
  scores to the candidate era; only speed is remeasured.
- **Fail:** any reproducible mismatch or validator failure blocks quality transfer and triggers the
  complete existing candidate quality suite for diagnosis. More parity items cannot rescue a mismatch.
- **Invalid window:** infrastructure, identity, affinity, or cleanup failure invalidates the affected
  role window. Repeat that entire role window; never replace one measured row.
- **Forced full regeneration:** model, prompt template, sampling, scorer, question pool, endpoint
  semantics, or generation-path changes; missing mandatory shape coverage; or a pack-schema change not
  covered by its prospective attestation.

The pack manifest, builder, capture runner, and comparator are pinned by commit/tree/source hash. Its
initial coverage and stopping rule require human ratification before replacing a full quality run.

Expected candidate parity time is 4–6 minutes, versus approximately 30 minutes for the current two-role
full regeneration. That estimate must be replaced by measured phase timing after the first dry campaign.

## 7. Resident per-request DFlash/DSpark certification

Refactor `laguna_pgpu1_dflash_runner.py::run_replicate()` into
`launch_resident()`, `run_request_cell()`, and `close_resident()`.

Launch one server with the complete target/drafter graph and its maximum validated draft cap. The
counterbalanced schedule sends baseline cells with request-local `speculative.n_max=0` and accelerated
cells with the enabled cap. A measured cell must prove:

- the echoed effective cap equals the requested bounded value;
- cap 0 reports `draft_n == 0` and enabled mode reports `draft_n > 0`;
- the request used `-np 1` until the multi-slot cache defect is independently closed;
- fixed per-prompt semantics, numerical sanity, token floor, identity, and live binding all pass;
- before/after-window clocks, power, temperature, utilization, VRAM, PID mapping, during-request
  residency, and final VRAM-settle/cleanup evidence are complete.

Use five ABBA request quartets for n=10 per arm when a ≤2% claim is possible. Warm once at both cap
shapes; discard those declared warmups and record graph recapture. Preserve P-DFLASH's pooled acceptance
and per-prompt `DFlash/base` ratios; never substitute an aggregate median.

Candidate GPU runs remain observation-only. The same pinned resident instrument must run again against
the production-named kernel after cutover for P-GPU-1 certification.

## 8. Resource and topology constraints

`epyc-orchestrator/scripts/server/stack_manifest.py::serving_shape_capacity_report()` currently reports:

- host required 578.0625 GiB = 334.0 GiB weights + 244.0625 GiB KV, versus 1069.4157 GiB budget;
- GPU required 57.715 GiB versus 62.0 GiB budget;
- GPU roles: architect-general 35.455 GiB and worker-vision 22.26 GiB.

One complete stack fits. Two complete stacks do not fit by declared arithmetic, especially on the
MI210. Paired v8/v9 CPU cohorts and one candidate full stack are valid; two full GPU stacks are not.

The GPU host lane uses logical CPUs 184–191, SMT siblings of physical 88–95 on q3
(`epyc-orchestrator/orchestration/stack_topology.yaml::architect_general`, lines 213–223). Full CPU
roles span q0–q3, and architect critic explicitly serializes all four regions (lines 195–205).
Consequences:

- full-CPU decision measurements and GPU inference do not overlap;
- disjoint half-A CPU functional work may overlap GPU work only when the current contention matrix
  explicitly permits the pair and both live affinity witnesses pass;
- builds, packaging, hashing, scoring/replay, and static tests should fill otherwise-idle resources;
- model load remains sequential, but all tests for that model run before teardown.

## 9. Protocol and ratification boundary

The implementation must land before the measurement amendment, but its output is observation-only until
the human trust boundary binds the exact sources and hashes.

1. **CPU resident promotion amendment.** P-BENCH-2 already describes a resident production-shaped stack;
   P-BENCH-4 already warms and sends five requests to one server. P-BENCH-4 is limited to one FG-4b-like
   shape, however. Add a narrow prospective kernel-promotion protocol covering paired resident servers,
   n=10 request-level ABBA, exact parity, idle-peer witness, linkage, topology/affinity, 0.98 incumbent
   guard, median/MAD, cleanup, and fresh-mode fallback.
2. **GPU/DFlash instrument amendment.** P-GPU-1 explicitly allows resident mode when declared, but
   P-DFLASH binds the current checked-in runner, warmup, schedule, and replicates. Ratify the refactored
   source/commit/hash and its cap-0/cap-enabled per-request schedule prospectively.
3. **Quality-transfer pack attestation.** Bind the pack manifest, builder, banked production outputs,
   coverage matrix, exact-parity comparator, and stopping rule before it substitutes for regeneration.
4. **No retroactivity.** Existing fresh-server v9 rows are not upgraded, downgraded, or mixed with
   resident rows. Candidate GPU rows never become P-GPU evidence by later promotion.

Relevant governing text lives in root `measurement/protocols/bench-cpu.md` (`P-BENCH-2`, lines 100–107;
`P-BENCH-4`, lines 116–178), `measurement/protocols/gpu-cross-device.md` (candidate provenance, lines
16–21; explicit resident mode, lines 28–47; P-DFLASH, lines 60–89), and
`agents/shared/MEASUREMENT_POLICY.md` (deterministic replay and cross-era quality transfer, lines 43–61).

## 10. Acceptance tests

### Unit/schema

- deterministic ABBA generation has ten unique request rows per arm and stable schedule SHA;
- grouping launches exactly two servers per CPU cohort and one server per DFlash cohort;
- every required schema field and referenced artifact hash is present; unknown schema versions fail;
- comparator detects missing/duplicate pairs, token/content mismatch, order bias, non-finite timing,
  short completion, retries, and mixed identity;
- request-local cap tests prove 0, positive, clamped, omitted-default, and reused-slot isolation;
- output finalizer refuses partial manifests and atomically publishes only after all hashes and `COMPLETE`.

### Process and resource safety

- injected health timeout, request timeout, server crash, SIGTERM refusal, port leak, and VRAM-settle
  failure each clean only captured PIDs and invalidate the complete window;
- wrong `libggml`, wrong binary SHA/version/head, dirty tree, wrong model hash, missing backend, or stale
  runtime stage fails before measurement;
- wrong TID affinity, stale contention matrix, missing region/GPU lock, overlapping disallowed claims,
  swap growth, or excessive idle-peer CPU triggers failure/fresh fallback;
- a deliberately unrelated server survives every cleanup test.

### Integration

- small-GGUF CPU fixture completes resident A/B, exact parity, sealing, and fresh fallback;
- production-vs-itself full-shape calibration meets the resident validity bounds in §4.3;
- isolated hot-stack smoke reaches every declared endpoint without touching production ports/symlinks;
- cap-0/cap-enabled DSpark fixture proves one PID, effective settings, zero/positive draft counters, and
  exact output parity;
- one deliberate parity mutation blocks quality transfer and selects the full-suite diagnostic path;
- dry cutover/rollback rehearsal restores the original runtime roots and proves the candidate processes
  gone.

The first full dry campaign is successful only if the fast and fresh instruments agree on the keep/reject
verdict and the resident path has complete identity, topology, parity, and cleanup evidence.

## 11. Wall-clock targets

| Gating component | Current evidence | Fast-path target | Basis |
|---|---:|---:|---|
| CPU K35 | ~26 min | 15–17 min | remove ~14 min repeated lifecycle cost; retain n=10/arm |
| Candidate quality | ~30 min | 4–6 min | exact parity pack + banked production quality |
| DFlash | ~7 min | 3–4 min | one resident graph, request-level counterbalancing |
| These components | ~63 min | 22–27 min | 57–65% reduction |

For a prebuilt candidate, target **35–50 minutes** end to end. With clean CPU/HIP builds running in
parallel, target **45–60 minutes** for a complete promotion cycle. If the parity pack is not ratified,
retain the full quality run and target **60–75 minutes** instead.

These are implementation targets, not measured performance claims. The first instrumented dry campaign
must record phase-level wall time and resource utilization, compare fast/fresh verdicts, and replace the
estimates with observed medians before making a scheduling SLA.
