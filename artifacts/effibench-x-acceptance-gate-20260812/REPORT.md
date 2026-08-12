# EffiBench-X canonical-solutions acceptance gate — 2026-08-12

**Task**: `handoffs/active/architect-model-selection-bench.md:597` — run the benchmark's own
canonical solutions through the (patched) harness BEFORE any model arm. If canonical solutions
do not score as canonical, the instrument is not measuring what its name says.

**Scope note**: adoption itself (row :596) is a **separate decision**, additionally gated on
ingesting arXiv 2607.01211 (row :601). This gate informs that decision; it does not decide it.

## Verdict

**FIT to score model arms, Python-only, on the DATED subset, with named exclusions and one
mandatory extra patch (04).** Grounds:

- 61,600 test executions (308 problems x 100 tests x 2 canonical arms) — **zero fail-open
  occurrences** (no `status=="done"` record with runtime `None` or `0.0` anywhere).
- Canonical solutions score as canonical on 293/308 (`canonical_runtime`) and 285/308
  (`canonical_memory`); **306/308 problems have at least one passing canonical arm** (the
  harness's own `good_problems` criterion); 272/308 pass both arms.
- Every failure is a **correctness-side dataset/harness curation defect** (harvested human
  solutions with leftover debug prints, runner-solution interface mismatches, import-header
  gaps) — not a measurement defect. Full classification below.
- Efficiency leg: minimum measured runtime across all passing records is **17.85 ms**
  (a sane Python floor — no zero/near-zero fabrications), and cross-arm scoring with the
  harness's own `compute_model_stats` (each canonical arm scored as if it were a model arm
  against the other) yields **zero degenerate scores** over 272 problems
  (`canonical_memory` vs `canonical_runtime`: min 0.42 / median 1.00;
  reverse: min 0.21 / median 1.00).
- The fail-open path is **real, not theoretical**: it fired twice in this run
  (`leetcode_3451`, `leetcode_3469`, `canonical_memory` arm — statistics binary returned
  EMPTY output on a successful execution, the exact upstream-issue-#4 trigger, ~1 in 30,000
  executions). Post-patch both surfaced as loud `MeasurementError` records and the problems
  failed closed. Pre-patch they would have been silent `runtime=0.0` "passes" — and a 0.0
  canonical runtime makes **every model arm score 0.0 on that problem** via
  `min(1, max(0, canonical/model))` with no error anywhere.

**Exclusions to carry into any model-arm run**: `leetcode_3264` (runner emits no output) and
`leetcode_3451` (canonical has debug prints; also the transient measurement failure) failed
in both arms; the other 34 failing problem-arms are rescued by the other arm under the
any-canonical rule. The third canonical arm (`canonical_normal`) was not run (see Bounds).

## Source and subset

- Upstream `github.com/EffiBench/EffiBench-X` @ `21b6668a49a342c843e679cc5ed16081064a5768`
  (Apache-2.0), cloned to `/workspace/tmp/effibench-x-upstream/`. The `effibench-enhanced`
  re-upload (intake-986) was NOT used, per dispatch (zero lineage).
- Dataset: HuggingFace `EffiBench/effibench-x`, split `test`, 623 rows. DATED subset =
  rows with non-null `release_timestamp`: **exactly 308, all `leetcode` functional**,
  matching intake-952 dive item (6).
- Canonical arms: `canonical_runtime` and `canonical_memory` (the two arms that define the
  metric denominators in `compute_model_metrics`), extracted with the harness's own
  `generate_solution.py merge-canonical-solutions`; 308/308 Python solutions present in each.

## The three known blockers — located, patched, mutation-tested

Patches in `patches/`, generated with `git diff`, verified to apply cleanly in order
(01→04) onto pristine upstream `21b6668`.

### B1 — deprecated `openjdk:21` image (`01-deprecated-openjdk-image.patch`)

- **Verified, not assumed**: `docker manifest inspect openjdk:21-jdk-bookworm` fails (gone
  from Docker Hub); `eclipse-temurin:21-jdk` exists.
- **NOT moot under Python-only**: `BaseExecutionManager.setup()`
  (`effibench/backends/backend_utils.py:297-300`) iterates ALL six registry languages
  unconditionally, so a default backend startup dies on the Java image even for a
  Python-only evaluation.
- Patched both sites: `effibench/utils.py:320` (EFFIBENCH_REGISTRY) and
  `third_party/llm-sandbox/llm_sandbox/const.py:25` (DefaultImage.JAVA).
- This gate run never pulled the Java image: the driver
  (`scripts/run_backend_python_only.py`) prunes the registry to `python3` in-process — an
  operational wrapper, not an upstream patch.

### B2 — `time.sleep` AttributeError (`02-time-sleep-attributeerror.patch`)

`generate_solution.py:5` `from time import time` shadows the module; line 123
`time.sleep(submit_gap)` raises AttributeError on the first submitted task. `time()` is never
called in the file (verified by grep), so the minimal patch is `import time`.

### B3 — fail-open-to-0.0 metric (`03-fail-open-to-zero-metric.patch`) — the critical one

Three coordinated sites; failure now surfaces as an ERROR that propagates loudly, never 0.0:

1. `third_party/llm-sandbox/llm_sandbox/docker.py` (`run()` statistics parse): upstream
   silently set `runtime = memory = integral = 0.0` on ANY parse failure. Now raises
   `RuntimeError` with the raw statistics output embedded; propagates through the manager's
   existing error path as `status="error"`.
2. `effibench/backends/backend_utils.py` (`execute_in_sandbox`): upstream coerced missing
   metrics on "done" executions to 0.0 and stamped 0.0 on all failed executions. Now a
   "done" execution with missing metrics becomes `status="error"` with a `MeasurementError:`
   text prefix; failed/unmeasured executions carry `None`, never 0.0. **Load-bearing, not
   defensive**: the bundled "local" backend's sessions
   (`effibench/backends/local_sandbox.py::run`) never set metric attributes at all, so under
   that backend upstream fabricated 0.0 for EVERY successful execution — upstream issue #4
   exactly.
3. `evaluate_solution.py::compute_model_stats`: the outlier filter
   `r["runtime"] < 10_000_000_000` would crash on honest `None` carriers; now drops only
   records with a real runtime >= 10 s, keeping failures counted as failures.

### B4 — NEW, found at runtime: `is_passed` empty-output fail-open (`04-is-passed-empty-output-fail-open.patch`)

`evaluate_solution.py::is_passed_item` counted any `status=="done"`, empty-text, exit-0
record as a PASS even when the evaluator judged it failed. Observed live on
`leetcode_3264`: evaluator 0/100 passed, harness `is_passed()` returned True — converting
"produced no output" into a pass, with measured-but-meaningless runtimes entering the
canonical denominators. For MODEL arms this is a correctness-inflation channel (a solution
printing nothing and exiting 0 auto-passes). Clause removed; the evaluator verdict and the
tty-echo rescue clause remain. **This patch is mandatory before any model arm.**

### Mutation test — visible and counted

`test_fail_open_patch.py`, 8 pytest tests (B1: 1, B2: 1, B3: 4 defect + 1
preserved-behavior, B4: 1). Runs recorded:

- **Unpatched tree**: `6 failed, 1 passed` (`prepatch_pytest_output.txt`) — every
  defect-detecting test then present fails, proving the tests see the defects. (B4's test
  was added after its live discovery; its defect is separately proven pre-patch by the
  recorded `leetcode_3264` demonstration: evaluator 0/100 vs `is_passed()` True.)
- **Patched tree**: `8 passed` (`postpatch_pytest_output.txt`).

Inputs that previously fail-opened (unparseable statistics line; metric-less "done" result;
empty-output exit-0 record) are constructed explicitly and shown to fail closed.

## Additional upstream defects found on the way

1. **`requirements.txt` omits `sortedcontainers`** — a hard client-side dependency:
   `materialize_function_from_code` (`effibench/utils.py:717`) prepends the python3 imports
   header (which does `from sortedcontainers import ...`) and `exec`s every problem's
   evaluator in the client process. With upstream requirements every evaluation dies with
   `ValueError: Error executing code: No module named 'sortedcontainers'` before any sandbox
   execution. (Installed into the gate venv; not patched — packaging, not code.)
2. **`hf_dataset.py download` broken under datasets 5.x** — passes the removed
   `use_auth_token=` kwarg. The gate used `scripts/prepare_dataset.py` instead.

## Run environment and bounds

- Host: EPYC 9655 (96C/192T), Docker 29.7.1. Sandbox: 10 workers, one
  `python:3.11.11-bookworm` container each, cpuset-pinned to physical cores 0-9 (+SMT),
  per-test `timeout 10` inside the container, memory limit 1024 MB, no GPU involvement.
  Server + clients ran under `nice -n 10`; server main process bound to core 11 (not
  upstream's last-core default, whose SMT sibling 191 is a GPU host thread on this host).
- **All 308/308 problems ran in both arms — no prefix, no silent caps.** Wall:
  `canonical_runtime` 12:40:49Z → 12:46:01Z; `canonical_memory` 12:46:07Z → 12:50:46Z
  (~10 min total; 33 problems in arm 2 served from the harness's own dedup cache).
- Ambient load during the run: 1-min loadavg 12-42 on 96 cores (shared box). The gate's
  correctness leg is load-insensitive; the efficiency-consistency leg is cross-arm relative
  under identical ambient conditions. **These numbers are NOT comparative authority and NOT
  ladder-grade timing** (single-execution timing is upstream's design — intake-952 dive
  item 4 — and no region claim was held).
- Docker hygiene: upstream's `commit_container=True` mutates the local
  `python:3.11.11-bookworm` tag (bakes in build tools + pip packages). After the run the
  tag was restored by re-pull (`d4372bb352f5`) and the committed image deleted. All sandbox
  containers were removed by the server's shutdown handler (verified against a pre-run
  `docker ps` snapshot).
- Third canonical arm (`canonical_normal`) not run — bounded choice; the two arms run are
  the two that define the metric denominators. Adding it could only enlarge the usable set
  (e.g., possibly rescuing `3264`/`3451`).

## Failure classification (38 problem-arm failures / 616 problem-arm evaluations)

| Class | Count | Problem-arms | Nature |
|---|---|---|---|
| Canonical solution vs runner interface mismatch (JSON/type errors parsing runner input) | 13 | rt: 3225, 3519, 3528, 3542; mem: 3231, 3308, 3355, 3388, 3405, 3466, 3475, 3485, 3495 | harness/dataset curation |
| Debug/demo `print()` left in harvested canonical solution | 10 | rt: 3353, 3451; mem: 3298, 3360, 3380, 3383, 3395, 3402, 3525, 3567 | genuinely defective canonical |
| Import-header / scaffold gaps (`TreeNode`, `np`, `repeat`, `pairwise`, `insort`, scipy, `import copy` shadowing, runner-name mismatch) | 8 | rt: 3509, 3622, 3630, 3633, 3649; mem: 3435, 3516, 3527 | harness/dataset curation |
| Runner emits no output (exit 0, empty stdout) | 4 | rt+mem: 3264; rt: 3436, 3581 | harness defect (also exposed B4) |
| Environment parity: RecursionError (LeetCode raises the recursion limit; harness does not) | 1 | rt: 3239 (80/100 passed) | environment |
| Transient measurement failure — **failed closed** (statistics binary returned empty output on a successful execution) | 2 | mem: 3451 (99/100), 3469 (99/100) | fail-open trigger, correctly refused |

Zero timeouts, zero OOM, zero fail-open scores.

## Files

- `patches/01..04-*.patch` — apply in order onto upstream `21b6668`.
- `test_fail_open_patch.py` + `prepatch_pytest_output.txt` / `postpatch_pytest_output.txt` —
  mutation tests and both recorded runs.
- `results.jsonl` — one row per problem-arm (616 rows): pass counts, statuses, reason,
  runtime sums; written incrementally during the run.
- `gate_analysis.txt` — full analysis output (counts, fail-open scan, cross-arm scoring).
- `gatestats_canonical_{runtime,memory}.json.gz` — the harness's own per-problem stats.
- `scripts/` — the exact drivers used (dataset prep, Python-only backend, eval arm, collector, analysis).
- Raw per-test records (363 MB) remain at `/workspace/tmp/effibench-gate/data/evaluation/`
  (too large for the repo; regenerable via `scripts/`).
