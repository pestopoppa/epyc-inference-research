# AutoKernel execution layer — runbook for the first real campaign

**Status: this code has never been run against a kernel.** No candidate has been
built by it, no benchmark has been taken by it, no number in this package came
off a machine. Every test in it runs on recorded tool output. What follows is
how the session that owns compute takes it from a cold start to a first
candidate — and, just as importantly, the three things that will stop it and
should be dealt with first.

Read §0 and §6 before you touch anything. §6 is short and it is the part that
decides whether today is a campaign or a plumbing session.

---

## 0. What exists, in one table

| Module | What it does | Ever run for real? |
|---|---|---|
| `cpu_region_claim.py` | Acquires the CPU region claim (real `flock`s, per-region, with a journal) | flocks yes, in tests; never around a benchmark |
| `worktree.py` | Resolves the production tip, adds a campaign worktree, configures + builds, emits a build receipt | never built anything |
| `t0_provider.py` | Runs `test-backend-ops`, `verify_ggml_linkage.sh`, generations, sanitizers; returns `correctness.T0Evidence` | never launched a tool |
| `microbench.py` | Runs the T1 paired-block `llama-bench` design under the claim | never spawned a bench |
| `control_runner.py` | Scores the five controls through the same dispatcher a candidate uses | fixtures only |
| `chain.py` | The **seams** between the above and the evaluator that reads them | pure projection, no I/O beyond hashing |

The evaluator, journal, storage, controller, release plane and operator surface
were built and green before any of this existed. They are not the risk.

---

## 1. Before you start — the preflight

Run all of these. Any one of them failing means stop and fix, not proceed.

### 1.1 The frozen trees are untouched, and stay that way

```bash
for t in /mnt/raid0/llm/llama.cpp /mnt/raid0/llm/whisper.cpp /mnt/raid0/llm/qwentts.cpp; do
  echo "=== $t"
  git -C "$t" rev-parse HEAD
  git -C "$t" rev-parse --abbrev-ref HEAD
  git -C "$t" status --porcelain
done
```

Expected, and **record this output**; you compare against it at the end:

```
/mnt/raid0/llm/llama.cpp   67a433bf45a8a091d83b4ea0b32ff0735fd51800  production-consolidated-v8
    ?? .gitnexusignore
    ?? tools/math-tools/            <- both pre-date 2026-07-22, leave them
/mnt/raid0/llm/whisper.cpp  b307379226d93d9c5ed790d7cea0626613c0ef4b  production-speech-v1   (clean)
/mnt/raid0/llm/qwentts.cpp  2c1b5182e7e9f1acaa04405ff21747d8a7acf4d5  production-speech-v1   (clean)
```

If `llama.cpp` is not at `67a433bf4` on `production-consolidated-v8`, **stop**.
Something else moved production and every anchor you are about to take is wrong.

### 1.2 The host is yours to measure on

```bash
uptime                                    # 1-minute load
grep -c ^processor /proc/cpuinfo          # expect 192 (96 physical, SMT on)
awk '{print $1}' /proc/loadavg
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   # expect ~3.5e6 kHz
df -h /mnt/raid0
```

Thresholds the code enforces, not suggestions:

* `microbench.HostStatePolicy` refuses a run when the observed frequency is off
  nominal — this host has silently sat at −60% for days before
  (`feedback_host_throttle_check`). Verify the frequency, do not assume it.
* `worktree.BuildPlan(parallelism=BuildParallelism(..., load_average_cap=X))`
  makes the cap a **precondition**: `run_build` reads `getloadavg()` before
  configure and raises `HostTooContended` above it.
* As of 2026-08-03 23:00 this host carried load ~67 with six resident
  `llama-server` instances from another session and 92% of ROCm0 VRAM
  allocated. **Under that, do not bench.** A number taken there is garbage and
  it steals from whoever is working.

### 1.3 Nobody else holds the CPU region

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, "/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd")
from autokernel.execution import cpu_region_claim as CRC
print(CRC.default_region_lock_dir())
print(CRC.roles_present())
import json; print(json.dumps(CRC.inspect_region_claims(), indent=2)[:2000])
PY
```

**`roles_present()` is not a holder list — read it carefully.** It reports every
role that has a lock FILE in the root, held or not. On this host it currently
returns 27 role names, all of them free; the orchestrator creates the file the
first time a role ever claims and never removes it. The thing to read is
`inspect_region_claims()`, whose per-region rows carry `"held": true/false`,
`holder_pids` and `payload_is_stale`:

```bash
python3 - <<'PY'
import sys, json
sys.path.insert(0, "/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd")
from autokernel.execution import cpu_region_claim as CRC
state = CRC.inspect_region_claims()
held = [r for rows in state["regions"].values() for r in rows if r["held"]]
print(json.dumps(held, indent=1) if held else "no region is held")
PY
```

Anything in `held` overlapping your footprint means another role is on those
cores. **Do not force it and do not unlink a lock file** — the lock root is
`/mnt/raid0/llm/tmp`, which this project's own storage plane lists as a
*sweepable scratch root*, so an unlinked lock leaves your `flock` alive on an
orphaned inode while the path everyone else tests is free. Coordinate, or claim a
disjoint region and accept that your footprint is not the canonical one — which
means your numbers are not comparable to the canonical baseline and must say so.

Also run the host-topology check once, before the first bench:

```bash
python3 -c "
import sys; sys.path.insert(0,'/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd')
from autokernel.execution import cpu_region_claim as CRC
c = CRC.verify_host_topology(); print(c.outcome, c.reasons)"
```

`PASS` means the SMT sibling fold this module assumes matches the machine.
`COULD_NOT_CHECK` is **not** a pass.

### 1.4 The package is green

From `/mnt/raid0/llm/epyc-inference-research`:

```bash
python3 -W error::ResourceWarning -m unittest discover \
    -s scripts/kernel_rnd/autokernel -t . -p "test_*.py"
ruff check scripts/kernel_rnd/autokernel
```

A red suite here is a red campaign. In particular
`execution/test_execution_chain.py` is the composition test; if it fails, the
stages do not fit together and building a candidate will only find that out
later and more expensively.

---

## 2. Cold start to a first candidate

The order below is the order the code enforces. Steps 3 and 5 are the two that
consume real machine time.

### Step 1 — acquire the claim, and bind it for BOTH consumers

```python
import sys; sys.path.insert(0, "/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd")
from autokernel.evaluator import recipes
from autokernel.execution import chain, cpu_region_claim as CRC

# The footprint is read OFF the ratified prefix. Never retype "0-95".
prefix = list(recipes.CANONICAL_PREFIX)            # taskset -c 0-95 numactl --interleave=all
CPU_LIST = prefix[prefix.index("-c") + 1]

journal = CRC.RegionClaimJournal("/mnt/raid0/llm/ak-claims/region.jsonl")
claim = CRC.acquire_cpu_region_claim(
    CPU_LIST, role="autokernel", purpose="AK first campaign, decode_b1",
    campaign_id="ak-0001", journal=journal,
    timeout_s=600.0, max_hold_s=6 * 3600)

binding = chain.bind_claim(claim, cpu_list=CPU_LIST)
check = chain.check_claim_satisfies_both_seams(claim, cpu_list=CPU_LIST)
assert check.outcome == "PASS", check.reasons
```

`bind_claim` exists because the two consumers want different Protocols:
`t0_provider` reads `verify_held()`/`covers()` off the claim itself, and
`microbench` calls `attest()` — which `CpuRegionClaim` does not have. Passing the
raw claim to `MicrobenchRunner` raises `TypeError` **an hour into your claim
window**, after the worktree is built. Bind once, up front.

Everything from here to Step 8 happens **inside** this claim.

### Step 2 — anchor on the CURRENT production tip and make a worktree

```python
from autokernel.execution import worktree as WT

repo = WT.GitRepo("/mnt/raid0/llm/llama.cpp")             # READ-ONLY by construction
anchor = WT.resolve_anchor(repo, "production-consolidated-v8",
                           expected_commit="67a433bf45a8a091d83b4ea0b32ff0735fd51800")
wt, proof = WT.create_campaign_worktree(anchor, "ak-0001")   # /mnt/raid0/llm/llama.cpp-ak-0001
assert proof.holds, proof.differences
```

`expected_commit` turns "I believe production is at v8" into a checked
precondition. `create_campaign_worktree` re-resolves the tip and raises
`StaleAnchor` if it moved — CLAUDE.md step 1, and
INC-20260706-iqk-missing-subsystem is what happens when it is skipped.

`GitRepo` carries no content-mutating verb, and `Worktree` requires a
`SandboxPath` that cannot name a frozen tree. That is why `create_campaign_worktree`
may address `/mnt/raid0/llm/llama.cpp` at all: `git worktree add` writes
`.git/worktrees/<name>/` there, which is administrative metadata, and `proof`
demonstrates the working tree, branch and index did not move.

Apply the candidate's mutation in `wt` and commit it **with an explicit
pathspec** — never `git add .` in a shared clone
(`feedback_no_wholesale_git_add_shared_files`).

### Step 3 — build (this is the first thing that costs real time)

```python
plan = WT.BuildPlan(
    source_root=wt.path,
    build_dir=WT.default_build_dir("ak-0001", "akc-0001"),   # /mnt/raid0/llm/ak-build/...
    actor_worktree=wt.path,
    parallelism=WT.BuildParallelism(jobs=64, load_average_cap=8.0),
    targets=("llama-cli", "llama-bench", "test-backend-ops"),
    cmake="/usr/bin/cmake")

result = WT.run_build(plan, log_path="/mnt/raid0/llm/ak-build/ak-0001/akc-0001.log")
```

Notes that matter:

* `build_dir` is **outside** the worktree, by construction. `BuildPlan` refuses
  otherwise, because `integrity.check_clean_build_from_snapshot` FAILs a build
  directory inside the actor's tree.
* `GGML_CCACHE=OFF` is forced unless you pass `allow_ccache=True`. Leave it off.
  With ccache on, `chain.build_evidence` sets `incremental_objects_present=True`
  and the clean-build gate FAILs — correctly: a cache populated by another tree
  makes the actor's build state part of the artifact.
* `load_average_cap` is now a precondition, not a note. It raises
  `HostTooContended` before configure.
* Expect the first full build to take a while. Do **not** pipe the build through
  anything — under the default shell a pipe loses the compiler's exit status
  (`feedback_pipe_hazards`), which is exactly the "exit 0 with `Error 2` in the
  log" case `BuildResult.log_disagrees_with_exit_code` detects.

Then the receipt and its projection:

```python
from autokernel.evaluator import integrity

snapshot = integrity.hash_source_tree(wt.path.path, exclude_dir_names=(".git",))
identity = WT.build_identity(
    result, candidate_id="akc-0001", campaign_id="ak-0001",
    worktree=wt, snapshot=snapshot,
    output_binary=f"{plan.build_dir.path}/bin/llama-cli",
    toolchain="cmake + GNU make",
    libraries={"libggml.so.0":     f"{plan.build_dir.path}/bin/libggml.so.0",
               "libggml-base.so.0": f"{plan.build_dir.path}/bin/libggml-base.so.0"})

build_evidence = chain.build_evidence(identity)      # -> correctness.BuildProvenance
assert build_evidence.worst.outcome == "PASS", build_evidence.worst.reasons
```

`chain.build_evidence` is not a rename. `worktree` and `correctness` have two
different records both called `BuildProvenance`, and one field **inverts**:
`BuildIdentity.production_tree_paths` is the denylist, while
`correctness.BuildProvenance.production_tree_paths_touched` is the violation
list. Copying it across by name FAILs every candidate forever; copying `()`
makes the check vacuous. Use the function.

### Step 4 — linkage, then the artifact identity, MEASURED

```bash
bash /mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh \
     /mnt/raid0/llm/ak-build/ak-0001/akc-0001/bin/llama-cli
```

It must report every ggml library resolving **inside the candidate's own bin
directory**. If any resolves out of `/mnt/raid0/llm/llama.cpp`, the launcher's
`LD_LIBRARY_PATH` is wrong and the binary is silently running production's ggml.
This is not hypothetical — `testdata/recorded_t0_linkage_fail.txt` is that exact
failure, recorded on this host.

```python
from autokernel.execution import t0_provider as T0
report = T0.parse_linkage_report(open("linkage.txt").read())
linkage_sha256 = T0.ExecutedT0EvidenceProvider.linkage_digest(report)

artifact = chain.measure_artifact_identity(
    source_root=wt.path.path,
    binary=f"{plan.build_dir.path}/bin/llama-cli",
    linkage_sha256=linkage_sha256)
```

`measure_artifact_identity` re-walks the source tree and re-hashes the binary. It
does **not** copy the receipt's digests, because the T0 clean-build gate compares
the two and a helper that filled both sides would make that comparison `x == x`.
Note that on the real `llama.cpp` the tree walk is over ~13 GiB and it is paid
once per candidate — budget for it.

### Step 5 — T0

```python
capture = T0.capture_anchor(plan=anchor_plan, runner=T0.SubprocessRunner(),
                            claim=binding.t0_claim)
t0_anchor = chain.bind_anchor(capture, tool="llama-cli")

provider = T0.ExecutedT0EvidenceProvider(
    plan=t0_plan, runner=T0.SubprocessRunner(), claim=binding.t0_claim,
    anchor_capture=t0_anchor.capture)
t0_report = correctness.T0CorrectnessRunner(provider=provider,
                                            policy=policy).evaluate(request)
```

`bind_anchor(..., tool=...)` is not decoration — see §6.2.

### Step 6 — T1

```python
from autokernel.execution import microbench as MB

t1_anchor = chain.bind_anchor(anchor_bench_capture, tool="llama-bench")
assert chain.check_anchor_build_is_one_build([t0_anchor, t1_anchor]).outcome == "PASS"

run = MB.MicrobenchRunner(claim=binding.microbench_claim,
                          policy=MB.HostStatePolicy(nominal_khz=<measured>),
                          spawner=MB.SubprocessSpawner()).run(t1_plan)
blocks = run.paired_blocks()          # RAISES if the run was refused
```

**Dry-run it first.** `MB.RecordedSpawner` replays recorded output through the
entire pipeline — argv, env, pairing, parsing, reduction — without spawning
anything. Do that once before spending an hour of claim on a plan that turns out
to be malformed.

### Step 7 — controls, verdict, bank

The controls sweep, the `api.TierDispatcher` dispatch and the controller walk are
composed exactly as `execution/test_execution_chain.py::ChainLeg` does it. That
class is the reference composition; read it rather than reconstructing the order
from prose.

### Step 8 — release the claim and tear down

```python
receipt = WT.teardown_worktree(wt, witness_trees=list(WT.PRODUCTION_TREES))
assert receipt.to_dict()["all_production_trees_unchanged"]
claim.release()
```

Then **re-run §1.1 and diff it against what you recorded.** Byte-identical, or
you have a finding that outranks everything else you did today.

---

## 3. What "it is working" looks like

* `roles_present()` shows `('autokernel',)` while you hold the claim and `()`
  after `release()`.
* The build log's last section ends with the linked outputs you asked for, and
  `BuildResult.log_disagrees_with_exit_code` is `False`.
* `verify_ggml_linkage.sh` says `PASS: all linked ggml libraries resolve inside
  <your build>/bin`.
* `t0_report` has **17** gates. Today, a good candidate looks like *8 PASS and 9
  COULD_NOT_CHECK, zero FAIL* — see §6.1. A `FAIL` on
  `t0.source_integrity.clean_build_from_snapshot` or
  `t0.affected_surface_reconciliation` is a real finding about your candidate.
* Every `llama-bench` argv begins with `taskset -c 0-95 numactl --interleave=all`
  and carries `-fa 1`, and every env carries the full OMP stack
  (`OMP_PROC_BIND=spread`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=active`,
  `OMP_DYNAMIC=false`, `GGML_IQK=1`). `llama-bench` defaults to `-fa 0`; a
  default-flags number is real and useless.
* The record grammar line ends with `SEARCH RECORD, NOT A CLAIM` and carries
  `controls=5/5`.
* The three frozen trees are byte-identical at the end.

---

## 4. What should abort it

Stop immediately, do not "just get a number", if any of these happen.

| Signal | Why it is an abort |
|---|---|
| `git status --porcelain` on any frozen tree differs from §1.1 | The inviolate boundary moved. Everything else waits. |
| `StaleAnchor` from `resolve_anchor` or `create_campaign_worktree` | Production moved under you; the fork point is wrong and the failure would be silent. |
| `HostTooContended` from `run_build` | Someone else is on the machine. |
| `check_frequency` non-PASS | The CPU is throttled. This host has sat at −60% for days undetected. |
| `ClaimNotHeld` anywhere | Denial 8. The claim was lost mid-window; every number after it is unclaimed. |
| Linkage resolving into `/mnt/raid0/llm/llama.cpp` | The binary is running production's ggml. The numbers will look plausible and be wrong. |
| `MicrobenchRun.complete == False` | Do not reach for the partial blocks. `paired_blocks()` raises on purpose; `raw_vector()` is there so the refusal is still durable. |
| `chain.AnchorNotOneAnchor` | Two stages of one leg are comparing against different anchors. |
| A `ProductionMutated` / `ProductionTreeViolation` from anywhere | A path escaped the sandbox. Stop and find out how. |

**Never** use a name-pattern process operation to clean up — no `pkill`, no
`pgrep`, no `killall`, no `ps | grep | kill`. INC-20260731: a name-pattern kill
took out another agent's `llama-server` twice and killed `earlyoom`, whose own
argv contains the names it guards. Kill only PIDs you captured, escalate
`TERM` → `KILL`, and confirm with `ps -p <pid>`.

---

## 5. Deterministic replay — do this before you spend a claim

Every executor takes its process seam as an injectable, so the entire pipeline
runs off recorded output with no host, no claim and no build:

* `T0.RecordedProcessRunner([...])` for T0 — raises on an argv it has never seen
  rather than synthesising a blank capture.
* `MB.RecordedSpawner({...})` for T1 — records the argv and env it was handed, so
  you can assert on what *would* have been executed.

`execution/test_execution_chain.py` is that dry run, end to end, and it takes
four seconds. Run it after any edit to a plan.

---

## 6. Before a first campaign can start — the honest list

These are ordered. The first one blocks a *win*; the rest shape what a green run
means.

### 6.1 T0 has no producer for eight of its seventeen surfaces

A candidate through today's `ExecutedT0EvidenceProvider` gets **8 PASS and 9
COULD_NOT_CHECK**. The COULD_NOT_CHECKs are not bugs — an unevaluated surface
reading COULD_NOT_CHECK instead of PASS is the whole design — but you should know
which they are before you interpret a report:

| Surface | Why it is unmeasured |
|---|---|
| `symbol_and_registration_preservation` | needs a `correctness.SymbolTableDiff`; `integrity.extract_elf_symbols` exists and is not wired |
| `semantic_diff_conformance`, `schema_and_diff_policy` | need `DiffPolicyEvidence`; `integrity.parse_unified_diff` exists and is not wired |
| `static_and_compile_checks` | produced only when the anchor capture carries `compiler_id`/`compiler_version`; pass them |
| `sanitizer.asan`, `sanitizer.ubsan`, `unseen_boundary_shapes`, `state_rollback_teardown_race` | all gate on `ChangeSurface.derived_touches_*`, which is `None` unless `evaluator/surface.py`'s derivation is supplied to the plan |
| `exact_reference_comparison` | recorded as a deliberate gap: `test-backend-ops` prints its error metric only on FAILURE, so a passing case yields no observed ULP. Needs an exact-reference harness. |

Wiring the diff and symbol producers is a few hours and turns four
COULD_NOT_CHECKs into real gates. Supplying the surface derivation turns four
more. Do that before you conclude anything from a T0 report.

Separately: `collect_state_safety` hardcodes `rollback_tested=False`, so with
`state_safety_probe=True` that gate can only be FAIL and with it False only
COULD_NOT_CHECK. **It cannot PASS today.** Leave the probe off until there is a
rollback probe.

### 6.2 The anchor triple cannot name two tools — bind two

`api.AnchorIdentity.binary_sha256` is single-valued. T0 hashes the anchor
`llama-cli`; `microbench` compares the plan's anchor digest against the anchor
`llama-bench` it is about to spawn. One triple cannot honestly name both. Use
two `chain.bind_anchor(..., tool=...)` bindings and tie them with
`chain.check_anchor_build_is_one_build`, which enforces what genuinely must hold
across tools of one build: same `source_commit`, same `linkage_sha256`.

Required follow-up in `evaluator/api.py` (owned elsewhere): either a per-tool
digest table on `AnchorIdentity`, or a documented rule that `binary_sha256` names
the tool the record's `metric` is measured with.
`controller/state_machine.AnchorIdentity` already keys its digests by backend, so
the shape exists in the codebase — just not in the field the record uses.

### 6.3 THE BLOCKER — no candidate can cross the evidence threshold

For the CPU decode cell the calibration solves `B_min = 5` and
`threshold = 10`, and the sign-martingale e-value over 5 same-sign blocks tops
out at **5.57**. No candidate can cross on the base segment alone, whatever its
true effect — verified, not inferred (`ChainLeg` at factor 1.08, 1.3, 1.6 and 3.0
all return `e_value = 5.5687`, because the construction is sign-based and the
magnitude does not enter). Every win must come from the declared extension round.

And the extension round **has no producer**:

* `statistics._check_extension_structure` already accepts base blocks followed by
  whole extension rounds;
* `microbench.plan_blocks` already takes `segment=` and `extension_round=`;
* but `MicrobenchPlan` has no field for either, and `MicrobenchRunner.run()`
  calls `plan_blocks` with the defaults — so the runner emits `SEGMENT_BASE` and
  nothing else.

Pinned by `test_execution_chain.TestTheExtensionRoundHasNoProducer`, which also
demonstrates that the far side is ready. The fix is two fields on
`MicrobenchPlan` threaded into that one `plan_blocks` call. It was deliberately
not made here: whether the order schedule is re-derived or extended across two
runner invocations is a statistical decision (`OrderSchedule.order_for` is
prefix-stable, which suggests extended), and guessing it would put an
unverifiable assumption under every effect estimate. **Settle this first; without
it the campaign runs and banks `evidence_below_threshold` forever.**

### 6.4 A claim id and a candidate id are spelled the same

`cpu_region_claim._CLAIM_ID_PREFIX` is `"akc-"`, which is the prefix
`api.EvaluationRequest` requires of a **candidate** id. A claim id passed where a
candidate id belongs therefore satisfies the one validator written to catch that
class of mistake, and the record grammar renders it as `res=akc-…` beside
`candidate=akc-…`. Change it to `akclaim-` (which `t0_provider`'s own fixtures
already use) and update that module's assertions.

### 6.5 Smaller, recorded rather than fixed

* `correctness.BuildProvenance` has no `produced_by`, so an evaluator-measured
  build record is indistinguishable from an actor-supplied one, and no field
  for "the objects came from an external compiler cache" — `chain.build_evidence`
  folds ccache into `incremental_objects_present` and keeps the precise reason in
  `BuildEvidence.checks`. Both in `chain.SEAM_NOTES` and
  `t0_provider.SCHEMA_FOLLOWUPS`.
* `release/test_release_integration.py:1572` needs `"execution"` added to
  `_RULE_ONE_UNSCOPED_DIRS`. It is the sole failing test in the package and it is
  another workflow's file.
* `worktree._run_owned` can block in `Popen.__exit__`'s unbounded `wait()` if a
  child survives `SIGKILL` in uninterruptible D-state — plausible during a large
  build on this raid0. It does not report success falsely; it hangs. Reported by
  the worktree red-team, not fixed.

---

## 7. Where to look when something does not fit

`execution/chain.py` holds the four seams and argues each one in its docstring.
`execution/test_execution_chain.py::ChainLeg` is the reference composition —
claim, worktree, build, T0, T1, controls, verdict, bank, teardown — and every
negative path below it names the thing it refuses. Between them they are the
documentation that is checked by the test suite, which the file you are reading
is not.
