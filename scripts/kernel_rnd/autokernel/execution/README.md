# AutoKernel execution layer — runbook for the first real campaign

**Status: the measurement and five-control path has run live; the mutation/build
campaign has not.** On 2026-08-05 `execution/live_controls.py` acquired q0–q3,
calibrated from 200 fresh A/A plus 60 neutral blocks, and produced a 5/5 control
panel with `may_rank=true`. Evidence:
[`data/autokernel_controls_3pct_20260805/`](../../../../data/autokernel_controls_3pct_20260805/README.md).
No candidate worktree has yet been built by the campaign entrypoint. What follows
is how the session that owns compute goes from that calibrated instrument to a
first candidate.

Read §0 and §6 before you touch anything. §6 is short and it is the part that
decides whether today is a campaign or a plumbing session.

---

## 0. What exists, in one table

| Module | What it does | Ever run for real? |
|---|---|---|
| `cpu_region_claim.py` | Acquires the CPU region claim (real `flock`s, per-region, with a journal) | flocks yes, in tests; never around a benchmark |
| `worktree.py` | Resolves the production tip, adds a campaign worktree, configures + builds, emits a build receipt | never built anything |
| `t0_provider.py` | Runs `test-backend-ops`, `verify_ggml_linkage.sh`, generations, sanitizers; returns `correctness.T0Evidence` | never launched a tool |
| `microbench.py` | Runs the T1 paired-block `llama-bench` design under the claim; requires unique content/address, ordinary/full-device-sync hybrid, stable thread-set, and bitwise output receipts | pre-v9 controls ran 2026-08-05; hardened hybrid awaits explicit inference permission |
| `powercap_broker.py` | Reads root-owned package counters through one captured networkless/read-only container while the candidate remains non-root | live preflight PASS on 2026-08-12 |
| `instrument_integrity.py` | Pins reward-bearing measurement source to the named anchor before every live invocation | candidate/anchor source roots |
| `physical_bounds.py` | Re-derives the per-shape physical time floor and throughput ceiling from predeclared work/peak receipts | live runner requires it |
| `control_runner.py` | Scores the five controls through the same dispatcher a candidate uses | live 5/5 panel, 2026-08-05 |
| `live_controls.py` | Predeclares, measures, calibrates and scores the live CPU controls, including the fixed five-block control extension needed to make a positive verdict reachable | live; dry-run by default |
| `chain.py` | The **seams** between the above and the evaluator that reads them, plus the four T0 evidence projections (build, symbols, diff, change surface) | projection only; reads ELF/diff/log text it is handed, spawns nothing |
| `../campaign.py` | **The entrypoint.** Composes everything above into one loop and gives it a `main()` | dry-run composition yes; no candidate built, no bench spawned |

A live T1 plan is not constructively admissible without one physical envelope
for every measurement-material unit. Work terms are conservative lower bounds;
hardware peaks are conservative upper bounds. Therefore elapsed time uses
`max(work / peak_compute, bytes / peak_memory)`, while throughput uses the
inverse `min(peak_compute / work, peak_memory / bytes)`. The runner checks every
emitted repetition, not only the median, and refuses a missing, partial,
cross-shape, non-finite, or above-ceiling vector before it can be ranked.
The envelope is also bound to the registered metric's delivered unit and to a
digest of the exact recipe, model path, and parameter frame; a ceiling derived
for another unit or an easier invocation cannot grade the run.

An anti-short-circuit case is also not a gate-only label. `MicrobenchPlan`
requires its per-unit recipe parameters to differ from a normal control, ensures
the base segment contains every declared unit, constructs a distinct receipted
argv for each unit and arm, and places all of them on the same block schedule
consumed by the reducer. A hard case therefore contributes its actual cost to
the rank; it cannot merely pass correctness while the easy path supplies speed.

The surviving evaluator, journal, storage, controller-memory path and operator
surface are on the lean campaign path. The former broad release/adapters plane
was deliberately deleted on 2026-08-04 and is restored only for a scheduled
freeze or speech campaign; it is not silently present behind this driver.

**Read `../campaign.py` before working through §2 by hand.** Until 2026-08-04 this
package had NO entrypoint at all — `grep -rln '__main__|argparse|def main('` over
every non-test module returned nothing — and that, not a missing gate, is why it
had produced no results. §2 below is now a driver rather than a procedure:

```bash
# from /mnt/raid0/llm/epyc-inference-research — composes every step, executes NOTHING
python3 -m scripts.kernel_rnd.autokernel.campaign --model /path/to/model.gguf
```

Dry run is the **default**; `--execute` additionally requires `--i-hold-the-host`
AND an ops object with no unimplemented seams (refused at argv time, before the
claim — see `HostOps.unimplemented_seams`). An executing run also requires a
validated `--proposal-manifest` using the current proposal schema, an exact-unit
`--physical-envelope` (or `--ranked-units`), and a current
`--calibration-bundle`. The bundle is identity-bound to the production commit,
measurement-instrument commit and recipe; the v8 control bundle is rejected on
v9. The proposal is fsynced before preflight or any host work, and an identical
resume reuses that event; the same proposal id with different bytes is refused.

The driver's accept rule is `min(delta) > 0` over N pre-committed paired blocks
AND `median(relative) > contribution_floor`; both N's minimum and the floor come
from the current calibration bundle. A separate adjacent-anchor movement gate
retains the paired-design A/A control. The stock driver has no extension round;
the reusable extension path is protected by the durable completed-run ledger
described in §6.5.

`OrderSchedule` is a per-block coin flip on the campaign seed, **not** an
alternation: five blocks land all one way once in sixteen runs, and such a run
is a sequential A/B. The driver refuses that draw twice — from the plan, before
the blocks are spent, and again in `decide()` from the recorded orders.

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
/mnt/raid0/llm/llama.cpp   0db32c06e3e550065b78311a6031ef3dd2c4f27c  production-consolidated-v9
    ?? .gitnexusignore
    ?? tools/math-tools/            <- both pre-date 2026-07-22, leave them
/mnt/raid0/llm/whisper.cpp  b307379226d93d9c5ed790d7cea0626613c0ef4b  production-speech-v1   (clean)
/mnt/raid0/llm/qwentts.cpp  2c1b5182e7e9f1acaa04405ff21747d8a7acf4d5  production-speech-v1   (clean)
```

If `llama.cpp` is not at `0db32c06e3e` on `production-consolidated-v9`, **stop**.
Something else moved production and every anchor you are about to take is wrong.

The reward instrument is a separate reviewed source anchor:

```
/mnt/raid0/llm/autokernel/worktrees/ak-final-direct-20260813  283b520b527a7b507d6cf05cd124a59f427f3629  experimental-v9-autokernel-t0-final-direct-20260813
```

That commit descends from the production v9 commit above. It is an
experimental measurement instrument; serving remains frozen. The live preflight
proves the branch, commit, clean source tree, and production-ancestry edge.

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

### Step 2 — anchor on the reviewed measurement overlay and make a worktree

```python
from autokernel.execution import worktree as WT

repo = WT.GitRepo("/mnt/raid0/llm/autokernel/worktrees/ak-final-direct-20260813") # READ-ONLY by construction
anchor = WT.resolve_anchor(repo, "experimental-v9-autokernel-t0-instrument-20260813",
                           expected_commit="283b520b527a7b507d6cf05cd124a59f427f3629")
wt, proof = WT.create_campaign_worktree(anchor, "ak-0001")   # /mnt/raid0/llm/llama.cpp-ak-0001
assert proof.holds, proof.differences
```

Production v9 is checked independently. The measurement commit is its reviewed
one-commit evaluator overlay; `expected_commit` turns that identity into a checked
precondition. `create_campaign_worktree` re-resolves the tip and raises
`StaleAnchor` if it moved — CLAUDE.md step 1, and
INC-20260706-iqk-missing-subsystem is what happens when it is skipped.

`GitRepo` carries no content-mutating verb, and `Worktree` requires a
`SandboxPath` that cannot name a frozen tree. That is why `create_campaign_worktree`
may address `/mnt/raid0/llm/llama.cpp` at all: `git worktree add` writes
`.git/worktrees/<name>/` there, which is administrative metadata, and `proof`
demonstrates the working tree, branch and index did not move. The same structural
guard applies when the addressed clone is the experimental measurement source.

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
# The anchor's toolchain is MEASURED off THE ANCHOR'S OWN build log, not typed
# and not the candidate's. Without it `collect_static_analysis` returns None and
# the static gate is COULD_NOT_CHECK; with the CANDIDATE's log it used to return
# a PASS on a self-comparison, so `candidate_build=` is required and the
# candidate's own log is refused (§6.1a item 2).
toolchain = chain.anchor_toolchain_from_build_log(
    anchor_build_log_text, log_ref=f"file://{anchor_log_path}",
    candidate_build=build_ev.provenance)
capture = T0.capture_anchor(plan=anchor_plan, runner=T0.SubprocessRunner(),
                            claim=binding.t0_claim, **toolchain.as_capture_kwargs())
t0_anchor = chain.bind_anchor(capture, tool="llama-cli")

provider = T0.ExecutedT0EvidenceProvider(
    plan=t0_plan, runner=T0.SubprocessRunner(), claim=binding.t0_claim,
    anchor_capture=t0_anchor.capture)
t0_report = correctness.T0CorrectnessRunner(provider=provider,
                                            policy=policy).evaluate(request)
```

`bind_anchor(..., tool=...)` is not decoration — see §6.2.

**Four of the plan's fields are evidence, not configuration**, and each one is a
T0 surface that reads COULD_NOT_CHECK when it is left `None`. The producers are
in `chain.py`; `test_execution_chain.ChainLeg.t0_evidence_inputs` is the
reference wiring.

```python
# symbols -> symbol_and_registration_preservation.
# The ANCHOR BINDING, not a path: SymbolTableDiff carries no anchor triple, so
# this is the only place "which anchor is this diff against?" is answerable, and
# `symbol_evidence` refuses an anchor binary the binding did not measure. The
# registration tables are mandatory — `removed_op_registrations=()` from an
# extractor that was never constructed is the fail-open the whole gate is for.
libggml_anchor = chain.bind_anchor(libggml_capture, tool="libggml.so.0")
symbols = chain.symbol_evidence(
    anchor_binary=f"{anchor_bin}/libggml.so.0",
    candidate_binary=f"{plan.build_dir.path}/bin/libggml.so.0",
    anchor=libggml_anchor,
    declared=integrity.DeclaredSymbolDeltas.from_proposal(proposal),
    anchor_op_registrations=…, candidate_op_registrations=…,
    anchor_dispatch_predicates=…, candidate_dispatch_predicates=…)

# diff -> semantic_diff_conformance AND schema_and_diff_policy.
# `commit_was_pathspec_limited` is DERIVED from the commit argv, not declared;
# `production_tree_paths` is measured by resolving each repo-relative diff path
# against the worktree, which is the only way a `..` escape is visible at all.
diff = chain.diff_policy_evidence(
    diff_text=subprocess_output_of_git_diff,
    worktree_root=wt.path.path,
    declared_surface_files=proposal["declared_surface_files"],
    envelope=correctness.ChangeClassEnvelope(change_class=…, max_changed_lines=…,
                                             max_files_touched=…),
    branch_name=wt.branch.name,
    commit_argv=("git", "commit", "-m", msg, "--", *paths),
    record_schema_violations=validate_candidate_records())

# change_surface -> asan, ubsan, unseen_boundary_shapes, state_rollback_teardown_race,
# and the derived-op set `check_backend_op_units` unions with the mandatory ops.
surface_evidence = chain.change_surface_from(
    surface.derive_affected_surface(candidate_id=…, diff=…, indexes=[…],
                                    registrations=…),
    diff_text=subprocess_output_of_git_diff)

t0_plan = T0.T0ExecutionPlan(
    …,
    build=build_evidence.provenance,
    **chain.t0_plan_evidence(
        symbols=symbols, diff=diff, change_surface=surface_evidence))
```

`t0_plan_evidence()` deliberately returns the three records and their routed
checks together. Do not copy only `.diff`, `.policy`, and `.surface`: each wrapper
can carry facts the far-side dataclass has no field for — an incomparable
registration arity, a binary file with no line count, an ELF extractor coverage
gap, or an underdetermined behavioural touch.

### Step 6 — T1

```python
from autokernel import journal as J
from autokernel.execution import microbench as MB

t1_anchor = chain.bind_anchor(anchor_bench_capture, tool="llama-bench")
assert chain.check_anchor_build_is_one_build([t0_anchor, t1_anchor]).outcome == "PASS"

book = J.Journal(<durable-journal-root>, campaign_id=campaign.campaign_id)
run_ledger = MB.CompletedRunLedger(book, campaign_id=campaign.campaign_id)

runner = MB.MicrobenchRunner(claim=binding.microbench_claim,
                             policy=MB.HostStatePolicy(nominal_khz=<measured>),
                             spawner=MB.SubprocessSpawner(),
                             run_ledger=run_ledger)
base_run = runner.run(t1_plan)        # t1_plan.base_blocks == campaign.b_min
```

`t1_plan.params` carries a non-zero `autokernel_seed` derived from the committed
campaign/candidate identity. The recipe always emits
`--autokernel-harden <seed>`. The instrument creates two simultaneously-live
contexts per reported repetition: the first sees unique content inside the timing
bracket; the second sees the same content at different input/context/output
addresses outside the bracket, and its logits must be bitwise identical. Every
context receives different warm-up content, so a stale pointer-keyed cache returns
different stale answers across the pair and is refused. The JSON receipt records
the content/output hashes, paired addresses, and rotated working-set size;
`microbench` rejects missing, reused, malformed, or non-invariant material.

The candidate and anchor must both be built from the reviewed experimental
instrument baseline. `instrument_integrity.compare_manifest_to_anchor()` re-hashes
`tools/llama-bench/llama-bench.cpp`, `tests/test-backend-ops.cpp`, and
`tests/test-quantize-perf.cpp` at run open and before every invocation. The plan
carries explicit source roots; a build directory cannot stand in for source identity.
Production serving source remains frozen; it is not patched in place to acquire
this measurement-only mode.

**The base segment cannot cross, so this is not the end of Step 6.** §6.3 is the
arithmetic: at `B_min = 5` the sign-martingale tops out at `e = 5.5687` against a
threshold of 10, at every true effect. A campaign that stops here banks nothing
and reports `evidence_below_threshold` forever, which reads like "no candidate
was good enough" and is actually "the instrument cannot resolve a win at all".
Run the declared extension round and pool:

```python
# The licence comes off the CAMPAIGN — the same `statistics.CampaignStatistics`
# the reduction runs under. There is no spelling that takes a rule and a
# commitment: a pair the caller mints verifies against itself and against
# nothing (§6.3).
rounds = []
for round_index in range(1, campaign.stopping_rule.extension.max_rounds + 1):
    licence = MB.ExtensionAuthorization(campaign=campaign, round_index=round_index)
    rounds.append(runner.run(t1_plan.extend(licence)))
    # WHEN to extend is the rule's, not yours: drive it from
    # `campaign.sequential_evaluation(...)`, which terminates itself.

blocks = MB.assemble_run_blocks(base_run, rounds, campaign=campaign,
                                run_ledger=run_ledger)
# RAISES on a refused run, a round licensed by another campaign, a base segment
# that is not B_min blocks, or two runs that are not the same plan.
```

`base_run.paired_blocks()` alone is the right call only for a campaign whose rule
declares `max_rounds = 0`, and such a rule can never bank a candidate in this
cell.

**Dry-run it first.** `MB.RecordedSpawner` replays recorded output through the
entire pipeline — argv, env, pairing, parsing, reduction — without spawning
anything. Do that once before spending an hour of claim on a plan that turns out
to be malformed.

### Step 7 — controls, verdict, bank

The controls sweep, the `api.TierDispatcher` dispatch and the controller walk are
composed exactly as `execution/test_execution_chain.py::ChainLeg` does it. That
class is the reference composition; read it rather than reconstructing the order
from prose. Its stages, in the order the code enforces them, are

```
claim -> worktree -> build -> artifact -> anchor -> t0
      -> t1 -> extend -> reduce -> controls -> dispatch
```

and **`extend` is the one that is easy to leave out.** `ChainLeg.extend_and_pool()`
runs every round `campaign.stopping_rule.extension` declares and pools them with
the base segment through `MB.assemble_run_blocks`; `ChainLeg.reduce()` then
reduces the **pooled** sequence, not `t1_run.paired_blocks()`. A leg without that
stage is green in every other respect and banks `evidence_below_threshold` for
every candidate at every true effect — this class itself did exactly that until
2026-08-04, while Step 6 above already said to run the round.

What a banked win looks like, and what a null looks like, are both pinned in
`test_execution_chain.TestAWinIsReachableAndANullIsRefused`:

| | base segment (5) | declared budget (10) | resolution | rule replay |
|---|---|---|---|---|
| candidate at a true +8% | `e = 5.5687` | `e = 42.2877`, crosses at block 7 | `improvement`, **ranked** | `evidence_threshold_crossed` -> `compose_into_champion_lineage` |
| candidate with NO true effect | `e = 5.5687` *(worst-case draw)* | `e = 5.5687`, never crosses | not rankable | `extension_exhausted` -> `abandon` |

Both rows end at `BANK_EVENT`. Banking is not the same as winning: a candidate
that did not earn a rank is banked as an abandon, with its reason.

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
* `t0_report` has **17** gates. The generic recorded fixture now looks like
  *12 PASS and 5 COULD_NOT_CHECK, zero FAIL* — see §6.1. That fixture predates
  the structured `AK_REF_V1` receipt; the current instrument closes exact
  reference when it emits one for every required op. One remaining
  projection limitation was previously silent; its appearance is the §6.1b
  fail-open closing, not a regression. Note that the five
  remaining COULD_NOT_CHECKs each have a NAMED reason. A `FAIL` on
  `t0.source_integrity.clean_build_from_snapshot` or
  `t0.affected_surface_reconciliation` is a real finding about your candidate.
  So, now, is a `FAIL` on `symbol_and_registration_preservation`,
  `semantic_diff_conformance`, `schema_and_diff_policy`,
  `static_and_compile_checks` or `unseen_boundary_shapes` — those five stopped
  being free.
* Every `llama-bench` argv begins with `taskset -c 0-95 numactl --interleave=all`
  and carries `-fa 1`, and every env carries the full OMP stack
  (`OMP_PROC_BIND=spread`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=active`,
  `OMP_DYNAMIC=false`, `GGML_IQK=1`). `llama-bench` defaults to `-fa 0`; a
  default-flags number is real and useless.
* **T1 ran the whole declared budget.** `len(blocks)` after
  `assemble_run_blocks` is `b_min + max_rounds * blocks_per_round` — **10** for
  this campaign, not 5 — and `reduction.mde.window_length` is the same number. If
  it is 5 you stopped at the base segment and nothing can cross (§6.3); if
  `mde.window_length` is 15 you are on a build from before §6.7.
* The reduction's `admissible` is PASS and `mde_window` is PASS.
* A win looks like `e_value >= threshold` **and**
  `verdict.effect_resolution == "improvement"` **and**
  `verdict.speed_rank_admissible` **and** the rule replaying to
  `evidence_threshold_crossed` -> `compose_into_champion_lineage`. All four, or
  it is not a win. A candidate that reaches `BANK_EVENT` with
  `evidence_below_threshold` has been banked, not won.
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

These are ordered. §6.3 — the one that blocked a *win* outright — is **closed as
of 2026-08-04**, and is kept here because the fact underneath it still governs
how a campaign must be planned: nothing crosses on the base segment. §6.5 is
**closed as of 2026-08-05** by the durable completed-run ledger; read it before
you run a second round of anything. §6.7 is **new on 2026-08-04**
and is closed: the pooled reduction that §6.3 made possible published an MDE for
a window the stopping rule cannot license. The rest shape what a green run means.

**Status, in one line each.**

| § | What | State |
|---|---|---|
| 6.1 | T0's unwired producers | **CLOSED** — 12 PASS / 5 COULD_NOT_CHECK after exported-version coverage became conditional and §6.1b made remaining limitations verdict-bearing |
| 6.1a | Four defects in what 6.1 produced | **CLOSED** — bite-tested |
| 6.1b | The projections' refusal channel reaches nothing | **CLOSED 2026-08-10** — projection checks merge into their existing integrity gates |
| 6.2 | One anchor triple cannot name two tools | **CLOSED** — `AnchorIdentity.tool`, enforced three ways |
| 6.3 | The extension round had no producer | **CLOSED** — producer, *and* a reference composition that runs it |
| 6.4 | Claim ids and candidate ids share a prefix | **CLOSED** — refused at import |
| 6.5 | A declared round can be re-run until it crosses | **CLOSED 2026-08-05** — every runner return is fsynced by declared key; reruns refuse before inference; pooling requires the same ledger identity |
| 6.6 | Smaller, recorded rather than fixed | mixed, each noted |
| 6.7 | The published MDE described an unlicensable window | **CLOSED** — `b_min`, plus a new `mde_window` check |
| 6.8 | How much block noise the declared budget tolerates | **OPEN as a PLANNING decision** — read it before you declare a rule |

### 6.1 T0's unwired producers — WIRED 2026-08-04; current receipts close exact reference

Was: *8 PASS and 9 COULD_NOT_CHECK*, because four of the nine were unevaluated
only for want of a line of code carrying `integrity.py`'s and `surface.py`'s
output across to `correctness.py`'s shape. Those lines are now
`chain.symbol_evidence`, `chain.diff_policy_evidence`,
`chain.anchor_toolchain_from_build_log` and `chain.change_surface_from`; the
wiring is in Step 5 above and in `test_execution_chain.ChainLeg`. **A candidate
now gets 12 PASS and 5 COULD_NOT_CHECK.** The remaining additional limitation is
the derived-surface absence proof §6.1b had previously dropped. ELF version
coverage closes when exported definitions are demonstrably unversioned; a
genuinely versioned exported surface still reports `COULD_NOT_CHECK`.

**Recorded-fixture census re-verified 2026-08-11: 17 gates, 12 PASS, 5
COULD_NOT_CHECK, 0 FAIL.** The fixture is deliberately retained in its historical
pre-receipt form, so exact reference remains one of its five named gaps. A run
from the current instrument emits `AK_REF_V1` with the actual comparator metric,
observed maximum, tolerance, comparison count, and activated CPU-reference oracle;
the provider projects it as `metric_bounded` evidence and refuses malformed,
missing-op, candidate-derived, or over-tolerance receipts. The symbol-version
closure is T0-local; the extension is T1 and the MDE is the reducer's. The
recorded-fixture count is pinned by
`TestTheChainFits.test_exactly_five_t0_surfaces_are_explicitly_unevaluated`, which
fails if it moves. The thirteen, by name:

`symbol_and_registration_preservation`, `clean_build_from_snapshot`,
`semantic_diff_conformance`, `schema_and_diff_policy`, `static_and_compile_checks`,
`backend_op_units`, `unseen_boundary_shapes`, `affected_surface_reconciliation`,
`no_fallback_dispatch_proof`, `output_coherence_vs_anchor`, `determinism_class`,
`binary_and_linkage_identity`, `anti_reward_hacking`.

Five surfaces moved, and each is a gate that can now FAIL your candidate:

| Surface | What produces it | What newly FAILs |
|---|---|---|
| `symbol_and_registration_preservation` | `chain.symbol_evidence` over `integrity.extract_elf_symbols` + two `RegistrationTable` pairs | an undeclared symbol removal, an undeclared arity change *(symbol **or** op-registration — the second added by the red team below)*, a removed op registration or dispatch case, >60% exported-surface shrinkage |
| `semantic_diff_conformance` | `chain.diff_policy_evidence` over `integrity.parse_unified_diff` | a file outside the declared surface, an unrelated deletion, a diff over its change-class envelope |
| `schema_and_diff_policy` | same record | a production-named branch, a diff path resolving into a frozen tree, a commit that was **not** pathspec-limited |
| `static_and_compile_checks` | `chain.anchor_toolchain_from_build_log(anchor_log, log_ref=…, candidate_build=…)` -> `capture_anchor(compiler_id=…, compiler_version=…, warning_count=…)` | a compiler error, an analyzer finding, a candidate/anchor toolchain mismatch, a new warning vs the anchor — **but only if you feed it the ANCHOR's log; see the red team below** |
| `unseen_boundary_shapes` | `chain.change_surface_from` (`derived_touches_dispatch`) + a `HoldoutPlan` | a dispatch change with no holdout at all, a holdout the planner could see, any unseen/boundary failure |

Two of those five need something the campaign must DECLARE, and there is no
default for either — supplying nothing means no evidence and a COULD_NOT_CHECK,
which is honest:

* the backend adapter's **registration patterns**, for the two
  `PatternRegistrationExtractor`s. `removed_op_registrations=()` from an
  extractor that was never constructed is indistinguishable from a clean one;
* the proposal's **declared surface files** and **declared symbol deltas**
  (`integrity.DeclaredSymbolDeltas.from_proposal`, which RAISES on an absent
  declaration rather than defaulting to empty).

The generic source-candidate limitations, with the reason each remains open:

| Surface | Why it is still unmeasured | Is it closable here? |
|---|---|---|
| `sanitizer.asan` | needs `derived_touches_memory` **or** `derived_touches_threading` to be `False` before it can PASS by non-applicability. `chain.classify_behavioural_surface` answers `True` or `None` and **never `False`** by construction — proving "no reachable path allocates" needs a whole-program analysis, not a token list. For THIS candidate the diff matches no memory or threading token, so the flag is `None`. | No. A candidate whose diff *does* touch memory gets a real, speed-blocking FAIL instead — `TestTheBehaviouralClassifierOnlyWidens` — so the surface is a gate whenever it can be one. |
| `sanitizer.ubsan` | identical; the two share `derived_touches_*` and the same PASS branch. | No, same reason. |
| `state_rollback_teardown_race` | `t0_provider.collect_state_safety` hardcodes `rollback_tested=False`, so `state_safety_probe=True` is a guaranteed FAIL for every candidate and `state_safety_probe=False` leaves the gate to the change derivation, which is `None` here. Proved by exhaustion in `test_t0_provider.TheStateSafetyGateCannotPass`. | No. It needs a rollback probe that does not exist. **Leave the probe off**; that test is the tripwire for when one lands. |
| `affected_surface_reconciliation` | the source classifier can widen memory/thread/state touches but cannot prove their absence for arbitrary source edits. | No for generic source edits without a whole-program closure proof. The registered source-free IQK parameter surface does prove all three false. |
| `exact_reference_comparison` | Historical captures have no positive receipt. The current `test-backend-ops` emits `AK_REF_V1` only after its separately activated CPU reference path ran and the case passed. | **Closed for current-instrument runs.** Missing or malformed receipts remain speed-blocking; old fixtures are not upgraded by assertion. |

None of the remaining generic limitations is a want of pass-through wiring.
For the first source-free parameter candidate, the registered surface proves
memory, threading, and persistent state are untouched, so ASAN, UBSAN, state
safety, and surface reconciliation resolve by non-applicability while the
structured backend-op receipt supplies the reference leg. Arbitrary source
proposals still need stronger reachability and rollback instruments.

**Read this before you over-read the sanitizer surface.**
`surface.AffectedSurface` has no memory / threading / persistent-state axis — its
axes are backends, link targets, op names, kernel symbols and dispatch
predicates. `chain.change_surface_from` therefore classifies those three flags
**lexically, from the diff body**, and answers `True` or `None` and **never
`False`**. That is deliberate: `False` is what licenses `check_asan`'s *"the
mechanical derivation finds it touches neither memory nor threading"* PASS, and
proving that needs a whole-program reachability pass, not a token list. So:

* a candidate whose diff visibly allocates or threads gets a **real, speed-
  blocking FAIL** if no ASAN/UBSAN run was recorded — that is the case that
  matters and it is now covered;
* a candidate whose diff does not gets COULD_NOT_CHECK, exactly as before.

Adding a token to `chain.BEHAVIOURAL_TOKENS` can only turn a COULD_NOT_CHECK into
a gate; it can never turn a FAIL into a PASS.

#### 6.1a The red team on the above — four defects, fixed 2026-08-04

The five surfaces really did move. Four of the gates they produced did not bite
where this section said they did, and all four are now bite-tested in
`test_execution_chain.py` section H.

1. **A declared ARITY CHANGE excused an undeclared REMOVAL.**
   `chain._declared_covering` matched a name in `SymbolDiff.removed` against
   *either* `declared.removed` *or* `declared.arity_changed`, and
   `SymbolDiff.removed` is by construction a removal with NO matching addition.
   So a proposal saying *"I will change the arity of
   `ggml::detail::kernel_dispatch`"* whose candidate DROPPED that specialization
   PASSed `symbol_and_registration_preservation`, byte-identically to a candidate
   that declared the removal honestly — §8.5.1's own headline example arriving
   through the gate written to catch it. A removal is now excused by
   `declared.removed` and nothing else; symmetrically, an arity change is excused
   by `declared.arity_changed` and nothing else.
   (`TestADeclaredArityChangeDoesNotExcuseARemoval`)

2. **The anchor toolchain was measured off the CANDIDATE's build log.**
   `ChainLeg.bind_anchor` — the reference composition this README tells you to
   copy — passed `self.build_log_text`, the candidate's. Both of
   `check_static_and_compile`'s cross-arm comparisons then had the same bytes on
   both sides: the toolchain-mismatch branch could never fire, and the
   new-warning delta was identically zero for every candidate that ever ran, so
   the gate's PASS was a self-comparison. Measured, not argued: with the old
   wiring, adding a warning to the candidate build takes the candidate count 2→3
   *and the anchor count 2→3*. `anchor_toolchain_from_build_log` now REQUIRES the
   candidate's `correctness.BuildProvenance` and refuses a log that resolves to
   the candidate's own, and `ChainWorld.anchor_tree()` writes the anchor's own
   `anchor-build.log`. (`TestTheAnchorToolchainIsMeasuredOffTheANCHORSLog`)

3. **`git commit -i -- <paths>` read as pathspec-limited.** `--include` means
   *"stage these paths IN ADDITION TO whatever is already staged"*; the default
   for a pathspec commit is `--only`, which disregards the index. `-i` is
   therefore the one spelling under which another session's staged files ride
   into the artifact WITH a pathspec present — exactly the hazard
   `commit_was_pathspec_limited` exists for — and it returned `True`. `-i`,
   `--include` and any bundled cluster containing `i` or `p` are now refusals,
   and the flag scan stops at `--` so a pathspec spelled like a flag is a
   filename.

4. **A registration ARITY change reached no gate at all.**
   `GGML_CPU_OP(MUL_MAT, 2)` → `GGML_CPU_OP(MUL_MAT, 5)` changes no exported
   name, links clean, and dispatches the op with the wrong operand count for
   every shape. `integrity.RegistrationDiff` has reported it since it was
   written; `correctness.SymbolTableDiff` had no field for it, so
   `chain.symbol_evidence` could only put it in a `checks` tuple **that nothing
   reads**, and T0 said PASS. `SymbolTableDiff.arity_changed_op_registrations`
   now exists and `check_symbol_and_registration_preservation` FAILs on any entry
   not declared. (`TestARegistrationArityChangeReachesTheGate`)

#### 6.1b CLOSED 2026-08-10 — projection refusals reach T0 integrity gates

`chain.t0_projection_checks()` now routes `SymbolEvidence.checks`,
`DiffEvidence.checks`, and `ChangeSurfaceEvidence.checks` through
`T0ExecutionPlan.projection_checks`. `evaluate_t0()` merges each finding into the
existing constitutional gate that owns it, preserving the 17-gate coverage
contract while making every non-PASS verdict-bearing:

* registration `arity_not_comparable` — exactly one side's declared pattern
  captured an arity, which is not "unchanged";
* **`elf_extraction_complete`** — the ELF extractor reported a coverage gap, so
  the symbol diff is over an incompletely-read table, and the gate still says
  PASS. This is the one that matters most: it is the difference between "no
  symbol was removed" and "no symbol was removed from the part we could read";
* `diff_is_textual` — a binary blob in the diff contributes no changed lines, so
  the §10.6 change-class envelope does not bound it;
* the three `derived_touches_*` UNDETERMINEDs.

The merged gate notes retain the projection check name and outcome. A malformed
triple or an attempt to target any other gate is refused when the plan/evidence
record is constructed.

### 6.2 The anchor triple cannot name two tools — bind two — CLOSED 2026-08-04

`api.AnchorIdentity.binary_sha256` is single-valued. T0 hashes the anchor
`llama-cli`; `microbench` compares the plan's anchor digest against the anchor
`llama-bench` it is about to spawn. One triple cannot honestly name both. Use
two `chain.bind_anchor(..., tool=...)` bindings and tie them with
`chain.check_anchor_build_is_one_build`, which enforces what genuinely must hold
across tools of one build: same `source_commit`, same `linkage_sha256`.

**The api.py half is now done.** Of the two options this section offered, the
second is implemented and ENFORCED rather than documented: `AnchorIdentity.tool`
names the tool whose binary `binary_sha256` is, and by rule that is the tool the
record's `metric` was measured with. The enforcement is three-part —
`identity_matches` FAILs two differently-named tools even when every digest
agrees, `for_tool()` refuses to re-label a triple as another tool, and `short()`
prefixes the tool into the record grammar (`vs anchor llama-bench:<commit>/…`).
`AnchorBinding.identity` stamps it, so the tool no longer evaporates between the
binding and the object the record reads. Not the per-tool digest table: this
object is the denominator of ONE ratio measured by one tool, where
`controller/state_machine.AnchorIdentity`'s per-backend table is the
campaign-wide production identity that must describe every backend the tree
serves. The argument is on the class; `test_api.AnchorNamesOneToolTest` and
`test_execution_chain.TestTheChainFits.test_one_capture_bound_for_two_tools_is_two_anchors`
hold it down.

`tool` is optional, because a record written before the field existed named no
tool and must stay readable. It is never *silently* compatible: named-against-
unnamed is COULD_NOT_CHECK, never PASS. Anything that builds an
`AnchorIdentity` by copying another one field by field will therefore drop the
name and degrade its own precondition — copy with `dataclasses.replace`.

**Red-team follow-up, same day, three defects — all now closed.** (a) The rule
*"`binary_sha256` is the digest of the tool the record's `metric` was measured
with"* had no enforcement point: `for_tool` refuses a RE-label, but the FIRST
label is a free string and `bind_anchor` accepts any of them, so an anchor
captured off `llama-bench` and bound `tool="llama-cli"` had a digest that MATCHED
the binary that ran, passed every check, and rendered `vs anchor llama-cli:…` as
the denominator of a ratio `llama-bench` produced. `MicrobenchRunner`'s new
`anchor_identity.tool` conjunct compares the plan's anchor against
`recipes.get_recipe(plan.recipe_id).tool` — the only place both halves of the
rule are present at once. (b) and (c) `identity_matches` now has THREE outcomes
and two `release/readiness.py` sites tested `!= PASS`, written when it had two:
`_check_anchor_agreement` filed an unobserved tool name as `BLOCK_ANCHOR_MOVED`
(a record asserting the denominator was REBUILT), and `T2Cell._require_same_window`
raised `CellInadmissible` saying the two halves were *"reductions of DIFFERENT
windows"*. Both now branch on FAIL; COULD_NOT_CHECK goes to
`BLOCK_ANCHOR_ABSENT`, which is that function's own existing bucket for an
unobserved component. **Grep for `identity_matches` before adding a consumer:
`!= PASS` is the wrong test.**

### 6.3 THE BLOCKER — CLOSED 2026-08-04: the extension round has a producer

**The fact, unchanged and still the reason this section exists.** For the CPU
decode cell the calibration solves `B_min = 5` and `threshold = 10`, and the
sign-martingale e-value over 5 same-sign blocks tops out at **5.5687**. No
candidate can cross on the base segment alone, whatever its true effect —
verified, not inferred (`ChainLeg` at factor 1.08, 1.3 and 3.0 all return
`e_value = 5.5687`, because the construction is sign-based and the magnitude
does not enter). Ten same-sign blocks reach **42.29**, crossing at block 7.
**Every win comes from the declared extension round.** Do not plan a campaign
that stops at `B_min`.

The producer now exists. `MicrobenchPlan` carries `segment` and an
`ExtensionAuthorization`, `MicrobenchRunner.run()` threads them into
`plan_blocks`, and `microbench.assemble_run_blocks(base_run, [round_1, ...])`
pools the runs into the one block sequence the reducer reads. Regression cover:
`test_execution_chain.TestTheExtensionRoundHasAProducer` (which replaced the pin
that used to live here) and four classes in `test_microbench.py`.

**And the reference composition now RUNS it — that half was still missing on the
morning of 2026-08-04, and a producer nobody calls is not a closed blocker.**
Step 6 above had been corrected to run the declared round; `ChainLeg`, which
Step 7 tells you to copy for everything after T1, still ended at
`run_t1()` + `reduce()` over five blocks. So the whole chain demonstrated a
green, seventeen-gate, five-control, fully-attested leg that reached
`BANK_EVENT` and banked **`evidence_below_threshold`** — the failure this
section exists to prevent, arrived at by copying the file the runbook points at.
`ChainLeg` now has an `extend` stage between `t1` and `reduce`
(`extend_and_pool()`), `reduce()` defaults to the pooled sequence, and
`walk()` runs the whole declared budget.

**The demonstration, both directions, is
`test_execution_chain.TestAWinIsReachableAndANullIsRefused`:** a candidate at a
true +8% pools to `e = 42.2877`, crosses at block 7, resolves `improvement`, is
speed-rank admissible, replays to `evidence_threshold_crossed` ->
`compose_into_champion_lineage`, and reaches `BANK_EVENT`; a candidate with no
true effect (`null_effect`, per-block factors centred exactly on 1.0 at the
calibration's own noise) pools to `e = 0.9000`, never crosses at any prefix, is
not rankable, and replays to `extension_exhausted` -> `abandon`. The hardest
null in that class reaches **exactly 5.5687** on its base segment — five
same-sign blocks from a true effect of zero — which is the clearest statement of
why 5.5687 is not a near miss.

**The schedule question is settled: EXTENDED, not re-derived.** One
`OrderSchedule`, whose `base_blocks` stays `B_min`, indexed straight through the
base segment and every round; round *r* starts at
`B_min + (r-1) * blocks_per_round`. The argument is in `microbench.py`'s module
docstring under *"the extension round is extended, not re-derived"* — five
pieces of code that only make sense under it, most decisively that
`OrderSchedule.order_for`'s reversed limb is unreachable under re-derivation, so
a re-derived round would repeat the base orders, which `BoundedExtension`
(`order="reversed"` only) cannot declare. A plan whose `base_blocks` disagrees
with its authorization raises `ScheduleMismatch`; two runs that are not the same
plan cannot be pooled.

**An extension is declared, never granted after the fact.**
`ExtensionAuthorization` takes the **campaign** — one
`statistics.CampaignStatistics` — and reads `max_rounds`, `blocks_per_round`,
`max_blocks_per_candidate` and the base length off it instead of accepting them.
Round `max_rounds + 1`, a round that would pass `max_blocks_per_candidate`, a
round index below 1, and `segment="extension"` with no authorization at all are
each `ExtensionNotDeclared` at construction — before a process is spawned.

It takes the campaign because taking a `(StoppingRule, StoppingRuleCommitment)`
pair did **not** hold, and the first version of this section claimed it did.
`commitment.verify(rule)` compares its two arguments to each other; a caller who
wanted a budget the campaign never declared never had to mutate a committed rule,
only to commit the rule it wanted. Red team, 2026-08-04, reproduced end to end:
a licence for round 3 of a `max_rounds=3` rule with a ceiling of 100, campaign id
`not-even-this-campaign` and `committed_at` in **2099**, constructed cleanly;
rounds 1, 2 and 3 were **spawned** — real benchmark minutes on a held claim —
and only the reducer refused the pooled 20 blocks afterwards. A single forged
round of the campaign's own shape was not refused at all: it reduced to
`admissible = PASS`, `e = 42.29`, banked, and its raw vector recorded
`campaign_id: "some-other-campaign"` as the licence for the round. Because the
campaign verifies its own commitment at construction and carries an accepted
calibration for the cell, a licence derived from it is issued by the same object
the evidence is reduced under.

A second campaign is still buildable, so the binding is re-checked where it
matters: `assemble_run_blocks(base_run, rounds, campaign=...)` takes the campaign
as a **required** keyword and refuses a round whose licence names another one
(`ExtensionAuthorization.licence_for`), a base segment that is not `B_min` blocks,
and a run planned under another committed seed. `PairedBlock` carries a segment
and a round number but never an authorization, so pooling is the last frame in
which the question can be asked at all.

What this does NOT do: decide *when* to extend. That is the rule's, and the rule
already has a driver — `statistics.SequentialEvaluation` issues the
`BlockRequest` for each block and terminates itself. Drive the round from it (see
`test_the_stopping_rule_replays_to_a_crossing_on_the_pooled_blocks`), do not
extend on your own judgement of whether the answer might still change.

One related change on the far side: `OrderSchedule.check_observed` now takes
`first_index=` so a single round can be order-controlled at its own window
(default `0`, so the reducer's whole-run call is unchanged). It cross-checks each
block's own `block_index` against that window, so the parameter cannot relabel a
run.

### 6.4 A claim id and a candidate id are spelled the same — CLOSED 2026-08-04

`cpu_region_claim._CLAIM_ID_PREFIX` was `"akc-"`, which is the prefix
`api.EvaluationRequest` requires of a **candidate** id. A claim id passed where a
candidate id belongs therefore satisfied the one validator written to catch that
class of mistake, and the record grammar rendered it as `res=akc-…` beside
`candidate=akc-…`.

It is now `"akclaim-"` (the spelling `t0_provider`'s own fixtures already used),
and the two namespaces cannot be re-merged by a later edit:
`_require_disjoint_id_namespaces` runs at IMPORT and refuses any claim prefix
that starts with, or is started by, the candidate prefix — prefix-disjointness in
both directions, because every id validator in the package tests by
`startswith`. `test_cpu_region_claim.TestAClaimIdIsNotACandidateId` resolves a
minted id against the real `api.EvaluationRequest` rather than against the
module's own copy of the prefix, with a real candidate id as the compliant-path
control.

### 6.5 CLOSED 2026-08-05 — a declared round cannot be re-run until it crosses

Found by the 2026-08-04 red team. It was the one remaining way
to manufacture a crossing from a null effect. **Re-reproduced against the
current code on 2026-08-04**, after `ExtensionAuthorization` was moved onto the
campaign and after the reference composition started pooling: run round 1, keep
the run, run round 1 again, pool the *second* run with the base segment —
`assemble_run_blocks` accepts it and returns ten blocks. (Pooling *both* runs
raises `ScheduleMismatch`: *"extension round 1 was submitted twice"*. Discarding
one is what is invisible.) `assemble_run_blocks` refuses the
same round *object* twice, and refuses a round from another plan or another
campaign — but nothing anywhere refuses a **second run of the same declared
round**, because a second run of the same plan is the same plan. Run round 1,
pool, read the e-value, and if it did not cross, run round 1 again and pool that
one instead. Every structural check passes: same seed, same candidate, same
attempt, same instrument, one contiguous index line.

Measured on the calibrated construction (`sign_martingale_predictable_lambda/v1`,
λ cap 0.5, threshold 10, B_min 5, 5 blocks per round, 10 000 trials per row):

| what was submitted | null crossing rate |
|---|---|
| base segment only (5 blocks) | 0.00 % |
| base + one declared round (10 blocks) | 1.36 % |
| best of 2 re-runs of round 1 | 2.63 % |
| best of 5 | 4.71 % |
| best of 10 | 6.33 % |
| best of 25 | 9.76 % |
| best of 50 | 13.12 % |

The declared budget is α = 0.1 (threshold = 1/α = 10). Twenty-five re-runs
exhaust the whole campaign's error budget on one candidate; fifty exceed it.
Nothing in the old package could see it, because detecting a **discarded** completed
round needs a durable per-candidate run ledger, and the runner wrote none:
`MicrobenchRun` is a value, `assemble_run_blocks` sees only the runs it is
handed, and `PairedBlock` carries a segment and a round number but no run
identity at all.

`CompletedRunLedger` now closes it in three places. `MicrobenchRunner.run()`
refuses an extension without the durable ledger, checks the declared key before
the first spawn, and appends the complete raw vector as
`MICROBENCH_RUN_COMPLETED` before returning. The key is exactly
`(campaign_id, candidate_id, attempt, segment, extension_round)` and `run_id` is
the content hash of the raw vector. `assemble_run_blocks()` requires the same
ledger whenever an extension is present and refuses an unjournaled, substituted,
or multiply-completed run. A run that must be repeated is therefore `attempt + 1`
with the retry schedule, not a re-roll of the observed attempt. The stock
executing campaign path also refuses a missing `--journal-root` during preflight,
before it acquires a claim or launches T0 inference.

### 6.6 Smaller, recorded rather than fixed

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
* `OrderSchedule.order_for` folds the index modulo `base_blocks`, so with
  `blocks_per_round == B_min` **round 2 draws round 1's orders exactly** (both
  are the reversed limb of base slots 0..B_min-1). It is deterministic,
  seed-derived and prefix-stable, and it does not affect the sign statistic — but
  it means rounds ≥ 2 are not independent order draws, and the run's order
  balance drifts by the base draw's own imbalance once per round (with the chain
  campaign's seed: 5/5 at 10 blocks, 7/8 at 15, 9/11 at 20). Harmless at
  `max_rounds = 1`, which is what the CPU decode cell declares. A campaign that
  declares more rounds should key the reversal on the ROUND, the way `derive()`
  now keys it on the attempt's parity.
* `api.EffectReducer.reduce_blocks` — the ratified seam — takes no `attempt`, so
  it always reduces at attempt 0. `PairedBlockReducer.reduce` does take one.
  Since `derive()` now reverses on the attempt's parity, a **retried** run must
  be reduced through `reduce()` with its attempt, or its order control fails.
  That is the correct direction (it fails closed) but the api seam cannot express
  it; thread the attempt at the call site.

### 6.7 CLOSED 2026-08-04 — the published MDE described a window the rule cannot license

Found while wiring §6.3's extension round into the reference composition, and it
is **the defect that closing §6.3 made live**. Latent before, because every run
in the package stopped at `B_min` and the two quantities below happened to be
the same number.

`statistics.solve_mde`'s `block_count` argument is the **base segment length**:
it replays the stopping rule with `b_min=block_count` and draws its resampling
windows at `rule.max_total_blocks(block_count)`, which ADDS
`max_rounds * blocks_per_round` on top. `PairedBlockReducer.reduce` handed it
`len(blocks)` — the **realized** count. For a pooled ten-block run that asks for
the MDE of a fifteen-block window, and `max_rounds = 1` means this campaign can
never license fifteen blocks for one candidate:

| passed as `block_count` | window built | selection MDE | confirmation MDE |
|---|---|---|---|
| `b_min = 5` (correct) | 10 — the rule's own budget | 0.008584 | 0.013294 |
| `len(blocks) = 10` (was) | 15 — **unlicensable** | 0.006972 | 0.007511 |
| | | **-18.8 %** | **-43.5 %** |

The direction overstates, and it reaches the verdict: `api._resolve_effect`
reads `magnitude < mde` as `no_detectable_difference`, so an understated MDE
admits effects the run could not resolve, and if the e-value crosses they are
RANKED as improvements. The record's MDE is not decorative — *"a record without
a published MDE is INVALID"* — so this is a headline number on every banked win.

**Fixed** in `PairedBlockReducer.reduce`: the MDE comes from `campaign.b_min`,
which is the only argument for which `solve_mde` builds a window the rule can
actually license. The over-extension detection that used to arrive here as *"no
MDE could be derived"* — a bookkeeping message for a rule violation — is now a
named check, **`mde_window`**, which bounds the realized count by
`b_min + max_rounds * blocks_per_round`. That bound is TIGHTER than
`block_count`'s `max_blocks_per_candidate` (10 vs 20 on this campaign), so it
also catches a run between the two, which nothing caught before. Cover:
`test_statistics.TestOverExtensionIsJournaledNotRaised` (three tests, including
the between-the-ceilings bite and a compliant-path control) and
`test_execution_chain.TestAWinIsReachableAndANullIsRefused.test_the_pooled_records_MDE_describes_a_window_the_RULE_can_license`.

**Residual, OPEN and lower severity:** `solve_mde` still cannot express *"the MDE
of a run that realized N blocks and stopped"* — it always adds the extension
budget. The design MDE at `b_min` is the right number to publish (it is the
campaign's declared power, fixed before any candidate ran, which is what makes
*"published WITH the result, not after seeing it"* checkable), but a reader who
takes it as this run's own resolution is reading it slightly generously.
Measured on this campaign: design MDE at `b_min` 0.008584 against roughly 0.0098
for a ten-block window with a look at every block. Closing it means a
`realized_blocks=` parameter on `solve_mde`, which is a ratified seam and wants
its own pass.

### 6.8 OPEN as a PLANNING decision — how little block noise the budget tolerates

Not a defect, and not fixable in code: it is arithmetic about the rule you are
about to declare, and nothing in the package will tell you before you spend the
claim. Enumerated over all 1024 sign sequences of a ten-block run under
`sign_martingale_predictable_lambda/v1` (the e-value is a deterministic function
of the sign sequence, so this is exhaustive, not sampled):

| blocks against the candidate | sequences | cross at 10 (selection) | cross at 20 (confirmation) |
|---|---|---|---|
| 0 | 1 | 1 | 1 |
| 1 | 10 | 9 | 1 |
| 2 | 45 | 3 | 0 |
| 3 | 120 | 1 | 0 |
| ≥ 4 | 848 | 0 | 0 |
| **total** | **1024** | **14 (1.37 %)** | **2 (0.20 %)** |

Read the last column first. **At the declared budget, a confirmation-stratum
candidate crosses on exactly two sequences out of 1024: ten of ten same-sign, or
nine of ten with the single adverse block LAST.** One adverse block anywhere
else sinks it. The selection stratum is looser but not loose: every placement of
even one adverse block is guaranteed to cross only at 15 blocks, and of three
only at 20.

| budget | selection (thr 10) | confirmation (thr 20) |
|---|---|---|
| 10 blocks (`max_rounds = 1`) | 0 adverse blocks tolerated in every placement | 0 |
| 15 blocks (`max_rounds = 2`) | 2 | 1 |
| 20 blocks (`max_rounds = 3`) | 3 | 3 |

The bottom row of the first table is also the null crossing rate: **1.37 %** at
selection, which is the exact value §6.5's Monte-Carlo table estimates at 1.36 %,
and 0.20 % at confirmation. Both are far under the declared α of 0.1 — the
instrument is honest; it is simply *tight*.

**The decision, and it is the operator's, before the claim is acquired:** a
`max_rounds = 1` campaign can bank a SELECTION win from a candidate whose effect
is large relative to per-block noise, and will struggle to CONFIRM it, because
confirmation demands a near-perfect run at half the α. If the campaign intends to
reach `compose_into_champion_lineage` — which needs
`confirmation_admission_count = 2` replications — declare `max_rounds = 2` and
budget 15 blocks per candidate. The cost is 50 % more claim time per candidate;
the alternative is a selection win that cannot be confirmed, banked, and
mistaken for a candidate that failed to replicate.

*(The MDE agrees and says it more gently: the solver finds a confirmation MDE at
`b_min` of 0.0133 versus a selection MDE of 0.0086 — the confirmation stratum
needs an effect ~55 % larger at the same budget.)*

---

## 7. Where to look when something does not fit

`execution/chain.py` holds the four seams and argues each one in its docstring.
`execution/test_execution_chain.py::ChainLeg` is the reference composition —
claim, worktree, build, artifact, anchor, T0, T1, **extend**, reduce, controls,
dispatch, bank, teardown — and every negative path below it names the thing it
refuses. Between them they are the documentation that is checked by the test
suite, which the file you are reading is not.

Two classes are worth reading before anything else:

* `TestAWinIsReachableAndANullIsRefused` — the end-to-end proof that a real
  effect banks and a null does not, with every number reproducible from the
  fixtures in that file and none of them a measurement of any kernel;
* `TestTheExtensionRoundHasAProducer` — the refusals around the extension: an
  undeclared round, a round licensed by another campaign, a re-derived schedule.
