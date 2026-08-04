#!/usr/bin/env python3
"""test_execution_chain.py — does the execution layer COMPOSE? No inference, no build.

WHAT THIS FILE IS
-----------------
Five agents built five executors in parallel against one evaluator. Each has its
own suite and each passes it. This file asks the only question none of those
suites can: when the output of one is handed to the next, does it fit — and when
it does not, does the composition refuse rather than produce a number?

The walk is the real one, in order:

    claim acquired (a REAL flock, in a temp lock root)
      -> worktree created from a production branch tip
      -> candidate "built" (a RECORDED cmake+make log; nothing is compiled)
      -> BuildIdentity -> chain.build_evidence -> correctness.BuildProvenance
      -> artifact digests MEASURED from disk (not copied off the receipt)
      -> T0 evidence collected through RecordedProcessRunner, anchor triple bound
      -> evaluate_t0 -> seventeen gates
      -> T1 paired blocks from RECORDED llama-bench JSON -> the reducer
      -> controls scored -> api.ControlPanel
      -> TierDispatcher -> a Verdict
      -> the controller walks BUILD -> T0_GATE -> T1_SEARCH_EVAL -> ... -> BANK_EVENT
      -> claim released, worktree torn down
      -> THE PRODUCTION TREES ARE BYTE-IDENTICAL TO BEFORE

WHAT IT DOES **NOT** PROVE — read this before trusting a green run
------------------------------------------------------------------
Nothing here compiles anything, runs a kernel, or takes a timing. Specifically
unproven, and each one is a real risk for tomorrow's first campaign:

1. **That a real llama.cpp build parses.** `parse_build_log` is exercised against
   recorded cmake+make output from this host, but a 13 GiB tree with ROCm,
   HIP and 2000 compile units may print lines these fixtures do not contain.
2. **That the recorded `llama-bench` argv is what the real binary accepts.** The
   argv is asserted against `recipes.CANONICAL_PREFIX`; the binary has never been
   handed it by this code.
3. **Scale.** `integrity.hash_source_tree` over the worktree here walks a
   two-file repository. Over `llama.cpp` it walks ~13 GiB, once per candidate,
   and `measure_artifact_identity` calls it.
4. **Any number.** Every effect in this file comes from recorded fixtures scaled
   by a stated factor. No claim about any kernel is derivable from a green run
   of this file, and none is made.
5. **Contention.** `HostStatePolicy` is exercised against a stubbed `/proc` and
   `/sys`. Whether the real host's throttle and load thresholds fire correctly
   is a fact about tomorrow's host state.

The frozen production trees ARE touched here, in exactly one way: they are read
(`git rev-parse`, `git status --porcelain`) before and after the whole walk and
the two readings must be byte-identical. Nothing else in this file addresses
them, and `TestFrozenTreesAreUntouched` fails if that ever stops being true.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[2]))

from autokernel import schemas                                          # noqa: E402
from autokernel.evaluator import (api, correctness, integrity,          # noqa: E402
                                  recipes, statistics)
from autokernel.evaluator import controls as CT                         # noqa: E402
from autokernel.evaluator import controls as controls_module            # noqa: E402
from autokernel import journal                                          # noqa: E402
from autokernel.controller import state_machine as SM                   # noqa: E402
from autokernel.execution import control_runner as CR                   # noqa: E402
from autokernel.execution import chain                                  # noqa: E402
from autokernel.execution import cpu_region_claim as CRC                # noqa: E402
from autokernel.execution import microbench as MB                       # noqa: E402
from autokernel.execution import t0_provider as T0                      # noqa: E402
from autokernel.execution import worktree as WT                         # noqa: E402

TESTDATA = _HERE.parent / "testdata"
MARKER = "---8<--- verbatim below this line ---8<---"

#: Where the chain's temporary worlds live. `/tmp` is a 120 GiB root SSD on this
#: host and the project rule is that files live on raid0.
SCRATCH_ROOT = "/mnt/raid0/llm/.scratch"

CAMPAIGN = "ak-chain-0001"
CANDIDATE = "akc-chain-0001"


def recorded(name: str) -> str:
    text = (TESTDATA / name).read_text(encoding="utf-8")
    if MARKER not in text:
        raise AssertionError(f"{name} carries no provenance marker; it may not be a capture")
    return text.split(MARKER, 1)[1].lstrip("\n")


def raw(name: str) -> str:
    """A recorded log with no provenance marker (the build logs are bare captures)."""
    return (TESTDATA / name).read_text(encoding="utf-8")


def clean_configure_log() -> str:
    """The recorded configure log, DERIVED into what a clean-build configure prints.

    Two edits, both named, both to lines this test's assertions depend on:

      * the `-- ccache found, compilation results will be cached` line is
        REMOVED. `BuildPlan` forces `-DGGML_CCACHE=OFF` (upstream
        `ggml/CMakeLists.txt` defaults it ON), so a compliant configure does not
        print it, and no recorded configure on this host was taken with it off.
      * `-- ggml commit: 2fdb4f97d-dirty` loses its `-dirty` suffix. The
        recorded configure was taken in a tree with uncommitted changes; a
        campaign worktree checked out at a commit has none.

    Labelled rather than presented as a capture, and the untransformed original
    is used for the negative-path tests, where those two lines are the point.
    """
    lines = []
    for line in raw("recorded_configure_ccache.log").splitlines(keepends=True):
        if line.startswith("-- ccache found"):
            continue
        lines.append(line.replace("2fdb4f97d-dirty", "2fdb4f97d"))
    return "".join(lines)


#: A `GGML_SCHED_DEBUG=2` trace with two nodes on the CPU backend. Shaped to
#: `t0_provider._SPLIT_RE`/`_NODE_RE`, which are read off the real ggml printer.
SCHED_TRACE = (
    "## SPLIT #0: CPU # 2 inputs\n"
    "node #  0 (       MUL_MAT):        ffn_up-0 (  f32) [ CPU        assigned ]\n"
    "node #  1 (    MUL_MAT_ID):     ffn_moe_up-0 (  f32) [ CPU        assigned ]\n"
)


def _disposition(argv, exit_code: int) -> WT.ProcessDisposition:
    """A disposition for a process that was NEVER LAUNCHED, and it says so.

    `pid=0`/`pgid=0` are not plausible pids: they are the sentinel that makes a
    reader of a chain-test artifact see immediately that no child existed. A
    real `run_build` fills these from the child it owned.
    """
    return WT.ProcessDisposition(
        argv=tuple(argv), pid=0, pgid=0, exit_code=exit_code, timed_out=False,
        signals_sent=(), verified_dead=True, duration_s=1.0,
        started_at="2026-08-03T23:00:00Z")


# =============================================================================
# The world: a stand-in production clone, a real claim, a real worktree
# =============================================================================

class ChainWorld:
    """Everything one campaign leg needs, built in a temp directory.

    The production clone here is a STAND-IN — a two-file git repository on a
    branch called `production-consolidated-v8`. It is not `/mnt/raid0/llm/llama.cpp`
    and this file never builds a worktree off the real one: a `git worktree add`
    against the real clone writes `.git/worktrees/<name>/` in a 13 GiB tree
    shared with production, and a unit test has no business doing that on a host
    where six other sessions are working.

    What the real trees get instead is the check that matters: `TestFrozenTrees
    AreUntouched` fingerprints all three before and after, and the walk below
    must not move them.
    """

    def __init__(self, root: str, *, tip_files: dict | None = None) -> None:
        self.root = root
        self.src = os.path.join(root, "llama.cpp")
        os.makedirs(self.src)
        self._git("init", "-q", "-b", "production-consolidated-v8")
        self._git("config", "user.email", "chain@test")
        self._git("config", "user.name", "chain")
        for name, body in (tip_files or {"ggml.c": "int mul_mat(void){return 0;}\n"}).items():
            (Path(self.src) / name).write_text(body, encoding="utf-8")
            self._git("add", name)
        self._git("commit", "-qm", "production tip")
        self.tip = self._git("rev-parse", "HEAD").strip()

        self.lock_root = os.path.join(root, "locks")
        self.journal = CRC.RegionClaimJournal(os.path.join(root, "claims.jsonl"))
        self.claim = None
        self.worktree = None
        self.worktree_proof = None

    def _git(self, *args: str) -> str:
        return subprocess.run(("git",) + args, cwd=self.src, check=True,
                              capture_output=True, text=True).stdout

    # -- stage 1: the claim ------------------------------------------------
    def acquire(self, cpu_list: str = "0-95"):
        self.claim = CRC.acquire_cpu_region_claim(
            cpu_list, role="autokernel", purpose="execution chain test",
            campaign_id=CAMPAIGN, journal=self.journal, lock_root=self.lock_root,
            timeout_s=5.0)
        return self.claim

    # -- stage 2: the worktree ---------------------------------------------
    def make_worktree(self, campaign_id: str = CAMPAIGN):
        anchor = WT.resolve_anchor(WT.GitRepo(self.src), "production-consolidated-v8")
        wt_root = os.path.join(self.root, "worktrees")
        os.makedirs(wt_root, exist_ok=True)
        self.worktree, self.worktree_proof = WT.create_campaign_worktree(
            anchor, campaign_id, root=wt_root)
        return self.worktree

    # -- stage 3: the "build" ----------------------------------------------
    def build_dir(self, name: str = "build-t0") -> WT.SandboxPath:
        path = os.path.join(self.root, "builds", name)
        return WT.SandboxPath.in_sandbox(path, sandbox_root=os.path.join(self.root, "builds"),
                                         label="build dir")

    def build_plan(self, **overrides) -> WT.BuildPlan:
        kwargs = dict(
            source_root=self.worktree.path,
            build_dir=self.build_dir(),
            actor_worktree=self.worktree.path,
            parallelism=WT.BuildParallelism(jobs=32),
            targets=("llama-cli", "test-backend-ops"),
            cmake="/usr/bin/cmake",
        )
        kwargs.update(overrides)
        return WT.BuildPlan(**kwargs)

    def recorded_build_result(self, plan: WT.BuildPlan, *, exit_code: int = 0,
                              log_text: str | None = None) -> WT.BuildResult:
        """A `BuildResult` assembled from a RECORDED log. Nothing is compiled.

        This is where "the build ran" enters the chain, and it is deliberately
        the one stage with no live process: a 96-way cmake build on a host at
        load 67 is both bad data and theft from whoever is measuring. The log is
        the concatenation `run_build` itself writes — a configure section and a
        build section — so `parse_build_log` sees the same text it would see
        from a real run.
        """
        if log_text is None:
            log_text = ("=== configure: " + " ".join(plan.configure_argv()) + "\n"
                        + raw("recorded_configure_ccache.log")
                        + "=== build: " + " ".join(plan.build_argv()) + "\n"
                        + raw("recorded_build_success.log"))
        os.makedirs(plan.build_dir.path, exist_ok=True)
        pre = integrity.hash_source_tree(plan.build_dir.path).sha256
        log_path = os.path.join(self.root, "build.log")
        Path(log_path).write_text(log_text, encoding="utf-8")
        disp = _disposition(plan.build_argv(), exit_code)
        conf = _disposition(plan.configure_argv(), 0)
        return WT.BuildResult(
            plan=plan, configure=conf, build=disp, log_path=log_path,
            log_sha256=WT._sha256_text(log_text), facts=WT.parse_build_log(log_text),
            build_dir_pre_build_digest=pre, build_dir_created_for_this_build=True,
            load_average_at_start=None)

    def write_artifacts(self, plan: WT.BuildPlan) -> dict:
        """The files a real build would have written. Bytes, not ELF."""
        bin_dir = os.path.join(plan.build_dir.path, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        paths = {}
        for name, body in (("llama-cli", b"\x7fELF chain-candidate llama-cli\n"),
                           ("llama-bench", b"\x7fELF chain-candidate llama-bench\n"),
                           ("test-backend-ops", b"\x7fELF chain-candidate tbo\n"),
                           ("libggml.so.0", b"\x7fELF chain-candidate libggml\n"),
                           ("libggml-base.so.0", b"\x7fELF chain-candidate libggml-base\n")):
            path = os.path.join(bin_dir, name)
            Path(path).write_bytes(body)
            paths[name] = path
        return paths

    def anchor_tree(self) -> dict:
        """The ANCHOR build's binaries — a separate tree, read-only in a real leg.

        On the real host this is `production-consolidated-v8`'s own build
        directory, which `recipes` permits the anchor arm to name and permits no
        candidate arm to. Here it is a sibling directory with real bytes so the
        digests the runner takes are real digests.
        """
        root = os.path.join(self.root, "anchor")
        bin_dir = os.path.join(root, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        out = {"root": root, "bin": bin_dir}
        for name, body in (("llama-cli", b"\x7fELF anchor-v8 llama-cli\n"),
                           ("llama-bench", b"\x7fELF anchor-v8 llama-bench\n"),
                           ("libggml.so.0", b"\x7fELF anchor-v8 libggml\n")):
            path = os.path.join(bin_dir, name)
            Path(path).write_bytes(body)
            out[name] = path
        return out

    def linkage_report_text(self, *, binary: str, bin_dir: str, libraries) -> str:
        """A `verify_ggml_linkage.sh` report over paths that really exist here.

        Same grammar as `testdata/recorded_t0_linkage_pass.txt`, re-pointed at
        this world's build directory so `collect_linkage` can hash every row
        instead of being monkeypatched out.
        """
        rows = "".join(
            f"  OK   {name:<28} -> {path}\n" for name, path in sorted(libraries.items()))
        return (f"binary : {binary}\n"
                f"expect : libraries under {bin_dir}\n\n"
                f"{rows}\n"
                "LD_LIBRARY_PATH order as the loader sees it:\n"
                f"         1\t{bin_dir}\n\n"
                f"PASS: all linked ggml libraries resolve inside {bin_dir}\n")

    # -- teardown ----------------------------------------------------------
    def release(self):
        if self.claim is not None:
            self.claim.release()
            self.claim = None


class _ChainCase(unittest.TestCase):
    """One temp world per test. Everything it creates is removed on exit."""

    def setUp(self) -> None:
        os.makedirs(SCRATCH_ROOT, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(prefix="ak-chain-", dir=SCRATCH_ROOT)
        self.addCleanup(self._tmp.cleanup)
        self.world = ChainWorld(os.path.join(self._tmp.name, "world"))
        self.addCleanup(self.world.release)


# =============================================================================
# A. The frozen trees, before and after everything
# =============================================================================

def fingerprint_frozen() -> dict:
    out = {}
    for tree in ("/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/whisper.cpp",
                 "/mnt/raid0/llm/qwentts.cpp"):
        if not os.path.isdir(os.path.join(tree, ".git")):
            out[tree] = None
            continue
        def g(*args):
            return subprocess.run(("git", "-C", tree) + args, capture_output=True,
                                  text=True, check=True).stdout
        out[tree] = {"head": g("rev-parse", "HEAD").strip(),
                     "branch": g("rev-parse", "--abbrev-ref", "HEAD").strip(),
                     "status": g("status", "--porcelain")}
    return out



# =============================================================================
# B. The leg — one candidate, all the way through, composed from the real parts
# =============================================================================

CANONICAL_CPU_LIST = list(recipes.CANONICAL_PREFIX)[
    list(recipes.CANONICAL_PREFIX).index("-c") + 1]


class ChainLeg:
    """One campaign leg, composed. Every stage stores its output for assertion.

    This is deliberately NOT a reusable campaign driver. `controller/state_machine.py`
    owns the loop and a second walk here would give the loop two spellings. It is
    the shortest composition that touches every seam, written so a test can reach
    into the middle of it.
    """

    def __init__(self, world: ChainWorld, *, anchor_source_commit=None,
                 build_exit_code: int = 0, claim: str = "acquire",
                 configure_log=None, build_dir=None) -> None:
        self.world = world
        self.anchor_source_commit = anchor_source_commit
        self.build_exit_code = build_exit_code
        self._claim_mode = claim
        self._configure_log = configure_log
        self._build_dir = build_dir
        self.claim = None
        self.claim_binding = None

    # -- 1. claim ----------------------------------------------------------
    def acquire_claim(self):
        if self._claim_mode == "none":
            self.claim = None
            return None
        self.claim = self.world.acquire(CANONICAL_CPU_LIST)
        self.claim_binding = chain.bind_claim(self.claim, cpu_list=CANONICAL_CPU_LIST)
        self.claim_receipt_at_open = self.claim.receipt().to_dict()
        self.claim_footprint_check = CRC.check_precondition_1(
            self.claim.receipt(), CANONICAL_CPU_LIST, lock_root=self.world.lock_root)
        return self.claim

    # -- 2. worktree -------------------------------------------------------
    def make_worktree(self):
        self.worktree = self.world.make_worktree()
        return self.worktree

    # -- 3. build (recorded) ----------------------------------------------
    def build(self):
        self.plan = (self.world.build_plan() if self._build_dir is None
                     else self.world.build_plan(build_dir=self._build_dir))
        configure = self._configure_log if self._configure_log is not None \
            else clean_configure_log()
        log_text = ("=== configure: " + " ".join(self.plan.configure_argv()) + "\n"
                    + configure
                    + "=== build: " + " ".join(self.plan.build_argv()) + "\n"
                    + raw("recorded_build_success.log"))
        self.result = self.world.recorded_build_result(
            self.plan, exit_code=self.build_exit_code, log_text=log_text)
        self.artifacts = self.world.write_artifacts(self.plan)
        self.snapshot = integrity.hash_source_tree(
            self.worktree.path.path, exclude_dir_names=(".git",))
        self.identity = WT.build_identity(
            self.result, candidate_id=CANDIDATE, campaign_id=CAMPAIGN,
            worktree=self.worktree, snapshot=self.snapshot,
            output_binary=self.artifacts["llama-cli"],
            toolchain="cmake 3.31 + GNU make",
            libraries={"libggml.so.0": self.artifacts["libggml.so.0"],
                       "libggml-base.so.0": self.artifacts["libggml-base.so.0"]})
        # SEAM 1.
        self.build_evidence = chain.build_evidence(self.identity)
        return self.identity

    # -- 4. the request's artifact identity, MEASURED ----------------------
    def measure_artifact(self):
        bin_dir = os.path.dirname(self.artifacts["llama-cli"])
        self.linkage_text = self.world.linkage_report_text(
            binary=self.artifacts["test-backend-ops"], bin_dir=bin_dir,
            libraries={k: v for k, v in self.artifacts.items() if k.startswith("libggml")})
        report = T0.parse_linkage_report(self.linkage_text)
        self.linkage_sha256 = T0.ExecutedT0EvidenceProvider.linkage_digest(report)
        # SEAM 2 — from disk, not from the receipt.
        self.artifact = chain.measure_artifact_identity(
            source_root=self.worktree.path.path,
            binary=self.artifacts["llama-cli"],
            linkage_sha256=self.linkage_sha256)
        return self.artifact

    # -- 5. the anchor, bound once -----------------------------------------
    def bind_anchor(self):
        commit = self.anchor_source_commit or ANCHOR_COMMIT
        self.anchor_binding = chain.bind_anchor(T0.AnchorCapture(
            source_commit=commit,
            binary_sha256=T0.sha256_text("anchor llama-cli bytes"),
            linkage_sha256=T0.sha256_text("anchor resolved library table"),
            output_digests=(T0.sha256_text("Paris."),), output_lengths=(6,),
            determinism_class="bitwise_stable", delivered_units=32,
            oracle_ids=("oracle://anchor-v8",)), tool="llama-cli")
        return self.anchor_binding

    # -- 6. T0 -------------------------------------------------------------
    def t0_plan(self):
        # SEAM 3 — the plan's paths come off the receipt, not off a literal.
        candidate = chain.candidate_build_for(
            self.identity, test_backend_ops=self.artifacts["test-backend-ops"])
        return T0.T0ExecutionPlan(
            candidate=candidate,
            tools=T0.ToolPaths(
                bash="/bin/bash",
                verify_ggml_linkage_sh=str(_HERE.parents[4] / "scripts" / "utils"
                                           / "verify_ggml_linkage.sh"),
                cmake="/usr/bin/cmake"),
            op_suite=T0.OpSuitePlan(
                backend_filter="CPU", ops=("MUL_MAT", "MUL_MAT_ID"),
                suite_id="test-backend-ops/v1",
                suite_source_sha256=self.identity.snapshot_sha256),
            dispatch=T0.DispatchTracePlan(derived_surface=("MUL_MAT", "MUL_MAT_ID")),
            generation=T0.GenerationPlan(prompt="The capital of France is",
                                         prompt_ref="ak-prompt-001", n_predict=32, seed=42),
            determinism_runs=2, cache_state="cold", state_safety_probe=False,
            oracle_ids=("oracle://anchor-v8",),
            build=self.build_evidence.provenance,
        )

    def _t0_runner(self, plan, *, op_suite_text):
        gen = T0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env, seed=plan.generation.seed)
        trace = T0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env,
            extra_env={"GGML_SCHED_DEBUG": "2"})
        ops = T0.build_backend_ops_invocation(
            binary=plan.candidate.test_backend_ops,
            library_path=plan.candidate.library_path,
            backend_filter=plan.op_suite.backend_filter, ops=plan.op_suite.ops,
            base_env=plan.base_env)
        link = T0.build_linkage_invocation(
            bash=plan.tools.bash, script=plan.tools.verify_ggml_linkage_sh,
            binary=plan.candidate.binary, expected_root=plan.candidate.library_path,
            library_path=plan.candidate.library_path, base_env=plan.base_env)
        perf = ("llama_perf_context_print:        eval time =   1234.56 ms /    32 runs   "
                "(   38.58 ms per token,    25.92 tokens per second)\n")

        def cap(argv, stdout="", stderr="", exit_code=0):
            return T0.CompletedProcess(
                argv=tuple(argv), env=(), cwd=plan.candidate.worktree,
                exit_code=exit_code, stdout=stdout, stderr=stderr, duration_s=0.5,
                timed_out=False, signalled=False, orphans=())

        return T0.RecordedProcessRunner([
            cap(ops.argv, stdout=op_suite_text),
            cap(trace.argv, stdout=SCHED_TRACE + "Paris.", stderr=perf),
            cap(link.argv, stdout=self.linkage_text.replace(
                plan.candidate.test_backend_ops, plan.candidate.binary)),
            cap(gen.argv, stdout="Paris.", stderr=perf),
        ])

    def evaluate_t0(self, *, op_suite_text=None):
        plan = self.t0_plan()
        self.provider = T0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._t0_runner(plan, op_suite_text=op_suite_text or _OPS_OK),
            claim=None if self.claim is None else self.claim_binding.t0_claim,
            anchor_capture=self.anchor_binding.capture)
        self.request = self.evaluation_request()
        self.t0_report = correctness.T0CorrectnessRunner(
            provider=self.provider, policy=_t0_policy()).evaluate(self.request)
        return self.t0_report

    def evaluation_request(self, **overrides):
        kwargs = dict(
            event_id="ake-chain-0001", campaign_id=CAMPAIGN, candidate_id=CANDIDATE,
            tier="T0", backend="llama_cpu", phase="decode",
            cell_class="operator_microbench", protocol_id=api.PROTOCOL_VERSIONED_ID,
            artifact=self.artifact,
            anchor=self.anchor_binding.identity,
            evaluator=api.EvaluatorIdentity(
                id="ak-eval/v1", bundle_sha256=self.identity.snapshot_sha256,
                runtime_source_label_ref="ref://autokernel/execution/chain"),
            scope_denominator=api.ScopeDenominator(
                machine_subset="full", numa_nodes=(), devices=(), cores=96),
            scope_manifest_sha256=self.identity.snapshot_sha256, co_residency="single",
            determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                              same_seed_repeat_runs=2),
            metric="tokens_per_second", metric_direction="higher_better", reps=10,
            created_at="2026-08-03T23:00:00Z",
            campaign_controls=ChainCampaign.get()[0],
            calibration=ChainCampaign.get()[4])
        kwargs.update(overrides)
        return api.EvaluationRequest(**kwargs)

    # -- 7. T1 — the SAME claim, through the other Protocol -----------------
    def run_t1(self, *, factor: float = 1.08, blocks: int = 5):
        """Paired blocks from recorded `llama-bench` JSON, under the bound claim.

        SEAM 4 in anger: `MicrobenchRunner` calls `claim.attest()` before every
        spawn, `CpuRegionClaim` has no `attest`, and the adapter
        `chain.bind_claim` produced is what makes the two fit. Nothing is
        spawned — `RecordedSpawner` replays — but the claim is a REAL flock and
        the attestation really goes back to the filesystem.
        """
        anchor_tree = self.world.anchor_tree()
        self.candidate_binding = bench_binding(
            os.path.join(self.world.root, "t1", MB.ARM_CANDIDATE), b"chain-candidate-bench-bytes")
        self.anchor_binding_tool = bench_binding(
            os.path.join(self.world.root, "t1", MB.ARM_ANCHOR), b"chain-anchor-bench-bytes")
        assert anchor_tree["root"]

        # SEAM 3: a SECOND anchor binding, for the OTHER tool, tied to the first
        # by commit and linkage. One `api.AnchorIdentity` cannot name two files.
        self.t1_anchor = chain.bind_anchor(T0.AnchorCapture(
            source_commit=self.anchor_binding.capture.source_commit,
            binary_sha256=integrity.sha256_file(self.anchor_binding_tool.binary),
            linkage_sha256=self.anchor_binding.capture.linkage_sha256,
        ), tool="llama-bench")
        # A T1 record compares RATES, so the anchor must name the measurement
        # events its side of the comparison came from — `schemas` refuses a T1
        # anchor block with an empty `measurement_event_ids`. T0 compares
        # artifacts and does not.
        self.t1_anchor_identity = api.AnchorIdentity(
            source_commit=self.t1_anchor.identity.source_commit,
            binary_sha256=self.t1_anchor.identity.binary_sha256,
            linkage_sha256=self.t1_anchor.identity.linkage_sha256,
            measurement_event_ids=("ake-chain-anchor-0001",))

        plan = MB.MicrobenchPlan(
            recipe_id=BENCH_RECIPE_ID, candidate_id=CANDIDATE,
            campaign_seed=CAMPAIGN_SEED,
            candidate_binding=self.candidate_binding,
            anchor_binding=self.anchor_binding_tool,
            anchor=self.t1_anchor_identity,
            params={"model": FIXTURE_MODEL, "n_gen": 128, "reps": 10,
                    "output_format": "json"},
            base_blocks=blocks, pairs_per_block=1, unit_ids=("chain-unit-0",))
        spawner = MB.RecordedSpawner({
            MB.ARM_CANDIDATE: scaled_bench(factor=factor, build_commit="cafe12345"),
            MB.ARM_ANCHOR: BENCH_FIXTURE.read_text(encoding="utf-8"),
        })
        self.microbench_spawner = spawner
        runner = MB.MicrobenchRunner(
            claim=self.claim_binding.microbench_claim, policy=HEALTHY_POLICY,
            spawner=spawner, host_state=HostStateStub(healthy_host_state()))
        self.t1_run = runner.run(plan)
        return self.t1_run

    # -- 8. reduce ----------------------------------------------------------
    def reduce(self):
        stats = ChainCampaign.get()[5]
        self.reducer = statistics.PairedBlockReducer(stats)
        self.t1_request = self.evaluation_request(
            tier="T1", event_id="ake-chain-0002",
            anchor=self.t1_anchor_identity,
            metric="tokens_per_second", metric_direction="higher_better")
        self.reduction = self.reducer.reduce(
            self.t1_request, self.t1_run.paired_blocks(),
            raw_samples_ref=f"ak-raw://{CAMPAIGN}/{CANDIDATE}/t1")
        return self.reduction

    # -- 9. controls, window, verdict ---------------------------------------
    def score_controls(self):
        self.controls = ControlStack()
        provisional = api.ControlPanel(
            positive=schemas.Check(schemas.PASS), neutral=schemas.Check(schemas.PASS),
            degraded_negative=schemas.Check(schemas.PASS), aa=schemas.Check(schemas.PASS),
            historical_replay=schemas.Check(schemas.PASS))
        # The window the CONTROLS run under carries the same real claim receipt as
        # the candidate's. `ExecutedControlRunner.open_window` refuses a window
        # whose `resource_claim_open` is not PASS — denial 8 enforced where
        # refusing is still free.
        self.control_window = window_attestations(
            claim_receipt=self.claim_binding.claim_id, controls=provisional,
            anchor=self.controls.anchor)
        self.sweep_result = self.controls.run(self.control_window)
        return self.sweep_result

    def dispatch(self):
        """The candidate's own T1 dispatch, with the MEASURED panel in the window."""
        panel = self.sweep_result.panel_result.panel
        if panel is None:
            raise AssertionError("the control sweep produced no panel; nothing may rank")
        attestations = controls_module.window_control_attestations(
            self.sweep_result.panel_result)
        self.window = window_attestations(
            claim_receipt=self.claim_binding.claim_id, controls=panel,
            anchor=self.t1_anchor_identity,
            control_definitions_immutable=attestations["control_definitions_immutable"])
        dispatcher = api.TierDispatcher(gate_runners={
            "T0": correctness.T0CorrectnessRunner(provider=self.provider,
                                                  policy=_t0_policy()),
            "T1": _AllPassGateRunner("T1"),
        })
        self.outcome = dispatcher.dispatch(self.t1_request, self.window,
                                           effect=self.reduction.estimate)
        return self.outcome

    # -- 10. the controller banks or abandons -------------------------------
    def bank(self, root: str):
        """Walk `BUILD -> T0_GATE -> T1_SEARCH_EVAL -> POST_RUN_CRITIC -> BANK_EVENT`.

        The controller owns the walk; this only proves that the verdict the
        execution layer produced is something the machine will accept at
        `BANK_EVENT`, and that a T0 failure takes the OTHER documented edge
        (`T0_GATE -> POST_RUN_CRITIC`) — *"compilation failures are valuable
        outcomes"*, so a candidate that fails T0 is still banked, as a failure.
        """
        journal_ = journal.Journal(os.path.join(root, "journal"))
        journal_.initialize()
        machine = SM.ControllerStateMachine(
            journal_=journal_, root=os.path.join(root, "controller"),
            campaign_id=CAMPAIGN)
        # A FOURTH anchor shape. `controller.state_machine.AnchorIdentity` keys
        # its digests BY BACKEND — which is the per-key table `api.AnchorIdentity`
        # does not have and whose absence is `chain.SEAM_NOTES` item 2. Building
        # it from the same capture is the only thing that keeps the controller's
        # anchor and the record's anchor one anchor.
        machine.bootstrap(anchor=SM.AnchorIdentity(
            source_tree="llama.cpp", branch="production-consolidated-v8",
            commit=self.t1_anchor.capture.source_commit,
            binary_sha256={"llama_cpu": self.t1_anchor.capture.binary_sha256},
            linkage_sha256={"llama_cpu": self.t1_anchor.capture.linkage_sha256}),
            views=journal.rebuild_views(journal_.read_all()))
        for state, trigger, reason in (
                (SM.SELECT_TARGET, "discover_complete", "decode_b1 selected"),
                (SM.PROPOSE, "target_selected", "one mechanism proposed"),
                (SM.PRE_RUN_CRITIC, "proposal_drafted", "critic admitted the proposal"),
                (SM.MUTATE, "critic_admitted", "the mutation was applied in the worktree"),
                (SM.BUILD, "mutation_applied",
                 f"built {self.identity.output_binary_sha256[:12]} from "
                 f"{self.identity.snapshot_sha256[:12]}")):
            machine.transition(state, trigger=trigger, reason=reason)
        t0_failed = bool(self.t0_report.failed)
        machine.transition(SM.T0_GATE, trigger="build_complete",
                           reason=f"{len(self.t0_report.gates)} T0 gates evaluated")
        if t0_failed:
            machine.transition(
                SM.POST_RUN_CRITIC, trigger="t0_failed",
                reason=f"T0 failed on {sorted(self.t0_report.failed)}; a compilation or "
                       "correctness failure is a valuable outcome and is banked as one")
        else:
            machine.transition(SM.T1_SEARCH_EVAL, trigger="t0_passed",
                               reason="no T0 gate failed; the candidate is admitted to T1")
            machine.transition(
                SM.POST_RUN_CRITIC, trigger="t1_complete",
                reason=f"verdict {self.outcome.verdict.status} "
                       f"({self.outcome.verdict.effect_resolution})")
        machine.transition(SM.BANK_EVENT, trigger="critic_complete",
                           reason="the evaluation event is durable")
        self.machine = machine
        return machine

    # -- 11. teardown -------------------------------------------------------
    def teardown(self):
        # The witnesses are the REAL frozen trees, read-only. `teardown_worktree`
        # was red-teamed for exactly this: witnessing a decoy used to satisfy
        # `all_production_trees_unchanged`, so the receipt now records which
        # production trees it actually observed and refuses to claim the property
        # over a set that contains none of them.
        self.teardown_receipt = WT.teardown_worktree(
            self.worktree, witness_trees=list(WT.PRODUCTION_TREES))
        self.world.release()
        return self.teardown_receipt

    #: The stages, in the order the code enforces. `up_to()` runs a prefix.
    STAGES = ("claim", "worktree", "build", "artifact", "anchor", "t0",
              "t1", "reduce", "controls", "dispatch")

    def up_to(self, stage: str):
        """Run every stage up to and including `stage`. Raises on an unknown name.

        A named prefix rather than a chain of calls at each call site: the ORDER
        is a property of the pipeline (the artifact digests cannot be measured
        before the build, the anchor cannot be bound before the linkage digest
        exists) and repeating it per test is how one test ends up running a
        different pipeline from the others.
        """
        if stage not in self.STAGES:
            raise ValueError(f"unknown stage {stage!r}; stages are {list(self.STAGES)}")
        steps = {
            "claim": self.acquire_claim, "worktree": self.make_worktree,
            "build": self.build, "artifact": self.measure_artifact,
            "anchor": self.bind_anchor, "t0": self.evaluate_t0,
            "t1": self.run_t1, "reduce": self.reduce,
            "controls": self.score_controls, "dispatch": self.dispatch,
        }
        for name in self.STAGES[:self.STAGES.index(stage) + 1]:
            steps[name]()
        return self

    # -- the whole walk ----------------------------------------------------
    def walk(self, *, through_t1: bool = True):
        self.acquire_claim()
        self.make_worktree()
        self.build()
        self.measure_artifact()
        self.bind_anchor()
        self.evaluate_t0()
        if through_t1:
            self.run_t1()
            self.reduce()
            self.score_controls()
            self.dispatch()
        self.teardown()
        return self


#: A `test-backend-ops` run in which both mandatory ops really were exercised.
#: Built by substituting op names into the RECORDED grammar and labelled as such:
#: no CPU-only build on this host emits a real MUL_MAT_ID case list under a shape
#: filter, which is the finding `recorded_t0_backend_ops_mandatory_ops.txt` holds.
_OPS_OK = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
           "  Device description: AMD EPYC 9655 96-Core Processor\n\n"
           "  MUL_MAT(type_a=f32,type_b=f32,m=16,n=1,k=256): OK\n"
           "  MUL_MAT_ID(type_a=f32,type_b=f32,n_mats=4,n_used=2): OK\n"
           "  2/2 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")


def _t0_policy() -> correctness.T0Policy:
    return correctness.T0Policy(
        required_backend_ops=correctness.MANDATORY_BACKEND_OPS,
        symbol_shrinkage_reject_ratio=0.6,
        diff_ceiling=correctness.DiffComplexityCeiling(
            backend="llama_cpu", max_changed_lines=400, max_files_touched=10,
            shared_core_forces_review=True),
        determinism_min_runs=2, coherence_tolerance_floor=0.98,
        policy_ref="ak-policy/v1")



# =============================================================================
# C. T1 — the same claim, the microbench seam, and the reducer
# =============================================================================

BENCH_RECIPE_ID = "t1b.llama_cpu.llama_bench_decode.v1"
BENCH_FIXTURE = TESTDATA / "recorded_llama_bench_cpu_decode_canonical.json"
FIXTURE_MODEL = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
FIXTURE_BUILD_COMMIT = "91745611f"
ANCHOR_COMMIT = FIXTURE_BUILD_COMMIT + "0" * 31


def scaled_bench(*, factor: float, build_commit: str | None = None) -> str:
    """A DERIVED arm: the recorded sample vector scaled by a stated factor.

    The one place a number in this file is not verbatim recorded output. An A/B
    needs two arms that differ and the host is too contended tonight to measure a
    second one, so the derivation is named here rather than shipped in
    `testdata/` where it could be mistaken for a capture. Same device as
    `test_microbench.scaled_fixture`, deliberately: two spellings of one
    transformation is how the two drift.
    """
    rows = json.loads(BENCH_FIXTURE.read_text(encoding="utf-8"))
    for row in rows:
        row["samples_ts"] = [round(v * factor, 6) for v in row["samples_ts"]]
        row["avg_ts"] = round(sum(row["samples_ts"]) / len(row["samples_ts"]), 6)
        if build_commit is not None:
            row["build_commit"] = build_commit
    return json.dumps(rows)


def bench_binding(root: str, payload: bytes) -> recipes.ToolBinding:
    bindir = Path(root) / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    binary = bindir / "llama-bench"
    binary.write_bytes(payload)
    binary.chmod(0o755)
    return recipes.ToolBinding(binary=str(binary), source_root=str(root),
                               library_path=str(bindir))


class HostStateStub:
    """`read_host_state` replacement. A healthy host, stated as a stub."""

    def __init__(self, state) -> None:
        self.state = state
        self.calls = 0

    def __call__(self, *, cpu_list: str, **kwargs):
        self.calls += 1
        return dataclasses.replace(self.state, cpu_list=cpu_list)


def healthy_host_state() -> MB.HostState:
    return MB.HostState(
        observed_at="2026-08-03T23:00:00+00:00", cpu_list=CANONICAL_CPU_LIST,
        khz_by_cpu=tuple((c, 3500000) for c in range(96)),
        driver_min_khz=400000, driver_max_khz=4510000, load1=2.0,
        source="chain-test stub (the real host is at load ~67 tonight)")


HEALTHY_POLICY = MB.HostStatePolicy(nominal_khz=3500000)


# =============================================================================
# D. The campaign's statistics — solved once, exactly as a real one is
# =============================================================================

def _cal_blocks(count, *, effect, noise, seed, stratum, prefix, split):
    rng = random.Random(seed)
    blocks = []
    for i in range(count):
        anchor_arm = tuple(100.0 + rng.gauss(0, noise * 100.0) for _ in range(3))
        med = sorted(anchor_arm)[1]
        cand_arm = tuple(med * (1.0 + effect) + rng.gauss(0, noise * 100.0)
                         for _ in range(3))
        unit = f"{prefix}-{i}"
        while split.assign(unit) != stratum:
            unit += "x"
        blocks.append(statistics.PairedBlock(
            block_index=i, unit_id=unit, stratum=stratum,
            order=(statistics.ORDER_ANCHOR_FIRST if i % 2 == 0
                   else statistics.ORDER_CANDIDATE_FIRST),
            anchor_samples=anchor_arm, candidate_samples=cand_arm,
            measured_at="2026-08-03T23:00:00+00:00"))
    return tuple(blocks)


CAMPAIGN_SEED = "ak-chain-seed-2026-08-03"


class ChainCampaign:
    """The calibration block, solved once for the whole file.

    The material is SYNTHETIC and says so: a real campaign's calibration block is
    200 measured A/A blocks under a held claim, which is exactly what tonight's
    contention forbids. What is real is the code path — `solve_calibration` is
    the shipped solver and `require_accepted()` refuses a calibration that did
    not converge, here as in production.
    """

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is not None:
            return cls._cache
        controls_decl = api.CampaignControls(
            calibration_block_count=200, contribution_floor=0.10, max_candidates=10,
            confirmation_admission_count=2, max_blocks_per_candidate=20,
            storage_floor_bytes_free=200 * 1024 ** 3)
        rule = statistics.StoppingRule(
            rule_id="ak-stop-chain/v1", final_table="t1_paired_block_table",
            decisions=(("evidence_threshold_crossed", "compose_into_champion_lineage"),
                       ("extension_exhausted", "abandon"),
                       ("block_ceiling_reached", "abandon")),
            extension=statistics.BoundedExtension(max_rounds=1, blocks_per_round=5),
            max_blocks_per_candidate=20)
        construction = statistics.select_construction(
            "sign_martingale_predictable_lambda/v1")
        split = statistics.StratumSplitRule(
            rule_id="ak-split-chain/v1", campaign_seed=CAMPAIGN_SEED,
            confirmation_fraction=0.3,
            rotation=statistics.RotationSchedule(schedule_id="ak-rot-chain/v1",
                                                 period_campaigns=4))
        rng = random.Random(3)
        inputs = statistics.CalibrationInputs(
            backend="llama_cpu", phase="decode", cell_class="operator_microbench",
            campaign_seed=CAMPAIGN_SEED, controls=controls_decl, stopping_rule=rule,
            construction=construction, effect_scale=statistics.EFFECT_SCALE_RELATIVE,
            metric_direction="higher_better",
            hypothesis=statistics.HYPOTHESIS_IMPROVEMENT, margin=0.0,
            aa_blocks=_cal_blocks(200, effect=0.0, noise=0.01, seed=1,
                                  stratum=api.STRATUM_SELECTION, prefix="aa", split=split),
            neutral_blocks=_cal_blocks(60, effect=0.0, noise=0.01, seed=2,
                                       stratum=api.STRATUM_SELECTION, prefix="nt",
                                       split=split),
            anchor_calibration_values=tuple(100.0 + rng.gauss(0, 1.0) for _ in range(200)),
            samples_ref=f"ak-raw://{CAMPAIGN}/calibration/0001")
        solve = statistics.solve_calibration(inputs)
        outputs = solve.require_accepted()
        stats = statistics.CampaignStatistics(
            campaign_id=CAMPAIGN, campaign_seed=CAMPAIGN_SEED,
            effect_scale=statistics.EFFECT_SCALE_RELATIVE,
            hypothesis=statistics.HYPOTHESIS_IMPROVEMENT, margin=0.0, stopping_rule=rule,
            stopping_rule_commitment=statistics.StoppingRuleCommitment.commit(
                rule, campaign_id=CAMPAIGN, committed_at="2026-08-03T23:00:00+00:00"),
            split_rule=split, construction=construction, calibration=outputs,
            aa_effect_pool=solve.aa_effect_pool,
            anchor_calibration_values=solve.anchor_calibration_values)
        cls._cache = (controls_decl, rule, split, solve, outputs, stats)
        return cls._cache



# =============================================================================
# E. Controls -> panel -> verdict -> the controller
# =============================================================================

def _control_arms(count, *, effect, noise, seed):
    rng = random.Random(seed)
    anchor_blocks, candidate_blocks = [], []
    for _ in range(count):
        anchor_arm = tuple(100.0 + rng.gauss(0, noise * 100.0) for _ in range(3))
        med = sorted(anchor_arm)[1]
        anchor_blocks.append(anchor_arm)
        candidate_blocks.append(
            tuple(med * (1.0 + effect) + rng.gauss(0, noise * 100.0) for _ in range(3)))
    return tuple(anchor_blocks), tuple(candidate_blocks)


def _control_fixture(control_id, *, tier, effect, seed, tag, blocks):
    definition = next(d for d in CT.CONTROL_DEFINITIONS if d.control_id == control_id)
    anchor_blocks, candidate_blocks = _control_arms(blocks, effect=effect, noise=0.01,
                                                    seed=seed)
    digest = hashlib.sha256(tag.encode()).hexdigest
    return CR.ControlFixture(
        fixture_id=definition.fixture_id, control_id=control_id, tier=tier,
        candidate_id=f"akc-control-{control_id.replace('_', '-')}",
        artifact=api.ArtifactIdentity(
            source_sha256=hashlib.sha256((tag + "-src").encode()).hexdigest(),
            binary_sha256=hashlib.sha256((tag + "-bin").encode()).hexdigest(),
            linkage_sha256=hashlib.sha256((tag + "-link").encode()).hexdigest()),
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=3),
        created_at=NOW, measured_at=NOW, stratum=api.STRATUM_SELECTION,
        anchor_samples=anchor_blocks, candidate_samples=candidate_blocks,
        available=True, unavailable_reason=None)
    del digest


NOW = "2026-08-03T23:00:00+00:00"

REPLAY_DECLARATION_KW = dict(
    win_id="iqk-prefill-port", backend="llama_cpu", phase="decode",
    reference_direction="higher_better",
    evidence_locator="data/ak-chain/iqk-prefill-port.json",
    durability_class="carried_in_git")


class ControlStack:
    """The five controls, scored through the SAME dispatcher a candidate uses.

    Every arm here is synthetic and labelled: the controls' own material is a
    measurement in a real campaign and there is no recorded control run on this
    host. What this exercises is the seam — `ExecutedControlRunner` ->
    `ControlHarness` -> `api.ControlPanel` -> `WindowAttestations.controls` ->
    `compute_verdict` — which is code, not data.
    """

    def __init__(self, *, degraded_effect=0.90):
        stats = ChainCampaign.get()[5]
        blocks = stats.b_min + stats.stopping_rule.extension.blocks_per_round
        fixtures = (
            _control_fixture(CT.CONTROL_POSITIVE, tier="T1", effect=0.30, seed=11,
                             tag="positive", blocks=blocks),
            _control_fixture(CT.CONTROL_NEUTRAL, tier="T1", effect=0.0, seed=12,
                             tag="neutral", blocks=blocks),
            _control_fixture(CT.CONTROL_DEGRADED_NEGATIVE, tier="T1",
                             effect=degraded_effect, seed=13, tag="degraded",
                             blocks=blocks),
            _control_fixture(CT.CONTROL_AA, tier="T1", effect=0.0, seed=14,
                             tag="aa", blocks=blocks),
            _control_fixture(CT.CONTROL_HISTORICAL_WIN_REPLAY, tier="T2", effect=0.36,
                             seed=15, tag="replay", blocks=blocks),
        )
        self.fixture_set = CR.resolve_fixture_set(
            fixtures=fixtures,
            pinned_digest=schemas.content_hash(CR._fixture_payload(fixtures)),
            source_label="evaluator-bundle@ak-chain")
        controls_decl, rule, _split, solve, outputs, stats = ChainCampaign.get()
        self.anchor = ANCHOR_IDENTITY
        self.binding = CR.CampaignBinding(
            campaign_id=CAMPAIGN, backend="llama_cpu", phase="decode",
            cell_class="operator_microbench", protocol_id=api.PROTOCOL_VERSIONED_ID,
            evaluator=api.EvaluatorIdentity(
                id="P-AK-SEARCH-1/v1",
                bundle_sha256=hashlib.sha256(b"ak-chain-bundle").hexdigest(),
                runtime_source_label_ref="ake-srclabel-chain"),
            scope_denominator=api.ScopeDenominator(
                machine_subset="full", numa_nodes=(), devices=(), cores=96),
            scope_manifest_sha256=hashlib.sha256(b"ak-chain-scope").hexdigest(),
            co_residency="single", metric="tokens_per_second",
            metric_direction="higher_better", reps=10, anchor=self.anchor,
            campaign_controls=controls_decl, calibration=outputs)
        self.declaration = CT.HistoricalWinReplayDeclaration(
            reference_band=CT.ReferenceBand(low=0.30, high=0.45), **REPLAY_DECLARATION_KW)
        self.dispatcher = api.TierDispatcher(gate_runners={
            "T0": _AllPassGateRunner("T0"), "T1": _AllPassGateRunner("T1"),
            "T2": _AllPassGateRunner("T2")})
        self.runner = CR.ExecutedControlRunner(
            pipeline=CR.DispatchPipeline(
                dispatcher=self.dispatcher,
                reducer=statistics.PairedBlockReducer(stats)),
            fixtures=self.fixture_set, binding=self.binding,
            campaign_statistics=stats)
        self.harness = CT.ControlHarness(
            bundle=CT.resolve_control_bundle(
                pinned_definitions_digest=CT.CONTROL_DEFINITIONS_DIGEST,
                aa_cadence=CT.AACadence(every_n_windows=5, every_n_seconds=3600.0,
                                        declared_at=NOW),
                seed_rotation=CT.SeedRotationSchedule(rotate_every_windows=10,
                                                      declared_at=NOW),
                historical_win_replays=(self.declaration,),
                source_label="evaluator-bundle@ak-chain"),
            runner=self.runner)
        self.sweep = CR.ControlSweep(harness=self.harness, campaign_seed=CAMPAIGN_SEED)
        self.solve = solve
        self.outputs = outputs

    def resolution(self):
        return CT.HistoricalWinResolution(
            backend="llama_cpu", available=True, declaration=self.declaration,
            durability_outcome=schemas.PASS,
            check=schemas.Check(schemas.PASS, ("fixture: resolves in-repo",)))

    def run(self, window, *, window_id="akw-chain-0001", windows_completed=0):
        return self.sweep.run(
            run_context=CT.ControlRunContext(
                campaign_id=CAMPAIGN, backend="llama_cpu", phase="decode",
                cell_class="operator_microbench", window_id=window_id, tier="T1",
                seed="PLACEHOLDER-SEED-MUST-NOT-BE-USED", anchor=self.anchor,
                declaration=self.declaration),
            context=CT.ControlContext(
                campaign_id=CAMPAIGN, backend="llama_cpu", phase="decode",
                cell_class="operator_microbench", window_id=window_id,
                historical=self.resolution(),
                neutral_dispersion=CT.neutral_dispersion_check(self.solve),
                calibration=self.outputs),
            window=window, aa_cadence=schemas.Check(schemas.PASS),
            windows_completed=windows_completed, last_rotation_epoch=0)


class _AllPassGateRunner:
    """A gate runner for the CONTROLS' own dispatch. It is not the T0 runner.

    It FAILs the degraded-negative control's artifact and passes everything else.
    That is not a convenience: control 3 is *"a deliberately degraded candidate
    that looks fast"* and it PASSES when the pipeline REFUSES to rank it. A stand-in
    that passed everything would make control 3 fail on correct behaviour — which
    is a gate that gets switched off, not a gate that works. The correctness
    verdict here is keyed to the fixture's own source digest, the same way
    `test_control_runner._GateRunner` keys it.

    The real T0 runner is exercised on the candidate's own leg, where its gates
    are measurements rather than a stand-in.
    """

    def __init__(self, tier: str) -> None:
        self.tier = tier
        self.requests: list = []

    def run_gates(self, request):
        self.requests.append(request)
        degraded = request.artifact.source_sha256 == DEGRADED_SOURCE_SHA256
        check = schemas.Check(
            schemas.FAIL,
            ("the op suite disagreed with the reference on 11/64 shapes; the kernel "
             "silently falls back and reports the cached result",)) if degraded \
            else schemas.Check(schemas.PASS)
        return (
            api.GateResult(gate_id="ops-suite", gate_class=api.GATE_CORRECTNESS,
                           check=check, requires_anchor=True,
                           evidence_ref="ak-raw://chain/ops/0001"),
            api.GateResult(gate_id="numerics", gate_class=api.GATE_NUMERICAL_SAFETY,
                           check=schemas.Check(schemas.PASS), requires_anchor=True),
        )


#: The degraded-negative control fixture's own source digest. Derived by the same
#: expression `_control_fixture` uses, so renaming the tag cannot leave the gate
#: runner pointing at a digest no fixture carries.
DEGRADED_SOURCE_SHA256 = hashlib.sha256(b"degraded-src").hexdigest()


ANCHOR_IDENTITY = api.AnchorIdentity(
    source_commit=ANCHOR_COMMIT,
    binary_sha256=hashlib.sha256(b"ak-chain-anchor-binary").hexdigest(),
    linkage_sha256=hashlib.sha256(b"ak-chain-anchor-linkage").hexdigest())


def window_attestations(*, claim_receipt: str, controls: api.ControlPanel,
                        anchor: api.AnchorIdentity,
                        control_definitions_immutable=schemas.Check(schemas.PASS),
                        resource_claim_open=schemas.Check(schemas.PASS)):
    """The window, with the REAL claim's receipt id in `resource_claim_receipt`.

    Deliberately built field by field with no `all_clear()` helper — `api.py`
    refuses to ship one, on the grounds that a fixture that fabricates PASS is
    the fixture that removes the signal under test. What is NOT fabricated here
    is the claim receipt: it is the id of the flock this test really held.
    """
    p = schemas.Check(schemas.PASS)
    return api.WindowAttestations(
        resource_claim_receipt=claim_receipt,
        resource_claim_open=resource_claim_open, resource_claim_close=p,
        resource_claim_same_holder=p, no_concurrent_inference=p,
        preflight_attestation_ref="ake-chain-preflight-0001",
        host_receipt="host-chain-20260803T2300Z", host_health=p,
        anchor_at_open=anchor, anchor_at_close=anchor, anchor_gate=p,
        evaluator_bundle=p, runtime_source_label=p,
        recipe=api.RecipeReceipt(
            constructor_id="ak.microbench.llama_cpu.decode/v1",
            constructor_sha256=hashlib.sha256(b"chain-constructor").hexdigest(),
            argv_sha256=hashlib.sha256(b"chain-argv").hexdigest()),
        storage_open=p, storage_close=p, strata=p,
        stopping_rule_id="ak-stop-chain/v1", rule_immutability=p,
        order_randomized=p, order_seed="PLACEHOLDER-SEED-MUST-NOT-BE-USED",
        aa_cadence=p, controls=controls, calibration=p,
        control_definitions_immutable=control_definitions_immutable,
        raw_evidence_ref=f"data/{CAMPAIGN}/raw/")



# =============================================================================
# F. THE POSITIVE PATH — the chain fits, end to end
# =============================================================================

class TestTheChainFits(_ChainCase):
    """One leg, all the way through. Each assertion names the seam it covers."""

    def setUp(self):
        super().setUp()
        self.leg = ChainLeg(self.world).walk()

    def test_the_claim_was_a_real_lock_and_covered_the_argv_footprint(self):
        """Seam 4, and denial 8's precondition 1."""
        self.assertTrue(self.leg.claim_receipt_at_open["lock_paths"])
        self.assertEqual(self.leg.claim_footprint_check.outcome, schemas.PASS,
                         self.leg.claim_footprint_check.reasons)

    def test_the_worktree_came_from_the_branch_tip_and_left_the_source_unchanged(self):
        self.assertTrue(self.leg.world.worktree_proof.holds,
                        self.leg.world.worktree_proof.differences)
        self.assertEqual(self.leg.worktree.source_commit, self.world.tip)

    def test_the_build_receipt_projects_into_the_gate_the_evaluator_actually_reads(self):
        """SEAM 1. The object `evaluate_t0` consumed is a correctness.BuildProvenance."""
        evidence = self.leg.build_evidence
        self.assertIsInstance(evidence.provenance, correctness.BuildProvenance)
        self.assertEqual(evidence.worst.outcome, schemas.PASS, evidence.worst.reasons)
        self.assertEqual(
            self.leg.t0_report.outcome(correctness.GID_CLEAN_BUILD), schemas.PASS,
            self.leg.t0_report.gate(correctness.GID_CLEAN_BUILD).check.reasons)

    def test_the_gates_equality_is_a_comparison_and_not_a_tautology(self):
        """SEAM 2: the receipt's digests and the record's were taken separately.

        They are equal here, and the point is that they CAN differ —
        `TestAnArtifactThatIsNotTheBuiltOne` makes them differ and the gate FAILs.
        """
        self.assertEqual(self.leg.artifact.source_sha256,
                         self.leg.build_evidence.provenance.built_from_snapshot_sha256)
        self.assertEqual(self.leg.artifact.binary_sha256,
                         self.leg.build_evidence.provenance.output_binary_sha256)

    def test_t0_produced_seventeen_gates_and_none_of_them_failed(self):
        report = self.leg.t0_report
        self.assertEqual(len(report.gates), len(correctness.T0_GATE_IDS))
        self.assertEqual(sorted(report.failed), [])

    def test_exactly_nine_t0_surfaces_still_have_no_producer(self):
        """The number `execution/README.md` tells tomorrow's session to expect.

        Not a vanity assertion: the runbook says "a good candidate today looks
        like 8 PASS and 9 COULD_NOT_CHECK", and a reader who sees a different
        shape has to know whether the campaign is broken or the runbook is stale.
        When a producer is wired the count moves and this test fails, which is
        the reminder to update §3 and §6.1 of the runbook.
        """
        report = self.leg.t0_report
        unproduced = sorted(g.gate_id for g in report.gates
                            if g.check.outcome == schemas.COULD_NOT_CHECK)
        self.assertEqual(unproduced, sorted([
            correctness.GID_ASAN,
            correctness.GID_BOUNDARY_SHAPES,
            correctness.GID_EXACT_REFERENCE,
            correctness.GID_SCHEMA_DIFF_POLICY,
            correctness.GID_SEMANTIC_DIFF,
            correctness.GID_STATE_SAFETY,
            correctness.GID_STATIC_COMPILE,
            correctness.GID_SYMBOLS,
            correctness.GID_UBSAN,
        ]), "the set of T0 surfaces with no producer has changed — update "
           "execution/README.md §3 and §6.1, which tell tomorrow's session what "
           "a healthy report looks like")
        passed = [g.gate_id for g in report.gates if g.check.outcome == schemas.PASS]
        self.assertEqual(len(passed), 8, sorted(passed))

    def test_the_two_anchor_bindings_are_one_anchor_build(self):
        """SEAM 3. Different binaries, one commit, one linkage."""
        check = chain.check_anchor_build_is_one_build(
            [self.leg.anchor_binding, self.leg.t1_anchor])
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)
        self.assertNotEqual(self.leg.anchor_binding.identity.binary_sha256,
                            self.leg.t1_anchor.identity.binary_sha256)

    def test_every_consumer_of_the_t1_anchor_names_the_t1_anchor(self):
        check = chain.check_anchor_matches(self.leg.t1_anchor, consumers={
            "the evaluation request": self.leg.t1_request.anchor,
            "the window at open": self.leg.window.anchor_at_open,
            "the window at close": self.leg.window.anchor_at_close,
        })
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_t1_emitted_paired_blocks_the_reducer_admitted(self):
        self.assertTrue(self.leg.t1_run.complete, self.leg.t1_run.refusals)
        blocks = self.leg.t1_run.paired_blocks()
        self.assertEqual(len(blocks), 5)
        for block in blocks:
            self.assertIsInstance(block, statistics.PairedBlock)
        self.assertEqual(self.leg.reduction.admissible.outcome, schemas.PASS,
                         self.leg.reduction.admissible.reasons)
        self.assertIsNotNone(self.leg.reduction.estimate)

    def test_the_bench_argv_carried_the_canonical_prefix_and_the_whole_omp_stack(self):
        """The measurement discipline, asserted on what WOULD have been executed."""
        calls = self.leg.microbench_spawner.calls
        self.assertTrue(calls)
        prefix = list(recipes.CANONICAL_PREFIX)
        for call in calls:
            self.assertEqual(list(call["argv"][:len(prefix)]), prefix)
            self.assertEqual(call["argv"][call["argv"].index("-fa") + 1], "1")
            for key, value in recipes.CANONICAL_OMP_ENV.items():
                self.assertEqual(call["env"][key], value)

    def test_five_controls_were_scored_and_the_panel_licenses_ranking(self):
        self.assertTrue(self.leg.sweep_result.may_rank,
                        self.leg.sweep_result.blocked_reason)
        self.assertEqual(self.leg.window.controls.marker(), "5/5")

    def test_the_dispatcher_returned_a_verdict_over_the_measured_panel(self):
        outcome = self.leg.outcome
        self.assertEqual(outcome.verdict.status, schemas.PASS.lower()
                         if hasattr(schemas.PASS, "lower") else "pass")
        self.assertEqual(outcome.void_scan.findings, ())
        self.assertEqual(outcome.event_violations, ())
        self.assertIsNotNone(outcome.record_content_hash)

    def test_the_record_grammar_line_is_complete(self):
        self.assertEqual(self.leg.outcome.grammar_complete.outcome, schemas.PASS,
                         self.leg.outcome.grammar_complete.reasons)
        self.assertIn("SEARCH RECORD, NOT A CLAIM", self.leg.outcome.grammar_line)

    def test_the_controller_banks_the_event(self):
        machine = self.leg.bank(os.path.join(self._tmp.name, "controller-root"))
        self.assertEqual(machine.state, SM.BANK_EVENT)

    def test_the_claim_was_released_and_the_worktree_torn_down(self):
        self.assertIsNone(self.world.claim)
        self.assertFalse(os.path.exists(self.leg.worktree.path.path))
        receipt = self.leg.teardown_receipt.to_dict()
        self.assertTrue(receipt["all_production_trees_unchanged"])
        self.assertTrue(receipt["production_trees_witnessed"])


# =============================================================================
# G. THE NEGATIVE PATHS — what the chain must REFUSE
# =============================================================================

class TestNoClaimMeansNoMeasurement(_ChainCase):
    """Denial 8, at both doors, with a compliant-path control beside each."""

    def test_the_t0_provider_refuses_to_collect_without_a_claim(self):
        leg = ChainLeg(self.world, claim="none")
        leg.make_worktree()
        leg.build()
        leg.measure_artifact()
        leg.bind_anchor()
        with self.assertRaises(T0.ClaimNotHeld):
            leg.evaluate_t0()

    def test_the_microbench_runner_cannot_even_be_constructed_without_a_claim(self):
        with self.assertRaises(MB.ClaimNotHeld):
            MB.MicrobenchRunner(claim=None, spawner=MB.RecordedSpawner({}))

    def test_a_released_claim_stops_the_next_measurement(self):
        """The mid-run revocation check, on the REAL claim rather than a stub."""
        leg = ChainLeg(self.world)
        leg.acquire_claim()
        binding = leg.claim_binding
        self.assertEqual(binding.microbench_claim.attest().check.outcome, schemas.PASS)
        leg.world.release()
        self.assertEqual(binding.microbench_claim.attest().check.outcome, schemas.FAIL)

    def test_the_compliant_path_still_measures(self):
        """The control: with the claim held, both seams say yes."""
        leg = ChainLeg(self.world)
        leg.acquire_claim()
        check = chain.check_claim_satisfies_both_seams(leg.claim,
                                                       cpu_list=CANONICAL_CPU_LIST)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_a_claim_over_a_narrower_region_does_not_authorise_the_canonical_run(self):
        """Precondition 1 is COVERAGE, not existence.

        A claim over cores 0-23 answers `held` exactly like a claim over 0-95,
        and the argv is pinned to 0-95 either way; the 72 cores outside it belong
        to whoever holds them. Without this the footprint branch of
        `check_claim_satisfies_both_seams` is decorative — verified by mutation:
        replacing `not covers(cpu_list)` with `False` left the suite green until
        this test existed.
        """
        narrow = self.world.acquire("0-23")
        self.addCleanup(narrow.release)
        check = chain.check_claim_satisfies_both_seams(narrow,
                                                       cpu_list=CANONICAL_CPU_LIST)
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        # The T0 seam's OWN refusal, matched on its own spelling. Matching a
        # phrase both seams use would have been satisfied by the adapter's
        # refusal alone — verified by mutation: with `covers()` disabled here the
        # adapter still FAILs and a loose assertion stayed green, so the branch
        # would have been decorative. It is not redundant: a claim that
        # implements `attest()` natively never reaches the adapter.
        self.assertIn(f"claim {narrow.claim_id!r} does not cover "
                      f"{CANONICAL_CPU_LIST!r}", check.reasons)
        # and the compliant half of the same claim: it DOES authorise its own region
        self.assertEqual(
            chain.check_claim_satisfies_both_seams(narrow, cpu_list="0-23").outcome,
            schemas.PASS)


class TestAFrozenTreeIsRefusedAtEveryDoor(_ChainCase):
    """Every constructor that could name a production tree, and the one that may."""

    FROZEN = "/mnt/raid0/llm/llama.cpp"

    def test_a_sandbox_path_inside_a_frozen_tree_is_refused(self):
        with self.assertRaises(WT.ProductionTreeViolation):
            WT.SandboxPath.create(f"{self.FROZEN}/build-ak")

    def test_a_traversal_into_a_frozen_tree_is_refused(self):
        with self.assertRaises(WT.WorktreeError):
            WT.SandboxPath.create("/mnt/raid0/llm/exp/../llama.cpp/build-ak")

    def test_a_candidate_build_inside_a_frozen_tree_is_refused(self):
        with self.assertRaises(T0.ProductionTreeRefusal):
            T0.CandidateBuild(
                worktree=self.FROZEN, build_dir=f"{self.FROZEN}/build",
                source_commit="0" * 39 + "1",
                source_sha256=integrity.sha256_file(__file__),
                binary=f"{self.FROZEN}/build/bin/llama-cli",
                library_path=f"{self.FROZEN}/build/bin",
                test_backend_ops=f"{self.FROZEN}/build/bin/test-backend-ops")

    def test_a_region_lock_root_inside_a_frozen_tree_is_refused(self):
        with self.assertRaises(CRC.LockRootDenied):
            CRC.plan_region_claim("0-23", role="autokernel",
                                  lock_root=f"{self.FROZEN}/.ak-locks")

    def test_a_build_whose_paths_land_in_a_frozen_tree_is_reported_as_touching_one(self):
        """SEAM 1's polarity, at the value the gate actually reads."""
        leg = ChainLeg(self.world)
        leg.up_to("build")
        forged = dataclasses.replace(leg.identity, build_dir=f"{self.FROZEN}/build")
        touched = chain.production_trees_touched_by(forged)
        # Every SPELLING of the tree, because `frozen_tree_paths()` carries the
        # aliases too (`/workspace/repos/epyc-llama` is a symlink to this path)
        # and a reader of the record must be able to match either.
        self.assertIn(self.FROZEN, touched)
        self.assertEqual(chain.build_evidence(forged)
                         .provenance.production_tree_paths_touched, touched)

    def test_the_compliant_campaign_worktree_is_not_inside_the_frozen_tree(self):
        """The control. `llama.cpp-ak-…` shares the prefix and is NOT contained.

        A `str.startswith` containment test refuses this path, which would block
        every campaign; the component-wise test admits it.
        """
        path = WT.SandboxPath.create("/mnt/raid0/llm/llama.cpp-ak-0001")
        self.assertEqual(path.path, "/mnt/raid0/llm/llama.cpp-ak-0001")
        leg = ChainLeg(self.world)
        leg.up_to("build")
        self.assertEqual(chain.production_trees_touched_by(leg.identity), ())

    def test_the_anchor_may_name_a_frozen_tree(self):
        """The other control: the anchor IS the frozen binary and reading it is not a write."""
        anchor = T0.AnchorBuild(worktree=self.FROZEN, source_commit="0" * 39 + "1",
                                binary=f"{self.FROZEN}/build/bin/llama-cli",
                                library_path=f"{self.FROZEN}/build/bin")
        self.assertEqual(anchor.worktree, self.FROZEN)


class TestAnAnchorMismatchRaises(_ChainCase):
    """It RAISES. A downgrade would file the campaign's bug as the candidate's property."""

    def test_two_stages_naming_different_anchors_raise(self):
        leg = ChainLeg(self.world)
        leg.up_to("anchor")
        other = api.AnchorIdentity(
            source_commit="1" * 40,
            binary_sha256=T0.sha256_text("some other anchor binary"),
            linkage_sha256=T0.sha256_text("some other anchor linkage"))
        with self.assertRaises(chain.AnchorNotOneAnchor):
            chain.require_anchor_matches(leg.anchor_binding,
                                         consumers={"the evaluation request": other})

    def test_two_tools_from_two_builds_are_refused(self):
        leg = ChainLeg(self.world)
        leg.up_to("anchor")
        stale = chain.bind_anchor(T0.AnchorCapture(
            source_commit="1" * 40,
            binary_sha256=T0.sha256_text("stale llama-bench"),
            linkage_sha256=T0.sha256_text("stale linkage")), tool="llama-bench")
        check = chain.check_anchor_build_is_one_build([leg.anchor_binding, stale])
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_an_unbound_anchor_is_could_not_check_and_does_not_raise(self):
        """Silence is not agreement, and it is not a defect in the campaign either."""
        leg = ChainLeg(self.world)
        leg.up_to("anchor")
        check = chain.check_anchor_matches(leg.anchor_binding,
                                           consumers={"the evaluation request": None})
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        chain.require_anchor_matches(leg.anchor_binding,
                                     consumers={"the evaluation request": None})

    def test_t0_evidence_captured_against_another_anchor_raises_in_the_evaluator(self):
        """`correctness._refuse_replay_mismatch` — reached through the real provider."""
        leg = ChainLeg(self.world)
        leg.up_to("anchor")
        plan = leg.t0_plan()
        provider = T0.ExecutedT0EvidenceProvider(
            plan=plan, runner=leg._t0_runner(plan, op_suite_text=_OPS_OK),
            claim=leg.claim_binding.t0_claim, anchor_capture=leg.anchor_binding.capture)
        request = leg.evaluation_request(anchor=api.AnchorIdentity(
            source_commit="2" * 40,
            binary_sha256=T0.sha256_text("a different anchor binary"),
            linkage_sha256=T0.sha256_text("a different anchor linkage")))
        evidence = provider.evidence_for(request)
        with self.assertRaises(correctness.EvidenceAnchorMismatch):
            correctness.evaluate_t0(request, evidence, _t0_policy())


class TestAnArtifactThatIsNotTheBuiltOne(_ChainCase):
    """SEAM 2's whole point: the gate's equality can fail, so it is a check."""

    def test_a_record_naming_another_binary_fails_the_clean_build_gate(self):
        leg = ChainLeg(self.world)
        leg.up_to("anchor")
        other = leg.artifacts["test-backend-ops"]
        leg.artifact = chain.measure_artifact_identity(
            source_root=leg.worktree.path.path, binary=other,
            linkage_sha256=leg.linkage_sha256)
        report = leg.evaluate_t0()
        self.assertIn(correctness.GID_CLEAN_BUILD, report.failed)


class TestACcacheBuildIsNotACleanBuild(_ChainCase):
    """The recorded configure log, unmodified: ccache ON and a dirty ggml commit."""

    def test_the_recorded_ccache_configure_fails_the_clean_build_gate(self):
        leg = ChainLeg(self.world,
                       configure_log=raw("recorded_configure_ccache.log"))
        leg.up_to("anchor")
        names = dict(leg.build_evidence.checks)
        self.assertEqual(names["no_external_object_cache"].outcome, schemas.FAIL)
        self.assertEqual(names["snapshot_is_what_built"].outcome, schemas.FAIL)
        self.assertTrue(leg.build_evidence.provenance.incremental_objects_present)
        self.assertIn(correctness.GID_CLEAN_BUILD, leg.evaluate_t0().failed)

    def test_the_clean_configure_is_the_compliant_control(self):
        """Without this the guard could be `incremental_objects_present = True`."""
        leg = ChainLeg(self.world)
        leg.up_to("build")
        self.assertFalse(leg.build_evidence.provenance.incremental_objects_present)
        self.assertTrue(leg.build_evidence.provenance.build_dir_was_fresh)


class TestAContendedRunEmitsNoNumber(_ChainCase):
    """Requirement 5: a refused run has no accessor that hands back partial blocks."""

    def _leg_to_t1(self):
        leg = ChainLeg(self.world)
        leg.up_to("t0")
        return leg

    def test_a_throttled_host_refuses_and_paired_blocks_raises(self):
        leg = self._leg_to_t1()
        throttled = dataclasses.replace(
            healthy_host_state(),
            khz_by_cpu=tuple((c, 1200000) for c in range(96)))
        original = HostStateStub
        run = self._run_with_state(leg, throttled)
        self.assertFalse(run.complete)
        with self.assertRaises(MB.RunRefused):
            run.paired_blocks()
        self.assertTrue(run.raw_vector(), "the refusal must still be durable")
        del original

    def test_a_loaded_host_refuses(self):
        leg = self._leg_to_t1()
        loaded = dataclasses.replace(healthy_host_state(), load1=67.0)
        run = self._run_with_state(leg, loaded)
        self.assertFalse(run.complete)

    def test_the_healthy_control_still_produces_blocks(self):
        leg = self._leg_to_t1()
        run = self._run_with_state(leg, healthy_host_state())
        self.assertTrue(run.complete, run.refusals)

    def _run_with_state(self, leg, state):
        leg.run_t1()                                  # builds the bindings and the plan
        plan = MB.MicrobenchPlan(
            recipe_id=BENCH_RECIPE_ID, candidate_id=CANDIDATE,
            campaign_seed=CAMPAIGN_SEED,
            candidate_binding=leg.candidate_binding,
            anchor_binding=leg.anchor_binding_tool,
            anchor=leg.t1_anchor.identity,
            params={"model": FIXTURE_MODEL, "n_gen": 128, "reps": 10,
                    "output_format": "json"},
            base_blocks=5, pairs_per_block=1, unit_ids=("chain-unit-0",))
        runner = MB.MicrobenchRunner(
            claim=leg.claim_binding.microbench_claim, policy=HEALTHY_POLICY,
            spawner=MB.RecordedSpawner({
                MB.ARM_CANDIDATE: scaled_bench(factor=1.08, build_commit="cafe12345"),
                MB.ARM_ANCHOR: BENCH_FIXTURE.read_text(encoding="utf-8")}),
            host_state=HostStateStub(state))
        return runner.run(plan)


# =============================================================================
# H. The seams' own unit surface
# =============================================================================

class TestTheBuildProvenanceProjection(unittest.TestCase):

    def test_the_compiler_split_refuses_to_invent_a_version(self):
        self.assertEqual(chain.split_compiler_identity("CXX GNU 15.2.0"),
                         ("CXX GNU", "15.2.0"))
        for bad in ("GNU", "ASM GNU", "", "clang"):
            with self.assertRaises(chain.BuildProvenanceUnprojectable, msg=bad):
                chain.split_compiler_identity(bad)

    def test_half_a_compiler_override_is_refused(self):
        with self.assertRaises(TypeError):
            chain.build_evidence(object())

    def test_the_build_log_ref_names_the_content(self):
        ref = "file:///tmp/x/build.log#sha256=" + "a1" * 32
        self.assertEqual(T0.resolve_build_log_ref(ref), "/tmp/x/build.log")
        self.assertEqual(T0.resolve_build_log_ref("/tmp/x/build.log"),
                         "/tmp/x/build.log")
        for bad in ("", None, "build.log", "https://example/build.log", "/tmp/a#b"):
            self.assertIsNone(T0.resolve_build_log_ref(bad), bad)


class TestTheExtensionRoundHasNoProducer(_ChainCase):
    """A gap that BLOCKS the first campaign, pinned so it cannot be forgotten.

    The calibrated threshold for this cell is 10 and the sign-martingale
    e-value over B_min=5 same-sign blocks tops out at 5.57 — so no candidate
    can cross on the base segment alone, whatever its true effect. Crossing
    needs the declared extension round, and:

      * `statistics._check_extension_structure` ALREADY accepts a submission of
        base blocks followed by whole extension rounds, and
      * `microbench.plan_blocks` ALREADY takes `segment=` and `extension_round=`,

    but `MicrobenchPlan` has no field for either and `MicrobenchRunner.run()`
    calls `plan_blocks` with the defaults, so the runner can only ever emit
    `SEGMENT_BASE`. The fix is two fields on the plan and passing them through;
    it is NOT made here because the order schedule across two runner
    invocations is a statistical decision, not a plumbing one.

    When it is made, this test should be replaced by one that runs an extension
    round — not deleted.
    """

    def test_the_runner_emits_only_base_segment_blocks(self):
        leg = ChainLeg(self.world)
        leg.up_to("t0")
        for block in leg.run_t1().paired_blocks():
            self.assertEqual(block.segment, statistics.SEGMENT_BASE)
            self.assertIsNone(block.extension_round)
        self.assertFalse(
            any(f.name in ("segment", "extension_round")
                for f in dataclasses.fields(MB.MicrobenchPlan)),
            "MicrobenchPlan now carries the extension fields — the gap this test "
            "pins is closed. Replace it with a test that runs an extension round "
            "and update execution/README.md, which tells tomorrow's session that "
            "no candidate can cross the evidence threshold.")

    def test_the_reducer_is_already_ready_for_the_extension(self):
        """The compliant-path control: the far side of the gap is not the problem."""
        stats = ChainCampaign.get()[5]
        self.assertEqual(stats.stopping_rule.extension.blocks_per_round, 5)
        self.assertEqual(stats.stopping_rule.extension.max_rounds, 1)
        base = statistics.OrderSchedule.derive(
            campaign_seed=CAMPAIGN_SEED, candidate_id=CANDIDATE,
            base_blocks=stats.b_min, attempt=0)
        plans = MB.plan_blocks(base, count=5, pairs=1, unit_ids=("u",),
                               stratum=api.STRATUM_SELECTION,
                               segment=statistics.SEGMENT_EXTENSION, extension_round=1)
        self.assertEqual([p.block_index for p in plans], [5, 6, 7, 8, 9])
        self.assertTrue(all(p.segment == statistics.SEGMENT_EXTENSION for p in plans))


class TestFrozenTreesAreUntouched(_ChainCase):
    """The hard boundary, checked around the whole walk rather than asserted."""

    def test_the_three_frozen_trees_are_byte_identical_after_a_full_leg(self):
        before = fingerprint_frozen()
        self.assertTrue(any(v is not None for v in before.values()),
                        "no frozen tree was found to check; this test would be vacuous")
        leg = ChainLeg(self.world)
        leg.walk()
        after = fingerprint_frozen()
        self.assertEqual(before, after)

    def test_llama_cpp_is_still_on_the_v8_freeze_commit(self):
        fp = fingerprint_frozen()["/mnt/raid0/llm/llama.cpp"]
        if fp is None:
            self.skipTest("the frozen llama.cpp clone is not present on this host")
        self.assertEqual(fp["branch"], "production-consolidated-v8")
        self.assertEqual(fp["head"], "67a433bf45a8a091d83b4ea0b32ff0735fd51800")
