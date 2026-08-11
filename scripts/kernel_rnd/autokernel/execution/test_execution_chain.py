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
      -> claim released, worktree torn down
      -> THE PRODUCTION TREES ARE BYTE-IDENTICAL TO BEFORE

The walk used to have one more step between the verdict and the teardown — the
controller walking `BUILD -> T0_GATE -> T1_SEARCH_EVAL -> ... -> BANK_EVENT` over
the verdict this chain produced. `controller/state_machine.py` was deleted on
2026-08-04 with the rest of AK4, and that step is excised rather than stubbed.
What it covered, and therefore what is no longer covered, is written out at
`ChainLeg` stage 10.

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

from autokernel import journal as J                                     # noqa: E402
from autokernel import schemas                                          # noqa: E402
from autokernel.evaluator import (api, correctness, integrity,          # noqa: E402
                                  recipes, statistics)
from autokernel.evaluator import surface as SU                          # noqa: E402
from autokernel.evaluator import controls as CT                         # noqa: E402
from autokernel.evaluator import controls as controls_module            # noqa: E402
# The ELF writer is imported from `evaluator/test_integrity.py` rather than
# copied. It is ~60 lines of struct-packing that produces a VALID ELF64 — the
# only kind `integrity.extract_elf_symbols` accepts, since it raises
# `ElfFormatError` rather than returning an empty table for anything else — and a
# second copy here would be a second thing to keep in step with the reader.
from autokernel.evaluator.test_integrity import build_elf64, fn as elf_fn        # noqa: E402
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

# =============================================================================
# Fixtures for the four producers wired on 2026-08-04 (README §6.1)
# =============================================================================

#: The anchor's exported ABI. Two C entry points and one template specialization,
#: so a removal, an arity change and a mangled/qualified declaration can each be
#: exercised separately.
ANCHOR_EXPORTS = [
    elf_fn("ggml_mul_mat"),
    elf_fn("ggml_mul_mat_id"),
    elf_fn("_ZN4ggml6detail15kernel_dispatchILi4EEEvPKfPfi"),
]
#: The compliant candidate: same ABI plus one addition. §8.5.1 makes only removal
#: and arity change hard, so an undeclared ADDITION must not fail the gate.
CANDIDATE_EXPORTS = ANCHOR_EXPORTS + [elf_fn("ggml_mul_mat_id_avx512")]

#: The backend adapter's declared registration patterns. `PatternRegistrationExtractor`
#: refuses an empty mapping and requires a named `key` group; `arity` is optional
#: and a missing one means "the pattern did not capture it", never "unchanged".
OP_REGISTRATION_PATTERNS = {
    "ggml_backend_cpu": r"GGML_CPU_OP\((?P<key>\w+)\s*,\s*(?P<arity>\d+)\)",
}
DISPATCH_PREDICATE_PATTERNS = {
    "cpu_supports_op": r"CPU_SUPPORTS\((?P<key>\w+)\)",
}
REGISTRATION_SOURCES = {
    "ggml/src/ggml-cpu.c": (
        "GGML_CPU_OP(MUL_MAT, 2)\n"
        "GGML_CPU_OP(MUL_MAT_ID, 3)\n"
        "CPU_SUPPORTS(MUL_MAT)\n"
        "CPU_SUPPORTS(MUL_MAT_ID)\n"
    ),
}


def registration_tables(label: str, sources=None):
    """`(op_registration_table, dispatch_predicate_table)` for one side."""
    body = REGISTRATION_SOURCES if sources is None else sources
    ops = integrity.PatternRegistrationExtractor(
        kind=integrity.KIND_OP_REGISTRATION, patterns=OP_REGISTRATION_PATTERNS,
        declared_by="ak-chain-backend-adapter/v1").extract_text(label, body)
    predicates = integrity.PatternRegistrationExtractor(
        kind=integrity.KIND_DISPATCH_PREDICATE, patterns=DISPATCH_PREDICATE_PATTERNS,
        declared_by="ak-chain-backend-adapter/v1").extract_text(label, body)
    return ops, predicates


#: The candidate's diff. Deliberately touches NEITHER memory nor threading tokens
#: — `TestTheBehaviouralClassifierOnlyWidens` supplies one that does. The healthy
#: leg therefore keeps ASAN/UBSAN at COULD_NOT_CHECK, which is the honest reading
#: for a change whose behavioural surface was not determined.
CANDIDATE_DIFF = """diff --git a/ggml/src/ggml-cpu.c b/ggml/src/ggml-cpu.c
index 1111111..2222222 100644
--- a/ggml/src/ggml-cpu.c
+++ b/ggml/src/ggml-cpu.c
@@ -10,5 +10,5 @@ GGML_CPU_OP(MUL_MAT, 2)
 GGML_CPU_OP(MUL_MAT_ID, 3)
 CPU_SUPPORTS(MUL_MAT)
 CPU_SUPPORTS(MUL_MAT_ID)
-    const int step = 4;
+    const int step = 8;
 }
"""

#: A diff that DOES touch memory, for the classifier's positive path.
MEMORY_TOUCHING_DIFF = """diff --git a/ggml/src/ggml-cpu.c b/ggml/src/ggml-cpu.c
index 1111111..3333333 100644
--- a/ggml/src/ggml-cpu.c
+++ b/ggml/src/ggml-cpu.c
@@ -10,3 +10,4 @@
 CPU_SUPPORTS(MUL_MAT)
-    float * tmp = params->wdata;
+    float * tmp = (float *) malloc(n * sizeof(float));
+    memcpy(tmp, src, n * sizeof(float));
 }
"""

CHANGE_ENVELOPE = correctness.ChangeClassEnvelope(
    change_class="arithmetic", max_changed_lines=400, max_files_touched=10)

COMMIT_ARGV = ("git", "commit", "-m", "akc-chain-0001: widen the CPU step",
               "--", "ggml/src/ggml-cpu.c")

#: A miniature build-system dependency index, in the shape `gcc -MD` and CMake's
#: `link.txt` really write. The closure is what makes `derive_affected_surface`'s
#: output a derivation rather than a list.
CHAIN_DEPFILE = ("CMakeFiles/ggml.dir/ggml-cpu.c.o: ../ggml/src/ggml-cpu.c "
                 "../ggml/include/ggml.h\n")
CHAIN_LINK = ("/usr/bin/c++ -O3 CMakeFiles/ggml.dir/ggml-cpu.c.o -o bin/llama-bench -lm")


def chain_affected_surface(*, touched=("ggml/src/ggml-cpu.c",), with_registrations=True):
    """`surface.derive_affected_surface` over the miniature index above."""
    index = SU.build_dependency_index(
        label="candidate", build_dir="build-t0", source_root="/repo/llama.cpp",
        dep_edges=SU.parse_make_depfile(CHAIN_DEPFILE, origin_ref="ggml-cpu.d"),
        link_edges=[SU.parse_cmake_link_txt(CHAIN_LINK, origin_ref="bench/link.txt")],
        backend_link_targets={"llama_cpu": ["bin/llama-bench"]})
    registrations = None
    if with_registrations:
        registrations = SU.SymbolRegistrationIndex(
            label="candidate",
            symbols_by_source={"ggml/src/ggml-cpu.c": ("ggml_mul_mat", "ggml_mul_mat_id")},
            registrations_by_symbol={
                "ggml_mul_mat": (SU.OpRegistration(op_name="MUL_MAT", backend="llama_cpu",
                                                   dispatch_predicate="cpu_supports_op"),),
                "ggml_mul_mat_id": (SU.OpRegistration(op_name="MUL_MAT_ID",
                                                      backend="llama_cpu",
                                                      dispatch_predicate="cpu_supports_op"),),
            })
    diff = SU.SourceDiff(
        base_commit=ANCHOR_COMMIT, candidate_commit="b" * 40,
        entries=tuple(SU.DiffEntry(path=p, change_kind="modified") for p in touched),
        origin_ref="git diff --name-status")
    return SU.derive_affected_surface(candidate_id=CANDIDATE, diff=diff, indexes=[index],
                                      registrations=registrations)


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


def anchor_build_log() -> str:
    """The ANCHOR build's OWN configure+build capture — a different file entirely.

    Its body is the same recorded cmake+make output as the candidate's, and that
    is the honest fixture: the anchor is the production tip built on this host
    with this toolchain, so it prints the same compiler identification and the
    same two warnings. What matters is that it is a SEPARATE capture in a
    separate file under the anchor tree, so that
    `check_static_and_compile`'s two cross-arm comparisons compare two
    measurements rather than one measurement with itself. Change the candidate's
    log alone — `TestANewWarningVersusTheAnchorIsVisible` does — and the delta
    fires; under the pre-red-team wiring it could not.
    """
    return ("=== configure: anchor (production-consolidated-v8)\n"
            + clean_configure_log()
            + "=== build: anchor (production-consolidated-v8)\n"
            + raw("recorded_build_success.log"))


#: A `GGML_SCHED_DEBUG=2` trace with two nodes on the CPU backend. Shaped to
#: `t0_provider._SPLIT_RE`/`_NODE_RE`, which are read off the real ggml printer.
SCHED_TRACE = (
    "## SPLIT #0: CPU # 2 inputs\n"
    "node #  0 (       MUL_MAT):        ffn_up-0 (  f32) [ CPU        assigned ]\n"
    "node #  1 (    MUL_MAT_ID):     ffn_moe_up-0 (  f32) [ CPU        assigned ]\n"
)


def _disposition(argv, exit_code: int, *, writable_root: str,
                 phase: str) -> WT.ProcessDisposition:
    """A disposition for a process that was NEVER LAUNCHED, and it says so.

    `pid=0`/`pgid=0` are not plausible pids: they are the sentinel that makes a
    reader of a chain-test artifact see immediately that no child existed. A
    real `run_build` fills these from the child it owned.
    """
    return WT.ProcessDisposition(
        argv=tuple(argv), pid=0, pgid=0, exit_code=exit_code, timed_out=False,
        signals_sent=(), verified_dead=True, duration_s=1.0,
        started_at="2026-08-03T23:00:00Z",
        sandbox_receipt={
            "sandbox_id": WT.process_sandbox.SANDBOX_ID,
            "writable_root": writable_root,
            "fixture_only": True,
            "phase": phase,
        },
        sandbox_teardown={
            "verified_empty": True,
            "removed": True,
            "fixture_only": True,
        })


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
        self.run_ledger = MB.CompletedRunLedger(
            J.Journal(os.path.join(root, "events"), campaign_id=CAMPAIGN),
            campaign_id=CAMPAIGN)
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
        disp = _disposition(
            plan.build_argv(), exit_code,
            writable_root=plan.build_dir.path, phase="build")
        conf = _disposition(
            plan.configure_argv(), 0,
            writable_root=plan.build_dir.path, phase="configure")
        return WT.BuildResult(
            plan=plan, configure=conf, build=disp, log_path=log_path,
            log_sha256=WT._sha256_text(log_text), facts=WT.parse_build_log(log_text),
            build_dir_pre_build_digest=pre, build_dir_created_for_this_build=True,
            load_average_at_start=None)

    def write_artifacts(self, plan: WT.BuildPlan, *, exports=None) -> dict:
        """The files a real build would have written.

        `libggml.so.0` is a REAL ELF64 with a real `.dynsym`, because
        `chain.symbol_evidence` reads it with `integrity.extract_elf_symbols` and
        that reader raises `ElfFormatError` on anything that is not one — an
        empty table diffs clean against everything, which is the fail-open the
        whole symbol gate exists to prevent. The other four are opaque bytes;
        nothing reads their contents, only their digests.
        """
        bin_dir = os.path.join(plan.build_dir.path, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        paths = {}
        for name, body in (("llama-cli", b"\x7fELF chain-candidate llama-cli\n"),
                           ("llama-bench", b"\x7fELF chain-candidate llama-bench\n"),
                           ("test-backend-ops", b"\x7fELF chain-candidate tbo\n"),
                           ("libggml.so.0",
                            build_elf64(list(CANDIDATE_EXPORTS if exports is None
                                             else exports))),
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
                           ("libggml.so.0", build_elf64(list(ANCHOR_EXPORTS)))):
            path = os.path.join(bin_dir, name)
            Path(path).write_bytes(body)
            out[name] = path
        # The ANCHOR's OWN configure+build log, in its own tree and its own file.
        # It is a separate capture and not the candidate's: `check_static_and_
        # compile` compares compiler identity and warning count ACROSS the two
        # arms, and until the 2026-08-04 red team this leg measured the "anchor"
        # toolchain off the candidate's log, so both comparisons were identities
        # and the gate PASSed on a self-comparison.
        # `anchor_toolchain_from_build_log` now refuses that wiring outright.
        log_path = os.path.join(root, "anchor-build.log")
        Path(log_path).write_text(anchor_build_log(), encoding="utf-8")
        out["build.log"] = log_path
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

    This is deliberately NOT a reusable campaign driver — `campaign.py` is the
    entrypoint and a second driver here would give the loop two spellings. It is
    the shortest composition that touches every seam, written so a test can reach
    into the middle of it. (Until 2026-08-04 the sentence above named
    `controller/state_machine.py` as the owner of the loop; that module is
    deleted, and the reason not to grow a second driver here is unchanged.)
    """

    def __init__(self, world: ChainWorld, *, anchor_source_commit=None,
                 build_exit_code: int = 0, claim: str = "acquire",
                 configure_log=None, build_dir=None,
                 candidate_effect=1.08) -> None:
        self.world = world
        self.anchor_source_commit = anchor_source_commit
        self.build_exit_code = build_exit_code
        self._claim_mode = claim
        self._configure_log = configure_log
        self._build_dir = build_dir
        self.claim = None
        self.claim_binding = None
        # The candidate's TRUE behaviour, for the whole declared budget. A float
        # is a constant factor on every block (the win fixture); a sequence is
        # one factor per block, indexed by the block's own `block_index`, which
        # is how a NULL candidate is expressed — its per-block factors straddle
        # 1.0, so the block signs are mixed rather than all positive.
        self.candidate_effect = candidate_effect
        self.pooled_blocks = None
        self.t1_extension_runs = ()

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
        self.build_log_text = log_text
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

    # -- 5. the anchor, bound once PER TOOL ---------------------------------
    def bind_anchor(self):
        """SEAM 3, and SEAM 7.

        Two bindings here, not one, and a third in `run_t1`. `llama-cli` is the
        tool T0 generates with; `libggml.so.0` is the tool the SYMBOL diff reads,
        and `api.AnchorIdentity.binary_sha256` is single-valued so it cannot name
        both. They are tied by `check_anchor_build_is_one_build`: same commit,
        same linkage.

        The anchor toolchain is MEASURED off the ANCHOR's own build log rather
        than typed. Without `compiler_id`/`compiler_version` on the capture,
        `collect_static_analysis` returns `None` and the static gate reads
        COULD_NOT_CHECK — README §6.1 item 3, which is closed by passing them.

        The log is `anchor_paths["build.log"]`, in the anchor tree, and NOT
        `self.build_log_text`, which is the candidate's. That was the wiring
        until the 2026-08-04 red team: one log on both sides of two cross-arm
        comparisons, so `static_and_compile_checks` PASSed on a self-comparison
        and neither the toolchain-mismatch branch nor the new-warning branch
        could ever fire. `anchor_toolchain_from_build_log` now takes the
        candidate's `BuildProvenance` and refuses that composition.
        """
        commit = self.anchor_source_commit or ANCHOR_COMMIT
        self.anchor_paths = self.world.anchor_tree()
        linkage = T0.sha256_text("anchor resolved library table")
        self.anchor_toolchain = chain.anchor_toolchain_from_build_log(
            Path(self.anchor_paths["build.log"]).read_text(encoding="utf-8"),
            log_ref=f"file://{self.anchor_paths['build.log']}",
            candidate_build=self.build_evidence.provenance)
        self.anchor_binding = chain.bind_anchor(T0.AnchorCapture(
            source_commit=commit,
            binary_sha256=T0.sha256_text("anchor llama-cli bytes"),
            linkage_sha256=linkage,
            output_digests=(T0.sha256_text("Paris."),), output_lengths=(6,),
            determinism_class="bitwise_stable", delivered_units=32,
            oracle_ids=("oracle://anchor-v8",),
            **self.anchor_toolchain.as_capture_kwargs()), tool="llama-cli")
        self.libggml_anchor = chain.bind_anchor(T0.AnchorCapture(
            source_commit=commit,
            binary_sha256=integrity.sha256_file(self.anchor_paths["libggml.so.0"]),
            linkage_sha256=linkage), tool="libggml.so.0")
        return self.anchor_binding

    # -- 6. T0 -------------------------------------------------------------
    def t0_evidence_inputs(self, *, diff_text=None):
        """SEAMS 5, 6 and 8 — the three producers §6.1 says exist and are unwired.

        Each returns a record `t0_provider` accepts as an input and never derives
        itself, and each is the reason one or two T0 surfaces stop reading
        COULD_NOT_CHECK. They are assembled here, in the reference composition,
        so tomorrow's session copies the wiring rather than the omission.
        """
        anchor_ops, anchor_predicates = registration_tables("anchor")
        cand_ops, cand_predicates = registration_tables("candidate")
        self.symbol_evidence = chain.symbol_evidence(
            anchor_binary=self.anchor_paths["libggml.so.0"],
            candidate_binary=self.artifacts["libggml.so.0"],
            anchor=self.libggml_anchor,
            declared=integrity.DeclaredSymbolDeltas(
                added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
            anchor_op_registrations=anchor_ops,
            candidate_op_registrations=cand_ops,
            anchor_dispatch_predicates=anchor_predicates,
            candidate_dispatch_predicates=cand_predicates)
        self.diff_evidence = chain.diff_policy_evidence(
            diff_text=CANDIDATE_DIFF if diff_text is None else diff_text,
            worktree_root=self.worktree.path.path,
            declared_surface_files=("ggml/src/ggml-cpu.c",),
            envelope=CHANGE_ENVELOPE,
            branch_name=self.worktree.branch.name,
            commit_argv=COMMIT_ARGV,
            record_schema_violations=())
        self.change_surface_evidence = chain.change_surface_from(
            chain_affected_surface(),
            diff_text=CANDIDATE_DIFF if diff_text is None else diff_text)
        return (self.symbol_evidence, self.diff_evidence, self.change_surface_evidence)

    def t0_plan(self):
        # SEAM 3 — the plan's paths come off the receipt, not off a literal.
        candidate = chain.candidate_build_for(
            self.identity, test_backend_ops=self.artifacts["test-backend-ops"])
        symbols, diff, surface_evidence = self.t0_evidence_inputs()
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
            # The derivation says this change reaches a dispatch predicate, so
            # `check_unseen_boundary_shapes` is a REAL gate now and FAILs
            # outright without a holdout: "a dispatch change validated only on
            # shapes it was written against is an overfit, not a kernel". Before
            # the surface was wired the same candidate got COULD_NOT_CHECK here.
            holdout=T0.HoldoutPlan(
                unseen_case_filter="unseen", boundary_case_filter="boundary",
                selection_rule_id="ak-holdout/v1", selection_seed="ak-chain-seed-0001",
                visible_to_planner=False),
            determinism_runs=2, cache_state="cold", state_safety_probe=False,
            oracle_ids=("oracle://anchor-v8",),
            candidate_diff_text=CANDIDATE_DIFF,
            build=self.build_evidence.provenance,
            **chain.t0_plan_evidence(
                symbols=symbols, diff=diff, change_surface=surface_evidence),
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

        captures = [
            cap(ops.argv, stdout=op_suite_text),
            cap(trace.argv, stdout=SCHED_TRACE + "Paris.", stderr=perf),
            cap(link.argv, stdout=self.linkage_text.replace(
                plan.candidate.test_backend_ops, plan.candidate.binary)),
            cap(gen.argv, stdout="Paris.", stderr=perf),
        ]
        if plan.holdout is not None:
            for case_filter, label in ((plan.holdout.unseen_case_filter, "unseen"),
                                       (plan.holdout.boundary_case_filter, "boundary")):
                held = T0.build_backend_ops_invocation(
                    binary=plan.candidate.test_backend_ops,
                    library_path=plan.candidate.library_path,
                    backend_filter=plan.op_suite.backend_filter, ops=plan.op_suite.ops,
                    base_env=plan.base_env, params_filter=case_filter)
                captures.append(cap(held.argv, stdout=_held_out_ops(label)))
        return T0.RecordedProcessRunner(captures)

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
            change_class="parameter", anchor_tier="T0", transfer_ratio_to=(),
            created_at="2026-08-03T23:00:00Z",
            campaign_controls=ChainCampaign.get()[0],
            calibration=ChainCampaign.get()[4])
        kwargs.update(overrides)
        return api.EvaluationRequest(**kwargs)

    # -- 7. T1 — the SAME claim, through the other Protocol -----------------
    def run_t1(self, *, factor=None, blocks: int = 5, attempt: int = 0):
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
        #
        # Copied with `replace`, NOT rebuilt field by field: a field-by-field copy
        # silently dropped `tool`, and an identity that no longer says which binary
        # its digest came off is not the same identity — `identity_matches` reads it
        # as COULD_NOT_CHECK against the binding it was copied from. That is the
        # composition this seam exists to catch, so the reference leg has to show
        # the copy that keeps the name.
        self.t1_anchor_identity = dataclasses.replace(
            self.t1_anchor.identity, measurement_event_ids=("ake-chain-anchor-0001",))

        plan = MB.MicrobenchPlan(
            recipe_id=BENCH_RECIPE_ID, candidate_id=CANDIDATE,
            campaign_seed=CAMPAIGN_SEED,
            candidate_binding=self.candidate_binding,
            anchor_binding=self.anchor_binding_tool,
            anchor=self.t1_anchor_identity,
            params={"model": FIXTURE_MODEL, "n_gen": 128, "reps": 10,
                    "output_format": "json"},
            base_blocks=blocks, pairs_per_block=1, unit_ids=("chain-unit-0",),
            attempt=attempt)
        self.t1_base_plan = plan
        self.t1_run = self._spawn_t1(plan, factor=factor)
        return self.t1_run

    def _spawn_t1(self, plan, *, factor=None):
        """Replay one run of `plan`. `factor` overrides the leg's own effect.

        A per-block effect is keyed by `(arm, invocation_index)` rather than by
        arm alone, because a constant per-arm response makes every block
        identical and therefore every block sign identical — which is the shape
        of a WIN and cannot express a null. `RecordedSpawner` already resolves
        `(arm, index)` before `arm`; this is that seam's first user.
        """
        effect = self.candidate_effect if factor is None else factor
        spawner = MB.RecordedSpawner(
            bench_responses(effect, first_block=plan.block_index_offset,
                            blocks=plan.blocks_to_run))
        self.microbench_spawner = spawner
        runner = MB.MicrobenchRunner(
            claim=self.claim_binding.microbench_claim, policy=HEALTHY_POLICY,
            spawner=spawner, host_state=HostStateStub(healthy_host_state()),
            run_ledger=self.world.run_ledger)
        return runner.run(plan)

    # -- 7b. the DECLARED extension round, same claim, same schedule ---------
    def run_t1_extension(self, *, round_index: int = 1, factor=None):
        """One declared extension round, planned off the base plan.

        The authorization is built from the CAMPAIGN — the same
        `CampaignStatistics` the reduction runs under, which is where the
        committed rule and the calibrated `B_min` both live — so a round this
        campaign did not declare cannot be planned here at all, and a licence
        another campaign issued cannot be pooled into this one's record.
        """
        stats = ChainCampaign.get()[5]
        authorization = MB.ExtensionAuthorization(campaign=stats,
                                                  round_index=round_index)
        self.t1_extension_plan = self.t1_base_plan.extend(authorization)
        self.t1_extension_run = self._spawn_t1(self.t1_extension_plan, factor=factor)
        return self.t1_extension_run

    # -- 7c. the DECLARED BUDGET, pooled into one block sequence -------------
    def extend_and_pool(self, *, factor=None):
        """Run every round the campaign DECLARED and pool them with the base.

        This stage is why the leg banks anything at all, and it is a stage
        rather than a line inside `run_t1` because it is the thing an operator
        following the runbook must not skip. §6.3's arithmetic: `B_min = 5`, the
        sign-martingale over five same-sign blocks tops out at `e = 5.5687`, and
        the threshold is 10. A leg that stops at `run_t1` reduces to
        `evidence_below_threshold` for EVERY candidate at EVERY true effect, and
        that reads as "no candidate was good enough" when it is really "the
        instrument cannot resolve a win at all".

        Until 2026-08-04 this class — the reference composition §2 Step 7 tells
        tomorrow's session to copy — did exactly that: `walk()` ended at
        `run_t1()` + `reduce()`, so the whole chain demonstrated a green,
        seventeen-gate, five-control, fully-attested leg that banked
        `evidence_below_threshold`. The runbook's Step 6 had been fixed to run
        the round; the composition it points at had not.

        The rounds are taken from the RULE (`extension.max_rounds`), never from
        a literal, so a campaign that declares a different budget runs a
        different number of rounds here without an edit. Pooling goes through
        `MB.assemble_run_blocks(..., campaign=...)`, which is the seam that
        refuses a round licensed by another campaign.
        """
        stats = ChainCampaign.get()[5]
        runs = []
        for round_index in range(1, stats.stopping_rule.extension.max_rounds + 1):
            runs.append(self.run_t1_extension(round_index=round_index, factor=factor))
        self.t1_extension_runs = tuple(runs)
        self.pooled_blocks = MB.assemble_run_blocks(
            self.t1_run, runs, campaign=stats, run_ledger=self.world.run_ledger)
        return self.pooled_blocks

    # -- 8. reduce ----------------------------------------------------------
    def reduce(self, blocks=None):
        """Reduce the POOLED budget when one exists, the base segment otherwise.

        The default is `self.pooled_blocks`, not `self.t1_run.paired_blocks()`.
        A leg that ran its declared rounds and then reduced only its base
        segment would discard the evidence it spent the claim on, and the
        discard would be invisible — the reduction is admissible either way and
        differs only in an e-value that cannot cross.
        """
        stats = ChainCampaign.get()[5]
        self.reducer = statistics.PairedBlockReducer(stats)
        self.t1_request = self.evaluation_request(
            tier="T1", event_id="ake-chain-0002",
            anchor=self.t1_anchor_identity,
            metric="tokens_per_second", metric_direction="higher_better")
        if blocks is None:
            blocks = (self.t1_run.paired_blocks() if self.pooled_blocks is None
                      else self.pooled_blocks)
        self.reduction = self.reducer.reduce(
            self.t1_request, blocks,
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

    # -- 10. the controller banked or abandoned — REMOVED 2026-08-04 ---------
    #
    # `ChainLeg.bank()` walked `controller/state_machine.py`'s
    # `SELECT_TARGET -> PROPOSE -> PRE_RUN_CRITIC -> MUTATE -> BUILD -> T0_GATE ->
    # T1_SEARCH_EVAL -> POST_RUN_CRITIC -> BANK_EVENT` over the leg's own verdict,
    # and three tests asserted the machine ended at `BANK_EVENT`. That plane was
    # deleted with the rest of AK4 on the operator's approval, so the stage is gone
    # rather than stubbed: a stage that walks nothing would leave three green
    # assertions about a machine that does not exist.
    #
    # WHAT THAT COST, stated rather than quietly absorbed — it is coverage this
    # file no longer has and nothing else in the package replaces:
    #
    #   * that a verdict produced by this chain is ACCEPTED by a controller at
    #     `BANK_EVENT` — the seam between the execution layer's output and whatever
    #     durably records it. There is no controller now; when campaign #1 grows
    #     one, this is the seam to re-assert.
    #   * that a T0 FAILURE takes the other documented edge (`T0_GATE ->
    #     POST_RUN_CRITIC`) and is still banked, as a failure — *"compilation
    #     failures are valuable outcomes"*. Nothing here asserts that any more. The
    #     T0-failure path itself is still covered (`TestACcacheBuildIsNotACleanBuild`,
    #     `TestAnArtifactThatIsNotTheBuiltOne`, `TestTheWiredProducersRefuse...`);
    #     what is lost is that a failed candidate is RECORDED rather than dropped.
    #   * the FOURTH anchor shape. `state_machine.AnchorIdentity` keyed its digests
    #     BY BACKEND, which is the per-key table `api.AnchorIdentity` does not
    #     have; building it from the same capture was what kept the controller's
    #     anchor and the record's anchor one anchor. That is `chain.SEAM_NOTES`
    #     item 2, and it is now an unexercised note.
    #
    # Everything else in the walk — claim, worktree, build, T0, blocks, controls,
    # verdict, teardown, production-unchanged — is untouched: none of it went
    # through the machine.

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
    #:
    #: `extend` sits between `t1` and `reduce` because that is where it sits in
    #: a real campaign: the base segment is measured, the declared rounds are
    #: measured under the same claim and the same schedule, and only the pooled
    #: sequence is reduced. There is no stage that reduces the base segment on
    #: its own, because no such reduction can bank anything (§6.3).
    STAGES = ("claim", "worktree", "build", "artifact", "anchor", "t0",
              "t1", "extend", "reduce", "controls", "dispatch")

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
            "t1": self.run_t1, "extend": self.extend_and_pool,
            "reduce": self.reduce,
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
            # The DECLARED budget, not the base segment. Removing this line
            # makes every candidate in this file resolve
            # `evidence_below_threshold` and is what
            # `TestAWinIsReachableAndANullIsRefused` bites on.
            self.extend_and_pool()
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


def _held_out_ops(label: str) -> str:
    """A `-p <filter>` run's console output, in the RECORDED grammar.

    Two cases per op so `reconcile()` has something to cross-check: that method
    is what caught the builder's own finding that a `-p` shape filter can empty a
    run while the tool still prints `OK`.
    """
    return ("Testing 1 devices\n\nBackend 1/1: CPU\n"
            "  Device description: AMD EPYC 9655 96-Core Processor\n\n"
            f"  MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=1,{label}=1): OK\n"
            f"  MUL_MAT_ID(type_a=f32,type_b=f32,n_mats=1,n_used=1,{label}=1): OK\n"
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
        reps = len(row["samples_ts"])
        row.update({
            "autokernel_hardened": True,
            "autokernel_output_invariant": True,
            "autokernel_input_working_set_bytes": 1 << 30,
            "autokernel_input_hashes": ",".join(
                f"{index + 1:016x}" for index in range(reps)),
            "autokernel_input_addresses": ",".join(
                f"0x{0x1000 + 2 * index:x}/0x{0x1001 + 2 * index:x}"
                for index in range(reps)),
            "autokernel_context_addresses": ",".join(
                f"0x{0x4000 + 2 * index:x}/0x{0x4001 + 2 * index:x}"
                for index in range(reps)),
            "autokernel_output_hashes": ",".join(
                f"{index + 101:016x}/{index + 101:016x}" for index in range(reps)),
            "autokernel_hybrid_ab_complete": True,
            "autokernel_unsynchronized_samples_ns": ",".join(
                str(value) for value in row["samples_ns"]),
            "autokernel_thread_set_stable": True,
            "autokernel_escape_checks_complete": True,
            "autokernel_thread_set_hashes": ",".join(
                "/".join([f"{index + 201:016x}"] * 4) for index in range(reps)),
            "autokernel_device_sync_mode": "cpu_not_applicable",
        })
    return json.dumps(rows)


def bench_responses(effect, *, first_block: int, blocks: int) -> dict:
    """`RecordedSpawner` responses for one run, with a PER-BLOCK candidate arm.

    `effect` is a constant factor or a sequence indexed by GLOBAL block index —
    so the extension round, whose `block_index_offset` is `B_min`, reads the
    same sequence at the same offsets the reducer will later see. That is what
    makes a null candidate expressible: a constant factor makes every block's
    effect identical and therefore every block SIGN identical, which is the
    shape of a win at any magnitude and cannot be the shape of a null.

    Both invocation slots of each block are filled for both arms. Which of the
    two the candidate occupies is decided by the block's `order`, which is
    derived from the campaign seed and is not knowable here; filling both and
    letting `RecordedSpawner`'s `(arm, index)` lookup pick is the only way to
    key on the block without re-deriving the order schedule in a fixture.
    """
    anchor_payload = scaled_bench(factor=1.0)
    out: dict = {}
    for i in range(blocks):
        index = first_block + i
        factor = effect[index] if isinstance(effect, (list, tuple)) else float(effect)
        payload = scaled_bench(factor=factor, build_commit="cafe12345")
        for call in (2 * i, 2 * i + 1):
            out[(MB.ARM_CANDIDATE, call)] = payload
            out[(MB.ARM_ANCHOR, call)] = anchor_payload
    return out


def null_effect(*, seed: int, blocks: int, sigma: float = 0.01) -> tuple:
    """A candidate with NO true effect: per-block factors centred exactly on 1.0.

    DERIVED and seeded, and named as such for the same reason `scaled_bench` is:
    it is not a capture. `sigma` is the calibration block's own per-sample noise
    (`_cal_blocks(noise=0.01)`), so the null candidate is as noisy as the A/A
    material the campaign's threshold was solved against — a quieter null would
    make the control easier than the campaign it is a control for.

    The mean is 1.0 exactly, not "about 1.0": the whole point of a null control
    is that the true effect is zero, so any crossing is manufactured rather than
    measured.
    """
    rng = random.Random(seed)
    return tuple(1.0 + rng.gauss(0.0, sigma) for _ in range(blocks))


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
            change_class="parameter",
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

    def test_exactly_five_t0_surfaces_are_explicitly_unevaluated(self):
        """The number `execution/README.md` tells tomorrow's session to expect.

        Not a vanity assertion: the runbook tells the session what a healthy
        report looks like, and a reader who sees a different shape has to know
        whether the campaign is broken or the runbook is stale. When a producer
        is wired the count moves and this test fails, which is the reminder to
        update §3 and §6.1 of the runbook.

        It was 8 PASS / 9 COULD_NOT_CHECK until 2026-08-04. Five surfaces moved
        when `chain.symbol_evidence`, `chain.diff_policy_evidence`,
        `chain.anchor_toolchain_from_build_log` and `chain.change_surface_from`
        were wired into the leg above. On 2026-08-10 the projection-side refusal
        channel stopped evaporating, revealing two more honestly unevaluated
        gates. On 2026-08-11 exported-symbol version coverage became conditional:
        an unversioned exported surface is complete, while a genuinely versioned
        export still reports the named gap. The five are all named findings,
        not surfaces nobody got to:

          * `exact_reference_comparison` — this historical recorded fixture
            predates the current instrument's `AK_REF_V1` positive receipt. A
            current-instrument run projects the observed comparator metric and
            tolerance; an old capture remains honestly uncovered.
          * `sanitizer.asan` / `sanitizer.ubsan` — the derivation determined
            neither memory nor threading for THIS candidate, and the behavioural
            classifier can only answer True or undetermined. A candidate that
            does touch memory gets a real gate: see
            `TestTheBehaviouralClassifierOnlyWidens`.
          * `state_rollback_teardown_race` — no rollback probe exists, so no
            state-safety measurement can pass at all
            (`t0_provider.STATE_SAFETY_CANNOT_PASS`).
          * `affected_surface_reconciliation` — the pure source classifier can
            widen memory/thread/state touches but cannot prove their absence.
        """
        report = self.leg.t0_report
        unproduced = sorted(g.gate_id for g in report.gates
                            if g.check.outcome == schemas.COULD_NOT_CHECK)
        self.assertEqual(unproduced, sorted([
            correctness.GID_SURFACE_RECONCILIATION,
            correctness.GID_ASAN,
            correctness.GID_EXACT_REFERENCE,
            correctness.GID_STATE_SAFETY,
            correctness.GID_UBSAN,
        ]), "the set of T0 surfaces with no producer has changed — update "
           "execution/README.md §3 and §6.1, which tell tomorrow's session what "
           "a healthy report looks like")
        passed = [g.gate_id for g in report.gates if g.check.outcome == schemas.PASS]
        self.assertEqual(len(passed), 12, sorted(passed))

    def test_the_wired_surfaces_are_gates_and_not_assertions(self):
        """Each outcome below is a comparison, including projection refusals.

        The negative side of every one of them is in
        `TestTheWiredProducersRefuseCleanShapedNothing` and
        `TestTheBehaviouralClassifierOnlyWidens`.
        """
        report = self.leg.t0_report
        for gate_id in (correctness.GID_SEMANTIC_DIFF, correctness.GID_SCHEMA_DIFF_POLICY,
                        correctness.GID_STATIC_COMPILE,
                        correctness.GID_BOUNDARY_SHAPES):
            self.assertEqual(report.outcome(gate_id), schemas.PASS,
                             report.gate(gate_id).check.reasons)
        self.assertEqual(report.outcome(correctness.GID_SYMBOLS), schemas.PASS)
        for gate_id, finding in (
                (correctness.GID_SURFACE_RECONCILIATION, "projection change_surface."),):
            self.assertEqual(report.outcome(gate_id), schemas.COULD_NOT_CHECK)
            self.assertTrue(any(finding in reason
                                for reason in report.gate(gate_id).check.reasons),
                            report.gate(gate_id).check.reasons)

    def test_plan_evidence_helper_keeps_records_and_refusals_together(self):
        fields = chain.t0_plan_evidence(
            symbols=self.leg.symbol_evidence,
            diff=self.leg.diff_evidence,
            change_surface=self.leg.change_surface_evidence)
        self.assertEqual(set(fields), {
            "symbols", "diff", "change_surface", "projection_checks"})
        self.assertIs(fields["symbols"], self.leg.symbol_evidence.diff)
        self.assertTrue(fields["projection_checks"])

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

    def test_the_identity_every_consumer_reads_names_the_tool_it_was_bound_for(self):
        """SEAM 3. The tool is on the BINDING; it must survive to the record.

        `bind_anchor(tool=…)` has always taken the tool, but `.identity` used to
        drop it, so the object T0's request, T1's plan and the journalled line all
        read carried no trace of which binary its single digest came off.
        """
        self.assertEqual(self.leg.anchor_binding.identity.tool, "llama-cli")
        self.assertEqual(self.leg.t1_anchor.identity.tool, "llama-bench")
        self.assertEqual(self.leg.t1_request.anchor.tool, "llama-bench")
        self.assertTrue(self.leg.t1_request.anchor.short().startswith("llama-bench:"))
        self.assertEqual(self.leg.request.anchor.tool, "llama-cli")

    def test_one_capture_bound_for_two_tools_is_two_anchors(self):
        """The composition that used to pass silently, and its compliant control.

        Deriving both stages' identity from ONE capture makes every digest agree
        by construction — which is exactly why the tool has to be part of the
        comparison. Nothing else in the triple can tell these two apart.
        """
        capture = self.leg.anchor_binding.capture
        as_cli = chain.bind_anchor(capture, tool="llama-cli")
        as_bench = chain.bind_anchor(capture, tool="llama-bench")
        self.assertEqual(as_cli.identity.binary_sha256, as_bench.identity.binary_sha256)

        consumers = {"the evaluation request": as_cli.identity}
        check = chain.check_anchor_matches(as_bench, consumers=consumers)
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        self.assertIn("anchor.tool differs", " ".join(check.reasons))
        with self.assertRaises(chain.AnchorNotOneAnchor):
            chain.require_anchor_matches(as_bench, consumers=consumers)

        # Compliant path: one tool's consumers reading that tool's anchor.
        self.assertEqual(
            chain.check_anchor_matches(
                as_bench, consumers={"the evaluation request": as_bench.identity}).outcome,
            schemas.PASS)
        # And the cross-tool tie is unaffected — two tools of ONE build still tie.
        self.assertEqual(
            chain.check_anchor_build_is_one_build([as_cli, as_bench]).outcome, schemas.PASS)

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

    # `test_the_controller_banks_the_event` stood here until 2026-08-04. It walked
    # the deleted `controller/state_machine.py` and asserted the machine reached
    # `BANK_EVENT` over this leg's verdict. See the note at ChainLeg stage 10 for
    # what its removal costs.

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
        forged_dir = f"{self.FROZEN}/build"
        forged_receipts = tuple(
            {**row,
             "activation": {**row["activation"], "writable_root": forged_dir}}
            for row in leg.identity.sandbox_receipts)
        forged = dataclasses.replace(
            leg.identity, build_dir=forged_dir,
            sandbox_receipts=forged_receipts)
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
                MB.ARM_ANCHOR: scaled_bench(factor=1.0)}),
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


class TestTheExtensionRoundHasAProducer(_ChainCase):
    """The gap that BLOCKED the first campaign, closed and held closed.

    THE FACT this class exists for: the calibrated threshold for this cell is 10
    and the sign-martingale e-value over B_min=5 same-sign blocks tops out at
    5.5687 — the statistic is the SIGN of each block's effect, so the magnitude
    never enters and a candidate at a true factor of 3.0 returns exactly what
    one at 1.08 returns. Nothing crosses on the base segment. Every win comes
    from the declared extension round, which is why the runner not producing one
    meant a campaign that accumulates `evidence_below_threshold` forever and
    looks like "no candidate was good enough".

    The schedule decision, argued from the code in `microbench`'s module
    docstring: the extension EXTENDS the base segment's `OrderSchedule` rather
    than re-deriving one. `test_the_extension_orders_are_the_reversed_base_orders`
    is what that means observably, and
    `test_a_re_derived_schedule_is_refused_not_relabelled` is the other one being
    refused.
    """

    def setUp(self):
        super().setUp()
        self.stats = ChainCampaign.get()[5]

    # -- the fact ---------------------------------------------------------
    def test_the_base_segment_alone_cannot_cross_at_any_true_effect(self):
        """The blocker, reproduced: sign-based evidence caps out below 10."""
        self.assertEqual(self.stats.b_min, 5)
        self.assertEqual(self.stats.threshold_for(api.STRATUM_SELECTION), 10.0)
        leg = ChainLeg(self.world)
        leg.up_to("t0")
        values = []
        for index, factor in enumerate((1.08, 1.3, 3.0)):
            # These are three independent mathematical fixtures, not three
            # attempts in one campaign. Keep the same schedule so magnitude is
            # the only changing variable, but give each fixture its own journal.
            self.world.run_ledger = MB.CompletedRunLedger(
                J.Journal(os.path.join(self.world.root, f"effect-fixture-{index}"),
                          campaign_id=CAMPAIGN), campaign_id=CAMPAIGN)
            leg.run_t1(factor=factor)
            values.append(leg.reduce().estimate.e_value)
        for value in values:
            self.assertAlmostEqual(value, 5.56875, places=5)
            self.assertLess(value, 10.0)
        self.assertEqual(len(set(values)), 1,
                         "the construction is sign-based; the magnitude must not enter")

    # -- the producer -----------------------------------------------------
    def test_the_runner_emits_a_declared_extension_round(self):
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        blocks = leg.run_t1_extension().paired_blocks()
        self.assertEqual(len(blocks), self.stats.stopping_rule.extension.blocks_per_round)
        self.assertEqual([b.block_index for b in blocks], [5, 6, 7, 8, 9])
        for block in blocks:
            self.assertEqual(block.segment, statistics.SEGMENT_EXTENSION)
            self.assertEqual(block.extension_round, 1)

    def test_the_extension_orders_are_the_reversed_base_orders(self):
        """THE DECISION, observable: extended, so `order_for`'s reversed limb runs.

        Under a RE-DERIVED schedule the round would be asked for indices 0..4 and
        would repeat the base orders exactly — which `BoundedExtension` cannot
        even declare, since it accepts no `order` but `"reversed"`.
        """
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        base = [b.order for b in leg.t1_run.paired_blocks()]
        extension = [b.order for b in leg.run_t1_extension().paired_blocks()]
        self.assertEqual(len(base), len(extension))
        for got, base_order in zip(extension, base):
            self.assertNotEqual(got, base_order)
        self.assertEqual(leg.t1_extension_plan.schedule(), leg.t1_base_plan.schedule(),
                         "the extension must run the SAME schedule object the base ran")

    def test_the_extension_round_ran_under_the_same_held_claim(self):
        """Denial 8 does not lapse because the blocks are a continuation."""
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        run = leg.run_t1_extension()
        self.assertTrue(run.claim_attestations)
        for attestation in run.claim_attestations:
            self.assertTrue(attestation.held)
            self.assertEqual(attestation.check.outcome, schemas.PASS)

    # -- the win ----------------------------------------------------------
    def test_a_candidate_with_a_real_effect_now_crosses_the_threshold(self):
        """THE POINT. Pooled to the PRE-DECLARED threshold, the same candidate crosses."""
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        base_only = leg.reduce().estimate
        self.assertLess(base_only.e_value, base_only.threshold)

        leg.run_t1_extension()
        pooled = MB.assemble_run_blocks(leg.t1_run, [leg.t1_extension_run],
                                       campaign=self.stats,
                                       run_ledger=leg.world.run_ledger)
        self.assertEqual(len(pooled), 10)
        reduction = leg.reduce(blocks=pooled)
        self.assertEqual(reduction.admissible.outcome, schemas.PASS,
                         reduction.admissible.reasons)
        estimate = reduction.estimate
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.paired_blocks, 10)
        self.assertGreaterEqual(estimate.e_value, estimate.threshold)
        self.assertAlmostEqual(estimate.e_value, 42.2876953125, places=5)

    def test_the_reducer_reads_the_pooled_set_as_base_then_whole_rounds(self):
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        leg.run_t1_extension()
        pooled = MB.assemble_run_blocks(leg.t1_run, [leg.t1_extension_run],
                                       campaign=self.stats,
                                       run_ledger=leg.world.run_ledger)
        checks = dict(leg.reduce(blocks=pooled).checks)
        for name in ("order_control", "extension_structure", "block_identity",
                     "block_count"):
            self.assertEqual(checks[name].outcome, schemas.PASS, checks[name].reasons)

    def test_the_stopping_rule_replays_to_a_crossing_on_the_pooled_blocks(self):
        """The rule's own replay, not just the e-value: outcome and decision.

        The e-value crossing is necessary and not sufficient — what banks a win
        is the pre-committed rule REPLAYED over the realized blocks returning
        `evidence_threshold_crossed`, at a block count the rule licenses.
        """
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        leg.run_t1_extension()
        pooled = MB.assemble_run_blocks(leg.t1_run, [leg.t1_extension_run],
                                       campaign=self.stats,
                                       run_ledger=leg.world.run_ledger)
        evaluation = self.stats.sequential_evaluation(
            candidate_id=CANDIDATE, stratum=api.STRATUM_SELECTION,
            metric_direction="higher_better")
        for block in pooled:
            request = evaluation.next_block_request()
            self.assertEqual(request.block_index, block.block_index)
            self.assertEqual(request.order, block.order)
            self.assertEqual(request.segment, block.segment)
            self.assertEqual(request.extension_round, block.extension_round)
            look = evaluation.submit_block(block)
            if look.terminal:
                break
        decision = evaluation.decide()
        self.assertEqual(decision.outcome, "evidence_threshold_crossed")
        self.assertEqual(decision.decision, "compose_into_champion_lineage")
        self.assertEqual(decision.extension_rounds_used, 1)
        self.assertTrue(decision.crossed)

    # -- the refusals -----------------------------------------------------
    def test_an_undeclared_second_round_cannot_be_planned(self):
        """`max_rounds=1`: round 2 is not a longer run, it is a different rule."""
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        self.assertEqual(self.stats.stopping_rule.extension.max_rounds, 1)
        with self.assertRaises(MB.ExtensionNotDeclared):
            leg.run_t1_extension(round_index=2)

    def test_a_rule_mutated_after_the_commitment_cannot_authorize_a_round(self):
        """The caller granting itself an extension after seeing the base segment.

        The mutated rule cannot become a campaign at all — `CampaignStatistics`
        verifies its own commitment — and there is no other object that licenses
        a round, so `max_rounds=3` has nowhere to be stated.
        """
        rule = self.stats.stopping_rule
        greedier = dataclasses.replace(
            rule, extension=statistics.BoundedExtension(max_rounds=3, blocks_per_round=5))
        with self.assertRaises(statistics.StoppingRuleMutated):
            dataclasses.replace(self.stats, stopping_rule=greedier)
        with self.assertRaises(MB.ExtensionNotDeclared):
            MB.ExtensionAuthorization(campaign=self.stats, round_index=2)

    def test_a_round_this_campaign_did_not_license_cannot_be_pooled_into_it(self):
        """A second campaign is buildable; a second campaign's licence is not usable.

        THE BITE for the pooling-seam half of the 2026-08-04 red team. The
        forged campaign declares the SAME rule shape, so nothing about the
        round's blocks — index line, orders, segment, round number — is
        different; only the commitment it was licensed under is.
        """
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        forged = statistics.CampaignStatistics(
            campaign_id="not-even-this-campaign", campaign_seed=CAMPAIGN_SEED,
            effect_scale=self.stats.effect_scale, hypothesis=self.stats.hypothesis,
            margin=self.stats.margin,
            stopping_rule=dataclasses.replace(self.stats.stopping_rule,
                                              rule_id="ak-stop-attacker/v9"),
            stopping_rule_commitment=statistics.StoppingRuleCommitment.commit(
                dataclasses.replace(self.stats.stopping_rule,
                                    rule_id="ak-stop-attacker/v9"),
                campaign_id="not-even-this-campaign",
                committed_at="2099-01-01T00:00:00+00:00"),
            split_rule=self.stats.split_rule, construction=self.stats.construction,
            calibration=self.stats.calibration,
            aa_effect_pool=self.stats.aa_effect_pool,
            anchor_calibration_values=self.stats.anchor_calibration_values)
        run = leg._spawn_t1(
            leg.t1_base_plan.extend(
                MB.ExtensionAuthorization(campaign=forged, round_index=1)),
            factor=1.08)
        self.assertTrue(run.complete, run.refusals)
        with self.assertRaises(MB.ExtensionNotDeclared) as caught:
            MB.assemble_run_blocks(leg.t1_run, [run], campaign=self.stats,
                                   run_ledger=leg.world.run_ledger)
        self.assertIn("not-even-this-campaign", str(caught.exception))

    def test_a_re_derived_schedule_is_refused_not_relabelled(self):
        """The OTHER answer to the schedule question, refused as a hard error."""
        leg = ChainLeg(self.world)
        leg.up_to("t1")
        authorization = MB.ExtensionAuthorization(campaign=self.stats, round_index=1)
        re_derived = dataclasses.replace(
            leg.t1_base_plan, base_blocks=self.stats.b_min + 5)
        with self.assertRaises(MB.ScheduleMismatch):
            re_derived.extend(authorization)

    def test_the_runbook_step_that_runs_t1_runs_the_extension_round(self):
        """THE BITE for the composition: there is no other driver.

        `assemble_run_blocks`, `ExtensionAuthorization` and `MicrobenchPlan.extend`
        have ZERO non-test callers — grep the package. The producer's only driver
        is §2 of `README.md`, and Step 6 used to end at `run.paired_blocks()`
        with no mention of an extension round anywhere in it. An operator
        following the runbook verbatim would run five blocks, read
        `e = 5.5687 < 10`, and bank nothing — which is precisely the failure
        §6.3 exists to prevent, arrived at by following the instructions.
        """
        readme = (_HERE.parent / "README.md").read_text(encoding="utf-8")
        start = readme.index("### Step 6 — T1")
        step6 = readme[start:readme.index("### Step 7", start)]
        for token in ("ExtensionAuthorization", "assemble_run_blocks",
                      "campaign=campaign", "sequential_evaluation"):
            self.assertIn(token, step6,
                          f"runbook Step 6 never names {token}; the extension producer "
                          f"has no other caller than this procedure")
        self.assertNotIn("blocks = run.paired_blocks()", step6,
                         "Step 6 still ends at the base segment, which cannot cross")

    def test_the_far_side_was_ready_all_along(self):
        """The compliant-path control kept from the pin this class replaces."""
        self.assertEqual(self.stats.stopping_rule.extension.blocks_per_round, 5)
        self.assertEqual(self.stats.stopping_rule.extension.max_rounds, 1)
        base = statistics.OrderSchedule.derive(
            campaign_seed=CAMPAIGN_SEED, candidate_id=CANDIDATE,
            base_blocks=self.stats.b_min, attempt=0)
        plans = MB.plan_blocks(base, count=5, pairs=1, unit_ids=("u",),
                               stratum=api.STRATUM_SELECTION,
                               segment=statistics.SEGMENT_EXTENSION, extension_round=1)
        self.assertEqual([p.block_index for p in plans], [5, 6, 7, 8, 9])
        self.assertTrue(all(p.segment == statistics.SEGMENT_EXTENSION for p in plans))


# =============================================================================
# F2. THE DELIVERABLE — a win is reachable, and a null is still refused
# =============================================================================

#: The three seeds used as null candidates below, and what each one is FOR.
#: Named rather than inlined because a null control chosen after seeing its
#: result is not a control — these are fixed here, and the crossing-rate test
#: further down is what says they are not a lucky draw.
NULL_SEED_MIXED = 101        #: signs mixed from block 1; e never leaves 1.0's neighbourhood
NULL_SEED_BASE_CEILING = 303 #: five same-sign blocks first — a null that MAXES the base segment
NULL_SEED_SECOND = 202       #: a second independent draw, so the first is not the argument


class TestAWinIsReachableAndANullIsRefused(_ChainCase):
    """THE DELIVERABLE. Both directions, end to end, with no inference.

    Everything in this file was green before this class existed and a campaign
    driven by it could still never bank anything, because the thing that was
    green was a leg that stopped at the base segment. §6.3 is the arithmetic:
    `B_min = 5`, the sign-martingale over five same-sign blocks tops out at
    `e = 5.5687`, the calibrated threshold is 10, and the statistic is the SIGN
    of each block so the magnitude never enters. A candidate at a true +8% and a
    candidate at a true +200% both return 5.5687 and both resolve
    `evidence_below_threshold`.

    That failure mode is expensive precisely because it looks like a result:
    every gate PASSes, all five controls score, the record grammar completes —
    and the verdict says "no candidate was good enough" when the truth is "the
    instrument cannot resolve a win at all".

    The two halves here are both required. A change that makes wins reachable
    and also makes nulls reachable has not improved the instrument, it has
    removed it.

      * `test_a_real_effect_crosses_and_is_BANKED_as_an_improvement`
      * `test_a_null_candidate_does_not_cross_and_is_not_ranked`

    Neither uses a live process. The candidate arm is `scaled_bench`, the
    recorded `llama-bench` sample vector scaled by a stated factor, replayed
    through `MB.RecordedSpawner`; the anchor arm is the capture verbatim.
    """

    def setUp(self):
        super().setUp()
        self.stats = ChainCampaign.get()[5]

    # -- the composition itself ------------------------------------------
    def test_the_reference_composition_runs_the_DECLARED_BUDGET_not_B_min(self):
        """THE BITE for the seam. `ChainLeg` is what §2 Step 7 says to copy.

        Until 2026-08-04 `walk()` ran `run_t1()` and reduced its five blocks.
        The runbook's Step 6 had already been corrected to run the declared
        round — §6.3 — but the reference composition the runbook points at for
        Steps 7 and 8 had not, so the two disagreed and the one that is
        executable was the one that banks nothing.

        Deleting `self.extend_and_pool()` from `walk()` fails this test, the two
        below it, and `TestTheChainFits.test_the_dispatcher_...` is unaffected —
        which is the point: nothing that was previously asserted could see it.
        """
        self.assertIn("extend", ChainLeg.STAGES)
        self.assertEqual(ChainLeg.STAGES.index("extend"),
                         ChainLeg.STAGES.index("t1") + 1)
        leg = ChainLeg(self.world).walk()
        declared = (self.stats.b_min
                    + self.stats.stopping_rule.extension.max_rounds
                    * self.stats.stopping_rule.extension.blocks_per_round)
        self.assertEqual(len(leg.pooled_blocks), declared)
        self.assertEqual(len(leg.reduction.blocks), declared,
                         "the leg measured the declared budget and then reduced "
                         "only part of it")
        self.assertEqual(len(leg.t1_extension_runs),
                         self.stats.stopping_rule.extension.max_rounds)

    # -- the win ----------------------------------------------------------
    def test_a_real_effect_crosses_and_is_BANKED_as_an_improvement(self):
        """A +8% candidate, walked once, banked as a ranked improvement.

        Every number below is reproducible from the fixtures in this file and
        none of them is a measurement of any kernel:

            base segment (5 blocks)   e = 5.568750  < 10   evidence_below_threshold
            declared budget (10)      e = 42.287695 >= 10  improvement, RANKED
            first crossing            block 7
            rule replay               evidence_threshold_crossed
                                      -> compose_into_champion_lineage
        """
        leg = ChainLeg(self.world, candidate_effect=1.08).walk()

        # What the base segment alone would have said, from the SAME blocks.
        base_only = leg.reducer.reduce(
            leg.t1_request, leg.t1_run.paired_blocks(),
            raw_samples_ref="ak-raw://chain/base-only")
        self.assertAlmostEqual(base_only.estimate.e_value, 5.56875, places=5)
        self.assertEqual(api._resolve_effect(base_only.estimate),
                         api.EFFECT_EVIDENCE_BELOW_THRESHOLD)

        estimate = leg.reduction.estimate
        self.assertEqual(leg.reduction.admissible.outcome, schemas.PASS,
                         leg.reduction.admissible.reasons)
        self.assertEqual(estimate.paired_blocks, 10)
        self.assertAlmostEqual(estimate.e_value, 42.2876953125, places=5)
        self.assertGreaterEqual(estimate.e_value, estimate.threshold)
        self.assertEqual(leg.reduction.e_process.first_crossing_block, 7)

        # The VERDICT, not just the e-value: a crossing that does not become a
        # rankable improvement has not banked anything either.
        verdict = leg.outcome.verdict
        self.assertEqual(verdict.status, "pass")
        self.assertEqual(verdict.effect_resolution, api.EFFECT_IMPROVEMENT)
        self.assertTrue(verdict.speed_rank_admissible)
        self.assertEqual(leg.outcome.grammar_complete.outcome, schemas.PASS,
                         leg.outcome.grammar_complete.reasons)

        # And the RULE's own replay over the realized blocks, which is what
        # licenses composing the candidate into the champion lineage.
        decision = self._replay(leg.pooled_blocks)
        self.assertEqual(decision.outcome, "evidence_threshold_crossed")
        self.assertEqual(decision.decision, "compose_into_champion_lineage")
        self.assertTrue(decision.crossed)
        self.assertEqual(decision.extension_rounds_used, 1)
        # The BANKED half of this test — `leg.bank(...)` reaching `BANK_EVENT` —
        # went with `controller/state_machine.py` on 2026-08-04. What the win
        # produces is still asserted above, in full; what is no longer asserted is
        # that anything durably accepts it. See ChainLeg stage 10.

    # -- the null ---------------------------------------------------------
    def test_a_null_candidate_does_not_cross_and_is_not_ranked(self):
        """The control. A candidate with NO true effect, through the same walk.

        `null_effect` centres the per-block factors on 1.0 exactly, at the
        calibration block's own noise (sigma = 0.01), so the block signs are
        mixed rather than all positive. Nothing else about the leg differs from
        the win above — same claim, same schedule, same declared budget, same
        seventeen T0 gates, same five controls.

            declared budget (10)   e = 0.900000 < 10, and no PREFIX ever crossed
            resolution             below_noise_floor -> NOT rankable
            rule replay            extension_exhausted -> abandon
        """
        leg = ChainLeg(self.world,
                       candidate_effect=null_effect(seed=NULL_SEED_MIXED,
                                                    blocks=10)).walk()
        estimate = leg.reduction.estimate
        self.assertEqual(leg.reduction.admissible.outcome, schemas.PASS,
                         leg.reduction.admissible.reasons)
        self.assertEqual(estimate.paired_blocks, 10)
        self.assertAlmostEqual(estimate.e_value, 0.9, places=6)
        self.assertLess(estimate.e_value, estimate.threshold)
        # `e_value` is the running MAXIMUM, so this is the statement that no
        # prefix of the run crossed — a null that crosses at block 6 and falls
        # back would still have banked under a rule that looks after each block.
        self.assertIsNone(leg.reduction.e_process.first_crossing_block)

        verdict = leg.outcome.verdict
        self.assertEqual(verdict.status, "pass")
        self.assertNotIn(verdict.effect_resolution,
                         (api.EFFECT_IMPROVEMENT, api.EFFECT_REGRESSION))
        self.assertFalse(verdict.speed_rank_admissible)

        decision = self._replay(leg.pooled_blocks)
        self.assertEqual(decision.outcome, "extension_exhausted")
        self.assertEqual(decision.decision, "abandon")
        self.assertFalse(decision.crossed)
        # "A null is BANKED too — as an abandon" was asserted here through the
        # controller and is gone with it. The abandon DECISION is still asserted
        # (three lines up); that it is recorded is not. See ChainLeg stage 10.

    def test_the_hardest_null_reaches_the_BASE_CEILING_and_still_does_not_cross(self):
        """A null whose first five blocks are all same-sign. It maxes the base.

        This is the null that would have been mistaken for a win by anyone
        reading `e = 5.5687` as "nearly there": 5.5687 is not a near miss, it is
        what a fair coin returns on five flips, and this null gets it with a
        TRUE EFFECT OF ZERO. The declared round then does not carry it over —
        which is the whole reason the extension is declared in advance rather
        than granted after the base segment is read (§6.3).
        """
        effect = null_effect(seed=NULL_SEED_BASE_CEILING, blocks=10)
        leg = ChainLeg(self.world, candidate_effect=effect)
        leg.up_to("reduce")
        signs = leg.reduction.e_process.signs
        self.assertEqual(signs[:5], (1.0,) * 5,
                         "this seed is chosen because its base segment is "
                         "all same-sign; if that stops being true the control "
                         "is no longer the hard case it claims to be")
        base_only = leg.reducer.reduce(
            leg.t1_request, leg.t1_run.paired_blocks(),
            raw_samples_ref="ak-raw://chain/null-base-only")
        self.assertAlmostEqual(base_only.estimate.e_value, 5.56875, places=5)
        self.assertAlmostEqual(leg.reduction.estimate.e_value, 5.56875, places=5)
        self.assertLess(leg.reduction.estimate.e_value,
                        leg.reduction.estimate.threshold)
        self.assertIsNone(leg.reduction.e_process.first_crossing_block)
        self.assertEqual(self._replay(leg.pooled_blocks).decision, "abandon")

    def test_a_second_independent_null_draw_is_also_refused(self):
        """One null is an anecdote. The rate test below is the general claim."""
        leg = ChainLeg(self.world,
                       candidate_effect=null_effect(seed=NULL_SEED_SECOND, blocks=10))
        leg.up_to("reduce")
        self.assertAlmostEqual(leg.reduction.estimate.e_value, 1.1, places=6)
        self.assertIsNone(leg.reduction.e_process.first_crossing_block)
        self.assertEqual(self._replay(leg.pooled_blocks).decision, "abandon")

    def test_the_null_crossing_rate_at_the_declared_budget_holds_villes_bound(self):
        """The claim the three legs above cannot make: nulls cross at <= alpha.

        A null CAN cross — ten same-sign blocks from a fair coin is 2^-10 — and
        a test asserting "a null never crosses" would be false and would have to
        be switched off the first time it fired. What is true, and is what the
        threshold means, is that the crossing rate is bounded by `1/threshold`
        = alpha = 0.1 at EVERY horizon, including running the campaign's whole
        declared budget and looking after every block.

        Run against `statistics.run_e_process` directly rather than through the
        reducer: the reducer solves an MDE per call, which is 10 000x the cost
        and answers a different question. This is the e-process core, which is
        the thing whose bound is being checked.
        """
        construction = statistics.select_construction(
            "sign_martingale_predictable_lambda/v1")
        rng = random.Random(20260804)
        alpha = 1.0 / 10.0
        for blocks, ceiling in ((5, alpha), (10, alpha),
                                (self.stats.stopping_rule.max_blocks_per_candidate,
                                 alpha)):
            crossed = 0
            trials = 4000
            for _ in range(trials):
                oriented = [rng.gauss(0.0, 0.01) for _ in range(blocks)]
                run = statistics.run_e_process(
                    oriented, construction=construction,
                    hypothesis=self.stats.hypothesis, margin=self.stats.margin,
                    threshold=self.stats.threshold_for(api.STRATUM_SELECTION))
                if run.first_crossing_block is not None:
                    crossed += 1
            rate = crossed / trials
            self.assertLessEqual(rate, ceiling,
                                 f"null crossing rate {rate:.4f} at {blocks} blocks "
                                 f"exceeds the declared alpha {ceiling}")
        # The compliant-path control, in the same construction: a REAL effect
        # does cross, and often. Without it this test passes on a construction
        # that never crosses at all.
        crossed = sum(
            1 for _ in range(200)
            if statistics.run_e_process(
                [abs(rng.gauss(0.08, 0.01)) for _ in range(10)],
                construction=construction, hypothesis=self.stats.hypothesis,
                margin=self.stats.margin,
                threshold=self.stats.threshold_for(api.STRATUM_SELECTION)
            ).first_crossing_block is not None)
        self.assertEqual(crossed, 200,
                         "a same-sign 10-block run must cross; if it does not, "
                         "the bound above is vacuous")

    def test_the_pooled_records_MDE_describes_a_window_the_RULE_can_license(self):
        """The defect the pooled reduction made live, and its bite.

        `solve_mde`'s `block_count` is the BASE SEGMENT: it replays the rule with
        `b_min=block_count` and draws its windows at
        `rule.max_total_blocks(block_count)`, which ADDS the declared extension
        budget on top. `PairedBlockReducer.reduce` used to hand it `len(blocks)`,
        which is the REALIZED count — the two coincide only while every run stops
        at `B_min`, which is exactly what every run in this file did until the
        extension round got a producer.

        With a pooled 10-block run the old call asked for the MDE of a 15-block
        window, and `max_rounds = 1` means this campaign cannot license 15 blocks
        for one candidate at all:

            mde_for(b_min=5)   window 10   selection 0.008584   confirmation 0.013294
            mde_for(len=10)    window 15   selection 0.006972   confirmation 0.007511
                                                 -18.8%              -43.5%

        The direction overstates. `api._resolve_effect` reads `magnitude < mde`
        as `no_detectable_difference`, so an understated MDE admits effects the
        run could not resolve — and if the e-value crosses, they are RANKED.
        This is the first place in the package where a real 10-block pooled
        reduction exists, so it is where the seam became checkable.
        """
        leg = ChainLeg(self.world).walk()
        rule = self.stats.stopping_rule
        licensed = rule.max_total_blocks(self.stats.b_min)
        self.assertEqual(licensed, len(leg.pooled_blocks),
                         "the leg ran exactly the window the rule licenses")

        published = leg.reduction.mde
        self.assertEqual(published.window_length, licensed)
        declared = leg.reducer.mde_for(self.stats.b_min,
                                       stratum=api.STRATUM_SELECTION,
                                       metric_direction="higher_better")
        self.assertEqual(published.value, declared.value)

        # And the number the old call site would have published, for a window
        # this rule cannot license.
        realized = leg.reducer.mde_for(len(leg.pooled_blocks),
                                       stratum=api.STRATUM_SELECTION,
                                       metric_direction="higher_better")
        self.assertEqual(realized.window_length, licensed + 5)
        self.assertLess(realized.value, published.value,
                        "if these are equal the assertion above is vacuous")
        self.assertEqual(leg.reduction.check("mde_window").outcome, schemas.PASS)

    def test_the_runbook_step_7_points_at_a_composition_that_pools(self):
        """§2 Step 7 delegates to `ChainLeg`. It must say WHICH stages that is.

        The bite for the documentation half: Step 7 used to name the controls,
        the dispatch and the controller walk and not the pooling, so a reader
        following it composed `run_t1 -> reduce` and never learned that the
        stage between them exists.
        """
        readme = (_HERE.parent / "README.md").read_text(encoding="utf-8")
        start = readme.index("### Step 7")
        step7 = readme[start:readme.index("### Step 8", start)]
        for token in ("extend_and_pool", "pooled"):
            self.assertIn(token, step7,
                          f"runbook Step 7 never names {token}; it delegates the "
                          "composition to ChainLeg without saying that the leg "
                          "reduces the POOLED budget")

    # -- helper -----------------------------------------------------------
    def _replay(self, blocks):
        """The pre-committed rule, replayed over the realized blocks."""
        evaluation = self.stats.sequential_evaluation(
            candidate_id=CANDIDATE, stratum=api.STRATUM_SELECTION,
            metric_direction="higher_better")
        for block in blocks:
            evaluation.next_block_request()
            if evaluation.submit_block(block).terminal:
                break
        return evaluation.decide()


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

    def test_llama_cpp_is_still_on_the_v9_freeze_commit(self):
        fp = fingerprint_frozen()["/mnt/raid0/llm/llama.cpp"]
        if fp is None:
            self.skipTest("the frozen llama.cpp clone is not present on this host")
        self.assertEqual(fp["branch"], "production-consolidated-v9")
        self.assertEqual(fp["head"], "0db32c06e3e550065b78311a6031ef3dd2c4f27c")


# =============================================================================
# G. The four producers wired for §6.1 — every one with the bite that pins it
# =============================================================================

class TestTheWiredProducersRefuseCleanShapedNothing(_ChainCase):
    """Each producer's PASS above is a comparison. Here is each one failing.

    The trap this whole section guards is that a producer emitting PASS for a
    surface it did not evaluate is strictly worse than the COULD_NOT_CHECK it
    replaced: COULD_NOT_CHECK is honest and a false PASS is not. So every
    producer either measures the thing or refuses to produce a record at all.
    """

    def setUp(self):
        super().setUp()
        self.leg = ChainLeg(self.world).up_to("anchor")

    def _tables(self):
        return registration_tables("anchor"), registration_tables("candidate")

    def _symbols(self, *, candidate_exports=None, declared=None, sources=None,
                 anchor_binding=None):
        (a_ops, a_pred), (c_ops, c_pred) = self._tables()
        if sources is not None:
            c_ops, c_pred = registration_tables("candidate", sources=sources)
        candidate_path = os.path.join(self._tmp.name, "candidate-libggml.so.0")
        Path(candidate_path).write_bytes(build_elf64(
            list(CANDIDATE_EXPORTS if candidate_exports is None else candidate_exports)))
        return chain.symbol_evidence(
            anchor_binary=self.leg.anchor_paths["libggml.so.0"],
            candidate_binary=candidate_path,
            anchor=anchor_binding or self.leg.libggml_anchor,
            declared=declared or integrity.DeclaredSymbolDeltas(
                added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
            anchor_op_registrations=a_ops, candidate_op_registrations=c_ops,
            anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=c_pred)

    def _gate(self, evidence):
        return correctness.check_symbol_and_registration_preservation(
            self.leg.evaluation_request(), evidence.diff, _t0_policy())

    # -- symbols ----------------------------------------------------------
    def test_an_undeclared_symbol_removal_fails_the_gate(self):
        evidence = self._symbols(candidate_exports=ANCHOR_EXPORTS[:-1])
        self.assertIn("_ZN4ggml6detail15kernel_dispatchILi4EEEvPKfPfi",
                      evidence.diff.removed_symbols)
        gate = self._gate(evidence)
        self.assertEqual(gate.check.outcome, schemas.FAIL, gate.check.reasons)

    def test_a_removal_declared_by_QUALIFIED_name_is_declared(self):
        """The join the far side cannot make: it has no demangler.

        `DeclaredSymbolDeltas.covers` matches `ggml::detail::kernel_dispatch`
        against the mangled name; `check_symbol_and_registration_preservation`
        does a plain set difference. Handing it the raw declaration would FAIL
        every honestly-declared removal, and a gate that fails on correct input
        is a gate that gets switched off. `_declared_covering` resolves it.
        """
        evidence = self._symbols(
            candidate_exports=ANCHOR_EXPORTS[:-1],
            declared=integrity.DeclaredSymbolDeltas(
                added=frozenset(), removed=frozenset({"ggml::detail::kernel_dispatch"}),
                arity_changed=frozenset()))
        self.assertEqual(evidence.diff.removed_symbols, evidence.diff.declared_removals)
        self.assertEqual(self._gate(evidence).check.outcome, schemas.PASS)

    def test_an_undeclared_ADDITION_is_not_a_failure(self):
        """§8.5.1 makes only removal and arity change hard (invariant 18)."""
        evidence = self._symbols()
        self.assertIn("ggml_mul_mat_id_avx512", evidence.diff.added_symbols)
        self.assertEqual(evidence.diff.removed_symbols, ())
        self.assertEqual(self._gate(evidence).check.outcome, schemas.PASS)

    def test_a_removed_op_registration_fails_the_gate(self):
        """The half a pure ELF diff cannot see: a registration is data, not a symbol."""
        thinned = {"ggml/src/ggml-cpu.c": (
            "GGML_CPU_OP(MUL_MAT, 2)\nCPU_SUPPORTS(MUL_MAT)\nCPU_SUPPORTS(MUL_MAT_ID)\n")}
        evidence = self._symbols(sources=thinned)
        self.assertEqual(evidence.diff.removed_op_registrations,
                         ("ggml_backend_cpu:MUL_MAT_ID",))
        gate = self._gate(evidence)
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("op registration" in r for r in gate.check.reasons))

    def test_a_registration_table_is_mandatory_and_none_is_refused(self):
        (a_ops, a_pred), (c_ops, c_pred) = self._tables()
        with self.assertRaises(TypeError) as ctx:
            chain.symbol_evidence(
                anchor_binary=self.leg.anchor_paths["libggml.so.0"],
                candidate_binary=self.leg.artifacts["libggml.so.0"],
                anchor=self.leg.libggml_anchor,
                declared=integrity.DeclaredSymbolDeltas(
                    added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
                anchor_op_registrations=a_ops, candidate_op_registrations=c_ops,
                anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=None)
        self.assertIn("never run", str(ctx.exception))

    def test_iqk_parameter_adapter_scans_both_real_source_roots(self):
        hooks = (
            "if (ggml_iqk_try_mul_mat(params, dst)) return;\n"
            "if (ggml_iqk_try_mul_mat_id(params, dst)) return;\n")
        predicates = (
            "switch (type) { case GGML_TYPE_Q4_K: break; "
            "case GGML_TYPE_IQ4_XS: break; }\n")
        for root in (self.leg.anchor_paths["root"], self.leg.worktree.path.path):
            cpu = Path(root, "ggml", "src", "ggml-cpu")
            (cpu / "iqk").mkdir(parents=True)
            (cpu / "ggml-cpu.c").write_text(hooks, encoding="utf-8")
            (cpu / "iqk" / "iqk_dispatch.cpp").write_text(
                predicates, encoding="utf-8")
        evidence = chain.iqk_parameter_symbol_evidence(
            anchor_binary=self.leg.anchor_paths["libggml.so.0"],
            candidate_binary=self.leg.artifacts["libggml.so.0"],
            anchor=self.leg.libggml_anchor,
            proposal={"declared_symbol_deltas": {
                "added": [], "removed": [], "arity_changed": []}},
            anchor_root=self.leg.anchor_paths["root"],
            candidate_root=self.leg.worktree.path.path)
        self.assertTrue(evidence.op_registration_diff.anchor_count)
        self.assertEqual(evidence.op_registration_diff.anchor_count, 2)
        self.assertEqual(evidence.dispatch_predicate_diff.anchor_count, 2)
        self.assertEqual(self._gate(evidence).check.outcome, schemas.PASS)

    def test_iqk_parameter_adapter_refuses_a_missing_ggml_source_universe(self):
        with self.assertRaisesRegex(chain.EmptyAnchorSurface, "source root is missing"):
            chain.iqk_parameter_symbol_evidence(
                anchor_binary=self.leg.anchor_paths["libggml.so.0"],
                candidate_binary=self.leg.artifacts["libggml.so.0"],
                anchor=self.leg.libggml_anchor,
                proposal={"declared_symbol_deltas": {
                    "added": [], "removed": [], "arity_changed": []}},
                anchor_root=self.leg.anchor_paths["root"],
                candidate_root=self.leg.worktree.path.path)

    def test_a_dispatch_table_diffed_as_an_op_table_is_refused(self):
        (a_ops, a_pred), (_c_ops, c_pred) = self._tables()
        with self.assertRaises(ValueError) as ctx:
            chain.symbol_evidence(
                anchor_binary=self.leg.anchor_paths["libggml.so.0"],
                candidate_binary=self.leg.artifacts["libggml.so.0"],
                anchor=self.leg.libggml_anchor,
                declared=integrity.DeclaredSymbolDeltas(
                    added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
                anchor_op_registrations=a_ops, candidate_op_registrations=c_pred,
                anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=c_pred)
        self.assertIn("two different registries", str(ctx.exception))

    def test_an_anchor_that_exports_nothing_is_refused_not_diffed_clean(self):
        empty = os.path.join(self._tmp.name, "empty-libggml.so.0")
        Path(empty).write_bytes(build_elf64([elf_fn("hidden", vis="HIDDEN")]))
        binding = chain.bind_anchor(T0.AnchorCapture(
            source_commit=ANCHOR_COMMIT,
            binary_sha256=integrity.sha256_file(empty),
            linkage_sha256=self.leg.libggml_anchor.capture.linkage_sha256),
            tool="libggml.so.0")
        (a_ops, a_pred), (c_ops, c_pred) = self._tables()
        with self.assertRaises(chain.EmptyAnchorSurface):
            chain.symbol_evidence(
                anchor_binary=empty, candidate_binary=self.leg.artifacts["libggml.so.0"],
                anchor=binding,
                declared=integrity.DeclaredSymbolDeltas(
                    added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
                anchor_op_registrations=a_ops, candidate_op_registrations=c_ops,
                anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=c_pred)

    def test_diffing_against_a_binary_the_binding_did_not_measure_is_refused(self):
        """SymbolTableDiff carries no anchor triple, so this is the only place to ask."""
        with self.assertRaises(chain.AnchorNotOneAnchor) as ctx:
            self._symbols(anchor_binding=self.leg.anchor_binding)   # the llama-cli one
        self.assertIn("bind_anchor", str(ctx.exception))

    def test_a_stripped_binary_raises_rather_than_diffing_clean(self):
        stripped = os.path.join(self._tmp.name, "stripped.so")
        Path(stripped).write_bytes(b"not an elf at all")
        (a_ops, a_pred), (c_ops, c_pred) = self._tables()
        with self.assertRaises(integrity.ElfFormatError):
            chain.symbol_evidence(
                anchor_binary=self.leg.anchor_paths["libggml.so.0"],
                candidate_binary=stripped, anchor=self.leg.libggml_anchor,
                declared=integrity.DeclaredSymbolDeltas(
                    added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
                anchor_op_registrations=a_ops, candidate_op_registrations=c_ops,
                anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=c_pred)

    # -- the diff ---------------------------------------------------------
    def _diff(self, **overrides):
        kwargs = dict(diff_text=CANDIDATE_DIFF, worktree_root=self.leg.worktree.path.path,
                      declared_surface_files=("ggml/src/ggml-cpu.c",),
                      envelope=CHANGE_ENVELOPE, branch_name=self.leg.worktree.branch.name,
                      commit_argv=COMMIT_ARGV, record_schema_violations=())
        kwargs.update(overrides)
        return chain.diff_policy_evidence(**kwargs)

    def test_a_bare_commit_is_not_pathspec_limited_and_fails_the_policy_gate(self):
        evidence = self._diff(commit_argv=("git", "commit", "-m", "kernel tweak"))
        self.assertFalse(evidence.policy.commit_was_pathspec_limited)
        gate, _review = correctness.check_schema_and_diff_policy(
            evidence.policy,
            chain.change_surface_from(chain_affected_surface(),
                                      diff_text=CANDIDATE_DIFF).surface,
            _t0_policy())
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("pathspec-limited" in r for r in gate.check.reasons))

    def test_empty_parameter_diff_needs_no_fabricated_commit(self):
        evidence = self._diff(
            diff_text="", declared_surface_files=(), commit_argv=(),
            envelope=correctness.ChangeClassEnvelope(
                change_class="parameter", max_changed_lines=1, max_files_touched=1))
        self.assertTrue(evidence.policy.commit_was_pathspec_limited)
        self.assertEqual(evidence.policy.changed_lines, 0)
        self.assertEqual(evidence.policy.files_touched, ())
        self.assertIn("no source commit exists", " ".join(evidence.notes))

    def test_dash_a_defeats_a_pathspec_and_is_read_as_such(self):
        for argv in (("git", "commit", "-a", "-m", "x", "--", "ggml/src/ggml-cpu.c"),
                     ("git", "commit", "-am", "x", "--", "ggml/src/ggml-cpu.c"),
                     ("git", "commit", "--all", "-m", "x", "--", "a")):
            limited, reason = chain.commit_was_pathspec_limited(argv)
            self.assertFalse(limited, argv)
            self.assertIn("commits more than the pathspec", reason)

    def test_dash_i_include_also_defeats_a_pathspec_and_is_read_as_such(self):
        """The 2026-08-04 red team's find: `-i` read as pathspec-limited.

        `git commit -i -- <paths>` means *"stage these paths IN ADDITION TO
        whatever is already staged"* (git-commit(1)); the default for a pathspec
        commit is `--only`, which disregards the index. So `-i` is the one
        spelling under which another session's staged files ride into the
        artifact WITH a pathspec present — precisely the hazard
        `commit_was_pathspec_limited` exists to catch — and it returned True.
        """
        for argv in (("git", "commit", "-i", "-m", "x", "--", "ggml/src/ggml-cpu.c"),
                     ("git", "commit", "--include", "-m", "x", "--", "ggml/src/ggml-cpu.c"),
                     ("git", "commit", "-im", "x", "--", "ggml/src/ggml-cpu.c")):
            limited, reason = chain.commit_was_pathspec_limited(argv)
            self.assertFalse(limited, argv)
            self.assertIn("commits more than the pathspec", reason)

    def test_the_compliant_commit_is_the_control(self):
        limited, reason = chain.commit_was_pathspec_limited(COMMIT_ARGV)
        self.assertTrue(limited, reason)

    def test_the_flag_scan_stops_at_the_separator_so_a_pathspec_is_a_filename(self):
        """A compliant-path control for the widened scan: `-i` AFTER `--` is a file."""
        limited, _reason = chain.commit_was_pathspec_limited(
            ("git", "commit", "-m", "x", "--", "ggml/src/-i.c"))
        self.assertTrue(limited)

    def test_a_file_outside_the_declared_surface_fails_semantic_conformance(self):
        evidence = self._diff(declared_surface_files=("ggml/src/ggml-quants.c",))
        gate = correctness.check_semantic_diff_conformance(evidence.policy)
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("outside the declared surface" in r for r in gate.check.reasons))

    def test_a_diff_over_its_class_envelope_fails(self):
        tight = correctness.ChangeClassEnvelope(change_class="arithmetic",
                                                max_changed_lines=1, max_files_touched=10)
        gate = correctness.check_semantic_diff_conformance(self._diff(envelope=tight).policy)
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("envelope" in r for r in gate.check.reasons))

    def test_a_traversal_out_of_the_worktree_into_a_frozen_tree_is_MEASURED(self):
        """A diff path is repo-relative, so the gate's own absolute-path test never fires.

        `production_tree_paths` is resolved against the worktree here, which is
        the only place the escape is visible at all.
        """
        escape = CANDIDATE_DIFF.replace(
            "ggml/src/ggml-cpu.c",
            os.path.relpath("/mnt/raid0/llm/llama.cpp/ggml/src/ggml.c",
                            self.leg.worktree.path.path))
        evidence = self._diff(diff_text=escape, declared_surface_files=())
        self.assertTrue(evidence.policy.production_tree_paths,
                        "the traversal out of the worktree was not measured")
        gate, _review = correctness.check_schema_and_diff_policy(
            evidence.policy,
            chain.change_surface_from(chain_affected_surface(),
                                      diff_text=CANDIDATE_DIFF).surface,
            _t0_policy())
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("denial 2" in r for r in gate.check.reasons))

    def test_a_production_named_branch_fails(self):
        gate, _review = correctness.check_schema_and_diff_policy(
            self._diff(branch_name="production-consolidated-v9").policy,
            chain.change_surface_from(chain_affected_surface(),
                                      diff_text=CANDIDATE_DIFF).surface,
            _t0_policy())
        self.assertEqual(gate.check.outcome, schemas.FAIL)

    def test_the_integrity_envelope_class_is_refused_where_correctness_is_required(self):
        with self.assertRaises(TypeError) as ctx:
            self._diff(envelope=integrity.ChangeClassEnvelope(
                change_class="arithmetic", max_changed_lines=400, max_files_touched=10,
                max_hunks=20, max_file_shrinkage_ratio=0.5, allows_file_creation=False,
                allows_file_deletion=False, allows_pure_deletion_hunks=False,
                declared_by="ak-chain-policy/v1"))
        self.assertIn("DIFFERENT class with the same name", str(ctx.exception))

    # -- the anchor toolchain ---------------------------------------------
    #: The candidate build record every call below has to carry, so that the log
    #: it is handed can be told apart from the candidate's own. See
    #: `TestTheAnchorToolchainIsMeasuredOffTheANCHORSLog`.
    def _candidate_build(self):
        return self.leg.build_evidence.provenance

    def test_the_anchor_toolchain_is_measured_from_the_log_not_typed(self):
        toolchain = chain.anchor_toolchain_from_build_log(
            anchor_build_log(), log_ref="file:///anchor/build.log",
            candidate_build=self._candidate_build())
        self.assertTrue(toolchain.compiler_id.startswith(("CXX", "C ", "HIP", "CUDA")))
        self.assertRegex(toolchain.compiler_version, r"^\d")
        self.assertGreaterEqual(toolchain.warning_count, 0)

    def test_a_log_with_no_compiler_identification_refuses_rather_than_guessing(self):
        with self.assertRaises(chain.BuildProvenanceUnprojectable) as ctx:
            chain.anchor_toolchain_from_build_log(
                "=== build: make\n[ 50%] Building CXX object x.o\n",
                log_ref="file:///anchor/cached.log",
                candidate_build=self._candidate_build())
        self.assertIn("CONFIGURE time only", str(ctx.exception))

    def test_an_empty_anchor_log_is_refused(self):
        with self.assertRaises(chain.BuildProvenanceUnprojectable) as ctx:
            chain.anchor_toolchain_from_build_log(
                "   ", log_ref="file:///anchor/empty.log",
                candidate_build=self._candidate_build())
        self.assertIn("strongest possible baseline", str(ctx.exception))


class TestTheBehaviouralClassifierOnlyWidens(unittest.TestCase):
    """§6.1 item 4 — and the one thing it must NEVER do.

    `derived_touches_memory=False` is what licenses `check_asan`'s PASS branch:
    *"ASAN/UBSAN is not mandatory for this change: the mechanical derivation
    finds it touches neither memory nor threading."* Nothing in this package can
    establish that, so the classifier is allowed to answer True or undetermined
    and nothing else. These tests are the enforcement.
    """

    def test_no_diff_can_make_a_behavioural_flag_False(self):
        """Swept over every fixture diff plus an empty one, not spot-checked."""
        for text in (CANDIDATE_DIFF, MEMORY_TOUCHING_DIFF, "", "diff --git a/x b/x\n"):
            for name, (flag, _matched) in chain.classify_behavioural_surface(text).items():
                self.assertIn(flag, (True, None), f"{name} answered False for {text[:40]!r}")

    def test_the_source_carries_no_False_branch_for_the_three_flags(self):
        """Structural, so a later edit cannot reintroduce the PASS branch quietly."""
        source = Path(chain.__file__).read_text(encoding="utf-8")
        self.assertIn("True if matched else None", source)
        for literal in ("derived_touches_memory=False", "derived_touches_threading=False",
                        "derived_touches_persistent_state=False"):
            self.assertNotIn(literal, source)

    def test_a_memory_touching_diff_makes_the_sanitizer_surface_MANDATORY(self):
        evidence = chain.change_surface_from(chain_affected_surface(),
                                             diff_text=MEMORY_TOUCHING_DIFF)
        self.assertIs(evidence.surface.derived_touches_memory, True)
        self.assertIs(evidence.surface.sanitizers_mandatory, True)

    def test_and_the_gate_then_FAILs_without_a_sanitizer_run_instead_of_shrugging(self):
        """The whole point: COULD_NOT_CHECK becomes a real, speed-blocking FAIL."""
        surface = chain.change_surface_from(chain_affected_surface(),
                                            diff_text=MEMORY_TOUCHING_DIFF).surface
        request = _request_for_surface()
        for gate in (correctness.check_asan(request, None, surface),
                     correctness.check_ubsan(request, None, surface)):
            self.assertEqual(gate.check.outcome, schemas.FAIL, gate.check.reasons)
            self.assertTrue(any("MANDATORY" in r for r in gate.check.reasons))

    def test_a_diff_with_no_behavioural_token_stays_UNDETERMINED(self):
        """The honest half. Not a PASS, and the reason says why."""
        surface = chain.change_surface_from(chain_affected_surface(),
                                            diff_text=CANDIDATE_DIFF).surface
        self.assertIsNone(surface.derived_touches_memory)
        self.assertIsNone(surface.sanitizers_mandatory)
        gate = correctness.check_asan(_request_for_surface(), None, surface)
        self.assertEqual(gate.check.outcome, schemas.COULD_NOT_CHECK)

    def test_the_file_headers_are_not_scanned_for_tokens(self):
        """`--- a/ggml/src/cache.c` must not score as 'touches persistent state'."""
        text = ("diff --git a/ggml/src/cache.c b/ggml/src/cache.c\n"
                "--- a/ggml/src/cache.c\n+++ b/ggml/src/cache.c\n"
                "@@ -1,1 +1,1 @@\n-int a = 1;\n+int a = 2;\n")
        self.assertIsNone(chain.classify_behavioural_surface(text)["persistent_state"][0])

    def test_the_derived_ops_come_off_the_build_system_closure(self):
        evidence = chain.change_surface_from(chain_affected_surface(),
                                             diff_text=CANDIDATE_DIFF)
        self.assertEqual(sorted(evidence.surface.derived_ops), ["MUL_MAT", "MUL_MAT_ID"])
        self.assertIn("ggml/src/ggml-cpu.c", evidence.surface.derived_files)

    def test_dispatch_is_undetermined_when_no_symbol_index_was_supplied(self):
        """An empty dispatch-predicate tuple is not 'no dispatch predicates'."""
        evidence = chain.change_surface_from(
            chain_affected_surface(with_registrations=False), diff_text=CANDIDATE_DIFF)
        self.assertIsNone(evidence.surface.derived_touches_dispatch)
        self.assertTrue(any("no SymbolRegistrationIndex" in n for n in evidence.notes))

    def test_a_surface_derivation_is_required_and_a_lookalike_is_refused(self):
        with self.assertRaises(TypeError) as ctx:
            chain.change_surface_from({"op_names": ("MUL_MAT",)}, diff_text=CANDIDATE_DIFF)
        self.assertIn("declaration wearing a derivation's name", str(ctx.exception))

    def test_the_derivation_ref_names_both_halves_and_their_content(self):
        """A reader must be able to tell WHICH token table produced a True."""
        ref = chain.change_surface_from(chain_affected_surface(),
                                        diff_text=MEMORY_TOUCHING_DIFF).surface.derivation_ref
        self.assertIn("derive_affected_surface@", ref)
        self.assertIn("classify_behavioural_surface/v1@", ref)


class TestRealizedEditClassification(unittest.TestCase):
    """AK-X-7: realized repair work cannot masquerade as a declared rewrite."""

    def test_empty_and_comment_only_diffs_are_no_op(self):
        for text in ("", "diff --git a/x b/x\n@@ -1 +1 @@\n-// old\n+// new\n"):
            result = chain.classify_realized_edit(text)
            self.assertEqual(result["edit_type"], chain.EDIT_NO_OP)
            self.assertEqual(result["substantive_lines"], 0)

    def test_each_specific_repair_class_is_recognized(self):
        cases = {
            chain.EDIT_MASK_FIX: "-if (i < n)\n+if (i < n && valid_index(mask))\n",
            chain.EDIT_DELEGATED_OP: "-local_mm(x)\n+rocblas_gemm_ex(x)\n",
            chain.EDIT_DTYPE_CAST: "-x = y\n+x = static_cast<bf16>(y)\n",
            chain.EDIT_OPTIMIZATION_REWRITE: "-for (int i=0;i<n;i++) f(i);\n+fused_tile(n);\n",
        }
        for expected, text in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(chain.classify_realized_edit(text)["edit_type"], expected)

    def test_change_surface_capture_retains_the_counts(self):
        evidence = chain.change_surface_from(
            chain_affected_surface(),
            diff_text="-x = y;\n+x = static_cast<bf16>(y);\n")
        self.assertEqual(evidence.realized_edit["edit_type"], chain.EDIT_DTYPE_CAST)
        self.assertTrue(any("realized edit type dtype_cast" in note
                            for note in evidence.notes))


def _request_for_surface() -> api.EvaluationRequest:
    """The minimum request the sanitizer gates read: an artifact and an anchor."""
    sha = "c" * 64
    return api.EvaluationRequest(
        event_id="ake-surface-0001", campaign_id=CAMPAIGN, candidate_id=CANDIDATE,
        tier="T0", backend="llama_cpu", phase="decode",
        cell_class="operator_microbench", protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(source_sha256=sha, binary_sha256="d" * 64,
                                      linkage_sha256="e" * 64),
        anchor=api.AnchorIdentity(source_commit=ANCHOR_COMMIT, binary_sha256="f" * 64,
                                  linkage_sha256="a" * 64),
        evaluator=api.EvaluatorIdentity(id="ak-eval/v1", bundle_sha256=sha,
                                        runtime_source_label_ref="ref://surface"),
        scope_denominator=api.ScopeDenominator(machine_subset="full", numa_nodes=(),
                                               devices=(), cores=96),
        scope_manifest_sha256=sha, co_residency="single",
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=2),
        metric="tokens_per_second", metric_direction="higher_better", reps=10,
        change_class="parameter", anchor_tier="T0", transfer_ratio_to=(),
        created_at="2026-08-04T00:00:00Z", campaign_controls=None, calibration=None)


# =============================================================================
# H. The 2026-08-04 red team on §6.1 — four fixes, each with the bite that pins it
# =============================================================================

class TestADeclaredArityChangeDoesNotExcuseARemoval(_ChainCase):
    """§8.5.1's headline example, arriving through the gate written to catch it.

    `SymbolDiff.removed` is by construction a removal with NO matching addition:
    `symbol_evidence` partitions the removal/addition pairs out into
    `signature_changes` first. So a name in `removed_symbols` was DROPPED, and a
    proposal that declared *"I will change the arity of X"* declared something
    else. `_declared_covering` accepted `declared.arity_changed` for it until
    this test existed, the name landed in `declared_removals`, and the gate
    PASSed — byte-identically to the candidate that honestly declared the
    removal.

    Bite: revert `which=declared.removed` to the old
    `covers(removed) or covers(arity_changed)` and
    `test_the_undeclared_case_is_the_bite` FAILs with a PASS.
    """

    #: The anchor's ABI minus the template specialization: a pure removal.
    DROPPED = "_ZN4ggml6detail15kernel_dispatchILi4EEEvPKfPfi"
    QUALIFIED = "ggml::detail::kernel_dispatch"

    def setUp(self):
        super().setUp()
        self.leg = ChainLeg(self.world).up_to("anchor")

    def _evidence(self, declared):
        (a_ops, a_pred), (c_ops, c_pred) = (registration_tables("anchor"),
                                            registration_tables("candidate"))
        candidate_path = os.path.join(self._tmp.name, "dropped-libggml.so.0")
        Path(candidate_path).write_bytes(build_elf64(list(ANCHOR_EXPORTS[:-1])))
        return chain.symbol_evidence(
            anchor_binary=self.leg.anchor_paths["libggml.so.0"],
            candidate_binary=candidate_path, anchor=self.leg.libggml_anchor,
            declared=declared,
            anchor_op_registrations=a_ops, candidate_op_registrations=c_ops,
            anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=c_pred)

    def _gate(self, evidence):
        return correctness.check_symbol_and_registration_preservation(
            self.leg.evaluation_request(), evidence.diff, _t0_policy())

    def test_the_undeclared_case_is_the_bite(self):
        evidence = self._evidence(integrity.DeclaredSymbolDeltas(
            added=frozenset(), removed=frozenset(),
            arity_changed=frozenset({self.QUALIFIED})))
        self.assertIn(self.DROPPED, evidence.diff.removed_symbols)
        self.assertNotIn(self.DROPPED, evidence.diff.declared_removals)
        gate = self._gate(evidence)
        self.assertEqual(gate.check.outcome, schemas.FAIL, gate.check.reasons)
        self.assertTrue(any("removed and not declared" in r for r in gate.check.reasons))

    def test_an_honestly_declared_removal_is_the_compliant_control(self):
        """The gate must still PASS what the proposal really did declare."""
        evidence = self._evidence(integrity.DeclaredSymbolDeltas(
            added=frozenset(), removed=frozenset({self.QUALIFIED}),
            arity_changed=frozenset()))
        self.assertEqual(evidence.diff.declared_removals, (self.DROPPED,))
        self.assertEqual(self._gate(evidence).check.outcome, schemas.PASS)

    def test_declaring_nothing_is_the_other_control(self):
        evidence = self._evidence(integrity.DeclaredSymbolDeltas(
            added=frozenset(), removed=frozenset(), arity_changed=frozenset()))
        self.assertEqual(evidence.diff.declared_removals, ())
        self.assertEqual(self._gate(evidence).check.outcome, schemas.FAIL)


class TestTheAnchorToolchainIsMeasuredOffTheANCHORSLog(_ChainCase):
    """`static_and_compile_checks` must not PASS on a self-comparison.

    Both of that gate's cross-arm branches — the toolchain confound and the
    new-warning delta — compare a field of the ANCHOR's build against the same
    field of the CANDIDATE's. `ChainLeg.bind_anchor` measured the anchor
    toolchain off `self.build_log_text`, which is the candidate's build log, so
    both comparisons were identities, neither branch could fire for any
    candidate, and the gate reported PASS. Nothing in a build log says whose
    build it was; `anchor_toolchain_from_build_log` now takes the candidate's
    `BuildProvenance` and refuses the composition.

    Bite: delete the same-file refusal and
    `test_the_candidates_own_log_is_refused_as_the_anchors` FAILs; re-point
    `ChainLeg.bind_anchor` at `self.build_log_text` and
    `test_a_new_candidate_warning_is_now_VISIBLE` FAILs with a PASS.
    """

    #: ONE leg per test, built by the test that needs it. A second leg in the
    #: same world blocks on the first one's CPU-region claim, which is the claim
    #: working exactly as designed.
    def _leg(self, stage, **kwargs):
        return ChainLeg(self.world, **kwargs).up_to(stage)

    def test_the_candidates_own_log_is_refused_as_the_anchors(self):
        leg = self._leg("anchor")
        with self.assertRaises(chain.BuildProvenanceUnprojectable) as ctx:
            chain.anchor_toolchain_from_build_log(
                leg.build_log_text,
                log_ref=f"file://{leg.result.log_path}",
                candidate_build=leg.build_evidence.provenance)
        self.assertIn("CANDIDATE's own build log", str(ctx.exception))

    def test_a_bare_string_cannot_stand_in_for_the_candidate_build_record(self):
        leg = self._leg("anchor")
        with self.assertRaises(TypeError):
            chain.anchor_toolchain_from_build_log(
                anchor_build_log(), log_ref="file:///anchor/build.log",
                candidate_build=f"file://{leg.result.log_path}")

    def test_the_legs_anchor_toolchain_came_from_a_different_file(self):
        """The composition, not just the guard."""
        leg = self._leg("anchor")
        anchor_log = T0.resolve_build_log_ref(leg.anchor_toolchain.log_ref)
        candidate_log = T0.resolve_build_log_ref(
            leg.build_evidence.provenance.build_log_ref)
        self.assertIsNotNone(anchor_log)
        self.assertIsNotNone(candidate_log)
        self.assertNotEqual(os.path.realpath(anchor_log),
                            os.path.realpath(candidate_log))

    def test_a_new_candidate_warning_is_now_VISIBLE(self):
        """The consequence: the gate FAILs a candidate that adds a warning.

        Under the old wiring the anchor's warning count was READ OFF THIS SAME
        LOG, so it rose with the candidate's and the delta was identically zero
        for every candidate that ever ran.
        """
        noisy = clean_configure_log() + (
            "ggml/src/ggml-cpu/ak-candidate.c:412:9: warning: unused variable ‘tmp’ "
            "[-Wunused-variable]\n")
        leg = self._leg("t0", configure_log=noisy)
        _errors, candidate_warnings, _f = T0.parse_compiler_diagnostics(leg.build_log_text)
        self.assertGreater(candidate_warnings, leg.anchor_toolchain.warning_count)
        gate = leg.t0_report.gate(correctness.GID_STATIC_COMPILE)
        self.assertEqual(gate.check.outcome, schemas.FAIL, gate.check.reasons)
        self.assertTrue(any("new compiler warning" in r for r in gate.check.reasons))

    def test_the_clean_leg_is_the_compliant_control(self):
        """Two separate captures of the same toolchain must still PASS."""
        leg = self._leg("t0")
        self.assertEqual(leg.t0_report.outcome(correctness.GID_STATIC_COMPILE),
                         schemas.PASS,
                         leg.t0_report.gate(correctness.GID_STATIC_COMPILE).check.reasons)


class TestARegistrationArityChangeReachesTheGate(_ChainCase):
    """A registration is data, not a symbol: the ELF diff sees nothing.

    `GGML_CPU_OP(MUL_MAT, 2)` becoming `GGML_CPU_OP(MUL_MAT, 5)` changes no
    exported name, links clean, and dispatches the op with the wrong operand
    count for every shape. `integrity.RegistrationDiff` has reported it since it
    was written; `correctness.SymbolTableDiff` had no field for it, so
    `chain.symbol_evidence` could only put it in a `checks` tuple that nothing
    reads, and T0 said PASS.

    Bite: drop `arity_changed_op_registrations` from the record (or the FAIL
    branch from `check_symbol_and_registration_preservation`) and
    `test_an_undeclared_registration_arity_change_FAILS_T0` FAILs with a PASS.
    """

    ARITY_CHANGED_SOURCES = {"ggml/src/ggml-cpu.c": (
        "GGML_CPU_OP(MUL_MAT, 5)\nGGML_CPU_OP(MUL_MAT_ID, 3)\n"
        "CPU_SUPPORTS(MUL_MAT)\nCPU_SUPPORTS(MUL_MAT_ID)\n")}

    def setUp(self):
        super().setUp()
        self.leg = ChainLeg(self.world).up_to("anchor")

    def _evidence(self, *, sources=None, declared=None):
        a_ops, a_pred = registration_tables("anchor")
        c_ops, c_pred = registration_tables("candidate", sources=sources)
        return chain.symbol_evidence(
            anchor_binary=self.leg.anchor_paths["libggml.so.0"],
            candidate_binary=self.leg.artifacts["libggml.so.0"],
            anchor=self.leg.libggml_anchor,
            declared=declared or integrity.DeclaredSymbolDeltas(
                added=frozenset(), removed=frozenset(), arity_changed=frozenset()),
            anchor_op_registrations=a_ops, candidate_op_registrations=c_ops,
            anchor_dispatch_predicates=a_pred, candidate_dispatch_predicates=c_pred)

    def _gate(self, evidence):
        return correctness.check_symbol_and_registration_preservation(
            self.leg.evaluation_request(), evidence.diff, _t0_policy())

    def test_an_undeclared_registration_arity_change_FAILS_T0(self):
        evidence = self._evidence(sources=self.ARITY_CHANGED_SOURCES)
        self.assertEqual(evidence.diff.arity_changed_op_registrations,
                         ("ggml_backend_cpu:MUL_MAT",))
        gate = self._gate(evidence)
        self.assertEqual(gate.check.outcome, schemas.FAIL, gate.check.reasons)
        self.assertTrue(any("changed arity undeclared" in r for r in gate.check.reasons))

    def test_a_declared_registration_arity_change_is_the_compliant_control(self):
        evidence = self._evidence(
            sources=self.ARITY_CHANGED_SOURCES,
            declared=integrity.DeclaredSymbolDeltas(
                added=frozenset(), removed=frozenset({"ggml_backend_cpu:MUL_MAT"}),
                arity_changed=frozenset()))
        self.assertEqual(self._gate(evidence).check.outcome, schemas.PASS)

    def test_the_unchanged_registration_is_the_other_control(self):
        evidence = self._evidence()
        self.assertEqual(evidence.diff.arity_changed_op_registrations, ())
        self.assertEqual(self._gate(evidence).check.outcome, schemas.PASS)


class EvidenceWithNoChecksDoesNotDerivePassTest(unittest.TestCase):
    """`chain._worst` delegates to `schemas.Check.worst_of`, empty case included.

    All four evidence records expose `worst` as a reduction over a plain `checks`
    tuple that nothing requires to be non-empty. It derived PASS from zero
    sub-checks, and `campaign.py` reads `build_ev.worst.outcome != schemas.PASS`
    to decide whether to abort the leg — so an evidence record carrying no checks
    LICENSED the run. That is the fail-open, and it is now COULD_NOT_CHECK.

    The records are built field-by-field rather than through their producers on
    purpose: the producers always append at least one check, so the hole is only
    reachable by construction, which is exactly the surface a projection seam
    exposes to its callers.
    """

    def _records(self, checks):
        return {
            "BuildEvidence": chain.BuildEvidence(
                provenance=None, checks=checks),
            "SymbolEvidence": chain.SymbolEvidence(
                diff=None, anchor_table=None, candidate_table=None,
                symbol_diff=None, op_registration_diff=None,
                dispatch_predicate_diff=None, checks=checks),
            "DiffEvidence": chain.DiffEvidence(
                policy=None, source_diff=None, checks=checks),
            "ChangeSurfaceEvidence": chain.ChangeSurfaceEvidence(
                surface=None, affected=None, checks=checks),
        }

    def test_no_evidence_record_derives_pass_from_zero_checks(self):
        for name, record in self._records(()).items():
            with self.subTest(record=name):
                self.assertEqual(record.worst.outcome, schemas.COULD_NOT_CHECK)
                self.assertFalse(record.worst.passed)
                self.assertEqual(record.worst.reasons,
                                 (schemas.EMPTY_CHECK_VECTOR_REASON,))

    def test_an_all_pass_vector_still_passes(self):
        """The fix must close the empty case without demoting a real clean result."""
        checks = (("a", schemas.Check(schemas.PASS)),
                  ("b", schemas.Check(schemas.PASS)))
        for name, record in self._records(checks).items():
            with self.subTest(record=name):
                self.assertEqual(record.worst.outcome, schemas.PASS)

    def test_reasons_carry_the_outcome_that_raised_them(self):
        checks = (("a", schemas.Check(schemas.COULD_NOT_CHECK, ("log unreadable",))),
                  ("b", schemas.Check(schemas.FAIL, ("production tree touched",))))
        worst = chain.BuildEvidence(provenance=None, checks=checks).worst
        self.assertEqual(worst.outcome, schemas.FAIL)
        self.assertEqual(worst.reasons, ("[COULD_NOT_CHECK] log unreadable",
                                         "[FAIL] production tree touched"))

    def test_a_non_check_in_the_checks_tuple_raises(self):
        with self.assertRaises(TypeError):
            chain.BuildEvidence(provenance=None,
                                checks=(("a", "PASS"),)).worst
