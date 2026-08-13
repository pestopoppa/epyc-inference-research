#!/usr/bin/env python3
"""t0_provider.py — the REAL `correctness.T0EvidenceProvider`.

WHY THIS MODULE EXISTS
----------------------
`evaluator/correctness.py` is a complete T0 gate with nothing behind it. Its own
docstring says so: *"It runs NO build, NO op suite, NO inference and NO
sanitizer"*, and `audit_no_write_or_process_paths()` proves that from its AST.
`T0EvidenceProvider` is a Protocol, and the only implementation in the tree —
`StaticEvidenceProvider` — serves a dict somebody else filled in. Every number
in every existing T0 test is synthetic.

This module is the producer. It builds argv, executes it, parses what came back,
and returns a `correctness.T0Evidence` whose fields are measurements.

THE ONE INVARIANT THAT SHAPES EVERYTHING HERE
---------------------------------------------
Five of the evidence types carry ANCHOR-derived material, and each names its
anchor by precondition 4's three components — `anchor_source_commit`,
`anchor_binary_sha256`, `anchor_linkage_sha256` — **all three or none**
(`correctness._validate_anchor_triple`). A partial name is refused because it
resolves to more than one artifact, and a placeholder is refused because a
fabricated identity reads as a resolved one.

So this provider may never *assemble* a triple. It holds an `AnchorCapture` —
an object that cannot exist unless all three components were MEASURED off a real
anchor binary in this session — and every anchor-derived field is filled from
that one object through `_anchor_triple()`, which returns three values or three
`None`s and has no third path. With no `AnchorCapture` there is no anchor-derived
evidence: `static_analysis` is `None` (its anchor compiler fields are mandatory
and unfakeable), `coherence` carries no anchor output, and every anchor field is
absent rather than borrowed from the request.

**The identity recorded is the identity MEASURED, never the identity
REQUESTED.** If the plan's anchor binary hashes to something other than
`request.anchor` says, this provider records what it hashed. The consumer then
raises (`CoherenceAnchorMismatch`) or FAILs (`check_binary_and_linkage_identity`)
— which is the point. Copying `request.anchor` into the evidence would make
those two consumers unfalsifiable: they would be comparing the request against
itself.

WHAT THE HOST TAUGHT US, MEASURED NOT ASSUMED
---------------------------------------------
Three facts below were established by running the real tools on 2026-08-03; the
verbatim captures are in `testdata/recorded_t0_*.txt` and the parsers are tested
against them.

1. **`test-backend-ops test` exits 0, prints `OK`, and runs ZERO cases when its
   `-b` filter matches no device.** Recorded:
   `test -o MUL_MAT_ID -b ROCm0` on a CPU-only build prints `Skipping`,
   `1/1 backends passed`, `OK`, exit 0. The skip path increments `n_ok`
   (tests/test-backend-ops.cpp:10366-10377), so a one-token typo in a backend
   name buys a clean pass over an unexercised op — `kernel_eval.sh`'s exact
   defect, still live in the instrument. `parse_backend_ops_console` therefore
   attributes cases to a backend ONLY from per-case result lines, and a skipped
   backend contributes nothing. Exit status is not consulted for coverage at all.
2. **`test` mode skips the CPU device unless `-b CPU` is passed explicitly**
   (same source, `backend_filter == NULL && dev_type == CPU`). A CPU op-suite
   run without the explicit filter tests nothing and says `OK`.
3. **A binary in an experimental worktree resolves production ggml under this
   container's ambient `LD_LIBRARY_PATH`.** `verify_ggml_linkage.sh` on the
   experimental `test-backend-ops`, ambient env: five libraries — all of ggml —
   resolve out of `/mnt/raid0/llm/llama.cpp/build/bin`, the frozen production
   tree. With the launcher's own `LD_LIBRARY_PATH` prepended: PASS. Linkage is
   not a formality here, it is the difference between measuring the candidate
   and measuring production.

PROCESS DISCIPLINE — the part that is not negotiable
----------------------------------------------------
INC-20260731: a name-pattern kill took out another agent's `llama-server` twice
and killed `earlyoom`, whose own argv contains the names it guards. This module
therefore:

  * spawns every child in its OWN session (`start_new_session=True`) and signals
    only the pid/pgid it captured itself — never a name, never a pattern;
  * escalates SIGTERM -> SIGKILL and VERIFIES death with `waitpid`/`os.kill(pid,
    0)` before reporting anything;
  * reports a child that would not die as an `orphan_process` on
    `StateSafetyEvidence` rather than swallowing it (invariant 10);
  * passes a FULLY DECLARED environment. Nothing is inherited: an ambient
    `LD_LIBRARY_PATH` is exactly how fact 3 above happens.

`audit_process_discipline()` proves the first clause from this module's own AST,
the same way `api.py` and `recipes.py` prove their own denials.

WHAT THIS MODULE DOES NOT PRODUCE
----------------------------------
`SymbolTableDiff`, `BuildProvenance` and `DiffPolicyEvidence` are accepted as
INPUTS and passed through. `integrity.py` already parses ELF tables, build
provenance and diffs against real artifacts, and duplicating it here would
create the second derivation `correctness.SEAMS` already records. Reference
evidence is instead projected from the instrument's structured per-case receipt;
without that receipt the exact-reference gate remains uncovered.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .. import schemas
from ..evaluator import api, correctness, recipes
from . import reward_hack_scan
from . import sandbox as process_sandbox

__all__ = [
    # errors
    "ExecutionError", "ClaimNotHeld", "ProductionTreeRefusal", "CaptureUnavailable",
    "OutputParseError", "AnchorCaptureIncomplete", "ProcessEscaped",
    # identity
    "PROVIDER_ID", "PRODUCER", "SCHEMA_FOLLOWUPS", "SEAMS", "STATE_SAFETY_CANNOT_PASS",
    # process seam
    "CompletedProcess", "ProcessRunner", "SubprocessRunner", "RecordedProcessRunner",
    # claims
    "HeldClaim", "require_claim",
    # captures
    "CaptureSink", "MemoryCaptureSink", "capture_ref",
    # plan
    "CandidateBuild", "AnchorBuild", "ToolPaths", "OpSuitePlan", "GenerationPlan",
    "HoldoutPlan", "T0ExecutionPlan",
    # parsers
    "BackendOpsReference", "BackendOpsCase", "BackendOpsBackend", "BackendOpsRun",
    "parse_backend_ops_console", "parse_backend_ops_csv",
    "LinkageRow", "LinkageReport", "parse_linkage_report",
    "parse_sanitizer_findings", "parse_compiler_diagnostics", "parse_sched_trace",
    "parse_delivered_tokens",
    # anchor
    "AnchorCapture", "capture_anchor_identity", "capture_anchor",
    # provider
    "ExecutedT0EvidenceProvider", "audit_process_discipline",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded capture
# =============================================================================

class ExecutionError(api.EvaluatorError):
    """Base class for every refusal this module makes."""


class ClaimNotHeld(ExecutionError):
    """An inference-bearing step was attempted outside a held claim.

    P-AK-SEARCH-1 denial 8: *"no inference run OUTSIDE A HELD CLAIM"*. This
    raises rather than recording a COULD_NOT_CHECK, because the run must not
    happen at all — a record explaining that an unclaimed run was inconclusive
    still means the host was stolen from.
    """


class ProductionTreeRefusal(ExecutionError):
    """A build, write, or candidate-arm execution named a frozen production tree."""


class CaptureUnavailable(ExecutionError):
    """A capture the provider was asked to replay does not exist.

    Deliberately a raise and not an empty capture: `RecordedProcessRunner`
    returning a blank `CompletedProcess` for an argv it never recorded would
    manufacture "the tool printed nothing", which every parser here reads as
    "zero cases" — a fabricated measurement wearing a real one's clothes.
    """


class OutputParseError(ExecutionError):
    """Tool output did not match the grammar this parser was written against.

    A parser that silently returns "nothing found" on an unrecognised format is
    the `feedback_filtered_log_masks_working_codepath` shape: the run looks
    clean because the reader could not read it.
    """


class AnchorCaptureIncomplete(ExecutionError):
    """An anchor capture was constructed without all three identity components."""


class ProcessEscaped(ExecutionError):
    """A child this module launched survived SIGTERM and SIGKILL."""


# =============================================================================
# Identity and recorded gaps
# =============================================================================

#: The producer stamped on every `produced_by` field this module fills. It is
#: the literal `"evaluator"` from `correctness.EVIDENCE_PRODUCERS` because this
#: code IS the evaluator's collection half: it runs the oracle, not the kernel.
#: A candidate-supplied number never reaches an evidence field from here — every
#: value below is parsed from a tool this module launched itself.
PRODUCER = "evaluator"

PROVIDER_ID = "ak.t0.execution.provider/v1"

#: `state_rollback_teardown_race` CANNOT PASS with today's collector, and this
#: string is that fact written where the collector can attach it to the record
#: rather than left to be rediscovered from the gate's source.
#:
#: The mechanism, end to end: `collect_state_safety` hardcodes
#: `rollback_tested=False` because nothing here exercises a rollback, and
#: `correctness.check_state_rollback_teardown_race` appends *"rollback was not
#: tested"* to its reasons whenever `not evidence.rollback_tested`. So:
#:
#:   * `state_safety_probe=True`  -> evidence EXISTS -> FAIL, always, for every
#:     candidate and every derived surface, whatever the candidate did. A gate
#:     that fails identically on every input carries no information about any of
#:     them, and a T0 FAIL is speed-blocking;
#:   * `state_safety_probe=False` -> no evidence -> the gate answers from the
#:     DERIVATION alone: PASS where the derivation determined the change touches
#:     neither persistent state nor threading, COULD_NOT_CHECK where it did not
#:     determine it, FAIL where it says the surface IS touched.
#:
#: The precise claim, and the one `TheStateSafetyGateCannotPass` in
#: `test_t0_provider.py` proves by exhaustion, is therefore: **no state-safety
#: MEASUREMENT can PASS.** Every PASS this surface can produce is a PASS by
#: non-applicability, granted by the change surface and not by anything this
#: module observed. That test is the tripwire: the day a real rollback probe
#: exists it will fail, which is the reminder to delete this constant, delete the
#: note, and wire the probe.
STATE_SAFETY_CANNOT_PASS = (
    "state_rollback_teardown_race CANNOT PASS ON EVIDENCE today: this collector has no "
    "rollback probe, so `rollback_tested` is hardcoded False and "
    "`check_state_rollback_teardown_race` FAILs on it unconditionally. With "
    "state_safety_probe=True the gate is a guaranteed FAIL for every candidate and every "
    "derived surface; with it False the gate is decided by the change derivation alone — "
    "PASS by non-applicability, COULD_NOT_CHECK, or FAIL where the derivation says the "
    "surface is touched. Leave the probe OFF until a rollback probe exists: a gate that "
    "fails identically on every input says nothing about any of them, and a T0 FAIL blocks "
    "speed ranking. The one real observation on this surface is `orphan_processes`, and it "
    "is collected either way.")

#: Schema follow-ups this module CANNOT close from here, reported rather than
#: patched. `schemas.py` and the evidence dataclasses in `correctness.py` are
#: owned by other agents this hour; denial 6 says a coverage gap is RECORDED.
#:
#: CLOSED 2026-08-04, two of three: `correctness.BuildProvenance` and
#: `correctness.DiffPolicyEvidence` now carry `produced_by`, validated by
#: `_req_producer`, and `check_clean_build_from_snapshot`,
#: `check_semantic_diff_conformance` and `check_schema_and_diff_policy` each FAIL
#: a record the evaluator did not produce. The entries are DELETED rather than
#: annotated, because a follow-up list that keeps closed items stops being read.
SCHEMA_FOLLOWUPS = (
    ("correctness.AntiRewardHackingEvidence.delivered_units_candidate is `int`, not "
     "`Optional[int]`, so a count that was NOT READ has no representation and must be written "
     "as 0 — the same value that means 'the candidate delivered nothing', which is a control-3 "
     "finding. `parse_delivered_tokens` is careful to return `None` rather than 0 for exactly "
     "this reason and the distinction dies at the schema. REQUIRED FOLLOW-UP: make the field "
     "Optional and have `check_anti_reward_hacking` read `None` as COULD_NOT_CHECK, the same "
     "way it already reads an absent `delivered_units_anchor`."),
)

#: Seams that still need something this module cannot supply. Recorded, not resolved.
SEAMS = (
    ("`SymbolTableDiff`, `BuildProvenance` and `DiffPolicyEvidence` are pass-through inputs. "
     "`integrity.py` already derives all three from real ELF tables, build logs and parsed "
     "diffs; producing them again here would create a second derivation of the §8.5.1 gates, "
     "which `correctness.SEAMS` item 1 already records as an open integration decision. This "
     "module accepts whatever `integrity.py` produced and does not compete with it. AS OF "
     "2026-08-04 the join exists: `execution/chain.py` carries `symbol_evidence`, "
     "`build_evidence` and `diff_policy_evidence`, which are the projections from "
     "integrity's records into correctness's, and `change_surface_from`, which is the same "
     "join for `evaluator/surface.py`'s derivation. Supplying them to `plan.symbols`, "
     "`plan.build`, `plan.diff` and `plan.change_surface` is what turns those surfaces from "
     "COULD_NOT_CHECK into gates. Nothing here derives them; the seam is still a seam."),
    ("The BEHAVIOURAL half of the change surface has no producer anywhere in this package, "
     "and `chain.change_surface_from` closes only part of it. `surface.AffectedSurface`'s "
     "axes are backends, link targets, op names, kernel symbols and dispatch predicates — "
     "there is no memory / threading / persistent-state axis, so `derived_touches_memory`, "
     "`derived_touches_threading` and `derived_touches_persistent_state` are classified "
     "lexically from the diff body and can only be TRUE or UNDETERMINED. `False` is what "
     "licenses `check_asan`'s PASS branch (\"the mechanical derivation finds it touches "
     "neither memory nor threading\") and no analysis in this tree can establish it. "
     "CONSEQUENCE, stated so a green report is not over-read: the sanitizer and state-safety "
     "surfaces become REAL gates for a candidate that visibly touches those surfaces and "
     "stay COULD_NOT_CHECK for one that does not. REQUIRED FOLLOW-UP: a whole-program or "
     "call-graph reachability pass over the affected closure, which is a real instrument and "
     "not a token list."),
    ("Dispatch-trace fallback detection sees INTER-backend assignment only. "
     "`GGML_SCHED_DEBUG=2` prints the scheduler's per-node backend assignment "
     "(ggml-backend.cpp:945-982), which catches an op that fell off the accelerator onto the "
     "CPU backend. It cannot see INTRA-backend kernel selection — a candidate whose optimized "
     "AVX-512 path silently declines to the generic ggml-cpu loop produces an identical "
     "trace. `DispatchTracePlan.fallback_scope` names which class the trace covers, and "
     "`fallback_instrumentation_active` is False whenever the campaign's fallback class is "
     "outside it, so the gate reads COULD_NOT_CHECK instead of PASS. Closing it needs "
     "instrumentation inside the backend, which is a source change and a normal candidate."),
    ("FOUR STATE-SAFETY FIELDS AND ONE ORACLE-USE FIELD ARE ASSERTED, NOT MEASURED. On "
     "`StateSafetyEvidence`: `race_detector_id=None`, `race_findings=()`, "
     "`leaked_resources=()` and `rollback_tested=False` — no race detector is run, no resource "
     "table is diffed across teardown, and no rollback is exercised. On "
     "`AntiRewardHackingEvidence`, `candidate_output_used_as_oracle=False` remains asserted. "
     "RVP-C6-9 now scans the committed candidate diff for environment probes and timing-"
     "dependent branches and records versioned detector ids; an absent diff records no detector "
     "id, so empty findings become COULD_NOT_CHECK rather than PASS. Only "
     "`orphan_processes` is a real observation. Two of these fail CLOSED and one does not: "
     "`rollback_tested=False` makes `check_state_rollback_teardown_race` FAIL outright "
     "whenever `state_safety_probe=True`, so the surface is currently a choice between a "
     "guaranteed FAIL and no evidence at all. Closing the remaining seam needs a real teardown "
     "probe and an observation proving whether candidate output entered the oracle; neither is "
     "synthesisable from `test-backend-ops` output."),
    ("`test-backend-ops test -b CPU` compares the CPU backend against the CPU backend: the "
     "reference in test mode is ggml's own CPU implementation of the same graph. For a CPU "
     "candidate this is a self-consistency check with real value (it catches a kernel that "
     "disagrees with the scalar reference path inside the same build) but it is NOT a "
     "comparison against the anchor's kernel, and no anchor triple is recorded on "
     "`OpSuiteEvidence` because none was involved. Cross-build op-level comparison needs the "
     "anchor's own suite run and a case-level join, which is a T1 instrument."),
)


# =============================================================================
# Small validators
# =============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: Names, not bodies. This module wrote the strict `_req_sha256` — the one that
#: rejects a placeholder digest — and two other modules wrote it without that
#: rejection. The body is now `schemas.require`, so the strict form is the only
#: form there is.
_req_str = schemas.require.str
_req_abs = schemas.require.abs_path
_req_sha256 = schemas.require.sha256
_req_commit = schemas.require.commit
_req_int = schemas.require.int
_req_bool = schemas.require.bool


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _lexically_normal(path: str) -> str:
    """`path` with `.` and `..` segments folded, WITHOUT touching the filesystem.

    `str(Path(x))` folds `//` and `.` and leaves `..` exactly where it was, so
    `/mnt/raid0/llm/exp/../llama.cpp/build` compares as "not a production tree"
    while naming one. This is the same fold `integrity._lexically_normal_parts`
    already does for the containment tests over there.
    """
    parts = Path(path).parts
    absolute = bool(parts) and parts[0] == "/"
    out: list = []
    for part in parts[1:] if absolute else parts:
        if part == ".":
            continue
        if part == "..":
            if out:
                out.pop()
            continue
        out.append(part)
    joined = "/".join(out)
    return "/" + joined if absolute else joined


def _containment_forms(path: str) -> tuple:
    """Every spelling of `path` a containment test has to answer for.

    Three, because a frozen tree can be named three ways and only one of them is
    the one the loader/compiler will actually write to:

      * the path as given (already folded for `//` and `.`),
      * the LEXICAL fold, which removes `..`,
      * the PHYSICAL resolution, which removes symlinks.

    `/workspace/repos/epyc-llama` is in `PRODUCTION_TREE_ROOTS` because that
    symlink is known; an ad-hoc symlink into a frozen tree is not, and only
    `realpath` sees it. `os.path.realpath` does not require the path to exist —
    a build directory that has not been created yet still resolves through the
    ancestors that do.
    """
    forms = {str(Path(path)), _lexically_normal(path)}
    try:
        forms.add(os.path.realpath(path))
    except (OSError, ValueError):  # pragma: no cover - realpath is total on Linux
        pass
    return tuple(sorted(forms))


def _production_roots() -> tuple:
    roots = set()
    for root in correctness.PRODUCTION_TREE_ROOTS:
        roots.add(root)
        try:
            roots.add(os.path.realpath(root))
        except (OSError, ValueError):  # pragma: no cover
            pass
    return tuple(sorted(roots))


def under_production_tree(path: str) -> bool:
    """True when `path` is inside a frozen production kernel tree.

    Takes the root list from `correctness.PRODUCTION_TREE_ROOTS` rather than
    re-listing it — two lists of frozen trees is one list plus a future
    divergence — but does NOT reuse that module's string comparison. A prefix
    test over `str(Path(path))` answers for exactly one spelling of the path, and
    boundary 1 has to hold for every spelling that reaches the same inode:
    `/mnt/raid0/llm/exp/../llama.cpp/build` and a symlink pointing at
    `/mnt/raid0/llm/llama.cpp` both used to read as "not production" here, and
    both are writes into a frozen tree. Every form in `_containment_forms` is
    tested against every root and its physical resolution.
    """
    roots = _production_roots()
    return any(form == root or form.startswith(root + "/")
               for form in _containment_forms(path)
               for root in roots)


def _refuse_production_write(path: str, label: str) -> str:
    """Refuse a WRITE path that names, or can reach, a frozen production tree.

    Two refusals, in order. An unnormalized path is refused outright — the same
    fail-closed reading `integrity._refuse_unnormalized_path` makes, because
    normalizing it here would be a guess about the caller's intent and the guess
    is what the evasion needs. Then containment, over all three spellings.
    """
    if any(part in (".", "..") for part in Path(path).parts):
        raise ProductionTreeRefusal(
            f"{label} {path!r} carries a '.' or '..' segment. An unnormalized write path "
            "cannot be read for containment by eye and folds to somewhere else entirely — "
            "'/mnt/raid0/llm/exp/../llama.cpp/build' is the frozen production tree. Record "
            "the resolved path.")
    if under_production_tree(path):
        raise ProductionTreeRefusal(
            f"{label} {path!r} is inside a frozen production kernel tree (it resolves to "
            f"{os.path.realpath(path)!r}). Production kernels are FROZEN: never build in one, "
            "never write to one, never switch its branch (invariant 3, denial 2). Reading one "
            "is permitted and is how anchoring works; this path would be written.")
    return path


def sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes. Raises rather than returning a placeholder."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise CaptureUnavailable(
            f"cannot hash {path}: {exc}. An identity that cannot be measured is recorded as "
            "absent, never as a placeholder.") from exc


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# The process seam
# =============================================================================

@dataclass(frozen=True)
class CompletedProcess:
    """One execution, with everything a record needs to be re-checkable.

    `env` is a tuple of pairs and not a dict so the whole object is hashable and
    canonical-JSON-able: the capture's own content hash is what `receipt_ref`
    and `evidence_ref` cite, and a dict would make that hash order-dependent.
    """

    argv: tuple
    env: tuple
    cwd: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    signalled: bool
    orphans: tuple = ()
    sandbox_receipt: Optional[dict] = None
    sandbox_teardown: Optional[dict] = None

    def __post_init__(self) -> None:
        for item in self.argv:
            _req_str(item, "process.argv[]")
        if not self.argv:
            raise ValueError("process.argv must be non-empty")
        for pair in self.env:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise TypeError("process.env must be a tuple of (name, value) pairs")
        if self.exit_code is not None and (isinstance(self.exit_code, bool)
                                           or not isinstance(self.exit_code, int)):
            raise ValueError("process.exit_code must be an int or None")

    @property
    def combined(self) -> str:
        """stdout then stderr. Tools here interleave; parsers scan both."""
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr

    def env_value(self, name: str) -> Optional[str]:
        for key, value in self.env:
            if key == name:
                return value
        return None

    def to_dict(self) -> dict:
        return {
            "argv": list(self.argv),
            "env": [list(pair) for pair in self.env],
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round(self.duration_s, 6),
            "timed_out": self.timed_out,
            "signalled": self.signalled,
            "orphans": list(self.orphans),
            "sandbox_receipt": self.sandbox_receipt,
            "sandbox_teardown": self.sandbox_teardown,
        }

    def content_sha256(self) -> str:
        return schemas.content_hash(self.to_dict())


def capture_ref(capture: CompletedProcess) -> str:
    """The `*_ref` string an evidence field cites. Content-addressed, not a path.

    A ref that is a filesystem path is only as durable as the directory; a
    content hash of the capture itself is checkable from the record forever, and
    a sink that stores the capture stores it UNDER this name.
    """
    return f"akcap:{capture.content_sha256()[:32]}"


class ProcessRunner(Protocol):
    """Executes one argv with a fully declared environment. The only exec seam."""

    def run(self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str,
            timeout_s: float) -> CompletedProcess:
        ...


class SubprocessRunner:
    """The real runner. Spawns, waits, and kills ONLY what it launched itself.

    Four properties, each one a scar:

    1. **The environment is passed, never inherited.** `env=` is exactly what
       the caller declared. An inherited `LD_LIBRARY_PATH` is how an
       experimental binary loads production ggml and runs silently wrong
       (INC-20260731; still reproducible on this host — see the module
       docstring, fact 3).
    2. **No shell, ever.** `shell=False` is the default and is passed
       explicitly. `argv` is a list; nothing here renders a command line, so
       there is no quoting bug for it to have.
    3. **No pipe into another process.** stdout/stderr go to `subprocess.PIPE`
       and are read by this process. `feedback_pipe_hazards`: piping a llama
       binary changes its behaviour, and a piped test's exit status becomes the
       tail's.
    4. **Kill by captured pid only.** The child gets its own session
       (`start_new_session=True`), so the only pgid this module ever signals is
       one it created. SIGTERM, then SIGKILL, then VERIFY — and a survivor is
       reported as an orphan rather than assumed dead.
    """

    def __init__(self, *, term_grace_s: float = 10.0, kill_grace_s: float = 5.0,
                 sandbox_policy: Optional[process_sandbox.SandboxPolicy] = None) -> None:
        if term_grace_s <= 0 or kill_grace_s <= 0:
            raise ValueError("grace periods must be positive")
        self._term_grace_s = float(term_grace_s)
        self._kill_grace_s = float(kill_grace_s)
        if sandbox_policy is not None and not isinstance(
                sandbox_policy, process_sandbox.SandboxPolicy):
            raise TypeError("sandbox_policy must be a SandboxPolicy or None")
        self._sandbox_policy = sandbox_policy

    def run(self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str,
            timeout_s: float) -> CompletedProcess:
        argv = tuple(str(token) for token in argv)
        if not argv:
            raise ValueError("argv must be non-empty")
        executed_env = {str(k): str(v) for k, v in env.items()}
        scratch = None
        receipt_path = None
        spawn_argv = argv
        if self._sandbox_policy is not None:
            executed_env["PYTHONDONTWRITEBYTECODE"] = "1"
            scratch = tempfile.TemporaryDirectory(
                prefix="autokernel-t0-", dir=self._sandbox_policy.writable_root)
            evaluator_dir = Path(scratch.name, "evaluator")
            candidate_dir = Path(scratch.name, "candidate")
            evaluator_dir.mkdir()
            candidate_dir.mkdir()
            receipt_path = evaluator_dir / "sandbox-receipt.json"
            executed_env["TMPDIR"] = str(candidate_dir)
            invocation_policy = process_sandbox.SandboxPolicy(
                writable_root=str(candidate_dir),
                cgroup_root=self._sandbox_policy.cgroup_root,
                limits=self._sandbox_policy.limits,
                token=self._sandbox_policy.token)
            spawn_argv = invocation_policy.wrap(
                argv, receipt_path=str(receipt_path))
        else:
            invocation_policy = None
        env_pairs = tuple(sorted(executed_env.items()))
        started = time.monotonic()
        try:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell, declared env
                list(spawn_argv),
                env=dict(env_pairs),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
                text=True,
            )
        except BaseException:
            if scratch is not None:
                scratch.cleanup()
            raise
        timed_out = False
        signalled = False
        orphans: tuple = ()
        sandbox_receipt = None
        sandbox_teardown = None
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            signalled = True
            orphans = self._terminate(proc)

            try:
                stdout, stderr = proc.communicate(timeout=self._kill_grace_s)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
                self._close_pipes(proc)
        except BaseException:
            # ANY other way out of `communicate` — MemoryError, a KeyboardInterrupt
            # at the console, an OSError on the pipes — used to propagate with the
            # child still running. `run()` is the only place holding that pid, so
            # nothing downstream could ever kill it, and on this host the escapee
            # is a `taskset -c 0-95` llama-cli with the full OMP stack that owns
            # the machine until someone finds it by name — which is the operation
            # INC-20260731 forbids. Terminate what we launched, THEN re-raise.
            try:
                self._terminate(proc)
            except BaseException:  # noqa: BLE001 - never mask the original failure
                pass
            self._close_pipes(proc)
            self._cleanup_sandbox_after_failure(proc.pid)
            if scratch is not None:
                scratch.cleanup()
            raise
        if self._sandbox_policy is not None:
            try:
                sandbox_receipt = process_sandbox.read_receipt(receipt_path)
                process_sandbox.verify_receipt(
                    sandbox_receipt, policy=invocation_policy, pid=proc.pid, argv=argv)
                sandbox_teardown = process_sandbox.cleanup_cgroup(
                    invocation_policy, proc.pid)
            except process_sandbox.SandboxError as exc:
                self._cleanup_sandbox_after_failure(proc.pid)
                raise ExecutionError(
                    f"candidate containment did not produce a verified receipt and "
                    f"teardown: {exc}") from exc
            finally:
                if scratch is not None:
                    scratch.cleanup()
        return CompletedProcess(
            argv=argv,
            env=env_pairs,
            cwd=cwd,
            exit_code=proc.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
            signalled=signalled,
            orphans=orphans,
            sandbox_receipt=sandbox_receipt,
            sandbox_teardown=sandbox_teardown,
        )

    def _cleanup_sandbox_after_failure(self, pid: int) -> None:
        """Best-effort drain on an exceptional path; never hides the first failure."""
        if self._sandbox_policy is None:
            return
        path = self._sandbox_policy.cgroup_path(pid)
        if not path.exists():
            return
        try:
            process_sandbox.cleanup_cgroup(self._sandbox_policy, pid)
        except process_sandbox.SandboxError:
            pass

    @staticmethod
    def _close_pipes(proc: "subprocess.Popen") -> None:
        """Close the stdio pipes on the paths that never reach `communicate`'s return.

        `communicate` closes them on a normal return; the two abnormal exits here
        do not, and the test suite runs under `-W error::ResourceWarning`.
        """
        for stream in (proc.stdout, proc.stderr, proc.stdin):
            if stream is not None:
                try:
                    stream.close()
                except OSError:  # pragma: no cover - already closed
                    pass

    def _terminate(self, proc: "subprocess.Popen") -> tuple:
        """SIGTERM -> SIGKILL -> verify, against the pgid WE created.

        Returns the pids still alive after SIGKILL — the honest orphan list that
        `StateSafetyEvidence.orphan_processes` carries.

        **Death is proved by REAPING, not by `os.kill(pid, 0)`.** The first
        version of this method used `os.kill(pid, 0)` alone and the timeout test
        failed with a live orphan every time: a killed child that has not been
        waited for is a ZOMBIE, and a zombie answers signal 0 exactly like a
        running process. Reporting it as an escaped process would have put a
        phantom orphan on every timed-out capture's state-safety evidence — and
        `check_state_rollback_teardown_race` FAILs on a non-empty orphan list,
        so every timeout would have failed a T0 gate for a process that was
        already dead. `proc.poll()` reaps and returns the status; that is the
        proof. `_pid_alive` is kept as the secondary check for the case where
        the child was reaped by someone else.
        """
        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return ()
        for sig, grace in ((signal.SIGTERM, self._term_grace_s),
                           (signal.SIGKILL, self._kill_grace_s)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return ()
            except PermissionError as exc:
                raise ProcessEscaped(
                    f"cannot signal process group {pgid} that this runner created: {exc}"
                ) from exc
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if proc.poll() is not None or not _pid_alive(pid):
                    return ()
                time.sleep(0.05)
        if proc.poll() is None and _pid_alive(pid):
            return (str(pid),)
        return ()


def _pid_alive(pid: int) -> bool:
    """True when `pid` still exists. `os.kill(pid, 0)` sends no signal.

    This is a check on ONE pid this module captured. It is not `pgrep`, it takes
    no name, and it can match nothing else on a shared host.

    It cannot distinguish a zombie from a running process — see `_terminate`,
    which is why the primary proof of death there is a reap.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RecordedProcessRunner:
    """Replays captures recorded from real runs. RAISES on an argv it has never seen.

    Two uses, one shape. Invariant 11 makes *"deterministic replay before
    regeneration"* the normal path for re-scoring saved material, and the tests
    drive the whole provider from real recorded tool output without a host.

    It never synthesises. An unmatched argv is `CaptureUnavailable`, because a
    blank `CompletedProcess` reads to every parser here as "the tool ran and
    found nothing", which is a fabricated measurement.
    """

    def __init__(self, captures: Sequence[CompletedProcess]) -> None:
        self._by_key: dict = {}
        self._calls: list = []
        for capture in captures:
            if not isinstance(capture, CompletedProcess):
                raise TypeError("captures must be CompletedProcess instances")
            self._by_key.setdefault(self._key(capture.argv), []).append(capture)

    @staticmethod
    def _key(argv: Sequence[str]) -> tuple:
        return tuple(str(token) for token in argv)

    @property
    def calls(self) -> tuple:
        """Every argv this runner was asked for, in order. The argv assertion seam."""
        return tuple(self._calls)

    def run(self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str,
            timeout_s: float) -> CompletedProcess:
        key = self._key(argv)
        self._calls.append((key, tuple(sorted((str(k), str(v)) for k, v in env.items()))))
        queue = self._by_key.get(key)
        if not queue:
            raise CaptureUnavailable(
                f"no recorded capture for argv {list(key)}. This runner replays real "
                f"captures and never synthesises one: an empty stdout reads to every parser "
                f"as 'the tool ran and found nothing'. Recorded argvs are "
                f"{[list(k) for k in sorted(self._by_key)]}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


# =============================================================================
# Claims — denial 8's "no inference run OUTSIDE A HELD CLAIM"
# =============================================================================

class HeldClaim(Protocol):
    """A resource claim currently held by THIS process.

    Structural, not nominal, and deliberately so: the GPU implementation is
    `resource/device_claim.DeviceClaim`, the CPU region claim is being built as
    `execution/cpu_region_claim.py` by a sibling agent this hour, and binding to
    either by import would bind to a module that may not exist yet. Anything
    with these three members satisfies it.
    """

    @property
    def claim_id(self) -> str:
        ...

    def is_held(self) -> bool:
        ...

    def describe(self) -> str:
        ...

    # `CpuRegionClaim` answers `verify_held()`/`held` and `covers(cpu_list)`
    # instead of `is_held()`/`describe()`; `require_claim` accepts either shape
    # and PREFERS `verify_held()` where it exists. See `_claim_is_held`.


def _claim_is_held(claim: Any) -> bool:
    """Ask a claim whether it is held, in any of the three shapes that exist.

    `resource/device_claim.DeviceClaim` answers `is_held()`.
    `execution/cpu_region_claim.CpuRegionClaim` — the ONLY CPU region claim in
    this tree, and the one every CPU measurement here needs — has neither
    `is_held` nor `describe`: it exposes a `held` property and a `verify_held()`
    returning a `Check`. Requiring `is_held` by name made the documented wiring
    (`claim=<CPU region claim>`) raise `TypeError` before any of these guards
    could run, so the strictest reading of the protocol also made it unusable.

    `verify_held()` is preferred where it exists: it RE-READS the lock, so it
    answers the recycled-pid and stale-holder question a cached boolean cannot.
    """
    verify = getattr(claim, "verify_held", None)
    if callable(verify):
        outcome = getattr(verify(), "outcome", None)
        if outcome is not None:
            return outcome == schemas.PASS
    is_held = getattr(claim, "is_held", None)
    if callable(is_held):
        return bool(is_held())
    held = getattr(claim, "held", None)
    if isinstance(held, bool):
        return held
    raise TypeError(
        f"claim must answer whether it is held — `verify_held()`, `is_held()` or a `held` "
        f"property; {type(claim).__name__} has none of them")


def _claim_describe(claim: Any) -> str:
    describe = getattr(claim, "describe", None)
    if callable(describe):
        return str(describe())
    plan = getattr(claim, "plan", None)
    return f"{type(claim).__name__}(plan={plan!r})"


def _canonical_cpu_list() -> str:
    """The cpu list the canonical prefix actually pins, read OFF the prefix.

    `recipes.CANONICAL_PREFIX` is the ratified constant and `recipes.py` reads
    the list out of it exactly this way. Retyping `"0-95"` here would be the
    hardcoded-taskset defect the measurement discipline names by hand.
    """
    prefix = list(recipes.CANONICAL_PREFIX)
    return prefix[prefix.index("-c") + 1]


def require_claim(claim: Any, *, what: str, cpu_list: Optional[str] = None) -> str:
    """Return the claim id, or REFUSE the run. Never a warning, never a downgrade.

    Denial 8 is the one process limit that binds every inference-bearing step
    here. A COULD_NOT_CHECK would be a record about the measurement; the point
    is that the measurement must not be taken — an unclaimed run steals the host
    from whoever does hold the claim, and its number is contaminated besides.

    `cpu_list` is the footprint the argv PINS. Precondition 1 asks for *"a CPU
    region claim covering the exact footprint measured"*, and "a claim exists"
    is not that: a claim over cores 0-7 answers `is_held()` exactly like a claim
    over 0-95, and the run underneath it is pinned to 0-95 either way. Where the
    claim can answer the coverage question (`CpuRegionClaim.covers`), it is
    asked; where it cannot, the gap is refused rather than assumed — a claim
    object that cannot state its footprint cannot authorise one.
    """
    if claim is None:
        raise ClaimNotHeld(
            f"{what} runs inference and no resource claim was supplied. P-AK-SEARCH-1 "
            "denial 8: 'no inference run OUTSIDE A HELD CLAIM'. Acquire the claim covering "
            "the exact footprint the argv pins, then pass it in.")
    if not hasattr(claim, "claim_id"):
        raise TypeError(
            f"claim must implement the HeldClaim protocol (claim_id, and one of "
            f"verify_held/is_held/held); {type(claim).__name__} has no 'claim_id'")
    if not _claim_is_held(claim):
        raise ClaimNotHeld(
            f"{what} runs inference and claim {claim.claim_id!r} reports itself NOT held "
            f"({_claim_describe(claim)}). A claim that was acquired and then lost is not a "
            "claim; invariant 9 makes a PASS an observation, never a claim.")
    if cpu_list is not None:
        covers = getattr(claim, "covers", None)
        if callable(covers) and not covers(cpu_list):
            raise ClaimNotHeld(
                f"{what} pins cpus {cpu_list!r} and claim {claim.claim_id!r} does not cover "
                f"them ({_claim_describe(claim)}). A held claim over a smaller region does "
                "not authorise a wider run: the cores outside it belong to whoever else "
                "holds them, and the number taken across them is contended besides.")
    return str(claim.claim_id)


# =============================================================================
# Capture sinks
# =============================================================================

class CaptureSink(Protocol):
    """Stores a capture under its content-addressed ref so a record can cite it."""

    def store(self, capture: CompletedProcess) -> str:
        ...


class MemoryCaptureSink:
    """An in-process sink. Writes nothing; the default when no durable sink is given."""

    def __init__(self) -> None:
        self._by_ref: dict = {}

    def store(self, capture: CompletedProcess) -> str:
        ref = capture_ref(capture)
        self._by_ref[ref] = capture
        return ref

    def get(self, ref: str) -> CompletedProcess:
        try:
            return self._by_ref[ref]
        except KeyError:
            raise CaptureUnavailable(f"no capture stored under {ref!r}") from None

    @property
    def refs(self) -> tuple:
        return tuple(sorted(self._by_ref))


# =============================================================================
# The plan — what to run, where, against what
# =============================================================================

@dataclass(frozen=True)
class ToolPaths:
    """Absolute paths to every external tool. No PATH lookup, no ambient resolution.

    A tool resolved through `PATH` is a tool whose identity depends on whoever's
    shell started the session, which is the same class of defect as an inherited
    `LD_LIBRARY_PATH`.
    """

    bash: str
    verify_ggml_linkage_sh: str
    cmake: Optional[str] = None
    git: Optional[str] = None

    def __post_init__(self) -> None:
        _req_abs(self.bash, "tools.bash")
        _req_abs(self.verify_ggml_linkage_sh, "tools.verify_ggml_linkage_sh")
        for name in ("cmake", "git"):
            value = getattr(self, name)
            if value is not None:
                _req_abs(value, f"tools.{name}")


@dataclass(frozen=True)
class CandidateBuild:
    """The candidate under test. Every path is refused inside a production tree."""

    worktree: str
    build_dir: str
    source_commit: str
    source_sha256: str
    binary: str
    library_path: str
    test_backend_ops: str
    build_log_ref: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("worktree", "build_dir", "binary", "library_path", "test_backend_ops"):
            _req_abs(getattr(self, name), f"candidate.{name}")
        _req_commit(self.source_commit, "candidate.source_commit")
        _req_sha256(self.source_sha256, "candidate.source_sha256")
        for name in ("worktree", "build_dir", "binary", "test_backend_ops"):
            _refuse_production_write(getattr(self, name), f"candidate.{name}")
        if str(Path(self.binary).parent) != str(Path(self.library_path)):
            raise ValueError(
                f"candidate.library_path must be the binary's own directory "
                f"({Path(self.binary).parent}), not {self.library_path}. Anything else lets "
                "the binary resolve another tree's libggml — the failure is silent and the "
                "numbers stay plausible (INC-20260731).")


@dataclass(frozen=True)
class AnchorBuild:
    """The anchor. READ-ONLY, and it may legitimately be a production tree.

    `recipes.py` draws the same line: *"a candidate arm whose binary or source
    root resolves inside a frozen production tree is refused. The anchor arm is
    allowed there, because the anchor IS the frozen production binary and
    executing it read-only is not a write."*
    """

    worktree: str
    source_commit: str
    binary: str
    library_path: str

    def __post_init__(self) -> None:
        for name in ("worktree", "binary", "library_path"):
            _req_abs(getattr(self, name), f"anchor.{name}")
        _req_commit(self.source_commit, "anchor.source_commit")
        if str(Path(self.binary).parent) != str(Path(self.library_path)):
            raise ValueError(
                f"anchor.library_path must be the binary's own directory "
                f"({Path(self.binary).parent}), not {self.library_path}")


@dataclass(frozen=True)
class OpSuitePlan:
    """`test-backend-ops` parameters. The backend filter is REQUIRED, and here is why.

    Recorded on this host, 2026-08-03:
    `test-backend-ops test -o ARANGE` with no `-b` prints `Skipping CPU backend`,
    `1/1 backends passed`, `OK`, exit 0 — zero cases. The CPU device is skipped
    outright in test mode unless the filter names it
    (tests/test-backend-ops.cpp:10372). An omitted filter is therefore not a
    wider run; it is an empty one that says OK.
    """

    backend_filter: str
    ops: tuple
    suite_id: str
    suite_source_sha256: str
    suite_seed: int = 0
    layout_probe: bool = False
    value_transform_probe: bool = False
    stateful_probe: bool = False
    timeout_s: float = 1800.0
    parallel_workers: Optional[int] = None

    def __post_init__(self) -> None:
        _req_str(self.backend_filter, "op_suite.backend_filter")
        if not self.ops:
            raise ValueError(
                "op_suite.ops is empty. A suite with no op filter is not a broader suite; "
                "combined with a backend filter that matches nothing it is a clean OK over "
                "nothing at all, which is the defect T0 exists to catch.")
        for op in self.ops:
            _req_str(op, "op_suite.ops[]")
        _req_str(self.suite_id, "op_suite.suite_id")
        _req_sha256(self.suite_source_sha256, "op_suite.suite_source_sha256")
        _req_int(self.suite_seed, "op_suite.suite_seed")
        _req_bool(self.layout_probe, "op_suite.layout_probe")
        _req_bool(self.value_transform_probe, "op_suite.value_transform_probe")
        _req_bool(self.stateful_probe, "op_suite.stateful_probe")
        if sum((self.layout_probe, self.value_transform_probe, self.stateful_probe)) > 1:
            raise ValueError(
                "op_suite layout, value-transform, and stateful probes must be separate passes")
        if self.parallel_workers is not None:
            _req_int(self.parallel_workers, "op_suite.parallel_workers", minimum=1)


@dataclass(frozen=True)
class GenerationPlan:
    """The coherence/determinism generation. Greedy is DERIVED from argv, not declared.

    `sampler_is_greedy` on `CoherenceEvidence` is the field that decides whether
    a byte difference means anything at all (`_derive_coherence` step 4). Taking
    it as a caller's boolean would let a sampled run be scored as a greedy one,
    so this plan carries the sampling PARAMETERS and `is_greedy()` reads them.
    """

    prompt: str
    prompt_ref: str
    n_predict: int
    seed: int
    temperature: float = 0.0
    top_k: int = 1
    threads: Optional[int] = None
    extra_argv: tuple = ()
    timeout_s: float = 1800.0

    def __post_init__(self) -> None:
        _req_str(self.prompt, "generation.prompt")
        _req_str(self.prompt_ref, "generation.prompt_ref")
        _req_int(self.n_predict, "generation.n_predict", minimum=1)
        _req_int(self.seed, "generation.seed", minimum=0)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("generation.temperature must be a number")
        _req_int(self.top_k, "generation.top_k", minimum=1)
        if self.threads is not None:
            _req_int(self.threads, "generation.threads", minimum=1)
        for token in self.extra_argv:
            _req_str(token, "generation.extra_argv[]")

    def is_greedy(self) -> bool:
        """Greedy iff temperature is exactly 0 AND top-k is exactly 1.

        Both, not either. `--temp 0` with `top_k > 1` still resolves ties
        through the sampler's RNG in llama.cpp's sampling chain, and `top_k 1`
        with a non-zero temperature is greedy only by accident of the
        implementation. A coherence label rests on this bit; it is not the place
        for an approximation.
        """
        return float(self.temperature) == 0.0 and int(self.top_k) == 1


@dataclass(frozen=True)
class HoldoutPlan:
    """Unseen/boundary shapes for a dispatch change.

    `visible_to_planner` is an input the CALLER must state, because this module
    cannot observe what the planner saw. It is passed straight through to
    `BoundaryShapeEvidence.held_out_from_planner` (negated), where
    `check_unseen_boundary_shapes` FAILs a holdout the planner could see: *"a
    search that can see its own holdout will find the kernel that special-cases
    it."* Defaulting it to "not visible" would be the provider asserting an
    anti-overfitting property it has no way to know.
    """

    unseen_case_filter: str
    boundary_case_filter: str
    selection_rule_id: str
    selection_seed: str
    visible_to_planner: bool
    timeout_s: float = 1800.0

    def __post_init__(self) -> None:
        for name in ("unseen_case_filter", "boundary_case_filter", "selection_rule_id",
                     "selection_seed"):
            _req_str(getattr(self, name), f"holdout.{name}")
        if not isinstance(self.visible_to_planner, bool):
            raise TypeError(
                "holdout.visible_to_planner must be stated explicitly as a bool; this module "
                "cannot observe what the planner saw and will not assume the safe answer")


@dataclass(frozen=True)
class DispatchTracePlan:
    """How the dispatch trace is collected and what class of fallback it can see."""

    derived_surface: tuple
    fallback_scope: str = "inter_backend"
    debug_level: int = 2
    timeout_s: float = 1800.0

    #: The only scope `GGML_SCHED_DEBUG` can actually observe. See SEAMS item 3.
    INSTRUMENTED_SCOPE = "inter_backend"

    def __post_init__(self) -> None:
        for op in self.derived_surface:
            _req_str(op, "dispatch.derived_surface[]")
        _req_str(self.fallback_scope, "dispatch.fallback_scope")
        _req_int(self.debug_level, "dispatch.debug_level", minimum=1)


@dataclass(frozen=True)
class T0ExecutionPlan:
    """Everything one T0 collection needs. No field decides anything by default."""

    candidate: CandidateBuild
    tools: ToolPaths
    op_suite: OpSuitePlan
    dispatch: DispatchTracePlan
    anchor: Optional[AnchorBuild] = None
    generation: Optional[GenerationPlan] = None
    determinism_runs: int = 0
    holdout: Optional[HoldoutPlan] = None
    sanitizer_target: Optional[str] = None
    sanitizer_build_dir: Optional[str] = None
    #: The instrumented binary's digest, measured OUTSIDE this process. Used only
    #: when `sha256_file` cannot reach the artifact (a replay, or a collection run
    #: on another host); never as a preference over the measured hash.
    sanitizer_binary_sha256: Optional[str] = None
    sanitizer_jobs: int = 32
    sanitizer_timeout_s: float = 7200.0
    backend: str = "llama_cpu"
    cache_state: str = "unknown"
    delivered_unit_name: str = "tokens_generated"
    oracle_ids: tuple = ()
    base_env: tuple = ()
    #: Arm-local variants licensed by the campaign parameter registry.  Kept
    #: separate from ``base_env`` so an ambient variable cannot override the
    #: canonical evaluator environment while a declared IQK comparison can.
    parameter_env: tuple = ()
    reference: Any = None
    symbols: Any = None
    build: Any = None
    diff: Any = None
    #: The MECHANICALLY DERIVED change surface, from `evaluator/surface.py`'s
    #: `derive_affected_surface` by way of `chain.change_surface_from`. `None`
    #: means no derivation was supplied, and `_change_surface` then emits a
    #: surface whose every `derived_touches_*` is `None` — which four T0 gates
    #: read as COULD_NOT_CHECK. It is a REAL FIELD and not a `getattr` target:
    #: `_change_surface` used to read `getattr(self._plan, "change_surface", None)`
    #: against a dataclass that had no such field, so the pass-through it
    #: documents was unreachable and no caller could have noticed.
    change_surface: Any = None
    #: Projection-side checks that do not fit in the projected correctness
    #: records. Each entry is ``(existing_gate_id, check_name, Check)``. The
    #: evaluator folds these into the named gate; they are not advisory notes.
    projection_checks: tuple = ()
    #: Hash-bound outputs from the sensitivity/hostile/checker reducers.  They
    #: become verdict-bearing only after ``evidence_for`` binds them to the live
    #: request's candidate source and evaluator bundle.
    source_prerequisites: tuple = ()
    state_safety_probe: bool = False
    #: Committed source delta from the reviewed measurement base. `None` means
    #: the two C6 source detectors did not run, and empty findings then remain
    #: UNKNOWN rather than being interpreted as clean.
    candidate_diff_text: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateBuild):
            raise TypeError("plan.candidate must be a CandidateBuild")
        if not isinstance(self.tools, ToolPaths):
            raise TypeError("plan.tools must be a ToolPaths")
        if not isinstance(self.op_suite, OpSuitePlan):
            raise TypeError("plan.op_suite must be an OpSuitePlan")
        if not isinstance(self.dispatch, DispatchTracePlan):
            raise TypeError("plan.dispatch must be a DispatchTracePlan")
        if self.anchor is not None and not isinstance(self.anchor, AnchorBuild):
            raise TypeError("plan.anchor must be an AnchorBuild or None")
        if self.generation is not None and not isinstance(self.generation, GenerationPlan):
            raise TypeError("plan.generation must be a GenerationPlan or None")
        if self.holdout is not None and not isinstance(self.holdout, HoldoutPlan):
            raise TypeError("plan.holdout must be a HoldoutPlan or None")
        if self.cache_state not in correctness.CACHE_STATES:
            raise ValueError(
                f"plan.cache_state: {self.cache_state!r} is not one of "
                f"{list(correctness.CACHE_STATES)}. 'unknown' is in the vocabulary so an "
                "unestablished state is SAYABLE — it reads as COULD_NOT_CHECK downstream, "
                "which is what it is; it is never silently 'cold'.")
        _req_int(self.determinism_runs, "plan.determinism_runs")
        if self.determinism_runs and self.generation is None:
            raise ValueError(
                "plan.determinism_runs > 0 with no generation plan: there is nothing to "
                "repeat")
        if self.sanitizer_build_dir is not None:
            _refuse_production_write(self.sanitizer_build_dir, "plan.sanitizer_build_dir")
        if self.sanitizer_binary_sha256 is not None:
            _req_sha256(self.sanitizer_binary_sha256, "plan.sanitizer_binary_sha256")
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"plan.backend: {self.backend!r} is not one of "
                             f"{sorted(schemas.BACKENDS)}")
        if self.candidate_diff_text is not None and not isinstance(
                self.candidate_diff_text, str):
            raise TypeError("plan.candidate_diff_text must be a string or None")
        _validated_parameter_env(self.parameter_env)
        # The four pass-through evidence inputs are TYPE-CHECKED, and the reason
        # is seam 1: `integrity.py` and `correctness.py` each define a
        # `BuildProvenance`, they share no field name, and the wrong one arriving
        # here is SILENT — `collect_static_analysis` reads `build.build_log_ref`
        # with a `getattr`, `integrity.BuildProvenance` spells it
        # `build_log_path`, and the whole static-analysis surface then reports
        # COULD_NOT_CHECK for a reason that names the anchor instead of naming
        # the mistake. Same shape for a `surface.AffectedSurface` handed straight
        # to `change_surface` instead of through `chain.change_surface_from`.
        for name, expected in (("symbols", correctness.SymbolTableDiff),
                               ("build", correctness.BuildProvenance),
                               ("diff", correctness.DiffPolicyEvidence),
                               ("reference", correctness.ReferenceEvidence),
                               ("change_surface", correctness.ChangeSurface)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(
                    f"plan.{name} must be a correctness.{expected.__name__} or None, got "
                    f"{type(value).__module__}.{type(value).__name__}. The projections that "
                    f"produce these live in execution/chain.py (symbol_evidence, "
                    f"build_evidence, diff_policy_evidence, change_surface_from); a record "
                    "from evaluator/integrity.py or evaluator/surface.py is a DIFFERENT "
                    "class with overlapping names, and passing one here fails silently "
                    "rather than loudly.")
        allowed_projection_gates = {
            correctness.GID_SYMBOLS,
            correctness.GID_SEMANTIC_DIFF,
            correctness.GID_SURFACE_RECONCILIATION,
        }
        for item in self.projection_checks:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError(
                    "plan.projection_checks entries must be "
                    "(gate_id, check_name, schemas.Check) triples")
            gate_id, check_name, check = item
            if gate_id not in allowed_projection_gates:
                raise ValueError(
                    f"plan.projection_checks gate {gate_id!r} is not a projection-owned "
                    f"T0 gate; allowed={sorted(allowed_projection_gates)}")
            _req_str(check_name, "plan.projection_checks[].check_name")
            if not isinstance(check, schemas.Check):
                raise TypeError("plan.projection_checks[].check must be a schemas.Check")
        prerequisite_ids = []
        for item in self.source_prerequisites:
            if not isinstance(item, correctness.SourcePrerequisiteEvidence):
                raise TypeError(
                    "plan.source_prerequisites entries must be "
                    "correctness.SourcePrerequisiteEvidence")
            prerequisite_ids.append(item.prerequisite_id)
            if item.candidate_source_sha256 != self.candidate.source_sha256:
                raise ValueError(
                    "plan source prerequisite names a different candidate source SHA-256")
        if len(prerequisite_ids) != len(set(prerequisite_ids)):
            raise ValueError("plan.source_prerequisites contains duplicate ids")
        if self.source_prerequisites and not (self.candidate_diff_text or "").strip():
            raise ValueError(
                "a parameter/no-source plan cannot carry source prerequisites")


# =============================================================================
# Parsers — pure functions over real recorded tool output
# =============================================================================

_CASE_RE = re.compile(r"^ {2}([A-Z][A-Z0-9_]*)\((.*)\):\s*(.*)$")
_TESTS_PASSED_RE = re.compile(r"^ {2}(\d+)/(\d+) tests passed\s*$")
_BACKENDS_PASSED_RE = re.compile(r"^(\d+)/(\d+) backends passed\s*$")
_BACKEND_INIT_RE = re.compile(r"^Backend (\d+)/(\d+): (\S+)\s*$")
_BACKEND_STATUS_RE = re.compile(r"^ {2}Backend (\S+):\s*(OK|FAIL)\s*$")
_TESTING_RE = re.compile(r"^Testing (\d+) devices\s*$")
_REFERENCE_RECEIPT_RE = re.compile(
    r"^AK_REF_V1 metric=([A-Za-z0-9_.:/-]+) "
    r"observed=([0-9]+(?:\.[0-9]*)?(?:e[+-]?\d+)?) "
    r"tolerance=([0-9]+(?:\.[0-9]*)?(?:e[+-]?\d+)?) "
    r"comparisons=([1-9][0-9]*) oracle=([A-Za-z0-9_.:/-]+)$")
_PROPERTY_RECEIPT_V1_RE = re.compile(
    r"^AK_PROP_V1 metric=([A-Za-z0-9_.:/-]+) "
    r"residual=([0-9]+(?:\.[0-9]*)?(?:e[+-]?\d+)?) "
    r"tolerance=([0-9]+(?:\.[0-9]*)?(?:e[+-]?\d+)?) "
    r"passed=([01]) suite_seed=([0-9]+)$")
_PROPERTY_RECEIPT_V2_RE = re.compile(
    r"^AK_PROP_V2 metric=([A-Za-z0-9_.:/-]+) "
    r"residual=([0-9]+(?:\.[0-9]*)?(?:e[+-]?\d+)?) "
    r"tolerance=([0-9]+(?:\.[0-9]*)?(?:e[+-]?\d+)?) "
    r"passed=([01]) suite_seed=([0-9]+) "
    r"transform=(identity|x3|x0p01|negate)$")
_LAYOUT_RECEIPT_RE = re.compile(
    r"^AK_LAYOUT_V1 families=((?:offset|transpose|stride_gap)"
    r"(?:,(?:offset|transpose|stride_gap))*) suite_seed=([0-9]+)$")
_VALUE_RECEIPT_RE = re.compile(
    r"^AK_VALUE_V1 transforms=(identity,x3,x0p01,negate) "
    r"completed=([0-4]) suite_seed=([0-9]+)$")
_STATE_RECEIPT_RE = re.compile(
    r"^AK_STATE_V1 inputs=([0-9]+) initial_equal=([01]) "
    r"input_immutable=([01]) final_outputs=([0-9]+) suite_seed=([0-9]+)$")

#: The two skip reasons the tool prints, verbatim
#: (tests/test-backend-ops.cpp:10368 and :10375). Both increment `n_ok`, so both
#: end in `N/N backends passed` + `OK` + exit 0 with nothing run.
_SKIP_REASONS = ("Skipping", "Skipping CPU backend")


@dataclass(frozen=True)
class BackendOpsReference:
    """A passing case's comparison against the activated CPU reference path."""

    metric_id: str
    observed: float
    tolerance: float
    comparisons: int
    oracle_id: str

    def __post_init__(self) -> None:
        for name in ("metric_id", "oracle_id"):
            _req_str(getattr(self, name), f"backend_ops.reference.{name}")
        for name in ("observed", "tolerance"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise OutputParseError(
                    f"backend-op reference {name} must be finite and non-negative")
        _req_int(self.comparisons, "backend_ops.reference.comparisons", minimum=1)


@dataclass(frozen=True)
class BackendOpsProperty:
    """One reference-free raw-buffer property residual emitted by the tool."""

    metric_id: str
    residual: float
    tolerance: float
    passed: bool
    suite_seed: int
    transform: str = "identity"
    receipt_version: int = 1

    def __post_init__(self) -> None:
        _req_str(self.metric_id, "backend_ops.property.metric_id")
        for name in ("residual", "tolerance"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise OutputParseError(
                    f"backend-op property {name} must be finite and non-negative")
        if not isinstance(self.passed, bool):
            raise OutputParseError("backend-op property passed must be a bool")
        _req_int(self.suite_seed, "backend_ops.property.suite_seed")
        if self.receipt_version not in (1, 2):
            raise OutputParseError("backend-op property receipt_version must be 1 or 2")
        if self.transform not in ("identity", "x3", "x0p01", "negate"):
            raise OutputParseError(
                "backend-op property transform must be identity|x3|x0p01|negate")
        derived = self.residual <= self.tolerance
        if self.passed != derived:
            raise OutputParseError(
                f"backend-op property {self.metric_id!r} says passed={self.passed} but "
                f"residual={self.residual} and tolerance={self.tolerance} derive {derived}")


@dataclass(frozen=True)
class BackendOpsLayout:
    """The explicitly exercised layout families for one backend-op case."""

    families: tuple
    suite_seed: int

    def __post_init__(self) -> None:
        allowed = {"offset", "transpose", "stride_gap"}
        if not self.families or len(set(self.families)) != len(self.families):
            raise OutputParseError(
                "backend-op layout families must be non-empty and unique")
        if any(item not in allowed for item in self.families):
            raise OutputParseError(
                f"backend-op layout families must be drawn from {sorted(allowed)}")
        _req_int(self.suite_seed, "backend_ops.layout.suite_seed")


@dataclass(frozen=True)
class BackendOpsValueTransforms:
    """The fixed four-transform pass and how many transforms completed."""

    transforms: tuple
    completed: int
    suite_seed: int

    def __post_init__(self) -> None:
        expected = ("identity", "x3", "x0p01", "negate")
        if self.transforms != expected:
            raise OutputParseError(
                f"backend-op value transforms must be exactly {expected}")
        _req_int(self.completed, "backend_ops.value_transforms.completed")
        if self.completed > len(expected):
            raise OutputParseError("backend-op value transforms completed exceeds four")
        _req_int(self.suite_seed, "backend_ops.value_transforms.suite_seed")


@dataclass(frozen=True)
class BackendOpsStateful:
    """The explicit-input, immutable-input, final-output state contract."""

    input_count: int
    initial_equal: bool
    input_immutable: bool
    final_output_count: int
    suite_seed: int

    def __post_init__(self) -> None:
        _req_int(self.input_count, "backend_ops.stateful.input_count")
        _req_bool(self.initial_equal, "backend_ops.stateful.initial_equal")
        _req_bool(self.input_immutable, "backend_ops.stateful.input_immutable")
        _req_int(self.final_output_count, "backend_ops.stateful.final_output_count")
        _req_int(self.suite_seed, "backend_ops.stateful.suite_seed")


@dataclass(frozen=True)
class BackendOpsCase:
    """One `test-backend-ops` case. `not supported` is NOT a pass."""

    op: str
    params: str
    status: str          # ok | fail | not_supported
    interleaved: str = ""
    reference: Optional[BackendOpsReference] = None
    properties: tuple = ()
    layout: Optional[BackendOpsLayout] = None
    value_transforms: Optional[BackendOpsValueTransforms] = None
    stateful: Optional[BackendOpsStateful] = None

    @property
    def passed(self) -> bool:
        return self.status == "ok"


def _case_with_reference(*, op: str, params: str, status: str,
                         interleaved: str = "") -> BackendOpsCase:
    diagnostics: list = []
    reference: Optional[BackendOpsReference] = None
    properties: list = []
    layout: Optional[BackendOpsLayout] = None
    value_transforms: Optional[BackendOpsValueTransforms] = None
    stateful: Optional[BackendOpsStateful] = None
    for part in (piece.strip() for piece in interleaved.split(" | ") if piece.strip()):
        if part.startswith("AK_REF_"):
            if reference is not None:
                raise OutputParseError(
                    f"case {op}({params}) emitted more than one reference receipt")
            match = _REFERENCE_RECEIPT_RE.fullmatch(part)
            if match is None:
                raise OutputParseError(
                    f"case {op}({params}) emitted malformed reference receipt {part!r}")
            reference = BackendOpsReference(
                metric_id=match.group(1), observed=float(match.group(2)),
                tolerance=float(match.group(3)), comparisons=int(match.group(4)),
                oracle_id=match.group(5))
        elif part.startswith("AK_PROP_"):
            match_v2 = _PROPERTY_RECEIPT_V2_RE.fullmatch(part)
            match_v1 = _PROPERTY_RECEIPT_V1_RE.fullmatch(part)
            match = match_v2 or match_v1
            if match is None:
                raise OutputParseError(
                    f"case {op}({params}) emitted malformed property receipt {part!r}")
            properties.append(BackendOpsProperty(
                metric_id=match.group(1), residual=float(match.group(2)),
                tolerance=float(match.group(3)), passed=match.group(4) == "1",
                suite_seed=int(match.group(5)),
                transform=match.group(6) if match_v2 is not None else "identity",
                receipt_version=2 if match_v2 is not None else 1))
        elif part.startswith("AK_LAYOUT_"):
            if layout is not None:
                raise OutputParseError(
                    f"case {op}({params}) emitted more than one layout receipt")
            match = _LAYOUT_RECEIPT_RE.fullmatch(part)
            if match is None:
                raise OutputParseError(
                    f"case {op}({params}) emitted malformed layout receipt {part!r}")
            layout = BackendOpsLayout(
                families=tuple(match.group(1).split(",")), suite_seed=int(match.group(2)))
        elif part.startswith("AK_VALUE_"):
            if value_transforms is not None:
                raise OutputParseError(
                    f"case {op}({params}) emitted more than one value-transform receipt")
            match = _VALUE_RECEIPT_RE.fullmatch(part)
            if match is None:
                raise OutputParseError(
                    f"case {op}({params}) emitted malformed value-transform receipt {part!r}")
            value_transforms = BackendOpsValueTransforms(
                transforms=tuple(match.group(1).split(",")), completed=int(match.group(2)),
                suite_seed=int(match.group(3)))
        elif part.startswith("AK_STATE_"):
            if stateful is not None:
                raise OutputParseError(
                    f"case {op}({params}) emitted more than one stateful receipt")
            match = _STATE_RECEIPT_RE.fullmatch(part)
            if match is None:
                raise OutputParseError(
                    f"case {op}({params}) emitted malformed stateful receipt {part!r}")
            stateful = BackendOpsStateful(
                input_count=int(match.group(1)), initial_equal=match.group(2) == "1",
                input_immutable=match.group(3) == "1",
                final_output_count=int(match.group(4)), suite_seed=int(match.group(5)))
        else:
            diagnostics.append(part)
    if reference is not None and status != "ok":
        raise OutputParseError(
            f"case {op}({params}) attached a passing reference receipt to status {status}")
    if any(not item.passed for item in properties) and status != "fail":
        raise OutputParseError(
            f"case {op}({params}) attached a failed property receipt to status {status}")
    if value_transforms is not None and status == "ok" and value_transforms.completed != 4:
        raise OutputParseError(
            f"case {op}({params}) passed after only {value_transforms.completed}/4 transforms")
    if stateful is not None and status == "ok" and (
            stateful.input_count == 0 or not stateful.initial_equal or
            not stateful.input_immutable or stateful.final_output_count == 0):
        raise OutputParseError(
            f"case {op}({params}) passed without satisfying the stateful triad")
    return BackendOpsCase(op=op, params=params, status=status,
                          interleaved=" | ".join(diagnostics), reference=reference,
                          properties=tuple(properties), layout=layout,
                          value_transforms=value_transforms, stateful=stateful)


@dataclass(frozen=True)
class BackendOpsBackend:
    """One device the run visited — including one it skipped."""

    name: str
    skipped: bool
    skip_reason: Optional[str]
    cases: tuple
    reported_passed: Optional[int]
    reported_total: Optional[int]
    status: Optional[str]


@dataclass(frozen=True)
class BackendOpsRun:
    """A whole `test-backend-ops` invocation, parsed.

    `exercised_ops()` is the load-bearing accessor and it reads CASES ONLY. The
    tool's exit status, its `N/N backends passed` line and its final `OK` are
    all clean when every backend was SKIPPED — measured, not inferred: see the
    module docstring, facts 1 and 2, and `testdata/recorded_t0_backend_ops_console_skip.txt`.
    """

    backends: tuple
    backends_passed: Optional[int]
    backends_total: Optional[int]
    overall: Optional[str]
    failing_tests: tuple

    @property
    def cases(self) -> tuple:
        out: list = []
        for backend in self.backends:
            out.extend(backend.cases)
        return tuple(out)

    @property
    def skipped_backends(self) -> tuple:
        return tuple(b.name for b in self.backends if b.skipped)

    def exercised_ops(self) -> tuple:
        """Ops with at least one SUPPORTED case. Never derived from the summary line.

        Supported, not merely present. A `not supported [CPU]` line means the
        backend declined the shape and no comparison happened; an op whose every
        case was declined was not exercised, and listing it would let
        `check_backend_op_units` tick off a required op that ran nothing. This
        is measured, not hypothetical: the recorded mandatory-op capture
        contains 108 declined `MUL_MAT` shapes beside 178 compared ones.
        """
        seen: list = []
        for case in self.cases:
            if case.status != "not_supported" and case.op not in seen:
                seen.append(case.op)
        return tuple(seen)

    def failed_ops(self) -> tuple:
        return tuple(sorted({c.op for c in self.cases if c.status == "fail"}))

    def interleaved_diagnostics(self) -> tuple:
        """Text that arrived inside a case line — a sanitizer writing to stderr.

        Retained, not discarded: on a sanitizer build these ARE the findings,
        and the recorded capture carries four real UBSAN misaligned-load reports
        from ggml's x86 quant kernels.
        """
        return tuple(dict.fromkeys(c.interleaved for c in self.cases if c.interleaved))

    def unsupported_by_op(self) -> tuple:
        counts: dict = {}
        for case in self.cases:
            if case.status == "not_supported":
                counts[case.op] = counts.get(case.op, 0) + 1
        return tuple(sorted(counts.items()))

    def layout_families(self) -> tuple:
        return tuple(sorted({family for case in self.cases if case.layout is not None
                             for family in case.layout.families}))

    def value_transforms(self) -> tuple:
        return tuple(sorted({transform for case in self.cases
                             if case.value_transforms is not None
                             for transform in case.value_transforms.transforms}))

    def stateful_ops(self) -> tuple:
        return tuple(sorted({case.op for case in self.cases if case.stateful is not None}))

    def cases_by_op(self) -> tuple:
        """`((op, total, passed), ...)` over the cases that were actually COMPARED.

        `total` counts supported cases only, and the reason is a false positive
        that would have made this gate useless. A real `MUL_MAT` run on the CPU
        backend legitimately declines every `type_a=f32,type_b=f16` shape
        (recorded: 108 of 286 lines). Counting those as failures would FAIL
        every honest MUL_MAT run, and a gate that fails on correct input gets
        switched off.

        The coverage question is answered by `exercised_ops()` instead: an op
        with zero supported cases never appears there, so a required op the
        backend declines outright is still a hard FAIL — it is reported as
        "not exercised at all", which is exactly what happened.
        """
        totals: dict = {}
        for case in self.cases:
            if case.status == "not_supported":
                continue
            total, passed = totals.get(case.op, (0, 0))
            totals[case.op] = (total + 1, passed + (1 if case.passed else 0))
        return tuple((op, total, passed) for op, (total, passed) in totals.items())

    def reconcile(self) -> None:
        """Cross-check this parse against the tool's OWN per-backend summary.

        `console_printer::print_summary` prints `  %zu/%zu tests passed` counting
        the cases it compared. If this parser's supported-case count disagrees
        with that number, one of the two is wrong about what ran — and a parser
        that quietly wins that argument is how a partially-read log becomes a
        clean report. Verified against the recorded capture: 286 case lines, 108
        declined, and the tool's own line reads `178/178 tests passed`.
        """
        for backend in self.backends:
            if backend.reported_total is None:
                continue
            compared = [c for c in backend.cases if c.status != "not_supported"]
            passed = [c for c in compared if c.passed]
            if len(compared) != backend.reported_total or len(passed) != backend.reported_passed:
                raise OutputParseError(
                    f"parser/tool disagreement on backend {backend.name!r}: this parser read "
                    f"{len(passed)}/{len(compared)} compared cases, the tool printed "
                    f"{backend.reported_passed}/{backend.reported_total}. One of them is not "
                    "reading the run that happened; refusing to pick a winner.")


def _classify_verdict(verdict: str, line: str) -> tuple:
    """`(status, interleaved_text)` for one case line.

    Returns `("pending", <text>)` when the verdict has not arrived yet — see
    `parse_backend_ops_console`, which then holds the case open.

    DISCOVERED FROM A REAL CAPTURE, not anticipated. The recorded mandatory-op
    run was taken with a UBSAN-instrumented build, and the tool prints a case in
    two halves — `printf("  %s(%s): ")`, `fflush(stdout)`, then the verdict
    (test-backend-ops.cpp:943-944). Anything the sanitizer writes in between
    lands between the halves, and a UBSAN report is MULTI-LINE, so the verdict
    ends up on its own line several lines later:

        MUL_MAT(type_a=q8_0,…): …/x86/quants.c:590:86: runtime error: load of
        misaligned address 0x… for type 'const uint32_t'
        0x…: note: pointer points here
         e7 b4 2e 38 …
        …/x86/quants.c:598:86: runtime error: …
        OK

    The first version of this parser refused that line. Refusing is the right
    default and the wrong answer here: it would make the op suite unreadable on
    exactly the build where it is most worth reading, and it would discard four
    real UBSAN findings in ggml's x86 quant kernels while doing it. The
    interleaved text is retained on the case.

    An unrecognised, non-diagnostic verdict still refuses.
    """
    if verdict.startswith("not supported"):
        return "not_supported", ""
    if verdict == "OK":
        return "ok", ""
    if verdict == "FAIL":
        return "fail", ""
    if verdict.endswith("OK"):
        return "ok", verdict[:-2].strip()
    if verdict.endswith("FAIL"):
        return "fail", verdict[:-4].strip()
    if ": runtime error:" in verdict or "Sanitizer:" in verdict:
        return "pending", verdict
    raise OutputParseError(
        f"unrecognised test-backend-ops case verdict {verdict!r} on line {line!r}. A parser "
        "that shrugs at an unknown verdict reports a clean run; this one refuses instead.")


def parse_backend_ops_console(text: str) -> BackendOpsRun:
    """Parse `--output console`. The format is `console_printer`, verbatim.

    Console rather than CSV is the default for T0 for one measured reason: CSV
    emits a header and per-case rows and NOTHING ELSE — no backend line, no skip
    reason (`csv_printer` has no `print_backend_init` output). A CSV run whose
    backend filter matched no device is byte-identical to a run that had no
    cases to select, and the recorded skip shape is invisible in it. Console
    carries the skip reason, so the parser can refuse it.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    clean = _strip_ansi(text)
    backends: list = []
    name: Optional[str] = None
    skipped = False
    skip_reason: Optional[str] = None
    cases: list = []
    reported: tuple = (None, None)
    status: Optional[str] = None
    backends_passed = backends_total = None
    overall = None
    failing: list = []
    in_failing = False
    saw_frame = False

    def flush() -> None:
        nonlocal name, skipped, skip_reason, cases, reported, status
        if name is not None:
            backends.append(BackendOpsBackend(
                name=name, skipped=skipped, skip_reason=skip_reason, cases=tuple(cases),
                reported_passed=reported[0], reported_total=reported[1], status=status))
        name, skipped, skip_reason, cases, reported, status = None, False, None, [], (None, None), None

    pending: Optional[list] = None    # [op, params, interleaved_lines]

    for raw in clean.splitlines():
        line = raw.rstrip("\n")
        if pending is not None:
            # A case whose verdict has not arrived yet: the tool prints nothing
            # else until it does, so everything until the bare OK/FAIL belongs
            # to this case's diagnostic block.
            stripped = line.strip()
            if stripped in ("OK", "FAIL"):
                cases.append(_case_with_reference(
                    op=pending[0], params=pending[1],
                    status="ok" if stripped == "OK" else "fail",
                    interleaved=" | ".join(p for p in pending[2] if p)))
                pending = None
            else:
                pending[2].append(stripped)
            continue
        if in_failing:
            if line.startswith("  ") and line.strip():
                failing.append(line.strip())
                continue
            in_failing = False
        if _TESTING_RE.match(line):
            saw_frame = True
            continue
        init = _BACKEND_INIT_RE.match(line)
        if init:
            saw_frame = True
            flush()
            name = init.group(3)
            continue
        if name is not None and line.strip() in _SKIP_REASONS and not cases:
            skipped = True
            skip_reason = line.strip()
            continue
        case = _CASE_RE.match(line)
        if case and name is not None:
            state, interleaved = _classify_verdict(case.group(3).strip(), line)
            if state == "pending":
                pending = [case.group(1), case.group(2), [interleaved]]
                continue
            cases.append(_case_with_reference(
                op=case.group(1), params=case.group(2), status=state,
                interleaved=interleaved))
            continue
        tests = _TESTS_PASSED_RE.match(line)
        if tests and name is not None:
            reported = (int(tests.group(1)), int(tests.group(2)))
            continue
        bstatus = _BACKEND_STATUS_RE.match(line)
        if bstatus:
            status = bstatus.group(2)
            continue
        summary = _BACKENDS_PASSED_RE.match(line)
        if summary:
            saw_frame = True
            flush()
            backends_passed, backends_total = int(summary.group(1)), int(summary.group(2))
            continue
        if line.strip() in ("OK", "FAIL") and backends_total is not None and overall is None:
            overall = line.strip()
            continue
        if line.strip() == "Failing tests:":
            in_failing = True
            continue
    if pending is not None:
        raise OutputParseError(
            f"case {pending[0]}({pending[1]}) never received a verdict before the output "
            "ended. A case left open is a case whose result is unknown, and an unknown "
            "result is not a pass.")
    flush()
    if not saw_frame:
        raise OutputParseError(
            "no test-backend-ops console frame found: the output carries no 'Testing N "
            "devices', no 'Backend i/n:' and no 'N/N backends passed' line. Refusing to "
            "report zero cases from output this parser could not read — 'no log' is not "
            "'no findings'.")
    return BackendOpsRun(
        backends=tuple(backends), backends_passed=backends_passed,
        backends_total=backends_total, overall=overall, failing_tests=tuple(failing))


_CSV_ROW_RE = re.compile(r'"((?:[^"]|"")*)"')


def parse_backend_ops_csv(text: str) -> BackendOpsRun:
    """Parse `--output csv`. Provided for completeness; NOT the T0 default.

    Historical recorded header, verbatim:
    `"backend_name","op_name","op_params","test_mode","supported","error_message","backend_reg_name"`.
    `error_message` is `""` on pass and carries the reason on failure
    (test-backend-ops.cpp:1495). Current instruments add a separate optional
    `reference_receipt` column; it never overloads failure semantics. There is no backend-init or skip line in this
    format at all, so a zero-row CSV cannot be distinguished from a skipped run
    — which is why `parse_backend_ops_console` is what the provider uses.
    """
    rows = [line for line in _strip_ansi(text).splitlines() if line.startswith('"')]
    if not rows:
        raise OutputParseError("no CSV rows found in test-backend-ops output")
    header = [cell.replace('""', '"') for cell in _CSV_ROW_RE.findall(rows[0])]
    required = {"backend_name", "op_name", "op_params", "supported", "error_message"}
    missing = sorted(required - set(header))
    if missing:
        raise OutputParseError(
            f"test-backend-ops CSV header is missing {missing}; got {header}. The tool's "
            "csv_printer filters columns through get_fields_csv(), so a header change is a "
            "tool change and not something to parse around.")
    index = {field_name: i for i, field_name in enumerate(header)}
    by_backend: dict = {}
    for row in rows[1:]:
        cells = [cell.replace('""', '"') for cell in _CSV_ROW_RE.findall(row)]
        if len(cells) < len(header):
            continue
        backend = cells[index["backend_name"]]
        supported = cells[index["supported"]] == "1"
        hard_failure = (cells[index["hard_failure"]] == "1"
                        if "hard_failure" in index else False)
        error = cells[index["error_message"]]
        layout_receipt = (cells[index["layout_receipt"]]
                          if "layout_receipt" in index else "")
        value_receipt = (cells[index["value_receipt"]]
                         if "value_receipt" in index else "")
        state_receipt = (cells[index["state_receipt"]]
                         if "state_receipt" in index else "")
        property_receipt = (cells[index["property_receipt"]]
                            if "property_receipt" in index else "")
        reference_receipt = (cells[index["reference_receipt"]]
                             if "reference_receipt" in index else "")
        state = ("fail" if hard_failure else
                 "not_supported" if not supported else
                 "ok" if not error else "fail")
        interleaved = " | ".join(
            value for value in (
                layout_receipt, value_receipt, state_receipt, property_receipt, reference_receipt,
                error) if value)
        by_backend.setdefault(backend, []).append(_case_with_reference(
            op=cells[index["op_name"]], params=cells[index["op_params"]], status=state,
            interleaved=interleaved))
    backends = tuple(
        BackendOpsBackend(name=name, skipped=False, skip_reason=None, cases=tuple(cases),
                          reported_passed=None, reported_total=None, status=None)
        for name, cases in by_backend.items())
    return BackendOpsRun(backends=backends, backends_passed=None, backends_total=None,
                         overall=None, failing_tests=())


@dataclass(frozen=True)
class LinkageRow:
    soname: str
    path: str
    inside_expected_root: bool


@dataclass(frozen=True)
class LinkageReport:
    binary: str
    expected_root: str
    rows: tuple
    verdict: str          # PASS | FAIL
    loader_path: tuple

    @property
    def stray(self) -> tuple:
        return tuple(r for r in self.rows if not r.inside_expected_root)


_LINK_ROW_RE = re.compile(r"^ {2}(OK|BAD)\s+(\S+)\s+->\s+(\S+)\s*$")
_LINK_BIN_RE = re.compile(r"^binary\s*:\s*(\S+)\s*$")
_LINK_EXPECT_RE = re.compile(r"^expect\s*:\s*libraries under (\S+)\s*$")


def parse_linkage_report(text: str) -> LinkageReport:
    """Parse `scripts/utils/verify_ggml_linkage.sh` output.

    Tested against BOTH recorded shapes: the PASS taken with the launcher's own
    `LD_LIBRARY_PATH`, and the FAIL taken under this container's ambient one, in
    which five libraries — all of ggml — resolve out of the frozen production
    tree. The FAIL fixture is not hypothetical; it is what the experimental
    binary does today if a launcher forgets.
    """
    clean = _strip_ansi(text)
    binary = expected = None
    rows: list = []
    loader: list = []
    verdict = None
    in_loader = False
    for line in clean.splitlines():
        match = _LINK_BIN_RE.match(line)
        if match:
            binary = match.group(1)
            continue
        match = _LINK_EXPECT_RE.match(line)
        if match:
            expected = match.group(1)
            continue
        match = _LINK_ROW_RE.match(line)
        if match:
            rows.append(LinkageRow(soname=match.group(2), path=match.group(3),
                                   inside_expected_root=match.group(1) == "OK"))
            continue
        if line.startswith("LD_LIBRARY_PATH order"):
            in_loader = True
            continue
        if in_loader:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                loader.append(parts[1])
                continue
            if line.strip():
                in_loader = False
        if line.startswith("PASS:"):
            verdict = schemas.PASS
        elif line.startswith("FAIL:"):
            verdict = schemas.FAIL
    if binary is None or expected is None or verdict is None:
        raise OutputParseError(
            "verify_ggml_linkage.sh output is missing its binary/expect header or its "
            "PASS/FAIL verdict. Refusing to report 'no stray libraries' from output this "
            "parser could not read: an unreadable linkage report is not a clean one.")
    return LinkageReport(binary=binary, expected_root=expected, rows=tuple(rows),
                         verdict=verdict, loader_path=tuple(loader))


_ASAN_RE = re.compile(r"^.*(?:ERROR|WARNING): AddressSanitizer: (.+)$")
_LSAN_RE = re.compile(r"^.*ERROR: LeakSanitizer: (.+)$")
_UBSAN_RE = re.compile(r"^(.*?):\s*runtime error:\s*(.+)$")


def parse_sanitizer_findings(text: str) -> tuple:
    """Return `(asan_findings, ubsan_findings)` from a sanitizer log.

    LeakSanitizer findings are filed under ASAN because they are memory-safety
    defects and `T0_GATE_SPEC` files ASAN under `stability`; UBSAN's
    `runtime error:` lines are undefined behaviour and land in
    `numerical_safety`. Two lists, because the record must say WHICH surface.
    """
    asan: list = []
    ubsan: list = []
    for line in _strip_ansi(text).splitlines():
        match = _ASAN_RE.match(line)
        if match:
            asan.append(f"AddressSanitizer: {match.group(1).strip()}")
            continue
        match = _LSAN_RE.match(line)
        if match:
            asan.append(f"LeakSanitizer: {match.group(1).strip()}")
            continue
        match = _UBSAN_RE.match(line)
        if match:
            ubsan.append(f"{match.group(1).strip()}: runtime error: {match.group(2).strip()}")
    return tuple(asan), tuple(ubsan)


_DIAG_RE = re.compile(r"^(?P<file>[^\s:]+):(?:\d+:)?(?:\d+:)?\s*(?P<sev>error|warning):\s*(?P<msg>.+)$")


def parse_compiler_diagnostics(text: str) -> tuple:
    """Return `(error_count, warning_count, findings)` from a build log.

    Counts DISTINCT diagnostic lines. A build log replays the same warning once
    per translation unit that includes the header, and counting repeats would
    make the anchor-vs-candidate warning delta a function of build parallelism.
    """
    errors: list = []
    warnings: list = []
    for line in _strip_ansi(text).splitlines():
        match = _DIAG_RE.match(line.strip())
        if not match:
            continue
        entry = f"{match.group('file')}: {match.group('msg').strip()}"
        target = errors if match.group("sev") == "error" else warnings
        if entry not in target:
            target.append(entry)
    return len(errors), len(warnings), tuple(errors)


_SPLIT_RE = re.compile(r"^## SPLIT #(\d+): (\S+) # (\d+) inputs")
_NODE_RE = re.compile(r"^node #\s*(\d+) \(\s*(\S+)\s*\):\s+(\S+)\s+\(\s*\S+\s*\)\s+\[\s*(\S+)\s+(\S+)\s*\]")


def parse_sched_trace(text: str) -> tuple:
    """Parse `GGML_SCHED_DEBUG=2` output.

    Returns `(instrument_emitted, splits, nodes)` where `nodes` is
    `((index, op, name, backend, cause), ...)`.

    `instrument_emitted` is a separate return value and not an inference from
    `len(nodes)`: an empty node list from a run that never enabled the
    instrument and an empty node list from a graph with no nodes are the same
    tuple, and `DispatchTraceEvidence.fallback_instrumentation_active` is the
    field that decides whether the no-fallback gate PASSes or reads
    COULD_NOT_CHECK. The `## SPLIT` marker is the proof the scheduler printed.
    """
    clean = _strip_ansi(text)
    splits: list = []
    nodes: list = []
    emitted = False
    for line in clean.splitlines():
        stripped = line.strip()
        match = _SPLIT_RE.match(stripped)
        if match:
            emitted = True
            splits.append((int(match.group(1)), match.group(2), int(match.group(3))))
            continue
        match = _NODE_RE.match(stripped)
        if match:
            emitted = True
            nodes.append((int(match.group(1)), match.group(2), match.group(3),
                          match.group(4), match.group(5)))
    return emitted, tuple(splits), tuple(nodes)


#: `correctness.BuildProvenance.build_log_ref` is a *ref*, and this module is the
#: only consumer that dereferences it. `execution/chain.py` emits
#: `file://<abs path>#sha256=<digest>` so the ref names the log's CONTENT rather
#: than a mutable path — six weeks later a bare path resolves to whatever is
#: there then. A bare absolute path is still accepted, because that is what a
#: hand-written provenance carries.
_BUILD_LOG_REF_RE = re.compile(r"^file://(?P<path>/[^#]*)(?:#sha256=(?P<sha>[0-9a-f]{64}))?$")


def resolve_build_log_ref(ref: Any) -> Optional[str]:
    """The local path a `build_log_ref` names, or `None` when it names none.

    `None` is a distinguishable outcome and its caller RECORDS it. Returning
    `None` silently — which is what an unguarded `Path(ref).read_text()` inside a
    bare `except OSError` did — made an unresolvable ref indistinguishable from
    an unmeasured anchor toolchain, and both came out as the same
    COULD_NOT_CHECK with the wrong reason attached.
    """
    if not isinstance(ref, str) or not ref.strip():
        return None
    match = _BUILD_LOG_REF_RE.match(ref.strip())
    if match:
        return match.group("path")
    if ref.startswith("/") and "#" not in ref:
        return ref
    return None


_EVAL_TOKENS_RE = re.compile(r"eval time =\s*[\d.]+ ms /\s*(\d+)\s+runs")


def parse_delivered_tokens(text: str) -> Optional[int]:
    """Decoded tokens from llama.cpp's own perf print, or `None` if absent.

    `llama_perf_context_print: eval time = %10.2f ms / %5d runs`
    (src/llama-context.cpp:4075). `None` means the line was not there, which is
    COULD_NOT_CHECK downstream; it is never coerced to 0, because 0 delivered
    units is a control-3 finding and 'we did not read it' is not one.
    """
    matches = _EVAL_TOKENS_RE.findall(_strip_ansi(text))
    if not matches:
        return None
    return int(matches[-1])


# =============================================================================
# The anchor capture — the only source of an anchor triple
# =============================================================================

@dataclass(frozen=True)
class AnchorCapture:
    """Everything measured off the anchor, including its three-component identity.

    This object is the reason a partial anchor triple cannot be produced by this
    module. It cannot exist without all three components (`__post_init__`
    refuses), so `_anchor_triple()` either has one of these and emits three
    values, or has `None` and emits three `None`s. There is no code path that
    fills in two.

    Every component is MEASURED: `binary_sha256` is the hash of the anchor
    binary's bytes, `linkage_sha256` is the digest of the anchor's own resolved
    library table, and `source_commit` is the commit the anchor tree reports.
    None of them is copied from `request.anchor` — see the module docstring.
    """

    source_commit: str
    binary_sha256: str
    linkage_sha256: str
    resolved_libraries: tuple = ()
    output_digests: tuple = ()
    output_lengths: tuple = ()
    determinism_class: str = "not_measured"
    compiler_id: Optional[str] = None
    compiler_version: Optional[str] = None
    warning_count: Optional[int] = None
    delivered_units: Optional[int] = None
    oracle_ids: tuple = ()
    capture_refs: tuple = ()
    #: Anything observed while measuring the anchor that the fields above cannot
    #: carry — chiefly an anchor generation that did not complete, whose digest
    #: is therefore absent from `output_digests`.
    notes: tuple = ()

    def __post_init__(self) -> None:
        try:
            _req_commit(self.source_commit, "anchor_capture.source_commit")
            _req_sha256(self.binary_sha256, "anchor_capture.binary_sha256")
            _req_sha256(self.linkage_sha256, "anchor_capture.linkage_sha256")
        except ValueError as exc:
            raise AnchorCaptureIncomplete(
                f"{exc}. Precondition 4 names an anchor by source commit, binary SHA-256 AND "
                "linkage SHA-256. This object exists so that a partially named anchor is "
                "unconstructible rather than merely discouraged: two of three components is "
                "not a weaker name, it is a different one, and it resolves to more than one "
                "artifact."
            ) from exc
        if self.determinism_class not in schemas.DETERMINISM_CLASSES:
            raise ValueError(
                f"anchor_capture.determinism_class: {self.determinism_class!r} is not one of "
                f"{sorted(schemas.DETERMINISM_CLASSES)}")

    def identity(self) -> api.AnchorIdentity:
        return api.AnchorIdentity(source_commit=self.source_commit,
                                  binary_sha256=self.binary_sha256,
                                  linkage_sha256=self.linkage_sha256)

    def first_output_digest(self) -> Optional[str]:
        return self.output_digests[0] if self.output_digests else None

    def first_output_length(self) -> Optional[int]:
        return self.output_lengths[0] if self.output_lengths else None


def _anchor_triple(capture: Optional[AnchorCapture]) -> tuple:
    """`(commit, binary, linkage)` or `(None, None, None)`. There is no third result.

    Every anchor field on every evidence type this module produces is filled
    from this ONE call. That is what makes `_validate_anchor_triple`'s
    all-or-none rule structural here rather than a convention three producers
    have to remember separately — the shape that drifts is the one with fewer
    tests, and this way there is only one shape.
    """
    if capture is None:
        return (None, None, None)
    if not isinstance(capture, AnchorCapture):
        raise TypeError("anchor capture must be an AnchorCapture or None")
    return (capture.source_commit, capture.binary_sha256, capture.linkage_sha256)


# =============================================================================
# Argv construction — every command carries a receipt
# =============================================================================

@dataclass(frozen=True)
class ConstructedInvocation:
    """argv + fully declared env + the receipt that binds them.

    Precondition 6's `HAND_TYPED_ARGV` void condition is detected by the ABSENCE
    of a receipt, so every command this module runs carries one — including the
    ones `recipes.py` has no registered recipe for (T0's correctness-mode
    `test-backend-ops` is not in the T1a/T1b registry, which only registers
    `perf`).
    """

    constructor_id: str
    argv: tuple
    env: tuple
    receipt: api.RecipeReceipt
    notes: tuple = ()

    def env_dict(self) -> dict:
        return dict(self.env)


def _module_sha256() -> str:
    try:
        return schemas.content_hash({
            "module": "autokernel.execution.t0_provider",
            "source": Path(__file__).read_text(encoding="utf-8"),
        })
    except OSError as exc:
        raise ExecutionError(
            f"cannot read {__file__} to compute this constructor's content hash: {exc}. "
            "Precondition 6 requires the constructor's identifier AND content hash on the "
            "record; there is no fallback value for it.") from exc


def _receipt(constructor_id: str, argv: Sequence[str], env: Sequence[tuple]) -> api.RecipeReceipt:
    return api.RecipeReceipt(
        constructor_id=constructor_id,
        constructor_sha256=_module_sha256(),
        argv_sha256=schemas.content_hash({"argv": list(argv), "env": [list(p) for p in env]}),
    )


_REGISTERED_PARAMETER_ENV = {"GGML_IQK": frozenset({"0", "1"})}


def _validated_parameter_env(parameter_env: Sequence[tuple]) -> dict:
    out: dict = {}
    for row in parameter_env:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise TypeError("parameter_env rows must be (name, value) pairs")
        key, value = str(row[0]), str(row[1])
        if key in out:
            raise ValueError(f"parameter_env names {key!r} more than once")
        choices = _REGISTERED_PARAMETER_ENV.get(key)
        if choices is None or value not in choices:
            raise ValueError(
                f"parameter_env {key}={value!r} is not a registered arm-local variant")
        out[key] = value
    return out


def _launch_env(library_path: str, base_env: Sequence[tuple],
                extra: Optional[Mapping[str, str]] = None,
                parameter_env: Sequence[tuple] = ()) -> tuple:
    """The launch environment: the binary's OWN library path first, nothing inherited.

    `LD_LIBRARY_PATH` is set to the binary's own directory and is not appended
    to anything ambient. That is not belt-and-braces: on this host today, the
    container's ambient `LD_LIBRARY_PATH` still carries
    `/mnt/raid0/llm/llama.cpp/build/bin`, and an experimental binary launched
    with it resolves all five of its ggml libraries out of the frozen production
    tree (recorded in `testdata/recorded_t0_linkage_fail.txt`). Every launcher
    sets its own, and proves it — which is what `_collect_linkage` then does.
    """
    env: dict = {str(k): str(v) for k, v in base_env}
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    # Applied after the canonical environment, but only through the closed
    # registry above.  This is the one-factor arm difference; ``base_env`` is
    # not allowed to win last-writer merely because it happened to carry the
    # same spelling.
    env.update(_validated_parameter_env(parameter_env))
    # The pin is applied LAST and refuses to be named twice. It used to be
    # applied BEFORE `extra`, so any caller-supplied mapping silently won
    # last-wins — and `collect_sanitizers` passes a mapping it did not build
    # itself (`SanitizerInvocation.env`). A pin a later dict can lift is a
    # convention, not a pin, and the thing it lifts is the difference between
    # measuring the candidate and measuring the frozen production tree.
    if extra and "LD_LIBRARY_PATH" in {str(k) for k in extra}:
        raise ExecutionError(
            "the extra environment sets LD_LIBRARY_PATH. A poisoned `base_env` is fine — the "
            "pin below defeats it, which is the whole point — but `extra` is where a mapping "
            "this module did not build itself arrives (`collect_sanitizers` forwards "
            "`SanitizerInvocation.env`), and it used to be applied AFTER the pin and win "
            "last-wins. A pin a later dict can lift is a convention, and what it lifts is the "
            "difference between measuring the candidate and measuring the frozen production "
            "tree (INC-20260731, still reproducible on this host).")
    env["LD_LIBRARY_PATH"] = str(Path(library_path))
    return tuple(sorted(env.items()))


def build_backend_ops_invocation(*, binary: str, library_path: str, backend_filter: str,
                                 ops: Sequence[str], base_env: Sequence[tuple],
                                 suite_seed: int = 0,
                                 layout_probe: bool = False,
                                 value_transform_probe: bool = False,
                                 stateful_probe: bool = False,
                                 parameter_env: Sequence[tuple] = (),
                                 output_format: str = "console",
                                 params_filter: Optional[str] = None,
                                 parallel_workers: Optional[int] = None,
                                 cpu_prefix: bool = True) -> ConstructedInvocation:
    """`test-backend-ops test` — the CORRECTNESS mode. Not in the T1a registry.

    `recipes.py` registers `test-backend-ops perf` for T1a and nothing for T0:
    the registry is a SPEED-recipe registry, and its `perf` builder emits
    `perf`, `--output` formats chosen for timing fields, and a
    `raw_samples_retained` finding about per-repetition sample vectors. None of
    that describes a correctness run. The canonical CPU prefix and OMP stack are
    still imported from `recipes` (which imports them from the ratified
    `scripts/lib/canonical_recipe.py`) rather than retyped, so the pinned
    footprint and the env stack are the ratified ones.

    `-b <filter>` is mandatory here and the plan refuses to omit it. Measured
    reason in the module docstring, fact 2.
    """
    _req_abs(binary, "backend_ops.binary")
    _req_str(backend_filter, "backend_ops.backend_filter")
    if not ops:
        raise ValueError("backend_ops.ops must be non-empty")
    _req_int(suite_seed, "backend_ops.suite_seed")
    _req_bool(layout_probe, "backend_ops.layout_probe")
    _req_bool(value_transform_probe, "backend_ops.value_transform_probe")
    _req_bool(stateful_probe, "backend_ops.stateful_probe")
    if sum((layout_probe, value_transform_probe, stateful_probe)) > 1:
        raise ValueError(
            "layout, value-transform, and stateful probes must be separate invocations")
    argv: list = list(recipes.CANONICAL_PREFIX) if cpu_prefix else []
    argv += [binary, "test", "-o", ",".join(ops), "-b", backend_filter]
    argv += ["--suite-seed", str(suite_seed), "--autokernel-properties"]
    if layout_probe:
        argv += ["--autokernel-layouts"]
    if value_transform_probe:
        argv += ["--autokernel-value-transforms"]
    if stateful_probe:
        argv += ["--autokernel-stateful"]
    if params_filter is not None:
        argv += ["-p", params_filter]
    if parallel_workers is not None:
        argv += ["-j", str(parallel_workers)]
    argv += ["--output", output_format]
    extra = dict(recipes.CANONICAL_OMP_ENV) if cpu_prefix else {}
    env = _launch_env(library_path, base_env, extra, parameter_env)
    constructor_id = "ak.t0.backend_ops_test/v3"
    return ConstructedInvocation(
        constructor_id=constructor_id, argv=tuple(argv), env=env,
        receipt=_receipt(constructor_id, argv, env),
        notes=(
            "correctness mode ('test'), not the T1a 'perf' recipe",
            "-b is explicit: test mode skips the CPU device outright without it "
            "(test-backend-ops.cpp:10372) and still prints OK",
            "--suite-seed fixes every tensor input for deterministic replay",
            "--autokernel-properties runs independent raw-buffer host-double properties "
            "and their planted-defect self-test",
            ("--autokernel-layouts selects only layout-variant cases and makes an "
             "unsupported layout a hard failure" if layout_probe else
             "the independent layout-variant pass is not requested"),
            ("--autokernel-value-transforms runs the fixed identity/x3/x0.01/negate "
             "fail-any pass" if value_transform_probe else
             "the independent value-transform pass is not requested"),
            ("--autokernel-stateful proves equal and immutable explicit state inputs and "
             "compares final state outputs" if stateful_probe else
             "the independent stateful pass is not requested"),
            "canonical prefix and OMP stack imported from evaluator.recipes, never retyped",
        ))


def build_linkage_invocation(*, bash: str, script: str, binary: str, expected_root: str,
                             library_path: str,
                             base_env: Sequence[tuple],
                             parameter_env: Sequence[tuple] = ()) -> ConstructedInvocation:
    """`verify_ggml_linkage.sh <binary> <root>`, under the launcher's own LD_LIBRARY_PATH.

    The script reports what the LOADER would do, so it must run under the same
    environment the measurement runs under. Running it with an ambient
    `LD_LIBRARY_PATH` and then measuring with a pinned one — or the reverse —
    verifies a linkage nothing was measured with.
    """
    _req_abs(bash, "linkage.bash")
    _req_abs(script, "linkage.script")
    _req_abs(binary, "linkage.binary")
    argv = [bash, script, binary, str(Path(expected_root))]
    env = _launch_env(library_path, base_env, parameter_env=parameter_env)
    constructor_id = "ak.t0.verify_ggml_linkage/v1"
    return ConstructedInvocation(
        constructor_id=constructor_id, argv=tuple(argv), env=env,
        receipt=_receipt(constructor_id, argv, env),
        notes=("runs under the SAME LD_LIBRARY_PATH the measurement runs under",))


def build_generation_invocation(*, binary: str, library_path: str, plan: GenerationPlan,
                                base_env: Sequence[tuple], seed: Optional[int] = None,
                                extra_env: Optional[Mapping[str, str]] = None,
                                parameter_env: Sequence[tuple] = (),
                                cpu_prefix: bool = True) -> ConstructedInvocation:
    """One llama-cli generation. Sampling parameters are EXPLICIT in the argv.

    `--temp`, `--top-k` and `--seed` are always emitted, never left to the
    binary's defaults, because `GenerationPlan.is_greedy()` reads the plan and
    the argv must be the same statement. `--no-warmup` and `-no-cnv` keep the
    captured stdout the generation and nothing else.
    """
    _req_abs(binary, "generation.binary")
    argv: list = list(recipes.CANONICAL_PREFIX) if cpu_prefix else []
    argv += [binary, "-p", plan.prompt, "-n", str(plan.n_predict),
             "--seed", str(plan.seed if seed is None else seed),
             "--temp", repr(float(plan.temperature)), "--top-k", str(plan.top_k),
             "--no-warmup", "-no-cnv"]
    if plan.threads is not None:
        argv += ["-t", str(plan.threads)]
    argv += list(plan.extra_argv)
    extra = dict(recipes.CANONICAL_OMP_ENV) if cpu_prefix else {}
    if extra_env:
        extra.update(extra_env)
    env = _launch_env(library_path, base_env, extra, parameter_env)
    constructor_id = "ak.t0.generation/v1"
    return ConstructedInvocation(
        constructor_id=constructor_id, argv=tuple(argv), env=env,
        receipt=_receipt(constructor_id, argv, env),
        notes=(f"greedy={plan.is_greedy()} (temp=={plan.temperature}, top_k=={plan.top_k})",))


# =============================================================================
# The provider
# =============================================================================

@dataclass
class _Collected:
    """Scratch space for one `evidence_for` call. Never shared between calls."""

    refs: list = field(default_factory=list)
    orphans: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    #: Every capture this collection took, in order. Held HERE and not read back
    #: out of the sink: `_delivered_units` used to require the sink to be a
    #: `MemoryCaptureSink`, so passing the durable sink a real campaign needs
    #: silently turned the delivered-work reading off.
    captures: list = field(default_factory=list)

    def ref(self, *, first: bool = False) -> str:
        """A citable `*_ref`, or a named absence. Never an empty string.

        Every evidence type validates `receipt_ref` as a non-empty string, and
        for good reason: `""` renders in a record as a reference that exists and
        resolves to nothing. Evidence assembled from declarations rather than
        from one capture says so, in a form a reader can act on.
        """
        if not self.refs:
            return f"{PROVIDER_ID}:no-capture"
        return self.refs[0] if first else self.refs[-1]


class ExecutedT0EvidenceProvider:
    """The real `correctness.T0EvidenceProvider`: it runs the tools.

    Wiring:

        provider = ExecutedT0EvidenceProvider(plan=plan, runner=SubprocessRunner(),
                                              claim=cpu_region_claim)
        runner = correctness.T0CorrectnessRunner(provider=provider, policy=policy)
        report = runner.evaluate(request)

    Nothing about it is host-specific: `runner` is the only execution seam and
    `RecordedProcessRunner` satisfies it, so the whole collection path is
    exercised from recorded output with no host, no claim on real hardware and
    no build. That is how it is tested, and it is also invariant 11's replay
    path.
    """

    def __init__(self, *, plan: T0ExecutionPlan, runner: Any,
                 claim: Any = None, sink: Any = None,
                 anchor_capture: Optional[AnchorCapture] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        if not isinstance(plan, T0ExecutionPlan):
            raise TypeError("plan must be a T0ExecutionPlan")
        if not hasattr(runner, "run"):
            raise TypeError("runner must implement run(argv, env=, cwd=, timeout_s=)")
        if anchor_capture is not None and not isinstance(anchor_capture, AnchorCapture):
            raise TypeError("anchor_capture must be an AnchorCapture or None")
        self._plan = plan
        self._runner = runner
        self._claim = claim
        self._sink = sink if sink is not None else MemoryCaptureSink()
        self._anchor = anchor_capture
        self._clock = clock or time.time

    # -- plumbing ---------------------------------------------------------

    @property
    def plan(self) -> T0ExecutionPlan:
        return self._plan

    @property
    def anchor_capture(self) -> Optional[AnchorCapture]:
        return self._anchor

    def _pinned_cpus(self) -> Optional[str]:
        """The cpu list this plan's argv pins, or `None` when it pins none.

        Every CPU-backend invocation this module builds carries
        `recipes.CANONICAL_PREFIX`, so the footprint is the canonical one and
        the claim must cover it.
        """
        return _canonical_cpu_list() if self._plan.backend == "llama_cpu" else None

    def _execute(self, invocation: ConstructedInvocation, *, timeout_s: float,
                 collected: _Collected, cwd: Optional[str] = None) -> tuple:
        """Run one constructed invocation and store its capture. Returns `(capture, ref)`."""
        capture = self._runner.run(
            invocation.argv, env=invocation.env_dict(),
            cwd=cwd or self._plan.candidate.worktree, timeout_s=timeout_s)
        if not isinstance(capture, CompletedProcess):
            raise TypeError(
                f"runner returned {type(capture).__name__}, expected CompletedProcess")
        ref = self._sink.store(capture)
        collected.refs.append(ref)
        collected.captures.append(capture)
        if capture.orphans:
            collected.orphans.extend(capture.orphans)
        return capture, ref

    # -- op suite ---------------------------------------------------------

    def collect_op_suite(self, collected: _Collected) -> correctness.OpSuiteEvidence:
        """Run `test-backend-ops test` and report ONLY what it actually exercised.

        A claim is required even though this runs no inference. The argv carries
        the canonical prefix, so it pins `taskset -c 0-95` and occupies the
        machine for as long as the suite runs; precondition 1 wants *"a CPU
        region claim covering the exact footprint measured"* and the footprint
        here is the whole of it. `verify_ggml_linkage.sh` is the exception this
        module makes — an `ldd` and a loop over its output take no core.
        """
        plan = self._plan.op_suite
        self._op_suite_reference = None
        require_claim(self._claim, what="the backend-op suite", cpu_list=self._pinned_cpus())
        invocation = build_backend_ops_invocation(
            binary=self._plan.candidate.test_backend_ops,
            library_path=self._plan.candidate.library_path,
            backend_filter=plan.backend_filter, ops=plan.ops,
            base_env=self._plan.base_env,
            suite_seed=plan.suite_seed,
            layout_probe=plan.layout_probe,
            value_transform_probe=plan.value_transform_probe,
            stateful_probe=plan.stateful_probe,
            parameter_env=self._plan.parameter_env,
            parallel_workers=plan.parallel_workers,
            cpu_prefix=self._plan.backend == "llama_cpu")
        capture, ref = self._execute(invocation, timeout_s=plan.timeout_s, collected=collected)
        # stdout ONLY. The case grammar is a stdout construct; stderr carries
        # loader lines and, on a sanitizer build, diagnostics that land mid-line
        # (see `_classify_verdict`). Parsing the merged stream would make the
        # case list a function of how the two happened to interleave.
        run = parse_backend_ops_console(capture.stdout or capture.combined)
        run.reconcile()
        for diagnostic in run.interleaved_diagnostics():
            collected.notes.append(f"op suite: diagnostic inside a case line: {diagnostic}")
        for op, count in run.unsupported_by_op():
            collected.notes.append(
                f"op suite: {op} had {count} shape(s) declined by backend "
                f"{plan.backend_filter} ('not supported'); those were not compared and are "
                "not counted as passes")
        missing = tuple(op for op in plan.ops if op not in run.exercised_ops())
        if missing:
            collected.notes.append(
                f"op suite: op(s) {list(missing)} were REQUESTED and never exercised — the "
                f"tool exited {capture.exit_code} and reported "
                f"{run.backends_passed}/{run.backends_total} backends passed anyway. An "
                "untested op is not a passing op.")
        if run.skipped_backends:
            collected.notes.append(
                f"op suite: backend(s) {list(run.skipped_backends)} were SKIPPED and "
                f"contributed no cases; the tool still reports "
                f"{run.backends_passed}/{run.backends_total} backends passed and exits "
                f"{capture.exit_code}")
        comparisons = tuple(
            correctness.ReferenceComparison(
                shape_id=f"{case.op}({case.params})#{ordinal}",
                op=case.op,
                mode="metric_bounded",
                mismatch_count=0,
                max_ulp_observed=None,
                tolerance_ulp=None,
                oracle_id=case.reference.oracle_id,
                oracle_is_candidate_derived=False,
                metric_id=case.reference.metric_id,
                max_error_observed=case.reference.observed,
                tolerance_error=case.reference.tolerance)
            for ordinal, case in enumerate(run.cases)
            if case.reference is not None)
        if comparisons:
            self._op_suite_reference = correctness.ReferenceEvidence(
                comparisons=comparisons,
                undefined_for=(),
                oracle_registry_ref=f"{ref}#backend-ops-reference-v1",
                produced_by=PRODUCER)
        property_measurements: list = []
        case_ordinal = 0
        for backend in run.backends:
            for case in backend.cases:
                shape_id = f"{case.op}({case.params})#{case_ordinal}"
                for property_result in case.properties:
                    property_measurements.append(correctness.PropertyMeasurement(
                        shape_id=shape_id,
                        op=case.op,
                        backend=backend.name,
                        metric_id=property_result.metric_id,
                        residual=property_result.residual,
                        tolerance=property_result.tolerance,
                        suite_seed=property_result.suite_seed,
                        passed=property_result.passed,
                        input_transform=property_result.transform))
                case_ordinal += 1
        layout_cases = tuple(case for case in run.cases if case.layout is not None)
        for case in layout_cases:
            if case.layout.suite_seed != plan.suite_seed:
                raise OutputParseError(
                    f"layout receipt for {case.op}({case.params}) carries suite_seed "
                    f"{case.layout.suite_seed}, expected {plan.suite_seed}")
        value_cases = tuple(case for case in run.cases
                            if case.value_transforms is not None)
        if plan.value_transform_probe and len(value_cases) != len(run.cases):
            missing = tuple(
                f"{case.op}({case.params})" for case in run.cases
                if case.value_transforms is None)
            raise OutputParseError(
                "value-transform pass emitted case(s) without AK_VALUE_V1 receipt: "
                f"{missing}")
        for case in value_cases:
            if case.value_transforms.suite_seed != plan.suite_seed:
                raise OutputParseError(
                    f"value-transform receipt for {case.op}({case.params}) carries suite_seed "
                    f"{case.value_transforms.suite_seed}, expected {plan.suite_seed}")
            legacy_properties = tuple(
                item.metric_id for item in case.properties if item.receipt_version != 2)
            if legacy_properties:
                raise OutputParseError(
                    f"value-transform receipt for {case.op}({case.params}) has property "
                    f"measurement(s) without AK_PROP_V2 transform binding: {legacy_properties}")
        stateful_cases = tuple(case for case in run.cases if case.stateful is not None)
        if plan.stateful_probe and len(stateful_cases) != len(run.cases):
            missing = tuple(
                f"{case.op}({case.params})" for case in run.cases if case.stateful is None)
            raise OutputParseError(
                "stateful pass emitted case(s) without AK_STATE_V1 receipt: "
                f"{missing}")
        for case in stateful_cases:
            if case.stateful.suite_seed != plan.suite_seed:
                raise OutputParseError(
                    f"stateful receipt for {case.op}({case.params}) carries suite_seed "
                    f"{case.stateful.suite_seed}, expected {plan.suite_seed}")
        return correctness.OpSuiteEvidence(
            suite_id=plan.suite_id,
            suite_source_sha256=plan.suite_source_sha256,
            suite_seed=plan.suite_seed,
            ops_exercised=run.exercised_ops(),
            ops_failed=run.failed_ops(),
            cases_by_op=run.cases_by_op(),
            shapes_ref=f"test-backend-ops:{plan.backend_filter}:{','.join(plan.ops)}",
            receipt_ref=ref,
            produced_by=PRODUCER,
            property_measurements=tuple(property_measurements),
            layout_probe=plan.layout_probe,
            layout_families=run.layout_families(),
            layout_case_count=len(layout_cases),
            value_transform_probe=plan.value_transform_probe,
            value_transforms=run.value_transforms(),
            value_transform_case_count=len(value_cases),
            stateful_probe=plan.stateful_probe,
            stateful_ops=run.stateful_ops(),
            stateful_case_count=len(stateful_cases),
        )

    # -- boundary shapes ---------------------------------------------------

    def collect_boundary_shapes(self, collected: _Collected):
        """Unseen/boundary shapes via `-p <case filter>`. `None` when no holdout is planned."""
        holdout = self._plan.holdout
        if holdout is None:
            return None
        require_claim(self._claim, what="the unseen/boundary shape suite", cpu_list=self._pinned_cpus())
        unseen_cases: list = []
        boundary_cases: list = []
        failures: list = []
        refs: list = []
        for label, case_filter, bucket in (
                ("unseen", holdout.unseen_case_filter, unseen_cases),
                ("boundary", holdout.boundary_case_filter, boundary_cases)):
            invocation = build_backend_ops_invocation(
                binary=self._plan.candidate.test_backend_ops,
                library_path=self._plan.candidate.library_path,
                backend_filter=self._plan.op_suite.backend_filter,
                ops=self._plan.op_suite.ops,
                base_env=self._plan.base_env,
                suite_seed=self._plan.op_suite.suite_seed,
                parameter_env=self._plan.parameter_env,
                params_filter=case_filter,
                cpu_prefix=self._plan.backend == "llama_cpu")
            capture, ref = self._execute(invocation, timeout_s=holdout.timeout_s,
                                         collected=collected)
            refs.append(ref)
            # stdout, and reconciled — the same two rules `collect_op_suite`
            # states and justifies. This path used to parse `capture.combined`
            # and skip `reconcile()`, so the one run that a `-p` shape filter can
            # silently empty (the builder's own recorded finding: `-p` excluded
            # every MUL_MAT_ID case and the tool still printed `OK`) was the one
            # run with no cross-check against the tool's own count.
            run = parse_backend_ops_console(capture.stdout or capture.combined)
            run.reconcile()
            for case in run.cases:
                name = f"{case.op}({case.params})"
                if case.status == "not_supported":
                    # A declined shape was not compared, so it is neither a pass
                    # nor a failure. Counting it as a failure FAILs
                    # `check_unseen_boundary_shapes` on the recorded, honest CPU
                    # run in which 108 of 286 MUL_MAT shapes are legitimately
                    # declined — and a gate that fails on correct input is a gate
                    # that gets switched off. `collect_op_suite` already draws
                    # this line; this path drew the opposite one.
                    collected.notes.append(
                        f"boundary shapes: {label} case {name} was declined by backend "
                        f"{self._plan.op_suite.backend_filter} ('not supported'); it was not "
                        "compared and is recorded as neither a shape exercised nor a failure")
                    continue
                bucket.append(name)
                if not case.passed:
                    failures.append(f"{label}:{name}:{case.status}")
        return correctness.BoundaryShapeEvidence(
            unseen_shapes=tuple(unseen_cases),
            boundary_shapes=tuple(boundary_cases),
            failures=tuple(failures),
            selection_rule_id=holdout.selection_rule_id,
            selection_seed=holdout.selection_seed,
            held_out_from_planner=not holdout.visible_to_planner,
            receipt_ref=refs[0] if refs else collected.ref(),
            produced_by=PRODUCER,
        )

    # -- dispatch trace ----------------------------------------------------

    def collect_dispatch_trace(self, collected: _Collected):
        """Collect `GGML_SCHED_DEBUG` output. `None` when there is nothing to trace."""
        plan = self._plan.dispatch
        generation = self._plan.generation
        if generation is None:
            return None
        require_claim(self._claim, what="the dispatch trace", cpu_list=self._pinned_cpus())
        invocation = build_generation_invocation(
            binary=self._plan.candidate.binary,
            library_path=self._plan.candidate.library_path,
            plan=generation, base_env=self._plan.base_env,
            extra_env={"GGML_SCHED_DEBUG": str(plan.debug_level)},
            parameter_env=self._plan.parameter_env,
            cpu_prefix=self._plan.backend == "llama_cpu")
        capture, ref = self._execute(invocation, timeout_s=plan.timeout_s, collected=collected)
        emitted, _splits, nodes = parse_sched_trace(capture.combined)
        derived = set(plan.derived_surface)
        affected_nodes = tuple(node for node in nodes if node[1] in derived)
        traced = tuple(dict.fromkeys(op for _, op, _, _, _ in affected_nodes))
        ignored = tuple(dict.fromkeys(op for _, op, _, _, _ in nodes if op not in derived))
        if ignored:
            collected.notes.append(
                f"dispatch trace: ignored {len(ignored)} executed op kind(s) outside the "
                f"mechanically derived affected surface: {list(ignored)}")
        # An op assigned to a backend other than the one under test is an
        # inter-backend fallback; that is the only class this instrument sees.
        expected_backend = self._plan.op_suite.backend_filter
        fallback_events = tuple(
            f"node #{index} {op} ran on {backend} (cause {cause}), not {expected_backend}"
            for index, op, _, backend, cause in affected_nodes
            if backend not in (expected_backend, "NULL"))
        in_scope = plan.fallback_scope == DispatchTracePlan.INSTRUMENTED_SCOPE
        if not in_scope:
            collected.notes.append(
                f"dispatch trace: campaign fallback scope {plan.fallback_scope!r} is outside "
                f"what GGML_SCHED_DEBUG can observe ({DispatchTracePlan.INSTRUMENTED_SCOPE}); "
                "reporting the instrumentation as INACTIVE so the no-fallback gate reads "
                "COULD_NOT_CHECK rather than PASS")
        return correctness.DispatchTraceEvidence(
            derived_surface=tuple(plan.derived_surface),
            traced_kernels=traced,
            fallback_events=fallback_events if (emitted and in_scope) else (),
            fallback_instrumentation_active=bool(emitted and in_scope),
            trace_ref=ref,
            produced_by=PRODUCER,
        )

    # -- linkage -----------------------------------------------------------

    def _collect_linkage_report(self, *, binary: str, library_path: str, expected_root: str,
                                collected: _Collected) -> tuple:
        invocation = build_linkage_invocation(
            bash=self._plan.tools.bash,
            script=self._plan.tools.verify_ggml_linkage_sh,
            binary=binary, expected_root=expected_root, library_path=library_path,
            base_env=self._plan.base_env, parameter_env=self._plan.parameter_env)
        capture, ref = self._execute(invocation, timeout_s=120.0, collected=collected)
        return parse_linkage_report(capture.combined), ref

    @staticmethod
    def linkage_digest(report: LinkageReport) -> str:
        """The `linkage_sha256` this module means: a digest of the RESOLVED table.

        Defined here, once, and used for both arms. It is a hash of
        `((soname, path), ...)` sorted — the identity of what the loader
        actually bound, not of the binary and not of the env string. Two builds
        with identical binaries that resolve different libraries have different
        linkage, which is the whole point of the field.
        """
        # A linkage report is tool-shaped, not build-shaped.  ``llama-cli``
        # normally reports libggml.so, libggml-base and libggml-cpu; asking
        # ldd about libggml.so itself reports only its direct base/cpu edges.
        # Those are one build, but the *sets* differ.  The shared ABI root is
        # libggml-base: fold its resolved path together with the verifier's
        # expected root.  This proves the common ggml generation while neither
        # treating a CLI-only dependency nor a direct-child-only dependency as
        # a second build.  Backend/tool-specific rows stay in the full receipt
        # and are still checked for confinement by the verifier.
        base_rows = [row for row in report.rows
                     if row.soname in {"libggml-base.so", "libggml-base.so.0"}]
        if len(base_rows) != 1:
            raise OutputParseError(
                "the linkage verifier must report exactly one libggml-base row; "
                "the common resolved ggml generation cannot be identified")
        return schemas.content_hash({
            "expected_root": report.expected_root,
            "resolved_ggml_base": [base_rows[0].soname, base_rows[0].path],
        })

    def collect_linkage(self, collected: _Collected):
        """`LinkageEvidence` for the candidate, with the anchor triple from the capture."""
        candidate = self._plan.candidate
        report, ref = self._collect_linkage_report(
            binary=candidate.binary, library_path=candidate.library_path,
            expected_root=candidate.library_path, collected=collected)
        rows = tuple((row.soname, row.path, sha256_file(row.path)) for row in report.rows)
        commit, binary_sha, linkage_sha = _anchor_triple(self._anchor)
        return correctness.LinkageEvidence(
            binary_sha256=sha256_file(candidate.binary),
            linkage_sha256=self.linkage_digest(report),
            anchor_source_commit=commit,
            anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            resolved_libraries=rows,
            expected_library_root=str(Path(candidate.library_path)),
            verifier_id=f"verify_ggml_linkage.sh@{sha256_file(self._plan.tools.verify_ggml_linkage_sh)[:12]}",
            receipt_ref=ref,
            produced_by=PRODUCER,
        )

    # -- sanitizers --------------------------------------------------------

    def collect_sanitizers(self, collected: _Collected):
        """Build and RUN the ASAN/UBSAN invocation `correctness` constructs.

        The argv is not retyped here: `correctness.build_sanitizer_invocation`
        emits it with a receipt, `check_sanitizer_invocation` refuses it if it
        would be fail-open, and this method only executes it. That split is the
        protocol's — a module that decides whether the sanitizer would gate must
        not be the module that runs it.
        """
        plan = self._plan
        if plan.sanitizer_target is None or plan.sanitizer_build_dir is None:
            return None
        require_claim(self._claim, what="the ASAN/UBSAN build and targeted run")
        # ONE expression for the instrumented artifact, used for the argv that
        # runs it, the LD_LIBRARY_PATH it runs under, and the digest recorded as
        # `sanitizer_build_binary_sha256`. It used to be three: the run named
        # `candidate.test_backend_ops` — the ORDINARY, uninstrumented build —
        # under the ordinary build's ggml, while the digest was taken off the
        # instrumented binary that was built and never executed. The build
        # succeeded, the uninstrumented binary ran clean, zero findings were
        # parsed, `executed=True` and exit 0 were recorded, and BOTH mandatory
        # gates PASSed. `_sanitizer_preamble`'s "the sanitizer build's binary is
        # not the record's binary" guard was satisfied the whole time — by a
        # hash of a file nothing ran. That is the guard-on-the-wrong-input shape,
        # and the only structural answer is to make the three unable to differ.
        sanitizer_binary = str(Path(plan.sanitizer_build_dir) / "bin" / plan.sanitizer_target)
        sanitizer_lib_dir = str(Path(sanitizer_binary).parent)
        run_argv = (
            sanitizer_binary, "test",
            "-o", ",".join(plan.op_suite.ops),
            "-b", plan.op_suite.backend_filter,
        )
        invocation = correctness.build_sanitizer_invocation(
            source_dir=plan.candidate.worktree,
            build_dir=plan.sanitizer_build_dir,
            target=plan.sanitizer_target,
            run_argv=run_argv,
            jobs=plan.sanitizer_jobs,
            backend=plan.backend,
        )
        gate = correctness.check_sanitizer_invocation(invocation)
        if gate.outcome != schemas.PASS:
            raise ExecutionError(
                "refusing to run a sanitizer invocation that would not gate: "
                + "; ".join(gate.reasons))
        if plan.tools.cmake is None:
            raise ExecutionError(
                "a sanitizer build was planned but tools.cmake is unset; the configure and "
                "build argv name `cmake` and this module resolves no tool through PATH")
        sanitizer_env = dict(invocation.env)
        exit_code: Optional[int] = None
        log_ref = ""
        findings_text = ""
        for stage_argv in (invocation.configure_argv, invocation.build_argv,
                           invocation.run_argv):
            argv = list(stage_argv)
            if argv[0] == "cmake":
                argv[0] = plan.tools.cmake
            is_run = tuple(stage_argv) == tuple(invocation.run_argv)
            # The RUN loads the instrumented tree's own ggml. Pointing it at
            # `candidate.library_path` would have run the sanitizer binary
            # against the uninstrumented libraries the finding would be in.
            env = _launch_env(
                sanitizer_lib_dir if is_run else str(Path(plan.sanitizer_build_dir)),
                plan.base_env, sanitizer_env, plan.parameter_env)
            stage = ConstructedInvocation(
                constructor_id=invocation.constructor_id, argv=tuple(argv), env=env,
                receipt=invocation.receipt)
            capture, ref = self._execute(stage, timeout_s=plan.sanitizer_timeout_s,
                                         collected=collected)
            findings_text += capture.combined + "\n"
            exit_code = capture.exit_code
            log_ref = ref
            if capture.exit_code != 0 and not is_run:
                # A configure/build that failed produced no instrumented binary,
                # so the "targeted run" would run something else or nothing. It
                # used to continue to the run stage and note the failure; an
                # empty findings list from a build that never happened reads as
                # a clean sanitizer surface.
                raise ExecutionError(
                    f"sanitizer stage {argv[0]!r} exited {capture.exit_code}; refusing to "
                    "report a sanitizer surface from a build that did not complete. An empty "
                    "findings list from an absent instrumented binary is indistinguishable "
                    "from a clean one, and §8.6 makes this surface MANDATORY, not advisory. "
                    f"stderr tail: {capture.stderr.strip().splitlines()[-3:]}")
        asan, ubsan = parse_sanitizer_findings(findings_text)
        try:
            sanitizer_binary_sha = sha256_file(sanitizer_binary)
        except CaptureUnavailable as exc:
            supplied = plan.sanitizer_binary_sha256
            if supplied is None:
                # The previous fallback hashed the LOG. That produces a
                # well-formed sha256 that names no binary at all, and it is the
                # value `_sanitizer_preamble` compares against
                # `request.artifact.binary_sha256` to prove the two builds are
                # distinct — a comparison that then always passes, whatever was
                # built. An identity that cannot be measured is absent, and this
                # field has no absent value, so the collection refuses.
                raise ExecutionError(
                    f"the instrumented binary {sanitizer_binary} could not be hashed after a "
                    f"successful build ({exc}). `sanitizer_build_binary_sha256` names the "
                    "binary that RAN; substituting any other digest — the log's included — "
                    "makes `_sanitizer_preamble`'s distinct-builds guard unfalsifiable. "
                    "Supply `plan.sanitizer_binary_sha256` if it was measured elsewhere."
                ) from exc
            sanitizer_binary_sha = supplied
            collected.notes.append(
                f"sanitizer: {sanitizer_binary} was not hashable in this process; the digest "
                "recorded is the one the caller measured (plan.sanitizer_binary_sha256), not "
                "a stand-in")
        return correctness.SanitizerEvidence(
            invocation=invocation,
            executed=True,
            exit_code=exit_code,
            asan_findings=asan,
            ubsan_findings=ubsan,
            sanitizer_build_binary_sha256=sanitizer_binary_sha,
            log_ref=log_ref,
            produced_by=PRODUCER,
        )

    # -- generations: coherence and determinism ----------------------------

    @staticmethod
    def _generation_defect(capture: CompletedProcess) -> Optional[str]:
        """Why this generation's OUTPUT may not be treated as an observation.

        Nothing here used to look at how a generation ended, and the digest of a
        run that produced nothing is a perfectly good sha256. Three llama-cli
        invocations that segfault instantly hash to the same empty-string digest
        three times, which `DeterminismEvidence.measured_class()` reads as
        `bitwise_stable` — a candidate that cannot run at all scoring the
        strongest determinism class there is. The same digest against an anchor
        that also crashed gives `token_agreement_ratio == 1.0`: perfect
        coherence between two things that never generated a token.

        A failed run is still RECORDED (the capture and its ref are kept and the
        note says what happened); what it is not is a measurement of the
        candidate's output.
        """
        if capture.timed_out:
            return f"timed out after {capture.duration_s:.1f}s"
        if capture.signalled:
            return "was signalled by this runner before it finished"
        if capture.exit_code != 0:
            return f"exited {capture.exit_code}"
        if not (capture.stdout or "").strip():
            return "produced no stdout"
        return None

    def _generate(self, *, binary: str, library_path: str, seed: Optional[int],
                  collected: _Collected) -> tuple:
        plan = self._plan.generation
        invocation = build_generation_invocation(
            binary=binary, library_path=library_path, plan=plan,
            base_env=self._plan.base_env, seed=seed,
            parameter_env=self._plan.parameter_env,
            cpu_prefix=self._plan.backend == "llama_cpu")
        capture, ref = self._execute(invocation, timeout_s=plan.timeout_s, collected=collected)
        return capture, ref

    def collect_coherence(self, collected: _Collected):
        """Candidate generation vs the anchor's, with the anchor named honestly."""
        plan = self._plan.generation
        if plan is None:
            return None
        require_claim(self._claim, what="the coherence generation", cpu_list=self._pinned_cpus())
        capture, ref = self._generate(binary=self._plan.candidate.binary,
                                      library_path=self._plan.candidate.library_path,
                                      seed=None, collected=collected)
        text = capture.stdout
        defect = self._generation_defect(capture)
        if defect is not None:
            collected.notes.append(
                f"coherence: the candidate generation {defect}; its output is recorded as "
                "ABSENT rather than compared. A partial or empty output compared byte-wise "
                "is a coherence verdict about a run that did not happen — the gate reads "
                "'not compared', which is what was established")
            text = ""
        commit, binary_sha, linkage_sha = _anchor_triple(self._anchor)
        anchor_digest = None if self._anchor is None else self._anchor.first_output_digest()
        anchor_len = None if self._anchor is None else self._anchor.first_output_length()
        anchor_class = None if self._anchor is None else self._anchor.determinism_class
        ratio = None
        if anchor_digest is not None and text:
            ratio = 1.0 if sha256_text(text) == anchor_digest else 0.0
        return correctness.CoherenceEvidence(
            candidate_output_sha256=sha256_text(text) if text else None,
            candidate_output_len=len(text),
            anchor_output_sha256=anchor_digest,
            anchor_output_len=anchor_len,
            sampler_id=f"llama-cli/temp={plan.temperature},top_k={plan.top_k}",
            sampler_is_greedy=plan.is_greedy(),
            seed=plan.seed,
            tokens_requested=plan.n_predict,
            token_agreement_ratio=ratio,
            divergence_first_index=None,
            anchor_determinism_class=anchor_class,
            anchor_source_commit=commit,
            anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            prompt_ref=plan.prompt_ref,
            receipt_ref=ref,
            produced_by=PRODUCER,
        )

    def collect_determinism(self, collected: _Collected):
        """Same-seed repeats on the candidate; anchor digests from the anchor capture."""
        plan = self._plan.generation
        runs = self._plan.determinism_runs
        if plan is None or runs < 1:
            return None
        require_claim(self._claim, what="the determinism repeats", cpu_list=self._pinned_cpus())
        digests: list = []
        ref = ""
        for index in range(runs):
            capture, ref = self._generate(binary=self._plan.candidate.binary,
                                          library_path=self._plan.candidate.library_path,
                                          seed=plan.seed, collected=collected)
            defect = self._generation_defect(capture)
            if defect is not None:
                # NOT a digest. `check_determinism_class` already refuses a run
                # count that disagrees with the digest count ("the reduction
                # cannot be recomputed from that"), so a failed repeat lands on
                # that existing FAIL instead of contributing an identical
                # empty-output digest that reads as bitwise stability.
                collected.notes.append(
                    f"determinism: same-seed run {index + 1}/{runs} {defect}; no output "
                    "digest is recorded for it. A run that produced nothing is not a "
                    "bitwise-stable repeat of a run that produced nothing")
                continue
            digests.append(sha256_text(capture.stdout))
        commit, binary_sha, linkage_sha = _anchor_triple(self._anchor)
        anchor_digests = () if self._anchor is None else self._anchor.output_digests
        anchor_class = "not_measured" if self._anchor is None else self._anchor.determinism_class
        return correctness.DeterminismEvidence(
            seed=plan.seed,
            runs=runs,
            candidate_output_digests=tuple(digests),
            anchor_output_digests=tuple(anchor_digests),
            anchor_determinism_class=anchor_class,
            anchor_source_commit=commit,
            anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            declared_class_change=False,
            declared_class_change_ref=None,
            receipt_ref=ref,
            produced_by=PRODUCER,
        )

    # -- static analysis ---------------------------------------------------

    def collect_static_analysis(self, collected: _Collected):
        """Compiler diagnostics for both arms. `None` without an anchor capture.

        `StaticAnalysisEvidence.anchor_compiler_id` and
        `anchor_compiler_version` are REQUIRED non-empty strings, and they
        describe the ANCHOR's build. With no anchor capture there is no honest
        value for them, and inventing one would be a toolchain comparison
        against a toolchain nobody measured. `None` here means the gate reads
        COULD_NOT_CHECK, which is what was established.
        """
        build = self._plan.build
        anchor = self._anchor
        if anchor is None or anchor.compiler_id is None or anchor.compiler_version is None:
            collected.notes.append(
                "static analysis: no anchor toolchain was captured, so no "
                "StaticAnalysisEvidence is produced; the gate reads COULD_NOT_CHECK rather "
                "than comparing against an unmeasured anchor build")
            return None
        if build is None or getattr(build, "build_log_ref", None) is None:
            return None
        ref = getattr(build, "build_log_ref")
        log_path = resolve_build_log_ref(ref)
        if log_path is None:
            collected.notes.append(
                f"static analysis: build_log_ref {ref!r} is not a resolvable local path, so "
                "the compiler diagnostics could not be read. The gate reads COULD_NOT_CHECK "
                "for THIS reason and not for the absent-anchor one — an unresolvable ref and "
                "an unmeasured anchor toolchain used to be the same silent `return None`.")
            return None
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            collected.notes.append(
                f"static analysis: build_log_ref {ref!r} resolved to {log_path!r} and could "
                f"not be read ({exc}); the gate reads COULD_NOT_CHECK because the log was "
                "unreachable, which is not the same finding as 'no diagnostics were emitted'")
            return None
        errors, warnings, findings = parse_compiler_diagnostics(text)
        commit, binary_sha, linkage_sha = _anchor_triple(anchor)
        return correctness.StaticAnalysisEvidence(
            # Attribute access, not `getattr(build, "compiler_id", "unknown")`.
            # `plan.build` is type-checked to a `correctness.BuildProvenance`,
            # whose two compiler fields are required non-empty, so the fallback
            # was reachable only from the WRONG record — and the wrong record
            # here is `integrity.BuildProvenance`, which spells its compiler
            # `compiler` and its log ref `build_log_path`. It produced
            # `compiler_id="unknown"`, a value no anchor can equal, so the gate
            # FAILed with "the candidate was built with unknown unknown but the
            # anchor with GNU 15.2.0" — a toolchain confound reported about a
            # candidate whose toolchain was never read. The plan refuses the
            # wrong record now; this reads the right one directly.
            compiler_id=build.compiler_id,
            compiler_version=build.compiler_version,
            anchor_compiler_id=anchor.compiler_id,
            anchor_compiler_version=anchor.compiler_version,
            error_count=errors,
            warning_count=warnings,
            anchor_warning_count=anchor.warning_count,
            anchor_source_commit=commit,
            anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            warnings_as_errors=build.warnings_as_errors,
            analyzer_id=None,
            analyzer_error_findings=findings,
            receipt_ref=str(log_path),
            produced_by=PRODUCER,
        )

    # -- state safety ------------------------------------------------------

    def collect_state_safety(self, collected: _Collected):
        """Teardown/rollback observation over the processes THIS provider launched.

        `orphan_processes` is the honest one: it is the set of pids this module
        spawned, signalled, and then FAILED to confirm dead. Invariant 10
        requires the loop to verify termination of what it launched, and a
        provider that reported an empty list because it never checked would be
        the fail-open shape.
        """
        if not self._plan.state_safety_probe:
            collected.notes.append(f"state safety: probe OFF. {STATE_SAFETY_CANNOT_PASS}")
            return None
        collected.notes.append(f"state safety: probe ON. {STATE_SAFETY_CANNOT_PASS}")
        return correctness.StateSafetyEvidence(
            rollback_tested=False,
            teardown_tested=True,
            race_detector_id=None,
            race_findings=(),
            leaked_resources=(),
            orphan_processes=tuple(sorted(set(collected.orphans))),
            receipt_ref=collected.ref(),
            produced_by=PRODUCER,
        )

    # -- anti-reward-hacking ----------------------------------------------

    def collect_anti_reward_hacking(self, delivered_candidate: Optional[int],
                                    collected: _Collected):
        """Control 3's detector, filled from what the run actually declared."""
        anchor = self._anchor
        if delivered_candidate is None:
            # `AntiRewardHackingEvidence.delivered_units_candidate` is `int`, not
            # `Optional[int]`, so "not read" has no representation and 0 is the
            # only sayable value — which reads as "delivered nothing", a control-3
            # finding. The distinction survives in the record as a note, and the
            # schema gap is reported in SCHEMA_FOLLOWUPS rather than papered over.
            collected.notes.append(
                "anti-reward-hacking: no delivered-unit count was READ from any capture "
                "(llama.cpp\'s `eval time = ... / N runs` line was absent); the field records "
                "0 because the schema has no absent value, and 0 here means UNREAD, not "
                "'the candidate delivered nothing'")
        commit, binary_sha, linkage_sha = _anchor_triple(anchor)
        oracle_ids = tuple(self._plan.oracle_ids) or (
            () if anchor is None else tuple(anchor.oracle_ids))
        scan = (None if self._plan.candidate_diff_text is None
                else reward_hack_scan.scan_unified_diff(
                    self._plan.candidate_diff_text))
        return correctness.AntiRewardHackingEvidence(
            cache_state=self._plan.cache_state,
            correctness_verdict_source=PRODUCER,
            candidate_output_used_as_oracle=False,
            oracle_ids=oracle_ids,
            delivered_unit_name=self._plan.delivered_unit_name,
            delivered_units_candidate=0 if delivered_candidate is None else delivered_candidate,
            delivered_units_anchor=None if anchor is None else anchor.delivered_units,
            anchor_source_commit=commit,
            anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            environment_probe_findings=(
                () if scan is None else scan.environment_probe_findings),
            timing_dependent_branch_findings=(
                () if scan is None else scan.timing_dependent_branch_findings),
            stream_creation_findings=(
                () if scan is None else scan.stream_creation_findings),
            async_escape_findings=(
                () if scan is None else scan.async_escape_findings),
            instrument_frame_findings=(
                () if scan is None else scan.instrument_frame_findings),
            pointer_memoization_findings=(
                () if scan is None else scan.pointer_memoization_findings),
            structured_short_circuit_findings=(
                () if scan is None else scan.structured_short_circuit_findings),
            receipt_ref=collected.ref(),
            environment_probe_detector_id=(
                None if scan is None else scan.environment_probe_detector_id),
            timing_dependent_branch_detector_id=(
                None if scan is None else scan.timing_dependent_branch_detector_id),
            stream_creation_detector_id=(
                None if scan is None else scan.stream_creation_detector_id),
            async_escape_detector_id=(
                None if scan is None else scan.async_escape_detector_id),
            instrument_frame_detector_id=(
                None if scan is None else scan.instrument_frame_detector_id),
            pointer_memoization_detector_id=(
                None if scan is None else scan.pointer_memoization_detector_id),
            structured_short_circuit_detector_id=(
                None if scan is None else scan.structured_short_circuit_detector_id),
        )

    # -- the Protocol method ----------------------------------------------

    def evidence_for(self, request: api.EvaluationRequest) -> correctness.T0Evidence:
        """Collect every T0 surface for one candidate. The `T0EvidenceProvider` seam."""
        if not isinstance(request, api.EvaluationRequest):
            raise TypeError("request must be an api.EvaluationRequest")
        collected = _Collected()
        surface = self._change_surface()

        op_suite = self.collect_op_suite(collected)
        boundary = self.collect_boundary_shapes(collected)
        dispatch = self.collect_dispatch_trace(collected)
        linkage = self.collect_linkage(collected)
        sanitizers = self.collect_sanitizers(collected)
        coherence = self.collect_coherence(collected)
        determinism = self.collect_determinism(collected)
        static_analysis = self.collect_static_analysis(collected)
        state_safety = self.collect_state_safety(collected)
        delivered = self._delivered_units(collected)
        anti_reward = self.collect_anti_reward_hacking(delivered, collected)

        self._notes = tuple(collected.notes)
        self._refs = tuple(collected.refs)
        return correctness.T0Evidence(
            control_role=None,
            change_surface=surface,
            symbols=self._plan.symbols,
            build=self._plan.build,
            diff=self._plan.diff,
            static_analysis=static_analysis,
            sanitizers=sanitizers,
            op_suite=op_suite,
            reference=(self._plan.reference if self._plan.reference is not None
                       else self._op_suite_reference),
            boundary_shapes=boundary,
            dispatch_trace=dispatch,
            state_safety=state_safety,
            coherence=coherence,
            determinism=determinism,
            linkage=linkage,
            anti_reward_hacking=anti_reward,
            source_candidate=bool((self._plan.candidate_diff_text or "").strip()),
            source_prerequisites=self._plan.source_prerequisites,
            projection_checks=self._plan.projection_checks,
        )

    @property
    def notes(self) -> tuple:
        """Collection notes from the last `evidence_for` call."""
        return getattr(self, "_notes", ())

    @property
    def capture_refs(self) -> tuple:
        return getattr(self, "_refs", ())

    def _delivered_units(self, collected: _Collected) -> Optional[int]:
        """Tokens the candidate actually generated, from llama.cpp's own perf print.

        Reads the captures this collection took, not the sink. The sink is a
        STORAGE seam: which one is installed decides where a capture is written,
        never whether a measurement happened. Gating the read on
        `isinstance(self._sink, MemoryCaptureSink)` meant the durable sink a real
        campaign installs returned `None` for every candidate, and `None` then
        became `delivered_units_candidate=0` — a control-3 "the candidate
        delivered less work than the anchor" FAIL manufactured by a storage
        choice.
        """
        for capture in reversed(collected.captures):
            tokens = parse_delivered_tokens(capture.combined)
            if tokens is not None:
                return tokens
        return None

    def _change_surface(self) -> correctness.ChangeSurface:
        """The mechanically derived surface. `surface.py` owns the real derivation.

        This provider passes through whatever the derivation produced. It never
        fills a `derived_touches_*` flag in from the plan, because
        `ChangeSurface`'s `None` means "the derivation did not determine it" and
        every consumer fails closed on it — turning that into a `False` here
        would PASS a behavioural surface on a fact nobody established.

        `plan.change_surface` is where the derivation arrives, by way of
        `chain.change_surface_from(derive_affected_surface(...), diff_text=…)`.
        It is a declared field on `T0ExecutionPlan` and is type-checked there.
        Until 2026-08-04 this method read it with
        `getattr(self._plan, "change_surface", None)` against a dataclass that
        had no such attribute, so the pass-through this docstring describes was
        DEAD: every candidate got the all-`None` surface below and four gates
        read COULD_NOT_CHECK with no way for a caller to supply otherwise.
        """
        existing = self._plan.change_surface
        if isinstance(existing, correctness.ChangeSurface):
            return existing
        return correctness.ChangeSurface(
            derived_touches_memory=None,
            derived_touches_threading=None,
            derived_touches_dispatch=None,
            derived_touches_persistent_state=None,
            derived_ops=tuple(self._plan.dispatch.derived_surface),
            derived_files=(),
            declared_touches_memory=None,
            declared_touches_threading=None,
            declared_ops=(),
            touches_shared_core_header=False,
            derivation_ref="autokernel.execution.t0_provider: no surface derivation supplied; "
                           "every derived_touches_* is UNDETERMINED and fails closed",
        )


# =============================================================================
# Anchor capture
# =============================================================================

_CMAKE_COMPILER_ID_RE = re.compile(
    r'^set\(CMAKE_CXX_COMPILER_ID\s+"(?P<value>[^"]+)"\)\s*$', re.MULTILINE)
_CMAKE_COMPILER_VERSION_RE = re.compile(
    r'^set\(CMAKE_CXX_COMPILER_VERSION\s+"(?P<value>[^"]+)"\)\s*$', re.MULTILINE)


def _measure_cmake_toolchain(binary: str) -> tuple:
    """Read the toolchain CMake recorded for the build containing ``binary``."""
    build_root = Path(binary).resolve().parent.parent
    compiler_files = sorted(
        (build_root / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if not compiler_files:
        return None, None
    measured = set()
    for path in compiler_files:
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CaptureUnavailable(
                f"anchor toolchain metadata {path} is unreadable: {exc}") from exc
        id_match = _CMAKE_COMPILER_ID_RE.search(body)
        version_match = _CMAKE_COMPILER_VERSION_RE.search(body)
        if id_match is None or version_match is None:
            raise CaptureUnavailable(
                f"anchor toolchain metadata {path} does not record both CXX compiler "
                "identity and version")
        measured.add((f"CXX {id_match.group('value')}", version_match.group("value")))
    if len(measured) != 1:
        raise CaptureUnavailable(
            f"anchor build {build_root} has conflicting CXX toolchains: {sorted(measured)}")
    return next(iter(measured))


def _complete_anchor_toolchain(binary: str, compiler_id: Optional[str],
                               compiler_version: Optional[str]) -> tuple:
    if (compiler_id is None) != (compiler_version is None):
        raise ValueError(
            "anchor compiler_id and compiler_version must be supplied together or measured "
            "together")
    if compiler_id is not None:
        return compiler_id, compiler_version
    return _measure_cmake_toolchain(binary)

def capture_anchor_identity(*, anchor: AnchorBuild, tools: ToolPaths, runner: Any,
                            base_env: Sequence[tuple] = (), sink: Any = None,
                            parameter_env: Sequence[tuple] = (),
                            compiler_id: Optional[str] = None,
                            compiler_version: Optional[str] = None,
                            warning_count: Optional[int] = None) -> AnchorCapture:
    """Measure one anchor tool's immutable identity without executing inference.

    T0 and T1 use different binaries.  The full ``capture_anchor`` entry point
    is intentionally shaped around a T0 plan because it may also execute seeded
    generation.  T1 needs only the hash and resolved-library table for
    ``llama-bench``; forcing callers to fabricate an op-suite plan to obtain
    those two read-only measurements made the stock campaign adapter impossible
    to wire honestly.
    """
    if not isinstance(anchor, AnchorBuild):
        raise TypeError("capture_anchor_identity.anchor must be an AnchorBuild")
    if not isinstance(tools, ToolPaths):
        raise TypeError("capture_anchor_identity.tools must be ToolPaths")
    compiler_id, compiler_version = _complete_anchor_toolchain(
        anchor.binary, compiler_id, compiler_version)
    sink = sink if sink is not None else MemoryCaptureSink()
    invocation = build_linkage_invocation(
        bash=tools.bash, script=tools.verify_ggml_linkage_sh,
        binary=anchor.binary, expected_root=anchor.library_path,
        library_path=anchor.library_path, base_env=base_env,
        parameter_env=parameter_env)
    capture = runner.run(
        invocation.argv, env=invocation.env_dict(), cwd=anchor.worktree, timeout_s=120.0)
    if not isinstance(capture, CompletedProcess):
        raise TypeError(
            f"runner returned {type(capture).__name__}, expected CompletedProcess")
    ref = sink.store(capture)
    report = parse_linkage_report(capture.combined)
    return AnchorCapture(
        source_commit=anchor.source_commit,
        binary_sha256=sha256_file(anchor.binary),
        linkage_sha256=ExecutedT0EvidenceProvider.linkage_digest(report),
        resolved_libraries=tuple((row.soname, row.path) for row in report.rows),
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        warning_count=warning_count,
        capture_refs=(ref,),
    )

def capture_anchor(*, plan: T0ExecutionPlan, runner: Any, claim: Any = None,
                   sink: Any = None, generation_seeds: Sequence[int] = (),
                   compiler_id: Optional[str] = None,
                   compiler_version: Optional[str] = None,
                   warning_count: Optional[int] = None,
                   oracle_ids: Sequence[str] = ()) -> AnchorCapture:
    """Measure the anchor's identity and behaviour. The ONLY producer of a triple.

    Runs read-only against the anchor tree, which may be a frozen production
    tree: executing a frozen binary is not a write, and it is how anchoring
    works. Nothing here builds, writes, or touches the anchor's branch or index.

    `generation_seeds` drives the anchor's same-seed repeats. Zero seeds is
    legitimate and yields `determinism_class="not_measured"` with no output
    digests — the anchor's identity is still fully measured, so anchor-derived
    IDENTITY evidence (linkage) is available while anchor-derived BEHAVIOURAL
    evidence (coherence, determinism) honestly reports that nothing was run.
    """
    if plan.anchor is None:
        raise ExecutionError(
            "capture_anchor was called with no AnchorBuild in the plan. There is no way to "
            "measure an anchor's identity without an anchor; an evidence set with no anchor "
            "capture records no anchor components at all, which is the correct answer.")
    anchor = plan.anchor
    compiler_id, compiler_version = _complete_anchor_toolchain(
        anchor.binary, compiler_id, compiler_version)
    sink = sink if sink is not None else MemoryCaptureSink()
    collected = _Collected()
    provider = ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=claim, sink=sink)
    report, _ = provider._collect_linkage_report(
        binary=anchor.binary, library_path=anchor.library_path,
        expected_root=anchor.library_path, collected=collected)
    digests: list = []
    lengths: list = []
    notes: list = []
    delivered: Optional[int] = None
    if generation_seeds:
        if plan.generation is None:
            raise ExecutionError(
                "anchor generation seeds were supplied but the plan has no GenerationPlan")
        require_claim(claim, what="the anchor generation",
                      cpu_list=provider._pinned_cpus())
        for seed in generation_seeds:
            capture, _ = provider._generate(binary=anchor.binary,
                                            library_path=anchor.library_path,
                                            seed=seed, collected=collected)
            defect = ExecutedT0EvidenceProvider._generation_defect(capture)
            if defect is not None:
                # Same rule as the candidate side. Two crashed anchor runs used
                # to hash identically and certify the ANCHOR as `bitwise_stable`,
                # which is the class every candidate comparison is then made
                # against. Fewer than two usable digests falls through to
                # `not_measured`, which is the truth.
                notes.append(f"anchor generation for seed {seed} {defect}; no digest recorded")
                continue
            digests.append(sha256_text(capture.stdout))
            lengths.append(len(capture.stdout))
            tokens = parse_delivered_tokens(capture.combined)
            if tokens is not None:
                delivered = tokens
    if len(digests) >= 2:
        determinism_class = ("bitwise_stable" if all(d == digests[0] for d in digests)
                             else "bitwise_unstable")
    else:
        determinism_class = "not_measured"
    return AnchorCapture(
        source_commit=anchor.source_commit,
        binary_sha256=sha256_file(anchor.binary),
        linkage_sha256=ExecutedT0EvidenceProvider.linkage_digest(report),
        resolved_libraries=tuple((row.soname, row.path) for row in report.rows),
        output_digests=tuple(digests),
        output_lengths=tuple(lengths),
        determinism_class=determinism_class,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        warning_count=warning_count,
        delivered_units=delivered,
        oracle_ids=tuple(oracle_ids),
        capture_refs=tuple(collected.refs),
        notes=tuple(notes),
    )


# =============================================================================
# Self-audit
# =============================================================================

#: Name-pattern process tools. Every one of them takes a NAME and matches
#: whatever else on this shared host happens to carry it — including a guard
#: process whose argv necessarily contains the names it guards
#: (INC-20260731-broad-process-pattern-kills: `earlyoom` died because its own
#: command line reads `--ignore ^(llama-server|sd-server)$`).
_FORBIDDEN_PROCESS_NAMES = frozenset({
    "pkill", "pgrep", "killall", "system", "popen", "spawnl", "spawnlp", "spawnv",
    "spawnvp", "getoutput", "getstatusoutput", "check_output", "call", "check_call",
})

_FORBIDDEN_IMPORTS = frozenset({"pty", "commands"})


def audit_process_discipline(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that it cannot make a name-pattern kill.

    Three things are checked, and each one is a rule this module could otherwise
    break silently:

    1. **No name-pattern process call.** `pkill`, `pgrep`, `killall`,
       `os.system`, `subprocess.call/check_output/...` — every one of them can
       either take a name or hand a command line to a shell.
    2. **No `shell=True`.** A shell turns argv into a string, and a string can
       carry a pipeline.
    3. **Signals go to a captured pid.** `os.kill` / `os.killpg` are permitted
       (they are how a launched child is stopped) and every call site is
       reported, so a reader can check by eye that the argument is a pid this
       module captured rather than one it discovered.

    COULD_NOT_CHECK when the source cannot be read: an unreadable module is not
    an audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module source: {exc}",))

    findings: list = []
    signal_sites: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in _FORBIDDEN_PROCESS_NAMES:
                findings.append(
                    f"line {node.lineno}: call to {name!r} — a name-pattern or shell-bearing "
                    "process call. Capture the pid you spawned and signal only that "
                    "(INC-20260731).")
            if name in ("kill", "killpg"):
                signal_sites.append(f"line {node.lineno}: {name}")
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) \
                        and keyword.value.value is True:
                    findings.append(
                        f"line {node.lineno}: shell=True. A shell renders argv as a string, "
                        "and a string can carry a pipeline; llama binaries must never be "
                        "piped (feedback_pipe_hazards).")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for module_name in names:
                if module_name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {module_name!r}")
    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS, (
        "no name-pattern process call, no shell=True, no pty/commands import",
        f"signal call sites (each must target a pid this module captured): {signal_sites}",
    ))
