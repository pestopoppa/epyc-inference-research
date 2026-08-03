#!/usr/bin/env python3
"""serving_runtime.py — the serving-runtime backend adapter (§11.6, §13.5).

WHY THIS MODULE EXISTS
----------------------
Scheduler, batching, admission and KV-policy work shares the AutoKernel research
loop with kernel work, and shares almost nothing else. Four properties differ,
and every one of them has already cost this project something when it was
elided:

  * **It must not travel the kernel-freeze path.** AK-D9/AK-D23: *"Scheduler wins
    are not kernel freezes."* A freeze crosses four human-only trust boundaries
    (`MEASUREMENT.md:140-142`) — the freeze itself, the era-registry row, the
    AutoPilot baseline apply, and the pinned human-only path list. A scheduler
    change owes none of them and is entitled to none of them. This adapter
    therefore **refuses** that path outright (`refuse_kernel_freeze`,
    `scan_for_kernel_freeze_actions`, `check_no_kernel_artifact_change`) rather
    than degrading to a "kernel freeze with the kernel bits skipped", which is
    how a scheduler change would acquire a kernel era row nobody could interpret
    (§11.4).

  * **The metric is `task_rate`, never tokens/s.** `MEASUREMENT.md:23-30` makes
    the two authoritative in their own scopes and forbids substituting one for
    the other, and §11.6 names the reason: a scheduler change is exactly the case
    where tokens are not commensurable across arms. Arm A may emit more tokens
    per second while completing fewer tasks. `ServingEvidence` cannot be
    constructed carrying a token-rate metric, and `check_metric_discipline()`
    routes the question through `schemas.check_metric_commensurability()` so
    there is one implementation of the rule, not two.

  * **The workload is variable-arrival replay, not fixed-shape benchmarking.**
    The P0.2 separation (§19.6): *"benchmark class is part of evaluator
    identity: fixed-shape feeds kernel campaigns, variable-request feeds
    `serving_runtime`."* Kernel and scheduler effects have historically been
    conflated here; a fixed-shape spec handed to this adapter is refused
    (`FixedShapeWorkloadRefused`), never re-labelled. Latency and SLO cells are
    REQUIRED outputs of a serving cell, not optional companions — a scheduler
    that raises throughput by starving the tail is a regression that a
    throughput-only record cannot express.

  * **Release is three gates, and none of them implies the next** (§11.6).
    Pipeline-green, the-stack-starts, and live-equals-config are distinct
    questions — this is `feedback_stack_change_three_gates`, learned the
    expensive way. The gate framework here refuses to satisfy one gate with
    another gate's evidence (`GateEvidenceMisuse`), refuses to record gate 3
    against the config file that was *supposed* to produce the live state
    (`feedback_verify_live_affinity_not_just_topology_hash`), and refuses to
    report an unattempted gate as a passed one.

WHAT THIS MODULE DOES NOT DO
----------------------------
It starts, stops and signals **no process**, and it starts **no stack**. Gate 1's
subprocess is a *seam* (`GuardRunner`) supplied by the caller; gates 2 and 3 are
modelled against observations supplied by the caller. That is not a promise in
prose: `audit_no_write_or_process_paths()` parses this module's own AST and FAILs
on a write-capable call, a process call, or an import of `os`/`subprocess`/
`signal`/`shutil`, and the test suite asserts it PASSes. Starting the stack is the
inference-owning session's action, executed by that session at a moment it
chooses (`OPERATING_CONSTRAINTS.md:41`); this module can only produce a
`ReloadRequest`, which is a record.

It also writes no file, runs no inference, and reads
`epyc-orchestrator/scripts/validate/stack_change_guard.py` as a **contract**, not
as a dependency: that repository is off limits for writes and this package must
not import from it (its `main()` inserts its own repo root on `sys.path` and
imports `src.registry`, which would give an AutoKernel process an orchestrator
import identity — the ambient-identity scar of §2.5 item 12).

WHICH SECTIONS THIS FILE IMPLEMENTS
-----------------------------------
§11.6 (the three-gate stack-change path, the three serving deltas, what it does
not touch), §13.5 (adapter responsibilities, the refusal), §1.5/AK-D9 (per-backend
campaigns, serving has no source tree and no frozen branch), §1.6 (per-phase
objective and the labelled-analysis escape for anything cross-scope), §9.7 (T2
composed-champion estimator — composition is measured, never added), §10.4
(operator waivers: PASS / FAIL / PASS_WITH_WAIVER, hash-pinned, merits never
judged), §10.5 (the incumbent is archived, not merely rebuildable — for serving
the incumbent is a *rendered configuration*), §10.6 (diff-complexity ceiling and
the `REQUIRES_HUMAN_CODE_REVIEW` marker), §11.3 (a cutover/reload is a routed
request, never an action), §12 (failure and abuse model), §4 invariants 5, 9, 10,
15, 18, and AK-D36/AK-D37 (the whole-stack llama.cpp-vs-vLLM ratio is not an
objective; every batch regime remains a legitimate direction).
"""
from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

from .. import schemas

__all__ = [
    # identity
    "BACKEND", "ADAPTER_ID", "RELEASE_PATH", "KERNEL_RELEASE_PATH", "RESOURCE_LANE",
    "METRIC", "METRIC_DIRECTION", "BENCHMARK_CLASS", "FIXED_SHAPE_BENCHMARK_CLASS",
    "WORKLOAD_CLASS", "SCHEMA_STACK_CHANGE_PACKAGE", "STACK_CHANGE_VERDICTS",
    "GUARD_REPO_ROOT", "GUARD_SCRIPT", "GUARD_SCOPE_WEAKENING_FLAGS",
    "LINKAGE_VERIFIER", "TREE_ROOTS",
    "NO_GGML_TREE", "SERVICE_TREES",
    "KERNEL_PROTOCOL_IDS", "RELOAD_GATE_BASIS", "AK_D36_NOTE", "AK_D37_NOTE",
    # errors
    "ServingAdapterError", "KernelFreezePathRefused", "KernelChangeMisrouted",
    "MetricSubstitutionRefused", "FixedShapeWorkloadRefused", "GateEvidenceMisuse",
    "GateOrderViolation", "GateVerdictTampering", "AdditiveCompositionRefused",
    "CrossEngineRatioObjectiveRefused", "ReloadGateBasisRefused",
    "GuardRunnerNotWired", "PackageTampering",
    # kernel-freeze refusal
    "FORBIDDEN_PRODUCTION_ACTIONS", "refuse_kernel_freeze", "release_path_for",
    "scan_for_kernel_freeze_actions", "check_no_kernel_artifact_change",
    "STABLE_KERNEL_PATH_SEGMENT", "classify_stable_kernel_path_use",
    # gate framework
    "GATE_1", "GATE_2", "GATE_3", "GATE_ORDER", "GATE_EVIDENCE_KINDS",
    "GateSubjectUnbound",
    "REFUSED_EVIDENCE_KINDS", "LIVE_OBSERVATION_SOURCES", "GateOutcome",
    "ThreeGateResult", "evaluate_three_gates",
    # gate 1
    "GuardInvocation", "GuardResult", "GuardRunner", "build_guard_invocation",
    "run_guard", "parse_guard_result", "gate_pipeline_green",
    # gate 2
    "LinkageReceipt", "ServiceStartObservation", "StackStartObserver",
    "gate_stack_starts",
    # gate 3
    "IntendedProcessConfig", "LiveProcessFact", "LiveProcessObserver",
    "gate_live_equals_config",
    # evidence
    "ArrivalTrace", "VariableArrivalReplaySpec", "TaskRateCell", "LatencyCell",
    "SloCell", "ServingEvidence", "PinnedProductionConfig", "check_metric_discipline",
    "check_comparison_anchor", "check_regime_admissible", "PINNED_CONFIG_MOVED",
    # objective
    "admit_objective", "admit_change_class", "CrossEngineAnalysisView",
    # T2
    "ComposedServingEstimate",
    # release
    "ComplexityCeiling", "SERVING_COMPLEXITY_CEILING", "BlastRadius",
    "classify_blast_radius", "RollbackPlan", "StackChangePackage",
    "assemble_stack_change_package", "ReloadRequest", "build_reload_request",
    # events
    "build_serving_evaluation_event",
    # audit
    "audit_no_write_or_process_paths",
]


# =============================================================================
# Identity — every constant here is a decision, not a default
# =============================================================================

BACKEND = "serving_runtime"

#: Versioned, because a record naming a mutable adapter id cannot be replayed.
ADAPTER_ID = "autokernel.adapters.serving_runtime/v1"

#: §11.6. The two release paths are named so that "which path is this on" is a
#: value a record carries, never a thing a reader infers from context.
RELEASE_PATH = "stack_change"
KERNEL_RELEASE_PATH = "kernel_freeze"

#: `schemas.RESOURCE_LANES` — the serving lane. It is deliberately neither `cpu`
#: nor `gpu`: a variable-arrival replay occupies the whole serving stack, and
#: borrowing a kernel lane's claim would misdescribe what is excluded.
RESOURCE_LANE = "stack"

#: `MEASUREMENT.md:23-30` and §11.6. Substitution is forbidden in both
#: directions, which is why this pair is stated once and consumed everywhere.
METRIC = "task_rate"
METRIC_DIRECTION = "higher_better"

#: §19.6 P0.2 — benchmark class is part of evaluator identity.
BENCHMARK_CLASS = "variable_request"
FIXED_SHAPE_BENCHMARK_CLASS = "fixed_shape"
WORKLOAD_CLASS = "variable_arrival_replay"

SCHEMA_STACK_CHANGE_PACKAGE = "epyc.autokernel.stack_change_package.v1"

#: §10.4 generalised to this path. A binary PASS/FAIL gate would have blocked v8.
STACK_CHANGE_VERDICTS = frozenset({"PASS", "FAIL", "PASS_WITH_WAIVER"})

#: Gate 1's instrument. READ-ONLY, and never imported: `epyc-orchestrator` is a
#: different repository and a different trust boundary, and its `main()` puts its
#: own repo root on `sys.path`.
GUARD_REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
GUARD_SCRIPT = GUARD_REPO_ROOT / "scripts" / "validate" / "stack_change_guard.py"

#: §10.2: the linkage verifier lives in **epyc-inference-research**, not
#: epyc-root — CLAUDE.md cites it unqualified, which is the same defect class as
#: the durability validator's path in `MEASUREMENT.md:155`. Naming it absolutely
#: here is the fix.
LINKAGE_VERIFIER = "/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"

#: Guard flags that narrow WHAT the guard checks while leaving its output shape
#: intact — the run still prints `OK:` and still exits 0, so nothing downstream
#: can tell that a smaller question was answered. `--strict` is checked
#: separately (it is absent rather than present); these are its mirror image, and
#: without them gate 1 is passable by DELETING the thing it inspects rather than
#: by satisfying it. Gate 1 records a FAIL rather than refusing the invocation:
#: the run happened, and the record of a run that could not answer the gate is
#: itself evidence (§3.2).
GUARD_SCOPE_WEAKENING_FLAGS = {
    "--skip-hardcoded-surface-scan": (
        "the hardcoded consumer-surface scan is the staleness gate 1 exists to "
        "catch, so with it skipped `OK:` means 'nothing was looked at'"
    ),
    "--allow-production-blocker-waivers": (
        "it downgrades a production-blocker hardcoded surface from a fail-closed "
        "error to a visible warning, which a pre-declared acceptance then absorbs "
        "— a production blocker laundered into a green gate in two steps"
    ),
}

#: The three source trees (§1.5). A serving change owns none of them — they
#: appear here only so gate 2 can prove a service's `LD_LIBRARY_PATH` resolves
#: into ITS OWN tree. The three trees run three ggml generations, so a binary
#: that inherits another tree's ggml runs silently wrong (CLAUDE.md, 2026-07-31
#: speech-kernel freeze).
TREE_ROOTS = {tree: f"/mnt/raid0/llm/{tree}" for tree in sorted(schemas.SOURCE_TREES)}

#: The directory the three trees live in. Named because its OTHER children are
#: the hazard: `llama.cpp-experimental`, `llama.cpp-v5`, `llama.cpp-mi210-hip` and
#: a dozen more sibling worktrees sit beside `llama.cpp`, each with its own ggml
#: build, and each is a string-prefix of a declared root.
_TREE_PARENT = "/mnt/raid0/llm"

#: Not every affected service links a ggml tree — the orchestrator API is a
#: uvicorn process. Such a service DECLARES that it links none, positively, so
#: "no linkage receipt" cannot mean either "it needs none" or "somebody forgot".
#: Those two states are indistinguishable if absence is the only signal, and
#: which one it is decides whether gate 2 passes.
NO_GGML_TREE = "none"
SERVICE_TREES = frozenset(TREE_ROOTS) | {NO_GGML_TREE}

#: Protocols that govern tokens/s kernel cells. A serving cell that cites one is
#: not a stricter serving cell, it is a category error: those protocols' decision
#: rules are written over a metric this backend may not report.
KERNEL_PROTOCOL_IDS = frozenset({
    "P-BENCH-1", "P-BENCH-PREFILL-1", "P-BENCH-4", "P-GPU-1", "P-SHED-1",
    "P-KERNEL-FREEZE-1",
})

#: `feedback_stack_reload_checks_cpu_bench_not_autopilot`: a reload gates on the
#: pinned bench. AutoPilot being down is NOT the gate — it is a fact about a
#: consumer, and it has been mistaken for the gate before.
RELOAD_GATE_BASIS = "pinned_bench"

AK_D36_NOTE = (
    "AK-D36: the whole-stack llama.cpp-vs-vLLM ratio MUST NOT become an objective. "
    "The headline 24-44x arrives at 16-64 concurrent users and is continuous "
    "batching, PagedAttention and the scheduler; at batch-1 the kernel delta on "
    "our MI210 is +11%. A cross-engine whole-stack ratio recruited as a target "
    "spends a campaign on a property the campaign is not measuring."
)
AK_D37_NOTE = (
    "AK-D37: AK-D36 excludes a TARGET, not a REGIME. Single-stream AND batched "
    "prefill and decode all remain legitimate directions; this adapter never "
    "rejects a direction for its concurrency level. Read as 'batch-1 only', "
    "AK-D36 would retire the highest-confidence GPU band available."
)


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class ServingAdapterError(Exception):
    """Base class for every refusal this adapter makes."""


class KernelFreezePathRefused(ServingAdapterError):
    """The kernel-freeze path was requested of the serving adapter.

    §13.5: *"its release path is the three-gate stack-change path (§11.6) … not
    the kernel-freeze path, which the adapter MUST refuse rather than degrade
    to."* Degrading would mean assembling a freeze-shaped package with the kernel
    parts empty, and that package's era row is the artifact nobody could later
    interpret (§11.4, `MEASUREMENT.md:233`).
    """


class KernelChangeMisrouted(ServingAdapterError):
    """A change that alters a kernel artifact was routed through the stack path.

    The refusal is symmetric and both halves matter. Sending a scheduler change
    down the freeze path invents an era; sending a *kernel* change down the stack
    path ships a new binary into production with no freeze, no era row, no
    rollback anchor and no T3 — which is the more dangerous of the two.
    """


class MetricSubstitutionRefused(ServingAdapterError):
    """tokens/s was offered where `task_rate` is authoritative, or vice versa."""


class FixedShapeWorkloadRefused(ServingAdapterError):
    """A fixed-shape benchmark was offered as serving evidence (§19.6 P0.2)."""


class GateEvidenceMisuse(ServingAdapterError):
    """A gate was answered with evidence that cannot answer it.

    §11.6: the three gates *"are distinct and none implies the next"*. The
    concrete failures this catches are a guard result offered as proof the stack
    started, and a config file offered as proof of what is live.
    """


class GateOrderViolation(ServingAdapterError):
    """The three gates were assembled out of order, or past a non-PASS gate."""


class GateVerdictTampering(ServingAdapterError):
    """A `ThreeGateResult` carries a status that does not follow from its gates."""


class GateSubjectUnbound(ServingAdapterError):
    """Gate 3's outcome is not bound to the processes gate 2's outcome is about.

    `gate_live_equals_config()` ties each live fact to the gate-2 observation of
    the same process. That tie lived entirely INSIDE the function, so it was
    obtainable by not calling it: a hand-built gate-3 `GateOutcome` — which is
    what a caller assembling a verdict from journalled outcomes holds — carried
    no tie at all, and `evaluate_three_gates()` derived `released == True` from
    it. A guarantee a verdict can be assembled without is not a guarantee
    (`feedback_verify_integrity_not_presence_of_own_edit`), so the release verdict
    itself now requires the binding, and requires it to name the same processes
    the gate-2 outcome beside it names.
    """


class AdditiveCompositionRefused(ServingAdapterError):
    """A composed estimate was requested by adding local percentages (§9.7)."""


class CrossEngineRatioObjectiveRefused(ServingAdapterError):
    """A cross-engine whole-stack ratio was offered as a campaign objective."""


class ReloadGateBasisRefused(ServingAdapterError):
    """A reload was justified by something other than the pinned bench."""


class GuardRunnerNotWired(ServingAdapterError):
    """`run_guard()` was called without a runner.

    There is deliberately no default runner. A default would either spawn a
    process from this module — which the AST audit forbids — or report an unrun
    guard as having produced no errors, which is a fail-open PASS.
    """


class PackageTampering(ServingAdapterError):
    """A stack-change package carries a verdict its own contents do not support."""


# =============================================================================
# Small shared helpers
# =============================================================================

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a non-empty string, got {value!r}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    _require_str(value, field)
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field}: {value!r} is not a sha256 hex digest")
    if schemas.is_placeholder_digest(value):
        raise ValueError(
            f"{field}: {value!r} is a placeholder digest, not a measured identity"
        )
    return value


def _require_tuple(value: Any, field: str, *, item_type=str, non_empty: bool = False):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field}: expected a sequence, got {type(value).__name__}")
    items = tuple(value)
    for i, item in enumerate(items):
        if not isinstance(item, item_type):
            raise ValueError(
                f"{field}[{i}]: expected {item_type.__name__}, got {type(item).__name__}"
            )
    if non_empty and not items:
        raise ValueError(f"{field}: must not be empty")
    return items


def _parse_instant(value: Any, field: str) -> Optional[datetime]:
    """Return an aware datetime, or None when the value cannot be read.

    None is the third outcome: an unparseable timestamp is not an ordering, and
    a gate that cannot order its observations reports COULD_NOT_CHECK rather
    than assuming they were sequential.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # A naive timestamp on a shared host is ambiguous across sessions, and
        # sequencing is the whole point of gate 2.
        return None
    return parsed


def _protocol_family(protocol_id: str) -> str:
    return protocol_id.split("/", 1)[0].strip()


# =============================================================================
# The refusal (§13.5, §11.6 "what it does not touch", cardinal rule)
# =============================================================================

#: Every action that belongs to a human, restated as an action id so a plan can
#: be scanned for it structurally. `MEASUREMENT.md:140-142` is the source list;
#: the branch/symlink/build entries come from §1.3 item 4 and invariant 3.
FORBIDDEN_PRODUCTION_ACTIONS = frozenset({
    "freeze",
    "cutover",
    "write_production_branch",
    "commit_to_production_branch",
    "build_in_production_tree",
    "move_stable_kernel_symlink",
    "write_era_registry_row",
    "apply_autopilot_baseline",
    "amend_human_only_paths",
    "seal_kernel_release_candidate",
    "write_registry_row",
    "write_lineup_row",
})

#: Textual patterns that betray a forbidden action inside a command string. The
#: scan is deliberately conservative — a false positive costs a human one look,
#: a false negative costs a production write nobody authorised. Each pattern
#: carries the reason it exists so a hit explains itself.
_FORBIDDEN_COMMAND_PATTERNS = (
    (re.compile(r"production-(consolidated|speech)-v\d+"),
     "names a frozen production kernel branch (§1.3 item 4; human_only_paths.yaml:42-49)"),
    (re.compile(r"instrument_eras\.ya?ml"),
     "writes the era registry (MEASUREMENT.md:140-142, human-only)"),
    (re.compile(r"autopilot_baseline\.ya?ml"),
     "applies an AutoPilot baseline (MEASUREMENT.md:140-142, human-only)"),
    (re.compile(r"human_only_paths"),
     "amends the pinned human-only path list (§1.3 item 4)"),
    (re.compile(r"freeze_v\d+_production"),
     "runs a production freeze script (§1.3, invariant 5)"),
    (re.compile(r"\bgit\s+(commit|push|tag|branch|checkout|switch)\b[^\n]*\bproduction-"),
     "operates git on a production-named branch (invariant 3)"),
    (re.compile(r"\breboot\b"),
     "reboots the host (MEASUREMENT.md:140-142, operator authority)"),
)

#: The stable production kernel path, as its own vocabulary rather than as one
#: more substring in `_FORBIDDEN_COMMAND_PATTERNS`.
#:
#: `/mnt/raid0/llm/kernels/production/<lane>` is a SYMLINK an operator repoints at
#: freeze (invariant 3). It is also the path every serving launcher legitimately
#: EXECUTES A BINARY THROUGH — that is the whole point of a stable path. A
#: substring match cannot tell the two apart, so it refused both, and a §11.6
#: package could not state its own launch command: the guard forbade its own
#: idiom (`feedback_guard_must_not_forbid_its_own_idiom`).
#:
#: The distinction this classifier draws is the shell's own, and it is drawn from
#: two facts about each occurrence, never from a verb blocklist (a blocklist of
#: mutating verbs fails OPEN on the verb nobody listed):
#:
#:   1. **Depth.** `…/kernels/production` and `…/kernels/production/<lane>` ADDRESS
#:      the trust boundary itself — the directory and the symlink. Everything at
#:      those depths is refused in every position, because repointing, removing
#:      and replacing the link all name it and nothing legitimate does.
#:      `…/kernels/production/<lane>/<something>/…` addresses content THROUGH the
#:      link, which is what execution does.
#:   2. **Position.** In shell grammar the command word is the program that runs;
#:      everything after it is an operand that the program acts ON. A below-link
#:      path in command-word position is an execution. The same path as an operand
#:      of some other program (`cp … <path>`, `rm <path>`, `install … <path>`) is
#:      that program acting on production, and is refused without asking which
#:      program it was.
#:
#: Three narrow, named allowances complete "executes a binary under the path":
#: a transparent launcher prefix (`taskset`/`numactl`/`env`/…) is stepped over to
#: the program it launches; a loader-search-path assignment in the segment's
#: assignment PREFIX (`LD_LIBRARY_PATH=…`, which CLAUDE.md's speech-kernel freeze
#: requires every launcher to set) is a read; and once the command word IS a
#: production binary, its own below-link operands are that binary's own tree.
#: Assignment prefixes are distinguished from `cmd NAME=value` operands by
#: position, exactly as the shell does — `make install DESTDIR=…` is an operand
#: and is refused.
#:
#: The residual this leaves, stated rather than implied: a segment whose command
#: word is a production binary may pass below-link operands, so a production
#: binary that itself writes into its own tree is not caught here. Nothing
#: text-level can catch that, and the link itself is still refused in that segment
#: — what is NOT relied on is any claim that the residual is empty.
STABLE_KERNEL_PATH_SEGMENT = "kernels/production"

#: Characters a plain path token may contain. Anything else — a substitution, a
#: glob, a quote, a metacharacter — makes the token undecidable, and an
#: undecidable token holding a production path is refused rather than parsed.
_PLAIN_PATH_RE = re.compile(r"^[\w./+@%~-]+$")

#: Shell control operators that end one command segment and begin the next.
_SEGMENT_SPLIT_RE = re.compile(r"\|\||&&|[;|&\n]")

#: A redirection operator, optionally file-descriptor-qualified. A redirect TARGET
#: is a write by definition, so a production path in one is refused in every case
#: — including inside a segment whose command word is a production binary.
_REDIRECT_RE = re.compile(r"^[0-9]*(?:>>|>&|&>|>|<<|<&|<)")

#: `NAME=value`. The name grammar is the shell's, so `--prefix=…`, `core.x=…` and
#: `-o key=value` are operands rather than assignments.
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

#: The only assignment names whose value may name the production tree. All three
#: are dynamic-loader search paths: they are READ by `ld.so`, and no write can
#: hide in one. Every other name — `DESTDIR`, `PREFIX`, `OUT` — is refused, which
#: is what keeps `DESTDIR=…/kernels/production/cpu/bin make install` out.
_READ_ONLY_PATH_ASSIGNMENTS = frozenset({
    "LD_LIBRARY_PATH", "LD_RUN_PATH", "PATH",
})

#: Programs that launch another program rather than acting on their operands.
#: Stepping over one reaches the program it launches; it never grants anything.
_TRANSPARENT_LAUNCHERS = frozenset({
    "env", "exec", "taskset", "numactl", "nohup", "setsid", "stdbuf", "nice",
    "ionice", "chrt", "time",
})

#: The ONLY non-option token shape stepped over while walking a launcher's own
#: arguments: a CPU/node list or `all`. Any other bare word ends the walk and
#: becomes the command word, so `taskset -c 0-95 rm …/kernels/production/cpu/bin/x`
#: reports `rm` — a launcher prefix may not be used to smuggle a mutating verb
#: into command position.
_LAUNCHER_SCALAR_RE = re.compile(r"^(?:[0-9][0-9,+-]*|all)$")


def _unquote(token: str) -> str:
    """Strip ONE matching pair of surrounding shell quotes. Nothing else.

    A quoted launch command is still a launch command. Stripping grants nothing:
    the unquoted token is depth- and position-analysed exactly as a bare one, so
    `ln -sfn X "…/kernels/production/cpu"` is still an operand at link depth.
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _plain_path_parts(token: str) -> tuple:
    """`(parts, decidable)` — the path-like pieces of one shell token.

    A `NAME=a:b` assignment and a bare `a:b` search path are both colon-separated
    lists of paths, so they are split before depth is measured. `decidable` is
    False when any piece is not a plain path, which is refused rather than read.
    """
    body = _unquote(token)
    match = _ASSIGNMENT_RE.match(body)
    if match is not None:
        body = match.group(2)
    parts = [part for part in body.split(":") if part]
    decidable = all(_PLAIN_PATH_RE.match(part) for part in parts)
    return tuple(parts), decidable


#: A run of two or more `/`. `a//b` and `a/b` are THE SAME PATH to every kernel
#: on this host, so a segment search that does not collapse them is passed by
#: typing one extra slash: `rm -rf /mnt/raid0/llm/kernels//production/cpu/bin/x`
#: names no `kernels/production` substring and produced no finding at all.
_SLASH_RUN_RE = re.compile(r"/{2,}")


def _collapse_slashes(text: str) -> str:
    """`a//b` -> `a/b`. Used for SEARCHING only; messages quote the raw token."""
    return _SLASH_RUN_RE.sub("/", text)


def _names_stable_path(text: str) -> bool:
    """True when `text` names the stable path under any slash spelling."""
    return STABLE_KERNEL_PATH_SEGMENT in _collapse_slashes(text)


def _stable_path_components(path: str):
    """Components addressed BELOW `kernels/production`, or None for no occurrence.

    `…/kernels/production` -> `()`; `…/kernels/production/cpu` -> `('cpu',)` — the
    lane symlink itself; `…/kernels/production/cpu/bin/llama-server` -> three, which
    is content reached THROUGH the link. Slash runs are collapsed first, because
    `kernels//production` is the same directory and a depth this function cannot
    see is a depth it reports as zero occurrences.
    """
    collapsed = _collapse_slashes(path)
    index = collapsed.find(STABLE_KERNEL_PATH_SEGMENT)
    if index < 0:
        return None
    tail = collapsed[index + len(STABLE_KERNEL_PATH_SEGMENT):]
    return tuple(part for part in tail.split("/") if part)


def _traverses_out_of_stable_path(path: str) -> bool:
    """True when a `..` component appears below `kernels/production`.

    Depth is the whole of the execute-vs-mutate distinction, and `..` makes the
    counted depth mean nothing: `…/kernels/production/cpu/../../../../usr/bin/
    install` counts SEVEN components — "well below the link" — while naming
    `/usr/bin/install`, a program that is not under the stable path at all. In
    command-word position that admitted an arbitrary host binary as "an execution
    through the link", and admitting it also flipped `launches_production`, which
    then admitted the segment's below-link OPERANDS. The pair
    `…/production/cpu/../../../../usr/bin/install -m755 evil
    …/production/cpu/bin/llama-server` classified PASS: a straight overwrite of
    the production serving binary.

    A path this function cannot measure is refused, not resolved: resolving it
    would mean following the very symlink whose target is the trust boundary.
    """
    components = _stable_path_components(path)
    return components is not None and ".." in components


def _is_below_stable_link(path: str) -> bool:
    components = _stable_path_components(path)
    if components is None or ".." in components:
        return False
    return len(components) >= 2


def _executes_under_stable_path(token: str) -> bool:
    """True when `token` is a plain path naming a file BELOW the lane symlink."""
    body = _unquote(token)
    if not _PLAIN_PATH_RE.match(body):
        return False
    return _is_below_stable_link(body)


def _segment_positions(tokens: Sequence) -> tuple:
    """`(command_word_index, assignment_prefix_indices, redirect_target_indices)`.

    The walk is fail-closed at every step: an unrecognised token ENDS it and
    becomes the command word, so nothing is stepped over by accident.
    """
    assignments: set = set()
    redirects: set = set()
    i = 0
    total = len(tokens)
    while i < total:
        token = tokens[i]
        if _REDIRECT_RE.match(token):
            redirects.add(i)
            # A bare operator takes the next token as its target; an attached one
            # (`>file`) carries its own.
            if _REDIRECT_RE.fullmatch(token) and i + 1 < total:
                redirects.add(i + 1)
                i += 1
            i += 1
            continue
        break
    command_index = None
    while i < total:
        token = tokens[i]
        if _ASSIGNMENT_RE.match(token):
            assignments.add(i)
            i += 1
            continue
        if token in _TRANSPARENT_LAUNCHERS:
            i += 1
            while i < total and (tokens[i].startswith("-")
                                 or _LAUNCHER_SCALAR_RE.match(tokens[i])):
                i += 1
            continue
        command_index = i
        break
    # Redirections may appear anywhere after the command word too.
    j = 0
    while j < total:
        token = tokens[j]
        if _REDIRECT_RE.match(token):
            redirects.add(j)
            if _REDIRECT_RE.fullmatch(token) and j + 1 < total:
                redirects.add(j + 1)
                j += 1
        j += 1
    return command_index, assignments, redirects


def _stable_kernel_path_findings(text: str) -> list:
    """Every use of the stable production kernel path in `text` that MUTATES it.

    Executions are not findings; everything else that names the path is, including
    every token the parser could not decide.
    """
    findings: list = []
    for segment in _SEGMENT_SPLIT_RE.split(text):
        tokens = segment.split()
        if not tokens:
            continue
        command_index, assignments, redirects = _segment_positions(tokens)
        command_word = "" if command_index is None else tokens[command_index]
        launches_production = _executes_under_stable_path(command_word)
        for i, token in enumerate(tokens):
            parts, decidable = _plain_path_parts(token)
            occurrences = [part for part in parts
                           if _stable_path_components(part) is not None]
            if not occurrences and not _names_stable_path(token):
                continue
            if not decidable or (not occurrences and _names_stable_path(token)):
                findings.append(
                    f"token {token!r} names {STABLE_KERNEL_PATH_SEGMENT!r} but is not "
                    f"a plain path token, so whether it executes a binary under the "
                    f"stable production kernel path or mutates it cannot be decided "
                    f"— an undecidable use of a trust boundary is refused "
                    f"(invariant 3)"
                )
                continue
            for part in occurrences:
                if _traverses_out_of_stable_path(part):
                    findings.append(
                        f"token {token!r} walks back out of the stable production "
                        f"kernel path with a '..' component ({part!r}); depth is what "
                        f"separates executing a binary THROUGH the link from acting "
                        f"ON it, and a path that traverses upward has no depth this "
                        f"scan can measure without resolving the very symlink that "
                        f"is the trust boundary (invariant 3)"
                    )
                    continue
                if not _is_below_stable_link(part):
                    findings.append(
                        f"token {token!r} addresses the stable production kernel path "
                        f"{part!r} ITSELF — that path is the symlink an operator "
                        f"repoints at freeze, never a binary anything executes "
                        f"(invariant 3, §1.3 item 4)"
                    )
                    continue
                if i in redirects:
                    findings.append(
                        f"token {token!r} is a redirection target under the stable "
                        f"production kernel path; a redirect writes it (invariant 3)"
                    )
                    continue
                if i == command_index:
                    continue
                match = _ASSIGNMENT_RE.match(_unquote(token))
                if (i in assignments and match is not None
                        and match.group(1) in _READ_ONLY_PATH_ASSIGNMENTS):
                    continue
                if launches_production:
                    continue
                acting = command_word or "<no command word>"
                findings.append(
                    f"token {token!r} passes a path under the stable production "
                    f"kernel path to {acting!r} as an operand; the command word is "
                    f"what executes, and an operand is what a command ACTS ON "
                    f"(invariant 3)"
                )
    return findings


def classify_stable_kernel_path_use(command: Any) -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK for one command's use of the stable path.

      * FAIL — at least one occurrence MUTATES the path (or could not be decided).
      * PASS — the command names the stable production kernel path and every
        occurrence EXECUTES a binary through it.
      * COULD_NOT_CHECK — the command does not name the path at all, so this
        classifier decided nothing about it. PASS is deliberately unreachable
        there: a classifier that certifies a string for not containing the thing
        it classifies is passed by deleting the thing it inspects, and the
        §11.6 package would then carry a green receipt for a command that never
        mentioned production.
    """
    if not isinstance(command, str):
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"expected a command string, got {type(command).__name__}; a "
             f"non-string was not classified",),
        )
    if not _names_stable_path(command):
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"the command does not name {STABLE_KERNEL_PATH_SEGMENT!r}, so there "
             f"was no use of the stable production kernel path to classify",),
        )
    findings = _stable_kernel_path_findings(command)
    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS)


_ACTION_KEYS = frozenset({"action", "actions", "kind", "op", "operation", "verb"})


def release_path_for(backend: str) -> str:
    """The release path a backend travels. Refuses to answer for other backends.

    This adapter answers for `serving_runtime` only. Returning `kernel_freeze`
    here for a llama backend would make this module look like a router with an
    opinion about paths it does not own; the kernel adapters own their own.
    """
    if backend != BACKEND:
        raise ServingAdapterError(
            f"release_path_for: this adapter answers for {BACKEND!r} only; "
            f"{backend!r} is another adapter's backend (§13.1-§13.4)"
        )
    return RELEASE_PATH


def refuse_kernel_freeze(request: str, *, detail: str = "") -> None:
    """Always raises. There is no argument that makes this path available.

    Callable so that every place the kernel-freeze path could be *reached* from
    serving work raises the same, greppable, self-explaining error — rather than
    each caller inventing its own quiet `return None`.
    """
    suffix = f" — {detail}" if detail else ""
    raise KernelFreezePathRefused(
        f"{BACKEND} cannot travel the {KERNEL_RELEASE_PATH} path; {request!r} was "
        f"requested{suffix}. A serving change creates no kernel version, no frozen "
        f"branch and no kernel-speed era row (§11.6). Its path is "
        f"{RELEASE_PATH!r}: the three gates of §11.6. Refusing is the behaviour "
        f"§13.5 requires — degrading to a freeze-shaped package with empty kernel "
        f"fields is the failure this refusal exists to prevent."
    )


def scan_for_kernel_freeze_actions(
    plan: Any, path: str = "$", *, match_command_strings: bool = True,
) -> schemas.Check:
    """FAIL if anything in `plan` would perform a human-only production write.

    Walks the whole structure: action ids are matched against
    `FORBIDDEN_PRODUCTION_ACTIONS`, and — when `match_command_strings` is on —
    every string is matched against the command patterns. Returns a `Check` so a
    caller that wants to journal the finding can; `assemble_stack_change_package()`
    turns a FAIL into a refusal, because a package is a thing a human executes and
    a hit means the package would execute a write nobody authorised.

    `match_command_strings=False` exists for one specific, load-bearing case and
    is not a convenience: **diagnostic prose is not an action.** A failing gate 3
    reports *"service 'worker' runs /mnt/raid0/llm/kernels/production/cpu/…"* —
    that string DESCRIBES production, it does not write it. Pattern-matching it
    would make the adapter refuse to *report a failure*, and a release gate that
    cannot express its own FAIL is worse than one that over-matches. So the
    package scans its command surface for command strings, and its whole body for
    action ids, which is where an action can actually be declared.

    COULD_NOT_CHECK is not reachable here on purpose: an empty or unreadable plan
    contains no forbidden action, and saying so is a fact, not an inference.
    """
    findings: list = []
    _scan_node(plan, path, findings, match_command_strings)
    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS)


def _declared_action_strings(value: Any, path: str):
    """Yield `(path, action)` for every forbidden action id declared at `value`.

    The declared value is WALKED, not merely tested when it happens to be a bare
    string. `_ACTION_KEYS` contains `actions` precisely because a plan's natural
    shape is a LIST — `{"actions": ["freeze", "write_era_registry_row"]}` — and a
    scalar-only test reads that as "no forbidden action declared", which is the
    worst answer this scan can give.

    Matching is case- and whitespace-insensitive: `"Freeze"` is the same
    declaration as `"freeze"`, and a production-write scan a caller can pass by
    changing the case of one letter is decoration rather than a check.
    """
    if isinstance(value, str):
        if value.strip().lower() in FORBIDDEN_PRODUCTION_ACTIONS:
            yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _declared_action_strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _declared_action_strings(item, f"{path}[{i}]")


def _scan_node(node: Any, path: str, findings: list, match_strings: bool) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}"
            if isinstance(key, str) and key.lower() in _ACTION_KEYS:
                for where, declared in _declared_action_strings(value, child):
                    findings.append(
                        f"{where}: declares the human-only action {declared!r} "
                        f"(MEASUREMENT.md:140-142); AutoKernel drafts it, a human "
                        f"executes it"
                    )
            _scan_node(value, child, findings, match_strings)
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _scan_node(value, f"{path}[{i}]", findings, match_strings)
    elif isinstance(node, str) and match_strings:
        for pattern, reason in _FORBIDDEN_COMMAND_PATTERNS:
            if pattern.search(node):
                findings.append(f"{path}: {reason} — matched {pattern.pattern!r}")
        # The stable kernel path is classified, not matched: see
        # `STABLE_KERNEL_PATH_SEGMENT`. It runs only where the substring patterns
        # run, so diagnostic prose (scanned with match_command_strings=False) is
        # untouched by it in either direction.
        for finding in _stable_kernel_path_findings(node):
            findings.append(f"{path}: {finding}")


def check_no_kernel_artifact_change(
    *, pinned: "PinnedProductionConfig", candidate_binary_sha256: Any,
    candidate_linkage_sha256: Any,
) -> schemas.Check:
    """FAIL when the candidate changes the kernel binary or its linkage.

    §11.6 *"what it does not touch"*: no new kernel version, no frozen branch, no
    era row for kernel speed. The stack-change path has no T3, no sealed
    candidate, no archived incumbent binary and no rollback kernel — so a change
    that alters the kernel artifact and travels this path ships an unfrozen
    binary into production. `assemble_stack_change_package()` raises
    `KernelChangeMisrouted` on a FAIL here rather than recording it.

    COULD_NOT_CHECK when either digest is absent: an unknown binary identity is
    not an unchanged one.
    """
    reasons: list = []
    for label, got, want in (
        ("binary_sha256", candidate_binary_sha256, pinned.kernel_binary_sha256),
        ("linkage_sha256", candidate_linkage_sha256, pinned.kernel_linkage_sha256),
    ):
        if not isinstance(got, str) or not _SHA256_RE.match(got):
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"candidate {label} is absent or not a sha256; an unknown kernel "
                 f"identity is not an unchanged one",),
            )
        if got != want:
            reasons.append(
                f"candidate {label} {got[:12]} differs from the pinned production "
                f"kernel's {want[:12]} — a kernel change may not travel the "
                f"{RELEASE_PATH} path (§11.6)"
            )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


# =============================================================================
# The gate framework (§11.6) — three gates, none implying the next
# =============================================================================

GATE_1 = "pipeline_green"
GATE_2 = "stack_starts"
GATE_3 = "live_equals_config"
GATE_ORDER = (GATE_1, GATE_2, GATE_3)

#: What each gate may be answered WITH. The mapping is the enforcement point for
#: *"distinct and none implies the next"*: a gate cannot borrow another gate's
#: evidence kind, so a green guard can never stand in for a started stack.
GATE_EVIDENCE_KINDS = {
    GATE_1: frozenset({"stack_change_guard_result"}),
    GATE_2: frozenset({"service_start_observation"}),
    GATE_3: frozenset({"live_process_observation"}),
}

#: Evidence kinds that are refused with a *named* reason rather than a generic
#: "not allowed", because each one is a mistake somebody has actually made.
REFUSED_EVIDENCE_KINDS = {
    "config_file": (
        "the config file is what was SUPPOSED to produce the live state; gate 3 "
        "verifies against live state (§11.6, feedback_verify_live_affinity_not_"
        "just_topology_hash)"
    ),
    "topology_hash": (
        "a topology hash describes the machine, not the process; live affinity is "
        "verified per process (feedback_verify_live_affinity_not_just_topology_hash)"
    ),
    "registry": (
        "the registry is a source artifact — gate 1's subject, not evidence that "
        "anything is running"
    ),
    "intended_config": (
        "the intended configuration is the question, not the answer"
    ),
    "assumed": (
        "an assumption is not evidence; an unobserved gate reports COULD_NOT_CHECK"
    ),
    "autopilot_state": (
        "autopilot being up or down is not a gate "
        "(feedback_stack_reload_checks_cpu_bench_not_autopilot)"
    ),
}

#: Sources that genuinely observe a running process. Anything else is refused by
#: `LiveProcessFact` — including the config file that was supposed to produce it.
LIVE_OBSERVATION_SOURCES = frozenset({
    "proc_status", "proc_exe", "proc_cmdline", "sched_getaffinity", "taskset_live",
    "cgroup_procs",
})


@dataclass(frozen=True)
class GateOutcome:
    """One gate's verdict, bound to the evidence kind that can answer it.

    `status` uses `schemas.PASS/FAIL/COULD_NOT_CHECK`: an unobservable gate is a
    third outcome, never a soft pass.
    """

    gate: str
    status: str
    evidence_kind: str
    evidence_ref: str
    reasons: tuple = ()
    notes: tuple = ()
    defect: bool = False
    #: The processes this outcome is ABOUT, as `service_id#pid` tokens. Gates 2
    #: and 3 fill it from their own observations; gate 1's subject is a guard run,
    #: not a process, so it stays empty there.
    subjects: tuple = ()
    #: The EARLIER gate's subjects this outcome was decided against. Gate 3 fills
    #: it with the gate-2 observations it tied each live fact to. It is a subject
    #: binding, never an answer: `GATE_EVIDENCE_KINDS` still requires gate 3 to be
    #: answered with `live_process_observation`, and nothing here is read as
    #: evidence for *"is live what was configured"*.
    tied_to: tuple = ()

    def __post_init__(self) -> None:
        if self.gate not in GATE_ORDER:
            raise ValueError(f"gate: {self.gate!r} is not one of {list(GATE_ORDER)}")
        if self.status not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise ValueError(f"status: {self.status!r} is not a Check outcome")
        _require_str(self.evidence_ref, "evidence_ref")
        object.__setattr__(self, "reasons", _require_tuple(self.reasons, "reasons"))
        object.__setattr__(self, "notes", _require_tuple(self.notes, "notes"))
        object.__setattr__(self, "subjects", _require_tuple(self.subjects, "subjects"))
        object.__setattr__(self, "tied_to", _require_tuple(self.tied_to, "tied_to"))
        if self.gate == GATE_1 and self.tied_to:
            raise ValueError(
                f"gate {GATE_1!r} carries tied_to {self.tied_to}: nothing precedes "
                f"gate 1, so there is no earlier subject to be bound to"
            )

        allowed = GATE_EVIDENCE_KINDS[self.gate]
        if self.evidence_kind not in allowed:
            reason = REFUSED_EVIDENCE_KINDS.get(self.evidence_kind)
            if reason is None:
                other = [g for g, kinds in GATE_EVIDENCE_KINDS.items()
                         if self.evidence_kind in kinds]
                reason = (
                    f"it answers {other[0]!r}, a different gate"
                    if other else "it is not a recognised evidence kind"
                )
            raise GateEvidenceMisuse(
                f"gate {self.gate!r} cannot be answered with evidence kind "
                f"{self.evidence_kind!r}: {reason}. The three gates are distinct and "
                f"none implies the next (§11.6); {self.gate!r} accepts "
                f"{sorted(allowed)}"
            )
        if self.status == schemas.PASS and self.reasons:
            raise ValueError(
                f"gate {self.gate!r}: a PASS carrying failure reasons {self.reasons} is "
                f"incoherent — put non-blocking observations in `notes`"
            )
        if self.status != schemas.PASS and not self.reasons:
            raise ValueError(
                f"gate {self.gate!r}: a {self.status} must say why; an unexplained "
                f"non-pass is not auditable"
            )

    @property
    def passed(self) -> bool:
        return self.status == schemas.PASS


@dataclass(frozen=True)
class ThreeGateResult:
    """The stack-change release verdict: all three gates, in order, or nothing.

    Constructed by `evaluate_three_gates()`. `__post_init__` RE-DERIVES the
    status from the gates it carries and raises `GateVerdictTampering` if the
    stored status differs, so no code path can attach a release-eligible status
    to a set of gates that does not support it.
    """

    pipeline_green: Optional[GateOutcome]
    stack_starts: Optional[GateOutcome]
    live_equals_config: Optional[GateOutcome]
    status: str
    blocked_at: Optional[str]
    reasons: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", _require_tuple(self.reasons, "reasons"))
        derived_status, derived_blocked, derived_reasons = _derive_three_gate_status(
            self.gates
        )
        if derived_status != self.status or derived_blocked != self.blocked_at:
            raise GateVerdictTampering(
                f"ThreeGateResult carries status={self.status!r} blocked_at="
                f"{self.blocked_at!r} but its own gates derive "
                f"{derived_status!r}/{derived_blocked!r}"
            )
        if tuple(derived_reasons) != tuple(self.reasons):
            raise GateVerdictTampering(
                "ThreeGateResult carries reasons that do not follow from its gates"
            )
        _require_gate_subject_binding(self.stack_starts, self.live_equals_config)

    @property
    def gates(self) -> tuple:
        return (self.pipeline_green, self.stack_starts, self.live_equals_config)

    @property
    def released(self) -> bool:
        """True only when all three gates PASSed. Nothing else is release-eligible."""
        return self.status == schemas.PASS

    def require_release_eligible(self) -> None:
        if not self.released:
            raise ServingAdapterError(
                f"stack change is not release-eligible: status={self.status!r}, "
                f"blocked at {self.blocked_at!r} — {'; '.join(self.reasons)}"
            )


def _require_gate_subject_binding(
    stack_starts: Optional[GateOutcome], live_equals_config: Optional[GateOutcome],
) -> None:
    """Refuse a three-gate result whose gate 3 is about processes gate 2 is not.

    Checked HERE rather than only inside `gate_live_equals_config()` because this
    is the one place a release verdict can come into existence. Every conjunct is
    required to be non-empty on purpose: two empty tuples compare equal, so a tie
    that only tested `tied_to == subjects` would have been satisfiable by
    supplying neither — the same shape as an audit passed by handing it nothing to
    audit.
    """
    if live_equals_config is None:
        return
    if stack_starts is None:
        raise GateSubjectUnbound(
            "a gate-3 outcome was assembled with no gate-2 outcome; gate 3 is "
            "decided about the processes gate 2 observed, and there are none"
        )
    if not stack_starts.subjects:
        raise GateSubjectUnbound(
            f"the gate-2 outcome {stack_starts.evidence_ref!r} names no subjects, so "
            f"there is nothing for gate 3 to be bound to. A binding to an empty set "
            f"is not a binding"
        )
    if not live_equals_config.tied_to:
        raise GateSubjectUnbound(
            f"the gate-3 outcome {live_equals_config.evidence_ref!r} declares no "
            f"gate-2 subjects it was decided against. Gate 2 could have proved that "
            f"one set of processes came up correctly linked while gate 3 proved a "
            f"different set matches its configuration, and the pair would read as "
            f"'the stack that started is the stack that is running' with nothing "
            f"having compared the two (§11.6)"
        )
    if set(live_equals_config.tied_to) != set(stack_starts.subjects):
        raise GateSubjectUnbound(
            f"gate 3 was decided against {sorted(live_equals_config.tied_to)} but "
            f"gate 2 is about {sorted(stack_starts.subjects)}; the two gates are "
            f"about different processes, so neither says anything about the other's"
        )


def _derive_three_gate_status(gates: Sequence) -> tuple:
    reasons: list = []
    for name, outcome in zip(GATE_ORDER, gates):
        if outcome is None:
            reasons.append(
                f"gate {name!r} was not attempted; an unattempted gate is not a "
                f"passed one (§11.6)"
            )
            return schemas.COULD_NOT_CHECK, name, tuple(reasons)
        if outcome.status != schemas.PASS:
            reasons.extend(f"gate {name!r}: {r}" for r in outcome.reasons)
            return outcome.status, name, tuple(reasons)
    return schemas.PASS, None, ()


def evaluate_three_gates(
    *,
    pipeline_green: Optional[GateOutcome],
    stack_starts: Optional[GateOutcome] = None,
    live_equals_config: Optional[GateOutcome] = None,
) -> ThreeGateResult:
    """Assemble the three gates in order and derive the release verdict.

    Enforces the two properties §11.6 states and one it implies:

      * each outcome must be for its own gate — a gate-2 outcome passed as gate 3
        is refused, not re-labelled;
      * a later gate may not be supplied while an earlier one is not PASS. That
        is `GateOrderViolation`, and it is the *reverse* of the usual confusion:
        holding a green gate 3 next to a red gate 1 invites reading the pair as
        "mostly fine", when the order exists because gate 1's subject (the
        generated priors) is what gates 2 and 3 are supposed to be running;
      * a gate nobody attempted yields COULD_NOT_CHECK, never PASS.
    """
    supplied = (pipeline_green, stack_starts, live_equals_config)
    for name, outcome in zip(GATE_ORDER, supplied):
        if outcome is None:
            continue
        if not isinstance(outcome, GateOutcome):
            raise TypeError(f"{name}: expected a GateOutcome, got {type(outcome).__name__}")
        if outcome.gate != name:
            raise GateOrderViolation(
                f"outcome for gate {outcome.gate!r} was supplied in the {name!r} "
                f"position; the gates are distinct and none substitutes for another"
            )

    for i, outcome in enumerate(supplied):
        if outcome is not None and outcome.status == schemas.PASS:
            continue
        for later_name, later in zip(GATE_ORDER[i + 1:], supplied[i + 1:]):
            if later is not None:
                blocker = GATE_ORDER[i]
                state = "not attempted" if outcome is None else outcome.status
                raise GateOrderViolation(
                    f"gate {later_name!r} was supplied while gate {blocker!r} is "
                    f"{state}; the three gates pass in order (§11.6)"
                )
        break

    status, blocked_at, reasons = _derive_three_gate_status(supplied)
    return ThreeGateResult(
        pipeline_green=pipeline_green,
        stack_starts=stack_starts,
        live_equals_config=live_equals_config,
        status=status,
        blocked_at=blocked_at,
        reasons=reasons,
    )


# =============================================================================
# Gate 1 — PIPELINE GREEN (`stack_change_guard.py`)
# =============================================================================

@dataclass(frozen=True)
class GuardInvocation:
    """The exact invocation of `stack_change_guard.py` a gate-1 run performed.

    Carried with the result so `gate_pipeline_green()` can check HOW the guard
    was run, not just what it printed. `--strict` is the difference between "no
    stale hashes" and "no known gaps", and a release gate needs the second.

    `--surface-summary-only` is deliberately not offered: it replaces the warning
    lines with counts, and an accepted-warning comparison needs the text. A count
    cannot be checked against an acceptance list.
    """

    argv: tuple
    cwd: str
    priors_path: str
    strict: bool
    script: str = str(GUARD_SCRIPT)

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv",
                           _require_tuple(self.argv, "argv", non_empty=True))
        _require_str(self.cwd, "cwd")
        _require_str(self.priors_path, "priors_path")
        _require_str(self.script, "script")
        if not isinstance(self.strict, bool):
            raise ValueError("strict: expected a bool")
        if ("--strict" in self.argv) != self.strict:
            raise ValueError(
                "strict: the flag and the argv disagree; the recorded invocation "
                "must be the invocation"
            )
        if "--surface-summary-only" in self.argv:
            raise ValueError(
                "--surface-summary-only replaces the warning lines with counts, so "
                "the accepted-warning comparison cannot run; gate 1 needs the text"
            )


def build_guard_invocation(
    *,
    priors_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    script: Optional[str] = None,
    python: str = "python3",
    strict: bool = True,
    accepted_gaps_path: Optional[str] = None,
    surface_exceptions_path: Optional[str] = None,
    all_hardcoded_surfaces: bool = False,
) -> GuardInvocation:
    """Construct gate 1's invocation. Codified, never hand-typed.

    `bench-cpu.md:8-10` and `MEASUREMENT_POLICY.md:37` — hand-typed argv voids a
    run. The same discipline applies to a release gate's instrument: the argv is
    constructed here, recorded, and checked, so "which guard ran, with which
    inputs" is a fact the package carries rather than shell history nobody kept.

    The paths default to the orchestrator repo's own defaults. They are
    overridable ONLY so a test can point the contract at a fixture; nothing in
    this module reads or writes either repository.
    """
    root = repo_root or str(GUARD_REPO_ROOT)
    target = script or str(GUARD_SCRIPT)
    priors = priors_path or str(Path(root) / "orchestration" / "derived" / "stack_priors.yaml")
    argv = [python, target, "--priors", priors]
    if strict:
        argv.append("--strict")
    argv.extend(["--repo-root", root])
    if accepted_gaps_path:
        argv.extend(["--accepted-gaps", accepted_gaps_path])
    if surface_exceptions_path:
        argv.extend(["--surface-exceptions", surface_exceptions_path])
    if all_hardcoded_surfaces:
        argv.append("--all-hardcoded-surfaces")
    return GuardInvocation(argv=tuple(argv), cwd=root, priors_path=priors,
                           strict=strict, script=target)


class GuardRunner(Protocol):
    """The seam that actually spawns `stack_change_guard.py`.

    This module models the guard's CLI contract and parses its output; it does
    not spawn. The caller supplies a runner returning `(returncode, stdout,
    stderr)`. Keeping the spawn outside is what lets `audit_no_write_or_process_
    paths()` prove this file cannot start a process.
    """

    def __call__(self, invocation: GuardInvocation) -> tuple:  # pragma: no cover
        ...


_GUARD_HEADER_RE = re.compile(r"^(OK|WARN|FAIL):\s*(.*)$")
_GUARD_COUNT_RE = re.compile(r"^(\d+)\s+")
_GUARD_BULLET_RE = re.compile(r"^\s+-\s+(.*)$")


@dataclass(frozen=True)
class GuardResult:
    """A parsed `stack_change_guard.py` run.

    `parsed=False` is the third outcome: a guard whose output this module cannot
    read is not a green guard, and gate 1 reports COULD_NOT_CHECK for it.
    """

    invocation: GuardInvocation
    returncode: int
    header: Optional[str]
    errors: tuple = ()
    warnings: tuple = ()
    defects: tuple = ()
    parsed: bool = False
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        for name in ("errors", "warnings", "defects"):
            object.__setattr__(self, name, _require_tuple(getattr(self, name), name))
        if self.header is not None and self.header not in ("OK", "WARN", "FAIL"):
            raise ValueError(f"header: {self.header!r} is not an OK/WARN/FAIL header")


def parse_guard_result(
    invocation: GuardInvocation, returncode: int, stdout: str, stderr: str = "",
) -> GuardResult:
    """Parse the guard's documented output contract.

    The contract (`stack_change_guard.py:main`) is:
      `OK: <priors path>`                 exit 0
      `WARN: N stack-prior warning(s)`    exit 0, then `  - <text>` per warning
      `FAIL: N stack-prior error(s)`      exit 1, then `  - <text>` per error

    Two disagreements are recorded as **defects** rather than resolved in favour
    of the cheaper answer, which is the §3.2 stage-disagreement rule applied to
    an instrument instead of a build: an exit code that contradicts the body, and
    a header count that contradicts the number of bullets. Either means the
    contract this gate is built on has moved, and a gate built on a moved
    contract is not evidence.
    """
    header = None
    detail: list = []
    declared_count = None
    for line in stdout.splitlines():
        match = _GUARD_HEADER_RE.match(line)
        if match and header is None:
            header = match.group(1)
            count_match = _GUARD_COUNT_RE.match(match.group(2))
            if count_match:
                declared_count = int(count_match.group(1))
            continue
        bullet = _GUARD_BULLET_RE.match(line)
        if bullet and header is not None:
            detail.append(bullet.group(1).strip())

    defects: list = []
    if header is None:
        return GuardResult(
            invocation=invocation, returncode=returncode, header=None,
            defects=("guard produced no OK/WARN/FAIL header line; its output does "
                     "not match the contract gate 1 is built on",),
            parsed=False, stdout=stdout, stderr=stderr,
        )

    expected_rc = 1 if header == "FAIL" else 0
    if returncode != expected_rc:
        defects.append(
            f"guard printed a {header} header but exited {returncode} (contract "
            f"says {expected_rc}); exit code and body disagree, which is a hard "
            f"finding, not a preference for the cheaper answer"
        )
    if declared_count is not None and declared_count != len(detail):
        defects.append(
            f"guard declared {declared_count} item(s) but printed {len(detail)} "
            f"detail line(s); the enumeration gate 1 checks is incomplete"
        )

    errors = tuple(detail) if header == "FAIL" else ()
    warnings = tuple(detail) if header == "WARN" else ()
    return GuardResult(
        invocation=invocation, returncode=returncode, header=header,
        errors=errors, warnings=warnings, defects=tuple(defects), parsed=True,
        stdout=stdout, stderr=stderr,
    )


def run_guard(invocation: GuardInvocation, runner: Optional[GuardRunner]) -> GuardResult:
    """Run the guard through the caller's runner and parse the result.

    `runner` is required and has no default (`GuardRunnerNotWired`). A default
    would either spawn from this module or silently report an unrun guard as
    clean, and a fail-open gate 1 is the whole class of defect §11.6 exists over.
    """
    if runner is None:
        raise GuardRunnerNotWired(
            "run_guard() needs a GuardRunner: this module models the guard's "
            "contract and never spawns a process. Supply a runner that executes "
            f"{invocation.argv!r} and returns (returncode, stdout, stderr)"
        )
    outcome = runner(invocation)
    if not isinstance(outcome, tuple) or len(outcome) != 3:
        raise ServingAdapterError(
            f"GuardRunner must return (returncode, stdout, stderr); got {outcome!r}"
        )
    returncode, stdout, stderr = outcome
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise ServingAdapterError(
            f"GuardRunner returncode must be an int, got {returncode!r}"
        )
    return parse_guard_result(invocation, returncode, str(stdout), str(stderr))


def gate_pipeline_green(
    result: GuardResult, *, accepted_warnings: Sequence = (),
) -> GateOutcome:
    """Gate 1: derived priors consistent with source, no retired role live, no
    stale consumer surface.

    Warning handling is the load-bearing part. The guard exits 0 on warnings, so
    the exit code alone conflates "clean" with "warned about a hardcoded consumer
    surface". A release gate that took exit 0 as green would pass exactly the
    staleness it exists to catch. Every warning must therefore be matched by a
    PRE-DECLARED acceptance; an unaccepted warning FAILs, and an acceptance that
    matched nothing is surfaced as a note rather than dropped — a stale
    acceptance list is how a real warning gets silently pre-forgiven later.
    """
    accepted = _require_tuple(accepted_warnings, "accepted_warnings")
    ref = f"guard:{result.invocation.priors_path}:rc={result.returncode}"

    if not result.parsed:
        return GateOutcome(
            gate=GATE_1, status=schemas.COULD_NOT_CHECK,
            evidence_kind="stack_change_guard_result", evidence_ref=ref,
            reasons=result.defects or ("guard output could not be parsed",),
            defect=True,
        )
    if result.defects:
        return GateOutcome(
            gate=GATE_1, status=schemas.FAIL,
            evidence_kind="stack_change_guard_result", evidence_ref=ref,
            reasons=result.defects, defect=True,
        )
    if not result.invocation.strict:
        return GateOutcome(
            gate=GATE_1, status=schemas.FAIL,
            evidence_kind="stack_change_guard_result", evidence_ref=ref,
            reasons=("guard ran without --strict: without it the guard fails only on "
                     "stale hashes and live-role invariants, so a known gap passes "
                     "and gate 1 would certify a pipeline it did not fully check",),
        )
    weakened = tuple(
        f"guard ran with {flag}: {reason}"
        for flag, reason in GUARD_SCOPE_WEAKENING_FLAGS.items()
        if flag in result.invocation.argv
    )
    if weakened:
        return GateOutcome(
            gate=GATE_1, status=schemas.FAIL,
            evidence_kind="stack_change_guard_result", evidence_ref=ref,
            reasons=weakened,
        )
    if result.errors:
        return GateOutcome(
            gate=GATE_1, status=schemas.FAIL,
            evidence_kind="stack_change_guard_result", evidence_ref=ref,
            reasons=tuple(f"guard error: {e}" for e in result.errors),
        )

    unaccepted = tuple(w for w in result.warnings if w not in accepted)
    if unaccepted:
        return GateOutcome(
            gate=GATE_1, status=schemas.FAIL,
            evidence_kind="stack_change_guard_result", evidence_ref=ref,
            reasons=tuple(
                f"unaccepted guard warning: {w} — the guard exits 0 on warnings, so "
                f"gate 1 requires each one to be pre-declared" for w in unaccepted
            ),
        )
    notes = [f"accepted guard warning: {w}" for w in result.warnings]
    notes.extend(
        f"stale acceptance (matched no warning this run): {a}"
        for a in accepted if a not in result.warnings
    )
    return GateOutcome(
        gate=GATE_1, status=schemas.PASS,
        evidence_kind="stack_change_guard_result", evidence_ref=ref,
        notes=tuple(notes),
    )


# =============================================================================
# Gate 2 — THE STACK STARTS
# =============================================================================

@dataclass(frozen=True)
class LinkageReceipt:
    """A `verify_ggml_linkage.sh` receipt for one service.

    §10.2 and CLAUDE.md's speech-kernel freeze: the three trees run three ggml
    generations, so *"every launcher must set its own `LD_LIBRARY_PATH` and prove
    it"* — a binary that inherits another tree's ggml runs silently wrong. The
    receipt names the verifier, so a receipt produced by something else is
    visible rather than assumed equivalent.
    """

    verifier: str
    status: str
    resolved_tree: str
    receipt_ref: str

    def __post_init__(self) -> None:
        _require_str(self.verifier, "verifier")
        _require_str(self.receipt_ref, "receipt_ref")
        if self.status not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise ValueError(f"status: {self.status!r} is not a Check outcome")
        if self.resolved_tree not in TREE_ROOTS:
            raise ValueError(
                f"resolved_tree: {self.resolved_tree!r} is not one of "
                f"{sorted(TREE_ROOTS)}"
            )


@dataclass(frozen=True)
class ServiceStartObservation:
    """One service's start, as observed — never as intended.

    `start_index`/`started_at`/`ready_at` exist because
    `feedback_sequential_model_loading` is a real constraint on this host:
    services come up sequentially, and two overlapping starts on a 1.1TB host
    with NUMA-sensitive placement is not the configuration anybody validated.
    """

    service_id: str
    tree: str
    start_index: int
    started_at: str
    ready_at: str
    pid: int
    state: str
    ld_library_path: str
    linkage: Optional[LinkageReceipt] = None

    def __post_init__(self) -> None:
        _require_str(self.service_id, "service_id")
        _require_str(self.started_at, "started_at")
        _require_str(self.ready_at, "ready_at")
        _require_str(self.state, "state")
        if self.tree not in SERVICE_TREES:
            raise ValueError(
                f"tree: {self.tree!r} is not one of {sorted(SERVICE_TREES)} — a "
                f"service that links no ggml tree declares {NO_GGML_TREE!r}"
            )
        if not isinstance(self.start_index, int) or isinstance(self.start_index, bool):
            raise ValueError("start_index: expected an int")
        if self.start_index < 0:
            raise ValueError("start_index: must be >= 0")
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 0:
            raise ValueError("pid: expected a positive int")
        if not isinstance(self.ld_library_path, str):
            raise ValueError("ld_library_path: expected a string")
        if self.linkage is not None and not isinstance(self.linkage, LinkageReceipt):
            raise TypeError("linkage: expected a LinkageReceipt or None")


class StackStartObserver(Protocol):
    """The seam that observes a stack coming up.

    It OBSERVES. Starting the stack is `orchestrator_stack.py`, executed by
    whoever owns the inference at a moment they choose
    (`OPERATING_CONSTRAINTS.md:41`, `feedback_use_orchestrator_stack_for_lifecycle`).
    This adapter has no start, stop or signal verb anywhere in it.
    """

    def observe(self, service_ids: Sequence) -> Sequence:  # pragma: no cover
        ...


def _ld_path_entries(value: str) -> list:
    return [entry for entry in value.split(":") if entry]


def _under_root(entry: str, root: str) -> bool:
    """True only when `entry` is inside `root`, on a path-COMPONENT boundary.

    A bare `startswith` is not a containment test, and the difference is the
    whole of gate 2. `/mnt/raid0/llm/llama.cpp-experimental/build/bin` starts with
    `/mnt/raid0/llm/llama.cpp` while being a DIFFERENT tree carrying a different
    ggml generation — and that host has a dozen such siblings on disk
    (`llama.cpp-experimental`, `llama.cpp-v5`, `llama.cpp-mi210-hip`, …), one of
    which is where CLAUDE.md says all kernel work happens. Under a prefix test,
    a service linked against the experimental tree reads as correctly linked
    against production AND is not reported as foreign: the silently-wrong-ggml
    failure the speech-kernel freeze was ratified over, passing its own gate.
    """
    return entry == root or entry.startswith(root + "/")


def gate_stack_starts(
    observations: Sequence, *, affected_services: Sequence,
) -> GateOutcome:
    """Gate 2: every affected service comes up, sequentially, with correct
    per-tree `LD_LIBRARY_PATH` proven by the linkage verifier.

    Four independent failures, none of which the other three would catch:

      * a service that did not come up at all, or came up not running;
      * starts that overlap — "sequentially" is part of the gate, not a style
        preference;
      * a missing or failing linkage receipt (COULD_NOT_CHECK for missing: an
        unproven linkage is not a proven one);
      * an `LD_LIBRARY_PATH` that reaches into another tree, which is the silent
        wrong-ggml failure the speech freeze was ratified over.

    An observation naming a service outside the declared affected set is a scope
    disagreement, not noise: invariant 18 makes the affected surface derived and
    traced, so the observer and the manifest disagreeing is a finding.
    """
    declared = tuple(_require_tuple(affected_services, "affected_services",
                                    non_empty=True))
    observed = list(observations)
    for i, obs in enumerate(observed):
        if not isinstance(obs, ServiceStartObservation):
            raise TypeError(
                f"observations[{i}]: expected a ServiceStartObservation, got "
                f"{type(obs).__name__}"
            )
    ref = "services:" + ",".join(sorted(o.service_id for o in observed))
    #: The processes this gate is about, for gate 3 to be bound to. Derived from
    #: the observations, never from `affected_services` — the declared set is the
    #: manifest's claim about what should have started, and binding gate 3 to it
    #: would tie gate 3 to an intention rather than to an observed process.
    subjects = tuple(sorted(f"{o.service_id}#{o.pid}" for o in observed))

    reasons: list = []
    notes: list = []
    undecidable = False

    by_id = {}
    for obs in observed:
        if obs.service_id in by_id:
            reasons.append(
                f"service {obs.service_id!r} appears twice in one observation set"
            )
        by_id[obs.service_id] = obs

    for service_id in declared:
        if service_id not in by_id:
            reasons.append(
                f"affected service {service_id!r} was not observed starting; gate 2 "
                f"is 'every affected service comes up', not 'the ones that did'"
            )
    for service_id in by_id:
        if service_id not in declared:
            reasons.append(
                f"observed service {service_id!r} is not in the declared affected "
                f"set {list(declared)}; declared equals traced (invariant 18), and a "
                f"disagreement is a finding rather than a service to ignore"
            )

    for obs in observed:
        if obs.state != "running":
            reasons.append(
                f"service {obs.service_id!r} is in state {obs.state!r}, not 'running'"
            )

    ordered = sorted(observed, key=lambda o: o.start_index)
    indices = [o.start_index for o in ordered]
    if len(set(indices)) != len(indices):
        reasons.append(
            f"start_index values {indices} are not distinct; two services claiming "
            f"the same position did not start sequentially"
        )
    elif indices and indices != list(range(len(indices))):
        reasons.append(
            f"start_index values {indices} are not contiguous from 0; the start "
            f"sequence has a hole, so an unobserved service started somewhere in it"
        )

    previous = None
    for obs in ordered:
        start = _parse_instant(obs.started_at, "started_at")
        ready = _parse_instant(obs.ready_at, "ready_at")
        if start is None or ready is None:
            undecidable = True
            notes.append(
                f"service {obs.service_id!r}: start/ready timestamps are absent, "
                f"naive, or unparseable, so sequencing cannot be checked"
            )
            previous = None
            continue
        if ready < start:
            reasons.append(
                f"service {obs.service_id!r} reports ready_at before started_at"
            )
        if previous is not None and start < previous[1]:
            reasons.append(
                f"service {obs.service_id!r} started at {obs.started_at} before "
                f"{previous[0]!r} was ready at {previous[1].isoformat()}; the stack "
                f"comes up sequentially (feedback_sequential_model_loading)"
            )
        previous = (obs.service_id, ready)

    for obs in observed:
        entries = _ld_path_entries(obs.ld_library_path)

        if obs.tree == NO_GGML_TREE:
            intruders = [
                entry for entry in entries
                if entry.startswith(_TREE_PARENT + "/")
            ]
            if intruders:
                reasons.append(
                    f"service {obs.service_id!r} declares it links no ggml tree but "
                    f"its LD_LIBRARY_PATH reaches into one: {intruders}"
                )
            if obs.linkage is not None:
                reasons.append(
                    f"service {obs.service_id!r} declares {NO_GGML_TREE!r} yet carries "
                    f"a linkage receipt resolving {obs.linkage.resolved_tree!r}; one of "
                    f"the two is wrong and neither may be assumed"
                )
            continue

        own_root = TREE_ROOTS[obs.tree]
        if not entries:
            reasons.append(
                f"service {obs.service_id!r} has an empty LD_LIBRARY_PATH; each "
                f"launcher must set its own (CLAUDE.md, speech-kernel freeze)"
            )
        else:
            if not any(_under_root(entry, own_root) for entry in entries):
                reasons.append(
                    f"service {obs.service_id!r} (tree {obs.tree!r}) has no "
                    f"LD_LIBRARY_PATH entry under {own_root}"
                )
            # An entry under NO declared tree root but under `/mnt/raid0/llm/` is
            # a sibling worktree (`llama.cpp-experimental`, `llama.cpp-v5`, …).
            # It is not "somebody else's declared tree", so the foreign-tree
            # reason above does not reach it — and it is exactly the wrong-ggml
            # case, so it is named on its own terms rather than passed over.
            sibling = [
                entry for entry in entries
                if entry.startswith(_TREE_PARENT + "/")
                and not any(_under_root(entry, root) for root in TREE_ROOTS.values())
            ]
            if sibling:
                reasons.append(
                    f"service {obs.service_id!r} (tree {obs.tree!r}) has "
                    f"LD_LIBRARY_PATH entries in a worktree that is not a declared "
                    f"source tree: {sibling}. A sibling of {own_root} carries its own "
                    f"ggml build; only {sorted(TREE_ROOTS.values())} are the trees "
                    f"gate 2 can vouch for"
                )
            foreign = [
                entry for entry in entries
                for tree, root in TREE_ROOTS.items()
                if tree != obs.tree and _under_root(entry, root)
            ]
            if foreign:
                reasons.append(
                    f"service {obs.service_id!r} (tree {obs.tree!r}) has "
                    f"LD_LIBRARY_PATH entries in another tree: {foreign}. The three "
                    f"trees run three ggml generations; a binary that inherits "
                    f"another tree's ggml runs silently wrong"
                )

        if obs.linkage is None:
            undecidable = True
            notes.append(
                f"service {obs.service_id!r} carries no linkage receipt; an unproven "
                f"linkage is not a proven one"
            )
            continue
        if obs.linkage.verifier != LINKAGE_VERIFIER:
            reasons.append(
                f"service {obs.service_id!r} linkage was produced by "
                f"{obs.linkage.verifier!r}, not the named verifier "
                f"{LINKAGE_VERIFIER!r} (§10.2)"
            )
        if obs.linkage.status == schemas.FAIL:
            reasons.append(
                f"service {obs.service_id!r} failed linkage verification "
                f"({obs.linkage.receipt_ref})"
            )
        elif obs.linkage.status == schemas.COULD_NOT_CHECK:
            undecidable = True
            notes.append(
                f"service {obs.service_id!r} linkage verification returned "
                f"COULD_NOT_CHECK ({obs.linkage.receipt_ref})"
            )
        if obs.linkage.resolved_tree != obs.tree:
            reasons.append(
                f"service {obs.service_id!r} declares tree {obs.tree!r} but its "
                f"linkage resolved into {obs.linkage.resolved_tree!r}"
            )

    if reasons:
        return GateOutcome(gate=GATE_2, status=schemas.FAIL,
                           evidence_kind="service_start_observation",
                           evidence_ref=ref, reasons=tuple(reasons),
                           notes=tuple(notes), subjects=subjects)
    if undecidable:
        return GateOutcome(gate=GATE_2, status=schemas.COULD_NOT_CHECK,
                           evidence_kind="service_start_observation",
                           evidence_ref=ref, subjects=subjects,
                           reasons=tuple(notes) or ("gate 2 could not be decided",))
    return GateOutcome(gate=GATE_2, status=schemas.PASS,
                       evidence_kind="service_start_observation",
                       evidence_ref=ref, notes=tuple(notes), subjects=subjects)


# =============================================================================
# Gate 3 — LIVE EQUALS CONFIG
# =============================================================================

@dataclass(frozen=True)
class IntendedProcessConfig:
    """What a service was configured to be.

    `flags_are_exhaustive` is required rather than defaulted because the two
    readings differ in what a surprise flag means. Exhaustive: a live flag nobody
    intended is drift and FAILs. Non-exhaustive: it is recorded and not judged.
    Defaulting either way would decide that silently for every caller.
    """

    service_id: str
    binary_path: str
    binary_sha256: str
    flags: tuple
    cpu_affinity: tuple
    config_sha256: str
    config_recorded_at: str
    flags_are_exhaustive: bool

    def __post_init__(self) -> None:
        _require_str(self.service_id, "service_id")
        _require_str(self.binary_path, "binary_path")
        _require_sha256(self.binary_sha256, "binary_sha256")
        _require_sha256(self.config_sha256, "config_sha256")
        _require_str(self.config_recorded_at, "config_recorded_at")
        object.__setattr__(self, "flags", _require_tuple(self.flags, "flags"))
        object.__setattr__(self, "cpu_affinity",
                           _require_tuple(self.cpu_affinity, "cpu_affinity",
                                          item_type=int))
        if not isinstance(self.flags_are_exhaustive, bool):
            raise ValueError("flags_are_exhaustive: expected a bool")


@dataclass(frozen=True)
class LiveProcessFact:
    """What a service actually IS, read from the running process.

    `observation_source` is validated against `LIVE_OBSERVATION_SOURCES`, and the
    refusals are named: the config file, the topology hash, and the registry are
    each *the thing gate 3 is supposed to be independent of*. Verifying live
    affinity against a topology hash is a mistake this project has already made
    (`feedback_verify_live_affinity_not_just_topology_hash`).
    """

    service_id: str
    pid: int
    observation_source: str
    binary_path: str
    binary_sha256: str
    argv: tuple
    cpu_affinity: tuple
    started_at: str

    def __post_init__(self) -> None:
        _require_str(self.service_id, "service_id")
        _require_str(self.binary_path, "binary_path")
        _require_sha256(self.binary_sha256, "binary_sha256")
        _require_str(self.started_at, "started_at")
        object.__setattr__(self, "argv",
                           _require_tuple(self.argv, "argv", non_empty=True))
        object.__setattr__(self, "cpu_affinity",
                           _require_tuple(self.cpu_affinity, "cpu_affinity",
                                          item_type=int))
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 0:
            raise ValueError("pid: expected a positive int")
        if self.observation_source not in LIVE_OBSERVATION_SOURCES:
            reason = REFUSED_EVIDENCE_KINDS.get(
                self.observation_source,
                "it is not a source that observes a running process",
            )
            raise GateEvidenceMisuse(
                f"observation_source {self.observation_source!r} cannot establish "
                f"live state: {reason}. Gate 3 verifies against live state rather "
                f"than against the config that was supposed to produce it (§11.6). "
                f"Live sources: {sorted(LIVE_OBSERVATION_SOURCES)}"
            )


class LiveProcessObserver(Protocol):
    """The seam that reads live process facts.

    Reads `/proc`, `sched_getaffinity`, cgroup membership. It never signals: this
    host is shared, and a name-pattern process action here has already killed
    another session's server twice (INC-20260731-broad-process-pattern-kills).
    """

    def observe(self, service_ids: Sequence) -> Sequence:  # pragma: no cover
        ...


def _flag_map(tokens: Sequence) -> dict:
    """Parse `--flag value` / `--flag=value` / bare `--flag` into a mapping.

    Repeated flags keep every value: `--override a --override b` is not the same
    configuration as `--override b`, and collapsing it would hide a difference.
    """
    out: dict = {}
    i = 0
    tokens = list(tokens)
    while i < len(tokens):
        token = tokens[i]
        if not isinstance(token, str) or not token.startswith("-"):
            i += 1
            continue
        if "=" in token:
            name, value = token.split("=", 1)
        else:
            name = token
            value = None
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if isinstance(nxt, str) and not nxt.startswith("-"):
                    value = nxt
                    i += 1
        out.setdefault(name, []).append(value)
        i += 1
    return out


def _program_name(path: str) -> str:
    """The last path component. `''` for the empty string, never an exception."""
    return path.rsplit("/", 1)[-1]


def gate_live_equals_config(
    intended: Sequence, live: Sequence, *, started: Sequence,
) -> GateOutcome:
    """Gate 3: the running processes match the intended configuration.

    Right binary, right argv, right flags, right affinity — not stale, and **the
    same processes gate 2 measured**. The staleness check is separate from the
    comparisons on purpose: a process started before its configuration was written
    can match every field and still be running the previous configuration's
    behaviour, because the file it would have read did not exist yet (CLAUDE.md,
    *"check the running process isn't stale"*).

    `started` is gate 2's own observations and is REQUIRED, not optional. Without
    it gate 3 graded whatever `LiveProcessFact` set it was handed: gate 2 could
    have proved that pid 1000 came up correctly linked while gate 3 proved that
    pid 4242 matched its config, and the pair read as "the stack that started is
    the stack that is running" when nothing had checked that they were the same
    process. Two ties close it, because a pid alone does not identify a process:

      * `LiveProcessFact.pid` must equal the pid gate 2 observed for that service;
      * that pid's start instant must fall inside the start→ready window gate 2
        watched it come up in. A pid is a reusable number, so matching pids with
        incompatible start times are two different processes wearing one number.

    Consuming gate 2's observation is not gate 2's evidence answering gate 3:
    `GATE_EVIDENCE_KINDS` still binds this outcome to `live_process_observation`,
    and no gate-2 fact is read as an answer to *"is live what was configured"*.
    It is read only to establish that the SUBJECT of both gates is one process.

    `argv[0]` is compared to the observed `binary_path` for the same reason. The
    two come from different `/proc` reads (`proc_cmdline` and `proc_exe`), so a
    disagreement means the observation stitched two processes together — or that
    the binary was exec'd under another program's name.
    """
    intended_list = list(intended)
    live_list = list(live)
    started_list = list(started)
    for i, obs in enumerate(started_list):
        if not isinstance(obs, ServiceStartObservation):
            raise TypeError(
                f"started[{i}]: expected a ServiceStartObservation, got "
                f"{type(obs).__name__}"
            )
    if not started_list:
        raise ValueError(
            "started: gate 3 is decided about the processes gate 2 observed. With no "
            "gate-2 observation there is nothing to tie the live facts to, and a PASS "
            "would certify that the stack which started is the stack which is running "
            "without anything having compared the two"
        )
    for i, item in enumerate(intended_list):
        if not isinstance(item, IntendedProcessConfig):
            raise TypeError(
                f"intended[{i}]: expected an IntendedProcessConfig, got "
                f"{type(item).__name__}"
            )
    for i, item in enumerate(live_list):
        if not isinstance(item, LiveProcessFact):
            raise TypeError(
                f"live[{i}]: expected a LiveProcessFact, got {type(item).__name__}"
            )
    if not intended_list:
        raise ValueError(
            "intended: gate 3 compares live state against an intended configuration; "
            "with nothing intended there is nothing to verify, and reporting PASS "
            "would certify an empty comparison"
        )

    # Gate 2 FAILs a service that "appears twice in one observation set", and the
    # live side of this very function FAILs two live processes for one service.
    # The gate-2 side was collapsed last-wins, so the SAME evidence in a different
    # list order decided the tie differently: `[pid 1000, pid 4242]` tied to 4242
    # and PASSed, `[pid 4242, pid 1000]` tied to 1000 and FAILed. A verdict that
    # depends on the order its evidence was listed in is not a verdict.
    started_by_id = {}
    started_pids: dict = {}
    tie_reasons: list = []
    for obs in started_list:
        if obs.service_id in started_by_id:
            tie_reasons.append(
                f"service {obs.service_id!r} carries two gate-2 start observations "
                f"(pids {started_by_id[obs.service_id].pid} and {obs.pid}); which of "
                f"them gate 3 is tied to is not determined here, and one of the two "
                f"processes is a leftover"
            )
        started_by_id[obs.service_id] = obs
        started_pids.setdefault(obs.service_id, []).append(obs.pid)

    # The tie is part of the evidence reference, so the durable record shows WHICH
    # gate-2 processes this outcome was decided against rather than only that some
    # tie happened. Every observed pid is named, not the surviving one.
    ref = ("live:" + ",".join(sorted(f.service_id for f in live_list))
           + "|gate2:" + ",".join(
               f"{sid}#" + "+".join(str(pid) for pid in started_pids[sid])
               for sid in sorted(started_pids)))
    live_by_id = {}
    reasons: list = list(tie_reasons)
    notes: list = []
    undecidable = False

    for fact in live_list:
        if fact.service_id in live_by_id:
            reasons.append(
                f"service {fact.service_id!r} has two live processes "
                f"({live_by_id[fact.service_id].pid} and {fact.pid}); one of them is "
                f"a leftover, and which one serves traffic is not determined here"
            )
        live_by_id[fact.service_id] = fact

    intended_ids = {cfg.service_id for cfg in intended_list}
    for fact in live_list:
        if fact.service_id not in intended_ids:
            reasons.append(
                f"live service {fact.service_id!r} (pid {fact.pid}) is not in the "
                f"intended configuration; live has something config does not"
            )

    for cfg in intended_list:
        fact = live_by_id.get(cfg.service_id)
        if fact is None:
            reasons.append(
                f"service {cfg.service_id!r} is configured but no live process was "
                f"observed for it"
            )
            continue
        if fact.binary_sha256 != cfg.binary_sha256:
            reasons.append(
                f"service {cfg.service_id!r} runs binary {fact.binary_sha256[:12]}, "
                f"configured {cfg.binary_sha256[:12]}"
            )
        if fact.binary_path != cfg.binary_path:
            reasons.append(
                f"service {cfg.service_id!r} runs {fact.binary_path!r}, configured "
                f"{cfg.binary_path!r}"
            )

        # argv[0] against the observed binary path. `LiveProcessFact.argv` is
        # non-empty by construction, so there is no missing-argv branch.
        argv0 = fact.argv[0]
        if _program_name(argv0) != _program_name(fact.binary_path):
            reasons.append(
                f"service {cfg.service_id!r}: argv[0] {argv0!r} names program "
                f"{_program_name(argv0)!r} while the observed binary is "
                f"{fact.binary_path!r}; the two live reads describe different "
                f"programs, so this fact is not about one process"
            )
        elif argv0.startswith("/"):
            if argv0 != fact.binary_path:
                reasons.append(
                    f"service {cfg.service_id!r}: argv[0] {argv0!r} is an absolute "
                    f"path that is not the observed binary {fact.binary_path!r}; "
                    f"same program name, different file"
                )
        else:
            # The compliant idiom: a launcher may exec with a bare or relative
            # argv[0]. The program NAME is still compared — only the directory is
            # unavailable — and saying which part could not be compared is the
            # honest record.
            notes.append(
                f"service {cfg.service_id!r}: argv[0] {argv0!r} is not an absolute "
                f"path, so it was compared to {fact.binary_path!r} by program name "
                f"only"
            )

        start_obs = started_by_id.get(cfg.service_id)
        if start_obs is None:
            reasons.append(
                f"service {cfg.service_id!r} has no gate-2 start observation, so the "
                f"live process cannot be tied to one that was observed starting; "
                f"gate 3 would otherwise grade a process gate 2 never measured"
            )
        elif start_obs.pid != fact.pid:
            reasons.append(
                f"service {cfg.service_id!r}: gate 3 observed pid {fact.pid} but "
                f"gate 2 observed pid {start_obs.pid}. The process was replaced "
                f"between the gates, so gate 2's start, linkage and sequencing "
                f"evidence is about a process that is no longer the one serving"
            )
        else:
            live_start = _parse_instant(fact.started_at, "started_at")
            window_open = _parse_instant(start_obs.started_at, "started_at")
            window_close = _parse_instant(start_obs.ready_at, "ready_at")
            if live_start is None or window_open is None or window_close is None:
                undecidable = True
                notes.append(
                    f"service {cfg.service_id!r}: pid {fact.pid} matches gate 2, but a "
                    f"timestamp is absent, naive, or unparseable, so it could not be "
                    f"shown to be the SAME process rather than a reused pid"
                )
            elif not window_open <= live_start <= window_close:
                reasons.append(
                    f"service {cfg.service_id!r}: pid {fact.pid} started at "
                    f"{fact.started_at}, outside the window gate 2 watched it come up "
                    f"in ({start_obs.started_at} .. {start_obs.ready_at}). A pid is a "
                    f"reusable number; matching pids with incompatible start times "
                    f"are two processes, not one"
                )

        live_flags = _flag_map(fact.argv)
        want_flags = _flag_map(cfg.flags)
        for name, values in want_flags.items():
            if name not in live_flags:
                reasons.append(
                    f"service {cfg.service_id!r}: configured flag {name!r} is absent "
                    f"from the live argv"
                )
            elif live_flags[name] != values:
                reasons.append(
                    f"service {cfg.service_id!r}: flag {name!r} is {live_flags[name]!r} "
                    f"live, configured {values!r}"
                )
        extra = [name for name in live_flags if name not in want_flags]
        if extra:
            if cfg.flags_are_exhaustive:
                reasons.append(
                    f"service {cfg.service_id!r}: live argv carries flags the "
                    f"configuration does not: {sorted(extra)}"
                )
            else:
                notes.append(
                    f"service {cfg.service_id!r}: live-only flags {sorted(extra)} "
                    f"(configuration declares itself non-exhaustive)"
                )

        if set(fact.cpu_affinity) != set(cfg.cpu_affinity):
            reasons.append(
                f"service {cfg.service_id!r}: live affinity "
                f"{sorted(set(fact.cpu_affinity))} differs from configured "
                f"{sorted(set(cfg.cpu_affinity))} — affinity is verified per live "
                f"process, never from the topology hash"
            )

        started = _parse_instant(fact.started_at, "started_at")
        recorded = _parse_instant(cfg.config_recorded_at, "config_recorded_at")
        if started is None or recorded is None:
            undecidable = True
            notes.append(
                f"service {cfg.service_id!r}: staleness cannot be decided — a "
                f"timestamp is absent, naive, or unparseable"
            )
        elif started < recorded:
            reasons.append(
                f"service {cfg.service_id!r} (pid {fact.pid}) has been running since "
                f"{fact.started_at}, before its configuration was recorded at "
                f"{cfg.config_recorded_at}; it is a stale process wearing a matching "
                f"configuration"
            )

    subjects = tuple(sorted(f"{f.service_id}#{f.pid}" for f in live_list))
    tied_to = tuple(sorted(f"{o.service_id}#{o.pid}" for o in started_list))
    if reasons:
        return GateOutcome(gate=GATE_3, status=schemas.FAIL,
                           evidence_kind="live_process_observation",
                           evidence_ref=ref, reasons=tuple(reasons),
                           notes=tuple(notes), subjects=subjects, tied_to=tied_to)
    if undecidable:
        return GateOutcome(gate=GATE_3, status=schemas.COULD_NOT_CHECK,
                           evidence_kind="live_process_observation",
                           evidence_ref=ref, subjects=subjects, tied_to=tied_to,
                           reasons=tuple(notes) or ("gate 3 could not be decided",))
    return GateOutcome(gate=GATE_3, status=schemas.PASS,
                       evidence_kind="live_process_observation",
                       evidence_ref=ref, notes=tuple(notes),
                       subjects=subjects, tied_to=tied_to)


# =============================================================================
# Serving evidence — task_rate, variable arrival, latency and SLO first-class
# =============================================================================

@dataclass(frozen=True)
class ArrivalTrace:
    """The recorded arrival process a replay reproduces.

    A replay whose arrivals are generated fresh each run is not the same
    instrument twice, and a scheduler comparison is precisely where arrival
    ordering changes the answer. The trace is identified by content hash and
    replayed under a recorded seed.
    """

    trace_id: str
    trace_sha256: str
    seed: int
    request_count: int
    duration_s: float
    roles: tuple

    def __post_init__(self) -> None:
        _require_str(self.trace_id, "trace_id")
        _require_sha256(self.trace_sha256, "trace_sha256")
        object.__setattr__(self, "roles",
                           _require_tuple(self.roles, "roles", non_empty=True))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed: expected an int")
        if not isinstance(self.request_count, int) or self.request_count <= 0:
            raise ValueError("request_count: expected a positive int")
        if not isinstance(self.duration_s, (int, float)) or self.duration_s <= 0:
            raise ValueError("duration_s: expected a positive number")


@dataclass(frozen=True)
class VariableArrivalReplaySpec:
    """The serving workload. Fixed-shape specs are refused, not re-labelled.

    §19.6 P0.2 makes benchmark class part of evaluator identity: fixed-shape
    feeds kernel campaigns, variable-request feeds `serving_runtime`. The two
    were historically conflated, which is how a scheduler effect got attributed
    to a kernel and vice versa.
    """

    spec_id: str
    benchmark_class: str
    trace: ArrivalTrace
    concurrency: int
    warmup_s: float
    recipe_constructor_id: str
    recipe_sha256: str

    def __post_init__(self) -> None:
        _require_str(self.spec_id, "spec_id")
        _require_str(self.recipe_constructor_id, "recipe_constructor_id")
        _require_sha256(self.recipe_sha256, "recipe_sha256")
        if not isinstance(self.trace, ArrivalTrace):
            raise TypeError("trace: expected an ArrivalTrace")
        if self.benchmark_class != BENCHMARK_CLASS:
            raise FixedShapeWorkloadRefused(
                f"benchmark_class {self.benchmark_class!r} is not {BENCHMARK_CLASS!r}: "
                f"{BACKEND} evidence is variable-arrival replay, and a fixed-shape "
                f"benchmark measures the kernel, not the scheduler (§19.6 P0.2, "
                f"§11.6). Route it to a kernel backend instead of re-labelling it"
            )
        if not isinstance(self.concurrency, int) or isinstance(self.concurrency, bool):
            raise ValueError("concurrency: expected an int")
        check = check_regime_admissible(self.concurrency)
        if not check.passed:
            raise ValueError(f"concurrency: {'; '.join(check.reasons)}")
        if not isinstance(self.warmup_s, (int, float)) or self.warmup_s < 0:
            raise ValueError("warmup_s: expected a non-negative number")


@dataclass(frozen=True)
class TaskRateCell:
    """The serving throughput cell. Its metric is fixed at construction.

    Not a parameter with a task_rate default: a default is something a caller can
    change, and `MEASUREMENT.md:25-26` forbids substituting tokens/s here in
    either direction.
    """

    value: float
    raw_samples_ref: str
    paired_blocks: int
    metric: str = METRIC
    direction: str = METRIC_DIRECTION
    unit: str = "tasks/s"

    def __post_init__(self) -> None:
        _require_str(self.raw_samples_ref, "raw_samples_ref")
        if self.metric != METRIC or self.direction != METRIC_DIRECTION:
            raise MetricSubstitutionRefused(
                f"a TaskRateCell is {METRIC}/{METRIC_DIRECTION}; got "
                f"{self.metric!r}/{self.direction!r}. tokens/s and task_rate are "
                f"authoritative in their own scopes and neither substitutes for the "
                f"other (MEASUREMENT.md:23-30)"
            )
        if not isinstance(self.value, (int, float)) or self.value <= 0:
            raise ValueError("value: expected a positive number")
        if not isinstance(self.paired_blocks, int) or self.paired_blocks <= 0:
            raise ValueError("paired_blocks: expected a positive int")


@dataclass(frozen=True)
class LatencyCell:
    """A latency cell — a first-class output, lower-better."""

    name: str
    value_ms: float
    raw_samples_ref: str
    direction: str = "lower_better"

    def __post_init__(self) -> None:
        _require_str(self.name, "name")
        _require_str(self.raw_samples_ref, "raw_samples_ref")
        if self.direction != "lower_better":
            raise ValueError(
                f"direction: a latency cell is lower_better, got {self.direction!r}"
            )
        if not isinstance(self.value_ms, (int, float)) or self.value_ms < 0:
            raise ValueError("value_ms: expected a non-negative number")


@dataclass(frozen=True)
class SloCell:
    """An SLO attainment cell — first-class, and stated before it is measured."""

    slo_id: str
    target_description: str
    attainment: float
    window_s: float
    raw_samples_ref: str

    def __post_init__(self) -> None:
        _require_str(self.slo_id, "slo_id")
        _require_str(self.target_description, "target_description")
        _require_str(self.raw_samples_ref, "raw_samples_ref")
        if not isinstance(self.attainment, (int, float)) or not 0.0 <= self.attainment <= 1.0:
            raise ValueError("attainment: expected a fraction in [0, 1]")
        if not isinstance(self.window_s, (int, float)) or self.window_s <= 0:
            raise ValueError("window_s: expected a positive number")


@dataclass(frozen=True)
class PinnedProductionConfig:
    """The comparison anchor: the pinned production configuration, not a memory.

    §11.6: *"the comparison is against the pinned production configuration at its
    production-optimal recipe"*. It carries the kernel binary and linkage digests
    because those must be IDENTICAL across arms — a serving comparison in which
    the kernel also moved measures neither (`check_no_kernel_artifact_change`).

    §10.5's rule applied to this path: the incumbent is archived, not merely
    rebuildable. For a kernel that means a preserved binary; for serving it means
    the RENDERED configuration — regenerating it from a drifted registry does not
    reproduce what was running, exactly as rebuilding an old commit under a
    drifted toolchain does not reproduce the incumbent binary.
    """

    config_id: str
    config_sha256: str
    engine: str
    roles: tuple
    kernel_binary_sha256: str
    kernel_linkage_sha256: str
    pinned_at: str
    pinned_bench_receipt: str
    archive_path: str
    archive_sha256: str

    def __post_init__(self) -> None:
        for name in ("config_id", "engine", "pinned_at", "pinned_bench_receipt",
                     "archive_path"):
            _require_str(getattr(self, name), name)
        for name in ("config_sha256", "kernel_binary_sha256", "kernel_linkage_sha256",
                     "archive_sha256"):
            _require_sha256(getattr(self, name), name)
        object.__setattr__(self, "roles",
                           _require_tuple(self.roles, "roles", non_empty=True))


#: The serving analogue of `ANCHOR_MOVED` (AK-D22): a comparison whose pinned
#: configuration has moved has a denominator that no longer exists.
PINNED_CONFIG_MOVED = "PINNED_CONFIG_MOVED"


def check_regime_admissible(concurrency: Any) -> schemas.Check:
    """Every positive concurrency is admissible. This function exists to say so.

    AK-D37: AK-D36 excludes a *target*, not a *regime*. Single-stream and batched
    are both legitimate, and a batch count is never a reason to reject a
    direction. The check is therefore only that the number is a usable positive
    integer — and it is written as a named function so a future reader looking
    for "where do we cap concurrency" finds the answer *nowhere*, deliberately.
    """
    if not isinstance(concurrency, int) or isinstance(concurrency, bool):
        return schemas.Check(schemas.FAIL, ("concurrency must be an int",))
    if concurrency < 1:
        return schemas.Check(schemas.FAIL, ("concurrency must be >= 1",))
    return schemas.Check(schemas.PASS)


def check_metric_discipline(claim_grammar: Mapping) -> schemas.Check:
    """Route metric commensurability through the one implementation of the rule.

    `schemas.check_metric_commensurability()` already encodes
    `MEASUREMENT.md:23-30` for every backend. Re-deriving it here would give the
    project two implementations of one rule, and the half that drifts is
    whichever one has fewer tests.
    """
    return schemas.check_metric_commensurability(BACKEND, claim_grammar)


@dataclass(frozen=True)
class ServingEvidence:
    """One serving cell's evidence bundle.

    Three refusals live in `__post_init__`, and each is a defect this backend
    would otherwise ship:

      * a token-rate metric (`MetricSubstitutionRefused`);
      * a fixed-shape workload (`FixedShapeWorkloadRefused`, via the spec);
      * throughput with no latency or SLO cell. A scheduler can raise task_rate
        by starving the tail, and a record with no latency cell cannot say so.
        §11.6 makes latency and SLO *"first-class outputs rather than
        secondary"*, which in a data contract means required.
    """

    campaign_id: str
    candidate_id: str
    workload: VariableArrivalReplaySpec
    task_rate: TaskRateCell
    latency: tuple
    slo: tuple
    comparison_config_id: str
    comparison_config_sha256: str
    engine: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "candidate_id", "comparison_config_id", "engine"):
            _require_str(getattr(self, name), name)
        _require_sha256(self.comparison_config_sha256, "comparison_config_sha256")
        if not isinstance(self.workload, VariableArrivalReplaySpec):
            raise TypeError("workload: expected a VariableArrivalReplaySpec")
        if not isinstance(self.task_rate, TaskRateCell):
            raise TypeError("task_rate: expected a TaskRateCell")
        object.__setattr__(self, "latency",
                           _require_tuple(self.latency, "latency",
                                          item_type=LatencyCell))
        object.__setattr__(self, "slo",
                           _require_tuple(self.slo, "slo", item_type=SloCell))
        if not self.latency:
            raise ValueError(
                "latency: a serving cell with no latency cell cannot express the "
                "regression it is most likely to cause — §11.6 makes latency and SLO "
                "first-class outputs, not optional companions"
            )
        if not self.slo:
            raise ValueError(
                "slo: a serving cell with no SLO cell reports throughput against no "
                "declared service objective (§11.6)"
            )

    def claim_grammar(self, *, protocol_id: str, reps: int,
                      attestation_ref: str) -> dict:
        """The claim-grammar block for this cell, with the metric fixed.

        `category=CANDIDATE` per `MEASUREMENT.md:85-95`: this is search/release
        evidence about a candidate configuration, never an OPTIMUM headline.
        """
        _require_str(protocol_id, "protocol_id")
        _require_str(attestation_ref, "attestation_ref")
        if _protocol_family(protocol_id) in KERNEL_PROTOCOL_IDS:
            raise MetricSubstitutionRefused(
                f"protocol {protocol_id!r} governs tokens/s kernel cells; a "
                f"{METRIC} cell cannot cite it (MEASUREMENT.md:23-30)"
            )
        if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
            raise ValueError("reps: expected a positive int; zero reps is not a "
                             "measurement")
        return {
            "category": "CANDIDATE",
            "protocol_id": protocol_id,
            "metric": METRIC,
            "metric_direction": METRIC_DIRECTION,
            "reps": reps,
            "attestation_ref": attestation_ref,
        }


def check_comparison_anchor(
    evidence: ServingEvidence, pinned: PinnedProductionConfig,
) -> schemas.Check:
    """FAIL unless the comparison arm IS the pinned production configuration.

    Two ways this fails and both matter: comparing against some other config
    (an off-recipe or convenience arm — invariant 15 makes those diagnostic and
    never release-justifying), and comparing against a config whose content hash
    has moved since it was pinned, which is `PINNED_CONFIG_MOVED`: the ratio's
    denominator no longer exists (AK-D22's shape, applied to a configuration
    instead of a binary).
    """
    if evidence.comparison_config_id != pinned.config_id:
        return schemas.Check(schemas.FAIL, (
            f"comparison arm is {evidence.comparison_config_id!r}, not the pinned "
            f"production configuration {pinned.config_id!r} (§11.6)",
        ))
    if evidence.comparison_config_sha256 != pinned.config_sha256:
        return schemas.Check(schemas.FAIL, (
            f"{PINNED_CONFIG_MOVED}: the comparison cites "
            f"{evidence.comparison_config_sha256[:12]} but the pinned configuration "
            f"is {pinned.config_sha256[:12]}; the denominator this ratio was "
            f"measured against no longer exists",
        ))
    if evidence.engine != pinned.engine:
        return schemas.Check(schemas.FAIL, (
            f"comparison engine {evidence.engine!r} is not the production engine "
            f"{pinned.engine!r}. {AK_D36_NOTE}",
        ))
    missing = [role for role in evidence.workload.trace.roles if role not in pinned.roles]
    if missing:
        return schemas.Check(schemas.FAIL, (
            f"replay drives roles the pinned configuration does not serve: {missing}",
        ))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Objective admission (§1.6, AK-D36/AK-D37)
# =============================================================================

def admit_change_class(change_class: str, *, underlying: Optional[str] = None) -> str:
    """Admit a change class to this backend, derived — not enumerated by hand.

    `schemas.CHANGE_CLASS_CHEAP_SUITE` already maps every change class to its
    cheap suite (§9.5), and exactly one class's suite is
    `variable_arrival_replay`. Deriving admission from that mapping means a new
    serving change class becomes admissible by declaring its suite, and a kernel
    change class can never become admissible by being added to a list here.

    `oracle_port` is the one indirection the vocabulary already has: its suite is
    *"underlying_change_class_suite"*, so it is admitted only with an underlying
    class that is itself admissible.
    """
    if change_class not in schemas.CHANGE_CLASSES:
        raise ValueError(
            f"change_class: {change_class!r} is not one of "
            f"{sorted(schemas.CHANGE_CLASSES)}"
        )
    suite = schemas.CHANGE_CLASS_CHEAP_SUITE[change_class]
    if suite == "underlying_change_class_suite":
        if underlying is None:
            raise ValueError(
                f"change_class {change_class!r} defers to an underlying class "
                f"(§9.5); name it so admission can be decided"
            )
        admit_change_class(underlying)
        return change_class
    if suite != WORKLOAD_CLASS:
        raise ValueError(
            f"change_class {change_class!r} has cheap suite {suite!r}, which is a "
            f"fixed-shape kernel suite; {BACKEND} measures {WORKLOAD_CLASS!r}. This "
            f"change belongs to a kernel backend (§13.1-§13.2), not to the "
            f"stack-change path"
        )
    return change_class


@dataclass(frozen=True)
class CrossEngineAnalysisView:
    """A cross-engine comparison, reportable and non-gating by construction.

    §1.6 permits cross-scope roll-ups as *"a labelled analysis view. They never
    gate."* AK-D36 is what happens when one becomes an objective instead. `gates`
    is not a field a caller sets — construction refuses any value but False, so
    the view cannot be handed to a decision as though it were evidence.
    """

    label: str
    incumbent_engine: str
    comparator_engine: str
    concurrency: int
    observed_ratio: float
    note: str = AK_D36_NOTE
    gates: bool = False

    def __post_init__(self) -> None:
        for name in ("label", "incumbent_engine", "comparator_engine"):
            _require_str(getattr(self, name), name)
        if self.gates is not False:
            raise CrossEngineRatioObjectiveRefused(
                f"a cross-engine analysis view never gates. {AK_D36_NOTE}"
            )
        if not isinstance(self.observed_ratio, (int, float)):
            raise ValueError("observed_ratio: expected a number")
        check = check_regime_admissible(self.concurrency)
        if not check.passed:
            raise ValueError(f"concurrency: {'; '.join(check.reasons)}")


def admit_objective(objective: Mapping, *, pinned: PinnedProductionConfig) -> schemas.Check:
    """Admit (or refuse) a serving campaign objective.

    Refuses exactly three things, and deliberately nothing about batch size:

      * a token-rate metric (`MEASUREMENT.md:25-26`);
      * a comparison arm on a different engine — that is the whole-stack ratio
        AK-D36 forbids as a target. It stays reportable as a
        `CrossEngineAnalysisView`;
      * an objective that gates on a non-production recipe class (invariant 15).

    Concurrency is read only to confirm it is a usable positive integer.
    AK-D37: *"The constraint is on the metric, never on the batch regime."*
    """
    if not isinstance(objective, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("objective is not a mapping",))
    reasons: list = []

    metric = objective.get("metric")
    if not isinstance(metric, str) or not metric:
        return schemas.Check(schemas.COULD_NOT_CHECK, ("objective.metric is absent",))
    grammar_check = check_metric_discipline({"metric": metric})
    if grammar_check.outcome == schemas.FAIL:
        reasons.extend(grammar_check.reasons)
    elif metric != METRIC:
        # The shared rule is a SUBSTITUTION detector: it recognises the token-rate
        # spellings it knows (`decode_tokens_s`) and passes everything else. So
        # `tok/s` and `t/s` — the spellings this project's own bench records and
        # llama-bench output actually use — clear it, and so does any metric at
        # all. Delegating the substitution question is right; inferring "therefore
        # admissible" from a non-FAIL is not. This backend's metric is fixed at
        # module level and `TaskRateCell` fixes it at construction; an objective is
        # held to the same rule rather than to a weaker one.
        reasons.append(
            f"objective metric {metric!r} is not {METRIC!r}; {BACKEND} reports "
            f"{METRIC} and nothing else (MEASUREMENT.md:23-30, §11.6). A metric the "
            f"shared commensurability rule does not recognise as a token rate is "
            f"still not this backend's metric"
        )

    protocol = objective.get("protocol_id")
    if isinstance(protocol, str) and _protocol_family(protocol) in KERNEL_PROTOCOL_IDS:
        reasons.append(
            f"objective cites {protocol!r}, a protocol whose decision rule is "
            f"written over tokens/s kernel cells"
        )

    arms = objective.get("comparison_arms")
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)) or not arms:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("objective.comparison_arms is absent or empty; an "
                              "objective with no named comparison arm has no anchor",))
    for i, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            reasons.append(f"comparison_arms[{i}]: expected a mapping")
            continue
        engine = arm.get("engine")
        if engine != pinned.engine:
            reasons.append(
                f"comparison_arms[{i}]: engine {engine!r} is not the production "
                f"engine {pinned.engine!r}. {AK_D36_NOTE} Report it as a "
                f"CrossEngineAnalysisView instead"
            )

    recipe_class = objective.get("recipe_class")
    if recipe_class is not None and recipe_class not in schemas.RECIPE_CLASSES:
        reasons.append(
            f"recipe_class {recipe_class!r} is not production-optimal; baseline and "
            f"off-recipe cells are diagnostic and never justify a release "
            f"(invariant 15)"
        )

    concurrency_levels = objective.get("concurrency_levels")
    if concurrency_levels is not None:
        if not isinstance(concurrency_levels, Sequence) or isinstance(
                concurrency_levels, (str, bytes)):
            reasons.append("concurrency_levels: expected a sequence")
        else:
            for level in concurrency_levels:
                check = check_regime_admissible(level)
                if not check.passed:
                    reasons.append(f"concurrency_levels: {level!r} — "
                                   f"{'; '.join(check.reasons)}")

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


# =============================================================================
# T2 — the composed-champion estimator (§9.7)
# =============================================================================

@dataclass(frozen=True)
class ComposedServingEstimate:
    """A T2 estimate of a composed serving champion.

    §9.7: T2 *"runs on the composed champion, never by adding local
    percentages"*. Interaction is the dominant uncertainty at composition time —
    two scheduler changes that each help alone can fight — so the composition is
    measured as one thing, in one window, against the pinned production
    configuration.
    """

    composition_id: str
    member_candidate_ids: tuple
    evidence: ServingEvidence
    measured_as_whole: bool
    window_ref: str

    def __post_init__(self) -> None:
        _require_str(self.composition_id, "composition_id")
        _require_str(self.window_ref, "window_ref")
        object.__setattr__(
            self, "member_candidate_ids",
            _require_tuple(self.member_candidate_ids, "member_candidate_ids",
                           non_empty=True),
        )
        if len(self.member_candidate_ids) < 2:
            raise ValueError(
                "member_candidate_ids: a composition has at least two members; one "
                "member is a candidate, and T2's subject is interaction"
            )
        if len(set(self.member_candidate_ids)) != len(self.member_candidate_ids):
            raise ValueError("member_candidate_ids: contains duplicates")
        if not isinstance(self.evidence, ServingEvidence):
            raise TypeError("evidence: expected a ServingEvidence")
        if self.measured_as_whole is not True:
            raise AdditiveCompositionRefused(
                "a composed estimate must be MEASURED as a whole (§9.7). An estimate "
                "assembled from per-member effects is not a T2 result: interaction is "
                "exactly what T2 is for, and addition assumes it away"
            )
        if len(self.evidence.workload.trace.roles) < 2 and len(self.evidence.slo) < 2:
            raise ValueError(
                "a T2 serving composition needs breadth: more than one role in the "
                "replay, or more than one SLO cell. §9.7's sentinel matrix is medium "
                "and production-weighted, not a single cell"
            )

    @classmethod
    def from_local_effects(cls, *args, **kwargs):
        """Always raises. Named so the refusal is discoverable where it is sought.

        Someone will eventually look for the convenient way to combine banked
        per-change percentages. This is that place, and it says no.
        """
        raise AdditiveCompositionRefused(
            "there is no additive constructor: §9.7 composes by re-measuring the "
            "composed champion, never by adding local percentages"
        )


# =============================================================================
# Diff-complexity ceiling (§10.6)
# =============================================================================

@dataclass(frozen=True)
class ComplexityCeiling:
    """A backend adapter's declared complexity / blast-radius ceiling (§10.6)."""

    max_diff_lines: int
    max_files_touched: int
    shared_core_permitted: bool
    rationale: str

    def __post_init__(self) -> None:
        _require_str(self.rationale, "rationale")
        for name in ("max_diff_lines", "max_files_touched"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name}: expected a positive int")


#: The serving adapter's ceiling. Above it the package is MARKED for human review
#: — never truncated, never rejected, and never silently accepted. §10.6's point
#: is that LLM-authored change should not reach a release package unreviewed at
#: arbitrary size, and the marker is what carries that to the operator's first
#: page.
SERVING_COMPLEXITY_CEILING = ComplexityCeiling(
    max_diff_lines=400,
    max_files_touched=12,
    shared_core_permitted=False,
    rationale=(
        "A serving change is scheduler/admission/KV policy in the orchestrator plus "
        "launcher flags — one conceptual mutation per proposal (invariant 13). A diff "
        "larger than this is either several changes or a rewrite, and either way the "
        "three gates cannot attribute what moved. Shared ggml core is never permitted "
        "on this path at any size: a core change reaches every op in both the CPU and "
        "GPU builds (AK-D30) and owes a kernel freeze, not a stack change."
    ),
)


@dataclass(frozen=True)
class BlastRadius:
    """The classification of one candidate diff against the ceiling."""

    diff_lines: int
    files_touched: int
    touches_shared_core: bool
    requires_human_code_review: bool
    reasons: tuple = ()


def classify_blast_radius(
    *, diff_lines: int, files_touched: int, touches_shared_core: bool,
    ceiling: ComplexityCeiling = SERVING_COMPLEXITY_CEILING,
) -> BlastRadius:
    """Classify a diff. Marks for review; never truncates and never caps.

    A shared-core touch is not "a big diff": it is a kernel change, and
    `KernelChangeMisrouted` is raised rather than marked, because no amount of
    human review makes a stack change the right vehicle for a new ggml core.
    """
    for name, value in (("diff_lines", diff_lines), ("files_touched", files_touched)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name}: expected a non-negative int")
    if not isinstance(touches_shared_core, bool):
        raise ValueError("touches_shared_core: expected a bool")

    if touches_shared_core and not ceiling.shared_core_permitted:
        raise KernelChangeMisrouted(
            f"this diff touches shared ggml core, which reaches every op in both the "
            f"CPU and GPU builds (AK-D30). It owes a kernel freeze under §10, not the "
            f"{RELEASE_PATH} path. {ceiling.rationale}"
        )

    reasons: list = []
    if diff_lines > ceiling.max_diff_lines:
        reasons.append(
            f"diff is {diff_lines} lines, above the declared ceiling of "
            f"{ceiling.max_diff_lines}"
        )
    if files_touched > ceiling.max_files_touched:
        reasons.append(
            f"diff touches {files_touched} files, above the declared ceiling of "
            f"{ceiling.max_files_touched}"
        )
    return BlastRadius(
        diff_lines=diff_lines, files_touched=files_touched,
        touches_shared_core=touches_shared_core,
        requires_human_code_review=bool(reasons), reasons=tuple(reasons),
    )


# =============================================================================
# Rollback, reload request, and the stack-change package
# =============================================================================

@dataclass(frozen=True)
class RollbackPlan:
    """The rollback target for a stack change: the archived rendered config.

    §10.5's rule, transposed. For a kernel the incumbent binary is archived
    because rebuilding an old commit under a drifted toolchain does not reproduce
    it. For serving, regenerating the incumbent configuration from a registry
    that has moved does not reproduce what was serving traffic — so the RENDERED
    configuration is archived, and the plan names it by hash.
    """

    incumbent_config_id: str
    incumbent_config_sha256: str
    archive_path: str
    archive_sha256: str
    archive_verified: bool
    restore_command: str

    def __post_init__(self) -> None:
        for name in ("incumbent_config_id", "archive_path", "restore_command"):
            _require_str(getattr(self, name), name)
        for name in ("incumbent_config_sha256", "archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if self.archive_verified is not True:
            raise ValueError(
                "archive_verified: an unverified archive is not a rollback target — "
                "§11.2 requires the packager to verify the archive target, and a "
                "rollback discovered to be missing at rollback time is no rollback"
            )
        scan = scan_for_kernel_freeze_actions(
            {"restore_command": self.restore_command}
        )
        if not scan.passed:
            raise KernelFreezePathRefused(
                "rollback restore_command would perform a human-only production "
                f"write: {'; '.join(scan.reasons)}"
            )


@dataclass(frozen=True)
class ReloadRequest:
    """A REQUEST to reload the stack, routed to whoever owns the inference.

    §11.3 / `OPERATING_CONSTRAINTS.md:41`: a reload *"must be executed BY THAT
    SESSION, at a moment it chooses"*. This object is a record, never an action —
    nothing in this module can execute it, and the AST audit proves it.

    `gate_basis` is fixed to the pinned bench. AutoPilot being down is a fact
    about a consumer, and it has been mistaken for the gate
    (`feedback_stack_reload_checks_cpu_bench_not_autopilot`); offering it here as
    an alternative basis would re-open that.
    """

    request_id: str
    owning_session: str
    services: tuple
    gate_basis: str
    pinned_bench_receipt: str
    three_gate_status: str
    notes: tuple = ()

    def __post_init__(self) -> None:
        for name in ("request_id", "owning_session", "pinned_bench_receipt"):
            _require_str(getattr(self, name), name)
        object.__setattr__(self, "services",
                           _require_tuple(self.services, "services", non_empty=True))
        object.__setattr__(self, "notes", _require_tuple(self.notes, "notes"))
        if self.gate_basis != RELOAD_GATE_BASIS:
            raise ReloadGateBasisRefused(
                f"gate_basis {self.gate_basis!r} is not {RELOAD_GATE_BASIS!r}: a stack "
                f"reload gates on the pinned bench. AutoPilot being down is not the "
                f"gate (feedback_stack_reload_checks_cpu_bench_not_autopilot)"
            )
        if self.three_gate_status != schemas.PASS:
            raise ServingAdapterError(
                f"a reload request may only be raised on a three-gate PASS; got "
                f"{self.three_gate_status!r}"
            )


def build_reload_request(
    *, request_id: str, owning_session: str, services: Sequence,
    pinned: PinnedProductionConfig, gates: ThreeGateResult,
    gate_basis: str = RELOAD_GATE_BASIS, autopilot_state: Optional[str] = None,
) -> ReloadRequest:
    """Draft the reload request. Drafting is the whole of what happens here.

    `autopilot_state` is accepted and recorded as a NOTE precisely so it is
    visible that it did not participate in the decision.
    """
    gates.require_release_eligible()
    notes = []
    if autopilot_state is not None:
        notes.append(
            f"autopilot state at drafting: {autopilot_state} — recorded, not a gate"
        )
    return ReloadRequest(
        request_id=request_id,
        owning_session=owning_session,
        services=tuple(services),
        gate_basis=gate_basis,
        pinned_bench_receipt=pinned.pinned_bench_receipt,
        three_gate_status=gates.status,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class StackChangePackage:
    """What the serving adapter hands a human. NOT a kernel release package.

    The id prefix is `aks-` where a kernel release package is `akr-`
    (`schemas.validate_release_package`), so the two can never be confused by a
    reader or a glob. `schemas.validate_release_package()` would reject this
    object anyway — it requires a `source_tree` from `schemas.SOURCE_TREES`, and
    `serving_runtime` deliberately has none (§1.5) — but relying on an incidental
    rejection is not a design.

    The verdict is RE-DERIVED in `__post_init__` from the gates and the waivers,
    so no path can attach PASS to a package whose gates do not support it.
    """

    package_id: str
    campaign_id: str
    verdict: str
    gates: ThreeGateResult
    evidence: ServingEvidence
    pinned: PinnedProductionConfig
    rollback: RollbackPlan
    blast_radius: BlastRadius
    operator_command_sequence: tuple
    active_waivers: tuple = ()
    suppressed_claims: tuple = ()
    requires_human_code_review: bool = False
    created_at: str = ""
    notes: tuple = ()
    schema: str = SCHEMA_STACK_CHANGE_PACKAGE
    release_path: str = RELEASE_PATH

    def __post_init__(self) -> None:
        _require_str(self.campaign_id, "campaign_id")
        if _parse_instant(self.created_at, "created_at") is None:
            raise ValueError(
                f"created_at: {self.created_at!r} is not an ISO-8601 timestamp with a "
                f"timezone offset; a naive timestamp on a shared host is ambiguous"
            )
        if not self.package_id.startswith("aks-"):
            raise ValueError(
                f"package_id: {self.package_id!r} must start with 'aks-' — 'akr-' is "
                f"the kernel release package, and the two paths must not share a "
                f"namespace"
            )
        if self.release_path != RELEASE_PATH:
            raise KernelFreezePathRefused(
                f"release_path {self.release_path!r}: this package is "
                f"{RELEASE_PATH!r} and cannot declare another"
            )
        if self.verdict not in STACK_CHANGE_VERDICTS:
            raise ValueError(
                f"verdict: {self.verdict!r} is not one of "
                f"{sorted(STACK_CHANGE_VERDICTS)}"
            )
        for name, kind in (("gates", ThreeGateResult), ("evidence", ServingEvidence),
                           ("pinned", PinnedProductionConfig),
                           ("rollback", RollbackPlan), ("blast_radius", BlastRadius)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name}: expected a {kind.__name__}")

        # The anchor check is re-run HERE and not only in
        # `assemble_stack_change_package()`. This class is exported, constructible,
        # and is the durable artifact; a refusal that lives only in one of the two
        # constructors is a refusal a caller passes by using the other one. Both
        # fields it needs are the package's own, so there is no reason for it to be
        # the assembler's private business (invariant 15: a comparison against
        # anything but the pinned production configuration never justifies a
        # release, whichever door the package came through).
        anchor = check_comparison_anchor(self.evidence, self.pinned)
        if not anchor.passed:
            raise ServingAdapterError(
                "package evidence is not anchored to its own pinned production "
                "configuration: " + "; ".join(anchor.reasons)
            )
        object.__setattr__(self, "notes", _require_tuple(self.notes, "notes"))
        object.__setattr__(
            self, "suppressed_claims",
            _require_tuple(self.suppressed_claims, "suppressed_claims"),
        )
        waivers = _require_tuple(self.active_waivers, "active_waivers",
                                 item_type=Mapping)
        object.__setattr__(self, "active_waivers", waivers)

        for i, waiver in enumerate(waivers):
            violations = schemas.validate_operator_waiver(waiver)
            if violations:
                raise ValueError(
                    f"active_waivers[{i}]: not a valid "
                    f"{schemas.SCHEMA_OPERATOR_WAIVER}: {violations}"
                )
            # §10.4 binds a waiver to its campaign. A waiver from another campaign
            # is not a looser waiver, it is a different question's answer.
            if waiver.get("campaign_id") != self.campaign_id:
                raise ValueError(
                    f"active_waivers[{i}]: bound to campaign "
                    f"{waiver.get('campaign_id')!r}, not this package's "
                    f"{self.campaign_id!r}"
                )

        derived = _derive_package_verdict(self.gates, waivers)
        if derived != self.verdict:
            raise PackageTampering(
                f"package verdict {self.verdict!r} does not follow from its gates "
                f"(status {self.gates.status!r}) and {len(waivers)} waiver(s); "
                f"derived {derived!r}"
            )
        if waivers and not self.suppressed_claims:
            raise ValueError(
                "suppressed_claims: a waived cell suppresses the corresponding claim "
                "in the release receipt (§10.4); a package pinning a waiver and "
                "suppressing nothing still makes the claim it waived"
            )
        if not waivers and self.suppressed_claims:
            raise ValueError(
                "suppressed_claims: nothing is waived, so nothing is suppressed"
            )
        if self.blast_radius.requires_human_code_review and not self.requires_human_code_review:
            raise ValueError(
                "requires_human_code_review: must be true when the diff is above the "
                "declared complexity ceiling (§10.6) — the marker is what puts it on "
                "the operator's first page"
            )

        commands = _require_tuple(self.operator_command_sequence,
                                 "operator_command_sequence", item_type=Mapping,
                                 non_empty=True)
        object.__setattr__(self, "operator_command_sequence", commands)
        for i, entry in enumerate(commands):
            prefix = f"operator_command_sequence[{i}]"
            _require_str(entry.get("command"), f"{prefix}.command")
            _require_str(entry.get("validation_receipt"), f"{prefix}.validation_receipt")
            if entry.get("validated") is not True:
                raise ValueError(
                    f"{prefix}.validated: every operator command is pre-validated "
                    f"end-to-end before it is handed over (MEASUREMENT.md:138-145)"
                )

        # Two scans, deliberately different in reach (see
        # `scan_for_kernel_freeze_actions`): the COMMAND SURFACE is scanned for
        # command strings, because that is what a human would execute; the whole
        # body is scanned for declared action ids, because that is where a
        # forbidden action can be declared. Scanning the whole body for command
        # strings would refuse to report a failing gate whose reason merely
        # quotes a production path.
        command_surface = {
            "operator_command_sequence": [dict(c) for c in commands],
            "restore_command": self.rollback.restore_command,
        }
        for scan, label in (
            (scan_for_kernel_freeze_actions(command_surface), "command surface"),
            (scan_for_kernel_freeze_actions(self.to_record(),
                                            match_command_strings=False),
             "declared actions"),
        ):
            if not scan.passed:
                raise KernelFreezePathRefused(
                    f"stack-change package would perform a human-only production "
                    f"write ({label}): " + "; ".join(scan.reasons)
                )
        authority = schemas.find_authority_flavoured_keys(self.to_record())
        if authority:
            raise ValueError(
                f"authority-flavoured keys in a machine-authored package: {authority} "
                f"(§1.3 — AutoKernel holds no authority to declare)"
            )

    def to_record(self) -> dict:
        """A JSON-able record of the package. No schema id is registered for it.

        `schemas.SCHEMA_REGISTRY` is not extended from here: registering a schema
        is an edit to the data-contract module, which is a different change with
        a different review. The schema string is carried so a future registration
        has something to bind to, and `schemas.canonical_json()` will refuse the
        record outright if it is not canonicalisable.
        """
        return {
            "schema": self.schema,
            "package_id": self.package_id,
            "campaign_id": self.campaign_id,
            # The record named the campaign but not the candidate the change IS,
            # so nothing in the durable artifact said which candidate produced its
            # numbers.
            "candidate_id": self.evidence.candidate_id,
            "backend": BACKEND,
            "release_path": self.release_path,
            "adapter": ADAPTER_ID,
            "verdict": self.verdict,
            "gates": {
                name: (None if outcome is None else {
                    "gate": outcome.gate,
                    "status": outcome.status,
                    "evidence_kind": outcome.evidence_kind,
                    "evidence_ref": outcome.evidence_ref,
                    "reasons": list(outcome.reasons),
                    "notes": list(outcome.notes),
                })
                for name, outcome in zip(GATE_ORDER, self.gates.gates)
            },
            "gate_status": self.gates.status,
            "gate_blocked_at": self.gates.blocked_at,
            "metric": METRIC,
            "metric_direction": METRIC_DIRECTION,
            "benchmark_class": self.evidence.workload.benchmark_class,
            "workload_class": WORKLOAD_CLASS,
            "task_rate": self.evidence.task_rate.value,
            # Every cell carries its raw-samples pointer, and the workload block
            # carries the instrument's identity. Without them the package records
            # a NUMBER whose instrument and samples cannot be recovered from the
            # record — durable evidence is the pointer, not the digest of a
            # summary (MEASUREMENT: "evidence must be durable, not merely
            # hashed"), and §10.5's archived-incumbent rule is the same idea
            # applied to the other arm.
            "task_rate_cell": {
                "value": self.evidence.task_rate.value,
                "unit": self.evidence.task_rate.unit,
                "metric": self.evidence.task_rate.metric,
                "direction": self.evidence.task_rate.direction,
                "paired_blocks": self.evidence.task_rate.paired_blocks,
                "raw_samples_ref": self.evidence.task_rate.raw_samples_ref,
            },
            "workload": {
                "spec_id": self.evidence.workload.spec_id,
                "benchmark_class": self.evidence.workload.benchmark_class,
                "concurrency": self.evidence.workload.concurrency,
                "warmup_s": self.evidence.workload.warmup_s,
                "recipe_constructor_id": self.evidence.workload.recipe_constructor_id,
                "recipe_sha256": self.evidence.workload.recipe_sha256,
                "trace": {
                    "trace_id": self.evidence.workload.trace.trace_id,
                    "trace_sha256": self.evidence.workload.trace.trace_sha256,
                    "seed": self.evidence.workload.trace.seed,
                    "request_count": self.evidence.workload.trace.request_count,
                    "duration_s": self.evidence.workload.trace.duration_s,
                    "roles": list(self.evidence.workload.trace.roles),
                },
            },
            "comparison": {
                "engine": self.evidence.engine,
                "config_id": self.evidence.comparison_config_id,
                "config_sha256": self.evidence.comparison_config_sha256,
            },
            "latency_cells": [
                {"name": c.name, "value_ms": c.value_ms, "direction": c.direction,
                 "raw_samples_ref": c.raw_samples_ref}
                for c in self.evidence.latency
            ],
            "slo_cells": [
                {"slo_id": c.slo_id, "attainment": c.attainment,
                 "target": c.target_description, "window_s": c.window_s,
                 "raw_samples_ref": c.raw_samples_ref}
                for c in self.evidence.slo
            ],
            "pinned_configuration": {
                "config_id": self.pinned.config_id,
                "config_sha256": self.pinned.config_sha256,
                "engine": self.pinned.engine,
                "pinned_bench_receipt": self.pinned.pinned_bench_receipt,
                "archive_path": self.pinned.archive_path,
                "archive_sha256": self.pinned.archive_sha256,
            },
            "rollback": {
                "incumbent_config_id": self.rollback.incumbent_config_id,
                "incumbent_config_sha256": self.rollback.incumbent_config_sha256,
                "archive_path": self.rollback.archive_path,
                "archive_sha256": self.rollback.archive_sha256,
                "archive_verified": self.rollback.archive_verified,
                "restore_command": self.rollback.restore_command,
            },
            "diff_complexity": {
                "diff_lines": self.blast_radius.diff_lines,
                "files_touched": self.blast_radius.files_touched,
                "touches_shared_core": self.blast_radius.touches_shared_core,
            },
            "requires_human_code_review": self.requires_human_code_review,
            "active_waivers": [dict(w) for w in self.active_waivers],
            "suppressed_claims": list(self.suppressed_claims),
            "operator_command_sequence": [dict(c) for c in self.operator_command_sequence],
            "does_not_touch": [
                "kernel version", "frozen production branch", "kernel-speed era row",
                "AutoPilot baseline",
            ],
            "created_at": self.created_at,
            "notes": list(self.notes),
        }


def _derive_package_verdict(gates: ThreeGateResult, waivers: Sequence) -> str:
    """PASS / PASS_WITH_WAIVER / FAIL — and a waiver NEVER rescues a gate.

    §10.4's waiver waives an evidence *cell* and forfeits the claim that cell
    would have supported. The three gates are not cells: gate 2 failing means the
    stack did not come up, and gate 3 failing means production is not running
    what anybody configured. There is no claim to forfeit in exchange for those,
    so a waiver alongside a failed gate leaves the verdict FAIL.
    """
    if gates.status == schemas.PASS:
        return "PASS_WITH_WAIVER" if waivers else "PASS"
    return "FAIL"


def assemble_stack_change_package(
    *,
    package_id: str,
    campaign_id: str,
    gates: ThreeGateResult,
    evidence: ServingEvidence,
    pinned: PinnedProductionConfig,
    rollback: RollbackPlan,
    blast_radius: BlastRadius,
    operator_command_sequence: Sequence,
    candidate_binary_sha256: str,
    candidate_linkage_sha256: str,
    created_at: str,
    active_waivers: Sequence = (),
    suppressed_claims: Sequence = (),
    notes: Sequence = (),
) -> StackChangePackage:
    """Assemble the package, refusing every route back to the kernel path.

    Order matters. The kernel-artifact check runs FIRST, because a candidate that
    changed the kernel binary is not a stack change with a problem — it is a
    different kind of change on the wrong path, and every later check would be
    answering the wrong question about it.
    """
    artifact_check = check_no_kernel_artifact_change(
        pinned=pinned,
        candidate_binary_sha256=candidate_binary_sha256,
        candidate_linkage_sha256=candidate_linkage_sha256,
    )
    if artifact_check.outcome == schemas.FAIL:
        raise KernelChangeMisrouted(
            "; ".join(artifact_check.reasons)
            + f" — this change owes a kernel freeze (§10), not the {RELEASE_PATH} path"
        )
    if artifact_check.outcome == schemas.COULD_NOT_CHECK:
        raise ServingAdapterError(
            "; ".join(artifact_check.reasons)
            + " — a package cannot be assembled while it is unknown whether the "
              "kernel moved; an unknown identity is not an unchanged one"
        )

    anchor_check = check_comparison_anchor(evidence, pinned)
    if not anchor_check.passed:
        raise ServingAdapterError(
            "comparison anchor is not the pinned production configuration: "
            + "; ".join(anchor_check.reasons)
        )

    verdict = _derive_package_verdict(gates, tuple(active_waivers))
    return StackChangePackage(
        package_id=package_id,
        campaign_id=campaign_id,
        verdict=verdict,
        gates=gates,
        evidence=evidence,
        pinned=pinned,
        rollback=rollback,
        blast_radius=blast_radius,
        operator_command_sequence=tuple(operator_command_sequence),
        active_waivers=tuple(active_waivers),
        suppressed_claims=tuple(suppressed_claims),
        requires_human_code_review=blast_radius.requires_human_code_review,
        created_at=created_at,
        notes=tuple(notes),
    )


# =============================================================================
# Journal record
# =============================================================================

def build_serving_evaluation_event(
    *,
    event_id: str,
    candidate_id: str,
    tier: str,
    evidence: ServingEvidence,
    pinned: PinnedProductionConfig,
    protocol_id: str,
    reps: int,
    attestation_ref: str,
    evaluator_id: str,
    evaluator_bundle_sha256: str,
    artifact_source_sha256: str,
    anchor_source_commit: str,
    anchor_measurement_event_ids: Sequence,
    scope_manifest_sha256: str,
    host_receipt: str,
    resource_claim_receipt: str,
    scope_denominator: Mapping,
    correctness: Mapping,
    quality: Mapping,
    stability: Mapping,
    mechanism: Mapping,
    performance: Mapping,
    determinism: Mapping,
    status: str,
    created_at: str,
    integrity_flags: Sequence = (),
    supersedes: Sequence = (),
    co_residency: Optional[str] = None,
) -> dict:
    """Build a v3 evaluation event for a serving cell, and REFUSE an invalid one.

    Two adapter-specific bindings sit on top of the shared contract:

      * **the anchor's binary and linkage are the PINNED PRODUCTION KERNEL's.**
        They are identical across arms by construction — that is what makes the
        comparison a scheduler comparison — and recording them is what lets a
        later reader prove the kernel did not move underneath the result;
      * **co-residency is derived, not asserted.** A replay driving more than one
        role is co-resident by definition; declaring it `single` would let a
        multi-role cell be compared against a single-role one
        (`feedback_benchmark_methodology`).

    The event is validated before it is returned and a violation RAISES. An
    invalid record that reaches the journal is worse than no record: it looks
    like evidence to every downstream reader.
    """
    claim = evidence.claim_grammar(protocol_id=protocol_id, reps=reps,
                                  attestation_ref=attestation_ref)
    metric_check = check_metric_discipline(claim)
    if metric_check.outcome != schemas.PASS:
        raise MetricSubstitutionRefused(
            f"claim grammar failed metric commensurability: {metric_check.reasons}"
        )
    anchor_check = check_comparison_anchor(evidence, pinned)
    if not anchor_check.passed:
        raise ServingAdapterError(
            "event's comparison arm is not the pinned production configuration: "
            + "; ".join(anchor_check.reasons)
        )

    # The `performance` block arrives from the caller and the `evidence` argument
    # never reaches it, so the two can describe different measurements while the
    # claim grammar vouches for one of them. `paired_blocks` is the binding that
    # admits no second reading: both counts are the number of paired blocks of
    # THIS cell, so a disagreement means the block and the cell are not the same
    # measurement, and the event would carry a task_rate claim over numbers the
    # cell never produced.
    if isinstance(performance, Mapping) and "paired_blocks" in performance:
        declared_blocks = performance.get("paired_blocks")
        if declared_blocks != evidence.task_rate.paired_blocks:
            raise ServingAdapterError(
                f"performance.paired_blocks {declared_blocks!r} contradicts the "
                f"task_rate cell's {evidence.task_rate.paired_blocks!r}; the "
                f"performance block and the evidence cell must be the same "
                f"measurement, and an event whose numbers came from somewhere its "
                f"claim grammar does not describe is not evidence"
            )

    roles = evidence.workload.trace.roles
    derived_co_residency = (
        "single" if len(roles) == 1 else f"co_resident:{evidence.workload.spec_id}"
    )
    if co_residency is not None and co_residency != derived_co_residency:
        raise ServingAdapterError(
            f"co_residency {co_residency!r} contradicts the replay, which drives "
            f"{len(roles)} role(s) and is therefore {derived_co_residency!r}"
        )

    event = {
        "schema": schemas.SCHEMA_EVALUATION_EVENT,
        "event_id": event_id,
        "campaign_id": evidence.campaign_id,
        "candidate_id": candidate_id,
        "tier": tier,
        "claim_grammar": claim,
        "evaluator": {"id": evaluator_id, "bundle_sha256": evaluator_bundle_sha256},
        "artifact": {
            "source_sha256": artifact_source_sha256,
            # Identical to the anchor's by construction: a serving change that
            # moved the kernel is `KernelChangeMisrouted`, not a serving cell.
            "binary_sha256": pinned.kernel_binary_sha256,
            "linkage_sha256": pinned.kernel_linkage_sha256,
        },
        "anchor": {
            "source_commit": anchor_source_commit,
            "binary_sha256": pinned.kernel_binary_sha256,
            "linkage_sha256": pinned.kernel_linkage_sha256,
            "measurement_event_ids": list(anchor_measurement_event_ids),
        },
        "scope_manifest_sha256": scope_manifest_sha256,
        "host_receipt": host_receipt,
        "resource_claim_receipt": resource_claim_receipt,
        "co_residency": derived_co_residency,
        "correctness": dict(correctness),
        "quality": dict(quality),
        "stability": dict(stability),
        "mechanism": dict(mechanism),
        "scope_denominator": dict(scope_denominator),
        "determinism": dict(determinism),
        "performance": dict(performance),
        "integrity_flags": list(integrity_flags),
        "status": status,
        "supersedes": list(supersedes),
        "created_at": created_at,
    }
    violations = schemas.validate_evaluation_event(event)
    if violations:
        raise ServingAdapterError(
            f"refusing to emit an invalid evaluation event: {violations}"
        )
    return event


# =============================================================================
# Self-audit — the no-process, no-write property, checked from the AST
# =============================================================================

_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__",
                                   "input"})

_FORBIDDEN_CALL_ATTRS = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "flush", "fsync",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "rmtree", "rename", "chmod",
    "chown", "utime", "symlink", "link", "touch", "move", "copy", "copyfile", "copytree",
    "system", "popen", "Popen", "run", "call", "spawnv", "fork", "kill", "killpg",
    "send_signal", "terminate", "check_call", "check_output", "communicate", "setxattr",
    # `open` as a BARE name is caught below, but `Path(p).open("w")` is an
    # attribute call and was not — the idiomatic write door stood open. The
    # pathlib link verbs are spelled `_to`, and `move_stable_kernel_symlink` is
    # item one on `FORBIDDEN_PRODUCTION_ACTIONS`: an audit that names `symlink`
    # and not `symlink_to` cannot see the exact call that performs it.
    "open", "symlink_to", "hardlink_to", "lchmod", "mknod", "dump",
    "execv", "execve", "execvp", "execvpe", "posix_spawn", "startfile",
})

#: NOT listed, deliberately: `replace`. `Path.replace()` is a rename, but
#: `str.replace()` is everywhere, and an audit that fires on ordinary string work
#: gets relaxed by the next person who trips it. A rename here would also have to
#: pass a Path through, which needs one of the listed verbs to have produced a
#: file first.

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "shlex", "asyncio",
    # Dynamic import is the way back to every module above: `__import__` was
    # already refused by name, `importlib.import_module` was not.
    "importlib", "posix", "pickle", "runpy",
})


#: Module-level names the audited source MUST define for the result to be ABOUT this
#: module. Same doctrine as the two sibling adapters: without a binding of this kind
#: the audit is a property of whatever string it was handed.
_AUDIT_IDENTITY_NAMES = (
    "release_path_for", "refuse_kernel_freeze", "scan_for_kernel_freeze_actions",
)


def _source_is_this_module(tree: Any) -> bool:
    """True when the parsed AST is recognisably THIS adapter's source."""
    backend = None
    defined = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "BACKEND"
                        and isinstance(node.value, ast.Constant)):
                    backend = node.value.value
    return backend == BACKEND and set(_AUDIT_IDENTITY_NAMES) <= defined


def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that it cannot write, spawn, or signal.

    §11.6 is a release path, and a release path that could start the stack itself
    would take the reload out of the hands of the session that owns the inference
    (`OPERATING_CONSTRAINTS.md:41`). Gate 1's guard runs behind a caller-supplied
    `GuardRunner`; gates 2 and 3 consume observations. That separation is only
    real if it is checked, so this parses the module and FAILs on a write-capable
    call, a process call, or an import that would grant either.

    COULD_NOT_CHECK when the source cannot be read or parsed: an unreadable
    module is not an audited one — and likewise when the supplied source is not
    THIS module's. The no-argument call anchors itself on `Path(__file__)`, but the
    `source=` seam does not, so a clean unrelated module (its two sibling adapters,
    for instance) returned PASS and read as a clean bill of health for this one.
    `whisper_stt` and `qwentts_tts` already bind their audits to their own module
    identity; this is the same guarantee in the same plane and it was the only one
    of the three left unbound. A FAIL is still returned unbound, because a forbidden
    construct is a finding about the text whoever the text belongs to.
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
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"could not parse module: {exc}",))
    if not tree.body:
        # This audit is passed by AUDITING a module, never by handing it nothing
        # to audit: an empty source contains no forbidden call, and reporting
        # PASS for it is the check certifying its own absence.
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("source parsed to an empty module; there was nothing to audit, which "
             "is not the same as auditing something and finding it clean",),
        )

    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                findings.append(f"line {node.lineno}: calls {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALL_ATTRS:
                findings.append(f"line {node.lineno}: calls .{func.attr}()")

    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    if not _source_is_this_module(tree):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the supplied source does not define this module's identity (BACKEND = "
            f"{BACKEND!r} plus {list(_AUDIT_IDENTITY_NAMES)}), so the AST audited is not "
            f"this adapter's. A clean audit of text nobody bound to the module is not "
            f"evidence about the module",))
    return schemas.Check(schemas.PASS)
