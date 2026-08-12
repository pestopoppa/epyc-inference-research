#!/usr/bin/env python3
"""plan.py — the release-plan compiler (AK5, §10.1, §3.2).

WHY THIS MODULE EXISTS
----------------------
Before a human may freeze a kernel, one question has to be answered without a
human answering it: **which cells must show no regression, on which backends,
under which protocol, protecting which roles?** A curated matrix is the wrong
shape for that answer — it goes stale the moment a role is repointed, and the v8
freeze script's hard-coded artifact list is named in §10.3 as something not to
reuse. The answer is *derived*: the cells that matter for backend B are the
production cells whose roles resolve to B in the compiled stack priors, joined to
the reconciled affected surface and to what the build system says actually
changed.

The seed for this compiler is
`epyc-orchestrator/scripts/validate/kernel_freeze_scope.py`, and its argument is
correct. Its **implementation** has four defects this module exists not to
inherit, each of which silently *shrinks* a freeze's evidence:

  1. **An unknown binary path is classified `cpu`.** `_backend_of()` tests three
     substrings and then `return "cpu"`. A role served by a binary nobody
     recognises therefore lands in the CPU gate, where the wrong protocol judges
     it. Here classification is a longest-prefix match at a path-component
     boundary against roots the caller *declares*, and no match is
     `ROLE_BINARY_UNCLASSIFIED` — a third outcome, never a default. (Component
     boundaries are not decoration: `…/llama.cpp/build` is a string prefix of
     `…/llama.cpp/build-hip/bin/llama-server`, so substring logic maps the whole
     HIP fleet onto the CPU kernel.)
  2. **Three silent `continue`s.** A role with no `serving`, no `launch`, or no
     `binary_path` vanishes from the scope with no record. Here every live role
     that cannot be planned is retained as an `UnplannableRole` carrying its
     reason, and every role excluded for not being live is retained with its
     deployment status. A role can leave the matrix; it cannot leave without a
     receipt.
  3. **Deduplication by printed basename.** The seed's `seen_models` set is keyed
     on `model_path.split("/")[-1]`, which both over-merges (two different files
     with one basename) and under-merges (one file reached by two paths). Here
     the dedup key is the full measurement identity — backend, phase, model
     identity, quant, context, KV, ubatch, speculation, concurrency, placement,
     co-residency, recipe class — and a cell records **which roles it protects**,
     so collapsing four roles onto one server's cell loses no protection
     (`feedback_model_not_role_indexing`, `feedback_same_model_roles_share_server`).
  4. **No backend-unchanged test at all.** CPU and GPU share one tree and one
     frozen branch (§1.5), so a CPU-only champion still produces a new binary on
     both paths whenever it touches shared ggml core. Without the test, a freeze
     either buys a full GPU matrix it may not need or drops GPU with no receipt.

THE BACKEND-UNCHANGED TEST (§3.2), AND WHY THE NAIVE FORM IS WORSE THAN NONE
----------------------------------------------------------------------------
A backend owes candidate-grade evidence **only if its binary changed**. The
tempting test — hash the two binaries — never fires: ROCm/llama.cpp builds embed
build ids, timestamps and absolute paths, so a fresh build is essentially never
byte-identical to one made months earlier somewhere else. A test that never fires
reads as "always changed", which is the expensive answer, and the moment someone
"fixes" it by comparing loosely it becomes the unsafe one.

The two stages are implemented in `evaluator/surface.py` and this module does not
re-implement them; it **consumes** them and decides what happens to the cells:

  * stage 1 (`backend_unchanged_stage1_source_closure`) — the gate: the build
    system's own generated dependency closure for the backend's link targets,
    diffed over `production_base..candidate`, plus toolchain identity;
  * stage 2 (`backend_unchanged_stage2_normalized_binary`) — the confirmation:
    normalized `.text`/`.rodata`/`.data.rel.ro`/dynsym comparison against a
    rebuild of the base **in the candidate's build environment**, so both sides
    share one non-determinism regime;
  * `backend_unchanged()` combines them, and a **disagreement is a hard finding
    filed against build identity**, never a silent preference for the cheaper
    answer.

This module adds the three release-side conditions the evaluator cannot know:

  * cells drop only when `may_drop_cells` is true **and** the incumbent evidence
    that is meant to transfer is actually named, with hashes, in a
    `TransferReceipt`. "Unchanged" plus nothing to transfer is not a transfer;
  * cells drop only when the reconciled affected surface (§6.4) is `PASS`. An
    unreconciled or escaped surface may not narrow anything, because narrowing on
    an unconfirmed manifest is exactly the actor-controls-its-own-scope failure
    invariant 18 exists to prevent; and
  * a backend the **dispatch trace observed executing** may not simultaneously be
    declared binary-unchanged. That contradiction (`TRACED_BACKEND_DECLARED_UNCHANGED`)
    is a FAIL, and it is the one an over-narrow closure would produce.

WHAT THIS MODULE IS NOT
-----------------------
It compiles a plan. It does not run one, seal a candidate, judge a waiver, or
assemble a transaction. It emits no argv: there is no T3 recipe family in
`evaluator/recipes.py`, and hand-typed argv voids a run, so a cell carries its
recipe *dimensions* and the constructor binding it will need — never a command
line.

**It never freezes and never cuts over.** It performs no write of any kind: no
file, no process, no branch, no symlink, no era row, no baseline. That is proved
from this module's own AST by `audit_plan_module_is_read_only()`, which delegates
to the auditor `surface.py` already uses rather than forking a second one.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §1.5, §1.6,
§3.2, §6.4, §10.1–§10.6, §11.6, §13.1–§13.5, invariants 2, 3, 6, 15, 18.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import schemas
from ..evaluator import surface

# =============================================================================
# Identity of the artifact this module produces
# =============================================================================

#: The plan is a versioned artifact in its own right: `schemas.validate_release_package`
#: requires a non-empty `release_plan` block, and a reader must be able to tell which
#: compiler shape it is looking at. The version is part of the name, never metadata
#: beside it (schemas.py CONVENTIONS).
PLAN_SCHEMA = "epyc.autokernel.release_plan.v1"

#: The compiler's own identity, recorded in the plan so a bundle names the code that
#: derived its scope. Bumped when the derivation changes, not when a comment does.
COMPILER_ID = "autokernel.release.plan/v1"

#: Invariant 15: baseline / off-recipe cells are diagnostic and never veto or justify
#: a release, so every cell this compiler emits is at the production-optimal recipe.
#: There is deliberately no parameter to relax this.
RECIPE_CLASS = "production_optimal"

#: A role is in release scope when the compiled priors say it is actually serving.
#: `benchmark_or_candidate` roles are real rows in the same file and are NOT protected
#: cells; they are recorded as out-of-scope with their status rather than dropped.
LIVE_DEPLOYMENT_STATUSES = frozenset({"live_stack"})

#: §11.6 / §13.5 / AK-D9, AK-D23: scheduler, KV-policy, batching and admission work
#: releases through the three-gate stack-change path, measured in `task_rate` under
#: variable arrival. It has no source tree, no frozen branch and no kernel era row.
#: The adapter *"MUST refuse the kernel-freeze path outright rather than degrading to
#: it"* — so this compiler raises on it instead of compiling a degenerate plan.
STACK_CHANGE_BACKEND = "serving_runtime"

#: Frozen production branches (CLAUDE.md, invariant 3). `schemas` carries the same
#: pattern for candidate/champion branch names; `test_plan.py` asserts the two agree
#: rather than trusting that they do.
PRODUCTION_BRANCH_RE = re.compile(r"^production-(consolidated|speech)-v\d+$")

#: `single` or `co_resident:<group>`, the vocabulary `schemas._CO_RESIDENCY_RE`
#: already accepts on an evaluation event's scope denominator. Restated as a constant
#: so the two never drift apart unnoticed (test_plan.py checks the produced labels
#: against the schema's own regex).
CO_RESIDENCY_SINGLE = "single"
CO_RESIDENCY_PREFIX = "co_resident:"


# =============================================================================
# Errors — wiring and authority defects RAISE; facts about the release are findings
# =============================================================================

class ReleasePlanError(Exception):
    """Base class for every refusal this module makes."""


class PlanInputError(ReleasePlanError):
    """An input is malformed, or two inputs are about different things.

    Distinct from a finding on purpose: a finding is a statement about the release
    under compilation, and this is a statement about the *caller*. Combining a
    stage-1 diff taken over one base with a stage-2 comparison taken against another
    is not a property of the candidate; it is a wiring bug, and folding it into the
    plan's check would let it be waived.
    """


class ProductionWriteRefused(ReleasePlanError):
    """The target would build in, commit to, or evaluate from a production tree.

    Invariant 3: *"Frozen means immutable. No actor builds in or modifies any
    production tree."* CLAUDE.md is blunter — production kernels *"must NEVER be
    modified, rebased, built, or committed to"*. This is a refusal, not a finding.
    """


class KernelFreezePathRefused(ReleasePlanError):
    """`serving_runtime` was routed at the kernel-freeze path (§11.6, §13.5)."""


# =============================================================================
# Findings — every one has a FIXED severity and a FIXED outcome
#
# The plan's overall check is DERIVED from its findings; it is never stamped. That
# is the same rule `evaluator/api.compute_verdict` enforces for a search verdict,
# for the same reason: a status a caller can set is a status a caller can set
# wrongly.
# =============================================================================

SEVERITY_BLOCKING = "blocking"
SEVERITY_ADVISORY = "advisory"
SEVERITIES = (SEVERITY_BLOCKING, SEVERITY_ADVISORY)

F_ROLE_BINARY_UNCLASSIFIED = "ROLE_BINARY_UNCLASSIFIED"
F_ROLE_BINARY_AMBIGUOUS = "ROLE_BINARY_AMBIGUOUS"
F_ROLE_RECIPE_INCOMPLETE = "ROLE_RECIPE_INCOMPLETE"
F_CAPACITY_FLOOR_INCOMPLETE = "CAPACITY_FLOOR_INCOMPLETE"
F_RELEASE_PROTOCOL_UNDEFINED = "RELEASE_PROTOCOL_UNDEFINED"
F_PHASE_THRESHOLDS_UNDECLARED = "PHASE_THRESHOLDS_UNDECLARED"
F_UNCOVERED_AFFECTED_OP = "UNCOVERED_AFFECTED_OP"
F_CORESIDENT_CELL_MISSING = "CORESIDENT_CELL_MISSING"
F_BACKEND_UNCHANGED_TEST_NOT_RUN = "BACKEND_UNCHANGED_TEST_NOT_RUN"
F_BUILD_IDENTITY_STAGE_DISAGREEMENT = "BUILD_IDENTITY_STAGE_DISAGREEMENT"
F_TRACED_BACKEND_DECLARED_UNCHANGED = "TRACED_BACKEND_DECLARED_UNCHANGED"
F_TRANSFER_RECEIPT_INCOMPLETE = "TRANSFER_RECEIPT_INCOMPLETE"
F_SURFACE_ESCAPE = "SURFACE_ESCAPE"
F_SURFACE_UNRECONCILED = "SURFACE_UNRECONCILED"
F_STABLE_PATH_RECEIPT_MISSING = "STABLE_PATH_RECEIPT_MISSING"
F_STABLE_PATH_NOT_IN_PRODUCTION_TREE = "STABLE_PATH_NOT_IN_PRODUCTION_TREE"
F_LINKAGE_REQUIREMENT_UNPROVEN = "LINKAGE_REQUIREMENT_UNPROVEN"
F_NO_PROTECTED_CELLS = "NO_PROTECTED_CELLS"
F_QUALITY_TRANSFER_REFUSED = "QUALITY_TRANSFER_REFUSED"
F_HOST_CAPACITY_BUDGET_UNDECLARED = "HOST_CAPACITY_BUDGET_UNDECLARED"
F_DIFF_COMPLEXITY_CEILING_EXCEEDED = "DIFF_COMPLEXITY_CEILING_EXCEEDED"
F_MODEL_IDENTITY_BY_PATH_ONLY = "MODEL_IDENTITY_BY_PATH_ONLY"
F_BACKEND_UNCHANGED_RESULT_UNANCHORED = "BACKEND_UNCHANGED_RESULT_UNANCHORED"
F_SINGLE_BACKEND_NOOP_CANDIDATE = "SINGLE_BACKEND_NOOP_CANDIDATE"

#: code -> (severity, outcome, what it means).
#:
#: The outcome column is the whole point. `FAIL` is reserved for a CONTRADICTION —
#: two instruments that cannot both be right. Everything that is merely a *gap* is
#: `COULD_NOT_CHECK`, because inability to evaluate is a third outcome and reporting
#: it as FAIL would put the blame on the candidate for a hole in the instrument
#: (house rule; `schemas.Check`).
FINDING_SPEC: Mapping[str, tuple] = {
    F_ROLE_BINARY_UNCLASSIFIED: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "a live role's serving binary resolves under no declared backend root, so the "
        "protocol that would judge it is unknown; the seed defaulted such a role to CPU"),
    F_ROLE_BINARY_AMBIGUOUS: (
        SEVERITY_BLOCKING, schemas.FAIL,
        "a live role's serving binary resolves under two backends' declared roots; the "
        "bindings contradict each other"),
    F_ROLE_RECIPE_INCOMPLETE: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "a live role's compiled prior is missing a field the production-optimal recipe "
        "needs, so no cell can be keyed for it"),
    F_CAPACITY_FLOOR_INCOMPLETE: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "a cell's capacity floor could not be derived from the incumbent's declared "
        "footprint, so §10.2 phase 7 has no fixed floor to hold the candidate to"),
    F_RELEASE_PROTOCOL_UNDEFINED: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "the backend binding declares no release protocol for a phase it serves; the "
        "compiler does not invent one (§13.3/§13.4: STT and TTS protocols are 'to be "
        "defined')"),
    F_PHASE_THRESHOLDS_UNDECLARED: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "a phase names its release protocol but carries no thresholds; a band supplied "
        "by this compiler would be a literal nobody ratified"),
    F_UNCOVERED_AFFECTED_OP: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "an op inside the reconciled affected surface has no observed op/shape coverage, "
        "so §10.2 phase 3 cannot exercise it"),
    F_CORESIDENT_CELL_MISSING: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "the backend requires a co-resident cell (§13.2, §9.7, §10.2 phase 4) but the "
        "compiled lineup shows a single resident server, so none could be derived"),
    F_BACKEND_UNCHANGED_TEST_NOT_RUN: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "no §3.2 result was supplied for a backend the tree serves; absence of the test "
        "is not a pass, so the backend keeps its full matrix"),
    F_BACKEND_UNCHANGED_RESULT_UNANCHORED: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "the §3.2 result offered as grounds for dropping cells does not record the "
        "commits its stages were taken over, so it cannot be tied to this release's base "
        "and candidate; `surface` documents a null commit as 'the caller built this "
        "result by hand and the cross-check cannot run', and cannot-run is not passed"),
    F_BUILD_IDENTITY_STAGE_DISAGREEMENT: (
        SEVERITY_BLOCKING, schemas.FAIL,
        "§3.2's two stages disagree: the closure is wrong or the build is "
        "non-deterministic. Filed against build identity, and the backend owes full "
        "evidence"),
    F_TRACED_BACKEND_DECLARED_UNCHANGED: (
        SEVERITY_BLOCKING, schemas.FAIL,
        "the dispatch trace observed this backend executing candidate kernels while the "
        "§3.2 test declared its binary unchanged; both cannot be true"),
    F_TRANSFER_RECEIPT_INCOMPLETE: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "cells may drop only 'with a transfer receipt naming the incumbent artifacts and "
        "their hashes' (§10.2 phase 1); no such incumbent evidence was supplied"),
    F_SURFACE_ESCAPE: (
        SEVERITY_BLOCKING, schemas.FAIL,
        "the reconciled affected surface reports `traced ⊄ derived` — a hard candidate "
        "failure (invariant 18). No cell may be narrowed on an escaped surface"),
    F_SURFACE_UNRECONCILED: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "the affected surface was not reconciled to PASS, so it may not narrow release "
        "scope; the full matrix stands"),
    F_STABLE_PATH_RECEIPT_MISSING: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "no receipt records what the stable production kernel path currently resolves "
        "to, so the incumbent whose evidence would transfer is unidentified"),
    F_STABLE_PATH_NOT_IN_PRODUCTION_TREE: (
        SEVERITY_BLOCKING, schemas.FAIL,
        "the stable production kernel path resolves outside the production tree the "
        "release names; the incumbent is not what the plan assumes"),
    F_LINKAGE_REQUIREMENT_UNPROVEN: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "a cell's launcher does not carry the per-tree LD_LIBRARY_PATH its binding "
        "requires; three ggml generations are live and a binary that inherits another "
        "tree's ggml runs silently wrong (CLAUDE.md speech-kernel freeze)"),
    F_NO_PROTECTED_CELLS: (
        SEVERITY_BLOCKING, schemas.COULD_NOT_CHECK,
        "a backend the tree serves has neither protected cells nor a transfer receipt; "
        "an empty matrix passes vacuously"),
    F_QUALITY_TRANSFER_REFUSED: (
        SEVERITY_ADVISORY, schemas.PASS,
        "a quality-transfer claim was supplied but does not hold, so the cell keeps its "
        "quality measurement (more work, not less — advisory)"),
    F_HOST_CAPACITY_BUDGET_UNDECLARED: (
        SEVERITY_ADVISORY, schemas.PASS,
        "no host/device capacity budget was declared, so the co-resident group's summed "
        "residency could not be checked against one"),
    F_DIFF_COMPLEXITY_CEILING_EXCEEDED: (
        SEVERITY_ADVISORY, schemas.PASS,
        "the diff exceeds the backend's declared complexity/blast-radius ceiling (§10.6); "
        "the package is marked REQUIRES_HUMAN_CODE_REVIEW on its first page"),
    F_MODEL_IDENTITY_BY_PATH_ONLY: (
        SEVERITY_ADVISORY, schemas.PASS,
        "a model was identified by declared path only; two paths reaching one GGUF (the "
        "lmstudio compat symlink farm) will not dedupe, which over-measures rather than "
        "under-measures"),
    F_SINGLE_BACKEND_NOOP_CANDIDATE: (
        SEVERITY_BLOCKING, schemas.FAIL,
        "a source tree serving exactly one backend cannot produce a release candidate "
        "while that backend is unchanged; there is no sibling backend whose change the "
        "candidate could contain"),
}

FINDING_CODES = tuple(sorted(FINDING_SPEC))

#: Marker the package's first page carries (§10.6).
REQUIRES_HUMAN_CODE_REVIEW = "REQUIRES_HUMAN_CODE_REVIEW"


@dataclass(frozen=True)
class PlanFinding:
    """One statement about this release, with a severity and outcome it cannot pick.

    `severity` and `outcome` are looked up from `FINDING_SPEC` by code and cannot be
    passed in: a finding that could choose its own severity is a finding that can be
    downgraded at the point it is inconvenient.
    """

    code: str
    detail: str
    backend: Optional[str] = None
    filed_against: str = "release_plan"

    def __post_init__(self) -> None:
        if self.code not in FINDING_SPEC:
            raise PlanInputError(
                f"finding code {self.code!r} is not one of {list(FINDING_CODES)}")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise PlanInputError(f"finding {self.code} must carry a detail string")
        if self.backend is not None and self.backend not in schemas.BACKENDS:
            raise PlanInputError(
                f"finding backend {self.backend!r} is not one of {sorted(schemas.BACKENDS)}")

    @property
    def severity(self) -> str:
        return FINDING_SPEC[self.code][0]

    @property
    def outcome(self) -> str:
        return FINDING_SPEC[self.code][1]

    @property
    def gating(self) -> bool:
        return self.severity == SEVERITY_BLOCKING

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "outcome": self.outcome,
                "backend": self.backend, "detail": self.detail,
                "filed_against": self.filed_against,
                "meaning": FINDING_SPEC[self.code][2]}


# =============================================================================
# Small helpers
# =============================================================================

_OUTCOME_RANK = {schemas.PASS: 0, schemas.COULD_NOT_CHECK: 1, schemas.FAIL: 2}


def worst_check(*checks: schemas.Check) -> schemas.Check:
    """Worst-of over checks, FAIL > COULD_NOT_CHECK > PASS, reasons preserved.

    `surface.py` has the same reducer as a private helper. It is re-derived here
    rather than imported through the private name so this module does not depend on
    another module's underscore surface; `test_plan.py` asserts the two agree on a
    truth table.
    """
    if not checks:
        return schemas.Check(schemas.PASS)
    worst = max(checks, key=lambda c: _OUTCOME_RANK[c.outcome])
    if worst.outcome == schemas.PASS:
        return schemas.Check(schemas.PASS)
    reasons: list = []
    for check in checks:
        if check.outcome == worst.outcome:
            reasons.extend(check.reasons)
    return schemas.Check(worst.outcome, tuple(reasons))


def _check_dict(check: schemas.Check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def normalize_path(path: str, *, label: str) -> str:
    """Collapse duplicate separators and trailing slashes on an ABSOLUTE path.

    Relative paths are refused: a backend root or a binary path that depends on a
    working directory cannot be compared across two processes, and the comparison is
    what decides which protocol judges a role.

    `.` and `..` COMPONENTS are refused for a sharper reason. This module performs no
    filesystem I/O by design, so it cannot resolve one — and an unresolved traversal
    breaks `path_is_under()` in BOTH directions at once:

      * `<production_tree>/../llama.cpp-experimental/build` reads as INSIDE the
        production tree, so a legitimate experimental build root is refused; and,
        far worse,
      * `<production_tree>-experimental/../<production_tree>/build` reads as OUTSIDE
        it, so a build root that really does resolve into the FROZEN production tree
        clears `ProductionWriteRefused` (invariant 3), and a stable-path receipt whose
        target really does escape the production tree clears
        `STABLE_PATH_NOT_IN_PRODUCTION_TREE`.

    A declared path carrying a traversal is a wiring defect, not a fact about the
    release, so it raises. Callers that receive paths from compiled priors
    (`extract_role_facts`, `LinkageRequirement.check_declared`) catch this and turn it
    into a record rather than an abort.
    """
    if not isinstance(path, str) or not path.strip():
        raise PlanInputError(f"{label} must be a non-empty string, got {path!r}")
    cleaned = re.sub(r"/{2,}", "/", path.strip())
    if not cleaned.startswith("/"):
        raise PlanInputError(f"{label} must be absolute, got {path!r}")
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    traversal = sorted({c for c in cleaned.split("/") if c in (".", "..")})
    if traversal:
        raise PlanInputError(
            f"{label} contains {traversal} component(s): {path!r}. This module resolves "
            "no path (it performs no I/O), so a traversing path would be compared "
            "unresolved, and containment at a component boundary is what decides both "
            "the production-write refusal and the stable-path receipt. Supply the "
            "resolved absolute path.")
    return cleaned


def path_is_under(root: str, path: str) -> bool:
    """True when `path` equals `root` or lies under it AT A COMPONENT BOUNDARY.

    `str.startswith` is not this function: `/mnt/raid0/llm/llama.cpp/build` is a
    string prefix of `/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server`, so a
    prefix test maps every HIP role onto the CPU kernel. That is the seed's
    substring-classification defect in its most damaging form.
    """
    root = normalize_path(root, label="root")
    path = normalize_path(path, label="path")
    return path == root or path.startswith(root + "/")


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanInputError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _sorted_unique(values: Iterable[str]) -> tuple:
    return tuple(sorted(set(values)))


# =============================================================================
# Declared inputs — every one is supplied, none is guessed
# =============================================================================

@dataclass(frozen=True)
class PhaseProtocol:
    """The owning protocol and thresholds for one (backend, phase) cell class.

    §1.6: each phase is judged under its own protocol — P-BENCH-1 for CPU decode,
    P-BENCH-PREFILL-1 for CPU prefill, P-GPU-1 for MI210 — *"so nothing crosses a
    protocol boundary"*. `thresholds` is supplied by the caller from the protocol
    text and is only ever RECORDED here. This module contains no numeric band: a
    literal in the compiler is a threshold nobody ratified, and
    `measurement/protocols/` is human-amendment-only.
    """

    phase: str
    protocol_id: str
    metric: str
    direction: str
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    threshold_source: Optional[str] = None

    def __post_init__(self) -> None:
        _require_str(self.phase, "PhaseProtocol.phase")
        _require_str(self.protocol_id, "PhaseProtocol.protocol_id")
        _require_str(self.metric, "PhaseProtocol.metric")
        if self.direction not in schemas.METRIC_DIRECTIONS:
            raise PlanInputError(
                f"PhaseProtocol.direction {self.direction!r} is not one of "
                f"{sorted(schemas.METRIC_DIRECTIONS)}")
        if not isinstance(self.thresholds, Mapping):
            raise PlanInputError("PhaseProtocol.thresholds must be a mapping")
        if self.thresholds and self.threshold_source is None:
            raise PlanInputError(
                "PhaseProtocol.threshold_source must name where the thresholds come from; "
                "an unsourced band is indistinguishable from one this compiler invented")

    def to_dict(self) -> dict:
        return {"phase": self.phase, "protocol_id": self.protocol_id,
                "metric": self.metric, "direction": self.direction,
                "thresholds": dict(self.thresholds),
                "threshold_source": self.threshold_source}


@dataclass(frozen=True)
class LinkageRequirement:
    """The per-tree runtime-linkage contract a cell's launcher must satisfy.

    CLAUDE.md: the three production trees run three different ggml generations, so
    *"every launcher must set its own `LD_LIBRARY_PATH`"* and prove it. §10.2 phase 2
    additionally records that the verifier lives in **epyc-inference-research**, not
    epyc-root, which CLAUDE.md cites unqualified — the same defect class as the
    durability validator's path. The resolved location is carried here rather than
    re-derived by every consumer.
    """

    source_tree: str
    ggml_generation: str
    required_ld_library_path: tuple = ()
    verifier: str = "scripts/utils/verify_ggml_linkage.sh"
    verifier_repo: str = "epyc-inference-research"

    def __post_init__(self) -> None:
        if self.source_tree not in schemas.SOURCE_TREES:
            raise PlanInputError(
                f"LinkageRequirement.source_tree {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}")
        _require_str(self.ggml_generation, "LinkageRequirement.ggml_generation")
        if not isinstance(self.required_ld_library_path, tuple):
            raise PlanInputError("LinkageRequirement.required_ld_library_path must be a tuple")
        for entry in self.required_ld_library_path:
            normalize_path(entry, label="LinkageRequirement.required_ld_library_path entry")

    def check_declared(self, declared: Sequence[str]) -> schemas.Check:
        """Does a role's declared `ld_library_path` cover this requirement?"""
        if not self.required_ld_library_path:
            return schemas.Check(schemas.PASS)
        if declared is None:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("the compiled prior records no ld_library_path for this launcher",))
        # A malformed entry is a fact about the compiled priors, so it becomes a
        # COULD_NOT_CHECK on this cell. Letting `normalize_path` raise through here
        # aborted the WHOLE compile on one relative or traversing string in one
        # launcher's environment — everywhere else in this module a data defect is
        # retained as a record, and an abort is the one outcome that leaves no record
        # at all.
        have: set = set()
        malformed: list = []
        for entry in declared:
            if not isinstance(entry, str) or not entry.strip():
                continue
            try:
                have.add(normalize_path(entry, label="declared ld_library_path entry"))
            except PlanInputError as exc:
                malformed.append(str(exc))
        if malformed:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                tuple(f"the launcher's declared ld_library_path is unusable: {m}"
                      for m in malformed))
        missing = [r for r in self.required_ld_library_path
                   if normalize_path(r, label="required entry") not in have]
        if missing:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"launcher does not declare {missing}; {self.source_tree} runs ggml "
                 f"{self.ggml_generation} and an inherited ggml runs silently wrong",))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"source_tree": self.source_tree, "ggml_generation": self.ggml_generation,
                "required_ld_library_path": list(self.required_ld_library_path),
                "verifier": self.verifier, "verifier_repo": self.verifier_repo}


@dataclass(frozen=True)
class ComplexityCeiling:
    """§10.6 — the backend adapter's declared complexity / blast-radius ceiling.

    *"LLM-authored kernel C++/HIP should not reach a release package unreviewed at
    arbitrary size."* Above the ceiling the package is marked
    `REQUIRES_HUMAN_CODE_REVIEW` and says so on its first page. `core_header` is a
    KIND of change, not a size band (AK-D30), so it forces review at any size.
    """

    max_diff_lines: Optional[int] = None
    max_files_touched: Optional[int] = None
    shared_core_requires_review: bool = True

    def __post_init__(self) -> None:
        for name in ("max_diff_lines", "max_files_touched"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise PlanInputError(f"ComplexityCeiling.{name} must be a non-negative int")
        if not isinstance(self.shared_core_requires_review, bool):
            raise PlanInputError("ComplexityCeiling.shared_core_requires_review must be a bool")

    def exceeded_by(self, *, diff_lines: Optional[int], files_touched: Optional[int],
                    touches_shared_core: bool, change_classes: Sequence[str]) -> tuple:
        """Return the reasons this diff exceeds the ceiling (empty when it does not)."""
        reasons: list = []
        if self.max_diff_lines is not None:
            if diff_lines is None:
                reasons.append("diff size is unknown and the ceiling is declared in lines")
            elif diff_lines > self.max_diff_lines:
                reasons.append(f"diff of {diff_lines} lines exceeds the ceiling of "
                               f"{self.max_diff_lines}")
        if self.max_files_touched is not None:
            if files_touched is None:
                reasons.append("files touched is unknown and the ceiling is declared in files")
            elif files_touched > self.max_files_touched:
                reasons.append(f"{files_touched} files touched exceeds the ceiling of "
                               f"{self.max_files_touched}")
        if self.shared_core_requires_review and touches_shared_core:
            reasons.append("the diff touches shared ggml core")
        if "core_header" in tuple(change_classes):
            reasons.append("change class `core_header` is its own risk tier (AK-D30)")
        return tuple(reasons)

    def to_dict(self) -> dict:
        return {"max_diff_lines": self.max_diff_lines,
                "max_files_touched": self.max_files_touched,
                "shared_core_requires_review": self.shared_core_requires_review}


@dataclass(frozen=True)
class BackendBinding:
    """Everything a backend adapter declares about itself (§13.1–§13.5).

    `binary_roots` are the absolute path roots under which this backend's serving
    binaries live — the stable production path plus the build directory it resolves
    into. They are DECLARED, not pattern-matched: §1.5 warns that the `tts` symlink
    points at `build`, not `build/bin`, so *"adapters must not assume uniformity"*,
    and §3.2 forbids a directory-prefix guess for the closure for the same reason.
    """

    backend: str
    stable_production_path: str
    production_tree_path: str
    binary_roots: tuple
    phases: tuple
    protocols: Mapping[str, PhaseProtocol] = field(default_factory=dict)
    prerequisites: Mapping[str, schemas.Check] = field(default_factory=dict)
    linkage: Optional[LinkageRequirement] = None
    ceiling: Optional[ComplexityCeiling] = None
    co_residency_required: bool = False
    host_capacity_budget_gb: Optional[float] = None
    canary_required: bool = True

    def __post_init__(self) -> None:
        if self.backend == STACK_CHANGE_BACKEND:
            raise KernelFreezePathRefused(
                f"{STACK_CHANGE_BACKEND!r} has no source tree and no frozen branch; its "
                "release path is the three-gate stack-change path on stack_change_guard.py "
                "(§11.6, AK-D23). The adapter refuses the kernel-freeze path rather than "
                "degrading to it.")
        if self.backend not in schemas.BACKENDS:
            raise PlanInputError(
                f"BackendBinding.backend {self.backend!r} is not one of "
                f"{sorted(schemas.BACKENDS)}")
        normalize_path(self.stable_production_path, label="stable_production_path")
        normalize_path(self.production_tree_path, label="production_tree_path")
        if not isinstance(self.binary_roots, tuple) or not self.binary_roots:
            raise PlanInputError(
                f"BackendBinding.binary_roots must be a non-empty tuple for "
                f"{self.backend!r}; without one no role can be attributed to it, and "
                "defaulting an unattributed role to a backend is the seed's defect")
        for root in self.binary_roots:
            normalize_path(root, label="BackendBinding.binary_roots entry")
        if not isinstance(self.phases, tuple) or not self.phases:
            raise PlanInputError(f"BackendBinding.phases must be a non-empty tuple for "
                                 f"{self.backend!r}")
        expected = schemas.PHASES_BY_BACKEND.get(self.backend)
        if expected is not None and set(self.phases) != set(expected):
            raise PlanInputError(
                f"backend {self.backend!r} serves phases {sorted(expected)} per §1.6, but "
                f"the binding declares {list(self.phases)}; the per-phase objective is "
                "non-inferiority on BOTH phases plus improvement on at least one, so a "
                "dropped phase silently drops half the objective")
        for phase, protocol in self.protocols.items():
            if not isinstance(protocol, PhaseProtocol):
                raise PlanInputError(
                    f"BackendBinding.protocols[{phase!r}] must be a PhaseProtocol")
            if protocol.phase != phase:
                raise PlanInputError(
                    f"BackendBinding.protocols[{phase!r}] declares phase "
                    f"{protocol.phase!r}")
            if phase not in self.phases:
                raise PlanInputError(
                    f"BackendBinding.protocols names phase {phase!r}, which the binding "
                    f"does not serve")
        if not isinstance(self.prerequisites, Mapping):
            raise PlanInputError("BackendBinding.prerequisites must be a mapping")
        for prerequisite_id, check in self.prerequisites.items():
            _require_str(prerequisite_id, "BackendBinding.prerequisite id")
            if not isinstance(check, schemas.Check):
                raise PlanInputError(
                    f"BackendBinding.prerequisites[{prerequisite_id!r}] must be a Check")
        if self.linkage is not None and not isinstance(self.linkage, LinkageRequirement):
            raise PlanInputError("BackendBinding.linkage must be a LinkageRequirement or None")
        if self.ceiling is not None and not isinstance(self.ceiling, ComplexityCeiling):
            raise PlanInputError("BackendBinding.ceiling must be a ComplexityCeiling or None")
        if self.host_capacity_budget_gb is not None:
            if (not isinstance(self.host_capacity_budget_gb, (int, float))
                    or self.host_capacity_budget_gb <= 0):
                raise PlanInputError(
                    "BackendBinding.host_capacity_budget_gb must be a positive number or None")
        if (self.linkage is not None
                and self.linkage.source_tree != schemas.SOURCE_TREE_BY_BACKEND[self.backend]):
            raise PlanInputError(
                f"backend {self.backend!r} is served by source tree "
                f"{schemas.SOURCE_TREE_BY_BACKEND[self.backend]!r} but its linkage "
                f"requirement names {self.linkage.source_tree!r}")

    @property
    def source_tree(self) -> str:
        return schemas.SOURCE_TREE_BY_BACKEND[self.backend]

    def claims_binary(self, binary_path: str) -> bool:
        return any(path_is_under(root, binary_path) for root in self.binary_roots)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "source_tree": self.source_tree,
            "stable_production_path": normalize_path(self.stable_production_path,
                                                     label="stable_production_path"),
            "production_tree_path": normalize_path(self.production_tree_path,
                                                   label="production_tree_path"),
            "binary_roots": [normalize_path(r, label="binary root") for r in self.binary_roots],
            "phases": list(self.phases),
            "protocols": {p: proto.to_dict() for p, proto in sorted(self.protocols.items())},
            "prerequisites": {
                prerequisite_id: _check_dict(check)
                for prerequisite_id, check in sorted(self.prerequisites.items())},
            "linkage": None if self.linkage is None else self.linkage.to_dict(),
            "ceiling": None if self.ceiling is None else self.ceiling.to_dict(),
            "co_residency_required": self.co_residency_required,
            "host_capacity_budget_gb": self.host_capacity_budget_gb,
            "canary_required": self.canary_required,
        }


@dataclass(frozen=True)
class StablePathReceipt:
    """What a stable production kernel path resolved to when it was read.

    §1.5's table is the map; this is the territory. It is a RECEIPT and not a live
    lookup on purpose: resolving a symlink inside the compiler would be I/O whose
    answer could change between the plan and its execution, and the plan has to be
    reproducible from its inputs.
    """

    backend: str
    stable_path: str
    resolved_target: str
    observed_at: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise PlanInputError(f"StablePathReceipt.backend {self.backend!r} is unknown")
        normalize_path(self.stable_path, label="StablePathReceipt.stable_path")
        normalize_path(self.resolved_target, label="StablePathReceipt.resolved_target")
        _require_str(self.observed_at, "StablePathReceipt.observed_at")

    def check_against(self, binding: BackendBinding) -> schemas.Check:
        if not path_is_under(binding.stable_production_path, self.stable_path):
            return schemas.Check(
                schemas.FAIL,
                (f"receipt is for {self.stable_path!r}, which is not the binding's stable "
                 f"path {binding.stable_production_path!r}",))
        if not path_is_under(binding.production_tree_path, self.resolved_target):
            return schemas.Check(
                schemas.FAIL,
                (f"{self.stable_path!r} resolves to {self.resolved_target!r}, which is "
                 f"outside the production tree {binding.production_tree_path!r}",))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "stable_path": self.stable_path,
                "resolved_target": self.resolved_target, "observed_at": self.observed_at}


@dataclass(frozen=True)
class IncumbentEvidence:
    """The incumbent artifacts whose evidence would transfer if a backend is unchanged.

    §10.5: incumbent builds are ARCHIVED, not merely rebuildable — *"rebuilding an old
    commit under a drifted toolchain does not reproduce that binary"*. So a transfer
    receipt names artifacts and their hashes, and this type refuses to exist without
    at least one.
    """

    backend: str
    era_id: str
    artifacts: tuple  # tuple[tuple[str, str], ...] — (reference, sha256)
    protocol_ids: tuple = ()
    archive_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise PlanInputError(f"IncumbentEvidence.backend {self.backend!r} is unknown")
        _require_str(self.era_id, "IncumbentEvidence.era_id")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise PlanInputError(
                "IncumbentEvidence.artifacts must name at least one (ref, sha256) pair; a "
                "receipt that names nothing transfers nothing (§10.2 phase 1)")
        for entry in self.artifacts:
            if (not isinstance(entry, tuple) or len(entry) != 2
                    or not all(isinstance(x, str) for x in entry)):
                raise PlanInputError(
                    f"IncumbentEvidence.artifacts entries must be (ref, sha256) string "
                    f"pairs, got {entry!r}")
            ref, digest = entry
            _require_str(ref, "IncumbentEvidence artifact ref")
            # `fullmatch`, not `match`: `re.match(r"^...$", x)` also accepts a value
            # with a TRAILING NEWLINE, because `$` matches before a final "\n". A
            # digest read off `sha256sum` output without `.strip()` therefore clears
            # the format check AND `is_placeholder_digest` (whose own regex has the
            # same shape, so `"0"*64 + "\n"` is not recognised as filler). A
            # fabricated hash is indistinguishable from a measured one downstream —
            # which is the exact thing the next branch exists to prevent.
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise PlanInputError(
                    f"incumbent artifact {ref!r} carries {digest!r}, which is not a "
                    "sha256 hex digest")
            if schemas.is_placeholder_digest(digest):
                raise PlanInputError(
                    f"incumbent artifact {ref!r} carries a placeholder digest {digest!r}; "
                    "a fabricated hash is indistinguishable from a measured one to every "
                    "downstream reader")

    def to_dict(self) -> dict:
        return {"backend": self.backend, "era_id": self.era_id,
                "artifacts": [{"ref": r, "sha256": d} for r, d in self.artifacts],
                "protocol_ids": list(self.protocol_ids),
                "archive_path": self.archive_path}


@dataclass(frozen=True)
class OpShapeCoverage:
    """Observed op/shape coverage, per backend, from the correctness corpus.

    §10.2 phase 3 needs *"exact and unseen op shapes"*. An affected op with no
    observed coverage is a hole in the release evidence, and this type is what makes
    the hole visible instead of absent.
    """

    covered: Mapping[str, Mapping[str, tuple]] = field(default_factory=dict)
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.covered, Mapping):
            raise PlanInputError("OpShapeCoverage.covered must be a mapping")
        for backend, ops in self.covered.items():
            if backend not in schemas.BACKENDS:
                raise PlanInputError(f"OpShapeCoverage backend {backend!r} is unknown")
            if not isinstance(ops, Mapping):
                raise PlanInputError(f"OpShapeCoverage.covered[{backend!r}] must be a mapping")
            for op, shapes in ops.items():
                if not isinstance(shapes, tuple):
                    raise PlanInputError(
                        f"OpShapeCoverage.covered[{backend!r}][{op!r}] must be a tuple of "
                        "shape ids")

    def shapes_for(self, backend: str, op_name: str) -> tuple:
        return tuple((self.covered.get(backend) or {}).get(op_name, ()))

    def to_dict(self) -> dict:
        return {"source_ref": self.source_ref,
                "covered": {b: {op: list(shapes) for op, shapes in sorted(ops.items())}
                            for b, ops in sorted(self.covered.items())}}


@dataclass(frozen=True)
class QualityTransfer:
    """§10.2 phase 5 — *"Transfer banked quality across kernel eras once paired parity
    proves transfer."*

    Every field is `Optional[bool]` and `None` means unknown, which yields
    COULD_NOT_CHECK. There is no default of True, for the same reason
    `surface.EvidenceTransferScope` has none: a transfer that defaults to permitted
    removes evidence by omission.
    """

    backend: str
    model_key: str
    paired_parity_proven: Optional[bool] = None
    deterministic_replay_valid: Optional[bool] = None
    era_boundary_crossed: Optional[bool] = None
    evidence_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise PlanInputError(f"QualityTransfer.backend {self.backend!r} is unknown")
        _require_str(self.model_key, "QualityTransfer.model_key")

    def check(self) -> schemas.Check:
        unknown: list = []
        failed: list = []
        if self.paired_parity_proven is None:
            unknown.append("paired parity is unknown")
        elif not self.paired_parity_proven:
            failed.append("paired parity does not hold, so banked quality does not transfer")
        if self.deterministic_replay_valid is None:
            unknown.append("deterministic-replay validity is unknown")
        elif not self.deterministic_replay_valid:
            failed.append("the generation path changed, so saved outputs cannot be rescored")
        if self.era_boundary_crossed is None:
            unknown.append("era_boundary_crossed is unknown")
        elif self.era_boundary_crossed:
            failed.append("an era boundary was crossed, so the banked evidence is a "
                          "different era's")
        if not self.evidence_ref:
            unknown.append("no evidence reference names what would transfer")
        if failed:
            return schemas.Check(schemas.FAIL, tuple(failed))
        if unknown:
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknown))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "model_key": self.model_key,
                "paired_parity_proven": self.paired_parity_proven,
                "deterministic_replay_valid": self.deterministic_replay_valid,
                "era_boundary_crossed": self.era_boundary_crossed,
                "evidence_ref": self.evidence_ref,
                "check": _check_dict(self.check())}


@dataclass(frozen=True)
class ReleaseTarget:
    """What is being compiled: one source tree, one base, one full candidate.

    Invariant 2 (*"Release evidence is produced by the same full candidate that is
    frozen; no promotion-time cherry-pick reconciliation"*) is why this names ONE
    candidate commit for the whole tree rather than a per-backend set.
    """

    source_tree: str
    production_base_commit: str
    candidate_commit: str
    candidate_branch: str
    candidate_build_root: str
    candidate_id: str
    backends: tuple
    change_classes: tuple = ()
    diff_lines: Optional[int] = None
    files_touched: Optional[int] = None
    touches_shared_core: bool = False

    def __post_init__(self) -> None:
        if self.source_tree not in schemas.SOURCE_TREES:
            raise PlanInputError(
                f"ReleaseTarget.source_tree {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}; `{STACK_CHANGE_BACKEND}` has no source "
                "tree and releases through the stack-change path (§11.6)")
        for name in ("production_base_commit", "candidate_commit"):
            value = getattr(self, name)
            # `fullmatch`: `re.match(r"^...$", …)` accepts a trailing newline, and a
            # commit read from `git rev-parse` without `.strip()` would then be
            # compared verbatim against the §3.2 stage commits — agreeing with itself
            # while naming a ref no git command resolves.
            if not re.fullmatch(r"[0-9a-f]{40}", str(value)):
                raise PlanInputError(
                    f"ReleaseTarget.{name} must be a full 40-hex commit, got {value!r}")
        if self.production_base_commit == self.candidate_commit:
            raise PlanInputError(
                "the candidate commit equals the production base; there is nothing to "
                "release, and a plan compiled over an empty diff would report every "
                "backend unchanged")
        _require_str(self.candidate_branch, "ReleaseTarget.candidate_branch")
        _require_str(self.candidate_id, "ReleaseTarget.candidate_id")
        # `match`, deliberately, where every ACCEPTANCE pattern in this module uses
        # `fullmatch`: this is a REFUSAL predicate, so its loose end anchoring errs
        # towards refusing (`"production-consolidated-v8\n"` is still caught).
        if PRODUCTION_BRANCH_RE.match(self.candidate_branch):
            raise ProductionWriteRefused(
                f"candidate branch {self.candidate_branch!r} is a FROZEN production branch. "
                "Invariant 3: no actor builds in or modifies a production tree. AutoKernel "
                "versions past production; it never patches it in place.")
        normalize_path(self.candidate_build_root, label="ReleaseTarget.candidate_build_root")
        if not isinstance(self.backends, tuple) or not self.backends:
            raise PlanInputError("ReleaseTarget.backends must be a non-empty tuple")
        if STACK_CHANGE_BACKEND in self.backends:
            raise KernelFreezePathRefused(
                f"{STACK_CHANGE_BACKEND!r} may not appear in a kernel-freeze release "
                "target: a scheduler win is not a kernel freeze (AK-D9). Its three-gate "
                "path is §11.6, measured in task_rate, and this compiler refuses rather "
                "than degrading to it.")
        for backend in self.backends:
            if backend not in schemas.BACKENDS:
                raise PlanInputError(f"ReleaseTarget backend {backend!r} is unknown")
        # §1.5: *"freeze scope is the union of backends served by the tree, narrowed only
        # by the mechanically derived affected-surface manifest"*. So the scope is not a
        # choice. Omitting `llama_gpu` from a CPU campaign's target would be the cheapest
        # possible scope exploit — the GPU binary still changes, and the freeze would ship
        # it unmeasured with nothing recording that it had been left out. Narrowing has
        # exactly one sanctioned route, the §3.2 test, and it produces a receipt.
        served = frozenset(b for b, tree in schemas.SOURCE_TREE_BY_BACKEND.items()
                           if tree == self.source_tree)
        if frozenset(self.backends) != served:
            raise PlanInputError(
                f"source tree {self.source_tree!r} serves {sorted(served)}, but the target "
                f"names {sorted(self.backends)}. Freeze scope is the union of backends "
                "served by the tree (§1.5); it is narrowed by the §3.2 backend-unchanged "
                "test with a transfer receipt, never by leaving a backend out of the "
                "target.")
        for change_class in self.change_classes:
            if change_class not in schemas.CHANGE_CLASSES:
                raise PlanInputError(f"unknown change class {change_class!r}")
        for name in ("diff_lines", "files_touched"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise PlanInputError(f"ReleaseTarget.{name} must be a non-negative int or None")
        if not isinstance(self.touches_shared_core, bool):
            raise PlanInputError("ReleaseTarget.touches_shared_core must be a bool")

    def to_dict(self) -> dict:
        return {"source_tree": self.source_tree,
                "production_base_commit": self.production_base_commit,
                "candidate_commit": self.candidate_commit,
                "candidate_branch": self.candidate_branch,
                "candidate_build_root": normalize_path(self.candidate_build_root,
                                                       label="candidate_build_root"),
                "candidate_id": self.candidate_id,
                "backends": list(self.backends),
                "change_classes": list(self.change_classes),
                "diff_lines": self.diff_lines,
                "files_touched": self.files_touched,
                "touches_shared_core": self.touches_shared_core}


# =============================================================================
# Role extraction from the compiled stack priors
# =============================================================================

@dataclass(frozen=True)
class ModelIdentity:
    """How a model is identified for deduplication.

    A digest is the real identity. A path is a weaker one: the project runs ONE model
    root with `lmstudio/models` as a compat symlink farm, so two paths can reach one
    GGUF. Path-keying therefore fails to merge them — it OVER-measures, which is the
    safe direction — and the weaker basis is recorded on the cell so a reader knows
    which it got. What it must never do is key on the basename, which merges two
    different files that happen to share a filename.
    """

    model_path: str
    sha256: Optional[str] = None
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        normalize_path(self.model_path, label="ModelIdentity.model_path")
        if self.sha256 is not None:
            # `fullmatch`: see IncumbentEvidence. Here the consequence is sharper still
            # — `key` is the DEDUP key, so two different models both carrying
            # `"0"*64 + "\n"` would collapse into one cell and one of them would stop
            # being measured.
            if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
                raise PlanInputError(
                    f"ModelIdentity.sha256 {self.sha256!r} is not a sha256 hex digest")
            if schemas.is_placeholder_digest(self.sha256):
                raise PlanInputError(
                    "ModelIdentity.sha256 is a placeholder digest; a fabricated model hash "
                    "silently merges two different models into one cell")

    @property
    def basis(self) -> str:
        return "sha256" if self.sha256 else "declared_path"

    @property
    def key(self) -> str:
        if self.sha256:
            return f"sha256:{self.sha256}"
        return f"path:{normalize_path(self.model_path, label='model_path')}"

    def to_dict(self) -> dict:
        return {"model_path": normalize_path(self.model_path, label="model_path"),
                "sha256": self.sha256, "model_id": self.model_id,
                "identity_basis": self.basis, "key": self.key}


@dataclass(frozen=True)
class RoleFacts:
    """One live role's launch entry, flattened out of the compiled stack priors."""

    role: str
    deployment_status: str
    binary_path: str
    model: ModelIdentity
    port: Optional[int]
    quant: Optional[str]
    architecture: Optional[str]
    model_family: Optional[str]
    params_b: Optional[float]
    active_b: Optional[float]
    context_tokens: Optional[int]
    kv_type_k: Optional[str]
    kv_type_v: Optional[str]
    ubatch: Optional[int]
    slots: Optional[int]
    flash_attn: Optional[bool]
    spec_enabled: Optional[bool]
    spec_type: Optional[str]
    draft_max: Optional[int]
    numa_policy: Optional[str]
    numa_instance: Optional[int]
    cpu_shape_class: Optional[str]
    model_mem_gb: Optional[float]
    ld_library_path: tuple = ()

    def to_dict(self) -> dict:
        return {
            "role": self.role, "deployment_status": self.deployment_status,
            "binary_path": self.binary_path, "model": self.model.to_dict(),
            "port": self.port, "quant": self.quant, "architecture": self.architecture,
            "model_family": self.model_family, "params_b": self.params_b,
            "active_b": self.active_b, "context_tokens": self.context_tokens,
            "kv_type_k": self.kv_type_k, "kv_type_v": self.kv_type_v,
            "ubatch": self.ubatch, "slots": self.slots, "flash_attn": self.flash_attn,
            "spec_enabled": self.spec_enabled, "spec_type": self.spec_type,
            "draft_max": self.draft_max, "numa_policy": self.numa_policy,
            "numa_instance": self.numa_instance, "cpu_shape_class": self.cpu_shape_class,
            "model_mem_gb": self.model_mem_gb,
            "ld_library_path": list(self.ld_library_path),
        }


@dataclass(frozen=True)
class UnplannableRole:
    """A live role that could not be turned into a cell, and why.

    The seed `continue`d past three of these shapes. A live role that leaves the
    matrix without a record is exactly how a freeze ships a role nobody measured.
    """

    role: str
    code: str
    reason: str

    def __post_init__(self) -> None:
        if self.code not in FINDING_SPEC:
            raise PlanInputError(f"UnplannableRole.code {self.code!r} is not a finding code")

    def to_dict(self) -> dict:
        return {"role": self.role, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class OutOfScopeRole:
    """A role the priors carry that is not a protected production cell."""

    role: str
    deployment_status: Optional[str]
    reason: str

    def to_dict(self) -> dict:
        return {"role": self.role, "deployment_status": self.deployment_status,
                "reason": self.reason}


def _as_mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _opt_int(value: Any) -> Optional[int]:
    # `bool` is an `int` in Python and `slots: true` would silently become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _opt_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _opt_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def extract_role_facts(priors: Mapping[str, Any], *,
                       live_statuses: frozenset = LIVE_DEPLOYMENT_STATUSES,
                       model_digests: Optional[Mapping[str, str]] = None,
                       ) -> tuple:
    """Flatten compiled stack priors into `(RoleFacts, UnplannableRole, OutOfScopeRole)`.

    One launch entry becomes one `RoleFacts`: a role with two entries is serving on
    two servers, and merging them would lose a placement the freeze has to protect.
    A role with no entries but a resolvable runtime still yields one entry-less fact,
    because a serving role with no recorded entry is still serving.

    Nothing is skipped silently. Every live role that cannot be flattened comes back
    as an `UnplannableRole` with a reason; every non-live role comes back as an
    `OutOfScopeRole` with its status.
    """
    if not isinstance(priors, Mapping):
        raise PlanInputError(f"priors must be a mapping, got {type(priors).__name__}")
    roles = priors.get("roles")
    if not isinstance(roles, Mapping):
        raise PlanInputError(
            "compiled stack priors must carry a `roles` mapping; without it there is no "
            "derivation and the only alternative is a curated list (§10.1)")
    digests = dict(model_digests or {})

    facts: list = []
    unplannable: list = []
    out_of_scope: list = []

    # `key=` on the sort: a YAML prior may key a role by something that is not a string
    # (`8072:` parses as an int), and a bare `sorted()` over mixed keys raises TypeError
    # out of the compiler — an unhandled crash where every other malformed input here
    # produces a record.
    for role, record in sorted(roles.items(), key=lambda item: str(item[0])):
        if not isinstance(record, Mapping):
            unplannable.append(UnplannableRole(
                role=str(role), code=F_ROLE_RECIPE_INCOMPLETE,
                reason=f"role record is {type(record).__name__}, not a mapping"))
            continue
        status = record.get("deployment_status")
        if status not in live_statuses:
            out_of_scope.append(OutOfScopeRole(
                role=str(role), deployment_status=status if isinstance(status, str) else None,
                reason=(f"deployment_status={status!r} is not one of "
                        f"{sorted(live_statuses)}; only serving roles are protected cells")))
            continue

        serving = _as_mapping(record.get("serving"))
        launch = _as_mapping(serving.get("launch"))
        runtime = _as_mapping(launch.get("runtime"))
        requirements = _as_mapping(launch.get("requirements"))
        cache = _as_mapping(runtime.get("cache"))
        flags = _as_mapping(runtime.get("flags"))
        spec = _as_mapping(flags.get("spec"))
        model_block = _as_mapping(record.get("model"))
        policy = _as_mapping(record.get("policy"))

        binary_path = runtime.get("binary_path")
        if not isinstance(binary_path, str) or not binary_path.strip():
            unplannable.append(UnplannableRole(
                role=str(role), code=F_ROLE_RECIPE_INCOMPLETE,
                reason=("a live role with no resolved serving.launch.runtime.binary_path "
                        "cannot be attributed to a backend")))
            continue
        model_path = requirements.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            unplannable.append(UnplannableRole(
                role=str(role), code=F_ROLE_RECIPE_INCOMPLETE,
                reason="a live role with no requirements.model_path has no model to protect"))
            continue

        try:
            normalized_binary = normalize_path(binary_path, label=f"{role}.binary_path")
            identity = ModelIdentity(
                model_path=model_path,
                sha256=digests.get(normalize_path(model_path, label=f"{role}.model_path")),
                model_id=record.get("model_id") if isinstance(record.get("model_id"), str)
                else None,
            )
        except PlanInputError as exc:
            unplannable.append(UnplannableRole(
                role=str(role), code=F_ROLE_RECIPE_INCOMPLETE, reason=str(exc)))
            continue

        ld_entries = runtime.get("ld_library_path")
        ld_tuple = tuple(e for e in ld_entries if isinstance(e, str) and e.strip()) \
            if isinstance(ld_entries, (list, tuple)) else ()

        entries = launch.get("entries")
        entry_list: list = []
        if isinstance(entries, (list, tuple)):
            for index, launch_entry in enumerate(entries):
                if isinstance(launch_entry, Mapping):
                    entry_list.append(launch_entry)
                else:
                    # A comprehension that filtered these out was the seed's silent
                    # `continue` one level down: an entry is a SERVER, so dropping one
                    # drops a placement from the matrix. Worse, dropping every entry
                    # falls through to the entry-less fallback below, and a role with
                    # two unparsable servers then looks like one server with no port.
                    unplannable.append(UnplannableRole(
                        role=str(role), code=F_ROLE_RECIPE_INCOMPLETE,
                        reason=(f"serving.launch.entries[{index}] is a "
                                f"{type(launch_entry).__name__}, not a mapping; that "
                                "launch entry is a server this plan does not protect")))
        if not entry_list:
            entry_list = [{}]

        for entry in entry_list:
            facts.append(RoleFacts(
                role=str(role),
                deployment_status=str(status),
                binary_path=normalized_binary,
                model=identity,
                port=_opt_int(entry.get("port")),
                quant=model_block.get("quant")
                if isinstance(model_block.get("quant"), str) else None,
                architecture=model_block.get("arch")
                if isinstance(model_block.get("arch"), str) else None,
                model_family=model_block.get("family")
                if isinstance(model_block.get("family"), str) else None,
                params_b=_opt_float(model_block.get("params_b")),
                active_b=_opt_float(model_block.get("active_b")),
                context_tokens=_opt_int(cache.get("context_tokens")),
                kv_type_k=cache.get("kv_type_k") if isinstance(cache.get("kv_type_k"), str)
                else None,
                kv_type_v=cache.get("kv_type_v") if isinstance(cache.get("kv_type_v"), str)
                else None,
                ubatch=_opt_int(cache.get("ubatch")),
                slots=_opt_int(entry.get("slots")) if _opt_int(entry.get("slots")) is not None
                else _opt_int(cache.get("slots")),
                flash_attn=_opt_bool(flags.get("flash_attn")),
                spec_enabled=_opt_bool(spec.get("enabled")),
                spec_type=spec.get("type") if isinstance(spec.get("type"), str) else None,
                draft_max=_opt_int(spec.get("draft_max")),
                numa_policy=serving.get("numa_policy")
                if isinstance(serving.get("numa_policy"), str) else None,
                numa_instance=_opt_int(entry.get("numa_instance")),
                cpu_shape_class=entry.get("cpu_shape_class")
                if isinstance(entry.get("cpu_shape_class"), str) else None,
                model_mem_gb=_opt_float(policy.get("model_mem_gb"))
                if _opt_float(policy.get("model_mem_gb")) is not None
                else _opt_float(model_block.get("mem_gb")),
                ld_library_path=ld_tuple,
            ))

    return tuple(facts), tuple(unplannable), tuple(out_of_scope)


# =============================================================================
# Co-residency — derived from the compiled lineup, never assumed
# =============================================================================

@dataclass(frozen=True)
class CoResidencyGroup:
    """The set of servers the compiled lineup shows resident together on one backend.

    §10.2 phase 4 requires the performance matrix to include co-resident cells, and
    §13.2 makes co-resident lineup cells a `llama_cpu` adapter responsibility. The
    group is derived from the priors AS COMPILED — it is the lineup that is actually
    configured, not an assumption about fleet modes (`full`/`half` are exclusive with
    `quarters` per role, so a different lineup compiles a different plan).
    """

    backend: str
    group_id: str
    members: tuple  # tuple[tuple[Optional[int], str], ...] — (port, model key)

    @property
    def label(self) -> str:
        if len(self.members) < 2:
            return CO_RESIDENCY_SINGLE
        return f"{CO_RESIDENCY_PREFIX}{self.group_id}"

    def to_dict(self) -> dict:
        return {"backend": self.backend, "group_id": self.group_id,
                "label": self.label,
                "members": [{"port": p, "model_key": m} for p, m in self.members]}


def derive_co_residency_group(backend: str, facts: Sequence[RoleFacts]) -> CoResidencyGroup:
    """Group `facts` for one backend into the servers that are resident together.

    Server identity is `(port, model key)`: roles that share a port share ONE server
    (`feedback_same_model_roles_share_server`), so four roles on 8072 are one resident
    instance, not four.
    """
    members = _sorted_unique(
        json.dumps([f.port, f.model.key], separators=(",", ":")) for f in facts)
    decoded = tuple(tuple(json.loads(m)) for m in members)
    group_id = schemas.content_hash(
        {"backend": backend, "members": [list(m) for m in decoded]})[:12]
    return CoResidencyGroup(backend=backend, group_id=group_id,
                            members=tuple((m[0], m[1]) for m in decoded))


# =============================================================================
# The cell
# =============================================================================

@dataclass(frozen=True)
class ReleaseCell:
    """One measurement the release gate owes, and every role it protects.

    The identity is the MEASUREMENT, not the role. Four roles served by one llama
    server at one recipe are one cell protecting four roles — dedup that loses the
    protection list is how a role stops being covered without anyone deciding it
    should (`feedback_model_not_role_indexing`).
    """

    cell_id: str
    backend: str
    phase: str
    model: ModelIdentity
    quant: Optional[str]
    architecture: Optional[str]
    context_tokens: Optional[int]
    kv_type_k: Optional[str]
    kv_type_v: Optional[str]
    ubatch: Optional[int]
    concurrency: Optional[int]
    speculation: Mapping[str, Any]
    placement: Mapping[str, Any]
    co_residency: str
    recipe_class: str
    protocol: Optional[PhaseProtocol]
    capacity_floor: Mapping[str, Any]
    linkage: Optional[LinkageRequirement]
    protected_roles: tuple
    protected_entries: tuple
    checks: Mapping[str, schemas.Check]
    quality_transfer: Optional[QualityTransfer] = None

    def __post_init__(self) -> None:
        # `fullmatch`: `match` + `$` would admit `"single\n"`, which this module claims
        # is inside the vocabulary an evaluation event's scope denominator accepts.
        if not re.fullmatch(rf"({CO_RESIDENCY_SINGLE}|{CO_RESIDENCY_PREFIX}[A-Za-z0-9._:-]+)",
                            self.co_residency):
            raise PlanInputError(
                f"co_residency {self.co_residency!r} is outside the vocabulary an "
                "evaluation event's scope denominator accepts")
        if self.recipe_class != RECIPE_CLASS:
            raise PlanInputError(
                f"recipe_class must be {RECIPE_CLASS!r} (invariant 15: baseline/off-recipe "
                f"cells are diagnostic and never justify or veto a release)")

    @property
    def check(self) -> schemas.Check:
        return worst_check(*self.checks.values()) if self.checks else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "backend": self.backend,
            "phase": self.phase,
            "model": self.model.to_dict(),
            "quant": self.quant,
            "architecture": self.architecture,
            "context_tokens": self.context_tokens,
            "kv_type_k": self.kv_type_k,
            "kv_type_v": self.kv_type_v,
            "ubatch": self.ubatch,
            "concurrency": self.concurrency,
            "speculation": dict(self.speculation),
            "placement": dict(self.placement),
            "co_residency": self.co_residency,
            "recipe_class": self.recipe_class,
            "protocol": None if self.protocol is None else self.protocol.to_dict(),
            "capacity_floor": dict(self.capacity_floor),
            "linkage": None if self.linkage is None else self.linkage.to_dict(),
            "protected_roles": list(self.protected_roles),
            "protected_entries": [{"role": r, "port": p} for r, p in self.protected_entries],
            "quality_transfer": (None if self.quality_transfer is None
                                 else self.quality_transfer.to_dict()),
            "checks": {k: _check_dict(v) for k, v in sorted(self.checks.items())},
            "check": _check_dict(self.check),
            # A cell names the recipe DIMENSIONS and requires a constructor; it never
            # carries argv. There is no T3 recipe family in evaluator/recipes.py, and
            # hand-typed argv voids a run (P-AK-SEARCH-1 precondition 6,
            # `bench-cpu.md:8-10`), so a command line invented here would be worse than
            # no command line at all.
            "command_line": None,
            "recipe_constructor_required": True,
        }


def _cell_key(*, backend: str, phase: str, facts: RoleFacts, co_residency: str) -> dict:
    """The dedup key: the full measurement identity, and nothing about the role."""
    return {
        "backend": backend,
        "phase": phase,
        "model": facts.model.key,
        "quant": facts.quant,
        "architecture": facts.architecture,
        "context_tokens": facts.context_tokens,
        "kv_type_k": facts.kv_type_k,
        "kv_type_v": facts.kv_type_v,
        "ubatch": facts.ubatch,
        "concurrency": facts.slots,
        "flash_attn": facts.flash_attn,
        "speculation": {"enabled": facts.spec_enabled, "type": facts.spec_type,
                        "draft_max": facts.draft_max},
        "placement": {"numa_policy": facts.numa_policy, "numa_instance": facts.numa_instance,
                      "cpu_shape_class": facts.cpu_shape_class},
        "co_residency": co_residency,
        "recipe_class": RECIPE_CLASS,
    }


def _capacity_floor(facts: RoleFacts) -> tuple:
    """The FIXED floor a candidate must stay inside (§10.2 phase 7).

    Derived from the incumbent's own declared production footprint, so it is fixed
    before any candidate number exists — which is the point of a floor. Returns
    `(floor, check)`; a floor with an unknown component is COULD_NOT_CHECK, never a
    floor of zero and never silently omitted.
    """
    missing: list = []
    if facts.model_mem_gb is None:
        missing.append("the incumbent's resident footprint (policy.model_mem_gb / "
                       "model.mem_gb) is not recorded")
    if facts.context_tokens is None:
        missing.append("the served context length (cache.context_tokens) is not recorded")
    floor = {
        "resident_gb_max": facts.model_mem_gb,
        "context_tokens_min": facts.context_tokens,
        "kv_type_k": facts.kv_type_k,
        "kv_type_v": facts.kv_type_v,
        "direction": "candidate must not need more memory nor serve less context",
        "basis": "incumbent production footprint, compiled stack priors",
    }
    if missing:
        return floor, schemas.Check(schemas.COULD_NOT_CHECK, tuple(missing))
    return floor, schemas.Check(schemas.PASS)


# =============================================================================
# Transfer receipts
# =============================================================================

@dataclass(frozen=True)
class TransferReceipt:
    """Why a backend's cells were dropped, and what evidence stands in their place.

    §10.2 phase 1: *"Confirmed unchanged and incumbent evidence still in scope ⇒ that
    backend's cells drop with a transfer receipt naming the incumbent artifacts and
    their hashes."* Both halves are required here: the §3.2 result AND the named
    artifacts. "Unchanged with nothing to transfer" is not a transfer, it is a gap.
    """

    backend: str
    production_base_commit: str
    candidate_commit: str
    unchanged_result: Mapping[str, Any]
    incumbent: Mapping[str, Any]
    dropped_cell_ids: tuple
    dropped_cell_count: int
    basis: str = "§3.2 backend-unchanged: source-closure gate + normalized-binary confirmation"

    def to_dict(self) -> dict:
        return {"backend": self.backend,
                "production_base_commit": self.production_base_commit,
                "candidate_commit": self.candidate_commit,
                "unchanged_result": dict(self.unchanged_result),
                "incumbent": dict(self.incumbent),
                "dropped_cell_ids": list(self.dropped_cell_ids),
                "dropped_cell_count": self.dropped_cell_count,
                "basis": self.basis}


# =============================================================================
# Per-backend plan
# =============================================================================

@dataclass(frozen=True)
class BackendPlan:
    """One backend's share of the release matrix — or its receipt for not having one."""

    backend: str
    binding_ref: Mapping[str, Any]
    cells: tuple
    transfer_receipt: Optional[TransferReceipt]
    co_residency_group: Optional[CoResidencyGroup]
    affected_ops: tuple
    uncovered_ops: tuple
    canary_roles: tuple
    findings: tuple
    checks: Mapping[str, schemas.Check]
    #: The §3.2 result verbatim, as a RECORD rather than as a gate. "This backend's
    #: binary changed" is the ordinary case and the reason the cells exist; folding it
    #: into the plan's own check would report every real release as failing.
    unchanged_ref: Optional[Mapping[str, Any]] = None
    #: Cell-scoped findings that a transfer receipt made moot. They are RETAINED rather
    #: than discarded: if the receipt is later withdrawn — the incumbent evidence goes
    #: out of scope, the closure turns out to be too narrow — these are exactly the gaps
    #: that come back, and a reader should not have to recompile to learn they existed.
    suppressed_findings: tuple = ()

    @property
    def cells_dropped(self) -> bool:
        return self.transfer_receipt is not None

    @property
    def check(self) -> schemas.Check:
        parts = list(self.checks.values()) + [c.check for c in self.cells]
        parts.extend(schemas.Check(f.outcome, (f"{f.code}: {f.detail}",))
                     for f in self.findings if f.gating)
        return worst_check(*parts) if parts else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "binding": dict(self.binding_ref),
            "cells": [c.to_dict() for c in self.cells],
            "cell_count": len(self.cells),
            "cells_dropped": self.cells_dropped,
            "transfer_receipt": (None if self.transfer_receipt is None
                                 else self.transfer_receipt.to_dict()),
            "backend_unchanged": (None if self.unchanged_ref is None
                                  else dict(self.unchanged_ref)),
            "co_residency_group": (None if self.co_residency_group is None
                                   else self.co_residency_group.to_dict()),
            "affected_ops": list(self.affected_ops),
            "uncovered_ops": list(self.uncovered_ops),
            "canary_roles": list(self.canary_roles),
            "findings": [f.to_dict() for f in self.findings],
            "suppressed_findings": [f.to_dict() for f in self.suppressed_findings],
            "checks": {k: _check_dict(v) for k, v in sorted(self.checks.items())},
            "check": _check_dict(self.check),
        }


# =============================================================================
# The plan
# =============================================================================

@dataclass(frozen=True)
class ReleasePlan:
    """The compiled release plan — the `release_plan` block of a release package.

    It says what must be measured. It does not measure, seal, judge a waiver, or
    execute anything, and it carries no authority field of any kind:
    `schemas.find_authority_flavoured_keys()` over `to_dict()` must return empty, and
    `test_plan.py` asserts it does.
    """

    target: ReleaseTarget
    compiler_id: str
    compiled_at: str
    backends: tuple
    unplannable_roles: tuple
    out_of_scope_roles: tuple
    findings: tuple
    narrowing_permitted: bool
    surface_ref: Mapping[str, Any]
    waiver_refs: tuple
    requires_human_code_review: bool
    review_reasons: tuple

    @property
    def cells(self) -> tuple:
        out: list = []
        for backend_plan in self.backends:
            out.extend(backend_plan.cells)
        return tuple(out)

    @property
    def check(self) -> schemas.Check:
        parts = [b.check for b in self.backends]
        parts.extend(schemas.Check(f.outcome, (f"{f.code}: {f.detail}",))
                     for f in self.findings if f.gating)
        return worst_check(*parts) if parts else schemas.Check(schemas.PASS)

    @property
    def blocking_findings(self) -> tuple:
        out = [f for f in self.findings if f.gating]
        for backend_plan in self.backends:
            out.extend(f for f in backend_plan.findings if f.gating)
        return tuple(out)

    def cells_for(self, backend: str) -> tuple:
        for backend_plan in self.backends:
            if backend_plan.backend == backend:
                return backend_plan.cells
        raise PlanInputError(f"no plan for backend {backend!r}")

    def protected_roles(self) -> tuple:
        return _sorted_unique(r for cell in self.cells for r in cell.protected_roles)

    def to_dict(self) -> dict:
        return {
            "schema": PLAN_SCHEMA,
            "compiler_id": self.compiler_id,
            "compiled_at": self.compiled_at,
            "target": self.target.to_dict(),
            "narrowing_permitted": self.narrowing_permitted,
            "surface": dict(self.surface_ref),
            "backends": [b.to_dict() for b in self.backends],
            "cell_count": len(self.cells),
            "protected_roles": list(self.protected_roles()),
            "unplannable_roles": [r.to_dict() for r in self.unplannable_roles],
            "out_of_scope_roles": [r.to_dict() for r in self.out_of_scope_roles],
            "findings": [f.to_dict() for f in self.findings],
            "blocking_finding_codes": list(
                _sorted_unique(f.code for f in self.blocking_findings)),
            "requires_human_code_review": self.requires_human_code_review,
            "review_marker": REQUIRES_HUMAN_CODE_REVIEW
            if self.requires_human_code_review else None,
            "review_reasons": list(self.review_reasons),
            "waiver_refs": [dict(w) for w in self.waiver_refs],
            "recipe_class": RECIPE_CLASS,
            "check": _check_dict(self.check),
            # Said in the artifact, not only in the docstring: this plan is an input to
            # a human-executed transaction (§1.3, §11.2, invariant 5).
            "executed_by": "operator",
            "notice": ("AutoKernel compiles this plan and never executes a freeze or a "
                       "cutover; a human does (MEASUREMENT.md:140-142)."),
        }

    def sha256(self) -> str:
        return schemas.content_hash(self.to_dict())


# =============================================================================
# The compiler
# =============================================================================

def _validate_bindings(target: ReleaseTarget,
                       bindings: Mapping[str, BackendBinding]) -> dict:
    if not isinstance(bindings, Mapping) or not bindings:
        raise PlanInputError("bindings must be a non-empty mapping of backend -> BackendBinding")
    resolved: dict = {}
    for backend, binding in bindings.items():
        if not isinstance(binding, BackendBinding):
            raise PlanInputError(
                f"bindings[{backend!r}] must be a BackendBinding, got "
                f"{type(binding).__name__}")
        if binding.backend != backend:
            raise PlanInputError(
                f"bindings[{backend!r}] declares backend {binding.backend!r}")
        resolved[backend] = binding
    missing = [b for b in target.backends if b not in resolved]
    if missing:
        raise PlanInputError(
            f"no binding for {missing}; freeze scope is the union of backends served by "
            "the tree (§1.5), so a backend without a binding cannot be shown to owe "
            "nothing")
    # Bindings for OTHER trees are welcome and are not a defect: the stack runs four
    # production binaries, and a whisper role must be classifiable as "another tree's"
    # rather than falling through to UNCLASSIFIED while llama.cpp is being released.
    for backend, binding in resolved.items():
        if path_is_under(binding.production_tree_path, target.candidate_build_root):
            raise ProductionWriteRefused(
                f"candidate build root {target.candidate_build_root!r} lies inside the "
                f"production tree {binding.production_tree_path!r}. Invariant 3: no actor "
                "builds in a production tree; ALL kernel work happens in "
                "llama.cpp-experimental worktrees.")
    return resolved


def _classify(binary_path: str, bindings: Mapping[str, BackendBinding]) -> tuple:
    """Return `(backend, claimants)` for a serving binary path.

    Longest-root-wins so that a binding declaring `…/llama.cpp/build` and one declaring
    `…/llama.cpp/build/bin` do not both claim a binary; genuinely ambiguous claims (two
    unrelated roots matching) come back as a list for the caller to file.
    """
    claimants = [b for b in bindings.values() if b.claims_binary(binary_path)]
    if not claimants:
        return None, ()
    if len(claimants) == 1:
        return claimants[0].backend, (claimants[0].backend,)
    best: list = []
    best_len = -1
    for binding in claimants:
        longest = max(len(normalize_path(r, label="root")) for r in binding.binary_roots
                      if path_is_under(r, binary_path))
        if longest > best_len:
            best_len, best = longest, [binding]
        elif longest == best_len:
            best.append(binding)
    if len(best) == 1:
        return best[0].backend, tuple(sorted(b.backend for b in claimants))
    return None, tuple(sorted(b.backend for b in claimants))


def drop_verdict_contradictions(result: surface.BackendUnchangedResult) -> tuple:
    """Reasons `result.may_drop_cells` contradicts the evidence on the same object.

    `surface.BackendUnchangedResult` is an ordinary frozen dataclass and
    `may_drop_cells` is a plain FIELD, not a property. `backend_unchanged()` derives it
    correctly — but nothing stops a caller (or a future refactor, or a resumed record
    rehydrated by hand) from constructing the result directly with
    `may_drop_cells=True` beside `unchanged=FAIL`. That single boolean deletes an entire
    backend's release matrix, so this compiler re-derives the precondition instead of
    reading it: a verdict a caller can set is a verdict a caller can set wrongly, which
    is the same rule this module already applies to `PlanFinding.severity` and to the
    plan's own check.

    The conditions are `backend_unchanged()`'s own, restated: stage 1 PASS, stage 2 ran
    and PASSed, the stages agree, the incumbent's evidence is in scope, and nothing was
    filed against build identity. An empty result means the object agrees with itself.
    """
    if not isinstance(result, surface.BackendUnchangedResult):
        raise PlanInputError("expected a surface.BackendUnchangedResult")
    reasons: list = []
    if result.unchanged.outcome != schemas.PASS:
        reasons.append(f"`unchanged` is {result.unchanged.outcome}, not PASS "
                       f"({'; '.join(result.unchanged.reasons) or 'no reason given'})")
    if result.agreement.outcome != schemas.PASS:
        reasons.append(f"`agreement` is {result.agreement.outcome}, not PASS "
                       f"({'; '.join(result.agreement.reasons) or 'no reason given'})")
    if result.stage1.check.outcome != schemas.PASS:
        reasons.append(f"stage 1 is {result.stage1.check.outcome}, not PASS")
    if result.stage2 is None:
        reasons.append("stage 2 (normalized binary confirmation) did not run; §3.2 "
                       "requires it before cells drop")
    elif result.stage2.check.outcome != schemas.PASS:
        reasons.append(f"stage 2 is {result.stage2.check.outcome}, not PASS")
    scope_check = result.transfer_scope.check()
    if scope_check.outcome != schemas.PASS:
        reasons.append(f"the incumbent's evidence is not in scope: {scope_check.outcome} "
                       f"({'; '.join(scope_check.reasons)})")
    if result.findings:
        reasons.append("build-identity findings are filed: "
                       + ", ".join(f.code for f in result.findings))
    if result.blocking_reasons:
        reasons.append("blocking reasons are recorded: "
                       + "; ".join(result.blocking_reasons))
    return tuple(reasons)


def _unanchored_stage_commits(result: surface.BackendUnchangedResult) -> tuple:
    """Which commit fields the §3.2 stages left null.

    `compile_release_plan` cross-checks a stage's commits against the release target
    only `if base is not None`. That is right for a result that is merely RECORDED, and
    fail-open for one that is about to DROP cells: a stage pair naming no commits at all
    passes the cross-check vacuously and takes the matrix with it.
    """
    missing: list = []
    if result.stage1.base_commit is None:
        missing.append("stage1.base_commit")
    if result.stage1.candidate_commit is None:
        missing.append("stage1.candidate_commit")
    if result.stage2 is not None and result.stage2.base_commit is None:
        missing.append("stage2.base_commit")
    return tuple(missing)


def _traced_backends(reconciliation: surface.SurfaceReconciliation) -> frozenset:
    traced = reconciliation.traced
    if traced is None:
        return frozenset()
    return frozenset(e.backend for e in traced.events)


def _affected_ops_for(reconciliation: surface.SurfaceReconciliation,
                      backend: str) -> tuple:
    ops = {r.op_name for r in reconciliation.derived.op_registrations
           if r.backend == backend}
    traced = reconciliation.traced
    if traced is not None:
        ops.update(e.op_name for e in traced.events if e.backend == backend)
    return tuple(sorted(ops))


def compile_release_plan(*,
                         target: ReleaseTarget,
                         bindings: Mapping[str, BackendBinding],
                         priors: Mapping[str, Any],
                         reconciliation: surface.SurfaceReconciliation,
                         compiled_at: str,
                         unchanged_by_backend: Optional[Mapping[str, Any]] = None,
                         incumbent_evidence: Optional[Mapping[str, IncumbentEvidence]] = None,
                         stable_path_receipts: Optional[Mapping[str, StablePathReceipt]] = None,
                         op_coverage: Optional[OpShapeCoverage] = None,
                         quality_transfer: Optional[Sequence[QualityTransfer]] = None,
                         model_digests: Optional[Mapping[str, str]] = None,
                         waiver_refs: Sequence[Mapping[str, Any]] = (),
                         live_statuses: frozenset = LIVE_DEPLOYMENT_STATUSES,
                         ) -> ReleasePlan:
    """Compile one source tree's release plan (§10.1).

    The join, in the order the design lists it: source tree and the backends it serves;
    stable production kernel paths; distinct production models and roles; quant,
    context, KV, speculation, concurrency, placement and co-residency recipes;
    architecture class; observed op/shape coverage; the reconciled affected-surface
    manifest; per-backend protocol ids and thresholds; correctness/quality transfer
    eligibility; capacity floors; linkage requirements. Equivalent cells deduplicate,
    and each keeps the roles it protects.

    Raises on wiring and authority defects (`PlanInputError`, `ProductionWriteRefused`,
    `KernelFreezePathRefused`). Everything that is a fact about the release under
    compilation comes back as a `PlanFinding`, so it lands on the record instead of
    stopping the compile.
    """
    if not isinstance(target, ReleaseTarget):
        raise PlanInputError("target must be a ReleaseTarget")
    if not isinstance(reconciliation, surface.SurfaceReconciliation):
        raise PlanInputError("reconciliation must be a surface.SurfaceReconciliation "
                             "(§6.4 stage 3); a derived manifest alone is not reconciled")
    if reconciliation.derived.candidate_id != target.candidate_id:
        raise PlanInputError(
            f"the reconciled surface is for candidate "
            f"{reconciliation.derived.candidate_id!r} but the target is "
            f"{target.candidate_id!r}; planning one candidate's scope from another's "
            "surface would protect the wrong cells")
    _require_str(compiled_at, "compiled_at")

    resolved_bindings = _validate_bindings(target, bindings)
    unchanged_by_backend = dict(unchanged_by_backend or {})
    incumbent_evidence = dict(incumbent_evidence or {})
    stable_path_receipts = dict(stable_path_receipts or {})
    coverage = op_coverage if op_coverage is not None else OpShapeCoverage()
    if not isinstance(coverage, OpShapeCoverage):
        raise PlanInputError("op_coverage must be an OpShapeCoverage or None")
    transfers: dict = {}
    for entry in (quality_transfer or ()):
        if not isinstance(entry, QualityTransfer):
            raise PlanInputError("quality_transfer entries must be QualityTransfer")
        transfers[(entry.backend, entry.model_key)] = entry

    for backend, result in unchanged_by_backend.items():
        if not isinstance(result, surface.BackendUnchangedResult):
            raise PlanInputError(
                f"unchanged_by_backend[{backend!r}] must be a surface.BackendUnchangedResult; "
                "this compiler consumes the §3.2 test, it does not re-implement it")
        if result.backend != backend:
            raise PlanInputError(
                f"unchanged_by_backend[{backend!r}] holds a result for {result.backend!r}")
        if backend not in target.backends:
            raise PlanInputError(
                f"a §3.2 result was supplied for {backend!r}, which the target does not "
                "serve")
        base = result.stage1.base_commit
        cand = result.stage1.candidate_commit
        if base is not None and base != target.production_base_commit:
            raise PlanInputError(
                f"the §3.2 stage-1 diff for {backend!r} was taken over base {base!r} but "
                f"the release names {target.production_base_commit!r}")
        if cand is not None and cand != target.candidate_commit:
            raise PlanInputError(
                f"the §3.2 stage-1 diff for {backend!r} ends at {cand!r} but the release "
                f"names candidate {target.candidate_commit!r}")
        # A result that claims cells may drop must agree with its own evidence. This is
        # a statement about the CALLER, not about the candidate, so it raises rather
        # than becoming a waivable finding.
        if result.may_drop_cells:
            contradictions = drop_verdict_contradictions(result)
            if contradictions:
                raise PlanInputError(
                    f"unchanged_by_backend[{backend!r}] claims may_drop_cells=True while "
                    f"its own evidence says otherwise: {'; '.join(contradictions)}. "
                    "`may_drop_cells` is a plain field on a constructible dataclass, so "
                    "it is re-derived here rather than read; a boolean that empties a "
                    "backend's release matrix is not taken on trust.")

    plan_findings: list = []

    # --- the surface decides whether narrowing is permitted at all (§6.4) -------
    if reconciliation.hard_failure:
        narrowing_permitted = False
        plan_findings.append(PlanFinding(
            code=F_SURFACE_ESCAPE,
            detail=("reconciled surface reports escapes "
                    f"{[f'{a}:{v}' for a, v in reconciliation.escaped][:10]}; the derived "
                    "manifest did not contain what the trace observed"),
            filed_against="candidate"))
    elif reconciliation.check.outcome != schemas.PASS:
        narrowing_permitted = False
        plan_findings.append(PlanFinding(
            code=F_SURFACE_UNRECONCILED,
            detail=("; ".join(reconciliation.check.reasons)
                    or "the affected surface did not reconcile to PASS")))
    else:
        narrowing_permitted = True

    # --- roles -----------------------------------------------------------------
    facts, unplannable, out_of_scope = extract_role_facts(
        priors, live_statuses=live_statuses, model_digests=model_digests)
    unplannable = list(unplannable)

    by_backend: dict = {b: [] for b in target.backends}
    foreign_roles: list = []
    for fact in facts:
        backend, claimants = _classify(fact.binary_path, resolved_bindings)
        if backend is None and claimants:
            unplannable.append(UnplannableRole(
                role=fact.role, code=F_ROLE_BINARY_AMBIGUOUS,
                reason=(f"{fact.binary_path!r} lies under the declared roots of "
                        f"{list(claimants)}")))
            plan_findings.append(PlanFinding(
                code=F_ROLE_BINARY_AMBIGUOUS,
                detail=(f"role {fact.role!r}: {fact.binary_path!r} is claimed by "
                        f"{list(claimants)}"),
                filed_against="backend_bindings"))
            continue
        if backend is None:
            unplannable.append(UnplannableRole(
                role=fact.role, code=F_ROLE_BINARY_UNCLASSIFIED,
                reason=(f"{fact.binary_path!r} lies under no declared backend root; the "
                        "seed would have defaulted it to `cpu`")))
            plan_findings.append(PlanFinding(
                code=F_ROLE_BINARY_UNCLASSIFIED,
                detail=(f"role {fact.role!r} is served by {fact.binary_path!r}, which "
                        "resolves under no declared backend root")))
            continue
        if backend not in by_backend:
            # A live role served by a backend this tree does not freeze — e.g. a whisper
            # role while llama.cpp is being released. It is not out of scope for the
            # STACK, only for THIS tree, and saying so is not the same as dropping it.
            foreign_roles.append(OutOfScopeRole(
                role=fact.role, deployment_status=fact.deployment_status,
                reason=(f"served by backend {backend!r}, which is not in this tree's "
                        f"freeze scope ({list(target.backends)})")))
            continue
        by_backend[backend].append(fact)

    if any(f.model.basis == "declared_path" for f in facts):
        plan_findings.append(PlanFinding(
            code=F_MODEL_IDENTITY_BY_PATH_ONLY,
            detail=("no content digest was supplied for "
                    f"{len({f.model.model_path for f in facts if not f.model.sha256})} "
                    "model path(s); dedup on those falls back to the declared path")))

    # --- complexity ceiling (§10.6) --------------------------------------------
    review_reasons: list = []
    for backend in sorted(target.backends):
        binding = resolved_bindings[backend]
        ceiling = binding.ceiling
        if ceiling is None:
            # NOT `continue`. §10.6's concern is *"LLM-authored kernel C++/HIP should
            # not reach a release package unreviewed at arbitrary size"*, so an
            # adapter that declares no ceiling is the case the section names, not an
            # exemption from it — skipping the backend made an undeclared band read as
            # an infinite one. Two things were being skipped with it:
            #   * `core_header` and shared-ggml-core are KINDS of change, not size
            #     bands (AK-D30). They force review at any size and have nothing to do
            #     with whether a band was declared; and
            #   * an unevaluated diff size is not a small one.
            # An all-default ceiling reproduces exactly the kind-based rules and no
            # size band, and the undeclared band is itself named as a review reason.
            ceiling = ComplexityCeiling()
            review_reasons.append(
                f"{backend}: no complexity/blast-radius ceiling is declared, so the "
                "diff's size was never evaluated against one (§10.6)")
        exceeded = ceiling.exceeded_by(
            diff_lines=target.diff_lines, files_touched=target.files_touched,
            touches_shared_core=target.touches_shared_core,
            change_classes=target.change_classes)
        for reason in exceeded:
            review_reasons.append(f"{backend}: {reason}")
    if review_reasons:
        plan_findings.append(PlanFinding(
            code=F_DIFF_COMPLEXITY_CEILING_EXCEEDED,
            detail="; ".join(review_reasons)))

    traced_backends = _traced_backends(reconciliation)

    backend_plans: list = []
    for backend in sorted(target.backends):
        backend_plans.append(_compile_backend(
            backend=backend,
            binding=resolved_bindings[backend],
            facts=tuple(by_backend[backend]),
            target=target,
            reconciliation=reconciliation,
            unchanged=unchanged_by_backend.get(backend),
            incumbent=incumbent_evidence.get(backend),
            receipt=stable_path_receipts.get(backend),
            coverage=coverage,
            transfers=transfers,
            traced_backends=traced_backends,
            narrowing_permitted=narrowing_permitted,
        ))

    return ReleasePlan(
        target=target,
        compiler_id=COMPILER_ID,
        compiled_at=compiled_at,
        backends=tuple(backend_plans),
        unplannable_roles=tuple(unplannable),
        out_of_scope_roles=tuple(out_of_scope) + tuple(foreign_roles),
        findings=tuple(plan_findings),
        narrowing_permitted=narrowing_permitted,
        surface_ref={
            "candidate_id": reconciliation.derived.candidate_id,
            "derived_sha256": reconciliation.derived.sha256(),
            "traced_sha256": (None if reconciliation.traced is None
                              else reconciliation.traced.sha256()),
            "check": _check_dict(reconciliation.check),
            "hard_failure": reconciliation.hard_failure,
            "full_tree": reconciliation.derived.full_tree,
        },
        waiver_refs=tuple(dict(w) for w in waiver_refs),
        requires_human_code_review=bool(review_reasons),
        review_reasons=tuple(review_reasons),
    )


def _compile_backend(*, backend: str, binding: BackendBinding, facts: tuple,
                     target: ReleaseTarget,
                     reconciliation: surface.SurfaceReconciliation,
                     unchanged: Optional[surface.BackendUnchangedResult],
                     incumbent: Optional[IncumbentEvidence],
                     receipt: Optional[StablePathReceipt],
                     coverage: OpShapeCoverage,
                     transfers: Mapping[tuple, QualityTransfer],
                     traced_backends: frozenset,
                     narrowing_permitted: bool) -> BackendPlan:
    """Compile one backend's cells, or its receipt for having none.

    `cell_findings` is separate from `findings` on purpose. Everything in it is a
    statement about EXECUTING this backend's cells — an incomplete recipe, a floor that
    could not be derived, an unproven linkage, a missing protocol band, a lineup with
    no co-resident pair. When the §3.2 test drops the cells, none of that work is owed,
    and filing it anyway would block a release on gaps in evidence the receipt has just
    shown nobody needs. Structural findings — the stable-path receipt, the §3.2 test
    itself, a stage disagreement, an empty matrix — are filed either way, because they
    are about the drop DECISION rather than about the dropped work.
    """
    findings: list = []
    cell_findings: list = []
    checks: dict = {}

    # Adapter-owned preconditions are facts the generic compiler cannot derive:
    # protocol registry membership, pinned cross-modality instruments and source-
    # closure traversal. They remain explicit checks, preserving FAIL vs
    # COULD_NOT_CHECK rather than collapsing both into one compiler finding.
    for prerequisite_id, check in sorted(binding.prerequisites.items()):
        checks[f"prerequisite.{prerequisite_id}"] = check

    # --- stable production kernel path (§1.5) ---------------------------------
    if receipt is None:
        checks["stable_path"] = schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"no receipt for {binding.stable_production_path!r}",))
        findings.append(PlanFinding(
            code=F_STABLE_PATH_RECEIPT_MISSING, backend=backend,
            detail=f"no receipt records what {binding.stable_production_path!r} resolves to"))
    else:
        if receipt.backend != backend:
            raise PlanInputError(
                f"stable-path receipt for {receipt.backend!r} was supplied for {backend!r}")
        path_check = receipt.check_against(binding)
        checks["stable_path"] = path_check
        if path_check.outcome == schemas.FAIL:
            findings.append(PlanFinding(
                code=F_STABLE_PATH_NOT_IN_PRODUCTION_TREE, backend=backend,
                detail="; ".join(path_check.reasons)))

    # --- co-residency group ----------------------------------------------------
    group = derive_co_residency_group(backend, facts) if facts else None
    co_resident = group is not None and len(group.members) >= 2
    if binding.co_residency_required and not co_resident:
        cell_findings.append(PlanFinding(
            code=F_CORESIDENT_CELL_MISSING, backend=backend,
            detail=("the compiled lineup shows "
                    f"{0 if group is None else len(group.members)} resident server(s), so "
                    "no co-resident cell could be derived, but this backend requires one "
                    "(§13.2, §10.2 phase 4)")))

    # --- cells -----------------------------------------------------------------
    cells_by_id: dict = {}
    for fact in facts:
        recipe_gaps: list = []
        if fact.context_tokens is None:
            recipe_gaps.append("cache.context_tokens is not recorded")
        if fact.quant is None:
            recipe_gaps.append("model.quant is not recorded")
        if fact.architecture is None:
            recipe_gaps.append("model.arch is not recorded")
        if recipe_gaps:
            cell_findings.append(PlanFinding(
                code=F_ROLE_RECIPE_INCOMPLETE, backend=backend,
                detail=f"role {fact.role!r}: " + "; ".join(recipe_gaps)))

        floor, floor_check = _capacity_floor(fact)
        if floor_check.outcome != schemas.PASS:
            cell_findings.append(PlanFinding(
                code=F_CAPACITY_FLOOR_INCOMPLETE, backend=backend,
                detail=f"role {fact.role!r}: " + "; ".join(floor_check.reasons)))

        linkage_check = (binding.linkage.check_declared(fact.ld_library_path)
                         if binding.linkage is not None else schemas.Check(schemas.PASS))
        if linkage_check.outcome != schemas.PASS:
            cell_findings.append(PlanFinding(
                code=F_LINKAGE_REQUIREMENT_UNPROVEN, backend=backend,
                detail=f"role {fact.role!r}: " + "; ".join(linkage_check.reasons)))

        transfer = transfers.get((backend, fact.model.key))
        if transfer is not None and transfer.check().outcome != schemas.PASS:
            cell_findings.append(PlanFinding(
                code=F_QUALITY_TRANSFER_REFUSED, backend=backend,
                detail=(f"model {fact.model.key}: "
                        + "; ".join(transfer.check().reasons))))

        co_labels = [CO_RESIDENCY_SINGLE]
        if co_resident:
            co_labels.append(group.label)

        for phase in sorted(binding.phases):
            protocol = binding.protocols.get(phase)
            protocol_check = schemas.Check(schemas.PASS)
            if protocol is None:
                protocol_check = schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    (f"no release protocol is declared for {backend}/{phase}",))
            elif not protocol.thresholds:
                protocol_check = schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    (f"{protocol.protocol_id} is named for {backend}/{phase} but carries "
                     "no thresholds",))

            for co_label in co_labels:
                key = _cell_key(backend=backend, phase=phase, facts=fact,
                                co_residency=co_label)
                cell_id = "akcell-" + schemas.content_hash(key)[:16]
                cell_checks = {"protocol": protocol_check,
                               "capacity_floor": floor_check,
                               "linkage": linkage_check}
                existing = cells_by_id.get(cell_id)
                if existing is not None:
                    cells_by_id[cell_id] = _merge_protection(
                        existing, fact, checks=cell_checks, floor=floor)
                    continue
                cells_by_id[cell_id] = ReleaseCell(
                    cell_id=cell_id,
                    backend=backend,
                    phase=phase,
                    model=fact.model,
                    quant=fact.quant,
                    architecture=fact.architecture,
                    context_tokens=fact.context_tokens,
                    kv_type_k=fact.kv_type_k,
                    kv_type_v=fact.kv_type_v,
                    ubatch=fact.ubatch,
                    concurrency=fact.slots,
                    speculation={"enabled": fact.spec_enabled, "type": fact.spec_type,
                                 "draft_max": fact.draft_max},
                    placement={"numa_policy": fact.numa_policy,
                               "numa_instance": fact.numa_instance,
                               "cpu_shape_class": fact.cpu_shape_class,
                               "flash_attn": fact.flash_attn},
                    co_residency=co_label,
                    recipe_class=RECIPE_CLASS,
                    protocol=protocol,
                    capacity_floor=floor,
                    linkage=binding.linkage,
                    protected_roles=(fact.role,),
                    protected_entries=((fact.role, fact.port),),
                    checks=cell_checks,
                    quality_transfer=transfer,
                )

    for phase in sorted(binding.phases):
        if phase not in binding.protocols:
            cell_findings.append(PlanFinding(
                code=F_RELEASE_PROTOCOL_UNDEFINED, backend=backend,
                detail=(f"backend {backend!r} serves phase {phase!r} but declares no "
                        "release protocol for it")))
        elif not binding.protocols[phase].thresholds:
            cell_findings.append(PlanFinding(
                code=F_PHASE_THRESHOLDS_UNDECLARED, backend=backend,
                detail=(f"{binding.protocols[phase].protocol_id} is named for "
                        f"{backend}/{phase} but carries no thresholds")))

    if binding.host_capacity_budget_gb is None and co_resident:
        cell_findings.append(PlanFinding(
            code=F_HOST_CAPACITY_BUDGET_UNDECLARED, backend=backend,
            detail=("the lineup is co-resident but no host/device capacity budget was "
                    "declared, so the summed residency floor could not be checked")))

    cells = tuple(sorted(cells_by_id.values(), key=lambda c: c.cell_id))

    # --- the §3.2 drop decision -------------------------------------------------
    # Decided BEFORE op coverage on purpose: a backend whose binary is confirmed
    # unchanged executes no candidate kernel, so an uncovered op on it is not a hole in
    # this release's evidence. Checking coverage first would file a blocking finding
    # about work the receipt has just shown nobody owes.
    transfer_receipt = None
    if unchanged is None:
        checks["backend_unchanged_test_present"] = schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("no §3.2 backend-unchanged result was supplied",))
        findings.append(PlanFinding(
            code=F_BACKEND_UNCHANGED_TEST_NOT_RUN, backend=backend,
            detail=("no §3.2 result was supplied, so this backend keeps its full matrix; "
                    "absence of the test is not a pass")))
    else:
        checks["backend_unchanged_test_present"] = schemas.Check(schemas.PASS)
        disagreements = [
            f.detail for f in unchanged.findings
            if f.code in (surface.FINDING_STAGE_DISAGREEMENT_SOURCE_CLEAN,
                          surface.FINDING_STAGE_DISAGREEMENT_SOURCE_DIRTY)]
        if not disagreements and unchanged.agreement.outcome == schemas.FAIL:
            # Read the AGREEMENT, not only the findings tuple. Filing solely off
            # `unchanged.findings` means the guarantee "a stage disagreement is a FAIL
            # filed against build identity" can be defeated by handing over a result
            # whose findings tuple is empty — deleting the thing the check inspects,
            # while `agreement` still says FAIL in the same object.
            disagreements = list(unchanged.agreement.reasons) or [
                "the two §3.2 stages disagree and the result carries no detail"]
        for detail in disagreements:
            findings.append(PlanFinding(
                code=F_BUILD_IDENTITY_STAGE_DISAGREEMENT, backend=backend,
                detail=detail, filed_against="build_identity"))

        claims_unchanged = unchanged.may_drop_cells
        served_by_tree = tuple(
            candidate_backend for candidate_backend, source_tree in
            schemas.SOURCE_TREE_BY_BACKEND.items()
            if source_tree == binding.source_tree)
        if claims_unchanged and len(served_by_tree) == 1:
            findings.append(PlanFinding(
                code=F_SINGLE_BACKEND_NOOP_CANDIDATE, backend=backend,
                detail=(f"{binding.source_tree} serves only {backend}; both unchanged "
                        "stages say its sole backend is unchanged, so this candidate "
                        "contains no releasable kernel change"),
                filed_against="build_identity"))
            claims_unchanged = False
        if claims_unchanged:
            unanchored = _unanchored_stage_commits(unchanged)
            if unanchored:
                findings.append(PlanFinding(
                    code=F_BACKEND_UNCHANGED_RESULT_UNANCHORED, backend=backend,
                    detail=(f"the §3.2 result records no {list(unanchored)}, so the "
                            "cross-check against this release's base "
                            f"{target.production_base_commit} and candidate "
                            f"{target.candidate_commit} passed vacuously; a result that "
                            "names no commits cannot be shown to be about this release"),
                    filed_against="build_identity"))
                claims_unchanged = False
        if claims_unchanged and backend in traced_backends:
            findings.append(PlanFinding(
                code=F_TRACED_BACKEND_DECLARED_UNCHANGED, backend=backend,
                detail=("the dispatch trace recorded this backend executing candidate "
                        "kernels, so its binary cannot be unchanged; the source closure "
                        "is too narrow"),
                filed_against="build_identity"))
            claims_unchanged = False

        if claims_unchanged and not narrowing_permitted:
            findings.append(PlanFinding(
                code=F_SURFACE_UNRECONCILED, backend=backend,
                detail=("the §3.2 test would drop this backend's cells, but the affected "
                        "surface is not reconciled, so nothing may be narrowed "
                        "(invariant 18)")))
            claims_unchanged = False

        if claims_unchanged and incumbent is None:
            findings.append(PlanFinding(
                code=F_TRANSFER_RECEIPT_INCOMPLETE, backend=backend,
                detail=("§3.2 confirms the binary unchanged, but no incumbent evidence was "
                        "named, so there is nothing to transfer and the cells stand")))
            claims_unchanged = False

        if claims_unchanged:
            if incumbent.backend != backend:
                raise PlanInputError(
                    f"incumbent evidence for {incumbent.backend!r} was supplied for "
                    f"{backend!r}")
            transfer_receipt = TransferReceipt(
                backend=backend,
                production_base_commit=target.production_base_commit,
                candidate_commit=target.candidate_commit,
                unchanged_result=unchanged.to_dict(),
                incumbent=incumbent.to_dict(),
                dropped_cell_ids=tuple(c.cell_id for c in cells),
                dropped_cell_count=len(cells),
            )
            cells = ()

    if transfer_receipt is None:
        findings.extend(cell_findings)
        if not cells:
            findings.append(PlanFinding(
                code=F_NO_PROTECTED_CELLS, backend=backend,
                detail=("this backend has neither protected cells nor a transfer receipt; "
                        "an empty matrix would pass vacuously")))

    # --- op/shape coverage (§10.2 phase 3) -------------------------------------
    affected_ops = _affected_ops_for(reconciliation, backend)
    if transfer_receipt is not None:
        uncovered: tuple = ()
        checks["op_coverage"] = schemas.Check(schemas.PASS)
    else:
        uncovered = tuple(op for op in affected_ops
                          if not coverage.shapes_for(backend, op))
        if uncovered:
            findings.append(PlanFinding(
                code=F_UNCOVERED_AFFECTED_OP, backend=backend,
                detail=(f"{len(uncovered)} affected op(s) have no observed shape coverage: "
                        f"{list(uncovered[:10])}")))
        checks["op_coverage"] = (
            schemas.Check(schemas.PASS) if not uncovered
            else schemas.Check(schemas.COULD_NOT_CHECK,
                               (f"{len(uncovered)} affected ops uncovered",)))

    if transfer_receipt is not None or not binding.canary_required:
        canary_roles: tuple = ()
    else:
        canary_roles = _sorted_unique(r for cell in cells for r in cell.protected_roles)

    return BackendPlan(
        backend=backend,
        binding_ref=binding.to_dict(),
        cells=cells,
        transfer_receipt=transfer_receipt,
        co_residency_group=group,
        affected_ops=affected_ops,
        uncovered_ops=uncovered,
        canary_roles=canary_roles,
        findings=tuple(findings),
        checks=checks,
        unchanged_ref=None if unchanged is None else unchanged.to_dict(),
        suppressed_findings=() if transfer_receipt is None else tuple(cell_findings),
    )


def _merge_protection(cell: ReleaseCell, fact: RoleFacts, *,
                      checks: Mapping[str, schemas.Check],
                      floor: Mapping[str, Any]) -> ReleaseCell:
    """Add a role to an existing cell's protection list, WORST-OF on its checks.

    This is the whole of deduplication: two roles whose measurement identity is equal
    are ONE measurement protecting TWO roles. The identity is what merged them, so the
    recipe fields cannot differ — but the per-ROLE facts around it can, and keeping only
    the first role's would be the dedup bug in its quiet form. If role A's launcher sets
    the tree's `LD_LIBRARY_PATH` and role B's does not, the merged cell must not report
    linkage PASS: B is protected by this cell too, and B is the one that would run
    against an inherited ggml.
    """
    roles = _sorted_unique(cell.protected_roles + (fact.role,))
    entries = tuple(sorted(set(cell.protected_entries + ((fact.role, fact.port),)),
                           key=lambda e: (e[0], -1 if e[1] is None else e[1])))
    merged_checks = dict(cell.checks)
    for key, check in checks.items():
        merged_checks[key] = (worst_check(merged_checks[key], check)
                              if key in merged_checks else check)

    merged_floor = dict(cell.capacity_floor)
    here, there = cell.capacity_floor.get("resident_gb_max"), floor.get("resident_gb_max")
    if here != there:
        # One model, two declared footprints. Take the STRICTER ceiling and say so;
        # picking either silently would hold the candidate to a floor half its
        # protected roles were never measured against.
        known = [v for v in (here, there) if v is not None]
        merged_floor["resident_gb_max"] = min(known) if known else None
        merged_floor["footprint_disagreement"] = sorted(
            str(v) for v in {here, there})
        merged_checks["capacity_floor"] = worst_check(
            merged_checks.get("capacity_floor", schemas.Check(schemas.PASS)),
            schemas.Check(schemas.COULD_NOT_CHECK,
                          (f"roles {list(roles)} declare different incumbent footprints "
                           f"({here} vs {there}) for one measurement identity; the "
                           "stricter ceiling is applied and the disagreement recorded",)))

    return ReleaseCell(
        cell_id=cell.cell_id, backend=cell.backend, phase=cell.phase, model=cell.model,
        quant=cell.quant, architecture=cell.architecture,
        context_tokens=cell.context_tokens, kv_type_k=cell.kv_type_k,
        kv_type_v=cell.kv_type_v, ubatch=cell.ubatch, concurrency=cell.concurrency,
        speculation=cell.speculation, placement=cell.placement,
        co_residency=cell.co_residency, recipe_class=cell.recipe_class,
        protocol=cell.protocol, capacity_floor=merged_floor, linkage=cell.linkage,
        protected_roles=roles, protected_entries=entries, checks=merged_checks,
        quality_transfer=cell.quality_transfer,
    )


# =============================================================================
# Reading compiled priors — the only I/O in this module, and it is read-only
# =============================================================================

def load_compiled_priors(path: Any) -> Mapping[str, Any]:
    """Read compiled stack priors from a `.json` or `.yaml` file.

    Kept deliberately thin, and separate from `compile_release_plan`, so the compiler
    itself is a pure function of its inputs: a plan has to be reproducible from the
    record, and a compiler that reads the filesystem mid-derivation is not.

    Raises rather than returning `{}` on a missing or unparsable file. An empty prior
    set compiles an empty matrix, and an empty matrix passes vacuously.
    """
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    suffix = target.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # declared in pyproject as `pyyaml>=6.0`
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise PlanInputError(
                f"cannot read {target}: PyYAML is not importable ({exc})") from exc
        loaded = yaml.safe_load(text)
    else:
        raise PlanInputError(
            f"cannot read {target}: expected a .json or .yaml compiled-priors file, got "
            f"suffix {suffix!r}")
    if not isinstance(loaded, Mapping):
        raise PlanInputError(
            f"{target} parsed to {type(loaded).__name__}, not a mapping of compiled priors")
    return loaded


# =============================================================================
# Self-audit — the cardinal rule, proved from this module's own AST
# =============================================================================

def audit_plan_module_is_read_only(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's AST that it writes nothing and signals nothing.

    Delegates to `surface.audit_surface_module_is_read_only`, which already encodes
    the forbidden-import, forbidden-call and read-only-`open` rules. Reusing it is the
    point: two copies of an auditor drift, and the copy that drifts is the one that
    stops catching things.

    COULD_NOT_CHECK when the source cannot be read — an unreadable module is not an
    audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    return surface.audit_surface_module_is_read_only(source)
