#!/usr/bin/env python3
"""correctness.py — the T0 correctness surfaces, which gate before any speed work.

WHY THIS MODULE EXISTS
----------------------
`scripts/kernel_rnd/kernel_eval.sh` is the defect this module is shaped around,
and T0 is where every one of its failures had to be caught and was not:

  * **It tested ONE op.** `test-backend-ops -o MUL_MAT`, and nothing else. Every
    MoE expert path — `MUL_MAT_ID`, the op that routes tokens to experts and the
    one this project's production worker (`gemma4-26B-A4B`, MTP) executes on every
    token — was never exercised. A kernel that breaks expert dispatch and leaves
    dense gemm alone passed that gate cleanly. Here `T0Policy.__post_init__`
    REFUSES a policy whose required-op set omits `MUL_MAT_ID`, and
    `check_backend_op_units` FAILs when a required op was not exercised: an
    untested op is not a passing op, and absence of a failure in a suite that
    never ran the op is not evidence of correctness.

  * **It called any non-empty generation "coherent".** `COH="coherent"` was set
    for a non-empty string, the baseline comparison ran only when `--baseline-env`
    happened to be passed, and `kernel_store.py:81` then admitted
    `coherence in ("byte-identical","coherent")` into a CORRECT-ONLY Pareto view.
    Here coherence is a COMPUTED verdict: `CoherenceVerdict` cannot be
    constructed except by `compute_coherence()`, re-derives its own label from
    the evidence stored on it, and RAISES `CoherenceWithoutAnchor` if an
    equivalence label is paired with no bound anchor. The anchor-less case is
    structurally `not_compared` -> `COULD_NOT_CHECK` -> (via
    `api.compute_verdict`) `INVALID`. *"A run without an explicit anchor is
    INVALID — never 'correct', never 'coherent', never 'byte-identical'."*

  * **It had no notion of a gate it forgot to run.** `T0Report.__post_init__`
    raises `GateCoverageGap` unless the report carries exactly one result for
    every id in `T0_GATE_IDS`. A future edit that deletes a surface fails loudly
    instead of shipping a shorter checklist that still says PASS.

WHICH PROTOCOL CLAUSES THIS FILE IMPLEMENTS
-------------------------------------------
`measurement/protocols/kernel-research.md` (Annex K, **P-AK-SEARCH-1**, RATIFIED
2026-08-03), by section name:

  * **"Correctness precedence"** — this is the whole subject of the module. Every
    gate it emits lands in a class that `api.SPEED_BLOCKING_GATE_CLASSES` treats
    as lexicographically prior, so a T0 failure makes `Verdict.rank_key()` raise:
    *"A candidate failing any of them receives no speed rank at all — not a
    penalised one."* It also implements the two sentences that follow:
    *"Correctness verdicts are produced by the evaluator against declared oracles
    and are NEVER self-reported by the candidate. A candidate output MUST NEVER be
    cached or reused as a correctness oracle. Cache state is declared in every
    record."* — `produced_by` / `correctness_verdict_source` must be the
    evaluator, `candidate_output_used_as_oracle` is a FAIL, and an undeclared
    cache state is a FAIL rather than a blank field.
  * **"Preconditions (all enforced or attested per run)"**, precondition 4 — the
    anchor is named by source commit, binary SHA-256 and linkage SHA-256, and
    every comparison gate here declares `requires_anchor=True` and returns
    `COULD_NOT_CHECK` when no anchor is bound. The EVIDENCE names it the same way:
    every evidence type carrying anchor-DERIVED material — `LinkageEvidence`,
    `CoherenceEvidence`, `DeterminismEvidence`, `StaticAnalysisEvidence` and
    `AntiRewardHackingEvidence` — carries all three components or none
    (`_validate_anchor_triple`), and its consumer REFUSES — a raised
    `EvidenceAnchorMismatch` subclass, not a quiet downgrade — when the anchor it
    is handed is not the anchor the capture was taken against, which is the
    failure invariant 11's *"deterministic replay before regeneration"* path
    invites. An UNRECORDED identity is COULD_NOT_CHECK everywhere, never an
    implicit match: absence of a record is not agreement.
  * **"Controls — four mandatory, plus one accept-side control"** — control 3
    (degraded-negative: *"cheating, silently falling back, reducing work, or
    serving a cached result"*) is what `check_no_fallback_dispatch_proof` and
    `check_anti_reward_hacking` exist to catch, and both sit in a speed-blocking
    class so the control *"MUST receive no speed rank at all"*. `control_role`
    also flips the binary-identity rule for control 4 (A/A), where measuring the
    anchor against itself is required rather than forbidden.
  * **"What voids a run"** — not evaluated here. Void conditions are window
    properties and belong to `api.check_void_conditions`; this module returns
    gate results and never a verdict.

Design context: `epyc-root/handoffs/active/autokernel-research-loop.md` §8.5.1
(source-integrity gates), §8.6 (T0_GATE, the enumeration this module implements
one-for-one), §6.4 (affected-surface derivation, `derived ⊇ traced`), §6.5
(oracle registry), §10.6 (diff-complexity ceiling), §15.2 (the five controls),
invariant 12 (determinism class is an interface), invariant 13 (one conceptual
mutation), invariant 18 (declared equals traced; the actor's declaration is a
scored prediction, never a scope input).

WHAT THIS MODULE IS NOT
-----------------------
It runs NO build, NO op suite, NO inference and NO sanitizer. It starts, stops
and signals NO process, and it writes NO file. `build_sanitizer_invocation()`
CONSTRUCTS the ASAN/UBSAN argv and its receipt; something else, holding a claim,
may later run it. That property is proved, not promised:
`audit_no_write_or_process_paths()` parses this module's own AST through
`api.audit_no_write_or_process_paths` and FAILs on any write call, process call,
or import that would grant either. Argv is only ever a tuple — this module never
renders a shell string, so there is no quoting bug for it to have.

The evidence types below are the seam. A tier runner that really compiles,
really runs `test-backend-ops`, and really traces dispatch fills them in; the
tests fill them in from fixtures. Every seam that needs a real artifact is named
in `SEAMS` at the bottom of this file.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

from .. import schemas
from . import api

__all__ = [
    # errors
    "CorrectnessError", "T0EvidenceUnavailable", "CoherenceWithoutAnchor",
    "CoherenceTampering", "EvidenceAnchorMismatch", "CoherenceAnchorMismatch",
    "DeterminismAnchorMismatch", "StaticAnalysisAnchorMismatch",
    "AntiRewardHackingAnchorMismatch", "GateCoverageGap", "ActorDeclaredScope",
    # vocabularies
    "T0_GATE_IDS", "T0_GATE_SPEC", "MANDATORY_BACKEND_OPS", "CONTROL_ROLES",
    "COHERENCE_LABELS", "COHERENCE_EQUIVALENCE_LABELS", "NA_SOURCES",
    "EVIDENCE_PRODUCERS", "REQUIRES_HUMAN_CODE_REVIEW", "CACHE_STATES",
    # policy and applicability
    "T0Policy", "DiffComplexityCeiling", "ChangeClassEnvelope", "NotApplicable",
    # evidence types
    "ChangeSurface", "SymbolTableDiff", "BuildProvenance", "DiffPolicyEvidence",
    "StaticAnalysisEvidence", "SanitizerInvocation", "SanitizerEvidence",
    "OpSuiteEvidence", "PropertyMeasurement", "ReferenceComparison", "ReferenceEvidence",
    "BoundaryShapeEvidence", "DispatchTraceEvidence", "StateSafetyEvidence",
    "CoherenceEvidence", "DeterminismEvidence", "LinkageEvidence",
    "AntiRewardHackingEvidence", "T0Evidence",
    "SOURCE_PREREQUISITE_IDS", "SourcePrerequisiteEvidence",
    # coherence
    "CoherenceVerdict", "compute_coherence",
    # sanitizer construction
    "build_sanitizer_invocation", "check_sanitizer_invocation",
    # gate checks
    "check_symbol_and_registration_preservation", "check_clean_build_from_snapshot",
    "check_semantic_diff_conformance", "check_schema_and_diff_policy",
    "check_static_and_compile", "check_asan", "check_ubsan",
    "check_backend_op_units", "check_exact_reference_comparison",
    "check_unseen_boundary_shapes", "check_affected_surface_reconciliation",
    "check_no_fallback_dispatch_proof", "check_state_rollback_teardown_race",
    "check_output_coherence", "check_determinism_class",
    "check_binary_and_linkage_identity", "check_anti_reward_hacking",
    # aggregation and seams
    "T0Report", "evaluate_t0", "demote_anchor_requiring_passes", "T0EvidenceProvider",
    "StaticEvidenceProvider", "T0CorrectnessRunner", "audit_no_write_or_process_paths",
    "SEAMS",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class CorrectnessError(api.EvaluatorError):
    """Base class for every refusal this module makes."""


class T0EvidenceUnavailable(CorrectnessError):
    """No T0 evidence exists for a candidate the runner was asked to gate.

    This RAISES rather than returning an all-COULD_NOT_CHECK report, because a
    provider with nothing to say about a candidate is a broken pipeline, not a
    finding about the kernel. A report is a statement that seventeen surfaces
    were examined; it must not be synthesisable from nothing.
    """


class CoherenceWithoutAnchor(CorrectnessError):
    """An equivalence label was paired with no bound anchor.

    Precondition 4: *"a coherence or identity label produced without a named
    anchor comparison is not a verdict, and any record carrying one is INVALID."*
    `kernel_eval.sh` produced exactly that label, from exactly that absence.
    """


class CoherenceTampering(CorrectnessError):
    """A `CoherenceVerdict` carries a label its own evidence does not imply."""


class EvidenceAnchorMismatch(CorrectnessError):
    """Evidence was consumed against an anchor it was not captured against.

    One defect, four surfaces. Every piece of T0 evidence that carries
    anchor-DERIVED material — the anchor's output digest, its determinism class,
    its toolchain, its delivered-work count — is only meaningful when it is read
    against the anchor that produced it. Invariant 11 makes re-scoring saved
    material the normal path (*"deterministic replay before regeneration"*), so a
    capture taken against anchor A reaching a consumer holding anchor B is the
    expected accident rather than an exotic one.

    Every subclass RAISES rather than returning a degraded outcome, for the reason
    spelled out on `CoherenceAnchorMismatch`: a mismatch is a defect in the replay
    path, not a finding about the candidate, and a gate result would file it as
    the latter. Catch this base to catch all four; catch a subclass to know which
    surface was replayed wrong.
    """


class CoherenceAnchorMismatch(EvidenceAnchorMismatch):
    """A coherence capture was compared against an anchor it was not taken against.

    Invariant 11 makes re-scoring SAVED outputs a first-class cost-control
    mechanism: *"deterministic replay before regeneration"*. That is precisely the
    path on which a `CoherenceEvidence` captured against anchor A can be handed to
    `compute_coherence(anchor=B, ...)`, and without this refusal it would produce a
    perfectly well-formed `byte_identical` verdict meaning nothing — the candidate
    output agreed with an anchor output nobody claimed it came from.

    This RAISES rather than returning `not_compared`, because the two are
    different facts. `not_compared` says *no comparison was possible*, which is a
    property of the evidence; a mismatch says *the caller replayed the wrong
    material*, which is a defect in the replay path. Downgrading the second into
    the first would leave the bug in place and the pipeline green.
    """


class DeterminismAnchorMismatch(EvidenceAnchorMismatch):
    """A determinism capture was read against an anchor it was not taken against.

    `DeterminismEvidence` carries the anchor's `anchor_output_digests` and its
    `anchor_determinism_class`. Both are what SOME anchor did, and invariant 12
    makes the class *"a declared, release-relevant property"* — an interface. A
    class comparison against another anchor's interface can report a change that
    never happened, or miss one that did.

    It has two consumers and is raised from both, because the second is where the
    defect is loudest. `check_determinism_class` reads the capture against the
    anchor the record names. `check_output_coherence` reconciles the anchor's
    determinism class between the coherence capture and this one, and that
    reconciliation is the only place in this module where a *self-declared* field
    turns a byte DIFFERENCE into an equivalence label — so two records that agree
    while describing different anchors would read as corroboration and buy a PASS.
    """


class StaticAnalysisAnchorMismatch(EvidenceAnchorMismatch):
    """A static-analysis capture names another anchor's toolchain.

    The gate exists to catch *"a toolchain comparison wearing a kernel
    comparison's clothes"*. `anchor_compiler_id`, `anchor_compiler_version` and
    `anchor_warning_count` describe the anchor's build; measured against a
    different anchor's build they can hide a compiler change or invent one, which
    is the same confound arriving by another door.
    """


class AntiRewardHackingAnchorMismatch(EvidenceAnchorMismatch):
    """A delivered-work count was compared against another anchor's count.

    `delivered_units_anchor` is control 3's floor: *"delivered work may not
    shrink"*. The comparison is exact and has no tolerance knob, which is exactly
    why the two counts must come from one anchor — a floor taken from a different
    anchor's run is a number, not a floor, and a candidate that reduced work can
    clear it.
    """


class GateCoverageGap(CorrectnessError):
    """A T0 report would omit, duplicate, or invent a gate.

    Denial 6: *"A controller that discovers a coverage gap in its evaluator
    RECORDS the gap … it does not patch the instrument."* This raise is how the
    gap is discovered at all: the report cannot be built while it exists.
    """


class ActorDeclaredScope(CorrectnessError):
    """The actor tried to declare a surface not applicable.

    Invariant 18: *"The affected-surface manifest is mechanically derived and
    dynamically confirmed; the actor's declaration is a scored prediction, never
    a scope input."* An actor that can mark its own risky surface "n/a" has
    delegated itself the gate.
    """


# =============================================================================
# Vocabularies
# =============================================================================

#: The two ops that are required on EVERY candidate, whatever the change class.
#:
#: `MUL_MAT` was the only op `kernel_eval.sh` tested. `MUL_MAT_ID` is the MoE
#: expert-routing gemm; the production general/worker role is a MoE model, so a
#: kernel change validated on `MUL_MAT` alone leaves the op that runs on every
#: token of the production workload completely unexercised. Adding it is the
#: single concrete coverage fix this module carries, so it is a constant and
#: `T0Policy` refuses a policy that drops it.
MANDATORY_BACKEND_OPS = ("MUL_MAT", "MUL_MAT_ID")

#: §15.2's five controls, plus `None` for an ordinary candidate. The role changes
#: what binary identity MEANS: control 4 (A/A) measures the anchor against
#: itself, so an identical binary is REQUIRED there and forbidden everywhere else.
CONTROL_ROLES = ("positive", "neutral", "degraded_negative", "aa", "historical_replay")

#: Who produced a piece of evidence. Only the evaluator may produce a correctness
#: verdict: *"Correctness verdicts are produced by the evaluator against declared
#: oracles and are NEVER self-reported by the candidate."*
#:
#: The tuple itself now lives in `schemas.py`, because `schemas.require.producer`
#: is the type of every `produced_by` field and the type and its domain may not
#: be two objects that can disagree. Re-exported under this name because it is
#: what `t0_provider` and the conformance suite already read.
EVIDENCE_PRODUCERS = schemas.EVIDENCE_PRODUCERS

COHERENCE_BYTE_IDENTICAL = "byte_identical"
COHERENCE_WITHIN_TOLERANCE = "equivalent_within_declared_tolerance"
COHERENCE_DIVERGENT = "divergent"
COHERENCE_EMPTY = "empty_generation"
COHERENCE_UNDECIDABLE = "undecidable_under_sampling"
COHERENCE_NOT_COMPARED = "not_compared"

COHERENCE_LABELS = (
    COHERENCE_BYTE_IDENTICAL, COHERENCE_WITHIN_TOLERANCE, COHERENCE_DIVERGENT,
    COHERENCE_EMPTY, COHERENCE_UNDECIDABLE, COHERENCE_NOT_COMPARED,
)

#: The labels that ASSERT equivalence. These are the ones that may never be
#: produced without a bound anchor — the `COH="coherent"` class of statement.
COHERENCE_EQUIVALENCE_LABELS = (COHERENCE_BYTE_IDENTICAL, COHERENCE_WITHIN_TOLERANCE)

#: Who may declare a surface not applicable. The actor is deliberately absent.
NA_SOURCES = ("static_derivation", "operator_waiver")

#: *"Cache state is declared in every record."* `unknown` is in the vocabulary so
#: that "we did not record it" is SAYABLE — and it is a COULD_NOT_CHECK, never an
#: empty field that reads as cold. `served_from_cache` is control 3's shape and
#: FAILs.
CACHE_STATES = (
    "cold", "warm_page_cache", "warm_kv_cache", "served_from_cache", "unknown",
)

#: §10.6's marker. Above the backend adapter's ceiling, or on any `core_header`
#: change, the candidate carries this into its release package regardless of how
#: clean T0 was.
REQUIRES_HUMAN_CODE_REVIEW = "REQUIRES_HUMAN_CODE_REVIEW"

# Mirrors `schemas._PRODUCTION_BRANCH_RE`. Duplicated deliberately and named as a
# mirror (the same thing `api.py` does for `_SHA256_RE`) rather than reaching into
# another module's private.
_PRODUCTION_BRANCH_RE = re.compile(r"^production-(consolidated|speech)-v\d+$")

#: Path prefixes that are production kernel trees. A diff or a build output that
#: names one violates invariant 3 ("no actor builds in or modifies any production
#: tree") and denial 2 ("No production write of any kind").
PRODUCTION_TREE_ROOTS = (
    "/mnt/raid0/llm/llama.cpp",
    "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
    "/workspace/repos/epyc-llama",
)


# =============================================================================
# Small validators — the scalar ones are `schemas.require`, the shaped ones are
# local because their domain is this module's and nobody else's
# =============================================================================
#
# These eight names used to be eight bodies. They are kept as names, and only as
# names, because ~40 call sites read `_req_sha256(...)` and renaming them would
# bury the one change that matters in a rename diff. The BODY is now the field
# type in `schemas.require`, which is the only place the placeholder-digest
# predicate lives — see the `require` header in `schemas.py` for why a ninth copy
# of `re.compile(r"^[0-9a-f]{64}$")` is now a test failure rather than a habit.

_req_str = schemas.require.str
_req_sha256 = schemas.require.sha256
_req_commit = schemas.require.commit
_req_bool = schemas.require.bool
_req_int = schemas.require.int
_req_tuple = schemas.require.tuple
_req_producer = schemas.require.producer


def _opt_sha256(value: Any, label: str) -> Optional[str]:
    return None if value is None else _req_sha256(value, label)


def _opt_commit(value: Any, label: str) -> Optional[str]:
    return None if value is None else _req_commit(value, label)


def _opt_bool(value: Any, label: str) -> Optional[bool]:
    """`None` means NOT DETERMINED. It is never coerced to False."""
    if value is None:
        return None
    return _req_bool(value, label)


def _opt_int(value: Any, label: str, *, minimum: int = 0) -> Optional[int]:
    return None if value is None else _req_int(value, label, minimum=minimum)


def _req_ratio(value: Any, label: str) -> float:
    """A POLICY ratio: a floor or a reject threshold, which zero would disable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number, got {value!r}")
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(f"{label}: expected a finite ratio in (0, 1], got {value!r}")
    return float(value)


def _req_observed_ratio(value: Any, label: str) -> float:
    """An OBSERVED ratio, on the CLOSED unit interval.

    Zero is a measurement, not a malformed one: a candidate whose output agrees
    with the anchor on no token at all has an agreement ratio of exactly 0.0, and
    that is the strongest evidence of divergence there is. Validating it against a
    policy floor's `(0, 1]` domain raised on the real observation and forced the
    provider to send `None` instead, which reads downstream as *"no token
    agreement ratio was measured"* — a measured extreme silently rewritten as a
    missing measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number, got {value!r}")
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{label}: expected a finite ratio in [0, 1], got {value!r}")
    return float(value)


def _validate_anchor_triple(*, source_commit: Any, binary_sha256: Any, linkage_sha256: Any,
                            label: str) -> None:
    """Precondition 4's THREE components on a piece of evidence: all, or none.

    *"names its anchor by source commit, binary SHA-256, and linkage SHA-256"* —
    the same rule `schemas._check_anchor_block_v3` enforces on the record, applied
    where the evidence that FEEDS the record is captured, so the two cannot say
    different things about how completely the anchor was named.

    Two refusals, and both are about a name that resolves to more than one thing:

      * **Partial naming.** Two of three components is not a weaker name, it is a
        different one: a rebuilt binary at the same commit and a binary rebuilt
        from a different commit are both admissible under a two-component name.
        Absent is a state this module can express (all three `None`, which reads
        as COULD_NOT_CHECK downstream); partially named is not.
      * **Placeholders.** `schemas.is_placeholder_digest` — `0`*64 in an anchor
        field is a CLAIM that an anchor was resolved, and every downstream reader
        takes it for one. Evidence with no anchor records none; it never fills the
        fields in.
    """
    _opt_commit(source_commit, f"{label}.anchor_source_commit")
    _opt_sha256(binary_sha256, f"{label}.anchor_binary_sha256")
    _opt_sha256(linkage_sha256, f"{label}.anchor_linkage_sha256")
    named = {
        f"{label}.anchor_source_commit": source_commit,
        f"{label}.anchor_binary_sha256": binary_sha256,
        f"{label}.anchor_linkage_sha256": linkage_sha256,
    }
    present = sorted(name for name, value in named.items() if value is not None)
    if present and len(present) != len(named):
        missing = sorted(name for name, value in named.items() if value is None)
        raise ValueError(
            f"{label}: an anchor is named by source commit, binary SHA-256 AND linkage "
            f"SHA-256 (precondition 4). This evidence names {present} and omits {missing}. "
            "A partially named anchor is the defect: it resolves to more than one artifact, "
            "so record all three components or none of them")
    for name, value in sorted(named.items()):
        if schemas.is_placeholder_digest(value):
            raise ValueError(
                f"{name}: {value!r} is a placeholder digest, not a measured identity. "
                "Evidence that compared against no anchor omits all three components; it "
                "never fills them in — a fabricated identity reads as a resolved one to "
                "every downstream reader, which is strictly worse than an absent one")


def _recorded_anchor(evidence: Any) -> Optional[api.AnchorIdentity]:
    """The anchor a piece of evidence was captured AGAINST, or `None` if unrecorded.

    One implementation for every evidence type that carries the triple, because
    four hand-copied accessors drift and the one that drifts is whichever has
    fewer tests. `_validate_anchor_triple` has already refused a partial naming,
    so a present `anchor_source_commit` guarantees the other two are present too.

    `None` is not "any anchor" and not "the current one": it is the absence of a
    record, and no consumer may read it as agreement.
    """
    if evidence.anchor_source_commit is None:
        return None
    return api.AnchorIdentity(
        source_commit=evidence.anchor_source_commit,
        binary_sha256=evidence.anchor_binary_sha256,
        linkage_sha256=evidence.anchor_linkage_sha256,
    )


def _refuse_replay_mismatch(anchor: Optional[api.AnchorIdentity],
                            recorded: Optional[api.AnchorIdentity],
                            *, label: str, consequence: str, error: type) -> None:
    """Refuse evidence captured against a DIFFERENT anchor than it is read against.

    The comparison is `api.AnchorIdentity.identity_matches` — all three
    components, the same comparator the record uses — and there are three
    outcomes, no fourth:

      * **PASS** — the capture and the consumer name one anchor. Returns; the
        caller proceeds to derive whatever it derives.
      * **FAIL** — they name different anchors. RAISES `error`. Not a degraded
        gate result: a mismatch is a defect in the REPLAY, and a COULD_NOT_CHECK
        would file it as a property of the evidence and leave the bug in place.
      * **COULD_NOT_CHECK** — one side recorded nothing (`anchor is None`, or
        evidence predating the field). Returns WITHOUT raising, because nothing
        disagrees. Silence here is not agreement: each caller separately routes an
        unrecorded identity to COULD_NOT_CHECK, and this function's return value
        must never be read as "the identities match".
    """
    if anchor is None or recorded is None:
        return
    match = anchor.identity_matches(recorded)
    if match.outcome == schemas.FAIL:
        raise error(
            f"this {label} capture was taken against anchor {recorded.short()} "
            f"and is being compared against anchor {anchor.short()}: "
            + "; ".join(match.reasons) + ". " + consequence
        )


def _fail(*reasons: str) -> schemas.Check:
    return schemas.Check(schemas.FAIL, tuple(reasons))


def _cnc(*reasons: str) -> schemas.Check:
    return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))


def _verdict(reasons: Sequence[str]) -> schemas.Check:
    """FAIL if anything was found, PASS otherwise. Never a soft pass."""
    return _fail(*reasons) if reasons else schemas.Check(schemas.PASS)


def _under_production_tree(path: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in PRODUCTION_TREE_ROOTS)


@functools.lru_cache(maxsize=1)
def _module_source_sha256() -> str:
    """SHA-256 of this module's own source — the recipe constructor's identity.

    RAISES if the source cannot be read. A constructor that cannot name itself
    cannot emit a receipt, and a receipt with a fabricated constructor hash is
    worse than no receipt (precondition 6).
    """
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except OSError as exc:
        raise CorrectnessError(
            f"cannot read {__file__} to compute the recipe constructor's content hash: "
            f"{exc}. Precondition 6 requires the constructor's identifier AND content "
            "hash on the record; there is no fallback value for it."
        ) from exc
    return schemas.content_hash({"module": "autokernel.evaluator.correctness", "source": source})


# =============================================================================
# Policy and applicability — inputs that live in the evaluator bundle
# =============================================================================

@dataclass(frozen=True)
class DiffComplexityCeiling:
    """§10.6. Above this, the package is marked `REQUIRES_HUMAN_CODE_REVIEW`.

    This is NOT a T0 failure and is deliberately not modelled as one: §10.6 says
    the package *"is marked … and says so on its first page"*, which is a review
    obligation on the release side, not a statement that the kernel is wrong.
    The §8.5.1 change-class size envelope is the one that FAILs — see
    `ChangeClassEnvelope`. Conflating the two would either let a huge diff ship
    unreviewed or fail a legitimate large-but-declared change at T0.
    """

    backend: str
    max_changed_lines: int
    max_files_touched: int
    shared_core_forces_review: bool

    def __post_init__(self) -> None:
        _req_str(self.backend, "diff_ceiling.backend")
        _req_int(self.max_changed_lines, "diff_ceiling.max_changed_lines", minimum=1)
        _req_int(self.max_files_touched, "diff_ceiling.max_files_touched", minimum=1)
        _req_bool(self.shared_core_forces_review, "diff_ceiling.shared_core_forces_review")


@dataclass(frozen=True)
class ChangeClassEnvelope:
    """§8.5.1 item 3: the diff must *"stay inside the change-class size envelope"*.

    Breaching this IS a T0 failure, because invariant 13 ("one conceptual mutation
    per proposal") is what keeps a candidate falsifiable and revertible, and a
    diff that has outgrown its class is no longer one mutation.
    """

    change_class: str
    max_changed_lines: int
    max_files_touched: int

    def __post_init__(self) -> None:
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise ValueError(
                f"envelope.change_class: {self.change_class!r} is not one of "
                f"{sorted(schemas.CHANGE_CLASSES)}")
        _req_int(self.max_changed_lines, "envelope.max_changed_lines", minimum=1)
        _req_int(self.max_files_touched, "envelope.max_files_touched", minimum=1)


@dataclass(frozen=True)
class NotApplicable:
    """A surface declared not applicable, by someone entitled to declare it.

    Invariant 18 is enforced structurally: `source` may be `static_derivation`
    (the mechanical derivation found nothing on this surface) or
    `operator_waiver` (a human took the call, on the record). It may NOT be the
    actor, and an attempt raises `ActorDeclaredScope` rather than being silently
    downgraded — an actor that can mark its own surface n/a has granted itself
    the gate.
    """

    reason: str
    source: str
    ref: str

    def __post_init__(self) -> None:
        _req_str(self.reason, "not_applicable.reason")
        if self.source not in NA_SOURCES:
            raise ActorDeclaredScope(
                f"not_applicable.source {self.source!r} is not one of {list(NA_SOURCES)}. "
                "The actor's declaration is a scored prediction, never a scope input "
                "(invariant 18); a surface is not applicable because the derivation says "
                "so or because the operator waived it, on the record."
            )
        _req_str(self.ref, "not_applicable.ref")

    def note(self) -> str:
        return f"not applicable ({self.source}={self.ref}): {self.reason}"


@dataclass(frozen=True)
class T0Policy:
    """The T0 gate's own parameters. Part of the evaluator bundle, read-only here.

    Denial 6: *"The controller MUST NOT modify this protocol, the evaluator
    bundle, the control definitions, the campaign objective, the calibrated
    thresholds, or any scoring contract."* Nothing in this module writes a policy;
    it is handed one, and it refuses a policy that would reproduce the defect it
    replaces.

    There are NO defaults. A campaign that has not decided its required op set,
    its symbol-shrinkage reject ratio, or its determinism repeat count has not
    decided what T0 means, and a default would decide it silently.
    """

    required_backend_ops: tuple
    symbol_shrinkage_reject_ratio: float
    diff_ceiling: DiffComplexityCeiling
    determinism_min_runs: int
    coherence_tolerance_floor: Optional[float]
    policy_ref: str

    def __post_init__(self) -> None:
        ops = _req_tuple(self.required_backend_ops, "policy.required_backend_ops")
        for op in ops:
            _req_str(op, "policy.required_backend_ops[]")
        missing = [op for op in MANDATORY_BACKEND_OPS if op not in ops]
        if missing:
            raise ValueError(
                f"policy.required_backend_ops omits {missing}. `kernel_eval.sh` ran "
                "`test-backend-ops -o MUL_MAT` and nothing else, so every MoE expert path "
                "(MUL_MAT_ID) went unexercised on a host whose production worker is a MoE "
                f"model. {list(MANDATORY_BACKEND_OPS)} are required on every candidate and "
                "this policy may not drop them."
            )
        if len(set(ops)) != len(ops):
            raise ValueError("policy.required_backend_ops contains duplicates")
        _req_ratio(self.symbol_shrinkage_reject_ratio, "policy.symbol_shrinkage_reject_ratio")
        if not isinstance(self.diff_ceiling, DiffComplexityCeiling):
            raise TypeError("policy.diff_ceiling must be a DiffComplexityCeiling")
        _req_int(self.determinism_min_runs, "policy.determinism_min_runs", minimum=2)
        if self.coherence_tolerance_floor is not None:
            _req_ratio(self.coherence_tolerance_floor, "policy.coherence_tolerance_floor")
        _req_str(self.policy_ref, "policy.policy_ref")

    def to_dict(self) -> dict:
        return {
            "required_backend_ops": list(self.required_backend_ops),
            "symbol_shrinkage_reject_ratio": self.symbol_shrinkage_reject_ratio,
            "diff_ceiling": {
                "backend": self.diff_ceiling.backend,
                "max_changed_lines": self.diff_ceiling.max_changed_lines,
                "max_files_touched": self.diff_ceiling.max_files_touched,
                "shared_core_forces_review": self.diff_ceiling.shared_core_forces_review,
            },
            "determinism_min_runs": self.determinism_min_runs,
            "coherence_tolerance_floor": self.coherence_tolerance_floor,
            "policy_ref": self.policy_ref,
        }


# =============================================================================
# Evidence types — the seam between "something ran" and "a verdict was computed"
# =============================================================================

@dataclass(frozen=True)
class ChangeSurface:
    """§6.4 item 1: the MECHANICALLY DERIVED surface, plus the actor's prediction.

    `derived_*` fields drive every decision. `declared_*` fields are retained and
    scored and drive nothing (invariant 18). A `derived_*` field of `None` means
    the derivation did not determine it — which is NOT False. Where a surface's
    applicability turns on it, `None` fails closed to `COULD_NOT_CHECK`, because
    "we could not tell whether this change touches memory" is not "it does not".
    """

    derived_touches_memory: Optional[bool]
    derived_touches_threading: Optional[bool]
    derived_touches_dispatch: Optional[bool]
    derived_touches_persistent_state: Optional[bool]
    derived_ops: tuple
    derived_files: tuple
    declared_touches_memory: Optional[bool]
    declared_touches_threading: Optional[bool]
    declared_ops: tuple
    touches_shared_core_header: bool
    derivation_ref: str

    def __post_init__(self) -> None:
        for name in ("derived_touches_memory", "derived_touches_threading",
                     "derived_touches_dispatch", "derived_touches_persistent_state",
                     "declared_touches_memory", "declared_touches_threading"):
            _opt_bool(getattr(self, name), f"change_surface.{name}")
        for name in ("derived_ops", "derived_files", "declared_ops"):
            for item in _req_tuple(getattr(self, name), f"change_surface.{name}"):
                _req_str(item, f"change_surface.{name}[]")
        _req_bool(self.touches_shared_core_header, "change_surface.touches_shared_core_header")
        _req_str(self.derivation_ref, "change_surface.derivation_ref")

    @property
    def sanitizers_mandatory(self) -> Optional[bool]:
        """True / False / None (undetermined). Never guesses."""
        mem, thr = self.derived_touches_memory, self.derived_touches_threading
        if mem is True or thr is True:
            return True
        if mem is None or thr is None:
            return None
        return False

    def prediction_score(self) -> tuple:
        """The actor's declaration scored against the derivation. NEVER a gate.

        Invariant 18: the declaration is *"a scored prediction, never a scope
        input"*. Every consumer of this method must treat it as telemetry about
        the planner's accuracy; nothing in this module reads it back.

        Rows are `(field, declared, derived, hit)`.
        """
        rows = []
        for name in ("touches_memory", "touches_threading"):
            declared = getattr(self, f"declared_{name}")
            derived = getattr(self, f"derived_{name}")
            rows.append((name, declared, derived, declared == derived))
        declared_ops, derived_ops = set(self.declared_ops), set(self.derived_ops)
        missed = tuple(sorted(derived_ops - declared_ops))
        over = tuple(sorted(declared_ops - derived_ops))
        rows.append(("ops_missed_by_actor", missed, tuple(sorted(derived_ops)), not missed))
        rows.append(("ops_over_declared", over, tuple(sorted(derived_ops)), not over))
        return tuple(rows)


@dataclass(frozen=True)
class SymbolTableDiff:
    """§8.5.1 item 1 — the C++ analogue of AutoPilot's public-name preservation.

    *"Any removal or arity change not declared in the proposal is a hard
    failure."* A dropped template specialization, a deleted dispatch case, or a
    removed op registration compiles cleanly and silently changes behaviour for
    every shape nobody happened to test.
    """

    removed_symbols: tuple
    arity_changed_symbols: tuple
    added_symbols: tuple
    removed_op_registrations: tuple
    removed_dispatch_predicates: tuple
    declared_removals: tuple
    anchor_symbol_count: int
    candidate_symbol_count: int
    tool_id: str
    receipt_ref: str
    produced_by: str
    #: Registrations present on BOTH sides whose declared arity changed —
    #: `GGML_CPU_OP(MUL_MAT, 2)` becoming `GGML_CPU_OP(MUL_MAT, 5)`. Added
    #: 2026-08-04: `integrity.RegistrationDiff` has produced this since it was
    #: written and this record had no field for it, so `chain.symbol_evidence`
    #: could only report it in a `checks` tuple that no gate reads. §8.5.1 makes
    #: an arity change hard for a SYMBOL and there is no reading under which a
    #: registration's arity is softer: an op registered with the wrong operand
    #: count dispatches wrong for every shape, and it compiles.
    #:
    #: Defaulted to `()` so records written before the field existed stay
    #: constructible — and `()` is honest for them only because the projection
    #: that fills it is the same one that produced them.
    arity_changed_op_registrations: tuple = ()

    def __post_init__(self) -> None:
        for name in ("removed_symbols", "arity_changed_symbols", "added_symbols",
                     "removed_op_registrations", "removed_dispatch_predicates",
                     "declared_removals", "arity_changed_op_registrations"):
            for item in _req_tuple(getattr(self, name), f"symbols.{name}"):
                _req_str(item, f"symbols.{name}[]")
        _req_int(self.anchor_symbol_count, "symbols.anchor_symbol_count", minimum=1)
        _req_int(self.candidate_symbol_count, "symbols.candidate_symbol_count", minimum=0)
        _req_str(self.tool_id, "symbols.tool_id")
        _req_str(self.receipt_ref, "symbols.receipt_ref")
        _req_producer(self.produced_by, "symbols.produced_by")


@dataclass(frozen=True)
class BuildProvenance:
    """§8.5.1 item 2 — *"clean build from the recorded snapshot"*, never incremental.

    *"An incremental build can link stale objects and hide the error that the
    snapshot would surface, which would make the actor's build state part of the
    artifact."*
    """

    built_from_snapshot_sha256: str
    build_dir: str
    build_dir_was_fresh: bool
    incremental_objects_present: bool
    compiler_id: str
    compiler_version: str
    build_log_ref: str
    production_tree_paths_touched: tuple
    output_binary_sha256: str
    #: WHO produced this record. Added 2026-08-04: eleven of this file's evidence
    #: types have carried `produced_by` since they were written and this one did
    #: not, so §8.5.1 item 2 — the clean-build claim, and the binary identity that
    #: rides on it — was the one piece of T0 evidence an ACTOR could hand in
    #: unattributed. `SymbolTableDiff` twenty lines above validates it the same
    #: way; `check_symbol_and_registration_preservation` reads it and FAILs a
    #: self-report. This field is what makes that reading possible here.
    produced_by: str
    #: True only when the build receipt proves compiler warnings were fatal.
    #: Static analysis may then compare clean compiler exits without needing a
    #: historical warning count from an anchor build log that no longer exists.
    warnings_as_errors: bool = False

    def __post_init__(self) -> None:
        _req_sha256(self.built_from_snapshot_sha256, "build.built_from_snapshot_sha256")
        _req_str(self.build_dir, "build.build_dir")
        _req_bool(self.build_dir_was_fresh, "build.build_dir_was_fresh")
        _req_bool(self.incremental_objects_present, "build.incremental_objects_present")
        _req_str(self.compiler_id, "build.compiler_id")
        _req_str(self.compiler_version, "build.compiler_version")
        _req_str(self.build_log_ref, "build.build_log_ref")
        for item in _req_tuple(self.production_tree_paths_touched,
                               "build.production_tree_paths_touched"):
            _req_str(item, "build.production_tree_paths_touched[]")
        _req_sha256(self.output_binary_sha256, "build.output_binary_sha256")
        _req_producer(self.produced_by, "build.produced_by")
        _req_bool(self.warnings_as_errors, "build.warnings_as_errors")


@dataclass(frozen=True)
class DiffPolicyEvidence:
    """§8.5.1 item 3 + §10.6 + the record-schema half of §8.6's "schema and diff policy"."""

    files_touched: tuple
    declared_surface_files: tuple
    unrelated_deletions: tuple
    changed_lines: int
    change_class: str
    envelope: ChangeClassEnvelope
    branch_name: str
    commit_was_pathspec_limited: bool
    production_tree_paths: tuple
    record_schema_violations: tuple
    diff_ref: str
    #: WHO produced this record. Added 2026-08-04, for the same reason as
    #: `BuildProvenance.produced_by`: §8.5.1 item 3 is a claim about what the diff
    #: touched, and `files_touched` / `unrelated_deletions` / `production_tree_paths`
    #: are exactly the fields an actor benefits from getting wrong. Without this
    #: field the gate cannot tell a derived diff from a declared one.
    produced_by: str

    def __post_init__(self) -> None:
        for name in ("files_touched", "declared_surface_files", "unrelated_deletions",
                     "production_tree_paths", "record_schema_violations"):
            for item in _req_tuple(getattr(self, name), f"diff.{name}"):
                _req_str(item, f"diff.{name}[]")
        _req_int(self.changed_lines, "diff.changed_lines", minimum=0)
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise ValueError(f"diff.change_class: {self.change_class!r} is not one of "
                             f"{sorted(schemas.CHANGE_CLASSES)}")
        if not isinstance(self.envelope, ChangeClassEnvelope):
            raise TypeError("diff.envelope must be a ChangeClassEnvelope")
        if self.envelope.change_class != self.change_class:
            raise ValueError(
                f"diff.envelope is for change class {self.envelope.change_class!r} but the "
                f"diff declares {self.change_class!r}; the envelope of a different class is "
                "not a bound on this one")
        _req_str(self.branch_name, "diff.branch_name")
        _req_bool(self.commit_was_pathspec_limited, "diff.commit_was_pathspec_limited")
        _req_str(self.diff_ref, "diff.diff_ref")
        _req_producer(self.produced_by, "diff.produced_by")


@dataclass(frozen=True)
class StaticAnalysisEvidence:
    """§8.6 "static/compile checks", plus toolchain identity against the anchor.

    A candidate built with a different compiler than the anchor is not a kernel
    comparison; it is a toolchain comparison wearing one. That is a confound, so
    it is checked here rather than discovered in the speed number.

    Three of its fields are the ANCHOR's — `anchor_compiler_id`,
    `anchor_compiler_version`, `anchor_warning_count` — so the capture also
    records WHICH anchor they were read from, as a triple: all three components or
    none (`_validate_anchor_triple`). Without it the confound this gate exists to
    catch can arrive through the gate itself, by comparing the candidate's
    toolchain against a toolchain belonging to some other anchor's build.
    """

    compiler_id: str
    compiler_version: str
    anchor_compiler_id: str
    anchor_compiler_version: str
    error_count: int
    warning_count: int
    anchor_warning_count: Optional[int]
    anchor_source_commit: Optional[str]
    anchor_binary_sha256: Optional[str]
    anchor_linkage_sha256: Optional[str]
    warnings_as_errors: bool
    analyzer_id: Optional[str]
    analyzer_error_findings: tuple
    receipt_ref: str
    produced_by: str

    def recorded_anchor(self) -> Optional[api.AnchorIdentity]:
        """The anchor whose toolchain this capture read, or `None` if unrecorded."""
        return _recorded_anchor(self)

    def __post_init__(self) -> None:
        for name in ("compiler_id", "compiler_version", "anchor_compiler_id",
                     "anchor_compiler_version", "receipt_ref"):
            _req_str(getattr(self, name), f"static.{name}")
        _req_int(self.error_count, "static.error_count")
        _req_int(self.warning_count, "static.warning_count")
        _opt_int(self.anchor_warning_count, "static.anchor_warning_count")
        _validate_anchor_triple(
            source_commit=self.anchor_source_commit,
            binary_sha256=self.anchor_binary_sha256,
            linkage_sha256=self.anchor_linkage_sha256,
            label="static")
        _req_bool(self.warnings_as_errors, "static.warnings_as_errors")
        if self.analyzer_id is not None:
            _req_str(self.analyzer_id, "static.analyzer_id")
        for item in _req_tuple(self.analyzer_error_findings, "static.analyzer_error_findings"):
            _req_str(item, "static.analyzer_error_findings[]")
        _req_producer(self.produced_by, "static.produced_by")


# -----------------------------------------------------------------------------
# ASAN / UBSAN — the invocation is CONSTRUCTED here and RUN elsewhere
# -----------------------------------------------------------------------------

#: Compile flags the sanitizer build must carry. `-fno-sanitize-recover=all` is
#: the load-bearing one: without it UBSAN prints and CONTINUES, which is a
#: fail-open sanitizer — the run finishes "successfully" with the diagnostic
#: buried in a log nobody gates on.
_SANITIZER_COMPILE_FLAGS = (
    "-fsanitize=address,undefined",
    "-fno-sanitize-recover=all",
    "-fno-omit-frame-pointer",
    "-g",
)

#: CLAUDE.md / `feedback_no_core_dumps`: **NEVER core dumps; GDB/ASAN.** A
#: multi-GB core from a GPU process on a shared 3.7 TB host is both a storage
#: event and a privacy event, and ASAN's own stack trace is strictly more useful.
#: `disable_coredump=1` is therefore not a preference, and
#: `check_sanitizer_invocation` FAILs without it.
_ASAN_OPTIONS = (
    "abort_on_error=0",
    "disable_coredump=1",
    "detect_leaks=1",
    "detect_stack_use_after_return=1",
    "strict_string_checks=1",
    "print_stacktrace=1",
    "symbolize=1",
)

_UBSAN_OPTIONS = (
    "halt_on_error=1",
    "print_stacktrace=1",
    "symbolize=1",
)

#: Substrings that would re-enable a core dump. Deliberately NOT `"coredump=1"`,
#: which is a substring of the compliant `"disable_coredump=1"` — a guard that
#: forbids its own compliant idiom fails the run it was meant to protect.
_CORE_DUMP_TOKENS = ("ulimit -c", "core_pattern", "disable_coredump=0")


@dataclass(frozen=True)
class SanitizerInvocation:
    """What an ASAN/UBSAN build and targeted run WOULD be. Nothing here runs it.

    argv is a tuple of tuples and never a shell string: a module that cannot
    render a command line cannot have a quoting bug, and `shlex` is one of the
    imports this module's AST audit forbids.
    """

    constructor_id: str
    configure_argv: tuple
    build_argv: tuple
    run_argv: tuple
    env: tuple
    receipt: api.RecipeReceipt
    notes: tuple

    def __post_init__(self) -> None:
        _req_str(self.constructor_id, "sanitizer.constructor_id")
        for name in ("configure_argv", "build_argv", "run_argv"):
            argv = _req_tuple(getattr(self, name), f"sanitizer.{name}")
            if not argv:
                raise ValueError(f"sanitizer.{name} must be non-empty")
            for item in argv:
                _req_str(item, f"sanitizer.{name}[]")
        for pair in _req_tuple(self.env, "sanitizer.env"):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise TypeError("sanitizer.env must be a tuple of (name, value) pairs")
            _req_str(pair[0], "sanitizer.env key")
            _req_str(pair[1], "sanitizer.env value")
        for note in _req_tuple(self.notes, "sanitizer.notes"):
            _req_str(note, "sanitizer.notes[]")
        if not isinstance(self.receipt, api.RecipeReceipt):
            raise TypeError("sanitizer.receipt must be an api.RecipeReceipt")

    def env_value(self, name: str) -> Optional[str]:
        for key, value in self.env:
            if key == name:
                return value
        return None

    def to_dict(self) -> dict:
        return {
            "constructor_id": self.constructor_id,
            "configure_argv": list(self.configure_argv),
            "build_argv": list(self.build_argv),
            "run_argv": list(self.run_argv),
            "env": [list(pair) for pair in self.env],
            "receipt": self.receipt.to_dict(),
            "notes": list(self.notes),
        }


def build_sanitizer_invocation(*,
                               source_dir: str,
                               build_dir: str,
                               target: str,
                               run_argv: Sequence[str],
                               jobs: int,
                               backend: str) -> SanitizerInvocation:
    """Construct the mandatory ASAN/UBSAN invocation and its receipt. Runs nothing.

    §8.6: *"mandatory ASAN/UBSAN build and targeted run for any change touching
    memory or threading."* This is the constructor half of precondition 6 for
    that build — the argv is emitted by code with a content hash, never typed.

    Refuses a production tree outright: invariant 3 and denial 2 forbid building
    in one, and a sanitizer build is still a build.
    """
    _req_str(source_dir, "source_dir")
    _req_str(build_dir, "build_dir")
    _req_str(target, "target")
    _req_str(backend, "backend")
    _req_int(jobs, "jobs", minimum=1)
    run = tuple(run_argv)
    if not run:
        raise ValueError(
            "run_argv is empty: a sanitizer BUILD with no targeted RUN proves nothing. "
            "§8.6 requires the build AND the targeted run.")
    for item in run:
        _req_str(item, "run_argv[]")
    for path, label in ((source_dir, "source_dir"), (build_dir, "build_dir")):
        if _under_production_tree(path):
            raise CorrectnessError(
                f"{label} {path!r} is inside a production kernel tree. Production kernels are "
                "FROZEN; no actor builds in or modifies one (invariant 3, denial 2). A "
                "sanitizer build is still a build."
            )

    flags = " ".join(_SANITIZER_COMPILE_FLAGS)
    configure = (
        "cmake", "-S", source_dir, "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_C_FLAGS={flags}",
        f"-DCMAKE_CXX_FLAGS={flags}",
        f"-DCMAKE_EXE_LINKER_FLAGS={flags}",
    )
    build = ("cmake", "--build", build_dir, "--target", target, "-j", str(jobs))
    env = (
        ("ASAN_OPTIONS", ":".join(_ASAN_OPTIONS)),
        ("UBSAN_OPTIONS", ":".join(_UBSAN_OPTIONS)),
    )
    constructor_id = f"ak.t0.sanitizer.{backend}/v1"
    argv_hash = schemas.content_hash({
        "configure_argv": list(configure),
        "build_argv": list(build),
        "run_argv": list(run),
        "env": [list(pair) for pair in env],
    })
    receipt = api.RecipeReceipt(
        constructor_id=constructor_id,
        constructor_sha256=_module_source_sha256(),
        argv_sha256=argv_hash,
    )
    return SanitizerInvocation(
        constructor_id=constructor_id,
        configure_argv=configure,
        build_argv=build,
        run_argv=run,
        env=env,
        receipt=receipt,
        notes=(
            "constructed, NOT executed: this module runs no build",
            "core dumps are disabled by policy (CLAUDE.md: NEVER core dumps; GDB/ASAN)",
            "-fno-sanitize-recover=all: a sanitizer that continues past an error is "
            "fail-open",
        ),
    )


def _compile_flag_tokens(argv: Sequence[str]) -> frozenset:
    """The set of compile-flag TOKENS an argv actually carries.

    Substring matching against `" ".join(argv)` is not a flag check: `-g` is a
    substring of a build directory named `.../build-gpu`, so an invocation that
    carries no debug-info flag at all "contains" one, and
    `-fsanitize=address,undefined` present only in `CMAKE_EXE_LINKER_FLAGS` reads
    as though the compiler got it. Exact token equality is the check; a `-D<NAME>=`
    prefix is stripped so the flags cmake carries inside `-DCMAKE_C_FLAGS=...`
    count as the tokens they are, and the constructed invocation's own idiom keeps
    passing.
    """
    tokens = set()
    for item in argv:
        for chunk in item.split():
            tokens.add(chunk)
            if chunk.startswith("-D") and "=" in chunk:
                tokens.add(chunk.split("=", 1)[1])
    return frozenset(tokens)


def check_sanitizer_invocation(invocation: Any) -> schemas.Check:
    """FAIL an invocation that would be fail-open, or that would dump core.

    Checked rather than assumed, because a hand-edited invocation that drops
    `-fno-sanitize-recover=all` still *runs*, still exits 0, and still produces a
    log — it just stops being a gate.
    """
    if not isinstance(invocation, SanitizerInvocation):
        return _cnc(f"no sanitizer invocation to inspect (got {type(invocation).__name__})")

    reasons = []
    configure_tokens = _compile_flag_tokens(invocation.configure_argv)
    for flag in _SANITIZER_COMPILE_FLAGS:
        if flag not in configure_tokens:
            reasons.append(f"configure argv is missing the required flag {flag!r}")
    if any(token.startswith("-fsanitize-recover") for token in configure_tokens) \
            and "-fno-sanitize-recover=all" not in configure_tokens:
        reasons.append("configure argv enables sanitizer recovery; a sanitizer that prints "
                       "and continues is fail-open")

    asan = invocation.env_value("ASAN_OPTIONS")
    ubsan = invocation.env_value("UBSAN_OPTIONS")
    if asan is None:
        reasons.append("ASAN_OPTIONS is not set; the runtime's platform defaults are not a "
                       "declared configuration")
    else:
        opts = tuple(part.strip() for part in asan.split(":"))
        if "disable_coredump=1" not in opts:
            reasons.append("ASAN_OPTIONS does not set disable_coredump=1; this project's "
                           "standing rule is NEVER core dumps, use GDB/ASAN")
        if "abort_on_error=1" in opts and "disable_coredump=1" not in opts:
            reasons.append("ASAN_OPTIONS aborts on error without disabling core dumps")
        if not any(o.startswith("detect_leaks=") for o in opts):
            reasons.append("ASAN_OPTIONS does not state detect_leaks explicitly")
        if "print_stacktrace=1" not in opts:
            reasons.append("ASAN_OPTIONS does not set print_stacktrace=1; a finding without a "
                           "stack is not actionable and invites a core dump instead")
    if ubsan is None:
        reasons.append("UBSAN_OPTIONS is not set")
    else:
        uopts = tuple(part.strip() for part in ubsan.split(":"))
        if "halt_on_error=1" not in uopts:
            reasons.append("UBSAN_OPTIONS does not set halt_on_error=1; undefined behaviour "
                           "would be logged and the run would continue")
        if "print_stacktrace=1" not in uopts:
            reasons.append("UBSAN_OPTIONS does not set print_stacktrace=1")

    for argv_name in ("configure_argv", "build_argv", "run_argv"):
        blob = " ".join(getattr(invocation, argv_name))
        for token in _CORE_DUMP_TOKENS:
            if token in blob:
                reasons.append(f"{argv_name} contains {token!r}, which would enable a core dump")
    for _, value in invocation.env:
        for token in _CORE_DUMP_TOKENS:
            if token in value:
                reasons.append(f"sanitizer env contains {token!r}, which would enable a core dump")

    return _verdict(reasons)


@dataclass(frozen=True)
class SanitizerEvidence:
    """The receipt of an ASAN/UBSAN build and targeted run that ACTUALLY happened."""

    invocation: SanitizerInvocation
    executed: bool
    exit_code: Optional[int]
    asan_findings: tuple
    ubsan_findings: tuple
    sanitizer_build_binary_sha256: str
    log_ref: str
    produced_by: str

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, SanitizerInvocation):
            raise TypeError("sanitizers.invocation must be a SanitizerInvocation")
        _req_bool(self.executed, "sanitizers.executed")
        if self.exit_code is not None and (isinstance(self.exit_code, bool)
                                           or not isinstance(self.exit_code, int)):
            raise ValueError("sanitizers.exit_code must be an int or None")
        for name in ("asan_findings", "ubsan_findings"):
            for item in _req_tuple(getattr(self, name), f"sanitizers.{name}"):
                _req_str(item, f"sanitizers.{name}[]")
        _req_sha256(self.sanitizer_build_binary_sha256,
                    "sanitizers.sanitizer_build_binary_sha256")
        _req_str(self.log_ref, "sanitizers.log_ref")
        _req_producer(self.produced_by, "sanitizers.produced_by")


@dataclass(frozen=True)
class PropertyMeasurement:
    """One candidate-only property residual, bound to its replay coordinates."""

    shape_id: str
    op: str
    backend: str
    metric_id: str
    residual: float
    tolerance: float
    suite_seed: int
    passed: bool
    input_transform: str = "identity"

    def __post_init__(self) -> None:
        for name in ("shape_id", "op", "backend", "metric_id"):
            _req_str(getattr(self, name), f"property_measurement.{name}")
        for name in ("residual", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"property_measurement.{name} must be finite and non-negative")
        _req_int(self.suite_seed, "property_measurement.suite_seed")
        _req_bool(self.passed, "property_measurement.passed")
        if self.input_transform not in ("identity", "x3", "x0p01", "negate"):
            raise ValueError(
                "property_measurement.input_transform must be identity|x3|x0p01|negate")
        derived = self.residual <= self.tolerance
        if self.passed != derived:
            raise ValueError(
                f"property_measurement.passed={self.passed} disagrees with "
                f"residual <= tolerance ({self.residual} <= {self.tolerance} is {derived})")

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.property_measurement.v1",
            "shape_id": self.shape_id,
            "op": self.op,
            "backend": self.backend,
            "metric_id": self.metric_id,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "suite_seed": self.suite_seed,
            "passed": self.passed,
            "input_transform": self.input_transform,
        }


@dataclass(frozen=True)
class OpSuiteEvidence:
    """§8.6 "targeted backend-op unit shapes" — the surface `kernel_eval.sh` truncated."""

    suite_id: str
    suite_source_sha256: str
    suite_seed: int
    ops_exercised: tuple
    ops_failed: tuple
    cases_by_op: tuple           # ((op, cases_total, cases_passed), ...)
    shapes_ref: str
    receipt_ref: str
    produced_by: str
    property_measurements: tuple = ()
    layout_probe: bool = False
    layout_families: tuple = ()
    layout_case_count: int = 0
    value_transform_probe: bool = False
    value_transforms: tuple = ()
    value_transform_case_count: int = 0
    stateful_probe: bool = False
    stateful_ops: tuple = ()
    stateful_case_count: int = 0

    def __post_init__(self) -> None:
        _req_str(self.suite_id, "op_suite.suite_id")
        _req_sha256(self.suite_source_sha256, "op_suite.suite_source_sha256")
        _req_int(self.suite_seed, "op_suite.suite_seed")
        for name in ("ops_exercised", "ops_failed"):
            for item in _req_tuple(getattr(self, name), f"op_suite.{name}"):
                _req_str(item, f"op_suite.{name}[]")
        for row in _req_tuple(self.cases_by_op, "op_suite.cases_by_op"):
            if not (isinstance(row, tuple) and len(row) == 3):
                raise TypeError("op_suite.cases_by_op rows must be (op, total, passed)")
            _req_str(row[0], "op_suite.cases_by_op op")
            _req_int(row[1], "op_suite.cases_by_op total")
            _req_int(row[2], "op_suite.cases_by_op passed")
        _req_str(self.shapes_ref, "op_suite.shapes_ref")
        _req_str(self.receipt_ref, "op_suite.receipt_ref")
        _req_producer(self.produced_by, "op_suite.produced_by")
        for item in _req_tuple(self.property_measurements,
                               "op_suite.property_measurements"):
            if not isinstance(item, PropertyMeasurement):
                raise TypeError(
                    "op_suite.property_measurements must contain PropertyMeasurement")
            if item.suite_seed != self.suite_seed:
                raise ValueError(
                    f"property measurement {item.shape_id!r} carries suite_seed "
                    f"{item.suite_seed}, expected {self.suite_seed}")
        _req_bool(self.layout_probe, "op_suite.layout_probe")
        _req_int(self.layout_case_count, "op_suite.layout_case_count")
        allowed_layouts = {"offset", "transpose", "stride_gap"}
        for item in _req_tuple(self.layout_families, "op_suite.layout_families"):
            if item not in allowed_layouts:
                raise ValueError(
                    f"op_suite.layout_families contains {item!r}, expected one of "
                    f"{sorted(allowed_layouts)}")
        if len(set(self.layout_families)) != len(self.layout_families):
            raise ValueError("op_suite.layout_families must be unique")
        if not self.layout_probe and (self.layout_families or self.layout_case_count):
            raise ValueError(
                "op_suite layout evidence cannot exist when layout_probe is false")
        _req_bool(self.value_transform_probe, "op_suite.value_transform_probe")
        _req_int(self.value_transform_case_count,
                 "op_suite.value_transform_case_count")
        allowed_transforms = {"identity", "x3", "x0p01", "negate"}
        for item in _req_tuple(self.value_transforms, "op_suite.value_transforms"):
            if item not in allowed_transforms:
                raise ValueError(
                    f"op_suite.value_transforms contains {item!r}, expected one of "
                    f"{sorted(allowed_transforms)}")
        if len(set(self.value_transforms)) != len(self.value_transforms):
            raise ValueError("op_suite.value_transforms must be unique")
        if not self.value_transform_probe and (
                self.value_transforms or self.value_transform_case_count):
            raise ValueError(
                "op_suite value-transform evidence cannot exist when the probe is false")
        _req_bool(self.stateful_probe, "op_suite.stateful_probe")
        _req_int(self.stateful_case_count, "op_suite.stateful_case_count")
        allowed_stateful = {
            "SSM_SCAN", "SSM_CONV", "FLASH_ATTN_EXT", "GATED_DELTA_NET"}
        for item in _req_tuple(self.stateful_ops, "op_suite.stateful_ops"):
            if item not in allowed_stateful:
                raise ValueError(
                    f"op_suite.stateful_ops contains {item!r}, expected one of "
                    f"{sorted(allowed_stateful)}")
        if len(set(self.stateful_ops)) != len(self.stateful_ops):
            raise ValueError("op_suite.stateful_ops must be unique")
        if not self.stateful_probe and (self.stateful_ops or self.stateful_case_count):
            raise ValueError(
                "op_suite stateful evidence cannot exist when stateful_probe is false")
        if sum((self.layout_probe, self.value_transform_probe, self.stateful_probe)) > 1:
            raise ValueError(
                "layout, value-transform, and stateful evidence must come from separate passes")

    def cases_for(self, op: str) -> Optional[tuple]:
        for row in self.cases_by_op:
            if row[0] == op:
                return (row[1], row[2])
        return None


@dataclass(frozen=True)
class ReferenceComparison:
    """One exact or declared-metric comparison against an oracle (§6.5)."""

    shape_id: str
    op: str
    mode: str                    # exact_bitwise | ulp_bounded | metric_bounded
    mismatch_count: int
    max_ulp_observed: Optional[float]
    tolerance_ulp: Optional[float]
    oracle_id: str
    oracle_is_candidate_derived: bool
    metric_id: Optional[str] = None
    max_error_observed: Optional[float] = None
    tolerance_error: Optional[float] = None

    def __post_init__(self) -> None:
        _req_str(self.shape_id, "reference.shape_id")
        _req_str(self.op, "reference.op")
        modes = ("exact_bitwise", "ulp_bounded", "metric_bounded")
        if self.mode not in modes:
            raise ValueError(f"reference.mode: {self.mode!r} must be one of {modes}")
        _req_int(self.mismatch_count, "reference.mismatch_count", minimum=0)
        for name in ("max_ulp_observed", "tolerance_ulp", "max_error_observed",
                     "tolerance_error"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"reference.{name} must be a finite non-negative number or None")
        if self.mode == "ulp_bounded" and self.tolerance_ulp is None:
            raise ValueError("reference: a ulp_bounded comparison with no declared tolerance "
                             "is not a comparison")
        if self.mode == "metric_bounded":
            _req_str(self.metric_id, "reference.metric_id")
            if self.max_error_observed is None or self.tolerance_error is None:
                raise ValueError(
                    "reference: a metric_bounded comparison needs both an observed error "
                    "and a declared tolerance")
            if self.max_ulp_observed is not None or self.tolerance_ulp is not None:
                raise ValueError(
                    "reference: metric_bounded evidence cannot also carry ULP fields")
        elif any(value is not None for value in
                 (self.metric_id, self.max_error_observed, self.tolerance_error)):
            raise ValueError(
                "reference: generic metric fields are valid only for metric_bounded mode")
        _req_str(self.oracle_id, "reference.oracle_id")
        _req_bool(self.oracle_is_candidate_derived, "reference.oracle_is_candidate_derived")


@dataclass(frozen=True)
class ReferenceEvidence:
    """§8.6 "exact reference comparisons where defined".

    `undefined_for` is what makes *"where defined"* checkable: an op with neither
    a comparison nor an entry here is neither compared nor declared undefined,
    and silence is not a reference.
    """

    comparisons: tuple
    undefined_for: tuple         # ((op, reason), ...)
    oracle_registry_ref: str
    produced_by: str

    def __post_init__(self) -> None:
        for item in _req_tuple(self.comparisons, "reference.comparisons"):
            if not isinstance(item, ReferenceComparison):
                raise TypeError("reference.comparisons must contain ReferenceComparison")
        for row in _req_tuple(self.undefined_for, "reference.undefined_for"):
            if not (isinstance(row, tuple) and len(row) == 2):
                raise TypeError("reference.undefined_for rows must be (op, reason)")
            _req_str(row[0], "reference.undefined_for op")
            _req_str(row[1], "reference.undefined_for reason")
        _req_str(self.oracle_registry_ref, "reference.oracle_registry_ref")
        _req_producer(self.produced_by, "reference.produced_by")


@dataclass(frozen=True)
class BoundaryShapeEvidence:
    """§8.6 "unseen/boundary shapes for dispatch changes".

    `held_out_from_planner` is the anti-overfitting property: a shape the planner
    saw is not an unseen shape, and a search that optimizes against its own test
    set will find the kernel that special-cases it. This mirrors the protocol's
    reason for the selection/confirmation split — *"the evidence that promotes a
    candidate is structurally unfit to report how ready it is."*
    """

    unseen_shapes: tuple
    boundary_shapes: tuple
    failures: tuple
    selection_rule_id: str
    selection_seed: str
    held_out_from_planner: bool
    receipt_ref: str
    produced_by: str

    def __post_init__(self) -> None:
        for name in ("unseen_shapes", "boundary_shapes", "failures"):
            for item in _req_tuple(getattr(self, name), f"boundary.{name}"):
                _req_str(item, f"boundary.{name}[]")
        _req_str(self.selection_rule_id, "boundary.selection_rule_id")
        _req_str(self.selection_seed, "boundary.selection_seed")
        _req_bool(self.held_out_from_planner, "boundary.held_out_from_planner")
        _req_str(self.receipt_ref, "boundary.receipt_ref")
        _req_producer(self.produced_by, "boundary.produced_by")


@dataclass(frozen=True)
class DispatchTraceEvidence:
    """§6.4 item 2/3 + §8.6's no-fallback proof, from one traced run.

    `fallback_instrumentation_active` exists because an empty `fallback_events`
    from an uninstrumented trace is not evidence of no fallback — it is evidence
    of no instrumentation. Concluding absence from one un-probed encoding is a
    named failure mode in this project.
    """

    derived_surface: tuple
    traced_kernels: tuple
    fallback_events: tuple
    fallback_instrumentation_active: bool
    trace_ref: str
    produced_by: str

    def __post_init__(self) -> None:
        for name in ("derived_surface", "traced_kernels", "fallback_events"):
            for item in _req_tuple(getattr(self, name), f"dispatch.{name}"):
                _req_str(item, f"dispatch.{name}[]")
        _req_bool(self.fallback_instrumentation_active,
                  "dispatch.fallback_instrumentation_active")
        _req_str(self.trace_ref, "dispatch.trace_ref")
        _req_producer(self.produced_by, "dispatch.produced_by")


@dataclass(frozen=True)
class StateSafetyEvidence:
    """§8.6 "state, rollback, teardown, and race tests where relevant"."""

    rollback_tested: bool
    teardown_tested: bool
    race_detector_id: Optional[str]
    race_findings: tuple
    leaked_resources: tuple
    orphan_processes: tuple
    receipt_ref: str
    produced_by: str

    def __post_init__(self) -> None:
        _req_bool(self.rollback_tested, "state.rollback_tested")
        _req_bool(self.teardown_tested, "state.teardown_tested")
        if self.race_detector_id is not None:
            _req_str(self.race_detector_id, "state.race_detector_id")
        for name in ("race_findings", "leaked_resources", "orphan_processes"):
            for item in _req_tuple(getattr(self, name), f"state.{name}"):
                _req_str(item, f"state.{name}[]")
        _req_str(self.receipt_ref, "state.receipt_ref")
        _req_producer(self.produced_by, "state.produced_by")


@dataclass(frozen=True)
class CoherenceEvidence:
    """The raw material of a coherence comparison. It carries NO label.

    Producing a label from this is `compute_coherence()`'s job and nothing else's;
    the whole point is that the label is derived, not attached at capture time.

    The three `anchor_*` identity fields record WHICH anchor produced
    `anchor_output_sha256`. Without them the capture says what the anchor produced
    and not whose output it was, so a capture taken against anchor A could be
    re-scored against anchor B — invariant 11's *"deterministic replay before
    regeneration"* is the designed path where exactly that happens — and yield a
    `byte_identical` label that names an agreement nobody claimed. They are
    optional as a triple only: all three or none (`_validate_anchor_triple`), and
    NONE means the identity was not recorded, which `compute_coherence` treats as
    COULD_NOT_CHECK-shaped and never as agreement.
    """

    candidate_output_sha256: Optional[str]
    candidate_output_len: int
    anchor_output_sha256: Optional[str]
    anchor_output_len: Optional[int]
    sampler_id: str
    sampler_is_greedy: Optional[bool]
    seed: Optional[int]
    tokens_requested: int
    token_agreement_ratio: Optional[float]
    divergence_first_index: Optional[int]
    anchor_determinism_class: Optional[str]
    anchor_source_commit: Optional[str]
    anchor_binary_sha256: Optional[str]
    anchor_linkage_sha256: Optional[str]
    prompt_ref: str
    receipt_ref: str
    produced_by: str

    def recorded_anchor(self) -> Optional[api.AnchorIdentity]:
        """The anchor this capture was taken AGAINST, or `None` if none was recorded.

        `None` is not "any anchor" and not "the current one": it is the absence of
        a record, and `compute_coherence` refuses to read it as agreement.
        """
        return _recorded_anchor(self)

    def __post_init__(self) -> None:
        _opt_sha256(self.candidate_output_sha256, "coherence.candidate_output_sha256")
        _opt_sha256(self.anchor_output_sha256, "coherence.anchor_output_sha256")
        _req_int(self.candidate_output_len, "coherence.candidate_output_len")
        _opt_int(self.anchor_output_len, "coherence.anchor_output_len")
        _req_str(self.sampler_id, "coherence.sampler_id")
        _opt_bool(self.sampler_is_greedy, "coherence.sampler_is_greedy")
        _opt_int(self.seed, "coherence.seed")
        _req_int(self.tokens_requested, "coherence.tokens_requested", minimum=1)
        if self.token_agreement_ratio is not None:
            _req_observed_ratio(self.token_agreement_ratio,
                                "coherence.token_agreement_ratio")
        _opt_int(self.divergence_first_index, "coherence.divergence_first_index")
        if self.anchor_determinism_class is not None and \
                self.anchor_determinism_class not in schemas.DETERMINISM_CLASSES:
            raise ValueError(
                f"coherence.anchor_determinism_class: {self.anchor_determinism_class!r} is not "
                f"one of {sorted(schemas.DETERMINISM_CLASSES)}")
        _validate_anchor_triple(
            source_commit=self.anchor_source_commit,
            binary_sha256=self.anchor_binary_sha256,
            linkage_sha256=self.anchor_linkage_sha256,
            label="coherence")
        _req_str(self.prompt_ref, "coherence.prompt_ref")
        _req_str(self.receipt_ref, "coherence.receipt_ref")
        _req_producer(self.produced_by, "coherence.produced_by")


@dataclass(frozen=True)
class DeterminismEvidence:
    """§8.6 determinism-class check; invariant 12 makes the class an interface.

    `anchor_output_digests` and `anchor_determinism_class` are the ANCHOR's
    behaviour, so the capture records the anchor's three-component identity beside
    them — all three or none (`_validate_anchor_triple`). Two consumers read that
    material, and both would otherwise accept another anchor's:
    `check_determinism_class` compares the class against the candidate's measured
    one, and `check_output_coherence` reconciles it against the coherence
    capture's copy of the same anchor property. That second one is why the
    identity matters most here — a reconciliation between two records is
    corroboration only if the two records describe the SAME anchor, and agreeing
    about a different anchor's class is not agreement about this one.
    """

    seed: int
    runs: int
    candidate_output_digests: tuple
    anchor_output_digests: tuple
    anchor_determinism_class: str
    anchor_source_commit: Optional[str]
    anchor_binary_sha256: Optional[str]
    anchor_linkage_sha256: Optional[str]
    declared_class_change: bool
    declared_class_change_ref: Optional[str]
    receipt_ref: str
    produced_by: str

    def recorded_anchor(self) -> Optional[api.AnchorIdentity]:
        """The anchor this capture's repeated runs were taken against, or `None`."""
        return _recorded_anchor(self)

    def __post_init__(self) -> None:
        _req_int(self.seed, "determinism.seed")
        _req_int(self.runs, "determinism.runs")
        for name in ("candidate_output_digests", "anchor_output_digests"):
            for item in _req_tuple(getattr(self, name), f"determinism.{name}"):
                _req_sha256(item, f"determinism.{name}[]")
        if self.anchor_determinism_class not in schemas.DETERMINISM_CLASSES:
            raise ValueError(
                f"determinism.anchor_determinism_class: "
                f"{self.anchor_determinism_class!r} is not one of "
                f"{sorted(schemas.DETERMINISM_CLASSES)}")
        _validate_anchor_triple(
            source_commit=self.anchor_source_commit,
            binary_sha256=self.anchor_binary_sha256,
            linkage_sha256=self.anchor_linkage_sha256,
            label="determinism")
        _req_bool(self.declared_class_change, "determinism.declared_class_change")
        if self.declared_class_change and not (
                isinstance(self.declared_class_change_ref, str)
                and self.declared_class_change_ref.strip()):
            raise ValueError(
                "determinism: a declared change of determinism class must name where it was "
                "declared; invariant 12 makes it 'a declared, release-relevant property', and "
                "an undocumented declaration is not one")
        _req_str(self.receipt_ref, "determinism.receipt_ref")
        _req_producer(self.produced_by, "determinism.produced_by")

    def measured_class(self) -> str:
        """The class the EVALUATOR computes. Never read from the candidate."""
        if self.runs < 2 or len(self.candidate_output_digests) < 2:
            return "not_measured"
        first = self.candidate_output_digests[0]
        stable = all(d == first for d in self.candidate_output_digests)
        return "bitwise_stable" if stable else "bitwise_unstable"


@dataclass(frozen=True)
class LinkageEvidence:
    """§8.6 "binary/linkage identity".

    The resolved-library check is the `verify_ggml_linkage.sh` scar in code form:
    the three production trees run three different ggml generations, and a binary
    that inherits another tree's ggml *"runs silently wrong"*. Silently wrong is
    exactly the failure a speed number cannot show.

    The anchor side is named by all THREE of precondition 4's components. Two
    digests without the commit named the anchor less completely here than in the
    `evaluation_event.v3` record this evidence feeds, and this is the gate whose
    whole subject is *"a rebuilt anchor is a different anchor"* — the component
    that says which source a rebuild came from cannot be the missing one.
    """

    binary_sha256: str
    linkage_sha256: str
    anchor_source_commit: Optional[str]
    anchor_binary_sha256: Optional[str]
    anchor_linkage_sha256: Optional[str]
    resolved_libraries: tuple      # ((soname, path, sha256), ...)
    expected_library_root: str
    verifier_id: str
    receipt_ref: str
    produced_by: str

    def __post_init__(self) -> None:
        _req_sha256(self.binary_sha256, "linkage.binary_sha256")
        _req_sha256(self.linkage_sha256, "linkage.linkage_sha256")
        _validate_anchor_triple(
            source_commit=self.anchor_source_commit,
            binary_sha256=self.anchor_binary_sha256,
            linkage_sha256=self.anchor_linkage_sha256,
            label="linkage")
        for row in _req_tuple(self.resolved_libraries, "linkage.resolved_libraries"):
            if not (isinstance(row, tuple) and len(row) == 3):
                raise TypeError("linkage.resolved_libraries rows must be (soname, path, sha256)")
            _req_str(row[0], "linkage.resolved_libraries soname")
            _req_str(row[1], "linkage.resolved_libraries path")
            _req_sha256(row[2], "linkage.resolved_libraries sha256")
        _req_str(self.expected_library_root, "linkage.expected_library_root")
        _req_str(self.verifier_id, "linkage.verifier_id")
        _req_str(self.receipt_ref, "linkage.receipt_ref")
        _req_producer(self.produced_by, "linkage.produced_by")


@dataclass(frozen=True)
class AntiRewardHackingEvidence:
    """§8.6 "anti-reward-hacking/integrity checks"; control 3's detector.

    `delivered_units_*` is deliberately not "FLOPs". A legitimate optimization may
    do less INTERNAL work — that is the point of it. What may never shrink is
    DELIVERED work: tokens generated, layers evaluated, elements written. Control
    3 cheats by *"reducing work"* in that second sense, so the comparison here is
    exact and has no tolerance knob to widen.

    `delivered_units_anchor` is the floor that comparison is made against, and the
    anchor triple records which anchor delivered it — all three components or none
    (`_validate_anchor_triple`). An exact comparison against a floor lifted from
    another anchor's run is not strict, it is arbitrary: it can clear a candidate
    that really did reduce work, which is the one thing this field exists to
    prevent.
    """

    cache_state: str
    correctness_verdict_source: str
    candidate_output_used_as_oracle: bool
    oracle_ids: tuple
    delivered_unit_name: str
    delivered_units_candidate: int
    delivered_units_anchor: Optional[int]
    anchor_source_commit: Optional[str]
    anchor_binary_sha256: Optional[str]
    anchor_linkage_sha256: Optional[str]
    environment_probe_findings: tuple
    timing_dependent_branch_findings: tuple
    receipt_ref: str
    environment_probe_detector_id: Optional[str] = None
    timing_dependent_branch_detector_id: Optional[str] = None
    stream_creation_findings: tuple = ()
    async_escape_findings: tuple = ()
    instrument_frame_findings: tuple = ()
    pointer_memoization_findings: tuple = ()
    structured_short_circuit_findings: tuple = ()
    stream_creation_detector_id: Optional[str] = None
    async_escape_detector_id: Optional[str] = None
    instrument_frame_detector_id: Optional[str] = None
    pointer_memoization_detector_id: Optional[str] = None
    structured_short_circuit_detector_id: Optional[str] = None

    def recorded_anchor(self) -> Optional[api.AnchorIdentity]:
        """The anchor that delivered `delivered_units_anchor`, or `None` if unrecorded."""
        return _recorded_anchor(self)

    def __post_init__(self) -> None:
        if self.cache_state not in CACHE_STATES:
            raise ValueError(
                f"anti_reward_hacking.cache_state: {self.cache_state!r} is not one of "
                f"{list(CACHE_STATES)}. 'Cache state is declared in every record' — and "
                "'unknown' is in the vocabulary so an unrecorded state is SAYABLE rather "
                "than an empty field that reads as cold.")
        _req_producer(self.correctness_verdict_source,
                      "anti_reward_hacking.correctness_verdict_source")
        _req_bool(self.candidate_output_used_as_oracle,
                  "anti_reward_hacking.candidate_output_used_as_oracle")
        for item in _req_tuple(self.oracle_ids, "anti_reward_hacking.oracle_ids"):
            _req_str(item, "anti_reward_hacking.oracle_ids[]")
        _req_str(self.delivered_unit_name, "anti_reward_hacking.delivered_unit_name")
        _req_int(self.delivered_units_candidate, "anti_reward_hacking.delivered_units_candidate")
        _opt_int(self.delivered_units_anchor, "anti_reward_hacking.delivered_units_anchor")
        _validate_anchor_triple(
            source_commit=self.anchor_source_commit,
            binary_sha256=self.anchor_binary_sha256,
            linkage_sha256=self.anchor_linkage_sha256,
            label="anti_reward_hacking")
        for name in ("environment_probe_findings", "timing_dependent_branch_findings",
                     "stream_creation_findings", "async_escape_findings",
                     "instrument_frame_findings", "pointer_memoization_findings",
                     "structured_short_circuit_findings"):
            for item in _req_tuple(getattr(self, name), f"anti_reward_hacking.{name}"):
                _req_str(item, f"anti_reward_hacking.{name}[]")
        for name in ("environment_probe_detector_id",
                     "timing_dependent_branch_detector_id",
                     "stream_creation_detector_id", "async_escape_detector_id",
                     "instrument_frame_detector_id", "pointer_memoization_detector_id",
                     "structured_short_circuit_detector_id"):
            value = getattr(self, name)
            if value is not None:
                _req_str(value, f"anti_reward_hacking.{name}")
        _req_str(self.receipt_ref, "anti_reward_hacking.receipt_ref")


#: A surface that may be evidence, an explicit `NotApplicable`, or `None`. The
#: three are distinct: evidence is evaluated, `NotApplicable` is a declared and
#: attributed absence, `None` is COULD_NOT_CHECK.
_Maybe = Union[Any, NotApplicable, None]


SOURCE_PREREQUISITE_IDS = (
    "input_sensitivity", "hostile_distributions", "checker_isolation",
)


@dataclass(frozen=True)
class SourcePrerequisiteEvidence:
    """Hash-bound pre-campaign correctness evidence for one source candidate.

    The offline reducers do not gain verdict authority merely by returning a
    report.  A report enters T0 only through this binding, which names the exact
    candidate source, evaluator bundle, producer, evidence object and capture
    mode.  ``dry_run`` is representable but can never carry a PASS.
    """

    prerequisite_id: str
    candidate_source_sha256: str
    evaluator_bundle_sha256: str
    suite_version: str
    producer_id: str
    capture_mode: str
    evidence_ref: str
    evidence_sha256: str
    check: schemas.Check

    def __post_init__(self) -> None:
        if self.prerequisite_id not in SOURCE_PREREQUISITE_IDS:
            raise ValueError(
                f"source prerequisite {self.prerequisite_id!r} is not one of "
                f"{list(SOURCE_PREREQUISITE_IDS)}")
        _req_sha256(self.candidate_source_sha256,
                    "source_prerequisite.candidate_source_sha256")
        _req_sha256(self.evaluator_bundle_sha256,
                    "source_prerequisite.evaluator_bundle_sha256")
        _req_str(self.suite_version, "source_prerequisite.suite_version")
        if self.producer_id != "trusted_evaluator":
            raise ValueError(
                "source prerequisite must be produced by trusted_evaluator")
        if self.capture_mode not in ("measured", "dry_run"):
            raise ValueError("source prerequisite capture_mode must be measured or dry_run")
        _req_str(self.evidence_ref, "source_prerequisite.evidence_ref")
        _req_sha256(self.evidence_sha256, "source_prerequisite.evidence_sha256")
        if not isinstance(self.check, schemas.Check):
            raise TypeError("source_prerequisite.check must be a schemas.Check")
        if self.capture_mode == "dry_run" and self.check.outcome == schemas.PASS:
            raise ValueError("dry-run source prerequisite cannot carry PASS")

    def measurement(self) -> dict:
        return {
            "schema": "epyc.autokernel.source_prerequisite.v1",
            "prerequisite_id": self.prerequisite_id,
            "candidate_source_sha256": self.candidate_source_sha256,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "suite_version": self.suite_version,
            "producer_id": self.producer_id,
            "capture_mode": self.capture_mode,
            "evidence_ref": self.evidence_ref,
            "evidence_sha256": self.evidence_sha256,
            "outcome": self.check.outcome,
        }


@dataclass(frozen=True)
class T0Evidence:
    """Everything T0 examines, for one candidate. No field has a default.

    A default would let a caller omit a surface and have the omission read as
    satisfied — which is the shape of the defect this whole module replaces. Every
    field must be passed, including the ones that are `None`; `None` means "no
    evidence" and produces `COULD_NOT_CHECK`, never PASS.
    """

    control_role: Optional[str]
    change_surface: ChangeSurface
    symbols: Optional[SymbolTableDiff]
    build: Optional[BuildProvenance]
    diff: Optional[DiffPolicyEvidence]
    static_analysis: Optional[StaticAnalysisEvidence]
    sanitizers: Optional[SanitizerEvidence]
    op_suite: Optional[OpSuiteEvidence]
    reference: _Maybe
    boundary_shapes: _Maybe
    dispatch_trace: Optional[DispatchTraceEvidence]
    state_safety: _Maybe
    coherence: Optional[CoherenceEvidence]
    determinism: Optional[DeterminismEvidence]
    linkage: Optional[LinkageEvidence]
    anti_reward_hacking: Optional[AntiRewardHackingEvidence]
    #: Source-changing candidates require all three hash-bound prerequisites.
    #: Parameter-only candidates leave this false and carry no bindings.
    source_candidate: bool = False
    source_prerequisites: tuple = ()
    #: Non-PASS findings made while projecting parsed artifact evidence into
    #: the records above. Entries are ``(gate_id, check_name, Check)`` and are
    #: folded into existing integrity gates by ``evaluate_t0``. Empty is honest
    #: for replay fixtures that supply the final records directly and therefore
    #: perform no projection.
    projection_checks: tuple = ()

    def __post_init__(self) -> None:
        if self.control_role is not None and self.control_role not in CONTROL_ROLES:
            raise ValueError(f"evidence.control_role: {self.control_role!r} is not one of "
                             f"{list(CONTROL_ROLES)}")
        if not isinstance(self.change_surface, ChangeSurface):
            raise TypeError("evidence.change_surface must be a ChangeSurface")
        if not isinstance(self.source_candidate, bool):
            raise TypeError("evidence.source_candidate must be a bool")
        typed = (
            ("symbols", SymbolTableDiff), ("build", BuildProvenance),
            ("diff", DiffPolicyEvidence), ("static_analysis", StaticAnalysisEvidence),
            ("sanitizers", SanitizerEvidence), ("op_suite", OpSuiteEvidence),
            ("dispatch_trace", DispatchTraceEvidence), ("coherence", CoherenceEvidence),
            ("determinism", DeterminismEvidence), ("linkage", LinkageEvidence),
            ("anti_reward_hacking", AntiRewardHackingEvidence),
        )
        for name, klass in typed:
            value = getattr(self, name)
            if value is not None and not isinstance(value, klass):
                raise TypeError(f"evidence.{name} must be a {klass.__name__} or None")
        for name, klass in (("reference", ReferenceEvidence),
                            ("boundary_shapes", BoundaryShapeEvidence),
                            ("state_safety", StateSafetyEvidence)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (klass, NotApplicable)):
                raise TypeError(
                    f"evidence.{name} must be a {klass.__name__}, a NotApplicable, or None")
        prerequisite_ids = []
        for item in self.source_prerequisites:
            if not isinstance(item, SourcePrerequisiteEvidence):
                raise TypeError(
                    "evidence.source_prerequisites entries must be "
                    "SourcePrerequisiteEvidence")
            prerequisite_ids.append(item.prerequisite_id)
        if len(prerequisite_ids) != len(set(prerequisite_ids)):
            raise ValueError("evidence.source_prerequisites contains duplicate ids")
        if not self.source_candidate and self.source_prerequisites:
            raise ValueError(
                "parameter/no-source evidence cannot carry source prerequisites")
        allowed_projection_gates = {
            GID_SYMBOLS, GID_SEMANTIC_DIFF, GID_SURFACE_RECONCILIATION,
        }
        for item in self.projection_checks:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError(
                    "evidence.projection_checks entries must be "
                    "(gate_id, check_name, schemas.Check) triples")
            gate_id, check_name, check = item
            if gate_id not in allowed_projection_gates:
                raise ValueError(
                    f"evidence.projection_checks gate {gate_id!r} is not projection-owned")
            _req_str(check_name, "evidence.projection_checks[].check_name")
            if not isinstance(check, schemas.Check):
                raise TypeError("evidence.projection_checks[].check must be a schemas.Check")


# =============================================================================
# Coherence — a COMPUTED verdict against an explicit anchor
# =============================================================================

_COHERENCE_MINT = object()


def _derive_coherence(*, anchor_bound: bool,
                      anchor_identity_recorded: bool,
                      candidate_output_sha256: Optional[str],
                      candidate_output_len: Optional[int],
                      anchor_output_sha256: Optional[str],
                      sampler_is_greedy: Optional[bool],
                      anchor_determinism_class: Optional[str],
                      token_agreement_ratio: Optional[float],
                      tolerance_floor: Optional[float]) -> tuple:
    """The ONLY place a coherence label comes from. Pure function of its arguments.

    Order, and why it is this order:

    1. **Empty generation first.** A candidate that generated nothing is broken,
       and that is true with or without an anchor. `kernel_eval.sh` recorded
       `COH="empty-generation"` and then emitted `"status":"OK"` anyway.
    2. **Then the anchor.** With no anchor, or no anchor output, the label is
       `not_compared`. It is NOT "coherent": *"Absence of a comparison is not
       evidence of equivalence."*
    3. **Then WHOSE anchor output it is.** An anchor output that names no
       capturing anchor cannot be attributed to the bound one, so the comparison
       is `not_compared` — the same COULD_NOT_CHECK-shaped outcome, for the same
       reason. It sits AFTER the digest check on purpose: with no anchor output
       there is nothing to attribute, and that case already has its own reason.
    4. **Then the sampler.** Under a non-greedy sampler a byte difference proves
       nothing and byte-identity proves nothing either, so the comparison is
       `undecidable_under_sampling` — a third outcome, not a pass and not a fail.
    5. **Then the digests**, with a declared tolerance admitted ONLY when the
       anchor's own determinism class is `bitwise_unstable`, i.e. when byte
       identity is unattainable by construction rather than merely inconvenient.
    """
    if candidate_output_len == 0:
        return COHERENCE_EMPTY, (
            "the candidate generated no output; an empty generation is a candidate defect "
            "and needs no anchor to establish",)
    if not anchor_bound:
        return COHERENCE_NOT_COMPARED, (
            "no anchor is bound, so no coherence comparison was performed; a coherence or "
            "identity label produced without a named anchor comparison is not a verdict",)
    if candidate_output_sha256 is None or anchor_output_sha256 is None:
        return COHERENCE_NOT_COMPARED, (
            "one side of the comparison has no output digest; absence of a comparison is not "
            "evidence of equivalence",)
    if not anchor_identity_recorded:
        return COHERENCE_NOT_COMPARED, (
            "the coherence capture records no anchor identity, so the anchor output it carries "
            "cannot be attributed to the bound anchor; an unrecorded identity is not agreement "
            "with the anchor that happens to be bound now",)
    if sampler_is_greedy is None:
        return COHERENCE_UNDECIDABLE, (
            "the sampler's determinism was not recorded, so neither a byte difference nor a "
            "byte identity can be interpreted",)
    if not sampler_is_greedy:
        return COHERENCE_UNDECIDABLE, (
            "coherence was measured under a non-deterministic sampler; a byte difference is "
            "not evidence of divergence and a byte identity is not evidence of equivalence",)
    if candidate_output_sha256 == anchor_output_sha256:
        return COHERENCE_BYTE_IDENTICAL, ()
    if anchor_determinism_class == "bitwise_unstable" \
            and tolerance_floor is not None and token_agreement_ratio is not None \
            and token_agreement_ratio >= tolerance_floor:
        return COHERENCE_WITHIN_TOLERANCE, (
            f"outputs differ but the anchor's own determinism class is bitwise_unstable, so "
            f"byte identity is unattainable; token agreement {token_agreement_ratio} meets the "
            f"declared floor {tolerance_floor}",)
    reasons = [f"candidate output {candidate_output_sha256[:12]} differs from anchor output "
               f"{anchor_output_sha256[:12]} under a greedy sampler"]
    if anchor_determinism_class != "bitwise_unstable":
        reasons.append(
            f"the anchor's determinism class is {anchor_determinism_class!r}, so a declared "
            "tolerance does not apply: byte identity is attainable here")
    elif tolerance_floor is None:
        reasons.append("no coherence tolerance floor was declared by the campaign")
    elif token_agreement_ratio is None:
        reasons.append("no token agreement ratio was measured to compare against the floor")
    else:
        reasons.append(f"token agreement {token_agreement_ratio} is below the declared floor "
                       f"{tolerance_floor}")
    return COHERENCE_DIVERGENT, tuple(reasons)


@dataclass(frozen=True)
class CoherenceVerdict:
    """A COMPUTED coherence label. There is no path that stamps one.

    Two locks, the same shape as `api.Verdict`:

    1. `__init__` refuses without the module-private mint token that only
       `compute_coherence()` holds; and
    2. `__post_init__` RE-DERIVES the label from the evidence stored on this very
       object and raises `CoherenceTampering` on disagreement — so taking the mint
       token buys nothing.

    Plus a third that is specific to this class and to the defect it replaces: an
    equivalence label with `anchor_bound=False` raises `CoherenceWithoutAnchor`.
    `COH="coherent"` is not expressible here.

    `anchor_identity_recorded` is stored rather than inferred because the second
    lock re-derives from THIS object: a verdict whose evidence never named the
    anchor it was captured against must not be able to carry an equivalence label,
    and flipping the flag on a minted verdict re-derives to `not_compared` and
    raises `CoherenceTampering`.
    """

    label: str
    anchor_bound: bool
    anchor_identity_recorded: bool
    candidate_output_sha256: Optional[str]
    candidate_output_len: Optional[int]
    anchor_output_sha256: Optional[str]
    sampler_is_greedy: Optional[bool]
    anchor_determinism_class: Optional[str]
    token_agreement_ratio: Optional[float]
    tolerance_floor: Optional[float]
    reasons: tuple
    mint: InitVar[Any] = None

    def __post_init__(self, mint: Any) -> None:
        if mint is not _COHERENCE_MINT:
            raise CoherenceTampering(
                "CoherenceVerdict is not constructible directly — a coherence label is derived "
                "from a comparison against a named anchor. Call compute_coherence()."
            )
        if self.label not in COHERENCE_LABELS:
            raise CoherenceTampering(f"label {self.label!r} is not one of "
                                     f"{list(COHERENCE_LABELS)}")
        if self.label in COHERENCE_EQUIVALENCE_LABELS and not self.anchor_bound:
            raise CoherenceWithoutAnchor(
                f"label {self.label!r} asserts equivalence but no anchor is bound. "
                "'A run without an explicit anchor is INVALID — never \"correct\", never "
                "\"coherent\", never \"byte-identical\".' This is the exact statement "
                "kernel_eval.sh produced from an absent baseline."
            )
        derived_label, derived_reasons = _derive_coherence(
            anchor_bound=self.anchor_bound,
            anchor_identity_recorded=self.anchor_identity_recorded,
            candidate_output_sha256=self.candidate_output_sha256,
            candidate_output_len=self.candidate_output_len,
            anchor_output_sha256=self.anchor_output_sha256,
            sampler_is_greedy=self.sampler_is_greedy,
            anchor_determinism_class=self.anchor_determinism_class,
            token_agreement_ratio=self.token_agreement_ratio,
            tolerance_floor=self.tolerance_floor,
        )
        if (self.label, self.reasons) != (derived_label, derived_reasons):
            raise CoherenceTampering(
                f"coherence label does not follow from its own evidence: stored "
                f"{self.label!r} {list(self.reasons)}, derived {derived_label!r} "
                f"{list(derived_reasons)}"
            )

    @property
    def asserts_equivalence(self) -> bool:
        return self.label in COHERENCE_EQUIVALENCE_LABELS

    def to_check(self) -> schemas.Check:
        """Map the label onto the three-outcome vocabulary. Never a soft pass."""
        if self.label in COHERENCE_EQUIVALENCE_LABELS:
            return schemas.Check(schemas.PASS, self.reasons)
        if self.label in (COHERENCE_DIVERGENT, COHERENCE_EMPTY):
            return _fail(*self.reasons)
        return _cnc(*self.reasons)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "anchor_bound": self.anchor_bound,
            "anchor_identity_recorded": self.anchor_identity_recorded,
            "candidate_output_sha256": self.candidate_output_sha256,
            "anchor_output_sha256": self.anchor_output_sha256,
            "sampler_is_greedy": self.sampler_is_greedy,
            "anchor_determinism_class": self.anchor_determinism_class,
            "token_agreement_ratio": self.token_agreement_ratio,
            "tolerance_floor": self.tolerance_floor,
            "reasons": list(self.reasons),
        }


def compute_coherence(*,
                      anchor: Optional[api.AnchorIdentity],
                      evidence: Optional[CoherenceEvidence],
                      tolerance_floor: Optional[float]) -> CoherenceVerdict:
    """Compute the coherence verdict. The ONLY constructor of `CoherenceVerdict`.

    Precondition 4 is structural here in two directions.

    *That* an anchor is bound: `anchor=None` can only ever produce `not_compared`
    (or `empty_generation`, which is a statement about the candidate alone),
    because `_derive_coherence` returns nothing else and
    `CoherenceVerdict.__post_init__` refuses an equivalence label without one.

    *Which* anchor is bound: the identity recorded on the evidence — the anchor
    whose output the capture actually holds — is compared against the anchor
    passed here, by `api.AnchorIdentity.identity_matches`, which is the same
    all-three-components comparator the record uses. Three outcomes, and no
    fourth:

      * **PASS** — the capture and the caller name the same anchor; derive.
      * **FAIL** — they name different anchors. RAISES `CoherenceAnchorMismatch`.
        Invariant 11 makes replaying saved outputs the normal path, so this is
        where a wrong replay actually happens; returning `not_compared` would
        record "no comparison was possible" for what is really "the caller
        compared the wrong material", and the replay bug would survive.
      * **COULD_NOT_CHECK** — the capture recorded no identity (it predates the
        field, or the producer omitted it). No raise: nothing disagrees. But it
        does not pass either — `anchor_identity_recorded=False` sends the
        derivation to `not_compared`, because absence of a recorded identity is
        not agreement.

    The mismatch check runs BEFORE the derivation, including before the
    empty-generation branch: material captured against another anchor is not this
    run's material, and no label — not even one about the candidate alone — should
    be minted from a mix-up the caller has not yet noticed.
    """
    if anchor is not None and not isinstance(anchor, api.AnchorIdentity):
        raise TypeError("anchor must be an api.AnchorIdentity or None")
    if evidence is not None and not isinstance(evidence, CoherenceEvidence):
        raise TypeError("evidence must be a CoherenceEvidence or None")
    if tolerance_floor is not None:
        _req_ratio(tolerance_floor, "tolerance_floor")

    anchor_bound = anchor is not None
    recorded_anchor = None if evidence is None else evidence.recorded_anchor()
    _refuse_replay_mismatch(
        anchor, recorded_anchor, label="coherence",
        consequence=("A coherence label computed across that mismatch would assert "
                     "agreement with an anchor output this run never produced"),
        error=CoherenceAnchorMismatch)

    if evidence is None:
        label, reasons = _derive_coherence(
            anchor_bound=anchor_bound, anchor_identity_recorded=False,
            candidate_output_sha256=None,
            candidate_output_len=None, anchor_output_sha256=None,
            sampler_is_greedy=None, anchor_determinism_class=None,
            token_agreement_ratio=None, tolerance_floor=tolerance_floor)
        # The reasons are passed through UNMODIFIED. Appending an extra reason here
        # would make the stored value disagree with the re-derivation in
        # __post_init__ and raise CoherenceTampering — which is the lock working,
        # but it also means the "why" for a missing capture belongs on the gate's
        # notes, not inside the derived verdict. `check_output_coherence` puts it
        # there.
        return CoherenceVerdict(
            label=label, anchor_bound=anchor_bound, anchor_identity_recorded=False,
            candidate_output_sha256=None,
            candidate_output_len=None, anchor_output_sha256=None, sampler_is_greedy=None,
            anchor_determinism_class=None, token_agreement_ratio=None,
            tolerance_floor=tolerance_floor, reasons=reasons, mint=_COHERENCE_MINT)

    label, reasons = _derive_coherence(
        anchor_bound=anchor_bound,
        anchor_identity_recorded=recorded_anchor is not None,
        candidate_output_sha256=evidence.candidate_output_sha256,
        candidate_output_len=evidence.candidate_output_len,
        anchor_output_sha256=evidence.anchor_output_sha256,
        sampler_is_greedy=evidence.sampler_is_greedy,
        anchor_determinism_class=evidence.anchor_determinism_class,
        token_agreement_ratio=evidence.token_agreement_ratio,
        tolerance_floor=tolerance_floor,
    )
    return CoherenceVerdict(
        label=label,
        anchor_bound=anchor_bound,
        anchor_identity_recorded=recorded_anchor is not None,
        candidate_output_sha256=evidence.candidate_output_sha256,
        candidate_output_len=evidence.candidate_output_len,
        anchor_output_sha256=evidence.anchor_output_sha256,
        sampler_is_greedy=evidence.sampler_is_greedy,
        anchor_determinism_class=evidence.anchor_determinism_class,
        token_agreement_ratio=evidence.token_agreement_ratio,
        tolerance_floor=tolerance_floor,
        reasons=reasons,
        mint=_COHERENCE_MINT,
    )


# =============================================================================
# The gate registry — the coverage contract
# =============================================================================

GID_SYMBOLS = "t0.source_integrity.symbol_and_registration_preservation"
GID_CLEAN_BUILD = "t0.source_integrity.clean_build_from_snapshot"
GID_SEMANTIC_DIFF = "t0.source_integrity.semantic_diff_conformance"
GID_SCHEMA_DIFF_POLICY = "t0.schema_and_diff_policy"
GID_STATIC_COMPILE = "t0.static_and_compile_checks"
GID_ASAN = "t0.sanitizer.asan"
GID_UBSAN = "t0.sanitizer.ubsan"
GID_OP_UNITS = "t0.backend_op_units"
GID_EXACT_REFERENCE = "t0.exact_reference_comparison"
GID_BOUNDARY_SHAPES = "t0.unseen_boundary_shapes"
GID_SURFACE_RECONCILIATION = "t0.affected_surface_reconciliation"
GID_NO_FALLBACK = "t0.no_fallback_dispatch_proof"
GID_STATE_SAFETY = "t0.state_rollback_teardown_race"
GID_COHERENCE = "t0.output_coherence_vs_anchor"
GID_DETERMINISM = "t0.determinism_class"
GID_LINKAGE = "t0.binary_and_linkage_identity"
GID_ANTI_REWARD_HACKING = "t0.anti_reward_hacking"

#: `(gate_id, gate_class, requires_anchor)`, in §8.5.1-then-§8.6 order.
#:
#: Class choice is load-bearing, not cosmetic. Every id below sits in a class
#: `api.SPEED_BLOCKING_GATE_CLASSES` contains, so ANY T0 failure makes
#: `Verdict.rank_key()` raise. In particular the two dispatch gates are
#: `integrity` and NOT `mechanism`: `mechanism` is not speed-blocking, and control
#: 3 — the candidate that *"silently falls back"* — *"MUST receive no speed rank
#: at all"*. Filing the no-fallback proof under `mechanism` would have let a
#: falling-back candidate be ranked.
T0_GATE_SPEC = (
    (GID_SYMBOLS, api.GATE_INTEGRITY, True),
    (GID_CLEAN_BUILD, api.GATE_INTEGRITY, False),
    (GID_SEMANTIC_DIFF, api.GATE_INTEGRITY, False),
    (GID_SCHEMA_DIFF_POLICY, api.GATE_INTEGRITY, False),
    (GID_STATIC_COMPILE, api.GATE_INTEGRITY, False),
    # ASAN findings are memory-safety defects -> stability. UBSAN findings are
    # undefined behaviour, overwhelmingly arithmetic/aliasing -> numerical_safety.
    # Two gates rather than one, so the record says WHICH surface failed.
    (GID_ASAN, api.GATE_STABILITY, False),
    (GID_UBSAN, api.GATE_NUMERICAL_SAFETY, False),
    (GID_OP_UNITS, api.GATE_CORRECTNESS, False),
    (GID_EXACT_REFERENCE, api.GATE_CORRECTNESS, True),
    (GID_BOUNDARY_SHAPES, api.GATE_CORRECTNESS, False),
    (GID_SURFACE_RECONCILIATION, api.GATE_INTEGRITY, False),
    (GID_NO_FALLBACK, api.GATE_INTEGRITY, False),
    (GID_STATE_SAFETY, api.GATE_STABILITY, False),
    (GID_COHERENCE, api.GATE_CORRECTNESS, True),
    (GID_DETERMINISM, api.GATE_DETERMINISM, True),
    (GID_LINKAGE, api.GATE_INTEGRITY, True),
    (GID_ANTI_REWARD_HACKING, api.GATE_INTEGRITY, False),
)

T0_GATE_IDS = tuple(gid for gid, _, _ in T0_GATE_SPEC)
_GATE_CLASS_BY_ID = {gid: cls for gid, cls, _ in T0_GATE_SPEC}
_GATE_ANCHOR_BY_ID = {gid: req for gid, _, req in T0_GATE_SPEC}

# Fail fast at import time if the registry ever disagrees with api's vocabulary.
for _gid, _cls, _ in T0_GATE_SPEC:
    if _cls not in api.SPEED_BLOCKING_GATE_CLASSES:
        raise AssertionError(
            f"T0 gate {_gid!r} is filed under {_cls!r}, which api.SPEED_BLOCKING_GATE_CLASSES "
            "does not contain. A T0 gate that does not block speed ranking is not a T0 gate: "
            "correctness is lexicographically prior to speed."
        )
del _gid, _cls


def _gate(gate_id: str, check: schemas.Check, *,
          evidence_ref: Optional[str] = None,
          notes: Sequence[str] = (),
          measurements: Sequence[dict] = ()) -> api.GateResult:
    """Build a `GateResult` from the registry, so class and anchor-ness are never
    supplied at the call site and cannot drift between gates."""
    return api.GateResult(
        gate_id=gate_id,
        gate_class=_GATE_CLASS_BY_ID[gate_id],
        check=check,
        requires_anchor=_GATE_ANCHOR_BY_ID[gate_id],
        evidence_ref=evidence_ref,
        notes=tuple(notes),
        measurements=tuple(measurements),
    )


def _no_evidence(surface: str) -> schemas.Check:
    return _cnc(f"no {surface} evidence was captured; an unevaluated surface is "
                "COULD_NOT_CHECK, never a pass")


def _no_anchor(what: str) -> schemas.Check:
    return _cnc(f"{what} is a comparison against the anchor and no anchor is bound "
                "(P-AK-SEARCH-1 precondition 4); absence of a comparison is not evidence "
                "of equivalence")


def _self_reported(surface: str, producer: str) -> str:
    return (f"{surface} evidence was produced by {producer!r}, not the evaluator; "
            "'Correctness verdicts are produced by the evaluator against declared oracles "
            "and are NEVER self-reported by the candidate'")


def _any_true(*flags: Optional[bool]) -> Optional[bool]:
    """Three-valued OR. `True` if any flag is True, `None` if any is undetermined.

    `False or None` is `None`, never `False`. `ChangeSurface.sanitizers_mandatory`
    already reasons this way; every other consumer of the derived flags must too,
    or "we could not tell whether this change touches threading" is reported as
    "it does not" and the surface is PASSed on a fact nobody established.
    """
    if any(flag is True for flag in flags):
        return True
    if any(flag is None for flag in flags):
        return None
    return False


def _na_contradiction(na: "NotApplicable", relevant: Optional[bool], what: str) -> Optional[str]:
    """A static derivation may not declare n/a a surface it itself says is touched.

    `NotApplicable` refuses `source="actor"` (invariant 18), but that only closes
    the front door. A `NotApplicable` stamped `static_derivation` while the SAME
    derivation's `derived_touches_*` flag says the surface IS touched is the back
    door: it turns a required behavioural surface into a PASS by attaching a note
    that contradicts its own cited source. An `operator_waiver` is a human taking
    the call on the record and is left alone.
    """
    if relevant is not True or na.source != "static_derivation":
        return None
    return (f"the surface is declared not applicable by {na.source}={na.ref} "
            f"({na.reason!r}), but the mechanical derivation says this change {what}. "
            "A derivation cannot waive a surface its own flags say is in scope; the "
            "actor's declaration is a scored prediction, never a scope input "
            "(invariant 18), and a self-contradicting derivation is not a stronger "
            "source than one.")


# =============================================================================
# The seventeen T0 checks
# =============================================================================

def check_symbol_and_registration_preservation(
        request: api.EvaluationRequest,
        evidence: Optional[SymbolTableDiff],
        policy: T0Policy) -> api.GateResult:
    """§8.5.1 item 1. The C++ analogue of public-name preservation."""
    if request.anchor is None:
        return _gate(GID_SYMBOLS, _no_anchor("symbol and registration preservation"))
    if evidence is None:
        return _gate(GID_SYMBOLS, _no_evidence("symbol-table diff"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_SYMBOLS, _fail(_self_reported("symbol-table", evidence.produced_by)),
                     evidence_ref=evidence.receipt_ref)

    declared = set(evidence.declared_removals)
    reasons = []
    for label, items in (("symbol", evidence.removed_symbols),
                         ("op registration", evidence.removed_op_registrations),
                         ("dispatch predicate", evidence.removed_dispatch_predicates)):
        undeclared = tuple(sorted(set(items) - declared))
        if undeclared:
            reasons.append(
                f"{len(undeclared)} {label}(s) removed and not declared in the proposal: "
                f"{list(undeclared)}. A dropped specialization or a deleted dispatch case "
                "compiles cleanly and silently changes behaviour for every shape nobody "
                "happened to test.")
    undeclared_arity = tuple(sorted(set(evidence.arity_changed_symbols) - declared))
    if undeclared_arity:
        reasons.append(f"{len(undeclared_arity)} symbol(s) changed arity undeclared: "
                       f"{list(undeclared_arity)}")
    undeclared_reg_arity = tuple(
        sorted(set(evidence.arity_changed_op_registrations) - declared))
    if undeclared_reg_arity:
        reasons.append(
            f"{len(undeclared_reg_arity)} op registration(s) changed arity undeclared: "
            f"{list(undeclared_reg_arity)}. A registration is data, not a symbol: the ELF "
            "diff sees nothing, the build is clean, and the op dispatches with the wrong "
            "operand count for every shape")

    shrinkage = 1.0 - (evidence.candidate_symbol_count / evidence.anchor_symbol_count)
    if shrinkage > policy.symbol_shrinkage_reject_ratio:
        reasons.append(
            f"the exported symbol table shrank by {shrinkage:.1%}, above the policy reject "
            f"ratio {policy.symbol_shrinkage_reject_ratio:.1%} "
            f"({evidence.anchor_symbol_count} -> {evidence.candidate_symbol_count}); this is "
            "the C++ analogue of the >60% shrinkage reject that AutoPilot's source mutator "
            "lacked")
    return _gate(GID_SYMBOLS, _verdict(reasons), evidence_ref=evidence.receipt_ref,
                 notes=(f"tool={evidence.tool_id}",
                        f"added_symbols={len(evidence.added_symbols)}"))


def check_clean_build_from_snapshot(
        request: api.EvaluationRequest,
        evidence: Optional[BuildProvenance]) -> api.GateResult:
    """§8.5.1 item 2. The snapshot, a fresh build directory, and no production path."""
    if evidence is None:
        return _gate(GID_CLEAN_BUILD, _no_evidence("build provenance"))
    if evidence.produced_by != "evaluator":
        # `build_dir_was_fresh`, `incremental_objects_present` and
        # `built_from_snapshot_sha256` are otherwise taken on the producer's word,
        # which makes an actor-supplied record indistinguishable from a measured
        # one. `t0_provider.SCHEMA_FOLLOWUPS` named this branch as the remedy.
        return _gate(GID_CLEAN_BUILD, _fail(_self_reported("clean-build", evidence.produced_by)),
                     evidence_ref=evidence.build_log_ref)
    reasons = []
    if evidence.built_from_snapshot_sha256 != request.artifact.source_sha256:
        reasons.append(
            f"built from snapshot {evidence.built_from_snapshot_sha256[:12]} but the record's "
            f"artifact names {request.artifact.source_sha256[:12]}; the binary under test is "
            "not the source under test")
    if not evidence.build_dir_was_fresh:
        reasons.append(f"build directory {evidence.build_dir!r} was not fresh; an incremental "
                       "build can link stale objects and hide the error the snapshot would "
                       "surface")
    if evidence.incremental_objects_present:
        reasons.append("stale object files were present in the build directory, so the actor's "
                       "build state is part of the artifact")
    if evidence.output_binary_sha256 != request.artifact.binary_sha256:
        reasons.append(
            f"the build produced binary {evidence.output_binary_sha256[:12]} but the record "
            f"names {request.artifact.binary_sha256[:12]}")
    touched = tuple(p for p in evidence.production_tree_paths_touched)
    if touched:
        reasons.append(
            f"the build touched production tree path(s) {list(touched)}; production kernels are "
            "FROZEN and no actor builds in or modifies one (invariant 3, denial 2)")
    for path in (evidence.build_dir,):
        if _under_production_tree(path):
            reasons.append(f"build directory {path!r} is inside a production kernel tree")
    return _gate(GID_CLEAN_BUILD, _verdict(reasons), evidence_ref=evidence.build_log_ref,
                 notes=(f"compiler={evidence.compiler_id} {evidence.compiler_version}",))


def check_semantic_diff_conformance(evidence: Optional[DiffPolicyEvidence]) -> api.GateResult:
    """§8.5.1 item 3. One conceptual mutation, inside its class envelope."""
    if evidence is None:
        return _gate(GID_SEMANTIC_DIFF, _no_evidence("diff"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_SEMANTIC_DIFF,
                     _fail(_self_reported("semantic-diff", evidence.produced_by)),
                     evidence_ref=evidence.diff_ref)
    reasons = []
    declared = set(evidence.declared_surface_files)
    outside = tuple(sorted(set(evidence.files_touched) - declared))
    if outside:
        reasons.append(f"the diff touches {len(outside)} file(s) outside the declared surface: "
                       f"{list(outside)}")
    if evidence.unrelated_deletions:
        reasons.append(f"the diff contains {len(evidence.unrelated_deletions)} unrelated "
                       f"deletion(s): {list(evidence.unrelated_deletions)}")
    if evidence.changed_lines > evidence.envelope.max_changed_lines:
        reasons.append(
            f"{evidence.changed_lines} changed lines exceeds the {evidence.change_class} "
            f"envelope of {evidence.envelope.max_changed_lines}; invariant 13 requires one "
            "conceptual mutation per proposal, and a diff that has outgrown its class is no "
            "longer one")
    if len(evidence.files_touched) > evidence.envelope.max_files_touched:
        reasons.append(
            f"{len(evidence.files_touched)} files touched exceeds the {evidence.change_class} "
            f"envelope of {evidence.envelope.max_files_touched}")
    return _gate(GID_SEMANTIC_DIFF, _verdict(reasons), evidence_ref=evidence.diff_ref)


def check_schema_and_diff_policy(evidence: Optional[DiffPolicyEvidence],
                                 surface: ChangeSurface,
                                 policy: T0Policy) -> tuple:
    """§8.6 "schema and diff policy, including the diff-complexity ceiling (§10.6)".

    Returns `(GateResult, human_review_reasons)`. The ceiling does NOT fail the
    gate — §10.6 marks the package `REQUIRES_HUMAN_CODE_REVIEW` and says so on its
    first page, which is a review obligation, not a claim that the kernel is
    wrong. The marker rides out on the gate's `notes` as well as in the returned
    tuple, because `api.TierGateRunner.run_gates` hands back only gate results and
    a marker that reaches nothing is not a marker (see `SEAMS`).
    """
    if evidence is None:
        return _gate(GID_SCHEMA_DIFF_POLICY, _no_evidence("diff/schema")), ()
    if evidence.produced_by != "evaluator":
        # `commit_was_pathspec_limited` is the field that matters most here: in a
        # shared clone an unrestricted commit sweeps another session's staged files
        # into the artifact, and the candidate was previously believed about it.
        return _gate(GID_SCHEMA_DIFF_POLICY,
                     _fail(_self_reported("schema/diff-policy", evidence.produced_by)),
                     evidence_ref=evidence.diff_ref), ()

    reasons = []
    if evidence.record_schema_violations:
        reasons.append(f"the candidate's own records fail their schema: "
                       f"{list(evidence.record_schema_violations)}")
    prod_paths = tuple(p for p in evidence.files_touched if _under_production_tree(p))
    prod_paths = tuple(sorted(set(prod_paths) | set(evidence.production_tree_paths)))
    if prod_paths:
        reasons.append(f"the diff names production tree path(s) {list(prod_paths)}; denial 2 "
                       "forbids a production write of any kind")
    if _PRODUCTION_BRANCH_RE.match(evidence.branch_name):
        reasons.append(f"the candidate is on production-named branch "
                       f"{evidence.branch_name!r}; ALL kernel work happens on "
                       "llama.cpp-experimental branches")
    if not evidence.commit_was_pathspec_limited:
        reasons.append("the candidate's commit was not pathspec-limited; in a shared clone an "
                       "unrestricted commit sweeps another session's staged files into the "
                       "artifact")

    review = []
    ceiling = policy.diff_ceiling
    if evidence.changed_lines > ceiling.max_changed_lines:
        review.append(f"{evidence.changed_lines} changed lines exceeds the {ceiling.backend} "
                      f"adapter ceiling of {ceiling.max_changed_lines} (§10.6)")
    if len(evidence.files_touched) > ceiling.max_files_touched:
        review.append(f"{len(evidence.files_touched)} files touched exceeds the "
                      f"{ceiling.backend} adapter ceiling of {ceiling.max_files_touched} "
                      "(§10.6)")
    if surface.touches_shared_core_header and ceiling.shared_core_forces_review:
        review.append("the change touches shared ggml core or a widely-included header; "
                      "core_header is a risk tier, not a size band, and forces human review "
                      "regardless of the §10.6 ceiling")
    if evidence.change_class == "core_header":
        review.append("change_class=core_header forces REQUIRES_HUMAN_CODE_REVIEW regardless "
                      "of diff size")

    notes = [f"change_class={evidence.change_class}", f"ceiling={ceiling.backend}"]
    if review:
        notes.append(f"{REQUIRES_HUMAN_CODE_REVIEW}: " + "; ".join(review))
    return (_gate(GID_SCHEMA_DIFF_POLICY, _verdict(reasons), evidence_ref=evidence.diff_ref,
                  notes=notes),
            tuple(review))


def check_static_and_compile(evidence: Optional[StaticAnalysisEvidence],
                             anchor: Optional[api.AnchorIdentity]) -> api.GateResult:
    """§8.6 "static/compile checks", plus toolchain identity against the anchor.

    `anchor` is a parameter — with no default — because three of the fields this
    gate reads are the ANCHOR's build, not the candidate's. A capture that names a
    different anchor raises `StaticAnalysisAnchorMismatch`; a capture that names
    none cannot have its toolchain fields attributed to the bound anchor, so the
    gate's PASS becomes COULD_NOT_CHECK. Neither is a default: the first is a
    refusal, and the second says out loud what was not established.

    With `anchor=None` there is no identity for a capture to disagree WITH, so
    this gate answers as it always did — an anchor-less T0 run is already
    structurally INVALID (`api.compute_verdict`), and this gate is not the place
    that says so.
    """
    if evidence is None:
        return _gate(GID_STATIC_COMPILE, _no_evidence("static/compile"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_STATIC_COMPILE,
                     _fail(_self_reported("static/compile", evidence.produced_by)),
                     evidence_ref=evidence.receipt_ref)

    recorded = evidence.recorded_anchor()
    _refuse_replay_mismatch(
        anchor, recorded, label="static-analysis",
        consequence=("The anchor compiler and warning count it carries describe another "
                     "anchor's build, so comparing them with this candidate's is the "
                     "toolchain confound this gate exists to catch, arriving through the "
                     "gate itself"),
        error=StaticAnalysisAnchorMismatch)

    reasons = []
    unknown = []
    if anchor is not None and recorded is None:
        unknown.append(
            "the static-analysis capture records no anchor identity, so the anchor compiler "
            "and warning count it carries cannot be attributed to the anchor this run names; "
            "an unrecorded identity is not agreement with the bound anchor")
    if evidence.error_count:
        reasons.append(f"{evidence.error_count} compiler error(s)")
    if evidence.analyzer_error_findings:
        reasons.append(f"{len(evidence.analyzer_error_findings)} static-analyzer finding(s) at "
                       f"error severity: {list(evidence.analyzer_error_findings)}")
    if (evidence.compiler_id, evidence.compiler_version) != \
            (evidence.anchor_compiler_id, evidence.anchor_compiler_version):
        reasons.append(
            f"the candidate was built with {evidence.compiler_id} "
            f"{evidence.compiler_version} but the anchor with {evidence.anchor_compiler_id} "
            f"{evidence.anchor_compiler_version}; that is a toolchain comparison wearing a "
            "kernel comparison's clothes")
    if evidence.warnings_as_errors:
        # A clean -Werror build already proves the new-warning delta is zero:
        # a new warning would have been an error and `error_count` would be > 0.
        pass
    elif evidence.anchor_warning_count is None:
        unknown.append("the build does not use -Werror and no anchor warning count was "
                       "recorded, so a new warning cannot be detected")
    elif evidence.warning_count > evidence.anchor_warning_count:
        reasons.append(f"{evidence.warning_count - evidence.anchor_warning_count} new "
                       f"compiler warning(s) versus the anchor "
                       f"({evidence.anchor_warning_count} -> {evidence.warning_count})")
    notes = (f"analyzer={evidence.analyzer_id}",
             f"capture_anchor={'unrecorded' if recorded is None else recorded.short()}")
    if reasons:
        return _gate(GID_STATIC_COMPILE, _fail(*reasons, *unknown),
                     evidence_ref=evidence.receipt_ref, notes=notes)
    if unknown:
        return _gate(GID_STATIC_COMPILE, _cnc(*unknown), evidence_ref=evidence.receipt_ref,
                     notes=notes)
    return _gate(GID_STATIC_COMPILE, schemas.Check(schemas.PASS),
                 evidence_ref=evidence.receipt_ref, notes=notes)


def _sanitizer_preamble(request: api.EvaluationRequest,
                        evidence: Optional[SanitizerEvidence],
                        surface: ChangeSurface) -> tuple:
    """Shared ASAN/UBSAN preconditions. Returns `(check_or_None, notes)`.

    A `check` means the sanitizer question is already answered for both gates
    (not run, not runnable, or not mandatory); `None` means go look at findings.
    """
    mandatory = surface.sanitizers_mandatory
    if evidence is None:
        if mandatory is True:
            return _fail(
                "this change touches memory or threading and no ASAN/UBSAN build and "
                "targeted run was recorded; §8.6 makes it MANDATORY, not advisory "
                f"(derivation {surface.derivation_ref})"), ()
        if mandatory is None:
            return _cnc(
                "the derivation did not determine whether this change touches memory or "
                "threading, so the mandatory ASAN/UBSAN surface cannot be ruled out; "
                "'we could not tell' is not 'it does not' "
                f"(derivation {surface.derivation_ref})"), ()
        return schemas.Check(
            schemas.PASS,
            ("ASAN/UBSAN is not mandatory for this change: the mechanical derivation at "
             f"{surface.derivation_ref} finds it touches neither memory nor threading",)), ()

    if evidence.produced_by != "evaluator":
        return _fail(_self_reported("sanitizer", evidence.produced_by)), ()
    if not evidence.executed:
        return _fail("an ASAN/UBSAN invocation was constructed but never executed; a built "
                     "sanitizer that did not run proves nothing"), ()
    invocation = check_sanitizer_invocation(evidence.invocation)
    if invocation.outcome != schemas.PASS:
        return schemas.Check(
            invocation.outcome,
            ("the sanitizer invocation would not have gated:",) + tuple(invocation.reasons)), ()
    notes = [f"recipe={evidence.invocation.receipt.render()}"]
    if evidence.sanitizer_build_binary_sha256 == request.artifact.binary_sha256:
        return _fail(
            "the sanitizer build's binary is the same binary the record measures; an "
            "ASAN/UBSAN build is instrumented and is not a performance artifact, so the "
            "two must be distinct builds"), ()
    # LAST, so it can never downgrade one of the FAILs above: an unrecorded exit
    # code is COULD_NOT_CHECK, not a pass. `halt_on_error=1` means UBSAN's answer
    # IS the exit status, and an empty findings list from a log nobody parsed is
    # evidence about the parser, not about the kernel.
    if evidence.exit_code is None:
        return _cnc(
            "the sanitizer run's exit code was not recorded, so a clean findings list "
            "cannot be distinguished from an unparsed log; -fno-sanitize-recover=all and "
            "halt_on_error=1 make the exit status the sanitizer's verdict"), ()
    return None, tuple(notes)


def check_asan(request: api.EvaluationRequest,
               evidence: Optional[SanitizerEvidence],
               surface: ChangeSurface) -> api.GateResult:
    """§8.6 mandatory ASAN half. Memory-safety findings -> `stability`."""
    early, notes = _sanitizer_preamble(request, evidence, surface)
    if early is not None:
        return _gate(GID_ASAN, early)
    reasons = []
    if evidence.asan_findings:
        reasons.append(f"{len(evidence.asan_findings)} AddressSanitizer finding(s): "
                       f"{list(evidence.asan_findings)}")
    if evidence.exit_code not in (0, None) and not evidence.asan_findings:
        reasons.append(f"the sanitizer run exited {evidence.exit_code} with no parsed finding; "
                       "an unexplained non-zero exit is not a pass")
    return _gate(GID_ASAN, _verdict(reasons), evidence_ref=evidence.log_ref, notes=notes)


def check_ubsan(request: api.EvaluationRequest,
                evidence: Optional[SanitizerEvidence],
                surface: ChangeSurface) -> api.GateResult:
    """§8.6 mandatory UBSAN half. Undefined behaviour -> `numerical_safety`."""
    early, notes = _sanitizer_preamble(request, evidence, surface)
    if early is not None:
        return _gate(GID_UBSAN, early)
    reasons = []
    if evidence.ubsan_findings:
        reasons.append(f"{len(evidence.ubsan_findings)} UndefinedBehaviorSanitizer finding(s): "
                       f"{list(evidence.ubsan_findings)}")
    return _gate(GID_UBSAN, _verdict(reasons), evidence_ref=evidence.log_ref, notes=notes)


def check_backend_op_units(request: api.EvaluationRequest,
                           evidence: Optional[OpSuiteEvidence],
                           surface: ChangeSurface,
                           policy: T0Policy) -> api.GateResult:
    """§8.6 "targeted backend-op unit shapes" — including MUL_MAT_ID.

    The required set is the policy's mandatory ops UNION the mechanically derived
    affected ops (§6.4 item 1). A required op that was not exercised is a FAIL and
    not a pass with a smaller denominator: `kernel_eval.sh` reported
    `MUL_MAT 4231/4231 OK` and that sentence was true — it was just not a
    statement about `MUL_MAT_ID`, which it never ran.
    """
    if evidence is None:
        return _gate(GID_OP_UNITS, _no_evidence("backend-op unit suite"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_OP_UNITS, _fail(_self_reported("op-suite", evidence.produced_by)),
                     evidence_ref=evidence.receipt_ref)

    required = tuple(dict.fromkeys(tuple(policy.required_backend_ops) + surface.derived_ops))
    exercised = set(evidence.ops_exercised)
    reasons = []

    if evidence.suite_source_sha256 != request.artifact.source_sha256:
        reasons.append(
            f"the op suite was built from source {evidence.suite_source_sha256[:12]} but the "
            f"candidate is {request.artifact.source_sha256[:12]}; a suite built from another "
            "tree says nothing about this candidate")
    missing = tuple(op for op in required if op not in exercised)
    if missing:
        reasons.append(
            f"required op(s) not exercised at all: {list(missing)}. An untested op is not a "
            "passing op — the replaced gate ran MUL_MAT only, and every MoE expert path went "
            f"unexercised. Required set = policy {list(policy.required_backend_ops)} + derived "
            f"surface {list(surface.derived_ops)}")
    if evidence.ops_failed:
        reasons.append(f"op(s) failed: {list(evidence.ops_failed)}")
    if evidence.layout_probe:
        required_layouts = {"offset", "transpose", "stride_gap"}
        missing_layouts = sorted(required_layouts - set(evidence.layout_families))
        if evidence.layout_case_count == 0:
            reasons.append(
                "the layout pass selected zero cases; a flag over no layout is not a probe")
        if missing_layouts:
            reasons.append(
                f"layout pass did not exercise required family/families {missing_layouts}; "
                "required=['offset', 'stride_gap', 'transpose']")
    if evidence.value_transform_probe:
        required_transforms = {"identity", "x3", "x0p01", "negate"}
        missing_transforms = sorted(
            required_transforms - set(evidence.value_transforms))
        if evidence.value_transform_case_count == 0:
            reasons.append(
                "the value-transform pass selected zero cases; a flag over no floating "
                "input is not a probe")
        if missing_transforms:
            reasons.append(
                f"value-transform pass did not exercise required transform(s) "
                f"{missing_transforms}; required=['identity', 'negate', 'x0p01', 'x3']")
    if evidence.stateful_probe:
        required_stateful = {
            "SSM_SCAN", "SSM_CONV", "FLASH_ATTN_EXT", "GATED_DELTA_NET"}
        missing_stateful = sorted(required_stateful - set(evidence.stateful_ops))
        if evidence.stateful_case_count == 0:
            reasons.append(
                "the stateful pass selected zero cases; a flag over no recurrent state is not a probe")
        if missing_stateful:
            reasons.append(
                f"stateful pass did not exercise required op(s) {missing_stateful}; "
                "required=['FLASH_ATTN_EXT', 'GATED_DELTA_NET', 'SSM_CONV', 'SSM_SCAN']")
    for op in required:
        if op not in exercised:
            continue
        cases = evidence.cases_for(op)
        if cases is None:
            reasons.append(f"op {op!r} is listed as exercised but reports no case counts; "
                           "'exercised' with no cases is a name in a list")
            continue
        total, passed = cases
        if total == 0:
            reasons.append(f"op {op!r} ran zero cases; a suite that selected no shape for an "
                           "op did not test it")
        elif passed != total:
            reasons.append(f"op {op!r}: {passed}/{total} cases passed")
    return _gate(GID_OP_UNITS, _verdict(reasons), evidence_ref=evidence.receipt_ref,
                 notes=(f"suite={evidence.suite_id}", f"required={list(required)}",
                        f"shapes={evidence.shapes_ref}",
                        f"layout_probe={evidence.layout_probe}",
                        f"layout_families={list(evidence.layout_families)}",
                        f"layout_cases={evidence.layout_case_count}",
                        f"value_transform_probe={evidence.value_transform_probe}",
                        f"value_transforms={list(evidence.value_transforms)}",
                        f"value_transform_cases={evidence.value_transform_case_count}",
                        f"stateful_probe={evidence.stateful_probe}",
                        f"stateful_ops={list(evidence.stateful_ops)}",
                        f"stateful_cases={evidence.stateful_case_count}"),
                 measurements=tuple(item.to_dict()
                                    for item in evidence.property_measurements))


def check_exact_reference_comparison(request: api.EvaluationRequest,
                                     evidence: _Maybe,
                                     surface: ChangeSurface,
                                     policy: T0Policy) -> api.GateResult:
    """§8.6 "exact reference comparisons where defined" (oracles per §6.5)."""
    if request.anchor is None:
        return _gate(GID_EXACT_REFERENCE, _no_anchor("exact reference comparison"))
    if isinstance(evidence, NotApplicable):
        return _gate(GID_EXACT_REFERENCE, schemas.Check(schemas.PASS, (evidence.note(),)),
                     notes=(evidence.note(),))
    if evidence is None:
        return _gate(GID_EXACT_REFERENCE, _no_evidence("exact reference comparison"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_EXACT_REFERENCE,
                     _fail(_self_reported("reference-comparison", evidence.produced_by)))

    reasons = []
    covered = set()
    for comparison in evidence.comparisons:
        covered.add(comparison.op)
        if comparison.oracle_is_candidate_derived:
            reasons.append(
                f"shape {comparison.shape_id!r} was compared against oracle "
                f"{comparison.oracle_id!r}, which is derived from the candidate's own output; "
                "'A candidate output MUST NEVER be cached or reused as a correctness oracle'")
            continue
        if comparison.mode == "exact_bitwise" and comparison.mismatch_count:
            reasons.append(f"shape {comparison.shape_id!r} ({comparison.op}): "
                           f"{comparison.mismatch_count} exact-reference mismatch(es)")
        if comparison.mode == "ulp_bounded":
            if comparison.max_ulp_observed is None:
                reasons.append(f"shape {comparison.shape_id!r}: ulp_bounded comparison recorded "
                               "no observed ULP, so the tolerance was never applied")
            elif comparison.max_ulp_observed > comparison.tolerance_ulp:
                reasons.append(
                    f"shape {comparison.shape_id!r} ({comparison.op}): max ULP "
                    f"{comparison.max_ulp_observed} exceeds the declared tolerance "
                    f"{comparison.tolerance_ulp}")
        if comparison.mode == "metric_bounded":
            if comparison.max_error_observed > comparison.tolerance_error:
                reasons.append(
                    f"shape {comparison.shape_id!r} ({comparison.op}): "
                    f"{comparison.metric_id} error {comparison.max_error_observed} exceeds "
                    f"the declared tolerance {comparison.tolerance_error}")
    declared_undefined = {op for op, _ in evidence.undefined_for}
    required = tuple(dict.fromkeys(tuple(policy.required_backend_ops) + surface.derived_ops))
    silent = tuple(op for op in required
                   if op not in covered and op not in declared_undefined)
    if silent:
        reasons.append(
            f"op(s) {list(silent)} were neither compared against a reference nor declared as "
            "having none; 'where defined' is checkable only if undefined is DECLARED, and "
            "silence is not a reference")
    return _gate(GID_EXACT_REFERENCE, _verdict(reasons),
                 evidence_ref=evidence.oracle_registry_ref,
                 notes=(f"comparisons={len(evidence.comparisons)}",
                        f"undefined_for={sorted(declared_undefined)}"))


def check_unseen_boundary_shapes(evidence: _Maybe,
                                 surface: ChangeSurface) -> api.GateResult:
    """§8.6 "unseen/boundary shapes for dispatch changes"."""
    touches_dispatch = surface.derived_touches_dispatch
    if isinstance(evidence, NotApplicable):
        contradiction = _na_contradiction(evidence, touches_dispatch, "modifies dispatch")
        if contradiction:
            return _gate(GID_BOUNDARY_SHAPES, _fail(contradiction), notes=(evidence.note(),))
        return _gate(GID_BOUNDARY_SHAPES, schemas.Check(schemas.PASS, (evidence.note(),)),
                     notes=(evidence.note(),))
    if evidence is None:
        if touches_dispatch is True:
            return _gate(GID_BOUNDARY_SHAPES, _fail(
                "this change modifies dispatch and no unseen/boundary shapes were exercised; "
                "a dispatch change validated only on shapes it was written against is an "
                f"overfit, not a kernel (derivation {surface.derivation_ref})"))
        if touches_dispatch is None:
            return _gate(GID_BOUNDARY_SHAPES, _cnc(
                "the derivation did not determine whether this change touches dispatch, so "
                "the unseen/boundary-shape requirement cannot be ruled out "
                f"(derivation {surface.derivation_ref})"))
        return _gate(GID_BOUNDARY_SHAPES, schemas.Check(
            schemas.PASS,
            ("not a dispatch change per the mechanical derivation at "
             f"{surface.derivation_ref}",)))

    reasons = []
    if evidence.produced_by != "evaluator":
        reasons.append(_self_reported("boundary-shape", evidence.produced_by))
    if not evidence.held_out_from_planner:
        reasons.append(
            "the 'unseen' shapes were visible to the planner, so they are not unseen; a "
            "search that can see its own holdout will find the kernel that special-cases it")
    if not evidence.unseen_shapes:
        reasons.append("no unseen shapes were exercised")
    if not evidence.boundary_shapes:
        reasons.append("no boundary shapes were exercised")
    if evidence.failures:
        reasons.append(f"{len(evidence.failures)} unseen/boundary shape failure(s): "
                       f"{list(evidence.failures)}")
    return _gate(GID_BOUNDARY_SHAPES, _verdict(reasons), evidence_ref=evidence.receipt_ref,
                 notes=(f"rule={evidence.selection_rule_id}",
                        f"seed={evidence.selection_seed}",
                        f"unseen={len(evidence.unseen_shapes)}",
                        f"boundary={len(evidence.boundary_shapes)}"))


def check_affected_surface_reconciliation(
        evidence: Optional[DispatchTraceEvidence],
        surface: ChangeSurface) -> api.GateResult:
    """§6.4 item 3: `derived ⊇ traced` must hold; `traced ⊄ derived` is a hard failure."""
    if evidence is None:
        return _gate(GID_SURFACE_RECONCILIATION, _no_evidence("dispatch trace"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_SURFACE_RECONCILIATION,
                     _fail(_self_reported("dispatch-trace", evidence.produced_by)),
                     evidence_ref=evidence.trace_ref)

    reasons = []
    derived = set(evidence.derived_surface)
    traced = set(evidence.traced_kernels)
    if not traced:
        reasons.append("the trace recorded no executed kernel at all, so the affected surface "
                       "was never exercised; an empty trace does not reconcile with anything")
    extra = tuple(sorted(traced - derived))
    if extra:
        reasons.append(
            f"traced ⊄ derived: {list(extra)} executed but is outside the derived affected "
            "surface. Over-approximation is expected and acceptable; under-approximation is a "
            "hard candidate failure, because freeze scope, lineage composition and sentinel "
            "selection all key off this manifest")
    if set(surface.derived_ops) - derived:
        reasons.append(
            f"the trace's derived surface omits ops the change-surface derivation found: "
            f"{sorted(set(surface.derived_ops) - derived)}; two derivations of one surface "
            "disagreeing is itself the defect")
    notes = [f"derived={len(derived)}", f"traced={len(traced)}"]
    for row in surface.prediction_score():
        notes.append(f"actor_prediction:{row[0]}={'hit' if row[3] else 'miss'}")
    return _gate(GID_SURFACE_RECONCILIATION, _verdict(reasons), evidence_ref=evidence.trace_ref,
                 notes=notes)


def check_no_fallback_dispatch_proof(
        evidence: Optional[DispatchTraceEvidence]) -> api.GateResult:
    """§8.6 no-fallback proof; control 3's *"silently falling back"*.

    Filed under `integrity` and not `mechanism` on purpose — see `T0_GATE_SPEC`.
    An uninstrumented trace yields COULD_NOT_CHECK, never PASS: zero fallback
    events from an instrument that cannot record one is a fact about the
    instrument.
    """
    if evidence is None:
        return _gate(GID_NO_FALLBACK, _no_evidence("dispatch trace"))
    # The SAME DispatchTraceEvidence object drives check_affected_surface_
    # reconciliation, which refuses a candidate-produced trace. This gate read the
    # object and skipped the field: a candidate could hand in
    # `fallback_events=(), fallback_instrumentation_active=True` and PASS control
    # 3's own detector on its own word.
    if evidence.produced_by != "evaluator":
        return _gate(GID_NO_FALLBACK, _fail(_self_reported("dispatch-trace",
                                                           evidence.produced_by)),
                     evidence_ref=evidence.trace_ref)
    if not evidence.fallback_instrumentation_active:
        return _gate(GID_NO_FALLBACK, _cnc(
            "the trace was collected without fallback instrumentation, so an empty "
            "fallback-event list is evidence of no instrumentation and not of no fallback"),
            evidence_ref=evidence.trace_ref)
    if evidence.fallback_events:
        return _gate(GID_NO_FALLBACK, _fail(
            f"{len(evidence.fallback_events)} fallback event(s) observed: "
            f"{list(evidence.fallback_events)}. A candidate that silently falls back to the "
            "generic path is control 3's degraded-negative shape and MUST receive no speed "
            "rank at all"), evidence_ref=evidence.trace_ref)
    return _gate(GID_NO_FALLBACK, schemas.Check(schemas.PASS), evidence_ref=evidence.trace_ref)


def check_state_rollback_teardown_race(evidence: _Maybe,
                                       surface: ChangeSurface) -> api.GateResult:
    """§8.6 "state, rollback, teardown, and race tests where relevant"."""
    # Three-valued, NOT `False or None -> False`. persistent_state=False with
    # threading=None used to return PASS with the reason "no persistent state and
    # no threading per the mechanical derivation" — a claim about threading the
    # derivation never made.
    relevant = _any_true(surface.derived_touches_persistent_state,
                         surface.derived_touches_threading)
    if isinstance(evidence, NotApplicable):
        contradiction = _na_contradiction(evidence, relevant,
                                          "touches persistent state or threading")
        if contradiction:
            return _gate(GID_STATE_SAFETY, _fail(contradiction), notes=(evidence.note(),))
        return _gate(GID_STATE_SAFETY, schemas.Check(schemas.PASS, (evidence.note(),)),
                     notes=(evidence.note(),))
    if evidence is None:
        if relevant is True:
            return _gate(GID_STATE_SAFETY, _fail(
                "this change touches persistent state or threading and no state/rollback/"
                "teardown/race evidence was recorded "
                f"(derivation {surface.derivation_ref})"))
        if relevant is None:
            return _gate(GID_STATE_SAFETY, _cnc(
                "the derivation did not determine whether this change touches persistent "
                "state or threading "
                f"(derivation {surface.derivation_ref})"))
        return _gate(GID_STATE_SAFETY, schemas.Check(
            schemas.PASS, ("no persistent state and no threading per the mechanical "
                           f"derivation at {surface.derivation_ref}",)))

    reasons = []
    if evidence.produced_by != "evaluator":
        reasons.append(_self_reported("state-safety", evidence.produced_by))
    if not evidence.rollback_tested:
        reasons.append("rollback was not tested")
    if not evidence.teardown_tested:
        reasons.append("teardown was not tested")
    if surface.derived_touches_threading is True and evidence.race_detector_id is None:
        reasons.append("this change touches threading and no race detector was run")
    if evidence.race_findings:
        reasons.append(f"race finding(s): {list(evidence.race_findings)}")
    if evidence.leaked_resources:
        reasons.append(f"leaked resource(s) after teardown: {list(evidence.leaked_resources)}")
    if evidence.orphan_processes:
        reasons.append(
            f"orphan process(es) survived teardown: {list(evidence.orphan_processes)}; "
            "invariant 10 requires the loop to verify termination of what it launched")
    return _gate(GID_STATE_SAFETY, _verdict(reasons), evidence_ref=evidence.receipt_ref)


def check_output_coherence(request: api.EvaluationRequest,
                           evidence: Optional[CoherenceEvidence],
                           policy: T0Policy,
                           determinism: Optional[DeterminismEvidence] = None) -> tuple:
    """The coherence gate. Returns `(GateResult, CoherenceVerdict)`.

    This is the gate `kernel_eval.sh` got wrong, and it is the reason
    `CoherenceVerdict` exists as a type at all rather than a string variable.

    `determinism` is passed so the ONE evidence field that can turn a byte
    difference into an equivalence label — the anchor's determinism class — is
    reconciled instead of taken on the coherence capture's word. Two records of
    one anchor property, never compared, is a second source of truth, and the
    tolerance branch is the only place in this module where a self-declared field
    can produce a PASS from differing outputs.

    That reconciliation compares the two records' ANCHOR IDENTITIES as well as
    their claims, because agreement between two records is corroboration only if
    they describe the same anchor. Two captures of *different* anchors that happen
    to agree read exactly like corroboration and are none, so:

      * identities named and different — RAISE `DeterminismAnchorMismatch`. The
        wrong material was reconciled; that is a replay defect, not a finding.
      * identities agree, classes disagree — the pre-existing FAIL, unchanged.
      * the determinism record names no anchor — COULD_NOT_CHECK. The two records
        agreeing establishes nothing about whether they describe one anchor, and
        the tolerance may not rest on it. (The coherence side cannot be unrecorded
        here: `_derive_coherence` sends an unrecorded capture to `not_compared`,
        so this branch is unreachable without it.)

    A capture taken against a DIFFERENT anchor than the record names does not
    reach a gate result at all: `compute_coherence` raises
    `CoherenceAnchorMismatch` and it propagates, exactly as `CoherenceTampering`
    does. A report is a statement that seventeen surfaces were examined, and one
    examined against the wrong anchor was not examined.
    """
    verdict = compute_coherence(anchor=request.anchor, evidence=evidence,
                                tolerance_floor=policy.coherence_tolerance_floor)
    check = verdict.to_check()
    if evidence is not None and evidence.produced_by != "evaluator" and \
            check.outcome == schemas.PASS:
        check = _fail(_self_reported("coherence", evidence.produced_by))
    if verdict.label == COHERENCE_WITHIN_TOLERANCE and check.outcome == schemas.PASS:
        if determinism is None:
            check = _cnc(
                "the outputs differ and the equivalence label rests entirely on the "
                f"coherence capture's own claim that the anchor is "
                f"{verdict.anchor_determinism_class!r}; no determinism evidence was "
                "captured to reconcile that against, so the tolerance was applied on an "
                "unchecked declaration",
                *check.reasons)
        else:
            # Before the classes are compared at all: reconciling two records that
            # describe different anchors produces agreement about nothing.
            _refuse_replay_mismatch(
                evidence.recorded_anchor(), determinism.recorded_anchor(),
                label="determinism",
                consequence=("It is being reconciled against a coherence capture taken "
                             "against a different anchor, and the tolerance branch would "
                             "read that as two independent records corroborating one "
                             "anchor's determinism class"),
                error=DeterminismAnchorMismatch)
            if determinism.anchor_determinism_class != verdict.anchor_determinism_class:
                check = _fail(
                    f"the coherence capture declares the anchor "
                    f"{verdict.anchor_determinism_class!r} and the determinism evidence "
                    f"declares it {determinism.anchor_determinism_class!r}. The declared "
                    "tolerance is admissible ONLY when byte identity is unattainable by "
                    "construction; two records of one anchor property disagreeing is not "
                    "that, and the equivalence label does not survive it",
                    *check.reasons)
            elif determinism.recorded_anchor() is None:
                check = _cnc(
                    "the outputs differ and the tolerance rests on two records agreeing that "
                    f"the anchor is {verdict.anchor_determinism_class!r}, but the determinism "
                    "evidence records no anchor identity, so their agreement does not "
                    "establish that they describe the SAME anchor; corroboration requires "
                    "both records to name the anchor they describe",
                    *check.reasons)
    notes = [f"label={verdict.label}"]
    if evidence is None:
        notes.append("no coherence evidence was captured for this candidate")
    else:
        notes.append(f"sampler={evidence.sampler_id}")
        notes.append(f"seed={evidence.seed}")
        recorded = evidence.recorded_anchor()
        notes.append(f"capture_anchor={'unrecorded' if recorded is None else recorded.short()}")
    gate = _gate(GID_COHERENCE, check,
                 evidence_ref=None if evidence is None else evidence.receipt_ref,
                 notes=notes)
    return gate, verdict


def check_determinism_class(request: api.EvaluationRequest,
                            evidence: Optional[DeterminismEvidence],
                            policy: T0Policy) -> tuple:
    """§8.6 determinism-class check; invariant 12. Returns `(GateResult, properties)`.

    `properties` is the release-relevant-property list: a DECLARED change of
    determinism class is not a failure, it is *"a declared, release-relevant
    property"* that must travel with the candidate. An UNDECLARED change is a
    failure, because the class is an interface and a candidate may not silently
    change it.

    Every comparison below that reads `anchor_*` off the evidence is a comparison
    against SOME anchor, so the capture is bound to the anchor the record names
    first: a different one raises `DeterminismAnchorMismatch` (the class change it
    would report is between two anchors, not between anchor and candidate), and an
    unrecorded one is COULD_NOT_CHECK-shaped — it never turns an anchor comparison
    that could not be attributed into a PASS, and never downgrades a FAIL either.
    """
    if request.anchor is None:
        return _gate(GID_DETERMINISM, _no_anchor("the determinism-class comparison")), ()
    if evidence is None:
        return _gate(GID_DETERMINISM, _no_evidence("determinism")), ()
    if evidence.produced_by != "evaluator":
        return (_gate(GID_DETERMINISM, _fail(_self_reported("determinism",
                                                            evidence.produced_by)),
                      evidence_ref=evidence.receipt_ref), ())

    recorded = evidence.recorded_anchor()
    _refuse_replay_mismatch(
        request.anchor, recorded, label="determinism",
        consequence=("The determinism class and output digests it carries are another "
                     "anchor's, so any class change derived from them is a difference "
                     "between two anchors rather than between this anchor and this "
                     "candidate"),
        error=DeterminismAnchorMismatch)

    measured = evidence.measured_class()
    reasons = []
    unknown = []
    properties = []

    if recorded is None:
        unknown.append(
            "the determinism capture records no anchor identity, so the anchor determinism "
            "class and anchor output digests it carries cannot be attributed to the anchor "
            "this run names; an unrecorded identity is not agreement with the bound anchor")

    if evidence.runs < policy.determinism_min_runs:
        unknown.append(
            f"{evidence.runs} same-seed run(s) is below the policy minimum "
            f"{policy.determinism_min_runs}; a determinism class cannot be established from "
            "fewer repeats than the policy requires")
    if len(evidence.candidate_output_digests) != evidence.runs:
        reasons.append(f"{evidence.runs} runs were declared but "
                       f"{len(evidence.candidate_output_digests)} output digests were "
                       "recorded; the reduction cannot be recomputed from that")
    if measured != request.determinism.determinism_class:
        reasons.append(
            f"the evaluator measured determinism class {measured!r} but the record carries "
            f"{request.determinism.determinism_class!r}; a correctness property is produced by "
            "the evaluator and is never self-reported by the candidate")
    if request.determinism.same_seed_repeat_runs != evidence.runs:
        reasons.append(
            f"the record declares {request.determinism.same_seed_repeat_runs} same-seed "
            f"repeats but {evidence.runs} were run")

    if measured != "not_measured" and measured != evidence.anchor_determinism_class:
        if evidence.declared_class_change:
            properties.append(
                f"determinism class changes {evidence.anchor_determinism_class} -> {measured} "
                f"(declared at {evidence.declared_class_change_ref}); invariant 12 makes this a "
                "release-relevant property that must travel with the candidate")
        else:
            reasons.append(
                f"the determinism class changed {evidence.anchor_determinism_class} -> "
                f"{measured} and the change was NOT declared; 'A candidate may not silently "
                "change same-seed run-to-run bitwise stability; a change of class is a "
                "declared, release-relevant property' (invariant 12)")
    if evidence.anchor_determinism_class == "bitwise_stable" and evidence.anchor_output_digests:
        anchor_first = evidence.anchor_output_digests[0]
        if not all(d == anchor_first for d in evidence.anchor_output_digests):
            reasons.append(
                "the anchor is recorded as bitwise_stable but its own repeated runs differ; "
                "the anchor's declared class and its measured behaviour disagree, so no "
                "determinism comparison against it means anything")

    notes = [f"measured={measured}", f"anchor={evidence.anchor_determinism_class}",
             f"seed={evidence.seed}", f"runs={evidence.runs}",
             f"capture_anchor={'unrecorded' if recorded is None else recorded.short()}"]
    for prop in properties:
        notes.append(f"RELEASE_RELEVANT_PROPERTY: {prop}")
    if reasons:
        return (_gate(GID_DETERMINISM, _fail(*reasons, *unknown),
                      evidence_ref=evidence.receipt_ref, notes=notes), tuple(properties))
    if unknown:
        return (_gate(GID_DETERMINISM, _cnc(*unknown), evidence_ref=evidence.receipt_ref,
                      notes=notes), tuple(properties))
    return (_gate(GID_DETERMINISM, schemas.Check(schemas.PASS, tuple(properties)),
                  evidence_ref=evidence.receipt_ref, notes=notes), tuple(properties))


def check_binary_and_linkage_identity(request: api.EvaluationRequest,
                                      evidence: Optional[LinkageEvidence],
                                      control_role: Optional[str]) -> api.GateResult:
    """§8.6 "binary/linkage identity", including the shared-ggml linkage scar."""
    if request.anchor is None:
        return _gate(GID_LINKAGE, _no_anchor("binary and linkage identity"))
    if evidence is None:
        return _gate(GID_LINKAGE, _no_evidence("binary/linkage identity"))
    if evidence.produced_by != "evaluator":
        return _gate(GID_LINKAGE, _fail(_self_reported("linkage", evidence.produced_by)),
                     evidence_ref=evidence.receipt_ref)

    reasons = []
    unknown = []
    if evidence.binary_sha256 != request.artifact.binary_sha256:
        reasons.append(f"the verified binary {evidence.binary_sha256[:12]} is not the binary "
                       f"the record names ({request.artifact.binary_sha256[:12]})")
    if evidence.linkage_sha256 != request.artifact.linkage_sha256:
        reasons.append(f"the verified linkage {evidence.linkage_sha256[:12]} is not the "
                       f"linkage the record names ({request.artifact.linkage_sha256[:12]})")
    if evidence.anchor_source_commit is None:
        # `_validate_anchor_triple` has already refused a partial naming, so one
        # absent component means all three are absent: nothing was captured to
        # re-verify, which is COULD_NOT_CHECK and never a pass.
        unknown.append("the anchor's source commit and binary/linkage digests were not "
                       "captured at verification time, so anchor identity could not be "
                       "re-verified here")
    else:
        if evidence.anchor_source_commit != request.anchor.source_commit:
            reasons.append(
                f"the anchor source commit verified ({evidence.anchor_source_commit[:12]}) is "
                f"not the anchor the record names ({request.anchor.source_commit[:12]}); "
                "precondition 4 names an anchor by all three of source commit, binary "
                "SHA-256 and linkage SHA-256, and a different commit is a different anchor "
                "even when a digest happens to agree")
        if evidence.anchor_binary_sha256 != request.anchor.binary_sha256:
            reasons.append(
                f"the anchor binary verified ({evidence.anchor_binary_sha256[:12]}) is not the "
                f"anchor the record names ({request.anchor.binary_sha256[:12]}); a rebuilt "
                "anchor is a different anchor")
        if evidence.anchor_linkage_sha256 != request.anchor.linkage_sha256:
            reasons.append(
                f"the anchor linkage verified ({evidence.anchor_linkage_sha256[:12]}) is not "
                f"the anchor the record names ({request.anchor.linkage_sha256[:12]})")
        identical = evidence.binary_sha256 == evidence.anchor_binary_sha256
        if control_role == "aa" and not identical:
            reasons.append(
                "this is the A/A control and the candidate binary is NOT the anchor binary; "
                "A/A is the anchor measured against itself through the full candidate "
                "pipeline, and measuring two different binaries there calibrates nothing")
        if control_role != "aa" and identical:
            reasons.append(
                "the candidate binary is byte-identical to the anchor binary, so there is "
                "nothing to rank: a self-comparison presented as a candidate result would "
                "report instrument noise as a kernel effect")

    if not evidence.resolved_libraries:
        unknown.append("no resolved shared libraries were recorded, so linkage could not be "
                       "verified")
    else:
        root = evidence.expected_library_root
        stray = tuple(f"{soname} -> {path}" for soname, path, _ in evidence.resolved_libraries
                      if not (path == root or path.startswith(root + "/")))
        if stray:
            reasons.append(
                f"resolved libraries outside the expected root {root!r}: {list(stray)}. The "
                "three production trees run three different ggml generations, and a binary "
                "that inherits another tree's ggml runs SILENTLY wrong — which is precisely "
                "the failure a speed number cannot show")
    if reasons:
        return _gate(GID_LINKAGE, _fail(*reasons, *unknown), evidence_ref=evidence.receipt_ref)
    if unknown:
        return _gate(GID_LINKAGE, _cnc(*unknown), evidence_ref=evidence.receipt_ref)
    return _gate(GID_LINKAGE, schemas.Check(schemas.PASS), evidence_ref=evidence.receipt_ref,
                 notes=(f"verifier={evidence.verifier_id}",
                        f"libraries={len(evidence.resolved_libraries)}",
                        f"control_role={control_role}"))


def check_anti_reward_hacking(evidence: Optional[AntiRewardHackingEvidence],
                              control_role: Optional[str],
                              anchor: Optional[api.AnchorIdentity]) -> api.GateResult:
    """§8.6 anti-reward-hacking/integrity checks; control 3's detector.

    Control 3 is *"deliberately fast-looking but wrong: cheating, silently falling
    back, reducing work, or serving a cached result"*. Falling back is the
    dispatch gate's; the other three are here.

    `anchor` is a parameter — with no default — for one field:
    `delivered_units_anchor` is the floor the *"delivered work may not shrink"*
    comparison is made against. A floor from a different anchor raises
    `AntiRewardHackingAnchorMismatch`; a floor with no recorded anchor is
    COULD_NOT_CHECK, the same outcome an ABSENT floor already gets, because a
    count that cannot be attributed to the anchor bounds nothing. With
    `anchor=None` there is nothing for the capture to disagree with and the gate
    answers as it always did; an anchor-less T0 run is already structurally
    INVALID, and this gate is not the place that says so.
    """
    if evidence is None:
        return _gate(GID_ANTI_REWARD_HACKING, _no_evidence("anti-reward-hacking"))
    recorded = evidence.recorded_anchor()
    _refuse_replay_mismatch(
        anchor, recorded, label="anti-reward-hacking",
        consequence=("The delivered-work floor it carries was produced by another anchor, "
                     "and an exact comparison against another anchor's count can clear a "
                     "candidate that really did reduce work"),
        error=AntiRewardHackingAnchorMismatch)
    reasons = []
    unknown = []
    if evidence.correctness_verdict_source != "evaluator":
        reasons.append(_self_reported("correctness-verdict",
                                      evidence.correctness_verdict_source))
    if evidence.candidate_output_used_as_oracle:
        reasons.append("a candidate output was used as a correctness oracle; 'A candidate "
                       "output MUST NEVER be cached or reused as a correctness oracle'")
    if evidence.cache_state == "served_from_cache":
        reasons.append("cache_state='served_from_cache': the measured run served a cached "
                       "result, which is control 3's shape")
    elif evidence.cache_state == "unknown":
        unknown.append("cache_state='unknown': the cache state was not established, and 'Cache "
                       "state is declared in every record'")
    if not evidence.oracle_ids:
        reasons.append("no oracle was named; a correctness verdict is produced 'against "
                       "declared oracles', and an undeclared oracle is not one")
    if evidence.delivered_units_anchor is None:
        unknown.append("the anchor's delivered-work count was not recorded, so work reduction "
                       "could not be checked")
    else:
        if anchor is not None and recorded is None:
            unknown.append(
                "the anchor's delivered-work count records no anchor identity, so the floor "
                "it sets cannot be attributed to the anchor this run names; an unrecorded "
                "identity is not agreement with the bound anchor")
        if evidence.delivered_units_candidate < evidence.delivered_units_anchor:
            reasons.append(
                f"the candidate delivered {evidence.delivered_units_candidate} "
                f"{evidence.delivered_unit_name} against the anchor's "
                f"{evidence.delivered_units_anchor}; delivered work may not shrink. Internal "
                "work may legitimately fall — that is what an optimization IS — but a "
                "candidate that produces less output for the same request is control 3's "
                "'reducing work'")
    if evidence.environment_probe_findings:
        reasons.append(f"the candidate probed its environment: "
                       f"{list(evidence.environment_probe_findings)}; a kernel that can tell "
                       "it is under test can behave differently when it is")
    if evidence.timing_dependent_branch_findings:
        reasons.append(f"timing-dependent branch(es) found: "
                       f"{list(evidence.timing_dependent_branch_findings)}")
    if evidence.stream_creation_findings:
        reasons.append(
            "candidate-added accelerator stream creation found: "
            f"{list(evidence.stream_creation_findings)}; the timed harness synchronizes "
            "only its declared stream, so work on another stream can escape the bracket")
    if evidence.async_escape_findings:
        reasons.append(
            "candidate-added thread/async creation found: "
            f"{list(evidence.async_escape_findings)}; work that outlives the timed bracket "
            "is not admissible CPU evidence")
    if evidence.instrument_frame_findings:
        reasons.append(
            "candidate edit touched the protected measurement frame: "
            f"{list(evidence.instrument_frame_findings)}")
    if evidence.pointer_memoization_findings:
        reasons.append(
            "candidate-added pointer-keyed memoization found: "
            f"{list(evidence.pointer_memoization_findings)}; address identity cannot "
            "stand in for delivered work")
    if evidence.structured_short_circuit_findings:
        reasons.append(
            "candidate-added structured-input/known-shape short circuit found: "
            f"{list(evidence.structured_short_circuit_findings)}")
    if evidence.environment_probe_detector_id is None:
        unknown.append(
            "the environment-probe detector did not run; empty findings are not PASS")
    if evidence.timing_dependent_branch_detector_id is None:
        unknown.append(
            "the timing-dependent-branch detector did not run; empty findings are not PASS")
    if evidence.stream_creation_detector_id is None:
        unknown.append(
            "the stream-creation detector did not run; empty findings are not PASS")
    if evidence.async_escape_detector_id is None:
        unknown.append(
            "the thread/async-escape detector did not run; empty findings are not PASS")
    if evidence.instrument_frame_detector_id is None:
        unknown.append(
            "the instrument-frame detector did not run; empty findings are not PASS")
    if evidence.pointer_memoization_detector_id is None:
        unknown.append(
            "the pointer-memoization detector did not run; empty findings are not PASS")
    if evidence.structured_short_circuit_detector_id is None:
        unknown.append(
            "the structured-short-circuit detector did not run; empty findings are not PASS")
    notes = (f"cache_state={evidence.cache_state}", f"control_role={control_role}",
             f"oracles={list(evidence.oracle_ids)}",
             f"environment_detector={evidence.environment_probe_detector_id or 'not_run'}",
             f"timing_detector={evidence.timing_dependent_branch_detector_id or 'not_run'}",
             f"stream_detector={evidence.stream_creation_detector_id or 'not_run'}",
             f"async_detector={evidence.async_escape_detector_id or 'not_run'}",
             f"frame_detector={evidence.instrument_frame_detector_id or 'not_run'}",
             f"pointer_detector={evidence.pointer_memoization_detector_id or 'not_run'}",
             f"short_circuit_detector="
             f"{evidence.structured_short_circuit_detector_id or 'not_run'}",
             f"capture_anchor={'unrecorded' if recorded is None else recorded.short()}")
    if reasons:
        # A FAIL is never downgraded by an unrelated COULD_NOT_CHECK: both are
        # reported, and the worse outcome governs.
        return _gate(GID_ANTI_REWARD_HACKING, _fail(*reasons, *unknown),
                     evidence_ref=evidence.receipt_ref, notes=notes)
    if unknown:
        return _gate(GID_ANTI_REWARD_HACKING, _cnc(*unknown),
                     evidence_ref=evidence.receipt_ref, notes=notes)
    return _gate(GID_ANTI_REWARD_HACKING, schemas.Check(schemas.PASS),
                 evidence_ref=evidence.receipt_ref, notes=notes)


# =============================================================================
# Aggregation
# =============================================================================

@dataclass(frozen=True)
class T0Report:
    """Everything one T0 evaluation produced. Its coverage is checked, not assumed.

    `__post_init__` raises `GateCoverageGap` unless there is exactly one result
    for every id in `T0_GATE_IDS`. That is the structural answer to a checklist
    that silently loses a line: a report cannot exist while a surface is missing,
    so a deleted gate is a raise at construction rather than a shorter list that
    still reads PASS.
    """

    event_id: str
    candidate_id: str
    tier: str
    gates: tuple
    coherence: CoherenceVerdict
    requires_human_code_review: bool
    human_review_reasons: tuple
    release_relevant_properties: tuple
    actor_prediction_score: tuple
    anchor_bound: bool
    demoted_gates: tuple
    policy_ref: str

    def __post_init__(self) -> None:
        # Type FIRST: reading `.gate_id` off an arbitrary object raised
        # AttributeError from inside a coverage check, which is not the refusal
        # this class is supposed to make.
        for gate in self.gates:
            if not isinstance(gate, api.GateResult):
                raise TypeError("T0Report.gates must contain api.GateResult instances")
        ids = tuple(g.gate_id for g in self.gates)
        if len(set(ids)) != len(ids):
            duplicates = sorted({gid for gid in ids if ids.count(gid) > 1})
            raise GateCoverageGap(f"T0 report contains duplicate gate ids: {duplicates}")
        missing = tuple(gid for gid in T0_GATE_IDS if gid not in ids)
        unknown = tuple(gid for gid in ids if gid not in T0_GATE_IDS)
        if missing or unknown:
            raise GateCoverageGap(
                f"T0 report does not cover the declared gate set. missing={list(missing)} "
                f"unknown={list(unknown)}. §8.6 enumerates the surfaces T0 runs on every "
                "source candidate; a report that omits one is a coverage gap, and denial 6 "
                "says a coverage gap is RECORDED and blocks release eligibility, never "
                "quietly tolerated."
            )
        # Coverage is not only "is the id present". A gate carrying a T0 id but
        # filed under a class `api.SPEED_BLOCKING_GATE_CLASSES` does not contain
        # would be a full seventeen-line report in which one failing surface does
        # not block ranking — the import-time assertion pins T0_GATE_SPEC, but
        # nothing pinned the results, and a T0Report is public.
        for gate in self.gates:
            expected_class = _GATE_CLASS_BY_ID[gate.gate_id]
            if gate.gate_class != expected_class:
                raise GateCoverageGap(
                    f"gate {gate.gate_id!r} is filed under {gate.gate_class!r}, but T0's "
                    f"registry files it under {expected_class!r}. Correctness is "
                    "lexicographically prior to speed; a T0 gate in a class that does not "
                    "block ranking is a failing candidate with a speed rank.")
            expected_anchor = _GATE_ANCHOR_BY_ID[gate.gate_id]
            if gate.requires_anchor != expected_anchor:
                raise GateCoverageGap(
                    f"gate {gate.gate_id!r} declares requires_anchor="
                    f"{gate.requires_anchor} but the registry declares "
                    f"{expected_anchor}. `requires_anchor` is what demotes an anchor-less "
                    "PASS (precondition 4); clearing it removes the demotion.")
        if not isinstance(self.coherence, CoherenceVerdict):
            raise TypeError("T0Report.coherence must be a CoherenceVerdict")
        if self.requires_human_code_review != bool(self.human_review_reasons):
            raise ValueError(
                "T0Report.requires_human_code_review must follow from its reasons; a marker "
                "with no reason cannot be acted on and a reason with no marker is lost")

    def gate(self, gate_id: str) -> api.GateResult:
        for result in self.gates:
            if result.gate_id == gate_id:
                return result
        raise KeyError(gate_id)

    def outcome(self, gate_id: str) -> str:
        return self.gate(gate_id).check.outcome

    @property
    def failed(self) -> tuple:
        return tuple(g.gate_id for g in self.gates if g.check.outcome == schemas.FAIL)

    @property
    def unevaluated(self) -> tuple:
        return tuple(g.gate_id for g in self.gates
                     if g.check.outcome == schemas.COULD_NOT_CHECK)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "candidate_id": self.candidate_id,
            "tier": self.tier,
            "gates": [g.to_dict() for g in self.gates],
            "coherence": self.coherence.to_dict(),
            "requires_human_code_review": self.requires_human_code_review,
            "human_review_reasons": list(self.human_review_reasons),
            "release_relevant_properties": list(self.release_relevant_properties),
            # Deep-converted: `schemas.canonical_json` refuses a tuple outright, so a
            # nested tuple here would make the whole report un-hashable and therefore
            # un-journalable. Discovered by the test that content-hashes the report.
            "actor_prediction_score": [
                [list(cell) if isinstance(cell, tuple) else cell for cell in row]
                for row in self.actor_prediction_score
            ],
            "anchor_bound": self.anchor_bound,
            "demoted_gates": list(self.demoted_gates),
            "policy_ref": self.policy_ref,
            "failed": list(self.failed),
            "unevaluated": list(self.unevaluated),
        }


def demote_anchor_requiring_passes(gates: Sequence[api.GateResult], *,
                                   anchor_bound: bool) -> tuple:
    """Demote every anchor-requiring PASS to COULD_NOT_CHECK when no anchor is bound.

    Returns `(gates, demoted_gate_ids)` — the demotion is REPORTED, never silent.

    Each individual check above already refuses to return PASS without an anchor,
    so on the current code path this finds nothing. That is the point: it is the
    second of two independent enforcements of precondition 4, and it is a
    module-level function rather than a few lines inside `evaluate_t0` so it can
    be tested against a fabricated anchor-less PASS — the shape a future check
    that forgets its `request.anchor is None` guard would produce.

    `api.compute_verdict` applies the same demotion a third time, downstream.
    Three, for the one defect that contaminated `kernel_store.py`'s correct-only
    Pareto view.
    """
    if not isinstance(anchor_bound, bool):
        raise TypeError("anchor_bound must be a bool")
    if anchor_bound:
        return tuple(gates), ()
    resolved, demoted = [], []
    for gate in gates:
        if not isinstance(gate, api.GateResult):
            raise TypeError(f"gates must be api.GateResult instances, got "
                            f"{type(gate).__name__}")
        if gate.requires_anchor and gate.check.outcome == schemas.PASS:
            demoted.append(gate.gate_id)
            resolved.append(api.GateResult(
                gate_id=gate.gate_id,
                gate_class=gate.gate_class,
                check=_cnc(
                    "this gate compares against the anchor and no anchor is bound; a "
                    "coherence or identity label produced without a named anchor "
                    "comparison is not a verdict (P-AK-SEARCH-1 precondition 4)",
                    *gate.check.reasons),
                requires_anchor=True,
                evidence_ref=gate.evidence_ref,
                notes=gate.notes + ("PASS demoted to COULD_NOT_CHECK at T0: no anchor bound",),
            ))
        else:
            resolved.append(gate)
    return tuple(resolved), tuple(demoted)


def evaluate_t0(request: api.EvaluationRequest,
                evidence: T0Evidence,
                policy: T0Policy) -> T0Report:
    """Run every T0 surface and return the report. Launches nothing.

    Order follows §8.6: the §8.5.1 source-integrity gates *"run before any
    behavioural check"*, then the behavioural surfaces. The order is presentation
    only — every gate is evaluated, because a report that stopped at the first
    failure would leave the other sixteen surfaces unknown while looking complete.

    Belt-and-braces on precondition 4: after the gates are built, any
    anchor-requiring gate that came back PASS with no anchor bound is demoted to
    `COULD_NOT_CHECK` here as well as in `api.compute_verdict`. Two independent
    demotions, because this is the single defect that contaminated the store.
    """
    if not isinstance(request, api.EvaluationRequest):
        raise TypeError("request must be an api.EvaluationRequest")
    if not isinstance(evidence, T0Evidence):
        raise TypeError("evidence must be a T0Evidence")
    if not isinstance(policy, T0Policy):
        raise TypeError("policy must be a T0Policy")

    surface = evidence.change_surface
    gates = [
        check_symbol_and_registration_preservation(request, evidence.symbols, policy),
        check_clean_build_from_snapshot(request, evidence.build),
        check_semantic_diff_conformance(evidence.diff),
    ]
    schema_gate, review_reasons = check_schema_and_diff_policy(evidence.diff, surface, policy)
    gates.append(schema_gate)
    gates.append(check_static_and_compile(evidence.static_analysis, request.anchor))
    gates.append(check_asan(request, evidence.sanitizers, surface))
    gates.append(check_ubsan(request, evidence.sanitizers, surface))
    gates.append(check_backend_op_units(request, evidence.op_suite, surface, policy))
    gates.append(check_exact_reference_comparison(request, evidence.reference, surface, policy))
    gates.append(check_unseen_boundary_shapes(evidence.boundary_shapes, surface))
    gates.append(check_affected_surface_reconciliation(evidence.dispatch_trace, surface))
    gates.append(check_no_fallback_dispatch_proof(evidence.dispatch_trace))
    gates.append(check_state_rollback_teardown_race(evidence.state_safety, surface))
    coherence_gate, coherence = check_output_coherence(request, evidence.coherence, policy,
                                                       evidence.determinism)
    gates.append(coherence_gate)
    determinism_gate, properties = check_determinism_class(request, evidence.determinism, policy)
    gates.append(determinism_gate)
    gates.append(check_binary_and_linkage_identity(request, evidence.linkage,
                                                   evidence.control_role))
    gates.append(check_anti_reward_hacking(evidence.anti_reward_hacking, evidence.control_role,
                                           request.anchor))

    # A projection can discover a refusal that its projected record has no
    # field for (for example an incomplete ELF extraction or a binary diff).
    # Merge it into the existing constitutional gate instead of minting an
    # eighteenth gate or demoting it to an advisory note. This preserves report
    # coverage while ensuring FAIL/COULD_NOT_CHECK cannot disappear at a seam.
    if evidence.projection_checks or evidence.source_candidate:
        grouped = {}
        for gate_id, check_name, check in evidence.projection_checks:
            grouped.setdefault(gate_id, []).append((check_name, check, None))
        if evidence.source_candidate:
            prerequisite_gate = {
                "input_sensitivity": GID_OP_UNITS,
                "hostile_distributions": GID_OP_UNITS,
                "checker_isolation": GID_EXACT_REFERENCE,
            }
            supplied = {item.prerequisite_id: item
                        for item in evidence.source_prerequisites}
            for prerequisite_id in SOURCE_PREREQUISITE_IDS:
                item = supplied.get(prerequisite_id)
                measurement = None
                if item is None:
                    check = schemas.Check(
                        schemas.COULD_NOT_CHECK,
                        (f"source candidate has no hash-bound {prerequisite_id} evidence",))
                else:
                    measurement = item.measurement()
                    reasons = []
                    if item.candidate_source_sha256 != request.artifact.source_sha256:
                        reasons.append(
                            "prerequisite names a different candidate source SHA-256")
                    if item.evaluator_bundle_sha256 != request.evaluator.bundle_sha256:
                        reasons.append(
                            "prerequisite names a different evaluator bundle SHA-256")
                    if item.capture_mode != "measured":
                        reasons.append("dry-run prerequisite is not correctness evidence")
                    check = (schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))
                             if reasons else item.check)
                grouped.setdefault(prerequisite_gate[prerequisite_id], []).append(
                    (f"source_prerequisite.{prerequisite_id}", check, measurement))
        merged = []
        for gate in gates:
            extras = grouped.get(gate.gate_id, ())
            if not extras:
                merged.append(gate)
                continue
            labelled = [gate.check]
            for check_name, check, _measurement in extras:
                reasons = check.reasons or (
                    f"projection check {check_name!r} returned {check.outcome}",)
                labelled.append(schemas.Check(
                    check.outcome,
                    tuple(f"projection {check_name}: {reason}" for reason in reasons)))
            merged.append(api.GateResult(
                gate_id=gate.gate_id,
                gate_class=gate.gate_class,
                check=schemas.Check.worst_of(labelled),
                requires_anchor=gate.requires_anchor,
                evidence_ref=gate.evidence_ref,
                notes=gate.notes + tuple(
                    f"projection_check={name}:{check.outcome}"
                    for name, check, _measurement in extras),
                measurements=gate.measurements + tuple(
                    measurement for _name, _check, measurement in extras
                    if measurement is not None),
            ))
        gates = merged

    gates, demoted = demote_anchor_requiring_passes(
        tuple(gates), anchor_bound=request.anchor is not None)

    return T0Report(
        event_id=request.event_id,
        candidate_id=request.candidate_id,
        tier=request.tier,
        gates=tuple(gates),
        coherence=coherence,
        requires_human_code_review=bool(review_reasons),
        human_review_reasons=tuple(review_reasons),
        release_relevant_properties=tuple(properties),
        actor_prediction_score=surface.prediction_score(),
        anchor_bound=request.anchor is not None,
        demoted_gates=tuple(demoted),
        policy_ref=policy.policy_ref,
    )


# =============================================================================
# Seams — the evidence provider and the `api.TierGateRunner` implementation
# =============================================================================

class T0EvidenceProvider(Protocol):
    """Supplies the T0 evidence for a candidate.

    The implementation that really compiles, really runs `test-backend-ops`, and
    really traces dispatch lives behind this Protocol, under a held claim. Nothing
    in this module is that implementation.
    """

    def evidence_for(self, request: api.EvaluationRequest) -> T0Evidence:
        ...


class StaticEvidenceProvider:
    """A provider over already-collected evidence, keyed by candidate id.

    Used for deterministic replay (invariant 11: *"Saved outputs, profiles, and
    raw samples are rescored without inference when the generation path remains
    valid"*) and by the tests. It RAISES on an unknown candidate: it does not
    synthesize, default, or return a blank `T0Evidence`.
    """

    def __init__(self, evidence_by_candidate: Mapping[str, T0Evidence]) -> None:
        if not isinstance(evidence_by_candidate, Mapping):
            raise TypeError("evidence_by_candidate must be a mapping of candidate_id -> "
                            "T0Evidence")
        for candidate_id, item in evidence_by_candidate.items():
            _req_str(candidate_id, "evidence_by_candidate key")
            if not isinstance(item, T0Evidence):
                raise TypeError(f"evidence for {candidate_id!r} is a "
                                f"{type(item).__name__}, not a T0Evidence")
        self._evidence = dict(evidence_by_candidate)

    def evidence_for(self, request: api.EvaluationRequest) -> T0Evidence:
        try:
            return self._evidence[request.candidate_id]
        except KeyError:
            raise T0EvidenceUnavailable(
                f"no T0 evidence for candidate {request.candidate_id!r}; known candidates are "
                f"{sorted(self._evidence)}. A report claims seventeen surfaces were examined "
                "and must not be synthesisable from nothing."
            ) from None


class T0CorrectnessRunner:
    """The `api.TierGateRunner` for T0. Aggregates evidence; launches nothing.

    `run_gates()` satisfies the Protocol the dispatcher calls. `evaluate()` is the
    richer entry point and is what a controller should call, because the
    dispatcher's seam carries only gate results — see `SEAMS` item 1.
    """

    tier = "T0"

    def __init__(self, *, provider: Any, policy: T0Policy) -> None:
        if not hasattr(provider, "evidence_for"):
            raise TypeError("provider must implement evidence_for(request) -> T0Evidence")
        if not isinstance(policy, T0Policy):
            raise TypeError("policy must be a T0Policy")
        self._provider = provider
        self._policy = policy

    @property
    def policy(self) -> T0Policy:
        return self._policy

    def evaluate(self, request: api.EvaluationRequest) -> T0Report:
        api.admit_tier(request.tier)
        if request.tier != self.tier:
            raise api.EvaluatorNotWired(
                f"T0CorrectnessRunner was handed a {request.tier} request. T0's gates are the "
                "correctness surfaces that run BEFORE any speed work; running them under a "
                "T1/T2 label would file a correctness verdict as a search measurement."
            )
        evidence = self._provider.evidence_for(request)
        if not isinstance(evidence, T0Evidence):
            raise TypeError(
                f"provider returned {type(evidence).__name__}, expected T0Evidence")
        return evaluate_t0(request, evidence, self._policy)

    def run_gates(self, request: api.EvaluationRequest) -> Sequence[api.GateResult]:
        return self.evaluate(request).gates


# =============================================================================
# Self-audit and recorded seams
# =============================================================================

def audit_no_write_or_process_paths() -> schemas.Check:
    """Prove from this module's own AST that it cannot write, spawn, or signal.

    Delegates to `api.audit_no_write_or_process_paths`, which is the same check
    `api.py` applies to itself, so the two modules cannot drift on what "no write
    path" means. COULD_NOT_CHECK when the source cannot be read — an unreadable
    module is not an audited one.
    """
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except OSError as exc:
        return _cnc(f"could not read {__file__}: {exc}")
    return api.audit_no_write_or_process_paths(source, module_id=MODULE_ID)


#: Seams that need a real artifact, implemented here against fakes, recorded so
#: the gap is a known one. Denial 6: *"A controller that discovers a coverage gap
#: in its evaluator RECORDS the gap, blocks release eligibility for the affected
#: lineage, continues unrelated research, and MAY draft an amendment for human
#: review. It does not patch the instrument."*
SEAMS = (
    ("DUPLICATION, recorded not resolved. Two sibling AK3 modules landed in this package "
     "while this one was being written and cover ground §8.6 also assigns to T0: "
     "`integrity.py` implements §8.5.1 against real artifacts (`check_symbol_preservation`, "
     "`check_clean_build_from_snapshot`, `check_semantic_diff_conformance`, "
     "`ComplexityCeiling`, `ChangeClassEnvelope`, `REQUIRES_HUMAN_CODE_REVIEW`, "
     "`SourceIntegrityGateRunner`), and `surface.py` implements §6.4 "
     "(`derive_affected_surface`, `reconcile_surface`, `score_actor_declaration`). This "
     "module's equivalents — GID_SYMBOLS, GID_CLEAN_BUILD, GID_SEMANTIC_DIFF, "
     "GID_SURFACE_RECONCILIATION, ChangeSurface.prediction_score — are evidence-level and "
     "fixture-driven; theirs parse ELF, depfiles and diffs. The correct end state is that "
     "those five delegate, with `integrity`/`surface` producing the evidence types here. "
     "That rewiring is NOT done unilaterally: at the time of writing both sibling suites are "
     "red, so binding to them would bind to a moving target, and choosing which module owns a "
     "gate id is an integration decision for whoever composes the bundle. The overlap is "
     "recorded here so it is discovered by reading, not by two gate ids colliding in a "
     "record."),
    ("api.TierGateRunner.run_gates returns only Sequence[GateResult], so a T0 report's "
     "REQUIRES_HUMAN_CODE_REVIEW marker, its release-relevant determinism properties, and its "
     "actor-prediction score cannot reach the record through api.TierDispatcher.dispatch(). "
     "MITIGATED, not closed: those three ride out on GateResult.notes, which api's to_dict() "
     "carries into the durable payload. Closing it properly means a structured field on the "
     "runner seam, which is a change in api.py — another agent's module in the same bundle."),
    ("The evidence types are the fixture seam. Nothing here compiles, links, runs "
     "test-backend-ops, traces dispatch, or executes the ASAN/UBSAN invocation it builds. A "
     "real T0EvidenceProvider does all of that under a held claim; the tests fill the same "
     "types from fixtures. The verdict logic is therefore fully exercised without a build, and "
     "the collection logic is entirely unexercised until AK3's runner lands."),
    ("build_sanitizer_invocation() emits cmake argv shaped for the llama.cpp trees. Its flag "
     "names are checked against _SANITIZER_COMPILE_FLAGS, not against a real CMakeLists, so a "
     "cmake option rename upstream would produce a well-formed invocation that configures "
     "nothing. Detecting that needs a real configure run, which is a build."),
    ("Delivered-work comparison uses whatever unit the provider names in "
     "AntiRewardHackingEvidence.delivered_unit_name. This module cannot verify that the unit "
     "is genuinely delivered work rather than internal work relabelled; a provider that "
     "reports FLOPs under that field would make a legitimate optimization look like control 3, "
     "or vice versa. The unit vocabulary belongs in the backend adapter contract (§13)."),
    ("Symbol-table shrinkage uses exported-symbol COUNTS. A candidate that removes N symbols "
     "and adds N unrelated ones has a shrinkage of zero; the undeclared-removal check is what "
     "actually catches that, and the ratio is only the blunt second line. A name-level diff is "
     "already carried in removed_symbols, so the data is present if a future policy wants a "
     "stricter rule."),
)
MODULE_ID = "autokernel.evaluator.correctness/v1"
