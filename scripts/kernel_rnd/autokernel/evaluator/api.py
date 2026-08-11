#!/usr/bin/env python3
"""api.py — the typed AK3 evaluator interface, and the place a verdict is COMPUTED.

WHY THIS MODULE EXISTS
----------------------
It replaces `scripts/kernel_rnd/kernel_eval.sh`, which is fenced off and exits 2.
That script is the live defect this module is shaped around, and its three
failures map one-to-one onto three structural properties here:

  * **It stamped a literal.** The shell emitted `"status":"OK"` unconditionally at
    the end of the happy path — the status was a string in a `printf`, not a
    function of anything it had measured. Here a `Verdict` cannot be constructed
    at all except by `compute_verdict()`, and even then `Verdict.__post_init__`
    RE-DERIVES the status from the gate results it carries and raises
    `VerdictTampering` if the stored status differs. There is no code path,
    private or public, that can attach a status to a record without the
    underlying checks. This is the single most important property in the file.

  * **It reported coherence with no anchor.** `COH="coherent"` was set for ANY
    non-empty generation, and the baseline comparison ran only when
    `--baseline-env` happened to be passed. `kernel_store.py:81` then admitted
    `coherence in ("byte-identical","coherent")` into its CORRECT-ONLY Pareto
    view, so every anchor-less run entered the frontier as if verified. Here
    `P-AK-SEARCH-1` precondition 4 is structural: an absent, incomplete, or
    mutated anchor is a `VoidFinding`, the verdict is `INVALID`, and any gate
    that declared `requires_anchor=True` has its PASS demoted to
    `COULD_NOT_CHECK` before the status is derived — *"Absence of a comparison is
    not evidence of equivalence."*

  * **It let speed be reported beside a correctness problem.** The shell's
    `emit_fail` exited early, which is the right shape, but nothing structurally
    prevented a later edit from ranking a failing candidate. Here
    `Verdict.rank_key()` RAISES `SpeedRankUnavailable` unless the status is
    `pass`, which requires every lexicographically-prior gate to have returned
    PASS. A failing candidate gets no rank at all — not a penalised one
    (`kernel-research.md`, "Correctness precedence").

WHICH PROTOCOL CLAUSES THIS FILE IMPLEMENTS
-------------------------------------------
`measurement/protocols/kernel-research.md` (Annex K, P-AK-SEARCH-1, RATIFIED
2026-08-03), by section name:

  * "Preconditions (all enforced or attested per run)" -> `PreconditionScan`,
    `check_preconditions()`, `PRECONDITION_IDS`.
  * "Campaign calibration block — every threshold is derived, none is supplied"
    -> `CalibrationOutputs` (whose `__post_init__` refuses a supplied literal
    that violates the derived relations, and refuses a block that does not
    record the normative solve order), `CampaignControls` (precondition 8).
  * "Statistical requirements" -> `EffectEstimate` (e-value + threshold + MDE
    published in the same object as the estimate; LCB carried only as a labelled
    `descriptive` field), `_resolve_effect()` (`|effect| < MDE` is *"a result and
    a decision, not a failed experiment"*).
  * "Controls — four mandatory, plus one accept-side control" -> `ControlPanel`,
    whose `__post_init__` refuses an unavailable control 5 that does not name
    its reason AND its operator escalation, so it is never a silent skip.
  * "Correctness precedence" -> `LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES`,
    `Verdict.rank_key()`, `rank_candidates()`.
  * "Search-grade requires ALL of" -> `SEARCH_GRADE_CONJUNCTS`,
    `evaluate_search_grade()`, `SearchGradeResult.failed`.
  * "Record grammar" -> `render_search_record_grammar()`,
    `check_record_grammar_complete()`, `build_evaluation_event()`.
  * "What voids a run" -> `VOID_REASONS`, `check_void_conditions()`,
    `VoidFinding`; a voided run is `INVALID` **with its reason** and is returned
    for journaling, never discarded.

Design context: `epyc-root/handoffs/active/autokernel-research-loop.md` §5.4
(trusted runner: "has no authority to modify candidate source or production
state"), §8.6 (T0_GATE), §9 (tier table), §9.2 (statistical machinery), §15.2
(five controls), phase AK3.

WHAT THIS MODULE IS NOT
-----------------------
It runs NO inference, NO benchmark, and NO build. It starts, stops, and signals
NO process. It writes NO file. Those are not promises in prose:
`audit_no_write_or_process_paths()` parses this module's own AST and FAILs if a
write-capable call, a process call, or an import of `os`/`subprocess`/`shutil`/
`signal` appears in it, and `test_api.py` asserts the audit PASSes. The tier
runners that DO launch work plug in through the `TierGateRunner` seam and hand
back `GateResult`s; this module only aggregates them.

T3 (kernel-freeze gate) and T4 (post-cutover watch) are NOT owned here. They are
release instruments governed by the release protocols, explicitly outside
P-AK-SEARCH-1's scope ("It does NOT apply to T3 or any release gate"). The seam
is `ReleaseTierEvaluator`; `admit_tier()` refuses them with `TierNotOwned`.
"""
from __future__ import annotations

import ast
import math
import re
from collections import namedtuple
from dataclasses import InitVar, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from .. import schemas
from . import devices

__all__ = [
    # identity
    "PROTOCOL_ID", "PROTOCOL_VERSIONED_ID", "PROTOCOL_RATIFIED_UTC", "RECORD_CLASS",
    # errors
    "EvaluatorError", "TierNotOwned", "EvaluatorNotWired", "StateMachineViolation",
    "VerdictTampering", "SpeedRankUnavailable", "AnchorMissing",
    # vocabularies
    "SEARCH_TIERS", "RELEASE_TIERS", "RELEASE_TIER_OWNER", "DISPATCH_STATES",
    "GATE_CLASSES", "LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES", "SPEED_BLOCKING_GATE_CLASSES",
    "STATUS_PASS", "STATUS_FAIL", "STATUS_INCONCLUSIVE", "STATUS_INVALID", "VERDICT_STATUSES",
    "VOID_REASONS", "VOID_REASON_PHRASES", "PRECONDITION_IDS", "SEARCH_GRADE_CONJUNCTS",
    "CALIBRATION_SOLVE_ORDER", "EFFECT_RESOLUTIONS", "SUB_FLOOR_RESOLUTIONS",
    "is_rankable_resolution", "is_sub_floor_resolution",
    "E_PROCESS_CONSTRUCTION_IDS",
    "STRATA", "STRATUM_SELECTION", "STRATUM_CONFIRMATION",
    # typed inputs
    "AnchorIdentity", "ArtifactIdentity", "EvaluatorIdentity", "ScopeDenominator",
    "DeterminismReport", "RecipeReceipt", "CampaignControls", "CalibrationOutputs",
    "ControlPanel", "EffectEstimate", "WindowAttestations", "EvaluationRequest",
    # typed outputs
    "GateResult", "VoidFinding", "VoidScan", "PreconditionScan", "SearchGradeResult",
    "Verdict", "EvaluationOutcome",
    # seams
    "TierGateRunner", "EffectReducer", "RecipeConstructor", "ReleaseTierEvaluator",
    # functions
    "admit_tier", "check_preconditions", "check_void_conditions", "evaluate_search_grade",
    "compute_verdict", "TierDispatcher", "build_evaluation_event",
    "render_search_record_grammar", "check_record_grammar_complete",
    "compose_attestation_ref", "rank_candidates", "audit_no_write_or_process_paths",
    "MODULE_ID",
]

# =============================================================================
# Protocol identity
# =============================================================================

PROTOCOL_ID = "P-AK-SEARCH-1"
PROTOCOL_VERSIONED_ID = "P-AK-SEARCH-1/v1"
PROTOCOL_RATIFIED_UTC = "20260803T083005Z"

#: This module's own identity, in the source, so `audit_no_write_or_process_paths()`
#: can PROVE the text it parsed is this file rather than assume it. Same spelling and
#: same purpose as `release/packager.MODULE_ID`.
MODULE_ID = "autokernel.evaluator.api/v1"

#: Annex K requires every protocol to state the class of record it emits.
#: P-AK-SEARCH-1 emits a verdict that is NOT a claim, and the grammar says so.
RECORD_CLASS = "SEARCH RECORD, NOT A CLAIM"

#: Not mirrored any more — BOUND. A local `re.compile(r"^[0-9a-f]{64}$")` is the
#: first line of a re-derived digest validator and it is free to write, which is
#: why nine modules wrote one and two of them then forgot the placeholder check.
#: The shape is `schemas`' to define.
_SHA256_RE = schemas.SHA256_RE
_COMMIT_RE = schemas.COMMIT_RE
_CO_RESIDENCY_RE = re.compile(r"^(single|co_resident:[A-Za-z0-9._:-]+)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# =============================================================================
# Errors — every one of these is a refusal, never a degraded result
# =============================================================================

class EvaluatorError(Exception):
    """Base class for every refusal this module makes."""


class TierNotOwned(EvaluatorError):
    """A release tier (T3/T4) was handed to the search evaluator.

    P-AK-SEARCH-1 "Scope": *"It does NOT apply to T3 or any release gate, which
    are governed by the release protocols."* Silently treating T3 as T2 would
    produce a release-shaped record under a search protocol, which is exactly the
    retro-certification route denial 3 forbids.
    """


class EvaluatorNotWired(EvaluatorError):
    """No gate runner is registered for a tier this evaluator does own.

    There is deliberately no default runner: a default would report an unrun
    tier as having produced no failures, which is a fail-open PASS.
    """


class StateMachineViolation(EvaluatorError):
    """A dispatch step was attempted out of order."""


class VerdictTampering(EvaluatorError):
    """A `Verdict` was constructed outside the aggregator, or carries a status
    that does not follow from its own gate results."""


class SpeedRankUnavailable(EvaluatorError):
    """A speed rank was requested for a candidate that has not earned one.

    "Correctness precedence": *"A candidate failing any of them receives no speed
    rank at all — not a penalised one."* This is raised rather than returning a
    sentinel so a ranking loop cannot treat "no rank" as a very low rank.
    """


class AnchorMissing(EvaluatorError):
    """An anchor was CLAIMED for this run and cannot be recorded as claimed.

    Precondition 4: *"A run without an explicit anchor is INVALID."* That case is
    no longer this exception's job — under `evaluation_event.v3` an anchor-less
    VOIDED run emits a valid record whose `anchor` block is structurally absent,
    which is what *"A voided run is journaled as INVALID with its reason, and is
    never silently discarded"* requires. This is now raised only when the record
    would have to LIE: an anchor object that is not an `AnchorIdentity`, one whose
    fields do not parse, one carrying a placeholder digest, or a run with no
    anchor whose verdict does not actually declare an anchor void — in which case
    omitting the block would hide a missing denominator behind a status that says
    nothing about anchors. Fabricating a digest was never the fix and is not one
    now.
    """


# =============================================================================
# Tiers — what this evaluator owns, and the seam for what it does not
# =============================================================================

#: Tiers governed by P-AK-SEARCH-1 ("Scope": tiers T0, T1 and T2).
SEARCH_TIERS = ("T0", "T1", "T1a", "T1b", "T1c", "T2")

#: Release instruments. Owned by AK5, governed by the release protocols.
RELEASE_TIERS = ("T3", "T4")
RELEASE_TIER_OWNER = "AK5"

DISPATCH_STATES = (
    "CREATED",
    "TIER_ADMITTED",
    "WINDOW_OPENED",
    "PRECONDITIONS_CHECKED",
    "ANCHOR_BOUND",
    "GATES_RUN",
    "WINDOW_CLOSED",
    "VERDICT_COMPUTED",
    "EMITTED",
    "REFUSED",
)

# Deliberately no "VOID" terminal state: a voided window walks the SAME path to a
# computed verdict of INVALID and an emitted record. "A voided run is journaled as
# INVALID with its reason, and is never silently discarded."
_TRANSITIONS: Mapping[str, tuple] = {
    "CREATED": ("TIER_ADMITTED", "REFUSED"),
    "TIER_ADMITTED": ("WINDOW_OPENED",),
    "WINDOW_OPENED": ("PRECONDITIONS_CHECKED",),
    "PRECONDITIONS_CHECKED": ("ANCHOR_BOUND",),
    "ANCHOR_BOUND": ("GATES_RUN",),
    "GATES_RUN": ("WINDOW_CLOSED",),
    "WINDOW_CLOSED": ("VERDICT_COMPUTED",),
    "VERDICT_COMPUTED": ("EMITTED",),
    "EMITTED": (),
    "REFUSED": (),
}


def admit_tier(tier: str) -> str:
    """Return `tier` if this evaluator owns it; raise otherwise.

    Implements P-AK-SEARCH-1 "Scope". T3/T4 are refused by name so the caller
    reads *why* rather than "unknown tier".
    """
    if not isinstance(tier, str):
        raise TypeError(f"tier must be a string, got {type(tier).__name__}")
    if tier in RELEASE_TIERS:
        raise TierNotOwned(
            f"tier {tier!r} is a release instrument and is outside P-AK-SEARCH-1's "
            f"scope ('It does NOT apply to T3 or any release gate'). It is owned by "
            f"{RELEASE_TIER_OWNER} and implements the ReleaseTierEvaluator seam; the "
            f"search evaluator refuses it rather than producing a release-shaped "
            f"record under a search protocol."
        )
    if tier not in SEARCH_TIERS:
        raise TierNotOwned(
            f"unknown tier {tier!r}; P-AK-SEARCH-1 governs {list(SEARCH_TIERS)} and "
            f"{RELEASE_TIER_OWNER} owns {list(RELEASE_TIERS)}"
        )
    return tier


# =============================================================================
# Gate classes and the lexicographic order
# =============================================================================

GATE_INTEGRITY = "integrity"
GATE_CORRECTNESS = "correctness"
GATE_QUALITY = "quality"
GATE_NUMERICAL_SAFETY = "numerical_safety"
GATE_STABILITY = "stability"
GATE_DETERMINISM = "determinism"
GATE_MECHANISM = "mechanism"
GATE_PERFORMANCE = "performance"

GATE_CLASSES = (
    GATE_INTEGRITY, GATE_CORRECTNESS, GATE_QUALITY, GATE_NUMERICAL_SAFETY,
    GATE_STABILITY, GATE_DETERMINISM, GATE_MECHANISM, GATE_PERFORMANCE,
)

#: Verbatim from "Correctness precedence": *"Correctness, quality, numerical
#: safety, integrity, and stability are lexicographically prior to speed."*
#: Exactly five, in the protocol's own enumeration — nothing added here.
LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES = (
    GATE_CORRECTNESS, GATE_QUALITY, GATE_NUMERICAL_SAFETY, GATE_INTEGRITY, GATE_STABILITY,
)

#: The protocol's five, plus `determinism`. Determinism is added on a DIFFERENT
#: authority, and the distinction is kept visible rather than folded in: design
#: §8.6 lists the determinism-class check among T0's gates and then says *"Any
#: failure ends speed ranking for that candidate"*, and invariant 12 makes the
#: determinism class an interface a candidate may not silently change.
SPEED_BLOCKING_GATE_CLASSES = LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES + (GATE_DETERMINISM,)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_INVALID = "invalid"
VERDICT_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_INCONCLUSIVE, STATUS_INVALID)

# The verdict vocabulary is a SUBSET of the record vocabulary, not a parallel one.
# `timeout`, `crash` and `rejected` are reported by the runner as gate results;
# this aggregator never invents them.
_UNKNOWN_STATUSES = [s for s in VERDICT_STATUSES if s not in schemas.EVENT_STATUSES]
if _UNKNOWN_STATUSES:  # pragma: no cover - import-time contract assertion
    raise ImportError(
        f"evaluator verdict statuses {_UNKNOWN_STATUSES} are not in "
        f"schemas.EVENT_STATUSES; the evaluator must not invent a record shape"
    )

_STATUS_SEVERITY = {
    STATUS_PASS: 0,
    STATUS_INCONCLUSIVE: 1,
    STATUS_FAIL: 2,
    STATUS_INVALID: 3,
}

# What a FAIL in each class escalates to.
#   * The speed-blocking classes escalate to FAIL.
#   * `mechanism` escalates to INCONCLUSIVE, per design §9.4: *"A failed mechanism
#     prediction withholds the mechanism bonus and normally makes the result
#     inconclusive until the mismatch is explained."*
#   * `performance` escalates to INCONCLUSIVE: "the experiment ran and did not
#     resolve" is distinct from "the experiment was not a measurement".
_ON_GATE_FAIL = {c: STATUS_FAIL for c in SPEED_BLOCKING_GATE_CLASSES}
_ON_GATE_FAIL[GATE_MECHANISM] = STATUS_INCONCLUSIVE
_ON_GATE_FAIL[GATE_PERFORMANCE] = STATUS_INCONCLUSIVE

# COULD_NOT_CHECK NEVER escalates to FAIL and NEVER passes. It is the third
# outcome, and the record says which of the two it was.
_ON_GATE_COULD_NOT_CHECK = {c: STATUS_INCONCLUSIVE for c in GATE_CLASSES}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_SEVERITY[a] >= _STATUS_SEVERITY[b] else b


# =============================================================================
# "What voids a run" — the protocol's enumeration, as a checked precondition set
# =============================================================================

VOID_CLAIM_NOT_HELD = "CLAIM_NOT_HELD"
VOID_HOST_HEALTH_TIER_VIOLATION = "HOST_HEALTH_TIER_VIOLATION"
VOID_ANCHOR_GATE_FAILED = "ANCHOR_GATE_FAILED"
VOID_AA_CONTROL_FAILED = "AA_CONTROL_FAILED"
VOID_EVALUATOR_BUNDLE_UNVERIFIED = "EVALUATOR_BUNDLE_UNVERIFIED"
VOID_ANCHOR_MISSING_OR_MUTATED = "ANCHOR_MISSING_OR_MUTATED"
VOID_HAND_TYPED_ARGV = "HAND_TYPED_ARGV"
VOID_CONCURRENT_INFERENCE = "CONCURRENT_INFERENCE_CONTAMINATION"
VOID_STORAGE_EXHAUSTED = "STORAGE_EXHAUSTED_MID_WINDOW"
VOID_STRATA_VIOLATION = "STRATA_VIOLATION"
VOID_POST_HOC_RULE_CHANGE = "POST_HOC_RULE_CHANGE"
VOID_INCOMPLETE_CALIBRATION = "INCOMPLETE_CALIBRATION_BLOCK"

VOID_REASONS = (
    VOID_CLAIM_NOT_HELD,
    VOID_HOST_HEALTH_TIER_VIOLATION,
    VOID_ANCHOR_GATE_FAILED,
    VOID_AA_CONTROL_FAILED,
    VOID_EVALUATOR_BUNDLE_UNVERIFIED,
    VOID_ANCHOR_MISSING_OR_MUTATED,
    VOID_HAND_TYPED_ARGV,
    VOID_CONCURRENT_INFERENCE,
    VOID_STORAGE_EXHAUSTED,
    VOID_STRATA_VIOLATION,
    VOID_POST_HOC_RULE_CHANGE,
    VOID_INCOMPLETE_CALIBRATION,
)

#: The protocol's own words, kept verbatim so the journaled reason is auditable
#: against the ratified text rather than against a paraphrase.
VOID_REASON_PHRASES = {
    VOID_CLAIM_NOT_HELD:
        "a resource claim not held, not re-verified, or held by a different holder "
        "at window close",
    VOID_HOST_HEALTH_TIER_VIOLATION: "a host-health tier violation",
    VOID_ANCHOR_GATE_FAILED: "a failed anchor gate",
    VOID_AA_CONTROL_FAILED: "a failed A/A control",
    VOID_EVALUATOR_BUNDLE_UNVERIFIED:
        "a missing, drifted, or unverifiable evaluator bundle hash or runtime "
        "source-label attestation",
    VOID_ANCHOR_MISSING_OR_MUTATED: "a missing or mutated anchor",
    VOID_HAND_TYPED_ARGV: "hand-typed argv",
    VOID_CONCURRENT_INFERENCE: "contamination by concurrent inference",
    VOID_STORAGE_EXHAUSTED: "storage exhaustion mid-window",
    VOID_STRATA_VIOLATION: "a strata violation",
    VOID_POST_HOC_RULE_CHANGE:
        "any post-hoc change to the stopping rule, the calibration outputs, the "
        "objective, or the control definitions",
    VOID_INCOMPLETE_CALIBRATION: "an incomplete calibration block",
}

# Void conditions whose subject is a rate cell. They are evaluated only when the
# record carries a rate comparison, and when they are NOT evaluated the scan says
# so explicitly (`VoidScan.not_applicable`) — never a silent skip.
_RATE_ONLY_VOIDS = frozenset({
    VOID_ANCHOR_GATE_FAILED,
    VOID_AA_CONTROL_FAILED,
    VOID_STRATA_VIOLATION,
    VOID_INCOMPLETE_CALIBRATION,
})


# =============================================================================
# Preconditions — the protocol's eight, by name
# =============================================================================

PRECONDITION_IDS = (
    "resource_claim_held_whole_window",
    "no_concurrent_inference",
    "host_health_tier",
    "explicit_immutable_anchor",
    "evaluator_identity",
    "codified_recipe",
    "storage_headroom",
    "declared_campaign_controls",
)

PRECONDITION_PHRASES = {
    "resource_claim_held_whole_window": "Resource claim held for the whole window",
    "no_concurrent_inference": "No concurrent inference",
    "host_health_tier": "Host-health tier satisfied",
    "explicit_immutable_anchor": "An EXPLICIT IMMUTABLE ANCHOR",
    "evaluator_identity": "Evaluator identity",
    "codified_recipe": "Codified recipe",
    "storage_headroom": "Storage headroom",
    "declared_campaign_controls": "Declared campaign controls",
}


# =============================================================================
# "Search-grade requires ALL of" — the conjunction, as an explicit predicate
# =============================================================================

_Conjunct = namedtuple("_Conjunct", "id phrase rate_only")

#: The protocol's list, in its own order, split on its own semicolons.
#: `rate_only` conjuncts are evaluated iff the record carries a rate comparison
#: (see `EffectEstimate`); the predicate reports which were not applicable so a
#: caller can journal the exemption instead of inferring it.
SEARCH_GRADE_CONJUNCTS = (
    _Conjunct("ratified_protocol", "this ratified protocol", False),
    _Conjunct("preconditions", "every precondition above", False),
    _Conjunct("calibration_block_accepted",
              "a completed and accepted calibration block for the cell's "
              "(backend, phase, cell class)", True),
    _Conjunct("stopping_rule_unmodified", "the pre-committed stopping rule unmodified", True),
    _Conjunct("b_min_paired_blocks_order_randomized",
              "B_min paired blocks under order-randomized interleaving", True),
    _Conjunct("anchor_gate_passing", "a passing anchor gate", True),
    _Conjunct("aa_control_within_cadence",
              "a passing A/A control within its declared cadence", True),
    _Conjunct("controls_1_4_available_and_passing",
              "controls 1-4 available and passing", True),
    _Conjunct("control_5_passing_or_recorded_unavailable",
              "control 5 either passing or explicitly recorded "
              "HISTORICAL_REPLAY_UNAVAILABLE with an operator escalation on the record",
              True),
    _Conjunct("e_value_against_calibrated_threshold",
              "an e-value against the calibrated threshold", True),
    _Conjunct("published_mde", "a published MDE", True),
    _Conjunct("correct_stratum", "the correct stratum", True),
    _Conjunct("complete_record_grammar", "the complete record grammar below", False),
    _Conjunct("raw_samples_reproducible",
              "raw samples from which the reduction is reproducible", False),
)

_CONJUNCT_BY_ID = {c.id: c for c in SEARCH_GRADE_CONJUNCTS}

STRATUM_SELECTION = "selection"
STRATUM_CONFIRMATION = "confirmation"
STRATA = (STRATUM_SELECTION, STRATUM_CONFIRMATION)

#: Normative solve order of the calibration block. A conforming implementation
#: *"MUST record that it did"* follow it, so `CalibrationOutputs` refuses a block
#: that does not carry this exact sequence.
CALIBRATION_SOLVE_ORDER = (
    "inputs_fixed_first",
    "alpha_sel_from_max_candidates",
    "phi_estimated_from_aa_control",
    "b_min_solved_upward",
    "alpha_sel_validated_at_b_min",
    "anchor_gate_band_computed",
)

#: The e-process constructions THIS evaluator bundle implements.
#:
#: *"The e-process construction itself (its supermartingale or betting form, its
#: reducer, and its resampling method) is a property of the evaluator bundle,
#: fixed at the bundle hash; a campaign selects among constructions the bundle
#: already implements and records which one it selected."*
#:
#: This tuple is the bundle's registry-of-record, and `CalibrationOutputs`
#: refuses an id outside it. It lives here rather than in `statistics.py` for a
#: dependency reason: `statistics` imports `api`, so `api` cannot ask
#: `statistics` what it implements without a cycle, and a registry that is only
#: populated when some other module happens to have been imported first would
#: read as "no constructions are implemented" in exactly the runs that skipped
#: the import. `statistics.py` asserts at IMPORT TIME that its own
#: `CONSTRUCTIONS` registry has exactly these ids, so the two cannot drift
#: without an `ImportError` — a divergence is loud rather than a record naming a
#: construction nothing can reproduce.
E_PROCESS_CONSTRUCTION_IDS = (
    "sign_martingale_fixed_lambda/v1",
    "sign_martingale_predictable_lambda/v1",
)


# =============================================================================
# Typed inputs
# =============================================================================

#: `schemas.require.str` under this module's name. Body hoisted; ~30 call sites
#: keep reading `_require_nonempty_str(...)`, which is what they should read.
_require_nonempty_str = schemas.require.str


def _require_sha256(value: Any, label: str) -> str:
    """SHAPE ONLY, and this is the one digest validator in the keep set that is.

    **This is a KNOWN WEAKER GATE, enumerated as such.** Every other digest field
    in the package is `schemas.require.sha256`, which also refuses
    `schemas.is_placeholder_digest` — sixty-four zeros are well-formed hex and
    name an artifact nobody hashed. `AnchorIdentity.binary_sha256` and
    `linkage_sha256`, the very fields precondition 4 exists to bind, accept one.

    It is not delegated yet because delegating it is a BEHAVIOUR change, not a
    hoist: 152 tests construct an `AnchorIdentity` or an `ArtifactIdentity` from
    filler, and eight of them are in `evaluator/test_surface.py` — a test module
    for code the simplification review condemned to deletion, which this refactor
    may not touch. Fixing the fixtures there is work that gets deleted; fixing
    only the others leaves the suite red. So the tightening waits for the
    deletion to land, and until then it is a NAMED debt with a test on it rather
    than a difference between two bodies that nobody notices.

    `test_schemas_require.TestNoKeepSetModuleReDerivesAScalarValidator` carries
    this module in `_KNOWN_WEAKER_DIGEST_VALIDATORS` and fails if a SECOND module
    joins it, or if this one is fixed and the entry is left behind.
    """
    if not isinstance(value, str) or not schemas.SHA256_RE.match(value):
        raise ValueError(f"{label}: expected a lowercase sha256 hex digest, got {value!r}")
    return value


def _require_check(value: Any, label: str) -> schemas.Check:
    if not isinstance(value, schemas.Check):
        raise TypeError(
            f"{label}: expected schemas.Check (PASS/FAIL/COULD_NOT_CHECK), got "
            f"{type(value).__name__}; a bare bool cannot express the third outcome"
        )
    return value


def _require_positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{label}: must be finite, got {value!r} (an unbounded "
                         "quantity cannot derive an error budget)")
    if value <= 0:
        raise ValueError(f"{label}: must be strictly positive, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class AnchorIdentity:
    """The anchor named by source commit, binary SHA-256 and linkage SHA-256.

    Precondition 4. A constructed `AnchorIdentity` is always well-formed; use
    `parse()` when the input may be malformed, because a malformed anchor must
    become an INVALID verdict, not an exception in the caller's face.

    **`binary_sha256` NAMES ONE TOOL, and `tool` is which one.** One anchor build
    ships several binaries — T0 hashes the anchor `llama-cli`, `microbench`
    compares the plan's anchor digest against the anchor `llama-bench` it is
    about to spawn — and a single-valued digest cannot honestly name both. The
    ENFORCED RULE is therefore: *`binary_sha256` is the digest of the tool the
    record's `metric` was measured with, and `tool` is that tool's name.* The
    rule is enforced, not merely documented, in three places:

      * `identity_matches` REFUSES to call two differently-named tools the same
        anchor, even when every digest agrees — see its docstring. That is the
        one comparison every consumer already goes through
        (`_anchor_precondition`, `correctness._refuse_replay_mismatch`,
        `chain.check_anchor_matches`, `release.readiness`), so evidence captured
        against tool A cannot reach evidence captured against tool B without the
        difference surfacing as a Check.
      * `for_tool()` is the only way to attach a name, and it REFUSES to rename a
        triple already bound to a different tool — relabelling one tool's digest
        as another's is precisely the lie this field exists to prevent.
      * `short()` renders the tool into the record grammar's anchor field, so a
        journalled line says which binary the denominator came from.

    `tool` is OPTIONAL because records written before it existed named no tool,
    and an unnamed anchor must stay readable. It is never *silently* compatible
    with a named one: named-vs-unnamed is COULD_NOT_CHECK, never PASS.

    WHY NOT A PER-TOOL DIGEST TABLE. `controller.state_machine.AnchorIdentity`
    keys `binary_sha256`/`linkage_sha256` BY BACKEND, and that shape is right
    *there*: it is the campaign-wide production identity, which must describe
    every backend the tree serves so that a repointed symlink is caught for all
    of them at once. This object is a different thing — the DENOMINATOR OF ONE
    RATIO, measured by one tool. A table here would put digests into the record
    that the record's own number was not measured against, which is the same
    class of defect as a single digest naming the wrong tool, arriving by the
    other door. Tools of one anchor build are tied instead by what genuinely must
    hold across them — same `source_commit`, same `linkage_sha256` —
    in `execution.chain.check_anchor_build_is_one_build`.
    """

    source_commit: str
    binary_sha256: str
    linkage_sha256: str
    measurement_event_ids: tuple = ()
    tool: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_commit, str) or not _COMMIT_RE.match(self.source_commit):
            raise ValueError(
                f"anchor.source_commit: expected a 40-hex commit, got {self.source_commit!r}"
            )
        _require_sha256(self.binary_sha256, "anchor.binary_sha256")
        _require_sha256(self.linkage_sha256, "anchor.linkage_sha256")
        if not isinstance(self.measurement_event_ids, tuple):
            raise TypeError("anchor.measurement_event_ids must be a tuple")
        for eid in self.measurement_event_ids:
            _require_nonempty_str(eid, "anchor.measurement_event_ids[]")
        if self.tool is not None:
            _require_nonempty_str(self.tool, "anchor.tool")
            # A NAME, not a path: `llama-bench`, the way `recipes.CellRecipe.tool`
            # spells it. `/mnt/.../bin/llama-bench` and `llama-bench` would be two
            # spellings of one tool and would compare unequal, which turns the
            # check below into a source of false FAILs. Deliberately NOT a closed
            # enum: `recipes` accepts an open tool string, and a second registry
            # here would drift and reject a tool the recipe layer admits.
            if self.tool != self.tool.strip() or any(c.isspace() for c in self.tool):
                raise ValueError(
                    f"anchor.tool: {self.tool!r} carries whitespace; it is a tool NAME "
                    "and two spellings of one name are two anchors to every comparison")
            if "/" in self.tool or "\\" in self.tool:
                raise ValueError(
                    f"anchor.tool: {self.tool!r} looks like a path. Name the tool "
                    "(`llama-bench`), not where this run happened to find it — the "
                    "path is `recipes.ToolBinding`'s field and it varies per worktree")

    @classmethod
    def parse(cls, obj: Any):
        """Return `(anchor_or_None, reasons)`. Never raises on bad input."""
        if not isinstance(obj, Mapping):
            return None, (f"anchor: expected a mapping, got {type(obj).__name__}",)
        try:
            ids = obj.get("measurement_event_ids", ())
            return cls(
                source_commit=obj.get("source_commit"),
                binary_sha256=obj.get("binary_sha256"),
                linkage_sha256=obj.get("linkage_sha256"),
                measurement_event_ids=tuple(ids) if isinstance(ids, (list, tuple)) else ids,
                tool=obj.get("tool"),
            ), ()
        except (ValueError, TypeError) as exc:
            return None, (str(exc),)

    def for_tool(self, tool: str) -> "AnchorIdentity":
        """This triple, naming the tool its `binary_sha256` was taken off.

        RAISES on a rename. A capture is hashed off exactly one file, so calling
        `for_tool("llama-bench")` on a triple already bound to `llama-cli` would
        not correct a label — it would assert that the cli's digest is the
        bench's, which is the single-valued-field defect written down as a fact.
        Re-naming with the SAME tool is the identity and returns self.
        """
        _require_nonempty_str(tool, "anchor.tool")
        if self.tool is not None and self.tool != tool.strip():
            raise ValueError(
                f"anchor.binary_sha256 is already bound to {self.tool!r} and cannot be "
                f"re-named {tool.strip()!r}: one digest is the hash of one file. Two "
                "tools of one anchor build need TWO identities, tied by "
                "`chain.check_anchor_build_is_one_build` (same commit, same linkage)")
        if self.tool == tool.strip():
            return self
        return replace(self, tool=tool.strip())

    def short(self) -> str:
        """`[<tool>:]<commit[:12]>/<binary[:12]>/<linkage[:12]>` — the grammar's anchor field.

        The tool is PREFIXED when it is named, so the record grammar's `vs anchor
        …` field says which binary the denominator was measured with. An unnamed
        anchor renders exactly as it always did: an old record is not retroactively
        made to claim a tool it never recorded.
        """
        triple = (f"{self.source_commit[:12]}/{self.binary_sha256[:12]}/"
                  f"{self.linkage_sha256[:12]}")
        return triple if self.tool is None else f"{self.tool}:{triple}"

    def identity_matches(self, other: Optional["AnchorIdentity"]) -> schemas.Check:
        """Byte-for-byte re-verification. *"A rebuilt anchor is a different anchor."*

        Three components plus the tool, and the tool is why this is not a pure
        digest comparison: two triples that agree on all three digests but name
        DIFFERENT tools are not one anchor, they are one impossible anchor — the
        same bytes cannot be both binaries — and the usual way to produce that
        pair is to derive both stages' identity from a single capture and let the
        tool evaporate at the boundary. That is FAIL, not a silent PASS.

        A tool named on one side and absent on the other is COULD_NOT_CHECK, on
        `controller.state_machine.check_anchor_identity`'s rule: a detected
        difference is a FACT and outranks an incomplete observation, and an
        unobserved component is never PASS. Both sides unnamed compares exactly
        as it did before the field existed.
        """
        if other is None:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 ("no anchor identity was captured to compare against",))
        reasons = []
        for name in ("source_commit", "binary_sha256", "linkage_sha256"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine != theirs:
                reasons.append(f"anchor.{name} moved: {mine!r} -> {theirs!r}")
        unnamed = []
        if self.tool is not None and other.tool is not None and self.tool != other.tool:
            reasons.append(
                f"anchor.tool differs: {self.tool!r} vs {other.tool!r}. `binary_sha256` "
                "names ONE tool, so these are two anchors and evidence measured with one "
                "is not comparable against the other. Tools of one build are tied by "
                "`chain.check_anchor_build_is_one_build`, not by this check")
        elif (self.tool is None) != (other.tool is None):
            unnamed.append(
                f"one side names its tool ({self.tool!r} vs {other.tool!r}) and the other "
                "does not, so which binary the digest belongs to is unobserved on one "
                "side; not naming a tool is not evidence that it is the same tool")
        if reasons:
            return schemas.Check(schemas.FAIL, tuple(reasons + unnamed))
        if unnamed:
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unnamed))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        """The record's anchor block. `tool` appears only when it is named.

        Omitted-when-unnamed rather than `"tool": null`, so absence has ONE
        representation — the same rule `schemas._check_anchor_block_v3` applies to
        the anchor block itself — and so a record written before this field
        existed and one written by a caller that named no tool are byte-identical
        rather than two spellings of the same silence.
        """
        block = {
            "source_commit": self.source_commit,
            "binary_sha256": self.binary_sha256,
            "linkage_sha256": self.linkage_sha256,
            "measurement_event_ids": list(self.measurement_event_ids),
        }
        if self.tool is not None:
            block["tool"] = self.tool
        return block


@dataclass(frozen=True)
class ArtifactIdentity:
    """The candidate under test: source snapshot, binary, and linkage hashes."""

    source_sha256: str
    binary_sha256: str
    linkage_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_sha256, "artifact.source_sha256")
        _require_sha256(self.binary_sha256, "artifact.binary_sha256")
        _require_sha256(self.linkage_sha256, "artifact.linkage_sha256")

    def to_dict(self) -> dict:
        return {
            "source_sha256": self.source_sha256,
            "binary_sha256": self.binary_sha256,
            "linkage_sha256": self.linkage_sha256,
        }


@dataclass(frozen=True)
class EvaluatorIdentity:
    """Precondition 5: pinned bundle hash PLUS the runtime source-label attestation.

    The attestation reference is required, not optional: *"so that 'the evaluator
    that ran' is a checkable fact rather than an inference from an import
    statement."*
    """

    id: str
    bundle_sha256: str
    runtime_source_label_ref: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.id, "evaluator.id")
        if "/v" not in self.id:
            raise ValueError(
                f"evaluator.id {self.id!r} has no '/vN' suffix; a mutable evaluator id "
                "cannot fail closed on drift (schemas._VERSIONED_ID_RE)"
            )
        _require_sha256(self.bundle_sha256, "evaluator.bundle_sha256")
        _require_nonempty_str(self.runtime_source_label_ref, "evaluator.runtime_source_label_ref")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bundle_sha256": self.bundle_sha256,
            "runtime_source_label_ref": self.runtime_source_label_ref,
        }


@dataclass(frozen=True)
class ScopeDenominator:
    """What the cell actually measured, so a gate refuses a scope mismatch (§7.4)."""

    machine_subset: str
    numa_nodes: tuple
    devices: tuple
    cores: int

    def __post_init__(self) -> None:
        if self.machine_subset not in schemas.MACHINE_SUBSETS:
            raise ValueError(
                f"scope_denominator.machine_subset: {self.machine_subset!r} is not one of "
                f"{sorted(schemas.MACHINE_SUBSETS)}"
            )
        if not isinstance(self.numa_nodes, tuple) or not isinstance(self.devices, tuple):
            raise TypeError("scope_denominator numa_nodes/devices must be tuples")
        if isinstance(self.cores, bool) or not isinstance(self.cores, int) or self.cores < 0:
            raise ValueError(f"scope_denominator.cores: expected a non-negative int, "
                             f"got {self.cores!r}")
        if self.machine_subset == "partial" and not self.numa_nodes and not self.devices:
            raise ValueError(
                "scope_denominator: machine_subset='partial' must name the numa nodes or "
                "devices it measured, otherwise the cell's denominator is unknown"
            )

    def render(self) -> str:
        """The grammar's `scope=<denominator of what was measured>` field."""
        parts = [self.machine_subset]
        if self.numa_nodes:
            parts.append("numa" + ",".join(str(n) for n in self.numa_nodes))
        if self.devices:
            parts.append("dev" + ",".join(self.devices))
        parts.append(f"{self.cores}c")
        return "/".join(parts)

    def to_dict(self) -> dict:
        return {
            "machine_subset": self.machine_subset,
            "numa_nodes": list(self.numa_nodes),
            "devices": list(self.devices),
            "cores": self.cores,
        }


@dataclass(frozen=True)
class DeterminismReport:
    """Invariant 12: a determinism class is an interface, and `not_measured` is sayable."""

    determinism_class: str
    same_seed_repeat_runs: int

    def __post_init__(self) -> None:
        if self.determinism_class not in schemas.DETERMINISM_CLASSES:
            raise ValueError(
                f"determinism.class: {self.determinism_class!r} is not one of "
                f"{sorted(schemas.DETERMINISM_CLASSES)}"
            )
        if isinstance(self.same_seed_repeat_runs, bool) or \
                not isinstance(self.same_seed_repeat_runs, int) or \
                self.same_seed_repeat_runs < 0:
            raise ValueError("determinism.same_seed_repeat_runs must be a non-negative int")
        if self.determinism_class in ("bitwise_stable", "bitwise_unstable") and \
                self.same_seed_repeat_runs == 0:
            raise ValueError(
                "determinism: a class cannot be claimed from zero same-seed repeats "
                "(use 'not_measured')"
            )

    def to_dict(self) -> dict:
        return {"class": self.determinism_class,
                "same_seed_repeat_runs": self.same_seed_repeat_runs}


@dataclass(frozen=True)
class RecipeReceipt:
    """Precondition 6: argv came from a recipe constructor, and here is which one.

    The ABSENCE of this object is what `HAND_TYPED_ARGV` detects. There is no
    "unknown recipe" value, because an unknown recipe is a hand-typed one for
    every purpose the protocol cares about (`bench-cpu.md:8-10`).
    """

    constructor_id: str
    constructor_sha256: str
    argv_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.constructor_id, "recipe.constructor_id")
        _require_sha256(self.constructor_sha256, "recipe.constructor_sha256")
        _require_sha256(self.argv_sha256, "recipe.argv_sha256")

    def render(self) -> str:
        """The grammar's `recipe=<recipe_constructor_id>@<recipe_sha256[:12]>` field."""
        return f"{self.constructor_id}@{self.constructor_sha256[:12]}"

    def to_dict(self) -> dict:
        return {"constructor_id": self.constructor_id,
                "constructor_sha256": self.constructor_sha256,
                "argv_sha256": self.argv_sha256}


@dataclass(frozen=True)
class CampaignControls:
    """Precondition 8 — the quantities the calibration block consumes.

    *"Each MUST be finite and strictly positive; a campaign that omits one, or
    declares it as zero or unbounded, cannot derive its error budgets and MUST
    NOT start."* `__post_init__` is that rule; `parse()` turns a bad manifest into
    a reason list instead of a traceback so the run can still be journaled.
    """

    calibration_block_count: int
    contribution_floor: float
    max_candidates: int
    confirmation_admission_count: int
    max_blocks_per_candidate: int
    storage_floor_bytes_free: int

    def __post_init__(self) -> None:
        for name in ("calibration_block_count", "max_candidates",
                     "confirmation_admission_count", "max_blocks_per_candidate",
                     "storage_floor_bytes_free"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"campaign_controls.{name}: expected an int, got {value!r}")
            _require_positive_finite(value, f"campaign_controls.{name}")
        _require_positive_finite(self.contribution_floor, "campaign_controls.contribution_floor")

    @classmethod
    def parse(cls, obj: Any):
        """Return `(controls_or_None, reasons)`. Never raises on bad input."""
        if not isinstance(obj, Mapping):
            return None, (f"campaign_controls: expected a mapping, got {type(obj).__name__}",)
        names = ("calibration_block_count", "contribution_floor", "max_candidates",
                 "confirmation_admission_count", "max_blocks_per_candidate",
                 "storage_floor_bytes_free")
        missing = [n for n in names if n not in obj]
        if missing:
            return None, tuple(
                f"campaign_controls.{n}: required by precondition 8 and not declared"
                for n in missing
            )
        try:
            return cls(**{n: obj[n] for n in names}), ()
        except (ValueError, TypeError) as exc:
            return None, (str(exc),)

    def alpha_sel_ceiling(self) -> float:
        """`α_sel` MUST NOT exceed the reciprocal of `max_candidates`."""
        return 1.0 / self.max_candidates

    def alpha_conf_ceiling(self, alpha_sel: float) -> float:
        """`α_conf` MUST NOT exceed `α_sel / confirmation_admission_count`."""
        return alpha_sel / self.confirmation_admission_count


@dataclass(frozen=True)
class CalibrationOutputs:
    """The four derived outputs of the calibration block, plus their provenance.

    *"No value in this list may be supplied as a literal."* This object cannot
    enforce that a number was derived rather than typed — only the reducer that
    produced it can — so it enforces the next best thing, which is every relation
    the protocol states between the outputs, plus the recorded solve order, plus
    the raw-sample reference the manifest must retain. A block missing any of
    those is refused here rather than accepted and quietly used.

    *"There is no fifth output"*: `storage_floor_bytes_free` is a CampaignControls
    field (precondition 7 / `MEASUREMENT.md` §5), not a calibration output. Two
    definitions of one manifest field is the defect that note exists to prevent.
    """

    backend: str
    phase: str
    cell_class: str
    noise_floor_phi: float
    b_min_blocks: int
    alpha_sel: float
    alpha_conf: float
    anchor_gate_band: tuple
    accepted: bool
    solve_order_recorded: tuple
    samples_ref: str
    e_process_construction_id: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.backend, "calibration.backend")
        _require_nonempty_str(self.phase, "calibration.phase")
        _require_nonempty_str(self.cell_class, "calibration.cell_class")
        _require_positive_finite(self.noise_floor_phi, "calibration.noise_floor_phi")
        if isinstance(self.b_min_blocks, bool) or not isinstance(self.b_min_blocks, int):
            raise ValueError("calibration.b_min_blocks must be an int")
        _require_positive_finite(self.b_min_blocks, "calibration.b_min_blocks")
        _require_positive_finite(self.alpha_sel, "calibration.alpha_sel")
        _require_positive_finite(self.alpha_conf, "calibration.alpha_conf")
        if self.alpha_sel >= 1.0 or self.alpha_conf >= 1.0:
            raise ValueError("calibration: error budgets must be < 1")
        if self.alpha_conf > self.alpha_sel:
            raise ValueError(
                f"calibration.alpha_conf ({self.alpha_conf}) is looser than alpha_sel "
                f"({self.alpha_sel}); the protocol forbids a confirmation budget looser "
                "than selection"
            )
        if not isinstance(self.anchor_gate_band, tuple) or len(self.anchor_gate_band) != 2:
            raise ValueError("calibration.anchor_gate_band must be a (low, high) tuple")
        low, high = self.anchor_gate_band
        for v, n in ((low, "low"), (high, "high")):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"calibration.anchor_gate_band.{n} must be a finite number")
        if low >= high:
            raise ValueError("calibration.anchor_gate_band must be (low < high)")
        if not isinstance(self.accepted, bool):
            raise TypeError("calibration.accepted must be a bool")
        if tuple(self.solve_order_recorded) != CALIBRATION_SOLVE_ORDER:
            raise ValueError(
                "calibration.solve_order_recorded does not record the normative solve "
                f"order; expected {list(CALIBRATION_SOLVE_ORDER)}, got "
                f"{list(self.solve_order_recorded)}. The outputs are mutually referential "
                "and the order is what makes them well-defined."
            )
        _require_nonempty_str(self.samples_ref, "calibration.samples_ref")
        _require_nonempty_str(self.e_process_construction_id,
                              "calibration.e_process_construction_id")
        if self.e_process_construction_id not in E_PROCESS_CONSTRUCTION_IDS:
            raise ValueError(
                f"calibration.e_process_construction_id "
                f"{self.e_process_construction_id!r} is not implemented by this evaluator "
                f"bundle; implemented constructions are {list(E_PROCESS_CONSTRUCTION_IDS)}. "
                "'A campaign selects among constructions the bundle already implements and "
                "records which one it selected' — a recorded construction the bundle does "
                "not implement is a reduction nobody can reproduce, and the calibrated "
                "threshold was validated for a procedure that never ran."
            )

    def threshold_for(self, stratum: str) -> float:
        """`1/α_sel` for selection, `1/α_conf` for confirmation."""
        if stratum == STRATUM_SELECTION:
            return 1.0 / self.alpha_sel
        if stratum == STRATUM_CONFIRMATION:
            return 1.0 / self.alpha_conf
        raise ValueError(f"unknown stratum {stratum!r}; expected one of {list(STRATA)}")

    def check_against_controls(self, controls: Optional[CampaignControls]) -> schemas.Check:
        """PASS/FAIL/COULD_NOT_CHECK on the derived-budget relations of output 3."""
        if controls is None:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("campaign controls were not declared; the error budgets cannot be "
                 "checked against max_candidates / confirmation_admission_count",))
        reasons = []
        ceiling = controls.alpha_sel_ceiling()
        if self.alpha_sel > ceiling:
            reasons.append(
                f"alpha_sel {self.alpha_sel} exceeds 1/max_candidates ({ceiling}); the "
                "expected number of false selections across the campaign would exceed one")
        conf_ceiling = controls.alpha_conf_ceiling(self.alpha_sel)
        if self.alpha_conf > conf_ceiling:
            reasons.append(
                f"alpha_conf {self.alpha_conf} exceeds alpha_sel/"
                f"confirmation_admission_count ({conf_ceiling})")
        if self.b_min_blocks > controls.max_blocks_per_candidate:
            reasons.append(
                f"b_min_blocks {self.b_min_blocks} exceeds the declared ceiling "
                f"max_blocks_per_candidate {controls.max_blocks_per_candidate}; the "
                "calibration FAILS and the campaign does not start")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "phase": self.phase,
            "cell_class": self.cell_class,
            "noise_floor_phi": self.noise_floor_phi,
            "b_min_blocks": self.b_min_blocks,
            "alpha_sel": self.alpha_sel,
            "alpha_conf": self.alpha_conf,
            "anchor_gate_band": list(self.anchor_gate_band),
            "accepted": self.accepted,
            "solve_order_recorded": list(self.solve_order_recorded),
            "samples_ref": self.samples_ref,
            "e_process_construction_id": self.e_process_construction_id,
        }


HISTORICAL_REPLAY_UNAVAILABLE = "HISTORICAL_REPLAY_UNAVAILABLE"


@dataclass(frozen=True)
class ControlPanel:
    """The four mandatory controls plus the accept-side control's declared contract.

    "Controls — four mandatory, plus one accept-side control run under a declared
    contract". Control 5's unavailable branch is *"normative, not a silent skip"*:
    this constructor REFUSES an unavailable control 5 that does not name both a
    reason and an operator escalation reference, because a campaign that runs
    four controls and reports as though it ran five is exactly what the branch
    exists to prevent.
    """

    positive: schemas.Check
    neutral: schemas.Check
    degraded_negative: schemas.Check
    aa: schemas.Check
    historical_replay: Optional[schemas.Check]
    historical_replay_unavailable_reason: Optional[str] = None
    operator_escalation_ref: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("positive", "neutral", "degraded_negative", "aa"):
            _require_check(getattr(self, name), f"controls.{name}")
        if self.historical_replay is None:
            if not (isinstance(self.historical_replay_unavailable_reason, str)
                    and self.historical_replay_unavailable_reason.strip()):
                raise ValueError(
                    "controls: historical_win_replay is unavailable but no reason was "
                    f"recorded; the protocol requires {HISTORICAL_REPLAY_UNAVAILABLE} "
                    "in the journal and the manifest, naming the backend and the reason"
                )
            if not (isinstance(self.operator_escalation_ref, str)
                    and self.operator_escalation_ref.strip()):
                raise ValueError(
                    "controls: historical_win_replay is unavailable and no operator "
                    "escalation reference was recorded; 'whether the campaign proceeds "
                    "on four controls is the operator's call, taken once, on the record'"
                )
        else:
            _require_check(self.historical_replay, "controls.historical_replay")

    @property
    def available(self) -> int:
        return 4 if self.historical_replay is None else 5

    def marker(self) -> str:
        """The grammar's `controls=<4/5|5/5>[ (HISTORICAL_REPLAY_UNAVAILABLE)]` field."""
        if self.historical_replay is None:
            return f"4/5 ({HISTORICAL_REPLAY_UNAVAILABLE})"
        return "5/5"

    def check_1_to_4(self) -> schemas.Check:
        """Controls 1-4 available AND passing. Any non-PASS blocks ranking."""
        reasons = []
        outcome = schemas.PASS
        for name in ("positive", "neutral", "degraded_negative", "aa"):
            chk = getattr(self, name)
            if chk.outcome != schemas.PASS:
                reasons.append(f"control {name}: {chk.outcome} {list(chk.reasons)}")
                if chk.outcome == schemas.FAIL:
                    outcome = schemas.FAIL
                elif outcome != schemas.FAIL:
                    outcome = schemas.COULD_NOT_CHECK
        return schemas.Check(outcome, tuple(reasons))

    def check_5(self) -> schemas.Check:
        """Control 5 passing, OR recorded unavailable with an operator escalation."""
        if self.historical_replay is None:
            return schemas.Check(
                schemas.PASS,
                (f"{HISTORICAL_REPLAY_UNAVAILABLE}: "
                 f"{self.historical_replay_unavailable_reason}; escalated to the operator "
                 f"at {self.operator_escalation_ref}",))
        if self.historical_replay.outcome == schemas.PASS:
            return schemas.Check(schemas.PASS)
        return schemas.Check(
            self.historical_replay.outcome,
            ("historical-win replay did not promote; this is a GATE DEFECT, not a "
             "research finding — it halts the campaign and is escalated to the "
             "operator",) + tuple(self.historical_replay.reasons))

    def to_dict(self) -> dict:
        return {
            "positive": self.positive.outcome,
            "neutral": self.neutral.outcome,
            "degraded_negative": self.degraded_negative.outcome,
            "aa": self.aa.outcome,
            "historical_replay": (None if self.historical_replay is None
                                  else self.historical_replay.outcome),
            "historical_replay_unavailable_reason": self.historical_replay_unavailable_reason,
            "operator_escalation_ref": self.operator_escalation_ref,
            "marker": self.marker(),
        }


EFFECT_NOT_MEASURED = "not_measured"
EFFECT_BELOW_NOISE_FLOOR = "below_noise_floor"
EFFECT_NO_DETECTABLE_DIFFERENCE = "no_detectable_difference"
EFFECT_EVIDENCE_BELOW_THRESHOLD = "evidence_below_threshold"
EFFECT_IMPROVEMENT = "improvement"
EFFECT_REGRESSION = "regression"

EFFECT_RESOLUTIONS = (
    EFFECT_NOT_MEASURED, EFFECT_BELOW_NOISE_FLOOR, EFFECT_NO_DETECTABLE_DIFFERENCE,
    EFFECT_EVIDENCE_BELOW_THRESHOLD, EFFECT_IMPROVEMENT, EFFECT_REGRESSION,
)

#: The two resolutions that place a candidate on the ordering at all. Below the
#: noise floor is not a small win; it is not a win. Below the MDE is *"no
#: detectable difference, which is a result and a decision, not a failed
#: experiment"* — and a result you cannot order.
_RANKABLE_RESOLUTIONS = (EFFECT_IMPROVEMENT, EFFECT_REGRESSION)

#: The two resolutions where the estimate never cleared the campaign's OWN
#: sensitivity: the magnitude is inside the calibrated noise floor, or inside the
#: MDE at the realized block count. They are the sub-floor half of what
#: `_RANKABLE_RESOLUTIONS` excludes, and they are PUBLIC because a downstream
#: reader that must not order such a cell needs this module's answer to *"which
#: resolutions carry no orderable magnitude"*, not a second copy of it — a second
#: copy is a second copy that drifts, and the half that drifts is whichever one
#: has fewer tests.
#:
#: `EFFECT_EVIDENCE_BELOW_THRESHOLD` is deliberately NOT here. That estimate
#: cleared both the floor and the MDE: it is a DETECTABLE magnitude with
#: insufficient evidence, which is a different thing to report and a different
#: thing to hide. It is unrankable for a reason that is not parity, and a reader
#: that folded the two together would either call a measured degradation "at
#: parity" or make it invisible.
SUB_FLOOR_RESOLUTIONS = (EFFECT_BELOW_NOISE_FLOOR, EFFECT_NO_DETECTABLE_DIFFERENCE)


def _require_known_resolution(resolution: str) -> str:
    if resolution not in EFFECT_RESOLUTIONS:
        raise ValueError(
            f"{resolution!r} is not one of {list(EFFECT_RESOLUTIONS)}; an unknown "
            "resolution must be a refusal, never a silent False — a mistyped one "
            "would read as 'rankable' or 'not sub-floor' by falling off the end of "
            "the vocabulary"
        )
    return resolution


def is_rankable_resolution(resolution: str) -> bool:
    """Does this resolution place a candidate on the ordering at all?

    The exported form of `_RANKABLE_RESOLUTIONS`. Selecting a cell as "the
    weakest" or "the best" IS a rank, so anything downstream that selects must
    ask this rather than re-deriving it.
    """
    return _require_known_resolution(resolution) in _RANKABLE_RESOLUTIONS


def is_sub_floor_resolution(resolution: str) -> bool:
    """Did the estimate fail to clear the campaign's own floor or MDE?

    True means parity — *"a result and a decision, not a failed experiment"* —
    and a result that carries no orderable magnitude at all.
    """
    return _require_known_resolution(resolution) in SUB_FLOOR_RESOLUTIONS


@dataclass(frozen=True)
class EffectEstimate:
    """One rate comparison, with everything the protocol requires published WITH it.

    "Statistical requirements": the e-value AND its threshold AND the MDE AND the
    calibrated noise floor live in this object, so the reduction cannot be
    reported without them. `lcb_descriptive` is carried only as a labelled
    magnitude summary — *"An LCB MAY be carried beside the e-value as a labelled
    descriptive statistic … no decision in the enumerated authority is taken on
    it"* — and nothing in `_resolve_effect` reads it.

    The presence of an `EffectEstimate` is what makes a record a RATE COMPARISON,
    which in turn is what makes the rate-only search-grade conjuncts and the
    rate-only void conditions applicable. That predicate is derived from the
    record, not declared by the tier, so a T0 record cannot dodge the statistical
    conjunction by reporting a number under a correctness label.
    """

    metric: str
    metric_direction: str
    value: float
    e_value: float
    threshold: float
    mde: float
    noise_floor: float
    paired_blocks: int
    stratum: str
    raw_samples: tuple
    raw_samples_ref: str
    lcb_descriptive: Optional[float] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.metric, "effect.metric")
        if self.metric_direction not in schemas.METRIC_DIRECTIONS:
            raise ValueError(
                f"effect.metric_direction: {self.metric_direction!r} is not one of "
                f"{sorted(schemas.METRIC_DIRECTIONS)}"
            )
        for name in ("value", "e_value", "threshold", "mde", "noise_floor"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"effect.{name}: expected a finite number, got {v!r}")
        _require_positive_finite(self.threshold, "effect.threshold")
        if self.e_value < 0:
            raise ValueError("effect.e_value must be non-negative")
        if self.mde < 0 or self.noise_floor < 0:
            raise ValueError("effect.mde and effect.noise_floor must be non-negative")
        if isinstance(self.paired_blocks, bool) or not isinstance(self.paired_blocks, int) \
                or self.paired_blocks < 1:
            raise ValueError("effect.paired_blocks must be a positive int")
        if self.stratum not in STRATA:
            raise ValueError(f"effect.stratum: {self.stratum!r} is not one of {list(STRATA)}")
        if not isinstance(self.raw_samples, tuple) or not self.raw_samples:
            raise ValueError(
                "effect.raw_samples must be a non-empty tuple; an estimate without raw "
                "samples is a self-reported score and is not reproducible (§7.4)"
            )
        _require_nonempty_str(self.raw_samples_ref, "effect.raw_samples_ref")

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "metric_direction": self.metric_direction,
            "value": self.value,
            "e_value": self.e_value,
            "threshold": self.threshold,
            "mde": self.mde,
            "noise_floor": self.noise_floor,
            "paired_blocks": self.paired_blocks,
            "stratum": self.stratum,
            "raw_samples_ref": self.raw_samples_ref,
            # Labelled, and labelled in the record itself, not just in prose.
            "lcb_descriptive": self.lcb_descriptive,
            "lcb_label": "descriptive",
            # AK-TR-2: the delta is never serialised without its own instrument
            # floor adjacent to it.  Consumers may style this string, but do not
            # have to rediscover whether the row is inside the noise.
            "delta_display": self.delta_display(),
            "inside_noise_floor": abs(self.value) <= self.noise_floor,
        }

    def delta_display(self) -> str:
        relation = ("INSIDE_NOISE_FLOOR" if abs(self.value) <= self.noise_floor
                    else "ABOVE_NOISE_FLOOR")
        return (f"delta={self.value:+.9g}; noise_floor={self.noise_floor:.9g}; "
                f"{relation}")


def _resolve_effect(effect: Optional[EffectEstimate]) -> str:
    """Classify an estimate. Pure; reads nothing but the estimate itself."""
    if effect is None:
        return EFFECT_NOT_MEASURED
    magnitude = abs(effect.value)
    if magnitude <= effect.noise_floor:
        return EFFECT_BELOW_NOISE_FLOOR
    if magnitude < effect.mde:
        return EFFECT_NO_DETECTABLE_DIFFERENCE
    if effect.e_value < effect.threshold:
        return EFFECT_EVIDENCE_BELOW_THRESHOLD
    improving = (effect.value > 0 if effect.metric_direction == "higher_better"
                 else effect.value < 0)
    return EFFECT_IMPROVEMENT if improving else EFFECT_REGRESSION


@dataclass(frozen=True)
class WindowAttestations:
    """Everything the protocol requires attested at window open AND window close.

    Every field is required and there are NO defaults, on purpose: a default
    would let a caller omit an attestation and have the omission read as
    satisfied. Tests build this explicitly; there is deliberately no
    `all_clear()` convenience constructor in this module, because a fixture that
    fabricates PASS is the fixture that removes the signal under test.
    """

    # Precondition 1 + void 1
    resource_claim_receipt: str
    resource_claim_open: schemas.Check
    resource_claim_close: schemas.Check
    resource_claim_same_holder: schemas.Check
    # Precondition 2 + void 8
    no_concurrent_inference: schemas.Check
    preflight_attestation_ref: str
    # Precondition 3 + void 2
    host_receipt: str
    host_health: schemas.Check
    # Precondition 4 + void 6
    anchor_at_open: Optional[AnchorIdentity]
    anchor_at_close: Optional[AnchorIdentity]
    # "Anchor gate" (statistical requirements) + void 3
    anchor_gate: schemas.Check
    # Precondition 5 + void 5
    evaluator_bundle: schemas.Check
    runtime_source_label: schemas.Check
    # Precondition 6 + void 7 (absence == hand-typed argv)
    recipe: Optional[RecipeReceipt]
    # Precondition 7 + void 9
    storage_open: schemas.Check
    storage_close: schemas.Check
    # Selection/confirmation split + void 10
    strata: schemas.Check
    # Pre-committed stopping rule + void 11
    stopping_rule_id: str
    rule_immutability: schemas.Check
    # Order control
    order_randomized: schemas.Check
    order_seed: str
    # A/A cadence + controls
    aa_cadence: schemas.Check
    controls: ControlPanel
    # Calibration + void 12
    calibration: schemas.Check
    # Void 11's OTHER subject. "What voids a run" names four things whose
    # post-hoc change voids: the stopping rule, the calibration outputs, the
    # objective, AND the control definitions. `rule_immutability` carries the
    # first three; the fourth is computed by `controls.verify_control_definitions`
    # (and its predicate digest) and, before this field existed, had nowhere to
    # go — a campaign could rebind a control predicate and every record in the
    # window still read as clean. `controls.window_control_attestations()` is the
    # projection that fills it.
    control_definitions_immutable: schemas.Check
    # Record grammar's `raw=` for non-rate records
    raw_evidence_ref: str

    def __post_init__(self) -> None:
        for name in ("resource_claim_open", "resource_claim_close",
                     "resource_claim_same_holder", "no_concurrent_inference",
                     "host_health", "anchor_gate", "evaluator_bundle",
                     "runtime_source_label", "storage_open", "storage_close", "strata",
                     "rule_immutability", "order_randomized", "aa_cadence", "calibration",
                     "control_definitions_immutable"):
            _require_check(getattr(self, name), f"window.{name}")
        for name in ("resource_claim_receipt", "preflight_attestation_ref", "host_receipt",
                     "stopping_rule_id", "order_seed", "raw_evidence_ref"):
            _require_nonempty_str(getattr(self, name), f"window.{name}")
        if not isinstance(self.controls, ControlPanel):
            raise TypeError("window.controls must be a ControlPanel")
        if self.recipe is not None and not isinstance(self.recipe, RecipeReceipt):
            raise TypeError("window.recipe must be a RecipeReceipt or None")
        for name in ("anchor_at_open", "anchor_at_close"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, AnchorIdentity):
                raise TypeError(f"window.{name} must be an AnchorIdentity or None")


@dataclass(frozen=True)
class TransferRatio:
    """One write-time correspondence from this event to a completed anchor tier.

    Both signed effects travel with the ratio.  A reader may verify the division,
    but may not invent a correspondence between two old events that did not name
    each other when the later event was written.
    """

    event_id: str
    tier: str
    source_effect: float
    target_effect: float

    def __post_init__(self) -> None:
        _require_nonempty_str(self.event_id, "transfer.event_id")
        if not self.event_id.startswith("ake-"):
            raise ValueError("transfer.event_id must start with 'ake-'")
        if self.tier not in schemas.TIERS:
            raise ValueError(f"transfer.tier {self.tier!r} is not one of "
                             f"{sorted(schemas.TIERS)}")
        for name in ("source_effect", "target_effect"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError(f"transfer.{name} must be finite")
        if self.target_effect == 0:
            raise ValueError("transfer.target_effect must be non-zero")

    @property
    def ratio(self) -> float:
        return self.source_effect / self.target_effect

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "tier": self.tier,
            "source_effect": self.source_effect,
            "target_effect": self.target_effect,
            "ratio": self.ratio,
        }


@dataclass(frozen=True)
class EvaluationRequest:
    """The identity of one evaluation. Carries no results and no verdict."""

    event_id: str
    campaign_id: str
    candidate_id: str
    tier: str
    backend: str
    phase: str
    cell_class: str
    protocol_id: str
    artifact: ArtifactIdentity
    anchor: Optional[AnchorIdentity]
    evaluator: EvaluatorIdentity
    scope_denominator: ScopeDenominator
    scope_manifest_sha256: str
    co_residency: str
    determinism: DeterminismReport
    metric: str
    metric_direction: str
    reps: int
    change_class: str
    anchor_tier: str
    transfer_ratio_to: tuple
    created_at: str
    campaign_controls: Optional[CampaignControls]
    calibration: Optional[CalibrationOutputs]
    device_state: Optional[devices.DeviceState] = None

    def __post_init__(self) -> None:
        for name, prefix in (("event_id", "ake-"), ("campaign_id", "ak-"),
                             ("candidate_id", "akc-")):
            value = getattr(self, name)
            _require_nonempty_str(value, name)
            if not value.startswith(prefix):
                raise ValueError(f"{name}: {value!r} must start with {prefix!r}")
        # Deliberately NOT admit_tier(): a request naming T3 is constructible, so
        # that the refusal happens at the dispatch boundary where it is journaled,
        # rather than as a TypeError in whoever built the request.
        if self.tier not in schemas.TIERS:
            raise ValueError(f"tier: {self.tier!r} is not one of {sorted(schemas.TIERS)}")
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"backend: {self.backend!r} is not one of "
                             f"{sorted(schemas.BACKENDS)}")
        for name in ("phase", "cell_class", "protocol_id", "metric"):
            _require_nonempty_str(getattr(self, name), name)
        if self.metric_direction not in schemas.METRIC_DIRECTIONS:
            raise ValueError(f"metric_direction: {self.metric_direction!r} is not one of "
                             f"{sorted(schemas.METRIC_DIRECTIONS)}")
        if isinstance(self.reps, bool) or not isinstance(self.reps, int) or self.reps < 1:
            raise ValueError("reps: zero reps is not a measurement (MEASUREMENT.md:13)")
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise ValueError(f"change_class: {self.change_class!r} is not one of "
                             f"{sorted(schemas.CHANGE_CLASSES)}")
        if self.anchor_tier not in schemas.TIERS:
            raise ValueError(f"anchor_tier: {self.anchor_tier!r} is not one of "
                             f"{sorted(schemas.TIERS)}")
        if not isinstance(self.transfer_ratio_to, tuple):
            raise TypeError("transfer_ratio_to must be a tuple of TransferRatio")
        for row in self.transfer_ratio_to:
            if not isinstance(row, TransferRatio):
                raise TypeError("transfer_ratio_to must contain only TransferRatio")
            if row.event_id == self.event_id:
                raise ValueError("a transfer target cannot be the source event")
            if row.tier != self.anchor_tier:
                raise ValueError(f"transfer target tier {row.tier!r} does not match "
                                 f"anchor_tier {self.anchor_tier!r}")
        _require_sha256(self.scope_manifest_sha256, "scope_manifest_sha256")
        if not _CO_RESIDENCY_RE.match(self.co_residency or ""):
            raise ValueError(f"co_residency: {self.co_residency!r} must be 'single' or "
                             "'co_resident:<lineup_id>'")
        if not _DATE_RE.match(self.created_at or ""):
            raise ValueError(f"created_at: {self.created_at!r} must be an ISO-8601 timestamp")
        for name, klass in (("artifact", ArtifactIdentity), ("evaluator", EvaluatorIdentity),
                            ("scope_denominator", ScopeDenominator),
                            ("determinism", DeterminismReport)):
            if not isinstance(getattr(self, name), klass):
                raise TypeError(f"{name} must be a {klass.__name__}")
        if self.anchor is not None and not isinstance(self.anchor, AnchorIdentity):
            raise TypeError("anchor must be an AnchorIdentity or None")
        if self.device_state is not None and not isinstance(
                self.device_state, devices.DeviceState):
            raise TypeError("device_state must be a devices.DeviceState or None")


# =============================================================================
# Gate results and scans
# =============================================================================

@dataclass(frozen=True)
class GateResult:
    """One gate's outcome, produced by a tier runner and never by this module.

    `requires_anchor` is what makes precondition 4 structural rather than
    conventional: a gate that declares itself a comparison against the anchor
    (coherence, byte-identity, reference parity, determinism-vs-anchor) has its
    PASS demoted to COULD_NOT_CHECK when no anchor is bound. `kernel_eval.sh`
    reported `COH="coherent"` with no baseline at all; a gate that answers
    "coherent" here without an anchor answers COULD_NOT_CHECK instead.
    """

    gate_id: str
    gate_class: str
    check: schemas.Check
    requires_anchor: bool = False
    evidence_ref: Optional[str] = None
    notes: tuple = ()

    def __post_init__(self) -> None:
        _require_nonempty_str(self.gate_id, "gate.gate_id")
        if self.gate_class not in GATE_CLASSES:
            raise ValueError(f"gate.gate_class: {self.gate_class!r} is not one of "
                             f"{list(GATE_CLASSES)}")
        _require_check(self.check, f"gate[{self.gate_id}].check")
        if not isinstance(self.requires_anchor, bool):
            raise TypeError("gate.requires_anchor must be a bool")
        if not isinstance(self.notes, tuple):
            raise TypeError("gate.notes must be a tuple")

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "gate_class": self.gate_class,
            "outcome": self.check.outcome,
            "reasons": list(self.check.reasons),
            "requires_anchor": self.requires_anchor,
            "evidence_ref": self.evidence_ref,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class VoidFinding:
    """One triggered void condition, with the protocol's own phrase for it."""

    reason: str
    protocol_phrase: str
    outcome: str
    detail: tuple = ()

    def __post_init__(self) -> None:
        if self.reason not in VOID_REASONS:
            raise ValueError(f"void reason {self.reason!r} is not one of {list(VOID_REASONS)}")
        # FAIL and COULD_NOT_CHECK both void the window, and the record says which
        # it was. Neither is conflated with the other.
        if self.outcome not in (schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise ValueError(f"void outcome must be FAIL or COULD_NOT_CHECK, got "
                             f"{self.outcome!r}")

    def to_dict(self) -> dict:
        return {"reason": self.reason, "protocol_phrase": self.protocol_phrase,
                "outcome": self.outcome, "detail": list(self.detail)}


@dataclass(frozen=True)
class VoidScan:
    """The result of checking every void condition, including the ones skipped."""

    findings: tuple
    evaluated: tuple
    not_applicable: tuple

    @property
    def voided(self) -> bool:
        return bool(self.findings)

    def reasons(self) -> tuple:
        return tuple(f.reason for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "evaluated": list(self.evaluated),
            "not_applicable": list(self.not_applicable),
        }


@dataclass(frozen=True)
class PreconditionScan:
    """The protocol's eight preconditions, each as a three-outcome Check."""

    checks: tuple  # ((precondition_id, Check), ...) — ordered, hashable

    def __post_init__(self) -> None:
        ids = tuple(pid for pid, _ in self.checks)
        if ids != PRECONDITION_IDS:
            raise ValueError(
                f"precondition scan must cover exactly {list(PRECONDITION_IDS)} in order, "
                f"got {list(ids)}"
            )

    def get(self, precondition_id: str) -> schemas.Check:
        for pid, chk in self.checks:
            if pid == precondition_id:
                return chk
        raise KeyError(precondition_id)

    @property
    def satisfied(self) -> bool:
        return all(chk.outcome == schemas.PASS for _, chk in self.checks)

    @property
    def unsatisfied(self) -> tuple:
        return tuple(pid for pid, chk in self.checks if chk.outcome != schemas.PASS)

    def to_dict(self) -> dict:
        return {pid: {"outcome": chk.outcome, "reasons": list(chk.reasons),
                      "phrase": PRECONDITION_PHRASES[pid]}
                for pid, chk in self.checks}


@dataclass(frozen=True)
class SearchGradeResult:
    """The "Search-grade requires ALL of" conjunction, with the failed conjuncts named."""

    satisfied: bool
    evaluated: tuple
    failed: tuple
    not_applicable: tuple
    reasons: tuple  # ((conjunct_id, (reason, ...)), ...)

    def reason_for(self, conjunct_id: str) -> tuple:
        for cid, reasons in self.reasons:
            if cid == conjunct_id:
                return reasons
        return ()

    def to_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "evaluated": list(self.evaluated),
            "failed": list(self.failed),
            "not_applicable": list(self.not_applicable),
            "reasons": {cid: list(rs) for cid, rs in self.reasons},
        }


# =============================================================================
# Precondition and void checking
# =============================================================================

def _combine(*checks: schemas.Check) -> schemas.Check:
    """Worst-of over Checks: FAIL beats COULD_NOT_CHECK beats PASS.

    Delegates to `schemas.Check.worst_of`, which is the package's one lattice.
    `_combine()` with no arguments used to answer PASS — the fail-open this
    module's own prose names at `check_gate_derivation_is_locked` ("an empty gate
    list derives to PASS and that is a fail-open verdict") — and now answers
    COULD_NOT_CHECK.
    """
    return schemas.Check.worst_of(checks)


def _anchor_precondition(request: EvaluationRequest,
                         window: WindowAttestations) -> schemas.Check:
    """Precondition 4, including the window-open/window-close re-verification."""
    if request.anchor is None:
        return schemas.Check(
            schemas.FAIL,
            ("no anchor was named for this run; 'A run without an explicit anchor is "
             "INVALID — never \"correct\", never \"coherent\", never \"byte-identical\"'",))
    open_check = request.anchor.identity_matches(window.anchor_at_open)
    close_check = request.anchor.identity_matches(window.anchor_at_close)
    return _combine(open_check, close_check)


def check_preconditions(request: EvaluationRequest,
                        window: WindowAttestations) -> PreconditionScan:
    """Evaluate P-AK-SEARCH-1 "Preconditions (all enforced or attested per run)".

    Every one is a three-outcome Check. An attestation that could not be read is
    COULD_NOT_CHECK — it is NOT a satisfied precondition, and `satisfied` is True
    only on all-PASS.
    """
    if not isinstance(request, EvaluationRequest):
        raise TypeError("request must be an EvaluationRequest")
    if not isinstance(window, WindowAttestations):
        raise TypeError("window must be a WindowAttestations")

    claim = _combine(window.resource_claim_open, window.resource_claim_close,
                     window.resource_claim_same_holder)

    recipe = (schemas.Check(schemas.PASS) if window.recipe is not None
              else schemas.Check(schemas.FAIL,
                                 ("no recipe constructor identity was recorded; every "
                                  "measurement command line inside this protocol's scope "
                                  "is emitted by a recipe constructor, and hand-typed "
                                  "argv voids the run",)))

    if request.campaign_controls is None:
        controls = schemas.Check(
            schemas.FAIL,
            ("the campaign manifest did not declare every quantity the calibration "
             "block consumes; a campaign that omits one, or declares it as zero or "
             "unbounded, cannot derive its error budgets and MUST NOT start",))
    else:
        controls = schemas.Check(schemas.PASS)

    host_health = _combine(window.host_health, _device_state_check(request))
    return PreconditionScan(checks=(
        ("resource_claim_held_whole_window", claim),
        ("no_concurrent_inference", window.no_concurrent_inference),
        ("host_health_tier", host_health),
        ("explicit_immutable_anchor", _anchor_precondition(request, window)),
        ("evaluator_identity", _combine(window.evaluator_bundle, window.runtime_source_label)),
        ("codified_recipe", recipe),
        ("storage_headroom", _combine(window.storage_open, window.storage_close)),
        ("declared_campaign_controls", controls),
    ))


def check_void_conditions(request: EvaluationRequest,
                          window: WindowAttestations,
                          *,
                          rate_comparison: bool) -> VoidScan:
    """Evaluate P-AK-SEARCH-1 "What voids a run" as a checked precondition set.

    FAIL and COULD_NOT_CHECK both void the window; the `VoidFinding` records which
    it was, so "the claim was held by someone else" and "the claim state could not
    be read" stay distinguishable in the journal while both fail closed. Observing
    that a device *looks* free is TOCTOU, not exclusion.

    `rate_comparison` selects the rate-only conditions (anchor gate, A/A control,
    strata, calibration). When False they appear in `not_applicable`, never
    silently omitted.
    """
    if not isinstance(request, EvaluationRequest):
        raise TypeError("request must be an EvaluationRequest")
    if not isinstance(window, WindowAttestations):
        raise TypeError("window must be a WindowAttestations")
    if not isinstance(rate_comparison, bool):
        raise TypeError("rate_comparison must be a bool")

    candidates = [
        (VOID_CLAIM_NOT_HELD,
         _combine(window.resource_claim_open, window.resource_claim_close,
                  window.resource_claim_same_holder)),
        (VOID_HOST_HEALTH_TIER_VIOLATION,
         _combine(window.host_health, _device_state_check(request))),
        (VOID_ANCHOR_GATE_FAILED, window.anchor_gate),
        (VOID_AA_CONTROL_FAILED, window.controls.aa),
        (VOID_EVALUATOR_BUNDLE_UNVERIFIED,
         _combine(window.evaluator_bundle, window.runtime_source_label)),
        (VOID_ANCHOR_MISSING_OR_MUTATED, _anchor_precondition(request, window)),
        (VOID_HAND_TYPED_ARGV,
         schemas.Check(schemas.PASS) if window.recipe is not None
         else schemas.Check(schemas.FAIL, ("argv was not emitted by a recipe constructor",))),
        (VOID_CONCURRENT_INFERENCE, window.no_concurrent_inference),
        (VOID_STORAGE_EXHAUSTED, _combine(window.storage_open, window.storage_close)),
        (VOID_STRATA_VIOLATION, window.strata),
        # "any post-hoc change to the stopping rule, the calibration outputs, the
        # objective, OR THE CONTROL DEFINITIONS". The control-definitions digest
        # is a separate observation from the stopping-rule commitment, and it is
        # combined here rather than folded into `rule_immutability` upstream so
        # the journaled reason names which of the two moved.
        (VOID_POST_HOC_RULE_CHANGE,
         _combine(window.rule_immutability, window.control_definitions_immutable)),
        (VOID_INCOMPLETE_CALIBRATION, _calibration_void_check(request, window)),
    ]

    findings, evaluated, skipped = [], [], []
    for reason, chk in candidates:
        if reason in _RATE_ONLY_VOIDS and not rate_comparison:
            skipped.append(reason)
            continue
        evaluated.append(reason)
        if chk.outcome != schemas.PASS:
            findings.append(VoidFinding(
                reason=reason,
                protocol_phrase=VOID_REASON_PHRASES[reason],
                outcome=chk.outcome,
                detail=tuple(chk.reasons),
            ))
    return VoidScan(findings=tuple(findings), evaluated=tuple(evaluated),
                    not_applicable=tuple(skipped))


def _device_state_check(request: EvaluationRequest) -> schemas.Check:
    """GPU state is verdict-bearing; CPU health remains in the host receipt."""
    if request.backend != "llama_gpu":
        return schemas.Check(schemas.PASS)
    if request.device_state is None:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("no parsed GPU device_state was attached; a rocm-smi text blob cannot "
             "establish that the device stayed unthrottled",))
    return request.device_state.check()


def _calibration_void_check(request: EvaluationRequest,
                            window: WindowAttestations) -> schemas.Check:
    if request.calibration is None:
        return schemas.Check(
            schemas.FAIL,
            ("no calibration block was resolved for this cell; a campaign that cannot "
             "complete its calibration block MUST NOT rank any candidate",))
    cal = request.calibration
    reasons = []
    if not cal.accepted:
        reasons.append("the calibration block for this cell was not accepted")
    if (cal.backend, cal.phase, cal.cell_class) != (request.backend, request.phase,
                                                    request.cell_class):
        reasons.append(
            f"calibration was solved for ({cal.backend}, {cal.phase}, {cal.cell_class}) "
            f"but the cell is ({request.backend}, {request.phase}, {request.cell_class}); "
            "values calibrated under a different host state, backend, phase, or cell "
            "class MUST NOT be reused")
    relations = cal.check_against_controls(request.campaign_controls)
    if relations.outcome != schemas.PASS:
        reasons.extend(relations.reasons)
    if reasons:
        outcome = (schemas.COULD_NOT_CHECK
                   if relations.outcome == schemas.COULD_NOT_CHECK and len(reasons) == len(
                       relations.reasons)
                   else schemas.FAIL)
        return schemas.Check(outcome, tuple(reasons))
    return _combine(window.calibration)


# =============================================================================
# Search-grade conjunction
# =============================================================================

def evaluate_search_grade(*,
                          request: EvaluationRequest,
                          window: WindowAttestations,
                          preconditions: PreconditionScan,
                          effect: Optional[EffectEstimate],
                          grammar_complete: schemas.Check) -> SearchGradeResult:
    """Implement "Search-grade requires ALL of" as an explicit conjunction.

    Returns which conjuncts failed and why, so the caller can journal the reason
    rather than journaling "not search-grade". *"Missing ANY of these makes the
    record INVALID. There is no weaker-but-usable state."*
    """
    rate = effect is not None
    results: dict = {}

    # 1 — this ratified protocol
    if request.protocol_id != PROTOCOL_VERSIONED_ID:
        results["ratified_protocol"] = (
            f"record cites protocol {request.protocol_id!r}, not {PROTOCOL_VERSIONED_ID!r}",)

    # 2 — every precondition above
    if not preconditions.satisfied:
        results["preconditions"] = tuple(
            f"{pid}: {preconditions.get(pid).outcome} {list(preconditions.get(pid).reasons)}"
            for pid in preconditions.unsatisfied)

    if rate:
        cal = request.calibration
        # 3 — a completed and accepted calibration block for this cell
        cal_check = _calibration_void_check(request, window)
        cal_reasons = list(cal_check.reasons) if cal_check.outcome != schemas.PASS else []
        # ... AND the record must actually be scored against THAT block's floor.
        # `_resolve_effect` reads `effect.noise_floor`, so a record carrying any
        # other number is ranked against a floor this cell never calibrated —
        # zeroing it turns a sub-floor estimate into an `improvement`. "An
        # estimate whose magnitude does not exceed phi MUST NOT be ranked,
        # banked, or composed, whatever its evidence value."
        if cal is not None and not math.isclose(
                effect.noise_floor, cal.noise_floor_phi, rel_tol=1e-12, abs_tol=0.0):
            cal_reasons.append(
                f"record carries floor={effect.noise_floor} but the calibrated noise floor "
                f"phi for ({cal.backend}, {cal.phase}, {cal.cell_class}) is "
                f"{cal.noise_floor_phi}; the effect resolution is decided against the "
                "floor ON THE RECORD, so a substituted floor is a substituted verdict")
        if cal_reasons:
            results["calibration_block_accepted"] = tuple(cal_reasons)

        # 4 — the pre-committed stopping rule unmodified
        if window.rule_immutability.outcome != schemas.PASS:
            results["stopping_rule_unmodified"] = (
                (f"stopping rule {window.stopping_rule_id!r} immutability: "
                 f"{window.rule_immutability.outcome}",) + tuple(window.rule_immutability.reasons))

        # 5 — B_min paired blocks under order-randomized interleaving
        block_reasons = []
        if cal is None:
            block_reasons.append("no calibrated B_min to compare the realized block count against")
        elif effect.paired_blocks < cal.b_min_blocks:
            block_reasons.append(
                f"{effect.paired_blocks} paired blocks is below the calibrated B_min "
                f"({cal.b_min_blocks})")
        if window.order_randomized.outcome != schemas.PASS:
            block_reasons.append(
                f"order randomization: {window.order_randomized.outcome} "
                f"{list(window.order_randomized.reasons)}; blocked designs are forbidden "
                "because thermal and page-cache drift alias onto the arm effect")
        if block_reasons:
            results["b_min_paired_blocks_order_randomized"] = tuple(block_reasons)

        # 6 — a passing anchor gate
        if window.anchor_gate.outcome != schemas.PASS:
            results["anchor_gate_passing"] = (
                (f"anchor gate: {window.anchor_gate.outcome}",) + tuple(window.anchor_gate.reasons))

        # 7 — a passing A/A control within its declared cadence
        aa_reasons = []
        if window.controls.aa.outcome != schemas.PASS:
            aa_reasons.append(f"A/A control: {window.controls.aa.outcome} "
                              f"{list(window.controls.aa.reasons)}")
        if window.aa_cadence.outcome != schemas.PASS:
            aa_reasons.append(f"A/A cadence: {window.aa_cadence.outcome} "
                              f"{list(window.aa_cadence.reasons)}")
        if aa_reasons:
            results["aa_control_within_cadence"] = tuple(aa_reasons)

        # 8 — controls 1-4 available and passing.
        # The definitions digest rides here rather than in its own conjunct
        # because it is what makes "controls 1-4" mean the ratified four:
        # "Control definitions, fixtures, expected directions, and seeds live
        # inside the evaluator bundle under the measurement trust boundary and
        # MUST NOT be modified by any process inside the loop."
        c14 = _combine(window.controls.check_1_to_4(),
                       window.control_definitions_immutable)
        if c14.outcome != schemas.PASS:
            results["controls_1_4_available_and_passing"] = tuple(c14.reasons)

        # 9 — control 5 passing or explicitly recorded unavailable + escalated
        c5 = window.controls.check_5()
        if c5.outcome != schemas.PASS:
            results["control_5_passing_or_recorded_unavailable"] = tuple(c5.reasons)

        # 10 — an e-value against the CALIBRATED threshold
        e_reasons = []
        if cal is None:
            e_reasons.append("no calibrated threshold exists to test the e-value against")
        else:
            expected = cal.threshold_for(effect.stratum)
            if not math.isclose(effect.threshold, expected, rel_tol=1e-9, abs_tol=0.0):
                e_reasons.append(
                    f"record carries threshold {effect.threshold} but the calibrated "
                    f"threshold for the {effect.stratum} stratum is {expected} "
                    f"(1/alpha)")
        if e_reasons:
            results["e_value_against_calibrated_threshold"] = tuple(e_reasons)

        # 11 — a published MDE, written into the same record as the estimate
        if not math.isfinite(effect.mde):
            results["published_mde"] = ("MDE is not a finite number",)

        # 12 — the correct stratum
        stratum_reasons = []
        if effect.stratum not in STRATA:
            stratum_reasons.append(f"unknown stratum {effect.stratum!r}")
        if window.strata.outcome != schemas.PASS:
            stratum_reasons.append(
                f"strata partition: {window.strata.outcome} {list(window.strata.reasons)}; "
                "no block may serve both strata")
        if stratum_reasons:
            results["correct_stratum"] = tuple(stratum_reasons)

    # 13 — the complete record grammar
    if grammar_complete.outcome != schemas.PASS:
        results["complete_record_grammar"] = (
            (f"record grammar: {grammar_complete.outcome}",) + tuple(grammar_complete.reasons))

    # 14 — raw samples from which the reduction is reproducible
    raw_reasons = []
    if rate and not effect.raw_samples:
        raw_reasons.append("the reduction has no raw samples to be recomputed from")
    if not window.raw_evidence_ref.strip():
        raw_reasons.append("no raw-evidence reference was recorded")
    if raw_reasons:
        results["raw_samples_reproducible"] = tuple(raw_reasons)

    evaluated, skipped = [], []
    for conjunct in SEARCH_GRADE_CONJUNCTS:
        if conjunct.rate_only and not rate:
            skipped.append(conjunct.id)
        else:
            evaluated.append(conjunct.id)

    failed = tuple(cid for cid in evaluated if cid in results)
    reasons = tuple((cid, results[cid]) for cid in failed)
    return SearchGradeResult(
        satisfied=not failed,
        evaluated=tuple(evaluated),
        failed=failed,
        not_applicable=tuple(skipped),
        reasons=reasons,
    )


# =============================================================================
# THE VERDICT — computed, never stamped
# =============================================================================

#: Held by this module alone. `compute_verdict()` is the only caller that passes
#: it. It is the FIRST of two locks; the second, and the one that actually
#: matters, is the re-derivation in `Verdict.__post_init__`, which holds even if
#: someone reaches in and takes this object.
_MINT_TOKEN = object()

_Derived = namedtuple(
    "_Derived",
    "status effect_resolution speed_rank_admissible integrity_flags derivation",
)


def _derive(*, gates: tuple, void_findings: tuple, search_grade: SearchGradeResult,
            anchor: Optional[AnchorIdentity], effect: Optional[EffectEstimate]) -> _Derived:
    """The ONLY place a status comes from. Pure function of its arguments.

    Order of precedence, and why:

    1. **Voids first.** A voided window is INVALID and *"MUST NOT be recorded as a
       candidate failure, because a drifted anchor says nothing whatever about
       the candidate."* INVALID therefore dominates FAIL.
    2. **No anchor is a void.** Belt and braces with (1): even if a caller built a
       `VoidScan` that missed it, an unbound anchor cannot yield anything but
       INVALID here.
    3. **Not search-grade is INVALID.** *"Missing ANY of these makes the record
       INVALID. There is no weaker-but-usable state."* Gate failures are still
       recorded in `integrity_flags` and in the derivation trail, so applying
       INVALID loses no signal — both statuses are non-ranking.
    4. **Otherwise the worst gate outcome**, by class severity.
    """
    flags: list = []
    trail: list = []

    gate_status = STATUS_PASS
    for gate in sorted(gates, key=lambda g: (g.gate_class, g.gate_id)):
        if gate.check.outcome == schemas.PASS:
            continue
        if gate.check.outcome == schemas.FAIL:
            escalation = _ON_GATE_FAIL[gate.gate_class]
        else:
            escalation = _ON_GATE_COULD_NOT_CHECK[gate.gate_class]
        gate_status = _worse(gate_status, escalation)
        flags.append(f"{gate.gate_class.upper()}:{gate.gate_id}:{gate.check.outcome}")
        trail.append(
            f"gate {gate.gate_id} ({gate.gate_class}) returned {gate.check.outcome} "
            f"-> {escalation}")

    if void_findings:
        status = STATUS_INVALID
        for finding in void_findings:
            flags.append(f"VOID:{finding.reason}:{finding.outcome}")
            trail.append(
                f"VOID: {finding.reason} — {finding.protocol_phrase} "
                f"({finding.outcome})")
        trail.append(
            "status INVALID: 'A voided run is journaled as INVALID with its reason, and "
            "is never silently discarded'; it is NOT recorded as a candidate failure")
    elif anchor is None:
        status = STATUS_INVALID
        flags.append(f"VOID:{VOID_ANCHOR_MISSING_OR_MUTATED}:{schemas.FAIL}")
        trail.append(
            "status INVALID: no anchor is bound. 'A run without an explicit anchor is "
            "INVALID — never \"correct\", never \"coherent\", never \"byte-identical\".'")
    elif not search_grade.satisfied:
        status = STATUS_INVALID
        for cid in search_grade.failed:
            flags.append(f"SEARCH_GRADE_MISSING:{cid}")
            trail.append(f"search-grade conjunct not satisfied: {cid} — "
                         f"{_CONJUNCT_BY_ID[cid].phrase}")
        trail.append(
            "status INVALID: 'Missing ANY of these makes the record INVALID. There is no "
            "weaker-but-usable state.'")
    else:
        status = gate_status
        trail.append(f"status {status}: derived from {len(gates)} gate result(s); "
                     "no void condition triggered and the search-grade conjunction holds")

    resolution = _resolve_effect(effect)
    admissible = status == STATUS_PASS and resolution in _RANKABLE_RESOLUTIONS

    if effect is not None and not admissible:
        if status != STATUS_PASS:
            trail.append(
                f"speed rank WITHHELD: status is {status}. 'A candidate failing any of "
                "them receives no speed rank at all — not a penalised one.'")
        else:
            trail.append(f"speed rank WITHHELD: effect resolution is {resolution}")
    elif effect is None:
        trail.append("speed rank WITHHELD: no rate comparison was measured")

    return _Derived(
        status=status,
        effect_resolution=resolution,
        speed_rank_admissible=admissible,
        integrity_flags=tuple(flags),
        derivation=tuple(trail),
    )


@dataclass(frozen=True)
class Verdict:
    """A COMPUTED verdict. There is no path that stamps a status.

    Two independent locks:

    1. `__init__` refuses to run unless it is handed the module-private mint
       token, which only `compute_verdict()` passes. This stops the accident.
    2. `__post_init__` RE-DERIVES `status`, `effect_resolution`,
       `speed_rank_admissible`, `integrity_flags` and `derivation` from the gate
       results, void findings, anchor and search-grade result stored on the very
       same object, and raises `VerdictTampering` on any disagreement. This stops
       the determined caller: reaching in and taking `_MINT_TOKEN` buys nothing,
       because the only status this object will accept is the one its own
       evidence implies.

    `kernel_eval.sh` ended with `"status":"OK"` in a `printf` format string. This
    class is the structural answer to that line.
    """

    tier: str
    status: str
    gates: tuple
    void_findings: tuple
    search_grade: SearchGradeResult
    anchor: Optional[AnchorIdentity]
    effect: Optional[EffectEstimate]
    effect_resolution: str
    speed_rank_admissible: bool
    integrity_flags: tuple
    derivation: tuple
    mint: InitVar[Any] = None

    def __post_init__(self, mint: Any) -> None:
        if mint is not _MINT_TOKEN:
            raise VerdictTampering(
                "Verdict is not constructible directly — a verdict object derives from "
                "the gate results it aggregates. Call compute_verdict()."
            )
        if self.status not in VERDICT_STATUSES:
            raise VerdictTampering(f"status {self.status!r} is not one of "
                                   f"{list(VERDICT_STATUSES)}")
        for gate in self.gates:
            if not isinstance(gate, GateResult):
                raise VerdictTampering("gates must all be GateResult instances")
        for finding in self.void_findings:
            if not isinstance(finding, VoidFinding):
                raise VerdictTampering("void_findings must all be VoidFinding instances")
        if not isinstance(self.search_grade, SearchGradeResult):
            raise VerdictTampering("search_grade must be a SearchGradeResult")

        recomputed = _derive(
            gates=self.gates,
            void_findings=self.void_findings,
            search_grade=self.search_grade,
            anchor=self.anchor,
            effect=self.effect,
        )
        for name in ("status", "effect_resolution", "speed_rank_admissible",
                     "integrity_flags", "derivation"):
            stored, derived = getattr(self, name), getattr(recomputed, name)
            if stored != derived:
                raise VerdictTampering(
                    f"verdict.{name} does not follow from this verdict's own evidence: "
                    f"stored {stored!r}, derived {derived!r}. A verdict is computed from "
                    f"its gate results; it is never supplied."
                )
        # The same invariant schemas.validate_evaluation_event enforces, asserted
        # here so a malformed verdict cannot reach the record builder.
        if self.status == STATUS_PASS and self.integrity_flags:
            raise VerdictTampering(
                "status 'pass' with non-empty integrity_flags: correctness is "
                "lexicographically first (invariant 6)")

    # -- the speed-rank boundary ------------------------------------------------

    def rank_key(self) -> tuple:
        """The sort key for this candidate. RAISES unless the rank was earned.

        "Correctness precedence": *"A candidate failing any of them receives no
        speed rank at all — not a penalised one. A penalised rank is still a rank,
        and any search that ranks incorrect candidates will eventually surface one
        whose penalty is smaller than its apparent speed gain."* Returning a
        sentinel here would recreate exactly that: a ranking loop would sort the
        sentinel somewhere. So this raises.
        """
        if not self.speed_rank_admissible:
            raise SpeedRankUnavailable(self.speed_rank_withheld_reason())
        if self.effect is None:  # pragma: no cover - _RANKABLE_RESOLUTIONS forbids it
            raise VerdictTampering(
                "speed_rank_admissible is True with no effect estimate; the resolution "
                "and the estimate disagree")
        signed = (self.effect.value if self.effect.metric_direction == "higher_better"
                  else -self.effect.value)
        return (signed, self.effect.e_value, self.effect.paired_blocks)

    def speed_rank_withheld_reason(self) -> str:
        if self.speed_rank_admissible:
            return ""
        failing = tuple(
            g.gate_id for g in self.gates
            if g.gate_class in SPEED_BLOCKING_GATE_CLASSES
            and g.check.outcome != schemas.PASS
        )
        if self.status != STATUS_PASS:
            detail = (f"status is {self.status!r}")
            if failing:
                detail += f"; failing prior gates: {list(failing)}"
            if self.void_findings:
                detail += f"; void reasons: {[f.reason for f in self.void_findings]}"
            if self.search_grade.failed:
                detail += f"; search-grade missing: {list(self.search_grade.failed)}"
            return (f"no speed rank at all — not a penalised one ({detail}); "
                    "correctness, quality, numerical safety, integrity and stability are "
                    "lexicographically prior to speed")
        return (f"no speed rank: effect resolution is {self.effect_resolution!r}; "
                "an estimate whose magnitude does not exceed the campaign noise floor "
                "MUST NOT be ranked, banked, or composed, and |effect| < MDE is "
                "'no detectable difference', which is a result and a decision")

    # -- projection -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "status": self.status,
            "gates": [g.to_dict() for g in self.gates],
            "void_findings": [f.to_dict() for f in self.void_findings],
            "search_grade": self.search_grade.to_dict(),
            "anchor": None if self.anchor is None else self.anchor.to_dict(),
            "effect": None if self.effect is None else self.effect.to_dict(),
            "effect_resolution": self.effect_resolution,
            "speed_rank_admissible": self.speed_rank_admissible,
            "speed_rank_withheld_reason": self.speed_rank_withheld_reason(),
            "integrity_flags": list(self.integrity_flags),
            "derivation": list(self.derivation),
        }


def compute_verdict(*,
                    tier: str,
                    gates: Sequence[GateResult],
                    void_scan: VoidScan,
                    search_grade: SearchGradeResult,
                    anchor: Optional[AnchorIdentity],
                    effect: Optional[EffectEstimate] = None) -> Verdict:
    """Aggregate gate results into a verdict. The ONLY constructor of `Verdict`.

    Demotes any anchor-requiring gate that returned PASS with no anchor bound.
    That demotion is the direct answer to `kernel_eval.sh`'s `COH="coherent"`:
    the gate is not lying, it simply had nothing to compare against, and
    COULD_NOT_CHECK is what that means.
    """
    admit_tier(tier)
    if not isinstance(void_scan, VoidScan):
        raise TypeError("void_scan must be a VoidScan")
    if not isinstance(search_grade, SearchGradeResult):
        raise TypeError("search_grade must be a SearchGradeResult")
    if anchor is not None and not isinstance(anchor, AnchorIdentity):
        raise TypeError("anchor must be an AnchorIdentity or None")
    if effect is not None and not isinstance(effect, EffectEstimate):
        raise TypeError("effect must be an EffectEstimate or None")

    resolved: list = []
    for gate in gates:
        if not isinstance(gate, GateResult):
            raise TypeError(f"gates must be GateResult instances, got "
                            f"{type(gate).__name__}")
        if anchor is None and gate.requires_anchor and gate.check.outcome == schemas.PASS:
            resolved.append(GateResult(
                gate_id=gate.gate_id,
                gate_class=gate.gate_class,
                check=schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    ("this gate compares against the anchor and no anchor is bound; "
                     "absence of a comparison is not evidence of equivalence "
                     "(P-AK-SEARCH-1 precondition 4)",) + tuple(gate.check.reasons)),
                requires_anchor=True,
                evidence_ref=gate.evidence_ref,
                notes=gate.notes + ("PASS demoted to COULD_NOT_CHECK: no anchor bound",),
            ))
        else:
            resolved.append(gate)

    gate_tuple = tuple(resolved)
    derived = _derive(
        gates=gate_tuple,
        void_findings=void_scan.findings,
        search_grade=search_grade,
        anchor=anchor,
        effect=effect,
    )
    return Verdict(
        tier=tier,
        status=derived.status,
        gates=gate_tuple,
        void_findings=void_scan.findings,
        search_grade=search_grade,
        anchor=anchor,
        effect=effect,
        effect_resolution=derived.effect_resolution,
        speed_rank_admissible=derived.speed_rank_admissible,
        integrity_flags=derived.integrity_flags,
        derivation=derived.derivation,
        mint=_MINT_TOKEN,
    )


def rank_candidates(verdicts: Iterable[Verdict]) -> tuple:
    """Return `(ranked, unrankable)` — never a silently truncated list.

    `ranked` is best-first by `rank_key()`. `unrankable` is
    `((verdict, reason), ...)` for everything that did not earn a rank, so the
    caller can journal WHY each candidate is absent instead of discovering a
    shorter list.
    """
    ranked, unrankable = [], []
    for verdict in verdicts:
        if not isinstance(verdict, Verdict):
            raise TypeError(f"expected Verdict, got {type(verdict).__name__}")
        if verdict.speed_rank_admissible:
            ranked.append(verdict)
        else:
            unrankable.append((verdict, verdict.speed_rank_withheld_reason()))
    ranked.sort(key=lambda v: v.rank_key(), reverse=True)
    return tuple(ranked), tuple(unrankable)


# =============================================================================
# Record grammar
# =============================================================================

def compose_attestation_ref(window: WindowAttestations,
                            evaluator: EvaluatorIdentity) -> str:
    """Build `claim_grammar.attestation_ref` from `res` + `host` + `srclabel`.

    "Record grammar", reconciliation note: *"where the design's own
    evaluation-event schema requires an `attestation_ref` field … that field is
    satisfied by `res` + `host` + `srclabel` together."* The grammar itself
    carries no `attest` field, because attestation refers to a claim and this
    record is not one.
    """
    return (f"res={window.resource_claim_receipt};host={window.host_receipt};"
            f"srclabel={evaluator.runtime_source_label_ref}")


_GRAMMAR_ALWAYS_REQUIRED = (
    "category", "tier", "eval", "srclabel", "res", "host", "anchor", "recipe",
    "stratum", "scope", "det", "raw", "campaign", "controls", "date",
)
_GRAMMAR_RATE_REQUIRED = ("metric", "value", "direction", "blocks", "e", "thr", "MDE", "floor")


def check_record_grammar_complete(*,
                                  request: EvaluationRequest,
                                  window: WindowAttestations,
                                  effect: Optional[EffectEstimate]) -> schemas.Check:
    """FAIL naming every grammar field that cannot be filled from this run.

    *"A record omitting any field of this template is INVALID."* The always-required
    set is the one the "Record grammar" paragraph enumerates in prose; the
    rate-required set is the statistical remainder of the template, applicable
    when the record carries a rate comparison. When it does not, `stratum` falls
    back to the window's declared stopping-rule context and the statistical fields
    render `n/a` — which is stated in the line, never elided from it.
    """
    missing: list = []

    if request.anchor is None:
        missing.append("anchor: no source commit / binary sha256 / linkage sha256")
    if window.recipe is None:
        missing.append("recipe: no recipe-constructor identity was recorded")
    if not window.resource_claim_receipt.strip():
        missing.append("res: no resource-claim receipt")
    if not window.host_receipt.strip():
        missing.append("host: no host-health receipt")
    if not request.evaluator.runtime_source_label_ref.strip():
        missing.append("srclabel: no runtime source-label attestation reference")
    if not window.raw_evidence_ref.strip() and effect is None:
        missing.append("raw: no reference to the raw samples")
    if effect is not None and not effect.raw_samples_ref.strip():
        missing.append("raw: no reference to the raw samples")
    if not _DATE_RE.match(request.created_at or ""):
        missing.append("date: created_at is not an ISO-8601 date")

    if effect is not None:
        for name, value in (("e", effect.e_value), ("thr", effect.threshold),
                            ("MDE", effect.mde), ("floor", effect.noise_floor),
                            ("value", effect.value)):
            if not math.isfinite(value):
                missing.append(f"{name}: not a finite number")
        if effect.paired_blocks < 1:
            missing.append("blocks: no paired blocks were realized")
        # The grammar's head is ONE triple: `<metric> <value> <direction>`. It
        # renders `request.metric` and `request.metric_direction` beside
        # `effect.value`, so a reduction of a different quantity would be printed
        # under this cell's metric name. `MEASUREMENT.md:25-26` forbids
        # substituting one metric for another and P-AK-SEARCH-1's "Metric"
        # section leaves that prohibition "unaffected here" — so a disagreement
        # is a grammar defect, not a rounding detail.
        if effect.metric != request.metric:
            missing.append(
                f"metric: the record's cell measures {request.metric!r} but the estimate "
                f"is of {effect.metric!r}; the grammar's <metric> <value> <direction> is "
                "one triple and substituting one metric for another is forbidden")
        if effect.metric_direction != request.metric_direction:
            missing.append(
                f"direction: the cell declares {request.metric_direction!r} but the "
                f"estimate declares {effect.metric_direction!r}; the rendered line would "
                "state a direction the estimate was not oriented to, and rank_key() signs "
                "the value by the ESTIMATE's direction")

    required = _GRAMMAR_ALWAYS_REQUIRED + (_GRAMMAR_RATE_REQUIRED if effect is not None else ())
    if missing:
        return schemas.Check(
            schemas.FAIL,
            (f"grammar fields required for this record: {list(required)}",) + tuple(missing))
    # Even on PASS the applied field set is stated, so a reader never has to infer
    # which template fields this record was actually held to.
    return schemas.Check(schemas.PASS,
                         (f"grammar fields required for this record: {list(required)}",))


def _fmt(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def render_search_record_grammar(*,
                                 request: EvaluationRequest,
                                 window: WindowAttestations,
                                 verdict: Verdict,
                                 effect: Optional[EffectEstimate]) -> str:
    """Render the protocol's grammar template for this record.

    Template, verbatim from "Record grammar"::

        <metric> <value> <higher-better|lower-better>, tier <T0|T1|T2>, vs anchor
        <anchor_commit[:12]>/<anchor_binary_sha256[:12]>/<anchor_linkage_sha256[:12]>
        — SEARCH RECORD, NOT A CLAIM [P-AK-SEARCH-1, category=CANDIDATE, blocks=<n>,
        e=<e-value>, thr=<1/α>, MDE=<mde>, floor=<φ>, stratum=<selection|confirmation>,
        det=<determinism-class>, scope=<denominator of what was measured>,
        controls=<4/5|5/5>[ (HISTORICAL_REPLAY_UNAVAILABLE)], campaign=<campaign_id>,
        eval=<bundle_sha256[:12]>, srclabel=<ref>, recipe=<id>@<sha[:12]>,
        res=<claim_receipt>, host=<host_receipt>, raw=<raw_samples_ref>, YYYY-MM-DD]

    The anchor field is `AnchorIdentity.short()`, which PREFIXES the tool when the
    anchor names one (`llama-bench:<commit>/<binary>/<linkage>`). The protocol's
    template is unchanged: the triple is still the triple, and the prefix says
    which binary of that build the single-valued digest is — without it a reader
    cannot tell whether the denominator came from the tool the metric was
    measured with.

    A field with no value for this record renders `n/a` rather than being dropped,
    and `check_record_grammar_complete()` is what decides whether that is legal.
    A record whose verdict is INVALID still renders a line: the line is how the
    void reason travels with the number.
    """
    direction = ("higher-better" if request.metric_direction == "higher_better"
                 else "lower-better")
    anchor_txt = request.anchor.short() if request.anchor is not None else "NO-ANCHOR"
    value_txt = _fmt(effect.value) if effect is not None else "n/a"
    recipe_txt = window.recipe.render() if window.recipe is not None else "HAND-TYPED"
    raw_txt = effect.raw_samples_ref if effect is not None else window.raw_evidence_ref
    date_txt = request.created_at[:10]

    fields = [
        f"{PROTOCOL_ID}",
        "category=CANDIDATE",
        f"blocks={effect.paired_blocks if effect is not None else 'n/a'}",
        f"e={_fmt(effect.e_value) if effect is not None else 'n/a'}",
        f"thr={_fmt(effect.threshold) if effect is not None else 'n/a'}",
        f"MDE={_fmt(effect.mde) if effect is not None else 'n/a'}",
        f"floor={_fmt(effect.noise_floor) if effect is not None else 'n/a'}",
        f"stratum={effect.stratum if effect is not None else 'n/a'}",
        f"det={request.determinism.determinism_class}",
        f"scope={request.scope_denominator.render()}",
        f"controls={window.controls.marker()}",
        f"campaign={request.campaign_id}",
        f"eval={request.evaluator.bundle_sha256[:12]}",
        f"srclabel={request.evaluator.runtime_source_label_ref}",
        f"recipe={recipe_txt}",
        f"res={window.resource_claim_receipt}",
        f"host={window.host_receipt}",
        f"raw={raw_txt}",
        date_txt,
    ]
    head = (f"{request.metric} {value_txt} {direction}, tier {verdict.tier}, "
            f"vs anchor {anchor_txt}")
    return f"{head} — {RECORD_CLASS} [{', '.join(fields)}]"


# =============================================================================
# Evaluation event emission
# =============================================================================

def _canonicalizable(value: Any, path: str) -> Any:
    """Convert a raw-sample structure into something `schemas.canonical_json` accepts.

    `EffectEstimate.raw_samples` is a TUPLE because the estimate is a frozen,
    hashable dataclass, and `statistics.PairedBlock.to_tuple()` therefore hands
    back nested tuples. `schemas.canonical_json` REFUSES tuples outright
    (*"tuple is not canonicalizable (use a list)"*), so every reduction the
    reducer actually produced raised `TypeError` out of `content_hash(event)` —
    the record could not be hashed, journaled, or emitted. Two modules were each
    right about their own requirement and nothing converted between them.

    The conversion is tuple -> list ONLY. Anything else `schemas` cannot
    canonicalize RAISES here, naming its path, rather than being coerced to a
    string: a sample the record cannot represent is a record defect, and
    stringifying it would put an unparseable value where a number belongs.
    """
    if isinstance(value, (list, tuple)):
        return [_canonicalizable(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        return {str(k): _canonicalizable(v, f"{path}.{k}") for k, v in value.items()}
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{path}: raw samples carry a non-finite float ({value!r}); the record "
                "cannot be canonicalized and a record that cannot be hashed cannot be "
                "journaled")
        return value
    raise TypeError(
        f"{path}: raw samples carry a {type(value).__name__}, which "
        f"schemas.canonical_json cannot represent. Coercing it to a string would put "
        f"an unparseable value where the reduction's inputs belong.")


def _vector(gates: Sequence[GateResult], *classes: str) -> dict:
    """Per-case vector, never a rolled-up verdict (§7.4)."""
    return {
        g.gate_id: {
            "outcome": g.check.outcome,
            "reasons": list(g.check.reasons),
            "requires_anchor": g.requires_anchor,
            "evidence_ref": g.evidence_ref,
        }
        for g in gates if g.gate_class in classes
    }


def _anchor_block_for_event(anchor: Any) -> dict:
    """The `anchor` block for a run that CLAIMS one. Raises rather than softening.

    `EvaluationRequest` already refuses a non-`AnchorIdentity` anchor and
    `AnchorIdentity.__post_init__` already refuses malformed fields, so reaching
    any raise below means the object was mutated past its own constructor
    (`object.__setattr__` on a frozen dataclass does exactly that). The record is
    the last place that can still catch it, and the alternative — writing the
    mutated block out — puts an unresolvable anchor into the primary journal
    under a status that says the run was fine.
    """
    if not isinstance(anchor, AnchorIdentity):
        raise AnchorMissing(
            f"an anchor was claimed as a {type(anchor).__name__}, not an AnchorIdentity; "
            "the record cannot name an anchor it cannot read"
        )
    block = anchor.to_dict()
    parsed, reasons = AnchorIdentity.parse(block)
    if parsed is None:
        raise AnchorMissing(
            "an anchor was CLAIMED but is malformed, so it names nothing that can be "
            "re-verified byte-for-byte at window close: " + "; ".join(reasons)
        )
    fabricated = [name for name in ("source_commit", "binary_sha256", "linkage_sha256")
                  if schemas.is_placeholder_digest(block.get(name))]
    if fabricated:
        raise AnchorMissing(
            f"the claimed anchor carries placeholder digests at {fabricated}; a "
            "placeholder reads as a resolved anchor to every downstream reader. A run "
            "with no anchor omits the block entirely and is journaled INVALID"
        )
    return block


def build_evaluation_event(*,
                           request: EvaluationRequest,
                           window: WindowAttestations,
                           verdict: Verdict,
                           effect: Optional[EffectEstimate],
                           preconditions: PreconditionScan) -> dict:
    """Project this run into the current evaluation-event schema (v5).

    **An anchor-less voided run emits a record.** `evaluation_event.v3` permits
    the `anchor` block to be structurally ABSENT when the record's status is
    `invalid` and its `integrity_flags` name an anchor void reason — exactly the
    ANCHOR-MISSING case — so *"A voided run is journaled as INVALID with its
    reason, and is never silently discarded"* is now satisfiable through the
    primary record and not only through `durable_payload`. No digest is invented
    to make that happen; the block is omitted, and `check_anchor_binding` still
    FAILs the record for having no anchor, which is the correct reading.

    RAISES `AnchorMissing` in the two cases where emitting would be a lie: an
    anchor claimed but unreadable or fabricated (see `_anchor_block_for_event`),
    and a run with NO anchor whose verdict does not declare an anchor void — the
    schema would reject that record, and the emitter refuses it here for the same
    reason rather than handing the journal something that cannot be appended.

    `anchor.source_commit` is now a REQUIRED, validated field of the record
    (precondition 4 names the anchor by all three components). Fields the
    protocol's grammar needs and v3 still has no top-level home for (`stratum`,
    `controls`, `recipe`, `srclabel`, the calibrated threshold/floor, the
    search-grade result) are carried INSIDE existing free-form blocks —
    `performance.search_discipline`, `evaluator` — rather than by adding
    top-level keys. `schemas.py` is the single source of truth for record shape
    and amending it is a separate, human-reviewed change.
    """
    if request.anchor is None:
        # The SAME fact `schemas._check_anchor_block_v3` admits the record on, read
        # from the same vector, so the emitter and the validator cannot disagree
        # about what "this run was voided for its anchor" means.
        declared = schemas.declared_anchor_void_reasons(
            {"integrity_flags": list(verdict.integrity_flags)})
        if verdict.status != STATUS_INVALID or not declared:
            raise AnchorMissing(
                "cannot emit an evaluation_event for a run with no anchor whose verdict "
                f"is status={verdict.status!r} declaring anchor voids {declared}: the "
                f"anchor block may be omitted only by an INVALID record naming one of "
                f"{sorted(schemas.ANCHOR_VOID_REASONS)}, and there is no digest to "
                "fabricate in its place. The run is still durable through "
                "durable_payload()."
            )
        anchor_block = None
    else:
        anchor_block = _anchor_block_for_event(request.anchor)

    correctness = _vector(verdict.gates, GATE_CORRECTNESS, GATE_NUMERICAL_SAFETY)
    quality = _vector(verdict.gates, GATE_QUALITY)
    stability = _vector(verdict.gates, GATE_STABILITY, GATE_INTEGRITY)
    mechanism = _vector(verdict.gates, GATE_MECHANISM)

    cal = request.calibration
    discipline = {
        "stratum": effect.stratum if effect is not None else None,
        "stopping_rule_id": window.stopping_rule_id,
        "order_randomization_seed": window.order_seed,
        "order_randomized": window.order_randomized.outcome,
        "recipe": None if window.recipe is None else window.recipe.to_dict(),
        "controls": window.controls.to_dict(),
        "calibration": None if cal is None else cal.to_dict(),
        "preconditions": preconditions.to_dict(),
        "search_grade": verdict.search_grade.to_dict(),
        "void_findings": [f.to_dict() for f in verdict.void_findings],
        "effect_resolution": verdict.effect_resolution,
        "speed_rank_admissible": verdict.speed_rank_admissible,
        "raw_evidence_ref": window.raw_evidence_ref,
        "preflight_attestation_ref": window.preflight_attestation_ref,
        "record_class": RECORD_CLASS,
    }

    uncertainty = None if effect is None else {
        "e_value": effect.e_value,
        "threshold": effect.threshold,
        "mde": effect.mde,
        "noise_floor": effect.noise_floor,
        "e_process_construction_id": (None if cal is None else cal.e_process_construction_id),
        "lcb_descriptive": effect.lcb_descriptive,
        "lcb_label": "descriptive",
    }

    performance = {
        "raw_samples": (_canonicalizable(effect.raw_samples, "raw_samples")
                        if effect is not None else []),
        "raw_samples_ref": effect.raw_samples_ref if effect is not None else window.raw_evidence_ref,
        "paired_blocks": effect.paired_blocks if effect is not None else 0,
        "estimate": effect.value if effect is not None else None,
        "delta_display": None if effect is None else effect.delta_display(),
        "uncertainty": uncertainty,
        "search_discipline": discipline,
    }

    evaluator = request.evaluator.to_dict()

    event = {
        "schema": schemas.SCHEMA_EVALUATION_EVENT,
        "event_id": request.event_id,
        "campaign_id": request.campaign_id,
        "candidate_id": request.candidate_id,
        "tier": request.tier,
        "backend": request.backend,
        "device_state": (None if request.device_state is None
                         else request.device_state.to_dict()),
        "change_class": request.change_class,
        "anchor_tier": request.anchor_tier,
        "transfer_ratio_to": [row.to_dict() for row in request.transfer_ratio_to],
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": request.protocol_id,
            "metric": request.metric,
            "metric_direction": request.metric_direction,
            "reps": request.reps,
            "attestation_ref": compose_attestation_ref(window, request.evaluator),
        },
        "evaluator": evaluator,
        "artifact": request.artifact.to_dict(),
        # OMITTED, never null and never a placeholder, when the run had no anchor.
        **({} if anchor_block is None else {"anchor": anchor_block}),
        "scope_manifest_sha256": request.scope_manifest_sha256,
        "host_receipt": window.host_receipt,
        "resource_claim_receipt": window.resource_claim_receipt,
        "co_residency": request.co_residency,
        "correctness": correctness,
        "quality": quality,
        "stability": stability,
        "mechanism": mechanism,
        "scope_denominator": request.scope_denominator.to_dict(),
        "determinism": request.determinism.to_dict(),
        "performance": performance,
        "integrity_flags": list(verdict.integrity_flags),
        "status": verdict.status,
        "supersedes": [],
        "created_at": request.created_at,
    }
    return event


# =============================================================================
# Seams — the interfaces other AK3 / AK5 modules plug into
# =============================================================================

class TierGateRunner(Protocol):
    """Runs one tier's gates and returns their results. It, not this module, may
    launch builds, op suites, microbenches and profilers — under a held claim."""

    tier: str

    def run_gates(self, request: EvaluationRequest) -> Sequence[GateResult]:
        ...


class EffectReducer(Protocol):
    """The e-process reducer seam. Implemented by `statistics.PairedBlockReducer`.

    Produces the `EffectEstimate` from paired blocks. The construction (its
    supermartingale or betting form, its reducer, its resampling method) is a
    property of the evaluator bundle, fixed at the bundle hash; a campaign selects
    among constructions the bundle already implements and records which one —
    see `E_PROCESS_CONSTRUCTION_IDS`, which `CalibrationOutputs` enforces.

    **`None` is NOT a legal answer for a non-conforming run**, and the return
    annotation is `Optional` only because a caller that has no rate comparison to
    make passes no reducer at all. `TierDispatcher.dispatch` reads `effect is
    None` as *"this record is not a rate comparison"* and then skips the rate-only
    void conditions and the rate-only search-grade conjuncts — so a reducer that
    answered `None` for a strata violation or an order-control violation would
    suppress the very void that violation must raise. A conforming reducer either
    returns a conforming `EffectEstimate` or RAISES, carrying the full reduction
    on the exception so the run is still journalable as `INVALID` with its reason
    (`statistics.ReductionInadmissible`).
    """

    construction_id: str

    def reduce_blocks(self, request: EvaluationRequest,
                      blocks: Sequence[Any]) -> Optional[EffectEstimate]:
        ...


class RecipeConstructor(Protocol):
    """The codified-recipe seam (AK3, separate task). Emits argv and its receipt.

    Precondition 6: hand-typed argv voids the run. This evaluator never builds a
    command line; it only checks that a `RecipeReceipt` exists.
    """

    constructor_id: str

    def construct(self, request: EvaluationRequest) -> tuple:
        ...


class ReleaseTierEvaluator(Protocol):
    """The T3/T4 seam. NOT implemented here and never called from here.

    T3 is the kernel-freeze gate and T4 the post-cutover watch; both are release
    instruments outside P-AK-SEARCH-1's scope, owned by AK5 and governed by the
    release protocols. `admit_tier()` refuses those tiers so a release-shaped
    decision can never be produced under a search protocol.
    """

    def evaluate_release(self, request: Any) -> Any:
        ...


# =============================================================================
# Tier dispatch — an explicit state machine
# =============================================================================

@dataclass(frozen=True)
class EvaluationOutcome:
    """Everything one dispatch produced. `durable_payload` is ALWAYS present."""

    verdict: Verdict
    states: tuple
    preconditions: PreconditionScan
    void_scan: VoidScan
    grammar_line: str
    grammar_complete: schemas.Check
    durable_payload: dict
    event: Optional[dict] = None
    event_violations: tuple = ()
    event_blocked_reason: Optional[str] = None
    record_content_hash: Optional[str] = None

    @property
    def emitted(self) -> bool:
        return self.event is not None


class TierDispatcher:
    """Dispatch one evaluation through T0 / T1 / T2 with an explicit state machine.

    States: `CREATED -> TIER_ADMITTED -> WINDOW_OPENED -> PRECONDITIONS_CHECKED ->
    ANCHOR_BOUND -> GATES_RUN -> WINDOW_CLOSED -> VERDICT_COMPUTED -> EMITTED`,
    plus the terminal `REFUSED` for a release tier. There is deliberately no VOID
    terminal state: a voided window walks the same path and lands on a computed
    INVALID verdict with its reason, because *"A voided run is journaled as
    INVALID with its reason, and is never silently discarded."*

    The dispatcher owns NO runners of its own. A tier with no registered runner
    raises `EvaluatorNotWired` rather than producing an empty gate list, because
    an empty gate list derives to PASS and that is a fail-open verdict.
    """

    def __init__(self, *, gate_runners: Mapping[str, Any]) -> None:
        if not isinstance(gate_runners, Mapping):
            raise TypeError("gate_runners must be a mapping of tier -> TierGateRunner")
        for tier in gate_runners:
            admit_tier(tier)  # raises TierNotOwned on T3/T4 at WIRING time
        for tier, runner in gate_runners.items():
            if not hasattr(runner, "run_gates"):
                raise TypeError(f"gate runner for {tier!r} has no run_gates(request)")
        self._runners = dict(gate_runners)

    @property
    def tiers(self) -> tuple:
        return tuple(sorted(self._runners))

    @staticmethod
    def _advance(states: list, target: str) -> None:
        current = states[-1]
        if target not in _TRANSITIONS[current]:
            raise StateMachineViolation(
                f"illegal dispatch transition {current} -> {target}; legal targets are "
                f"{list(_TRANSITIONS[current])}"
            )
        states.append(target)

    def dispatch(self,
                 request: EvaluationRequest,
                 window: WindowAttestations,
                 *,
                 effect: Optional[EffectEstimate] = None) -> EvaluationOutcome:
        """Run the state machine and return the outcome. Never raises on a bad run.

        It DOES raise on a bad *wiring* — an unowned tier, a missing runner, a
        runner that returns something other than `GateResult`s — because those are
        defects in the evaluator, not findings about the candidate.
        """
        states = ["CREATED"]
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        if not isinstance(window, WindowAttestations):
            raise TypeError("window must be a WindowAttestations")

        try:
            admit_tier(request.tier)
        except TierNotOwned:
            self._advance(states, "REFUSED")
            raise

        self._advance(states, "TIER_ADMITTED")
        runner = self._runners.get(request.tier)
        if runner is None:
            raise EvaluatorNotWired(
                f"no gate runner registered for tier {request.tier!r}; registered tiers "
                f"are {list(self.tiers)}. There is no default runner: an unrun tier with "
                f"no gate results would derive to PASS."
            )

        self._advance(states, "WINDOW_OPENED")
        self._advance(states, "PRECONDITIONS_CHECKED")
        preconditions = check_preconditions(request, window)

        self._advance(states, "ANCHOR_BOUND")
        anchor = request.anchor

        self._advance(states, "GATES_RUN")
        gates = runner.run_gates(request)
        if not isinstance(gates, (list, tuple)):
            raise TypeError(
                f"runner for tier {request.tier!r} returned {type(gates).__name__}; "
                "expected a sequence of GateResult")
        gates = tuple(gates)
        if not gates:
            # The same fail-open the unregistered-tier branch above refuses, one
            # step later: `_derive` walks the gate list and a list with nothing
            # in it contributes nothing to worsen, so zero gates derive to PASS.
            # A runner that produced no result has not evaluated the candidate,
            # and "did not evaluate" is never a pass.
            raise EvaluatorNotWired(
                f"the gate runner registered for tier {request.tier!r} "
                f"({type(runner).__name__}) returned zero gate results. An empty gate "
                "list derives to PASS, so it is refused here for the same reason a "
                "missing runner is: a tier that produced no findings because it ran "
                "nothing is not a tier that found nothing."
            )

        self._advance(states, "WINDOW_CLOSED")
        rate_comparison = effect is not None
        void_scan = check_void_conditions(request, window, rate_comparison=rate_comparison)
        grammar_complete = check_record_grammar_complete(
            request=request, window=window, effect=effect)
        search_grade = evaluate_search_grade(
            request=request, window=window, preconditions=preconditions,
            effect=effect, grammar_complete=grammar_complete)

        self._advance(states, "VERDICT_COMPUTED")
        verdict = compute_verdict(
            tier=request.tier, gates=gates, void_scan=void_scan,
            search_grade=search_grade, anchor=anchor, effect=effect)

        self._advance(states, "EMITTED")
        grammar_line = render_search_record_grammar(
            request=request, window=window, verdict=verdict, effect=effect)

        event: Optional[dict] = None
        violations: tuple = ()
        blocked: Optional[str] = None
        content_hash: Optional[str] = None
        try:
            event = build_evaluation_event(
                request=request, window=window, verdict=verdict, effect=effect,
                preconditions=preconditions)
        except AnchorMissing as exc:
            blocked = str(exc)
        else:
            violations = tuple(schemas.validate_evaluation_event(event))
            content_hash = schemas.content_hash(event)

        durable = {
            "record_class": RECORD_CLASS,
            "protocol_id": PROTOCOL_VERSIONED_ID,
            "protocol_ratified_utc": PROTOCOL_RATIFIED_UTC,
            "dispatch_states": list(states),
            "tier": request.tier,
            "event_id": request.event_id,
            "campaign_id": request.campaign_id,
            "candidate_id": request.candidate_id,
            "backend": request.backend,
            "phase": request.phase,
            "cell_class": request.cell_class,
            "grammar_line": grammar_line,
            "grammar_complete": {"outcome": grammar_complete.outcome,
                                 "reasons": list(grammar_complete.reasons)},
            "preconditions": preconditions.to_dict(),
            "void_scan": void_scan.to_dict(),
            "verdict": verdict.to_dict(),
            "event_emitted": event is not None,
            "event_blocked_reason": blocked,
            "event_violations": list(violations),
            "record_content_hash": content_hash,
        }

        return EvaluationOutcome(
            verdict=verdict,
            states=tuple(states),
            preconditions=preconditions,
            void_scan=void_scan,
            grammar_line=grammar_line,
            grammar_complete=grammar_complete,
            durable_payload=durable,
            event=event,
            event_violations=violations,
            event_blocked_reason=blocked,
            record_content_hash=content_hash,
        )


# =============================================================================
# Self-audit — "no write path outside its own output root", proved not promised
# =============================================================================

#: Bare-name calls that can create or destroy state, or execute code.
_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input"})

#: Attribute calls that write, delete, move, execute, or signal. The check is
#: blunt on purpose — it does not try to prove the receiver's type, so this module
#: simply does not use these method names on anything.
_FORBIDDEN_CALL_ATTRS = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "flush", "fsync",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "rmtree", "rename", "chmod",
    "chown", "utime", "symlink", "link", "touch", "move", "copy", "copyfile", "copytree",
    "system", "popen", "Popen", "spawnv", "fork", "kill", "killpg", "send_signal",
    "terminate", "check_call", "check_output", "communicate", "setxattr",
})

#: Modules whose mere import would give this file the ability to write or signal.
_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "shlex", "asyncio",
})


def _defines_module(tree: ast.AST, module_id: str) -> bool:
    """True when parsed source binds ``MODULE_ID`` to the expected identity.

    Copied from `release/packager._defines_this_module`, for the same reason it
    exists there: a self-audit that does not prove it read its OWN module is a
    clean bill of health for whatever text it was handed.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODULE_ID" and \
                        isinstance(node.value, ast.Constant) and \
                        node.value.value == module_id:
                    return True
    return False


def _is_an_audited_module(tree: ast.AST) -> bool:
    """True when the parsed source has a module BODY — something to audit.

    `""`, whitespace, a comment, a lone docstring and `x = 1` all contain no
    forbidden construct, so a search for forbidden constructs certified every one
    of them. That is the guarantee obtained by DELETING the thing under
    inspection, and it is the one shape of this call that is always wrong.
    """
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for node in getattr(tree, "body", ()))


#: A clean result over source the caller chose is not evidence about this module.
_NOT_A_MODULE = (
    "the audited source defines no function or class, so there was nothing to audit. A "
    "PASS here would be the guarantee obtained by deleting the thing it inspects")
_NOT_THIS_MODULE = (
    "the audited source does not define MODULE_ID = {module_id!r}, so the AST parsed is "
    "not this module's. Call with no argument to audit this module")


def audit_no_write_or_process_paths(source: Optional[str] = None, *,
                                    module_id: Optional[str] = None) -> schemas.Check:
    """Prove from an AST that it cannot write or signal. No argument = THIS module.

    Design §5.4: the trusted runner *"has no authority to modify candidate source
    or production state."* Prose cannot enforce that, and neither can a code
    review that happens once. This parses the module and FAILs if it finds a
    write-capable call, a process call, or an import of a module that would grant
    either. `test_api.py` asserts the result is PASS, so the property becomes a
    regression barrier rather than an intention.

    TWO CALLS, and the difference is what the PASS means:

      * **No argument.** The self-audit. The text is read from `__file__` and then
        BOUND with `_defines_this_module` — the assumption "we read our own file"
        is checked rather than trusted. This is the call `test_conformance.py` and
        `evaluator/__init__.py` mean by "the evaluator proves it cannot write".
      * **A supplied `source`.** The shared ENGINE. Several modules reuse this
        rather than copy the denylists — one definition of "cannot write, cannot
        signal" for the package, which is why `controls.py` and `correctness.py`
        delegate here instead of re-typing the tables. A result over supplied text
        is a finding about THAT TEXT, and the caller is the one that binds it to a
        module by passing its expected ``module_id``. Supplied source without
        that binding can find a violation, but can never receive PASS.

    `audit_no_write_or_process_paths("")` used to return PASS in both readings.
    It no longer does: source with no function or class in it is COULD_NOT_CHECK,
    because a search for forbidden constructs over no constructs certifies
    nothing. A FAIL is still returned unbound — a forbidden construct is a finding
    about the text whoever wrote it.

    COULD_NOT_CHECK when the source cannot be read or parsed — an unreadable
    module is not an audited one.
    """
    own = source is None
    if own:
        if module_id is not None:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "module_id is only accepted with supplied source; the no-argument audit "
                "binds itself to api.MODULE_ID",))
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module: {exc}",))
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
        # Returned UNBOUND and before the module-shape test: a forbidden construct
        # is a finding about the text, whoever wrote it and however small it is.
        return schemas.Check(schemas.FAIL, tuple(findings))
    if not _is_an_audited_module(tree):
        return schemas.Check(schemas.COULD_NOT_CHECK, (_NOT_A_MODULE,))
    expected_id = MODULE_ID if own else module_id
    if not isinstance(expected_id, str) or not expected_id.strip():
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "supplied source has no expected module_id binding; a clean AST alone cannot "
            "prove which module was audited",))
    if not _defines_module(tree, expected_id):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (_NOT_THIS_MODULE.format(module_id=expected_id),))
    return schemas.Check(schemas.PASS)
