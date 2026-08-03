"""critic.py — the AK4 pre-run and post-run critics (design §6.3, §8.4, §8.8).

WHY THIS MODULE EXISTS
----------------------
The critic is the only component whose job is to be WRONG about the planner, and
the only one a plausible-sounding model could turn into a rubber stamp. Its
authority is therefore deliberately asymmetric, and that asymmetry is the whole
module:

  * **The critic may reject or revise. It can NEVER waive an evaluator gate.**
    (§6.3.) Nothing here mutates a `Verdict`, nothing here constructs one — the
    evaluator's `Verdict` re-derives its own status from its own gates and raises
    `VerdictTampering` on any disagreement, so a waiver would have to arrive as a
    *field*. `find_gate_waiver_keys()` refuses those fields anywhere in a critic
    payload, and `apply_pre_run_verdict()` refuses a revision that shrinks
    `evaluation_plan.required_t0/required_t1`. A critic that could remove a
    required gate would be an actor grading itself, which §4 invariant 4 and
    P-AK-SEARCH-1 denial 6 both forbid.

  * **The post-run critic INTERPRETS; it does not DECLARE** (§8.8: *"The
    deterministic controller checks the classification against the raw gates"*;
    AK-D4). `reconcile_classification()` derives, from the evaluator's raw gate
    results alone, the set of classifications those gates ADMIT, and refuses any
    other. `classify_run()` raises rather than returning an unreconciled
    interpretation, so nothing downstream can consume one by forgetting to look.

  * **A wrong suppression silently closes a research family** (§12, §19.3). A
    do-not-repeat entry blocks a proposal only when it carries a receipt that
    PARSES — a commit:path:line locator or an artifact hash — is bound to a
    production commit, and is not `conflicted`. Confident prose blocks nothing.
    This is the direction most likely to be got backwards: the safe-looking
    default (block on any recorded negative) is the one that quietly ends
    research directions nobody will re-open.

  * **Authorship is not evidence** (§8.4.0, AK-D38). The deterministic gates
    never read `hypothesis_origin`. An operator hypothesis that repeats a
    receipted negative is rejected exactly as a controller-authored one is, and
    the only thing origin controls is a CEILING: `operator_hypothesis` may not
    carry an evidence grade above `design_prior`.

TWO MODELS, ON PURPOSE
----------------------
§6.3: *"Prefer a different provider/model from the planner; a critic sharing the
planner's blind spots mostly agrees."* `check_critic_independence()` makes that
configurable rather than assumed, and a campaign that deliberately shares one
model must say so in a recorded reason — never by silence.

ORDER OF WORK
-------------
Deterministic gates run BEFORE the metered critic call, and a deterministic
rejection can skip the call entirely (§8.4: *"Cheap deterministic checks run
before metered drafting, not after — the reverse ordering cost roughly 38
draft-and-critique cycles that were paid for and then thrown away"*).

This module runs NO inference of its own, opens NO socket, launches NO process,
and writes NO file; `planner.audit_no_provider_side_effects()` is run against
this file by the test suite.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §6.3, §6.4,
§6.5, §8.4, §8.4.0, §8.4.1, §8.8, §8.9, §12, §19.2, §19.3; governing instrument
`epyc-root/measurement/protocols/kernel-research.md` (P-AK-SEARCH-1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from .. import journal, schemas
from . import oracles
from ..evaluator import api as evaluator_api
from .planner import (
    GRADE_DESIGN_PRIOR,
    ORIGIN_OPERATOR_HYPOTHESIS,
    ROLE_POST_RUN_CRITIC,
    ROLE_PRE_RUN_CRITIC,
    Completion,
    ModelBinding,
    ModelRequest,
    PromptBundle,
    Provider,
    ProviderResponseInvalid,
    RealizedCost,
    ResponseContract,
    check_binding_honoured,
)

__all__ = [
    # errors
    "CriticError", "CriticIndependenceError", "GateWaiverAttempt",
    "ClassificationMismatch", "RevisionRefused",
    # oracle registry (§6.5)
    "HARVEST_CLASSES", "OracleRow", "ORACLE_REGISTRY", "oracle_row",
    # do-not-repeat ledger (§19.2, §19.3)
    "LEDGER_CLASSES", "SUPPRESSING_LEDGER_CLASSES", "LedgerEntry",
    "check_receipt", "LedgerDisposition", "evaluate_ledger",
    # facts
    "BudgetEnvelope", "ProposalFacts",
    # pre-run critic (§6.3)
    "PreRunQuestion", "PRE_RUN_QUESTIONS", "PRE_RUN_RESPONSE_CONTRACT",
    "PreRunGate", "PreRunAnswer", "PreRunCritique",
    "DISPOSITION_ACCEPT", "DISPOSITION_REVISE", "DISPOSITION_REJECT",
    "DISPOSITIONS", "evaluate_pre_run_gates", "critique_proposal",
    "apply_pre_run_verdict", "FORBIDDEN_REVISION_PATHS",
    # post-run critic (§8.8)
    "HYPOTHESIS_STATUSES", "HYPOTHESIS_KINDS", "MECHANISM_STATUSES",
    "SIGNAL_CLASSES", "CHAMPION_INTERACTIONS", "POST_RUN_RESPONSE_CONTRACT",
    "WallShareTranslation", "NextExperiment", "DurableLesson",
    "PostRunClassification", "PostRunCritique",
    "derive_signal_class", "admissible_hypothesis_statuses",
    "admissible_mechanism_statuses", "reconcile_classification", "classify_run",
    "lesson_journal_payload", "LESSON_JOURNAL_KIND", "critic_cost",
    # guards
    "check_critic_independence", "find_gate_waiver_keys",
]


# =============================================================================
# Errors
# =============================================================================

class CriticError(Exception):
    """Base for every refusal this module raises."""


class CriticIndependenceError(CriticError):
    """The critic binding is the planner binding and no reason was declared (§6.3)."""


class GateWaiverAttempt(CriticError):
    """A critic payload carried a field shaped like a gate waiver.

    Refused rather than ignored. An ignored waiver field is indistinguishable
    from an honoured one to anybody reading the record later, and the attempt is
    itself evidence worth journaling.
    """

    def __init__(self, paths: Sequence[str]) -> None:
        super().__init__(
            "critic payload carries waiver-flavoured key(s) "
            f"{sorted(paths)}: the critic may reject or revise; it can never waive "
            "an evaluator gate (§6.3, §4 invariant 4)"
        )
        self.paths = tuple(paths)


class ClassificationMismatch(CriticError):
    """The post-run classification does not follow from the raw gate results.

    Carries the failed `Check` and the classification so the caller journals both
    — a refused interpretation is durable evidence (invariant 7), and the pair is
    what makes a mis-calibrated critic visible instead of merely absent.
    """

    def __init__(self, check: schemas.Check, classification: Any) -> None:
        super().__init__(
            "post-run classification contradicts the raw gates: "
            + "; ".join(check.reasons)
        )
        self.check = check
        self.classification = classification


class RevisionRefused(CriticError):
    """A critic revision touched something the critic does not own."""


# =============================================================================
# §6.5 Oracle registry — declared, read-only reference implementations
# =============================================================================

#: The axis is ARCHITECTURAL PORTABILITY, not licensing (AK-D34). Standing policy
#: is open-source self-hosted, non-commercial, licences not blockers; what decides
#: the cost of a harvest is whether the artifact runs on gfx90a or EPYC.
#:
#: Sourced from `oracles.py`, not restated. This module had three classes and
#: `context.py` had four, so §6.5's FlashAttention/FlashInfer row
#: (*"portable_source where a HIP path exists, else reimplement"* = `conditional`)
#: was inexpressible here and the two planes disagreed about what a legal class is.
HARVEST_CLASSES = oracles.HARVEST_CLASSES


@dataclass(frozen=True)
class OracleRow:
    """One declared oracle. A RETIRED row stays visible with its correction.

    §6.5 keeps the AITER row *"rather than deleting it: the row was wrong, and a
    future reader reaching for AMD's inference kernels should meet the correction
    rather than re-add it"*. Deleting a retired row is how the same wrong entry
    gets re-derived; `retirement_note` is therefore mandatory when `retired`.
    """

    oracle_id: str
    harvest_class: str
    why: str
    retired: bool = False
    retirement_note: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.oracle_id.strip():
            raise ValueError("OracleRow.oracle_id must be non-empty")
        if self.harvest_class not in HARVEST_CLASSES:
            raise ValueError(f"OracleRow.harvest_class: {self.harvest_class!r} not in "
                             f"{list(HARVEST_CLASSES)}")
        if self.retired and not (self.retirement_note or "").strip():
            raise ValueError(
                f"OracleRow {self.oracle_id!r}: a retired row must carry the "
                "correction that retired it (§6.5)"
            )


def _row_for(oracle_id: str, fact: oracles.OracleFact) -> OracleRow:
    return OracleRow(
        oracle_id=oracle_id,
        harvest_class=fact.harvest_class,
        why=fact.why,
        retired=fact.retired,
        retirement_note=fact.retirement_note(),
    )


#: Every id §6.5 makes nameable: the table ROW (what `context.py` renders into the
#: planner brief) and each member TREE (what a port actually names). Both are here
#: because the planner cites what it was shown and this gate rejects what it
#: cannot resolve — and before the AK4 integration pass the two id sets
#: intersected in exactly ONE oracle out of nineteen, so a planner that cited its
#: own context was told *"not in the declared registry"*.
ORACLE_REGISTRY = tuple(
    _row_for(oracle_id, fact)
    for fact in oracles.REGISTRY
    for oracle_id in fact.ids()
)

_ORACLE_BY_ID = {row.oracle_id: row for row in ORACLE_REGISTRY}


def oracle_row(oracle_id: Any) -> Optional[OracleRow]:
    """Look up a declared oracle. `None` means "not in the registry", which is a
    REJECT for an `oracle_port`: new oracles enter through `research-intake`, not
    by an agent adding a row (AK-D34)."""
    return _ORACLE_BY_ID.get(oracle_id) if isinstance(oracle_id, str) else None


# =============================================================================
# §19.2 do-not-repeat ledger, §19.3 receipt rule
# =============================================================================

LEDGER_CLASSES = (
    "HARD_CONSTRAINT", "MATCHED_NEGATIVE", "CONDITIONAL_NEGATIVE",
    "CONFOUNDED_RESULT", "SUPERSEDED_FACT", "LOW_VALUE",
)

#: The three §19.3 names a receipt is mandatory for. The other three do not
#: suppress a family: `CONDITIONAL_NEGATIVE` excludes matched cells,
#: `CONFOUNDED_RESULT` demands a repaired experiment, `LOW_VALUE` deprioritizes.
SUPPRESSING_LEDGER_CLASSES = frozenset({
    "HARD_CONSTRAINT", "MATCHED_NEGATIVE", "SUPERSEDED_FACT",
})

#: A receipt is a LOCATOR, not a sentence: `<40-hex commit>:<path>:<line[-line]>`
#: or an artifact hash. §19.3 — *"a source receipt … not a confident sentence"*.
_RECEIPT_COMMIT_PATH_LINE = re.compile(r"^[0-9a-f]{40}:\S+:\d+(-\d+)?$")
_RECEIPT_ARTIFACT_HASH = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

#: §19.3: *"Required evidence grade scales with breadth"* — a family-wide
#: suppression needs `source_verified` or `protocol_bound`; a single-cell
#: exclusion may rest on an observation.
_FAMILY_WIDE_GRADES = frozenset({"source_verified", "protocol_bound"})


def check_receipt(value: Any) -> schemas.Check:
    """Does this receipt RESOLVE to something a reader could go and look at?

    PASS for a commit:path:line locator or an artifact hash. FAIL for prose, an
    empty string, or a bare commit with nothing to find in it. This is the
    difference between a suppression that can be re-verified on anchor move and
    one that can only be believed.
    """
    if not isinstance(value, str) or not value.strip():
        return schemas.Check(schemas.FAIL, ("receipt: missing or empty",))
    text = value.strip()
    if _RECEIPT_COMMIT_PATH_LINE.match(text) or _RECEIPT_ARTIFACT_HASH.match(text):
        # A well-formed digest that no measurement produced is the WORST case, not
        # an edge case: it satisfies the regex, reads as evidence, and closes a
        # research family nobody will re-open. `planner.resolve_context_binding()`
        # already refuses these for the context hash; a suppression receipt is the
        # higher-stakes of the two. (§19.3; `schemas.is_placeholder_digest`.)
        digest = text.split(":")[0] if _RECEIPT_COMMIT_PATH_LINE.match(text) \
            else text.split(":")[-1]
        if schemas.is_placeholder_digest(digest):
            return schemas.Check(schemas.FAIL, (
                f"receipt {text!r} resolves to placeholder digest {digest!r}, which no "
                "measurement produced. A fabricated receipt is a CLAIM that a "
                "suppression was verified — an absent receipt is loud, a fabricated "
                "one is silent and closes a family forever (§19.3)",
            ))
        return schemas.Check(schemas.PASS)
    return schemas.Check(schemas.FAIL, (
        f"receipt {text!r} is not a resolvable locator; §19.3 requires "
        "'<40-hex commit>:<path>:<line>' or an artifact sha256, not a confident "
        "sentence — a suppression nobody can re-verify is never re-tested",
    ))


@dataclass(frozen=True)
class LedgerEntry:
    """One do-not-repeat / constraint entry (§19.2).

    `conflicted` is a first-class state, not an error: §19.2 requires
    contradiction detection against live operator decisions and sibling entries,
    and *"anything that contradicts either becomes `conflicted` and is never
    authoritative"*. A conflicted entry is carried, reported, and blocks nothing.
    """

    entry_id: str
    ledger_class: str
    statement: str
    match_dimensions: Mapping[str, Any]
    reopen_when: str
    receipt: Optional[str] = None
    verified_against_commit: Optional[str] = None
    evidence_grade: str = "observation"
    scope: str = "cell"          # "cell" | "family"
    conflicted: bool = False
    reopen_satisfied: bool = False

    def __post_init__(self) -> None:
        if self.ledger_class not in LEDGER_CLASSES:
            raise ValueError(f"LedgerEntry.ledger_class: {self.ledger_class!r} not in "
                             f"{list(LEDGER_CLASSES)}")
        if not isinstance(self.match_dimensions, Mapping) or not self.match_dimensions:
            raise ValueError(
                "LedgerEntry.match_dimensions: required and non-empty — "
                "'do not repeat' without regime identity is dangerous because this "
                "project repeatedly observes sign changes across architecture, "
                "substrate, batch, context, and quant (§19.2)"
            )
        if not isinstance(self.reopen_when, str) or not self.reopen_when.strip():
            raise ValueError("LedgerEntry.reopen_when: required and non-empty (§19.2)")
        if self.scope not in ("cell", "family"):
            raise ValueError("LedgerEntry.scope must be 'cell' or 'family'")

    def authority(self) -> schemas.Check:
        """Is this entry allowed to suppress anything at all? (§19.3.)

        FAIL means the entry is carried but toothless. That is the deliberate
        direction: a suppression that cannot be re-verified must not be able to
        close a research family, because nothing will ever test it again.
        """
        if self.conflicted:
            return schemas.Check(schemas.FAIL, (
                f"{self.entry_id}: entry is `conflicted` (§19.2 contradiction "
                "detection) and is never authoritative",
            ))
        if self.ledger_class not in SUPPRESSING_LEDGER_CLASSES:
            return schemas.Check(schemas.PASS)
        reasons: list = []
        receipt = check_receipt(self.receipt)
        if receipt.outcome != schemas.PASS:
            reasons.extend(f"{self.entry_id}: {r}" for r in receipt.reasons)
        if not isinstance(self.verified_against_commit, str) or \
                not _COMMIT_RE.match(self.verified_against_commit or "") or \
                schemas.is_placeholder_digest(self.verified_against_commit):
            reasons.append(
                f"{self.entry_id}: verified_against_commit must be a full 40-hex "
                "production commit that is not a placeholder — §19.3 binds every "
                "suppression to the commit it was verified against, so an anchor move "
                "can re-check it, and '0'*40 re-checks against nothing"
            )
        if self.scope == "family" and self.evidence_grade not in _FAMILY_WIDE_GRADES:
            reasons.append(
                f"{self.entry_id}: a family-wide suppression needs evidence grade in "
                f"{sorted(_FAMILY_WIDE_GRADES)}, got {self.evidence_grade!r} (§19.3, "
                "breadth scaling)"
            )
        if reasons:
            return schemas.Check(schemas.FAIL, tuple(reasons))
        return schemas.Check(schemas.PASS)


@dataclass(frozen=True)
class LedgerDisposition:
    """What the ledger says about ONE proposal. `blocking` is the only verdict."""

    blocking: tuple
    excluded_cells: tuple
    advisory: tuple
    toothless: tuple

    def to_dict(self) -> dict:
        return {
            "blocking": [e.entry_id for e in self.blocking],
            "excluded_cells": [e.entry_id for e in self.excluded_cells],
            "advisory": [e.entry_id for e in self.advisory],
            "toothless": [e.entry_id for e in self.toothless],
        }


def evaluate_ledger(matches: Sequence[LedgerEntry]) -> LedgerDisposition:
    """Apply the §19.2 planner-behaviour column. Deterministic; origin-blind.

    The table, verbatim in behaviour:
      * `HARD_CONSTRAINT` — reject matching proposal;
      * `MATCHED_NEGATIVE` — reject unless an explicit reopen predicate is newly
        satisfied;
      * `CONDITIONAL_NEGATIVE` — exclude matched cells, other regimes eligible;
      * `CONFOUNDED_RESULT` — do not learn its sign, require a repaired experiment;
      * `SUPERSEDED_FACT` — do not execute the stale proposal;
      * `LOW_VALUE` — deprioritize.

    Every suppressing entry first has to pass `authority()`; one that does not is
    reported in `toothless` and blocks nothing.
    """
    blocking: list = []
    excluded: list = []
    advisory: list = []
    toothless: list = []
    for entry in matches:
        if not isinstance(entry, LedgerEntry):
            raise TypeError(f"ledger match must be a LedgerEntry, got "
                            f"{type(entry).__name__}")
        if entry.ledger_class in SUPPRESSING_LEDGER_CLASSES or entry.conflicted:
            if entry.authority().outcome != schemas.PASS:
                toothless.append(entry)
                continue
        if entry.ledger_class in ("HARD_CONSTRAINT", "SUPERSEDED_FACT"):
            blocking.append(entry)
        elif entry.ledger_class == "MATCHED_NEGATIVE":
            (advisory if entry.reopen_satisfied else blocking).append(entry)
        elif entry.ledger_class == "CONDITIONAL_NEGATIVE":
            excluded.append(entry)
        else:  # CONFOUNDED_RESULT, LOW_VALUE
            advisory.append(entry)
    return LedgerDisposition(
        blocking=tuple(blocking), excluded_cells=tuple(excluded),
        advisory=tuple(advisory), toothless=tuple(toothless),
    )


# =============================================================================
# The deterministic facts a gate reads. None of these come from a model.
# =============================================================================

@dataclass(frozen=True)
class BudgetEnvelope:
    """What is left. §8.4 rejects a proposal whose estimate exceeds it."""

    minutes_remaining: float
    storage_gb_remaining: float
    candidates_remaining: int
    controller_tokens_remaining: int

    def __post_init__(self) -> None:
        for name in ("minutes_remaining", "storage_gb_remaining"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"BudgetEnvelope.{name} must be a non-negative number")
        for name in ("candidates_remaining", "controller_tokens_remaining"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"BudgetEnvelope.{name} must be a non-negative int")


@dataclass(frozen=True)
class ProposalFacts:
    """Everything the deterministic gates consume, supplied by the CONTROLLER.

    Two fields are load-bearing in a way that is easy to get backwards:

      * `derived_affected_surface` is the MECHANICALLY derived surface (§6.4), and
        it is what the oracle-coverage gate reads. The proposal's own
        `change.predicted_affected_surface` is a scored prediction and is never a
        scope input (invariant 18) — using it here would let the actor set its own
        release scope, which is the exact §12 row this separation answers.

      * `confirmation_shapes` lives HERE and never in `planner.ContextManifest`.
        P-AK-SEARCH-1 forbids confirmation-stratum contents from appearing in
        planner context AND rejects a proposal that targets a confirmation shape;
        the controller therefore has to know them while the planner must not.

    `roofline_utilisation` is carried for the critic's prose context and is read
    by NO gate. AK-D35: utilisation is *"a diagnostic and a routing input, never a
    gate"*, and `test_planner_critic` asserts that dropping it changes no gate.
    """

    derived_affected_surface: tuple = ()
    correctness_oracles_by_surface: Mapping[str, Sequence[str]] = field(default_factory=dict)
    real_graph_shapes: frozenset = frozenset()
    confirmation_shapes: frozenset = frozenset()
    wall_share_receipts: frozenset = frozenset()
    microkernel_only: bool = False
    architectural_campaign: bool = False
    lineage_steps: tuple = ()
    lineage_end_state: str = ""
    lineage_step_index: Optional[int] = None
    fusion_explanation: str = ""
    prospective_shapes: Mapping[str, Any] = field(default_factory=dict)
    ledger_matches: tuple = ()
    existing_dispatch_receipts: tuple = ()
    oracle_coverage: tuple = ()
    backend_owned_domains: frozenset = frozenset()
    proposal_domains: frozenset = frozenset()
    evaluator_change_required: bool = False
    budget: Optional[BudgetEnvelope] = None
    surface_reconciled: schemas.Check = field(
        default_factory=lambda: schemas.Check(schemas.COULD_NOT_CHECK,
                                              ("no reconciliation supplied",)))
    capability_objective_met: Optional[bool] = None
    roofline_utilisation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_reconciled, schemas.Check):
            raise TypeError("ProposalFacts.surface_reconciled must be a schemas.Check")
        if self.budget is not None and not isinstance(self.budget, BudgetEnvelope):
            raise TypeError("ProposalFacts.budget must be a BudgetEnvelope or None")
        for entry in self.ledger_matches:
            if not isinstance(entry, LedgerEntry):
                raise TypeError("ProposalFacts.ledger_matches must hold LedgerEntry values")


# =============================================================================
# §6.3 PRE_RUN_CRITIC — the ten structured questions
# =============================================================================

DISPOSITION_ACCEPT = "accept"
DISPOSITION_REVISE = "revise"
DISPOSITION_REJECT = "reject"
DISPOSITIONS = (DISPOSITION_ACCEPT, DISPOSITION_REVISE, DISPOSITION_REJECT)
_DISPOSITION_SEVERITY = {DISPOSITION_ACCEPT: 0, DISPOSITION_REVISE: 1,
                         DISPOSITION_REJECT: 2}


@dataclass(frozen=True)
class PreRunQuestion:
    """One §6.3 question, with its POLARITY stated.

    `pass_means` exists because the design's phrasing is not uniformly oriented —
    *"Is the hypothesis falsifiable?"* wants yes, *"Is a faster-but-wrong path
    plausible?"* wants no. Leaving that implicit is a defect waiting to happen, so
    every answer is a `schemas.Check` where PASS uniformly means "this axis does
    not block", and `pass_means` says what that is for this question.
    """

    qid: str
    design_question: str
    pass_means: str
    blocking: bool
    on_could_not_check: str      # "reject" | "advisory"

    def __post_init__(self) -> None:
        if self.on_could_not_check not in ("reject", "advisory"):
            raise ValueError("on_could_not_check must be 'reject' or 'advisory'")


PRE_RUN_QUESTIONS = (
    PreRunQuestion(
        "falsifiable", "Is the hypothesis falsifiable?",
        "the hypothesis names an observation that would refute it", True, "reject"),
    PreRunQuestion(
        "measurement_discriminates",
        "Does the proposed measurement distinguish the claimed mechanism from "
        "alternatives?",
        "the measurement separates the claimed mechanism from at least one named "
        "alternative", True, "reject"),
    PreRunQuestion(
        "shapes_identified",
        "Are exact target and non-target shapes identified?",
        "both sets are exact, not a description of a family", True, "reject"),
    PreRunQuestion(
        "faster_but_wrong",
        "Is a faster-but-wrong path plausible?",
        "no faster-but-wrong path is plausible, or the evaluation plan already "
        "catches the named one", True, "reject"),
    PreRunQuestion(
        "already_in_our_tree",
        "Does an existing dispatch/path in OUR tree already implement this?",
        "no existing path implements it, cited against source", True, "advisory"),
    PreRunQuestion(
        "oracle_already_implements",
        "Does a declared oracle already implement this, and is porting cheaper "
        "than authoring?",
        "either no oracle implements it, or the proposal is the port", True,
        "advisory"),
    PreRunQuestion(
        "one_conceptual_change",
        "Is the proposal actually one conceptual change?",
        "one conceptual change, or one STEP of a declared architectural lineage",
        True, "reject"),
    PreRunQuestion(
        "value_within_ceiling",
        "Can the claimed end-to-end value exceed the measured wall-share ceiling?",
        "the claim stays inside the measured ceiling, or an architectural campaign "
        "supplies a predicted post-change profile instead", True, "reject"),
    PreRunQuestion(
        "cost_proportional",
        "Is the resource cost proportional to expected information gain?",
        "the cost is proportional to the information the experiment buys", True,
        "advisory"),
    PreRunQuestion(
        "repeats_receipted_negative",
        "Does it repeat a recorded negative without new evidence — and does that "
        "negative carry a receipt?",
        "it repeats no negative that carries a receipt", True, "reject"),
)

_QUESTION_BY_ID = {q.qid: q for q in PRE_RUN_QUESTIONS}

PRE_RUN_RESPONSE_CONTRACT = ResponseContract(
    name="autokernel_pre_run_critique",
    required_keys=("answers", "disposition", "reasons"),
    optional_keys=("revisions", "notes"),
)


@dataclass(frozen=True)
class PreRunGate:
    """One deterministic pre-run gate. The model cannot move it."""

    gate_id: str
    check: schemas.Check
    blocking: bool = True

    def to_dict(self) -> dict:
        return {"gate_id": self.gate_id, "outcome": self.check.outcome,
                "reasons": list(self.check.reasons), "blocking": self.blocking}


@dataclass(frozen=True)
class PreRunAnswer:
    """The model's answer to one §6.3 question."""

    qid: str
    check: schemas.Check

    def to_dict(self) -> dict:
        return {"qid": self.qid, "outcome": self.check.outcome,
                "reasons": list(self.check.reasons)}


def _gate(gate_id: str, ok: bool, reasons: Sequence[str] = (),
          *, blocking: bool = True) -> PreRunGate:
    outcome = schemas.PASS if ok else schemas.FAIL
    return PreRunGate(gate_id, schemas.Check(outcome, tuple(reasons)), blocking)


def _cnc(gate_id: str, reasons: Sequence[str], *, blocking: bool = True) -> PreRunGate:
    return PreRunGate(gate_id, schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons)),
                      blocking)


def evaluate_pre_run_gates(manifest: Mapping[str, Any],
                           facts: ProposalFacts) -> tuple:
    """The §8.4 rejection conditions, computed from records. No model is consulted.

    §8.4.1 replaces three of them inside a declared architectural campaign — the
    wall-share ceiling by a predicted post-change profile, real-graph shapes by
    prospective shapes with a stated mechanism and observation, and one-conceptual-
    change by one change per declared lineage step. Replaced, never waived: each
    substitute is strictly more falsifiable than the condition it stands in for,
    and an architectural campaign that declares no lineage gets no substitution.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    if not isinstance(facts, ProposalFacts):
        raise TypeError("facts must be a ProposalFacts")

    gates: list = []

    violations = schemas.validate_proposal(manifest)
    gates.append(_gate("schema_valid", not violations, violations))
    if violations:
        # Every gate below reads fields the schema just said are wrong; running
        # them would report derived nonsense as evidence.
        return tuple(gates)

    # `manifest["change"]["predicted_affected_surface"]` is deliberately read by NO
    # gate below: it is a scored prediction and never a scope input (§6.4,
    # invariant 18). The oracle-coverage gate reads `facts.derived_affected_surface`.
    mechanism = dict(manifest["mechanism_prediction"])
    target = dict(manifest["target"])
    request = dict(manifest["resource_request"])
    origin_block = dict(manifest.get("hypothesis_origin") or {})

    # --- one conceptual change (invariant 13, §8.4.1) -----------------------
    if facts.architectural_campaign:
        reasons: list = []
        if not facts.lineage_steps:
            reasons.append("architectural campaign declares no lineage steps")
        if not facts.lineage_end_state.strip():
            reasons.append("architectural campaign declares no end-state")
        if facts.lineage_step_index is None or \
                not (0 <= facts.lineage_step_index < len(facts.lineage_steps)):
            reasons.append(
                f"this proposal does not name which lineage step it is "
                f"(step_index={facts.lineage_step_index!r}, "
                f"{len(facts.lineage_steps)} step(s) declared)")
        gates.append(_gate("one_conceptual_change", not reasons, reasons))
    else:
        gates.append(_gate(
            "one_conceptual_change", not facts.lineage_steps,
            ["a multi-step lineage was declared but the campaign is not an "
             "architectural campaign; invariant 13 binds each STEP only inside a "
             "declared lineage with a stated end-state (§8.4.1)"]
            if facts.lineage_steps else []))

    # --- wall-share ceiling (§8.4) / predicted post-change profile (§8.4.1) --
    ceiling = mechanism.get("expected_wall_share_ceiling")
    claimed = mechanism.get("expected_end_to_end_gain")
    receipt_id = mechanism.get("wall_share_receipt_id")
    if facts.architectural_campaign:
        profile = mechanism.get("predicted_post_change_profile")
        reasons = []
        if not isinstance(profile, Mapping) or not profile:
            reasons.append(
                "an architectural campaign replaces the wall-share ceiling with a "
                "PREDICTED POST-CHANGE PROFILE per op family; none is declared "
                "(§8.4.1). The substitution is not a waiver")
        elif not (target.get("ops") or []):
            # "Covers every target op" is satisfied by a proposal that declares no
            # target ops — the coverage check reduces to a loop over nothing. The
            # substitution §8.4.1 grants is a MORE falsifiable claim, and a profile
            # that predicts nothing about anything is less.
            reasons.append(
                "the proposal declares no target ops, so the predicted post-change "
                "profile covers nothing and cannot be wrong; §8.4.1 substitutes a "
                "profile PER OP FAMILY for the ceiling, which requires op families")
        else:
            missing = [op for op in (target.get("ops") or []) if op not in profile]
            if missing:
                reasons.append(
                    f"predicted post-change profile does not cover target op(s) "
                    f"{missing}; a profile that omits the ops it changes cannot be "
                    "wrong in a way the profiler can see")
        gates.append(_gate("wall_share_ceiling", not reasons, reasons))
    else:
        reasons = []
        if not isinstance(claimed, (int, float)) or isinstance(claimed, bool):
            reasons.append(
                "mechanism_prediction.expected_end_to_end_gain is missing; §7.2 "
                "rejects a proposal without a wall-share prediction before it "
                "consumes a benchmark window")
        elif float(claimed) > float(ceiling) and not facts.fusion_explanation.strip():
            reasons.append(
                f"claimed end-to-end gain {claimed} exceeds this change's own "
                f"measured wall-share ceiling {ceiling} and no fusion explanation is "
                "declared (§8.4)")
        if receipt_id not in facts.wall_share_receipts:
            reasons.append(
                f"wall_share_receipt_id {receipt_id!r} does not resolve to a measured "
                "wall-share receipt; an unreceipted ceiling is a number the proposal "
                "chose for itself")
        gates.append(_gate("wall_share_ceiling", not reasons, reasons))

    # --- correctness oracle coverage over the DERIVED surface (§6.4, §8.4) ---
    if not facts.derived_affected_surface:
        gates.append(_cnc("correctness_oracle_coverage", (
            "no mechanically derived affected surface was supplied; coverage cannot "
            "be checked, and the proposal's own predicted surface is a scored "
            "prediction, never a scope input (§6.4, invariant 18)",)))
    else:
        uncovered = [s for s in facts.derived_affected_surface
                     if not facts.correctness_oracles_by_surface.get(s)]
        gates.append(_gate("correctness_oracle_coverage", not uncovered, [
            f"no correctness oracle covers affected surface {s!r} (§8.4)"
            for s in uncovered
        ]))

    # --- target shapes occur in a real graph (§8.4) / prospective (§8.4.1) ---
    declared_shapes = [str(s) for s in (target.get("shapes") or [])]
    unseen = [s for s in declared_shapes if s not in facts.real_graph_shapes]
    if not declared_shapes and not facts.microkernel_only:
        # `target.shapes: []` is schema-valid, and it made BOTH this gate and the
        # confirmation-stratum gate pass by having nothing to iterate: delete the
        # thing the check inspects and the check reports PASS. A proposal with no
        # exact target shape has nothing whose real-graph occurrence could be
        # verified, which is COULD_NOT_CHECK — and COULD_NOT_CHECK blocks (§8.4).
        gates.append(_cnc("real_graph_shapes", (
            "the proposal declares no exact target shapes, so real-graph occurrence "
            "cannot be checked at all; §8.4 requires exact target shapes, and an "
            "empty target set is not a proposal that passes — it is one nobody can "
            "evaluate (declare `microkernel_only` if the campaign genuinely has no "
            "graph shapes)",)))
    elif not unseen or facts.microkernel_only:
        gates.append(_gate("real_graph_shapes", True))
    elif facts.architectural_campaign:
        reasons = []
        for shape in unseen:
            declaration = facts.prospective_shapes.get(shape)
            if not isinstance(declaration, Mapping):
                reasons.append(
                    f"shape {shape!r} does not occur in a real graph and no "
                    "prospective-shape declaration exists for it (§8.4.1)")
                continue
            for key in ("mechanism", "observation"):
                if not str(declaration.get(key) or "").strip():
                    reasons.append(
                        f"prospective shape {shape!r} declares no {key}; §8.4.1 "
                        "admits it only with the mechanism by which it comes to occur "
                        "AND a way to observe that it did")
        gates.append(_gate("real_graph_shapes", not reasons, reasons))
    else:
        gates.append(_gate("real_graph_shapes", False, [
            f"target shape {s!r} does not occur in a real graph and the campaign is "
            "not microkernel-only (§8.4)" for s in unseen
        ]))

    # --- selection/confirmation split (P-AK-SEARCH-1) -----------------------
    leaked = sorted(set(declared_shapes) & set(facts.confirmation_shapes))
    gates.append(_gate("confirmation_stratum", not leaked, [
        f"target shape {s!r} belongs to the CONFIRMATION stratum; a proposal that "
        "targets a confirmation shape is rejected before it consumes a window "
        "(P-AK-SEARCH-1, selection/confirmation split)" for s in leaked
    ]))

    # --- do-not-repeat ledger (§8.4, §19.2, §19.3) --------------------------
    disposition = evaluate_ledger(facts.ledger_matches)
    gates.append(_gate("do_not_repeat", not disposition.blocking, [
        f"{e.entry_id} ({e.ledger_class}): {e.statement} — reopen_when: "
        f"{e.reopen_when}" for e in disposition.blocking
    ]))
    if disposition.toothless:
        gates.append(PreRunGate("do_not_repeat_toothless", schemas.Check(
            schemas.COULD_NOT_CHECK,
            tuple(f"{e.entry_id} ({e.ledger_class}) matched but does not suppress: "
                  + "; ".join(e.authority().reasons) for e in disposition.toothless),
        ), blocking=False))
    # `excluded_cells` and `advisory` were computed and then dropped on the floor:
    # a matched CONDITIONAL_NEGATIVE ("exclude the matched cells, other regimes
    # eligible") and a matched CONFOUNDED_RESULT ("do not learn its sign, require a
    # repaired experiment") left NO trace on the critique record, so no downstream
    # reader could act on either. §19.2 gives each ledger class a planner-behaviour
    # column; a class whose behaviour is not "block" still has a behaviour.
    if disposition.excluded_cells:
        gates.append(PreRunGate("do_not_repeat_excluded_cells", schemas.Check(
            schemas.COULD_NOT_CHECK,
            tuple(f"{e.entry_id} ({e.ledger_class}) excludes matched cell(s) "
                  f"{sorted(e.match_dimensions)}: {e.statement} — other regimes remain "
                  f"eligible; reopen_when: {e.reopen_when}"
                  for e in disposition.excluded_cells),
        ), blocking=False))
    if disposition.advisory:
        gates.append(PreRunGate("do_not_repeat_advisory", schemas.Check(
            schemas.COULD_NOT_CHECK,
            tuple(f"{e.entry_id} ({e.ledger_class}): {e.statement} — reopen_when: "
                  f"{e.reopen_when}" for e in disposition.advisory),
        ), blocking=False))

    # --- budget (§8.4) ------------------------------------------------------
    if facts.budget is None:
        gates.append(_cnc("budget", (
            "no budget envelope supplied; a proposal cannot be checked against a "
            "budget nobody declared",)))
    else:
        reasons = []
        if float(request["expected_minutes"]) > facts.budget.minutes_remaining:
            reasons.append(
                f"expected_minutes {request['expected_minutes']} exceeds remaining "
                f"{facts.budget.minutes_remaining}")
        if float(request["expected_storage_gb"]) > facts.budget.storage_gb_remaining:
            reasons.append(
                f"expected_storage_gb {request['expected_storage_gb']} exceeds "
                f"remaining {facts.budget.storage_gb_remaining}")
        if facts.budget.candidates_remaining < 1:
            reasons.append("no candidate budget remains (max_candidates reached)")
        gates.append(_gate("budget", not reasons, reasons))

    # --- repo/release domain ownership (§8.4, AK-D9/AK-D23) -----------------
    outside = sorted(facts.proposal_domains - facts.backend_owned_domains)
    gates.append(_gate("domain_ownership", not outside, [
        f"domain {d!r} is not owned by this backend adapter; a change crossing a "
        "repo/release domain routes to that domain's own gate (§8.4, AK-D9)"
        for d in outside
    ]))

    # --- the evaluator is not modifiable (§8.4, P-AK-SEARCH-1 denial 6) -----
    gates.append(_gate("evaluator_unchanged", not facts.evaluator_change_required, [
        "the proposed evaluation step would require changing the evaluator; the "
        "controller RECORDS the gap, blocks release eligibility for the affected "
        "lineage, continues unrelated research, and MAY draft an amendment — it does "
        "not patch the instrument (P-AK-SEARCH-1 denial 6, AK-D10)",
    ] if facts.evaluator_change_required else []))

    # --- change_class maps to a cheap suite (§9.5, §7.2) --------------------
    change_class = manifest["change_class"]
    suite = schemas.CHANGE_CLASS_CHEAP_SUITE.get(change_class)
    gates.append(_gate("cheap_suite", suite is not None, [
        f"change_class {change_class!r} maps to no cheap suite (§9.5)"
    ] if suite is None else []))

    # --- oracle registry (§6.5) --------------------------------------------
    gates.append(_oracle_gate(manifest))

    # --- origin ceiling (§8.4.0, AK-D38) ------------------------------------
    origin = origin_block.get("origin")
    grade = origin_block.get("evidence_grade")
    gates.append(_gate(
        "origin_grade_ceiling",
        not (origin == ORIGIN_OPERATOR_HYPOTHESIS and grade != GRADE_DESIGN_PRIOR),
        [f"origin {ORIGIN_OPERATOR_HYPOTHESIS!r} carries evidence_grade {grade!r}; an "
         f"operator hypothesis enters at {GRADE_DESIGN_PRIOR!r} and can never be "
         "promoted by its origin (AK-D38, §19.0 rule 4)"]
        if origin == ORIGIN_OPERATOR_HYPOTHESIS and grade != GRADE_DESIGN_PRIOR
        else []))

    return tuple(gates)


def _oracle_gate(manifest: Mapping[str, Any]) -> PreRunGate:
    """§6.5: a port names a DECLARED, non-retired oracle and records its class."""
    reference = dict(manifest.get("oracle_reference") or {})
    named = isinstance(reference.get("oracle"), str) and reference["oracle"].strip()
    is_port = manifest.get("campaign_kind") == "oracle_port"
    if not is_port and not named:
        return _gate("oracle_registry", True)
    if not is_port:
        # `campaign_kind` is a MODEL-owned field, and gating the whole registry
        # check on it meant a proposal that names AITER — retired because gfx90a
        # is not on its supported-hardware table — passed `oracle_registry`
        # unconditionally by calling itself a `dispatch` campaign. Registry
        # membership and retirement are properties of the ORACLE, so they are
        # checked wherever an oracle is named. The harvest-class and attribution
        # requirements stay tied to an actual port, which is what §6.5 scopes them to.
        row = oracle_row(reference.get("oracle"))
        reasons = []
        if row is None:
            reasons.append(
                f"oracle {reference.get('oracle')!r} is not in the declared registry; "
                "new oracles enter through research-intake, not by an agent adding a "
                "row (AK-D34) — and relabelling the campaign_kind does not make the "
                "reference disappear from the record")
        elif row.retired:
            reasons.append(f"oracle {row.oracle_id!r} is RETIRED: {row.retirement_note}")
        return _gate("oracle_registry", not reasons, reasons)
    row = oracle_row(reference.get("oracle"))
    reasons: list = []
    if row is None:
        reasons.append(
            f"oracle {reference.get('oracle')!r} is not in the declared registry; new "
            "oracles enter through research-intake, which verifies real gfx90a/EPYC "
            "support and assigns the harvest class — not by an agent adding a row "
            "(AK-D34)")
    elif row.retired:
        reasons.append(f"oracle {row.oracle_id!r} is RETIRED: {row.retirement_note}")
    declared_class = reference.get("harvest_class")
    if declared_class not in HARVEST_CLASSES:
        reasons.append(
            f"oracle_reference.harvest_class {declared_class!r} is not one of "
            f"{list(HARVEST_CLASSES)}; §6.5 requires every port to record the harvest "
            "class it relied on")
    elif row is not None and row.harvest_class not in oracles.SPLIT_HARVEST_CLASSES \
            and declared_class != row.harvest_class:
        # A SPLIT row (`mixed` splits by part, `conditional` by availability) does
        # not fix one class for the whole tree, so §6.5's *"records the harvest
        # class it relied on"* is a narrower answer than the row's own label and
        # must not be compared for equality against it. `conditional` was missing
        # from this module's vocabulary entirely, which made the design's own
        # FlashAttention row unrepresentable.
        reasons.append(
            f"oracle {row.oracle_id!r} is classified {row.harvest_class!r}; the "
            f"proposal relies on {declared_class!r}. Misclassifying is a SCHEDULE "
            "problem: a `reimplement` oracle costs authoring effort a "
            "`portable_source` one does not")
    if not str(reference.get("attribution") or "").strip():
        reasons.append(
            "oracle_reference.attribution is required — recorded as courtesy and "
            "provenance, never as a condition of entry (§6.5)")
    return _gate("oracle_registry", not reasons, reasons)


@dataclass(frozen=True)
class PreRunCritique:
    """The disposed pre-run critique. `disposition` is COMPUTED, never copied.

    `model_disposition` is retained beside it precisely so the two can disagree on
    the record: a critic that keeps accepting what the gates reject is a critic
    worth replacing, and that is only visible if both are kept.
    """

    proposal_id: str
    disposition: str
    gates: tuple
    answers: tuple
    reasons: tuple
    revisions: Mapping[str, Any]
    binding: Optional[ModelBinding]
    usage_tokens: int
    model_consulted: bool
    model_disposition: Optional[str]
    independence: schemas.Check
    decided_at: str

    @property
    def accepted(self) -> bool:
        return self.disposition == DISPOSITION_ACCEPT

    def blocking_failures(self) -> tuple:
        return tuple(g for g in self.gates
                     if g.blocking and g.check.outcome != schemas.PASS)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "disposition": self.disposition,
            "gates": [g.to_dict() for g in self.gates],
            "answers": [a.to_dict() for a in self.answers],
            "reasons": list(self.reasons),
            "revisions": dict(self.revisions),
            "binding": None if self.binding is None else self.binding.to_dict(),
            "usage_tokens": self.usage_tokens,
            "model_consulted": self.model_consulted,
            "model_disposition": self.model_disposition,
            "independence": {"outcome": self.independence.outcome,
                             "reasons": list(self.independence.reasons)},
            "decided_at": self.decided_at,
        }

    def verdict_block(self) -> dict:
        """The `critic_verdict` block for the manifest.

        `schemas.CRITIC_STATUSES` is a CLOSED three-value vocabulary
        (pending/pass/fail) and this module does not get to extend it, so `revise`
        and `reject` both land on `fail` and the first reason names which. The
        distinction lives in `disposition`, on the critique record, where it is not
        lost — a revised proposal is redrafted, a rejected one is not.
        """
        status = "pass" if self.accepted else "fail"
        reasons = list(self.reasons)
        if not self.accepted:
            reasons.insert(0, f"disposition={self.disposition}")
        return {"status": status, "reasons": reasons}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z")


def check_critic_independence(
    planner_binding: ModelBinding,
    critic_binding: ModelBinding,
    *,
    shared_model_reason: Optional[str] = None,
) -> schemas.Check:
    """§6.3: prefer a different provider/model for the critic.

    PASS when the provider or the model differs. When they are identical, a
    campaign may still proceed — but only by DECLARING why, and the reason lands
    on the critique record. Silence is refused, because *"a critic sharing the
    planner's blind spots mostly agrees"* and an agreeing critic is
    indistinguishable from a working one until something expensive gets through.
    """
    if not isinstance(planner_binding, ModelBinding) or \
            not isinstance(critic_binding, ModelBinding):
        raise TypeError("both bindings must be ModelBinding values")
    if planner_binding.identity() != critic_binding.identity():
        return schemas.Check(schemas.PASS)
    if isinstance(shared_model_reason, str) and shared_model_reason.strip():
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"critic shares the planner's model {critic_binding.model_id!r} on "
            f"{critic_binding.provider!r}; declared reason: {shared_model_reason}. "
            "Independence is NOT established — the critique is admitted with this "
            "limitation on the record (§6.3)",
        ))
    return schemas.Check(schemas.FAIL, (
        f"critic binding is identical to the planner binding "
        f"({critic_binding.provider}/{critic_binding.model_id}) and no reason was "
        "declared; a critic sharing the planner's blind spots mostly agrees (§6.3)",
    ))


#: Key stems that would make a critic payload into a gate waiver. Blunt on
#: purpose — this module simply never uses these words as field names.
_WAIVER_STEMS = (
    "waiv", "override", "bypass", "exempt", "suppress_gate", "force_pass",
    "skip_gate", "accept_despite", "ignore_gate", "disable_gate", "downgrade_gate",
)


def find_gate_waiver_keys(obj: Any, path: str = "$") -> list:
    """Dotted paths of every waiver-flavoured key anywhere inside `obj` (§6.3)."""
    found: list = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{path}.{key}"
            flat = re.sub(r"[^a-z0-9]+", "", str(key).lower())
            if any(stem.replace("_", "") in flat for stem in _WAIVER_STEMS):
                found.append(child)
            found.extend(find_gate_waiver_keys(value, child))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(find_gate_waiver_keys(value, f"{path}[{index}]"))
    return found


def _parse_answers(data: Mapping[str, Any]) -> tuple:
    raw = data.get("answers")
    if not isinstance(raw, Mapping):
        raise ProviderResponseInvalid("critique.answers must be an object")
    missing = [q.qid for q in PRE_RUN_QUESTIONS if q.qid not in raw]
    if missing:
        raise ProviderResponseInvalid(
            f"critique.answers omits question(s) {missing}; §6.3's set is answered in "
            "full or the critique is incomplete")
    unknown = sorted(k for k in raw if k not in _QUESTION_BY_ID)
    if unknown:
        raise ProviderResponseInvalid(f"critique.answers has unknown question(s) {unknown}")
    answers: list = []
    for question in PRE_RUN_QUESTIONS:
        entry = raw[question.qid]
        if not isinstance(entry, Mapping):
            raise ProviderResponseInvalid(
                f"critique.answers.{question.qid} must be an object with 'outcome' "
                "and 'reasons'")
        outcome = entry.get("outcome")
        if outcome not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise ProviderResponseInvalid(
                f"critique.answers.{question.qid}.outcome must be one of "
                f"{[schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK]}, got "
                f"{outcome!r}")
        reasons = entry.get("reasons") or []
        if not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons):
            raise ProviderResponseInvalid(
                f"critique.answers.{question.qid}.reasons must be a list of strings")
        if outcome != schemas.PASS and not reasons:
            raise ProviderResponseInvalid(
                f"critique.answers.{question.qid}: a non-PASS answer must state why; "
                "an unexplained rejection cannot be acted on or contested")
        answers.append(PreRunAnswer(question.qid, schemas.Check(outcome, tuple(reasons))))
    return tuple(answers)


def _dispose_pre_run(gates: Sequence[PreRunGate],
                     answers: Sequence[PreRunAnswer],
                     model_disposition: Optional[str]) -> tuple:
    """Compute the disposition. Severity only ever goes UP.

    The model can make a proposal's fate worse (reject, revise) and can never make
    it better: a deterministic gate failure is not answerable by an opinion. This
    is the whole of *"the critic may reject or revise; it cannot waive"* in one
    max().
    """
    severity = _DISPOSITION_SEVERITY[DISPOSITION_ACCEPT]
    reasons: list = []
    for gate in gates:
        if not gate.blocking or gate.check.outcome == schemas.PASS:
            continue
        severity = max(severity, _DISPOSITION_SEVERITY[DISPOSITION_REJECT])
        reasons.append(f"gate {gate.gate_id} -> {gate.check.outcome}: "
                       + "; ".join(gate.check.reasons))
    for answer in answers:
        question = _QUESTION_BY_ID[answer.qid]
        if not question.blocking or answer.check.outcome == schemas.PASS:
            continue
        if answer.check.outcome == schemas.FAIL:
            severity = max(severity, _DISPOSITION_SEVERITY[DISPOSITION_REVISE])
            reasons.append(f"critic answered FAIL to {answer.qid!r} "
                           f"({question.design_question}): "
                           + "; ".join(answer.check.reasons))
        elif question.on_could_not_check == "reject":
            severity = max(severity, _DISPOSITION_SEVERITY[DISPOSITION_REJECT])
            reasons.append(
                f"critic could not check {answer.qid!r} and that axis blocks: "
                + "; ".join(answer.check.reasons))
        else:
            reasons.append(f"advisory: {answer.qid!r} COULD_NOT_CHECK — "
                           + "; ".join(answer.check.reasons))
    if model_disposition is not None:
        if model_disposition not in _DISPOSITION_SEVERITY:
            # A KeyError here would surface as a crash rather than a refusal, and
            # this is the one function whose whole job is to be un-influenceable.
            raise ProviderResponseInvalid(
                f"critique.disposition {model_disposition!r} is not one of "
                f"{list(DISPOSITIONS)}; an unrecognised disposition is refused, never "
                "treated as the most permissive one")
        severity = max(severity, _DISPOSITION_SEVERITY[model_disposition])
        if _DISPOSITION_SEVERITY[model_disposition] > 0:
            reasons.append(f"critic requested disposition {model_disposition!r}")
    disposition = next(d for d in DISPOSITIONS if _DISPOSITION_SEVERITY[d] == severity)
    return disposition, tuple(reasons)


def critique_proposal(
    *,
    manifest: Mapping[str, Any],
    facts: ProposalFacts,
    provider: Optional[Provider] = None,
    binding: Optional[ModelBinding] = None,
    bundle: Optional[PromptBundle] = None,
    planner_binding: Optional[ModelBinding] = None,
    shared_model_reason: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    skip_model_on_deterministic_reject: bool = True,
    clock=_iso_now,
) -> PreRunCritique:
    """Run PRE_RUN_CRITIC: deterministic gates first, then the metered call.

    With `skip_model_on_deterministic_reject` (the default) a proposal the gates
    already reject never reaches the model. §8.4: *"Cheap deterministic checks run
    before metered drafting, not after — the reverse ordering cost roughly 38
    draft-and-critique cycles that were paid for and then thrown away."*

    `provider` may be omitted entirely, which runs the deterministic half alone.
    That is a legitimate mode, not a degraded one: the gates are the part with
    authority.
    """
    gates = evaluate_pre_run_gates(manifest, facts)
    independence = schemas.Check(
        schemas.COULD_NOT_CHECK, ("no planner binding supplied for comparison",))
    if planner_binding is not None and binding is not None:
        independence = check_critic_independence(
            planner_binding, binding, shared_model_reason=shared_model_reason)
        if independence.outcome == schemas.FAIL:
            raise CriticIndependenceError("; ".join(independence.reasons))

    deterministic_reject = any(
        g.blocking and g.check.outcome != schemas.PASS for g in gates)
    consult = provider is not None and not (
        deterministic_reject and skip_model_on_deterministic_reject)

    answers: tuple = ()
    usage_tokens = 0
    model_disposition: Optional[str] = None
    revisions: dict = {}

    if consult:
        if binding is None or bundle is None:
            raise ValueError("a provider requires both `binding` and `bundle`")
        if bundle.role != ROLE_PRE_RUN_CRITIC:
            raise ValueError(
                f"critique_proposal requires a {ROLE_PRE_RUN_CRITIC!r} bundle, got "
                f"{bundle.role!r}")
        request = ModelRequest(role=ROLE_PRE_RUN_CRITIC, bundle=bundle,
                               contract=PRE_RUN_RESPONSE_CONTRACT, binding=binding,
                               max_output_tokens=max_output_tokens)
        completion = provider.complete(request)
        if not isinstance(completion, Completion):
            raise ProviderResponseInvalid(
                f"provider returned {type(completion).__name__}, not a Completion")
        honoured = check_binding_honoured(request, completion)
        if honoured.outcome != schemas.PASS:
            raise ProviderResponseInvalid(
                "provider did not honour the requested critic binding: "
                + "; ".join(honoured.reasons))
        violations = PRE_RUN_RESPONSE_CONTRACT.validate(completion.data)
        if violations:
            raise ProviderResponseInvalid(
                "pre-run critique does not satisfy its response contract: "
                + "; ".join(violations))
        waivers = find_gate_waiver_keys(dict(completion.data))
        if waivers:
            raise GateWaiverAttempt(waivers)
        answers = _parse_answers(completion.data)
        model_disposition = completion.data.get("disposition")
        if model_disposition not in DISPOSITIONS:
            raise ProviderResponseInvalid(
                f"critique.disposition must be one of {list(DISPOSITIONS)}, got "
                f"{model_disposition!r}")
        raw_revisions = completion.data.get("revisions") or {}
        if not isinstance(raw_revisions, Mapping):
            raise ProviderResponseInvalid("critique.revisions must be an object")
        revisions = dict(raw_revisions)
        usage_tokens = completion.usage.total

    disposition, reasons = _dispose_pre_run(gates, answers, model_disposition)
    return PreRunCritique(
        proposal_id=str(manifest.get("proposal_id")),
        disposition=disposition,
        gates=tuple(gates),
        answers=answers,
        reasons=reasons,
        revisions=revisions,
        binding=binding,
        usage_tokens=usage_tokens,
        model_consulted=consult,
        model_disposition=model_disposition,
        independence=independence,
        decided_at=clock(),
    )


#: Paths a critic revision may never touch. The first group is controller
#: provenance and disposition; the rest is THE GATE SURFACE — every field
#: `evaluate_pre_run_gates()` reads.
#:
#: The gate surface has to be complete, not representative. The gates are
#: computed against the pre-revision manifest and are NOT re-run here, so any
#: gate-read field a revision could move is a gate the critic passes by editing
#: rather than by waiving: revise `target.shapes` and `real_graph_shapes` and
#: `confirmation_stratum` were decided about a different proposal; revise
#: `resource_request.expected_minutes` and the `budget` gate was. Listing only
#: the two wall-share fields left the other seven open.
FORBIDDEN_REVISION_PATHS = (
    "schema", "proposal_id", "campaign_id", "parent_candidate_id", "controller",
    "realized_cost", "critic_verdict", "created_at", "narrative_retrievable",
    "hypothesis_origin", "novelty_basis",
    "declared_symbol_deltas",
    # gate surface
    "mechanism_prediction",
    "target",
    "non_target",
    "change_class",
    "campaign_kind",
    "oracle_reference",
    "resource_request",
)


#: The evaluator-gate lists. A revision may ADD to these and may never remove
#: from them, at any spelling of the path (§6.3).
_REQUIRED_GATE_PATHS = ("evaluation_plan.required_t0", "evaluation_plan.required_t1")


def _get_path(obj: Mapping[str, Any], path: str) -> Any:
    node: Any = obj
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _set_path(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = obj
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, Mapping):
            raise RevisionRefused(f"revision path {path!r} does not exist on this manifest")
        node[part] = dict(child)
        node = node[part]
    if parts[-1] not in node:
        raise RevisionRefused(
            f"revision path {path!r} does not exist on this manifest; a critic revises "
            "a field, it does not invent one")
    node[parts[-1]] = value


def apply_pre_run_verdict(manifest: Mapping[str, Any],
                          critique: PreRunCritique) -> dict:
    """Stamp `critic_verdict` and apply admissible revisions. Re-validates.

    Two refusals, both structural rather than advisory:
      * a revision touching `FORBIDDEN_REVISION_PATHS` is refused outright; and
      * a revision that REMOVES any entry from `evaluation_plan.required_t0` or
        `required_t1` is refused — that is a gate waiver written as an edit, and
        it is the only shape a waiver could take from here (§6.3).
    """
    if not isinstance(critique, PreRunCritique):
        raise TypeError("critique must be a PreRunCritique")
    if str(manifest.get("proposal_id")) != critique.proposal_id:
        raise RevisionRefused(
            f"critique is for proposal {critique.proposal_id!r}, not "
            f"{manifest.get('proposal_id')!r}")
    # `critique.disposition` is computed by `_dispose_pre_run()`, but this function
    # is the one that STAMPS the verdict, and it was taking the disposition on
    # trust. A `PreRunCritique` carrying its own failing gates and an `accept`
    # disposition is internally contradictory, and stamping `critic_verdict: pass`
    # from it puts a deterministic gate failure behind a passing record. The
    # evidence is already on the object; the only question was whether anything read it.
    if critique.accepted:
        failures = critique.blocking_failures()
        if failures:
            raise RevisionRefused(
                "refusing to stamp a passing critic_verdict on a proposal whose own "
                "blocking gate(s) did not pass: "
                + "; ".join(f"{g.gate_id} -> {g.check.outcome}" for g in failures)
                + ". The disposition is computed from the gates, never asserted "
                  "alongside them (§6.3, §4 invariant 4)")
    updated = {k: (dict(v) if isinstance(v, Mapping) else v)
               for k, v in manifest.items()}

    waivers = find_gate_waiver_keys(dict(critique.revisions))
    if waivers:
        raise GateWaiverAttempt(waivers)

    for path, value in sorted(critique.revisions.items()):
        if path in FORBIDDEN_REVISION_PATHS or \
                any(path.startswith(f"{p}.") for p in FORBIDDEN_REVISION_PATHS):
            raise RevisionRefused(
                f"revision path {path!r} is controller-owned or gate-defining; the "
                "critic may reject or revise a proposal, never edit the record's "
                "provenance or the surface it is measured against (§6.3)")
        if path in _REQUIRED_GATE_PATHS:
            before = set(_get_path(manifest, path) or [])
            after = set(value or [])
            dropped = sorted(before - after)
            if dropped:
                raise RevisionRefused(
                    f"revision to {path!r} removes required gate(s) {dropped}; the "
                    "critic can never waive an evaluator gate (§6.3, invariant 4). "
                    "Adding gates is admissible; removing them is not")
        _set_path(updated, path, value)

    # Comparing the revision PATHS is not enough: a revision to any ANCESTOR of a
    # required-gate list rewrites it without ever naming it. `{"evaluation_plan":
    # {...}}` passed every path check above and still dropped a T0 gate. So the
    # authoritative comparison is over the RESULT — what the required set was
    # before, and what it is now — which no spelling of the path can route around.
    for path in _REQUIRED_GATE_PATHS:
        before = set(_get_path(manifest, path) or [])
        after = set(_get_path(updated, path) or [])
        dropped = sorted(before - after)
        if dropped:
            raise RevisionRefused(
                f"the revised proposal drops required gate(s) {dropped} from {path!r}; "
                "the critic can never waive an evaluator gate (§6.3, invariant 4), and "
                "revising a parent of the gate list is the same waiver spelled "
                "differently. Adding gates is admissible; removing them is not")

    updated["critic_verdict"] = critique.verdict_block()
    violations = schemas.validate_proposal(updated)
    if violations:
        raise RevisionRefused(
            "the revised proposal no longer validates: " + "; ".join(violations))
    return updated


# =============================================================================
# §8.8 POST_RUN_CRITIC — interpretation, reconciled against the raw gates
# =============================================================================

HYPOTHESIS_STATUSES = ("confirmed", "refuted", "inconclusive")
HYPOTHESIS_KINDS = ("rate", "mechanism", "capability")
MECHANISM_STATUSES = ("confirmed", "refuted", "unavailable")

SIGNAL_SIGNAL = "signal"
SIGNAL_NOISE = "noise"
SIGNAL_NO_DETECTABLE_DIFFERENCE = "no_detectable_difference"
SIGNAL_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SIGNAL_NOT_MEASURED = "not_measured"
SIGNAL_CLASSES = (SIGNAL_SIGNAL, SIGNAL_NOISE, SIGNAL_NO_DETECTABLE_DIFFERENCE,
                  SIGNAL_INSUFFICIENT_EVIDENCE, SIGNAL_NOT_MEASURED)

CHAMPION_INTERACTIONS = ("compatible", "conflicts", "unknown")

#: One-to-one with the evaluator's own effect resolution. There is no judgement
#: here on purpose: "signal versus noise" is ALREADY decided by the calibrated
#: floor, the MDE and the e-value, all of which live in the `EffectEstimate`. A
#: critic that could disagree would be re-deciding a calibrated threshold in prose.
_SIGNAL_BY_RESOLUTION = {
    evaluator_api.EFFECT_NOT_MEASURED: SIGNAL_NOT_MEASURED,
    evaluator_api.EFFECT_BELOW_NOISE_FLOOR: SIGNAL_NOISE,
    evaluator_api.EFFECT_NO_DETECTABLE_DIFFERENCE: SIGNAL_NO_DETECTABLE_DIFFERENCE,
    evaluator_api.EFFECT_EVIDENCE_BELOW_THRESHOLD: SIGNAL_INSUFFICIENT_EVIDENCE,
    evaluator_api.EFFECT_IMPROVEMENT: SIGNAL_SIGNAL,
    evaluator_api.EFFECT_REGRESSION: SIGNAL_SIGNAL,
}

_RATE_ADMISSIBLE_BY_RESOLUTION = {
    # `no detectable difference` is a RESULT and a decision, not a failed
    # experiment (P-AK-SEARCH-1, MDE clause). Admitting `inconclusive` for it is
    # how a decided negative stays alive and re-spends budget forever.
    evaluator_api.EFFECT_IMPROVEMENT: frozenset({"confirmed"}),
    evaluator_api.EFFECT_REGRESSION: frozenset({"refuted"}),
    evaluator_api.EFFECT_BELOW_NOISE_FLOOR: frozenset({"refuted"}),
    evaluator_api.EFFECT_NO_DETECTABLE_DIFFERENCE: frozenset({"refuted"}),
    evaluator_api.EFFECT_EVIDENCE_BELOW_THRESHOLD: frozenset({"inconclusive"}),
    evaluator_api.EFFECT_NOT_MEASURED: frozenset({"inconclusive"}),
}


def _mechanism_gates(verdict: evaluator_api.Verdict) -> tuple:
    return tuple(g for g in verdict.gates
                 if g.gate_class == evaluator_api.GATE_MECHANISM)


def derive_signal_class(verdict: evaluator_api.Verdict) -> str:
    """The signal/noise call, derived. The critic reports it; it never decides it."""
    if not isinstance(verdict, evaluator_api.Verdict):
        raise TypeError("verdict must be an evaluator_api.Verdict")
    return _SIGNAL_BY_RESOLUTION[verdict.effect_resolution]


def admissible_hypothesis_statuses(verdict: evaluator_api.Verdict,
                                   kind: str,
                                   *,
                                   capability_objective_met: Optional[bool] = None
                                   ) -> frozenset:
    """Which §8.8 hypothesis classifications the RAW GATES admit.

    Precedence mirrors `evaluator.api._derive` exactly, and for the same reasons:

      1. **A voided window admits only `inconclusive`.** *"A voided run … MUST NOT
         be recorded as a candidate failure, because a drifted anchor says nothing
         whatever about the candidate."* Calling it `refuted` would bank a negative
         the measurement never supports.
      2. **A gate failure admits `refuted` or `inconclusive`, never `confirmed`.**
         Whether a correctness failure refutes THIS hypothesis depends on whether
         the hypothesis included correctness; what it can never do is confirm it.
      3. **Otherwise the effect resolution decides** for a rate hypothesis, the
         mechanism gates for a mechanism hypothesis, and the evaluator's own
         capability outcome for a capability hypothesis (§9.8).
    """
    if not isinstance(verdict, evaluator_api.Verdict):
        raise TypeError("verdict must be an evaluator_api.Verdict")
    if kind not in HYPOTHESIS_KINDS:
        raise ValueError(f"kind: {kind!r} not in {list(HYPOTHESIS_KINDS)}")
    if verdict.status == evaluator_api.STATUS_INVALID:
        return frozenset({"inconclusive"})
    if verdict.status == evaluator_api.STATUS_FAIL:
        return frozenset({"refuted", "inconclusive"})
    if verdict.status == evaluator_api.STATUS_INCONCLUSIVE:
        return frozenset({"inconclusive"})
    if kind == "rate":
        return _RATE_ADMISSIBLE_BY_RESOLUTION[verdict.effect_resolution]
    if kind == "mechanism":
        mechanism = admissible_mechanism_statuses(verdict)
        if mechanism == frozenset({"confirmed"}):
            return frozenset({"confirmed"})
        if mechanism == frozenset({"refuted"}):
            return frozenset({"refuted"})
        return frozenset({"inconclusive"})
    if capability_objective_met is True:
        return frozenset({"confirmed"})
    if capability_objective_met is False:
        return frozenset({"refuted"})
    return frozenset({"inconclusive"})


def admissible_mechanism_statuses(verdict: evaluator_api.Verdict) -> frozenset:
    """Which mechanism classifications the mechanism-class gates admit.

    `unavailable` is the honest answer when no mechanism gate ran or every one of
    them returned COULD_NOT_CHECK — an unsupported counter is explicit (§6, the
    profiler surface row), and reporting `refuted` for a counter nobody could read
    would manufacture a negative out of missing instrumentation.
    """
    if not isinstance(verdict, evaluator_api.Verdict):
        raise TypeError("verdict must be an evaluator_api.Verdict")
    if verdict.status == evaluator_api.STATUS_INVALID:
        return frozenset({"unavailable"})
    gates = _mechanism_gates(verdict)
    if not gates:
        return frozenset({"unavailable"})
    outcomes = {g.check.outcome for g in gates}
    if outcomes == {schemas.COULD_NOT_CHECK}:
        return frozenset({"unavailable"})
    if schemas.FAIL in outcomes:
        return frozenset({"refuted"})
    if outcomes == {schemas.PASS}:
        return frozenset({"confirmed"})
    return frozenset({"confirmed", "unavailable"})


@dataclass(frozen=True)
class WallShareTranslation:
    """Op-level movement translated to the graph (§8.8), with its receipt.

    `graph_delta_claimed` is `None` when no graph-level claim is admissible, which
    is not a missing value: §12's *"Profiler metric moves but wall time does not"*
    row says the mechanism bonus never substitutes for real graph gain, so a run
    with no rankable effect reports NO graph delta rather than a small one.
    """

    op_share_before: float
    op_delta_observed: float
    graph_delta_claimed: Optional[float]
    receipt_id: str
    explanation: str = ""

    def __post_init__(self) -> None:
        for name in ("op_share_before", "op_delta_observed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"WallShareTranslation.{name} must be a number")
        if self.graph_delta_claimed is not None and (
                isinstance(self.graph_delta_claimed, bool)
                or not isinstance(self.graph_delta_claimed, (int, float))):
            raise ValueError("WallShareTranslation.graph_delta_claimed must be a "
                             "number or None")
        if not str(self.receipt_id or "").strip():
            raise ValueError(
                "WallShareTranslation.receipt_id: required — an op-to-graph "
                "translation with no wall-share receipt is arithmetic on a number "
                "nobody measured")

    def to_dict(self) -> dict:
        return {
            "op_share_before": float(self.op_share_before),
            "op_delta_observed": float(self.op_delta_observed),
            "graph_delta_claimed": (None if self.graph_delta_claimed is None
                                    else float(self.graph_delta_claimed)),
            "receipt_id": self.receipt_id,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class NextExperiment:
    """The next DISCRIMINATING experiment (§8.8).

    "Discriminating" is enforced, not hoped for: it must name at least two
    competing mechanisms and the observation that separates them. §8.4 ranks
    expected information gain first, and an experiment that cannot separate two
    hypotheses has none to offer.
    """

    question: str
    distinguishes: tuple
    observation: str
    tier: str
    estimated_cost_class: str

    def __post_init__(self) -> None:
        for name in ("question", "observation", "estimated_cost_class"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"NextExperiment.{name}: required and non-empty")
        if not isinstance(self.distinguishes, tuple) or len(self.distinguishes) < 2:
            raise ValueError(
                "NextExperiment.distinguishes must name at least TWO competing "
                "mechanisms; an experiment that separates nothing is not a "
                "discriminating experiment (§8.8)")
        if self.tier not in schemas.TIERS:
            raise ValueError(f"NextExperiment.tier: {self.tier!r} not in "
                             f"{sorted(schemas.TIERS)}")

    def to_dict(self) -> dict:
        return {"question": self.question, "distinguishes": list(self.distinguishes),
                "observation": self.observation, "tier": self.tier,
                "estimated_cost_class": self.estimated_cost_class}


@dataclass(frozen=True)
class DurableLesson:
    """The §8.8 durable do-not-repeat lesson, WITH its receipt (§19.3).

    Built as a `LedgerEntry` so the lesson is admitted on exactly the terms a
    ledger entry is admitted on — there is no second, laxer path into the ledger
    through the post-run critic.
    """

    entry: LedgerEntry
    derived_from_event_ids: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entry, LedgerEntry):
            raise TypeError("DurableLesson.entry must be a LedgerEntry")

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry.entry_id,
            "ledger_class": self.entry.ledger_class,
            "statement": self.entry.statement,
            "match_dimensions": dict(self.entry.match_dimensions),
            "reopen_when": self.entry.reopen_when,
            "receipt": self.entry.receipt,
            "verified_against_commit": self.entry.verified_against_commit,
            "evidence_grade": self.entry.evidence_grade,
            "scope": self.entry.scope,
            "derived_from_event_ids": list(self.derived_from_event_ids),
        }


@dataclass(frozen=True)
class PostRunClassification:
    """The critic's INTERPRETATION of one run. Reconciled before it is believed."""

    hypothesis_kind: str
    hypothesis_status: str
    mechanism_status: str
    signal_class: str
    wall_share: WallShareTranslation
    target_behaviour: Mapping[str, str]
    non_target_behaviour: Mapping[str, str]
    champion_interaction: str
    champion_reason: str
    next_experiment: NextExperiment
    durable_lesson: Optional[DurableLesson] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.hypothesis_kind not in HYPOTHESIS_KINDS:
            raise ValueError(f"hypothesis_kind: {self.hypothesis_kind!r} not in "
                             f"{list(HYPOTHESIS_KINDS)}")
        if self.hypothesis_status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"hypothesis_status: {self.hypothesis_status!r} not in "
                             f"{list(HYPOTHESIS_STATUSES)}")
        if self.mechanism_status not in MECHANISM_STATUSES:
            raise ValueError(f"mechanism_status: {self.mechanism_status!r} not in "
                             f"{list(MECHANISM_STATUSES)}")
        if self.signal_class not in SIGNAL_CLASSES:
            raise ValueError(f"signal_class: {self.signal_class!r} not in "
                             f"{list(SIGNAL_CLASSES)}")
        if self.champion_interaction not in CHAMPION_INTERACTIONS:
            raise ValueError(f"champion_interaction: {self.champion_interaction!r} not "
                             f"in {list(CHAMPION_INTERACTIONS)}")
        for name in ("target_behaviour", "non_target_behaviour"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"PostRunClassification.{name} must be a mapping")
        if not str(self.champion_reason or "").strip():
            raise ValueError("champion_reason: required and non-empty")

    def to_dict(self) -> dict:
        return {
            "hypothesis_kind": self.hypothesis_kind,
            "hypothesis_status": self.hypothesis_status,
            "mechanism_status": self.mechanism_status,
            "signal_class": self.signal_class,
            "wall_share": self.wall_share.to_dict(),
            "target_behaviour": dict(self.target_behaviour),
            "non_target_behaviour": dict(self.non_target_behaviour),
            "champion_interaction": self.champion_interaction,
            "champion_reason": self.champion_reason,
            "next_experiment": self.next_experiment.to_dict(),
            "durable_lesson": (None if self.durable_lesson is None
                               else self.durable_lesson.to_dict()),
            "notes": self.notes,
        }


POST_RUN_RESPONSE_CONTRACT = ResponseContract(
    name="autokernel_post_run_classification",
    required_keys=(
        "hypothesis_kind", "hypothesis_status", "mechanism_status", "signal_class",
        "wall_share", "target_behaviour", "non_target_behaviour",
        "champion_interaction", "champion_reason", "next_experiment",
    ),
    optional_keys=("durable_lesson", "notes"),
)


def reconcile_classification(classification: PostRunClassification,
                             verdict: evaluator_api.Verdict,
                             *,
                             manifest: Mapping[str, Any],
                             facts: ProposalFacts) -> schemas.Check:
    """THE deterministic check: does the interpretation follow from the raw gates?

    §8.8 — *"The deterministic controller checks the classification against the raw
    gates."* Every reason returned names the classification field and the gate
    evidence it contradicts, so a FAIL is actionable rather than merely negative.
    """
    if not isinstance(classification, PostRunClassification):
        raise TypeError("classification must be a PostRunClassification")
    if not isinstance(verdict, evaluator_api.Verdict):
        raise TypeError("verdict must be an evaluator_api.Verdict")
    if not isinstance(facts, ProposalFacts):
        raise TypeError("facts must be a ProposalFacts")

    reasons: list = []

    waivers = find_gate_waiver_keys(classification.to_dict())
    if waivers:
        raise GateWaiverAttempt(waivers)

    derived_signal = derive_signal_class(verdict)
    if classification.signal_class != derived_signal:
        reasons.append(
            f"signal_class {classification.signal_class!r} contradicts the evaluator's "
            f"effect resolution {verdict.effect_resolution!r}, which is "
            f"{derived_signal!r}. The floor, the MDE and the e-value already decided "
            "this; a critic does not re-decide a calibrated threshold")

    admissible = admissible_hypothesis_statuses(
        verdict, classification.hypothesis_kind,
        capability_objective_met=facts.capability_objective_met)
    if classification.hypothesis_status not in admissible:
        reasons.append(
            f"hypothesis_status {classification.hypothesis_status!r} is not admitted by "
            f"the raw gates (status={verdict.status!r}, "
            f"effect_resolution={verdict.effect_resolution!r}); admissible: "
            f"{sorted(admissible)}")

    mech_admissible = admissible_mechanism_statuses(verdict)
    if classification.mechanism_status not in mech_admissible:
        reasons.append(
            f"mechanism_status {classification.mechanism_status!r} is not admitted by "
            f"the mechanism-class gates; admissible: {sorted(mech_admissible)}")

    claimed = classification.wall_share.graph_delta_claimed
    if claimed is not None:
        if not verdict.speed_rank_admissible:
            reasons.append(
                f"wall_share.graph_delta_claimed={claimed} while the run earned no "
                f"speed rank ({verdict.speed_rank_withheld_reason()}). A mechanism "
                "bonus never substitutes for real graph gain (§12)")
        ceiling = (manifest.get("mechanism_prediction") or {}).get(
            "expected_wall_share_ceiling")
        architectural = facts.architectural_campaign
        numeric_ceiling = isinstance(ceiling, (int, float)) and \
            not isinstance(ceiling, bool)
        if not numeric_ceiling and not architectural:
            # Skipping the comparison when the ceiling is missing made the §12
            # summed-local-gains check passable by DELETING the number it compares
            # against: a 95% graph delta reconciled clean on a manifest with no
            # ceiling. Inability to check is not a pass anywhere else in this
            # module and is not one here.
            reasons.append(
                f"wall_share.graph_delta_claimed={claimed} cannot be checked against a "
                f"wall-share ceiling: mechanism_prediction.expected_wall_share_ceiling "
                f"is {ceiling!r}. A graph-level claim with no ceiling on the record is "
                "exactly the summed-local-gain shape §12 rejects")
        elif numeric_ceiling and abs(float(claimed)) > float(ceiling) and \
                not architectural and not facts.fusion_explanation.strip():
            reasons.append(
                f"wall_share.graph_delta_claimed={claimed} exceeds the proposal's own "
                f"measured wall-share ceiling {ceiling} with no fusion explanation; "
                "summed local gains inflate readiness (§12)")
    if classification.wall_share.receipt_id not in facts.wall_share_receipts:
        reasons.append(
            f"wall_share.receipt_id {classification.wall_share.receipt_id!r} does not "
            "resolve to a measured wall-share receipt")

    target = dict(manifest.get("target") or {})
    non_target = dict(manifest.get("non_target") or {})
    for label, declared, reported in (
        ("target", target.get("regimes") or [], classification.target_behaviour),
        ("non_target", non_target.get("regimes") or [],
         classification.non_target_behaviour),
    ):
        missing = [r for r in declared if r not in reported]
        if missing:
            reasons.append(
                f"{label}_behaviour omits declared regime(s) {missing}; a declared "
                "regime with no reported behaviour is how a non-target regression "
                "leaves the record (§8.8, invariant 18)")

    if classification.champion_interaction == "compatible" and \
            facts.surface_reconciled.outcome != schemas.PASS:
        reasons.append(
            "champion_interaction 'compatible' requires a reconciled affected-surface "
            f"map; reconciliation is {facts.surface_reconciled.outcome} "
            f"({'; '.join(facts.surface_reconciled.reasons)}). Only changes with "
            "reconciled maps may be combined (§8.9)")

    lesson = classification.durable_lesson
    if lesson is None:
        if verdict.status != evaluator_api.STATUS_INVALID:
            reasons.append(
                "durable_lesson is absent. §8.8 requires a durable do-not-repeat "
                "lesson with its receipt; the ONLY run that owes none is a VOIDED "
                f"window, and this one is {verdict.status!r}")
    else:
        authority = lesson.entry.authority()
        if authority.outcome != schemas.PASS:
            reasons.append(
                "durable_lesson would enter the ledger without §19.3 standing: "
                + "; ".join(authority.reasons))
        # EVERY suppressing class, not just MATCHED_NEGATIVE. `HARD_CONSTRAINT` and
        # `SUPERSEDED_FACT` suppress at least as hard — `HARD_CONSTRAINT` is the one
        # that closes a family outright — and constraining only the middle one let
        # the post-run critic mint a family-closing HARD_CONSTRAINT off a run whose
        # hypothesis was CONFIRMED. A suppression must be a negative THIS run
        # produced (§8.8, §19.2, §12 "a wrong suppression closes a family").
        if lesson.entry.ledger_class in SUPPRESSING_LEDGER_CLASSES and \
                classification.hypothesis_status != "refuted":
            reasons.append(
                f"durable_lesson class {lesson.entry.ledger_class} suppresses future "
                f"proposals and therefore requires a refuted hypothesis; this run "
                f"classified it {classification.hypothesis_status!r}. A negative that "
                "closes a mechanism must be a negative this run produced")

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


@dataclass(frozen=True)
class PostRunCritique:
    """A RECONCILED post-run interpretation. It cannot exist unreconciled."""

    candidate_id: str
    classification: PostRunClassification
    reconciliation: schemas.Check
    verdict_status: str
    effect_resolution: str
    binding: Optional[ModelBinding]
    usage_tokens: int
    independence: schemas.Check
    decided_at: str

    def __post_init__(self) -> None:
        if self.reconciliation.outcome != schemas.PASS:
            raise ClassificationMismatch(self.reconciliation, self.classification)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "classification": self.classification.to_dict(),
            "reconciliation": {"outcome": self.reconciliation.outcome,
                               "reasons": list(self.reconciliation.reasons)},
            "verdict_status": self.verdict_status,
            "effect_resolution": self.effect_resolution,
            "binding": None if self.binding is None else self.binding.to_dict(),
            "usage_tokens": self.usage_tokens,
            "independence": {"outcome": self.independence.outcome,
                             "reasons": list(self.independence.reasons)},
            "decided_at": self.decided_at,
        }


def _parse_classification(data: Mapping[str, Any]) -> PostRunClassification:
    """Build a `PostRunClassification` from a provider's structured output.

    Every structural requirement (two competing mechanisms, a receipt id, a
    champion reason) is enforced by the dataclasses, so a malformed
    interpretation fails HERE rather than becoming a half-filled record.
    """
    wall = data.get("wall_share")
    if not isinstance(wall, Mapping):
        raise ProviderResponseInvalid("classification.wall_share must be an object")
    nxt = data.get("next_experiment")
    if not isinstance(nxt, Mapping):
        raise ProviderResponseInvalid("classification.next_experiment must be an object")
    lesson_raw = data.get("durable_lesson")
    lesson: Optional[DurableLesson] = None
    if lesson_raw is not None:
        if not isinstance(lesson_raw, Mapping):
            raise ProviderResponseInvalid("classification.durable_lesson must be an object")
        try:
            entry = LedgerEntry(
                entry_id=str(lesson_raw.get("entry_id") or ""),
                ledger_class=str(lesson_raw.get("ledger_class") or ""),
                statement=str(lesson_raw.get("statement") or ""),
                match_dimensions=dict(lesson_raw.get("match_dimensions") or {}),
                reopen_when=str(lesson_raw.get("reopen_when") or ""),
                receipt=lesson_raw.get("receipt"),
                verified_against_commit=lesson_raw.get("verified_against_commit"),
                evidence_grade=str(lesson_raw.get("evidence_grade") or "observation"),
                scope=str(lesson_raw.get("scope") or "cell"),
            )
        except (TypeError, ValueError) as exc:
            raise ProviderResponseInvalid(f"classification.durable_lesson: {exc}") from exc
        lesson = DurableLesson(
            entry=entry,
            derived_from_event_ids=tuple(
                str(e) for e in (lesson_raw.get("derived_from_event_ids") or [])),
        )
    try:
        return PostRunClassification(
            hypothesis_kind=str(data.get("hypothesis_kind") or ""),
            hypothesis_status=str(data.get("hypothesis_status") or ""),
            mechanism_status=str(data.get("mechanism_status") or ""),
            signal_class=str(data.get("signal_class") or ""),
            wall_share=WallShareTranslation(
                op_share_before=wall.get("op_share_before"),
                op_delta_observed=wall.get("op_delta_observed"),
                graph_delta_claimed=wall.get("graph_delta_claimed"),
                receipt_id=str(wall.get("receipt_id") or ""),
                explanation=str(wall.get("explanation") or ""),
            ),
            target_behaviour=dict(data.get("target_behaviour") or {}),
            non_target_behaviour=dict(data.get("non_target_behaviour") or {}),
            champion_interaction=str(data.get("champion_interaction") or ""),
            champion_reason=str(data.get("champion_reason") or ""),
            next_experiment=NextExperiment(
                question=str(nxt.get("question") or ""),
                distinguishes=tuple(str(x) for x in (nxt.get("distinguishes") or [])),
                observation=str(nxt.get("observation") or ""),
                tier=str(nxt.get("tier") or ""),
                estimated_cost_class=str(nxt.get("estimated_cost_class") or ""),
            ),
            durable_lesson=lesson,
            notes=str(data.get("notes") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise ProviderResponseInvalid(f"post-run classification: {exc}") from exc


def classify_run(
    *,
    provider: Provider,
    binding: ModelBinding,
    bundle: PromptBundle,
    verdict: evaluator_api.Verdict,
    manifest: Mapping[str, Any],
    facts: ProposalFacts,
    candidate_id: str,
    planner_binding: Optional[ModelBinding] = None,
    shared_model_reason: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    clock=_iso_now,
) -> PostRunCritique:
    """Run POST_RUN_CRITIC and RECONCILE it before returning (§8.8).

    RAISES `ClassificationMismatch` when the interpretation does not follow from
    the gates. Returning an unreconciled critique with a flag would mean the
    caller could consume it by forgetting to check — and the whole point of the
    reconciliation is that no downstream reader has to remember.
    """
    if bundle.role != ROLE_POST_RUN_CRITIC:
        raise ValueError(
            f"classify_run requires a {ROLE_POST_RUN_CRITIC!r} bundle, got "
            f"{bundle.role!r}")
    independence = schemas.Check(
        schemas.COULD_NOT_CHECK, ("no planner binding supplied for comparison",))
    if planner_binding is not None:
        independence = check_critic_independence(
            planner_binding, binding, shared_model_reason=shared_model_reason)
        if independence.outcome == schemas.FAIL:
            raise CriticIndependenceError("; ".join(independence.reasons))

    request = ModelRequest(role=ROLE_POST_RUN_CRITIC, bundle=bundle,
                           contract=POST_RUN_RESPONSE_CONTRACT, binding=binding,
                           max_output_tokens=max_output_tokens)
    completion = provider.complete(request)
    if not isinstance(completion, Completion):
        raise ProviderResponseInvalid(
            f"provider returned {type(completion).__name__}, not a Completion")
    honoured = check_binding_honoured(request, completion)
    if honoured.outcome != schemas.PASS:
        raise ProviderResponseInvalid(
            "provider did not honour the requested critic binding: "
            + "; ".join(honoured.reasons))
    violations = POST_RUN_RESPONSE_CONTRACT.validate(completion.data)
    if violations:
        raise ProviderResponseInvalid(
            "post-run classification does not satisfy its response contract: "
            + "; ".join(violations))
    waivers = find_gate_waiver_keys(dict(completion.data))
    if waivers:
        raise GateWaiverAttempt(waivers)

    classification = _parse_classification(completion.data)
    reconciliation = reconcile_classification(
        classification, verdict, manifest=manifest, facts=facts)
    return PostRunCritique(
        candidate_id=candidate_id,
        classification=classification,
        reconciliation=reconciliation,
        verdict_status=verdict.status,
        effect_resolution=verdict.effect_resolution,
        binding=binding,
        usage_tokens=completion.usage.total,
        independence=independence,
        decided_at=clock(),
    )


#: §19.4 names the bootstrap-knowledge kinds; `CONSTRAINT_COMPILED` is the one
#: whose subject is a compiled constraint/negative-ledger entry, which is exactly
#: what a durable lesson is. Named as a constant because `journal.KINDS` is a
#: closed vocabulary and a typo would be rejected at append time, not here.
LESSON_JOURNAL_KIND = "CONSTRAINT_COMPILED"


def lesson_journal_payload(lesson: DurableLesson, *, campaign_id: str,
                           candidate_id: str) -> dict:
    """Build the payload for a durable lesson. The CALLER appends it.

    This module holds no journal and writes no file. The kind is asserted against
    `journal.BOOTSTRAP_KNOWLEDGE_KINDS` here so a rename in `journal.py` fails at
    the seam rather than at 3am inside an append.
    """
    if not isinstance(lesson, DurableLesson):
        raise TypeError("lesson must be a DurableLesson")
    if LESSON_JOURNAL_KIND not in journal.BOOTSTRAP_KNOWLEDGE_KINDS:
        raise CriticError(
            f"{LESSON_JOURNAL_KIND!r} is no longer a journal kind; the lesson has "
            "nowhere durable to go and must not be dropped (invariant 7)")
    authority = lesson.entry.authority()
    if authority.outcome != schemas.PASS:
        raise CriticError(
            "refusing to journal a lesson without §19.3 standing: "
            + "; ".join(authority.reasons))
    payload = {
        "campaign_id": campaign_id,
        "candidate_id": candidate_id,
        **lesson.to_dict(),
    }
    schemas.canonical_json(payload)
    return payload


def critic_cost(critique: Any) -> RealizedCost:
    """The controller-token cost of one critique, for `planner.attribute_cost`.

    Critic tokens land on the PROPOSAL's `realized_cost`, not on a separate
    ledger: §12's zero-yield row asks what a proposal cost, and a proposal that
    was drafted once and critiqued three times cost all four calls.
    """
    tokens = getattr(critique, "usage_tokens", None)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise TypeError("critique must carry a non-negative int `usage_tokens`")
    return RealizedCost(controller_tokens=tokens)
