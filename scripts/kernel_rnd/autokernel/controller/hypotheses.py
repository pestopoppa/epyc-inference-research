"""hypotheses.py — the operator hypothesis channel and still-open tracking (§8.4.0, AK-D38).

WHY THIS MODULE EXISTS
----------------------
Two halves of one mechanism, and neither works without the other.

**1. The operator channel.** The operator sees things the profile does not. Without a
channel, that steering arrives out-of-band — as a standing instruction with no
falsifier, no grade and no resolution record. §8.4.0 gives it a channel and makes it
*safe by grading*: a hypothesis is stated WITH a falsifier ("if a current wall-share
map shows the cluster under 20% I am wrong"), enters at `design_prior` evidence grade,
and **can never be promoted by its origin**. §19.0 rule 4 already forbids upgrading
evidence on import, and an operator hunch is exactly the input most likely to be
treated as settled because of who said it. It is a proposal SOURCE, never an
authority: it faces the pre-run critic unchanged, obeys every §8.4 rejection
condition, and is subject to the do-not-repeat ledger like any other proposal.
Authorship is not new evidence (AK-D38).

**2. Still-open tracking, for every hypothesis regardless of origin.** AutoKernel had
the proposal `hypothesis` field (`schemas.validate_proposal`) and nothing that tracked
it, so a hypothesis evaporated the moment its proposal was dispositioned — *including*
when that proposal failed for an unrelated reason. That is the failure mode that
leaves a question feeling "already tried" with no receipt. The sibling loop does not
have this either, verified against source rather than assumed (§8.4.0 correction,
2026-08-03): `ExperimentJournal.unfalsified_hypotheses()` is a five-trial *recency
window*, its own docstring calls resolution-checking *"intentionally minimal … presence
of the falsifier string only"*, nothing marks a hypothesis resolved, the still-open
block is stagnation-gated, and the falsifier defaults to an empty string.

So, here: every hypothesis carries a falsifier, stays open until resolved
confirmed / refuted / inconclusive **with the evidence that resolved it**, and the open
set is re-surfaced into every planning round.

THE GRADING IS THE SAFETY PROPERTY, AND IT IS STRUCTURAL
--------------------------------------------------------
There is no code path in which origin raises grade, and the absence is checked rather
than intended:

* `Hypothesis` has **no `evidence_grade` field** — it is a read-only property returning
  `entry_grade(origin)`, and `entry_grade` is total over `ORIGINS` and constant. A new
  origin that forgot to be `design_prior` cannot exist, because the function does not
  branch on origin at all; `audit_no_origin_grade_promotion()` proves both facts from
  the objects.
* The operator store **refuses an `evidence_grade` key outright** (`_REFUSED_ENTRY_KEYS`).
  A store that could state its own grade is a store that can launder a hunch into a
  measured fact by typing one word.
* A resolution's `evidence_grade` belongs to the *evidence*, lives on
  `ResolutionEvidence`, and never touches the hypothesis's entry grade. The rendered
  planner block names the two separately (`entry_evidence_grade` vs
  `resolution_evidence_grade`) so no reader can conflate them.

WHAT THIS MODULE REFUSES TO BECOME (§8.4.0 "What it must not become")
---------------------------------------------------------------------
* **Not a queue-jumping mechanism.** There is no priority, rank, weight or boost field
  anywhere in this module, and `still_open()` orders by the sequence in which questions
  were opened — a stable order, not a ranking.
* **Not a bypass.** Nothing here waives a §8.4 rejection condition, a wall-share
  ceiling, or a correctness gate. `find_authority_flavoured_keys()` is run over every
  operator entry and over every rendered block, so an authority flag added later fails
  at load instead of quietly becoming load-bearing.
* **Not a route to mark something resolved without evidence.** `resolve()` demands
  evidence references, an observation stated against the falsifier, and an explicit
  declaration that the evidence bears on that falsifier. A proposal disposition is an
  ATTEMPT (`note_attempt()`), and an attempt never closes a question — which is the
  whole point of the second half of this mechanism.

A MALFORMED STORE RAISES
------------------------
`OperatorHypothesisStore.load()` never degrades to an empty list. An empty list is a
statement — *"the operator has no open hypotheses"* — and the planner acts on it. A
truncated file, a bad mount, a JSON typo or a missing falsifier must therefore be
distinguishable from that statement, and the only distinguishable outcome is a refusal.
"No operator channel is configured" is expressed by passing no store at all.

SUBSTRATE NOTE — where a hypothesis event lands
-----------------------------------------------
`journal.KINDS` is a CLOSED vocabulary with no hypothesis kind, and `journal.py` is
outside this task's write scope. Events therefore land in `HypothesisLedger`: the same
discipline as `state_machine.TransitionLedger` (O_APPEND, fsync per record, torn tail
discarded and reported, strictly increasing seq), written under `Journal.write_lock()`
so its order is total with respect to journal appends. Co-opting a §19.4
bootstrap-knowledge kind (`PRIOR_ATOMIZED`, `PRIOR_SUPERSEDED`) was rejected for the
reason `state_machine` rejected co-opting `STOP_STATE`: those kinds name the bootstrap
corpus, other planes fold them, and a second meaning makes the first one unreadable.
**Follow-up: add `KIND_HYPOTHESIS_OPENED` / `_ATTEMPTED` / `_RESOLVED` / `_REOPENED` to
`journal.py`; `HypothesisRecorder` is the seam that makes the swap a wiring change with
no contract change here.**

NO CACHED STATE
---------------
`HypothesisTracker` holds no folded state. Every query re-reads the ledger from disk,
for the same reason `state_machine` re-reads the control latch: a cached copy is a copy
that can be written back over the record. This is O(ledger) per call and is the right
price at this scale; it is noted here rather than silently optimized.

This module runs NO inference, NO benchmark and NO build; it starts, stops and signals
NO process; it calls NO model. It writes exactly one file, the ledger, under the
controller root it is given.

Governing instrument: `measurement/protocols/kernel-research.md` (P-AK-SEARCH-1,
RATIFIED 2026-08-03). Nothing here ranks, banks, composes or contributes to readiness.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, Sequence

from .. import journal, schemas
from .state_machine import ControllerError

__all__ = [
    # errors
    "HypothesisError", "HypothesisStoreError", "FalsifierMissing",
    "HypothesisLedgerCorruption", "UnknownHypothesis", "HypothesisNotOpen",
    "HypothesisAlreadyTracked", "QuestionRewritten", "ResolutionEvidenceMissing",
    # vocabulary
    "ORIGIN_OPERATOR", "ORIGIN_PLANNER", "ORIGIN_CRITIC", "ORIGIN_CONTROLLER",
    "ORIGIN_IMPORT", "ORIGINS",
    "GRADE_DESIGN_PRIOR", "GRADE_OBSERVATION", "GRADE_SOURCE_VERIFIED",
    "GRADE_IMPORTED_CLAIM", "GRADE_PROTOCOL_BOUND", "EVIDENCE_GRADES", "ENTRY_GRADE",
    "STATUS_OPEN", "RESOLUTION_CONFIRMED", "RESOLUTION_REFUTED",
    "RESOLUTION_INCONCLUSIVE", "RESOLUTIONS", "STATUSES",
    "EVENT_OPENED", "EVENT_ATTEMPTED", "EVENT_RESOLVED", "EVENT_REOPENED",
    "LEDGER_EVENT_KINDS",
    "MATCH_CLASS_HARD_CONSTRAINT", "MATCH_CLASS_MATCHED_NEGATIVE",
    "MATCH_CLASS_CONDITIONAL_NEGATIVE", "MATCH_CLASS_CONFOUNDED_RESULT",
    "MATCH_CLASS_SUPERSEDED_FACT", "MATCH_CLASS_LOW_VALUE", "MATCH_CLASSES",
    "REJECTING_MATCH_CLASSES",
    "STORE_SCHEMA", "ROUND_BLOCK_SCHEMA", "LEDGER_FILENAME",
    # types
    "Hypothesis", "Attempt", "ResolutionEvidence", "TrackedHypothesis",
    "LedgerEvent", "LedgerRead", "IntakeReport", "LedgerMatch",
    # seams
    "HypothesisRecorder", "DoNotRepeatLedger",
    # implementations
    "OperatorHypothesisStore", "HypothesisLedger", "JournalOrderedRecorder",
    "HypothesisTracker",
    # pure functions and checks
    "entry_grade", "fold_ledger", "check_do_not_repeat",
    "audit_no_origin_grade_promotion",
]


# =============================================================================
# Errors — every one is a refusal. None of them has a degraded-result sibling.
# =============================================================================

class HypothesisError(ControllerError):
    """Base for every refusal here.

    It extends `state_machine.ControllerError` so a driver can catch the CONTROLLER
    PLANE rather than one module of it; a loop that has to enumerate module bases
    eventually forgets one, and the one it forgets is the one that matters.
    """


class HypothesisStoreError(HypothesisError):
    """The operator store could not be read as a store.

    Never an empty list: empty means *"the operator has no hypotheses"*, the planner
    acts on that, and a mount failure must not be able to say it.
    """


class FalsifierMissing(HypothesisStoreError):
    """A hypothesis was stated without a falsifier.

    The falsifier is what makes it a resolvable hypothesis rather than a standing
    instruction. This is the refusal §8.4.0 turns on, so it has its own type.
    """


class HypothesisLedgerCorruption(HypothesisError):
    """The ledger does not describe one coherent history."""


class UnknownHypothesis(HypothesisError):
    """An operation named a hypothesis that was never opened."""


class HypothesisNotOpen(HypothesisError):
    """Resolve on a resolved question, or reopen on an open one."""


class HypothesisAlreadyTracked(HypothesisError):
    """An id that is already in the ledger was opened again."""


class QuestionRewritten(HypothesisError):
    """A tracked hypothesis's statement or falsifier changed under its own id.

    Rewriting a falsifier after evidence exists is how any hypothesis becomes
    "confirmed". A new question gets a new id; the old one stays open with its
    original falsifier until evidence closes it.
    """


class ResolutionEvidenceMissing(HypothesisError):
    """A resolution that does not carry the evidence that resolved it."""


# =============================================================================
# Vocabulary
# =============================================================================

# Mirrors `state_machine.STOP_REQUEST_ORIGINS` and adds `import`: §19.5 imports
# historical research as typed legacy events, and an "unresolved contradiction"
# prior (§19.1) is a hypothesis with an author who is not in the room.
ORIGIN_OPERATOR = "operator"
ORIGIN_PLANNER = "planner"
ORIGIN_CRITIC = "critic"
ORIGIN_CONTROLLER = "controller"
ORIGIN_IMPORT = "import"

ORIGINS = frozenset({
    ORIGIN_OPERATOR, ORIGIN_PLANNER, ORIGIN_CRITIC, ORIGIN_CONTROLLER, ORIGIN_IMPORT,
})

# §19.1's `evidence_grade` vocabulary, verbatim.
GRADE_DESIGN_PRIOR = "design_prior"
GRADE_OBSERVATION = "observation"
GRADE_SOURCE_VERIFIED = "source_verified"
GRADE_IMPORTED_CLAIM = "imported_claim"
GRADE_PROTOCOL_BOUND = "protocol_bound"

EVIDENCE_GRADES = frozenset({
    GRADE_DESIGN_PRIOR, GRADE_OBSERVATION, GRADE_SOURCE_VERIFIED,
    GRADE_IMPORTED_CLAIM, GRADE_PROTOCOL_BOUND,
})

#: The grade EVERY hypothesis enters at, whoever stated it (§8.4.0, AK-D38, §19.0
#: rule 4). `design_prior` means "worth considering", not "probably true" (§19.1).
ENTRY_GRADE = GRADE_DESIGN_PRIOR

STATUS_OPEN = "open"
RESOLUTION_CONFIRMED = "confirmed"
RESOLUTION_REFUTED = "refuted"
RESOLUTION_INCONCLUSIVE = "inconclusive"

#: `inconclusive` is a real resolution and NOT a synonym for open: the experiment ran
#: and did not resolve the falsifier. It is reopenable on new evidence, which is what
#: stops it from becoming a permanent silent close. (Same distinction
#: `schemas.EVENT_STATUSES` draws between `inconclusive` and `invalid`.)
RESOLUTIONS = frozenset({
    RESOLUTION_CONFIRMED, RESOLUTION_REFUTED, RESOLUTION_INCONCLUSIVE,
})
STATUSES = frozenset({STATUS_OPEN}) | RESOLUTIONS

EVENT_OPENED = "HYPOTHESIS_OPENED"
EVENT_ATTEMPTED = "HYPOTHESIS_ATTEMPTED"
EVENT_RESOLVED = "HYPOTHESIS_RESOLVED"
EVENT_REOPENED = "HYPOTHESIS_REOPENED"

LEDGER_EVENT_KINDS = (EVENT_OPENED, EVENT_ATTEMPTED, EVENT_RESOLVED, EVENT_REOPENED)

# §19.2's do-not-repeat / constraint ledger classes. This module does not BUILD that
# ledger (the memory-update plane owns it); it consumes matches and disposes them.
MATCH_CLASS_HARD_CONSTRAINT = "HARD_CONSTRAINT"
MATCH_CLASS_MATCHED_NEGATIVE = "MATCHED_NEGATIVE"
MATCH_CLASS_CONDITIONAL_NEGATIVE = "CONDITIONAL_NEGATIVE"
MATCH_CLASS_CONFOUNDED_RESULT = "CONFOUNDED_RESULT"
MATCH_CLASS_SUPERSEDED_FACT = "SUPERSEDED_FACT"
MATCH_CLASS_LOW_VALUE = "LOW_VALUE"

MATCH_CLASSES = frozenset({
    MATCH_CLASS_HARD_CONSTRAINT, MATCH_CLASS_MATCHED_NEGATIVE,
    MATCH_CLASS_CONDITIONAL_NEGATIVE, MATCH_CLASS_CONFOUNDED_RESULT,
    MATCH_CLASS_SUPERSEDED_FACT, MATCH_CLASS_LOW_VALUE,
})

#: The two classes whose §19.2 planner behaviour is "reject". The other four exclude
#: cells, demand a repaired experiment, regenerate from current source, or
#: deprioritize — all of which are advisory and none of which closes the question.
REJECTING_MATCH_CLASSES = frozenset({
    MATCH_CLASS_HARD_CONSTRAINT, MATCH_CLASS_MATCHED_NEGATIVE,
})

STORE_SCHEMA = "epyc.autokernel.operator_hypotheses.v1"
ROUND_BLOCK_SCHEMA = "epyc.autokernel.hypothesis_round_block.v1"

LEDGER_FILENAME = "hypotheses.jsonl"

_HYPOTHESIS_ID_RE = re.compile(r"^akh-[A-Za-z0-9][A-Za-z0-9._-]*$")

#: A falsifier that is one of these is an empty string wearing a hat. AutoPilot's
#: falsifier defaulted to `""` and nothing objected; these are the strings that
#: reproduce that state while passing a non-empty check.
_PLACEHOLDER_FALSIFIERS = frozenset({
    "", "-", "--", "n/a", "n.a.", "na", "none", "null", "nil", "tbd", "todo", "tbc",
    "?", "??", "???", "x", "unknown", "unclear", "unstated", "not applicable",
    "see above", "as above", "pending", "n/a — see statement",
})


def entry_grade(origin: str) -> str:
    """The evidence grade a hypothesis ENTERS at. Constant, by construction.

    The parameter exists so the constancy is ENUMERABLE — `audit_no_origin_grade_
    promotion()` and the suite both range over `ORIGINS` and assert one value comes
    back — and for no other reason: there is no branch on `origin` in this body, so a
    future origin cannot be given a higher grade without deleting this line, which the
    audit and the tests would both catch.

    §8.4.0: *"An operator hypothesis enters at `design_prior` evidence grade and can
    never be promoted by its origin"*. §19.0 rule 4: never upgrade evidence on import.
    """
    if origin not in ORIGINS:
        raise ValueError(f"origin: {origin!r} is not a declared origin {sorted(ORIGINS)}")
    return ENTRY_GRADE


# =============================================================================
# Small helpers — the durable-write pair is deliberately local
#
# `journal.py` and `state_machine.py` each carry their own copy of these four lines.
# Importing another module's private helper would couple two planes through a name
# neither exports; a four-line fsync wrapper is the cheaper duplication.
# =============================================================================

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_text(value: Any, what: str, *, error=ValueError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{what} is required and must be a non-empty string")
    return value


def _require_refs(value: Any, what: str, *, error=ValueError) -> tuple:
    """A non-empty tuple of non-empty strings. An empty ref list is no receipt."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error(f"{what} must be a sequence of reference strings")
    refs = tuple(value)
    if not refs:
        raise error(f"{what} must name at least one reference; an empty list is not evidence")
    for index, ref in enumerate(refs):
        if not isinstance(ref, str) or not ref.strip():
            raise error(f"{what}[{index}] must be a non-empty reference string")
    return refs


def _refs_from_record(value: Any, what: str, *, error) -> tuple:
    """Refs read back OUT of a record, before `_require_refs` sees them.

    `tuple("akj-1")` is `('a','k','j','-','1')` — five perfectly valid-looking
    references — so a `from_dict` that normalises with `tuple(...)` hands
    `_require_refs` something that passes every check it makes, and the guard that
    exists precisely to refuse a bare string never fires. A record whose ref list is a
    string (or a mapping, whose `tuple()` is its KEYS) is refused here instead.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error(
            f"{what} must be a LIST of reference strings, not "
            f"{type(value).__name__}; a bare string would be exploded into one "
            "'reference' per character and would then satisfy every later check"
        )
    return tuple(value)


# =============================================================================
# The hypothesis itself
# =============================================================================

@dataclass(frozen=True)
class Hypothesis:
    """One question, with the falsifier that makes it resolvable.

    NOTE WHAT IS ABSENT: there is no `evidence_grade` field, no `priority`, no `rank`,
    and no `status`. Grade is derived (`entry_grade`), ranking does not exist here at
    all (§8.4.0: not a queue-jumping mechanism), and status lives in the ledger because
    a status that travels with the statement is a status somebody can edit.
    """

    hypothesis_id: str
    statement: str
    falsifier: str
    origin: str
    author: str
    regime: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        if not _HYPOTHESIS_ID_RE.match(self.hypothesis_id):
            raise ValueError(
                f"hypothesis_id: {self.hypothesis_id!r} must start with 'akh-' "
                "(the id prefixes are the record's family, per §7)"
            )
        _require_text(self.statement, "statement")
        self._check_falsifier()
        if self.origin not in ORIGINS:
            raise ValueError(f"origin: {self.origin!r} not in {sorted(ORIGINS)}")
        _require_text(self.author, "author")
        for name in ("regime", "source"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            # Canonicalizability is checked at construction, not at write time: a
            # hypothesis that cannot be serialized must fail before it is tracked,
            # never halfway through being recorded.
            schemas.canonical_json(dict(value))
        flagged = schemas.find_authority_flavoured_keys(
            {"regime": dict(self.regime), "source": dict(self.source)}
        )
        if flagged:
            raise ValueError(
                f"authority-flavoured keys {flagged} in a hypothesis; §8.4.0 — a "
                "hypothesis is a proposal SOURCE, never an authority, and it carries "
                "no freeze/cutover/promotion flag (§1.3, invariant 5)"
            )
        if self.created_at is not None:
            _require_text(self.created_at, "created_at")

    def _check_falsifier(self) -> None:
        """§8.4.0: a ONE-LINE predicted outcome whose absence invalidates it."""
        if not isinstance(self.falsifier, str) or not self.falsifier.strip():
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier is required and must be a non-empty "
                "string — a hypothesis without a falsifier is a standing instruction, "
                "not a resolvable question (§8.4.0)"
            )
        collapsed = self.falsifier.strip().lower()
        if collapsed in _PLACEHOLDER_FALSIFIERS:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier {self.falsifier!r} is a placeholder; "
                "the predecessor loop's falsifier defaulted to the empty string and "
                "nothing ever objected"
            )
        if "\n" in self.falsifier or "\r" in self.falsifier:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier must be ONE LINE — a predicted "
                "outcome whose absence invalidates the hypothesis, not a paragraph "
                "of reasoning (§8.4.0)"
            )
        if collapsed == self.statement.strip().lower():
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier restates the hypothesis; it must "
                "predict an OUTCOME that could fail to appear"
            )

    @property
    def evidence_grade(self) -> str:
        """Always `design_prior`. A PROPERTY, deliberately: there is no field to set.

        This is the §8.4.0 safety property in its structural form — the object cannot
        express a promoted grade, so no code path can produce one.
        """
        return entry_grade(self.origin)

    @property
    def fingerprint(self) -> str:
        """Identity of the QUESTION: statement + falsifier + regime.

        Two entries with one id and different fingerprints are two questions, and the
        second is refused rather than applied (`QuestionRewritten`).
        """
        return schemas.content_hash({
            "statement": self.statement,
            "falsifier": self.falsifier,
            "regime": dict(self.regime),
        })

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "falsifier": self.falsifier,
            "origin": self.origin,
            "author": self.author,
            "regime": dict(self.regime),
            "source": dict(self.source),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Hypothesis":
        if not isinstance(obj, Mapping):
            raise TypeError("hypothesis record must be a mapping")
        missing = sorted(
            {"hypothesis_id", "statement", "falsifier", "origin", "author"} - set(obj)
        )
        if missing:
            raise ValueError(f"hypothesis record is missing {missing}")
        return Hypothesis(
            hypothesis_id=obj["hypothesis_id"],
            statement=obj["statement"],
            falsifier=obj["falsifier"],
            origin=obj["origin"],
            author=obj["author"],
            regime=dict(obj.get("regime") or {}),
            source=dict(obj.get("source") or {}),
            created_at=obj.get("created_at"),
        )


# =============================================================================
# Attempts and resolutions
# =============================================================================

@dataclass(frozen=True)
class Attempt:
    """A proposal that was dispositioned while this question was open.

    An attempt NEVER resolves anything, and that is the entire point: AutoKernel's
    hypothesis used to evaporate when its proposal was dispositioned, *including when
    that proposal failed for an unrelated reason* — a build break, a skipped proposal,
    a voided window. Recording the attempt is what turns "this feels already tried"
    into a receipt that says what was tried and what it did or did not bear on.
    """

    hypothesis_id: str
    proposal_id: str
    disposition: str
    bears_on_falsifier: bool
    note: str
    refs: tuple = ()
    at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        _require_text(self.proposal_id, "proposal_id")
        _require_text(self.disposition, "disposition")
        _require_text(self.note, "note")
        if not isinstance(self.bears_on_falsifier, bool):
            raise TypeError(
                "bears_on_falsifier must be an explicit bool — 'we did not consider "
                "whether this bore on the falsifier' is not a value this field takes"
            )
        if isinstance(self.refs, (str, bytes)) or not isinstance(self.refs, Sequence):
            raise TypeError("refs must be a sequence of reference strings")
        for index, ref in enumerate(self.refs):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"refs[{index}] must be a non-empty reference string")

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "proposal_id": self.proposal_id,
            "disposition": self.disposition,
            "bears_on_falsifier": self.bears_on_falsifier,
            "note": self.note,
            "refs": list(self.refs),
            "at": self.at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Attempt":
        if not isinstance(obj, Mapping):
            raise TypeError("attempt record must be a mapping")
        missing = sorted(
            {"hypothesis_id", "proposal_id", "disposition", "bears_on_falsifier", "note"}
            - set(obj)
        )
        if missing:
            raise ValueError(f"attempt record is missing {missing}")
        return Attempt(
            hypothesis_id=obj["hypothesis_id"],
            proposal_id=obj["proposal_id"],
            disposition=obj["disposition"],
            bears_on_falsifier=obj["bears_on_falsifier"],
            note=obj["note"],
            refs=_refs_from_record(obj.get("refs"), "refs", error=TypeError),
            at=obj.get("at"),
        )


@dataclass(frozen=True)
class ResolutionEvidence:
    """What closed a question, and what was observed against its falsifier.

    `evidence_grade` here is the grade of the EVIDENCE. It is never written back onto
    the hypothesis, whose entry grade is `design_prior` forever: an operator hypothesis
    that the loop confirms is confirmed BY THE EVIDENCE, and the record says which
    evidence, so a later reader can check it.
    """

    outcome: str
    evidence_grade: str
    evidence_refs: tuple
    falsifier_observed: str
    bears_on_falsifier: bool
    resolved_by: str
    at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in RESOLUTIONS:
            raise ValueError(f"outcome: {self.outcome!r} not in {sorted(RESOLUTIONS)}")
        if self.evidence_grade not in EVIDENCE_GRADES:
            raise ValueError(
                f"evidence_grade: {self.evidence_grade!r} not in {sorted(EVIDENCE_GRADES)}"
            )
        object.__setattr__(self, "evidence_refs", _require_refs(
            self.evidence_refs, "evidence_refs", error=ResolutionEvidenceMissing
        ))
        _require_text(
            self.falsifier_observed, "falsifier_observed",
            error=ResolutionEvidenceMissing,
        )
        if self.bears_on_falsifier is not True:
            # The one rule that keeps §8.4.0's second half honest. Evidence that does
            # not bear on the falsifier is an ATTEMPT; routing it here would close a
            # question on a failure that had nothing to do with it, which is the exact
            # defect this module exists to remove.
            raise ResolutionEvidenceMissing(
                "bears_on_falsifier must be exactly True to RESOLVE a hypothesis; "
                "evidence that does not bear on the falsifier is an attempt "
                "(note_attempt), and an attempt leaves the question open"
            )
        _require_text(self.resolved_by, "resolved_by")

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "evidence_grade": self.evidence_grade,
            "evidence_refs": list(self.evidence_refs),
            "falsifier_observed": self.falsifier_observed,
            "bears_on_falsifier": self.bears_on_falsifier,
            "resolved_by": self.resolved_by,
            "at": self.at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "ResolutionEvidence":
        if not isinstance(obj, Mapping):
            raise TypeError("resolution record must be a mapping")
        missing = sorted({
            "outcome", "evidence_grade", "evidence_refs", "falsifier_observed",
            "bears_on_falsifier", "resolved_by",
        } - set(obj))
        if missing:
            raise ValueError(f"resolution record is missing {missing}")
        return ResolutionEvidence(
            outcome=obj["outcome"],
            evidence_grade=obj["evidence_grade"],
            evidence_refs=_refs_from_record(
                obj["evidence_refs"], "evidence_refs", error=ResolutionEvidenceMissing
            ),
            falsifier_observed=obj["falsifier_observed"],
            bears_on_falsifier=obj["bears_on_falsifier"],
            resolved_by=obj["resolved_by"],
            at=obj.get("at"),
        )


# =============================================================================
# The operator store — an intake, never a source of truth about resolution
# =============================================================================

#: Keys the operator may write. This set is exactly what is CARRIED — anything else is
#: refused, because a key this loader silently ignores is a key the operator believes
#: had an effect.
_ALLOWED_ENTRY_KEYS = frozenset({
    "hypothesis_id", "statement", "falsifier", "author", "regime", "created_at",
})

#: Keys refused with a POINTED message. Each one is a specific laundering route:
#: stating your own grade, stating your own resolution, or stating an origin other
#: than the one the file's existence already establishes.
_REFUSED_ENTRY_KEYS: Mapping[str, str] = {
    "evidence_grade": (
        "evidence grade is DERIVED and is always 'design_prior' for a hypothesis "
        "(§8.4.0, AK-D38, §19.0 rule 4); a store that could state its own grade is a "
        "store that can launder a hunch into a measured fact"
    ),
    "grade": "see 'evidence_grade': grade is derived, never stated",
    "status": (
        "status lives in the ledger, not in the store; a hypothesis is resolved by "
        "evidence (resolve()), never by editing a file"
    ),
    "resolution": "see 'status': resolution requires the evidence that resolved it",
    "resolved": "see 'status': resolution requires the evidence that resolved it",
    "outcome": "see 'status': resolution requires the evidence that resolved it",
    "origin": (
        "every entry in the operator store has origin 'operator' by construction; "
        "stating another origin would let the channel relabel its own provenance"
    ),
    "priority": (
        "there is no priority anywhere in this mechanism — §8.4.0: an operator "
        "hypothesis is not a queue-jumping mechanism"
    ),
    "rank": "see 'priority': this mechanism has no ranking",
    "weight": "see 'priority': this mechanism has no ranking",
    "notes": (
        "a hypothesis is its statement and its falsifier; there is no third prose "
        "field, because free text beside a tracked question is exactly the planner "
        "narrative invariant 20 keeps out of a later planning context"
    ),
}


class OperatorHypothesisStore:
    """The operator-editable file of stated hypotheses.

    JSON rather than YAML on purpose. Every other durable file in this package is
    JSON, and YAML's implicit typing is a hazard for exactly the field that carries the
    safety property: a falsifier reading `no` or `NO` becomes the boolean `False`, and
    an unquoted `1.5-3` becomes a string that no longer says what it said. A falsifier
    that silently changes type is a falsifier that silently stops being one.

    Shape::

        {
          "schema": "epyc.autokernel.operator_hypotheses.v1",
          "hypotheses": [
            {
              "hypothesis_id": "akh-g15-elementwise-fusion",
              "statement": "G15's elementwise/norm cluster is where the B=128 decode time is, and fusing it lands >= 15%",
              "falsifier": "a current wall-share map shows the cluster under 20%",
              "author": "operator",
              "regime": {"backend": "llama_gpu", "phase": "decode", "batch_band": "b128"}
            }
          ]
        }

    An empty `hypotheses` list is VALID and means the operator has none. That is
    precisely why every malformed input raises instead.
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(_require_text(path, "store path"))

    def exists(self) -> bool:
        """Whether the store file is there.

        `load()` RAISES on absence rather than returning nothing, so a caller that
        legitimately has no operator channel expresses that by passing no store at all
        — not by pointing at a path that may or may not have been mounted.
        """
        return os.path.exists(self.path)

    def read_bytes(self) -> bytes:
        try:
            with open(self.path, "rb") as handle:
                return handle.read()
        except FileNotFoundError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store does not exist. An absent "
                "store is not an empty one — pass no store at all to declare that "
                "this campaign has no operator channel"
            ) from exc
        except OSError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store could not be read ({exc})"
            ) from exc

    def _digest(self, raw: bytes) -> str:
        """SHA-256 of the store's bytes, or a REFUSAL if they are not UTF-8.

        `errors="surrogateescape"` used to be here, which made this the one public
        store method whose failure escaped the module's error contract: the lone
        surrogates it produces are legal in a `str` but cannot be re-encoded, so
        `content_hash` raised a bare `UnicodeEncodeError` — not a `HypothesisStoreError`,
        not even a `ControllerError`, so a driver catching the controller plane died on
        it. A store whose bytes are not UTF-8 is an unreadable store, and this module
        has exactly one way to say that.
        """
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store is not valid UTF-8 ({exc}); "
                "an undecodable store is unreadable, not empty"
            ) from exc
        return schemas.content_hash({"bytes": text})

    def content_sha256(self) -> str:
        return self._digest(self.read_bytes())

    def load(self) -> tuple:
        """Every stated hypothesis, or a refusal. NEVER a degraded empty list."""
        return self.load_with_digest()[0]

    def load_with_digest(self) -> tuple:
        """`(hypotheses, store_sha256)` from ONE read of the file.

        One read, not two: a digest taken from a second read describes a file that may
        no longer be the one that was parsed, and the digest exists precisely so a
        later reader can tell which bytes a tracked hypothesis came from.
        """
        raw = self.read_bytes()
        if not raw.strip():
            raise HypothesisStoreError(
                f"{self.path}: store file is empty. An unreadable store is not an "
                "absent one, and an absent one is not 'the operator has no "
                "hypotheses' — that statement is written as an empty 'hypotheses' list"
            )
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HypothesisStoreError(
                f"{self.path}: unparseable operator hypothesis store: {exc}"
            ) from exc
        if not isinstance(obj, Mapping):
            raise HypothesisStoreError(
                f"{self.path}: store must be a JSON object with 'schema' and "
                f"'hypotheses', got {type(obj).__name__}"
            )
        if obj.get("schema") != STORE_SCHEMA:
            raise HypothesisStoreError(
                f"{self.path}: schema is {obj.get('schema')!r}, expected "
                f"{STORE_SCHEMA!r}; the schema string is the file's identity, not "
                "metadata beside it"
            )
        entries = obj.get("hypotheses")
        if entries is None:
            raise HypothesisStoreError(
                f"{self.path}: 'hypotheses' key is absent. Write [] to say the "
                "operator has none; absence is a truncated file"
            )
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise HypothesisStoreError(
                f"{self.path}: 'hypotheses' must be a list, got "
                f"{type(entries).__name__}"
            )
        unknown_top = sorted(set(obj) - {"schema", "hypotheses"})
        if unknown_top:
            raise HypothesisStoreError(
                f"{self.path}: unknown top-level keys {unknown_top}; a key this "
                "loader ignores is a key the operator believes had an effect"
            )

        store_sha = self._digest(raw)
        loaded: list = []
        seen: dict = {}
        for index, entry in enumerate(entries):
            hypothesis = self._load_entry(entry, index, store_sha)
            if hypothesis.hypothesis_id in seen:
                raise HypothesisStoreError(
                    f"{self.path}: hypothesis_id {hypothesis.hypothesis_id!r} appears "
                    f"at entries {seen[hypothesis.hypothesis_id]} and {index}; one id "
                    "is one question"
                )
            seen[hypothesis.hypothesis_id] = index
            loaded.append(hypothesis)
        return tuple(loaded), store_sha

    def _load_entry(self, entry: Any, index: int, store_sha: str) -> Hypothesis:
        where = f"{self.path}[{index}]"
        if not isinstance(entry, Mapping):
            raise HypothesisStoreError(
                f"{where}: each hypothesis must be an object, got "
                f"{type(entry).__name__}"
            )
        for key, why in _REFUSED_ENTRY_KEYS.items():
            if key in entry:
                raise HypothesisStoreError(f"{where}: key {key!r} is refused — {why}")
        unknown = sorted(set(entry) - _ALLOWED_ENTRY_KEYS)
        if unknown:
            raise HypothesisStoreError(
                f"{where}: unknown keys {unknown}; allowed keys are "
                f"{sorted(_ALLOWED_ENTRY_KEYS)}"
            )
        flagged = schemas.find_authority_flavoured_keys(dict(entry))
        if flagged:
            raise HypothesisStoreError(
                f"{where}: authority-flavoured keys {flagged}. §8.4.0 — the channel is "
                "steering WITHOUT authority; a hypothesis is a proposal source and "
                "carries no freeze, cutover, promotion or ratification flag"
            )

        for key in ("hypothesis_id", "statement"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise HypothesisStoreError(f"{where}: {key} is required and non-empty")
        if "falsifier" not in entry:
            raise FalsifierMissing(
                f"{where} ({entry.get('hypothesis_id')!r}): no falsifier. §8.4.0 — a "
                "hypothesis is stated WITH a one-line predicted outcome whose absence "
                "invalidates it; without one it is a standing instruction, and the "
                "loop has no way to ever resolve it"
            )
        regime = entry.get("regime", {})
        if not isinstance(regime, Mapping):
            raise HypothesisStoreError(
                f"{where}: regime must be an object of match dimensions "
                f"(§19.2 — 'do not repeat' without regime identity is dangerous)"
            )
        author = entry.get("author", ORIGIN_OPERATOR)
        if not isinstance(author, str) or not author.strip():
            raise HypothesisStoreError(f"{where}: author must be a non-empty string")

        source = {
            "kind": "operator_store",
            "path": self.path,
            "entry_index": index,
            "store_sha256": store_sha,
        }
        try:
            return Hypothesis(
                hypothesis_id=entry["hypothesis_id"],
                statement=entry["statement"],
                falsifier=entry["falsifier"],
                origin=ORIGIN_OPERATOR,
                author=author,
                regime=dict(regime),
                source=source,
                created_at=entry.get("created_at"),
            )
        except FalsifierMissing as exc:
            raise FalsifierMissing(f"{where}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HypothesisStoreError(f"{where}: {exc}") from exc


# =============================================================================
# The ledger
# =============================================================================

@dataclass(frozen=True)
class LedgerEvent:
    """One append-only line. `seq` is the ordering truth; `at` is diagnostic."""

    seq: int
    kind: str
    hypothesis_id: str
    at: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ValueError("seq must be a positive int")
        if self.kind not in LEDGER_EVENT_KINDS:
            raise ValueError(
                f"kind: {self.kind!r} not in {list(LEDGER_EVENT_KINDS)}; the ledger "
                "vocabulary is closed"
            )
        _require_text(self.hypothesis_id, "hypothesis_id")
        _require_text(self.at, "at")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        schemas.canonical_json(dict(self.payload))

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "hypothesis_id": self.hypothesis_id,
            "at": self.at,
            "payload": dict(self.payload),
        }

    @property
    def receipt(self) -> str:
        return schemas.content_hash(self.to_dict())

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "LedgerEvent":
        if not isinstance(obj, Mapping):
            raise TypeError("ledger record must be a mapping")
        missing = sorted({"seq", "kind", "hypothesis_id", "at"} - set(obj))
        if missing:
            raise ValueError(f"ledger record is missing {missing}")
        return LedgerEvent(
            seq=obj["seq"],
            kind=obj["kind"],
            hypothesis_id=obj["hypothesis_id"],
            at=obj["at"],
            payload=dict(obj.get("payload") or {}),
        )


@dataclass(frozen=True)
class LedgerRead:
    events: tuple
    discarded_tail_bytes: int


class HypothesisLedger:
    """Append-only, fsynced, one JSON line per event.

    Same discipline as `state_machine.TransitionLedger` — a trailing fragment with no
    newline is a TORN APPEND (the process died mid-write), so the event never took
    effect and discarding it restores a position the record supports. The discarded
    byte count is REPORTED, never swallowed: invariant 7 says outcomes are durable, and
    a silently truncated ledger is a lost question.
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(_require_text(path, "ledger path"))

    def initialize(self) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(directory)

    def append(self, event: LedgerEvent) -> LedgerEvent:
        """Write and fsync one event. Returning means DURABLE.

        REFUSES to write onto a ledger that cannot be read back, and in particular onto
        a TORN TAIL. `read()` discards a trailing fragment because the record it
        describes never took effect — but an O_APPEND write lands immediately after
        those bytes, fusing the fragment and the new record into one unparseable line.
        That line is not a torn tail (it ends in a newline), so from the next read
        onward `read()` raises and EVERY question in the ledger becomes unreadable,
        permanently, from one write that looked like it succeeded. `journal.py` faces
        the same hazard and repairs the tail under the write lock before appending
        (`_repair_torn_tail_locked`); this module has no kind in which to receipt a
        discard, so it refuses and leaves the bytes for repair rather than destroying
        the record or silently dropping them.
        """
        if not isinstance(event, LedgerEvent):
            raise TypeError(f"event must be a LedgerEvent, got {type(event).__name__}")
        # Validates through the reader on purpose: "readable back" is defined in exactly
        # one place, the same discipline `HypothesisTracker._record` applies by
        # re-folding the candidate ledger before writing. O(ledger) per append, at the
        # same scale and for the same reason as the rest of this module.
        before = self.read()
        if before.discarded_tail_bytes:
            raise HypothesisLedgerCorruption(
                f"{self.path}: {before.discarded_tail_bytes} byte(s) of torn tail have "
                "not been repaired; appending would fuse them to this record and make "
                "the whole ledger unparseable. Truncate the unterminated trailing "
                "fragment (it describes an event that never took effect) and retry"
            )
        line = (schemas.canonical_json(event.to_dict()) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(fd, line)
            if written != len(line):  # pragma: no cover - short write on a regular file
                raise HypothesisLedgerCorruption(
                    f"{self.path}: short write ({written} of {len(line)} bytes)"
                )
            os.fsync(fd)
        finally:
            os.close(fd)
        return event

    def read(self) -> LedgerRead:
        if not os.path.exists(self.path):
            return LedgerRead((), 0)
        with open(self.path, "rb") as handle:
            data = handle.read()
        if not data:
            return LedgerRead((), 0)
        cut = data.rfind(b"\n")
        if cut == -1:
            return LedgerRead((), len(data))
        tail = len(data) - (cut + 1)
        # `data[:cut + 1]` always ends in the newline `cut` names, so `split` always
        # yields one trailing empty element that is a SEPARATOR ARTEFACT, not a line.
        # Dropping exactly that element is what lets an EMPTY line be refused: skipping
        # every empty element (the shape `state_machine.TransitionLedger.read()` still
        # has) silently discards a real blank line in the middle of the file while
        # `discarded_tail_bytes` stays 0, so a ledger that lost a record reads as intact.
        lines = data[: cut + 1].split(b"\n")[:-1]
        events: list = []
        for line_number, raw in enumerate(lines, start=1):
            if not raw.strip():
                raise HypothesisLedgerCorruption(
                    f"{self.path}:{line_number}: blank line — an append-only ledger has "
                    "no empty records, so a blank line is a lost or overwritten one"
                )
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HypothesisLedgerCorruption(
                    f"{self.path}:{line_number}: unparseable ledger line: {exc}"
                ) from exc
            try:
                events.append(LedgerEvent.from_dict(obj))
            except (TypeError, ValueError) as exc:
                raise HypothesisLedgerCorruption(
                    f"{self.path}:{line_number}: invalid ledger event: {exc}"
                ) from exc
        seqs = [e.seq for e in events]
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise HypothesisLedgerCorruption(
                f"{self.path}: event seq numbers are not strictly increasing: {seqs}"
            )
        return LedgerRead(tuple(events), tail)


class HypothesisRecorder(Protocol):
    """The seam every ledger event is recorded through.

    `record()` returns the event AS RECORDED and RAISES if it could not be made
    durable. Nothing observable changes until it returns, so a recorder that raises is
    an event that did not happen. This is the swap point for the day `journal.py`
    grows hypothesis kinds.
    """

    def record(self, event: LedgerEvent) -> LedgerEvent:
        ...


class JournalOrderedRecorder:
    """The shipped recorder: ledger append under the JOURNAL write lock.

    The lock is not for this file's own consistency — a single O_APPEND write does not
    need one. It is so the hypothesis ledger's order is TOTAL with respect to journal
    appends, which is what lets a later reader put "this evaluation event" and "this
    hypothesis was resolved by it" in one order rather than two.
    """

    __slots__ = ("_journal", "_ledger")

    def __init__(self, journal_: journal.Journal, ledger: HypothesisLedger) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("ledger must be a HypothesisLedger")
        self._journal = journal_
        self._ledger = ledger

    def record(self, event: LedgerEvent) -> LedgerEvent:
        with self._journal.write_lock():
            return self._ledger.append(event)


# =============================================================================
# The fold — ledger events to current state
# =============================================================================

@dataclass(frozen=True)
class TrackedHypothesis:
    """One question and everything the ledger says has happened to it."""

    hypothesis: Hypothesis
    status: str
    opened_seq: int
    opened_at: str
    attempts: tuple = ()
    resolution: Optional[ResolutionEvidence] = None
    resolved_at: Optional[str] = None
    reopen_count: int = 0
    #: Resolutions that a later reopen replaced. Kept, never dropped: derived views
    #: may rewind, evidence does not disappear (invariant 8).
    superseded_resolutions: tuple = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status: {self.status!r} not in {sorted(STATUSES)}")

    @property
    def hypothesis_id(self) -> str:
        return self.hypothesis.hypothesis_id

    @property
    def is_open(self) -> bool:
        return self.status == STATUS_OPEN

    @property
    def evidence_grade(self) -> str:
        """The hypothesis's ENTRY grade — `design_prior`, open or resolved.

        It delegates rather than storing, so there is no slot for a resolution to
        write a higher grade into.
        """
        return self.hypothesis.evidence_grade


def _decode_payload(factory, record: Any, what: str, event: LedgerEvent):
    """Decode one ledger payload, classifying EVERY failure as ledger corruption.

    Without this, a ledger line whose payload is malformed surfaced as whatever its
    dataclass happened to raise: a bare `ValueError` (which is not a `HypothesisError`
    at all, so the driver that catches the controller plane — the stated reason
    `HypothesisError` extends `ControllerError` — does not catch it), or a
    `FalsifierMissing`, which is a *store* error and would send a reader looking for a
    bad operator file when the defect is in the ledger.
    """
    try:
        return factory(record)
    except (TypeError, ValueError, HypothesisError) as exc:
        raise HypothesisLedgerCorruption(
            f"seq {event.seq}: {event.kind} payload {what!r} is not a valid record "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def fold_ledger(events: Sequence[LedgerEvent]) -> dict:
    """Fold ledger events into `{hypothesis_id: TrackedHypothesis}`. Pure.

    Contradictions are REFUSED, not reconciled: an attempt or resolution naming an
    unopened question, a second OPENED for one id, a resolution of a resolved
    question, or a reopen of an open one all raise. A fold that repairs its input
    cannot tell a defect from a history.
    """
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("events must be a sequence of LedgerEvent")
    # Typed BEFORE sorting: `sorted(..., key=lambda e: e.seq)` touches every element
    # first, so a per-element type check inside the loop below could only ever fire for
    # something that already has a `.seq` — the documented TypeError was unreachable for
    # a dict, which is the exact thing a caller reaching for this function has.
    for index, event in enumerate(events):
        if not isinstance(event, LedgerEvent):
            raise TypeError(
                f"events[{index}]: expected LedgerEvent, got {type(event).__name__}"
            )
    state: dict = {}
    for event in sorted(events, key=lambda e: e.seq):
        key = event.hypothesis_id
        if event.kind == EVENT_OPENED:
            if key in state:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} was opened twice; one id is one question"
                )
            hypothesis = _decode_payload(
                Hypothesis.from_dict, event.payload.get("hypothesis") or {},
                "hypothesis", event,
            )
            if hypothesis.hypothesis_id != key:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the payload holds "
                    f"{hypothesis.hypothesis_id!r}"
                )
            state[key] = TrackedHypothesis(
                hypothesis=hypothesis,
                status=STATUS_OPEN,
                opened_seq=event.seq,
                opened_at=event.at,
            )
            continue

        current = state.get(key)
        if current is None:
            raise HypothesisLedgerCorruption(
                f"seq {event.seq}: {event.kind} names {key!r}, which was never "
                "opened; an append-only ledger cannot repair a dangling reference"
            )
        if event.kind == EVENT_ATTEMPTED:
            attempt = _decode_payload(
                Attempt.from_dict, event.payload.get("attempt") or {}, "attempt", event,
            )
            if attempt.hypothesis_id != key:
                # The same binding the OPENED branch enforces. Without it the receipt
                # for what was tried can be filed against a question it was never about
                # — and "what was tried, and what it bore on" is the entire product of
                # this half of the mechanism.
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the attempt holds "
                    f"{attempt.hypothesis_id!r}; a receipt filed under the wrong "
                    "question is worse than no receipt"
                )
            # An attempt is recorded whatever the status: a proposal dispositioned
            # after a question was resolved is still part of that question's history.
            state[key] = dataclasses.replace(
                current, attempts=current.attempts + (attempt,)
            )
        elif event.kind == EVENT_RESOLVED:
            if not current.is_open:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} is already {current.status}; resolving a "
                    "resolved question would overwrite the evidence that closed it"
                )
            resolution = _decode_payload(
                ResolutionEvidence.from_dict, event.payload.get("resolution") or {},
                "resolution", event,
            )
            state[key] = dataclasses.replace(
                current,
                status=resolution.outcome,
                resolution=resolution,
                resolved_at=event.at,
            )
        elif event.kind == EVENT_REOPENED:
            if current.is_open:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} is already open; reopening it would "
                    "record a state change that did not happen"
                )
            state[key] = dataclasses.replace(
                current,
                status=STATUS_OPEN,
                resolution=None,
                resolved_at=None,
                reopen_count=current.reopen_count + 1,
                superseded_resolutions=(
                    current.superseded_resolutions
                    + ((current.resolution,) if current.resolution is not None else ())
                ),
            )
        else:  # pragma: no cover - LedgerEvent.__post_init__ closes the vocabulary
            raise HypothesisLedgerCorruption(f"seq {event.seq}: unknown kind {event.kind!r}")
    return state


# =============================================================================
# Do-not-repeat disposition (§8.4, §19.2, §19.3)
# =============================================================================

@dataclass(frozen=True)
class LedgerMatch:
    """One do-not-repeat / constraint-ledger entry that matches a hypothesis.

    This module CONSUMES matches; the §19.2 ledger that produces them belongs to the
    memory-update plane. `receipt` is `None` when the entry carries none, and that is
    decisive: §8.4 rejects a repeat only when the negative was recorded *"by an entry
    carrying a receipt"*, and §19.3 exists because a wrong suppression is invisible —
    nothing ever tests it again.
    """

    entry_id: str
    entry_class: str
    match_dimensions: Mapping[str, Any] = field(default_factory=dict)
    receipt: Optional[str] = None
    conflicted: bool = False
    reopen_predicate_satisfied: bool = False

    def __post_init__(self) -> None:
        _require_text(self.entry_id, "entry_id")
        if self.entry_class not in MATCH_CLASSES:
            raise ValueError(
                f"entry_class: {self.entry_class!r} not in {sorted(MATCH_CLASSES)}"
            )
        if not isinstance(self.match_dimensions, Mapping):
            raise TypeError("match_dimensions must be a mapping")
        if self.receipt is not None:
            _require_text(self.receipt, "receipt")
        for name in ("conflicted", "reopen_predicate_satisfied"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_class": self.entry_class,
            "match_dimensions": dict(self.match_dimensions),
            "receipt": self.receipt,
            "conflicted": self.conflicted,
            "reopen_predicate_satisfied": self.reopen_predicate_satisfied,
        }


class DoNotRepeatLedger(Protocol):
    """The §19.2 ledger, as this module consumes it. Nothing here implements it."""

    def matches_for(self, regime: Mapping[str, Any], statement: str) -> Sequence:
        ...


def check_do_not_repeat(
    *, regime: Mapping[str, Any], matches: Optional[Sequence]
) -> schemas.Check:
    """FAIL when a hypothesis repeats a RECEIPTED negative (§8.4, §19.2, §19.3).

    **This function cannot see who stated the hypothesis.** It takes the regime and
    the matches, and nothing else — the structural form of AK-D38's *"being the
    operator's idea is not new evidence"*. There is no origin parameter to consult and
    therefore no path along which one could be.

    Three outcomes, all real:

    * `matches is None` means the ledger was NOT consulted — COULD_NOT_CHECK. A
      hypothesis nobody checked against the ledger is not a hypothesis that clears it.
    * an empty sequence means it WAS consulted and matched nothing — PASS.
    * a rejecting match with a receipt — FAIL.

    Class dispositions follow §19.2's table exactly: `HARD_CONSTRAINT` and
    `MATCHED_NEGATIVE` reject; `CONDITIONAL_NEGATIVE`, `CONFOUNDED_RESULT`,
    `SUPERSEDED_FACT` and `LOW_VALUE` are advisory — they exclude cells, demand a
    repaired experiment, regenerate from current source, or deprioritize, and none of
    them closes the question. Two further §19.2/§19.3 rules bite here: a `conflicted`
    entry is never authoritative, and a `MATCHED_NEGATIVE` whose reopen predicate is
    newly satisfied does not reject.
    """
    if matches is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the do-not-repeat ledger was not consulted; not checking is not a clear "
            "result (§8.4 rejects a repeat of a receipted negative, and this call "
            "cannot tell whether one exists)",
        ))
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        raise TypeError("matches must be None or a sequence of LedgerMatch")
    for index, match in enumerate(matches):
        if not isinstance(match, LedgerMatch):
            raise TypeError(f"matches[{index}] must be a LedgerMatch")
    if not isinstance(regime, Mapping):
        raise TypeError("regime must be a mapping")

    # A hypothesis with no regime has no match dimensions (§19.2 — "'do not repeat'
    # without regime identity is dangerous, because this project repeatedly observes
    # sign changes across architecture, substrate, batch, context and quant"). It is
    # collected as an incompleteness rather than returned immediately: a concrete
    # receipted match is a FACT and outranks an incomplete comparison, the same
    # precedence `state_machine.check_anchor_identity` applies.
    incomplete: list = []
    if not regime:
        incomplete.append(
            "the hypothesis declares no regime, so no match dimensions exist to "
            "compare against (§19.2)"
        )

    rejecting: list = []
    advisory: list = []
    unreceipted: list = []
    for match in matches:
        if match.conflicted:
            advisory.append(
                f"{match.entry_id} ({match.entry_class}) is CONFLICTED and is never "
                "authoritative (§19.2)"
            )
            continue
        if match.entry_class not in REJECTING_MATCH_CLASSES:
            advisory.append(
                f"{match.entry_id} ({match.entry_class}) is advisory under §19.2: it "
                "excludes cells, demands a repair, or deprioritizes — it does not "
                "close the question"
            )
            continue
        if (match.entry_class == MATCH_CLASS_MATCHED_NEGATIVE
                and match.reopen_predicate_satisfied):
            advisory.append(
                f"{match.entry_id} (MATCHED_NEGATIVE) has a newly satisfied reopen "
                "predicate, so §19.2 admits it"
            )
            continue
        if match.receipt is None:
            unreceipted.append(
                f"{match.entry_id} ({match.entry_class}) matches but carries NO "
                "receipt; §8.4 rejects a repeat only when the negative carries one, "
                "and §19.3 requires a source receipt bound to the production commit"
            )
            continue
        rejecting.append(
            f"{match.entry_id} ({match.entry_class}) already records this under "
            f"matching conditions, receipt {match.receipt}"
        )

    if rejecting:
        return schemas.Check(
            schemas.FAIL, tuple(rejecting + unreceipted + incomplete + advisory)
        )
    if unreceipted or incomplete:
        # A suppression without a receipt neither rejects nor clears: §19.3 makes the
        # receipt the price of closing a family, and a match we cannot verify is
        # exactly the "wrong suppression silently closes a research family" row of §12.
        return schemas.Check(
            schemas.COULD_NOT_CHECK, tuple(unreceipted + incomplete + advisory)
        )
    return schemas.Check(schemas.PASS, tuple(advisory))


# =============================================================================
# The tracker
# =============================================================================

@dataclass(frozen=True)
class IntakeReport:
    """What one reconciliation of the operator store against the ledger did."""

    store_path: Optional[str]
    store_sha256: Optional[str]
    opened: tuple = ()
    already_tracked: tuple = ()
    #: In the store AND already resolved. Leaving the line in the file does NOT
    #: reopen it — reopening requires new evidence (§19.2's reopen predicate).
    resolved_but_still_in_store: tuple = ()
    #: Tracked, open, and NO LONGER in the store. Deleting the line does not close the
    #: question; that is the evaporation this module exists to prevent.
    open_but_absent_from_store: tuple = ()


class HypothesisTracker:
    """Still-open tracking for every hypothesis, whatever its origin.

    Holds NO folded state: every query re-reads the ledger from disk under the journal
    write lock, for the same reason `state_machine` re-reads the control latch from
    disk. A cached fold is a fold that can be written back over the record.
    `__slots__` is load-bearing — there is no slot for a state cache, so an edit that
    adds one fails at runtime rather than quietly reintroducing the shape.
    """

    __slots__ = ("_journal", "_root", "_ledger", "_recorder", "_clock", "_campaign_id")

    def __init__(
        self,
        *,
        journal_: journal.Journal,
        root: str,
        campaign_id: Optional[str] = None,
        recorder: Optional[HypothesisRecorder] = None,
        clock=None,
    ) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        try:
            journal_.shards()
        except journal.JournalError as exc:
            raise HypothesisError(
                f"{journal_.root}: the journal is not readable "
                f"({type(exc).__name__}: {exc}); call Journal.initialize() before "
                "constructing a tracker over it"
            ) from exc
        self._journal = journal_
        self._root = os.path.abspath(_require_text(root, "root"))
        self._campaign_id = campaign_id
        self._clock = clock if clock is not None else _iso_now
        os.makedirs(self._root, exist_ok=True)
        self._ledger = HypothesisLedger(os.path.join(self._root, LEDGER_FILENAME))
        self._ledger.initialize()
        self._recorder = recorder if recorder is not None else JournalOrderedRecorder(
            journal_, self._ledger
        )

    # ---- position ---------------------------------------------------------

    @property
    def root(self) -> str:
        return self._root

    @property
    def ledger(self) -> HypothesisLedger:
        return self._ledger

    @property
    def campaign_id(self) -> Optional[str]:
        return self._campaign_id

    def read(self) -> LedgerRead:
        """The ledger as it is on disk right now, plus any torn tail."""
        return self._ledger.read()

    def state(self) -> dict:
        """`{hypothesis_id: TrackedHypothesis}`, folded fresh from disk."""
        return fold_ledger(self._ledger.read().events)

    def get(self, hypothesis_id: str) -> TrackedHypothesis:
        tracked = self.state().get(_require_text(hypothesis_id, "hypothesis_id"))
        if tracked is None:
            raise UnknownHypothesis(
                f"{hypothesis_id!r} was never opened in {self._ledger.path}"
            )
        return tracked

    def still_open(self) -> tuple:
        """Every unresolved question, in the order the questions were opened.

        The order is stable and is NOT a ranking: §8.4.0 — an operator hypothesis is
        not a queue-jumping mechanism, so nothing in this module ranks anything.
        """
        return tuple(sorted(
            (t for t in self.state().values() if t.is_open),
            key=lambda t: t.opened_seq,
        ))

    def resolved(self) -> tuple:
        return tuple(sorted(
            (t for t in self.state().values() if not t.is_open),
            key=lambda t: t.opened_seq,
        ))

    # ---- writes -----------------------------------------------------------

    def _record(
        self, kind: str, hypothesis_id: str, payload: Mapping[str, Any]
    ) -> LedgerEvent:
        """Build, record, and return one event. Recording is what makes it real.

        Called under the journal write lock by every writer here, so the read that
        assigns `seq` and the append that consumes it cannot be interleaved.
        """
        read = self._ledger.read()
        seq = (read.events[-1].seq + 1) if read.events else 1
        event = LedgerEvent(
            seq=seq,
            kind=kind,
            hypothesis_id=hypothesis_id,
            at=self._clock(),
            payload=dict(payload),
        )
        # Re-fold with the candidate appended BEFORE writing: the fold owns every
        # legality rule, so validating through it means a caller cannot reach a state
        # the fold would refuse to read back.
        fold_ledger(tuple(read.events) + (event,))
        recorded = self._recorder.record(event)
        if not isinstance(recorded, LedgerEvent):
            raise HypothesisError(
                f"recorder returned {type(recorded).__name__}, not a LedgerEvent; an "
                "event whose record cannot be identified did not happen"
            )
        # The ENVELOPE and the PAYLOAD, not just seq and kind. Comparing only the
        # envelope left the whole decision inside the record substitutable: a recorder
        # asked to record `refuted` with one evidence ref could record `confirmed` with
        # another, return an event whose seq and kind matched, and the tracker would
        # then report the question confirmed on evidence nobody supplied. This seam is
        # the documented swap point for a future journal-kind adapter, which is exactly
        # the position from which a substituted payload would be invisible.
        for name in ("seq", "kind", "hypothesis_id", "at"):
            if getattr(recorded, name) != getattr(event, name):
                raise HypothesisError(
                    "recorder returned an event that is not the one it was asked to "
                    f"record ({name}: {getattr(recorded, name)!r} vs "
                    f"{getattr(event, name)!r})"
                )
        # A recorder MAY add to the payload — binding in a journal event id is the
        # documented reason this seam exists (`state_machine.JournalTransitionRecorder`
        # fills its binding in the same way). It may not change or drop what it was
        # given.
        for name, value in event.payload.items():
            if name not in recorded.payload:
                raise HypothesisError(
                    "recorder returned an event that is not the one it was asked to "
                    f"record (payload key {name!r} was dropped)"
                )
            if recorded.payload[name] != value:
                raise HypothesisError(
                    "recorder returned an event that is not the one it was asked to "
                    f"record (payload key {name!r} was rewritten: "
                    f"{recorded.payload[name]!r} vs {value!r})"
                )
        # And what the recorder SAYS it recorded is checked against what the ledger now
        # holds, because every query on this class reads that file and nothing else. A
        # recorder is a seam, not an authority: "recording is what makes it real" has to
        # mean the RECORD, not the return value.
        after = self._ledger.read()
        landed = after.events[-1] if after.events else None
        if (landed is None or landed.seq != event.seq or landed.kind != event.kind
                or landed.hypothesis_id != event.hypothesis_id):
            raise HypothesisError(
                f"the ledger does not end with the event just recorded ({event.kind} "
                f"#{event.seq} for {event.hypothesis_id}); it ends with "
                + (f"{landed.kind} #{landed.seq} for {landed.hypothesis_id}"
                   if landed is not None else "nothing")
            )
        for name, value in event.payload.items():
            if landed.payload.get(name) != value:
                raise HypothesisError(
                    f"the ledger holds a different payload for {event.kind} "
                    f"#{event.seq}: key {name!r} reads {landed.payload.get(name)!r}, "
                    f"not {value!r}"
                )
        return recorded

    def open_hypothesis(self, hypothesis: Hypothesis) -> LedgerEvent:
        """Track a new question. Refuses a rewrite of a tracked one."""
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis")
        with self._journal.write_lock():
            state = self.state()
            existing = state.get(hypothesis.hypothesis_id)
            if existing is not None:
                if existing.hypothesis.fingerprint != hypothesis.fingerprint:
                    raise QuestionRewritten(
                        f"{hypothesis.hypothesis_id}: a DIFFERENT question is already "
                        "tracked under this id (statement, falsifier or regime "
                        "changed). A rewritten falsifier is how any hypothesis "
                        "becomes 'confirmed' — state the new question under a new id"
                    )
                raise HypothesisAlreadyTracked(
                    f"{hypothesis.hypothesis_id} is already tracked "
                    f"({existing.status}); opening it again would restart its history"
                )
            payload = {
                "hypothesis": hypothesis.to_dict(),
                "entry_evidence_grade": hypothesis.evidence_grade,
                "fingerprint": hypothesis.fingerprint,
            }
            if self._campaign_id is not None:
                payload["campaign_id"] = self._campaign_id
            return self._record(EVENT_OPENED, hypothesis.hypothesis_id, payload)

    def note_attempt(
        self,
        hypothesis_id: str,
        *,
        proposal_id: str,
        disposition: str,
        bears_on_falsifier: bool,
        note: str,
        refs: Sequence = (),
    ) -> LedgerEvent:
        """Record that a proposal was dispositioned while this question was open.

        THIS DOES NOT RESOLVE ANYTHING, by construction — there is no branch here that
        can set a status, whatever `bears_on_falsifier` says. §8.4.0: a hypothesis used
        to evaporate when its proposal was dispositioned, including when that proposal
        failed for an unrelated reason. Resolution is `resolve()`, and it costs
        evidence.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        attempt = Attempt(
            hypothesis_id=hypothesis_id,
            proposal_id=proposal_id,
            disposition=disposition,
            bears_on_falsifier=bears_on_falsifier,
            note=note,
            refs=tuple(refs),
            at=self._clock(),
        )
        with self._journal.write_lock():
            state = self.state()
            if hypothesis_id not in state:
                raise UnknownHypothesis(
                    f"{hypothesis_id!r} was never opened; an attempt on an untracked "
                    "question is the receipt-less 'already tried' this module removes"
                )
            return self._record(
                EVENT_ATTEMPTED, hypothesis_id, {"attempt": attempt.to_dict()}
            )

    def resolve(self, hypothesis_id: str, evidence: ResolutionEvidence) -> LedgerEvent:
        """Close a question WITH the evidence that closed it.

        `ResolutionEvidence` has already refused an empty ref list, an absent
        observation against the falsifier, and `bears_on_falsifier != True`. What is
        added here is that the question must be OPEN: overwriting a resolution would
        destroy the receipt that closed it.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        if not isinstance(evidence, ResolutionEvidence):
            raise TypeError("evidence must be a ResolutionEvidence")
        with self._journal.write_lock():
            state = self.state()
            tracked = state.get(hypothesis_id)
            if tracked is None:
                raise UnknownHypothesis(f"{hypothesis_id!r} was never opened")
            if not tracked.is_open:
                raise HypothesisNotOpen(
                    f"{hypothesis_id} is already {tracked.status}; reopen() it on new "
                    "evidence rather than overwriting the evidence that closed it"
                )
            stamped = evidence if evidence.at is not None else dataclasses.replace(
                evidence, at=self._clock()
            )
            return self._record(
                EVENT_RESOLVED, hypothesis_id, {"resolution": stamped.to_dict()}
            )

    def reopen(
        self,
        hypothesis_id: str,
        *,
        reason: str,
        new_evidence_refs: Sequence,
        reopened_by: str,
    ) -> LedgerEvent:
        """Reopen a resolved question, on NEW EVIDENCE.

        `new_evidence_refs` is mandatory and non-empty for the same reason
        `resolve()`'s is: §19.2's reopen predicate is a fact about the world, and
        "someone asked again" is not one. An `inconclusive` resolution is reopenable
        exactly like the other two — otherwise "inconclusive" would silently close a
        question forever, which is the failure mode in a new costume.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        _require_text(reason, "reason")
        _require_text(reopened_by, "reopened_by")
        refs = _require_refs(
            new_evidence_refs, "new_evidence_refs", error=ResolutionEvidenceMissing
        )
        with self._journal.write_lock():
            state = self.state()
            tracked = state.get(hypothesis_id)
            if tracked is None:
                raise UnknownHypothesis(f"{hypothesis_id!r} was never opened")
            if tracked.is_open:
                raise HypothesisNotOpen(
                    f"{hypothesis_id} is already open; reopening it would record a "
                    "state change that did not happen"
                )
            return self._record(EVENT_REOPENED, hypothesis_id, {
                "reason": reason,
                "new_evidence_refs": list(refs),
                "reopened_by": reopened_by,
                "superseded_outcome": tracked.status,
            })

    # ---- intake -----------------------------------------------------------

    def intake(self, store: Optional[OperatorHypothesisStore]) -> IntakeReport:
        """Reconcile the operator store into the ledger. Idempotent.

        `store=None` is how a campaign says it has NO operator channel. A configured
        store that cannot be read RAISES (`load()`), because a mount failure must not
        be able to say "the operator has no hypotheses".

        Four dispositions, and the last two are the interesting ones:

        * new id                       -> opened;
        * known id, same fingerprint   -> already tracked, no event;
        * known id, resolved           -> stays resolved. Leaving the line in the file
          does not reopen it; `reopen()` and new evidence do.
        * tracked and open, no longer in the file -> STAYS OPEN and is reported. The
          operator deleting a line means "stop offering this at intake", never "this
          question is answered".

        A known id whose statement, falsifier or regime changed raises
        `QuestionRewritten` — see `open_hypothesis()`.
        """
        if store is None:
            state = self.state()
            return IntakeReport(
                store_path=None,
                store_sha256=None,
                open_but_absent_from_store=tuple(sorted(
                    t.hypothesis_id for t in state.values()
                    if t.is_open and t.hypothesis.origin == ORIGIN_OPERATOR
                )),
            )
        if not isinstance(store, OperatorHypothesisStore):
            raise TypeError("store must be an OperatorHypothesisStore or None")

        stated, store_sha = store.load_with_digest()
        opened: list = []
        already: list = []
        resolved_in_store: list = []
        with self._journal.write_lock():
            state = self.state()
            # TWO PASSES, and the order is the point: classify every stated entry
            # BEFORE opening any of them. Opening as we went meant a rewritten entry
            # anywhere in the file raised after earlier entries had already been
            # committed, and the IntakeReport — the only account of what this intake
            # did — was destroyed by the exception that travelled past it. An intake
            # that half-applied and reported nothing is the shape of defect this
            # module exists to remove.
            to_open: list = []
            for hypothesis in stated:
                tracked = state.get(hypothesis.hypothesis_id)
                if tracked is None:
                    to_open.append(hypothesis)
                    continue
                if tracked.hypothesis.fingerprint != hypothesis.fingerprint:
                    raise QuestionRewritten(
                        f"{hypothesis.hypothesis_id}: the store now states a DIFFERENT "
                        "question under a tracked id (statement, falsifier or regime "
                        "changed). Rewriting a falsifier after evidence exists is how "
                        "any hypothesis becomes 'confirmed' — give the new question a "
                        "new id"
                    )
                if tracked.is_open:
                    already.append(hypothesis.hypothesis_id)
                else:
                    resolved_in_store.append(hypothesis.hypothesis_id)
            for hypothesis in to_open:
                self.open_hypothesis(hypothesis)
                opened.append(hypothesis.hypothesis_id)
            stated_ids = {h.hypothesis_id for h in stated}
            absent = sorted(
                t.hypothesis_id for t in self.state().values()
                if t.is_open and t.hypothesis.origin == ORIGIN_OPERATOR
                and t.hypothesis_id not in stated_ids
            )
        return IntakeReport(
            store_path=store.path,
            store_sha256=store_sha,
            opened=tuple(opened),
            already_tracked=tuple(already),
            resolved_but_still_in_store=tuple(resolved_in_store),
            open_but_absent_from_store=tuple(absent),
        )

    # ---- the planner surface ----------------------------------------------

    def planner_round_block(
        self,
        *,
        round_id: str,
        matches_by_hypothesis: Optional[Mapping[str, Optional[Sequence]]] = None,
        include_resolved: bool = True,
    ) -> dict:
        """The open set, re-surfaced for one planning round (§8.4.0, §6.1).

        Structured facts, not a prompt: the planner adapter owns prompt assembly, and a
        renderer that emitted instruction-shaped prose would be putting operator
        content in an instruction position, which §6.1 forbids for imported content and
        §8.4.0 forbids for this content specifically — *a proposal source, never an
        authority*.

        Every entry carries its falsifier (structurally: `Hypothesis` cannot exist
        without one) and its `entry_evidence_grade`, which is `design_prior` for every
        origin. The resolution's own grade is rendered under a DIFFERENT name,
        `resolution_evidence_grade`, so no reader can read one as the other.

        `matches_by_hypothesis` supplies §19.2 ledger matches per id. A missing id
        means the ledger was not consulted for it, and the entry carries
        COULD_NOT_CHECK — never a silent pass.
        """
        _require_text(round_id, "round_id")
        if matches_by_hypothesis is not None and not isinstance(
            matches_by_hypothesis, Mapping
        ):
            raise TypeError("matches_by_hypothesis must be a mapping or None")

        def _render(tracked: TrackedHypothesis) -> dict:
            hypothesis = tracked.hypothesis
            matches = None
            if matches_by_hypothesis is not None:
                matches = matches_by_hypothesis.get(hypothesis.hypothesis_id)
            verdict = check_do_not_repeat(regime=hypothesis.regime, matches=matches)
            entry = {
                "hypothesis_id": hypothesis.hypothesis_id,
                "statement": hypothesis.statement,
                "falsifier": hypothesis.falsifier,
                "origin": hypothesis.origin,
                "author": hypothesis.author,
                "entry_evidence_grade": hypothesis.evidence_grade,
                "regime": dict(hypothesis.regime),
                "provenance": dict(hypothesis.source),
                "status": tracked.status,
                "opened_at": tracked.opened_at,
                "reopen_count": tracked.reopen_count,
                "attempt_count": len(tracked.attempts),
                "attempts": [a.to_dict() for a in tracked.attempts],
                "do_not_repeat": {
                    "outcome": verdict.outcome,
                    "reasons": list(verdict.reasons),
                },
            }
            if tracked.resolution is not None:
                entry["resolution"] = {
                    "outcome": tracked.resolution.outcome,
                    "resolution_evidence_grade": tracked.resolution.evidence_grade,
                    "evidence_refs": list(tracked.resolution.evidence_refs),
                    "falsifier_observed": tracked.resolution.falsifier_observed,
                    "resolved_by": tracked.resolution.resolved_by,
                    "resolved_at": tracked.resolved_at,
                }
            return entry

        open_entries = [_render(t) for t in self.still_open()]
        block = {
            "schema": ROUND_BLOCK_SCHEMA,
            "round_id": round_id,
            "campaign_id": self._campaign_id,
            "compiled_at": self._clock(),
            # §8.4.0, AK-D38, and the reason this whole module is safe. Rendered as a
            # field so a consumer that drops it is missing a key rather than silently
            # promoting a prior.
            "authority": "proposal_source_not_authority",
            "entry_evidence_grade": ENTRY_GRADE,
            "open_count": len(open_entries),
            "still_open": open_entries,
        }
        if include_resolved:
            block["resolved"] = [_render(t) for t in self.resolved()]
        flagged = schemas.find_authority_flavoured_keys(block)
        if flagged:  # pragma: no cover - no construction here produces such a key
            raise HypothesisError(
                f"round block carries authority-flavoured keys {flagged}; a planner "
                "block is a set of priors, never a grant (§1.3, invariant 5)"
            )
        # Serializability is proven, not hoped for: the block is handed to a context
        # compiler that will hash it.
        schemas.canonical_json(block)
        return block


# =============================================================================
# The structural audit (§8.4.0's safety property, checked from the objects)
# =============================================================================

#: The entry the audit hands the loader. Its falsifier is a real one-line predicted
#: outcome so that a refusal can only be about the key under test.
_AUDIT_PROBE_ENTRY: Mapping[str, Any] = {
    "hypothesis_id": "akh-audit-probe",
    "statement": "the operator store refuses to state its own evidence grade",
    "falsifier": "a probe entry carrying a grade key loads without a refusal",
}


def _store_refuses_key(key: str) -> Optional[str]:
    """`None` if the loader REFUSES an entry carrying `key`, else why that matters.

    Behavioural, not declarative. Reading `_REFUSED_ENTRY_KEYS` proves a dict has a
    string in it; it does not prove `_load_entry` consults the dict, so the whole
    refusal could be deleted from the loader with the table left in place and the audit
    would still have said PASS. This calls the loader.
    """
    probe = dict(_AUDIT_PROBE_ENTRY)
    probe[key] = GRADE_PROTOCOL_BOUND
    store = OperatorHypothesisStore("<audit-probe>")  # no I/O: _load_entry reads none
    try:
        store._load_entry(probe, 0, "<audit-probe>")
    except HypothesisStoreError:
        return None
    except Exception as exc:  # noqa: BLE001 - any refusal is a refusal, but say so
        return (
            f"the operator store rejects a stated {key!r} with {type(exc).__name__} "
            "rather than a HypothesisStoreError; the refusal is not the typed one a "
            "caller can catch"
        )
    return (
        f"the operator store LOADS an entry stating {key!r}; the one operator-editable "
        "input in this system can state its own evidence grade, which is exactly the "
        "laundering §8.4.0/AK-D38/§19.0 rule 4 forbid (the refusal table may still list "
        "the key — a table is not an enforcement)"
    )


def _store_admits_a_clean_entry() -> Optional[str]:
    """`None` if the probe entry WITHOUT a grade key loads.

    The control that stops `_store_refuses_key` from passing vacuously: a loader that
    refused everything — including every real operator hypothesis — would otherwise
    look like the strongest possible enforcement.
    """
    store = OperatorHypothesisStore("<audit-probe>")
    try:
        store._load_entry(dict(_AUDIT_PROBE_ENTRY), 0, "<audit-probe>")
    except Exception as exc:  # noqa: BLE001
        return (
            f"the operator store refuses a well-formed entry ({type(exc).__name__}: "
            f"{exc}); the grade-refusal probes below cannot be told apart from a loader "
            "that refuses everything"
        )
    return None


def audit_no_origin_grade_promotion() -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on "can origin raise a hypothesis's grade?".

    Proved from the objects rather than asserted in prose, in the shape of
    `state_machine.audit_no_cached_control_state()`:

    1. `entry_grade` returns `design_prior` for EVERY declared origin — enumerated,
       so a new origin cannot arrive with a higher grade;
    2. `Hypothesis` declares NO dataclass field whose name mentions a grade — the only
       grade on it is a read-only property, so there is nothing to assign;
    3. `Hypothesis` is frozen, so even the property's backing objects cannot be
       reassigned after construction; and
    4. the operator store REFUSES an `evidence_grade` key, so the one editable input in
       the system cannot state its own.

    COULD_NOT_CHECK is returned when introspection itself fails, never a soft PASS.
    """
    try:
        grades = {origin: entry_grade(origin) for origin in sorted(ORIGINS)}
        fields = {f.name for f in dataclasses.fields(Hypothesis)}
        frozen = Hypothesis.__dataclass_params__.frozen
    except Exception as exc:  # pragma: no cover - introspection failure is not a pass
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"could not introspect the hypothesis contract ({type(exc).__name__}: "
            f"{exc}); inability to evaluate is a third outcome",
        ))

    reasons: list = []
    promoted = {o: g for o, g in grades.items() if g != ENTRY_GRADE}
    if promoted:
        reasons.append(
            f"entry_grade() returns a non-{ENTRY_GRADE} grade for {promoted}; §8.4.0 — "
            "a hypothesis can never be promoted by its origin"
        )
    grade_fields = sorted(f for f in fields if "grade" in f.lower())
    if grade_fields:
        reasons.append(
            f"Hypothesis declares settable grade field(s) {grade_fields}; grade must "
            "stay a derived property with nothing to assign to"
        )
    if not frozen:
        reasons.append(
            "Hypothesis is not a frozen dataclass; a mutable hypothesis can be "
            "re-graded after it has been read"
        )
    if "evidence_grade" not in _REFUSED_ENTRY_KEYS:
        reasons.append(
            "the operator store no longer refuses an 'evidence_grade' key; the one "
            "operator-editable input in this system could state its own grade"
        )
    # …and the refusal is EXERCISED, not read off a table. Point 4 used to be satisfied
    # by the presence of a string in `_REFUSED_ENTRY_KEYS`, so deleting the loop in
    # `_load_entry` that consults it left the audit reporting PASS while the store
    # happily loaded `evidence_grade: protocol_bound`.
    vacuous = _store_admits_a_clean_entry()
    if vacuous is not None:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (vacuous,))
    for key in ("evidence_grade", "grade"):
        why = _store_refuses_key(key)
        if why is not None:
            reasons.append(why)
    grade_attr = getattr(Hypothesis, "evidence_grade", None)
    if not isinstance(grade_attr, property):
        reasons.append(
            "Hypothesis.evidence_grade is not a read-only property; a plain attribute "
            "is a slot something can assign a promoted grade into"
        )
    elif grade_attr.fset is not None or grade_attr.fdel is not None:
        reasons.append(
            "Hypothesis.evidence_grade has a setter or deleter; the grade must have "
            "no write path at all"
        )

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)
