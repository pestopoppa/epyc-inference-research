"""state_machine.py — the explicit AK4 controller state machine (design §8.1).

WHY THIS MODULE EXISTS
----------------------
Every other AK4 module (context compiler, planner adapter, pre-/post-run critic,
champion composer, budget guard) plugs into ONE machine, and the machine — not
any model output — owns disposition. Four failures it is written against, each
with a receipt in the owning design:

1. **A state ahead of its record.** A crash between "act" and "journal" leaves a
   loop whose position on disk is behind its position in memory, and the next
   restart replays or skips work with no way to tell which. Every transition here
   is JOURNALED FIRST and takes effect only if the record was fsynced
   (`_journal_then_act`). A transition that fails to journal did not happen.
2. **A control that was requested but never verified** (§4 invariant 19, §12 row
   *"An operator control is silently ignored"*). AutoPilot's pause was a silent
   no-op for months because state was cached in memory and written back over the
   operator's change. Here the latch is a file, it is re-read FROM DISK at the top
   of every iteration UNDER the journal write lock, no object retains it as
   attribute state, and `audit_no_cached_control_state()` proves that from the
   object rather than from a comment. An ack that exists without its latch — or a
   latch without its ack — is `UnackedControlError`, a hard failure.
3. **A restart that came up empty with nothing objecting** (§8.2 step 10, §2.5).
   `bootstrap()` asserts the derived view against the journal and REFUSES to
   start on an empty frontier with a non-empty journal. A deliberate rebase
   passes an explicit escape and lands its reason in the record, so it is never
   indistinguishable from the failure.
4. **A denominator that quietly died** (§8.9, §12, AK-D22). Anchor identity is
   re-verified at every campaign boundary, not only at freeze; a mismatch is
   `ANCHOR_MOVED` and an *unverifiable* anchor is refused rather than assumed
   good, because a fail-open anchor check is the same shape as no check.

WHAT THIS MODULE IS NOT
-----------------------
It runs no inference, no benchmark and no build; it starts, stops and signals no
process; it calls no model. It writes exactly three kinds of file, all under the
controller root it is given: the transition ledger, the operator-control latch,
and the recorded anchor identity.

Governing instrument: `measurement/protocols/kernel-research.md` (P-AK-SEARCH-1,
RATIFIED 2026-08-03). T3/T4 are release instruments outside that protocol's
scope: the `SEAL -> T3_RELEASE_GATE -> PACKAGE` branch is declared as a seam and
the tier is REFUSED (via `evaluator.api.admit_tier`) until AK5 wires a runner.

SUBSTRATE NOTE — where a transition record lands
------------------------------------------------
`journal.KINDS` is a CLOSED vocabulary and holds no `STATE_TRANSITION` kind, and
extending it is outside this task's write scope. So the durable transition record
is split, deliberately and visibly:

  * a transition that ENTERS A STOP STATE is appended to the journal as
    `STOP_STATE` — the kind whose meaning is exactly that, and the one
    `rebuild_views()` folds into `Views.stop_states`;
  * an operator control ack is appended as `OPERATOR_CONTROL_ACK`, and a
    deliberate view rebase as `VIEW_REBASED` — both through the journal's own
    helpers;
  * and EVERY transition, stop or not, is appended to `TransitionLedger`, an
    append-only fsynced ledger written UNDER `Journal.write_lock()` so its order
    is total with respect to journal appends.

Using `STOP_STATE` for a non-stop transition was rejected: `Views.stop_states`
would stop meaning "this campaign stopped", which is a derived view other planes
read. The follow-up is a `STATE_TRANSITION` kind in `journal.py`, after which the
ledger collapses into the journal with no change to this module's contract —
`TransitionRecorder` is the seam that makes that swap a one-line wiring change.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .. import journal, schemas
from ..evaluator import api as evaluator_api

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")      # mirrors schemas._SHA256_RE
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")      # mirrors schemas._COMMIT_RE

__all__ = [
    # errors
    "ControllerError", "IllegalTransition", "TransitionNotRecorded",
    "ControlLatchError", "UnackedControlError", "BootstrapRefused",
    "AnchorUncheckable", "StopEvidenceMissing", "LedgerCorruption",
    # state vocabulary
    "BOOTSTRAP", "DISCOVER", "SELECT_TARGET", "PROPOSE", "PRE_RUN_CRITIC",
    "MUTATE", "BUILD", "T0_GATE", "T1_SEARCH_EVAL", "POST_RUN_CRITIC",
    "BANK_EVENT", "UPDATE_SEARCH_STATE", "CHAMPION_GUARD",
    "T2_LINEAGE_ESTIMATOR", "SEAL", "T3_RELEASE_GATE", "PACKAGE",
    "RELEASE_PACKAGE_READY", "PLATEAU_STOP", "PLANNER_DEGRADED",
    "OPERATOR_STOP_REQUESTED", "BUDGET_STOP", "DISK_PRESSURE",
    "EXHAUSTED_SURFACE", "EVALUATOR_COVERAGE_GAP", "RESOURCE_UNAVAILABLE",
    "HOST_REBOOT_REQUIRED", "INTEGRITY_STOP", "OPERATOR_INPUT_REQUIRED",
    "ANCHOR_MOVED",
    "LOOP_STATES", "RELEASE_STATES", "LIVE_STATES", "STOP_STATES", "STATES",
    "UNIVERSAL_STOPS", "SEARCH_CLOSURE_STOPS", "EDGES", "REOPEN_EDGES",
    "STOP_RECOVERY", "STOP_EVIDENCE_REQUIREMENTS", "RESERVED_CLOSURE_PHRASES",
    "reserved_closure_findings",
    # controls
    "CONTROL_PAUSE", "CONTROL_DRAIN", "CONTROL_ABORT", "CONTROL_RESUME",
    "CONTROLS", "HALTING_CONTROLS", "DISPOSITION_LATCHED", "DISPOSITION_RELEASED",
    "STOP_REQUEST_ORIGINS",
    # types
    "Transition", "ControlLatch", "AnchorIdentity", "IterationDecision",
    "BootstrapReport", "RestoreReport", "StopRequest",
    # seams
    "TransitionRecorder", "ReleaseGateRunner",
    # implementations
    "TransitionLedger", "JournalTransitionRecorder", "ControlLatchStore",
    "AnchorIdentityStore", "ControllerStateMachine",
    # deterministic checks
    "check_anchor_identity", "check_closure_enumeration", "check_stop_evidence",
    "audit_no_cached_control_state",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class ControllerError(Exception):
    """Base for every refusal this module raises."""


class IllegalTransition(ControllerError):
    """An edge that is not declared. The machine owns disposition, so an
    undeclared transition is a defect in the caller, not a state to reconcile."""


class TransitionNotRecorded(ControllerError):
    """The durable record could not be written, so the transition did not happen.

    Raised in place of the state change, never after it: journal-then-act means a
    recording failure leaves the machine exactly where it was.
    """


class ControlLatchError(ControllerError):
    """The operator-control latch on disk is missing, malformed, or contradicted."""


class UnackedControlError(ControllerError):
    """Invariant 19: an unacked control is a HARD failure, not a slow one.

    Two shapes, both fatal: a latch whose acknowledgement does not resolve to a
    journaled `OPERATOR_CONTROL_ACK`, and a journaled halting ack that no latch
    on disk claims. The second is the crash-between-ack-and-latch window; taking
    it for "no control pending" is precisely how a pause becomes a no-op.
    """


class BootstrapRefused(ControllerError):
    """§8.2 step 10: the journal and the derived view disagree and no explicit
    rebase was declared. Refusing is the whole point — AutoPilot proceeded."""


class AnchorUncheckable(ControllerError):
    """The anchor identity could not be compared (§8.9, AK-D22).

    Not a stop state, because §8.10 declares none for it and continuing past an
    uncheckable denominator is the fail-open shape the check exists to prevent.
    """


class StopEvidenceMissing(ControllerError):
    """A stop was requested without the evidence §8.10 requires for that state."""


class LedgerCorruption(ControllerError):
    """The transition ledger holds a line that will not parse. A partial history
    that reads like a complete one is worse than no history."""


# =============================================================================
# States (§8.1)
# =============================================================================

BOOTSTRAP = "BOOTSTRAP"
DISCOVER = "DISCOVER"
SELECT_TARGET = "SELECT_TARGET"
PROPOSE = "PROPOSE"
PRE_RUN_CRITIC = "PRE_RUN_CRITIC"
MUTATE = "MUTATE"
BUILD = "BUILD"
T0_GATE = "T0_GATE"
T1_SEARCH_EVAL = "T1_SEARCH_EVAL"
POST_RUN_CRITIC = "POST_RUN_CRITIC"
BANK_EVENT = "BANK_EVENT"
UPDATE_SEARCH_STATE = "UPDATE_SEARCH_STATE"
CHAMPION_GUARD = "CHAMPION_GUARD"
#: §8.1 "optional T2_LINEAGE_ESTIMATOR -> update readiness signal". Present
#: because CHAMPION_GUARD's optional branch is otherwise unrepresentable, and an
#: unrepresentable branch gets taken by side effect instead of by an edge.
T2_LINEAGE_ESTIMATOR = "T2_LINEAGE_ESTIMATOR"

#: The operator-requested release branch. AK5 owns what runs inside it.
SEAL = "SEAL"
T3_RELEASE_GATE = "T3_RELEASE_GATE"
PACKAGE = "PACKAGE"

# ---- stop states (§8.10) ----------------------------------------------------

RELEASE_PACKAGE_READY = "RELEASE_PACKAGE_READY"
PLATEAU_STOP = "PLATEAU_STOP"
PLANNER_DEGRADED = "PLANNER_DEGRADED"
OPERATOR_STOP_REQUESTED = "OPERATOR_STOP_REQUESTED"
BUDGET_STOP = "BUDGET_STOP"
DISK_PRESSURE = "DISK_PRESSURE"
EXHAUSTED_SURFACE = "EXHAUSTED_SURFACE"
EVALUATOR_COVERAGE_GAP = "EVALUATOR_COVERAGE_GAP"
RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
HOST_REBOOT_REQUIRED = "HOST_REBOOT_REQUIRED"
INTEGRITY_STOP = "INTEGRITY_STOP"
OPERATOR_INPUT_REQUIRED = "OPERATOR_INPUT_REQUIRED"
ANCHOR_MOVED = "ANCHOR_MOVED"

LOOP_STATES = (
    BOOTSTRAP, DISCOVER, SELECT_TARGET, PROPOSE, PRE_RUN_CRITIC, MUTATE, BUILD,
    T0_GATE, T1_SEARCH_EVAL, POST_RUN_CRITIC, BANK_EVENT, UPDATE_SEARCH_STATE,
    CHAMPION_GUARD, T2_LINEAGE_ESTIMATOR,
)
RELEASE_STATES = (SEAL, T3_RELEASE_GATE, PACKAGE)
LIVE_STATES = LOOP_STATES + RELEASE_STATES

STOP_STATES = (
    RELEASE_PACKAGE_READY, PLATEAU_STOP, PLANNER_DEGRADED,
    OPERATOR_STOP_REQUESTED, BUDGET_STOP, DISK_PRESSURE, EXHAUSTED_SURFACE,
    EVALUATOR_COVERAGE_GAP, RESOURCE_UNAVAILABLE, HOST_REBOOT_REQUIRED,
    INTEGRITY_STOP, OPERATOR_INPUT_REQUIRED, ANCHOR_MOVED,
)

STATES = LIVE_STATES + STOP_STATES

# `BLOCKED_INSTRUMENT` appears in §8.1's terminal-alternatives line but NOT in
# §8.10's enumeration of deterministic stop states, and §8.10 is the clause that
# defines what each stop means and what it demands. Its two concrete causes have
# their own stops here — an evaluator that cannot cover the surface is
# EVALUATOR_COVERAGE_GAP, an instrument that cannot be held is
# RESOURCE_UNAVAILABLE — so adding a third name would give one condition two
# spellings and let a guard test the one that never fires.

#: Host, resource, integrity, budget, planner-health and operator conditions can
#: interrupt ANY live state; they are not properties of a loop position.
UNIVERSAL_STOPS = (
    PLANNER_DEGRADED, OPERATOR_STOP_REQUESTED, BUDGET_STOP, DISK_PRESSURE,
    EVALUATOR_COVERAGE_GAP, RESOURCE_UNAVAILABLE, HOST_REBOOT_REQUIRED,
    INTEGRITY_STOP, OPERATOR_INPUT_REQUIRED, ANCHOR_MOVED,
)

#: Search-closure stops are reachable only where search state is actually known.
#: Declaring the surface closed from inside BUILD would be a claim about evidence
#: the machine is not holding at that point.
SEARCH_CLOSURE_STOPS = (PLATEAU_STOP, EXHAUSTED_SURFACE)
_SEARCH_CLOSURE_SOURCES = (
    DISCOVER, SELECT_TARGET, UPDATE_SEARCH_STATE, CHAMPION_GUARD,
)

#: Live -> live edges. §8.1's spine, plus the four documented back-edges:
#: PRE_RUN_CRITIC -> PROPOSE (§6.3 "The critic may reject or revise"),
#: BUILD -> MUTATE (§8.5.1 repair from the clean parent, capped),
#: T0_GATE -> POST_RUN_CRITIC (a T0 failure is an outcome, §8.5 "Compilation
#: failures are valuable outcomes"), and T3_RELEASE_GATE -> CHAMPION_GUARD
#: (a failed release gate returns to research, §10).
_FORWARD_EDGES: Mapping[str, tuple] = {
    BOOTSTRAP: (DISCOVER,),
    DISCOVER: (SELECT_TARGET,),
    SELECT_TARGET: (PROPOSE,),
    PROPOSE: (PRE_RUN_CRITIC,),
    PRE_RUN_CRITIC: (MUTATE, PROPOSE),
    MUTATE: (BUILD,),
    BUILD: (T0_GATE, MUTATE),
    T0_GATE: (T1_SEARCH_EVAL, POST_RUN_CRITIC),
    T1_SEARCH_EVAL: (POST_RUN_CRITIC,),
    POST_RUN_CRITIC: (BANK_EVENT,),
    BANK_EVENT: (UPDATE_SEARCH_STATE,),
    UPDATE_SEARCH_STATE: (CHAMPION_GUARD,),
    CHAMPION_GUARD: (DISCOVER, SELECT_TARGET, T2_LINEAGE_ESTIMATOR, SEAL),
    T2_LINEAGE_ESTIMATOR: (CHAMPION_GUARD,),
    SEAL: (T3_RELEASE_GATE,),
    T3_RELEASE_GATE: (PACKAGE, CHAMPION_GUARD),
    PACKAGE: (RELEASE_PACKAGE_READY,),
}


def _build_edges() -> dict:
    edges: dict = {}
    for state in LIVE_STATES:
        targets = list(_FORWARD_EDGES[state])
        targets.extend(UNIVERSAL_STOPS)
        if state in _SEARCH_CLOSURE_SOURCES:
            targets.extend(SEARCH_CLOSURE_STOPS)
        # dict.fromkeys keeps declaration order and drops the duplicates a
        # forward edge into a stop would otherwise create.
        edges[state] = tuple(dict.fromkeys(targets))
    for state in STOP_STATES:
        # Terminal. `reopen()` is the only way out and it is not an edge.
        edges[state] = ()
    return edges


#: The ONLY legal transitions. `transition()` consults this and nothing else.
EDGES: Mapping[str, tuple] = _build_edges()

# ---- recovery ---------------------------------------------------------------

#: How each stop is LEFT. `reopen()` reads this; a class of HANDOFF_COMPLETE
#: cannot be reopened at all.
RECOVERY_OPERATOR_RESUME = "OPERATOR_RESUME"
RECOVERY_REANCHOR = "REANCHOR"
RECOVERY_NEW_CAMPAIGN = "NEW_CAMPAIGN"
RECOVERY_OPERATOR_REVIEW = "OPERATOR_REVIEW"
RECOVERY_HANDOFF_COMPLETE = "HANDOFF_COMPLETE"

STOP_RECOVERY: Mapping[str, str] = {
    RELEASE_PACKAGE_READY: RECOVERY_HANDOFF_COMPLETE,
    PLATEAU_STOP: RECOVERY_NEW_CAMPAIGN,
    PLANNER_DEGRADED: RECOVERY_OPERATOR_REVIEW,
    OPERATOR_STOP_REQUESTED: RECOVERY_OPERATOR_RESUME,
    BUDGET_STOP: RECOVERY_NEW_CAMPAIGN,
    DISK_PRESSURE: RECOVERY_OPERATOR_RESUME,
    EXHAUSTED_SURFACE: RECOVERY_NEW_CAMPAIGN,
    EVALUATOR_COVERAGE_GAP: RECOVERY_OPERATOR_REVIEW,
    RESOURCE_UNAVAILABLE: RECOVERY_OPERATOR_RESUME,
    HOST_REBOOT_REQUIRED: RECOVERY_OPERATOR_RESUME,
    INTEGRITY_STOP: RECOVERY_OPERATOR_REVIEW,
    OPERATOR_INPUT_REQUIRED: RECOVERY_OPERATOR_RESUME,
    ANCHOR_MOVED: RECOVERY_REANCHOR,
}

#: A reopened campaign re-enters at BOOTSTRAP, never mid-loop: §8.2 is what makes
#: the chain reconstructible, and a stop always invalidated something that
#: BOOTSTRAP re-establishes (identity, claim, storage, controls).
REOPEN_EDGES: Mapping[str, tuple] = {
    state: (() if recovery == RECOVERY_HANDOFF_COMPLETE else (BOOTSTRAP,))
    for state, recovery in STOP_RECOVERY.items()
}


# =============================================================================
# Stop evidence (§8.10) — what each stop must carry to be emittable
# =============================================================================

#: §8.10: bare "exhausted" and "all paths" are RESERVED WORDS the validator
#: rejects. Closure inflation is this project's most-repeated documented habit,
#: surviving even explicit awareness of the rule.
#:
#: "exhausted" is in this list because it was NOT, and that was the whole gap:
#: `guards.check_closure_language` scanned for it, this validator did not, and a
#: stop assembled without going through a guard — `stop()` and
#: `dispose_stop_request()` are both public — reached the record saying *"the
#: surface is exhausted"* with every check passing. A guard only the polite path
#: performs is not a guard, and the disposer is the one that must hold the line.
RESERVED_CLOSURE_PHRASES = (
    "exhausted", "all paths", "every path", "all avenues", "nothing left",
    "fully explored", "exhausted all", "no more options", "completely exhausted",
)

#: Word-bounded, and whitespace-tolerant between tokens, so "closure: ALL PATHS
#: covered" and "Exhausted." both match while a longer word containing one does
#: not. `guards.check_closure_language` compiles nothing of its own and calls
#: `reserved_closure_findings()` — two regex dialects over one vocabulary is how
#: the two planes came to disagree about "exhausted" in the first place.
#:
#: `_` is a boundary character on BOTH sides, so the identifier `EXHAUSTED_SURFACE`
#: — the stop state's own name, which a reason may legitimately mention — is not a
#: closure claim. The claim this rejects is the English word.
_RESERVED_PHRASE_RES = tuple(
    (phrase,
     re.compile(r"(?<![a-z0-9_])" + phrase.replace(" ", r"\s+") + r"(?![a-z0-9_])"))
    for phrase in RESERVED_CLOSURE_PHRASES
)


def reserved_closure_findings(text: Any) -> tuple:
    """Reserved closure phrases present in `text`, in declaration order.

    A non-string is no finding rather than an error: callers scan heterogeneous
    detail mappings, and "this field is not text" is a different complaint that
    the enumeration checks make in their own words.
    """
    if not isinstance(text, str):
        return ()
    lowered = text.lower()
    return tuple(phrase for phrase, pattern in _RESERVED_PHRASE_RES
                 if pattern.search(lowered))

#: Required keys in a stop's `detail`, per §8.10. Absent or empty => refusal.
STOP_EVIDENCE_REQUIREMENTS: Mapping[str, tuple] = {
    RELEASE_PACKAGE_READY: ("package_id",),
    PLATEAU_STOP: ("closed", "deferred", "planner_health"),
    PLANNER_DEGRADED: ("signal", "receipt"),
    OPERATOR_STOP_REQUESTED: ("control", "control_id"),
    BUDGET_STOP: ("budget", "limit", "consumed"),
    DISK_PRESSURE: ("path", "free_bytes", "floor_bytes"),
    EXHAUSTED_SURFACE: ("closed", "deferred"),
    EVALUATOR_COVERAGE_GAP: (
        "missing_coverage_class", "blocked_lineage", "owner", "deadline",
    ),
    RESOURCE_UNAVAILABLE: ("resource", "claim_kind"),
    HOST_REBOOT_REQUIRED: ("uptime_seconds", "ceiling_seconds"),
    INTEGRITY_STOP: ("signal", "occurrences", "receipt"),
    # §18 item 7: every operator escalation is rendered Context / Options /
    # Recommendation / Default. An open-ended question is not an escalation.
    OPERATOR_INPUT_REQUIRED: ("context", "options", "recommendation", "default"),
    ANCHOR_MOVED: ("recorded_anchor", "observed_anchor", "affected_backends"),
}

#: Where a stop request came from. Recorded, and deliberately NOT consulted by
#: any validator: §8.4.0/AK-D38 — authorship is not evidence, and the input most
#: likely to be waved through is the one whose author is trusted.
STOP_REQUEST_ORIGINS = frozenset({"controller", "planner", "critic", "operator"})


# =============================================================================
# Operator controls (§4 invariant 19, §6 observability row)
# =============================================================================

CONTROL_PAUSE = "pause"
CONTROL_DRAIN = "drain"
CONTROL_ABORT = "abort"
CONTROL_RESUME = "resume"

CONTROLS = frozenset({CONTROL_PAUSE, CONTROL_DRAIN, CONTROL_ABORT, CONTROL_RESUME})

#: The three that HALT. `resume` is the only one that clears a latch.
HALTING_CONTROLS = frozenset({CONTROL_PAUSE, CONTROL_DRAIN, CONTROL_ABORT})

DISPOSITION_LATCHED = "latched"
DISPOSITION_RELEASED = "released"


# =============================================================================
# Time and small durable-write helpers
# =============================================================================

def _iso_now() -> str:
    """Timezone-aware UTC; naive timestamps are rejected across this package."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: str, obj: Mapping[str, Any]) -> None:
    """Replace `path` with `obj`, durably. A torn control latch is a lost control."""
    directory = os.path.dirname(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    data = schemas.canonical_json(obj).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_dir(directory)


def _require_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} is required and must be a non-empty string")
    return value


# =============================================================================
# Transition record
# =============================================================================

@dataclass(frozen=True)
class Transition:
    """One edge, as it is written down BEFORE it takes effect.

    `journal_kind` / `journal_event_id` bind the ledger line to the journal event
    that carried it, when one exists. They are None for a transition the closed
    journal vocabulary has no kind for (see the module docstring's substrate
    note), and that is a visible absence rather than a silent one.
    """

    seq: int
    from_state: str
    to_state: str
    trigger: str
    reason: str
    at: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    journal_kind: Optional[str] = None
    journal_event_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ValueError("seq must be a positive int")
        for name in ("from_state", "to_state"):
            value = getattr(self, name)
            if value not in STATES:
                raise ValueError(f"{name}: {value!r} is not a declared state")
        _require_text(self.trigger, "trigger")
        _require_text(self.reason, "reason")
        _require_text(self.at, "at")
        if not isinstance(self.detail, Mapping):
            raise TypeError("detail must be a mapping")
        # Canonicalizability is checked HERE, not at write time: a detail that
        # cannot be serialized must fail before the machine has committed to the
        # transition, never halfway through recording it.
        schemas.canonical_json(dict(self.detail))

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "trigger": self.trigger,
            "reason": self.reason,
            "at": self.at,
            "detail": dict(self.detail),
            "journal_kind": self.journal_kind,
            "journal_event_id": self.journal_event_id,
        }

    @property
    def receipt(self) -> str:
        """Content hash of the ledger line — the durable id of this transition."""
        return schemas.content_hash(self.to_dict())

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Transition":
        if not isinstance(obj, Mapping):
            raise TypeError("transition record must be a mapping")
        missing = [
            key for key in ("seq", "from_state", "to_state", "trigger", "reason", "at")
            if key not in obj
        ]
        if missing:
            raise ValueError(f"transition record is missing {missing}")
        return Transition(
            seq=obj["seq"],
            from_state=obj["from_state"],
            to_state=obj["to_state"],
            trigger=obj["trigger"],
            reason=obj["reason"],
            at=obj["at"],
            detail=obj.get("detail") or {},
            journal_kind=obj.get("journal_kind"),
            journal_event_id=obj.get("journal_event_id"),
        )


# =============================================================================
# The transition ledger
# =============================================================================

@dataclass(frozen=True)
class LedgerRead:
    transitions: tuple
    discarded_tail_bytes: int


class TransitionLedger:
    """Append-only, fsynced, one JSON line per transition.

    Same discipline as `journal.py`: O_APPEND, fsync per record, directory fsync
    on creation, and a trailing fragment without its newline treated as a TORN
    APPEND rather than as data. A torn tail means the process died while writing
    the record — and because the record is written BEFORE the act, the transition
    it describes never took effect, so discarding it restores the machine to a
    position the record supports. The discarded byte count is reported, never
    swallowed.
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

    def append(self, transition: Transition) -> Transition:
        """Write and fsync one transition. Returning means DURABLE."""
        if not isinstance(transition, Transition):
            raise TypeError(
                f"transition must be a Transition, got {type(transition).__name__}"
            )
        line = (schemas.canonical_json(transition.to_dict()) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(fd, line)
            if written != len(line):  # pragma: no cover - short write on a regular file
                raise TransitionNotRecorded(
                    f"{self.path}: short write ({written} of {len(line)} bytes)"
                )
            os.fsync(fd)
        finally:
            os.close(fd)
        return transition

    def read(self) -> LedgerRead:
        """Every complete transition in order, plus the size of any torn tail."""
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
        body = data[: cut + 1]
        transitions: list = []
        for line_number, raw in enumerate(body.split(b"\n"), start=1):
            if not raw:
                continue
            if not raw.strip():
                raise LedgerCorruption(f"{self.path}:{line_number}: blank line")
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerCorruption(
                    f"{self.path}:{line_number}: unparseable transition line: {exc}"
                ) from exc
            try:
                transitions.append(Transition.from_dict(obj))
            except (TypeError, ValueError) as exc:
                raise LedgerCorruption(
                    f"{self.path}:{line_number}: invalid transition record: {exc}"
                ) from exc
        seqs = [t.seq for t in transitions]
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise LedgerCorruption(
                f"{self.path}: transition seq numbers are not strictly increasing: "
                f"{seqs}"
            )
        for previous, current in zip(transitions, transitions[1:]):
            if previous.to_state != current.from_state:
                raise LedgerCorruption(
                    f"{self.path}: transition {current.seq} starts in "
                    f"{current.from_state!r} but {previous.seq} left the machine in "
                    f"{previous.to_state!r}; the ledger does not describe one machine"
                )
        return LedgerRead(tuple(transitions), tail)


class TransitionRecorder(Protocol):
    """The seam every transition is recorded through.

    `record()` returns the transition AS RECORDED — the same object with its
    journal binding filled in — and RAISES if the record could not be made
    durable. `ControllerStateMachine` changes no state until it returns, so a
    recorder that raises is a transition that did not happen. This is also the
    swap point for the day `journal.py` grows a `STATE_TRANSITION` kind.
    """

    def record(self, transition: Transition) -> Transition:
        ...


class JournalTransitionRecorder:
    """The shipped recorder: journal event where a kind exists, ledger always.

    Order inside the write lock is journal-then-ledger so that
    `Views.stop_states` — which other planes read — is never behind the ledger,
    and so the ledger line can carry the journal event id it was bound to.
    """

    __slots__ = ("_journal", "_ledger", "_campaign_id")

    def __init__(
        self,
        journal_: journal.Journal,
        ledger: TransitionLedger,
        *,
        campaign_id: Optional[str] = None,
    ) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        if not isinstance(ledger, TransitionLedger):
            raise TypeError("ledger must be a TransitionLedger")
        self._journal = journal_
        self._ledger = ledger
        self._campaign_id = campaign_id

    def record(self, transition: Transition) -> Transition:
        with self._journal.write_lock():
            bound = transition
            if transition.to_state in STOP_STATES:
                entry = self._journal.append(journal.KIND_STOP_STATE, {
                    "state": transition.to_state,
                    "from_state": transition.from_state,
                    "trigger": transition.trigger,
                    "reason": transition.reason,
                    "transition_seq": transition.seq,
                    "at": transition.at,
                    "detail": dict(transition.detail),
                }, campaign_id=self._campaign_id)
                bound = replace(
                    transition,
                    journal_kind=entry.kind,
                    journal_event_id=entry.event_id,
                )
            return self._ledger.append(bound)


# =============================================================================
# Operator control latch (§4 invariant 19)
# =============================================================================

@dataclass(frozen=True)
class ControlLatch:
    """An operator control, as it exists ON DISK.

    This type is never held as attribute state by anything in this module. It is
    read from the file at the moment it is needed and dropped. That is the whole
    defense against the AutoPilot shape: there is no in-memory copy to write back
    over the operator's change, because there is no in-memory copy.
    """

    control: str
    control_id: str
    received_at: str
    requested_by: str
    reason: str
    disposition: str
    latched_at: str
    acked_event_id: str

    def __post_init__(self) -> None:
        if self.control not in CONTROLS:
            raise ValueError(f"control: {self.control!r} not in {sorted(CONTROLS)}")
        for name in ("control_id", "received_at", "requested_by", "reason",
                     "disposition", "latched_at", "acked_event_id"):
            _require_text(getattr(self, name), name)

    @property
    def halting(self) -> bool:
        return self.control in HALTING_CONTROLS

    def to_dict(self) -> dict:
        return {
            "control": self.control,
            "control_id": self.control_id,
            "received_at": self.received_at,
            "requested_by": self.requested_by,
            "reason": self.reason,
            "disposition": self.disposition,
            "latched_at": self.latched_at,
            "acked_event_id": self.acked_event_id,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "ControlLatch":
        if not isinstance(obj, Mapping):
            raise ControlLatchError("latch file does not hold a JSON object")
        missing = sorted({
            "control", "control_id", "received_at", "requested_by", "reason",
            "disposition", "latched_at", "acked_event_id",
        } - set(obj))
        if missing:
            raise ControlLatchError(f"latch file is missing {missing}")
        try:
            return ControlLatch(**{key: obj[key] for key in (
                "control", "control_id", "received_at", "requested_by", "reason",
                "disposition", "latched_at", "acked_event_id",
            )})
        except (TypeError, ValueError) as exc:
            raise ControlLatchError(f"latch file is malformed: {exc}") from exc


class ControlLatchStore:
    """The latch file. Every read hits the disk; there is no cache to stale.

    The API is deliberately missing the one method that caused the original
    defect: there is no `save(current_state)`. A latch is created by `latch()`
    from arguments, and removed by `release()` naming the id it removes. Nothing
    can write back a snapshot it took earlier, because nothing can express it.
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(_require_text(path, "latch path"))

    def read(self) -> Optional[ControlLatch]:
        """The latch on disk right now, or None. Never memoized."""
        try:
            with open(self.path, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        if not raw.strip():
            raise ControlLatchError(
                f"{self.path}: latch file is empty; an unreadable latch is not an "
                "absent one"
            )
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlLatchError(f"{self.path}: unparseable latch: {exc}") from exc
        return ControlLatch.from_dict(obj)

    def latch(self, latch: ControlLatch) -> ControlLatch:
        """Write a NEW latch. Refuses to overwrite a different one."""
        if not isinstance(latch, ControlLatch):
            raise TypeError("latch must be a ControlLatch")
        existing = self.read()
        if existing is not None and existing.control_id != latch.control_id:
            raise ControlLatchError(
                f"{self.path}: control {existing.control_id!r} ({existing.control}) "
                f"is already latched; resume it before latching "
                f"{latch.control_id!r}. Overwriting an operator's control is the "
                "defect this refusal exists for"
            )
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        _atomic_write_json(self.path, latch.to_dict())
        return latch

    def release(self, control_id: str) -> ControlLatch:
        """Remove the latch named by `control_id`. Refuses on any mismatch."""
        _require_text(control_id, "control_id")
        existing = self.read()
        if existing is None:
            raise ControlLatchError(
                f"{self.path}: no latch to release; releasing a control that was "
                "never latched would report a halt as cleared that never held"
            )
        if existing.control_id != control_id:
            raise ControlLatchError(
                f"{self.path}: latched control is {existing.control_id!r}, not "
                f"{control_id!r}"
            )
        os.unlink(self.path)
        _fsync_dir(os.path.dirname(self.path))
        return existing


# =============================================================================
# Anchor identity (§8.9, AK-D22)
# =============================================================================

@dataclass(frozen=True)
class AnchorIdentity:
    """Production identity of one source tree, per backend it serves.

    §8.9: an emergency hot-fix, a rollback, or any operator action that repoints
    a production symlink leaves every in-flight champion forked from a dead
    anchor — *"every ratio in the journal has a denominator that no longer
    exists"*. This is the value recorded at BOOTSTRAP and re-compared at every
    campaign boundary.
    """

    source_tree: str
    branch: str
    commit: str
    binary_sha256: Mapping[str, str]
    linkage_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_text(self.source_tree, "source_tree")
        _require_text(self.branch, "branch")
        if not isinstance(self.commit, str) or not _COMMIT_RE.match(self.commit):
            raise ValueError("commit must be a 40-character lowercase hex sha")
        for name in ("binary_sha256", "linkage_sha256"):
            table = getattr(self, name)
            if not isinstance(table, Mapping) or not table:
                raise ValueError(f"{name} must be a non-empty backend -> sha256 map")
            for backend, digest in table.items():
                if backend not in schemas.BACKENDS:
                    raise ValueError(
                        f"{name}: {backend!r} is not a declared backend "
                        f"{sorted(schemas.BACKENDS)}"
                    )
                if not isinstance(digest, str) or not _SHA256_RE.match(digest):
                    raise ValueError(f"{name}[{backend}] must be a lowercase sha256")
        if set(self.binary_sha256) != set(self.linkage_sha256):
            raise ValueError(
                "binary_sha256 and linkage_sha256 must cover the same backends; a "
                "backend with a binary hash and no linkage hash is half an identity"
            )

    @property
    def backends(self) -> tuple:
        return tuple(sorted(self.binary_sha256))

    def to_dict(self) -> dict:
        return {
            "source_tree": self.source_tree,
            "branch": self.branch,
            "commit": self.commit,
            "binary_sha256": dict(self.binary_sha256),
            "linkage_sha256": dict(self.linkage_sha256),
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "AnchorIdentity":
        if not isinstance(obj, Mapping):
            raise TypeError("anchor identity must be a mapping")
        missing = sorted(
            {"source_tree", "branch", "commit", "binary_sha256", "linkage_sha256"}
            - set(obj)
        )
        if missing:
            raise ValueError(f"anchor identity is missing {missing}")
        return AnchorIdentity(
            source_tree=obj["source_tree"],
            branch=obj["branch"],
            commit=obj["commit"],
            binary_sha256=dict(obj["binary_sha256"]),
            linkage_sha256=dict(obj["linkage_sha256"]),
        )


class AnchorIdentityStore:
    """The anchor recorded at BOOTSTRAP, on disk, re-read at every boundary."""

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(_require_text(path, "anchor path"))

    def read(self) -> Optional[AnchorIdentity]:
        try:
            with open(self.path, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerError(
                f"{self.path}: unparseable anchor identity: {exc}"
            ) from exc
        return AnchorIdentity.from_dict(obj)

    def record(self, anchor: AnchorIdentity) -> AnchorIdentity:
        if not isinstance(anchor, AnchorIdentity):
            raise TypeError("anchor must be an AnchorIdentity")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        _atomic_write_json(self.path, anchor.to_dict())
        return anchor


def check_anchor_identity(
    recorded: Optional[AnchorIdentity], observed: Optional[AnchorIdentity]
) -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on "is this still the same anchor?" (§8.9).

    A detected mismatch is a FACT and outranks an incomplete observation, so a
    concrete difference reports FAIL even when another backend could not be
    observed. Absence of an observation is COULD_NOT_CHECK and never PASS: an
    anchor nobody looked at is not an anchor that did not move.
    """
    if recorded is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no anchor identity was recorded at BOOTSTRAP; there is nothing to "
            "compare this campaign's ratios against",
        ))
    if observed is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no anchor identity was observed at this boundary; not observing is "
            "not evidence that production stayed put",
        ))

    mismatches: list = []
    unobserved: list = []
    for name in ("source_tree", "branch", "commit"):
        was, now = getattr(recorded, name), getattr(observed, name)
        if was != now:
            mismatches.append(f"{name}: recorded {was!r}, observed {now!r}")
    for name in ("binary_sha256", "linkage_sha256"):
        was_table = getattr(recorded, name)
        now_table = getattr(observed, name)
        for backend in sorted(was_table):
            if backend not in now_table:
                unobserved.append(f"{name}[{backend}] was not observed")
                continue
            if was_table[backend] != now_table[backend]:
                mismatches.append(
                    f"{name}[{backend}]: recorded {was_table[backend][:12]}, "
                    f"observed {now_table[backend][:12]}"
                )
        for backend in sorted(set(now_table) - set(was_table)):
            mismatches.append(
                f"{name}[{backend}] appeared since BOOTSTRAP; the tree serves a "
                "backend the recorded anchor does not describe"
            )

    if mismatches:
        return schemas.Check(schemas.FAIL, tuple(mismatches + unobserved))
    if unobserved:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unobserved))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Deterministic stop validation (§8.10)
# =============================================================================

def _closure_text_surfaces(
    reason: Any, detail: Mapping[str, Any]
) -> list:
    """Every free-text surface a closure claim can be written on, with its path.

    Deliberately broad. The reserved words exist because closure inflation
    survives explicit awareness of the rule, and an author who cannot write
    "exhausted" in `reason` writes it in a `sub_scope` instead.
    """
    surfaces: list = [("reason", reason if isinstance(reason, str) else "")]
    for key, gate_key in (("closed", "gates_met"), ("deferred", "gates_unrun")):
        entries = detail.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                continue
            for field_name in ("sub_scope", "reason", "note"):
                value = entry.get(field_name)
                if isinstance(value, str):
                    surfaces.append((f"{key}[{index}].{field_name}", value))
            gates = entry.get(gate_key)
            if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
                for gate_index, gate in enumerate(gates):
                    if isinstance(gate, str):
                        surfaces.append((f"{key}[{index}].{gate_key}[{gate_index}]", gate))
    layers = detail.get("hierarchy_layers_considered")
    if isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
        for index, layer in enumerate(layers):
            if isinstance(layer, str):
                surfaces.append((f"hierarchy_layers_considered[{index}]", layer))
    return surfaces


def check_closure_enumeration(
    reason: str, detail: Optional[Mapping[str, Any]]
) -> schemas.Check:
    """§8.10: closure must ENUMERATE what was closed and what was not.

    *"closed for sub-scope X (gates A, B, C met); sub-scope Y deferred (gates D,
    E un-run)"*. Bare "exhausted" and "all paths" are reserved words. An empty
    `deferred` list is a legitimate answer; an ABSENT one is not, because the
    claim being made is precisely about what remains.

    The reserved-word scan covers the ENUMERATION as well as the reason. It did
    not, and a scan that stops at the field the author was thinking about is the
    field the habit moves into: "all paths" reads the same in a `sub_scope` as it
    does in `reason`, and the sub-scope is where a closure claim actually lives.
    """
    if not isinstance(detail, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no detail mapping was supplied, so the enumeration cannot be read",
        ))
    reasons: list = []
    for where, text in _closure_text_surfaces(reason, detail):
        for phrase in reserved_closure_findings(text):
            reasons.append(
                f"{where} contains the reserved closure phrase {phrase!r}; §8.10 "
                "rejects it in favour of an enumeration"
            )

    closed = detail.get("closed")
    if not isinstance(closed, Sequence) or isinstance(closed, (str, bytes)) or not closed:
        reasons.append(
            "closed: required, a non-empty list of {sub_scope, gates_met} entries"
        )
    else:
        for index, entry in enumerate(closed):
            if not isinstance(entry, Mapping):
                reasons.append(f"closed[{index}]: must be a mapping")
                continue
            if not isinstance(entry.get("sub_scope"), str) or not entry["sub_scope"].strip():
                reasons.append(f"closed[{index}].sub_scope: required and non-empty")
            gates = entry.get("gates_met")
            if (not isinstance(gates, Sequence) or isinstance(gates, (str, bytes))
                    or not gates):
                reasons.append(
                    f"closed[{index}].gates_met: required, a non-empty list of the "
                    "gates that were actually met"
                )

    deferred = detail.get("deferred")
    if deferred is None:
        reasons.append(
            "deferred: required (an empty list is an answer, an absent one is "
            "closure inflation)"
        )
    elif not isinstance(deferred, Sequence) or isinstance(deferred, (str, bytes)):
        reasons.append("deferred: must be a list")
    else:
        for index, entry in enumerate(deferred):
            if not isinstance(entry, Mapping):
                reasons.append(f"deferred[{index}]: must be a mapping")
                continue
            if not isinstance(entry.get("sub_scope"), str) or not entry["sub_scope"].strip():
                reasons.append(f"deferred[{index}].sub_scope: required and non-empty")
            gates = entry.get("gates_unrun")
            if (not isinstance(gates, Sequence) or isinstance(gates, (str, bytes))
                    or not gates):
                reasons.append(
                    f"deferred[{index}].gates_unrun: required, a non-empty list of "
                    "the gates that were NOT run"
                )

    if reasons:
        # Deduplicated: 'exhausted' is a substring of 'exhausted all', so one
        # phrase can match twice and the same defect must read the same way once.
        return schemas.Check(schemas.FAIL, tuple(dict.fromkeys(reasons)))
    return schemas.Check(schemas.PASS)


def check_stop_evidence(
    state: str, reason: str, detail: Optional[Mapping[str, Any]]
) -> schemas.Check:
    """Everything §8.10 demands of a stop, checked deterministically.

    Origin is not an input. An operator's stop request and a planner's are held
    to the same evidence, because §8.4.0's grading rule — *"can never be promoted
    by its origin"* — is about exactly this temptation.
    """
    if state not in STOP_STATES:
        return schemas.Check(schemas.FAIL, (
            f"{state!r} is not a declared stop state {list(STOP_STATES)}",
        ))
    if not isinstance(reason, str) or not reason.strip():
        return schemas.Check(schemas.FAIL, ("reason: required and non-empty",))
    if not isinstance(detail, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"detail is {type(detail).__name__}, not a mapping; the evidence "
            f"{state} requires cannot be read",
        ))

    reasons: list = []
    for key in STOP_EVIDENCE_REQUIREMENTS[state]:
        if key not in detail:
            reasons.append(f"detail.{key}: required for {state}")
            continue
        value = detail[key]
        if value is None or (isinstance(value, (str, bytes)) and not value.strip()):
            reasons.append(f"detail.{key}: required for {state} and must not be empty")

    if state in SEARCH_CLOSURE_STOPS:
        enumeration = check_closure_enumeration(reason, detail)
        if enumeration.outcome != schemas.PASS:
            reasons.extend(enumeration.reasons)

    if state == PLATEAU_STOP:
        # §8.10: "plateau means the search is done, degraded means the searcher is
        # broken, and conflating them once cost this project months of paid
        # no-ops". A plateau that never looked at planner health is that
        # conflation, so the receipt is mandatory. The THRESHOLDS are derived by
        # the campaign calibration and are not this module's business (§8.4.1) —
        # what is checked here is that the question was asked and answered.
        health = detail.get("planner_health")
        if not isinstance(health, Mapping):
            reasons.append(
                "detail.planner_health: required, a mapping recording that "
                "PLANNER_DEGRADED was ruled out before declaring a plateau"
            )
        else:
            for key in ("proposal_skipped_count", "repeated_fingerprint_count"):
                value = health.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    reasons.append(
                        f"detail.planner_health.{key}: required, a non-negative int"
                    )
            if health.get("degraded_ruled_out") is not True:
                reasons.append(
                    "detail.planner_health.degraded_ruled_out: must be exactly True; "
                    "a plateau declared without ruling out a broken searcher is the "
                    "§8.10 conflation"
                )

    if state == OPERATOR_INPUT_REQUIRED:
        options = detail.get("options")
        if (not isinstance(options, Sequence) or isinstance(options, (str, bytes))
                or len(options) < 2):
            reasons.append(
                "detail.options: an operator decision package carries 2-4 concrete "
                "options; a single option is not a decision and an open-ended "
                "question is not an escalation (§18 item 7)"
            )

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


@dataclass(frozen=True)
class StopRequest:
    """A REQUEST to stop. §8.10: *"The LLM may request a stop. The controller
    owns disposition from records."* Constructing one decides nothing."""

    state: str
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    origin: str = "controller"

    def __post_init__(self) -> None:
        if self.origin not in STOP_REQUEST_ORIGINS:
            raise ValueError(
                f"origin: {self.origin!r} not in {sorted(STOP_REQUEST_ORIGINS)}"
            )


# =============================================================================
# Reports
# =============================================================================

@dataclass(frozen=True)
class RestoreReport:
    """What the ledger said the machine's position was, at construction.

    It carries `latch_present` and `latch_control_id`, deliberately NOT the
    `ControlLatch` itself: a report is held by the machine, and a held latch is
    the cached copy this module exists to make impossible.
    """

    state: str
    seq: int
    transition_count: int
    discarded_tail_bytes: int
    latch_present: bool
    latch_control_id: Optional[str]


@dataclass(frozen=True)
class IterationDecision:
    """The top-of-iteration verdict. `proceed` is the only thing callers act on."""

    proceed: bool
    state: str
    reason: str
    control: Optional[str] = None
    control_id: Optional[str] = None


@dataclass(frozen=True)
class BootstrapReport:
    """§8.2's outcome. `view_check` is the step-10 consistency verdict."""

    view_check: schemas.Check
    deliberate_rebase: bool
    event_count: int
    anchor: AnchorIdentity
    transition: Transition


class ReleaseGateRunner(Protocol):
    """The AK5 seam for `SEAL -> T3_RELEASE_GATE -> PACKAGE`.

    Nothing in AK4 implements it. Until one is wired,
    `ControllerStateMachine.run_release_gate()` refuses the tier by calling
    `evaluator.api.admit_tier("T3")`, which raises `TierNotOwned` naming AK5 —
    the same refusal the search evaluator makes, from the same table, so the two
    planes cannot drift into disagreeing about who owns T3.
    """

    def evaluate_release(self, request: Any) -> Any:
        ...


# =============================================================================
# The machine
# =============================================================================

LEDGER_FILENAME = "transitions.jsonl"
LATCH_FILENAME = "control.latch.json"
ANCHOR_FILENAME = "anchor.json"


class ControllerStateMachine:
    """The AK4 state machine. Deterministic, journal-then-act, terminal at stops.

    Construction is not lazy: the ledger is read and the position restored in
    `__init__`, because a machine that has to be told to catch up is a machine
    somebody forgets to tell.

    `__slots__` is load-bearing, not a micro-optimization: there is no slot for a
    control latch, so a future edit that tries to cache one fails at runtime
    instead of quietly reintroducing the defect.
    """

    __slots__ = (
        "_journal", "_root", "_ledger", "_latch_store", "_anchor_store",
        "_recorder", "_clock", "_campaign_id", "_release_gate", "_state", "_seq",
        "_restore_report",
    )

    def __init__(
        self,
        *,
        journal_: journal.Journal,
        root: str,
        campaign_id: Optional[str] = None,
        recorder: Optional[TransitionRecorder] = None,
        clock: Optional[Callable[[], str]] = None,
        release_gate: Optional[ReleaseGateRunner] = None,
    ) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        # An uninitialized journal is a missing input, not a state to create:
        # this plane records THROUGH the journal and does not own its layout.
        try:
            journal_.shards()
        except journal.JournalError as exc:
            raise ControllerError(
                f"{journal_.root}: the journal is not readable "
                f"({type(exc).__name__}: {exc}); call Journal.initialize() before "
                "constructing a controller over it"
            ) from exc
        self._journal = journal_
        self._root = os.path.abspath(_require_text(root, "root"))
        self._campaign_id = campaign_id
        self._clock = clock if clock is not None else _iso_now
        self._release_gate = release_gate

        os.makedirs(self._root, exist_ok=True)
        self._ledger = TransitionLedger(os.path.join(self._root, LEDGER_FILENAME))
        self._ledger.initialize()
        self._latch_store = ControlLatchStore(os.path.join(self._root, LATCH_FILENAME))
        self._anchor_store = AnchorIdentityStore(os.path.join(self._root, ANCHOR_FILENAME))
        self._recorder = recorder if recorder is not None else JournalTransitionRecorder(
            journal_, self._ledger, campaign_id=campaign_id
        )

        read = self._ledger.read()
        if read.transitions:
            last = read.transitions[-1]
            self._state = last.to_state
            self._seq = last.seq
        else:
            self._state = BOOTSTRAP
            self._seq = 0
        latch = self._latch_store.read()
        self._restore_report = RestoreReport(
            state=self._state,
            seq=self._seq,
            transition_count=len(read.transitions),
            discarded_tail_bytes=read.discarded_tail_bytes,
            latch_present=latch is not None,
            latch_control_id=None if latch is None else latch.control_id,
        )

    # ---- position ---------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def root(self) -> str:
        return self._root

    @property
    def restore_report(self) -> RestoreReport:
        return self._restore_report

    @property
    def latch_store(self) -> ControlLatchStore:
        return self._latch_store

    @property
    def anchor_store(self) -> AnchorIdentityStore:
        return self._anchor_store

    @property
    def ledger(self) -> TransitionLedger:
        return self._ledger

    def is_stopped(self) -> bool:
        return self._state in STOP_STATES

    # ---- the one transition primitive -------------------------------------

    def transition(
        self,
        to_state: str,
        *,
        trigger: str,
        reason: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Transition:
        """Journal, then act. The ONLY way the machine's position changes.

        The order is not a style choice: recording after the fact leaves a window
        in which the machine is ahead of its record, and a crash inside that
        window is unrecoverable because nothing distinguishes "did the work" from
        "was about to". If the recorder raises, the state is untouched.
        """
        if to_state not in STATES:
            raise IllegalTransition(f"{to_state!r} is not a declared state")
        legal = EDGES[self._state]
        if not legal:
            raise IllegalTransition(
                f"{self._state} is terminal ({STOP_RECOVERY.get(self._state)}); no "
                f"transition leaves it. Recovery is reopen(), which is not an edge"
            )
        if to_state not in legal:
            raise IllegalTransition(
                f"{self._state} -> {to_state} is not a declared edge; legal targets "
                f"are {list(legal)}"
            )
        if to_state == T3_RELEASE_GATE and self._release_gate is None:
            # Structural, not procedural: the refusal lives on the EDGE, so it
            # cannot be walked around by calling the primitive instead of
            # `run_release_gate()`. A guard only the polite path passes is not a
            # guard.
            evaluator_api.admit_tier("T3")
        return self._journal_then_act(
            to_state, trigger=trigger, reason=reason, detail=detail
        )

    def _journal_then_act(
        self,
        to_state: str,
        *,
        trigger: str,
        reason: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Transition:
        candidate = Transition(
            seq=self._seq + 1,
            from_state=self._state,
            to_state=to_state,
            trigger=trigger,
            reason=reason,
            at=self._clock(),
            detail=dict(detail or {}),
        )
        try:
            recorded = self._recorder.record(candidate)
        except ControllerError:
            raise
        except Exception as exc:
            # Wrapped, never swallowed: the caller must be able to tell "the
            # transition did not happen" from "the transition happened and
            # something else went wrong", and only this frame knows which.
            raise TransitionNotRecorded(
                f"{self._state} -> {to_state} was NOT recorded ({type(exc).__name__}: "
                f"{exc}); the transition did not happen"
            ) from exc
        if not isinstance(recorded, Transition):
            raise TransitionNotRecorded(
                f"recorder returned {type(recorded).__name__}, not a Transition; a "
                "transition whose record cannot be identified did not happen"
            )
        if recorded.to_state != to_state or recorded.seq != candidate.seq:
            raise TransitionNotRecorded(
                "recorder returned a transition that is not the one it was asked to "
                f"record ({recorded.from_state} -> {recorded.to_state} #{recorded.seq} "
                f"vs {candidate.from_state} -> {candidate.to_state} #{candidate.seq})"
            )
        self._state = to_state
        self._seq = recorded.seq
        return recorded

    # ---- operator controls (invariant 19) ---------------------------------

    def submit_control(
        self,
        control: str,
        *,
        control_id: str,
        requested_by: str,
        reason: str,
        received_at: Optional[str] = None,
    ) -> ControlLatch:
        """Acknowledge in the journal, THEN latch on disk. Both under the lock.

        Ack first: a latch nobody acknowledged cannot be audited, whereas an ack
        whose latch is missing is DETECTABLE — `_verify_controls()` treats it as
        `UnackedControlError` on the next iteration rather than as "no control
        pending". Of the two crash windows, only one is recoverable, so the
        write order puts the crash in that one.
        """
        if control not in CONTROLS:
            raise ValueError(f"control: {control!r} not in {sorted(CONTROLS)}")
        if control == CONTROL_RESUME:
            raise ValueError(
                "resume is not submitted as a control; call resume_control(), which "
                "must name the latched control it releases"
            )
        _require_text(control_id, "control_id")
        _require_text(requested_by, "requested_by")
        _require_text(reason, "reason")
        stamp = received_at if received_at is not None else self._clock()
        _require_text(stamp, "received_at")

        with self._journal.write_lock():
            existing = self._latch_store.read()
            if existing is not None and existing.control_id == control_id:
                return existing
            entry = self._journal.append_control_ack(
                control=control,
                control_id=control_id,
                received_at=stamp,
                disposition=DISPOSITION_LATCHED,
            )
            return self._latch_store.latch(ControlLatch(
                control=control,
                control_id=control_id,
                received_at=stamp,
                requested_by=requested_by,
                reason=reason,
                disposition=DISPOSITION_LATCHED,
                latched_at=self._clock(),
                acked_event_id=entry.event_id,
            ))

    def resume_control(
        self, control_id: str, *, requested_by: str, reason: str
    ) -> ControlLatch:
        """Clear a latched halt. Only an operator resumes; the loop cannot.

        §8.10: an operator stop *"stays stopped across restart until resumed"*.
        The journal keeps the released ack, so "was this halt ever cleared, and
        by whom" is answerable from the record.
        """
        _require_text(control_id, "control_id")
        _require_text(requested_by, "requested_by")
        _require_text(reason, "reason")
        with self._journal.write_lock():
            released = self._latch_store.read()
            if released is None or released.control_id != control_id:
                raise ControlLatchError(
                    f"no latched control {control_id!r} to resume; resuming a halt "
                    "that is not held would report a clear that never happened"
                )
            self._journal.append_control_ack(
                control=CONTROL_RESUME,
                control_id=control_id,
                received_at=self._clock(),
                disposition=DISPOSITION_RELEASED,
            )
            return self._latch_store.release(control_id)

    def _verify_controls(self) -> Optional[ControlLatch]:
        """Reconcile the disk latch with the journal's acks. Caller holds the lock.

        Returns the latch (or None) and RAISES on either failure shape. This is
        the routine invariant 19 is about, and it is why the latch is a return
        value rather than a field.
        """
        latch = self._latch_store.read()
        acks = [
            entry for entry in self._journal.read_all()
            if entry.kind == journal.KIND_OPERATOR_CONTROL_ACK
        ]
        ack_ids = {entry.event_id for entry in acks}
        released_ids = {
            entry.payload.get("control_id") for entry in acks
            if entry.payload.get("disposition") == DISPOSITION_RELEASED
        }
        halting_ids = {
            entry.payload.get("control_id") for entry in acks
            if entry.payload.get("control") in HALTING_CONTROLS
        }
        outstanding = {cid for cid in (halting_ids - released_ids) if cid is not None}

        if latch is not None:
            if latch.acked_event_id not in ack_ids:
                raise UnackedControlError(
                    f"control {latch.control_id!r} ({latch.control}) is latched on "
                    f"disk citing ack {latch.acked_event_id!r}, which resolves to no "
                    "OPERATOR_CONTROL_ACK in the journal. An unacked control is a "
                    "hard failure (invariant 19)"
                )
            matching = [
                entry for entry in acks
                if entry.event_id == latch.acked_event_id
                and entry.payload.get("control_id") == latch.control_id
            ]
            if not matching:
                raise UnackedControlError(
                    f"latched control {latch.control_id!r} cites ack "
                    f"{latch.acked_event_id!r}, which acknowledges a DIFFERENT "
                    "control; the latch and the record do not describe one command"
                )
            outstanding.discard(latch.control_id)

        if outstanding:
            raise UnackedControlError(
                f"halting control(s) {sorted(outstanding)} are acknowledged in the "
                "journal and held by no latch on disk. This is the "
                "crash-between-ack-and-latch window; treating it as 'no control "
                "pending' is exactly how a pause becomes a silent no-op"
            )
        return latch

    def begin_iteration(self) -> IterationDecision:
        """Top of every iteration: re-read the latch FROM DISK under the lock.

        Not from a field, not from a cached snapshot, not from whatever the last
        iteration believed — from the file, every time. If a halting control is
        latched, the machine transitions to `OPERATOR_STOP_REQUESTED` (journaling
        first, as always) and reports `proceed=False`. A halt therefore survives
        restart: the next process constructs a new machine, reads the same file,
        and stops again.
        """
        with self._journal.write_lock():
            latch = self._verify_controls()
            if latch is not None and latch.halting:
                if self._state not in STOP_STATES:
                    self._journal_then_act(
                        OPERATOR_STOP_REQUESTED,
                        trigger="operator_control",
                        reason=(
                            f"operator control {latch.control!r} latched on disk by "
                            f"{latch.requested_by}: {latch.reason}"
                        ),
                        detail={
                            "control": latch.control,
                            "control_id": latch.control_id,
                            "requested_by": latch.requested_by,
                            "received_at": latch.received_at,
                            "acked_event_id": latch.acked_event_id,
                        },
                    )
                return IterationDecision(
                    proceed=False,
                    state=self._state,
                    reason=(
                        f"halted by operator control {latch.control!r} "
                        f"({latch.control_id}); resume is operator authority"
                    ),
                    control=latch.control,
                    control_id=latch.control_id,
                )
            if self._state in STOP_STATES:
                return IterationDecision(
                    proceed=False,
                    state=self._state,
                    reason=(
                        f"{self._state} is terminal; recovery class "
                        f"{STOP_RECOVERY[self._state]}"
                    ),
                )
            return IterationDecision(
                proceed=True, state=self._state, reason="no control latched"
            )

    # ---- BOOTSTRAP (§8.2) -------------------------------------------------

    def bootstrap(
        self,
        *,
        anchor: AnchorIdentity,
        views: Optional[journal.Views] = None,
        deliberate_rebase: bool = False,
        rebase_reason: Optional[str] = None,
    ) -> BootstrapReport:
        """§8.2 steps 10-12, and the anchor record §8.9 re-verifies against.

        `views` is the derived view AS THE CONTROLLER HOLDS IT. It defaults to a
        fresh rebuild, but the failure this step exists for is a *derived store*
        that came up empty while the journal was full — which a fresh rebuild can
        never reproduce. Passing the real store is the point.

        Refuses on disagreement. `deliberate_rebase=True` with a stated
        `rebase_reason` is the explicit escape, journaled as `VIEW_REBASED`, so
        an intentional wipe is never indistinguishable from the AutoPilot loss.
        """
        if self._state != BOOTSTRAP:
            raise IllegalTransition(
                f"bootstrap() runs in {BOOTSTRAP}, not {self._state}"
            )
        if not isinstance(anchor, AnchorIdentity):
            raise TypeError("anchor must be an AnchorIdentity")

        with self._journal.write_lock():
            latch = self._verify_controls()
            if latch is not None and latch.halting:
                raise ControlLatchError(
                    f"cannot BOOTSTRAP while control {latch.control_id!r} "
                    f"({latch.control}) is latched; a halt survives restart until an "
                    "operator resumes it"
                )
            events = self._journal.read_all()
            # One path, not two. `Journal.bootstrap_views()` is the no-argument
            # convenience over exactly these two primitives and cannot accept a
            # SUPPLIED view, which is the case that matters here — so the
            # primitives are composed directly rather than the convenience being
            # called for one branch and reimplemented for the other.
            if views is None:
                views = journal.rebuild_views(events)
            elif not isinstance(views, journal.Views):
                raise TypeError("views must be a journal.Views")
            try:
                view_check = journal.assert_views_consistent(
                    events, views,
                    deliberate_rebase=deliberate_rebase,
                    rebase_reason=rebase_reason,
                )
            except journal.ViewConsistencyError as exc:
                raise BootstrapRefused(
                    "BOOTSTRAP refuses to start: " + str(exc)
                ) from exc
            if view_check.outcome != schemas.PASS:
                self._journal.append(journal.KIND_VIEW_REBASED, {
                    "rebase_reason": rebase_reason,
                    "suppressed_reasons": list(view_check.reasons),
                    "entry_count": len(events),
                })

            self._anchor_store.record(anchor)
            transition = self._journal_then_act(
                DISCOVER,
                trigger="bootstrap_ready",
                reason=(
                    "§8.2 READY: journal/derived-view consistency asserted and anchor "
                    "identity recorded"
                ),
                detail={
                    "event_count": len(events),
                    "view_check": view_check.outcome,
                    "deliberate_rebase": bool(deliberate_rebase),
                    "anchor": anchor.to_dict(),
                    "anchor_content_hash": schemas.content_hash(anchor.to_dict()),
                },
            )
        return BootstrapReport(
            view_check=view_check,
            deliberate_rebase=bool(deliberate_rebase),
            event_count=len(events),
            anchor=anchor,
            transition=transition,
        )

    # ---- campaign boundary (§8.9, AK-D22) ---------------------------------

    def campaign_boundary(self, *, observed_anchor: AnchorIdentity) -> schemas.Check:
        """Re-verify anchor identity. FAIL => `ANCHOR_MOVED`, and it is terminal.

        §8.9: a hot-fix or a rollback leaves every in-flight ratio with a
        denominator that no longer exists. COULD_NOT_CHECK raises
        `AnchorUncheckable` rather than continuing, because the fail-open branch
        here is worth more than the check.

        This method detects and stops. Superseding the affected T1/T2 records
        (§8.9 items 2-5) belongs to the memory-update module and to AK5's
        re-anchor path; the stop's `detail` carries both identities so that work
        has what it needs.
        """
        if not isinstance(observed_anchor, AnchorIdentity):
            raise TypeError("observed_anchor must be an AnchorIdentity")
        recorded = self._anchor_store.read()
        check = check_anchor_identity(recorded, observed_anchor)
        if check.outcome == schemas.COULD_NOT_CHECK:
            raise AnchorUncheckable(
                "anchor identity could not be compared at a campaign boundary: "
                + "; ".join(check.reasons)
            )
        if check.outcome == schemas.FAIL:
            affected = sorted(
                set(recorded.backends) | set(observed_anchor.backends)
            ) if recorded is not None else list(observed_anchor.backends)
            self.stop(
                ANCHOR_MOVED,
                reason=(
                    "production identity changed outside a loop-initiated freeze: "
                    + "; ".join(check.reasons)
                ),
                detail={
                    "recorded_anchor": recorded.to_dict() if recorded else None,
                    "observed_anchor": observed_anchor.to_dict(),
                    "affected_backends": affected,
                    "supersession_marker": "superseded_by_anchor_move",
                    "operator_notice": (
                        "an unexpected anchor move usually means something happened "
                        "that the loop should not silently absorb (§8.9)"
                    ),
                },
                trigger="anchor_identity_check",
            )
        return check

    # ---- stops (§8.10) ----------------------------------------------------

    def stop(
        self,
        state: str,
        *,
        reason: str,
        detail: Mapping[str, Any],
        trigger: str = "controller",
    ) -> Transition:
        """Enter a stop state, after the deterministic evidence check passes."""
        check = check_stop_evidence(state, reason, detail)
        if check.outcome != schemas.PASS:
            raise StopEvidenceMissing(
                f"{state} refused ({check.outcome}): " + "; ".join(check.reasons)
            )
        return self.transition(state, trigger=trigger, reason=reason, detail=detail)

    def dispose_stop_request(self, request: StopRequest) -> Transition:
        """Dispose an LLM-originated stop request. The origin buys nothing.

        §8.10: *"The LLM may request a stop. The controller owns disposition from
        records."* This is the whole of that sentence's implementation — the
        request is re-validated against the same table an internally generated
        stop faces, and the transition is refused identically.
        """
        if not isinstance(request, StopRequest):
            raise TypeError("request must be a StopRequest")
        return self.stop(
            request.state,
            reason=request.reason,
            detail=request.detail,
            trigger=f"stop_request:{request.origin}",
        )

    # ---- recovery ---------------------------------------------------------

    def reopen(self, *, reason: str, authorized_by: str) -> Transition:
        """Leave a stop state. NOT an edge — `transition()` can never do this.

        Stop states are terminal for the loop; a reopen is an operator-authorized
        restart that re-enters at BOOTSTRAP, because every stop invalidated
        something §8.2 re-establishes. Refuses while any halt is latched, so
        `OPERATOR_STOP_REQUESTED` cannot be shrugged off without an operator
        actually resuming the control.
        """
        _require_text(reason, "reason")
        _require_text(authorized_by, "authorized_by")
        if self._state not in STOP_STATES:
            raise IllegalTransition(
                f"reopen() applies to a stop state, not {self._state}"
            )
        targets = REOPEN_EDGES[self._state]
        if not targets:
            raise IllegalTransition(
                f"{self._state} has recovery class {STOP_RECOVERY[self._state]} and "
                "is not reopenable; the package is handed to the operator and AK6 "
                "owns what follows"
            )
        with self._journal.write_lock():
            latch = self._verify_controls()
            if latch is not None and latch.halting:
                raise ControlLatchError(
                    f"cannot reopen while control {latch.control_id!r} "
                    f"({latch.control}) is latched; a halt survives restart until an "
                    "operator resumes it (invariant 19)"
                )
            return self._journal_then_act(
                targets[0],
                trigger="operator_reopen",
                reason=reason,
                detail={
                    "reopened_from": self._state,
                    "recovery_class": STOP_RECOVERY[self._state],
                    "authorized_by": authorized_by,
                },
            )

    # ---- release branch seam (AK5) ----------------------------------------

    def request_freeze(
        self, *, requested_by: str, reason: str, detail: Optional[Mapping[str, Any]] = None
    ) -> Transition:
        """CHAMPION_GUARD -> SEAL, on an OPERATOR request only (§1.3, invariant 5).

        `requested_by` is mandatory because the loop may not seal itself: T3 runs
        *"On operator request only"* (§9 tier table), and an unattributed freeze
        request is the loop deciding to release.
        """
        _require_text(requested_by, "requested_by")
        _require_text(reason, "reason")
        payload = dict(detail or {})
        payload["requested_by"] = requested_by
        return self.transition(
            SEAL, trigger="operator_freeze_request", reason=reason, detail=payload
        )

    def run_release_gate(self, request: Any) -> Any:
        """SEAL -> T3_RELEASE_GATE. Refuses the tier until AK5 wires a runner.

        The refusal comes from `evaluator.api.admit_tier`, not from a second
        opinion held here, so the two planes cannot drift into disagreeing about
        who owns T3.
        """
        if self._release_gate is None:
            evaluator_api.admit_tier("T3")
            raise AssertionError(  # pragma: no cover - admit_tier always raises on T3
                "admit_tier('T3') returned; the release-tier refusal is broken"
            )
        transition = self.transition(
            T3_RELEASE_GATE,
            trigger="release_gate",
            reason=f"T3 dispatched to the wired {type(self._release_gate).__name__}",
            detail={"tier": "T3", "owner": evaluator_api.RELEASE_TIER_OWNER},
        )
        outcome = self._release_gate.evaluate_release(request)
        return transition, outcome


# =============================================================================
# Structural audit
# =============================================================================

def audit_no_cached_control_state(obj: Any) -> schemas.Check:
    """Prove from the OBJECT that it holds no operator-control state.

    Invariant 19's failure was not a missing check; it was a cached copy that
    got written back. Prose cannot enforce its absence and a review enforces it
    once, so this reads the object's declared slots and instance dictionary and
    FAILs on anything that holds a `ControlLatch` or is named like a latch cache.

    COULD_NOT_CHECK when the object exposes neither slots nor a `__dict__` —
    an object that cannot be inspected has not been cleared.
    """
    findings: list = []
    inspected = 0

    slot_names: list = []
    for klass in type(obj).__mro__:
        declared = klass.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        slot_names.extend(declared)

    instance_dict = getattr(obj, "__dict__", None)
    names = list(slot_names) + list(instance_dict or ())
    for name in names:
        inspected += 1
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, ControlLatch):
            findings.append(
                f"{name}: holds a ControlLatch; the latch must be read from disk "
                "at the point of use, never retained"
            )
        elif (
            "latch" in name.lower()
            and not isinstance(value, (ControlLatchStore, bool, str, int, float))
            and value is not None
        ):
            # A scalar named `latch_present` or `latch_control_id` is a REPORT,
            # not a cache — it cannot be written back as a control. A structured
            # value under a latch-shaped name is the thing that can.
            findings.append(
                f"{name}: a latch-named attribute that is not a ControlLatchStore "
                f"(it is a {type(value).__name__}); the store reads through to disk, "
                "anything else is a cache"
            )

    if not inspected:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{type(obj).__name__} exposes neither __slots__ nor __dict__; it could "
            "not be inspected, which is not the same as being clean",
        ))
    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS)
