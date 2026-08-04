"""guards.py — the deterministic guards that own every AK4 stop condition (§8.10).

WHY THIS MODULE EXISTS
----------------------
`state_machine.py` owns *disposition*: it validates the SHAPE of a stop's evidence
and refuses an illegal edge. It does not, and must not, decide whether a stop
condition actually HOLDS. That decision is this module, and it is written against
five documented failures:

1. **Closure inflation.** §8.10: bare *"exhausted"* and *"all paths"* are reserved
   words the validator rejects, because closure inflation is *"this project's
   most-repeated documented habit, surviving even explicit awareness of the
   rule"*. `check_closure_language()` rejects the bare tokens as WORDS, not only
   inside the longer phrases `state_machine.RESERVED_CLOSURE_PHRASES` lists — a
   substring list that misses "the surface is exhausted" is exactly the shape of
   guard that fails the habit it was written for.
2. **A broken searcher reading as a finished one.** §8.10: *"plateau means the
   search is done, degraded means the searcher is broken, and conflating them
   once cost this project months of paid no-ops."* Here a closure decision cannot
   be CONSTRUCTED without a clean `guard_planner_degraded()` verdict computed
   over the SAME planner-health snapshot, bound by content hash. A stale clean
   verdict cannot be paired with a fresh plateau.
3. **A dead gate reading as an exhausted surface.** §12: *"The gate can still
   reject but can no longer promote, and reads as 'exhausted'."* Both closure
   guards require the accept-side historical-win replay to have PROMOTED within
   its cadence; `HISTORICAL_REPLAY_UNAVAILABLE` yields `COULD_NOT_EVALUATE`, never
   a closure.
4. **A spend breaker that halts the loop.** §2.5 row 4: a $250 budget that was
   only a status string, and *"a spend breaker whose naive form stopped the
   loop"*. The breaker here REFUSES metered drafting and forces local planning;
   only the declared ceiling STOPs, and the two are different functions.
5. **A permanent silent block.** §8.10: `EVALUATOR_COVERAGE_GAP` *"has an owner
   and a deadline, or it becomes a permanent silent block"*. A `CoverageGap`
   without both cannot be constructed, and the deadline is compared against a
   SUPPLIED `now` — never a clock this module reads.

THE AUTHORITY LINE
------------------
§8.10's last sentence is the whole contract: *"The LLM may request a stop. The
controller owns disposition from records."* `dispose_requested_stop()` is that
sentence's implementation. A request is honoured ONLY when a guard independently
reached the same stop state from records, and what is journaled is then the
GUARD's evidence, never the requester's detail. `StopRequest.origin` is recorded
and consulted by nothing (AK-D38: authorship is not evidence).

PURITY, AND WHY IT IS STRUCTURAL
--------------------------------
Every guard is a pure function of values the caller already read. There is no
clock (`now` is an argument), no randomness, no filesystem, no process, and no
network. `audit_no_write_process_or_wait_paths()` proves that from this module's
own AST — it forbids importing `os`/`time`/`random`/`subprocess`/`signal`, calling
`.now()`/`.utcnow()`, and calling `.sleep()`/`.wait()`/`.poll()`. That last group
is the machine-checked form of §8.10's *"`RESOURCE_UNAVAILABLE` — persist and
drain, never busy-wait"*: the DIRECTIVES vocabulary contains no WAIT, POLL, SLEEP
or RETRY member, so a busy-wait is not merely forbidden, it is inexpressible.

WHAT THIS MODULE IS NOT
-----------------------
It runs no inference, no benchmark, no build; it reads and writes no file; it
starts, stops and signals no process; it calls no model. It computes no
readiness value — §4 invariant 14 and P-AK-SEARCH-1's *advisory readiness signal*
clause put that in AK5's deterministic reducer; this module CONSUMES the reduced
series and never derives one.

Governing instrument: `measurement/protocols/kernel-research.md` (Annex K,
P-AK-SEARCH-1, RATIFIED 2026-08-03) — preconditions 3 (host-health tier), 7
(storage headroom and the expiry-backlog clause), 8 (declared campaign controls,
*"finite and strictly positive"*), the *Controls* clause (accept-side control and
its unavailable branch), and *No self-amendment* (a coverage gap is RECORDED and
blocks release; the instrument is not patched).

Owning design: `handoffs/active/autokernel-research-loop.md` §4 invariants 6/7/14,
§7.1 `budgets`/`stop_policy`, §8.4.1, §8.5.1, §8.10, §10.6, §10.7, §12, §17
(AK-D4, AK-D22, AK-D27, AK-D29, AK-D32, AK-D38), §18 item 7.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .. import schemas, storage
from ..evaluator import api as evaluator_api
from . import state_machine as sm

__all__ = [
    # errors
    "GuardError", "GuardInputError", "GuardEvidenceError", "GuardVocabularyError",
    "ParityHasNoMagnitude",
    # outcome vocabulary
    "CONTINUE", "REFUSE", "STOP", "COULD_NOT_EVALUATE", "OUTCOMES",
    # directives
    "DIRECTIVE_PERSIST_AND_DRAIN", "DIRECTIVE_ROOT_CAUSE_ANALYSIS",
    "DIRECTIVE_LOCAL_PLANNING_ONLY", "DIRECTIVE_REPAIR_FORBIDDEN",
    "DIRECTIVE_RECLAIM_EXPIRABLE_FIRST", "DIRECTIVE_RELEASE_BLOCKED",
    "DIRECTIVE_ESCALATE_TO_OPERATOR", "DIRECTIVE_REANCHOR",
    "DIRECTIVE_REQUEST_DENIED", "DIRECTIVES", "FORBIDDEN_DIRECTIVE_TOKENS",
    # guard identity and precedence
    "GUARD_INTEGRITY", "GUARD_ANCHOR", "GUARD_HOST_UPTIME", "GUARD_RESOURCE",
    "GUARD_STORAGE", "GUARD_BUDGET", "GUARD_COVERAGE", "GUARD_OPERATOR_INPUT",
    "GUARD_PLANNER_HEALTH", "GUARD_EXHAUSTED", "GUARD_PLATEAU",
    "GUARD_COMMAND_RETRY", "GUARD_REPAIR_CAP", "GUARD_SPEND_BREAKER",
    "GUARD_STOP_REQUEST", "GUARD_IDS",
    "PLATEAU_BASIS_MEASURED_IMPROVEMENT", "PLATEAU_BASIS_NO_DETECTABLE_EFFECT",
    "PLATEAU_BASES",
    "STOP_PRECEDENCE", "GUARD_BY_STOP", "NON_GUARD_STOPS", "ESCALATING_STOPS",
    # constitutional constants
    "HOST_UPTIME_CEILING_SECONDS", "MAX_COMMAND_RETRIES", "BUDGET_DIMENSIONS",
    "RESERVED_CLOSURE_WORDS",
    # decision package (§18 item 7)
    "DecisionOption", "DecisionPackage",
    # decisions
    "GuardDecision", "GuardDisposition",
    # observations
    "IntegritySignal", "IntegrityLedger", "HostHealth", "ResourceClaimObservation",
    "StorageObservation", "BudgetDimension", "BudgetLedger", "SpendBreakerPolicy",
    "CoverageGap", "OperatorQuestion", "PlannerHealth", "PlannerHealthPolicy",
    "ClosedSubScope", "DeferredSubScope", "ClosureLedger",
    "AcceptSideControlReceipt", "ACCEPT_CONTROL_PROMOTED",
    "ACCEPT_CONTROL_FAILED_TO_PROMOTE", "ACCEPT_CONTROL_UNAVAILABLE",
    "ACCEPT_CONTROL_STATUSES",
    "ReadinessSeriesEntry", "ReadinessObservation", "ParityObservation",
    "observation_from_fields", "PlateauPolicy",
    "CommandRetryLedger", "RepairLedger",
    # guards
    "guard_integrity", "guard_anchor_moved", "guard_host_uptime",
    "guard_resource_available", "guard_storage_headroom", "guard_budget",
    "guard_controller_spend", "guard_evaluator_coverage", "guard_operator_input",
    "guard_planner_degraded", "guard_exhausted_surface", "guard_plateau",
    "guard_command_retries", "guard_repair_cap",
    # disposition
    "dispose", "dispose_requested_stop",
    # deterministic checks and audits
    "check_closure_language", "audit_no_write_process_or_wait_paths",
    "audit_stop_coverage_totality", "audit_directive_vocabulary",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class GuardError(Exception):
    """Base for every refusal this module raises."""


class GuardInputError(GuardError):
    """A required input is missing, malformed, or self-contradictory.

    House rule: a MISSING input raises. `COULD_NOT_EVALUATE` is for an input that
    is present and honestly reports that it could not be observed — those are
    different facts and collapsing them is how a fail-open check is born.
    """


class GuardEvidenceError(GuardError):
    """A STOP decision was assembled whose evidence §8.10 would refuse.

    Raised at CONSTRUCTION, so a decision that the state machine would reject
    cannot exist as an object and therefore cannot be reported to an operator,
    logged as a stop, or counted in a disposition.
    """


class GuardVocabularyError(GuardError):
    """The declared vocabulary drifted from §8.10's enumeration.

    Raised at import. A stop state that no guard decides is a stop that can only
    be entered by narration, and a guard for a state §8.10 does not declare is a
    condition with two spellings.
    """


class ParityHasNoMagnitude(GuardError):
    """Something read a round that produced NO readiness as though it had one.

    A `ParityObservation` records a round in which every protected cell was
    measured and none was orderable. There is no magnitude on it, because the
    release plane refused to invent one: sub-floor does not mean zero, it means
    the sign and the size are both unknown. Substituting `0.0` would make the
    plateau rule trend a number nobody measured, and substituting `None` would
    make a completed round look like a missing one.
    """


# =============================================================================
# Outcome vocabulary — four, because "cannot evaluate" and "not now" differ
# =============================================================================

#: No stop condition holds and the campaign may proceed.
CONTINUE = "CONTINUE"
#: A specific ACTION is refused (another retry, another repair, metered drafting,
#: an allocation before reclamation) while the campaign continues. This is the
#: outcome §2.5 row 4 is about: the breaker forces local planning, it does not halt.
REFUSE = "REFUSE"
#: A §8.10 stop condition holds, with the evidence that state demands.
STOP = "STOP"
#: The inputs were present and said they could not be observed. NEVER a pass.
COULD_NOT_EVALUATE = "COULD_NOT_EVALUATE"

OUTCOMES = (CONTINUE, REFUSE, STOP, COULD_NOT_EVALUATE)


# =============================================================================
# Directives — the closed set of things a guard may tell the controller to do
# =============================================================================

DIRECTIVE_PERSIST_AND_DRAIN = "PERSIST_AND_DRAIN"
DIRECTIVE_ROOT_CAUSE_ANALYSIS = "ROOT_CAUSE_ANALYSIS"
DIRECTIVE_LOCAL_PLANNING_ONLY = "LOCAL_PLANNING_ONLY"
DIRECTIVE_REPAIR_FORBIDDEN = "REPAIR_FORBIDDEN"
DIRECTIVE_RECLAIM_EXPIRABLE_FIRST = "RECLAIM_EXPIRABLE_FIRST"
DIRECTIVE_RELEASE_BLOCKED = "RELEASE_BLOCKED"
DIRECTIVE_ESCALATE_TO_OPERATOR = "ESCALATE_TO_OPERATOR"
DIRECTIVE_REANCHOR = "REANCHOR"
DIRECTIVE_REQUEST_DENIED = "REQUEST_DENIED"

DIRECTIVES = frozenset({
    DIRECTIVE_PERSIST_AND_DRAIN, DIRECTIVE_ROOT_CAUSE_ANALYSIS,
    DIRECTIVE_LOCAL_PLANNING_ONLY, DIRECTIVE_REPAIR_FORBIDDEN,
    DIRECTIVE_RECLAIM_EXPIRABLE_FIRST, DIRECTIVE_RELEASE_BLOCKED,
    DIRECTIVE_ESCALATE_TO_OPERATOR, DIRECTIVE_REANCHOR, DIRECTIVE_REQUEST_DENIED,
})

#: §8.10 `RESOURCE_UNAVAILABLE` — *"persist and drain, never busy-wait"*. The
#: prohibition is enforced by ABSENCE: there is no directive a guard could emit
#: that means "spin until it frees up", and `audit_directive_vocabulary()` proves
#: none was added later. A rule expressible only in prose is a rule that gets
#: re-litigated by whoever adds the next directive.
FORBIDDEN_DIRECTIVE_TOKENS = ("WAIT", "POLL", "SLEEP", "RETRY", "SPIN", "BUSY")


# =============================================================================
# Guard identity, the stop table, and precedence
# =============================================================================

GUARD_INTEGRITY = "integrity"
GUARD_ANCHOR = "anchor_identity"
GUARD_HOST_UPTIME = "host_uptime"
GUARD_RESOURCE = "resource_claim"
GUARD_STORAGE = "storage_headroom"
GUARD_BUDGET = "budget"
GUARD_COVERAGE = "evaluator_coverage"
GUARD_OPERATOR_INPUT = "operator_input"
GUARD_PLANNER_HEALTH = "planner_health"
GUARD_EXHAUSTED = "surface_closure"
GUARD_PLATEAU = "plateau"
GUARD_COMMAND_RETRY = "command_retry"
GUARD_REPAIR_CAP = "repair_cap"
GUARD_SPEND_BREAKER = "spend_breaker"
GUARD_STOP_REQUEST = "stop_request_disposal"

#: How a `PLATEAU_STOP` was reached. ONE §8.10 stop state, TWO kinds of evidence
#: for it, and they are not interchangeable:
#:
#:  * `measured_improvement_below_floor` — the window opened on a magnitude, the
#:    best round in it is a magnitude, and the SUBTRACTION came out at or below
#:    the campaign's derived floor.
#:  * `no_detectable_effect_in_any_round` — every round in the window measured
#:    its protected cells and NONE produced an orderable effect. There is no
#:    subtraction here and the detail carries no `improvement`, no
#:    `opening_readiness` and no `best_readiness`: substituting `0.0` for the
#:    rounds that had no magnitude would report "readiness improved by 0.0" for a
#:    quantity nobody measured — a plateau of zeros — and it would look exactly
#:    like the first basis in the journal.
#:
#: They are separate words because the second is only admissible when the rounds
#: could have SEEN the campaign's own target, and an operator auditing a stop has
#: to be able to ask which question was answered.
PLATEAU_BASIS_MEASURED_IMPROVEMENT = "measured_improvement_below_floor"
PLATEAU_BASIS_NO_DETECTABLE_EFFECT = "no_detectable_effect_in_any_round"
PLATEAU_BASES = (PLATEAU_BASIS_MEASURED_IMPROVEMENT, PLATEAU_BASIS_NO_DETECTABLE_EFFECT)

GUARD_IDS = (
    GUARD_INTEGRITY, GUARD_ANCHOR, GUARD_HOST_UPTIME, GUARD_RESOURCE,
    GUARD_STORAGE, GUARD_BUDGET, GUARD_COVERAGE, GUARD_OPERATOR_INPUT,
    GUARD_PLANNER_HEALTH, GUARD_EXHAUSTED, GUARD_PLATEAU, GUARD_COMMAND_RETRY,
    GUARD_REPAIR_CAP, GUARD_SPEND_BREAKER, GUARD_STOP_REQUEST,
)

#: Which guard is allowed to emit which §8.10 stop. One condition, one spelling.
GUARD_BY_STOP: Mapping[str, str] = {
    sm.INTEGRITY_STOP: GUARD_INTEGRITY,
    sm.ANCHOR_MOVED: GUARD_ANCHOR,
    sm.HOST_REBOOT_REQUIRED: GUARD_HOST_UPTIME,
    sm.RESOURCE_UNAVAILABLE: GUARD_RESOURCE,
    sm.DISK_PRESSURE: GUARD_STORAGE,
    sm.BUDGET_STOP: GUARD_BUDGET,
    sm.EVALUATOR_COVERAGE_GAP: GUARD_COVERAGE,
    sm.OPERATOR_INPUT_REQUIRED: GUARD_OPERATOR_INPUT,
    sm.PLANNER_DEGRADED: GUARD_PLANNER_HEALTH,
    sm.EXHAUSTED_SURFACE: GUARD_EXHAUSTED,
    sm.PLATEAU_STOP: GUARD_PLATEAU,
}

#: Stops this module deliberately does NOT decide, each with the plane that does.
#: Re-deriving either here would create a second source of truth for a fact that
#: already has one.
NON_GUARD_STOPS: Mapping[str, str] = {
    # The latch file IS the evidence, and `begin_iteration()` re-reads it from
    # disk under the journal write lock every iteration (invariant 19). A guard
    # that also decided it would be a cached second opinion — the exact shape
    # that made AutoPilot's pause a no-op.
    sm.OPERATOR_STOP_REQUESTED: "state_machine.begin_iteration (invariant 19)",
    # AK6 assembles the package; P-AK-SEARCH-1 denial 7 forbids this plane any
    # release activity at all.
    sm.RELEASE_PACKAGE_READY: "AK6 release packager (P-AK-SEARCH-1 denial 7)",
}

#: Order of adjudication when several conditions hold at once, highest first.
#: The principle, stated once so the next row has somewhere to go:
#:   (1) whatever INVALIDATES the evidence other guards reason from comes first —
#:       tamper invalidates the record, a moved anchor removes the denominator of
#:       every ratio, an over-uptime host invalidates every subsequent number;
#:   (2) then what makes a measurement impossible to START — no claim, no disk;
#:   (3) then what the campaign has SPENT;
#:   (4) then conditions requiring a human;
#:   (5) then the searcher's health;
#:   (6) and CLOSURE LAST, always, because a claim that the search is finished is
#:       only meaningful over evidence that nothing above has invalidated.
STOP_PRECEDENCE = (
    sm.INTEGRITY_STOP,
    sm.ANCHOR_MOVED,
    sm.HOST_REBOOT_REQUIRED,
    sm.RESOURCE_UNAVAILABLE,
    sm.DISK_PRESSURE,
    sm.BUDGET_STOP,
    sm.EVALUATOR_COVERAGE_GAP,
    sm.OPERATOR_INPUT_REQUIRED,
    sm.PLANNER_DEGRADED,
    sm.EXHAUSTED_SURFACE,
    sm.PLATEAU_STOP,
)

#: §18 item 7: *"Every operator escalation — `OPERATOR_INPUT_REQUIRED`,
#: `EVALUATOR_COVERAGE_GAP`, a reboot request, a phase-trade exception, a release
#: package — is rendered as Context / Options / Recommendation / Default."* A
#: STOP in this set without a `DecisionPackage` cannot be constructed.
ESCALATING_STOPS = frozenset({
    sm.OPERATOR_INPUT_REQUIRED, sm.EVALUATOR_COVERAGE_GAP, sm.HOST_REBOOT_REQUIRED,
})


def _assert_vocabulary_total() -> None:
    """Fail at IMPORT if the stop vocabulary and this module have drifted.

    A new §8.10 stop that no guard decides would be reachable only by narration,
    and a guard for a state §8.10 does not declare gives one condition two
    spellings. Both are silent until the day they matter, so the check runs at
    import rather than in a test somebody can forget to run.
    """
    declared = set(sm.STOP_STATES)
    covered = set(STOP_PRECEDENCE) | set(NON_GUARD_STOPS)
    if covered != declared:
        raise GuardVocabularyError(
            "guards.py and state_machine.STOP_STATES disagree: "
            f"undecided stops {sorted(declared - covered)}, "
            f"guards for undeclared stops {sorted(covered - declared)}"
        )
    if set(GUARD_BY_STOP) != set(STOP_PRECEDENCE):
        raise GuardVocabularyError(
            "GUARD_BY_STOP and STOP_PRECEDENCE must cover the same stops; "
            f"{sorted(set(GUARD_BY_STOP) ^ set(STOP_PRECEDENCE))} is in one only"
        )
    if len(STOP_PRECEDENCE) != len(set(STOP_PRECEDENCE)):
        raise GuardVocabularyError("STOP_PRECEDENCE holds a duplicate")
    for directive in DIRECTIVES:
        for token in FORBIDDEN_DIRECTIVE_TOKENS:
            if token in directive:
                raise GuardVocabularyError(
                    f"directive {directive!r} contains {token!r}; §8.10 says persist "
                    "and drain, NEVER busy-wait, and a directive that can express "
                    "waiting is a directive somebody will emit"
                )


_assert_vocabulary_total()


# =============================================================================
# Constitutional constants — supplied by an instrument, never by a campaign
# =============================================================================

#: §10.7 / `bench-cpu.md:17-19`: uptime ≥ 1 week requires a reboot before any
#: further search measurement, and reboots are operator authority. A campaign may
#: declare a STRICTER ceiling; a looser one is discarded, not applied — the same
#: shape as P-AK-SEARCH-1's *"a calibration that would license fewer blocks than
#: the owning protocol already requires is discarded"*.
HOST_UPTIME_CEILING_SECONDS = 7 * 24 * 60 * 60

#: `OPERATING_CONSTRAINTS.md:44-46`, mirrored by §7.1 `stop_policy.max_command_retries`
#: whose schema maximum is already 3. After the third retry the answer is
#: root-cause analysis, not a fourth attempt.
MAX_COMMAND_RETRIES = 3

#: §7.1 `budgets`. The set is CLOSED and a ledger must cover all of it: a budget
#: dimension nobody declared is an unbounded budget wearing a different name, and
#: P-AK-SEARCH-1 precondition 8 refuses to start a campaign that declares one as
#: zero or unbounded.
BUDGET_DIMENSIONS = (
    "max_wall_hours", "max_gpu_hours", "max_cpu_region_hours",
    "max_candidates", "max_controller_tokens", "max_storage_gb",
)

#: §8.10's reserved words, as WORDS — the two the design names explicitly.
#:
#: They are now a SUBSET of `state_machine.RESERVED_CLOSURE_PHRASES`, and that is
#: the fix: this list existed because the machine's list did not contain the bare
#: word, so "the surface is exhausted" passed the DISPOSER while failing here.
#: A stop built without a guard (`stop()` and `dispose_stop_request()` are both
#: public) went straight through. The vocabulary and the matching now live in the
#: machine and this module compiles nothing of its own; the constant stays because
#: it names what §8.10 says out loud, and `audit_stop_coverage_totality()` checks
#: the subset relation still holds.
RESERVED_CLOSURE_WORDS = ("exhausted", "all paths")

ACCEPT_CONTROL_PROMOTED = "PROMOTED"
ACCEPT_CONTROL_FAILED_TO_PROMOTE = "FAILED_TO_PROMOTE"
#: Spelled by the evaluator, not respelled here: two spellings of one marker is
#: how a guard ends up testing the one that never fires.
ACCEPT_CONTROL_UNAVAILABLE = evaluator_api.HISTORICAL_REPLAY_UNAVAILABLE

ACCEPT_CONTROL_STATUSES = (
    ACCEPT_CONTROL_PROMOTED, ACCEPT_CONTROL_FAILED_TO_PROMOTE,
    ACCEPT_CONTROL_UNAVAILABLE,
)


# =============================================================================
# Small validators — a missing input RAISES
# =============================================================================

def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GuardInputError(f"{what}: required, a non-empty string")
    return value


def _opt_text(value: Any, what: str) -> Optional[str]:
    if value is None:
        return None
    return _text(value, what)


def _nonneg_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuardInputError(f"{what}: required, an int (got {type(value).__name__})")
    if value < 0:
        raise GuardInputError(f"{what}: must be >= 0, got {value}")
    return value


def _positive_int(value: Any, what: str) -> int:
    value = _nonneg_int(value, what)
    if value == 0:
        raise GuardInputError(
            f"{what}: must be strictly positive; zero is an unbounded budget "
            "spelled as a number (P-AK-SEARCH-1 precondition 8)"
        )
    return value


def _finite_number(value: Any, what: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardInputError(f"{what}: required, a number (got {type(value).__name__})")
    if not math.isfinite(value):
        raise GuardInputError(
            f"{what}: must be finite, got {value!r}; every NaN comparison is False, "
            "so a NaN here disables the bound it is supposed to impose"
        )
    if minimum is not None and value < minimum:
        raise GuardInputError(f"{what}: must be >= {minimum}, got {value!r}")
    return float(value)


def _timestamp(value: Any, what: str) -> datetime:
    """Parse a tz-aware ISO-8601 stamp. Naive is refused, per `schemas._need_timestamp`."""
    text = _text(value, what)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise GuardInputError(f"{what}: {text!r} is not an ISO-8601 timestamp ({exc})") from exc
    if parsed.tzinfo is None:
        raise GuardInputError(
            f"{what}: {text!r} has no timezone offset; a naive stamp on a shared "
            "host is ambiguous across sessions"
        )
    return parsed


def _str_tuple(value: Any, what: str, *, non_empty: bool = True) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GuardInputError(f"{what}: required, a list of strings")
    items = tuple(_text(item, f"{what}[{i}]") for i, item in enumerate(value))
    if non_empty and not items:
        raise GuardInputError(f"{what}: required, a NON-EMPTY list of strings")
    return items


def _detached_detail(value: Any) -> Any:
    """Deep copy of an already-canonicalizable `detail`, detached from the caller.

    `GuardDecision` validates a STOP's evidence at construction. If the mapping it
    validated is still the caller's own dict, that validation expires the moment
    the caller touches it: the object then carries evidence
    `state_machine.check_stop_evidence` would refuse, while claiming to be a
    decision that passed. Only `None`/`bool`/`int`/`float`/`str`/`Mapping`/`list`
    can reach here, because `schemas.canonical_json` has already refused anything
    else, so this recursion is total.
    """
    if isinstance(value, Mapping):
        return {key: _detached_detail(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detached_detail(item) for item in value]
    return value


def _typed_tuple(value: Any, what: str, klass: type, *, non_empty: bool = True) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GuardInputError(f"{what}: required, a list of {klass.__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, klass):
            raise GuardInputError(
                f"{what}[{index}]: expected {klass.__name__}, got {type(item).__name__}"
            )
    items = tuple(value)
    if non_empty and not items:
        raise GuardInputError(f"{what}: required, a NON-EMPTY list of {klass.__name__}")
    return items


# =============================================================================
# The four-part decision package (§18 item 7)
# =============================================================================

@dataclass(frozen=True)
class DecisionOption:
    """One concrete option in an operator decision package.

    `tradeoffs` is mandatory and non-empty: an option list without tradeoffs is a
    menu, and the project's rule is options **plus tradeoffs plus a
    recommendation** (`feedback_present_decisions_as_options_with_tradeoffs`).
    `reversible` exists so the package's DEFAULT can be constrained to a branch
    that can be undone — the default is what happens when nobody answers, and an
    irreversible default is a decision taken by silence.
    """

    option_id: str
    summary: str
    tradeoffs: tuple
    consequence_if_chosen: str
    reversible: bool

    def __post_init__(self) -> None:
        _text(self.option_id, "option_id")
        _text(self.summary, "summary")
        _text(self.consequence_if_chosen, "consequence_if_chosen")
        if not isinstance(self.reversible, bool):
            raise GuardInputError("reversible: required, a bool")
        object.__setattr__(self, "tradeoffs", _str_tuple(self.tradeoffs, "tradeoffs"))

    def to_dict(self) -> dict:
        return {
            "option_id": self.option_id,
            "summary": self.summary,
            "tradeoffs": list(self.tradeoffs),
            "consequence_if_chosen": self.consequence_if_chosen,
            "reversible": self.reversible,
        }


@dataclass(frozen=True)
class DecisionPackage:
    """Context / Options / Recommendation / Default — §18 item 7, in that shape.

    *"An open-ended question is not an escalation."* Two to four concrete
    options, a recommendation that names one of them, and a default that names a
    REVERSIBLE one, because the default is the branch silence selects.
    """

    context: str
    options: tuple
    recommendation: str
    default: str
    default_rationale: str
    owner: str
    deadline: str

    def __post_init__(self) -> None:
        _text(self.context, "context")
        _text(self.default_rationale, "default_rationale")
        _text(self.owner, "owner")
        _timestamp(self.deadline, "deadline")
        options = _typed_tuple(self.options, "options", DecisionOption)
        if not 2 <= len(options) <= 4:
            raise GuardInputError(
                f"options: an operator decision package carries 2-4 concrete options, "
                f"got {len(options)}; one option is not a decision"
            )
        ids = [option.option_id for option in options]
        if len(set(ids)) != len(ids):
            raise GuardInputError(f"options: duplicate option_id in {ids}")
        object.__setattr__(self, "options", options)
        for name in ("recommendation", "default"):
            value = _text(getattr(self, name), name)
            if value not in ids:
                raise GuardInputError(
                    f"{name}: {value!r} names no declared option; declared are {ids}"
                )
        chosen_default = next(o for o in options if o.option_id == self.default)
        if not chosen_default.reversible:
            raise GuardInputError(
                f"default: option {self.default!r} is declared irreversible; the "
                "default is what happens when the operator does not answer, so an "
                "irreversible default is a decision taken by silence"
            )

    def to_detail(self) -> dict:
        """The mapping `state_machine.check_stop_evidence` reads for a §8.10 stop."""
        return {
            "context": self.context,
            "options": [option.to_dict() for option in self.options],
            "recommendation": self.recommendation,
            "default": self.default,
            "default_rationale": self.default_rationale,
            "owner": self.owner,
            "deadline": self.deadline,
        }

    def render(self) -> str:
        """Deterministic plain-text rendering, for the bus artifact a session lands.

        §18 item 6: *"The running loop does not write handoffs, index rows, or
        intake entries."* This returns a STRING; nothing here writes it anywhere.
        """
        lines = [
            "CONTEXT",
            f"  {self.context}",
            "",
            "OPTIONS",
        ]
        for option in self.options:
            marks = []
            if option.option_id == self.recommendation:
                marks.append("RECOMMENDED")
            if option.option_id == self.default:
                marks.append("DEFAULT")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            lines.append(f"  {option.option_id}: {option.summary}{suffix}")
            for tradeoff in option.tradeoffs:
                lines.append(f"      - {tradeoff}")
            lines.append(f"      => {option.consequence_if_chosen}")
            lines.append(
                f"      reversible: {'yes' if option.reversible else 'NO'}"
            )
        lines.extend([
            "",
            "RECOMMENDATION",
            f"  {self.recommendation}",
            "",
            "DEFAULT",
            f"  {self.default} (if no answer by {self.deadline}) — {self.default_rationale}",
            f"  owner: {self.owner}",
        ])
        return "\n".join(lines)


# =============================================================================
# The decision a guard returns
# =============================================================================

@dataclass(frozen=True)
class GuardDecision:
    """One guard's verdict, plus the reason and the receipts it rests on.

    A `STOP` is validated against `state_machine.check_stop_evidence` HERE, at
    construction. That is the load-bearing property of this type: a stop decision
    the machine would refuse cannot be built, so it can never be reported to an
    operator, counted in a disposition, or rendered into a digest as though the
    campaign had stopped for that reason.

    `evidence` is mandatory for a STOP and holds journal event ids or receipt
    locators. A stop with no receipt is narration, and §4 invariant 14 is
    precisely the rule that narration decides nothing.
    """

    guard_id: str
    outcome: str
    reason: str
    stop_state: Optional[str] = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    directives: tuple = ()
    decision_package: Optional[DecisionPackage] = None
    evidence: tuple = ()

    def __post_init__(self) -> None:
        _text(self.guard_id, "guard_id")
        if self.guard_id not in GUARD_IDS:
            raise GuardInputError(
                f"guard_id: {self.guard_id!r} is not a declared guard {list(GUARD_IDS)}"
            )
        if self.outcome not in OUTCOMES:
            raise GuardInputError(f"outcome: {self.outcome!r} not in {list(OUTCOMES)}")
        _text(self.reason, "reason")
        if not isinstance(self.detail, Mapping):
            raise GuardInputError("detail: must be a mapping")
        # Canonicalizability is checked here so a detail that cannot be journaled
        # fails while the decision is still being built, not while it is being
        # recorded. `schemas.canonical_json` rejects tuples on purpose.
        schemas.canonical_json(dict(self.detail))
        # Snapshot it. A validated STOP whose `detail` is still the caller's live
        # dict is a decision whose validation expires when the caller next writes
        # to that dict — the object would keep reporting as machine-accepted while
        # carrying evidence the machine now refuses.
        object.__setattr__(self, "detail", _detached_detail(dict(self.detail)))
        object.__setattr__(
            self, "directives", _str_tuple(self.directives, "directives", non_empty=False)
        )
        for directive in self.directives:
            if directive not in DIRECTIVES:
                raise GuardInputError(
                    f"directives: {directive!r} is not a declared directive "
                    f"{sorted(DIRECTIVES)}"
                )
        object.__setattr__(
            self, "evidence", _str_tuple(self.evidence, "evidence", non_empty=False)
        )
        if self.decision_package is not None and not isinstance(
            self.decision_package, DecisionPackage
        ):
            raise GuardInputError("decision_package: must be a DecisionPackage or None")

        if self.outcome == STOP:
            if self.stop_state not in GUARD_BY_STOP:
                raise GuardInputError(
                    f"stop_state: {self.stop_state!r} is not a guard-decidable stop "
                    f"{sorted(GUARD_BY_STOP)}"
                )
            owner = GUARD_BY_STOP[self.stop_state]
            if owner != self.guard_id:
                raise GuardInputError(
                    f"{self.stop_state} is decided by guard {owner!r}, not "
                    f"{self.guard_id!r}; one condition gets one spelling"
                )
            if not self.evidence:
                raise GuardEvidenceError(
                    f"{self.stop_state}: a stop with no receipt is narration; "
                    "`evidence` must name at least one journaled record"
                )
            check = sm.check_stop_evidence(self.stop_state, self.reason, self.detail)
            if check.outcome != schemas.PASS:
                raise GuardEvidenceError(
                    f"{self.stop_state} refused at construction ({check.outcome}): "
                    + "; ".join(check.reasons)
                )
            if self.stop_state in ESCALATING_STOPS and self.decision_package is None:
                raise GuardEvidenceError(
                    f"{self.stop_state} is an operator escalation and §18 item 7 "
                    "requires it to render as Context / Options / Recommendation / "
                    "Default; no DecisionPackage was attached"
                )
        elif self.stop_state is not None:
            raise GuardInputError(
                f"stop_state is only meaningful on a STOP; outcome is {self.outcome}"
            )

        if self.outcome == REFUSE and not self.directives:
            raise GuardInputError(
                "REFUSE must say what is refused: at least one directive is required"
            )
        if DIRECTIVE_ESCALATE_TO_OPERATOR in self.directives and self.decision_package is None:
            raise GuardEvidenceError(
                "a decision that escalates to the operator must carry the four-part "
                "package (§18 item 7); an open-ended question is not an escalation"
            )

    @property
    def clears(self) -> bool:
        """True ONLY for CONTINUE. `COULD_NOT_EVALUATE` is deliberately falsy."""
        return self.outcome == CONTINUE

    def to_check(self) -> schemas.Check:
        """The three-outcome view, for callers that compose with evaluator checks."""
        if self.outcome == CONTINUE:
            return schemas.Check(schemas.PASS)
        if self.outcome == COULD_NOT_EVALUATE:
            return schemas.Check(schemas.COULD_NOT_CHECK, (self.reason,))
        return schemas.Check(schemas.FAIL, (self.reason,))

    def to_dict(self) -> dict:
        return {
            "guard_id": self.guard_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "stop_state": self.stop_state,
            # Deep, not `dict(...)`: a shallow copy still hands out the nested
            # mappings, and `to_dict()["planner_health"]["degraded_ruled_out"]`
            # would be a writable handle on validated stop evidence.
            "detail": _detached_detail(dict(self.detail)),
            "directives": list(self.directives),
            "decision_package": (
                None if self.decision_package is None
                else self.decision_package.to_detail()
            ),
            "evidence": list(self.evidence),
        }


# =============================================================================
# Closure enumeration (§8.10) — shared by EXHAUSTED_SURFACE and PLATEAU_STOP
# =============================================================================

@dataclass(frozen=True)
class ClosedSubScope:
    """*"closed for sub-scope X (gates A, B, C met)"* — with the receipt.

    §19.3's receipt rule generalises: a suppression that closes a research family
    needs a receipt bound to the current production commit. Closing a sub-scope
    is that suppression, so the receipt is a field, not a convention.
    """

    sub_scope: str
    gates_met: tuple
    receipt: str

    def __post_init__(self) -> None:
        _text(self.sub_scope, "sub_scope")
        _text(self.receipt, "receipt")
        object.__setattr__(self, "gates_met", _str_tuple(self.gates_met, "gates_met"))

    def to_dict(self) -> dict:
        return {
            "sub_scope": self.sub_scope,
            "gates_met": list(self.gates_met),
            "receipt": self.receipt,
        }


@dataclass(frozen=True)
class DeferredSubScope:
    """*"sub-scope Y deferred (gates D, E un-run)"* — and WHY it was deferred."""

    sub_scope: str
    gates_unrun: tuple
    reason: str

    def __post_init__(self) -> None:
        _text(self.sub_scope, "sub_scope")
        _text(self.reason, "reason")
        object.__setattr__(self, "gates_unrun", _str_tuple(self.gates_unrun, "gates_unrun"))

    def to_dict(self) -> dict:
        return {
            "sub_scope": self.sub_scope,
            "gates_unrun": list(self.gates_unrun),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClosureLedger:
    """What was closed and what was not — the enumeration §8.10 demands.

    `deferred` MAY be empty and MUST be present: an empty list is an answer, an
    absent one is closure inflation. `closed` may not be empty, because a closure
    claim over nothing closed is not a claim about a surface at all.
    """

    closed: tuple
    deferred: tuple
    hierarchy_layers_considered: tuple

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "closed", _typed_tuple(self.closed, "closed", ClosedSubScope)
        )
        object.__setattr__(
            self, "deferred",
            _typed_tuple(self.deferred, "deferred", DeferredSubScope, non_empty=False),
        )
        object.__setattr__(
            self, "hierarchy_layers_considered",
            _str_tuple(self.hierarchy_layers_considered, "hierarchy_layers_considered"),
        )
        names = [entry.sub_scope for entry in self.closed]
        names += [entry.sub_scope for entry in self.deferred]
        if len(set(names)) != len(names):
            raise GuardInputError(
                "a sub-scope appears twice across closed/deferred; a scope cannot be "
                f"both closed and deferred: {sorted(names)}"
            )

    def to_detail(self) -> dict:
        return {
            "closed": [entry.to_dict() for entry in self.closed],
            "deferred": [entry.to_dict() for entry in self.deferred],
            "hierarchy_layers_considered": list(self.hierarchy_layers_considered),
        }

    @property
    def receipts(self) -> tuple:
        return tuple(entry.receipt for entry in self.closed)


def check_closure_language(reason: str, ledger: "ClosureLedger") -> schemas.Check:
    """Reject the reserved closure words, as WORDS, everywhere they can hide.

    `state_machine.check_closure_enumeration` rejects the multi-word phrases; not
    one of them matches *"the surface is exhausted"*, which is the sentence §8.10
    is actually about. So this scans the reason AND every enumerated sub-scope and
    gate string for the bare tokens. The scan is deliberately broad: the design
    records that this habit *"survived even explicit awareness of the rule"*, and
    a checker that only inspects the field the author was thinking about is a
    checker the habit walks around.

    Returns COULD_NOT_CHECK when there is no text to read — an unread reason is
    not a clean one.
    """
    if not isinstance(ledger, ClosureLedger):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"ledger is a {type(ledger).__name__}, not a ClosureLedger; the "
            "enumeration cannot be read",
        ))
    if not isinstance(reason, str) or not reason.strip():
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "reason is empty; there is no closure statement to inspect",
        ))

    surfaces: list = [("reason", reason)]
    for index, entry in enumerate(ledger.closed):
        surfaces.append((f"closed[{index}].sub_scope", entry.sub_scope))
        for gate_index, gate in enumerate(entry.gates_met):
            surfaces.append((f"closed[{index}].gates_met[{gate_index}]", gate))
    for index, entry in enumerate(ledger.deferred):
        surfaces.append((f"deferred[{index}].sub_scope", entry.sub_scope))
        for gate_index, gate in enumerate(entry.gates_unrun):
            surfaces.append((f"deferred[{index}].gates_unrun[{gate_index}]", gate))
        surfaces.append((f"deferred[{index}].reason", entry.reason))
    # The layer list is part of the enumeration and is free text like the rest of
    # it. A scan that stops at the fields the author was thinking about is the
    # scan this habit walks around, and "all paths" reads the same in
    # `hierarchy_layers_considered` as it does in `reason`.
    for index, layer in enumerate(ledger.hierarchy_layers_considered):
        surfaces.append((f"hierarchy_layers_considered[{index}]", layer))

    findings: list = []
    for where, text in surfaces:
        for phrase in sm.reserved_closure_findings(text):
            findings.append(
                f"{where}: contains the reserved closure phrase {phrase!r}; §8.10 "
                "requires the enumeration instead — 'closed for sub-scope X "
                "(gates A, B, C met); sub-scope Y deferred (gates D, E un-run)'"
            )
    if findings:
        # Deduplicated and ordered so the same defect reads the same way twice.
        return schemas.Check(schemas.FAIL, tuple(dict.fromkeys(findings)))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Observations — journal-derived state, each one validated on construction
# =============================================================================

@dataclass(frozen=True)
class IntegritySignal:
    """One tamper / reward-hacking observation, with the receipt that found it."""

    signal: str
    at: str
    receipt: str
    candidate_id: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.signal, "signal")
        _timestamp(self.at, "at")
        _text(self.receipt, "receipt")
        _opt_text(self.candidate_id, "candidate_id")

    def to_dict(self) -> dict:
        return {
            "signal": self.signal, "at": self.at, "receipt": self.receipt,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class IntegrityLedger:
    """Consecutive integrity failures against the campaign's declared tolerance.

    §7.1 declares `max_consecutive_integrity_failures`, and the template's value
    is 0 — invariant 6 makes integrity lexicographically prior, so a tolerance of
    zero means the first signal stops the campaign.
    """

    signals: tuple
    consecutive_failures: int
    max_consecutive_integrity_failures: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "signals",
            _typed_tuple(self.signals, "signals", IntegritySignal, non_empty=False),
        )
        _nonneg_int(self.consecutive_failures, "consecutive_failures")
        _nonneg_int(
            self.max_consecutive_integrity_failures,
            "max_consecutive_integrity_failures",
        )
        if self.consecutive_failures > len(self.signals):
            raise GuardInputError(
                f"consecutive_failures={self.consecutive_failures} exceeds the "
                f"{len(self.signals)} signal(s) supplied; a count without its "
                "receipts is the narration invariant 14 forbids"
            )


@dataclass(frozen=True)
class HostHealth:
    """Host uptime against the §10.7 ceiling.

    `observable` is explicit rather than implied by a sentinel: a host readout
    that failed is a different fact from a host that has been up for zero
    seconds, and the second reads as healthy.
    """

    uptime_seconds: int
    observed_at: str
    receipt: str
    ceiling_seconds: int = HOST_UPTIME_CEILING_SECONDS
    observable: bool = True

    def __post_init__(self) -> None:
        _nonneg_int(self.uptime_seconds, "uptime_seconds")
        _timestamp(self.observed_at, "observed_at")
        _text(self.receipt, "receipt")
        _positive_int(self.ceiling_seconds, "ceiling_seconds")
        if not isinstance(self.observable, bool):
            raise GuardInputError("observable: required, a bool")
        if self.ceiling_seconds > HOST_UPTIME_CEILING_SECONDS:
            raise GuardInputError(
                f"ceiling_seconds={self.ceiling_seconds} is looser than the §10.7 "
                f"ceiling of {HOST_UPTIME_CEILING_SECONDS}; a campaign may declare a "
                "STRICTER ceiling, and a looser one is discarded rather than applied"
            )


@dataclass(frozen=True)
class ResourceClaimObservation:
    """Whether the claim the window needs is HELD — acquired, never inferred.

    §4 invariant 9 and P-AK-SEARCH-1 precondition 1: *"Both are ACQUIRED, never
    inferred: observing that a device or region looks free is TOCTOU, not
    exclusion."* So there is no "looked free" branch here; `acquired` means a
    receipt exists, and the receipt is required when it is True.
    """

    resource: str
    claim_kind: str
    acquired: bool
    observed_at: str
    receipt: Optional[str] = None
    held_by: Optional[str] = None
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.resource, "resource")
        _text(self.claim_kind, "claim_kind")
        _timestamp(self.observed_at, "observed_at")
        if not isinstance(self.acquired, bool):
            raise GuardInputError("acquired: required, a bool")
        _opt_text(self.held_by, "held_by")
        _opt_text(self.unavailable_reason, "unavailable_reason")
        if self.acquired:
            _text(self.receipt, "receipt")
        else:
            _opt_text(self.receipt, "receipt")
            if self.unavailable_reason is None:
                raise GuardInputError(
                    "unavailable_reason: required when the claim was not acquired; "
                    "'it was not available' with no reason cannot be escalated or "
                    "resumed from"
                )


@dataclass(frozen=True)
class StorageObservation:
    """A storage reading plus the backlog that is ALREADY eligible for expiry.

    P-AK-SEARCH-1 precondition 7: *"Reclamation outside the enumerated expirable
    classes of that clause is operator authority; when the already-eligible expiry
    backlog does not clear the floor, the campaign stops."* Both halves are
    needed to decide, so both are fields.
    """

    path: str
    state: storage.StorageState
    expirable_backlog_bytes: int
    receipt: str

    def __post_init__(self) -> None:
        _text(self.path, "path")
        if not isinstance(self.state, storage.StorageState):
            raise GuardInputError(
                "state: must be a storage.StorageState produced by "
                "storage.disk_pressure(); this plane does not read the filesystem"
            )
        _nonneg_int(self.expirable_backlog_bytes, "expirable_backlog_bytes")
        _text(self.receipt, "receipt")


@dataclass(frozen=True)
class BudgetDimension:
    """One declared budget and what has been attributed against it.

    §12 zero-yield row: *"`realized_cost` attributed per proposal"* — cost was a
    stop threshold and never an attributed ledger, so cost-per-banked-win was
    unauditable. `consumed` here is the attributed figure, and `receipt` names
    where the attribution lives.
    """

    name: str
    limit: float
    consumed: float
    receipt: str

    def __post_init__(self) -> None:
        _text(self.name, "name")
        if self.name not in BUDGET_DIMENSIONS:
            raise GuardInputError(
                f"name: {self.name!r} is not a declared budget dimension "
                f"{list(BUDGET_DIMENSIONS)}"
            )
        limit = _finite_number(self.limit, "limit", minimum=0.0)
        if limit <= 0:
            raise GuardInputError(
                f"limit: {self.name} must be finite and strictly positive "
                "(P-AK-SEARCH-1 precondition 8: a campaign that declares a budget as "
                "zero or unbounded cannot derive its error budgets and MUST NOT start)"
            )
        object.__setattr__(self, "limit", limit)
        object.__setattr__(
            self, "consumed", _finite_number(self.consumed, "consumed", minimum=0.0)
        )
        _text(self.receipt, "receipt")

    @property
    def exhausted(self) -> bool:
        return self.consumed >= self.limit

    @property
    def fraction(self) -> float:
        return self.consumed / self.limit

    def to_dict(self) -> dict:
        return {
            "budget": self.name, "limit": self.limit, "consumed": self.consumed,
            "fraction": self.fraction, "receipt": self.receipt,
        }


@dataclass(frozen=True)
class BudgetLedger:
    """Every §7.1 budget dimension. Partial coverage is refused.

    A ledger missing a dimension is an unbounded budget for that dimension, and
    the only difference between that and a declared budget of infinity is which
    one you can see.
    """

    dimensions: tuple

    def __post_init__(self) -> None:
        dims = _typed_tuple(self.dimensions, "dimensions", BudgetDimension)
        names = tuple(dimension.name for dimension in dims)
        if len(set(names)) != len(names):
            raise GuardInputError(f"dimensions: duplicate budget name in {list(names)}")
        missing = [name for name in BUDGET_DIMENSIONS if name not in names]
        if missing:
            raise GuardInputError(
                f"dimensions: the ledger must cover every §7.1 budget; {missing} "
                "is absent, which is an unbounded budget under another name"
            )
        # Declared order, so two campaigns with the same state report the same
        # governing dimension.
        ordered = tuple(
            next(d for d in dims if d.name == name) for name in BUDGET_DIMENSIONS
        )
        object.__setattr__(self, "dimensions", ordered)

    def by_name(self, name: str) -> BudgetDimension:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise KeyError(name)

    @property
    def exhausted(self) -> tuple:
        return tuple(d for d in self.dimensions if d.exhausted)


@dataclass(frozen=True)
class SpendBreakerPolicy:
    """The fraction at which metered drafting stops and local planning starts.

    §2.5 row 4: the naive spend breaker *"stopped the loop"*, and the correction
    is that the breaker *"forces local planning rather than halting"*. The
    fraction is a declared campaign policy with a receipt, never a literal chosen
    here — this module supplies no threshold of its own.
    """

    breaker_fraction: float
    policy_receipt: str

    def __post_init__(self) -> None:
        fraction = _finite_number(self.breaker_fraction, "breaker_fraction")
        if not 0 < fraction < 1:
            raise GuardInputError(
                "breaker_fraction: must be strictly between 0 and 1; at 1 the breaker "
                "is the ceiling and forces nothing, at 0 it never lets the loop draft"
            )
        object.__setattr__(self, "breaker_fraction", fraction)
        _text(self.policy_receipt, "policy_receipt")


@dataclass(frozen=True)
class CoverageGap:
    """An evaluator coverage gap — with an OWNER and a DEADLINE, or not at all.

    §8.10: *"It has an owner and a deadline, or it becomes a permanent silent
    block."* Both are required fields, so a gap without them cannot be
    constructed and therefore cannot be sitting quietly in a ledger.
    P-AK-SEARCH-1 *No self-amendment*: the gap is RECORDED, release is blocked
    for the affected lineage, unrelated research continues, and an amendment may
    be DRAFTED — the instrument is never patched from inside the loop.
    """

    gap_id: str
    missing_coverage_class: str
    blocked_lineage: str
    owner: str
    deadline: str
    opened_at: str
    receipt: str
    boundaries_open: int = 0
    freeze_cycles_open: int = 0
    amendment_draft_ref: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("gap_id", "missing_coverage_class", "blocked_lineage", "owner",
                     "receipt"):
            _text(getattr(self, name), name)
        opened = _timestamp(self.opened_at, "opened_at")
        deadline = _timestamp(self.deadline, "deadline")
        if deadline < opened:
            raise GuardInputError(
                f"deadline {self.deadline!r} precedes opened_at {self.opened_at!r}; an "
                "escalation deadline in the past at creation is a block with no clock"
            )
        _nonneg_int(self.boundaries_open, "boundaries_open")
        _nonneg_int(self.freeze_cycles_open, "freeze_cycles_open")
        _opt_text(self.amendment_draft_ref, "amendment_draft_ref")

    def to_dict(self) -> dict:
        return {
            "gap_id": self.gap_id,
            "missing_coverage_class": self.missing_coverage_class,
            "blocked_lineage": self.blocked_lineage,
            "owner": self.owner,
            "deadline": self.deadline,
            "opened_at": self.opened_at,
            "receipt": self.receipt,
            "boundaries_open": self.boundaries_open,
            "freeze_cycles_open": self.freeze_cycles_open,
            "amendment_draft_ref": self.amendment_draft_ref,
        }


@dataclass(frozen=True)
class OperatorQuestion:
    """A question the loop may not answer itself, in decision-package form.

    An `answered` question with no `answered_event_id` is refused: an answer that
    exists only in a controller's belief is the fail-open shape — it clears a
    block using a fact nothing recorded.
    """

    question_id: str
    package: DecisionPackage
    raised_at: str
    answered: bool = False
    answered_event_id: Optional[str] = None
    blocking: bool = True
    receipt: str = ""

    def __post_init__(self) -> None:
        _text(self.question_id, "question_id")
        if not isinstance(self.package, DecisionPackage):
            raise GuardInputError("package: must be a DecisionPackage")
        _timestamp(self.raised_at, "raised_at")
        for name in ("answered", "blocking"):
            if not isinstance(getattr(self, name), bool):
                raise GuardInputError(f"{name}: required, a bool")
        _text(self.receipt, "receipt")
        if self.answered:
            _text(self.answered_event_id, "answered_event_id")
        else:
            _opt_text(self.answered_event_id, "answered_event_id")


@dataclass(frozen=True)
class PlannerHealthPolicy:
    """Thresholds for the four §8.10 degradation families, plus the §8.5.1 caps.

    Every threshold is a DECLARED campaign control with a receipt. This module
    supplies no number: P-AK-SEARCH-1 is explicit that *"no value in this list may
    be supplied as a literal — not by a controller, not by a proposal … and not by
    this protocol"*, and a guard that invented one would be deciding the
    explore/exploit tradeoff by guess (AK-D32).
    """

    window_rounds: int
    max_consecutive_noop_rounds: int
    max_repeated_fingerprints: int
    max_invalid_dispatches: int
    max_contradicted_narratives: int
    max_unavailable_dependency_rounds: int
    max_consecutive_build_failures: int
    max_repair_cap_exceedances: int
    policy_receipt: str

    def __post_init__(self) -> None:
        for name in ("window_rounds", "max_consecutive_noop_rounds",
                     "max_repeated_fingerprints", "max_invalid_dispatches",
                     "max_contradicted_narratives", "max_unavailable_dependency_rounds",
                     "max_repair_cap_exceedances"):
            _positive_int(getattr(self, name), name)
        # §7.1's template value is 0: zero tolerance is a legitimate declaration
        # for build failures, unlike a zero WINDOW which would decide from nothing.
        _nonneg_int(self.max_consecutive_build_failures, "max_consecutive_build_failures")
        _text(self.policy_receipt, "policy_receipt")


@dataclass(frozen=True)
class PlannerHealth:
    """The four §8.10 degradation families, counted from journaled records.

    *"repeating no-ops, dispatching invalid actions, narrating a condition the
    receipts contradict, or looping against an unavailable dependency"* — plus
    §8.5.1's repair cap, whose exceedance the design calls *"a `PLANNER_DEGRADED`
    signal, not another retry"*.

    `receipts` maps a signal name to the journal locator that evidences it. A
    signal that crosses its threshold with no receipt yields COULD_NOT_EVALUATE,
    never a stop: §8.10 requires `signal` AND `receipt`, and a degraded verdict
    with no receipt is the same narration it is meant to detect.
    """

    rounds_observed: int
    consecutive_noop_rounds: int
    proposal_skipped_count: int
    repeated_fingerprint_count: int
    invalid_dispatch_count: int
    contradicted_narrative_count: int
    unavailable_dependency_rounds: int
    consecutive_build_failures: int
    repair_cap_exceedances: int
    banked_count: int
    receipts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("rounds_observed", "consecutive_noop_rounds",
                     "proposal_skipped_count", "repeated_fingerprint_count",
                     "invalid_dispatch_count", "contradicted_narrative_count",
                     "unavailable_dependency_rounds", "consecutive_build_failures",
                     "repair_cap_exceedances", "banked_count"):
            _nonneg_int(getattr(self, name), name)
        if not isinstance(self.receipts, Mapping):
            raise GuardInputError("receipts: must be a mapping of signal -> locator")
        for key, value in self.receipts.items():
            _text(key, "receipts key")
            _text(value, f"receipts[{key}]")
        if self.consecutive_noop_rounds > self.rounds_observed:
            raise GuardInputError(
                f"consecutive_noop_rounds={self.consecutive_noop_rounds} exceeds "
                f"rounds_observed={self.rounds_observed}"
            )
        object.__setattr__(self, "receipts", dict(self.receipts))

    def to_dict(self) -> dict:
        return {
            "rounds_observed": self.rounds_observed,
            "consecutive_noop_rounds": self.consecutive_noop_rounds,
            "proposal_skipped_count": self.proposal_skipped_count,
            "repeated_fingerprint_count": self.repeated_fingerprint_count,
            "invalid_dispatch_count": self.invalid_dispatch_count,
            "contradicted_narrative_count": self.contradicted_narrative_count,
            "unavailable_dependency_rounds": self.unavailable_dependency_rounds,
            "consecutive_build_failures": self.consecutive_build_failures,
            "repair_cap_exceedances": self.repair_cap_exceedances,
            "banked_count": self.banked_count,
            "receipts": dict(self.receipts),
        }

    @property
    def digest(self) -> str:
        """Content hash of the snapshot, so a clean verdict cannot be re-used later."""
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class AcceptSideControlReceipt:
    """Control 5 — the historical-win replay that MUST promote (AK-D27).

    §12: *"The gate can still reject but can no longer promote, and reads as
    'exhausted'."* The other four controls test the gate's ability to REJECT;
    without this one a dead gate is indistinguishable from a closed surface, so
    no closure guard may conclude without it. `cadence` is the evaluator's own
    verdict on whether the control ran within its declared schedule — computed
    there, consumed here, never recomputed.
    """

    status: str
    event_id: str
    observed_at: str
    cadence: schemas.Check
    win_id: Optional[str] = None
    control_id: str = evaluator_api.HISTORICAL_REPLAY_UNAVAILABLE

    def __post_init__(self) -> None:
        if self.status not in ACCEPT_CONTROL_STATUSES:
            raise GuardInputError(
                f"status: {self.status!r} not in {list(ACCEPT_CONTROL_STATUSES)}"
            )
        _text(self.event_id, "event_id")
        _timestamp(self.observed_at, "observed_at")
        if not isinstance(self.cadence, schemas.Check):
            raise GuardInputError("cadence: must be a schemas.Check from the evaluator")
        _opt_text(self.win_id, "win_id")
        _text(self.control_id, "control_id")
        if self.status == ACCEPT_CONTROL_PROMOTED and self.win_id is None:
            raise GuardInputError(
                "win_id: required when the accept-side control PROMOTED; a promotion "
                "with no win named is not a replay of anything"
            )
        # `control_id`'s default is the UNAVAILABLE *status* sentinel, so a receipt
        # built without naming its control silently journals
        # `accept_side_control.control_id == "HISTORICAL_REPLAY_UNAVAILABLE"` inside
        # the evidence of a closure stop — the record of a live gate reading as the
        # record of a gate nobody could observe. A status is not an identity.
        if (self.control_id == ACCEPT_CONTROL_UNAVAILABLE
                and self.status != ACCEPT_CONTROL_UNAVAILABLE):
            raise GuardInputError(
                f"control_id: {ACCEPT_CONTROL_UNAVAILABLE!r} is a STATUS, not the "
                f"identity of a control; a {self.status} receipt must name the "
                "accept-side control that produced it"
            )

    @property
    def promoted(self) -> bool:
        return self.status == ACCEPT_CONTROL_PROMOTED

    def to_dict(self) -> dict:
        return {
            "status": self.status, "event_id": self.event_id,
            "observed_at": self.observed_at, "win_id": self.win_id,
            "control_id": self.control_id, "cadence": self.cadence.outcome,
        }


class ReadinessSeriesEntry:
    """One round's readiness RESULT — of which there are two kinds, and the split
    between them is a TYPE, not a field.

    A round either produced an orderable readiness magnitude
    (`ReadinessObservation`) or it did not (`ParityObservation`), and the plateau
    rule's arithmetic is only defined on the first. The alternative design — one
    class with a number and an `at_parity` boolean beside it — puts the whole
    guarantee in the consumer's hands: `max(o.readiness for o in window)` reads
    the number whether or not anyone checked the flag, and every "flag it and
    hope the consumer checks" design in this package has turned out to be a
    defect. Here `ParityObservation` has no magnitude to read at all, so the
    wrong reading is not discouraged, it is unavailable.
    """

    __slots__ = ()


@dataclass(frozen=True)
class ReadinessObservation(ReadinessSeriesEntry):
    """One reduced readiness figure, from the CONFIRMATION stratum, with its event.

    Two refusals live in the constructor:
      * a `selection`-stratum figure is rejected — P-AK-SEARCH-1: *"The readiness
        signal is computed ONLY from confirmation-stratum evidence"*, because
        selecting the maximum over many candidates biases the selected estimate
        upward;
      * a figure with no `source_event_id` is rejected — §4 invariant 14: the
        controller computes readiness from records and *"the LLM may request,
        never declare"*, so a readiness number with no record behind it is not a
        readiness number.

    A third refusal is structural rather than written: `readiness` is a REQUIRED
    argument, so the release plane's parity mapping — which carries no
    `readiness` key — cannot construct one of these at all.
    """

    round_index: int
    readiness: float
    at: str
    source_event_id: str
    stratum: str = evaluator_api.STRATUM_CONFIRMATION

    def __post_init__(self) -> None:
        _nonneg_int(self.round_index, "round_index")
        object.__setattr__(self, "readiness", _finite_number(self.readiness, "readiness"))
        _timestamp(self.at, "at")
        _text(self.source_event_id, "source_event_id")
        if self.stratum != evaluator_api.STRATUM_CONFIRMATION:
            raise GuardInputError(
                f"stratum: {self.stratum!r} — a readiness series admits only "
                f"{evaluator_api.STRATUM_CONFIRMATION!r} evidence; selection evidence "
                "is structurally unfit to report how ready a candidate is"
            )

    def to_dict(self) -> dict:
        return {
            "round_index": self.round_index, "readiness": self.readiness,
            "at": self.at, "source_event_id": self.source_event_id,
            "stratum": self.stratum,
            # Carried on BOTH kinds of round, because `ParityObservation` publishes
            # `orderable: false` and a reader will branch on it. A discriminator
            # present only on the negative side makes
            # `entry.get("orderable", False)` answer "no round is orderable", which
            # empties the window silently instead of failing — and `guard_plateau`'s
            # detail serialises the whole window, so that reader is downstream of
            # every stop decision this guard records.
            "orderable": True,
        }


@dataclass(frozen=True)
class ParityObservation(ReadinessSeriesEntry):
    """One round in which every protected cell was measured and NONE was orderable.

    This is a completed round with a result, not a missing round: under a
    non-inferiority objective, cells at `no_detectable_difference` are *"a result
    and a decision, not a failed experiment"*. So it enters the series — a
    plateau rule that simply never saw these rounds would be trending a
    subsequence it chose, and would report "no observations" for a campaign that
    ran every one of them.

    What it does NOT carry is a magnitude, and `readiness` raises rather than
    being merely absent so that `getattr(observation, "readiness", 0.0)` cannot
    quietly reintroduce one. `protected_cells`, `cells_at_parity`, `mde` and
    `noise_floor` are carried instead, because "nothing moved" is only meaningful
    against the sensitivity it was judged at.
    """

    round_index: int
    protected_cells: int
    cells_at_parity: int
    mde: float
    noise_floor: float
    sensitivity_bound: float
    at: str
    source_event_id: str
    stratum: str = evaluator_api.STRATUM_CONFIRMATION
    reference_gain: Optional[float] = None

    def __post_init__(self) -> None:
        _nonneg_int(self.round_index, "round_index")
        _nonneg_int(self.protected_cells, "protected_cells")
        _nonneg_int(self.cells_at_parity, "cells_at_parity")
        if self.protected_cells < 1:
            raise GuardInputError(
                "protected_cells: a parity observation with no protected cell behind "
                "it reports a result for a round that measured nothing")
        if self.cells_at_parity > self.protected_cells:
            raise GuardInputError(
                f"cells_at_parity {self.cells_at_parity} exceeds protected_cells "
                f"{self.protected_cells}")
        object.__setattr__(self, "mde", _finite_number(self.mde, "mde"))
        object.__setattr__(
            self, "noise_floor", _finite_number(self.noise_floor, "noise_floor"))
        object.__setattr__(self, "sensitivity_bound", _finite_number(
            self.sensitivity_bound, "sensitivity_bound", minimum=0.0))
        # The producer computes the bound (`ParityFigure.sensitivity_bound`) and
        # this side does not recompute it — a second derivation is a second thing
        # to drift. What it DOES refuse is a bound sharper than the two numbers
        # published beside it, because that is the only direction that hurts: a
        # bound smaller than the cell's own MDE or floor claims the round could
        # see finer than the cell that produced it, and a round that looks
        # sharper than it is turns "we could not have seen it" into "nothing
        # happened". A COARSER bound is admitted: the producer may bind on
        # something this side has never heard of, and being told the run was
        # blinder than these two numbers can only make the guard more reluctant.
        published = max(self.mde, self.noise_floor)
        if self.sensitivity_bound < published:
            raise GuardInputError(
                f"sensitivity_bound {self.sensitivity_bound} is sharper than the "
                f"round's own MDE {self.mde} / floor {self.noise_floor}; a parity "
                "round cannot resolve finer than the cell it came from, and an "
                "understated bound reads an underpowered round as a clean result")
        _timestamp(self.at, "at")
        _text(self.source_event_id, "source_event_id")
        if self.reference_gain is not None:
            object.__setattr__(self, "reference_gain", _finite_number(
                self.reference_gain, "reference_gain"))
        if self.stratum != evaluator_api.STRATUM_CONFIRMATION:
            raise GuardInputError(
                f"stratum: {self.stratum!r} — a readiness series admits only "
                f"{evaluator_api.STRATUM_CONFIRMATION!r} evidence; selection evidence "
                "is structurally unfit to report how ready a candidate is"
            )

    def could_have_detected(self, magnitude: float) -> bool:
        """Would an effect of this size have been visible in this round?

        The seam mirror of `release.readiness.ParityFigure.could_have_detected`,
        and deliberately one line over a bound the producer published rather than
        one this side derived. False means this round cannot tell "nothing moved"
        from "an effect of `magnitude` moved and we could not see it", so nothing
        may be concluded from it about a change of that size.

        The two planes cannot import each other, so the anti-drift mechanism is a
        test that runs BOTH over one figure and asserts they answer the same
        (`test_the_two_planes_answer_the_power_question_identically`) rather than
        a comment asking the next author to keep them aligned.
        """
        return _finite_number(magnitude, "magnitude") > self.sensitivity_bound

    @property
    def readiness(self) -> float:
        raise ParityHasNoMagnitude(
            f"round {self.round_index}: {self.cells_at_parity} of "
            f"{self.protected_cells} protected cell(s) resolved below the campaign's "
            f"own sensitivity (MDE {self.mde}, floor {self.noise_floor}) and none was "
            "orderable. There is no readiness magnitude on this round, and a plateau "
            "computed through a substituted one would be a trend in a quantity nobody "
            "measured")

    def to_dict(self) -> dict:
        """The serialized round — carrying NO `readiness` key, not a null one.

        `readiness` raises on the object and the dict must not undo that. These
        dicts are what land in `GuardDecision.detail["window"]` and from there in
        a journal, where the type is gone and only the mapping survives; a
        `"readiness": None` sitting in that window is `entry["readiness"] or 0.0`
        away from being trended as a round that measured zero improvement. Absent
        means a reader that assumes a magnitude gets a `KeyError` at the line that
        assumed it, which is the same refusal one layer out.
        """
        return {
            "round_index": self.round_index, "at": self.at,
            "protected_cells": self.protected_cells,
            "cells_at_parity": self.cells_at_parity,
            "mde": self.mde, "noise_floor": self.noise_floor,
            "sensitivity_bound": self.sensitivity_bound,
            "reference_gain": self.reference_gain,
            "source_event_id": self.source_event_id, "stratum": self.stratum,
            "orderable": False, "no_magnitude_reason": (
                f"{self.cells_at_parity} of {self.protected_cells} protected cell(s) "
                f"resolved below the campaign's own sensitivity (MDE {self.mde}, floor "
                f"{self.noise_floor}); sub-floor does not mean zero"),
        }


def _required(fields: Mapping, key: str):
    """Read a key the producer must have sent, or refuse. Never a fallback.

    A default here would be a number nobody measured wearing the shape of one
    that was measured, and the seam's whole job is that the shape decides.
    """
    if key not in fields:
        raise GuardInputError(
            f"fields is missing {key!r} ({sorted(fields)}); a figure's "
            "observation_fields() always carries it, so a mapping without it came "
            "from somewhere else. Substituting a default would put a value the "
            "producer never sent into the series — and for the sensitivity keys the "
            "default reads as the SHARPEST possible measurement, not a missing one")
    return fields[key]


def observation_from_fields(*, round_index: int, at: str,
                            fields: Mapping) -> ReadinessSeriesEntry:
    """Build the RIGHT series entry from a release-plane figure's own fields.

    AK5 does not import AK4 — the controller consumes the release plane, not the
    other way round — so what crosses the seam is a mapping produced by
    `release.readiness.<figure>.observation_fields()`. This is the ONE place that
    reads its shape, and it reads it as an EITHER-OR: a mapping carrying
    `readiness` builds a `ReadinessObservation`, a mapping carrying
    `cells_at_parity` builds a `ParityObservation`, and a mapping carrying both
    or neither is refused rather than resolved by precedence.

    There is deliberately no path here that can produce a magnitude for a round
    that had none, and no caller that has to remember to check anything: the
    branch is on which keys EXIST, and the parity mapping does not contain a
    number that could be mistaken for readiness.

    NOTHING HERE IS DEFAULTED, and that is the second mechanism. The branch is on
    which keys exist, so a `.get(key, fallback)` for anything else would mean an
    ABSENT key silently becomes a value the producer never sent — on the parity
    side the fallbacks would have been `mde=0.0` and `noise_floor=0.0`, which is
    not a missing sensitivity but the SHARPEST one expressible: "we resolved to
    zero and nothing moved" is the strongest parity claim there is, invented from
    a dropped key. `stratum` is worse again: defaulting an absent one to
    `confirmation` enforces P-AK-SEARCH-1 only against producers that volunteer
    the field, which is not enforcement. A mapping that lost a key is a mapping
    nobody can interpret, and this refuses it.
    """
    if not isinstance(fields, Mapping):
        raise GuardInputError(
            f"fields: expected a mapping of a figure's observation_fields(), got "
            f"{type(fields).__name__}")
    has_readiness = "readiness" in fields
    has_parity = "cells_at_parity" in fields
    if has_readiness and has_parity:
        raise GuardInputError(
            "fields carries both 'readiness' and 'cells_at_parity'; a round either "
            "produced an orderable magnitude or it did not, and a mapping claiming "
            "both would be resolved here by whichever branch happened to be first")
    if not has_readiness and not has_parity:
        raise GuardInputError(
            f"fields carries neither 'readiness' nor 'cells_at_parity' "
            f"({sorted(fields)}); a series entry with no result is not a round that "
            "happened")
    stratum = _required(fields, "stratum")
    if has_readiness:
        return ReadinessObservation(
            round_index=round_index, readiness=fields["readiness"], at=at,
            source_event_id=_required(fields, "source_event_id"), stratum=stratum)
    return ParityObservation(
        round_index=round_index,
        protected_cells=_required(fields, "protected_cells"),
        cells_at_parity=fields["cells_at_parity"],
        mde=_required(fields, "mde"),
        noise_floor=_required(fields, "noise_floor"),
        # Required even though `None` is a legal VALUE. A campaign that declared
        # no target and a producer that dropped the key are different facts, and
        # `.get("reference_gain")` would render them identically — as the first,
        # which is the one that silently disables the only branch able to
        # conclude anything from an all-parity window.
        sensitivity_bound=_required(fields, "sensitivity_bound"),
        reference_gain=_required(fields, "reference_gain"), at=at,
        source_event_id=_required(fields, "source_event_id"), stratum=stratum)


@dataclass(frozen=True)
class PlateauPolicy:
    """The plateau window and the improvement floor it is measured against.

    `improvement_floor` is the campaign's DERIVED floor (the calibration block's
    `φ`, or a quantity derived from it), and `floor_receipt` must name the
    calibration record it came from. P-AK-SEARCH-1: *"No value in this list may be
    supplied as a literal."* A floor with no receipt is a literal with a
    plausible name, so it is refused here rather than trusted.
    """

    window_rounds: int
    improvement_floor: float
    floor_receipt: str

    def __post_init__(self) -> None:
        window = _positive_int(self.window_rounds, "window_rounds")
        if window < 2:
            raise GuardInputError(
                "window_rounds: a plateau is a statement about a TREND; one "
                "observation cannot show the absence of improvement"
            )
        floor = _finite_number(self.improvement_floor, "improvement_floor", minimum=0.0)
        if floor <= 0:
            raise GuardInputError(
                "improvement_floor: must be strictly positive; a floor of zero calls "
                "any nonzero jitter an improvement and the plateau never fires"
            )
        object.__setattr__(self, "improvement_floor", floor)
        _text(self.floor_receipt, "floor_receipt")


@dataclass(frozen=True)
class CommandRetryLedger:
    """Attempts against `stop_policy.max_command_retries` (`OPERATING_CONSTRAINTS.md:44-46`).

    `attempts` counts every attempt including the first, so the retries used are
    `attempts - 1`. A ledger declaring a cap above `MAX_COMMAND_RETRIES` is
    refused: the constraint is a project rule and a campaign may tighten it, never
    loosen it.
    """

    command_id: str
    attempts: int
    last_error: str
    receipt: str
    max_retries: int = MAX_COMMAND_RETRIES

    def __post_init__(self) -> None:
        _text(self.command_id, "command_id")
        _text(self.last_error, "last_error")
        _text(self.receipt, "receipt")
        attempts = _nonneg_int(self.attempts, "attempts")
        if attempts == 0:
            raise GuardInputError(
                "attempts: a retry ledger for a command that never ran describes "
                "nothing; the first attempt counts as one"
            )
        _nonneg_int(self.max_retries, "max_retries")
        if self.max_retries > MAX_COMMAND_RETRIES:
            raise GuardInputError(
                f"max_retries={self.max_retries} exceeds the project cap of "
                f"{MAX_COMMAND_RETRIES} (OPERATING_CONSTRAINTS.md:44-46); a campaign "
                "may tighten this, never loosen it"
            )

    @property
    def retries_used(self) -> int:
        return self.attempts - 1

    @property
    def exhausted(self) -> bool:
        return self.retries_used >= self.max_retries


@dataclass(frozen=True)
class RepairLedger:
    """§8.5.1 repair and build-failure caps for one proposal.

    *"Repairs are capped per proposal; exceeding the cap is a `PLANNER_DEGRADED`
    signal, not another retry."* This ledger's guard therefore REFUSES the repair
    and emits the signal; the STOP is `guard_planner_degraded`'s to make from the
    accumulated ledger, so one condition keeps one spelling.
    """

    proposal_id: str
    repairs_attempted: int
    max_repairs: int
    consecutive_build_failures: int
    max_consecutive_build_failures: int
    receipt: str

    def __post_init__(self) -> None:
        _text(self.proposal_id, "proposal_id")
        _text(self.receipt, "receipt")
        _nonneg_int(self.repairs_attempted, "repairs_attempted")
        _positive_int(self.max_repairs, "max_repairs")
        _nonneg_int(self.consecutive_build_failures, "consecutive_build_failures")
        _nonneg_int(
            self.max_consecutive_build_failures, "max_consecutive_build_failures"
        )


# =============================================================================
# The guards
# =============================================================================

def guard_integrity(ledger: IntegrityLedger) -> GuardDecision:
    """`INTEGRITY_STOP` — repeated tamper / reward-hacking signal (§8.10).

    First in `STOP_PRECEDENCE` because integrity is lexicographically prior
    (invariant 6) and because a tampered record poisons every other guard's
    input: a budget computed from forged costs and a plateau computed from forged
    rates are not weaker conclusions, they are conclusions about nothing.
    """
    if not isinstance(ledger, IntegrityLedger):
        raise GuardInputError("ledger: must be an IntegrityLedger")

    tolerance = ledger.max_consecutive_integrity_failures
    # The CONSECUTIVE run, never the campaign's whole signal history. Slicing by
    # `len - n` rather than `[-n:]` because `signals[-0:]` is the entire list, and
    # that is how a CONTINUE ends up carrying one receipt per signal ever seen —
    # a decision whose size grows monotonically with campaign length, journaled
    # every round.
    run = ledger.signals[len(ledger.signals) - ledger.consecutive_failures:]
    if ledger.consecutive_failures <= tolerance:
        return GuardDecision(
            guard_id=GUARD_INTEGRITY,
            outcome=CONTINUE,
            reason=(
                f"{ledger.consecutive_failures} consecutive integrity failure(s) is "
                f"within the declared tolerance of {tolerance}"
            ),
            detail={
                "consecutive_failures": ledger.consecutive_failures,
                "max_consecutive_integrity_failures": tolerance,
                # The history is COUNTED rather than dropped: bounding the evidence
                # must not make a campaign that has survived 500 integrity signals
                # look like one that has survived none.
                "signals_recorded": len(ledger.signals),
            },
            evidence=tuple(signal.receipt for signal in run),
        )

    recent = run
    return GuardDecision(
        guard_id=GUARD_INTEGRITY,
        outcome=STOP,
        stop_state=sm.INTEGRITY_STOP,
        reason=(
            f"{ledger.consecutive_failures} consecutive integrity signal(s) exceed the "
            f"declared tolerance of {tolerance}; correctness and integrity are "
            "lexicographically prior to speed (invariant 6)"
        ),
        detail={
            "signal": recent[-1].signal,
            "occurrences": ledger.consecutive_failures,
            "receipt": recent[-1].receipt,
            "max_consecutive_integrity_failures": tolerance,
            "signals": [signal.to_dict() for signal in recent],
        },
        evidence=tuple(signal.receipt for signal in recent),
    )


def guard_anchor_moved(
    *,
    recorded: Optional[sm.AnchorIdentity],
    observed: Optional[sm.AnchorIdentity],
    receipt: str,
) -> GuardDecision:
    """`ANCHOR_MOVED` — the denominator of every ratio changed (§8.9, AK-D22).

    The comparison itself is `state_machine.check_anchor_identity`, called rather
    than reimplemented: two implementations of "is this the same anchor" is two
    answers, and the one that gets tested is not necessarily the one that runs.
    `COULD_NOT_CHECK` becomes `COULD_NOT_EVALUATE` — an anchor nobody looked at is
    not an anchor that did not move.
    """
    _text(receipt, "receipt")
    for name, value in (("recorded", recorded), ("observed", observed)):
        if value is not None and not isinstance(value, sm.AnchorIdentity):
            raise GuardInputError(f"{name}: must be a state_machine.AnchorIdentity or None")

    check = sm.check_anchor_identity(recorded, observed)
    if check.outcome == schemas.PASS:
        return GuardDecision(
            guard_id=GUARD_ANCHOR,
            outcome=CONTINUE,
            reason="anchor identity is byte-for-byte what BOOTSTRAP recorded",
            detail={"anchor_check": check.outcome},
            evidence=(receipt,),
        )
    if check.outcome == schemas.COULD_NOT_CHECK:
        return GuardDecision(
            guard_id=GUARD_ANCHOR,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                "anchor identity could not be compared: " + "; ".join(check.reasons)
            ),
            detail={"anchor_check": check.outcome, "reasons": list(check.reasons)},
            evidence=(receipt,),
        )

    affected = sorted(
        set(recorded.backends) | set(observed.backends)
    ) if recorded is not None and observed is not None else []
    return GuardDecision(
        guard_id=GUARD_ANCHOR,
        outcome=STOP,
        stop_state=sm.ANCHOR_MOVED,
        reason=(
            "production identity changed outside a loop-initiated freeze: "
            + "; ".join(check.reasons)
        ),
        detail={
            "recorded_anchor": recorded.to_dict() if recorded else None,
            "observed_anchor": observed.to_dict() if observed else None,
            "affected_backends": affected,
            "supersession_marker": "superseded_by_anchor_move",
            "mismatches": list(check.reasons),
        },
        directives=(DIRECTIVE_REANCHOR, DIRECTIVE_PERSIST_AND_DRAIN),
        evidence=(receipt,),
    )


def guard_host_uptime(
    host: HostHealth, *, owner: str, escalation_deadline: str, now: str
) -> GuardDecision:
    """`HOST_REBOOT_REQUIRED` — the §10.7 one-week uptime ceiling.

    *"Any decision-grade unattended campaign is therefore capped at roughly one
    week of host uptime. The loop must request the reboot as a decision package,
    persist fully, and resume."* Reboots are operator authority
    (`feedback_operator_owns_host_reboots`), so this guard REQUESTS and never
    performs — and the module cannot perform one, since it may not signal or
    execute anything at all.
    """
    if not isinstance(host, HostHealth):
        raise GuardInputError("host: must be a HostHealth")
    _text(owner, "owner")
    now_dt = _timestamp(now, "now")
    deadline_dt = _timestamp(escalation_deadline, "escalation_deadline")
    if deadline_dt <= now_dt:
        raise GuardInputError(
            f"escalation_deadline {escalation_deadline!r} is not after now {now!r}; a "
            "deadline already past is a default that has silently taken effect"
        )

    if not host.observable:
        return GuardDecision(
            guard_id=GUARD_HOST_UPTIME,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                "host uptime could not be observed; the §10.7 ceiling cannot be "
                "checked and an unchecked host-health tier is not a satisfied one "
                "(P-AK-SEARCH-1 precondition 3)"
            ),
            detail={"receipt": host.receipt, "observed_at": host.observed_at},
            evidence=(host.receipt,),
        )
    if host.uptime_seconds < host.ceiling_seconds:
        return GuardDecision(
            guard_id=GUARD_HOST_UPTIME,
            outcome=CONTINUE,
            reason=(
                f"uptime {host.uptime_seconds}s is under the {host.ceiling_seconds}s "
                "ceiling"
            ),
            detail={
                "uptime_seconds": host.uptime_seconds,
                "ceiling_seconds": host.ceiling_seconds,
            },
            evidence=(host.receipt,),
        )

    package = DecisionPackage(
        context=(
            f"Host uptime is {host.uptime_seconds}s, at or above the §10.7 ceiling of "
            f"{host.ceiling_seconds}s (bench-cpu.md:17-19). No decision-grade search "
            "measurement may proceed until the host is rebooted. Reboots are operator "
            "authority; the loop has persisted its state and can resume from it."
        ),
        options=(
            DecisionOption(
                option_id="reboot_now",
                summary="Reboot the host at the next session boundary, then resume the campaign",
                tradeoffs=(
                    "costs a pre-reboot wrap-up including commit and push "
                    "(SESSION_LIFECYCLE.md:43-55)",
                    "restores the host-health tier, so measurement can continue",
                ),
                consequence_if_chosen="campaign resumes from its persisted state after the reboot",
                reversible=True,
            ),
            DecisionOption(
                option_id="hold_drained",
                summary="Leave the campaign drained and stopped until a reboot is scheduled",
                tradeoffs=(
                    "no compute is spent and no evidence is produced",
                    "the champion lineage ages against a moving production tip",
                ),
                consequence_if_chosen="campaign stays stopped; nothing is measured or discarded",
                reversible=True,
            ),
            DecisionOption(
                option_id="non_measurement_only",
                summary="Continue only work that produces no measurement (analysis, drafting, review)",
                tradeoffs=(
                    "keeps the planner productive without touching the host-health tier",
                    "banks nothing: no candidate can be ranked while the tier is violated",
                ),
                consequence_if_chosen="no search record is emitted until after the reboot",
                reversible=True,
            ),
        ),
        recommendation="reboot_now",
        default="hold_drained",
        default_rationale=(
            "the safe branch is the one that measures nothing: a search record taken "
            "over the uptime ceiling is INVALID, and producing one is worse than "
            "producing none"
        ),
        owner=owner,
        deadline=escalation_deadline,
    )
    return GuardDecision(
        guard_id=GUARD_HOST_UPTIME,
        outcome=STOP,
        stop_state=sm.HOST_REBOOT_REQUIRED,
        reason=(
            f"host uptime {host.uptime_seconds}s reached the §10.7 ceiling of "
            f"{host.ceiling_seconds}s; no decision-grade measurement proceeds and a "
            "reboot is operator authority"
        ),
        detail={
            "uptime_seconds": host.uptime_seconds,
            "ceiling_seconds": host.ceiling_seconds,
            "observed_at": host.observed_at,
            **package.to_detail(),
        },
        directives=(DIRECTIVE_PERSIST_AND_DRAIN, DIRECTIVE_ESCALATE_TO_OPERATOR),
        decision_package=package,
        evidence=(host.receipt,),
    )


def guard_resource_available(observation: ResourceClaimObservation) -> GuardDecision:
    """`RESOURCE_UNAVAILABLE` — *"persist and drain, never busy-wait"* (§8.10).

    The no-busy-wait rule is enforced by what this guard CANNOT say: the
    `DIRECTIVES` vocabulary has no WAIT, POLL, SLEEP or RETRY member, and this
    module imports no clock and no `time`, so the only expressible response is to
    persist and drain. `audit_directive_vocabulary()` and
    `audit_no_write_process_or_wait_paths()` keep it that way.
    """
    if not isinstance(observation, ResourceClaimObservation):
        raise GuardInputError("observation: must be a ResourceClaimObservation")

    if observation.acquired:
        return GuardDecision(
            guard_id=GUARD_RESOURCE,
            outcome=CONTINUE,
            reason=(
                f"{observation.claim_kind} claim on {observation.resource} is HELD "
                f"under receipt {observation.receipt}"
            ),
            detail={
                "resource": observation.resource,
                "claim_kind": observation.claim_kind,
                "receipt": observation.receipt,
            },
            evidence=(observation.receipt,),
        )

    evidence = (observation.receipt,) if observation.receipt else (
        f"resource_observation:{observation.resource}@{observation.observed_at}",
    )
    return GuardDecision(
        guard_id=GUARD_RESOURCE,
        outcome=STOP,
        stop_state=sm.RESOURCE_UNAVAILABLE,
        reason=(
            f"{observation.claim_kind} claim on {observation.resource} was not "
            f"acquired: {observation.unavailable_reason}. The loop persists and "
            "drains; it does not wait for the resource to look free, because looking "
            "free is TOCTOU and not exclusion"
        ),
        detail={
            "resource": observation.resource,
            "claim_kind": observation.claim_kind,
            "held_by": observation.held_by,
            "unavailable_reason": observation.unavailable_reason,
            "observed_at": observation.observed_at,
            "busy_wait": "forbidden by §8.10",
        },
        directives=(DIRECTIVE_PERSIST_AND_DRAIN,),
        evidence=evidence,
    )


def guard_storage_headroom(observation: StorageObservation) -> GuardDecision:
    """`DISK_PRESSURE` — storage headroom below the campaign floor (§5.8, §8.10).

    P-AK-SEARCH-1 precondition 7 has a branch most implementations skip: when the
    ALREADY-ELIGIBLE expiry backlog would clear the floor, the campaign reclaims
    and continues; only when it would not does the campaign stop. Reclamation
    outside the enumerated expirable classes is operator authority, so this guard
    never proposes it.
    """
    if not isinstance(observation, StorageObservation):
        raise GuardInputError("observation: must be a StorageObservation")

    state = observation.state
    base = {
        "path": observation.path,
        "free_bytes": state.free_bytes,
        "floor_bytes": state.floor_bytes,
        "expirable_backlog_bytes": observation.expirable_backlog_bytes,
    }

    # `StorageState.pressured` is `state == DISK_PRESSURE` — a STATUS STRING. An
    # observation whose string says anything else reads as healthy no matter what
    # its numbers say, which is §2.5 row 4's defect ("the budget was only a status
    # string") reproduced on the disk. The numbers and the label are two readings
    # of one fact; when they disagree, the honest answer is that the fact was not
    # observed, not whichever of the two happens to be cheaper to believe.
    if (state.free_bytes < state.floor_bytes) != state.pressured:
        return GuardDecision(
            guard_id=GUARD_STORAGE,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"storage observation for {observation.path} is self-contradictory: "
                f"free {state.free_bytes} vs floor {state.floor_bytes} bytes says "
                f"pressured={state.free_bytes < state.floor_bytes}, while its state "
                f"string {state.state!r} says pressured={state.pressured}. A headroom "
                "verdict cannot be read off a label its own numbers contradict"
            ),
            detail={**base, "state": state.state},
            evidence=(observation.receipt,),
        )

    if not state.pressured:
        return GuardDecision(
            guard_id=GUARD_STORAGE,
            outcome=CONTINUE,
            reason=(
                f"free {state.free_bytes} bytes is at or above the "
                f"{state.floor_bytes}-byte floor on {observation.path}"
            ),
            detail=base,
            evidence=(observation.receipt,),
        )

    reclaimable_free = state.free_bytes + observation.expirable_backlog_bytes
    if reclaimable_free >= state.floor_bytes:
        return GuardDecision(
            guard_id=GUARD_STORAGE,
            outcome=REFUSE,
            reason=(
                f"free {state.free_bytes} bytes is below the {state.floor_bytes}-byte "
                f"floor, but the already-eligible expiry backlog of "
                f"{observation.expirable_backlog_bytes} bytes clears it; reclaim before "
                "allocating (P-AK-SEARCH-1 precondition 7)"
            ),
            detail={**base, "free_after_expiry_bytes": reclaimable_free},
            directives=(DIRECTIVE_RECLAIM_EXPIRABLE_FIRST,),
            evidence=(observation.receipt,),
        )

    return GuardDecision(
        guard_id=GUARD_STORAGE,
        outcome=STOP,
        stop_state=sm.DISK_PRESSURE,
        reason=(
            f"free {state.free_bytes} bytes is below the {state.floor_bytes}-byte floor "
            f"on {observation.path} and the eligible expiry backlog of "
            f"{observation.expirable_backlog_bytes} bytes does not clear it; further "
            "reclamation is operator authority, so the campaign stops"
        ),
        detail={
            **base,
            "free_after_expiry_bytes": reclaimable_free,
            "shortfall_bytes": state.floor_bytes - reclaimable_free,
            "state_reasons": list(state.reasons),
        },
        directives=(DIRECTIVE_PERSIST_AND_DRAIN,),
        evidence=(observation.receipt,),
    )


def guard_budget(ledger: BudgetLedger) -> GuardDecision:
    """`BUDGET_STOP` — wall / resource / candidate / token / storage (§8.10).

    Reports the FIRST exhausted dimension in §7.1's declared order and lists every
    exhausted one in `detail`, so two runs in the same state name the same
    governing budget. §12's zero-yield row is why `consumed` is an attributed
    figure with a receipt rather than a running counter: AutoPilot's budget was
    *"only a status string"* and cost-per-banked-win was unauditable.
    """
    if not isinstance(ledger, BudgetLedger):
        raise GuardInputError("ledger: must be a BudgetLedger")

    exhausted = ledger.exhausted
    if not exhausted:
        return GuardDecision(
            guard_id=GUARD_BUDGET,
            outcome=CONTINUE,
            reason="every declared budget dimension has headroom",
            detail={"dimensions": [d.to_dict() for d in ledger.dimensions]},
            evidence=tuple(d.receipt for d in ledger.dimensions),
        )

    governing = exhausted[0]
    return GuardDecision(
        guard_id=GUARD_BUDGET,
        outcome=STOP,
        stop_state=sm.BUDGET_STOP,
        reason=(
            f"budget {governing.name} is exhausted: {governing.consumed} consumed of "
            f"{governing.limit} declared"
        ),
        detail={
            "budget": governing.name,
            "limit": governing.limit,
            "consumed": governing.consumed,
            "exhausted": [d.to_dict() for d in exhausted],
            "dimensions": [d.to_dict() for d in ledger.dimensions],
        },
        directives=(DIRECTIVE_PERSIST_AND_DRAIN,),
        evidence=tuple(d.receipt for d in exhausted),
    )


def guard_controller_spend(
    ledger: BudgetLedger, policy: SpendBreakerPolicy
) -> GuardDecision:
    """The controller token/spend BREAKER — forces local planning, never halts.

    §2.5 row 4: *"a spend breaker whose naive form **stopped the loop**"*, and the
    correction recorded against it is that the breaker *"forces local planning
    rather than halting"*. So this guard never returns STOP: at the ceiling the
    stop belongs to `guard_budget`, which is a different function with a different
    receipt, and the two cannot be confused for each other.
    """
    if not isinstance(ledger, BudgetLedger):
        raise GuardInputError("ledger: must be a BudgetLedger")
    if not isinstance(policy, SpendBreakerPolicy):
        raise GuardInputError("policy: must be a SpendBreakerPolicy")

    tokens = ledger.by_name("max_controller_tokens")
    detail = {
        "budget": tokens.name,
        "limit": tokens.limit,
        "consumed": tokens.consumed,
        "fraction": tokens.fraction,
        "breaker_fraction": policy.breaker_fraction,
        "policy_receipt": policy.policy_receipt,
    }
    if tokens.fraction < policy.breaker_fraction:
        return GuardDecision(
            guard_id=GUARD_SPEND_BREAKER,
            outcome=CONTINUE,
            reason=(
                f"controller spend at {tokens.fraction:.4f} of budget is below the "
                f"declared breaker fraction {policy.breaker_fraction}"
            ),
            detail=detail,
            evidence=(tokens.receipt, policy.policy_receipt),
        )
    return GuardDecision(
        guard_id=GUARD_SPEND_BREAKER,
        outcome=REFUSE,
        reason=(
            f"controller spend at {tokens.fraction:.4f} of budget reached the declared "
            f"breaker fraction {policy.breaker_fraction}; metered drafting is refused "
            "and planning continues locally. The breaker does not halt the loop — the "
            "ceiling stop belongs to guard_budget"
        ),
        detail=detail,
        directives=(DIRECTIVE_LOCAL_PLANNING_ONLY,),
        evidence=(tokens.receipt, policy.policy_receipt),
    )


def guard_evaluator_coverage(
    gaps: Sequence[CoverageGap],
    *,
    now: str,
    covered_surfaces_remaining: int,
    escalation_owner: str,
    escalation_deadline: str,
) -> GuardDecision:
    """`EVALUATOR_COVERAGE_GAP` — release blocked, research continues (§8.10).

    The gap is not a plain stop, and treating it as one would be wrong in the
    expensive direction: *"release blocked for the affected lineage, research
    continues on covered surfaces."* So:

      * open, owned, inside its deadline → `REFUSE` with `RELEASE_BLOCKED`;
      * still open at the next campaign boundary → `REFUSE` and ESCALATE, with the
        four-part package;
      * past its deadline, open across two consecutive freeze cycles, or no
        covered surface left to research → `STOP`.

    The two escalating branches are what stop it becoming *"a permanent silent
    block"*. P-AK-SEARCH-1 *No self-amendment*: the loop may draft an amendment
    for human review; it does not patch the instrument and it does not route
    around it, and neither branch here does either.
    """
    entries = _typed_tuple(gaps, "gaps", CoverageGap, non_empty=False)
    now_dt = _timestamp(now, "now")
    _text(escalation_owner, "escalation_owner")
    deadline_dt = _timestamp(escalation_deadline, "escalation_deadline")
    if deadline_dt <= now_dt:
        raise GuardInputError(
            f"escalation_deadline {escalation_deadline!r} is not after now {now!r}"
        )
    _nonneg_int(covered_surfaces_remaining, "covered_surfaces_remaining")

    if not entries:
        return GuardDecision(
            guard_id=GUARD_COVERAGE,
            outcome=CONTINUE,
            reason="no evaluator coverage gap is open",
            detail={"open_gaps": 0},
        )

    overdue = [gap for gap in entries if _timestamp(gap.deadline, "deadline") < now_dt]
    program_defects = [gap for gap in entries if gap.freeze_cycles_open >= 2]
    escalating = [gap for gap in entries if gap.boundaries_open >= 1]

    def _package(governing: CoverageGap, why: str, stopping: bool) -> DecisionPackage:
        options = [
            DecisionOption(
                option_id="amend_evaluator",
                summary=(
                    "Review and apply the drafted amendment that adds the missing "
                    f"coverage class {governing.missing_coverage_class!r}"
                ),
                tradeoffs=(
                    "human amendment under the measurement trust boundary; the loop "
                    "may draft but never apply (P-AK-SEARCH-1: no self-amendment)",
                    "unblocks release for the affected lineage once applied",
                ),
                consequence_if_chosen=(
                    f"lineage {governing.blocked_lineage} becomes release-eligible "
                    "again after the amendment is reviewed and applied"
                ),
                reversible=True,
            ),
            DecisionOption(
                option_id="narrow_lineage",
                summary=(
                    f"Abandon lineage {governing.blocked_lineage} and continue only on "
                    "surfaces the evaluator covers"
                ),
                tradeoffs=(
                    "discards the blocked lineage's accumulated work for this campaign",
                    "no evaluator change is required and research continues immediately",
                ),
                consequence_if_chosen="campaign continues with a reduced surface",
                reversible=True,
            ),
            DecisionOption(
                option_id="hold_open",
                summary="Leave the gap open and keep release blocked for that lineage",
                tradeoffs=(
                    "costs nothing now",
                    "a gap open across two consecutive freeze cycles is a program-level "
                    "defect, not a standing condition (§8.10)",
                ),
                consequence_if_chosen=(
                    "release stays blocked for the lineage; the gap re-escalates at the "
                    "next campaign boundary"
                ),
                reversible=True,
            ),
        ]
        return DecisionPackage(
            context=(
                f"Evaluator coverage gap {governing.gap_id}: missing coverage class "
                f"{governing.missing_coverage_class!r} blocks release for lineage "
                f"{governing.blocked_lineage}. Owner {governing.owner}, deadline "
                f"{governing.deadline}, open across {governing.boundaries_open} campaign "
                f"boundary/ies and {governing.freeze_cycles_open} freeze cycle(s). "
                f"Drafted amendment: {governing.amendment_draft_ref or 'none yet'}. "
                f"{why}"
            ),
            options=tuple(options),
            recommendation="amend_evaluator",
            default="hold_open" if not stopping else "narrow_lineage",
            default_rationale=(
                "the default may not be the one that unblocks a release: an evaluator "
                "amendment is human-only, so silence must not apply one"
            ),
            owner=escalation_owner,
            deadline=escalation_deadline,
        )

    stopping_reason = None
    if overdue:
        governing = overdue[0]
        stopping_reason = (
            f"coverage gap {governing.gap_id} passed its escalation deadline "
            f"{governing.deadline}"
        )
    elif program_defects:
        governing = program_defects[0]
        stopping_reason = (
            f"coverage gap {governing.gap_id} has been open across "
            f"{governing.freeze_cycles_open} consecutive freeze cycles, which §8.10 "
            "reports as a program-level defect rather than a standing condition"
        )
    elif covered_surfaces_remaining == 0:
        governing = entries[0]
        stopping_reason = (
            f"coverage gap {governing.gap_id} blocks lineage "
            f"{governing.blocked_lineage} and no covered surface remains to research"
        )

    if stopping_reason is not None:
        package = _package(governing, stopping_reason, stopping=True)
        return GuardDecision(
            guard_id=GUARD_COVERAGE,
            outcome=STOP,
            stop_state=sm.EVALUATOR_COVERAGE_GAP,
            reason=stopping_reason,
            detail={
                "missing_coverage_class": governing.missing_coverage_class,
                "blocked_lineage": governing.blocked_lineage,
                "owner": governing.owner,
                "deadline": governing.deadline,
                "gap_id": governing.gap_id,
                "open_gaps": [gap.to_dict() for gap in entries],
                "covered_surfaces_remaining": covered_surfaces_remaining,
                **package.to_detail(),
            },
            directives=(DIRECTIVE_RELEASE_BLOCKED, DIRECTIVE_ESCALATE_TO_OPERATOR),
            decision_package=package,
            evidence=tuple(gap.receipt for gap in entries),
        )

    governing = entries[0]
    if escalating:
        governing = escalating[0]
        package = _package(
            governing,
            "The gap was still open at a campaign boundary, so §8.10 escalates it.",
            stopping=False,
        )
        return GuardDecision(
            guard_id=GUARD_COVERAGE,
            outcome=REFUSE,
            reason=(
                f"coverage gap {governing.gap_id} is still open at a campaign boundary; "
                f"release stays blocked for lineage {governing.blocked_lineage} and the "
                "gap escalates to the operator. Research continues on covered surfaces"
            ),
            detail={
                "gap_id": governing.gap_id,
                "open_gaps": [gap.to_dict() for gap in entries],
                "covered_surfaces_remaining": covered_surfaces_remaining,
                **package.to_detail(),
            },
            directives=(DIRECTIVE_RELEASE_BLOCKED, DIRECTIVE_ESCALATE_TO_OPERATOR),
            decision_package=package,
            evidence=tuple(gap.receipt for gap in entries),
        )

    return GuardDecision(
        guard_id=GUARD_COVERAGE,
        outcome=REFUSE,
        reason=(
            f"coverage gap {governing.gap_id} blocks release for lineage "
            f"{governing.blocked_lineage} (owner {governing.owner}, deadline "
            f"{governing.deadline}); research continues on covered surfaces"
        ),
        detail={
            "gap_id": governing.gap_id,
            "open_gaps": [gap.to_dict() for gap in entries],
            "covered_surfaces_remaining": covered_surfaces_remaining,
        },
        directives=(DIRECTIVE_RELEASE_BLOCKED,),
        evidence=tuple(gap.receipt for gap in entries),
    )


def guard_operator_input(
    questions: Sequence[OperatorQuestion], *, now: str
) -> GuardDecision:
    """`OPERATOR_INPUT_REQUIRED` — rendered as a four-part package (§18 item 7).

    The stop carries the package verbatim, so the artifact an operator reads and
    the evidence the journal holds are the same object. An "answered" question
    whose answer has no journaled event id yields `COULD_NOT_EVALUATE`, because
    clearing a block on an unrecorded answer is the fail-open shape: the block was
    real, the clearance was a belief.
    """
    entries = _typed_tuple(questions, "questions", OperatorQuestion, non_empty=False)
    _timestamp(now, "now")

    unrecorded = [
        q for q in entries if q.answered and not (q.answered_event_id or "").strip()
    ]
    if unrecorded:
        return GuardDecision(
            guard_id=GUARD_OPERATOR_INPUT,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                "question(s) "
                + ", ".join(q.question_id for q in unrecorded)
                + " are marked answered with no journaled answer event; an answer that "
                "exists only in the controller's belief cannot clear a block"
            ),
            detail={"unrecorded": [q.question_id for q in unrecorded]},
        )

    open_blocking = [q for q in entries if q.blocking and not q.answered]
    if not open_blocking:
        return GuardDecision(
            guard_id=GUARD_OPERATOR_INPUT,
            outcome=CONTINUE,
            reason="no blocking operator question is open",
            detail={"open_questions": 0, "questions": len(entries)},
        )

    governing = open_blocking[0]
    return GuardDecision(
        guard_id=GUARD_OPERATOR_INPUT,
        outcome=STOP,
        stop_state=sm.OPERATOR_INPUT_REQUIRED,
        reason=(
            f"operator decision {governing.question_id} is open and blocking; the loop "
            "may not answer it"
        ),
        detail={
            "question_id": governing.question_id,
            "raised_at": governing.raised_at,
            "open_question_ids": [q.question_id for q in open_blocking],
            **governing.package.to_detail(),
        },
        directives=(DIRECTIVE_ESCALATE_TO_OPERATOR, DIRECTIVE_PERSIST_AND_DRAIN),
        decision_package=governing.package,
        evidence=tuple(q.receipt for q in open_blocking),
    )


#: The §8.10 degradation families, in the order they are adjudicated. Each row is
#: (signal name, health attribute, policy attribute). Order is declared once so
#: two identical journals report the same governing signal.
_DEGRADATION_SIGNALS = (
    ("repeated_no_ops", "consecutive_noop_rounds", "max_consecutive_noop_rounds"),
    ("repeated_fingerprints", "repeated_fingerprint_count", "max_repeated_fingerprints"),
    ("invalid_dispatches", "invalid_dispatch_count", "max_invalid_dispatches"),
    ("contradicted_narrative", "contradicted_narrative_count",
     "max_contradicted_narratives"),
    ("unavailable_dependency_loop", "unavailable_dependency_rounds",
     "max_unavailable_dependency_rounds"),
    ("consecutive_build_failures", "consecutive_build_failures",
     "max_consecutive_build_failures"),
    ("repair_cap_exceeded", "repair_cap_exceedances", "max_repair_cap_exceedances"),
)


def guard_planner_degraded(
    health: PlannerHealth, policy: PlannerHealthPolicy
) -> GuardDecision:
    """`PLANNER_DEGRADED` — the searcher is broken, not finished (§8.10, AK-D32).

    *"Distinct from plateau: plateau means the search is done, degraded means the
    searcher is broken, and conflating them once cost this project months of paid
    no-ops."* The four families §8.10 names are checked in a declared order, plus
    §8.5.1's build-failure and repair caps whose exceedance the design calls a
    `PLANNER_DEGRADED` signal explicitly.

    A crossed signal with no receipt returns `COULD_NOT_EVALUATE`: §8.10 requires
    both `signal` and `receipt`, and a degraded verdict asserted without one is
    the same narration this stop exists to catch.

    A partial window also returns `COULD_NOT_EVALUATE`. Declaring the searcher
    broken on fewer rounds than the campaign declared is the identical error to
    declaring a plateau on a partial window, and it is worth the same refusal.
    """
    if not isinstance(health, PlannerHealth):
        raise GuardInputError("health: must be a PlannerHealth")
    if not isinstance(policy, PlannerHealthPolicy):
        raise GuardInputError("policy: must be a PlannerHealthPolicy")

    base = {
        "planner_health": health.to_dict(),
        "health_digest": health.digest,
        "policy_receipt": policy.policy_receipt,
        "window_rounds": policy.window_rounds,
    }

    crossed: list = []
    for signal, health_attr, policy_attr in _DEGRADATION_SIGNALS:
        observed = getattr(health, health_attr)
        threshold = getattr(policy, policy_attr)
        if observed > threshold:
            crossed.append((signal, observed, threshold))

    if not crossed:
        if health.rounds_observed < policy.window_rounds:
            return GuardDecision(
                guard_id=GUARD_PLANNER_HEALTH,
                outcome=COULD_NOT_EVALUATE,
                reason=(
                    f"only {health.rounds_observed} of the declared "
                    f"{policy.window_rounds}-round window has been observed; planner "
                    "health is not yet decidable, and 'no signal yet' is not 'healthy'"
                ),
                detail=base,
            )
        return GuardDecision(
            guard_id=GUARD_PLANNER_HEALTH,
            outcome=CONTINUE,
            reason=(
                f"no degradation signal crossed its declared threshold over "
                f"{health.rounds_observed} round(s)"
            ),
            detail=base,
            evidence=(policy.policy_receipt,),
        )

    signal, observed, threshold = crossed[0]
    receipt = health.receipts.get(signal)
    if not isinstance(receipt, str) or not receipt.strip():
        return GuardDecision(
            guard_id=GUARD_PLANNER_HEALTH,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"signal {signal!r} crossed its threshold ({observed} > {threshold}) but "
                "no receipt was supplied for it; §8.10 requires signal AND receipt, and "
                "a degraded verdict with no receipt is the narration it exists to catch"
            ),
            detail={
                **base,
                "crossed": [
                    {"signal": s, "observed": o, "threshold": t} for s, o, t in crossed
                ],
            },
        )

    return GuardDecision(
        guard_id=GUARD_PLANNER_HEALTH,
        outcome=STOP,
        stop_state=sm.PLANNER_DEGRADED,
        reason=(
            f"planner degradation signal {signal!r}: {observed} exceeds the declared "
            f"threshold of {threshold}. The searcher is broken, which is not the same "
            "finding as a search that is done"
        ),
        detail={
            **base,
            "signal": signal,
            "receipt": receipt,
            "observed": observed,
            "threshold": threshold,
            "crossed": [
                {"signal": s, "observed": o, "threshold": t} for s, o, t in crossed
            ],
        },
        evidence=(receipt, policy.policy_receipt),
    )


def _closure_preconditions(
    *,
    guard_id: str,
    reason: str,
    ledger: ClosureLedger,
    accept_control: AcceptSideControlReceipt,
    planner_decision: GuardDecision,
    health: PlannerHealth,
) -> Optional[GuardDecision]:
    """Everything both closure stops owe, checked once.

    Returns a non-CONTINUE decision when a precondition fails, or None when the
    caller may proceed to build the stop. The three preconditions, each with its
    receipt in the design:

      1. **The accept-side control PROMOTED, within cadence** (§12, AK-D27). The
         other four controls test the gate's ability to REJECT. Without a test of
         its ability to ACCEPT, a quietly dead gate is indistinguishable from an
         exhausted search surface — which is precisely the shape this project
         already shipped once: *"8 of 1,055 trials were of a type the gate could
         promote; 0 of 121 refutations came from futility."*
      2. **Planner health is CLEAN, and clean for THIS snapshot** (§8.10, §8.4.1).
         The decision must be a CONTINUE from `guard_planner_degraded`, and its
         `health_digest` must equal the digest of the health being reported — so a
         clean verdict from an earlier window cannot be paired with a fresh
         closure claim. §8.10 mandates this for `PLATEAU_STOP` only; it is applied
         to `EXHAUSTED_SURFACE` too because the conflation it prevents is
         identical, and one condition should not be guarded in one spelling and
         unguarded in the other.
      3. **The enumeration is real and the reserved words are absent** (§8.10).
    """
    if not isinstance(ledger, ClosureLedger):
        raise GuardInputError("closure: must be a ClosureLedger")
    if not isinstance(accept_control, AcceptSideControlReceipt):
        raise GuardInputError("accept_control: must be an AcceptSideControlReceipt")
    if not isinstance(planner_decision, GuardDecision):
        raise GuardInputError("planner_decision: must be a GuardDecision")
    if not isinstance(health, PlannerHealth):
        raise GuardInputError("health: must be a PlannerHealth")
    _text(reason, "reason")

    if planner_decision.guard_id != GUARD_PLANNER_HEALTH:
        raise GuardInputError(
            f"planner_decision must come from {GUARD_PLANNER_HEALTH!r}, got "
            f"{planner_decision.guard_id!r}; a closure claim needs the planner-health "
            "verdict, not some other guard's"
        )

    if not accept_control.promoted:
        return GuardDecision(
            guard_id=guard_id,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"the accept-side historical-win replay reports {accept_control.status}; "
                "a gate that can still reject but can no longer promote reads exactly "
                "like a closed surface, so closure is not decidable (§12, AK-D27)"
            ),
            detail={"accept_side_control": accept_control.to_dict()},
        )
    if accept_control.cadence.outcome != schemas.PASS:
        return GuardDecision(
            guard_id=guard_id,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                "the accept-side control's cadence check is "
                f"{accept_control.cadence.outcome}: "
                + "; ".join(accept_control.cadence.reasons)
                + " — a promotion outside its declared cadence does not evidence a live "
                "gate now"
            ),
            detail={"accept_side_control": accept_control.to_dict()},
        )

    if planner_decision.outcome != CONTINUE:
        return GuardDecision(
            guard_id=guard_id,
            outcome=REFUSE,
            reason=(
                "planner health is "
                f"{planner_decision.outcome} ({planner_decision.reason}); a closure "
                "claim may not be made while the searcher's health is unproven. "
                "Plateau means the search is done, degraded means the searcher is "
                "broken (§8.10)"
            ),
            detail={
                "planner_decision": planner_decision.to_dict(),
                "closure": ledger.to_detail(),
            },
            directives=(DIRECTIVE_ROOT_CAUSE_ANALYSIS,),
        )
    if planner_decision.detail.get("health_digest") != health.digest:
        return GuardDecision(
            guard_id=guard_id,
            outcome=REFUSE,
            reason=(
                "the clean planner-health verdict was computed over a different health "
                "snapshot than the one being reported "
                f"({planner_decision.detail.get('health_digest')!r} vs "
                f"{health.digest!r}); a stale clearance is not a clearance"
            ),
            detail={
                "planner_decision_digest": planner_decision.detail.get("health_digest"),
                "reported_digest": health.digest,
            },
            directives=(DIRECTIVE_ROOT_CAUSE_ANALYSIS,),
        )

    language = check_closure_language(reason, ledger)
    if language.outcome != schemas.PASS:
        return GuardDecision(
            guard_id=guard_id,
            outcome=REFUSE,
            reason=(
                "the closure statement uses reserved language instead of an "
                "enumeration: " + "; ".join(language.reasons)
            ),
            detail={"closure": ledger.to_detail(), "language_check": language.outcome},
            directives=(DIRECTIVE_ROOT_CAUSE_ANALYSIS,),
        )

    enumeration = sm.check_closure_enumeration(reason, ledger.to_detail())
    if enumeration.outcome != schemas.PASS:
        return GuardDecision(
            guard_id=guard_id,
            outcome=REFUSE,
            reason=(
                "the closure enumeration §8.10 requires is incomplete: "
                + "; ".join(enumeration.reasons)
            ),
            detail={"closure": ledger.to_detail()},
            directives=(DIRECTIVE_ROOT_CAUSE_ANALYSIS,),
        )
    return None


def guard_exhausted_surface(
    *,
    reason: str,
    closure: ClosureLedger,
    accept_control: AcceptSideControlReceipt,
    planner_decision: GuardDecision,
    health: PlannerHealth,
    eligible_layers_remaining: int,
) -> GuardDecision:
    """`EXHAUSTED_SURFACE` — every eligible hierarchy layer measured or falsified.

    *"Emitting it requires enumerating what was closed and what was not"*, and the
    bare reserved words are rejected. `eligible_layers_remaining > 0` is a
    CONTINUE, not a refusal: a surface with layers left is simply not closed, and
    saying so is the whole point of the enumeration.
    """
    _nonneg_int(eligible_layers_remaining, "eligible_layers_remaining")
    blocked = _closure_preconditions(
        guard_id=GUARD_EXHAUSTED, reason=reason, ledger=closure,
        accept_control=accept_control, planner_decision=planner_decision, health=health,
    )
    if blocked is not None:
        return blocked

    if eligible_layers_remaining > 0:
        return GuardDecision(
            guard_id=GUARD_EXHAUSTED,
            outcome=CONTINUE,
            reason=(
                f"{eligible_layers_remaining} eligible hierarchy layer(s) remain "
                "unmeasured and unfalsified; the surface is not closed"
            ),
            detail={
                "closure": closure.to_detail(),
                "eligible_layers_remaining": eligible_layers_remaining,
            },
            evidence=closure.receipts,
        )

    detail = closure.to_detail()
    detail.update({
        "eligible_layers_remaining": 0,
        "accept_side_control": accept_control.to_dict(),
        "planner_health": health.to_dict(),
        "health_digest": health.digest,
    })
    return GuardDecision(
        guard_id=GUARD_EXHAUSTED,
        outcome=STOP,
        stop_state=sm.EXHAUSTED_SURFACE,
        reason=reason,
        detail=detail,
        evidence=closure.receipts + (accept_control.event_id,),
    )


def guard_plateau(
    *,
    reason: str,
    series: Sequence[ReadinessSeriesEntry],
    policy: PlateauPolicy,
    closure: ClosureLedger,
    accept_control: AcceptSideControlReceipt,
    planner_decision: GuardDecision,
    health: PlannerHealth,
) -> GuardDecision:
    """`PLATEAU_STOP` — no meaningful readiness improvement across the window.

    *"Emitting it requires the same enumeration `EXHAUSTED_SURFACE` does"*, so the
    same `ClosureLedger` and the same reserved-word rejection apply, plus the
    §8.10 `planner_health.degraded_ruled_out` receipt — which this guard sets to
    True only because `_closure_preconditions` proved it from a matching-digest
    planner verdict, never because a caller asserted it.

    "Meaningful" is `policy.improvement_floor`, the campaign's DERIVED floor with
    its calibration receipt. The improvement measured is the best readiness in the
    window minus the readiness the window opened at — a non-negative quantity, so
    a declining series plateaus rather than reading as an improvement of the wrong
    sign.

    ROUNDS WITH NO MAGNITUDE. The series admits `ParityObservation`s, which are
    completed rounds that produced no orderable readiness. They are counted as
    rounds — dropping them would let the guard trend a subsequence of its own
    choosing — and they contribute NOTHING to `best`, because there is nothing on
    them to contribute. A parity round therefore cannot raise `best`, so a
    campaign returning parity is a campaign that is not improving, which is
    precisely what a plateau is.

    But "contributes nothing to the subtraction" is not the same as "says
    nothing", and reading it as the second is how this guard stalls. Under a
    NON-INFERIORITY objective parity is the most common HEALTHY outcome, and a
    converged campaign goes all-parity and stays there — so a rule that answers
    COULD_NOT_EVALUATE to an all-parity window answers it forever, on exactly the
    campaign that most needs stopping. Two misreadings sit either side of that
    and they point opposite ways: reading parity as `0.0` STOPS on an invented
    trend, and refusing to read it at all never stops. Neither is taken:

      * ALL ROUNDS AT PARITY — there is no subtraction to do, so none is done and
        the detail carries no improvement. What the window says instead is
        direct: `window_rounds` consecutive rounds measured their protected cells
        and not one produced a detectable effect. That is a `PLATEAU_STOP` on the
        `no_detectable_effect_in_any_round` basis — but ONLY if every round could
        have SEEN the campaign's own target (`reference_gain`, published by the
        release plane and compared through `could_have_detected`). A window too
        coarse to resolve the effect being hunted has not observed its absence,
        and a campaign that declared no target has not said what it would take to
        be found; both answer COULD_NOT_EVALUATE, and both are conditions a
        campaign can FIX, unlike the categorical refusal they replace.
      * A WINDOW THAT OPENS AT PARITY but contains orderable rounds has no
        opening magnitude, so `best - opening` stays undefined and the guard
        still answers COULD_NOT_EVALUATE. That is not a stall: the window slides,
        so this shape is transient by construction — the next rounds either push
        the parity round out of the window or make the window all-parity, and
        both of those have answers.
      * A PARITY ROUND INSIDE A STOPPING WINDOW is checked before the stop is
        taken. Its true readiness is unknown within `±sensitivity_bound`, so it
        could have been the best round in the window by more than the floor
        without anyone seeing it — if it could, the window has not shown the
        absence of improvement and the guard says so instead of stopping. This
        only ever runs on the STOP side: a window that MEASURED an improvement
        above the floor continues on that measurement, and a blind round cannot
        argue with it.
    """
    if not isinstance(policy, PlateauPolicy):
        raise GuardInputError("policy: must be a PlateauPolicy")
    observations = _typed_tuple(series, "series", ReadinessSeriesEntry, non_empty=False)
    rounds = [observation.round_index for observation in observations]
    if rounds != sorted(set(rounds)):
        raise GuardInputError(
            f"series: round_index must be strictly increasing, got {rounds}; an "
            "unordered readiness series makes 'across the window' undefined"
        )

    blocked = _closure_preconditions(
        guard_id=GUARD_PLATEAU, reason=reason, ledger=closure,
        accept_control=accept_control, planner_decision=planner_decision, health=health,
    )
    if blocked is not None:
        return blocked

    if len(observations) < policy.window_rounds:
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"only {len(observations)} confirmation-stratum readiness observation(s) "
                f"are available for a declared {policy.window_rounds}-round window; a "
                "plateau over a partial window is a guess about a trend"
            ),
            detail={
                "observations": len(observations),
                "window_rounds": policy.window_rounds,
                "floor_receipt": policy.floor_receipt,
            },
        )

    window = observations[-policy.window_rounds:]
    orderable = [observation for observation in window
                 if isinstance(observation, ReadinessObservation)]
    at_parity = [observation for observation in window
                 if isinstance(observation, ParityObservation)]
    parity_detail = {
        "parity_rounds": len(at_parity),
        "parity_round_indices": [observation.round_index for observation in at_parity],
    }

    def _stop(*, basis: str, extra: Mapping) -> GuardDecision:
        """Assemble the §8.10 stop. ONE assembly, so the two bases cannot diverge.

        `basis` rides in the detail rather than in `reason`, because `reason` is
        the caller's narrative and §8.10 reserves its vocabulary; an auditor
        needs to know which question the guard answered without parsing prose.
        """
        detail = closure.to_detail()
        detail.update({
            "window_rounds": policy.window_rounds,
            "improvement_floor": policy.improvement_floor,
            "floor_receipt": policy.floor_receipt,
            "orderable_rounds": len(orderable),
            "window": [observation.to_dict() for observation in window],
            "plateau_basis": basis,
            **parity_detail,
        })
        detail.update(extra)
        detail.update({
            "accept_side_control": accept_control.to_dict(),
            # §8.10's mandatory receipt. `degraded_ruled_out` is True because
            # `_closure_preconditions` verified a CONTINUE from
            # guard_planner_degraded over this exact health digest — never
            # because it was asserted.
            "planner_health": {
                **health.to_dict(),
                "degraded_ruled_out": True,
                "ruled_out_by": planner_decision.reason,
                "health_digest": health.digest,
            },
        })
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=STOP,
            stop_state=sm.PLATEAU_STOP,
            reason=reason,
            detail=detail,
            evidence=(
                tuple(o.source_event_id for o in window)
                + closure.receipts
                + (policy.floor_receipt, accept_control.event_id)
            ),
        )

    if not orderable:
        return _all_parity_plateau(
            window=window, policy=policy, parity_detail=parity_detail,
            observations=observations, stop=_stop)

    if not isinstance(window[0], ReadinessObservation):
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"the window opens on round {window[0].round_index}, which produced no "
                "orderable readiness, so 'improvement across the window' has no opening "
                "magnitude to be measured from. Reading that round as zero would "
                "manufacture an improvement equal to whatever the best round happened "
                "to be"
            ),
            detail={
                "observations": len(observations),
                "window_rounds": policy.window_rounds,
                "orderable_rounds": len(orderable),
                "floor_receipt": policy.floor_receipt,
                "window": [observation.to_dict() for observation in window],
                **parity_detail,
            },
        )

    opening = window[0].readiness
    best = max(observation.readiness for observation in orderable)
    improvement = best - opening
    measured_detail = {
        "improvement": improvement,
        "opening_readiness": opening,
        "best_readiness": best,
    }

    if improvement > policy.improvement_floor:
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=CONTINUE,
            reason=(
                f"readiness improved by {improvement} across the window, above the "
                f"derived floor of {policy.improvement_floor}"
            ),
            detail={
                "window_rounds": policy.window_rounds,
                "improvement_floor": policy.improvement_floor,
                "floor_receipt": policy.floor_receipt,
                "orderable_rounds": len(orderable),
                "window": [observation.to_dict() for observation in window],
                **measured_detail,
                **parity_detail,
            },
            evidence=tuple(o.source_event_id for o in window) + (policy.floor_receipt,),
        )

    # The subtraction says "no improvement". Before that becomes a stop, the
    # rounds with no magnitude get their say: one of them could have been the
    # best round in the window by more than the floor and nobody would have seen
    # it. `opening + floor` is the readiness a round would need to reach to
    # contradict the stop; a round that could not have resolved that magnitude
    # has not shown it did not happen.
    blind = [observation for observation in at_parity
             if not observation.could_have_detected(opening + policy.improvement_floor)]
    if blind:
        blindest = max(blind, key=lambda observation: observation.sensitivity_bound)
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"the window measured no improvement above the derived floor of "
                f"{policy.improvement_floor}, but round {blindest.round_index} produced "
                f"no orderable readiness at a sensitivity of +/-"
                f"{blindest.sensitivity_bound} — coarser than the "
                f"{opening + policy.improvement_floor} it would have had to reach to "
                "beat the window's opening by the floor. That round could have been the "
                "best in the window and gone unseen, so this window has not shown the "
                "absence of improvement"
            ),
            detail={
                "observations": len(observations),
                "window_rounds": policy.window_rounds,
                "improvement_floor": policy.improvement_floor,
                "floor_receipt": policy.floor_receipt,
                "orderable_rounds": len(orderable),
                "window": [observation.to_dict() for observation in window],
                "blind_round_indices": [o.round_index for o in blind],
                "magnitude_that_would_contradict": opening + policy.improvement_floor,
                **measured_detail,
                **parity_detail,
            },
        )

    return _stop(basis=PLATEAU_BASIS_MEASURED_IMPROVEMENT, extra=measured_detail)


def _all_parity_plateau(*, window: Sequence["ParityObservation"],
                        policy: "PlateauPolicy", parity_detail: Mapping,
                        observations: Sequence, stop) -> GuardDecision:
    """Every round in the window measured its cells and none produced an effect.

    This is the branch that keeps a converged non-inferiority campaign from
    running forever. There is no opening magnitude and no best, so nothing is
    subtracted and no `improvement` is reported — the evidence is the ABSENCE of
    a detectable effect across `window_rounds` consecutive rounds, which is a
    different fact from a subtraction that came out small, and it is named as
    one (`plateau_basis`).

    An absence is only evidence at a sensitivity fine enough to have seen the
    thing. The campaign's own advisory reference gain is what the search is
    looking for, so `could_have_detected(reference_gain)` on EVERY round is the
    admission test: pass, and the window has observed the absence of the effect
    being hunted; fail, and it has observed nothing. A campaign that declared no
    target has not said what "found it" would mean and gets the same answer —
    and unlike the blanket refusal this replaces, both of those are things a
    campaign can change (declare the reference policy, or buy sensitivity with
    blocks).

    The window is not required to be a `ParityObservation` sequence by type here
    because the caller reached this branch by finding no `ReadinessObservation`
    in it; the guard's own `_typed_tuple` already refused anything that is
    neither.
    """
    detail = {
        "observations": len(observations),
        "window_rounds": policy.window_rounds,
        "improvement_floor": policy.improvement_floor,
        "floor_receipt": policy.floor_receipt,
        "orderable_rounds": 0,
        "window": [observation.to_dict() for observation in window],
        **parity_detail,
    }
    coarsest = max(window, key=lambda observation: observation.sensitivity_bound)
    targets = {observation.reference_gain for observation in window}
    if None in targets or len(targets) != 1:
        declared = sorted(t for t in targets if t is not None)
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"every one of the {len(window)} rounds in the window measured its "
                "protected cells and none produced a detectable effect, but the window "
                f"does not carry one campaign target to judge that against (declared: "
                f"{declared or 'none'}). 'Nothing moved' is only a result against a "
                "magnitude worth ruling out; without one there is nothing to say the "
                "run was sensitive enough to have found"
            ),
            detail={**detail,
                    "coarsest_sensitivity_bound": coarsest.sensitivity_bound,
                    "reference_gains_declared": declared},
        )
    target = targets.pop()
    blind = [observation for observation in window
             if not observation.could_have_detected(target)]
    if blind:
        return GuardDecision(
            guard_id=GUARD_PLATEAU,
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"every one of the {len(window)} rounds in the window is at parity, but "
                f"round {coarsest.round_index} resolved no finer than +/-"
                f"{coarsest.sensitivity_bound} — coarser than the campaign's own target "
                f"of {target}. A run too blind to see the effect it is hunting has not "
                "observed its absence, and reading this window as a plateau would stop "
                "the campaign on a measurement rather than on a result"
            ),
            detail={**detail, "reference_gain": target,
                    "coarsest_sensitivity_bound": coarsest.sensitivity_bound,
                    "blind_round_indices": [o.round_index for o in blind]},
        )
    return stop(
        basis=PLATEAU_BASIS_NO_DETECTABLE_EFFECT,
        extra={
            "reference_gain": target,
            "coarsest_sensitivity_bound": coarsest.sensitivity_bound,
            # Named so that nothing downstream reads the ABSENCE of `improvement`
            # as a serialization accident and helpfully supplies a zero.
            "no_improvement_magnitude_reason": (
                f"all {len(window)} rounds in the window are at parity, so there is no "
                "opening magnitude and no best magnitude to subtract. Every round could "
                f"have resolved the campaign's target of {target} (coarsest sensitivity "
                f"+/-{coarsest.sensitivity_bound}) and none did: the plateau is the "
                "absence of a detectable effect, not an improvement measured at zero"),
        })


def guard_command_retries(ledger: CommandRetryLedger) -> GuardDecision:
    """Retries bounded at 3, then ROOT-CAUSE ANALYSIS — not a fourth attempt.

    `OPERATING_CONSTRAINTS.md:44-46`, carried into §7.1 as
    `stop_policy.max_command_retries` with a schema maximum of 3. Exceeding it is
    not a §8.10 stop on its own: the loop stops retrying THIS command and
    diagnoses it. What escalates the pattern to `PLANNER_DEGRADED` is the
    *"looping against an unavailable dependency"* family, counted in
    `PlannerHealth` — so the condition keeps one spelling and one guard.
    """
    if not isinstance(ledger, CommandRetryLedger):
        raise GuardInputError("ledger: must be a CommandRetryLedger")

    detail = {
        "command_id": ledger.command_id,
        "attempts": ledger.attempts,
        "retries_used": ledger.retries_used,
        "max_retries": ledger.max_retries,
        "last_error": ledger.last_error,
    }
    if not ledger.exhausted:
        return GuardDecision(
            guard_id=GUARD_COMMAND_RETRY,
            outcome=CONTINUE,
            reason=(
                f"{ledger.command_id}: {ledger.retries_used} of {ledger.max_retries} "
                "retries used"
            ),
            detail=detail,
            evidence=(ledger.receipt,),
        )
    return GuardDecision(
        guard_id=GUARD_COMMAND_RETRY,
        outcome=REFUSE,
        reason=(
            f"{ledger.command_id}: {ledger.retries_used} retries used of a bound of "
            f"{ledger.max_retries}; the next step is root-cause analysis, not another "
            f"attempt. Last error: {ledger.last_error}"
        ),
        detail=detail,
        directives=(DIRECTIVE_ROOT_CAUSE_ANALYSIS,),
        evidence=(ledger.receipt,),
    )


def guard_repair_cap(ledger: RepairLedger) -> GuardDecision:
    """§8.5.1 repair and build-failure caps — a signal, never another retry.

    *"Repairs are capped per proposal; exceeding the cap is a `PLANNER_DEGRADED`
    signal, not another retry. AutoPilot's scar here was a loop compounding edits
    onto an already-corrupted file."* So this guard REFUSES the repair and names
    the signal; the stop belongs to `guard_planner_degraded`, which counts the
    exceedances across proposals. Emitting the stop from here as well would give
    one condition two spellings, and the guard that fires would be whichever
    ledger the caller happened to consult.
    """
    if not isinstance(ledger, RepairLedger):
        raise GuardInputError("ledger: must be a RepairLedger")

    detail = {
        "proposal_id": ledger.proposal_id,
        "repairs_attempted": ledger.repairs_attempted,
        "max_repairs": ledger.max_repairs,
        "consecutive_build_failures": ledger.consecutive_build_failures,
        "max_consecutive_build_failures": ledger.max_consecutive_build_failures,
    }
    reasons: list = []
    signals: list = []
    if ledger.repairs_attempted >= ledger.max_repairs:
        reasons.append(
            f"{ledger.repairs_attempted} repair(s) attempted against a cap of "
            f"{ledger.max_repairs}"
        )
        signals.append("repair_cap_exceeded")
    if ledger.consecutive_build_failures > ledger.max_consecutive_build_failures:
        reasons.append(
            f"{ledger.consecutive_build_failures} consecutive build failure(s) exceed "
            f"the declared {ledger.max_consecutive_build_failures}"
        )
        signals.append("consecutive_build_failures")

    if not reasons:
        return GuardDecision(
            guard_id=GUARD_REPAIR_CAP,
            outcome=CONTINUE,
            reason=(
                f"{ledger.proposal_id}: repairs and build failures are within their "
                "declared caps"
            ),
            detail=detail,
            evidence=(ledger.receipt,),
        )
    return GuardDecision(
        guard_id=GUARD_REPAIR_CAP,
        outcome=REFUSE,
        reason=(
            f"{ledger.proposal_id}: " + "; ".join(reasons)
            + ". A further repair would compound edits onto an already-broken tree; "
            "this is a PLANNER_DEGRADED signal, not another retry (§8.5.1)"
        ),
        detail={**detail, "planner_degraded_signals": signals},
        directives=(DIRECTIVE_REPAIR_FORBIDDEN, DIRECTIVE_ROOT_CAUSE_ANALYSIS),
        evidence=(ledger.receipt,),
    )


# =============================================================================
# Disposition — the controller reduces many guard verdicts to one
# =============================================================================

@dataclass(frozen=True)
class GuardDisposition:
    """The single governing verdict over a round's guard decisions."""

    outcome: str
    reason: str
    governing: Optional[GuardDecision] = None
    stop_state: Optional[str] = None
    decisions: tuple = ()
    stops: tuple = ()
    refusals: tuple = ()
    unevaluable: tuple = ()

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise GuardInputError(f"outcome: {self.outcome!r} not in {list(OUTCOMES)}")
        _text(self.reason, "reason")

    @property
    def clears(self) -> bool:
        return self.outcome == CONTINUE

    @property
    def directives(self) -> tuple:
        """Every directive any decision emitted, deduplicated in decision order."""
        out: list = []
        for decision in self.decisions:
            out.extend(decision.directives)
        return tuple(dict.fromkeys(out))

    @property
    def decision_packages(self) -> tuple:
        return tuple(
            decision.decision_package for decision in self.decisions
            if decision.decision_package is not None
        )


def dispose(decisions: Sequence[GuardDecision]) -> GuardDisposition:
    """Reduce a round's guard decisions to one governing verdict.

    Deterministic and total: STOPs are ordered by `STOP_PRECEDENCE` (ties broken
    by supplied order, so the same input always yields the same governing
    decision); with no STOP, an unevaluable guard governs over a refusal, and a
    refusal governs over a continue. `COULD_NOT_EVALUATE` never reads as CONTINUE
    — an inability to evaluate a stop condition is not evidence the condition is
    absent, and treating it as one is the fail-open shape every check in this
    package is written against.
    """
    entries = _typed_tuple(decisions, "decisions", GuardDecision, non_empty=False)

    # A round in which NOTHING was checked is not a round in which no stop
    # condition holds. Reducing the empty set to CONTINUE — `clears is True` —
    # makes the whole guard plane passable by deleting the thing it inspects: a
    # caller whose guard collection raised, short-circuited, or was never wired
    # gets the same verdict as a caller whose fifteen guards all cleared.
    if not entries:
        return GuardDisposition(
            outcome=COULD_NOT_EVALUATE,
            reason=(
                "no guard decision was supplied, so no stop condition was evaluated; "
                "an unchecked round is COULD_NOT_EVALUATE and never CONTINUE"
            ),
        )

    stops = tuple(d for d in entries if d.outcome == STOP)
    refusals = tuple(d for d in entries if d.outcome == REFUSE)
    unevaluable = tuple(d for d in entries if d.outcome == COULD_NOT_EVALUATE)

    if stops:
        ranked = sorted(
            enumerate(stops), key=lambda pair: (
                STOP_PRECEDENCE.index(pair[1].stop_state), pair[0]
            )
        )
        governing = ranked[0][1]
        return GuardDisposition(
            outcome=STOP,
            reason=(
                f"{governing.stop_state} governs "
                f"({len(stops)} stop condition(s) held): {governing.reason}"
            ),
            governing=governing,
            stop_state=governing.stop_state,
            decisions=entries,
            stops=stops,
            refusals=refusals,
            unevaluable=unevaluable,
        )
    if unevaluable:
        governing = unevaluable[0]
        return GuardDisposition(
            outcome=COULD_NOT_EVALUATE,
            reason=(
                f"{len(unevaluable)} guard(s) could not evaluate their stop condition; "
                f"{governing.guard_id}: {governing.reason}"
            ),
            governing=governing,
            decisions=entries,
            refusals=refusals,
            unevaluable=unevaluable,
        )
    if refusals:
        governing = refusals[0]
        return GuardDisposition(
            outcome=REFUSE,
            reason=(
                f"no stop condition holds, but {len(refusals)} action(s) are refused; "
                f"{governing.guard_id}: {governing.reason}"
            ),
            governing=governing,
            decisions=entries,
            refusals=refusals,
        )
    return GuardDisposition(
        outcome=CONTINUE,
        reason=f"{len(entries)} guard(s) reported no stop condition and no refusal",
        decisions=entries,
    )


def dispose_requested_stop(
    request: sm.StopRequest, decisions: Sequence[GuardDecision]
) -> GuardDecision:
    """§8.10's last sentence, implemented: the LLM requests, the controller disposes.

    A request is honoured ONLY when a guard independently reached the same stop
    state from records, and what is returned is then the GUARD's decision — its
    reason, its detail, its receipts. The request's own narrative is never
    promoted into the record, because a stop that reads as evidenced but whose
    evidence is the requester's prose is worse than an unevidenced one.

    `request.origin` is copied into the refusal's detail and consulted by NOTHING:
    AK-D38 grades an operator hypothesis at `design_prior` *"and never above it"*,
    and the same principle applies here — the input most likely to be waved
    through is the one whose author is trusted.
    """
    if not isinstance(request, sm.StopRequest):
        raise GuardInputError("request: must be a state_machine.StopRequest")
    entries = _typed_tuple(decisions, "decisions", GuardDecision, non_empty=False)

    for decision in entries:
        if decision.outcome == STOP and decision.stop_state == request.state:
            return decision

    reached = sorted({d.stop_state for d in entries if d.outcome == STOP})
    return GuardDecision(
        guard_id=GUARD_STOP_REQUEST,
        outcome=REFUSE,
        reason=(
            f"stop {request.state!r} was REQUESTED but no guard reached it from "
            f"records; guards that did stop: {reached or 'none'}. A stop request is "
            "not evidence (§8.10: the controller owns disposition from records)"
        ),
        detail={
            "requested_state": request.state,
            "requested_reason": request.reason,
            "request_origin": request.origin,
            "origin_is_not_evidence": True,
            "stops_reached_by_guards": reached,
        },
        directives=(DIRECTIVE_REQUEST_DENIED,),
    )


# =============================================================================
# Structural audits — properties proven from the module, not asserted in prose
# =============================================================================

#: Calls that write, delete, execute, signal, WAIT, or read a clock. The check is
#: blunt on purpose — it does not prove the receiver's type, so this module simply
#: does not use these names on anything.
_FORBIDDEN_CALL_ATTRS = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "flush", "fsync",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "rmtree", "rename", "chmod",
    "symlink", "link", "touch", "move", "copy", "copyfile", "copytree",
    "system", "popen", "Popen", "spawnv", "fork", "kill", "killpg", "send_signal",
    "terminate", "check_call", "check_output", "communicate",
    # A busy-wait, and the clocks that would make one possible. §8.10:
    # RESOURCE_UNAVAILABLE persists and drains, never busy-waits — and every guard
    # here is a PURE function whose `now` is an argument, so reading a clock would
    # make two runs over one journal disagree.
    "sleep", "wait", "poll", "select", "now", "utcnow", "today", "monotonic",
    "perf_counter", "time_ns",
})

_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input"})

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "threading", "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl",
    "shlex", "asyncio", "time", "random", "secrets",
})


def audit_no_write_process_or_wait_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that a guard cannot act, wait, or drift.

    Three properties in one pass, because all three are invisible in prose and a
    code review enforces them once:

      * **no write, no process** — a guard is a pure function over journaled state;
      * **no wait** — §8.10's *"persist and drain, never busy-wait"* is a property
        of the code, not of the docstring that claims it;
      * **no clock, no randomness** — `now` is an argument, so two dispositions
        over one journal are the same disposition. `datetime` is imported for
        PARSING supplied stamps; `.now()`/`.utcnow()`/`.today()` are refused, so
        the import cannot become an ambient clock.

    COULD_NOT_CHECK when the source cannot be read or parsed — an unreadable
    module is not an audited one.
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
        return schemas.Check(schemas.FAIL, tuple(findings))
    # An empty, truncated or comment-only source parses cleanly, walks to nothing,
    # and reports PASS — the audit passing by having had its subject deleted. A
    # module with no definition in it is not an audited module. This is checked
    # only AFTER the findings, so `"import os\n"` still FAILs rather than being
    # excused for defining no function.
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    ):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the source audited defines no function or class; an empty or truncated "
            "module has nothing to audit, and its silence is not a clean bill",
        ))
    return schemas.Check(schemas.PASS)


def audit_stop_coverage_totality() -> schemas.Check:
    """Every §8.10 stop is either guard-decided here or owned elsewhere, by name.

    `_assert_vocabulary_total()` already raises at import on drift; this is the
    reportable form, so a caller can render the ownership table instead of
    discovering it by crashing.
    """
    declared = set(sm.STOP_STATES)
    guarded = set(STOP_PRECEDENCE)
    elsewhere = set(NON_GUARD_STOPS)
    reasons: list = []
    for state in sorted(declared - guarded - elsewhere):
        reasons.append(f"{state}: no guard decides it and no other plane claims it")
    for state in sorted((guarded | elsewhere) - declared):
        reasons.append(f"{state}: decided here but not declared by §8.10")
    for state in sorted(guarded & elsewhere):
        reasons.append(f"{state}: claimed by a guard AND by another plane")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def audit_directive_vocabulary() -> schemas.Check:
    """No directive can express waiting, polling, sleeping, spinning, or retrying.

    §8.10's `RESOURCE_UNAVAILABLE` clause is *"persist and drain, never
    busy-wait"*. A rule that lives only in prose is re-litigated by whoever adds
    the next directive; this makes the addition fail instead.
    """
    if not DIRECTIVES:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the directive vocabulary is empty; a vocabulary with nothing in it "
            "satisfies every content check by having no content, and no REFUSE "
            "could be expressed at all",
        ))
    if not FORBIDDEN_DIRECTIVE_TOKENS:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "FORBIDDEN_DIRECTIVE_TOKENS is empty; the audit would pass every "
            "possible vocabulary, including one that spells WAIT",
        ))
    findings: list = []
    for directive in sorted(DIRECTIVES):
        for token in FORBIDDEN_DIRECTIVE_TOKENS:
            if token in directive:
                findings.append(
                    f"{directive}: contains {token!r}; §8.10 forbids busy-waiting, so "
                    "no directive may name it"
                )
    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS)
