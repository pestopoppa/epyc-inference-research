"""Typed trial lifecycle (S3-AKU-01): one transition table, one persistence guard.

Two inventories live here, deliberately separate:

* `LifecycleState` + `TRANSITIONS` -- the minimal HANDOFF section 8.12 state set
  (starting, recovering, running, waiting_resource, waiting_actor, paused, draining,
  drained, validation_required, failed, complete). `TRANSITIONS` is the ONLY source of
  legality; `guard_transition` and `TERMINAL_STATES` are derived from it.
* `OutcomeStatus` -- the closed inventory of every literal string `loop.Outcome.status`
  takes. It is a `str` Enum, so every existing `outcome.status == "kept"` comparison,
  set membership test and JSON dump stays valid. It does not change any status value.

Pure Python; no I/O, no process or journal authority.
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class LifecycleState(str, Enum):
    STARTING = "starting"
    RECOVERING = "recovering"
    RUNNING = "running"
    WAITING_RESOURCE = "waiting_resource"
    WAITING_ACTOR = "waiting_actor"
    PAUSED = "paused"
    DRAINING = "draining"
    DRAINED = "drained"
    VALIDATION_REQUIRED = "validation_required"
    FAILED = "failed"
    COMPLETE = "complete"


S = LifecycleState

#: Legal successors. Pause closes admissions and keeps the service (-> PAUSED; resume goes
#: through RECOVERING, which reconciles identities and reacquires claims). Drain closes
#: admissions and stops workers (-> DRAINING -> DRAINED; resume likewise via RECOVERING).
#: FAILED and COMPLETE have no successors: terminal states are absorbing.
TRANSITIONS: Mapping[LifecycleState, frozenset[LifecycleState]] = {
    S.STARTING: frozenset({S.RECOVERING, S.RUNNING, S.WAITING_RESOURCE, S.WAITING_ACTOR,
                           S.PAUSED, S.DRAINING, S.FAILED}),
    S.RECOVERING: frozenset({S.RUNNING, S.WAITING_RESOURCE, S.WAITING_ACTOR, S.PAUSED,
                             S.DRAINING, S.FAILED}),
    S.RUNNING: frozenset({S.WAITING_RESOURCE, S.WAITING_ACTOR, S.PAUSED, S.DRAINING,
                          S.VALIDATION_REQUIRED, S.FAILED, S.COMPLETE}),
    S.WAITING_RESOURCE: frozenset({S.RUNNING, S.WAITING_ACTOR, S.PAUSED, S.DRAINING,
                                   S.FAILED}),
    S.WAITING_ACTOR: frozenset({S.RUNNING, S.WAITING_RESOURCE, S.PAUSED, S.DRAINING,
                                S.FAILED}),
    S.PAUSED: frozenset({S.RECOVERING, S.DRAINING, S.FAILED}),
    S.DRAINING: frozenset({S.DRAINED, S.FAILED}),
    S.DRAINED: frozenset({S.RECOVERING, S.FAILED}),
    S.VALIDATION_REQUIRED: frozenset({S.RUNNING, S.PAUSED, S.DRAINING, S.FAILED,
                                      S.COMPLETE}),
    S.FAILED: frozenset(),
    S.COMPLETE: frozenset(),
}

#: Derived, never hand-listed: a state with no successors.
TERMINAL_STATES: frozenset[LifecycleState] = frozenset(
    state for state, successors in TRANSITIONS.items() if not successors)


class TransitionRefused(RuntimeError):
    pass


class Decision(str, Enum):
    APPLY = "apply"
    NOOP = "noop"
    REFUSE = "refuse"


def _state(value) -> LifecycleState:
    try:
        return LifecycleState(value)
    except ValueError as exc:
        raise TransitionRefused(f"unknown lifecycle state {value!r}") from exc


def guard_transition(current, target) -> Decision:
    """Persistence guard derived from `TRANSITIONS` alone.

    * target in TRANSITIONS[current]            -> APPLY
    * same non-terminal state (idempotent write) -> NOOP
    * current terminal and target terminal       -> NOOP (duplicate / racing completion
      cannot rewrite the recorded terminal, including a second DIFFERENT terminal)
    * anything else                              -> REFUSE
    """
    current, target = _state(current), _state(target)
    if target in TRANSITIONS[current]:
        return Decision.APPLY
    if current == target:
        return Decision.NOOP
    if current in TERMINAL_STATES and target in TERMINAL_STATES:
        return Decision.NOOP
    return Decision.REFUSE


def apply_transition(current, target) -> LifecycleState:
    """Return the state to persist. NOOP keeps `current`; REFUSE raises."""
    decision = guard_transition(current, target)
    if decision is Decision.REFUSE:
        raise TransitionRefused(f"illegal lifecycle transition {_state(current).value} "
                                f"-> {_state(target).value}")
    return _state(target) if decision is Decision.APPLY else _state(current)


class OutcomeStatus(str, Enum):
    """Every literal `Outcome.status` takes (loop/loop.py, loop/pipeline.py, loop/run.py)."""
    # measured
    KEPT = "kept"
    KEEP_CANDIDATE = "keep_candidate"
    MEASURED_NULL = "measured_null"
    REGRESSION = "regression"
    INTERACTION_REGRESSION = "interaction_regression"
    RUNTIME_OBSERVED = "runtime_observed"
    # measurement problems
    MEASUREMENT_INVALID = "measurement_invalid"
    BENCH_FAILED = "bench_failed"
    NO_OP_BUILD = "no_op_build"
    INTEGRITY_REFUSED = "integrity_refused"
    RUNTIME_REFUSED = "runtime_refused"
    # refusals before measurement
    ABSTAINED = "abstained"
    REFUSED_AT_FORMATION = "refused_at_formation"
    REFUSED_DUPLICATE = "refused_duplicate"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    PATCH_REJECTED = "patch_rejected"
    GATE_REFUSED = "gate_refused"
    RESUME_REJECTED = "resume_rejected"
    # authoring
    PATCH_ROUNDS_EXHAUSTED = "patch_rounds_exhausted"
    SCOPE_BLOCKED = "scope_blocked"
    AUTHORING_FAILED = "authoring_failed"
    AUTHORING_HARNESS_FAILURE = "authoring_harness_failure"
    HYPOTHESIS_RETIRED = "hypothesis_retired"
    # control / harness
    SUPERSEDED = "superseded"
    #: AKPL-P3: formed and handed to the rowflow row chain (measured there, not in-process).
    HANDED_OFF = "handed_off"
    STOPPED_MID_FORMATION = "stopped_mid_formation"
    STOPPED_BEFORE_RESCHEDULE = "stopped_before_reschedule"
    PLANNER_TRANSIENT = "planner_transient"
    PLANNER_SAME_STATE_SKIP = "planner_same_state_skip"
    BELIEF_REFUSED = "belief_refused"
    LANE_ERROR = "lane_error"
    RUN_ABORTED = "run_aborted"
    # whole-source validation (run.py: "source_validation_" + disposition)
    SOURCE_VALIDATION_PASSED = "source_validation_passed"
    SOURCE_VALIDATION_FAILED = "source_validation_failed"
    SOURCE_VALIDATION_PENDING = "source_validation_pending"


#: Statuses a later draw may resume (mirrors `loop.RESUMABLE_STATUSES`; asserted equal in tests).
RESUMABLE_OUTCOME_STATUSES = frozenset({OutcomeStatus.STOPPED_MID_FORMATION,
                                        OutcomeStatus.PLANNER_TRANSIENT})

OUTCOME_STATUS_VALUES: frozenset[str] = frozenset(item.value for item in OutcomeStatus)


def outcome_status(value) -> OutcomeStatus:
    """Closed lookup; an unknown status string is a refusal, not a silent pass."""
    try:
        return OutcomeStatus(value)
    except ValueError as exc:
        raise TransitionRefused(f"unknown Outcome.status {value!r}") from exc


def guard_outcome_persist(recorded, new) -> Decision:
    """Completion-persistence guard for one iteration outcome.

    An iteration's outcome is terminal once recorded: the first write is APPLY (recorded is
    None), and any later write -- the same status again or a racing different one -- is a
    NOOP that must not overwrite the first.
    """
    outcome_status(new)
    if recorded is None:
        return Decision.APPLY
    outcome_status(recorded)
    return Decision.NOOP


__all__ = ["Decision", "LifecycleState", "OUTCOME_STATUS_VALUES", "OutcomeStatus",
           "RESUMABLE_OUTCOME_STATUSES", "TERMINAL_STATES", "TRANSITIONS",
           "TransitionRefused", "apply_transition", "guard_outcome_persist",
           "guard_transition", "outcome_status"]
