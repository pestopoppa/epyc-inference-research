"""Provider circuit breaker: park a run whose actor provider keeps failing.

Measured (campaign store `store-b0ba1d427`): 171 consecutive iterations ended
`planner_transient` because the actor provider failed terminally ("failed 4 times"),
and the run kept dispatching. `loop.iterate` deliberately ends only the ITERATION on
an `ActorTransient`; nothing acted on the streak. This breaker does.

Classification of one recorded iteration outcome:
  * PROVIDER  -- `planner_transient`, or an authoring outcome whose every author
    member failed with outcome `transient` (the author's provider, not the science).
    Increments the streak.
  * NEUTRAL   -- `stopped_mid_formation` / `lane_error`: no provider answer is
    implied either way. Leaves the streak unchanged.
  * anything else (abstain, refusals, gate fails, measured results, bench failures,
    ...) required at least one provider call to SUCCEED, so it RESETS the streak.
    Science outcomes therefore never count toward the streak, and they reset it.

When the streak reaches the threshold the breaker trips. `run.py` folds `tripped`
into its `should_stop`, so the pool's budget declines further draws at the next
iteration boundary exactly as for the STOP file; nothing is discarded and custody of
every produced artifact follows the existing stop path. The run then ends with the
continuation terminal `provider_parked` -- distinct from an operator `stopped`, and
resumable with `--resume-run` once the provider is healthy.
"""
from __future__ import annotations

import threading
from typing import Any, Mapping

PROVIDER_PARKED = "provider_parked"
#: Continuation terminals that end a batch short of its budget. The serial supervisor
#: stops scheduling on either; only `stopped` refuses a `--resume-run`.
#: `same_state_stop.SAME_STATE_PARKED`: the planner abstained on an unchanged state.
SAME_STATE_PARKED = "planner_same_state_parked"
#: The granted window reached its cleanup reserve (`deadline_reserve`).
DEADLINE_PARKED = "deadline_parked"
STOP_TERMINALS = frozenset({"stopped", PROVIDER_PARKED, SAME_STATE_PARKED, DEADLINE_PARKED})
TERMINALS = frozenset({"complete"}) | STOP_TERMINALS

DEFAULT_PARK_AFTER = 8

_AUTHORING = frozenset({"authoring_harness_failure", "authoring_failed",
                        "hypothesis_retired"})
NEUTRAL_STATUSES = frozenset({"stopped_mid_formation", "lane_error"})


def validate_threshold(value: Any, label: str = "provider_park_after") -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def classify(outcome: Any) -> str:
    """'provider' | 'neutral' | 'success' for one recorded iteration outcome."""
    status = getattr(outcome, "status", None)
    if status == "planner_transient":
        return "provider"
    if status in NEUTRAL_STATUSES:
        return "neutral"
    if status in _AUTHORING:
        pending = getattr(outcome, "hypothesis_pending", None)
        members = (pending.get("authoring_failures")
                   if isinstance(pending, Mapping) else None) or []
        if members and all(isinstance(m, Mapping) and m.get("outcome") == "transient"
                           for m in members):
            return "provider"
    return "success"


class ProviderBreaker:
    def __init__(self, threshold: int = DEFAULT_PARK_AFTER) -> None:
        self.threshold = validate_threshold(threshold)
        self.streak = 0
        self.tripped = False
        self.last_reason: str | None = None
        self._lock = threading.Lock()

    def observe(self, outcome: Any) -> bool:
        """Fold one recorded outcome; True exactly when this observation trips."""
        kind = classify(outcome)
        with self._lock:
            if kind == "provider":
                self.streak += 1
                reasons = getattr(outcome, "reasons", None) or []
                self.last_reason = str(reasons[0])[:500] if reasons else None
            elif kind == "success":
                self.streak = 0
            if not self.tripped and self.streak >= self.threshold:
                self.tripped = True
                return True
            return False

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "streak": self.streak,
                "tripped": self.tripped, "last_reason": self.last_reason}


def terminal(*, stop_asked: bool, breaker: ProviderBreaker | None,
             same_state: Any = None, deadline: Any = None) -> str:
    """The continuation terminal: an operator stop wins over any park."""
    if stop_asked:
        return "stopped"
    if breaker is not None and breaker.tripped:
        return PROVIDER_PARKED
    if same_state is not None and getattr(same_state, "tripped", False):
        return SAME_STATE_PARKED
    if deadline is not None and getattr(deadline, "tripped", False):
        return DEADLINE_PARKED
    return "complete"


__all__ = ["DEFAULT_PARK_AFTER", "NEUTRAL_STATUSES", "PROVIDER_PARKED", "ProviderBreaker",
           "SAME_STATE_PARKED",
           "STOP_TERMINALS", "TERMINALS", "classify", "terminal", "validate_threshold"]
