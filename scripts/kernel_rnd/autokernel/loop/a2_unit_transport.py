"""Opt-in pure glue (AKU-04f): selected-unit launch bracketed by v2 permit records.

    INTENT + PERMIT_ISSUED  ->  launcher(permit)  ->  LAUNCHED  ->  TERMINAL
                                     \\-> AcquisitionFailed -> durable NOT_ACQUIRED

`append` is any durable sink taking one v2 record dict (production: a bound
`CampaignController.append_a2_unit_record`); a raising sink aborts BEFORE the launch.
Prior history is the replayed record list: UNKNOWN (intent/permit/launched without a durable
NOT_ACQUIRED, or a terminal) never authorizes a launch. Nothing here discovers or schedules
work, and it is not called from any launch site until a caller opts in.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from . import a2_unit_permit as P


class AcquisitionFailed(RuntimeError):
    """Raised by a launcher that provably did not acquire the unit/resource."""


def launch_selected_unit(*, plan, unit_range: Mapping[str, Any], unit_index: int,
                         execution_id: str, frame_digest: str,
                         producer_identity: Mapping[str, Any], owner_incarnation: int,
                         prior_records: Sequence[Mapping[str, Any]],
                         append: Callable[[dict], Any],
                         launcher: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """Run one launch. `launcher(permit_id)` returns {"status": valid|invalid|diagnostic,
    "reason": str|None, "result_digest": sha|None}. Returns the final unit state record
    summary {"phase", "attempt", "records"}."""
    prior = list(prior_records)
    attempt, prev = 1, None
    if prior:
        state = P.project_unit(prior)
        if not P.may_relaunch(state):
            raise P.A2UnitRefused(
                f"unit is {state.phase}; replayed unknown/terminal state never authorizes "
                "a relaunch")
        attempt, prev = state.attempt + 1, prior[-1]["record_digest"]
    written: list[dict] = []

    def emit(record: dict) -> dict:
        append(record)   # a raising sink aborts before any launch or later record
        written.append(record)
        return record

    intent = emit(P.make_intent(plan=plan, unit_range=unit_range, unit_index=unit_index,
                                execution_id=execution_id, frame_digest=frame_digest,
                                producer_identity=producer_identity,
                                owner_incarnation=owner_incarnation, attempt=attempt,
                                prev_digest=prev))
    issued = emit(P.make_follow("PERMIT_ISSUED", intent, owner_incarnation=owner_incarnation,
                                prev_digest=intent["record_digest"]))
    state = P.project_unit([*prior, *written])
    permit = issued["payload"]["permit_id"]
    if not P.authorize_launch(state, current_owner=owner_incarnation, permit=permit):
        raise P.A2UnitRefused("permit does not authorize this launch")
    try:
        result = launcher(permit)
    except AcquisitionFailed as exc:
        emit(P.make_follow("NOT_ACQUIRED", intent, owner_incarnation=owner_incarnation,
                           prev_digest=written[-1]["record_digest"],
                           payload={"reason": str(exc) or "acquisition failed"}))
        return _summary([*prior, *written])
    launched = emit(P.make_follow("LAUNCHED", intent, owner_incarnation=owner_incarnation,
                                  prev_digest=written[-1]["record_digest"]))
    emit(P.make_follow("TERMINAL", intent, owner_incarnation=owner_incarnation,
                       prev_digest=launched["record_digest"],
                       payload={"status": result["status"], "reason": result.get("reason"),
                                "result_digest": result.get("result_digest")}))
    return _summary([*prior, *written])


def _summary(records: list[dict]) -> dict[str, Any]:
    state = P.project_unit(records)
    return {"phase": state.phase, "attempt": state.attempt, "records": records}


__all__ = ["AcquisitionFailed", "launch_selected_unit"]
