"""AKU-04f glue: one selected-unit launch bracketed by v2 permit records.

    INTENT + PERMIT_ISSUED  ->  launch(permit)  ->  LAUNCHED  ->  TERMINAL
                                     \\-> provable non-acquisition -> durable NOT_ACQUIRED

`append` is a durable sink taking one v2 record dict (production: the bound
`CampaignController.append_a2_unit_record`); a raising sink aborts BEFORE the launch.
Prior history is the replayed record list: UNKNOWN (intent/permit/launched without a durable
NOT_ACQUIRED, or a terminal) never authorizes a launch. The installed consumer is
`CampaignController.run_worker_stage` for a Prepared v4 selected-unit invocation: identity is
derived only from the real Prepared objects and the controller's owner fields
(`selected_launch_identity`) and the permit stays inside the controller's reservation.
Nothing here discovers or schedules work.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

from . import a2_execution_state as AX
from . import a2_unit_permit as P
from . import experiment_plan as ep

LOGICAL_ID = "aku-04f-selected-unit"
_OWNER_FIELDS = ("supervisor_id", "supervisor_incarnation")  # change per owner, not per frame


class AcquisitionFailed(RuntimeError):
    """Raised by a launcher that provably did not acquire the unit/resource."""


def selected_prepared(planned_invocation: Any):
    """The Prepared v4 of a concrete planned invocation with a selected dispatch, else None."""
    from .unified_worker import PlannedWorkerInvocation
    if type(planned_invocation) is not PlannedWorkerInvocation:
        return None
    prepared = planned_invocation.prepared
    return prepared if prepared.selected_dispatch is not None else None


def selected_launch_identity(prepared, *, campaign_id: str, config_generation: int,
                             config_digest: str) -> dict[str, Any]:
    """Launch identity of the ONE selected unit, from the real Prepared v4 objects only.

    Membership is rederived from the unchanged full plan (no one-unit rehash); the frame is
    the execution frame minus transport attempt, artifact location and owner incarnation,
    so a new attempt id, root or restart cannot open a fresh chain for the same unit."""
    dispatch = prepared.selected_dispatch
    if dispatch is None:
        raise P.A2UnitRefused("launch is not a selected-unit transport")
    plan = ep.ExperimentPlan.from_dict(prepared.plan.to_dict())
    base = prepared.body()["capture_context_base"]
    if (plan.digest != dispatch.plan_digest or plan.campaign_id != campaign_id
            or (base["campaign_id"], base["config_generation"], base["config_digest"])
            != (campaign_id, config_generation, config_digest)):
        raise P.A2UnitRefused("selected Prepared is not this controller's campaign/config")
    units = dispatch.selection.units(plan)
    if len(units) != 1:
        raise P.A2UnitRefused("selected launch must carry exactly one original unit")
    unit = units[0]
    frame = P._digest({
        "schema": prepared.schema, "plan_digest": plan.digest,
        "prompt_manifest_digest": prepared.prompts.digest,
        "runtime_pair": prepared.runtime_pair.to_dict(),
        "capture_context": {k: v for k, v in base.items() if k not in _OWNER_FIELDS},
        "max_stage_seconds": prepared.max_stage_seconds,
        "teardown_seconds": prepared.teardown_seconds})
    recipe = getattr(prepared.runtime_pair, unit.arm)
    return {"plan": plan, "unit_index": unit.order_index, "frame_digest": frame,
            "unit_range": P.select_unit_range(plan, dispatch.selection.start_order,
                                              dispatch.selection.stop_order - 1),
            "execution_id": AX.execution_identity(
                campaign_id=campaign_id, config_generation=config_generation,
                config_digest=config_digest, logical_id=LOGICAL_ID,
                plan_digest=plan.digest, frame_digest=frame),
            "producer_identity": {"frame_digest": frame,
                                  "recipe_snapshot_digest": recipe.snapshot_digest,
                                  "recipe_execution_digest": recipe.execution_digest}}


def identity_key(identity: Mapping[str, Any]) -> str:
    """`a2_unit_permit.unit_key` of the chain this launch identity belongs to."""
    plan = identity["plan"]
    unit = next(u for u in plan.expected_units if u.order_index == identity["unit_index"])
    return P.unit_key({**identity, "plan_digest": plan.digest, "unit": unit.to_dict()})


@dataclasses.dataclass
class Reservation:
    """Held by the launching owner only; the permit never leaves this object."""
    intent: dict
    permit: str
    owner: int
    records: list


def reserve(identity: Mapping[str, Any], *, owner_incarnation: int,
            prior_records: Sequence[Mapping[str, Any]],
            append: Callable[[dict], Any]) -> Reservation:
    """Write the fsynced INTENT and first-issue PERMIT; refuse replayed unknown/terminal."""
    prior = list(prior_records)
    attempt, prev = 1, None
    if prior:
        state = P.project_unit(prior)
        if not P.may_relaunch(state):
            raise P.A2UnitRefused(
                f"unit is {state.phase}; replayed unknown/terminal state never authorizes "
                "a relaunch")
        attempt, prev = state.attempt + 1, prior[-1]["record_digest"]
    held = Reservation({}, "", owner_incarnation, prior)
    intent = _emit(held, append, P.make_intent(
        **identity, owner_incarnation=owner_incarnation, attempt=attempt, prev_digest=prev))
    issued = _emit(held, append, P.make_follow(
        "PERMIT_ISSUED", intent, owner_incarnation=owner_incarnation,
        prev_digest=intent["record_digest"]))
    held.intent, held.permit = intent, issued["payload"]["permit_id"]
    if not P.authorize_launch(P.project_unit(held.records),
                              current_owner=owner_incarnation, permit=held.permit):
        raise P.A2UnitRefused("permit does not authorize this launch")
    return held


def settle(held: Reservation, *, append: Callable[[dict], Any],
           status: str | None = None, reason: str | None = None,
           result_digest: str | None = None, not_acquired: str | None = None) -> str:
    """Record LAUNCHED+TERMINAL, or durable NOT_ACQUIRED; returns the projected phase."""
    def follow(kind: str, payload: Mapping[str, Any] | None = None) -> None:
        _emit(held, append, P.make_follow(
            kind, held.intent, owner_incarnation=held.owner,
            prev_digest=held.records[-1]["record_digest"], payload=payload))
    if not_acquired is not None:
        follow("NOT_ACQUIRED", {"reason": not_acquired or "acquisition failed"})
    else:
        follow("LAUNCHED")
        follow("TERMINAL", {"status": status, "reason": reason,
                            "result_digest": result_digest})
    return P.project_unit(held.records).phase


def settle_terminal(held: Reservation, terminal: Any, *, append: Callable[[dict], Any]) -> str:
    """Bind the exact lifecycle terminal: accepted -> valid, otherwise invalid."""
    return settle(held, append=append, status="valid" if terminal.accepted else "invalid",
                  reason=None if terminal.accepted else (
                      terminal.reason or "lifecycle terminal was not accepted"),
                  result_digest=P._digest(dataclasses.asdict(terminal)))


def _emit(held: Reservation, append: Callable[[dict], Any], record: dict) -> dict:
    append(record)   # a raising sink aborts before any launch or later record
    held.records.append(record)
    return record


def launch_selected_unit(*, plan, unit_range: Mapping[str, Any], unit_index: int,
                         execution_id: str, frame_digest: str,
                         producer_identity: Mapping[str, Any], owner_incarnation: int,
                         prior_records: Sequence[Mapping[str, Any]],
                         append: Callable[[dict], Any],
                         launcher: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """Run one launch. `launcher(permit_id)` returns {"status": valid|invalid|diagnostic,
    "reason": str|None, "result_digest": sha|None}. Returns {"phase", "attempt", "records"}."""
    held = reserve({"plan": plan, "unit_range": unit_range, "unit_index": unit_index,
                    "execution_id": execution_id, "frame_digest": frame_digest,
                    "producer_identity": producer_identity},
                   owner_incarnation=owner_incarnation, prior_records=prior_records,
                   append=append)
    try:
        result = launcher(held.permit)
    except AcquisitionFailed as exc:
        settle(held, append=append, not_acquired=str(exc))
    else:
        settle(held, append=append, status=result["status"], reason=result.get("reason"),
               result_digest=result.get("result_digest"))
    state = P.project_unit(held.records)
    return {"phase": state.phase, "attempt": state.attempt, "records": held.records}


__all__ = ["AcquisitionFailed", "LOGICAL_ID", "Reservation", "identity_key",
           "launch_selected_unit",
           "reserve", "selected_launch_identity", "selected_prepared", "settle",
           "settle_terminal"]
