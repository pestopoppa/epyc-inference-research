"""Native per-unit A2 execution records (AKU-04f), additive and closed.

Versioned unit record chain (`epyc.autokernel.a2_unit_record.v2`) for ONE original unit of
the unchanged full ExperimentPlan:

    INTENT -> PERMIT_ISSUED -> LAUNCHED -> TERMINAL(valid|invalid|diagnostic)
          \\-> NOT_ACQUIRED -> INTENT(attempt+1, identical identity)

Rules this module enforces (and nothing else; it launches nothing and grants no authority):

* The permit is a one-shot, first-issue, current-owner permit. It is issued in the same
  owner incarnation that wrote the INTENT and is consumed by one LAUNCHED record.
* A replayed INTENT (or a permit from another incarnation) never authorizes a launch.
* A replayed non-terminal state with no durable NOT_ACQUIRED is UNKNOWN, never "absent":
  relaunch is refused. Only a NOT_ACQUIRED written by the permit's owner before it
  released, or a DIAGNOSTIC terminal, moves the unit on; a retry preserves the exact
  execution/plan/frame/range/unit/producer identity and increments only `attempt`.
* A diagnostic terminal is terminal: nothing may follow it, and it carries no scientific
  claim (a scientific rerun is a new plan unit, not a relaunch).
* Membership is rederived from the full plan (no one-unit plan rehash, no hidden
  multi-unit launch): the intent's unit must be the plan's unit at its order index and
  lie inside the selected contiguous range.

The existing transition validator (`a2_execution_state`) and its v1 events are untouched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import experiment_plan as ep

RECORD_SCHEMA = "epyc.autokernel.a2_unit_record.v2"
KINDS = ("INTENT", "PERMIT_ISSUED", "LAUNCHED", "NOT_ACQUIRED", "TERMINAL")
TERMINAL_STATUSES = ("valid", "invalid", "diagnostic")
MAX_ATTEMPTS = 8
MAX_RECORDS = 5 * MAX_ATTEMPTS


class A2UnitRefused(RuntimeError):
    pass


def _digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise A2UnitRefused(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(raw.encode()).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise A2UnitRefused(f"{label} must be lowercase SHA-256")
    return value


def _int(value: Any, label: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise A2UnitRefused(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise A2UnitRefused(f"{label} must be bounded nonempty text")
    return value


def select_unit_range(plan: ep.ExperimentPlan, first: int, last: int) -> dict[str, Any]:
    """Digest-covered contiguous absolute range of the UNCHANGED full plan."""
    try:
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
    except Exception as exc:
        raise A2UnitRefused(f"plan is invalid: {exc}") from exc
    ordered = sorted(plan.expected_units, key=lambda item: item.order_index)
    indices = [unit.order_index for unit in ordered]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise A2UnitRefused("plan order indices are not contiguous")
    if not (indices[0] <= first <= last <= indices[-1]):
        raise A2UnitRefused("selected range is outside the full plan")
    members = [unit.to_dict() for unit in ordered if first <= unit.order_index <= last]
    return {"plan_digest": plan.digest, "first": first, "last": last,
            "members_digest": _digest(members)}


def _identity(execution_id, plan_digest, frame_digest, unit_range, unit, producer) -> dict:
    return {"execution_id": execution_id, "plan_digest": plan_digest,
            "frame_digest": frame_digest, "unit_range": dict(unit_range),
            "unit": dict(unit), "producer_identity": dict(producer)}


def permit_id(intent_digest: str, attempt: int, owner_incarnation: int) -> str:
    return _digest({"purpose": "a2-first-issue-permit", "intent_digest": intent_digest,
                    "attempt": attempt, "owner_incarnation": owner_incarnation})


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": _digest(body)}


def make_intent(*, plan: ep.ExperimentPlan, unit_range: Mapping[str, Any], unit_index: int,
                execution_id: str, frame_digest: str, producer_identity: Mapping[str, Any],
                owner_incarnation: int, attempt: int = 1,
                prev_digest: str | None = None) -> dict[str, Any]:
    expected = select_unit_range(plan, unit_range["first"], unit_range["last"])
    if dict(unit_range) != expected:
        raise A2UnitRefused("unit range differs from the digest-covered full-plan range")
    ordered = {u.order_index: u for u in plan.expected_units}
    if unit_index not in ordered or not (expected["first"] <= unit_index <= expected["last"]):
        raise A2UnitRefused("unit is not a member of the selected range")
    return _seal({
        "schema": RECORD_SCHEMA, "kind": "INTENT",
        **_identity(_sha(execution_id, "execution_id"), plan.digest,
                    _sha(frame_digest, "frame_digest"), expected,
                    ordered[unit_index].to_dict(), producer_identity),
        "attempt": _int(attempt, "attempt"),
        "owner_incarnation": _int(owner_incarnation, "owner_incarnation"),
        "intent_digest": None, "prev_digest": prev_digest, "payload": None})


def _intent_digest(intent: Mapping[str, Any]) -> str:
    return intent["record_digest"]


def make_follow(kind: str, intent: Mapping[str, Any], *, owner_incarnation: int,
                prev_digest: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """PERMIT_ISSUED | LAUNCHED | NOT_ACQUIRED | TERMINAL bound to one exact intent."""
    if kind not in KINDS[1:]:
        raise A2UnitRefused("follow record kind is unsupported")
    base = {k: intent[k] for k in ("execution_id", "plan_digest", "frame_digest",
                                    "unit_range", "unit", "producer_identity", "attempt")}
    ipd = _intent_digest(intent)
    body = {"schema": RECORD_SCHEMA, "kind": kind, **base,
            "owner_incarnation": _int(owner_incarnation, "owner_incarnation"),
            "intent_digest": ipd, "prev_digest": prev_digest,
            "payload": dict(payload) if payload is not None else None}
    if kind in ("PERMIT_ISSUED", "LAUNCHED", "NOT_ACQUIRED"):
        pid = permit_id(ipd, intent["attempt"], intent["owner_incarnation"])
        body["payload"] = {"permit_id": pid, **(body["payload"] or {})}
    return _seal(body)


_FIELDS = {"schema", "kind", "execution_id", "plan_digest", "frame_digest", "unit_range",
           "unit", "producer_identity", "attempt", "owner_incarnation", "intent_digest",
           "prev_digest", "payload", "record_digest"}


def validate_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise A2UnitRefused("A2 unit record fields differ")
    row = dict(value)
    if row["schema"] != RECORD_SCHEMA or row["kind"] not in KINDS:
        raise A2UnitRefused("A2 unit record schema/kind is unsupported")
    for name in ("execution_id", "plan_digest", "frame_digest", "record_digest"):
        _sha(row[name], name)
    _int(row["attempt"], "attempt")
    _int(row["owner_incarnation"], "owner_incarnation")
    if row["prev_digest"] is not None:
        _sha(row["prev_digest"], "prev_digest")
    try:
        ep.UnitSpec.from_dict(row["unit"])
    except Exception as exc:
        raise A2UnitRefused(f"A2 unit is invalid: {exc}") from exc
    unit_range = row["unit_range"]
    if (not isinstance(unit_range, Mapping)
            or set(unit_range) != {"plan_digest", "first", "last", "members_digest"}
            or unit_range["plan_digest"] != row["plan_digest"]
            or not row["unit"]["order_index"] in range(unit_range["first"],
                                                        unit_range["last"] + 1)):
        raise A2UnitRefused("A2 unit range is not bound to the full plan and unit")
    if row["kind"] == "INTENT":
        if row["intent_digest"] is not None or row["payload"] is not None:
            raise A2UnitRefused("A2 intent carries no intent digest or payload")
    else:
        _sha(row["intent_digest"], "intent_digest")
        payload = row["payload"]
        if not isinstance(payload, Mapping):
            raise A2UnitRefused("A2 follow record needs a payload")
        if row["kind"] == "TERMINAL":
            if (set(payload) != {"status", "reason", "result_digest"}
                    or payload["status"] not in TERMINAL_STATUSES):
                raise A2UnitRefused("A2 terminal payload fields differ")
            if payload["status"] == "diagnostic":
                _text(payload["reason"], "diagnostic reason")
            elif payload["result_digest"] is None:
                _sha(None, "result_digest")
            if payload["result_digest"] is not None:
                _sha(payload["result_digest"], "result_digest")
        else:
            _sha(payload.get("permit_id"), "permit_id")
            if row["kind"] == "NOT_ACQUIRED":
                _text(payload.get("reason"), "not-acquired reason")
    if _digest({k: v for k, v in row.items() if k != "record_digest"}) != row["record_digest"]:
        raise A2UnitRefused("A2 unit record digest differs")
    return row


@dataclass(frozen=True)
class UnitState:
    phase: str            # intent_only | permit_issued | launched_unknown | not_acquired | terminal
    attempt: int
    intent: Mapping[str, Any]
    permit_id: str | None
    permit_owner: int | None
    terminal_status: str | None
    history_digest: str

    @property
    def absence_proven(self) -> bool:
        """True ONLY after a durable NOT_ACQUIRED. Replayed unknown is never absence."""
        return self.phase == "not_acquired"


def project_unit(records: Sequence[Mapping[str, Any]]) -> UnitState:
    if not isinstance(records, (list, tuple)) or not records:
        raise A2UnitRefused("A2 unit history is empty")
    if len(records) > MAX_RECORDS:
        raise A2UnitRefused("A2 unit history exceeds its bound")
    rows = [validate_record(r) for r in records]
    ident = lambda r: _digest({k: r[k] for k in ("execution_id", "plan_digest",
                                                  "frame_digest", "unit_range", "unit",
                                                  "producer_identity")})
    fixed = ident(rows[0])
    phase, attempt, intent, pid, owner, terminal = None, 0, None, None, None, None
    prev = None
    for row in rows:
        if ident(row) != fixed:
            raise A2UnitRefused("A2 unit history changes identity")
        if row["prev_digest"] != prev:
            raise A2UnitRefused("A2 unit history is not a chain")
        prev = row["record_digest"]
        kind = row["kind"]
        if terminal is not None:
            raise A2UnitRefused("A2 unit record follows a terminal")
        if kind == "INTENT":
            if phase not in (None, "not_acquired"):
                raise A2UnitRefused("A2 intent is duplicate; replay cannot re-intend")
            if row["attempt"] != attempt + 1 or row["attempt"] > MAX_ATTEMPTS:
                raise A2UnitRefused("A2 intent attempt is out of order")
            intent, phase, pid, owner = row, "intent_only", None, None
            attempt = row["attempt"]
            continue
        if intent is None or row["intent_digest"] != _intent_digest(intent) \
                or row["attempt"] != attempt:
            raise A2UnitRefused("A2 record does not name the current intent")
        expected_pid = permit_id(_intent_digest(intent), attempt, intent["owner_incarnation"])
        if kind != "TERMINAL" and row["payload"]["permit_id"] != expected_pid:
            raise A2UnitRefused("A2 permit identity differs")
        if kind == "PERMIT_ISSUED":
            if phase != "intent_only" or row["owner_incarnation"] != intent["owner_incarnation"]:
                raise A2UnitRefused("A2 permit is not first-issue by the intent's owner")
            phase, pid, owner = "permit_issued", expected_pid, row["owner_incarnation"]
        elif kind == "LAUNCHED":
            if phase != "permit_issued" or row["owner_incarnation"] != owner:
                raise A2UnitRefused("A2 launch lacks its unconsumed owner permit")
            phase = "launched_unknown"
        elif kind == "NOT_ACQUIRED":
            if (phase not in ("intent_only", "permit_issued")
                    or row["owner_incarnation"] != intent["owner_incarnation"]):
                raise A2UnitRefused("A2 not-acquired needs the owner and an unlaunched permit")
            phase = "not_acquired"
        else:
            status = row["payload"]["status"]
            if status != "diagnostic" and phase != "launched_unknown":
                raise A2UnitRefused("A2 valid/invalid terminal requires a recorded launch")
            if phase == "not_acquired":
                raise A2UnitRefused("A2 terminal after not-acquired needs a new intent")
            phase, terminal = "terminal", status
    return UnitState(phase, attempt, intent, pid, owner, terminal,
                     _digest([r["record_digest"] for r in rows]))


def unit_key(record: Mapping[str, Any]) -> str:
    """Stable key of the one unit chain a record belongs to (fixed identity, no attempt)."""
    try:
        return _digest({k: record[k] for k in ("execution_id", "plan_digest", "frame_digest",
                                                "unit_range", "unit", "producer_identity")})
    except (KeyError, TypeError) as exc:
        raise A2UnitRefused("A2 unit record lacks its identity fields") from exc


def replay_units(payloads: Sequence[Mapping[str, Any]]) -> dict[str, UnitState]:
    """Replay journal payloads (v2 only; others ignored) into per-unit permit state.

    Every state here came from replay: `authorize_launch` can only succeed for the CURRENT
    owner holding an unconsumed permit, and `launched_unknown` is never absence."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for payload in payloads:
        if isinstance(payload, Mapping) and payload.get("schema") == RECORD_SCHEMA:
            groups.setdefault(unit_key(payload), []).append(payload)
    return {key: project_unit(rows) for key, rows in groups.items()}


def authorize_launch(state: UnitState, *, current_owner: int, permit: str) -> bool:
    """True only for an unconsumed permit presented by the current owner that issued it.

    A state obtained by replay (any phase but `permit_issued`, or a permit issued by a
    different incarnation) can never authorize."""
    _int(current_owner, "current_owner")
    return (state.phase == "permit_issued" and state.permit_owner == current_owner
            and state.permit_id == permit)


def may_relaunch(state: UnitState) -> bool:
    """A new attempt needs durable proof of non-acquisition; unknown is not proof."""
    return state.phase == "not_acquired" and state.attempt < MAX_ATTEMPTS


__all__ = ["A2UnitRefused", "KINDS", "MAX_ATTEMPTS", "RECORD_SCHEMA", "TERMINAL_STATUSES",
           "UnitState", "authorize_launch", "make_follow", "make_intent", "may_relaunch",
           "permit_id", "project_unit", "replay_units", "select_unit_range", "unit_key",
           "validate_record"]
