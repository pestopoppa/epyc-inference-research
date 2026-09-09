"""Closed durable state for controller-owned A2 runtime phase events.

This module validates and projects journal payloads only.  It grants no execution,
worker, replay-attestation, evidence, or nomination authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import discovery_screen
from . import experiment_plan as ep

TRANSITION_SCHEMA = "epyc.autokernel.a2_runtime_execution_transition.v1"
BANK_REFERENCE_SCHEMA = "epyc.autokernel.a2_bank_reference.v1"
MAX_PHASE_EVENTS = 14
MAX_BANK_REFERENCES = 1
PHASES = ("anchor_bank", "candidate_screen")


class A2ExecutionStateRefused(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise A2ExecutionStateRefused(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise A2ExecutionStateRefused(f"{label} must be bounded nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label, 64)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise A2ExecutionStateRefused(f"{label} must be lowercase SHA-256")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise A2ExecutionStateRefused(f"{label} must be a positive integer")
    return value


def _bank_reference(value: Any) -> dict[str, Any]:
    fields = {"schema", "target_plan_digest", "target_frame_digest",
              "source_execution_id", "source_logical_id", "source_plan_digest",
              "source_frame_digest", "source_anchor_history_digest",
              "source_anchor_event_digests", "source_anchor_journal_entry_ids",
              "source_anchor_seal_event_digest", "bank_digest", "reference_digest"}
    if not isinstance(value, Mapping) or set(value) != fields \
            or value.get("schema") != BANK_REFERENCE_SCHEMA:
        raise A2ExecutionStateRefused("A2 bank reference fields/schema differ")
    row = dict(value)
    for name in ("target_plan_digest", "target_frame_digest", "source_execution_id",
                 "source_plan_digest", "source_frame_digest",
                 "source_anchor_history_digest", "source_anchor_seal_event_digest",
                 "bank_digest", "reference_digest"):
        _sha(row[name], name)
    _text(row["source_logical_id"], "source_logical_id")
    event_digests = row["source_anchor_event_digests"]
    entry_ids = row["source_anchor_journal_entry_ids"]
    if (not isinstance(event_digests, list) or len(event_digests) != 7
            or any(_sha(item, "source anchor event digest") != item
                   for item in event_digests)
            or len(set(event_digests)) != 7):
        raise A2ExecutionStateRefused(
            "A2 bank reference requires seven distinct anchor event digests")
    if row["source_anchor_seal_event_digest"] != event_digests[-1]:
        raise A2ExecutionStateRefused(
            "A2 bank reference seal is not the final source anchor event")
    if (not isinstance(entry_ids, list) or len(entry_ids) != 7
            or any(_text(item, "source anchor Journal entry ID", 128) != item
                   for item in entry_ids)
            or len(set(entry_ids)) != 7):
        raise A2ExecutionStateRefused(
            "A2 bank reference requires seven distinct Journal entry IDs")
    body = {name: row[name] for name in fields - {"reference_digest"}}
    if row["reference_digest"] != _digest(body):
        raise A2ExecutionStateRefused("A2 bank reference digest differs")
    return row


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("schema") == BANK_REFERENCE_SCHEMA:
        return _bank_reference(value)
    return discovery_screen.validate_phase_event(value)


def _record_plan_frame(value: Mapping[str, Any]) -> tuple[str, str]:
    if value["schema"] == BANK_REFERENCE_SCHEMA:
        return value["target_plan_digest"], value["target_frame_digest"]
    return value["plan_digest"], value["frame_digest"]


def execution_identity(*, campaign_id: str, config_generation: int,
                       config_digest: str, logical_id: str,
                       plan_digest: str, frame_digest: str) -> str:
    body = {"campaign_id": _text(campaign_id, "campaign_id"),
            "config_generation": _positive(config_generation, "config_generation"),
            "config_digest": _sha(config_digest, "config_digest"),
            "logical_id": _text(logical_id, "logical_id"),
            "plan_digest": _sha(plan_digest, "plan_digest"),
            "frame_digest": _sha(frame_digest, "frame_digest")}
    return _digest(body)


def attempt_identity(execution_id: str, event: Mapping[str, Any]) -> dict[str, str] | None:
    execution_id = _sha(execution_id, "execution_id")
    event = _record(event)
    if event["schema"] == BANK_REFERENCE_SCHEMA or event["state"] == "SEALED":
        return None
    phase, index = event["phase"], event["index"]
    return {"request_id": _digest({"execution_id": execution_id, "phase": phase,
                                    "index": index}),
            "plan_digest": event["plan_digest"],
            "lineage_id": f"a2:{execution_id}:{phase}",
            "stage_id": f"a2-unit:{phase}:{index}"}


def _validate_event_payload(event: Mapping[str, Any]) -> None:
    payload = event["payload"]
    if not isinstance(payload, Mapping):
        raise A2ExecutionStateRefused("A2 phase payload must be an object")
    if event["state"] == "INTENT":
        if set(payload) != {"unit", "producer_identity"}:
            raise A2ExecutionStateRefused("A2 intent payload fields differ")
        try:
            unit = ep.UnitSpec.from_dict(payload["unit"])
        except Exception as exc:
            raise A2ExecutionStateRefused(f"A2 intent unit is invalid: {exc}") from exc
        producer = payload["producer_identity"]
        if not isinstance(producer, Mapping) or set(producer) != {
                "frame_digest", "recipe_snapshot_digest", "recipe_execution_digest"}:
            raise A2ExecutionStateRefused("A2 intent producer identity fields differ")
        for name, value in producer.items():
            _sha(value, f"producer_identity.{name}")
        expected_arm = "anchor" if event["phase"] == "anchor_bank" else "candidate"
        expected_order = event["index"] if expected_arm == "anchor" else event["index"] + 3
        if (unit.arm != expected_arm or unit.order_index != expected_order
                or producer["frame_digest"] != event["frame_digest"]):
            raise A2ExecutionStateRefused("A2 intent differs from fixed phase membership")
    elif event["state"] == "TERMINAL":
        if set(payload) != {"status", "reason", "result"} \
                or payload["status"] not in {"valid", "invalid"} \
                or (payload["reason"] is not None
                    and (not isinstance(payload["reason"], str) or not payload["reason"])):
            raise A2ExecutionStateRefused("A2 terminal payload fields differ")
        try:
            discovery_screen.InvocationResult.from_dict(payload["result"])
        except Exception as exc:
            raise A2ExecutionStateRefused(f"A2 terminal result is invalid: {exc}") from exc
    elif event["phase"] == "anchor_bank":
        try:
            bank = discovery_screen.BaselineBank.from_dict(payload)
        except Exception as exc:
            raise A2ExecutionStateRefused(f"A2 bank seal is invalid: {exc}") from exc
        if _digest(bank.to_dict()["frame"]) != event["frame_digest"]:
            raise A2ExecutionStateRefused("A2 bank seal frame differs")
    else:
        try:
            receipt = discovery_screen.ScreenReceipt.from_dict(payload)
        except Exception as exc:
            raise A2ExecutionStateRefused(f"A2 screen seal is invalid: {exc}") from exc
        if (receipt.plan_digest != event["plan_digest"]
                or receipt.frame_digest != event["frame_digest"]):
            raise A2ExecutionStateRefused("A2 screen seal identity differs")


def make_bank_reference(*, source_values: Sequence[Mapping[str, Any]],
                        source_journal_entry_ids: Sequence[str],
                        target_plan_digest: str,
                        target_frame_digest: str) -> dict[str, Any]:
    """Describe one actual sealed anchor source without copying its measurements."""
    if (not isinstance(source_values, (list, tuple))
            or not isinstance(source_journal_entry_ids, (list, tuple))
            or len(source_values) != len(source_journal_entry_ids)):
        raise A2ExecutionStateRefused(
            "A2 bank source transitions and Journal identities differ")
    projection = project_transitions(source_values)
    anchor_rows = [validate_transition(value) for value in source_values
                   if value["event"].get("schema") != BANK_REFERENCE_SCHEMA
                   and value["event"].get("phase") == "anchor_bank"]
    anchor_ids = [entry_id for value, entry_id in zip(
        source_values, source_journal_entry_ids, strict=True)
        if value["event"].get("schema") != BANK_REFERENCE_SCHEMA
        and value["event"].get("phase") == "anchor_bank"]
    if (projection.bank_reference is not None or "anchor_bank" not in projection.sealed_phases
            or len(anchor_rows) != 7 or len(anchor_ids) != 7):
        raise A2ExecutionStateRefused(
            "A2 bank source must be an original seven-event sealed anchor phase")
    if projection.frame_digest != _sha(target_frame_digest, "target_frame_digest"):
        raise A2ExecutionStateRefused(
            "A2 bank source frame differs from the current target frame")
    anchor_events = [row["event"] for row in anchor_rows]
    seal = anchor_events[-1]
    if seal["state"] != "SEALED":
        raise A2ExecutionStateRefused("A2 bank source lacks its final anchor seal")
    bank = discovery_screen.BaselineBank.from_dict(seal["payload"])
    body = {
        "schema": BANK_REFERENCE_SCHEMA,
        "target_plan_digest": _sha(target_plan_digest, "target_plan_digest"),
        "target_frame_digest": target_frame_digest,
        "source_execution_id": projection.execution_id,
        "source_logical_id": projection.logical_id,
        "source_plan_digest": projection.plan_digest,
        "source_frame_digest": projection.frame_digest,
        "source_anchor_history_digest": _digest(anchor_rows),
        "source_anchor_event_digests": [event["event_digest"] for event in anchor_events],
        "source_anchor_journal_entry_ids": [
            _text(item, "source anchor Journal entry ID", 128) for item in anchor_ids],
        "source_anchor_seal_event_digest": seal["event_digest"],
        "bank_digest": bank.bank_digest,
    }
    return _bank_reference({**body, "reference_digest": _digest(body)})


def validate_event_membership(event: Mapping[str, Any],
                              plan: ep.ExperimentPlan) -> dict[str, Any]:
    """Bind a phase event to the exact frozen A2 plan member at its index."""
    try:
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
    except Exception as exc:
        raise A2ExecutionStateRefused(f"A2 membership plan is invalid: {exc}") from exc
    event = discovery_screen.validate_phase_event(event, plan_digest=plan.digest)
    _validate_event_payload(event)
    if event["state"] == "SEALED":
        return event
    offset = 0 if event["phase"] == "anchor_bank" else 3
    ordered = tuple(sorted(plan.expected_units, key=lambda item: item.order_index))
    if len(ordered) != 6:
        raise A2ExecutionStateRefused("A2 membership plan does not contain six fixed units")
    expected = ordered[offset + event["index"]]
    if event["state"] == "INTENT":
        actual = ep.UnitSpec.from_dict(event["payload"]["unit"])
        actual_values = (actual.unit_id, actual.arm, actual.process_id, actual.order_index)
    else:
        raw = discovery_screen.InvocationResult.from_dict(
            event["payload"]["result"]).raw_unit
        if (raw.plan_digest != plan.digest
                or tuple(raw.prompt_ids) != tuple(expected.expected_prompt_ids)):
            raise A2ExecutionStateRefused("A2 terminal differs from frozen plan membership")
        actual_values = (raw.unit_id, raw.arm, raw.process_id, raw.observed_order_index)
    expected_values = (expected.unit_id, expected.arm, expected.process_id,
                       expected.order_index)
    if actual_values != expected_values:
        raise A2ExecutionStateRefused("A2 event differs from frozen plan membership")
    return event


def make_transition(*, campaign_id: str, config_generation: int,
                    config_digest: str, supervisor_incarnation: int,
                    logical_id: str, event: Mapping[str, Any]) -> dict[str, Any]:
    event = _record(event)
    plan_digest, frame_digest = _record_plan_frame(event)
    execution_id = execution_identity(
        campaign_id=campaign_id, config_generation=config_generation,
        config_digest=config_digest, logical_id=logical_id,
        plan_digest=plan_digest, frame_digest=frame_digest)
    return {"schema": TRANSITION_SCHEMA, "campaign_id": campaign_id,
            "config_generation": config_generation, "config_digest": config_digest,
            "supervisor_incarnation": _positive(
                supervisor_incarnation, "supervisor_incarnation"),
            "logical_id": logical_id, "execution_id": execution_id,
            "attempt_identity": attempt_identity(execution_id, event), "event": event}


def validate_transition(value: Any) -> dict[str, Any]:
    fields = {"schema", "campaign_id", "config_generation", "config_digest",
              "supervisor_incarnation", "logical_id", "execution_id",
              "attempt_identity", "event"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise A2ExecutionStateRefused("A2 execution transition fields differ")
    row = dict(value)
    if row["schema"] != TRANSITION_SCHEMA:
        raise A2ExecutionStateRefused("A2 execution transition schema is unsupported")
    campaign_id = _text(row["campaign_id"], "campaign_id")
    generation = _positive(row["config_generation"], "config_generation")
    config_digest = _sha(row["config_digest"], "config_digest")
    _positive(row["supervisor_incarnation"], "supervisor_incarnation")
    logical_id = _text(row["logical_id"], "logical_id")
    event = _record(row["event"])
    if event["schema"] != BANK_REFERENCE_SCHEMA:
        _validate_event_payload(event)
    plan_digest, frame_digest = _record_plan_frame(event)
    expected = execution_identity(
        campaign_id=campaign_id, config_generation=generation,
        config_digest=config_digest, logical_id=logical_id,
        plan_digest=plan_digest, frame_digest=frame_digest)
    if _sha(row["execution_id"], "execution_id") != expected:
        raise A2ExecutionStateRefused("A2 execution identity differs from its fixed frame")
    expected_attempt = attempt_identity(expected, event)
    if row["attempt_identity"] != expected_attempt:
        raise A2ExecutionStateRefused("A2 attempt identity differs from its phase member")
    row["event"] = event
    return row


@dataclass(frozen=True)
class PhaseProjection:
    execution_id: str
    logical_id: str
    plan_digest: str
    frame_digest: str
    events: tuple[Mapping[str, Any], ...]
    pending_intents: tuple[Mapping[str, Any], ...]
    sealed_phases: tuple[str, ...]
    bank_reference: Mapping[str, Any] | None
    history_digest: str


def project_transitions(values: Sequence[Mapping[str, Any]]) -> PhaseProjection:
    if not isinstance(values, (list, tuple)) or not values:
        raise A2ExecutionStateRefused("A2 execution history is empty")
    if len(values) > MAX_PHASE_EVENTS + MAX_BANK_REFERENCES:
        raise A2ExecutionStateRefused("A2 execution exceeds its bounded transition count")
    rows = [validate_transition(value) for value in values]
    first = rows[0]
    fixed = (first["campaign_id"], first["config_generation"], first["config_digest"],
             first["logical_id"], first["execution_id"])
    if any((row["campaign_id"], row["config_generation"], row["config_digest"],
            row["logical_id"], row["execution_id"]) != fixed for row in rows):
        raise A2ExecutionStateRefused("A2 execution history changes fixed identity")
    plan_digest, frame_digest = _record_plan_frame(first["event"])
    if any(_record_plan_frame(row["event"])
           != (plan_digest, frame_digest) for row in rows):
        raise A2ExecutionStateRefused("A2 execution history changes plan/frame")
    references = [row["event"] for row in rows
                  if row["event"]["schema"] == BANK_REFERENCE_SCHEMA]
    if len(references) > MAX_BANK_REFERENCES:
        raise A2ExecutionStateRefused("A2 execution repeats a bank reference")
    bank_reference = references[0] if references else None
    events = [row["event"] for row in rows
              if row["event"]["schema"] != BANK_REFERENCE_SCHEMA]
    if len(events) > MAX_PHASE_EVENTS:
        raise A2ExecutionStateRefused("A2 execution exceeds its fixed phase-event bound")
    record_digests = [event.get("event_digest", event.get("reference_digest"))
                      for event in (row["event"] for row in rows)]
    if len(record_digests) != len(set(record_digests)):
        raise A2ExecutionStateRefused("A2 execution history repeats a transition record")
    if bank_reference is not None:
        reference_position = next(index for index, row in enumerate(rows)
                                  if row["event"]["schema"] == BANK_REFERENCE_SCHEMA)
        if reference_position != 0:
            raise A2ExecutionStateRefused("A2 bank reference must precede local phase events")

    pending: dict[str, Mapping[str, Any] | None] = {phase: None for phase in PHASES}
    completed: dict[str, int] = {phase: 0 for phase in PHASES}
    results: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    sealed: list[str] = []
    for event in events:
        phase, state, index = event["phase"], event["state"], event["index"]
        if phase == "anchor_bank" and bank_reference is not None:
            raise A2ExecutionStateRefused(
                "A2 execution cannot combine reused and locally measured anchors")
        if (phase == "candidate_screen" and "anchor_bank" not in sealed
                and bank_reference is None):
            raise A2ExecutionStateRefused("candidate phase precedes sealed anchor bank")
        if state == "INTENT":
            if pending[phase] is not None or index != completed[phase] or phase in sealed:
                raise A2ExecutionStateRefused("A2 intent is duplicate or out of fixed order")
            pending[phase] = event
        elif state == "TERMINAL":
            intent = pending[phase]
            if intent is None or index != intent["index"]:
                raise A2ExecutionStateRefused("A2 terminal lacks its exact pending intent")
            unit = ep.UnitSpec.from_dict(intent["payload"]["unit"])
            producer = intent["payload"]["producer_identity"]
            result = discovery_screen.InvocationResult.from_dict(
                event["payload"]["result"])
            raw = result.raw_unit
            if (raw.unit_id != unit.unit_id or raw.arm != unit.arm
                    or raw.process_id != unit.process_id
                    or raw.observed_order_index != unit.order_index
                    or tuple(raw.prompt_ids) != tuple(unit.expected_prompt_ids)
                    or dict(result.proof.producer_identity) != dict(producer)):
                raise A2ExecutionStateRefused(
                    "A2 terminal differs from its declared fixed unit/producer")
            results[phase].append(result.to_dict())
            pending[phase] = None
            completed[phase] += 1
        elif (pending[phase] is not None or completed[phase] != 3
              or phase in sealed):
            raise A2ExecutionStateRefused("A2 phase sealed before three exact terminals")
        else:
            sealed_results = (event["payload"]["anchor_results"]
                              if phase == "anchor_bank"
                              else event["payload"]["candidate_results"])
            if sealed_results != results[phase]:
                raise A2ExecutionStateRefused(
                    "A2 phase seal differs from its exact terminal results")
            sealed.append(phase)
    pending_rows = tuple({"phase": phase, "index": event["index"],
                          "intent_event_digest": event["event_digest"],
                          "request_identity": attempt_identity(first["execution_id"], event)}
                         for phase, event in pending.items() if event is not None)
    return PhaseProjection(
        first["execution_id"], first["logical_id"], plan_digest, frame_digest,
        tuple(events), pending_rows, tuple(sealed), bank_reference,
        _digest({"plan_digest": plan_digest, "frame_digest": frame_digest,
                 "bank_reference": bank_reference, "events": events}))


__all__ = ["A2ExecutionStateRefused", "BANK_REFERENCE_SCHEMA",
           "MAX_BANK_REFERENCES", "MAX_PHASE_EVENTS", "PHASES",
           "PhaseProjection", "TRANSITION_SCHEMA", "execution_identity",
           "attempt_identity", "make_bank_reference", "make_transition", "project_transitions",
           "validate_event_membership", "validate_transition"]
