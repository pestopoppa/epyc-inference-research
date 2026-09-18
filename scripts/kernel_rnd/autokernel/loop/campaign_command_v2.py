"""Closed supervisor-fenced campaign command v2 and journal transition v3."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from typing import Any, Mapping

from . import worker_lifecycle

COMMAND_SCHEMA = "epyc.autokernel.campaign_command.v2"
TRANSITION_SCHEMA = "epyc.autokernel.campaign_command_transition.v3"
COMMAND_FIELDS = frozenset({
    "schema", "campaign_id", "config_generation", "config_digest",
    "supervisor_incarnation", "request_id", "operation", "payload",
    "payload_digest", "expected_control_revision",
})


class CommandV2Refused(ValueError):
    pass


def command_digest(*, operation: str, payload: Mapping[str, Any], campaign_id: str,
                   config_generation: int, config_digest: str,
                   supervisor_incarnation: int, request_id: str,
                   expected_control_revision: int) -> str:
    body = {"schema": COMMAND_SCHEMA, "operation": operation, "payload": dict(payload),
            "campaign_id": campaign_id, "config_generation": config_generation,
            "config_digest": config_digest,
            "supervisor_incarnation": supervisor_incarnation,
            "request_id": request_id,
            "expected_control_revision": expected_control_revision}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise CommandV2Refused(f"{label} must be lowercase SHA-256")
    return value


def validate_command(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != COMMAND_FIELDS:
        raise CommandV2Refused("v2 command has missing/unknown fields")
    row = dict(value)
    if row["schema"] != COMMAND_SCHEMA:
        raise CommandV2Refused("unsupported v2 command schema")
    for name in ("campaign_id", "request_id"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise CommandV2Refused(f"v2 command {name} is invalid")
    _sha(row["config_digest"], "v2 command config_digest")
    _sha(row["payload_digest"], "v2 command payload_digest")
    for name, minimum in (("config_generation", 1), ("supervisor_incarnation", 1),
                          ("expected_control_revision", 0)):
        if not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] < minimum:
            raise CommandV2Refused(f"v2 command {name} is invalid")
    if row["operation"] not in {"pause", "resume", "drain"}:
        raise CommandV2Refused("v2 command operation is invalid")
    if not isinstance(row["payload"], Mapping) or row["payload"]:
        raise CommandV2Refused("v2 command payload must be empty")
    expected = command_digest(
        operation=row["operation"], payload={}, campaign_id=row["campaign_id"],
        config_generation=row["config_generation"], config_digest=row["config_digest"],
        supervisor_incarnation=row["supervisor_incarnation"],
        request_id=row["request_id"],
        expected_control_revision=row["expected_control_revision"])
    if not hmac.compare_digest(row["payload_digest"], expected):
        raise CommandV2Refused("v2 command payload_digest differs")
    row["payload"] = {}
    return row


def validate_transition(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "phase", "campaign_id", "config_digest", "config_generation",
              "supervisor_id", "supervisor_incarnation", "control_revision",
              "occurred_at", "command", "result"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CommandV2Refused("v3 command transition has missing/unknown fields")
    row = dict(value)
    if row["schema"] != TRANSITION_SCHEMA or row["phase"] not in {"ACCEPTED", "COMPLETED"}:
        raise CommandV2Refused("v3 command transition schema/phase is invalid")
    try:
        worker_lifecycle.CampaignBinding(
            row["campaign_id"], row["config_digest"], row["config_generation"],
            row["supervisor_id"], row["supervisor_incarnation"])
        occurred = datetime.fromisoformat(row["occurred_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise CommandV2Refused("v3 command transition writer/time is invalid") from exc
    if occurred.tzinfo is None:
        raise CommandV2Refused("v3 command transition time lacks timezone")
    command = validate_command(row["command"])
    result = worker_lifecycle.validate_command_result_v2(row["result"])
    if (row["campaign_id"] != command["campaign_id"]
            or row["config_digest"] != command["config_digest"]
            or row["config_generation"] != command["config_generation"]
            or row["control_revision"] != command["expected_control_revision"] + 1
            or result["control_revision"] != row["control_revision"]
            or any(result[name] != command[name]
                   for name in ("request_id", "operation", "payload_digest"))):
        raise CommandV2Refused("v3 command transition binding differs")
    if row["phase"] == "ACCEPTED" \
            and row["supervisor_incarnation"] != command["supervisor_incarnation"]:
        raise CommandV2Refused("v3 acceptance writer differs from command supervisor")
    if row["phase"] == "COMPLETED" and not result["completed"]:
        raise CommandV2Refused("v3 completion transition is incomplete")
    row["command"], row["result"] = command, result
    return row


__all__ = ["COMMAND_FIELDS", "COMMAND_SCHEMA", "CommandV2Refused",
           "TRANSITION_SCHEMA", "command_digest", "validate_command",
           "validate_transition"]
