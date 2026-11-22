"""Campaign command validation, split verbatim from campaign_control.py (code-budget split)."""
from __future__ import annotations
import hashlib
import hmac
import json
from typing import Any, Mapping
from .campaign_snapshot_validation import ControlRefused


COMMAND_SCHEMA = "epyc.autokernel.campaign_command.v1"


OPERATIONS = frozenset({"pause", "resume", "drain"})


def command_digest(*, operation: str, payload: Mapping[str, Any], campaign_id: str,
                   config_generation: int) -> str:
    body = {"operation": operation, "payload": dict(payload),
            "campaign_id": campaign_id, "config_generation": config_generation}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_command(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlRefused("command must be an object")
    required = {"schema", "campaign_id", "config_generation", "request_id",
                "operation", "payload", "payload_digest", "expected_control_revision"}
    if set(value) != required:
        raise ControlRefused(
            f"command fields differ: missing={sorted(required-set(value))}, "
            f"unknown={sorted(set(value)-required)}")
    row = dict(value)
    if row["schema"] != COMMAND_SCHEMA:
        raise ControlRefused("unsupported command schema")
    for key in ("campaign_id", "request_id", "payload_digest"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ControlRefused(f"{key} must be a non-empty string")
    for key in ("config_generation", "expected_control_revision"):
        if (not isinstance(row[key], int) or isinstance(row[key], bool)
                or row[key] < (1 if key == "config_generation" else 0)):
            raise ControlRefused(f"{key} is invalid")
    if not isinstance(row["operation"], str) or row["operation"] not in OPERATIONS:
        raise ControlRefused("operation must be pause, resume, or drain")
    if not isinstance(row["payload"], Mapping) or row["payload"]:
        raise ControlRefused("payload must be an empty mapping in v1")
    expected = command_digest(operation=row["operation"], payload=row["payload"],
                              campaign_id=row["campaign_id"],
                              config_generation=row["config_generation"])
    if not hmac.compare_digest(row["payload_digest"], expected):
        raise ControlRefused("payload_digest does not match command semantics")
    row["payload"] = {}
    return row
