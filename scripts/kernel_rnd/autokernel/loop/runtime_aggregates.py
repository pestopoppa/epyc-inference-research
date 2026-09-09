"""Bounded diagnostic carrier; no execution, evidence or qualification authority."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from types import MappingProxyType
from typing import Mapping

MAX_ROWS = 64
MAX_BYTES = 32768
ROW_LIMITS = {"profile": 24, "calibration": 24, "actor": 16}
KINDS = ("evidence", "profile", "calibration", "actor")
HEADER = frozenset({"schema", "status", "reason", "observed_at", "attempted_at", "generation", "error", "data"})
BUDGETS = frozenset({"actor_calls_per_target", "patch_repairs_per_target", "provider_seconds_per_target",
                    "resource_failures_per_target", "contamination_events_per_target", "actor_calls_per_campaign"})
FIELDS = {
    "evidence": frozenset({"reader_id", "epoch", "owner_state", "ready", "readiness", "source_frontier",
        "cursor_frontier", "projection_frontier", "admission_frontier", "last_admitted_frontier",
        "projection_checksum", "proof_pending", "lag_events", "lag_seconds", "quarantine_count", "cached_finding_count"}),
    "profile": frozenset({"configured_count", "usable_count", "mechanism_count", "debt_count",
        "planning_observed_at", "items", "items_total", "items_truncated"}),
    "calibration": frozenset({"request_count", "pending_count", "collected_count", "exhausted_count",
        "failed_count", "contaminated_count", "qualification", "ranking_authorized", "items", "items_total", "items_truncated"}),
    "actor": frozenset({"pending_count", "finished_count", "backend_count", "event_count", "executor_installed",
        "reserved", "spent", "items", "items_total", "items_truncated"}),
}
ITEMS = {
    "profile": frozenset({"target_revision", "profile_digest", "transition_id", "available_at_planning",
        "settled", "remaining_seconds", "clock_known", "consumed_request_debt"}),
    "calibration": frozenset({"request_digest", "chunk_digest", "outcome"}),
    "actor": frozenset({"request_digest", "target_revision", "transition_id", "phase", "settlement_outcome",
        "retry_remaining_seconds", "clock_known"}),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def source_identity():
    """Loaded codec implementation and closed constants, never runtime cache values."""
    from . import lifecycle_observation as lo
    functions = (utc_now, plain, freeze, _text, _date, _number, _scalar,
                 validate, observation, validate_bundle, source_identity)
    return {"functions": [lo.callable_identity(item) for item in functions],
            "constants": {"max_rows": MAX_ROWS, "max_bytes": MAX_BYTES, "row_limits": ROW_LIMITS,
                "kinds": list(KINDS), "header": sorted(HEADER), "budgets": sorted(BUDGETS),
                "fields": {key: sorted(value) for key, value in FIELDS.items()},
                "items": {key: sorted(value) for key, value in ITEMS.items()}}}


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def _text(value, *, nullable=False, maximum=512):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not 0 < len(value) <= maximum:
        raise ValueError("aggregate text is missing or exceeds its bound")


def _date(value):
    if value is None:
        return
    _text(value, maximum=64)
    if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("aggregate date lacks timezone")


def _number(value, *, integer=False, nullable=False):
    if nullable and value is None:
        return
    if (type(value) not in ((int,) if integer else (int, float)) or
            not 0 <= value <= 2**63 - 1 or not math.isfinite(value)):
        raise ValueError("aggregate numeric value is invalid")


def _scalar(key, value):
    if key.endswith(("_at",)):
        _date(value)
    elif key.endswith(("_count", "_frontier")) or key in {"items_total", "lag_events"}:
        _number(value, integer=True, nullable=True)
    elif key.endswith("seconds"):
        _number(value, nullable=True)
    elif key.endswith(("_digest", "_revision")) or key in {"transition_id", "projection_checksum"}:
        if value is not None and (not isinstance(value, str) or len(value) != 64
                                 or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("aggregate identity is not SHA-256")
    elif key in {"ready", "proof_pending", "items_truncated", "ranking_authorized", "executor_installed",
                 "available_at_planning", "settled", "clock_known", "consumed_request_debt"}:
        if type(value) is not bool and not (key == "consumed_request_debt" and value is None):
            raise ValueError("aggregate flag is invalid")
    else:
        _text(value, nullable=True)


def validate(kind, value):
    if kind not in KINDS or not isinstance(value, Mapping) or set(value) != HEADER:
        raise ValueError("aggregate header fields differ")
    if value["schema"] != f"epyc.autokernel.{kind}_observation.v1":
        raise ValueError("aggregate schema differs")
    if value["status"] not in ("available", "unknown", "not_connected"):
        raise ValueError("aggregate status differs")
    _text(value["reason"])
    _text(value["error"], nullable=True)
    _date(value["observed_at"])
    _date(value["attempted_at"])
    _number(value["generation"], integer=True)
    if value["error"] is not None and value["status"] != "unknown":
        raise ValueError("aggregate publication error must remain unknown")
    data = value["data"]
    if data is None:
        if value["status"] == "available" or value["observed_at"] is not None:
            raise ValueError("missing aggregate cannot be available or dated")
    else:
        if not isinstance(data, Mapping) or set(data) != FIELDS[kind] or value["observed_at"] is None:
            raise ValueError("aggregate data fields/date differ")
        for key, item in data.items():
            if key == "items":
                if not isinstance(item, (tuple, list)) or len(item) > ROW_LIMITS[kind]:
                    raise ValueError("aggregate row bound exceeded")
                for row in item:
                    if not isinstance(row, Mapping) or set(row) != ITEMS[kind]:
                        raise ValueError("aggregate item fields differ")
                    for name, part in row.items():
                        _scalar(name, part)
            elif key in {"reserved", "spent"}:
                if not isinstance(item, Mapping) or set(item) != BUDGETS:
                    raise ValueError("aggregate budget keys differ")
                for part in item.values():
                    _number(part)
            else:
                _scalar(key, item)
        if "items" in data and (type(data["items_total"]) is not int or data["items_total"] < len(data["items"]) or
                data["items_truncated"] != (data["items_total"] > len(data["items"]))):
            raise ValueError("aggregate sample completeness differs")
        if kind == "calibration" and (data["qualification"] != "unavailable" or data["ranking_authorized"]):
            raise ValueError("calibration observation cannot confer qualification")
        if kind == "evidence" and data["lag_seconds"] is not None:
            raise ValueError("event frontier cannot invent time lag")
        if kind == "evidence":
            if (data["owner_state"] not in {"ready", "pending", "failed", "closed"}
                    or data["readiness"] not in {"unknown", "projected", "outage"}
                    or data["ready"] != (data["owner_state"] == "ready")
                    or (value["status"] == "available" and not data["ready"])):
                raise ValueError("evidence readiness state differs")
        if kind == "profile" and data["planning_observed_at"] != value["observed_at"]:
            raise ValueError("profile reduction timestamp differs")
        for item in data.get("items", ()):
            if kind == "calibration" and item["outcome"] not in {None, "calibration", "invalid", "failed"}:
                raise ValueError("calibration observation outcome differs")
            if kind == "actor" and (item["phase"] not in {"pending", "settled", "finished_unsettled"}
                    or item["settlement_outcome"] not in {None, "prerequisite", "failed", "invalid"}
                    or (item["phase"] == "settled") != (item["settlement_outcome"] is not None)):
                raise ValueError("actor sampled settlement state differs")
            seconds = "remaining_seconds" if kind == "profile" else "retry_remaining_seconds"
            if kind in {"profile", "actor"} and not item["clock_known"] and item[seconds] is not None:
                raise ValueError("unknown clock cannot supply validity remainder")
    if value["observed_at"] and value["attempted_at"] and datetime.fromisoformat(
            value["observed_at"].replace("Z", "+00:00")) > datetime.fromisoformat(
                value["attempted_at"].replace("Z", "+00:00")):
        raise ValueError("aggregate observation is after its attempt")
    result = plain(value)
    if len(json.dumps(result, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()) > MAX_BYTES:
        raise ValueError("aggregate byte bound exceeded")
    return result


def observation(kind, *, status="unknown", reason="owner has not reported", data=None,
                observed_at=None, attempted_at=None, generation=0, error=None):
    return freeze(validate(kind, {"schema": f"epyc.autokernel.{kind}_observation.v1",
        "status": status, "reason": reason[:512], "observed_at": observed_at,
        "attempted_at": attempted_at, "generation": generation,
        "error": None if error is None else str(error)[:512], "data": data}))


def validate_bundle(value):
    if not isinstance(value, Mapping) or set(value) != set(KINDS):
        raise ValueError("aggregate bundle kinds differ")
    result = {kind: validate(kind, value[kind]) for kind in KINDS}
    if sum(len((row["data"] or {}).get("items", ())) for row in result.values()) > MAX_ROWS:
        raise ValueError("combined aggregate row bound exceeded")
    if len(json.dumps(result, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()) > MAX_BYTES:
        raise ValueError("combined aggregate byte bound exceeded")
    return result
