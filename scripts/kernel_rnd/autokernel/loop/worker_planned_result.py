"""Planned worker result record and helpers, split verbatim from unified_worker.py (code-budget split)."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any
from . import experiment_plan as ep
from . import observation_binding as ob
from . import planned_serving as ps
from . import planned_unit_selection as selection


RESULT_SCHEMA = "epyc.autokernel.planned_worker_result.v1"


RESULT_SCHEMA_V2 = "epyc.autokernel.planned_worker_result.v2"


RESULT_SCHEMA_V3 = "epyc.autokernel.planned_worker_result.v3"


class WorkerBridgeRefused(RuntimeError):
    """The closed request, live authority, placement, or result boundary was refused."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise WorkerBridgeRefused("objects require non-empty text keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise WorkerBridgeRefused("value is not finite canonical JSON")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise WorkerBridgeRefused(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise WorkerBridgeRefused(f"{label} must be lowercase SHA-256")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise WorkerBridgeRefused(f"{label} must be a positive integer")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WorkerBridgeRefused(f"{label} has missing/unknown fields")
    return dict(value)


@dataclass(frozen=True)
class PlannedWorkerResult:
    body: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "PlannedWorkerResult":
        schema = value.get("schema") if isinstance(value, Mapping) else None
        fields = {"schema", "nonce", "prepared_digest", "plan_digest",
            "lineage_id", "stage_id", "worker_id", "worker_generation", "grant_id",
            "grant_generation", "container_id", "completed_unit_ids", "run", "captures",
            "result_digest"}
        if schema in {RESULT_SCHEMA_V2, RESULT_SCHEMA_V3}:
            fields |= {"lifecycle_observation_references"}
        if schema == RESULT_SCHEMA_V3:
            fields |= {"selected_range"}
        row = _exact(value, fields, "planned worker result")
        if row["schema"] not in {RESULT_SCHEMA, RESULT_SCHEMA_V2, RESULT_SCHEMA_V3}:
            raise WorkerBridgeRefused("planned worker result schema is unsupported")
        supplied = _sha(row.pop("result_digest"), "result_digest")
        if supplied != _digest(row):
            raise WorkerBridgeRefused("planned worker result digest mismatch")
        for name in ("nonce", "lineage_id", "stage_id", "worker_id", "grant_id",
                     "container_id"):
            _text(row[name], name)
        for name in ("prepared_digest", "plan_digest"):
            _sha(row[name], name)
        for name in ("worker_generation", "grant_generation"):
            _positive(row[name], name)
        completed = row["completed_unit_ids"]
        if (not isinstance(completed, (list, tuple))
                or any(not isinstance(item, str) or not item for item in completed)
                or len(set(completed)) != len(completed)):
            raise WorkerBridgeRefused("completed_unit_ids must be unique text")
        try:
            run_fields = {"schema", "plan_digest", "prompt_manifest_digest",
                "lineage_id", "anchor_identity", "candidate_identity", "raw_units",
                "admissible_view", "use_status", "execution_complete", "paused_reason",
                "capture_receipts"}
            if schema in {RESULT_SCHEMA_V2, RESULT_SCHEMA_V3}:
                run_fields |= {"lifecycle_observation_references"}
            if schema == RESULT_SCHEMA_V3:
                run_fields |= {"selected_range", "selected_range_complete"}
            run_row = _exact(row["run"], run_fields, "planned serving run")
            expected_run_schema = {RESULT_SCHEMA: ps.RUN_SCHEMA,
                RESULT_SCHEMA_V2: ps.RUN_SCHEMA_V2, RESULT_SCHEMA_V3: ps.RUN_SCHEMA_V3}[schema]
            if run_row["schema"] != expected_run_schema:
                raise WorkerBridgeRefused("planned serving run schema is unsupported")
            if (run_row["plan_digest"] != row["plan_digest"]
                    or run_row["lineage_id"] != row["lineage_id"]
                    or type(run_row["execution_complete"]) is not bool):
                raise WorkerBridgeRefused("planned serving run binding differs from result")
            for raw in run_row["raw_units"]:
                ep.RawUnit.from_dict(raw)
            captures = row["captures"]
            if (not isinstance(captures, (list, tuple)) or len(captures) > 2):
                raise WorkerBridgeRefused("result captures must contain at most two arms")
            for item in captures:
                capture = _exact(item, {"measurement_id", "payload", "payload_digest",
                                        "artifact"}, "deferred capture")
                if (_sha(capture["measurement_id"], "measurement_id")
                        != capture["payload"].get("measurement_id")
                        or _sha(capture["payload_digest"], "payload_digest")
                        != _digest(capture["payload"])):
                    raise WorkerBridgeRefused("deferred capture digest/binding mismatch")
            if schema == RESULT_SCHEMA_V3:
                selected = selection.SelectedPlanUnitRange.from_dict(row["selected_range"])
                if (selected.plan_digest != row["plan_digest"]
                        or len(selected.unit_ids) != 1
                        or _plain(run_row["selected_range"]) != selected.to_dict()
                        or list(completed) != list(selected.unit_ids[:len(completed)])
                        or type(run_row["selected_range_complete"]) is not bool
                        or run_row["selected_range_complete"] != (
                            [raw["unit_id"] for raw in run_row["raw_units"]] == list(selected.unit_ids)
                            and all(raw["terminal"] for raw in run_row["raw_units"]))):
                    raise WorkerBridgeRefused("selected result range/completed units differ")
            if schema in {RESULT_SCHEMA_V2, RESULT_SCHEMA_V3}:
                references = [ob.LifecycleObservationReference.from_dict(
                                  _plain(item)).to_dict()
                              for item in row["lifecycle_observation_references"]]
                if (references != _plain(run_row["lifecycle_observation_references"])
                        or len(references) not in {len(completed), len(completed) + 1}
                        or len({item["unit_id"] for item in references}) != len(references)
                        or [item["unit_id"] for item in references[:len(completed)]]
                           != list(completed)):
                    raise WorkerBridgeRefused(
                        "v2 result observation references differ from fixed completed units")
        except WorkerBridgeRefused:
            raise
        except Exception as exc:
            raise WorkerBridgeRefused(f"planned worker result payload is invalid: {exc}") from exc
        row["result_digest"] = supplied
        return cls(_freeze(row))

    @property
    def result_digest(self) -> str:
        return self.body["result_digest"]

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.body)
