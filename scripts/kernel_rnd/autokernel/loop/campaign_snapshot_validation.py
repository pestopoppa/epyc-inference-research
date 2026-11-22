"""Campaign snapshot v2/v3 validation, split verbatim from campaign_control.py (code-budget split)."""
from __future__ import annotations
import copy
from datetime import datetime
import math
from typing import Any, Mapping
from . import runtime_aggregates as aggregates
from . import scheduling
from . import worker_lifecycle as worker_lifecycle_module


SNAPSHOT_SCHEMA_V2 = "epyc.autokernel.campaign_snapshot.v2"


SNAPSHOT_SCHEMA_V3 = "epyc.autokernel.campaign_snapshot.v3"


UNIFIED_PROJECTION_SCHEMA = "epyc.autokernel.unified_campaign_projection.v1"


UNIFIED_PROJECTION_SCHEMA_V2 = "epyc.autokernel.unified_campaign_projection.v2"


UNIFIED_PROJECTION_SCHEMA_V3 = "epyc.autokernel.unified_campaign_projection.v3"


UNIFIED_PROJECTION_SCHEMA_V4 = "epyc.autokernel.unified_campaign_projection.v4"


RUNTIME_OBSERVATION_FIELDS = frozenset({
    "status", "reason", "reason_truncated", "observed_at", "observation_sequence",
    "retry_after_seconds", "work_kind", "target_revision", "transition_id",
    "settlement_outcome", "installed_work_kinds", "publication_error"})


SNAPSHOT_V2_FIELDS = frozenset({
    "schema", "producer_build", "producer_schema", "campaign_id",
    "config_generation", "config_digest", "requested_manifest_digest",
    "supervisor_incarnation", "stream_epoch", "sequence", "journal_cursor",
    "control_revision", "generated_at", "desired_state", "observed_state",
    "command_results", "active_worker", "producer_heartbeat_at",
    "last_scientific_result_at", "worker_activity_at", "execution_authorized",
    "execution_capability_available", "worker_lifecycle_revision",
    "prerequisite_reason",
})


SNAPSHOT_V3_FIELDS = SNAPSHOT_V2_FIELDS | {"unified"}


ACTIVE_WORKER_V2_FIELDS = frozenset({
    "worker_id", "worker_generation", "request_id", "plan_digest", "lineage_id",
    "stage_id", "state", "grant_id", "grant_generation", "container_id",
    "provider_deadline", "deadline_clock_domain", "control_revision", "started_at",
    "activity_at", "termination_deadline", "unresolved_reason",
})


class ControlRefused(RuntimeError):
    """A control or supervisor transition failed closed."""


def validate_snapshot_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SNAPSHOT_V2_FIELDS:
        raise ControlRefused("v2 snapshot has missing/unknown fields")
    row = copy.deepcopy(dict(value))
    if row["schema"] != SNAPSHOT_SCHEMA_V2 or row["producer_schema"] != SNAPSHOT_SCHEMA_V2:
        raise ControlRefused("v2 snapshot schema identity is invalid")
    for name in ("campaign_id", "requested_manifest_digest"):
        if not isinstance(row[name], str) or not row[name]:
            raise ControlRefused(f"v2 snapshot {name} is invalid")
    for name in ("config_digest", "requested_manifest_digest"):
        if (not isinstance(row[name], str) or len(row[name]) != 64
                or any(char not in "0123456789abcdef" for char in row[name])):
            raise ControlRefused(f"v2 snapshot {name} must be lowercase SHA-256")
    for name, minimum in (("config_generation", 1), ("supervisor_incarnation", 1),
                          ("stream_epoch", 1), ("sequence", 1),
                          ("journal_cursor", 0), ("control_revision", 0)):
        if (not isinstance(row[name], int) or isinstance(row[name], bool)
                or row[name] < minimum):
            raise ControlRefused(f"v2 snapshot {name} is invalid")
    for name in ("generated_at", "producer_heartbeat_at"):
        if not isinstance(row[name], str):
            raise ControlRefused(f"v2 snapshot {name} is invalid")
        try:
            parsed = datetime.fromisoformat(row[name].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ControlRefused(f"v2 snapshot {name} is not ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ControlRefused(f"v2 snapshot {name} lacks timezone")
    for name in ("worker_activity_at", "last_scientific_result_at"):
        if row[name] is not None:
            if not isinstance(row[name], str):
                raise ControlRefused(f"v2 snapshot {name} is invalid")
            try:
                parsed = datetime.fromisoformat(row[name].replace("Z", "+00:00"))
            except ValueError as exc:
                raise ControlRefused(f"v2 snapshot {name} is not ISO-8601") from exc
            if parsed.tzinfo is None:
                raise ControlRefused(f"v2 snapshot {name} lacks timezone")
    if row["desired_state"] not in {"paused", "running", "drained"}:
        raise ControlRefused("v2 snapshot desired_state is invalid")
    if row["observed_state"] not in {
            "paused", "running", "drained", "pausing", "draining",
            "waiting_prerequisite", "ownership_unresolved"}:
        raise ControlRefused("v2 snapshot observed_state is invalid")
    if row["prerequisite_reason"] is not None \
            and (not isinstance(row["prerequisite_reason"], str)
                 or not row["prerequisite_reason"]):
        raise ControlRefused("v2 snapshot prerequisite_reason is invalid")
    build = row["producer_build"]
    build_fields = {"schema", "scope", "module", "identity_basis", "included_symbols",
                    "excluded_scope", "sha256"}
    if (not isinstance(build, Mapping) or set(build) != build_fields
            or not isinstance(build["included_symbols"], list)
            or not isinstance(build["excluded_scope"], list)
            or any(not isinstance(item, str) or not item
                   for item in [build["schema"], build["scope"], build["module"],
                                build["identity_basis"], *build["included_symbols"],
                                *build["excluded_scope"]])
            or not isinstance(build["sha256"], str) or len(build["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in build["sha256"])):
        raise ControlRefused("v2 snapshot producer_build is invalid")
    if type(row["execution_authorized"]) is not bool \
            or type(row["execution_capability_available"]) is not bool:
        raise ControlRefused("v2 snapshot execution flags must be boolean")
    if (not isinstance(row["worker_lifecycle_revision"], int)
            or isinstance(row["worker_lifecycle_revision"], bool)
            or row["worker_lifecycle_revision"] < 0):
        raise ControlRefused("v2 worker lifecycle revision is invalid")
    if not isinstance(row["command_results"], list):
        raise ControlRefused("v2 command_results must be a list")
    row["command_results"] = [
        worker_lifecycle_module.validate_command_result_v2(item)
        for item in row["command_results"]]
    active = row["active_worker"]
    if active is not None:
        if not isinstance(active, Mapping) or set(active) != ACTIVE_WORKER_V2_FIELDS:
            raise ControlRefused("v2 active_worker has missing/unknown fields")
        for name in ("worker_id", "request_id", "plan_digest", "lineage_id", "stage_id",
                     "state", "grant_id", "container_id", "deadline_clock_domain",
                     "started_at"):
            if not isinstance(active[name], str) or not active[name]:
                raise ControlRefused(f"v2 active_worker {name} is invalid")
        if (len(active["plan_digest"]) != 64
                or any(char not in "0123456789abcdef" for char in active["plan_digest"])):
            raise ControlRefused("v2 active_worker plan_digest is invalid")
        if active["state"] not in {
                "intent", "container_created", "child_captured", "exec_release_intent",
                "executing", "result_retained", "terminal_pending_result",
                "tearing_down", "teardown_failed", "unresolved"}:
            raise ControlRefused("v2 active_worker state is invalid")
        for name in ("worker_generation", "grant_generation"):
            if not isinstance(active[name], int) or isinstance(active[name], bool) \
                    or active[name] < 1:
                raise ControlRefused(f"v2 active_worker {name} is invalid")
        if not isinstance(active["control_revision"], int) \
                or isinstance(active["control_revision"], bool) \
                or active["control_revision"] < 0:
            raise ControlRefused("v2 active_worker control_revision is invalid")
        for name in ("provider_deadline", "termination_deadline"):
            if active[name] is not None and (not isinstance(active[name], (int, float))
                    or isinstance(active[name], bool) or not math.isfinite(active[name])):
                raise ControlRefused(f"v2 active_worker {name} is invalid")
        for name in ("activity_at", "unresolved_reason"):
            if active[name] is not None \
                    and (not isinstance(active[name], str) or not active[name]):
                raise ControlRefused(f"v2 active_worker {name} is invalid")
        for name in ("started_at", "activity_at"):
            if active[name] is not None:
                try:
                    parsed = datetime.fromisoformat(active[name].replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ControlRefused(
                        f"v2 active_worker {name} is not ISO-8601") from exc
                if parsed.tzinfo is None:
                    raise ControlRefused(f"v2 active_worker {name} lacks timezone")
        if row["worker_activity_at"] != active["activity_at"]:
            raise ControlRefused("v2 snapshot worker activity clocks disagree")
        if row["observed_state"] in {"paused", "drained"}:
            raise ControlRefused(
                "v2 snapshot cannot certify quiescence with an active worker")
        row["active_worker"] = dict(active)
    elif row["worker_activity_at"] is not None or row["execution_authorized"]:
        raise ControlRefused("v2 snapshot has worker activity/authority without active worker")
    return row


def validate_runtime_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != RUNTIME_OBSERVATION_FIELDS:
        raise ControlRefused("runtime observation fields differ")
    row = copy.deepcopy(dict(value))
    kinds = {"runtime_comparison", "actor_preparation", "profile_preparation",
             "calibration_preparation"}
    if (not isinstance(row["status"], str)
            or row["status"] not in {"not_reported", "recovered", "waiting", "settled",
                              "stopped", "recovery_required"}
            or not isinstance(row["reason"], str) or not 0 < len(row["reason"]) <= 4096
            or type(row["reason_truncated"]) is not bool
            or type(row["observation_sequence"]) is not int or row["observation_sequence"] < 0
            or not isinstance(row["installed_work_kinds"], list)
            or any(not isinstance(kind, str) or kind not in kinds for kind in row["installed_work_kinds"])
            or len(set(row["installed_work_kinds"])) != len(row["installed_work_kinds"])):
        raise ControlRefused("runtime observation status/bounds differ")
    error = row["publication_error"]
    if error is not None and (not isinstance(error, str) or not 0 < len(error) <= 4096):
        raise ControlRefused("runtime publication error is invalid")
    delay = row["retry_after_seconds"]
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ControlRefused("runtime observation retry delay is invalid")
    missing = row["status"] == "not_reported"
    if (missing != (row["observed_at"] is None) or (row["observation_sequence"] == 0 and not missing)
            or missing and row["observation_sequence"] > 0 and error is None):
        raise ControlRefused("runtime observation date/sequence differs")
    if not missing:
        try:
            stamp = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00"))
        except (ValueError, AttributeError) as exc:
            raise ControlRefused("runtime observation date is invalid") from exc
        if stamp.tzinfo is None:
            raise ControlRefused("runtime observation date lacks timezone")
    for name in ("target_revision", "transition_id"):
        item = row[name]
        if item is not None and (not isinstance(item, str) or len(item) != 64
                                or any(c not in "0123456789abcdef" for c in item)):
            raise ControlRefused("runtime observation identity is invalid")
    selected = row["work_kind"] is not None
    if (selected and (not isinstance(row["work_kind"], str) or row["work_kind"] not in kinds)
            or selected != (row["target_revision"] is not None)
            or selected != (row["transition_id"] is not None)
            or missing and (selected or delay != 0 or row["reason_truncated"])
            or row["settlement_outcome"] is not None
            and (not isinstance(row["settlement_outcome"], str)
                 or row["settlement_outcome"] not in scheduling.OUTCOMES)
            or (row["status"] == "settled") != (row["settlement_outcome"] is not None)
            or row["status"] == "settled" and not selected):
        raise ControlRefused("runtime observation work/settlement differs")
    return row


def validate_snapshot_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SNAPSHOT_V3_FIELDS:
        raise ControlRefused("v3 snapshot has missing/unknown fields")
    row = copy.deepcopy(dict(value))
    if isinstance(row.get("unified"), Mapping) and row["unified"].get("schema") == UNIFIED_PROJECTION_SCHEMA_V4:
        unified, legacy = row["unified"], copy.deepcopy(row)  # v4 = v3 + owner-derived resource/candidate
        try:
            resources = aggregates.validate_resources(unified.get("resources"))
            aggregates.validate_candidate(unified.get("candidate"))
            if any(resources["used"][key] != unified["scheduler"]["accounting"][key]
                   for key in ("receipt_count", "view_digest")):
                raise ValueError("resource usage differs from scheduler accounting")
            for part, key in (("requested", "occurred_at"), ("granted", "recorded_at")):
                if resources[part] is not None and datetime.fromisoformat(resources[part][key].replace("Z", "+00:00")) \
                        > datetime.fromisoformat(row["generated_at"].replace("Z", "+00:00")):
                    raise ValueError("resource record is newer than snapshot")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ControlRefused(str(exc)) from exc
        legacy["unified"].update(schema=UNIFIED_PROJECTION_SCHEMA_V3, resources={
            "schema": "epyc.autokernel.unified_resource_status.v1", "status": "not_connected", "reason": "v4 carrier",
            "requested": None, "granted": None, "held": None, "used": None}, candidate={
            "schema": "epyc.autokernel.unified_candidate_status.v1", "status": "not_connected", "reason": "v4 carrier",
            "accumulated_identity": None, "validated_identity": None, "frozen_production_identity": None,
            "validation_debt": None})
        validate_snapshot_v3(legacy)
        return row
    if isinstance(row.get("unified"), Mapping) and row["unified"].get("schema") == UNIFIED_PROJECTION_SCHEMA_V3:
        unified = row["unified"]
        if set(unified) != {"schema", "scheduler", "resources", "actors", "evidence", "candidate",
                            "targets", "runtime", "worker_timing"}:
            raise ControlRefused("aggregate unified projection fields differ")
        actors = unified["actors"]
        if (not isinstance(actors, Mapping) or set(actors) != {"schema", "status", "reason", "actor", "profile", "calibration"}
                or actors["schema"] != "epyc.autokernel.unified_actor_status.v2"
                or actors["status"] not in ("available", "unknown")
                or not isinstance(actors["reason"], str) or not 0 < len(actors["reason"]) <= 512):
            raise ControlRefused("aggregate preparation wrapper differs")
        # Validate unchanged scalar/time/worker fields before using their types.
        legacy = copy.deepcopy(row)
        legacy["unified"]["schema"] = UNIFIED_PROJECTION_SCHEMA_V2
        legacy["unified"]["actors"] = {"schema": "epyc.autokernel.unified_actor_status.v1",
            "status": "not_connected", "reason": "legacy validation carrier", "items": [],
            "clock_semantics": "UTC wall-clock projection; runtime fences remain monotonic"}
        legacy["unified"]["evidence"] = {"schema": "epyc.autokernel.unified_evidence_status.v1",
            "status": "not_connected", "reason": "legacy validation carrier", "frontier_digest": None, "lag_seconds": None}
        validate_snapshot_v3(legacy)
        try:
            aggregate_rows = aggregates.validate_bundle({"evidence": unified["evidence"],
                **{kind: actors[kind] for kind in ("actor", "profile", "calibration")}})
            if actors["status"] != ("unknown" if any(aggregate_rows[kind]["status"] == "unknown"
                    for kind in ("actor", "profile", "calibration")) else "available"):
                raise ValueError("preparation aggregate status contradicts observations")
            for item in aggregate_rows.values():
                for name in ("observed_at", "attempted_at"):
                    if item[name] is not None and datetime.fromisoformat(item[name].replace("Z", "+00:00")) > \
                            datetime.fromisoformat(row["generated_at"].replace("Z", "+00:00")):
                        raise ValueError("aggregate observation is newer than snapshot")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ControlRefused(str(exc)) from exc
        return row
    unified = row.pop("unified")
    base = dict(row)
    base["schema"] = SNAPSHOT_SCHEMA_V2
    base["producer_schema"] = SNAPSHOT_SCHEMA_V2
    validate_snapshot_v2(base)
    if row["schema"] != SNAPSHOT_SCHEMA_V3 or row["producer_schema"] != SNAPSHOT_SCHEMA_V3:
        raise ControlRefused("v3 snapshot schema identity is invalid")
    expected = {"schema", "scheduler", "resources", "actors", "evidence",
                "candidate", "targets"}
    runtime_aware = isinstance(unified, Mapping) and unified.get("schema") == UNIFIED_PROJECTION_SCHEMA_V2
    if runtime_aware:
        expected.update({"runtime", "worker_timing"})
    if not isinstance(unified, Mapping) or set(unified) != expected \
            or unified.get("schema") not in {UNIFIED_PROJECTION_SCHEMA, UNIFIED_PROJECTION_SCHEMA_V2}:
        raise ControlRefused("v3 unified projection fields/schema differ")
    if runtime_aware:
        unified["runtime"] = validate_runtime_observation(unified["runtime"])
        stamp = unified["runtime"]["observed_at"]
        if stamp is not None and datetime.fromisoformat(stamp.replace("Z", "+00:00")) > \
                datetime.fromisoformat(row["generated_at"].replace("Z", "+00:00")):
            raise ControlRefused("runtime observation is newer than its snapshot")
        timing, worker = unified["worker_timing"], row["active_worker"]
        if worker is None:
            if timing is not None:
                raise ControlRefused("runtime worker timing has no original worker")
        else:
            fields = {"worker_id", "worker_generation", "lifecycle_revision", "checked_at",
                      "state", "remaining_seconds"}
            if (not isinstance(timing, Mapping) or set(timing) != fields
                    or timing["worker_id"] != worker["worker_id"]
                    or type(timing["worker_generation"]) is not int
                    or timing["worker_generation"] != worker["worker_generation"]
                    or type(timing["lifecycle_revision"]) is not int
                    or timing["lifecycle_revision"] != row["worker_lifecycle_revision"]
                    or timing["checked_at"] != row["generated_at"]
                    or timing["state"] not in ("within_deadline", "deadline_elapsed", "clock_unavailable")):
                raise ControlRefused("runtime worker timing identity differs")
            remaining = timing["remaining_seconds"]
            if timing["state"] == "clock_unavailable":
                if remaining is not None:
                    raise ControlRefused("unknown worker clock cannot supply a remainder")
            elif (type(remaining) not in (int, float) or not math.isfinite(remaining)
                  or remaining < 0 or (remaining > 0) != (timing["state"] == "within_deadline")):
                raise ControlRefused("runtime worker timing remainder differs")
    closed = {
        "resources": {"schema", "status", "reason", "requested", "granted", "held", "used"},
        "actors": {"schema", "status", "reason", "clock_semantics", "items"},
        "evidence": {"schema", "status", "reason", "frontier_digest", "lag_seconds"},
        "candidate": {"schema", "status", "reason", "accumulated_identity",
                      "validated_identity", "frozen_production_identity", "validation_debt"},
        "targets": {"schema", "status", "reason", "total", "ready", "prerequisite",
                    "production_enrolled", "seed_enrolled", "items_page_ref"},
    }
    nested_schemas = {
        "resources": "epyc.autokernel.unified_resource_status.v1",
        "actors": "epyc.autokernel.unified_actor_status.v1",
        "evidence": "epyc.autokernel.unified_evidence_status.v1",
        "candidate": "epyc.autokernel.unified_candidate_status.v1",
        "targets": "epyc.autokernel.unified_target_status.v1",
    }
    for name, fields in closed.items():
        item = unified.get(name)
        if (not isinstance(item, Mapping) or set(item) != fields
                or item.get("schema") != nested_schemas[name]):
            raise ControlRefused(f"v3 unified {name} fields differ")
        if item["status"] not in {"available", "unknown", "not_connected"}:
            raise ControlRefused(f"v3 unified {name} status is invalid")
        if not isinstance(item["reason"], str) or not item["reason"]:
            raise ControlRefused(f"v3 unified {name} reason is invalid")
    scheduler_row = unified.get("scheduler")
    scheduler_fields = {"schema", "projection_digest", "config_digest", "policy_digest",
                        "round_number", "accounting_epoch", "capacity", "pending_selection_digest",
                        "status", "reason", "campaign_attempts", "campaign_charged_seconds",
                        "accounting", "coverage_debt_count"}
    if not isinstance(scheduler_row, Mapping) or set(scheduler_row) != scheduler_fields:
        raise ControlRefused("v3 unified scheduler fields differ")
    if scheduler_row.get("schema") != "epyc.autokernel.unified_scheduler_projection.v1":
        raise ControlRefused("v3 unified scheduler schema differs")
    for name in ("projection_digest", "config_digest", "policy_digest"):
        value = scheduler_row[name]
        if not isinstance(value, str) or len(value) != 64 \
                or any(char not in "0123456789abcdef" for char in value):
            raise ControlRefused(f"v3 scheduler {name} is invalid")
    if scheduler_row["status"] not in {"available", "unknown", "not_connected"} \
            or not isinstance(scheduler_row["reason"], str) or not scheduler_row["reason"]:
        raise ControlRefused("v3 scheduler status/reason is invalid")
    try:
        scheduling.ResourceVector.from_dict(scheduler_row["capacity"])
        scheduling.AccountingView.from_dict(scheduler_row["accounting"])
    except Exception as exc:
        raise ControlRefused(f"v3 scheduler capacity/accounting is invalid: {exc}") from exc
    pending = scheduler_row["pending_selection_digest"]
    if pending is not None and (not isinstance(pending, str) or len(pending) != 64
                                or any(char not in "0123456789abcdef" for char in pending)):
        raise ControlRefused("v3 scheduler pending selection digest is invalid")
    for name in ("round_number", "accounting_epoch", "campaign_attempts",
                 "coverage_debt_count"):
        if (not isinstance(scheduler_row[name], int) or isinstance(scheduler_row[name], bool)
                or scheduler_row[name] < 0):
            raise ControlRefused(f"v3 scheduler {name} is invalid")
    if (isinstance(scheduler_row["campaign_charged_seconds"], bool)
            or not isinstance(scheduler_row["campaign_charged_seconds"], (int, float))
            or not math.isfinite(scheduler_row["campaign_charged_seconds"])
            or scheduler_row["campaign_charged_seconds"] < 0):
        raise ControlRefused("v3 scheduler charged seconds is invalid")
    for name in ("total", "ready", "prerequisite", "production_enrolled", "seed_enrolled"):
        value = unified["targets"][name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ControlRefused(f"v3 targets {name} is invalid")
    if unified["targets"]["ready"] + unified["targets"]["prerequisite"] \
            != unified["targets"]["total"]:
        raise ControlRefused("v3 target counts disagree")
    if (unified["targets"]["production_enrolled"] > unified["targets"]["total"]
            or unified["targets"]["seed_enrolled"] > unified["targets"]["total"]
            or unified["targets"]["items_page_ref"] is not None):
        raise ControlRefused("v3 target enrollment counts/page reference are invalid")
    for name in ("resources", "actors", "evidence", "candidate"):
        item = unified[name]
        if item["status"] != "not_connected":
            raise ControlRefused(f"v3 {name} v1 supports only not_connected")
    resources = unified["resources"]
    if any(resources[name] is not None for name in ("requested", "granted", "held", "used")):
        raise ControlRefused("v3 disconnected resources must be null")
    actors = unified["actors"]
    if actors["clock_semantics"] != (
            "UTC wall-clock projection; runtime fences remain monotonic") \
            or not isinstance(actors["items"], list) or actors["items"]:
        raise ControlRefused("v3 disconnected actor projection is invalid")
    evidence = unified["evidence"]
    if evidence["frontier_digest"] is not None or evidence["lag_seconds"] is not None:
        raise ControlRefused("v3 disconnected evidence values must be null")
    candidate = unified["candidate"]
    if any(candidate[name] is not None for name in (
            "accumulated_identity", "validated_identity", "frozen_production_identity",
            "validation_debt")):
        raise ControlRefused("v3 disconnected candidate values must be null")
    row["unified"] = unified
    return row
