#!/usr/bin/env python3
"""Durable, fenced campaign controls; no worker or resource authority."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import stat
import types
import threading
import time
from typing import Any, Callable, Mapping

from .. import journal as journal_module, schemas
from ..controller.discovery_supervisor_secure import (
    RuntimeRoot, SecureRuntimeError, object_identity, read_stable_fd,
)
from . import status
from . import actor_preparation_state as actor_state_module
from . import campaign_command_v2
from . import maintenance_execution as maintenance_module
from . import native_retention_catalog as retention_catalog_module
from . import scheduling
from . import worker_lifecycle as worker_lifecycle_module
from .native_capture_control import (CurrentOwnerToken, NativeCaptureRefused,
                                     NativeCaptureValidator,
                                     PrevalidatedNativeCapture)
from .campaign import ResolvedCampaign

COMMAND_SCHEMA = "epyc.autokernel.campaign_command.v1"
SNAPSHOT_SCHEMA = "epyc.autokernel.campaign_snapshot.v1"
SNAPSHOT_SCHEMA_V2 = "epyc.autokernel.campaign_snapshot.v2"
SNAPSHOT_SCHEMA_V3 = "epyc.autokernel.campaign_snapshot.v3"
UNIFIED_PROJECTION_SCHEMA = "epyc.autokernel.unified_campaign_projection.v1"
DRIVER_SETTLEMENT_SCHEMA = "epyc.autokernel.unified_driver_settlement_request.v1"
DRIVER_SETTLEMENT_RECEIPT_SCHEMA = "epyc.autokernel.unified_driver_settlement_receipt.v1"
_MAX_WORKER_STDOUT_BYTES = 64 * 1024 * 1024
SNAPSHOT_FILE = "campaign-snapshot.json"
_CANDIDATE_REPLAYER_TOKEN = object()
OPERATIONS = frozenset({"pause", "resume", "drain"})
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


class DriverAdmissionClosed(ControlRefused):
    """A driver issue lost the race with a durable admission close."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_digest(*, operation: str, payload: Mapping[str, Any], campaign_id: str,
                   config_generation: int) -> str:
    body = {"operation": operation, "payload": dict(payload),
            "campaign_id": campaign_id, "config_generation": config_generation}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def resolved_config_digest(resolved: ResolvedCampaign) -> str:
    """Hash every frozen resolved identity, not merely the requested manifest."""
    return hashlib.sha256(json.dumps(
        resolved.to_dict(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


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
                "executing", "result_retained", "tearing_down", "teardown_failed",
                "unresolved"}:
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


def validate_snapshot_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SNAPSHOT_V3_FIELDS:
        raise ControlRefused("v3 snapshot has missing/unknown fields")
    row = copy.deepcopy(dict(value))
    unified = row.pop("unified")
    base = dict(row)
    base["schema"] = SNAPSHOT_SCHEMA_V2
    base["producer_schema"] = SNAPSHOT_SCHEMA_V2
    validate_snapshot_v2(base)
    if row["schema"] != SNAPSHOT_SCHEMA_V3 or row["producer_schema"] != SNAPSHOT_SCHEMA_V3:
        raise ControlRefused("v3 snapshot schema identity is invalid")
    expected = {"schema", "scheduler", "resources", "actors", "evidence",
                "candidate", "targets"}
    if not isinstance(unified, Mapping) or set(unified) != expected \
            or unified.get("schema") != UNIFIED_PROJECTION_SCHEMA:
        raise ControlRefused("v3 unified projection fields/schema differ")
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


@dataclass(frozen=True)
class TrustedGrant:
    identity: str
    generation: int
    deadline: float
    revoked: bool = False
    renewal_ok: bool = True


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str


def _stable_code_projection(code: types.CodeType) -> dict[str, Any]:
    """Return behavior-bearing code attributes without adaptive runtime state."""
    def constant(value: Any) -> Any:
        if isinstance(value, types.CodeType):
            return {"code": _stable_code_projection(value)}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [constant(item) for item in value]}
        if isinstance(value, frozenset):
            items = [constant(item) for item in value]
            return {"frozenset": sorted(
                items, key=lambda item: json.dumps(item, sort_keys=True))}
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return {"float": repr(value)}
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                "repr": repr(value)}

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "exceptiontable": code.co_exceptiontable.hex(),
        "consts": [constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _loaded_producer_build_identity() -> dict[str, Any]:
    """Hash loaded callable bytecode and selected constants, with exact scope."""
    digest = hashlib.sha256()
    included: list[str] = []

    def add_callable(symbol: str, function: types.FunctionType, kind: str) -> None:
        if function.__code__.co_filename == "<string>":
            kind = f"generated_{kind}"
        label = f"{kind}:{symbol}"
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(json.dumps(
            _stable_code_projection(function.__code__), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode())
        included.append(label)

    for name, value in sorted(globals().items()):
        if isinstance(value, types.FunctionType) and value.__module__ == __name__:
            add_callable(name, value, "module_function")
        elif isinstance(value, type) and value.__module__ == __name__:
            for member_name, member in sorted(vars(value).items()):
                symbol = f"{name}.{member_name}"
                if isinstance(member, types.FunctionType):
                    add_callable(symbol, member, "method")
                elif isinstance(member, staticmethod):
                    add_callable(symbol, member.__func__, "staticmethod")
                elif isinstance(member, classmethod):
                    add_callable(symbol, member.__func__, "classmethod")
                elif isinstance(member, property):
                    for accessor, function in (("fget", member.fget),
                                               ("fset", member.fset),
                                               ("fdel", member.fdel)):
                        if function is not None:
                            add_callable(f"{symbol}.{accessor}", function,
                                         f"property_{accessor}")
        elif name.isupper() and isinstance(value, (str, int, float, frozenset)):
            label = f"constant:{name}"
            digest.update(label.encode())
            digest.update(b"\0")
            encoded = repr(sorted(value)) if isinstance(value, frozenset) else repr(value)
            digest.update(encoded.encode())
            included.append(label)
    return {"schema": "epyc.autokernel.loaded_producer_build.v1",
            "scope": "campaign_control_callable_bytecode_and_selected_constants",
            "module": __name__,
            "identity_basis": "loaded_callable_bytecode_and_constants_sha256",
            "included_symbols": included,
            "excluded_scope": ["campaign_service_transport", "journal_validator",
                               "whole_package_launch_attestation"],
            "sha256": digest.hexdigest()}


def may_start_stage(*, desired_state: str, current_control_revision: int,
                    control_revision: int, current_supervisor_incarnation: int,
                    supervisor_incarnation: int, grant: TrustedGrant,
                    grant_identity: str, grant_generation: int, now: float,
                    max_stage_seconds: float, teardown_seconds: float,
                    dependency_check: Callable[[], bool | None]) -> AdmissionDecision:
    """Pure consumer fence for an injected trusted grant; creates no authority."""
    integers = ((current_control_revision, 0), (control_revision, 0),
                (current_supervisor_incarnation, 1), (supervisor_incarnation, 1))
    if any(not isinstance(value, int) or isinstance(value, bool) or value < minimum
           for value, minimum in integers):
        return AdmissionDecision(False, "invalid_fence_revision")
    if not isinstance(grant, TrustedGrant):
        return AdmissionDecision(False, "malformed_untrusted_grant")
    if not isinstance(grant_identity, str) or not grant_identity:
        return AdmissionDecision(False, "invalid_expected_grant_identity")
    if (not isinstance(grant_generation, int) or isinstance(grant_generation, bool)
            or grant_generation < 1):
        return AdmissionDecision(False, "invalid_expected_grant_generation")
    if (not isinstance(grant.generation, int) or isinstance(grant.generation, bool)
            or grant.generation < 1):
        return AdmissionDecision(False, "invalid_grant_identity")
    values = (now, max_stage_seconds, teardown_seconds, grant.deadline)
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(float(v)) for v in values):
        return AdmissionDecision(False, "invalid_time_bound")
    if max_stage_seconds <= 0 or teardown_seconds < 0:
        return AdmissionDecision(False, "invalid_time_bound")
    if desired_state != "running":
        return AdmissionDecision(False, f"admissions_closed:{desired_state}")
    if control_revision != current_control_revision:
        return AdmissionDecision(False, "stale_control_revision")
    if supervisor_incarnation != current_supervisor_incarnation:
        return AdmissionDecision(False, "stale_supervisor_incarnation")
    if not isinstance(grant.identity, str) or not grant.identity:
        return AdmissionDecision(False, "invalid_grant_identity")
    if not isinstance(grant.revoked, bool) or not isinstance(grant.renewal_ok, bool):
        return AdmissionDecision(False, "invalid_grant_state")
    if grant_identity != grant.identity or grant_generation != grant.generation:
        return AdmissionDecision(False, "stale_grant_identity_or_generation")
    if grant.revoked:
        return AdmissionDecision(False, "grant_revoked")
    if not grant.renewal_ok:
        return AdmissionDecision(False, "renewal_failed_future_admission")
    if grant.deadline - now < max_stage_seconds + teardown_seconds:
        return AdmissionDecision(False, "insufficient_grant_deadline")
    try:
        dependencies = dependency_check()
    except Exception:
        return AdmissionDecision(False, "dependencies_unknown:callback_failed")
    if dependencies is not True:
        return AdmissionDecision(False, "dependencies_unknown" if dependencies is None
                                 else "dependencies_unsatisfied")
    return AdmissionDecision(True, "admitted")


class _CandidateTransactionContext:
    """Ephemeral journal capability valid only under its controller's RLock."""

    def __init__(self, owner: "CampaignController", entries: tuple[Any, ...],
                 entry_offset: int) -> None:
        self._owner = owner
        self._active = True
        self._thread_id = threading.get_ident()
        self._lifetime_token = owner._lifetime_token
        self._context_token = owner._candidate_context_token
        self.entries = entries
        self.entry_offset = entry_offset
        self.store = owner.store
        self.campaign_id = owner.resolved.campaign_id
        self.config_generation = owner.config_generation
        self.config_digest = owner.config_digest
        self.supervisor_incarnation = owner.supervisor_incarnation

    def append(self, *, phase: str, transaction_id: str, operation: str,
               payload_digest: str, data: Mapping[str, Any]):
        if not self._active:
            raise ControlRefused("candidate transaction context is no longer active")
        if threading.get_ident() != self._thread_id:
            raise ControlRefused("candidate transaction context belongs to another thread")
        payload = {
            "schema": journal_module.CANDIDATE_TRANSACTION_SCHEMA,
            "phase": phase,
            "campaign_id": self.campaign_id,
            "config_generation": self.config_generation,
            "config_digest": self.config_digest,
            "supervisor_incarnation": self.supervisor_incarnation,
            "transaction_id": transaction_id,
            "operation": operation,
            "payload_digest": payload_digest,
            "data": dict(data),
        }
        return self._owner._append_candidate_event_locked(
            payload, lifetime_token=self._lifetime_token,
            context_token=self._context_token, thread_id=self._thread_id)

    def _close(self) -> None:
        self._active = False

    def projection_cache(self) -> Any:
        if not self._active or threading.get_ident() != self._thread_id:
            raise ControlRefused("candidate transaction context is not current")
        return copy.deepcopy(self._owner._candidate_projection_cache)

    def update_projection_cache(self, value: Any) -> None:
        del value
        raise ControlRefused(
            "candidate projection replacement is reserved for the trusted replayer")

    def _update_projection_cache_trusted(self, value: Any, *, authority: object) -> None:
        if not self._active or threading.get_ident() != self._thread_id:
            raise ControlRefused("candidate transaction context is not current")
        self._owner._update_candidate_projection_cache_locked(
            value, lifetime_token=self._lifetime_token,
            context_token=self._context_token, thread_id=self._thread_id,
            authority=authority)

    def completed_candidate(self, request_id: str) -> Any:
        if not self._active or threading.get_ident() != self._thread_id:
            raise ControlRefused("candidate transaction context is not current")
        return copy.deepcopy(self._owner._candidate_completed_records.get(request_id))

    def completed_candidate_ids(self) -> tuple[str, ...]:
        if not self._active or threading.get_ident() != self._thread_id:
            raise ControlRefused("candidate transaction context is not current")
        return tuple(sorted(self._owner._candidate_completed_records))


class CampaignController:
    """One lock-owning durable control projection for one resolved generation."""

    def __init__(self, resolved: ResolvedCampaign, store: Path, *,
                 config_generation: int = 1,
                 readiness_check: Callable[[], tuple[bool, str | None]] | None = None,
                 snapshot_version: int = 1,
                 scheduler_engine: scheduling.SchedulerEngine | None = None,
                 lifecycle_provider: Any = None,
                 lifecycle_dependency_check: Callable[[], bool | None] | None = None,
                 clock: Callable[[], str] = _now) -> None:
        if not isinstance(resolved, ResolvedCampaign):
            raise TypeError("resolved must be ResolvedCampaign")
        if not isinstance(config_generation, int) or isinstance(config_generation, bool) \
                or config_generation < 1:
            raise ValueError("config_generation must be positive")
        if snapshot_version not in {1, 2, 3}:
            raise ValueError("snapshot_version must be 1, 2, or 3")
        if snapshot_version == 1 and lifecycle_provider is not None:
            raise ControlRefused("worker lifecycle provider requires explicit snapshot v2")
        if scheduler_engine is not None and snapshot_version != 3:
            raise ControlRefused("unified scheduler requires explicit snapshot v3")
        if snapshot_version == 3 and scheduler_engine is None:
            raise ControlRefused("snapshot v3 requires the controller-owned unified scheduler")
        if scheduler_engine is not None and not isinstance(
                scheduler_engine, scheduling.SchedulerEngine):
            raise TypeError("scheduler_engine must be SchedulerEngine")
        try:
            normalized = ResolvedCampaign.from_dict(resolved.to_dict())
        except Exception as exc:
            raise ControlRefused(f"invalid resolved campaign: {exc}") from exc
        if normalized != resolved:
            raise ControlRefused("resolved campaign is not in canonical normalized form")
        self.resolved = normalized
        self.store = Path(store).absolute()
        self.config_generation = config_generation
        self.requested_manifest_digest = normalized.manifest_digest
        self.config_digest = resolved_config_digest(normalized)
        self._producer_build = _loaded_producer_build_identity()
        self.snapshot_version = snapshot_version
        self._lifecycle_provider = lifecycle_provider
        self._lifecycle_dependency_check = lifecycle_dependency_check or (lambda: True)
        self._worker_lifecycle = None
        self._active_worker_events: list[dict[str, Any]] = []
        self._active_acquisition_events: list[dict[str, Any]] = []
        self._acquisition_projection = worker_lifecycle_module.project_acquisitions([])
        self._worker_lifecycle_revision = 0
        self._worker_last_generation = 0
        self._worker_projection = worker_lifecycle_module.project_events([])
        self._worker_run_active = False
        self._worker_no_acquisition: tuple[Any, ...] | None = None
        self._worker_historical_logical_attempts: set[tuple[Any, ...]] = set()
        self.readiness_check = readiness_check or (lambda: (False, "execution authority absent"))
        self._cached_driver_readiness = (False, "provider readiness not refreshed")
        self.clock = clock
        self._mutex = threading.RLock()
        self._shutdown_condition = threading.Condition(self._mutex)
        self._lease_fd: int | None = None
        self._lock_identity: dict[str, int] | None = None
        self._journal_root_identity: dict[str, int] | None = None
        self._runtime_root: RuntimeRoot | None = None
        self._journal: journal_module.Journal | None = None
        self._entered = False
        self._ever_entered = False
        self._poisoned = False
        self._journal_cursor = 0
        self.supervisor_incarnation = 0
        self.stream_epoch = 0
        self.sequence = 0
        self.control_revision = 0
        self.desired_state = "paused"
        self.observed_state = "paused"
        self.prerequisite_reason: str | None = "explicit resume required"
        self._command_results: dict[str, dict[str, Any]] = {}
        self._command_requests: dict[str, dict[str, Any]] = {}
        self._shutdown_requested = False
        self._shutdown_drain_request_id: str | None = None
        self._candidate_pending: tuple[str, str, str] | None = None
        self._candidate_pending_payload: Mapping[str, Any] | None = None
        self._candidate_prepared = False
        self._candidate_completed: set[str] = set()
        self._candidate_completed_records: dict[str, dict[str, Mapping[str, Any]]] = {}
        self._candidate_entries: list[Any] = []
        self._candidate_projection_cache: Any = None
        self._candidate_projection_position = 0
        self._lifetime_token: object | None = None
        self._candidate_context_token: object | None = None
        self._native_records: dict[str, Any] = {}
        self._native_payload_digests: dict[str, str] = {}
        self._native_validator: NativeCaptureValidator | None = None
        self._native_capabilities: set[object] = set()
        self._maintenance_events: list[dict[str, Any]] = []
        self._maintenance_state = maintenance_module.MaintenanceState()
        self._maintenance_tombstone_intent = False
        self._retention_catalog_seed = None
        self._retention_catalog_event = None
        self._prepared_retention_jobs: dict[str, tuple[Any, ...]] = {}
        self._scheduler_engine = scheduler_engine
        self._driver_issued: dict[str, dict[str, Any]] = {}
        self._driver_settled: dict[str, dict[str, Any]] = {}
        self._driver_settlement_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
        self._driver_artifact_store: Any = None
        self._a2_execution_entries: dict[str, list[Any]] = {}
        self._a2_logical_executions: dict[str, str] = {}
        self._a2_bank_sources: dict[str, list[str]] = {}
        self._supervisor_id: str | None = None
        self._actor_preparation_events: list[dict[str, Any]] = []
        self._actor_preparation_state = actor_state_module.ActorPreparationProjection()
        self._actor_profile_producer: Any = None
        self._current_actor_profile_receipts: dict[str, Any] = {}
        self._actor_trusted_held_receipts: dict[str, tuple[Any, Any]] = {}
        self._actor_profile_execution_reservations: dict[str, Any] = {}
        self._actor_profile_execution_settlements: dict[str, Any] = {}
        self._actor_profile_cancelled_attempts: set[tuple[Any, ...]] = set()
        self._actor_profile_attempt_phases: dict[tuple[Any, ...], str] = {}
        if scheduler_engine is not None and (
                scheduler_engine.scheduler_id != normalized.campaign_id
                or scheduler_engine.config.config_id != normalized.campaign_id):
            raise ControlRefused(
                "unified scheduler IDs must exactly bind the resolved campaign")

    def _acquire(self) -> None:
        runtime: RuntimeRoot | None = None
        fd: int | None = None
        try:
            runtime = RuntimeRoot.create_or_open(self.store)
            fd = runtime.open_leaf(".supervisor.lock", os.O_RDWR | os.O_CREAT, 0o600)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ControlRefused("supervisor lock has unsafe object identity")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(fd)
                fd = None
                raise
        except BlockingIOError as exc:
            if runtime is not None:
                runtime.close()
            raise ControlRefused("another campaign controller owns the supervisor lock") from exc
        except (OSError, SecureRuntimeError) as exc:
            if fd is not None:
                os.close(fd)
            if runtime is not None:
                runtime.close()
            detail = ("supervisor lock symlink is refused"
                      if isinstance(exc, OSError) and exc.errno == errno.ELOOP
                      else f"cannot acquire supervisor lock: {exc}")
            raise ControlRefused(detail) from exc
        except BaseException:
            if fd is not None:
                os.close(fd)
            if runtime is not None:
                runtime.close()
            raise
        self._runtime_root = runtime
        self._lease_fd = fd
        self._lock_identity = object_identity(os.fstat(fd))

    def __enter__(self) -> "CampaignController":
        with self._mutex:
            if (self._ever_entered or self._entered or self._lease_fd is not None
                    or self._runtime_root is not None):
                raise ControlRefused("controller instances are single-lifetime")
            self._acquire()
            self._ever_entered = True
            self._lifetime_token = object()
            try:
                journal_root = self.store / "journal"
                fresh = not os.path.lexists(journal_root)
                if not fresh:
                    self._verify_journal_layout(journal_root)
                    self._journal_root_identity = object_identity(os.lstat(journal_root))
                self._journal = journal_module.Journal(
                    str(journal_root), campaign_id=self.resolved.campaign_id)
                self._verify_store()
                if fresh:
                    self._journal.initialize()
                    self._verify_journal_layout(journal_root)
                    self._journal_root_identity = object_identity(os.lstat(journal_root))
                    entries = []
                else:
                    entries = self._journal.read_all()
                self._verify_store()
                self._verify_journal_layout(journal_root)
                self._supervisor_id = "supervisor-" + schemas.content_hash({
                    "campaign_id": self.resolved.campaign_id,
                    "config_digest": self.config_digest,
                    "store": str(self.store),
                })
                self._replay(entries)
                self.supervisor_incarnation += 1
                self.stream_epoch += 1
                self.sequence = 0
                start_observed = self.observed_state
                pending_runtime = False
                for issued in self._driver_issued.values():
                    if issued["transition_id"] in self._driver_settled:
                        continue
                    selection = issued.get("selection")
                    catalog = issued.get("catalog")
                    if not isinstance(selection, Mapping) or not isinstance(catalog, Mapping):
                        continue
                    work = catalog.get("work_by_stage_digest")
                    selected = (work.get(selection.get("proposal_digest"))
                                if isinstance(work, Mapping) else None)
                    if (isinstance(selected, Mapping)
                            and selected.get("kind") == "runtime_comparison"):
                        pending_runtime = True
                        break
                if (self.snapshot_version == 3
                        and start_observed == "ownership_unresolved"
                        and pending_runtime):
                    start_observed = "waiting_prerequisite"
                self._append_event("START", {
                    "desired_state": self.desired_state,
                    "observed_state": start_observed,
                    "prerequisite_reason": self.prerequisite_reason,
                    "lock_identity": copy.deepcopy(self._lock_identity),
                })
                self._entered = True
                self._poisoned = False
                if self.snapshot_version in {2, 3}:
                    self._initialize_worker_lifecycle_locked()
                return self
            except BaseException:
                self.close()
                raise

    def _replay(self, entries) -> None:
        last_incarnation = 0
        last_epoch = 0
        last_revision = 0
        saw_start = False
        saw_v2_start = False
        saw_v3_start = False
        candidate_pending: tuple[str, str, str] | None = None
        candidate_pending_payload: Mapping[str, Any] | None = None
        candidate_prepared = False
        candidate_completed: set[str] = set()
        candidate_completed_records: dict[str, dict[str, Mapping[str, Any]]] = {}
        candidate_entries = []
        native_records: dict[str, Any] = {}
        native_payload_digests: dict[str, str] = {}
        worker_events: list[dict[str, Any]] = []
        acquisition_events: list[dict[str, Any]] = []
        acquisition_revision = 0
        acquisition_last_generation = 0
        maintenance_events: list[dict[str, Any]] = []
        maintenance_state = maintenance_module.MaintenanceState()
        a2_execution_entries: dict[str, list[Any]] = {}
        a2_logical_executions: dict[str, str] = {}
        a2_bank_sources: dict[str, list[str]] = {}
        actor_preparation_events: list[dict[str, Any]] = []
        actor_preparation_state = actor_state_module.ActorPreparationProjection()
        retention_catalog_event = None
        for entry in entries:
            self._journal_cursor = entry.seq
            if entry.campaign_id not in (None, self.resolved.campaign_id):
                raise ControlRefused("store contains another campaign identity")
            if entry.kind == journal_module.KIND_A2_RUNTIME_EXECUTION:
                from . import a2_execution_state
                try:
                    row = a2_execution_state.validate_transition(entry.payload)
                except a2_execution_state.A2ExecutionStateRefused as exc:
                    raise journal_module.JournalCorruption(
                        f"invalid A2 runtime execution history: {exc}") from exc
                if (not saw_start or entry.record_id != row["execution_id"]
                        or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "A2 runtime execution event breaks durable owner binding")
                prior_execution = a2_logical_executions.get(row["logical_id"])
                if prior_execution not in {None, row["execution_id"]}:
                    raise journal_module.JournalCorruption(
                        "A2 logical execution changes its fixed plan/frame identity")
                candidate = [*a2_execution_entries.get(row["execution_id"], []),
                             copy.deepcopy(entry)]
                try:
                    projection = a2_execution_state.project_transitions(
                        [item.payload for item in candidate])
                    reference = projection.bank_reference
                    if reference is not None:
                        source = a2_execution_entries.get(reference["source_execution_id"])
                        if not source:
                            raise a2_execution_state.A2ExecutionStateRefused(
                                "A2 bank reference source is absent or ordered after its reuse")
                        expected_reference = a2_execution_state.make_bank_reference(
                            source_values=[item.payload for item in source],
                            source_journal_entry_ids=[item.event_id for item in source],
                            target_plan_digest=projection.plan_digest,
                            target_frame_digest=projection.frame_digest)
                        if reference != expected_reference:
                            raise a2_execution_state.A2ExecutionStateRefused(
                                "A2 bank reference differs from its original source history")
                except a2_execution_state.A2ExecutionStateRefused as exc:
                    raise journal_module.JournalCorruption(
                        f"A2 runtime execution replay is inconsistent: {exc}") from exc
                a2_logical_executions[row["logical_id"]] = row["execution_id"]
                a2_execution_entries[row["execution_id"]] = candidate
                if "anchor_bank" in projection.sealed_phases:
                    seal = next(event for event in projection.events
                                if event["phase"] == "anchor_bank"
                                and event["state"] == "SEALED")
                    bank_digest = seal["payload"]["bank_digest"]
                    sources = a2_bank_sources.setdefault(bank_digest, [])
                    if row["execution_id"] not in sources:
                        sources.append(row["execution_id"])
                continue
            if entry.kind == journal_module.KIND_RETENTION_CATALOG_INSTALLED:
                try:
                    row = retention_catalog_module.validate_install_event(entry.payload)
                except retention_catalog_module.NativeRetentionCatalogRefused as exc:
                    raise journal_module.JournalCorruption(
                        f"invalid native retention catalog history: {exc}") from exc
                if (self.snapshot_version != 3 or not saw_start
                        or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation
                        or retention_catalog_event is not None):
                    raise journal_module.JournalCorruption(
                        "native retention catalog breaks durable controller binding")
                retention_catalog_event = retention_catalog_module._plain(row)
                continue
            if entry.kind == journal_module.KIND_WORKER_LIFECYCLE:
                violations = journal_module._validate_native_payload(entry.kind, entry.payload)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid worker lifecycle history: " + "; ".join(violations))
                if self.snapshot_version not in {2, 3}:
                    raise ControlRefused(
                        "store contains worker lifecycle v2; reopen explicitly as v2")
                row = entry.payload
                if (not saw_start or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest):
                    raise journal_module.JournalCorruption(
                        "worker lifecycle event breaks controller binding")
                if row["event"] == "OWNED_LAUNCH_INTENT":
                    if row["supervisor_incarnation"] != last_incarnation:
                        raise journal_module.JournalCorruption(
                            "worker lifecycle intent breaks current supervisor binding")
                else:
                    prior = next((item for item in reversed(worker_events)
                                  if item["worker_id"] == row["worker_id"]), None)
                    if (prior is None or row["supervisor_id"] != prior["supervisor_id"]
                            or row["supervisor_incarnation"]
                            != prior["supervisor_incarnation"]):
                        raise journal_module.JournalCorruption(
                            "worker lifecycle continuation breaks durable owner binding")
                worker_events.append(copy.deepcopy(dict(entry.payload)))
                self._worker_historical_logical_attempts.add(
                    self._worker_attempt_key(row)[5:])
                continue
            if entry.kind == journal_module.KIND_ACTOR_PREPARATION:
                try:
                    row = actor_state_module.validate_event(entry.payload)
                    if (not saw_start
                            or row["campaign_id"] != self.resolved.campaign_id
                            or row["config_generation"] != self.config_generation
                            or row["config_digest"] != self.config_digest
                            or row["supervisor_incarnation"] != last_incarnation
                            or row["supervisor_id"] != self._supervisor_id):
                        raise actor_state_module.ActorStateRefused(
                            "actor event breaks current owner binding")
                    actor_preparation_events.append(copy.deepcopy(row))
                    actor_preparation_state = actor_state_module.project_events(
                        actor_preparation_events)
                except actor_state_module.ActorStateRefused as exc:
                    raise journal_module.JournalCorruption(
                        f"actor preparation replay is inconsistent: {exc}") from exc
                continue
            if entry.kind == journal_module.KIND_MAINTENANCE_EXECUTION:
                try:
                    row = maintenance_module.validate_event(entry.payload)
                    token = maintenance_module.ExclusionToken.from_dict(row["token"])
                    if (not saw_start
                            or token.campaign_id != self.resolved.campaign_id
                            or token.config_generation != self.config_generation
                            or token.config_digest != self.config_digest
                            or token.supervisor_incarnation != last_incarnation
                            or token.supervisor_id
                            != self._maintenance_supervisor_id()):
                        raise maintenance_module.MaintenanceExecutionRefused(
                            "maintenance event breaks current owner binding")
                    maintenance_events.append(copy.deepcopy(row))
                    maintenance_state = maintenance_module.project_events(maintenance_events)
                except maintenance_module.MaintenanceExecutionRefused as exc:
                    raise journal_module.JournalCorruption(
                        f"maintenance execution replay is inconsistent: {exc}") from exc
                continue
            if entry.kind == journal_module.KIND_WORKER_ACQUISITION:
                violations = journal_module._validate_native_payload(entry.kind, entry.payload)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid worker acquisition history: " + "; ".join(violations))
                if self.snapshot_version not in {2, 3}:
                    raise ControlRefused(
                        "store contains worker acquisition v2; reopen explicitly as v2")
                row = worker_lifecycle_module.validate_acquisition_transition(entry.payload)
                self._worker_historical_logical_attempts.add(
                    self._worker_attempt_key(row)[5:])
                if row["phase"] == "INTENT":
                    if (acquisition_events or not saw_start
                            or row["campaign_id"] != self.resolved.campaign_id
                            or row["config_generation"] != self.config_generation
                            or row["config_digest"] != self.config_digest
                            or row["supervisor_incarnation"] != last_incarnation
                            or (acquisition_last_generation
                                and row["worker_generation"]
                                != acquisition_last_generation + 1)):
                        raise journal_module.JournalCorruption(
                            "worker acquisition intent breaks controller binding")
                    acquisition_events = [copy.deepcopy(row)]
                else:
                    if not acquisition_events:
                        raise journal_module.JournalCorruption(
                            "worker acquisition resolution lacks indexed intent")
                    candidate = [*acquisition_events, copy.deepcopy(row)]
                    if row["data"]["outcome"] == "lifecycle_handoff":
                        matches = [item for item in worker_events
                                   if item["event"] == "OWNED_LAUNCH_INTENT"
                                   and item["worker_id"] == row["worker_id"]
                                   and item["worker_generation"]
                                   == row["worker_generation"]]
                        if len(matches) != 1:
                            raise journal_module.JournalCorruption(
                                "acquisition handoff lacks prior exact journal lifecycle intent")
                        try:
                            handoff_grant = worker_lifecycle_module.validate_lifecycle_handoff(
                                acquisition_events[0], matches[0])
                        except worker_lifecycle_module.LifecycleRefused as exc:
                            raise journal_module.JournalCorruption(
                                "acquisition handoff differs from lifecycle intent") from exc
                        if ((handoff_grant.grant_id, handoff_grant.generation)
                                != (row["data"]["grant_id"],
                                    row["data"]["grant_generation"])):
                            raise journal_module.JournalCorruption(
                                "acquisition handoff grant identity differs")
                    acquisition_events = candidate
                try:
                    acquisition_projection = worker_lifecycle_module.project_acquisitions(
                        acquisition_events)
                except worker_lifecycle_module.LifecycleRefused as exc:
                    raise journal_module.JournalCorruption(
                        f"worker acquisition replay is inconsistent: {exc}") from exc
                acquisition_revision += 1
                acquisition_last_generation = max(
                    acquisition_last_generation, row["worker_generation"])
                if (row["phase"] == "RESOLVED"
                        and row["data"]["outcome"] == "denied"):
                    self._worker_no_acquisition = self._worker_attempt_key(row)
                if acquisition_projection.pending is None:
                    acquisition_events = []
                continue
            if entry.kind == journal_module.KIND_CAMPAIGN_COMMAND_V2:
                violations = journal_module._validate_native_payload(entry.kind, entry.payload)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid v2 command history: " + "; ".join(violations))
                if self.snapshot_version not in {2, 3}:
                    raise ControlRefused(
                        "store contains campaign controls v2; reopen explicitly as v2")
                row = worker_lifecycle_module.validate_command_transition_v2(entry.payload)
                if (not saw_start
                        or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "v2 command breaks campaign/config/supervisor binding")
                result = row["result"]
                command = dict(row["command"])
                request_id = result["request_id"]
                if row["phase"] == "ACCEPTED":
                    if row["control_revision"] != last_revision + 1 \
                            or request_id in self._command_results:
                        raise journal_module.JournalCorruption(
                            "v2 command acceptance breaks revision/idempotency")
                    last_revision = row["control_revision"]
                else:
                    prior = self._command_results.get(request_id)
                    if (prior is None or prior.get("completed") is not False
                            or row["control_revision"] > last_revision
                            or prior["payload_digest"] != result["payload_digest"]):
                        raise journal_module.JournalCorruption(
                            "v2 command completion lacks exact pending acceptance")
                if row["control_revision"] == last_revision:
                    self.desired_state = result["desired_state"]
                    self.observed_state = result["observed_state"]
                    self.prerequisite_reason = result["prerequisite_reason"]
                self._command_results[request_id] = copy.deepcopy(result)
                self._command_requests[request_id] = command
                continue
            if entry.kind == journal_module.KIND_CAMPAIGN_COMMAND_V3:
                violations = journal_module._validate_native_payload(entry.kind, entry.payload)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid v3 command history: " + "; ".join(violations))
                if self.snapshot_version not in {2, 3}:
                    raise ControlRefused(
                        "store contains campaign controls v3; reopen explicitly as v2/v3")
                row = campaign_command_v2.validate_transition(entry.payload)
                if (not saw_start or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "v3 command transition breaks current writer binding")
                result = row["result"]
                command = row["command"]
                request_id = result["request_id"]
                if row["phase"] == "ACCEPTED":
                    if (row["control_revision"] != last_revision + 1
                            or request_id in self._command_results):
                        raise journal_module.JournalCorruption(
                            "v3 command acceptance breaks revision/idempotency")
                    last_revision = row["control_revision"]
                else:
                    prior = self._command_results.get(request_id)
                    accepted = self._command_requests.get(request_id)
                    if (prior is None or prior.get("completed") is not False
                            or accepted != command or row["control_revision"] > last_revision
                            or prior["payload_digest"] != result["payload_digest"]):
                        raise journal_module.JournalCorruption(
                            "v3 completion lacks exact accepted command")
                if row["control_revision"] == last_revision:
                    self.desired_state = result["desired_state"]
                    self.observed_state = result["observed_state"]
                    self.prerequisite_reason = result["prerequisite_reason"]
                self._command_results[request_id] = copy.deepcopy(result)
                self._command_requests[request_id] = copy.deepcopy(command)
                continue
            if entry.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED:
                row = entry.payload
                violations = journal_module._validate_native_payload(entry.kind, row)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid native capture history: " + "; ".join(violations))
                context = row["carrier"]["capture_context"]
                plan = row["carrier"]["plan"]
                measurement_id = row["measurement_id"]
                if (not saw_start or entry.record_id != measurement_id
                        or context["campaign_id"] != self.resolved.campaign_id
                        or plan["campaign_id"] != self.resolved.campaign_id
                        or context["config_generation"] != self.config_generation
                        or context["config_digest"] != self.config_digest
                        or context["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "native capture event breaks campaign/config/incarnation binding")
                digest = schemas.content_hash(row)
                if measurement_id in native_records:
                    raise journal_module.JournalCorruption(
                        "native capture history repeats a measurement_id")
                native_records[measurement_id] = copy.deepcopy(entry)
                native_payload_digests[measurement_id] = digest
                continue
            if entry.kind == journal_module.KIND_UNIFIED_DRIVER_ISSUED:
                violations = journal_module._validate_native_payload(entry.kind, entry.payload)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid unified driver history: " + "; ".join(violations))
                if self.snapshot_version != 3 or self._scheduler_engine is None:
                    raise ControlRefused(
                        "store contains unified scheduler history; reopen with v3 scheduler")
                from . import unified_driver as driver_module
                row = copy.deepcopy(dict(entry.payload))
                if (not saw_start or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "unified driver issue breaks controller binding")
                catalog_row = dict(row["catalog"])
                supplied_id = catalog_row.pop("catalog_id", None)
                catalog = driver_module.PlanningCatalog(**catalog_row)
                if supplied_id != catalog.catalog_id or row["catalog_id"] != catalog.catalog_id:
                    raise journal_module.JournalCorruption(
                        "unified driver catalog identity differs")
                current = self._scheduler_engine.operational_projection().projection_digest
                if current != row["prior_projection_digest"]:
                    raise journal_module.JournalCorruption(
                        "unified scheduler replay prior projection differs")
                preview = self._scheduler_engine.preview_selection(
                    [scheduling.StageProposal.from_dict(item)
                     for item in catalog.stage_proposals], now=catalog.observed_at)
                expected_transition = driver_module._digest({
                    "catalog_id": catalog.catalog_id,
                    "selection": preview.selection.to_dict()})
                if (preview.selection.to_dict() != row["selection"]
                        or preview.after.projection_digest != row["after_projection_digest"]
                        or preview.prior.projection_digest != row["prior_projection_digest"]
                        or expected_transition != row["transition_id"]):
                    raise journal_module.JournalCorruption(
                        "unified scheduler replay transition differs")
                self._scheduler_engine.apply_preview(preview)
                if row["catalog_id"] in self._driver_issued:
                    raise journal_module.JournalCorruption(
                        "unified driver catalog is issued more than once")
                self._driver_issued[row["catalog_id"]] = row
                continue
            if entry.kind == journal_module.KIND_UNIFIED_DRIVER_SETTLED:
                violations = journal_module._validate_native_payload(entry.kind, entry.payload)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid unified driver settlement: " + "; ".join(violations))
                if self.snapshot_version != 3 or self._scheduler_engine is None:
                    raise ControlRefused(
                        "store contains unified settlement history; reopen with v3 scheduler")
                row = copy.deepcopy(dict(entry.payload))
                issued = self._driver_issued.get(row["catalog_id"])
                if (issued is None or row["transition_id"] != issued["transition_id"]
                        or row["selection"] != issued["selection"]
                        or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "unified settlement breaks issued/controller binding")
                preview = self._scheduler_engine.preview_accounting(
                    row["selection"], row["receipt"], outcome=row["outcome"])
                if (preview.prior.projection_digest != row["prior_projection_digest"]
                        or preview.after.projection_digest != row["after_projection_digest"]):
                    raise journal_module.JournalCorruption(
                        "unified settlement replay projection differs")
                self._scheduler_engine.apply_accounting_preview(preview)
                if row["transition_id"] in self._driver_settled:
                    raise journal_module.JournalCorruption(
                        "unified driver transition is settled more than once")
                self._driver_settled[row["transition_id"]] = row
                continue
            if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION:
                candidate_entries.append(copy.deepcopy(entry))
                row = entry.payload
                violations = journal_module._validate_native_payload(entry.kind, row)
                if violations:
                    raise journal_module.JournalCorruption(
                        "invalid candidate transaction history: "
                        + "; ".join(violations))
                if (not saw_start
                        or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_generation"] != self.config_generation
                        or row["config_digest"] != self.config_digest
                        or row["supervisor_incarnation"] != last_incarnation):
                    raise journal_module.JournalCorruption(
                        "candidate transaction event breaks supervisor binding")
                transaction_key = (row["transaction_id"], row["operation"],
                                   row["payload_digest"])
                if row["phase"] == "INTENT":
                    if candidate_pending is not None \
                            or row["transaction_id"] in candidate_completed:
                        raise journal_module.JournalCorruption(
                            "candidate transaction intent overlaps or reuses an id")
                    candidate_pending = transaction_key
                    candidate_pending_payload = copy.deepcopy(row)
                    candidate_prepared = False
                elif row["phase"] == "PREPARED":
                    if candidate_pending != transaction_key or candidate_prepared:
                        raise journal_module.JournalCorruption(
                            "candidate preparation lacks exact intent or is duplicated")
                    candidate_prepared = True
                elif candidate_pending != transaction_key:
                    raise journal_module.JournalCorruption(
                        "candidate transaction completion lacks exact intent")
                else:
                    candidate_completed.add(row["transaction_id"])
                    assert candidate_pending_payload is not None
                    candidate_completed_records[row["transaction_id"]] = {
                        "intent": candidate_pending_payload,
                        "completion": copy.deepcopy(row),
                    }
                    candidate_pending = None
                    candidate_pending_payload = None
                    candidate_prepared = False
                continue
            if entry.kind != journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT:
                continue
            row = entry.payload
            violations = journal_module._validate_native_payload(entry.kind, row)
            if violations:
                raise journal_module.JournalCorruption(
                    "invalid campaign supervisor history: " + "; ".join(violations))
            if row["schema"] == journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA_V3:
                if self.snapshot_version != 3:
                    raise ControlRefused(
                        "store is fenced for controller v3; older-reader downgrade is refused")
                saw_v2_start = True
                saw_v3_start = True
            elif row["schema"] == journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA_V2:
                if self.snapshot_version not in {2, 3}:
                    raise ControlRefused(
                        "store is fenced for controller v2; v1 downgrade is refused")
                if saw_v3_start:
                    raise journal_module.JournalCorruption(
                        "v2 supervisor START cannot follow the durable v3 fence")
                saw_v2_start = True
            elif saw_v2_start:
                raise journal_module.JournalCorruption(
                    "v1 supervisor events cannot follow the durable v2 fence")
            if (row["campaign_id"] != self.resolved.campaign_id
                    or row["config_generation"] != self.config_generation
                    or row["config_digest"] != self.config_digest):
                raise ControlRefused(
                    "store belongs to a different campaign/config generation; "
                    "create an explicit new generation")
            if row["event"] == "START":
                if (row["supervisor_incarnation"] != last_incarnation + 1
                        or row["stream_epoch"] != last_epoch + 1
                        or row["control_revision"] != last_revision):
                    raise journal_module.JournalCorruption(
                        "campaign supervisor START breaks monotonic replay identity")
                data = row["data"]
                if data["lock_identity"] != self._lock_identity:
                    raise journal_module.JournalCorruption(
                        "campaign supervisor lock identity changed")
                if not saw_start:
                    if data["desired_state"] != "paused" or data["observed_state"] != "paused":
                        raise journal_module.JournalCorruption(
                            "first campaign supervisor START must be paused")
                elif (data["desired_state"] != self.desired_state
                      or data["observed_state"] != self.observed_state
                      or data.get("prerequisite_reason") != self.prerequisite_reason):
                    raise journal_module.JournalCorruption(
                        "campaign supervisor START does not preserve replayed state")
                last_incarnation = row["supervisor_incarnation"]
                last_epoch = row["stream_epoch"]
                self.desired_state = data["desired_state"]
                self.observed_state = data["observed_state"]
                self.prerequisite_reason = data.get("prerequisite_reason")
                saw_start = True
            elif row["event"] == "CONTROL_ACCEPTED":
                if (not saw_start or row["supervisor_incarnation"] != last_incarnation
                        or row["stream_epoch"] != last_epoch
                        or row["control_revision"] != last_revision + 1):
                    raise journal_module.JournalCorruption(
                        "campaign control event breaks monotonic replay identity")
                data = dict(row["data"])
                command = data["command"]
                result = data.get("result")
                request_id = data.get("request_id")
                desired = data.get("desired_state")
                observed = data.get("observed_state")
                if (not isinstance(result, Mapping) or not isinstance(request_id, str)
                        or desired not in {"paused", "running", "drained"}
                        or observed not in {"paused", "running", "drained",
                                            "waiting_prerequisite"}):
                    raise journal_module.JournalCorruption(
                        "campaign control replay payload is malformed")
                if (command["campaign_id"] != self.resolved.campaign_id
                        or command["config_generation"] != self.config_generation
                        or command["expected_control_revision"] != last_revision
                        or result["control_revision"] != row["control_revision"]):
                    raise journal_module.JournalCorruption(
                        "campaign control replay binding/revision is inconsistent")
                if request_id in self._command_results:
                    raise journal_module.JournalCorruption(
                        "campaign control history repeats a request_id")
                if self.desired_state == "drained":
                    raise journal_module.JournalCorruption(
                        "campaign control history continues after terminal drain")
                last_revision = row["control_revision"]
                self.desired_state = desired
                self.observed_state = observed
                self.prerequisite_reason = data.get("prerequisite_reason")
                self._command_results[request_id] = dict(result)
        self.supervisor_incarnation = last_incarnation
        self.stream_epoch = last_epoch
        self._worker_lifecycle_revision = len(worker_events) + acquisition_revision
        self._worker_last_generation = max(
            [row["worker_generation"] for row in worker_events]
            + [acquisition_last_generation], default=0)
        try:
            self._worker_projection = worker_lifecycle_module.project_events(worker_events)
        except worker_lifecycle_module.LifecycleRefused as exc:
            raise journal_module.JournalCorruption(
                f"worker lifecycle replay is inconsistent: {exc}") from exc
        if self._worker_projection.active:
            active_id = next(iter(self._worker_projection.active))
            self._active_worker_events = [
                row for row in worker_events if row["worker_id"] == active_id]
        else:
            self._active_worker_events = []
        self._active_acquisition_events = acquisition_events
        self._acquisition_projection = worker_lifecycle_module.project_acquisitions(
            acquisition_events)
        if self._acquisition_projection.pending is not None:
            self.observed_state = "ownership_unresolved"
            self.prerequisite_reason = (
                "worker_acquisition_pending:"
                + str(self._acquisition_projection.pending["request_id"]))
        elif self._worker_projection.active:
            self.observed_state = "ownership_unresolved"
            self.prerequisite_reason = "owned worker reconciliation required"
        elif self.prerequisite_reason == "owned worker reconciliation required":
            self.observed_state = self.desired_state
            self.prerequisite_reason = None
        self.control_revision = last_revision
        self._candidate_pending = candidate_pending
        self._candidate_pending_payload = candidate_pending_payload
        self._candidate_prepared = candidate_prepared
        self._candidate_completed = candidate_completed
        self._candidate_completed_records = candidate_completed_records
        self._candidate_entries = candidate_entries
        self._native_records = native_records
        self._native_payload_digests = native_payload_digests
        self._maintenance_events = maintenance_events
        self._maintenance_state = maintenance_state
        if maintenance_state.owned and maintenance_state.phase != "UNRESOLVED":
            self._maintenance_state = replace(
                maintenance_state, phase="UNRESOLVED",
                reason="controller restarted before exact maintenance settlement")
        self._maintenance_tombstone_intent = maintenance_state.owned
        self._retention_catalog_event = retention_catalog_event
        self._retention_catalog_seed = None
        if retention_catalog_event is not None:
            seed_row = dict(retention_catalog_event["seed"])
            supplied = seed_row.pop("seed_digest")
            self._retention_catalog_seed = retention_catalog_module.NativeRetentionCatalogSeed(
                seed_row["campaign_id"], seed_row["config_digest"],
                seed_row["manifest_digest"],
                tuple(retention_catalog_module.CatalogArtifact.from_dict(item)
                      for item in seed_row["artifacts"]),
                tuple(seed_row["model_inventories"]),
                tuple(seed_row["uncertain_scopes"]), supplied,
                seed_row["schema"], retention_catalog_module._TOKEN)
        self._a2_execution_entries = a2_execution_entries
        self._a2_logical_executions = a2_logical_executions
        self._a2_bank_sources = a2_bank_sources
        self._actor_preparation_events = actor_preparation_events
        self._actor_preparation_state = actor_preparation_state
        if actor_preparation_state.profiles:
            from .actor_lifecycle import TargetProfileReceipt
            self._current_actor_profile_receipts = {}
            for profile_row in actor_preparation_state.profiles.values():
                receipt = TargetProfileReceipt(
                    campaign_digest=profile_row["config_digest"],
                    profile_request=profile_row["profile_request"],
                    profile_request_digest=profile_row["profile_request_digest"],
                    target_revision_digest=profile_row["target_revision_digest"],
                    target_profile_digest=profile_row["target_profile_digest"],
                    verified_at=profile_row["verified_at"],
                    valid_until=profile_row["valid_until"],
                    clock_domain=profile_row["clock_domain"],
                    verifier_ref=profile_row["verifier_ref"])
                self._current_actor_profile_receipts[receipt.digest] = receipt

    def _maintenance_supervisor_id(self) -> str:
        if self._supervisor_id is not None:
            return self._supervisor_id
        return "supervisor-" + schemas.content_hash({
            "campaign_id": self.resolved.campaign_id,
            "config_digest": self.config_digest,
            "store": str(self.store),
        })

    def _initialize_worker_lifecycle_locked(self) -> None:
        if not self._mutex._is_owned():
            raise ControlRefused("worker lifecycle initialization requires controller lock")
        if self._runtime_root is None:
            raise ControlRefused("worker lifecycle runtime root is unavailable")
        supervisor_id = "supervisor-" + schemas.content_hash({
            "campaign_id": self.resolved.campaign_id,
            "config_digest": self.config_digest,
            "store": str(self.store),
        })
        self._supervisor_id = supervisor_id
        binding = worker_lifecycle_module.CampaignBinding(
            self.resolved.campaign_id, self.config_digest, self.config_generation,
            supervisor_id, self.supervisor_incarnation)

        def admission(request, grant, now, required_until):
            with self._mutex:
                self._require_active_locked()
                remaining_stage = required_until - now - request.teardown_seconds
                if remaining_stage <= 0:
                    return worker_lifecycle_module.StageAdmission(
                        False, "stage budget expired before admission")
                decision = may_start_stage(
                    desired_state=self.desired_state,
                    current_control_revision=self.control_revision,
                    control_revision=request.control_revision,
                    current_supervisor_incarnation=self.supervisor_incarnation,
                    supervisor_incarnation=binding.supervisor_incarnation,
                    grant=TrustedGrant(grant.grant_id, grant.generation, grant.deadline,
                                       grant.revoked, grant.renewal_ok),
                    grant_identity=grant.grant_id, grant_generation=grant.generation,
                    now=now, max_stage_seconds=remaining_stage,
                    teardown_seconds=request.teardown_seconds,
                    dependency_check=self._lifecycle_dependency_check)
                return worker_lifecycle_module.StageAdmission(
                    decision.allowed, decision.reason)

        def binding_current(candidate):
            with self._mutex:
                return bool(
                    self._entered and not self._poisoned
                    and candidate.campaign_id == self.resolved.campaign_id
                    and candidate.config_digest == self.config_digest
                    and candidate.config_generation == self.config_generation
                    and candidate.supervisor_id == supervisor_id
                    and candidate.supervisor_incarnation == self.supervisor_incarnation)

        def runtime_fence(_request, _now):
            with self._mutex:
                self._require_active_locked()
                if self.desired_state == "drained":
                    return worker_lifecycle_module.RuntimeDirective(
                        "drain", "accepted drain reached the stage boundary")
                return worker_lifecycle_module.RuntimeDirective(
                    "continue", ("pause closes successors; held stage may settle"
                                 if self.desired_state == "paused"
                                 else "current held stage remains authorized"))

        self._worker_lifecycle = worker_lifecycle_module.WorkerLifecycle(
            binding=binding, runtime=self._runtime_root,
            event_sink=self._append_worker_event,
            provider=self._lifecycle_provider, admission_fence=admission,
            binding_fence=binding_current, runtime_fence=runtime_fence,
            wall_clock=self.clock)
        self._worker_lifecycle._worker_generation = self._worker_last_generation
        if self._acquisition_projection.pending is not None:
            self._worker_lifecycle._pending_acquisition = (
                worker_lifecycle_module.prospective_identity_from_transition(
                    self._acquisition_projection.pending))
            self._worker_lifecycle._ownership_unresolved = True
        if self._worker_projection.active:
            self._worker_lifecycle._ownership_unresolved = True

    def _append_worker_event(self, value: Mapping[str, Any]) -> None:
        if isinstance(value, Mapping) and value.get("schema") \
                == worker_lifecycle_module.ACQUISITION_SCHEMA:
            self._append_acquisition_event(value)
            return
        row = worker_lifecycle_module.validate_event(value)
        with self._mutex:
            self._require_active_locked()
            binding = self._worker_lifecycle.binding if self._worker_lifecycle else None
            if (binding is None or row["campaign_id"] != self.resolved.campaign_id
                    or row["config_digest"] != self.config_digest
                    or row["config_generation"] != self.config_generation):
                raise ControlRefused("worker event binding is not current")
            assert self._journal is not None
            if row["event"] == "OWNED_LAUNCH_INTENT":
                if (self._active_worker_events
                        or row["supervisor_id"] != binding.supervisor_id
                        or row["supervisor_incarnation"] != self.supervisor_incarnation):
                    raise ControlRefused("worker intent overlaps active lifecycle")
                pending = self._acquisition_projection.pending
                if (pending is None
                        or any(row[name] != pending[name] for name in (
                            "worker_id", "worker_generation", "request_id", "plan_digest",
                            "lineage_id", "stage_id", "container_id", "control_revision"))):
                    raise ControlRefused(
                        "worker lifecycle intent lacks exact durable prospective acquisition")
                candidate_events = [copy.deepcopy(row)]
            else:
                if (not self._active_worker_events
                        or self._active_worker_events[0]["worker_id"] != row["worker_id"]):
                    raise ControlRefused("worker event lacks its indexed active intent")
                candidate_events = [*self._active_worker_events, copy.deepcopy(row)]
            projection = worker_lifecycle_module.project_events(candidate_events)
            try:
                entry = self._journal.append(journal_module.KIND_WORKER_LIFECYCLE, row)
                self._verify_journal_layout(self.store / "journal")
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._worker_lifecycle_revision += 1
            self._worker_last_generation = max(
                self._worker_last_generation, row["worker_generation"])
            self._worker_projection = projection
            self._worker_historical_logical_attempts.add(
                self._worker_attempt_key(row)[5:])
            self._active_worker_events = (
                candidate_events if projection.active else [])
            if (not projection.active
                    and self._acquisition_projection.pending is None
                    and self.prerequisite_reason == "owned worker reconciliation required"):
                self.observed_state = self.desired_state
                self.prerequisite_reason = None

    def _append_acquisition_event(self, value: Mapping[str, Any]) -> None:
        row = worker_lifecycle_module.validate_acquisition_transition(value)
        with self._mutex:
            self._require_active_locked()
            binding = self._worker_lifecycle.binding if self._worker_lifecycle else None
            assert self._journal is not None
            if row["phase"] == "INTENT":
                if (binding is None or self._active_acquisition_events
                        or self._active_worker_events
                        or row["campaign_id"] != self.resolved.campaign_id
                        or row["config_digest"] != self.config_digest
                        or row["config_generation"] != self.config_generation
                        or row["supervisor_id"] != binding.supervisor_id
                        or row["supervisor_incarnation"] != self.supervisor_incarnation
                        or row["worker_generation"] != self._worker_last_generation + 1):
                    raise ControlRefused(
                        "prospective acquisition overlaps or breaks current binding/generation")
                candidate_events = [copy.deepcopy(row)]
            else:
                pending = self._acquisition_projection.pending
                if pending is None or not self._active_acquisition_events:
                    raise ControlRefused("acquisition resolution lacks indexed pending intent")
                candidate_events = [*self._active_acquisition_events, copy.deepcopy(row)]
                if row["data"]["outcome"] == "lifecycle_handoff":
                    matches = [item for item in self._active_worker_events
                               if item["event"] == "OWNED_LAUNCH_INTENT"
                               and item["worker_id"] == row["worker_id"]
                               and item["worker_generation"] == row["worker_generation"]]
                    if len(matches) != 1:
                        raise ControlRefused(
                            "acquisition handoff lacks actual durable lifecycle intent")
                    try:
                        handoff_grant = worker_lifecycle_module.validate_lifecycle_handoff(
                            self._active_acquisition_events[0], matches[0])
                    except worker_lifecycle_module.LifecycleRefused as exc:
                        raise ControlRefused(
                            "acquisition handoff differs from lifecycle intent") from exc
                    if ((handoff_grant.grant_id, handoff_grant.generation)
                            != (row["data"]["grant_id"],
                                row["data"]["grant_generation"])):
                        raise ControlRefused("acquisition handoff grant identity differs")
                elif self._active_worker_events:
                    raise ControlRefused(
                        "grant-free acquisition resolution conflicts with active lifecycle")
            projection = worker_lifecycle_module.project_acquisitions(candidate_events)
            try:
                entry = self._journal.append(
                    journal_module.KIND_WORKER_ACQUISITION, row)
                self._verify_journal_layout(self.store / "journal")
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._worker_lifecycle_revision += 1
            self._worker_last_generation = max(
                self._worker_last_generation, row["worker_generation"])
            self._acquisition_projection = projection
            self._worker_historical_logical_attempts.add(
                self._worker_attempt_key(row)[5:])
            self._active_acquisition_events = (
                candidate_events if projection.pending is not None else [])
            if (row["phase"] == "RESOLVED"
                    and row["data"]["outcome"] == "denied"):
                self._worker_no_acquisition = self._worker_attempt_key(row)
            if projection.pending is not None:
                self.observed_state = "ownership_unresolved"
                self.prerequisite_reason = (
                    "worker_acquisition_pending:" + str(row["request_id"]))
            elif (isinstance(self.prerequisite_reason, str)
                  and self.prerequisite_reason.startswith("worker_acquisition_pending:")):
                if self._worker_projection.active:
                    launch = self._active_worker_events[0]
                    if launch["supervisor_incarnation"] == self.supervisor_incarnation:
                        self.observed_state = self.desired_state
                        self.prerequisite_reason = None
                    else:
                        self.observed_state = "ownership_unresolved"
                        self.prerequisite_reason = "owned worker reconciliation required"
                else:
                    self.observed_state = self.desired_state
                    self.prerequisite_reason = None

    @staticmethod
    def _worker_attempt_key_fields(binding, request_id: str, plan_digest: str,
                                   lineage_id: str, stage_id: str) -> tuple[Any, ...]:
        return (
            binding.campaign_id, binding.config_digest, binding.config_generation,
            binding.supervisor_id, binding.supervisor_incarnation,
            request_id, plan_digest, lineage_id, stage_id,
        )

    @classmethod
    def _worker_attempt_key(cls, value, binding=None) -> tuple[Any, ...]:
        if binding is not None:
            return cls._worker_attempt_key_fields(
                binding, value.request_id, value.plan_digest,
                value.lineage_id, value.stage_id)
        owner = worker_lifecycle_module.CampaignBinding(
            value["campaign_id"], value["config_digest"], value["config_generation"],
            value["supervisor_id"], value["supervisor_incarnation"])
        return cls._worker_attempt_key_fields(
            owner, value["request_id"], value["plan_digest"],
            value["lineage_id"], value["stage_id"])

    def run_worker_stage(self, request: worker_lifecycle_module.StageRequest,
                         *, planned_invocation=None):
        """Run one provider-authorized stage without holding the command lock."""
        with self._mutex:
            self._require_active_locked()
            if self._maintenance_state.owned:
                raise ControlRefused("worker admission is fenced by maintenance exclusion")
            if not isinstance(request, worker_lifecycle_module.StageRequest):
                raise TypeError("request must be StageRequest")
            binding = (self._worker_lifecycle.binding
                       if self._worker_lifecycle is not None else None)
            attempt_key = (None if binding is None else
                           self._worker_attempt_key(request, binding))
            if attempt_key in self._actor_profile_cancelled_attempts:
                raise ControlRefused("target-profile admission was cancelled by this owner")
            if attempt_key is not None and self._worker_no_acquisition == attempt_key:
                # A newer call for the same logical request supersedes any earlier
                # denial proof before it can touch provider or lifecycle state.
                self._worker_no_acquisition = None

            def pre_engine_refusal(exc):
                if attempt_key is not None:
                    self._worker_no_acquisition = attempt_key
                raise exc

            if self.snapshot_version not in {2, 3} or self._worker_lifecycle is None:
                pre_engine_refusal(
                    ControlRefused("worker lifecycle requires explicit snapshot v2"))
            if (self._worker_run_active or self._worker_projection.active
                    or self._acquisition_projection.pending is not None):
                pre_engine_refusal(
                    ControlRefused("an owned worker is active or unresolved"))
            if self.desired_state != "running":
                pre_engine_refusal(
                    ControlRefused(f"worker admission is closed: {self.desired_state}"))
            if request.control_revision != self.control_revision:
                pre_engine_refusal(ControlRefused("worker request has stale control revision"))
            if self._lifecycle_provider is None:
                pre_engine_refusal(worker_lifecycle_module.WaitingAuthority(
                    "trusted grant provider is unavailable"))
            engine = self._worker_lifecycle
            if attempt_key in self._actor_profile_attempt_phases:
                if self._actor_profile_attempt_phases[attempt_key] != "prelaunch":
                    raise ControlRefused("target-profile admission is already attempting")
                self._actor_profile_attempt_phases[attempt_key] = "attempting"
            self._worker_run_active = True
        try:
            return engine.run_stage(request, planned_invocation=planned_invocation)
        finally:
            with self._mutex:
                self._worker_run_active = False
                self._settle_v2_commands_locked()

    def append_a2_runtime_transition(self, *, logical_id: str, plan,
                                     event: Mapping[str, Any]) -> Mapping[str, Any]:
        """Append or exactly replay one current-owner A2 phase event."""
        from . import a2_execution_state

        with self._mutex:
            self._require_active_locked()
            assert self._journal is not None
            try:
                from . import experiment_plan as experiment_plan_module
                if not isinstance(plan, experiment_plan_module.ExperimentPlan):
                    raise a2_execution_state.A2ExecutionStateRefused(
                        "A2 append requires an ExperimentPlan")
                plan = experiment_plan_module.ExperimentPlan.from_dict(plan.to_dict())
                if plan.campaign_id != self.resolved.campaign_id:
                    raise a2_execution_state.A2ExecutionStateRefused(
                        "A2 plan and controller campaign differ")
                event = a2_execution_state.validate_event_membership(event, plan)
                transition = a2_execution_state.make_transition(
                    campaign_id=self.resolved.campaign_id,
                    config_generation=self.config_generation,
                    config_digest=self.config_digest,
                    supervisor_incarnation=self.supervisor_incarnation,
                    logical_id=logical_id, event=event)
            except a2_execution_state.A2ExecutionStateRefused as exc:
                raise ControlRefused(f"invalid A2 runtime transition: {exc}") from exc
            execution_id = transition["execution_id"]
            prior_execution = self._a2_logical_executions.get(transition["logical_id"])
            if prior_execution not in {None, execution_id}:
                raise ControlRefused("A2 logical execution changes fixed identity")
            entries = self._a2_execution_entries.get(execution_id, [])
            event_digest = transition["event"]["event_digest"]
            prior = next((item for item in entries
                          if item.payload["event"].get("event_digest") == event_digest), None)
            if prior is not None:
                return copy.deepcopy(prior.payload["event"])
            try:
                a2_execution_state.project_transitions(
                    [*(item.payload for item in entries), transition])
            except a2_execution_state.A2ExecutionStateRefused as exc:
                raise ControlRefused(f"A2 runtime transition is inconsistent: {exc}") from exc
            try:
                entry = self._journal.append(
                    journal_module.KIND_A2_RUNTIME_EXECUTION, transition,
                    campaign_id=self.resolved.campaign_id, record_id=execution_id)
                self._verify_journal_layout(self.store / "journal")
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._a2_logical_executions[transition["logical_id"]] = execution_id
            self._a2_execution_entries.setdefault(execution_id, []).append(
                copy.deepcopy(entry))
            if transition["event"]["phase"] == "anchor_bank" \
                    and transition["event"]["state"] == "SEALED":
                bank_digest = transition["event"]["payload"]["bank_digest"]
                sources = self._a2_bank_sources.setdefault(bank_digest, [])
                if execution_id not in sources:
                    sources.append(execution_id)
            return copy.deepcopy(transition["event"])

    def _a2_resolve_bank_source_locked(self, bank, *, target_plan_digest: str,
                                       target_frame_digest: str):
        from . import a2_execution_state, discovery_screen

        if not self._mutex._is_owned():
            raise ControlRefused("A2 bank source resolution requires controller lock")
        try:
            bank = discovery_screen.BaselineBank.from_dict(bank.to_dict())
        except Exception as exc:
            raise ControlRefused(f"A2 reused bank is invalid: {exc}") from exc
        matches = []
        for source_id in self._a2_bank_sources.get(bank.bank_digest, []):
            entries = self._a2_execution_entries.get(source_id)
            if not entries:
                continue
            try:
                reference = a2_execution_state.make_bank_reference(
                    source_values=[item.payload for item in entries],
                    source_journal_entry_ids=[item.event_id for item in entries],
                    target_plan_digest=target_plan_digest,
                    target_frame_digest=target_frame_digest)
                projection = a2_execution_state.project_transitions(
                    [item.payload for item in entries])
                seal = next(event for event in projection.events
                            if event["phase"] == "anchor_bank"
                            and event["state"] == "SEALED")
                stored = discovery_screen.BaselineBank.from_dict(seal["payload"])
            except (a2_execution_state.A2ExecutionStateRefused,
                    discovery_screen.DiscoveryScreenRefused, StopIteration):
                continue
            if (reference["source_frame_digest"] == target_frame_digest
                    and stored.to_dict() == bank.to_dict()):
                matches.append((reference, stored))
        if len(matches) != 1:
            raise ControlRefused(
                "A2 reused bank lacks one exact indexed original sealed source")
        return matches[0]

    def _bank_for_a2_reference_locked(self, reference: Mapping[str, Any]):
        from . import a2_execution_state, discovery_screen

        if not self._mutex._is_owned():
            raise ControlRefused("A2 bank reference reopening requires controller lock")
        source = self._a2_execution_entries.get(reference["source_execution_id"])
        if not source:
            raise ControlRefused("A2 bank reference source is missing")
        try:
            expected = a2_execution_state.make_bank_reference(
                source_values=[item.payload for item in source],
                source_journal_entry_ids=[item.event_id for item in source],
                target_plan_digest=reference["target_plan_digest"],
                target_frame_digest=reference["target_frame_digest"])
            projection = a2_execution_state.project_transitions(
                [item.payload for item in source])
            seal = next(event for event in projection.events
                        if event["phase"] == "anchor_bank"
                        and event["state"] == "SEALED")
            bank = discovery_screen.BaselineBank.from_dict(seal["payload"])
        except (a2_execution_state.A2ExecutionStateRefused,
                discovery_screen.DiscoveryScreenRefused, StopIteration) as exc:
            raise ControlRefused(f"A2 bank reference source cannot be reopened: {exc}") from exc
        if expected != dict(reference):
            raise ControlRefused("A2 bank reference differs from original source history")
        return bank

    def append_a2_bank_reference(self, *, logical_id: str, plan,
                                 frame_digest: str, bank) -> Mapping[str, Any]:
        """Durably bind one imported bank to its indexed original anchor history."""
        from . import a2_execution_state
        from . import experiment_plan as experiment_plan_module

        with self._mutex:
            self._require_active_locked()
            assert self._journal is not None
            if not isinstance(plan, experiment_plan_module.ExperimentPlan):
                raise ControlRefused("A2 bank reuse requires an ExperimentPlan")
            plan = experiment_plan_module.ExperimentPlan.from_dict(plan.to_dict())
            if plan.campaign_id != self.resolved.campaign_id:
                raise ControlRefused("A2 bank reuse plan and controller campaign differ")
            execution_id = a2_execution_state.execution_identity(
                campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation,
                config_digest=self.config_digest, logical_id=logical_id,
                plan_digest=plan.digest, frame_digest=frame_digest)
            prior_execution = self._a2_logical_executions.get(logical_id)
            if prior_execution not in {None, execution_id}:
                raise ControlRefused("A2 logical execution changes fixed identity")
            entries = self._a2_execution_entries.get(execution_id, [])
            if entries:
                try:
                    projection = a2_execution_state.project_transitions(
                        [item.payload for item in entries])
                except a2_execution_state.A2ExecutionStateRefused as exc:
                    raise ControlRefused(f"A2 bank reuse history is invalid: {exc}") from exc
                reference = projection.bank_reference
                if reference is None:
                    raise ControlRefused(
                        "A2 bank reference must precede every local phase event")
                expected, stored = self._a2_resolve_bank_source_locked(
                    bank, target_plan_digest=plan.digest,
                    target_frame_digest=frame_digest)
                if reference != expected:
                    raise ControlRefused("A2 bank reuse retry changes its original source")
                return {"bank_reference": copy.deepcopy(reference),
                        "bank": stored.to_dict()}
            reference, stored = self._a2_resolve_bank_source_locked(
                bank, target_plan_digest=plan.digest,
                target_frame_digest=frame_digest)
            transition = a2_execution_state.make_transition(
                campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation,
                config_digest=self.config_digest,
                supervisor_incarnation=self.supervisor_incarnation,
                logical_id=logical_id, event=reference)
            try:
                a2_execution_state.project_transitions([transition])
                entry = self._journal.append(
                    journal_module.KIND_A2_RUNTIME_EXECUTION, transition,
                    campaign_id=self.resolved.campaign_id, record_id=execution_id)
                self._verify_journal_layout(self.store / "journal")
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._a2_logical_executions[logical_id] = execution_id
            self._a2_execution_entries[execution_id] = [copy.deepcopy(entry)]
            return {"bank_reference": copy.deepcopy(reference),
                    "bank": stored.to_dict()}

    def _attest_a2_bank_reference(self, *, execution_id: str,
                                  bank_reference: Mapping[str, Any], bank) -> bool:
        from . import a2_execution_state, discovery_screen

        with self._mutex:
            self._require_active_locked()
            entries = self._a2_execution_entries.get(execution_id)
            if not entries:
                return False
            try:
                projection = a2_execution_state.project_transitions(
                    [item.payload for item in entries])
                if projection.bank_reference != bank_reference:
                    return False
                stored = self._bank_for_a2_reference_locked(bank_reference)
            except (a2_execution_state.A2ExecutionStateRefused,
                    discovery_screen.DiscoveryScreenRefused, ControlRefused):
                return False
            return stored.to_dict() == dict(bank)

    def _attest_a2_runtime_history(self, *, execution_id: str,
                                   plan_digest: str, frame_digest: str,
                                   events) -> bool:
        """Private half of discovery_screen's actual-owner verifier mint."""
        from . import a2_execution_state

        with self._mutex:
            self._require_active_locked()
            entries = self._a2_execution_entries.get(execution_id)
            if not entries:
                return False
            try:
                projection = a2_execution_state.project_transitions(
                    [item.payload for item in entries])
            except a2_execution_state.A2ExecutionStateRefused:
                return False
            return (projection.plan_digest == plan_digest
                    and projection.frame_digest == frame_digest
                    and tuple(projection.events) == tuple(events))

    def replay_a2_runtime_execution(self, *, logical_id: str,
                                    plan_digest: str,
                                    frame_digest: str) -> Mapping[str, Any]:
        """Return one bounded indexed replay plus live current-owner attestation."""
        from . import a2_execution_state, discovery_screen

        with self._mutex:
            self._require_active_locked()
            execution_id = a2_execution_state.execution_identity(
                campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation,
                config_digest=self.config_digest, logical_id=logical_id,
                plan_digest=plan_digest, frame_digest=frame_digest)
            prior_execution = self._a2_logical_executions.get(logical_id)
            if prior_execution not in {None, execution_id}:
                raise ControlRefused("A2 logical execution has identity drift")
            entries = self._a2_execution_entries.get(execution_id, [])
            if not entries:
                return {"execution_id": execution_id, "logical_id": logical_id,
                        "plan_digest": plan_digest, "frame_digest": frame_digest,
                        "events": (), "pending_intents": (), "sealed_phases": (),
                        "bank_reference": None, "reused_bank": None,
                        "bank_verifier": None,
                        "journal_entry_ids": (), "journal_cursor": 0,
                        "history_digest": None, "phase_verifier": None}
            try:
                projection = a2_execution_state.project_transitions(
                    [item.payload for item in entries])
                verifier = (discovery_screen.attest_controller_phase_history(
                    self, execution_id=execution_id, plan_digest=plan_digest,
                    frame_digest=frame_digest, events=projection.events)
                    if projection.events else None)
                reused_bank = None
                bank_verifier = None
                if projection.bank_reference is not None:
                    stored = self._bank_for_a2_reference_locked(
                        projection.bank_reference)
                    reused_bank = stored.to_dict()
                    bank_verifier = discovery_screen.attest_controller_bank_reference(
                        self, execution_id=execution_id,
                        bank_reference=projection.bank_reference, bank=reused_bank)
            except (a2_execution_state.A2ExecutionStateRefused,
                    discovery_screen.DiscoveryScreenRefused) as exc:
                raise ControlRefused(f"A2 runtime replay refused: {exc}") from exc
            return {"execution_id": execution_id, "logical_id": logical_id,
                    "plan_digest": plan_digest, "frame_digest": frame_digest,
                    "events": tuple(copy.deepcopy(list(projection.events))),
                    "pending_intents": tuple(copy.deepcopy(list(
                        projection.pending_intents))),
                    "sealed_phases": projection.sealed_phases,
                    "bank_reference": copy.deepcopy(projection.bank_reference),
                    "reused_bank": copy.deepcopy(reused_bank),
                    "bank_verifier": bank_verifier,
                    "journal_entry_ids": tuple(item.event_id for item in entries),
                    "journal_cursor": entries[-1].seq,
                    "history_digest": projection.history_digest,
                    "phase_verifier": verifier}

    def worker_terminal_for_request(self, *, request_id: str, plan_digest: str,
                                    lineage_id: str, stage_id: str):
        """Return one exact current-owner terminal without restoring result authority."""
        with self._mutex:
            self._require_active_locked()
            if self._worker_lifecycle is None:
                raise ControlRefused("worker lifecycle is unavailable")
            return self._worker_lifecycle.terminal_for_request(
                request_id=request_id, plan_digest=plan_digest,
                lineage_id=lineage_id, stage_id=stage_id)

    def worker_held_claim_receipt(self, terminal):
        """Return provider-authored held facts for an exact owned terminal."""
        with self._mutex:
            self._require_active_locked()
            if self._worker_lifecycle is None:
                raise ControlRefused("worker lifecycle is unavailable")
            return self._worker_lifecycle.trusted_held_claim_receipt(terminal)

    def read_worker_stdout(
            self, *, request_id: str, plan_digest: str, lineage_id: str,
            stage_id: str, worker_id: str, worker_generation: int,
            result_digest: str, max_bytes: int) -> bytes:
        """Read one exact terminal's pinned private stdout without exposing a path."""
        if (not isinstance(worker_id, str) or not worker_id
                or not isinstance(worker_generation, int)
                or isinstance(worker_generation, bool) or worker_generation < 1
                or not isinstance(result_digest, str) or len(result_digest) != 64
                or any(char not in "0123456789abcdef" for char in result_digest)
                or not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
                or max_bytes < 1 or max_bytes > _MAX_WORKER_STDOUT_BYTES):
            raise worker_lifecycle_module.LifecycleRefused(
                "retained worker stdout request is invalid or exceeds its byte ceiling")
        with self._mutex:
            self._require_active_locked()
            lifecycle = self._worker_lifecycle
            runtime = self._runtime_root
            lifetime = self._lifetime_token
            if lifecycle is None or runtime is None or lifetime is None:
                raise worker_lifecycle_module.LifecycleRefused(
                    "retained worker stdout owner is unavailable")
            terminal = lifecycle.terminal_for_request(
                request_id=request_id, plan_digest=plan_digest,
                lineage_id=lineage_id, stage_id=stage_id)
            if (terminal is None
                    or terminal.worker_id != worker_id
                    or terminal.worker_generation != worker_generation
                    or terminal.result_digest != result_digest):
                raise worker_lifecycle_module.LifecycleRefused(
                    "retained worker stdout terminal/result identity differs")
            pinned_identity = dict(lifecycle.stdout_identity_for_terminal(terminal))
            pinned_proof = dict(lifecycle.stdout_proof_for_terminal(terminal))
            if pinned_proof["stdout_truncated"]:
                raise worker_lifecycle_module.LifecycleRefused(
                    "retained worker stdout was truncated")

        leaf = lifecycle.stdout_leaf(worker_id, worker_generation)
        fd = -1
        try:
            runtime.verify()
            fd = runtime.open_leaf(leaf, os.O_RDONLY | os.O_NONBLOCK)
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            raw, observed_identity = read_stable_fd(
                fd, limit=max_bytes, require_owned=True, require_single_link=True)
            named = os.stat(leaf, dir_fd=runtime.fd, follow_symlinks=False)
            if (not stat.S_ISREG(named.st_mode)
                    or stat.S_IMODE(named.st_mode) != 0o600
                    or observed_identity != pinned_identity
                    or object_identity(named) != observed_identity):
                raise SecureRuntimeError("retained worker stdout object was replaced")
            runtime.verify()
        except (OSError, SecureRuntimeError) as exc:
            raise worker_lifecycle_module.LifecycleRefused(
                "retained worker stdout is unavailable, oversized, or replaced") from exc
        finally:
            if fd >= 0:
                os.close(fd)

        if (len(raw) != pinned_proof["stdout_bytes"]
                or hashlib.sha256(raw).hexdigest() != pinned_proof["stdout_sha256"]):
            raise worker_lifecycle_module.LifecycleRefused(
                "retained worker stdout content differs from authenticated outcome")

        with self._mutex:
            self._require_active_locked()
            if (self._lifetime_token is not lifetime
                    or self._worker_lifecycle is not lifecycle
                    or self._runtime_root is not runtime):
                raise worker_lifecycle_module.LifecycleRefused(
                    "retained worker stdout owner changed during read")
            current = lifecycle.terminal_for_request(
                request_id=request_id, plan_digest=plan_digest,
                lineage_id=lineage_id, stage_id=stage_id)
            if (current != terminal
                    or dict(lifecycle.stdout_identity_for_terminal(current))
                       != pinned_identity
                    or dict(lifecycle.stdout_proof_for_terminal(current))
                       != pinned_proof):
                raise worker_lifecycle_module.LifecycleRefused(
                    "retained worker stdout terminal changed during read")
        return raw

    def worker_attempt_status(self, *, request_id: str, plan_digest: str,
                              lineage_id: str, stage_id: str) -> str:
        """Classify an exact request; only an issued-before-acquisition absence is proof."""
        with self._mutex:
            self._require_active_locked()
            if self._worker_lifecycle is None:
                raise ControlRefused("worker lifecycle is unavailable")
            keys = ("request_id", "plan_digest", "lineage_id", "stage_id")
            expected = (request_id, plan_digest, lineage_id, stage_id)
            active = (*self._active_acquisition_events, *self._active_worker_events)
            if self._worker_run_active or any(
                    tuple(row.get(key) for key in keys) == expected for row in active):
                return "unresolved"
            terminal = self._worker_lifecycle.terminal_for_request(
                request_id=request_id, plan_digest=plan_digest,
                lineage_id=lineage_id, stage_id=stage_id)
            if terminal is not None:
                return "terminal"
            attempt_key = self._worker_attempt_key_fields(
                self._worker_lifecycle.binding, request_id, plan_digest,
                lineage_id, stage_id)
            if attempt_key == self._worker_no_acquisition:
                return "not_acquired"
            if attempt_key[5:] in self._worker_historical_logical_attempts:
                return "unknown"
            for issued in self._driver_issued.values():
                if issued["transition_id"] in self._driver_settled:
                    continue
                selection = issued["selection"]
                proposal = selection.get("proposal") if isinstance(selection, Mapping) else None
                if (not isinstance(proposal, Mapping)
                        or proposal.get("proposal_id") != request_id
                        or lineage_id != f"driver:{issued['transition_id']}"
                        or stage_id != f"runtime:{issued['transition_id']}"):
                    continue
                work = issued["catalog"]["work_by_stage_digest"].get(
                    selection.get("proposal_digest"))
                if (isinstance(work, Mapping)
                        and work.get("stage_plan_binding") == "experiment_plan"
                        and work.get("stage_plan_digest") == plan_digest):
                    return "not_acquired"
            return "unknown"

    def reconcile_workers(self):
        """Reconcile retained v2 ownership outside the controller command lock."""
        with self._mutex:
            self._require_active_locked()
            if self.snapshot_version not in {2, 3} or self._worker_lifecycle is None:
                raise ControlRefused("worker reconciliation requires explicit snapshot v2")
            if self._worker_run_active:
                raise ControlRefused("worker execution is already active")
            engine = self._worker_lifecycle
            events = copy.deepcopy(self._active_worker_events)
            acquisition_events = copy.deepcopy(self._active_acquisition_events)
            self._worker_run_active = True
        try:
            if acquisition_events:
                status = engine.reconcile_acquisition(acquisition_events, events)
                if status != "handoff":
                    return None
            return engine.reconcile(events)
        finally:
            with self._mutex:
                self._worker_run_active = False
                self._settle_v2_commands_locked()

    def worker_result_fence(self, terminal):
        with self._mutex:
            self._require_active_locked()
            if self._worker_lifecycle is None:
                raise ControlRefused("worker lifecycle is unavailable")
            return self._worker_lifecycle.trusted_result_fence(terminal)

    def _append_v2_command_locked(self, phase: str, command: Mapping[str, Any],
                                  result: Mapping[str, Any]) -> None:
        assert self._journal is not None and self._worker_lifecycle is not None
        fenced = command.get("schema") == campaign_command_v2.COMMAND_SCHEMA
        validator = (campaign_command_v2.validate_transition if fenced
                     else worker_lifecycle_module.validate_command_transition_v2)
        row = validator({
            "schema": (campaign_command_v2.TRANSITION_SCHEMA if fenced
                       else worker_lifecycle_module.COMMAND_TRANSITION_SCHEMA),
            "phase": phase, "campaign_id": self.resolved.campaign_id,
            "config_digest": self.config_digest,
            "config_generation": self.config_generation,
            "supervisor_id": self._worker_lifecycle.binding.supervisor_id,
            "supervisor_incarnation": self.supervisor_incarnation,
            "control_revision": result["control_revision"],
            "occurred_at": self.clock(), "command": dict(command),
            "result": dict(result),
        })
        try:
            entry = self._journal.append(
                journal_module.KIND_CAMPAIGN_COMMAND_V3 if fenced
                else journal_module.KIND_CAMPAIGN_COMMAND_V2, row)
            self._verify_journal_layout(self.store / "journal")
        except BaseException:
            self._poisoned = True
            raise
        self._journal_cursor = entry.seq

    def _settle_v2_commands_locked(self) -> None:
        if self.snapshot_version not in {2, 3} or self._worker_run_active \
                or self._worker_projection.active \
                or self._acquisition_projection.pending is not None \
                or self._maintenance_state.owned:
            return
        for request_id, prior in tuple(self._command_results.items()):
            if prior.get("completed") is not False:
                continue
            operation = prior["operation"]
            if operation not in {"pause", "drain"}:
                continue
            result = dict(prior)
            result.update({
                "completed": True, "completed_at": self.clock(),
                "completion_reason": "owned workers quiesced and claims released",
                "observed_state": "paused" if operation == "pause" else "drained",
            })
            command = copy.deepcopy(self._command_requests.get(request_id)) or {
                "schema": COMMAND_SCHEMA, "campaign_id": self.resolved.campaign_id,
                "config_generation": self.config_generation, "request_id": request_id,
                "operation": operation, "payload": {},
                "payload_digest": result["payload_digest"],
                "expected_control_revision": result["control_revision"] - 1,
            }
            result = worker_lifecycle_module.validate_command_result_v2(result)
            self._append_v2_command_locked("COMPLETED", command, result)
            self._command_results[request_id] = result
            if result["control_revision"] == self.control_revision:
                self.observed_state = result["observed_state"]
                self.prerequisite_reason = result["prerequisite_reason"]
        self._shutdown_condition.notify_all()

    def _supersede_pending_pause_locked(self) -> None:
        for request_id, prior in tuple(self._command_results.items()):
            if prior.get("operation") != "pause" or prior.get("completed") is not False:
                continue
            result = dict(prior)
            result.update({
                "completed": True, "completed_at": self.clock(),
                "completion_reason": "superseded by a later accepted drain",
                "desired_state": "drained", "observed_state": "draining",
                "prerequisite_reason": None,
            })
            command = copy.deepcopy(self._command_requests.get(request_id)) or {
                "schema": COMMAND_SCHEMA, "campaign_id": self.resolved.campaign_id,
                "config_generation": self.config_generation, "request_id": request_id,
                "operation": "pause", "payload": {},
                "payload_digest": result["payload_digest"],
                "expected_control_revision": result["control_revision"] - 1,
            }
            result = worker_lifecycle_module.validate_command_result_v2(result)
            self._append_v2_command_locked("COMPLETED", command, result)
            self._command_results[request_id] = result
        self._shutdown_condition.notify_all()

    def _apply_command_v2_locked(self, value: Mapping[str, Any]) -> dict[str, Any]:
        fenced = isinstance(value, Mapping) \
            and value.get("schema") == campaign_command_v2.COMMAND_SCHEMA
        row = (campaign_command_v2.validate_command(value) if fenced
               else validate_command(value))
        prior = self._command_results.get(row["request_id"])
        if prior is not None:
            accepted = self._command_requests.get(row["request_id"])
            if ((fenced and accepted != row)
                    or (not fenced and prior["payload_digest"] != row["payload_digest"])):
                raise ControlRefused("request_id was already used for different semantics")
            self._publish_snapshot_locked()
            return copy.deepcopy(prior)
        if self._shutdown_requested and row["operation"] == "resume":
            raise ControlRefused("supervisor shutdown has latched workload admission closed")
        if row["campaign_id"] != self.resolved.campaign_id \
                or row["config_generation"] != self.config_generation:
            raise ControlRefused("command campaign/config generation does not match")
        if fenced and (row["config_digest"] != self.config_digest
                       or row["supervisor_incarnation"] != self.supervisor_incarnation):
            raise ControlRefused("command config/supervisor identity does not match")
        pending = [result for result in self._command_results.values()
                   if result.get("completed") is False]
        if pending and not (row["operation"] == "drain"
                            and all(item["operation"] == "pause" for item in pending)):
            raise ControlRefused("a prior lifecycle command is accepted but incomplete")
        if row["expected_control_revision"] != self.control_revision:
            raise ControlRefused(f"stale control revision; current={self.control_revision}")
        if self.desired_state == "drained":
            raise ControlRefused("drained is terminal for this campaign generation")
        revision = self.control_revision + 1
        accepted_at = self.clock()
        active = (self._worker_run_active or bool(self._worker_projection.active)
                  or self._acquisition_projection.pending is not None)
        if row["operation"] == "resume":
            try:
                ready, reason = self.readiness_check()
            except Exception as exc:
                raise ControlRefused(f"readiness prerequisite check failed: {exc}") from exc
            if type(ready) is not bool or (reason is not None and
                    (not isinstance(reason, str) or not reason.strip())):
                raise ControlRefused("readiness prerequisite returned malformed result")
            desired = "running"
            observed = "running" if ready else "waiting_prerequisite"
            prerequisite = None if ready else (reason or "prerequisite unavailable")
            completed = True
            completion_reason = "running" if ready else "waiting on named prerequisite"
        else:
            desired = "paused" if row["operation"] == "pause" else "drained"
            prerequisite = None
            completed = not active
            observed = (desired if completed
                        else ("pausing" if row["operation"] == "pause" else "draining"))
            completion_reason = ("already quiescent" if completed else None)
        result = worker_lifecycle_module.validate_command_result_v2({
            "schema": worker_lifecycle_module.COMMAND_RESULT_SCHEMA,
            "request_id": row["request_id"], "operation": row["operation"],
            "payload_digest": row["payload_digest"], "accepted": True,
            "accepted_at": accepted_at, "completed": completed,
            "completed_at": accepted_at if completed else None,
            "completion_reason": completion_reason,
            "control_revision": revision, "desired_state": desired,
            "observed_state": observed, "prerequisite_reason": prerequisite,
        })
        old_revision = self.control_revision
        self.control_revision = revision
        try:
            self._append_v2_command_locked("ACCEPTED", row, result)
        except BaseException:
            self.control_revision = old_revision
            raise
        self.desired_state, self.observed_state = desired, observed
        self.prerequisite_reason = prerequisite
        self._command_results[row["request_id"]] = result
        self._command_requests[row["request_id"]] = copy.deepcopy(row)
        if row["operation"] == "drain":
            self._supersede_pending_pause_locked()
        self._publish_snapshot_locked()
        self._shutdown_condition.notify_all()
        return copy.deepcopy(result)

    def request_shutdown_drain(self) -> dict[str, Any]:
        """Latch shutdown and reuse or persist the compatible durable drain."""
        with self._mutex:
            self._require_active_locked()
            self._shutdown_requested = True
            drains = [result for result in self._command_results.values()
                      if result.get("accepted") is True
                      and result.get("operation") == "drain"
                      and result.get("desired_state") == "drained"]
            if drains:
                prior = max(drains, key=lambda result: result["control_revision"])
                if prior["control_revision"] != self.control_revision \
                        or self.desired_state != "drained":
                    raise ControlRefused("durable drain is not the current control state")
                self._shutdown_drain_request_id = prior["request_id"]
                return copy.deepcopy(prior)
            if self.snapshot_version == 1:
                request_id = (f"service-sigterm-drain:{self.resolved.campaign_id}:"
                              f"{self.config_generation}:{self.supervisor_incarnation}")
                command = {
                    "schema": COMMAND_SCHEMA,
                    "campaign_id": self.resolved.campaign_id,
                    "config_generation": self.config_generation,
                    "request_id": request_id, "operation": "drain", "payload": {},
                    "expected_control_revision": self.control_revision,
                }
                command["payload_digest"] = command_digest(
                    operation="drain", payload={}, campaign_id=self.resolved.campaign_id,
                    config_generation=self.config_generation)
                result = self.apply_command(command)
                self._shutdown_drain_request_id = request_id
                return result
            request_id = (f"service-sigterm-drain:{self.resolved.campaign_id}:"
                          f"{self.config_generation}:{self.supervisor_incarnation}")
            prior_command = self._command_requests.get(request_id)
            if prior_command is not None:
                result = self._apply_command_v2_locked(copy.deepcopy(prior_command))
                self._shutdown_drain_request_id = request_id
                return result
            command = {
                "schema": campaign_command_v2.COMMAND_SCHEMA,
                "campaign_id": self.resolved.campaign_id,
                "config_generation": self.config_generation,
                "config_digest": self.config_digest,
                "supervisor_incarnation": self.supervisor_incarnation,
                "request_id": request_id, "operation": "drain", "payload": {},
                "expected_control_revision": self.control_revision,
            }
            command["payload_digest"] = campaign_command_v2.command_digest(
                operation="drain", payload={}, campaign_id=self.resolved.campaign_id,
                config_generation=self.config_generation, config_digest=self.config_digest,
                supervisor_incarnation=self.supervisor_incarnation, request_id=request_id,
                expected_control_revision=self.control_revision)
            result = self._apply_command_v2_locked(command)
            self._shutdown_drain_request_id = request_id
            return result

    def await_shutdown_drain(self, deadline: float) -> dict[str, Any]:
        """Wait boundedly for durable drain; elapsed time grants no cleanup authority."""
        if (not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
                or not math.isfinite(float(deadline))):
            raise ControlRefused("shutdown deadline is invalid")
        with self._shutdown_condition:
            while True:
                self._require_active_locked()
                request_id = self._shutdown_drain_request_id
                if request_id is None:
                    raise ControlRefused("shutdown drain was not requested")
                result = self._command_results.get(request_id)
                clean = bool(result and result.get("completed") is True
                             and result.get("observed_state") == "drained"
                             and self._shutdown_requested
                             and self.desired_state == "drained"
                             and not self._worker_run_active
                             and not self._worker_projection.active
                             and self._acquisition_projection.pending is None
                             and not self._maintenance_state.owned)
                if clean:
                    return copy.deepcopy(result)
                remaining = float(deadline) - time.monotonic()
                if remaining <= 0:
                    raise ControlRefused(
                        "shutdown drain deadline expired; ownership retained")
                self._shutdown_condition.wait(timeout=remaining)

    def reconcile_shutdown_ownership(self) -> bool:
        """Use only this controller's lifecycle engine to reconcile stranded ownership."""
        with self._mutex:
            self._require_active_locked()
            if self._worker_run_active:
                return False
            needs_recovery = bool(
                self._worker_projection.active
                or self._acquisition_projection.pending is not None)
            if not needs_recovery:
                self._settle_v2_commands_locked()
                return True
        self.reconcile_workers()
        return True

    def register_native_capture(self, validator: NativeCaptureValidator) -> None:
        """Install the explicit trusted worker-result consumer for this lifetime.

        The controller has no worker, grant, or supervisor-id authority of its own.
        Its lifecycle owner must construct this validator from those trusted sources.
        """
        if not isinstance(validator, NativeCaptureValidator):
            raise TypeError("validator must be NativeCaptureValidator")
        with self._mutex:
            self._require_active_locked()
            binding = validator.binding
            if (binding.campaign_id != self.resolved.campaign_id
                    or binding.config_digest != self.config_digest
                    or binding.config_generation != self.config_generation
                    or binding.supervisor_incarnation != self.supervisor_incarnation):
                raise ControlRefused("native capture validator binding is not current")
            if self._native_validator is not None:
                raise ControlRefused("native capture validator is already installed")
            self._native_validator = validator

    def native_capture(self, measurement_id: str):
        """Return a detached recorded event; this does not restore eligibility."""
        with self._mutex:
            self._require_active_locked()
            if not isinstance(measurement_id, str):
                raise TypeError("measurement_id must be text")
            return copy.deepcopy(self._native_records.get(measurement_id))

    @contextmanager
    def native_capture_callback(self):
        """Yield a same-thread, lifetime-bound callback for NativeMeasurementSink."""
        with self._mutex:
            self._require_active_locked()
            token = object()
            lifetime = self._lifetime_token
            thread_id = threading.get_ident()
            self._native_capabilities.add(token)

        def capture(measurement_id: str, payload: Mapping[str, Any]):
            if threading.get_ident() != thread_id:
                raise ControlRefused("native capture callback belongs to another thread")
            return self._capture_native(
                measurement_id, payload, token=token,
                lifetime_token=lifetime, thread_id=thread_id)

        try:
            yield capture
        finally:
            with self._mutex:
                self._native_capabilities.discard(token)

    def _capture_native(
            self, measurement_id: str, payload: Mapping[str, Any], *, token: object,
            lifetime_token: object | None, thread_id: int):
        """Prevalidate v2 artifacts outside the mutex; retain v1's exact path."""
        is_v2 = isinstance(payload, Mapping) and payload.get("schema") in (
            "epyc.autokernel.unified_arm_capture.v2", "epyc.autokernel.unified_arm_capture.v3")
        if not is_v2:
            with self._mutex:
                return self._capture_native_locked(
                    measurement_id, payload, token=token,
                    lifetime_token=lifetime_token, thread_id=thread_id)
        if not isinstance(measurement_id, str) or not isinstance(payload, Mapping):
            raise NativeCaptureRefused("native capture identity/payload is malformed")
        try:
            supplied_digest = schemas.content_hash(dict(payload))
        except Exception as exc:
            raise NativeCaptureRefused("native capture payload is not canonical JSON") from exc
        with self._mutex:
            self._require_native_capability_locked(
                token=token, lifetime_token=lifetime_token, thread_id=thread_id)
            prior = self._native_records.get(measurement_id)
            if prior is not None:
                if self._native_payload_digests[measurement_id] != supplied_digest:
                    raise NativeCaptureRefused(
                        "measurement_id was already used for different capture bytes")
                return copy.deepcopy(prior)
            if self.desired_state == "drained":
                raise NativeCaptureRefused("drained controller refuses new native captures")
            validator = self._native_validator
            if validator is None:
                raise NativeCaptureRefused(
                    "trusted native capture validator is not connected")
        prevalidated = validator.prevalidate(measurement_id, payload)
        with self._mutex:
            return self._capture_native_locked(
                measurement_id, payload, token=token,
                lifetime_token=lifetime_token, thread_id=thread_id,
                supplied_digest=supplied_digest, prevalidated=prevalidated,
                expected_validator=validator)

    def _require_native_capability_locked(self, *, token: object,
                                          lifetime_token: object | None,
                                          thread_id: int) -> None:
        if (threading.get_ident() != thread_id or not self._mutex._is_owned()
                or lifetime_token is None or lifetime_token is not self._lifetime_token
                or token not in self._native_capabilities):
            raise ControlRefused("native capture callback is not current owner")
        self._require_active_locked()

    def _capture_native_locked(
            self, measurement_id: str, payload: Mapping[str, Any], *, token: object,
            lifetime_token: object | None, thread_id: int,
            supplied_digest: str | None = None,
            prevalidated: PrevalidatedNativeCapture | None = None,
            expected_validator: NativeCaptureValidator | None = None):
        self._require_native_capability_locked(
            token=token, lifetime_token=lifetime_token, thread_id=thread_id)
        if not isinstance(measurement_id, str) or not isinstance(payload, Mapping):
            raise NativeCaptureRefused("native capture identity/payload is malformed")
        if supplied_digest is None:
            try:
                supplied_digest = schemas.content_hash(dict(payload))
            except Exception as exc:
                raise NativeCaptureRefused(
                    "native capture payload is not canonical JSON") from exc
        prior = self._native_records.get(measurement_id)
        if prior is not None:
            if self._native_payload_digests[measurement_id] != supplied_digest:
                raise NativeCaptureRefused(
                    "measurement_id was already used for different capture bytes")
            return copy.deepcopy(prior)
        if self._maintenance_state.owned:
            raise NativeCaptureRefused(
                "new native dependency publication is fenced by maintenance exclusion")
        if self.desired_state == "drained":
            raise NativeCaptureRefused("drained controller refuses new native captures")
        if self._native_validator is None:
            raise NativeCaptureRefused(
                "trusted native capture validator is not connected")
        if prevalidated is None:
            validated = self._native_validator.validate(measurement_id, payload)
        else:
            if self._native_validator is not expected_validator:
                raise NativeCaptureRefused(
                    "native capture validator changed during prevalidation")
            if self._worker_lifecycle is None:
                raise NativeCaptureRefused("worker lifecycle is unavailable")
            try:
                fence, grant_generation = self._worker_lifecycle.current_result_owner(
                    worker_id=prevalidated.context.worker_id,
                    worker_generation=prevalidated.context.worker_incarnation,
                    grant_id=prevalidated.context.grant_id,
                    container_id=prevalidated.context.container_id,
                    lineage_id=prevalidated.context.lineage_id)
            except Exception as exc:
                raise NativeCaptureRefused(
                    "v2 native capture current lifecycle owner is unavailable") from exc
            owner_token = CurrentOwnerToken(
                prevalidated.measurement_id, prevalidated.payload_digest,
                grant_generation, fence)
            validated = self._native_validator.validate_prevalidated(
                prevalidated, owner_token)
        row = validated.payload()
        if validated.payload_digest != supplied_digest:
            raise NativeCaptureRefused("validated native payload differs from callback bytes")
        assert self._journal is not None
        try:
            entry = self._journal.append(
                journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED, row,
                record_id=measurement_id)
            self._verify_journal_layout(self.store / "journal")
            self._journal_cursor = entry.seq
            self._native_records[measurement_id] = copy.deepcopy(entry)
            self._native_payload_digests[measurement_id] = supplied_digest
        except BaseException:
            self._poisoned = True
            raise
        return copy.deepcopy(entry)

    def _maintenance_require_token_locked(
            self, token: maintenance_module.ExclusionToken) -> None:
        if not isinstance(token, maintenance_module.ExclusionToken):
            raise TypeError("token must be an ExclusionToken")
        current = self._maintenance_state.token
        if not self._maintenance_state.owned or current is None or token != current:
            raise ControlRefused("maintenance exclusion token is not the indexed owner")
        if (token.campaign_id != self.resolved.campaign_id
                or token.config_digest != self.config_digest
                or token.config_generation != self.config_generation
                or token.supervisor_id != self._maintenance_supervisor_id()
                or token.supervisor_incarnation != self.supervisor_incarnation):
            raise ControlRefused("maintenance exclusion token has stale owner binding")

    def _maintenance_append_locked(self, row: Mapping[str, Any]):
        self._require_active_locked()
        validated = maintenance_module.validate_event(row)
        try:
            projected = maintenance_module.project_events(
                [*self._maintenance_events, validated])
        except maintenance_module.MaintenanceExecutionRefused as exc:
            raise ControlRefused(f"invalid maintenance transition: {exc}") from exc
        assert self._journal is not None
        try:
            entry = self._journal.append(
                journal_module.KIND_MAINTENANCE_EXECUTION, validated)
            self._verify_journal_layout(self.store / "journal")
        except BaseException:
            self._poisoned = True
            raise
        self._journal_cursor = entry.seq
        self._maintenance_events.append(copy.deepcopy(validated))
        self._maintenance_state = projected
        return copy.deepcopy(entry)

    def maintenance_admit(self, job):
        """Reserve only from a complete native catalog; unavailable until connected."""
        if not isinstance(job, maintenance_module.consumer.RetentionJob):
            raise TypeError("job must be a native RetentionJob")
        with self._mutex:
            self._require_active_locked()
            if (self._worker_run_active or self._worker_projection.active
                    or self._acquisition_projection.pending is not None):
                raise ControlRefused("worker ownership prevents maintenance admission")
            if self._candidate_pending is not None or self._candidate_context_token is not None:
                raise ControlRefused("candidate mutation prevents maintenance admission")
            if self._maintenance_state.owned:
                previous = self._maintenance_state.token
                if self._maintenance_state.phase != "UNRESOLVED" or previous is None:
                    raise ControlRefused("maintenance exclusion is already owned")
                if (job.plan.snapshot_id != previous.snapshot_id
                        or job.plan.snapshot_generation != previous.snapshot_generation
                        or job.plan.snapshot_digest != previous.snapshot_digest
                        or job.plan.plan_digest != previous.plan_digest
                        or job.policy_digest != previous.policy_digest
                        or job.selected_artifact_ids != previous.selected_artifact_ids):
                    raise ControlRefused(
                        "maintenance recovery job differs from unresolved intent")
                token = maintenance_module.ExclusionToken(
                    "maintenance-recovery-" + schemas.content_hash({
                        "predecessor": previous.token_digest,
                        "supervisor_incarnation": self.supervisor_incarnation,
                    })[:24],
                    self.resolved.campaign_id, self.config_digest,
                    self.config_generation, self._maintenance_supervisor_id(),
                    self.supervisor_incarnation, previous.snapshot_id,
                    previous.snapshot_generation, previous.snapshot_digest,
                    previous.plan_digest, previous.policy_digest,
                    previous.selected_artifact_ids, self.clock(), previous.token_digest)
                self._maintenance_append_locked(maintenance_module.make_event(
                    "INTENT", token, occurred_at=self.clock()))
                self._maintenance_tombstone_intent = True
                return token
            prepared = self._prepared_retention_jobs.get(job.plan.plan_digest)
            frontier = self._retention_frontier_locked()
            if (prepared is None or prepared[0] != frontier
                    or job.plan.snapshot_digest != prepared[1]
                    or job.policy_digest != prepared[2]
                    or tuple(job.selected_artifact_ids) != prepared[3]):
                raise ControlRefused(
                    "maintenance job lacks exact current native catalog preparation")
            token = maintenance_module.ExclusionToken(
                "maintenance-" + job.plan.plan_digest[:24],
                self.resolved.campaign_id, self.config_digest,
                self.config_generation, self._maintenance_supervisor_id(),
                self.supervisor_incarnation, job.plan.snapshot_id,
                job.plan.snapshot_generation, job.plan.snapshot_digest,
                job.plan.plan_digest, job.policy_digest,
                job.selected_artifact_ids, self.clock(), None)
            self._maintenance_append_locked(maintenance_module.make_event(
                "INTENT", token, occurred_at=self.clock()))
            self._maintenance_tombstone_intent = True
            return token

    def install_native_retention_catalog(
            self, seed, *, runtime_anchors, model_preparations, artifact_root,
            runtime_recipes=None, runtime_recipe_snapshots=None) -> dict[str, Any]:
        """Durably install the one token-built static catalog before driver work."""
        if not isinstance(seed, retention_catalog_module.NativeRetentionCatalogSeed):
            raise TypeError("seed must be NativeRetentionCatalogSeed")
        expected = retention_catalog_module.build_seed(
            self.resolved, runtime_anchors, config_digest=self.config_digest,
            model_preparations=model_preparations, runtime_recipes=runtime_recipes,
            runtime_recipe_snapshots=runtime_recipe_snapshots,
            artifact_root=artifact_root)
        if seed is not expected and seed != expected:
            raise ControlRefused("native retention catalog was not rederived from owning inputs")
        with self._mutex:
            self._require_active_locked()
            if self.snapshot_version != 3 or seed.campaign_id != self.resolved.campaign_id \
                    or seed.config_digest != self.config_digest \
                    or seed.manifest_digest != self.resolved.manifest_digest:
                raise ControlRefused("native retention catalog binding differs")
            if self._retention_catalog_event is not None:
                if self._retention_catalog_seed != seed:
                    raise ControlRefused("native retention catalog is already installed differently")
                return copy.deepcopy(self._retention_catalog_event)
            row = retention_catalog_module.make_install_event(
                seed, config_generation=self.config_generation,
                supervisor_incarnation=self.supervisor_incarnation)
            assert self._journal is not None
            try:
                entry = self._journal.append(
                    journal_module.KIND_RETENTION_CATALOG_INSTALLED, row,
                    record_id=seed.seed_digest)
                self._verify_journal_layout(self.store / "journal")
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._retention_catalog_seed = seed
            self._retention_catalog_event = copy.deepcopy(row)
            return copy.deepcopy(row)

    def _retention_frontier_locked(self) -> tuple[Any, ...]:
        seed = self._retention_catalog_seed
        if seed is None:
            raise ControlRefused("native retention catalog is unavailable")
        return (seed.seed_digest,
                schemas.content_hash(seed.to_dict()),
                self._candidate_projection_position,
                self._worker_lifecycle_revision,
                tuple(sorted(self._native_payload_digests.items())),
                tuple(sorted((key, row["transition_id"])
                             for key, row in self._driver_issued.items())),
                tuple(sorted(self._driver_settled)),
                self.supervisor_incarnation, self.control_revision)

    def native_retention_catalog_capture(self) -> Mapping[str, Any]:
        """Capture declarations and exact dynamic owner frontier; performs no file I/O."""
        with self._mutex:
            self._require_active_locked()
            frontier = self._retention_frontier_locked()
            return {
                "seed": self._retention_catalog_seed,
                "frontier": frontier,
                "candidate_records": copy.deepcopy(self._candidate_completed_records),
                "active_workers": copy.deepcopy(self._active_worker_events),
                "active_acquisitions": copy.deepcopy(self._active_acquisition_events),
                "native_records": copy.deepcopy(self._native_records),
                "driver_issued": copy.deepcopy(self._driver_issued),
                "driver_settled": copy.deepcopy(self._driver_settled),
            }

    def _native_retention_catalog_capture_locked(self, candidate_state) -> Mapping[str, Any]:
        """Capture a settled candidate state while its replay context owns the mutex."""
        if not self._mutex._is_owned() or self._candidate_context_token is None:
            raise ControlRefused("native retention capture requires candidate owner context")
        capture = dict(self.native_retention_catalog_capture())
        capture["candidate_state"] = copy.deepcopy(candidate_state.to_dict())
        return capture

    def validate_native_retention_frontier(self, frontier: tuple[Any, ...]) -> None:
        """Recheck a completed external catalog collection against current authority."""
        if not isinstance(frontier, tuple):
            raise TypeError("frontier must be a tuple")
        with self._mutex:
            self._require_active_locked()
            if frontier != self._retention_frontier_locked():
                raise ControlRefused("native retention catalog changed during collection")

    def bind_prepared_retention_job(self, job, *, prepared, policy) -> None:
        """Bind an externally collected immutable plan to the unchanged owner frontier."""
        from .retention_consumer import RetentionJob
        if (not isinstance(job, RetentionJob)
                or not isinstance(prepared, retention_catalog_module.PreparedCatalogView)):
            raise TypeError("prepared retention binding requires native typed capability")
        from . import retention as retention_module
        from . import retention_consumer as retention_consumer_module
        snapshot = retention_consumer_module.collect_native_snapshot(prepared.view)
        expected_plan = retention_module.plan_retention(snapshot)
        if (job.plan != expected_plan
                or job.policy_digest != retention_consumer_module._policy_digest(policy)
                or tuple(job.selected_artifact_ids) != job.plan.expirable_ids[:len(
                    job.selected_artifact_ids)]):
            raise ControlRefused("maintenance job differs from native preparation")
        with self._mutex:
            self._require_active_locked()
            current = self._retention_frontier_locked()
            if (prepared.frontier != current
                    or job.plan.snapshot_id != self._retention_catalog_seed.seed_digest
                    or job.plan.snapshot_generation != max(1, int(current[2]) + 1)
                    or job.plan.snapshot_digest != prepared.view_digest):
                raise ControlRefused("native retention catalog changed during preparation")
            self._prepared_retention_jobs[job.plan.plan_digest] = (
                current, prepared.view_digest, job.policy_digest,
                tuple(job.selected_artifact_ids))

    def native_retention_catalog_seed_digest(self) -> str | None:
        with self._mutex:
            self._require_active_locked()
            return (None if self._retention_catalog_seed is None
                    else self._retention_catalog_seed.seed_digest)

    def maintenance_revalidate(self, token, hold) -> None:
        with self._mutex:
            self._require_active_locked()
            self._maintenance_require_token_locked(token)
            maintenance_module._validate_hold(hold, token,
                                              previous=self._maintenance_state.hold)
            try:
                current = datetime.fromisoformat(self.clock().replace("Z", "+00:00"))
            except ValueError as exc:
                raise ControlRefused("controller clock is not ISO-8601") from exc
            if hold.deadline <= current.timestamp():
                raise ControlRefused("maintenance hold deadline expired")
            phase = self._maintenance_state.phase
            if phase not in {"INTENT", "PROVIDER_HELD", "MUTATION_REVALIDATED"}:
                raise ControlRefused("maintenance exclusion is not revalidatable")
            event = "PROVIDER_HELD" if phase == "INTENT" else "MUTATION_REVALIDATED"
            self._maintenance_append_locked(maintenance_module.make_event(
                event, token, hold=hold, occurred_at=self.clock()))

    def maintenance_append_tombstone(
            self, token, kind: str, payload: Mapping[str, Any], campaign_id: str | None):
        with self._mutex:
            self._require_active_locked()
            self._maintenance_require_token_locked(token)
            if self._maintenance_state.phase != "MUTATION_REVALIDATED":
                raise ControlRefused("tombstone append lacks immediate maintenance revalidation")
            if kind != journal_module.KIND_TOMBSTONE:
                raise ControlRefused("maintenance may append only native tombstones")
            if campaign_id != self.resolved.campaign_id:
                raise ControlRefused("maintenance tombstone campaign binding differs")
            assert self._journal is not None
            try:
                entry = self._journal.append(kind, payload, campaign_id=campaign_id)
                self._verify_journal_layout(self.store / "journal")
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            if payload.get("reclamation_state") == "intent":
                self._maintenance_tombstone_intent = True
            self._maintenance_state = replace(
                self._maintenance_state, phase="PROVIDER_HELD")
            return copy.deepcopy(entry)

    def maintenance_io_complete(self, token, cost) -> None:
        if not isinstance(cost, maintenance_module.MaintenanceCost):
            raise TypeError("cost must be MaintenanceCost")
        with self._mutex:
            self._require_active_locked()
            if (self._maintenance_state.token == token
                    and self._maintenance_state.phase == "IO_COMPLETE"
                    and self._maintenance_state.cost == cost):
                return
            self._maintenance_require_token_locked(token)
            if self._maintenance_state.phase not in {
                    "PROVIDER_HELD", "MUTATION_REVALIDATED"}:
                raise ControlRefused("maintenance I/O completion is out of order")
            self._maintenance_append_locked(maintenance_module.make_event(
                "IO_COMPLETE", token, hold=self._maintenance_state.hold, cost=cost,
                occurred_at=self.clock()))

    def maintenance_complete(self, token, accounting) -> None:
        if not isinstance(accounting, maintenance_module.AccountingReceipt):
            raise TypeError("accounting must be AccountingReceipt")
        with self._mutex:
            self._require_active_locked()
            if (self._maintenance_state.token == token
                    and self._maintenance_state.phase == "COMPLETED"
                    and self._maintenance_state.accounting_receipt_digest
                    == accounting.receipt_digest):
                return
            self._maintenance_require_token_locked(token)
            if (self._maintenance_state.phase != "IO_COMPLETE"
                    or accounting.token_digest != token.token_digest
                    or self._maintenance_state.hold is None
                    or accounting.hold_receipt_digest
                    != self._maintenance_state.hold.receipt_digest
                    or accounting.cost != self._maintenance_state.cost
                    or accounting.disposition != "complete"):
                raise ControlRefused("maintenance accounting is out of order or misbound")
            self._maintenance_append_locked(maintenance_module.make_event(
                "COMPLETED", token, hold=self._maintenance_state.hold,
                cost=self._maintenance_state.cost,
                accounting_receipt_digest=accounting.receipt_digest,
                occurred_at=self.clock()))
            self._maintenance_tombstone_intent = False
            self._settle_v2_commands_locked()

    def maintenance_abort(self, token, reason: str, receipt, hold=None) -> None:
        with self._mutex:
            self._require_active_locked()
            receipt_digest = getattr(receipt, "receipt_digest", None)
            if (self._maintenance_state.token == token
                    and self._maintenance_state.phase == "ABORTED"
                    and self._maintenance_state.reason == reason
                    and self._maintenance_state.abort_receipt_digest == receipt_digest):
                return
            self._maintenance_require_token_locked(token)
            if self._maintenance_tombstone_intent:
                raise ControlRefused("durable tombstone intent requires unresolved recovery")
            if isinstance(receipt, maintenance_module.NoHoldReceipt):
                if hold is not None or self._maintenance_state.phase != "INTENT":
                    raise ControlRefused("no-hold refusal is out of order")
                try:
                    maintenance_module._validate_no_hold(receipt, token)
                except maintenance_module.MaintenanceExecutionRefused as exc:
                    raise ControlRefused("no-hold refusal receipt is misbound") from exc
                event_hold = event_cost = accounting_digest = None
            elif isinstance(receipt, maintenance_module.AccountingReceipt):
                if not isinstance(hold, maintenance_module.HoldReceipt):
                    raise ControlRefused("aborted accounting lacks its provider hold")
                previous = self._maintenance_state.hold
                try:
                    maintenance_module._validate_hold(hold, token, previous=previous)
                    event_cost = maintenance_module.MaintenanceCost(0, 0, 0, 0)
                    maintenance_module._validate_accounting(
                        receipt, hold, token, event_cost, "aborted")
                except maintenance_module.MaintenanceExecutionRefused as exc:
                    raise ControlRefused("aborted accounting receipt is misbound") from exc
                event_hold = hold
                accounting_digest = receipt.receipt_digest
            else:
                raise TypeError("receipt must be AccountingReceipt or NoHoldReceipt")
            self._maintenance_append_locked(maintenance_module.make_event(
                "ABORTED", token, hold=event_hold, cost=event_cost,
                accounting_receipt_digest=accounting_digest,
                abort_receipt=receipt,
                reason=reason, occurred_at=self.clock()))
            self._settle_v2_commands_locked()

    def maintenance_unresolved(self, token, reason: str) -> None:
        with self._mutex:
            self._require_active_locked()
            if (self._maintenance_state.token == token
                    and self._maintenance_state.phase == "UNRESOLVED"
                    and self._maintenance_state.reason == reason):
                return
            self._maintenance_require_token_locked(token)
            self._maintenance_append_locked(maintenance_module.make_event(
                "UNRESOLVED", token, hold=self._maintenance_state.hold,
                cost=self._maintenance_state.cost, reason=reason,
                occurred_at=self.clock()))


    @property
    def command_results(self) -> dict[str, dict[str, Any]]:
        with self._mutex:
            self._require_active_locked()
            return copy.deepcopy(self._command_results)

    @property
    def producer_build(self) -> dict[str, Any]:
        with self._mutex:
            self._require_active_locked()
            return copy.deepcopy(self._producer_build)

    def _require_active_locked(self) -> None:
        if (not self._entered or self._poisoned or self._journal is None
                or self._lease_fd is None or self._runtime_root is None):
            state = "poisoned; replay required" if self._poisoned else "closed"
            raise ControlRefused(f"controller is {state}")
        self._verify_store()
        self._verify_journal_layout(self.store / "journal")

    def _verify_store(self) -> None:
        if self._runtime_root is None:
            raise ControlRefused("controller store is not pinned")
        try:
            self._runtime_root.verify()
        except SecureRuntimeError as exc:
            raise ControlRefused(f"service store identity changed: {exc}") from exc
        if self._lease_fd is None or self._lock_identity is None:
            raise ControlRefused("supervisor lock is not pinned")
        held = object_identity(os.fstat(self._lease_fd))
        if held != self._lock_identity:
            raise ControlRefused("held supervisor lock identity changed")
        named_fd = self._runtime_root.open_leaf(".supervisor.lock", os.O_RDONLY)
        try:
            named = object_identity(os.fstat(named_fd))
        finally:
            os.close(named_fd)
        if named != self._lock_identity:
            raise ControlRefused("named supervisor lock was replaced")

    def _verify_journal_layout(self, root: Path) -> None:
        try:
            root_info = os.lstat(root)
            if (not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode)
                    or root_info.st_uid != os.getuid()):
                raise ControlRefused("journal root has unsafe object identity")
            if self._journal_root_identity is not None:
                current = object_identity(root_info)
                stable_fields = ("dev", "ino", "uid", "mode")
                if any(current[field] != self._journal_root_identity[field]
                       for field in stable_fields):
                    raise ControlRefused("journal root identity changed")
            directories = [root]
            for name in (journal_module.ARCHIVE_DIRNAME, journal_module.CURSOR_DIRNAME):
                child = root / name
                if os.path.lexists(child):
                    info = os.lstat(child)
                    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                            or info.st_uid != os.getuid()):
                        raise ControlRefused(f"journal {name} directory is unsafe")
                    directories.append(child)
            for directory in directories:
                for child in directory.iterdir():
                    if child.name == journal_module.LOCK_NAME \
                            or (child.name.startswith("events")
                                and child.name.endswith(".jsonl")):
                        info = os.lstat(child)
                        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                                or info.st_uid != os.getuid() or info.st_nlink != 1):
                            raise ControlRefused(
                                f"journal critical file is unsafe: {child.name}")
        except OSError as exc:
            raise ControlRefused(f"cannot verify journal layout: {exc}") from exc

    def _append_event(self, event: str, data: Mapping[str, Any]):
        self._verify_store()
        if self._journal is None:
            raise ControlRefused("controller journal is unavailable")
        if self.snapshot_version in {2, 3} and event != "START":
            raise ControlRefused("versioned supervisor schema is restricted to START")
        schema = (journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA_V3
                  if self.snapshot_version == 3 else
                  journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA_V2
                  if self.snapshot_version == 2 else
                  journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA)
        entry = self._journal.append(journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, {
            "schema": schema,
            "event": event, "campaign_id": self.resolved.campaign_id,
            "config_generation": self.config_generation,
            "config_digest": self.config_digest,
            "supervisor_incarnation": self.supervisor_incarnation,
            "stream_epoch": self.stream_epoch,
            "control_revision": self.control_revision,
            "data": dict(data),
        })
        self._journal_cursor = entry.seq
        self._verify_journal_layout(self.store / "journal")
        return entry

    def _append_candidate_event_locked(
            self, payload: Mapping[str, Any], *, lifetime_token: object | None,
            context_token: object | None, thread_id: int):
        if (threading.get_ident() != thread_id
                or lifetime_token is None or lifetime_token is not self._lifetime_token
                or context_token is None
                or context_token is not self._candidate_context_token
                or not self._mutex._is_owned()):
            raise ControlRefused("candidate transaction capability is not current owner")
        self._require_active_locked()
        violations = journal_module._validate_native_payload(
            journal_module.KIND_CANDIDATE_TRANSACTION, payload)
        if violations:
            raise ControlRefused("invalid candidate transaction event: "
                                 + "; ".join(violations))
        if (payload.get("campaign_id") != self.resolved.campaign_id
                or payload.get("config_generation") != self.config_generation
                or payload.get("config_digest") != self.config_digest
                or payload.get("supervisor_incarnation") != self.supervisor_incarnation):
            raise ControlRefused("candidate transaction binding does not match controller")
        transaction_key = (str(payload["transaction_id"]), str(payload["operation"]),
                           str(payload["payload_digest"]))
        if payload["phase"] == "INTENT":
            if self._maintenance_state.owned:
                raise ControlRefused(
                    "candidate mutation is fenced by maintenance exclusion")
            if (self._candidate_pending is not None
                    or transaction_key[0] in self._candidate_completed):
                raise ControlRefused(
                    "candidate transaction intent overlaps or reuses an id")
        elif payload["phase"] == "PREPARED":
            if self._candidate_pending != transaction_key or self._candidate_prepared:
                raise ControlRefused(
                    "candidate preparation lacks exact active intent")
        elif self._candidate_pending != transaction_key or not self._candidate_prepared:
            raise ControlRefused(
                "candidate transaction completion lacks exact prepared intent")
        self._verify_store()
        if self._journal is None:
            raise ControlRefused("controller journal is unavailable")
        try:
            entry = self._journal.append(
                journal_module.KIND_CANDIDATE_TRANSACTION, payload,
                record_id=str(payload["transaction_id"]))
            self._verify_journal_layout(self.store / "journal")
        except BaseException:
            self._poisoned = True
            raise
        self._journal_cursor = entry.seq
        self._candidate_entries.append(copy.deepcopy(entry))
        if payload["phase"] == "INTENT":
            self._candidate_pending = transaction_key
            self._candidate_pending_payload = copy.deepcopy(payload)
            self._candidate_prepared = False
        elif payload["phase"] == "PREPARED":
            self._candidate_prepared = True
        else:
            self._candidate_completed.add(transaction_key[0])
            assert self._candidate_pending_payload is not None
            self._candidate_completed_records[transaction_key[0]] = {
                "intent": self._candidate_pending_payload,
                "completion": copy.deepcopy(payload),
            }
            self._candidate_pending = None
            self._candidate_pending_payload = None
            self._candidate_prepared = False
        return entry

    def _update_candidate_projection_cache_locked(
            self, value: Any, *, lifetime_token: object | None,
            context_token: object | None, thread_id: int, authority: object) -> None:
        if (threading.get_ident() != thread_id
                or authority is not _CANDIDATE_REPLAYER_TOKEN
                or lifetime_token is None or lifetime_token is not self._lifetime_token
                or context_token is None
                or context_token is not self._candidate_context_token
                or not self._mutex._is_owned()):
            raise ControlRefused("candidate projection cache writer is not current owner")
        self._require_active_locked()
        position = getattr(value, "position", None)
        if (not isinstance(position, int) or isinstance(position, bool)
                or not 0 <= position <= len(self._candidate_entries)):
            raise ControlRefused("candidate projection cache position is invalid")
        self._candidate_projection_cache = copy.deepcopy(value)
        self._candidate_projection_position = position

    def candidate_transaction(self, callback: Callable[[Any], Any]) -> Any:
        """Run one candidate operation under the sole controller/journal owner.

        The callback receives a short-lived append capability and a detached copy
        of the unprojected candidate-event suffix. It cannot retain that capability
        after return. Candidate append uncertainty poisons this incarnation until
        replay.
        """
        if not callable(callback):
            raise TypeError("candidate transaction callback must be callable")
        with self._mutex:
            self._require_active_locked()
            if self._candidate_context_token is not None:
                raise ControlRefused("nested candidate transaction is refused")
            assert self._journal is not None
            offset = self._candidate_projection_position
            entries = tuple(copy.deepcopy(self._candidate_entries[offset:]))
            self._verify_store()
            self._candidate_context_token = object()
            context = _CandidateTransactionContext(self, entries, offset)
            try:
                return callback(context)
            finally:
                context._close()
                self._candidate_context_token = None

    def _unified_driver_readiness_locked(self) -> dict[str, Any]:
        self._require_active_locked()
        if self._scheduler_engine is None:
            raise ControlRefused("unified scheduler is unavailable")
        provider_ready, provider_reason = self._cached_driver_readiness
        admission_open = self.desired_state == "running" and not self._poisoned
        reason = ("ready" if admission_open and provider_ready else
                  (provider_reason or self.prerequisite_reason
                   or f"admissions_closed:{self.desired_state}"))
        return {
            "schema": "epyc.autokernel.unified_driver_readiness.v1",
            "campaign_id": self.resolved.campaign_id,
            "config_digest": self.config_digest,
            "config_generation": self.config_generation,
            "supervisor_incarnation": self.supervisor_incarnation,
            "control_revision": self.control_revision,
            "admission_open": admission_open,
            "provider_available": provider_ready,
            "reason": reason,
            "scheduler_projection_digest":
                self._scheduler_engine.operational_projection().projection_digest,
        }

    def unified_driver_readiness(self) -> dict[str, Any]:
        with self._mutex:
            return copy.deepcopy(self._unified_driver_readiness_locked())

    def unified_driver_materialization_binding(
            self, *, catalog_id: str, transition_id: str,
            selection: Mapping[str, Any]) -> dict[str, Any]:
        """Return the current private artifact/capture base without execution authority."""
        with self._mutex:
            self._require_active_locked()
            if self.snapshot_version != 3 or self._supervisor_id is None:
                raise ControlRefused("unified runtime materialization requires snapshot v3")
            issued = self._driver_issued.get(catalog_id)
            issued_work = None
            if issued is not None:
                issued_selection = issued.get("selection")
                issued_catalog = issued.get("catalog")
                work_by_digest = (issued_catalog.get("work_by_stage_digest")
                                  if isinstance(issued_catalog, Mapping) else None)
                if isinstance(issued_selection, Mapping) and isinstance(work_by_digest, Mapping):
                    issued_work = work_by_digest.get(
                        issued_selection.get("proposal_digest"))
            if (issued is None or issued["transition_id"] != transition_id
                    or transition_id in self._driver_settled
                    or issued["selection"] != copy.deepcopy(selection)
                    or (issued["supervisor_incarnation"] != self.supervisor_incarnation
                        and (not isinstance(issued_work, Mapping)
                             or issued_work.get("kind") != "runtime_comparison"))
                    or issued["config_generation"] != self.config_generation
                    or issued["config_digest"] != self.config_digest):
                raise ControlRefused(
                    "runtime materialization requires a current exact issued selection")
            if self._driver_artifact_store is None:
                from .measurement_capture import ArtifactStore
                self._driver_artifact_store = ArtifactStore(
                    self.store / "unified-native-artifacts")
            return {
                "schema": "epyc.autokernel.unified_materialization_binding.v1",
                "campaign_id": self.resolved.campaign_id,
                "config_digest": self.config_digest,
                "config_generation": self.config_generation,
                "supervisor_id": self._supervisor_id,
                "supervisor_incarnation": self.supervisor_incarnation,
                "artifact_root": str(self._driver_artifact_store.root),
            }

    def unified_driver_pending_intent(self) -> Mapping[str, Any] | None:
        """Return the one exact replayed issued-but-unsettled driver record."""
        with self._mutex:
            self._require_active_locked()
            if self.snapshot_version != 3 or self._scheduler_engine is None:
                raise ControlRefused("pending unified intent requires snapshot v3")
            pending = [copy.deepcopy(row) for row in self._driver_issued.values()
                       if row["transition_id"] not in self._driver_settled]
            if len(pending) > 1:
                raise ControlRefused("multiple unsettled unified driver intents conflict")
            return None if not pending else pending[0]

    def refresh_unified_driver_readiness(self) -> dict[str, Any]:
        """Refresh provider readiness outside the controller's serialization mutex."""
        with self._mutex:
            self._require_active_locked()
            lifetime_token = self._lifetime_token
            incarnation = self.supervisor_incarnation
            readiness_check = self.readiness_check
        try:
            result = readiness_check()
        except Exception as exc:
            result = (False, f"driver readiness check failed: {exc}")
        if (not isinstance(result, tuple) or len(result) != 2
                or type(result[0]) is not bool or (result[1] is not None and (
                    not isinstance(result[1], str) or not result[1].strip()))):
            raise ControlRefused("driver readiness check returned malformed result")
        with self._mutex:
            self._require_active_locked()
            if (lifetime_token is None or lifetime_token is not self._lifetime_token
                    or incarnation != self.supervisor_incarnation
                    or readiness_check is not self.readiness_check):
                raise ControlRefused(
                    "driver readiness result belongs to a stale controller lifetime")
            self._cached_driver_readiness = (result[0], result[1])
            return copy.deepcopy(self._unified_driver_readiness_locked())

    def unified_driver_transaction(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Derive and durably issue one bounded scheduler selection under ownership."""
        with self._mutex:
            readiness = self._unified_driver_readiness_locked()
            if not isinstance(value, Mapping):
                raise ControlRefused("unified driver catalog must be a mapping")
            from . import unified_driver as driver_module
            raw = copy.deepcopy(dict(value))
            supplied_id = raw.pop("catalog_id", None)
            try:
                catalog = driver_module.PlanningCatalog(**raw)
            except Exception as exc:
                raise ControlRefused(f"invalid unified driver catalog: {exc}") from exc
            if supplied_id != catalog.catalog_id:
                raise ControlRefused("unified driver catalog_id differs from content")
            prior = self._driver_issued.get(catalog.catalog_id)
            if prior is not None:
                return {
                    "schema": driver_module.TRANSACTION_RECEIPT_SCHEMA,
                    "catalog_id": catalog.catalog_id,
                    "transition_id": prior["transition_id"], "status": "duplicate",
                    "selection": copy.deepcopy(prior["selection"]),
                }
            if not readiness["admission_open"]:
                raise DriverAdmissionClosed(f"admissions_closed:{self.desired_state}")
            if not readiness["provider_available"]:
                raise ControlRefused(readiness["reason"])
            if (catalog.campaign_digest != resolved_config_digest(self.resolved)
                    or dict(catalog.controller_binding) != readiness
                    or catalog.scheduler_projection_digest
                       != readiness["scheduler_projection_digest"]):
                raise ControlRefused("unified driver catalog binding is stale or foreign")
            assert self._scheduler_engine is not None
            preview = self._scheduler_engine.preview_selection(
                [scheduling.StageProposal.from_dict(item)
                 for item in catalog.stage_proposals], now=catalog.observed_at)
            transition_id = driver_module._digest({
                "catalog_id": catalog.catalog_id,
                "selection": preview.selection.to_dict()})
            if preview.selection.status != "selected":
                return {
                    "schema": driver_module.TRANSACTION_RECEIPT_SCHEMA,
                    "catalog_id": catalog.catalog_id, "transition_id": transition_id,
                    "status": "not_selected", "selection": preview.selection.to_dict(),
                }
            event = {
                "schema": journal_module.UNIFIED_DRIVER_ISSUED_SCHEMA,
                "catalog_id": catalog.catalog_id,
                "campaign_id": self.resolved.campaign_id,
                "config_generation": self.config_generation,
                "config_digest": self.config_digest,
                "supervisor_incarnation": self.supervisor_incarnation,
                "catalog": catalog.to_dict(), "selection": preview.selection.to_dict(),
                "prior_projection_digest": preview.prior.projection_digest,
                "after_projection_digest": preview.after.projection_digest,
                "transition_id": transition_id,
            }
            self._verify_store()
            assert self._journal is not None
            try:
                entry = self._journal.append(
                    journal_module.KIND_UNIFIED_DRIVER_ISSUED, event,
                    record_id=transition_id)
                self._verify_journal_layout(self.store / "journal")
                self._scheduler_engine.apply_preview(preview)
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._driver_issued[catalog.catalog_id] = copy.deepcopy(event)
            return {
                "schema": driver_module.TRANSACTION_RECEIPT_SCHEMA,
                "catalog_id": catalog.catalog_id, "transition_id": transition_id,
                "status": "accepted", "selection": preview.selection.to_dict(),
            }

    def register_target_profile_producer(self, producer) -> None:
        """Install the one producer capability; serialized receipts grant nothing."""
        from .target_profile_execution import TargetProfileExecution
        if (not isinstance(producer, TargetProfileExecution)
                or producer.controller is not self):
            raise TypeError("profile producer must be this owner's TargetProfileExecution")
        with self._mutex:
            self._require_active_locked()
            if self._actor_profile_producer is not None:
                raise ControlRefused("target-profile producer is already registered")
            self._actor_profile_producer = producer

    def actor_held_claim_receipt(self, terminal):
        """Bind provider-authored cost to an exact terminal for actor settlement."""
        supplied = self.worker_held_claim_receipt(terminal)
        if not isinstance(supplied, scheduling.HeldClaimReceipt):
            raise ControlRefused("actor held accounting receipt is untyped")
        if (supplied.ownership_generation != terminal.worker_generation
                or supplied.allocation_generation != terminal.grant_generation):
            raise ControlRefused("actor held accounting receipt binding differs")
        with self._mutex:
            self._require_active_locked()
            exact = self.worker_terminal_for_request(
                request_id=terminal.request_id, plan_digest=terminal.plan_digest,
                lineage_id=terminal.lineage_id, stage_id=terminal.stage_id)
            if exact != terminal:
                raise ControlRefused("actor held accounting terminal is not current")
            self._actor_trusted_held_receipts[terminal.request_id] = (supplied, terminal)
            return supplied

    def _selected_profile_work_locked(self, *, catalog_id: str, transition_id: str,
                                      request_digest: str,
                                      stage_plan_digest: str) -> tuple[dict, dict]:
        issued = self._driver_issued.get(catalog_id)
        if not isinstance(issued, Mapping) or issued.get("transition_id") != transition_id:
            raise ControlRefused("profile result lacks exact issued transition")
        selection = issued.get("selection", {})
        catalog = issued.get("catalog", {})
        work_map = catalog.get("work_by_stage_digest", {})
        work = work_map.get(selection.get("proposal_digest")) \
            if isinstance(work_map, Mapping) else None
        if (selection.get("status") != "selected" or not isinstance(work, Mapping)
                or work.get("kind") != "profile_preparation"
                or work.get("stage_plan_digest") != stage_plan_digest
                or actor_state_module.digest(work.get("payload")) != request_digest):
            raise ControlRefused("profile result differs from selected work")
        return copy.deepcopy(dict(issued)), copy.deepcopy(dict(work))

    def reserve_target_profile_execution(
            self, *, profile_request: Mapping[str, Any], profile_request_digest: str,
            catalog_id: str, transition_id: str, stage_plan_digest: str,
            max_output_bytes: int, selected_work=None):
        """Admit exact current selected profile work before any producer I/O."""
        from .target_profile_execution import TargetProfileExecutionReservation
        if (actor_state_module.digest(profile_request) != profile_request_digest
                or isinstance(max_output_bytes, bool)
                or not isinstance(max_output_bytes, int) or max_output_bytes < 1):
            raise ControlRefused("profile execution admission is malformed")
        request_id = "profile-" + profile_request_digest[:24]
        stage_id = "target-profile-" + profile_request_digest[:24]
        with self._mutex:
            self._require_active_locked()
            if self.desired_state != "running":
                raise ControlRefused("profile execution requires running controller")
            self._selected_profile_work_locked(
                catalog_id=catalog_id, transition_id=transition_id,
                request_digest=profile_request_digest,
                stage_plan_digest=stage_plan_digest)
            if self.snapshot_version == 3 and selected_work is None:
                raise ControlRefused(
                    "unified profile execution requires public selected profile advice")
            if selected_work is not None:
                from . import unified_driver
                if not isinstance(selected_work, unified_driver.SelectedProfileWork):
                    raise ControlRefused("selected profile advice is untyped")
                expected_binding = self.unified_driver_materialization_binding(
                    catalog_id=catalog_id, transition_id=transition_id,
                    selection=selected_work.selection.to_dict())
                if (selected_work.catalog_id != catalog_id
                        or selected_work.transition_id != transition_id
                        or selected_work.profile_request.to_dict() != dict(profile_request)
                        or selected_work.stage_plan_digest != stage_plan_digest
                        or dict(selected_work.controller_binding) != expected_binding
                        or selected_work.execution_authorized is not False):
                    raise ControlRefused("selected profile advice binding differs")
            if any(item.request_id == request_id
                   for item in self._actor_profile_execution_reservations.values()):
                raise ControlRefused("profile execution is already admitted")
            attempt_status = self.worker_attempt_status(
                request_id=request_id, plan_digest=stage_plan_digest,
                lineage_id=transition_id, stage_id=stage_id)
            attempt_key = self._worker_attempt_key_fields(
                self._worker_lifecycle.binding, request_id, stage_plan_digest,
                transition_id, stage_id)
            if (attempt_status in {"terminal", "unresolved"}
                    or attempt_key in self._actor_profile_attempt_phases):
                raise ControlRefused("profile attempt identity is already used or unresolved")
            capability = object()
            reservation_id = actor_state_module.digest({
                "catalog_id": catalog_id, "transition_id": transition_id,
                "profile_request_digest": profile_request_digest,
                "stage_plan_digest": stage_plan_digest,
                "supervisor_incarnation": self.supervisor_incarnation,
                "control_revision": self.control_revision})
            reservation = TargetProfileExecutionReservation(
                reservation_id, catalog_id, transition_id, profile_request_digest,
                stage_plan_digest, request_id, stage_id, self.supervisor_incarnation,
                self.control_revision, max_output_bytes, capability)
            self._actor_profile_execution_reservations[reservation_id] = reservation
            self._actor_profile_attempt_phases[attempt_key] = "prelaunch"
            return reservation

    def record_verified_target_profile(
            self, value: Mapping[str, Any], *, reservation, terminal,
            provider_cost_receipt):
        """Join selected work, exact terminal/cost and bounded owner-read output."""
        from .target_profile_execution import (
            PROFILE_OUTPUT_SCHEMA, TargetProfileExecutionReservation)
        from .actor_lifecycle import TargetProfileReceipt
        row = actor_state_module.validate_event(value)
        if row["event"] != "PROFILE_VERIFIED":
            raise ControlRefused("target-profile publication requires PROFILE_VERIFIED")
        if not isinstance(reservation, TargetProfileExecutionReservation):
            raise ControlRefused("target-profile reservation is untyped")
        with self._mutex:
            self._require_active_locked()
            if self._actor_profile_execution_reservations.get(
                    reservation.reservation_id) is not reservation:
                raise ControlRefused("target-profile reservation is not current owner authority")
            self._selected_profile_work_locked(
                catalog_id=row["catalog_id"], transition_id=row["transition_id"],
                request_digest=row["profile_request_digest"],
                stage_plan_digest=row["stage_plan_digest"])
            trusted = self._actor_trusted_held_receipts.get(terminal.request_id)
            if (trusted != (provider_cost_receipt, terminal)
                    or not isinstance(provider_cost_receipt, scheduling.HeldClaimReceipt)
                    or reservation.request_id != terminal.request_id
                    or reservation.stage_id != terminal.stage_id
                    or reservation.stage_plan_digest != terminal.plan_digest
                    or reservation.transition_id != terminal.lineage_id
                    or terminal.return_code != 0 or not terminal.accepted
                    or terminal.result_digest is None
                    or row["verifier_ref"] !=
                       f"controller-worker:{terminal.worker_id}:{terminal.worker_generation}"
                    or row["campaign_id"] != self.resolved.campaign_id
                    or row["config_digest"] != self.config_digest
                    or row["config_generation"] != self.config_generation
                    or row["supervisor_id"] != self._supervisor_id
                    or row["supervisor_incarnation"] != self.supervisor_incarnation
                    or row["control_revision"] != self.control_revision):
                raise ControlRefused("profile result differs from selected current owner work")
            owner_binding = (self.supervisor_incarnation, self.control_revision,
                             self.config_digest, self.desired_state)
        raw = self.read_worker_stdout(
            request_id=terminal.request_id, plan_digest=terminal.plan_digest,
            lineage_id=terminal.lineage_id, stage_id=terminal.stage_id,
            worker_id=terminal.worker_id, worker_generation=terminal.worker_generation,
            result_digest=terminal.result_digest,
            max_bytes=reservation.max_output_bytes)
        try:
            output = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlRefused("trusted profile stdout is not UTF-8 JSON") from exc
        expected_output = {
            "schema": PROFILE_OUTPUT_SCHEMA, "profile_content": row["profile_content"],
            "loaded_identity": row["loaded_identity"],
            "artifact_identity": row["artifact_identity"],
            "measurement_carrier": row["measurement_carrier"],
        }
        if output != expected_output:
            raise ControlRefused("profile event differs from controller-read worker stdout")
        with self._mutex:
            self._require_active_locked()
            if (owner_binding != (self.supervisor_incarnation, self.control_revision,
                                  self.config_digest, self.desired_state)
                    or self._actor_profile_execution_reservations.get(
                        reservation.reservation_id) is not reservation
                    or self._actor_trusted_held_receipts.get(terminal.request_id)
                       != (provider_cost_receipt, terminal)):
                raise ControlRefused("profile owner changed during bounded output read")
            self._selected_profile_work_locked(
                catalog_id=row["catalog_id"], transition_id=row["transition_id"],
                request_digest=row["profile_request_digest"],
                stage_plan_digest=row["stage_plan_digest"])
            receipt = TargetProfileReceipt(
                campaign_digest=row["config_digest"], profile_request=row["profile_request"],
                profile_request_digest=row["profile_request_digest"],
                target_revision_digest=row["target_revision_digest"],
                target_profile_digest=row["target_profile_digest"],
                verified_at=row["verified_at"], valid_until=row["valid_until"],
                clock_domain=row["clock_domain"], verifier_ref=row["verifier_ref"])
            prior = self._actor_preparation_state.profiles.get(
                row["target_revision_digest"])
            if prior is not None:
                if prior != row:
                    raise ControlRefused(
                        "target profile replacement requires a new owner epoch")
                return receipt
            self._append_actor_event_locked(row)
            self._current_actor_profile_receipts[receipt.digest] = receipt
            del self._actor_profile_execution_reservations[reservation.reservation_id]
            self._actor_profile_execution_settlements[reservation.reservation_id] = (
                provider_cost_receipt, terminal)
            return receipt

    def cancel_target_profile_execution(self, reservation) -> None:
        """Cancel only a proved pre-launch profile admission."""
        from .target_profile_execution import TargetProfileExecutionReservation
        if not isinstance(reservation, TargetProfileExecutionReservation):
            raise ControlRefused("target-profile reservation is untyped")
        with self._mutex:
            self._require_active_locked()
            if self._actor_profile_execution_reservations.get(
                    reservation.reservation_id) is not reservation:
                raise ControlRefused("target-profile reservation is not current owner authority")
            status = self.worker_attempt_status(
                request_id=reservation.request_id, plan_digest=reservation.stage_plan_digest,
                lineage_id=reservation.transition_id, stage_id=reservation.stage_id)
            attempt_key = self._worker_attempt_key_fields(
                self._worker_lifecycle.binding, reservation.request_id,
                reservation.stage_plan_digest, reservation.transition_id, reservation.stage_id)
            phase = self._actor_profile_attempt_phases.get(attempt_key)
            # A freshly issued phase is owner-maintained proof. Absence of held
            # cost, terminal, or provider records is never such proof by itself.
            if phase != "prelaunch" or status in {"terminal", "unresolved"}:
                raise ControlRefused(
                    f"target-profile cancellation lacks negative-admission proof: {phase}/{status}")
            self._actor_profile_attempt_phases[attempt_key] = "cancelled"
            self._actor_profile_cancelled_attempts.add(attempt_key)
            del self._actor_profile_execution_reservations[reservation.reservation_id]

    def finish_target_profile_execution(
            self, *, reservation, terminal, provider_cost_receipt):
        """Close a completed profile admission with its exact provider-held cost."""
        from .target_profile_execution import TargetProfileExecutionReservation
        if not isinstance(reservation, TargetProfileExecutionReservation):
            raise ControlRefused("target-profile reservation is untyped")
        with self._mutex:
            self._require_active_locked()
            if self._actor_profile_execution_reservations.get(
                    reservation.reservation_id) is not reservation:
                raise ControlRefused("target-profile reservation is not current owner authority")
            if (self._actor_trusted_held_receipts.get(terminal.request_id)
                    != (provider_cost_receipt, terminal)
                    or not isinstance(provider_cost_receipt, scheduling.HeldClaimReceipt)):
                raise ControlRefused("target-profile cost is not provider-authored")
            del self._actor_profile_execution_reservations[reservation.reservation_id]
            self._actor_profile_execution_settlements[reservation.reservation_id] = (
                provider_cost_receipt, terminal)
            return provider_cost_receipt

    def verified_target_profile(self, **request):
        """Ask the producer outside the mutex, then recheck exact controller binding."""
        with self._mutex:
            self._require_active_locked()
            producer = self._actor_profile_producer
            binding = (self.supervisor_incarnation, self.control_revision,
                       self.config_digest, self.desired_state)
            if producer is None:
                return None
        receipt = producer.verified_target_profile(**request)
        if receipt is None:
            return None
        from .actor_lifecycle import TargetProfileReceipt
        if not isinstance(receipt, TargetProfileReceipt):
            raise ControlRefused("profile producer returned an untyped receipt")
        with self._mutex:
            self._require_active_locked()
            if binding != (self.supervisor_incarnation, self.control_revision,
                           self.config_digest, self.desired_state):
                raise ControlRefused("controller changed during profile verification")
            if (receipt.campaign_digest != self.config_digest
                    or receipt.target_revision_digest
                       != request.get("target_revision_digest")):
                raise ControlRefused("profile receipt differs from current owner request")
            self._current_actor_profile_receipts[receipt.digest] = receipt
            return receipt

    def _selected_actor_work_locked(self, request_digest: str,
                                    stage_plan_digest: str) -> tuple[dict, dict]:
        issued_rows = getattr(self, "_driver_issued", {})
        matches = []
        for issued in issued_rows.values():
            catalog = issued.get("catalog", {})
            selection = issued.get("selection", {})
            work_map = catalog.get("work_by_stage_digest", {})
            selected_stage = selection.get("proposal_digest")
            work = work_map.get(selected_stage) if isinstance(work_map, Mapping) else None
            if (isinstance(work, Mapping) and work.get("kind") == "actor_preparation"
                    and work.get("stage_plan_digest") == stage_plan_digest
                    and actor_state_module.digest(work.get("payload")) == request_digest
                    and selection.get("status") == "selected"):
                matches.append((copy.deepcopy(dict(issued)), copy.deepcopy(dict(work))))
        if len(matches) != 1:
            raise ControlRefused("actor request lacks one exact issued selected transition")
        return matches[0]

    def _append_actor_event_locked(self, row: Mapping[str, Any]):
        self._require_active_locked()
        validated = actor_state_module.validate_event(row)
        projected = actor_state_module.project_events(
            [*self._actor_preparation_events, validated])
        assert self._journal is not None
        try:
            entry = self._journal.append(
                journal_module.KIND_ACTOR_PREPARATION, validated,
                record_id=(validated["reservation_id"]
                           if validated["event"] != "PROFILE_VERIFIED"
                           else validated["target_profile_digest"]))
            self._verify_journal_layout(self.store / "journal")
        except BaseException:
            self._poisoned = True
            raise
        self._journal_cursor = entry.seq
        self._actor_preparation_events.append(copy.deepcopy(validated))
        self._actor_preparation_state = projected
        return copy.deepcopy(entry)

    def reserve_actor_preparation(
            self, *, request: Mapping[str, Any], request_digest: str,
            stage_plan_digest: str, actor_profile: Mapping[str, Any],
            actor_profile_digest: str, target_profile_receipt: Mapping[str, Any],
            target_profile_receipt_digest: str, budgets: Mapping[str, int],
            now: float, clock_domain: str) -> Mapping[str, Any]:
        """Revalidate selected work and durably reserve all budgets before launch."""
        from . import actor_preparation as actor_module
        from .actor_lifecycle import TargetProfileReceipt
        profile = actor_module.ActorProfile.from_dict(actor_profile)
        receipt = TargetProfileReceipt(**dict(target_profile_receipt))
        actor_state_module._budget_map(budgets, "budgets", integral=True)
        if (isinstance(now, bool) or not isinstance(now, (int, float))
                or not math.isfinite(now)):
            raise ControlRefused("actor reservation time must be finite")
        if (request_digest != actor_state_module.digest(request)
                or actor_profile_digest != profile.digest
                or target_profile_receipt_digest != receipt.digest
                or receipt.clock_domain != clock_domain
                or receipt.verified_at > now or receipt.valid_until <= now):
            raise ControlRefused("actor reservation inputs lack current producer authority")
        with self._mutex:
            self._require_active_locked()
            if self._current_actor_profile_receipts.get(receipt.digest) != receipt:
                raise ControlRefused(
                    "actor reservation lacks current producer receipt authority")
            if self.desired_state != "running" or self._worker_run_active:
                raise ControlRefused("actor admission is closed or worker is active")
            issued, work = self._selected_actor_work_locked(
                request_digest, stage_plan_digest)
            target = work["payload"]["proposal"]["target_revision_digest"]
            if target != receipt.target_revision_digest:
                raise ControlRefused("selected actor target differs from verified profile")
            prior_budgets = [row["budgets"] for row in self._actor_preparation_events
                             if row["event"] == "INTENT"]
            if any(dict(previous) != dict(budgets) for previous in prior_budgets):
                raise ControlRefused(
                    "actor budgets differ from the durable campaign configuration")
            debits = {key: 0.0 for key in actor_state_module.BUDGET_KEYS}
            debits["actor_calls_per_target"] = 1.0
            debits["actor_calls_per_campaign"] = 1.0
            target_events = [row for row in self._actor_preparation_events
                             if row["target_revision_digest"] == target]
            target_projection = actor_state_module.project_events(target_events)
            for key in actor_state_module.BUDGET_KEYS:
                used = target_projection.spent[key] + target_projection.reserved[key]
                if key == "actor_calls_per_campaign":
                    used = (self._actor_preparation_state.spent[key]
                            + self._actor_preparation_state.reserved[key])
                if used + debits[key] > float(budgets[key]):
                    raise ControlRefused(f"actor budget exhausted: {key}")
            reservation_id = "actor-" + actor_state_module.digest({
                "transition_id": issued["transition_id"], "request": request_digest,
                "actor": actor_profile_digest, "profile": receipt.digest,
            })[:24]
            if reservation_id in self._actor_preparation_state.pending:
                raise ControlRefused("actor reservation is unresolved; reinvocation forbidden")
            if reservation_id in self._actor_preparation_state.finished:
                raise ControlRefused("actor reservation already settled; reinvocation forbidden")
            deadline = min(receipt.valid_until,
                           float(now) + float(budgets["provider_seconds_per_target"]))
            if deadline <= now:
                raise ControlRefused("actor reservation deadline is not future")
            row = {
                "schema": actor_state_module.INTENT_SCHEMA, "event": "INTENT",
                "reservation_id": reservation_id,
                "campaign_id": self.resolved.campaign_id,
                "config_generation": self.config_generation,
                "config_digest": self.config_digest,
                "supervisor_id": self._supervisor_id,
                "supervisor_incarnation": self.supervisor_incarnation,
                "control_revision": self.control_revision,
                "catalog_id": issued["catalog_id"],
                "transition_id": issued["transition_id"],
                "request_digest": request_digest,
                "stage_plan_digest": stage_plan_digest,
                "target_revision_digest": target,
                "target_profile_digest": receipt.target_profile_digest,
                "target_profile_receipt_digest": receipt.digest,
                "actor_profile_digest": actor_profile_digest,
                "backend_key": profile.backend().describe(),
                "clock_domain": clock_domain, "deadline": deadline,
                "occurred_at": self.clock(), "budgets": dict(budgets), "debits": debits,
            }
            self._append_actor_event_locked(row)
            return {
                "schema": actor_module.RESERVATION_SCHEMA,
                "reservation_id": reservation_id, "request_digest": request_digest,
                "stage_plan_digest": stage_plan_digest,
                "transition_id": issued["transition_id"],
                "target_profile_digest": receipt.target_profile_digest,
                "target_profile_receipt_digest": receipt.digest,
                "actor_profile_digest": actor_profile_digest, "deadline": deadline,
                "clock_domain": clock_domain, "control_revision": self.control_revision,
            }

    def finish_actor_preparation(
            self, *, reservation: Mapping[str, Any], outcome: Mapping[str, Any],
            disposition: str, provider_cost_receipt=None) -> None:
        """Settle one exact reservation; identical retry returns without another append."""
        from . import actor_preparation as actor_module
        reserved = actor_module.StageReservation.from_dict(reservation)
        result = actor_module.StageOutcome.from_dict(outcome)
        with self._mutex:
            self._require_active_locked()
            prior = self._actor_preparation_state.finished.get(reserved.reservation_id)
            if prior is not None:
                if (prior["outcome_digest"] == actor_state_module.digest(result.to_dict())
                        and prior["disposition"] == disposition):
                    return
                raise ControlRefused("actor finish retry differs from durable settlement")
            intent = self._actor_preparation_state.pending.get(reserved.reservation_id)
            if intent is None:
                raise ControlRefused("actor finish lacks unresolved durable intent")
            if any(reserved.to_dict()[key] != intent[key] for key in (
                    "reservation_id", "request_digest", "stage_plan_digest", "transition_id",
                    "target_profile_digest", "target_profile_receipt_digest",
                    "actor_profile_digest", "deadline", "clock_domain", "control_revision")):
                raise ControlRefused("actor finish reservation differs from durable intent")
            if result.charged_seconds:
                held = provider_cost_receipt
                issued_cost = self._actor_trusted_held_receipts.get(
                    reserved.reservation_id)
                if (not isinstance(held, scheduling.HeldClaimReceipt)
                        or issued_cost is None or issued_cost[0] is not held
                        or issued_cost[1].request_id != reserved.reservation_id
                        or issued_cost[1].plan_digest != reserved.stage_plan_digest
                        or held.ownership_generation
                           != issued_cost[1].worker_generation
                        or held.allocation_generation
                           != issued_cost[1].grant_generation
                        or held.ended_at - held.started_at
                           != result.charged_seconds):
                    raise ControlRefused(
                        "actor finish lacks exact provider-authored held cost")
            elif provider_cost_receipt is not None:
                raise ControlRefused("zero-cost actor finish carries a provider receipt")
            charges = {key: 0.0 for key in actor_state_module.BUDGET_KEYS}
            charges["actor_calls_per_target"] = 1.0
            charges["actor_calls_per_campaign"] = 1.0
            charges["provider_seconds_per_target"] = result.charged_seconds
            failed = result.status != "completed"
            charges["resource_failures_per_target"] = float(
                failed and not result.resource_enforced)
            previous = self._actor_preparation_state.availability.get(intent["backend_key"], {})
            streak = 0 if not failed else int(previous.get("consecutive_failures", 0)) + 1
            retry = None if not failed else intent["deadline"]
            row = {key: value for key, value in intent.items()
                   if key not in {"budgets", "debits"}}
            row.update({
                "schema": actor_state_module.FINISH_SCHEMA, "event": "FINISH",
                "occurred_at": self.clock(),
                "outcome_digest": actor_state_module.digest(result.to_dict()),
                "status": result.status, "failure_class": result.failure_class,
                "charged_seconds": result.charged_seconds,
                "resource_enforced": result.resource_enforced,
                "descendants_clean": result.descendants_clean,
                "disposition": disposition, "charges": charges,
                "consecutive_failures": streak,
                "last_success": None if failed else intent["deadline"],
                "retry_after": retry, "reset_at": None,
                "next_eligible_at": retry,
            })
            self._append_actor_event_locked(row)

    def current_actor_profile(self, target_revision_digest: str):
        """Return only the current producer-verified receipt for one exact target."""
        with self._mutex:
            self._require_active_locked()
            matches = [receipt for receipt in self._current_actor_profile_receipts.values()
                       if receipt.target_revision_digest == target_revision_digest]
            if len(matches) != 1:
                return None
            return matches[0]

    def register_unified_settlement_validator(
            self, validator: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
        """Install the trusted worker-owner terminal/held-receipt verifier."""
        if not callable(validator):
            raise TypeError("unified settlement validator must be callable")
        with self._mutex:
            self._require_active_locked()
            if self._driver_settlement_validator is not None:
                raise ControlRefused("unified settlement validator is already registered")
            self._driver_settlement_validator = validator

    def unified_driver_settle(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Verify outside the mutex, then journal and apply exact held accounting."""
        with self._mutex:
            self._require_active_locked()
            if not isinstance(value, Mapping):
                raise ControlRefused("unified settlement request must be a mapping")
            supplied = copy.deepcopy(dict(value))
            transition_id = supplied.get("transition_id")
            if isinstance(transition_id, str):
                prior_settlement = self._driver_settled.get(transition_id)
                if prior_settlement is not None:
                    semantic_names = (
                        "catalog_id", "transition_id", "selection", "receipt", "outcome",
                        "terminal_refs")
                    expected_fields = {"schema", *semantic_names}
                    if (set(supplied) != expected_fields
                            or supplied.get("schema") != DRIVER_SETTLEMENT_SCHEMA):
                        raise ControlRefused(
                            "settlement retry conflicts with durable content")
                    semantic = {name: supplied[name] for name in semantic_names}
                    durable = {name: prior_settlement[name] for name in semantic_names}
                    if semantic != durable:
                        raise ControlRefused(
                            "settlement retry conflicts with durable content")
                    return {
                        "schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                        "transition_id": transition_id, "status": "duplicate",
                        "accounting_projection_digest":
                            prior_settlement["after_projection_digest"],
                    }
            validator = self._driver_settlement_validator
            if validator is None:
                raise ControlRefused("trusted unified settlement validator is unavailable")
            lifetime_token = self._lifetime_token
            incarnation = self.supervisor_incarnation
        try:
            verified = validator(supplied)
        except Exception as exc:
            raise ControlRefused(f"trusted unified settlement verification failed: {exc}") from exc
        if not isinstance(verified, Mapping):
            raise ControlRefused("trusted unified settlement verifier returned malformed data")
        row = copy.deepcopy(dict(verified))
        expected = {"schema", "catalog_id", "transition_id", "selection", "receipt",
                    "outcome", "terminal_refs"}
        if set(row) != expected or row.get("schema") != DRIVER_SETTLEMENT_SCHEMA:
            raise ControlRefused("unified settlement request fields/schema differ")
        if (not isinstance(row["terminal_refs"], list) or not row["terminal_refs"]
                or any(not isinstance(item, str) or not item for item in row["terminal_refs"])
                or len(row["terminal_refs"]) != len(set(row["terminal_refs"]))):
            raise ControlRefused("unified settlement terminal refs are invalid")
        try:
            selection = scheduling.Selection.from_dict(row["selection"])
            receipt = scheduling.HeldClaimReceipt.from_dict(row["receipt"])
        except Exception as exc:
            raise ControlRefused(
                f"trusted unified settlement verifier returned invalid bindings: {exc}") from exc
        if row["outcome"] not in scheduling.OUTCOMES:
            raise ControlRefused("unified settlement outcome is unsupported")
        with self._mutex:
            self._require_active_locked()
            if (lifetime_token is None or lifetime_token is not self._lifetime_token
                    or incarnation != self.supervisor_incarnation
                    or validator is not self._driver_settlement_validator):
                raise ControlRefused(
                    "settlement verification belongs to a stale controller lifetime")
            prior_settlement = self._driver_settled.get(row["transition_id"])
            if prior_settlement is not None:
                semantic = {name: row[name] for name in (
                    "catalog_id", "transition_id", "selection", "receipt", "outcome",
                    "terminal_refs")}
                durable = {name: prior_settlement[name] for name in semantic}
                if semantic != durable:
                    raise ControlRefused("settlement retry conflicts with durable content")
                return {"schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                        "transition_id": row["transition_id"], "status": "duplicate",
                        "accounting_projection_digest":
                            prior_settlement["after_projection_digest"]}
            issued = self._driver_issued.get(row["catalog_id"])
            if (issued is None or issued["transition_id"] != row["transition_id"]
                    or issued["selection"] != selection.to_dict()):
                raise ControlRefused("settlement does not bind an issued transition")
            assert self._scheduler_engine is not None
            preview = self._scheduler_engine.preview_accounting(
                selection, receipt, outcome=row["outcome"])
            event = {
                "schema": journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA,
                "catalog_id": row["catalog_id"], "transition_id": row["transition_id"],
                "campaign_id": self.resolved.campaign_id,
                "config_generation": self.config_generation,
                "config_digest": self.config_digest,
                "supervisor_incarnation": self.supervisor_incarnation,
                "selection": selection.to_dict(), "receipt": receipt.to_dict(),
                "outcome": row["outcome"], "terminal_refs": list(row["terminal_refs"]),
                "prior_projection_digest": preview.prior.projection_digest,
                "after_projection_digest": preview.after.projection_digest,
            }
            self._verify_store()
            assert self._journal is not None
            try:
                entry = self._journal.append(
                    journal_module.KIND_UNIFIED_DRIVER_SETTLED, event,
                    record_id=row["transition_id"])
                self._verify_journal_layout(self.store / "journal")
                self._scheduler_engine.apply_accounting_preview(preview)
            except BaseException:
                self._poisoned = True
                raise
            self._journal_cursor = entry.seq
            self._driver_settled[row["transition_id"]] = copy.deepcopy(event)
            return {"schema": DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
                    "transition_id": row["transition_id"], "status": "accepted",
                    "accounting_projection_digest": preview.after.projection_digest}

    def apply_command(self, value: Mapping[str, Any]) -> dict[str, Any]:
        with self._mutex:
            self._require_active_locked()
            if self.snapshot_version in {2, 3}:
                return self._apply_command_v2_locked(value)
            row = validate_command(value)
            if row["campaign_id"] != self.resolved.campaign_id \
                    or row["config_generation"] != self.config_generation:
                raise ControlRefused("command campaign/config generation does not match")
            prior = self._command_results.get(row["request_id"])
            if prior is not None:
                if prior["payload_digest"] != row["payload_digest"]:
                    raise ControlRefused("request_id was already used for different semantics")
                self._publish_snapshot_locked()
                return copy.deepcopy(prior)
            if row["expected_control_revision"] != self.control_revision:
                raise ControlRefused(
                    f"stale control revision; current={self.control_revision}")
            if self.desired_state == "drained":
                raise ControlRefused("drained is terminal for this campaign generation")
            revision = self.control_revision + 1
            if row["operation"] == "resume":
                try:
                    readiness = self.readiness_check()
                except Exception as exc:
                    raise ControlRefused(f"readiness prerequisite check failed: {exc}") from exc
                if (not isinstance(readiness, tuple) or len(readiness) != 2
                        or not isinstance(readiness[0], bool)
                        or (readiness[1] is not None
                            and (not isinstance(readiness[1], str)
                                 or not readiness[1].strip()))):
                    raise ControlRefused("readiness prerequisite returned malformed result")
                ready, reason = readiness
                desired, observed = "running", ("running" if ready else "waiting_prerequisite")
                prerequisite = None if ready else (reason or "prerequisite unavailable")
                completed = bool(ready)
            elif row["operation"] == "pause":
                desired, observed, prerequisite, completed = "paused", "paused", None, True
            else:
                desired, observed, prerequisite, completed = "drained", "drained", None, True
            result = {"request_id": row["request_id"], "operation": row["operation"],
                      "payload_digest": row["payload_digest"], "accepted": True,
                      "completed": completed, "control_revision": revision,
                      "desired_state": desired, "observed_state": observed,
                      "prerequisite_reason": prerequisite}
            payload = {"request_id": row["request_id"], "command": row,
                       "desired_state": desired, "observed_state": observed,
                       "prerequisite_reason": prerequisite, "result": result}
            old_revision = self.control_revision
            self.control_revision = revision
            try:
                self._append_event("CONTROL_ACCEPTED", payload)
            except BaseException:
                self.control_revision = old_revision
                self._poisoned = True
                raise
            self.desired_state, self.observed_state = desired, observed
            self.prerequisite_reason = prerequisite
            self._command_results[row["request_id"]] = result
            self._publish_snapshot_locked()
            return copy.deepcopy(result)

    def _snapshot_locked(self) -> dict[str, Any]:
        self._require_active_locked()
        if self.snapshot_version == 3:
            base = self._snapshot_v2_locked()
            base["schema"] = SNAPSHOT_SCHEMA_V3
            base["producer_schema"] = SNAPSHOT_SCHEMA_V3
            base["unified"] = self._unified_projection_locked()
            return validate_snapshot_v3(base)
        if self.snapshot_version == 2:
            return self._snapshot_v2_locked()
        self.sequence += 1
        generated_at = self.clock()
        return {"schema": SNAPSHOT_SCHEMA,
                    "producer_build": copy.deepcopy(self._producer_build),
                    "producer_schema": SNAPSHOT_SCHEMA,
                    "campaign_id": self.resolved.campaign_id,
                    "config_generation": self.config_generation,
                    "config_digest": self.config_digest,
                    "requested_manifest_digest": self.requested_manifest_digest,
                    "supervisor_incarnation": self.supervisor_incarnation,
                    "stream_epoch": self.stream_epoch, "sequence": self.sequence,
                    "journal_cursor": self._journal_cursor,
                    "control_revision": self.control_revision,
                    "generated_at": generated_at, "desired_state": self.desired_state,
                    "observed_state": self.observed_state,
                    "command_results": copy.deepcopy(list(self._command_results.values())),
                    "active_worker": None, "producer_heartbeat_at": generated_at,
                    "last_scientific_result_at": None, "worker_activity_at": None,
                    "execution_authorized": False,
                    "prerequisite_reason": self.prerequisite_reason}

    def _snapshot_v2_locked(self) -> dict[str, Any]:
        self.sequence += 1
        generated_at = self.clock()
        observed_state = self.observed_state
        prerequisite_reason = self.prerequisite_reason
        if self._acquisition_projection.pending is not None:
            observed_state = "ownership_unresolved"
            prerequisite_reason = (
                "worker_acquisition_pending:"
                + str(self._acquisition_projection.pending["request_id"]))
        active_worker = None
        worker_activity_at = None
        if self._worker_projection.active:
            worker_id, latest = next(iter(self._worker_projection.active.items()))
            rows = self._active_worker_events
            intent = next(row for row in rows if row["event"] == "OWNED_LAUNCH_INTENT")
            stage_rows = [row for row in rows if row["event"] == "WORKER_STAGE"]
            teardown_rows = [row for row in rows
                             if row["event"] == "OWNED_TEARDOWN_STARTED"]
            unresolved_rows = [row for row in rows if row["event"] in {
                "WORKER_UNRESOLVED", "OWNED_TEARDOWN_FAILED"}]
            worker_activity_at = (stage_rows[-1]["data"]["activity_at"]
                                  if stage_rows else None)
            state = {
                "OWNED_LAUNCH_INTENT": "intent",
                "OWNED_CONTAINER_CREATED": "container_created",
                "OWNED_CHILD_CAPTURED": "child_captured",
                "OWNED_EXEC_RELEASE_INTENT": "exec_release_intent",
                "OWNED_EXEC_RELEASED": "executing",
                "WORKER_STAGE": "executing",
                "WORKER_RESULT_RETAINED": "result_retained",
                "OWNED_TEARDOWN_STARTED": "tearing_down",
                "OWNED_TEARDOWN_FAILED": "teardown_failed",
                "WORKER_UNRESOLVED": "unresolved",
                "WORKER_RESULT_STALE": "unresolved",
            }[latest["event"]]
            active_worker = {
                "worker_id": worker_id,
                "worker_generation": intent["worker_generation"],
                "request_id": intent["request_id"],
                "plan_digest": intent["plan_digest"],
                "lineage_id": intent["lineage_id"],
                "stage_id": intent["stage_id"], "state": state,
                "grant_id": intent["grant_id"],
                "grant_generation": intent["grant_generation"],
                "container_id": intent["container_id"],
                "provider_deadline": intent["data"]["provider_deadline"],
                "deadline_clock_domain": intent["data"]["clock_domain"],
                "control_revision": intent["control_revision"],
                "started_at": intent["occurred_at"],
                "activity_at": worker_activity_at,
                "termination_deadline": (teardown_rows[-1]["data"]["termination_deadline"]
                                         if teardown_rows else None),
                "unresolved_reason": (unresolved_rows[-1]["data"]["reason"]
                                      if unresolved_rows else None),
            }
        return validate_snapshot_v2({
            "schema": SNAPSHOT_SCHEMA_V2,
            "producer_build": copy.deepcopy(self._producer_build),
            "producer_schema": SNAPSHOT_SCHEMA_V2,
            "campaign_id": self.resolved.campaign_id,
            "config_generation": self.config_generation,
            "config_digest": self.config_digest,
            "requested_manifest_digest": self.requested_manifest_digest,
            "supervisor_incarnation": self.supervisor_incarnation,
            "stream_epoch": self.stream_epoch, "sequence": self.sequence,
            "journal_cursor": self._journal_cursor,
            "control_revision": self.control_revision,
            "generated_at": generated_at, "desired_state": self.desired_state,
            "observed_state": observed_state,
            "command_results": copy.deepcopy(list(self._command_results.values())),
            "active_worker": active_worker,
            "producer_heartbeat_at": generated_at,
            "last_scientific_result_at": None,
            "worker_activity_at": worker_activity_at,
            # A snapshot does not call the provider under the controller lock;
            # therefore it makes no live grant claim from cached history.
            "execution_authorized": False,
            "execution_capability_available": self._lifecycle_provider is not None,
            "worker_lifecycle_revision": self._worker_lifecycle_revision,
            "prerequisite_reason": prerequisite_reason,
        })

    def _unified_projection_locked(self) -> dict[str, Any]:
        assert self._scheduler_engine is not None
        projection = self._scheduler_engine.operational_projection()
        body = projection.body
        accounting = self._scheduler_engine.accounting_view().to_dict()
        targets = tuple(self.resolved.targets)
        ready = sum(item.status == "ready" for item in targets)
        prerequisite = len(targets) - ready
        seed_enrolled = sum("seed" in item.enrolled_as for item in targets)
        production_enrolled = sum("production" in item.enrolled_as for item in targets)
        return {
            "schema": UNIFIED_PROJECTION_SCHEMA,
            "scheduler": {
                "schema": "epyc.autokernel.unified_scheduler_projection.v1",
                "projection_digest": projection.projection_digest,
                "config_digest": body["config_digest"], "policy_digest": body["policy_digest"],
                "round_number": body["round_number"],
                "accounting_epoch": body["accounting_epoch"],
                "capacity": self._scheduler_engine.capacity.to_dict(),
                "pending_selection_digest": (body["issued_selection_digests"][-1]
                                             if body["issued_selection_digests"] else None),
                "status": "available", "reason": "controller-owned scheduler projection",
                "campaign_attempts": body["campaign_attempts"],
                "campaign_charged_seconds": body["campaign_charged_seconds"],
                "accounting": accounting,
                "coverage_debt_count": len(body["coverage_debt"]),
            },
            "resources": {
                "schema": "epyc.autokernel.unified_resource_status.v1",
                "status": "not_connected", "reason": "resource telemetry not connected",
                "requested": None, "granted": None, "held": None, "used": None,
            },
            "actors": {
                "schema": "epyc.autokernel.unified_actor_status.v1",
                "status": "not_connected", "reason": "actor result cache not connected",
                "clock_semantics": "UTC wall-clock projection; runtime fences remain monotonic",
                "items": [],
            },
            "evidence": {
                "schema": "epyc.autokernel.unified_evidence_status.v1",
                "status": "not_connected", "reason": "evidence frontier not connected",
                "frontier_digest": None, "lag_seconds": None,
            },
            "candidate": {
                "schema": "epyc.autokernel.unified_candidate_status.v1",
                "status": "not_connected", "reason": "candidate projection not connected",
                "accumulated_identity": None, "validated_identity": None,
                "frozen_production_identity": None, "validation_debt": None,
            },
            "targets": {
                "schema": "epyc.autokernel.unified_target_status.v1",
                "status": "available", "reason": "resolved campaign summary",
                "total": len(targets), "ready": ready, "prerequisite": prerequisite,
                "production_enrolled": production_enrolled, "seed_enrolled": seed_enrolled,
                "items_page_ref": None,
            },
        }

    def snapshot(self) -> dict[str, Any]:
        with self._mutex:
            return self._snapshot_locked()

    def _publish_snapshot_locked(self) -> dict[str, Any]:
        body = self._snapshot_locked()
        status.write_json(self.store, SNAPSHOT_FILE, body, prefix=".campaign-")
        self._verify_store()
        return copy.deepcopy(body)

    def publish_snapshot(self) -> dict[str, Any]:
        with self._mutex:
            return self._publish_snapshot_locked()

    def may_start_stage(self, **kwargs) -> AdmissionDecision:
        with self._mutex:
            self._require_active_locked()
            return may_start_stage(
                desired_state=self.desired_state,
                current_control_revision=self.control_revision,
                current_supervisor_incarnation=self.supervisor_incarnation,
                **kwargs)

    def close(self) -> None:
        with self._mutex:
            if self._entered and self._maintenance_state.owned:
                raise ControlRefused(
                    "maintenance ownership is active/unresolved; "
                    "controller ownership retained")
            if self.snapshot_version in {2, 3} and self._entered \
                    and (self._worker_run_active or self._worker_projection.active
                         or self._acquisition_projection.pending is not None):
                raise ControlRefused(
                    "owned worker is active/unresolved; controller ownership retained")
            fd, self._lease_fd = self._lease_fd, None
            runtime, self._runtime_root = self._runtime_root, None
            self._lock_identity = None
            self._journal_root_identity = None
            self._entered = False
            self._journal = None
            self._candidate_pending = None
            self._candidate_pending_payload = None
            self._candidate_prepared = False
            self._candidate_completed = set()
            self._candidate_completed_records = {}
            self._candidate_entries = []
            self._candidate_projection_cache = None
            self._candidate_projection_position = 0
            self._candidate_context_token = None
            self._native_records = {}
            self._native_payload_digests = {}
            self._native_validator = None
            self._native_capabilities = set()
            self._retention_catalog_seed = None
            self._retention_catalog_event = None
            self._prepared_retention_jobs = {}
            self._command_requests = {}
            self._driver_issued = {}
            self._driver_settled = {}
            self._driver_settlement_validator = None
            artifact_store, self._driver_artifact_store = self._driver_artifact_store, None
            self._a2_execution_entries = {}
            self._a2_logical_executions = {}
            self._a2_bank_sources = {}
            self._actor_preparation_events = []
            self._actor_preparation_state = actor_state_module.ActorPreparationProjection()
            self._actor_profile_producer = None
            self._current_actor_profile_receipts = {}
            self._actor_trusted_held_receipts = {}
            self._actor_profile_execution_reservations = {}
            self._actor_profile_execution_settlements = {}
            self._actor_profile_cancelled_attempts = set()
            self._actor_profile_attempt_phases = {}
            self._supervisor_id = None
            self._worker_lifecycle = None
            self._active_worker_events = []
            self._active_acquisition_events = []
            self._acquisition_projection = worker_lifecycle_module.project_acquisitions([])
            self._worker_lifecycle_revision = 0
            self._worker_last_generation = 0
            self._worker_projection = worker_lifecycle_module.project_events([])
            self._lifetime_token = None
            if artifact_store is not None:
                artifact_store.close()
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            if runtime is not None:
                runtime.close()

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = ["ACTIVE_WORKER_V2_FIELDS", "AdmissionDecision", "CampaignController",
           "COMMAND_SCHEMA", "ControlRefused", "DriverAdmissionClosed",
           "SNAPSHOT_FILE", "SNAPSHOT_SCHEMA",
           "SNAPSHOT_SCHEMA_V2", "SNAPSHOT_V2_FIELDS", "TrustedGrant",
           "SNAPSHOT_SCHEMA_V3", "SNAPSHOT_V3_FIELDS", "UNIFIED_PROJECTION_SCHEMA",
           "DRIVER_SETTLEMENT_SCHEMA", "DRIVER_SETTLEMENT_RECEIPT_SCHEMA",
           "command_digest", "may_start_stage", "resolved_config_digest",
           "validate_command", "validate_snapshot_v2", "validate_snapshot_v3"]
