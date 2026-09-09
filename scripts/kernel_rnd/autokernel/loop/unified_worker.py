#!/usr/bin/env python3
"""Closed planned-serving child bridge for one already-owned lifecycle worker.

This module owns neither admission nor a Journal.  Production execution requires a live
inherited unit-authority channel and real cgroup membership; JSON fields never manufacture
either capability.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import secrets
import select
import socket
import stat
import sys
import threading
import time
from types import MappingProxyType
import types
from typing import Any, Protocol

if __package__ in {None, ""}:  # fixed absolute-script entry under isolated Python
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # This worker consumes already-parsed closed objects and never loads campaign YAML.
    # Keep isolated mode independent of user site-packages while allowing the shared
    # campaign type module to import its otherwise-unused optional parser dependency.
    if "yaml" not in sys.modules:
        isolated_yaml = types.ModuleType("yaml")
        class _UnavailableYamlError(Exception):
            pass
        def _yaml_unavailable(*_args: Any, **_kwargs: Any) -> Any:
            raise _UnavailableYamlError("planned worker cannot load YAML")
        isolated_yaml.YAMLError = _UnavailableYamlError
        isolated_yaml.safe_load = _yaml_unavailable
        sys.modules["yaml"] = isolated_yaml
    from autokernel.loop import experiment_plan as ep
    from autokernel.loop import measurement_capture as mc
    from autokernel.loop import native_capture_control as nc
    from autokernel.loop import planned_serving as ps
    from autokernel.loop import serving
    from autokernel.loop import unified_planner as up
    from autokernel.loop import worker_lifecycle as wl
else:
    from . import experiment_plan as ep
    from . import measurement_capture as mc
    from . import native_capture_control as nc
    from . import planned_serving as ps
    from . import serving
    from . import unified_planner as up
    from . import worker_lifecycle as wl


PREPARED_SCHEMA = "epyc.autokernel.prepared_planned_serving_stage.v1"
HELLO_SCHEMA = "epyc.autokernel.planned_worker_hello.v1"
INVOCATION_SCHEMA = "epyc.autokernel.planned_worker_invocation.v1"
START_SCHEMA = "epyc.autokernel.planned_worker_start.v1"
UNIT_REQUEST_SCHEMA = "epyc.autokernel.planned_worker_unit_request.v1"
UNIT_PERMIT_SCHEMA = "epyc.autokernel.planned_worker_unit_permit.v1"
UNIT_COMPLETION_REQUEST_SCHEMA = "epyc.autokernel.planned_worker_unit_completion_request.v1"
UNIT_COMPLETION_SCHEMA = "epyc.autokernel.planned_worker_unit_completion.v1"
CONTINUATION_REQUEST_SCHEMA = "epyc.autokernel.planned_worker_continuation_request.v1"
CONTINUATION_SCHEMA = "epyc.autokernel.planned_worker_continuation.v1"
RESULT_SCHEMA = "epyc.autokernel.planned_worker_result.v1"
RESULT_REFERENCE_SCHEMA = "epyc.autokernel.planned_worker_result_reference.v1"
MAX_MESSAGE_BYTES = 256 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024


class WorkerBridgeRefused(RuntimeError):
    """The closed request, live authority, placement, or result boundary was refused."""


class _ParentUnitEvidenceProtocol(Protocol):
    """Parent-owned evidence checks; lifecycle itself retains admission authority."""

    def complete(self, *, start: "WorkerStart", sequence: int,
                 fence: ps.StageFence,
                 observation: Mapping[str, Any]) -> ps.StageCompletion: ...

    def verify_continuation(self, *, start: "WorkerStart", raw: ep.RawUnit,
                            plan: ep.ExperimentPlan,
                            prompts: ps.FrozenPromptManifest,
                            previous_lineage_id: str) -> bool: ...


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


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or (positive and value <= 0)):
        raise WorkerBridgeRefused(f"{label} must be finite"
                                  + (" and positive" if positive else ""))
    return float(value)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WorkerBridgeRefused(f"{label} has missing/unknown fields")
    return dict(value)


@dataclass(frozen=True)
class PreparedPlannedServingStage:
    dispatch: Mapping[str, Any]
    plan: ep.ExperimentPlan
    prompts: ps.FrozenPromptManifest
    runtime_pair: up.RuntimeArmPair
    capture_context_base: Mapping[str, Any]
    artifact_root: Path
    previous: Mapping[str, Any] | None
    max_stage_seconds: float
    teardown_seconds: float
    schema: str = PREPARED_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "PreparedPlannedServingStage":
        row = _exact(value, {"schema", "dispatch", "plan", "prompt_manifest",
                             "runtime_pair", "capture_context_base", "artifact_root",
                             "previous", "max_stage_seconds", "teardown_seconds",
                             "prepared_digest"}, "prepared planned-serving stage")
        if row["schema"] != PREPARED_SCHEMA:
            raise WorkerBridgeRefused("prepared stage schema is unsupported")
        supplied_digest = _sha(row.pop("prepared_digest"), "prepared_digest")
        if supplied_digest != _digest(row):
            raise WorkerBridgeRefused("prepared stage digest mismatch")
        try:
            dispatch = up.DispatchRequest(**_plain(row["dispatch"]))
            plan = ep.ExperimentPlan.from_dict(_plain(row["plan"]))
            prompts = ps.FrozenPromptManifest.from_dict(_plain(row["prompt_manifest"]))
            pair = up.RuntimeArmPair.from_dict(_plain(row["runtime_pair"]))
        except Exception as exc:
            raise WorkerBridgeRefused(f"prepared typed input is invalid: {exc}") from exc
        proposal = up.UnifiedProposal.from_dict(_plain(dispatch.proposal))
        bindings = (
            dispatch.execution_authorized is False,
            dispatch.experiment_intent["experiment_plan_digest"] == plan.digest,
            proposal.experiment_plan_digest == plan.digest,
            _plain(proposal.runtime_pair) == pair.to_dict(),
        )
        if not all(bindings):
            raise WorkerBridgeRefused(
                f"dispatch, plan, and runtime pair bindings differ: {bindings}")
        if (dict(plan.anchor_identity) != ps.arm_identity(pair.anchor.template, pair.anchor)
                or dict(plan.candidate_identity)
                   != ps.arm_identity(pair.candidate.template, pair.candidate)):
            raise WorkerBridgeRefused("plan arm identities differ from resolved recipes")
        base = _exact(row["capture_context_base"], {
            "campaign_id", "config_digest", "supervisor_id", "supervisor_incarnation",
            "config_generation", "instrument_id", "protocol_id", "protocol_status",
            "source_identities"}, "capture context base")
        if (base["campaign_id"] != plan.campaign_id
                or base["protocol_id"] != plan.protocol_ref
                or base["protocol_status"] != plan.protocol_status):
            raise WorkerBridgeRefused("capture base differs from plan campaign/protocol")
        # Reuse the accepted validator by supplying inert, syntactically valid dynamic fields.
        try:
            mc.CaptureContext.from_dict({**base, "worker_id": "prepared-worker",
                "worker_incarnation": 1, "grant_id": "prepared-grant",
                "container_id": "prepared-container", "lineage_id": "prepared-lineage"})
        except Exception as exc:
            raise WorkerBridgeRefused(f"capture context base is invalid: {exc}") from exc
        root = Path(_text(row["artifact_root"], "artifact_root"))
        if not root.is_absolute():
            raise WorkerBridgeRefused("artifact_root must be absolute")
        previous = row["previous"]
        if previous is not None:
            previous = _exact(previous, {"raw_units", "previous_lineage_id",
                                         "continuation_proof_ref",
                                         "continuation_proof_digest"}, "continuation")
            if not plan.continuation_allowed:
                raise WorkerBridgeRefused("plan forbids continuation")
            if not isinstance(previous["raw_units"], list) or not previous["raw_units"]:
                raise WorkerBridgeRefused("continuation raw_units must be non-empty")
            try:
                raws = [ep.RawUnit.from_dict(item).to_dict()
                        for item in previous["raw_units"]]
            except Exception as exc:
                raise WorkerBridgeRefused(f"continuation raws are invalid: {exc}") from exc
            expected = [item.unit_id for item in sorted(
                plan.expected_units, key=lambda item: item.order_index)]
            if [item["unit_id"] for item in raws] != expected[:len(raws)]:
                raise WorkerBridgeRefused("continuation is not the fixed completed prefix")
            previous = {**previous, "raw_units": raws,
                        "previous_lineage_id": _text(
                            previous["previous_lineage_id"], "previous_lineage_id"),
                        "continuation_proof_ref": _text(
                            previous["continuation_proof_ref"], "continuation_proof_ref"),
                        "continuation_proof_digest": _sha(
                            previous["continuation_proof_digest"],
                            "continuation_proof_digest")}
        return cls(_freeze(dispatch.to_dict()), plan, prompts, pair, _freeze(base), root,
                   None if previous is None else _freeze(previous),
                   _finite(row["max_stage_seconds"], "max_stage_seconds", positive=True),
                   _finite(row["teardown_seconds"], "teardown_seconds", positive=True))

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "dispatch": _plain(self.dispatch),
                "plan": self.plan.to_dict(), "prompt_manifest": self.prompts.to_dict(),
                "runtime_pair": self.runtime_pair.to_dict(),
                "capture_context_base": _plain(self.capture_context_base),
                "artifact_root": str(self.artifact_root),
                "previous": None if self.previous is None else _plain(self.previous),
                "max_stage_seconds": self.max_stage_seconds,
                "teardown_seconds": self.teardown_seconds}

    @property
    def prepared_digest(self) -> str:
        return _digest(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "prepared_digest": self.prepared_digest}


@dataclass(frozen=True)
class WorkerStart:
    nonce: str
    request_id: str
    prepared_digest: str
    plan_digest: str
    lineage_id: str
    stage_id: str
    campaign_id: str
    config_digest: str
    config_generation: int
    supervisor_id: str
    supervisor_incarnation: int
    worker_id: str
    worker_generation: int
    grant_id: str
    grant_generation: int
    container_id: str
    child_process: wl.ProcessIdentity
    cgroup_identity: Mapping[str, Any]
    clock_domain: str
    provider_deadline: float
    sequence: int = 0
    schema: str = START_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "WorkerStart":
        row = _exact(value, {"schema", "nonce", "request_id", "prepared_digest", "plan_digest",
            "lineage_id", "stage_id", "campaign_id", "config_digest",
            "config_generation", "supervisor_id", "supervisor_incarnation", "worker_id",
            "worker_generation", "grant_id", "grant_generation", "container_id",
            "child_process", "cgroup_identity", "clock_domain", "provider_deadline",
            "sequence", "start_digest"}, "worker start")
        if row["schema"] != START_SCHEMA or row["sequence"] != 0:
            raise WorkerBridgeRefused("worker start schema/sequence is invalid")
        supplied = _sha(row.pop("start_digest"), "start_digest")
        if supplied != _digest(row):
            raise WorkerBridgeRefused("worker start digest mismatch")
        process = _exact(row["child_process"], {"pid", "start_ticks", "boot_id"},
                         "child process identity")
        child = wl.ProcessIdentity(_positive(process["pid"], "child pid"),
                                   _positive(process["start_ticks"], "child start_ticks"),
                                   _text(process["boot_id"], "child boot_id"))
        cgroup = _exact(row["cgroup_identity"], {"path", "dev", "ino", "uid", "nlink",
                                                    "mode"}, "cgroup identity")
        _text(cgroup["path"], "cgroup path")
        for name in ("dev", "ino", "uid", "nlink", "mode"):
            if not isinstance(cgroup[name], int) or isinstance(cgroup[name], bool) \
                    or cgroup[name] < 0:
                raise WorkerBridgeRefused(f"cgroup {name} is invalid")
        return cls(_text(row["nonce"], "nonce"), _text(row["request_id"], "request_id"),
            _sha(row["prepared_digest"], "prepared_digest"),
            _sha(row["plan_digest"], "plan_digest"), _text(row["lineage_id"], "lineage_id"),
            _text(row["stage_id"], "stage_id"), _text(row["campaign_id"], "campaign_id"),
            _sha(row["config_digest"], "config_digest"),
            _positive(row["config_generation"], "config_generation"),
            _text(row["supervisor_id"], "supervisor_id"),
            _positive(row["supervisor_incarnation"], "supervisor_incarnation"),
            _text(row["worker_id"], "worker_id"),
            _positive(row["worker_generation"], "worker_generation"),
            _text(row["grant_id"], "grant_id"),
            _positive(row["grant_generation"], "grant_generation"),
            _text(row["container_id"], "container_id"), child, _freeze(cgroup),
            _text(row["clock_domain"], "clock_domain"),
            _finite(row["provider_deadline"], "provider_deadline"))

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "nonce": self.nonce, "request_id": self.request_id,
            "prepared_digest": self.prepared_digest, "plan_digest": self.plan_digest,
            "lineage_id": self.lineage_id, "stage_id": self.stage_id,
            "campaign_id": self.campaign_id, "config_digest": self.config_digest,
            "config_generation": self.config_generation, "supervisor_id": self.supervisor_id,
            "supervisor_incarnation": self.supervisor_incarnation,
            "worker_id": self.worker_id, "worker_generation": self.worker_generation,
            "grant_id": self.grant_id, "grant_generation": self.grant_generation,
            "container_id": self.container_id, "child_process": self.child_process.to_dict(),
            "cgroup_identity": _plain(self.cgroup_identity), "clock_domain": self.clock_domain,
            "provider_deadline": self.provider_deadline, "sequence": self.sequence}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "start_digest": _digest(self.body())}


@dataclass(frozen=True)
class UnitPermit:
    sequence: int
    unit_id: str
    process_generation_id: str
    fence_id: str
    valid_until: float
    grant_id: str
    grant_generation: int
    container_id: str
    allowed: bool
    reason: str

    @classmethod
    def from_dict(cls, value: Any, *, start: WorkerStart,
                  expected_sequence: int) -> "UnitPermit":
        row = _exact(value, {"schema", "nonce", "sequence", "unit_id",
            "process_generation_id", "fence_id", "valid_until", "grant_id",
            "grant_generation", "container_id", "allowed", "reason", "permit_digest"},
            "unit permit")
        if row["schema"] != UNIT_PERMIT_SCHEMA or row["nonce"] != start.nonce \
                or row["sequence"] != expected_sequence:
            raise WorkerBridgeRefused("unit permit schema/nonce/sequence differs")
        supplied = _sha(row.pop("permit_digest"), "permit_digest")
        if supplied != _digest(row) or type(row["allowed"]) is not bool:
            raise WorkerBridgeRefused("unit permit digest/allowed field is invalid")
        return cls(row["sequence"], _text(row["unit_id"], "unit_id"),
                   _text(row["process_generation_id"], "process_generation_id"),
                   _text(row["fence_id"], "fence_id"),
                   _finite(row["valid_until"], "valid_until"),
                   _text(row["grant_id"], "grant_id"),
                   _positive(row["grant_generation"], "grant_generation"),
                   _text(row["container_id"], "container_id"), row["allowed"],
                   _text(row["reason"], "reason"))


class ParentUnitEvidenceAuthority:
    """Bounded parent cache fed by an external trusted observation producer.

    The lifecycle watchdog only submits a bounded notice and polls this cache. It never
    invokes producer, provider, Journal, filesystem, or network callbacks.
    """

    def __init__(self, *, completions: Mapping[str, ps.StageCompletion] | None = None,
                 continuations: Mapping[str, bool] | None = None,
                 max_records: int = 64) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool) \
                or not 1 <= max_records <= 1024:
            raise WorkerBridgeRefused("parent evidence cache bound is invalid")
        self._max_records = max_records
        self._lock = threading.Lock()
        self._notices: queue.Queue[Mapping[str, Any]] = queue.Queue(maxsize=max_records)
        self._requested: dict[str, str] = {}
        self._completions: dict[str, ps.StageCompletion] = {}
        for key, value in dict(completions or {}).items():
            _sha(key, "completion cache key")
            if not isinstance(value, ps.StageCompletion):
                raise WorkerBridgeRefused("completion cache requires typed completions")
            self._completions[key] = value
        self._continuations: dict[str, bool] = {}
        for key, value in dict(continuations or {}).items():
            _sha(key, "continuation cache key")
            if type(value) is not bool:
                raise WorkerBridgeRefused("continuation cache values must be boolean")
            self._continuations[key] = value
        if set(self._completions) & set(self._continuations):
            raise WorkerBridgeRefused("parent evidence cache key changes evidence kind")
        if len(set(self._completions) | set(self._continuations)) > max_records:
            raise WorkerBridgeRefused("parent evidence cache exceeds its record bound")

    def _reserve_request(self, key: str, kind: str,
                         notice: Mapping[str, Any]) -> None:
        prior_kind = self._requested.get(key)
        if prior_kind is not None:
            if prior_kind != kind:
                raise WorkerBridgeRefused("parent evidence request key changes kind")
            return
        retained = (set(self._requested) | set(self._completions)
                    | set(self._continuations))
        if key not in retained and len(retained) >= self._max_records:
            raise WorkerBridgeRefused("parent evidence retained-key bound exceeded")
        try:
            self._notices.put_nowait(notice)
        except queue.Full as exc:
            raise WorkerBridgeRefused("parent evidence notice queue is full") from exc
        self._requested[key] = kind

    @staticmethod
    def completion_key(*, start: WorkerStart, sequence: int, fence: ps.StageFence,
                       observation: Mapping[str, Any]) -> str:
        return _digest({"start": start.to_dict(), "sequence": sequence,
                        "fence": {"fence_id": fence.fence_id,
                                  "unit_id": fence.unit_id,
                                  "process_generation_id": fence.process_generation_id},
                        "observation": _plain(observation)})

    @staticmethod
    def continuation_key(*, start: WorkerStart, raw: ep.RawUnit,
                         plan: ep.ExperimentPlan, prompts: ps.FrozenPromptManifest,
                         previous_lineage_id: str) -> str:
        return _digest({"start": start.to_dict(), "raw": raw.to_dict(),
                        "plan_digest": plan.digest,
                        "prompt_manifest_digest": prompts.digest,
                        "previous_lineage_id": previous_lineage_id})

    def request_completion(self, *, start: WorkerStart, sequence: int,
                           fence: ps.StageFence,
                           observation: Mapping[str, Any]) -> tuple[str,
                                                                   ps.StageCompletion | None]:
        key = self.completion_key(start=start, sequence=sequence, fence=fence,
                                  observation=observation)
        notice = MappingProxyType({"kind": "completion", "key": key,
            "start": start, "sequence": sequence, "fence": fence,
            "observation": _freeze(_plain(observation))})
        with self._lock:
            found = self._completions.get(key)
            if key in self._continuations:
                raise WorkerBridgeRefused("completion key belongs to continuation evidence")
            if found is None:
                self._reserve_request(key, "completion", notice)
            return key, found

    def poll_completion(self, key: str) -> ps.StageCompletion | None:
        _sha(key, "completion evidence key")
        with self._lock:
            return self._completions.get(key)

    def publish_completion(self, key: str, completion: ps.StageCompletion) -> None:
        _sha(key, "completion evidence key")
        if not isinstance(completion, ps.StageCompletion):
            raise WorkerBridgeRefused("published completion evidence is untyped")
        with self._lock:
            if self._requested.get(key) != "completion":
                raise WorkerBridgeRefused("completion evidence was not requested")
            if key in self._completions and self._completions[key] != completion:
                raise WorkerBridgeRefused("completion evidence conflicts")
            self._completions[key] = completion

    def request_continuation(self, *, start: WorkerStart, raw: ep.RawUnit,
                             plan: ep.ExperimentPlan, prompts: ps.FrozenPromptManifest,
                             previous_lineage_id: str) -> tuple[str, bool | None]:
        key = self.continuation_key(
            start=start, raw=raw, plan=plan, prompts=prompts,
            previous_lineage_id=previous_lineage_id)
        notice = MappingProxyType({"kind": "continuation", "key": key,
            "start": start, "raw": raw, "plan": plan, "prompts": prompts,
            "previous_lineage_id": previous_lineage_id})
        with self._lock:
            found = self._continuations.get(key)
            if key in self._completions:
                raise WorkerBridgeRefused("continuation key belongs to completion evidence")
            if found is None:
                self._reserve_request(key, "continuation", notice)
            return key, found

    def poll_continuation(self, key: str) -> bool | None:
        _sha(key, "continuation evidence key")
        with self._lock:
            return self._continuations.get(key)

    def publish_continuation(self, key: str, accepted: bool) -> None:
        _sha(key, "continuation evidence key")
        if type(accepted) is not bool:
            raise WorkerBridgeRefused("published continuation verdict is untyped")
        with self._lock:
            if self._requested.get(key) != "continuation":
                raise WorkerBridgeRefused("continuation evidence was not requested")
            if key in self._continuations and self._continuations[key] != accepted:
                raise WorkerBridgeRefused("continuation evidence conflicts")
            self._continuations[key] = accepted

    def next_notice(self, timeout: float | None = None) -> Mapping[str, Any]:
        """Producer-side wait; lifecycle/watchdog code never calls this method."""
        return self._notices.get(timeout=timeout)


class UnitAuthority(Protocol):
    """Live inherited capability implemented by the parent-side watcher."""

    def admit(self, *, sequence: int, plan_digest: str, unit: ep.UnitSpec,
              prior_completion_digest: str | None) -> UnitPermit: ...
    def complete(self, *, sequence: int, fence: ps.StageFence,
                 observation: Mapping[str, Any]) -> ps.StageCompletion: ...
    def verify_continuation(self, raw: ep.RawUnit, plan: ep.ExperimentPlan,
                            prompts: ps.FrozenPromptManifest,
                            previous_lineage_id: str) -> bool: ...


class InheritedUnitAuthority:
    """Bounded request/response adapter over one inherited private Unix socket."""

    def __init__(self, sock: socket.socket, *, start: WorkerStart,
                 clock: Callable[[], float] = time.monotonic,
                 total_limit: int = 4 * 1024 * 1024) -> None:
        if (not isinstance(sock, socket.socket) or sock.family != socket.AF_UNIX
                or sock.type & socket.SOCK_STREAM != socket.SOCK_STREAM):
            raise WorkerBridgeRefused("unit authority requires an inherited Unix stream socket")
        if total_limit < MAX_MESSAGE_BYTES:
            raise WorkerBridgeRefused("unit authority total byte limit is too small")
        self.sock = sock
        self.start = WorkerStart.from_dict(start.to_dict())
        self.clock = clock
        self.total_limit = total_limit
        self.total_bytes = 0
        self.closed = False

    def _exchange(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.closed:
            raise WorkerBridgeRefused("unit authority channel is closed")
        deadline = self.start.provider_deadline
        if deadline - self.clock() <= 0:
            raise WorkerBridgeRefused("comparison-wide provider deadline expired")
        raw = exact_canonical(request)
        self.total_bytes += len(raw)
        if self.total_bytes > self.total_limit:
            self.close()
            raise WorkerBridgeRefused("unit authority total byte limit exceeded")
        try:
            write_bounded_message(self.sock, request, deadline=deadline, clock=self.clock)
            response = read_bounded_message(self.sock, deadline=deadline, clock=self.clock)
        except (OSError, TimeoutError, WorkerBridgeRefused) as exc:
            self.close()
            raise WorkerBridgeRefused("unit authority channel timed out or failed") from exc
        self.total_bytes += len(exact_canonical(response))
        if self.total_bytes > self.total_limit:
            self.close()
            raise WorkerBridgeRefused("unit authority total byte limit exceeded")
        return response

    def admit(self, *, sequence: int, plan_digest: str, unit: ep.UnitSpec,
              prior_completion_digest: str | None) -> UnitPermit:
        body = {"schema": UNIT_REQUEST_SCHEMA, "nonce": self.start.nonce,
                "sequence": sequence, "plan_digest": plan_digest,
                "lineage_id": self.start.lineage_id, "unit": unit.to_dict(),
                "prior_completion_digest": prior_completion_digest}
        response = self._exchange({**body, "request_digest": _digest(body)})
        return UnitPermit.from_dict(response, start=self.start,
                                    expected_sequence=sequence)

    def complete(self, *, sequence: int, fence: ps.StageFence,
                 observation: Mapping[str, Any]) -> ps.StageCompletion:
        body = {"schema": UNIT_COMPLETION_REQUEST_SCHEMA, "nonce": self.start.nonce,
                "sequence": sequence, "fence_id": fence.fence_id,
                "observation": _plain(observation)}
        row = _exact(self._exchange({**body, "request_digest": _digest(body)}), {
            "schema", "nonce", "sequence", "fence_id", "terminal", "stage_witnesses",
            "recorded_screen", "reason", "completion_digest"}, "unit completion")
        if (row["schema"] != UNIT_COMPLETION_SCHEMA or row["nonce"] != self.start.nonce
                or row["sequence"] != sequence or row["fence_id"] != fence.fence_id):
            raise WorkerBridgeRefused("unit completion binding differs")
        supplied = _sha(row.pop("completion_digest"), "completion_digest")
        if supplied != _digest(row) or type(row["terminal"]) is not bool \
                or not isinstance(row["stage_witnesses"], Mapping):
            raise WorkerBridgeRefused("unit completion fields/digest are invalid")
        try:
            witnesses = {key: ep.Witness.from_dict(value, f"witness {key}")
                         for key, value in row["stage_witnesses"].items()}
        except Exception as exc:
            raise WorkerBridgeRefused("unit completion witnesses are invalid") from exc
        return ps.StageCompletion(row["fence_id"], row["terminal"], witnesses,
                                  row["recorded_screen"], row["reason"])

    def verify_continuation(self, raw: ep.RawUnit, plan: ep.ExperimentPlan,
                            prompts: ps.FrozenPromptManifest,
                            previous_lineage_id: str) -> bool:
        body = {"schema": CONTINUATION_REQUEST_SCHEMA, "nonce": self.start.nonce,
                "sequence": raw.observed_order_index + 1, "raw_unit": raw.to_dict(),
                "plan_digest": plan.digest, "prompt_manifest_digest": prompts.digest,
                "previous_lineage_id": previous_lineage_id}
        row = _exact(self._exchange({**body, "request_digest": _digest(body)}), {
            "schema", "nonce", "sequence", "raw_artifact_digest", "accepted", "reason",
            "continuation_digest"}, "continuation response")
        if (row["schema"] != CONTINUATION_SCHEMA or row["nonce"] != self.start.nonce
                or row["sequence"] != raw.observed_order_index + 1
                or row["raw_artifact_digest"] != raw.artifact_digest
                or type(row["accepted"]) is not bool):
            raise WorkerBridgeRefused("continuation response binding is invalid")
        supplied = _sha(row.pop("continuation_digest"), "continuation_digest")
        if supplied != _digest(row):
            raise WorkerBridgeRefused("continuation response digest differs")
        return row["accepted"]

    def close(self) -> None:
        self.closed = True
        if self.sock.fileno() >= 0:
            self.sock.close()


class MembershipProbe(Protocol):
    def __call__(self, start: WorkerStart) -> None: ...


def verify_linux_membership(start: WorkerStart) -> None:
    """Prove this exact process and cgroup inode from kernel-backed files."""
    current = wl.process_identity(os.getpid())
    if current != start.child_process:
        raise ps.UnsupportedContainment("worker process identity differs from captured identity")
    path = Path(str(start.cgroup_identity["path"]))
    cgroup_root = Path("/sys/fs/cgroup")
    try:
        resolved_root = cgroup_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        relative = resolved_path.relative_to(resolved_root)
        candidates = []
        for line in Path("/proc/self/mountinfo").read_text(
                encoding="ascii").splitlines():
            left, separator, right = line.partition(" - ")
            fields, tail = left.split(), right.split()
            if (separator and len(fields) >= 5 and tail
                    and fields[4] == str(resolved_root)):
                candidates.append((fields[2], tail[0]))
        root_info = resolved_root.stat()
        device = f"{os.major(root_info.st_dev)}:{os.minor(root_info.st_dev)}"
        if (candidates != [(device, "cgroup2")]
                or resolved_path.stat().st_dev != root_info.st_dev):
            raise ps.UnsupportedContainment(
                "cgroup root is not the exact cgroup-v2 mount")
    except ps.UnsupportedContainment:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ps.UnsupportedContainment("owned cgroup is outside the cgroup-v2 mount") from exc
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ps.UnsupportedContainment("cannot read process cgroup membership") from exc
    unified = [row.split("::", 1)[1] for row in rows if "::" in row]
    expected_path = ("/" if relative == Path(".")
                     else "/" + relative.as_posix().lstrip("/"))
    if len(unified) != 1 or unified[0].rstrip("/") != expected_path.rstrip("/"):
        raise ps.UnsupportedContainment("process unified-cgroup path differs from owned cgroup")
    try:
        fd = os.open(resolved_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            actual = {"path": str(resolved_path), "dev": info.st_dev, "ino": info.st_ino,
                      "uid": info.st_uid, "nlink": info.st_nlink, "mode": info.st_mode}
            if actual != _plain(start.cgroup_identity) or not stat.S_ISDIR(info.st_mode):
                raise ps.UnsupportedContainment("owned cgroup identity changed")
            member_fd = os.open("cgroup.procs", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=fd)
            try:
                members = os.read(member_fd, 1024 * 1024).decode("ascii").split()
            finally:
                os.close(member_fd)
        finally:
            os.close(fd)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ps.UnsupportedContainment("cannot prove owned cgroup membership") from exc
    if str(os.getpid()) not in members:
        raise ps.UnsupportedContainment("worker PID is absent from exact owned cgroup")


class OwnedWorkerStageProvider:
    """Per-unit planned-serving facade over one live comparison-wide authority."""

    def __init__(self, *, prepared: PreparedPlannedServingStage, start: WorkerStart,
                 authority: UnitAuthority, membership_probe: MembershipProbe | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.prepared = prepared
        self.start = start
        self.authority = authority
        self._membership = membership_probe or verify_linux_membership
        self.clock = clock
        self._ordered = tuple(sorted(prepared.plan.expected_units,
                                     key=lambda item: item.order_index))
        self._next = len(prepared.previous["raw_units"]) if prepared.previous else 0
        self._active: tuple[int, ps.StageFence] | None = None
        self._prior_digest: str | None = None
        self._validate_start()

    def _validate_start(self) -> None:
        if (self.start.prepared_digest != self.prepared.prepared_digest
                or self.start.plan_digest != self.prepared.plan.digest
                or self.start.campaign_id != self.prepared.plan.campaign_id
                or self.start.config_digest != self.prepared.capture_context_base["config_digest"]
                or self.start.config_generation
                   != self.prepared.capture_context_base["config_generation"]
                or self.start.supervisor_id
                   != self.prepared.capture_context_base["supervisor_id"]
                or self.start.supervisor_incarnation
                   != self.prepared.capture_context_base["supervisor_incarnation"]):
            raise WorkerBridgeRefused("worker start differs from prepared stage")
        self._membership(self.start)

    def admit(self, plan_digest: str, unit: ep.UnitSpec,
              stages: tuple[str, ...]) -> ps.StageFence:
        if stages != ps.STAGES or plan_digest != self.prepared.plan.digest:
            raise WorkerBridgeRefused("planned stage request changed")
        if self._active is not None or self._next >= len(self._ordered) \
                or unit != self._ordered[self._next]:
            raise WorkerBridgeRefused("unit is duplicate, out of order, or undeclared")
        self._membership(self.start)
        sequence = self._next + 1
        permit = self.authority.admit(sequence=sequence, plan_digest=plan_digest, unit=unit,
                                      prior_completion_digest=self._prior_digest)
        if not isinstance(permit, UnitPermit):
            raise WorkerBridgeRefused("live unit authority returned an untyped permit")
        if not permit.allowed:
            raise ps.StagePaused(_text(permit.reason, "unit stop reason"))
        if (permit.sequence != sequence or permit.unit_id != unit.unit_id
                or permit.process_generation_id != unit.process_id
                or permit.grant_id != self.start.grant_id
                or permit.grant_generation != self.start.grant_generation
                or permit.container_id != self.start.container_id
                or permit.valid_until > self.start.provider_deadline
                or self.clock() >= permit.valid_until):
            raise WorkerBridgeRefused("unit permit differs from held allocation/fixed unit")
        fence = ps.StageFence(
            _text(permit.fence_id, "fence_id"), unit.unit_id, unit.process_id,
            self.start.lineage_id, permit.grant_id, permit.container_id,
            self.start.clock_domain, permit.valid_until, self.start.supervisor_id,
            self.start.supervisor_incarnation, self.start.config_generation,
            self.start.worker_id, self.start.worker_generation)
        self._active = (sequence, fence)
        return fence

    @contextmanager
    def guard(self, fence: ps.StageFence):
        if self._active is None or self._active[1] != fence:
            raise ps.UnsupportedContainment("unit fence is not the live admitted fence")
        self._membership(self.start)
        if self.clock() >= fence.valid_until:
            raise ps.UnsupportedContainment("unit deadline expired before guarded execution")
        yield ps.ExecutionGuard(fence.fence_id, fence.unit_id,
                                fence.process_generation_id, fence.lineage_id,
                                fence.grant_id, fence.container_id, True, True)

    def complete(self, fence: ps.StageFence,
                 observation: Mapping[str, Any]) -> ps.StageCompletion:
        if self._active is None or self._active[1] != fence:
            raise WorkerBridgeRefused("completion does not match active unit")
        sequence = self._active[0]
        completion = self.authority.complete(
            sequence=sequence, fence=fence, observation=_freeze(_plain(observation)))
        if not isinstance(completion, ps.StageCompletion):
            raise WorkerBridgeRefused("live unit authority returned untyped completion")
        self._prior_digest = _digest({"sequence": sequence, "fence_id": fence.fence_id,
                                      "observation": observation,
                                      "completion": {"fence_id": completion.fence_id,
                                          "terminal": completion.terminal,
                                          "stage_witnesses": {key: item.to_dict() for key, item
                                                              in completion.stage_witnesses.items()},
                                          "recorded_screen": completion.recorded_screen,
                                          "reason": completion.reason}})
        self._active = None
        self._next += 1
        return completion

    def verify_continuation(self, raw: ep.RawUnit, plan: ep.ExperimentPlan,
                            prompts: ps.FrozenPromptManifest,
                            previous_lineage_id: str) -> bool:
        return self.authority.verify_continuation(raw, plan, prompts, previous_lineage_id) is True


@dataclass(frozen=True)
class PlannedWorkerResult:
    body: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "PlannedWorkerResult":
        row = _exact(value, {"schema", "nonce", "prepared_digest", "plan_digest",
            "lineage_id", "stage_id", "worker_id", "worker_generation", "grant_id",
            "grant_generation", "container_id", "completed_unit_ids", "run", "captures",
            "result_digest"}, "planned worker result")
        if row["schema"] != RESULT_SCHEMA:
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
            run_row = _exact(row["run"], {"schema", "plan_digest", "prompt_manifest_digest",
                "lineage_id", "anchor_identity", "candidate_identity", "raw_units",
                "admissible_view", "use_status", "execution_complete", "paused_reason",
                "capture_receipts"}, "planned serving run")
            if run_row["schema"] != ps.RUN_SCHEMA:
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


@dataclass(frozen=True)
class PlannedWorkerResultReference:
    nonce: str
    prepared_digest: str
    worker_id: str
    worker_generation: int
    result_digest: str
    result_locator: str
    result_sha256: str
    schema: str = RESULT_REFERENCE_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "PlannedWorkerResultReference":
        row = _exact(value, {"schema", "nonce", "prepared_digest", "worker_id",
                             "worker_generation", "result_digest", "result_locator",
                             "result_sha256", "reference_digest"}, "result reference")
        if row["schema"] != RESULT_REFERENCE_SCHEMA:
            raise WorkerBridgeRefused("result reference schema is unsupported")
        supplied = _sha(row.pop("reference_digest"), "reference_digest")
        if supplied != _digest(row):
            raise WorkerBridgeRefused("result reference digest mismatch")
        return cls(_text(row["nonce"], "nonce"),
                   _sha(row["prepared_digest"], "prepared_digest"),
                   _text(row["worker_id"], "worker_id"),
                   _positive(row["worker_generation"], "worker_generation"),
                   _sha(row["result_digest"], "result_digest"),
                   _text(row["result_locator"], "result_locator"),
                   _sha(row["result_sha256"], "result_sha256"))

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "nonce": self.nonce,
                "prepared_digest": self.prepared_digest, "worker_id": self.worker_id,
                "worker_generation": self.worker_generation,
                "result_digest": self.result_digest, "result_locator": self.result_locator,
                "result_sha256": self.result_sha256}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "reference_digest": _digest(self.body())}


class PlannedWorkerInvocation:
    """Parent-owned three-channel invocation for the fixed planned worker."""

    def __init__(self, prepared: PreparedPlannedServingStage,
                 parent_authority: ParentUnitEvidenceAuthority) -> None:
        self.prepared = PreparedPlannedServingStage.from_dict(prepared.to_dict())
        if type(parent_authority) is not ParentUnitEvidenceAuthority:
            raise WorkerBridgeRefused(
                "planned invocation requires the concrete parent evidence cache")
        self.parent_authority = parent_authority
        self._worker_path = Path(__file__).resolve(strict=True)
        self._interpreter = str(Path(sys.executable).resolve(strict=True))
        owned: list[int] = []
        child_control: socket.socket | None = None
        parent_control: socket.socket | None = None
        try:
            self._start_read, self._start_write = os.pipe2(os.O_CLOEXEC)
            owned.extend((self._start_read, self._start_write))
            self._result_read, self._result_write = os.pipe2(os.O_CLOEXEC)
            owned.extend((self._result_read, self._result_write))
            child_control, parent_control = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
            self._control_child = child_control.detach()
            owned.append(self._control_child)
            self._control = parent_control
            parent_control = None
            os.set_blocking(self._start_write, False)
            os.set_blocking(self._result_read, False)
            self._control.setblocking(False)
        except BaseException:
            if child_control is not None:
                child_control.close()
            if parent_control is not None:
                parent_control.close()
            active_control = getattr(self, "_control", None)
            if isinstance(active_control, socket.socket):
                active_control.close()
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        self.argv = (self._interpreter, "-I", "-B", str(self._worker_path),
                     "--start-fd", str(self._start_read),
                     "--control-fd", str(self._control_child),
                     "--result-fd", str(self._result_write))
        self._read_buffers = {self._control.fileno(): bytearray(),
                              self._result_read: bytearray()}
        self._write_buffers = {self._start_write: bytearray(),
                               self._control.fileno(): bytearray()}
        self._total_read = 0
        self._total_written = 0
        self._hello_nonce: str | None = None
        self.start: WorkerStart | None = None
        self._reference: PlannedWorkerResultReference | None = None
        self._reference_digest: str | None = None
        self._accepted_digest: str | None = None
        self._next = len(self.prepared.previous["raw_units"]) \
            if self.prepared.previous else 0
        self._continuation_next = 0
        self._prior_completion_digest: str | None = None
        self._active: tuple[int, ps.StageFence] | None = None
        self._pending_completion: tuple[str, dict[str, Any], int,
                                        ps.StageFence] | None = None
        self._pending_continuation: tuple[str, ep.RawUnit, int] | None = None
        self._launched = False
        self._closed = False

    @classmethod
    def open(cls, prepared: PreparedPlannedServingStage,
             parent_authority: ParentUnitEvidenceAuthority) -> "PlannedWorkerInvocation":
        return cls(prepared, parent_authority)

    def stage_request(self, *, request_id: str, lineage_id: str, stage_id: str,
                      control_revision: int, stage: str = "sampling",
                      cwd: Path | None = None) -> wl.StageRequest:
        return wl.StageRequest(
            request_id, self.prepared.plan.digest, lineage_id, stage_id, stage,
            self.argv, {"PYTHONDONTWRITEBYTECODE": "1"},
            Path(cwd) if cwd is not None else self.prepared.artifact_root,
            self.prepared.prepared_digest, self.prepared.max_stage_seconds,
            self.prepared.teardown_seconds, control_revision)

    def validate_request(self, request: wl.StageRequest) -> None:
        if self._launched or self._closed:
            raise WorkerBridgeRefused("planned invocation is already launched or closed")
        if (tuple(request.argv) != self.argv
                or request.artifact_contract_digest != self.prepared.prepared_digest
                or request.plan_digest != self.prepared.plan.digest
                or request.lineage_id == "" or request.stage_id == ""
                or request.max_stage_seconds != self.prepared.max_stage_seconds
                or request.teardown_seconds != self.prepared.teardown_seconds):
            raise WorkerBridgeRefused("stage request differs from planned invocation")

    def child_fds(self) -> tuple[int, int, int]:
        if self._launched or self._closed:
            raise WorkerBridgeRefused("planned invocation descriptors are unavailable")
        self._launched = True
        result = (self._start_read, self._control_child, self._result_write)
        for fd in result:
            os.set_inheritable(fd, True)
        return result

    def close_child_fds(self) -> None:
        for name in ("_start_read", "_control_child", "_result_write"):
            fd = getattr(self, name)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, -1)

    def read_fds(self) -> tuple[int, ...]:
        return tuple(fd for fd in self._read_buffers if fd >= 0)

    def write_fds(self) -> tuple[int, ...]:
        return tuple(fd for fd, data in self._write_buffers.items() if fd >= 0 and data)

    def _queue(self, fd: int, value: Mapping[str, Any], limit: int) -> None:
        raw = _canonical(value)
        if len(raw) > limit:
            raise WorkerBridgeRefused("planned-worker parent message exceeds its byte limit")
        framed = len(raw).to_bytes(4, "big") + raw
        self._total_written += len(framed)
        if self._total_written > 4 * 1024 * 1024:
            raise WorkerBridgeRefused("planned-worker parent byte budget exceeded")
        self._write_buffers[fd].extend(framed)

    def flush_ready(self, fd: int) -> None:
        data = self._write_buffers.get(fd)
        if data is None or not data:
            return
        try:
            written = os.write(fd, data)
        except BlockingIOError:
            return
        except OSError as exc:
            raise WorkerBridgeRefused("planned-worker parent write failed") from exc
        if written <= 0:
            raise WorkerBridgeRefused("planned-worker parent write made no progress")
        del data[:written]
        if fd == self._start_write and not data:
            os.close(fd)
            del self._write_buffers[fd]
            self._start_write = -1

    def receive_ready(self, fd: int) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        buffer = self._read_buffers.get(fd)
        if buffer is None:
            raise WorkerBridgeRefused("planned-worker parent read descriptor is unknown")
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            return ()
        except OSError as exc:
            raise WorkerBridgeRefused("planned-worker parent read failed") from exc
        if not chunk:
            if buffer:
                raise WorkerBridgeRefused("planned-worker channel closed mid-message")
            del self._read_buffers[fd]
            if fd == self._result_read and self._reference is None:
                raise WorkerBridgeRefused("planned worker closed without a result reference")
            return ()
        self._total_read += len(chunk)
        if self._total_read > 4 * 1024 * 1024:
            raise WorkerBridgeRefused("planned-worker parent byte budget exceeded")
        buffer.extend(chunk)
        rows: list[tuple[str, Mapping[str, Any]]] = []
        limit = MAX_MESSAGE_BYTES
        while len(buffer) >= 4:
            size = int.from_bytes(buffer[:4], "big")
            if size < 2 or size > limit:
                raise WorkerBridgeRefused("planned-worker parent frame length is invalid")
            if len(buffer) < size + 4:
                break
            raw = bytes(buffer[4:size + 4])
            del buffer[:size + 4]
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise WorkerBridgeRefused("planned-worker parent frame is malformed") from exc
            if not isinstance(value, dict) or _canonical(value) != raw:
                raise WorkerBridgeRefused("planned-worker parent frame is not canonical")
            kind = "result" if fd == self._result_read else "control"
            rows.append((kind, _freeze(value)))
        return tuple(rows)

    def accept_hello(self, value: Mapping[str, Any]) -> wl.ProcessIdentity:
        if self._hello_nonce is not None:
            raise WorkerBridgeRefused("planned worker sent duplicate hello")
        row = _exact(value, {"schema", "nonce", "child_process", "sequence",
                             "hello_digest"}, "planned worker hello")
        if row["schema"] != HELLO_SCHEMA or row["sequence"] != 0:
            raise WorkerBridgeRefused("planned worker hello schema/sequence is invalid")
        supplied = _sha(row.pop("hello_digest"), "hello_digest")
        if supplied != _digest(row):
            raise WorkerBridgeRefused("planned worker hello digest differs")
        process = _exact(row["child_process"], {"pid", "start_ticks", "boot_id"},
                         "planned child process")
        identity = wl.ProcessIdentity(_positive(process["pid"], "child pid"),
                                      _positive(process["start_ticks"], "child start ticks"),
                                      _text(process["boot_id"], "child boot id"))
        self._hello_nonce = _text(row["nonce"], "hello nonce")
        return identity

    def queue_start(self, start: WorkerStart) -> None:
        if self._hello_nonce is None or self.start is not None:
            raise WorkerBridgeRefused("planned worker start is out of order")
        start = WorkerStart.from_dict(start.to_dict())
        if start.nonce != self._hello_nonce:
            raise WorkerBridgeRefused("planned worker start nonce differs from hello")
        body = {"schema": INVOCATION_SCHEMA, "nonce": start.nonce,
                "prepared": self.prepared.to_dict(), "start": start.to_dict()}
        self._queue(self._start_write, {**body, "invocation_digest": _digest(body)},
                    MAX_RESULT_BYTES)
        self.start = start

    def accept_result(self, value: Mapping[str, Any]) -> None:
        if self._reference is not None or self.start is None:
            raise WorkerBridgeRefused("planned worker result is duplicate or premature")
        reference = PlannedWorkerResultReference.from_dict(value)
        if (reference.nonce != self.start.nonce
                or reference.prepared_digest != self.prepared.prepared_digest
                or reference.worker_id != self.start.worker_id
                or reference.worker_generation != self.start.worker_generation):
            raise WorkerBridgeRefused("planned worker result reference binding differs")
        self._reference = reference
        self._reference_digest = _digest(reference.to_dict())

    def validate_unit_request(self, value: Mapping[str, Any]) -> tuple[int, ep.UnitSpec,
                                                                       str | None]:
        if self.start is None or self._active is not None:
            raise WorkerBridgeRefused("planned unit request is premature or overlaps")
        row = _exact(value, {"schema", "nonce", "sequence", "plan_digest",
                             "lineage_id", "unit", "prior_completion_digest",
                             "request_digest"}, "planned unit request")
        supplied = _sha(row.pop("request_digest"), "request_digest")
        sequence = self._next + 1
        ordered = tuple(sorted(self.prepared.plan.expected_units,
                               key=lambda item: item.order_index))
        if (row["schema"] != UNIT_REQUEST_SCHEMA or row["nonce"] != self.start.nonce
                or row["sequence"] != sequence
                or row["plan_digest"] != self.prepared.plan.digest
                or row["lineage_id"] != self.start.lineage_id
                or row["prior_completion_digest"] != self._prior_completion_digest
                or self._next >= len(ordered) or supplied != _digest(row)):
            raise WorkerBridgeRefused("planned unit request binding/order differs")
        try:
            unit = ep.UnitSpec.from_dict(row["unit"])
        except Exception as exc:
            raise WorkerBridgeRefused("planned unit request is invalid") from exc
        if unit != ordered[self._next]:
            raise WorkerBridgeRefused("planned unit request changed the frozen unit")
        return sequence, unit, row["prior_completion_digest"]

    def queue_unit_permit(self, *, sequence: int, unit: ep.UnitSpec,
                          fence: ps.StageFence, allowed: bool, reason: str) -> None:
        if self.start is None:
            raise WorkerBridgeRefused("planned permit lacks worker start")
        body = {"schema": UNIT_PERMIT_SCHEMA, "nonce": self.start.nonce,
                "sequence": sequence, "unit_id": unit.unit_id,
                "process_generation_id": unit.process_id, "fence_id": fence.fence_id,
                "valid_until": fence.valid_until, "grant_id": fence.grant_id,
                "grant_generation": self.start.grant_generation,
                "container_id": fence.container_id, "allowed": allowed, "reason": reason}
        self._queue(self._control.fileno(),
                    {**body, "permit_digest": _digest(body)}, MAX_MESSAGE_BYTES)
        if allowed:
            self._active = (sequence, fence)

    def handle_completion(self, value: Mapping[str, Any]) -> None:
        if (self.start is None or self._active is None
                or self._pending_completion is not None):
            raise WorkerBridgeRefused("planned completion has no active unit")
        row = _exact(value, {"schema", "nonce", "sequence", "fence_id",
                             "observation", "request_digest"},
                     "planned unit completion request")
        supplied = _sha(row.pop("request_digest"), "request_digest")
        sequence, fence = self._active
        if (row["schema"] != UNIT_COMPLETION_REQUEST_SCHEMA
                or row["nonce"] != self.start.nonce or row["sequence"] != sequence
                or row["fence_id"] != fence.fence_id or supplied != _digest(row)
                or not isinstance(row["observation"], Mapping)):
            raise WorkerBridgeRefused("planned completion request binding differs")
        key, completion = self.parent_authority.request_completion(
            start=self.start, sequence=sequence, fence=fence,
            observation=_freeze(_plain(row["observation"])))
        self._pending_completion = (key, row, sequence, fence)
        if completion is not None:
            self._finish_completion(completion)

    def _finish_completion(self, completion: ps.StageCompletion) -> None:
        if self.start is None or self._pending_completion is None:
            raise WorkerBridgeRefused("parent completion evidence is out of order")
        _key, row, sequence, fence = self._pending_completion
        if not isinstance(completion, ps.StageCompletion) or completion.fence_id != fence.fence_id:
            raise WorkerBridgeRefused("parent returned an invalid typed completion")
        body = {"schema": UNIT_COMPLETION_SCHEMA, "nonce": self.start.nonce,
                "sequence": sequence, "fence_id": fence.fence_id,
                "terminal": completion.terminal,
                "stage_witnesses": {key: item.to_dict()
                                    for key, item in completion.stage_witnesses.items()},
                "recorded_screen": completion.recorded_screen,
                "reason": completion.reason}
        self._queue(self._control.fileno(),
                    {**body, "completion_digest": _digest(body)}, MAX_MESSAGE_BYTES)
        self._prior_completion_digest = _digest({
            "sequence": sequence, "fence_id": fence.fence_id,
            "observation": row["observation"], "completion": {
                "fence_id": completion.fence_id, "terminal": completion.terminal,
                "stage_witnesses": {key: item.to_dict()
                                    for key, item in completion.stage_witnesses.items()},
                "recorded_screen": completion.recorded_screen,
                "reason": completion.reason}})
        self._active = None
        self._pending_completion = None
        self._next += 1

    def handle_continuation(self, value: Mapping[str, Any]) -> None:
        if (self.start is None or self.prepared.previous is None
                or self._pending_continuation is not None):
            raise WorkerBridgeRefused("planned continuation is unavailable")
        row = _exact(value, {"schema", "nonce", "sequence", "raw_unit",
                             "plan_digest", "prompt_manifest_digest",
                             "previous_lineage_id", "request_digest"},
                     "planned continuation request")
        supplied = _sha(row.pop("request_digest"), "request_digest")
        raw = ep.RawUnit.from_dict(row["raw_unit"])
        expected = tuple(ep.RawUnit.from_dict(item)
                         for item in self.prepared.previous["raw_units"])
        sequence = raw.observed_order_index + 1
        if (row["schema"] != CONTINUATION_REQUEST_SCHEMA
                or row["nonce"] != self.start.nonce or row["sequence"] != sequence
                or row["plan_digest"] != self.prepared.plan.digest
                or row["prompt_manifest_digest"] != self.prepared.prompts.digest
                or row["previous_lineage_id"]
                   != self.prepared.previous["previous_lineage_id"]
                or self._continuation_next >= len(expected)
                or raw != expected[self._continuation_next] or supplied != _digest(row)):
            raise WorkerBridgeRefused("planned continuation request binding/order differs")
        key, accepted = self.parent_authority.request_continuation(
            start=self.start, raw=raw, plan=self.prepared.plan,
            prompts=self.prepared.prompts,
            previous_lineage_id=row["previous_lineage_id"])
        self._pending_continuation = (key, raw, sequence)
        if accepted is not None:
            self._finish_continuation(accepted)

    def _finish_continuation(self, accepted: bool) -> None:
        if self.start is None or self._pending_continuation is None:
            raise WorkerBridgeRefused("parent continuation evidence is out of order")
        _key, raw, sequence = self._pending_continuation
        body = {"schema": CONTINUATION_SCHEMA, "nonce": self.start.nonce,
                "sequence": sequence, "raw_artifact_digest": raw.artifact_digest,
                "accepted": accepted,
                "reason": "trusted continuation accepted" if accepted
                else "trusted continuation refused"}
        self._queue(self._control.fileno(),
                    {**body, "continuation_digest": _digest(body)}, MAX_MESSAGE_BYTES)
        self._pending_continuation = None
        self._continuation_next += 1

    def poll_evidence(self) -> None:
        if self._pending_completion is not None:
            completion = self.parent_authority.poll_completion(
                self._pending_completion[0])
            if completion is not None:
                self._finish_completion(completion)
        if self._pending_continuation is not None:
            accepted = self.parent_authority.poll_continuation(
                self._pending_continuation[0])
            if accepted is not None:
                self._finish_continuation(accepted)

    def result_reference(self) -> PlannedWorkerResultReference:
        if self._reference is None or self._accepted_digest is None:
            raise WorkerBridgeRefused("planned result has not been terminally accepted")
        return self._reference

    def bind_terminal_digest(self, digest: str) -> None:
        digest = _sha(digest, "accepted planned result digest")
        if digest != self._reference_digest:
            raise WorkerBridgeRefused(
                "terminal digest differs from the exact planned result reference")
        self._accepted_digest = digest

    @property
    def reference_digest(self) -> str | None:
        return self._reference_digest

    @property
    def channels_drained(self) -> bool:
        return not self._read_buffers and not any(self._write_buffers.values())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_child_fds()
        for fd in (self._start_write, self._result_read):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if self._control.fileno() >= 0:
            self._control.close()
        self._read_buffers.clear()
        self._write_buffers.clear()


def run_prepared_stage(prepared: PreparedPlannedServingStage, start: WorkerStart, *,
                       authority: InheritedUnitAuthority | None = None,
                       clock: Callable[[], float] = time.monotonic,
                       wall_clock: Callable[[], str] = ps._utc_now,
                       _test_authority: UnitAuthority | None = None,
                       _test_membership_probe: MembershipProbe | None = None,
                       _test_measure: Callable[..., float] | None = None) \
        -> PlannedWorkerResultReference:
    """Run the actual planned consumer and seal one result; tests may inject measurement."""
    prepared = PreparedPlannedServingStage.from_dict(prepared.to_dict())
    start = WorkerStart.from_dict(start.to_dict())
    if authority is None:
        if _test_authority is None:
            raise ps.TrustedStageProviderRequired(
                "inherited lifecycle unit authority is not connected")
        active_authority = _test_authority
    elif not isinstance(authority, InheritedUnitAuthority) or _test_authority is not None:
        raise WorkerBridgeRefused("production authority must be the inherited socket adapter")
    else:
        active_authority = authority
    try:
        provider = OwnedWorkerStageProvider(
            prepared=prepared, start=start, authority=active_authority,
            membership_probe=_test_membership_probe, clock=clock)
    except BaseException:
        if authority is not None:
            authority.close()
        raise
    context = mc.CaptureContext.from_dict({**_plain(prepared.capture_context_base),
        "worker_id": start.worker_id, "worker_incarnation": start.worker_generation,
        "grant_id": start.grant_id, "container_id": start.container_id,
        "lineage_id": start.lineage_id})
    store = mc.ArtifactStore(prepared.artifact_root)
    try:
        sink = mc.DeferredNativeMeasurementSink(context=context, store=store)
        previous_raws: Sequence[ep.RawUnit] = ()
        previous_lineage: str | None = None
        continuation = None
        if prepared.previous is not None:
            previous_raws = tuple(ep.RawUnit.from_dict(item)
                                  for item in prepared.previous["raw_units"])
            previous_lineage = prepared.previous["previous_lineage_id"]
            continuation = provider.verify_continuation
        pair = prepared.runtime_pair
        run = ps.run_planned_comparison(
            prepared.plan, anchor_template=pair.anchor.template,
            candidate_template=pair.candidate.template, anchor_recipe=pair.anchor,
            candidate_recipe=pair.candidate, prompts=prepared.prompts,
            stage_provider=provider, artifact_sink=sink, lineage_id=start.lineage_id,
            clock=clock, clock_domain=start.clock_domain, wall_clock=wall_clock,
            measure=_test_measure or serving._measure_once, previous_raws=previous_raws,
            previous_lineage_id=previous_lineage, continuation_verifier=continuation)
        completed = [item.unit_id for item in run.raw_units]
        body = {"schema": RESULT_SCHEMA, "nonce": start.nonce,
            "prepared_digest": prepared.prepared_digest, "plan_digest": prepared.plan.digest,
            "lineage_id": start.lineage_id, "stage_id": start.stage_id,
            "worker_id": start.worker_id, "worker_generation": start.worker_generation,
            "grant_id": start.grant_id, "grant_generation": start.grant_generation,
            "container_id": start.container_id, "completed_unit_ids": completed,
            "run": run.to_dict(), "captures": [_plain(item) for item in sink.captures]}
        result = PlannedWorkerResult.from_dict({**body, "result_digest": _digest(body)})
        namespace = f"planned-worker-result:{prepared.prepared_digest}:{start.nonce}"
        sealed = store.write(namespace, result.to_dict())
        return PlannedWorkerResultReference(
            start.nonce, prepared.prepared_digest, start.worker_id, start.worker_generation,
            result.result_digest, sealed.locator, sealed.sha256)
    finally:
        store.close()
        if authority is not None:
            authority.close()


def ingest_deferred_result(reference: PlannedWorkerResultReference, *,
                           prepared: PreparedPlannedServingStage, start: WorkerStart,
                           terminal: wl.TerminalWorker,
                           fence: nc.TrustedWorkerResultFence,
                           capture_transaction: mc.CaptureTransaction) \
        -> tuple[Any, ...]:
    """Parent-only generation-fenced ingestion into the current native callback."""
    reference = PlannedWorkerResultReference.from_dict(reference.to_dict())
    prepared = PreparedPlannedServingStage.from_dict(prepared.to_dict())
    start = WorkerStart.from_dict(start.to_dict())
    if not isinstance(terminal, wl.TerminalWorker) or not isinstance(
            fence, nc.TrustedWorkerResultFence):
        raise WorkerBridgeRefused("terminal result fence is untyped")
    fence = nc.TrustedWorkerResultFence.from_dict(fence.to_dict())
    if (not terminal.accepted or not fence.current or not fence.result_accepted
            or terminal.worker_id != start.worker_id
            or terminal.worker_generation != start.worker_generation
            or terminal.request_id != start.request_id
            or terminal.plan_digest != start.plan_digest
            or terminal.stage_id != start.stage_id
            or terminal.grant_id != start.grant_id
            or terminal.grant_generation != start.grant_generation
            or terminal.container_id != start.container_id
            or terminal.lineage_id != start.lineage_id
            or fence.worker_id != start.worker_id
            or fence.worker_incarnation != start.worker_generation
            or fence.grant_id != start.grant_id or fence.container_id != start.container_id
            or fence.lineage_id != start.lineage_id
            or fence.config_digest != start.config_digest
            or fence.config_generation != start.config_generation
            or fence.supervisor_id != start.supervisor_id
            or fence.supervisor_incarnation != start.supervisor_incarnation
            or reference.nonce != start.nonce
            or reference.prepared_digest != prepared.prepared_digest
            or reference.worker_id != start.worker_id
            or reference.worker_generation != start.worker_generation
            or terminal.result_digest != _digest(reference.to_dict())):
        raise WorkerBridgeRefused("deferred result is stale or differs from current worker fence")
    if not callable(capture_transaction):
        raise WorkerBridgeRefused("current parent native capture callback is required")
    store = mc.ArtifactStore(prepared.artifact_root)
    try:
        body = store.read(reference.result_locator, reference.result_sha256)
        result = PlannedWorkerResult.from_dict(body)
        if result.result_digest != reference.result_digest:
            raise WorkerBridgeRefused("result reference points to a different result digest")
        store.verify(f"planned-worker-result:{prepared.prepared_digest}:{start.nonce}",
                     result.to_dict())
        row = result.to_dict()
        if (row["nonce"] != start.nonce or row["prepared_digest"] != prepared.prepared_digest
                or row["plan_digest"] != prepared.plan.digest
                or row["lineage_id"] != start.lineage_id or row["stage_id"] != start.stage_id
                or row["worker_id"] != start.worker_id
                or row["worker_generation"] != start.worker_generation
                or row["grant_id"] != start.grant_id
                or row["grant_generation"] != start.grant_generation
                or row["container_id"] != start.container_id):
            raise WorkerBridgeRefused("sealed result identity differs from current worker")
        expected = [item.unit_id for item in sorted(
            prepared.plan.expected_units, key=lambda item: item.order_index)]
        completed = row["completed_unit_ids"]
        if completed != expected[:len(completed)]:
            raise WorkerBridgeRefused("sealed result completed units are not a fixed prefix")
        run_row = row["run"]
        if (run_row["prompt_manifest_digest"] != prepared.prompts.digest
                or run_row["anchor_identity"] != _plain(prepared.plan.anchor_identity)
                or run_row["candidate_identity"]
                   != _plain(prepared.plan.candidate_identity)
                or [item["unit_id"] for item in run_row["raw_units"]] != completed):
            raise WorkerBridgeRefused("sealed run differs from prepared inputs/completed prefix")
        expected_receipts = [{"measurement_id": item["measurement_id"],
                              "payload_digest": item["payload_digest"],
                              "artifact": item["artifact"]}
                             for item in row["captures"]]
        if run_row["capture_receipts"] != expected_receipts:
            raise WorkerBridgeRefused("sealed run receipts differ from deferred captures")
        validated_captures: list[tuple[str, Mapping[str, Any]]] = []
        arms = []
        validator = nc.NativeCaptureValidator(
            binding=nc.NativeCaptureBinding(
                start.campaign_id, start.config_digest, start.config_generation,
                start.supervisor_id, start.supervisor_incarnation),
            store=store, fence_provider=lambda _measurement_id, _context: fence)
        for capture in row["captures"]:
            payload = capture["payload"]
            if _digest(payload) != capture["payload_digest"]:
                raise WorkerBridgeRefused("deferred payload digest changed")
            receipt = capture["artifact"]
            store.verify(f"carrier:{capture['measurement_id']}", payload["carrier"])
            if receipt != payload["artifact"]:
                raise WorkerBridgeRefused("deferred carrier receipt differs from payload")
            normalized = validator.validate(capture["measurement_id"], payload)
            arm = payload["carrier"].get("arm")
            arms.append(arm)
            validated_captures.append((capture["measurement_id"], normalized.payload()))
        if arms != [arm for arm in ("anchor", "candidate") if arm in arms]:
            raise WorkerBridgeRefused("deferred captures are duplicate or out of arm order")
        return tuple(capture_transaction(measurement_id, payload)
                     for measurement_id, payload in validated_captures)
    finally:
        store.close()


def write_bounded_message(sock: socket.socket, value: Mapping[str, Any], *,
                          limit: int = MAX_MESSAGE_BYTES,
                          deadline: float | None = None,
                          clock: Callable[[], float] = time.monotonic) -> None:
    """Write one length-prefixed canonical message on an explicitly owned socket."""
    raw = _canonical(value)
    if len(raw) > limit:
        raise WorkerBridgeRefused("planned-worker message exceeds its byte limit")
    view = memoryview(len(raw).to_bytes(4, "big") + raw)
    while view:
        if deadline is None:
            timeout = None
        else:
            timeout = deadline - clock()
            if timeout <= 0:
                raise WorkerBridgeRefused("planned-worker write deadline expired")
        _readable, writable, _errors = select.select([], [sock], [], timeout)
        if not writable:
            raise WorkerBridgeRefused("planned-worker write deadline expired")
        written = sock.send(view)
        if written <= 0:
            raise WorkerBridgeRefused("planned-worker socket write made no progress")
        view = view[written:]


def read_bounded_message(sock: socket.socket, *, limit: int = MAX_MESSAGE_BYTES,
                         deadline: float | None = None,
                         clock: Callable[[], float] = time.monotonic) \
        -> Mapping[str, Any]:
    """Read one bounded length-prefixed object; timeout is owned by the caller/socket."""
    def exact(size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            if deadline is None:
                timeout = None
            else:
                timeout = deadline - clock()
                if timeout <= 0:
                    raise WorkerBridgeRefused("planned-worker read deadline expired")
            readable, _writable, _errors = select.select([sock], [], [], timeout)
            if not readable:
                raise WorkerBridgeRefused("planned-worker read deadline expired")
            chunk = sock.recv(remaining)
            if not chunk:
                raise WorkerBridgeRefused("planned-worker channel closed mid-message")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    size = int.from_bytes(exact(4), "big")
    if size < 2 or size > limit:
        raise WorkerBridgeRefused("planned-worker message length is invalid")
    raw = exact(size)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkerBridgeRefused("planned-worker message is malformed") from exc
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise WorkerBridgeRefused("planned-worker message is not a canonical object")
    return _freeze(value)


def exact_canonical(value: Any) -> bytes:
    """Named helper for tests and parent watcher implementations."""
    return _canonical(value)


def _read_fd_message(fd: int, *, limit: int) -> Mapping[str, Any]:
    def exact(size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                raise WorkerBridgeRefused("planned-worker pipe closed mid-message")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    size = int.from_bytes(exact(4), "big")
    if size < 2 or size > limit:
        raise WorkerBridgeRefused("planned-worker pipe message length is invalid")
    raw = exact(size)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkerBridgeRefused("planned-worker pipe message is malformed") from exc
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise WorkerBridgeRefused("planned-worker pipe message is not canonical")
    return _freeze(value)


def _write_fd_message(fd: int, value: Mapping[str, Any], *, limit: int) -> None:
    raw = _canonical(value)
    if len(raw) > limit:
        raise WorkerBridgeRefused("planned-worker pipe message exceeds its byte limit")
    view = memoryview(len(raw).to_bytes(4, "big") + raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise WorkerBridgeRefused("planned-worker pipe write made no progress")
        view = view[written:]


def run_from_fds(*, start_fd: int, control_fd: int, result_fd: int,
                 _test_membership_probe: MembershipProbe | None = None,
                 _test_measure: Callable[..., float] | None = None) -> None:
    """Fixed production child entry: hello, one invocation, one sealed result reference."""
    if len({start_fd, control_fd, result_fd}) != 3 or min(start_fd, control_fd, result_fd) < 3:
        raise WorkerBridgeRefused("planned-worker descriptors must be distinct inherited FDs")
    for fd in (start_fd, control_fd, result_fd):
        os.set_inheritable(fd, False)
    control = socket.socket(fileno=control_fd)
    hello_nonce = secrets.token_hex(16)
    process = wl.process_identity(os.getpid())
    hello_body = {"schema": HELLO_SCHEMA, "nonce": hello_nonce,
                  "child_process": process.to_dict(), "sequence": 0}
    write_bounded_message(control, {**hello_body, "hello_digest": _digest(hello_body)})
    try:
        invocation = _exact(_read_fd_message(start_fd, limit=MAX_RESULT_BYTES), {
            "schema", "nonce", "prepared", "start", "invocation_digest"},
            "planned worker invocation")
        if invocation["schema"] != INVOCATION_SCHEMA or invocation["nonce"] != hello_nonce:
            raise WorkerBridgeRefused("invocation schema/hello nonce differs")
        supplied = _sha(invocation.pop("invocation_digest"), "invocation_digest")
        if supplied != _digest(invocation):
            raise WorkerBridgeRefused("invocation digest mismatch")
        prepared = PreparedPlannedServingStage.from_dict(invocation["prepared"])
        start = WorkerStart.from_dict(invocation["start"])
        if start.nonce != hello_nonce or start.child_process != process:
            raise WorkerBridgeRefused("worker start does not bind this hello process")
        authority = InheritedUnitAuthority(control, start=start)
        control = None
        reference = run_prepared_stage(
            prepared, start, authority=authority,
            _test_membership_probe=_test_membership_probe,
            _test_measure=_test_measure)
        _write_fd_message(result_fd, reference.to_dict(), limit=MAX_MESSAGE_BYTES)
    finally:
        os.close(start_fd)
        os.close(result_fd)
        if control is not None:
            control.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-fd", type=int, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--result-fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_from_fds(start_fd=args.start_fd, control_fd=args.control_fd,
                     result_fd=args.result_fd)
    except (WorkerBridgeRefused, ps.PlannedServingError) as exc:
        print(f"planned worker refused: {exc}", file=sys.stderr)
        return 125
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTINUATION_REQUEST_SCHEMA", "CONTINUATION_SCHEMA", "HELLO_SCHEMA",
           "INVOCATION_SCHEMA", "MAX_MESSAGE_BYTES", "MAX_RESULT_BYTES", "PREPARED_SCHEMA", "RESULT_SCHEMA",
           "RESULT_REFERENCE_SCHEMA", "START_SCHEMA", "UNIT_COMPLETION_REQUEST_SCHEMA",
           "UNIT_COMPLETION_SCHEMA", "UNIT_PERMIT_SCHEMA", "UNIT_REQUEST_SCHEMA",
           "InheritedUnitAuthority", "MembershipProbe", "OwnedWorkerStageProvider",
           "ParentUnitEvidenceAuthority", "PlannedWorkerInvocation",
           "PlannedWorkerResult", "PlannedWorkerResultReference",
           "PreparedPlannedServingStage", "UnitAuthority", "UnitPermit", "WorkerBridgeRefused",
           "WorkerStart", "exact_canonical", "ingest_deferred_result",
           "main", "read_bounded_message", "run_from_fds", "run_prepared_stage", "verify_linux_membership",
           "write_bounded_message"]
