#!/usr/bin/env python3
"""Durable consumer for one exactly owned, provider-authorized worker stage.

The module deliberately owns no journal, grant policy, or scientific grading.  A
campaign controller supplies the serialized/fsyncing event sink and control fence;
an operator-owned provider supplies a bounded grant and exact cgroup-like container.
Without both, launch refuses before a process is created.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

from ..controller.discovery_supervisor_secure import (
    RuntimeRoot,
    SecureRuntimeError,
    object_identity,
)
from .native_capture_control import TrustedWorkerResultFence
from .worker_bootstrap import MAX_OUTCOME_BYTES, OUTCOME_SCHEMA, make_contract


EVENT_SCHEMA = "epyc.autokernel.worker_lifecycle_event.v1"
ACQUISITION_SCHEMA = "epyc.autokernel.worker_acquisition_transition.v1"
RESULT_SCHEMA = "epyc.autokernel.worker_terminal_result.v1"
COMMAND_TRANSITION_SCHEMA = "epyc.autokernel.campaign_command_transition.v2"
COMMAND_RESULT_SCHEMA = "epyc.autokernel.campaign_command_result.v2"
COMMAND_RESULT_FIELDS = frozenset({
    "schema", "request_id", "operation", "payload_digest", "accepted", "accepted_at",
    "completed", "completed_at", "completion_reason", "control_revision",
    "desired_state", "observed_state", "prerequisite_reason",
})
EXPENSIVE_STAGES = frozenset({
    "setup", "build", "load", "warmup", "sampling", "correctness", "maintenance",
})
EVENTS = frozenset({
    "OWNED_LAUNCH_INTENT", "OWNED_CONTAINER_CREATED", "OWNED_CHILD_CAPTURED",
    "OWNED_EXEC_RELEASE_INTENT", "OWNED_EXEC_RELEASED", "WORKER_STAGE",
    "WORKER_RESULT_RETAINED", "OWNED_TEARDOWN_STARTED", "OWNED_TEARDOWN_FAILED",
    "OWNED_TERMINAL", "WORKER_RESULT_ACCEPTED", "WORKER_RESULT_STALE",
    "WORKER_UNRESOLVED",
})
_SECRET_MARKERS = (
    "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE_KEY",
    "API_KEY", "AUTHORIZATION", "BEARER",
)
_CONTAINER_IDENTITY_FIELDS = frozenset({"path", "dev", "ino", "uid", "nlink", "mode"})


class LifecycleRefused(RuntimeError):
    """Launch, recovery, or result use failed closed."""


class WaitingAuthority(LifecycleRefused):
    """No trusted grant/containment provider can currently authorize launch."""


class ContainmentFailure(LifecycleRefused):
    """The exact owned container could not enforce or prove cleanup."""


class SimulatedCrash(BaseException):
    """Test-only fault hook marker; callers must reconcile exact owned state."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return types.MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise LifecycleRefused(f"{label} must be lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> str:
    value = _text(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LifecycleRefused(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise LifecycleRefused(f"{label} must include timezone")
    return value


def _text(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip() or "\0" in value
            or len(value) > 16 * 1024):
        raise LifecycleRefused(f"{label} must be non-empty text")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LifecycleRefused(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or (positive and value <= 0)):
        raise LifecycleRefused(f"{label} must be finite" + (" and positive" if positive else ""))
    return float(value)


@dataclass(frozen=True)
class CampaignBinding:
    campaign_id: str
    config_digest: str
    config_generation: int
    supervisor_id: str
    supervisor_incarnation: int

    def __post_init__(self) -> None:
        _text(self.campaign_id, "campaign_id")
        _sha(self.config_digest, "config_digest")
        _positive(self.config_generation, "config_generation")
        _text(self.supervisor_id, "supervisor_id")
        _positive(self.supervisor_incarnation, "supervisor_incarnation")


@dataclass(frozen=True)
class StageRequest:
    request_id: str
    plan_digest: str
    lineage_id: str
    stage_id: str
    stage: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    artifact_contract_digest: str
    max_stage_seconds: float
    teardown_seconds: float
    control_revision: int

    def __post_init__(self) -> None:
        for name in ("request_id", "lineage_id", "stage_id"):
            _text(getattr(self, name), name)
        for name in ("plan_digest", "artifact_contract_digest"):
            _sha(getattr(self, name), name)
        if self.stage not in EXPENSIVE_STAGES:
            raise LifecycleRefused("stage is not a closed expensive lifecycle stage")
        if (not isinstance(self.argv, tuple) or not self.argv or len(self.argv) > 256
                or any(not isinstance(item, str) or not item or "\0" in item
                       or len(item) > 16 * 1024 for item in self.argv)):
            raise LifecycleRefused("argv is invalid")
        if Path(self.argv[0]).name in {"sh", "bash", "dash", "zsh", "fish", "ksh"}:
            raise LifecycleRefused("shell interpreters are forbidden worker executables")
        if (not isinstance(self.env, Mapping) or len(self.env) > 256
                or any(not isinstance(key, str) or not key or "=" in key or "\0" in key
                       or len(key) > 1024 or not isinstance(value, str) or "\0" in value
                       or len(value) > 16 * 1024
                       for key, value in self.env.items())):
            raise LifecycleRefused("env is invalid")
        if any(any(marker in key.upper() for marker in _SECRET_MARKERS) for key in self.env):
            raise LifecycleRefused("secret-bearing environment keys are forbidden")
        if (not isinstance(self.cwd, Path) or not self.cwd.is_absolute()
                or len(str(self.cwd)) > 16 * 1024):
            raise LifecycleRefused("cwd must be an absolute Path")
        _finite(self.max_stage_seconds, "max_stage_seconds", positive=True)
        _finite(self.teardown_seconds, "teardown_seconds", positive=True)
        if (not isinstance(self.control_revision, int) or isinstance(self.control_revision, bool)
                or self.control_revision < 0):
            raise LifecycleRefused("control_revision must be a non-negative integer")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "env", types.MappingProxyType(dict(self.env)))
        object.__setattr__(self, "cwd", Path(str(self.cwd)))

    @property
    def contract_body(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "env": dict(self.env), "cwd": str(self.cwd),
                "artifact_contract_digest": self.artifact_contract_digest,
                "stage": self.stage}


@dataclass(frozen=True)
class GrantReceipt:
    grant_id: str
    generation: int
    deadline: float
    clock_domain: str
    revoked: bool = False
    renewal_ok: bool = True

    def __post_init__(self) -> None:
        _text(self.grant_id, "grant_id")
        _positive(self.generation, "grant_generation")
        _finite(self.deadline, "grant deadline")
        _text(self.clock_domain, "grant clock_domain")
        if type(self.revoked) is not bool or type(self.renewal_ok) is not bool:
            raise LifecycleRefused("grant status flags must be boolean")

    def revalidated(self) -> "GrantReceipt":
        return GrantReceipt(self.grant_id, self.generation, self.deadline,
                            self.clock_domain, self.revoked, self.renewal_ok)


def monotonic_clock_domain() -> str:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise LifecycleRefused("cannot bind monotonic clock to boot identity") from exc
    return f"linux-monotonic:{_text(boot_id, 'boot_id')}"


class OwnedContainer(Protocol):
    path: Path

    def create(self) -> None: ...
    def identity(self) -> Mapping[str, Any]: ...
    def add(self, pid: int) -> None: ...
    def pids(self) -> tuple[int, ...]: ...
    def populated(self) -> bool: ...
    def signal_all(self, signum: int, identities: Mapping[int, int]) -> bool: ...
    def kill(self) -> None: ...
    def wait_empty(self, timeout: float) -> bool: ...
    def close_and_remove(self) -> None: ...


@dataclass(frozen=True)
class AuthorizedLaunch:
    grant: GrantReceipt
    container_id: str
    container: OwnedContainer

    def __post_init__(self) -> None:
        if not isinstance(self.grant, GrantReceipt):
            raise LifecycleRefused("authorized launch grant is untyped")
        if (not isinstance(self.container_id, str)
                or not self.container_id.startswith("epyc-autokernel-")
                or "/" in self.container_id):
            raise LifecycleRefused("authorized launch container_id is invalid")
        required = ("create", "identity", "add", "pids", "populated", "signal_all",
                    "kill", "wait_empty", "close_and_remove")
        if not isinstance(getattr(self.container, "path", None), Path) \
                or any(not callable(getattr(self.container, name, None)) for name in required):
            raise LifecycleRefused("authorized launch container is malformed")
        object.__setattr__(self, "grant", self.grant.revalidated())


@dataclass(frozen=True)
class AuthorizationDenied:
    request_id: str
    container_id: str
    reason: str

    def __post_init__(self) -> None:
        _text(self.request_id, "denial request_id")
        _text(self.reason, "denial reason")
        if (not isinstance(self.container_id, str)
                or not self.container_id.startswith("epyc-autokernel-")
                or "/" in self.container_id):
            raise LifecycleRefused("denial container_id is invalid")


@dataclass(frozen=True)
class ProspectiveAcquisitionIdentity:
    binding: CampaignBinding
    worker_id: str
    worker_generation: int
    request_id: str
    plan_digest: str
    lineage_id: str
    stage_id: str
    request_digest: str
    container_id: str
    control_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CampaignBinding):
            raise LifecycleRefused("prospective campaign binding is untyped")
        for name in ("worker_id", "request_id", "lineage_id", "stage_id"):
            _text(getattr(self, name), f"prospective {name}")
        _positive(self.worker_generation, "prospective worker_generation")
        _sha(self.plan_digest, "prospective plan_digest")
        _sha(self.request_digest, "prospective request_digest")
        if (not isinstance(self.container_id, str)
                or not self.container_id.startswith("epyc-autokernel-")
                or "/" in self.container_id):
            raise LifecycleRefused("prospective container_id is invalid")
        if (not isinstance(self.control_revision, int)
                or isinstance(self.control_revision, bool) or self.control_revision < 0):
            raise LifecycleRefused("prospective control_revision is invalid")

    def common_fields(self) -> dict[str, Any]:
        return {
            "campaign_id": self.binding.campaign_id,
            "config_digest": self.binding.config_digest,
            "config_generation": self.binding.config_generation,
            "supervisor_id": self.binding.supervisor_id,
            "supervisor_incarnation": self.binding.supervisor_incarnation,
            "worker_id": self.worker_id,
            "worker_generation": self.worker_generation,
            "request_id": self.request_id,
            "plan_digest": self.plan_digest,
            "lineage_id": self.lineage_id,
            "stage_id": self.stage_id,
            "request_digest": self.request_digest,
            "container_id": self.container_id,
            "control_revision": self.control_revision,
        }


@dataclass(frozen=True)
class PendingAcquisitionInspection:
    status: str
    authorization: AuthorizedLaunch | None
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"absent", "exact", "unknown"}:
            raise LifecycleRefused("pending acquisition inspection status is invalid")
        _text(self.reason, "pending acquisition inspection reason")
        if (self.status == "exact") != isinstance(self.authorization, AuthorizedLaunch):
            raise LifecycleRefused(
                "pending acquisition inspection authorization disagrees with status")


def prospective_request_digest(binding: CampaignBinding, request: StageRequest, *,
                               worker_id: str, worker_generation: int,
                               container_id: str) -> str:
    """Bind the full frozen request and its preassigned campaign/worker routing."""
    if not isinstance(binding, CampaignBinding) or not isinstance(request, StageRequest):
        raise LifecycleRefused("prospective digest inputs are untyped")
    _text(worker_id, "worker_id")
    _positive(worker_generation, "worker_generation")
    if (not isinstance(container_id, str)
            or not container_id.startswith("epyc-autokernel-") or "/" in container_id):
        raise LifecycleRefused("container_id is invalid")
    body = {
        "campaign_id": binding.campaign_id,
        "config_digest": binding.config_digest,
        "config_generation": binding.config_generation,
        "supervisor_id": binding.supervisor_id,
        "supervisor_incarnation": binding.supervisor_incarnation,
        "worker_id": worker_id,
        "worker_generation": worker_generation,
        "container_id": container_id,
        "request": {
            "request_id": request.request_id,
            "plan_digest": request.plan_digest,
            "lineage_id": request.lineage_id,
            "stage_id": request.stage_id,
            "stage": request.stage,
            "argv": list(request.argv),
            "env": dict(request.env),
            "cwd": str(request.cwd),
            "artifact_contract_digest": request.artifact_contract_digest,
            "max_stage_seconds": request.max_stage_seconds,
            "teardown_seconds": request.teardown_seconds,
            "control_revision": request.control_revision,
        },
    }
    return _digest(body)


@dataclass(frozen=True)
class RecoveryInspection:
    status: str
    authorization: AuthorizedLaunch | None
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"absent_released", "exact", "unknown"}:
            raise LifecycleRefused("recovery inspection status is invalid")
        _text(self.reason, "recovery inspection reason")
        if (self.status == "exact") != isinstance(self.authorization, AuthorizedLaunch):
            raise LifecycleRefused("recovery inspection authorization disagrees with status")


class TrustedGrantProvider(Protocol):
    """Trusted synchronous adapter whose methods enforce each supplied deadline.

    Post-return checks cannot preempt a callback that never returns.  A real adapter
    must bound its own provider and containment I/O; this consumer supplies no
    thread, daemon, broker, or out-of-band cancellation authority.
    """

    def authorize(self, request: StageRequest, container_id: str,
                  deadline: float) -> AuthorizedLaunch | AuthorizationDenied: ...
    def inspect_pending(self, identity: ProspectiveAcquisitionIdentity,
                        deadline: float) -> PendingAcquisitionInspection: ...
    def refresh(self, authorization: AuthorizedLaunch, deadline: float) -> GrantReceipt: ...
    def release(self, authorization: AuthorizedLaunch, deadline: float) -> bool: ...
    def inspect(self, grant: GrantReceipt, container_id: str,
                deadline: float) -> RecoveryInspection: ...
    def close_held_receipt(self, *, authorization: AuthorizedLaunch,
                           request: StageRequest, worker_id: str,
                           worker_generation: int,
                           container_identity: Mapping[str, Any] | None,
                           lifecycle_started_at: float, released_at: float,
                           deadline: float) -> TrustedHeldClaimReceipt: ...


@dataclass(frozen=True)
class StageAdmission:
    allowed: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise LifecycleRefused("stage admission allowed must be boolean")
        _text(self.reason, "stage admission reason")


@dataclass(frozen=True)
class RuntimeDirective:
    action: str
    reason: str
    deadline: float | None = None

    def __post_init__(self) -> None:
        if self.action not in {"continue", "drain", "revoke"}:
            raise LifecycleRefused("runtime directive action is invalid")
        _text(self.reason, "runtime directive reason")
        if self.deadline is not None:
            _finite(self.deadline, "runtime directive deadline")


AdmissionFence = Callable[[StageRequest, GrantReceipt, float, float], StageAdmission]
RuntimeFence = Callable[[StageRequest, float], RuntimeDirective]
EventSink = Callable[[Mapping[str, Any]], Any]
BindingFence = Callable[[CampaignBinding], bool]


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    boot_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "start_ticks": self.start_ticks, "boot_id": self.boot_id}


@dataclass(frozen=True)
class TerminalWorker:
    worker_id: str
    worker_generation: int
    request_id: str
    plan_digest: str
    lineage_id: str
    stage_id: str
    grant_id: str
    grant_generation: int
    container_id: str
    return_code: int | None
    result_digest: str | None
    accepted: bool
    reason: str | None


@dataclass(frozen=True)
class TrustedHeldClaimReceipt:
    """Provider-authored accounting receipt bound to one exact lifecycle owner."""

    request_id: str
    plan_digest: str
    worker_id: str
    worker_generation: int
    grant_id: str
    grant_generation: int
    container_id: str
    clock_domain: str
    held_started_at: float
    held_ended_at: float
    receipt: Any

    def __post_init__(self) -> None:
        from .scheduling import HeldClaimReceipt
        for name in ("request_id", "worker_id", "grant_id", "container_id",
                     "clock_domain"):
            _text(getattr(self, name), name)
        _sha(self.plan_digest, "plan_digest")
        _positive(self.worker_generation, "worker_generation")
        _positive(self.grant_generation, "grant_generation")
        start = _finite(self.held_started_at, "held_started_at")
        end = _finite(self.held_ended_at, "held_ended_at")
        if end <= start:
            raise LifecycleRefused("held accounting interval is invalid")
        parsed = (self.receipt if isinstance(self.receipt, HeldClaimReceipt)
                  else HeldClaimReceipt.from_dict(self.receipt))
        if (parsed.started_at != start or parsed.ended_at != end
                or parsed.ownership_generation != self.worker_generation
                or parsed.allocation_generation != self.grant_generation):
            raise LifecycleRefused("held receipt interval/generations differ")
        object.__setattr__(self, "receipt", parsed)


@dataclass(frozen=True)
class LifecycleProjection:
    revision: int
    worker_generations: Mapping[str, int]
    active: Mapping[str, Mapping[str, Any]]
    terminal: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class AcquisitionProjection:
    revision: int
    last_generation: int
    pending: Mapping[str, Any] | None


def project_acquisitions(events: Sequence[Mapping[str, Any]]) -> AcquisitionProjection:
    pending: dict[str, Any] | None = None
    last_generation = 0
    seen: set[tuple[str, int]] = set()
    for supplied in events:
        row = validate_acquisition_transition(supplied)
        identity = (row["worker_id"], row["worker_generation"])
        if row["phase"] == "INTENT":
            if pending is not None or identity in seen:
                raise LifecycleRefused("prospective acquisition intent overlaps or reuses identity")
            if last_generation and row["worker_generation"] != last_generation + 1:
                raise LifecycleRefused("prospective worker generations are not contiguous")
            pending = row
            seen.add(identity)
            last_generation = row["worker_generation"]
        else:
            if pending is None:
                raise LifecycleRefused("acquisition resolution lacks pending intent")
            if any(row[name] != pending[name] for name in pending
                   if name not in {"phase", "occurred_at", "data"}):
                raise LifecycleRefused("acquisition resolution identity changed")
            pending = None
    return AcquisitionProjection(
        len(events), last_generation,
        _freeze_json(pending) if pending is not None else None)


def project_events(events: Sequence[Mapping[str, Any]]) -> LifecycleProjection:
    states: dict[str, str] = {}
    bindings: dict[str, tuple[Any, ...]] = {}
    generations: dict[str, int] = {}
    latest: dict[str, dict[str, Any]] = {}
    terminal: dict[str, dict[str, Any]] = {}
    ownership_terminal: set[str] = set()
    last_new_generation: int | None = None
    allowed = {
        "OWNED_LAUNCH_INTENT": {None},
        "OWNED_CONTAINER_CREATED": {"OWNED_LAUNCH_INTENT"},
        "OWNED_CHILD_CAPTURED": {"OWNED_CONTAINER_CREATED"},
        "OWNED_EXEC_RELEASE_INTENT": {"OWNED_CHILD_CAPTURED"},
        "OWNED_EXEC_RELEASED": {"OWNED_EXEC_RELEASE_INTENT"},
        "WORKER_STAGE": {"OWNED_EXEC_RELEASED"},
        "WORKER_RESULT_RETAINED": {"WORKER_STAGE"},
        "OWNED_TEARDOWN_STARTED": {
            "OWNED_LAUNCH_INTENT", "OWNED_CONTAINER_CREATED", "OWNED_CHILD_CAPTURED",
            "OWNED_EXEC_RELEASE_INTENT", "OWNED_EXEC_RELEASED", "WORKER_STAGE",
            "WORKER_RESULT_RETAINED", "OWNED_TEARDOWN_FAILED", "WORKER_UNRESOLVED",
            "WORKER_RESULT_STALE",
        },
        "OWNED_TERMINAL": {"OWNED_TEARDOWN_STARTED"},
        "OWNED_TEARDOWN_FAILED": {"OWNED_TEARDOWN_STARTED"},
        "WORKER_RESULT_ACCEPTED": {"OWNED_TERMINAL"},
        "WORKER_RESULT_STALE": {"OWNED_TERMINAL", "OWNED_TEARDOWN_FAILED"},
        "WORKER_UNRESOLVED": set(EVENTS),
    }
    for supplied in events:
        row = validate_event(supplied)
        worker = row["worker_id"]
        identity = tuple(row[name] for name in (
            "campaign_id", "config_digest", "config_generation", "worker_id",
            "worker_generation", "request_id", "plan_digest", "lineage_id", "stage_id",
            "grant_id", "grant_generation", "container_id", "control_revision"))
        if worker in bindings and bindings[worker] != identity:
            raise LifecycleRefused("worker lifecycle identity changed during replay")
        if worker not in bindings:
            if (row["event"] != "OWNED_LAUNCH_INTENT"
                    or (last_new_generation is not None
                        and row["worker_generation"] <= last_new_generation)):
                raise LifecycleRefused("launched worker generations are not strictly increasing")
            last_new_generation = row["worker_generation"]
        bindings[worker] = identity
        prior_generation = generations.get(worker)
        if prior_generation is not None and prior_generation != row["worker_generation"]:
            raise LifecycleRefused("worker_id was reused with another generation")
        generations[worker] = row["worker_generation"]
        event = row["event"]
        if event != "WORKER_UNRESOLVED" and states.get(worker) not in allowed[event]:
            raise LifecycleRefused(
                f"worker lifecycle transition {states.get(worker)!r} -> {event!r} is invalid")
        states[worker] = event
        latest[worker] = row
        if event == "OWNED_TERMINAL":
            ownership_terminal.add(worker)
        if event in {"WORKER_RESULT_ACCEPTED", "WORKER_RESULT_STALE"} \
                and worker in ownership_terminal:
            terminal[worker] = row
    active = {worker: _freeze_json(row) for worker, row in latest.items()
              if worker not in terminal}
    frozen_terminal = {worker: _freeze_json(row) for worker, row in terminal.items()}
    return LifecycleProjection(
        len(events), types.MappingProxyType(dict(generations)),
        types.MappingProxyType(active), types.MappingProxyType(frozen_terminal))


def process_identity(pid: int) -> ProcessIdentity:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise LifecycleRefused("PID must be positive")
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise LifecycleRefused(f"cannot capture PID {pid} identity") from exc
    close = raw.rfind(b")")
    fields = raw[close + 1:].split()
    if close < 0 or len(fields) < 20:
        raise LifecycleRefused("process stat identity is malformed")
    return ProcessIdentity(pid, int(fields[19]), _text(boot_id, "boot_id"))


def same_process(identity: ProcessIdentity) -> bool:
    try:
        return process_identity(identity.pid) == identity
    except LifecycleRefused:
        return False


def _reap_if_child(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass


def _container_identity(value: Mapping[str, Any], expected_path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTAINER_IDENTITY_FIELDS:
        raise ContainmentFailure("owned container identity has missing/unknown fields")
    row = dict(value)
    if row["path"] != str(expected_path):
        raise ContainmentFailure("owned container path differs from preassigned identity")
    for name in ("dev", "ino", "uid", "nlink", "mode"):
        if not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] < 0:
            raise ContainmentFailure(f"owned container {name} is invalid")
    if row["uid"] != os.getuid():
        raise ContainmentFailure("owned container uid differs")
    return row


def _same_container(container: OwnedContainer, expected: Mapping[str, Any]) -> None:
    current = _container_identity(container.identity(), Path(str(expected["path"])))
    if current != dict(expected):
        raise ContainmentFailure("owned container identity changed")


def _exact_data(data: Mapping[str, Any], fields: set[str], event: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise LifecycleRefused(f"{event} data has missing/unknown fields")
    return dict(data)


def _validate_event_data(event: str, supplied: Mapping[str, Any]) -> dict[str, Any]:
    if event == "OWNED_LAUNCH_INTENT":
        data = _exact_data(supplied, {"contract", "contract_digest", "provider_deadline",
                                     "clock_domain", "max_stage_seconds", "teardown_seconds",
                                     "stage_plus_teardown_deadline"}, event)
        contract = data["contract"]
        if not isinstance(contract, Mapping) or set(contract) != {
                "argv", "env", "cwd", "artifact_contract_digest", "stage"}:
            raise LifecycleRefused("launch intent contract is not closed")
        # Reuse StageRequest's bounded primitives without accepting a second loose shape.
        if (not isinstance(contract["argv"], list) or not contract["argv"]
                or len(contract["argv"]) > 256):
            raise LifecycleRefused("launch intent argv is invalid")
        if (any(not isinstance(item, str) or not item or "\0" in item
                or len(item) > 16 * 1024
                for item in contract["argv"])
                or Path(contract["argv"][0]).name in {
                    "sh", "bash", "dash", "zsh", "fish", "ksh"}):
            raise LifecycleRefused("launch intent argv contains a forbidden value")
        if (not isinstance(contract["env"], Mapping)
                or any(not isinstance(key, str) or not key or "=" in key or "\0" in key
                       or len(key) > 1024 or not isinstance(value, str) or "\0" in value
                       or len(value) > 16 * 1024
                       for key, value in contract["env"].items())
                or any(any(marker in key.upper() for marker in _SECRET_MARKERS)
                       for key in contract["env"])):
            raise LifecycleRefused("launch intent env is invalid")
        _sha(contract["artifact_contract_digest"], "artifact_contract_digest")
        if contract["stage"] not in EXPENSIVE_STAGES:
            raise LifecycleRefused("launch intent stage is invalid")
        _text(contract["cwd"], "contract cwd")
        _sha(data["contract_digest"], "contract_digest")
        for name in ("provider_deadline", "max_stage_seconds", "teardown_seconds",
                     "stage_plus_teardown_deadline"):
            _finite(data[name], name, positive=True)
        _text(data["clock_domain"], "clock_domain")
        data["contract"] = {"argv": list(contract["argv"]), "env": dict(contract["env"]),
                            "cwd": contract["cwd"],
                            "artifact_contract_digest": contract["artifact_contract_digest"],
                            "stage": contract["stage"]}
        return data
    if event == "OWNED_CONTAINER_CREATED":
        data = _exact_data(supplied, {"identity"}, event)
        identity = data["identity"]
        if not isinstance(identity, Mapping) or set(identity) != _CONTAINER_IDENTITY_FIELDS:
            raise LifecycleRefused("container event identity is not closed")
        if not isinstance(identity["path"], str) or not Path(identity["path"]).is_absolute():
            raise LifecycleRefused("container event path is invalid")
        if any(not isinstance(identity[name], int) or isinstance(identity[name], bool)
               or identity[name] < 0 for name in _CONTAINER_IDENTITY_FIELDS - {"path"}):
            raise LifecycleRefused("container event identity fields are invalid")
        data["identity"] = dict(identity)
        return data
    if event == "OWNED_CHILD_CAPTURED":
        data = _exact_data(supplied, {"process", "container", "contract_digest"}, event)
        process = data["process"]
        if not isinstance(process, Mapping) or set(process) != {"pid", "start_ticks", "boot_id"}:
            raise LifecycleRefused("captured process identity is not closed")
        for name in ("pid", "start_ticks"):
            _positive(process[name], f"process {name}")
        _text(process["boot_id"], "process boot_id")
        if not isinstance(data["container"], Mapping) \
                or set(data["container"]) != _CONTAINER_IDENTITY_FIELDS:
            raise LifecycleRefused("captured container identity is not closed")
        if any(not isinstance(data["container"][name], int)
               or isinstance(data["container"][name], bool)
               or data["container"][name] < 0
               for name in _CONTAINER_IDENTITY_FIELDS - {"path"}):
            raise LifecycleRefused("captured container identity fields are invalid")
        _sha(data["contract_digest"], "contract_digest")
        data["process"], data["container"] = dict(process), dict(data["container"])
        return data
    if event in {"OWNED_EXEC_RELEASE_INTENT", "OWNED_EXEC_RELEASED"}:
        data = _exact_data(supplied, {"contract_digest"}, event)
        _sha(data["contract_digest"], "contract_digest")
        return data
    if event == "WORKER_STAGE":
        data = _exact_data(supplied, {"stage", "activity_at"}, event)
        if data["stage"] not in EXPENSIVE_STAGES:
            raise LifecycleRefused("worker stage is invalid")
        _timestamp(data["activity_at"], "activity_at")
        return data
    if event == "WORKER_RESULT_RETAINED":
        data = _exact_data(supplied, {"schema", "result_digest", "return_code",
                                     "bootstrap_return_code"}, event)
        if data["schema"] != RESULT_SCHEMA:
            raise LifecycleRefused("terminal result schema is invalid")
        _sha(data["result_digest"], "result_digest")
        for name in ("return_code", "bootstrap_return_code"):
            if (data[name] is not None and
                    (not isinstance(data[name], int) or isinstance(data[name], bool))):
                raise LifecycleRefused(f"{name} is invalid")
        return data
    if event == "OWNED_TEARDOWN_STARTED":
        data = _exact_data(supplied, {"termination_deadline", "clock_domain"}, event)
        _finite(data["termination_deadline"], "termination_deadline", positive=True)
        _text(data["clock_domain"], "clock_domain")
        return data
    if event == "OWNED_TERMINAL":
        data = _exact_data(supplied, {"captured_identity_dead", "container_empty",
                                     "container_removed", "claim_released",
                                     "publishers_joined"}, event)
        if any(type(value) is not bool for value in data.values()):
            raise LifecycleRefused("terminal proofs must be boolean")
        return data
    if event in {"WORKER_RESULT_ACCEPTED", "WORKER_RESULT_STALE"}:
        data = _exact_data(supplied, {"result_digest", "accepted", "reason"}, event)
        if data["result_digest"] is not None:
            _sha(data["result_digest"], "result_digest")
        if type(data["accepted"]) is not bool:
            raise LifecycleRefused("result acceptance must be boolean")
        if data["reason"] is not None:
            _text(data["reason"], "result reason")
        if (event == "WORKER_RESULT_ACCEPTED") != data["accepted"]:
            raise LifecycleRefused("result event and acceptance disagree")
        return data
    if event in {"OWNED_TEARDOWN_FAILED", "WORKER_UNRESOLVED"}:
        data = _exact_data(supplied, {"reason"}, event)
        _text(data["reason"], "failure reason")
        return data
    raise LifecycleRefused("unsupported lifecycle event")


def validate_event(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "event", "campaign_id", "config_digest", "config_generation",
              "supervisor_id", "supervisor_incarnation", "worker_id", "worker_generation",
              "request_id", "plan_digest", "lineage_id", "stage_id", "grant_id",
              "grant_generation", "container_id", "control_revision", "occurred_at", "data"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LifecycleRefused("lifecycle event has missing/unknown fields")
    row = dict(value)
    if row["schema"] != EVENT_SCHEMA or row["event"] not in EVENTS:
        raise LifecycleRefused("lifecycle event schema/type is unsupported")
    CampaignBinding(row["campaign_id"], row["config_digest"], row["config_generation"],
                    row["supervisor_id"], row["supervisor_incarnation"])
    for name in ("worker_id", "request_id", "lineage_id", "stage_id", "grant_id",
                 "container_id"):
        _text(row[name], name)
    _timestamp(row["occurred_at"], "occurred_at")
    _positive(row["worker_generation"], "worker_generation")
    _positive(row["grant_generation"], "grant_generation")
    if (not isinstance(row["control_revision"], int)
            or isinstance(row["control_revision"], bool) or row["control_revision"] < 0):
        raise LifecycleRefused("event control_revision is invalid")
    _sha(row["plan_digest"], "event plan_digest")
    row["data"] = _validate_event_data(row["event"], row["data"])
    _canonical(row)
    return row


def validate_acquisition_transition(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "phase", "campaign_id", "config_digest", "config_generation",
              "supervisor_id", "supervisor_incarnation", "worker_id",
              "worker_generation", "request_id", "plan_digest", "lineage_id",
              "stage_id", "request_digest", "container_id", "control_revision",
              "occurred_at", "data"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LifecycleRefused("acquisition transition has missing/unknown fields")
    row = dict(value)
    if row["schema"] != ACQUISITION_SCHEMA or row["phase"] not in {"INTENT", "RESOLVED"}:
        raise LifecycleRefused("acquisition transition schema/phase is invalid")
    CampaignBinding(row["campaign_id"], row["config_digest"], row["config_generation"],
                    row["supervisor_id"], row["supervisor_incarnation"])
    ProspectiveAcquisitionIdentity(
        CampaignBinding(row["campaign_id"], row["config_digest"],
                        row["config_generation"], row["supervisor_id"],
                        row["supervisor_incarnation"]),
        row["worker_id"], row["worker_generation"], row["request_id"],
        row["plan_digest"], row["lineage_id"], row["stage_id"],
        row["request_digest"], row["container_id"], row["control_revision"])
    _timestamp(row["occurred_at"], "acquisition occurred_at")
    if row["phase"] == "INTENT":
        data = _exact_data(row["data"], {"authorization_deadline", "clock_domain"},
                           "acquisition INTENT")
        _finite(data["authorization_deadline"], "authorization_deadline", positive=True)
        _text(data["clock_domain"], "acquisition clock_domain")
    else:
        data = _exact_data(
            row["data"], {"outcome", "reason", "grant_id", "grant_generation"},
            "acquisition RESOLVED")
        if data["outcome"] not in {
                "denied", "absent_released", "exact_released", "lifecycle_handoff"}:
            raise LifecycleRefused("acquisition resolution outcome is invalid")
        _text(data["reason"], "acquisition resolution reason")
        has_grant = data["outcome"] in {"exact_released", "lifecycle_handoff"}
        if has_grant:
            _text(data["grant_id"], "acquisition resolution grant_id")
            _positive(data["grant_generation"], "acquisition resolution grant_generation")
        elif data["grant_id"] is not None or data["grant_generation"] is not None:
            raise LifecycleRefused("grant-free acquisition resolution carries grant identity")
    row["data"] = data
    _canonical(row)
    return row


def prospective_identity_from_transition(
        value: Mapping[str, Any]) -> ProspectiveAcquisitionIdentity:
    row = validate_acquisition_transition(value)
    return ProspectiveAcquisitionIdentity(
        CampaignBinding(row["campaign_id"], row["config_digest"],
                        row["config_generation"], row["supervisor_id"],
                        row["supervisor_incarnation"]),
        row["worker_id"], row["worker_generation"], row["request_id"],
        row["plan_digest"], row["lineage_id"], row["stage_id"],
        row["request_digest"], row["container_id"], row["control_revision"])


def validate_lifecycle_handoff(acquisition: Mapping[str, Any],
                               launch: Mapping[str, Any]) -> GrantReceipt:
    """Verify that an actual durable launch is the prospective intent's exact handoff."""
    acquisition_row = validate_acquisition_transition(acquisition)
    launch_row = validate_event(launch)
    if launch_row["event"] != "OWNED_LAUNCH_INTENT":
        raise LifecycleRefused("acquisition handoff target is not a launch intent")
    for name in (
            "campaign_id", "config_digest", "config_generation", "supervisor_id",
            "supervisor_incarnation", "worker_id", "worker_generation", "request_id",
            "plan_digest", "lineage_id", "stage_id", "container_id", "control_revision"):
        if launch_row[name] != acquisition_row[name]:
            raise LifecycleRefused(f"acquisition/lifecycle handoff {name} differs")
    contract = launch_row["data"]["contract"]
    request = StageRequest(
        launch_row["request_id"], launch_row["plan_digest"], launch_row["lineage_id"],
        launch_row["stage_id"], contract["stage"], tuple(contract["argv"]),
        contract["env"], Path(contract["cwd"]), contract["artifact_contract_digest"],
        launch_row["data"]["max_stage_seconds"],
        launch_row["data"]["teardown_seconds"], launch_row["control_revision"])
    identity = prospective_identity_from_transition(acquisition_row)
    actual_digest = prospective_request_digest(
        identity.binding, request, worker_id=identity.worker_id,
        worker_generation=identity.worker_generation, container_id=identity.container_id)
    if actual_digest != identity.request_digest:
        raise LifecycleRefused("acquisition/lifecycle request digest differs")
    return GrantReceipt(
        launch_row["grant_id"], launch_row["grant_generation"],
        launch_row["data"]["provider_deadline"], launch_row["data"]["clock_domain"])


def validate_command_result_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != COMMAND_RESULT_FIELDS:
        raise LifecycleRefused("v2 command result has missing/unknown fields")
    row = dict(value)
    if row["schema"] != COMMAND_RESULT_SCHEMA:
        raise LifecycleRefused("v2 command result schema is unsupported")
    for name in ("request_id", "payload_digest"):
        _text(row[name], f"command result {name}")
    _sha(row["payload_digest"], "command result payload_digest")
    if row["operation"] not in {"pause", "resume", "drain"}:
        raise LifecycleRefused("v2 command result operation is invalid")
    if row["desired_state"] not in {"paused", "running", "drained"}:
        raise LifecycleRefused("v2 command desired_state is invalid")
    if row["observed_state"] not in {
            "paused", "running", "drained", "pausing", "draining",
            "waiting_prerequisite", "ownership_unresolved"}:
        raise LifecycleRefused("v2 command observed_state is invalid")
    if row["accepted"] is not True or type(row["completed"]) is not bool:
        raise LifecycleRefused("v2 command acceptance/completion flags are invalid")
    accepted_at = _timestamp(row["accepted_at"], "command accepted_at")
    if row["completed"]:
        completed_at = _timestamp(row["completed_at"], "command completed_at")
        _text(row["completion_reason"], "command completion_reason")
        accepted_time = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
        completed_time = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        if completed_time < accepted_time:
            raise LifecycleRefused("command completion precedes acceptance")
    elif row["completed_at"] is not None or row["completion_reason"] is not None:
        raise LifecycleRefused("incomplete command cannot carry completion fields")
    if (not isinstance(row["control_revision"], int)
            or isinstance(row["control_revision"], bool) or row["control_revision"] < 1):
        raise LifecycleRefused("v2 command control_revision is invalid")
    if row["prerequisite_reason"] is not None:
        _text(row["prerequisite_reason"], "command prerequisite_reason")
    semantic = (
        row["operation"], row["completed"], row["desired_state"],
        row["observed_state"], row["completion_reason"])
    ordinary_quiet_reasons = {
        "already quiescent", "owned workers quiesced and claims released"}
    allowed = False
    if semantic == ("resume", True, "running", "running", "running"):
        allowed = row["prerequisite_reason"] is None
    elif semantic == (
            "resume", True, "running", "waiting_prerequisite",
            "waiting on named prerequisite"):
        allowed = row["prerequisite_reason"] is not None
    elif (row["operation"] in {"pause", "drain"} and not row["completed"]
          and row["desired_state"]
          == ("paused" if row["operation"] == "pause" else "drained")
          and row["observed_state"]
          == ("pausing" if row["operation"] == "pause" else "draining")
          and row["completion_reason"] is None
          and row["prerequisite_reason"] is None):
        allowed = True
    elif (row["operation"] in {"pause", "drain"} and row["completed"]
          and row["desired_state"]
          == ("paused" if row["operation"] == "pause" else "drained")
          and row["observed_state"] == row["desired_state"]
          and row["completion_reason"] in ordinary_quiet_reasons
          and row["prerequisite_reason"] is None):
        allowed = True
    elif semantic == (
            "pause", True, "drained", "draining",
            "superseded by a later accepted drain"):
        allowed = row["prerequisite_reason"] is None
    if not allowed:
        raise LifecycleRefused("v2 command result semantics are not producer-defined")
    return row


def validate_command_transition_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "phase", "campaign_id", "config_digest", "config_generation",
              "supervisor_id", "supervisor_incarnation", "control_revision",
              "occurred_at", "command", "result"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LifecycleRefused("v2 command transition has missing/unknown fields")
    row = dict(value)
    if row["schema"] != COMMAND_TRANSITION_SCHEMA or row["phase"] not in {
            "ACCEPTED", "COMPLETED"}:
        raise LifecycleRefused("v2 command transition schema/phase is invalid")
    CampaignBinding(row["campaign_id"], row["config_digest"], row["config_generation"],
                    row["supervisor_id"], row["supervisor_incarnation"])
    _timestamp(row["occurred_at"], "command transition occurred_at")
    command = row["command"]
    command_fields = {"schema", "campaign_id", "config_generation", "request_id",
                      "operation", "payload", "payload_digest", "expected_control_revision"}
    if not isinstance(command, Mapping) or set(command) != command_fields:
        raise LifecycleRefused("v2 transition command is not the closed v1 request")
    if command["schema"] != "epyc.autokernel.campaign_command.v1" \
            or command["operation"] not in {"pause", "resume", "drain"} \
            or not isinstance(command["payload"], Mapping) or command["payload"]:
        raise LifecycleRefused("v2 transition command semantics are invalid")
    expected_digest = hashlib.sha256(_canonical({
        "operation": command["operation"], "payload": {},
        "campaign_id": command["campaign_id"],
        "config_generation": command["config_generation"],
    })).hexdigest()
    if command["payload_digest"] != expected_digest:
        raise LifecycleRefused("v2 transition command digest differs")
    result = validate_command_result_v2(row["result"])
    if (command["campaign_id"] != row["campaign_id"]
            or command["config_generation"] != row["config_generation"]
            or command["expected_control_revision"] != row["control_revision"] - 1
            or result["control_revision"] != row["control_revision"]
            or any(result[name] != command[name]
                   for name in ("request_id", "operation", "payload_digest"))):
        raise LifecycleRefused("v2 command transition binding/revision differs")
    if row["phase"] == "COMPLETED" and not result["completed"]:
        raise LifecycleRefused("v2 command transition phase/completion disagrees")
    row["command"], row["result"] = dict(command), result
    return row


class WorkerLifecycle:
    """Synchronous lifecycle engine for one bounded expensive stage at a time."""

    def __init__(self, *, binding: CampaignBinding, runtime: RuntimeRoot,
                 event_sink: EventSink, provider: TrustedGrantProvider | None,
                 admission_fence: AdmissionFence, binding_fence: BindingFence,
                 runtime_fence: RuntimeFence,
                 wall_clock: Callable[[], str], monotonic: Callable[[], float] = time.monotonic,
                 id_factory: Callable[[], str] | None = None,
                 fault_hook: Callable[[str], None] | None = None) -> None:
        if not isinstance(binding, CampaignBinding) or not isinstance(runtime, RuntimeRoot):
            raise TypeError("binding/runtime have wrong type")
        for callback, label in ((event_sink, "event_sink"),
                                (admission_fence, "admission_fence"),
                                (binding_fence, "binding_fence"),
                                (runtime_fence, "runtime_fence"),
                                (wall_clock, "wall_clock"), (monotonic, "monotonic")):
            if not callable(callback):
                raise TypeError(f"{label} must be callable")
        self.binding = binding
        self.runtime = runtime
        self.event_sink = event_sink
        self.provider = provider
        self.admission_fence = admission_fence
        self.binding_fence = binding_fence
        self.runtime_fence = runtime_fence
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.fault_hook = fault_hook or (lambda _phase: None)
        self._worker_generation = 0
        self._active = False
        self._ownership_unresolved = False
        self._pending_acquisition: ProspectiveAcquisitionIdentity | None = None
        self._unattached_identity: ProcessIdentity | None = None
        self._terminals: dict[tuple[str, int], TerminalWorker] = {}
        self._terminal_requests: dict[tuple[str, str, str, str], set[tuple[str, int]]] = {}
        self._terminal_bindings: dict[tuple[str, int], CampaignBinding] = {}
        self._held_receipts: dict[tuple[str, int], TrustedHeldClaimReceipt] = {}
        self._stdout_identities: dict[tuple[str, int], dict[str, int]] = {}
        self._stdout_proofs: dict[tuple[str, int], dict[str, Any]] = {}

    def _register_terminal(self, terminal: TerminalWorker,
                           binding: CampaignBinding | None = None,
                           stdout_identity: Mapping[str, int] | None = None,
                           stdout_proof: Mapping[str, Any] | None = None) -> None:
        """Index only a terminal whose final lifecycle event was durably emitted."""
        owner = binding or self.binding
        worker_key = (terminal.worker_id, terminal.worker_generation)
        request_key = (terminal.request_id, terminal.plan_digest,
                       terminal.lineage_id, terminal.stage_id)
        self._terminals[worker_key] = terminal
        self._terminal_bindings[worker_key] = owner
        self._terminal_requests.setdefault(request_key, set()).add(worker_key)
        if stdout_identity is not None:
            self._stdout_identities[worker_key] = dict(stdout_identity)
        if stdout_proof is not None:
            self._stdout_proofs[worker_key] = dict(stdout_proof)

    @staticmethod
    def stdout_leaf(worker_id: str, worker_generation: int) -> str:
        _text(worker_id, "worker_id")
        _positive(worker_generation, "worker_generation")
        leaf = hashlib.sha256(f"{worker_id}:{worker_generation}".encode()).hexdigest()[:24]
        return f"worker-{leaf}.stdout.log"

    def _stdout_identity(self, worker_id: str, worker_generation: int) \
            -> dict[str, int] | None:
        name = self.stdout_leaf(worker_id, worker_generation)
        fd = -1
        try:
            fd = self.runtime.open_leaf(name, os.O_RDONLY | os.O_NONBLOCK)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                return None
            return object_identity(info)
        except (OSError, SecureRuntimeError):
            return None
        finally:
            if fd >= 0:
                os.close(fd)

    def _emit(self, event: str, *, worker_id: str, worker_generation: int,
              request: StageRequest, grant: GrantReceipt, container_id: str,
              data: Mapping[str, Any], event_binding: CampaignBinding | None = None) -> None:
        binding = event_binding or self.binding
        row = validate_event({
            "schema": EVENT_SCHEMA, "event": event,
            "campaign_id": binding.campaign_id,
            "config_digest": binding.config_digest,
            "config_generation": binding.config_generation,
            "supervisor_id": binding.supervisor_id,
            "supervisor_incarnation": binding.supervisor_incarnation,
            "worker_id": worker_id, "worker_generation": worker_generation,
            "request_id": request.request_id, "plan_digest": request.plan_digest,
            "lineage_id": request.lineage_id, "stage_id": request.stage_id,
            "grant_id": grant.grant_id, "grant_generation": grant.generation,
            "container_id": container_id, "control_revision": request.control_revision,
            "occurred_at": self.wall_clock(), "data": dict(data),
        })
        self.event_sink(row)
        self.fault_hook(event)

    def _emit_acquisition(self, phase: str, identity: ProspectiveAcquisitionIdentity,
                          *, deadline: float | None = None,
                          outcome: str | None = None, reason: str | None = None,
                          grant: GrantReceipt | None = None) -> None:
        data: dict[str, Any]
        if phase == "INTENT":
            if deadline is None:
                raise LifecycleRefused("acquisition intent lacks deadline")
            data = {"authorization_deadline": deadline,
                    "clock_domain": monotonic_clock_domain()}
        else:
            if outcome is None or reason is None:
                raise LifecycleRefused("acquisition resolution lacks outcome/reason")
            data = {"outcome": outcome, "reason": reason,
                    "grant_id": grant.grant_id if grant is not None else None,
                    "grant_generation": grant.generation if grant is not None else None}
        row = validate_acquisition_transition({
            "schema": ACQUISITION_SCHEMA, "phase": phase,
            **identity.common_fields(), "occurred_at": self.wall_clock(), "data": data,
        })
        self.event_sink(row)
        self.fault_hook(f"WORKER_ACQUISITION_{phase}")

    def _resolve_acquisition(self, identity: ProspectiveAcquisitionIdentity, *,
                             outcome: str, reason: str,
                             grant: GrantReceipt | None = None) -> None:
        self._emit_acquisition(
            "RESOLVED", identity, outcome=outcome, reason=reason, grant=grant)
        self._pending_acquisition = None
        self._ownership_unresolved = False

    def _admit(self, request: StageRequest, grant: GrantReceipt,
               required_until: float) -> GrantReceipt:
        now = _finite(self.monotonic(), "provider clock")
        if now >= required_until - request.teardown_seconds:
            raise WaitingAuthority("stage budget expired before admission")
        decision = self.admission_fence(request, grant, now, required_until)
        if not isinstance(decision, StageAdmission):
            raise LifecycleRefused("control fence returned an untyped admission")
        if not decision.allowed:
            raise WaitingAuthority(decision.reason)
        if grant.revoked:
            raise WaitingAuthority("grant_revoked")
        if not grant.renewal_ok:
            raise WaitingAuthority("renewal_failed_future_admission")
        if grant.deadline < required_until:
            raise WaitingAuthority("insufficient_grant_deadline")
        return grant

    def run_stage(self, request: StageRequest, *, planned_invocation: Any = None) \
            -> TerminalWorker:
        try:
            return self._run_stage(request, planned_invocation=planned_invocation)
        finally:
            if planned_invocation is not None:
                close = getattr(planned_invocation, "close", None)
                if callable(close):
                    close()

    def _run_stage(self, request: StageRequest, *, planned_invocation: Any = None) \
            -> TerminalWorker:
        if not isinstance(request, StageRequest):
            raise TypeError("request must be StageRequest")
        request = StageRequest(
            request.request_id, request.plan_digest, request.lineage_id, request.stage_id,
            request.stage, tuple(request.argv), dict(request.env), Path(str(request.cwd)),
            request.artifact_contract_digest, request.max_stage_seconds,
            request.teardown_seconds, request.control_revision)
        if planned_invocation is not None:
            from .unified_worker import PlannedWorkerInvocation
            if type(planned_invocation) is not PlannedWorkerInvocation:
                raise TypeError("planned_invocation must be the closed planned-worker type")
            try:
                planned_invocation.validate_request(request)
            except BaseException:
                planned_invocation.close()
                raise
        if self._active:
            if planned_invocation is not None:
                planned_invocation.close()
            raise LifecycleRefused("a worker stage is already active")
        if self._ownership_unresolved:
            if planned_invocation is not None:
                planned_invocation.close()
            raise ContainmentFailure("prior owned worker state remains unresolved")
        if self.provider is None:
            if planned_invocation is not None:
                planned_invocation.close()
            raise WaitingAuthority("trusted grant provider is unavailable")
        self.runtime.verify()
        worker_id = f"worker-{self.id_factory()}"
        container_id = f"epyc-autokernel-{self.id_factory()}"
        self._worker_generation += 1
        generation = self._worker_generation
        started_at = _finite(self.monotonic(), "stage budget start")
        overall_deadline = started_at + request.max_stage_seconds + request.teardown_seconds
        prospective = ProspectiveAcquisitionIdentity(
            self.binding, worker_id, generation, request.request_id, request.plan_digest,
            request.lineage_id, request.stage_id,
            prospective_request_digest(
                self.binding, request, worker_id=worker_id,
                worker_generation=generation, container_id=container_id),
            container_id, request.control_revision)
        self._pending_acquisition = prospective
        self._ownership_unresolved = True
        self._emit_acquisition("INTENT", prospective, deadline=overall_deadline)
        try:
            authorization = self.provider.authorize(request, container_id, overall_deadline)
        except BaseException as exc:
            raise ContainmentFailure(
                "provider authorization outcome is ambiguous; exact claim reconciliation required"
            ) from exc
        if isinstance(authorization, AuthorizationDenied):
            denial = AuthorizationDenied(
                authorization.request_id, authorization.container_id, authorization.reason)
            if (denial.request_id != request.request_id
                    or denial.container_id != container_id):
                raise ContainmentFailure("provider denial binding is ambiguous")
            self._resolve_acquisition(
                prospective, outcome="denied", reason=denial.reason)
            raise WaitingAuthority(denial.reason)
        if not isinstance(authorization, AuthorizedLaunch):
            raise ContainmentFailure(
                "provider authorization result is malformed and acquisition is unresolved")
        if self.monotonic() > overall_deadline:
            if (authorization.container_id != container_id
                    or not self.provider.release(authorization, overall_deadline)):
                raise ContainmentFailure(
                    "late provider acquisition could not be exactly released")
            self._resolve_acquisition(
                prospective, outcome="exact_released",
                reason="late authorization was exactly released",
                grant=authorization.grant)
            raise WaitingAuthority("trusted provider authorize exceeded lifecycle deadline")
        grant = authorization.grant.revalidated()
        authorization = AuthorizedLaunch(grant, authorization.container_id,
                                         authorization.container)
        if authorization.container_id != container_id:
            try:
                self.provider.release(authorization, overall_deadline)
            except BaseException:
                pass
            raise ContainmentFailure(
                "provider changed preassigned container identity; pending identity is unresolved")
        if grant.clock_domain != monotonic_clock_domain():
            if not self.provider.release(authorization, overall_deadline):
                raise ContainmentFailure("wrong-clock grant could not be released")
            self._resolve_acquisition(
                prospective, outcome="exact_released",
                reason="wrong-clock authorization was exactly released", grant=grant)
            raise WaitingAuthority("grant clock domain is not current boot monotonic")
        try:
            grant = self._admit(request, grant, overall_deadline)
        except BaseException as exc:
            if (not self.provider.release(authorization, overall_deadline)
                    or self.monotonic() > overall_deadline):
                raise ContainmentFailure("unadmitted provider grant release failed")
            self._resolve_acquisition(
                prospective, outcome="exact_released",
                reason="unadmitted authorization was exactly released", grant=grant)
            raise exc
        self._active = True
        process: subprocess.Popen[bytes] | None = None
        gate_read = gate_write = contract_read = contract_write = None
        outcome_read = outcome_write = None
        stdout_fd = stderr_fd = None
        owned_fds: set[int] = set()

        def owned_pipe() -> tuple[int, int]:
            read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
            owned_fds.update((read_fd, write_fd))
            return read_fd, write_fd

        def close_owned(fd: int) -> None:
            if fd not in owned_fds:
                return
            owned_fds.remove(fd)
            os.close(fd)

        captured: ProcessIdentity | None = None
        result: dict[str, Any] | None = None
        retained_digest: str | None = None
        cleanup_ok = False
        container_identity: dict[str, Any] | None = None
        container_create_attempted = False
        failure: BaseException | None = None
        simulated_crash = False
        held_receipt: TrustedHeldClaimReceipt | None = None
        try:
            nonce = self.id_factory()
            contract = make_contract(nonce=nonce, argv=request.argv, env=request.env,
                                     cwd=str(request.cwd))
            self._emit("OWNED_LAUNCH_INTENT", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id, data={
                           "contract": request.contract_body,
                           "contract_digest": contract["contract_digest"],
                           "provider_deadline": grant.deadline,
                           "clock_domain": grant.clock_domain,
                           "max_stage_seconds": request.max_stage_seconds,
                           "teardown_seconds": request.teardown_seconds,
                           "stage_plus_teardown_deadline": overall_deadline,
                       })
            self._resolve_acquisition(
                prospective, outcome="lifecycle_handoff",
                reason="real grant handed to durable owned lifecycle", grant=grant)
            container_create_attempted = True
            authorization.container.create()
            container_identity = _container_identity(
                authorization.container.identity(), authorization.container.path)
            self._emit("OWNED_CONTAINER_CREATED", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id, data={"identity": container_identity})

            gate_read, gate_write = owned_pipe()
            contract_read, contract_write = owned_pipe()
            outcome_read, outcome_write = owned_pipe()
            for fd in (gate_read, contract_read, outcome_write):
                os.set_inheritable(fd, True)
            leaf = hashlib.sha256(f"{worker_id}:{generation}".encode()).hexdigest()[:24]
            stdout_fd = self.runtime.open_append(f"worker-{leaf}.stdout.log")
            owned_fds.add(stdout_fd)
            stderr_fd = self.runtime.open_append(f"worker-{leaf}.stderr.log")
            owned_fds.add(stderr_fd)
            for fd in (stdout_fd, stderr_fd):
                os.set_inheritable(fd, True)
            planned_child_fds: tuple[int, int, int] | None = None
            if planned_invocation is not None:
                planned_child_fds = planned_invocation.child_fds()
            bootstrap_path = Path(__file__).with_name("worker_bootstrap.py").resolve(strict=True)
            bootstrap: tuple[str, ...] = (
                sys.executable, "-I", "-S", "-B", str(bootstrap_path),
                "--gate-fd", str(gate_read), "--contract-fd", str(contract_read),
                "--outcome-fd", str(outcome_write), "--stdout-fd", str(stdout_fd),
                "--stderr-fd", str(stderr_fd),
            )
            if planned_child_fds is not None:
                bootstrap += (
                    "--planned-start-fd", str(planned_child_fds[0]),
                    "--planned-control-fd", str(planned_child_fds[1]),
                    "--planned-result-fd", str(planned_child_fds[2]),
                )
            bootstrap_env = {"PYTHONDONTWRITEBYTECODE": "1"}
            inherited = (gate_read, contract_read, outcome_write, stdout_fd, stderr_fd)
            if planned_child_fds is not None:
                inherited += planned_child_fds
            process = subprocess.Popen(
                bootstrap, cwd=str(request.cwd), env=bootstrap_env,
                stdin=subprocess.DEVNULL, stdout=stderr_fd, stderr=stderr_fd,
                close_fds=True,
                pass_fds=inherited)
            if planned_invocation is not None:
                planned_invocation.close_child_fds()
            self._unattached_identity = process_identity(process.pid)
            self.fault_hook("AFTER_POPEN_BEFORE_ATTACH")
            for fd in (gate_read, contract_read, outcome_write, stdout_fd, stderr_fd):
                close_owned(fd)
            gate_read = contract_read = outcome_write = None
            stdout_fd = stderr_fd = None
            _same_container(authorization.container, container_identity)
            authorization.container.add(process.pid)
            captured = process_identity(process.pid)
            self._unattached_identity = None
            if process.pid not in authorization.container.pids():
                raise ContainmentFailure("bootstrap is absent from its exact owned container")
            self._emit("OWNED_CHILD_CAPTURED", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id, data={
                           "process": captured.to_dict(), "container": container_identity,
                           "contract_digest": contract["contract_digest"],
                       })
            raw_contract = _canonical(contract)
            self._write_pipe(contract_write, raw_contract,
                             min(overall_deadline - request.teardown_seconds,
                                 grant.deadline - request.teardown_seconds))
            close_owned(contract_write)
            contract_write = None

            refreshed = self.provider.refresh(authorization, overall_deadline)
            if self.monotonic() > overall_deadline:
                raise WaitingAuthority("trusted provider refresh exceeded lifecycle deadline")
            if not isinstance(refreshed, GrantReceipt):
                raise WaitingAuthority("trusted provider refresh is malformed")
            refreshed = refreshed.revalidated()
            if ((refreshed.grant_id, refreshed.generation)
                    != (grant.grant_id, grant.generation)):
                raise WaitingAuthority("grant identity/generation changed before execution")
            grant = self._admit(request, refreshed, overall_deadline)
            _same_container(authorization.container, container_identity)
            self._emit("OWNED_EXEC_RELEASE_INTENT", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id, data={"contract_digest": contract["contract_digest"]})
            os.write(gate_write, b"G")
            close_owned(gate_write)
            gate_write = None
            self._emit("OWNED_EXEC_RELEASED", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id, data={"contract_digest": contract["contract_digest"]})
            self._emit("WORKER_STAGE", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id,
                       data={"stage": request.stage, "activity_at": self.wall_clock()})
            result = self._wait_outcome(
                process, outcome_read, nonce, contract["contract_digest"], request,
                authorization, grant,
                min(overall_deadline - request.teardown_seconds,
                    grant.deadline - request.teardown_seconds),
                planned_invocation=planned_invocation,
                container_identity=container_identity,
                worker_id=worker_id, worker_generation=generation,
                container_id=container_id)
            close_owned(outcome_read)
            outcome_read = None
            retained_digest = (planned_invocation.reference_digest
                               if planned_invocation is not None else _digest(result))
            if retained_digest is None:
                raise LifecycleRefused("planned result reference was not retained")
            self._emit("WORKER_RESULT_RETAINED", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=container_id,
                       data={"schema": RESULT_SCHEMA, "result_digest": retained_digest,
                             "return_code": result["return_code"],
                             "bootstrap_return_code": process.poll()})
        except SimulatedCrash:
            simulated_crash = True
            raise
        except BaseException as exc:
            failure = exc
        finally:
            while owned_fds:
                fd = owned_fds.pop()
                try:
                    os.close(fd)
                except OSError:
                    pass
            if planned_invocation is not None:
                planned_invocation.close()
            if not simulated_crash:
                try:
                    cleanup_ok = self._teardown(
                        authorization, request, worker_id, generation, grant, process, captured,
                        container_identity, container_create_attempted, overall_deadline)
                except BaseException as exc:
                    cleanup_ok = False
                    # Cleanup/durability failure is the controlling terminal
                    # error even when stage execution had already failed.
                    failure = exc
                self._active = not cleanup_ok
                self._ownership_unresolved = not cleanup_ok
                close_receipt = getattr(self.provider, "close_held_receipt", None)
                if cleanup_ok and callable(close_receipt):
                    try:
                        released_at = _finite(self.monotonic(), "held release time")
                        if released_at > overall_deadline:
                            raise WaitingAuthority(
                                "held receipt deadline expired after exact release")
                        supplied = close_receipt(
                            authorization=authorization, request=request,
                            worker_id=worker_id, worker_generation=generation,
                            container_identity=(None if container_identity is None
                                                else dict(container_identity)),
                            lifecycle_started_at=started_at,
                            released_at=released_at, deadline=overall_deadline)
                        if not isinstance(supplied, TrustedHeldClaimReceipt):
                            raise WaitingAuthority(
                                "trusted provider held receipt is unavailable")
                        if (supplied.request_id != request.request_id
                                or supplied.plan_digest != request.plan_digest
                                or supplied.worker_id != worker_id
                                or supplied.worker_generation != generation
                                or supplied.grant_id != grant.grant_id
                                or supplied.grant_generation != grant.generation
                                or supplied.container_id != container_id
                                or supplied.clock_domain != grant.clock_domain
                                or supplied.held_started_at > started_at
                                or supplied.held_ended_at != released_at):
                            raise WaitingAuthority(
                                "trusted provider held receipt binding differs")
                        held_receipt = supplied
                        returned_at = _finite(
                            self.monotonic(), "held receipt return time")
                        if returned_at > overall_deadline:
                            raise WaitingAuthority(
                                "trusted provider held receipt returned after lifecycle deadline")
                    except BaseException as exc:
                        failure = exc

        if not cleanup_ok:
            if failure is not None:
                raise failure
            raise ContainmentFailure("owned cleanup failed without a terminal proof")
        accepted = bool(cleanup_ok and failure is None and result is not None
                        and result["return_code"] == 0 and self.binding_fence(self.binding))
        reason = None if accepted else (
            str(failure) if failure is not None else
            "worker returned nonzero" if result is not None and result["return_code"] != 0 else
            "binding is stale" if cleanup_ok else "owned cleanup failed")
        terminal = TerminalWorker(
            worker_id, generation, request.request_id, request.plan_digest,
            request.lineage_id, request.stage_id, grant.grant_id, grant.generation,
            container_id, result["return_code"] if result is not None else None,
            retained_digest, accepted, reason)
        event = "WORKER_RESULT_ACCEPTED" if accepted else "WORKER_RESULT_STALE"
        stdout_identity = self._stdout_identity(worker_id, generation)
        stdout_proof = None if result is None else {
            "stdout_bytes": result["stdout_bytes"],
            "stdout_sha256": result["stdout_sha256"],
            "stdout_truncated": result["stdout_truncated"],
        }
        self._emit(event, worker_id=worker_id, worker_generation=generation,
                   request=request, grant=grant, container_id=container_id,
                   data={"result_digest": retained_digest, "accepted": accepted,
                         "reason": reason})
        self._register_terminal(
            terminal, stdout_identity=stdout_identity, stdout_proof=stdout_proof)
        if held_receipt is not None:
            self._held_receipts[(worker_id, generation)] = held_receipt
        if planned_invocation is not None and accepted and retained_digest is not None:
            planned_invocation.bind_terminal_digest(retained_digest)
        if failure is not None:
            raise LifecycleRefused(reason or "worker stage failed") from failure
        return terminal

    def _write_pipe(self, fd: int, raw: bytes, deadline: float) -> None:
        os.set_blocking(fd, False)
        view = memoryview(raw)
        while view:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise LifecycleRefused("bootstrap contract transfer deadline expired")
            _readable, writable, _errors = select.select([], [fd], [], min(remaining, 0.05))
            if not writable:
                continue
            try:
                written = os.write(fd, view)
            except BrokenPipeError as exc:
                raise LifecycleRefused("bootstrap closed its contract pipe") from exc
            if written <= 0:
                raise LifecycleRefused("bootstrap contract pipe made no progress")
            view = view[written:]

    def _wait_outcome(self, process: subprocess.Popen[bytes], fd: int,
                      nonce: str, contract_digest: str, request: StageRequest,
                      authorization: AuthorizedLaunch, held_grant: GrantReceipt,
                      deadline: float, *, planned_invocation: Any = None,
                      container_identity: Mapping[str, Any] | None = None,
                      worker_id: str | None = None,
                      worker_generation: int | None = None,
                      container_id: str | None = None) -> dict[str, Any]:
        chunks: list[bytes] = []
        size = 0
        eof = False
        while not eof or (planned_invocation is not None
                          and not planned_invocation.channels_drained):
            now = self.monotonic()
            if now >= deadline:
                raise LifecycleRefused("worker stage deadline expired")
            if not self.binding_fence(self.binding):
                raise LifecycleRefused("campaign/supervisor binding became stale")
            directive = self.runtime_fence(request, now)
            if not isinstance(directive, RuntimeDirective):
                raise LifecycleRefused("runtime control fence returned an untyped directive")
            if directive.action != "continue":
                stop_at = (directive.deadline if directive.deadline is not None
                           else (deadline if directive.action == "drain" else now))
                deadline = min(deadline, stop_at)
                if now >= stop_at:
                    raise LifecycleRefused(directive.reason)
            refreshed = self.provider.refresh(authorization, deadline) if self.provider else None
            if self.monotonic() > deadline:
                raise LifecycleRefused("provider refresh exceeded active-stage deadline")
            if not isinstance(refreshed, GrantReceipt):
                raise LifecycleRefused("active grant refresh is malformed")
            refreshed = refreshed.revalidated()
            if ((refreshed.grant_id, refreshed.generation, refreshed.clock_domain)
                    != (held_grant.grant_id, held_grant.generation,
                        held_grant.clock_domain)):
                raise LifecycleRefused("active grant identity/generation changed")
            if refreshed.revoked:
                raise LifecycleRefused("active grant was revoked")
            # renewal_ok=false only closes successors.  This admitted stage keeps
            # its original held deadline and is not killed merely for an outage.
            deadline = min(deadline, refreshed.deadline - request.teardown_seconds)
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise LifecycleRefused("worker stage deadline expired")
            reads = [] if eof else [fd]
            writes: list[int] = []
            if planned_invocation is not None:
                planned_invocation.poll_evidence()
                reads.extend(planned_invocation.read_fds())
                writes.extend(planned_invocation.write_fds())
            ready, writable, _ = select.select(
                reads, writes, [], min(remaining, 0.05))
            for write_fd in writable:
                planned_invocation.flush_ready(write_fd)
            if not eof and fd in ready:
                chunk = os.read(fd, min(65536, MAX_OUTCOME_BYTES + 1 - size))
                if not chunk:
                    eof = True
                else:
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_OUTCOME_BYTES:
                        raise LifecycleRefused("worker outcome exceeded size limit")
            if planned_invocation is not None:
                for ready_fd in ready:
                    if ready_fd == fd:
                        continue
                    for kind, message in planned_invocation.receive_ready(ready_fd):
                        if kind == "result":
                            planned_invocation.accept_result(message)
                            continue
                        schema = message.get("schema")
                        if schema == "epyc.autokernel.planned_worker_hello.v1":
                            child = planned_invocation.accept_hello(message)
                            if (container_identity is None or worker_id is None
                                    or worker_generation is None or container_id is None):
                                raise LifecycleRefused("planned worker binding is incomplete")
                            _same_container(authorization.container, container_identity)
                            if (not same_process(child)
                                    or child.pid not in authorization.container.pids()):
                                raise ContainmentFailure(
                                    "planned child is absent from its exact owned container")
                            from .unified_worker import WorkerStart
                            start = WorkerStart(
                                str(message["nonce"]),
                                request.request_id,
                                planned_invocation.prepared.prepared_digest,
                                request.plan_digest, request.lineage_id, request.stage_id,
                                self.binding.campaign_id, self.binding.config_digest,
                                self.binding.config_generation, self.binding.supervisor_id,
                                self.binding.supervisor_incarnation, worker_id,
                                worker_generation, held_grant.grant_id,
                                held_grant.generation, container_id, child,
                                dict(container_identity), held_grant.clock_domain,
                                deadline)
                            planned_invocation.queue_start(start)
                        elif schema == "epyc.autokernel.planned_worker_unit_request.v1":
                            sequence, unit, _prior = \
                                planned_invocation.validate_unit_request(message)
                            start = planned_invocation.start
                            if start is None or not same_process(start.child_process) \
                                    or start.child_process.pid not in authorization.container.pids():
                                raise ContainmentFailure("planned child membership became stale")
                            directive = self.runtime_fence(request, self.monotonic())
                            allowed = directive.action == "continue"
                            if allowed:
                                self._admit(
                                    request, refreshed, deadline + request.teardown_seconds)
                            reason = ("held lifecycle allocation admitted fixed unit" if allowed
                                      else directive.reason)
                            from .planned_serving import StageFence
                            fence = StageFence(
                                f"fence-{self.id_factory()}", unit.unit_id, unit.process_id,
                                request.lineage_id, held_grant.grant_id, container_id,
                                held_grant.clock_domain, deadline, self.binding.supervisor_id,
                                self.binding.supervisor_incarnation,
                                self.binding.config_generation, worker_id, worker_generation)
                            planned_invocation.queue_unit_permit(
                                sequence=sequence, unit=unit, fence=fence,
                                allowed=allowed, reason=reason)
                        elif schema == "epyc.autokernel.planned_worker_unit_completion_request.v1":
                            planned_invocation.handle_completion(message)
                        elif schema == "epyc.autokernel.planned_worker_continuation_request.v1":
                            planned_invocation.handle_continuation(message)
                        else:
                            raise LifecycleRefused("planned worker control schema is unsupported")
        try:
            row = json.loads(b"".join(chunks))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LifecycleRefused("worker outcome is malformed") from exc
        fields = {"schema", "nonce", "contract_digest", "child_pid", "return_code",
                  "forwarded_signal", "stdout_bytes", "stderr_bytes", "stdout_sha256",
                  "stderr_sha256", "stdout_truncated", "stderr_truncated"}
        if (not isinstance(row, Mapping) or set(row) != fields
                or row["schema"] != OUTCOME_SCHEMA
                or row["nonce"] != nonce or row["contract_digest"] != contract_digest
                or not isinstance(row["return_code"], int)
                or isinstance(row["return_code"], bool)
                or not isinstance(row["child_pid"], int) or row["child_pid"] < 1
                or (row["forwarded_signal"] is not None
                    and not isinstance(row["forwarded_signal"], int))
                or any(not isinstance(row[name], int) or isinstance(row[name], bool)
                       or row[name] < 0 for name in ("stdout_bytes", "stderr_bytes"))
                or any(type(row[name]) is not bool
                       for name in ("stdout_truncated", "stderr_truncated"))
                or any(not isinstance(row[name], str) or len(row[name]) != 64
                       or any(char not in "0123456789abcdef" for char in row[name])
                       for name in ("stdout_sha256", "stderr_sha256"))):
            raise LifecycleRefused("worker outcome schema is invalid")
        process.wait(timeout=max(0.0, deadline - self.monotonic()))
        if planned_invocation is not None and planned_invocation.reference_digest is None:
            raise LifecycleRefused("planned worker outcome lacks its sealed result reference")
        return dict(row)

    def _member_identities(self, container: OwnedContainer) -> dict[int, int]:
        identities: dict[int, int] = {}
        for pid in container.pids():
            try:
                identities[pid] = process_identity(pid).start_ticks
            except LifecycleRefused:
                continue
        return identities

    def _teardown(self, authorization: AuthorizedLaunch, request: StageRequest,
                  worker_id: str, generation: int, grant: GrantReceipt,
                  process: subprocess.Popen[bytes] | None,
                  captured: ProcessIdentity | None,
                  container_identity: Mapping[str, Any] | None,
                  container_create_attempted: bool,
                  overall_deadline: float,
                  event_binding: CampaignBinding | None = None) -> bool:
        deadline = min(grant.deadline, overall_deadline)
        publication_failure: BaseException | None = None
        try:
            self._emit("OWNED_TEARDOWN_STARTED", worker_id=worker_id,
                       worker_generation=generation, request=request, grant=grant,
                       container_id=authorization.container_id,
                       data={"termination_deadline": deadline,
                             "clock_domain": grant.clock_domain},
                       event_binding=event_binding)
        except SimulatedCrash:
            raise
        except BaseException as exc:
            # Durability is now uncertain, but exact owned cleanup is still
            # mandatory.  Preserve this original publication error after the
            # cleanup attempt and do not certify a terminal transition.
            publication_failure = exc
        container = authorization.container
        try:
            if container_identity is None:
                if container_create_attempted:
                    raise ContainmentFailure(
                        "container creation was attempted without a pinned identity")
                # No container operation was attempted; only the provider claim needs release.
                if not self.provider or not self.provider.release(authorization, deadline):
                    raise ContainmentFailure("trusted provider did not release pre-container claim")
            else:
                _same_container(container, container_identity)
                if process is not None:
                    process.poll()  # Reap an already-finished bootstrap before emptiness checks.
                if container.populated():
                    container.signal_all(signal.SIGTERM, self._member_identities(container))
                remaining = max(0.0, deadline - self.monotonic())
                if process is not None and process.poll() is None:
                    try:
                        process.wait(timeout=min(remaining, request.teardown_seconds / 2))
                    except subprocess.TimeoutExpired:
                        pass
                if container.populated():
                    container.kill()
                remaining = max(0.0, deadline - self.monotonic())
                if process is not None:
                    try:
                        process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired as exc:
                        raise ContainmentFailure("captured bootstrap did not reap") from exc
                elif captured is not None:
                    while self.monotonic() < deadline:
                        _reap_if_child(captured.pid)
                        if not container.populated():
                            break
                        time.sleep(min(0.005, max(0.0, deadline - self.monotonic())))
                remaining = max(0.0, deadline - self.monotonic())
                if not container.wait_empty(remaining):
                    raise ContainmentFailure("owned container survived kill/wait-empty")
                if captured is not None and same_process(captured):
                    raise ContainmentFailure("captured PID identity remains alive")
                _same_container(container, container_identity)
                container.close_and_remove()
                if getattr(container, "path", Path("/")).exists():
                    raise ContainmentFailure("owned container survived exact removal")
                if not self.provider or not self.provider.release(authorization, deadline):
                    raise ContainmentFailure("trusted provider did not acknowledge claim release")
                if self.monotonic() > deadline:
                    raise ContainmentFailure("owned teardown exceeded its one deadline")
        except BaseException as exc:
            if publication_failure is not None:
                raise publication_failure from exc
            try:
                self._emit("OWNED_TEARDOWN_FAILED", worker_id=worker_id,
                           worker_generation=generation, request=request, grant=grant,
                           container_id=authorization.container_id,
                           data={"reason": f"{type(exc).__name__}: {exc}"},
                           event_binding=event_binding)
            except SimulatedCrash:
                raise
            except BaseException:
                # Cleanup failure is the primary evidence; a second journal
                # failure must not replace it.
                pass
            raise
        if publication_failure is not None:
            raise publication_failure
        self._emit("OWNED_TERMINAL", worker_id=worker_id,
                   worker_generation=generation, request=request, grant=grant,
                   container_id=authorization.container_id,
                   data={"captured_identity_dead": True, "container_empty": True,
                         "container_removed": True, "claim_released": True,
                         "publishers_joined": True}, event_binding=event_binding)
        return True

    def trusted_result_fence(self, terminal: TerminalWorker) -> TrustedWorkerResultFence:
        current = self._terminals.get((terminal.worker_id, terminal.worker_generation))
        if current != terminal:
            raise LifecycleRefused("terminal worker receipt is not owned by this lifecycle")
        is_current = bool(
            terminal.accepted
            and terminal.worker_generation == self._worker_generation
            and self.binding_fence(self.binding)
        )
        return TrustedWorkerResultFence(
            self.binding.campaign_id, self.binding.config_digest,
            self.binding.config_generation, self.binding.supervisor_id,
            self.binding.supervisor_incarnation, terminal.worker_id,
            terminal.worker_generation, terminal.grant_id, terminal.container_id,
            terminal.lineage_id, is_current, is_current)

    def terminal_for_request(self, *, request_id: str, plan_digest: str,
                             lineage_id: str, stage_id: str) -> TerminalWorker | None:
        """Return one exact current durably emitted terminal without restoring authority."""
        request_key = (_text(request_id, "request_id"), _sha(plan_digest, "plan_digest"),
                       _text(lineage_id, "lineage_id"), _text(stage_id, "stage_id"))
        if not self.binding_fence(self.binding):
            raise LifecycleRefused("terminal lookup campaign binding is not current")
        matches = tuple(self._terminal_requests.get(request_key, ()))
        if not matches:
            return None
        current = [key for key in matches
                   if self._terminal_bindings.get(key) == self.binding]
        if len(current) != 1:
            raise LifecycleRefused("terminal lookup is ambiguous across worker generations")
        terminal = self._terminals[current[0]]
        return TerminalWorker(**terminal.__dict__)

    def trusted_held_claim_receipt(self, terminal: TerminalWorker):
        """Return provider facts only for an exactly owned durably terminal worker."""
        current = self._terminals.get((terminal.worker_id, terminal.worker_generation))
        if current != terminal:
            raise LifecycleRefused("terminal worker receipt is not owned by this lifecycle")
        try:
            return self._held_receipts[(terminal.worker_id, terminal.worker_generation)].receipt
        except KeyError as exc:
            raise WaitingAuthority("trusted provider held receipt is unavailable") from exc

    def stdout_identity_for_terminal(self, terminal: TerminalWorker) -> Mapping[str, int]:
        """Return the detached file identity captured for one exact live terminal."""
        current = self._terminals.get((terminal.worker_id, terminal.worker_generation))
        if current != terminal:
            raise LifecycleRefused("terminal worker receipt is not owned by this lifecycle")
        identity = self._stdout_identities.get(
            (terminal.worker_id, terminal.worker_generation))
        if identity is None:
            raise LifecycleRefused("retained worker stdout is unavailable")
        return dict(identity)

    def stdout_proof_for_terminal(self, terminal: TerminalWorker) -> Mapping[str, Any]:
        """Return authenticated bootstrap stdout facts for an exact live terminal."""
        current = self._terminals.get((terminal.worker_id, terminal.worker_generation))
        if current != terminal:
            raise LifecycleRefused("terminal worker receipt is not owned by this lifecycle")
        proof = self._stdout_proofs.get((terminal.worker_id, terminal.worker_generation))
        if proof is None:
            raise LifecycleRefused("retained worker stdout proof is unavailable")
        return dict(proof)

    def reconcile_acquisition(self, events: Sequence[Mapping[str, Any]],
                              lifecycle_events: Sequence[Mapping[str, Any]]) -> str:
        """Resolve one prospective acquisition without launching or repeating its stage."""
        projection = project_acquisitions(events)
        if projection.pending is None:
            self._pending_acquisition = None
            self._ownership_unresolved = bool(self._active)
            return "resolved"
        intent = validate_acquisition_transition(projection.pending)
        identity = prospective_identity_from_transition(intent)
        self._worker_generation = max(self._worker_generation, identity.worker_generation)
        self._pending_acquisition = identity
        self._ownership_unresolved = True

        matching_launch = None
        for supplied in lifecycle_events:
            row = validate_event(supplied)
            if (row["event"] == "OWNED_LAUNCH_INTENT"
                    and row["worker_id"] == identity.worker_id
                    and row["worker_generation"] == identity.worker_generation
                    and row["request_id"] == identity.request_id
                    and row["container_id"] == identity.container_id):
                matching_launch = row
                break
        if matching_launch is not None:
            try:
                grant = validate_lifecycle_handoff(intent, matching_launch)
            except LifecycleRefused as exc:
                raise ContainmentFailure("durable lifecycle handoff differs") from exc
            self._resolve_acquisition(
                identity, outcome="lifecycle_handoff",
                reason="replay found matching durable owned lifecycle", grant=grant)
            return "handoff"

        if self.provider is None:
            raise WaitingAuthority("trusted provider is unavailable for pending acquisition")
        cleanup_deadline = self.monotonic() + 5.0
        try:
            inspection = self.provider.inspect_pending(identity, cleanup_deadline)
        except BaseException as exc:
            raise ContainmentFailure("pending acquisition inspection is ambiguous") from exc
        if (self.monotonic() > cleanup_deadline
                or not isinstance(inspection, PendingAcquisitionInspection)):
            raise ContainmentFailure("pending acquisition inspection failed its deadline/type")
        inspection = PendingAcquisitionInspection(
            inspection.status, inspection.authorization, inspection.reason)
        if inspection.status == "unknown":
            raise ContainmentFailure(inspection.reason)
        if inspection.status == "absent":
            self._resolve_acquisition(
                identity, outcome="absent_released", reason=inspection.reason)
            return "resolved"

        assert inspection.authorization is not None
        authorization = AuthorizedLaunch(
            inspection.authorization.grant, inspection.authorization.container_id,
            inspection.authorization.container)
        if (authorization.container_id != identity.container_id
                or not authorization.container.path.is_absolute()
                or authorization.container.path.name != identity.container_id):
            raise ContainmentFailure("pending inspection changed preassigned container identity")
        container = authorization.container
        if os.path.lexists(container.path):
            pinned = _container_identity(container.identity(), container.path)
            _same_container(container, pinned)
            if container.populated():
                raise ContainmentFailure(
                    "pending acquisition container is unexpectedly populated")
            _same_container(container, pinned)
            container.close_and_remove()
            if os.path.lexists(container.path):
                raise ContainmentFailure("empty pending acquisition container survived removal")
        if (not self.provider.release(authorization, cleanup_deadline)
                or self.monotonic() > cleanup_deadline):
            raise ContainmentFailure("pending exact acquisition release failed")
        self._resolve_acquisition(
            identity, outcome="exact_released", reason=inspection.reason,
            grant=authorization.grant)
        return "resolved"

    def reconcile(self, events: Sequence[Mapping[str, Any]]) -> TerminalWorker | None:
        """Reconcile one stranded exact container; never repeat its stage."""
        projection = project_events(events)
        if projection.worker_generations:
            self._worker_generation = max(self._worker_generation,
                                          max(projection.worker_generations.values()))
        if not projection.active:
            self._active = False
            self._ownership_unresolved = False
            return None
        if len(projection.active) != 1:
            self._ownership_unresolved = True
            raise ContainmentFailure("multiple unresolved workers require operator reconciliation")
        worker_id, _latest = next(iter(projection.active.items()))
        rows = [validate_event(row) for row in events if row.get("worker_id") == worker_id]
        intent = next((row for row in rows if row["event"] == "OWNED_LAUNCH_INTENT"), None)
        if intent is None:
            self._ownership_unresolved = True
            raise ContainmentFailure("active worker lacks durable launch intent")
        event_binding = CampaignBinding(
            intent["campaign_id"], intent["config_digest"], intent["config_generation"],
            intent["supervisor_id"], intent["supervisor_incarnation"])
        contract = intent["data"]["contract"]
        request = StageRequest(
            intent["request_id"], intent["plan_digest"], intent["lineage_id"],
            intent["stage_id"], contract["stage"], tuple(contract["argv"]),
            contract["env"], Path(contract["cwd"]), contract["artifact_contract_digest"],
            intent["data"]["max_stage_seconds"], intent["data"]["teardown_seconds"],
            intent["control_revision"])
        old_grant = GrantReceipt(
            intent["grant_id"], intent["grant_generation"],
            intent["data"]["provider_deadline"], intent["data"]["clock_domain"])
        deadline = self.monotonic() + request.teardown_seconds
        if self.provider is None:
            self._ownership_unresolved = True
            raise WaitingAuthority("trusted provider is unavailable for recovery")
        inspection = self.provider.inspect(old_grant, intent["container_id"], deadline)
        if self.monotonic() > deadline or not isinstance(inspection, RecoveryInspection):
            self._ownership_unresolved = True
            raise ContainmentFailure("trusted recovery inspection failed its deadline/type")

        def unresolved(reason: str) -> None:
            self._emit("WORKER_UNRESOLVED", worker_id=worker_id,
                       worker_generation=intent["worker_generation"], request=request,
                       grant=old_grant, container_id=intent["container_id"],
                       data={"reason": reason}, event_binding=event_binding)
            self._active = True
            self._ownership_unresolved = True

        created = next((row for row in rows if row["event"] == "OWNED_CONTAINER_CREATED"), None)
        if inspection.status == "unknown":
            unresolved(inspection.reason)
            raise ContainmentFailure(inspection.reason)
        if inspection.status == "absent_released":
            teardown_seen = any(row["event"] == "OWNED_TEARDOWN_STARTED" for row in rows)
            captured_prior = next((row for row in rows
                                   if row["event"] == "OWNED_CHILD_CAPTURED"), None)
            if captured_prior is not None and same_process(
                    ProcessIdentity(**captured_prior["data"]["process"])):
                unresolved("captured PID remains alive while provider reports container absent")
                raise ContainmentFailure("container absence conflicts with live captured PID")
            if created is not None and not teardown_seen:
                unresolved("previously created container is now absent without terminal receipt")
                raise ContainmentFailure("created container disappeared before recovery")
            if rows[-1]["event"] != "OWNED_TEARDOWN_STARTED":
                self._emit("OWNED_TEARDOWN_STARTED", worker_id=worker_id,
                           worker_generation=intent["worker_generation"], request=request,
                           grant=old_grant, container_id=intent["container_id"],
                           data={"termination_deadline": deadline,
                                 "clock_domain": old_grant.clock_domain},
                           event_binding=event_binding)
            self._emit("OWNED_TERMINAL", worker_id=worker_id,
                       worker_generation=intent["worker_generation"], request=request,
                       grant=old_grant, container_id=intent["container_id"],
                       data={"captured_identity_dead": True, "container_empty": True,
                             "container_removed": True, "claim_released": True,
                             "publishers_joined": True}, event_binding=event_binding)
            cleanup_ok = True
            grant = old_grant
        else:
            assert inspection.authorization is not None
            authorization = inspection.authorization
            grant = authorization.grant.revalidated()
            if (authorization.container_id != intent["container_id"]
                    or (grant.grant_id, grant.generation)
                    != (old_grant.grant_id, old_grant.generation)):
                unresolved("recovery provider changed ownership identity/generation")
                raise ContainmentFailure("recovery ownership identity differs")
            if created is None and authorization.container.populated():
                unresolved("container populated before durable container/PID receipt")
                raise ContainmentFailure("unknown side effects before PID receipt")
            expected_identity = created["data"]["identity"] if created else None
            if expected_identity is None:
                expected_identity = _container_identity(
                    authorization.container.identity(), authorization.container.path)
            if expected_identity is not None:
                try:
                    _same_container(authorization.container, expected_identity)
                except ContainmentFailure as exc:
                    unresolved(str(exc))
                    raise
            captured_row = next((row for row in rows
                                 if row["event"] == "OWNED_CHILD_CAPTURED"), None)
            if captured_row is None:
                unresolved(
                    "container exists without durable PID receipt; spawn side effects are unknown")
                raise ContainmentFailure("ownership is unknown before PID receipt")
            captured = (ProcessIdentity(**captured_row["data"]["process"])
                        if captured_row else None)
            try:
                cleanup_ok = self._teardown(
                    authorization, request, worker_id, intent["worker_generation"], grant,
                    None, captured, expected_identity, True, deadline,
                    event_binding=event_binding)
            except BaseException as exc:
                unresolved(f"recovery teardown failed: {type(exc).__name__}: {exc}")
                raise ContainmentFailure("recovery teardown failed") from exc
        retained = next((row for row in rows if row["event"] == "WORKER_RESULT_RETAINED"), None)
        result_digest = retained["data"]["result_digest"] if retained else None
        return_code = retained["data"]["return_code"] if retained else None
        terminal = TerminalWorker(
            worker_id, intent["worker_generation"], request.request_id,
            request.plan_digest, request.lineage_id, request.stage_id, grant.grant_id,
            grant.generation, intent["container_id"], return_code, result_digest, False,
            "recovered old/uncertain worker result is diagnostic only")
        self._emit("WORKER_RESULT_STALE", worker_id=worker_id,
                   worker_generation=intent["worker_generation"], request=request,
                   grant=grant, container_id=intent["container_id"],
                   data={"result_digest": result_digest, "accepted": False,
                         "reason": terminal.reason}, event_binding=event_binding)
        self._register_terminal(terminal, event_binding)
        self._active = not cleanup_ok
        self._ownership_unresolved = not cleanup_ok
        return terminal

    def planned_serving_guard(self, *_args: Any, **_kwargs: Any):
        # PlannedServing's measurement callback executes in the caller process.
        # This lifecycle encloses a separate child, so claiming that callback is
        # inside the worker cgroup would be false until a worker-RPC adapter exists.
        from .planned_serving import UnsupportedContainment
        raise UnsupportedContainment(
            "planned-serving callback placement is not proven inside the owned worker")


__all__ = ["ACQUISITION_SCHEMA", "AcquisitionProjection", "AuthorizationDenied",
           "AuthorizedLaunch", "CampaignBinding", "COMMAND_RESULT_FIELDS",
           "COMMAND_RESULT_SCHEMA", "COMMAND_TRANSITION_SCHEMA", "ContainmentFailure",
           "EVENTS", "EVENT_SCHEMA", "EXPENSIVE_STAGES", "GrantReceipt", "LifecycleRefused",
           "LifecycleProjection", "OwnedContainer", "PendingAcquisitionInspection",
           "ProcessIdentity", "ProspectiveAcquisitionIdentity", "RecoveryInspection",
           "RESULT_SCHEMA", "RuntimeDirective", "SimulatedCrash", "StageAdmission",
           "StageRequest", "TerminalWorker", "TrustedGrantProvider",
           "TrustedHeldClaimReceipt",
           "WaitingAuthority", "WorkerLifecycle", "process_identity", "same_process",
           "monotonic_clock_domain", "project_acquisitions", "project_events",
           "prospective_identity_from_transition", "prospective_request_digest",
           "validate_acquisition_transition", "validate_command_result_v2",
           "validate_command_transition_v2", "validate_event",
           "validate_lifecycle_handoff"]
