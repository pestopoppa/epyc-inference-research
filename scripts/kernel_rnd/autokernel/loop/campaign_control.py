#!/usr/bin/env python3
"""Durable, fenced campaign controls; no worker or resource authority."""
from __future__ import annotations

import copy
from dataclasses import dataclass
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
from typing import Any, Callable, Mapping

from .. import journal as journal_module
from ..controller.discovery_supervisor_secure import (
    RuntimeRoot, SecureRuntimeError, object_identity,
)
from . import status
from .campaign import ResolvedCampaign

COMMAND_SCHEMA = "epyc.autokernel.campaign_command.v1"
SNAPSHOT_SCHEMA = "epyc.autokernel.campaign_snapshot.v1"
SNAPSHOT_FILE = "campaign-snapshot.json"
_CANDIDATE_REPLAYER_TOKEN = object()
OPERATIONS = frozenset({"pause", "resume", "drain"})


class ControlRefused(RuntimeError):
    """A control or supervisor transition failed closed."""


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
                 clock: Callable[[], str] = _now) -> None:
        if not isinstance(resolved, ResolvedCampaign):
            raise TypeError("resolved must be ResolvedCampaign")
        if not isinstance(config_generation, int) or isinstance(config_generation, bool) \
                or config_generation < 1:
            raise ValueError("config_generation must be positive")
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
        self.readiness_check = readiness_check or (lambda: (False, "execution authority absent"))
        self.clock = clock
        self._mutex = threading.RLock()
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
                self._replay(entries)
                self.supervisor_incarnation += 1
                self.stream_epoch += 1
                self.sequence = 0
                self._append_event("START", {
                    "desired_state": self.desired_state,
                    "observed_state": self.observed_state,
                    "prerequisite_reason": self.prerequisite_reason,
                    "lock_identity": copy.deepcopy(self._lock_identity),
                })
                self._entered = True
                self._poisoned = False
                return self
            except BaseException:
                self.close()
                raise

    def _replay(self, entries) -> None:
        last_incarnation = 0
        last_epoch = 0
        last_revision = 0
        saw_start = False
        candidate_pending: tuple[str, str, str] | None = None
        candidate_pending_payload: Mapping[str, Any] | None = None
        candidate_prepared = False
        candidate_completed: set[str] = set()
        candidate_completed_records: dict[str, dict[str, Mapping[str, Any]]] = {}
        candidate_entries = []
        for entry in entries:
            self._journal_cursor = entry.seq
            if entry.campaign_id not in (None, self.resolved.campaign_id):
                raise ControlRefused("store contains another campaign identity")
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
        self.control_revision = last_revision
        self._candidate_pending = candidate_pending
        self._candidate_pending_payload = candidate_pending_payload
        self._candidate_prepared = candidate_prepared
        self._candidate_completed = candidate_completed
        self._candidate_completed_records = candidate_completed_records
        self._candidate_entries = candidate_entries

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
        entry = self._journal.append(journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, {
            "schema": journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA,
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

    def apply_command(self, value: Mapping[str, Any]) -> dict[str, Any]:
        with self._mutex:
            self._require_active_locked()
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
            self._lifetime_token = None
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            if runtime is not None:
                runtime.close()

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = ["AdmissionDecision", "CampaignController", "COMMAND_SCHEMA",
           "ControlRefused", "SNAPSHOT_FILE", "SNAPSHOT_SCHEMA", "TrustedGrant",
           "command_digest", "may_start_stage", "resolved_config_digest",
           "validate_command"]
