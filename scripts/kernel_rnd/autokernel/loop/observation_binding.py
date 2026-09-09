#!/usr/bin/env python3
"""Versioned binding between contained planned units and lifecycle observations.

This module seals and reopens evidence.  It deliberately owns neither process
launch/admission nor protocol policy.  In particular, child-authored verifier
rows are candidates for parent verification, never parent authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat as stat_module
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from .. import schemas
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import serving


INSTRUMENT_REFERENCE_SCHEMA = "epyc.autokernel.loaded_serving_instrument_reference.v1"
UNIT_BINDING_SCHEMA = "epyc.autokernel.observation_unit_binding.v1"
OBSERVATION_REFERENCE_SCHEMA = "epyc.autokernel.lifecycle_observation_reference.v1"
OBSERVATION_LINK_SCHEMA = "epyc.autokernel.lifecycle_observation_link.v1"
SEMANTIC_PHASES = ("load", "placement", "warmup", "measurement", "teardown")
BUILTIN_PROVENANCE_SCHEMA = "epyc.autokernel.loaded_builtin_callable.v1"
_MAX_PROVIDER_BYTES = 128 * 1024 * 1024
_MAX_MAPS_BYTES = 4 * 1024 * 1024


class ObservationBindingError(RuntimeError):
    """A lifecycle observation is malformed, stale, or not parent-verified."""


@dataclass(frozen=True)
class ParentObservationConfiguration:
    """Explicit parent configuration for v2 observation contexts.

    Keys are resolved-recipe execution digests.  This is configuration, not provider
    ownership evidence; the lifecycle separately obtains and verifies the held claim.
    """

    requested_effective_states: Mapping[str, Mapping[str, Any]]
    required_gpu_dsos: Mapping[str, tuple[Mapping[str, Any], ...]]
    cadence_s: float
    gap_limit_s: float
    budgets: Mapping[str, Any]

    def __post_init__(self) -> None:
        states: dict[str, Any] = {}
        dsos: dict[str, Any] = {}
        for digest, supplied in dict(self.requested_effective_states).items():
            _sha(digest, "observation recipe digest")
            row = _closed(supplied, {"logical_cpus", "numa_nodes", "thp_mode"},
                          "requested effective state")
            states[digest] = _freeze(_plain(row))
        for digest, supplied in dict(self.required_gpu_dsos).items():
            _sha(digest, "observation GPU recipe digest")
            if not isinstance(supplied, (list, tuple)):
                raise ObservationBindingError("required GPU DSOs must be an array")
            dsos[digest] = tuple(_freeze(_plain(item)) for item in supplied)
        if set(dsos) - set(states):
            raise ObservationBindingError("GPU DSO configuration lacks a recipe state")
        cadence = float(self.cadence_s)
        gap = float(self.gap_limit_s)
        if not (cadence > 0 and gap >= cadence):
            raise ObservationBindingError("observation cadence/gap configuration is invalid")
        if not isinstance(self.budgets, Mapping) or set(self.budgets) != lo.BUDGET_FIELDS:
            raise ObservationBindingError("observation budgets are not the closed observer set")
        object.__setattr__(self, "requested_effective_states", _freeze(states))
        object.__setattr__(self, "required_gpu_dsos", _freeze(dsos))
        object.__setattr__(self, "cadence_s", cadence)
        object.__setattr__(self, "gap_limit_s", gap)
        object.__setattr__(self, "budgets", _freeze(_plain(self.budgets)))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    # Round-trip scalars to reject non-finite/non-JSON values at the boundary.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _closed(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ObservationBindingError(f"{label} has missing or unknown fields")
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ObservationBindingError(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ObservationBindingError(f"{label} must be lowercase SHA-256")
    return value


def _artifact(value: Any, label: str) -> mc.StoredArtifact:
    row = _closed(value, {"locator", "sha256", "verified"}, label)
    if row["verified"] is not True:
        raise ObservationBindingError(f"{label} must be a verified sealed reference")
    return mc.StoredArtifact(_text(row["locator"], f"{label}.locator"),
                             _sha(row["sha256"], f"{label}.sha256"), True)


def _hashed(row: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    body = _plain(row)
    return body | {digest_field: schemas.content_hash(body)}


def _verify_hash(row: dict[str, Any], digest_field: str, label: str) -> dict[str, Any]:
    supplied = _sha(row.pop(digest_field), f"{label}.{digest_field}")
    if schemas.content_hash(row) != supplied:
        raise ObservationBindingError(f"{label} digest mismatch")
    return row


def _artifact_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    opened = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(opened)
        if (not stat_module.S_ISREG(before.st_mode)
                or before.st_size > _MAX_PROVIDER_BYTES):
            raise ObservationBindingError("loaded provider artifact is not a bounded file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining > 0:
            chunk = os.read(opened, min(1024 * 1024, remaining))
            if not chunk:
                raise ObservationBindingError("loaded provider artifact changed while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(opened, 1):
            raise ObservationBindingError("loaded provider artifact changed while hashing")
        after = os.fstat(opened)
        path_after = os.stat(path, follow_symlinks=False)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid",
                         "st_size", "st_mtime_ns", "st_ctime_ns")
        if (any(getattr(before, field) != getattr(after, field)
                for field in stable_fields)
                or (path_after.st_dev, path_after.st_ino)
                   != (after.st_dev, after.st_ino)):
            raise ObservationBindingError("loaded provider artifact changed while hashing")
    finally:
        os.close(opened)
    return {"path": str(path), "dev": after.st_dev, "ino": after.st_ino,
            "size": after.st_size, "sha256": digest.hexdigest()}


def _mapped_executable_provider(pointer: int) -> dict[str, Any]:
    opened = -1
    try:
        opened = os.open("/proc/self/maps", os.O_RDONLY | os.O_NONBLOCK)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(opened, min(65536, _MAX_MAPS_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > _MAX_MAPS_BYTES:
                raise ObservationBindingError("loaded provider maps exceed byte bound")
        raw = b"".join(chunks)
    except OSError as exc:
        raise ObservationBindingError("loaded provider maps are unavailable") from exc
    finally:
        if opened >= 0:
            os.close(opened)
    selected = None
    for encoded in raw.splitlines():
        fields = encoded.decode("utf-8", "strict").split(maxsplit=5)
        if len(fields) < 5:
            continue
        limits, permissions, _, device, inode = fields[:5]
        try:
            lower, upper = (int(item, 16) for item in limits.split("-", 1))
        except ValueError:
            continue
        if lower <= pointer < upper:
            if "x" not in permissions or len(fields) != 6 or not fields[5].startswith("/"):
                raise ObservationBindingError("builtin function is not in a named executable map")
            major, minor = (int(item, 16) for item in device.split(":", 1))
            if fields[5].endswith(" (deleted)"):
                raise ObservationBindingError("builtin provider mapping was deleted")
            selected = {"path": fields[5],
                        "dev": os.makedev(major, minor), "ino": int(inode)}
            break
    if selected is None:
        raise ObservationBindingError("builtin function provider mapping was not found")
    artifact = _artifact_identity(Path(selected["path"]))
    if (artifact["dev"], artifact["ino"]) != (selected["dev"], selected["ino"]):
        raise ObservationBindingError("loaded provider mapping identity changed")
    return artifact


def _loaded_builtin_identity(value: Callable[..., Any], *, clock_name: str) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    """Pin an actual loaded CPython builtin and its clock/runtime provenance."""
    if sys.implementation.name != "cpython":
        raise ObservationBindingError("builtin callable provenance currently requires CPython")
    getter = ctypes.pythonapi.PyCFunction_GetFunction
    getter.argtypes = [ctypes.py_object]
    getter.restype = ctypes.c_void_p
    pointer = getter(value)
    if not pointer:
        raise ObservationBindingError("selected clock is not a CPython C function")
    provider = _mapped_executable_provider(int(pointer))
    interpreter = _artifact_identity(Path("/proc/self/exe"))
    import time as time_module
    info = time_module.get_clock_info(clock_name)
    provenance = {"schema": BUILTIN_PROVENANCE_SCHEMA,
        "module": value.__module__, "qualname": value.__qualname__,
        "c_api": "PyCFunction_GetFunction",
        "provider_artifact": provider, "interpreter_artifact": interpreter,
        "runtime": {"implementation": sys.implementation.name,
                    "version": list(sys.version_info[:5]),
                    "cache_tag": sys.implementation.cache_tag,
                    "byteorder": sys.byteorder},
        "clock": {"name": clock_name, "implementation": info.implementation,
                  "monotonic": info.monotonic, "adjustable": info.adjustable,
                  "resolution": info.resolution}}
    implementation = {key: provenance[key] for key in (
        "schema", "module", "qualname", "c_api", "provider_artifact")}
    configuration = {key: provenance[key] for key in (
        "interpreter_artifact", "runtime", "clock")}
    identity = {"module": value.__module__, "qualname": value.__qualname__,
                "kind": "builtin_or_extension", "implementation_status": "pinned",
                "implementation_sha256": schemas.content_hash(implementation),
                "configuration_status": "pinned",
                "configuration_sha256": schemas.content_hash(configuration)}
    return identity, provenance


@dataclass(frozen=True)
class LoadedInstrumentReference:
    identity_sha256: str
    configuration_complete: bool
    artifact: mc.StoredArtifact
    schema: str = INSTRUMENT_REFERENCE_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "LoadedInstrumentReference":
        row = _verify_hash(_closed(value, {"schema", "identity_sha256",
            "configuration_complete", "artifact", "reference_digest"},
            "instrument reference"), "reference_digest", "instrument reference")
        if row["schema"] != INSTRUMENT_REFERENCE_SCHEMA \
                or not isinstance(row["configuration_complete"], bool):
            raise ObservationBindingError("instrument reference schema/state is invalid")
        return cls(_sha(row["identity_sha256"], "instrument identity"),
                   row["configuration_complete"], _artifact(row["artifact"], "instrument artifact"))

    def to_dict(self) -> dict[str, Any]:
        return _hashed({"schema": self.schema, "identity_sha256": self.identity_sha256,
                        "configuration_complete": self.configuration_complete,
                        "artifact": self.artifact.to_dict()}, "reference_digest")


@dataclass(frozen=True)
class ObservationUnitBinding:
    observation_id: str
    unit_id: str
    process_generation_id: str
    fence_id: str
    clock_domain: str
    boot_id: str
    worker_binding: Mapping[str, Any]
    container_id: str
    active_claim_ref: str
    held_claim: Mapping[str, Any]
    requested_effective_state: Mapping[str, Any]
    runtime_witness_keys: tuple[str, ...]
    required_gpu_dsos: tuple[Mapping[str, Any], ...]
    cadence_s: float
    gap_limit_s: float
    budgets: Mapping[str, Any]
    instrument: LoadedInstrumentReference
    schema: str = UNIT_BINDING_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "ObservationUnitBinding":
        fields = {"schema", "observation_id", "unit_id", "process_generation_id",
                  "fence_id", "clock_domain", "boot_id", "worker_binding",
                  "container_id", "active_claim_ref", "held_claim", "requested_effective_state",
                  "runtime_witness_keys", "required_gpu_dsos", "cadence_s",
                  "gap_limit_s", "budgets", "instrument", "binding_digest"}
        row = _verify_hash(_closed(value, fields, "observation unit binding"),
                           "binding_digest", "observation unit binding")
        if row["schema"] != UNIT_BINDING_SCHEMA:
            raise ObservationBindingError("observation unit binding schema is unsupported")
        instrument = LoadedInstrumentReference.from_dict(row["instrument"])
        held = row["held_claim"]
        if not isinstance(held, Mapping) or set(held) != {"logical_cpus", "gpu_devices"}:
            raise ObservationBindingError("held_claim has missing or unknown fields")
        backend = "gpu" if held["gpu_devices"] else "cpu"
        context = lo.validate_context({"schema": lo.CONTEXT_SCHEMA,
            "observation_id": row["observation_id"], "backend": backend,
            "instrument_identity_digest": instrument.identity_sha256,
            "recipe_identity_digest": "0" * 64, "clock_domain": row["clock_domain"],
            "cadence_s": row["cadence_s"], "gap_limit_s": row["gap_limit_s"],
            "boot_id": row["boot_id"], "worker_binding": row["worker_binding"],
            "requested_effective_state": row["requested_effective_state"],
            "held_claim": {key: row["held_claim"][key] for key in ("logical_cpus", "gpu_devices")},
            "runtime_witness_keys": row["runtime_witness_keys"],
            "required_gpu_dsos": row["required_gpu_dsos"], "budgets": row["budgets"]})
        return cls(*(_text(row[key], key) for key in ("observation_id", "unit_id",
                   "process_generation_id", "fence_id", "clock_domain", "boot_id")),
                   _freeze(context["worker_binding"]), _text(row["container_id"], "container_id"),
                   _text(row["active_claim_ref"], "active_claim_ref"),
                   _freeze(row["held_claim"]), _freeze(context["requested_effective_state"]),
                   tuple(context["runtime_witness_keys"]), tuple(_freeze(context["required_gpu_dsos"])),
                   context["cadence_s"], context["gap_limit_s"], _freeze(context["budgets"]), instrument)

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": self.schema, "observation_id": self.observation_id,
                "unit_id": self.unit_id, "process_generation_id": self.process_generation_id,
                "fence_id": self.fence_id, "clock_domain": self.clock_domain,
                "boot_id": self.boot_id, "worker_binding": _plain(self.worker_binding),
                "container_id": self.container_id,
                "active_claim_ref": self.active_claim_ref, "held_claim": _plain(self.held_claim),
                "requested_effective_state": _plain(self.requested_effective_state),
                "runtime_witness_keys": list(self.runtime_witness_keys),
                "required_gpu_dsos": _plain(self.required_gpu_dsos),
                "cadence_s": self.cadence_s, "gap_limit_s": self.gap_limit_s,
                "budgets": _plain(self.budgets), "instrument": self.instrument.to_dict()}
        return _hashed(body, "binding_digest")


@dataclass(frozen=True)
class LifecycleObservationReference:
    observation_id: str
    unit_id: str
    process_generation_id: str
    fence_id: str
    active_claim_ref: str
    target_pid: int
    target_start_ticks: int
    descendant_binding_ref: str
    worker_id: str
    worker_generation: int
    grant_id: str
    grant_generation: int
    container_id: str
    instrument_identity_sha256: str
    observation_content_sha256: str
    shutdown_status: str
    successor_permitted: bool
    artifact: mc.StoredArtifact
    schema: str = OBSERVATION_REFERENCE_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "LifecycleObservationReference":
        fields = {"schema", "observation_id", "unit_id", "process_generation_id",
                  "fence_id", "active_claim_ref", "target_pid", "target_start_ticks",
                  "descendant_binding_ref",
                  "worker_id", "worker_generation", "grant_id", "grant_generation",
                  "container_id", "instrument_identity_sha256", "observation_content_sha256",
                  "shutdown_status", "successor_permitted", "artifact", "reference_digest"}
        row = _verify_hash(_closed(value, fields, "observation reference"),
                           "reference_digest", "observation reference")
        if row["schema"] != OBSERVATION_REFERENCE_SCHEMA:
            raise ObservationBindingError("observation reference schema is unsupported")
        for key in ("target_pid", "target_start_ticks", "worker_generation", "grant_generation"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 1:
                raise ObservationBindingError(f"{key} must be a positive integer")
        if row["shutdown_status"] not in {"resolved", "unresolved"} \
                or row["successor_permitted"] is not (row["shutdown_status"] == "resolved"):
            raise ObservationBindingError("observation shutdown fence is inconsistent")
        return cls(*(_text(row[key], key) for key in ("observation_id", "unit_id",
                   "process_generation_id", "fence_id", "active_claim_ref")),
                   row["target_pid"], row["target_start_ticks"],
                   _text(row["descendant_binding_ref"], "descendant_binding_ref"),
                   _text(row["worker_id"], "worker_id"), row["worker_generation"],
                   _text(row["grant_id"], "grant_id"), row["grant_generation"],
                   _text(row["container_id"], "container_id"),
                   _sha(row["instrument_identity_sha256"], "instrument identity"),
                   _sha(row["observation_content_sha256"], "observation content"),
                   row["shutdown_status"], row["successor_permitted"],
                   _artifact(row["artifact"], "observation artifact"))

    def to_dict(self) -> dict[str, Any]:
        body = {key: getattr(self, key) for key in (
            "schema", "observation_id", "unit_id", "process_generation_id", "worker_id",
            "fence_id", "active_claim_ref", "target_pid", "target_start_ticks",
            "descendant_binding_ref",
            "worker_generation", "grant_id", "grant_generation", "container_id",
            "instrument_identity_sha256", "observation_content_sha256", "shutdown_status",
            "successor_permitted")}
        body["artifact"] = self.artifact.to_dict()
        return _hashed(body, "reference_digest")


class ParentEvidenceVerifier(Protocol):
    def __call__(self, kind: str, candidate: Mapping[str, Any], *,
                 observation: Mapping[str, Any], expected: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ContainedObservationAuthority(Protocol):
    """Child-side view of the inherited, bounded parent authority socket."""

    def observation_binding(self, *, sequence: int, unit: Any, fence: Any,
                            recipe_identity_digest: str
                            ) -> ObservationUnitBinding: ...

    def observation_target(self, *, sequence: int, unit: Any, fence: Any,
                           pid: int) -> Mapping[str, Any]: ...


class ContainedObservationFactory:
    """Create and seal exactly one whole-lifecycle observer per fixed unit."""

    def __init__(self, *, authority: ContainedObservationAuthority,
                 store: mc.ArtifactStore, instrument: LoadedInstrumentReference,
                 probe: lo.FilesystemProbe,
                 foreign_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 runtime_verifier: Callable[[str, Mapping[str, Any] | None],
                                            Mapping[str, Any]] | None = None,
                 monotonic: Callable[[], float], wall_clock: Callable[[], str],
                 max_units: int) -> None:
        if (not isinstance(store, mc.ArtifactStore)
                or not isinstance(instrument, LoadedInstrumentReference)
                or not isinstance(probe, lo.FilesystemProbe)
                or not callable(monotonic) or not callable(wall_clock)
                or isinstance(max_units, bool) or not 1 <= max_units <= 1024):
            raise ObservationBindingError("contained observer factory inputs are invalid")
        for name in ("observation_binding", "observation_target"):
            if not callable(getattr(authority, name, None)):
                raise ObservationBindingError("contained observer authority is incomplete")
        self.authority = authority
        self.store = store
        self.instrument = LoadedInstrumentReference.from_dict(instrument.to_dict())
        self.probe = probe
        self.foreign_verifier = foreign_verifier
        self.runtime_verifier = runtime_verifier
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.max_units = max_units
        self._sessions: dict[str, tuple[Any, ObservationUnitBinding]] = {}
        self._references: dict[str, LifecycleObservationReference] = {}

    def create(self, *, unit: Any, fence: Any, recipe: Any) -> lo.ObservationSession:
        unit_id = _text(getattr(unit, "unit_id", None), "unit_id")
        if unit_id in self._sessions or len(self._sessions) >= self.max_units:
            raise ObservationBindingError("contained observer unit is duplicate or over bound")
        process_id = _text(getattr(unit, "process_id", None), "process_generation_id")
        order_index = getattr(unit, "order_index", None)
        recipe_digest = _sha(getattr(recipe, "execution_digest", None),
                             "recipe identity digest")
        if isinstance(order_index, bool) or not isinstance(order_index, int) or order_index < 0:
            raise ObservationBindingError("unit order index is invalid")
        binding = self.authority.observation_binding(
            sequence=order_index + 1, unit=unit, fence=fence,
            recipe_identity_digest=recipe_digest)
        if not isinstance(binding, ObservationUnitBinding):
            raise ObservationBindingError("parent returned an untyped observation binding")
        binding = ObservationUnitBinding.from_dict(binding.to_dict())
        if (binding.unit_id != unit_id or binding.process_generation_id != process_id
                or binding.fence_id != getattr(fence, "fence_id", None)
                or binding.instrument != self.instrument):
            raise ObservationBindingError("parent observation binding differs from fixed unit")
        context = {"schema": lo.CONTEXT_SCHEMA,
            "observation_id": binding.observation_id,
            "backend": "gpu" if binding.held_claim["gpu_devices"] else "cpu",
            "instrument_identity_digest": self.instrument.identity_sha256,
            "recipe_identity_digest": recipe_digest,
            "clock_domain": binding.clock_domain, "cadence_s": binding.cadence_s,
            "gap_limit_s": binding.gap_limit_s, "boot_id": binding.boot_id,
            "worker_binding": _plain(binding.worker_binding),
            "requested_effective_state": _plain(binding.requested_effective_state),
            "held_claim": _plain(binding.held_claim),
            "runtime_witness_keys": list(binding.runtime_witness_keys),
            "required_gpu_dsos": _plain(binding.required_gpu_dsos),
            "budgets": _plain(binding.budgets)}

        def resolve(pid: int) -> Mapping[str, Any]:
            return self.authority.observation_target(
                sequence=order_index + 1, unit=unit, fence=fence, pid=pid)

        session = lo.ObservationSession(
            context, probe=self.probe, owned_identity_resolver=resolve,
            foreign_verifier=self.foreign_verifier,
            runtime_verifier=self.runtime_verifier, monotonic=self.monotonic,
            wall_clock=self.wall_clock)
        self._sessions[unit_id] = (session, binding)
        return session

    def finish_reference(self, *, unit: Any,
                         session: lo.ObservationSession) -> Mapping[str, Any]:
        unit_id = _text(getattr(unit, "unit_id", None), "unit_id")
        owned = self._sessions.get(unit_id)
        if owned is None or owned[0] is not session:
            raise ObservationBindingError("observation session is not owned by this unit")
        prior = self._references.get(unit_id)
        if prior is None:
            prior = seal_observation(store=self.store, binding=owned[1],
                                     record=session.record())
            self._references[unit_id] = prior
        return prior.to_dict()


@dataclass(frozen=True)
class ParentObservationVerifiers:
    observation: ParentEvidenceVerifier | None = None
    purpose: ParentEvidenceVerifier | None = None
    runtime: ParentEvidenceVerifier | None = None
    gpu: ParentEvidenceVerifier | None = None


@dataclass(frozen=True)
class ValidatedObservationLink:
    reference: LifecycleObservationReference
    phase_facts: Mapping[str, Any]
    observation_status: str
    purpose_status: str
    runtime_status: str
    gpu_status: str
    schema: str = OBSERVATION_LINK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": self.schema, "reference": self.reference.to_dict(),
                "phase_facts": _plain(self.phase_facts),
                "observation_status": self.observation_status,
                "purpose_status": self.purpose_status, "runtime_status": self.runtime_status,
                "gpu_status": self.gpu_status}
        return _hashed(body, "link_digest")


def loaded_planned_serving_identity(*, measurement_callable: Callable[..., Any],
                                    fence_clock: Callable[..., Any],
                                    serving_timer: Callable[..., Any]) -> Mapping[str, Any]:
    """Identify the actual selected measurement/timers and enumerated direct support."""
    if (getattr(fence_clock, "__module__", None),
            getattr(fence_clock, "__qualname__", None)) != ("time", "monotonic"):
        raise ObservationBindingError(
            "supported production fence clock requires loaded time.monotonic provenance")
    if (getattr(serving_timer, "__module__", None),
            getattr(serving_timer, "__qualname__", None)) != ("time", "time"):
        raise ObservationBindingError(
            "supported production serving timer requires loaded time.time provenance")
    clock_identity, clock_provenance = _loaded_builtin_identity(
        fence_clock, clock_name="monotonic")
    timer_identity, timer_provenance = _loaded_builtin_identity(
        serving_timer, clock_name="time")
    base = _plain(lo.loaded_instrument_identity(
        measurement_callable=measurement_callable, clock_callable=fence_clock,
        supporting_callables=(serving_timer, serving.verify_env_readback,
            serving.covers_request_phase, serving._residency_record,
            serving._refuse_if_not_resident, lo.ObservationSession.start,
            lo.ObservationSession.attach_target, lo.ObservationSession.phase,
            lo.ObservationSession.checkpoint, lo.ObservationSession.finish,
            lo.ObservationSession.reconcile_shutdown, lo.FilesystemProbe.capture),
        used_constants={"serving_residency_schema": serving.RESIDENCY_SCHEMA,
            "observer_context_schema": lo.CONTEXT_SCHEMA,
            "observer_sample_schema": lo.SAMPLE_SCHEMA,
            "observer_record_schema": lo.OBSERVATION_SCHEMA,
            "observer_instrument_schema": lo.INSTRUMENT_SCHEMA,
            "detector_version": lo.DETECTOR_VERSION, "phases": list(lo.PHASES),
            "budget_fields": sorted(lo.BUDGET_FIELDS),
            "builtin_callable_provenance": {
                "fence_clock": clock_provenance,
                "serving_timer": timer_provenance}}, dependency_packages=()))
    base["clock_callable"] = clock_identity
    base["supporting_callables"][0] = timer_identity
    rows = [base["measurement_callable"], base["clock_callable"],
            *base["supporting_callables"]]
    base["configuration_complete"] = all(base["dependency_versions"].values()) and all(
        row["implementation_status"] == "pinned"
        and row["configuration_status"] == "pinned" for row in rows)
    base.pop("sha256")
    base["sha256"] = schemas.content_hash(base)
    return _freeze(lo.validate_instrument_identity(base))


def seal_loaded_instrument(*, store: mc.ArtifactStore,
                           measurement_callable: Callable[..., Any],
                           fence_clock: Callable[..., Any],
                           serving_timer: Callable[..., Any]) -> LoadedInstrumentReference:
    identity = _plain(loaded_planned_serving_identity(
        measurement_callable=measurement_callable, fence_clock=fence_clock,
        serving_timer=serving_timer))
    validated = lo.validate_instrument_identity(identity)
    artifact = store.write(f"loaded-instrument:{validated['sha256']}", validated)
    return LoadedInstrumentReference(validated["sha256"],
                                     validated["configuration_complete"], artifact)


def seal_observation(*, store: mc.ArtifactStore, binding: ObservationUnitBinding,
                     record: Mapping[str, Any]) -> LifecycleObservationReference:
    record = lo.validate_observation(record)
    worker = record["worker_binding"]
    expected = binding.worker_binding
    if (record["observation_id"] != binding.observation_id
            or record["instrument_identity_digest"] != binding.instrument.identity_sha256
            or record["boot_id"] != binding.boot_id
            or worker != _plain(expected)
            or record["requested_effective_state"] !=
               _plain(binding.requested_effective_state)
            or {key: record["held_claim"][key] for key in ("logical_cpus", "gpu_devices")}
               != _plain(binding.held_claim)):
        raise ObservationBindingError("observation differs from its fixed unit binding")
    shutdown = record["shutdown"]
    target = record["target_binding"]
    if target is None:
        raise ObservationBindingError("observation has no parent-resolved target binding")
    artifact = store.write(f"lifecycle-observation:{binding.observation_id}", record)
    return LifecycleObservationReference(
        binding.observation_id, binding.unit_id, binding.process_generation_id,
        binding.fence_id, binding.active_claim_ref, target["pid"], target["start_ticks"],
        target["binding_ref"],
        worker["worker_id"], worker["worker_incarnation"], worker["grant_id"],
        worker["grant_generation"], binding.container_id,
        binding.instrument.identity_sha256, record["content_sha256"], shutdown["status"],
        shutdown["successor_permitted"], artifact)


def _parent_status(verifier: ParentEvidenceVerifier | None, kind: str, candidate: Mapping[str, Any],
                   observation: Mapping[str, Any], expected: Mapping[str, Any]
                   ) -> tuple[str, str | None, str | None]:
    if verifier is None:
        return "unknown", None, None
    try:
        result = verifier(kind, _plain(candidate), observation=_plain(observation),
                          expected=_plain(expected))
    except Exception:
        return "unknown", None, None
    if not isinstance(result, Mapping) or set(result) != {"status", "kind", "evidence_ref"} \
            or result["status"] not in {"verified", "unknown"}:
        return "unknown", None, None
    ref = result["evidence_ref"]
    verified_kind = result["kind"]
    if (result["status"] == "verified"
            and (not isinstance(ref, str) or not ref.strip()
                 or not isinstance(verified_kind, str) or not verified_kind.strip())):
        return "unknown", None, None
    return (result["status"], verified_kind if isinstance(verified_kind, str) else None,
            ref if isinstance(ref, str) else None)


def validate_reopened_observation(reference: LifecycleObservationReference, *,
                                  store: mc.ArtifactStore,
                                  expected: Mapping[str, Any],
                                  instrument: LoadedInstrumentReference,
                                  verifiers: ParentObservationVerifiers = ParentObservationVerifiers()
                                  ) -> ValidatedObservationLink:
    """Reopen immutable bytes, then derive only parent-verified per-phase facts."""
    reference = LifecycleObservationReference.from_dict(reference.to_dict())
    instrument = LoadedInstrumentReference.from_dict(instrument.to_dict())
    identity = lo.validate_instrument_identity(_plain(store.read(
        instrument.artifact.locator, instrument.artifact.sha256)))
    store.verify(f"loaded-instrument:{identity['sha256']}", identity)
    observation = lo.validate_observation(_plain(store.read(
        reference.artifact.locator, reference.artifact.sha256)))
    store.verify(f"lifecycle-observation:{reference.observation_id}", observation)
    worker = observation["worker_binding"]
    capture = expected.get("capture_context", {})
    checks = (identity["sha256"] == instrument.identity_sha256
              == reference.instrument_identity_sha256,
              observation["content_sha256"] == reference.observation_content_sha256,
              observation["observation_id"] == reference.observation_id,
              expected.get("fence_id") == reference.fence_id,
              expected.get("active_claim_ref") == reference.active_claim_ref,
              observation["target_binding"] is not None,
              observation["target_binding"]["pid"] == reference.target_pid,
              observation["target_binding"]["start_ticks"] == reference.target_start_ticks,
              observation["target_binding"]["binding_ref"] == reference.descendant_binding_ref,
              worker["worker_id"] == reference.worker_id,
              worker["worker_incarnation"] == reference.worker_generation,
              worker["grant_id"] == reference.grant_id,
              worker["grant_generation"] == reference.grant_generation,
              expected.get("unit_id") == reference.unit_id,
              expected.get("process_generation_id") == reference.process_generation_id,
              expected.get("container_id") == reference.container_id,
              capture.get("worker_id") == reference.worker_id,
              capture.get("worker_incarnation") == reference.worker_generation,
              capture.get("grant_id") == reference.grant_id,
              observation["shutdown"]["status"] == reference.shutdown_status,
              observation["shutdown"]["successor_permitted"] == reference.successor_permitted)
    if not all(checks):
        raise ObservationBindingError("reopened observation identity/fence mismatch")
    observation_status, _, _ = _parent_status(
        verifiers.observation, "observation", {}, observation, expected)
    phase_facts: dict[str, Any] = {}
    purpose_states: list[str] = []
    for phase in SEMANTIC_PHASES:
        facts = {"inference": [], "ordinary": [], "unknown": [], "interval_statuses": []}
        for interval in observation["intervals"]:
            if interval["phase"] != phase:
                continue
            facts["interval_statuses"].append(interval["status"])
            for overlap in interval["potential_foreign_overlap"]:
                status, verified_kind, ref = _parent_status(
                    verifiers.purpose, "purpose", {**overlap, "phase": phase},
                    observation, expected)
                purpose_states.append(status)
                bucket = ("inference" if status == "verified"
                          and verified_kind == "model_inference" else
                          "ordinary" if status == "verified"
                          and verified_kind == "ordinary" else "unknown")
                facts[bucket].append({"pid": overlap["pid"],
                    "start_ticks": overlap["start_ticks"], "evidence_ref": ref,
                    "process_total_cpu_tick_delta": overlap["process_total_cpu_tick_delta"],
                    "potential_physical_claim_overlap": overlap["potential_physical_claim_overlap"]})
        phase_facts[phase] = facts
    runtime_candidates = [row for sample in observation["samples"]
                          for row in sample["runtime_witnesses"]]
    runtime_states = [_parent_status(verifiers.runtime, "runtime", row, observation, expected)[0]
                      for row in runtime_candidates] or ["unknown"]
    gpu_candidates = [sample["gpu"]["target_attribution"] for sample in observation["samples"]
                      if sample["gpu"] is not None]
    gpu_states = [_parent_status(verifiers.gpu, "gpu", row, observation, expected)[0]
                  for row in gpu_candidates] or ["unknown"]
    def summarize(states: list[str]) -> str:
        return "verified" if states and all(x == "verified" for x in states) else "unknown"
    return ValidatedObservationLink(reference, _freeze(phase_facts), observation_status,
                                    summarize(purpose_states or ["unknown"]),
                                    summarize(runtime_states), summarize(gpu_states))


__all__ = ["ContainedObservationAuthority", "ContainedObservationFactory",
           "INSTRUMENT_REFERENCE_SCHEMA", "LifecycleObservationReference",
           "LoadedInstrumentReference", "OBSERVATION_LINK_SCHEMA",
           "OBSERVATION_REFERENCE_SCHEMA", "ObservationBindingError",
           "ObservationUnitBinding", "ParentEvidenceVerifier",
           "ParentObservationConfiguration",
           "ParentObservationVerifiers", "SEMANTIC_PHASES", "UNIT_BINDING_SCHEMA",
           "ValidatedObservationLink", "loaded_planned_serving_identity",
           "seal_loaded_instrument", "seal_observation", "validate_reopened_observation"]
