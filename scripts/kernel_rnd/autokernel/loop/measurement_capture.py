"""Prospective, per-arm evidence capture for :mod:`planned_serving`.

This module records facts; it does not decide eligibility, statistical significance, or
promotion.  Journal idempotence is supplied by the controller's serialized transaction
boundary, never by inspecting the artifact directory.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import median
import sys
import threading
from types import MappingProxyType
from typing import Any

from .. import schemas
from ..controller.discovery_supervisor_secure import RuntimeRoot, SecureRuntimeError
from . import experiment_plan as ep


CAPTURE_SCHEMA = "epyc.autokernel.unified_arm_capture.v1"
JOURNAL_KIND = "PLANNED_SERVING_ARM_CAPTURED"
PRODUCER_ID = "epyc.autokernel.measurement_capture/v1"
_ARMS = frozenset({"anchor", "candidate"})
_PROTOCOL_STATUSES = frozenset({"ratified", "unratified", "unknown"})


class CaptureError(RuntimeError):
    """A prospective capture is malformed, conflicting, or cannot be sealed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureError(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise CaptureError(f"{label} must be lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> str:
    value = _text(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CaptureError(f"{label} must include a timezone")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise CaptureError("objects must have non-empty text keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CaptureError("values must be finite canonical JSON data")


def _frozen_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CaptureError(f"{label} must be a non-empty object")
    copied = json.loads(json.dumps(_plain(value)))
    if not isinstance(copied, dict) or any(not isinstance(key, str) or not key for key in copied):
        raise CaptureError(f"{label} must have non-empty text keys")
    return _freeze(copied)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CaptureContext:
    campaign_id: str
    config_digest: str
    supervisor_id: str
    supervisor_incarnation: int
    config_generation: int
    worker_id: str
    worker_incarnation: int
    grant_id: str
    container_id: str
    lineage_id: str
    instrument_id: str
    protocol_id: str | None
    protocol_status: str
    source_identities: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaptureContext":
        fields = {"campaign_id", "config_digest", "supervisor_id",
                  "supervisor_incarnation", "config_generation", "worker_id",
                  "worker_incarnation", "grant_id", "container_id", "lineage_id",
                  "instrument_id", "protocol_id", "protocol_status",
                  "source_identities"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CaptureError("capture context has missing or unknown fields")
        sources = _frozen_mapping(value["source_identities"], "source_identities")
        if set(sources) != _ARMS or any(not isinstance(sources[arm], Mapping)
                                        for arm in _ARMS):
            raise CaptureError("source_identities must contain anchor and candidate objects")
        for arm in _ARMS:
            source = sources[arm]
            if set(source) != {"source_revision", "model_sha256", "build_sha256",
                              "recipe_hash"}:
                raise CaptureError(f"{arm} source identity has missing or unknown fields")
            revision = source["source_revision"]
            if (not isinstance(revision, str) or len(revision) not in {40, 64}
                    or any(char not in "0123456789abcdef" for char in revision)):
                raise CaptureError(f"{arm} source_revision must be a Git object ID")
            for name in ("model_sha256", "build_sha256", "recipe_hash"):
                _sha(source[name], f"{arm}.{name}")
        protocol = value["protocol_id"]
        if protocol is not None:
            protocol = _text(protocol, "protocol_id")
        protocol_status = value["protocol_status"]
        if protocol_status not in _PROTOCOL_STATUSES:
            raise CaptureError("protocol_status must be ratified, unratified, or unknown")
        if protocol_status == "ratified" and protocol is None:
            raise CaptureError("ratified protocol status requires protocol_id")
        for name in ("supervisor_incarnation", "config_generation", "worker_incarnation"):
            if isinstance(value[name], bool) or not isinstance(value[name], int) \
                    or value[name] <= 0:
                raise CaptureError(f"{name} must be a positive integer")
        return cls(_text(value["campaign_id"], "campaign_id"),
                   _sha(value["config_digest"], "config_digest"),
                   _text(value["supervisor_id"], "supervisor_id"),
                   value["supervisor_incarnation"], value["config_generation"],
                   _text(value["worker_id"], "worker_id"),
                   value["worker_incarnation"], _text(value["grant_id"], "grant_id"),
                   _text(value["container_id"], "container_id"),
                   _text(value["lineage_id"], "lineage_id"),
                   _text(value["instrument_id"], "instrument_id"),
                   protocol, protocol_status, sources)

    def to_dict(self) -> dict[str, Any]:
        return {"campaign_id": self.campaign_id, "config_digest": self.config_digest,
                "supervisor_id": self.supervisor_id,
                "supervisor_incarnation": self.supervisor_incarnation,
                "config_generation": self.config_generation,
                "worker_id": self.worker_id, "worker_incarnation": self.worker_incarnation,
                "grant_id": self.grant_id, "container_id": self.container_id,
                "lineage_id": self.lineage_id,
                "instrument_id": self.instrument_id, "protocol_id": self.protocol_id,
                "protocol_status": self.protocol_status,
                "source_identities": _plain(self.source_identities)}


@dataclass(frozen=True)
class StoredArtifact:
    locator: str
    sha256: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {"locator": self.locator, "sha256": self.sha256,
                "verified": self.verified}


class ArtifactStore:
    """Small content-addressed store rooted at an explicit caller-owned directory."""

    def __init__(self, root: Path):
        requested = Path(root).absolute()
        if not requested.exists() and not requested.parent.is_dir():
            raise CaptureError("artifact root parent must already exist for durable creation")
        runtime: RuntimeRoot | None = None
        try:
            runtime = RuntimeRoot.create_or_open(requested)
            # Always fsync the parent. A prior constructor may have created the root and then
            # failed before the parent fsync; existence alone does not prove durable creation.
            parent = os.open(requested.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                             | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except (OSError, SecureRuntimeError) as exc:
            if runtime is not None:
                runtime.close()
            raise CaptureError(f"unsafe artifact root: {exc}") from exc
        self._runtime = runtime
        self.root = self._runtime.path
        self._thread_lock = threading.RLock()
        self._thread_state = threading.local()

    @contextmanager
    def _exclusive(self):
        with self._thread_lock:
            if self._runtime.fd < 0:
                raise CaptureError("artifact store is closed")
            depth = getattr(self._thread_state, "exclusive_depth", 0)
            if depth == 0:
                fcntl.flock(self._runtime.fd, fcntl.LOCK_EX)
            self._thread_state.exclusive_depth = depth + 1
            try:
                self._runtime.verify()
                yield
            finally:
                remaining = self._thread_state.exclusive_depth - 1
                self._thread_state.exclusive_depth = remaining
                if remaining == 0:
                    fcntl.flock(self._runtime.fd, fcntl.LOCK_UN)

    @contextmanager
    def exclusive(self):
        """Hold this store instance's verified, reentrant publication exclusion.

        The capability exists only for the lifetime of this context and this open
        ``ArtifactStore`` instance.  It is not a transferable root authority.  Public
        methods such as :meth:`write` and :meth:`verify` may be called inside it; their
        nested acquisition is intentionally reentrant on the current thread.
        """
        with self._exclusive():
            yield

    def write(self, namespace: str, body: Mapping[str, Any]) -> StoredArtifact:
        with self._exclusive():
            return self._write_locked(namespace, body)

    def _write_locked(self, namespace: str, body: Mapping[str, Any]) -> StoredArtifact:
        name, encoded, raw_body = self._identity(namespace, body)
        temporary = f".{name}.stage"
        created_stage = False
        stage_identity: tuple[int, int] | None = None
        try:
            self._runtime.verify()
            if self._runtime.exists(temporary):
                self._recover_stage(temporary, name, encoded)
            if self._runtime.exists(name):
                raw = self._durable_read(name)
                if raw != encoded:
                    raise CaptureError("existing content-addressed artifact has different bytes")
            else:
                descriptor = self._runtime.open_leaf(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created_stage = True
                try:
                    opened_stage = os.fstat(descriptor)
                    stage_identity = (opened_stage.st_dev, opened_stage.st_ino)
                    view = memoryview(encoded)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise CaptureError("artifact staging write made no progress")
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                try:
                    os.link(temporary, name, src_dir_fd=self._runtime.fd,
                            dst_dir_fd=self._runtime.fd, follow_symlinks=False)
                    os.unlink(temporary, dir_fd=self._runtime.fd)
                    created_stage = False
                    os.fsync(self._runtime.fd)
                except FileExistsError:
                    pass
                raw = self._durable_read(name)
                if raw != encoded:
                    raise CaptureError("published content-addressed artifact bytes conflict")
                self._runtime.verify()
        except (OSError, SecureRuntimeError) as exc:
            raise CaptureError(f"artifact publication failed: {exc}") from exc
        finally:
            active_error = sys.exc_info()[1]
            if created_stage:
                # Only remove the inode this call exclusively created. If publication linked it
                # and cleanup itself failed, preserve both names for deterministic retry recovery.
                try:
                    self._cleanup_created_stage(temporary, name, stage_identity)
                except BaseException:
                    if active_error is None:
                        raise
            try:
                self._runtime.verify()
            except BaseException:
                if active_error is None:
                    raise
        reread = json.loads(raw)
        if reread != raw_body:
            raise CaptureError("published artifact bytes do not decode to the sealed record")
        return StoredArtifact(name, hashlib.sha256(raw).hexdigest(), True)

    def _cleanup_created_stage(self, temporary: str, name: str,
                               expected: tuple[int, int] | None) -> None:
        if expected is None:
            return
        try:
            stage = os.stat(temporary, dir_fd=self._runtime.fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (stage.st_dev, stage.st_ino) != expected:
            raise CaptureError("created stage name was replaced during cleanup")
        try:
            target = os.stat(name, dir_fd=self._runtime.fd, follow_symlinks=False)
        except FileNotFoundError:
            target = None
        if target is not None and (target.st_dev, target.st_ino) == expected:
            return  # exact post-link/pre-unlink state; deterministic retry owns recovery
        # The stage name still denotes the exact inode exclusively created by this call.
        os.unlink(temporary, dir_fd=self._runtime.fd)
        os.fsync(self._runtime.fd)

    def verify(self, namespace: str, body: Mapping[str, Any]) -> StoredArtifact:
        """Verify an existing exact object without creating or recovering anything."""
        with self._exclusive():
            return self._verify_locked(namespace, body)

    def read(self, locator: str, sha256: str) -> Mapping[str, Any]:
        """Read one exact sealed locator through the same pinned private-store checks."""
        locator = _text(locator, "artifact locator")
        _sha(sha256, "artifact sha256")
        if ("/" in locator or locator.startswith(".") or not locator.endswith(".json")
                or len(locator) > 256):
            raise CaptureError("artifact locator is not a private store leaf")
        with self._exclusive():
            try:
                self._runtime.verify()
                raw = self._durable_read(locator)
            except (OSError, SecureRuntimeError) as exc:
                raise CaptureError(f"artifact read failed: {exc}") from exc
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise CaptureError("artifact bytes differ from requested digest")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CaptureError("artifact bytes are not JSON") from exc
        if not isinstance(value, dict):
            raise CaptureError("artifact body is not an object")
        return _freeze(value)

    def _verify_locked(self, namespace: str, body: Mapping[str, Any]) -> StoredArtifact:
        name, encoded, raw_body = self._identity(namespace, body)
        temporary = f".{name}.stage"
        try:
            self._runtime.verify()
            if self._runtime.exists(temporary):
                raise CaptureError("artifact has an unresolved staged publication")
            if not self._runtime.exists(name):
                raise CaptureError("artifact does not exist")
            raw = self._durable_read(name)
        except (OSError, SecureRuntimeError) as exc:
            raise CaptureError(f"artifact verification failed: {exc}") from exc
        if raw != encoded or json.loads(raw) != raw_body:
            raise CaptureError("existing artifact does not match exact expected bytes")
        return StoredArtifact(name, hashlib.sha256(raw).hexdigest(), True)

    @staticmethod
    def _identity(namespace: str, body: Mapping[str, Any]) \
            -> tuple[str, bytes, dict[str, Any]]:
        namespace = _text(namespace, "artifact namespace")
        raw_body = _plain(body)
        if not isinstance(raw_body, dict):
            raise CaptureError("artifact body must be an object")
        content_digest = hashlib.sha256(_canonical(raw_body)).hexdigest()
        name = f"{hashlib.sha256(namespace.encode()).hexdigest()}-{content_digest}.json"
        encoded = json.dumps(raw_body, indent=2, sort_keys=True).encode("utf-8")
        return name, encoded, raw_body

    def _durable_read(self, name: str) -> bytes:
        raw, _ = self._read_pinned(name, allowed_links={1})
        return raw

    def _read_pinned(self, name: str, *, allowed_links: set[int],
                     limit: int = 64 * 1024 * 1024) -> tuple[bytes, os.stat_result]:
        descriptor = self._runtime.open_leaf(name, os.O_RDONLY)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_nlink not in allowed_links or before.st_size > limit):
                raise CaptureError("artifact leaf has unsafe type, owner, mode, links, or size")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise CaptureError("artifact ended before its pinned size")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise CaptureError("artifact grew while being read")
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(name, dir_fd=self._runtime.fd, follow_symlinks=False)
            def identity(value: os.stat_result) -> tuple[int, ...]:
                return (value.st_dev, value.st_ino, value.st_mode, value.st_uid,
                        value.st_nlink, value.st_size, value.st_mtime_ns)
            if identity(before) != identity(after) or identity(after) != identity(named):
                raise CaptureError("artifact identity changed while being verified")
        finally:
            os.close(descriptor)
        os.fsync(self._runtime.fd)
        self._runtime.verify()
        return b"".join(chunks), after

    def _recover_stage(self, temporary: str, name: str, encoded: bytes) -> None:
        stage_raw, stage = self._read_pinned(temporary, allowed_links={1, 2})
        final_exists = self._runtime.exists(name)
        if final_exists:
            final_raw, target = self._read_pinned(name, allowed_links={1, 2})
            if ((stage.st_dev, stage.st_ino) != (target.st_dev, target.st_ino)
                    or stage.st_nlink != 2 or target.st_nlink != 2
                    or stage_raw != encoded or final_raw != encoded):
                raise CaptureError("published final conflicts with staged artifact")
        elif stage.st_nlink != 1:
            raise CaptureError("unpublished staged artifact has invalid identity")
        elif stage_raw != encoded:
            self._quarantine_stage(temporary, stage, stage_raw)
            return
        else:
            # _read_pinned fsynced this exact stage descriptor before publication.
            os.link(temporary, name, src_dir_fd=self._runtime.fd,
                    dst_dir_fd=self._runtime.fd, follow_symlinks=False)
        current = os.stat(temporary, dir_fd=self._runtime.fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (stage.st_dev, stage.st_ino):
            raise CaptureError("staged artifact name was replaced before cleanup")
        os.unlink(temporary, dir_fd=self._runtime.fd)
        os.fsync(self._runtime.fd)
        self._runtime.verify()

    def _quarantine_stage(self, temporary: str, stage: os.stat_result, raw: bytes) -> None:
        quarantine = (f".{temporary.lstrip('.')}.quarantine-"
                      f"{stage.st_dev:x}-{stage.st_ino:x}-{hashlib.sha256(raw).hexdigest()[:16]}")
        try:
            os.stat(quarantine, dir_fd=self._runtime.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CaptureError("precisely named stage quarantine already exists")
        current = os.stat(temporary, dir_fd=self._runtime.fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (stage.st_dev, stage.st_ino):
            raise CaptureError("staged artifact changed before quarantine")
        os.rename(temporary, quarantine, src_dir_fd=self._runtime.fd,
                  dst_dir_fd=self._runtime.fd)
        moved = os.stat(quarantine, dir_fd=self._runtime.fd, follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != (stage.st_dev, stage.st_ino):
            raise CaptureError("quarantined stage identity mismatch")
        os.fsync(self._runtime.fd)
        self._runtime.verify()

    def close(self) -> None:
        with self._thread_lock:
            if getattr(self._thread_state, "exclusive_depth", 0):
                raise CaptureError("cannot close artifact store inside an active transaction")
            self._runtime.close()


# This callback is implemented by the controller.  It MUST perform lookup, exact-payload
# idempotence, append, cursor update, and append-uncertainty poisoning under one lock/capability.
CaptureTransaction = Callable[[str, Mapping[str, Any]], Any]


class _MeasurementCaptureBuilder:
    """Shared raw/carrier builder; subclasses choose deferred or direct publication."""

    def __init__(self, *, context: CaptureContext, store: ArtifactStore):
        self.context = CaptureContext.from_dict(context.to_dict())
        self.store = store
        self._artifacts: list[tuple[dict[str, Any], StoredArtifact]] = []

    def __call__(self, artifact: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(artifact, Mapping):
            raise CaptureError("planned-serving artifact must be an object")
        copied = _plain(artifact)
        digest = copied.pop("artifact_digest", None)
        if _sha(digest, "planned-serving artifact digest") != schemas.content_hash(copied):
            raise CaptureError("planned-serving artifact digest mismatch")
        copied["artifact_digest"] = digest
        for field, expected in (("lineage_id", self.context.lineage_id),
                                ("grant_id", self.context.grant_id),
                                ("container_id", self.context.container_id)):
            if field in copied and copied[field] != expected:
                raise CaptureError(f"planned-serving artifact {field} conflicts with context")
        if copied.get("kind") != "continued_unit":
            worker = copied.get("worker_identity")
            if (not isinstance(worker, Mapping)
                    or worker.get("worker_id") != self.context.worker_id
                    or worker.get("worker_incarnation") != self.context.worker_incarnation
                    or worker.get("supervisor_id") != self.context.supervisor_id
                    or worker.get("supervisor_incarnation") != self.context.supervisor_incarnation
                    or worker.get("config_generation") != self.context.config_generation):
                raise CaptureError("planned-serving artifact worker identity conflicts with context")
        stored = self.store.write(f"raw:{digest}", copied)
        self._artifacts.append((copied, stored))
        return stored.to_dict()

    def _build_payloads(self, summary: Mapping[str, Any]) \
            -> tuple[tuple[str, Mapping[str, Any], StoredArtifact], ...]:
        """Seal one carrier per arm without crossing the controller boundary."""
        plan = ep.ExperimentPlan.from_dict(_plain(summary["plan"]))
        if summary.get("plan_digest") != plan.digest:
            raise CaptureError("final capture plan digest mismatch")
        if (self.context.campaign_id != plan.campaign_id
                or self.context.protocol_id != plan.protocol_ref
                or self.context.protocol_status != plan.protocol_status):
            raise CaptureError("capture context campaign/protocol differs from frozen plan")
        supplied_view = summary.get("admissible_view")
        if not isinstance(supplied_view, Mapping) or supplied_view.get("view_digest") is None:
            raise CaptureError("final capture requires the immutable admissible view")
        raw_objects = summary.get("raw_units")
        if not isinstance(raw_objects, (list, tuple)):
            raise CaptureError("final capture raw_units must be an array")
        try:
            raw_units = tuple(ep.RawUnit.from_dict(item) for item in raw_objects)
            view = ep.admissible_units(plan, raw_units).to_dict()
        except Exception as exc:
            raise CaptureError("final capture raw units are invalid") from exc
        if _plain(supplied_view) != view:
            raise CaptureError("supplied admissible view differs from canonical raw-unit view")
        if summary.get("lineage_id") != self.context.lineage_id:
            raise CaptureError("final capture lineage differs from context")
        identities = summary.get("comparison_identities")
        if not isinstance(identities, Mapping) or set(identities) != _ARMS:
            raise CaptureError("final capture comparison identities are malformed")
        for arm in _ARMS:
            source = self.context.source_identities[arm]
            identity = identities[arm]
            if (source["model_sha256"] != identity.get("model_digest")
                    or source["build_sha256"] != identity.get("executable_digest")
                    or source["recipe_hash"] != identity.get("template_hash")):
                raise CaptureError(f"{arm} source identity conflicts with resolved execution")
        rows = view.get("selected_rows")
        if not isinstance(rows, list):
            raise CaptureError("admissible view selected_rows must be an array")
        results: list[tuple[str, Mapping[str, Any], StoredArtifact]] = []
        for arm in ("anchor", "candidate"):
            arm_rows = [row for row in rows if isinstance(row, Mapping)
                        and row.get("arm") == arm]
            attempts = [item for item, stored in self._artifacts
                        if item.get("kind") == "completed_attempt" and item.get("arm") == arm]
            raw_artifacts = [{"document": item, "stored": stored.to_dict()}
                             for item, stored in self._artifacts if item.get("arm") == arm]
            if not raw_artifacts:
                continue
            diagnostic = self._diagnostic(plan, arm, arm_rows, attempts, raw_artifacts)
            values = [float(row["value"]) for row in arm_rows]
            measurement = None
            if diagnostic is None and values:
                if plan.estimator_id != "median.v1":
                    diagnostic = f"unsupported estimator {plan.estimator_id!r}"
                else:
                    measurement = {"metric": plan.metric, "value": median(values),
                                   "unit": "t/s", "independent_unit": plan.unit,
                                   "direction": plan.metric_direction,
                                   "independent_n": len(values),
                                   "reps_basis": "scored independent process launches",
                                   "per_launch_values": values}
            measurement_id = schemas.content_hash({
                "producer": PRODUCER_ID, "plan_digest": plan.digest,
                "lineage_id": summary["lineage_id"], "arm": arm})
            body = _plain({"schema": CAPTURE_SCHEMA, "producer": PRODUCER_ID,
                    "measurement_id": measurement_id, "arm": arm,
                    "arm_locator": f"planned-serving:{plan.digest}:{summary['lineage_id']}:{arm}",
                    "plan": plan.to_dict(),
                    "prompt_manifest": summary["prompt_manifest"],
                    "prompt_manifest_digest": summary["prompt_manifest_digest"],
                    "lineage_id": summary["lineage_id"],
                    "comparison_identities": _plain(identities),
                    "source_identity": _plain(self.context.source_identities[arm]),
                    "capture_context": self.context.to_dict(),
                    "admissible_view": _plain(view), "raw_artifacts": raw_artifacts,
                    "environment_verdicts": self._environment(
                        attempts, identities[arm].get("backend")),
                    "status": "measurement" if measurement is not None else "diagnostic",
                    "diagnostic_reason": diagnostic, "measurement": measurement,
                    "claim": (f"{arm} {plan.metric} for frozen plan {plan.plan_id}"),
                    "category": plan.category, "phase": plan.phase,
                    "record_class": plan.record_class, "intended_use": plan.intended_use,
                    "protocol_id": self.context.protocol_id,
                    "protocol_status": self.context.protocol_status,
                    "instrument_id": self.context.instrument_id,
                    "interval": self._interval(raw_artifacts) if any(
                        item["document"].get("kind") == "native_observation"
                        for item in raw_artifacts) else None})
            carrier = dict(body, carrier_digest=schemas.content_hash(body))
            sealed = self.store.write(f"carrier:{measurement_id}", carrier)
            payload = {"schema": CAPTURE_SCHEMA, "measurement_id": measurement_id,
                       "carrier": carrier, "artifact": sealed.to_dict()}
            results.append((measurement_id, payload, sealed))
        return tuple(results)

    @staticmethod
    def _environment(attempts: Sequence[Mapping[str, Any]], backend: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        labels = {
            "contention": {"pass": "clean", "fail": "contaminated", "unknown": "unknown"},
            "placement": {"pass": "proven", "fail": "refuted", "unknown": "unknown"},
            "residency": {"pass": "proven", "fail": "refuted", "unknown": "unknown"},
        }
        for attempt in attempts:
            witnesses = attempt.get("stage_witnesses")
            if not isinstance(witnesses, Mapping):
                witnesses = {}
            row: dict[str, Any] = {"unit_id": attempt.get("unit_id")}
            for name, states in labels.items():
                witness = witnesses.get(name)
                status = witness.get("status") if isinstance(witness, Mapping) else "unknown"
                ref = witness.get("ref") if isinstance(witness, Mapping) else None
                if name == "residency" and backend == "cpu":
                    row[name] = {"verdict": "not_applicable", "ref": None}
                else:
                    row[name] = {"verdict": states.get(status, "unknown"), "ref": ref}
            out.append(row)
        return out

    @staticmethod
    def _interval(raw_artifacts: Sequence[Mapping[str, Any]]) -> Mapping[str, str]:
        intervals: list[tuple[datetime, datetime, str, str]] = []
        for item in raw_artifacts:
            document = item.get("document")
            if not isinstance(document, Mapping) or document.get("kind") != "native_observation":
                continue
            start_text = _timestamp(document.get("observed_started_at"),
                                    "observed_started_at")
            end_text = _timestamp(document.get("observed_ended_at"), "observed_ended_at")
            start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
            if end < start:
                raise CaptureError("a producer-authored measurement interval is reversed")
            intervals.append((start, end, start_text, end_text))
        if not intervals:
            raise CaptureError("arm has no complete producer-authored measurement interval")
        first = min(intervals, key=lambda item: item[0])
        last = max(intervals, key=lambda item: item[1])
        return MappingProxyType({"start": first[2], "end": last[3]})

    @staticmethod
    def _diagnostic(plan: ep.ExperimentPlan, arm: str, rows: Sequence[Mapping[str, Any]],
                    attempts: Sequence[Mapping[str, Any]],
                    raw_artifacts: Sequence[Mapping[str, Any]]) -> str | None:
        if (plan.instrument_class != "serving" or plan.unit != "process"
                or plan.estimator_id != "median.v1" or plan.estimand != "level"):
            return "unsupported serving unit/estimator/estimand combination"
        if not rows:
            return "zero scored independent launches"
        if len(rows) != int(plan.stopping["n_per_arm"]):
            return "incomplete admissible arm"
        by_digest = {item["document"].get("artifact_digest"): item
                     for item in raw_artifacts
                     if item["document"].get("kind") == "native_observation"}
        for row in rows:
            attempts_for_row = [item for item in attempts if item.get("unit_id") == row["unit_id"]]
            if len(attempts_for_row) != 1:
                return "missing or duplicate completed attempt"
            attempt = attempts_for_row[0]
            native = by_digest.get(attempt.get("native_observation_digest"))
            if native is None:
                return "completed attempt does not bind a native observation"
            document = native["document"]
            selected = document.get("selected_observation")
            requests = selected.get("requests") if isinstance(selected, Mapping) else None
            if not isinstance(requests, list) or not requests:
                return "native observation has no request rows"
            measured = [item for item in requests if isinstance(item, Mapping)
                        and item.get("phase") == "measurement"]
            if len(measured) != len(row["prompt_ids"]):
                return "partial slot metric"
            rates: list[float] = []
            for item in measured:
                rate = item.get("predicted_per_second")
                if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                        or not math.isfinite(float(rate)) or item.get("terminal") is not True
                        or item.get("error") is not None):
                    return "partial or invalid slot metric"
                rates.append(float(rate))
            if sum(rates) != float(row["value"]):
                return "raw slot sum differs from admitted unit value"
            witnesses = attempt.get("stage_witnesses")
            if not isinstance(witnesses, Mapping):
                return "provider witness set missing"
            for name in ("contention", "placement"):
                witness = witnesses.get(name)
                if not isinstance(witness, Mapping) or witness.get("status") != "pass" \
                        or not witness.get("ref"):
                    return f"{name} verdict unknown or failed"
            if arm == "candidate" or arm == "anchor":
                # Backend is exact in the resolved arm identity.  CPU records retain residency
                # as explicitly not-applicable; GPU records require an in-window pass.
                backend = document["comparison_identities"][arm].get("backend")
                residency = witnesses.get("residency")
                if backend == "gpu":
                    if not isinstance(residency, Mapping) or residency.get("status") != "pass" \
                            or not residency.get("ref"):
                        return "GPU in-window residency verdict unknown or failed"
                elif backend != "cpu":
                    return "unsupported or missing backend identity"
        return None


class DeferredNativeMeasurementSink(_MeasurementCaptureBuilder):
    """Child-only sink that seals exact native payloads but has no Journal callback."""

    def __init__(self, *, context: CaptureContext, store: ArtifactStore):
        super().__init__(context=context, store=store)
        self._captures: tuple[Mapping[str, Any], ...] = ()

    @property
    def captures(self) -> tuple[Mapping[str, Any], ...]:
        return self._captures

    def finalize_run(self, summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        captures: list[Mapping[str, Any]] = []
        receipts: list[Mapping[str, Any]] = []
        for measurement_id, payload, sealed in self._build_payloads(summary):
            plain = _plain(payload)
            digest = schemas.content_hash(plain)
            captures.append(MappingProxyType({
                "measurement_id": measurement_id,
                "payload": _freeze(plain),
                "payload_digest": digest,
                "artifact": _freeze(sealed.to_dict()),
            }))
            receipts.append(MappingProxyType({
                "measurement_id": measurement_id,
                "payload_digest": digest,
                "artifact": _freeze(sealed.to_dict()),
            }))
        self._captures = tuple(captures)
        return tuple(receipts)


class NativeMeasurementSink(_MeasurementCaptureBuilder):
    """Real planned-serving sink that seals raw artifacts then journals arm carriers."""

    def __init__(self, *, context: CaptureContext, store: ArtifactStore,
                 capture_transaction: CaptureTransaction):
        super().__init__(context=context, store=store)
        if not callable(capture_transaction):
            raise CaptureError("serialized capture transaction is required")
        self.capture_transaction = capture_transaction

    def finalize_run(self, summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """Preserve the direct same-thread serialized transaction semantics."""
        results: list[Mapping[str, Any]] = []
        for measurement_id, payload, sealed in self._build_payloads(summary):
            # Idempotence/conflict handling belongs inside this callback's single serialized
            # lookup+append boundary.  The content-addressed artifact is evidence, not a WAL.
            entry = self.capture_transaction(measurement_id, payload)
            results.append(MappingProxyType({"measurement_id": measurement_id,
                                             "artifact": sealed.to_dict(),
                                             "journal_entry": entry}))
        return tuple(results)


__all__ = ["ArtifactStore", "CAPTURE_SCHEMA", "CaptureContext", "CaptureError",
           "DeferredNativeMeasurementSink", "JOURNAL_KIND", "NativeMeasurementSink",
           "PRODUCER_ID", "StoredArtifact"]
