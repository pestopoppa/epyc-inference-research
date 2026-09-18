#!/usr/bin/env python3
"""Bounded, provenance-preserving snapshots of legacy AutoKernel state.

The snapshot is historical input.  It is deliberately not a Journal, candidate
manifest, validation receipt, or serving-eligibility record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any, Callable, Iterable, Mapping

from .. import journal
from ..controller import experiments
from ..controller.discovery_supervisor_secure import (
    SecureRuntimeError,
    open_beneath,
    read_stable_fd,
)
from . import accumulate
from .measurement_capture import ArtifactStore, CaptureError


SNAPSHOT_SCHEMA = "epyc.autokernel.legacy_migration_snapshot.v1"
SNAPSHOT_VERSION = 1
ARTIFACT_NAMESPACE_PREFIX = "epyc.autokernel.legacy_migration/"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORDS = 100_000
MAX_DESTINATION_ENTRIES = 4_096
EXPERIMENT_FIELDS = (
    "attempt_id", "recorded_at", "campaign_id", "deployment", "epoch_sha256",
    "hypothesis_id", "mechanism_id", "target_surface", "target_symbol", "statement",
    "falsifier", "status", "effect_fraction", "exact_effect", "target_effect",
    "refusal_reason", "result_sha256", "payload",
)
PRODUCTION_ROOTS = (
    Path("/mnt/raid0/llm/llama.cpp"),
    Path("/mnt/raid0/llm/whisper.cpp"),
    Path("/mnt/raid0/llm/qwentts.cpp"),
)


class MigrationRefused(RuntimeError):
    """The requested snapshot cannot be assembled without guessing or mutation."""


class UnsupportedSnapshot(MigrationRefused):
    """A reader cannot safely interpret the snapshot version."""


@dataclass(frozen=True)
class MigrationRequest:
    import_id: str
    campaign_id: str
    source_root: Path
    destination_root: Path
    source_repo: Path
    anchor_commit: str
    config_path: str | None = None
    artifact_paths: tuple[str, ...] = ()
    max_bytes: int = DEFAULT_MAX_BYTES
    max_records: int = DEFAULT_MAX_RECORDS

    def validated(self) -> "MigrationRequest":
        for value, label in ((self.import_id, "import_id"),
                             (self.campaign_id, "campaign_id"),
                             (self.anchor_commit, "anchor_commit")):
            if not isinstance(value, str) or not value.strip():
                raise MigrationRefused(f"{label} must be non-empty text")
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
               for char in self.import_id):
            raise MigrationRefused("import_id must be a safe identifier")
        for value, maximum, label in (
            (self.max_bytes, DEFAULT_MAX_BYTES, "max_bytes"),
            (self.max_records, DEFAULT_MAX_RECORDS, "max_records"),
        ):
            if type(value) is not int or value <= 0 or value > maximum:
                raise MigrationRefused(
                    f"{label} must be a positive integer no greater than {maximum}")
        source = _safe_existing_directory(self.source_root, "source root")
        repo = _safe_existing_directory(self.source_repo, "source repository")
        destination = _safe_destination(self.destination_root)
        if _overlaps(source, destination) or _same_directory(source, destination):
            raise MigrationRefused("destination may not alias or contain the source root")
        if _overlaps(repo, destination) or _same_directory(repo, destination):
            raise MigrationRefused("destination may not alias or contain the source repository")
        for protected in PRODUCTION_ROOTS:
            if protected.exists() and (_overlaps(protected, destination)
                                       or _same_directory(protected, destination)):
                raise MigrationRefused("destination overlaps a protected production root")
        paths = tuple(_relative(value, "artifact path") for value in self.artifact_paths)
        config = None if self.config_path is None else _relative(self.config_path, "config path")
        return MigrationRequest(self.import_id, self.campaign_id, source, destination,
                                repo, self.anchor_commit, config, paths,
                                self.max_bytes, self.max_records)


@dataclass(frozen=True)
class MigrationResult:
    mode: str
    snapshot: Mapping[str, Any]
    locator: str
    sha256: str
    created: bool

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "locator": self.locator, "sha256": self.sha256,
                "created": self.created, "snapshot_schema": self.snapshot["schema"],
                "campaign_id": self.snapshot["campaign_id"],
                "authority": self.snapshot["authority"]}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _relative(value: str, label: str) -> str:
    path = Path(value)
    if (not value or path.is_absolute() or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise MigrationRefused(f"{label} must be a safe relative path")
    return path.as_posix()


def _safe_existing_directory(value: Path, label: str) -> Path:
    path = Path(value).absolute()
    _reject_symlink_components(path)
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise MigrationRefused(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise MigrationRefused(f"{label} is not a directory")
    return path


def _safe_destination(value: Path) -> Path:
    path = Path(value).absolute()
    _reject_symlink_components(path, allow_missing_leaf=True)
    if path.exists():
        info = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise MigrationRefused(
                "existing destination must be an owned mode-0700 directory")
    elif not path.parent.is_dir():
        raise MigrationRefused("destination parent must already exist")
    return path


def _reject_symlink_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(path.parts[1:]) - 1:
                return
            raise MigrationRefused(f"path component does not exist: {current}")
        if stat.S_ISLNK(info.st_mode):
            raise MigrationRefused(f"symlink path component refused: {current}")


def _overlaps(left: Path, right: Path) -> bool:
    try:
        return left == right or left in right.parents or right in left.parents
    except RuntimeError:
        return True


def _same_directory(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    try:
        a = os.stat(left, follow_symlinks=False)
        b = os.stat(right, follow_symlinks=False)
    except OSError as exc:
        raise MigrationRefused(f"cannot compare directory identities: {exc}") from exc
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _identity(info: os.stat_result) -> dict[str, int]:
    return {"device": info.st_dev, "inode": info.st_ino, "mode": stat.S_IMODE(info.st_mode),
            "uid": info.st_uid, "links": info.st_nlink, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def _root_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), **_identity(os.stat(path, follow_symlinks=False))}


def _read_relative(root: Path, relative: str, *, limit: int) -> tuple[bytes, dict[str, Any]]:
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = open_beneath(root_fd, relative)
        try:
            raw, identity = read_stable_fd(fd, limit=limit)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise MigrationRefused(f"source is not a regular file: {relative}")
            named = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
            if _identity(named) != _identity(info):
                raise MigrationRefused(f"source identity changed while reading: {relative}")
            return raw, {"path": relative, **_identity(info),
                         "sha256": hashlib.sha256(raw).hexdigest()}
        finally:
            os.close(fd)
    except (OSError, SecureRuntimeError, ValueError) as exc:
        raise MigrationRefused(f"cannot read bounded source {relative}: {exc}") from exc
    finally:
        os.close(root_fd)


def _journal_paths(source: Path) -> list[str]:
    root = source / accumulate.JOURNAL_DIRNAME
    if not root.exists():
        return []
    try:
        book = journal.Journal(str(root))
        paths = [Path(ref.path).relative_to(source).as_posix() for ref in book.shards()]
    except (OSError, journal.JournalError, ValueError) as exc:
        raise MigrationRefused(f"legacy journal cannot be inventoried: {exc}") from exc
    lock = root / journal.LOCK_NAME
    if lock.exists() or lock.is_symlink():
        paths.append(lock.relative_to(source).as_posix())
    return sorted(paths)


def _managed_paths(request: MigrationRequest) -> list[str]:
    paths: set[str] = set(request.artifact_paths)
    if request.config_path:
        paths.add(request.config_path)
    for name in (accumulate.Bundle.FILENAME, "experiments.db",
                 "experiments.db-wal", "experiments.db-shm"):
        if (request.source_root / name).exists() or (request.source_root / name).is_symlink():
            paths.add(name)
    paths.update(_journal_paths(request.source_root))
    return sorted(paths)


def _frontier(request: MigrationRequest, paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    remaining = request.max_bytes
    result: dict[str, dict[str, Any]] = {}
    for relative in paths:
        raw, identity = _read_relative(request.source_root, relative, limit=remaining)
        remaining -= len(raw)
        if remaining < 0:
            raise MigrationRefused("source exceeds max_bytes")
        result[relative] = identity
    return result


def _load_experiments(request: MigrationRequest) -> dict[str, Any]:
    path = request.source_root / "experiments.db"
    if not path.exists():
        return {"schema_version": "unknown_legacy", "records": [],
                "status": "missing_unknown_legacy"}
    if ((request.source_root / "experiments.db-wal").exists()
            or (request.source_root / "experiments.db-shm").exists()):
        raise MigrationRefused(
            "active SQLite WAL/SHM source is unsupported; stable checkpoint required")
    # immutable=1 prevents a nominally read-only inspection from creating sidecars.
    # It is safe only because active WAL/SHM state is refused immediately above.
    uri = f"{path.as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(experiments)")]
        expected = list(EXPERIMENT_FIELDS)
        if columns != expected:
            raise MigrationRefused("unsupported ExperimentStore schema")
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY recorded_at, rowid LIMIT ?",
            (request.max_records + 1,)).fetchall()
        if len(rows) > request.max_records:
            raise MigrationRefused("experiment history exceeds max_records")
        records = []
        payload_bytes = 0
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, str):
                raise MigrationRefused("ExperimentStore payload is not text")
            payload_bytes += len(payload.encode("utf-8"))
            if payload_bytes > request.max_bytes:
                raise MigrationRefused("experiment payloads exceed max_bytes")
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise MigrationRefused("ExperimentStore contains invalid payload JSON") from exc
            records.append({key: row[key] for key in expected if key != "payload"})
            records[-1]["payload"] = parsed
        return {"schema_version": experiments.SCHEMA_VERSION, "records": records,
                "status": "historical_records"}
    except sqlite3.Error as exc:
        raise MigrationRefused(f"ExperimentStore read failed: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()


def assemble_snapshot(request: MigrationRequest, *,
                      is_ancestor: Callable[[str, str], bool]) -> dict[str, Any]:
    """Read one bounded stable frontier and return an authority-neutral snapshot."""
    request = request.validated()
    paths = _managed_paths(request)
    before = _frontier(request, paths)
    try:
        bundle, note = accumulate.load_bundle(
            request.source_root, anchor_commit=request.anchor_commit,
            is_ancestor=is_ancestor, read_only=True)
    except accumulate.BundleRecoveryRequired as exc:
        raise MigrationRefused(str(exc)) from exc
    if accumulate.Bundle.FILENAME in before:
        bundle_raw, bundle_identity = _read_relative(
            request.source_root, accumulate.Bundle.FILENAME, limit=request.max_bytes)
        try:
            original_bundle: Any = json.loads(bundle_raw)
        except json.JSONDecodeError as exc:
            raise MigrationRefused("legacy bundle projection is invalid JSON") from exc
    else:
        original_bundle = "unknown_legacy_missing_projection"
        bundle_identity = {"status": "unknown_legacy_missing_projection"}
    history = _load_experiments(request)
    source_campaign_ids = sorted({row["campaign_id"] for row in history["records"]
                                  if isinstance(row.get("campaign_id"), str)
                                  and row["campaign_id"]})
    config = ({"status": "unknown_legacy"} if request.config_path is None else
              {"status": "identified", "identity": before[request.config_path]})
    artifacts = [{"status": "identified", **before[path]}
                 for path in request.artifact_paths]
    after_paths = _managed_paths(request)
    after = _frontier(request, after_paths)
    if paths != after_paths or before != after:
        raise MigrationRefused("source durable frontier changed during snapshot")
    source_material = {"root": _root_identity(request.source_root),
                       "repository": _root_identity(request.source_repo),
                       "anchor_commit": request.anchor_commit, "frontier": before}
    request_material = {"import_id": request.import_id,
                        "campaign_id": request.campaign_id,
                        "source": source_material,
                        "config_path": request.config_path,
                        "artifact_paths": list(request.artifact_paths)}
    return {
        "schema": SNAPSHOT_SCHEMA,
        "schema_version": SNAPSHOT_VERSION,
        "engine_compatibility": {"readers": [SNAPSHOT_VERSION],
                                 "live_mutation": "unsupported_historical_only",
                                 "accumulator_reader": "accumulate.load_bundle/read_only",
                                 "experiment_store_schema": history["schema_version"]},
        "import_id": request.import_id,
        "campaign_id": request.campaign_id,
        "request_sha256": hashlib.sha256(_canonical(request_material)).hexdigest(),
        "source": {**source_material,
                   "campaign_ids": (source_campaign_ids if source_campaign_ids
                                    else "unknown_legacy")},
        "config": config,
        "artifacts": artifacts,
        "history": {
            "accumulator": {"original_snapshot": original_bundle,
                            "original_identity": bundle_identity,
                            "historical_view": bundle.to_dict(),
                            "recovery_note": note,
                            "authority": "unknown_legacy"},
            "experiments": history,
        },
        "authority": {"classification": "historical_input_only",
                      "measurement": "unknown_legacy",
                      "candidate_state": "not_created",
                      "validated": False,
                      "serving_eligible": False},
    }


def _expected_artifact(snapshot: Mapping[str, Any]) -> tuple[str, bytes, str]:
    namespace = ARTIFACT_NAMESPACE_PREFIX + str(snapshot["import_id"])
    locator, encoded, _ = ArtifactStore._identity(namespace, snapshot)
    return locator, encoded, hashlib.sha256(encoded).hexdigest()


def _destination_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries = []
    with os.scandir(path) as iterator:
        for entry in iterator:
            entries.append(entry.name)
            if len(entries) > MAX_DESTINATION_ENTRIES:
                raise MigrationRefused("destination entry bound exceeded")
    return sorted(entries)


def _check_destination(path: Path, locator: str, encoded: bytes) -> tuple[list[str], bool]:
    entries = _destination_entries(path)
    allowed = {locator, f".{locator}.stage"}
    if any(name not in allowed for name in entries):
        raise MigrationRefused(
            "destination is not dedicated to this exact import or contains conflicting state")
    for name in entries:
        raw, _ = _read_relative(path, name, limit=len(encoded))
        if raw != encoded:
            raise MigrationRefused("destination contains conflicting import bytes")
    return entries, locator in entries


def migrate(request: MigrationRequest, *, is_ancestor: Callable[[str, str], bool],
            dry_run: bool = True) -> MigrationResult:
    """Dry-run without creating anything, or publish one exact dedicated artifact."""
    request = request.validated()
    snapshot = assemble_snapshot(request, is_ancestor=is_ancestor)
    _parse_v1(snapshot)
    locator, encoded, digest = _expected_artifact(snapshot)
    if len(encoded) > min(request.max_bytes, DEFAULT_MAX_BYTES):
        raise MigrationRefused("serialized snapshot exceeds the supported reader byte bound")
    _entries, existed = _check_destination(request.destination_root, locator, encoded)
    if dry_run:
        return MigrationResult("dry_run", snapshot, locator, digest, False)
    store = ArtifactStore(request.destination_root)
    try:
        # The dedicated-destination check and publication must share ArtifactStore's
        # existing lock; otherwise another campaign can publish between them.
        with store.exclusive():
            _locked_entries, existed = _check_destination(
                request.destination_root, locator, encoded)
            artifact = store.write(ARTIFACT_NAMESPACE_PREFIX + request.import_id, snapshot)
        return MigrationResult("import", snapshot, artifact.locator,
                               artifact.sha256, not existed)
    except CaptureError as exc:
        raise MigrationRefused(f"durable snapshot publication failed: {exc}") from exc
    finally:
        store.close()


def _closed(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MigrationRefused(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationRefused(f"{label} must be non-empty text")
    return value


def _validate_experiment_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise MigrationRefused("snapshot experiment history must be an object")
    _closed(value, {"schema_version", "records", "status"}, "experiment history")
    records = value["records"]
    if not isinstance(records, list) or len(records) > DEFAULT_MAX_RECORDS:
        raise MigrationRefused("snapshot experiment records must be a bounded list")
    if value["status"] == "missing_unknown_legacy":
        if value["schema_version"] != "unknown_legacy" or records:
            raise MigrationRefused("missing experiment history is internally inconsistent")
        return records
    if value["status"] != "historical_records" or value["schema_version"] != 1:
        raise MigrationRefused("unsupported experiment history schema or status")
    nullable_text = {"deployment", "hypothesis_id", "mechanism_id", "target_surface",
                     "target_symbol", "statement", "falsifier", "refusal_reason",
                     "result_sha256"}
    numeric = {"effect_fraction", "exact_effect", "target_effect"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MigrationRefused(f"experiment record {index} must be an object")
        _closed(record, set(EXPERIMENT_FIELDS), f"experiment record {index}")
        for field in ("attempt_id", "recorded_at", "campaign_id", "epoch_sha256", "status"):
            _nonempty(record[field], f"experiment record {index}.{field}")
        for field in nullable_text:
            if record[field] is not None and not isinstance(record[field], str):
                raise MigrationRefused(f"experiment record {index}.{field} has invalid type")
        for field in numeric:
            number = record[field]
            if (number is not None
                    and (isinstance(number, bool) or not isinstance(number, (int, float))
                         or not math.isfinite(float(number)))):
                raise MigrationRefused(f"experiment record {index}.{field} has invalid type")
        if not isinstance(record["payload"], dict):
            raise MigrationRefused(f"experiment record {index}.payload must be an object")
    return records


def _parse_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    _closed(value, {"schema", "schema_version", "engine_compatibility", "import_id",
                    "campaign_id", "request_sha256", "source", "config", "artifacts",
                    "history", "authority"}, "snapshot")
    if value["schema"] != SNAPSHOT_SCHEMA or value["schema_version"] != SNAPSHOT_VERSION:
        raise UnsupportedSnapshot("snapshot is not an installed v1 schema")
    compatibility = value["engine_compatibility"]
    if not isinstance(compatibility, dict):
        raise MigrationRefused("engine_compatibility must be an object")
    _closed(compatibility, {"readers", "live_mutation", "accumulator_reader",
                            "experiment_store_schema"}, "engine_compatibility")
    if (compatibility["readers"] != [1]
            or compatibility["live_mutation"] != "unsupported_historical_only"
            or compatibility["accumulator_reader"] != "accumulate.load_bundle/read_only"
            or compatibility["experiment_store_schema"] not in (1, "unknown_legacy")):
        raise MigrationRefused("snapshot engine compatibility is unsupported")
    authority = value["authority"]
    expected_authority = {"classification": "historical_input_only",
                          "measurement": "unknown_legacy",
                          "candidate_state": "not_created", "validated": False,
                          "serving_eligible": False}
    if authority != expected_authority:
        raise MigrationRefused("snapshot authority is not immutable historical-only authority")
    _nonempty(value["import_id"], "import_id")
    _nonempty(value["campaign_id"], "campaign_id")
    source = value["source"]
    config = value["config"]
    artifacts = value["artifacts"]
    if not isinstance(source, dict):
        raise MigrationRefused("snapshot source must be an object")
    _closed(source, {"root", "repository", "anchor_commit", "frontier", "campaign_ids"},
            "snapshot source")
    _nonempty(source["anchor_commit"], "source.anchor_commit")
    if not isinstance(source["root"], dict) or not isinstance(source["repository"], dict):
        raise MigrationRefused("source root and repository identities must be objects")
    if not isinstance(source["frontier"], dict):
        raise MigrationRefused("source frontier must be an object")
    if (source["campaign_ids"] != "unknown_legacy"
            and (not isinstance(source["campaign_ids"], list)
                 or any(not isinstance(item, str) or not item for item in source["campaign_ids"]))):
        raise MigrationRefused("source campaign identities are malformed")
    if not isinstance(config, dict) or config.get("status") not in {"identified", "unknown_legacy"}:
        raise MigrationRefused("snapshot config identity is malformed")
    _closed(config, ({"status", "identity"} if config["status"] == "identified"
                     else {"status"}), "snapshot config")
    if config["status"] == "identified" and not isinstance(config["identity"], dict):
        raise MigrationRefused("identified snapshot config requires an identity object")
    if not isinstance(artifacts, list) or any(not isinstance(item, dict) for item in artifacts):
        raise MigrationRefused("snapshot artifacts must be a list of identities")
    for item in artifacts:
        if item.get("status") != "identified" or not isinstance(item.get("path"), str):
            raise MigrationRefused("snapshot artifact identity is malformed")
    history = value["history"]
    if not isinstance(history, dict):
        raise MigrationRefused("snapshot history must be an object")
    _closed(history, {"accumulator", "experiments"}, "snapshot history")
    accumulator = history["accumulator"]
    if not isinstance(accumulator, dict):
        raise MigrationRefused("snapshot accumulator history must be an object")
    _closed(accumulator, {"original_snapshot", "original_identity", "historical_view",
                          "recovery_note", "authority"}, "accumulator history")
    if accumulator["authority"] != "unknown_legacy":
        raise MigrationRefused("accumulator authority must remain unknown_legacy")
    if not isinstance(accumulator["recovery_note"], str):
        raise MigrationRefused("accumulator recovery_note must be text")
    view = accumulator["historical_view"]
    if not isinstance(view, dict):
        raise MigrationRefused("historical accumulator view must be an object")
    try:
        restored = accumulate.Bundle.from_dict(view)
    except (TypeError, ValueError) as exc:
        raise MigrationRefused(f"historical accumulator view is malformed: {exc}") from exc
    if restored.to_dict() != view:
        raise MigrationRefused("historical accumulator view is not canonical")
    records = _validate_experiment_history(history["experiments"])
    if compatibility["experiment_store_schema"] != history["experiments"]["schema_version"]:
        raise MigrationRefused("experiment schema conflicts with engine compatibility")
    source_material = {key: source[key] for key in
                       ("root", "repository", "anchor_commit", "frontier")}
    config_path = (config["identity"].get("path")
                   if config["status"] == "identified" else None)
    artifact_paths = [item.get("path") for item in artifacts]
    request_material = {"import_id": value["import_id"],
                        "campaign_id": value["campaign_id"], "source": source_material,
                        "config_path": config_path, "artifact_paths": artifact_paths}
    expected_digest = hashlib.sha256(_canonical(request_material)).hexdigest()
    if value["request_sha256"] != expected_digest:
        raise MigrationRefused("snapshot request digest is inconsistent")
    source_ids = sorted({record["campaign_id"] for record in records})
    expected_ids: list[str] | str = source_ids if source_ids else "unknown_legacy"
    if source["campaign_ids"] != expected_ids:
        raise MigrationRefused("source campaign identities conflict with experiment history")
    return {"authority": authority, "history": history, "view": view}


_PARSERS = {SNAPSHOT_VERSION: _parse_v1}


def inspect_snapshot(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    """Verify an ArtifactStore object and return its closed historical v1 view."""
    raw, identity = _read_relative(path.parent.absolute(), path.name, limit=max_bytes)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MigrationRefused("snapshot is invalid JSON") from exc
    if not isinstance(value, dict):
        raise MigrationRefused("snapshot must be an object")
    version = value.get("schema_version")
    parser = (_PARSERS.get(version)
              if isinstance(version, int) and not isinstance(version, bool) else None)
    if value.get("schema") != SNAPSHOT_SCHEMA or parser is None:
        raise UnsupportedSnapshot(
            f"snapshot version {version!r} is unsupported; no mutation or fallback allowed")
    parsed = parser(value)
    authority, history, view = parsed["authority"], parsed["history"], parsed["view"]
    cor = view.get("champion_of_record")
    if not isinstance(cor, str) or not cor:
        raise MigrationRefused("historical champion_of_record is unknown; anchor fallback refused")
    store = ArtifactStore(path.parent.absolute())
    try:
        verified = store.verify(ARTIFACT_NAMESPACE_PREFIX + value["import_id"], value)
    except CaptureError as exc:
        raise MigrationRefused(f"snapshot is not a verified ArtifactStore object: {exc}") from exc
    finally:
        store.close()
    if path.name != verified.locator:
        raise MigrationRefused(
            "requested snapshot filename does not match its verified ArtifactStore locator")
    return {"schema": SNAPSHOT_SCHEMA, "schema_version": version,
            "mode": "read_only_historical",
            "identity": {**identity, **verified.to_dict()},
            "campaign_id": value.get("campaign_id"), "import_id": value.get("import_id"),
            "champion_of_record": cor, "tip": view.get("tip"),
            "keeps": list(view.get("keeps", [])), "authority": authority,
            "experiment_records": len(history.get("experiments", {}).get("records", []))}


__all__ = ["MigrationRefused", "MigrationRequest", "MigrationResult",
           "SNAPSHOT_SCHEMA", "SNAPSHOT_VERSION", "UnsupportedSnapshot",
           "assemble_snapshot", "inspect_snapshot", "migrate"]
