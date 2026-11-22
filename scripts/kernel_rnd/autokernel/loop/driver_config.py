"""Unified driver config record and document loader, split verbatim from unified_driver.py (code-budget split)."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping


CONFIG_SCHEMA = "epyc.autokernel.unified_driver_config.v1"


class DriverRefused(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise DriverRefused(f"value is not finite canonical JSON: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DriverRefused(f"{label} must be an object with string keys")
    _canonical(value)
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class DriverConfig:
    resolved_campaign_path: str
    store_path: str
    scheduler_config: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    runtime_anchors: Mapping[str, Any]
    runtime_dimensions: Mapping[str, Any]
    profiles: Mapping[str, Any]
    experiment_plans: Mapping[str, Any]
    profile_requests: Mapping[str, Any]
    execution_inputs: Mapping[str, Any]
    native_artifact_sink_ref: str
    config_generation: int
    schema: str = CONFIG_SCHEMA
    kfd_gpu_ids: Mapping[str, str] | None = None  # AKU-07o2, additive (absent = legacy)

    @classmethod
    def from_dict(cls, value: Any) -> "DriverConfig":
        fields = {"schema", "resolved_campaign_path", "store_path", "scheduler_config",
                  "scheduler_state", "runtime_anchors", "runtime_dimensions", "profiles",
                  "experiment_plans", "profile_requests", "native_artifact_sink_ref",
                  "execution_inputs", "config_generation"}
        row = dict(_mapping(value, "driver config"))
        from .kfd_topology import validate_map
        row["kfd_gpu_ids"] = validate_map(row.pop("kfd_gpu_ids", None))
        if set(row) != fields | {"kfd_gpu_ids"} or row.pop("schema") != CONFIG_SCHEMA:
            raise DriverRefused("driver config fields/schema differ")
        for name in ("resolved_campaign_path", "store_path"):
            path = Path(row[name]) if isinstance(row[name], str) else Path()
            if not path.is_absolute():
                raise DriverRefused(f"{name} must be absolute")
            row[name] = str(path)
        if (not isinstance(row["config_generation"], int)
                or isinstance(row["config_generation"], bool)
                or row["config_generation"] < 1):
            raise DriverRefused("config_generation must be positive")
        if not isinstance(row["native_artifact_sink_ref"], str) \
                or not row["native_artifact_sink_ref"]:
            raise DriverRefused("native_artifact_sink_ref must be nonempty")
        for name in ("scheduler_config", "scheduler_state", "runtime_anchors",
                     "runtime_dimensions", "profiles", "experiment_plans",
                     "profile_requests", "execution_inputs"):
            row[name] = _freeze(_mapping(row[name], name))
        return cls(**row)


def load_config_document(path: Path) -> Mapping[str, Any]:
    try:
        named_before = os.stat(path, follow_symlinks=False)
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_nlink != 1 or before.st_size > 4 * 1024 * 1024):
                raise DriverRefused("unified driver config file identity/size is unsafe")
            chunks = []
            remaining = 4 * 1024 * 1024 + 1
            while remaining:
                chunk = os.read(fd, min(remaining, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(fd)
            named_after = os.stat(path, follow_symlinks=False)
            payload = b"".join(chunks)
            before_identity = (before.st_dev, before.st_ino, before.st_mode,
                               before.st_uid, before.st_nlink, before.st_size,
                               before.st_mtime_ns, before.st_ctime_ns)
            after_identity = (after.st_dev, after.st_ino, after.st_mode,
                              after.st_uid, after.st_nlink, after.st_size,
                              after.st_mtime_ns, after.st_ctime_ns)
            named_before_identity = (
                named_before.st_dev, named_before.st_ino, named_before.st_mode,
                named_before.st_uid, named_before.st_nlink, named_before.st_size,
                named_before.st_mtime_ns, named_before.st_ctime_ns)
            named_after_identity = (
                named_after.st_dev, named_after.st_ino, named_after.st_mode,
                named_after.st_uid, named_after.st_nlink, named_after.st_size,
                named_after.st_mtime_ns, named_after.st_ctime_ns)
            if (len(payload) > 4 * 1024 * 1024
                    or before_identity != after_identity
                    or named_before_identity != before_identity
                    or named_after_identity != after_identity):
                raise DriverRefused("unified driver config changed during bounded read")
        finally:
            os.close(fd)
        return _mapping(json.loads(payload.decode("utf-8")), "unified driver config")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverRefused(f"cannot load unified driver config: {exc}") from exc
