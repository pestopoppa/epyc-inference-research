"""Bounded, secret-free live telemetry for discovery actors.

The actor transports may contain prompts, source excerpts, credentials, and model
text.  None of those bytes belong on an operator dashboard.  This module writes
only a fixed vocabulary of controller-owned facts to durable JSONL streams.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA = "epyc.autokernel.discovery_live_event.v2"
LEGACY_SCHEMA = "epyc.autokernel.discovery_live_event.v1"
CHANNELS = frozenset({"autokernel", "planner"})
EVENTS = frozenset({
    "planner_started", "planner_completed", "planner_failed", "planner_refused",
    "critic_started", "critic_completed", "critic_failed",
})
_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,160}")
_SHA = re.compile(r"[0-9a-f]{64}")
_EVENT_ID = re.compile(r"ake-[0-9a-f]{64}")
_TS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z")
_BASE_KEYS = frozenset({
    "schema", "ts", "channel", "event", "campaign_id", "hypothesis_id",
    "provider", "model", "effort",
})
_V2_KEYS = _BASE_KEYS | {"operation_key", "event_id"}


class TelemetryError(ValueError):
    pass


def _text(value: object, label: str, *, digest: bool = False) -> str:
    if not isinstance(value, str) or not (_SHA if digest else _ID).fullmatch(value):
        raise TelemetryError(f"invalid {label}")
    return value


def _project_result(event: str,
                    result: Mapping[str, object] | None) -> dict[str, object] | None:
    if result is None:
        if event in {"planner_completed", "planner_refused", "critic_completed"}:
            raise TelemetryError(
                "completed/refused telemetry lacks its exact typed result")
        return None
    if set(result) - {"returncode", "stdout_sha256", "stderr_sha256",
                      "decision", "refusal_type", "refusal_reason_sha256"}:
        raise TelemetryError("telemetry result contains a non-allowlisted field")
    projected: dict[str, object] = {}
    if "returncode" in result:
        code = result["returncode"]
        if isinstance(code, bool) or not isinstance(code, int):
            raise TelemetryError("returncode must be an integer")
        projected["returncode"] = code
    for key in ("stdout_sha256", "stderr_sha256"):
        if key in result:
            projected[key] = _text(result[key], key, digest=True)
    if "decision" in result:
        decision = result["decision"]
        if decision not in ("accept", "reject", "revise"):
            raise TelemetryError("invalid critic decision")
        projected["decision"] = decision
    if "refusal_type" in result:
        refusal_type = result["refusal_type"]
        if event != "planner_refused" or refusal_type != "planner_output_refusal":
            raise TelemetryError("invalid planner refusal type")
        projected["refusal_type"] = refusal_type
    if "refusal_reason_sha256" in result:
        if event != "planner_refused":
            raise TelemetryError("refusal digest is only valid for planner refusal")
        projected["refusal_reason_sha256"] = _text(
            result["refusal_reason_sha256"],
            "refusal_reason_sha256", digest=True)
    if event == "planner_refused" and set(projected) != {
            "returncode", "stdout_sha256", "stderr_sha256",
            "refusal_type", "refusal_reason_sha256"}:
        raise TelemetryError(
            "planner refusal telemetry lacks its exact typed result")
    if event == "planner_refused" and projected["returncode"] != 0:
        raise TelemetryError("planner refusal telemetry must bind a successful actor exit")
    if (event == "planner_completed"
            and (set(projected) != {
                "returncode", "stdout_sha256", "stderr_sha256"}
                 or projected["returncode"] != 0)):
        raise TelemetryError("planner completion telemetry is invalid")
    if (event == "planner_failed"
            and (set(projected) != {
                "returncode", "stdout_sha256", "stderr_sha256"}
                 or projected["returncode"] == 0)):
        raise TelemetryError("planner failure telemetry is invalid")
    if event == "critic_completed" and set(projected) != {
            "stdout_sha256", "stderr_sha256", "decision"}:
        raise TelemetryError("critic completion telemetry is invalid")
    if event in {"planner_started", "critic_started", "critic_failed"}:
        raise TelemetryError("lifecycle marker cannot carry a result")
    return projected


def _event_id(row: Mapping[str, object]) -> str:
    identity = {key: value for key, value in row.items()
                if key not in {"result", "channel", "event_id", "ts"}}
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode(
            "ascii")).hexdigest()
    return f"ake-{digest}"


def _validate_existing_row(row: Mapping[str, object]) -> None:
    schema = row.get("schema")
    allowed = _V2_KEYS if schema == SCHEMA else _BASE_KEYS
    if schema not in {SCHEMA, LEGACY_SCHEMA}:
        raise TelemetryError("unknown telemetry row schema")
    expected = allowed | ({"result"} if "result" in row else set())
    if set(row) != expected:
        raise TelemetryError("telemetry row has an invalid field set")
    channel = row.get("channel")
    event = row.get("event")
    if channel not in CHANNELS or event not in EVENTS:
        raise TelemetryError("unknown telemetry channel or event")
    expected_channel = "planner" if str(event).startswith("planner_") else "autokernel"
    if channel != expected_channel:
        raise TelemetryError("telemetry event has the wrong channel")
    for key in ("campaign_id", "hypothesis_id", "provider", "model", "effort"):
        _text(row.get(key), key)
    ts = row.get("ts")
    if not isinstance(ts, str) or not _TS.fullmatch(ts):
        raise TelemetryError("invalid telemetry timestamp")
    try:
        parsed_ts = datetime.fromisoformat(ts[:-1] + "+00:00")
    except ValueError as exc:
        raise TelemetryError("invalid telemetry timestamp") from exc
    if parsed_ts.utcoffset() != timezone.utc.utcoffset(parsed_ts):
        raise TelemetryError("invalid telemetry timestamp")
    projected = _project_result(str(event), row.get("result"))
    if ("result" in row and projected != row["result"]):
        raise TelemetryError("telemetry result is not canonical")
    if schema == SCHEMA:
        _text(row.get("operation_key"), "operation_key", digest=True)
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id):
            raise TelemetryError("invalid telemetry event identity")
        if event_id != _event_id(row):
            raise TelemetryError("telemetry event identity does not match its row")


class DiscoveryTelemetry:
    """Append allowlisted actor lifecycle facts to two durable streams."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute() or root.is_symlink():
            raise TelemetryError("telemetry root must be an absolute non-symlink path")
        self.root = root

    def emit(self, channel: str, event: str, *, campaign_id: str,
             hypothesis_id: str, provider: str, model: str, effort: str,
             operation_key: str,
             result: Mapping[str, object] | None = None) -> None:
        if channel not in CHANNELS or event not in EVENTS:
            raise TelemetryError("unknown telemetry channel or event")
        expected_channel = "planner" if event.startswith("planner_") else "autokernel"
        if channel != expected_channel:
            raise TelemetryError("telemetry event has the wrong channel")
        semantic: dict[str, object] = {
            "schema": SCHEMA,
            "channel": channel,
            "event": event,
            "campaign_id": _text(campaign_id, "campaign_id"),
            "hypothesis_id": _text(hypothesis_id, "hypothesis_id"),
            "provider": _text(provider, "provider"),
            "model": _text(model, "model"),
            "effort": _text(effort, "effort"),
        }
        semantic["operation_key"] = _text(
            operation_key, "operation_key", digest=True)
        projected = _project_result(event, result)
        if projected is not None:
            semantic["result"] = projected
        row = {**semantic, "event_id": _event_id(semantic),
               "ts": datetime.now(timezone.utc).isoformat().replace(
                   "+00:00", "Z")}
        self.root.mkdir(parents=True, exist_ok=True)
        paths = [self.root / "autokernel.jsonl"]
        if channel == "planner":
            paths.append(self.root / "planner.jsonl")
        self._append_transaction(paths, row)

    @staticmethod
    def _rows(fd: int, path: Path) -> list[dict[str, Any]]:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TelemetryError(f"unsafe telemetry stream: {path.name}")
        raw = os.pread(fd, info.st_size, 0)
        if not raw:
            return []
        if not raw.endswith(b"\n") or b"\n\n" in raw:
            raise TelemetryError(f"malformed telemetry stream: {path.name}")
        try:
            rows = [json.loads(line) for line in raw.splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TelemetryError(
                f"malformed telemetry stream: {path.name}") from exc
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise TelemetryError(f"malformed telemetry stream: {path.name}")
            _validate_existing_row(row)
            if path.name == "planner.jsonl" and row["channel"] != "planner":
                raise TelemetryError(
                    "planner telemetry stream contains a non-planner row")
            event_id = row.get("event_id")
            if event_id is not None:
                if event_id in seen:
                    raise TelemetryError("duplicate telemetry event identity")
                seen.add(str(event_id))
        return rows

    @staticmethod
    def _write_event(fd: int, encoded: bytes) -> None:
        os.lseek(fd, 0, os.SEEK_END)
        if os.write(fd, encoded) != len(encoded):
            raise OSError("short telemetry write")

    @classmethod
    def _append_transaction(cls, paths: list[Path],
                            row: Mapping[str, object]) -> None:
        fds: list[int] = []
        sizes: list[int] = []
        semantic = {key: value for key, value in row.items() if key != "ts"}
        event_id = row["event_id"]
        try:
            for path in paths:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
                             | os.O_NOFOLLOW, 0o640)
                fds.append(fd)
            for fd in fds:
                fcntl.flock(fd, fcntl.LOCK_EX)
            stream_rows: list[list[dict[str, Any]]] = []
            existing: list[dict[str, Any] | None] = []
            for fd, path in zip(fds, paths, strict=True):
                rows = cls._rows(fd, path)
                stream_rows.append(rows)
                matches = [item for item in rows
                           if item.get("event_id") == event_id]
                if len(matches) > 1:
                    raise TelemetryError("duplicate telemetry event identity")
                if matches and ({key: value for key, value in matches[0].items()
                                 if key != "ts"} != semantic):
                    raise TelemetryError("telemetry event identity collision")
                existing.append(matches[0] if matches else None)
            if len(stream_rows) == 2:
                autokernel_planner = [
                    item for item in stream_rows[0]
                    if item.get("channel") == "planner"
                ]
                planner_projection = stream_rows[1]
                if autokernel_planner != planner_projection:
                    # The old writer and this writer both append the global
                    # stream first.  A process death may therefore leave the
                    # current v2 event as one unmatched tail row.  Only replay
                    # of that exact event may repair it; legacy gaps, order
                    # drift, or an unrelated partial remain corruption.
                    global_has_current_tail = (
                        autokernel_planner[:-1] == planner_projection
                        and autokernel_planner
                        and autokernel_planner[-1].get("event_id") == event_id)
                    projection_has_current_tail = (
                        planner_projection[:-1] == autokernel_planner
                        and planner_projection
                        and planner_projection[-1].get("event_id") == event_id)
                    if not (global_has_current_tail
                            or projection_has_current_tail):
                        raise TelemetryError(
                            "telemetry mirror sequences disagree")
            present = [item for item in existing if item is not None]
            if len(present) > 1 and any(item != present[0]
                                        for item in present[1:]):
                raise TelemetryError("telemetry mirror rows disagree")
            canonical = next((item for item in existing if item is not None),
                             dict(row))
            encoded = (json.dumps(canonical, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("ascii")
            sizes = [os.fstat(fd).st_size for fd in fds]
            for fd, prior in zip(fds, existing, strict=True):
                if prior is None:
                    cls._write_event(fd, encoded)
                    os.fsync(fd)
        except Exception:
            for fd, size in zip(fds, sizes, strict=False):
                try:
                    os.ftruncate(fd, size)
                    os.fsync(fd)
                except OSError:
                    pass
            raise
        finally:
            for fd in reversed(fds):
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


__all__ = [
    "CHANNELS", "EVENTS", "LEGACY_SCHEMA", "SCHEMA", "DiscoveryTelemetry",
    "TelemetryError",
]
