"""Bounded, secret-free live telemetry for discovery actors.

The actor transports may contain prompts, source excerpts, credentials, and model
text.  None of those bytes belong on an operator dashboard.  This module writes
only a fixed vocabulary of controller-owned facts to durable JSONL streams.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Mapping


SCHEMA = "epyc.autokernel.discovery_live_event.v1"
CHANNELS = frozenset({"autokernel", "planner"})
EVENTS = frozenset({
    "planner_started", "planner_completed", "planner_failed",
    "critic_started", "critic_completed", "critic_failed",
})
_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,160}")
_SHA = re.compile(r"[0-9a-f]{64}")


class TelemetryError(ValueError):
    pass


def _text(value: object, label: str, *, digest: bool = False) -> str:
    if not isinstance(value, str) or not (_SHA if digest else _ID).fullmatch(value):
        raise TelemetryError(f"invalid {label}")
    return value


class DiscoveryTelemetry:
    """Append allowlisted actor lifecycle facts to two durable streams."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute() or root.is_symlink():
            raise TelemetryError("telemetry root must be an absolute non-symlink path")
        self.root = root

    def emit(self, channel: str, event: str, *, campaign_id: str,
             hypothesis_id: str, provider: str, model: str, effort: str,
             result: Mapping[str, object] | None = None) -> None:
        if channel not in CHANNELS or event not in EVENTS:
            raise TelemetryError("unknown telemetry channel or event")
        row: dict[str, object] = {
            "schema": SCHEMA,
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "channel": channel,
            "event": event,
            "campaign_id": _text(campaign_id, "campaign_id"),
            "hypothesis_id": _text(hypothesis_id, "hypothesis_id"),
            "provider": _text(provider, "provider"),
            "model": _text(model, "model"),
            "effort": _text(effort, "effort"),
        }
        if result is not None:
            if set(result) - {"returncode", "stdout_sha256", "stderr_sha256", "decision"}:
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
            row["result"] = projected
        self.root.mkdir(parents=True, exist_ok=True)
        self._append(self.root / "autokernel.jsonl", row)
        if channel == "planner":
            self._append(self.root / "planner.jsonl", row)

    @staticmethod
    def _append(path: Path, row: Mapping[str, object]) -> None:
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC, 0o640)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.write(fd, encoded) != len(encoded):
                raise OSError("short telemetry write")
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


__all__ = ["CHANNELS", "EVENTS", "SCHEMA", "DiscoveryTelemetry", "TelemetryError"]
