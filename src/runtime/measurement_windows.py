"""Measurement-window guard for optional LOCAL inference (coherence judge, 2026-10-04).

Two windows on this host say "a measurement owns the hardware right now":

* the MI210 GPU window, ``/mnt/raid0/llm/tmp/gpu-window/mi210.json``
  (``src/runtime/gpu_window.py`` documents the protocol): held while
  ``holder`` is anything but ``production``;
* the AutoKernel CPU window, ``/mnt/raid0/llm/autokernel/cpu-window.json``
  (schema ``epyc.autokernel.cpu_window.v1``): held while ``state`` is
  ``closing``/``closed`` or ``loop_holds_claim`` is true, and also while it is
  ``open`` but due to close within :data:`CLOSING_MARGIN_S` (a peer job must
  fit before ``est_close_at``; INC-20260929-peer-window-closing-overrun).

A held window refuses EVERY local model, GPU or CPU: a GPU-resident server still
burns host threads that move the CPU floor, and the CPU window covers the whole
socket. Cloud backends never touch this host's compute and are exempt — the
caller decides that, this module only answers "is a window held".

Failure semantics differ from ``gpu_window`` ON PURPOSE. ``gpu_window`` fails OPEN
because a broken file must never take a serving role down. The judge is an
OPTIONAL extra inference, so here:

* a MISSING file means "no window" (no campaign is publishing one);
* an unreadable / garbled / unknown-schema / unknown-state file means HELD
  (fail closed: a judge call is cheap to refuse and expensive to get wrong);
* a CPU window whose ``expires_at`` has passed is STALE (the loop that writes it
  is dead) and does not hold — the stale fact is still reported in
  :func:`snapshot`.

Paths: ``COHERENCE_JUDGE_GPU_WINDOW_FILE`` / ``COHERENCE_JUDGE_CPU_WINDOW_FILE``
override the defaults. There is deliberately no "off" value.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GPU_WINDOW_DEFAULT = "/mnt/raid0/llm/tmp/gpu-window/mi210.json"
CPU_WINDOW_DEFAULT = "/mnt/raid0/llm/autokernel/cpu-window.json"
GPU_WINDOW_ENV = "COHERENCE_JUDGE_GPU_WINDOW_FILE"
CPU_WINDOW_ENV = "COHERENCE_JUDGE_CPU_WINDOW_FILE"
CPU_WINDOW_SCHEMA = "epyc.autokernel.cpu_window.v1"
CPU_STATES = ("open", "closing", "closed")
#: An ``open`` CPU window closing sooner than this is treated as held.
CLOSING_MARGIN_S = 120.0
RETRY_DEFAULT_S = 60
RETRY_MAX_S = 6 * 3600


@dataclass(frozen=True)
class WindowHold:
    """One held window: which, where, why, and when to retry."""

    window: str  # "gpu" | "cpu"
    path: str
    reason: str
    retry_after_s: int = RETRY_DEFAULT_S
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_ts(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _retry(seconds: float | None) -> int:
    if seconds is None or seconds <= 0:
        return RETRY_DEFAULT_S
    return int(min(RETRY_MAX_S, max(5, round(seconds))))


def gpu_window_path() -> Path:
    return Path(os.environ.get(GPU_WINDOW_ENV) or GPU_WINDOW_DEFAULT)


def cpu_window_path() -> Path:
    return Path(os.environ.get(CPU_WINDOW_ENV) or CPU_WINDOW_DEFAULT)


def _read_json(path: Path) -> tuple[str, dict[str, Any] | None]:
    """("missing"|"unreadable"|"ok", data)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "unreadable", None
    try:
        data = json.loads(text)
    except ValueError:
        return "unreadable", None
    if not isinstance(data, dict):
        return "unreadable", None
    return "ok", data


def gpu_hold(path: Path | None = None, *, now: float | None = None) -> WindowHold | None:
    """The GPU measurement window, or None when production holds the GPU."""
    path = path or gpu_window_path()
    now = time.time() if now is None else now
    status, data = _read_json(path)
    if status == "missing":
        return None
    if status != "ok" or data is None:
        return WindowHold("gpu", str(path), "gpu window file unreadable (fail closed)")
    holder = str(data.get("holder") or "").strip().lower()
    if holder == "production":
        return None
    if holder not in ("autokernel", "released"):
        return WindowHold(
            "gpu", str(path), f"gpu window holder {data.get('holder')!r} unknown (fail closed)"
        )
    end = _parse_ts(data.get("expected_end"))
    retry = 30 if holder == "released" else _retry(end - now if end is not None else None)
    return WindowHold(
        "gpu",
        str(path),
        f"gpu measurement window held (holder={holder})",
        retry,
        {
            "holder": holder,
            "since": data.get("since"),
            "expected_end": data.get("expected_end"),
            "parked_roles": data.get("parked_roles") or [],
            "parked_ports": data.get("parked_ports") or [],
        },
    )


def cpu_hold(path: Path | None = None, *, now: float | None = None) -> WindowHold | None:
    """The AutoKernel CPU window, or None when it is open (or stale/absent)."""
    path = path or cpu_window_path()
    now = time.time() if now is None else now
    status, data = _read_json(path)
    if status == "missing":
        return None
    if status != "ok" or data is None:
        return WindowHold("cpu", str(path), "cpu window file unreadable (fail closed)")
    if data.get("schema") != CPU_WINDOW_SCHEMA:
        return WindowHold(
            "cpu", str(path), f"cpu window schema {data.get('schema')!r} unknown (fail closed)"
        )
    expires = _parse_ts(data.get("expires_at"))
    if expires is not None and now > expires:
        return None  # stale: the loop that heartbeats the window is gone
    state = str(data.get("state") or "").strip().lower()
    if state not in CPU_STATES:
        return WindowHold(
            "cpu", str(path), f"cpu window state {data.get('state')!r} unknown (fail closed)"
        )
    detail = {
        "state": state,
        "phase": data.get("phase"),
        "campaign": data.get("campaign"),
        "loop_holds_claim": bool(data.get("loop_holds_claim")),
        "est_close_at": data.get("est_close_at"),
        "expires_at": data.get("expires_at"),
    }
    if state != "open" or data.get("loop_holds_claim"):
        return WindowHold(
            "cpu",
            str(path),
            f"autokernel cpu window held (state={state}, "
            f"loop_holds_claim={bool(data.get('loop_holds_claim'))})",
            RETRY_DEFAULT_S,
            detail,
        )
    close_at = _parse_ts(data.get("est_close_at"))
    if close_at is not None and close_at - now < CLOSING_MARGIN_S:
        return WindowHold(
            "cpu",
            str(path),
            f"autokernel cpu window closes within {int(CLOSING_MARGIN_S)} s",
            RETRY_DEFAULT_S,
            detail,
        )
    return None


def local_inference_holds(*, now: float | None = None) -> list[WindowHold]:
    """Every held window. Non-empty means: no local judge call now."""
    return [hold for hold in (gpu_hold(now=now), cpu_hold(now=now)) if hold is not None]


def snapshot(*, now: float | None = None) -> dict[str, Any]:
    """Compact per-call record of both windows (for the judge-call log)."""
    out: dict[str, Any] = {}
    for name, fn in (("gpu", gpu_hold), ("cpu", cpu_hold)):
        hold = fn(now=now)
        out[name] = {"held": hold is not None, "reason": hold.reason if hold else None}
    return out


__all__ = [
    "CLOSING_MARGIN_S",
    "CPU_WINDOW_DEFAULT",
    "CPU_WINDOW_ENV",
    "GPU_WINDOW_DEFAULT",
    "GPU_WINDOW_ENV",
    "WindowHold",
    "cpu_hold",
    "gpu_hold",
    "local_inference_holds",
    "snapshot",
]
