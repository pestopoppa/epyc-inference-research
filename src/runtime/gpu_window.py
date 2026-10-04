"""Parked roles: a GPU model server lent to AutoKernel work (the MI210 window).

Protocol (agreed 2026-10-03 between the stack owner workspace-ec and the
AutoKernel owner workspace-89, operator-directed). One JSON file,
``/mnt/raid0/llm/tmp/gpu-window/mi210.json``::

    {
      "holder": "production" | "autokernel" | "released",
      "since": "<iso8601>",
      "expected_end": "<iso8601>",
      "preempt_requested_at": "<iso8601>" | null,
      "preempt_reason": "<text>" | null,
      "parked_roles": ["architect_critic", ...],
      "parked_ports": [8083, ...]
    }

* The STACK-OWNED EXECUTOR (``src/runtime/gpu_window_executor.py``, G1) drains,
  stops the server and writes ``holder=autokernel`` with the parked roles/ports and
  ``grant_state`` (draining -> granted -> restoring); on close it proves serving and
  only then writes ``holder=production``. Its cron watchdog restores at
  ``expected_end`` + 10 min even with AutoKernel and the bus dead.
* The AUTOKERNEL campaign NEVER writes this file. It checks
  ``preempt_requested_at`` at each batch boundary, drains, and releases by dropping
  its ``gpu_device.mi210_0`` flock; the executor's watchdog sees that and restores.
* The ORCHESTRATOR (this module) refuses requests that resolve to a parked role
  or port with a fast, explicit ``role_parked`` error, and a real request calls
  :func:`request_preempt` so the drain starts on demand.

A role is parked while ``holder`` is anything but ``production`` and it is named
in ``parked_roles`` or every port it is served on is in ``parked_ports``
(``released`` = AK has drained but the server is not back yet; it still cannot
serve).

Reads FAIL OPEN: a missing, unreadable or garbled file means "not parked" (the
garbled case is logged once per file version) — a broken window file must never
take a serving role down. The read is a ``stat`` plus a JSON parse keyed on
(mtime, size, inode), so the per-request check costs one ``stat``.

``ORCHESTRATOR_GPU_WINDOW_FILE`` overrides the path; ``off`` disables the check.

CLI (stack owner)::

    python -m src.runtime.gpu_window park --roles architect_critic,coder_escalation,\
ingest_long_context --ports 8083 --holder autokernel --expected-end +45m
    python -m src.runtime.gpu_window restore     # = gpu_window_executor close

Windows for AutoKernel are opened by the G1 executor
(``python -m src.runtime.gpu_window_executor open ...``), the only writer of this
file besides the orchestrator's ``request_preempt``. ``park`` refuses while the
holder is not ``production``; ``--expected-end`` is capped at +60 min; ``restore``
runs the executor's serving proof before writing ``holder=production``.
    python -m src.runtime.gpu_window status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - fcntl exists on every supported host
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_PATH = "/mnt/raid0/llm/tmp/gpu-window/mi210.json"
PATH_ENV = "ORCHESTRATOR_GPU_WINDOW_FILE"
_DISABLED_VALUES = {"off", "0", "none", "false", "disabled"}

HOLDERS = ("production", "autokernel", "released")
ERROR_TYPE = "role_parked"

#: ``retry_after_s`` bounds. Unknown/overdue end -> the default; ``released`` (AK
#: drained, stack owner restoring) -> the short restore estimate.
RETRY_DEFAULT_S = 60
RETRY_RELEASED_S = 30
RETRY_MIN_S = 5
RETRY_MAX_S = 6 * 3600

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"key": None, "window": None}
_warned_keys: set[tuple] = set()


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value: Any) -> float | None:
    """ISO 8601 string or epoch seconds -> epoch seconds; None when unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


_REL_RE = re.compile(r"^\+(\d+(?:\.\d+)?)([smhd]?)$")


def parse_expected_end(value: str, *, now: float | None = None) -> str:
    """CLI value -> ISO 8601: ``+90m`` / ``+4h`` / ``+3600`` / ISO / epoch."""
    now = time.time() if now is None else now
    text = value.strip()
    match = _REL_RE.match(text)
    if match:
        amount = float(match.group(1))
        unit = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
        ts = now + amount * unit
    else:
        parsed = _parse_ts(float(text) if re.fullmatch(r"\d+(?:\.\d+)?", text) else text)
        if parsed is None:
            raise ValueError(f"unparseable --expected-end {value!r} (use +4h, +90m, ISO 8601)")
        ts = parsed
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def _norm_role(role: Any) -> str:
    value = getattr(role, "value", role)
    return str(value or "").strip()


def _norm_ports(values: Any) -> tuple[int, ...]:
    out: list[int] = []
    for value in values or ():
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if port > 0:
            out.append(port)
    return tuple(out)


@dataclass(frozen=True)
class GpuWindow:
    """One parsed window file."""

    holder: str
    since: str | None = None
    expected_end: str | None = None
    preempt_requested_at: str | None = None
    preempt_reason: str | None = None
    parked_roles: tuple[str, ...] = ()
    parked_ports: tuple[int, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GpuWindow":
        holder = str(data.get("holder") or "").strip().lower()
        if holder not in HOLDERS:
            raise ValueError(f"holder {data.get('holder')!r} not in {HOLDERS}")
        roles = data.get("parked_roles") or []
        if not isinstance(roles, list):
            raise ValueError("parked_roles must be a list")
        ports = data.get("parked_ports") or []
        if not isinstance(ports, list):
            raise ValueError("parked_ports must be a list")
        return cls(
            holder=holder,
            since=data.get("since"),
            expected_end=data.get("expected_end"),
            preempt_requested_at=data.get("preempt_requested_at") or None,
            preempt_reason=data.get("preempt_reason") or None,
            parked_roles=tuple(r for r in (_norm_role(x) for x in roles) if r),
            parked_ports=_norm_ports(ports),
            raw=dict(data),
        )

    @property
    def active(self) -> bool:
        """True while production does NOT hold the GPU."""
        return self.holder != "production"

    def retry_after_s(self, now: float | None = None) -> int:
        if self.holder == "released":
            return RETRY_RELEASED_S
        end = _parse_ts(self.expected_end)
        if end is None:
            return RETRY_DEFAULT_S
        remaining = end - (time.time() if now is None else now)
        if remaining <= 0:
            return RETRY_DEFAULT_S  # overdue: the window should be closing
        return int(min(RETRY_MAX_S, max(RETRY_MIN_S, round(remaining))))


@dataclass(frozen=True)
class ParkedInfo:
    """Why a role/port is refused right now."""

    role: str | None
    port: int | None
    holder: str
    retry_after_s: int
    expected_end: str | None
    since: str | None
    preempt_requested_at: str | None
    window_file: str
    matched_by: str  # "role" | "port"

    def refusal(self, **extra: Any) -> dict[str, Any]:
        """Structured ``refusal`` block (serving-call record, HTTP body)."""
        out: dict[str, Any] = {
            "gate": ERROR_TYPE,
            "holder": self.holder,
            "role": self.role,
            "port": self.port,
            "retry_after_s": self.retry_after_s,
            "expected_end": self.expected_end,
            "since": self.since,
            "matched_by": self.matched_by,
            "preempt_requested_at": self.preempt_requested_at,
            "window_file": self.window_file,
        }
        out.update({k: v for k, v in extra.items() if v is not None})
        return out


def window_path() -> Path | None:
    """The window file, or None when the check is disabled."""
    override = os.environ.get(PATH_ENV, "").strip()
    if override.lower() in _DISABLED_VALUES:
        return None
    return Path(override or DEFAULT_PATH)


def _warn_once(key: tuple, msg: str, *args: Any) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(msg, *args)


def read_window(path: Path | None = None) -> GpuWindow | None:
    """The current window, or None (missing / unreadable / garbled / disabled).

    Fail-open: any defect reads as "no window" and is logged once per file
    version. Cached on (path, mtime_ns, size, inode).
    """
    path = window_path() if path is None else path
    if path is None:
        return None
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        _warn_once((str(path), "stat", type(exc).__name__),
                   "gpu_window: cannot stat %s (%s) — treating as not parked", path, exc)
        return None
    key = (str(path), st.st_mtime_ns, st.st_size, st.st_ino)
    with _cache_lock:
        if _cache["key"] == key:
            return _cache["window"]
    window: GpuWindow | None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("top level is not a JSON object")
        window = GpuWindow.from_dict(data)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        _warn_once(key, "gpu_window: %s is garbled (%s) — treating as not parked", path, exc)
        window = None
    with _cache_lock:
        _cache["key"] = key
        _cache["window"] = window
    return window


def _role_ports(role: str) -> tuple[int, ...]:
    """Ports the role is served on, from the config ``server_urls`` map."""
    try:
        from urllib.parse import urlparse

        from src.config import get_config

        url = str(get_config().server_urls.as_dict().get(role) or "")
    except Exception:
        return ()
    ports = []
    for part in url.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            port = urlparse(part).port
        except ValueError:
            port = None
        if port:
            ports.append(int(port))
    return tuple(ports)


def parked_info(
    role: Any = None,
    port: int | None = None,
    *,
    window: GpuWindow | None = None,
) -> ParkedInfo | None:
    """``ParkedInfo`` when ``role`` or ``port`` is parked right now, else None.

    A role matches by name (``parked_roles``) or when every port it is served on
    is parked (aliases sharing the parked server follow it). A port matches
    ``parked_ports`` exactly. Never raises.
    """
    try:
        if window is None:
            window = read_window()
        if window is None or not window.active:
            return None
        role_name = _norm_role(role) or None
        matched_by = None
        if port is not None and int(port) in window.parked_ports:
            matched_by = "port"
        elif role_name and role_name in window.parked_roles:
            matched_by = "role"
        elif role_name and window.parked_ports:
            ports = _role_ports(role_name)
            if ports and all(p in window.parked_ports for p in ports):
                matched_by = "port"
                port = ports[0] if port is None else port
        if matched_by is None:
            return None
        path = window_path()
        return ParkedInfo(
            role=role_name,
            port=int(port) if port is not None else None,
            holder=window.holder,
            retry_after_s=window.retry_after_s(),
            expected_end=window.expected_end,
            since=window.since,
            preempt_requested_at=window.preempt_requested_at,
            window_file=str(path) if path is not None else "",
            matched_by=matched_by,
        )
    except Exception:  # fail open
        logger.debug("gpu_window: parked_info failed", exc_info=True)
        return None


def is_parked(target: Any = None, *, role: Any = None, port: int | None = None) -> bool:
    """``is_parked("architect_critic")`` / ``is_parked(8083)`` / keyword forms."""
    if target is not None:
        if isinstance(target, int) and not isinstance(target, bool):
            port = target
        elif isinstance(target, str) and target.isdigit():
            port = int(target)
        else:
            role = target
    return parked_info(role=role, port=port) is not None


# ---------------------------------------------------------------------------
# Writes (atomic tmp + rename under a sidecar flock)
# ---------------------------------------------------------------------------


class _Locked:
    """Exclusive flock on ``<window>.lock`` for read-modify-write."""

    def __init__(self, path: Path) -> None:
        self.lock_path = path.with_name(path.name + ".lock")
        self.fh = None

    def __enter__(self) -> "_Locked":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.lock_path, "a")
        if fcntl is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.fh is not None:
            if fcntl is not None:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()


def _read_raw(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def request_preempt(
    reason: str,
    *,
    role: Any = None,
    request_id: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Ask the AutoKernel holder to drain. Idempotent; never raises.

    Sets ``preempt_requested_at`` / ``preempt_reason`` / ``preempt_requested_by``
    only when the window is held by ``autokernel`` and no preempt is pending.
    Returns ``{"requested": bool, "status": ...}`` where status is ``requested``
    (this call set it), ``already_requested``, ``not_held`` (production /
    released / no window) or ``error``.
    """
    path = window_path() if path is None else path
    if path is None:
        return {"requested": False, "status": "disabled"}
    try:
        with _Locked(path):
            data = _read_raw(path)
            if data is None:
                return {"requested": False, "status": "not_held"}
            holder = str(data.get("holder") or "").lower()
            if holder != "autokernel":
                return {"requested": False, "status": "not_held", "holder": holder}
            if data.get("preempt_requested_at"):
                return {
                    "requested": False,
                    "status": "already_requested",
                    "preempt_requested_at": data.get("preempt_requested_at"),
                }
            stamp = _now_iso()
            data["preempt_requested_at"] = stamp
            data["preempt_reason"] = str(reason)[:500]
            data["preempt_requested_by"] = {
                "role": _norm_role(role) or None,
                "request_id": request_id,
                "pid": os.getpid(),
                "source": "orchestrator",
            }
            _atomic_write(path, data)
            logger.warning(
                "gpu_window: preempt requested (%s) role=%s request_id=%s", reason, role, request_id
            )
            return {"requested": True, "status": "requested", "preempt_requested_at": stamp}
    except Exception as exc:
        logger.warning("gpu_window: preempt request failed: %s", exc)
        return {"requested": False, "status": "error", "error": str(exc)[:200]}


class WindowHeld(ValueError):
    """A second park while the window is not ``production`` (one window at a time)."""


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def park(
    *,
    roles: Iterable[str],
    ports: Iterable[int],
    holder: str = "autokernel",
    expected_end: str | None = None,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stack owner: write a fresh window (clears any previous preempt).

    One window at a time: refuses (``WindowHeld``) unless the file is missing or
    reads ``holder=production``. The G1 executor
    (``src/runtime/gpu_window_executor.py``) is the normal caller; the CLI
    ``park`` takes the executor's lease lock first.
    """
    if holder not in ("autokernel", "released"):
        raise ValueError("park --holder must be autokernel or released")
    path = window_path() if path is None else path
    if path is None:
        raise RuntimeError(f"{PATH_ENV} disables the window file")
    data = {
        "holder": holder,
        "since": _now_iso(),
        "expected_end": expected_end,
        "preempt_requested_at": None,
        "preempt_reason": None,
        "parked_roles": sorted({_norm_role(r) for r in roles if _norm_role(r)}),
        "parked_ports": sorted(set(_norm_ports(ports))),
        "written_by": "stack_owner",
        "boot_id": _boot_id(),
    }
    data.update(extra or {})
    if not data["parked_roles"] and not data["parked_ports"]:
        raise ValueError("park needs at least one --roles or --ports entry")
    with _Locked(path):
        current = _read_raw(path)
        if current is not None and str(current.get("holder") or "").lower() != "production":
            raise WindowHeld(
                f"window already held (holder={current.get('holder')}); "
                "one window at a time — close it first"
            )
        _atomic_write(path, data)
    return data


def restore(*, path: Path | None = None) -> dict[str, Any]:
    """Stack owner: production holds the GPU again; nothing is parked."""
    path = window_path() if path is None else path
    if path is None:
        raise RuntimeError(f"{PATH_ENV} disables the window file")
    with _Locked(path):
        previous = _read_raw(path) or {}
        data = {
            "holder": "production",
            "since": _now_iso(),
            "expected_end": None,
            "preempt_requested_at": None,
            "preempt_reason": None,
            "parked_roles": [],
            "parked_ports": [],
            "written_by": "stack_owner",
            "previous_holder": previous.get("holder"),
        }
        _atomic_write(path, data)
    return data


def status(*, path: Path | None = None) -> dict[str, Any]:
    """Raw file + the parsed verdict (CLI ``status``)."""
    path = window_path() if path is None else path
    if path is None:
        return {"enabled": False}
    raw = _read_raw(path) if path.exists() else None
    window = read_window(path)
    out: dict[str, Any] = {
        "enabled": True,
        "path": str(path),
        "exists": path.exists(),
        "valid": window is not None,
        "raw": raw,
    }
    if window is not None:
        out.update(
            holder=window.holder,
            active=window.active,
            parked_roles=list(window.parked_roles) if window.active else [],
            parked_ports=list(window.parked_ports) if window.active else [],
            retry_after_s=window.retry_after_s() if window.active else 0,
        )
    return out


# ---------------------------------------------------------------------------
# Orchestrator choke-point helper
# ---------------------------------------------------------------------------


def refuse_if_parked(
    role: Any = None,
    port: int | None = None,
    *,
    request_id: str | None = None,
    base_url: str | None = None,
    preempt: bool = True,
    record: bool = True,
    caller: dict[str, Any] | None = None,
    method: str = "refused",
) -> None:
    """Raise ``RoleParkedError`` when ``role``/``port`` is parked; else return.

    On refusal: requests a preempt (``preempt=True``: a real request starts the
    drain) and writes a ``serving_call.v1`` record with ``outcome="refused"``
    (``record=True``; the backend layer passes False because its own wrapper
    records the raise). The not-parked path is one cached ``stat``.
    """
    info = parked_info(role=role, port=port)
    if info is None:
        return
    from src.exceptions import RoleParkedError

    preempt_result = None
    if preempt:
        role_label = info.role or (f"port {info.port}" if info.port else "parked role")
        preempt_result = request_preempt(
            f"{role_label} request {request_id or 'unknown'}",
            role=info.role,
            request_id=request_id,
        )
    exc = RoleParkedError.from_info(info, request_id=request_id, preempt=preempt_result)
    if record:
        try:
            from src.backends import serving_calls

            serving_calls.record_refusal(
                role=info.role,
                base_url=base_url or (f"http://localhost:{info.port}" if info.port else None),
                refusal=exc.refusal,
                caller=caller or {"source": "orchestrator", "request_id": request_id},
                method=method,
                exc=exc,
            )
        except Exception:
            logger.debug("gpu_window: refusal record failed", exc_info=True)
    raise exc


_SENTINEL_RE = re.compile(
    r"role_parked: role=(?P<role>\S*) port=(?P<port>\S*) holder=(?P<holder>\S+) "
    r"retry_after_s=(?P<retry>\d+)"
)


def parse_parked_sentinel(text: str | None) -> dict[str, Any] | None:
    """Recover the structured refusal from an in-band ``[ERROR: role_parked: ...]``."""
    if not text or ERROR_TYPE not in text:
        return None
    match = _SENTINEL_RE.search(text)
    if not match:
        return None
    port = match.group("port")
    return {
        "gate": ERROR_TYPE,
        "role": match.group("role") or None,
        "port": int(port) if port.isdigit() else None,
        "holder": match.group("holder"),
        "retry_after_s": int(match.group("retry")),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_csv(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.runtime.gpu_window",
        description="Park / restore GPU roles lent to AutoKernel (MI210 window file).",
    )
    parser.add_argument("--file", help=f"window file (default ${PATH_ENV} or {DEFAULT_PATH})")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_park = sub.add_parser("park", help="write holder=autokernel with the parked roles/ports")
    p_park.add_argument("--roles", action="append", default=[], help="comma-separated roles")
    p_park.add_argument("--ports", action="append", default=[], help="comma-separated ports")
    p_park.add_argument("--holder", default="autokernel", choices=("autokernel", "released"))
    p_park.add_argument("--expected-end", required=True,
                        help="ISO 8601, epoch seconds or relative (+45m); at most +60m")
    sub.add_parser("restore", help="executor close: serving proof, then holder=production")
    sub.add_parser("status", help="print the window and the parsed verdict")
    args = parser.parse_args(argv)
    path = Path(args.file) if args.file else None

    try:
        if args.cmd in ("park", "restore"):
            # Manual park/restore go through the executor's lease (one window at a
            # time; restore = close with the serving proof).
            from src.runtime import gpu_window_executor as gwe

            window = gwe._window_path(path)
            if args.cmd == "restore":
                out = gwe.Executor(window, gwe.live_ops()).close(reason="manual_cli")
            else:
                ports = []
                for value in _split_csv(args.ports):
                    if not value.isdigit():
                        parser.error(f"--ports: {value!r} is not a port number")
                    ports.append(int(value))
                end = parse_expected_end(args.expected_end)
                end_ts = _parse_ts(end) or 0.0
                if end_ts - time.time() > gwe.MAX_WINDOW_S + 5:
                    raise ValueError(f"--expected-end beyond +{gwe.MAX_WINDOW_S // 60} min "
                                     "needs the operator")
                with gwe.LeaseLock(window):
                    out = park(roles=_split_csv(args.roles), ports=ports, holder=args.holder,
                               expected_end=end, path=window)
        else:
            out = status(path=path)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
