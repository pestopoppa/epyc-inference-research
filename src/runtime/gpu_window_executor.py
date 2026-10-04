"""G1 executor: the stack-owned MI210 window handover (park -> grant -> restore).

The ONLY writer of the window file (``src/runtime/gpu_window.py``) for windows lent
to AutoKernel. AutoKernel never writes it: AK asks with a ``compute-request`` routed
to the coordinator-daemon, holds the ``gpu_device.mi210_0`` flock while it runs, and
releases by dropping that flock. This module runs the sequence, nothing else:

``open``  (stack-owner session, or an operator token — never a bus message alone)
    1. refuse unless: expected_end <= now + 60 min; no stack change pending
       (machine-readable marker); holder == production and no open lease (one
       window at a time); a valid schedule entry covers the window; the AK device
       flock is free.
    2. write the window ``grant_state=draining`` (parked roles answer 503
       ``role_parked`` from here on; AK must not start), then drain via ``/slots``:
       wait up to 10 min for every slot to go idle. On timeout REFUSE — the window
       goes straight back to production; nothing is stopped, nothing is killed.
    3. ``orchestrator_stack.py stop <component>``; prove the ports are free; write
       ``grant_state=granted``. AK may now take the device flock.

``close`` (stack owner, operator, or the watchdog)
    refuse while the device is BUSY (see below); mark ``grant_state=restoring``; for
    every parked port run the serving proof (``/health`` 200, ``/v1/models``
    non-empty, a real ``/v1/completions`` with non-empty text, and the live pid's
    ggml libraries all mapped from its own binary dir with ``libggml-hip`` resident);
    ``orchestrator_stack.py reload`` only the components whose port fails, then
    prove again. ``holder=production`` is written only after every port proves.

``tick``  (the durable watchdog; cron every minute plus ``@reboot``, see
    ``scripts/server/gpu_window_watchdog.cron``). Needs neither AK nor the bus:
    * after a reboot (boot_id changed) a non-production window is neutralised
      (``holder=released``, ``grant_state=restoring``) and flips to production as
      soon as the serving proof passes — it never starts servers on its own then;
    * at ``expected_end + 10 min`` (or once AK released the device after holding
      it, or released after a preempt request) it runs ``close``;
    * writes ``<window>.executor-status.json`` — the hub's freshness envelope.

Manual ``gpu_window park`` takes the same lease lock and refuses a second window;
manual ``gpu_window restore`` is ``close`` (serving proof included).

DEVICE BUSY (``Executor.device_busy``) — open, close and every watchdog close refuse
while ANY of these holds; the watchdog logs, records the refusal in the lease, raises
the hub alarm once overdue, and retries on the next tick — it never reloads onto it:
    (a) any KFD process (``/sys/class/kfd/kfd/proc/<pid>``, the list
        ``rocm-smi --showpids`` prints) other than the listener of a parked port
        (that is the server being restored); an unreadable list counts as busy;
    (b) gpu-quiet held EXCLUSIVE (``gpu_quiet_lock.holders``; held with an unknown
        mode, or unreadable, counts as busy);
    (c) the AK device flock ``gpu_device.mi210_0.lock``.

LEGACY / NO-LEASE PARKS (a park this executor did not open: written by the
``gpu_window park`` CLI or by hand, no lease tracking it) are never auto-closed by
the watchdog before ``expected_end + RESTORE_GRACE_S`` (never, without an
expected_end), and then only through ``close`` — device-free plus the same serving
proof. Holding one is logged at WARNING every tick; closing one at ERROR.

Env: ``ORCHESTRATOR_GPU_WINDOW_FILE`` (window), ``ORCHESTRATOR_GPU_WINDOW_SCHEDULE``
(schedule), ``ORCHESTRATOR_STACK_CHANGE_PENDING_FILE`` (marker),
``ORCHESTRATOR_GPU_WINDOW_OPERATOR_TOKENS`` (sha256 per line, operator-written; the
file must be 0600 in a 0700 dir, both owned by the executor's uid, or it is refused),
``ORCHESTRATOR_STACK_OWNER_SESSION`` (default ``workspace-ec``),
``ORCHESTRATOR_TMP_DIR`` (device flock + gpu-quiet root),
``ORCHESTRATOR_KFD_PROC_DIR`` (KFD process list, default ``/sys/class/kfd/kfd/proc``).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from src.runtime import gpu_window as gw

logger = logging.getLogger(__name__)

MAX_WINDOW_S = 3600
RESTORE_GRACE_S = 600
DRAIN_TIMEOUT_S = 600
DRAIN_POLL_S = 5.0
MAX_RELOAD_ATTEMPTS = 2
DEVICE_ID = "mi210_0"
WRITER = "gpu_window_executor"
STATUS_SCHEMA = "epyc.orchestrator.gpu_window_executor_status.v1"
LEASE_SCHEMA = "epyc.orchestrator.gpu_window_lease.v1"
SCHEDULE_SCHEMA = "epyc.gpu_window_schedule.v1"
PENDING_SCHEMA = "epyc.orchestrator.stack_change_pending.v1"
LEGACY_ORIGIN = "legacy_park"
KFD_PROC_ENV = "ORCHESTRATOR_KFD_PROC_DIR"
DEFAULT_KFD_PROC = "/sys/class/kfd/kfd/proc"
MAX_HISTORY = 200

SCHEDULE_ENV = "ORCHESTRATOR_GPU_WINDOW_SCHEDULE"
PENDING_ENV = "ORCHESTRATOR_STACK_CHANGE_PENDING_FILE"
TOKENS_ENV = "ORCHESTRATOR_GPU_WINDOW_OPERATOR_TOKENS"
OWNER_ENV = "ORCHESTRATOR_STACK_OWNER_SESSION"
DEFAULT_STACK_OWNER = "workspace-ec"
# F3: never under the world-writable /mnt/raid0/llm/tmp — a private 0700 dir, file 0600,
# both owned by the executor's uid (checked on every read, see _read_private_tokens).
DEFAULT_TOKENS = "/mnt/raid0/llm/private/gpu-window/operator_tokens.sha256"
ORCH_ROOT = Path(__file__).resolve().parents[2]
PROOF_PROMPT = "The capital of France is"


def default_components(ports: Iterable[int]) -> list[str]:
    """The stack component that owns each port: ``server_<port>`` (stack state key)."""
    return [f"server_{int(p)}" for p in ports]


class WindowRefused(RuntimeError):
    """The executor declined; nothing was stopped or killed."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _window_path(path: Path | None = None) -> Path:
    path = gw.window_path() if path is None else path
    if path is None:
        raise WindowRefused("window_disabled", f"{gw.PATH_ENV} disables the window file")
    return path


def lease_lock_path(window: Path) -> Path:
    return window.with_name(window.name + ".executor.lock")


def lease_path(window: Path) -> Path:
    return window.with_name(window.name + ".lease.json")


def status_path(window: Path) -> Path:
    return window.with_name(window.name + ".executor-status.json")


def schedule_path(window: Path) -> Path:
    return Path(os.environ.get(SCHEDULE_ENV) or window.with_name("schedule.json"))


def _tmp_dir() -> Path:
    for name in ("ORCHESTRATOR_TMP_DIR", "ORCHESTRATOR_PATHS_TMP_DIR"):
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value)
    return Path("/mnt/raid0/llm/tmp")


def pending_marker_path() -> Path:
    return Path(os.environ.get(PENDING_ENV) or (_tmp_dir() / "stack_change_pending.json"))


def device_lock_path(device_id: str = DEVICE_ID) -> Path:
    """Same path as AK ``resource/device_claim.device_lock_path`` (same env roots)."""
    return _tmp_dir() / f"gpu_device.{device_id}.lock"


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Lease lock: one executor transition at a time
# ---------------------------------------------------------------------------


class LeaseLock:
    """Non-blocking exclusive flock on ``<window>.executor.lock``."""

    def __init__(self, window: Path) -> None:
        self.path = lease_lock_path(window)
        self.fh = None

    def __enter__(self) -> "LeaseLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            self.fh = None
            raise WindowRefused("executor_busy", "another window transition holds the lease lock")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.fh is not None:
            self.fh.close()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    gw._atomic_write(path, data)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stack-change-pending marker
# ---------------------------------------------------------------------------


def pending_stack_change() -> dict[str, Any] | None:
    """The marker body when a stack change is pending apply/bring-up, else None.

    Fails CLOSED: an unreadable marker still means pending.
    """
    path = pending_marker_path()
    if not path.exists():
        return None
    return _read_json(path) or {"unreadable": True, "path": str(path)}


def set_pending(change_id: str, phase: str, *, by: str) -> dict[str, Any]:
    if phase not in ("apply", "bring-up"):
        raise ValueError("phase must be apply or bring-up")
    data = {"schema": PENDING_SCHEMA, "change_id": change_id, "phase": phase,
            "set_by": by, "set_at": _iso(time.time())}
    _write_json(pending_marker_path(), data)
    return data


def clear_pending() -> bool:
    try:
        pending_marker_path().unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Schedule file + validator
# ---------------------------------------------------------------------------


def validate_schedule(data: Any) -> list[str]:
    """Errors in a schedule document (empty list = valid)."""
    if not isinstance(data, dict):
        return ["top level is not an object"]
    errors: list[str] = []
    if data.get("schema") != SCHEDULE_SCHEMA:
        errors.append(f"schema must be {SCHEDULE_SCHEMA}")
    entries = data.get("entries")
    if not isinstance(entries, list):
        return errors + ["entries must be a list"]
    seen: set[str] = set()
    spans: dict[str, list[tuple[float, float, str]]] = {}
    for i, entry in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}: not an object")
            continue
        eid = str(entry.get("id") or "")
        if not eid:
            errors.append(f"{where}: missing id")
        elif eid in seen:
            errors.append(f"{where}: duplicate id {eid!r}")
        seen.add(eid)
        for key in ("device_id", "holder", "campaign_id", "purpose"):
            if not entry.get(key):
                errors.append(f"{where}: missing {key}")
        if entry.get("holder") not in (None, "autokernel"):
            errors.append(f"{where}: holder must be autokernel")
        start, end = gw._parse_ts(entry.get("start")), gw._parse_ts(entry.get("end"))
        if start is None or end is None:
            errors.append(f"{where}: start/end must be ISO 8601")
            continue
        if end <= start:
            errors.append(f"{where}: end must be after start")
        elif end - start > MAX_WINDOW_S and entry.get("operator_approved") is not True:
            errors.append(f"{where}: longer than {MAX_WINDOW_S}s needs operator_approved: true")
        spans.setdefault(str(entry.get("device_id")), []).append((start, end, eid))
    for device, items in spans.items():
        items.sort()
        for (s1, e1, a), (s2, _e2, b) in zip(items, items[1:]):
            if s2 < e1:
                errors.append(f"{device}: entries {a!r} and {b!r} overlap")
    return errors


def resolve_schedule_entry(path: Path, ref: str, start: float, end: float,
                           device_id: str = DEVICE_ID) -> dict[str, Any]:
    data = _read_json(path)
    if data is None:
        raise WindowRefused("schedule_missing", str(path))
    errors = validate_schedule(data)
    if errors:
        raise WindowRefused("schedule_invalid", "; ".join(errors[:5]))
    for entry in data["entries"]:
        if entry.get("id") != ref:
            continue
        if entry.get("device_id") != device_id:
            raise WindowRefused("schedule_mismatch", f"{ref} is for {entry.get('device_id')}")
        s, e = gw._parse_ts(entry["start"]), gw._parse_ts(entry["end"])
        if not (s <= start + 60 and end <= e):  # 1-min slack on the start edge
            raise WindowRefused("schedule_mismatch",
                                f"{ref} [{entry['start']}, {entry['end']}] does not cover the window")
        return entry
    raise WindowRefused("schedule_unscheduled", f"no entry {ref!r} in {path}")


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Authority:
    """Who may open a window: the stack-owner session or an operator token.

    A bus message is never one of these. ``compute_grant`` records the
    coordinator-daemon ``compute-grant`` id the request was routed through.
    """

    kind: str  # "stack_owner" | "operator_token" | "watchdog"
    principal: str
    compute_grant: str | None = None

    def record(self) -> dict[str, Any]:
        return {"kind": self.kind, "principal": self.principal,
                "compute_grant": self.compute_grant}


def _read_private_tokens(path: Path) -> set[str]:
    """Read the operator token digests, refusing any file another user could have written.

    Fail-closed: a missing, symlinked, foreign-owned, group/other-accessible file, or one
    whose directory is group/other-writable, yields no tokens (F3).
    """
    try:
        dir_info = os.stat(path.parent)
        if dir_info.st_uid != os.getuid() or dir_info.st_mode & 0o022:
            raise WindowRefused("unsafe_token_file",
                                f"{path.parent} must be owned by uid {os.getuid()} and 0700")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise WindowRefused("unsafe_token_file", f"{path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise WindowRefused("unsafe_token_file",
                                f"{path} must be a regular 0600 file owned by uid {os.getuid()}")
        with os.fdopen(fd, "r", closefd=False) as fh:
            text = fh.read()
    finally:
        os.close(fd)
    return {ln.strip().lower() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")}


def authorize(*, stack_owner_session: str | None, operator_token: str | None,
              compute_grant: str | None) -> Authority:
    if operator_token:
        digest = hashlib.sha256(operator_token.strip().encode()).hexdigest()
        tokens_file = Path(os.environ.get(TOKENS_ENV) or DEFAULT_TOKENS)
        allowed = _read_private_tokens(tokens_file)
        if digest not in allowed:
            raise WindowRefused("unauthorized", "operator token not in the operator token file")
        return Authority("operator_token", f"sha256:{digest[:12]}", compute_grant)
    owner = os.environ.get(OWNER_ENV) or DEFAULT_STACK_OWNER
    if stack_owner_session and stack_owner_session == owner:
        if not compute_grant:
            raise WindowRefused("unauthorized",
                                "stack-owner open needs the coordinator-daemon --compute-grant id")
        return Authority("stack_owner", owner, compute_grant)
    raise WindowRefused("unauthorized",
                        "open needs --stack-owner-session <owner> or --operator-token")


# ---------------------------------------------------------------------------
# Device-busy readers: KFD process list + gpu-quiet exclusive holder
# ---------------------------------------------------------------------------


def kfd_proc_dir() -> Path:
    return Path(os.environ.get(KFD_PROC_ENV) or DEFAULT_KFD_PROC)


def kfd_pids() -> list[int] | None:
    """PIDs with an open KFD context — the list ``rocm-smi --showpids`` reads.

    sysfs is not pid-namespaced, so this works from the container the watchdog runs
    in. None when the list cannot be read (callers treat that as busy).
    """
    try:
        entries = list(kfd_proc_dir().iterdir())
    except OSError:
        return None
    return sorted(int(e.name) for e in entries if e.name.isdigit())


def gpu_quiet_exclusive() -> list[str] | None:
    """Who holds gpu-quiet EXCLUSIVE ([] when free or only shared); None if unreadable.

    Uses the region-lock module's realized-first holder reader
    (``gpu_quiet_lock.holders``). Held with an unknown mode (``/proc/locks``
    unreadable) is reported as a holder: cannot prove it is shared.
    """
    try:
        from src.runtime.gpu_quiet_lock import GPU_QUIET_EXCLUSIVE, holders

        info = holders()
    except Exception:  # noqa: BLE001 - fail closed
        logger.warning("gpu_window_executor: gpu-quiet holders unreadable", exc_info=True)
        return None
    if not info.get("held"):
        return []
    mode = info.get("mode")
    if mode not in (None, GPU_QUIET_EXCLUSIVE):
        return []
    who = [f"{r.get('role') or '?'}(pid {r.get('pid')})"
           for r in info.get("holders") or []
           if isinstance(r, dict) and r.get("mode") in (None, GPU_QUIET_EXCLUSIVE)]
    if mode is None:
        return [f"mode_unknown:{','.join(who) or '?'}"]
    return who or ["unrecorded_holder"]


# ---------------------------------------------------------------------------
# Stack operations (injectable for tests)
# ---------------------------------------------------------------------------


@dataclass
class StackOps:
    stop: Callable[[str], bool]
    reload: Callable[[str], bool]
    get_json: Callable[[str, float], tuple[int, Any]]
    post_json: Callable[[str, dict, float], tuple[int, Any]]
    pids_on_port: Callable[[int], list[int]]
    proc_maps: Callable[[int], list[str]]
    proc_exe: Callable[[int], str]
    device_held: Callable[[str], bool]
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    kfd_pids: Callable[[], list[int] | None] = kfd_pids
    gpu_quiet_exclusive: Callable[[], list[str] | None] = gpu_quiet_exclusive


def _stack_cmd(action: str, component: str) -> bool:
    script = ORCH_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    try:
        result = subprocess.run([sys.executable, str(script), action, component],
                                cwd=str(ORCH_ROOT), timeout=600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("gpu_window_executor: %s %s failed: %s", action, component, exc)
        return False
    return result.returncode == 0


def _http(url: str, timeout: float, body: dict | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def _pids_on_port(port: int) -> list[int]:
    from scripts.server.stack_processes import pids_on_port

    return pids_on_port(port)


def _proc_maps(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/maps").read_text().splitlines()
    except OSError:
        return []


def _proc_exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def device_flock_held(device_id: str = DEVICE_ID) -> bool:
    """True when some process holds the AK device flock (non-blocking probe)."""
    path = device_lock_path(device_id)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return True  # cannot prove free -> treat as held


def live_ops() -> StackOps:
    return StackOps(
        stop=lambda c: _stack_cmd("stop", c),
        reload=lambda c: _stack_cmd("reload", c),
        get_json=lambda url, t: _http(url, t),
        post_json=lambda url, body, t: _http(url, t, body),
        pids_on_port=_pids_on_port,
        proc_maps=_proc_maps,
        proc_exe=_proc_exe,
        device_held=device_flock_held,
    )


# ---------------------------------------------------------------------------
# Drain + serving proof
# ---------------------------------------------------------------------------


def busy_slots(ops: StackOps, port: int) -> int | None:
    """Number of processing slots on ``port``; None when /slots cannot say."""
    code, body = ops.get_json(f"http://localhost:{port}/slots", 10.0)
    if code != 200 or not isinstance(body, list):
        return None
    return sum(1 for slot in body if isinstance(slot, dict) and slot.get("is_processing"))


def drain(ops: StackOps, ports: Iterable[int], timeout_s: float = DRAIN_TIMEOUT_S) -> None:
    """Wait for every slot on every port to go idle; raise WindowRefused on timeout."""
    deadline = ops.now() + timeout_s
    while True:
        state = {port: busy_slots(ops, port) for port in ports}
        unknown = [p for p, n in state.items() if n is None]
        if unknown:
            raise WindowRefused("drain_unverifiable", f"/slots unavailable on {unknown}")
        if all(n == 0 for n in state.values()):
            return
        if ops.now() >= deadline:
            raise WindowRefused("drain_timeout",
                                f"in-flight slots after {timeout_s:.0f}s: {state}")
        ops.sleep(DRAIN_POLL_S)


def linkage_ok(ops: StackOps, pid: int) -> tuple[bool, str]:
    """Every mapped libggml* comes from the binary's own dir and libggml-hip is resident."""
    exe = ops.proc_exe(pid)
    if not exe:
        return False, f"pid {pid}: exe unreadable"
    own = str(Path(exe).parent)
    libs = {line.split()[-1] for line in ops.proc_maps(pid)
            if "libggml" in line and line.split()[-1].startswith("/")}
    if not libs:
        return False, f"pid {pid}: no libggml mapped"
    foreign = sorted(lib for lib in libs if str(Path(lib).parent) != own)
    if foreign:
        return False, f"pid {pid}: ggml from outside {own}: {foreign[:3]}"
    if not any("libggml-hip" in Path(lib).name for lib in libs):
        return False, f"pid {pid}: libggml-hip not resident"
    return True, own


def serving_proof(ops: StackOps, port: int) -> tuple[bool, str]:
    base = f"http://localhost:{port}"
    code, _ = ops.get_json(f"{base}/health", 10.0)
    if code != 200:
        return False, f"/health {code}"
    code, body = ops.get_json(f"{base}/v1/models", 10.0)
    models = (body or {}).get("data") if isinstance(body, dict) else None
    if code != 200 or not models:
        return False, f"/v1/models {code} empty={not models}"
    model = models[0].get("id") if isinstance(models[0], dict) else None
    code, body = ops.post_json(f"{base}/v1/completions",
                               {"model": model, "prompt": PROOF_PROMPT, "max_tokens": 16,
                                "temperature": 0}, 120.0)
    try:
        text = body["choices"][0]["text"]
    except (TypeError, KeyError, IndexError):
        text = ""
    if code != 200 or not str(text).strip():
        return False, f"/v1/completions {code} empty_text={not str(text).strip()}"
    pids = ops.pids_on_port(port)
    if not pids:
        return False, "no listening pid"
    for pid in pids:
        ok, why = linkage_ok(ops, pid)
        if not ok:
            return False, f"linkage: {why}"
    return True, "ok"


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


@dataclass
class Executor:
    window: Path
    ops: StackOps = field(default_factory=live_ops)

    # -- state helpers ------------------------------------------------------

    def _lease(self) -> dict[str, Any] | None:
        return _read_json(lease_path(self.window))

    def _write_lease(self, lease: dict[str, Any]) -> None:
        lease["updated_at"] = _iso(self.ops.now())
        _write_json(lease_path(self.window), lease)

    def _set_window(self, **fields: Any) -> dict[str, Any]:
        with gw._Locked(self.window):
            data = gw._read_raw(self.window) or {}
            data.update(fields)
            data["written_by"] = WRITER
            gw._atomic_write(self.window, data)
        return data

    # -- device busy --------------------------------------------------------

    def device_busy(self, ports: Iterable[int] = ()) -> list[str]:
        """Why the MI210 is busy right now ([] = free). See DEVICE BUSY in the module doc.

        ``ports`` are the window's parked ports: their own listeners are the servers
        being restored, so their KFD contexts do not make the device busy.
        """
        reasons: list[str] = []
        if self.ops.device_held(DEVICE_ID):
            reasons.append(f"ak_device_flock:{DEVICE_ID}")
        quiet = self.ops.gpu_quiet_exclusive()
        if quiet is None:
            reasons.append("gpu_quiet_unreadable")
        elif quiet:
            reasons.append("gpu_quiet_exclusive:" + ",".join(quiet))
        pids = self.ops.kfd_pids()
        if pids is None:
            reasons.append(f"kfd_unreadable:{kfd_proc_dir()}")
        elif pids:
            own: set[int] = set()
            for port in ports:
                own.update(self.ops.pids_on_port(int(port)))
            foreign = [pid for pid in pids if pid not in own]
            if foreign:
                reasons.append(f"kfd_processes:{foreign[:8]}")
        return reasons

    @staticmethod
    def _busy_reason(reasons: list[str]) -> str:
        # ``device_held`` keeps its meaning (AK still holds the device); anything
        # else that occupies the GPU is ``device_busy``.
        return "device_held" if any(r.startswith("ak_device_flock") for r in reasons) \
            else "device_busy"

    @staticmethod
    def _note(lease: dict[str, Any], at: str, event: str, **fields: Any) -> None:
        """Append to the lease history; a repeat of the last event bumps its count."""
        history = lease.setdefault("history", [])
        last = history[-1] if history else None
        if isinstance(last, dict) and last.get("event") == event and all(
                last.get(k) == v for k, v in fields.items()):
            last["count"] = int(last.get("count") or 1) + 1
            last["last_at"] = at
        else:
            history.append({"at": at, "event": event, **fields})
            del history[:-MAX_HISTORY]

    @staticmethod
    def _tracks(lease: dict[str, Any], current: dict[str, Any]) -> bool:
        """True when ``lease`` is the live record of the window in ``current``."""
        if not lease.get("window_id") or lease.get("state") in (None, "closed"):
            return False
        if lease.get("origin") == LEGACY_ORIGIN:
            return lease.get("window_since") == current.get("since")
        return current.get("window_id") == lease.get("window_id")

    def _is_legacy(self, lease: dict[str, Any], current: dict[str, Any]) -> bool:
        """A non-production window this executor did not open (CLI/hand park, no lease)."""
        if current.get("holder") in (None, "production"):
            return False
        if not self._tracks(lease, current):
            return True
        return lease.get("origin") == LEGACY_ORIGIN

    def _write_production(self, previous: dict[str, Any] | None) -> dict[str, Any]:
        with gw._Locked(self.window):
            data = {
                "holder": "production", "since": _iso(self.ops.now()), "expected_end": None,
                "preempt_requested_at": None, "preempt_reason": None, "parked_roles": [],
                "parked_ports": [], "grant_state": None, "window_id": None,
                "written_by": WRITER,
                "previous_holder": (previous or {}).get("holder"),
                "previous_window_id": (previous or {}).get("window_id"),
            }
            gw._atomic_write(self.window, data)
        return data

    # -- open ---------------------------------------------------------------

    def open(self, *, roles: list[str], ports: list[int], components: list[str],
             expected_end: str, authority: Authority, schedule_ref: str,
             campaign_id: str, drain_timeout_s: float = DRAIN_TIMEOUT_S) -> dict[str, Any]:
        now = self.ops.now()
        end = gw._parse_ts(expected_end)
        if end is None or end <= now:
            raise WindowRefused("bad_expected_end", str(expected_end))
        if end - now > MAX_WINDOW_S + 5:
            raise WindowRefused("window_too_long",
                                f"{end - now:.0f}s > {MAX_WINDOW_S}s; longer needs the operator")
        if not ports or not components:
            raise WindowRefused("bad_request", "open needs --ports and --components")
        if len(set(ports)) != len(ports):
            raise WindowRefused("bad_request", f"duplicate port in {ports}")
        if len(components) != len(ports):
            # F1: components pair 1:1 with ports (stop/reload a port's own server); a
            # mismatch would make close reload every component, not the failing one.
            raise WindowRefused("bad_request",
                                f"{len(components)} components for {len(ports)} ports; "
                                "pass one stack component per port (default server_<port>)")
        pairs = sorted(zip(ports, components))
        ports = [p for p, _ in pairs]
        components = [c for _, c in pairs]
        with LeaseLock(self.window):
            pending = pending_stack_change()
            if pending is not None:
                raise WindowRefused("stack_change_pending", json.dumps(pending)[:300])
            current = gw._read_raw(self.window)
            if current is not None and str(current.get("holder") or "") != "production":
                raise WindowRefused("window_held", f"holder={current.get('holder')}")
            lease = self._lease()
            if lease and lease.get("state") not in (None, "closed"):
                raise WindowRefused("lease_open", f"window {lease.get('window_id')} "
                                    f"state={lease.get('state')}")
            entry = resolve_schedule_entry(schedule_path(self.window), schedule_ref, now, end)
            busy = self.device_busy(ports)
            if busy:
                raise WindowRefused("device_busy",
                                    f"{DEVICE_ID} busy before the grant: {'; '.join(busy)}")
            window_id = f"gw-{uuid.uuid4().hex[:12]}"
            lease = {
                "schema": LEASE_SCHEMA, "window_id": window_id, "state": "draining",
                "boot_id": boot_id(), "opened_at": _iso(now), "expected_end": _iso(end),
                "restore_by": _iso(end + RESTORE_GRACE_S), "roles": sorted(set(roles)),
                "ports": list(ports), "components": list(components),
                "authority": authority.record(), "schedule_ref": schedule_ref,
                "schedule_entry": entry, "campaign_id": campaign_id,
                "ak_seen_holding": False, "reload_attempts": 0, "history": [],
            }
            self._write_lease(lease)
            gw.park(roles=roles, ports=ports, holder="autokernel", expected_end=_iso(end),
                    path=self.window,
                    extra={"window_id": window_id, "device_id": DEVICE_ID,
                           "grant_state": "draining", "written_by": WRITER,
                           "schedule_ref": schedule_ref, "campaign_id": campaign_id})
            try:
                drain(self.ops, lease["ports"], drain_timeout_s)
            except WindowRefused as exc:
                # Nothing was stopped: hand the roles straight back.
                self._write_production(gw._read_raw(self.window))
                lease.update(state="closed", outcome=f"refused:{exc.reason}")
                self._write_lease(lease)
                raise
            for component in components:
                if not self.ops.stop(component):
                    lease["history"].append({"at": _iso(self.ops.now()), "event": "stop_failed",
                                             "component": component})
                    lease["state"] = "restoring"
                    self._write_lease(lease)
                    result = self._close_locked(lease, reason="stop_failed")
                    raise WindowRefused("stop_failed", f"{component}; close={result['outcome']}")
            still = {p: self.ops.pids_on_port(p) for p in lease["ports"]}
            if any(still.values()):
                lease["state"] = "restoring"
                self._write_lease(lease)
                result = self._close_locked(lease, reason="ports_not_free")
                raise WindowRefused("ports_not_free", f"{still}; close={result['outcome']}")
            self._set_window(grant_state="granted", granted_at=_iso(self.ops.now()))
            lease["state"] = "open"
            self._write_lease(lease)
            return lease

    # -- close --------------------------------------------------------------

    def close(self, *, reason: str, allow_reload: bool = True) -> dict[str, Any]:
        with LeaseLock(self.window):
            lease = self._lease() or {}
            return self._close_locked(lease, reason=reason, allow_reload=allow_reload)

    def _close_locked(self, lease: dict[str, Any], *, reason: str,
                      allow_reload: bool = True) -> dict[str, Any]:
        current = gw._read_raw(self.window)
        if (current is None or current.get("holder") == "production") and \
                lease.get("state") in (None, "closed"):
            return {"outcome": "noop", "holder": "production"}
        current = current or {}
        if current.get("holder") not in (None, "production") and \
                not self._tracks(lease, current):
            # Manual / legacy park (or a stale lease from an earlier window): track it
            # under a fresh lease of its own, marked legacy so the watchdog's
            # legacy rule keeps applying to it on every later tick.
            previous_id = lease.get("window_id")
            lease.clear()
            lease.update(schema=LEASE_SCHEMA, origin=LEGACY_ORIGIN,
                         boot_id=current.get("boot_id") or boot_id(),
                         window_id=current.get("window_id")
                         or f"manual-{current.get('since') or _iso(self.ops.now())}",
                         window_since=current.get("since"),
                         expected_end=current.get("expected_end"),
                         ports=list(current.get("parked_ports") or []),
                         components=default_components(current.get("parked_ports") or []),
                         state="adopted", reload_attempts=0, history=[],
                         previous_lease_window_id=previous_id)
        history = lease.setdefault("history", [])
        ports = list(lease.get("ports") or [])
        components = list(lease.get("components") or [])
        busy = self.device_busy(ports)
        if busy:
            why = self._busy_reason(busy)
            self._note(lease, _iso(self.ops.now()), f"close_refused_{why}", reason=reason,
                       busy=busy)
            self._write_lease(lease)
            logger.error("gpu_window_executor: REFUSING close (%s) of window %s: %s busy: %s "
                         "-- no reload; retry next tick", reason, lease.get("window_id"),
                         DEVICE_ID, "; ".join(busy))
            raise WindowRefused(why, f"{DEVICE_ID} busy: {'; '.join(busy)}")
        if current.get("reboot_reconciled"):
            allow_reload = False  # after a reboot, bring-up belongs to the stack owner
        if current.get("holder") not in (None, "production"):
            self._set_window(holder="released", grant_state="restoring",
                             restoring_since=current.get("restoring_since")
                             or _iso(self.ops.now()), restore_reason=reason)
        lease.update(state="restoring", close_reason=reason)
        self._write_lease(lease)
        failing = self._failing(ports)
        if failing and allow_reload and lease.get("reload_attempts", 0) < MAX_RELOAD_ATTEMPTS:
            lease["reload_attempts"] = lease.get("reload_attempts", 0) + 1
            for component in self._components_for(failing, ports, components):
                ok = self.ops.reload(component)
                history.append({"at": _iso(self.ops.now()), "event": "reload",
                                "component": component, "ok": ok})
            failing = self._failing(ports)
        if failing or not ports:
            lease.update(state="restore_failed",
                         last_proof_failure={str(p): w for p, w in failing.items()}
                         if failing else {"_": "no parked ports known"})
            self._write_lease(lease)
            return {"outcome": "restore_failed", "failing": lease["last_proof_failure"]}
        self._write_production(current)
        lease.update(state="closed", outcome="restored", closed_at=_iso(self.ops.now()))
        self._write_lease(lease)
        return {"outcome": "restored", "holder": "production"}

    def _failing(self, ports: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        for port in ports:
            ok, why = serving_proof(self.ops, port)
            if not ok:
                out[port] = why
        return out

    @staticmethod
    def _components_for(failing: dict[int, str], ports: list[int],
                        components: list[str]) -> list[str]:
        if len(components) == len(ports):
            return [c for c, p in zip(components, ports) if p in failing]
        # Unpaired (legacy lease): reload only the failing ports' own servers, never
        # every component — that was up to two needless 27B restarts (F1).
        return default_components([p for p in ports if p in failing])

    # -- watchdog -----------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        """One watchdog pass. Never raises; always writes the status file."""
        action, detail = "none", ""
        try:
            action, detail = self._tick()
        except WindowRefused as exc:
            action, detail = f"refused:{exc.reason}", exc.detail
        except Exception as exc:  # pragma: no cover - the watchdog must keep ticking
            logger.exception("gpu_window_executor: tick failed")
            action, detail = "error", str(exc)[:300]
        return self.write_status(action, detail)

    def _tick(self) -> tuple[str, str]:
        current = gw._read_raw(self.window)
        lease = self._lease() or {}
        if (current or {}).get("holder") in (None, "production") and \
                lease.get("state") in (None, "closed"):
            return "none", "production"
        try:
            lock = LeaseLock(self.window)
            lock.__enter__()
        except WindowRefused:
            return "busy", "a transition is in progress"
        try:
            return self._tick_locked()
        finally:
            lock.__exit__(None, None, None)

    def _tick_locked(self) -> tuple[str, str]:
        now = self.ops.now()
        lease = self._lease() or {}
        current = gw._read_raw(self.window) or {}
        if self._is_legacy(lease, current):
            return self._tick_legacy(lease, current, now)
        recorded_boot = lease.get("boot_id") or current.get("boot_id")
        if current.get("holder") not in (None, "production") and recorded_boot \
                and recorded_boot != boot_id():
            # Reboot: neutralise the stale grant; never start servers from here.
            self._set_window(boot_id=boot_id(), reboot_reconciled=True)
            if lease.get("window_id"):
                lease["boot_id"] = boot_id()
            result = self._close_locked(lease, reason="reboot_reconcile")
            return "reboot_reconcile", result["outcome"]
        if lease.get("state") == "draining":
            # An ``open`` died between park and grant (we hold the lease lock, so it
            # is not running): nothing was handed to AK — close at once.
            result = self._close_locked(lease, reason="interrupted_open")
            return "interrupted_open", result["outcome"]
        if lease.get("state") in ("restoring", "restore_failed"):
            result = self._close_locked(lease, reason="retry_restore")
            return "retry_restore", result["outcome"]
        held = self.ops.device_held(DEVICE_ID)
        if held and lease.get("state") == "open" and not lease.get("ak_seen_holding"):
            lease["ak_seen_holding"] = True
            self._write_lease(lease)
        end = gw._parse_ts(current.get("expected_end"))
        overdue = end is None or now >= end + RESTORE_GRACE_S
        is_open = lease.get("state") == "open"
        released = is_open and not held and bool(lease.get("ak_seen_holding"))
        preempted = is_open and not held and bool(current.get("preempt_requested_at"))
        if overdue or released or preempted:
            why = "expiry" if overdue else "ak_released" if released else "preempt"
            result = self._close_locked(lease, reason=f"watchdog_{why}")
            return f"close_{why}", result["outcome"]
        return "none", f"window open until {current.get('expected_end')}"

    def _tick_legacy(self, lease: dict[str, Any], current: dict[str, Any],
                     now: float) -> tuple[str, str]:
        """A park this executor did not open: close only when device-free AND past
        ``expected_end + RESTORE_GRACE_S``, and then through ``_close_locked``
        (busy check + serving proof). Never on reboot, release or preempt alone."""
        end = gw._parse_ts(current.get("expected_end"))
        desc = (f"legacy/no-lease park holder={current.get('holder')} "
                f"ports={current.get('parked_ports')} expected_end={current.get('expected_end')}")
        if end is None or now < end + RESTORE_GRACE_S:
            logger.warning("gpu_window_executor: NOT auto-closing %s: this executor did not "
                           "open it; it closes only once device-free and past expected_end "
                           "+ %ss (never without an expected_end)", desc, RESTORE_GRACE_S)
            return "hold_legacy", f"{desc}; not before expected_end + {RESTORE_GRACE_S}s"
        logger.error("gpu_window_executor: AUTO-CLOSING %s (past expected_end + %ss); "
                     "device-busy check and serving proof first", desc, RESTORE_GRACE_S)
        result = self._close_locked(lease, reason="watchdog_legacy_expiry")
        return "close_legacy_expiry", result["outcome"]

    # -- status (hub freshness envelope) ------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = self.ops.now()
        current = gw._read_raw(self.window) or {}
        lease = self._lease() or {}
        end = gw._parse_ts(current.get("expected_end"))
        holder = current.get("holder") or "production"
        overdue = holder != "production" and (end is None or now >= end + RESTORE_GRACE_S)
        if holder == "production":
            verdict = "ok"
        elif lease.get("state") == "restore_failed" or overdue:
            verdict = "alarm"
        else:
            verdict = "window_open"
        busy = (self.device_busy(current.get("parked_ports") or lease.get("ports") or [])
                if holder != "production" else [])
        return {
            "device_busy": busy, "legacy_park": self._is_legacy(lease, current),
            "holder": holder, "grant_state": current.get("grant_state"),
            "window_id": current.get("window_id"), "expected_end": current.get("expected_end"),
            "parked_ports": current.get("parked_ports") or [],
            "preempt_requested_at": current.get("preempt_requested_at"),
            "lease_state": lease.get("state"), "overdue": overdue, "verdict": verdict,
            "stack_change_pending": pending_stack_change() is not None,
            "device_held": self.ops.device_held(DEVICE_ID),
        }

    def write_status(self, action: str, detail: str) -> dict[str, Any]:
        now = self.ops.now()
        status = {"schema": STATUS_SCHEMA, "generated_at": _iso(now), "generated_at_epoch": now,
                  "boot_id": boot_id(), "last_action": action, "detail": detail,
                  "fresh_for_s": 180, **self.snapshot()}
        try:
            _write_json(status_path(self.window), status)
        except OSError:
            logger.warning("gpu_window_executor: cannot write status file", exc_info=True)
        return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.runtime.gpu_window_executor",
                                     description="Stack-owned MI210 window executor (G1).")
    parser.add_argument("--file", help="window file (default $ORCHESTRATOR_GPU_WINDOW_FILE)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_open = sub.add_parser("open", help="drain, stop, grant (stack owner / operator token)")
    p_open.add_argument("--roles", required=True, help="comma-separated parked roles")
    p_open.add_argument("--ports", required=True, help="comma-separated parked ports")
    p_open.add_argument("--components",
                        help="stack components to stop/reload, one per port in --ports "
                             "order (default server_<port>)")
    p_open.add_argument("--expected-end", required=True, help="+45m / ISO; at most +60m")
    p_open.add_argument("--schedule-ref", required=True)
    p_open.add_argument("--campaign-id", required=True)
    p_open.add_argument("--compute-grant", help="coordinator-daemon compute-grant id")
    p_open.add_argument("--stack-owner-session")
    p_open.add_argument("--operator-token")
    p_close = sub.add_parser("close", help="serving proof, then holder=production")
    p_close.add_argument("--reason", default="manual")
    sub.add_parser("tick", help="one watchdog pass (cron)")
    sub.add_parser("status", help="print the status snapshot")
    p_sched = sub.add_parser("validate-schedule", help="validate the schedule file")
    p_sched.add_argument("path", nargs="?")
    p_pend = sub.add_parser("stack-change-pending", help="set/clear/show the pending marker")
    p_pend.add_argument("action", choices=("set", "clear", "show"))
    p_pend.add_argument("--change-id")
    p_pend.add_argument("--phase", choices=("apply", "bring-up"))
    p_pend.add_argument("--by", default="stack-change")
    args = parser.parse_args(argv)

    try:
        window = _window_path(Path(args.file) if args.file else None)
        if args.cmd == "validate-schedule":
            path = Path(args.path) if args.path else schedule_path(window)
            data = _read_json(path)
            errors = ["unreadable or not a JSON object"] if data is None else validate_schedule(data)
            print(json.dumps({"path": str(path), "valid": not errors, "errors": errors}, indent=2))
            return 0 if not errors else 1
        if args.cmd == "stack-change-pending":
            if args.action == "set":
                if not args.change_id or not args.phase:
                    parser.error("set needs --change-id and --phase")
                out: Any = set_pending(args.change_id, args.phase, by=args.by)
            elif args.action == "clear":
                out = {"cleared": clear_pending()}
            else:
                out = {"pending": pending_stack_change()}
            print(json.dumps(out, indent=2))
            return 0
        executor = Executor(window)
        if args.cmd == "open":
            authority = authorize(stack_owner_session=args.stack_owner_session,
                                  operator_token=args.operator_token,
                                  compute_grant=args.compute_grant)
            roles = gw._split_csv([args.roles])
            ports = [int(p) for p in gw._split_csv([args.ports])]
            out = executor.open(
                roles=roles, ports=ports,
                components=(gw._split_csv([args.components]) if args.components
                            else default_components(ports)),
                expected_end=gw.parse_expected_end(args.expected_end), authority=authority,
                schedule_ref=args.schedule_ref, campaign_id=args.campaign_id)
        elif args.cmd == "close":
            out = executor.close(reason=args.reason)
        elif args.cmd == "tick":
            out = executor.tick()
        else:
            out = executor.snapshot()
    except WindowRefused as exc:
        print(json.dumps({"refused": exc.reason, "detail": exc.detail}), file=sys.stderr)
        return 3
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0 if not (isinstance(out, dict) and out.get("outcome") == "restore_failed") else 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
