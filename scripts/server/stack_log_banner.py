"""Wall-clock launch / stop banners for stack-managed server logs.

llama-server timestamps its log lines relative to process start (``M.SS.mmm.uuu``)
and the stack opens each log in append mode, so one file holds many launches with
no wall clock anywhere in it. Reconstructing a week of traffic (workspace-89,
2026-10-03) meant chaining "cleaning up before exit" lines back from a state file.
These banners make each segment self-anchoring:

    === launch 2026-10-03T03:45:48.505Z launch_id=... port=8070 roles=frontdoor,... ===
    === launch-detail {"argv": [...], "binary": ..., ...} ===
    === launched 2026-10-03T03:45:48.512Z launch_id=... pid=26641 ===
    ... llama-server output ...
    === stop 2026-10-03T09:12:01.003Z pid=26641 by=orchestrator_stack reload frontdoor ===
    === stopped 2026-10-03T09:12:03.110Z pid=26641 result=exited ===

The launch identity (argv sha256, binary, served model) is also written to
``<log dir>/server_launches/<port>.json`` (latest launch for that port) and
appended to ``<log dir>/server_launches/launches.jsonl`` (history). The
orchestrator's per-call serving record (``src/backends/serving_calls.py``) joins on
the sidecar to say which launch served each request.

Everything here is best-effort: a banner that cannot be written never blocks a
launch or a stop.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_DIRNAME = "server_launches"
LAUNCH_SCHEMA = "epyc.stack.server_launch.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stack_commit() -> str | None:
    try:
        from src.runtime.git_head import resolve_git_head

        return resolve_git_head(_REPO_ROOT)
    except Exception:
        return None


def _model_from_argv(argv: list[str]) -> str | None:
    for flag in ("-m", "--model"):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                return argv[idx + 1]
    return None


def _binary_from_argv(argv: list[str]) -> str | None:
    for arg in argv:
        if Path(arg).name.startswith("llama-server"):
            return arg
    return None


def argv_sha256(argv: Iterable[str]) -> str:
    """Stable digest of an argv (NUL-joined, so arg boundaries count)."""
    return hashlib.sha256("\0".join(str(a) for a in argv).encode()).hexdigest()


def _write_line(log: IO[str], line: str) -> None:
    try:
        log.write(line + "\n")
        log.flush()
    except Exception:
        pass


def write_launch_banner(
    log: IO[str],
    *,
    port: int,
    roles: Iterable[str],
    argv: list[str],
    env: Mapping[str, str] | None = None,
    binary: str | None = None,
) -> dict[str, Any]:
    """Write the pre-launch banner; return the launch record for :func:`write_launch_pid`.

    Written BEFORE ``Popen`` so it precedes every line the server prints.
    """
    argv = [str(a) for a in argv]
    binary = binary or _binary_from_argv(argv)
    try:
        binary_realpath = str(Path(binary).resolve()) if binary else None
    except OSError:
        binary_realpath = None
    record: dict[str, Any] = {
        "schema": LAUNCH_SCHEMA,
        "launch_id": uuid.uuid4().hex,
        "launched_at": _now_iso(),
        "port": int(port),
        "roles": [str(r) for r in roles],
        "argv": argv,
        "argv_sha256": argv_sha256(argv),
        "binary": binary,
        "binary_realpath": binary_realpath,
        "model_path": _model_from_argv(argv),
        "ld_library_path": (env or {}).get("LD_LIBRARY_PATH"),
        "launcher_pid": os.getpid(),
        "launcher_argv": [str(a) for a in sys.argv],
        "stack_commit": _stack_commit(),
    }
    try:
        _write_line(
            log,
            f"=== launch {record['launched_at']} launch_id={record['launch_id']} "
            f"port={record['port']} roles={','.join(record['roles'])} "
            f"argv_sha256={record['argv_sha256'][:16]} ===",
        )
        detail = {k: record[k] for k in ("argv", "binary", "binary_realpath", "model_path",
                                         "ld_library_path", "stack_commit", "launcher_argv")}
        _write_line(log, f"=== launch-detail {json.dumps(detail, separators=(',', ':'))} ===")
    except Exception:
        pass
    return record


def _sidecar_dir_for(log: IO[str]) -> Path | None:
    name = getattr(log, "name", None)
    if not isinstance(name, str) or not name:
        return None
    return Path(name).parent / SIDECAR_DIRNAME


def write_launch_pid(log: IO[str], record: dict[str, Any], pid: int) -> None:
    """Record the spawned pid (banner line + sidecar). Called right after ``Popen``."""
    try:
        record = dict(record, pid=int(pid), pid_recorded_at=_now_iso())
        _write_line(
            log,
            f"=== launched {record['pid_recorded_at']} launch_id={record.get('launch_id')} "
            f"pid={pid} ===",
        )
        sidecar_dir = _sidecar_dir_for(log)
        if sidecar_dir is None:
            return
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
        target = sidecar_dir / f"{int(record['port'])}.json"
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        tmp.write_text(payload + "\n")
        os.replace(tmp, target)
        with open(sidecar_dir / "launches.jsonl", "a") as fh:
            fh.write(payload + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stop banners — resolved from the process itself, so every kill path gets one
# ---------------------------------------------------------------------------


def stdout_log_of(pid: int, log_dir: Path | None = None) -> Path | None:
    """The ``.log`` file ``pid``'s stdout points at, or None.

    Read from ``/proc/<pid>/fd/1``. Only a regular ``*.log`` file counts, and when
    ``log_dir`` is given only one inside it, so a banner never lands in a terminal,
    a pipe, or a file the stack does not own.
    """
    try:
        target = Path(os.readlink(f"/proc/{int(pid)}/fd/1"))
    except (OSError, ValueError):
        return None
    if target.suffix != ".log" or not target.is_file():
        return None
    if log_dir is not None:
        try:
            target.resolve().relative_to(Path(log_dir).resolve())
        except (OSError, ValueError):
            return None
    return target


def _default_log_dir() -> Path:
    return Path(os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", str(_REPO_ROOT / "logs")))


def stop_logs_for(pids: Iterable[int], log_dir: Path | None = None) -> dict[Path, list[int]]:
    """Group ``pids`` by the stack log their stdout writes to."""
    log_dir = log_dir if log_dir is not None else _default_log_dir()
    grouped: dict[Path, list[int]] = {}
    for pid in pids:
        path = stdout_log_of(pid, log_dir)
        if path is not None:
            grouped.setdefault(path, []).append(int(pid))
    return grouped


def _append(path: Path, line: str) -> None:
    try:
        with open(path, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def write_stop_banner(logs: Mapping[Path, list[int]], *, reason: str | None = None) -> None:
    """``=== stop <ISO> pid=... by=... ===`` before the first signal is sent."""
    by = reason or " ".join(Path(sys.argv[0]).name.split()[:1] + sys.argv[1:])[:200]
    for path, pids in logs.items():
        _append(path, f"=== stop {_now_iso()} pid={','.join(map(str, pids))} by={by} ===")


def write_stopped_banner(logs: Mapping[Path, list[int]], *, result: str) -> None:
    """``=== stopped <ISO> pid=... result=... ===`` once the kill outcome is known."""
    for path, pids in logs.items():
        _append(path, f"=== stopped {_now_iso()} pid={','.join(map(str, pids))} result={result} ===")
