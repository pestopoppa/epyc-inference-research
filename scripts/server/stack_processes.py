"""Process and port helpers for orchestrator stack management."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

from scripts.server import stack_log_banner as _log_banner


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Return True when a TCP connection to host:port succeeds."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def pids_on_port(port: int, timeout: int = 3) -> list[int]:
    """Best-effort discovery of LISTEN pids on a TCP port.

    The LISTEN filter is intentional: raw `lsof -i :PORT` also returns client
    processes with established connections, which must not be killed during
    stack reload or stale-port cleanup.
    """
    try:
        result = subprocess.run(
            ["lsof", "-t", "-sTCP:LISTEN", f"-i:{port}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return []

    pids: list[int] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def pid_alive(pid: int) -> bool:
    """Return True when a pid currently exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_cmdline(pid: int) -> list[str]:
    """Return /proc cmdline tokens for a pid, or [] when unavailable."""
    if pid <= 0:
        return []
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return []
    return [
        part.decode(errors="replace")
        for part in raw.split(b"\0")
        if part
    ]


def child_pids(pid: int, timeout: int = 3) -> list[int]:
    """Return direct child pids for a process."""
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", str(pid)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return []

    children: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            children.append(int(line))
        except ValueError:
            continue
    return children


def collect_descendants(root_pid: int) -> list[int]:
    """Collect all descendants of root_pid breadth-first."""
    descendants: list[int] = []
    queue = [root_pid]
    seen = {root_pid}
    while queue:
        parent = queue.pop(0)
        for child in child_pids(parent):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            queue.append(child)
    return descendants


def kill_process_tree(pid: int, timeout: int = 5) -> bool:
    """Kill a process tree gracefully, then forcefully."""
    if pid <= 0:
        return True

    this_pid = os.getpid()
    targets = [p for p in (collect_descendants(pid) + [pid]) if p > 0 and p != this_pid]
    if not targets:
        return True

    # Wall-clock stop banner in the server's own log (its lines carry only
    # process-relative time). Resolved from /proc/<pid>/fd/1, so every kill path —
    # stop, reload, failed-health cleanup — gets one. Best-effort.
    stop_logs = _stop_logs(targets)
    _log_banner.write_stop_banner(stop_logs)
    try:
        for target in reversed(targets):
            try:
                os.kill(target, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"  [!] Permission denied killing PID {target}")
        for _ in range(timeout):
            time.sleep(1)
            if not any(pid_alive(target) for target in targets):
                _log_banner.write_stopped_banner(stop_logs, result="exited_on_sigterm")
                return True
        for target in reversed(targets):
            if not pid_alive(target):
                continue
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"  [!] Permission denied force-killing PID {target}")
        time.sleep(1)
        dead = not any(pid_alive(target) for target in targets)
        _log_banner.write_stopped_banner(
            stop_logs, result="killed_sigkill" if dead else "still_alive"
        )
        return dead
    except Exception as exc:
        print(f"  [!] Failed to kill PID {pid}: {exc}")
        _log_banner.write_stopped_banner(stop_logs, result="error")
        return False


def _stop_logs(targets: list[int]) -> dict[Path, list[int]]:
    try:
        return _log_banner.stop_logs_for(targets)
    except Exception:
        return {}


def scan_known_ports(ports: Iterable[int]) -> dict[int, list[int]]:
    """Return listener pids for known ports."""
    found: dict[int, list[int]] = {}
    for port in sorted(set(ports)):
        pids = pids_on_port(port, timeout=5)
        if pids:
            found[port] = pids
    return found


def free_memory_gb() -> int:
    """Return MemAvailable from /proc/meminfo in GB (rounded down)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb // (1024 * 1024)
    return 0


def renice_all_threads(pid: int, nice: int) -> None:
    """Renice every thread of `pid` to `nice`.

    `renice -p PID` from CLI only renices the lead thread; OMP team threads
    spawned during model load keep their original priority. This iterates
    /proc/<pid>/task/<tid> and sets each one explicitly. Idempotent.

    Used to deprioritize binary_override (gemma4 MTP via ik_llama.cpp PR
    #1744) which busy-spins 96 cores idle and contaminates other-role
    measurements unless reniced. Verified 2026-05-09: post-renice, frontdoor
    4.55 → 7.11 t/s, coder 4.02 → 12.34, ingest 10.46 → 28.99.

    Going from nice=0 to nice=19 (lower priority) is allowed for the owner
    without sudo.
    """
    task_dir = Path(f"/proc/{pid}/task")
    if not task_dir.exists():
        return
    ok = 0
    fail = 0
    for tid_dir in task_dir.iterdir():
        try:
            tid = int(tid_dir.name)
        except ValueError:
            continue
        try:
            os.setpriority(os.PRIO_PROCESS, tid, nice)
            ok += 1
        except (PermissionError, ProcessLookupError, OSError):
            fail += 1
    print(f"    [renice] {ok} thread(s) → nice={nice}"
          + (f" ({fail} failed)" if fail else ""))


def set_oom_score_adj(pids: Iterable[int], adj: int = -1000, timeout: int = 5) -> int:
    """Set oom_score_adj for each pid so earlyoom (and the kernel OOM killer) spare them.

    earlyoom skips processes whose oom_score_adj == -1000 in BOTH its oom_score and
    --sort-by-rss modes (see epyc-root handoffs/active/earlyoom-oom-protection.md). The
    orchestrator API master + its uvicorn workers are comm=python, so they cannot be
    earlyoom --ignore'd by name (they collide with runaway python evals) — oom_score_adj
    is the durable control-plane protection, replacing the manual one-shot `choom` that
    did not survive an API restart.

    Lowering oom_score_adj below 0 needs CAP_SYS_RESOURCE, so this uses `sudo -n` (the
    same NOPASSWD pattern as stack_host.py / host_health.py), preferring `choom` and
    falling back to `tee /proc/<pid>/oom_score_adj`. Best-effort and idempotent: a
    missing or password-denied sudo/choom logs a warning and is skipped so a stack
    start never fails on it. Returns the count of pids successfully set.
    """
    targets = [int(p) for p in pids]
    if not targets:
        return 0
    if not shutil.which("sudo"):
        print(f"    [oom-protect] sudo not found — skipping oom_score_adj={adj} for "
              f"{len(targets)} pid(s); control plane unprotected from earlyoom")
        return 0
    use_choom = shutil.which("choom") is not None
    ok = 0
    failed: list[int] = []
    for pid in targets:
        try:
            if use_choom:
                subprocess.run(
                    ["sudo", "-n", "choom", "-n", str(adj), "-p", str(pid)],
                    capture_output=True, timeout=timeout, check=True,
                )
            else:
                subprocess.run(
                    ["sudo", "-n", "tee", f"/proc/{pid}/oom_score_adj"],
                    input=f"{adj}\n", text=True, capture_output=True,
                    timeout=timeout, check=True,
                )
            ok += 1
        except Exception:
            failed.append(pid)
    msg = f"    [oom-protect] oom_score_adj={adj} on {ok}/{len(targets)} pid(s)"
    if failed:
        msg += (" — could not set "
                + ",".join(str(p) for p in failed)
                + " (configure NOPASSWD sudo for choom to protect the control plane "
                "from earlyoom)")
    print(msg)
    return ok
