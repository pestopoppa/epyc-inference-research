"""Every thread of a CPU-pinned serving launch stays on the recipe's ``cpu_list``.

Research-side twin of epyc-orchestrator ``scripts/server/launch_affinity.py`` (same
semantics, same env key; keep the two in step). The repos do not import each other,
so the ~60 lines that matter are duplicated rather than path-hacked.

Why (workspace-58, 2026-10-22): a ``taskset -c 184-191`` HIP llama-server had ONE
early ROCr thread on ``0-191``. The installed ROCr (ROCm 6.2.0,
libhsa-runtime64.so.1.14.60200) hands every thread it creates an explicit all-CPU
affinity unless ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG`` is exactly ``"0"``. After health,
``serving._measure_once`` calls ``pin_own_child``: it re-pins anything still outside the
placement, and the launch fails if a task escapes. The env key is exported for the
stack launcher's use. The AutoKernel loader does NOT set it, because its launch env is
identity-bearing (STACKCHG-GPUPIN-20261022 D2).

One-off research drivers (GPS-PIN-2, 2026-10-23) use the same two layers through
``apply_pinned_launch_env`` (layer 1, the env knob on a ``taskset -c`` launch; probe P1
2026-10-23 showed it alone keeps 196/196 ROCr threads inside the taskset) plus either
``enforce_pinned_child`` after health (a server) or ``PinWatch`` while it runs (a
run-to-completion tool such as llama-bench). Both refuse on an escape (D1); both return
the per-task read for the driver's record. Shell drivers call the CLI:

    launch_affinity.py --enforce PID CPUS [--json]   # exit 0 clean, 1 refuse
    launch_affinity.py --check PID CPUS [--json]     # read-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROCR_AFFINITY_ENV_KEY = "HSA_OVERRIDE_CPU_AFFINITY_DEBUG"
ROCR_AFFINITY_ENV_VALUE = "0"
DEFAULT_PROC_ROOT = Path("/proc")


def parse_cpu_list(spec: str) -> frozenset[int]:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"empty cpu list: {spec!r}")
    cpus: set[int] = set()
    for part in spec.strip().split(","):
        lo_s, sep, hi_s = part.strip().partition("-")
        if not lo_s.isdigit() or (sep and not hi_s.isdigit()):
            raise ValueError(f"malformed segment {part!r} in cpu list {spec!r}")
        lo, hi = int(lo_s), int(hi_s) if sep else int(lo_s)
        if hi < lo:
            raise ValueError(f"descending range {part!r} in cpu list {spec!r}")
        cpus.update(range(lo, hi + 1))
    return frozenset(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    ordered = sorted(set(cpus))
    runs: list[str] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        runs.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(runs)


def task_masks(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> list[tuple[int, str, frozenset[int]]]:
    """(tid, comm, cpus) for every task of ``pid``; [] when the pid is gone."""
    root = Path(proc_root) / str(pid) / "task"
    try:
        entries = sorted((e for e in root.iterdir() if e.name.isdigit()), key=lambda e: int(e.name))
    except OSError:
        return []
    out = []
    for entry in entries:
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        mask = next((ln.split(":", 1)[1].strip() for ln in status.splitlines()
                     if ln.startswith("Cpus_allowed_list:")), None)
        if mask is None:
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            comm = "?"
        out.append((int(entry.name), comm, parse_cpu_list(mask)))
    return out


def escapes(pid: int, cpu_list: str, proc_root: Path = DEFAULT_PROC_ROOT) -> list[dict]:
    allowed = parse_cpu_list(cpu_list)
    return [{"tid": t, "comm": c, "cpus": format_cpu_list(m)}
            for t, c, m in task_masks(pid, proc_root) if not m <= allowed]


@dataclass
class PinResult:
    pid: int
    cpus: str
    tasks_seen: int = 0
    repinned: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    escapes_after: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.tasks_seen > 0 and not self.escapes_after

    def as_dict(self) -> dict:
        return {"pid": self.pid, "cpus": self.cpus, "tasks_seen": self.tasks_seen,
                "repinned": self.repinned, "errors": self.errors,
                "escapes_after": self.escapes_after, "ok": self.ok}


def parent_pid(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> int | None:
    try:
        stat = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        return int(stat.rsplit(") ", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def pin_own_child(pid: int, cpu_list: str, *, proc_root: Path = DEFAULT_PROC_ROOT,
                  setter: Callable[[int, Iterable[int]], None] = os.sched_setaffinity,
                  launcher_pid: int | None = None) -> PinResult | None:
    """``pin_all_tasks`` for a pid that is provably OUR direct child; None (untouched) otherwise.

    The spawn is ``taskset -c ... llama-server`` (taskset execs), so the Popen pid is the
    server and its parent is this process. A recycled pid or a test double is never
    re-pinned.
    """
    me = os.getpid() if launcher_pid is None else launcher_pid
    if parent_pid(pid, proc_root) != me:
        return None
    return pin_all_tasks(pid, cpu_list, proc_root=proc_root, setter=setter)


def pin_all_tasks(pid: int, cpu_list: str, *, proc_root: Path = DEFAULT_PROC_ROOT,
                  setter: Callable[[int, Iterable[int]], None] = os.sched_setaffinity,
                  passes: int = 4) -> PinResult:
    """Re-pin each task NOT inside ``cpu_list`` (sub-masks are left alone); repeat until clean."""
    allowed = parse_cpu_list(cpu_list)
    result = PinResult(pid=pid, cpus=format_cpu_list(allowed))
    seen: set[int] = set()
    for _ in range(max(1, passes)):
        moved = 0
        for tid, comm, mask in task_masks(pid, proc_root):
            seen.add(tid)
            if mask <= allowed:
                continue
            row = {"tid": tid, "comm": comm, "cpus": format_cpu_list(mask)}
            try:
                setter(tid, allowed)
            except OSError as exc:
                if getattr(exc, "errno", None) != 3:  # ESRCH: exited mid-walk, not an escape
                    result.errors.append({**row, "error": str(exc)})
                continue
            result.repinned.append(row)
            moved += 1
        if moved == 0:
            break
    result.tasks_seen = len(seen)
    result.escapes_after = escapes(pid, cpu_list, proc_root)
    return result


class LaunchAffinityError(ValueError):
    """A launch whose tasks cannot be proven inside its placement (D1: refuse)."""


def prefix_cpuset(argv: Sequence[str]) -> str | None:
    """The cpu list ``argv`` pins with ``taskset -c``; None when it does not pin."""
    argv = list(argv)
    if "taskset" not in argv:
        return None
    idx = argv.index("taskset")
    if len(argv) > idx + 2 and argv[idx + 1] == "-c":
        return argv[idx + 2]
    raise LaunchAffinityError(f"taskset without a `-c <cpu_list>` argument: {argv!r}")


def apply_pinned_launch_env(env: MutableMapping[str, str], argv: Sequence[str]) -> bool:
    """Layer 1: ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0`` on a pinned launch (overrides an
    inherited value); an unpinned launch is untouched. True when applied."""
    if prefix_cpuset(argv) is None:
        return False
    env[ROCR_AFFINITY_ENV_KEY] = ROCR_AFFINITY_ENV_VALUE
    return True


def census(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> dict[str, int]:
    """The per-task read as ``{cpu_list: task_count}``."""
    out: dict[str, int] = {}
    for _tid, _comm, mask in task_masks(pid, proc_root):
        key = format_cpu_list(mask)
        out[key] = out.get(key, 0) + 1
    return out


def enforce_pinned_child(pid: int, cpu_list: str, *, label: str = "",
                         proc_root: Path = DEFAULT_PROC_ROOT,
                         setter: Callable[[int, Iterable[int]], None] = os.sched_setaffinity,
                         launcher_pid: int | None = None) -> dict:
    """Layer 2 after health: re-pin OUR child's stray tasks, then require zero escapes.

    Returns the record (``PinResult.as_dict`` plus ``census`` and ``env_knob`` read from
    /proc/<pid>/environ). Raises ``LaunchAffinityError`` on a pid that is not our direct
    child (never touched) or on any residual escape (D1 = refuse, no warn mode).
    """
    pin = pin_own_child(pid, cpu_list, proc_root=proc_root, setter=setter,
                        launcher_pid=launcher_pid)
    if pin is None:
        raise LaunchAffinityError(f"{label or 'launch'}: pid {pid} is not a child of this "
                                  "launcher; not touched")
    record = {**pin.as_dict(), "census": census(pid, proc_root),
              "env_knob": env_knob(pid, proc_root)}
    if not pin.ok:
        raise LaunchAffinityError(f"{label or 'launch'}: tasks escape {cpu_list} after re-pin: "
                                  f"{pin.escapes_after or 'no tasks readable'}"
                                  + (f"; errors {pin.errors}" if pin.errors else ""))
    return record


def env_knob(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> str | None:
    """The live process's ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG`` value; None if unset/unreadable."""
    try:
        raw = (Path(proc_root) / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    prefix = (ROCR_AFFINITY_ENV_KEY + "=").encode()
    for item in raw.split(b"\0"):
        if item.startswith(prefix):
            return item[len(prefix):].decode(errors="replace")
    return None


class PinWatch:
    """Layer 2 for a run-to-completion child (llama-bench, test-backend-ops): poll while
    it runs, re-pin strays, record the per-task read. ``record["ok"]`` is False (D1:
    the run is refused) on a non-child pid, zero readable polls, a setter error, or a
    residual escape at any poll. Use as ``with PinWatch(p.pid, cpus) as w: p.wait()``.
    """

    def __init__(self, pid: int, cpu_list: str, *, interval_s: float = 0.5,
                 proc_root: Path = DEFAULT_PROC_ROOT,
                 setter: Callable[[int, Iterable[int]], None] = os.sched_setaffinity,
                 launcher_pid: int | None = None):
        self.pid, self.cpu_list, self.interval_s = pid, cpu_list, interval_s
        self.proc_root, self.setter = proc_root, setter
        me = os.getpid() if launcher_pid is None else launcher_pid
        self.child = parent_pid(pid, proc_root) == me
        self.polls, self.tasks_max = 0, 0
        self.repinned: list[dict] = []
        self.errors: list[dict] = []
        self.escapes_after: list[dict] = []
        self.last_census: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> None:
        if not self.child:
            return
        pin = pin_all_tasks(self.pid, self.cpu_list, proc_root=self.proc_root, setter=self.setter)
        if pin.tasks_seen == 0:
            return  # not exec'd yet, or already exited
        self.polls += 1
        self.tasks_max = max(self.tasks_max, pin.tasks_seen)
        self.repinned.extend(pin.repinned)
        self.errors.extend(pin.errors)
        self.escapes_after.extend(pin.escapes_after)
        self.last_census = census(self.pid, self.proc_root) or self.last_census

    def _run(self) -> None:
        while True:
            self.poll()
            if self._stop.wait(self.interval_s):
                return

    def __enter__(self) -> "PinWatch":
        if self.child:
            self._thread = threading.Thread(target=self._run, name="pin-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def record(self) -> dict:
        ok = (self.child and self.polls > 0 and not self.errors and not self.escapes_after)
        return {"pid": self.pid, "cpus": self.cpu_list, "child": self.child,
                "polls": self.polls, "tasks_max": self.tasks_max, "repinned": self.repinned,
                "errors": self.errors, "escapes_after": self.escapes_after,
                "census": self.last_census, "ok": ok}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="per-task affinity check / enforce")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only: list escaping tasks")
    mode.add_argument("--enforce", action="store_true",
                      help="re-pin strays of a pid whose parent is the caller, then require 0 escapes")
    parser.add_argument("pid", type=int)
    parser.add_argument("cpus")
    parser.add_argument("--launcher-pid", type=int, default=None,
                        help="the pid that spawned PID (default: this CLI's parent shell)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--proc-root", type=Path, default=DEFAULT_PROC_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.enforce:
        launcher = os.getppid() if args.launcher_pid is None else args.launcher_pid
        try:
            payload = enforce_pinned_child(args.pid, args.cpus, proc_root=args.proc_root,
                                           launcher_pid=launcher)
            ok = True
        except LaunchAffinityError as exc:
            payload, ok = {"pid": args.pid, "cpus": args.cpus, "ok": False, "refused": str(exc),
                           "census": census(args.pid, args.proc_root)}, False
    else:
        found = escapes(args.pid, args.cpus, args.proc_root)
        tasks = census(args.pid, args.proc_root)
        ok = bool(tasks) and not found
        payload = {"pid": args.pid, "cpus": args.cpus, "census": tasks, "escapes": found, "ok": ok}
    text = json.dumps(payload, indent=2 if args.json else None, sort_keys=True)
    print(text if args.json else ("OK " if ok else "FAIL ") + text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
