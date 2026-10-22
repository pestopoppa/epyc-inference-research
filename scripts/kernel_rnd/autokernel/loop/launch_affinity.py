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
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
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
