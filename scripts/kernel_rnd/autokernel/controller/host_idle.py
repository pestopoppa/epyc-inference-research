"""Per-invocation host-idle observation for bench cells (AK-BENCH-IDLE-GUARD).

OBSERVE ONLY. The box requires the discard threshold to be chosen by replay of stored
rows with pre/post idle readings, and no such rows exist yet: benches never recorded
idleness. So this records `host_idle` pre/post readings around each invocation and
carries `busy_threshold_pct: None`; it discards nothing and never changes a verdict.
A threshold is a later, replay-derived change once these records have accumulated.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

SCHEMA = "epyc.autokernel.host_idle.v1"
WINDOW_S = 0.5
#: Deliberately None until a replay of stored rows sets it. See the module docstring.
BUSY_THRESHOLD_PCT = None


def _parse_cpus(cpu_list: str | None) -> set[int] | None:
    if not cpu_list:
        return None
    cpus: set[int] = set()
    for part in str(cpu_list).split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _ticks(stat_text: str, cpus: set[int] | None) -> tuple[int, int]:
    busy = total = 0
    for line in stat_text.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        name, *fields = line.split()
        if cpus is not None and int(name[3:]) not in cpus:
            continue
        values = [int(value) for value in fields[:8]]
        idle = values[3] + values[4]  # idle + iowait
        total += sum(values)
        busy += sum(values) - idle
    return busy, total


def observe(phase: str, cpu_list: str | None, *, proc_root: Path | str = "/proc",
            sleep: Callable[[float], None] = time.sleep, window_s: float = WINDOW_S) -> dict:
    """One typed reading. Never raises: an unreadable host is recorded, not a failure."""
    row = {"schema": SCHEMA, "phase": phase, "cpu_list": cpu_list, "window_s": window_s,
           "busy_pct": None, "loadavg1": None, "busy_threshold_pct": BUSY_THRESHOLD_PCT,
           "error": None}
    root = Path(proc_root)
    try:
        cpus = _parse_cpus(cpu_list)
        busy0, total0 = _ticks((root / "stat").read_text(encoding="utf-8"), cpus)
        sleep(window_s)
        busy1, total1 = _ticks((root / "stat").read_text(encoding="utf-8"), cpus)
        if total1 > total0:
            row["busy_pct"] = round(100.0 * (busy1 - busy0) / (total1 - total0), 2)
        row["loadavg1"] = float((root / "loadavg").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


__all__ = ["BUSY_THRESHOLD_PCT", "SCHEMA", "WINDOW_S", "observe"]
