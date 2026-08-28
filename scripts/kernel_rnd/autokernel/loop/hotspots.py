#!/usr/bin/env python3
"""Where the device time actually goes on the tree being edited.

The old loop ran `rocprofv3 --kernel-trace` on EVERY attempt, sealed a full
per-signature table, and read one float out of it. The planner chose which kernel to
attack while blind to the profile of the tree it was editing, and its flagship
hypothesis targeted a Q5_0 path production never dispatches.

This refreshes when the champion moves, not every iteration: a profile is only stale
once the tree changes, and re-profiling per iteration would spend GPU time to learn
what it already knows. But it must not be a FROZEN list either -- every accepted patch
moves the distribution, and a static portfolio goes stale exactly when the loop starts
working.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
import tempfile

from . import residency

#: NOT /opt/rocm/bin. This host's ROCm 6.2 ships no profiler binary at all -- only
#: librocprofiler-register.so -- and the first real loop run reported
#: `profile UNAVAILABLE` because of it. The working rocprofv3 is a VENDORED SDK at a
#: non-standard root, which is what the superseded loop's sealed closure pointed at
#: (`operations/config/rocprofv3-sdk-closure.json`). Candidates are tried in order and
#: the first that exists wins, so a host that later gains a system profiler needs no
#: change here.
ROCPROF_CANDIDATES = (
    "/mnt/raid0/llm/tools/rocprofiler-sdk-6.2.0-66/opt/rocm-6.2.0/bin/rocprofv3",
    "/opt/rocm/bin/rocprofv3",
    "/opt/rocm/bin/rocprofv2",
)


def _resolve_rocprof() -> str | None:
    for candidate in ROCPROF_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


ROCPROF = _resolve_rocprof() or ROCPROF_CANDIDATES[0]


class ProfileFailed(RuntimeError):
    """The profiler did not produce a usable kernel trace."""


@dataclass(frozen=True)
class Hotspot:
    signature: str
    total_duration_ns: int
    calls: int
    share_of_device_time: float

    def to_dict(self) -> dict:
        return {"signature": self.signature,
                "total_duration_ns": self.total_duration_ns,
                "calls": self.calls,
                "share_of_device_time": self.share_of_device_time}


def parse_kernel_trace(csv_text: str, *, limit: int = 12) -> list[Hotspot]:
    """Reduce a rocprofv3 kernel trace to a ranked table.

    Ranked by total duration, which is what a planner needs: a mechanism aimed at a
    route with a negligible share cannot move the target runtime no matter how
    correct it is.
    """
    totals: dict[str, list[int]] = {}
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        name = (row.get("Kernel_Name") or row.get("kernel_name")
                or row.get("Name") or "").strip()
        if not name:
            continue
        start = row.get("Start_Timestamp") or row.get("start_ns") or "0"
        end = row.get("End_Timestamp") or row.get("end_ns") or "0"
        try:
            duration = int(end) - int(start)
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        bucket = totals.setdefault(name, [0, 0])
        bucket[0] += duration
        bucket[1] += 1

    grand_total = sum(duration for duration, _ in totals.values())
    if grand_total <= 0:
        return []
    rows = [Hotspot(signature=name, total_duration_ns=duration, calls=calls,
                    share_of_device_time=duration / grand_total)
            for name, (duration, calls) in totals.items()]
    rows.sort(key=lambda row: -row.total_duration_ns)
    return rows[:limit]


def profile(binary: Path, model: Path, *, pp: int = 0, tg: int = 32,
            limit: int = 12, timeout_s: int = 1800) -> list[Hotspot]:
    """Profile one short generation and return the ranked hotspots."""
    resolved = _resolve_rocprof()
    if resolved is None:
        raise ProfileFailed(
            "no rocprofv3 found; tried " + ", ".join(ROCPROF_CANDIDATES))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        argv = [resolved, "--kernel-trace", "-d", str(out), "-o", "trace", "--",
                str(binary), "-m", str(model), "-p", str(pp), "-n", str(tg),
                "-r", "1", "-ngl", "99", "-fa", "1", "-o", "json"]
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_s, env=residency.loader_env(binary))
        if done.returncode != 0:
            raise ProfileFailed(f"rocprofv3 rc={done.returncode}: {done.stderr[-400:]}")
        traces = sorted(out.rglob("*kernel_trace.csv")) or sorted(out.rglob("*.csv"))
        if not traces:
            raise ProfileFailed("rocprofv3 produced no kernel trace csv")
        return parse_kernel_trace(traces[0].read_text(encoding="utf-8"), limit=limit)


__all__ = ["Hotspot", "ProfileFailed", "ROCPROF", "ROCPROF_CANDIDATES",
           "parse_kernel_trace", "profile"]
