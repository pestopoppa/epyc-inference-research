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
import os
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
#: rocprofv3 injects an interceptor into the measured binary, which then needs
#: libhsa-amd-aqlprofile64.so.1 -- and that ships in a SECOND vendored tree, not with
#: the SDK and not in /opt/rocm. Without it the measured binary dies rc=127 before it
#: runs, and the loop reports "profile UNAVAILABLE" for a reason no part of the
#: profiler's own error text explains.
ROCPROF_SUPPORT_LIBS = (
    "/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0/lib",
)

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


def _profiler_env(binary: Path, rocprof: str) -> dict[str, str]:
    """The profiler's OWN libs must precede the system ones.

    The vendored SDK ships `lib/librocprofiler-sdk.so` and
    `lib/rocprofiler-sdk/librocprofiler-sdk-tool.so`, while /opt/rocm carries only
    `librocprofiler-register.so` -- the registration shim. Inherit the plain loader
    path and rocprofv3 finds the shim instead of its own tool library and produces no
    trace. This is the same three-generations hazard the ggml linkage rule exists for,
    one layer up.

    The measured binary's own directory still comes FIRST, because the arm under test
    must load ITS ggml, not the profiler's.
    """
    env = residency.loader_env(binary)
    sdk_root = Path(rocprof).resolve().parent.parent
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(binary.parent),
        str(sdk_root / "lib"),
        str(sdk_root / "lib" / "rocprofiler-sdk"),
        *[path for path in ROCPROF_SUPPORT_LIBS if Path(path).is_dir()],
        "/opt/rocm/lib",
    ])
    # The counter XMLs live under share/, not lib/. Pointing at lib/ makes
    # rocprofiler_iterate_agent_supported_counters abort (SIGABRT) before the trace
    # starts -- a crash whose backtrace names no missing file.
    env["ROCPROFILER_METRICS_PATH"] = str(sdk_root / "share" / "rocprofiler-sdk")
    env["ROCP_METRICS_PATH"] = str(sdk_root / "share" / "rocprofiler-sdk")
    return env


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


def profile(binary: Path, model: Path, *, pp: int, tg: int,
            limit: int = 12, timeout_s: int = 1800) -> list[Hotspot]:
    """Profile the MEASURED surface and return the ranked hotspots.

    `pp`/`tg` are REQUIRED, deliberately. They used to default to `pp=0, tg=32` and
    `run.py` never overrode them, so the loop profiled DECODE and then A/B-tested
    PREFILL: every hypothesis was aimed at a hotspot the contracted measurement
    cannot see, and no mechanism derived from that table could ever move the number.
    The loop's own critic caught it on run 8 -- "the 17.73% quantize_q8_1 hotspot is
    from the decode profile (-p 0 -n 32), while the contracted measurement is pp512".

    A default here is not a convenience; it is a silent way to profile the wrong
    thing. Callers must say which surface they are about to measure.
    """
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
                              timeout=timeout_s,
                              env=_profiler_env(binary, resolved))
        if done.returncode != 0:
            raise ProfileFailed(
                f"rocprofv3 rc={done.returncode} via {resolved}: "
                f"{(done.stderr or done.stdout)[-400:]}")
        traces = sorted(out.rglob("*kernel_trace.csv")) or sorted(out.rglob("*.csv"))
        if not traces:
            raise ProfileFailed("rocprofv3 produced no kernel trace csv")
        return parse_kernel_trace(traces[0].read_text(encoding="utf-8"), limit=limit)


__all__ = ["Hotspot", "ProfileFailed", "ROCPROF", "ROCPROF_CANDIDATES",
           "ROCPROF_SUPPORT_LIBS",
           "parse_kernel_trace", "profile"]
