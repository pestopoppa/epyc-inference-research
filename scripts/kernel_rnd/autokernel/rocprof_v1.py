#!/usr/bin/env python3
"""`profile()` on ROCm 6.2 `rocprof` v1 `--stats` (INF48-LIVE-3: RVP-C4-5).

Same contract as `loop.hotspots.profile` (pp/tg REQUIRED, ranked `Hotspot` rows),
different profiler. It lives OUTSIDE loop/ on purpose: loop/hotspots.py sits at its
ratified code-line ceiling, and a backend is not a reason to raise it.

Why v1 `--stats` and not rocprofv2/v3: v1's stats view
(libexec/rocprofiler/dform.py gen_data_bins) writes exactly
Name,Calls,TotalDurationNs,AverageNs,Percentage to `<output>.stats.csv` -- the five
columns the counter-free kernel-trace path in intake-1219#record parses, byte for
byte. So the port is an argv change plus a glob change (`*kernel_stats*.csv` ->
`*.stats.csv`) and zero parser work. rocprofv2 has no stats view. This deliberately
does NOT depend on rocprofv3.

Standing caveats from the side-load (scripts/benchmark/rocm_profilers_env.sh): v1's
SQ/TA COUNTERS read zero on this box -- irrelevant here, --stats collects no counters
-- and v1 aborts at init on graph-enabled builds, so graphs are disabled in the env.
"""
from __future__ import annotations

import csv
import io
import os
import subprocess
import tempfile
from pathlib import Path

from .loop import residency
from .loop.hotspots import Hotspot, ProfileFailed

ROCPROF_V1_ROOT = "/mnt/raid0/llm/tools/rocm-profilers-6.2"
ROCPROF_V1_CANDIDATES = (
    f"{ROCPROF_V1_ROOT}/opt/rocm-6.2.0/bin/rocprof",
    "/opt/rocm/bin/rocprof",
)
STATS_COLUMNS = ("Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage")
STATS_GLOB = "*.stats.csv"


def resolve() -> str | None:
    for candidate in ROCPROF_V1_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def profiler_env(binary: Path, rocprof: str) -> dict[str, str]:
    """Mirror rocm_profilers_env.sh, with the measured binary's own dir FIRST so the
    arm under test loads ITS ggml (the three-ggml-generations hazard)."""
    env = residency.loader_env(binary)
    prefix = Path(rocprof).resolve().parent.parent
    root = prefix.parent.parent
    env["ROCM_PATH"] = "/opt/rocm"
    env["PATH"] = os.pathsep.join([str(prefix / "bin"), "/opt/rocm/bin",
                                   env.get("PATH", "/usr/bin:/bin")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(binary.parent), str(prefix / "lib"),
        str(root / "usr" / "lib" / "x86_64-linux-gnu"), "/opt/rocm/lib"])
    env["ROCP_METRICS"] = str(prefix / "lib" / "rocprofiler" / "metrics.xml")
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    return env


def argv(rocprof: str, out_dir: Path, binary: Path, model: Path, *,
         pp: int, tg: int) -> list[str]:
    """The same llama-bench surface `hotspots.profile` measures, under `rocprof --stats`."""
    return [rocprof, "--stats", "-o", str(out_dir / "trace.csv"), "--",
            str(binary), "-m", str(model), "-p", str(pp), "-n", str(tg),
            "-r", "1", "-ngl", "99", "-fa", "1", "-o", "json"]


def parse_stats_csv(csv_text: str, *, limit: int = 12) -> list[Hotspot]:
    """Reduce `<output>.stats.csv` to the ranked table `hotspots` produces.

    The header must be exactly the five dform.py columns; anything else is a different
    profiler and is refused, not guessed at. The share is recomputed from
    TotalDurationNs rather than read from `Percentage`, so it is exact and comparable
    with `hotspots.parse_kernel_trace`.
    """
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    if header is None:
        return []
    header = tuple(cell.strip().strip('"') for cell in header)
    if header != STATS_COLUMNS:
        raise ProfileFailed(f"not a rocprof v1 stats.csv header: {header!r}")
    totals: dict[str, list[int]] = {}
    for row in reader:
        if len(row) != len(header):
            continue
        name = row[0].strip()
        try:
            calls, total = int(row[1]), int(row[2])
        except ValueError:
            continue
        if not name or total <= 0 or calls <= 0:
            continue
        bucket = totals.setdefault(name, [0, 0])
        bucket[0] += total
        bucket[1] += calls
    grand_total = sum(duration for duration, _ in totals.values())
    if grand_total <= 0:
        return []
    rows = [Hotspot(signature=name, total_duration_ns=duration, calls=calls,
                    share_of_device_time=duration / grand_total)
            for name, (duration, calls) in totals.items()]
    rows.sort(key=lambda row: (-row.total_duration_ns, row.signature))
    return rows[:limit]


def profile(binary: Path, model: Path, *, pp: int, tg: int,
            limit: int = 12, timeout_s: int = 1800) -> list[Hotspot]:
    """`hotspots.profile` contract on rocprof v1. pp/tg are REQUIRED, as there."""
    resolved = resolve()
    if resolved is None:
        raise ProfileFailed(
            "no rocprof v1 found; tried " + ", ".join(ROCPROF_V1_CANDIDATES))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        done = subprocess.run(argv(resolved, out, binary, model, pp=pp, tg=tg),
                              capture_output=True, text=True, timeout=timeout_s,
                              env=profiler_env(binary, resolved), cwd=tmp)
        if done.returncode != 0:
            raise ProfileFailed(
                f"rocprof v1 rc={done.returncode} via {resolved}: "
                f"{(done.stderr or done.stdout)[-400:]}")
        stats = sorted(out.rglob(STATS_GLOB))
        if not stats:
            raise ProfileFailed(f"rocprof v1 produced no {STATS_GLOB}")
        return parse_stats_csv(stats[0].read_text(encoding="utf-8"), limit=limit)
