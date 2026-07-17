#!/usr/bin/env python3
"""Clean-host preflight for CPU llama-bench A/B runs.

K34 showed that same-source CPU guard cells can swing by >2x when host/runtime
state drifts. This preflight records the state that matters before expensive CPU
A/Bs and can optionally run the cheap recovered frontdoor decode sentinel.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


SCHEMA = "epyc_cpu_bench_clean_preflight.v1"

EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
DEFAULT_BINARY = EXPERIMENTAL_ROOT / "build-k24-cpu" / "bin" / "llama-bench"
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")
DEFAULT_SENTINEL_MIN_TPS = 18.0
DEFAULT_THREADS = 96
DEFAULT_TOKENS = 256
DEFAULT_REPS = 3
DEFAULT_LIBRARY_SUFFIX = "/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib"

PROCESS_BASENAME_BLOCKERS = {"llama-server", "llama-bench", "llama-cli", "rocprof", "rocprofv2"}
AUTOPILOT_MARKERS = (
    "scripts/autopilot/autopilot.py start",
    "start_fable_authority_daemon.py",
    "autopilot_supervisor.py",
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def run_capture(
    argv: list[str],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "argv": argv,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "argv": argv,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_ps(stdout: str, *, current_pid: int | None = None) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_text, comm, args = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if current_pid is not None and pid == current_pid:
            continue

        reason = None
        argv0 = Path(args.split(maxsplit=1)[0]).name if args else comm
        if comm in PROCESS_BASENAME_BLOCKERS or argv0 in PROCESS_BASENAME_BLOCKERS:
            reason = f"blocked process basename {comm!r}/{argv0!r}"
        elif any(marker in args for marker in AUTOPILOT_MARKERS):
            reason = "blocked AutoPilot process"
        elif comm == "perf" and (" stat " in f" {args} " or " record " in f" {args} "):
            reason = "blocked perf profiler"

        if reason:
            blockers.append({"pid": pid, "comm": comm, "args": args, "reason": reason})
    return blockers


def collect_process_blockers(
    *, runner: Runner = subprocess.run, current_pid: int | None = None
) -> dict[str, Any]:
    proc = run_capture(
        ["ps", "-eo", "pid=,comm=,args="],
        runner=runner,
        timeout=10,
    )
    if not proc["ok"]:
        return {"ok": False, "blockers": [], "error": proc["stderr"], "ps": proc}
    return {
        "ok": True,
        "blockers": parse_ps(proc["stdout"], current_pid=current_pid or os.getpid()),
    }


def active_thp_mode(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split():
        if part.startswith("[") and part.endswith("]"):
            return part[1:-1]
    return None


def unique_sysfs_values(pattern: str) -> dict[str, Any]:
    values: dict[str, int] = {}
    unreadable = 0
    paths = sorted(glob.glob(pattern))
    for path_text in paths:
        value = read_text(Path(path_text))
        if value is None:
            unreadable += 1
            continue
        values[value] = values.get(value, 0) + 1
    return {"pattern": pattern, "count": len(paths), "values": values, "unreadable": unreadable}


def collect_host_state(*, runner: Runner = subprocess.run) -> dict[str, Any]:
    commands = {
        "numactl_hardware": run_capture(["numactl", "--hardware"], runner=runner, timeout=20),
        "free_h": run_capture(["free", "-h"], runner=runner, timeout=20),
        "lscpu": run_capture(["lscpu"], runner=runner, timeout=20),
    }
    return {
        "commands": commands,
        "governors": unique_sysfs_values("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"),
        "energy_performance_preference": unique_sysfs_values(
            "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference"
        ),
        "boost": read_text(Path("/sys/devices/system/cpu/cpufreq/boost")),
        "transparent_hugepage": {
            "enabled": read_text(Path("/sys/kernel/mm/transparent_hugepage/enabled")),
            "enabled_active": active_thp_mode(
                read_text(Path("/sys/kernel/mm/transparent_hugepage/enabled"))
            ),
            "defrag": read_text(Path("/sys/kernel/mm/transparent_hugepage/defrag")),
            "defrag_active": active_thp_mode(
                read_text(Path("/sys/kernel/mm/transparent_hugepage/defrag"))
            ),
        },
        "numa_balancing": read_text(Path("/proc/sys/kernel/numa_balancing")),
        "perf_event_paranoid": read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
    }


def default_library_path(binary: Path) -> str:
    return f"{binary.parent}:{DEFAULT_LIBRARY_SUFFIX}"


def build_sentinel_command(
    *,
    binary: Path,
    model: Path,
    threads: int,
    tokens: int,
    reps: int,
    numa: str | None,
) -> list[str]:
    argv = [
        str(binary),
        "-m",
        str(model),
        "-p",
        "0",
        "-n",
        str(tokens),
        "-r",
        str(reps),
        "-t",
        str(threads),
        "-ngl",
        "0",
        "-dev",
        "none",
        "-o",
        "json",
    ]
    if numa:
        argv.extend(["--numa", numa])
    return argv


def parse_llama_bench_avg_ts(stdout: str) -> float | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    rows = payload if isinstance(payload, list) else [payload]
    values: list[float] = []
    for row in rows:
        if isinstance(row, dict) and row.get("avg_ts") is not None:
            try:
                values.append(float(row["avg_ts"]))
            except (TypeError, ValueError):
                pass
    if not values:
        return None
    return sum(values) / len(values)


def run_sentinel(
    *,
    binary: Path,
    model: Path,
    library_path: str,
    threads: int,
    tokens: int,
    reps: int,
    numa: str | None,
    runner: Runner = subprocess.run,
    timeout: float = 180.0,
) -> dict[str, Any]:
    argv = build_sentinel_command(
        binary=binary,
        model=model,
        threads=threads,
        tokens=tokens,
        reps=reps,
        numa=numa,
    )
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = library_path
    env["GGML_IQK"] = "1"
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "attempted": True,
            "ok": False,
            "argv": argv,
            "env": {"LD_LIBRARY_PATH": library_path, "GGML_IQK": "1"},
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "avg_ts": None,
        }
    return {
        "attempted": True,
        "ok": proc.returncode == 0,
        "argv": argv,
        "env": {"LD_LIBRARY_PATH": library_path, "GGML_IQK": "1"},
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "avg_ts": parse_llama_bench_avg_ts(proc.stdout),
    }


def collect_build_info(
    *, binary: Path, source_root: Path, library_path: str, runner: Runner = subprocess.run
) -> dict[str, Any]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = library_path
    help_probe = run_capture([str(binary), "--help"], runner=runner, timeout=20, env=env)
    git_head = run_capture(["git", "-C", str(source_root), "rev-parse", "HEAD"], runner=runner, timeout=20)
    git_status = run_capture(["git", "-C", str(source_root), "status", "--short"], runner=runner, timeout=20)
    return {
        "binary": str(binary),
        "binary_exists": binary.exists(),
        "model_safe_binary": str(binary).startswith(str(source_root)),
        "source_root": str(source_root),
        "library_path": library_path,
        "help_probe_ok": help_probe["ok"],
        "binary_build_commit_note": "llama-bench emits build_commit in benchmark JSON; run --run-sentinel to record the binary build id.",
        "git_head": git_head["stdout"].strip() if git_head["ok"] else None,
        "git_status_short": git_status["stdout"].splitlines() if git_status["ok"] else None,
    }


def host_warnings(host: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    governors = host["governors"]["values"]
    if governors and set(governors) != {"performance"}:
        warnings.append(f"CPU governors are not uniformly performance: {governors}")
    epp = host["energy_performance_preference"]["values"]
    if epp and any(value not in {"performance", "balance_performance"} for value in epp):
        warnings.append(f"energy_performance_preference includes non-performance values: {epp}")
    if host.get("boost") not in {None, "1"}:
        warnings.append(f"CPU boost is not enabled: {host.get('boost')}")
    thp = host["transparent_hugepage"]
    if thp.get("enabled_active") != "always":
        warnings.append(f"THP enabled active mode is {thp.get('enabled_active')!r}, expected 'always'")
    if thp.get("defrag_active") != "always":
        warnings.append(f"THP defrag active mode is {thp.get('defrag_active')!r}, expected 'always'")
    if host.get("numa_balancing") not in {None, "0"}:
        warnings.append(f"kernel.numa_balancing is {host.get('numa_balancing')!r}, expected '0'")
    return warnings


def decide_status(report: dict[str, Any], *, min_frontdoor_tps: float) -> tuple[str, str]:
    if report["processes"].get("blockers"):
        return (
            "blocked",
            "Stop AutoPilot, llama-server/bench/CLI, and profiler processes before collecting CPU A/B evidence.",
        )
    if not report["build"]["binary_exists"]:
        return "blocked", "Build or point --binary at the intended llama-bench before running CPU A/Bs."
    if report["sentinel"]["attempted"]:
        if not report["sentinel"].get("ok") or report["sentinel"].get("avg_ts") is None:
            return "blocked", "Frontdoor sentinel failed or did not emit parseable llama-bench JSON."
        if float(report["sentinel"]["avg_ts"]) < min_frontdoor_tps:
            return (
                "retry",
                "Frontdoor sentinel is below the recovered K34.1 band; clear/retry host state before blaming source.",
            )
    if report["host_warnings"]:
        return "warn", "Host drift warnings recorded; fix or explicitly waive before decision-grade CPU A/Bs."
    return "ok", "Clean-run preflight passed."


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    binary = args.binary.expanduser().resolve()
    model = args.model.expanduser().resolve()
    library_path = args.library_path or default_library_path(binary)
    host = collect_host_state(runner=runner)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "execute_sentinel" if args.run_sentinel else "record_only",
        "processes": collect_process_blockers(runner=runner, current_pid=os.getpid()),
        "host": host,
        "host_warnings": host_warnings(host),
        "build": collect_build_info(
            binary=binary,
            source_root=args.source_root.expanduser().resolve(),
            library_path=library_path,
            runner=runner,
        ),
        "sentinel": {"attempted": False},
        "policy": {
            "frontdoor_recovered_reference_tps": 20.57,
            "frontdoor_retry_threshold_tps": args.min_frontdoor_tps,
            "retry_before_source_blame": True,
        },
    }
    if args.run_sentinel:
        report["sentinel"] = run_sentinel(
            binary=binary,
            model=model,
            library_path=library_path,
            threads=args.threads,
            tokens=args.tokens,
            reps=args.reps,
            numa=args.numa,
            runner=runner,
            timeout=args.timeout,
        )
    status, recommendation = decide_status(report, min_frontdoor_tps=args.min_frontdoor_tps)
    report["status"] = status
    report["recommendation"] = recommendation
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--source-root", type=Path, default=EXPERIMENTAL_ROOT)
    parser.add_argument("--library-path", default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-sentinel", action="store_true")
    parser.add_argument("--min-frontdoor-tps", type=float, default=DEFAULT_SENTINEL_MIN_TPS)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--numa", choices=["distribute"], default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    payload = canonical_json(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.strict and report["status"] not in {"ok", "warn"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
