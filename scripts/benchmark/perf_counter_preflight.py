#!/usr/bin/env python3
"""No-inference preflight for EPYC canonical perf counter availability."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "amd_perf_counter_preflight.v1"

CANONICAL_PERF_EVENTS = [
    "fp_ops_retired_by_type.vector_mac",
    "fp_ops_retired_by_type.vector_all",
    "fp_ops_retired_by_type.scalar_all",
    "ls_dmnd_fills_from_sys.dram_io_all",
    "ls_hw_pf_dc_fills.dram_io_all",
    "cycles",
    "instructions",
    "task-clock",
]

SMOKE_EVENTS = ["cycles", "instructions", "task-clock"]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _first_cpuinfo_value(key: str) -> str | None:
    prefix = f"{key}\t:"
    cpuinfo = _read_text(Path("/proc/cpuinfo")) or ""
    for line in cpuinfo.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def collect_host_info() -> dict[str, Any]:
    return {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "vendor_id": _first_cpuinfo_value("vendor_id"),
        "model_name": _first_cpuinfo_value("model name"),
        "perf_event_paranoid": _read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
    }


def _event_present(perf_list_text: str, event: str) -> bool:
    # `perf list` can expose canonical aliases mid-line, e.g.
    # `cpu-cycles OR cycles [Hardware event]`. Treat each `OR` segment as an
    # exact alias rather than requiring the requested event at line start.
    for line in perf_list_text.splitlines():
        alias_text = line.split("[", 1)[0].strip()
        for alias in re.split(r"\s+OR\s+|\s+", alias_text):
            if alias.strip().strip("/") == event:
                return True
    return False


def inspect_perf_list(perf_binary: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [perf_binary, "list", "--no-desc"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def run_smoke_probe(perf_binary: str, *, duration_s: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                perf_binary,
                "stat",
                "-x,",
                "-e",
                ",".join(SMOKE_EVENTS),
                "--",
                "sleep",
                f"{duration_s:.3f}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(5.0, duration_s + 5.0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "attempted": True,
            "ok": False,
            "returncode": None,
            "stderr": str(exc),
        }

    return {
        "attempted": True,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr,
    }


def build_report(
    *,
    perf_binary: str | None = None,
    probe: bool = False,
    probe_duration_s: float = 0.1,
) -> dict[str, Any]:
    resolved_perf = perf_binary or shutil.which("perf")
    host = collect_host_info()

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "host": host,
        "perf": {
            "binary": resolved_perf,
            "found": bool(resolved_perf),
        },
        "events": {
            "canonical": CANONICAL_PERF_EVENTS,
            "present": [],
            "missing": list(CANONICAL_PERF_EVENTS),
        },
        "probe": {
            "attempted": False,
        },
        "status": "blocked",
        "recommendation": "",
    }

    if not resolved_perf:
        report["recommendation"] = (
            "Install or expose linux-tools/perf for the running kernel before "
            "using bench_canonical.sh --perf or accepting roofline evidence."
        )
        return report

    perf_list = inspect_perf_list(resolved_perf)
    report["perf"]["list_ok"] = perf_list["ok"]
    report["perf"]["list_returncode"] = perf_list["returncode"]
    if perf_list["stderr"]:
        report["perf"]["list_stderr"] = perf_list["stderr"].strip()

    if not perf_list["ok"]:
        report["recommendation"] = (
            "perf exists but `perf list --no-desc` failed; fix the perf install "
            "or permissions before collecting canonical counter evidence."
        )
        return report

    present = [
        event for event in CANONICAL_PERF_EVENTS
        if _event_present(perf_list["stdout"], event)
    ]
    missing = [event for event in CANONICAL_PERF_EVENTS if event not in present]
    report["events"]["present"] = present
    report["events"]["missing"] = missing

    if probe:
        report["probe"] = run_smoke_probe(resolved_perf, duration_s=probe_duration_s)

    if missing:
        report["status"] = "blocked"
        report["recommendation"] = (
            "Canonical AMD roofline events are not all visible in `perf list`; "
            "do not use --perf results for decision-grade roofline claims."
        )
    elif probe and not report["probe"].get("ok"):
        report["status"] = "blocked"
        report["recommendation"] = (
            "Canonical events are listed, but the smoke probe failed; check "
            "permissions and perf_event_paranoid before benchmarking."
        )
    else:
        report["status"] = "ok"
        report["recommendation"] = (
            "Canonical AMD perf events are visible"
            + (" and the smoke probe passed." if probe else ".")
        )

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AMD Perf Counter Preflight",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Status: **{report['status']}**",
        "",
        "## Host",
        "",
        f"- Model: `{report['host'].get('model_name')}`",
        f"- Vendor: `{report['host'].get('vendor_id')}`",
        f"- Kernel: `{report['host'].get('kernel')}`",
        f"- perf_event_paranoid: `{report['host'].get('perf_event_paranoid')}`",
        "",
        "## Perf",
        "",
        f"- Binary: `{report['perf'].get('binary')}`",
        f"- Found: `{report['perf'].get('found')}`",
        f"- perf list ok: `{report['perf'].get('list_ok')}`",
        "",
        "## Canonical Events",
        "",
        "| Event | Status |",
        "|---|---|",
    ]
    present = set(report["events"].get("present", []))
    for event in report["events"]["canonical"]:
        lines.append(f"| `{event}` | {'present' if event in present else 'missing'} |")
    lines.extend(
        [
            "",
            "## Probe",
            "",
            f"- Attempted: `{report['probe'].get('attempted')}`",
            f"- OK: `{report['probe'].get('ok')}`",
            "",
            "## Recommendation",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def _write_report(report: dict[str, Any], output_json: Path | None, output_md: Path | None) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-binary", help="Override perf binary path.")
    parser.add_argument("--probe", action="store_true", help="Run a short perf stat smoke probe.")
    parser.add_argument("--probe-duration-s", type=float, default=0.1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless status is ok.")
    parser.add_argument("--print-event-csv", action="store_true", help="Print the canonical event list and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_event_csv:
        print(",".join(CANONICAL_PERF_EVENTS))
        return 0

    report = build_report(
        perf_binary=args.perf_binary,
        probe=args.probe,
        probe_duration_s=args.probe_duration_s,
    )
    _write_report(report, args.output_json, args.output_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
