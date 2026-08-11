#!/usr/bin/env python3
"""Capture an AutoKernel ROCm counter profile through Omniperf/rocprof v1.

This is the independent fallback for shapes that crash ``rocprofv2``.  It is a
diagnostic producer, never a performance verdict: a seeded correctness preflight
must pass first, every GPU run holds the device claim, and the receipt binds the
binary, source commit, profiler, counter table, and device-state sampler.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.capture_autokernel_c4_profile import (
    artifact_inventory,
    assert_source_identity,
)
from scripts.benchmark.autokernel_claimed_sampling import (
    error_payload,
    stop_sampler_and_release,
)
from scripts.benchmark.run_autokernel_gpu_factorial import (
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.omniperf_fallback.v1"
PROFILER_NAME = "omniperf-2.0.1/rocprofv1"
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_]+")
_QUANT_RE = re.compile(r"[A-Za-z0-9_]+")
_LOCALE_COMPAT = (
    "import locale,os,runpy,sys; _s=locale.setlocale; "
    "locale.setlocale=lambda c,l=None:_s(c,'C.UTF-8' if l=='en_US.UTF-8' else l); "
    "sys.path.insert(0,os.path.dirname(sys.argv[1])); sys.argv=sys.argv[1:]; "
    "runpy.run_path(sys.argv[0],run_name='__main__')"
)
_REQUIRED_COUNTERS = (
    "SQ_INSTS_VALU_INT32",
    "SQ_INSTS_VMEM_RD",
    "SQ_WAIT_ANY",
    "TCC_REQ_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
)
_REQUIRED_BINARY_FLAGS = ("--suite-seed", "--repeat-suite")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def op_pattern(args: argparse.Namespace) -> str:
    if not _QUANT_RE.fullmatch(args.quant_type):
        raise RuntimeError("--quant-type must contain only letters, digits, and underscores")
    if min(args.op_m, args.op_n, args.op_k) < 1:
        raise RuntimeError("op dimensions must be positive")
    # Deliberately stop before layout fields. Omniperf 2.0.1 sends the command
    # through rocprof v1's shell wrapper, which consumes bracket escapes once;
    # selecting the complete fixed-shape family is safer than pretending one
    # bracketed layout survived that boundary.
    return (
        rf"^type_a={args.quant_type},type_b=f32,m={args.op_m},"
        rf"n={args.op_n},k={args.op_k}.*$"
    )


def backend_command(binary: Path, args: argparse.Namespace) -> tuple[str, ...]:
    return (
        str(binary), "test", "-o", "MUL_MAT", "-b", args.backend,
        "-p", op_pattern(args), "--suite-seed", str(args.suite_seed),
        "--repeat-suite", str(args.repetitions), "--output", "csv",
    )


def omniperf_command(binary: Path, output_dir: Path,
                     args: argparse.Namespace) -> tuple[str, ...]:
    if not _IDENTIFIER_RE.fullmatch(args.workload_name):
        raise RuntimeError("--workload-name must contain only letters, digits, and underscores")
    return (
        str(Path(args.omniperf_python).resolve()), "-c", _LOCALE_COMPAT,
        str(Path(args.omniperf).resolve()), "profile",
        "-n", args.workload_name, "-p", str(output_dir),
        "-b", "SQ", "TCC", "--no-roof", "--",
        *backend_command(binary, args),
    )


def profiler_environment(binary: Path, args: argparse.Namespace) -> dict[str, str]:
    prefix = Path(args.profiler_prefix).resolve()
    profiler_root = Path(args.profiler_root).resolve()
    env = os.environ.copy()
    env.update({
        "ROCPROF": str(prefix / "bin" / "rocprof"),
        "ROCM_PATH": "/opt/rocm",
        "ROCP_METRICS": str(prefix / "lib" / "rocprofiler" / "metrics.xml"),
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "PATH": f"{prefix / 'bin'}:/opt/rocm/bin:{env['PATH']}",
        "LD_LIBRARY_PATH": (
            f"{prefix / 'lib'}:{profiler_root / 'usr/lib/x86_64-linux-gnu'}:"
            f"{binary.parent}:/opt/rocm/lib"),
    })
    return env


def run_owned(command: tuple[str, ...], *, env: dict[str, str],
              timeout_s: float) -> tuple[int, str, str, float]:
    started = time.monotonic()
    process = subprocess.Popen(
        command, env=env, stdin=subprocess.DEVNULL, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except BaseException:
        if process.poll() is None:
            terminate_owned(process)
        raise
    return process.returncode, stdout, stderr, time.monotonic() - started


def validate_binary_capabilities(binary: Path, *, env: dict[str, str]) -> dict[str, Any]:
    """Refuse an instrument that cannot reproduce the seeded active shape."""
    rc, stdout, stderr, _ = run_owned(
        (str(binary), "--help"), env=env, timeout_s=30.0)
    help_text = stdout + stderr
    missing = [flag for flag in _REQUIRED_BINARY_FLAGS if flag not in help_text]
    if missing:
        raise RuntimeError(
            "AutoKernel test instrument lacks required seeded/repeated flags: "
            + ", ".join(missing))
    return {
        "help_returncode": rc,
        "required_flags": list(_REQUIRED_BINARY_FLAGS),
    }


def python_environment_identity(python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        (str(python), "-m", "pip", "freeze", "--all"),
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=60.0)
    packages = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    payload = "\n".join(packages) + "\n"
    return {
        "packages": packages,
        "packages_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def validate_preflight(text: str, *, repetitions: int) -> dict[str, int]:
    rows = [row for row in csv.DictReader(io.StringIO(text))
            if row.get("op_name") == "MUL_MAT"]
    if not rows:
        raise RuntimeError("Omniperf fallback preflight emitted no MUL_MAT rows")
    if len(rows) % repetitions:
        raise RuntimeError("preflight row count is not divisible by --repeat-suite")
    failed = [row for row in rows if row.get("supported") != "1"
              or row.get("hard_failure") == "1" or row.get("error_message")]
    if failed:
        raise RuntimeError(f"Omniperf fallback preflight has {len(failed)} failure(s)")
    return {
        "rows": len(rows),
        "cases_per_repetition": len(rows) // repetitions,
        "repetitions": repetitions,
    }


def kernel_family(name: str) -> str:
    if "mul_mat_vec_q" in name:
        return "mul_mat_vec_q"
    if "quantize_q8_1" in name:
        return "quantize_q8_1"
    if "fillBuffer" in name:
        return "buffer_fill"
    return "other"


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in ("", None):
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Omniperf field {field} is not numeric: {value!r}") from exc


def summarize_profile(path: Path, *, quant_type: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    required = {
        "Dispatch_ID", "Kernel_Name", "Start_Timestamp", "End_Timestamp",
        "Grid_Size", "Workgroup_Size", *_REQUIRED_COUNTERS,
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Omniperf counter table is missing fields: {missing}")
    if not rows:
        raise RuntimeError("Omniperf counter table is empty")
    if not any("mul_mat_vec_q" in row["Kernel_Name"] for row in rows):
        raise RuntimeError("Omniperf counter table has no mul_mat_vec_q dispatch")
    if not any("quantize_q8_1" in row["Kernel_Name"] for row in rows):
        raise RuntimeError("Omniperf counter table has no quantize_q8_1 dispatch")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[kernel_family(row["Kernel_Name"])].append(row)
    families = []
    for family in sorted(grouped):
        selected = grouped[family]
        durations = []
        counter_totals = {field: 0.0 for field in _REQUIRED_COUNTERS}
        for row in selected:
            start = _number(row, "Start_Timestamp")
            end = _number(row, "End_Timestamp")
            if end < start:
                raise RuntimeError("Omniperf emitted an inverted device timestamp")
            durations.append(end - start)
            for field in _REQUIRED_COUNTERS:
                counter_totals[field] += _number(row, field)
        families.append({
            "family": family,
            "dispatches": len(selected),
            "device_duration_ns_total": sum(durations),
            "device_duration_ns_median": statistics.median(durations),
            "counter_totals": counter_totals,
        })
    return {
        "quant_type": quant_type,
        "rows": len(rows),
        "columns": len(fields),
        "required_counters": list(_REQUIRED_COUNTERS),
        "families": families,
    }


def tool_identity(binary: Path, source_root: Path,
                  args: argparse.Namespace) -> dict[str, Any]:
    prefix = Path(args.profiler_prefix).resolve()
    profiler = prefix / "bin" / "rocprof"
    omniperf = Path(args.omniperf).resolve()
    python_requested = Path(args.omniperf_python).absolute()
    python = python_requested.resolve()
    requirements = omniperf.parent / "requirements.txt"
    enumerator = prefix / "bin" / "rocm_agent_enumerator"
    for path, label in ((binary, "binary"), (profiler, "rocprof v1"),
                        (omniperf, "Omniperf"), (python, "Omniperf Python"),
                        (requirements, "Omniperf requirements"),
                        (enumerator, "rocm_agent_enumerator")):
        if not path.is_file() or not os.access(path, os.X_OK if label != "Omniperf requirements" else os.R_OK):
            raise RuntimeError(f"{label} is unavailable: {path}")
    commit = assert_source_identity(source_root, args.source_commit)
    return {
        "source_root": str(source_root),
        "source_commit": commit,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "profiler_name": PROFILER_NAME,
        "rocprof": str(profiler),
        "rocprof_sha256": sha256_file(profiler),
        "omniperf": str(omniperf),
        "omniperf_sha256": sha256_file(omniperf),
        "omniperf_python": str(python),
        "omniperf_python_requested": str(python_requested),
        "omniperf_python_sha256": sha256_file(python),
        "omniperf_python_environment": python_environment_identity(python_requested),
        "requirements_sha256": sha256_file(requirements),
        "agent_enumerator": str(enumerator.resolve()),
        "agent_enumerator_sha256": sha256_file(enumerator.resolve()),
        "locale_compat_sha256": hashlib.sha256(_LOCALE_COMPAT.encode()).hexdigest(),
    }


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = Path(args.binary).resolve()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="Omniperf fallback evidence directory"))
    if output_dir.exists():
        raise RuntimeError(f"Omniperf fallback output already exists: {output_dir}")
    identity = tool_identity(binary, source_root, args)
    args.source_commit = identity["source_commit"]
    env = profiler_environment(binary, args)
    identity["binary_capabilities"] = validate_binary_capabilities(binary, env=env)
    preflight_command = backend_command(binary, args)
    profile_command = omniperf_command(binary, output_dir, args)
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose=f"AutoKernel Omniperf fallback {args.quant_type}",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_omniperf_fallback.py",
        timeout_s=args.claim_timeout_s, max_hold_s=args.profile_timeout_s + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling_receipt = None
    captured_error: BaseException | None = None
    preflight_stdout = ""
    preflight_stderr = ""
    profile_stdout = ""
    profile_stderr = ""
    preflight = None
    profile = None
    started_at = utc_now()
    started = time.monotonic()
    preflight_duration = 0.0
    profile_duration = 0.0
    teardown_errors: tuple[BaseException, ...] = ()
    try:
        sampler = device_sampler.RocmSmiSampler(
            device_index=0, interval_s=0.250).start()
        rc, preflight_stdout, preflight_stderr, preflight_duration = run_owned(
            preflight_command, env=env, timeout_s=args.preflight_timeout_s)
        if rc != 0:
            raise RuntimeError(f"correctness preflight exited {rc}")
        preflight = validate_preflight(
            preflight_stdout, repetitions=args.repetitions)
        rc, profile_stdout, profile_stderr, profile_duration = run_owned(
            profile_command, env=env, timeout_s=args.profile_timeout_s)
        if rc != 0:
            raise RuntimeError(f"Omniperf exited {rc}")
    except BaseException as exc:
        captured_error = exc
    finally:
        sampling_receipt, released_receipt, teardown_errors = stop_sampler_and_release(
            sampler=sampler, claim=claim)
        released = released_receipt.to_dict() if released_receipt is not None else None

    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "preflight.stdout.csv", preflight_stdout)
    write_text(output_dir / "preflight.stderr.txt", preflight_stderr)
    write_text(output_dir / "omniperf.stdout.txt", profile_stdout)
    write_text(output_dir / "omniperf.stderr.txt", profile_stderr)
    if teardown_errors and captured_error is None:
        captured_error = teardown_errors[0]
    if captured_error is None:
        try:
            profile = summarize_profile(
                output_dir / "pmc_perf.csv", quant_type=args.quant_type)
        except BaseException as exc:
            captured_error = exc
    payload = {
        "schema": SCHEMA,
        "status": "failed" if captured_error is not None else "passed",
        "authority": "diagnostic_only",
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "identity": identity,
        "workload": {
            "quant_type": args.quant_type,
            "shape": {"m": args.op_m, "n": args.op_n, "k": args.op_k},
            "suite_seed": args.suite_seed,
            "repetitions": args.repetitions,
            "backend": args.backend,
        },
        "preflight": preflight,
        "profile": profile,
        "preflight_command": list(preflight_command),
        "profile_command": list(profile_command),
        "preflight_duration_s": preflight_duration,
        "profile_duration_s": profile_duration,
        "device_claim_open": opened,
        "device_claim_released": released,
        "device_sampling": sampling_receipt.to_dict() if sampling_receipt is not None else None,
        "teardown_errors": error_payload(teardown_errors),
        "error": None if captured_error is None else {
            "type": type(captured_error).__name__,
            "message": str(captured_error),
        },
    }
    payload["artifacts"] = artifact_inventory(output_dir)
    write_json_atomic(output_dir / "receipt.json", payload)
    if captured_error is not None:
        raise RuntimeError(
            f"Omniperf fallback failed; durable receipt: {output_dir / 'receipt.json'}") from captured_error
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True)
    result.add_argument("--source-commit")
    result.add_argument("--binary", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="autokernel-omniperf-fallback-20260811")
    result.add_argument("--workload-name", default="autokernel_iq2xxs_fallback")
    result.add_argument("--quant-type", default="iq2_xxs")
    result.add_argument("--op-m", type=int, default=16)
    result.add_argument("--op-n", type=int, default=1)
    result.add_argument("--op-k", type=int, default=256)
    result.add_argument("--suite-seed", type=int, default=4711)
    result.add_argument("--repetitions", type=int, default=5)
    result.add_argument("--backend", default="ROCm0")
    result.add_argument("--profiler-root", default="/mnt/raid0/llm/tools/rocm-profilers-6.2")
    result.add_argument("--profiler-prefix", default="/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0")
    result.add_argument("--omniperf", default="/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0/libexec/omniperf/omniperf")
    result.add_argument("--omniperf-python", default="/mnt/raid0/llm/tools/omniperf-venv-6.2/bin/python")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--preflight-timeout-s", type=float, default=300.0)
    result.add_argument("--profile-timeout-s", type=float, default=900.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.repetitions < 1:
        raise RuntimeError("--repetitions must be positive")
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "status": payload["status"],
        "preflight_rows": payload["preflight"]["rows"],
        "profile_rows": payload["profile"]["rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
