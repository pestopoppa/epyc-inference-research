#!/usr/bin/env python3
"""Bounded RVP-C4-10 PC-sampling probe for gfx90a.

Default invocation is plan-only and cannot touch the GPU.  Live execution is
separately gated by ``--execute --i-have-exclusive-gpu-window``, an exclusive
device claim, an exact clean source commit, and a hard 30-minute ceiling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROFILER_ROOT = Path("/mnt/raid0/llm/tools/rocm-profilers-6.2")
PROFILER_PREFIX = PROFILER_ROOT / "opt/rocm-6.2.0"
PROFILER = PROFILER_PREFIX / "bin/rocprofv3"
HIPCC = Path("/opt/rocm/bin/hipcc")
PROBE_SOURCE = REPO_ROOT / "scripts/benchmark/rocprofv3_pc_sampling_probe.cpp"
MAX_TOTAL_SECONDS = 1800.0
SCHEMA = "epyc.rvp.rocprofv3_pc_sampling_probe.v1"
HOST_TRAP_FIELDS = (
    "Sample_Timestamp", "Exec_Mask", "Dispatch_Id", "Instruction",
    "Instruction_Comment", "Correlation_Id",
)
STALL_FIELDS = ("Wave_Issued_Instruction", "Instruction_Type", "Stall_Reason")
HOST_TRAP_WITH_STALL_FIELDS = HOST_TRAP_FIELDS + STALL_FIELDS
PC_SAMPLING_OPTIONS = (
    "--pc-sampling-beta-enabled", "--pc-sampling-method",
    "--pc-sampling-unit", "--pc-sampling-interval",
)
CLI_OPTION_REFUSAL = re.compile(
    r"\b(?:unrecognized (?:argument|arguments|option)|unknown option|"
    r"invalid option|no such option)\b\s*:?\s*[\"']?"
    r"(--pc-sampling-(?:beta-enabled|method|unit|interval))[\"']?(?:\s|$)",
    re.IGNORECASE,
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ProbeContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")


def seal_receipt(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    sealed = dict(unsigned)
    sealed["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)).hexdigest()
    return sealed


def validate_receipt_self_hash(value: dict[str, Any]) -> None:
    observed = value.get("receipt_sha256")
    if not isinstance(observed, str) or SHA256_HEX.fullmatch(observed) is None:
        raise ProbeContractError("receipt_sha256 must be exact lowercase SHA-256")
    unsigned = dict(value)
    del unsigned["receipt_sha256"]
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise ProbeContractError("receipt_sha256 does not match canonical receipt")


def git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def assert_source_identity(expected_commit: str) -> str:
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        raise ProbeContractError("--source-commit must be an exact 40-hex commit")
    try:
        int(expected_commit, 16)
    except ValueError as exc:
        raise ProbeContractError(
            "--source-commit must be an exact 40-hex commit") from exc
    observed = git_output("rev-parse", "HEAD")
    if observed != expected_commit:
        raise ProbeContractError(
            f"source commit mismatch: expected {expected_commit}, observed {observed}")
    dirty = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ProbeContractError("research source tree must be clean")
    return observed


def profiler_environment(binary: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = "/opt/rocm"
    env["PATH"] = f"{PROFILER_PREFIX / 'bin'}:/opt/rocm/bin:{env.get('PATH', '')}"
    libraries = (
        f"{PROFILER_PREFIX / 'lib'}:"
        f"{PROFILER_ROOT / 'usr/lib/x86_64-linux-gnu'}:"
        f"{binary.parent}:/opt/rocm/lib")
    env["LD_LIBRARY_PATH"] = libraries
    return env


def build_command(binary: Path) -> list[str]:
    return [
        str(HIPCC), "-O2", "--offload-arch=gfx90a", "-std=c++17",
        str(PROBE_SOURCE), "-o", str(binary),
    ]


def list_command() -> list[str]:
    return [str(PROFILER), "-L"]


def profile_command(binary: Path, raw_dir: Path) -> list[str]:
    return [
        str(PROFILER), "--pc-sampling-beta-enabled",
        "--pc-sampling-method", "host_trap",
        "--pc-sampling-unit", "time",
        "--pc-sampling-interval", "1",
        "--output-format", "csv", "-d", str(raw_dir),
        "--", str(binary),
    ]


def prepared_plan(output_dir: Path) -> dict[str, Any]:
    binary = output_dir / "pc_sampling_probe"
    return {
        "schema": SCHEMA,
        "status": "prepared_not_run",
        "authority": "diagnostic_only",
        "target": "gfx90a",
        "method": "host_trap",
        "max_total_seconds": MAX_TOTAL_SECONDS,
        "commands": {
            "build": build_command(binary),
            "list": list_command(),
            "profile": profile_command(binary, output_dir / "raw"),
        },
        "claim_rule": "exclusive mi210_0 device claim required",
        "residency_claim": (
            "none; this probe does not bind both KFD and VRAM residency"),
        "interpretation_rule": (
            "classify only emitted records or an exact CLI refusal; documentation "
            "alone is not evidence of absence"),
    }


def is_exact_profile_invocation(command: Sequence[str]) -> bool:
    if len(command) != 14:
        return False
    expected = profile_command(Path(command[13]), Path(command[11]))
    return tuple(command) == tuple(expected)


def classify_cli_failure(stderr: str, *, command: Sequence[str]) -> str:
    if not is_exact_profile_invocation(command):
        return "infrastructure_profiler_failure"
    match = CLI_OPTION_REFUSAL.search(stderr)
    if match is not None and match.group(1).casefold() in PC_SAMPLING_OPTIONS:
        return "pc_sampling_cli_unavailable_on_rocm_6_2"
    return "infrastructure_profiler_failure"


def classify_host_trap_csv(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        raise ProbeContractError("PC-sampling CSV must not be blank")
    if "\r" in text:
        raise ProbeContractError("PC-sampling CSV must use exact LF newlines")
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ProbeContractError(
            "PC-sampling CSV must have exactly one trailing LF")
    try:
        parsed = list(csv.reader(io.StringIO(text), strict=True))
    except (csv.Error, UnicodeError) as exc:
        raise ProbeContractError("PC-sampling CSV is malformed") from exc
    if not parsed or any(not row for row in parsed):
        raise ProbeContractError("PC-sampling CSV contains a blank row")
    fieldnames = tuple(parsed[0])
    if fieldnames not in (HOST_TRAP_FIELDS, HOST_TRAP_WITH_STALL_FIELDS):
        raise ProbeContractError("PC-sampling CSV header is not an exact schema")
    rows = parsed[1:]
    for row in rows:
        if len(row) != len(fieldnames):
            raise ProbeContractError("PC-sampling CSV row width is not exact")
        if any(not row[index].strip() for index in range(len(HOST_TRAP_FIELDS))):
            raise ProbeContractError(
                "PC-sampling CSV required base field is blank")
    if not rows:
        return {
            "classification": "inconclusive_no_samples",
            "record_count": 0, "fields": list(fieldnames)}
    present_stall_fields = (
        STALL_FIELDS if fieldnames == HOST_TRAP_WITH_STALL_FIELDS else ())
    if not present_stall_fields:
        classification = "host_trap_hotspot_only_no_stall_reason_fields"
    else:
        populated = any(
            row[index].strip() not in {"", "0", "N/A", "None"}
            for row in rows
            for index in range(len(HOST_TRAP_FIELDS), len(fieldnames)))
        classification = (
            "unexpected_stall_reason_input_review_required" if populated else
            "host_trap_stall_reason_fields_unpopulated")
    return {
        "classification": classification,
        "record_count": len(rows),
        "fields": list(fieldnames),
        "stall_fields": list(present_stall_fields),
    }


def run_command(command: Sequence[str], *, env: dict[str, str] | None,
                stdout: Path, stderr: Path, timeout_s: float) -> tuple[int, float]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ProbeContractError("command timeout must be positive and finite")
    started = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            tuple(command), env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_s)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    process.wait(timeout=10)
            raise
    return returncode, time.monotonic() - started


def remaining(started: float, requested: float) -> float:
    left = MAX_TOTAL_SECONDS - (time.monotonic() - started)
    if left <= 0:
        raise ProbeContractError("30-minute total probe ceiling exhausted")
    return min(requested, left)


def inventory(root: Path) -> list[dict[str, Any]]:
    return [{
        "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in sorted(root.rglob("*"))
      if path.is_file() and path.name != "receipt.json"]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not args.i_have_exclusive_gpu_window:
        raise ProbeContractError(
            "--execute requires --i-have-exclusive-gpu-window")
    if args.profile_timeout_s > 900 or args.build_timeout_s > 300:
        raise ProbeContractError(
            "profile/build timeouts may not exceed 900/300 seconds")
    source_commit = assert_source_identity(args.source_commit)
    if not PROFILER.is_file() or not HIPCC.is_file():
        raise ProbeContractError("pinned rocprofv3 or hipcc is unavailable")

    from scripts.benchmark.autokernel_claimed_sampling import (
        error_payload, stop_sampler_and_release)
    from scripts.kernel_rnd.autokernel import storage
    from scripts.kernel_rnd.autokernel.execution import device_sampler
    from scripts.kernel_rnd.autokernel.resource import device_claim

    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="RVP-C4-10 PC-sampling evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    binary = output_dir / "pc_sampling_probe"
    started_at = utc_now()
    started = time.monotonic()
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="RVP-C4-10 bounded rocprofv3 PC-sampling probe",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_rocprofv3_pc_sampling_probe.py",
        timeout_s=min(args.claim_timeout_s, 60.0),
        max_hold_s=MAX_TOTAL_SECONDS)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling = None
    released = None
    teardown_errors: list[BaseException] = []
    failure: BaseException | None = None
    payload: dict[str, Any] = {}
    try:
        sampler = device_sampler.RocmSmiSampler(
            device_index=0, interval_s=0.250).start()
        build = build_command(binary)
        build_rc, build_s = run_command(
            build, env=None, stdout=output_dir / "build.stdout.txt",
            stderr=output_dir / "build.stderr.txt",
            timeout_s=remaining(started, args.build_timeout_s))
        if build_rc != 0 or not binary.is_file():
            raise ProbeContractError(f"probe build failed with rc={build_rc}")
        listing = list_command()
        list_rc, list_s = run_command(
            listing, env=profiler_environment(binary),
            stdout=output_dir / "list.stdout.txt",
            stderr=output_dir / "list.stderr.txt",
            timeout_s=remaining(started, 120.0))
        profile = profile_command(binary, raw_dir)
        profile_rc, profile_s = run_command(
            profile, env=profiler_environment(binary),
            stdout=output_dir / "profile.stdout.txt",
            stderr=output_dir / "profile.stderr.txt",
            timeout_s=remaining(started, args.profile_timeout_s))
        if profile_rc != 0:
            classification = classify_cli_failure(
                (output_dir / "profile.stderr.txt").read_text(
                    encoding="utf-8", errors="replace"), command=profile)
            analysis = {"classification": classification, "record_count": 0}
        else:
            candidates = [
                path for path in sorted(raw_dir.rglob("*.csv"))
                if "pc_sampling_host_trap" in path.name]
            if len(candidates) != 1:
                analysis = {
                    "classification": "inconclusive_output_cardinality",
                    "record_count": 0,
                    "candidate_csvs": [str(path.relative_to(output_dir))
                                       for path in candidates],
                }
            else:
                analysis = classify_host_trap_csv(candidates[0].read_text(
                    encoding="utf-8", errors="strict"))
                analysis["csv"] = str(candidates[0].relative_to(output_dir))
                analysis["csv_sha256"] = sha256_file(candidates[0])
        payload = {
            "schema": SCHEMA, "status": "observed",
            "authority": "diagnostic_only", "target": "gfx90a",
            "method": "host_trap", "campaign_id": args.campaign_id,
            "started_at": started_at, "ended_at": utc_now(),
            "duration_s": time.monotonic() - started,
            "source": {
                "commit": source_commit,
                "probe_path": str(PROBE_SOURCE.relative_to(REPO_ROOT)),
                "probe_sha256": sha256_file(PROBE_SOURCE),
            },
            "toolchain": {
                "rocprofv3": str(PROFILER),
                "rocprofv3_sha256": sha256_file(PROFILER),
                "hipcc": str(HIPCC), "hipcc_sha256": sha256_file(HIPCC),
            },
            "commands": {"build": build, "list": listing, "profile": profile},
            "returncodes": {"build": build_rc, "list": list_rc,
                            "profile": profile_rc},
            "timing_s": {"build": build_s, "list": list_s,
                         "profile": profile_s},
            "analysis": analysis,
            "claim_boundary": (
                "This result describes only the captured in-window run. It does "
                "not infer absent fields from documentation or from a missed window. "
                "Device sampling is telemetry only: this probe does not bind both "
                "KFD and VRAM residency and makes no HIP-residency claim."),
            "device_claim": {"opened": opened},
        }
    except BaseException as exc:
        failure = exc
    finally:
        sampling, released_receipt, teardown_errors = stop_sampler_and_release(
            sampler=sampler, claim=claim)
        released = released_receipt.to_dict() if released_receipt else None
    if teardown_errors and failure is None:
        failure = teardown_errors[0]
    if not payload:
        payload = {
            "schema": SCHEMA, "status": "failed",
            "authority": "diagnostic_only", "target": "gfx90a",
            "method": "host_trap", "campaign_id": args.campaign_id,
            "started_at": started_at, "ended_at": utc_now(),
            "duration_s": time.monotonic() - started,
            "source": {"commit": source_commit},
            "failure": repr(failure), "device_claim": {"opened": opened},
        }
    if sampling is not None:
        payload["device_sampling"] = sampling.to_dict()
    payload["device_claim"]["released"] = released
    payload["teardown_errors"] = error_payload(teardown_errors)
    if failure is not None:
        payload["status"] = "failed"
        payload["failure"] = repr(failure)
    payload["artifacts"] = inventory(output_dir)
    payload = seal_receipt(payload)
    validate_receipt_self_hash(payload)
    write_json_atomic(output_dir / "receipt.json", payload)
    if failure is not None:
        raise failure
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="rvp-c4-10")
    result.add_argument("--source-commit")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--i-have-exclusive-gpu-window", action="store_true")
    result.add_argument("--build-timeout-s", type=float, default=300.0)
    result.add_argument("--profile-timeout-s", type=float, default=900.0)
    result.add_argument("--claim-timeout-s", type=float, default=60.0)
    result.add_argument(
        "--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.execute:
        print(json.dumps(prepared_plan(Path(args.output_dir)), sort_keys=True))
        return 0
    payload = execute(args)
    print(json.dumps({
        "status": payload["status"],
        "classification": payload.get("analysis", {}).get("classification"),
        "receipt": str(Path(args.output_dir).resolve() / "receipt.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
