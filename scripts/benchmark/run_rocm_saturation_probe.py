#!/usr/bin/env python3
"""Run RVP-T0-1 under an exclusive GPU claim and retain numeric evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim
from scripts.benchmark import autokernel_rocm_diagnostic_beliefs as beliefs


SCHEMA = "epyc.rvp_t0_1_saturation_probe.v1"
POWER_CAP_W = 300.0
CAP_APPROACH_FRACTION = 0.90
NOMINAL_SCLK_MHZ = 1700.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def terminate_owned(proc: subprocess.Popen, *, grace_s: float = 10.0) -> int:
    proc.terminate()
    try:
        return proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=grace_s)


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"probe binary is not executable: {binary}")
    source = Path(args.source).resolve()
    if not source.is_file():
        raise RuntimeError(f"probe source is not a file: {source}")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="RVP-T0-1 evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "gemm.stdout.jsonl"
    stderr_path = output_dir / "gemm.stderr.txt"

    command = (
        str(binary), "--duration-s", str(args.duration_s),
        "--m", str(args.m), "--n", str(args.n), "--k", str(args.k),
        "--device", str(args.device_index))
    journal = device_claim.ClaimJournal(args.claim_journal)
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="RVP-T0-1 60-second gfx90a saturation probe",
        campaign_id=args.campaign_id, journal=journal,
        holder_label="run_rocm_saturation_probe.py", timeout_s=args.claim_timeout_s,
        max_hold_s=args.duration_s + 300.0)
    opened_receipt = claim.receipt().to_dict()
    sampler = device_sampler.RocmSmiSampler(
        device_index=args.device_index, interval_s=0.250)
    started_at = utc_now()
    process_started = time.monotonic()
    sampling_session = None
    sampling_receipt = None
    returncode = None
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            proc = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout_handle,
                stderr=stderr_handle, start_new_session=True)
            try:
                sampling_session = sampler.start()
                returncode = proc.wait(timeout=args.duration_s + 240.0)
            except BaseException:
                if proc.poll() is None:
                    terminate_owned(proc)
                raise
            finally:
                if sampling_session is not None:
                    sampling_receipt = sampling_session.stop()
    finally:
        released_receipt = claim.release().to_dict()

    ended_at = utc_now()
    process_duration_s = time.monotonic() - process_started
    stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if returncode != 0:
        raise RuntimeError(f"saturation workload exited {returncode}: {stderr_tail!r}")
    if sampling_receipt is None:
        raise RuntimeError("saturation workload completed without a device sampling receipt")
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"saturation workload emitted {len(lines)} JSON rows, expected 1")
    workload = json.loads(lines[0])
    if workload.get("schema") != "epyc.rocm_gemm_saturation.v1":
        raise RuntimeError("saturation workload emitted an unknown schema")
    if not str(workload.get("arch", "")).startswith("gfx90a"):
        raise RuntimeError(f"saturation workload ran on {workload.get('arch')!r}, not gfx90a")
    if float(workload.get("elapsed_s", 0)) < args.duration_s:
        raise RuntimeError("saturation workload ended before the declared duration")

    samples = sampling_receipt.samples
    powers = tuple(row.sample.power_w for row in samples)
    sclks = tuple(row.sample.sclk_mhz for row in samples)
    max_power = max(powers)
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "workload": workload,
        "workload_binary": str(binary),
        "workload_binary_sha256": sha256_file(binary),
        "workload_source": str(source),
        "workload_source_sha256": sha256_file(source),
        "workload_command": list(command),
        "process_duration_s": process_duration_s,
        "device_claim_open": opened_receipt,
        "device_claim_released": released_receipt,
        "device_sampling": sampling_receipt.to_dict(),
        "power_cap_w": POWER_CAP_W,
        "cap_approach_fraction": CAP_APPROACH_FRACTION,
        "max_power_w": max_power,
        "min_power_w": min(powers),
        "approached_power_cap": max_power >= POWER_CAP_W * CAP_APPROACH_FRACTION,
        "nominal_sclk_mhz": NOMINAL_SCLK_MHZ,
        "min_sclk_mhz": min(sclks),
        "max_sclk_mhz": max(sclks),
        "nominal_sclk_sample_fraction": (
            sum(value >= NOMINAL_SCLK_MHZ for value in sclks) / len(sclks)),
        "stderr_tail": stderr_tail,
    }
    payload = beliefs.attach_beliefs(payload, producer_path=Path(__file__).resolve())
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True)
    result.add_argument(
        "--source",
        default=str(Path(__file__).with_name("rocm_gemm_saturation.cpp")),
        help="source file used to build --binary (hash-bound into the receipt)")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="ak-rvp-t0-1-20260811")
    result.add_argument("--claim-journal",
                        default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--duration-s", type=float, default=60.0)
    result.add_argument("--m", type=int, default=8192)
    result.add_argument("--n", type=int, default=8192)
    result.add_argument("--k", type=int, default=8192)
    result.add_argument("--device-index", type=int, default=0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        print(f"RVP-T0-1 REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "samples": payload["device_sampling"]["sample_count"],
        "max_gap_s": payload["device_sampling"]["max_gap_s"],
        "max_power_w": payload["max_power_w"],
        "min_sclk_mhz": payload["min_sclk_mhz"],
        "max_sclk_mhz": payload["max_sclk_mhz"],
        "approached_power_cap": payload["approached_power_cap"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
