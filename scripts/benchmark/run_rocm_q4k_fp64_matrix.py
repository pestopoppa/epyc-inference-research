#!/usr/bin/env python3
"""Characterize the stock gfx90a Q4_K fp64 ratio gate across build cells."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.run_autokernel_gpu_factorial import (
    cache_bool,
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.rvp_c2_6a_rocm_q4k_fp64_matrix.v1"
METRIC_ID = "fp64_error_ratio/host-double-gguf-wire/v1"
PROPERTY_RE = re.compile(
    r"AK_PROP_V2 metric=(?P<metric>\S+) residual=(?P<residual>\S+) "
    r"tolerance=(?P<tolerance>\S+) passed=(?P<passed>[01]) "
    r"suite_seed=(?P<seed>\d+) transform=(?P<transform>\S+)"
)


def resolve_variants(build_dirs: list[str]) -> list[dict]:
    variants = []
    seen = set()
    for value in build_dirs:
        root = Path(value).resolve()
        binary = root / "bin" / "test-backend-ops"
        cache = root / "CMakeCache.txt"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"test-backend-ops is not executable: {binary}")
        flags = {
            "rocwmma_fattn": cache_bool(cache, "GGML_HIP_ROCWMMA_FATTN"),
            "mmq_mfma": cache_bool(cache, "GGML_HIP_MMQ_MFMA"),
            "force_cublas": cache_bool(cache, "GGML_CUDA_FORCE_CUBLAS"),
            "force_mmq": cache_bool(cache, "GGML_CUDA_FORCE_MMQ"),
        }
        identity = tuple(sorted(flags.items()))
        if identity in seen:
            raise RuntimeError(f"duplicate Q4_K dispatch build flags: {flags}")
        seen.add(identity)
        variants.append({
            "build_root": str(root),
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            **flags,
        })
    return variants


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_property_receipt(text: str, *, suite_seed: int) -> dict:
    match = PROPERTY_RE.fullmatch(text)
    if match is None or match.group("metric") != METRIC_ID:
        raise RuntimeError(f"malformed fp64 property receipt: {text!r}")
    if int(match.group("seed")) != suite_seed:
        raise RuntimeError("fp64 property receipt suite seed drifted")
    residual = float(match.group("residual"))
    tolerance = float(match.group("tolerance"))
    passed = match.group("passed") == "1"
    sibling_error = tolerance / 1.5
    return {
        "metric_id": METRIC_ID,
        "candidate_error": residual,
        "sibling_error": sibling_error,
        "tolerance": tolerance,
        "error_ratio": residual / max(sibling_error, 1e-12),
        "passed": passed,
        "transform": match.group("transform"),
    }


def run_arm(*, variant: dict, args: argparse.Namespace, output_dir: Path) -> dict:
    build_root = Path(variant["build_root"])
    binary = build_root / "bin" / "test-backend-ops"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"test-backend-ops is not executable: {binary}")
    if variant["force_cublas"]:
        arm_id = "force-cublas"
    elif variant["force_mmq"]:
        arm_id = "force-mmq"
    else:
        arm_id = f"r{int(variant['rocwmma_fattn'])}m{int(variant['mmq_mfma'])}"
    command = (
        str(binary), "test", "-o", "MUL_MAT", "-b", "ROCm0",
        "-p", r"^type_a=q4_K,type_b=f32.*$",
        "--suite-seed", str(args.suite_seed), "--autokernel-properties",
        "--output", "csv",
    )
    stdout_path = output_dir / f"{arm_id}.stdout.csv"
    stderr_path = output_dir / f"{arm_id}.stderr.txt"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=stdout_handle,
            stderr=stderr_handle, start_new_session=True)
        try:
            returncode = process.wait(timeout=args.arm_timeout_s)
        except BaseException:
            if process.poll() is None:
                terminate_owned(process)
            raise

    rows = []
    with stdout_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            receipt = raw.get("property_receipt", "")
            if METRIC_ID not in receipt:
                continue
            parsed = parse_property_receipt(receipt, suite_seed=args.suite_seed)
            parsed.update({
                "op_params": raw["op_params"],
                "supported": raw["supported"] == "1",
                "hard_failure": raw["hard_failure"] == "1",
                "error_message": raw["error_message"],
                "reference_receipt": raw["reference_receipt"],
            })
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"{arm_id} emitted no {METRIC_ID} receipts")
    failures = [row for row in rows if not row["passed"]]
    expected_returncode = 1 if failures else 0
    if returncode != expected_returncode:
        raise RuntimeError(
            f"{arm_id} exited {returncode}, expected {expected_returncode} for "
            f"{len(failures)} fp64 failure(s)")
    if any(not row["supported"] or row["hard_failure"] for row in rows):
        raise RuntimeError(f"{arm_id} mixed unsupported/hard-failure rows into the fp64 matrix")
    return {
        "arm_id": arm_id,
        "rocwmma_fattn": variant["rocwmma_fattn"],
        "mmq_mfma": variant["mmq_mfma"],
        "force_cublas": variant["force_cublas"],
        "force_mmq": variant["force_mmq"],
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "command": list(command),
        "duration_s": time.monotonic() - started,
        "returncode": returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "cases": rows,
        "case_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "max_error_ratio": max(row["error_ratio"] for row in rows),
    }


def run(args: argparse.Namespace) -> dict:
    variants = resolve_variants(args.build_dir)
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="RVP-C2-6a evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="RVP-C2-6a stock ROCm Q4_K fp64 matrix",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_rocm_q4k_fp64_matrix.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=4 * args.arm_timeout_s + 120.0)
    opened_receipt = claim.receipt().to_dict()
    session = None
    sampling_receipt = None
    arms = []
    started_at = utc_now()
    started_mono = time.monotonic()
    try:
        session = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        for variant in variants:
            arms.append(run_arm(
                variant=variant, args=args, output_dir=output_dir))
    finally:
        if session is not None:
            sampling_receipt = session.stop()
        released_receipt = claim.release().to_dict()
    if sampling_receipt is None:
        raise RuntimeError("RVP-C2-6a completed without a device sampling receipt")

    by_mfma = {}
    for enabled in (False, True):
        selected = [arm for arm in arms if arm["mmq_mfma"] is enabled]
        if not selected:
            continue
        by_mfma[str(enabled).lower()] = {
            "arms": len(selected),
            "cases": sum(arm["case_count"] for arm in selected),
            "failures": sum(arm["failure_count"] for arm in selected),
            "max_error_ratio": max(arm["max_error_ratio"] for arm in selected),
        }
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "metric_id": METRIC_ID,
        "kappa": 1.5,
        "suite_seed": args.suite_seed,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started_mono,
        "variants": variants,
        "arms": arms,
        "summary": {
            "cases": sum(arm["case_count"] for arm in arms),
            "failures": sum(arm["failure_count"] for arm in arms),
            "by_mmq_mfma": by_mfma,
        },
        "device_claim_open": opened_receipt,
        "device_claim_released": released_receipt,
        "device_sampling": sampling_receipt.to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--build-dir", action="append", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="rvp-c2-6a-20260811")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--arm-timeout-s", type=float, default=300.0)
    result.add_argument("--suite-seed", type=int, default=4711)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "cases": payload["summary"]["cases"],
        "failures": payload["summary"]["failures"],
        "by_mmq_mfma": payload["summary"]["by_mmq_mfma"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
