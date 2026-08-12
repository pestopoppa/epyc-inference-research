#!/usr/bin/env python3
"""Run AutoKernel hostile-distribution and checker-isolation probes on ROCm."""
from __future__ import annotations

import argparse
import csv
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

from scripts.benchmark.run_autokernel_gpu_factorial import (
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.evaluator import oracle_integrity
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.oracle_integrity.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def source_identity(source_tree: Path) -> tuple[str, str]:
    source_tree = source_tree.resolve()
    frozen = Path("/mnt/raid0/llm/llama.cpp").resolve()
    if source_tree == frozen or frozen in source_tree.parents:
        raise RuntimeError("oracle-integrity probes refuse the frozen production tree")
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=source_tree, text=True).strip()
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=source_tree, text=True)
    if status:
        raise RuntimeError(
            "oracle-integrity source tree is dirty; receipts require a durable suite identity")
    return head, head[:9]


def run_probe(*, probe: str, binary: Path, args: argparse.Namespace,
              output_dir: Path) -> dict:
    flag = {
        "hostile": "--autokernel-hostile-distributions",
        "checker": "--autokernel-properties",
    }[probe]
    command = [
        str(binary), "test", "-o", args.ops, "-b", args.backend,
        "--suite-seed", str(args.suite_seed), "--output", "csv", flag,
    ]
    if args.params:
        command[4:4] = ["-p", args.params]
    stdout_path = output_dir / f"{probe}.stdout.csv"
    stderr_path = output_dir / f"{probe}.stderr.txt"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=stdout_handle,
            stderr=stderr_handle, start_new_session=True)
        try:
            returncode = process.wait(timeout=args.probe_timeout_s)
        except BaseException:
            if process.poll() is None:
                terminate_owned(process)
            raise
    with stdout_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "probe": probe, "command": command, "returncode": returncode,
        "duration_s": time.monotonic() - started,
        "stdout": str(stdout_path), "stderr": str(stderr_path),
        "row_count": len(rows), "rows": rows,
    }


def check_to_dict(check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def run(args: argparse.Namespace) -> dict:
    source_tree = Path(args.source_tree).resolve()
    binary = Path(args.binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"test-backend-ops is not executable: {binary}")
    head, suite_version = source_identity(source_tree)
    if source_tree not in binary.parents:
        raise RuntimeError("oracle-integrity binary is not inside its declared source tree")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AutoKernel oracle-integrity evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)

    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="RVP-C2-8/C2-9 ROCm oracle-integrity probes",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_oracle_integrity.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=2 * args.probe_timeout_s + 120.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling = None
    probes = []
    started_at = utc_now()
    started = time.monotonic()
    try:
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        probes.append(run_probe(
            probe="hostile", binary=binary, args=args, output_dir=output_dir))
        probes.append(run_probe(
            probe="checker", binary=binary, args=args, output_dir=output_dir))
    finally:
        if sampler is not None:
            sampling = sampler.stop()
        released = claim.release().to_dict()
    if sampling is None:
        raise RuntimeError("oracle-integrity probes completed without device sampling")

    hostile = oracle_integrity.evaluate_hostile_rows(
        probes[0]["rows"], expected_seed=args.suite_seed,
        expected_suite_version=suite_version)
    checker = oracle_integrity.evaluate_checker_rows(
        probes[1]["rows"], expected_suite_version=suite_version)
    for probe in probes:
        probe.pop("rows")
    payload = {
        "schema": SCHEMA, "campaign_id": args.campaign_id,
        "started_at": started_at, "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "source_tree": str(source_tree), "source_commit": head,
        "suite_version": suite_version, "binary": str(binary),
        "binary_sha256": sha256_file(binary), "backend": args.backend,
        "suite_seed": args.suite_seed, "ops": args.ops, "params": args.params,
        "probes": probes, "hostile_distribution_check": check_to_dict(hostile),
        "checker_isolation_check": check_to_dict(checker),
        "device_claim_open": opened, "device_claim_released": released,
        "device_sampling": sampling.to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-tree", required=True)
    result.add_argument("--binary", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="rvp-c2-8-c2-9-20260811")
    result.add_argument("--backend", default="ROCm0")
    result.add_argument("--suite-seed", type=int, default=4711)
    result.add_argument("--ops", default="MUL_MAT")
    result.add_argument("--params")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--probe-timeout-s", type=float, default=1200.0)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "hostile": payload["hostile_distribution_check"]["outcome"],
        "checker": payload["checker_isolation_check"]["outcome"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
