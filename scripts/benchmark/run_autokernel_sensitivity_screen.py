#!/usr/bin/env python3
"""Run the reference-only AutoKernel multi-seed input-sensitivity screen."""
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
from scripts.kernel_rnd.autokernel.evaluator import sensitivity
from scripts.kernel_rnd.autokernel.execution import cpu_region_claim


SCHEMA = "epyc.autokernel.input_sensitivity_screen.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_seed(*, binary: Path, seed: int, output_dir: Path,
             timeout_s: float, ops: str | None) -> dict:
    command = [
        str(binary), "test", "-b", "CPU", "--suite-seed", str(seed),
        "--autokernel-value-transforms", "--output", "csv",
    ]
    if ops:
        command[2:2] = ["-o", ops]
    stdout_path = output_dir / f"seed-{seed}.csv"
    stderr_path = output_dir / f"seed-{seed}.stderr.txt"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=stdout_handle,
            stderr=stderr_handle, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_s)
        except BaseException:
            if process.poll() is None:
                terminate_owned(process)
            raise
    if returncode != 0:
        raise RuntimeError(f"sensitivity seed {seed} exited {returncode}")

    rows = []
    with stdout_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("sensitivity_receipt"):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"sensitivity seed {seed} emitted no AK_SENS_V1 receipts")
    return {
        "suite_seed": seed,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "duration_s": time.monotonic() - started,
        "case_count": len(rows),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"test-backend-ops is not executable: {binary}")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AutoKernel sensitivity evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    seeds = tuple(args.suite_seed)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct --suite-seed values are required")

    journal = cpu_region_claim.RegionClaimJournal(args.claim_journal)
    started_at = utc_now()
    started = time.monotonic()
    with cpu_region_claim.acquire_cpu_region_claim(
            "0-191", purpose="AutoKernel reference-only input-sensitivity screen",
            campaign_id=args.campaign_id, journal=journal,
            holder_label="run_autokernel_sensitivity_screen.py",
            timeout_s=args.claim_timeout_s,
            max_hold_s=len(seeds) * args.seed_timeout_s + 300.0) as claim:
        claim_open = claim.receipt().to_dict()
        seed_runs = [run_seed(
            binary=binary, seed=seed, output_dir=output_dir,
            timeout_s=args.seed_timeout_s, ops=args.ops) for seed in seeds]
        claim_held = claim.verify_held()
    all_rows = [row for seed_run in seed_runs for row in seed_run.pop("rows")]
    observations = sensitivity.observations_from_csv_rows(
        all_rows, expected_seeds=seeds)
    report = sensitivity.reduce_input_sensitivity(observations)
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "suite_seeds": list(seeds),
        "ops": args.ops,
        "reference_only": True,
        "producer": sensitivity.TRUSTED_PRODUCER,
        "seed_runs": seed_runs,
        "observation_count": len(observations),
        "report": report.to_dict(),
        "unscoreable_unit_count": len(report.unscoreable_units),
        "cpu_region_claim_open": claim_open,
        "cpu_region_claim_held_after_runs": {
            "outcome": claim_held.outcome, "reasons": list(claim_held.reasons)},
        "cpu_region_claim_released": claim.receipt().to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="rvp-c2-7-sensitivity-20260811")
    result.add_argument("--suite-seed", action="append", type=int,
                        default=[])
    result.add_argument("--ops", help="optional comma-separated op filter")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/cpu.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--seed-timeout-s", type=float, default=1200.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.suite_seed:
        args.suite_seed = [4711, 6841, 8117]
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "outcome": payload["report"]["check"]["outcome"],
        "unscoreable_units": payload["unscoreable_unit_count"],
        "observations": payload["observation_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
