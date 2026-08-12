#!/usr/bin/env python3
"""Run AK-BH-1 under a device claim and retain paired baseline evidence."""
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


SCHEMA = "epyc.ak_bh_1_gemm_baseline_compare.v1"


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


def terminate_owned(process: subprocess.Popen, *, grace_s: float = 10.0) -> int:
    process.terminate()
    try:
        return process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=grace_s)


def parse_rows(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or rows[0].get("schema") != "epyc.rocm.gemm_baseline.meta.v1":
        raise RuntimeError("baseline comparator did not emit its metadata row")
    if not str(rows[0].get("arch", "")).startswith("gfx90a"):
        raise RuntimeError(f"baseline comparator ran on {rows[0].get('arch')!r}, not gfx90a")
    results = rows[1:]
    expected = int(rows[0].get("shape_count", 0)) * 2
    if len(results) != expected:
        raise RuntimeError(f"baseline comparator emitted {len(results)} result rows, expected {expected}")
    if any(row.get("schema") != "epyc.rocm.gemm_baseline.v1" for row in results):
        raise RuntimeError("baseline comparator emitted an unknown result schema")
    grouped: dict[tuple[int, int, int], dict[str, dict]] = {}
    for row in results:
        key = (int(row["m"]), int(row["n"]), int(row["k"]))
        grouped.setdefault(key, {})[str(row["library"])] = row
    if any(set(pair) != {"rocblas", "hipblaslt"} for pair in grouped.values()):
        raise RuntimeError("every shape must contain one rocBLAS and one hipBLASLt row")
    return rows[0], results


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"comparator binary is not executable: {binary}")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AK-BH-1 evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "baseline.stdout.jsonl"
    stderr_path = output_dir / "baseline.stderr.txt"
    command = (str(binary), str(args.repetitions), str(args.warmup))

    journal = device_claim.ClaimJournal(args.claim_journal)
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AK-BH-1 rocBLAS versus hipBLASLt baseline comparison",
        campaign_id=args.campaign_id, journal=journal,
        holder_label="run_rocm_gemm_baseline_compare.py",
        timeout_s=args.claim_timeout_s, max_hold_s=args.timeout_s + 120.0)
    opened_receipt = claim.receipt().to_dict()
    sampling_session = None
    sampling_receipt = None
    returncode = None
    started_at = utc_now()
    started_mono = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout_handle,
                stderr=stderr_handle, start_new_session=True)
            try:
                sampling_session = device_sampler.RocmSmiSampler(
                    device_index=0, interval_s=0.250).start()
                returncode = process.wait(timeout=args.timeout_s)
            except BaseException:
                if process.poll() is None:
                    terminate_owned(process)
                raise
            finally:
                if sampling_session is not None:
                    sampling_receipt = sampling_session.stop()
    finally:
        released_receipt = claim.release().to_dict()

    stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if returncode != 0:
        raise RuntimeError(f"baseline comparator exited {returncode}: {stderr_tail!r}")
    if sampling_receipt is None:
        raise RuntimeError("baseline comparator completed without a device sampling receipt")
    metadata, results = parse_rows(stdout_path)
    comparisons = []
    by_shape: dict[tuple[int, int, int], dict[str, dict]] = {}
    for row in results:
        key = (int(row["m"]), int(row["n"]), int(row["k"]))
        by_shape.setdefault(key, {})[str(row["library"])] = row
    for (m, n, k), pair in sorted(by_shape.items()):
        rocblas_tflops = float(pair["rocblas"]["tflops"])
        hipblaslt_tflops = float(pair["hipblaslt"]["tflops"])
        comparisons.append({
            "m": m, "n": n, "k": k,
            "rocblas_tflops": rocblas_tflops,
            "hipblaslt_tflops": hipblaslt_tflops,
            "hipblaslt_over_rocblas": hipblaslt_tflops / rocblas_tflops,
        })
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "process_duration_s": time.monotonic() - started_mono,
        "comparator_binary": str(binary),
        "comparator_binary_sha256": sha256_file(binary),
        "comparator_source": str(Path(args.source).resolve()),
        "comparator_source_sha256": sha256_file(Path(args.source).resolve()),
        "command": list(command),
        "metadata": metadata,
        "raw_results": results,
        "comparisons": comparisons,
        "device_claim_open": opened_receipt,
        "device_claim_released": released_receipt,
        "device_sampling": sampling_receipt.to_dict(),
        "stderr_tail": stderr_tail,
    }
    payload = beliefs.attach_beliefs(payload, producer_path=Path(__file__).resolve())
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="ak-bh-1-20260811")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--timeout-s", type=float, default=300.0)
    result.add_argument("--repetitions", type=int, default=30)
    result.add_argument("--warmup", type=int, default=10)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        print(f"AK-BH-1 REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ratios = [row["hipblaslt_over_rocblas"] for row in payload["comparisons"]]
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "shape_count": len(ratios),
        "hipblaslt_wins": sum(ratio > 1.0 for ratio in ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "samples": payload["device_sampling"]["sample_count"],
        "max_gap_s": payload["device_sampling"]["max_gap_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
