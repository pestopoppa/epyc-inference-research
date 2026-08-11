#!/usr/bin/env python3
"""Run the governed INF-37 IQ2_XXS fancy-SIMD diagnostic A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import cpu_region_claim


SCHEMA = "epyc.inf37.iq2_fancy_simd_ab.v1"
CPU_LIST = "0-191"
EXPECTED_CELLS = ((1, 4096, 14336), (512, 4096, 14336))
SOURCE_FILE = "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def git_status(root: Path) -> str:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=root, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    return result.stdout.rstrip("\n")


def assert_sources(baseline: Path, candidate: Path, *, commit: str,
                   candidate_diff_sha256: str) -> dict:
    for root, label in ((baseline, "baseline"), (candidate, "candidate")):
        if Path(git_output(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise RuntimeError(f"{label} source root is not its git toplevel")
        if git_output(root, "rev-parse", "HEAD") != commit:
            raise RuntimeError(f"{label} source is not at {commit}")
    baseline_status = git_status(baseline)
    if baseline_status:
        raise RuntimeError(f"baseline source is dirty: {baseline_status}")
    candidate_status = git_status(candidate)
    if candidate_status != f" M {SOURCE_FILE}":
        raise RuntimeError(
            f"candidate must modify only {SOURCE_FILE}: {candidate_status!r}")
    diff = subprocess.run(
        ("git", "diff", "--binary", "--", SOURCE_FILE), cwd=candidate,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    observed_diff = sha256_bytes(diff)
    if observed_diff != candidate_diff_sha256:
        raise RuntimeError(
            f"candidate diff mismatch: expected {candidate_diff_sha256}, "
            f"observed {observed_diff}")
    return {
        "commit": commit,
        "baseline_root": str(baseline),
        "candidate_root": str(candidate),
        "candidate_diff_sha256": observed_diff,
        "candidate_source_sha256": sha256_file(candidate / SOURCE_FILE),
        "baseline_source_sha256": sha256_file(baseline / SOURCE_FILE),
    }


def linkage(binary: Path) -> dict:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"binary is not executable: {binary}")
    result = subprocess.run(
        ("ldd", str(binary)), check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = tuple(line.strip() for line in result.stdout.splitlines())
    cpu_rows = tuple(line for line in lines if "libggml-cpu" in line)
    if len(cpu_rows) != 1 or str(binary.parent) not in cpu_rows[0]:
        raise RuntimeError(
            f"{binary} does not resolve exactly one build-local libggml-cpu: {cpu_rows}")
    return {
        "path": str(binary),
        "sha256": sha256_file(binary),
        "ldd": list(lines),
        "ggml_cpu_row": cpu_rows[0],
    }


def balanced_orders(blocks: int) -> tuple[tuple[str, str], ...]:
    if blocks < 4 or blocks % 2:
        raise ValueError("blocks must be even and at least 4")
    return tuple(("baseline", "candidate") if i % 2 == 0
                 else ("candidate", "baseline") for i in range(blocks))


def parse_sql_rows(stdout: str) -> tuple[dict, ...]:
    database = sqlite3.connect(":memory:")
    try:
        database.executescript(stdout)
        cursor = database.execute(
            "SELECT op_params, supported, passed, time_us, n_runs "
            "FROM test_backend_ops ORDER BY op_params")
        raw = cursor.fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"invalid test-backend-ops SQL output: {exc}") from exc
    finally:
        database.close()
    rows = []
    for params, supported, passed, time_us, n_runs in raw:
        fields = dict(part.split("=", 1) for part in params.split(",")
                      if "=" in part and "[" not in part)
        if fields.get("type_a") != "iq2_xxs":
            raise RuntimeError(f"foreign quant row: {params}")
        row = {
            "op_params": params,
            "n": int(fields["n"]),
            "m": int(fields["m"]),
            "k": int(fields["k"]),
            "supported": bool(supported),
            "passed": bool(passed),
            "time_us": float(time_us),
            "n_runs": int(n_runs),
        }
        if not row["supported"] or not row["passed"] or row["time_us"] <= 0:
            raise RuntimeError(f"invalid performance row: {row}")
        rows.append(row)
    observed = tuple(sorted((row["n"], row["m"], row["k"]) for row in rows))
    if observed != EXPECTED_CELLS:
        raise RuntimeError(
            f"expected exact cells {EXPECTED_CELLS}, observed {observed}")
    return tuple(sorted(rows, key=lambda row: row["n"]))


def summarize(invocations: list[dict], blocks: int) -> dict:
    cells = []
    for n, m, k in EXPECTED_CELLS:
        paired = []
        for block in range(blocks):
            values = {}
            for invocation in invocations:
                if invocation["block"] != block:
                    continue
                row = next(item for item in invocation["rows"] if item["n"] == n)
                values[invocation["arm"]] = row["time_us"]
            if set(values) != {"baseline", "candidate"}:
                raise RuntimeError(f"block {block} is missing an arm")
            paired.append({
                "block": block,
                "baseline_time_us": values["baseline"],
                "candidate_time_us": values["candidate"],
                "candidate_speedup_fraction": (
                    values["baseline"] / values["candidate"] - 1.0),
            })
        deltas = sorted(row["candidate_speedup_fraction"] for row in paired)
        median = (deltas[len(deltas) // 2 - 1] + deltas[len(deltas) // 2]) / 2
        cells.append({
            "n": n, "m": m, "k": k, "paired_blocks": paired,
            "median_candidate_speedup_fraction": median,
            "all_candidate_faster": all(value > 0 for value in deltas),
            "min_candidate_speedup_fraction": min(deltas),
            "max_candidate_speedup_fraction": max(deltas),
        })
    return {
        "metric": "time_us",
        "direction": "lower_is_better",
        "cells": cells,
        "screening_only": True,
        "promotion_authority": False,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    output = Path(storage.assert_not_scratch(
        args.output_dir, what="INF-37 IQ2 fancy-SIMD evidence directory"))
    output.mkdir(parents=True, exist_ok=False)
    baseline_source = Path(args.baseline_source).resolve()
    candidate_source = Path(args.candidate_source).resolve()
    binaries = {
        "baseline": Path(args.baseline_binary).resolve(),
        "candidate": Path(args.candidate_binary).resolve(),
    }
    identity = assert_sources(
        baseline_source, candidate_source, commit=args.source_commit,
        candidate_diff_sha256=args.candidate_diff_sha256)
    binary_identity = {arm: linkage(binary) for arm, binary in binaries.items()}
    runner = Path(__file__).resolve()
    command_tail = (
        "perf", "-b", "CPU", "-o", "MUL_MAT", "-p",
        r"type_a=iq2_xxs.*n=(1|512),", "--output", "sql")
    journal = cpu_region_claim.RegionClaimJournal(output / "region_claim.jsonl")
    claim = cpu_region_claim.acquire_cpu_region_claim(
        CPU_LIST, purpose="INF-37 IQ2 fancy-SIMD diagnostic A/B",
        campaign_id=args.campaign_id, journal=journal, role="autokernel",
        holder_label="run_inf37_iq2_fancy_simd_ab.py", timeout_s=args.claim_timeout_s,
        max_hold_s=args.timeout_s * args.blocks * 2 + 300.0)
    opened = claim.receipt().to_dict()
    invocations = []
    captured_error = None
    try:
        env = os.environ.copy()
        env["GGML_IQK"] = "1"
        for block, order in enumerate(balanced_orders(args.blocks)):
            for position, arm in enumerate(order):
                held = claim.verify_held()
                if held.outcome != cpu_region_claim.PASS:
                    raise RuntimeError(
                        f"CPU claim failed before block {block}/{arm}: {held}")
                command = (
                    "taskset", "-c", CPU_LIST, "numactl", "--interleave=all",
                    str(binaries[arm]), *command_tail)
                result = subprocess.run(
                    command, env=env, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=args.timeout_s)
                stdout_path = output / f"block-{block:02d}-{position}-{arm}.stdout.sql"
                stderr_path = output / f"block-{block:02d}-{position}-{arm}.stderr.txt"
                stdout_path.write_text(result.stdout, encoding="utf-8")
                stderr_path.write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(
                        f"block {block}/{arm} exited {result.returncode}: "
                        f"{result.stderr[-2000:]!r}")
                rows = parse_sql_rows(result.stdout)
                if "[iqk] ACTIVE" not in result.stderr:
                    raise RuntimeError(f"block {block}/{arm} did not activate IQK")
                invocations.append({
                    "block": block, "position": position, "arm": arm,
                    "command": list(command), "rows": list(rows),
                    "stdout_path": str(stdout_path),
                    "stdout_sha256": sha256_file(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stderr_sha256": sha256_file(stderr_path),
                })
    except BaseException as exc:
        captured_error = exc
    finally:
        released = claim.release().to_dict()
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "created_at": utc_now(),
        "status": "failed" if captured_error else "complete",
        "evidence_grade": "diagnostic_screening",
        "package_energy_required": False,
        "package_energy_reason": (
            "screening-only op microbenchmark; formal model controls retain the RAPL gate"),
        "runner": str(runner),
        "runner_sha256": sha256_file(runner),
        "source_identity": identity,
        "binary_identity": binary_identity,
        "cpu_list": CPU_LIST,
        "blocks": args.blocks,
        "orders": [list(row) for row in balanced_orders(args.blocks)],
        "device_claim_open": opened,
        "device_claim_released": released,
        "invocations": invocations,
        "summary": summarize(invocations, args.blocks) if captured_error is None else None,
        "error": None if captured_error is None else {
            "type": type(captured_error).__name__, "message": str(captured_error)},
    }
    write_json_atomic(output / "receipt.json", payload)
    if captured_error is not None:
        raise RuntimeError(
            f"INF-37 A/B failed; durable receipt: {output / 'receipt.json'}") from captured_error
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-source", required=True)
    result.add_argument("--candidate-source", required=True)
    result.add_argument("--baseline-binary", required=True)
    result.add_argument("--candidate-binary", required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--candidate-diff-sha256", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="inf37-iq2-fancy-simd-20260811")
    result.add_argument("--blocks", type=int, default=10)
    result.add_argument("--timeout-s", type=float, default=120.0)
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        output = Path(args.output_dir)
        receipt = output / "receipt.json"
        if output.is_dir() and not receipt.exists():
            write_json_atomic(receipt, {
                "schema": SCHEMA,
                "campaign_id": args.campaign_id,
                "created_at": utc_now(),
                "status": "failed_preclaim",
                "runner": str(Path(__file__).resolve()),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
        print(f"INF-37 A/B REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "cells": payload["summary"]["cells"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
