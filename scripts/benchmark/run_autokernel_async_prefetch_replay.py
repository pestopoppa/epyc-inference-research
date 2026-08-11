#!/usr/bin/env python3
"""Replay the known-real gfx90a Q8_0 async-prefetch win on frozen v9.

The experiment is a parameter comparison on one immutable production binary:
``GGML_CUDA_Q8_PREFETCH=1`` versus ``0``.  It keeps source, binary, linkage,
model, order, raw samples, the MI210 claim, and 250 ms device state in one
receipt.  This is a GPU-lane historical replay, not the v9 CPU calibration or a
production mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.async_prefetch_replay.v1"
PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
GFX90A_DURATION_FLOOR_NS = 250_090_903
GFX90A_DURATION_FLOOR_REF = (
    "rvp-t0-1-20260811T0906Z/receipt.json@sha256:"
    "07788e1d488ecec062e8133dd9e11d379e5075afbcc20f80b6da37e345533431"
    "#device_sampling.samples[1]")
SOURCE_ROOT = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_BINARY = SOURCE_ROOT / "build-hip/bin/llama-bench"
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf")
ARMS = ("anchor", "candidate")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_output(command: tuple[str, ...], *, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True).stdout


def balanced_orders(blocks: int, seed: int) -> list[tuple[str, str]]:
    if blocks < 2 or blocks % 2:
        raise ValueError("async-prefetch replay requires an even block count >= 2")
    orders = [("anchor", "candidate")] * (blocks // 2)
    orders += [("candidate", "anchor")] * (blocks // 2)
    random.Random(seed).shuffle(orders)
    return orders


def parse_row(path: Path, *, repetitions: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("{")]
    if len(rows) != 1:
        raise RuntimeError(f"{path} emitted {len(rows)} JSON rows, expected one")
    row = rows[0]
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise RuntimeError("async-prefetch arm did not execute on the MI210 ROCm backend")
    if row.get("n_prompt") != 0 or row.get("n_gen") != 128 or row.get("n_gpu_layers") != 99:
        raise RuntimeError("async-prefetch arm drifted from the predeclared tg128/full-offload cell")
    samples = row.get("samples_ns")
    if not isinstance(samples, list) or len(samples) != repetitions:
        raise RuntimeError("async-prefetch arm did not retain every raw repetition")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
           for value in samples):
        raise RuntimeError("async-prefetch arm emitted a non-positive timing sample")
    speed = row.get("avg_ts")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0:
        raise RuntimeError("async-prefetch arm emitted no positive avg_ts")
    return row


def run_arm(*, arm: str, block: int, binary: Path, model: Path,
            output_dir: Path, repetitions: int, timeout_s: float,
            warmup: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown async-prefetch arm: {arm}")
    prefix = "warmup" if warmup else f"block-{block:02d}"
    stdout_path = output_dir / f"{prefix}-{arm}.stdout.jsonl"
    stderr_path = output_dir / f"{prefix}-{arm}.stderr.txt"
    command = (
        str(binary), "-m", str(model), "-p", "0", "-n", "128",
        "-r", str(repetitions), "-ngl", "99", "-fa", "1", "-o", "jsonl")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    env["GGML_CUDA_Q8_PREFETCH"] = "1" if arm == "candidate" else "0"
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
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"async-prefetch {prefix} {arm} exited {returncode}: {tail!r}")
    row = parse_row(stdout_path, repetitions=repetitions)
    timing_window_ns = sum(float(value) for value in row["samples_ns"])
    if timing_window_ns < GFX90A_DURATION_FLOOR_NS:
        raise RuntimeError(
            f"async-prefetch {prefix} {arm} timing window {timing_window_ns:.0f} ns "
            f"is below the measured gfx90a floor {GFX90A_DURATION_FLOOR_NS} ns")
    return {
        "arm": arm, "block": block, "warmup": warmup,
        "environment": {"GGML_CUDA_Q8_PREFETCH": env["GGML_CUDA_Q8_PREFETCH"]},
        "command": list(command), "duration_s": time.monotonic() - started,
        "stdout": str(stdout_path), "stderr": str(stderr_path), "result": row,
        "duration_admission": {
            "outcome": "PASS", "timing_window_ns": timing_window_ns,
            "minimum_ns": GFX90A_DURATION_FLOOR_NS,
            "evidence_ref": GFX90A_DURATION_FLOOR_REF,
        },
    }


def summarize(block_rows: list[dict], *, contribution_floor: float) -> dict:
    by_block: dict[int, dict[str, float]] = {}
    for run in block_rows:
        by_block.setdefault(run["block"], {})[run["arm"]] = float(run["result"]["avg_ts"])
    if any(set(values) != set(ARMS) for values in by_block.values()):
        raise RuntimeError("a paired block is missing an async-prefetch arm")
    paired = []
    for block, values in sorted(by_block.items()):
        delta = values["candidate"] / values["anchor"] - 1.0
        paired.append({"block": block, **values, "relative_delta": delta})
    deltas = sorted(row["relative_delta"] for row in paired)
    mid = len(deltas) // 2
    median = (deltas[mid - 1] + deltas[mid]) / 2.0
    all_positive = all(value > 0 for value in deltas)
    reproduced = all_positive and median > contribution_floor
    return {
        "paired_blocks": paired,
        "minimum_relative_delta": min(deltas),
        "median_relative_delta": median,
        "contribution_floor": contribution_floor,
        "all_blocks_positive": all_positive,
        "verdict": "REPRODUCED_KNOWN_WIN" if reproduced else "NOT_REPRODUCED",
    }


def run(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    if source_root == Path("/mnt/raid0/llm/llama.cpp-experimental").resolve():
        raise RuntimeError("replay requires the frozen production-v9 source identity")
    source_commit = command_output(("git", "-C", str(source_root), "rev-parse", "HEAD")).strip()
    source_branch = command_output(
        ("git", "-C", str(source_root), "branch", "--show-current")).strip()
    if source_commit != PRODUCTION_COMMIT or source_branch != "production-consolidated-v9":
        raise RuntimeError(
            f"source identity drifted: {source_branch}@{source_commit}; required production v9")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-bench is not executable: {binary}")
    if not model.is_file():
        raise RuntimeError(f"model is missing: {model}")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    orders = balanced_orders(args.blocks, args.order_seed)
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AutoKernel async-prefetch replay evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    linkage = command_output(("ldd", str(binary)), env=env)
    (output_dir / "linkage.txt").write_text(linkage, encoding="utf-8")
    producer_paths = {
        "runner": Path(__file__).resolve(),
        "gpu_factorial_helpers": Path(
            sys.modules["scripts.benchmark.run_autokernel_gpu_factorial"].__file__).resolve(),
        "storage": Path(storage.__file__).resolve(),
        "device_sampler": Path(device_sampler.__file__).resolve(),
        "device_claim": Path(device_claim.__file__).resolve(),
    }
    declaration = {
        "schema": SCHEMA, "campaign_id": args.campaign_id,
        "declared_at": utc_now(), "source_root": str(source_root),
        "source_commit": source_commit, "source_branch": source_branch,
        "binary": str(binary), "binary_sha256": sha256_file(binary),
        "linkage_sha256": hashlib.sha256(linkage.encode()).hexdigest(),
        "producer_sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(producer_paths.items())
        },
        "model": str(model), "model_sha256": sha256_file(model),
        "cell": {"n_prompt": 0, "n_gen": 128, "n_gpu_layers": 99,
                 "flash_attention": True, "repetitions": args.repetitions},
        "blocks": args.blocks, "order_seed": args.order_seed,
        "orders": [list(order) for order in orders],
        "contribution_floor": args.contribution_floor,
        "gfx90a_duration_floor": {
            "minimum_ns": GFX90A_DURATION_FLOOR_NS,
            "evidence_ref": GFX90A_DURATION_FLOOR_REF,
        },
        "candidate_parameter": {"GGML_CUDA_Q8_PREFETCH": "1"},
        "anchor_parameter": {"GGML_CUDA_Q8_PREFETCH": "0"},
    }
    write_json_atomic(output_dir / "declaration.json", declaration)

    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AutoKernel known-real async-prefetch replay",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_async_prefetch_replay.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=(2 * args.blocks + 2) * args.arm_timeout_s + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling = None
    warmups = []
    runs = []
    started_at = utc_now()
    started = time.monotonic()
    try:
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        for arm in ARMS:
            warmups.append(run_arm(
                arm=arm, block=-1, binary=binary, model=model,
                output_dir=output_dir, repetitions=1,
                timeout_s=args.arm_timeout_s, warmup=True))
        for block, order in enumerate(orders):
            for arm in order:
                runs.append(run_arm(
                    arm=arm, block=block, binary=binary, model=model,
                    output_dir=output_dir, repetitions=args.repetitions,
                    timeout_s=args.arm_timeout_s))
        held = device_claim.check_device_claim_held(claim.receipt())
    finally:
        if sampler is not None:
            sampling = sampler.stop()
        released = claim.release().to_dict()
    if sampling is None:
        raise RuntimeError("async-prefetch replay completed without device sampling")
    result = summarize(runs, contribution_floor=args.contribution_floor)
    payload = {
        **declaration, "started_at": started_at, "ended_at": utc_now(),
        "duration_s": time.monotonic() - started, "warmups": warmups, "runs": runs,
        "result": result, "device_claim_open": opened,
        "device_claim_held_after_runs": {
            "outcome": held.outcome, "reasons": list(held.reasons)},
        "device_claim_released": released,
        "device_sampling": sampling.to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", default=str(SOURCE_ROOT))
    result.add_argument("--binary", default=str(DEFAULT_BINARY))
    result.add_argument("--model", default=str(DEFAULT_MODEL))
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="ak-gpu-prefetch-v9-20260811-r1")
    result.add_argument("--blocks", type=int, default=20)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--order-seed", type=int, default=2026081104)
    result.add_argument("--contribution-floor", type=float, default=0.02)
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--arm-timeout-s", type=float, default=600.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        print(f"ASYNC-PREFETCH REPLAY REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "verdict": payload["result"]["verdict"],
        "median_relative_delta": payload["result"]["median_relative_delta"],
        "minimum_relative_delta": payload["result"]["minimum_relative_delta"],
        "samples": payload["device_sampling"]["sample_count"],
        "max_gap_s": payload["device_sampling"]["max_gap_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
