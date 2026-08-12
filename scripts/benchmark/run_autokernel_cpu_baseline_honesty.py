#!/usr/bin/env python3
"""Run the AK-BH-3 CPU flash-attention baseline-honesty comparison."""
from __future__ import annotations

import argparse
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
from scripts.kernel_rnd.autokernel.evaluator import recipes
from scripts.kernel_rnd.autokernel.execution import cpu_region_claim, microbench


SCHEMA = "epyc.ak_bh_3_cpu_baseline_honesty.v1"
ARMS = (
    ("implicit-auto", None, -1),
    ("explicit-off", "off", 0),
    ("explicit-on", "on", 1),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check_to_dict(check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def command_for(*, binary: Path, model: Path, arm: tuple, args: argparse.Namespace) -> tuple:
    arm_id, flash_attention, _expected = arm
    command = list(recipes.CANONICAL_PREFIX)
    command += [
        str(binary), "-t", "96", "-mmp", "0", "-m", str(model),
        "-p", str(args.prompt_tokens), "-n", "0", "-r", str(args.repetitions),
        "--autokernel-harden", str(args.suite_seed), "-o", "jsonl",
    ]
    if flash_attention is not None:
        command[command.index("-mmp"):command.index("-mmp")] = ["-fa", flash_attention]
    return tuple(command)


def run_arm(*, binary: Path, model: Path, arm: tuple, args: argparse.Namespace,
            output_dir: Path) -> dict:
    arm_id, flash_attention, expected_flash_attention = arm
    command = command_for(binary=binary, model=model, arm=arm, args=args)
    stdout_path = output_dir / f"{arm_id}.stdout.jsonl"
    stderr_path = output_dir / f"{arm_id}.stderr.txt"
    env = {key: os.environ[key] for key in microbench.DEFAULT_BASE_ENV_KEYS if key in os.environ}
    env.update(recipes.CANONICAL_OMP_ENV)
    env["LD_LIBRARY_PATH"] = str(binary.parent)
    open_state = microbench.read_host_state(cpu_list=args.cpu_list)
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
    close_state = microbench.read_host_state(cpu_list=args.cpu_list)
    duration_s = time.monotonic() - started
    stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if returncode != 0:
        raise RuntimeError(f"AK-BH-3 arm {arm_id} exited {returncode}: {stderr_tail!r}")
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"AK-BH-3 arm {arm_id} emitted {len(lines)} rows, expected 1")
    row = json.loads(lines[0])
    if row.get("backends") != "CPU":
        raise RuntimeError(f"AK-BH-3 arm {arm_id} did not use the CPU backend")
    if row.get("flash_attn") != expected_flash_attention:
        raise RuntimeError(
            f"AK-BH-3 arm {arm_id} recorded flash_attn={row.get('flash_attn')!r}, "
            f"expected {expected_flash_attention}")
    required_true = (
        "autokernel_hardened", "autokernel_output_invariant",
        "autokernel_hybrid_ab_complete", "autokernel_thread_set_stable",
        "autokernel_escape_checks_complete",
    )
    if any(row.get(field) is not True for field in required_true):
        raise RuntimeError(f"AK-BH-3 arm {arm_id} has an incomplete hardening receipt")
    samples = row.get("samples_ns")
    if not isinstance(samples, list) or len(samples) != args.repetitions:
        raise RuntimeError(f"AK-BH-3 arm {arm_id} did not retain every repetition")
    power_check, power = microbench.derive_package_power_attestation(open_state, close_state)
    return {
        "arm_id": arm_id,
        "flash_attention_argument": flash_attention,
        "flash_attention_recorded": row["flash_attn"],
        "command": list(command),
        "environment": {key: env[key] for key in sorted(recipes.CANONICAL_OMP_ENV)},
        "duration_s": duration_s,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stderr_tail": stderr_tail,
        "result": row,
        "host_state_open": open_state.to_dict(),
        "host_state_close": close_state.to_dict(),
        "package_power_check": check_to_dict(power_check),
        "package_power": power.to_dict() if power is not None else None,
    }


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-bench is not executable: {binary}")
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    if args.cpu_list != recipes.CANONICAL_PREFIX[2]:
        raise RuntimeError(
            f"AK-BH-3 is the canonical full-host arm and requires CPU list "
            f"{recipes.CANONICAL_PREFIX[2]}, got {args.cpu_list}")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AK-BH-3 evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    journal = cpu_region_claim.RegionClaimJournal(args.claim_journal)
    arms = list(ARMS)
    random.Random(args.suite_seed).shuffle(arms)
    started_at = utc_now()
    started = time.monotonic()
    with cpu_region_claim.acquire_cpu_region_claim(
            args.cpu_list, purpose="AK-BH-3 CPU baseline-honesty arm",
            campaign_id=args.campaign_id, journal=journal,
            holder_label="run_autokernel_cpu_baseline_honesty.py",
            timeout_s=args.claim_timeout_s,
            max_hold_s=len(arms) * args.arm_timeout_s + 300.0) as claim:
        claim_open = claim.receipt().to_dict()
        results = [run_arm(
            binary=binary, model=model, arm=arm, args=args, output_dir=output_dir)
            for arm in arms]
        claim_held = check_to_dict(claim.verify_held())
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "cpu_list": args.cpu_list,
        "prompt_tokens": args.prompt_tokens,
        "repetitions": args.repetitions,
        "suite_seed": args.suite_seed,
        "randomized_arm_order": [arm[0] for arm in arms],
        "arms": results,
        "cpu_region_claim_open": claim_open,
        "cpu_region_claim_held_after_arms": claim_held,
        "cpu_region_claim_released": claim.receipt().to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="ak-bh-3-20260811")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/cpu.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--arm-timeout-s", type=float, default=600.0)
    result.add_argument("--cpu-list", default="0-95")
    result.add_argument("--prompt-tokens", type=int, default=512)
    result.add_argument("--repetitions", type=int, default=30)
    result.add_argument("--suite-seed", type=int, default=4711)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        print(f"AK-BH-3 REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ranked = sorted(
        ((arm["result"]["avg_ts"], arm["arm_id"]) for arm in payload["arms"]),
        reverse=True)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "winner": ranked[0][1],
        "winner_ts": ranked[0][0],
        "slowest": ranked[-1][1],
        "slowest_ts": ranked[-1][0],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
