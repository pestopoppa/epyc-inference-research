#!/usr/bin/env python3
"""Run AK-LN-2 and AK-X-5a across every registered CPU split depth."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
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
from scripts.kernel_rnd.autokernel import lanes, storage
from scripts.kernel_rnd.autokernel.evaluator import recipes
from scripts.kernel_rnd.autokernel.execution import cpu_region_claim, microbench


SCHEMA = "epyc.autokernel.cpu_lane_calibration.v1"
DEPTHS = (1, 4, 8, 16, 32, 48)
PREDICTION = {
    "registered_before_measurement": True,
    "bandwidth_bound": "rank fidelity decreases as concurrent split depth increases",
    "instruction_level": "rank fidelity remains at or above 0.8 across split depths",
    "fixed_candidate_set": ["anchor", "iqk-off", "flash-attention-off"],
}
CANDIDATES = (
    ("anchor", "on", "1", "mixed-anchor"),
    ("iqk-off", "on", "0", "instruction-level"),
    ("flash-attention-off", "off", "1", "bandwidth-bound"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check_to_dict(check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


@dataclass
class RunningLane:
    lane: lanes.LaneSpec
    candidate_id: str
    process: subprocess.Popen
    stdout_handle: object
    stderr_handle: object
    stdout_path: Path
    stderr_path: Path
    open_state: microbench.HostState
    started_mono: float
    loaded_state: microbench.HostState | None = None
    close_state: microbench.HostState | None = None
    returncode: int | None = None


def split_lanes(depth: int) -> tuple[lanes.LaneSpec, ...]:
    if depth == 1:
        return (lanes.default_lane_registry()["cpu-full"],)
    prefix = f"cpu-{depth}x{192 // depth}t-"
    selected = tuple(sorted(
        (lane for lane in lanes.historical_split_lane_registry().values()
         if lane.lane_id.startswith(prefix)), key=lambda lane: lane.lane_id))
    if len(selected) != depth:
        raise RuntimeError(f"lane registry supplied {len(selected)} lanes for depth {depth}")
    lanes.validate_concurrent_cpu_lanes(selected)
    return selected


def command_for(*, binary: Path, model: Path, lane: lanes.LaneSpec,
                candidate: tuple, args: argparse.Namespace) -> tuple[str, ...]:
    candidate_id, flash_attention, _iqk, _change_class = candidate
    if lane.verification_lane:
        prefix = list(recipes.CANONICAL_PREFIX)
    else:
        if len(lane.membind_nodes) != 1:
            raise RuntimeError(f"partition lane {lane.lane_id} needs one membind node")
        prefix = [
            "taskset", "-c", lane.cpu_list, "numactl",
            f"--membind={lane.membind_nodes[0]}",
        ]
    threads = len(cpu_region_claim.parse_cpu_list(lane.cpu_list))
    command = prefix + [
        str(binary), "-t", str(threads), "-fa", flash_attention, "-mmp", "0",
        "-m", str(model), "-p", str(args.prompt_tokens), "-n", "0",
        "-r", str(args.repetitions), "--autokernel-harden", str(args.suite_seed),
        "-o", "jsonl",
    ]
    if candidate_id not in {item[0] for item in CANDIDATES}:
        raise RuntimeError(f"unknown fixed candidate {candidate_id!r}")
    return tuple(command)


def arm_environment(binary: Path, candidate: tuple) -> dict[str, str]:
    env = {key: os.environ[key] for key in microbench.DEFAULT_BASE_ENV_KEYS if key in os.environ}
    env.update(recipes.CANONICAL_OMP_ENV)
    env["GGML_IQK"] = candidate[2]
    env["LD_LIBRARY_PATH"] = str(binary.parent)
    return env


def spawn_lane(*, binary: Path, model: Path, lane: lanes.LaneSpec, candidate: tuple,
               args: argparse.Namespace, arm_dir: Path) -> RunningLane:
    stdout_path = arm_dir / f"{lane.lane_id}.stdout.jsonl"
    stderr_path = arm_dir / f"{lane.lane_id}.stderr.txt"
    stdout_handle = stdout_path.open("wb")
    stderr_handle = stderr_path.open("wb")
    command = command_for(
        binary=binary, model=model, lane=lane, candidate=candidate, args=args)
    open_state = microbench.read_host_state(cpu_list=lane.cpu_list)
    try:
        process = subprocess.Popen(
            command, env=arm_environment(binary, candidate), stdin=subprocess.DEVNULL,
            stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return RunningLane(
        lane=lane, candidate_id=candidate[0], process=process,
        stdout_handle=stdout_handle, stderr_handle=stderr_handle,
        stdout_path=stdout_path, stderr_path=stderr_path,
        open_state=open_state, started_mono=time.monotonic())


def close_running(running: RunningLane) -> None:
    running.stdout_handle.close()
    running.stderr_handle.close()


def collect_wave(*, binary: Path, model: Path, depth: int, candidate: tuple,
                 args: argparse.Namespace, output_dir: Path) -> dict:
    selected_lanes = split_lanes(depth)
    arm_dir = output_dir / f"depth-{depth:02d}" / candidate[0]
    arm_dir.mkdir(parents=True, exist_ok=False)
    running: list[RunningLane] = []
    deadline = time.monotonic() + args.wave_timeout_s
    try:
        for lane in selected_lanes:
            running.append(spawn_lane(
                binary=binary, model=model, lane=lane, candidate=candidate,
                args=args, arm_dir=arm_dir))
        time.sleep(args.loaded_sample_delay_s)
        for item in running:
            item.loaded_state = microbench.read_host_state(cpu_list=item.lane.cpu_list)
        pending = {item.process.pid: item for item in running}
        while pending:
            for pid, item in tuple(pending.items()):
                returncode = item.process.poll()
                if returncode is None:
                    continue
                item.returncode = returncode
                item.close_state = microbench.read_host_state(cpu_list=item.lane.cpu_list)
                close_running(item)
                pending.pop(pid)
            if pending and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"depth {depth} candidate {candidate[0]} exceeded wave timeout")
            if pending:
                time.sleep(0.020)
    except BaseException:
        for item in running:
            if item.process.poll() is None:
                terminate_owned(item.process)
            if not item.stdout_handle.closed:
                close_running(item)
        raise

    results = []
    for item in running:
        stderr_tail = item.stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        if item.returncode != 0:
            raise RuntimeError(
                f"{item.lane.lane_id}/{candidate[0]} exited {item.returncode}: "
                f"{stderr_tail!r}")
        rows = [json.loads(line) for line in
                item.stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(
                f"{item.lane.lane_id}/{candidate[0]} emitted {len(rows)} rows")
        row = rows[0]
        if row.get("backends") != "CPU" or len(row.get("samples_ns", ())) != args.repetitions:
            raise RuntimeError(f"{item.lane.lane_id}/{candidate[0]} has incomplete samples")
        for field in (
                "autokernel_hardened", "autokernel_output_invariant",
                "autokernel_hybrid_ab_complete", "autokernel_thread_set_stable",
                "autokernel_escape_checks_complete"):
            if row.get(field) is not True:
                raise RuntimeError(
                    f"{item.lane.lane_id}/{candidate[0]} failed hardening field {field}")
        power_check, power = microbench.derive_package_power_attestation(
            item.open_state, item.close_state)
        results.append({
            "lane_id": item.lane.lane_id,
            "cpu_list": item.lane.cpu_list,
            "membind_nodes": list(item.lane.membind_nodes),
            "avg_ts": row["avg_ts"],
            "result": row,
            "stdout": str(item.stdout_path),
            "stderr": str(item.stderr_path),
            "stderr_tail": stderr_tail,
            "host_state_open": item.open_state.to_dict(),
            "host_state_loaded": item.loaded_state.to_dict(),
            "host_state_close": item.close_state.to_dict(),
            "package_power_check": check_to_dict(power_check),
            "package_power": power.to_dict() if power is not None else None,
        })
    values = [float(item["avg_ts"]) for item in results]
    median_value = statistics.median(values)
    max_position_deviation = max(abs(value / median_value - 1.0) for value in values)
    loaded_khz = [float(item["host_state_loaded"]["median_khz"])
                  for item in results if item["host_state_loaded"]["median_khz"] is not None]
    return {
        "depth": depth,
        "candidate_id": candidate[0],
        "change_class": candidate[3],
        "lane_count": len(results),
        "lanes": results,
        "aggregate_ts": sum(values),
        "median_lane_ts": median_value,
        "max_lane_position_deviation": max_position_deviation,
        "median_loaded_khz": statistics.median(loaded_khz) if loaded_khz else None,
        "all_power_windows_pass": all(
            item["package_power_check"]["outcome"] == "PASS" for item in results),
    }


def summarize(waves: list[dict], args: argparse.Namespace) -> list[dict]:
    full = {wave["candidate_id"]: wave for wave in waves if wave["depth"] == 1}
    full_order = tuple(sorted(full, key=lambda item: (-full[item]["aggregate_ts"], item)))
    full_frequency = statistics.median(
        wave["median_loaded_khz"] for wave in full.values()
        if wave["median_loaded_khz"] is not None)
    summaries = []
    for depth in DEPTHS:
        current = {wave["candidate_id"]: wave for wave in waves if wave["depth"] == depth}
        order = tuple(sorted(current, key=lambda item: (-current[item]["aggregate_ts"], item)))
        fidelity = lanes.spearman_rank_fidelity(order, full_order)
        anchor = current["anchor"]
        frequency_ratio = (anchor["median_loaded_khz"] / full_frequency
                           if anchor["median_loaded_khz"] is not None else None)
        power_frequency_pass = (
            anchor["all_power_windows_pass"]
            and anchor["max_lane_position_deviation"] <= args.max_position_deviation
            and frequency_ratio is not None
            and frequency_ratio >= args.min_loaded_frequency_ratio)
        summaries.append({
            "depth": depth,
            "screening_order": list(order),
            "verification_order": list(full_order),
            "spearman_rank_fidelity": fidelity,
            "rank_fidelity_pass": fidelity >= args.min_rank_fidelity,
            "anchor_max_lane_position_deviation": anchor["max_lane_position_deviation"],
            "anchor_loaded_frequency_ratio_to_full": frequency_ratio,
            "anchor_all_power_windows_pass": anchor["all_power_windows_pass"],
            "power_frequency_acceptance_pass": power_frequency_pass,
        })
    return summaries


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-bench is not executable: {binary}")
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AK-LN-2/AK-X-5a evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    schedule = [(depth, candidate) for depth in DEPTHS for candidate in CANDIDATES]
    random.Random(args.suite_seed).shuffle(schedule)
    journal = cpu_region_claim.RegionClaimJournal(args.claim_journal)
    started_at = utc_now()
    started = time.monotonic()
    with cpu_region_claim.acquire_cpu_region_claim(
            "0-191", purpose="AK-LN-2 and AK-X-5a CPU lane calibration",
            campaign_id=args.campaign_id, journal=journal,
            holder_label="run_autokernel_cpu_lane_calibration.py",
            timeout_s=args.claim_timeout_s,
            max_hold_s=len(schedule) * args.wave_timeout_s + 600.0) as claim:
        claim_open = claim.receipt().to_dict()
        waves = [collect_wave(
            binary=binary, model=model, depth=depth, candidate=candidate,
            args=args, output_dir=output_dir) for depth, candidate in schedule]
        claim_held = check_to_dict(claim.verify_held())
    summaries = summarize(waves, args)
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "prediction": PREDICTION,
        "thresholds": {
            "min_rank_fidelity": args.min_rank_fidelity,
            "max_position_deviation": args.max_position_deviation,
            "min_loaded_frequency_ratio": args.min_loaded_frequency_ratio,
        },
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "prompt_tokens": args.prompt_tokens,
        "repetitions": args.repetitions,
        "suite_seed": args.suite_seed,
        "randomized_schedule": [
            {"depth": depth, "candidate_id": candidate[0]}
            for depth, candidate in schedule],
        "waves": waves,
        "depth_summaries": summaries,
        "cpu_region_claim_open": claim_open,
        "cpu_region_claim_held_after_waves": claim_held,
        "cpu_region_claim_released": claim.receipt().to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="ak-ln-2-x5a-20260811")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/cpu.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--wave-timeout-s", type=float, default=900.0)
    result.add_argument("--loaded-sample-delay-s", type=float, default=0.5)
    result.add_argument("--prompt-tokens", type=int, default=512)
    result.add_argument("--repetitions", type=int, default=10)
    result.add_argument("--suite-seed", type=int, default=4711)
    result.add_argument("--min-rank-fidelity", type=float, default=0.8)
    result.add_argument("--max-position-deviation", type=float, default=0.10)
    result.add_argument("--min-loaded-frequency-ratio", type=float, default=0.80)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        print(f"CPU LANE CALIBRATION REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "depths": payload["depth_summaries"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
