#!/usr/bin/env python3
"""Build and replay one sealed historical AutoKernel expert task on ROCm."""
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

from scripts.benchmark.run_autokernel_gpu_factorial import (
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.evaluator import historical_tasks
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.historical_replay.v1"
ARM_ORDER = (
    ("parent", "expert_off", "expert_on"),
    ("expert_on", "parent", "expert_off"),
    ("expert_off", "expert_on", "parent"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_owned(command: list[str], *, cwd: Path, env: dict[str, str],
              stdout_path: Path, stderr_path: Path, timeout_s: float) -> int:
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        try:
            return process.wait(timeout=timeout_s)
        except BaseException:
            if process.poll() is None:
                terminate_owned(process)
            raise


def create_worktree(*, repo: Path, path: Path, commit: str) -> None:
    if path.exists():
        raise RuntimeError(f"historical worktree path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ("git", "worktree", "add", "--detach", str(path), commit), cwd=repo,
        check=True, stdin=subprocess.DEVNULL)


def build_arm(*, arm: str, tree: Path, output_dir: Path,
              jobs: int, timeout_s: float) -> dict:
    build = tree / "build-autokernel-historical"
    configure = [
        "cmake", "-S", str(tree), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON",
        "-DAMDGPU_TARGETS=gfx90a", "-DGGML_CCACHE=OFF",
    ]
    env = os.environ.copy()
    configure_rc = run_owned(
        configure, cwd=tree, env=env,
        stdout_path=output_dir / f"{arm}.configure.stdout.txt",
        stderr_path=output_dir / f"{arm}.configure.stderr.txt", timeout_s=timeout_s)
    if configure_rc != 0:
        raise RuntimeError(f"{arm} historical configure exited {configure_rc}")
    build_command = [
        "cmake", "--build", str(build), "--target", "llama-batched-bench",
        f"-j{jobs}",
    ]
    build_rc = run_owned(
        build_command, cwd=tree, env=env,
        stdout_path=output_dir / f"{arm}.build.stdout.txt",
        stderr_path=output_dir / f"{arm}.build.stderr.txt", timeout_s=timeout_s)
    if build_rc != 0:
        raise RuntimeError(f"{arm} historical build exited {build_rc}")
    binary = build / "bin" / "llama-batched-bench"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"{arm} historical binary is not executable: {binary}")
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=tree, text=True).strip()
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=tree, text=True)
    if status:
        raise RuntimeError(f"{arm} historical source became dirty during build")
    return {
        "arm": arm, "source_tree": str(tree), "source_commit": commit,
        "build_root": str(build), "binary": str(binary),
        "binary_sha256": sha256_file(binary), "configure_command": configure,
        "build_command": build_command,
    }


def benchmark_argv(descriptor: historical_tasks.HistoricalTaskDescriptor,
                   binary: Path) -> list[str]:
    return [
        str(binary) if item == "llama-batched-bench" else
        descriptor.model_path if item == "{model}" else item
        for item in descriptor.benchmark_argv
    ]


def parse_benchmark(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    if len(rows) != 1:
        raise RuntimeError(f"historical benchmark emitted {len(rows)} JSONL rows, expected one")
    row = rows[0]
    expected = {
        "n_kv_max": 8192, "n_batch": 2048, "n_ubatch": 512,
        "flash_attn": 0, "is_pp_shared": 0, "n_gpu_layers": 99,
        "pp": 128, "tg": 128, "pl": 32, "n_kv": 8192,
    }
    drift = {name: (row.get(name), value) for name, value in expected.items()
             if row.get(name) != value}
    if drift:
        raise RuntimeError(f"historical benchmark surface drifted: {drift}")
    speed = row.get("speed_tg")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0:
        raise RuntimeError("historical benchmark speed_tg is not positive")
    return row


def run_benchmark(*, arm: str, block: int, build: dict, descriptor,
                  output_dir: Path, timeout_s: float) -> dict:
    binary = Path(build["binary"])
    command = benchmark_argv(descriptor, binary)
    stdout_path = output_dir / f"block-{block:02d}-{arm}.stdout.jsonl"
    stderr_path = output_dir / f"block-{block:02d}-{arm}.stderr.txt"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    if arm == "expert_off":
        env["GGML_CUDA_GDN_STATE_BF16"] = "0"
    elif arm == "expert_on":
        env["GGML_CUDA_GDN_STATE_BF16"] = "1"
    started = time.monotonic()
    returncode = run_owned(
        command, cwd=Path(build["source_tree"]), env=env,
        stdout_path=stdout_path, stderr_path=stderr_path, timeout_s=timeout_s)
    if returncode != 0:
        raise RuntimeError(f"historical {arm} block {block} exited {returncode}")
    row = parse_benchmark(stdout_path)
    return {
        "block": block, "arm": arm, "command": command,
        "environment": {
            "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
            "GGML_CUDA_GDN_STATE_BF16": env.get("GGML_CUDA_GDN_STATE_BF16"),
        },
        "returncode": returncode, "duration_s": time.monotonic() - started,
        "stdout": str(stdout_path), "stderr": str(stderr_path), "metrics": row,
    }


def run(args: argparse.Namespace) -> dict:
    descriptor_path = Path(args.descriptor).resolve()
    descriptor = historical_tasks.HistoricalTaskDescriptor.from_dict(
        json.loads(descriptor_path.read_text(encoding="utf-8")))
    source_repo = Path(descriptor.source_repo).resolve()
    frozen = Path("/mnt/raid0/llm/llama.cpp").resolve()
    if source_repo == frozen or frozen in source_repo.parents:
        raise RuntimeError("historical replay refuses the frozen production tree")
    model = Path(descriptor.model_path)
    if sha256_path(model) != descriptor.model_sha256:
        raise RuntimeError("historical task model SHA-256 drifted")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AutoKernel historical replay evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    workspace = Path(args.workspace_dir).resolve()
    parent_tree = workspace / "parent"
    expert_tree = workspace / "expert"
    if workspace.exists():
        if not args.reuse_workspace:
            raise RuntimeError(f"historical replay workspace already exists: {workspace}")
        for tree, commit in ((parent_tree, descriptor.parent_commit),
                             (expert_tree, descriptor.expert_commit)):
            actual = subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=tree, text=True).strip()
            if actual != commit:
                raise RuntimeError(f"reused historical worktree drifted: {tree}")
    else:
        workspace.mkdir(parents=True)
        create_worktree(repo=source_repo, path=parent_tree, commit=descriptor.parent_commit)
        create_worktree(repo=source_repo, path=expert_tree, commit=descriptor.expert_commit)
    builds = {
        "parent": build_arm(
            arm="parent", tree=parent_tree, output_dir=output_dir,
            jobs=args.build_jobs, timeout_s=args.build_timeout_s),
        "expert": build_arm(
            arm="expert", tree=expert_tree, output_dir=output_dir,
            jobs=args.build_jobs, timeout_s=args.build_timeout_s),
    }
    if builds["parent"]["source_commit"] != descriptor.parent_commit:
        raise RuntimeError("parent historical worktree commit drifted")
    if builds["expert"]["source_commit"] != descriptor.expert_commit:
        raise RuntimeError("expert historical worktree commit drifted")
    arm_builds = {
        "parent": builds["parent"], "expert_off": builds["expert"],
        "expert_on": builds["expert"],
    }

    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="RVP-C5-R matched historical expert replay",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_historical_replay.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=3 * descriptor.minimum_repeats * args.run_timeout_s + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling = None
    runs = []
    started_at = utc_now()
    started = time.monotonic()
    try:
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        for block in range(descriptor.minimum_repeats):
            for arm in ARM_ORDER[block % len(ARM_ORDER)]:
                runs.append(run_benchmark(
                    arm=arm, block=block, build=arm_builds[arm],
                    descriptor=descriptor, output_dir=output_dir,
                    timeout_s=args.run_timeout_s))
        held = device_claim.check_device_claim_held(claim.receipt())
    finally:
        if sampler is not None:
            sampling = sampler.stop()
        released = claim.release().to_dict()
    if sampling is None:
        raise RuntimeError("historical replay completed without device sampling")
    samples = {
        arm: [run["metrics"]["speed_tg"] for run in runs if run["arm"] == arm]
        for arm in ("parent", "expert_off", "expert_on")
    }
    ceiling = historical_tasks.score_expert_ceiling(
        baseline_samples=samples["parent"], expert_samples=samples["expert_on"],
        candidate_samples=None, minimum_repeats=descriptor.minimum_repeats,
        metric_direction=descriptor.metric_direction)
    off_mean = sum(samples["expert_off"]) / len(samples["expert_off"])
    parent_mean = sum(samples["parent"]) / len(samples["parent"])
    payload = {
        "schema": SCHEMA, "campaign_id": args.campaign_id,
        "started_at": started_at, "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "descriptor": descriptor.to_dict(),
        "descriptor_sha256": descriptor.canonical_sha256(),
        "descriptor_path": str(descriptor_path), "workspace_dir": str(workspace),
        "model_sha256_verified": True, "builds": builds, "runs": runs,
        "samples": samples, "expert_ceiling": ceiling.to_dict(),
        "same_binary_off_vs_parent_pct": 100.0 * (off_mean / parent_mean - 1.0),
        "device_claim_open": opened,
        "device_claim_held_after_runs": {
            "outcome": held.outcome, "reasons": list(held.reasons)},
        "device_claim_released": released,
        "device_sampling": sampling.to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--descriptor", required=True)
    result.add_argument("--workspace-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="rvp-c5-r-gdn-bf16-20260811")
    result.add_argument("--build-jobs", type=int, default=96)
    result.add_argument("--build-timeout-s", type=float, default=3600.0)
    result.add_argument("--run-timeout-s", type=float, default=600.0)
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--reuse-workspace", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "parent_mean": payload["expert_ceiling"]["baseline"],
        "expert_mean": payload["expert_ceiling"]["expert"],
        "expert_gain_pct": payload["expert_ceiling"]["expert_gain_pct"],
        "candidate_check": payload["expert_ceiling"]["check"]["outcome"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
