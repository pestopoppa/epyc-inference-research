#!/usr/bin/env python3
"""Run the AK-BH-2 gfx90a baseline-honesty factorial with retained evidence."""
from __future__ import annotations

import argparse
import hashlib
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

from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.ak_bh_2_gpu_factorial.v1"
VARIANT_RE = re.compile(r"r([01])m([01])$")


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


def cache_bool(cache: Path, name: str) -> bool:
    prefix = f"{name}:BOOL="
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):]
            if value in {"ON", "OFF"}:
                return value == "ON"
    raise RuntimeError(f"{cache} does not declare {name}:BOOL=ON|OFF")


def resolve_variants(build_dirs: list[str]) -> list[dict]:
    variants = []
    seen = set()
    for value in build_dirs:
        root = Path(value).resolve()
        binary = root / "bin" / "llama-bench"
        cache = root / "CMakeCache.txt"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"llama-bench is not executable: {binary}")
        match = VARIANT_RE.search(root.name)
        if match is None:
            raise RuntimeError(
                f"factorial build directory must end in r0m0, r0m1, r1m0, or r1m1: {root}")
        expected = (match.group(1) == "1", match.group(2) == "1")
        observed = (
            cache_bool(cache, "GGML_HIP_ROCWMMA_FATTN"),
            cache_bool(cache, "GGML_HIP_MMQ_MFMA"),
        )
        if observed != expected:
            raise RuntimeError(f"{root} name declares {expected}, CMake cache declares {observed}")
        if observed in seen:
            raise RuntimeError(f"duplicate factorial build for rocWMMA/MFMA={observed}")
        seen.add(observed)
        variants.append({
            "build_root": str(root),
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "rocwmma_fattn": observed[0],
            "mmq_mfma": observed[1],
        })
    if seen != {(False, False), (False, True), (True, False), (True, True)}:
        raise RuntimeError("AK-BH-2 requires all four rocWMMA x MMQ-MFMA build cells")
    return sorted(variants, key=lambda item: (item["rocwmma_fattn"], item["mmq_mfma"]))


def run_arm(*, arm_id: str, variant: dict, flash_attention: str, model: Path,
            args: argparse.Namespace, output_dir: Path) -> dict:
    binary = Path(variant["binary"])
    command = (
        str(binary), "-m", str(model), "-p", str(args.prompt_tokens),
        "-n", "0", "-r", str(args.repetitions), "-ngl", "99",
        "-fa", flash_attention, "--autokernel-harden", str(args.suite_seed),
        "-o", "jsonl")
    stdout_path = output_dir / f"{arm_id}.stdout.jsonl"
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
    stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if returncode != 0:
        raise RuntimeError(f"factorial arm {arm_id} exited {returncode}: {stderr_tail!r}")
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"factorial arm {arm_id} emitted {len(lines)} JSON rows, expected 1")
    row = json.loads(lines[0])
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise RuntimeError(f"factorial arm {arm_id} did not run on the MI210 ROCm backend")
    if row.get("flash_attn") != (1 if flash_attention == "on" else 0):
        raise RuntimeError(f"factorial arm {arm_id} did not preserve the explicit -fa setting")
    required_true = (
        "autokernel_hardened", "autokernel_output_invariant",
        "autokernel_hybrid_ab_complete", "autokernel_thread_set_stable",
        "autokernel_escape_checks_complete")
    if any(row.get(field) is not True for field in required_true):
        raise RuntimeError(f"factorial arm {arm_id} has an incomplete hardening receipt")
    samples = row.get("samples_ns")
    if not isinstance(samples, list) or len(samples) != args.repetitions:
        raise RuntimeError(f"factorial arm {arm_id} did not retain every raw repetition")
    return {
        "arm_id": arm_id,
        "rocwmma_fattn": variant["rocwmma_fattn"],
        "mmq_mfma": variant["mmq_mfma"],
        "flash_attention": flash_attention,
        "binary": str(binary),
        "binary_sha256": variant["binary_sha256"],
        "command": list(command),
        "duration_s": time.monotonic() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stderr_tail": stderr_tail,
        "result": row,
        "timing_window_ns": sum(float(value) for value in samples),
    }


def run(args: argparse.Namespace) -> dict:
    model = Path(args.model).resolve()
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    variants = resolve_variants(args.build_dir)
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="AK-BH-2 evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AK-BH-2 ROCm baseline-honesty factorial",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_gpu_factorial.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=8 * args.arm_timeout_s + 300.0)
    opened_receipt = claim.receipt().to_dict()
    session = None
    sampling_receipt = None
    arms = []
    started_at = utc_now()
    started_mono = time.monotonic()
    try:
        session = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        for variant in variants:
            for flash_attention in ("off", "on"):
                arm_id = (
                    f"r{int(variant['rocwmma_fattn'])}"
                    f"m{int(variant['mmq_mfma'])}-fa-{flash_attention}")
                arms.append(run_arm(
                    arm_id=arm_id, variant=variant, flash_attention=flash_attention,
                    model=model, args=args, output_dir=output_dir))
    finally:
        if session is not None:
            sampling_receipt = session.stop()
        released_receipt = claim.release().to_dict()
    if sampling_receipt is None:
        raise RuntimeError("AK-BH-2 completed without a device sampling receipt")
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started_mono,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "prompt_tokens": args.prompt_tokens,
        "repetitions": args.repetitions,
        "suite_seed": args.suite_seed,
        "variants": variants,
        "arms": arms,
        "device_claim_open": opened_receipt,
        "device_claim_released": released_receipt,
        "device_sampling": sampling_receipt.to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--build-dir", action="append", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="ak-bh-2-20260811")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--arm-timeout-s", type=float, default=300.0)
    result.add_argument("--prompt-tokens", type=int, default=512)
    result.add_argument("--repetitions", type=int, default=30)
    result.add_argument("--suite-seed", type=int, default=4711)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        print(f"AK-BH-2 REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ranked = sorted(
        ((arm["result"]["avg_ts"], arm["arm_id"]) for arm in payload["arms"]),
        reverse=True)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "arms": len(payload["arms"]),
        "winner": ranked[0][1],
        "winner_ts": ranked[0][0],
        "slowest": ranked[-1][1],
        "slowest_ts": ranked[-1][0],
        "samples": payload["device_sampling"]["sample_count"],
        "max_gap_s": payload["device_sampling"]["max_gap_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
