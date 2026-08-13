#!/usr/bin/env python3
"""Six-call, non-promotable MI210 discovery screen over one build factor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import schemas, storage
from scripts.kernel_rnd.autokernel.execution import cpu_region_claim, device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA_BANK = "epyc.autokernel.gpu_screening_baseline.v1"
SCHEMA_RESULT = "epyc.autokernel.gpu_candidate_only_screen.v1"
SOURCE_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
CPU_LIST = "184-191"
DEVICE_ID = "mi210_0"
VRAM_USED = Path("/sys/class/drm/card2/device/mem_info_vram_used")
KFD_PROCS = Path("/sys/class/kfd/kfd/proc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cache_bool(build: Path, name: str) -> bool:
    prefix = f"{name}:BOOL="
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):]
            if value in {"ON", "OFF"}:
                return value == "ON"
    raise RuntimeError(f"{build} does not declare {name}:BOOL=ON|OFF")


def artifacts(build: Path) -> dict:
    binary = build / "bin" / "llama-bench"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-bench is not executable: {binary}")
    libraries = {
        path.name: sha256_file(path)
        for path in sorted((build / "bin").glob("*.so*")) if path.is_file()
    }
    return {"binary": str(binary.resolve()), "binary_sha256": sha256_file(binary),
            "libraries": libraries}


def build_identity(build: Path) -> dict:
    identity = {
        "source_commit": SOURCE_COMMIT,
        "rocwmma_fattn": cache_bool(build, "GGML_HIP_ROCWMMA_FATTN"),
        "mmq_mfma": cache_bool(build, "GGML_HIP_MMQ_MFMA"),
        "artifacts": artifacts(build),
    }
    return identity


def _kfd_pids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(path.name) for path in KFD_PROCS.iterdir()
                            if path.name.isdigit()))
    except OSError as exc:
        raise RuntimeError(f"KFD process inventory unreadable: {exc}") from exc


def invoke(*, build: Path, model: Path, seed: int, baseline_vram: int) -> dict:
    binary = build / "bin" / "llama-bench"
    argv = ("taskset", "-c", CPU_LIST, "numactl", "--interleave=all", str(binary),
            "-m", str(model), "-p", "512", "-n", "0", "-r", "1", "-ngl", "99",
            "-fa", "on", "--autokernel-harden", str(seed), "-o", "jsonl")
    env = {**os.environ, "LD_LIBRARY_PATH": f"{binary.parent}:/opt/rocm/lib"}
    process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    samples = []
    while process.poll() is None:
        kfd = _kfd_pids()
        try:
            vram = int(VRAM_USED.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError(f"VRAM residency counter unreadable: {exc}") from exc
        owned = []
        foreign = []
        for pid in kfd:
            try:
                (owned if os.getpgid(pid) == process.pid else foreign).append(pid)
            except (ProcessLookupError, PermissionError):
                continue
        if foreign:
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError(f"foreign KFD inference overlapped discovery: {foreign}")
        samples.append({"offset_s": time.monotonic(), "kfd_pids": list(kfd),
                        "owned_kfd_pids": owned, "vram_used_bytes": vram})
        time.sleep(0.05)
    stdout, stderr = process.communicate(timeout=10)
    if process.returncode != 0:
        raise RuntimeError(f"GPU discovery invocation exited {process.returncode}: {stderr[-2000:]}")
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"GPU discovery invocation emitted {len(rows)} rows")
    row = rows[0]
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise RuntimeError("GPU discovery invocation did not report MI210 ROCm execution")
    reported_commit = str(row.get("build_commit", ""))
    if len(reported_commit) < 7 or not SOURCE_COMMIT.startswith(reported_commit):
        raise RuntimeError("GPU discovery binary does not report the sealed source commit")
    if row.get("n_prompt") != 512 or row.get("n_gen") != 0 or row.get("flash_attn") != 1:
        raise RuntimeError("GPU discovery result differs from the sealed pp512/-fa-on frame")
    if not any(sample["owned_kfd_pids"] for sample in samples):
        raise RuntimeError("GPU discovery window has no owned KFD residency sample")
    if max(sample["vram_used_bytes"] for sample in samples) <= baseline_vram:
        raise RuntimeError("GPU discovery window has no positive VRAM residency delta")
    return {"argv": list(argv), "env": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]},
            "metric": float(row["avg_ts"]), "raw_row": row,
            "stderr_tail": stderr[-2000:], "residency": samples,
            "hip_residency_proved": True}


def run(args: argparse.Namespace) -> dict:
    out = Path(storage.assert_not_scratch(args.output_dir, what="GPU discovery output"))
    out.mkdir(parents=True, exist_ok=False)
    model = Path(args.model).resolve()
    anchor_build, candidate_build = Path(args.anchor_build).resolve(), Path(args.candidate_build).resolve()
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    anchor_identity, candidate_identity = build_identity(anchor_build), build_identity(candidate_build)
    if not anchor_identity["rocwmma_fattn"] or not candidate_identity["rocwmma_fattn"]:
        raise RuntimeError("both arms must keep ROCWMMA_FATTN=ON")
    if anchor_identity["mmq_mfma"] is not True or candidate_identity["mmq_mfma"] is not False:
        raise RuntimeError("sole factor must be GGML_HIP_MMQ_MFMA ON->OFF")
    if _kfd_pids():
        raise RuntimeError("MI210 already has KFD users")
    baseline_vram = int(VRAM_USED.read_text(encoding="utf-8").strip())
    purpose = "AutoKernel GPU candidate-only discovery MMQ_MFMA ON->OFF"
    cpu_journal = cpu_region_claim.RegionClaimJournal(args.cpu_claim_journal)
    gpu_journal = device_claim.ClaimJournal(args.device_claim_journal)
    cpu = cpu_region_claim.acquire_cpu_region_claim(
        CPU_LIST, purpose=purpose, campaign_id=args.campaign_id, journal=cpu_journal,
        timeout_s=0, max_hold_s=300)
    gpu = None
    sampler = None
    try:
        gpu = device_claim.acquire_device_claim(
            DEVICE_ID, purpose=purpose, campaign_id=args.campaign_id,
            journal=gpu_journal, timeout_s=0, max_hold_s=300)
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        anchor_runs = [invoke(build=anchor_build, model=model, seed=args.seed + i,
                              baseline_vram=baseline_vram) for i in range(3)]
        bank_body = {
            "schema": SCHEMA_BANK, "campaign_id": args.campaign_id,
            "authority": "nonpromotable_candidate_only_discovery",
            "frame": {"backend": "llama_gpu", "recipe": "pp512-fa-on-ngl99",
                      "metric": "prefill_tokens_per_s", "metric_direction": "higher_better",
                      "model": str(model), "model_sha256": sha256_file(model),
                      "source_commit": SOURCE_COMMIT, "cpu_list": CPU_LIST,
                      "device": "AMD Instinct MI210", "architecture": "gfx90a"},
            "sole_build_factor": {"name": "GGML_HIP_MMQ_MFMA", "anchor": "ON",
                                  "candidate": "OFF"},
            "anchor_identity": anchor_identity,
            "candidate_identity": candidate_identity,
            "anchor_samples": [run["metric"] for run in anchor_runs],
            "anchor_runs": anchor_runs,
        }
        bank = {**bank_body, "baseline_sha256": schemas.content_hash(bank_body)}
        atomic_json(out / "baseline-bank.json", bank)
        candidate_runs = [invoke(build=candidate_build, model=model, seed=args.seed + 3 + i,
                                 baseline_vram=baseline_vram) for i in range(3)]
        center = sum(bank["anchor_samples"]) / 3
        values = [run["metric"] for run in candidate_runs]
        effects = [(value - center) / center for value in values]
        numeric = sampler.stop().to_dict()
        sampler = None
        result_body = {
            "schema": SCHEMA_RESULT, "campaign_id": args.campaign_id,
            "state": "decided", "ok": True, "non_promotable": True,
            "nomination": "top_k_candidate_only_not_a_keep",
            "baseline_sha256": bank["baseline_sha256"],
            "anchor_invocations": 3, "candidate_invocations": 3,
            "baseline_center": center, "candidate_samples": values,
            "relative_effects": effects, "median_relative": median(effects),
            "host_noise_policy": "ordinary_host_activity_recorded_not_blocking",
            "sole_build_factor": bank["sole_build_factor"],
            "candidate_runs": candidate_runs, "device_sampling": numeric,
            "hip_residency_proved": all(run["hip_residency_proved"]
                                         for run in anchor_runs + candidate_runs),
            "cpu_claim_open": cpu.receipt().to_dict(),
            "device_claim_open": gpu.receipt().to_dict(),
        }
        result = {**result_body, "result_sha256": schemas.content_hash(result_body)}
        atomic_json(out / "result.json", result)
        return result
    finally:
        if sampler is not None:
            sampler.stop()
        if gpu is not None:
            gpu.release()
        cpu.release()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anchor-build", required=True)
    result.add_argument("--candidate-build", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--seed", type=int, default=8613)
    result.add_argument("--cpu-claim-journal", default="/mnt/raid0/llm/ak-claims/region.jsonl")
    result.add_argument("--device-claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    return result


def main() -> int:
    try:
        payload = run(parser().parse_args())
    except Exception as exc:
        print(f"GPU discovery REFUSED: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps({key: payload[key] for key in (
        "state", "baseline_center", "candidate_samples", "median_relative",
        "hip_residency_proved", "result_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
