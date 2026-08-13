#!/usr/bin/env python3
"""Six-call, non-promotable MI210 discovery screen over one sealed factor."""
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
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import schemas, storage
from scripts.kernel_rnd.autokernel.execution import (
    cpu_region_claim, device_sampler, inference_window)
from scripts.kernel_rnd.autokernel.resource import device_claim
from scripts.benchmark import autokernel_gpu_discovery_beliefs as gpu_beliefs
from scripts.benchmark import autokernel_progression


SCHEMA_BANK = "epyc.autokernel.gpu_screening_baseline.v2"
SCHEMA_RESULT = "epyc.autokernel.gpu_candidate_only_screen.v2"
SOURCE_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
CPU_LIST = "184-191"
DEVICE_ID = "mi210_0"
SMALL_MODEL_OVERLAP_MAX_BYTES = 512 * 1024 * 1024
VRAM_USED = Path("/sys/class/drm/card2/device/mem_info_vram_used")
KFD_PROCS = Path("/sys/class/kfd/kfd/proc")
MODEL_CALL_WINDOW = inference_window.InferenceCallWindow(timeout_s=600.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def invoke(*, build: Path, model: Path, seed: int, baseline_vram: int,
           flash_attention: bool, campaign_id: str,
           cpu_journal: cpu_region_claim.RegionClaimJournal,
           allow_small_model_cpu_overlap: bool = False) -> dict:
    """Run one model load/measurement while excluding CPU model calls only."""
    if allow_small_model_cpu_overlap:
        model_bytes = model.stat().st_size
        if model_bytes > SMALL_MODEL_OVERLAP_MAX_BYTES:
            raise RuntimeError(
                f"small-model CPU-overlap mode refuses {model_bytes} bytes; "
                f"limit is {SMALL_MODEL_OVERLAP_MAX_BYTES}")
        claims = cpu_region_claim.inspect_region_claims()
        concurrent = []
        for region, entries in (claims.get("regions") or {}).items():
            for entry in entries:
                if not entry.get("held"):
                    continue
                concurrent.append({
                    "region": region, "role": entry.get("role"),
                    "holder_pids": entry.get("holder_pids") or [],
                    "attribution": entry.get("attribution"),
                })
        result = _invoke_locked(
            build=build, model=model, seed=seed, baseline_vram=baseline_vram,
            flash_attention=flash_attention)
        result["inference_call_window"] = None
        result["cpu_coverage"] = {
            "schema": "epyc.autokernel.discovery_cpu_overlap.v1",
            "cpu_overlap_policy": "allowed_discovery_noise",
            "cpu_exclusivity": False, "borrowed": False,
            "model_size_bytes": model_bytes,
            "small_model_threshold_bytes": SMALL_MODEL_OVERLAP_MAX_BYTES,
            "concurrent_claims": concurrent,
            "promotion_claim": False,
        }
        return result
    with MODEL_CALL_WINDOW.hold() as lease:
        owned_claim = None
        try:
            try:
                owned_claim = cpu_region_claim.acquire_cpu_region_claim(
                    CPU_LIST, purpose="AutoKernel GPU model-call helper window",
                    campaign_id=campaign_id, journal=cpu_journal,
                    role="autokernel-gpu-discovery", timeout_s=0, max_hold_s=300)
                coverage = owned_claim
                coverage_receipt = {
                    "schema": "epyc.autokernel.owned_cpu_coverage.v1",
                    "borrowed": False,
                    "claim": owned_claim.receipt().to_dict(),
                }
            except cpu_region_claim.CpuRegionClaimTimeout:
                coverage = inference_window.borrow_windowed_cpu_coverage(CPU_LIST)
                coverage_receipt = coverage.to_dict()
            result = _invoke_locked(
                build=build, model=model, seed=seed, baseline_vram=baseline_vram,
                flash_attention=flash_attention)
            if getattr(coverage, "borrowed", False):
                coverage.validate()
        finally:
            if owned_claim is not None:
                owned_claim.release()
    result["inference_call_window"] = {
        "schema": "epyc.autokernel.inference_call_window.v1",
        "lock_path": str(lease.path),
        "waited_s": lease.waited_s,
        "scope": "model_load_and_inference_only",
    }
    result["cpu_coverage"] = coverage_receipt
    return result


def _invoke_locked(*, build: Path, model: Path, seed: int, baseline_vram: int,
                   flash_attention: bool) -> dict:
    binary = build / "bin" / "llama-bench"
    argv = ("taskset", "-c", CPU_LIST, "numactl", "--interleave=all", str(binary),
            "-m", str(model), "-p", "512", "-n", "0", "-r", "1", "-ngl", "99",
            "-fa", "on" if flash_attention else "off",
            "--autokernel-harden", str(seed), "-o", "jsonl")
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
    expected_flash = 1 if flash_attention else 0
    if (row.get("n_prompt") != 512 or row.get("n_gen") != 0
            or row.get("flash_attn") != expected_flash):
        raise RuntimeError("GPU discovery result differs from the sealed pp512 frame")
    if not any(sample["owned_kfd_pids"] for sample in samples):
        raise RuntimeError("GPU discovery window has no owned KFD residency sample")
    if max(sample["vram_used_bytes"] for sample in samples) <= baseline_vram:
        raise RuntimeError("GPU discovery window has no positive VRAM residency delta")
    return {"argv": list(argv), "env": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]},
            "metric": float(row["avg_ts"]), "raw_row": row,
            "stderr_tail": stderr[-2000:], "residency": samples,
            "hip_residency_proved": True}


def factor_spec(*, factor: str, anchor_build: Path, candidate_build: Path,
                anchor_identity: dict, candidate_identity: dict) -> dict:
    """Validate and describe the only difference admitted by this screen."""
    if factor == "mmq_mfma":
        if not anchor_identity["rocwmma_fattn"] or not candidate_identity["rocwmma_fattn"]:
            raise RuntimeError("both MMQ arms must keep ROCWMMA_FATTN=ON")
        if anchor_identity["mmq_mfma"] is not True or candidate_identity["mmq_mfma"] is not False:
            raise RuntimeError("sole factor must be GGML_HIP_MMQ_MFMA ON->OFF")
        return {
            "name": "GGML_HIP_MMQ_MFMA", "anchor": "ON", "candidate": "OFF",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "flash_attention":
        if anchor_build != candidate_build:
            raise RuntimeError("flash_attention screen requires one identical build path for both arms")
        if anchor_identity != candidate_identity:
            raise RuntimeError("flash_attention screen requires identical sealed build identities")
        if not anchor_identity["rocwmma_fattn"] or anchor_identity["mmq_mfma"]:
            raise RuntimeError("flash_attention screen requires the r1m0 build (ROCWMMA ON, MMQ MFMA OFF)")
        return {
            "name": "flash_attention", "anchor": "OFF", "candidate": "ON",
            "anchor_flash_attention": False, "candidate_flash_attention": True,
        }
    if factor == "rocwmma_fattn":
        if anchor_identity["mmq_mfma"] or candidate_identity["mmq_mfma"]:
            raise RuntimeError("both ROCWMMA arms must keep MMQ_MFMA=OFF")
        if anchor_identity["rocwmma_fattn"] is not False or candidate_identity["rocwmma_fattn"] is not True:
            raise RuntimeError("sole factor must be GGML_HIP_ROCWMMA_FATTN OFF->ON")
        return {
            "name": "GGML_HIP_ROCWMMA_FATTN", "anchor": "OFF", "candidate": "ON",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    raise RuntimeError(f"unsupported GPU discovery factor: {factor}")


def preflight(args: argparse.Namespace) -> dict:
    model = Path(args.model).resolve()
    anchor_build = Path(args.anchor_build).resolve()
    candidate_build = Path(args.candidate_build).resolve()
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    anchor_identity = build_identity(anchor_build)
    candidate_identity = build_identity(candidate_build)
    factor = factor_spec(
        factor=args.factor, anchor_build=anchor_build, candidate_build=candidate_build,
        anchor_identity=anchor_identity, candidate_identity=candidate_identity)
    return {
        "schema": "epyc.autokernel.gpu_discovery_preflight.v1",
        "campaign_id": args.campaign_id,
        "authority": "nonpromotable_candidate_only_discovery",
        "model": str(model),
        "model_sha256": sha256_file(model),
        "anchor_build": str(anchor_build),
        "candidate_build": str(candidate_build),
        "anchor_identity": anchor_identity,
        "candidate_identity": candidate_identity,
        "sole_factor": {key: factor[key] for key in ("name", "anchor", "candidate")},
        "anchor_flash_attention": factor["anchor_flash_attention"],
        "candidate_flash_attention": factor["candidate_flash_attention"],
        "frame": "pp512-ngl99",
        "invocations": {"anchor": 3, "candidate": 3},
        "inference_executed": False,
    }


def run(args: argparse.Namespace) -> dict:
    sealed = preflight(args)
    started_at = utc_now()
    out = Path(storage.assert_not_scratch(args.output_dir, what="GPU discovery output"))
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "preflight.json", sealed)
    model = Path(sealed["model"])
    anchor_build = Path(sealed["anchor_build"])
    candidate_build = Path(sealed["candidate_build"])
    anchor_identity = sealed["anchor_identity"]
    candidate_identity = sealed["candidate_identity"]
    sole_factor = sealed["sole_factor"]
    if _kfd_pids():
        raise RuntimeError("MI210 already has KFD users")
    baseline_vram = int(VRAM_USED.read_text(encoding="utf-8").strip())
    purpose = ("AutoKernel GPU candidate-only discovery "
               f"{sole_factor['name']} {sole_factor['anchor']}->{sole_factor['candidate']}")
    cpu_journal = cpu_region_claim.RegionClaimJournal(args.cpu_claim_journal)
    gpu_journal = device_claim.ClaimJournal(args.device_claim_journal)
    gpu = None
    sampler = None
    try:
        gpu = device_claim.acquire_device_claim(
            DEVICE_ID, purpose=purpose, campaign_id=args.campaign_id,
            journal=gpu_journal, timeout_s=0, max_hold_s=300)
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        anchor_runs = [invoke(
            build=anchor_build, model=model, seed=args.seed + i,
            baseline_vram=baseline_vram,
            flash_attention=sealed["anchor_flash_attention"],
            campaign_id=args.campaign_id, cpu_journal=cpu_journal,
            allow_small_model_cpu_overlap=args.allow_small_model_cpu_overlap)
            for i in range(3)]
        bank_body = {
            "schema": SCHEMA_BANK, "campaign_id": args.campaign_id,
            "status": "complete", "started_at": started_at, "ended_at": utc_now(),
            "authority": "nonpromotable_candidate_only_discovery",
            "frame": {"backend": "llama_gpu", "recipe": "pp512-ngl99",
                      "metric": "prefill_tokens_per_s", "metric_direction": "higher_better",
                      "model": str(model), "model_sha256": sha256_file(model),
                      "source_commit": SOURCE_COMMIT, "cpu_list": CPU_LIST,
                      "device": "AMD Instinct MI210", "architecture": "gfx90a"},
            "sole_factor": sole_factor,
            "anchor_identity": anchor_identity,
            "candidate_identity": candidate_identity,
            "anchor_samples": [run["metric"] for run in anchor_runs],
            "anchor_runs": anchor_runs,
        }
        bank = gpu_beliefs.attach_baseline_beliefs(
            bank_body, producer_path=Path(__file__).resolve())
        atomic_json(out / "baseline-bank.json", bank)
        candidate_runs = [invoke(
            build=candidate_build, model=model, seed=args.seed + 3 + i,
            baseline_vram=baseline_vram,
            flash_attention=sealed["candidate_flash_attention"],
            campaign_id=args.campaign_id, cpu_journal=cpu_journal,
            allow_small_model_cpu_overlap=args.allow_small_model_cpu_overlap)
            for i in range(3)]
        center = sum(bank["anchor_samples"]) / 3
        values = [run["metric"] for run in candidate_runs]
        effects = [(value - center) / center for value in values]
        numeric = sampler.stop().to_dict()
        sampler = None
        result_body = {
            "schema": SCHEMA_RESULT, "campaign_id": args.campaign_id,
            "status": "complete", "started_at": started_at, "ended_at": utc_now(),
            "authority": "nonpromotable_candidate_only_discovery",
            "state": "decided", "ok": True, "non_promotable": True,
            "nomination": "top_k_candidate_only_not_a_keep",
            "baseline_sha256": bank["baseline_sha256"],
            "anchor_invocations": 3, "candidate_invocations": 3,
            "baseline_center": center, "candidate_samples": values,
            "relative_effects": effects, "median_relative": median(effects),
            "host_noise_policy": "ordinary_host_activity_recorded_not_blocking",
            "cpu_overlap_policy": ("allowed_discovery_noise"
                                   if args.allow_small_model_cpu_overlap
                                   else "shared_model_call_window"),
            "model_size_bytes": model.stat().st_size,
            "small_model_overlap_max_bytes": SMALL_MODEL_OVERLAP_MAX_BYTES,
            "promotion_claim": False,
            "frame": bank["frame"], "sole_factor": bank["sole_factor"],
            "candidate_identity": bank["candidate_identity"],
            "candidate_runs": candidate_runs, "device_sampling": numeric,
            "hip_residency_proved": all(run["hip_residency_proved"]
                                         for run in anchor_runs + candidate_runs),
            "cpu_coverage_windows": [
                run["cpu_coverage"] for run in anchor_runs + candidate_runs],
            "device_claim_open": gpu.receipt().to_dict(),
        }
        result = gpu_beliefs.attach_result_beliefs(
            result_body, bank=bank, producer_path=Path(__file__).resolve())
        atomic_json(out / "result.json", result)
        # A derived operator view, kept separate from the strict terminal
        # campaign contract.  The immutable result above is already durable, so
        # an export failure must not erase or reclassify the measurement.
        try:
            autokernel_progression.export_progression()
        except Exception as exc:
            print(f"WARNING: GPU result is durable but progression export failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return result
    finally:
        if sampler is not None:
            sampler.stop()
        if gpu is not None:
            gpu.release()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anchor-build", required=True)
    result.add_argument("--candidate-build", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--factor", choices=("mmq_mfma", "flash_attention", "rocwmma_fattn"),
                        default="mmq_mfma")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--preflight-output")
    result.add_argument("--seed", type=int, default=8613)
    result.add_argument("--allow-small-model-cpu-overlap", action="store_true",
                        help="nonpromotable discovery only: treat CPU inference as noise "
                             "for models no larger than the sealed small-model limit")
    result.add_argument("--cpu-claim-journal", default="/mnt/raid0/llm/ak-claims/region.jsonl")
    result.add_argument("--device-claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        payload = preflight(args) if args.preflight_only else run(args)
        if args.preflight_output:
            atomic_json(Path(args.preflight_output), payload)
    except Exception as exc:
        print(f"GPU discovery REFUSED: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    if args.preflight_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({key: payload[key] for key in (
            "state", "baseline_center", "candidate_samples", "median_relative",
            "hip_residency_proved", "result_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
