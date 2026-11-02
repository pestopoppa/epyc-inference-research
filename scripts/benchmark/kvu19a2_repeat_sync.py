#!/usr/bin/env python3
"""Predeclared two-case, five-pair native FA diagnostic (no model inference)."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from kvu19a2_custody import process_birth, sample_process, sha_file

sys.path.insert(0, str(Path(os.environ["EPYC_VIDYA_ROOT"]).resolve()))
from adapters.kvu19a2_native import record_run  # noqa: E402

EXE = Path(os.environ["KVU19A2_EXE"]).resolve()
OUT = Path(os.environ["KVU19A2_REPEAT_OUT"]).resolve()
ARCHIVE = Path(os.environ["KVU19A2_SOURCE_ARCHIVE"]).resolve()
ROOT = Path("/dev/shm/kvu19a2-immutable-build-v2-20261102")
BUILD_RECEIPT = ROOT / "build-receipt.json"
BUILD_RECEIPT_SHA = os.environ["KVU19A2_BUILD_RECEIPT_SHA256"]
EXPECTED_EXE = os.environ["KVU19A2_EXE_SHA256"]
HARNESS_RECEIPT_SHA = os.environ["KVU19A2_HARNESS_RECEIPT_SHA256"]
EXPECTED_ARCHIVE = "87e3cdc13b1859a31dfd3ced43c596e6b9653e2194b1ebd0d5eb6772593c04c2"
PLAN_SHA = "1746ce6590047d6f021c7a94d2f2d123f70bca6f4527be3734dcb6670dcc8910"
SYNC_RECEIPT = Path("/dev/shm/kvu19a2-sync-build-20261102/receipt.json")
SYNC_RECEIPT_SHA = "1d00db9e5fcf394d59d7548d1269fdfa2333bc3559da4a69e446ad1efc85dafc"
SYNC_ROOT = Path("/dev/shm/kvu19a2-sync-build-20261102/build/bin")
SYNC_ARCHIVE_SHA = "34251fa02113cbccc5a2d911bc6d0274ef2730bd31474075aa1129772fb65559"
INPUT_PIN_SHA = "f9fc6b1f96d359c7dad6023794a9b804e71910a2d6ae31acc25f2faaf317f1e8"


def time_now():
    return datetime.now(timezone.utc).isoformat()


def vram_sample(pid):
    root = Path("/sys/class/kfd/kfd/proc") / str(pid)
    present = root.exists()
    try:
        files = list(root.glob("vram_*")) if present else []
        values = {p.name: int(p.read_text().strip()) for p in files}
        total = sum(values.values()) if values else None
    except (OSError, ValueError):
        values, total = {}, None
    return {"at": time_now(), "kfd": present, "vram_bytes": total, "vram_files": values}


def compare(results, canonical):
    comparisons = {}
    for case in (78, 80):
        rows = [r for r in results if r["case"] == case]
        expected_inputs = 3 if case == 78 else 4
        common = len(rows) == 10 and len(canonical[str(case)]) == expected_inputs and all(
            r["input_hashes"] == canonical[str(case)] for r in rows) and all(
            r["input_hashes"] == rows[0]["input_hashes"] for r in rows)
        comparisons[str(case)] = {"input_match": common, "arms": {}}
        for arm in ("base", "candidate"):
            group = [r for r in rows if r["arm"] == arm]
            complete = common and len(group) == 5 and {r["repeat"] for r in group} == set(range(1, 6)) and all(
                r["rc"] == 0 and r["output_sha256"] and r["mapping"] and any(
                    s["kfd"] and s["vram_bytes"] is not None and s["vram_bytes"] > 0 for s in r["samples"])
                for r in group)
            hashes = sorted({r["output_sha256"] for r in group if r["output_sha256"]})
            comparisons[str(case)]["arms"][arm] = {"verdict": ("BIT_IDENTICAL" if len(hashes) == 1 else "VARIATION")
                                                    if complete else "UNKNOWN", "output_sha256": hashes}
    return comparisons


def run():
    artifact = Path(__file__).resolve().parents[2] / "artifacts/kvu19a2-repeat-20261102"
    plan = artifact.parent / "kvu19a2-sync-build-20261102/PREDECLARATION.md"
    input_pin = artifact / "canonical-input-hashes.json"
    harness_receipt = EXE.parent / "harness-receipt.json"
    if (sha_file(plan) != PLAN_SHA or sha_file(input_pin) != INPUT_PIN_SHA or sha_file(EXE) != EXPECTED_EXE
            or sha_file(harness_receipt) != HARNESS_RECEIPT_SHA
            or sha_file(ARCHIVE) != EXPECTED_ARCHIVE or sha_file(BUILD_RECEIPT) != BUILD_RECEIPT_SHA
            or sha_file(SYNC_RECEIPT) != SYNC_RECEIPT_SHA
            or sha_file(Path("/dev/shm/kvu19a2-sync-source-20261102.tar")) != SYNC_ARCHIVE_SHA):
        raise ValueError("plan, executable or archive hash changed")
    build = json.loads(BUILD_RECEIPT.read_text())
    if build.get("schema") != "kvu19a2.root-owned-source-build.v1" or build.get("source_archive_sha256") != EXPECTED_ARCHIVE:
        raise ValueError("build receipt source mismatch")
    harness = json.loads(harness_receipt.read_text())
    if (harness.get("schema") != "kvu19a2.harness-build.v1" or harness.get("executable_sha256") != EXPECTED_EXE
            or harness.get("pair_build_receipt_sha256") != BUILD_RECEIPT_SHA):
        raise ValueError("harness receipt mismatch")
    sync = json.loads(SYNC_RECEIPT.read_text())
    if sync.get("commit") != "18d030309a57071c32ffd0db42994ae473d2722a" or sync.get("source_archive_sha256") != SYNC_ARCHIVE_SHA:
        raise ValueError("synchronized build source mismatch")
    canonical = json.loads(input_pin.read_text())["cases"]
    hips = {"base": build["outputs"]["candidate"]["hip_sha256"], "candidate": sync["outputs"]["build/bin/libggml-hip.so"]}
    libdirs = {"base": ROOT / "candidate/bin", "candidate": SYNC_ROOT}
    preflight_libs = {arm: {str(p.resolve()): sha_file(p.resolve())
                            for p in libdirs[arm].glob("libggml*.so.0.16.0")}
                      for arm in ("base", "candidate")}
    OUT.mkdir(exist_ok=False)
    results, observations = [], []
    inroot = Path(tempfile.mkdtemp(prefix="kvu19a2-inputs-", dir="/dev/shm"))
    for case in (78, 80):
        for repeat in range(1, 6):
            arms = ("base", "candidate") if repeat % 2 else ("candidate", "base")
            for arm in arms:
                run_id = f"case{case}-{arm}-r{repeat}"
                output = OUT / run_id
                inputs = inroot / run_id
                output.mkdir(); inputs.mkdir()
                libdir = libdirs[arm]
                hip = (libdir / "libggml-hip.so").resolve()
                if sha_file(hip) != hips[arm]:
                    raise ValueError("HIP digest mismatch")
                libs = preflight_libs[arm]
                if not libs or any(sha_file(Path(path)) != digest for path, digest in libs.items()):
                    raise ValueError("runtime library changed after preflight")
                env = {key: value for key, value in os.environ.items() if not key.startswith("GGML_")}
                env.update(FA_CASE=str(case), FA_DUMP_DIR=str(output), FA_INPUT_DIR=str(inputs),
                           LD_LIBRARY_PATH=str(libdir) + ":/opt/rocm/lib", GGML_BACKEND_PATH=str(hip),
                           GGML_CUDA_FA_MASK_SKIP="0")
                samples, mapping = [], None
                start = time_now()
                log = output / "native.log"
                with log.open("xb") as stream:
                    child = subprocess.Popen([str(EXE), "ROCm0", "exact"], cwd=output, env=env,
                                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = time.monotonic() + 90
                    birth = process_birth(Path("/proc") / str(child.pid))
                    try:
                        while child.poll() is None and time.monotonic() < deadline:
                            sample = vram_sample(child.pid)
                            samples.append(sample)
                            if sample["vram_bytes"] and mapping is None:
                                try:
                                    mapping = sample_process(child.pid, birth, EXPECTED_EXE, libs)
                                except (OSError, ValueError, ProcessLookupError):
                                    pass
                            time.sleep(.01)
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                child.kill(); child.wait()
                            raise TimeoutError("native child timed out")
                        rc = child.wait()
                    finally:
                        if child.poll() is None:
                            child.kill(); child.wait()
                end = time_now()
                input_hashes = {p.name: sha_file(p) for p in sorted(inputs.glob("input_*.bin"))}
                target = output / f"out_{case:02d}.bin"
                captured = target.is_file() and rc == 0
                info = {"case": case, "arm": arm, "repeat": repeat, "pid": child.pid, "birth_ticks": birth,
                        "started_at": start, "ended_at": end, "rc": rc, "input_hashes": input_hashes,
                        "input_dir": str(inputs), "output_sha256": sha_file(target) if captured else None,
                        "log_sha256": sha_file(log), "samples": samples, "mapping": mapping}
                results.append(info)
                obs = {"arm": arm, "case": f"FA_CASE={case},repeat={repeat}",
                       "source_archive_sha256": EXPECTED_ARCHIVE if arm == "base" else SYNC_ARCHIVE_SHA, "executable_sha256": EXPECTED_EXE,
                       "hip_sha256": hips[arm], "mapped_ggml": mapping["mapped_ggml"] if mapping else {},
                       "mapping_sample_present": bool(mapping), "samples": samples,
                       "source_build_binding": "root-owned read-only to build UID; transitive toolchain not fully pinned",
                       "build_receipt_sha256": BUILD_RECEIPT_SHA if arm == "base" else SYNC_RECEIPT_SHA, "harness_receipt_sha256": HARNESS_RECEIPT_SHA,
                       "mask_skip": "0", "input_hashes": input_hashes,
                       "started_at": start, "ended_at": end}
                device = re.search(r"Device 0:.*?\b(gfx[0-9a-f]+)\b", log.read_text(errors="replace"))
                if device and mapping and any(s["kfd"] and s["vram_bytes"] is not None and s["vram_bytes"] > 0
                                              for s in samples):
                    obs.update(applicability={"backend": "gpu", "device": device.group(1)},
                               log_path=str(log), log_sha256=sha_file(log))
                if captured:
                    raw = target.read_bytes()
                    if not raw or len(raw) % 4:
                        raise ValueError("malformed output")
                    obs.update(observed=True, raw_path=str(target), raw_sha256=sha_file(target),
                               nan_elements=sum(math.isnan(x[0]) for x in struct.iter_unpack("<f", raw)))
                else:
                    obs.update(observed=False, nan_elements=None, missing_reason="no completed native output")
                observations.append(obs)
    result = {"schema": "kvu19a2.repeat-diagnostic.v1", "plan_sha256": PLAN_SHA,
              "executable_sha256": EXPECTED_EXE, "source_archive_sha256": EXPECTED_ARCHIVE,
              "source_build_binding": "root-owned read-only to build UID; transitive toolchain not fully pinned",
              "build_receipt_sha256": BUILD_RECEIPT_SHA, "harness_receipt_sha256": HARNESS_RECEIPT_SHA,
              "sync_source_archive_sha256": SYNC_ARCHIVE_SHA, "sync_build_receipt_sha256": SYNC_RECEIPT_SHA,
              "arm_definition": {"base": "original guarded candidate", "candidate": "guarded candidate plus producer barrier fix"},
              "runtime_library_pins": preflight_libs, "mask_skip": "0", "input_root_tmpfs": str(inroot), "runs": results,
              "comparisons": compare(results, canonical), "scope": "model-free diagnostic; no promotion"}
    with (OUT / "result.json").open("x") as f:
        json.dump(result, f, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    record_run(OUT, run_id=OUT.name, producer=str(Path(__file__).resolve()), observations=observations)


if __name__ == "__main__":
    run()
