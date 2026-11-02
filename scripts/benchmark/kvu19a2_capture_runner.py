import hashlib
import json
import os
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from kvu19a2_custody import process_birth, sample_process, sha_file
adapter_root = Path(os.environ["EPYC_VIDYA_ROOT"]).resolve()
sys.path.insert(0, str(adapter_root))
from adapters.kvu19a2_native import record_run

root = Path("/mnt/raid0/llm/tmp/kvu/kvu19a2-preserved-20261102")
out = Path(os.environ["KVU19A2_OUTPUT_DIR"]).resolve()
out.mkdir(exist_ok=False)
archive = Path(os.environ["KVU19A2_SOURCE_ARCHIVE"]).resolve()
archive_sha = sha_file(archive)
if archive_sha != os.environ["KVU19A2_SOURCE_ARCHIVE_SHA256"]:
    raise ValueError("source archive digest mismatch")
exe = root / "fa_harness"
expected_exe = "6bb3b80930695241498cd7e1513ec903886dcb3a8a1301d5df13214d0ee64c8a"
if sha_file(exe) != expected_exe:
    raise ValueError("capture binary mismatch")
build = {"base": "1970ea22b7045312b904604df4d50d06ace6f1c8fd8d493925807905462c07c1",
         "candidate": "ba4c98dacbb3c1b90f360131cd1e50450f022b95fc9c02e833e857127a5a6e8c"}
results = []
observations = []
for arm in ("base", "candidate"):
    libdir = root / arm / "bin"
    hip = (libdir / "libggml-hip.so").resolve()
    if sha_file(hip) != build[arm]:
        raise ValueError("HIP build digest mismatch")
    libhashes = {str(p.resolve()): sha_file(p.resolve()) for p in libdir.glob("libggml*.so.0.16.0")}
    env = dict(os.environ, FA_CASE="78", FA_DUMP_DIR=str(out),
               LD_LIBRARY_PATH=str(libdir) + ":/opt/rocm/lib",
               GGML_BACKEND_PATH=str(hip))
    log = out / f"{arm}.log"
    sample = None
    own_kfd = False
    with log.open("xb") as stream:
        child = subprocess.Popen([str(exe), "ROCm0", "exact"], cwd=out, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 90
        try:
            birth = process_birth(Path("/proc") / str(child.pid))
            while child.poll() is None and time.monotonic() < deadline:
                own_kfd |= (Path("/sys/class/kfd/kfd/proc") / str(child.pid)).exists()
                try:
                    sample = sample_process(child.pid, birth, expected_exe, libhashes)
                except (OSError, ValueError, ProcessLookupError):
                    pass
                time.sleep(.02)
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
                raise TimeoutError("owned child exceeded deadline")
            rc = child.wait()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    dump = out / "out_78.bin"
    target = out / f"{arm}-out_78.bin"
    if dump.exists():
        dump.rename(target)
    results.append({"arm": arm, "pid": child.pid, "birth_ticks": birth, "rc": rc,
                    "own_kfd_seen": own_kfd, "mapping_sample": sample,
                    "log_sha256": sha_file(log), "dump_sha256": sha_file(target) if target.exists() else None})
    observation = {"arm": arm, "case": "FA_CASE=78,D576/512,KV8192,F16", "source_archive_sha256": archive_sha,
                   "executable_sha256": expected_exe, "hip_sha256": build[arm],
                   "mapped_ggml": sample["mapped_ggml"] if sample else {}, "pid": child.pid,
                   "birth_ticks": birth, "own_kfd_seen": own_kfd, "mapping_sample_present": bool(sample),
                   "gpu_vram_observed": None, "source_build_binding": "unproven"}
    if rc == 0 and target.exists():
        raw = target.read_bytes()
        if not raw or len(raw) % 4:
            raise ValueError("malformed native output")
        observation.update(observed=True, raw_path=str(target), raw_sha256=sha_file(target),
                           nan_elements=sum(math.isnan(x[0]) for x in struct.iter_unpack("<f", raw)))
    else:
        observation.update(observed=False, nan_elements=None, missing_reason="native output absent or nonzero exit")
    observations.append(observation)
with (out / "result.json").open("x") as stream:
    json.dump({"scope": "one model-free capture smoke per arm; no repeatability verdict", "runs": results}, stream)
record_run(out, run_id=out.name, producer=str(Path(__file__).resolve()), observations=observations)
