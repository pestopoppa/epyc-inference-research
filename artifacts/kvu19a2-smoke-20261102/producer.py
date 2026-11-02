import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "scripts/benchmark"))
from kvu19a2_custody import process_birth, sample_process, sha_file

root = Path("/mnt/raid0/llm/tmp/kvu/kvu19a2-preserved-20261102")
out = Path("/mnt/raid0/llm/tmp/rtg57-fa4b-accounting-20261102/kvu19a2-smoke")
out.mkdir(exist_ok=False)
exe = root / "fa_harness"
expected_exe = "6bb3b80930695241498cd7e1513ec903886dcb3a8a1301d5df13214d0ee64c8a"
if sha_file(exe) != expected_exe:
    raise ValueError("capture binary mismatch")
build = {"base": "1970ea22b7045312b904604df4d50d06ace6f1c8fd8d493925807905462c07c1",
         "candidate": "ba4c98dacbb3c1b90f360131cd1e50450f022b95fc9c02e833e857127a5a6e8c"}
results = []
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
with (out / "result.json").open("x") as stream:
    json.dump({"scope": "one model-free capture smoke per arm; no repeatability verdict", "runs": results}, stream)
