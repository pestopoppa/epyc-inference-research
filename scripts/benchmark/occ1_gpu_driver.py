#!/usr/bin/env python3
"""OCC-1 GPU run, executed per `scripts/benchmark/occ1/README.md` (research 0d3ca467) "GPU runner recipe".

Steps, in the recipe's order, under the autokernel GPU claim:
  1. VRAM baseline -> <run>/vram_baseline_bytes and <pilot>/vram_baseline_bytes (before the reader exists)
  2. `launch_reader.sh --port 18431` (execs llama-server; the captured PID is the server)
  3. pilot `run` (3 chunks, --tokenizer '') + `report`; stop if the pilot is VOID
  4. full `run` + `report` (SC85 belief sidecar via EPYC_ROOT = a root origin/main checkout)
  5. teardown of the captured PID only, verified gone
Plans already exist (pilot suite e4cbd4448e91, full suite 261d8ac1eaed) and are not re-planned.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "scripts/kernel_rnd"))
from autokernel.loop import claim  # noqa: E402

OCC = Path("/mnt/raid0/llm/worktrees/sub-gpu-runner-occ1-0d3ca467")
ROOT = Path("/mnt/raid0/llm/worktrees/sub-gpu-runner-root-occ1")
RUN = Path("/mnt/raid0/llm/tmp/occ1-run-20260916")
PILOT = Path("/mnt/raid0/llm/tmp/occ1-run-20260916-pilot")
PORT = 18431
VRAM = Path("/sys/class/drm/card2/device/mem_info_vram_used")
UV = ["uv", "run", "--no-project", "--with", "pillow==12.3.0", "python", "scripts/benchmark/occ1/run_occ1.py"]
UV_REPORT = ["uv", "run", "--no-project", "python", "scripts/benchmark/occ1/run_occ1.py"]


def sh(cmd, log: Path) -> int:
    env = dict(os.environ, OCC1_PORT=str(PORT), EPYC_ROOT=str(ROOT))
    with log.open("ab") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n".encode())
        fh.flush()
        return subprocess.run(cmd, cwd=str(OCC), stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def verdict(d: Path):
    try:
        s = json.loads((d / "summary.json").read_text())
        return s.get("overall"), s.get("void_reasons")
    except Exception as exc:
        return None, str(exc)


def main() -> int:
    log = RUN / "driver.log"
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        base = VRAM.read_text().strip()
        (RUN / "vram_baseline_bytes").write_text(base + "\n")
        (PILOT / "vram_baseline_bytes").write_text(base + "\n")
        print(f"VRAM baseline {int(base) / 2**30:.2f} GiB", flush=True)
        with (RUN / "server.log").open("ab") as slog:
            proc = subprocess.Popen(["scripts/benchmark/occ1/launch_reader.sh", "--port", str(PORT)],
                                    cwd=str(OCC), stdout=slog, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OCC1_PORT=str(PORT)), start_new_session=True)
        pid = proc.pid
        (RUN / "server.pid").write_text(f"{pid}\n")
        print(f"reader pid {pid}", flush=True)
        result = {"pid": pid}
        try:
            deadline = time.time() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"reader exited rc={proc.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("reader not healthy in 900s")
                time.sleep(5)
            print(f"healthy; VRAM {int(VRAM.read_text()) / 2**30:.2f} GiB", flush=True)
            rc = sh(UV + ["run", "--out", str(PILOT), "--limit-chunks", "3", "--tokenizer", "",
                          "--server-pid", str(pid), "--port", str(PORT)], log)
            sh(UV_REPORT + ["report", "--out", str(PILOT)], log)
            result["pilot"] = {"run_rc": rc, "verdict": verdict(PILOT)}
            print(f"pilot: {result['pilot']}", flush=True)
            if result["pilot"]["verdict"][0] in (None, "VOID"):
                print("pilot VOID -> full run NOT started", flush=True)
            else:
                rc = sh(UV + ["run", "--out", str(RUN), "--server-pid", str(pid), "--port", str(PORT)], log)
                sh(UV_REPORT + ["report", "--out", str(RUN)], log)
                result["full"] = {"run_rc": rc, "verdict": verdict(RUN)}
                print(f"full: {result['full']}", flush=True)
        finally:
            if proc.poll() is None:
                os.kill(pid, signal.SIGTERM)
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    os.kill(pid, signal.SIGKILL)
                    proc.wait(30)
            alive = subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0
            result["teardown"] = {"rc": proc.returncode, "still_alive": alive}
            print(f"reader {pid} stopped rc={proc.returncode} alive={alive}", flush=True)
    (RUN / "gpu_driver_result.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
