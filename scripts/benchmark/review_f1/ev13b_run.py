#!/usr/bin/env python3
"""EV-13b run leg on the MI210: reader -> judge (cross-family) -> swap judge -> score.

Models are served ONE AT A TIME (sequential by construction), each by a fresh llama-server owned
by this process, with GPU residency sampled across its lifetime and the GPU claim held throughout.

  reader  Qwen3.8-27B-Q8_0 -- the production `architect_general` argv resolved from the orchestrator's
          own builder (v9 build-hip, MTP n-max 8, q8_0 KV, terse template, reasoning off), with
          declared deltas: port, scratch --slot-save-path, and `-np 1 -c 131072` (the largest
          Augment-v1 diff is ~250 KB, ~70k tokens; production's 32k/slot would refuse it).
          harness.py, 50 PRs x 3 runs, temp 0.6, seed 42+i, chat_template_kwargs enable_thinking=false.
  judge   gemma-4-26B-A4B-it-ORIG-Q4_K_M (production worker_general file) -- cross-family.
  swap    Qwen3.6-35B-A3B-MTP-Q8_0 -- same persisted reader findings re-judged (EV-6 delta).
          Both judges: v9 build-hip, -ngl 99 -fa on -np 4 -c 32768 f16 KV --jinja, host threads
          184-191; the Qwen judge keeps its MTP self-draft (n-max 4, the serving recipe's value).
          Each judge is calibrated first (positive/negative controls >= 95%); an invalid judge's
          scored leg is still recorded but marked invalid.
  beliefs Sidecars are written by epyc-root's `review_f1_capture.py`, resolved from `EPYC_ROOT`. The driver
          refuses to start without it (exit 2, before the GPU claim). A failed capture is printed,
          recorded under `belief_capture` in run_record.json, and makes the exit code 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "scripts/kernel_rnd"))
sys.path.insert(0, str(WT / "scripts/benchmark"))
from autokernel.loop import claim, residency  # noqa: E402
import belief_capture  # noqa: E402

ORCH = Path("/workspace/repos/epyc-orchestrator")
V9 = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin")
EXT = WT / "data/external/review_f1/augment_v1"
GOLDEN = EXT / "golden_set.json"
MANIFEST = WT / "data/review_f1/augment_v1_manifest.json"
READER = ("Qwen3.8-27B", "Q8_0")
JUDGES = [
    ("gemma-4-26B-A4B-it-ORIG", "Q4_K_M", "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf", []),
    ("Qwen3.6-35B-A3B-MTP", "Q8_0", "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
     ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]),
]
RPORT, JPORT = 18391, 18392
RESOLVE = r"""
import json, sys
sys.path.insert(0, 'scripts/server'); sys.path.insert(0, '.')
import orchestrator_stack as o
from stack_numa import _numa_prefix
reg = o.RegistryLoader()
rc = reg.get_role_config('architect_general') if hasattr(reg, 'get_role_config') else reg.roles['architect_general']
print(json.dumps({'prefix': _numa_prefix('architect_general'),
                  'cmd': o.build_server_command(rc, %d, prepare_runtime_dirs=False)}))
"""


def reader_argv(slot_dir: Path) -> tuple[list[str], dict]:
    out = subprocess.run([str(ORCH / ".venv/bin/python3"), "-c", RESOLVE % RPORT], cwd=str(ORCH),
                         capture_output=True, text=True, check=True, timeout=120)
    spec = json.loads(out.stdout.strip().splitlines()[-1])
    cmd = list(spec["cmd"])
    prod = list(cmd)
    for flag, val in (("--slot-save-path", str(slot_dir)), ("-np", "1"), ("-c", "131072")):
        cmd[cmd.index(flag) + 1] = val
    return spec["prefix"] + cmd, {"resolved_production_cmd": prod, "resolved_prefix": spec["prefix"],
                                  "deltas": {"--port": RPORT, "--slot-save-path": str(slot_dir),
                                             "-np": 1, "-c": 131072}}


def judge_argv(model: str, extra: list[str]) -> list[str]:
    return ["taskset", "-c", "184-191", str(V9 / "llama-server"), "-m", model,
            "--host", "127.0.0.1", "--port", str(JPORT), "-np", "4", "-c", "32768", "-t", "8",
            "-b", "2048", "-ub", "2048", "-ngl", "99", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
            "--jinja", "--reasoning", "off", "--device", "ROCm0", "--metrics", *extra]


class Server:
    def __init__(self, argv, port, log: Path):
        self.argv, self.port, self.log = argv, port, log

    def __enter__(self):
        env = dict(os.environ)
        env.pop("HSA_OVERRIDE_GFX_VERSION", None)
        env.pop("GGML_NOHUGEPAGE_PROCESS", None)
        env["LD_LIBRARY_PATH"] = f"{V9}:/opt/rocm/lib"
        self.sampler = residency.Sampler(interval=1.0).__enter__()
        self.err = self.log.open("ab")
        self.proc = subprocess.Popen(self.argv, stdout=subprocess.DEVNULL, stderr=self.err, env=env)
        print(f"server pid {self.proc.pid}: {' '.join(self.argv)}", flush=True)
        deadline = time.time() + 900
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError("server not healthy")
            time.sleep(3)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/props", timeout=10) as r:
            self.props = json.loads(r.read())
        vram = residency.vram_bytes()
        print(f"  healthy: model {self.props.get('model_path')} VRAM {vram / 2**30:.1f}G "
              f"template {len(self.props.get('chat_template') or '')}B", flush=True)
        if vram < 8 * 2**30:
            raise RuntimeError("not GPU-resident")
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        self.err.close()
        self.sampler.__exit__()
        self.record = {"pid": self.proc.pid, "rc": self.proc.returncode, "argv": self.argv,
                       "props_model_path": self.props.get("model_path") if hasattr(self, "props") else None,
                       "residency": self.sampler.proof}
        print(f"  stopped pid {self.proc.pid} rc={self.proc.returncode}", flush=True)
        return False


def py(args: list[str], log: Path) -> int:
    with log.open("ab") as fh:
        return subprocess.run([sys.executable, *args], cwd=str(WT), stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    # VB-RUNNER-PATHS: resolve the root capture writer from EPYC_ROOT before any server starts.
    try:
        cap = belief_capture.preflight("review_f1_capture")["review_f1_capture"]
    except belief_capture.CaptureUnavailable as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    captures = belief_capture.CaptureLog()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rq = f"{READER[0]}__{READER[1]}"
    reader_root = out / rq
    record = {"schema": "epyc.ev13b.run_leg.v1", "golden": str(GOLDEN),
              "golden_sha256": hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
              "v9_llama_server_sha256": hashlib.sha256((V9 / "llama-server").read_bytes()).hexdigest(),
              "servers": {}, "steps": {}}
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        done = reader_root.exists() and all(
            len(json.loads(p.read_text()).get("runs", [])) >= args.runs
            for p in [reader_root / f"{c['case_id']}.json" for c in json.loads(GOLDEN.read_text())["cases"]]
            if p.exists()) and len(list(reader_root.glob("*__pr-*.json"))) == 50
        if not done:
            (out / "slots").mkdir(exist_ok=True)
            argv, prov = reader_argv(out / "slots")
            record["reader_provenance"] = prov
            srv = Server(argv, RPORT, out / "reader.server.log")
            with srv:
                rc = py(["scripts/benchmark/review_f1/harness.py", "--golden", str(GOLDEN),
                         "--context-dir", str(EXT), "--server-url", f"http://127.0.0.1:{RPORT}",
                         "--model", READER[0], "--quant", READER[1],
                         "--judge-model", JUDGES[0][0], "--judge-quant", JUDGES[0][1],
                         "--runs", str(args.runs), "--seed", "42", "--out", str(out), "--resume"],
                        out / "reader.harness.log")
                record["steps"]["reader_harness_rc"] = rc
            record["servers"]["reader"] = srv.record
            if rc != 0:
                (out / "run_record.json").write_text(json.dumps(record, indent=2))
                return rc
        for name, quant, model, extra in JUDGES:
            key = f"{name}__{quant}"
            srv = Server(judge_argv(model, extra), JPORT, out / f"judge.{key}.server.log")
            with srv:
                common = ["--golden", str(GOLDEN), "--judge-model", name, "--judge-quant", quant,
                          "--judge-url", f"http://127.0.0.1:{JPORT}", "--reader-model", READER[0]]
                cal = out / f"judge_calibration.{key}.json"
                if not cal.exists():
                    record["steps"][f"calibrate_{key}"] = py(
                        ["scripts/benchmark/review_f1/semantic_judge.py", "calibrate", *common, "--out", str(cal)],
                        out / f"judge.{key}.log")
                record["steps"][f"judge_{key}"] = py(
                    ["scripts/benchmark/review_f1/semantic_judge.py", "judge", *common,
                     "--reader-root", str(reader_root), "--context-dir", str(EXT)], out / f"judge.{key}.log")
            record["servers"][key] = srv.record
    for i, (name, quant, _, _) in enumerate(JUDGES):
        other = JUDGES[1 - i]
        py(["scripts/benchmark/review_f1/semantic_judge.py", "score", "--golden", str(GOLDEN),
            "--judge-model", name, "--judge-quant", quant, "--reader-root", str(reader_root),
            "--reader-model", READER[0], "--manifest", str(MANIFEST),
            "--swap-judge-key", f"{other[0]}__{other[1]}"], out / "score.log")
        summ = reader_root / f"_summary.semantic.{name}__{quant}.json"
        # Belief kernel write side (VB-REVIEW-F1): emitted at write time, never backfilled.
        def _capture(summ=summ, name=name, quant=quant):
            cal = json.loads((out / f"judge_calibration.{name}__{quant}.json").read_text())
            return cap.write_belief_measurements(
                summ, run_id=f"ev13b-{time.strftime('%Y%m%d')}-{name}",
                producer="epyc-inference-research/scripts/benchmark/review_f1/ev13b_run.py",
                served={"reader": record["servers"].get("reader", {}).get("props_model_path"),
                        "judge_calibration_valid": cal.get("valid"),
                        "judge_positive_rate": cal.get("positive_rate"),
                        "judge_negative_rate": cal.get("negative_rate"),
                        "v9_llama_server_sha256": record["v9_llama_server_sha256"]})
        record["steps"][f"beliefs_{name}"] = captures.run(f"beliefs_{name}", _capture)
    record["belief_capture"] = captures.as_record()
    (out / "run_record.json").write_text(json.dumps(record, indent=2))
    print("done", flush=True)
    return captures.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
