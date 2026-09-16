#!/usr/bin/env python3
"""ERNIE-Image-Turbo MI210 (ROCm/HIP) rebench + same-binary f32-patch A/B.

Recipe: `/mnt/raid0/llm/worktrees/sub-gpu-prep-stable-diffusion.cpp/RUNNER_RECIPE.md` (§3-§5), executed
as written: build `build-rocm-gpuprep-20260916` (sd-server sha256 5335caad...), port 18190,
`taskset -c 184-191 ... -t 8 --diffusion-fa --diffusion-conv-direct --vae-conv-direct`, env
`HIP_PATH/ROCM_PATH=/opt/rocm`, `LD_LIBRARY_PATH=<build>/bin:/opt/rocm/lib`, `HIP_VISIBLE_DEVICES=0`,
`OMP_NUM_THREADS=1`. Phases: A warm-up 512^2 (discarded); B patch ON 768/896/960/1024 x3;
D patch ON 832x1248 x3; C patch OFF (`SD_ERNIE_ROCM_F32=0`, fresh server, 512^2 warm-up discarded)
896/1024 x3. Phase E (content audit) is out of scope.

Per request: W x H, HTTP status, wall time, PNG bytes, blankness (pixel mean/stddev/fraction>250,
computed with PIL from the orchestrator venv), and the server log's sampling/decode/generate
timings. Residency: VRAM + KFD sampled across each server lifetime; the server pid must appear in
/sys/class/kfd/kfd/proc during generation; the log must show `found 1 ROCm devices` and
`RAM 0.00MB` params. The production CPU sd-server (pid 910274, :8190) is never touched.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
from autokernel.loop import claim, residency  # noqa: E402

B = Path("/mnt/raid0/llm/worktrees/sub-gpu-prep-stable-diffusion.cpp/build-rocm-gpuprep-20260916")
SHA = "5335caad06f0891e7d27520e495bc1e203fdce71e214c6f857d41c271fee4862"
MD = Path("/mnt/raid0/llm/models/diffusion")
PORT = 18190
PIL_PY = "/workspace/repos/epyc-orchestrator/.venv/bin/python3"
PROD_PID = 910274
BLANK = r"""
import sys, json
from PIL import Image, ImageStat
im = Image.open(sys.argv[1]).convert('L')
st = ImageStat.Stat(im)
hist = im.histogram()
print(json.dumps({'mean': st.mean[0], 'stddev': st.stddev[0],
                  'frac_gt250': sum(hist[251:]) / (im.width * im.height), 'size': [im.width, im.height]}))
"""


def launch(out: Path, label: str, f32_off: bool):
    d = out / label
    d.mkdir(parents=True, exist_ok=True)
    argv = ["taskset", "-c", "184-191", str(B / "bin/sd-server"),
            "--diffusion-model", str(MD / "ernie-image-turbo-gguf/ernie-image-turbo-Q8_0.gguf"),
            "--vae", str(MD / "ernie-image-turbo-comfy/vae/flux2-vae.safetensors"),
            "--llm", str(MD / "ernie-image-turbo-comfy/text_encoders/ministral-3-3b.safetensors"),
            "-t", "8", "--diffusion-fa", "--diffusion-conv-direct", "--vae-conv-direct",
            "--listen-ip", "127.0.0.1", "--listen-port", str(PORT)]
    env = dict(os.environ)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env.pop("SD_ERNIE_ROCM_F32", None)
    env.update(HIP_PATH="/opt/rocm", ROCM_PATH="/opt/rocm", LD_LIBRARY_PATH=f"{B}/bin:/opt/rocm/lib",
               HIP_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    if f32_off:
        env["SD_ERNIE_ROCM_F32"] = "0"
    (d / "server.argv").write_text(" ".join(argv) + f"\nSD_ERNIE_ROCM_F32={env.get('SD_ERNIE_ROCM_F32', '<unset: patch ON>')}\n")
    err = (d / "server.stderr").open("wb")
    proc = subprocess.Popen(argv, stdout=(d / "server.stdout").open("wb"), stderr=err, env=env)
    (d / "server.pid").write_text(str(proc.pid))
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"sd-server exited rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/sdapi/v1/samplers", timeout=3) as r:
                if r.status == 200:
                    return proc, d
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("sd-server not ready in 300s")


def stop(proc) -> str:
    if proc.pid == PROD_PID:
        raise RuntimeError("refusing to signal the production sd-server")
    if proc.poll() is not None:
        return "exited"
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(30)
        return "terminated"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
        return "killed"


def txt2img(d: Path, tag: str, w: int, h: int, pid: int) -> dict:
    body = {"prompt": "a lovely cat, colorful, detailed, studio lighting",
            "negative_prompt": "blurry, low quality, blank white image",
            "width": w, "height": h, "steps": 8, "cfg_scale": 1.0, "seed": 2026072732, "batch_size": 1}
    log = d / "server.stdout"
    elog = d / "server.stderr"
    off_o, off_e = log.stat().st_size, elog.stat().st_size
    kfd_seen = []
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/sdapi/v1/txt2img", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    import threading
    stop_evt = threading.Event()

    def watch():
        while not stop_evt.is_set():
            kfd_seen.append(Path(f"/sys/class/kfd/kfd/proc/{pid}").exists())
            stop_evt.wait(1.0)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    status, data = None, b""
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            status, data = r.status, r.read()
    except urllib.error.HTTPError as exc:
        status, data = exc.code, exc.read()
    finally:
        stop_evt.set()
        th.join(5)
    wall = time.time() - t0
    row = {"tag": tag, "w": w, "h": h, "http": status, "wall_s": round(wall, 3),
           "kfd_pid_listed_samples": sum(kfd_seen), "kfd_samples": len(kfd_seen)}
    try:
        img = base64.b64decode(json.loads(data)["images"][0])
        png = d / f"{tag}.png"
        png.write_bytes(img)
        row["png_bytes"] = len(img)
        row["png_sha256"] = hashlib.sha256(img).hexdigest()
        out = subprocess.run([PIL_PY, "-c", BLANK, str(png)], capture_output=True, text=True, timeout=60)
        row["blankness"] = json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr[-300:]}
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    time.sleep(0.5)
    seg = log.read_bytes()[off_o:].decode(errors="replace") + elog.read_bytes()[off_e:].decode(errors="replace")
    for key, pat in (("sampling_s", r"sampling completed, taking ([0-9.]+)s"),
                     ("decode_s", r"decode_first_stage completed, taking ([0-9.]+)s"),
                     ("generate_s", r"generate_image completed in ([0-9.]+)s")):
        m = re.findall(pat, seg)
        row[key] = float(m[-1]) if m else None
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256((B / "bin/sd-server").read_bytes()).hexdigest()
    if sha != SHA:
        print(f"REFUSED: sd-server sha256 {sha} != {SHA}", file=sys.stderr)
        return 64
    lk = subprocess.run(["bash", "/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh",
                         str(B / "bin/sd-server"), str(B)], capture_output=True, text=True,
                        env=dict(os.environ, LD_LIBRARY_PATH=f"{B}/bin:/opt/rocm/lib"))
    (args.out / "linkage.txt").write_text(lk.stdout + lk.stderr)
    if lk.returncode != 0:
        print(f"REFUSED: linkage exit {lk.returncode}", file=sys.stderr)
        return 64
    free_gib = (65520 * 2**20 - residency.vram_bytes()) / 2**30
    print(f"sd-server {sha[:12]} linkage PASS; VRAM free ~{free_gib:.1f} GiB", flush=True)
    if free_gib < 30:
        print("REFUSED: <30 GiB VRAM free", file=sys.stderr)
        return 75
    rows_path = args.out / "requests.jsonl"
    servers = []
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        for label, f32_off, cases in (
            ("patch_on", False, [("A_warm_512", 512, 512, 1)]
             + [(f"B_{s}", s, s, args.repeats) for s in (768, 896, 960, 1024)]
             + [("D_832x1248", 832, 1248, args.repeats)]),
            ("patch_off", True, [("C_warm_512", 512, 512, 1)]
             + [(f"C_{s}", s, s, args.repeats) for s in (896, 1024)]),
        ):
            sampler = residency.Sampler(interval=1.0)
            with sampler:
                proc, d = launch(args.out, label, f32_off)
                info = {"label": label, "pid": proc.pid, "f32_off": f32_off}
                try:
                    for tag, w, h, reps in cases:
                        for rep in range(reps):
                            row = txt2img(d, f"{tag}_r{rep}", w, h, proc.pid)
                            row.update(server=label, f32_off=f32_off, rep=rep, vram_now=residency.vram_bytes())
                            with rows_path.open("a") as fh:
                                fh.write(json.dumps(row) + "\n")
                            b = row.get("blankness") or {}
                            print(f"{label} {tag} r{rep}: http {row['http']} {row['wall_s']}s "
                                  f"png {row.get('png_bytes')} sd {b.get('stddev')} gen {row.get('generate_s')}", flush=True)
                finally:
                    info["teardown"] = stop(proc)
            text = (d / "server.stdout").read_text(errors="replace") + (d / "server.stderr").read_text(errors="replace")
            info["log_rocm_device"] = bool(re.search(r"found 1 ROCm devices", text))
            info["log_params_lines"] = re.findall(r"total params memory size = .*", text)[:4]
            info["residency"] = sampler.proof
            servers.append(info)
            print(json.dumps(info), flush=True)
    (args.out / "servers.json").write_text(json.dumps(servers, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
