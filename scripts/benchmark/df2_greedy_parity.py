#!/usr/bin/env python3
"""DF2-6 greedy-parity check, with the controls that make a verdict attributable.

dFlash2 claims losslessness. The naive test -- compare `--spec-type draft-dflash`
against `--spec-type none` at temp 0 -- cannot support a conclusion either way,
because three separate mechanisms produce non-parity here and none of them is
DFlash2:

1. **Batched speculative verification alone diverges.** Upstream #27407 reproduces
   deterministic near-tie divergence with `draft-simple` and no DFlash code present.
   -> the `draft_simple` arm.
2. **Our own MMQ patch splits on batch width.** EPYC-local `a6b4b5263`
   (`ggml/src/ggml-cuda/mmvq.cu:341-344`) deliberately routes Q8_0 to a different
   kernel at `ne11 >= 2`, and its own commit message calls it "numerically-valid
   (not bit-exact)", taken for +17.4% single-stream MTP. A verify batch is
   `ne11 = n_max+1 >= 2`; the non-speculative baseline is `ne11 = 1`. So a parity
   failure on gfx90a may be entirely attributable to a performance patch we
   knowingly accepted. -> `GGML_CUDA_LOG_MMVQ_ROUTE=1` captured on EVERY arm.
   NOTE `--spec-draft-n-max 1` is NOT a safe bit-exact reference: it still produces
   a 2-column batch, so it takes the same MMQ route as the arms under test.
3. **ngram is the discriminator.** Upstream #25618 (five weeks older than #27407,
   never cited by it, reproduced on Vulkan/Metal/ROCm) reports ngram staying
   byte-identical through the same `common_sampler_sample_and_accept_n` path while
   external drafters diverge. If a multi-token verify batch alone were sufficient,
   ngram would break too. ngram byte-identical WHILE draft-simple diverges would
   localise the defect to the external-drafter verify path and overturn #27407's
   headline for our stack. -> the `ngram_simple` arm.

Protocol fixes from DF2-6c, all mandatory:
  (i)   >= 5 prompts. #27407's own draft-simple arm was byte-identical on one
        workload and divergent on another; a third-party run went 0/5 -> 4/5 on the
        same patch. A single-prompt check returns a false clean sheet near 50%.
  (ii)  Fresh process per arm -- one reporter measured 1/5 reused vs 4/5 fresh
        DESPITE cache_prompt=false.
  (iii) `-ctk f16 -ctv f16`. Quantized KV moves greedy output even at --spec-type
        none, so a q8_0-KV arm would need its own baseline first.
  (iv)  Report per-prompt PASS/FAIL with the first-differing GENERATION-TOKEN index,
        never an aggregate verdict.

Negative controls, carried from the k35 dspark method: the baseline arm must show
`draft_n == 0` and every speculative arm `draft_n > 0`. Without them a PASS can be
a false clean sheet produced by speculation silently never engaging.
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

TARGET = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
DFLASH2 = Path("/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf")
QUESTIONS = Path("/workspace/tmp/questions_mtp_ab.json")

PORT = 18098
HOST = "127.0.0.1"
N_MAX = 8
VRAM_RESIDENT_FLOOR = 8 * 1024**3
VRAM_SYSFS = Path("/sys/class/drm/card2/device/mem_info_vram_used")

#: baseline first -- every other arm is compared against it.
ARMS = ("baseline", "dflash2", "draft_simple", "ngram_simple")


class ArmRefused(RuntimeError):
    """An arm could not be run under conditions that make it interpretable."""


def read_vram() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


def arm_argv(build_bin: Path, arm: str, ctx: int,
             pin_host_cores: str | None = None) -> list[str]:
    # Optional because the 2026-08-28 originals ran UNPINNED (every arm equally).
    # The standing GPU recipe pins llama-server host threads to the codified list
    # (`evaluator/recipes.py:gpu_host_cpu_list()` -- 184-191, NOT 88-95); a refresh
    # passes it. Parity verdicts are token comparisons, so pinning cannot change a
    # PASS/FAIL -- it standardises the timing environment only.
    argv = ["taskset", "-c", pin_host_cores] if pin_host_cores else []
    argv += [
        str(build_bin / "llama-server"),
        "-m", str(TARGET),
        "-np", "1", "-c", str(ctx),
        "-t", "8", "-tb", "8", "-b", "2048", "-ub", "2048",
        # DF2-6c-iii: unquantized KV, or non-parity is unattributable.
        "-ctk", "f16", "-ctv", "f16",
        "--device", "ROCm0", "-ngl", "99", "-fa", "on",
        "--host", HOST, "--port", str(PORT), "--metrics", "--slots",
    ]
    if arm == "baseline":
        argv += ["--spec-type", "none"]
    elif arm == "dflash2":
        argv += ["-md", str(DFLASH2), "-ngld", "99",
                 "--spec-type", "draft-dflash", "--spec-draft-n-max", str(N_MAX)]
    elif arm == "draft_simple":
        # Control 1: multi-token verify with NO DFlash code involved.
        argv += ["-md", str(DFLASH2), "-ngld", "99",
                 "--spec-type", "draft-simple", "--spec-draft-n-max", str(N_MAX)]
    elif arm == "ngram_simple":
        # Control 2 (DF2-6b): same verify path, no external drafter at all.
        argv += ["--spec-type", "ngram-simple", "--spec-draft-n-max", str(N_MAX)]
    else:
        raise ArmRefused(f"unknown arm {arm!r}")
    return argv


def wait_ready(proc: subprocess.Popen, log: Path, timeout_s: float = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ArmRefused(f"server exited rc={proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise ArmRefused(f"server not ready in {timeout_s}s; see {log}")


def stop(proc: subprocess.Popen) -> None:
    """Signal only the pid we spawned; verify it is gone. Never a pattern kill."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=60)


def ask(prompt: str, max_tokens: int) -> dict:
    """Greedy, cache-free, raw token ids returned."""
    body = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "typical_p": 1.0, "repeat_penalty": 1.0,
        "seed": 42, "stream": False,
        "cache_prompt": False, "return_tokens": True,
        "samplers": ["top_k", "temperature"],
    }
    req = urllib.request.Request(f"http://{HOST}:{PORT}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def run_arm(build_bin: Path, out: Path, arm: str, prompts: list[dict],
            max_tokens: int, ctx: int,
            pin_host_cores: str | None = None) -> dict:
    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    log = arm_dir / "server.stderr"
    argv = arm_argv(build_bin, arm, ctx, pin_host_cores)
    (arm_dir / "server_command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{build_bin}:/opt/rocm/lib"
    env["GGML_CUDA_LOG_MMVQ_ROUTE"] = "1"   # every arm, per DF2-6 second consequence
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)

    records = []
    with log.open("wb") as errf:
        # DF2-6c-ii: a FRESH process per arm.
        proc = subprocess.Popen(argv, stdout=errf, stderr=subprocess.STDOUT,
                                env=env, cwd=str(build_bin.parent.parent))
        try:
            wait_ready(proc, log)
            vram = read_vram()
            if vram < VRAM_RESIDENT_FLOOR:
                raise ArmRefused(
                    f"{arm}: VRAM {vram} below residency floor -- not GPU-resident")
            for q in prompts:
                d = ask(q["prompt"], max_tokens)
                t = d.get("timings") or {}
                records.append({
                    "id": q["id"],
                    "content": d.get("content", ""),
                    "tokens": d.get("tokens") or [],
                    "content_sha256": hashlib.sha256(
                        (d.get("content") or "").strip().encode()).hexdigest(),
                    "draft_n": t.get("draft_n"),
                    "draft_n_accepted": t.get("draft_n_accepted"),
                    "predicted_n": t.get("predicted_n"),
                })
        finally:
            stop(proc)

    text = log.read_text(errors="replace")
    (arm_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {
        "arm": arm, "records": records, "vram_bytes": vram,
        "mmvq_route_lines": text.count("GGML_CUDA_MUL_MAT_ROUTE"),
    }


def first_diff_index(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing GENERATION token, or None if one is a prefix."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser(description="DF2-6 greedy parity with controls")
    ap.add_argument("--build-bin", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--only-arm", action="append", default=None)
    ap.add_argument("--pin-host-cores", default=None,
                    help="taskset the server to this cpu list (pass the codified "
                         "GPU host-thread list; the 2026-08-28 originals ran "
                         "unpinned)")
    args = ap.parse_args()

    prompts = json.loads(QUESTIONS.read_text())[: args.n_prompts]
    if len(prompts) < 5:
        print("REFUSED: DF2-6c-i requires >= 5 prompts", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    arms = tuple(args.only_arm) if args.only_arm else ARMS
    if "baseline" not in arms:
        print("REFUSED: the baseline arm is the reference; it cannot be skipped",
              file=sys.stderr)
        return 2

    results, refusals = {}, []
    for arm in arms:
        print(f"[{time.strftime('%H:%M:%S')}] arm {arm}", flush=True)
        try:
            results[arm] = run_arm(args.build_bin, args.out, arm, prompts,
                                   args.max_tokens, args.ctx,
                                   args.pin_host_cores)
        except (ArmRefused, Exception) as exc:  # noqa: BLE001 - recorded, not hidden
            print(f"  REFUSED: {exc}", flush=True)
            refusals.append({"arm": arm, "reason": str(exc)})

    if "baseline" not in results:
        print("REFUSED: baseline arm did not complete; no comparison is possible",
              file=sys.stderr)
        (args.out / "refusals.json").write_text(json.dumps(refusals, indent=2))
        return 2

    base = {r["id"]: r for r in results["baseline"]["records"]}
    # Negative control: the baseline must genuinely not speculate.
    base_drafted = [r["id"] for r in results["baseline"]["records"] if (r["draft_n"] or 0) > 0]

    report = {"arms": {}, "refusals": refusals,
              "baseline_negative_control_ok": not base_drafted,
              "baseline_arms_that_drafted": base_drafted}

    for arm, res in results.items():
        if arm == "baseline":
            continue
        rows, drafted_any = [], False
        for rec in res["records"]:
            b = base.get(rec["id"])
            if b is None:
                continue
            same = rec["tokens"] == b["tokens"]
            drafted_any = drafted_any or (rec["draft_n"] or 0) > 0
            rows.append({
                "id": rec["id"],
                "verdict": "PASS" if same else "FAIL",
                "first_diff_generation_token_index":
                    None if same else first_diff_index(b["tokens"], rec["tokens"]),
                "content_sha256_baseline": b["content_sha256"],
                "content_sha256_arm": rec["content_sha256"],
                "draft_n": rec["draft_n"], "draft_n_accepted": rec["draft_n_accepted"],
                "predicted_n_baseline": b["predicted_n"],
                "predicted_n_arm": rec["predicted_n"],
            })
        report["arms"][arm] = {
            # DF2-6c-iv: per-prompt verdicts. The aggregate is reported for
            # convenience but is NOT the verdict.
            "per_prompt": rows,
            "n_pass": sum(1 for r in rows if r["verdict"] == "PASS"),
            "n_fail": sum(1 for r in rows if r["verdict"] == "FAIL"),
            "negative_control_arm_drafted": drafted_any,
            "mmvq_route_lines": res["mmvq_route_lines"],
        }

    (args.out / "parity_report.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
    print("\n=== DF2-6 per-arm parity (per-prompt verdicts in parity_report.json) ===")
    print(f"  baseline negative control (drafted nothing): "
          f"{'OK' if report['baseline_negative_control_ok'] else 'VIOLATED'}")
    for arm, a in report["arms"].items():
        print(f"  {arm:<13} PASS={a['n_pass']:<3} FAIL={a['n_fail']:<3} "
              f"drafted={a['negative_control_arm_drafted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
