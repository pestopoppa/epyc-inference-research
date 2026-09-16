#!/usr/bin/env python3
"""fable5 window-2 §5 measurement #1, MoE half -- quantized MoE batched `-npl` sweep on the MI210.

Spec: `handoffs/active/fable5-window2-findings-05-intake-sweep-and-roofline.md` (box "§5 measurement #1,
MoE half", sub-gpu-prep note 2026-09-16), executed verbatim:
  `/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-batched-bench` (frozen v9, RUN only; the champion
  has no batched-bench), LD_LIBRARY_PATH = that bin (+/opt/rocm/lib), `taskset -c 184-191`,
  `-ngl 99 -fa on -ctk f16 -ctv f16 -c 16384 -b 2048 -ub 2048 -t 8 -npp 128 -ntg 128
  -npl 1,2,4,8,16,32`; n=3 launches per arm in rotated order; unit = launch; GPU residency sampled
  DURING each launch (autokernel `residency.Sampler`); GPU claim held.
Arms: MoE targets gemma-4-26B-A4B Q4_K_M + Q8_0 and Qwen3.6-35B-A3B-MTP Q8_0 (batched-bench runs
the trunk; the MTP head is not exercised), plus the two DENSE same-family controls the PASS rule
needs: gemma-4-31B Q4_K_M and Qwen3.6-27B Q8_0.

PRE-REGISTERED READING (written to PREREGISTRATION.json before the first launch):
  scaling(arm) = median S_TG(B=32) / median S_TG(B=1)
  PASS if scaling(gemma-26B-A4B Q4_K_M) >= 0.8 x scaling(gemma-31B Q4_K_M) AND the gemma-26B-A4B
       Q4_K/Q8_0 S_TG ratio does not fall from B=1 to B=32;
  FAIL if scaling(gemma-26B-A4B Q4_K_M) < 0.5 x scaling(gemma-31B Q4_K_M)
       -> L3-MoE MUL_MAT_ID/MMQ kernel gains ROI;
  otherwise INTERMEDIATE. The Qwen3.6 MoE/dense Q8_0 pair is reported beside it (same statistic).
Observation grade, speed only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
from autokernel.loop import claim, residency  # noqa: E402

BIN = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin")
M = Path("/mnt/raid0/llm/models")
ARMS = {
    "gemma26a4b_q4km": M / "gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf",
    "gemma26a4b_q8": M / "gemma-4-26B-A4B-it-ORIG-Q8_0.gguf",
    "gemma31_dense_q4km": M / "gemma-4-31B-it-Q4_K_M.gguf",
    "qwen36_35ba3b_q8": M / "Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
    "qwen36_27b_dense_q8": M / "Qwen_Qwen3.6-27B-Q8_0.gguf",
}
NPL = (1, 2, 4, 8, 16, 32)
PREREG = {
    "statistic": "scaling(arm) = median over launches of S_TG(B=32) / median S_TG(B=1)",
    "PASS": "scaling(gemma26a4b_q4km) >= 0.8 * scaling(gemma31_dense_q4km) AND "
            "S_TG(gemma26a4b_q4km)/S_TG(gemma26a4b_q8) at B=32 >= the same ratio at B=1",
    "FAIL": "scaling(gemma26a4b_q4km) < 0.5 * scaling(gemma31_dense_q4km) -> L3-MoE MUL_MAT_ID/MMQ kernel gains ROI",
    "otherwise": "INTERMEDIATE",
    "reported_beside": "scaling(qwen36_35ba3b_q8) / scaling(qwen36_27b_dense_q8)",
    "grade": "observation, speed only",
}


def argv_for(model: Path) -> list[str]:
    return ["taskset", "-c", "184-191", str(BIN / "llama-batched-bench"), "-m", str(model),
            "-ngl", "99", "-fa", "on", "-ctk", "f16", "-ctv", "f16", "-c", "16384",
            "-b", "2048", "-ub", "2048", "-t", "8", "-npp", "128", "-ntg", "128",
            "-npl", ",".join(map(str, NPL)), "--output-format", "jsonl"]


def run_one(out: Path, arm: str, launch: int) -> dict:
    d = out / arm / f"launch{launch}"
    d.mkdir(parents=True, exist_ok=True)
    argv = argv_for(ARMS[arm])
    env = dict(os.environ)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env.pop("GGML_NOHUGEPAGE_PROCESS", None)
    env["LD_LIBRARY_PATH"] = f"{BIN}:/opt/rocm/lib"
    sampler = residency.Sampler(interval=0.25)
    t0 = time.time()
    with sampler, (d / "stdout.jsonl").open("wb") as o, (d / "stderr.log").open("wb") as e:
        proc = subprocess.Popen(argv, stdout=o, stderr=e, env=env)
        pid = proc.pid
        try:
            rc = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(30)
    rows = []
    text = (d / "stdout.jsonl").read_text(errors="replace") + "\n" + (d / "stderr.log").read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    s_tg = {int(r["pl"]): float(r["speed_tg"]) for r in rows if "pl" in r and "speed_tg" in r}
    s_pp = {int(r["pl"]): float(r["speed_pp"]) for r in rows if "pl" in r and "speed_pp" in r}
    proof = sampler.proof
    status = "ok" if rc == 0 and len(s_tg) == len(NPL) else "failed"
    if status == "ok" and not proof["resident"]:
        status = "refused_not_resident"
    return {"arm": arm, "launch": launch, "pid": pid, "rc": rc, "status": status, "argv": argv,
            "seconds": round(time.time() - t0, 1), "S_TG": s_tg, "S_PP": s_pp, "residency": proof,
            "offload_line": next((l for l in (d / "stderr.log").read_text(errors="replace").splitlines()
                                  if "offloaded" in l), None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--launches", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prereg = args.out / "PREREGISTRATION.json"
    if not prereg.exists():
        body = {**PREREG, "arms": {k: str(v) for k, v in ARMS.items()}, "npl": NPL,
                "launches": args.launches, "argv_template": argv_for(Path("<model>")),
                "binary_sha256": hashlib.sha256((BIN / "llama-batched-bench").read_bytes()).hexdigest(),
                "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        prereg.write_text(json.dumps(body, indent=2))
    print(f"prereg sha256 {hashlib.sha256(prereg.read_bytes()).hexdigest()}", flush=True)
    names = list(ARMS)
    plan = []
    for launch in range(args.launches):
        k = launch % len(names)
        plan += [(a, launch) for a in (names[k:] + names[:k])]
    rows_path = args.out / "launches.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["arm"], r["launch"]))
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        for arm, launch in plan:
            if (arm, launch) in done:
                continue
            row = run_one(args.out, arm, launch)
            with rows_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{arm:22s} L{launch} {row['status']:8s} S_TG {row['S_TG']} peakVRAM "
                  f"{row['residency']['peak_vram_bytes'] / 2**30:.1f}G [{row['seconds']}s]", flush=True)
    ok = [json.loads(l) for l in rows_path.read_text().splitlines()]
    ok = [r for r in ok if r["status"] == "ok"]
    summ = {}
    for arm in names:
        rs = [r for r in ok if r["arm"] == arm]
        if not rs:
            continue
        med = {b: statistics.median(r["S_TG"][str(b)] if str(b) in r["S_TG"] else r["S_TG"][b] for r in rs)
               for b in NPL}
        spread = {b: [min(r["S_TG"].get(str(b), r["S_TG"].get(b)) for r in rs),
                      max(r["S_TG"].get(str(b), r["S_TG"].get(b)) for r in rs)] for b in NPL}
        summ[arm] = {"n": len(rs), "median_S_TG": med, "min_max_S_TG": spread,
                     "scaling_32_over_1": med[32] / med[1]}
    verdict = None
    if all(a in summ for a in ("gemma26a4b_q4km", "gemma26a4b_q8", "gemma31_dense_q4km")):
        rel = summ["gemma26a4b_q4km"]["scaling_32_over_1"] / summ["gemma31_dense_q4km"]["scaling_32_over_1"]
        r1 = summ["gemma26a4b_q4km"]["median_S_TG"][1] / summ["gemma26a4b_q8"]["median_S_TG"][1]
        r32 = summ["gemma26a4b_q4km"]["median_S_TG"][32] / summ["gemma26a4b_q8"]["median_S_TG"][32]
        verdict = {"moe_over_dense_scaling": rel, "q4k_over_q8_B1": r1, "q4k_over_q8_B32": r32,
                   "verdict": ("PASS" if rel >= 0.8 and r32 >= r1 else "FAIL" if rel < 0.5 else "INTERMEDIATE")}
    if "qwen36_35ba3b_q8" in summ and "qwen36_27b_dense_q8" in summ:
        (verdict or {}).update(qwen_moe_over_dense_scaling=summ["qwen36_35ba3b_q8"]["scaling_32_over_1"]
                               / summ["qwen36_27b_dense_q8"]["scaling_32_over_1"])
    (args.out / "summary.json").write_text(json.dumps({"arms": summ, "reading": verdict}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
