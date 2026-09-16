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
    "qwen36_27b_dense_q8": M / "Qwen3.6-27B-MTP-Q8_0.gguf",  # MTP twin: both Qwen arms carry the unused head
}
#: Operator approval of the Option-A thresholds (2026-09-16-sub-s5-thresholds.md) must exist before launch.
APPROVAL = Path("/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/s5/APPROVED")
THRESHOLDS_DOC = Path("/workspace/progress/2026-09/2026-09-16-sub-s5-thresholds.md")
NPL = (1, 2, 4, 8, 16, 32)
PREREG = {
    "source": "Option A, /workspace/progress/2026-09/2026-09-16-sub-s5-thresholds.md (sections 3-6)",
    "n": "5 launches/arm; one pre-committed top-up (+5 at B in {1,32}) only on INCONCLUSIVE",
    "delta_eff": "max(8%, measured p95_dev of the verdict statistic); a statistic with p95_dev > 16% is INVALID",
    "K(B)": "S_TG(B)/S_TG(1) within one launch",
    "Q-A": "F(B)=S_TG,G4(B)/S_TG,G8(B) over paired adjacent launches; E=F(32)/F(1) median. "
           "GO if E >= 1-delta_eff; NO-GO if E < 1-2*delta_eff AND F(32) < 1.0; else INCONCLUSIVE",
    "Q-B": "M2=K_moe(32)/K_dense(32) per pair (G4:Dg4, Q8m:Dq8), median. PASS >= 0.80; FAIL < 0.50; "
           "else INCONCLUSIVE; INCONCLUSIVE also when M2 lies within delta_eff of 0.80 or 0.50",
    "secondary": "E and M2 at B=16 reported as a consistency check",
    "grade": "observation proxy, category BASELINE, never a headline",
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
    ap.add_argument("--launches", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not APPROVAL.exists():
        print(f"HELD: operator approval of the pre-registered thresholds not recorded ({APPROVAL})", flush=True)
        return 3
    prereg = args.out / "PREREGISTRATION.json"
    if not prereg.exists():
        body = {**PREREG, "arms": {k: str(v) for k, v in ARMS.items()}, "npl": NPL,
                "launches": args.launches, "argv_template": argv_for(Path("<model>")),
                "binary_sha256": hashlib.sha256((BIN / "llama-batched-bench").read_bytes()).hexdigest(),
                "approval_file": str(APPROVAL), "approval": APPROVAL.read_text().strip(),
                "thresholds_doc_sha256": hashlib.sha256(THRESHOLDS_DOC.read_bytes()).hexdigest(),
                "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        prereg.write_text(json.dumps(body, indent=2))
    print(f"prereg sha256 {hashlib.sha256(prereg.read_bytes()).hexdigest()}", flush=True)
    names = list(ARMS)
    units = [("gemma26a4b_q4km", "gemma26a4b_q8"), ("gemma31_dense_q4km",), ("qwen36_35ba3b_q8",),
             ("qwen36_27b_dense_q8",)]
    plan = []
    for launch in range(args.launches):
        k = launch % len(units)
        for unit in units[k:] + units[:k]:
            pair = unit if launch % 2 == 0 else tuple(reversed(unit))
            plan += [(a, launch) for a in pair]
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
    import sys as _s
    _s.path.insert(0, str(REPO / "scripts/kernel_rnd"))
    from autokernel.loop.serving import _spread

    def per_launch(arm):
        return {r["launch"]: {int(k): v for k, v in r["S_TG"].items()} for r in ok if r["arm"] == arm}

    def med(xs):
        return statistics.median(xs) if xs else None

    reading = {}
    g4, g8 = per_launch("gemma26a4b_q4km"), per_launch("gemma26a4b_q8")
    paired = sorted(set(g4) & set(g8))
    if paired:
        F = {b: [g4[l][b] / g8[l][b] for l in paired] for b in (1, 16, 32)}
        E32 = [F[32][i] / F[1][i] for i in range(len(paired))]
        E16 = [F[16][i] / F[1][i] for i in range(len(paired))]
        sp = _spread(E32)["p95_dev_pct"] if len(E32) >= 2 else None
        d = max(8.0, sp or 0.0) / 100
        e, f32 = med(E32), med(F[32])
        qa = ("INVALID" if sp is not None and sp > 16 else "GO" if e >= 1 - d else
              "NO-GO" if (e < 1 - 2 * d and f32 < 1.0) else "INCONCLUSIVE")
        reading["Q-A"] = {"n_pairs": len(paired), "E32": E32, "E32_median": e, "E32_p95_dev_pct": sp,
                          "delta_eff": d, "F1_median": med(F[1]), "F32_median": f32,
                          "E16_median": med(E16), "verdict": qa}
    for moe, dense in (("gemma26a4b_q4km", "gemma31_dense_q4km"), ("qwen36_35ba3b_q8", "qwen36_27b_dense_q8")):
        a, b = per_launch(moe), per_launch(dense)
        if not a or not b:
            continue
        ka = {bb: [v[bb] / v[1] for v in a.values()] for bb in (16, 32)}
        kb = {bb: [v[bb] / v[1] for v in b.values()] for bb in (16, 32)}
        m2 = med(ka[32]) / med(kb[32])
        m2_16 = med(ka[16]) / med(kb[16])
        spa = _spread(ka[32])["p95_dev_pct"] if len(ka[32]) >= 2 else 0.0
        spb = _spread(kb[32])["p95_dev_pct"] if len(kb[32]) >= 2 else 0.0
        sp = max(spa, spb)
        d = max(8.0, sp) / 100
        near = abs(m2 - 0.80) <= d * 0.80 or abs(m2 - 0.50) <= d * 0.50
        qb = ("INVALID" if sp > 16 else "INCONCLUSIVE" if near else
              "PASS" if m2 >= 0.80 else "FAIL" if m2 < 0.50 else "INCONCLUSIVE")
        reading[f"Q-B {moe}:{dense}"] = {"M2_32": m2, "M2_16": m2_16, "K32_moe_median": med(ka[32]),
                                          "K32_dense_median": med(kb[32]), "K32_p95_dev_pct": [spa, spb],
                                          "delta_eff": d, "verdict": qb}
    verdict = reading
    (args.out / "summary.json").write_text(json.dumps({"arms": summ, "reading": verdict}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
