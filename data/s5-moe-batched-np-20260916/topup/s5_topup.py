#!/usr/bin/env python3
"""fable5 window-2 §5 #1 (MoE half) -- the ONE pre-registered top-up, Q-B only.

Prereg (PREREGISTRATION.json, frozen 2026-09-16T14:01:31Z): "one pre-committed top-up (+5 at B in {1,32})
only on INCONCLUSIVE". First pass (research 6cbdd856): Q-A GO, Q-B INCONCLUSIVE on both pairs, so the
top-up covers the four Q-B arms (G4, Dg4, Q8m, Dq8) at -npl 1,32, launches 5..9, rotated order.

It reuses the first-pass driver verbatim (s5_moe_batched_np_sweep.run_one / argv_for / claim / residency).
The only change is the module's NPL = (1, 32). Pooling applies the driver's own Q-B statistic to the pooled
n=10 per arm (first-pass K(32) + top-up K(32), unit = launch):
M2 = median K_moe(32) / median K_dense(32); delta_eff = max(8%, max p95_dev of the two K32 sets);
INVALID if p95_dev > 16%; PASS >= 0.80; FAIL < 0.50; INCONCLUSIVE otherwise or within delta_eff of an edge.
INCONCLUSIVE after the top-up is a BOUNDED NULL (thresholds doc §6), and the table is final.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

DRIVER = Path("/mnt/raid0/llm/worktrees/sub-gpu-runner-epyc-inference-research/scripts/benchmark/"
              "s5_moe_batched_np_sweep.py")
FIRST = Path("/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/s5")
OUT = Path("/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/s5-topup")
PAIRS = (("gemma26a4b_q4km", "gemma31_dense_q4km"), ("qwen36_35ba3b_q8", "qwen36_27b_dense_q8"))
LAUNCHES = range(5, 10)

spec = importlib.util.spec_from_file_location("s5drv", DRIVER)
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
drv.NPL = (1, 32)
from autokernel.loop.serving import _spread  # noqa: E402  (path set up by the driver import)


def k32(rows, arm):
    return {r["launch"]: float(r["S_TG"]["32"]) / float(r["S_TG"]["1"]) for r in rows if r["arm"] == arm}


def qb(ka: list[float], kb: list[float]) -> dict:
    m2 = statistics.median(ka) / statistics.median(kb)
    spa, spb = _spread(ka)["p95_dev_pct"], _spread(kb)["p95_dev_pct"]
    sp = max(spa, spb)
    d = max(8.0, sp) / 100
    near = abs(m2 - 0.80) <= d * 0.80 or abs(m2 - 0.50) <= d * 0.50
    v = ("INVALID" if sp > 16 else "INCONCLUSIVE" if near else
         "PASS" if m2 >= 0.80 else "FAIL" if m2 < 0.50 else "INCONCLUSIVE")
    return {"M2_32": m2, "n": [len(ka), len(kb)], "K32_moe_median": statistics.median(ka),
            "K32_dense_median": statistics.median(kb), "K32_moe_minmax": [min(ka), max(ka)],
            "K32_dense_minmax": [min(kb), max(kb)], "K32_p95_dev_pct": [spa, spb], "delta_eff": d,
            "verdict": v}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((FIRST / "PREREGISTRATION.json").read_text())
    sha = hashlib.sha256((drv.BIN / "llama-batched-bench").read_bytes()).hexdigest()
    if sha != prereg["binary_sha256"]:
        print(f"REFUSED: binary sha {sha} != prereg {prereg['binary_sha256']}", flush=True)
        return 3
    arms = [a for p in PAIRS for a in p]
    units = [(a,) for a in arms]
    plan = []
    for launch in LAUNCHES:
        k = launch % len(units)
        plan += [(u[0], launch) for u in units[k:] + units[:k]]
    rows_path = OUT / "launches.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["arm"], r["launch"]))
    meta = {"prereg_sha256": hashlib.sha256((FIRST / "PREREGISTRATION.json").read_bytes()).hexdigest(),
            "binary_sha256": sha, "npl": list(drv.NPL), "arms": arms, "launches": list(LAUNCHES),
            "plan": plan, "driver": str(DRIVER),
            "driver_sha256": hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
            "argv_template": drv.argv_for(Path("<model>")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "TOPUP_META.json").write_text(json.dumps(meta, indent=2))
    print(f"prereg sha256 {meta['prereg_sha256']} binary {sha}", flush=True)
    # The claim is non-blocking by design; the chain already serializes the queue, so a refusal here is a
    # short overlap (e.g. a collector). Retry for up to 2 h, then refuse.
    for attempt in range(120):
        try:
            with drv.claim.hold() as receipt:
                print(f"GPU claim held: {dict(receipt)}", flush=True)
                for arm, launch in plan:
                    if (arm, launch) in done:
                        continue
                    row = drv.run_one(OUT, arm, launch)
                    with rows_path.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(f"{arm:22s} L{launch} {row['status']:8s} S_TG {row['S_TG']} peakVRAM "
                          f"{row['residency']['peak_vram_bytes'] / 2**30:.1f}G "
                          f"kfd {row['residency']['peak_kfd_processes']} [{row['seconds']}s]", flush=True)
            break
        except drv.claim.ClaimRefused as exc:
            print(f"claim refused (attempt {attempt}): {exc}", flush=True)
            time.sleep(60)
    else:
        print("REFUSED: GPU claim not acquired in 2 h", flush=True)
        return 4

    top = [json.loads(l) for l in rows_path.read_text().splitlines()]
    top = [r for r in top if r["status"] == "ok"]
    first = [json.loads(l) for l in (FIRST / "launches.jsonl").read_text().splitlines()]
    first = [r for r in first if r["status"] == "ok"]
    adm = {}
    for arm in arms:
        rs = [r for r in top if r["arm"] == arm]
        adm[arm] = {"n_ok": len(rs), "all_resident": all(r["residency"]["resident"] for r in rs),
                    "peak_kfd_max": max((r["residency"]["peak_kfd_processes"] for r in rs), default=None),
                    "sclk_min_mhz": min((r["residency"]["sclk_min_mhz"] for r in rs), default=None),
                    "launches_clock_not_flat": sum(not r["residency"]["clock_stable"] for r in rs)}
    reading = {}
    for moe, dense in PAIRS:
        f_a, f_b, t_a, t_b = k32(first, moe), k32(first, dense), k32(top, moe), k32(top, dense)
        reading[f"{moe}:{dense}"] = {
            "pooled": qb(list(f_a.values()) + list(t_a.values()), list(f_b.values()) + list(t_b.values())),
            "first_pass": qb(list(f_a.values()), list(f_b.values())),
            "topup_only": qb(list(t_a.values()), list(t_b.values())) if len(t_a) >= 2 and len(t_b) >= 2 else None,
        }
    per_arm = {arm: {"n": len([r for r in top if r["arm"] == arm]),
                     "median_S_TG": {b: statistics.median(float(r["S_TG"][str(b)]) for r in top if r["arm"] == arm)
                                     for b in (1, 32)}}
               for arm in arms if any(r["arm"] == arm for r in top)}
    summ = {"meta": meta, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "admissibility": adm, "topup_arms": per_arm, "Q-B": reading,
            "rule": "pooled n=10 per arm; INCONCLUSIVE after the top-up = BOUNDED NULL; table final"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ["Q-B"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
