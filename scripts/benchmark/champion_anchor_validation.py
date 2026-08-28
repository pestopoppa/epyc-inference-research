#!/usr/bin/env python3
"""CH-4: validate the composed champion against the sealed production anchor.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
`champion.py` records `last_t0/last_t1/last_t2` by reading tier events out of a
campaign journal; those events are produced by a full AutoKernel campaign, not by a
standalone tool. This script therefore does **not** emit formal T0/T1/T2 events and
must not be described as "the champion passed T0/T1/T2". It runs the MEASUREMENTS
those tiers stand for, against the same anchor, so the composed champion is not
adopted on assumption:

  * T0-equivalent  -- correctness: `test-backend-ops` on the champion build.
  * T1/T2-equivalent -- no regression vs the frozen production anchor on the
    production model, prefill and decode.

CH-4's actual subject is MoE-Spec, which is in the champion source but has never
earned anything. Two facts make the shape of this test specific:

  1. `--moe-spec-budget` defaults to 0 and `llama-graph.cpp` guards on
     `moe_spec_budget > 0`, so the champion carries the CAPABILITY and not the
     behaviour. The default-path arm must therefore be INDISTINGUISHABLE from
     production; a difference there is a regression, not a feature.
  2. Its measured evidence was n=3 where MEASUREMENT_POLICY requires >=5 for a >=5%
     claim, and the 5-rep confirm was declined. So the budget>0 arm here is a
     directional observation, not a claim.

DESIGN, carrying forward what the CH-6 re-check taught: arms ALTERNATE rather than
running in blocks (block-sequential design confounds drift with arm and is how the
+46.9% ubatch artifact happened), full sample vectors are printed rather than medians
alone (the failure mode is bimodality, which a median hides), and any arm whose
max/min exceeds 1.3x is flagged.

GPU residency is proven per run, never inferred: llama.cpp dlopens libggml-hip.so, so
neither the binary nor `ldd` can tell you a HIP run happened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

MODEL = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
VRAM_SYSFS = Path("/sys/class/drm/card2/device/mem_info_vram_used")
VRAM_FLOOR = 8 * 1024**3


def read_vram() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


def bench(build_bin: Path, *, pp: int, tg: int, env_extra: dict[str, str] | None = None
          ) -> float:
    argv = ["taskset", "-c", "184-191", "numactl", "--interleave=all",
            str(build_bin / "llama-bench"), "-m", str(MODEL),
            "-p", str(pp), "-n", str(tg), "-r", "1", "-ngl", "99", "-fa", "1",
            "-o", "json"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{build_bin}:/opt/rocm/lib"
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env.update(env_extra or {})
    out = subprocess.run(argv, capture_output=True, text=True, timeout=3600, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"llama-bench rc={out.returncode}: {out.stderr[-300:]}")
    rows = json.loads(out.stdout)
    key = "n_prompt" if pp else "n_gen"
    want = pp or tg
    for r in rows:
        if int(r[key]) == want:
            return float(r["avg_ts"])
    raise RuntimeError(f"no row for {key}={want}")


def flag(samples: list[float]) -> str:
    if not samples or min(samples) <= 0:
        return ""
    ratio = max(samples) / min(samples)
    return f"   ** SUSPECT max/min={ratio:.2f}x" if ratio > 1.3 else ""


def report(label: str, anchor: list[float], cand: list[float]) -> dict:
    ma, mc = st.median(anchor), st.median(cand)
    delta = (mc / ma - 1.0) * 100.0
    print(f"\n  {label}")
    print(f"    production anchor  median {ma:9.2f}  {sorted(round(v,1) for v in anchor)}{flag(anchor)}")
    print(f"    champion           median {mc:9.2f}  {sorted(round(v,1) for v in cand)}{flag(cand)}")
    print(f"    champion vs anchor: {delta:+.2f}%")
    return {"surface": label, "anchor_median": ma, "champion_median": mc,
            "delta_pct": delta, "anchor_samples": anchor, "champion_samples": cand}


def main() -> int:
    ap = argparse.ArgumentParser(description="CH-4 champion vs production anchor")
    ap.add_argument("--anchor-bin", required=True, type=Path)
    ap.add_argument("--champion-bin", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--moe-spec-budget", type=int, default=0,
                    help="if >0, adds a third directional arm with MoE-Spec ENABLED")
    args = ap.parse_args()

    for b in (args.anchor_bin, args.champion_bin):
        if not (b / "llama-bench").is_file():
            print(f"REFUSED: no llama-bench in {b}", file=sys.stderr)
            return 2
    args.out.mkdir(parents=True, exist_ok=True)

    results, moe_rows = [], []
    for label, pp, tg in (("Qwen3.8-27B-Q8_0 pp512 (prefill)", 512, 0),
                          ("Qwen3.8-27B-Q8_0 tg128 (decode)", 0, 128)):
        anchor, cand, moe = [], [], []
        print(f"\n[{time.strftime('%H:%M:%S')}] {label} — {args.reps} alternating pairs")
        for i in range(args.reps):
            for sink, b, extra in ((anchor, args.anchor_bin, None),
                                   (cand, args.champion_bin, None)):
                try:
                    sink.append(bench(b, pp=pp, tg=tg, env_extra=extra))
                except Exception as exc:  # noqa: BLE001
                    print(f"    rep {i}: FAILED {exc}")
            if args.moe_spec_budget > 0:
                try:
                    moe.append(bench(args.champion_bin, pp=pp, tg=tg, env_extra={
                        "LLAMA_ARG_MOE_SPEC_BUDGET": str(args.moe_spec_budget)}))
                except Exception as exc:  # noqa: BLE001
                    print(f"    rep {i} moe-spec: FAILED {exc}")
            if read_vram() < 0:
                print("    (VRAM sysfs unreadable — residency not provable this rep)")
        if anchor and cand:
            results.append(report(label, anchor, cand))
        if moe:
            mm, mc = st.median(moe), st.median(cand)
            print(f"    MoE-Spec budget={args.moe_spec_budget}: median {mm:9.2f} "
                  f"({(mm/mc-1)*100:+.2f}% vs champion default) {sorted(round(v,1) for v in moe)}")
            moe_rows.append({"surface": label, "budget": args.moe_spec_budget,
                             "median": mm, "vs_champion_default_pct": (mm/mc-1)*100,
                             "samples": moe})

    payload = {"results": results, "moe_spec": moe_rows,
               "anchor_bin": str(args.anchor_bin), "champion_bin": str(args.champion_bin),
               "note": "T1/T2-EQUIVALENT MEASUREMENTS, not formal campaign tier events"}
    (args.out / "champion_anchor_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'champion_anchor_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
