#!/usr/bin/env python3
"""CH-6: re-check `GGML_HIP_MMQ_MFMA ON vs OFF` as a BUILD-CONFIG A/B on the champion.

Framed deliberately as a build-config comparison, not a champion member: `champion.py`
requires source evidence (`source_tree`, `candidate_source_commit`, a `source_snapshot`
patch digest) for every member, and `discovery_static_registry` accepts no CMake flag
from planner output. A build flag is structurally inexpressible as a champion arm, so the
question this answers is "should the champion's BUILD RECIPE carry it", nothing else.

WHAT THE ORIGINAL SCREEN GOT WRONG, AND WHAT THIS FIXES
-------------------------------------------------------
`ak-gpu-mmq-mfma-screen-20260813-s2` reported +26.6% at n=3 per arm with all three
anchors run consecutively and then all three candidates -- a BLOCK-SEQUENTIAL design in
which any drift over the measurement window is fully confounded with the arm. Its
sibling `ubatch_up` screen shows what that costs: a null arm reported +46.9% because a
bimodal sample's median landed on the fast mode.

So:
  * arms ALTERNATE (A,B,A,B,...) instead of running in blocks -- drift now hits both
    arms equally instead of loading onto one;
  * n=10 per arm instead of 3;
  * the full sample vector is printed, not just the median, because the failure mode
    that produced the +46.9% artifact is BIMODALITY, which a median hides;
  * a spread check flags any arm whose max/min exceeds 1.3x as suspect.

THE OPEN QUESTION IS NOT THE 0.5B NUMBER
-----------------------------------------
+26.6% was measured on `Qwen2.5-Coder-0.5B-Q4_K_M @ pp512, np=1` -- the smallest model
in the fleet, single-stream prefill, which is precisely the regime where MFMA has least
to offer. Countervailing evidence exists at batch>1: the MMQ weight tile is dequantized
once and reused across B columns and MFMA engages on the dequantized tiles, and the
champion's own state file records MMQ forcing INVERTING on MoE workloads at low batch
(B2 -30%, B4 -21%, B8 -10.5%). So this runs BOTH surfaces: the original 0.5B cell as a
replication, and the real production model, where the answer actually matters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import re
import statistics as st
import subprocess
import sys
import time

SMALL = Path("/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/"
             "Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
BIG = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
VRAM_SYSFS = Path("/sys/class/drm/card2/device/mem_info_vram_used")


def read_vram() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


def run_bench(build_bin: Path, model: Path, *, pp: int, tg: int, reps: int = 1
              ) -> dict[str, float]:
    """One llama-bench invocation; returns {test_name: t/s}."""
    argv = ["taskset", "-c", "184-191", "numactl", "--interleave=all",
            str(build_bin / "llama-bench"), "-m", str(model),
            "-p", str(pp), "-n", str(tg), "-r", str(reps),
            "-ngl", "99", "-fa", "1", "-o", "json"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{build_bin}:/opt/rocm/lib"
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    out = subprocess.run(argv, capture_output=True, text=True, timeout=3600, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"llama-bench rc={out.returncode}: {out.stderr[-400:]}")
    try:
        rows = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"llama-bench emitted non-JSON: {out.stdout[:300]}")
    return {r["n_prompt"] and f"pp{r['n_prompt']}" or f"tg{r['n_gen']}":
            float(r["avg_ts"]) for r in rows}


def spread_flag(samples: list[float]) -> str:
    if not samples or min(samples) <= 0:
        return ""
    ratio = max(samples) / min(samples)
    return f"  ** SUSPECT: max/min = {ratio:.2f}x (bimodal?)" if ratio > 1.3 else ""


def report(label: str, on: list[float], off: list[float]) -> dict:
    med_on, med_off = st.median(on), st.median(off)
    delta = (med_off / med_on - 1.0) * 100.0
    print(f"\n  {label}")
    print(f"    MMQ_MFMA=ON   median {med_on:10.2f}  n={len(on)}  {sorted(round(v,1) for v in on)}"
          f"{spread_flag(on)}")
    print(f"    MMQ_MFMA=OFF  median {med_off:10.2f}  n={len(off)}  {sorted(round(v,1) for v in off)}"
          f"{spread_flag(off)}")
    print(f"    OFF vs ON: {delta:+.2f}%")
    return {"surface": label, "median_on": med_on, "median_off": med_off,
            "off_vs_on_pct": delta, "samples_on": on, "samples_off": off,
            "n_per_arm": len(on)}


def main() -> int:
    ap = argparse.ArgumentParser(description="CH-6 MMQ_MFMA build-config A/B")
    ap.add_argument("--build-on", required=True, type=Path, help="MMQ_MFMA=ON bin dir")
    ap.add_argument("--build-off", required=True, type=Path, help="MMQ_MFMA=OFF bin dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=10, help="alternating pairs per surface")
    args = ap.parse_args()

    for b in (args.build_on, args.build_off):
        if not (b / "llama-bench").is_file():
            print(f"REFUSED: no llama-bench in {b}", file=sys.stderr)
            return 2

    surfaces = [
        ("Qwen2.5-Coder-0.5B-Q4_K_M pp512 (the ORIGINAL +26.6% surface)", SMALL, 512, 0),
        ("Qwen3.8-27B-Q8_0 pp512 (the surface that matters)", BIG, 512, 0),
        ("Qwen3.8-27B-Q8_0 tg128 (decode, where MFMA should matter least)", BIG, 0, 128),
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for label, model, pp, tg in surfaces:
        if not model.exists():
            print(f"  SKIPPED (model absent): {label}")
            continue
        key = f"pp{pp}" if pp else f"tg{tg}"
        on, off = [], []
        print(f"\n[{time.strftime('%H:%M:%S')}] {label} — alternating {args.reps} pairs")
        for i in range(args.reps):
            # ALTERNATE, never block-sequential: drift must hit both arms equally.
            for which, bin_dir, sink in (("ON", args.build_on, on),
                                         ("OFF", args.build_off, off)):
                try:
                    got = run_bench(bin_dir, model, pp=pp, tg=tg)
                    sink.append(got[key])
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    print(f"    rep {i} {which}: FAILED {exc}")
            if read_vram() < 0:
                print("    (VRAM sysfs unreadable — residency not provable this rep)")
        if on and off:
            results.append(report(label, on, off))

    (args.out / "mmq_mfma_recheck.json").write_text(
        json.dumps({"results": results,
                    "build_on": str(args.build_on), "build_off": str(args.build_off)},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'mmq_mfma_recheck.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
