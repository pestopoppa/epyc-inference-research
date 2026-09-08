#!/usr/bin/env python3
"""Maximum-performance sweep for Qwen3.6-35B-A3B-MTP-Q8_0 on the CURRENT champion (ef81196d5 + everything folded).

No baseline. The question is only: how fast does it go, at its best, right now.
Sweeps parallel slots with the MTP speculative drafter, several launches per point so the
number carries a between-LAUNCH spread rather than one session's tightness.
"""
import dataclasses, json, sys, time
from pathlib import Path
sys.path.insert(0, "/mnt/raid0/llm/worktrees/mains/ak-rebuild-research/scripts/kernel_rnd")
from autokernel.loop import serving  # noqa: E402

BUILD = Path("/mnt/raid0/llm/tmp/build-fold-ef81196d5")
RECIPE = Path("/mnt/raid0/llm/worktrees/mains/ak-rebuild-research/artifacts/serving-recipes/qwen3.6-35b-a3b-q8-gpu-mtp.json")
OUT = Path("/mnt/raid0/llm/tmp/maxperf-35b-20260908/sweep.json")
NPS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "4", "8"])]
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 3

base = serving.Recipe.load(RECIPE)
print(f"build {BUILD}  recipe {base.name}  np sweep {NPS}  x{SAMPLES} launches each", flush=True)
results = {}
for np_ in NPS:
    r = dataclasses.replace(base, np=np_, name=f"{base.name}-maxperf-np{np_}")
    t0 = time.time()
    try:
        out = serving.calibrate_floor(r, BUILD, samples=SAMPLES)
    except Exception as e:
        print(f"np={np_}: FAILED {type(e).__name__}: {e}", flush=True)
        results[np_] = {"error": f"{type(e).__name__}: {e}"}
        OUT.write_text(json.dumps(results, indent=2, default=str)); continue
    med = out["median_tok_s"]
    sp = out.get("spread", {})
    results[np_] = {"median_aggregate_tok_s": med, "per_slot_tok_s": med / np_,
                    "runs": out.get("runs"), "cv_pct": out.get("cv_pct"),
                    "p95_dev_pct": out.get("floor_pct"), "sd_pct": sp.get("sd"),
                    "min": sp.get("min"), "max": sp.get("max"),
                    "residency": out.get("residency"), "recipe_hash": r.recipe_hash,
                    "seconds": round(time.time() - t0, 1)}
    print(f"np={np_:2d}  aggregate {med:8.2f} tok/s   per-slot {med/np_:7.2f}   "
          f"spread p95dev {out.get('floor_pct'):.2f}%  runs {[round(x,1) for x in out.get('runs',[])]}  [{time.time()-t0:.0f}s]", flush=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))

print("\n=== MAXIMUM PERFORMANCE, Qwen3.6-35B-A3B-MTP-Q8_0, champion ef81196d5 + MTP ===", flush=True)
ok = {k: v for k, v in results.items() if "median_aggregate_tok_s" in v}
if ok:
    best_agg = max(ok.items(), key=lambda kv: kv[1]["median_aggregate_tok_s"])
    print(f"  best AGGREGATE : {best_agg[1]['median_aggregate_tok_s']:.2f} tok/s at np={best_agg[0]}")
    print(f"  single stream  : {ok.get(1,{}).get('median_aggregate_tok_s','n/a')} tok/s (np=1)")
print(f"written: {OUT}", flush=True)
