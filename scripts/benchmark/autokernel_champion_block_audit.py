#!/usr/bin/env python3
"""Measure a BLOCK of champion commits directly against the base they started from.

    python3 scripts/benchmark/autokernel_champion_block_audit.py \
        --base-build <dir> --champion-tree <dir> --out <dir>

WHY THIS EXISTS
---------------
When the anchor fails to advance -- run 13, run 14, and run 17, three different causes
-- every recorded per-commit effect becomes CUMULATIVE, and the marginal value of any
one commit is unrecoverable from the record. Inferring it from two cumulative figures
carries roughly sqrt(2) their uncertainty, which is how a real +1.088% gain was once
demoted on a comparison against a floor calibrated for a direct measurement.

The block audit sidesteps the whole problem: build the current champion, build (or
reuse) the base, and run ONE paired A/B between them. That answers the only question
that matters after a frozen-anchor run -- is the tree actually faster than what it
started from -- with the ordinary floor rather than an inflated one.

Run 17: 30 commits, +3.942%, decisive, no drift. They were kept on that evidence
rather than rolled back on arithmetic.

Non-promotable screening under P-AK-SEARCH-1. Holds the mi210_0 claim and proves
residency on every invocation.
"""
from pathlib import Path
from autokernel.controller import build_recipe
from autokernel.loop import bench, claim, gates
CHAMP = Path("/mnt/raid0/llm/tmp/build-champ-8fd1b23a")
BASE = Path("/mnt/raid0/llm/tmp/build-anchor-champ")
print("building current champion 8fd1b23a ...", flush=True)
v = gates.compiles(Path("/mnt/raid0/llm/tmp/ak-loop-tree"), CHAMP,
                   cmake_defines=tuple(build_recipe.HOUSE_GPU_RECIPE.cmake_defines()),
                   jobs=64, cpu_list="96-183")
print("  build:", v.gate, v.passed, v.reason, flush=True)
if not v.passed:
    print(v.detail[-800:]); raise SystemExit(1)
ok = gates.op_correctness(CHAMP)
print("  oracle:", ok.gate, ok.passed, ok.reason, flush=True)
if not ok.passed:
    print(ok.detail[-600:]); raise SystemExit(1)
with claim.hold() as receipt:
    print("  claim held on", receipt["device_id"], flush=True)
    c = bench.compare(bench.Arm("run17-start", BASE / "bin" / "llama-bench"),
                      bench.Arm("champion-now", CHAMP / "bin" / "llama-bench"),
                      Path("/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
                      pp=0, tg=128, pairs=20, noise_floor_pct=1.188)
d = c.to_dict()
out = Path("/mnt/raid0/llm/autokernel/loop-memory/run17-audit")
out.mkdir(parents=True, exist_ok=True)
(out / "total.json").write_text(json.dumps(d, indent=2))
print("\nTOTAL effect of run 17's 30 commits: %+.3f%%" % d["effect_pct"])
print("  floor %.3f%%  decisive=%s  drifting=%s" % (d["noise_floor_pct"], d["decisive"], d["drifting"]))
print("  drift anchor %+.3f%%  cand %+.3f%%" % (d["anchor_drift_pct"], d["candidate_drift_pct"]))
r = d["residency"]
print("  residency %s/%s  clock %s-%s stable=%s" % (r["resident"], r["invocations"], r["sclk_min_mhz"], r["sclk_max_mhz"], r["clock_stable"]))
