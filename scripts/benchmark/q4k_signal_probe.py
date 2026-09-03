#!/usr/bin/env python3
"""UNGATED signal probe: does the champion's Q4_K-gated work do anything on a Q4_K model?

NOT CLAIM-GRADE, and deliberately so. There is no calibrated noise floor for
(surface, Q4_K model), so this cannot say "decisive" and must never be published
as a headline. It exists to answer ONE cheap question — is there an effect large
enough to justify the ~3 h A/A calibration that WOULD make it claim-grade?

Why it still uses the loop's own `bench.compare` rather than a hand-rolled
llama-bench loop: that gets the paired alternating design, residency sampling and
drift detection for free, so a "no effect" answer is trustworthy rather than an
artifact of a sloppy harness. Only the floor is missing, so `calibrated=False` is
passed and `decisive` comes back None by construction.

Context (R23-29): champion commits `7d2ea88b` and `732389d6` are hard-gated on
`GGML_TYPE_Q4_K` and therefore CANNOT fire on the Q8_0 production model — which is
why the champion measures production-neutral there. On a Q4_K workload they should
fire. If they do not move the needle here, the "banked for a future Q4_K target"
thesis is weaker than it looks and should be recorded as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.loop import bench  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion-build", type=Path, required=True)
    ap.add_argument("--baseline-build", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--surface", default="dec-b4", choices=tuple(bench.SURFACES))
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for label, b in (("champion", args.champion_build), ("baseline", args.baseline_build)):
        if not (b / "bin" / "llama-bench").is_file():
            print(f"REFUSE: no llama-bench in {label} build {b}", file=sys.stderr)
            return 2
    if not args.model.is_file():
        print(f"REFUSE: model {args.model} not found", file=sys.stderr)
        return 2

    pp, tg, ubatch = bench.SURFACES[args.surface]
    print(f"UNGATED SIGNAL PROBE — not claim-grade, no calibrated floor")
    print(f"  model    : {args.model.name}")
    print(f"  surface  : {args.surface} (pp={pp} tg={tg} ubatch={ubatch}), {args.pairs} pairs")
    print(f"  baseline : {args.baseline_build}")
    print(f"  champion : {args.champion_build}")

    cmp_ = bench.compare(
        bench.Arm("production_v9", args.baseline_build / "bin" / "llama-bench"),
        bench.Arm("champion", args.champion_build / "bin" / "llama-bench"),
        args.model, pp=pp, tg=tg, pairs=args.pairs,
        noise_floor_pct=None, surface=args.surface, ubatch=ubatch,
        calibrated=False)

    d = cmp_.to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(d, indent=2), encoding="utf-8")

    a, c = d["anchor_samples"], d["candidate_samples"]
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"\n  production median : {med(a):.2f} t/s")
    print(f"  champion median   : {med(c):.2f} t/s")
    print(f"  effect            : {d['effect_pct']:+.3f}%   (decisive={d['decisive']}, "
          f"drifting={d['drifting']})")
    r = d.get("residency", {})
    print(f"  residency         : {r.get('resident')}/{r.get('invocations')} resident, "
          f"peak VRAM {r.get('peak_vram_bytes', 0)/1e9:.1f} GB, "
          f"clocks {r.get('sclk_min_mhz')}-{r.get('sclk_max_mhz')} stable={r.get('clock_stable')}")
    print(f"\n  record: {args.out}")
    print("  REMINDER: no floor -> this is a SIGNAL, not a verdict. Calibrate before claiming.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
