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

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.controller import build_recipe                      # noqa: E402
from autokernel.loop import bench, claim, gates                     # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--champion-tree", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/ak-loop-tree"),
                        help="worktree at the champion commit to be audited")
    parser.add_argument("--base-build", type=Path, required=True,
                        help="an EXISTING build of the commit the block started from")
    parser.add_argument("--champion-build", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/build-champion-audit"),
                        help="where to build the champion; rebuilt if absent")
    parser.add_argument("--model", type=Path,
                        default=Path("/mnt/raid0/llm/models/"
                                     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"))
    parser.add_argument("--surface", choices=("pp512", "tg128"), default="tg128")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pp, tg = (512, 0) if args.surface == "pp512" else (0, 128)
    floor = bench.MEASURED_FLOOR_PCT[args.surface].get(args.pairs, 1.188)

    print(f"champion  {args.champion_tree}")
    print(f"base      {args.base_build}")
    verdict = gates.compiles(
        args.champion_tree, args.champion_build,
        cmake_defines=tuple(build_recipe.HOUSE_GPU_RECIPE.cmake_defines()),
        jobs=64, cpu_list="96-183")
    print(f"build     {verdict.gate} passed={verdict.passed} {verdict.reason}")
    if not verdict.passed:
        print(verdict.detail[-800:])
        return 1

    oracle = gates.op_correctness(args.champion_build)
    print(f"oracle    {oracle.gate} passed={oracle.passed} {oracle.reason}")
    if not oracle.passed:
        print(oracle.detail[-600:])
        return 1

    with claim.hold() as receipt:
        print(f"claim     held on {receipt['device_id']}")
        comparison = bench.compare(
            bench.Arm("base", args.base_build / "bin" / "llama-bench"),
            bench.Arm("champion", args.champion_build / "bin" / "llama-bench"),
            args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor)

    body = comparison.to_dict()
    body.update({"schema": "epyc.autokernel.champion_block_audit.v1",
                 "authority": "screening_non_promotable",
                 "champion_tree": str(args.champion_tree),
                 "base_build": str(args.base_build)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "champion-block-audit.json").write_text(
        json.dumps(body, indent=2), encoding="utf-8")

    residency = body["residency"]
    print(f"\nBLOCK effect {comparison.effect * 100:+.3f}%  floor {floor:.3f}%")
    print(f"  decisive={comparison.decisive}  drifting={comparison.drifting}")
    print(f"  drift anchor {comparison.anchor_drift_pct:+.3f}%  "
          f"candidate {comparison.candidate_drift_pct:+.3f}%")
    print(f"  residency {residency['resident']}/{residency['invocations']}  "
          f"clock {residency['sclk_min_mhz']}-{residency['sclk_max_mhz']} "
          f"stable={residency['clock_stable']}")
    print(f"\nwrote {args.out / 'champion-block-audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
