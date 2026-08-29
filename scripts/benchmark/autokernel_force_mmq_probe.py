#!/usr/bin/env python3
"""Does MMQ beat dequant+rocBLAS at ne11=512 on gfx90a? One build, one A/B.

    python3 -m scripts.benchmark.autokernel_force_mmq_probe --out <dir>

WHY THIS EXISTS
---------------
`ggml_cuda_should_use_mmq()` (`ggml/src/ggml-cuda/mmq.cu:240`) sends the contracted
pp512 Q4_K workload down the dequantize -> convert -> rocBLAS/Tensile path, purely
because line :309 gates Q4_K/Q5_K at `ne11 <= 256` and our prompt gives `ne11 = 512`.
That routes ~76% of measured device time (dequant 15.06% + convert 9.32% + Tensile
51.33%) away from our own kernels.

`-DGGML_CUDA_FORCE_MMQ=ON` makes line :288 return true unconditionally, so the same
question is answerable as a BUILD ARM with no source patch, no critic rounds and no
planner spend. If MMQ loses here, hypothesis AK-H-MMQ-1 is dead and the loop should
move to decode; if it wins, the one-line threshold change has a measured effect size
before anyone writes it.

This is a screening measurement under `P-AK-SEARCH-1`: non-promotable by
construction. It holds the mi210_0 claim and proves residency on every invocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.controller import build_recipe                      # noqa: E402
from autokernel.loop import bench, claim, gates                     # noqa: E402

SURFACE_PP = 512
FLOOR_PCT = 2.175 / (5 ** 0.5)          # the enforced prefill floor, 0.973%


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/ak-loop-tree"))
    parser.add_argument("--anchor-build", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/build-anchor-j64"))
    parser.add_argument("--candidate-build", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/build-forcemmq"))
    parser.add_argument("--model", type=Path,
                        default=Path("/mnt/raid0/llm/models/"
                                     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"))
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    recipe = build_recipe.HOUSE_GPU_RECIPE
    defines = tuple(recipe.cmake_defines()) + (("GGML_CUDA_FORCE_MMQ", "ON"),)
    print("recipe    " + recipe.name + " + GGML_CUDA_FORCE_MMQ=ON")
    for name, value in defines:
        print(f"            {name}={value}")

    with claim.hold() as receipt:
        print(f"claim     held on {receipt['device_id']}")

        started = time.monotonic()
        verdict = gates.compiles(args.worktree, args.candidate_build,
                                 cmake_defines=defines, jobs=64,
                                 cpu_list="96-183")
        print(f"build     {verdict.gate} passed={verdict.passed} "
              f"{verdict.reason} ({time.monotonic() - started:.0f}s)")
        if not verdict.passed:
            print(verdict.detail[-1500:])
            return 1

        # MMQ is existing, already-correct code, but this is a different BUILD --
        # prove the op oracle still passes before believing any number from it.
        correctness = gates.op_correctness(args.candidate_build)
        print(f"oracle    MUL_MAT passed={correctness.passed} {correctness.reason}")
        if not correctness.passed:
            print(correctness.detail[-1500:])
            return 1

        comparison = bench.compare(
            bench.Arm("anchor-blas", args.anchor_build / "bin" / "llama-bench"),
            bench.Arm("force-mmq", args.candidate_build / "bin" / "llama-bench"),
            args.model, pp=SURFACE_PP, tg=0, pairs=args.pairs,
            noise_floor_pct=FLOOR_PCT)

    body = comparison.to_dict()
    body.update({
        "schema": "epyc.autokernel.force_mmq_probe.v1",
        "authority": "screening_non_promotable",
        "question": ("does MMQ beat dequantize+rocBLAS at ne11=512 on gfx90a, i.e. is "
                     "the ne11<=256 cutoff at mmq.cu:309 mis-set for CDNA2"),
        "cmake_defines": [list(pair) for pair in defines],
        "model": str(args.model),
    })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "force-mmq-probe.json").write_text(
        json.dumps(body, indent=2), encoding="utf-8")

    effect = comparison.effect * 100.0
    print(f"\neffect    {effect:+.3f}% on {comparison.surface} over "
          f"{comparison.pairs} alternating pairs "
          f"({comparison.estimator}), floor {FLOOR_PCT:.3f}%")
    print(f"decisive  {comparison.decisive}")
    if bench.spread_is_suspect(comparison.anchor_samples) or \
            bench.spread_is_suspect(comparison.candidate_samples):
        print("WARNING   bimodal spread — the median is hiding something; do not "
              "report this effect without explaining the spread")
    verdict_line = ("MMQ WINS — AK-H-MMQ-1 is live, the one-line threshold change has "
                    "a measured effect size" if comparison.decisive and effect > 0 else
                    "MMQ LOSES — AK-H-MMQ-1 is dead; the cutoff is correctly placed"
                    if comparison.decisive else
                    "INSIDE THE FLOOR — not a measurement of anything; MMQ and the "
                    "vendor path are indistinguishable at this shape")
    print(f"verdict   {verdict_line}")
    print(f"\nwrote {args.out / 'force-mmq-probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
