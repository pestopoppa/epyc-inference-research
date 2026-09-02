#!/usr/bin/env python3
"""Measure champion-vs-production on the CONFIRM rung (the production model).

Why this exists (R23-19, 2026-09-02). The published headline for champion
`732389d6` reads +27.363%, but its evidence record carries `peak_vram_bytes`
1.49 GB and 536-684 t/s samples: it was measured on the 1.5B SCREEN rung, not
on production's Qwen3.8-27B-Q8_0 (~29 GB, ~65 t/s on the same surface). Run 23
predates R23-11, so `headline_model` fell back to `--model` by construction.
D4 moves the headline to the confirm rung for FUTURE runs; this one-shot
measures the number for the champion that already exists, so the program's
headline stops being a screen-shape claim. CH-6 is the precedent for why it
matters: MMQ_MFMA measured +23.09% on the 0.5B and +0.50% on the 27B.

It reuses the loop's own `production.refresh` + `bench.compare` -- no second
implementation of the measurement, and the published bundle is schema-identical
to the one the loop writes (including R23-11's `model` field, which is exactly
the provenance whose absence made the screen-rung headline unreadable).

The floor is re-keyed to the confirm model via the loop's `noise_floor_pct`, so
the 27B number is judged against the 27B A/A floor, never the 1.5B's 0.668%.

Fail-closed: refuses on a missing/unbuilt arm, an uncalibrated confirm surface,
or a champion build whose provenance does not name the champion commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.loop import bench, production  # noqa: E402
from autokernel.loop.run import noise_floor_pct  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("/mnt/raid0/llm/autokernel/loop-memory"))
    ap.add_argument("--champion-build", type=Path, required=True,
                    help="guard-verified anchor-gen dir holding bin/llama-bench")
    ap.add_argument("--champion-commit", required=True)
    ap.add_argument("--model", type=Path, required=True, help="the CONFIRM rung model")
    ap.add_argument("--surface", default="dec-b4", choices=tuple(bench.SURFACES))
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--baseline-build", type=Path, default=None,
                    help="cached frozen-production build (skips a rebuild if present)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    champ_bench = args.champion_build / "bin" / "llama-bench"
    if not champ_bench.is_file():
        print(f"REFUSE: no llama-bench at {champ_bench}", file=sys.stderr); return 2
    if not args.model.is_file():
        print(f"REFUSE: confirm model {args.model} not found", file=sys.stderr); return 2

    prov = args.champion_build / "provenance.json"
    if prov.is_file():
        declared = json.loads(prov.read_text()).get("champion_commit", "")
        if declared and not declared.startswith(args.champion_commit[:12]):
            print(f"REFUSE: {args.champion_build} declares champion {declared[:12]}, "
                  f"not {args.champion_commit[:12]} -- that build does not hold this champion",
                  file=sys.stderr)
            return 2

    floor = noise_floor_pct(args.surface, args.pairs, args.model, store=args.store)
    if floor is None:
        print(f"REFUSE: {args.surface} is UNCALIBRATED for {args.model.stem} -- "
              f"publishing would state a headline against a floor nobody measured. "
              f"Run the A/A calibration for this (surface, model) first.", file=sys.stderr)
        return 3

    pp, tg, ubatch = bench.SURFACES[args.surface]
    print(f"headline rung : {args.model.name}")
    print(f"surface       : {args.surface} (pp={pp} tg={tg} ubatch={ubatch})")
    print(f"pairs         : {args.pairs}")
    print(f"floor         : {floor}%  (keyed to THIS model, not the screen rung)")
    print(f"champion      : {args.champion_commit[:12]} @ {args.champion_build}")
    if args.dry_run:
        print("\nDRY RUN -- wiring proven, no device time spent.")
        return 0

    outcome = production.refresh(
        store=args.store,
        champion_commit=args.champion_commit,
        champion_build=args.champion_build,
        baseline_build=args.baseline_build,
        build_baseline=None,   # never build production here; a cached slot or refuse
        compare=lambda base, champ: bench.compare(
            bench.Arm("production_v9", base / "bin" / "llama-bench"),
            bench.Arm("champion", champ / "bin" / "llama-bench"),
            args.model, pp=pp, tg=tg, pairs=args.pairs,
            noise_floor_pct=floor, surface=args.surface, ubatch=ubatch,
            calibrated=True),
        on_step=lambda label: print(f"  .. {label}"),
        note="R23-19 one-shot: headline re-measured on the CONFIRM rung; the prior "
             "bundle for this champion was measured on the 1.5B screen rung.")
    print(f"\nheadline  {outcome.reason}")
    return 0 if outcome.published else 4


if __name__ == "__main__":
    raise SystemExit(main())
