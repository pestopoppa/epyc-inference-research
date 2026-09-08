#!/usr/bin/env python3
"""Re-calibrate the SERVING floor under a given recipe (R23-49).

    python3 -m autokernel.loop.recal_serving_floor \
        --build /mnt/raid0/llm/autokernel/loop-memory/anchor-gen-021 \
        [--samples 5] [--execute] [--apply]

WHY A RECALIBRATION IS EVER NEEDED. A floor is a property of the measured CONDITION.
Pinning the llama-server host threads (`Recipe.cpu_list`) changed the condition, so the
3.536% floor calibrated unpinned was void for the pinned recipe -- and nothing on disk
said so. An `env` arm makes it sharper still: its entire purpose is to change DISPERSION,
which is to change the floor itself.

This runs the loop's own A/A (`serving.calibrate_floor`: N launches of ONE build under the
recipe, p95 |deviation from median|) and files the result through `serving.write_floor`.

WHAT CHANGED WHEN THIS CAME INTO THE PACKAGE. The scratch original wrote
`STORE / f"serving-floor.{recipe.name}.json"` with a bare `write_text`. That is the WRITER
half of the same defect: no atomic write (a crashed calibration could leave a half-written
floor for a gate to read), and no identity stamp or refusal (a row produced by one recipe
could be filed under another's name, which is a bar nobody would question). `write_floor`
is now the one writer, so the identity is IN the file and `load_floor` can check it.

Run ONLY with the loop dead and the GPU claim free, and under the same foreign load the
condition is meant to include (the recipe's own cpu_list pin stays in force).

Dry by default: `--execute` takes the A/A and prints the floor; `--apply` additionally
writes the floor file, keeping the previous one as a dated backup.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

from . import instruments, serving


def measure(recipe: serving.Recipe, build: Path, *, samples: int) -> dict:
    """THE MEASUREMENT SEAM -- N server launches. Stubbed by every test here."""
    return serving.calibrate_floor(recipe, build, samples=samples)


def backup_path(target: Path, label: str, *, day: str | None = None) -> Path:
    """Where the floor being replaced is kept. Derived from `serving.floor_path`'s answer,
    never from a second hand-built name -- a backup filed beside the wrong floor is a
    record of a condition nobody can identify."""
    day = day or time.strftime("%Y%m%d")
    return target.with_name(f"{target.stem}.{label}-{day}{target.suffix}.bak")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autokernel.loop.recal_serving_floor", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    instruments.add_recipe_args(parser)
    parser.add_argument("--build", type=Path, required=True,
                        help="the ONE build the A/A is run on")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--require-cpu-list", action="store_true",
                        help="refuse a recipe with no cpu_list. The R23-49 recalibration "
                             "was for the PINNED condition specifically; this makes that "
                             "intent explicit instead of assuming it.")
    parser.add_argument("--backup-label", default="prev",
                        help="label in the replaced floor's backup filename "
                             "(default: %(default)s)")
    parser.add_argument("--host-state", default=None,
                        help="free-form note recorded in the floor's `conditions` block, "
                             "e.g. 'autokernel loop DOWN (fold window 2026-09-08)'")
    instruments.add_posture_args(parser, apply_help="write the floor file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    posture = instruments.resolve_posture(args)
    try:
        recipe = serving.Recipe.load(args.recipe)
        if args.require_cpu_list and not recipe.cpu_list:
            raise instruments.InstrumentRefusal(
                "recipe has no cpu_list -- this recal was asked for the PINNED condition only")
        instruments.require_binary(args.build, "llama-server")
        target = serving.floor_path(args.store, recipe)
        print(f"recipe    {recipe.describe()}")
        print(f"build     {args.build} | samples {args.samples}")
        print(f"target    {target}")
        print(f"posture   {posture.describe()}")
        if posture.dry_run:
            print(f"\nDRY RUN -- would run {args.samples} A/A launches and file the floor "
                  f"at the path above. Pass --execute to measure.")
            return 0
        started = time.time()
        row = measure(recipe, args.build, samples=args.samples)
    except (instruments.InstrumentRefusal, serving.ServingFloorMismatch) as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    conditions = {
        "cpu_list": recipe.cpu_list,
        "host_state": args.host_state or "hand-run recalibration (autokernel loop DOWN)",
        "harness": "serving.calibrate_floor A/A, one build, fresh server per sample",
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    runs = [round(value, 2) for value in row.get("runs", [])]
    print(f"floor_pct {row['floor_pct']:.3f}% (cv {row['cv_pct']:.3f}%, median "
          f"{row['median_tok_s']:.2f} tok/s, runs {runs}) [{time.time() - started:.0f}s]")
    if not posture.apply:
        print("DRY (no --apply): floor NOT written")
        return 0
    if target.exists():
        backup = backup_path(target, args.backup_label)
        shutil.copy2(target, backup)
        print("previous floor backed up:", backup)
    # `write_floor` refuses a row produced by a DIFFERENT recipe, stamps the identity at
    # top level and replaces atomically. The refusal is not defensive noise: filing one
    # arm's A/A under another arm's name is the copy-paste this whole module exists to
    # make impossible.
    try:
        written = serving.write_floor(args.store, recipe, row, conditions=conditions)
    except serving.ServingFloorMismatch as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    print("floor written:", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
