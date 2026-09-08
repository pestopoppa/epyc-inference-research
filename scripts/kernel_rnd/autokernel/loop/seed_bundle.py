#!/usr/bin/env python3
"""R23-51a -- seed the durable accumulator bundle with a MEASURED tip-vs-cor number.

    python3 -m autokernel.loop.seed_bundle --cor 445e93a8 \
        --cor-build /mnt/raid0/llm/tmp/build-cor-445e93a8 \
        --tip-build /mnt/raid0/llm/autokernel/loop-memory/anchor-gen-021 \
        [--champion-worktree /mnt/raid0/llm/tmp/ak-loop-tree] [--execute] [--apply]

Run ONLY in a fold window, with the loop verified dead and the GPU claim free.

NEVER WITH A PRODUCT OF SOLOS. `compounded_bench_pct` is what the serving gate will be
asked to confirm, and keeps INTERACT: multiplying each keep's marginal effect together
produces a number no measurement ever took. So the value written here is the loop's own
paired A/B (`bench.compare`) of the champion-of-record build against the accumulator tip
build -- one measurement of the whole bundle, exactly as `run.py` re-measures it after
every keep.

The keep list is read from the champion worktree's commit topology (`akm-` subjects
between the champion of record and the tip), and the instrument refuses outright if the
champion of record is not an ancestor of the tip: a bundle describing a lineage the tree
does not contain is the laundering `accumulate.load_bundle` exists to refuse.

Dry by default: `--execute` takes the A/B and records the measurement; `--apply`
additionally writes the bundle into the store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import accumulate, bench, instruments, pool, status


def measure(cor_build: Path, tip_build: Path, model: Path, *, pairs: int,
            floor_pct: float) -> dict:
    """THE MEASUREMENT SEAM -- the loop's own paired A/B on the tg128 surface."""
    pp, tg, ubatch = bench.SURFACES["tg128"]
    return bench.compare(bench.Arm("champion_of_record", cor_build / "bin" / "llama-bench"),
                         bench.Arm("accumulator_tip", tip_build / "bin" / "llama-bench"),
                         model, pp=pp, tg=tg, pairs=pairs, noise_floor_pct=floor_pct,
                         surface="tg128", ubatch=ubatch, calibrated=True).to_dict()


def _git(worktree: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(worktree), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def is_ancestor(worktree: Path, ancestor: str, descendant: str) -> bool:
    """`git merge-base --is-ancestor`, read from the EXIT CODE.

    The scratch original asserted on the command's stdout being empty, which is true of
    both answers -- the check could not fail. The refusal it was reaching for is real, so
    it is implemented here rather than dropped.
    """
    return subprocess.run(
        ["git", "-C", str(worktree), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True, text=True).returncode == 0


def keeps_between(worktree: Path, cor: str, tip: str) -> list[str]:
    """The mechanism ids of the keeps the bundle batches: `akm-...` commit subjects."""
    return [line.split(":")[0]
            for line in _git(worktree, "log", "--format=%s", f"{cor}..{tip}").splitlines()
            if line.startswith("akm-")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autokernel.loop.seed_bundle", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cor", required=True, help="champion-of-record commit")
    parser.add_argument("--cor-build", type=Path, required=True)
    parser.add_argument("--tip-build", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=instruments.DEFAULT_STORE)
    parser.add_argument("--model", type=Path, default=instruments.DEFAULT_MODEL)
    parser.add_argument("--champion-worktree", type=Path, default=pool.CHAMPION_TREE,
                        help="tree whose topology names the keeps (default: the loop's "
                             "own champion tree, %(default)s)")
    parser.add_argument("--tip", default=None,
                        help="accumulator tip commit (default: the champion worktree's HEAD)")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--floor", type=float, default=0.638,
                        help="tg128 noise floor %%, calibrated 2026-09-04 at 20 pairs")
    parser.add_argument("--measurement-out", type=Path, default=None,
                        help="where the raw comparison is recorded "
                             "(default: <store>/seed-measurement.json)")
    instruments.add_posture_args(
        parser, apply_help="write the seeded bundle into the store.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    posture = instruments.resolve_posture(args)
    try:
        worktree = Path(args.champion_worktree)
        tip = args.tip or _git(worktree, "rev-parse", "HEAD")
        if not is_ancestor(worktree, args.cor, tip):
            raise instruments.InstrumentRefusal(
                f"cor {args.cor} is not an ancestor of the tip {tip[:12]} in {worktree}: "
                f"the bundle would describe a lineage this tree does not contain")
        keeps = keeps_between(worktree, args.cor, tip)
        instruments.require_binary(args.cor_build, "llama-bench")
        instruments.require_binary(args.tip_build, "llama-bench")
        # The loop's flock file is the caller's responsibility: this instrument is run by
        # hand in a fold window, and the operator proves the GPU claim is free before
        # invoking it. The scratch original had a no-op `if lock.exists(): pass` here that
        # checked nothing; a check that cannot fail is worse than an honest precondition.
        print(f"lineage   cor {args.cor} -> tip {tip[:12]}: {len(keeps)} keeps: {keeps}")
        print(f"builds    cor {args.cor_build}\n          tip {args.tip_build}")
        print(f"surface   tg128, {args.pairs} alternating pairs, floor {args.floor}%")
        print(f"posture   {posture.describe()}")
        if posture.dry_run:
            print("\nDRY RUN -- nothing measured. Pass --execute to run the paired A/B "
                  "that supplies compounded_bench_pct.")
            return 0
        comparison = measure(args.cor_build, args.tip_build, args.model,
                             pairs=args.pairs, floor_pct=args.floor)
    except instruments.InstrumentRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    print(f"MEASURED compounded tip-vs-cor: {comparison['effect_pct']:+.3f}%  "
          f"decisive={comparison['decisive']} drifting={comparison.get('drifting')}")
    out = Path(args.measurement_out) if args.measurement_out else None
    if out is None:
        written = status.write_json(Path(args.store), "seed-measurement.json",
                                    comparison, prefix=".seed-")
    else:
        written = status.write_json(out.parent, out.name, comparison, prefix=".seed-")
    print("measurement recorded:", written)
    bundle = accumulate.Bundle(champion_of_record=args.cor, tip=tip, keeps=keeps,
                               compounded_bench_pct=comparison["effect_pct"])
    if posture.apply:
        path = bundle.save(Path(args.store))
        print("bundle written:", path, json.dumps(bundle.to_dict()))
    else:
        print("DRY (no --apply). Would write:", json.dumps(bundle.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
