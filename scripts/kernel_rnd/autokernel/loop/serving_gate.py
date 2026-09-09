#!/usr/bin/env python3
"""Run the R23-44 serving gate ONCE on the seeded bundle, BY HAND.

    python3 -m autokernel.loop.serving_gate \
        --cor-build /mnt/raid0/llm/tmp/build-cor-445e93a8 \
        --tip-build /mnt/raid0/llm/autokernel/loop-memory/anchor-gen-021 \
        [--pairs 5] [--execute] [--apply]

Run ONLY with the loop dead and the GPU claim free, AFTER `seed_bundle --apply` wrote the
bundle with a MEASURED compounded tip-vs-cor. Uses the loop's own `serving.compare`
(llama-server under the canonical recipe, paired alternating, per-request aggregate) and
`accumulate.resolve`, so the record is the same schema the loop would have written
(`epyc.autokernel.serving_ab.v1`) -- a hand-run gate that wrote a different shape would be
evidence nothing downstream could read.

WHAT CHANGED WHEN THIS CAME INTO THE PACKAGE. The scratch original built the floor path
itself (`STORE / f"serving-floor.{recipe.name}.json"`) and read `["floor_pct"]` off it.
That is the READER half of the defect `run.py` was fixed for: a floor is a property of the
measured CONDITION, not of the recipe's NAME, so a hand-run gate could be -- and under
R23-49's cpu_list pin nearly was -- judged against a floor calibrated under another
condition, with nothing raised. It now goes through `serving.load_floor`, which refuses a
mismatch and stamps the provenance onto the record.

Dry by default (`instruments.Posture`). `--execute` spends the gate and writes the
evidence record; `--apply` additionally advances the champion of record on a PROMOTE. The
headline is left to the relaunch's publish (the loop re-publishes on its first keep or
guard).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from . import accumulate, instruments, pool, serving, status


def measure(recipe: serving.Recipe, cor_build: Path, tip_build: Path, *,
            pairs: int, floor_pct: float) -> dict:
    """THE MEASUREMENT SEAM -- the only thing here that touches the GPU.

    A thin wrapper on purpose: everything above it (floor identity, the scheduling
    decision, the resolve/record/apply chain) is then testable without hardware, and a
    test that stubs this cannot accidentally launch a server.
    """
    return serving.compare(recipe, cor_build, tip_build, pairs=pairs, floor_pct=floor_pct)


def _is_ancestor(worktree: Path, ancestor: str, descendant: str) -> bool:
    """Read-only lineage check for the explicitly pinned serving-gate tip."""
    done = subprocess.run(
        ["git", "-C", str(worktree), "merge-base", "--is-ancestor",
         ancestor, descendant], capture_output=True, text=True)
    if done.returncode not in (0, 1):
        raise RuntimeError(
            f"git ancestry check failed in {worktree}: {done.stderr.strip()}"
        )
    return done.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autokernel.loop.serving_gate", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    instruments.add_recipe_args(parser)
    parser.add_argument("--cor-build", type=Path, required=True,
                        help="build of the champion OF RECORD (the anchor arm)")
    parser.add_argument("--tip-build", type=Path, required=True,
                        help="build of the accumulator TIP (the candidate arm)")
    parser.add_argument("--tip", required=True,
                        help="source commit exactly represented by --tip-build")
    parser.add_argument("--champion-worktree", type=Path, default=pool.CHAMPION_TREE,
                        help="read-only source topology for Bundle ancestry checks")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--fire-multiple", type=float, default=2.5)
    instruments.add_posture_args(
        parser, apply_help="on PROMOTE, advance the bundle's champion of record and save "
                           "a fresh bundle.")
    parser.add_argument("--force", action="store_true",
                        help="ONE-OFF operator permission (2026-09-08 fold window): spend "
                             "the serving gate even if the bundle is below fire_multiple x "
                             "floor. The loop's own scheduling rule is unchanged; promotion "
                             "still requires the serving A/B to clear the calibrated floor.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    posture = instruments.resolve_posture(args)
    try:
        recipe = serving.Recipe.load(args.recipe)
        print(f"recipe    {recipe.describe()}")
        reading = instruments.read_floor(args.store, recipe)
        floor = reading.floor_pct
        bundle, _ = accumulate.load_bundle(
            Path(args.store), anchor_commit=args.tip,
            is_ancestor=lambda older, newer: _is_ancestor(
                args.champion_worktree, older, newer),
            read_only=posture.dry_run)
        policy = accumulate.AccumulatorPolicy(fire_multiple=args.fire_multiple)
        thr = policy.fire_threshold_pct(floor)
        print(f"floor     {floor:.3f}% [{reading.provenance}] from {reading.path.name} | "
              f"fire threshold {thr:.2f}%")
        magnitude_label = (
            "current combined measurement"
            if bundle.measurement_validity == accumulate.MEASUREMENT_CURRENT
            else "historical-only magnitude; threshold disabled"
        )
        print(f"bundle    cor {bundle.champion_of_record[:12]} tip {bundle.tip[:12]} "
              f"keeps {len(bundle.keeps)} compounded {bundle.compounded_bench_pct:+.3f}% "
              f"validity={bundle.measurement_validity} ({magnitude_label})")
        decision = accumulate.decide_after_keep(bundle, floor, policy)
        print(f"schedule  decide_after_keep -> {decision}  (fire_multiple x floor = "
              f"{thr:.2f}% is only the loop's SCHEDULING heuristic)")
        if decision is not accumulate.Decision.FIRE_SERVING and not args.force:
            print("bundle does not clear the scheduling threshold and --force not given. "
                  "Nothing measured.")
            return instruments.REFUSED
        # Operator 2026-09-08: "just make sure it passes the noise floor". In the fold
        # window the gate is spent regardless of the scheduling threshold; PROMOTE still
        # requires the serving A/B to be DECISIVE, i.e. |effect| >= the calibrated serving
        # floor AND positive -- that check lives in `accumulate.resolve()` and is not
        # bypassable here.
        cor_binary = instruments.require_binary(args.cor_build, "llama-server")
        tip_binary = instruments.require_binary(args.tip_build, "llama-server")
        print(f"builds    anchor {cor_binary}\n          candidate {tip_binary}")
        print(f"posture   {posture.describe()}")
        if posture.dry_run:
            print(f"\nDRY RUN -- would run {args.pairs} alternating serving pairs against "
                  f"floor {floor:.3f}%. Pass --execute to spend the gate.")
            return 0
        started = time.time()
        row = measure(recipe, args.cor_build, args.tip_build,
                      pairs=args.pairs, floor_pct=floor)
    except (instruments.InstrumentRefusal, serving.ServingFloorMismatch,
            accumulate.BundleRecoveryRequired) as refusal:
        # A floor calibrated under a DIFFERENT recipe is a refusal, not a verdict: it
        # exits REFUSED with the message rather than a traceback, so the caller can tell
        # "no reading" from "a bad reading".
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    plan = accumulate.resolve(bundle, row, policy)
    record = {"outcome": plan["outcome"].value, "reason": plan["reason"],
              "bundled_keeps": list(bundle.keeps),
              "planner_evidence": plan.get("planner_evidence"),
              "hand_run": "autokernel.loop.serving_gate",
              # The provenance of the bar travels with the verdict: a reader cannot
              # otherwise tell a checked floor from an assumed one (run.py does the same).
              "floor_provenance": reading.provenance, **row}
    written = status.write_json(Path(args.store) / "serving",
                                f"bundle-{bundle.tip[:12]}.json", record, prefix=".sv-")
    print(f"serving   {plan['reason']}  [{time.time() - started:.0f}s]  record: {written}")
    print(f"effect    anchor {row['anchor_tok_s']:.2f} tok/s vs tip "
          f"{row['candidate_tok_s']:.2f} tok/s -> {row['effect_pct']:+.2f}% "
          f"(floor {floor:.3f}%, decisive={row['decisive']})")
    if plan["outcome"] is accumulate.Outcome.PROMOTE:
        print("PROMOTE: champion of record advances to", bundle.tip[:12])
        if posture.apply:
            fresh = accumulate.Bundle(champion_of_record=bundle.tip, tip=bundle.tip)
            fresh.save(Path(args.store))
            print("bundle reset and saved:", json.dumps(fresh.to_dict()))
        else:
            print("(dry: pass --apply to advance the champion of record)")
    else:
        print("HOLD: champion of record unchanged; bundle kept; divergence is planner "
              "evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
