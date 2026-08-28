#!/usr/bin/env python3
"""Drive the discovery loop end to end on real hardware.

    python3 -m scripts.kernel_rnd.autokernel.loop.run \
        --worktree /mnt/raid0/llm/tmp/ak-loop-tree \
        --anchor-build /mnt/raid0/llm/tmp/build-anchor-j64 \
        --model /mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
        --store /mnt/raid0/llm/autokernel/loop-memory \
        --iterations 10

Holds the mi210_0 flock for the whole run, refuses a workload that does not dispatch
production's kernels, and records every iteration -- kept or not -- into durable
memory that outlives this process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ..controller import build_recipe, workload_contract
from . import actors, archive, bench, claim, gates, hotspots, loop, status

#: Measured 2026-08-28, n=20 alternating pairs. Prefill is the cheaper surface to
#: detect on; decode has heavier tails.
NOISE_FLOOR_PCT = {"pp512": 2.175 / (5 ** 0.5), "tg128": 3.452 / (5 ** 0.5)}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", type=Path, required=True,
                        help="candidate source tree the planner edits")
    parser.add_argument("--anchor-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/build-candidate-loop"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=bench.MIN_PAIRS)
    parser.add_argument("--surface", choices=("pp512", "tg128"), default="pp512")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="prove the wiring without a provider call or a build")
    args = parser.parse_args(argv)

    # The workload must dispatch the kernels production dispatches. Refuse loudly.
    census = workload_contract.verify_workload(args.model)
    recipe = build_recipe.HOUSE_GPU_RECIPE
    print(f"workload  {args.model.name}: n_embd={census.n_embd}, "
          f"dominant {census.dominant_quant}")
    print(f"recipe    {recipe.name} {recipe.sha256()[:12]}  "
          f"divergences={[f.name for f in recipe.divergences()] or 'none'}")

    anchor_commit = _git(args.worktree, "rev-parse", "HEAD")
    epoch = archive.epoch_for(anchor_commit=anchor_commit,
                              build_recipe=recipe.to_dict())
    print(f"anchor    {anchor_commit[:12]}   epoch {epoch[:12]}")

    pp, tg = (512, 0) if args.surface == "pp512" else (0, 128)
    floor = NOISE_FLOOR_PCT[args.surface]
    print(f"surface   {args.surface}, {args.pairs} alternating pairs, "
          f"noise floor {floor:.3f}%")

    if args.dry_run:
        print("\nDRY RUN — wiring proven, nothing spent.")
        return 0

    planner = actors.CodexPlanner(workspace=args.worktree)
    critic = actors.CodexCritic(workspace=args.worktree)

    def read_inbox() -> list[str]:
        """Re-read every iteration, not once at startup.

        The inbox is how a hypothesis reaches the planner from outside the loop -- from a
        handoff, a backlog row, or the operator mid-run. Reading it once means anything
        dropped in after launch is invisible until the next restart, which is how the
        channel stayed empty while the backlog held measured levers for the exact kernels
        the planner was re-deriving.
        """
        inbox_dir = args.store / "inbox"
        if not inbox_dir.is_dir():
            return []
        return [path.read_text(encoding="utf-8").strip()
                for path in sorted(inbox_dir.glob("*.md"))]

    def build_context() -> dict:
        return {
            "program": loop.PROGRAM.read_text(encoding="utf-8"),
            "kernel_hotspots": [row.to_dict() for row in hotspot_rows],
            "prior_experiments": archive.recall(args.store, epoch=epoch),
            "inbox": read_inbox(),
        }

    def gate(hypothesis, paths):
        return gates.run_all(
            gates.compiles(args.worktree, args.candidate_build,
                           cmake_defines=recipe.cmake_defines(),
                           jobs=64, cpu_list="96-183"),
            gates.op_correctness(args.candidate_build),
        )

    def measure(hypothesis, paths):
        return bench.compare(
            bench.Arm("anchor", args.anchor_build / "bin" / "llama-bench"),
            bench.Arm("candidate", args.candidate_build / "bin" / "llama-bench"),
            args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor)

    def reset_tree() -> None:
        """Return the candidate tree to the champion before each iteration.

        Run 5 left `mmq.cu` modified by an authoring attempt that never passed, so the
        next iteration's worktree ground-truth check would have been satisfied by that
        leftover rather than by anything the planner did. Resets to HEAD, not to a
        fixed commit, so a kept patch stays on the champion branch.
        """
        subprocess.run(["git", "-C", str(args.worktree), "reset", "--hard", "HEAD"],
                       capture_output=True, text=True, timeout=600, check=False)
        subprocess.run(["git", "-C", str(args.worktree), "clean", "-fd",
                        "ggml/", "src/"],
                       capture_output=True, text=True, timeout=600, check=False)

    def commit(hypothesis, paths, comparison):
        return archive.keep(
            args.worktree, branch="HEAD",
            message=(f"{hypothesis.mechanism_id}: {comparison.effect * 100:+.3f}% "
                     f"on {comparison.surface} over {comparison.pairs} pairs"),
            paths=tuple(paths))

    def gpu_reading(outcomes=()) -> dict:
        """Held versus busy. Both halves, or the number means nothing.

        Held comes from the claim, busy from the comparisons that actually ran. The
        superseded loop held the MI210 for 1.403 hours across its entire life while
        compiling for 29.0, and nothing reported it -- because the surface reported
        iterations and receipts and had no number for "am I using what I hold".
        """
        if claim_started is None:
            return {}
        held = time.time() - claim_started
        busy = sum(float((o.comparison.to_dict() or {}).get("device_seconds") or 0.0)
                   for o in outcomes if o.comparison is not None)
        return {
            "claim_held_s": round(held, 1),
            "device_seconds_under_load": round(busy, 1),
            "gpu_seconds_idle_while_claimed": round(max(0.0, held - busy), 1),
            "idle_fraction_while_claimed": (
                round(max(0.0, 1.0 - busy / held), 4) if held > 0 else None),
        }

    def publish(state: str, outcomes=(), gpu=None, hotspot_rows=(),
                step: str | None = None) -> None:
        """A loop that only reports when it succeeds looks identical to a stuck one."""
        status.write(
            args.store, state=state, epoch=epoch, campaign_id="ak-loop",
            anchor_commit=anchor_commit, surface=args.surface, pairs=args.pairs,
            noise_floor_pct=floor,
            outcomes=[o.to_attempt() for o in outcomes],
            iterations_planned=args.iterations, step=step,
            champion_head=_git(args.worktree, "rev-parse", "HEAD"),
            gpu=gpu if gpu is not None else gpu_reading(outcomes),
            hotspots=[row.to_dict() for row in hotspot_rows])

    latest: list = []

    def _remember(rows) -> None:
        latest[:] = list(rows)
        publish("running", latest, hotspot_rows=hotspot_rows)

    claim_started = None
    publish("starting")
    started = time.time()
    with claim.hold() as receipt:
        claim_started = time.time()
        print(f"claim     held on {receipt['device_id']}\n")
        try:
            # The SAME surface the A/B will measure. Profiling decode and then
            # measuring prefill aims every hypothesis at a route the instrument
            # cannot see.
            hotspot_rows = hotspots.profile(
                args.anchor_build / "bin" / "llama-bench", args.model,
                pp=pp, tg=tg)
            print(f"profile   {len(hotspot_rows)} hotspots; top: "
                  f"{hotspot_rows[0].signature[:60] if hotspot_rows else '(none)'}")
        except hotspots.ProfileFailed as exc:
            hotspot_rows = []
            print(f"profile   UNAVAILABLE ({exc}); the planner is told so rather "
                  f"than left to guess")

        publish("running", hotspot_rows=hotspot_rows)
        try:
            outcomes = loop.run(
                planner=planner, critic=critic, build_context=build_context,
                measure=measure, gate=gate, commit=commit,
                store_root=args.store, epoch=epoch, campaign_id="ak-loop",
                iterations=args.iterations, reset=reset_tree,
                on_iteration=_remember,
                on_step=lambda label: publish("running", latest,
                                              hotspot_rows=hotspot_rows,
                                              step=label))
        except BaseException:
            # A crashed loop must SAY it crashed. Going quiet reads as "slow".
            publish("failed", hotspot_rows=hotspot_rows)
            raise

    elapsed = time.time() - started
    publish("complete", outcomes, hotspot_rows=hotspot_rows)
    kept = sum(1 for outcome in outcomes if outcome.status == "kept")
    measured = sum(1 for outcome in outcomes
                   if outcome.status in {"kept", "measured_null"})
    print(f"\n{len(outcomes)} iterations in {elapsed / 60:.1f} min: "
          f"{measured} reached a measurement, {kept} kept")
    for index, outcome in enumerate(outcomes, start=1):
        effect = (f"{outcome.comparison.effect * 100:+.3f}%"
                  if outcome.comparison else "—")
        print(f"  {index:>2}. {outcome.status:<22} {effect:>10}  "
              f"{outcome.hypothesis.mechanism_id if outcome.hypothesis else ''}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "loop-run.json").write_text(json.dumps({
            "schema": "epyc.autokernel.loop_run.v1",
            "epoch": epoch, "anchor_commit": anchor_commit,
            "surface": args.surface, "pairs": args.pairs,
            "noise_floor_pct": floor, "elapsed_s": round(elapsed, 1),
            "iterations": [outcome.to_attempt() for outcome in outcomes],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out / 'loop-run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
