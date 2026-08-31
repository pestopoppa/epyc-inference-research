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
import signal
import subprocess
import sys
import time

from ..controller import build_recipe, workload_contract
from . import (actors, anchor, archive, bench, claim, gates, hotspots, loop, pipeline,
               pool, status)

#: Single-pair p95, measured 2026-08-28 over n=20 alternating A/A pairs. Prefill is
#: the cheaper surface to detect on; decode has heavier tails.
SINGLE_PAIR_P95 = {"pp512": 2.175, "tg128": 3.452}


def noise_floor_pct(surface: str, pairs: int) -> float:
    """The bar for THIS run, scaled to the pairs actually being run.

    This was a dict of constants computed at 5 pairs, so `--pairs 9` still enforced
    the 5-pair bar -- 1.544% on decode where the measured 9-pair floor is 1.175%, a
    bar 31% higher than the instrument needs. Conservative rather than unsafe, but it
    throws away the sensitivity the extra pairs were bought for.

    Returns the MAX of two bounds, because neither dominates:

      * sigma/sqrt(n), the parametric bound. Conservative where the tail is light.
      * the exhaustively MEASURED floor for that pair count (`bench.MEASURED_FLOOR_PCT`).

    Decode does not average down at sqrt(n): its measured floor goes 3.452 -> 1.502 (5)
    -> 1.175 (9), while sqrt(n) predicts 1.544 -> 1.151. So at 9 pairs the parametric
    bound sits BELOW what the instrument actually resolves, and using it alone would let
    pure noise clear the bar. The guard test caught exactly this.

    For a pair count with no measured row, the largest measured row at or below it is
    used -- more pairs only ever lower the floor, so that is the conservative choice.
    """
    pairs = max(1, pairs)
    parametric = SINGLE_PAIR_P95[surface] / (pairs ** 0.5)
    rows = bench.MEASURED_FLOOR_PCT[surface]
    usable = [count for count in rows if count <= pairs]
    measured = rows[max(usable)] if usable else rows[min(rows)]
    return max(parametric, measured)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600).stdout.strip()


def prior_experiments(args, epoch: str) -> list[dict]:
    """The history the planner gets, and the one place `-A3` is turned on.

    A named function rather than three lines inside `build_context`, because the CLI
    flag existing and the flag REACHING the store are different facts, and only one
    of them was testable inline. A mutation that parsed `--rank-prior-experiments` and
    then recalled with the authority hardcoded off passed every test written against
    the parser; this is the seam that catches it.
    """
    return archive.recall(args.store, epoch=epoch,
                          ranking_authorized=args.rank_prior_experiments)


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
    parser.add_argument("--iterations", type=int, default=10,
                        help="0 means run CONTINUOUSLY until stopped: drop a STOP "
                             "file in the store, or send SIGTERM/SIGINT. Either is "
                             "honoured at the next iteration boundary, so a lane "
                             "holding the device finishes and publishes first.")
    parser.add_argument("--pairs", type=int, default=bench.MIN_PAIRS)
    parser.add_argument("--surface", choices=("pp512", "tg128"), default="pp512")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="prove the wiring without a provider call or a build")
    # `P-AK-SEARCH-1-A3` (RATIFIED 2026-08-31) narrows denial 4 to permit epoch-scoped
    # ranking. Off by default and named on the command line, so an ordering that
    # influenced a run is attributable to a flag someone typed. It grants ranking and
    # nothing else -- banking, composition, readiness contribution and promotion are
    # untouched, and the campaign still derives its own thresholds.
    parser.add_argument("--rank-prior-experiments", action="store_true",
                        help="P-AK-SEARCH-1-A3: order recalled records by merit "
                             "instead of recency, cross-epoch magnitudes redacted")
    # ---- concurrency (DRAFT). 1 is today's sequential path, unchanged. ----------
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent lanes; 1 (default) is the sequential path "
                             "and reaches none of the pooled code")
    parser.add_argument("--worker-root", type=Path, default=pool.WORKER_ROOT,
                        help="parent of the per-lane detached worktrees")
    parser.add_argument("--worker-build-root", type=Path,
                        default=pool.WORKER_BUILD_ROOT,
                        help="parent of the per-lane candidate build directories")
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
    floor = noise_floor_pct(args.surface, args.pairs)
    print(f"surface   {args.surface}, {args.pairs} alternating pairs, "
          f"noise floor {floor:.3f}%")

    if args.dry_run:
        print("\nDRY RUN — wiring proven, nothing spent.")
        return 0

    planner = actors.CodexPlanner(workspace=args.worktree)
    critic = actors.CodexCritic(workspace=args.worktree)

    #: Wall seconds per phase, accumulated across the run. The loop is ~84% NOT
    #: benchmarking (run 11: 11.9 of 75.1 min on device), and until now nothing
    #: recorded WHICH phase held the rest -- turn_recorded_at is stamped at write
    #: time, so per-iteration gaps all read 0.0. You cannot shorten what you have
    #: not measured.
    phase_seconds: dict[str, float] = {}
    phase_mark: list = [None, None]

    def note_phase(label: str) -> None:
        previous, started = phase_mark
        if previous is not None:
            phase_seconds[previous] = phase_seconds.get(previous, 0.0) + (
                time.monotonic() - started)
        phase_mark[0], phase_mark[1] = label, time.monotonic()

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
            "prior_experiments": prior_experiments(args, epoch),
            "inbox": read_inbox(),
        }

    #: The sequential run is one lane. Making it a `Worker` is what lets the gate,
    #: the measurement and the patch archive be written ONCE and used by both paths --
    #: a second copy of the build recipe wiring is a second thing to drift.
    solo = pipeline.Worker("solo", args.worktree, args.candidate_build)

    def keep_the_diff(worker, hypothesis) -> Path | None:
        """Preserve every candidate patch, kept or not.

        `reset_tree` returns the worktree to the champion before the next iteration, so
        a refused patch exists nowhere afterwards. Run 9 lost all ten: seven died on
        `MUL_MAT failed on ROCm0` and not one of them can now be reproduced, re-read or
        diagnosed. A negative written up without its diff is not evidence anyone can
        act on -- and the whole point of durable memory is that the next iteration does
        not re-derive what this one paid for.

        The filename carries the LANE. Mechanism ids repeat -- one bit-deposit rewrite
        of `vec_dot_q5_0_q8_1_impl` was proposed 38 times -- so with concurrent lanes a
        bare `<mechanism>.patch` is two lanes overwriting one file, which is run 9's
        lost-diffs defect returning by a different route.
        """
        diff = _git(worker.worktree, "diff")
        if not diff.strip():
            return None
        out = args.store / "patches"
        out.mkdir(parents=True, exist_ok=True)
        name = getattr(hypothesis, "mechanism_id", None) or "unnamed"
        target = out / f"{name}.{worker.name}.patch"
        target.write_text(diff + "\n", encoding="utf-8")
        return target

    def gate_for(worker):
        def gate(hypothesis, paths):
            # The diff first: a build that fails still leaves a patch worth reading,
            # and this is the last moment it exists on disk.
            keep_the_diff(worker, hypothesis)
            # Callables, so a failed build actually short-circuits: an eagerly
            # evaluated op_correctness ran the suite against a stale binary and blamed
            # this patch.
            #
            # `jobs=64, cpu_list="96-183"` is per BUILD, not per run. Under `--workers`
            # this is safe only because the build runs inside the serialized tail: two
            # concurrent 64-job builds would oversubscribe an 88-core lane and every
            # build time recorded during the overlap would be a measurement of
            # contention.
            return gates.run_all(
                lambda: gates.compiles(worker.worktree, worker.build_dir,
                                       cmake_defines=recipe.cmake_defines(),
                                       jobs=64, cpu_list="96-183"),
                lambda: gates.op_correctness(worker.build_dir),
            )
        return gate

    #: The anchor ADVANCES with the champion. It used to be a fixed binary while the
    #: candidate worktree accumulated every kept patch, so a reported effect was
    #: CUMULATIVE against original v9 rather than the marginal value of that patch --
    #: and a patch that made the champion WORSE still cleared the floor as long as the
    #: accumulated total did. Run 13 kept four that way: +5.574% marginal for the
    #: first, then -0.209%, -0.478% and -2.864%. The champion ended at +1.846% having
    #: been +5.574% after a single patch.
    #:
    #: "Screen against the champion so gains compound" was the requirement from the
    #: start. A static anchor asks "does the accumulated tree beat v9"; the question
    #: that decides a keep is "does THIS patch improve on the best we have".
    anchor_build = [args.anchor_build]
    # Run 19 advanced twice while the status published the run's STARTING commit, so a
    # working anchor read as stuck. `epoch` still pins the start for comparability.
    current_anchor_commit = [anchor_commit]

    def measure_for(worker):
        def measure(hypothesis, paths):
            # The anchor build is SHARED across lanes and only ever read, so it needs
            # no per-lane copy; the candidate binary is per lane because each lane
            # built it from its own patch.
            return bench.compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor)
        return measure

    gate = gate_for(solo)
    measure = measure_for(solo)

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

    hotspot_rows: list = []

    def reprofile() -> None:
        """Re-derive the hotspots from the CURRENT champion.

        A profile names where the time goes in one binary. Once a patch is kept that
        binary no longer exists, and the accepted change moved the very distribution
        the next hypothesis should aim at. A continuous run that profiled once would
        spend hours aiming at a distribution it had already altered.
        """
        try:
            rows = hotspots.profile(anchor_build[0] / "bin" / "llama-bench",
                                    args.model, pp=pp, tg=tg)
        except hotspots.ProfileFailed as exc:
            print(f"profile   UNAVAILABLE ({exc}); the planner is told so rather than "
                  f"left to guess")
            return
        hotspot_rows[:] = rows
        print(f"profile   {len(rows)} hotspots; top: "
              f"{rows[0].signature[:60] if rows else '(none)'}")

    stopping = {"asked": False}

    def _ask_stop(signum, _frame) -> None:
        # Never abort mid-measurement: a killed A/B wastes the device time already
        # spent and leaves a half-written candidate. Flag it and let the boundary
        # handle it, exactly as the STOP file does.
        stopping["asked"] = True
        print(f"\nstopping  signal {signum} received — finishing the current "
              f"iteration, then winding down")

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _ask_stop)

    def should_stop() -> bool:
        return stopping["asked"] or pool.stop_requested(args.store)

    anchor_guard_seen: list = []

    def build_champion(dest: Path):
        """The loop's recipe, compiled AT the path used. Shared by promotion and guard."""
        return gates.compiles(args.worktree, dest, cmake_defines=recipe.cmake_defines(),
                              jobs=64, cpu_list="96-183")

    def verify_anchor() -> None:
        """Prove the promoted binary IS the champion; `RunAborted` if not. Runs in the
        serialized tail holding the claim: `commit` is called inside `tail_session`."""
        def keep_verdict(verdict) -> None:
            # Both outcomes, before any abort raises: store + status, so the dashboard
            # says WHY a run stopped and the check is auditable after the fact.
            archive.record(args.store, verdict.to_attempt(), epoch=epoch,
                           recorded_at=loop._now(), campaign_id="ak-loop")
            anchor_guard_seen.append(verdict.to_dict())
            publish("running", latest, hotspot_rows=hotspot_rows)
            print(f"anchor    {verdict.detail}")

        anchor.verify(
            champion_commit=_git(args.worktree, "rev-parse", "HEAD"),
            anchor_build=anchor_build[0], noise_floor_pct=floor,
            on_verdict=keep_verdict, build=build_champion,
            compare=lambda promoted, fresh: bench.compare(
                bench.Arm("promoted_anchor", promoted / "bin" / "llama-bench"),
                bench.Arm("fresh_champion", fresh / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor),
            on_step=lambda label: publish("running", latest,
                                          hotspot_rows=hotspot_rows, step=label))

    def promote_anchor() -> None:
        """Advance the anchor by BUILDING the champion into the new slot, never by
        renaming a build directory in (CMake dirs are not relocatable). `pool` owns the
        mechanics so a test can EXECUTE them rather than grep for them."""
        anchor_build[0] = pool.promote_anchor(
            args.store, build=build_champion, recipe=recipe.to_dict(),
            champion_commit=_git(args.worktree, "rev-parse", "HEAD"))
        current_anchor_commit[0] = _git(args.worktree, "rev-parse", "HEAD")
        print(f"anchor    advanced to {anchor_build[0].name} — subsequent effects are "
              f"MARGINAL against this champion, not cumulative")
        # FIRST, before the loop draws any further work: nothing below is worth doing
        # against an anchor that is not the champion (run 18: 114 candidates, 6.5 h).
        verify_anchor()
        # The champion moved, so the profile that named the hotspots is stale: the
        # accepted patch changed the very distribution the next hypothesis should aim
        # at. Re-profiling here is what makes a long run keep aiming at the truth
        # rather than at wherever the time went hours ago.
        reprofile()
        dropped = pool.prune_anchor_generations(args.store, current=anchor_build[0])
        if dropped:
            print(f"anchor    pruned {len(dropped)} superseded generation(s) "
                  f"(~{201 * len(dropped)} MB)")

    def commit(hypothesis, paths, comparison):
        # `branch="HEAD"` is correct HERE and only here: the sequential run has the
        # champion branch checked out, so advancing HEAD advances the branch. A lane is
        # DETACHED, and the same call there would leave the commit unreferenced --
        # which is why the pooled path uses `pool.advance_champion` instead.
        head = archive.keep(
            args.worktree, branch="HEAD",
            message=pool.commit_message(hypothesis, comparison),
            paths=tuple(paths))
        # Only after the commit succeeds: an anchor advanced for a patch that did not
        # land would silently raise the bar for everything after it.
        promote_anchor()
        return head

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
            anchor_commit=current_anchor_commit[0], surface=args.surface, pairs=args.pairs,
            noise_floor_pct=floor,
            outcomes=[o.to_attempt() for o in outcomes],
            iterations_planned=args.iterations, step=step,
            champion_head=_git(args.worktree, "rev-parse", "HEAD"),
            anchor_guard=anchor_guard_seen[-1] if anchor_guard_seen else None,
            gpu=gpu if gpu is not None else gpu_reading(outcomes),
            hotspots=[row.to_dict() for row in hotspot_rows])

    latest: list = []

    def _remember(rows) -> None:
        latest[:] = list(rows)
        publish("running", latest, hotspot_rows=hotspot_rows)

    # ---------------------------------------------------------------------------
    # THE CONCURRENT PATH (DRAFT). Everything above is the sequential run and is
    # reached identically when `--workers 1`. Nothing below runs unless asked for.
    # ---------------------------------------------------------------------------

    def run_pooled() -> pool.PoolResult:
        """Drive the same loop across N detached lanes.

        WHAT IS PER-LANE THAT USED TO BE GLOBAL
          * the worktree and the candidate build directory (`pipeline.Worker`);
          * the planner and the critic, because each holds a workspace path;
          * the saved patch filename, which now carries the lane;
          * the phase clock -- `note_phase` keeps ONE (label, started) mark, so with
            several lanes the interval is charged to whichever lane wrote last. The
            pooled path uses `pool.PhaseClock`, whose totals are LANE-seconds and can
            legitimately sum to more than the wall clock.

        WHAT STAYS GLOBAL
          * the GPU claim: one `flock` for the process. A second `hold()` on a second
            descriptor in this same process would refuse itself.
          * the anchor build, which is only ever read;
          * `build_context`, `args.store` and the status file. `record` and the status
            publish both happen under the pipeline's outcomes lock, so they are
            serialized -- at the cost of a slow status write briefly blocking every
            lane's recording.
          * `epoch`: pinned to the champion the run STARTED from, which is what makes
            the archive rows comparable across the run.
        """
        def record_pooled(outcome) -> None:
            archive.record(args.store, outcome.to_attempt(), epoch=epoch,
                           recorded_at=loop._now(), campaign_id="ak-loop")
            latest.append(outcome)
            publish("running", latest, hotspot_rows=hotspot_rows)

        def step_pooled(worker_name: str, label: str) -> None:
            # The step line names the lane: an unattributed "building and gating" on a
            # pooled run says nothing about which of N lanes is where.
            publish("running", latest, hotspot_rows=hotspot_rows,
                    step=f"[{worker_name}] {label}")

        def commit_pooled(worker, hypothesis, paths, comparison) -> str:
            """Advance the champion branch, then the ANCHOR.

            The sequential path promotes the anchor inside its own `commit`; the
            pooled path has its own commit and did not, which would have left the
            anchor static across every lane -- reproducing run 13's defect (cumulative
            effects reported as marginal, a -2.864% regression committed as a keep) at
            seven times the rate.

            The anchor is BUILT from the champion tree, not taken from this lane: a
            lane's build directory is not relocatable, and the champion tree is what
            `advance_champion` just reset onto the accepted commit. Other lanes are
            mid-formation against the old base and are refused as `superseded` on their
            next tail entry -- their candidates were never built on this champion.
            """
            head = pool.advance_champion(worker, hypothesis, paths, comparison,
                                         champion_tree=args.worktree)
            promote_anchor()
            return head

        return pool.drive(
            commit=commit_pooled,
            workers=pool.provision(args.workers, champion_tree=args.worktree,
                                   root=args.worker_root,
                                   build_root=args.worker_build_root,
                                   execute=True),
            make_planner=lambda worker: actors.CodexPlanner(workspace=worker.worktree),
            make_critic=lambda worker: actors.CodexCritic(workspace=worker.worktree),
            build_context=build_context, make_gate=gate_for,
            make_measure=measure_for, record=record_pooled,
            iterations=(args.iterations or None), should_stop=should_stop,
            champion_tree=args.worktree,
            on_step=step_pooled)

    claim_started = None
    publish("starting")
    started = time.time()
    with claim.hold() as receipt:
        claim_started = time.time()
        print(f"claim     held on {receipt['device_id']}\n")
        # Profiles the CURRENT anchor on the SAME surface the A/B will measure, and
        # is re-run whenever a keep advances the champion.
        reprofile()

        publish("running", hotspot_rows=hotspot_rows)
        pooled: pool.PoolResult | None = None
        try:
            if args.workers > 1:
                pooled = run_pooled()
                outcomes = pooled.outcomes
            else:
                outcomes = loop.run(
                    planner=planner, critic=critic, build_context=build_context,
                    measure=measure, gate=gate, commit=commit,
                    store_root=args.store, epoch=epoch, campaign_id="ak-loop",
                    iterations=args.iterations, reset=reset_tree,
                    should_stop=should_stop,
                    on_iteration=_remember,
                    on_step=lambda label: (note_phase(label),
                                           publish("running", latest,
                                                   hotspot_rows=hotspot_rows,
                                                   step=label)) and None)
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
    if pooled is not None:
        # The number that decides the lane count: once the tail approaches the wall
        # clock every extra lane only queues on it.
        print(f"pool      {args.workers} lanes, tail {pooled.tail_seconds / 60:.1f} "
              f"min of {pooled.wall_seconds / 60:.1f} wall, "
              f"{pooled.superseded} superseded")
    for index, outcome in enumerate(outcomes, start=1):
        effect = (f"{outcome.comparison.effect * 100:+.3f}%"
                  if outcome.comparison else "—")
        print(f"  {index:>2}. {outcome.status:<22} {effect:>10}  "
              f"{outcome.hypothesis.mechanism_id if outcome.hypothesis else ''}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        body = {
            "schema": "epyc.autokernel.loop_run.v1",
            "epoch": epoch, "anchor_commit": anchor_commit,
            "surface": args.surface, "pairs": args.pairs,
            "noise_floor_pct": floor, "elapsed_s": round(elapsed, 1),
            "workers": args.workers,
            "iterations": [outcome.to_attempt() for outcome in outcomes],
            "phase_seconds": {k: round(v, 1) for k, v in sorted(
                phase_seconds.items(), key=lambda kv: -kv[1])},
        }
        if pooled is not None:
            # The pooled block REPLACES `phase_seconds` rather than sitting beside it:
            # the sequential accumulator is never written on this path, so leaving it
            # at {} beside a populated pooled block invites reading the empty one.
            pooled_body = pooled.to_dict(workers=args.workers)
            body["phase_seconds"] = pooled_body.pop("phase_lane_seconds")
            body["phase_seconds_are_lane_seconds"] = True
            body["pool"] = pooled_body
        (args.out / "loop-run.json").write_text(json.dumps(body, indent=2),
                                                encoding="utf-8")
        print(f"\nwrote {args.out / 'loop-run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
