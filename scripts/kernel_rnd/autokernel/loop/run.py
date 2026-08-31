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
from . import (actors, anchor, archive, bench, champion, claim, gates, hotspots, loop,
               pipeline, pool, production, status)


def noise_floor_pct(surface: str, pairs: int) -> float:
    """The bar for THIS run, scaled to the pairs actually being run.

    This was a dict of constants computed at 5 pairs, so `--pairs 9` still enforced
    the 5-pair bar -- 1.544% on decode where the measured 9-pair floor is 1.175%, a
    bar 31% higher than the instrument needs. Conservative rather than unsafe, but it
    throws away the sensitivity the extra pairs were bought for.

    Returns the MAX of two bounds, because neither dominates:

      * sigma/sqrt(n), the parametric bound, seeded from the MEASURED single-pair p95
        (`bench.MEASURED_FLOOR_PCT[surface][1]` -- the same exhaustive A/A table, so
        there is exactly ONE copy of that number and nothing to drift; a byte-copy of
        the k=1 column used to live here with nothing enforcing agreement).
        Conservative where the tail is light.
      * the exhaustively MEASURED floor for that pair count, taking the largest
        measured row at or below it -- more pairs only ever lower the floor, so that
        is the conservative choice. `max(...)` never sees an empty sequence: the
        table carries a k=1 row (the parametric seed) and `pairs` is clamped to >= 1.

    Decode does not average down at sqrt(n): its measured floor goes 3.452 -> 1.502 (5)
    -> 1.175 (9), while sqrt(n) predicts 1.544 -> 1.151. So at 9 pairs the parametric
    bound sits BELOW what the instrument actually resolves, and using it alone would let
    pure noise clear the bar. The guard test caught exactly this.
    """
    pairs = max(1, pairs)
    rows = bench.MEASURED_FLOOR_PCT[surface]
    parametric = rows[1] / (pairs ** 0.5)
    measured = rows[max(count for count in rows if count <= pairs)]
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
    # THE single champion branch; the worktree must have it checked out at its tip or
    # the loop refuses to start (`champion.verify_startup`).
    parser.add_argument("--champion-branch", default=champion.CANONICAL_BRANCH)
    # Proceed (loudly) on a hand-built anchor that carries no provenance.json;
    # anchor-gen-* dirs never need or honour this.
    parser.add_argument("--allow-unverified-anchor", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10,
                        help="0 means run CONTINUOUSLY until stopped: drop a STOP "
                             "file in the store, or send SIGTERM/SIGINT. On a stop, "
                             "a lane still FORMING abandons at its next stage "
                             "boundary (no further planner/critic call is drawn); "
                             "the lane holding the serialized tail finishes "
                             "build/oracle/A-B/commit and publishes first.")
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
    # ---- concurrency. EVERY run is pooled; --workers 1 is a one-lane pool. The
    # separate sequential path was deleted 2026-08-31 once the pool owned the
    # consecutive-error breaker -- two run paths were two things to drift.
    parser.add_argument("--workers", type=int, default=pipeline.DEFAULT_WORKERS,
                        help="concurrent lanes (default: the measured tail-saturation "
                             "point; see pipeline.DEFAULT_WORKERS)")
    parser.add_argument("--worker-root", type=Path, default=pool.WORKER_ROOT,
                        help="parent of the per-lane detached worktrees")
    parser.add_argument("--worker-build-root", type=Path,
                        default=pool.WORKER_BUILD_ROOT,
                        help="parent of the per-lane candidate build directories")
    args = parser.parse_args(argv)

    # FIRST, before the claim, the census, even the dry run's wiring proof: the loop
    # optimises THE single champion branch or it does not start. See `champion` for
    # the 2026-08-31 incident this refusal exists to make unrepeatable.
    verified_head = champion.verify_startup(
        worktree=args.worktree, branch=args.champion_branch,
        anchor_build=args.anchor_build,
        allow_unverified_anchor=args.allow_unverified_anchor)
    print(f"champion  {args.champion_branch} @ {verified_head[:12]} — verified")

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

    def keep_the_diff(worker, hypothesis) -> Path | None:
        """Preserve every candidate patch, kept or not.

        `pool.reset_to_champion` returns a lane to the champion before each iteration, so
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
        # spent and leaves a half-written candidate. Flag it and let the stage
        # boundaries handle it, exactly as the STOP file does: forming lanes abandon
        # before their next actor call, the tail holder finishes and publishes.
        stopping["asked"] = True
        print(f"\nstopping  signal {signum} received — forming lanes abandon at "
              f"their next stage boundary; a lane holding the tail finishes its "
              f"measurement and publishes first")

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _ask_stop)

    def should_stop() -> bool:
        return stopping["asked"] or pool.stop_requested(args.store)

    anchor_guard_seen: list = []

    def build_champion(dest: Path):
        """The loop's recipe, compiled AT the path used. Shared by promotion and guard."""
        return gates.compiles(args.worktree, dest, cmake_defines=recipe.cmake_defines(),
                              jobs=64, cpu_list="96-183")

    def build_baseline(dest: Path, commit: str):
        """The frozen production kernel, built at most once PER FREEZE. Never in the
        production tree itself.

        `commit` is the LIVE-resolved freeze (`production.resolve_frozen`), not a
        pinned constant: after a promotion the headline must follow the newly frozen
        kernel, never stale v9. The build-source copy is checked against it before a
        single compiler is invoked -- a copy that has not followed the promotion would
        produce a binary published under the new freeze's name, which is the one way
        this headline can be quietly wrong. A refusal here is a `Verdict`, not an
        exception -- `production.refresh` turns it into a skipped refresh (the panel
        reads SUPERSEDED, naming why), and the run carries on.
        """
        head = _git(production.BASELINE_TREE, "rev-parse", "HEAD")
        if head != commit:
            return gates.Verdict("baseline-tree", False,
                                 f"{production.BASELINE_TREE} is at {head[:12]}, not "
                                 f"the frozen production kernel {commit[:12]}; "
                                 f"refresh the copy to follow the promotion")
        return gates.compiles(production.BASELINE_TREE, dest,
                              cmake_defines=recipe.cmake_defines(),
                              jobs=64, cpu_list="96-183")

    def publish_headline() -> None:
        """Refresh the dashboard headline for the champion that was just promoted.

        The champion arm is `anchor_build[0]` -- the slot `pool.promote_anchor` built
        from this commit and `anchor.verify` just proved holds the champion. No second
        build is paid for here, and nothing below is allowed to end the run.
        """
        outcome = production.refresh(
            store=args.store, champion_commit=current_anchor_commit[0],
            champion_build=anchor_build[0], build_baseline=build_baseline,
            compare=lambda base, champ: bench.compare(
                bench.Arm("production_v9", base / "bin" / "llama-bench"),
                bench.Arm("champion", champ / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor),
            on_step=lambda label: publish("running", latest,
                                          hotspot_rows=hotspot_rows, step=label))
        archive.record(args.store, outcome.to_attempt(), epoch=epoch,
                       recorded_at=loop._now(), campaign_id="ak-loop")
        print(f"headline  {outcome.reason}")

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
        # AFTER the guard, never before: a headline measured against a slot that does
        # not hold the champion is the same void number, published where the operator
        # reads it. The guard aborts, so reaching this line means the arm is the
        # champion.
        publish_headline()
        # The champion moved, so the profile that named the hotspots is stale: the
        # accepted patch changed the very distribution the next hypothesis should aim
        # at. Re-profiling here is what makes a long run keep aiming at the truth
        # rather than at wherever the time went hours ago.
        reprofile()
        dropped = pool.prune_anchor_generations(args.store, current=anchor_build[0])
        if dropped:
            print(f"anchor    pruned {len(dropped)} superseded generation(s) "
                  f"(~{201 * len(dropped)} MB)")

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

    def run_pooled() -> pool.PoolResult:
        """Drive the loop across N detached lanes. THE run path -- the sequential
        `loop.run` wiring was deleted 2026-08-31 (it survives only as a test seam).

        WHAT IS PER-LANE
          * the worktree and the candidate build directory (`pipeline.Worker`);
          * the planner and the critic, because each holds a workspace path;
          * the saved patch filename, which carries the lane;
          * the phase clock -- `pool.PhaseClock`, whose totals are LANE-seconds and
            can legitimately sum to more than the wall clock.

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

            The promotion once lived only in the sequential path's commit, which
            would have left the anchor static across every lane -- reproducing run
            13's defect (cumulative effects reported as marginal, a -2.864%
            regression committed as a keep) at seven times the rate.

            The anchor is BUILT from the champion tree, not taken from this lane: a
            lane's build directory is not relocatable, and the champion tree is what
            `advance_champion` just reset onto the accepted commit. Other lanes are
            mid-formation against the old base and are refused as `superseded` on their
            next tail entry -- their candidates were never built on this champion.
            """
            head = pool.advance_champion(worker, hypothesis, paths, comparison,
                                         champion_tree=args.worktree,
                                         branch=args.champion_branch)
            promote_anchor()
            return head

        return pool.drive(
            commit=commit_pooled,
            workers=pool.provision(args.workers, champion_tree=args.worktree,
                                   champion_branch=args.champion_branch,
                                   root=args.worker_root,
                                   build_root=args.worker_build_root,
                                   execute=True),
            make_planner=lambda worker: actors.CodexPlanner(workspace=worker.worktree),
            make_critic=lambda worker: actors.CodexCritic(workspace=worker.worktree),
            build_context=build_context, make_gate=gate_for,
            make_measure=measure_for, record=record_pooled,
            iterations=(args.iterations or None), should_stop=should_stop,
            champion_tree=args.worktree, branch=args.champion_branch,
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
        try:
            pooled = run_pooled()
            outcomes = pooled.outcomes
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
        # `phase_seconds` are LANE-seconds (`pool.PhaseClock`): with N lanes they can
        # legitimately sum to more than the wall clock, and the flag beside them says
        # so to any reader that predates the pooled accounting.
        pooled_body = pooled.to_dict(workers=args.workers)
        body = {
            "schema": "epyc.autokernel.loop_run.v1",
            "epoch": epoch, "anchor_commit": anchor_commit,
            "surface": args.surface, "pairs": args.pairs,
            "noise_floor_pct": floor, "elapsed_s": round(elapsed, 1),
            "workers": args.workers,
            "iterations": [outcome.to_attempt() for outcome in outcomes],
            "phase_seconds": pooled_body.pop("phase_lane_seconds"),
            "phase_seconds_are_lane_seconds": True,
            "pool": pooled_body,
        }
        (args.out / "loop-run.json").write_text(json.dumps(body, indent=2),
                                                encoding="utf-8")
        print(f"\nwrote {args.out / 'loop-run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
