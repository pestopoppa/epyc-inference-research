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
import shutil
import subprocess
import sys
import time

from ..controller import (anchor_integrity, build_recipe, inbox, rung_confirm,
                          workload_contract)
from . import (accumulate, actors, anchor, archive, bench, champion, claim, gates,
               hotspots, loop, serving,
               pipeline, pool, production, status)


def noise_floor_pct(surface: str, pairs: int, model: Path | str,
                    store: Path | None = None) -> float | None:
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

    None -- never a borrowed or guessed number -- when the surface is UNCALIBRATED
    (`bench.floor_rows`): no built-in row and no store-written A/A calibration. The
    run still measures and records on such a surface, but every comparison carries
    `decisive: None` and `refuse_uncalibrated_keep` blocks the commit path.

    `model` keys the lookup alongside the surface (§5.2): floors are workload
    properties, and a second rung must never inherit the first rung's floor.
    """
    pairs = max(1, pairs)
    rows = bench.floor_rows(surface, model, store)
    if rows is None:
        return None
    parametric = rows[1] / (pairs ** 0.5)
    measured = rows[max(count for count in rows if count <= pairs)]
    return max(parametric, measured)


def refuse_uncalibrated_keep(surface: str, calibrated: bool, comparison) -> None:
    """The commit path's OWN check, independent of `Comparison.decisive`.

    `iterate` only calls commit when decisive is truthy, so this looks redundant --
    it exists because the historical defect was precisely a comparison object whose
    decisive read True off a floor nobody had calibrated. The commit path re-derives
    the refusal from the run-level calibration fact, so a doctored or stale
    comparison cannot advance the champion on an uncalibrated surface.
    """
    if not calibrated or comparison.decisive is not True:
        raise loop.RunAborted(f"keep refused on {surface}: " + (
            "UNCALIBRATED surface — run --calibrate-surface" if not calibrated
            else "comparison is not decisive"))


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


def calibrate(args, run=subprocess.run) -> int:
    """A/A bootstrap-calibrate `--surface` into the store, then exit.

    The METHOD lives in exactly one place -- `scripts/benchmark/
    autokernel_aa_campaign.py`, the 2026-08-29 D8 instrument-characterisation
    campaign (anchor against ANCHOR so the true effect is zero by construction;
    three conditions, SETTLED / PREHEATED / POST_BUILD, isolating device settling
    from host load; floor bootstrapped over fresh pairs via
    `bench.bootstrap_floor`). This mode only points that instrument at the store:
    `--write-calibration` makes it write `calibration/<surface>.json` with the
    floor rows plus full provenance (all three condition records, model, anchor
    commit), which is precisely the file `bench.floor_rows` reads and without
    which this surface refuses decisive keeps. Two copies of the method would
    drift; the run gets a MODE, the method keeps its one home.
    """
    script = Path(__file__).resolve().parents[3] / "benchmark" / "autokernel_aa_campaign.py"
    return run([sys.executable, str(script), "--surface", args.surface,
                "--pairs", str(args.calibrate_surface),
                "--anchor-build", str(args.anchor_build),
                "--worktree", str(args.worktree), "--model", str(args.model),
                "--out", str(args.store / "calibration"
                             / f"aa-{args.surface}.{args.model.stem}"),
                "--write-calibration", str(args.store)]).returncode


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
    parser.add_argument("--surface", choices=tuple(bench.SURFACES), default="pp512")
    parser.add_argument("--calibrate-surface", type=int, metavar="N", default=None,
                        help="A/A bootstrap-calibrate --surface: N pairs x 3 "
                             "conditions (D8 method); floor lands in the store")
    # ---- the two-rung screen/confirm keep gate (§5.3, operator-approved D1-D6).
    # OFF unless --confirm-model is given: single-rung behavior is bit-identical
    # without it, so the running run's semantics are untouched until the boundary
    # that enables it.
    parser.add_argument("--confirm-model", type=Path, default=None,
                        help="production-shaped confirm rung: a screen keep is a "
                             "KEEP_CANDIDATE until it survives this model's "
                             "confirm surfaces (D1); headline moves to this rung")
    parser.add_argument("--confirm-pairs", type=int,
                        default=rung_confirm.DEFAULT_PAIRS,
                        help="pairs per confirm surface (D3; 5 = calibrated k=5 row)")
    parser.add_argument("--confirm-surfaces",
                        default=",".join(rung_confirm.DEFAULT_SURFACES),
                        help="comma-separated confirm gate surfaces (D2)")
    # R23-43: the SERVING keep gate. When set, a screen keep must ALSO improve serving
    # throughput on llama-server under the champion's canonical recipe -- the only
    # performance that matters (operator 2026-09-04). Supersedes the bench confirm rung.
    parser.add_argument("--serving-recipe", type=Path, default=None,
                        help="canonical serving recipe JSON; a keep must improve serving "
                             "throughput under it (llama-server), not just the bench screen")
    parser.add_argument("--serving-pairs", type=int, default=5,
                        help="paired serving A/B runs per bundle at the serving gate")
    parser.add_argument("--fire-multiple", type=float, default=2.5,
                        help="R23-44: run the serving gate once the accumulator's compounded "
                             "bench gain over the champion of record reaches this multiple of "
                             "the serving floor (operator: 2-3x; default 2.5)")
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
    parser.add_argument("--planner-model", default=actors.PLANNER_DEFAULT.model,
                        help="planner/author model; claude-* routes via the claude CLI, "
                             "anything else via codex (default: %(default)s)")
    parser.add_argument("--planner-effort", default=actors.PLANNER_DEFAULT.effort)
    parser.add_argument("--critic-model", default=actors.CRITIC_DEFAULT.model,
                        help="critic model, both passes (default: %(default)s)")
    parser.add_argument("--critic-effort", default=actors.CRITIC_DEFAULT.effort)
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

    pp, tg, ubatch = bench.SURFACES[args.surface]
    if args.calibrate_surface:
        return calibrate(args)
    floor = noise_floor_pct(args.surface, args.pairs, args.model, store=args.store)
    calibrated = floor is not None
    print(f"surface   {args.surface}, {args.pairs} alternating pairs, "
          + (f"noise floor {floor:.3f}%" if calibrated else
             "UNCALIBRATED — decisive=None on every comparison; keeps refused"))
    # §5.3: configured once at startup, refused loudly here if misconfigured -- a
    # confirm rung that is not production-shaped must never gate a keep.
    confirm = None if args.confirm_model is None else rung_confirm.configure(
        model=args.confirm_model, pairs=args.confirm_pairs,
        surfaces=args.confirm_surfaces, store=args.store, screen_census=census,
        known_surfaces=tuple(bench.SURFACES),
        floor_for=lambda s: noise_floor_pct(s, args.confirm_pairs,
                                            args.confirm_model, store=args.store))
    if confirm is not None:
        print(f"confirm   {confirm.describe()}")
    serving_recipe = None
    serving_floor_pct = None
    if args.serving_recipe is not None:
        serving_recipe = serving.Recipe.load(args.serving_recipe)
        floor_path = (args.store / f"serving-floor.{serving_recipe.name}.json")
        if floor_path.is_file():
            serving_floor_pct = json.loads(floor_path.read_text()).get("floor_pct")
        print(f"serving   {serving_recipe.describe()} — keep gate on llama-server; "
              + (f"floor {serving_floor_pct}%" if serving_floor_pct is not None
                 else "UNCALIBRATED (keeps refused until the serving floor is calibrated)"))
    planner_backend = actors.backend_for(args.planner_model, args.planner_effort)
    critic_backend = actors.backend_for(args.critic_model, args.critic_effort)
    print(f"actors    planner={planner_backend.describe()}  "
          f"critic={critic_backend.describe()}")
    # D4: with the two-rung gate on, the champion-vs-production headline is measured
    # on the confirm rung -- the standing +17.9% was the screen shape, which is the
    # "headline must be the production recipe" defect. Floor re-keyed to that model.
    headline_model = args.confirm_model or args.model
    headline_floor = floor if args.confirm_model is None else noise_floor_pct(
        args.surface, args.pairs, headline_model, store=args.store)

    if args.dry_run:
        print("\nDRY RUN — wiring proven, nothing spent.")
        return 0

    def build_context() -> dict:
        return {
            "program": loop.PROGRAM.read_text(encoding="utf-8"),
            "kernel_hotspots": [row.to_dict() for row in hotspot_rows],
            "prior_experiments": prior_experiments(args, epoch),
            # Re-read EVERY iteration, never cached at startup, and hardened so one
            # unreadable file cannot kill the run through the breaker (R22-6): the
            # rationale for both lives on `controller.inbox.read_inbox`'s docstring.
            "inbox": inbox.read_inbox(args.store / "inbox"),
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
    # R23-44 two-tier champion (operator 2026-09-04): the anchor above is the ACCUMULATOR,
    # advancing on every bench keep so keeps compound. The CHAMPION OF RECORD is the last
    # commit a serving gate DEMONSTRATED, the one the headline shows and a promotion would
    # ship. Its build is the serving A-arm and must survive the accumulator's own pruning,
    # so it lives in a `cor-build` slot -- prune_anchor_generations only touches `anchor-gen-*`.
    cor_commit = [anchor_commit]
    cor_slot = args.store / "cor-build"
    cor_build = [args.anchor_build]
    accum_policy = accumulate.AccumulatorPolicy(fire_multiple=args.fire_multiple)
    bundle = [accumulate.Bundle(champion_of_record=anchor_commit, tip=anchor_commit)]

    def snapshot_cor(from_build: Path) -> None:
        """Copy the champion-of-record's built binaries into the protected `cor-build` slot.
        A copy, not a rebuild: llama-server/llama-bench dlopen their ggml libs from their own
        bin/ via LD_LIBRARY_PATH (serving/bench set it), so the binaries run from a copy and
        no CMakeCache is needed -- and a copy cannot drift from the build the loop measured."""
        shutil.rmtree(cor_slot, ignore_errors=True)
        shutil.copytree(from_build, cor_slot)
        cor_build[0] = cor_slot

    def measure_for(worker):
        def measure(hypothesis, paths):
            # The anchor build is SHARED across lanes and only ever read, so it needs
            # no per-lane copy; the candidate binary is per lane because each lane
            # built it from its own patch.
            return bench.compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor,
                surface=args.surface, ubatch=ubatch, calibrated=calibrated)
        return measure

    def confirm_measure(worker):
        """The confirm rung's A/B for one keep-candidate (§5.3): same arms, the
        production-shaped model, the confirm surface's own keyed floor."""
        def measure(surface, floor_pct):
            cpp, ctg, cub = bench.SURFACES[surface]
            return bench.compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.confirm_model, pp=cpp, tg=ctg, pairs=args.confirm_pairs,
                noise_floor_pct=floor_pct, surface=surface, ubatch=cub,
                calibrated=floor_pct is not None)
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

    def build_champion(dest: Path, targets: tuple = gates.DEFAULT_TARGETS):
        """The loop's recipe, compiled AT the path used. Shared by promotion and guard —
        at DIFFERENT widths (R22-7). `pool.promote_anchor` calls this with
        `gates.PROMOTION_TARGETS` (llama-server included: a champion that is not
        production-complete is not promotable — operator ruling, 2026-09-01), while
        the anchor guard's throwaway fresh build and its heal retry take the narrow
        default: the guard answers "is the anchor slot the champion", its digest
        hashes `bin/libggml-hip.so` alone, and paying server link time per keep for
        a binary nobody runs would buy nothing. Candidate lane builds (`gate_for`)
        stay narrow for the same reason at hundreds of iterations per run."""
        # R23-40 (2026-09-03): jobs=1, NOT 64. This recipe feeds BOTH the promoted
        # anchor and the guard's fresh comparison build, and `-j64` HIP builds of one
        # commit are NON-reproducible on this host -- three same-recipe builds of
        # 445e93a8 differed in every code section (.text/.hip_fatbin/.rodata), so the
        # digest guard aborted the run (Run-18 fault class). The build-path sections
        # are already excluded from the digest (R21-10), so this is genuine parallel-
        # build non-determinism. Serial build makes the promoted anchor and the fresh
        # guard build bit-identical. Cost is per-KEEP only (rare), never per-iteration:
        # lane candidate builds (`gate_for`) keep jobs=64. A future toolchain-flag fix
        # (hipcc determinism at -j64) could restore parallel anchor builds; filed R23-41.
        return gates.compiles(args.worktree, dest, cmake_defines=recipe.cmake_defines(),
                              jobs=1, cpu_list="96-183", targets=targets)

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
            # An excursion-flagged promotion still publishes (the anchor is
            # hash-proven), but the bundle must carry the session-health note.
            note=next((g["detail"] for g in anchor_guard_seen[-1:]
                       if g.get("excursion")), None),
            compare=lambda base, champ: bench.compare(
                bench.Arm("production_v9", base / "bin" / "llama-bench"),
                bench.Arm("champion", champ / "bin" / "llama-bench"),
                headline_model, pp=pp, tg=tg, pairs=args.pairs,
                noise_floor_pct=headline_floor, surface=args.surface, ubatch=ubatch,
                calibrated=headline_floor is not None),
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
            # 2026-09-06: OBJECT digest, not the linked .so. The compiler is reproducible
            # (0/379 objects ever differed); the linker is not (four distinct .so digests
            # for one commit aborted every keep on link noise). Objects prove identity.
            digest=anchor_integrity.object_digest,
            on_verdict=keep_verdict, build=build_champion,
            compare=lambda promoted, fresh: bench.compare(
                bench.Arm("promoted_anchor", promoted / "bin" / "llama-bench"),
                bench.Arm("fresh_champion", fresh / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor,
                surface=args.surface, ubatch=ubatch, calibrated=calibrated),
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
        # R23-44: the headline no longer publishes here. This is the ACCUMULATOR advancing;
        # the headline follows the CHAMPION OF RECORD, which advances only when a bundle
        # passes the serving gate (`accumulate_after_keep`). Between serving promotions the
        # dashboard headline correctly holds at the last serving-demonstrated champion.
        # The champion moved, so the profile that named the hotspots is stale: the
        # accepted patch changed the very distribution the next hypothesis should aim
        # at. Re-profiling here is what makes a long run keep aiming at the truth
        # rather than at wherever the time went hours ago.
        reprofile()
        dropped = pool.prune_anchor_generations(args.store, current=anchor_build[0])
        if dropped:
            print(f"anchor    pruned {len(dropped)} superseded generation(s) "
                  f"(~{201 * len(dropped)} MB)")

    def accumulate_after_keep(mechanism_id: str) -> None:
        """R23-44 compound-then-gate. The accumulator just advanced on a bench keep; batch it
        and, only when the bundle's compounded bench gain over the champion of record clears
        `fire_multiple` x the serving floor, spend the serving gate ONCE on the whole bundle.

        No serving_recipe -> no serving tier: the loop reverts to a pure bench-keep loop.

        On a serving win the champion of record advances to the accumulator tip and the
        headline follows it; on a divergence (bundle cleared bench, serving did not confirm)
        the champion of record HOLDS, the bundle is KEPT, and the divergence is journaled as
        planner evidence naming the bundled keeps (operator 2026-09-04)."""
        if serving_recipe is None:
            return
        head = _git(args.worktree, "rev-parse", "HEAD")
        # compounded bench: champion-of-record build (A) vs the just-advanced accumulator (B),
        # re-measured (never a product of marginal effects -- keeps interact) because this is
        # the number the fire threshold reads and the serving gate will be asked to confirm.
        comp = bench.compare(
            bench.Arm("champion_of_record", cor_build[0] / "bin" / "llama-bench"),
            bench.Arm("accumulator", anchor_build[0] / "bin" / "llama-bench"),
            args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor,
            surface=args.surface, ubatch=ubatch, calibrated=calibrated).to_dict()
        bundle[0].add_keep(mechanism_id, head, comp["effect"] * 100.0)
        thr = (f"{accum_policy.fire_threshold_pct(serving_floor_pct):.2f}"
               if serving_floor_pct is not None else "uncalibrated")
        print(f"accum     bundle {len(bundle[0].keeps)} keep(s), "
              f"{bundle[0].compounded_bench_pct:+.2f}% compounded bench vs champion of record "
              f"(serving gate fires at {thr}%)")
        if accumulate.decide_after_keep(bundle[0], serving_floor_pct,
                                        accum_policy) is not accumulate.Decision.FIRE_SERVING:
            return
        # The bundle is big enough for the ~3.5% serving floor to resolve -- spend the gate ONCE.
        sv_row = serving.compare(serving_recipe, cor_build[0], anchor_build[0],
                                 pairs=args.serving_pairs, floor_pct=serving_floor_pct)
        plan = accumulate.resolve(bundle[0], sv_row, accum_policy)
        status.write_json(
            args.store / "serving", f"bundle-{head[:12]}.json",
            {"outcome": plan["outcome"].value, "reason": plan["reason"],
             "bundled_keeps": list(bundle[0].keeps),
             "planner_evidence": plan.get("planner_evidence"), **sv_row}, prefix=".sv-")
        print(f"serving   {plan['reason']}")
        if plan["outcome"] is accumulate.Outcome.PROMOTE:
            # The champion of record advances to the accumulator tip. Snapshot its build into
            # the protected slot, publish the headline against it, and start a fresh bundle.
            cor_commit[0] = plan["new_champion_of_record"]
            snapshot_cor(anchor_build[0])
            publish_headline()
            bundle[0] = accumulate.Bundle(champion_of_record=head, tip=head)
        else:
            # DIVERGED + HOLD: hand the divergence to the planner as journal evidence so it can
            # revert/revise a bundled keep or re-aim; the champion of record and the bundle hold.
            archive.record(
                args.store,
                {"schema": "epyc.autokernel.attempt.v1", "campaign_id": "ak-loop",
                 "mechanism_id": f"serving-divergence-{head[:12]}", "status": "measured_divergence",
                 "hypothesis": plan["reason"], "planner_evidence": plan["planner_evidence"],
                 "serving": sv_row},
                epoch=epoch, recorded_at=loop._now(), campaign_id="ak-loop")

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

    def accumulator_state() -> dict | None:
        """The two-tier champion's live bundle, for the dashboard (R23-44): how many
        bench keeps have accumulated on the champion of record and how far their
        compounded gain has climbed toward the serving gate's fire threshold. None when
        there is no serving tier (no --serving-recipe)."""
        if serving_recipe is None:
            return None
        thr = (accum_policy.fire_threshold_pct(serving_floor_pct)
               if serving_floor_pct is not None else None)
        comp = bundle[0].compounded_bench_pct
        return {
            "champion_of_record": cor_commit[0],
            "accumulator_tip": bundle[0].tip,
            "keeps": list(bundle[0].keeps),
            "n_keeps": len(bundle[0].keeps),
            "compounded_bench_pct": round(comp, 3),
            "serving_floor_pct": serving_floor_pct,
            "fire_multiple": accum_policy.fire_multiple,
            "fire_threshold_pct": round(thr, 3) if thr is not None else None,
            "progress_fraction": (round(min(comp / thr, 1.0), 4)
                                  if thr and thr > 0 else None),
            "fires_next": bool(thr is not None and comp >= thr),
        }

    def publish(state: str, outcomes=(), gpu=None, hotspot_rows=(),
                step: str | None = None) -> None:
        """A loop that only reports when it succeeds looks identical to a stuck one."""
        status.write(
            args.store, state=state, epoch=epoch, campaign_id="ak-loop",
            anchor_commit=current_anchor_commit[0], surface=args.surface, pairs=args.pairs,
            noise_floor_pct=floor, model=str(args.model),
            outcomes=[o.to_attempt() for o in outcomes],
            iterations_planned=args.iterations, step=step,
            champion_head=_git(args.worktree, "rev-parse", "HEAD"),
            anchor_guard=anchor_guard_seen[-1] if anchor_guard_seen else None,
            accumulator=accumulator_state(),
            gpu=gpu if gpu is not None else gpu_reading(outcomes),
            hotspots=[row.to_dict() for row in hotspot_rows])

    latest: list = []

    def run_pooled() -> pool.PoolResult:
        """Drive the loop across N detached lanes. THE run path -- the sequential
        `loop.run` wiring was deleted 2026-08-31, and the `loop.run` seam itself on
        2026-09-01 (R21-7): `iterate` under `pipeline.run_pool` is the only loop.

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
            refuse_uncalibrated_keep(args.surface, calibrated, comparison)
            # R23-44: the BENCH confirm rung is the KEEP GATE -- a cheap, deterministic screen
            # a keep must clear to enter the accumulator (§5.3: one extra bench.compare per
            # confirm surface, in this same serialized tail; the veto lands the candidate as
            # keep_candidate, never kept). The SERVING gate is NO LONGER per-keep: it cannot
            # resolve a 1-3% keep against the ~3.5% serving floor, so it fires on the BUNDLE in
            # accumulate_after_keep once the compounded gain clears the floor.
            if confirm is not None:
                verdict = confirm.gate(hypothesis.mechanism_id, comparison,
                                       confirm_measure(worker))
                if not verdict["promoted"]:
                    raise loop.ConfirmVetoed(verdict["reason"])
            head = pool.advance_champion(worker, hypothesis, paths, comparison,
                                         champion_tree=args.worktree,
                                         branch=args.champion_branch)
            promote_anchor()
            # The accumulator advanced; batch this keep and, if the bundle now clears the
            # serving floor, spend the one serving gate that can advance the champion of record.
            accumulate_after_keep(hypothesis.mechanism_id)
            return head

        return pool.drive(
            commit=commit_pooled,
            workers=pool.provision(args.workers, champion_tree=args.worktree,
                                   champion_branch=args.champion_branch,
                                   root=args.worker_root,
                                   build_root=args.worker_build_root,
                                   execute=True),
            make_planner=lambda worker: actors.AgentPlanner(
                workspace=worker.worktree, backend=planner_backend),
            make_critic=lambda worker: actors.AgentCritic(
                workspace=worker.worktree, backend=critic_backend),
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
        # R23-44: snapshot the starting champion into the protected champion-of-record slot
        # BEFORE the accumulator can advance and prune. The serving gate reads cor_build as
        # its A-arm; without this snapshot the first accumulator prune could delete it.
        if serving_recipe is not None:
            snapshot_cor(args.anchor_build)
            print(f"cor       champion of record {cor_commit[0][:12]} -> cor-build "
                  f"(serving A-arm; headline follows serving-demonstrated advances only)")
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
                   if outcome.status in {"kept", "measured_null", "keep_candidate"})
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
