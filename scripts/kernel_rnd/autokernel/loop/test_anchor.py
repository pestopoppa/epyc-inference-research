#!/usr/bin/env python3
"""The promotion A/A guard, mutation-tested against the defect it exists to catch.

This guard exists BECAUSE verification failed, so its own verification is written to a
higher bar than "the test passes". Every assertion below names the number it would read
if the thing under test were broken, and the two directions are tested separately:

  * the FAILING direction is the literal run-18 signature -- a promoted anchor whose
    bench result differs from a fresh build of the champion by ~9.5%. The guard must
    fire AND the run must stop drawing work.
  * the PASSING direction is two builds of one commit, effect ~0. It must NOT fire, or
    the guard aborts every healthy run and gets turned off within a day.

Nothing here touches a GPU, a ROCm toolchain, an API key or a real build: `anchor.verify`
takes `build` and `compare` as injected callables, and `pool.promote_anchor` takes the
build too, so the whole promotion path is executable in a temporary directory.
"""
from __future__ import annotations

from pathlib import Path
import statistics as st
import tempfile
import threading
import unittest

from autokernel.loop import anchor, bench, gates, loop, pipeline, pool
from autokernel.loop import run as run_mod
from autokernel.loop.test_loop import drive_single_lane

#: The bar the run itself enforces on prefill at 5 pairs. Read from `run.noise_floor_pct`
#: rather than typed, so a change to the run's floor cannot leave this suite testing a
#: bar the loop no longer uses. It is 0.973% today.
FLOOR = run_mod.noise_floor_pct("pp512", 5,
                                f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf")

#: Run 18's post-promotion median effect (n=114), and the champion advance the promotion
#: had just kept. The guard must fire on either sign: the anchor being 9.5% too fast and
#: the anchor being 9.3% too slow are the same defect seen from two directions.
RUN_18_EFFECT_PCT = -9.539
CHAMPION_ADVANCE_PCT = 9.321

CHAMPION = "5ad3e36d" + "0" * 32


def _comparison(effect_pct, *, anchor_samples=None, candidate_samples=None,
                pairs=5, surface="pp512", drift=False):
    """A `Comparison` whose `effect` is COMPUTED from its own samples.

    The fixture therefore cannot lie about its own number -- if the samples do not
    produce the effect a test names, the test's assertion on `effect_pct` fails before
    anything about the guard is exercised.
    """
    left = list(anchor_samples or [100.0] * 9)
    right = list(candidate_samples
                 or [100.0 * (1 + effect_pct / 100.0)] * 9)
    return bench.Comparison(
        surface=surface, anchor_samples=left, candidate_samples=right,
        effect=st.median(right) / st.median(left) - 1.0,
        estimator="median_over_median", pairs=pairs, noise_floor_pct=FLOOR,
        residency={"invocations": 2 * pairs, "resident": 2 * pairs},
        anchor_drift_pct=bench.drift_pct(left) if drift else 0.0,
        candidate_drift_pct=bench.drift_pct(right) if drift else 0.0)


def _ramping_run18():
    """The run-18 effect on a DRIFTING pair of arms.

    -9.539% with a candidate arm climbing +4.975% across the run. `Comparison.decisive`
    is False for this -- the drift veto swallows it -- so a guard that asked
    `not comparison.decisive` instead of comparing the magnitude would read this as
    "indistinguishable" and pass. See `TheGuardMustNotLeanOnDecisive`.
    """
    return _comparison(RUN_18_EFFECT_PCT,
                       candidate_samples=[86.461 + i for i in range(9)], drift=True)


class _Recorder:
    """Collects everything the guard does, in order."""

    def __init__(self, comparison, *, build_passes=True, build_reason=""):
        self.comparison = comparison
        self.build_verdict = gates.Verdict(
            "compile", build_passes, build_reason or ("" if build_passes else "boom"))
        self.order: list[str] = []
        self.built_into: list[Path] = []
        self.cleaned: list[Path] = []
        self.compared: list[tuple[Path, Path]] = []
        self.verdicts: list[anchor.AnchorVerdict] = []
        self.steps: list[str] = []

    def clean(self, path):
        self.order.append("clean")
        self.cleaned.append(Path(path))

    def build(self, dest):
        self.order.append("build")
        self.built_into.append(Path(dest))
        return self.build_verdict

    def compare(self, left, right):
        self.order.append("compare")
        self.compared.append((Path(left), Path(right)))
        return self.comparison

    def on_verdict(self, verdict):
        self.order.append("verdict")
        self.verdicts.append(verdict)

    def run(self, **kwargs):
        return anchor.verify(
            champion_commit=CHAMPION, anchor_build=Path("/store/anchor-gen-002"),
            scratch_build=Path("/scratch/build-anchor-verify"),
            noise_floor_pct=FLOOR, build=self.build, compare=self.compare,
            clean=self.clean, on_verdict=self.on_verdict,
            on_step=self.steps.append, **kwargs)


# ------------------------------------------------------- the failing direction


class TheRun18DefectMustAbortTheRun(unittest.TestCase):
    """A promotion where the anchor binary is NOT the champion.

    Run 18 promoted 5ad3e36d at 11:03; the measured median went from -1.441% (n=16,
    best +0.060%) to -9.539% (n=114, best -5.642%) and the run continued for 6.5 hours.
    """

    def test_the_run_18_signature_aborts(self):
        recorder = _Recorder(_comparison(RUN_18_EFFECT_PCT))
        with self.assertRaises(loop.RunAborted) as caught:
            recorder.run()
        # BROKEN READS: no exception at all, and verdict.passed True.
        verdict = recorder.verdicts[0]
        self.assertFalse(verdict.passed)
        self.assertAlmostEqual(verdict.effect_pct, RUN_18_EFFECT_PCT, places=3,
                               msg="the fixture is not reproducing run 18's number")
        self.assertAlmostEqual(FLOOR, 0.973, places=3)
        self.assertIn("-9.539%", str(caught.exception))
        self.assertIn("NOT hold the champion", str(caught.exception))

    def test_the_opposite_sign_aborts_too(self):
        """A promoted anchor that is the PREVIOUS binary reads +9.321%, not -9.539%.

        An implementation that compared the signed effect against the floor would let
        this through: +9.321 <= 0.973 is False, but `effect <= floor` written without
        `abs` on a NEGATIVE-signed convention passes silently for one of the two signs.
        """
        recorder = _Recorder(_comparison(CHAMPION_ADVANCE_PCT))
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        self.assertAlmostEqual(recorder.verdicts[0].effect_pct, CHAMPION_ADVANCE_PCT,
                               places=3)

    def test_the_verdict_reaches_the_recorder_before_the_abort(self):
        """A guard whose failure is invisible after the fact is the shape of check that
        let run 18 happen. BROKEN READS: `recorder.verdicts == []` and the operator has
        only a traceback."""
        recorder = _Recorder(_comparison(RUN_18_EFFECT_PCT))
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        self.assertEqual(len(recorder.verdicts), 1)
        self.assertEqual(recorder.order, ["build", "compare", "verdict"])
        row = recorder.verdicts[0].to_attempt()
        self.assertEqual(row["status"], "anchor_mismatch")
        self.assertAlmostEqual(row["effect_fraction"], RUN_18_EFFECT_PCT / 100.0,
                               places=5)
        self.assertEqual(row["mechanism_id"], anchor.MECHANISM_ID)


class TheGuardMustNotLeanOnDecisive(unittest.TestCase):
    """`Comparison.decisive` is False whenever an arm is drifting, whatever the effect.

    Reusing it as "the two builds are indistinguishable" would have waved run 18's
    -9.539% straight through on any run whose arms happened to be moving -- a check that
    passes for a reason unrelated to the question it asks. This test constructs exactly
    that case and requires the guard to fire anyway.
    """

    def test_the_fixture_really_is_the_hole(self):
        comparison = _ramping_run18()
        self.assertAlmostEqual(comparison.effect * 100.0, RUN_18_EFFECT_PCT, places=3)
        self.assertTrue(comparison.drifting)
        self.assertFalse(comparison.decisive,
                         "this fixture is only interesting if `decisive` says False")

    def test_it_aborts_a_drifting_nine_percent_anyway(self):
        recorder = _Recorder(_ramping_run18())
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        # BROKEN READS: a guard written as `if comparison.decisive: raise` returns
        # normally here, having just certified a -9.539% anchor as the champion.
        self.assertFalse(recorder.verdicts[0].passed)


class AChampionThatWillNotBuildIsNotAPass(unittest.TestCase):
    """An unbuildable champion means the anchor is UNCHECKED, which is the state run 18
    was in for 6.5 hours. Silently continuing is the failure; so is measuring against a
    scratch directory that has no binary in it."""

    def test_a_failed_build_aborts_before_any_comparison(self):
        recorder = _Recorder(_comparison(0.0), build_passes=False,
                             build_reason="undefined symbol __ockl_wfred_add_i32")
        with self.assertRaises(loop.RunAborted) as caught:
            recorder.run()
        self.assertIn("would not build", str(caught.exception))
        self.assertIn("undefined symbol", str(caught.exception))
        # BROKEN READS: order == ["clean", "build", "compare"] -- an A/A taken against a
        # scratch tree containing no llama-bench, i.e. BenchFailed blamed on the loop.
        self.assertEqual(recorder.order, ["build"])
        self.assertEqual(recorder.compared, [])


# ------------------------------------------------------- the passing direction


class TwoBuildsOfOneCommitMustNotTripIt(unittest.TestCase):
    """The other half of the mutation test. A guard that fires on a healthy promotion
    aborts every run and is removed within a day, so the passing direction is not
    optional coverage -- it is the reason the guard is allowed to exist."""

    def test_an_identical_pair_passes(self):
        recorder = _Recorder(_comparison(0.0))
        verdict = recorder.run()
        # BROKEN READS: RunAborted, on a promotion where nothing at all is wrong.
        self.assertTrue(verdict.passed)
        self.assertAlmostEqual(verdict.effect_pct, 0.0, places=6)
        self.assertIn("inside the floor", verdict.detail)

    def test_realistic_aa_noise_passes(self):
        """+0.420% is inside the 0.973% prefill floor at 5 pairs: real A/A jitter."""
        recorder = _Recorder(_comparison(0.420))
        self.assertTrue(recorder.run().passed)

    def test_a_drifting_but_tiny_effect_still_passes(self):
        """Drift must not become a second, accidental reason to abort. The guard asks
        one question -- is the magnitude inside the floor -- and 0.1% is."""
        recorder = _Recorder(_comparison(
            0.1, candidate_samples=[99.6 + i * 0.125 for i in range(9)], drift=True))
        self.assertTrue(recorder.run().passed)

    def test_the_boundary_is_where_it_says_it_is(self):
        """0.9717% passes, 0.9824% aborts -- the bar is the run's own floor (0.9727%)
        and nothing else. BROKEN READS: a guard hard-coding a looser bar (say 3%, the
        superseded loop's) passes BOTH rows, including a 2.9% anchor mismatch.

        A hair either side rather than the floor exactly: `_comparison` derives its
        effect from the samples, so an exact-equality probe tests float rounding.
        """
        self.assertTrue(_Recorder(_comparison(FLOOR * 0.999)).run().passed)
        with self.assertRaises(loop.RunAborted):
            _Recorder(_comparison(FLOOR * 1.01)).run()
        with self.assertRaises(loop.RunAborted):
            _Recorder(_comparison(2.9)).run()


class TheScratchTreeIsWipedFirst(unittest.TestCase):
    """An incremental rebuild is precisely the mechanism capable of producing a binary
    that does not correspond to its source. A guard that reused a dirty scratch tree
    would be exposed to the fault class it exists to detect."""

    def test_clean_runs_before_the_build(self):
        recorder = _Recorder(_comparison(0.0))
        recorder.run()
        # BROKEN READS: ["build", "clean", "compare"] -- the champion compiled on top of
        # the PREVIOUS champion's object files, then the wipe removes the evidence.
        # 2026-09-06: the scratch is NO LONGER wiped first. The build is INCREMENTAL --
        # CMake recompiles only the changed sources -- because the digest is over the
        # OBJECTS (deterministic per source file), never the relinked .so. A wipe here
        # would re-impose the 20-min clean rebuild that doubled every keep's cost for
        # nothing. The heal (on a digest mismatch) is the one place a clean happens.
        self.assertEqual(recorder.order[0], "build")
        self.assertNotIn("clean", recorder.order)
        self.assertEqual(recorder.cleaned, [])  # nothing wiped on the normal path
        self.assertEqual(recorder.built_into, [Path("/scratch/build-anchor-verify")])

    def test_the_anchor_is_the_first_arm(self):
        """So the guard's number is on the same scale and sign as the run's own rows."""
        recorder = _Recorder(_comparison(0.0))
        recorder.run()
        self.assertEqual(recorder.compared,
                         [(Path("/store/anchor-gen-002"),
                           Path("/scratch/build-anchor-verify"))])


# ------------------------------------------------- the abort must stop the loop


def _hypothesis():
    return loop.Hypothesis(
        mechanism_id="akm-q5-bit-deposit", statement="bit-deposit the qh scatter",
        falsifier="no VGPR reduction below 64",
        target_surface="ggml/src/ggml-cuda/vecdotq.cuh",
        target_symbol="vec_dot_q5_0_q8_1_impl")


class _Planner:
    def __init__(self):
        self.proposals = 0

    def propose(self, context):
        self.proposals += 1
        return _hypothesis()

    def author(self, hypothesis, context):
        return ("ggml/src/ggml-cuda/vecdotq.cuh",)


class _Critic:
    def review_hypothesis(self, hypothesis, context):
        return loop.Review(True)

    def review_patch(self, hypothesis, paths, context):
        return loop.Review(True)


class TheAbortMustStopASingleLanePool(unittest.TestCase):
    """Asserting that `RunAborted` was constructed proves nothing about the loop. This
    asserts the loop's BEHAVIOUR: how many iterations it drew after the refusal.

    Single lane, so the count is EXACT — proposals == 1, not a bound. That exactness
    is the detector for abort LAUNDERING: a pool that catches `RunAborted` in its
    blanket handler still aborts eventually (three lane_errors arm the breaker), so
    `assertRaises` alone stays green; only the draw count says the refusal was
    honoured at the first opportunity rather than re-proven three times."""

    def _run(self, commit, iterations=6):
        planner = _Planner()
        outcomes = drive_single_lane(
            planner=planner, critic=_Critic(),
            measure=lambda h, p: _comparison(5.0),
            gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
            commit=commit, iterations=iterations)
        return planner, outcomes

    def test_it_stops_drawing_work_after_a_refused_anchor(self):
        def commit(hypothesis, paths, comparison):
            raise loop.RunAborted("anchor guard: the anchor slot does NOT hold the "
                                  "champion")
        planner = _Planner()
        with self.assertRaises(loop.RunAborted):
            drive_single_lane(
                planner=planner, critic=_Critic(),
                measure=lambda h, p: _comparison(5.0),
                gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
                commit=commit, iterations=6)
        # BROKEN READS: proposals == 3 and the breaker's RunAborted instead of the
        # guard's -- a blanket handler laundering the abort into `lane_error` keeps
        # drawing until the breaker trips. That is run 18 continuing against a bad
        # anchor, in miniature, with the brake taking the credit.
        self.assertEqual(planner.proposals, 1)

    def test_a_healthy_keep_still_draws_its_whole_budget(self):
        """The mutation in the other direction: `RunAborted` must not be raised by a
        promotion that verified. BROKEN READS: proposals == 1 and the run dies."""
        planner, outcomes = self._run(lambda h, p, c: "deadbeef", iterations=4)
        self.assertEqual(planner.proposals, 4)
        self.assertEqual([o.status for o in outcomes], ["kept"] * 4)


class TheAbortMustStopEveryLane(unittest.TestCase):
    """The anchor is SHARED, so a refused anchor voids every lane's next measurement --
    not only the lane that promoted it. At seven lanes a per-lane abort would keep six
    lanes measuring against the bad binary."""

    def _pool(self, iterations, *, lanes=3, cap=40):
        drawn = [0]
        lock = threading.Lock()

        def gate(hypothesis, paths):
            with lock:
                drawn[0] += 1
            return True, [gates.Verdict("compile", True)]

        def commit(worker, hypothesis, paths, comparison):
            raise loop.RunAborted("anchor guard: the anchor slot does NOT hold the "
                                  "champion")

        workers = [pipeline.Worker(f"lane{i}", Path(f"/w/{i}"), Path(f"/b/{i}"))
                   for i in range(lanes)]
        outcomes = []
        with self.assertRaises(loop.RunAborted):
            outcomes = pipeline.run_pool(
                workers=workers, make_planner=lambda w: _Planner(),
                make_critic=lambda w: _Critic(), build_context=dict,
                make_gate=lambda w: gate,
                make_measure=lambda w: (lambda h, p: _comparison(5.0)),
                commit=commit, champion_head=lambda: "c0",
                reset_to_champion=lambda w: "c0", record=lambda o: None,
                iterations=iterations,
                should_stop=lambda: drawn[0] >= cap)
        return drawn[0], outcomes

    def test_a_bounded_pool_stops_early(self):
        drawn, _ = self._pool(24)
        # BROKEN READS: drawn == 24 and NO exception -- `run_pool`'s blanket handler
        # turns the refusal into one `lane_error` per lane and every lane keeps drawing.
        self.assertLessEqual(drawn, 6, f"lanes kept drawing after the abort: {drawn}")

    def test_a_CONTINUOUS_pool_stops_too(self):
        """`iterations=None` is how the loop actually runs. Zeroing the budget is what
        stops it: `remaining` stops being None, so `take()` refuses. The `cap` here is a
        test safety net, not the mechanism -- BROKEN READS: drawn == 40, the cap."""
        drawn, _ = self._pool(None, cap=40)
        self.assertLess(drawn, 40, "only the safety cap stopped a continuous pool")


# ------------------------------------------- promotion BUILDS, never moves, the anchor


class PromotionBuildsTheAnchorWhereItIsUsed(unittest.TestCase):
    """`shutil.move(build_dir, anchor_slot)` was the leading root cause of run 18.

    A CMake build directory is not relocatable: `CMakeCache.txt` records absolute
    binary-dir, source-dir and compiler paths. run 18's `anchor-gen-001/CMakeCache.txt`
    is stamped 06:01 and its libraries 08:29 -- both before the run started at 09:37 and
    hours before the 11:03 promotion. Whatever was in that slot was not the build the
    loop made when it promoted.
    """

    def _promote(self, store, *, passes=True, commit=CHAMPION, recipe=None):
        seen = []

        def build(dest, targets=gates.DEFAULT_TARGETS):
            seen.append(Path(dest))
            if passes:
                (Path(dest) / "bin").mkdir(parents=True, exist_ok=True)
                (Path(dest) / "bin" / "llama-bench").write_text("elf", encoding="utf-8")
            return gates.Verdict("compile", passes, "" if passes else "build failed")

        promoted = pool.promote_anchor(store, build=build, champion_commit=commit,
                                       recipe=recipe or {"name": "house-gpu"})
        return promoted, seen

    def test_the_build_happens_AT_the_anchor_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            promoted, seen = self._promote(store)
            # BROKEN READS: seen == [.../build-candidate-loop] and the artifact is then
            # renamed -- a CMakeCache naming a directory that no longer exists.
            self.assertEqual(seen, [promoted])
            self.assertEqual(promoted.name, "anchor-gen-001")
            self.assertTrue((promoted / "bin" / "llama-bench").is_file())

    def test_no_candidate_directory_is_consumed(self):
        """The old contract MOVED the lane's build out of the candidate slot. Nothing
        may do that now: the lane's directory is its own and stays where it was built."""
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            candidate = store / "build-candidate"
            (candidate / "bin").mkdir(parents=True)
            (candidate / "bin" / "llama-bench").write_text("old", encoding="utf-8")
            promoted, _ = self._promote(store)
            # BROKEN READS under the move-based promotion: candidate.exists() is False
            # and the anchor's llama-bench reads "old".
            self.assertTrue((candidate / "bin" / "llama-bench").is_file())
            self.assertEqual((promoted / "bin" / "llama-bench").read_text(), "elf")

    def test_provenance_names_the_commit_and_the_recipe(self):
        """What makes the champion PROMOTABLE: the freeze runbook wants a full build
        from a named commit with a named recipe, never a relocated artifact."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            promoted, _ = self._promote(Path(tmp),
                                        recipe={"name": "house-gpu", "sha": "abc"})
            body = json.loads((promoted / "provenance.json").read_text())
            self.assertEqual(body["champion_commit"], CHAMPION)
            self.assertEqual(body["build_recipe"]["name"], "house-gpu")
            self.assertEqual(body["built_at"], str(promoted))

    def test_a_champion_that_will_not_build_refuses_the_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                self._promote(Path(tmp), passes=False)
            self.assertIn("would not build", str(caught.exception))

    def test_generations_never_collide_under_pruning(self):
        """Run 17 lost 23 of 30 advances to count-based numbering. With keep=1 the count
        is always 1, so every promotion targeted anchor-gen-002 forever."""
        with tempfile.TemporaryDirectory() as tmp:
            store, seen = Path(tmp), []
            for _ in range(5):
                promoted, _ = self._promote(store)
                pool.prune_anchor_generations(store, current=promoted)
                seen.append(promoted.name)
            self.assertEqual(len(set(seen)), 5, f"generations collided: {seen}")

    def test_it_refuses_to_build_over_an_existing_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "anchor-gen-001").mkdir()
            (store / "anchor-gen-002").mkdir()
            promoted, _ = self._promote(store)
            self.assertEqual(promoted.name, "anchor-gen-003")


class PromoteThenVerifyEndToEnd(unittest.TestCase):
    """The real `pool.promote_anchor` and the real `anchor.verify`, composed the way
    `run.main` composes them, driven single-lane through the real `pipeline.run_pool`
    -- the production driver. Only the build and the benchmark are doubles --
    everything between them is the shipping code."""

    def _drive(self, effect_pct, *, iterations=3):
        planner = _Planner()
        verdicts = []
        tmp = tempfile.TemporaryDirectory()
        store = Path(tmp.name)

        def build(dest, targets=gates.DEFAULT_TARGETS):
            (Path(dest) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(dest) / "bin" / "llama-bench").write_text("elf", encoding="utf-8")
            return gates.Verdict("compile", True)

        def commit(hypothesis, paths, comparison):
            promoted = pool.promote_anchor(store, build=build,
                                           champion_commit=CHAMPION,
                                           recipe={"name": "house-gpu"})
            anchor.verify(champion_commit=CHAMPION, anchor_build=promoted,
                          scratch_build=store / "scratch", noise_floor_pct=FLOOR,
                          build=build, compare=lambda a, b: _comparison(effect_pct),
                          on_verdict=verdicts.append)
            return "deadbeef"

        def run():
            return drive_single_lane(
                planner=planner, critic=_Critic(),
                measure=lambda h, p: _comparison(5.0),
                gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
                commit=commit, iterations=iterations)
        return planner, verdicts, store, run, tmp

    def test_a_bad_anchor_stops_the_run_after_one_iteration(self):
        planner, verdicts, store, run, tmp = self._drive(RUN_18_EFFECT_PCT)
        try:
            with self.assertRaises(loop.RunAborted):
                run()
            # The promotion HAPPENED -- and was then caught. BROKEN READS:
            # planner.proposals == 3, no exception, and three more iterations measured
            # against anchor-gen-001.
            self.assertEqual(planner.proposals, 1)
            self.assertTrue((store / "anchor-gen-001" / "provenance.json").is_file())
            self.assertEqual(len(verdicts), 1)
            self.assertFalse(verdicts[0].passed)
        finally:
            tmp.cleanup()

    def test_a_good_anchor_lets_the_run_continue(self):
        planner, verdicts, store, run, tmp = self._drive(0.2)
        try:
            outcomes = run()
            # BROKEN READS: RunAborted on the first keep, and proposals == 1.
            self.assertEqual(planner.proposals, 3)
            self.assertEqual([o.status for o in outcomes], ["kept"] * 3)
            self.assertEqual(len(verdicts), 3)
            self.assertTrue(all(v.passed for v in verdicts))
            self.assertTrue((store / "anchor-gen-003").is_dir())
        finally:
            tmp.cleanup()


class TheVerdictIsOnTheStatusSurface(unittest.TestCase):
    """"Recorded in the store" is two places: the experiments memory (`to_attempt`,
    covered above) and the status file the dashboard polls. A check whose result exists
    only in a terminal that has since scrolled is not auditable after the fact."""

    def _write(self, **kwargs):
        from autokernel.loop import status
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e", campaign_id="c",
                         anchor_commit=CHAMPION, surface="pp512", pairs=5,
                         noise_floor_pct=FLOOR, **kwargs)
            return status.read(Path(tmp))

    def test_a_failed_guard_is_visible_on_the_surface(self):
        recorder = _Recorder(_comparison(RUN_18_EFFECT_PCT))
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        body = self._write(anchor_guard=recorder.verdicts[0].to_dict())
        # BROKEN READS: KeyError -- the dashboard shows a run that stopped and no reason.
        self.assertFalse(body["anchor_guard"]["passed"])
        self.assertAlmostEqual(body["anchor_guard"]["effect_pct"], RUN_18_EFFECT_PCT,
                               places=3)

    def test_no_promotion_yet_is_null_not_a_pass(self):
        """`absent` is not `passed`. Collapsing them is how a check that never ran reads
        as a check that succeeded."""
        self.assertIsNone(self._write()["anchor_guard"])


# ------------------------------------------- the hash pre-check triad (R22-3)
#
# Builds of one commit are DETERMINISTIC on this host (R21-10: two builds of
# `ce1df3aa` differed by exactly one `.dynstr` byte -- the RUNPATH). So the guard's
# question is answerable by code-section digest before a pair is spent, and an
# above-floor A/A on hash-identical binaries indicts the SESSION, not the anchor.

#: Run 21's abort: a +1.765% A/A reading against a pooled A/A sigma of 0.417% --
#: a 4.2-sigma instrument excursion on binaries that were provably code-identical.
RUN_21_EXCURSION_PCT = 1.765

DIGEST = "6385a354c413f83465c03aa6b8acc50d5886ddd8b7248c3e8866d9aa1f027fed"


class _DigestRecorder(_Recorder):
    """A `_Recorder` whose injected `digest` returns per-path values in sequence.

    `plan` maps a path substring to the list of digests successive calls return
    (the last value repeats), so a heal that rebuilds the scratch can be given a
    different answer the second time -- or the same one, for the never-converges
    case."""

    def __init__(self, comparison, plan, **kwargs):
        super().__init__(comparison, **kwargs)
        self.plan = {key: list(values) for key, values in plan.items()}
        self.digest_calls: list[Path] = []

    def digest(self, path):
        self.digest_calls.append(Path(path))
        for key, values in self.plan.items():
            if key in str(path):
                return values.pop(0) if len(values) > 1 else values[0]
        return None

    def run(self, **kwargs):
        return super().run(digest=self.digest, **kwargs)


class TheRun21ExcursionMustNotAbortAHashProvenAnchor(unittest.TestCase):
    """Historical scenario (a): run 21's abort, reconstructed.

    The promoted anchor and the fresh champion build were code-identical (the
    R21-10 probe proved it after the fact), and the A/A read +1.765% -- 4.2 sigma
    of instrument excursion. The old guard aborted a healthy run; the triad must
    record `anchor_guard_excursion` and CONTINUE, because the hash has already
    answered the guard's actual question.
    """

    def _excursion(self):
        return _DigestRecorder(_comparison(RUN_21_EXCURSION_PCT),
                               {"anchor-gen": [DIGEST], "scratch": [DIGEST]})

    def test_the_fixture_is_above_the_floor(self):
        self.assertGreater(RUN_21_EXCURSION_PCT, FLOOR,
                           "run 21's excursion must exceed the floor or this suite "
                           "is not testing the abort-vs-continue decision at all")

    def test_it_records_an_excursion_and_returns(self):
        recorder = self._excursion()
        # BROKEN READS (the pre-R22-3 guard, i.e. the hash-check-dropped mutant):
        # loop.RunAborted here, ending a healthy run on an instrument excursion.
        verdict = recorder.run()
        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.excursion)
        self.assertAlmostEqual(verdict.effect_pct, RUN_21_EXCURSION_PCT, places=3)
        self.assertIn("IDENTICAL code digests", verdict.detail)

    def test_the_archived_row_says_excursion_not_verified(self):
        row = self._excursion().run().to_attempt()
        # BROKEN READS: "anchor_verified" -- an above-floor session reading
        # laundered into a clean bill of health nobody can find afterwards.
        self.assertEqual(row["status"], "anchor_guard_excursion")

    def test_the_aa_still_runs_as_a_session_health_sample(self):
        recorder = self._excursion()
        recorder.run()
        # BROKEN READS: no "compare" -- a hash short-circuit that stops sampling
        # session health, which is the only signal run 21's fault class leaves.
        self.assertIn("compare", recorder.order)
        self.assertEqual(recorder.order.count("build"), 1,
                         "identical digests must not trigger the heal rebuild")

    def test_an_inside_floor_reading_is_not_an_excursion(self):
        verdict = _DigestRecorder(_comparison(0.420),
                                  {"anchor-gen": [DIGEST], "scratch": [DIGEST]}).run()
        # BROKEN READS: excursion True on healthy A/A noise -- every clean
        # promotion flagged, and the flag stops meaning anything.
        self.assertFalse(verdict.excursion)
        self.assertEqual(verdict.to_attempt()["status"], "anchor_verified")

    def test_a_digest_of_none_is_not_proof(self):
        """None == None must never read as "hash-proven". A build whose library
        cannot be hashed falls back to the A/A-only contract, where run 21's
        reading aborts -- conservative, and exactly the pre-triad behaviour."""
        recorder = _DigestRecorder(_comparison(RUN_21_EXCURSION_PCT), {})
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        self.assertFalse(recorder.verdicts[0].excursion)


class TheRun18MismatchIsProvenByHashAlone(unittest.TestCase):
    """Historical scenario (b): run 18's fault class, caught deterministically.

    The binary in the anchor slot is not the champion, so its `.hip_fatbin` (and
    `.text`) digest cannot match a fresh build's. The triad aborts on the hash,
    spending ZERO of the 20 pairs the A/A would have cost, and names both digests
    so the abort carries its own evidence.
    """

    def _mismatch(self, comparison=None):
        return _DigestRecorder(comparison or _comparison(0.0),
                               {"anchor-gen": ["a" * 64], "scratch": ["f" * 64]})

    def test_differing_digests_abort_without_spending_a_pair(self):
        recorder = self._mismatch()
        with self.assertRaises(loop.RunAborted) as caught:
            recorder.run()
        # BROKEN READS (digests-differ-continues mutant): no exception, and a
        # "compare" entry -- 20 pairs of device time spent measuring a slot the
        # hash had already indicted, then run 18's void numbers after it.
        self.assertNotIn("compare", recorder.order)
        self.assertIn("a" * 64, str(caught.exception))
        self.assertIn("f" * 64, str(caught.exception))
        self.assertIn("run 18", str(caught.exception).lower())

    def test_the_abort_verdict_is_recorded_with_both_digests(self):
        recorder = self._mismatch()
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        self.assertEqual(len(recorder.verdicts), 1)
        row = recorder.verdicts[0].to_attempt()
        self.assertEqual(row["status"], "anchor_mismatch")
        self.assertEqual(row["anchor_guard"]["anchor_digest"], "a" * 64)
        self.assertEqual(row["anchor_guard"]["fresh_digest"], "f" * 64)
        self.assertEqual(row["anchor_guard"]["pairs"], 0)

    def test_one_heal_is_attempted_then_the_abort_stands(self):
        recorder = self._mismatch()
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        # Exactly two builds (the incremental one, then ONE heal) and ONE clean: the
        # heal's rmtree precedes its rebuild. (2026-09-06: the initial clean is gone --
        # the first build is incremental and the digest is over objects, so a stale
        # object cannot pass.) BROKEN READS: build-count 1 (no heal -- a transient
        # scratch corruption aborts a healthy run) or 3+ (the heal loops).
        self.assertEqual(recorder.order.count("build"), 2)
        self.assertEqual(recorder.order.count("clean"), 1)

    def test_the_heal_never_loops(self):
        recorder = self._mismatch()
        original = recorder.build

        def counting_build(dest):
            if recorder.order.count("build") >= 4:
                raise AssertionError("heal-once is rebuilding in a loop")
            return original(dest)

        recorder.build = counting_build
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        self.assertEqual(recorder.order.count("build"), 2)

    def test_a_transient_scratch_corruption_heals_and_the_run_continues(self):
        """First scratch digest wrong, the rebuilt one right: the guard's OWN
        artifact glitched, not the anchor. Abort here and every transient build
        corruption kills a healthy run -- the backstop exists for this case."""
        recorder = _DigestRecorder(
            _comparison(0.420),
            {"anchor-gen": ["a" * 64], "scratch": ["f" * 64, "a" * 64]})
        verdict = recorder.run()
        self.assertTrue(verdict.passed)
        self.assertFalse(verdict.excursion, "a healed pair is proven, and 0.420% "
                                            "is inside the floor: not an excursion")
        self.assertEqual(recorder.order.count("build"), 2)
        self.assertIn("compare", recorder.order)

    def test_a_heal_whose_rebuild_fails_still_aborts(self):
        recorder = _DigestRecorder(_comparison(0.0),
                                   {"anchor-gen": ["a" * 64], "scratch": ["f" * 64]})
        calls = {"n": 0}
        original = recorder.build

        def failing_second_build(dest):
            calls["n"] += 1
            if calls["n"] >= 2:
                recorder.order.append("build")
                return gates.Verdict("compile", False, "OOM during heal rebuild")
            return original(dest)

        recorder.build = failing_second_build
        with self.assertRaises(loop.RunAborted) as caught:
            recorder.run()
        self.assertIn("unavailable", str(caught.exception))


class TheComparisonIsPersistedOnEveryMeasuredVerdict(unittest.TestCase):
    """Run 21's abort left no samples, drift or clock record -- the verdict said
    +1.765% and discarded the `Comparison` that could say WHY. Every verdict that
    measured must now embed `comparison.to_dict()` in the archived row."""

    def _row(self, recorder):
        return recorder.verdicts[0].to_attempt()["anchor_guard"]

    def test_a_passing_verdict_carries_its_comparison(self):
        recorder = _Recorder(_comparison(0.420))
        recorder.run()
        embedded = self._row(recorder)["comparison"]
        # BROKEN READS: KeyError -- the archived row again carries only a number.
        self.assertEqual(embedded["anchor_samples"], [100.0] * 9)
        self.assertEqual(embedded["pairs"], 5)
        self.assertIn("anchor_drift_pct", embedded)

    def test_an_aborting_verdict_carries_it_too(self):
        recorder = _Recorder(_comparison(RUN_18_EFFECT_PCT))
        with self.assertRaises(loop.RunAborted):
            recorder.run()
        embedded = self._row(recorder)["comparison"]
        self.assertAlmostEqual(embedded["effect_pct"], RUN_18_EFFECT_PCT, places=3)
        self.assertEqual(len(embedded["candidate_samples"]), 9)

    def test_an_excursion_verdict_carries_comparison_and_digests(self):
        recorder = _DigestRecorder(_comparison(RUN_21_EXCURSION_PCT),
                                   {"anchor-gen": [DIGEST], "scratch": [DIGEST]})
        recorder.run()
        row = self._row(recorder)
        self.assertAlmostEqual(row["comparison"]["effect_pct"],
                               RUN_21_EXCURSION_PCT, places=3)
        self.assertEqual(row["anchor_digest"], DIGEST)
        self.assertEqual(row["fresh_digest"], DIGEST)


if __name__ == "__main__":
    unittest.main()
