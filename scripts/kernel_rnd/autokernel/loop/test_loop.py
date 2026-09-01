"""The loop's control flow, exercised without a GPU, an LLM, or a ROCm toolchain.

Every side effect is injected, so these tests are about the thing that actually went
wrong: not whether a kernel is fast, but whether a rejection reaches the actor that
can act on it, whether budgets stay independent, and whether a candidate that is not
faster can be reported as faster.
"""
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import bench, gates, loop, pipeline


def drive_single_lane(*, planner, critic, measure, gate, commit, iterations,
                      build_context=dict, record=None):
    """Drive `pipeline.run_pool` — THE production path — as a one-lane pool.

    This replaced the deleted `loop.run` test seam (R21-7): the properties these
    suites pin (fault containment, abort propagation, budget draw) belong to the
    path production actually runs, so the driver is the pool at `--workers 1` —
    which is exactly what `run.py --workers 1` builds — not a private sequential
    loop kept alive for the fakes. The champion head is pinned, so the staleness
    check never fires and one lane runs strictly in order.
    """
    worker = pipeline.Worker("lane0", Path("/w/lane0"), Path("/b/lane0"))
    return pipeline.run_pool(
        workers=[worker], make_planner=lambda w: planner,
        make_critic=lambda w: critic, build_context=build_context,
        make_gate=lambda w: gate, make_measure=lambda w: measure,
        commit=lambda w, h, p, c: commit(h, p, c),
        champion_head=lambda: "c0", reset_to_champion=lambda w: "c0",
        record=record or (lambda outcome: None), iterations=iterations)


def _hypothesis(mechanism="akm-q5-bit-deposit"):
    return loop.Hypothesis(
        mechanism_id=mechanism, statement="bit-deposit the qh scatter",
        falsifier="no VGPR reduction below 64",
        target_surface="ggml/src/ggml-cuda/vecdotq.cuh",
        target_symbol="vec_dot_q5_0_q8_1_impl")


def _comparison(effect, *, floor=1.0):
    return bench.Comparison(
        surface="tg128", anchor_samples=[100.0], candidate_samples=[100.0 * (1 + effect)],
        effect=effect, estimator="median_over_median", pairs=5,
        noise_floor_pct=floor, residency={"invocations": 10, "resident": 10})


class _Planner:
    """Records what it was told, so a lost reason is visible."""

    def __init__(self, hypotheses=None):
        self.hypotheses = list(hypotheses or [_hypothesis()])
        self.seen_hypothesis_rejections = []
        self.seen_patch_rejections = []
        self.proposals = 0
        self.authorings = 0

    def propose(self, context):
        self.seen_hypothesis_rejections.append(
            list(context.get("prior_hypothesis_rejections") or []))
        self.proposals += 1
        index = min(self.proposals - 1, len(self.hypotheses) - 1)
        return self.hypotheses[index]

    def author(self, hypothesis, context):
        self.seen_patch_rejections.append(
            list(context.get("prior_patch_rejections") or []))
        self.authorings += 1
        return ("ggml/src/ggml-cuda/vecdotq.cuh",)


class _Critic:
    def __init__(self, hypothesis_verdicts, patch_verdicts):
        self.hypothesis_verdicts = list(hypothesis_verdicts)
        self.patch_verdicts = list(patch_verdicts)

    def review_hypothesis(self, hypothesis, context):
        return (self.hypothesis_verdicts.pop(0) if self.hypothesis_verdicts
                else loop.Review(True))

    def review_patch(self, hypothesis, paths, context):
        return (self.patch_verdicts.pop(0) if self.patch_verdicts
                else loop.Review(True))


def _run(planner, critic, *, effect=0.05, gate_ok=True, floor=1.0):
    committed = {}

    def measure(hypothesis, paths):
        return _comparison(effect, floor=floor)

    def gate(hypothesis, paths):
        if gate_ok:
            return True, [gates.Verdict("compile", True)]
        return False, [gates.Verdict("compile", False, "build failed: undefined symbol")]

    def commit(hypothesis, paths, comparison):
        committed["head"] = "abc1234"
        return "abc1234"

    outcome = loop.iterate(planner=planner, critic=critic, context={},
                           measure=measure, gate=gate, commit=commit)
    return outcome, committed


class RejectionsMustCarryAReason(unittest.TestCase):

    def test_a_rejection_without_a_reason_is_refused_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            loop.Review(False)
        self.assertIn("must carry a reason", str(caught.exception))

    def test_whitespace_is_not_a_reason(self):
        with self.assertRaises(ValueError):
            loop.Review(False, "   ")

    def test_an_acceptance_needs_no_reason(self):
        self.assertTrue(loop.Review(True).accepted)


class TheLoopback(unittest.TestCase):
    """The exit criterion: a rejection whose reason reaches the next proposal."""

    def test_pass_one_rejection_reaches_the_planner_verbatim_and_it_refines(self):
        planner = _Planner([_hypothesis("akm-bad"), _hypothesis("akm-good")])
        critic = _Critic([loop.Review(False, "already measured null in epoch 4de6")], [])
        outcome, committed = _run(planner, critic)

        self.assertEqual(outcome.status, "kept")
        self.assertEqual(planner.proposals, 2, "the planner must get a second turn")
        # The COMPLETED loopback: the reason appears verbatim in the next proposal.
        self.assertEqual(planner.seen_hypothesis_rejections[0], [])
        self.assertEqual(planner.seen_hypothesis_rejections[1],
                         ["already measured null in epoch 4de6"])
        self.assertEqual(outcome.hypothesis.mechanism_id, "akm-good")
        self.assertEqual(committed["head"], "abc1234")

    def test_pass_two_rejection_reaches_the_planner_and_leaves_the_hypothesis_alone(self):
        planner = _Planner()
        critic = _Critic([], [loop.Review(False, "diff derives undeclared symbols")])
        outcome, _ = _run(planner, critic)

        self.assertEqual(outcome.status, "kept")
        self.assertEqual(planner.authorings, 2, "the planner must rewrite the patch")
        self.assertEqual(planner.seen_patch_rejections[1],
                         ["diff derives undeclared symbols"])
        # A bad patch is not evidence against the idea: one hypothesis, reproposed
        # zero times.
        self.assertEqual(planner.proposals, 1)

    def test_a_gate_failure_loops_back_with_the_toolchain_message(self):
        planner = _Planner()
        critic = _Critic([], [])
        outcome, _ = _run(planner, critic, gate_ok=False)
        self.assertEqual(outcome.status, "refused_at_formation")
        self.assertIn("build failed: undefined symbol", " ".join(outcome.reasons))


class BudgetsAreIndependent(unittest.TestCase):
    """critic_revise must never be charged to the hypothesis's counter."""

    def test_patch_rounds_are_bounded_and_then_return_to_the_hypothesis_loop(self):
        planner = _Planner()
        critic = _Critic([], [loop.Review(False, "scope creep")] * 10)
        outcome, _ = _run(planner, critic)
        # 3 hypothesis rounds x 2 patch rounds: the patch budget is spent inside each
        # hypothesis round, not shared across them.
        self.assertEqual(planner.authorings, loop.HYPOTHESIS_ROUNDS * loop.PATCH_ROUNDS)
        self.assertEqual(planner.proposals, loop.HYPOTHESIS_ROUNDS)
        self.assertEqual(outcome.status, "refused_at_formation")

    def test_a_hypothesis_budget_exhaustion_does_not_retire_the_hypothesis(self):
        planner = _Planner()
        critic = _Critic([loop.Review(False, "unsupported by the profile")] * 5, [])
        outcome, _ = _run(planner, critic)
        self.assertEqual(outcome.status, "refused_at_formation")
        self.assertEqual(len(outcome.reasons), loop.HYPOTHESIS_ROUNDS)
        # This used to read `assertIsNone(outcome.hypothesis)` as a PROXY for "not
        # retired". The proxy cost the operator the mechanism_id on the row the
        # dashboard shows most often -- 5 of 5 iterations in run 6 read `None`. Naming
        # what was refused is not retiring it, so assert the real contract instead.
        row = outcome.to_attempt()
        self.assertTrue(row.get("mechanism_id"), "the row must name what was refused")
        for marker in ("retired", "banned", "do_not_repeat"):
            self.assertNotIn(marker, row)

    def test_a_mechanism_refused_before_can_still_be_proposed_and_kept(self):
        """The property that `assertIsNone` proxy was standing in for.

        Nothing in the loop excludes a mechanism because it appears in history: a
        refusal is information the planner must answer, never a gate. This is what
        stops the v33 failure where three turns retired a hypothesis that was never
        tested.
        """
        refused, _ = _run(_Planner(),
                          _Critic([loop.Review(False, "unsupported")] * 5, []))
        self.assertEqual(refused.status, "refused_at_formation")
        again, committed = _run(_Planner(), _Critic([], []), effect=0.05, floor=1.0)
        self.assertEqual(again.status, "kept")
        self.assertEqual(again.hypothesis.mechanism_id,
                         refused.hypothesis.mechanism_id)
        self.assertTrue(committed)


class TheDecision(unittest.TestCase):
    """A candidate that is not faster must not be reported as faster."""

    def test_a_clearly_faster_candidate_is_kept(self):
        outcome, committed = _run(_Planner(), _Critic([], []), effect=0.05, floor=1.0)
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(committed["head"], "abc1234")

    def test_a_slower_candidate_is_never_kept(self):
        outcome, committed = _run(_Planner(), _Critic([], []), effect=-0.05, floor=1.0)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(committed, {}, "a regression must not advance the champion")

    def test_an_effect_inside_the_noise_floor_is_not_a_win(self):
        """The defect in one assertion.

        The old loop's 3% bar sat BELOW its instrument's own measured decode floor of
        3.452%, so noise could clear it. An effect inside the floor is not a small
        win; it is not a measurement of anything.
        """
        outcome, committed = _run(_Planner(), _Critic([], []), effect=0.02, floor=3.452)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(committed, {})

    def test_the_same_effect_outside_a_lower_floor_is_a_win(self):
        """Guards against the above passing for the wrong reason."""
        outcome, committed = _run(_Planner(), _Critic([], []), effect=0.02, floor=1.5)
        self.assertEqual(outcome.status, "kept")

    def test_a_comparison_with_no_declared_floor_is_never_decisive(self):
        self.assertFalse(_comparison(0.5, floor=None).decisive)

    def test_a_null_result_is_recorded_with_its_evidence(self):
        outcome, _ = _run(_Planner(), _Critic([], []), effect=0.02, floor=3.452)
        row = outcome.to_attempt()
        self.assertEqual(row["status"], "measured_null")
        self.assertAlmostEqual(row["effect_fraction"], 0.02)
        self.assertIn("comparison", row)
        self.assertEqual(row["mechanism_id"], "akm-q5-bit-deposit")


class BenchRefusals(unittest.TestCase):

    def test_a_non_resident_run_is_refused_not_reported(self):
        """'I invoked the HIP build' is not evidence of a HIP run."""
        comparison = bench.Comparison(
            surface="tg128", anchor_samples=[1.0], candidate_samples=[2.0],
            effect=1.0, estimator="median_over_median", pairs=1,
            noise_floor_pct=1.0, residency={"invocations": 2, "resident": 0})
        # The refusal happens in `compare`; this pins the shape the record carries.
        self.assertEqual(comparison.residency["resident"], 0)

    def test_bimodality_is_flagged(self):
        self.assertTrue(bench.spread_is_suspect([25409, 18083, 25372, 16175, 25381]))
        self.assertFalse(bench.spread_is_suspect([400.0, 401.0, 402.0]))

    def test_the_minimum_pair_count_matches_the_measured_floor(self):
        self.assertGreaterEqual(bench.MIN_PAIRS, 5)


if __name__ == "__main__":
    unittest.main()


class ProviderTransientsEndAnIterationNotTheRun(unittest.TestCase):
    """A codex 401 took down 284 attempts in 23 minutes. Reproduced here, once.

    The FIRST real end-to-end run of this loop hit exactly that class: the planner's
    authoring step raised, the exception escaped `iterate`, and the whole run died on
    iteration 1. That is the superseded controller's defect -- provider faults
    escaping as terminal -- reproduced in the replacement. These tests pin the fix.
    """

    class _Exploding:
        def __init__(self, where): self.where = where
        def propose(self, context):
            if self.where == "propose":
                raise loop.ActorTransient("provider 401")
            return _hypothesis()
        def author(self, hypothesis, context):
            if self.where == "author":
                raise loop.ActorTransient("authoring returned no changed paths")
            return ("a.cu",)

    def test_a_transient_while_proposing_ends_the_iteration(self):
        outcome, _ = _run(self._Exploding("propose"), _Critic([], []))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIn("provider 401", outcome.reasons[0])

    def test_a_transient_while_authoring_ends_the_iteration(self):
        outcome, committed = _run(self._Exploding("author"), _Critic([], []))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIn("no changed paths", outcome.reasons[0])
        self.assertEqual(committed, {})

    def test_the_transient_is_recorded_with_its_reason(self):
        outcome, _ = _run(self._Exploding("author"), _Critic([], []))
        row = outcome.to_attempt()
        self.assertEqual(row["status"], "planner_transient")
        self.assertIn("no changed paths", row["reason"])

    def test_a_run_continues_past_a_transient(self):
        """The property that matters: one bad provider call is not a dead campaign.

        Driven through the production pool. A transient is contained INSIDE
        `iterate`, so it reaches the pool as a recorded outcome, never trips the
        consecutive-error breaker, and the lane keeps drawing its budget."""
        outcomes = drive_single_lane(
            planner=self._Exploding("author"), critic=_Critic([], []),
            measure=lambda h, p: _comparison(0.05),
            gate=lambda h, p: (True, []),
            commit=lambda h, p, c: "head", iterations=3)
        self.assertEqual([o.status for o in outcomes], ["planner_transient"] * 3)


# `TheTreeIsResetBeforeEachIteration` was deleted with the sequential CLI path: the
# reset-before-every-iteration property now lives where production runs it —
# `pool.reset_to_champion` at the top of every `run_pool` lane — and is pinned by
# `test_pipeline.NoLaneMayDieOutsideTheTry` plus the staleness tests, which need the
# reset's returned base to pass at all.


class ARefusalNamesWhatWasRefused(unittest.TestCase):
    """A refused_at_formation row with an empty mechanism_id tells the operator that
    something was refused without saying what -- and it is the row the dashboard shows
    most often (run 6: 5 of 5 iterations)."""

    def test_the_refusal_carries_the_last_hypothesis_proposed(self):
        h = loop.Hypothesis("akm-q4k-branchless", "s", "f", "a.cu", "sym")
        planner = mock.Mock(); planner.propose.return_value = h
        critic = mock.Mock()
        critic.review_hypothesis.return_value = loop.Review(False, "already in v9")
        outcome = loop.iterate(planner=planner, critic=critic, context={},
                               measure=mock.Mock(), gate=mock.Mock(), commit=mock.Mock())
        self.assertEqual(outcome.status, "refused_at_formation")
        self.assertIsNotNone(outcome.hypothesis)
        self.assertEqual(outcome.to_attempt()["mechanism_id"], "akm-q4k-branchless")
        self.assertIn("already in v9", outcome.to_attempt()["reason"])


class TheEnforcedFloorNeverSitsBelowTheMeasuredOne(unittest.TestCase):
    """A bar below the instrument's own resolution is the defect this rebuild exists
    to close -- the superseded loop used a 3% bar while 4 of 20 pure-noise decode
    pairs already exceeded it.

    Three different floor figures were in circulation at once: bench.py's docstring,
    a hand-written table in program.md (0.753%/1.848% at k=5, reproducible by no
    method from the raw pairs), and run.py's sigma/sqrt(n). This pins the only
    relationship that actually matters between them.
    """

    def test_every_enforced_floor_is_at_or_above_its_measured_row(self):
        from autokernel.loop import run as run_mod
        for surface in bench.MEASURED_FLOOR_PCT:
            for pair_count in (5, 9):
                enforced = run_mod.noise_floor_pct(
                    surface, pair_count, f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf")
                measured = bench.MEASURED_FLOOR_PCT[surface][pair_count]
                self.assertGreaterEqual(
                    enforced, measured,
                    f"{surface}: enforced {enforced:.3f}% sits BELOW the measured "
                    f"{measured:.3f}% at {pair_count} pairs -- noise would pass")

    def test_the_floor_scales_with_the_pairs_ACTUALLY_run(self):
        """It was a dict of constants computed at 5, so `--pairs 9` still enforced the
        5-pair bar: 1.544% on decode where the measured 9-pair floor is 1.175%. Safe,
        but it throws away the sensitivity the extra pairs were bought for."""
        from autokernel.loop import run as run_mod
        for surface in ("pp512", "tg128"):
            model = f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
            self.assertGreater(run_mod.noise_floor_pct(surface, 5, model),
                               run_mod.noise_floor_pct(surface, 9, model),
                               "more pairs must lower the bar")

    def test_decode_does_not_average_down_at_root_n(self):
        """The parametric bound alone is UNSAFE on decode: sqrt(n) predicts 1.151% at
        9 pairs while the instrument actually resolves 1.175%, so pure noise would
        clear the bar. The floor takes the max of parametric and measured."""
        from autokernel.loop import run as run_mod
        self.assertLess(3.452 / (9 ** 0.5), bench.MEASURED_FLOOR_PCT["tg128"][9])
        self.assertAlmostEqual(
            run_mod.noise_floor_pct("tg128", 9,
                                    f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf"),
                               bench.MEASURED_FLOOR_PCT["tg128"][9], places=6)

    def test_the_measured_table_is_monotonic_in_pair_count(self):
        """Averaging more pairs must not raise the floor; a non-monotonic row means
        the table was transcribed rather than derived."""
        for surface, rows in bench.MEASURED_FLOOR_PCT.items():
            counts = sorted(rows)
            values = [rows[k] for k in counts]
            self.assertEqual(values, sorted(values, reverse=True), surface)

    def test_min_pairs_has_a_measured_row_at_all(self):
        for surface in ("pp512", "tg128"):
            self.assertIn(bench.MIN_PAIRS, bench.MEASURED_FLOOR_PCT[surface])


class AnInstrumentFailureEndsTheIterationNotTheRun(unittest.TestCase):
    """Run 12 died on iteration 1: llama-bench was SIGKILLed (rc=-9) mid-measurement
    and BenchFailed escaped `iterate`, ending a ten-iteration run that had already
    paid for its profile and was holding the device.

    A provider transient was already handled this way. An instrument failure is the
    same shape -- not the science failing -- and earlyoom on this host ignores
    llama-server but not llama-bench, so an external kill is a standing hazard.
    """

    def test_a_bench_failure_is_recorded_and_the_run_continues(self):
        def measure(hypothesis, paths):
            raise bench.BenchFailed("llama-bench rc=-9: killed")
        outcome = loop.iterate(
            planner=_Planner(), critic=_Critic([], []), context={},
            measure=measure,
            gate=lambda *a: (True, [gates.Verdict("compile", True)]),
            commit=lambda *a: "abc1234")
        self.assertEqual(outcome.status, "bench_failed")
        self.assertIn("rc=-9", " ".join(outcome.reasons))

    def test_it_is_not_conflated_with_a_provider_transient(self):
        """Merging them would hide a failing instrument behind a flaky API."""
        def measure(hypothesis, paths):
            raise bench.BenchFailed("only 3/10 invocations were sampled resident")
        outcome = loop.iterate(
            planner=_Planner(), critic=_Critic([], []), context={},
            measure=measure,
            gate=lambda *a: (True, [gates.Verdict("compile", True)]),
            commit=lambda *a: "abc1234")
        self.assertNotEqual(outcome.status, "planner_transient")

    def test_the_run_keeps_going_after_one(self):
        """`bench_failed` is contained inside `iterate`, so the pool sees a recorded
        outcome — not an error status — and the breaker never arms."""
        calls = {"n": 0}

        def measure(hypothesis, paths):
            calls["n"] += 1
            raise bench.BenchFailed("killed")

        outcomes = drive_single_lane(
            planner=_Planner(), critic=_Critic([], []), measure=measure,
            gate=lambda *a: (True, [gates.Verdict("compile", True)]),
            commit=lambda *a: "abc1234", iterations=3)
        self.assertEqual(len(outcomes), 3, "one killed bench must not end the run")
        self.assertEqual([o.status for o in outcomes], ["bench_failed"] * 3)
        self.assertEqual(calls["n"], 3)


class NoSingleIterationMayEndTheRun(unittest.TestCase):
    """Run 12 lost ten iterations, a profile and a held device to one SIGKILLed
    benchmark. Before that a codex 401 took down 284 attempts. The iteration is the
    unit of failure; the run is what has to survive.

    Driven through the production pool: an exception that escapes `iterate` is the
    lane's blanket containment (`lane_error`), one-off faults reset the breaker's
    count, and a blanket catch that merely swallowed would be worse than the crash,
    so the traceback is kept. The BREAKER itself is tested in `test_pipeline.py`
    (`ThePoolBreaker`).
    """

    def _run(self, measure, iterations=5):
        return drive_single_lane(
            planner=_Planner(), critic=_Critic([], []), measure=measure,
            gate=lambda *a: (True, [gates.Verdict("compile", True)]),
            commit=lambda *a: "abc1234", iterations=iterations)

    def test_an_unexpected_exception_becomes_a_recorded_outcome(self):
        calls = {"n": 0}

        def measure(hypothesis, paths):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient disk hiccup")
            return _comparison(0.05, floor=1.0)

        outcomes = self._run(measure, iterations=3)
        self.assertEqual(len(outcomes), 3, "the run must continue past the fault")
        self.assertEqual(outcomes[0].status, "lane_error")
        self.assertEqual([o.status for o in outcomes[1:]], ["kept"] * 2,
                         "one fault, then the lane is back to work")
        self.assertIn("OSError", " ".join(outcomes[0].reasons))

    def test_the_traceback_is_kept_not_swallowed(self):
        """The pool records the traceback's LAST 1500 chars, so the header line can
        be truncated away on a deep stack; the frames are the evidence that survives.
        Asserting on the "Traceback" banner would pin a spelling, not the record."""
        outcomes = self._run(
            lambda h, p: (_ for _ in ()).throw(RuntimeError("boom")), iterations=1)
        recorded = " ".join(outcomes[0].reasons)
        self.assertIn('File "', recorded,
                      "a swallowed exception is worse than the crash it replaced")
        self.assertIn("RuntimeError: boom", recorded)


class AnUncalibratedMeasurementCannotBecomeAKeep(unittest.TestCase):
    """Run-22 discipline at the loop's own keep gate: `decisive=None` is falsy, so a
    positive raw effect on an uncalibrated surface records as measured_null with an
    UNDECIDABLE reason -- and commit is never drawn. Mutation partner:
    `AnUncalibratedSurfaceRefusesToDecide` (test_bench) pins the None itself."""

    def _uncalibrated(self, effect=0.10, floor=1.0):
        return bench.Comparison(
            surface="dec-b4", anchor_samples=[100.0],
            candidate_samples=[100.0 * (1 + effect)], effect=effect,
            estimator="median_over_median", pairs=5, noise_floor_pct=floor,
            residency={}, calibrated=False)

    def test_a_huge_effect_on_an_uncalibrated_surface_is_not_kept(self):
        committed = {}

        def commit(hypothesis, paths, comparison):
            committed["head"] = "deadbeef"      # reaching here IS the defect
            return "deadbeef"

        outcome = loop.iterate(
            planner=_Planner(), critic=_Critic([], []), context={},
            measure=lambda h, p: self._uncalibrated(),
            gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
            commit=commit)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(committed, {}, "commit must never be drawn on decisive=None")

    def test_the_reason_says_undecidable_and_names_the_surface(self):
        reason = loop._null_reason(self._uncalibrated())
        self.assertIn("UNDECIDABLE", reason)
        self.assertIn("dec-b4", reason)
        self.assertIn("--calibrate-surface", reason,
                      "the reason must say how to make the surface decisive")

    def test_a_calibrated_null_still_reads_as_a_floor_miss(self):
        comparison = bench.Comparison(
            surface="tg128", anchor_samples=[100.0], candidate_samples=[100.4],
            effect=0.004, estimator="median_over_median", pairs=5,
            noise_floor_pct=1.0, residency={})
        self.assertNotIn("UNDECIDABLE", loop._null_reason(comparison))
        self.assertIn("did not clear", loop._null_reason(comparison))
