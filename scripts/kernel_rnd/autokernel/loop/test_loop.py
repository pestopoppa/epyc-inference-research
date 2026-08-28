"""The loop's control flow, exercised without a GPU, an LLM, or a ROCm toolchain.

Every side effect is injected, so these tests are about the thing that actually went
wrong: not whether a kernel is fast, but whether a rejection reaches the actor that
can act on it, whether budgets stay independent, and whether a candidate that is not
faster can be reported as faster.
"""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import bench, gates, loop


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
        """The property that matters: one bad provider call is not a dead campaign."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            outcomes = loop.run(
                planner=self._Exploding("author"), critic=_Critic([], []),
                build_context=dict,
                measure=lambda h, p: _comparison(0.05),
                gate=lambda h, p: (True, []),
                commit=lambda h, p, c: "head",
                store_root=Path(tmp), epoch="e" * 64,
                campaign_id="ak-test", iterations=3)
        self.assertEqual([o.status for o in outcomes], ["planner_transient"] * 3)


class TheTreeIsResetBeforeEachIteration(unittest.TestCase):
    """A failed authoring attempt must not satisfy the NEXT iteration's ground truth.

    Run 5 ended with `mmq.cu` still modified by an attempt that never passed. The
    worktree check asks "did the actor actually change something", and a leftover
    answers yes on the previous iteration's behalf -- a check that passes without the
    thing it checks for having happened.
    """

    def _planner(self):
        planner = mock.Mock()
        planner.propose.side_effect = loop.ActorTransient("no hypothesis")
        return planner

    def test_reset_runs_before_every_iteration(self):
        order = []
        planner = self._planner()
        planner.propose.side_effect = lambda ctx: order.append("propose") or (_ for _ in ()).throw(
            loop.ActorTransient("no hypothesis"))
        with tempfile.TemporaryDirectory() as tmp:
            loop.run(planner=planner, critic=mock.Mock(),
                         build_context=dict, measure=mock.Mock(), gate=mock.Mock(),
                         commit=mock.Mock(), store_root=Path(tmp), epoch="e" * 64,
                         campaign_id="ak-loop", iterations=3,
                         reset=lambda: order.append("reset"))
        self.assertEqual(order, ["reset", "propose"] * 3,
                         "every iteration must start from the champion, not from "
                         "the previous attempt's leftovers")

    def test_the_loop_still_runs_without_a_reset_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcomes = loop.run(
                planner=self._planner(), critic=mock.Mock(), build_context=dict,
                measure=mock.Mock(), gate=mock.Mock(), commit=mock.Mock(),
                store_root=Path(tmp), epoch="e" * 64, campaign_id="ak-loop",
                iterations=2)
        self.assertEqual(len(outcomes), 2)


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
