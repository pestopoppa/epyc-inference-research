"""The actor layer, tested without spending a provider call.

The two things that must hold before this touches a real API: consecutive failures
back off (a codex 401 produced 284 failures in 23 minutes with zero delay), and the
context bundle actually carries what the old planner never received.
"""
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import actors
from autokernel.loop.loop import Hypothesis


class Backoff(unittest.TestCase):

    def test_a_transient_streak_backs_off_exponentially(self):
        slept = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise actors.ProviderTransient("401")
            return "ok"

        result, streak = actors._with_backoff(flaky, sleep=slept.append)
        self.assertEqual(result, "ok")
        self.assertEqual(streak, 2)
        self.assertEqual(slept, [actors.BACKOFF_S[0], actors.BACKOFF_S[1]])

    def test_it_gives_up_rather_than_spinning(self):
        """284 failures in 23 minutes is what no bound looks like."""
        slept = []

        def always():
            raise actors.ProviderTransient("401")

        with self.assertRaises(actors.ProviderTransient) as caught:
            actors._with_backoff(always, sleep=slept.append)
        self.assertIn("consecutive", str(caught.exception))
        self.assertEqual(len(slept), len(actors.BACKOFF_S) - 1)

    def test_a_first_try_success_sleeps_not_at_all(self):
        slept = []
        result, streak = actors._with_backoff(lambda: "fine", sleep=slept.append)
        self.assertEqual((result, streak, slept), ("fine", 0, []))


class JsonExtraction(unittest.TestCase):

    def test_it_takes_the_last_complete_object(self):
        text = ('thinking out loud {"draft": 1}\nfinal answer:\n'
                '{"mechanism_id": "akm-x", "statement": "s"}\n')
        self.assertEqual(actors._extract_json(text)["mechanism_id"], "akm-x")

    def test_nested_objects_do_not_confuse_it(self):
        self.assertEqual(
            actors._extract_json('{"a": {"b": 2}, "c": 3}')["c"], 3)

    def test_no_json_is_a_transient_not_a_crash(self):
        with self.assertRaises(actors.ProviderTransient):
            actors._extract_json("I could not complete this task.")

    def test_malformed_json_is_a_transient(self):
        with self.assertRaises(actors.ProviderTransient):
            actors._extract_json("{not: valid}")


class ContextBundle(unittest.TestCase):
    """Everything rendered here is something the old loop measured and discarded."""

    def test_the_profile_reaches_the_actor(self):
        text = actors.render_context({"kernel_hotspots": [
            {"signature": "mul_mat_vec_q<Q4_K>", "total_duration_ns": 700000,
             "calls": 13803, "share_of_device_time": 0.7}]})
        self.assertIn("mul_mat_vec_q<Q4_K>", text)
        self.assertIn("70.00%", text)
        self.assertIn("13803", text)

    def test_an_absent_profile_says_so_rather_than_inviting_a_guess(self):
        text = actors.render_context({})
        self.assertIn("no profile yet", text)

    def test_prior_refusals_are_rendered_as_things_to_answer(self):
        text = actors.render_context(
            {"prior_hypothesis_rejections": ["already measured null in epoch 4de6"]})
        self.assertIn("already measured null in epoch 4de6", text)
        self.assertIn("answer these, do not re-derive", text)

    def test_a_stale_epoch_record_is_marked_not_comparable(self):
        text = actors.render_context({"prior_experiments": [
            {"mechanism_id": "akm-old", "status": "screened_out",
             "effect_fraction": 0.001, "stale_epoch": True}]})
        self.assertIn("akm-old", text)
        self.assertIn("STALE EPOCH", text)
        self.assertIn("NUMBER is not", text)

    def test_a_refusal_reason_from_memory_is_carried(self):
        text = actors.render_context({"prior_experiments": [
            {"mechanism_id": "akm-old", "status": "authoring_refused",
             "refusal_reason": "derives undeclared symbols ['<file-scope>']"}]})
        self.assertIn("derives undeclared symbols", text)

    def test_the_operator_inbox_is_surfaced(self):
        text = actors.render_context({"inbox": ["try IQ4_XS at the 64-VGPR knee"]})
        self.assertIn("IQ4_XS", text)
        self.assertIn("Operator suggestions", text)


class PlannerContract(unittest.TestCase):

    def test_a_complete_hypothesis_parses(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        payload = ('{"mechanism_id": "akm-q4k-branchless", "statement": "s", '
                   '"falsifier": "f", "target_surface": "ggml/src/ggml-cuda/mmvq.cu", '
                   '"target_symbol": "vec_dot_q4_K_q8_1"}')
        with mock.patch.object(actors, "_run_agent", return_value=payload):
            got = planner.propose({})
        self.assertIsInstance(got, Hypothesis)
        self.assertEqual(got.mechanism_id, "akm-q4k-branchless")

    def test_an_incomplete_hypothesis_is_a_transient(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"mechanism_id": "akm-x"}'):
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.propose({})
        self.assertIn("missing", str(caught.exception))

    def test_authoring_with_no_paths_is_a_transient(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent", return_value='{"paths": []}'):
            with self.assertRaises(actors.ProviderTransient):
                planner.author(
                    Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})


class CriticContract(unittest.TestCase):

    def test_a_reasonless_rejection_is_made_explicit_not_crashed_on(self):
        """The loop refuses a reasonless rejection; the critic must not hand it one."""
        critic = actors.CodexCritic(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": false}'):
            review = critic.review_hypothesis(
                Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertFalse(review.accepted)
        self.assertIn("without stating a reason", review.reason)

    def test_an_acceptance_passes_through(self):
        critic = actors.CodexCritic(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": true}'):
            self.assertTrue(critic.review_hypothesis(
                Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {}).accepted)


class PlaceholderEchoes(unittest.TestCase):
    """The first real run parsed the prompt's own template as the answer.

    `_extract_json` takes the LAST complete JSON object in stdout, and codex echoes
    the requested shape while reasoning. So `{"paths": ["<relative path you
    changed>"]}` -- the spec, not the reply -- was parsed as a result, and the loop
    recorded a transient while codex was still working.
    """

    def test_the_template_is_recognised_as_a_placeholder(self):
        for echoed in ("<relative path you changed>", "akm-<short-slug>",
                       "<the function you will change>", "e.g. mmvq.cu",
                       "  YOUR PATH HERE  "):
            self.assertTrue(actors._is_placeholder(echoed), echoed)

    def test_a_real_path_is_not_a_placeholder(self):
        for real in ("ggml/src/ggml-cuda/vecdotq.cuh", "mmvq.cu",
                     "akm-q4k-branchless"):
            self.assertFalse(actors._is_placeholder(real), real)

    def test_authoring_refuses_an_echoed_template(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"paths": ["<relative path you changed>"]}'):
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIn("echoed the prompt template", str(caught.exception))

    def test_a_hypothesis_echoing_the_template_is_refused(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        payload = ('{"mechanism_id": "akm-<short-slug>", "statement": "s", '
                   '"falsifier": "f", "target_surface": "a.cu", '
                   '"target_symbol": "<the function you will change>"}')
        with mock.patch.object(actors, "_run_agent", return_value=payload):
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.propose({})
        self.assertIn("echoed the prompt template", str(caught.exception))


class TheWorktreeIsTheGroundTruth(unittest.TestCase):
    """An actor that SAYS it changed a file and did not must not pass."""

    def test_an_unchanged_worktree_refuses_the_claimed_path(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"paths": ["ggml/src/ggml-cuda/mmvq.cu"]}'), \
             mock.patch.object(actors.subprocess, "run") as ran:
            ran.return_value = mock.Mock(stdout="")          # git status: clean
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIn("worktree is unchanged", str(caught.exception))

    def test_a_genuinely_changed_worktree_passes(self):
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"paths": ["ggml/src/ggml-cuda/mmvq.cu"]}'), \
             mock.patch.object(actors.subprocess, "run") as ran:
            ran.return_value = mock.Mock(stdout=" M ggml/src/ggml-cuda/mmvq.cu")
            got = planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertEqual(got, ("ggml/src/ggml-cuda/mmvq.cu",))


class TheActorMustNotSpendTheLoopsCompute(unittest.TestCase):
    def test_the_authoring_prompt_forbids_building(self):
        """codex started its own `cmake --build -j 16` on the first real run."""
        planner = actors.CodexPlanner(workspace=Path("/tmp"))
        captured = {}

        def capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"paths": ["ggml/src/ggml-cuda/mmvq.cu"]}'

        with mock.patch.object(actors, "_run_agent", side_effect=capture), \
             mock.patch.object(actors.subprocess, "run") as ran:
            ran.return_value = mock.Mock(stdout=" M ggml/src/ggml-cuda/mmvq.cu")
            planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIn("DO NOT BUILD", captured["prompt"])
        self.assertIn("the loop owns the build", captured["prompt"].lower())


if __name__ == "__main__":
    unittest.main()
