"""A loop nobody can see is a loop nobody can trust.

The operator's first question about the rebuilt loop was "I have zero visibility on
what's going on", while the dashboard correctly reported the SUPERSEDED deployment as
stopped. These tests pin the contract that closes that gap -- and specifically the
three-valued freshness, because collapsing `absent` into `stale` is how a dead
producer previously rendered as a clean, empty, trusted page.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from autokernel.loop import status


def _outcomes():
    return [
        {"status": "kept", "mechanism_id": "akm-a", "effect_fraction": 0.031},
        {"status": "measured_null", "mechanism_id": "akm-b", "effect_fraction": 0.002},
        {"status": "planner_transient", "reason": "actor exited 1"},
        {"status": "refused_at_formation", "reason": "already measured"},
    ]


class Write(unittest.TestCase):

    def test_it_counts_every_disposition_not_just_the_wins(self):
        """A board that shows only keeps is how 0 promotions looked like progress."""
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=0.973,
                         outcomes=_outcomes(), iterations_planned=10)
            body = status.read(Path(tmp))
        self.assertEqual(body["dispositions"], {
            "kept": 1, "measured_null": 1, "planner_transient": 1,
            "refused_at_formation": 1})
        self.assertEqual(body["iterations_done"], 4)
        self.assertEqual(body["measurements_reached"], 2)

    def test_the_noise_floor_is_on_the_surface(self):
        """The bar a candidate must clear, and it must be visible."""
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=0.973)
            self.assertAlmostEqual(status.read(Path(tmp))["noise_floor_pct"], 0.973)

    def test_recent_is_newest_first_and_includes_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=1.0,
                         outcomes=_outcomes())
            recent = status.read(Path(tmp))["recent"]
        self.assertEqual(recent[0]["status"], "refused_at_formation")
        self.assertIn("planner_transient", [row["status"] for row in recent])

    def test_the_write_is_atomic(self):
        """A dashboard polling a half-written file reports something never true."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(20):
                status.write(root, state="running", epoch="e" * 64,
                             campaign_id="ak-loop", anchor_commit="a" * 40,
                             surface="pp512", pairs=5, noise_floor_pct=1.0,
                             outcomes=_outcomes()[:index % 4])
                json.loads((root / status.STATUS_FILENAME).read_text())
            # No temp files survive a clean run.
            self.assertEqual([p.name for p in root.glob(".status-*")], [])

    def test_it_creates_the_store_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "does-not-exist-yet"
            status.write(root, state="starting", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=1.0)
            self.assertIsNotNone(status.read(root))


class Freshness(unittest.TestCase):
    """Three-valued, never two."""

    def _body(self, age_s: float, stale_after: int = 1800):
        written = datetime.now(timezone.utc) - timedelta(seconds=age_s)
        return {"generated_at": written.isoformat().replace("+00:00", "Z"),
                "stale_after_s": stale_after}

    def test_absent_is_not_stale(self):
        got = status.freshness(None)
        self.assertEqual(got["state"], "absent")
        self.assertIsNone(got["age_s"])
        self.assertIn("never run", got["detail"])

    def test_fresh_within_the_envelope(self):
        self.assertEqual(status.freshness(self._body(60))["state"], "fresh")

    def test_stale_past_the_envelope(self):
        got = status.freshness(self._body(3600))
        self.assertEqual(got["state"], "stale")
        self.assertGreater(got["age_s"], 1800)

    def test_the_envelope_is_taken_from_the_record_not_a_constant(self):
        """A loop that declares a longer cadence must not read as stale."""
        self.assertEqual(
            status.freshness(self._body(3600, stale_after=7200))["state"], "fresh")

    def test_a_malformed_stamp_is_malformed_not_fresh(self):
        got = status.freshness({"generated_at": "not a timestamp"})
        self.assertEqual(got["state"], "malformed")

    def test_a_missing_stamp_is_malformed_not_fresh(self):
        self.assertEqual(status.freshness({})["state"], "malformed")

    def test_the_four_states_are_distinct(self):
        """If any two collapse, a dead producer can render as a live one."""
        states = {
            status.freshness(None)["state"],
            status.freshness(self._body(1))["state"],
            status.freshness(self._body(99999))["state"],
            status.freshness({"generated_at": "x"})["state"],
        }
        self.assertEqual(states, {"absent", "fresh", "stale", "malformed"})


class Read(unittest.TestCase):

    def test_a_missing_file_reads_as_none_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(status.read(Path(tmp)))

    def test_a_corrupt_file_reads_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / status.STATUS_FILENAME).write_text("{ half written",
                                                            encoding="utf-8")
            self.assertIsNone(status.read(Path(tmp)))

    def test_a_non_object_file_reads_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / status.STATUS_FILENAME).write_text("[1,2,3]", encoding="utf-8")
            self.assertIsNone(status.read(Path(tmp)))


class TheUtilizationLegMustBeWired(unittest.TestCase):
    """The dashboard found `gpu` empty on every publish: `publish()` accepted a
    `gpu=` argument and no callsite passed one.

    A contract leg that is declared and never populated is worse than an absent one,
    because the page renders a field that will never fill. This is the number that
    would have caught 1.403 hours of GPU held against 29.0 hours of compiling.
    """

    def test_a_populated_reading_survives_the_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=1.0,
                         gpu={"claim_held_s": 3600.0,
                              "device_seconds_under_load": 60.0,
                              "gpu_seconds_idle_while_claimed": 3540.0,
                              "idle_fraction_while_claimed": 0.9833})
            gpu = status.read(Path(tmp))["gpu"]
        self.assertAlmostEqual(gpu["claim_held_s"], 3600.0)
        self.assertAlmostEqual(gpu["idle_fraction_while_claimed"], 0.9833)

    def test_the_runner_passes_a_reading_rather_than_an_empty_map(self):
        """Pins the defect: run.py must not call publish() without gpu data."""
        import re
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertIn("def gpu_reading(", source)
        # publish() must default to the computed reading, never to {}.
        self.assertIn("gpu=gpu if gpu is not None else gpu_reading(outcomes)", source)
        self.assertNotIn('gpu=gpu or {}', source)

    def test_idle_is_derived_from_both_halves(self):
        """Held without busy, or busy without held, is not a utilization figure."""
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=1.0, gpu={})
            self.assertEqual(status.read(Path(tmp))["gpu"], {},
                             "an unreported reading stays empty rather than "
                             "fabricating 0s busy / 100% idle")


if __name__ == "__main__":
    unittest.main()


class TheLoopBeatsWithinAnIteration(unittest.TestCase):
    """An iteration can outlive the freshness envelope.

    Once the bundle carried program.md plus the seeds, a single planner call ran past
    18 minutes against a 30-minute envelope -- so a HEALTHY loop was on course to read
    `stale`. Raising the envelope to cover it is the wrong fix: it makes a genuinely
    dead loop look alive for longer. The loop reports what it is doing instead.
    """

    def test_the_step_is_carried_on_the_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=1.0,
                         step="critic pass 1: reviewing the hypothesis")
            body = status.read(Path(tmp))
        self.assertEqual(body["step"], "critic pass 1: reviewing the hypothesis")

    def test_no_step_is_null_rather_than_a_stale_previous_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e" * 64,
                         campaign_id="ak-loop", anchor_commit="a" * 40,
                         surface="pp512", pairs=5, noise_floor_pct=1.0)
            self.assertIsNone(status.read(Path(tmp))["step"])


class EveryLongCallBeats(unittest.TestCase):
    """A beat that only fires on the cheap steps proves nothing."""

    def test_each_actor_and_device_call_is_preceded_by_a_beat(self):
        from unittest import mock
        from autokernel.loop import bench, gates, loop as loop_mod
        beats = []
        h = loop_mod.Hypothesis("akm-x", "s", "f", "a.cu", "sym")
        planner = mock.Mock()
        planner.propose.return_value = h
        planner.author.return_value = ("a.cu",)
        critic = mock.Mock()
        critic.review_hypothesis.return_value = loop_mod.Review(True)
        critic.review_patch.return_value = loop_mod.Review(True)
        comparison = bench.Comparison(
            surface="pp512", anchor_samples=[1.0], candidate_samples=[1.05],
            effect=0.05, estimator="median_over_median", pairs=5,
            noise_floor_pct=1.0, residency={})
        loop_mod.iterate(
            planner=planner, critic=critic, context={},
            measure=lambda *a: comparison,
            gate=lambda *a: (True, [gates.Verdict("compile", True)]),
            commit=lambda *a: "abc1234",
            on_step=beats.append)
        for expected in ("proposing", "critic pass 1", "authoring",
                         "critic pass 2", "building", "measuring"):
            self.assertTrue(any(expected in b for b in beats),
                            f"no beat before {expected}: {beats}")

    def test_a_raising_hook_does_not_kill_the_iteration(self):
        from unittest import mock
        from autokernel.loop import loop as loop_mod
        planner = mock.Mock()
        planner.propose.side_effect = loop_mod.ActorTransient("provider down")
        outcome = loop_mod.iterate(
            planner=planner, critic=mock.Mock(), context={},
            measure=mock.Mock(), gate=mock.Mock(), commit=mock.Mock(),
            on_step=mock.Mock(side_effect=RuntimeError("disk full")))
        self.assertEqual(outcome.status, "planner_transient")
