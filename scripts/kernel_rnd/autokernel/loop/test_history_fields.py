"""AK-HR-3: the nullable, additive history fields are written, null when unknown, and
leave every existing reader unaffected."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from autokernel.controller import experiments
from autokernel.loop import actors, archive, history_fields, loop
from autokernel.loop.test_loop import _Critic, _Planner, _hypothesis, _run

NEW_KEYS = set(history_fields.ATTEMPT_KEYS)


def _hyp(mechanism, statement="reorder the scatter", symbol="scatter_rows"):
    return loop.Hypothesis(mechanism_id=mechanism, statement=statement, falsifier="no gain",
                           target_surface="ggml/src/x.cpp", target_symbol=symbol)


class TrialFields(unittest.TestCase):

    def test_outside_a_trial_everything_is_null(self):
        self.assertIsNone(history_fields.current())
        self.assertEqual(history_fields.metrics_fields(),
                         {"trial_id": None, "mechanism_family_bucket": None})
        history_fields.note_call("planner", 3.0, None, None)    # no trial: a no-op

    def test_stage_seconds_are_null_until_a_stage_runs(self):
        with history_fields.trial_scope() as trial:
            history_fields.note_call("planner", 2.5, None, None)
            history_fields.note_call("planner", 1.0, "budget_exhausted", None)
            snap = trial.snapshot()
        self.assertEqual(snap["stage_seconds"]["planner"], 3.5)
        self.assertIsNone(snap["stage_seconds"]["critic"])
        self.assertIsNone(snap["trial_tokens"])
        self.assertEqual(snap["failure_class"], "budget_exhausted")

    def test_tokens_sum_only_when_a_call_reported_them(self):
        with history_fields.trial_scope() as trial:
            history_fields.note_call("author", 1.0, None,
                                     {"totals": {"decoded_tokens": 10, "prompt_tokens": 5}})
            self.assertEqual(trial.snapshot()["trial_tokens"], 15)

    def test_timed_wrapper_changes_nothing_but_the_clock(self):
        with history_fields.trial_scope() as trial:
            self.assertEqual(history_fields.timed("gate", lambda a, b=1: a + b)(2, b=3), 5)
            with self.assertRaises(KeyError):
                history_fields.timed("measure", lambda: {}["x"])()
            snap = trial.snapshot()
        self.assertIsNotNone(snap["stage_seconds"]["gate"])
        self.assertIsNotNone(snap["stage_seconds"]["measure"])

    @staticmethod
    def _fake_build(seconds, returncode=0):
        from types import SimpleNamespace
        import time as _time

        def run(*_a, **_k):
            _time.sleep(seconds / 2)
            return SimpleNamespace(returncode=returncode, stdout="", stderr="")
        return run

    def _compile(self, seconds, returncode=0):
        from unittest import mock
        from autokernel.loop import gates
        with mock.patch.object(gates.subprocess, "run", side_effect=self._fake_build(seconds, returncode)):
            return gates.compiles("/src", "/build", cmake_defines=(), jobs=1, cpu_list=None)

    def test_run_all_splits_build_and_correctness_inside_the_gate(self):
        """AK-HR-3b: `compiles` times the build; run_all times every other check."""
        from autokernel.loop import gates
        import time as _time

        def check(name, seconds):
            def run():
                _time.sleep(seconds)
                return gates.Verdict(name, True)
            return run
        with history_fields.trial_scope() as trial:
            passed, verdicts = history_fields.timed("gate", gates.run_all)(
                lambda: self._compile(0.04), check("op_correctness", 0.03),
                check("cpu_route_reference", 0.01))
            snap = trial.snapshot()["stage_seconds"]
        self.assertTrue(passed)
        self.assertEqual(verdicts[0], gates.Verdict("compile", True))
        self.assertGreaterEqual(snap["build"], 0.04)
        self.assertGreaterEqual(snap["correctness"], 0.04)
        self.assertLess(snap["correctness"], 0.04 + 0.04)          # the build is not counted twice
        self.assertLessEqual(snap["build"] + snap["correctness"], snap["gate"] + 0.002)

    def test_a_build_outside_run_all_is_still_build_time(self):
        """The GPU gate calls its build before the quiet window, outside run_all."""
        with history_fields.trial_scope() as trial:
            verdict = self._compile(0.02, returncode=1)
            snap = trial.snapshot()["stage_seconds"]
        self.assertEqual(verdict.gate, "configure")
        self.assertGreaterEqual(snap["build"], 0.01)
        self.assertIsNone(snap["correctness"])

    def test_a_failed_build_leaves_correctness_null_and_no_trial_is_a_noop(self):
        from autokernel.loop import gates
        with history_fields.trial_scope() as trial:
            gates.run_all(lambda: self._compile(0.0, returncode=1),
                          lambda: gates.Verdict("op_correctness", True))
            snap = trial.snapshot()["stage_seconds"]
        self.assertIsNotNone(snap["build"])
        self.assertIsNone(snap["correctness"])          # short-circuited: never ran
        self.assertEqual(self._compile(0.0).gate, "compile")       # outside a trial: no-op

    def test_a_runtime_arm_gate_has_correctness_but_no_build(self):
        from autokernel.loop import gates
        with history_fields.trial_scope() as trial:
            gates.run_all(lambda: gates.Verdict("op_correctness", True))
            snap = trial.snapshot()["stage_seconds"]
        self.assertIsNone(snap["build"])
        self.assertIsNotNone(snap["correctness"])

    def test_family_bucket_reuses_the_existing_four_family_helper(self):
        row = {"mechanism_id": "akm-prefetch-next-row", "statement": "x"}
        self.assertEqual(history_fields.family_bucket(row), actors._mechanism_family(row))
        self.assertEqual(history_fields.family_bucket(row), "prefetch/cache")
        self.assertIsNone(history_fields.family_bucket({"mechanism_id": "akm-zzz"}))


class IterateWritesTheFields(unittest.TestCase):

    def test_a_kept_attempt_carries_the_fields_and_every_old_key(self):
        outcome, _ = _run(_Planner([_hyp("akm-prefetch-rows")]), _Critic([], []))
        row = outcome.to_attempt()
        self.assertTrue(NEW_KEYS <= set(row))
        self.assertEqual(len(row["trial_id"]), 32)
        self.assertEqual(row["mechanism_family_bucket"], "prefetch/cache")
        self.assertIsNone(row["knob_descriptor"])           # a source candidate: no parameters
        self.assertIsNotNone(row["stage_seconds"]["measure"])
        self.assertIsNotNone(row["stage_seconds"]["gate"])
        self.assertIsNone(row["stage_seconds"]["critic"])   # fakes make no actor-call line
        self.assertIsNone(row["failure_class"])
        # The trial context does not outlive the turn.
        self.assertIsNone(history_fields.current())

    def test_unknown_family_is_null_not_invented(self):
        outcome, _ = _run(_Planner([_hyp("akm-zzz")]), _Critic([], []))
        self.assertIn("mechanism_family_bucket", outcome.to_attempt())
        self.assertIsNone(outcome.to_attempt()["mechanism_family_bucket"])

    def test_an_outcome_built_outside_iterate_keeps_its_historical_shape(self):
        row = loop.Outcome("kept", _hypothesis()).to_attempt()
        self.assertFalse(NEW_KEYS & set(row))

    def test_behavior_is_unchanged_by_the_wrappers(self):
        outcome, committed = _run(_Planner(), _Critic([], []))
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(committed["head"], "abc1234")

    def test_two_turns_get_distinct_trial_ids(self):
        a, _ = _run(_Planner(), _Critic([], []))
        b, _ = _run(_Planner(), _Critic([], []))
        self.assertNotEqual(a.history_fields["trial_id"], b.history_fields["trial_id"])


class MetricsLine(unittest.TestCase):

    def test_the_sibling_line_carries_trial_id_and_null_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            backend = actors.Backend(kind="codex", model="m", effort="low", binary="codex")
            with history_fields.trial_scope("t" * 32) as trial:
                actors._record_metrics(workspace, backend, role="planner", returncode=0,
                                       wall_s=4.0, timed_out=False, before_ids=set(),
                                       collect_metrics=False, schema=None, final_text=None,
                                       salvaged=False)
                self.assertEqual(trial.snapshot()["stage_seconds"]["planner"], 4.0)
            log = Path(tmp) / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
            line = json.loads(log.read_text().splitlines()[-1])
            self.assertEqual(line["trial_id"], "t" * 32)
            self.assertIsNone(line["mechanism_family_bucket"])
            self.assertEqual(line["schema"], "epyc.autokernel.actor_call_metrics.v1")
            # No trial: both keys present and null, every old key unchanged.
            actors._record_metrics(workspace, backend, role="critic", returncode=0, wall_s=1.0,
                                   timed_out=False, before_ids=set(), collect_metrics=False,
                                   schema=None, final_text=None, salvaged=False)
            bare = json.loads(log.read_text().splitlines()[-1])
            self.assertIsNone(bare["trial_id"])
            self.assertTrue({"wall_s", "failure_class", "opencode", "seat_arm"} <= set(bare))


class StoreCounters(unittest.TestCase):

    def _attempt(self, tokens, i):
        return {"status": "measured_null", "mechanism_id": f"m{i}", "reason": f"r{i}",
                "trial_id": f"{i:032d}", "trial_tokens": tokens, "stage_seconds": {},
                "turn_recorded_at": f"2026-10-20T00:00:0{i}Z"}

    def test_ordinal_and_cumulative_tokens_are_monotone_and_old_rows_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with experiments.ExperimentStore(root) as store:     # a pre-AK-HR-3 row
                store.record({"status": "kept", "mechanism_id": "old", "reason": "old"},
                             epoch="E", recorded_at="2026-10-19T00:00:00Z", campaign_id="c")
            for i, tokens in enumerate((100, None, 50), start=1):
                archive.record(root, self._attempt(tokens, i), epoch="E",
                               recorded_at=f"2026-10-20T00:00:0{i}Z", campaign_id="c")
            db = sqlite3.connect(root / "experiments.db")
            payloads = [json.loads(p) for (p,) in db.execute(
                "SELECT payload FROM experiments ORDER BY rowid")]
            self.assertNotIn("trial_ordinal", payloads[0])          # old row untouched
            self.assertEqual([p["trial_ordinal"] for p in payloads[1:]], [2, 3, 4])
            self.assertEqual([p["cumulative_tokens"] for p in payloads[1:]], [100, 100, 150])
            with experiments.ExperimentStore(root) as store:
                self.assertEqual(len(store.recall(epoch="E")), 4)   # existing reader unaffected

    def test_a_row_without_a_trial_is_stored_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive.record(root, {"status": "kept", "mechanism_id": "m", "reason": "r"},
                           epoch="E", recorded_at="2026-10-20T00:00:00Z", campaign_id="c")
            (payload,) = sqlite3.connect(root / "experiments.db").execute(
                "SELECT payload FROM experiments").fetchone()
            self.assertEqual(json.loads(payload),
                             {"status": "kept", "mechanism_id": "m", "reason": "r"})


if __name__ == "__main__":
    unittest.main()
