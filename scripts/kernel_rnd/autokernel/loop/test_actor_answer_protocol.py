"""UFH14-B1 F2: the planner's budgeted answer protocol (`ActorSeat.answer_protocol="f2"`),
offline (`_run_agent` mocked, as in test_planner_salvage_turn.py).

Proven on the 27B in DS41-C95 F12 (arm cxf12): JSON-first rule, forced answer at 65% of
the budget with no resume-compaction, a refine turn for the rest, final answer = the last
COMPLETE answer. What must hold here:

* a natural end is ONE call (budget = force_frac x planner budget), parsed as today;
* a budget cut with no complete reply -> forced turn (session continued, compaction OFF,
  <= ANSWER_PHASE_CAP_S) -> refine turn (normal config) with the rest of the budget;
* a cut call that already printed a complete reply skips the forced turn;
* a later incomplete answer never overrides an earlier complete one; an abstention is a
  complete answer;
* nothing complete -> the salvage turn, resumed with compaction off;
* a stop during a turn is a stop; protocol off = the historical call.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from autokernel.loop import actor_metrics, actors
from autokernel.loop import loop as loop_mod

MODEL = "test/answer"
HYP = {"mechanism_id": "akm-forced", "statement": "fuse the two loads",
       "falsifier": "no tg gain in the matched A/B",
       "target_surface": "ggml/src/ggml-cuda/mmq.cuh", "target_symbol": "load_tiles"}
REFINED = {**HYP, "mechanism_id": "akm-refined"}
PROVISIONAL = {**HYP, "mechanism_id": "akm-provisional"}
SALVAGED = {**HYP, "mechanism_id": "akm-salvaged"}
JUNK = "Let me look at one more file before answering."


def _cut(provisional=None, session_id="ses_root") -> actors.ActorBudgetExhausted:
    exc = actors.ActorBudgetExhausted("budget_exhausted: the planner call spent its 2925s budget")
    exc.session_id = session_id
    exc.started_monotonic = time.monotonic()
    exc.provisional_text = json.dumps(provisional) if provisional else None
    return exc


class AnswerProtocol(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        patch = mock.patch.object(actors, "_schema_repair", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)

    def _planner(self, protocol="f2", salvage_s=900, timeout_s=7200) -> actors.AgentPlanner:
        return actors.AgentPlanner(
            workspace=self.ws, backend=actors.backend_for(MODEL, "high"), timeout_s=timeout_s,
            seat=actors.ActorSeat(bounded=False, planner_budget_s=4500,
                                  planner_salvage_s=salvage_s, answer_protocol=protocol))

    def _run(self, planner, *replies):
        calls, queue = [], list(replies)

        def fake(prompt, **kw):
            config = (kw.get("env") or {}).get("OPENCODE_CONFIG")
            content = (json.loads(Path(config).read_text())
                       if config and Path(config).is_file() else None)
            calls.append({"prompt": prompt, **kw, "config": config, "config_content": content})
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            try:
                return planner.propose({}), calls
            except BaseException as exc:  # noqa: BLE001
                return exc, calls

    def _rows(self) -> list[dict]:
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        if not log.exists():
            return []
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        return [r for r in rows if r.get("schema") == actor_metrics.ANSWER_PROTOCOL_SCHEMA]

    def test_natural_end_is_one_call_with_the_json_first_rule(self):
        result, calls = self._run(self._planner(), json.dumps(HYP))
        self.assertEqual(len(calls), 1)
        self.assertIn(actors.ANSWER_JSON_FIRST_RULE, calls[0]["prompt"])
        self.assertAlmostEqual(calls[0]["budget_s"], 4500 * 0.65)
        self.assertTrue(calls[0]["raise_provisional"])
        self.assertEqual(result.mechanism_id, "akm-forced")
        self.assertEqual(result.planner_report_source, "")
        self.assertEqual(self._rows(), [])

    def test_cut_then_forced_then_refine_takes_the_refined_answer(self):
        result, calls = self._run(self._planner(), _cut(), json.dumps(HYP), json.dumps(REFINED))
        self.assertEqual(len(calls), 3)
        main, forced, refine = calls
        self.assertTrue(forced["prompt"].startswith("Checkpoint: 65% of your time budget"))
        self.assertEqual(forced["session_id"], "ses_root")
        self.assertLessEqual(forced["budget_s"], actors.ANSWER_PHASE_CAP_S)
        # The forced turn resumes with compaction OFF, on a sibling copy of the config.
        self.assertTrue(forced["config"].endswith(actors.NO_COMPACTION_SUFFIX))
        self.assertIs(forced["config_content"]["compaction"]["auto"], False)
        self.assertEqual({k: v for k, v in forced["config_content"].items() if k != "compaction"},
                         main["config_content"])
        self.assertTrue(forced["env"][actors.SEAT_ENV_ARM].endswith("+forced"))
        # The refine turn: the rest of the budget, normal config (compaction allowed).
        self.assertIn("minutes left", refine["prompt"])
        self.assertEqual(refine["config"], main["config"])
        self.assertGreater(refine["budget_s"], 4000)
        self.assertTrue(refine["env"][actors.SEAT_ENV_ARM].endswith("+refine"))
        self.assertEqual(result.mechanism_id, "akm-refined")
        self.assertEqual(result.planner_report_source, actors.REPORT_SOURCE_REFINE_TURN)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer_phase"], "refine")
        self.assertEqual([p["phase"] for p in rows[0]["phases"]], ["main", "forced", "refine"])
        self.assertEqual([p["result"] for p in rows[0]["phases"]],
                         ["no_complete_answer", "proposal", "proposal"])

    def test_an_incomplete_refine_never_overrides_the_forced_answer(self):
        incomplete = json.dumps({"mechanism_id": "akm-half"})
        for refine in (JUNK, incomplete, actors.ActorBudgetExhausted("budget_exhausted: refine"),
                       actors.ActorTimedOut("actor exceeded"), OSError("disk")):
            with self.subTest(refine=str(refine)[:30]):
                result, calls = self._run(self._planner(), _cut(), json.dumps(HYP), refine)
                self.assertEqual(len(calls), 3)
                self.assertEqual(result.mechanism_id, "akm-forced")
                self.assertEqual(result.planner_report_source, actors.REPORT_SOURCE_FORCED_TURN)

    def test_a_provisional_reply_skips_the_forced_turn(self):
        result, calls = self._run(self._planner(), _cut(provisional=PROVISIONAL), JUNK)
        self.assertEqual(len(calls), 2)
        self.assertIn("minutes left", calls[1]["prompt"])
        self.assertEqual(result.mechanism_id, "akm-provisional")
        self.assertEqual(result.planner_report_source, actors.REPORT_SOURCE_PROVISIONAL)
        self.assertEqual(self._rows()[0]["answer_phase"], "main")

    def test_an_abstention_is_a_complete_answer(self):
        abstain = json.dumps({"abstain": "the profile shows no headroom"})
        result, calls = self._run(self._planner(), _cut(), abstain, JUNK)
        self.assertEqual(len(calls), 3)
        self.assertIsInstance(result, loop_mod.Abstain)
        self.assertTrue(self._rows()[0]["abstained"])
        # ... and a later complete hypothesis still wins over it (last complete wins).
        result, _ = self._run(self._planner(), _cut(), abstain, json.dumps(REFINED))
        self.assertEqual(result.mechanism_id, "akm-refined")

    def test_nothing_complete_falls_back_to_the_salvage_turn_without_compaction(self):
        result, calls = self._run(self._planner(), _cut(), JUNK, json.dumps(SALVAGED))
        self.assertEqual(len(calls), 3)          # main, forced (junk), salvage; no refine
        salvage = calls[2]
        self.assertEqual(salvage["prompt"], actors.PLANNER_SALVAGE_MESSAGE)
        self.assertTrue(salvage["config"].endswith(actors.NO_COMPACTION_SUFFIX))
        self.assertEqual(result.mechanism_id, "akm-salvaged")
        self.assertEqual(result.planner_report_source, actors.REPORT_SOURCE_SALVAGE_TURN)
        self.assertEqual(self._rows()[0]["result"], "no_complete_answer")

    def test_salvage_off_and_nothing_complete_ends_budget_exhausted(self):
        cut = _cut()
        result, calls = self._run(self._planner(salvage_s=0), cut, JUNK)
        self.assertEqual(len(calls), 2)
        self.assertIs(result, cut)

    def test_an_unknown_session_cannot_continue(self):
        result, calls = self._run(self._planner(salvage_s=0), _cut(session_id=None))
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(result, actors.ActorBudgetExhausted)
        self.assertEqual(self._rows()[0]["phases"][1]["reason"], "session_unknown")

    def test_a_stop_during_a_turn_is_a_stop(self):
        result, calls = self._run(self._planner(), _cut(), json.dumps(HYP),
                                  actors.ActorStopped("stop asked"))
        self.assertIsInstance(result, actors.ActorStopped)
        self.assertEqual(self._rows()[-1]["result"], "stopped")

    def test_protocol_off_is_the_historical_call(self):
        result, calls = self._run(self._planner(protocol="off"), json.dumps(HYP))
        self.assertEqual(len(calls), 1)
        self.assertNotIn(actors.ANSWER_JSON_FIRST_RULE, calls[0]["prompt"])
        self.assertEqual(calls[0]["budget_s"], 4500)
        self.assertNotIn("raise_provisional", calls[0])

    def test_the_hard_timeout_bounds_every_turn(self):
        result, calls = self._run(self._planner(timeout_s=3000), _cut(), json.dumps(HYP),
                                  json.dumps(REFINED))
        for call in calls[1:]:
            self.assertLessEqual(call["budget_s"], 3000 - actors.STOP_GRACE_S)
            self.assertLessEqual(call["timeout_s"], 3000 - actors.STOP_GRACE_S)


class ProvisionalRaise(unittest.TestCase):
    """`_budget_exhausted(raise_provisional=True)` keeps the complete reply on the
    exception instead of returning it; the default path is unchanged."""

    def _call(self, stdout, **kw):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(actors, "_persist_reply", return_value=None), \
                mock.patch.object(actors, "_record_metrics", return_value=None), \
                mock.patch.object(actors, "_record_call", return_value=None):
            return actors._budget_exhausted(
                Path(tmp), actors.backend_for(MODEL, "high"), "p", argv=["x"], returncode=-15,
                stdout=stdout, stderr="", started=time.monotonic(), started_at=time.time(),
                before_ids=set(), arm=None, env=None, collect_metrics=False,
                schema=actors.HYPOTHESIS_SCHEMA, budget_s=2925.0, **kw)

    def test_default_returns_a_complete_reply(self):
        self.assertEqual(json.loads(self._call(json.dumps(HYP))), HYP)

    def test_raise_provisional_carries_it(self):
        with self.assertRaises(actors.ActorBudgetExhausted) as ctx:
            self._call(json.dumps(HYP), raise_provisional=True)
        self.assertEqual(json.loads(ctx.exception.provisional_text), HYP)
        self.assertIn("provisional complete reply", str(ctx.exception))
        with self.assertRaises(actors.ActorBudgetExhausted) as ctx:
            self._call(JUNK, raise_provisional=True)
        self.assertIsNone(ctx.exception.provisional_text)
        self.assertIn("without a complete reply", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
