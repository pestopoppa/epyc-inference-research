"""The planner salvage turn (`--actor-planner-salvage-s`), offline.

DS41 run 10m (2026-09-28): the opencode planner (local 27B, plain seat, thinking on)
often never wrote its report -- 21-46 tool steps, up to ~87k output tokens, ended
`budget_exhausted` at the 4500 s wall budget. 5 of 9 planner calls produced nothing,
each wasting 45-75 min, though the investigation was usually done. What must hold:

* a PLANNER call ended by its wall budget with no complete reply, whose root opencode
  session is known, gets ONE follow-up call continuing that session
  (`opencode run ... --session <id>`, same backend/model/variant, same per-call
  OPENCODE_CONFIG, same lane) with `PLANNER_SALVAGE_MESSAGE` on stdin, under
  min(salvage budget, what is left of the hard timeout minus the stop grace);
* a valid proposal from it proceeds, stamped `planner_report_source: "salvage_turn"`
  (hypothesis, experiments row) and recorded in actor-calls.jsonl (`SALVAGE_SCHEMA` row,
  metrics row `salvage_turn`, arm suffix `+salvage`);
* anything else (junk, an abstention, an error, its own budget or timeout) ends the
  iteration exactly as before (`ActorBudgetExhausted`, reason `budget_exhausted...`),
  the attempt recorded; a stop during it is a stop;
* the knob at 0, an unknown session, the author and the critic: no second call at all;
* the continuation argv really continues the session in the installed opencode 1.18.31
  (wire test against a recording mock, no model server anywhere).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from autokernel.loop import actor_metrics, actors, cpu_window
from autokernel.loop import loop as loop_mod

#: A provider absent from the global opencode config: `_provider_base_url` is None, so
#: no schema-repair turn can ever reach a real server from these tests.
MODEL = "test/salvage"
HYPOTHESIS = {"mechanism_id": "akm-salvaged", "statement": "fuse the two loads",
              "falsifier": "no tg gain in the matched A/B",
              "target_surface": "ggml/src/ggml-cuda/mmq.cuh", "target_symbol": "load_tiles"}
JUNK = "I have reviewed the kernel extensively and summarized my findings above."


def _spent(session_id="ses_root", started=None) -> actors.ActorBudgetExhausted:
    exc = actors.ActorBudgetExhausted(
        "budget_exhausted: the planner call spent its 4500s budget without a complete "
        "reply and was ended (rc -15) after 4512s [opencode:test/salvage@high] -- not retried")
    exc.session_id = session_id
    exc.started_monotonic = time.monotonic() if started is None else started
    return exc


class _Base(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.backend = actors.backend_for(MODEL, "high")
        # Belt and braces: never a network repair turn from a test.
        patch = mock.patch.object(actors, "_schema_repair", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _planner(self, salvage_s=900, timeout_s=7200, **kw) -> actors.AgentPlanner:
        return actors.AgentPlanner(
            workspace=self.ws, backend=self.backend, timeout_s=timeout_s,
            seat=actors.ActorSeat(bounded=False, planner_budget_s=4500,
                                  planner_salvage_s=salvage_s), **kw)

    def _log_rows(self, schema=None) -> list[dict]:
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        if not log.exists():
            return []
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        return [r for r in rows if schema is None or r.get("schema") == schema]

    def _salvage_rows(self) -> list[dict]:
        return self._log_rows(actor_metrics.SALVAGE_SCHEMA)


class SalvageDecision(_Base):
    """AgentPlanner's salvage logic, `_run_agent` mocked."""

    def _run(self, planner, *replies):
        """propose() with `_run_agent` answering from `replies` in order (an exception
        instance is raised). Returns (result or exception, the recorded calls)."""
        calls, queue = [], list(replies)

        def fake(prompt, **kw):
            config = (kw.get("env") or {}).get("OPENCODE_CONFIG")
            # Whether the per-call config still exists DURING the call (the call scope
            # releases it when propose returns).
            calls.append({"prompt": prompt, **kw,
                          "config_exists": bool(config) and Path(config).is_file()})
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            try:
                return planner.propose({}), calls
            except BaseException as exc:  # noqa: BLE001 -- the test inspects it
                return exc, calls

    def test_a_valid_salvage_proposal_proceeds_stamped_salvage_turn(self):
        result, calls = self._run(self._planner(), _spent(), json.dumps(HYPOTHESIS))
        self.assertIsInstance(result, loop_mod.Hypothesis)
        self.assertEqual(result.mechanism_id, "akm-salvaged")
        self.assertEqual(result.planner_report_source, actors.REPORT_SOURCE_SALVAGE_TURN)
        self.assertEqual(result.to_dict()["planner_report_source"], "salvage_turn")
        self.assertEqual(len(calls), 2)
        first, salvage = calls
        self.assertIsNone(first.get("session_id"))
        self.assertEqual(first["budget_s"], 4500)
        # Same backend, same schema, same per-call config; the session continued.
        self.assertEqual(salvage["prompt"], actors.PLANNER_SALVAGE_MESSAGE)
        self.assertEqual(salvage["session_id"], "ses_root")
        self.assertIs(salvage["backend"], first["backend"])
        self.assertIs(salvage["schema"], actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(salvage["env"]["OPENCODE_CONFIG"], first["env"]["OPENCODE_CONFIG"])
        self.assertTrue(salvage["config_exists"])
        self.assertEqual(salvage["env"][actors.SEAT_ENV_ARM],
                         first["env"][actors.SEAT_ENV_ARM] + actors.SALVAGE_ARM_SUFFIX)
        self.assertEqual(salvage["budget_s"], 900)
        # The whole planner phase stays under the hard timeout (minus the stop grace).
        self.assertLessEqual(salvage["timeout_s"], 7200 - actors.STOP_GRACE_S)
        rows = self._salvage_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "proposal")
        self.assertEqual(rows[0]["mechanism_id"], "akm-salvaged")
        self.assertEqual(rows[0]["session_id"], "ses_root")
        self.assertEqual(rows[0]["budget_s"], 900)
        self.assertTrue(rows[0]["planner_reason"].startswith("budget_exhausted"))

    def test_the_salvaged_hypothesis_round_trips_through_a_checkpoint(self):
        result, _ = self._run(self._planner(), _spent(), json.dumps(HYPOTHESIS))
        again = loop_mod.Hypothesis(**result.to_dict())
        self.assertEqual(again, result)
        row = loop_mod.Outcome("kept", result).to_attempt()
        self.assertEqual(row["planner_report_source"], "salvage_turn")

    def test_an_ordinary_proposal_is_byte_identical(self):
        result, calls = self._run(self._planner(), json.dumps(HYPOTHESIS))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.planner_report_source, "")
        self.assertNotIn("planner_report_source", result.to_dict())
        self.assertEqual(result.to_dict(), HYPOTHESIS)
        self.assertEqual(self._salvage_rows(), [])

    def test_junk_from_the_salvage_turn_ends_budget_exhausted_and_is_recorded(self):
        spent = _spent()
        result, calls = self._run(self._planner(), spent, JUNK)
        self.assertEqual(len(calls), 2)
        self.assertIsInstance(result, actors.ActorBudgetExhausted)
        self.assertTrue(str(result).startswith(str(spent)))
        self.assertIn("salvage turn: no_proposal", str(result))
        self.assertIs(result.__cause__, spent)
        rows = self._salvage_rows()
        self.assertEqual([r["result"] for r in rows], ["no_proposal"])
        # iterate records it exactly as before: a planner transient, budget_exhausted.
        planner = mock.Mock()
        planner.propose.side_effect = result
        outcome = loop_mod.iterate(
            planner=planner, critic=mock.Mock(), context={},
            measure=mock.Mock(side_effect=AssertionError("no measurement")),
            gate=mock.Mock(side_effect=AssertionError("no gate")),
            commit=mock.Mock(side_effect=AssertionError("no commit")))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertTrue(outcome.reasons[0].startswith("budget_exhausted"))

    def test_an_abstention_is_not_a_proposal(self):
        result, calls = self._run(self._planner(), _spent(),
                                  json.dumps({"abstain": "the profile shows no headroom"}))
        self.assertEqual(len(calls), 2)
        self.assertIsInstance(result, actors.ActorBudgetExhausted)
        self.assertIn("salvage turn: abstained", str(result))
        self.assertEqual(self._salvage_rows()[0]["reason"], "the profile shows no headroom")

    def test_salvage_errors_budget_and_timeout_all_end_budget_exhausted(self):
        for failure, label in ((actors.ActorBudgetExhausted("budget_exhausted: salvage"),
                                "budget_exhausted"),
                               (actors.ActorTimedOut("actor exceeded 885s"), "timed_out"),
                               (actors.ProviderTransient("actor exited 1"), "no_proposal"),
                               (OSError("disk"), "error")):
            with self.subTest(label=label):
                result, calls = self._run(self._planner(), _spent(), failure)
                self.assertEqual(len(calls), 2)
                self.assertIsInstance(result, actors.ActorBudgetExhausted)
                self.assertIn(f"salvage turn: {label}", str(result))
                self.assertEqual(self._salvage_rows()[-1]["result"], label)

    def test_a_stop_during_the_salvage_turn_is_a_stop(self):
        result, calls = self._run(self._planner(), _spent(),
                                  loop_mod.ActorStopped("stop asked during the actor call"))
        self.assertEqual(len(calls), 2)
        self.assertIsInstance(result, loop_mod.ActorStopped)
        self.assertEqual(self._salvage_rows()[-1]["result"], "stopped")

    def test_flag_zero_is_byte_identical_no_second_call(self):
        spent = _spent()
        result, calls = self._run(self._planner(salvage_s=0), spent)
        self.assertIs(result, spent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._log_rows(), [])

    def test_the_library_default_seat_never_salvages(self):
        self.assertEqual(actors.ActorSeat().planner_salvage_s, 0)
        planner = actors.AgentPlanner(workspace=self.ws, backend=self.backend,
                                      seat=actors.ActorSeat(bounded=False, planner_budget_s=9))
        spent = _spent()
        result, calls = self._run(planner, spent)
        self.assertIs(result, spent)
        self.assertEqual(len(calls), 1)

    def test_session_unknown_means_no_salvage(self):
        spent = _spent(session_id=None)
        result, calls = self._run(self._planner(), spent)
        self.assertIs(result, spent)
        self.assertEqual(len(calls), 1)
        self.assertEqual([(r["result"], r["reason"]) for r in self._salvage_rows()],
                         [("skipped", "session_unknown")])

    def test_a_stop_asked_before_the_turn_skips_it(self):
        spent, calls = _spent(), []
        planner = self._planner(should_stop=lambda: bool(calls))   # asked mid-call

        def fake(prompt, **kw):
            calls.append(kw)
            raise spent
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            with self.assertRaises(actors.ActorBudgetExhausted) as caught:
                planner.propose({})
        self.assertIs(caught.exception, spent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._salvage_rows()[-1]["reason"], "stop_asked")

    def test_the_salvage_turn_never_runs_past_the_actor_timeout(self):
        # 600 s timeout, call started 100 s ago: 600 - 100 - grace left.
        started = time.monotonic() - 100
        _, calls = self._run(self._planner(timeout_s=600), _spent(started=started),
                             json.dumps(HYPOTHESIS))
        salvage = calls[1]
        left = 600 - 100 - actors.STOP_GRACE_S
        self.assertLessEqual(salvage["budget_s"], left + 1)
        self.assertLess(salvage["budget_s"], 900)
        self.assertLessEqual(salvage["timeout_s"], left + 1)
        # Too little left: no turn at all.
        spent = _spent(started=time.monotonic() - 590)
        result, calls = self._run(self._planner(timeout_s=600), spent)
        self.assertIs(result, spent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._salvage_rows()[-1]["reason"],
                         "no_time_left_under_actor_timeout")

    def test_a_non_opencode_backend_never_salvages(self):
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=actors.backend_for("gpt-5.6-sol", "high"),
            seat=actors.ActorSeat(bounded=False, planner_budget_s=9, planner_salvage_s=900))
        spent = _spent()
        result, calls = self._run(planner, spent)
        self.assertIs(result, spent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._log_rows(), [])

    def test_author_budget_exhaustion_never_triggers_it(self):
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=self.backend,
            seat=actors.ActorSeat(bounded=False, author_budget_s=4500, planner_salvage_s=900))
        hypothesis = loop_mod.Hypothesis(**HYPOTHESIS)
        spent = _spent()
        calls = []

        def fake(prompt, **kw):
            calls.append(kw)
            raise spent
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            with self.assertRaises(actors.ActorBudgetExhausted) as caught:
                planner.author(hypothesis, {})
        self.assertIs(caught.exception, spent)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].get("session_id"))
        self.assertEqual(self._salvage_rows(), [])

    def test_critic_budget_exhaustion_never_triggers_it(self):
        critic = actors.AgentCritic(workspace=self.ws, backend=self.backend,
                                    seat=actors.ActorSeat(bounded=False, planner_salvage_s=900))
        spent = _spent()
        calls = []

        def fake(prompt, **kw):
            calls.append(kw)
            raise spent
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            with self.assertRaises(actors.ActorBudgetExhausted) as caught:
                critic.review_hypothesis(loop_mod.Hypothesis(**HYPOTHESIS), {})
        self.assertIs(caught.exception, spent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._salvage_rows(), [])


# ------------------------------------------------------------------ real child processes

#: A stand-in opencode. Records its argv and stdin; WITHOUT `--session` it prints some
#: thinking and lingers (a planner that never stops to answer), WITH `--session` it prints
#: sys.argv[2] (the salvage reply) and exits -- or lingers too when that is empty.
SCRIPT = r"""
import json, os, sys, time
data = sys.stdin.read()
with open(sys.argv[1], "a") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "stdin": data, "pid": os.getpid()}) + "\n")
if "--session" in sys.argv and sys.argv[2]:
    sys.stdout.write(sys.argv[2]); sys.stdout.flush()
    sys.exit(0)
sys.stdout.write("thinking..."); sys.stdout.flush()
time.sleep(120)
"""


class _ScriptBackend(actors.Backend):
    marker: str = ""
    reply: str = ""

    def argv(self, prompt, workspace, *, read_only=False):
        return [sys.executable, "-c", SCRIPT, self.marker, self.reply]


class SalvageProcess(_Base):
    """The real call path (`_run_agent_in`, `_budget_exhausted`, `_run_stoppable`) with a
    child process standing in for opencode; `collect_metrics` on (`OPENCODE` pointed at
    the stand-in) and only the opencode CLI helpers (`session list`, `export`) faked."""

    def setUp(self):
        super().setUp()
        self.marker = Path(self._tmp.name) / "calls.jsonl"
        self.exports = Path(self._tmp.name) / "exports"
        self.exports.mkdir()
        self.call_started_ms = int(time.time() * 1000)
        for patch in (mock.patch.object(actors, "STOP_POLL_S", 0.05),
                      mock.patch.object(actors, "STOP_GRACE_S", 3.0),
                      mock.patch.object(actors, "OPENCODE", sys.executable),
                      mock.patch.object(actor_metrics, "list_session_ids",
                                        side_effect=self._sessions),
                      mock.patch.object(actor_metrics, "export_session",
                                        side_effect=self._export)):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        for line in (self.marker.read_text().splitlines() if self.marker.exists() else ()):
            try:
                os.kill(json.loads(line)["pid"], signal.SIGKILL)
            except (ProcessLookupError, ValueError, KeyError):
                pass
        super().tearDown()

    def _sessions(self, workspace, **kw):
        # The before-listing (no filter) sees nothing; after the call the root session
        # (and a fan-out scout, a child session) are new.
        return {"ses_root", "ses_scout"} if kw.get("strict") else set()

    def _export(self, workspace, session_id, out_path, **kw):
        now = int(time.time() * 1000)
        msgs = [
            {"info": {"role": "user", "sessionID": session_id,
                      "time": {"created": self.call_started_ms - 5000}}, "parts": []},
            {"info": {"role": "assistant", "sessionID": session_id, "finish": "tool-calls",
                      "time": {"created": self.call_started_ms - 4000},
                      "tokens": {"input": 10, "output": 700}}, "parts": []},
            {"info": {"role": "user", "sessionID": session_id, "time": {"created": now}},
             "parts": []},
            {"info": {"role": "assistant", "sessionID": session_id, "finish": "stop",
                      "time": {"created": now}, "tokens": {"input": 20, "output": 42}},
             "parts": []},
        ]
        info = {"id": session_id, **({"parentID": "ses_root"} if session_id == "ses_scout" else {})}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({"info": info, "messages": msgs}))

    def _backend(self, salvage_reply: str) -> _ScriptBackend:
        backend = _ScriptBackend(kind="opencode", model=MODEL, effort="high",
                                 binary=sys.executable)
        object.__setattr__(backend, "marker", str(self.marker))
        object.__setattr__(backend, "reply", salvage_reply)
        return backend

    def _launches(self) -> list[dict]:
        return [json.loads(line) for line in self.marker.read_text().splitlines()]

    def _planner(self, backend, salvage_s=30, timeout_s=60, **kw):
        return actors.AgentPlanner(
            workspace=self.ws, backend=backend, timeout_s=timeout_s,
            seat=actors.ActorSeat(bounded=False, planner_budget_s=1,
                                  planner_salvage_s=salvage_s), **kw)

    def test_budget_exhausted_then_salvage_turn_continues_the_root_session(self):
        started = time.monotonic()
        hypothesis = self._planner(self._backend(json.dumps(HYPOTHESIS))).propose({})
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(hypothesis.mechanism_id, "akm-salvaged")
        self.assertEqual(hypothesis.planner_report_source, "salvage_turn")
        first, salvage = self._launches()
        self.assertNotIn("--session", first["argv"])
        # The continuation: the planner call's argv plus `--session <root id>`, never
        # the scout's; the salvage message on stdin.
        self.assertEqual(salvage["argv"][:2], first["argv"][:2])
        self.assertEqual(salvage["argv"][-2:], ["--session", "ses_root"])
        self.assertEqual(salvage["stdin"], actors.PLANNER_SALVAGE_MESSAGE)
        metrics = self._log_rows(actor_metrics.METRICS_SCHEMA)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0]["failure_class"], "budget_exhausted")
        self.assertNotIn("salvage_turn", metrics[0])
        self.assertTrue(metrics[1]["salvage_turn"])
        self.assertEqual(metrics[1]["continued_session_id"], "ses_root")
        self.assertTrue(metrics[1]["seat_arm"].endswith("+budget1s+salvage"))
        self.assertEqual(metrics[1]["returncode"], 0)
        # Only the continuation's own steps are counted (not the original call's).
        self.assertTrue(metrics[1]["opencode"]["continued"])
        self.assertEqual(metrics[1]["opencode"]["totals"]["steps"], 1)
        self.assertEqual(metrics[1]["opencode"]["totals"]["decoded_tokens"], 42)
        v1 = self._log_rows("epyc.autokernel.actor_call.v1")
        if v1:   # the ROOT contract module may be absent on a bare checkout
            self.assertTrue(v1[-1]["seat"]["arm"].endswith("+salvage"))
        rows = self._salvage_rows()
        self.assertEqual([r["result"] for r in rows], ["proposal"])
        self.assertEqual(rows[0]["session_id"], "ses_root")

    def test_salvage_junk_keeps_budget_exhausted_and_records_the_attempt(self):
        with self.assertRaises(actors.ActorBudgetExhausted) as caught:
            self._planner(self._backend(JUNK)).propose({})
        self.assertTrue(str(caught.exception).startswith("budget_exhausted"))
        self.assertIn("salvage turn: no_proposal", str(caught.exception))
        self.assertEqual(len(self._launches()), 2)
        self.assertEqual([r["result"] for r in self._salvage_rows()], ["no_proposal"])

    def test_the_salvage_turn_has_its_own_budget_and_is_ended_through_the_stop_path(self):
        # An empty salvage reply makes the stand-in linger in the salvage call too.
        with mock.patch.object(actors, "PLANNER_SALVAGE_MIN_S", 1):
            started = time.monotonic()
            with self.assertRaises(actors.ActorBudgetExhausted) as caught:
                self._planner(self._backend(""), salvage_s=1).propose({})
        self.assertLess(time.monotonic() - started, 30)
        self.assertIn("salvage turn: budget_exhausted", str(caught.exception))
        launches = self._launches()
        self.assertEqual(len(launches), 2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
                Path(f"/proc/{item['pid']}").exists() and _alive(item["pid"])
                for item in launches):
            time.sleep(0.05)
        for item in launches:
            self.assertFalse(_alive(item["pid"]), "an actor survived its budget")
        metrics = self._log_rows(actor_metrics.METRICS_SCHEMA)
        self.assertEqual([m["failure_class"] for m in metrics],
                         ["budget_exhausted", "budget_exhausted"])
        self.assertTrue(metrics[1]["salvage_turn"])

    def test_flag_zero_launches_once_exactly_as_before(self):
        with self.assertRaises(actors.ActorBudgetExhausted) as caught:
            self._planner(self._backend(json.dumps(HYPOTHESIS)), salvage_s=0).propose({})
        self.assertNotIn("salvage turn", str(caught.exception))
        self.assertEqual(caught.exception.session_id, "ses_root")
        self.assertEqual(len(self._launches()), 1)
        self.assertEqual(self._salvage_rows(), [])

    def test_no_session_without_the_export(self):
        # A test double that is not the real opencode binary collects no metrics, so
        # no session is known and no salvage turn runs.
        with mock.patch.object(actors, "OPENCODE", "/nonexistent/opencode"):
            with self.assertRaises(actors.ActorBudgetExhausted) as caught:
                self._planner(self._backend(json.dumps(HYPOTHESIS))).propose({})
        self.assertIsNone(caught.exception.session_id)
        self.assertEqual(len(self._launches()), 1)
        self.assertEqual([r["reason"] for r in self._salvage_rows()], ["session_unknown"])


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state not in ("Z", "X")


class Helpers(unittest.TestCase):

    def test_continue_session_argv(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        base = backend.argv("p", Path("/lane"), read_only=True)
        self.assertEqual(actors._continue_session_argv(backend, base, "ses_x"),
                         [*base, "--session", "ses_x"])
        self.assertIs(actors._continue_session_argv(backend, base, None), base)
        codex = actors.backend_for("gpt-5.6-sol", "high")
        argv = codex.argv("p", Path("/lane"))
        self.assertIs(actors._continue_session_argv(codex, argv, "ses_x"), argv)

    def test_root_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            def export(name, info):
                path = Path(tmp) / f"{name}.json"
                path.write_text(json.dumps({"info": info, "messages": []}))
                return {"session_id": name, "export": {"path": str(path)}}
            root = export("ses_a", {"id": "ses_a"})
            scout = export("ses_b", {"id": "ses_b", "parentID": "ses_a"})
            other = export("ses_c", {"id": "ses_c"})
            self.assertEqual(actor_metrics.root_session_id(
                {"metrics_error": None, "sessions": [scout, root]}), "ses_a")
            self.assertIsNone(actor_metrics.root_session_id(
                {"metrics_error": None, "sessions": [root, other]}))   # ambiguous
            self.assertIsNone(actor_metrics.root_session_id(
                {"metrics_error": "no new opencode session", "sessions": []}))
            self.assertIsNone(actor_metrics.root_session_id(
                {"metrics_error": None, "sessions": [{"session_id": "x",
                                                      "export": {"path": tmp + "/gone"}}]}))
            self.assertIsNone(actor_metrics.root_session_id(None))

    def test_parse_export_is_unchanged_by_the_refactor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.json"
            msgs = [{"info": {"role": "user", "sessionID": "s"}, "parts": []},
                    {"info": {"role": "assistant", "finish": "stop",
                              "tokens": {"input": 3, "output": 5}}, "parts": []}]
            path.write_text(json.dumps({"messages": msgs}))
            self.assertEqual(actor_metrics.parse_export(path),
                             actor_metrics.parse_export_data({"messages": msgs}))

    def test_cpu_window_skips_salvage_rows_in_the_planner_median(self):
        self.assertEqual(cpu_window.SALVAGE_ARM_SUFFIX, actors.SALVAGE_ARM_SUFFIX)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "actor-calls.jsonl"
            rows = [{"schema": "epyc.autokernel.actor_call.v1", "role": "planner",
                     "wall_s": wall, "seat": {"arm": arm}}
                    for wall, arm in ((1500.0, "plain+budget4500s"),
                                      (4510.0, "plain+budget4500s"),
                                      (120.0, "plain+budget4500s+salvage"))]
            log.write_text("".join(json.dumps(r) + "\n" for r in rows))
            estimator = cpu_window.PhaseEstimator(log, planner_budget_s=4500,
                                                  author_budget_s=3000, critic_timeout_s=7200)
            self.assertEqual(estimator._walls()["planner"], [1500.0, 4510.0])


class CliFlag(unittest.TestCase):

    def test_flag_default_mapping_and_validation(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text(encoding="utf-8")
        block = source[source.index('"--actor-planner-salvage-s"'):][:200]
        self.assertIn("default=900", block)
        self.assertEqual(run._actor_salvage(argparse.Namespace(actor_planner_salvage_s=900)),
                         {"planner_salvage_s": 900})
        self.assertEqual(run._actor_salvage(argparse.Namespace()), {"planner_salvage_s": 0})
        base = dict(actor_context_limit=0, actor_output_limit=0,
                    actor_planner_output_limit=0, actor_author_output_limit=0,
                    actor_planner_budget_s=4500, actor_author_budget_s=0)
        self.assertIsNone(run._actor_budget_error(argparse.Namespace(
            **base, actor_planner_salvage_s=900)))
        self.assertIn(">= 0", run._actor_budget_error(argparse.Namespace(
            **base, actor_planner_salvage_s=-1)))
        self.assertEqual(run._actor_config(argparse.Namespace(
            actor_planner_salvage_s=900))["actor_planner_salvage_s"], 900)

    def test_only_the_proposal_planner_seat_takes_it(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text(encoding="utf-8")
        planner = source[source.index("def make_planner(worker)"):][:1600]
        self.assertIn("**_actor_salvage(args)", planner)
        self.assertEqual(source.count("**_actor_salvage(args)"), 1)


# ------------------------------------------------------------------ the wire: opencode 1.18.31

try:
    from autokernel.loop import test_actor_author_thinking as _wire
except Exception:  # noqa: BLE001 -- the wire test is skipped without its helpers
    _wire = None


def _wire_ready() -> bool:
    return bool(_wire is not None and _wire.OPENCODE and _wire.REAL_GLOBAL.is_file()
                and "http://127.0.0.1:8083/v1" in _wire.REAL_GLOBAL.read_text(errors="replace"))


FIRST = "FIRST-MESSAGE investigate."
TOOL_CALL_ID = "call_salvage_1"


def _base_recorder():
    return _wire._Recorder if _wire is not None else object


class _ToolThenAnswer(_base_recorder()):
    """The recording mock, except that the FIRST turn of the first message answers with
    a `bash` tool call (`sleep 60`) -- a planner mid-investigation -- and every other
    turn answers with a complete hypothesis."""
    bodies: list = []

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw)
        type(self).bodies.append({"method": "POST", "path": self.path, "body": body})
        if not self.path.endswith("/chat/completions"):
            self._send(404, "application/json", b"{}")
            return
        base = {"id": "chatcmpl-mock", "created": int(time.time()), "model": "qwen3.8-27b",
                "object": "chat.completion.chunk"}
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if not body.get("stream"):   # opencode's title request
            self._send(200, "application/json", json.dumps({
                **base, "object": "chat.completion", "usage": usage,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "title"}}]}).encode())
            return
        messages = body.get("messages") or []
        if len(messages) <= 2 and FIRST in json.dumps(messages):
            chunks = [
                {**base, "choices": [{"index": 0, "finish_reason": None, "delta": {
                    "role": "assistant", "tool_calls": [{
                        "index": 0, "id": TOOL_CALL_ID, "type": "function",
                        "function": {"name": "bash", "arguments": ""}}]}}]},
                {**base, "choices": [{"index": 0, "finish_reason": None, "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": json.dumps(
                        {"command": "sleep 60", "description": "wait"})}}]}}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                 "usage": usage},
            ]
        else:
            chunks = [
                {**base, "choices": [{"index": 0, "finish_reason": None, "delta": {
                    "role": "assistant", "content": json.dumps(HYPOTHESIS)}}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": usage},
            ]
        payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(200, "text/event-stream", payload.encode())


@unittest.skipUnless(_wire_ready(), "needs the installed opencode and the host's qwen-gpu "
                                    "global config")
class OpencodeSessionContinuation(unittest.TestCase):
    """The installed opencode, a scratch HOME/XDG, and a recording mock in place of the
    model server (the helpers of `test_actor_author_thinking`): a first `opencode run`
    creates a session; the salvage argv (`_continue_session_argv`) with a new message on
    stdin must send a request that carries the FIRST exchange -- the session continued --
    and must not create a new session."""

    def _open(self, handler):
        """Scratch HOME/lane/mock; returns (lane, env, backend, planner argv)."""
        from http.server import ThreadingHTTPServer
        base = _wire.SCRATCH_BASE if _wire.SCRATCH_BASE.is_dir() else None
        tmp = tempfile.TemporaryDirectory(prefix="ak-salvage-mock-", dir=base)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        handler.bodies = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        self.assertNotEqual(port, 8083)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        home = root / "home"
        (home / ".config" / "opencode").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.jsonc").write_text(
            _wire._jsonc_text_rebased(_wire.REAL_GLOBAL.read_text(encoding="utf-8"),
                                      f"http://127.0.0.1:{port}/v1"), encoding="utf-8")
        lane = root / "work" / "lane"
        lane.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(lane)], check=True, timeout=30)
        seat = actors.ActorSeat(bounded=False, planner_budget_s=4500, planner_salvage_s=900)
        backend = actors.Backend("opencode", "qwen-gpu/qwen3.8-27b", "high", _wire.OPENCODE)
        extra = actors._seat_call(seat, backend, "planner", lane, {})
        self.assertNotIn("8083", Path(extra["OPENCODE_CONFIG"]).read_text())
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("OPENCODE_") and not k.startswith("XDG_")}
        env.update({"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
                    "XDG_DATA_HOME": str(home / ".local" / "share"),
                    "XDG_STATE_HOME": str(home / ".local" / "state"),
                    "XDG_CACHE_HOME": str(home / ".cache"),
                    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                    **extra})
        return lane, env, backend, backend.argv("", lane, read_only=False)

    def _spawn(self, argv, env, lane, message):
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, cwd=str(lane),
                                start_new_session=True, text=True)
        self.addCleanup(self._reap, proc)
        return proc

    @staticmethod
    def _reap(proc):
        try:   # nothing of our own session may outlive the test (bash, MCP children)
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _run(self, argv, env, lane, message, timeout_s=120):
        proc = self._spawn(argv, env, lane, message)
        try:
            out, err = proc.communicate(message, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
        return proc.returncode, out, err

    def _sessions(self, env, lane) -> list[str]:
        listed = subprocess.run(
            [_wire.OPENCODE, "session", "list", "--format", "json", "-n", "5"],
            cwd=str(lane), env=env, capture_output=True, text=True, timeout=60)
        return [row["id"] for row in json.loads(listed.stdout or "[]")]

    def _continue(self, handler, env, lane, backend, argv, sid):
        first_posts = len(_wire._chat_bodies(handler.bodies))
        salvage_argv = actors._continue_session_argv(backend, argv, sid)
        self.assertEqual(salvage_argv, [*argv, "--session", sid])
        rc, out, err = self._run(salvage_argv, env, lane, actors.PLANNER_SALVAGE_MESSAGE)
        self.assertEqual(rc, 0, err[-1500:])
        bodies = _wire._chat_bodies(handler.bodies)[first_posts:]
        self.assertTrue(bodies, "the continuation sent no chat completion")
        messages = bodies[-1].get("messages") or []
        self.assertIn(FIRST, json.dumps(messages))        # the session's history came along
        self.assertIn("Stop investigating now", json.dumps(messages[-1]))  # stdin message
        self.assertEqual(self._sessions(env, lane), [sid],
                         "the continuation created a new session")
        return out, messages

    def test_session_argv_continues_a_finished_session(self):
        handler = _wire._Recorder
        lane, env, backend, argv = self._open(handler)
        rc, _out, err = self._run(argv, env, lane, FIRST)
        self.assertEqual(rc, 0, err[-1500:])
        sessions = self._sessions(env, lane)
        self.assertEqual(len(sessions), 1)
        self._continue(handler, env, lane, backend, argv, sessions[0])

    def test_session_argv_continues_a_session_ended_by_the_budget_path(self):
        """The realistic case: the planner's process group was TERM'd (then KILL'd) by
        `_end_group` in the middle of a tool call, exactly as `_run_stoppable` ends a
        call at its budget. The continuation still works: opencode 1.18.31 closes the
        dangling tool call with a tool result before the new message."""
        handler = _ToolThenAnswer
        lane, env, backend, argv = self._open(handler)
        proc = self._spawn(argv, env, lane, FIRST)
        proc.stdin.write(FIRST)
        proc.stdin.close()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not _wire._chat_bodies(handler.bodies):
            time.sleep(0.2)
        self.assertTrue(_wire._chat_bodies(handler.bodies), "opencode never called the mock")
        time.sleep(4)   # the bash tool (sleep 60) is now running
        self.assertIsNone(proc.poll(), "opencode ended before the budget path could")
        actors._end_group(proc, grace_s=15)
        sessions = self._sessions(env, lane)
        self.assertEqual(len(sessions), 1)
        out, messages = self._continue(handler, env, lane, backend, argv, sessions[0])
        tool_results = [m for m in messages if m.get("role") == "tool"
                        and m.get("tool_call_id") == TOOL_CALL_ID]
        self.assertEqual(len(tool_results), 1, json.dumps(messages)[-1500:])
        self.assertIn('"akm-salvaged"', out)   # the reply reaches stdout for the parser


if __name__ == "__main__":
    unittest.main()
