"""Author report recovery, per-role output limits and `output_capped_empty`.

DS41 run 10b (2026-09-25 19:58Z): the planner formed a hypothesis, the critic accepted
it, and the AUTHOR call ran 608 s and ended on one step that hit the 8,192-token output
cap. opencode ends the session on finish=length, so the reply was empty and the loop
recorded `planner_transient: actor produced no final report (0 chars)`. ~51 min of
planner+critic+author produced nothing.

Pinned here:
- an empty author reply over a lane the loop reset for this draw, holding a diff, is
  answered from `git diff --name-only <base>` (+ untracked files inside the kernel
  allowlist), recorded `report_source: "lane_diff"`, and the normal gates judge it;
- no diff (or no change by this call), no reset lane, a moved HEAD, a stray untracked
  file or a diff outside the target surface keeps today's refusal;
- the planner and the critic are never derived;
- per-role `limit.output` in the generated configs, and output < context/2;
- `failure_class: output_capped_empty` on the metrics row (and the refusal reason);
- the planner_transient row carries the author resume checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from autokernel.loop import (actor_metrics, actor_opencode_config as aoc, actors, bench,
                             gates, integrity, pipeline, resume)
from autokernel.loop import loop as loop_mod

MODEL = "qwen-gpu/qwen3.8-27b"
TARGET = "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"


def _git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def _hypothesis() -> loop_mod.Hypothesis:
    return loop_mod.Hypothesis(
        mechanism_id="akm-q4k-x4t-avx512", statement="512-bit inner loop",
        falsifier="no clear of the decode floor",
        target_surface=f"{TARGET}: mul_mat_qX_K_q8_2_X4_T body (lines 795-876)",
        target_symbol="mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>")


def _comparison(effect=0.05):
    return bench.Comparison(
        surface="tg128", anchor_samples=[100.0], candidate_samples=[100.0 * (1 + effect)],
        effect=effect, estimator="median_over_median", pairs=5,
        noise_floor_pct=1.0, residency={"invocations": 10, "resident": 10})


class _Critic:
    def __init__(self):
        self.patch_reviews = []

    def review_hypothesis(self, hypothesis, context):
        return loop_mod.Review(True)

    def review_patch(self, hypothesis, paths, context):
        self.patch_reviews.append(tuple(paths))
        return loop_mod.Review(True)


class _Lane(unittest.TestCase):
    """A real git lane: one committed kernel file, HEAD detached at `base`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.ws = root / "workers" / "lane0"
        (self.ws / Path(TARGET).parent).mkdir(parents=True)
        (self.ws / TARGET).write_text("int kernel() { return 0; }\n")
        (self.ws / "README").write_text("tree\n")
        _git(self.ws, "init", "-q")
        _git(self.ws, "add", "-A")
        _git(self.ws, "commit", "-q", "-m", "anchor")
        self.base = _git(self.ws, "rev-parse", "HEAD")
        _git(self.ws, "checkout", "-q", "--detach", self.base)

    def tearDown(self):
        self._tmp.cleanup()

    def edit(self, text="int kernel() { return 1; }\n", path=TARGET):
        target = self.ws / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def planner(self, reply: str = "", *, edit=None, capped=True) -> actors.AgentPlanner:
        """An AgentPlanner whose author call (optionally) edits the lane, then replies
        `reply` -- the run 10b shape when empty and `capped`."""
        planner = actors.AgentPlanner(workspace=self.ws)

        def call(*_args, **_kw):
            if edit is not None:
                edit()
            if capped and not reply.strip():
                return actors._Reply(reply, failure_class=actor_metrics.OUTPUT_CAPPED_EMPTY)
            return reply
        self._run_agent = mock.patch.object(actors, "_run_agent", side_effect=call)
        self._run_agent.start()
        self.addCleanup(self._run_agent.stop)
        return planner

    def iterate(self, planner, *, author_lane="default", critic=None, gate_calls=None,
                patch_rounds=loop_mod.PATCH_ROUNDS):
        gate_calls = gate_calls if gate_calls is not None else []

        def gate(hypothesis, paths):
            gate_calls.append(tuple(paths))
            return True, [gates.Verdict("compile", True)]

        hypothesis = _hypothesis()
        proposer = mock.Mock(wraps=planner)
        proposer.propose = mock.Mock(return_value=hypothesis)
        proposer.author = planner.author
        return loop_mod.iterate(
            planner=proposer, critic=critic or _Critic(), context={},
            measure=lambda h, p: _comparison(), gate=gate,
            commit=lambda h, p, c: "abc1234", hypothesis_rounds=1,
            patch_rounds=patch_rounds,
            validate_candidate=lambda h, p: integrity.validate_candidate(self.ws, p).to_dict(),
            author_lane=((self.ws, self.base) if author_lane == "default" else author_lane))


class EmptyAuthorReplyOverALaneDiff(_Lane):

    def test_the_report_is_derived_from_the_diff_and_the_gates_proceed(self):
        critic = _Critic()
        gate_calls = []
        outcome = self.iterate(self.planner("", edit=self.edit), critic=critic,
                               gate_calls=gate_calls)
        self.assertEqual(gate_calls, [(TARGET,)])            # the build gate saw it
        self.assertEqual(critic.patch_reviews, [(TARGET,)])  # critic pass 2 judged it
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(outcome.report_source, "lane_diff")
        row = outcome.to_attempt()
        self.assertEqual(row["report_source"], "lane_diff")
        recovery = row["author_report_recovery"]
        self.assertEqual(recovery["paths"], [TARGET])
        self.assertEqual(recovery["base"], self.base)
        self.assertEqual(recovery["failure_class"], actor_metrics.OUTPUT_CAPPED_EMPTY)
        self.assertIn("no final report", recovery["reply_refusal"])
        # ...and beside the author call's metrics row, under its own schema.
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        [source] = [json.loads(line) for line in log.read_text().splitlines()
                    if json.loads(line).get("schema") == actor_metrics.REPORT_SOURCE_SCHEMA]
        self.assertEqual((source["report_source"], source["role"], source["paths"]),
                         ("lane_diff", "author", [TARGET]))
        self.assertEqual(source["failure_class"], actor_metrics.OUTPUT_CAPPED_EMPTY)

    def test_an_untracked_file_inside_the_allowlist_is_part_of_the_report(self):
        extra = "ggml/src/ggml-cpu/iqk/iqk_q4k_x4t.h"

        def edit():
            self.edit()
            self.edit("#pragma once\n", path=extra)
        gate_calls = []
        self.iterate(self.planner("", edit=edit), gate_calls=gate_calls)
        self.assertEqual(gate_calls, [tuple(sorted((TARGET, extra)))])

    def test_an_unparsable_reply_is_recovered_too(self):
        with mock.patch.object(actors, "_schema_repair", return_value=None):
            outcome = self.iterate(self.planner("I edited the kernel. Done.", edit=self.edit,
                                                capped=False))
        self.assertEqual(outcome.report_source, "lane_diff")
        self.assertIsNone(outcome.author_report_recovery["failure_class"])

    def test_the_real_pool_passes_the_lane_it_reset(self):
        worker = pipeline.Worker("lane0", self.ws, self.ws.parent / "build")
        planner = self.planner("", edit=self.edit)
        hypothesis = _hypothesis()
        planner.propose = mock.Mock(return_value=hypothesis)
        gate_calls = []

        def gate(h, p):
            gate_calls.append(tuple(p))
            return True, [gates.Verdict("compile", True)]
        outcomes = pipeline.run_pool(
            workers=[worker], make_planner=lambda w: planner, make_critic=lambda w: _Critic(),
            build_context=dict, make_gate=lambda w: gate,
            make_measure=lambda w: (lambda h, p: _comparison()),
            commit=lambda w, h, p, c: "abc1234", champion_head=lambda: self.base,
            reset_to_champion=lambda w: self.base, record=lambda o: None, iterations=1)
        self.assertEqual([o.status for o in outcomes], ["kept"])
        self.assertEqual(outcomes[0].report_source, "lane_diff")
        self.assertEqual(gate_calls, [(TARGET,)])


class TodaysRefusalStands(_Lane):

    def _assert_transient(self, outcome, *needles):
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIsNone(outcome.report_source)
        for needle in needles:
            self.assertIn(needle, outcome.reasons[0])
        return outcome

    def test_no_diff_keeps_the_refusal_and_the_author_checkpoint(self):
        outcome = self._assert_transient(self.iterate(self.planner("")),
                                         "output_capped_empty",
                                         "refusing to repair an empty reply")
        self.assertNotIn("lane diff", outcome.reasons[0])
        [checkpoint] = outcome.resume_checkpoints
        self.assertEqual(checkpoint["stage"], "author")
        self.assertEqual(checkpoint["schema"], loop_mod.CHECKPOINT_SCHEMA)
        self.assertEqual(checkpoint["hypothesis"]["mechanism_id"], "akm-q4k-x4t-avx512")
        self.assertEqual(checkpoint["patch_rounds_remaining"], loop_mod.PATCH_ROUNDS)
        self.assertTrue(checkpoint["critic_hypothesis"]["accepted"])

    def test_a_lane_the_loop_did_not_reset_is_never_salvaged(self):
        self._assert_transient(self.iterate(self.planner("", edit=self.edit),
                                            author_lane=None),
                               "no final report")

    def test_a_moved_head_is_refused(self):
        def commit_it():
            self.edit()
            _git(self.ws, "commit", "-qam", "author committed")
        self._assert_transient(self.iterate(self.planner("", edit=commit_it)),
                               "lane diff not usable", "is not the reset base")

    def test_a_stray_untracked_file_is_refused(self):
        def edit():
            self.edit()
            self.edit("notes\n", path="notes.txt")
        self._assert_transient(self.iterate(self.planner("", edit=edit)),
                               "untracked files outside")

    def test_a_diff_that_misses_the_target_surface_is_refused(self):
        self._assert_transient(
            self.iterate(self.planner("", edit=lambda: self.edit(
                "x\n", path="ggml/src/ggml-cpu/other.cpp"))),
            "is the hypothesis's target surface")

    def test_a_later_round_that_changed_nothing_does_not_resubmit_the_rejected_edit(self):
        critic = _Critic()
        critic.review_patch = mock.Mock(side_effect=[loop_mod.Review(False, "too broad")])
        replies = iter(['{"paths": ["%s"]}' % TARGET, ""])
        planner = actors.AgentPlanner(workspace=self.ws)

        def call(*_a, **_kw):
            reply = next(replies)
            if reply:
                self.edit()
            return reply
        with mock.patch.object(actors, "_run_agent", side_effect=call):
            outcome = self.iterate(planner, critic=critic)
        self._assert_transient(outcome, "no final report")
        self.assertEqual(outcome.resume_checkpoints[0]["prior_patch_rejections"],
                         ["too broad"])


class OnlyTheAuthorIsDerived(_Lane):

    def test_an_empty_planner_reply_is_a_plain_transient(self):
        planner = self.planner("", edit=self.edit)
        with self.assertRaises(actors.ProviderTransient) as caught:
            planner.propose({})
        self.assertNotIsInstance(caught.exception, loop_mod.AuthorReportMissing)
        self.assertIn("output_capped_empty", str(caught.exception))

    def test_iterate_never_derives_a_hypothesis_from_a_dirty_lane(self):
        self.edit()
        planner = self.planner("")
        outcome = loop_mod.iterate(
            planner=planner, critic=_Critic(), context={},
            measure=mock.Mock(side_effect=AssertionError("no measurement")),
            gate=mock.Mock(side_effect=AssertionError("no gate")),
            commit=mock.Mock(side_effect=AssertionError("no commit")),
            author_lane=(self.ws, self.base))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIsNone(outcome.report_source)

    def test_an_empty_critic_reply_is_a_plain_transient(self):
        self.planner("")
        critic = actors.AgentCritic(workspace=self.ws)
        with self.assertRaises(actors.ProviderTransient) as caught:
            critic.review_hypothesis(_hypothesis(), {})
        self.assertNotIsInstance(caught.exception, loop_mod.AuthorReportMissing)

    def test_the_author_raises_the_recoverable_type(self):
        with self.assertRaises(actors.AuthorReplyMissing) as caught:
            self.planner("").author(_hypothesis(), {})
        self.assertIsInstance(caught.exception, loop_mod.AuthorReportMissing)
        self.assertIsInstance(caught.exception, actors.ProviderTransient)
        self.assertEqual(caught.exception.failure_class, actor_metrics.OUTPUT_CAPPED_EMPTY)

    def test_an_abstention_is_never_overridden_by_the_lane(self):
        self.edit()
        got = self.planner('{"abstain": "infeasible"}').author(_hypothesis(), {})
        self.assertIsInstance(got, actors.Abstain)


class PerRoleOutputLimits(unittest.TestCase):

    SEAT = dict(context_limit=aoc.DEFAULT_CONTEXT_LIMIT, output_limit=aoc.DEFAULT_OUTPUT_LIMIT,
                planner_output_limit=aoc.DEFAULT_PLANNER_OUTPUT_LIMIT,
                author_output_limit=aoc.DEFAULT_AUTHOR_OUTPUT_LIMIT, concise=True)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.backend = actors.backend_for(MODEL, "high")

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _limit(env) -> dict:
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        return config["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]["limit"]

    def test_defaults(self):
        self.assertEqual((aoc.DEFAULT_PLANNER_OUTPUT_LIMIT, aoc.DEFAULT_AUTHOR_OUTPUT_LIMIT,
                          aoc.DEFAULT_CONTEXT_LIMIT), (16384, 32768, 131072))

    def test_each_plain_role_config_carries_its_own_output_limit(self):
        seat = actors.ActorSeat(bounded=False, **self.SEAT)
        expected = {"planner": 16384, "critic": 16384, "author": 32768}
        for role, output in expected.items():
            env = actors._seat_call(seat, self.backend, role, self.ws, {})
            self.assertEqual(self._limit(env), {"context": 131072, "output": output}, role)
            self.assertIn(f"+out{output // 1024}k", env[actors.SEAT_ENV_ARM], role)
            self.assertEqual(json.loads(env[actors.SEAT_ENV_BUDGETS])["output_limit"], output)

    def test_the_bounded_author_config_carries_the_author_limit(self):
        planner = actors.AgentPlanner(workspace=self.ws, backend=self.backend,
                                      seat=actors.ActorSeat(bounded=True, **self.SEAT))
        _backend, env = planner._seated("author", {})
        self.assertEqual(self._limit(env), {"context": 131072, "output": 32768})
        _backend, env = planner._seated("planner", {})
        self.assertEqual(self._limit(env), {"context": 131072, "output": 16384})

    def test_a_zero_role_limit_falls_back_to_the_shared_one(self):
        seat = actors.ActorSeat(bounded=False, context_limit=131072, output_limit=8192)
        for role in ("planner", "critic", "author"):
            self.assertEqual(seat.output_limit_for(role), 8192, role)

    def _args(self, **kw):
        base = dict(actor_context_limit=aoc.DEFAULT_CONTEXT_LIMIT,
                    actor_output_limit=aoc.DEFAULT_OUTPUT_LIMIT,
                    actor_planner_output_limit=aoc.DEFAULT_PLANNER_OUTPUT_LIMIT,
                    actor_author_output_limit=aoc.DEFAULT_AUTHOR_OUTPUT_LIMIT,
                    actor_concise="on", actor_planner_budget_s=2700,
                    actor_author_budget_s=0, actor_timeout_s=7200)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_run_maps_and_validates_the_per_role_flags(self):
        from autokernel.loop import run
        args = self._args()
        self.assertIsNone(run._actor_budget_error(args))
        self.assertEqual(run._effective_output_limits(args),
                         {"planner": 16384, "critic": 16384, "author": 32768})
        seat = actors.ActorSeat(bounded=False, **run._actor_limits(args))
        self.assertEqual(seat.output_limit_for("author"), 32768)
        self.assertEqual(seat.output_limit_for("critic"), 16384)
        for flag in ("actor_author_output_limit", "actor_planner_output_limit",
                     "actor_output_limit"):
            error = run._actor_budget_error(self._args(**{flag: 65536}))
            self.assertIn("below half", error, flag)
        self.assertIn(">= 0", run._actor_budget_error(self._args(actor_author_output_limit=-1)))
        self.assertEqual(run._effective_output_limits(self._args(
            actor_planner_output_limit=0, actor_author_output_limit=0)),
            {"planner": 8192, "critic": 8192, "author": 8192})

    def test_the_parser_defaults(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text(encoding="utf-8")
        for flag, default in (("--actor-planner-output-limit", "DEFAULT_PLANNER_OUTPUT_LIMIT"),
                              ("--actor-author-output-limit", "DEFAULT_AUTHOR_OUTPUT_LIMIT"),
                              ("--actor-output-limit", "DEFAULT_OUTPUT_LIMIT"),
                              ("--actor-context-limit", "DEFAULT_CONTEXT_LIMIT")):
            block = source[source.index(f'"{flag}"'):][:300]
            self.assertIn(default, block, flag)


class _ScriptBackend(actors.Backend):
    reply: str = ""

    def argv(self, prompt, workspace, *, read_only=False):
        return [sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", self.reply]


class OutputCappedEmpty(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _export(self, finishes) -> Path:
        path = Path(self._tmp.name) / "export.json"
        msgs = [{"info": {"role": "user", "sessionID": "s"}, "parts": []}]
        for finish in finishes:
            msgs.append({"info": {"role": "assistant", "finish": finish,
                                  "tokens": {"input": 1, "output": 1}}, "parts": []})
        path.write_text(json.dumps({"messages": msgs}))
        return path

    def test_the_export_says_whether_the_final_step_was_capped(self):
        self.assertTrue(actor_metrics.parse_export(
            self._export(["tool-calls", "length"]))["final_step_capped"])
        self.assertFalse(actor_metrics.parse_export(
            self._export(["length", "stop"]))["final_step_capped"])

    def _call(self, reply: str, *, final_step_capped: bool, role_schema=actors.PATHS_SCHEMA):
        backend = _ScriptBackend(kind="opencode", model="test/m", effort="high",
                                 binary=actors.OPENCODE)
        object.__setattr__(backend, "reply", reply)
        stats = {"metrics_error": None, "totals": {"output_capped_steps": 1},
                 "sessions": [], "session_ids": ["s"], "context_max_tokens": 1,
                 "final_step_capped": final_step_capped}
        with mock.patch.object(actor_metrics, "list_session_ids", return_value=set()), \
                mock.patch.object(actor_metrics, "collect", return_value=stats), \
                mock.patch.object(actors, "_record_call"):
            text = actors._run_agent("p", workspace=self.ws, backend=backend,
                                     schema=role_schema)
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        return text, rows[-1]

    def test_a_capped_final_step_with_empty_text_is_classified(self):
        text, row = self._call("", final_step_capped=True)
        self.assertEqual(row["failure_class"], actor_metrics.OUTPUT_CAPPED_EMPTY)
        self.assertEqual(text, "")
        self.assertEqual(text.failure_class, actor_metrics.OUTPUT_CAPPED_EMPTY)
        with self.assertRaises(actors.ProviderTransient) as caught:
            actors._parse_reply(text, schema=actors.PATHS_SCHEMA,
                                backend=actors.CRITIC_DEFAULT, workspace=self.ws)
        self.assertTrue(str(caught.exception).startswith("output_capped_empty: "))

    def test_every_opencode_role_is_classified(self):
        for schema in (actors.HYPOTHESIS_SCHEMA, actors.REVIEW_SCHEMA):
            _text, row = self._call("", final_step_capped=True, role_schema=schema)
            self.assertEqual(row["failure_class"], actor_metrics.OUTPUT_CAPPED_EMPTY)

    def test_a_reply_or_an_uncapped_final_step_is_not(self):
        _text, row = self._call('{"paths": ["a.cpp"]}', final_step_capped=True)
        self.assertIsNone(row["failure_class"])
        text, row = self._call("", final_step_capped=False)
        self.assertIsNone(row["failure_class"])
        self.assertIsNone(getattr(text, "failure_class", None))

    def test_the_summary_counts_it(self):
        rows = [{"failure_class": actor_metrics.OUTPUT_CAPPED_EMPTY}, {"failure_class": None}]
        self.assertEqual(actor_metrics._summary(rows)["output_capped_empty"], 1)


class TheLostHypothesisResumesAtAuthoring(_Lane):
    """The run 10b shape end to end: the planner_transient row's author checkpoint,
    bound by the owner, is what `resume.scan` / `prepare` would queue."""

    def test_the_bound_checkpoint_is_eligible_at_author(self):
        outcome = self.iterate(self.planner(""))
        outcome.spawn_parent = self.base
        row = resume.bind_checkpoints(outcome.to_attempt(), epoch="e" * 64,
                                      anchor_commit=self.base,
                                      target={"measurement_surface": "s", "model": "m"})
        [checkpoint] = row["resume_checkpoints"]
        self.assertEqual((checkpoint["stage"], checkpoint["anchor_commit"]),
                         ("author", self.base))
        candidate = resume.Candidate("row#0", "row", "t", row["status"],
                                     "akm-q4k-x4t-avx512", checkpoint)
        self.assertIsNone(resume.ineligible_reason(candidate, rules_fingerprint="x"))


if __name__ == "__main__":
    unittest.main()
