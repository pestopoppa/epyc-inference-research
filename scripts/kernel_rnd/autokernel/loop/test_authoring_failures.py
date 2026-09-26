"""An accepted hypothesis is not discarded when only its AUTHORING failed.

Origin: DS41 run 10h batch-000000 (2026-09-26). The reinstated `akm-q4k-x4t-avx512`
(de5eeef0...#0) went to a best-of-2 panel: a0-off (thinking off) wrote a zmm inner loop,
ran ak-check (compile PASS, op-test FAIL 79/130) and ABSTAINED; a1-medium (thinking
medium, 16,384 output cap) spent 71 minutes, hit the cap on 3 steps -- the last one its
edit -- and ended `report_missing`. The panel returned `Abstain`, the iteration ended
`abstained`, the resumed claim was consumed and the hypothesis was gone.

Batch-000001: the planner's 31-minute hypothesis `akm-verify-batch-solo-2rows` was lost
when critic pass 1 returned an empty reply -- opencode auto-rejected the read-only
critic's `external_directory` read of `<store>/experiments.md`, which ended its session.

Pinned run:
    taskset -c 72-79 timeout 900 python3 -m pytest -q -p no:cacheprovider \\
        autokernel/loop/test_authoring_failures.py
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from autokernel.controller import experiments
from autokernel.loop import (actor_metrics, actor_opencode_config as aoc, actors, ak_check,
                             bestof, loop, process_metrics, resume, serial_scheduling)
from autokernel.loop import test_bestof as tb
from autokernel.loop import test_pending_hypotheses as tp
from autokernel.loop import test_resume as base

EPOCH, TARGET = base.EPOCH, base.TARGET
A0_REASON = ("zmm 512-bit inner loop compiles (ak-check PASS) but op-test FAILS 79/130 "
             "(ERR=1.0) on q4_K and q5_K MUL_MAT/MUL_MAT_ID; abstaining")
A1_REASON = ("authoring reported ['ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp'] but the "
             "worktree is unchanged there")
AK_CHECK = {"calls": 7, "pass": 1, "fail": 5, "last_status": "nothing",
            "by_mode": {"compile": {"calls": 5, "pass": 1, "fail": 4, "seconds": 8.3},
                        "op-test": {"calls": 2, "pass": 0, "fail": 1, "seconds": 5.0}},
            "last_op_test": {"status": "fail", "passed": 51, "total": 130,
                             "types": ["q4_K", "q5_K"], "ops": ["MUL_MAT", "MUL_MAT_ID"]}}


def metrics_row(*, thinking, capped=False, capped_steps=0, output_limit=16384,
                ak_check=None, role="author", failure_class=None, rejected=0, finish=None):
    """One actor-call metrics row as `actors._record_metrics` writes it."""
    return {"schema": actor_metrics.METRICS_SCHEMA, "role": role, "wall_s": 2.0,
            "failure_class": failure_class, "timed_out": False, "ak_check": ak_check,
            "budgets": {"thinking": thinking, "output_limit": output_limit,
                        "context_limit": 90112, "output_capped_steps": capped_steps},
            "opencode": {"final_step_capped": capped, "permission_rejected": rejected,
                         "final_finish": finish,
                         "totals": {"steps": 14, "decoded_tokens": 76154, "compactions": 2,
                                    "output_capped_steps": capped_steps}}}


def write_rows(log: Path, *rows) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ 1. classification


class ClassifyAuthorFailures(unittest.TestCase):

    def test_the_table(self):
        cases = [
            (("abstained", {}), "authoring"),
            (("abstained", {"final_step_capped": True}), "authoring"),   # an explicit reply
            (("report_missing", {}), "authoring"),
            (("report_missing", {"final_step_capped": True}), "harness"),
            (("report_missing", {"failure_class": "output_capped_empty"}), "harness"),
            (("report_missing", {"failure_class": "permission_rejected"}), "harness"),
            (("report_missing", {"timed_out": True}), "harness"),
            (("transient", {}), "harness"),
            (("error", {}), "harness"),
            (("stopped", {}), "harness"),
        ]
        for (outcome, evidence), expected in cases:
            with self.subTest(outcome=outcome, evidence=evidence):
                self.assertEqual(loop.classify_author_failure(outcome, **evidence)["class"],
                                 expected)

    def test_a_capped_final_step_with_no_diff_is_output_capped_empty(self):
        record = loop.author_failure_record(
            label="a1-medium", thinking="medium", outcome="report_missing", reason=A1_REASON,
            evidence=actor_metrics.failure_evidence(
                metrics_row(thinking="medium", capped=True, capped_steps=3)))
        self.assertEqual((record["class"], record["failure_class"]),
                         ("harness", "output_capped_empty"))
        text = loop.author_failure_feedback(record)
        self.assertIn("not charged", text)
        self.assertIn("3 step(s) hit the 16384-token output cap", text)

    def test_feedback_carries_the_ak_check_result_and_the_retained_patch(self):
        record = loop.author_failure_record(
            label="a0-off", thinking="off", outcome="abstained", reason=A0_REASON,
            evidence={"ak_check": AK_CHECK},
            patch={"patch_file": "/store/patches/a0.patch", "patch_sha256": "9c" * 32})
        text = loop.author_failure_feedback(record)
        self.assertIn("authoring failed [a0-off, thinking off] abstained", text)
        self.assertIn("ak-check: compile 1/5 pass, op-test 0/2 pass", text)
        self.assertIn("last op-test FAIL 51/130 on q4_K,q5_K", text)
        self.assertIn("/store/patches/a0.patch", text)

    def test_ak_check_usage_summary_reports_the_last_op_test(self):
        log = Path(self.id().replace(".", "_"))
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            write_rows(log, {"call_id": "c", "mode": "compile", "status": "pass"},
                       {"call_id": "c", "mode": "op-test", "status": "fail",
                        "oracle": {"passed": 51, "total": 130, "types": ["q4_K"],
                                   "ops": ["MUL_MAT"]}})
            summary = ak_check.usage_summary(log, "c")
        self.assertEqual(summary["last_op_test"]["passed"], 51)
        self.assertEqual(summary["last_op_test"]["status"], "fail")


# ------------------------------------------------------------------ 2. single path, resumed


class Abstainer(base.Planner):
    """Authors nothing: the author abstains (or raises)."""

    def __init__(self, case, repo, *, reply=None, raises=None):
        super().__init__(repo, [])
        self.case, self.reply, self.raises = case, reply, raises

    def propose(self, context):
        self.case.fail("a pending accepted hypothesis resumes before the planner is asked")

    def author(self, hypothesis, context):
        self.authored.append((hypothesis, list(context.get("prior_patch_rejections", ()))))
        if self.raises is not None:
            raise self.raises
        return self.reply


class ResumedAuthorFailures(tp.Fixture):
    """A resumed author checkpoint whose authoring fails stays pending."""

    def pending_point(self, author_attempts=3):
        self.attempt([tp.rejected(tp.AUTHORING_1), tp.rejected(tp.AUTHORING_2)],
                     author_attempts=author_attempts)
        point, _ = self.take()
        self.assertEqual(point.stage, "author")
        return point

    def run_point(self, point, planner, *, author_attempts=3, author_lane=None):
        outcome = loop.iterate(
            planner=planner, critic=base.Critic(), context={},
            measure=lambda h, p: self.fail("never measured"),
            gate=lambda h, p: self.fail("never gated"), commit=lambda h, p, c: None,
            record_abandoned=self.owner.record_abandoned, resume=point,
            author_attempts=author_attempts, author_lane=author_lane)
        self.owner.record(outcome)
        base.reset(self.repo)
        return outcome

    def test_an_author_abstention_stays_pending_with_feedback_and_one_attempt_charged(self):
        point = self.pending_point()
        outcome = self.run_point(point, Abstainer(self, self.repo,
                                                  reply=loop.Abstain(A0_REASON)))
        self.assertEqual(outcome.status, loop.AUTHORING_FAILED)
        self.assertEqual(outcome.resumed_from, point.checkpoint_id)
        # The claim is consumed by a PENDING outcome: its row carries the next checkpoint.
        self.assertEqual(self.owner.dispositions[-1], "settled")
        self.assertEqual(self.claim(point.checkpoint_id)["result_status"], loop.AUTHORING_FAILED)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["stage"], checkpoint["author_attempts_used"],
                          checkpoint["author_harness_failures"]), ("author", 2, 0))
        self.assertEqual(checkpoint["prior_patch_rejections"][:2],
                         [tp.AUTHORING_1, tp.AUTHORING_2])
        self.assertIn(A0_REASON, checkpoint["prior_patch_rejections"][-1])
        self.assertEqual(checkpoint["authoring_failures"][0]["class"], "authoring")
        self.assertEqual(outcome.hypothesis_pending["author_attempts_remaining"], 1)
        # The next draw re-authors it from that evidence, before the planner.
        successor, report = self.take()
        self.assertEqual([(q["stage"], q["row_status"]) for q in report["queued"]],
                         [("author", loop.AUTHORING_FAILED)])
        self.assertIn(A0_REASON, successor.prior_patch_rejections[-1])
        (view,) = resume.pending_hypotheses(self.store, epoch=EPOCH, anchor_commit=self.anchor)
        self.assertEqual((view["status"], view["author_attempts_used"]),
                         (loop.AUTHORING_FAILED, 2))

    def test_a_truncated_report_is_a_harness_failure_and_not_charged(self):
        point = self.pending_point()
        lane_log = self.repo.parent / actor_metrics.REPLY_DIR_NAME / actor_metrics.CALL_LOG_NAME

        class Truncated(Abstainer):
            def author(inner, hypothesis, context):
                write_rows(lane_log, metrics_row(thinking="medium", capped=True, capped_steps=3))
                raise actors.AuthorReplyMissing(A1_REASON)

        outcome = self.run_point(point, Truncated(self, self.repo),
                                 author_lane=(self.repo, self.anchor))
        self.assertEqual(outcome.status, loop.AUTHORING_HARNESS_FAILURE)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["author_attempts_used"],
                          checkpoint["author_harness_failures"]), (1, 1))
        (failure,) = checkpoint["authoring_failures"]
        self.assertEqual((failure["outcome"], failure["class"], failure["failure_class"]),
                         ("report_missing", "harness", "output_capped_empty"))
        self.assertIn("not charged", checkpoint["prior_patch_rejections"][-1])

    def test_consecutive_harness_failures_are_charged_at_the_cap(self):
        point = self.pending_point()
        transient = actors.ProviderTransient("provider 503")
        for streak in range(1, loop.AUTHOR_HARNESS_FAILURE_CAP):
            outcome = self.run_point(point, Abstainer(self, self.repo, raises=transient))
            self.assertEqual(outcome.status, loop.AUTHORING_HARNESS_FAILURE)
            self.assertEqual(outcome.resume_checkpoints[0]["author_harness_failures"], streak)
            self.assertEqual(outcome.resume_checkpoints[0]["author_attempts_used"], 1)
            point, _ = self.take()
        outcome = self.run_point(point, Abstainer(self, self.repo, raises=transient))
        self.assertEqual(outcome.status, loop.AUTHORING_FAILED)
        self.assertTrue(outcome.hypothesis_pending["charged"])
        self.assertEqual((outcome.resume_checkpoints[0]["author_attempts_used"],
                          outcome.resume_checkpoints[0]["author_harness_failures"]), (2, 0))
        self.assertIn("consecutive authoring harness failures",
                      " ".join(outcome.resume_checkpoints[0]["prior_patch_rejections"]))

    def test_the_spent_budget_retires_the_hypothesis(self):
        point = self.pending_point(author_attempts=2)
        outcome = self.run_point(point, Abstainer(self, self.repo,
                                                  reply=loop.Abstain("cannot")),
                                 author_attempts=2)
        self.assertEqual(outcome.status, loop.HYPOTHESIS_RETIRED)
        self.assertEqual(outcome.resume_checkpoints, [])
        self.assertIn("2/2", outcome.reasons[0])
        self.assertEqual(outcome.to_attempt()["refusal_gate"], "author_attempts")
        queue, _ = self.prepare()
        self.assertEqual(len(queue), 0)

    def test_a_fresh_accepted_hypothesis_is_kept_too_and_planner_abstention_is_not(self):
        planner = base.Planner(self.repo, [base.hyp()])
        planner.author = lambda h, c: loop.Abstain("infeasible")
        outcome = loop.iterate(planner=planner, critic=base.Critic(), context={},
                               measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None, author_attempts=3)
        self.assertEqual(outcome.status, loop.AUTHORING_FAILED)
        self.assertEqual(outcome.resume_checkpoints[0]["author_attempts_used"], 1)
        # The PLANNER declining to propose is unaffected: still `abstained`.
        quiet = mock.Mock()
        quiet.propose.return_value = loop.Abstain("nothing left")
        outcome = loop.iterate(planner=quiet, critic=base.Critic(), context={},
                               measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None)
        self.assertEqual((outcome.status, outcome.hypothesis), ("abstained", None))


# ------------------------------------------------------------------ 3. best-of panel


class PanelNoWinner(tb.Fixture):
    """The run 10h round, reproduced: a0 abstains over a diff, a1's edit is truncated."""

    def a0(self, spec, ws, stop):
        case = self

        class A0(tb.Editor):
            def author(inner, hypothesis, context):
                (inner.ws / tb.TARGET).write_text("int f(void) { return 512; }\n")
                write_rows(inner.ws.parent / actor_metrics.REPLY_DIR_NAME
                           / actor_metrics.CALL_LOG_NAME,
                           metrics_row(thinking="off", ak_check=AK_CHECK))
                return loop.Abstain(A0_REASON)
        del case
        return A0(spec, ws, stop)

    def a1(self, spec, ws, stop, *, capped=True):
        class A1(tb.Editor):
            def author(inner, hypothesis, context):
                write_rows(inner.ws.parent / actor_metrics.REPLY_DIR_NAME
                           / actor_metrics.CALL_LOG_NAME,
                           metrics_row(thinking="medium", capped=capped, capped_steps=3))
                raise actors.AuthorReplyMissing(A1_REASON)
        return A1(spec, ws, stop)

    def test_a_mixed_panel_returns_an_authoring_failure_with_every_member(self):
        panel = self.panel({"off": self.a0, "medium": self.a1})
        result, records = self.call(panel)
        self.assertIsInstance(result, loop.AuthoringFailure)
        members = {m["label"]: m for m in result.members}
        self.assertEqual((members["a0-off"]["class"], members["a1-medium"]["class"]),
                         ("authoring", "harness"))
        self.assertTrue(members["a0-off"]["patch"]["patch_file"].endswith(".patch"))
        self.assertEqual(members["a0-off"]["ak_check"]["by_mode"]["op-test"]["fail"], 1)
        # The panel record: the truncated member is surfaced as output_capped_empty.
        row = {m["label"]: m for m in records[0]["members"]}
        self.assertEqual(row["a1-medium"]["failure_class"], "output_capped_empty")
        self.assertEqual(row["a1-medium"]["failure"]["class"], "harness")
        self.assertTrue(row["a1-medium"]["final_step_capped"])
        self.assertEqual(row["a0-off"]["failure"]["class"], "authoring")
        self.assert_no_scratch_left()

    def test_an_all_harness_panel_raises_with_the_records_attached(self):
        panel = self.panel({"off": self.a1, "medium": self.a1})
        with self.assertRaises(loop.AuthorReportMissing) as caught:
            self.call(panel)
        self.assertEqual({r["class"] for r in caught.exception.author_failures}, {"harness"})

    def test_through_iterate_a_mixed_panel_charges_one_attempt(self):
        panel = self.panel({"off": self.a0, "medium": self.a1})
        outcome = loop.iterate(
            planner=tb._Planner(), critic=tb._Critic(), context={}, patch_rounds=2,
            hypothesis_rounds=1, measure=lambda h, p: None, gate=lambda h, p: (True, []),
            commit=lambda *a: "head", author_lane=(self.lane, self.base), author_panel=panel,
            author_attempts=3)
        self.assertEqual(outcome.status, loop.AUTHORING_FAILED)
        self.assertEqual(outcome.hypothesis_pending["author_attempts_used"], 1)   # not 2
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual(len(checkpoint["authoring_failures"]), 2)
        feedback = checkpoint["prior_patch_rejections"]
        self.assertTrue(any("a0-off" in line and "op-test 0/2 pass" in line
                            and ".patch" in line for line in feedback))
        self.assertTrue(any("a1-medium" in line and "not charged" in line
                            for line in feedback))
        self.assertEqual(len(outcome.author_panels), 1)

    def test_through_iterate_an_all_harness_panel_charges_nothing(self):
        panel = self.panel({"off": self.a1, "medium": self.a1})
        outcome = loop.iterate(
            planner=tb._Planner(), critic=tb._Critic(), context={}, patch_rounds=2,
            hypothesis_rounds=1, measure=lambda h, p: None, gate=lambda h, p: (True, []),
            commit=lambda *a: "head", author_lane=(self.lane, self.base), author_panel=panel)
        self.assertEqual(outcome.status, loop.AUTHORING_HARNESS_FAILURE)
        self.assertEqual(outcome.hypothesis_pending["author_attempts_used"], 0)


# ------------------------------------------------------------------ 4. member budgets


class AsymmetricMemberBudgets(unittest.TestCase):

    def test_the_default_off_medium_split(self):
        budget = bestof.panel_budget(bestof.parse_authors("off,medium"))
        self.assertEqual([(m.label, m.context_limit, m.output_limit) for m in budget.members],
                         [("a0-off", 65_536, 16_384), ("a1-medium", 114_688, 40_960)])
        self.assertEqual(sum(m.context_limit for m in budget.members), 196_608 - 16_384)
        self.assertEqual(budget.concurrent_peak, 196_608)
        for member in budget.members:
            self.assertLessEqual(member.output_limit,
                                 member.context_limit - bestof.MIN_COMPACTION_HEADROOM)
        body = budget.to_dict()
        self.assertEqual(body["members"][1]["compaction_at"], 114_688 - 40_960)
        self.assertIsNone(body["context_limit"])

    def test_same_modes_are_symmetric_and_overrides_merge(self):
        same = bestof.panel_budget(bestof.parse_authors("off,off"))
        self.assertEqual((same.context_limit, same.output_limit), (90_112, 16_384))
        modes = bestof.parse_mode_budgets("medium=4:16384")
        self.assertEqual(modes["off"], bestof.DEFAULT_MODE_BUDGETS["off"])
        even = bestof.panel_budget(bestof.parse_authors("off,medium"), modes=modes)
        self.assertEqual((even.context_limit, even.output_limit), (90_112, 16_384))

    def test_refusals(self):
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "outside"):
            bestof.panel_budget(bestof.parse_authors("off") * 5)
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "reserve"):
            bestof.panel_budget(bestof.parse_authors("off,medium"), pool_tokens=16_384)
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "compacts"):
            bestof.panel_budget(bestof.parse_authors("off,medium,medium"))   # off: 40,960
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "compacts"):
            bestof.panel_budget(bestof.parse_authors("off,medium"),
                                modes=bestof.parse_mode_budgets("medium=7:90000"))
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "no author budget"):
            bestof.panel_budget(bestof.parse_authors("off,medium"),
                                modes={"off": (4, 16_384)})
        for bad in ("medium", "medium=7", "medium=x:1", "medium=0:40960", "=1:2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bestof.parse_mode_budgets(bad)

    def test_run_py_wires_each_members_limits_and_the_output_ceiling_env(self):
        from autokernel.loop import run
        args = SimpleNamespace(actor_authors=None, workers=1, actor_pool_tokens=196_608,
                               actor_authors_budget="")
        with mock.patch.object(run.actor_opencode_config, "THINKING_CHOICES",
                               ("default", "off", "medium")):
            plan = run._author_plan(args, "opencode")
        self.assertIn("a1-medium(context=114688 output=40960", plan.note)
        with self.assertRaises(bestof.PoolBudgetRefused):
            with mock.patch.object(run.actor_opencode_config, "THINKING_CHOICES",
                                   ("default", "off", "medium")):
                run._author_plan(SimpleNamespace(**{**vars(args),
                                                    "actor_authors_budget": "off=4:60000"}),
                                 "opencode")
        source = Path(run.__file__).read_text()
        self.assertIn('"context_limit": budget.for_member(spec).context_limit', source)
        self.assertIn('"author_output_limit": budget.for_member(spec).output_limit', source)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "lane"
            ws.mkdir()
            backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
            for member in plan.budget.members:
                seat = actors.ActorSeat(bounded=False).for_author(
                    thinking=member.thinking, context_limit=member.context_limit,
                    output_limit=member.output_limit)
                env = actors._seat_call(seat, backend, "author", ws / member.label, {})
                config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
                limit = next(iter(next(iter(config["provider"].values()))["models"].values()))
                self.assertEqual(limit["limit"], {"context": member.context_limit,
                                                  "output": member.output_limit})
                if member.output_limit > aoc.OPENCODE_OUTPUT_TOKEN_MAX:
                    self.assertEqual(env[aoc.OUTPUT_TOKEN_MAX_ENV], str(member.output_limit))
                else:
                    self.assertNotIn(aoc.OUTPUT_TOKEN_MAX_ENV, env)

    def test_a_panel_refuses_a_budget_missing_a_member(self):
        specs = bestof.parse_authors("off,medium")
        other = bestof.panel_budget(bestof.parse_authors("medium,off"))
        with self.assertRaisesRegex(ValueError, "no budget for panel member"):
            bestof.AuthorPanel(lane="lane0", specs=specs, make_author=None, scratch=None,
                               budget=bestof.PoolBudget(2, 196_608, 16_384, None, None,
                                                        other.members[:1]))


# ------------------------------------------------------------------ 5. critic pass 1


class CriticTransient:
    """Critic pass 1 fails `fails` times (an empty reply unless `message` says else)."""

    def __init__(self, fails, message="actor produced no final report (0 chars); refusing "
                 "to repair an empty reply", verdict=True):
        self.fails, self.message, self.verdict, self.calls = fails, message, verdict, 0

    def review_hypothesis(self, hypothesis, context):
        self.calls += 1
        if self.calls <= self.fails:
            raise actors.ProviderTransient(self.message)
        return loop.Review(self.verdict, "" if self.verdict else "unsupported premise")

    def review_patch(self, hypothesis, paths, context):
        return loop.Review(True)


class CriticPassOneTransients(tp.Fixture):

    def test_an_empty_reply_is_retried_once_in_the_iteration(self):
        critic = CriticTransient(1)
        outcome = loop.iterate(planner=base.Planner(self.repo, [base.hyp()]), critic=critic,
                               context={}, measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None)
        self.assertEqual((outcome.status, critic.calls), ("measured_null", 2))

    def test_a_critic_transient_keeps_the_planner_hypothesis_at_critic1(self):
        planner = base.Planner(self.repo, [base.hyp()])
        outcome = loop.iterate(planner=planner, critic=CriticTransient(2, "provider 503"),
                               context={}, measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None, hypothesis_rounds=1)
        self.owner.record(outcome)
        self.assertEqual(outcome.status, "planner_transient")
        self.assertEqual(outcome.hypothesis.mechanism_id, "akm-demo-hoist")
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["stage"], checkpoint["critic1_attempts_used"],
                          checkpoint["critic_hypothesis"]), ("critic1", 1, None))
        # The next draw resumes it AT critic pass 1: no planner call.
        point, report = self.take()
        self.assertEqual([q["stage"] for q in report["queued"]], ["critic1"])
        critic = CriticTransient(0)
        resumed = loop.iterate(planner=tp.NoPlanner(self, self.repo), critic=critic,
                               context={}, measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None,
                               resume=point)
        self.owner.record(resumed)
        self.assertEqual((resumed.status, critic.calls), ("measured_null", 1))
        self.assertEqual(resumed.resumed_from, point.checkpoint_id)
        self.assertEqual([row["decision"] for row in resumed.validator_provenance][:1],
                         ["critic:hypothesis"])

    def test_the_retry_budget_drops_it_after_the_last_transient(self):
        outcome = loop.iterate(planner=base.Planner(self.repo, [base.hyp()]),
                               critic=CriticTransient(9, "provider 503"), context={},
                               measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None, hypothesis_rounds=1)
        self.owner.record(outcome)
        for used in range(2, loop.CRITIC1_RETRIES):
            point, _ = self.take()
            outcome = loop.iterate(planner=tp.NoPlanner(self, self.repo),
                                   critic=CriticTransient(9, "provider 503"), context={},
                                   measure=lambda h, p: None, gate=lambda h, p: None,
                                   commit=lambda h, p, c: None, resume=point,
                                   hypothesis_rounds=0)
            self.owner.record(outcome)
            self.assertEqual(outcome.resume_checkpoints[0]["critic1_attempts_used"], used)
        point, _ = self.take()
        outcome = loop.iterate(planner=tp.NoPlanner(self, self.repo),
                               critic=CriticTransient(9, "provider 503"), context={},
                               measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None, resume=point, hypothesis_rounds=0)
        self.owner.record(outcome)
        self.assertEqual((outcome.status, outcome.resume_checkpoints), ("planner_transient", []))
        self.assertIn("retry budget 3 spent", outcome.reasons[0])
        queue, _ = self.prepare()
        self.assertEqual(len(queue), 0)

    def test_a_stop_before_critic_pass_one_keeps_it(self):
        outcome = loop.iterate(planner=base.Planner(self.repo, [base.hyp()]),
                               critic=CriticTransient(0), context={},
                               measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None,
                               should_abandon=iter([False, True, True]).__next__)
        self.assertEqual(outcome.status, "stopped_mid_formation")
        self.assertEqual([c["stage"] for c in outcome.resume_checkpoints], ["critic1"])
        self.assertEqual(outcome.resume_checkpoints[0]["critic1_attempts_used"], 0)

    def test_a_resumed_critic1_rejection_is_a_verdict(self):
        self.owner.record(loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()]), critic=CriticTransient(9, "503"),
            context={}, measure=lambda h, p: None, gate=lambda h, p: None,
            commit=lambda h, p, c: None, hypothesis_rounds=1))
        point, _ = self.take()
        outcome = loop.iterate(planner=base.Planner(self.repo, [base.hyp("akm-fresh")]),
                               critic=CriticTransient(0, verdict=False), context={},
                               measure=lambda h, p: None, gate=lambda h, p: None,
                               commit=lambda h, p, c: None, resume=point,
                               hypothesis_rounds=0, record_abandoned=self.owner.record_abandoned)
        self.assertEqual(outcome.status, "refused_at_formation")
        self.assertEqual(self.owner.claims_at_disposal, ["hypothesis_rejected"])
        self.assertEqual(self.claim(point.checkpoint_id)["result_status"], "hypothesis_rejected")


# ------------------------------------------------------------------ 6. permission rejection


class PermissionRejectedSessions(unittest.TestCase):

    def export(self, tmp: Path) -> Path:
        body = {"info": {}, "messages": [
            {"info": {"role": "assistant", "finish": "tool-calls", "sessionID": "s",
                      "tokens": {"input": 1, "output": 2}},
             "parts": [{"type": "tool", "tool": "read", "state": {
                 "status": "error", "input": {"filePath": "/store/experiments.md"},
                 "error": "The user rejected permission to use this specific tool call."}}]}]}
        path = tmp / "export.json"
        path.write_text(json.dumps(body))
        return path

    def test_the_export_says_a_permission_rejection_ended_the_session(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            parsed = actor_metrics.parse_export(self.export(Path(tmp)))
        self.assertEqual((parsed["permission_rejected"], parsed["final_finish"]),
                         (1, "tool-calls"))

    def test_an_empty_reply_after_it_is_a_named_harness_failure(self):
        raw = actors._Reply("", failure_class=actor_metrics.PERMISSION_REJECTED)
        with self.assertRaises(actors.ProviderTransient) as caught:
            actors._parse_reply(raw, schema=actors.REVIEW_SCHEMA,
                                backend=actors.backend_for("deepseek/flash", "max"),
                                workspace=Path("/nonexistent"))
        self.assertEqual(caught.exception.failure_class, actor_metrics.PERMISSION_REJECTED)
        self.assertIn("refused tool permission", str(caught.exception))
        self.assertTrue(loop._empty_reply(caught.exception))
        self.assertEqual(loop.classify_author_failure(
            "report_missing", failure_class="permission_rejected")["class"], "harness")

    def test_the_read_only_critic_may_read_the_store_its_context_points_at(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "lane"
            ws.mkdir()
            store = Path(tmp) / "store"
            backend = actors.backend_for("deepseek/deepseek-flash", "max")
            seat = actors.ActorSeat(bounded=False, lane_guard=True)
            context = {"actor_read_roots": [str(store)]}
            env = actors._seat_call(seat, backend, "critic", ws, context)
            config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
            self.assertEqual(config["permission"]["external_directory"][f"{store}/*"], "allow")
            self.assertEqual(config["permission"]["edit"], "deny")   # reads only
            author = actors._seat_call(seat, backend, "author", ws, context)
            author_config = json.loads(Path(author["OPENCODE_CONFIG"]).read_text())
            self.assertNotIn(f"{store}/*", author_config["permission"].get(
                "external_directory", {}))
        self.assertEqual(actors._read_roots({"actor_read_roots": ["relative", "/"]}), ())

    def test_run_py_hands_the_store_to_the_critic(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text()
        block = source[source.index('"actor_read_roots"'):][:300]
        # The store and the campaign root holding it (inputs/, state-*/): run 10i's
        # critic grepped the campaign root.
        self.assertIn("(Path(args.store).resolve(), Path(args.store).resolve().parent)", block)


# ------------------------------------------------------------------ 7. reinstate / backfill


class ReinstateFromAnAbstainedPanelRow(tp.Fixture):
    """Run 10h's shape: a reinstated author checkpoint, a resumed panel iteration that
    ended `abstained` under the old policy and consumed the claim."""

    def seed(self):
        self.attempt([tp.rejected(tp.AUTHORING_1), tp.rejected(tp.AUTHORING_2)])
        point, _ = self.take()
        panel = {"schema": bestof.PANEL_SCHEMA, "panel_id": "0c1e34bcc06df2fb",
                 "winner": None, "selection": "none",
                 "pool": {"context_limit": 90112, "output_limit": 16384},
                 "members": [
                     {"label": "a0-off", "thinking": "off", "outcome": "abstained",
                      "reason": A0_REASON, "result": "lost", "validation": None,
                      "patch": {"patch_file": str(self.store / "patches" / "a0.patch"),
                                "patch_sha256": "9c" * 32, "bytes": 5311}},
                     {"label": "a1-medium", "thinking": "medium",
                      "outcome": "report_missing", "reason": A1_REASON, "result": "lost",
                      "validation": None, "patch": None, "failure_class": None}]}
        old = loop.Outcome("abstained", point.hypothesis, [A0_REASON],
                           resumed_from=point.checkpoint_id, resume_stage="author",
                           author_panels=[panel]).to_attempt()
        old["spawn_parent"] = self.anchor
        with experiments.ExperimentStore(self.store) as store:
            store.record(old, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
        with resume.ClaimLedger(self.store) as ledger:
            ledger.settle(point.checkpoint_id, self.anchor, result_status="abstained")
        calls = self.tmp / "actor-calls.jsonl"
        for label, row in (("a0-off", metrics_row(thinking="off", ak_check=AK_CHECK)),
                           ("a1-medium", metrics_row(thinking="medium", capped=True,
                                                     capped_steps=3))):
            write_rows(calls, {**row, "author_panel": {"panel_id": panel["panel_id"],
                                                       "label": label}})
        with experiments.ExperimentStore(self.store) as store:
            ids = {status: found[0] for status, found in (
                (status, store._connection.execute(
                    "SELECT attempt_id FROM experiments WHERE status=? ORDER BY rowid DESC",
                    (status,)).fetchone())
                for status in (loop.PATCH_ROUNDS_EXHAUSTED, "abstained"))}
        return ids, calls, point

    def test_the_dry_run_plans_an_authoring_failed_row_from_the_panel(self):
        ids, calls, point = self.seed()
        queue, _ = self.prepare()
        self.assertEqual(len(queue), 0)                         # dropped, as in run 10h
        plan = resume.reinstate_plan(
            self.store, row_id=ids[loop.PATCH_ROUNDS_EXHAUSTED][:12],
            rejection_rows=[ids["abstained"][:12]], epoch=EPOCH, measurement_epoch="5" * 64,
            actor_calls=calls, attempts_used=1)
        attempt = plan["attempt"]
        self.assertEqual(attempt["status"], loop.AUTHORING_FAILED)
        (checkpoint,) = attempt["resume_checkpoints"]
        self.assertEqual((checkpoint["author_attempts_used"], checkpoint["resumed_from"]),
                         (1, point.checkpoint_id))
        failures = {f["label"]: f for f in checkpoint["authoring_failures"]}
        self.assertEqual((failures["a0-off"]["class"], failures["a1-medium"]["class"]),
                         ("authoring", "harness"))
        self.assertEqual(failures["a1-medium"]["failure_class"], "output_capped_empty")
        feedback = checkpoint["prior_patch_rejections"]
        self.assertEqual(feedback[:2], [tp.AUTHORING_1, tp.AUTHORING_2])
        self.assertTrue(any("a0.patch" in line and "op-test 0/2 pass" in line
                            for line in feedback))
        self.assertTrue(plan["checks"]["source_checkpoint_consumed"])
        self.assertIs(plan["checks"]["prevalidates_at_anchor"], True)
        self.assertEqual(attempt["reinstated_from"]["charged_lineages"], [point.checkpoint_id])
        # The default would charge a0's abstention: 1 (carried) + 1.
        default = resume.reinstate_plan(
            self.store, row_id=ids[loop.PATCH_ROUNDS_EXHAUSTED][:12],
            rejection_rows=[ids["abstained"][:12]], epoch=EPOCH, actor_calls=calls)
        self.assertEqual(default["attempt"]["resume_checkpoints"][0]["author_attempts_used"], 2)
        self.assertTrue(resume.reinstate_apply(self.store, plan))
        successor, report = self.take()
        self.assertEqual([(q["stage"], q["row_status"]) for q in report["queued"]],
                         [("author", loop.AUTHORING_FAILED)])
        self.assertIn(A0_REASON, " ".join(successor.prior_patch_rejections))

    def test_the_cli_takes_the_panel_row_and_refuses_a_planner_abstention(self):
        ids, calls, _point = self.seed()
        out = io.StringIO()
        with redirect_stdout(out):
            code = resume.main(["reinstate", "--store", str(self.store), "--row",
                                ids[loop.PATCH_ROUNDS_EXHAUSTED][:12], "--rejection",
                                ids["abstained"][:12], "--epoch", EPOCH, "--attempts-used",
                                "1", "--actor-calls", str(calls)])
        self.assertEqual(code, 0)
        self.assertIn("dry-run: nothing written", out.getvalue())
        self.assertIn('"class": "authoring_failed"', out.getvalue())
        planner_abstained = loop.Outcome("abstained", None, ["nothing left"]).to_attempt()
        with experiments.ExperimentStore(self.store) as store:
            store.record(planner_abstained, epoch=EPOCH, recorded_at=loop._now(),
                         campaign_id="ak-loop")
            quiet = store._connection.execute(
                "SELECT attempt_id FROM experiments WHERE status='abstained' "
                "AND json_extract(payload, '$.mechanism_id') IS NULL").fetchone()[0]
        with self.assertRaisesRegex(ValueError, "different hypothesis"):
            resume.reinstate_plan(self.store, row_id=ids[loop.PATCH_ROUNDS_EXHAUSTED][:12],
                                  rejection_rows=[quiet], epoch=EPOCH)


class BackfillCriticOne(tp.Fixture):

    def seed(self):
        row = loop.Outcome("planner_transient", None, [
            "actor produced no final report (0 chars); refusing to repair an empty reply"
        ]).to_attempt()
        row.update(spawn_parent=self.anchor, branch_id="detached:lane0",
                   research_scope={"measurement_surface": "serving:demo",
                                   "model": {"path": "/models/demo.gguf"}})
        with experiments.ExperimentStore(self.store) as store:
            store.record(row, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
            lost = store._connection.execute(
                "SELECT attempt_id FROM experiments WHERE status='planner_transient'"
            ).fetchone()[0]
        reply = self.tmp / "20260926T124612-opencode-planner-rc0.stdout"
        reply.write_text(json.dumps({**base.hyp("akm-verify-batch-solo-2rows").to_dict()},
                                    indent=1))
        import os
        os.utime(reply, (0, 0))
        return lost, reply

    def test_a_lost_planner_hypothesis_is_requeued_at_critic1(self):
        lost, reply = self.seed()
        out = io.StringIO()
        with redirect_stdout(out):
            code = resume.main(["backfill-critic1", "--store", str(self.store), "--row",
                                lost[:12], "--reply", str(reply), "--epoch", EPOCH,
                                "--measurement-epoch", "4" * 64])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("dry-run: nothing written", out.getvalue())
        plan = resume.backfill_critic1_plan(self.store, row_id=lost[:12], reply=reply,
                                            epoch=EPOCH)
        (checkpoint,) = plan["attempt"]["resume_checkpoints"]
        self.assertEqual((checkpoint["stage"], checkpoint["critic1_attempts_used"],
                          checkpoint["hypothesis"]["mechanism_id"]),
                         ("critic1", 1, "akm-verify-batch-solo-2rows"))
        self.assertEqual(checkpoint["target"], TARGET)
        self.assertTrue(plan["checks"]["reply_precedes_row"])
        self.assertIs(plan["checks"]["prevalidates_at_anchor"], True)
        self.assertTrue(resume.backfill_critic1_apply(self.store, plan))
        self.assertFalse(resume.backfill_critic1_apply(self.store, plan))   # idempotent
        point, report = self.take()
        self.assertEqual((point.stage, report["queued"][0]["mechanism_id"]),
                         ("critic1", "akm-verify-batch-solo-2rows"))

    def test_refusals(self):
        lost, reply = self.seed()
        abstain = self.tmp / "abstain.stdout"
        abstain.write_text('{"abstain": "nothing"}')
        with self.assertRaisesRegex(ValueError, "abstention"):
            resume.backfill_critic1_plan(self.store, row_id=lost, reply=abstain, epoch=EPOCH)
        with self.assertRaisesRegex(ValueError, "--epoch-reason"):
            resume.backfill_critic1_plan(self.store, row_id=lost, reply=reply, epoch="d" * 64)
        with self.assertRaisesRegex(ValueError, "attempts-used"):
            resume.backfill_critic1_plan(self.store, row_id=lost, reply=reply, epoch=EPOCH,
                                         attempts_used=loop.CRITIC1_RETRIES)


# ------------------------------------------------------------------ 8. classifiers


class TheNewStatusesAreKnown(unittest.TestCase):

    def test_every_classifier_knows_them(self):
        self.assertEqual(serial_scheduling.one_iteration_outcome(
            "complete", {loop.AUTHORING_FAILED: 1}), "invalid")
        self.assertEqual(serial_scheduling.one_iteration_outcome(
            "complete", {loop.AUTHORING_HARNESS_FAILURE: 1}), "failed")
        for name in (loop.AUTHORING_FAILED, loop.AUTHORING_HARNESS_FAILURE):
            self.assertIn(name, process_metrics.VALID_DENOMINATOR)
            self.assertIn(name, experiments._STATUS_MERIT)
            self.assertIn(name, resume.SCANNED_STATUSES)
            self.assertIn(name, loop.PENDING_HYPOTHESIS_STATUSES)

    def test_the_resume_chain_allows_the_harness_retries(self):
        self.assertEqual(resume.depth_limit({"author_attempts_budget": 3}),
                         resume.MAX_RESUME_DEPTH + 3 * loop.AUTHOR_HARNESS_FAILURE_CAP)


if __name__ == "__main__":
    unittest.main()
