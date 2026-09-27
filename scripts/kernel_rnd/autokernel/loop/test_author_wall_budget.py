"""Best-of-N author wall budgets (`bestof.WallBudget`), with fake panel members.

Origin: DS41 run 10j, round 1 on `akm-q4k-x4t-avx512` (2026-09-26/27). a0-off gave up
after 61 min; a1-medium hit the 7200 s actor timeout, was RETRIED in-round as an
uncharged harness failure and spent another 6667 s: ~3.9 h for one round, whose winning
patch critic2 then proved wrong. Operator 2026-09-27: per-member wall budgets by
thinking mode (off 2700 s, medium/default 4500 s), a 5400 s panel wall, no in-round
retry of a timed-out or budget-stopped member, and early cancel at 2x a finisher's wall.

The members here are `test_bestof`'s fakes on the same real-repo fixture; budgets are
fractions of a second so the stop predicate (polled by the fakes as the actor's C22
path polls it) trips within the test.

Pinned run:
    nice -n 19 taskset -c 96-111 timeout 900 python3 -m pytest -q -p no:cacheprovider \\
        autokernel/loop/test_author_wall_budget.py
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest import mock

from autokernel.loop import actor_metrics, actors, bestof, loop
from autokernel.loop import test_bestof as tb
from autokernel.loop.loop import Abstain, ActorStopped, ActorTransient


def walls(off=5.0, medium=5.0, panel=10.0, factor=2.0) -> bestof.WallBudget:
    return bestof.WallBudget({"off": off, "medium": medium, "default": medium}, panel, factor)


class Abstainer(tb.Editor):
    """Abstains (a genuine finish) after `delay` seconds."""

    def author(self, hypothesis, context):
        time.sleep(self.delay)
        return Abstain(f"{self.spec.label} cannot")


class Waiter(tb.Blocker):
    """Runs until the panel's stop reaches it; `edit` writes a target diff first."""

    def __init__(self, spec, ws, stop, *, edit=True, calls=None):
        super().__init__(spec, ws, stop, edit=edit)
        self.calls = calls

    def author(self, hypothesis, context):
        if self.calls is not None:
            self.calls.append(self.spec.label)
        return super().author(hypothesis, context)


def by_label(row) -> dict:
    return {m["label"]: m for m in row["members"]}


# --------------------------------------------------------------------------- knobs


class Knobs(unittest.TestCase):

    def test_defaults_are_the_operator_budgets(self):
        budget = bestof.wall_budget()
        self.assertEqual(dict(budget.member_s), {"off": 2700, "medium": 4500, "default": 4500})
        self.assertEqual((budget.panel_s, budget.cancel_factor), (5400, 2.0))
        specs = bestof.parse_authors("off,medium,default")
        self.assertEqual([budget.for_member(s) for s in specs], [2700, 4500, 4500])
        self.assertEqual(budget.to_dict(), {"member_s": {"default": 4500, "medium": 4500,
                                                         "off": 2700},
                                            "panel_s": 5400, "cancel_factor": 2.0})

    def test_member_walls_merge_and_the_panel_wall_caps_them(self):
        budget = bestof.wall_budget("off=1200, medium=9000", "6000", "0")
        self.assertEqual(dict(budget.member_s), {"off": 1200, "medium": 9000, "default": 4500})
        medium = bestof.parse_authors("off,medium")[1]
        self.assertEqual(budget.for_member(medium), 6000)      # capped by the panel wall
        self.assertEqual(budget.cancel_factor, 0.0)
        self.assertEqual(bestof.parse_member_walls(""), bestof.DEFAULT_MEMBER_WALL_S)
        self.assertEqual(bestof.parse_member_walls(None), bestof.DEFAULT_MEMBER_WALL_S)

    def test_refusals(self):
        for bad in ("off", "off=", "=10", "off=x", "off=0", "off=-5", "off=1.5",
                    "high=10", "off=10,off=20"):
            with self.subTest(wall=bad), self.assertRaises(ValueError):
                bestof.parse_member_walls(bad)
        with self.assertRaisesRegex(ValueError, "unknown thinking mode"):
            bestof.parse_member_walls("medium=10", allowed=("default", "off"))
        for bad in ("0", "-1", "abc", "", "1.5", None):
            with self.subTest(panel=bad), self.assertRaises(ValueError):
                bestof.parse_panel_wall(bad)
        for bad in ("-1", "0.5", "nan", "inf", "abc", ""):
            with self.subTest(factor=bad), self.assertRaises(ValueError):
                bestof.parse_cancel_factor(bad)
        self.assertEqual(bestof.parse_cancel_factor("1"), 1.0)
        self.assertEqual(bestof.parse_cancel_factor(2.5), 2.5)

    def args(self, **kw):
        base = dict(actor_authors=None, workers=1, actor_pool_tokens=196_608,
                    actor_authors_budget="", actor_authors_wall="",
                    actor_authors_panel_wall="5400", actor_authors_cancel_factor="2.0")
        return SimpleNamespace(**{**base, **kw})

    def test_run_py_resolves_and_refuses_the_knobs(self):
        from autokernel.loop import run
        with mock.patch.object(run.actor_opencode_config, "THINKING_CHOICES",
                               ("default", "off", "medium")):
            plan = run._author_plan(self.args(), "opencode")
            self.assertTrue(plan.panel)
            self.assertEqual(plan.walls, bestof.wall_budget())
            self.assertIn("a0-off(context=65536 output=16384 compaction@49152 wall=2700s)",
                          plan.note)
            self.assertIn("panel_wall=5400s cancel_factor=2", plan.note)
            # Refused whatever N is: a typo must not wait for the first panel.
            for bad in (dict(actor_authors_wall="off=0"),
                        dict(actor_authors_panel_wall="0"),
                        dict(actor_authors_cancel_factor="0.5"),
                        dict(actor_authors="single", actor_authors_wall="high=10")):
                with self.subTest(**bad), self.assertRaises(ValueError):
                    run._author_plan(self.args(**bad), "opencode")
            # Knobs absent (an older caller's namespace): the defaults.
            legacy = SimpleNamespace(actor_authors=None, workers=1,
                                     actor_pool_tokens=196_608)
            self.assertEqual(run._author_plan(legacy, "opencode").walls, bestof.wall_budget())

    def test_run_parser_defaults_and_member_seats_never_retry_a_timeout(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text(encoding="utf-8")
        self.assertIn('"--actor-authors-panel-wall", default=str(bestof.DEFAULT_PANEL_WALL_S)',
                      source)
        self.assertIn('"--actor-authors-cancel-factor",\n'
                      '                        default=str(bestof.DEFAULT_CANCEL_FACTOR)',
                      source)
        self.assertIn('parser.add_argument("--actor-authors-wall", default=""', source)
        # Only the panel member's seat turns the timeout retry off; the single path
        # (make_planner) never names it.
        self.assertEqual(source.count("retry_timeouts=False"), 1)
        member = source[source.index("def make_author(spec, workspace, member_stop):"):]
        self.assertIn("retry_timeouts=False", member[:member.index("return bestof.AuthorPanel(")])
        self.assertIn("should_stop=should_stop, walls=author_plan.walls", source)

    def test_serial_common_args_admit_the_knobs(self):
        import json as _json
        import tempfile
        from autokernel.loop import serial_roster, serial_run as sr
        from autokernel.loop.test_serial_roster import _inputs
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _resolved, _owners, argv = _inputs(tmp, backends=("cpu",))
            common = tmp / "common.json"
            common.write_text(_json.dumps(["--actor-authors-wall", "off=2400,medium=4200",
                                           "--actor-authors-panel-wall=5000",
                                           "--actor-authors-cancel-factor", "0"]))
            targets, _skipped, _cpus = serial_roster.build_targets(
                Path(sr.option(argv, "--resolved-campaign")),
                Path(sr.option(argv, "--owned-targets")),
                target_root=Path(sr.option(argv, "--state-dir")) / "targets",
                common_path=common)
            self.assertEqual(sr.option(targets[0], "--actor-authors-wall"),
                             "off=2400,medium=4200")
            self.assertEqual(sr.option(targets[0], "--actor-authors-cancel-factor"), "0")
            common.write_text(_json.dumps(["--actor-authors-wall"]))
            with self.assertRaisesRegex(sr.SerialRefused, "no valid value"):
                serial_roster.build_targets(
                    Path(sr.option(argv, "--resolved-campaign")),
                    Path(sr.option(argv, "--owned-targets")),
                    target_root=Path(sr.option(argv, "--state-dir")) / "targets",
                    common_path=common)


# --------------------------------------------------------------------------- timeouts


class TimeoutsAreNotRetriedInAPanel(unittest.TestCase):

    def test_with_backoff_retries_a_timeout_only_on_the_single_path(self):
        calls, slept = [], []

        def timed_out():
            calls.append(1)
            if len(calls) == 1:
                raise actors.ActorTimedOut("actor exceeded 7200s")
            return "reply"

        # The single path (default): retried, byte for byte as before.
        self.assertEqual(actors._with_backoff(timed_out, sleep=slept.append), ("reply", 1))
        self.assertEqual((len(calls), slept), (2, [actors.BACKOFF_S[0]]))
        calls.clear()
        with self.assertRaisesRegex(actors.ActorTimedOut, "7200s"):
            actors._with_backoff(timed_out, sleep=slept.append, retry_timeouts=False)
        self.assertEqual(len(calls), 1)
        # Other provider transients keep their retry inside a member's wall.
        flaky = iter([actors.ProviderTransient("503"), "ok"])

        def other():
            item = next(flaky)
            if isinstance(item, Exception):
                raise item
            return item
        self.assertEqual(actors._with_backoff(other, sleep=lambda _s: None,
                                              retry_timeouts=False), ("ok", 1))
        self.assertTrue(issubclass(actors.ActorTimedOut, actors.ProviderTransient))
        self.assertTrue(actors.ActorTimedOut("x").timed_out)

    def test_agent_planner_forwards_the_switch_and_defaults_to_retry(self):
        self.assertTrue(actors.AgentPlanner(workspace=Path(".")).retry_timeouts)
        import tempfile
        seen = {}

        class Seen(Exception):
            pass

        def spy(call, **kwargs):
            seen.update(kwargs)
            raise Seen()

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "lane"
            ws.mkdir()
            for flag, expected in ((True, None), (False, False)):
                seen.clear()
                planner = actors.AgentPlanner(workspace=ws, retry_timeouts=flag)
                with mock.patch.object(actors, "_with_backoff", spy), \
                        mock.patch.object(actors.AgentPlanner, "_context_block",
                                          lambda self, role, context: ("", None)), \
                        self.assertRaises(Seen):
                    planner.author(tb.HYP, {})
                self.assertEqual(seen.get("retry_timeouts"), expected)


# --------------------------------------------------------------------------- the panel


class PanelWallBudgets(tb.Fixture):

    def test_a_budget_stop_salvages_the_diff_and_it_can_win(self):
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           walls=walls(off=0.3, medium=5.0))
        paths, records = self.call(panel)
        self.assertEqual(tuple(paths), (tb.TARGET,))
        self.assertEqual((self.lane / tb.TARGET).read_text(),
                         "int f(void) { return 99; } /* partial */\n")
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("first_passing", "a0-off"))
        members = by_label(row)
        off, medium = members["a0-off"], members["a1-medium"]
        self.assertEqual((off["result"], off["outcome"], off["report_source"]),
                         ("won", "diff", "lane_diff"))
        self.assertTrue(off["budget_stopped"])
        self.assertEqual((off["stop_cause"], off["wall_budget_s"]), ("member_budget", 0.3))
        self.assertEqual(off["failure_class"], "member_budget_exhausted")
        self.assertEqual(off["salvage"], {"usable": True, "paths": [tb.TARGET]})
        self.assertTrue(off["validation"]["passed"])
        self.assertGreaterEqual(off["wall_s"], 0.3)
        # The other member lost to the winner (not a budget stop of its own).
        self.assertEqual((medium["result"], medium["stop_cause"], medium["budget_stopped"]),
                         ("cancelled", "winner", False))
        self.assertEqual(medium["wall_budget_s"], 5)
        self.assertEqual(row["walls"]["member_s"]["off"], 0.3)
        self.assertFalse(row["walls"]["panel_wall_hit"])
        self.assertGreater(row["walls"]["panel_wall_s"], 0.0)
        self.assert_no_scratch_left()

    def test_a_budget_stop_with_no_diff_never_retries_and_the_other_diff_wins(self):
        calls = []
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st, edit=False, calls=calls),
                            "medium": lambda s, w, st: tb.Editor(s, w, st, delay=0.6)},
                           walls=walls(off=0.2, medium=5.0, factor=0))
        paths, records = self.call(panel)
        self.assertEqual(tuple(paths), (tb.TARGET,))
        self.assertEqual(calls, ["a0-off"])      # one call: no in-round retry
        members = by_label(records[0])
        self.assertEqual((members["a1-medium"]["result"], records[0]["winner"]),
                         ("won", "a1-medium"))
        off = members["a0-off"]
        self.assertEqual((off["result"], off["outcome"], off["budget_stopped"]),
                         ("lost", "stopped", True))
        self.assertEqual(off["failure_class"], "member_budget_exhausted")
        self.assertEqual(off["failure"]["class"], "harness")
        self.assertEqual(off["salvage"]["usable"], False)
        self.assertTrue(off["reason"].startswith("member_budget_exhausted: stopped at its "
                                                 "wall budget (0.2 s"))
        self.assert_no_scratch_left()

    def test_a_timed_out_member_is_not_retried_and_its_diff_is_salvaged(self):
        calls = []

        class TimesOut(tb.Editor):
            def author(inner, hypothesis, context):
                calls.append(inner.spec.label)
                (inner.ws / tb.TARGET).write_text("int f(void) { return 42; }\n")
                raise actors.ActorTimedOut("actor exceeded 7200s")

        panel = self.panel({"off": lambda s, w, st: tb.Editor(s, w, st, path=tb.OTHER,
                                                              text="int g(void){return 3;}\n"),
                            "medium": TimesOut}, walls=walls(factor=0))
        paths, records = self.call(panel)
        self.assertEqual(calls, ["a1-medium"])
        members = by_label(records[0])
        medium = members["a1-medium"]
        self.assertEqual((records[0]["winner"], medium["result"], medium["outcome"]),
                         ("a1-medium", "won", "diff"))
        self.assertTrue(medium["timed_out"])
        self.assertFalse(medium["budget_stopped"])
        self.assertEqual(medium["report_source"], "lane_diff")
        self.assertEqual((self.lane / tb.TARGET).read_text(), "int f(void) { return 42; }\n")
        self.assert_no_scratch_left()

    def test_every_member_budget_stopped_is_an_uncharged_harness_failure(self):
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st, edit=False),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           walls=walls(off=0.2, medium=0.3))
        with self.assertRaises(ActorTransient) as caught:
            self.call(panel)
        failure = caught.exception
        self.assertNotIsInstance(failure, ActorStopped)     # never read as a run stop
        records = failure.author_failures
        self.assertEqual({r["class"] for r in records}, {"harness"})
        self.assertEqual({r["failure_class"] for r in records}, {"member_budget_exhausted"})
        self.assertEqual({r["harness_reason"] for r in records}, {"member_budget_exhausted"})
        self.assertEqual(sorted(r["wall_budget_s"] for r in records), [0.2, 0.3])
        self.assertTrue(all(r["budget_stopped"] for r in records))
        self.assert_no_scratch_left()

    def iterate(self, panel, **kw):
        return loop.iterate(
            planner=tb._Planner(), critic=tb._Critic(), context={}, patch_rounds=2,
            hypothesis_rounds=1, measure=lambda h, p: None, gate=lambda h, p: (True, []),
            commit=lambda *a: "head", author_lane=(self.lane, self.base), author_panel=panel,
            author_attempts=3, **kw)

    def test_through_iterate_all_budget_stopped_charges_nothing(self):
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st, edit=False),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           walls=walls(off=0.2, medium=0.2))
        outcome = self.iterate(panel)
        self.assertEqual(outcome.status, loop.AUTHORING_HARNESS_FAILURE)
        pending = outcome.hypothesis_pending
        self.assertEqual((pending["author_attempts_used"], pending["charged"],
                          pending["author_harness_failures"]), (0, False, 1))
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual(checkpoint["stage"], "author")     # retried at the next draw
        self.assertEqual({m["failure_class"] for m in checkpoint["authoring_failures"]},
                         {"member_budget_exhausted"})
        self.assertTrue(any("not charged" in line and "member_budget_exhausted" in line
                            for line in checkpoint["prior_patch_rejections"]))
        self.assertEqual(len(outcome.author_panels), 1)     # one panel: no in-round retry

    def test_the_consecutive_harness_cap_still_charges_the_third(self):
        # The cap patched to 1: this all-budget-stopped round is the "third in a row".
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st, edit=False),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           walls=walls(off=0.1, medium=0.1))
        with mock.patch.object(loop, "AUTHOR_HARNESS_FAILURE_CAP", 1):
            outcome = self.iterate(panel)
        self.assertEqual(outcome.status, loop.AUTHORING_FAILED)
        self.assertTrue(outcome.hypothesis_pending["charged"])
        self.assertEqual(outcome.hypothesis_pending["author_attempts_used"], 1)

    def test_early_cancel_at_twice_the_finishers_wall(self):
        panel = self.panel({"off": lambda s, w, st: Abstainer(s, w, st, delay=0.3),
                            "medium": lambda s, w, st: Waiter(s, w, st)},
                           walls=walls(off=5.0, medium=5.0, factor=2.0))
        result, records = self.call(panel)
        self.assertIsInstance(result, loop.AuthoringFailure)     # a genuine abstention
        row = records[0]
        medium = by_label(row)["a1-medium"]
        self.assertEqual((medium["result"], medium["reason"], medium["stop_cause"]),
                         ("cancelled", "early_cancel", "early_cancel"))
        self.assertTrue(medium["early_cancelled"])
        self.assertFalse(medium["budget_stopped"])
        early = row["walls"]["early_cancel"]
        self.assertEqual((early["label"], early["cancelled"]), ("a0-off", ["a1-medium"]))
        self.assertAlmostEqual(early["threshold_s"], 2 * early["wall_s"], places=6)
        # Ended past 2x the finisher's wall, long before its own 5 s budget.
        self.assertGreater(medium["wall_s"], early["threshold_s"])
        self.assertLess(medium["wall_s"], 3.0)
        # Its partial edit is retained for the record; it was not a selection candidate.
        self.assertTrue(medium["patch"]["patch_file"].endswith(".patch"))
        failures = {m["label"]: m for m in result.members}
        self.assertTrue(failures["a1-medium"]["early_cancelled"])
        self.assertEqual(failures["a0-off"]["class"], "authoring")
        self.assert_no_scratch_left()

    def test_early_cancel_off_and_an_ak_check_pass_are_left_alone(self):
        # factor 0: the slow member runs to its own budget instead.
        panel = self.panel({"off": lambda s, w, st: Abstainer(s, w, st, delay=0.1),
                            "medium": lambda s, w, st: Waiter(s, w, st)},
                           walls=walls(off=5.0, medium=0.8, factor=0))
        _paths, records = self.call(panel)
        medium = by_label(records[0])["a1-medium"]
        self.assertEqual((medium["stop_cause"], medium["early_cancelled"]),
                         ("member_budget", False))
        self.assertIsNone(records[0]["walls"]["early_cancel"])

        # A member whose own sandbox ak-check last PASSED is not cancelled early.
        class Checked(Waiter):
            def author(inner, hypothesis, context):
                log = inner.ws.parent / actor_metrics.REPLY_DIR_NAME / f"ak-check-{inner.ws.name}.jsonl"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(json.dumps({"status": "fail", "mode": "compile"}) + "\n"
                               + json.dumps({"status": "pass", "mode": "op-test"}) + "\n")
                return super().author(hypothesis, context)

        panel = self.panel({"off": lambda s, w, st: Abstainer(s, w, st, delay=0.1),
                            "medium": lambda s, w, st: Checked(s, w, st)},
                           walls=walls(off=5.0, medium=0.9, factor=2.0))
        _paths, records = self.call(panel)
        medium = by_label(records[0])["a1-medium"]
        self.assertEqual((medium["stop_cause"], medium["early_cancelled"]),
                         ("member_budget", False))
        self.assertEqual(records[0]["walls"]["early_cancel"]["cancelled"], [])
        self.assert_no_scratch_left()

    def test_a_harness_failure_does_not_start_the_early_cancel_clock(self):
        class Flaky(tb.Editor):
            def author(inner, hypothesis, context):
                raise actors.ProviderTransient("503")

        panel = self.panel({"off": Flaky, "medium": lambda s, w, st: Waiter(s, w, st)},
                           walls=walls(off=5.0, medium=0.6, factor=2.0))
        paths, records = self.call(panel)
        medium = by_label(records[0])["a1-medium"]
        self.assertEqual((medium["stop_cause"], records[0]["winner"]),
                         ("member_budget", "a1-medium"))
        self.assertIsNone(records[0]["walls"]["early_cancel"])
        self.assert_no_scratch_left()

    def test_the_panel_wall_caps_every_member_and_the_winner_check(self):
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st, edit=False),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           walls=walls(off=30.0, medium=30.0, panel=0.4))
        with self.assertRaises(ActorTransient) as caught:
            self.call(panel)
        self.assertEqual({r["wall_budget_s"] for r in caught.exception.author_failures}, {0.4})
        self.assertIn("the panel's hard wall", caught.exception.author_failures[0]["reason"])

        # The winner check runs under the panel wall too (not the member's budget).
        def slow_check(hypothesis, workspace, base, paths, should_stop, context=None):
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if should_stop():
                    raise bestof.ValidationStopped("panel wall")
                time.sleep(0.01)
            raise AssertionError("the panel wall never reached the check")

        records = []
        panel = self.panel({"off": lambda s, w, st: tb.Editor(s, w, st),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           validator=slow_check, walls=walls(off=0.05, medium=30.0, panel=0.5))
        started = time.monotonic()
        paths, records = self.call(panel)
        self.assertLess(time.monotonic() - started, 5.0)
        row = records[0]
        self.assertTrue(row["walls"]["panel_wall_hit"])
        members = by_label(row)
        off = members["a0-off"]
        # a0-off authored inside its 0.05 s budget? No: the Editor returns at once, so
        # its budget never stopped it; its CHECK was ended by the panel wall.
        self.assertEqual((off["outcome"], off["validation"]["validator"], off["stop_cause"]),
                         ("diff", "stopped", "panel_wall"))
        self.assertFalse(off["budget_stopped"])
        self.assertEqual(members["a1-medium"]["stop_cause"], "panel_wall")
        self.assertEqual((row["selection"], row["winner"]), ("best_failing", "a0-off"))
        self.assert_no_scratch_left()

    def test_a_run_stop_is_still_a_stop(self):
        panel = self.panel({"off": lambda s, w, st: Waiter(s, w, st, edit=False),
                            "medium": lambda s, w, st: Waiter(s, w, st, edit=False)},
                           walls=walls(off=30.0, medium=30.0))
        import threading
        timer = threading.Timer(0.2, self.stop.set)
        timer.start()
        try:
            with self.assertRaises(ActorStopped):
                self.call(panel)
        finally:
            timer.cancel()
        self.assert_no_scratch_left()


class SinglePathUnchanged(tb.Fixture):

    def test_the_default_panel_budgets_do_not_touch_a_fast_round(self):
        # No walls given: the library default (the operator's budgets) applies and a
        # round of seconds is untouched by it.
        panel = self.panel({"off": tb.Editor, "medium": lambda s, w, st: tb.Blocker(s, w, st)})
        self.assertEqual(panel.walls, bestof.wall_budget())
        _paths, records = self.call(panel)
        members = by_label(records[0])
        self.assertEqual((members["a0-off"]["wall_budget_s"],
                          members["a1-medium"]["wall_budget_s"]), (2700, 4500))
        self.assertFalse(any(m["budget_stopped"] or m["early_cancelled"]
                             for m in members.values()))
        self.assert_no_scratch_left()

    def test_n1_never_builds_a_panel_and_the_single_author_keeps_its_retry(self):
        from autokernel.loop import run
        args = SimpleNamespace(actor_authors="medium", workers=1, actor_pool_tokens=196_608,
                               actor_authors_wall="off=60")
        with mock.patch.object(run.actor_opencode_config, "THINKING_CHOICES",
                               ("default", "off", "medium")):
            plan = run._author_plan(args, "opencode")
        self.assertFalse(plan.panel)
        self.assertIsNone(plan.walls)
        with self.assertRaisesRegex(ValueError, "N >= 2"):
            bestof.AuthorPanel(lane="l", specs=bestof.parse_authors("off"),
                               make_author=None, scratch=None,
                               budget=bestof.author_budget(1))
        # The single author's seat is built without the switch: timeouts retried.
        self.assertTrue(actors.AgentPlanner(workspace=Path(".")).retry_timeouts)


if __name__ == "__main__":
    unittest.main()
