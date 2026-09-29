"""Per-lane actor models (`--lane-actor-models`, operator 2026-09-29) -- fakes only.

No actor process, provider, server or hardware: the two-lane run drives `run.main`
through the keep fixture of `test_promotion_targets` with fake planners/critics, and
the external model is only ever a string on a recorded `Backend`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from . import actors, ak_check, cpu_window, lane_actors, pipeline, pool, run, serial_run
from .test_promotion_targets import TheKeepBuildsAProductionCompleteAnchor

QWEN = "qwen-gpu/qwen3.8-27b"
DEEPSEEK = "deepseek/deepseek-flash"


# ------------------------------------------------------------------ parsing

def test_parse_accepts_lanes_above_zero_with_optional_effort():
    assert lane_actors.parse("", workers=2) == {}
    assert lane_actors.parse(None, workers=1) == {}
    lanes = lane_actors.parse(f"1={DEEPSEEK}, 2={DEEPSEEK}@max", workers=3)
    assert lanes == {1: lane_actors.LaneActor(1, DEEPSEEK, None),
                     2: lane_actors.LaneActor(2, DEEPSEEK, "max")}
    assert lanes[1].effort_or("high") == "high" and lanes[2].effort_or("high") == "max"


@pytest.mark.parametrize("spec,match", [
    (f"0={DEEPSEEK}", "lane 0"),
    (f"2={DEEPSEEK}", "does not exist"),
    (f"1={DEEPSEEK},1={DEEPSEEK}", "named twice"),
    ("1=gpt-5.6-sol", "provider/model"),
    ("1=claude-fable-5", "provider/model"),
    ("1=orch:auto", "provider/model"),
    (f"1={DEEPSEEK}@", "empty effort"),
    (f"x={DEEPSEEK}", "K=provider/model"),
    ("1=", "K=provider/model"),
])
def test_parse_refuses_what_cannot_run_as_written(spec, match):
    with pytest.raises(ValueError, match=match):
        lane_actors.parse(spec, workers=2)


def test_lane_index_shared_pool_and_provenance():
    assert lane_actors.lane_index(SimpleNamespace(name="lane1")) == 1
    assert lane_actors.lane_index(SimpleNamespace(name="member-x")) is None
    lanes = lane_actors.parse(f"1={DEEPSEEK}", workers=2)
    assert lane_actors.shared_pool_lanes(2, {}, QWEN) == 2
    assert lane_actors.shared_pool_lanes(2, lanes, QWEN) == 1
    # Another model on the SAME server still shares its pool.
    assert lane_actors.shared_pool_lanes(
        2, lane_actors.parse("1=qwen-gpu/other", workers=2), QWEN) == 2
    assert lane_actors.provenance({}, "high") == {}
    assert lane_actors.provenance(lanes, "high") == {
        "lane_actor_models": {"1": {"model": DEEPSEEK, "effort": "high"}}}


def test_overridden_seat_drops_only_the_qwen_template_reasoning_kwargs():
    seat = actors.ActorSeat(author_thinking="medium", context_limit=180224,
                            author_output_limit=40960, planner_salvage_s=900,
                            author_sandbox=True)
    assert lane_actors.seat_for(None, seat) is seat
    lane = lane_actors.LaneActor(1, DEEPSEEK)
    moved = lane_actors.seat_for(lane, seat)
    assert moved.thinking_for("author") == "default"
    assert moved.limits_for("author") == seat.limits_for("author")
    assert (moved.planner_salvage_s, moved.author_sandbox) == (900, True)
    # The per-call config carries no chat_template_kwargs for the external model.
    from . import actor_opencode_config as seat_config
    config = seat_config.build_plain_config(
        role="author", lane=Path("/lanes/lane1"), model=DEEPSEEK,
        thinking=moved.thinking_for("author"), **moved.limits_for("author"))
    assert "options" not in json.dumps(config)
    assert config["provider"]["deepseek"]["models"]["deepseek-flash"]["limit"] == {
        "context": 180224, "output": 40960}


# ------------------------------------------------------------------ run.py seams

def _plan_args(**overrides):
    base = dict(actor_authors=None, workers=2, actor_pool_tokens=196_608,
                planner_model=QWEN, lane_actor_models="")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_best_of_panel_survives_two_lanes_only_when_lane_one_is_off_the_pool():
    degraded = run._author_plan(_plan_args(), "opencode")
    assert not degraded.panel and "unified pool" in degraded.note
    kept = run._author_plan(_plan_args(lane_actor_models=f"1={DEEPSEEK}"), "opencode")
    assert kept.panel and kept.note.startswith("best-of-2")
    # The panel budget holds exactly one lane's authors, as with --workers 1.
    single_lane = run._author_plan(_plan_args(workers=1), "opencode")
    assert kept.budget == single_lane.budget
    with pytest.raises(ValueError, match="unified pool"):
        run._author_plan(_plan_args(actor_authors="off,medium"), "opencode")


def test_actor_config_records_lane_models_only_when_set():
    plain = run._actor_config(argparse.Namespace(planner_model=QWEN, planner_effort="high"))
    assert "lane_actor_models" not in plain
    lanes = run._actor_config(argparse.Namespace(
        planner_model=QWEN, planner_effort="high", workers=2,
        lane_actor_models=f"1={DEEPSEEK}"))
    assert lanes["lane_actor_models"] == {"1": {"model": DEEPSEEK, "effort": "high"}}
    assert {key: value for key, value in lanes.items() if key != "lane_actor_models"} == plain


def test_lane_models_are_neither_continuation_nor_epoch_identity(tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text("{}")
    base = ["--worktree", "/w", "--cpu-serving-launch", str(launch), "--workers", "1",
            "--planner-model", QWEN]
    two = [*base, "--workers", "2", "--lane-actor-models", f"1={DEEPSEEK}"]
    assert serial_run.resume_binding(two) == serial_run.resume_binding(base)
    # The measurement epoch takes epoch inputs, never argv (OP-60): nothing to fold.
    inputs = {"cpu_execution_digest": "e" * 64, "frozen_prompt_digest": "f" * 64}
    assert run.measurement_epoch_inputs(inputs) == inputs


def test_serial_common_args_admit_the_lane_flag(tmp_path):
    from .test_serial_roster import _inputs
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    common = tmp_path / "common.json"
    common.write_text(json.dumps(["--workers", "2", "--lane-actor-models", f"1={DEEPSEEK}"]))
    from . import serial_roster
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(serial_run.option(argv, "--resolved-campaign")),
        Path(serial_run.option(argv, "--owned-targets")),
        target_root=tmp_path / "targets", common_path=common)
    assert serial_run.option(targets[0], "--lane-actor-models") == f"1={DEEPSEEK}"


# ------------------------------------------------------------------ the check fence

def test_checks_take_the_pools_shared_fence_for_the_run_only(tmp_path, monkeypatch):
    monkeypatch.delenv(ak_check.ENV_FENCE, raising=False)
    member = tmp_path / "scratch" / "author" / "tree"
    assert ak_check.check_fence_dir(member) == member.parent / ak_check.FENCE_DIR_NAME
    seen = []

    def fake_run_pool(**kwargs):
        seen.append(os.environ.get(ak_check.ENV_FENCE))
        return []

    workers = [pipeline.Worker(f"lane{index}", tmp_path / "workers" / f"lane{index}",
                               tmp_path / "builds" / f"lane{index}") for index in range(2)]
    common = dict(workers=workers, make_planner=None, make_critic=None, build_context=dict,
                  make_gate=None, make_measure=None, record=lambda _o: None, iterations=2,
                  reset=lambda _w: "0" * 40, commit=lambda *_a: "0" * 40)
    with mock.patch.object(pool.pipeline, "run_pool", fake_run_pool):
        pool.drive(**common)
        pool.drive(**common, author_sandbox=True)
    shared = tmp_path / "workers" / ak_check.FENCE_DIR_NAME
    assert seen == [None, str(shared)]
    assert ak_check.ENV_FENCE not in os.environ
    # During the pool, a member tree's check fences where the tail waits.
    assert ak_check.check_fence_dir(member, {ak_check.ENV_FENCE: str(shared)}) == shared
    # Lanes under different parents: no single fence names them all -> unchanged.
    with ak_check.shared_fence_env([tmp_path / "a", tmp_path / "b"]):
        assert ak_check.ENV_FENCE not in os.environ


def test_a_member_check_is_refused_while_the_tail_holds_the_shared_fence(tmp_path):
    shared = tmp_path / "workers" / ak_check.FENCE_DIR_NAME
    with ak_check.tail_fence([shared]):
        with pytest.raises(ak_check.Refused):
            with ak_check.sandbox_slot(ak_check.check_fence_dir(
                    tmp_path / "scratch" / "tree", {ak_check.ENV_FENCE: str(shared)})):
                pass


# ------------------------------------------------------------------ the CPU window

def _mixed_log(tmp_path: Path) -> Path:
    path = tmp_path / "actor-calls.jsonl"
    rows = ([{"role": "planner", "wall_s": 3000, "backend": {"model": QWEN}}] * 3
            + [{"role": "planner", "wall_s": 300, "backend": {"model": DEEPSEEK}}] * 3
            + [{"role": "critic", "wall_s": 100, "backend": {"model": DEEPSEEK}}])
    path.write_text("".join(json.dumps({"schema": "epyc.autokernel.actor_call.v1", **row}) + "\n"
                            for row in rows))
    return path


def test_phase_estimates_key_on_the_lanes_model_when_lanes_differ(tmp_path):
    estimator = cpu_window.PhaseEstimator(_mixed_log(tmp_path), planner_budget_s=4500,
                                          author_budget_s=4500, critic_timeout_s=7200)
    pooled = estimator.estimate(("planner",))[1]["planner"]
    assert pooled["samples"] == 6 and pooled["median_wall_s"] == 1650
    local = estimator.estimate(("planner",), models={"planner": QWEN})[1]["planner"]
    external = estimator.estimate(("planner",), models={"planner": DEEPSEEK})[1]["planner"]
    assert (local["median_wall_s"], external["median_wall_s"]) == (3000, 300)
    # A model with no rows yet falls back to the pooled median, never to nothing.
    fresh = estimator.estimate(("planner",), models={"planner": "x/new"})[1]["planner"]
    assert fresh["median_wall_s"] == 1650


def test_the_window_closes_when_the_first_of_several_lanes_needs_the_cpu(tmp_path):
    now = [1_000_000.0]
    window = cpu_window.CpuWindow(
        campaign="ak-test", path=tmp_path / "global" / "cpu-window.json",
        estimator=cpu_window.PhaseEstimator(_mixed_log(tmp_path), planner_budget_s=4500,
                                            author_budget_s=4500, critic_timeout_s=7200),
        heartbeat_s=None, clock=lambda: now[0], log=lambda _text: None,
        lane_models={"lane0": {"planner": QWEN, "author": QWEN, "critic": DEEPSEEK},
                     "lane1": {"planner": DEEPSEEK, "author": DEEPSEEK, "critic": DEEPSEEK}})
    window.lease = SimpleNamespace(held=False, generation=1, release=lambda **_kw: None)
    published = []
    window._publish = lambda **fields: published.append(fields)
    window.note_step("lane0", "authoring the patch")
    alone = published[-1]
    assert "lanes" not in alone["est_close_basis"]
    now[0] += 10
    window.note_step("lane1", "proposing a hypothesis")
    both = published[-1]
    # lane0's author+critic2 ends before lane1's planner..critic2 chain.
    assert both["est_close_at"] == alone["est_close_at"]
    assert set(both["est_close_basis"]["lanes"]) == {"lane0", "lane1"}
    window.note_step("lane0", "critic pass 2: reviewing the diff")
    assert published[-1]["state"] == "closing"


# ------------------------------------------------------------------ the two-lane run

class TwoLaneRun(TheKeepBuildsAProductionCompleteAnchor):
    """`run.main` with two pooled lanes; lane 1 on the external model."""

    def runTest(self):  # pragma: no cover - driven explicitly below
        pass

    def two_lanes(self, extra):
        recorded = []
        original_main = run.main

        def selected_main(argv):
            argv = list(argv)
            argv[argv.index("--workers") + 1] = "2"
            argv[argv.index("--iterations") + 1] = "2"
            argv += ["--planner-model", QWEN, "--planner-effort", "high",
                     "--critic-model", DEEPSEEK, "--critic-effort", "max",
                     "--actor-authors", "single", *extra]
            installed = run.actors.AgentPlanner
            installed_provision = run.pool.provision

            def planner(workspace, **kwargs):
                recorded.append((Path(workspace).name, kwargs))
                return installed(workspace, **kwargs)

            def provision(count, **kwargs):
                assert count == 2
                first = installed_provision(count, **kwargs)[0]
                lane, build = self.root / "lane1", self.root / "lane1-build"
                if not (lane / ".git").exists():
                    subprocess.run(["git", "-C", str(self.repo), "worktree", "add", "--detach",
                                    str(lane), self.tip], capture_output=True, check=True,
                                   timeout=60)
                build.mkdir(exist_ok=True)
                return [first, pipeline.Worker("lane1", lane, build)]

            with mock.patch.object(run.actors, "AgentPlanner", planner), \
                    mock.patch.object(run.pool, "provision", provision):
                return original_main(argv)

        with mock.patch.object(run, "main", selected_main):
            rc, _calls, _planners, _scratch, log = self._run_one_keep()
        return rc, recorded, log


def test_two_lane_run_puts_lane_one_on_the_external_model_and_leaves_lane_zero_alone():
    runs = {}
    for label, extra in (("global", []), ("lanes", ["--lane-actor-models", f"1={DEEPSEEK}"])):
        fixture = TwoLaneRun()
        fixture.setUp()
        try:
            rc, recorded, log = fixture.two_lanes(extra)
        finally:
            fixture.doCleanups()
        assert rc == 0, log
        runs[label] = (dict(recorded), log)
    lanes, log = runs["lanes"]
    assert set(lanes) == {"lane0", "lane1"}
    assert lanes["lane1"]["backend"] == actors.Backend("opencode", DEEPSEEK, "high",
                                                       actors.OPENCODE)
    assert lanes["lane1"]["seat"].thinking_for("author") == "default"
    assert f"lane1 planner+author=opencode:{DEEPSEEK}@high" in log
    # Lane 0 is constructed exactly as in the same run without the flag.
    global_lanes, _ = runs["global"]
    assert lanes["lane0"]["backend"] == global_lanes["lane0"]["backend"]
    assert lanes["lane0"]["backend"].model == QWEN
    assert lanes["lane0"]["seat"] == global_lanes["lane0"]["seat"]
    assert lanes["lane0"]["seat"].thinking_for("author") == "medium"
    assert {key: value for key, value in lanes["lane0"].items()
            if key not in {"should_stop", "sandbox_scratch"}} == \
        {key: value for key, value in global_lanes["lane0"].items()
         if key not in {"should_stop", "sandbox_scratch"}}
    # Without the flag both lanes are the global planner (the historical pool).
    assert global_lanes["lane1"]["backend"].model == QWEN
