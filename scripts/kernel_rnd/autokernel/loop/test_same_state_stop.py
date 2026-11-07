"""Identical-state abstention stop (same_state_stop.py) and the retry-budget shadow."""
import subprocess
from types import SimpleNamespace

import pytest

from . import (bench, campaign, gates, loop, pipeline, provider_breaker as pb,
               same_state_stop as ss, serial_scheduling)


def _git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "f.c").write_text("int x;\n")
    subprocess.run(["git", "-C", str(path), "add", "f.c"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "c0"], check=True)
    return path


class _Abstainer:
    def __init__(self):
        self.calls = 0
        self.backend = SimpleNamespace(describe=lambda: "codex:test")

    def propose(self, context):
        self.calls += 1
        return loop.Abstain("no admitted route carries a live mechanism")

    def author(self, hypothesis, context):  # pragma: no cover - never reached
        raise AssertionError


def _ctx(**extra):
    return {"program": "p", "inbox": [{"name": "10.md", "sha": "a"}],
            "prior_experiments": [{"status": "patch_rejected", "mechanism_id": "akm-x",
                                   "turn_recorded_at": "t0"}], **extra}


def _recorded(outcome, lane="lane0", attempt_id="att-1"):
    outcome.branch_id = f"detached:{lane}"
    outcome.journal_receipt = {"attempt_id": attempt_id}
    return outcome


def _iterate(planner, context):
    return loop.iterate(planner=planner, critic=None, context=context,
                        measure=lambda h, p: None, gate=lambda h, p: (True, []),
                        commit=lambda h, p, c: "x", record_abandoned=lambda c: None)


def test_second_identical_abstention_is_skipped_typed_and_parks(tmp_path):
    repo = _git_repo(tmp_path / "wt")
    guard = ss.SameStateGuard(1)
    inner = _Abstainer()
    planner = ss.GuardedPlanner(inner, guard, workspace=repo, lane="lane0")
    first = _iterate(planner, _ctx())
    assert first.status == "abstained" and inner.calls == 1
    assert not guard.observe(_recorded(first, attempt_id="att-7"))
    # The abstention itself is appended to history, and the health alarm moves:
    # neither is a state change.
    ctx = _ctx(loop_health_alarm={"n": 1})
    ctx["prior_experiments"].append({"status": "abstained", "reason": "same"})
    second = _iterate(planner, ctx)
    assert inner.calls == 1                       # planner NOT called
    assert second.status == ss.PLANNER_SAME_STATE_SKIP
    row = second.to_attempt()
    assert row["planner_same_state"]["prior_abstention_id"] == "att-7"
    assert row["planner_same_state"]["lane"] == "lane0"
    assert guard.observe(_recorded(second)) and guard.tripped
    assert not guard.observe(_recorded(second))   # trips once
    assert pb.terminal(stop_asked=False, breaker=pb.ProviderBreaker(),
                       same_state=guard) == ss.SAME_STATE_PARKED


def test_operator_stop_wins_over_same_state_park():
    guard = ss.SameStateGuard(1)
    guard.tripped = True
    assert pb.terminal(stop_asked=True, breaker=None, same_state=guard) == "stopped"
    assert ss.SAME_STATE_PARKED in pb.STOP_TERMINALS
    assert serial_scheduling.one_iteration_outcome(ss.SAME_STATE_PARKED, {}) == "failed"


@pytest.mark.parametrize("change", ["inbox", "row", "head"])
def test_any_real_state_change_asks_the_planner(tmp_path, change):
    repo = _git_repo(tmp_path / "wt")
    guard = ss.SameStateGuard(1)
    inner = _Abstainer()
    planner = ss.GuardedPlanner(inner, guard, workspace=repo, lane="lane0")
    _iterate(planner, _ctx())
    ctx = _ctx()
    if change == "inbox":
        ctx["inbox"].append({"name": "20.md", "sha": "b"})
    elif change == "row":
        ctx["prior_experiments"].append({"status": "measured_null", "mechanism_id": "akm-y"})
    else:
        (repo / "f.c").write_text("int y;\n")
        subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c",
                        "user.name=t", "commit", "-qam", "c1"], check=True)
    assert _iterate(planner, ctx).status == "abstained"
    assert inner.calls == 2


def test_threshold_counts_repeats_and_zero_disables(tmp_path):
    repo = _git_repo(tmp_path / "wt")
    for after, calls in ((2, 2), (0, 4)):
        guard = ss.SameStateGuard(after)
        inner = _Abstainer()
        planner = ss.GuardedPlanner(inner, guard, workspace=repo, lane="lane0")
        statuses = [_iterate(planner, _ctx()).status for _ in range(4)]
        assert inner.calls == calls, statuses


def test_a_proposal_clears_the_lane_and_lanes_are_independent(tmp_path):
    repo = _git_repo(tmp_path / "wt")
    guard = ss.SameStateGuard(1)
    guard.note_answer("lane0", "k", abstained=True)
    guard.check("lane1", "k")                       # other lane: no skip
    with pytest.raises(ss.PlannerSameStateSkip):
        guard.check("lane0", "k")
    guard.note_answer("lane0", "k", abstained=False)
    guard.check("lane0", "k")


def test_unreadable_worktree_never_skips(tmp_path):
    guard = ss.SameStateGuard(1)
    inner = _Abstainer()
    planner = ss.GuardedPlanner(inner, guard, workspace=tmp_path / "missing", lane="l")
    for _ in range(3):
        assert _iterate(planner, _ctx()).status == "abstained"
    assert inner.calls == 3


def test_skip_stops_dispatch_at_the_iteration_boundary():
    guard = ss.SameStateGuard(1)
    budget = pipeline.Budget(10, should_stop=lambda: guard.tripped)
    drawn = 0
    while budget.take():
        drawn += 1
        guard.observe(SimpleNamespace(status=ss.PLANNER_SAME_STATE_SKIP, branch_id="d:l"))
    assert drawn == 1 and budget.remaining == 9


def test_config_knob_and_env_override():
    def res(**extra):
        return {"schema": campaign.RESOURCE_SCHEMA, "cpu_logical": [0], "gpu_ids": [],
                "stage_timeout_s": 60, "build_timeout_s": 60, "build_jobs": 1,
                "max_builds": 1, **extra}
    default = campaign.ResourceRequest.from_dict(res())
    assert "planner_same_state_skip_after" not in default.to_dict()   # digests unchanged
    assert ss.resolve_skip_after(default.planner_same_state_skip_after, env={}) == 1
    off = campaign.ResourceRequest.from_dict(res(planner_same_state_skip_after=0))
    assert campaign.ResourceRequest.from_dict(off.to_dict()) == off
    assert ss.resolve_skip_after(0, env={}) == 0
    assert ss.resolve_skip_after(0, env={ss.ENV: "3"}) == 3
    assert ss.resolve_skip_after(None, env={ss.ENV: "0"}) == 0
    for bad in (-1, True, "1", 1.5):
        with pytest.raises(campaign.ManifestError):
            campaign.ResourceRequest.from_dict(res(planner_same_state_skip_after=bad))
    with pytest.raises(ValueError):
        ss.resolve_skip_after(None, env={ss.ENV: "x"})


def test_retry_budget_shadow_annotates_only_at_threshold(tmp_path):
    from ..controller import experiments
    store = tmp_path / "store"
    store.mkdir()
    attempt = {"mechanism_id": "akm-x"}
    with experiments.ExperimentStore(store) as handle:
        for index in range(4):
            assert ss.retry_budget_shadow(store, attempt) is None
            handle.record({"status": "patch_rejected", "mechanism_id": "akm-x",
                           "attempt_id": f"a{index}", "reason": f"r{index}"},
                          epoch="e" * 64, recorded_at=f"2026-11-07T00:00:0{index}Z",
                          campaign_id="t")
    shadow = ss.retry_budget_shadow(store, attempt)
    assert shadow == {"mode": "log_only", "mechanism_id": "akm-x",
                      "prior_patch_rejected": 4, "threshold": 4, "would_refuse": True}
    assert ss.retry_budget_shadow(store, {"mechanism_id": "akm-other"}) is None
    assert ss.retry_budget_shadow(store, {}) is None
    assert ss.retry_budget_shadow(tmp_path / "nope" / "x", attempt) is None
