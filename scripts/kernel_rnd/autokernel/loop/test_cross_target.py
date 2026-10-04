"""Per-target champion lineage (cross_target.py) -- temp git repos, injected measurement.

The scenario is the DS41 / Q38FN pair. A DS41 keep that also helps (or is neutral on)
Q38FN goes to the trunk, and Q38FN picks it up at its next advancement. A DS41 keep that
regresses Q38FN (b3e0b0902's -29..-41% prefill class) stays on DS41's branch only. It is
re-checked on Q38FN at Q38FN's advancements: it propagates if it helps, or if it is
neutral and simplifies. Otherwise it stays.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from . import cross_target as ct, lane_targets


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _write(path: Path, body) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path):
    """One repo; trunk + two lane branches at a common champion; a champion worktree
    per lane (branch checked out) and a worker worktree for q38's rechecks."""
    repo = tmp_path / "llama"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    (repo / "ggml").mkdir()
    (repo / "ggml" / "ops.cpp").write_text("a\nb\nc\nd\ne\nf\ng\nh\n")
    (repo / "src").mkdir()
    (repo / "src" / "deepseek.cpp").write_text("ds\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "champion")
    _git(repo, "branch", "trunk")      # a bare ref: never checked out in a deployment
    _git(repo, "branch", "ds41-champ")
    _git(repo, "branch", "q38-champ")
    trees = {}
    for name, branch in (("ds41", "ds41-champ"), ("q38", "q38-champ")):
        trees[name] = tmp_path / f"champ-{name}"
        _git(repo, "worktree", "add", "-q", str(trees[name]), branch)
    worker = tmp_path / "worker-q38"
    _git(repo, "worktree", "add", "-q", "--detach", str(worker), "q38-champ")
    owned0 = _write(tmp_path / "ds41" / "owned.json", {"ds41-t": {
        "launch": str(tmp_path / "l0.json"), "frozen_prompts": str(tmp_path / "p0.json")}})
    owned1 = _write(tmp_path / "q38" / "owned.json", {"q38-t": {
        "launch": str(tmp_path / "l1.json"), "frozen_prompts": str(tmp_path / "p1.json")}})
    binding = _write(tmp_path / "bind" / "lane-targets.json", {
        "schema": lane_targets.SCHEMA, "trunk": {"branch": "trunk"}, "lanes": {
            "lane0": {"target_id": "ds41-t", "owned_targets": str(owned0),
                      "exclusive_paths": ["src/deepseek*.cpp"], "as_peer_floor_file": None,
                      "as_peer_max_regression_pct": 2.0, "champion_branch": "ds41-champ"},
            "lane1": {"target_id": "q38-t", "owned_targets": str(owned1),
                      "exclusive_paths": ["src/qwen*.cpp"], "as_peer_floor_file": None,
                      "as_peer_max_regression_pct": 2.0, "champion_branch": "q38-champ"}}})
    ds41 = lane_targets.resolve(binding, "lane0", target_id="ds41-t", workers=1, lock=False,
                                champion_branch="ds41-champ")
    q38 = lane_targets.resolve(binding, "lane1", target_id="q38-t", workers=1, lock=False,
                               champion_branch="q38-champ")
    return {"repo": repo, "trees": trees, "worker": worker, "ds41": ds41, "q38": q38}


def _keep(tree: Path, rel: str, text: str, message: str) -> str:
    (tree / rel).write_text(text)
    _git(tree, "commit", "-q", "-am", message)
    return _git(tree, "rev-parse", "HEAD")


def _cross(passed=True, effect=0.0, decisive=False, required=True):
    return {"required": required, "passed": passed, "peers": [] if not required else [
        {"lane": "lane1", "target_id": "q38-t", "passed": passed, "effect": effect,
         "decisive": decisive, "reason": "peer regressed" if not passed else "ok"}]}


UNCHANGED = {"q38-t": {"observed": True, "unchanged": True, "losses": [], "changed": []}}


# ------------------------------------------------------------------ pure decisions

def test_decide():
    assert ct.decide(_cross(required=False), {})["decision"] == ct.SHARED
    assert ct.decide(_cross(), UNCHANGED)["decision"] == ct.SHARED
    regressed = ct.decide(_cross(passed=False, effect=-0.3, decisive=True), UNCHANGED)
    assert regressed["decision"] == ct.TARGET_ONLY and "regressed" in regressed["reason"]
    lost = {"q38-t": {"observed": True, "losses": [{"key": "iqk.gemm:Q8_0:act=Q8_0"}],
                      "changed": ["s:iqk.gemm:Q8_0:act=Q8_0"]}}
    assert ct.decide(_cross(effect=0.2, decisive=True), lost)["decision"] == ct.TARGET_ONLY
    changed = {"q38-t": {"observed": True, "losses": [], "changed": ["s:iqk.gemm:Q4_K"]}}
    assert ct.decide(_cross(effect=0.001), changed)["decision"] == ct.TARGET_ONLY
    assert ct.decide(_cross(effect=0.05, decisive=True), changed)["decision"] == ct.SHARED
    assert ct.decide(_cross(), {})["decision"] == ct.TARGET_ONLY      # coverage unobserved


@pytest.mark.parametrize("row,simpler,expected", [
    ({"effect": 0.04, "decisive": True}, False, ("propagate", "helps")),
    ({"effect": -0.04, "decisive": True}, True, ("stay", "regressed")),
    ({"effect": 0.001, "decisive": False}, True, ("propagate", "neutral_simplifies")),
    ({"effect": 0.001, "decisive": False}, False, ("stay", "neutral_not_simpler")),
    (None, True, ("stay", "failed")),
])
def test_recheck_decision(row, simpler, expected):
    out = ct.recheck_decision(row, simplifies=simpler)
    assert (out["result"], out["why"]) == expected


# ------------------------------------------------------------------ lineage moves

def test_shared_keep_goes_to_trunk_and_syncs_into_the_peer(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k1")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m-shared")
    assert row["decision"] == ct.SHARED and row["trunk"]["result"] == "applied"
    assert _git(repo, "show", "trunk:ggml/ops.cpp").startswith("A\n")
    synced = ct.sync_from_trunk(q38, repo=repo, champion_tree=world["trees"]["q38"],
                                branch="q38-champ")
    assert [r["applied"] for r in synced] == [True]
    assert (world["trees"]["q38"] / "ggml/ops.cpp").read_text().startswith("A\n")
    # Idempotent: nothing more to bring over, and DS41's own branch sees it as present.
    assert ct.sync_from_trunk(q38, repo=repo, champion_tree=world["trees"]["q38"],
                              branch="q38-champ") == []
    assert ct.sync_from_trunk(ds41, repo=repo, champion_tree=world["trees"]["ds41"],
                              branch="ds41-champ") == []
    events = [r["event"] for r in ct.read(ds41)]
    assert events == ["keep_decision", "trunk_sync_commit"]


def test_target_only_keep_stays_off_trunk_and_rechecks_on_the_peer(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    trunk_before = _git(repo, "rev-parse", "trunk")
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "a\nb\nc\nd\ne\nf\ng\nH\n", "k2")
    cross = _cross(passed=False, effect=-0.35, decisive=True)
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep, decision=ct.decide(cross, UNCHANGED),
                         cross=cross, coverage=UNCHANGED, mechanism_id="m-ds41-only")
    assert row["decision"] == ct.TARGET_ONLY and "trunk" not in row
    assert _git(repo, "rev-parse", "trunk") == trunk_before
    head = _git(repo, "rev-parse", "q38-champ")
    assert [p["keep_commit"] for p in ct.pending(q38, repo=repo, head=head)] == [keep]
    assert ct.pending(ds41, repo=repo, head=_git(repo, "rev-parse", "ds41-champ")) == []

    def measure_regressed(entry):
        commit = ct.apply_in_worktree(world["worker"], entry["keep_commit"])
        return {"row": {"effect": -0.30, "decisive": True}, "candidate_commit": commit}

    advanced = []
    event = ct.recheck_one(q38, repo=repo, champion_tree=world["trees"]["q38"],
                           branch="q38-champ", measure=measure_regressed,
                           advance=advanced.append)
    assert (event["result"], event["why"], event["applied"]) == ("stay", "regressed", False)
    assert advanced == [] and _git(repo, "rev-parse", "q38-champ") == head
    # Re-checked once per champion head: nothing pending until q38 advances.
    assert ct.recheck_one(q38, repo=repo, champion_tree=world["trees"]["q38"],
                          branch="q38-champ", measure=measure_regressed,
                          advance=advanced.append) is None
    _keep(world["trees"]["q38"], "ggml/ops.cpp", "a\nB\nc\nd\ne\nf\ng\nh\n", "q38 own keep")
    _git(world["worker"], "checkout", "-q", "--detach", "q38-champ")

    def measure_helps(entry):
        commit = ct.apply_in_worktree(world["worker"], entry["keep_commit"])
        return {"row": {"effect": 0.06, "decisive": True}, "candidate_commit": commit}

    event = ct.recheck_one(q38, repo=repo, champion_tree=world["trees"]["q38"],
                           branch="q38-champ", measure=measure_helps, advance=advanced.append)
    assert (event["result"], event["why"], event["applied"]) == ("propagate", "helps", True)
    assert advanced == [event["new_head"]] == [_git(repo, "rev-parse", "q38-champ")]
    assert (world["trees"]["q38"] / "ggml/ops.cpp").read_text() == "a\nB\nc\nd\ne\nf\ng\nH\n"
    # Both targets carry it now, so it went onto the trunk too.
    assert event["trunk"]["result"] == "applied"
    assert _git(repo, "show", "trunk:ggml/ops.cpp").endswith("H")   # _git strips
    assert ct.pending(q38, repo=repo, head=_git(repo, "rev-parse", "q38-champ")) == []


def test_recheck_failure_is_recorded_not_raised(world):
    repo, ds41, q38 = world["repo"], world["ds41"], world["q38"]
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "a\nb\nC\nd\ne\nf\ng\nh\n", "k3")
    cross = _cross(passed=False, effect=-0.2, decisive=True)
    ct.record_keep(ds41, repo=repo, keep_commit=keep, decision=ct.decide(cross, UNCHANGED),
                   cross=cross, coverage=UNCHANGED, mechanism_id="m3")

    def boom(_entry):
        raise RuntimeError("build failed")

    event = ct.recheck_one(q38, repo=repo, champion_tree=world["trees"]["q38"],
                           branch="q38-champ", measure=boom, advance=lambda _c: None)
    assert (event["result"], event["why"]) == ("stay", "failed")
    assert "build failed" in event["reason"]


def test_simplifies_and_patch_id(world):
    repo, tree = world["repo"], world["trees"]["ds41"]
    shrink = _keep(tree, "ggml/ops.cpp", "a\n", "shrink")
    grow = _keep(tree, "ggml/ops.cpp", "a\nb\nc\n", "grow")
    assert ct.simplifies(repo, shrink) and not ct.simplifies(repo, grow)
    assert len(ct.patch_id(repo, grow)) == 40


def test_conflicting_trunk_pick_is_a_recorded_miss(world):
    repo, ds41 = world["repo"], world["ds41"]
    # The trunk moved on the same line independently: the cherry-pick conflicts.
    _git(repo, "worktree", "add", "-q", str(repo.parent / "trunk-tree"), "trunk")
    _keep(repo.parent / "trunk-tree", "ggml/ops.cpp", "Z\nb\nc\nd\ne\nf\ng\nh\n", "trunk edit")
    keep = _keep(world["trees"]["ds41"], "ggml/ops.cpp", "A\nb\nc\nd\ne\nf\ng\nh\n", "k4")
    row = ct.record_keep(ds41, repo=repo, keep_commit=keep,
                         decision=ct.decide(_cross(), UNCHANGED), cross=_cross(),
                         coverage=UNCHANGED, mechanism_id="m4")
    assert row["decision"] == ct.SHARED
    assert row["trunk"] == {"branch": "trunk", "commit": None, "result": "conflict"}


# ------------------------------------------------------------------ binding

def test_trunk_binding_requires_distinct_lane_branches(tmp_path, world):
    path = world["ds41"].path
    body = json.loads(path.read_text())
    body["lanes"]["lane1"]["champion_branch"] = "ds41-champ"
    bad = _write(tmp_path / "bad.json", body)
    with pytest.raises(lane_targets.LaneBindingError, match="distinct"):
        lane_targets.load_binding(bad)
    body["lanes"]["lane1"]["champion_branch"] = "trunk"
    with pytest.raises(lane_targets.LaneBindingError, match="distinct"):
        lane_targets.load_binding(_write(tmp_path / "bad2.json", body))
    with pytest.raises(lane_targets.LaneBindingError, match="not --champion-branch"):
        lane_targets.resolve(path, "lane0", target_id="ds41-t", workers=1, lock=False,
                             champion_branch="other")
    assert lane_targets.load_binding(path)[1] == "trunk"
