"""Lane -> target binding (`--lane-targets`/`--lane`, operator 2026-10-03) -- fakes only.

No server, build or hardware: the cross-target comparison is injected, exactly as the
loop injects `serving.compare` around its own builds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import lane_targets, serial_run


def _write(path: Path, body) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _binding(tmp_path: Path, **lane1_overrides) -> Path:
    owned0 = _write(tmp_path / "ds41" / "owned-targets.json", {"ds41-t": {
        "launch": str(tmp_path / "ds41" / "launch.json"),
        "frozen_prompts": str(tmp_path / "ds41" / "prompts.json"),
        "store": str(tmp_path / "ds41" / "store")}})
    owned1 = _write(tmp_path / "q38" / "owned-targets.json", {"q38-t": {
        "launch": str(tmp_path / "q38" / "launch.json"),
        "frozen_prompts": str(tmp_path / "q38" / "prompts.json")}})
    floor = _write(tmp_path / "ds41" / "floor.json", {"unit": "process", "floor_pct": 3.412})
    lane1 = {"target_id": "q38-t", "owned_targets": str(owned1),
             "exclusive_paths": ["src/models/qwen4exp*.cpp"],
             "as_peer_floor_file": None, "as_peer_max_regression_pct": 2.0,
             "cpu_window_path": str(tmp_path / "cpu-window-q38.json")}
    lane1.update(lane1_overrides)
    return _write(tmp_path / "lane-targets.json", {"schema": lane_targets.SCHEMA, "lanes": {
        "lane0": {"target_id": "ds41-t", "owned_targets": str(owned0),
                  "exclusive_paths": ["src/models/deepseek*.cpp"],
                  "as_peer_floor_file": str(floor),
                  "as_peer_max_regression_pct": None},
        "lane1": lane1}})


# ------------------------------------------------------------------ binding

def test_resolve_binds_the_lane_and_its_peer(tmp_path):
    bound = lane_targets.resolve(_binding(tmp_path), "lane1", target_id="q38-t", workers=1)
    try:
        assert bound.lane.target_id == "q38-t"
        assert bound.lane.cpu_window_path == tmp_path / "cpu-window-q38.json"
        (peer,) = bound.peers
        assert peer.entry.name == "lane0" and peer.entry.target_id == "ds41-t"
        assert peer.launch_path == tmp_path / "ds41" / "launch.json"
        assert peer.store == tmp_path / "ds41" / "store"
    finally:
        bound.release()


@pytest.mark.parametrize("lane,target,workers,needle", [
    ("lane1", "ds41-t", 1, "bound to q38-t"),       # target on the wrong lane
    ("lane2", "q38-t", 1, "not bound"),
    ("lane1", "q38-t", 2, "ONE worker"),
])
def test_resolve_refuses_what_cannot_run_as_bound(tmp_path, lane, target, workers, needle):
    with pytest.raises(lane_targets.LaneBindingError, match=needle):
        lane_targets.resolve(_binding(tmp_path), lane, target_id=target, workers=workers)


def test_one_lane_one_instance(tmp_path):
    path = _binding(tmp_path)
    first = lane_targets.resolve(path, "lane1", target_id="q38-t", workers=1)
    try:
        with pytest.raises(lane_targets.LaneBindingError, match="already has a running"):
            lane_targets.resolve(path, "lane1", target_id="q38-t", workers=1)
        # The other lane is a different instance; a dry run takes no lock.
        lane_targets.resolve(path, "lane0", target_id="ds41-t", workers=1).release()
        lane_targets.resolve(path, "lane1", target_id="q38-t", workers=1, lock=False)
    finally:
        first.release()
    lane_targets.resolve(path, "lane1", target_id="q38-t", workers=1).release()


@pytest.mark.parametrize("override,needle", [
    ({"exclusive_paths": ["ggml/src/ggml-cpu/*"]}, "shared by every target"),
    ({"as_peer_max_regression_pct": 0}, r"\(0, 20\]"),
    ({"owned_targets": "relative.json"}, "absolute"),
    ({"target_id": "ds41-t"}, "two lanes"),
    ({"unknown": 1}, "expected fields"),
])
def test_load_refuses_malformed_entries(tmp_path, override, needle):
    with pytest.raises(lane_targets.LaneBindingError, match=needle):
        lane_targets.load(_binding(tmp_path, **override))


def test_needs_cross_check_only_outside_exclusive_paths():
    own = ("src/models/qwen4exp*.cpp",)
    assert not lane_targets.needs_cross_check(["src/models/qwen4exp.cpp"], own)
    assert lane_targets.needs_cross_check(["src/models/qwen4exp.cpp",
                                           "ggml/src/ggml-cpu/ops.cpp"], own)
    assert lane_targets.needs_cross_check([], own)      # cannot prove exclusivity


# ------------------------------------------------------------------ verdicts

def test_decide_floor_and_point_bars():
    floor = {"mode": "floor", "floor_pct": 3.0}
    assert lane_targets.decide({"effect": -0.05, "decisive": True}, floor)[0] is False
    assert lane_targets.decide({"effect": -0.02, "decisive": False}, floor)[0] is True
    assert lane_targets.decide({"effect": 0.05, "decisive": True}, floor)[0] is True
    point = {"mode": "point", "max_regression_pct": 2.0}
    assert lane_targets.decide({"effect": -0.021}, point)[0] is False
    assert lane_targets.decide({"effect": -0.019}, point)[0] is True
    assert lane_targets.decide(None, point)[0] is False
    assert lane_targets.decide({"effect": 0.1}, {"mode": "absent"})[0] is False


def test_cross_check_records_and_vetoes_a_shared_regression(tmp_path):
    bound = lane_targets.resolve(_binding(tmp_path), "lane1", target_id="q38-t", workers=1,
                                 lock=False)
    seen = []

    def compare(peer, bar):
        seen.append((peer.entry.name, bar["mode"]))
        kwargs = lane_targets.compare_kwargs(bar, None, None) if bar["mode"] != "floor" else {}
        assert kwargs in ({}, {"floor_pct": None})
        return {"effect": -0.06, "decisive": True, "pairs": 5}

    store = tmp_path / "store"
    verdict = lane_targets.cross_check(bound, changed=["ggml/src/ggml-cpu/ops.cpp"],
                                       compare=compare, store=store, mechanism_id="akm-x",
                                       now=lambda: 1.0)
    assert seen == [("lane0", "floor")]
    assert verdict["required"] is True and verdict["passed"] is False
    assert "3.412% matched floor" in verdict["reason"]
    (record,) = (store / lane_targets.CHECK_DIR).iterdir()
    assert json.loads(record.read_text())["peers"][0]["effect"] == -0.06


def test_cross_check_skips_exclusive_keeps_and_fails_closed_on_errors(tmp_path):
    bound = lane_targets.resolve(_binding(tmp_path), "lane1", target_id="q38-t", workers=1,
                                 lock=False)

    def never(peer, bar):
        raise AssertionError("an exclusive keep needs no peer measurement")

    ok = lane_targets.cross_check(bound, changed=["src/models/qwen4exp.cpp"], compare=never,
                                  store=tmp_path / "s", mechanism_id="akm-own", now=lambda: 1.0)
    assert ok["required"] is False and ok["passed"] is True

    def broken(peer, bar):
        raise RuntimeError("server died")

    bad = lane_targets.cross_check(bound, changed=["ggml/x.c"], compare=broken,
                                   store=tmp_path / "s", mechanism_id="akm-bad", now=lambda: 2.0)
    assert bad["passed"] is False and "server died" in bad["reason"]


def test_peer_without_a_bar_refuses_shared_keeps(tmp_path):
    path = _binding(tmp_path)
    body = json.loads(path.read_text())
    body["lanes"]["lane0"]["as_peer_floor_file"] = None
    path.write_text(json.dumps(body), encoding="utf-8")
    bound = lane_targets.resolve(path, "lane1", target_id="q38-t", workers=1, lock=False)
    verdict = lane_targets.cross_check(
        bound, changed=["ggml/x.c"], store=tmp_path / "s", mechanism_id="akm-nobar",
        compare=lambda peer, bar: pytest.fail("no bar, no measurement"), now=lambda: 1.0)
    assert verdict["passed"] is False and "no cross-check bar" in verdict["reason"]


# ------------------------------------------------------------------ serial binding

def test_lane_flags_never_orphan_a_continuation(tmp_path):
    campaign = _write(tmp_path / "c.json", {"campaign": 1})
    base = ["--resolved-campaign", str(campaign), "--target-id", "ds41-t", "--workers", "2"]
    bound = [*base[:4], "--workers", "1", "--lane-targets", str(tmp_path / "lt.json"),
             "--lane", "lane0"]
    assert serial_run.LANE_BINDING_FLAGS == frozenset(lane_targets.FLAGS)
    assert serial_run.resume_binding(base) == serial_run.resume_binding(bound)


# ------------------------------------------------------------------ Q38FN recipe flag

@pytest.mark.parametrize("value,supported", [("0.5", True), ("0", True), ("1.5", False),
                                             ("nan", False), ("x", False)])
def test_draft_p_min_is_a_normalized_extension_flag(value, supported):
    from . import resolved_recipe, serving
    template = serving.Recipe(name="q38", model="/m.gguf", device="none", ngl=0,
                              extra_flags=("--device-draft", "none", "--draft-p-min", value))
    report = resolved_recipe._capability(template, "cpu", None, ())
    codes = {reason.code for reason in report.reasons}
    assert ("extra_flags_unsupported" not in codes) is supported
