import pytest

from . import typed_planner as tp

LIVE = [{"mechanism_id": "akm-a", "summary": "fuse"}]
DNR = [{"mechanism_id": "akm-prefetch", "summary": "prefetch (DEAD)"}]


def test_options_keep_dnr_selectable_and_typed_controls():
    opts = tp.build_options(LIVE, DNR)
    ids = [o["option"] for o in opts]
    assert ids == ["akm-a", "akm-prefetch", "explore", "exhausted"]
    assert opts[1]["premise_superseded"] is True
    reply = {"choice": "akm-prefetch", "statement": "s", "falsifier": "f",
             "target_surface": "x.c", "target_symbol": "g"}
    assert tp.parse_choice(reply, opts)["premise_superseded"] is True


def test_exhausted_is_distinct_and_carries_no_free_text():
    opts = tp.build_options(LIVE)
    out = tp.parse_choice('{"choice": "exhausted", "reason": "akm-a spent"}', opts)
    assert out["choice"] == "exhausted" and out["text"] == {"reason": "akm-a spent"}
    with pytest.raises(tp.TypedPlannerRefused):
        tp.parse_choice({"choice": "exhausted", "reason": "r", "statement": "s"}, opts)
    with pytest.raises(tp.TypedPlannerRefused):
        tp.parse_choice({"choice": "akm-zzz"}, opts)
    with pytest.raises(tp.TypedPlannerRefused):
        tp.parse_choice({"choice": "akm-a", "statement": "s"}, opts)


def test_explore_reserve_is_enforced():
    assert tp.explore_required([]) is True
    assert tp.explore_required(["explore", "akm-a", "akm-a"]) is False
    assert tp.explore_required(["explore", "akm-a", "akm-a", "akm-a"]) is True
    opts = tp.build_options(LIVE)
    with pytest.raises(tp.TypedPlannerRefused, match="reserved explore"):
        tp.parse_choice({"choice": "akm-a", "statement": "s", "falsifier": "f",
                         "target_surface": "x", "target_symbol": "g"}, opts, must_explore=True)
    assert "MUST choose" in tp.render_choice_prompt(opts, must_explore=True)


def test_option_cap_refuses():
    with pytest.raises(tp.TypedPlannerRefused, match="Choice cap"):
        tp.build_options([{"mechanism_id": f"akm-{i}"} for i in range(260)])


def test_replay_reachability_needs_conditions():
    calls = [{"wall_min": 30, "category": "ABSTAIN-MAPPED", "downstream": ["abstained"]},
             {"wall_min": 30, "category": "FAMILY", "downstream": ["kept"]}]
    keeps = [{"mechanism_id": m, "category": c} for m, c in
             (("k1", "FAMILY"), ("k2", "NONE"), ("k3", "FAMILY-DNR"))]
    full = tp.replay_yield(calls, keeps)
    bare = tp.replay_yield(calls, keeps, conditions=(False, False, False))
    assert full["keeps_reachable"] == 3 and bare["keeps_reachable"] == 1
    assert full["yield_per_hour_typed"] > full["yield_per_hour_free"]
    assert tp.replay_yield(calls[:1], keeps)["yield_per_hour_free"] is None
