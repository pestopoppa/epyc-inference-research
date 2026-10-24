"""The trigger audit replays the real `actors.diminishing_returns_triggers`."""
from autokernel.history_replay import render_trigger_audit as audit


def _rec(i, status, mech):
    return {"campaign_dir": "c", "status": status, "mechanism_id": mech, "statement": None,
            "recorded_at": f"2026-10-01T00:00:{i:02d}Z",
            "task_key": {"target_surface": "a.c", "target_symbol": "f"}}


def test_an_abstention_supplying_the_third_distinct_idea_is_attributed():
    records = [_rec(0, "measured_null", "m1"), _rec(1, "measured_null", "m2"),
               _rec(2, "abstained", "m3"), _rec(3, "planner_transient", None)]
    out = audit.audit(records)["campaigns"]["c"]
    assert out["arms"]["pre_decision"]["stagnation_block_fires"] == 1
    assert out["arms"]["no_abstained"]["stagnation_block_fires"] == 0
    assert out["family_firings_only_with"]["stagnation_families_needing_abstained"] == 1
    assert out["identity"] == {"abstained:mechanism:surface_symbol": 1}
