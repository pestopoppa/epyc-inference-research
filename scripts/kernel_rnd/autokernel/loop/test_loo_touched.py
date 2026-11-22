import pytest

from . import loo_touched as lt

SCREENS = {"k1": {"touched_ids": ["w1"], "inert_ids": ["w2"], "unknown_ids": []},
           "k2": {"touched_ids": ["w2"], "inert_ids": [], "unknown_ids": ["w1"]}}
FULL = {"w1": 100.0, "w2": 50.0}


def test_plan_skips_inert_keeps_unknown():
    p = lt.loo_plan(SCREENS)
    assert p["k1"] == {"required": ["w1"], "skipped_inert": ["w2"]}
    assert p["k2"]["required"] == ["w1", "w2"]


def test_count_restricted_to_touched_ignores_inert_arm():
    plan = lt.loo_plan(SCREENS)
    arms = [{"keep_id": "k1", "keep_time_effect": 0.02, "loo": {"w1": 90.0, "w2": 999.0}},
            {"keep_id": "k2", "keep_time_effect": 0.03, "loo": {"w1": 99.0, "w2": 50.0}}]
    r = lt.count_touched(arms, full_bundle=FULL, plan=plan)
    # k1 w2 is INERT so its 999 does not convict; k2 w2 equal -> convicted
    assert r["goodharted_keep_ids"] == ["k2"] and r["goodharted_count"] == 1
    assert r["ignored_surfaces"] == {"k1": ["w2"]}


def test_missing_touched_value_or_arm_refuses():
    plan = lt.loo_plan(SCREENS)
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.count_touched([{"keep_id": "k1", "keep_time_effect": 0.0, "loo": {"w1": 1.0}}],
                         full_bundle=FULL, plan=plan)
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.count_touched([{"keep_id": "k1", "keep_time_effect": 0.0, "loo": {"w1": 1.0}},
                          {"keep_id": "k2", "keep_time_effect": 0.0, "loo": {"w2": 1.0}}],
                         full_bundle=FULL, plan=plan)


def test_overlapping_classes_and_unplanned_arm_refuse():
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.loo_plan({"k": {"touched_ids": ["w"], "inert_ids": ["w"], "unknown_ids": []}})
    plan = lt.loo_plan({"k1": SCREENS["k1"]})
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.count_touched([{"keep_id": "zz", "keep_time_effect": 0, "loo": {"w1": 1.0}}],
                         full_bundle=FULL, plan=plan)


def test_promotion_record_refuses_without_count_or_receipts():
    plan = lt.loo_plan(SCREENS)
    arms = [{"keep_id": "k1", "keep_time_effect": 0.02, "loo": {"w1": 90.0}},
            {"keep_id": "k2", "keep_time_effect": 0.03, "loo": {"w1": 90.0, "w2": 40.0}}]
    c = lt.count_touched(arms, full_bundle=FULL, plan=plan)
    assert lt.promotion_loo_record(plan=plan, count=c, loo_receipts=["a", "b"])["goodharted_count"] == 0
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.promotion_loo_record(plan=plan, count=c, loo_receipts=[])
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.promotion_loo_record(plan=plan, count=c, loo_receipts=["a"])
    with pytest.raises(lt.LOOTouchedRefusal):
        lt.promotion_loo_record(plan=plan, count={}, loo_receipts=["a", "b"])


def test_owed_keeps_splits_by_surface_and_refuses_unclassed():
    p = lt.loo_plan(SCREENS)
    assert lt.owed_keeps(["k1", "k2"], p, "w2") == (["k2"], ["k1"])
    assert lt.owed_keeps(["k1", "k2"], p, "w1") == (["k1", "k2"], [])
    for args in ((["k3"], p, "w1"), (["k1"], p, "w9"), (["k1"], p, "")):
        with pytest.raises(lt.LOOTouchedRefusal):
            lt.owed_keeps(*args)
