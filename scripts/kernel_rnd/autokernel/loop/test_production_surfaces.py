"""AKX-ALL-LB2: production-shape surfaces and the not-yet-transferable label."""
from __future__ import annotations

import pytest

from . import production_surfaces as ps


def test_surfaces_cover_n8_n16_n32_and_the_ragged_mix():
    assert {s.surface_id: s.ne11 for s in ps.SURFACES} == {
        "n8": 8, "n16": 16, "n32": 32, "ragged_1_8_8_8": 25}
    assert [s.surface_id for s in ps.SURFACES if s.anchor] == ["n8"]
    assert ps.BY_ID["ragged_1_8_8_8"].slot_rows == (1, 8, 8, 8)
    assert ps.BY_ID["ragged_1_8_8_8"].kind == "ragged"


def test_production_surfaces_exclude_the_anchor_width():
    assert [s.surface_id for s in ps.production_surfaces()] == ["n16", "n32", "ragged_1_8_8_8"]


def test_a_keep_measured_only_at_n8_is_not_yet_transferable_and_refused():
    a = ps.assess({"n8": 7.1}, floor_pct=2.7)
    assert a.label == ps.NOT_YET_TRANSFERABLE and not a.promotable
    assert a.missing == ("n16", "n32", "ragged_1_8_8_8")
    assert "not-yet-transferable" in ps.promotion_refusal(a)


def test_a_partial_production_measurement_is_still_not_transferable():
    a = ps.assess({"n8": 7.1, "n16": 5.0, "n32": None}, floor_pct=2.7)
    assert a.label == ps.NOT_YET_TRANSFERABLE and a.missing == ("n32", "ragged_1_8_8_8")


def test_transferable_only_when_every_production_surface_is_measured_and_none_regresses():
    ok = ps.assess({"n8": 7.1, "n16": 5.0, "n32": 2.0, "ragged_1_8_8_8": 1.0}, floor_pct=2.7)
    assert ok.promotable and ps.promotion_refusal(ok) is None
    # inside the noise floor is not a regression
    edge = ps.assess({"n16": -2.6, "n32": 0.0, "ragged_1_8_8_8": 0.0}, floor_pct=2.7)
    assert edge.label == ps.TRANSFERABLE


def test_a_measured_regression_on_a_production_surface_refuses_promotion():
    a = ps.assess({"n8": 9.0, "n16": 4.0, "n32": -6.0, "ragged_1_8_8_8": 1.0}, floor_pct=2.7)
    assert a.label == ps.REGRESSES and a.regressed == ("n32",)
    assert "n32" in ps.promotion_refusal(a)


def test_missing_wins_over_regressed_in_the_label_but_both_are_reported():
    a = ps.assess({"n16": -9.0}, floor_pct=2.7)
    assert a.label == ps.NOT_YET_TRANSFERABLE and a.regressed == ("n16",)


def test_unknown_surface_and_negative_floor_are_caller_errors():
    with pytest.raises(ValueError):
        ps.assess({"n64": 1.0}, floor_pct=2.7)
    with pytest.raises(ValueError):
        ps.assess({}, floor_pct=-1)
