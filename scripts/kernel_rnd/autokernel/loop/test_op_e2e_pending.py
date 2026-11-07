"""AK-OP-E2E-PENDING: op wins stay pending until a same-run ABBA e2e confirms; skewed routing."""
from __future__ import annotations

import pytest

from . import keep_screen as ks
from . import served_shape_cases as ssc
from . import serving_gate as sg

H8B_OP = {"ratio": 1.29, "decisive": True}


def _e2e(effect, floor=1.0, decisive=True, **kw):
    return {"design": "ABBA", "same_run": True, "effect_pct": effect, "floor_pct": floor,
            "decisive": decisive, **kw}


def test_op_win_without_e2e_is_pending():
    assert ks.op_keep_status(H8B_OP)["status"] == ks.OP_PENDING_E2E


def test_h8b_op_win_e2e_loss_is_refuted_not_kept():
    assert ks.op_keep_status(H8B_OP, _e2e(-5.1))["status"] == ks.OP_REFUTED_E2E


def test_h14_e2e_noise_stays_pending():
    r = ks.op_keep_status({"ratio": 1.13, "decisive": True}, _e2e(0.2, decisive=False))
    assert r["status"] == ks.OP_PENDING_E2E
    # non-vacuity: same effect size with a decisive flag but below floor must also stay pending
    assert ks.op_keep_status(H8B_OP, _e2e(0.2, decisive=True))["status"] == ks.OP_PENDING_E2E


def test_confirmed_only_by_same_run_abba_decisive_positive():
    assert ks.op_keep_status(H8B_OP, _e2e(4.8))["status"] == ks.OP_CONFIRMED_E2E


@pytest.mark.parametrize("bad", [_e2e(4.8, same_run=False), _e2e(4.8, design="AB"),
                                 _e2e(None), _e2e(4.8, floor=0), _e2e(float("nan")),
                                 _e2e(True)])
def test_negative_control_cross_run_or_malformed_never_confirms(bad):
    assert ks.op_keep_status(H8B_OP, bad)["status"] == ks.OP_PENDING_E2E


@pytest.mark.parametrize("op", [None, {}, {"ratio": 0.9, "decisive": True},
                                {"ratio": 1.3, "decisive": False}, {"ratio": True, "decisive": True}])
def test_no_op_win_is_not_pending(op):
    assert ks.op_keep_status(op)["status"] == ks.OP_NO_WIN


def test_awaiting_e2e_lists_only_pending():
    recs = [{"id": "h8b", "op": H8B_OP, "e2e": _e2e(-5.1)},
            {"id": "h14", "op": H8B_OP},
            {"id": "h10", "op": H8B_OP, "e2e": _e2e(9.0)},
            {"id": "lose", "op": {"ratio": 0.9, "decisive": True}}]
    assert ks.awaiting_e2e(recs) == ["h14"]


def test_serving_gate_projection_uses_noise_floor_and_row_decisive():
    row = {"effect_pct": 4.0, "noise_floor_pct": 1.0, "decisive": True}
    assert sg.op_win_verdict(H8B_OP, row)["status"] == ks.OP_CONFIRMED_E2E
    assert sg.op_win_verdict(H8B_OP, None)["status"] == ks.OP_PENDING_E2E
    assert sg.op_win_verdict(H8B_OP, {"effect_pct": 4.0, "decisive": None})["status"] == ks.OP_PENDING_E2E


def test_uniform_default_unchanged():
    for n, u in ((512, 10), (384, 8)):
        for row in range(6):
            assert ssc.routed_ids(n, u, row) == ssc.route_ids(n, u, row)
    assert ssc.route_ids(512, 10, 0)[:3] == (511, 460, 409)


def test_replay_is_deterministic_distinct_and_skewed():
    hist = ssc.zipf_histogram(256, 1.2)
    a = [ssc.routed_ids(256, 8, r, distribution="replay", histogram=hist) for r in range(64)]
    b = [ssc.routed_ids(256, 8, r, distribution="replay", histogram=hist) for r in range(64)]
    assert a == b
    assert all(len(set(x)) == 8 and all(0 <= i < 256 for i in x) for x in a)
    hot = sum(1 for x in a for i in x if i < 8)
    uni = [ssc.route_ids(256, 8, r) for r in range(64)]
    hot_uni = sum(1 for x in uni for i in x if i < 8)
    assert hot > 4 * max(hot_uni, 1)          # negative control: uniform has no skew


def test_replay_refuses_silent_uniform_fallback():
    with pytest.raises(ValueError):
        ssc.routed_ids(64, 4, 0, distribution="replay")
    with pytest.raises(ValueError):
        ssc.routed_ids(64, 4, 0, distribution="replay", histogram=[1.0] * 63)
    with pytest.raises(ValueError):
        ssc.routed_ids(64, 4, 0, distribution="zipf")
    with pytest.raises(ValueError):
        ssc.replay_route_ids([1.0, 0, 0, 0], 2, 0)


def test_skew_summary():
    s = ssc.skew_summary(ssc.zipf_histogram(256, 1.2))
    assert s["top_share"] > 3 * s["uniform_top_share"]
    assert ssc.skew_summary([1.0] * 256)["top_share"] == pytest.approx(8 / 256)
