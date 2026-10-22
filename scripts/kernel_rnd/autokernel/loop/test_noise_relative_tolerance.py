"""AK-PORT-LB1: the noise-floor-relative model-level tolerance (pure logit-row fixtures)."""
from __future__ import annotations

import json
import random

import pytest

from . import gates
from . import noise_relative_tolerance as nrt

V = 12          # tiny vocab
N = 16          # positions (>= DEFAULT_MIN_POSITIONS)


def _rows(seed=7, n=N, v=V, spread=4.0):
    """Reference logits with a clear top-1 (margin ~ spread) at every position."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        row = [rnd.uniform(-1.0, 1.0) for _ in range(v)]
        row[rnd.randrange(v)] += spread
        rows.append(row)
    return rows


def _perturb(rows, eps, seed):
    rnd = random.Random(seed)
    return [[x + rnd.uniform(-eps, eps) for x in r] for r in rows]


def _shift_to_max(rows, eps):
    """Candidate whose max divergence from `rows` is exactly eps at every position."""
    return [[x + (eps if i == 0 else 0.0) for i, x in enumerate(r)] for r in rows]


def test_candidate_inside_the_band_passes():
    single = _rows()
    inc_b = _shift_to_max(single, 0.010)       # floor 0.010
    cand = _shift_to_max(single, 0.014)        # 1.4x floor
    r = nrt.evaluate(single, inc_b, cand)
    assert r.status == nrt.PASS, r.reason
    assert r.detail["ratio_observed"] == pytest.approx(1.4)


def test_candidate_just_outside_the_band_is_wrong():
    single = _rows()
    r = nrt.evaluate(single, _shift_to_max(single, 0.010), _shift_to_max(single, 0.016))
    assert r.status == nrt.WRONG and "1.5x" in r.reason


def test_a_loose_absolute_tolerance_would_have_admitted_this_but_the_floor_relative_gate_does_not():
    # incumbent noise 1e-4; candidate 1e-2 is "small" in absolute terms yet 100x the floor
    single = _rows()
    r = nrt.evaluate(single, _shift_to_max(single, 1e-4), _shift_to_max(single, 1e-2))
    assert r.status == nrt.WRONG


def test_a_noisy_incumbent_widens_the_band():
    # the SAME 0.05 candidate divergence is wrong against a quiet incumbent and fine against a noisy one
    single = _rows()
    cand = _shift_to_max(single, 0.05)
    assert nrt.evaluate(single, _shift_to_max(single, 0.01), cand).status == nrt.WRONG
    assert nrt.evaluate(single, _shift_to_max(single, 0.04), cand).status == nrt.PASS


def test_greedy_flip_at_a_confident_position_is_wrong_even_inside_the_divergence_band():
    # A flip needs |delta logit| >= margin/2, so it can hide inside the 1.5x band only when
    # margin_gate <= margin <= ~3x floor: floor 0.11 -> band 0.165, margin gate 0.25, margin 0.30.
    single = _rows(spread=4.0)
    single[3] = [-1.0] * V
    single[3][4], single[3][9] = 1.00, 0.70
    cand = [list(r) for r in single]
    cand[3][4], cand[3][9] = 0.85, 0.855          # argmax flips; max delta 0.155 <= 0.165
    inc_b = _shift_to_max(single, 0.11)
    r = nrt.evaluate(single, inc_b, cand)
    assert r.detail["candidate_divergence"] <= r.detail["ratio_bound"]
    assert r.status == nrt.WRONG and "greedy flip" in r.reason
    assert r.detail["flips"][0]["position"] == 3


def test_flip_at_a_near_tie_position_is_exempt():
    single = _rows(spread=4.0)
    single[5] = [0.0] * V
    single[5][2], single[5][7] = 1.00, 0.95     # margin 0.05 < max(0.25, 2*floor)
    cand = [list(r) for r in single]
    cand[5][2], cand[5][7] = cand[5][7], cand[5][2]
    inc_b = _shift_to_max(single, 0.04)
    cand = _shift_to_max(cand, 0.04)
    r = nrt.evaluate(single, inc_b, cand)
    assert r.status == nrt.PASS, r.reason
    assert r.detail["near_tie_exempt"] >= 1


def test_margin_gate_scales_with_the_floor():
    single = _rows(spread=1.0)
    # with floor 0.5 the margin gate is max(0.25, 1.0) = 1.0
    r = nrt.evaluate(single, _shift_to_max(single, 0.5), _shift_to_max(single, 0.5))
    assert r.detail["margin_gate"] == pytest.approx(1.0)


def test_zero_floor_requires_bit_exactness():
    single = _rows()
    assert nrt.evaluate(single, [list(r) for r in single], [list(r) for r in single]).status == nrt.PASS
    r = nrt.evaluate(single, [list(r) for r in single], _shift_to_max(single, 1e-6))
    assert r.status == nrt.WRONG and "row-invariant" in r.reason
    assert r.detail["floor_is_zero"] is True


@pytest.mark.parametrize("single,inc_b,cand,needle", [
    ([], [], [], "no positions"),
    (_rows(n=3), _rows(n=3), _rows(n=3), "min_positions"),
    (_rows(), _rows(n=N - 1), _rows(), "position counts differ"),
    ([[1.0]] * N, [[1.0]] * N, [[1.0]] * N, "fewer than 2 logits"),
])
def test_unusable_input_is_unavailable_never_a_pass(single, inc_b, cand, needle):
    r = nrt.evaluate(single, inc_b, cand)
    assert r.status == nrt.UNAVAILABLE and needle in r.reason


def test_non_finite_reference_is_unavailable_and_non_finite_candidate_is_wrong():
    single = _rows()
    bad = [list(r) for r in single]
    bad[2][0] = float("nan")
    assert nrt.evaluate(bad, single, single).status == nrt.UNAVAILABLE
    assert nrt.evaluate(single, bad, single).status == nrt.UNAVAILABLE
    r = nrt.evaluate(single, _shift_to_max(single, 0.01), bad)
    assert r.status == nrt.WRONG and "non-finite" in r.reason


def test_no_confident_position_means_the_flip_half_observed_nothing():
    single = [[0.01 * i for i in range(V)] for _ in range(N)]     # top-2 margin 0.01 everywhere
    r = nrt.evaluate(single, _shift_to_max(single, 0.01), _shift_to_max(single, 0.012))
    assert r.status == nrt.UNAVAILABLE and "observed nothing" in r.reason


def test_stat_variants_are_applied_to_floor_and_candidate_alike():
    single = _rows()
    inc_b, cand = _perturb(single, 0.01, 1), _perturb(single, 0.012, 2)
    for stat in nrt.STATS:
        r = nrt.evaluate(single, inc_b, cand, nrt.Config(stat=stat))
        assert r.detail["stat"] == stat and r.detail["floor"] > 0


def test_config_validation():
    with pytest.raises(ValueError):
        nrt.Config(stat="median")
    with pytest.raises(ValueError):
        nrt.Config(ratio=0.5)


def test_gate_wrapper_maps_status_to_verdict_gate_names():
    single = _rows()
    ok = gates.check_noise_relative_tolerance(single, _shift_to_max(single, 0.01), _shift_to_max(single, 0.012))
    assert (ok.gate, ok.passed) == ("reference_comparison", True)
    bad = gates.check_noise_relative_tolerance(single, _shift_to_max(single, 0.01), _shift_to_max(single, 0.05))
    assert (bad.gate, bad.passed) == ("reference_comparison", False)
    na = gates.check_noise_relative_tolerance([], [], [])
    assert (na.gate, na.passed) == ("oracle_unavailable", False)
    assert json.loads(ok.detail)["positions"] == N
