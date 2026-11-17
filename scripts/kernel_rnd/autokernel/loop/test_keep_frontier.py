import pytest

from . import keep_frontier as kf


def P(i, e, s):
    return kf.make_point(i, e, s)


def test_variance_reduction_survives_a_faster_noisy_keep():
    fast_noisy, slow_tight = P("a", 5.0, 4.0), P("b", 3.0, 0.5)
    assert {p.candidate_id for p in kf.frontier([fast_noisy, slow_tight])} == {"a", "b"}


def test_dominated_point_is_dropped_and_order_preserved():
    pts = [P("a", 5.0, 1.0), P("b", 4.0, 2.0), P("c", 6.0, 3.0)]
    assert [p.candidate_id for p in kf.frontier(pts)] == ["a", "c"]


def test_exact_duplicates_both_retained_and_ids_unique():
    assert len(kf.frontier([P("a", 1, 1), P("b", 1, 1)])) == 2
    with pytest.raises(ValueError):
        kf.frontier([P("a", 1, 1), P("a", 2, 2)])


def test_retain_evicts_dominated_members_and_rejects_dominated_newcomer():
    cur = kf.frontier([P("a", 5.0, 2.0), P("b", 3.0, 0.5)])
    after, evicted = kf.retain(cur, P("c", 6.0, 1.0))
    assert [p.candidate_id for p in after] == ["b", "c"] and [p.candidate_id for p in evicted] == ["a"]
    after2, evicted2 = kf.retain(after, P("d", 2.0, 2.0))
    assert after2 == after and evicted2 == []


@pytest.mark.parametrize("effect,spread", [(float("nan"), 1), (1, None), (1, -0.1), (True, 1), ("1", 1)])
def test_missing_or_invalid_coordinate_refuses(effect, spread):
    with pytest.raises(ValueError):
        kf.make_point("x", effect, spread)


def test_uses_the_orchestrator_dominates_not_a_copy():
    assert kf._load_dominates().__module__ == "src.autopilot_core.pareto_math"


def test_record_carries_both_coordinates():
    assert P("a", 2.0, 0.3).record() == {"schema": kf.SCHEMA, "candidate_id": "a",
                                         "effect": 2.0, "spread": 0.3}
