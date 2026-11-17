"""AK-PROF-ATTR-1 controls: synthetic overlap (union < sum), serial trace (union == sum), cross-run subtraction refused."""
import pytest

import prior_art
import trace_attribution as ta


def _d(i, s, e):
    return prior_art.ProfileDispatch(dispatch_id=str(i), kernel_name=f"k{i}", kernel_family="f", start_ns=s, end_ns=e)


def test_synthetic_overlap_union_less_than_sum():
    t = ta.attribute([_d(0, 0, 100), _d(1, 50, 150), _d(2, 300, 400)], run_id="r1")
    assert t.sum_ns == 300
    assert t.active_union_ns == 250
    assert t.overlap_ns == 50
    assert t.span_ns == 400
    assert t.idle_gap_ns == 150
    assert t.active_union_ns < t.sum_ns
    assert t.as_dict()["additive"] is False


def test_serial_trace_is_additive_and_mapping_input():
    t = ta.attribute([{"start_ns": 0, "end_ns": 10}, {"start_ns": 10, "end_ns": 30}], run_id="r1")
    assert (t.sum_ns, t.active_union_ns, t.overlap_ns, t.idle_gap_ns) == (30, 30, 0, 0)


def test_nested_and_empty():
    assert ta.attribute([_d(0, 0, 100), _d(1, 10, 20)], "r").active_union_ns == 100
    assert ta.attribute([], "r").sum_ns == 0


def test_same_run_subtraction_allowed():
    t = ta.attribute([_d(0, 0, 100)], run_id="r1")
    assert ta.unattributed_ns(160, "r1", t) == 60


def test_cross_run_subtraction_rejected():
    t = ta.attribute([_d(0, 0, 100)], run_id="profiler-run")
    with pytest.raises(ta.CrossRunSubtractionError):
        ta.unattributed_ns(160, "bench-run", t)
    with pytest.raises(ta.CrossRunSubtractionError):
        ta.unattributed_ns(160, "", t)


def test_reversed_interval_rejected():
    with pytest.raises(ValueError):
        ta.attribute([{"start_ns": 5, "end_ns": 1}], "r")
