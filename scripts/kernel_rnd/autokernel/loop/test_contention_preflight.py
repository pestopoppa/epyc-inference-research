"""AK-EVAL-QUARTERS: the 3/2/1 contention fallback. Synthetic probes only; no bandwidth probe runs."""
import pytest

from . import contention_preflight as cp


SOLO = {"q0": 100.0, "q1": 100.0, "q2": 100.0}


@pytest.mark.parametrize("concurrent,admitted", [
    ({"q0+q1+q2": {"q0": 96.0, "q1": 95.0, "q2": 97.0}}, ("q0", "q1", "q2")),
    ({"q0+q1+q2": {"q0": 96.0, "q1": 90.0, "q2": 97.0}, "q0+q2": {"q0": 98.0, "q2": 99.0}}, ("q0", "q2")),
    ({"q0+q1+q2": {"q0": 80.0, "q1": 80.0, "q2": 80.0}, "q0+q2": {"q0": 94.9, "q2": 99.0}}, ("q0",)),
    ({"q0+q1+q2": {"q0": None, "q1": 99.0, "q2": 99.0}, "q0+q2": {}}, ("q0",)),
])
def test_three_two_one_fallback_at_95_percent_of_solo(concurrent, admitted):
    assert cp.admit(SOLO, concurrent)[0] == admitted


def test_preflight_stops_at_the_first_admitted_set_and_receipt_gates_regions(tmp_path):
    calls = []

    def probe(quarter):
        calls.append(quarter)
        return 100.0
    receipt = cp.run_preflight(probe)
    assert receipt["admitted"] == ["q0", "q1", "q2"] and receipt["streams"] == 3
    assert len(calls) == 6        # 3 solo + one concurrent set; q0+q2 never tried
    path = cp.write_receipt(tmp_path / "preflight.json", receipt)
    assert cp.admitted_region(path, "q1") == "q1"
    with pytest.raises(ValueError, match="not admitted"):
        cp.admitted_region(path, "q3")
    path.write_text(path.read_text().replace('"q2"', '"q3"'))
    with pytest.raises(ValueError, match="altered"):
        cp.admitted_region(path, "q3")



def test_two_stream_receipt_admits_q0_q2_only(tmp_path):
    solo = dict(SOLO)
    admitted, tried = cp.admit(solo, {"q0+q1+q2": {"q0": 90.0, "q1": 90.0, "q2": 90.0},
                                      "q0+q2": {"q0": 97.0, "q2": 96.0}})
    receipt = {"schema": cp.SCHEMA, "solo": solo, "tried": tried, "admitted": list(admitted),
               "streams": len(admitted), "solo_fraction_min": cp.SOLO_FRACTION_MIN}
    path = cp.write_receipt(tmp_path / "p.json", receipt)
    assert cp.admitted_region(path, "q2") == "q2"
    with pytest.raises(ValueError, match="not admitted"):
        cp.admitted_region(path, "q1")
