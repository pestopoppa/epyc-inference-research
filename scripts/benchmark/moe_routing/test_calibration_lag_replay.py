"""RT-6.2 model-free policy fidelity and chronological calibration controls."""
import numpy as np
import pytest

import calibration_lag_replay as lag


@pytest.fixture(scope="module")
def reference():
    return lag.replay.place_deepinfer


def test_historical_surrogate_is_preserved_and_not_relabelled_upstream():
    loads = np.array([[2, 1, 3, 3], [0, 0, 0, 0], [2, 1, 1, 0]])
    # Real upstream774ce244 CPU reference gives copies[2,1,1,2], per audit artifact.
    placement = lag.replay.place_deepinfer(loads, 2, 2)
    assert [len(x) for x in placement] == [3, 1, 1, 1]


def test_all_cells_use_same_targets_and_never_cross_sequences(reference):
    loads = np.ones((12, 4))
    windows = np.repeat([10, 20], 6)
    rows = lag.sweep(loads, windows, [1, 2], [0, 1], 2, 0, reference)
    selected = [[(r["window"], r["target_block"]) for r in cell["targets"]] for cell in rows]
    assert all(targets == selected[0] for targets in selected)
    assert selected[0] == [(10, 3), (10, 4), (10, 5), (20, 9), (20, 10), (20, 11)]
    for cell in rows:
        for target in cell["targets"]:
            assert target["history_begin"] >= (0 if target["window"] == 10 else 6)
            assert target["history_end_exclusive"] <= target["target_block"]


def test_lag_and_target_mutations_do_not_leak_into_calibration(reference):
    loads = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    base = lag.sweep(loads, np.zeros(4), [2], [1], 2, 0, reference)[0]
    changed = loads.copy()
    changed[2:] = [[4, 0, 0, 0], [3, 0, 0, 0]]
    mutated = lag.sweep(changed, np.zeros(4), [2], [1], 2, 0, reference)[0]
    assert base["targets"][0]["calibration_sha256"] == mutated["targets"][0]["calibration_sha256"]
    assert base["targets"][0]["target_sha256"] != mutated["targets"][0]["target_sha256"]
    assert base["targets"][0]["history_begin"] == 0
    assert base["targets"][0]["history_end_exclusive"] == 2


@pytest.mark.parametrize("histories,lags", [([], [0]), ([0], [0]), ([1], [-1])])
def test_invalid_history_lag_refuses(histories, lags):
    with pytest.raises(ValueError):
        lag.select_targets(np.zeros(3), histories, lags)


def test_short_sequence_is_unsupported_and_invalid_bins_refuse():
    def never_called(*_):
        pytest.fail("unsupported policy must not execute")
    cell = lag.sweep(np.ones((3, 4)), np.zeros(3), [3], [1], 2, 0, never_called)[0]
    assert cell["status"] == "unsupported"
    assert cell["target_count"] == 0
    assert cell["policies"]["historical_surrogate"]["realizable_max_over_mean"] is None
    with pytest.raises(ValueError, match="bins positive"):
        lag.sweep(np.ones((4, 4)), np.zeros(4), [1], [0], 0, 0, never_called)



def test_zero_target_is_explicit_unknown_not_nan_or_zero():
    rows = lag.sweep(np.zeros((2, 4)), np.zeros(2), [1], [0], 2, 0,
                     lambda *_: pytest.fail("undefined denominator"))
    assert rows[0]["status"] == "unsupported"
    assert rows[0]["unknown_targets"][0]["reason"] == "zero_target_load"
    assert rows[0]["policies"]["historical_surrogate"]["n"] == 0


def test_window_reappearance_refuses_chronological_ambiguity():
    with pytest.raises(ValueError, match="contiguous"):
        lag.select_targets(np.array([1, 1, 2, 1]), [1], [0])
