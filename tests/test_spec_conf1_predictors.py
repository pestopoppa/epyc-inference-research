import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/speculative/spec_conf1"))
import predictors as P  # noqa: E402

# Recorded-trace fixture (hand-built, 3 rounds): chain drafts, target prob of the drafted token.
TRACE = [
    {"slot": 0, "round": 0, "n_drafted": 4, "n_accepted": 2, "top1_prob": [0.9, 0.8, 0.5, 0.4],
     "cand_probs": [[0.92], [0.85], [0.10]]},
    {"slot": 0, "round": 1, "n_drafted": 4, "n_accepted": 4, "top1_prob": [0.95, 0.9, 0.85, 0.8],
     "cand_probs": [[0.9], [0.9], [0.8], [0.7]]},
    {"slot": 1, "round": 0, "n_drafted": 3, "n_accepted": 0, "top1_prob": [0.4, 0.3, 0.2],
     "cand_probs": [[0.05]]},
]


def test_frontier_gap_definition_tree():
    assert abs(P.frontier_gap([0.5, 0.3]) - 0.2) < 1e-12
    assert P.frontier_gap([0.7, 0.6]) == 0.0  # clipped


def test_round_fields_first_reject_and_unverified_none():
    f = P.round_fields(TRACE[0])
    assert abs(f["frontier_gap_first_reject"] - 0.90) < 1e-12  # gap at position 2 (the rejection)
    assert f["frontier_gap"][3] is None  # never verified
    assert P.round_fields(TRACE[1])["frontier_gap_first_reject"] is None  # fully accepted


def test_fifth_predictor_row_present_and_signed():
    t = P.build_table(TRACE)
    assert set(t) == set(P.PREDICTORS) and "frontier_gap" in t
    assert t["frontier_gap"]["n"] == 3 + 4 + 1  # verified positions only
    assert t["frontier_gap"]["auc"] == 1.0  # low gap separates accepted from rejected here


def test_missing_cand_probs_rejected():
    bad = dict(TRACE[0], cand_probs=[[0.9]])
    try:
        P.round_fields(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_auc_ties_and_empty():
    assert P.auc([1, 1], [1, 0]) == 0.5
    assert P.auc([1, 2], [1, 1]) is None


def test_json_roundtrip():
    assert P.build_table(json.loads(json.dumps(TRACE)))["top1"]["n"] == 11
