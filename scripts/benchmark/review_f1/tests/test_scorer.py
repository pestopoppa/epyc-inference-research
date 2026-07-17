"""Tests for the review_f1 deterministic scorer.

Runnable BOTH under pytest and stand-alone (`python test_scorer.py`) because
the research .venv has no pytest. Test functions take no fixture args and build
data via conftest builders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conftest  # noqa: E402
import scorer  # noqa: E402


def _finding(criterion, file, start, end=None):
    return {"criterion": criterion, "location": {"file": file, "line_start": start, "line_end": end}}


def test_perfect_recall_and_precision():
    cases = conftest.make_golden_cases()
    run = {
        "repoA__pr-1": [_finding("logic_bug", "a.py", 10, 12), _finding("security", "a.py", 30)],
        "repoB__pr-2": [_finding("runtime_error", "b.go", 40, 44)],
    }
    res = scorer.score_run(cases, run)
    # 3 scored goldens (a-g0, a-g1, b-g0), all matched, no FP.
    assert res["tp"] == 3 and res["fp"] == 0 and res["fn"] == 0
    assert res["precision"] == 1.0 and res["recall"] == 1.0 and res["f1"] == 1.0


def test_low_severity_golden_is_never_fn():
    cases = conftest.make_golden_cases()
    # Reviewer finds nothing at all. The 2 low-severity goldens must NOT count.
    res = scorer.score_run(cases, {"repoA__pr-1": [], "repoB__pr-2": []})
    assert res["fn"] == 3  # only the 3 scored goldens are FN, not the 2 low ones
    assert res["tp"] == 0 and res["fp"] == 0


def test_low_severity_match_is_neutral_not_fp():
    cases = conftest.make_golden_cases()
    # Reviewer emits exactly one finding that matches ONLY the low-severity golden a-g2.
    run = {"repoA__pr-1": [_finding("logic_bug", "a.py", 5, 5)], "repoB__pr-2": []}
    res = scorer.score_run(cases, run)
    # Neutral: not a TP (low golden not scored) and NOT an FP.
    assert res["tp"] == 0 and res["fp"] == 0
    assert res["fn"] == 3  # all scored goldens still unmatched


def test_false_positive_counted():
    cases = conftest.make_golden_cases()
    run = {
        "repoA__pr-1": [_finding("security", "a.py", 999)],  # matches no golden
        "repoB__pr-2": [],
    }
    res = scorer.score_run(cases, run)
    assert res["fp"] == 1 and res["tp"] == 0 and res["fn"] == 3


def test_dedup_by_golden_index():
    cases = conftest.make_golden_cases()
    # Two reviewer findings both overlap the single golden a-g0 -> one TP, one FP.
    run = {
        "repoA__pr-1": [_finding("logic_bug", "a.py", 10, 12), _finding("logic_bug", "a.py", 11, 11)],
        "repoB__pr-2": [],
    }
    res = scorer.score_pr(cases[0]["golden_findings"], run["repoA__pr-1"])
    assert res.tp == 1 and res.fp == 1  # golden matched once; second is an FP


def test_micro_average_pools_across_prs():
    # PR1: 1 TP, 0 FP, 1 FN. PR2: 0 TP, 1 FP, 0 FN.
    c1 = scorer.Counts(tp=1, fp=0, fn=1)
    c2 = scorer.Counts(tp=0, fp=1, fn=0)
    res = scorer.micro_average([c1, c2])
    # pooled tp=1, fp=1, fn=1 -> P=R=F=0.5
    assert res["precision"] == 0.5 and res["recall"] == 0.5 and res["f1"] == 0.5


def test_criterion_mismatch_blocks_match():
    golden = [{"golden_id": "g", "criterion": "security",
               "location": {"file": "x.py", "line_start": 1, "line_end": 3}, "severity": "high"}]
    finding = _finding("performance", "x.py", 2)  # same location, wrong criterion
    res = scorer.score_pr(golden, [finding])
    assert res.tp == 0 and res.fp == 1 and res.fn == 1


def test_aggregate_runs_mean_and_std():
    cases = conftest.make_golden_cases()
    perfect = {
        "repoA__pr-1": [_finding("logic_bug", "a.py", 10, 12), _finding("security", "a.py", 30)],
        "repoB__pr-2": [_finding("runtime_error", "b.go", 40, 44)],
    }
    empty = {"repoA__pr-1": [], "repoB__pr-2": []}
    agg = scorer.aggregate_runs(cases, [perfect, perfect, empty])
    assert agg["n_runs"] == 3 and agg["protocol_ok"] is True
    # F1s = [1.0, 1.0, 0.0] -> mean 2/3, population std = sqrt(2/9)
    assert abs(agg["mean_f1"] - (2 / 3)) < 1e-9
    assert abs(agg["std_f1"] - (2 / 9) ** 0.5) < 1e-9


def test_protocol_flags_under_three_runs():
    cases = conftest.make_golden_cases()
    agg = scorer.aggregate_runs(cases, [{"repoA__pr-1": [], "repoB__pr-2": []}])
    assert agg["protocol_ok"] is False  # <3 runs fails the stability protocol


def test_synthetic_golden_scores_deterministically():
    golden = conftest.load_synthetic_golden()
    cases = golden["cases"]
    # Perfect reviewer for the 6 scored findings across 3 synthetic PRs.
    run = {}
    for case in cases:
        run[case["case_id"]] = [
            {"criterion": g["criterion"], "location": g["location"]}
            for g in case["golden_findings"] if g["severity"] != "low"
        ]
    res = scorer.score_run(cases, run)
    assert res["tp"] == golden["n_golden_scored"] and res["fn"] == 0 and res["fp"] == 0
    assert res["f1"] == 1.0


# --------------------------------------------------------------------------- #
def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
