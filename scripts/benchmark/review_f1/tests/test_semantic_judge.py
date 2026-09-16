"""Tests for the EV-13b semantic matcher (mock judge; NO inference)."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import semantic_judge as sj  # noqa: E402

CASE = {
    "case_id": "repo__pr-1",
    "pr_ref": {"title": "t", "diff_path": "d.diff"},
    "golden_findings": [
        {"golden_id": "repo__pr-1-g0", "comment": "null deref in parse()", "severity": "high"},
        {"golden_id": "repo__pr-1-g1", "comment": "off by one in loop", "severity": "medium"},
        {"golden_id": "repo__pr-1-g2", "comment": "typo in log", "severity": "low"},
    ],
}
CASE2 = {
    "case_id": "repo__pr-2", "pr_ref": {"title": "u"},
    "golden_findings": [{"golden_id": "repo__pr-2-g0", "comment": "race on cache", "severity": "high"}],
}
DIFF = "--- a/src/x.py\n+++ b/src/x.py\n@@ -10,3 +20,5 @@\n+a\n"


def test_parse_maps_shuffled_numbers_and_counts_invalid():
    order = sj.golden_order(CASE, 42, 0)
    out = sj.parse_judge('```json\n{"matches": [1, 9, "x"], "confidence": "high"}\n```', order)
    assert out["parse_ok"] and out["matches"] == [order[0]["golden_id"]] and out["invalid_index"] == 2
    assert sj.parse_judge("no json", order)["parse_ok"] is False


def test_validity_window():
    files = sj.touched(DIFF)
    assert files == {"src/x.py": [(20, 24)]}
    loc = lambda s, e: {"location": {"file": "x.py", "line_start": s, "line_end": e}}  # noqa: E731
    assert sj.validity(loc(30, 31), files) == (True, True)     # within +10
    assert sj.validity(loc(40, 41), files) == (True, False)
    assert sj.validity({"location": {"file": "other.py", "line_start": 1}}, files) == (False, False)


def test_assignment_tp_duplicate_neutral_fp_fn():
    edges = [["repo__pr-1-g0"], ["repo__pr-1-g0"], ["repo__pr-1-g2"], [], ["repo__pr-1-g0", "repo__pr-1-g1"]]
    c = sj.assign(CASE, edges)
    # f0->g0, f4->g1 (augmenting), f1 duplicate FP, f2 neutral, f3 FP; no FN
    assert c == {"tp": 2, "fp": 2, "fn": 0, "duplicate_fp": 1, "neutral_low": 1}


def _mock_call(truth):
    """truth(comment) -> set of golden comments it should match."""
    def call(messages, seed):
        user = messages[1]["content"]
        comment = user.split("):\n", 1)[1].split("\n\nGOLDEN COMMENTS:")[0]
        goldens = re.findall(r"^\[(\d+)\] (.*)$", user, flags=re.M)
        hits = [int(n) for n, text in goldens if text in truth(comment)]
        return json.dumps({"matches": hits, "confidence": "high", "rationale": "x"})
    return call


def test_calibration_controls_pass_with_a_faithful_judge():
    def truth(comment):
        return {g["comment"] for c in (CASE, CASE2) for g in c["golden_findings"]
                if comment == sj.paraphrase(g["comment"])}
    res = sj.calibrate([CASE, CASE2], _mock_call(truth), 42, workers=2)
    assert res["n_positive"] == 3 and res["n_negative"] == 2
    assert res["valid"] and res["positive_rate"] == 1.0 and res["negative_rate"] == 1.0


def test_judge_and_score_end_to_end(tmp_path):
    root = tmp_path / "results" / "reader__q8"
    root.mkdir(parents=True)
    runs = [[{"comment": "null deref in parse()", "location": {"file": "src/x.py", "line_start": 21}},
             {"comment": "unrelated", "location": None}]] * 3
    (root / "repo__pr-1.json").write_text(json.dumps({"runs": runs}))
    (tmp_path / "d.diff").write_text(DIFF)
    exact = lambda c: {c}  # noqa: E731
    sj.judge_all([CASE], root, "judge__q4", _mock_call(exact), tmp_path, workers=2)
    s = sj.score([CASE], root, "judge__q4")
    assert s["n_runs"] == 3 and s["protocol_ok"]
    r0 = s["per_run"][0]
    assert (r0["tp"], r0["fp"], r0["fn"]) == (1, 1, 1)
    assert r0["location_validity_rate"] == 0.5 and not r0["malfunction"]
    assert abs(s["mean_f1"] - 0.5) < 1e-9 and s["std_f1"] == 0.0


def test_parse_failures_mark_malfunction(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "repo__pr-1.json").write_text(json.dumps({"runs": [[{"comment": "a"}]]}))
    sj.judge_all([CASE], root, "j__q", lambda m, s: "garbage", None, workers=1)
    s = sj.score([CASE], root, "j__q")
    assert s["per_run"][0]["judge_parse_fail_rate"] == 1.0 and s["per_run"][0]["malfunction"]
    assert s["n_runs"] == 0
