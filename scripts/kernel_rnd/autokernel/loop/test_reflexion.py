from types import SimpleNamespace as NS

from scripts.kernel_rnd.autokernel.loop import reflexion as rf


def cmp(effect, decisive, floor=0.5):
    return NS(effect=effect, noise_floor_pct=floor, decisive=decisive)


def test_no_comparison_everything_null():
    r = rf.build(None, "abstained")
    assert r["actual_outcome"] is None and r["was_fix_effective"] is None
    assert r["was_diagnosis_correct"] is None and r["expected_outcome"] is None


def test_fix_effective_derived_from_comparison_only():
    assert rf.build(cmp(0.03, True), "kept")["was_fix_effective"] is True
    assert rf.build(cmp(-0.03, True), "regression")["was_fix_effective"] is False
    assert rf.build(cmp(0.001, False), "measured_null")["was_fix_effective"] is False
    assert rf.build(cmp(0.03, None), "x")["was_fix_effective"] is None


def test_diagnosis_needs_declared_expectation():
    assert rf.build(cmp(0.03, True), "kept")["was_diagnosis_correct"] is None
    d = {"expected_outcome": {"direction": "improve", "min_effect_fraction": 0.02}}
    assert rf.build(cmp(0.03, True), "kept", d)["was_diagnosis_correct"] is True
    assert rf.build(cmp(0.01, True), "kept", d)["was_diagnosis_correct"] is False
    assert rf.build(cmp(0.0, False), "measured_null", d)["was_diagnosis_correct"] is False
    nc = {"expected_outcome": {"direction": "no_change", "min_effect_fraction": None}}
    assert rf.build(cmp(0.0, False), "measured_null", nc)["was_diagnosis_correct"] is True
    assert rf.build(cmp(0.03, None), "x", d)["was_diagnosis_correct"] is None


def test_invalid_declaration_is_dropped_not_repaired():
    r = rf.build(cmp(0.03, True), "kept", {"expected_outcome": {"direction": "faster"},
                                           "lessons": ["", "ok"], "avoid_patterns": ["a"]})
    assert r["expected_outcome"] is None and r["was_diagnosis_correct"] is None
    assert r["lessons"] == [] and r["avoid_patterns"] == ["a"]
    assert len(r["lint_violations"]) == 2


def test_accuracy_reports_coverage():
    rows = [{"reflexion": {"was_diagnosis_correct": True}},
            {"reflexion": {"was_diagnosis_correct": False}},
            {"reflexion": {"was_diagnosis_correct": None}},
            {"status": "legacy"}]
    s = rf.diagnosis_accuracy(rows)
    assert (s["rows"], s["known"], s["correct"]) == (3, 2, 1)
    assert s["accuracy"] == 0.5 and abs(s["coverage"] - 2 / 3) < 1e-9
    assert rf.diagnosis_accuracy([])["accuracy"] is None


def test_broadcast_dedups_newest_first_and_bounds():
    rows = [{"reflexion": {"avoid_patterns": ["old"], "try_patterns": ["t"]}},
            {"reflexion": {"avoid_patterns": ["new", "old"]}}]
    v = rf.broadcast_view(rows)
    assert v["avoid_patterns"] == ["new", "old"] and v["try_patterns"] == ["t"]
    assert rf.broadcast_view(rows, limit=1)["try_patterns"] == []
