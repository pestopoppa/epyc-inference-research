#!/usr/bin/env python3
"""CJ-8 contract tests for the v7 kernel quality-gate comparator.

Before CJ-8 this comparator had one bit of output. Three unrelated events shared
it:

  * a real REGRESSION,
  * INSUFFICIENT evidence (`candidate_n < min_n`) — which `check_suite`'s own
    docstring called "advisory, not blocking" while the code blocked on it,
  * a suite MISSING from either input file,

and a fourth event was worse than mislabelled: `bl.get("accuracy", 0)` read a
MISSING baseline accuracy as a measured 0.0, which made every delta positive and
**silently passed the suite**. A corrupt baseline file disarmed the gate that
exists to stop a quality-losing kernel from promoting.

The tests below pin, in order: that nothing which used to block now promotes,
that the previously-passing disarm now blocks, and that a real regression is
still exactly a regression.

Runs under pytest or standalone (`python3 test_v7_quality_gate_compare.py`) —
the research repo ships no test infra.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_SPEC = importlib.util.spec_from_file_location(
    "v7_quality_gate_compare", _HERE / "v7_quality_gate_compare.py")
v7 = importlib.util.module_from_spec(_SPEC)
sys.modules["v7_quality_gate_compare"] = v7
_SPEC.loader.exec_module(v7)

PASS, FAIL, OOC = v7.PASS, v7.FAIL, v7.OUT_OF_COVERAGE


def _suites(*rows):
    return {"suites": list(rows)}


def _row(suite, acc=None, n=None):
    r = {"suite": suite}
    if acc is not None:
        r["accuracy"] = acc
    if n is not None:
        r["n"] = n
    return r


def _cmp(baseline, candidate, threshold=0.05, min_n=50):
    return v7.compare(baseline, candidate, threshold, min_n)


# --------------------------------------------------------------------------- #
# THE SAFETY RULE: every previously-blocking input still blocks
# --------------------------------------------------------------------------- #
def test_every_undecidable_input_still_exits_non_zero():
    cases = {
        "insufficient_n": (_suites(_row("g", 0.60, 200)), _suites(_row("g", 0.60, 10))),
        "missing_candidate_accuracy": (
            _suites(_row("g", 0.60, 200)), _suites(_row("g", n=200))),
        "missing_baseline_accuracy": (
            _suites(_row("g", n=200)), _suites(_row("g", 0.60, 200))),
        "suite_absent_from_candidate": (
            _suites(_row("g", 0.60, 200)), _suites()),
        "suite_absent_from_baseline": (
            _suites(), _suites(_row("g", 0.60, 200))),
        "no_suites_at_all": (_suites(), _suites()),
    }
    for label, (bl, cand) in cases.items():
        _, _, verdict, _ = _cmp(bl, cand)
        assert verdict == OOC, f"{label}: expected out-of-coverage, got {verdict}"
        assert v7.gvv.EXIT_BY_VERDICT[verdict] != 0, (
            f"{label}: an undecidable input MUST keep blocking")


def test_missing_baseline_accuracy_no_longer_silently_passes():
    """The gate-disarming bug. `bl.get('accuracy', 0)` made delta positive."""
    _, _, verdict, text = _cmp(_suites(_row("g", n=200)), _suites(_row("g", 0.365, 200)))
    assert verdict != PASS
    assert verdict == OOC
    assert "NOT ASSESSABLE" in text
    # Mutation guard: with the accuracy PRESENT the same shape passes, so the
    # assertion above is about the absent key and not about the numbers.
    _, _, ok, _ = _cmp(_suites(_row("g", 0.36, 200)), _suites(_row("g", 0.365, 200)))
    assert ok == PASS


# --------------------------------------------------------------------------- #
# A decision is still a decision
# --------------------------------------------------------------------------- #
def test_real_regression_is_still_fail_exit_1():
    rows, _, verdict, _ = _cmp(_suites(_row("g", 0.60, 200)), _suites(_row("g", 0.40, 200)))
    assert verdict == FAIL
    assert v7.gvv.EXIT_BY_VERDICT[verdict] == 1
    assert rows[0]["cause"] is None, "a DECIDED verdict must carry no cause code"
    assert "REGRESSION" in rows[0]["status"]


def test_clean_comparison_is_still_pass_exit_0():
    _, summary, verdict, _ = _cmp(
        _suites(_row("a", 0.60, 200), _row("b", 0.30, 200)),
        _suites(_row("a", 0.62, 200), _row("b", 0.29, 200)))
    assert verdict == PASS
    assert v7.gvv.EXIT_BY_VERDICT[verdict] == 0
    assert summary["resolved_coverage"] == 1.0


def test_a_decided_regression_outranks_an_undecided_suite():
    """`fail` wins over `out-of-coverage`: a suite that ran and rejected the
    candidate is the finding; the unreadable remainder is noise beside it."""
    _, _, verdict, _ = _cmp(
        _suites(_row("a", 0.60, 200), _row("b", 0.60, 200)),
        _suites(_row("a", 0.40, 200), _row("b", 0.60, 10)))
    assert verdict == FAIL


# --------------------------------------------------------------------------- #
# The names and the causes
# --------------------------------------------------------------------------- #
def test_insufficient_evidence_is_no_longer_counted_as_a_regression():
    rows, summary, verdict, text = _cmp(
        _suites(_row("g", 0.60, 200)), _suites(_row("g", 0.60, 10)))
    assert verdict == OOC
    assert rows[0]["cause"] == v7.gvv.CAUSE_INSUFFICIENT_COVERAGE
    assert summary["n_fail"] == 0, "a thin run is NOT a suite the candidate failed"
    assert summary["n_out_of_coverage"] == 1
    assert "NOT a regression finding" in rows[0]["status"]
    assert "BLOCKS promotion" in text


def test_absent_accuracy_and_measured_zero_are_different_events():
    """The collapse that made a serialization defect readable as a 0% score."""
    absent_rows, _, absent_verdict, _ = _cmp(
        _suites(_row("g", 0.60, 200)), _suites(_row("g", n=200)))
    zero_rows, _, zero_verdict, _ = _cmp(
        _suites(_row("g", 0.60, 200)), _suites(_row("g", 0.0, 200)))

    assert absent_verdict == OOC
    assert absent_rows[0]["cause"] == v7.gvv.CAUSE_ABSENT
    assert absent_rows[0]["candidate_acc"] is None
    # Mutation guard: a GENUINE zero is still a decided regression. Without this
    # the test above would pass for a comparator that refused everything.
    assert zero_verdict == FAIL
    assert zero_rows[0]["candidate_acc"] == 0.0


def test_check_suite_returns_a_cause_only_when_undecided():
    decided = [
        v7.check_suite(0.6, 0.6, 200, 200, 0.05, 50),
        v7.check_suite(0.6, 0.1, 200, 200, 0.05, 50),
    ]
    for verdict, cause, _ in decided:
        assert verdict in (PASS, FAIL)
        assert cause is None
    verdict, cause, _ = v7.check_suite(0.6, 0.6, 200, 5, 0.05, 50)
    assert verdict == OOC and cause is not None


def test_report_marks_an_undecided_suite_differently_from_a_failed_one():
    rows, summary, _, text = _cmp(
        _suites(_row("a", 0.60, 200), _row("b", 0.60, 200)),
        _suites(_row("a", 0.20, 200), _row("b", 0.60, 5)))
    md = v7.render_markdown(rows, summary, text, {}, {})
    assert "✗" in md and "?" in md, "a thin suite must not wear the failure glyph"
    assert "Not decided (out-of-coverage): 1" in md
    assert "insufficient_coverage=1" in md


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
