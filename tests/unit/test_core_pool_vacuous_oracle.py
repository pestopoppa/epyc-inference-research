"""The pool-build guard against oracles satisfied by echoing the input.

Origin: epyc-root `artifacts/audit/debugbench-oracle-vacuity-20260812.md` (`mainC`,
2026-08-12). Every debugbench row in both core pools carried an `expected` that was a
byte-exact 100-character PREFIX of the upstream reference solution, scored `substring`.
That prefix was already present in the buggy code the model was handed, on 4 of 4 rows,
so a model that changed nothing and echoed its input scored a PASS. Corpus-wide the
construction is vacuous on 3,233 of 4,250 upstream rows (76.1%).

The test that matters is NOT "is `expected` short or uninformative" — a longer prefix is
just as vacuous. It is "is the oracle already satisfied by the input?"
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "autopilot"), str(REPO / "src"), str(REPO)]
_spec = importlib.util.spec_from_file_location(
    "cvs", REPO / "scripts" / "autopilot" / "core_v2_select.py")
cvs = importlib.util.module_from_spec(_spec)
sys.modules["cvs"] = cvs
_spec.loader.exec_module(cvs)


def _row(**kw):
    base = {"id": "r1", "suite": "s", "scoring_method": "substring"}
    base.update(kw)
    return base


def test_the_debugbench_shape_is_caught() -> None:
    """The real defect: expected is a long prefix of the code the model was handed."""
    solution = "class Solution { public int atoms(String f) { Map<String,Integer> m = new HashMap<>();"
    hits = cvs.vacuous_rows([_row(expected=solution[:60], buggy_code=solution + " /*bug*/ }")])
    assert len(hits) == 1
    assert hits[0]["severity"] == "structural"


def test_a_genuine_oracle_is_not_flagged() -> None:
    """The compliant path — this must not forbid its own idiom."""
    assert cvs.vacuous_rows([_row(expected="42", prompt="What is six times seven?")]) == []


def test_short_incidental_containment_is_reported_but_not_structural() -> None:
    """A 1-char answer appears in almost any prompt; that is chance, not a broken oracle."""
    hits = cvs.vacuous_rows([_row(expected="4", prompt="Compute 2+2 given 4 options")])
    assert [h["severity"] for h in hits] == ["incidental"]


def test_exact_scorer_is_not_made_vacuous_by_containment() -> None:
    """An exact scorer does not inherit substring containment semantics."""
    assert cvs.vacuous_rows(
        [_row(expected="x" * 80, prompt="x" * 200, scoring_method="exact")]) == []


def test_programmatic_echo_uses_the_existing_verifier() -> None:
    row = _row(expected="", scoring_method="programmatic",
               scoring_config={"verifier": "non_empty"}, prompt="copy this input")
    hits = cvs.vacuous_rows([row])
    assert [hit["severity"] for hit in hits] == ["structural"]
    assert hits[0]["reason"] == "programmatic_verifier_accepts_input_echo"


def test_programmatic_echo_that_fails_the_verifier_is_not_flagged() -> None:
    row = _row(expected="unused", scoring_method="programmatic",
               scoring_config={"verifier": "word_count_min", "threshold": 3},
               prompt="only two")
    assert cvs.vacuous_rows([row]) == []


def test_programmatic_verifier_errors_are_unresolved_not_safe() -> None:
    row = _row(expected="unused", scoring_method="programmatic",
               scoring_config={"verifier": "future_unknown"}, prompt="copy this input")
    hits = cvs.vacuous_rows([row])
    assert [hit["severity"] for hit in hits] == ["unresolved"]
    assert hits[0]["reason_type"] == "ValueError"


def test_a_longer_prefix_is_not_a_fix() -> None:
    """Guards the tempting wrong repair: widening 100 -> 500 keeps it vacuous."""
    body = "y" * 900
    hits = cvs.vacuous_rows([_row(expected=body[:500], buggy_code=body)])
    assert hits and hits[0]["severity"] == "structural"


def test_missing_or_empty_expected_is_ignored_not_crashed() -> None:
    assert cvs.vacuous_rows([_row(prompt="p"), _row(expected="", prompt="p"),
                             _row(expected=None, prompt="p")]) == []


# ---------------------------------------------------------------------------
# constant_oracle_suites — the sibling guard.
#
# Origin: epyc-root `artifacts/audit/unscoreable-rows-livecodebench-cruxeval-mah-20260812.md`
# (`mainD`, 2026-08-12). All 2,360 `livecodebench` rows carry expected == "def ", 2,349 of
# them scored by `substring`. `vacuous_rows` reports nothing actionable for them: the needle
# is in the prompt of only 16 of 2,349 and is 4 chars, i.e. below _STRUCTURAL_MIN. A
# different property, needing a different test — an oracle constant across a suite grades
# response FORMAT, not the answer.
# ---------------------------------------------------------------------------


def _suite(suite, n, expected, method="substring", **kw):
    return [dict({"id": f"{suite}-{i}", "suite": suite, "expected": expected,
                  "scoring_method": method}, **kw) for i in range(n)]


def test_the_livecodebench_shape_is_caught() -> None:
    """The real defect: one four-character oracle shared by every row in the suite."""
    hits = cvs.constant_oracle_suites(_suite("livecodebench", 2360, "def "))
    assert [h["suite"] for h in hits] == ["livecodebench"]
    assert hits[0]["expected"] == "def " and hits[0]["rows"] == 2360


def test_vacuous_rows_does_not_already_catch_it() -> None:
    """States the gap this guard exists to close. If this ever fails, one of the two
    guards has drifted and the docstrings above are lying."""
    rows = _suite("livecodebench", 3, "def ", prompt="Write a Python function to solve.")
    assert cvs.vacuous_rows(rows) == []


def test_a_varying_oracle_is_not_flagged() -> None:
    """The compliant path — this must not forbid its own idiom."""
    rows = [dict(r, expected=f"answer-{i}")
            for i, r in enumerate(_suite("math", 40, "placeholder"))]
    assert cvs.constant_oracle_suites(rows) == []


def test_constant_expected_with_the_oracle_elsewhere_is_left_alone() -> None:
    """bigcodebench/usaco: expected is constant but scoring is code_execution and the real
    oracle is per-row `test_code`. Flagging these trains readers to ignore the field."""
    rows = [dict(r, scoring_config={"test_code": f"assert f({i})"})
            for i, r in enumerate(
                _suite("bigcodebench", 1140, "task_func", method="code_execution"))]
    assert cvs.constant_oracle_suites(rows) == []


def test_small_suites_are_not_judged() -> None:
    """On a 50-row core pool a suite contributes 1-3 rows and constancy is coincidence.
    Guards against computing this over `emitted` instead of the source pool — an input too
    small to disagree passes any predicate."""
    assert cvs.constant_oracle_suites(_suite("tiny", 3, "def ")) == []
    assert cvs.constant_oracle_suites(
        _suite("tiny", cvs._CONSTANT_ORACLE_MIN_ROWS, "def "))


def test_empty_expected_across_a_suite_is_not_a_constant_oracle() -> None:
    """`usaco` and `instruction_precision` carry expected=='' everywhere; the needle lives
    in `scoring_config`. An empty oracle is a different defect, already gated upstream."""
    assert cvs.constant_oracle_suites(_suite("usaco", 520, "")) == []


def test_write_core_jsonl_records_the_flag_and_scopes_it_to_drawn_suites(tmp_path) -> None:
    """Verify THE consumer, not the helper: the finding has to reach the artifact, and a
    warning about a suite this core never draws from is noise that gets the field ignored."""
    pool = {("livecodebench", f"q{i}"): dict(r, id=f"q{i}")
            for i, r in enumerate(_suite("livecodebench", 30, "def "))}
    pool.update({("needle_parameterized", f"n{i}"): dict(r, id=f"n{i}")
                 for i, r in enumerate(_suite("needle_parameterized", 30, "the needle"))})
    report = {"generated_at": "2026-08-12T00:00:00Z",
              "parameters": {"target_size": 1}, "selected_count": 1}

    cvs.write_core_jsonl(
        path=tmp_path / "core.jsonl", core_id="c",
        selected=[cvs.ItemStats(qid="q0", suite="livecodebench", attempts=3, correct=2)],
        pool_lookup=pool, report=report)

    flagged = report["constant_oracle_suites"]
    assert [e["suite"] for e in flagged] == ["livecodebench"], (
        "needle_parameterized is constant too, but this core does not draw from it")
    assert flagged[0]["expected"] == "def "


def test_suites_are_partitioned_not_pooled() -> None:
    """Two suites that each vary internally must not be flagged because the union happens
    to look constant, and one broken suite must not be masked by a healthy neighbour."""
    rows = _suite("livecodebench", 30, "def ") + [
        dict(r, expected=f"a-{i}") for i, r in enumerate(_suite("math", 30, "x"))]
    assert [h["suite"] for h in cvs.constant_oracle_suites(rows)] == ["livecodebench"]
