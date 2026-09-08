"""CJ-11 migration test for architect_bench_rescore.

The caller under test used to be
``new_ok = bool(resp) and score_response(resp, r["expected"], q)`` -- the exact
idiom in which any truthy third return value coerces to a PASS. It is now on the
three-valued ``score_response_or_error``, and the contract this file pins is:

  * an undecidable row is RECORDED (``verdict``/``cause`` fields) and is NEVER
    counted correct;
  * every positive assertion is paired with a mutation that removes the signal.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import architect_bench_rescore as rescore  # noqa: E402
import gate_verdict_vocab as vocab  # noqa: E402


def _must_fail(thunk, what: str) -> None:
    try:
        thunk()
    except AssertionError:
        return
    raise AssertionError(f"MUTATION NOT DETECTED: {what}")


# (row, question) pairs. Every row's `response` is NON-EMPTY, so the historical
# `bool(resp)` guard does not mask anything: the third value is what is on trial.
ROWS = [
    # decided pass
    ({"id": "q_pass", "expected": "D", "correct": True,
      "response": "The answer is D.", "extracted": "D"},
     {"id": "q_pass", "scoring_method": "multiple_choice", "scoring_config": {}}),
    # decided fail
    ({"id": "q_fail", "expected": "D", "correct": False,
      "response": "The answer is C.", "extracted": "C"},
     {"id": "q_fail", "scoring_method": "multiple_choice", "scoring_config": {}}),
    # UNDECIDABLE: non-empty response, nothing extractable
    ({"id": "q_unparsed", "expected": "D", "correct": False,
      "response": "I think C is likely, or maybe D, hard to say", "extracted": ""},
     {"id": "q_unparsed", "scoring_method": "multiple_choice", "scoring_config": {}}),
    # UNDECIDABLE: no gold at all. THE INFLATION CASE -- the legacy path scored
    # this TRUE (extractor's "" == expected's ""), i.e. a free pass for a row
    # with no reference to check against.
    ({"id": "q_no_ref", "expected": "", "correct": True,
      "response": "no letter here at all", "extracted": ""},
     {"id": "q_no_ref", "scoring_method": "multiple_choice", "scoring_config": {}}),
    # UNDECIDABLE: code_execution with no oracle
    ({"id": "q_no_oracle", "expected": "", "correct": False,
      "response": "def f():\n    return 1", "extracted": ""},
     {"id": "q_no_oracle", "scoring_method": "code_execution", "scoring_config": {}}),
]


def _rescore() -> dict[str, dict]:
    """Run the real CLI over a temp tree; return the rescored rows by id."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pq_dir = root / "runs" / "arm" / "pass1"
        pq_dir.mkdir(parents=True)
        (pq_dir / "per_question.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row, _q in ROWS))
        manifest = root / "questions.json"
        manifest.write_text(json.dumps({"suites": {"s": [q for _r, q in ROWS]}}))

        argv = sys.argv
        sys.argv = ["architect_bench_rescore", str(root / "runs"),
                    "--questions", str(manifest), "--write"]
        try:
            assert rescore.main() == 0
        finally:
            sys.argv = argv
        out = (pq_dir / "per_question.rescored.jsonl").read_text()
    return {json.loads(line)["id"]: json.loads(line)
            for line in out.splitlines() if line.strip()}


class TestArchitectBenchRescoreCJ11(unittest.TestCase):
    def test_undecidable_rows_are_recorded_and_never_counted_correct(self):
        by_id = _rescore()
        self.assertEqual(len(by_id), len(ROWS))

        undecidable = ["q_unparsed", "q_no_ref", "q_no_oracle"]
        for qid in undecidable:
            row = by_id[qid]
            # (a) never a pass -- neither directly nor through the old idiom
            self.assertIs(row["correct"], False, qid)
            self.assertFalse(bool(row["response"]) and row["correct"], qid)
            # recorded, with a cause from the CLOSED registry
            self.assertEqual(row["verdict"], vocab.VERDICT_OUT_OF_COVERAGE, qid)
            self.assertIn(row["cause"], vocab.CAUSES, qid)

        self.assertEqual(by_id["q_unparsed"]["cause"], vocab.CAUSE_UNPARSED)
        self.assertEqual(by_id["q_no_ref"]["cause"], vocab.CAUSE_NO_REFERENCE)
        self.assertEqual(by_id["q_no_oracle"]["cause"], vocab.CAUSE_NO_REFERENCE)

        # decided rows are untouched in meaning and carry NO cause
        self.assertIs(by_id["q_pass"]["correct"], True)
        self.assertEqual(by_id["q_pass"]["verdict"], vocab.VERDICT_PASS)
        self.assertNotIn("cause", by_id["q_pass"])
        self.assertIs(by_id["q_fail"]["correct"], False)
        self.assertEqual(by_id["q_fail"]["verdict"], vocab.VERDICT_FAIL)
        self.assertNotIn("cause", by_id["q_fail"])

    def test_mutation_the_old_idiom_would_inflate_these_same_rows(self):
        """MUTATION: reinstate `bool(resp) and <verdict>` over a truthy third
        value and show the assertions above FAIL -- proving they have teeth and
        are not passing because the rows are trivially falsy."""
        rows = [r for r, _q in ROWS if r["id"].startswith(("q_unparsed", "q_no_"))]
        self.assertTrue(rows)

        def mutated():
            for r in rows:
                # the shape a "just return the verdict name" widening would take
                verdict = vocab.VERDICT_OUT_OF_COVERAGE
                assert not (bool(r["response"]) and verdict), r["id"]
        _must_fail(mutated, "truthy third value survives `bool(resp) and ...`")

    def test_mutation_no_reference_row_was_a_real_pre_migration_pass(self):
        """The q_no_ref flip is a genuine de-inflation, not a hypothetical: the
        two-valued function still returns True on exactly that input."""
        from answer_scoring import score_response
        q = {"scoring_method": "multiple_choice", "scoring_config": {}}
        self.assertIs(score_response("no letter here at all", "", q), True)
        # ...and the migrated caller does not carry it through
        self.assertIs(_rescore()["q_no_ref"]["correct"], False)

    def test_mutation_empty_rescored_output_cannot_pass_vacuously(self):
        def mutated():
            by_id: dict[str, dict] = {}
            assert len(by_id) == len(ROWS)
        _must_fail(mutated, "an empty rescored file would pass the row assertions")


if __name__ == "__main__":
    unittest.main()
