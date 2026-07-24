"""Canonical tests for the shared answer_scoring primitives.

This is the single home for scoring-primitive regression tests as consumers
migrate off their private copies (handoffs/active/scoring-infra-standardization.md).
Run: `python -m pytest scripts/benchmark/test_answer_scoring.py` (or import + call).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import answer_scoring as s


def test_letter_explicit_and_terse():
    assert s.extract_letter_answer("I think the answer is C.") == "C"
    assert s.extract_letter_answer("C.") == "C"
    assert s.extract_letter_answer("I think C is likely") == ""


def test_letter_verbose_bare_final_line_no_penalty():
    """The bug that manufactured the 2026-07 A4 gpqa deficit: a verbose arm
    that reasons then puts a bare letter on the final line HAS answered."""
    assert s.extract_letter_answer(
        "Let me work through it. This matches option D.\n\nD"
    ) == "D"
    assert s.extract_letter_answer("...so the product is the ether.\n\n**B**") == "B"
    assert s.extract_letter_answer("reasoning...\n(A)") == "A"
    # a genuinely truncated derivation must still fail to parse (not credited)
    assert s.extract_letter_answer(
        "Step 1: balance the redox couple. Step 2: the half reaction for"
    ) == ""


def test_boxed_takes_last_complete():
    # the truncated-fragment bug: an incomplete trailing \boxed{ must be skipped
    assert s.extract_boxed(r"first \boxed{42} then cut \boxed{9") == "42"
    assert s.extract_boxed(r"answer is \boxed{7}") == "7"


def test_score_response_dispatch():
    assert s.score_response("The answer is D.", "D",
                            {"scoring_method": "multiple_choice"})
    assert not s.score_response("The answer is C.", "D",
                               {"scoring_method": "multiple_choice"})
    assert s.score_response(r"so \boxed{70}", "70",
                            {"scoring_method": "math_numeric"})


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
