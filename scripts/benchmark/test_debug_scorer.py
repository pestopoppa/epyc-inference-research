#!/usr/bin/env python3
"""Regression tests for benchmark debug scoring."""

from debug_scorer import score_answer


def test_programmatic_word_count_null_count_fails_closed():
    assert not score_answer(
        answer="one two three",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "word_count",
            "count": None,
            "relation": "at_least",
        },
    )


def test_programmatic_sentence_count_null_count_fails_closed():
    assert not score_answer(
        answer="One. Two.",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "sentence_count",
            "count": None,
            "relation": "at_least",
        },
    )


def test_programmatic_word_count_relation_accepts_numeric_string():
    assert score_answer(
        answer="one two three",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "word_count",
            "count": "3",
            "relation": "exactly",
        },
    )
