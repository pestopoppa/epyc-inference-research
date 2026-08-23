#!/usr/bin/env python3
"""Regression tests for benchmark debug scoring.

Since 2026-08-23 this repo's debug_scorer is a delegation shim to the
orchestrator's B7-hardened copy (epyc-root
`handoffs/active/scorer-fork-drift-audit-2026-07-22.md` residual row), so these
tests pin B7 semantics. Where the old pre-B7 fork differed (null `count`
failing closed; numeric-string `count` coercion), B7's behavior is pinned here
instead — the research path must not drift from the canonical scorer.
"""

from debug_scorer import score_answer


def test_programmatic_word_count_null_count_is_vacuous_under_b7():
    # B7: `count = config.get("count") or threshold or 0` — a null count becomes 0,
    # so `at_least` always passes. The pre-B7 fork failed closed here; B7 does not,
    # and the research path delegates to B7.
    assert score_answer(
        answer="one two three",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "word_count",
            "count": None,
            "relation": "at_least",
        },
    )


def test_programmatic_sentence_count_null_count_is_vacuous_under_b7():
    assert score_answer(
        answer="One. Two.",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "sentence_count",
            "count": None,
            "relation": "at_least",
        },
    )


def test_programmatic_word_count_relation_exactly_uses_int_count():
    # B7 compares `wc == count` type-strictly: an int count matches, a numeric
    # string never does. The pre-B7 fork coerced "3" to int; B7 does not.
    assert score_answer(
        answer="one two three",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "word_count",
            "count": 3,
            "relation": "exactly",
        },
    )
    assert not score_answer(
        answer="one two three",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "word_count",
            "count": 4,
            "relation": "exactly",
        },
    )
    assert not score_answer(
        answer="one two three",
        expected="",
        scoring_method="programmatic",
        scoring_config={
            "verifier": "word_count",
            "count": "3",
            "relation": "exactly",
        },
    )
