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


def _force_surface_matching():
    """Pin the no-spaCy fallback so these tests mean the same thing on every host."""
    s._SPACY_NLP = None
    s._SPACY_TRIED = True


def _naive_lemmas(text):
    """Deterministic toy lemmatizer: clean tokenize + strip ing/ed/s suffixes."""
    import re
    return [re.sub(r"(ing|ed|s)$", "", w) for w in re.findall(r"[a-z]+", text.lower())]


def test_ordered_subsequence_basic_directions():
    _force_surface_matching()
    # in order: both metrics saturate
    r = s.score_ordered_subsequence(
        "First we mix the acid, then we heat the flask, finally we titrate.",
        ["acid", "heat", "titrate"])
    assert r["all_in_order"] and r["coverage"] == 1.0 and r["coverage_in_order"] == 1.0
    # all present but order broken: coverage stays 1.0, ordered metrics drop —
    # the two-metric divergence the row exists to capture
    r = s.score_ordered_subsequence(
        "We titrate, after heating, having mixed the acid first.",
        ["acid", "heat", "titrate"], lemmatizer=_naive_lemmas)
    assert r["coverage"] == 1.0 and not r["all_in_order"] and r["coverage_in_order"] < 1.0


def test_ordered_subsequence_partial_and_missing():
    _force_surface_matching()
    r = s.score_ordered_subsequence("only the acid appears", ["acid", "heat", "titrate"])
    assert not r["all_in_order"] and r["missing"] == ["heat", "titrate"]
    assert abs(r["coverage"] - 1 / 3) < 1e-9 and abs(r["coverage_in_order"] - 1 / 3) < 1e-9


def test_ordered_subsequence_multiword_contiguous():
    _force_surface_matching()
    ok = s.score_ordered_subsequence(
        "compute the energy level then the Larmor precession",
        ["energy level", "larmor precession"])
    assert ok["all_in_order"]
    # split multi-word must NOT match; hyphens split like spaces on both paths
    split = s.score_ordered_subsequence(
        "the energy of this level", ["energy level"])
    assert split["missing"] == ["energy level"] and split["coverage"] == 0.0
    hyph = s.score_ordered_subsequence("the energy-level diagram", ["energy level"])
    assert hyph["all_in_order"]


def test_ordered_subsequence_early_mention_never_shadows():
    _force_surface_matching()
    # 'c' appears early (out of order) AND again after 'a' — the DP must find
    # the in-order assignment, not greedily bind the first occurrence
    r = s.score_ordered_subsequence("c comes early, then a, then c again", ["a", "c"])
    assert r["all_in_order"]


def test_ordered_subsequence_duplicates_need_repeats():
    _force_surface_matching()
    assert s.score_ordered_subsequence("a b a", ["a", "a"])["all_in_order"]
    assert not s.score_ordered_subsequence("a b", ["a", "a"])["all_in_order"]


def test_ordered_subsequence_empty_concepts_refused():
    # the vacuity guard: empty config must raise, never score 1.0
    _force_surface_matching()
    try:
        s.score_ordered_subsequence("anything", [])
        assert False, "empty concept list must be refused"
    except ValueError:
        pass
    # dispatch path fails closed too: a suite row missing its concepts raises
    try:
        s.score_response("anything", "", {"scoring_method": "ordered_subsequence",
                                          "scoring_config": {}})
        assert False, "dispatch must not silently pass a bad config"
    except ValueError:
        pass


def test_ordered_subsequence_injected_lemmatizer_and_conservative_fallback():
    _force_surface_matching()
    text = "he runs the tests then ships the build"
    naive = lambda t: [w[:-1] if w.endswith("s") else w
                       for w in t.lower().replace(",", " ").split()]
    assert s.score_ordered_subsequence(text, ["run", "test", "ship"],
                                       lemmatizer=naive)["all_in_order"]
    # without lemmatization the inflected forms miss — the CONSERVATIVE direction
    # (never a false match), flagged so a reader can tell which regime scored
    r = s.score_ordered_subsequence(text, ["run", "test", "ship"])
    assert not r["all_in_order"] and r["lemmatized"] is False


def test_ordered_subsequence_dispatch_binary_arm():
    _force_surface_matching()
    q = {"scoring_method": "ordered_subsequence",
         "scoring_config": {"concepts": ["mix", "heat"]}}
    assert s.score_response("mix then heat", "", q)
    assert not s.score_response("heat then mix", "", q)


# ── CJ-11: score_response_or_error (three-valued sibling) ────────────────────
# Every positive assertion below is paired with a MUTATION that removes the
# signal it claims to test, so a check that would pass vacuously fails loudly.

import gate_verdict_vocab as vocab  # noqa: E402


def _must_fail(thunk, what: str):
    """Run `thunk` (a mutated variant) and require it to raise AssertionError.

    A test that passes under its own mutation is testing nothing.
    """
    try:
        thunk()
    except AssertionError:
        return
    raise AssertionError(f"MUTATION NOT DETECTED: {what}")


# (response, expected, q, expected_cause) — one row per undecidable shape.
UNDECIDABLE_CASES = [
    ("", "D", {"scoring_method": "multiple_choice"}, vocab.CAUSE_EMPTY),
    ("   \n ", "D", {"scoring_method": "multiple_choice"}, vocab.CAUSE_EMPTY),
    ("I think C is likely, or maybe D, hard to say",
     "D", {"scoring_method": "multiple_choice"}, vocab.CAUSE_UNPARSED),
    ("The answer is D.", "  ", {"scoring_method": "multiple_choice"},
     vocab.CAUSE_NO_REFERENCE),
    ("reasoning with no boxed answer", "42", {"scoring_method": "math_numeric"},
     vocab.CAUSE_UNPARSED),
    (r"so \boxed{42}", "not-a-number-at-all", {"scoring_method": "math_numeric"},
     vocab.CAUSE_NO_REFERENCE),
    ("anything", "D", {"scoring_method": "vibes_based"}, vocab.CAUSE_UNSUPPORTED),
    ("anything", "", {"scoring_method": "ordered_subsequence",
                      "scoring_config": {}}, vocab.CAUSE_NO_REFERENCE),
    ("def f(): pass", "", {"scoring_method": "code_execution",
                           "scoring_config": {}}, vocab.CAUSE_NO_REFERENCE),
    ("def f(): pass", "", {"scoring_method": "code_execution",
                           "scoring_config": {"test": "check(f)"}},
     vocab.CAUSE_NO_REFERENCE),
]

DECIDED_CASES = [
    ("The answer is D.", "D", {"scoring_method": "multiple_choice"}, True),
    ("The answer is C.", "D", {"scoring_method": "multiple_choice"}, False),
    (r"so \boxed{70}", "70", {"scoring_method": "math_numeric"}, True),
    (r"so \boxed{71}", "70", {"scoring_method": "math_numeric"}, False),
    ("mix then heat", "", {"scoring_method": "ordered_subsequence",
                           "scoring_config": {"concepts": ["mix", "heat"]}}, True),
    ("heat then mix", "", {"scoring_method": "ordered_subsequence",
                           "scoring_config": {"concepts": ["mix", "heat"]}}, False),
    ("42", "42", {"scoring_method": "exact_match"}, True),
    ("43", "42", {"scoring_method": "exact_match"}, False),
]


def test_or_error_causes_are_from_the_closed_registry():
    _force_surface_matching()
    for response, expected, q, want in UNDECIDABLE_CASES:
        verdict, cause = s.score_response_or_error(response, expected, q)
        assert verdict is None, (q, response, verdict)
        assert cause == want, (q, response, cause, want)
        assert cause in vocab.CAUSES, cause
    # MUTATION: a free-text cause -- the shape the seeding precedent returns --
    # must NOT satisfy the closed-registry membership check.
    def mutated():
        cause = "scoring_unavailable: boom"
        assert cause in vocab.CAUSES, cause
    _must_fail(mutated, "membership check accepts a free-text cause")


def test_or_error_decided_cases_carry_no_cause():
    _force_surface_matching()
    for response, expected, q, want in DECIDED_CASES:
        verdict, cause = s.score_response_or_error(response, expected, q)
        assert cause is None, (q, response, cause)
        assert verdict is want, (q, response, verdict, want)


def test_undecidable_never_becomes_a_pass_through_the_bool_and_idiom():
    """(a) THE load-bearing property.

    The live call idiom is `bool(resp) and score_response(...)`. A truthy third
    value would coerce to a PASS there and INFLATE quality. Prove that the
    three-valued verdict cannot do that at any undecidable input — including the
    ones where the response is NON-empty, so `bool(resp)` does not save us.
    """
    _force_surface_matching()
    non_empty = [c for c in UNDECIDABLE_CASES if c[0].strip()]
    assert non_empty, "corpus must exercise the non-empty branch of `bool(resp)`"
    for response, expected, q, _cause in UNDECIDABLE_CASES:
        verdict, cause = s.score_response_or_error(response, expected, q)
        assert not (bool(response) and verdict), (q, response, verdict)
        assert verdict is not True

    # MUTATION: replace the None with the string the vocabulary uses for the
    # undecided verdict -- a plausible "just return the verdict name" design.
    # It is TRUTHY, so the same assertion must now fail.
    def mutated():
        for response, expected, q, _cause in UNDECIDABLE_CASES:
            verdict = vocab.VERDICT_OUT_OF_COVERAGE  # truthy third value
            assert not (bool(response) and verdict), (q, response, verdict)
    _must_fail(mutated, "truthy third value survives the `bool(resp) and` idiom")


def test_or_error_verdict_agrees_with_score_response_wherever_decided():
    """(b) The sibling never re-decides an item score_response already decides."""
    _force_surface_matching()
    for response, expected, q, _want in DECIDED_CASES:
        verdict, cause = s.score_response_or_error(response, expected, q)
        assert cause is None
        assert verdict == s.score_response(response, expected, q)

    # MUTATION: negate the reference and require disagreement to be detected.
    def mutated():
        for response, expected, q, _want in DECIDED_CASES:
            verdict, _ = s.score_response_or_error(response, expected, q)
            assert verdict == (not s.score_response(response, expected, q))
    _must_fail(mutated, "equivalence check does not compare against score_response")


def test_score_response_two_valued_behaviour_is_unchanged():
    """(b) score_response keeps its exact legacy behaviour, raises included.

    These are the outcomes on inputs the new taxonomy calls UNDECIDABLE: the
    two-valued function must still return what it always returned, so no existing
    number moves until a caller deliberately migrates.
    """
    _force_surface_matching()
    assert s.score_response("", "D", {"scoring_method": "multiple_choice"}) is False
    assert s.score_response("I think C is likely, or maybe D, hard to say", "D",
                            {"scoring_method": "multiple_choice"}) is False
    # the historical INFLATED pass: no gold at all, and the extractor's "" matches
    assert s.score_response("no letter here at all", "",
                            {"scoring_method": "multiple_choice"}) is True
    # ...which the three-valued sibling refuses to call a pass
    assert s.score_response_or_error("no letter here at all", "",
                                     {"scoring_method": "multiple_choice"}) == (
        None, vocab.CAUSE_NO_REFERENCE)
    # no-oracle code_execution is still a (wrong) False on the legacy path
    assert s.score_response("def f(): pass", "", {"scoring_method": "code_execution",
                                                  "scoring_config": {}}) is False
    # unknown method still falls through to the strip-compare fallback
    assert s.score_response("x", "x", {"scoring_method": "vibes_based"}) is True
    # and the vacuity guard still RAISES on the two-valued path
    try:
        s.score_response("anything", "", {"scoring_method": "ordered_subsequence",
                                          "scoring_config": {}})
        raise AssertionError("empty concept list must still raise on score_response")
    except ValueError:
        pass


def test_or_error_converts_a_raising_checker_instead_of_propagating():
    _force_surface_matching()
    # A malformed suite config that makes the dispatch raise INSIDE the checker
    # (past the pure pre-check): concepts present but not a usable sequence.
    q = {"scoring_method": "ordered_subsequence",
         "scoring_config": {"concepts": ["mix"]}}
    boom = object()  # not a str -> _lemma_tokens raises inside the checker
    verdict, cause = s.score_response_or_error(boom, "", q)  # type: ignore[arg-type]
    assert verdict is None and cause == vocab.CAUSE_CHECKER_ERROR

    # MUTATION: the same input on the two-valued path must still raise, proving
    # the conversion is done by the wrapper and not by a change to score_response.
    raised = False
    try:
        s.score_response(boom, "", q)  # type: ignore[arg-type]
    except Exception:
        raised = True
    assert raised, "score_response must keep propagating checker exceptions"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
