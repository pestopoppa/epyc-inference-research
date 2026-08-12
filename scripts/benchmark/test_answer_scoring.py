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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
