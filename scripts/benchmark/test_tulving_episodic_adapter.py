#!/usr/bin/env python3
"""Unit tests for P3b: TulvingEpisodicAdapter + deterministic F1 scorer.

Tests cover:
- Token normalisation (_normalise_token, _tokenise)
- Single-pair F1 (_token_f1)
- List-level F1 (score_f1_list)
- Response parsing (_extract_list_from_response)
- Prompt construction (_row_to_prompt)
- Correct-answer parsing (_parse_correct_answer)
- Tier assignment (_get_tier_for_index)
- Graceful failure when data is absent
- Composite scores (compute_simple_recall_score, compute_chronological_awareness_score)
- Dataset adapter registration

Run with:
    pytest scripts/benchmark/test_tulving_episodic_adapter.py -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tulving_episodic_adapter import (
    TulvingEpisodicAdapter,
    _normalise_token,
    _tokenise,
    _token_f1,
    score_f1_list,
    _extract_list_from_response,
    compute_simple_recall_score,
    compute_chronological_awareness_score,
)


# ── Token normalisation ───────────────────────────────────────────────────────

class TestNormaliseToken:
    def test_lowercase(self):
        assert _normalise_token("New York City") == "new york city"

    def test_punctuation_stripped(self):
        assert _normalise_token("Sep 22, 2026.") == "sep 22 2026"

    def test_whitespace_collapsed(self):
        assert _normalise_token("  hello   world  ") == "hello world"

    def test_empty(self):
        assert _normalise_token("") == ""

    def test_digits_preserved(self):
        assert "2026" in _normalise_token("Year 2026.")

    def test_unicode_nfc(self):
        # Composed vs decomposed 'é' — both should normalise to same
        import unicodedata
        composed = unicodedata.normalize("NFC", "é")
        decomposed = unicodedata.normalize("NFD", "é")
        assert _normalise_token(composed) == _normalise_token(decomposed)


class TestTokenise:
    def test_splits_correctly(self):
        assert _tokenise("New York City") == ["new", "york", "city"]

    def test_date_tokens(self):
        tokens = _tokenise("Sep 22, 2026")
        assert "sep" in tokens
        assert "22" in tokens
        assert "2026" in tokens

    def test_empty(self):
        assert _tokenise("") == []


# ── Token F1 ─────────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_identical(self):
        assert _token_f1("New York City", "New York City") == 1.0

    def test_completely_different(self):
        assert _token_f1("apple", "banana") == 0.0

    def test_partial_overlap(self):
        f1 = _token_f1("Sep 22 2026", "Sep 30 2026")
        assert 0.0 < f1 < 1.0

    def test_both_empty(self):
        assert _token_f1("", "") == 1.0

    def test_pred_empty(self):
        assert _token_f1("", "something") == 0.0

    def test_gt_empty(self):
        assert _token_f1("something", "") == 0.0

    def test_date_format_variants(self):
        # "Sep 22, 2026" vs "September 22 2026" — partial match
        f1 = _token_f1("Sep 22, 2026", "September 22, 2026")
        assert f1 > 0.5  # At least "22" and "2026" match

    def test_entity_name_exact(self):
        assert _token_f1("Jackson Ramos", "Jackson Ramos") == 1.0

    def test_entity_name_partial(self):
        f1 = _token_f1("Jackson", "Jackson Ramos")
        assert 0.0 < f1 < 1.0


# ── List-level F1 ─────────────────────────────────────────────────────────────

class TestScoreF1List:
    def test_perfect_match(self):
        result = score_f1_list(
            ["Jackson Ramos", "Emilia Hooks"],
            ["Jackson Ramos", "Emilia Hooks"],
        )
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_empty_both(self):
        result = score_f1_list([], [])
        assert result["f1"] == 1.0

    def test_empty_pred_nonempty_gt(self):
        result = score_f1_list([], ["Jackson Ramos"])
        assert result["f1"] == 0.0
        assert result["nb_gt"] == 1

    def test_nonempty_pred_empty_gt_hallucination(self):
        # Predicting something when GT is empty = hallucination
        result = score_f1_list(["Jackson Ramos"], [])
        assert result["f1"] == 0.0
        assert result["nb_pred"] == 1
        assert result["nb_gt"] == 0

    def test_partial_match(self):
        result = score_f1_list(
            ["Jackson Ramos", "Unknown Person"],
            ["Jackson Ramos", "Emilia Hooks"],
        )
        assert 0.0 < result["f1"] < 1.0

    def test_single_correct(self):
        result = score_f1_list(
            ["New York"],
            ["New York"],
        )
        assert result["f1"] == 1.0

    def test_dates(self):
        result = score_f1_list(
            ["Sep 22, 2026", "Feb 27, 2026"],
            ["Sep 22, 2026", "Feb 27, 2026", "Aug 24, 2026"],
        )
        # 2/3 items found correctly
        assert result["recall"] > 0.5
        assert result["nb_gt"] == 3

    def test_lenient_capping(self):
        # Predicting more items than GT should not inflate precision past 1.0
        result = score_f1_list(
            ["A", "B", "C", "D", "E"],  # 5 predictions
            ["A", "B"],                 # 2 GT items
        )
        assert result["precision"] <= 1.0

    def test_threshold_param(self):
        # With a strict threshold, a weak match should not count
        result_strict = score_f1_list(
            ["Sep 22"],  # Missing "2026" token
            ["Sep 22 2026"],
            threshold=0.9,
        )
        result_lenient = score_f1_list(
            ["Sep 22"],
            ["Sep 22 2026"],
            threshold=0.3,
        )
        assert result_strict["f1"] <= result_lenient["f1"]

    def test_matched_gt_items(self):
        result = score_f1_list(
            ["Jackson Ramos"],
            ["Jackson Ramos", "Emilia Hooks"],
        )
        assert "Jackson Ramos" in result["matched_gt_items"]

    def test_locations(self):
        result = score_f1_list(
            ["New South Wales"],
            ["New South Wales"],
        )
        assert result["f1"] == 1.0


# ── Response parsing ──────────────────────────────────────────────────────────

class TestExtractListFromResponse:
    def test_bullet_list(self):
        response = "- Sep 22, 2026\n- Feb 27, 2026\n- Aug 24, 2026"
        items = _extract_list_from_response(response)
        assert len(items) == 3
        assert "Sep 22, 2026" in items

    def test_numbered_list(self):
        response = "1. Jackson Ramos\n2. Emilia Hooks"
        items = _extract_list_from_response(response)
        assert "Jackson Ramos" in items
        assert "Emilia Hooks" in items

    def test_asterisk_bullets(self):
        response = "* New York\n* London"
        items = _extract_list_from_response(response)
        assert "New York" in items

    def test_comma_separated(self):
        response = "New York, London, Paris"
        items = _extract_list_from_response(response)
        assert len(items) == 3
        assert "New York" in items

    def test_abstention_returns_empty(self):
        assert _extract_list_from_response("I don't know") == []
        assert _extract_list_from_response("I'm not sure") == []
        assert _extract_list_from_response("No information available") == []

    def test_explanatory_not_mentioned_does_not_erase_list(self):
        response = "- High Line\n- Woolworth Building\n\nSome other locations are not mentioned."
        assert _extract_list_from_response(response) == ["High Line", "Woolworth Building"]

    def test_single_item(self):
        items = _extract_list_from_response("New York")
        assert len(items) >= 1

    def test_none_keyword_is_abstention(self):
        # "None" is the prompt-instructed abstention token → maps to empty list
        assert _extract_list_from_response("None") == []
        assert _extract_list_from_response("none") == []


# ── Correct answer parsing ────────────────────────────────────────────────────

class TestParseCorrectAnswer:
    def test_list(self):
        result = TulvingEpisodicAdapter._parse_correct_answer(["A", "B"])
        assert result == ["A", "B"]

    def test_numpy_array_like(self):
        class ArrayLike:
            def tolist(self):
                return ["A", "B"]

        result = TulvingEpisodicAdapter._parse_correct_answer(ArrayLike())
        assert result == ["A", "B"]

    def test_json_string(self):
        result = TulvingEpisodicAdapter._parse_correct_answer('["A", "B"]')
        assert result == ["A", "B"]

    def test_python_list_string(self):
        result = TulvingEpisodicAdapter._parse_correct_answer("['A', 'B']")
        assert result == ["A", "B"]

    def test_plain_string(self):
        result = TulvingEpisodicAdapter._parse_correct_answer("A single answer")
        assert result == ["A single answer"]

    def test_empty_list(self):
        result = TulvingEpisodicAdapter._parse_correct_answer([])
        assert result == []

    def test_empty_string(self):
        result = TulvingEpisodicAdapter._parse_correct_answer("")
        assert result == []

    def test_none_items_filtered(self):
        result = TulvingEpisodicAdapter._parse_correct_answer([None, "A", None])
        assert result == ["A"]


# ── Prompt construction ────────────────────────────────────────────────────────

class TestRowToPrompt:
    def _make_adapter_with_row(self, row):
        adapter = TulvingEpisodicAdapter(data_dir="/nonexistent")
        adapter._dataset = [row]
        return adapter

    def _sample_row(self, retrieval_type="Times", get="all", nb_gt=3):
        return {
            "question": "List all dates for Jackson Ramos.",
            "correct_answer": ["Sep 22, 2026", "Feb 27, 2026", "Aug 24, 2026"][:nb_gt],
            "retrieval_type": retrieval_type,
            "get": get,
            "cue": "(*, *, ent, *)",
            "chapter": 5,
        }

    def test_suite_name(self):
        a = self._make_adapter_with_row(self._sample_row())
        p = a._row_to_prompt(0, self._sample_row())
        assert p["suite"] == "tulving_episodic"

    def test_expected_is_json(self):
        row = self._sample_row()
        a = self._make_adapter_with_row(row)
        p = a._row_to_prompt(0, row)
        parsed = json.loads(p["expected"])
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_scoring_method(self):
        a = self._make_adapter_with_row(self._sample_row())
        p = a._row_to_prompt(0, self._sample_row())
        assert p["scoring_method"] == "f1_list"

    def test_metadata_ground_truth(self):
        a = self._make_adapter_with_row(self._sample_row())
        p = a._row_to_prompt(0, self._sample_row())
        assert p["metadata"]["ground_truth_items"] == ["Sep 22, 2026", "Feb 27, 2026", "Aug 24, 2026"]

    def test_prompt_contains_question(self):
        row = self._sample_row()
        a = self._make_adapter_with_row(row)
        p = a._row_to_prompt(0, row)
        assert "Jackson Ramos" in p["prompt"]

    def test_spaces_retrieval_prompt(self):
        row = self._sample_row(retrieval_type="Spaces")
        a = self._make_adapter_with_row(row)
        p = a._row_to_prompt(0, row)
        assert "location" in p["prompt"].lower()

    def test_entities_retrieval_prompt(self):
        row = self._sample_row(retrieval_type="Entities")
        a = self._make_adapter_with_row(row)
        p = a._row_to_prompt(0, row)
        assert "entity" in p["prompt"].lower() or "name" in p["prompt"].lower()


# ── Tier assignment ───────────────────────────────────────────────────────────

class TestTierAssignment:
    def _adapter(self, rows):
        a = TulvingEpisodicAdapter(data_dir="/nonexistent")
        a._dataset = rows
        return a

    def test_tier1_zero_gt(self):
        a = self._adapter([{"correct_answer": [], "get": "all"}])
        assert a._get_tier_for_index(0) == 1

    def test_tier1_latest_single(self):
        a = self._adapter([{"correct_answer": ["Paris"], "get": "latest"}])
        assert a._get_tier_for_index(0) == 1

    def test_tier3_chronological(self):
        a = self._adapter([{"correct_answer": ["A", "B", "C"], "get": "chronological"}])
        assert a._get_tier_for_index(0) == 3

    def test_tier3_many_items(self):
        items = ["A", "B", "C", "D", "E", "F", "G"]
        a = self._adapter([{"correct_answer": items, "get": "all"}])
        assert a._get_tier_for_index(0) == 3

    def test_tier2_medium(self):
        a = self._adapter([{"correct_answer": ["A", "B", "C"], "get": "all"}])
        assert a._get_tier_for_index(0) == 2

    def test_tier_uses_python_list_string_count(self):
        a = self._adapter([{"correct_answer": "['A', 'B', 'C']", "get": "all"}])
        assert a._get_tier_for_index(0) == 2


# ── Graceful failure ──────────────────────────────────────────────────────────

class TestGracefulFailure:
    def test_no_data_returns_empty(self):
        adapter = TulvingEpisodicAdapter(data_dir="/nonexistent/path")
        adapter._ensure_loaded()
        assert adapter._dataset == []
        assert adapter.total_available == 0

    def test_sample_empty_returns_empty(self):
        adapter = TulvingEpisodicAdapter(data_dir="/nonexistent/path")
        adapter._ensure_loaded()
        assert adapter.sample(n=5) == []


# ── Composite score computation ───────────────────────────────────────────────

class TestCompositeScores:
    def test_simple_recall_score_perfect(self):
        results = [
            {"f1": 1.0, "nb_gt": 0},
            {"f1": 1.0, "nb_gt": 1},
            {"f1": 1.0, "nb_gt": 2},
            {"f1": 1.0, "nb_gt": 4},
            {"f1": 1.0, "nb_gt": 7},
        ]
        score = compute_simple_recall_score(results)
        assert score == 1.0

    def test_simple_recall_score_zero(self):
        results = [
            {"f1": 0.0, "nb_gt": 1},
            {"f1": 0.0, "nb_gt": 2},
        ]
        score = compute_simple_recall_score(results)
        assert score == 0.0

    def test_simple_recall_score_partial(self):
        results = [
            {"f1": 1.0, "nb_gt": 0},  # group 0: perfect
            {"f1": 0.5, "nb_gt": 1},  # group 1: half
        ]
        score = compute_simple_recall_score(results)
        assert 0.5 < score < 1.0

    def test_simple_recall_score_empty(self):
        score = compute_simple_recall_score([])
        assert score == 0.0

    def test_simple_recall_buckets(self):
        """Verify equal-weight grouping: groups with no questions don't affect average."""
        # Only groups 1 and 2+ present
        results = [{"f1": 0.5, "nb_gt": 1}, {"f1": 0.0, "nb_gt": 2}]
        score = compute_simple_recall_score(results)
        # 2 groups: avg(0.5) = 0.5, avg(0.0) = 0.0 → final = 0.25
        assert abs(score - 0.25) < 1e-9

    def test_chronological_awareness_both_halves(self):
        latest = [{"f1": 0.8}, {"f1": 0.6}]    # avg = 0.7
        chrono = [{"kendall_tau": 0.5}, {"kendall_tau": 0.3}]  # avg = 0.4
        score = compute_chronological_awareness_score(latest, chrono)
        assert abs(score - 0.55) < 1e-9

    def test_chronological_awareness_latest_only(self):
        latest = [{"f1": 0.6}]
        score = compute_chronological_awareness_score(latest, [])
        assert score == 0.6

    def test_chronological_awareness_chrono_only(self):
        chrono = [{"kendall_tau": 0.4}]
        score = compute_chronological_awareness_score([], chrono)
        assert score == 0.4

    def test_chronological_awareness_empty(self):
        score = compute_chronological_awareness_score([], [])
        assert score == 0.0

    def test_chronological_awareness_perfect(self):
        latest = [{"f1": 1.0}]
        chrono = [{"kendall_tau": 1.0}]
        score = compute_chronological_awareness_score(latest, chrono)
        assert score == 1.0


# ── compute_f1_for_result ─────────────────────────────────────────────────────

class TestComputeF1ForResult:
    def _make_prompt_dict(self, ground_truth, retrieval_type="Times", get_style="all"):
        return {
            "metadata": {
                "ground_truth_items": ground_truth,
                "retrieval_type": retrieval_type,
                "get_style": get_style,
            }
        }

    def test_perfect_list_answer(self):
        pd = self._make_prompt_dict(["Sep 22, 2026", "Feb 27, 2026"])
        result = TulvingEpisodicAdapter.compute_f1_for_result(
            "- Sep 22, 2026\n- Feb 27, 2026",
            pd,
        )
        assert result["f1"] == 1.0

    def test_empty_gt_abstention_perfect(self):
        pd = self._make_prompt_dict([])
        result = TulvingEpisodicAdapter.compute_f1_for_result("None", pd)
        # GT empty, pred empty (abstention) → F1 = 1.0
        assert result["f1"] == 1.0

    def test_hallucination_penalised(self):
        pd = self._make_prompt_dict([])
        result = TulvingEpisodicAdapter.compute_f1_for_result(
            "- Jackson Ramos\n- Emilia Hooks",
            pd,
        )
        assert result["f1"] == 0.0

    def test_partial_recall(self):
        pd = self._make_prompt_dict(["A", "B", "C"])
        result = TulvingEpisodicAdapter.compute_f1_for_result("- A\n- B", pd)
        assert 0.0 < result["f1"] < 1.0

    def test_llm_judge_override(self):
        pd = self._make_prompt_dict(["something"])

        def fake_judge(preds, gts, rtype):
            return 0.99

        result = TulvingEpisodicAdapter.compute_f1_for_result(
            "irrelevant response", pd, llm_judge=fake_judge
        )
        assert result["f1"] == 0.99
        assert result["source"] == "llm_judge"

    def test_source_deterministic_when_no_judge(self):
        pd = self._make_prompt_dict(["A"])
        result = TulvingEpisodicAdapter.compute_f1_for_result("- A", pd)
        assert result["source"] == "deterministic"


# ── Dataset adapter registration ─────────────────────────────────────────────

class TestRegistration:
    def test_in_adapter_suites(self):
        from dataset_adapters import ADAPTER_SUITES
        assert "tulving_episodic" in ADAPTER_SUITES

    def test_get_adapter_returns_instance(self):
        from dataset_adapters import get_adapter
        adapter = get_adapter("tulving_episodic")
        assert adapter is not None
        assert isinstance(adapter, TulvingEpisodicAdapter)

    def test_suite_name(self):
        assert TulvingEpisodicAdapter.suite_name == "tulving_episodic"

    def test_ingest_role_includes_tulving(self):
        from suites import ROLE_SUITE_MAP
        assert "tulving_episodic" in ROLE_SUITE_MAP["ingest"]

    def test_long_context_role_includes_tulving(self):
        from suites import ROLE_SUITE_MAP
        assert "tulving_episodic" in ROLE_SUITE_MAP["long_context"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
