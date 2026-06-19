#!/usr/bin/env python3
"""Unit tests for EV-3: ScoringVerifiersAdapter.

Tests cover:
- Label normalisation (_normalise_label)
- Tier estimation (_estimate_tier)
- Prompt structure (_row_to_prompt)
- Local JSONL loading (_load_from_local)
- Graceful failure when dataset is absent
- ADAPTER_SUITES / get_adapter registration

Run with:
    pytest scripts/benchmark/test_scoring_verifiers_adapter.py -q
"""

import json
import sys
from pathlib import Path
import tempfile

# Ensure the benchmark directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from scoring_verifiers_adapter import ScoringVerifiersAdapter


# ── Label normalisation ───────────────────────────────────────────────────────

class TestNormaliseLabel:
    def test_string_correct(self):
        assert ScoringVerifiersAdapter._normalise_label("correct") == "correct"

    def test_string_incorrect(self):
        assert ScoringVerifiersAdapter._normalise_label("incorrect") == "incorrect"

    def test_truthy_int(self):
        assert ScoringVerifiersAdapter._normalise_label(1) == "correct"

    def test_falsy_int(self):
        assert ScoringVerifiersAdapter._normalise_label(0) == "incorrect"

    def test_bool_true(self):
        assert ScoringVerifiersAdapter._normalise_label(True) == "correct"

    def test_bool_false(self):
        assert ScoringVerifiersAdapter._normalise_label(False) == "incorrect"

    def test_string_pass(self):
        assert ScoringVerifiersAdapter._normalise_label("pass") == "correct"

    def test_string_fail(self):
        assert ScoringVerifiersAdapter._normalise_label("fail") == "incorrect"

    def test_float_above_half(self):
        assert ScoringVerifiersAdapter._normalise_label("0.9") == "correct"

    def test_float_below_half(self):
        assert ScoringVerifiersAdapter._normalise_label("0.3") == "incorrect"

    def test_float_exact_half(self):
        assert ScoringVerifiersAdapter._normalise_label("0.5") == "correct"

    def test_none(self):
        assert ScoringVerifiersAdapter._normalise_label(None) == "incorrect"

    def test_yes(self):
        assert ScoringVerifiersAdapter._normalise_label("yes") == "correct"

    def test_no(self):
        assert ScoringVerifiersAdapter._normalise_label("no") == "incorrect"


# ── Tier estimation ───────────────────────────────────────────────────────────

class TestEstimateTier:
    def _row(self, solution_lines=1, problem_len=50):
        return {
            "solution": "\n".join(["x = 1"] * solution_lines),
            "problem": "a" * problem_len,
        }

    def test_tier1_short(self):
        row = self._row(solution_lines=2, problem_len=100)
        assert ScoringVerifiersAdapter._estimate_tier(row) == 1

    def test_tier2_medium(self):
        row = self._row(solution_lines=10, problem_len=200)
        assert ScoringVerifiersAdapter._estimate_tier(row) == 2

    def test_tier3_long_solution(self):
        row = self._row(solution_lines=25, problem_len=100)
        assert ScoringVerifiersAdapter._estimate_tier(row) == 3

    def test_tier3_long_problem(self):
        row = self._row(solution_lines=2, problem_len=500)
        assert ScoringVerifiersAdapter._estimate_tier(row) == 3


# ── Prompt structure ──────────────────────────────────────────────────────────

class TestRowToPrompt:
    def _make_row(self, label=1, solution="def f(): return 42", problem="What does f() return?"):
        return {
            "id": "test_001",
            "subset": "he_r_plus",
            "problem": problem,
            "solution": solution,
            "label": label,
        }

    def _get_prompt(self, row):
        adapter = ScoringVerifiersAdapter()
        adapter._dataset = [row]
        return adapter._row_to_prompt(0, row)

    def test_suite_name(self):
        p = self._get_prompt(self._make_row())
        assert p["suite"] == "scoring_verifiers"

    def test_expected_correct(self):
        p = self._get_prompt(self._make_row(label=1))
        assert p["expected"] == "correct"

    def test_expected_incorrect(self):
        p = self._get_prompt(self._make_row(label=0))
        assert p["expected"] == "incorrect"

    def test_prompt_contains_problem(self):
        p = self._get_prompt(self._make_row(problem="Is 2+2=5?"))
        assert "Is 2+2=5?" in p["prompt"]

    def test_prompt_contains_solution(self):
        p = self._get_prompt(self._make_row(solution="return False"))
        assert "return False" in p["prompt"]

    def test_scoring_method(self):
        p = self._get_prompt(self._make_row())
        assert p["scoring_method"] == "multiple_choice"

    def test_scoring_config_choices(self):
        p = self._get_prompt(self._make_row())
        assert "correct" in p["scoring_config"]["choices"]
        assert "incorrect" in p["scoring_config"]["choices"]

    def test_tier_range(self):
        p = self._get_prompt(self._make_row())
        assert p["tier"] in (1, 2, 3)

    def test_id_contains_subset(self):
        p = self._get_prompt(self._make_row())
        assert "he_r_plus" in p["id"]

    def test_prompt_asks_for_correct_or_incorrect(self):
        p = self._get_prompt(self._make_row())
        assert "correct" in p["prompt"].lower() and "incorrect" in p["prompt"].lower()


# ── Local JSONL loading ───────────────────────────────────────────────────────

class TestLocalJSONLLoading:
    def _make_temp_jsonl(self, rows):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "test.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return Path(tmp)

    def test_loads_rows(self):
        rows = [
            {"id": 1, "problem": "Q1", "solution": "A1", "label": 1},
            {"id": 2, "problem": "Q2", "solution": "A2", "label": 0},
        ]
        tmp_dir = self._make_temp_jsonl(rows)
        loaded = ScoringVerifiersAdapter._load_from_local(tmp_dir)
        assert len(loaded) == 2

    def test_row_content(self):
        rows = [{"problem": "test", "solution": "sol", "label": "pass"}]
        tmp_dir = self._make_temp_jsonl(rows)
        loaded = ScoringVerifiersAdapter._load_from_local(tmp_dir)
        assert loaded[0]["problem"] == "test"

    def test_empty_dir_returns_empty(self):
        tmp = Path(tempfile.mkdtemp())
        loaded = ScoringVerifiersAdapter._load_from_local(tmp)
        assert loaded == []

    def test_skips_invalid_json_lines(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "mixed.jsonl"
        path.write_text('{"id": 1}\n{broken json\n{"id": 2}\n')
        loaded = ScoringVerifiersAdapter._load_from_local(Path(tmp))
        assert len(loaded) == 2

    def test_records_subset_from_jsonl_filename(self):
        rows = [{"problem": "Q1", "all_solutions": []}]
        tmp_dir = self._make_temp_jsonl(rows)
        (tmp_dir / "test.jsonl").rename(tmp_dir / "HE-R+.jsonl")
        loaded = ScoringVerifiersAdapter._load_from_local(tmp_dir)
        assert loaded[0]["subset"] == "HE-R+"


class TestSolutionExpansion:
    def test_expands_all_solutions_to_labeled_items(self):
        rows = [
            {
                "task_id": "HumanEval/0",
                "subset": "HE-R+",
                "prompt": "Write f().",
                "all_solutions": [
                    {"rank": 1, "average_test_score": 1.0, "solution": "def f(): return 1"},
                    {"rank": 2, "average_test_score": 0.0, "solution": "def f(): return 0"},
                ],
            }
        ]
        expanded = ScoringVerifiersAdapter._expand_solution_rows(rows)

        assert len(expanded) == 2
        assert expanded[0]["label"] == 1.0
        assert expanded[1]["label"] == 0.0
        assert expanded[0]["solution"] == "def f(): return 1"
        assert expanded[0]["id"] == "HumanEval/0::sol0"

    def test_prompt_uses_expanded_solution_score(self):
        adapter = ScoringVerifiersAdapter()
        row = {
            "id": "HumanEval/0::sol1",
            "subset": "HE-R+",
            "problem": "Write f().",
            "solution": "def f(): return 0",
            "label": 0.0,
            "average_test_score": 0.0,
            "rank": 2,
            "task_id": "HumanEval/0",
        }

        prompt = adapter._row_to_prompt(0, row)

        assert prompt["expected"] == "incorrect"
        assert "def f(): return 0" in prompt["prompt"]
        assert prompt["metadata"]["average_test_score"] == 0.0
        assert prompt["metadata"]["task_id"] == "HumanEval/0"


# ── Graceful failure with missing dataset ────────────────────────────────────

class TestGracefulFailure:
    def test_empty_dataset_on_no_data(self):
        """Adapter should not raise; just return empty dataset."""
        adapter = ScoringVerifiersAdapter(
            local_path="/nonexistent/path/that/does/not/exist"
        )
        # Patch HF loading to also fail
        original = adapter._ensure_loaded

        def patched():
            adapter._dataset = []

        adapter._ensure_loaded = patched
        adapter._ensure_loaded()
        assert adapter._dataset == []
        assert adapter.total_available == 0
        assert adapter.sample(n=5) == []


# ── Dataset adapter registration ─────────────────────────────────────────────

class TestRegistration:
    def test_in_adapter_suites(self):
        from dataset_adapters import ADAPTER_SUITES
        assert "scoring_verifiers" in ADAPTER_SUITES

    def test_get_adapter_returns_instance(self):
        from dataset_adapters import get_adapter
        adapter = get_adapter("scoring_verifiers")
        assert adapter is not None
        assert isinstance(adapter, ScoringVerifiersAdapter)

    def test_suite_name(self):
        assert ScoringVerifiersAdapter.suite_name == "scoring_verifiers"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
