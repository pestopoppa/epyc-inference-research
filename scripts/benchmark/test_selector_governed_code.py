"""RM-11 selector code scoring uses the existing governed executable oracle."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import debug_scorer
import qwable_verifier_selector_runner as runner


@pytest.fixture(autouse=True)
def private_governed_scorer_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_scorer._orb, "_SCORER_TMP_ROOT", tmp_path)


def question(config):
    return {"id": "synthetic-code", "prompt": "Implement solve(n)",
            "suite": "synthetic", "tier": 1, "scoring_method": "code_execution",
            "expected": "legacy expected text must not become a zero-argument call",
            "scoring_config": config}


@pytest.mark.parametrize("fenced", [False, True])
def test_entry_point_cases_and_complete_module_are_actually_executed(fenced):
    raw = ("from __future__ import annotations\n\nimport math\n\n"
           "def helper(n):\n    return math.isqrt(n)\n\n"
           "def solve(n: MissingType):\n    return helper(n) + 1\n")
    answer = f"```python\n{raw}```" if fenced else raw
    q = question({"entry_point": "solve", "entry_point_cases": [([16], 5), ([25], 6)]})
    assert runner.score_candidate(q, answer) is True
    assert runner.score_candidate(q, answer.replace("helper(n) + 1", "helper(n) - 1")) is False


def test_missing_executable_oracle_is_unavailable_not_wrong():
    q = question({"entry_point": "solve"})
    with pytest.raises(debug_scorer.ScoringUnavailableError, match="entry_point_cases"):
        runner.score_candidate(q, "def solve(n):\n    return n + 1\n")


def test_vacuous_test_code_cannot_pass_a_candidate():
    q = question({"test_code": "# assert solve(4) == 5"})
    assert runner.score_candidate(q, "def solve(n):\n    return 0\n") is False


def test_unittest_oracle_runs_instead_of_accepting_an_idle_test_class():
    q = question({"test_code": "import unittest\nclass Cases(unittest.TestCase):\n"
                                "    def test_answer(self):\n        self.assertEqual(solve(4), 5)"})
    assert runner.score_candidate(q, "def solve(n):\n    return n + 1\n") is True
    assert runner.score_candidate(q, "def solve(n):\n    return n - 1\n") is False


def test_relative_files_stay_inside_governed_scorer_scratch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answer = ("from pathlib import Path\n\ndef solve():\n"
              "    Path('selector-side-effect').write_text('private')\n    return 5\n")
    q = question({"test_code": "assert solve() == 5"})
    assert runner.score_candidate(q, answer) is True
    assert not (tmp_path / "selector-side-effect").exists()


def test_run_question_records_real_candidate_outcomes_before_selection(tmp_path, monkeypatch):
    q = question({"entry_point": "solve", "entry_point_cases": [([4], 5)]})
    args = runner.parse_args(["--n-candidates", "2"])
    replies = iter(["def solve(n):\n    return n + 1\n",
                    "def solve(n):\n    return n - 1\n", "FINAL: 0"])
    calls = []

    def chat(port, payload, timeout):
        calls.append(port)
        return {"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}]}

    monkeypatch.setattr(runner, "chat", chat)
    row = runner.run_question(q, args, SimpleNamespace(port=19000),
                              SimpleNamespace(port=19001), tmp_path)
    assert calls == [19000, 19000, 19001]
    assert [candidate["correct"] for candidate in row["candidates"]] == [True, False]
    assert row["pass_at_1"] is True and row["verifier_pass"] is True
    assert row["n_passing"] == 1 and row["oracle_pass_at_n"] is True
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_execute_persists_unavailable_oracle_and_continues_next_question(tmp_path, monkeypatch):
    missing = question({"entry_point": "solve"})
    missing["id"] = "missing-oracle"
    valid = question({"entry_point": "solve", "entry_point_cases": [([4], 5)]})
    valid["id"] = "valid-oracle"
    args = runner.parse_args(["--output-dir", str(tmp_path), "--limit", "2", "--n-candidates", "1"])
    specs = [SimpleNamespace(name="beneficiary", port=19000, model_path=tmp_path),
             SimpleNamespace(name="verifier", port=19001, model_path=tmp_path)]
    monkeypatch.setattr(runner, "server_specs", lambda _args: specs)
    monkeypatch.setattr(runner, "load_questions", lambda *_args: [missing, valid])
    monkeypatch.setattr(runner, "process_snapshot", lambda: "synthetic transport")
    monkeypatch.setattr(runner, "launch_server", lambda *_args: SimpleNamespace(pid=0))
    monkeypatch.setattr(runner, "wait_for_health", lambda *_args: None)
    monkeypatch.setattr(runner, "terminate_server", lambda *_args: None)

    def chat(port, payload, timeout):
        content = "FINAL: 0" if port == 19001 else "def solve(n):\n    return n + 1\n"
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(runner, "chat", chat)
    assert runner.execute(args, {}) == 0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert "entry_point_cases" in rows[0]["error"]
    assert rows[1]["pass_at_1"] is True
