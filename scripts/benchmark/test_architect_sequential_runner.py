from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import architect_sequential_runner as runner


def _question(qid: str, difficulty: int, expected: str = "A") -> dict:
    return {
        "id": qid,
        "prompt": f"prompt-{qid}",
        "expected": expected,
        "difficulty_key": difficulty,
        "scoring_method": "multiple_choice",
        "scoring_config": {},
    }


def _meta(text: str, *, error: str = "") -> dict:
    return {"text": text, "error": error, "finish_reason": "stop"}


class TestArchitectSequentialRunner(unittest.TestCase):
    def test_order_questions_is_descending_and_requires_a_priori_key(self):
        ordered = runner.order_questions(
            [_question("tie-b", 3), _question("easy", 1), _question("tie-a", 3)],
            difficulty_field="difficulty_key",
        )

        self.assertEqual([item["id"] for item in ordered], ["tie-a", "tie-b", "easy"])
        with self.assertRaisesRegex(ValueError, "a-priori difficulty"):
            runner.order_questions([{"id": "missing"}], difficulty_field="difficulty_key")

    def test_interleaves_complete_pairs_and_stops_on_separation(self):
        calls: list[tuple[str, str]] = []

        def fake_query(url: str, prompt: str, **_kwargs):
            calls.append((url, prompt))
            return _meta("A" if url.endswith("candidate") else "B")

        policy = runner.SequentialPolicy(confirm_e=1.05, budget=99)
        with self.subTest("interleaved pair"):
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as directory:
                temp = Path(directory)
                result = runner.run_interleaved(
                    suite="future_hard", questions=[_question("hard", 5), _question("easy", 1)],
                    arms=[runner.Arm("baseline", "http://baseline"), runner.Arm("candidate", "http://candidate")],
                    baseline_arm="baseline", candidate_arm="candidate", difficulty_field="difficulty_key",
                    saturation=runner.SaturationPolicy(2, 1.0), output=temp / "result.json",
                    capture_out=temp / "capture.jsonl", seed=42, max_tokens=64, temperature=0.7,
                    policy=policy, query=fake_query,
                )

                self.assertEqual(calls, [("http://baseline", "prompt-hard"), ("http://candidate", "prompt-hard")])
                self.assertEqual(result["stop_reason"], "separation:candidate")
                self.assertEqual(result["candidate_eprocess"]["k"], 1)
                capture = [json.loads(line) for line in (temp / "capture.jsonl").read_text().splitlines()]
                self.assertEqual([row["arm"] for row in capture], ["baseline", "candidate"])

    def test_saturation_stops_only_after_a_completed_difficulty_tier(self):
        calls: list[str] = []

        def fake_query(url: str, _prompt: str, **_kwargs):
            calls.append(url)
            return _meta("A")

        policy = runner.SequentialPolicy(first_lambda=0.0, lambda_cap=0.0, budget=99)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            result = runner.run_interleaved(
                suite="future_hard", questions=[_question("hard", 5), _question("easy", 1)],
                arms=[runner.Arm("baseline", "http://baseline"), runner.Arm("candidate", "http://candidate")],
                baseline_arm="baseline", candidate_arm="candidate", difficulty_field="difficulty_key",
                saturation=runner.SaturationPolicy(1, 1.0), output=temp / "result.json",
                capture_out=temp / "capture.jsonl", seed=42, max_tokens=64, temperature=0.7,
                policy=policy, query=fake_query,
            )

        self.assertEqual(calls, ["http://baseline", "http://candidate"])
        self.assertEqual(result["stop_reason"], "saturation:difficulty_key=5")
        self.assertEqual(result["candidate_eprocess"]["k"], 1)

    def test_transport_failure_is_recorded_without_eprocess_update(self):
        def fake_query(url: str, _prompt: str, **_kwargs):
            return _meta("A" if url.endswith("candidate") else "", error="timeout" if url.endswith("baseline") else "")

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            result = runner.run_interleaved(
                suite="future_hard", questions=[_question("hard", 5)],
                arms=[runner.Arm("baseline", "http://baseline"), runner.Arm("candidate", "http://candidate")],
                baseline_arm="baseline", candidate_arm="candidate", difficulty_field="difficulty_key",
                saturation=runner.SaturationPolicy(1, 1.0), output=temp / "result.json",
                capture_out=temp / "capture.jsonl", seed=42, max_tokens=64, temperature=0.7,
                query=fake_query,
            )

        self.assertEqual(result["provisional_transport_pairs"], 1)
        self.assertEqual(result["candidate_eprocess"]["k"], 0)
        self.assertEqual(result["pairs"][0]["sequential"]["state"], "not_updated_transport_failure")
