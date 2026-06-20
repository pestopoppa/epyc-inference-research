from __future__ import annotations

import math
import unittest

import score_aa_omniscience_run as scorer


class ScoreAAOmniscienceRunTests(unittest.TestCase):
    def test_extract_answer_uses_final_answer_tag(self) -> None:
        response = "<answer>wrong</answer>\ntext\n<answer>right</answer>"

        self.assertEqual(scorer.extract_answer(response), "right")

    def test_score_response_labels_correct_abstention_partial_and_incorrect(self) -> None:
        config = {
            "extract_pattern": r"<answer>(.*?)</answer>",
            "threshold": 0.8,
            "abstention_patterns": [r"(?i)i don'?t know"],
        }

        self.assertEqual(
            scorer.score_response("<answer>ASC 606-10-25-15</answer>", "ASC 606-10-25-15", config).label,
            "CORRECT",
        )
        self.assertEqual(
            scorer.score_response("<answer>I don't know</answer>", "ASC 606-10-25-15", config).label,
            "NOT_ATTEMPTED",
        )
        self.assertEqual(
            scorer.score_response("<answer>ASC 606</answer>", "ASC 606-10-25-15", config).label,
            "PARTIAL_ANSWER",
        )
        self.assertEqual(
            scorer.score_response("<answer>banana</answer>", "ASC 606-10-25-15", config).label,
            "INCORRECT",
        )

    def test_summary_computes_aa_metrics(self) -> None:
        rows = [
            {"config": "baseline", "domain": "Finance", "label_4class": "CORRECT", "f1": 1.0, "tokens_per_second": 10.0},
            {"config": "baseline", "domain": "Finance", "label_4class": "INCORRECT", "f1": 0.0, "tokens_per_second": 20.0},
            {"config": "baseline", "domain": "Finance", "label_4class": "PARTIAL_ANSWER", "f1": 0.5, "tokens_per_second": 30.0},
            {"config": "baseline", "domain": "Finance", "label_4class": "NOT_ATTEMPTED", "f1": 0.0, "tokens_per_second": None},
        ]

        summary = scorer.summarize(rows)
        baseline = summary["configs"]["baseline"]

        self.assertEqual(baseline["total"], 4)
        self.assertTrue(math.isclose(baseline["accuracy"], 0.25))
        self.assertTrue(math.isclose(baseline["hallucination_rate"], 1 / 3))
        self.assertTrue(math.isclose(baseline["omniscience_index"], 0.5 * 0.25 + 0.5 * (1 - 1 / 3)))
        self.assertTrue(math.isclose(baseline["avg_f1"], 0.375))
        self.assertTrue(math.isclose(baseline["avg_tokens_per_second"], 20.0))


if __name__ == "__main__":
    unittest.main()
