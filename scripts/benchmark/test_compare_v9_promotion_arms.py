from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_v9_promotion_arms.py")
SPEC = importlib.util.spec_from_file_location("compare_v9_promotion_arms", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def record(scenario: str, block: int, rep: int, tps: float, content: str = "same") -> dict:
    return {
        "scenario": scenario,
        "nominal_context": 2048,
        "block": block,
        "rep": rep,
        "prompt_sha256": "prompt",
        "request_sha256": "request",
        "response_sha256": f"response-{scenario}-{block}-{rep}-{tps}",
        "content": content,
        "completion_tokens": 128,
        "token_ids": None,
        "decode_tps": tps,
    }


def arms(candidate_ratio: float = 1.0, content: str = "same") -> tuple[list[dict], list[dict]]:
    baseline = [record("role", block, rep, 10.0) for block in (1, 2) for rep in range(1, 6)]
    candidate = [
        record("role", block, rep, 10.0 * candidate_ratio, content)
        for block in (1, 2)
        for rep in range(1, 6)
    ]
    return baseline, candidate


class ComparisonTests(unittest.TestCase):
    def test_exact_parity_and_ratio_pass(self) -> None:
        baseline, candidate = arms(0.99)
        result = runner.compare(baseline, candidate, minimum_reps=10, gate_throughput=True)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["quality_transfer"]["status"], "pass")
        self.assertAlmostEqual(result["throughput"]["rows"][0]["candidate_over_baseline"], 0.99)

    def test_gray_or_content_drift_fail_closed(self) -> None:
        baseline, candidate = arms(0.97)
        self.assertEqual(
            runner.compare(baseline, candidate, minimum_reps=10, gate_throughput=True)["status"],
            "fail",
        )
        baseline, candidate = arms(1.01, content="changed")
        result = runner.compare(baseline, candidate, minimum_reps=10, gate_throughput=True)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["quality_transfer"]["status"], "fail")

    def test_observation_mode_does_not_gate_throughput(self) -> None:
        baseline, candidate = arms(0.90)
        result = runner.compare(baseline, candidate, minimum_reps=10, gate_throughput=False)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["throughput"]["status"], "fail")

    def test_mismatched_keys_and_short_arms_refuse(self) -> None:
        baseline, candidate = arms()
        with self.assertRaisesRegex(runner.ComparisonError, "cardinality mismatch"):
            runner.compare(baseline, candidate[:-1], minimum_reps=10, gate_throughput=True)
        with self.assertRaisesRegex(runner.ComparisonError, "fewer than"):
            runner.compare(baseline[:5], candidate[:5], minimum_reps=10, gate_throughput=True)


if __name__ == "__main__":
    unittest.main()
