#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).parent))

import qwable_task_quality_runner as runner


class QwableTaskQualityRunnerTests(unittest.TestCase):
    def test_plan_defaults_to_gpu_iq4_and_q8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--output-dir", tmp])
            with mock.patch.object(runner.base, "glm_download_active", return_value=False):
                plan = runner.build_plan(args)

        self.assertEqual(plan["schema"], "qwable_task_quality_plan.v1")
        self.assertEqual(plan["selected_arms"], ["iq4_gpu", "q8_gpu"])
        self.assertEqual(len(plan["tasks"]), 6)
        self.assertIn("deterministic task-quality slice", plan["classification"])

    def test_plan_all_arms_includes_cpu(self) -> None:
        args = runner.parse_args(["--all-arms"])
        with mock.patch.object(runner.base, "glm_download_active", return_value=False):
            plan = runner.build_plan(args)
        self.assertEqual(plan["selected_arms"], ["iq4_gpu", "q8_gpu", "iq4_cpu", "q8_cpu"])

    def test_score_task_json_exact_accepts_fenced_json(self) -> None:
        task = runner.TASKS[0]
        score = runner.score_task(task, '```json\n{"answer":"55"}\n```')
        self.assertTrue(score["passed"])
        self.assertEqual(score["json_mode"], "fenced")

    def test_score_task_exact_and_word_count(self) -> None:
        exact = runner.TASKS[3]
        self.assertTrue(runner.score_task(exact, "B\n")["passed"])
        self.assertFalse(runner.score_task(exact, "b\n")["passed"])

        words = runner.TASKS[4]
        self.assertTrue(runner.score_task(words, "fresh tests block benchmark leakage")["passed"])
        self.assertFalse(runner.score_task(words, "Fresh tests block benchmark leakage")["passed"])
        self.assertFalse(runner.score_task(words, "fresh tests block benchmark leakage today")["passed"])

    def test_run_arm_writes_results_with_mocked_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args = runner.parse_args(["--output-dir", tmp, "--max-tokens", "96"])
            fake_proc = mock.Mock()
            fake_proc.pid = 123
            fake_proc.poll.return_value = 0

            responses = {
                "arithmetic_sum_json": {"choices": [{"message": {"content": '{"answer":"55"}'}}]},
                "logic_transitive_json": {"choices": [{"message": {"content": '{"answer":"yes"}'}}]},
                "time_addition_json": {"choices": [{"message": {"content": '{"finish_24h":"10:35"}'}}]},
                "option_reasoning_letter": {"choices": [{"message": {"content": "B"}}]},
                "lowercase_five_words": {"choices": [{"message": {"content": "fresh tests block benchmark leakage"}}]},
                "sorted_keys_json": {
                    "choices": [{"message": {"content": '{"label":"qwable","status":"ready"}'}}]
                },
            }

            def fake_query(_port, payload, _timeout):  # noqa: ANN001
                prompt = payload["messages"][-1]["content"]
                task_id = next(task.task_id for task in runner.TASKS if task.prompt == prompt)
                response = {
                    **responses[task_id],
                    "timings": {"predicted_per_second": 100.0, "prompt_per_second": 200.0},
                    "usage": {"completion_tokens": 4},
                }
                return response, json.dumps(response)

            with (
                mock.patch.object(runner.base, "launch_server", return_value=fake_proc),
                mock.patch.object(runner.base, "wait_for_health", return_value=None),
                mock.patch.object(runner.base, "terminate_server", return_value=None),
            ):
                result = runner.run_arm(args, output_dir, 0, query=fake_query)

            self.assertEqual(result["arm"], "iq4_gpu")
            self.assertEqual(result["passed"], 6)
            self.assertEqual(result["total"], 6)
            self.assertEqual(result["mean_decode_tps"], 100.0)
            self.assertTrue((output_dir / "results" / "iq4_gpu.json").exists())


if __name__ == "__main__":
    unittest.main()
