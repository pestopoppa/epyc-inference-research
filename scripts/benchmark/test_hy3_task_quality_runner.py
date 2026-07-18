#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).parent))

import hy3_task_quality_runner as runner


class Hy3TaskQualityRunnerTests(unittest.TestCase):
    def test_plan_defaults_to_hybrid_then_cpu(self) -> None:
        args = runner.parse_args([])
        with (
            mock.patch.object(runner.base, "glm_download_active", return_value=False),
            mock.patch.object(runner, "list_llama_server_pids", return_value=[]),
            mock.patch.object(runner, "validate_experimental_server", return_value=runner.base.SERVER_BIN),
        ):
            plan = runner.build_plan(args)

        self.assertEqual(plan["schema"], "hy3_task_quality_plan.v1")
        self.assertEqual(plan["selected_arms"], ["hybrid_nospec", "cpu_nospec"])
        self.assertIn("not runnable on one MI210", plan["gpu_only_disposition"])
        self.assertEqual(len(plan["tasks"]), 6)

    def test_selected_arms_can_target_cpu_only(self) -> None:
        args = runner.parse_args(["--only", "cpu_nospec"])
        self.assertEqual(runner.selected_arm_indices(args), [1])

    def test_launch_argv_records_hybrid_flags_only_for_hybrid(self) -> None:
        args = runner.parse_args([])

        hybrid = runner.launch_argv(runner.ARMS[0], 19240, args)
        cpu = runner.launch_argv(runner.ARMS[1], 19250, args)

        self.assertIn("--cpu-moe", hybrid)
        self.assertIn("--fit", hybrid)
        self.assertIn("ROCm0", hybrid)
        self.assertNotIn("--cpu-moe", cpu)
        self.assertIn("none", cpu)

    def test_score_reused_tasks(self) -> None:
        self.assertTrue(runner.quality.score_task(runner.TASKS[0], '{"status":"ok","model":"hy3"}')["passed"])
        self.assertTrue(runner.quality.score_task(runner.TASKS[1], "95")["passed"])
        self.assertTrue(
            runner.quality.score_task(runner.TASKS[2], "routing saves work by activating experts")["passed"]
        )

    def test_run_arm_writes_result_with_mocked_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args = runner.parse_args(["--output-dir", tmp])
            fake_proc = mock.Mock()
            fake_proc.pid = 12345
            fake_proc.poll.return_value = 0

            response_by_prompt = {
                runner.TASKS[0].prompt: '{"status":"ok","model":"hy3"}',
                runner.TASKS[1].prompt: "95",
                runner.TASKS[2].prompt: "routing saves work by activating experts",
                runner.TASKS[3].prompt: (
                    "Sparse routing reduces compute and memory bandwidth pressure. "
                    "It needs load balancing because routing overhead can erase wins."
                ),
                runner.TASKS[4].prompt: "HY3-DELTA-9421",
                runner.TASKS[5].prompt: "def binary_search(arr, target):\n    while False:\n        pass\n    return -1",
            }

            def fake_query(_port: int, payload: dict, _timeout: int) -> tuple[dict, str]:
                prompt = payload["messages"][-1]["content"]
                content = response_by_prompt[prompt]
                response = {
                    "choices": [{"message": {"content": content}}],
                    "timings": {"predicted_per_second": 10.0, "prompt_per_second": 20.0},
                    "usage": {"completion_tokens": 4},
                }
                return response, json.dumps(response)

            with (
                mock.patch.object(runner.base, "launch_server", return_value=fake_proc),
                mock.patch.object(runner.base, "wait_for_health", return_value=None),
                mock.patch.object(runner.base, "terminate_server", return_value=None),
            ):
                result = runner.run_arm(args, output_dir, 0, query=fake_query)

            self.assertEqual(result["arm"], "hybrid_nospec")
            self.assertEqual(result["passed"], 6)
            self.assertEqual(result["total"], 6)
            self.assertEqual(result["mean_decode_tps"], 10.0)
            self.assertTrue((output_dir / "results" / "hybrid_nospec.json").exists())

    def test_run_execute_checks_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args = runner.parse_args(["--output-dir", tmp, "--only", "cpu_nospec"])
            plan = {
                "classification": "test classification",
                "gpu_only_disposition": "not runnable",
                "preexisting_llama_server_pids": [111],
            }
            arm_result = {
                "arm": "cpu_nospec",
                "passed": 6,
                "total": 6,
                "mean_decode_tps": 3.0,
                "mean_prompt_tps": 20.0,
            }

            with (
                mock.patch.object(runner, "run_arm", return_value=arm_result),
                mock.patch.object(runner, "verify_cleanup", return_value={"passed": True, "extra_pids": []}),
            ):
                summary = runner.run_execute(args, output_dir, plan)

            self.assertTrue(summary["quality_gate_passed"])
            self.assertTrue((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
