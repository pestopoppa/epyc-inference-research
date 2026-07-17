#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).parent))

import nemotron_nano_task_quality_runner as runner


class NemotronNanoTaskQualityRunnerTests(unittest.TestCase):
    def test_extract_message_content_ignores_reasoning_content(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "reasoning_content": "hidden chain",
                    }
                }
            ]
        }
        self.assertEqual(runner.extract_message_content(response), "ok")

    def test_build_plan_records_existing_pids_and_q8_kv(self) -> None:
        args = runner.parse_args(["--port", "19122"])
        with (
            mock.patch.object(runner, "detect_glm_pids", return_value=[2362944]),
            mock.patch.object(runner, "list_llama_server_pids", return_value=[2362958]),
            mock.patch.object(runner, "detect_q8_kv_support", return_value=True),
            mock.patch.object(runner, "pick_available_port", return_value=19122),
        ):
            plan = runner.build_plan(args)

        self.assertEqual(plan["preexisting_server_pids"], [2362958])
        self.assertEqual(plan["concurrent_glm_probe_pids"], [2362944])
        self.assertEqual(plan["server"]["kv_cache"], "q8_0/q8_0")
        self.assertEqual(plan["server"]["port"], 19122)

    def test_score_task_json_and_word_count(self) -> None:
        self.assertTrue(runner.score_task(runner.TASKS[0], "ok\n")["passed"])
        self.assertTrue(runner.score_task(runner.TASKS[1], '{"status":"ok","model":"nemotron"}')["passed"])
        self.assertTrue(
            runner.score_task(runner.TASKS[3], "fresh benchmarks catch hidden leakage")["passed"]
        )
        self.assertFalse(
            runner.score_task(runner.TASKS[3], "Fresh benchmarks catch hidden leakage")["passed"]
        )

    def test_run_execute_writes_summary_and_cleanup_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args = runner.parse_args(["--output-dir", tmp, "--max-tokens", "160"])
            plan = {
                "classification": "quality-only gate; throughput is contaminated/non-decision while concurrent CPU GLM work is active",
                "server": {
                    "argv": ["numactl", "--interleave=all", str(runner.SERVER_BIN)],
                    "port": 19122,
                    "q8_kv_supported": True,
                },
                "cleanup_expectation": {"allowed_pids_after_run": [2362958]},
            }
            fake_proc = mock.Mock()
            fake_proc.pid = 2500001
            fake_proc.poll.return_value = 0

            def fake_query(_port: int, payload: dict, _timeout: int) -> tuple[dict, str]:
                prompt = payload["messages"][-1]["content"]
                response_map = {
                    runner.TASKS[0].prompt: "ok",
                    runner.TASKS[1].prompt: '{"status":"ok","model":"nemotron"}',
                    runner.TASKS[2].prompt: "95",
                    runner.TASKS[3].prompt: "fresh benchmarks catch hidden leakage",
                    runner.TASKS[4].prompt: "NN-4242-DELTA",
                }
                content = response_map[prompt]
                response = {
                    "choices": [{"message": {"content": content, "reasoning_content": "ignore me"}}],
                    "timings": {"predicted_per_second": 100.0, "prompt_per_second": 250.0},
                    "usage": {"completion_tokens": 4},
                }
                return response, json.dumps(response)

            with (
                mock.patch.object(runner, "launch_server", return_value=fake_proc),
                mock.patch.object(runner, "wait_for_health", return_value=None),
                mock.patch.object(runner, "query_chat", side_effect=fake_query),
                mock.patch.object(runner, "terminate_server", return_value=None),
                mock.patch.object(
                    runner,
                    "verify_cleanup",
                    return_value={
                        "allowed_pids": [2362958],
                        "observed_pids": [2362958],
                        "extra_pids": [],
                        "missing_allowed_pids": [],
                        "passed": True,
                    },
                ),
            ):
                summary = runner.run_execute(args, output_dir, plan)

            self.assertTrue(summary["quality_gate_passed"])
            self.assertTrue(summary["cleanup"]["passed"])
            self.assertTrue((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
