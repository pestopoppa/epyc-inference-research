#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import sys

sys.path.insert(0, str(Path(__file__).parent))

import qwable_reasoning_economics_runner as runner


class TestQwableReasoningEconomicsRunner(TestCase):
    def test_dry_run_writes_plan_with_expected_arms_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "qwable"
            stdout = io.StringIO()
            with mock.patch.object(runner, "glm_download_active", return_value=False):
                with contextlib.redirect_stdout(stdout):
                    rc = runner.main(["--output-dir", str(output_dir), "--port-base", "19100"])

            self.assertEqual(rc, 0)
            self.assertIn("mode: dry_run", stdout.getvalue())
            self.assertTrue((output_dir / "plan.json").exists())
            self.assertTrue((output_dir / "commands.sh").exists())

            plan = json.loads((output_dir / "plan.json").read_text())
            self.assertEqual(plan["schema"], "qwable_reasoning_economics_plan.v1")
            self.assertEqual(plan["mode"], "dry_run")
            self.assertEqual(
                plan["model_paths"]["iq4_xs"],
                "/mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf",
            )
            self.assertEqual(
                plan["model_paths"]["q8_0"],
                "/mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf",
            )
            self.assertEqual(plan["resource_summary"]["iq4_xs"], "can co-reside more plausibly")
            self.assertEqual(plan["resource_summary"]["q8_0"], "sequential or smaller-beneficiary only")

            arm_names = [arm["name"] for arm in plan["arms"]]
            self.assertEqual(
                arm_names,
                [
                    "standalone_iq4_gpu",
                    "standalone_q8_gpu",
                    "strict_iq4_json_gpu",
                    "cpu_iq4_baseline",
                    "scaffold_then_beneficiary_stub",
                    "verifier_selector_stub",
                ],
            )
            self.assertEqual(plan["execution"]["selected_smoke_arms"], ["standalone_iq4_gpu"])

            first_arm = plan["arms"][0]
            self.assertIn("env -i", first_arm["commands"]["launch"])
            self.assertIn(
                "LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin",
                first_arm["commands"]["launch"],
            )
            self.assertIn(
                "/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server",
                first_arm["commands"]["launch"],
            )
            self.assertIn("--device ROCm0", first_arm["commands"]["launch"])
            self.assertIn("--port 19100", first_arm["commands"]["launch"])
            self.assertIn("curl -fsS", first_arm["commands"]["smoke"])

            q8_arm = plan["arms"][1]
            self.assertIn(
                "/mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf",
                q8_arm["commands"]["launch"],
            )
            self.assertEqual(q8_arm["resource_notes"]["co_residency_policy"], "sequential_only")
            self.assertEqual(q8_arm["resource_notes"]["beneficiary_policy"], "smaller-beneficiary-only")

            strict_arm = plan["arms"][2]
            self.assertEqual(strict_arm["role"], "strict_json_reasoner")
            self.assertIn("strict_iq4_json_gpu", strict_arm["commands"]["smoke"])

            cpu_arm = plan["arms"][3]
            self.assertIn("--device none", cpu_arm["commands"]["launch"])
            self.assertIn("-ngl 0", cpu_arm["commands"]["launch"])

    def test_dry_run_records_selected_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "qwable"
            with mock.patch.object(runner, "glm_download_active", return_value=False):
                rc = runner.main(
                    [
                        "--output-dir",
                        str(output_dir),
                        "--only",
                        "strict_iq4_json_gpu",
                        "--only",
                        "standalone_q8_gpu",
                    ]
                )

            self.assertEqual(rc, 0)
            plan = json.loads((output_dir / "plan.json").read_text())
            self.assertEqual(
                plan["execution"]["selected_smoke_arms"],
                ["standalone_q8_gpu", "strict_iq4_json_gpu"],
            )

    def test_response_summary_classifies_strict_and_fenced_json(self) -> None:
        strict = runner.response_summary(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"arm":"x"}'}}
                ],
                "usage": {"completion_tokens": 4},
            }
        )
        self.assertEqual(strict["content_json_mode"], "strict")
        self.assertEqual(strict["content_json"], {"arm": "x"})

        fenced = runner.response_summary(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '```json\n{"arm":"x"}\n```'}}
                ]
            }
        )
        self.assertEqual(fenced["content_json_mode"], "fenced")

    def test_execute_refuses_active_glm_download_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "qwable"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(runner, "glm_download_active", return_value=True):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    rc = runner.main(["--execute", "--output-dir", str(output_dir)])

            self.assertEqual(rc, 75)
            self.assertIn("GLM-5.2 download is active", stderr.getvalue())
            self.assertFalse((output_dir / "plan.json").exists())
            self.assertFalse((output_dir / "commands.sh").exists())
