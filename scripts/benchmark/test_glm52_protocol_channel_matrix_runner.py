#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))

import glm52_protocol_channel_matrix_runner as runner


def make_shard_dir(root: Path) -> Path:
    model_dir = root / "GLM-5.2-UD-IQ2_M"
    model_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 7):
        (model_dir / f"glm-shard-{idx:02d}.gguf").write_bytes(b"x" * (idx * 17))
    return model_dir


def make_binary(root: Path) -> tuple[Path, Path]:
    binary_dir = root / "bin"
    binary_dir.mkdir()
    binary = binary_dir / "llama-server"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary, binary_dir


class TestGlm52ProtocolChannelMatrixRunner(unittest.TestCase):
    def test_validate_ready_prefers_content_then_combined(self) -> None:
        mode = runner.RUNTIME_MODES["free_reasoning_off"]

        self.assertTrue(
            runner.validate_response(mode, {"content": "READY", "combined": "READY"})["passed"]
        )
        self.assertTrue(
            runner.validate_response(mode, {"content": "", "combined": "READY"})["passed"]
        )
        failed = runner.validate_response(mode, {"content": "READY now", "combined": "READY now"})
        self.assertFalse(failed["passed"])
        self.assertTrue(failed["combined_contains_expected"])

    def test_validate_json_decision_requires_exact_object(self) -> None:
        mode = runner.RUNTIME_MODES["json_reasoning_off"]

        self.assertTrue(
            runner.validate_response(mode, {"content": '{"decision":"allow"}', "combined": ""})["passed"]
        )
        failed = runner.validate_response(mode, {"content": "explain {\"decision\":\"allow\"}", "combined": ""})
        self.assertFalse(failed["passed"])
        self.assertIn("json_error", failed)

    def test_build_plan_expands_selected_cells_and_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = make_shard_dir(root)
            binary, binary_dir = make_binary(root)
            args = runner.parse_args(
                [
                    "--output-dir",
                    str(root / "matrix"),
                    "--model-dir",
                    str(model_dir),
                    "--binary",
                    str(binary),
                    "--library-path",
                    str(binary_dir),
                    "--bands",
                    "p2168_tk4096",
                    "--modes",
                    "free_reasoning_off,json_reasoning_off",
                    "--endpoints",
                    "chat,completion",
                ]
            )

            plan = runner.build_plan(args)

            self.assertEqual(plan["schema"], runner.SCHEMA)
            self.assertTrue(plan["execution_allowed"])
            self.assertEqual(len(plan["cells"]), 2)
            self.assertEqual(plan["cells"][0]["band"]["indexer_top_k"], 4096)
            self.assertEqual(plan["cells"][0]["request"]["endpoints"], ["chat", "completion"])
            self.assertIn("--reasoning", plan["cells"][0]["server"]["server_command"])
            self.assertIn("--json-schema", plan["cells"][1]["server"]["server_command"])

    def test_extract_channels_handles_chat_v1_and_raw_shapes(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"content": "chat", "reasoning_content": "reason"},
                    "text": "v1",
                }
            ],
            "content": "raw",
        }

        channels = runner.extract_channels(response)

        self.assertEqual(channels["content"], "chat")
        self.assertEqual(channels["reasoning_content"], "reason")
        self.assertEqual(channels["text"], "v1")
        self.assertEqual(channels["raw_content"], "raw")
        self.assertEqual(channels["combined"], "reasonchatv1raw")


if __name__ == "__main__":
    unittest.main()
