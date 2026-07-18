#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import unittest
import tempfile
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import k11_gemma4_determinism_runner as runner


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class TestK11Gemma4DeterminismRunner(unittest.TestCase):
    def test_build_server_argv_pins_experimental_v7_build(self) -> None:
        args = runner.parse_args(["--runs", "2"])
        argv = runner.build_server_argv(args, 31337)

        self.assertEqual(
            argv[:4],
            [
                "env",
                f"LD_LIBRARY_PATH={runner.SERVER_LIB_DIR}",
                "numactl",
                "--interleave=all",
            ],
        )
        self.assertIn(str(runner.SERVER_BIN), argv)
        self.assertEqual(argv[argv.index("-m") + 1], str(runner.DEFAULT_TARGET_MODEL))
        self.assertEqual(argv[argv.index("-md") + 1], str(runner.DEFAULT_DRAFT_MODEL))
        self.assertEqual(argv[argv.index("-np") + 1], str(runner.DEFAULT_SLOTS))
        self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(
            argv[argv.index("--spec-draft-n-max") + 1],
            str(runner.DEFAULT_SPEC_DRAFT_N_MAX),
        )
        self.assertEqual(argv[argv.index("--device") + 1], "ROCm0")
        self.assertEqual(argv[argv.index("--device-draft") + 1], "ROCm0")
        self.assertEqual(argv[argv.index("-rea") + 1], "off")
        self.assertEqual(argv[argv.index("--port") + 1], "31337")

    def test_build_server_argv_supports_no_spec_control(self) -> None:
        args = runner.parse_args(["--spec-type", "none", "--slots", "1"])
        argv = runner.build_server_argv(args, 31337)

        self.assertEqual(argv[argv.index("-np") + 1], "1")
        self.assertEqual(argv[argv.index("--spec-type") + 1], "none")
        self.assertNotIn("-md", argv)
        self.assertNotIn("--device-draft", argv)
        self.assertNotIn("--spec-draft-ngl", argv)

    def test_score_word_task(self) -> None:
        passed = runner.score_word_task("benchmark benchmark", "benchmark", 2)
        failed = runner.score_word_task("benchmark other", "benchmark", 2)

        self.assertEqual(passed["observed_word_count"], 2)
        self.assertTrue(passed["passed"])
        self.assertEqual(failed["bad_word_count"], 1)
        self.assertFalse(failed["passed"])

    def test_query_chat_parses_semantic_response(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            seen["url"] = req.full_url
            seen["payload"] = json.loads(req.data.decode("utf-8"))
            seen["timeout"] = timeout
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "OK",
                                "reasoning_content": "",
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 2},
                    "timings": {"draft_n": 4, "draft_n_accepted": 4},
                }
            )

        with mock.patch.object(runner.urllib.request, "urlopen", fake_urlopen):
            response, raw = runner.query_chat(18080, "prompt", 64, 0.0, 42, 5)

        self.assertEqual(seen["url"], "http://127.0.0.1:18080/v1/chat/completions")
        self.assertEqual(seen["payload"]["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(seen["payload"]["seed"], 42)
        self.assertEqual(seen["payload"]["temperature"], 0.0)
        self.assertEqual(seen["payload"]["top_k"], 1)
        self.assertEqual(response["choices"][0]["message"]["content"], "OK")
        self.assertEqual(json.loads(raw)["timings"]["draft_n_accepted"], 4)

    def test_terminate_server_stops_process_group(self) -> None:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        try:
            runner.terminate_server(proc)
            self.assertIsNotNone(proc.poll())
            self.assertFalse(runner.is_pid_alive(proc.pid))
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_dry_run_writes_plan_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "k11"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("k11_gemma4_determinism_runner.py")),
                    "--output-dir",
                    str(output_dir),
                    "--runs",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("mode: dry_run", result.stdout)
            self.assertIn("Dry run only. No server will be launched.", result.stdout)
            self.assertTrue((output_dir / "plan.json").exists())
            self.assertTrue((output_dir / "commands.sh").exists())

            plan = json.loads((output_dir / "plan.json").read_text())
            self.assertEqual(plan["meta"]["mode"], "dry_run")
            self.assertEqual(plan["meta"]["spec_type"], "draft-mtp")
            self.assertEqual(plan["meta"]["seed"], 42)
            self.assertEqual(plan["meta"]["slots"], runner.DEFAULT_SLOTS)
            self.assertEqual(len(plan["runs"]), 2)
            self.assertEqual(plan["runs"][0]["label"], "run_01")

            commands = (output_dir / "commands.sh").read_text()
            self.assertIn(str(runner.SERVER_BIN), commands)
            self.assertIn("LD_LIBRARY_PATH", commands)
            self.assertIn("-np 4", commands)
            self.assertIn("--device ROCm0", commands)
            self.assertIn("--device-draft ROCm0", commands)
            self.assertIn("--spec-type draft-mtp", commands)


if __name__ == "__main__":
    unittest.main()
