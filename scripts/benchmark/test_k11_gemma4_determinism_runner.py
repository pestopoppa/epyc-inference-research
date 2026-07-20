#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
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


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid


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
        self.assertEqual(argv[argv.index("--device") + 1], runner.DEFAULT_TARGET_DEVICE)
        self.assertEqual(argv[argv.index("--device-draft") + 1], runner.DEFAULT_DRAFT_DEVICE)
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

    def test_build_server_argv_supports_cpu_no_spec_control(self) -> None:
        args = runner.parse_args(
            [
                "--spec-type",
                "none",
                "--target-device",
                "none",
                "--n-gpu-layers",
                "0",
            ]
        )
        argv = runner.build_server_argv(args, 31337)

        self.assertEqual(argv[argv.index("--device") + 1], "none")
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        self.assertEqual(argv[argv.index("--spec-type") + 1], "none")
        self.assertNotIn("--device-draft", argv)
        self.assertNotIn("--spec-draft-ngl", argv)

    def test_build_server_argv_can_disable_draft_backend_sampling(self) -> None:
        args = runner.parse_args(["--draft-backend-sampling", "off"])
        argv = runner.build_server_argv(args, 31337)

        self.assertIn("--no-spec-draft-backend-sampling", argv)
        self.assertNotIn("--spec-draft-backend-sampling", argv)

    def test_build_server_argv_records_extra_server_env(self) -> None:
        args = runner.parse_args(["--server-env", "GGML_CUDA_DISABLE_GRAPHS=1"])
        argv = runner.build_server_argv(args, 31337)

        self.assertEqual(
            argv[:5],
            [
                "env",
                f"LD_LIBRARY_PATH={runner.SERVER_LIB_DIR}",
                "GGML_CUDA_DISABLE_GRAPHS=1",
                "numactl",
                "--interleave=all",
            ],
        )

    def test_server_env_requires_assignment(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--server-env", "GGML_CUDA_DISABLE_GRAPHS"])

            with self.assertRaises(SystemExit):
                runner.parse_args(["--server-env", "=1"])

    def test_apply_request_sampler_mode_explicit_greedy(self) -> None:
        payload = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}

        runner.apply_request_sampler_mode(payload, "explicit-greedy")

        self.assertEqual(payload["samplers"], ["temperature"])
        self.assertEqual(payload["top_k"], 0)
        self.assertEqual(payload["min_p"], 0.0)
        self.assertIs(payload["backend_sampling"], False)

    def test_apply_request_sampler_mode_cpu_top_k(self) -> None:
        payload = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}

        runner.apply_request_sampler_mode(payload, "cpu-top-k")

        self.assertEqual(payload["samplers"], ["top_k", "temperature"])
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["min_p"], 0.0)
        self.assertIs(payload["backend_sampling"], False)

    def test_score_word_task(self) -> None:
        passed = runner.score_word_task("benchmark benchmark", "benchmark", 2)
        failed = runner.score_word_task("benchmark other", "benchmark", 2)

        self.assertEqual(passed["observed_word_count"], 2)
        self.assertTrue(passed["passed"])
        self.assertEqual(failed["bad_word_count"], 1)
        self.assertFalse(failed["passed"])

    def test_schema_task_builds_word_array_schema(self) -> None:
        schema = runner.json_schema_for_schema_task("word-array-200")

        self.assertEqual(schema["properties"]["words"]["minItems"], 200)
        self.assertEqual(schema["properties"]["words"]["maxItems"], 200)
        self.assertEqual(schema["properties"]["words"]["items"]["enum"], ["benchmark"])
        self.assertEqual(schema["properties"]["done"]["enum"], ["END"])
        self.assertFalse(schema["additionalProperties"])

    def test_score_schema_task(self) -> None:
        passed_payload = {"words": ["benchmark"] * 200, "done": "END"}
        failed_payload = {"words": ["benchmark"] * 199 + ["other"], "done": "END"}

        passed = runner.score_schema_task(json.dumps(passed_payload), "word-array-200")
        failed = runner.score_schema_task(json.dumps(failed_payload), "word-array-200")
        parse_failed = runner.score_schema_task("not json", "word-array-200")

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["observed_word_count"], 200)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["bad_word_count"], 1)
        self.assertFalse(parse_failed["json_ok"])

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
        self.assertNotIn("return_tokens", seen["payload"])
        self.assertNotIn("n_probs", seen["payload"])
        self.assertEqual(response["choices"][0]["message"]["content"], "OK")
        self.assertEqual(json.loads(raw)["timings"]["draft_n_accepted"], 4)

    def test_query_chat_sends_stop_strings(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            seen["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse({"choices": [{"message": {"content": "OK"}}]})

        with mock.patch.object(runner.urllib.request, "urlopen", fake_urlopen):
            runner.query_chat(18080, "prompt", 64, 0.0, 42, 5, stop=["DONE", "END"])

        self.assertEqual(seen["payload"]["stop"], ["DONE", "END"])

    def test_query_chat_sends_json_schema(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            seen["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse({"choices": [{"message": {"content": "OK"}}]})

        schema = runner.json_schema_for_schema_task("word-array-200")
        with mock.patch.object(runner.urllib.request, "urlopen", fake_urlopen):
            runner.query_chat(18080, "prompt", 64, 0.0, 42, 5, json_schema=schema)

        self.assertEqual(seen["payload"]["json_schema"], schema)

    def test_query_chat_can_request_token_trace_metadata(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            seen["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse({"choices": [{"message": {"content": "OK"}}]})

        with mock.patch.object(runner.urllib.request, "urlopen", fake_urlopen):
            runner.query_chat(
                18080,
                "prompt",
                64,
                0.0,
                42,
                5,
                trace_token_divergence=True,
                trace_n_probs=7,
                trace_post_sampling_probs=True,
                trace_response_fields=["choices", "tokens"],
            )

        self.assertIs(seen["payload"]["return_tokens"], True)
        self.assertEqual(seen["payload"]["n_probs"], 7)
        self.assertIs(seen["payload"]["post_sampling_probs"], True)
        self.assertEqual(seen["payload"]["response_fields"], ["choices", "tokens"])
        self.assertIs(seen["payload"]["verbose"], True)

    def test_query_chat_can_send_explicit_greedy_payload(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            seen["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "choices": [{"message": {"content": "OK", "reasoning_content": ""}}],
                    "timings": {},
                }
            )

        with mock.patch.object(runner.urllib.request, "urlopen", fake_urlopen):
            runner.query_chat(18080, "prompt", 64, 0.0, 42, 5, "explicit-greedy")

        self.assertEqual(seen["payload"]["samplers"], ["temperature"])
        self.assertEqual(seen["payload"]["top_k"], 0)
        self.assertIs(seen["payload"]["backend_sampling"], False)

    def test_build_token_trace_compacts_chat_logprobs(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "alpha beta"},
                    "logprobs": {
                        "content": [
                            {
                                "id": 11,
                                "token": "alpha",
                                "bytes": [97],
                                "prob": 0.8,
                                "top_probs": [
                                    {"id": 11, "token": "alpha", "prob": 0.8},
                                    {"id": 12, "token": "beta", "prob": 0.2},
                                ],
                            },
                            {"id": 13, "token": " beta", "bytes": [32, 98], "prob": 0.9},
                        ]
                    },
                }
            ],
        }

        trace = runner.build_token_trace(
            label="run_01",
            response=response,
            content="alpha beta",
            finish_reason="stop",
        )

        self.assertEqual(trace["finish_reason"], "stop")
        self.assertEqual(trace["content_word_count"], 2)
        self.assertEqual(trace["token_count"], 2)
        self.assertEqual(trace["sequence"], [11, 13])
        self.assertEqual(trace["sequence_source"], "choices[0].logprobs.content")
        self.assertEqual(trace["tokens"][0]["top_probs"][1]["token"], "beta")

    def test_build_token_trace_prefers_returned_token_ids(self) -> None:
        response = {
            "__verbose": {"tokens": [101, 102]},
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "alpha"},
                    "logprobs": {"content": [{"id": 201, "token": "alpha", "logprob": -0.1}]},
                }
            ],
        }

        trace = runner.build_token_trace(
            label="run_01",
            response=response,
            content="alpha",
            finish_reason="length",
        )

        self.assertEqual(trace["sequence"], [101, 102])
        self.assertEqual(trace["sequence_source"], "__verbose.tokens")
        self.assertEqual(trace["probability_source"], "choices[0].logprobs.content")

    def test_build_token_divergence_summary_reports_first_divergent_token(self) -> None:
        traces = [
            {
                "label": "run_01",
                "finish_reason": "stop",
                "content_word_count": 3,
                "token_count": 3,
                "sequence_source": "tokens",
                "probability_source": "choices[0].logprobs.content",
                "sequence": [10, 20, 30],
                "tokens": [{"id": 10}, {"id": 20}, {"id": 30, "token": "A", "prob": 0.7}],
            },
            {
                "label": "run_02",
                "finish_reason": "stop",
                "content_word_count": 3,
                "token_count": 3,
                "sequence_source": "tokens",
                "probability_source": "choices[0].logprobs.content",
                "sequence": [10, 20, 31],
                "tokens": [{"id": 10}, {"id": 20}, {"id": 31, "token": "B", "prob": 0.6}],
            },
        ]
        results = [{"label": "run_01", "status": "ok"}, {"label": "run_02", "status": "ok"}]

        summary = runner.build_token_divergence_summary(
            enabled=True,
            traces=traces,
            results=results,
        )

        self.assertTrue(summary["available"])
        self.assertEqual(summary["common_prefix_length"], 2)
        self.assertFalse(summary["all_token_sequences_identical"])
        self.assertEqual(summary["first_divergence"]["index"], 2)
        self.assertEqual(summary["first_divergence"]["by_run"][0]["token"]["id"], 30)
        self.assertEqual(summary["first_divergence"]["by_run"][1]["token"]["token"], "B")

    def test_run_execute_writes_token_traces_and_summary_without_server(self) -> None:
        responses = [
            (
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "alpha beta"},
                            "logprobs": {
                                "content": [
                                    {"id": 1, "token": "alpha", "prob": 0.8},
                                    {"id": 2, "token": " beta", "prob": 0.7},
                                ]
                            },
                        }
                    ],
                    "usage": {"completion_tokens": 2},
                    "timings": {"draft_n": 2, "draft_n_accepted": 1},
                },
                '{"run":1}',
            ),
            (
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "alpha gamma"},
                            "logprobs": {
                                "content": [
                                    {"id": 1, "token": "alpha", "prob": 0.8},
                                    {"id": 3, "token": " gamma", "prob": 0.6},
                                ]
                            },
                        }
                    ],
                    "usage": {"completion_tokens": 2},
                    "timings": {"draft_n": 2, "draft_n_accepted": 2},
                },
                '{"run":2}',
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "k11"
            args = runner.parse_args(
                [
                    "--execute",
                    "--runs",
                    "2",
                    "--output-dir",
                    str(output_dir),
                    "--trace-token-divergence",
                ]
            )
            with (
                mock.patch.object(runner, "pick_ephemeral_port", side_effect=[18081, 18082]),
                mock.patch.object(runner, "launch_server", side_effect=[_FakeProc(101), _FakeProc(102)]),
                mock.patch.object(runner, "wait_for_health"),
                mock.patch.object(runner, "query_chat", side_effect=responses),
                mock.patch.object(runner, "terminate_server"),
            ):
                summary = runner.run_execute(args, output_dir)

            self.assertEqual(summary["finish_reasons"], {"stop": 1, "length": 1})
            self.assertEqual(summary["content_word_counts"], [2, 2])
            self.assertEqual(summary["token_divergence"]["common_prefix_length"], 1)
            self.assertEqual(summary["token_divergence"]["first_divergence"]["index"], 1)
            self.assertTrue((output_dir / "token_traces" / "run_01.tokens.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            run_record = json.loads((output_dir / "runs" / "run_01.json").read_text())
            self.assertEqual(run_record["finish_reason"], "stop")
            self.assertEqual(run_record["content_word_count"], 2)
            self.assertEqual(run_record["token_trace"]["token_count"], 2)

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
                    "--stop",
                    "DONE",
                    "--stop",
                    "END",
                    "--schema-task",
                    "word-array-200",
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
            self.assertEqual(plan["meta"]["stop"], ["DONE", "END"])
            self.assertEqual(plan["meta"]["schema_task"], "word-array-200")
            self.assertEqual(plan["meta"]["json_schema"]["properties"]["done"]["enum"], ["END"])
            self.assertFalse(plan["meta"]["trace_token_divergence"])
            self.assertEqual(plan["meta"]["slots"], runner.DEFAULT_SLOTS)
            self.assertEqual(plan["meta"]["target_device"], runner.DEFAULT_TARGET_DEVICE)
            self.assertEqual(plan["meta"]["draft_device"], runner.DEFAULT_DRAFT_DEVICE)
            self.assertEqual(len(plan["runs"]), 2)
            self.assertEqual(plan["runs"][0]["label"], "run_01")

            commands = (output_dir / "commands.sh").read_text()
            self.assertIn(str(runner.SERVER_BIN), commands)
            self.assertIn("LD_LIBRARY_PATH", commands)
            self.assertIn("-np 4", commands)
            self.assertIn("--device ROCm0", commands)
            self.assertIn("--device-draft ROCm0", commands)
            self.assertIn("--spec-type draft-mtp", commands)

    def test_dry_run_records_token_trace_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "k11"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("k11_gemma4_determinism_runner.py")),
                    "--output-dir",
                    str(output_dir),
                    "--runs",
                    "1",
                    "--trace-token-divergence",
                    "--trace-n-probs",
                    "7",
                    "--no-trace-post-sampling-probs",
                    "--trace-response-field",
                    "choices",
                    "--trace-response-field",
                    "tokens",
                    "--server-env",
                    "GGML_CUDA_DISABLE_GRAPHS=1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            plan = json.loads((output_dir / "plan.json").read_text())
            self.assertTrue(plan["meta"]["trace_token_divergence"])
            self.assertEqual(plan["meta"]["trace_n_probs"], 7)
            self.assertFalse(plan["meta"]["trace_post_sampling_probs"])
            self.assertEqual(plan["meta"]["trace_response_fields"], ["choices", "tokens"])
            self.assertEqual(plan["meta"]["server_env"], ["GGML_CUDA_DISABLE_GRAPHS=1"])
            self.assertIn(
                "GGML_CUDA_DISABLE_GRAPHS=1",
                (output_dir / "commands.sh").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
