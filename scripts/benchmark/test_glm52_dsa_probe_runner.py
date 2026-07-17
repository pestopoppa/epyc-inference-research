#!/usr/bin/env python3
from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

import sys

sys.path.insert(0, str(Path(__file__).parent))

import glm52_dsa_probe_runner as runner


def _make_shard_dir(root: Path, *, include_blocker: bool = False) -> Path:
    model_dir = root / "GLM-5.2-UD-IQ2_M"
    model_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 7):
        (model_dir / f"glm-shard-{idx:02d}.gguf").write_bytes(b"x" * (idx * 11))
    if include_blocker:
        (model_dir / "download.partial.incomplete").write_text("", encoding="utf-8")
    return model_dir


def _write_matching_hf_tree(model_dir: Path) -> None:
    tree_dir = model_dir / ".cache" / "huggingface" / "trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for shard in sorted(model_dir.glob("*.gguf")):
        rel = shard.relative_to(model_dir).as_posix()
        size = shard.stat().st_size
        files[rel] = {"size": size, "lfs_size": size, "lfs_sha256": f"sha-{shard.name}"}
    (tree_dir / "revision.json").write_text(json.dumps({"files": files}), encoding="utf-8")


class TestGlm52DsaProbeRunner(unittest.TestCase):
    def test_collect_inventory_blocks_incomplete_download_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_shard_dir(Path(tmp), include_blocker=True)

            inventory = runner.collect_inventory(model_dir)

            self.assertEqual(inventory["status"], "blocked")
            self.assertEqual(inventory["non_cache_shard_count"], 6)
            self.assertTrue(any(item["path"].endswith(".incomplete") for item in inventory["blocker_files"]))
            self.assertTrue(any("download.partial.incomplete" in reason for reason in inventory["refusal_reasons"]))

    def test_collect_inventory_ignores_stale_hf_incomplete_after_manifest_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_shard_dir(Path(tmp))
            _write_matching_hf_tree(model_dir)
            stale_dir = model_dir / ".cache" / "huggingface" / "download" / "UD-IQ2_M"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "old-body.incomplete").write_bytes(b"stale")

            inventory = runner.collect_inventory(model_dir)

            self.assertEqual(inventory["status"], "ready")
            self.assertEqual(inventory["hf_tree_manifest"]["status"], "complete")
            self.assertEqual(inventory["blocker_files"], [])
            self.assertEqual(len(inventory["stale_cache_marker_files"]), 1)

    def test_build_plan_uses_experimental_binary_and_sanitized_library_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _make_shard_dir(tmp_path)
            binary_dir = tmp_path / "bin"
            binary_dir.mkdir()
            binary = binary_dir / "llama-server"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)

            args = runner.parse_args(
                [
                    "--output",
                    str(tmp_path / "plan.json"),
                    "--model-dir",
                    str(model_dir),
                    "--binary",
                    str(binary),
                    "--library-path",
                    str(binary_dir),
                    "--kv-contexts",
                    "4096",
                    "8192",
                ]
            )
            inventory = runner.collect_inventory(model_dir)
            plan = runner.build_plan(args, inventory, runner.resolve_binary(binary), runner.resolve_library_path(binary, binary_dir))

            short_stage = plan["stages"][1]
            kv_stage = plan["stages"][3]
            expected_model = str((model_dir / "glm-shard-01.gguf").resolve())

            self.assertEqual(plan["schema"], runner.SCHEMA)
            self.assertTrue(plan["execution_allowed"])
            self.assertEqual(plan["model_path"], expected_model)
            self.assertEqual(short_stage["server"]["server_command"][:5], [
                "env",
                "-i",
                "PATH=/usr/bin:/bin",
                f"LD_LIBRARY_PATH={binary_dir.resolve()}",
                "OMP_NUM_THREADS=1",
            ])
            self.assertEqual(short_stage["server"]["server_command"][5:8], ["numactl", "--interleave=all", str(binary.resolve())])
            self.assertEqual(
                short_stage["server"]["server_command"][
                    short_stage["server"]["server_command"].index("-m") + 1
                ],
                expected_model,
            )
            self.assertEqual(
                short_stage["server"]["server_command"][
                    short_stage["server"]["server_command"].index("--override-kv") + 1
                ],
                f"{runner.INDEXER_TOP_K_OVERRIDE_KEY}=int:{runner.DEFAULT_INDEXER_TOP_K}",
            )
            self.assertEqual(short_stage["server"]["server_command"][short_stage["server"]["server_command"].index("-c") + 1], str(runner.DEFAULT_SHORT_CONTEXT))
            self.assertEqual(kv_stage["fixed_indexer_top_k"], runner.DEFAULT_INDEXER_TOP_K)
            self.assertEqual([item["context_length"] for item in kv_stage["series"]], [4096, 8192])
            self.assertIn("--device", short_stage["server"]["server_command"])
            self.assertIn("none", short_stage["server"]["server_command"])
            self.assertIn("--log-disable", short_stage["server"]["server_command"])
            self.assertEqual(short_stage["request"]["endpoint"], "/v1/chat/completions")

    def test_trace_logs_and_stage_selection_are_reflected_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _make_shard_dir(tmp_path)
            binary_dir = tmp_path / "bin"
            binary_dir.mkdir()
            binary = binary_dir / "llama-server"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            output = tmp_path / "probe" / "plan.json"

            args = runner.parse_args(
                [
                    "--output",
                    str(output),
                    "--model-dir",
                    str(model_dir),
                    "--binary",
                    str(binary),
                    "--library-path",
                    str(binary_dir),
                    "--trace-logs",
                    "--only-stage",
                    "long_context_dsa_probe",
                    "--only-stage",
                    "kv_length_scaling",
                ]
            )
            inventory = runner.collect_inventory(model_dir)
            plan = runner.build_plan(args, inventory, runner.resolve_binary(binary), runner.resolve_library_path(binary, binary_dir))

            self.assertEqual(plan["selected_stages"], ["kv_length_scaling", "long_context_dsa_probe"])
            long_command = plan["stages"][2]["server"]["server_command"]
            self.assertNotIn("--log-disable", long_command)
            self.assertIn("--log-verbosity", long_command)
            self.assertIn("--log-file", long_command)
            self.assertEqual(
                plan["stages"][2]["server"]["log_file"],
                str(output.parent / "logs" / "long_context_dsa_probe.server.log"),
            )
            self.assertEqual(plan["stages"][2]["prompt"]["min_prompt_tokens"], runner.DEFAULT_MIN_PROMPT_TOKENS)
            self.assertEqual(
                plan["stages"][2]["prompt"]["prompt_context_guard_tokens"],
                runner.DEFAULT_PROMPT_CONTEXT_GUARD_TOKENS,
            )

    def test_long_output_plan_enables_metrics_and_completion_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _make_shard_dir(tmp_path)
            binary_dir = tmp_path / "bin"
            binary_dir.mkdir()
            binary = binary_dir / "llama-server"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)

            args = runner.parse_args(
                [
                    "--output",
                    str(tmp_path / "plan.json"),
                    "--model-dir",
                    str(model_dir),
                    "--binary",
                    str(binary),
                    "--library-path",
                    str(binary_dir),
                    "--long-output",
                    "--throughput-max-tokens",
                    "640",
                    "--min-completion-tokens",
                    "320",
                    "--progress-poll-interval",
                    "7",
                    "--server-extra-arg=--spec-type",
                    "--server-extra-arg=ngram-mod",
                ]
            )
            inventory = runner.collect_inventory(model_dir)
            plan = runner.build_plan(args, inventory, runner.resolve_binary(binary), runner.resolve_library_path(binary, binary_dir))

            short_stage = plan["stages"][1]
            long_stage = plan["stages"][2]

            self.assertFalse(short_stage["server"]["metrics"])
            self.assertNotIn("--metrics", short_stage["server"]["server_command"])
            self.assertTrue(long_stage["server"]["metrics"])
            self.assertIn("--metrics", long_stage["server"]["server_command"])
            self.assertTrue(long_stage["server"]["trace_logs"])
            self.assertNotIn("--log-disable", long_stage["server"]["server_command"])
            self.assertIn("--log-file", long_stage["server"]["server_command"])
            self.assertEqual(
                long_stage["server"]["log_file"],
                str(tmp_path / "logs" / "long_context_dsa_probe.server.log"),
            )
            self.assertEqual(long_stage["request"]["purpose"], "coherence_plus_throughput")
            self.assertTrue(long_stage["request"]["stream"])
            self.assertEqual(long_stage["request"]["max_tokens"], 640)
            self.assertEqual(long_stage["request"]["min_completion_tokens"], 320)
            self.assertEqual(long_stage["request"]["progress_poll_interval_s"], 7)
            self.assertEqual(long_stage["server"]["extra_args"], ["--spec-type", "ngram-mod"])
            self.assertIn("--spec-type", long_stage["server"]["server_command"])
            self.assertIn("ngram-mod", long_stage["server"]["server_command"])
            self.assertIn("tokenstream", long_stage["prompt"]["task_line"])
            self.assertEqual(long_stage["prompt"]["answer_instruction"], runner.LONG_OUTPUT_ANSWER_INSTRUCTION)

    def test_long_context_needle_options_are_reflected_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _make_shard_dir(tmp_path)
            binary_dir = tmp_path / "bin"
            binary_dir.mkdir()
            binary = binary_dir / "llama-server"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)

            args = runner.parse_args(
                [
                    "--output",
                    str(tmp_path / "plan.json"),
                    "--model-dir",
                    str(model_dir),
                    "--binary",
                    str(binary),
                    "--library-path",
                    str(binary_dir),
                    "--long-task-line",
                    "Return only the recovery code hidden in the context.",
                    "--long-answer-instruction",
                    "Answer with the recovery code only.",
                    "--long-needle-text",
                    "Recovery code: GLM52-NEEDLE-7F3A.",
                    "--long-needle-depth",
                    "0.75",
                    "--long-expected-substring",
                    "GLM52-NEEDLE-7F3A",
                ]
            )
            inventory = runner.collect_inventory(model_dir)
            plan = runner.build_plan(args, inventory, runner.resolve_binary(binary), runner.resolve_library_path(binary, binary_dir))

            long_prompt = plan["stages"][2]["prompt"]

            self.assertEqual(long_prompt["task_line"], "Return only the recovery code hidden in the context.")
            self.assertEqual(long_prompt["answer_instruction"], "Answer with the recovery code only.")
            self.assertEqual(long_prompt["needle_text"], "Recovery code: GLM52-NEEDLE-7F3A.")
            self.assertEqual(long_prompt["needle_depth"], 0.75)
            self.assertEqual(long_prompt["expected_substring"], "GLM52-NEEDLE-7F3A")

    def test_prompt_builder_inserts_needle_text(self) -> None:
        prompt = runner._build_prompt_from_chars(
            "Return the recovery code.",
            4000,
            answer_instruction="Answer with only the code.",
            needle_text="Recovery code: GLM52-NEEDLE-7F3A.",
            needle_depth=0.5,
        )

        self.assertIn("--- NEEDLE RECORD ---", prompt)
        self.assertIn("GLM52-NEEDLE-7F3A", prompt)
        self.assertIn("Return the recovery code.", prompt)

    def test_prompt_token_floor_expands_until_live_tokenizer_count_passes_minimum(self) -> None:
        counts: list[int] = []

        def fake_counter(prompt: str) -> int:
            count = len(prompt) // 6
            counts.append(count)
            return count

        result = runner.build_prompt_with_token_floor(
            task_line="Return READY.",
            context_length=90000,
            min_prompt_tokens=65536,
            max_completion_tokens=16,
            prompt_context_guard_tokens=512,
            token_counter=fake_counter,
        )

        self.assertGreaterEqual(result["prompt_token_count"], 65536)
        self.assertLessEqual(result["prompt_token_count"], result["prompt_token_max"])
        self.assertGreater(result["prompt_token_adjustment_attempts"], 1)
        self.assertGreater(len(counts), 1)

    def test_prompt_token_floor_rejects_unreachable_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds safe prompt budget"):
            runner.build_prompt_with_token_floor(
                task_line="Return READY.",
                context_length=65536,
                min_prompt_tokens=65536,
                max_completion_tokens=16,
                prompt_context_guard_tokens=512,
                token_counter=lambda prompt: len(prompt),
            )

    def test_run_execution_skips_unselected_stages(self) -> None:
        plan = {
            "selected_stages": ["long_context_dsa_probe"],
            "stages": [
                {"name": "shard_integrity_inventory", "kind": "inventory", "status": "ready"},
                {
                    "name": "long_context_dsa_probe",
                    "kind": "long_context_probe",
                    "status": "ready",
                    "prompt": {"task_line": "x", "context_length": 1, "kind": "long_context_probe"},
                    "server": {"server_command": [], "port": 1, "context_length": 1, "log_file": None},
                    "request": {"max_tokens": 1, "temperature": 0.0, "seed": 1},
                },
                {"name": "kv_length_scaling", "kind": "kv_length_scaling", "status": "ready", "series": []},
            ],
        }
        expected = {
            "name": "long_context_dsa_probe",
            "status": "ok",
            "port": 1,
            "context_length": 1,
            "prompt_kind": "long_context_probe",
        }
        original = runner.run_stage
        try:
            runner.run_stage = lambda stage: expected  # type: ignore[assignment]
            result = runner.run_execution(plan)
        finally:
            runner.run_stage = original  # type: ignore[assignment]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stages"][0]["status"], "skipped")
        self.assertEqual(result["stages"][1], expected)
        self.assertEqual(result["stages"][2]["reason"], "not selected")

    def test_run_execution_fails_on_acceptance_failure(self) -> None:
        plan = {
            "selected_stages": ["long_context_dsa_probe"],
            "stages": [
                {
                    "name": "long_context_dsa_probe",
                    "kind": "long_context_probe",
                    "status": "ready",
                    "prompt": {"task_line": "x", "context_length": 1, "kind": "long_context_probe"},
                    "server": {"server_command": [], "port": 1, "context_length": 1, "log_file": None},
                    "request": {"max_tokens": 1, "temperature": 0.0, "seed": 1},
                },
            ],
        }
        expected = {
            "name": "long_context_dsa_probe",
            "status": "failed_acceptance",
            "port": 1,
            "context_length": 1,
            "prompt_kind": "long_context_probe",
        }
        original = runner.run_stage
        try:
            runner.run_stage = lambda stage: expected  # type: ignore[assignment]
            result = runner.run_execution(plan)
        finally:
            runner.run_stage = original  # type: ignore[assignment]

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"][0], expected)

    def test_run_execution_fails_on_nested_kv_series_failure(self) -> None:
        plan = {
            "selected_stages": ["kv_length_scaling"],
            "stages": [
                {
                    "name": "kv_length_scaling",
                    "kind": "kv_length_scaling",
                    "status": "ready",
                    "fixed_indexer_top_k": 32,
                    "series": [
                        {
                            "prompt": {"task_line": "x", "context_length": 1, "kind": "kv_length_scaling"},
                            "server": {"server_command": [], "port": 1, "context_length": 1, "log_file": None},
                            "request": {"max_tokens": 1, "temperature": 0.0, "seed": 1},
                        }
                    ],
                },
            ],
        }
        original = runner.run_stage
        try:
            runner.run_stage = lambda stage: {"name": "kv_length_scaling", "status": "failed_acceptance"}  # type: ignore[assignment]
            result = runner.run_execution(plan)
        finally:
            runner.run_stage = original  # type: ignore[assignment]

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"][0]["status"], "ok")
        self.assertEqual(result["stages"][0]["series"][0]["status"], "failed_acceptance")

    def test_run_stage_records_completion_floor_pass(self) -> None:
        stage = {
            "name": "long_context_dsa_probe",
            "prompt": {
                "task_line": "Return READY.",
                "context_length": 4096,
                "kind": "long_context_probe",
                "answer_instruction": runner.SHORT_ANSWER_INSTRUCTION,
            },
            "server": {"server_command": [], "port": 1, "context_length": 4096, "log_file": None, "metrics": False},
            "request": {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 1,
                "timeout_s": 60,
                "min_completion_tokens": 2,
                "progress_poll_interval_s": 0,
                "purpose": "coherence_plus_throughput",
            },
        }

        class FakeProc:
            pid = None

        originals = (
            runner.launch_server,
            runner.wait_for_health,
            runner.count_prompt_tokens,
            runner.call_completion,
            runner.terminate_server,
        )
        try:
            runner.launch_server = lambda command: FakeProc()  # type: ignore[assignment]
            runner.wait_for_health = lambda port: None  # type: ignore[assignment]
            runner.count_prompt_tokens = lambda port, prompt, timeout_s: 100  # type: ignore[assignment]
            runner.call_completion = lambda port, prompt, max_tokens, temperature, seed, timeout_s: {  # type: ignore[assignment]
                "usage": {"prompt_tokens": 100, "completion_tokens": 3},
                "timings": {"predicted_per_second": 12.5},
                "choices": [{"message": {"content": "READY tokenstream tokenstream"}, "finish_reason": "stop"}],
            }
            runner.terminate_server = lambda proc: None  # type: ignore[assignment]
            result = runner.run_stage(stage)
        finally:
            (
                runner.launch_server,
                runner.wait_for_health,
                runner.count_prompt_tokens,
                runner.call_completion,
                runner.terminate_server,
            ) = originals  # type: ignore[assignment]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["request_purpose"], "coherence_plus_throughput")
        self.assertEqual(result["completion_token_min"], 2)
        self.assertEqual(result["completion_token_count"], 3)
        self.assertTrue(result["completion_token_min_passed"])
        self.assertIsNone(result["expected_substring_passed"])

    def test_run_stage_rejects_missing_expected_substring(self) -> None:
        stage = {
            "name": "long_context_dsa_probe",
            "prompt": {
                "task_line": "Return the hidden code.",
                "context_length": 4096,
                "kind": "long_context_probe",
                "answer_instruction": runner.SHORT_ANSWER_INSTRUCTION,
                "needle_text": "Recovery code: GLM52-NEEDLE-7F3A.",
                "needle_depth": 0.5,
                "expected_substring": "GLM52-NEEDLE-7F3A",
            },
            "server": {"server_command": [], "port": 1, "context_length": 4096, "log_file": None, "metrics": False},
            "request": {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 1,
                "timeout_s": 60,
                "min_completion_tokens": 1,
                "progress_poll_interval_s": 0,
                "purpose": "coherence_plus_throughput",
            },
        }

        class FakeProc:
            pid = None

        originals = (
            runner.launch_server,
            runner.wait_for_health,
            runner.count_prompt_tokens,
            runner.call_completion,
            runner.terminate_server,
        )
        try:
            runner.launch_server = lambda command: FakeProc()  # type: ignore[assignment]
            runner.wait_for_health = lambda port: None  # type: ignore[assignment]
            runner.count_prompt_tokens = lambda port, prompt, timeout_s: 100  # type: ignore[assignment]
            runner.call_completion = lambda port, prompt, max_tokens, temperature, seed, timeout_s: {  # type: ignore[assignment]
                "usage": {"prompt_tokens": 100, "completion_tokens": 3},
                "timings": {"predicted_per_second": 12.5},
                "choices": [{"message": {"content": "READY"}, "finish_reason": "stop"}],
            }
            runner.terminate_server = lambda proc: None  # type: ignore[assignment]
            result = runner.run_stage(stage)
        finally:
            (
                runner.launch_server,
                runner.wait_for_health,
                runner.count_prompt_tokens,
                runner.call_completion,
                runner.terminate_server,
            ) = originals  # type: ignore[assignment]

        self.assertEqual(result["status"], "failed_acceptance")
        self.assertFalse(result["expected_substring_passed"])

    def test_run_stage_records_http_500_as_failed_request(self) -> None:
        stage = {
            "name": "long_context_dsa_probe",
            "prompt": {
                "task_line": "Return the hidden code.",
                "context_length": 4096,
                "kind": "long_context_probe",
                "answer_instruction": runner.SHORT_ANSWER_INSTRUCTION,
                "needle_text": "Recovery code: GLM52-NEEDLE-7F3A.",
                "needle_depth": 0.5,
                "expected_substring": "GLM52-NEEDLE-7F3A",
            },
            "server": {"server_command": [], "port": 1, "context_length": 4096, "log_file": None, "metrics": False},
            "request": {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 1,
                "timeout_s": 60,
                "min_completion_tokens": 1,
                "progress_poll_interval_s": 0,
                "purpose": "coherence_plus_throughput",
            },
        }

        class FakeProc:
            pid = None

        def fake_completion(port, prompt, max_tokens, temperature, seed, timeout_s):
            raise HTTPError(
                url="http://127.0.0.1:1/v1/chat/completions",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"bad peg-native"}}'),
            )

        originals = (
            runner.launch_server,
            runner.wait_for_health,
            runner.count_prompt_tokens,
            runner.call_completion,
            runner.terminate_server,
        )
        try:
            runner.launch_server = lambda command: FakeProc()  # type: ignore[assignment]
            runner.wait_for_health = lambda port: None  # type: ignore[assignment]
            runner.count_prompt_tokens = lambda port, prompt, timeout_s: 100  # type: ignore[assignment]
            runner.call_completion = fake_completion  # type: ignore[assignment]
            runner.terminate_server = lambda proc: None  # type: ignore[assignment]
            result = runner.run_stage(stage)
        finally:
            (
                runner.launch_server,
                runner.wait_for_health,
                runner.count_prompt_tokens,
                runner.call_completion,
                runner.terminate_server,
            ) = originals  # type: ignore[assignment]

        self.assertEqual(result["status"], "failed_request")
        self.assertEqual(result["request_error"]["http_code"], 500)
        self.assertIn("bad peg-native", result["request_error"]["body_preview"])
        self.assertFalse(result["expected_substring_passed"])

    def test_run_stage_streaming_falls_back_to_completion_tokenize_count(self) -> None:
        stage = {
            "name": "long_context_dsa_probe",
            "prompt": {
                "task_line": "Return READY.",
                "context_length": 4096,
                "kind": "long_context_probe",
                "answer_instruction": runner.SHORT_ANSWER_INSTRUCTION,
            },
            "server": {"server_command": [], "port": 1, "context_length": 4096, "log_file": None, "metrics": False},
            "request": {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 1,
                "timeout_s": 60,
                "min_completion_tokens": 4,
                "progress_poll_interval_s": 0,
                "purpose": "coherence_plus_throughput",
                "stream": True,
            },
        }

        class FakeProc:
            pid = None

        originals = (
            runner.launch_server,
            runner.wait_for_health,
            runner.count_prompt_tokens,
            runner.call_completion_streaming,
            runner.terminate_server,
        )

        def fake_count(port: int, prompt: str, timeout_s: int) -> int:
            return 5 if "READY tokenstream" in prompt else 100

        def fake_stream(port, prompt, max_tokens, temperature, seed, timeout_s, progress_callback=None):
            if progress_callback is not None:
                progress_callback({"status": "stream_chunk", "chunk_count": 1})
            return {
                "usage": {},
                "timings": {},
                "streaming": {"enabled": True, "chunk_count": 1},
                "choices": [
                    {
                        "message": {"content": "READY tokenstream tokenstream tokenstream"},
                        "finish_reason": "stop",
                    }
                ],
            }

        try:
            runner.launch_server = lambda command: FakeProc()  # type: ignore[assignment]
            runner.wait_for_health = lambda port: None  # type: ignore[assignment]
            runner.count_prompt_tokens = fake_count  # type: ignore[assignment]
            runner.call_completion_streaming = fake_stream  # type: ignore[assignment]
            runner.terminate_server = lambda proc: None  # type: ignore[assignment]
            result = runner.run_stage(stage)
        finally:
            (
                runner.launch_server,
                runner.wait_for_health,
                runner.count_prompt_tokens,
                runner.call_completion_streaming,
                runner.terminate_server,
            ) = originals  # type: ignore[assignment]

        self.assertEqual(result["completion_token_count"], 5)
        self.assertEqual(result["completion_token_count_source"], "tokenize_completion_text")
        self.assertTrue(result["completion_token_min_passed"])
        self.assertEqual(result["streaming"], {"enabled": True, "chunk_count": 1})
        self.assertEqual(result["stream_progress_samples_tail"], [{"status": "stream_chunk", "chunk_count": 1}])

    def test_run_stage_rejects_short_completion_for_throughput_evidence(self) -> None:
        stage = {
            "name": "long_context_dsa_probe",
            "prompt": {
                "task_line": "Return READY.",
                "context_length": 4096,
                "kind": "long_context_probe",
                "answer_instruction": runner.SHORT_ANSWER_INSTRUCTION,
            },
            "server": {"server_command": [], "port": 1, "context_length": 4096, "log_file": None, "metrics": False},
            "request": {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 1,
                "timeout_s": 60,
                "min_completion_tokens": 8,
                "progress_poll_interval_s": 0,
                "purpose": "coherence_plus_throughput",
            },
        }

        class FakeProc:
            pid = None

        originals = (
            runner.launch_server,
            runner.wait_for_health,
            runner.count_prompt_tokens,
            runner.call_completion,
            runner.terminate_server,
        )
        try:
            runner.launch_server = lambda command: FakeProc()  # type: ignore[assignment]
            runner.wait_for_health = lambda port: None  # type: ignore[assignment]
            runner.count_prompt_tokens = lambda port, prompt, timeout_s: 100  # type: ignore[assignment]
            runner.call_completion = lambda port, prompt, max_tokens, temperature, seed, timeout_s: {  # type: ignore[assignment]
                "usage": {"prompt_tokens": 100, "completion_tokens": 3},
                "timings": {},
                "choices": [{"message": {"content": "READY"}, "finish_reason": "stop"}],
            }
            runner.terminate_server = lambda proc: None  # type: ignore[assignment]
            result = runner.run_stage(stage)
        finally:
            (
                runner.launch_server,
                runner.wait_for_health,
                runner.count_prompt_tokens,
                runner.call_completion,
                runner.terminate_server,
            ) = originals  # type: ignore[assignment]

        self.assertEqual(result["status"], "failed_completion_floor")
        self.assertEqual(result["completion_token_count"], 3)
        self.assertFalse(result["completion_token_min_passed"])

    def test_summarize_server_log_extracts_prompt_and_decode_timings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            log_path.write_text(
                "\n".join(
                    [
                        "I slot print_timing: id 3 | task 0 | prompt processing, n_tokens =   2048, progress = 0.17, t =  76.30 s / 26.84 tokens per second",
                        "I slot print_timing: id 3 | task 0 | n_decoded =    508, tg =   2.53 t/s, tg_3s =   2.53 t/s",
                        "I slot print_timing: id 3 | task 0 | prompt eval time =  645895.03 ms / 11952 tokens (   54.04 ms per token,    18.50 tokens per second)",
                        "I slot print_timing: id 3 | task 0 |        eval time =  202448.52 ms /   512 tokens (  395.41 ms per token,     2.53 tokens per second)",
                    ]
                ),
                encoding="utf-8",
            )

            summary = runner.summarize_server_log(str(log_path))

        self.assertEqual(summary["prompt_eval_tokens"], 11952)
        self.assertEqual(summary["prompt_eval_tps"], 18.5)
        self.assertEqual(summary["decode_tokens"], 512)
        self.assertEqual(summary["decode_tps"], 2.53)
        self.assertEqual(summary["max_decoded_checkpoint"], 508)

    def test_main_writes_dry_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _make_shard_dir(tmp_path)
            binary_dir = tmp_path / "bin"
            binary_dir.mkdir()
            binary = binary_dir / "llama-server"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            output = tmp_path / "plan.json"

            rc = runner.main(
                [
                    "--output",
                    str(output),
                    "--model-dir",
                    str(model_dir),
                    "--binary",
                    str(binary),
                    "--library-path",
                    str(binary_dir),
                ]
            )

            self.assertEqual(rc, 0)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["mode"], "dry-run")
            self.assertEqual(plan["inventory"]["status"], "ready")
            self.assertEqual(plan["stages"][0]["kind"], "inventory")


if __name__ == "__main__":
    unittest.main()
