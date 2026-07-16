#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
