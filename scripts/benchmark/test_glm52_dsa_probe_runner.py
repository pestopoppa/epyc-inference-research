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


class TestGlm52DsaProbeRunner(unittest.TestCase):
    def test_collect_inventory_blocks_incomplete_download_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_shard_dir(Path(tmp), include_blocker=True)

            inventory = runner.collect_inventory(model_dir)

            self.assertEqual(inventory["status"], "blocked")
            self.assertEqual(inventory["non_cache_shard_count"], 6)
            self.assertTrue(any(item["path"].endswith(".incomplete") for item in inventory["blocker_files"]))
            self.assertTrue(any("download.partial.incomplete" in reason for reason in inventory["refusal_reasons"]))

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
            self.assertEqual(short_stage["request"]["endpoint"], "/v1/chat/completions")

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
