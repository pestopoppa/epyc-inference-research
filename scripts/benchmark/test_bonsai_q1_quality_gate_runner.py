#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import bonsai_q1_quality_gate_runner as runner


class _CompletedProcess:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _fake_runner_factory(mapping: dict[str, list[str]]):
    def _runner(cmd, capture_output=True, text=True):  # noqa: ANN001
        if cmd[:3] == ["ps", "-eo", "pid=,args="]:
            matches = mapping.get("ps", [])
            if matches:
                return _CompletedProcess(0, "\n".join(matches))
            return _CompletedProcess(0, "")
        pattern = cmd[-1]
        matches = mapping.get(pattern, [])
        if matches:
            return _CompletedProcess(0, "\n".join(matches))
        return _CompletedProcess(1, "")

    return _runner


class BonsaiQ1QualityGateRunnerTests(unittest.TestCase):
    def test_build_manifest_emits_pinned_experimental_v7_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "Bonsai-27B-Q1_0.gguf"
            model.write_text("placeholder", encoding="utf-8")
            binary = Path(tmpdir) / "llama-cli"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            with (
                mock.patch.object(runner, "EXPERIMENTAL_ROOT", Path(tmpdir)),
                mock.patch.object(runner, "EXPERIMENTAL_BIN_DIR", binary.parent),
                mock.patch.object(runner, "MODEL_PATH", model),
                mock.patch.object(runner, "EXPERIMENTAL_LLAMA_CLI", binary),
                mock.patch.object(runner, "EXPERIMENTAL_LD_LIBRARY_PATH", str(binary.parent)),
            ):
                guards = runner.GuardState(
                    quiet_host_blockers=[],
                    glm_download_active=False,
                    glm_download_blockers=[],
                )

                manifest = runner.build_manifest(guards)

                self.assertEqual(manifest["gate"]["gate_id"], "bonsai_q1_role_claim_gate")
                self.assertTrue(manifest["gate"]["dry_run_only"])
                self.assertEqual(manifest["meta"]["experimental_root"], str(runner.EXPERIMENTAL_ROOT))
                self.assertEqual(manifest["meta"]["experimental_binary"], str(binary))
                self.assertEqual(manifest["gate"]["model_path"], str(model))
                self.assertEqual(len(manifest["gate"]["probes"]), 4)
                self.assertEqual(len(manifest["gate"]["command_templates"]), 8)

                cpu = manifest["gate"]["command_templates"][0]
                mi210 = manifest["gate"]["command_templates"][1]
                self.assertIn(str(runner.EXPERIMENTAL_LLAMA_CLI), cpu["shell"])
                self.assertIn("--device none", cpu["shell"])
                self.assertIn("-ngl 0", cpu["shell"])
                self.assertIn("Return exactly: ok", cpu["shell"])
                self.assertIn("--device ROCm0", mi210["shell"])
                self.assertIn("-ngl 99", mi210["shell"])
                self.assertNotIn("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-cli", cpu["shell"])
                self.assertIn("probe-specific expected output", manifest["gate"]["acceptance_rule"])
                self.assertTrue(any(command["probe_id"] == "strict_json" for command in manifest["gate"]["command_templates"]))

    def test_collect_guard_state_detects_glm_and_quiet_host_blockers(self):
        fake_runner = _fake_runner_factory(
            {
                runner.AUTOPILOT_PATTERN: ["1234 scripts/autopilot/autopilot.py start"],
                "ps": ["5678 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m model.gguf"],
                runner.GLM_PATTERN: ["3862528 hf download unsloth/GLM-5.2-GGUF"],
            }
        )

        guards = runner.collect_guard_state(runner=fake_runner)

        self.assertFalse(guards.quiet_host_ready)
        self.assertTrue(guards.glm_download_active)
        self.assertTrue(any("quiet host guard" in blocker for blocker in guards.quiet_host_blockers))
        self.assertTrue(any("GLM HF writer" in blocker for blocker in guards.glm_download_blockers))

    def test_write_artifacts_emits_manifest_and_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "Bonsai-27B-Q1_0.gguf"
            model.write_text("placeholder", encoding="utf-8")
            binary = Path(tmpdir) / "llama-cli"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            with (
                mock.patch.object(runner, "EXPERIMENTAL_ROOT", Path(tmpdir)),
                mock.patch.object(runner, "EXPERIMENTAL_BIN_DIR", binary.parent),
                mock.patch.object(runner, "MODEL_PATH", model),
                mock.patch.object(runner, "EXPERIMENTAL_LLAMA_CLI", binary),
                mock.patch.object(runner, "EXPERIMENTAL_LD_LIBRARY_PATH", str(binary.parent)),
            ):
                guards = runner.GuardState(
                    quiet_host_blockers=[],
                    glm_download_active=False,
                    glm_download_blockers=[],
                )
                manifest = runner.build_manifest(guards)
                output_dir = Path(tmpdir) / "out"
                runner.write_artifacts(output_dir, manifest)

                self.assertTrue((output_dir / "manifest.json").exists())
                self.assertTrue((output_dir / "gate.json").exists())
                self.assertTrue((output_dir / "commands.sh").exists())
                disk_manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(disk_manifest["gate"]["gate_id"], "bonsai_q1_role_claim_gate")
                commands = (output_dir / "commands.sh").read_text()
                self.assertIn(str(runner.EXPERIMENTAL_LLAMA_CLI), commands)
                self.assertNotIn("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-cli", commands)


if __name__ == "__main__":
    unittest.main()
