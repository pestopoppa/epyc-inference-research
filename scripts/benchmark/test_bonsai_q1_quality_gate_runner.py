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
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


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
    def _model_spec(self, key: str, model_path: Path) -> runner.ModelSpec:
        base = runner.MODEL_SPECS[key]
        return runner.ModelSpec(
            key=base.key,
            gate_id=base.gate_id,
            title=base.title,
            model_path=model_path,
            output_subdir=base.output_subdir,
            arm_prefix=base.arm_prefix,
            role_claim_label=base.role_claim_label,
        )

    def test_build_manifest_emits_pinned_experimental_v7_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "Bonsai-27B-Q1_0.gguf"
            model.write_text("placeholder", encoding="utf-8")
            binary = Path(tmpdir) / "llama-cli"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            with (
                mock.patch.object(runner, "EXPERIMENTAL_ROOT", Path(tmpdir)),
                mock.patch.object(runner, "EXPERIMENTAL_BIN_DIR", binary.parent),
                mock.patch.dict(runner.MODEL_SPECS, {"bonsai_q1": self._model_spec("bonsai_q1", model)}),
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
                self.assertIn("-no-cnv", cpu["argv"])
                self.assertIn("--reasoning", cpu["argv"])
                self.assertIn("--reasoning-budget", cpu["argv"])
                self.assertIn("--no-show-timings", cpu["argv"])
                self.assertIn("-ngl 0", cpu["shell"])
                self.assertIn("Return exactly: ok", cpu["shell"])
                self.assertIn("--device ROCm0", mi210["shell"])
                self.assertIn("-ngl 99", mi210["shell"])
                self.assertNotIn("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-cli", cpu["shell"])
                self.assertIn("probe-specific expected output", manifest["gate"]["acceptance_rule"])
                self.assertTrue(any(command["probe_id"] == "strict_json" for command in manifest["gate"]["command_templates"]))

    def test_build_manifest_can_target_ternary_q2_g64(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "Ternary-Bonsai-27B-Q2_g64.gguf"
            model.write_text("placeholder", encoding="utf-8")
            binary = Path(tmpdir) / "llama-cli"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            with (
                mock.patch.object(runner, "EXPERIMENTAL_ROOT", Path(tmpdir)),
                mock.patch.object(runner, "EXPERIMENTAL_BIN_DIR", binary.parent),
                mock.patch.dict(runner.MODEL_SPECS, {"ternary_q2_g64": self._model_spec("ternary_q2_g64", model)}),
                mock.patch.object(runner, "EXPERIMENTAL_LLAMA_CLI", binary),
                mock.patch.object(runner, "EXPERIMENTAL_LD_LIBRARY_PATH", str(binary.parent)),
            ):
                guards = runner.GuardState([], False, [])
                manifest = runner.build_manifest(guards, model_key="ternary_q2_g64")

        self.assertEqual(manifest["gate"]["gate_id"], "ternary_q2_g64_quality_gate")
        self.assertEqual(manifest["meta"]["model"], "ternary_q2_g64")
        self.assertEqual(manifest["gate"]["model_path"], str(model))
        self.assertTrue(all(command["arm"].startswith("ternary_q2_g64_") for command in manifest["gate"]["command_templates"]))

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
                mock.patch.dict(runner.MODEL_SPECS, {"bonsai_q1": self._model_spec("bonsai_q1", model)}),
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

    def test_evaluate_probe_exact_rules(self):
        self.assertTrue(runner.evaluate_probe("exact_ok", "ok\n")["passed"])
        self.assertFalse(runner.evaluate_probe("exact_ok", "OK\n")["passed"])
        self.assertTrue(runner.evaluate_probe("strict_json", '{"status":"ok","model":"bonsai"}')["passed"])
        self.assertFalse(runner.evaluate_probe("strict_json", '```json\n{"status":"ok","model":"bonsai"}\n```')["passed"])
        self.assertTrue(runner.evaluate_probe("simple_math", "95\n")["passed"])
        self.assertTrue(runner.evaluate_probe("short_instruction", "held out tests prevent benchmark leakage")["passed"])
        self.assertFalse(runner.evaluate_probe("short_instruction", "Held out tests prevent benchmark leakage")["passed"])

    def test_evaluate_probe_scores_generated_segment_from_llama_cli_stdout(self):
        stdout = """
Loading model...

> Return exactly: ok
ok


Exiting...
"""
        result = runner.evaluate_probe("exact_ok", stdout, "Return exactly: ok")
        self.assertTrue(result["passed"])
        self.assertEqual(result["generated_text"], "ok")

    def test_run_execute_writes_summary_with_mocked_subprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "Bonsai-27B-Q1_0.gguf"
            model.write_text("placeholder", encoding="utf-8")
            binary = Path(tmpdir) / "llama-cli"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            def fake_run(argv, capture_output=True, text=True, timeout=300, check=False):  # noqa: ANN001
                return _CompletedProcess(0, "ok\n", "")

            with (
                mock.patch.object(runner, "EXPERIMENTAL_ROOT", Path(tmpdir)),
                mock.patch.object(runner, "EXPERIMENTAL_BIN_DIR", binary.parent),
                mock.patch.dict(runner.MODEL_SPECS, {"bonsai_q1": self._model_spec("bonsai_q1", model)}),
                mock.patch.object(runner, "EXPERIMENTAL_LLAMA_CLI", binary),
                mock.patch.object(runner, "EXPERIMENTAL_LD_LIBRARY_PATH", str(binary.parent)),
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run),
            ):
                guards = runner.GuardState(
                    quiet_host_blockers=[],
                    glm_download_active=False,
                    glm_download_blockers=[],
                )
                manifest = runner.build_manifest(guards, execute=True)
                output_dir = Path(tmpdir) / "out"
                runner.write_artifacts(output_dir, manifest)
                summary = runner.run_execute(output_dir, manifest, ["bonsai_q1_cpu_exact_ok"], timeout_s=30)

                self.assertEqual(summary["status"], "pass")
                self.assertEqual(summary["passed"], 1)
                self.assertTrue((output_dir / "arms" / "bonsai_q1_cpu_exact_ok" / "stdout.txt").exists())
                self.assertTrue((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
