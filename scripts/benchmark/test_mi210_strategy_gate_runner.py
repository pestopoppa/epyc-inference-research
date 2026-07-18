#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).parent))

import mi210_strategy_gate_runner as runner


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


class Mi210StrategyGateRunnerTests(unittest.TestCase):
    def test_build_manifest_emits_three_gate_plans_with_pinned_experimental_ld_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.gguf"
            model.write_text("placeholder", encoding="utf-8")
            skew_evidence = Path(tmpdir) / "skew.json"
            skew_evidence.write_text("{}", encoding="utf-8")

            args = Namespace(
                output_dir=Path(tmpdir) / "plans",
                model=model,
                skew_evidence=skew_evidence,
                force_no_skew_evidence=False,
                allow_glm_download=False,
                execute=False,
            )
            guards = runner.GuardState(
                quiet_host_blockers=[],
                glm_download_active=False,
                glm_download_blockers=[],
            )

            manifest = runner.build_manifest(args, guards)

            self.assertEqual([gate["gate_id"] for gate in manifest["gates"]], [
                "frontdoor_residency_p_gpu1",
                "hybrid_moe_offload_cpu_experts",
                "ngram_mtp_quality_monitoring_stub",
            ])
            self.assertEqual(
                manifest["meta"]["experimental_ld_library_path"],
                str(runner.EXPERIMENTAL_BIN_DIR),
            )

            frontdoor = manifest["gates"][0]
            self.assertTrue(frontdoor["dry_run_only"])
            self.assertIn("stack stopped", frontdoor["evidence_fields"])
            self.assertIn("residency VRAM", frontdoor["evidence_fields"])

            hybrid = manifest["gates"][1]
            self.assertEqual(hybrid["model_path"], str(model))
            self.assertTrue(hybrid["prerequisite_evidence"]["expert_routing_skew_profile"]["present"])
            self.assertEqual(
                hybrid["prerequisite_evidence"]["expert_routing_skew_profile"]["runner"],
                str(runner.SCRIPT_DIR / "expert_routing_skew_profile.sh"),
            )
            hybrid_shells = [command["shell"] for command in hybrid["command_templates"]]
            self.assertTrue(any(f"LD_LIBRARY_PATH={runner.EXPERIMENTAL_LD_LIBRARY_PATH}" in shell for shell in hybrid_shells))
            self.assertTrue(any("env -i" in shell for shell in hybrid_shells))
            self.assertTrue(any("-ot exps=CPU" in shell for shell in hybrid_shells))
            self.assertTrue(any("--device none" in shell for shell in hybrid_shells))

            ngram = manifest["gates"][2]
            self.assertTrue(ngram["dry_run_only"])
            self.assertFalse(ngram["command_templates"])

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

    def test_probe_pattern_ignores_current_process_chain(self):
        self_pid = runner.os.getpid()
        with mock.patch.object(runner, "_current_pid_chain", return_value={self_pid, 2222}):
            fake_runner = _fake_runner_factory(
                {
                    runner.AUTOPILOT_PATTERN: [
                        f"{self_pid} /bin/bash -c pgrep -af {runner.AUTOPILOT_PATTERN!r}",
                        f"2222 timeout wrapper {runner.AUTOPILOT_PATTERN}",
                        "3333 python scripts/autopilot/autopilot.py start",
                        f"4444 pgrep -af {runner.AUTOPILOT_PATTERN}",
                    ],
                }
            )

            matches = runner._probe_pattern(runner.AUTOPILOT_PATTERN, runner=fake_runner)

        self.assertEqual(matches, ["3333 python scripts/autopilot/autopilot.py start"])

    def test_execute_refuses_glm_writer_without_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.gguf"
            model.write_text("placeholder", encoding="utf-8")
            skew_evidence = Path(tmpdir) / "skew.json"
            skew_evidence.write_text("{}", encoding="utf-8")

            fake_guards = runner.GuardState(
                quiet_host_blockers=[],
                glm_download_active=True,
                glm_download_blockers=["GLM HF writer is active"],
            )
            args = Namespace(
                output_dir=Path(tmpdir) / "plans",
                model=model,
                skew_evidence=skew_evidence,
                force_no_skew_evidence=False,
                allow_glm_download=False,
                execute=True,
            )

            manifest = runner.build_manifest(args, fake_guards)
            blockers = runner._execution_blockers(manifest, args)

            self.assertIn("GLM HF writer is active", blockers)

    def test_execute_guard_allows_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.gguf"
            model.write_text("placeholder", encoding="utf-8")
            skew_evidence = Path(tmpdir) / "skew.json"
            skew_evidence.write_text("{}", encoding="utf-8")

            fake_guards = runner.GuardState(
                quiet_host_blockers=[],
                glm_download_active=True,
                glm_download_blockers=["GLM HF writer is active"],
            )
            args = Namespace(
                output_dir=Path(tmpdir) / "plans",
                model=model,
                skew_evidence=skew_evidence,
                force_no_skew_evidence=False,
                allow_glm_download=True,
                execute=True,
            )

            manifest = runner.build_manifest(args, fake_guards)
            blockers = runner._execution_blockers(manifest, args)

            self.assertNotIn("GLM HF writer is active", blockers)

    def test_write_artifacts_emits_manifest_and_per_gate_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.gguf"
            model.write_text("placeholder", encoding="utf-8")
            skew_evidence = Path(tmpdir) / "skew.json"
            skew_evidence.write_text("{}", encoding="utf-8")
            args = Namespace(
                output_dir=Path(tmpdir) / "plans",
                model=model,
                skew_evidence=skew_evidence,
                force_no_skew_evidence=False,
                allow_glm_download=False,
                execute=False,
            )
            guards = runner.GuardState(
                quiet_host_blockers=[],
                glm_download_active=False,
                glm_download_blockers=[],
            )

            manifest = runner.build_manifest(args, guards)
            runner.write_artifacts(args.output_dir, manifest)

            self.assertTrue((args.output_dir / "manifest.json").exists())
            self.assertTrue((args.output_dir / "frontdoor_residency_p_gpu1.json").exists())
            self.assertTrue((args.output_dir / "hybrid_moe_offload_cpu_experts.json").exists())
            self.assertTrue((args.output_dir / "ngram_mtp_quality_monitoring_stub.json").exists())
            manifest_disk = json.loads((args.output_dir / "manifest.json").read_text())
            self.assertEqual(manifest_disk["gates"][1]["gate_id"], "hybrid_moe_offload_cpu_experts")


if __name__ == "__main__":
    unittest.main()
