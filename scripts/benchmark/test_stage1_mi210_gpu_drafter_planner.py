#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import stage1_mi210_gpu_drafter_planner as planner


class _CompletedProcess:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _fake_runner_factory(mapping: dict[str, list[str]]):
    def _runner(cmd, capture_output=True, text=True):  # noqa: ANN001
        if cmd[:3] == ["ps", "-eo", "pid=,args="]:
            matches = mapping.get("ps", [])
            return _CompletedProcess(0, "\n".join(matches))
        pattern = cmd[-1]
        matches = mapping.get(pattern, [])
        if matches:
            return _CompletedProcess(0, "\n".join(matches))
        return _CompletedProcess(1, "")

    return _runner


class Stage1Mi210GpuDrafterPlannerTests(unittest.TestCase):
    def test_validate_experimental_server_accepts_pinned_experimental_path(self):
        resolved = planner.validate_experimental_server(planner.EXPERIMENTAL_SERVER)
        self.assertEqual(resolved, planner.EXPERIMENTAL_SERVER.resolve())

    def test_collect_guard_state_detects_autopilot_and_llama_server(self):
        fake_runner = _fake_runner_factory(
            {
                planner.AUTOPILOT_PATTERN: ["1234 python scripts/autopilot/autopilot.py start"],
                "ps": ["5678 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m target.gguf"],
            }
        )

        guards = planner.collect_guard_state(runner=fake_runner)

        self.assertFalse(guards.quiet_host_ready)
        self.assertEqual(len(guards.quiet_host_blockers), 2)
        self.assertTrue(any("AutoPilot pattern" in blocker for blocker in guards.quiet_host_blockers))
        self.assertTrue(any("process basename 'llama-server'" in blocker for blocker in guards.quiet_host_blockers))

    def test_write_artifacts_emits_manifest_commands_and_gate_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_model = Path(tmpdir) / "target.gguf"
            draft_model = Path(tmpdir) / "draft.gguf"
            target_model.write_text("target", encoding="utf-8")
            draft_model.write_text("draft", encoding="utf-8")

            args = Namespace(
                target_model=target_model,
                draft_model=draft_model,
                baseline_port=19187,
                stage1_port=19188,
                output_dir=Path(tmpdir) / "plan",
            )
            guards = planner.GuardState(quiet_host_blockers=[])

            manifest = planner.build_manifest(args, guards, planner.EXPERIMENTAL_SERVER.resolve())
            gate_summary = planner.build_gate_summary(manifest)
            planner.write_artifacts(args.output_dir, manifest, gate_summary)

            manifest_disk = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
            gate_summary_disk = json.loads((args.output_dir / "gate_summary.json").read_text(encoding="utf-8"))
            commands = (args.output_dir / "commands.sh").read_text(encoding="utf-8")

            self.assertEqual(manifest_disk["mode"], "dry_run")
            self.assertEqual(
                manifest_disk["prerequisites"]["n5_frontdoor_retest"]["command"],
                planner.N5_PREREQUISITE_COMMAND,
            )
            self.assertEqual(
                manifest_disk["gate"]["pass_metadata"]["speedup_gte"],
                planner.PASS_SPEEDUP_THRESHOLD,
            )
            stage1_shell = manifest_disk["gate"]["server_command_templates"][1]["shell"]
            self.assertIn(str(planner.EXPERIMENTAL_SERVER.resolve()), stage1_shell)
            self.assertIn("--spec-draft-device ROCm0", stage1_shell)
            self.assertIn("--spec-draft-ngl 99", stage1_shell)
            self.assertIn("--spec-type draft-tree", stage1_shell)
            baseline_shell = manifest_disk["gate"]["server_command_templates"][0]["shell"]
            self.assertIn("--device none", baseline_shell)
            self.assertIn("--spec-type none", baseline_shell)
            self.assertIn(planner.N5_PREREQUISITE_COMMAND, commands)
            self.assertTrue((args.output_dir / "manifest.json").exists())
            self.assertTrue((args.output_dir / "commands.sh").exists())
            self.assertTrue((args.output_dir / "gate_summary.json").exists())
            self.assertEqual(gate_summary_disk["pass_speedup_gte"], planner.PASS_SPEEDUP_THRESHOLD)
            self.assertEqual(gate_summary_disk["next_window"], "after GLM completes and the host is quiet")

    def test_validate_experimental_server_refuses_production_path(self):
        with self.assertRaisesRegex(ValueError, "production v6"):
            planner.validate_experimental_server(planner.PRODUCTION_SERVER)


if __name__ == "__main__":
    unittest.main()
