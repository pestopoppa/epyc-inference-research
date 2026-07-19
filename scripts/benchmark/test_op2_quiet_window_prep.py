#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import op2_quiet_window_prep as op2


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # noqa: ANN001 - subprocess-compatible test double
        self.calls.append(list(argv))
        stdout = ""
        if argv[:2] == ["git", "branch"]:
            stdout = "main\n"
        elif argv[:2] == ["git", "rev-parse"] and "HEAD" in argv:
            stdout = "abc123\n"
        elif argv[:2] == ["git", "rev-parse"] and "@{u}" in argv:
            stdout = "origin/main\n"
        elif argv[:2] == ["git", "status"]:
            stdout = ""
        elif argv[:2] == ["ps", "-eo"]:
            stdout = "123 Sun Jul 19 00:00:00 2026 bash /usr/bin/bash\n"
        elif argv == ["uptime"]:
            stdout = " 00:00:00 up 1 day\n"
        elif argv == ["free", "-h"]:
            stdout = "Mem: 1T\n"
        elif argv == ["uname", "-a"]:
            stdout = "Linux test\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")


class OP2QuietWindowPrepTests(unittest.TestCase):
    def test_stage_plan_keeps_only_remaining_op2_payload(self) -> None:
        plan = op2.stage_plan()

        self.assertEqual(
            [row["stage"] for row in plan["remaining_payload"]],
            ["live_v6_iqk_role_garbage_verification", "clean_canonical_cpu_decode_bench"],
        )
        self.assertIn("skipped_not_staged", {row["status"] for row in plan["skipped_or_closed"]})
        self.assertIn("closed_no_go", {row["status"] for row in plan["skipped_or_closed"]})

    def test_process_parser_ignores_earlyoom_policy_arguments(self) -> None:
        text = (
            "1849379 earlyoom /usr/local/bin/earlyoom --ignore ^(llama-server|sd-server)$ "
            "--prefer ^llama-bench$\n"
            "42 llama-server /mnt/raid0/llm/llama.cpp/build/bin/llama-server --port 8080\n"
        )

        matches = op2.parse_matching_processes(text)

        self.assertEqual([match["comm"] for match in matches], ["llama-server"])

    def test_write_bundle_is_no_inference_and_records_measurement_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            measurement = tmp_path / "MEASUREMENT.md"
            measurement.write_text("### P-GPU-1 — GPU canonical (DEFERRED — hardware not acquired)\n")
            args = op2.parse_args(
                [
                    "--run-id",
                    "op2-test",
                    "--output-dir",
                    str(tmp_path / "bundle"),
                    "--measurement-path",
                    str(measurement),
                    "--root-repo",
                    str(tmp_path),
                    "--research-repo",
                    str(tmp_path),
                    "--orchestrator-repo",
                    str(tmp_path),
                    "--production-llama-repo",
                    str(tmp_path),
                    "--experimental-llama-repo",
                    str(tmp_path),
                ]
            )
            fake = FakeRunner()

            manifest = op2.write_bundle(args, runner=fake)

            self.assertEqual(manifest["status"], "prepared_no_inference")
            self.assertTrue(manifest["measurement"]["p_gpu_1_deferred"])
            self.assertIn("pre-MI210 defer reason", manifest["measurement"]["p_gpu_1_line_note"])
            self.assertIn("production-named MI210 GPU claims only", manifest["measurement"]["p_gpu_1_certification_note"])
            self.assertFalse(manifest["autopilot_restart_authorized"])
            self.assertFalse(manifest["production_v6_touch_authorized"])
            called = [" ".join(call) for call in fake.calls]
            forbidden = ("llama-server", "llama-bench", "orchestrator_stack.py", "rocprof", "perf")
            self.assertFalse(any(any(word in call for word in forbidden) for call in called))

            bundle = tmp_path / "bundle"
            self.assertTrue((bundle / "manifest.json").exists())
            self.assertTrue((bundle / "stage_plan.json").exists())
            commands = (bundle / "operator_next_commands.sh").read_text()
            self.assertIn("OP2_RUN_ID:=op2-canonical-bench-window-$(date -u +%Y%m%dT%H%M%SZ)", commands)
            self.assertIn("/mnt/raid0/llm/epyc-inference-research/data/op2_canonical_bench_window", commands)
            self.assertNotIn(f'OP2_RUN_ROOT:-{bundle.resolve()}', commands)
            self.assertIn("bench_canonical.sh", commands)
            self.assertIn("python3 scripts/server/preflight_gate.py", commands)
            self.assertIn("--affinity-live-only", commands)
            self.assertIn("--server-health-only", commands)
            self.assertIn("--contention-observation-only", commands)
            self.assertIn("--roles frontdoor worker_general architect_general ingest_long_context worker_vision vision_escalation", commands)
            self.assertIn("python3 scripts/server/orchestrator_stack.py status", commands)
            self.assertNotIn("python scripts/server/", commands)
            self.assertIn("role_smoke_ports.tsv", commands)
            self.assertIn("/v1/chat/completions", commands)
            self.assertIn("role_smoke_aggregate.json", commands)
            self.assertIn("P-GPU-1 caveat", commands)
            self.assertIn("process_blockers.json", commands)
            self.assertIn("raise SystemExit(74)", commands)
            self.assertIn("production-named MI210 GPU claims only", commands)
            self.assertNotIn("README.role_smokes.md", commands)
            summary = (bundle / "summary.md").read_text()
            self.assertIn("Raw P-GPU-1 MEASUREMENT line", summary)
            self.assertIn("Raw-line caveat", summary)
            self.assertIn("pre-MI210 defer reason", summary)
            self.assertIn("P-GPU-1 certification caveat", summary)
            self.assertIn("production-named MI210 GPU claims only", summary)

            loaded = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(loaded["schema"], op2.SCHEMA)
            self.assertEqual(loaded["operator_execution"]["mode"], "dynamic_timestamped")

    def test_static_execution_output_dir_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            measurement = tmp_path / "MEASUREMENT.md"
            measurement.write_text("### P-GPU-1 — GPU canonical (DEFERRED — hardware not acquired)\n")
            execution_dir = tmp_path / "execution"
            args = op2.parse_args(
                [
                    "--run-id",
                    "op2-test",
                    "--output-dir",
                    str(tmp_path / "bundle"),
                    "--execution-output-dir",
                    str(execution_dir),
                    "--measurement-path",
                    str(measurement),
                    "--root-repo",
                    str(tmp_path),
                    "--research-repo",
                    str(tmp_path),
                    "--orchestrator-repo",
                    str(tmp_path),
                    "--production-llama-repo",
                    str(tmp_path),
                    "--experimental-llama-repo",
                    str(tmp_path),
                ]
            )

            manifest = op2.write_bundle(args, runner=FakeRunner())

            commands = (tmp_path / "bundle" / "operator_next_commands.sh").read_text()
            self.assertIn(f'OP2_RUN_ROOT:-{execution_dir.resolve()}', commands)
            self.assertEqual(manifest["operator_execution"]["mode"], "static")


if __name__ == "__main__":
    unittest.main()
