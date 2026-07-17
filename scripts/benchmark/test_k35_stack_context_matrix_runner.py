import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_stack_context_matrix_runner as k35


class K35StackContextMatrixRunnerTests(unittest.TestCase):
    def test_worker_command_preserves_composed_spec_knobs(self):
        scenario = k35.scenario_by_name("worker_general_cpu_composed_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19123,
            nominal_context=2048,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("--spec-type ngram-mod,draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 2", joined)
        self.assertIn("--spec-draft-threads 16", joined)
        self.assertIn("--spec-draft-device none", joined)
        self.assertIn("--no-mmap", argv)
        self.assertIn("-ctk q8_0 -ctv q8_0", joined)

    def test_frontdoor_command_uses_gpu_no_spec(self):
        scenario = k35.scenario_by_name("frontdoor_gpu_resident_no_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19124,
            nominal_context=8192,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("--device ROCm0", joined)
        self.assertIn("-ngl 99", joined)
        self.assertIn("--spec-type none", joined)
        self.assertIn("--reasoning off", joined)
        self.assertEqual(scenario.enable_thinking, False)

    def test_architect_command_preserves_native_mtp_and_thinking_off(self):
        scenario = k35.scenario_by_name("architect_general_cpu_native_mtp")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19125,
            nominal_context=2048,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("-np 2", joined)
        self.assertIn("--spec-type draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 4", joined)
        self.assertIn("-ctk q4_0 -ctv f16", joined)
        self.assertIn("--mlock", argv)
        self.assertIn("--slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general", joined)
        self.assertNotIn("-md", argv)
        self.assertEqual(scenario.enable_thinking, False)

    def test_architect_context_accounts_for_parallel_slots(self):
        scenario = k35.scenario_by_name("architect_general_cpu_native_mtp")
        self.assertEqual(
            k35.server_context(scenario, nominal_context=8192, max_tokens=512),
            16384,
        )

    def test_ingest_command_uses_default_experts_without_spec(self):
        scenario = k35.scenario_by_name("ingest_long_context_cpu_default_experts")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19126,
            nominal_context=8192,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("--spec-type none", joined)
        self.assertNotIn("--override-kv", argv)
        self.assertNotIn("qwen3next.expert_used_count", joined)
        self.assertIn("-ctk q4_0 -ctv q4_0", joined)
        self.assertIn("--mlock", argv)

    def test_plan_skips_contexts_above_scenario_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(
                [
                    "--only",
                    "worker_general_cpu_composed_spec",
                    "--context",
                    "2048",
                    "--context",
                    "32768",
                    "--max-tokens",
                    "128",
                    "--output-dir",
                    tmp,
                ]
            )
            plan = k35.build_plan(args)
            self.assertEqual([cell["nominal_context"] for cell in plan["cells"]], [2048])

    def test_plan_skips_architect_contexts_above_per_slot_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(
                [
                    "--only",
                    "architect_general_cpu_native_mtp",
                    "--context",
                    "8192",
                    "--context",
                    "14000",
                    "--max-tokens",
                    "128",
                    "--output-dir",
                    tmp,
                ]
            )
            plan = k35.build_plan(args)
            self.assertEqual([cell["nominal_context"] for cell in plan["cells"]], [8192])

    def test_main_dry_run_writes_plan_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = k35.main(
                [
                    "--only",
                    "frontdoor_gpu_resident_no_spec",
                    "--context",
                    "2048",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            plan = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual(len(plan["cells"]), 1)
            self.assertTrue((Path(tmp) / "commands.sh").exists())


if __name__ == "__main__":
    unittest.main()
