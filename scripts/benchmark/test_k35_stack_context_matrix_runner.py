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
