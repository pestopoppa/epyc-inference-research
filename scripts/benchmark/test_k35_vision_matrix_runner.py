import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_vision_matrix_runner as k35v


class K35VisionMatrixRunnerTests(unittest.TestCase):
    def test_worker_vision_command_preserves_production_launch_shape(self):
        scenario = k35v.scenario_by_name("worker_vision_cpu_qwen25vl")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19251,
        )
        joined = " ".join(argv)
        self.assertIn("--mmproj", argv)
        self.assertIn("-np 2", joined)
        self.assertIn("-c 8192", joined)
        self.assertIn("-t 24", joined)
        self.assertIn("--device none", joined)
        self.assertNotIn("--override-kv", argv)

    def test_vision_escalation_command_preserves_moe4_override(self):
        scenario = k35v.scenario_by_name("vision_escalation_cpu_qwen3vl30b_moe4")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19252,
        )
        joined = " ".join(argv)
        self.assertIn("-np 1", joined)
        self.assertIn("-c 16384", joined)
        self.assertIn("-t 96", joined)
        self.assertIn("--override-kv qwen3vlmoe.expert_used_count=int:4", joined)

    def test_vision_escalation_image1024_candidate_adds_warned_bounds(self):
        scenario = k35v.scenario_by_name("vision_escalation_cpu_qwen3vl30b_moe4_image1024")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19253,
        )
        joined = " ".join(argv)
        self.assertIn("--image-min-tokens 1024", joined)
        self.assertIn("--image-max-tokens 1024", joined)

    def test_vision_escalation_default_experts_omits_moe4_override(self):
        scenario = k35v.scenario_by_name("vision_escalation_cpu_qwen3vl30b_default_experts")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19254,
        )
        joined = " ".join(argv)
        self.assertNotIn("--override-kv", argv)
        self.assertNotIn("qwen3vlmoe.expert_used_count", joined)

    def test_qwen3vl8b_cpu_candidate_uses_local_artifacts(self):
        scenario = k35v.scenario_by_name("vision_candidate_cpu_qwen3vl8b_q4")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19255,
        )
        joined = " ".join(argv)
        self.assertIn("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf", argv)
        self.assertIn("mmproj-Qwen3VL-8B-Instruct-F16.gguf", joined)
        self.assertIn("--device none", joined)

    def test_qwen3vl8b_mi210_candidate_offloads_and_sets_image_tokens(self):
        scenario = k35v.scenario_by_name("vision_candidate_mi210_qwen3vl8b_q4")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19256,
        )
        joined = " ".join(argv)
        self.assertIn("--device ROCm0", joined)
        self.assertIn("--image-min-tokens 1024", joined)
        self.assertIn("--image-max-tokens 1024", joined)

    def test_qwen3vl8b_mi210_default_image_candidate_uses_default_bounds(self):
        scenario = k35v.scenario_by_name("vision_candidate_mi210_qwen3vl8b_q4_default_image")
        argv = k35v.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19257,
        )
        joined = " ".join(argv)
        self.assertIn("--device ROCm0", joined)
        self.assertNotIn("--image-min-tokens", joined)
        self.assertNotIn("--image-max-tokens", joined)

    def test_score_response_normalizes_expected_terms(self):
        fixture = k35v.fixture_by_id("receipt_doc_number")
        score = k35v.score_response("The document number is CS 00012465.", fixture)
        self.assertTrue(score["pass"])
        self.assertEqual(score["missing_terms"], [])

    def test_image_data_url_has_png_prefix(self):
        fixture = k35v.fixture_by_id("ocr_digit_7500")
        data_url = k35v.image_data_url(fixture.image)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_main_dry_run_writes_plan_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = k35v.main(
                [
                    "--only",
                    "worker_vision_cpu_qwen25vl",
                    "--fixture",
                    "ocr_digit_7500",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            plan = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual(plan["schema"], "epyc.k35_vision_matrix.plan.v1")
            self.assertEqual(len(plan["cells"]), 1)
            self.assertEqual(len(plan["cells"][0]["fixtures"]), 1)
            self.assertTrue((Path(tmp) / "commands.sh").exists())

    def test_default_plan_excludes_candidate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35v.parse_args(["--output-dir", tmp])
            plan = k35v.build_plan(args)
            names = [cell["scenario"] for cell in plan["cells"]]
            self.assertEqual(
                names,
                ["worker_vision_cpu_qwen25vl", "vision_escalation_cpu_qwen3vl30b_moe4"],
            )


if __name__ == "__main__":
    unittest.main()
