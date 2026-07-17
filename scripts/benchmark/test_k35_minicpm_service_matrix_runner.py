import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_minicpm_service_matrix_runner as runner


class K35MiniCPMServiceMatrixRunnerTests(unittest.TestCase):
    def test_dry_run_writes_plan_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = runner.main(
                [
                    "--context",
                    "2048",
                    "--fixture",
                    "chart_tanzania",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            plan = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual(plan["schema"], "epyc.k35_minicpm_service_matrix.plan.v1")
            self.assertEqual(plan["frontdoor"]["scenario"], "frontdoor_gpu_resident_no_spec")
            self.assertEqual(plan["minicpm"]["scenario"], "vision_candidate_mi210_minicpm_o45_q4")
            self.assertEqual(plan["minicpm"]["fixtures"], ["chart_tanzania"])
            self.assertTrue((Path(tmp) / "commands.sh").exists())

    def test_plan_uses_realistic_optimized_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--output-dir", tmp])
            plan = runner.build_plan(args)
            frontdoor = " ".join(plan["frontdoor"]["server_argv"])
            minicpm = " ".join(plan["minicpm"]["server_argv"])
            self.assertIn("--device ROCm0", frontdoor)
            self.assertIn("-ctk q8_0", frontdoor)
            self.assertIn("-ctv q8_0", frontdoor)
            self.assertIn("--spec-type none", frontdoor)
            self.assertIn("MiniCPM-o-4_5-Q4_K_M.gguf", minicpm)
            self.assertIn("vision/MiniCPM-o-4_5-vision-F16.gguf", minicpm)
            self.assertIn("--device ROCm0", minicpm)
            self.assertIn("--reasoning off", minicpm)

    def test_compact_summary_reports_service_tax_buckets(self):
        results = [
            {
                "arm": "frontdoor_alone_control",
                "frontdoor_results": [{"decode_tps": 100.0}, {"decode_tps": 90.0}],
            },
            {
                "arm": "frontdoor_minicpm_pair",
                "idle_results": [{"decode_tps": 101.0}],
                "active_results": [
                    {
                        "frontdoor": {"decode_tps": 80.0, "passed_min_completion": True},
                        "vision": {"decode_tps": 110.0, "score": {"pass": True}},
                    }
                ],
            },
        ]
        summary = runner.compact_summary(results)
        self.assertEqual(summary["frontdoor_alone_decode_tps"]["n"], 2)
        self.assertAlmostEqual(summary["frontdoor_alone_decode_tps"]["mean"], 95.0)
        self.assertEqual(summary["frontdoor_active_overlap_decode_tps"]["mean"], 80.0)
        self.assertEqual(summary["minicpm_active_overlap_decode_tps"]["mean"], 110.0)
        self.assertTrue(summary["active_overlap_passed"])


if __name__ == "__main__":
    unittest.main()
