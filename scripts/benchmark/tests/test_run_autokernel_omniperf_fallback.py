#!/usr/bin/env python3
"""Zero-device tests for the AutoKernel Omniperf fallback runner."""
from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts.benchmark import run_autokernel_omniperf_fallback as R


class OmniperfFallbackTest(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "quant_type": "iq2_xxs", "op_m": 16, "op_n": 1, "op_k": 256,
            "suite_seed": 4711, "repetitions": 5, "backend": "ROCm0",
            "workload_name": "iq2xxs_fallback",
            "omniperf_python": "/venv/bin/python",
            "omniperf": "/tools/omniperf",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_shape_filter_avoids_bracketed_layout_boundary(self):
        self.assertEqual(
            R.op_pattern(self.args()),
            r"^type_a=iq2_xxs,type_b=f32,m=16,n=1,k=256.*$")
        with self.assertRaisesRegex(RuntimeError, "letters, digits"):
            R.op_pattern(self.args(quant_type="iq2;bad"))

    def test_commands_pin_seed_repetitions_profiler_blocks_and_no_roof(self):
        args = self.args()
        backend = R.backend_command(Path("/bin/test"), args)
        self.assertEqual(backend[backend.index("--suite-seed") + 1], "4711")
        self.assertEqual(backend[backend.index("--repeat-suite") + 1], "5")
        profile = R.omniperf_command(Path("/bin/test"), Path("/evidence"), args)
        self.assertIn(R._LOCALE_COMPAT, profile)
        self.assertIn("--no-roof", profile)
        self.assertEqual(profile[profile.index("-b") + 1:profile.index("-b") + 3],
                         ("SQ", "TCC"))

    def test_preflight_requires_supported_complete_repetitions(self):
        header = ["backend_name", "op_name", "supported", "hard_failure",
                  "error_message"]
        out = tempfile.SpooledTemporaryFile(mode="w+")
        writer = csv.DictWriter(out, fieldnames=header)
        writer.writeheader()
        for _ in range(10):
            writer.writerow({"backend_name": "ROCm0", "op_name": "MUL_MAT",
                             "supported": "1", "hard_failure": "0",
                             "error_message": ""})
        out.seek(0)
        self.assertEqual(R.validate_preflight(out.read(), repetitions=5)["cases_per_repetition"], 2)
        out.close()
        with self.assertRaisesRegex(RuntimeError, "divisible"):
            R.validate_preflight(
                "op_name,supported,hard_failure,error_message\nMUL_MAT,1,0,\n",
                repetitions=5)

    def test_binary_capability_gate_requires_seed_and_repeat(self):
        with mock.patch.object(
                R, "run_owned", return_value=(1, "", "Usage: test --suite-seed", 0.0)):
            with self.assertRaisesRegex(RuntimeError, "--repeat-suite"):
                R.validate_binary_capabilities(Path("/bin/test"), env={})
        with mock.patch.object(
                R, "run_owned",
                return_value=(1, "", "Usage: test --suite-seed --repeat-suite", 0.0)):
            result = R.validate_binary_capabilities(Path("/bin/test"), env={})
        self.assertEqual(result["required_flags"], ["--suite-seed", "--repeat-suite"])

    def test_profile_summary_requires_counter_and_target_kernel_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pmc_perf.csv"
            fields = [
                "Dispatch_ID", "Kernel_Name", "Start_Timestamp", "End_Timestamp",
                "Grid_Size", "Workgroup_Size", *R._REQUIRED_COUNTERS,
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for dispatch, name in enumerate(("quantize_q8_1.kd", "mul_mat_vec_q<16>.kd")):
                    row = {field: "1" for field in fields}
                    row.update({"Dispatch_ID": str(dispatch), "Kernel_Name": name,
                                "Start_Timestamp": "10", "End_Timestamp": "20"})
                    writer.writerow(row)
            summary = R.summarize_profile(path, quant_type="iq2_xxs")
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(
                {row["family"] for row in summary["families"]},
                {"mul_mat_vec_q", "quantize_q8_1"})

    def test_parser_defaults_to_the_proven_fallback(self):
        args = R.parser().parse_args([
            "--source-root", "/source", "--binary", "/bin/test",
            "--output-dir", "/evidence",
        ])
        self.assertEqual(args.quant_type, "iq2_xxs")
        self.assertTrue(args.omniperf_python.endswith("/bin/python"))
        self.assertEqual(args.repetitions, 5)

    def test_runner_writes_prospective_belief_measurements_only(self):
        text = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn('"belief_measurements": belief_measurements', text)
        self.assertIn('"reps_basis": "scored:seeded repeated backend-op suites"', text)


if __name__ == "__main__":
    unittest.main()
