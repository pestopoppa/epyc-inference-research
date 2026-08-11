#!/usr/bin/env python3
"""Zero-device tests for the AutoKernel G15 profiler runner."""
from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import run_autokernel_g15_profile as R


class G15ProfileTest(unittest.TestCase):
    def test_parallel_values_are_positive_and_unique(self):
        self.assertEqual(R.positive_ints("64,128"), (64, 128))
        with self.assertRaisesRegex(Exception, "positive"):
            R.positive_ints("64,0")
        with self.assertRaisesRegex(Exception, "unique"):
            R.positive_ints("64,64")

    def test_bench_command_pins_the_g15_regime(self):
        command = R.bench_command(
            Path("/bin/bench"), Path("/model.gguf"), parallel=128,
            prompt_tokens=128, generation_tokens=128, context=32768,
            batch=2048, ubatch=512)
        self.assertEqual(command[command.index("-npl") + 1], "128")
        self.assertEqual(command[command.index("-fa") + 1], "off")
        self.assertEqual(command[-2:], ("--output-format", "jsonl"))

    def test_bench_result_binds_the_exact_cell(self):
        row = {
            "n_gpu_layers": 99, "flash_attn": 0,
            "pl": 128, "pp": 128, "tg": 128,
            "t_tg": 1.25, "speed_tg": 13107.2,
        }
        self.assertEqual(
            R.parse_bench_result(
                json.dumps(row), "", parallel=128,
                prompt_tokens=128, generation_tokens=128), row)
        row["flash_attn"] = 1
        with self.assertRaisesRegex(RuntimeError, "flash-attention"):
            R.parse_bench_result(
                json.dumps(row), "", parallel=128,
                prompt_tokens=128, generation_tokens=128)

    def test_kernel_taxonomy_prioritizes_matrix_before_elementwise_mul(self):
        self.assertEqual(R.kernel_family("mul_mat_q_q8_0"), "matrix")
        self.assertEqual(R.kernel_family("Cijk_Alik_Bljk_HSS.kd"), "matrix")
        self.assertEqual(R.kernel_family("rms_norm_f32"), "norm")
        self.assertEqual(R.kernel_family("silu_f32"), "activation")
        self.assertEqual(R.kernel_family("mul_f32"), "elementwise")
        self.assertEqual(R.kernel_family("cpy_f32_f16"), "copy_convert")
        self.assertEqual(R.kernel_family("k_get_rows_float"), "gather_scatter")
        self.assertEqual(R.kernel_family("dequantize_block_q8_0"), "quantization")
        self.assertEqual(R.kernel_family("rope_multi<true, false, float>"), "position")
        self.assertEqual(R.kernel_family("unary_op_kernel<op_softplus>"), "activation")

    def test_timestamp_summary_emits_exact_target_share_and_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timestamps.csv"
            fields = ["Index", "KernelName", "gpu-id", "BeginNs", "EndNs"]
            rows = (
                ("mul_mat_q", 0, 50),
                ("cpy_f32_f16", 50, 60),
                ("rms_norm_f32", 60, 70),
                ("silu_f32", 70, 85),
                ("mul_f32", 85, 100),
            )
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, (kernel, start, end) in enumerate(rows):
                    writer.writerow({
                        "Index": index, "KernelName": kernel, "gpu-id": 0,
                        "BeginNs": start, "EndNs": end,
                    })
            result = R.summarize_timestamps(path)
        self.assertAlmostEqual(result["elementwise_norm_target_share"], 0.40)
        self.assertAlmostEqual(result["adjacent_fusion_surface_share"], 0.50)
        self.assertEqual(
            result["target_cluster_table"][0]["family_sequence"],
            ["norm", "activation", "elementwise"])
        self.assertEqual(
            result["target_cluster_table"][0]["kernel_sequence"],
            ["rms_norm_f32", "silu_f32", "mul_f32"])
        self.assertEqual(
            R.hypothesis_result(result["elementwise_norm_target_share"])["verdict"],
            "READY_PROFILE_SELECTED")

    def test_a_subthreshold_profile_falsifies_target_selection(self):
        result = R.hypothesis_result(0.199)
        self.assertEqual(result["verdict"], "FALSIFIED_PROFILE_TARGET")
        self.assertEqual(result["authority"], "target_selection_only")

    def test_large_trace_reduction_happens_after_sampler_teardown(self):
        source = inspect.getsource(R.run)
        self.assertLess(
            source.index("stop_sampler_and_release("),
            source.index("summarize_timestamps(Path("))


if __name__ == "__main__":
    unittest.main()
