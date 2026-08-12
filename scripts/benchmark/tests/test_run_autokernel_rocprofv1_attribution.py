#!/usr/bin/env python3
"""Zero-device tests for whole-model rocprof-v1 attribution."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import run_autokernel_rocprofv1_attribution as R


class RocprofV1AttributionTest(unittest.TestCase):
    def test_prompt_tokens_are_non_negative_unique(self):
        self.assertEqual(R.prompt_tokens("2048,8192,32768"), (2048, 8192, 32768))
        self.assertEqual(R.prompt_tokens("0"), (0,))
        with self.assertRaisesRegex(Exception, "non-negative"):
            R.prompt_tokens("32,-1")
        with self.assertRaisesRegex(Exception, "unique"):
            R.prompt_tokens("32,32")

    def test_profile_command_is_v1_timestamp_only(self):
        command = R.profile_command(
            Path("/bin/bench"), Path("/model.gguf"), tokens=2048,
            repetitions=1, profiler=Path("/bin/rocprof"),
            input_file=Path("/evidence/timestamps.txt"),
            output_file=Path("/evidence/p2048.csv"))
        self.assertEqual(command[:5], (
            "/bin/rocprof", "--tool-version", "1", "--timestamp", "on"))
        self.assertEqual(command[command.index("/usr/bin/taskset") + 1:command.index("/usr/bin/taskset") + 3],
                         ("-c", "184-191"))
        self.assertIn("GGML_CUDA_DISABLE_GRAPHS", R.profiler_environment(
            Path("/build/bin/bench"), R.parser().parse_args([
                "--source-root", "/source", "--binary", "/bin/bench",
                "--model", "/model", "--output-dir", "/evidence",
            ])))

    def test_profile_command_defaults_to_explicit_prefill(self):
        command = R.profile_command(
            Path("/bin/bench"), Path("/model.gguf"), tokens=2048,
            repetitions=1, profiler=Path("/bin/rocprof"),
            input_file=Path("/evidence/timestamps.txt"),
            output_file=Path("/evidence/p2048.csv"))
        self.assertEqual(command[command.index("-n") + 1], "0")

    def test_profile_command_threads_decode_tokens_to_llama_bench(self):
        command = R.profile_command(
            Path("/bin/bench"), Path("/model.gguf"), tokens=2048,
            repetitions=1, profiler=Path("/bin/rocprof"),
            input_file=Path("/evidence/timestamps.txt"),
            output_file=Path("/evidence/p2048.csv"), gen_tokens=128)
        self.assertEqual(command[command.index("-n") + 1], "128")

    def test_profile_command_binds_requested_gpu_layers(self):
        default_command = R.profile_command(
            Path("/bin/bench"), Path("/model.gguf"), tokens=2048,
            repetitions=1, profiler=Path("/bin/rocprof"),
            input_file=Path("/evidence/timestamps.txt"),
            output_file=Path("/evidence/p2048.csv"))
        partial_command = R.profile_command(
            Path("/bin/bench"), Path("/model.gguf"), tokens=2048,
            repetitions=1, profiler=Path("/bin/rocprof"),
            input_file=Path("/evidence/timestamps.txt"),
            output_file=Path("/evidence/p2048.csv"), gpu_layers=48)
        self.assertEqual(default_command[default_command.index("-ngl") + 1], "99")
        self.assertEqual(partial_command[partial_command.index("-ngl") + 1], "48")

    def test_profile_command_admits_explicit_decode_only_surface(self):
        command = R.profile_command(
            Path("/bin/bench"), Path("/model.gguf"), tokens=0,
            repetitions=1, profiler=Path("/bin/rocprof"),
            input_file=Path("/evidence/timestamps.txt"),
            output_file=Path("/evidence/p0.csv"), gen_tokens=128)
        self.assertEqual(command[command.index("-p") + 1], "0")
        self.assertEqual(command[command.index("-n") + 1], "128")
        self.assertEqual(command[command.index("-t") + 1], "8")
        with self.assertRaisesRegex(ValueError, "p0 requires"):
            R.bench_command(
                Path("/bin/bench"), Path("/model.gguf"), tokens=0,
                repetitions=1, gen_tokens=0)

    def test_generation_tokens_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            R.bench_command(
                Path("/bin/bench"), Path("/model.gguf"), tokens=2048,
                repetitions=1, gen_tokens=-1)

    def test_workload_phase_matches_generation_surface(self):
        self.assertEqual(R.workload_phase(0), "prefill")
        self.assertEqual(R.workload_phase(128), "prefill+decode")
        self.assertEqual(R.workload_phase(128, (0,)), "decode")
        self.assertEqual(R.workload_phase(128, (2048,)), "prefill+decode")
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            R.workload_phase(128, (0, 2048))

    def test_belief_claim_is_derived_from_bound_model_and_phase(self):
        digest = "ab" * 32
        claim = R.attribution_claim(
            model=Path("/models/Qwen3.5-122B-A10B-UD-IQ2_M.gguf"),
            model_sha256=digest, prompt_tokens=2048, gen_tokens=128,
            gpu_layers=48, share=0.125)
        self.assertIn("Qwen3.5-122B-A10B-UD-IQ2_M.gguf", claim)
        self.assertIn(f"SHA-256 {digest}", claim)
        self.assertIn("p2048/tg128/ngl48 prefill+decode", claim)
        self.assertNotIn("Qwen3.6-35B-A3B Q8", claim)

    def test_receipt_workload_binds_generation_tokens_and_phase(self):
        text = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn('"gen_tokens": args.gen_tokens', text)
        self.assertIn('"gpu_layers": args.gpu_layers', text)
        self.assertIn('"phase": workload_phase(args.gen_tokens, args.prompt_tokens)', text)

    def test_parser_exposes_generation_tokens(self):
        args = R.parser().parse_args([
            "--source-root", "/source", "--binary", "/bin/bench",
            "--model", "/model", "--output-dir", "/evidence",
            "--gen-tokens", "128",
        ])
        self.assertEqual(args.gen_tokens, 128)

    def test_parser_exposes_validated_gpu_layers(self):
        args = R.parser().parse_args([
            "--source-root", "/source", "--binary", "/bin/bench",
            "--model", "/model", "--output-dir", "/evidence",
            "--gpu-layers", "48",
        ])
        self.assertEqual(args.gpu_layers, 48)
        self.assertEqual(R.parser().parse_args([
            "--source-root", "/source", "--binary", "/bin/bench",
            "--model", "/model", "--output-dir", "/evidence",
        ]).gpu_layers, 99)
        for invalid in ("0", "-1", "not-an-int"):
            with self.assertRaises(SystemExit):
                R.parser().parse_args([
                    "--source-root", "/source", "--binary", "/bin/bench",
                    "--model", "/model", "--output-dir", "/evidence",
                    "--gpu-layers", invalid,
                ])

    def test_bench_result_requires_mi210_and_complete_samples(self):
        row = {
            "n_prompt": 32, "backends": "ROCm", "gpu_info": "AMD Instinct MI210",
            "flash_attn": 1, "n_gpu_layers": 99, "samples_ns": [100],
        }
        self.assertEqual(
            R.parse_bench_result(json_line(row), "", tokens=32, repetitions=1), row)
        row["n_gpu_layers"] = 48
        self.assertEqual(R.parse_bench_result(
            json_line(row), "", tokens=32, repetitions=1, gpu_layers=48), row)
        with self.assertRaisesRegex(RuntimeError, "requested GPU layer count"):
            R.parse_bench_result(
                json_line(row), "", tokens=32, repetitions=1, gpu_layers=47)
        row["gpu_info"] = "other"
        with self.assertRaisesRegex(RuntimeError, "MI210"):
            R.parse_bench_result(json_line(row), "", tokens=32, repetitions=1)

    def test_timestamp_summary_requires_gdn_and_reports_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timestamps.csv"
            fields = [
                "Dispatch_ID", "Kernel_Name", "GPU_ID",
                "Start_Timestamp", "End_Timestamp",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "Dispatch_ID": "0", "Kernel_Name": "gated_delta_net_cuda",
                    "GPU_ID": "4", "Start_Timestamp": "10", "End_Timestamp": "30",
                })
                writer.writerow({
                    "Dispatch_ID": "1", "Kernel_Name": "mul_mat_q",
                    "GPU_ID": "4", "Start_Timestamp": "30", "End_Timestamp": "90",
                })
            result = R.summarize_timestamps(path)
        self.assertEqual(result["dispatches"], 2)
        self.assertAlmostEqual(result["gated_delta_net_share"], 0.25)

    def test_timestamp_summary_accepts_direct_rocprof_v1_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timestamps.csv"
            fields = ["Index", "KernelName", "gpu-id", "BeginNs", "EndNs"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "Index": "0", "KernelName": "gated_delta_net_cuda",
                    "gpu-id": "4", "BeginNs": "10", "EndNs": "30",
                })
            result = R.summarize_timestamps(path)
        self.assertEqual(result["dispatches"], 1)
        self.assertEqual(result["gated_delta_net_share"], 1.0)

    def test_runner_writes_prospective_belief_measurements_only(self):
        text = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn('"producer": producer_identity()', text)
        self.assertIn('"cpu_claim_open": cpu_opened', text)
        self.assertIn('"cpu_claim_released": cpu_released', text)
        self.assertIn('"belief_measurements": belief_measurements', text)
        self.assertIn('"metric_direction": "lower_better"', text)
        self.assertIn('"reps_basis": "scored:llama-bench prompt repetitions"', text)
        self.assertIn('"model_sha256": tool_identity["model_sha256"]', text)
        self.assertIn('"phase": workload_phase(args.gen_tokens, (tokens,))', text)


def json_line(value):
    import json
    return json.dumps(value)


if __name__ == "__main__":
    unittest.main()
