#!/usr/bin/env python3
"""Zero-device tests for the INF-37 Q4_K unpack attribution runner."""
from __future__ import annotations

import argparse
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.benchmark import run_autokernel_q4k_unpack_attribution as R


class Q4KUnpackAttributionTest(unittest.TestCase):
    def _valid_profile(self, *, duration: float = 20.0,
                       int32_per_wave: float = 4.0,
                       valu_per_wave: float = 10.0) -> dict:
        counters = {counter: 10.0 for counter in R.LEGACY_DETERMINISTIC_SQ_COUNTERS}
        counters["SQ_WAVES"] = 10.0
        counters["SQ_INSTS_VALU_INT32"] = int32_per_wave * 10.0
        counters["SQ_INSTS_VALU"] = valu_per_wave * 10.0
        return {
            "counter_transport_valid": True,
            "device_duration_ns_median": duration,
            "counter_medians": counters,
            "raw_dispatch_evidence": [
                {"counters": dict(counters)}, {"counters": dict(counters)}],
            "counter_per_wave": {
                "SQ_INSTS_VALU_INT32": int32_per_wave,
                "SQ_INSTS_VALU": valu_per_wave,
            },
        }

    def test_generic_op_rows_are_exact_contiguous_production_shapes(self):
        q4k = R.test_file_line("q4_K", m=17408, n=1, k=5120).split()
        self.assertEqual([int(value) for value in q4k[:6]], [29, 0, 17408, 1, 1, 1])
        # op/count + 16 params + source count, then src0 type/k/m and strides.
        self.assertEqual(int(q4k[23]), 2)
        self.assertEqual(int(q4k[24]), 12)
        self.assertEqual([int(value) for value in q4k[25:29]], [5120, 17408, 1, 1])
        self.assertEqual([int(value) for value in q4k[29:33]], [144, 2880, 50135040, 50135040])
        q8 = R.test_file_line("q8_0", m=17408, n=1, k=5120).split()
        self.assertEqual(int(q8[24]), 8)
        self.assertEqual([int(value) for value in q8[29:33]], [34, 5440, 94699520, 94699520])

    def test_frozen_v9_enum_derives_mul_mat_as_29(self):
        frozen_v9_prefix = """\
enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_DUP, GGML_OP_ADD, GGML_OP_ADD_ID, GGML_OP_ADD1, GGML_OP_ACC,
    GGML_OP_SUB, GGML_OP_MUL, GGML_OP_DIV, GGML_OP_SQR, GGML_OP_SQRT,
    GGML_OP_LOG, GGML_OP_SIN, GGML_OP_COS, GGML_OP_SUM, GGML_OP_SUM_ROWS,
    GGML_OP_CUMSUM, GGML_OP_MEAN, GGML_OP_ARGMAX, GGML_OP_COUNT_EQUAL,
    GGML_OP_REPEAT, GGML_OP_REPEAT_BACK, GGML_OP_CONCAT, GGML_OP_SILU_BACK,
    GGML_OP_NORM, GGML_OP_RMS_NORM, GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM, GGML_OP_L2_NORM, GGML_OP_MUL_MAT,
};
"""
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / "ggml.h"
            header.write_text(frozen_v9_prefix, encoding="utf-8")
            self.assertEqual(R.derive_ggml_op_value(header, "GGML_OP_MUL_MAT"), 29)

    def test_test_file_repetition_is_explicit_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q4k.txt"
            receipt = R.write_test_file(path, "q4_K", m=17408, n=1, k=5120,
                                        repetitions=5)
            self.assertEqual(len(path.read_text().splitlines()), 5)
            self.assertEqual(receipt["rows"], 5)
            self.assertEqual(len(receipt["sha256"]), 64)

    def test_commands_use_test_file_and_sq_tcc_blocks(self):
        args = argparse.Namespace(
            omniperf_python="/venv/bin/python", omniperf="/tools/omniperf")
        command = R.omniperf_command(
            Path("/build/test-backend-ops"), Path("/evidence/q4k.txt"),
            Path("/evidence/block"), workload_name="inf37_q4k", backend="ROCm0",
            args=args)
        self.assertIn("--test-file", command)
        self.assertEqual(command[command.index("-b") + 1:command.index("-b") + 3],
                         ("SQ", "TCC"))
        self.assertIn("--no-roof", command)

    def test_omniperf_command_preserves_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system_python = root / "system-python"
            system_python.write_text("", encoding="utf-8")
            venv_python = root / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(system_python)
            args = argparse.Namespace(
                omniperf_python=str(venv_python), omniperf="/tools/omniperf")
            command = R.omniperf_command(
                Path("/build/test-backend-ops"), Path("/evidence/q4k.txt"),
                Path("/evidence/block"), workload_name="inf37_q4k", backend="ROCm0",
                args=args)
            self.assertEqual(command[0], str(venv_python))
            self.assertNotEqual(command[0], str(system_python))

    def test_rocprofv2_command_is_one_file_plugin_pass(self):
        args = argparse.Namespace(profiler_prefix="/profiler")
        command = R.rocprofv2_command(
            Path("/build/test-backend-ops"), Path("/evidence/q4k.txt"),
            Path("/evidence/counters.txt"), Path("/evidence/raw"),
            workload_name="inf37_q4k", backend="ROCm0", args=args)
        self.assertEqual(command[:9], (
            "/profiler/bin/rocprofv2", "-i", "/evidence/counters.txt",
            "--plugin", "file", "--plugin-version", "2", "-d", "/evidence/raw"))
        self.assertEqual(command.count("-i"), 1)
        self.assertIn("--test-file", command)
        self.assertEqual(
            R.ROCPROFV2_PMC_LINE,
            "pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VALU_INT32")

    def test_rocprofv2_listing_requires_exact_gfx90a_minimal_set(self):
        listing = "\n".join(
            f"  gfx90a:0 : {counter} : semantics for {counter}"
            for counter in R.ROCPROFV2_COUNTERS)
        parsed = R.parse_rocprofv2_counter_listing(listing)
        self.assertEqual(set(parsed), set(R.ROCPROFV2_COUNTERS))
        with self.assertRaisesRegex(RuntimeError, "lacks an exact"):
            R.parse_rocprofv2_counter_listing(
                "\n".join(listing.splitlines()[:-1]))

    def test_rocprofv2_support_probe_accepts_only_documented_rc1_quirk(self):
        listing = "\n".join(
            f"gfx90a:0 : {counter} : semantics for {counter}"
            for counter in R.ROCPROFV2_COUNTERS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiler = root / "prefix" / "bin" / "rocprofv2"
            profiler.parent.mkdir(parents=True)
            profiler.write_text("stub", encoding="utf-8")
            args = argparse.Namespace(profiler_prefix=str(root / "prefix"))
            with mock.patch.object(
                    R.O, "run_owned", return_value=(1, listing, "", 0.25)):
                result = R.validate_rocprofv2_counter_support(
                    args, env={}, output_dir=root)
            self.assertTrue(result["single_pass_group"])
            self.assertEqual(result["returncode"], 1)
            with mock.patch.object(
                    R.O, "run_owned", return_value=(1, listing, "unexpected", 0.25)):
                with self.assertRaisesRegex(RuntimeError, "failed rc=1"):
                    R.validate_rocprofv2_counter_support(args, env={}, output_dir=root)

    def test_rocprofv2_csv_selection_and_minimal_counter_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "pmc_1"
            raw.mkdir(parents=True)
            csv_path = raw / "results_cell.csv"
            fields = ["Kernel_Name", "Grid_Size", "Workgroup_Size",
                      "Start_Timestamp", "End_Timestamp", *R.ROCPROFV2_COUNTERS]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for rep in range(2):
                    writer.writerow({
                        "Kernel_Name": "void mul_mat_vec_q<(ggml_type)12, 1>()",
                        "Grid_Size": "17408", "Workgroup_Size": "128",
                        "Start_Timestamp": str(100 + 20 * rep),
                        "End_Timestamp": str(110 + 20 * rep),
                        "SQ_WAVES": "10", "SQ_INSTS_VALU": "20",
                        "SQ_INSTS_VALU_INT32": "30",
                    })
            self.assertEqual(R.select_rocprofv2_counter_csv(Path(tmp) / "raw"), csv_path)
            result = R.summarize_counter_table(
                csv_path, quant="q4_K", expected_dispatches=2,
                counter_fields=R.ROCPROFV2_COUNTERS,
                deterministic_counters=R.ROCPROFV2_COUNTERS)
            self.assertTrue(result["claim_eligible"])
            self.assertEqual(result["counter_per_wave"]["SQ_INSTS_VALU_INT32"], 3.0)

    def _write_v2_profile(self, raw_dir: Path, *, zero_counter: str | None = None) -> None:
        destination = raw_dir / "pmc_1"
        destination.mkdir(parents=True)
        fields = ["Kernel_Name", "Grid_Size", "Workgroup_Size",
                  "Start_Timestamp", "End_Timestamp", *R.ROCPROFV2_COUNTERS]
        with (destination / "results_cell.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rep in range(2):
                counters = {counter: "10" for counter in R.ROCPROFV2_COUNTERS}
                if zero_counter is not None:
                    counters[zero_counter] = "0"
                writer.writerow({
                    "Kernel_Name": "void mul_mat_vec_q<(ggml_type)12, 1>()",
                    "Grid_Size": "17408", "Workgroup_Size": "128",
                    "Start_Timestamp": str(100 + 20 * rep),
                    "End_Timestamp": str(110 + 20 * rep), **counters,
                })

    def _retry_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            transport_attempts=2, counter_transport="rocprofv2",
            profiler_prefix="/profiler", backend="ROCm0",
            profile_timeout_s=30.0, active_repetitions=2)

    def test_transport_retries_nonzero_exit_and_preserves_both_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm = Path(tmp) / "arm"
            arm.mkdir()
            calls = 0

            def run_owned(command, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return 139, "first-out", "first-segv", 0.5
                raw_dir = Path(command[command.index("-d") + 1])
                self._write_v2_profile(raw_dir)
                return 0, "second-out", "", 0.6

            with mock.patch.object(R.O, "run_owned", side_effect=run_owned):
                profile = R.profile_cell(
                    Path("/binary"), Path("/test-file"), arm,
                    workload_name="cell", quant="q4_K", args=self._retry_args(),
                    env={}, counter_file=Path("/counters"))
            self.assertEqual(profile["accepted_attempt"], 2)
            self.assertEqual([row["returncode"] for row in profile["attempts"]], [139, 0])
            self.assertEqual(profile["attempts"][0]["result"], "nonzero_profiler_exit")
            self.assertTrue((arm / "attempt-01/rocprofv2.stderr.txt").is_file())
            self.assertTrue((arm / "attempt-02/raw/pmc_1/results_cell.csv").is_file())

    def test_transport_retries_missing_artifact_but_not_parsed_zero_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm = Path(tmp) / "missing"
            arm.mkdir()
            calls = 0

            def missing_then_success(command, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    self._write_v2_profile(Path(command[command.index("-d") + 1]))
                return 0, "", "", 0.5

            with mock.patch.object(R.O, "run_owned", side_effect=missing_then_success):
                profile = R.profile_cell(
                    Path("/binary"), Path("/test-file"), arm,
                    workload_name="cell", quant="q4_K", args=self._retry_args(),
                    env={}, counter_file=Path("/counters"))
            self.assertEqual(profile["accepted_attempt"], 2)
            self.assertEqual(profile["attempts"][0]["result"], "missing_profiler_artifact")

            zero_arm = Path(tmp) / "zero"
            zero_arm.mkdir()
            calls = 0

            def parsed_zero(command, **_kwargs):
                nonlocal calls
                calls += 1
                self._write_v2_profile(
                    Path(command[command.index("-d") + 1]), zero_counter="SQ_WAVES")
                return 0, "", "", 0.5

            with mock.patch.object(R.O, "run_owned", side_effect=parsed_zero):
                failed = R.profile_cell(
                    Path("/binary"), Path("/test-file"), zero_arm,
                    workload_name="cell", quant="q4_K", args=self._retry_args(),
                    env={}, counter_file=Path("/counters"))
            self.assertEqual(calls, 1)
            self.assertEqual(failed["accepted_attempt"], 1)
            self.assertFalse(failed["counter_transport_valid"])
            self.assertEqual(
                failed["attempts"][0]["result"], "accepted_parsed_counter_failure")

    def test_correctness_output_requires_every_row(self):
        text = "backend_name,op_name,supported,hard_failure,error_message\n" + "\n".join(
            "ROCm0,MUL_MAT,1,0," for _ in range(3)) + "\n"
        self.assertEqual(R.validate_test_output(text, expected_rows=3)["rows"], 3)
        with self.assertRaisesRegex(RuntimeError, "expected 4"):
            R.validate_test_output(text, expected_rows=4)

    def _counter_csv(self, path: Path, *, quant: str, zero: str | None = None) -> None:
        fields = ["Kernel_Name", "Grid_Size", "Workgroup_Size",
                  "Start_Timestamp", "End_Timestamp", *R.PRIMARY_COUNTERS]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rep in range(2):
                row = {field: "10" for field in R.PRIMARY_COUNTERS}
                row.update({
                    "Kernel_Name": (
                        f"void mul_mat_vec_q<(ggml_type){R.QUANTS[quant]['type_id']}, 1, false, false>()"),
                    "Grid_Size": "17408", "Workgroup_Size": "128",
                    "Start_Timestamp": str(100 + rep * 20),
                    "End_Timestamp": str(110 + rep * 20),
                })
                if zero:
                    row[zero] = "0"
                writer.writerow(row)

    def test_counter_summary_requires_nonzero_sq_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pmc.csv"
            self._counter_csv(path, quant="q4_K")
            summary = R.summarize_counter_table(path, quant="q4_K", expected_dispatches=2)
            self.assertEqual(summary["counter_per_wave"]["SQ_INSTS_VALU_INT32"], 1.0)
            self._counter_csv(path, quant="q4_K", zero="SQ_INSTS_VALU_INT32")
            failed = R.summarize_counter_table(path, quant="q4_K", expected_dispatches=2)
            self.assertFalse(failed["counter_transport_valid"])
            self.assertIsNone(failed["counter_per_wave"])
            self.assertEqual(len(failed["raw_dispatch_evidence"]), 2)

    def test_summary_refuses_to_invent_inside_kernel_wall_share(self):
        arm = self._valid_profile()
        q40 = self._valid_profile(duration=15.0, int32_per_wave=2.0, valu_per_wave=7.0)
        q80 = self._valid_profile(duration=12.0, int32_per_wave=1.0, valu_per_wave=6.0)
        result = R.paired_block_summary([
            {"block": 0, "arms": {"q4_K": arm, "q4_0": q40, "q8_0": q80}}])
        self.assertIsNone(result["inside_unpack_wall_share"])
        self.assertEqual(
            result["comparisons"]["q4_K_minus_q4_0"][0]["int32_insts_per_wave_delta"], 2.0)

    def test_summary_omits_comparisons_when_any_arm_transport_is_invalid(self):
        valid = self._valid_profile()
        invalid = {
            "counter_transport_valid": False,
            "device_duration_ns_median": 15.0,
            "counter_per_wave": None,
        }
        result = R.paired_block_summary([{
            "block": 0, "arms": {"q4_K": invalid, "q4_0": valid, "q8_0": valid}}])
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(result["comparisons"]["q4_K_minus_q4_0"], [])
        self.assertEqual(result["comparison_eligibility"][0]["invalid_arms"], ["q4_K"])

    def test_cross_block_sq_drift_invalidates_every_comparison(self):
        blocks = []
        for block_id in range(2):
            blocks.append({
                "block": block_id,
                "arms": {quant: self._valid_profile() for quant in R.QUANTS},
            })
        drifted = copy.deepcopy(blocks[1]["arms"]["q4_K"])
        drifted["counter_medians"]["SQ_WAVES"] = 11.0
        for row in drifted["raw_dispatch_evidence"]:
            row["counters"]["SQ_WAVES"] = 11.0
        blocks[1]["arms"]["q4_K"] = drifted
        result = R.paired_block_summary(blocks, expected_blocks=2)
        waves = result["transport_integrity"]["quants"]["q4_K"]["counters"]["SQ_WAVES"]
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(waves["raw_unique_values"], [10.0, 11.0])
        self.assertEqual(waves["cross_block_median_unique_values"], [10.0, 11.0])
        self.assertTrue(waves["cross_block_drift"])
        self.assertFalse(waves["exactly_invariant"])
        self.assertTrue(all(not row["eligible"] for row in result["comparison_eligibility"]))
        self.assertEqual(result["comparisons"]["q4_K_minus_q4_0"], [])

    def test_missing_block_invalidates_complete_campaign_coverage(self):
        block = {"block": 0, "arms": {
            quant: self._valid_profile() for quant in R.QUANTS}}
        result = R.paired_block_summary([block], expected_blocks=2)
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(
            result["transport_integrity"]["quants"]["q4_K"]["missing_blocks"], [1])

    def test_rocprofv2_minimal_contract_can_gate_complete_invariant_blocks(self):
        blocks = [{
            "block": block_id,
            "arms": {quant: self._valid_profile() for quant in R.QUANTS},
        } for block_id in range(2)]
        result = R.paired_block_summary(
            blocks, expected_blocks=2,
            deterministic_counters=R.ROCPROFV2_COUNTERS)
        self.assertTrue(result["claim_eligible"])
        self.assertEqual(len(result["comparisons"]["q4_K_minus_q4_0"]), 2)
        self.assertEqual(
            result["transport_integrity"]["deterministic_counters"],
            list(R.ROCPROFV2_COUNTERS))

    def test_parser_defaults_are_model_derived_and_balanced(self):
        args = R.parser().parse_args([
            "--source-root", "/source", "--binary", "/binary",
            "--output-dir", "/evidence"])
        self.assertEqual((args.op_m, args.op_k), (17408, 5120))
        self.assertEqual(args.blocks, 4)
        self.assertEqual(args.active_repetitions, 5)
        self.assertEqual(args.source_commit, R.FROZEN_V9_COMMIT)
        self.assertEqual(args.counter_transport, "rocprofv2")
        self.assertEqual(args.transport_attempts, 2)


if __name__ == "__main__":
    unittest.main()
