"""CPU-only contract and parser tests for the prepared RVP-C4-10 probe."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/benchmark/run_rocprofv3_pc_sampling_probe.py"
PROBE = ROOT / "scripts/benchmark/rocprofv3_pc_sampling_probe.cpp"

SPEC = importlib.util.spec_from_file_location("pc_sampling_probe_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def csv_text(*, stall_fields: bool = False, populated: bool = False) -> str:
    fields = [
        "Sample_Timestamp", "Exec_Mask", "Dispatch_Id", "Instruction",
        "Instruction_Comment", "Correlation_Id"]
    values = ["1", "255", "2", "s_waitcnt", "probe.cpp:1", "3"]
    if stall_fields:
        fields += ["Wave_Issued_Instruction", "Instruction_Type", "Stall_Reason"]
        values += (["1", "VALU", "WAITCNT"] if populated else ["", "", ""])
    return ",".join(fields) + "\n" + ",".join(values) + "\n"


class PcSamplingProbeContractTests(unittest.TestCase):
    def test_default_invocation_is_plan_only(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--output-dir", directory],
                check=True, text=True, stdout=subprocess.PIPE)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["status"], "prepared_not_run")
            self.assertEqual(plan["max_total_seconds"], 1800.0)
            self.assertIn("--pc-sampling-beta-enabled",
                          plan["commands"]["profile"])
            self.assertIn("host_trap", plan["commands"]["profile"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_live_execution_requires_explicit_window_ack(self):
        args = MODULE.parser().parse_args([
            "--output-dir", "/tmp/not-created", "--execute",
            "--source-commit", "a" * 40])
        with self.assertRaisesRegex(
                MODULE.ProbeContractError, "exclusive-gpu-window"):
            MODULE.execute(args)

    def test_host_trap_without_stall_fields_is_decision_grade_negative(self):
        result = MODULE.classify_host_trap_csv(csv_text())
        self.assertEqual(
            result["classification"],
            "host_trap_hotspot_only_no_stall_reason_fields")
        self.assertEqual(result["record_count"], 1)

    def test_stall_fields_are_distinguished_by_population(self):
        blank = MODULE.classify_host_trap_csv(csv_text(stall_fields=True))
        self.assertEqual(
            blank["classification"],
            "host_trap_stall_reason_fields_unpopulated")
        populated = MODULE.classify_host_trap_csv(
            csv_text(stall_fields=True, populated=True))
        self.assertEqual(
            populated["classification"],
            "unexpected_stall_reason_input_review_required")

    def test_empty_or_wrong_schema_is_inconclusive(self):
        empty = MODULE.classify_host_trap_csv(
            "Sample_Timestamp,Exec_Mask,Dispatch_Id,Instruction,"
            "Instruction_Comment,Correlation_Id\n")
        self.assertEqual(empty["classification"], "inconclusive_no_samples")
        wrong = MODULE.classify_host_trap_csv("a,b\n1,2\n")
        self.assertEqual(wrong["classification"],
                         "inconclusive_schema_mismatch")

    def test_only_exact_option_failure_closes_on_rocm_6_2(self):
        self.assertEqual(MODULE.classify_cli_failure(
            "error: unrecognized option --pc-sampling-beta-enabled"),
            "pc_sampling_cli_unavailable_on_rocm_6_2")
        self.assertEqual(MODULE.classify_cli_failure("segmentation fault"),
                         "inconclusive_profiler_failure")

    def test_static_governance_and_exact_arch_refusal(self):
        runner = RUNNER.read_text(encoding="utf-8")
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("acquire_device_claim", runner)
        self.assertIn("RocmSmiSampler", runner)
        self.assertIn("assert_source_identity", runner)
        self.assertIn("MAX_TOTAL_SECONDS = 1800.0", runner)
        self.assertIn('authority": "diagnostic_only"', runner)
        self.assertNotIn("torch", runner.casefold())
        self.assertIn("expected exact gfx90a", source)
        self.assertIn("pc_sampling_spin", source)
        self.assertNotIn("llama", source.casefold())


if __name__ == "__main__":
    unittest.main()
