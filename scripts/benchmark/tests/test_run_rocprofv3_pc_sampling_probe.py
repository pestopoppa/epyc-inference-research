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


def profile_command() -> list[str]:
    return MODULE.profile_command(Path("/tmp/probe"), Path("/tmp/raw"))


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

    def test_exact_header_only_is_inconclusive(self):
        result = MODULE.classify_host_trap_csv(
            "Sample_Timestamp,Exec_Mask,Dispatch_Id,Instruction,"
            "Instruction_Comment,Correlation_Id\n")
        self.assertEqual(result["classification"], "inconclusive_no_samples")

    def test_blank_and_noncanonical_newlines_refuse(self):
        for text in ("", "\n", csv_text().replace("\n", "\r\n"),
                     csv_text().rstrip("\n"), csv_text() + "\n"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(MODULE.ProbeContractError):
                    MODULE.classify_host_trap_csv(text)

    def test_blank_internal_row_refuses(self):
        lines = csv_text().splitlines()
        with self.assertRaisesRegex(MODULE.ProbeContractError, "blank row"):
            MODULE.classify_host_trap_csv(
                lines[0] + "\n\n" + lines[1] + "\n")

    def test_row_width_must_be_exact(self):
        for text in (csv_text().replace(",3\n", ",3,extra\n"),
                     csv_text().replace(",3\n", "\n")):
            with self.subTest(text=text):
                with self.assertRaisesRegex(
                        MODULE.ProbeContractError, "row width"):
                    MODULE.classify_host_trap_csv(text)

    def test_header_must_be_exact_and_unique(self):
        fields = list(MODULE.HOST_TRAP_FIELDS)
        variants = (
            fields + ["Extra"],
            fields[:-1],
            fields[:-1] + [fields[0]],
            [fields[1], fields[0], *fields[2:]],
        )
        for header in variants:
            with self.subTest(header=header):
                with self.assertRaisesRegex(
                        MODULE.ProbeContractError, "exact schema"):
                    MODULE.classify_host_trap_csv(",".join(header) + "\n")

    def test_required_base_cells_must_be_populated(self):
        for index in range(len(MODULE.HOST_TRAP_FIELDS)):
            values = ["1", "255", "2", "s_waitcnt", "probe.cpp:1", "3"]
            values[index] = ""
            text = ",".join(MODULE.HOST_TRAP_FIELDS) + "\n"
            text += ",".join(values) + "\n"
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                        MODULE.ProbeContractError, "base field"):
                    MODULE.classify_host_trap_csv(text)

    def test_only_exact_option_failure_closes_on_rocm_6_2(self):
        self.assertEqual(MODULE.classify_cli_failure(
            "error: unrecognized option --pc-sampling-beta-enabled",
            command=profile_command()),
            "pc_sampling_cli_unavailable_on_rocm_6_2")
        self.assertEqual(MODULE.classify_cli_failure(
            "segmentation fault", command=profile_command()),
            "infrastructure_profiler_failure")

    def test_unrelated_unknown_option_cannot_launder_into_unavailable(self):
        diagnostic = (
            "error: unknown option --foo; usage includes "
            "--pc-sampling-beta-enabled")
        self.assertEqual(MODULE.classify_cli_failure(
            diagnostic, command=profile_command()),
            "infrastructure_profiler_failure")

    def test_nonexact_invocation_cannot_claim_unavailable(self):
        command = profile_command()
        command[7] = "2"
        self.assertEqual(MODULE.classify_cli_failure(
            "error: unknown option --pc-sampling-beta-enabled",
            command=command), "infrastructure_profiler_failure")

    def test_receipt_self_hash_is_canonical_and_tamper_evident(self):
        first = MODULE.seal_receipt({"z": [1, 2], "a": {"b": True}})
        second = MODULE.seal_receipt({"a": {"b": True}, "z": [1, 2]})
        self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])
        MODULE.validate_receipt_self_hash(first)

        tampered = dict(first)
        tampered["z"] = [1, 3]
        with self.assertRaisesRegex(
                MODULE.ProbeContractError, "does not match"):
            MODULE.validate_receipt_self_hash(tampered)

        tampered["receipt_sha256"] = "A" * 64
        with self.assertRaisesRegex(MODULE.ProbeContractError, "lowercase"):
            MODULE.validate_receipt_self_hash(tampered)

    def test_coherent_reseal_is_valid_but_changes_identity(self):
        original = MODULE.seal_receipt({"analysis": {"record_count": 1}})
        changed = MODULE.seal_receipt({"analysis": {"record_count": 2}})
        MODULE.validate_receipt_self_hash(changed)
        self.assertNotEqual(
            original["receipt_sha256"], changed["receipt_sha256"])

    def test_static_governance_and_exact_arch_refusal(self):
        runner = RUNNER.read_text(encoding="utf-8")
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("acquire_device_claim", runner)
        self.assertIn("RocmSmiSampler", runner)
        self.assertIn("assert_source_identity", runner)
        self.assertIn("MAX_TOTAL_SECONDS = 1800.0", runner)
        self.assertIn('authority": "diagnostic_only"', runner)
        self.assertIn("makes no HIP-residency claim", runner)
        self.assertIn("validate_receipt_self_hash(payload)", runner)
        self.assertNotIn("torch", runner.casefold())
        self.assertIn("expected exact gfx90a", source)
        self.assertIn("pc_sampling_spin", source)
        self.assertNotIn("llama", source.casefold())


if __name__ == "__main__":
    unittest.main()
