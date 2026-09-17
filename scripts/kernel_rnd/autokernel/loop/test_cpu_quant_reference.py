"""Hardware-free tests for the independent fixed CPU quant oracle."""
import json
import struct
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import cpu_quant_reference as fixture


def q8_encoded_row(expert: int, row: int) -> bytes:
    blocks = []
    for block in range(8):
        ints = [int(16 * fixture._source_weight(expert, row, block * 32 + column))
                for column in range(32)]
        blocks.append(struct.pack("<e32b", 1 / 16, *ints))
    return b"".join(blocks)


def q8_fused_output() -> str:
    op = fixture.FUSED_OP
    lines = [f"{fixture.MARKER} Q8_0 {op} 256 40 2 272"]
    up, gate = {}, {}
    for expert in range(2):
        for row in range(fixture.ROWS):
            up[expert, row] = q8_encoded_row(expert, row)
            gate[expert, row] = q8_encoded_row(expert + 3, row + 5)
            lines.append(f"A {expert} {row} {up[expert, row].hex()}")
            lines.append(f"G {expert} {row} {gate[expert, row].hex()}")
    for token in range(fixture.TOKENS):
        for row in range(fixture.ROWS):
            value = fixture._reference("Q8_0", op, up, token, row, gate)
            lines.append(f"O {token} {row} {value.hex()}")
    return "\n".join(lines) + "\n"


def q8_output(op: str = "MUL_MAT_ID") -> str:
    experts = 2 if op == "MUL_MAT_ID" else 1
    lines = [f"{fixture.MARKER} Q8_0 {op} 256 40 2 272"]
    rows = {}
    for expert in range(experts):
        for row in range(fixture.ROWS):
            encoded = q8_encoded_row(expert, row)
            rows[expert, row] = encoded
            lines.append(f"A {expert} {row} {encoded.hex()}")
    for token in range(fixture.TOKENS):
        for row in range(fixture.ROWS):
            value = fixture._reference("Q8_0", op, rows, token, row)
            lines.append(f"O {token} {row} {value.hex()}")
    return "\n".join(lines) + "\n"


class QuantReferenceTest(unittest.TestCase):
    def test_fused_reference_checks_both_matrices_and_output(self):
        output = q8_fused_output()
        self.assertEqual(fixture._parse_and_compare(output, "Q8_0", fixture.FUSED_OP).status,
                         "pass")
        lines = output.splitlines()
        lines = [line for line in lines if not line.startswith("G 1 39 ")]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixture._parse_and_compare("\n".join(lines), "Q8_0", fixture.FUSED_OP)
        lines = output.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        lines[index] = "O 0 0 0x1.0p+10"
        self.assertEqual(fixture._parse_and_compare(
            "\n".join(lines), "Q8_0", fixture.FUSED_OP).status, "wrong")
        lines = output.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("G 0 0 "))
        lines[index] = "G 0 0 " + bytes(fixture.ROW_BYTES["Q8_0"]).hex()
        self.assertEqual(fixture._parse_and_compare(
            "\n".join(lines), "Q8_0", fixture.FUSED_OP).status, "wrong")

    def test_existing_cpu_fixture_reports_observed_error_without_new_gate(self):
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        baseline = float.fromhex(lines[index].split()[3])
        lines[index] = f"O 0 0 {(baseline + 0.009).hex()}"
        result = fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID")
        self.assertEqual(result.status, "pass")
        marker, receipt = result.detail.split(" ", 1)
        self.assertEqual(marker, fixture.METRIC_MARKER)
        metric = json.loads(receipt)
        self.assertEqual((metric["quant"], metric["op"], metric["outputs"]),
                         ("Q8_0", "MUL_MAT_ID", 80))
        self.assertEqual(metric["max_quant_abs_error"], 0)
        self.assertAlmostEqual(metric["max_output_abs_error"], 0.009)
        self.assertAlmostEqual(metric["max_output_limit_fraction"], 0.9)
        self.assertEqual((metric["output_abs_tol"], metric["output_rel_tol"]),
                         (fixture.ABS_TOL, fixture.REL_TOL))

    def test_suite_carries_each_existing_case_metric(self):
        completed = [subprocess.CompletedProcess([], 0, "", "")]
        completed += [subprocess.CompletedProcess([], 0, q8_output(op), "")
                      for op in fixture.OPS]
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(subprocess, "run", side_effect=completed):
            result = fixture.check_cpu_quant_suite(
                Path("/build"), Path("/source"), quants=("Q8_0",), ops=fixture.OPS)
        self.assertEqual(result.status, "pass")
        rows = [json.loads(line.split(" ", 1)[1]) for line in result.detail.splitlines()]
        self.assertEqual([row["op"] for row in rows], list(fixture.OPS))
        self.assertTrue(all(row["schema"] == "epyc.autokernel.cpu_quant_metric.v1"
                            for row in rows))

    def test_q8_scalar_oracle_accepts_both_ops_without_claiming_dispatch(self):
        for op in fixture.OPS:
            with self.subTest(op=op):
                result = fixture._parse_and_compare(q8_output(op), "Q8_0", op)
                self.assertEqual(result.status, "pass")
                self.assertFalse(result.path_verified)

    def test_changed_output_is_wrong_not_unavailable(self):
        output = q8_output().replace("O 0 0 ", "O 0 0 0x1.0p+10\nO 0 0 ", 1)
        with self.assertRaisesRegex(ValueError, "duplicate output"):
            fixture._parse_and_compare(output, "Q8_0", "MUL_MAT_ID")
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        lines[index] = "O 0 0 0x1.0p+10"
        result = fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID")
        self.assertEqual(result.status, "wrong")
        self.assertIn("mismatch", result.reason)

    def test_near_boundary_wrong_output_is_rejected(self):
        # The exact synthetic reference at this point is 0.625. A +0.012
        # perturbation is only 1.2% of the output, but exceeds both the
        # 0.01 absolute and 0.005 relative allowances.
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        baseline = float.fromhex(lines[index].split()[3])
        self.assertEqual(baseline, 0.625)
        lines[index] = f"O 0 0 {(baseline + 0.009).hex()}"
        self.assertEqual(fixture._parse_and_compare("\n".join(lines),
                                                    "Q8_0", "MUL_MAT_ID").status, "pass")
        lines[index] = f"O 0 0 {(baseline + 0.012).hex()}"
        self.assertEqual(fixture._parse_and_compare("\n".join(lines),
                                                    "Q8_0", "MUL_MAT_ID").status, "wrong")

    def test_broken_quant_encoding_is_wrong(self):
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("A 0 0 "))
        lines[index] = "A 0 0 " + bytes(fixture.ROW_BYTES["Q8_0"]).hex()
        result = fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID")
        self.assertEqual(result.status, "wrong")
        self.assertIn("quant encoding mismatch", result.reason)

    def test_q4_and_q5_packed_high_bits(self):
        scales = bytes([1] * 12)
        q4 = struct.pack("<ee", 1, 0) + scales + bytes([0x21] * 128)
        decoded4 = fixture._decode_row("Q4_K", q4)
        self.assertEqual(decoded4[:32], (1.0,) * 32)
        self.assertEqual(decoded4[32:64], (2.0,) * 32)
        high = bytes([0x01] * 32)
        q5 = struct.pack("<ee", 1, 0) + scales + high + bytes([0x21] * 128)
        decoded5 = fixture._decode_row("Q5_K", q5)
        self.assertEqual(decoded5[:32], (17.0,) * 32)
        self.assertEqual(decoded5[32:64], (2.0,) * 32)

    def test_missing_rows_and_identity_are_untrusted(self):
        output = q8_output()
        with self.assertRaisesRegex(ValueError, "identity"):
            fixture._parse_and_compare(output, "Q8_0", "MUL_MAT")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixture._parse_and_compare("\n".join(output.splitlines()[:-1]),
                                       "Q8_0", "MUL_MAT_ID")


if __name__ == "__main__":
    unittest.main()
