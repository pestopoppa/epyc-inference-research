#!/usr/bin/env python3
"""Tests for deterministic Laguna Q4-versus-IQ2 paired-read generation."""

import importlib.util
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("paired_read", BASE / "regenerate_paired_quality_read.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairedReadTests(unittest.TestCase):
    def test_sealed_reports_produce_expected_paired_result(self) -> None:
        read = MODULE.build_read()
        self.assertEqual(read["q4"]["resolved"], 19)
        self.assertEqual(read["iq2"]["resolved"], 17)
        self.assertEqual(read["paired_read"]["overlap"], 13)
        self.assertEqual(len(read["paired_read"]["q4_only"]), 6)
        self.assertEqual(len(read["paired_read"]["iq2_only"]), 4)
        self.assertEqual(read["paired_read"]["exact_binomial_two_sided_p"], 0.75390625)

    def test_rendering_is_byte_stable(self) -> None:
        first = MODULE.build_read()
        second = MODULE.build_read()
        self.assertEqual(MODULE.render_json(first), MODULE.render_json(second))
        self.assertEqual(MODULE.render_markdown(first), MODULE.render_markdown(second))

    def test_denominator_and_harness_error_contracts_are_enforced(self) -> None:
        report = {"total_instances": 39, "submitted_instances": 40, "error_instances": 0}
        with self.assertRaises(MODULE.ValidationError):
            MODULE.validate_report(report, "bad")

    def test_pinned_official_report_hashes_are_enforced(self) -> None:
        self.assertEqual(
            MODULE.validate_pinned_report_hash(MODULE.Q4_REPORT, "Q4"),
            MODULE.EXPECTED_REPORT_SHA256["Q4"],
        )
        report = {"total_instances": 40, "submitted_instances": 40, "error_instances": 1}
        with self.assertRaises(MODULE.ValidationError):
            MODULE.validate_report(report, "bad")


if __name__ == "__main__":
    unittest.main()
