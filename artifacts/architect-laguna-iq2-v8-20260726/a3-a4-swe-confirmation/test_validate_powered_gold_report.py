#!/usr/bin/env python3
"""Direct regression tests for the powered-SWE gold acceptance validator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("validate_powered_gold_report.py")
SPEC = importlib.util.spec_from_file_location("validate_powered_gold_report", MODULE_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def manifest() -> dict:
    return {"candidate_ids": ["a", "b", "c"], "candidate_count": 3, "gold_validated_target_count": 2}


def report() -> dict:
    return {
        "schema_version": 2,
        "submitted_ids": ["a", "b", "c"],
        "completed_ids": ["a", "c"],
        "resolved_ids": ["c", "a"],
        "unresolved_ids": [],
        "empty_patch_ids": ["b"],
        "incomplete_ids": [],
        "error_ids": [],
        "total_instances": 3,
        "submitted_instances": 3,
        "completed_instances": 2,
        "resolved_instances": 2,
        "unresolved_instances": 0,
        "empty_patch_instances": 1,
        "error_instances": 0,
    }


class PoweredGoldAcceptanceTests(unittest.TestCase):
    def test_accepts_full_terminal_report_in_manifest_order(self) -> None:
        accepted, summary = VALIDATOR.validate(manifest(), report())
        self.assertEqual(accepted, ["a", "c"])
        self.assertEqual(summary["accepted_count"], 2)

    def test_rejects_partial_submission_even_with_target_resolved(self) -> None:
        candidate_report = report()
        candidate_report["submitted_ids"] = ["a", "c"]
        candidate_report["total_instances"] = 2
        candidate_report["submitted_instances"] = 2
        with self.assertRaisesRegex(ValueError, "submitted_ids must exactly match"):
            VALIDATOR.validate(manifest(), candidate_report)

    def test_rejects_duplicate_report_id(self) -> None:
        candidate_report = report()
        candidate_report["submitted_ids"] = ["a", "b", "b"]
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            VALIDATOR.validate(manifest(), candidate_report)

    def test_rejects_nonterminal_report(self) -> None:
        candidate_report = report()
        candidate_report["incomplete_ids"] = ["b"]
        with self.assertRaisesRegex(ValueError, "no incomplete or error IDs"):
            VALIDATOR.validate(manifest(), candidate_report)

    def test_rejects_inconsistent_partition(self) -> None:
        candidate_report = report()
        candidate_report["completed_ids"] = ["a"]
        candidate_report["completed_instances"] = 1
        with self.assertRaisesRegex(ValueError, "completed/resolved/unresolved partition"):
            VALIDATOR.validate(manifest(), candidate_report)

    def test_refuses_inconsistent_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.ids"
            path.write_text("wrong\n")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                VALIDATOR.write_atomic_or_verify(path, "a\nc\n")

    def test_atomic_writer_allows_idempotent_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.ids"
            VALIDATOR.write_atomic_or_verify(path, "a\nc\n")
            VALIDATOR.write_atomic_or_verify(path, "a\nc\n")
            self.assertEqual(path.read_text(), "a\nc\n")

    def test_second_output_conflict_is_preflighted_before_first_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / "accepted.ids"
            summary = root / "summary.json"
            summary.write_text("inconsistent\n")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                VALIDATOR.preflight_outputs(((accepted, "a\nc\n"), (summary, "expected\n")))
            self.assertFalse(accepted.exists())


if __name__ == "__main__":
    unittest.main()
