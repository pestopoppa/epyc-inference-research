"""Offline checks for prospective KVU-12.6 sizes; no server or inference."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import kvu_cpu_matrix as matrix


class LengthSupportTest(unittest.TestCase):
    def test_comparison_cells_are_present_and_overflow_cell_is_absent(self):
        command = [sys.executable, str(Path(matrix.__file__)), "--dry-run", "--kv-modes", "unified",
                   "--nps", "4,8", "--pools", "524288", "--lens", "48k,96k", "--pmodes", "depth"]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        np4 = result.stdout.split("== launch unified_np4_pool524288", 1)[1].split("== launch", 1)[0]
        np8 = result.stdout.split("== launch unified_np8_pool524288", 1)[1].split("== launch", 1)[0]
        self.assertIn("cell  96k depth", np4)
        self.assertIn("cell  48k depth", np8)
        self.assertNotIn("cell  96k depth", np8)

    def test_unknown_length_fails_instead_of_silently_producing_no_cells(self):
        result = subprocess.run([sys.executable, str(Path(matrix.__file__)), "--dry-run", "--lens", "48x"],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown sizes", result.stderr)

    def test_bank_mode_includes_new_sizes_in_split_and_unified(self):
        command = [sys.executable, str(Path(matrix.__file__)), "--dry-run", "--from-bank", "--nps", "4,8",
                   "--pools", "524288", "--lens", "48k,96k", "--pmodes", "depth"]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertIn("cell  48k bank", result.stdout)
        self.assertIn("cell  96k bank", result.stdout)
        self.assertIn("split_np4_slot131072", result.stdout)

    def test_extended_bank_sizes_are_opt_in_and_ordered(self):
        self.assertEqual(matrix.BANK_NAMES, ["8k", "32k", "64k", "128k"])
        self.assertEqual(matrix.validate_names("48k,96k", matrix.AVAILABLE_BANK_NAMES, "bank", True), ["48k", "96k"])
        with self.assertRaises(ValueError):
            matrix.validate_names("96k,48k", matrix.AVAILABLE_BANK_NAMES, "bank", True)
        with self.assertRaises(ValueError):
            matrix.validate_names("48k,48k", matrix.AVAILABLE_BANK_NAMES, "bank", True)

    def test_short_old_bank_is_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "doc0.tokens.json"
            doc.write_text(json.dumps([1, 2, 3]))
            with self.assertRaisesRegex(ValueError, "use a new bank directory"):
                matrix.make_bank_tokens(0, None, 1, 4, tmp)
            self.assertEqual(json.loads(doc.read_text()), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
