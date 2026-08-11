from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .. import schemas
from . import instrument_integrity as I


class TestInstrumentSourcePin(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.anchor = self.root / "anchor"
        self.candidate = self.root / "candidate"
        for root in (self.anchor, self.candidate):
            for units in I.TRANSLATION_UNITS.values():
                for relative in units:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"reviewed {relative}\n".encode())

    def check(self):
        return I.compare_to_anchor(
            tool="llama-bench", candidate_root=str(self.candidate),
            anchor_root=str(self.anchor))

    def test_identical_instrument_passes(self):
        self.assertEqual(self.check().outcome, schemas.PASS)

    def test_one_byte_candidate_edit_fails(self):
        (self.candidate / "tools/llama-bench/llama-bench.cpp").write_bytes(
            b"weakened instrument\n")
        check = self.check()
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("may not edit", " ".join(check.reasons))

    def test_deleting_candidate_source_fails(self):
        (self.candidate / "tools/llama-bench/llama-bench.cpp").unlink()
        self.assertEqual(self.check().outcome, schemas.FAIL)

    def test_missing_anchor_source_fails(self):
        (self.anchor / "tools/llama-bench/llama-bench.cpp").unlink()
        self.assertEqual(self.check().outcome, schemas.FAIL)

    def test_unregistered_tool_is_not_a_pass(self):
        check = I.compare_to_anchor(
            tool="candidate-invented-bench", candidate_root=str(self.candidate),
            anchor_root=str(self.anchor))
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_complete_manifest_pins_all_three_instruments(self):
        check = I.compare_manifest_to_anchor(
            candidate_root=str(self.candidate), anchor_root=str(self.anchor))
        self.assertEqual(check.outcome, schemas.PASS)
        for relative in (
                "tests/test-backend-ops.cpp", "tests/test-quantize-perf.cpp"):
            with self.subTest(relative=relative):
                path = self.candidate / relative
                original = path.read_bytes()
                path.write_bytes(original + b"candidate edit\n")
                self.assertEqual(I.compare_manifest_to_anchor(
                    candidate_root=str(self.candidate),
                    anchor_root=str(self.anchor)).outcome, schemas.FAIL)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
