"""The guards must actually fire. A guard that never fires is the thing it guards against.

The first version of the doc detector used a regex over assert lines and reported
ZERO hits against `test_readme.py` -- which does pin README prose -- because the real
assertions compare a module constant against `self.text`, not an inline literal.
These tests exist so that failure cannot recur silently.
"""
from pathlib import Path
import tempfile
import unittest

from autokernel import check_regrowth_guards as guards

PACKAGE = Path(__file__).resolve().parent


class DocCouplingDetector(unittest.TestCase):

    def test_it_finds_the_known_coupled_tests(self):
        """If this stops finding test_readme.py, the detector has gone vacuous."""
        found = {name for name, _, _ in guards.doc_coupled_tests(PACKAGE)}
        self.assertIn("test_readme.py", found)
        self.assertIn("test_campaign_footprint.py", found)
        self.assertIn("test_program_md.py", found)

    def test_it_reports_which_document_couples_each_file(self):
        rows = {name: doc for name, _, doc in guards.doc_coupled_tests(PACKAGE)}
        self.assertEqual(rows["test_campaign_footprint.py"], "FOOTPRINT.md")
        self.assertEqual(rows["test_readme.py"], "README.md")

    def test_an_uncoupled_test_file_is_not_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_clean.py").write_text(
                "def test_x():\n    assert 1 == 1\n", encoding="utf-8")
            self.assertEqual(guards.doc_coupled_tests(root), [])

    def test_a_coupled_test_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_doc.py").write_text(
                'TEXT = (ROOT / "FOOTPRINT.md").read_text()\n', encoding="utf-8")
            rows = guards.doc_coupled_tests(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][2], "FOOTPRINT.md")

    def test_non_test_files_are_not_scanned(self):
        """The guard is about TESTS pinning docs, not about docs being mentioned."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "module.py").write_text('# see README.md\n', encoding="utf-8")
            self.assertEqual(guards.doc_coupled_tests(root), [])


class LocBudget(unittest.TestCase):

    def test_an_absent_loop_package_is_not_a_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            total, rows = guards.loop_package_loc(Path(tmp) / "nope")
            self.assertEqual((total, rows), (0, []))

    def test_it_counts_source_and_ignores_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "loop.py").write_text("a\nb\nc\n", encoding="utf-8")
            (root / "test_loop.py").write_text("x\n" * 500, encoding="utf-8")
            total, rows = guards.loop_package_loc(root)
            self.assertEqual(total, 3)
            self.assertEqual([name for name, _ in rows], ["loop.py"])

    def test_the_budget_fires_when_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "big.py").write_text("x\n" * 50, encoding="utf-8")
            self.assertEqual(
                guards.main(["--package", str(root), "--loop-package", str(root),
                             "--budget", "10"]), 1)

    def test_the_budget_passes_when_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "small.py").write_text("x\n" * 5, encoding="utf-8")
            self.assertEqual(
                guards.main(["--package", str(root), "--loop-package", str(root),
                             "--budget", "1000"]), 0)

    def test_the_declared_budget_is_the_one_documented(self):
        """3,450 against a subject of 42,494 -- the ratio is the point.

        A tripwire, not an arithmetic check: it exists so the budget cannot move
        without someone editing this literal and saying why. It was raised
        3,000 -> 3,400 on 2026-08-29 for concurrency (pipeline.py plus pool.py,
        ~640 lines, 5.6x throughput), with the rationale stated at the constant --
        but this test was not updated with it, and CI could not report the drift,
        because CI had been dying on a missing pytest since its very first run.
        Those are one defect, not two: a guard nobody could see failing.
        """
        self.assertEqual(guards.LOOP_LOC_BUDGET, 3450)


if __name__ == "__main__":
    unittest.main()
