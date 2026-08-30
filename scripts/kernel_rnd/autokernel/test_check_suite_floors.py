#!/usr/bin/env python3
"""The suite-floor guard, mutation-tested.

A guard is only worth its line count if it goes red when the thing it watches breaks.
Every test here drives the guard through a REAL pytest collection in a temporary
directory, because the failure this guard exists to catch -- a suite that quietly got
smaller -- is invisible to any check that trusts a number it was handed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autokernel import check_suite_floors as floors


_ONE_TEST = """
def test_a():
    assert True
"""

_THREE_TESTS = _ONE_TEST + """
def test_b():
    assert True


def test_c():
    assert True
"""

_DOES_NOT_IMPORT = """
import a_module_that_is_not_installed_anywhere  # noqa: F401


def test_a():
    assert True
"""


class Collection(unittest.TestCase):
    def suite(self, body: str):
        tmp = tempfile.TemporaryDirectory()
        (Path(tmp.name) / "test_sample.py").write_text(body, encoding="utf-8")
        return tmp

    def test_it_counts_what_pytest_actually_collects(self):
        with self.suite(_THREE_TESTS) as tmp:
            self.assertEqual(floors.collected(("test_sample.py",), root=Path(tmp)), 3)

    def test_a_suite_that_shrank_is_a_different_number(self):
        """The mutation: the same path, fewer tests. This is the whole point."""
        with self.suite(_THREE_TESTS) as tmp:
            before = floors.collected(("test_sample.py",), root=Path(tmp))
        with self.suite(_ONE_TEST) as tmp:
            after = floors.collected(("test_sample.py",), root=Path(tmp))
        self.assertEqual((before, after), (3, 1))

    def test_a_collection_error_raises_and_is_never_laundered_to_zero(self):
        """A suite that fails to import must not read as a suite that shrank to 0.

        Returning 0 here would be the exact defect the guard refuses -- an error
        rendered as a plausible number -- and it would then be reported as a floor
        violation, which names the wrong cause.
        """
        with self.suite(_DOES_NOT_IMPORT) as tmp:
            with self.assertRaises(RuntimeError):
                floors.collected(("test_sample.py",), root=Path(tmp))

    def test_a_path_matching_nothing_raises_rather_than_returning_zero(self):
        with self.suite(_ONE_TEST) as tmp:
            with self.assertRaises(RuntimeError):
                floors.collected(("no_such_test_file.py",), root=Path(tmp))


class Floors(unittest.TestCase):
    def test_every_declared_floor_is_met_by_the_real_suites(self):
        """The guard's own subject. If this fails, a suite really did shrink."""
        for name, (floor, paths) in floors.SUITE_FLOORS.items():
            with self.subTest(suite=name):
                self.assertGreaterEqual(floors.collected(paths), floor)

    def test_the_declared_paths_all_exist(self):
        """A floor over a path that no longer exists is a floor over nothing."""
        for name, (_floor, paths) in floors.SUITE_FLOORS.items():
            for path in paths:
                with self.subTest(suite=name, path=path):
                    self.assertTrue((floors.REPO_ROOT / path).exists(), path)

    def test_the_workflow_runs_exactly_what_the_floors_declare(self):
        """A floor over paths CI does not run guards a suite nobody executes.

        The two lists are written in different files and different languages, so they
        drift the moment someone adds a test module to one and not the other -- and
        the drift is invisible, because both halves stay green: the floor still counts
        the module, and CI still passes without running it.
        """
        workflow = (floors.REPO_ROOT
                    / ".github/workflows/autokernel-guards.yml").read_text("utf-8")
        for name, (_floor, paths) in floors.SUITE_FLOORS.items():
            for path in paths:
                with self.subTest(suite=name, path=path):
                    self.assertIn(path, workflow,
                                  f"{path} carries a floor but CI never runs it")

    def test_main_refuses_a_floor_the_suite_cannot_meet(self):
        """Mutation-test of the exit code, not just of the count."""
        original = dict(floors.SUITE_FLOORS)
        try:
            floors.SUITE_FLOORS.clear()
            floors.SUITE_FLOORS["impossible"] = (
                10_000, ("scripts/kernel_rnd/autokernel/loop/",))
            self.assertEqual(floors.main(), 1)
            floors.SUITE_FLOORS.clear()
            floors.SUITE_FLOORS["met"] = (1, ("scripts/kernel_rnd/autokernel/loop/",))
            self.assertEqual(floors.main(), 0)
        finally:
            floors.SUITE_FLOORS.clear()
            floors.SUITE_FLOORS.update(original)


if __name__ == "__main__":
    unittest.main()
