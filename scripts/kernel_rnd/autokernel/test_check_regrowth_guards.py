"""The guards must actually fire. A guard that never fires is the thing it guards against.

The first version of the doc detector used a regex over assert lines and reported
ZERO hits against `test_readme.py` -- which does pin README prose -- because the real
assertions compare a module constant against `self.text`, not an inline literal.
These tests exist so that failure cannot recur silently.
"""
from pathlib import Path
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
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


class PerFileCodeBaseline(unittest.TestCase):

    def _manifest(self, path: Path, files: dict[str, int], *,
                  status: str = "GENERATED") -> Path:
        path.write_text(json.dumps(guards.manifest_for(files, status=status)),
                        encoding="utf-8")
        return path

    def _package(self, root: Path) -> tuple[Path, Path]:
        package = root / "package"
        loop = package / "loop"
        loop.mkdir(parents=True)
        return package, loop

    def _main(self, package: Path, baseline: Path, *args: str,
              stdout: io.StringIO | None = None,
              stderr: io.StringIO | None = None) -> int:
        with patch.object(guards, "DEFAULT_CODE_BASELINE", baseline):
            with redirect_stdout(stdout or io.StringIO()), redirect_stderr(stderr or io.StringIO()):
                return guards.main(["--package", str(package), *args])

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

    def test_source_map_preserves_recursive_python_scope_and_test_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.py").write_text("x = 1\n", encoding="utf-8")
            (root / "test_skip.py").write_text("x = 1\n" * 500, encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "also_keep.py").write_text("y = 2\n", encoding="utf-8")
            self.assertEqual(guards.composition_by_file(root),
                             {"keep.py": 1, "nested/also_keep.py": 1})

    def test_metric_preserves_docstring_comment_blank_and_code_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.py"
            path.write_text('"""module docs"""\nx = 1\n# note\n\n', encoding="utf-8")
            self.assertEqual(guards.file_composition(path), (1, 2, 1))
            path.write_text('"""module docs"""\nx = 1\n# note\n\n# added\n\n',
                            encoding="utf-8")
            self.assertEqual(guards.file_composition(path), (1, 3, 2))

    def test_ast_and_inline_comment_line_semantics_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.py"
            path.write_text('def f():\n    """first\n    second"""\n'
                            '    x = 1  # inline comment\n    return x\n', encoding="utf-8")
            self.assertEqual(guards.file_composition(path), (2, 3, 0))
            self.assertEqual(guards.file_composition(path), (2, 3, 0))

    def test_growth_in_one_file_cannot_be_subsidized_by_another_file_shrinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.py", root / "b.py"
            a.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            b.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"a.py": 3, "b.py": 3})
            a.write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
            b.write_text("a = 1\nb = 2\n", encoding="utf-8")
            self.assertEqual(sum(guards.composition_by_file(root).values()), 6)
            errors = guards.check_code_baseline(baseline, guards.composition_by_file(root))
            self.assertTrue(any("a.py" in error and "exceeds ceiling" in error
                                for error in errors))

    def test_added_prose_and_blank_lines_do_not_exceed_code_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.py"
            source.write_text('"""module docs"""\nx = 1\n', encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"a.py": 1})
            source.write_text('"""module docs"""\nx = 1\n# note\n\n',
                              encoding="utf-8")
            self.assertEqual(guards.check_code_baseline(
                baseline, guards.composition_by_file(root)), [])

    def test_reduction_passes_and_does_not_raise_its_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.py"
            source.write_text("a = 1\nb = 2\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"a.py": 2})
            source.write_text("a = 1\n", encoding="utf-8")
            self.assertEqual(guards.check_code_baseline(
                baseline, guards.composition_by_file(root)), [])

    def test_new_source_and_stale_manifest_path_both_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json",
                                      {"a.py": 1, "removed.py": 4})
            (root / "new.py").write_text("b = 2\n", encoding="utf-8")
            errors = guards.check_code_baseline(baseline, guards.composition_by_file(root))
            self.assertTrue(any("new in-scope source" in error and "new.py" in error
                                for error in errors))
            self.assertTrue(any("stale baseline path" in error and "removed.py" in error
                                for error in errors))

    def test_ungenerated_template_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._manifest(root / "baseline.json", {}, status="UNGENERATED")
            self.assertTrue(any("UNGENERATED" in error for error in
                                guards.check_code_baseline(baseline, {"a.py": 1})))

    def test_duplicate_keys_and_bad_schema_paths_and_counts_fail(self):
        valid = guards.manifest_for({"a.py": 1})
        encoded = json.dumps(valid)
        bad = [(encoded.replace('"schema_version": 1',
                                '"schema_version": 1, "schema_version": 1'), "duplicate JSON key"),
               (json.dumps({**valid, "schema_version": True}), "schema_version must be integer")]
        for name, value in (("../a.py", 1), ("/a.py", 1), ("a.py", True),
                            ("a.py", -1)):
            message = "invalid relative POSIX source path" if isinstance(value, int) and type(value) is int and value >= 0 else "must be a nonnegative integer"
            bad.append((json.dumps({**valid, "files": {name: value}}), message))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            for raw, message in bad:
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    self.assertTrue(any(message in error for error in
                                        guards.check_code_baseline(path, {"a.py": 1})))

    def test_scope_metric_schema_and_extra_field_drift_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            base = guards.manifest_for({"a.py": 1})
            for key, value in (("schema_version", 2), ("source_revision", "wrong"), ("scope", "different.scope"),
                               ("metric", "different.metric"), ("unexpected", 1)):
                with self.subTest(key=key):
                    candidate = dict(base)
                    candidate[key] = value
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    self.assertTrue(guards.check_code_baseline(path, {"a.py": 1}))

    def test_explicit_exception_is_bounded_and_requires_approval_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            manifest = guards.manifest_for({"a.py": 1})
            manifest["exceptions"] = {"a.py": {"ceiling": 2, "approval": "operator decision 1"},
                                      "new.py": {"ceiling": 1, "approval": "operator decision 2"}}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(guards.check_code_baseline(path, {"a.py": 2, "new.py": 1}), [])
            self.assertTrue(guards.check_code_baseline(path, {"a.py": 3, "new.py": 1}))
            self.assertTrue(guards.check_code_baseline(path, {"a.py": 2}))
            manifest["exceptions"]["a.py"]["approval"] = ""
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(guards.check_code_baseline(path, {"a.py": 2, "new.py": 1}))

    def test_main_uses_default_baseline_clean_passes_and_one_file_growth_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            source = loop / "a.py"
            source.write_text("a = 1\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"a.py": 1})
            self.assertEqual(self._main(package, baseline), 0)
            source.write_text("a = 1\nb = 2\n", encoding="utf-8")
            errors = io.StringIO()
            self.assertEqual(self._main(package, baseline, stderr=errors), 1)
            self.assertIn("a.py", errors.getvalue())

    def test_generator_prints_json_without_mutating_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            (loop / "a.py").write_text("a = 1\n", encoding="utf-8")
            baseline = root / "baseline.json"
            original = json.dumps(guards.manifest_for({}, status="UNGENERATED")) + "\n"
            baseline.write_text(original, encoding="utf-8")
            output = io.StringIO()
            self.assertEqual(self._main(package, baseline, "--emit-baseline-json",
                                        stdout=output), 0)
            self.assertEqual(baseline.read_text(encoding="utf-8"), original)
            self.assertEqual(json.loads(output.getvalue()),
                             guards.manifest_for({"a.py": 1}))

    def test_obsolete_code_budget_option_cannot_bypass_per_file_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            (loop / "a.py").write_text("a = 1\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {}, status="UNGENERATED")
            with patch.object(guards, "DEFAULT_CODE_BASELINE", baseline):
                with self.assertRaises(SystemExit) as raised:
                    guards.main(["--package", str(package), "--code-budget", "9999999"])
            self.assertEqual(raised.exception.code, 2)

    def test_prose_alone_can_never_trip_the_code_baseline(self):
        """The whole reason the instrument changed, asserted.

        Under the old total-line count the cheapest way to land a fix was to delete the
        incident record the guard's own note calls load bearing -- so a package that is
        almost entirely comment MUST still pass, and the same file's code MUST NOT.
        Reads exit 1 under a total-line count; exit 0 is the property being bought.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            (loop / "prose.py").write_text('"""d."""\n' + "# note\n" * 400 + "x = 1\n",
                                           encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"prose.py": 1})
            self.assertEqual(self._main(package, baseline, "--budget", "10"), 0)

    def test_a_total_over_budget_is_reported_and_not_a_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            (loop / "big.py").write_text("# c\n" * 400 + "x = 1\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"big.py": 1})
            output = io.StringIO()
            self.assertEqual(self._main(package, baseline, "--budget", "10",
                                        stdout=output), 0)
            self.assertIn("informational budget 10", output.getvalue())

    def test_the_budget_fires_when_exceeded(self):
        """CODE over the code budget fails. The subject is code, not lines."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            (loop / "big.py").write_text("x = 1\n" * 50, encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"big.py": 10})
            self.assertEqual(self._main(package, baseline), 1)

    def test_per_file_budget_passes_when_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, loop = self._package(root)
            (loop / "small.py").write_text("x\n" * 5, encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"small.py": 5})
            self.assertEqual(self._main(package, baseline), 0)


if __name__ == "__main__":
    unittest.main()
