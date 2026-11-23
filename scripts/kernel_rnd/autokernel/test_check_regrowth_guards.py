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
        """If this stops finding a known doc-reading test, the detector has gone vacuous.

        The original anchors (`test_readme.py`, `test_campaign_footprint.py`,
        `test_program_md.py`) were deleted with the retired `campaign.py` driver
        (C5-05/C5-06, 2026-11-02); these are surviving files that still read a
        guarded document.
        """
        found = {name for name, _, _ in guards.doc_coupled_tests(PACKAGE)}
        self.assertIn("execution/test_worktree.py", found)
        self.assertIn("loop/test_patch_retention.py", found)
        self.assertIn("loop/test_loop.py", found)

    def test_it_reports_which_document_couples_each_file(self):
        rows = {name: doc for name, _, doc in guards.doc_coupled_tests(PACKAGE)}
        self.assertEqual(rows["execution/test_worktree.py"], "README.md")
        self.assertEqual(rows["loop/test_loop.py"], "program.md")

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

    def test_growth_in_one_file_is_paid_for_by_another_file_shrinking(self):
        """Total-budget redesign (2026-11-23): the FAIL is the total; a per-file
        overrun is advisory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.py", root / "b.py"
            a.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            b.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json", {"a.py": 3, "b.py": 3})
            a.write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
            b.write_text("a = 1\nb = 2\n", encoding="utf-8")
            self.assertEqual(sum(guards.composition_by_file(root).values()), 6)
            self.assertEqual(guards.check_code_baseline(
                baseline, guards.composition_by_file(root)), [])

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

    def test_new_source_is_admitted_and_stale_path_only_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            baseline = self._manifest(root / "baseline.json",
                                      {"a.py": 1, "removed.py": 4})
            (root / "new.py").write_text("b = 2\n", encoding="utf-8")
            report = guards.evaluate(baseline, guards.composition_by_file(root))
            self.assertEqual(report["errors"], [])
            self.assertTrue(any("auto-admitted" in w and "new.py" in w for w in report["warnings"]))
            self.assertTrue(any("stale baseline path" in w and "removed.py" in w
                                for w in report["warnings"]))

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
            self.assertEqual(guards.check_code_baseline(path, {"a.py": 2}), [])
            manifest["exceptions"]["a.py"]["approval"] = ""
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(guards.check_code_baseline(path, {"a.py": 2, "new.py": 1}))

    def test_main_uses_default_baseline_clean_passes_and_total_growth_fails(self):
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
            self.assertIn("exceeds", errors.getvalue())

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
            self.assertEqual(self._main(package, baseline), 0)

    def test_the_removed_informational_loc_budget_flag_is_gone(self):
        self.assertFalse(hasattr(guards, "LOOP_LOC_BUDGET"))
        self.assertFalse(hasattr(guards, "LOOP_CODE_BUDGET"))

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


class TotalBudget(unittest.TestCase):
    """The 2026-11-23 total-budget redesign (epyc-root code-budget audit §5)."""

    def _write(self, path: Path, files: dict[str, int], **extra) -> Path:
        manifest = guards.manifest_for(files)
        manifest.update(extra)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    def _signed(self, path: Path, files: dict[str, int], total: int) -> Path:
        return self._write(path, files, total_ceiling=total,
                           total_ceiling_approval="RATIFY-TEST")

    def test_fallback_is_sum_of_per_file_ceilings_with_exception_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp) / "b.json", {"a.py": 3, "b.py": 4},
                               exceptions={"b.py": {"ceiling": 6, "approval": "op"}})
            self.assertEqual(guards.evaluate(path, {"a.py": 9})["ceiling"], 9)
            self.assertEqual(guards.check_code_baseline(path, {"a.py": 9}), [])
            errors = guards.check_code_baseline(path, {"a.py": 10})
            self.assertTrue(any("exceeds fallback" in e and "by 1" in e for e in errors))

    def test_signed_total_ceiling_is_the_only_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._signed(Path(tmp) / "b.json", {"a.py": 1}, 5)
            self.assertEqual(guards.check_code_baseline(path, {"a.py": 3, "new.py": 2}), [])
            self.assertTrue(guards.check_code_baseline(path, {"a.py": 3, "new.py": 3}))

    def test_exceptions_block_is_optional_and_total_needs_its_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            manifest = guards.manifest_for({"a.py": 1})
            del manifest["exceptions"]
            manifest["total_ceiling"] = 4
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("present together" in e for e in
                                guards.check_code_baseline(path, {"a.py": 1})))
            manifest["total_ceiling_approval"] = "RATIFY-X"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(guards.check_code_baseline(path, {"a.py": 1}), [])
            manifest["total_ceiling"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(guards.check_code_baseline(path, {"a.py": 1}))

    def test_band_is_max_ten_lines_or_five_percent(self):
        self.assertEqual(guards.warn_band(10), 10)
        self.assertEqual(guards.warn_band(200), 10)
        self.assertEqual(guards.warn_band(201), 11)
        self.assertEqual(guards.warn_band(1000), 50)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._signed(Path(tmp) / "b.json", {"a.py": 100, "b.py": 1000}, 10**6)
            report = guards.evaluate(path, {"a.py": 110, "b.py": 1051})
            self.assertEqual(report["errors"], [])
            self.assertEqual(len(report["candidates"]), 1)
            self.assertIn("b.py", report["candidates"][0])

    def test_tighten_lowers_per_file_only_drops_stale_and_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._signed(Path(tmp) / "b.json", {"a.py": 10, "b.py": 5, "gone.py": 7}, 40)
            self.assertEqual(guards.tighten(path, {"a.py": 6, "b.py": 8, "new.py": 3},
                                            per_file=True, total=False), [])
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["files"], {"a.py": 6, "b.py": 5})
            self.assertEqual(raw["total_ceiling"], 40)

    def test_tighten_total_takes_half_the_deletion_credit_and_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._signed(Path(tmp) / "b.json", {"a.py": 10}, 100)
            self.assertEqual(guards.tighten(path, {"a.py": 60}, per_file=False, total=True), [])
            self.assertEqual(json.loads(path.read_text())["total_ceiling"], 80)
            self.assertEqual(guards.tighten(path, {"a.py": 80}, per_file=False, total=True), [])
            self.assertEqual(json.loads(path.read_text())["total_ceiling"], 80)

    def test_tighten_refuses_red_runs_and_the_unsigned_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            red = self._signed(Path(tmp) / "red.json", {"a.py": 1}, 1)
            before = red.read_text()
            self.assertTrue(guards.tighten(red, {"a.py": 2}, per_file=True, total=True))
            self.assertEqual(red.read_text(), before)
            fallback = self._write(Path(tmp) / "f.json", {"a.py": 5})
            before = fallback.read_text()
            refusals = guards.tighten(fallback, {"a.py": 1}, per_file=True, total=False)
            self.assertTrue(any("signed in" in r for r in refusals))
            self.assertEqual(fallback.read_text(), before)

    def test_unsigned_raise_of_total_ceiling_is_refused(self):
        old = {**guards.manifest_for({"a.py": 5}), "total_ceiling": 10,
               "total_ceiling_approval": "RATIFY-A"}
        up = {**old, "total_ceiling": 11}
        down = {**old, "total_ceiling": 9}
        self.assertTrue(guards.check_total_raise(old, up, None))
        self.assertTrue(guards.check_total_raise(old, up, "autokernel: fix"))
        self.assertTrue(guards.check_total_raise(old, {**up, "total_ceiling_approval": "me"},
                                                 "x\n\nOperator-Ratified: yes"))
        self.assertEqual(guards.check_total_raise(old, {**up, "total_ceiling_approval": "RATIFY-B"},
                                                  "x\n\nOperator-Ratified: yes"), [])
        self.assertEqual(guards.check_total_raise(old, down, None), [])
        unsigned = guards.manifest_for({"a.py": 5})  # fallback 5: sign-in to 10 is a raise
        self.assertTrue(guards.check_total_raise(unsigned, old, "plain"))

    def test_raise_check_reads_git_history_and_the_working_copy(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def git(*args):
                subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                                "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)
            git("init", "-q")
            path = self._signed(repo / "b.json", {"a.py": 5}, 10)
            git("add", "b.json"); git("commit", "-q", "--no-verify", "-m", "base")
            self.assertEqual(guards.raise_check_from_git(path)[0], [])
            self._signed(path, {"a.py": 5}, 12)
            self.assertTrue(guards.raise_check_from_git(path)[0])  # uncommitted raise
            git("commit", "-q", "--no-verify", "-am", "sneaky")
            self.assertTrue(guards.raise_check_from_git(path)[0])  # unsigned committed raise
            path.write_text(path.read_text().replace("RATIFY-TEST", "unsigned-approval"))
            git("commit", "-q", "--no-verify", "--amend", "-am", "x\n\nOperator-Ratified: yes")
            self.assertTrue(guards.raise_check_from_git(path)[0])  # approval not RATIFY-*
            path.write_text(path.read_text().replace("unsigned-approval", "RATIFY-OK"))
            git("commit", "-q", "--no-verify", "--amend", "-am", "x\n\nOperator-Ratified: yes")
            self.assertEqual(guards.raise_check_from_git(path)[0], [])


if __name__ == "__main__":
    unittest.main()
