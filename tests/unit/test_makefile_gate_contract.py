"""Run the real Makefile with isolated fake tools; no live service or formatting."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class MakefileGateContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="make-gates-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        shutil.copyfile(Path(__file__).resolve().parents[2] / "Makefile", self.tmp / "Makefile")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for tool in ("bash", "find"):
            (self.bin / tool).symlink_to(shutil.which(tool))
        (self.tmp / "scripts").mkdir()
        (self.tmp / "scripts" / "sample.sh").write_text("#!/bin/bash\necho sample\n")
        (self.tmp / "README.md").write_text("# Sample\n")
        self.calls = self.tmp / "calls"
        for tool in ("shellcheck", "shfmt", "markdownlint", "curl"):
            self.install(tool)

    def install(self, tool, status=0):
        path = self.bin / tool
        path.write_text(f'#!/bin/bash\necho "{tool} $*" >> "$GATE_CALLS"\nexit {status}\n')
        path.chmod(0o755)

    def run_make(self, target):
        return subprocess.run(
            [shutil.which("make"), "--no-print-directory", target, "SHELL=/bin/bash"],
            cwd=self.tmp, env={**os.environ, "PATH": str(self.bin), "CI": "",
                               "GATE_CALLS": str(self.calls)},
            capture_output=True, text=True, timeout=15)

    def test_gates_require_each_linter(self):
        for tool in ("shellcheck", "shfmt", "markdownlint"):
            with self.subTest(tool=tool):
                (self.bin / tool).unlink()
                result = self.run_make("gates")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{tool} not installed", result.stderr)
                self.assertNotIn("All gates passed", result.stdout)
                self.install(tool)

    def test_gates_check_without_mutating_or_contacting_service(self):
        sample = self.tmp / "scripts" / "sample.sh"
        before = sample.read_bytes()
        result = self.run_make("gates")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text()
        self.assertIn("shfmt -d ", calls)
        self.assertNotIn("shfmt -w ", calls)
        self.assertNotIn("curl ", calls)
        self.assertEqual(sample.read_bytes(), before)

    def test_linter_failures_propagate(self):
        for tool, target in (("shellcheck", "shellcheck"), ("shfmt", "format"),
                             ("markdownlint", "mdlint")):
            with self.subTest(tool=tool):
                self.install(tool, status=7)
                self.assertNotEqual(self.run_make(target).returncode, 0)
                self.install(tool)

    def test_configured_markdown_failure_is_not_retried_without_config(self):
        (self.tmp / ".markdownlint.json").write_text("{}")
        self.install("markdownlint", status=7)
        self.assertNotEqual(self.run_make("mdlint").returncode, 0)
        self.assertEqual(self.calls.read_text().count("markdownlint "), 1)

    def test_explicit_reindex_fails_when_service_is_unavailable(self):
        self.install("curl", status=7)
        result = self.run_make("nextplaid-reindex")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NextPLAID not running", result.stderr)

    def test_dead_numerics_targets_are_removed(self):
        for target in ("check-numerics", "report-numerics"):
            self.assertIn("No rule to make target", self.run_make(target).stderr)


if __name__ == "__main__":
    unittest.main()
