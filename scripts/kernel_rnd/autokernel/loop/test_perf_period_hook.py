"""The restored DS41 prompt passes the root PII hook; a same-table account does not."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROMPT = Path(__file__).with_name("fixtures") / "ds41-run8-planner-prompt.txt"


class PerfPeriodHookControl(unittest.TestCase):
    def _hook(self):
        root = os.environ.get("EPYC_ROOT")
        if not root:
            self.skipTest("set EPYC_ROOT to run the cross-repository staged hook control")
        hook = Path(root) / "scripts/hooks/pii_precommit.sh"
        self.assertTrue(hook.is_file(), hook)
        return hook

    def _scan(self, content):
        hook = self._hook()
        with tempfile.TemporaryDirectory(prefix="toc-rd1b-hook-") as folder:
            repo = Path(folder)
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True,
                           capture_output=True)
            fixture = repo / "research/perf.md"
            fixture.parent.mkdir()
            fixture.write_text(content, encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "research/perf.md"],
                           check=True, capture_output=True)
            return subprocess.run(["bash", str(hook)], cwd=repo,
                                  capture_output=True, timeout=60).returncode

    def test_exact_recorded_prompt_is_allowed(self):
        self.assertEqual(self._scan(PROMPT.read_text(encoding="utf-8")), 0)

    def test_same_table_account_number_row_is_blocked(self):
        prompt = PROMPT.read_text(encoding="utf-8")
        period_row = next(line for line in prompt.splitlines()
                          if line.startswith("| 50.02% |"))
        # Build the synthetic account number from safe fragments: the committed
        # test itself must not require a PII hook exemption.
        synthetic_account = "123456" + "789012"
        account_row = ("| 0.00% | " + synthetic_account
                       + " | runtime | account_id=" + synthetic_account + " |")
        self.assertIn(period_row, prompt)
        self.assertEqual(self._scan(prompt.replace(period_row,
                                                    period_row + "\n" + account_row, 1)), 1)


if __name__ == "__main__":
    unittest.main()
