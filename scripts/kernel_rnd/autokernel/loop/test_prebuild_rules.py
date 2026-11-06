"""AK-FAILRULE-1 rules: empty-suite preflight and blank-only-hunk lint."""
import tempfile
import unittest
from pathlib import Path

from . import loop, prebuild_rules as R, resume

NORMAL = """diff --git a/a.c b/a.c
--- a/a.c
+++ b/a.c
@@ -1,2 +1,2 @@
-int x;
+int y;
@@ -10,0 +11,2 @@
+
+  // note
"""
EOF_BLANK = """diff --git a/a.c b/a.c
--- a/a.c
+++ b/a.c
@@ -1,2 +1,2 @@
-int x;
+int y;
@@ -99,0 +100 @@
+
"""


def _anchor(root, content):
    (Path(root) / "bin").mkdir(parents=True, exist_ok=True)
    (Path(root) / "bin" / "test-backend-ops").write_bytes(content)
    return Path(root)


class BlankHunk(unittest.TestCase):
    def test_fires_on_blank_only_hunk(self):
        reason = R.blank_only_hunk_refusal(EOF_BLANK)
        self.assertIn("blank_only_hunk", reason)
        self.assertIn("a.c#hunk2", reason)

    def test_blank_removal_only_hunk(self):
        self.assertTrue(R.blank_only_hunks(
            "diff --git a/b b/b\n@@ -3,2 +3,0 @@\n-\n-  \n"))

    def test_quiet_on_normal_and_mixed_hunks(self):
        self.assertIsNone(R.blank_only_hunk_refusal(NORMAL))
        self.assertIsNone(R.blank_only_hunk_refusal(""))


class SuitePreflight(unittest.TestCase):
    def test_empty_refuses_nonempty_passes(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            empty, full = _anchor(a, b"E"), _anchor(b, b"F")
            counts = {"GLU": 0, "MUL_MAT": 12}
            pre = R.SuitePreflight(lambda _bin, sel, _be: counts[sel])
            self.assertIn(R.SUITE_EMPTY_GATE, pre.refusal(full, ["MUL_MAT", "GLU"], "CPU"))
            self.assertIsNone(pre.refusal(full, ["MUL_MAT"], "CPU"))
            self.assertIsNone(pre.refusal(empty, ["MUL_MAT"], "CPU"))

    def test_cache_keyed_by_binary_digest_not_path_or_surface(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            one, same, other = _anchor(a, b"X"), _anchor(b, b"X"), _anchor(c, b"Y")
            pre = R.SuitePreflight(lambda *_: 3)
            pre.refusal(one, ["ADD"], "CPU")
            pre.refusal(one, ["ADD"], "CPU")
            self.assertEqual(pre.listings, 1)
            pre.refusal(same, ["ADD"], "CPU")        # identical digest, different path
            self.assertEqual(pre.listings, 1)
            pre.refusal(other, ["ADD"], "CPU")       # different binary -> relisted
            pre.refusal(one, ["MUL"], "CPU")         # different op -> relisted
            self.assertEqual(pre.listings, 3)

    def test_lister_fault_and_missing_binary_fall_through(self):
        with tempfile.TemporaryDirectory() as a:
            def boom(*_):
                raise OSError("x")
            self.assertIsNone(R.SuitePreflight(boom).refusal(_anchor(a, b"Z"), ["ADD"], "CPU"))
            self.assertIsNone(R.SuitePreflight(lambda *_: 0).refusal(Path(a) / "nope", ["ADD"], "CPU"))

    def test_default_lister_counts_supported_lines(self):
        import unittest.mock as m
        out = ("Backend 1/1: CPU\n  GLU(type=swiglu): \x1b[1;32mSUPPORTED\x1b[0m\n"
               "  GLU(type=x): \x1b[1;31mNOT SUPPORTED\x1b[0m\n")
        with m.patch("subprocess.run", return_value=m.Mock(stdout=out, stderr="")):
            self.assertEqual(R.list_suite_cases(Path("x"), "SWIGLU", "CPU"), 1)


class IntegrityWiring(unittest.TestCase):
    def _repo(self, root):
        import subprocess
        run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True,
                                        capture_output=True)
        run("init", "-q")
        run("config", "user.email", "t@t")
        run("config", "user.name", "t")
        path = Path(root) / "ggml" / "src" / "k.c"
        path.parent.mkdir(parents=True)
        path.write_text("int a;\n")
        run("add", ".")
        run("commit", "-qm", "base")
        return path

    def test_validate_candidate_refuses_eof_blank_hunk(self):
        from . import integrity
        with tempfile.TemporaryDirectory() as root:
            path = self._repo(root)
            path.write_text("int a;\n\n")
            with self.assertRaises(integrity.IntegrityRefused) as ctx:
                integrity.validate_candidate(Path(root), ["ggml/src/k.c"])
            self.assertEqual(ctx.exception.refusal_class, R.BLANK_HUNK_CLASS)


class RuleGates(unittest.TestCase):
    def test_new_gate_is_a_rule_gate_in_both_mirrors(self):
        self.assertIn(R.SUITE_EMPTY_GATE, loop._RULE_GATES)
        self.assertEqual(loop._RULE_GATES, resume.RULE_GATES)
        self.assertEqual(loop.classify_patch_rejection(
            "x", source="gate:oracle_suite_empty")["class"], "scope")


if __name__ == "__main__":
    unittest.main()
