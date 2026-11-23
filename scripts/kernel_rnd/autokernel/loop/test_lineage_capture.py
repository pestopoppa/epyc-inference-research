"""No live-loop call: fixture-only source-capture tests."""
import json
import tempfile
import unittest
from pathlib import Path

from .lineage_capture import capture


class LineageCaptureTests(unittest.TestCase):
    def test_root_source_has_no_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = capture(Path(directory), capture_id="anchor-1", parent_id=None,
                              base_commit="a" * 40, patch_bytes=None,
                              source_files={"src/a.cpp": b"int value = 1;\n"})
            self.assertIsNone(receipt["patch_sha256"])

    def test_seals_original_bytes_and_refuses_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = dict(capture_id="attempt-1", parent_id="attempt-0",
                          base_commit="a" * 40, patch_bytes=b"diff --git a/a b/a\n",
                          source_files={"src/a.cpp": b"int value = 2;\n"})
            receipt = capture(root, **kwargs)
            self.assertEqual(capture(root, **kwargs), receipt)
            stored = json.loads((root / "lineage/captures/attempt-1.json").read_text())
            self.assertEqual(stored["solution_sha256"], receipt["solution_sha256"])
            blob = root / "lineage/blobs" / receipt["solution_sha256"][:2] / (receipt["solution_sha256"] + ".txt")
            self.assertIn(b"int value = 2;", blob.read_bytes())
            with self.assertRaisesRegex(ValueError, "conflicts"):
                capture(root, **{**kwargs, "patch_bytes": b"different"})

    def test_refuses_path_escape_and_missing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(capture_id="attempt-1", parent_id="attempt-0",
                          base_commit="a" * 40, patch_bytes=b"patch",
                          source_files={"../escaped.cpp": b"x"})
            with self.assertRaisesRegex(ValueError, "unsafe"):
                capture(Path(directory), **kwargs)
            kwargs["source_files"] = {"ok.cpp": b"x"}
            kwargs["capture_id"] = "bad/id"
            with self.assertRaisesRegex(ValueError, "identity"):
                capture(Path(directory), **kwargs)


if __name__ == "__main__":
    unittest.main()


class CaptureAtKeepTests(unittest.TestCase):
    def _repo(self, root):
        import subprocess
        run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)
        run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
        (root / "ggml/src").mkdir(parents=True)
        (root / "ggml/src/a.c").write_text("int a = 1;\n")
        run("add", "."); run("commit", "-qm", "base")
        (root / "ggml/src/a.c").write_text("int a = 2;\n")
        return run

    def test_off_never_invokes_producer(self):
        from .lineage_capture import capture_at_keep
        calls = []
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(capture_at_keep(Path(d), Path(d), lane="l", mechanism_id="m",
                                              environ={}, _capture=lambda *a, **k: calls.append(k)))
        self.assertEqual(calls, [])

    def test_on_via_env_or_knob_invokes_with_right_record(self):
        from .lineage_capture import capture_at_keep
        for kw in ({"environ": {"AK_LINEAGE_CAPTURE": "1"}}, {"environ": {}, "enabled": True}):
            calls = []
            with tempfile.TemporaryDirectory() as d:
                repo = Path(d) / "wt"; repo.mkdir(); self._repo(repo)
                capture_at_keep(Path(d) / "store", repo, lane="l", mechanism_id="m",
                                _capture=lambda store, **k: calls.append(k), **kw)
            (k,) = calls
            self.assertEqual(k["source_files"], {"ggml/src/a.c": b"int a = 2;\n"})
            self.assertIn(b"int a = 2;", k["patch_bytes"])
            self.assertRegex(k["base_commit"], r"[0-9a-f]{40}")
            self.assertTrue(k["capture_id"].startswith("ak-"))

    def test_real_capture_and_failure_does_not_raise(self):
        from .lineage_capture import capture_at_keep
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "wt"; repo.mkdir(); self._repo(repo)
            receipt = capture_at_keep(Path(d) / "store", repo, lane="l", mechanism_id="m",
                                      environ={"AK_LINEAGE_CAPTURE": "1"})
            self.assertEqual(receipt["authority"], "offline_lineage_source_only")
            def boom(*a, **k):
                raise OSError("disk full")
            self.assertIsNone(capture_at_keep(Path(d) / "s2", repo, lane="l", mechanism_id="m",
                                              environ={"AK_LINEAGE_CAPTURE": "1"}, _capture=boom))
            self.assertIsNone(capture_at_keep(Path(d) / "s3", Path(d) / "nonrepo", lane="l",
                                              mechanism_id="m", enabled=True))
