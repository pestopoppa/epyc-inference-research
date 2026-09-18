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
