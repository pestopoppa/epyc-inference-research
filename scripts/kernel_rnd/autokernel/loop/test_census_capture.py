"""The journal detects tampering and retains failure events."""
import json
import tempfile
import unittest
from pathlib import Path

from autokernel.loop import census_capture as capture


class JournalTest(unittest.TestCase):
    def test_chain_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            first = capture.append_event(path, {"kind": "start", "argv": ["probe"]})
            capture.append_event(path, {"kind": "terminal", "error": "timeout"})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[1]["previous_sha256"], first)
            self.assertEqual(records[1]["error"], "timeout")
            records[0]["argv"] = ["changed"]
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaises(ValueError):
                capture.append_event(path, {"kind": "terminal"})

    def test_reserved_fields_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                capture.append_event(Path(directory) / "capture", {"content_sha256": "fake"})

    def test_incomplete_prelaunch_is_recorded_as_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture"
            with self.assertRaises(ValueError):
                capture.begin(path, attempt_id="a", started_ns=1, argv=["probe"], env={},
                              shape={}, grant="", deadline="", kernel_commit="", inputs={})
            self.assertEqual(json.loads(path.read_text())["kind"], "refusal")
