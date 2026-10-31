"""Exercise the campaign entry, native writer, and provenance refusal."""
import hashlib
import json
import subprocess
import sys
import unittest

from autokernel.lesson_native import verify_artifacts
from autokernel.test_lesson_native import NativeTest


class SubmissionTest(unittest.TestCase):
    setUp = NativeTest.setUp

    def submit(self, provenance, *, control=False):
        config = dict(self.args, artifact_root=str(self.root / "cli-run"),
                      argv=[str(self.exe)], cap_provenance=provenance)
        config["task_id"] = "CONTROL-fixture" if control else "operational-fixture"
        path = self.root / "submission.json"
        path.write_text(json.dumps(config))
        argv = [sys.executable, "-m", "autokernel.campaign", "submit-lesson",
                "--config", str(path), "--journal", str(self.root / "journal")]
        if control:
            argv.append("--control")
        result = subprocess.run(argv, capture_output=True, text=True, check=True)
        return json.loads(result.stdout), self.j.read_all()[-1]

    def test_unqualified_cap_is_unknown_and_never_executes(self):
        output, entry = self.submit(None)
        self.assertEqual((output["decision"], entry.payload["reason"]), ("UNKNOWN", "cap_missing"))
        self.assertFalse((self.root / "cli-run").exists())

    def test_control_runs_and_legacy_manifest_cannot_be_reappended(self):
        output, entry = self.submit({"kind": "control", "source": "synthetic fixture only"}, control=True)
        self.assertTrue(output["control"])
        self.assertTrue(verify_artifacts(entry.payload))
        self.assertEqual(self.j.retrieve(), [])
        manifest = self.root / "cli-run/manifest.json"
        record = json.loads(manifest.read_text())
        self.assertEqual(record["cap_provenance"]["kind"], "control")
        del record["cap_provenance"]
        data = json.dumps(record, sort_keys=True).encode()
        manifest.write_bytes(data)
        payload = dict(entry.payload)
        verification = dict(payload["verification"])
        path, frag = verification["run_ref"].rsplit("#", 1)
        parts = dict(part.split("=", 1) for part in frag.split(";"))
        parts["manifest"] = hashlib.sha256(data).hexdigest()
        verification["run_ref"] = path + "#" + ";".join(k + "=" + parts[k] for k in ("manifest", "baseline", "verification"))
        payload["verification"] = verification
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_artifacts(payload)
        with self.assertRaises(ValueError):
            self.j.append("LESSON_ADMISSION_EVALUATED", payload)
        # An existing legacy row remains readable and withheld even if its
        # artifact cannot qualify for a fresh append.
        journal_path = self.root / "journal/events.jsonl"
        legacy_row = json.loads(journal_path.read_text())
        legacy_row["payload"] = payload
        journal_path.write_text(json.dumps(legacy_row) + "\n")
        self.assertEqual(self.j.read_all()[0].payload, payload)
        self.assertEqual(self.j.retrieve(), [])

    def test_asserted_roofline_source_cannot_authorize_execution(self):
        output, entry = self.submit({"kind": "workload_roofline", "source": "unverified assertion"})
        self.assertEqual((output["decision"], entry.payload["reason"]), ("UNKNOWN", "cap_missing"))
        self.assertFalse((self.root / "cli-run").exists())
