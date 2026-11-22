import tempfile
import unittest
from pathlib import Path

from autokernel.journal import Journal
from autokernel.loop import keep_lesson

COMMIT = "a" * 40


def _kept(**over):
    row = {"status": "kept", "mechanism_id": "akm-fuse-x", "statement": "fuse the epilogue",
           "falsifier": "decode tok/s does not rise", "target_surface": "ggml-cpu.c",
           "champion_head": COMMIT}
    row.update(over)
    return row


class KeepLessonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.journal = Journal(str(Path(tmp.name) / "j"))
        self.journal.initialize()

    def _receipts(self):
        return [e for e in self.journal.read_all()
                if (e.payload or {}).get("schema") == "autokernel.lesson_admission.v1"]

    def test_keep_records_unknown_cap_missing_without_native_run(self):
        out = keep_lesson.submit_keep(self.journal, _kept())
        self.assertEqual(out, {"recorded": True, "reason": "cap_missing", "decision": "UNKNOWN"})
        (entry,) = self._receipts()
        self.assertEqual(entry.payload["commit"], COMMIT)
        self.assertIsNone(entry.payload["verification"])

    def test_replay_is_idempotent(self):
        keep_lesson.submit_keep(self.journal, _kept())
        again = keep_lesson.submit_keep(self.journal, _kept())
        self.assertEqual(again["reason"], "already_recorded")
        self.assertEqual(len(self._receipts()), 1)

    def test_ineligible_rows_record_nothing(self):
        for row, reason in ((_kept(status="rejected"), "not_a_keep"),
                            (_kept(champion_head="abc"), "no_full_commit"),
                            (_kept(statement=""), "no_declared_lesson")):
            self.assertEqual(keep_lesson.submit_keep(self.journal, row)["reason"], reason)
        self.assertEqual(self._receipts(), [])

    def test_asserted_roofline_cap_cannot_authorize_a_run(self):
        task = f"ak-loop:ggml-cpu.c:akm-fuse-x"
        caps = {task: {"upper_cap": 3.0, "cap_provenance":
                       {"kind": "workload_roofline", "source": "paper estimate"}}}
        out = keep_lesson.submit_keep(self.journal, _kept(), caps=caps)
        self.assertEqual(out["decision"], "UNKNOWN")
        self.assertEqual(out["reason"], "cap_missing")

    def test_producer_failure_never_raises_into_the_loop(self):
        def boom(*_a, **_k):
            raise RuntimeError("native down")
        out = keep_lesson.submit_keep(self.journal, _kept(), producer=boom)
        self.assertFalse(out["recorded"])
        self.assertTrue(out["reason"].startswith("producer_refused"))


if __name__ == "__main__":
    unittest.main()
