"""No-hardware tests for prospective RB-lineage write-side observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import archive, lineage_beliefs, loop


EPOCH = "c" * 64
PARENT = "a" * 40
PRODUCER = "b" * 40
FILE_SHA = "d" * 64


class LineageBeliefReceipt(unittest.TestCase):
    def test_committed_journal_row_and_run_artifact_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            outcome = loop.Outcome("measured_null", spawn_parent=PARENT,
                                   branch_id="detached:lane0", width=2, depth=3)
            journal_receipts = []
            self.assertTrue(archive.record(
                store, outcome.to_attempt(), epoch=EPOCH,
                recorded_at="2026-09-17T12:00:00Z", campaign_id="ak-loop",
                journal_receipt_out=journal_receipts))
            self.assertEqual(len(journal_receipts), 1)
            from ..controller import experiments
            with experiments.ExperimentStore(store, read_only=True) as reader:
                persisted = reader._connection.execute(
                    "SELECT payload FROM experiments WHERE attempt_id=?",
                    (journal_receipts[0]["attempt_id"],)).fetchone()
            self.assertEqual(journal_receipts[0]["payload_sha256"],
                             hashlib.sha256(persisted["payload"].encode()).hexdigest())
            outcome.journal_receipt = journal_receipts[0]
            run_artifact = Path(tmp) / "loop-run.json"
            run_artifact.write_text('{"schema":"epyc.autokernel.loop_run.v1"}')
            path = lineage_beliefs.publish(
                store, [outcome], epoch=EPOCH, anchor_commit=PARENT,
                producer_commit=PRODUCER, producer_file_sha256=FILE_SHA,
                run_artifact=run_artifact)
            receipt = json.loads(path.read_text())
            self.assertEqual(
                receipt["receipt_sha256"],
                hashlib.sha256(lineage_beliefs._canonical({
                    key: value for key, value in receipt.items()
                    if key != "receipt_sha256"
                })).hexdigest())
            measurement = receipt["belief_measurements"][0]
            self.assertEqual(measurement["value"], 1.0)
            self.assertEqual(measurement["protocol_id"], "")
            self.assertEqual(measurement["attestation_sha256"],
                             hashlib.sha256(run_artifact.read_bytes()).hexdigest())
            self.assertEqual(receipt["journal"]["rows"][0]["journal"], journal_receipts[0])
            self.assertEqual(receipt["anchor_commit"], PARENT)
            self.assertEqual(receipt["producer_source"]["research_commit"], PRODUCER)
            self.assertEqual(lineage_beliefs.publish(
                store, [outcome], epoch=EPOCH, anchor_commit=PARENT,
                producer_commit=PRODUCER, producer_file_sha256=FILE_SHA,
                run_artifact=run_artifact), path)

    def test_missing_parent_or_duplicate_journal_receipt_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            outcome = loop.Outcome("lane_error", branch_id="detached:lane0",
                                   width=1, depth=1)
            receipts = []
            attempt = outcome.to_attempt()
            self.assertTrue(archive.record(
                store, attempt, epoch=EPOCH,
                recorded_at="2026-09-17T12:00:00Z", campaign_id="ak-loop",
                journal_receipt_out=receipts))
            duplicate_receipts = []
            self.assertFalse(archive.record(
                store, attempt, epoch=EPOCH,
                recorded_at="2026-09-17T12:00:00Z", campaign_id="ak-loop",
                journal_receipt_out=duplicate_receipts))
            self.assertEqual(duplicate_receipts, [])
            outcome.journal_receipt = receipts[0]
            receipt = lineage_beliefs.build(
                [outcome], epoch=EPOCH, anchor_commit=PARENT,
                producer_commit=PRODUCER, producer_file_sha256=FILE_SHA,
                journal_root=store)
            self.assertEqual(receipt["belief_measurements"][0]["value"], 0.0)
            self.assertIsNone(receipt["loop_run"])
            self.assertEqual(receipt["belief_measurements"][0]["attestation_sha256"], "")
            self.assertFalse(receipt["journal"]["rows"][0]["capture_complete"])

    def test_zero_outcomes_emit_no_zero_rate_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = lineage_beliefs.build(
                [], epoch=EPOCH, anchor_commit=PARENT,
                producer_commit=PRODUCER, producer_file_sha256=FILE_SHA,
                journal_root=Path(tmp))
            self.assertEqual(receipt["belief_measurements"], [])

    def test_receipt_for_different_lineage_cannot_complete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            outcome = loop.Outcome("measured_null", spawn_parent=PARENT,
                                   branch_id="detached:lane0", width=2, depth=3)
            receipts = []
            self.assertTrue(archive.record(
                store, outcome.to_attempt(), epoch=EPOCH,
                recorded_at="2026-09-17T12:00:00Z", campaign_id="ak-loop",
                journal_receipt_out=receipts))
            outcome.journal_receipt = {**receipts[0], "depth": 4}
            receipt = lineage_beliefs.build(
                [outcome], epoch=EPOCH, anchor_commit=PARENT,
                producer_commit=PRODUCER, producer_file_sha256=FILE_SHA,
                journal_root=store)
            self.assertEqual(receipt["belief_measurements"][0]["value"], 0.0)
            self.assertIsNone(receipt["journal"]["rows"][0]["journal"])


if __name__ == "__main__":
    unittest.main()
