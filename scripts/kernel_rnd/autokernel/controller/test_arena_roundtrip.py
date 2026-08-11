from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import arena_roundtrip as R


SHA = "a" * 64


def receipt(**overrides):
    values = {
        "campaign_id": "inf03-arena-a-b-001",
        "task_id": "instruction2triton/add",
        "controller_id": "geak_v1",
        "started_at": "2026-08-11T12:00:00Z",
        "ended_at": "2026-08-11T12:01:00Z",
        "correctness": R.ScoredCount(3, 4, "scored correctness cases"),
        "timing_validity": R.ScoredCount(98, 100, "scored timing repetitions"),
        "preflight_locator": "/evidence/preflight.json",
        "preflight_sha256": SHA,
        "source": {"arena_commit": "b" * 40},
        "artifacts": {"correctness.json": "c" * 64, "timing.json": "d" * 64},
    }
    values.update(overrides)
    return R.build_receipt(**values)


class ArenaRoundTripReceiptTest(unittest.TestCase):
    def test_emits_separate_directional_correctness_and_timing_rows(self):
        value = receipt()
        self.assertEqual(value["status"], "pass")
        self.assertEqual(
            [row["measurement_id"] for row in value["belief_measurements"]],
            ["arena_correctness_pass_rate", "arena_timing_harness_validity_rate"],
        )
        self.assertEqual(
            [row["value"] for row in value["belief_measurements"]], [0.75, 0.98]
        )
        self.assertEqual(
            [row["reps"] for row in value["belief_measurements"]], [4, 100]
        )
        self.assertTrue(all(
            row["metric_direction"] == "higher_better"
            for row in value["belief_measurements"]
        ))
        self.assertTrue(all(
            row["category"] == "CANDIDATE"
            for row in value["belief_measurements"]
        ))

    def test_preflight_is_dependency_evidence_and_never_a_measurement(self):
        value = receipt()
        preflight = value["dependencies"]["preflight"]
        self.assertEqual(preflight["classification"], "dependency_evidence_only")
        self.assertFalse(preflight["belief_measurement_emitted"])
        rendered = json.dumps(value["belief_measurements"])
        self.assertNotIn("license", rendered)
        self.assertNotIn("preflight", rendered)

    def test_invalid_counts_controller_or_digest_fail_closed(self):
        with self.assertRaisesRegex(R.RoundTripReceiptError, "scored counts"):
            R.ScoredCount(2, 1, "cases")
        with self.assertRaisesRegex(R.RoundTripReceiptError, "registered"):
            receipt(controller_id="unknown")
        with self.assertRaisesRegex(R.RoundTripReceiptError, "SHA-256"):
            receipt(preflight_sha256="not-a-digest")

    def test_receipt_hash_is_stable_and_atomic_writer_round_trips(self):
        first = receipt()
        second = receipt()
        self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            R.write_receipt(path, first)
            self.assertEqual(json.loads(path.read_text()), first)
            self.assertFalse(any(path.parent.glob(".receipt.json.tmp-*")))


if __name__ == "__main__":
    unittest.main()
