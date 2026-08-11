from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import autokernel_mmq_wgm_receipt as receipt


SHA = "a" * 64
CAMPAIGN = "inf36-mmq-wgm-successor-r1"


def device_claim(*, campaign_id: str = CAMPAIGN) -> dict:
    shared = {
        "schema": "epyc.autokernel.device_claim_receipt.v1",
        "claim_id": "akd-1234",
        "device_id": "mi210_0",
        "campaign_id": campaign_id,
        "acquired_at": "2026-08-11T18:00:00Z",
    }
    return {
        "opened": {**shared, "released_at": None},
        "released": {**shared, "released_at": "2026-08-11T18:10:00Z"},
    }


def build(**overrides) -> dict:
    values = {
        "campaign_id": CAMPAIGN,
        "started_at": "2026-08-11T18:00:00Z",
        "ended_at": "2026-08-11T18:10:00Z",
        "wgm_arm": 8,
        "category": "CANDIDATE",
        "wall_time_samples_ms": [8.5, 8.7, 8.6],
        "wall_time_reps_basis": "scored: three matched end-to-end repetitions",
        "counter_samples": [
            receipt.CounterSample(70, 30, 1000, 20),
            receipt.CounterSample(60, 40, 1200, 20),
        ],
        "counter_reps_basis": "scored: two all-MMQ counter-only repetitions",
        "surface": {"surface_id": "q4_k_pp32_pl128", "quant": "Q4_K"},
        "source": {
            "repo": "epyc-llama",
            "base_commit": "b" * 40,
            "state": "uncommitted_experimental",
            "source_path": "ggml/src/ggml-cuda/mmq.cu",
            "source_sha256": "c" * 64,
            "source_diff_sha256": "d" * 64,
        },
        "device_claim": device_claim(),
        "wall_time_evidence": {"locator": "wall.jsonl", "sha256": "e" * 64},
        "counter_evidence": {"locator": "counters.csv", "sha256": "f" * 64},
    }
    values.update(overrides)
    return receipt.build_receipt(**values)


class MmqWgmReceiptTest(unittest.TestCase):
    def test_emits_three_per_arm_directional_measurements(self) -> None:
        value = build()
        rows = value["belief_measurements"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["metric_direction"] for row in rows],
            ["lower_better", "higher_better", "lower_better"],
        )
        self.assertEqual([row["reps"] for row in rows], [3, 2, 2])
        self.assertEqual([row["value"] for row in rows], [8.6, 0.65, 1100.0])
        self.assertTrue(all(row["extra"]["wgm_arm"] == 8 for row in rows))
        self.assertEqual(value["authority"], "diagnostic_only")
        self.assertEqual(value["wgm_arm"], {"value": 8, "label": "8"})

    def test_binds_source_producer_claim_evidence_and_digest(self) -> None:
        value = build()
        self.assertEqual(value["source"]["source_sha256"], "c" * 64)
        self.assertEqual(value["producer"]["producer_id"], receipt.PRODUCER_ID)
        self.assertEqual(value["device_claim"]["opened"]["claim_id"], "akd-1234")
        self.assertEqual(value["evidence"]["wall_time"]["sha256"], "e" * 64)
        self.assertEqual(value["receipt_sha256"], receipt.receipt_sha256(value))

    def test_invalid_observations_arm_and_identity_fail_closed(self) -> None:
        with self.assertRaisesRegex(receipt.WgmReceiptError, "wall_time_samples"):
            build(wall_time_samples_ms=[])
        with self.assertRaisesRegex(receipt.WgmReceiptError, "non-negative integer"):
            build(wgm_arm=-1)
        with self.assertRaisesRegex(receipt.WgmReceiptError, "must begin with 'scored:'"):
            build(wall_time_reps_basis="attempted: three repetitions")
        with self.assertRaisesRegex(receipt.WgmReceiptError, "TCC lookup"):
            receipt.CounterSample(0, 0, 1, 1)
        with self.assertRaisesRegex(receipt.WgmReceiptError, "base_commit"):
            build(source={
                "repo": "epyc-llama", "base_commit": "bad", "state": "dirty",
                "source_path": "mmq.cu", "source_sha256": SHA,
            })

    def test_mismatched_or_unreleased_device_claim_fails_closed(self) -> None:
        mismatched = device_claim()
        mismatched["released"]["claim_id"] = "akd-other"
        with self.assertRaisesRegex(receipt.WgmReceiptError, "claim_id changed"):
            build(device_claim=mismatched)
        unreleased = device_claim()
        unreleased["released"]["released_at"] = None
        with self.assertRaisesRegex(receipt.WgmReceiptError, "released_at"):
            build(device_claim=unreleased)

    def test_receipt_hash_is_stable_and_writer_rejects_mutation(self) -> None:
        first = build()
        second = build()
        self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            receipt.write_receipt(path, first)
            self.assertEqual(json.loads(path.read_text()), first)
            self.assertFalse(any(path.parent.glob(".receipt.json.tmp-*")))
            mutated = copy.deepcopy(first)
            mutated["wgm_arm"]["value"] = 16
            with self.assertRaisesRegex(receipt.WgmReceiptError, "does not bind"):
                receipt.write_receipt(path, mutated)


if __name__ == "__main__":
    unittest.main()
