from __future__ import annotations

import json
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark import run_autokernel_async_prefetch_replay as replay


class AsyncPrefetchReplayTests(unittest.TestCase):
    def test_balanced_orders_are_deterministic_and_balanced(self):
        first = replay.balanced_orders(20, 17)
        self.assertEqual(first, replay.balanced_orders(20, 17))
        self.assertEqual(first.count(("anchor", "candidate")), 10)
        self.assertEqual(first.count(("candidate", "anchor")), 10)

    def test_balanced_orders_refuse_non_even_design(self):
        for blocks in (0, 1, 3):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                replay.balanced_orders(blocks, 17)

    def test_summarize_requires_positive_everywhere_and_floor(self):
        rows = []
        for block, anchor, candidate in ((0, 100.0, 104.0), (1, 101.0, 105.04)):
            for arm, speed in (("anchor", anchor), ("candidate", candidate)):
                rows.append({"block": block, "arm": arm, "result": {"avg_ts": speed}})
        summary = replay.summarize(rows, contribution_floor=0.03)
        self.assertEqual(summary["verdict"], "REPRODUCED_KNOWN_WIN")
        self.assertTrue(summary["all_blocks_positive"])

    def test_summarize_refuses_a_single_negative_block(self):
        rows = []
        for block, anchor, candidate in ((0, 100.0, 104.0), (1, 100.0, 99.9)):
            for arm, speed in (("anchor", anchor), ("candidate", candidate)):
                rows.append({"block": block, "arm": arm, "result": {"avg_ts": speed}})
        self.assertEqual(
            replay.summarize(rows, contribution_floor=0.0)["verdict"],
            "NOT_REPRODUCED")

    def test_parse_row_binds_mi210_and_raw_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.jsonl"
            path.write_text(json.dumps({
                "backends": "ROCm", "gpu_info": "AMD Instinct MI210",
                "n_prompt": 0, "n_gen": 128, "n_gpu_layers": 99,
                "samples_ns": [1, 2, 3], "avg_ts": 31.0,
            }) + "\n", encoding="utf-8")
            self.assertEqual(replay.parse_row(path, repetitions=3)["avg_ts"], 31.0)

    def test_parse_row_refuses_wrong_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.jsonl"
            path.write_text(json.dumps({
                "backends": "CPU", "gpu_info": "", "n_prompt": 0, "n_gen": 128,
                "n_gpu_layers": 99, "samples_ns": [1], "avg_ts": 1.0,
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "MI210"):
                replay.parse_row(path, repetitions=1)

    def test_gfx90a_duration_floor_is_bound_to_local_evidence(self):
        self.assertEqual(replay.GFX90A_DURATION_FLOOR_NS, 250_090_903)
        self.assertIn("rvp-t0-1-20260811T0906Z", replay.GFX90A_DURATION_FLOOR_REF)

    def test_future_receipt_row_binds_verdict_blocks_and_all_identities(self):
        declaration = {
            "source_root": "/source", "source_branch": "production-consolidated-v9",
            "source_commit": replay.PRODUCTION_COMMIT, "binary": "/bin/llama-bench",
            "binary_sha256": "a" * 64, "linkage_sha256": "b" * 64,
            "model": "/models/q8.gguf", "model_sha256": "c" * 64,
            "blocks": 2, "cell": {"repetitions": 3},
            "orders": [["anchor", "candidate"], ["candidate", "anchor"]],
            "order_seed": 17,
            "candidate_parameter": {"GGML_CUDA_Q8_PREFETCH": "1"},
            "anchor_parameter": {"GGML_CUDA_Q8_PREFETCH": "0"},
        }
        runs = []
        for block, anchor, candidate in ((0, 100.0, 104.0), (1, 101.0, 105.04)):
            for arm, speed in (("anchor", anchor), ("candidate", candidate)):
                runs.append({"block": block, "arm": arm, "result": {"avg_ts": speed}})
        result = replay.summarize(runs, contribution_floor=0.03)
        opened = {"claim_id": "akd-1", "device_id": "mi210_0"}
        released = {**opened, "released_at": "2026-08-12T02:00:00Z"}
        rows, source_sha, claim_sha = replay.belief_measurements(
            declaration=declaration, result=result, opened_claim=opened,
            released_claim=released, producer_sha256="d" * 64)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["native_verdict"], "REPRODUCED_KNOWN_WIN")
        self.assertEqual(row["protocol_id"], replay.SCHEMA)
        self.assertEqual(row["reps"], 2)
        self.assertEqual(row["extra"]["source_identity_sha256"], source_sha)
        self.assertEqual(row["extra"]["claim_identity_sha256"], claim_sha)
        unsigned = copy.deepcopy(row)
        stored = unsigned.pop("measurement_sha256")
        self.assertEqual(stored, replay.canonical_sha256(unsigned))
        self.assertEqual(
            row["extra"]["evidence_sha256"],
            replay.canonical_sha256(row["extra"]["evidence_basis"]))


if __name__ == "__main__":
    unittest.main()
