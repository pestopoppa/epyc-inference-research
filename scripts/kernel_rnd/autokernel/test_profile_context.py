#!/usr/bin/env python3
"""Tests for the C4 AutoKernel / external-evaluator bridge."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import profile_context as C
from .controller import authoring_contract as A


class ProfileContextTest(unittest.TestCase):
    def payload(self):
        receipt = {"source_commit": "deadbeef", "profile_sha256": "2" * 64}
        return {
            "schema": "epyc.autokernel.c4_profile_report.v1",
            "manifest_sha256": "1" * 64,
            "comparison_id": "c4-q4",
            "stage": "decode",
            "capture_protocol": {
                "mapping": {"receipt": dict(receipt)},
                "formal": {"receipt": dict(receipt)},
            },
            "kernel_table": [{
                "kernel_family": "mul_mat_vec_q", "dispatches": 5,
                "duration_ns": 100, "gpu_time_share": 0.7,
            }, {
                "kernel_family": "quantize_q8_1", "dispatches": 5,
                "duration_ns": 40, "gpu_time_share": 0.3,
            }],
            "overlap_opportunity_table": [{
                "pattern_id": "requant-overlap", "formal_time_share": 0.3,
                "attribution_status": "mapped",
                "source_paths": ["sealed/evaluator/path"],
            }],
            "fuse_pattern_table": [],
            "architecture_shape_table": [{
                "block_id": "decode-layer", "exact_sequence_occurrences": 5,
                "kernel_families": ["quantize_q8_1", "mul_mat_vec_q"],
                "source_paths": ["tests/test-backend-ops.cpp"],
            }],
            "coverage_gaps": [],
        }

    def write(self, payload=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "report.json"
        path.write_text(json.dumps(payload or self.payload(), sort_keys=True),
                        encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_bridge_retains_hash_and_emits_neutral_metrics(self):
        path, digest = self.write()
        context = C.load_profile_context(path, expected_sha256=digest)
        observation = context.evaluator_observation()
        self.assertEqual(observation["authority"], "diagnostic_only")
        self.assertEqual(observation["evidence"]["report_sha256"], digest)
        self.assertEqual(len(observation["metrics"]), 3)

    def test_authoring_context_omits_evaluator_paths_and_is_priced(self):
        path, digest = self.write()
        item = A.c4_profile_context_item(str(path), expected_sha256=digest)
        self.assertNotIn("test-backend-ops", item.content)
        self.assertNotIn("sealed/evaluator/path", item.content)
        priced = A.price_context(
            round_id="r1", budget=A.ContextBudget(2000, 2000, 1), items=(item,))
        self.assertEqual(priced.items, (item,))

    def test_hash_mismatch_and_invalid_share_refuse(self):
        path, _ = self.write()
        with self.assertRaisesRegex(C.ProfileContextError, "hash mismatch"):
            C.load_profile_context(path, expected_sha256="f" * 64)
        payload = self.payload()
        payload["kernel_table"][0]["gpu_time_share"] = 0.8
        path, digest = self.write(payload)
        with self.assertRaisesRegex(C.ProfileContextError, "sum above one"):
            C.load_profile_context(path, expected_sha256=digest)


if __name__ == "__main__":
    unittest.main()
