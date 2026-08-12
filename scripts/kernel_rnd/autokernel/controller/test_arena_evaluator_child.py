#!/usr/bin/env python3
"""Unit tests for strict broker-to-evaluator serialization; no GPU work."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from . import arena_evaluator_child as C


class ArenaEvaluatorChildTest(unittest.TestCase):
    def test_baseline_roundtrip_uses_only_declared_strict_json_fields(self):
        cases = [SimpleNamespace(
            test_case_id="shape-1", shape=[1, 2], execution_time_ms=1.25,
            metadata={"dtype": "float16", "nested": [True, None]})]
        document = C.serialize_baseline_cases(cases)

        class PinnedCase:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        restored = C.reconstruct_baseline_cases(document, PinnedCase)
        self.assertEqual(restored[0].test_case_id, "shape-1")
        self.assertEqual(restored[0].execution_time_ms, 1.25)
        self.assertEqual(document["receipt_sha256"], C.canonical_sha256({
            key: value for key, value in document.items()
            if key != "receipt_sha256"}))

    def test_baseline_rejects_nonfinite_or_non_json_metadata(self):
        with self.assertRaisesRegex(C.EvaluatorChildError, "timing"):
            C.serialize_baseline_cases([SimpleNamespace(
                test_case_id="bad", shape=[], execution_time_ms=float("nan"),
                metadata={})])
        with self.assertRaisesRegex(C.EvaluatorChildError, "strict JSON"):
            C.serialize_baseline_cases([SimpleNamespace(
                test_case_id="bad", shape=[], execution_time_ms=1.0,
                metadata={"opaque": object()})])

    def test_baseline_tamper_is_rejected_before_vendor_reconstruction(self):
        document = C.serialize_baseline_cases([SimpleNamespace(
            test_case_id="case", shape=None, execution_time_ms=1.0,
            metadata=None)])
        document["cases"][0]["execution_time_ms"] = 2.0
        with self.assertRaisesRegex(C.EvaluatorChildError, "self-hash"):
            C.reconstruct_baseline_cases(document, SimpleNamespace)


if __name__ == "__main__":
    unittest.main()
