from __future__ import annotations

import json
import unittest

from . import datatype_targets as D


class DatatypeTargetsTest(unittest.TestCase):
    def test_fp8_is_software_upcast_target_not_native_capability(self):
        target = D.TARGETS[D.FP8_TARGET_ID]
        self.assertFalse(target.native_gfx90a_mfma)
        self.assertEqual(target.decode_path, "software_decode_and_upcast_to_bf16")
        self.assertIn("bf16_vector_gemv", target.compute_paths)
        self.assertIn("upcast_cost_attribution", target.prerequisites)

    def test_nvfp4_is_mechanically_deferred_behind_fp8(self):
        target = D.TARGETS[D.NVFP4_TARGET_ID]
        self.assertEqual(target.compute_paths, ())
        self.assertIn("deferred", target.state)
        self.assertEqual(
            target.prerequisites, ("fp8_weight_bf16_compute_gfx90a_terminal_result",)
        )

    def test_context_is_non_numeric_and_excludes_cross_vendor_claims(self):
        item = D.target_context_item((D.FP8_TARGET_ID,))
        payload = json.loads(item.content)
        self.assertEqual(payload["authority"], "design_target_only")
        self.assertFalse(payload["hardware_facts"]["native_fp8_mfma"])
        self.assertNotIn("upstream_latency", item.content)
        self.assertNotIn("speedup_pct", item.content)

    def test_bad_or_duplicate_selection_fails_closed(self):
        with self.assertRaisesRegex(D.DatatypeTargetError, "unknown"):
            D.select(("fp3",))
        with self.assertRaisesRegex(D.DatatypeTargetError, "unique"):
            D.select((D.FP8_TARGET_ID, D.FP8_TARGET_ID))


if __name__ == "__main__":
    unittest.main()
