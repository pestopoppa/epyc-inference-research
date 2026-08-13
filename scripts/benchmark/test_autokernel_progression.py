import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import autokernel_progression as progression


class ProgressionTest(unittest.TestCase):
    def test_projects_screen_and_preflight_without_promotion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "screens" / "cpu" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({
                "schema": "epyc.autokernel.campaign_result.v1", "screening_only": True,
                "non_promotable": True, "ok": True, "state": "decided", "error": None,
                "campaign_id": "cpu-s1", "candidate_id": "c1",
                "screening_report": {"candidate_invocations": 3,
                    "nomination": "top_k_candidate_only_not_a_keep",
                    "median_relative": .12, "sole_intended_factor": {
                        "name": "GGML_IQK", "anchor": "0", "candidate": "1"}},
                "spec": {"n_prompt": 512, "metric": "prefill_tokens_per_s",
                         "created_at": "2026-08-13T00:00:00Z"}}))
            pf = root / "screens" / "gpu.preflight.json"
            pf.write_text(json.dumps({"campaign_id": "gpu-s1", "inference_executed": False,
                "sole_factor": {"name": "ROCWMMA", "anchor": "OFF", "candidate": "ON"}}))
            doc = progression.build_progression(root)
            self.assertEqual(doc["funnel"], {"candidate": 1, "strict_keep": 0,
                                              "champion": 0, "promotable": 0})
            self.assertFalse(doc["promotion_claim"])
            self.assertEqual(doc["candidates"][0]["effect_fraction"], .12)
            self.assertEqual(doc["unexplored"][0]["state"], "preflight_ready")

    def test_invalid_screen_is_not_projected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "screens" / "bad" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"schema": "epyc.autokernel.campaign_result.v1",
                                          "screening_only": True, "ok": False,
                                          "state": "error", "non_promotable": True}))
            self.assertEqual(progression.build_progression(root)["candidates"], [])

    def test_noisy_gpu_overlap_is_inconclusive_and_dedupes_factor_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "screens" / "rocwmma-terminal" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({
                "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
                "campaign_id": "rocwmma-terminal", "non_promotable": True,
                "promotion_claim": False, "ok": True, "state": "decided",
                "candidate_invocations": 3, "anchor_invocations": 3,
                "hip_residency_proved": True, "median_relative": -.21,
                "relative_effects": [-.25, .15, -.21],
                "cpu_overlap_policy": "allowed_discovery_noise",
                "sole_factor": {"name": "GGML_HIP_ROCWMMA_FATTN",
                                "anchor": "OFF", "candidate": "ON"},
            }))
            # The campaign name differs, but the terminal receipt already covers
            # this exact factor.  It must not remain in the opportunity queue.
            preflight = root / "screens" / "rocwmma-old-name.preflight.json"
            preflight.write_text(json.dumps({
                "campaign_id": "ak-gpu-rocwmma-screen-old-name",
                "inference_executed": False,
                "sole_factor": {"name": "GGML_HIP_ROCWMMA_FATTN",
                                "anchor": "OFF", "candidate": "ON"},
            }))

            doc = progression.build_progression(root)
            candidate = doc["candidates"][0]
            self.assertEqual(candidate["stage"], "inconclusive")
            self.assertTrue(candidate["noise"]["sign_conflict"])
            self.assertAlmostEqual(candidate["noise"]["effect_spread_fraction"], .40)
            self.assertIn("CPU-overlap discovery noise", candidate["confidence"])
            self.assertEqual(doc["funnel"]["candidate"], 1)
            self.assertEqual(doc["strategy"]["pursued"], [])
            self.assertEqual(doc["strategy"]["abandoned"], [candidate])
            self.assertEqual(doc["unexplored"], [])
            self.assertFalse(doc["promotion_claim"])

    def test_five_call_gpu_retest_supersedes_three_call_screen(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            screens = root / "screens"
            for campaign, calls, effect, effects in (
                    ("poll-s1", 3, .09, [.08, .09, .10]),
                    ("poll-s2", 5, -.01, [-.02, .40, -.01, .35, -.15])):
                result = screens / campaign / "result.json"
                result.parent.mkdir(parents=True)
                result.write_text(json.dumps({
                    "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
                    "campaign_id": campaign, "non_promotable": True,
                    "promotion_claim": False, "ok": True, "state": "decided",
                    "candidate_invocations": calls, "anchor_invocations": calls,
                    "hip_residency_proved": True, "median_relative": effect,
                    "relative_effects": effects,
                    "cpu_overlap_policy": "allowed_discovery_noise",
                    "sole_factor": {"name": "gpu_poll", "anchor": 50,
                                    "candidate": 0},
                }))
            doc = progression.build_progression(root)
            self.assertEqual(len(doc["candidates"]), 1)
            candidate = doc["candidates"][0]
            self.assertEqual(candidate["effect_fraction"], -.01)
            self.assertEqual(candidate["stage"], "inconclusive")
            self.assertIn("5+5", candidate["confidence"])
            self.assertEqual(candidate["evidence"][0]["campaign_id"], "poll-s2")

    def test_sign_consistent_negative_gpu_screen_is_abandoned(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "screens" / "kv-offload" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({
                "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
                "campaign_id": "kv-offload", "non_promotable": True,
                "promotion_claim": False, "ok": True, "state": "decided",
                "candidate_invocations": 3, "anchor_invocations": 3,
                "hip_residency_proved": True, "median_relative": -.49,
                "relative_effects": [-.51, -.49, -.47],
                "cpu_overlap_policy": "allowed_discovery_noise",
                "sole_factor": {"name": "kv_offload", "anchor": "ON",
                                "candidate": "OFF"},
            }))
            doc = progression.build_progression(root)
            candidate = doc["candidates"][0]
            self.assertEqual(candidate["stage"], "screened_out")
            self.assertEqual(doc["strategy"]["abandoned"], [candidate])
            self.assertEqual(doc["strategy"]["pursued"], [])


if __name__ == "__main__":
    unittest.main()
