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


if __name__ == "__main__":
    unittest.main()
