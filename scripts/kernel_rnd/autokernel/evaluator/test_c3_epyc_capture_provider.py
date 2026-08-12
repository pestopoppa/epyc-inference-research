from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.kernel_rnd.autokernel.evaluator import c3_epyc_capture_provider as P


class C3EpycCaptureProviderTest(unittest.TestCase):
    def test_absent_real_model_hook_is_one_typed_external_artifact(self):
        result = P.preflight("epyc.attention.mla_paged_prefill.k228", None)
        self.assertEqual(result["status"], "COULD_NOT_CHECK")
        self.assertIn("real-model provider-hook manifest", result["missing_external_artifact"])
        self.assertIn("reference-only", result["missing_external_artifact"])

    def test_manifest_hash_drift_refuses_before_hook_import(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            provider = root / "provider.json"
            provider.write_text("{}", encoding="utf-8")
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps({
                "schema": P.RECIPE_SCHEMA, "provider_manifest": str(provider),
                "provider_manifest_sha256": "a" * 64}), encoding="utf-8")
            plan = {"recipe_ref": str(recipe),
                    "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(P.ProviderRefusal, "manifest hash"):
                P.load_provider(plan)


if __name__ == "__main__":
    unittest.main()
