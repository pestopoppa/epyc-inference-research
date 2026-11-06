"""AK-GFX90A-ASM-SURFACE slice 3: planner prompt + actor-output validation of v2 manifests."""
from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .. import schemas
from .. import source_candidate as S
from . import discovery_controller as D
from . import test_discovery_controller as T

PATH = "ggml/src/ggml-hip/fattn.cu"
PATCH = (
    b"diff --git a/ggml/src/ggml-hip/fattn.cu b/ggml/src/ggml-hip/fattn.cu\n"
    b"--- a/ggml/src/ggml-hip/fattn.cu\n+++ b/ggml/src/ggml-hip/fattn.cu\n"
    b"@@ -1,1 +1,1 @@ fattn_dispatch\n-old\n+new\n")
CO_SHA = hashlib.sha256(b"co bytes").hexdigest()


def _payload(*, v2, change_class="isa_schedule", mechanism="gfx90a.iglp_opt1", **art_over):
    payload = {
        "schema": S.SCHEMA_SOURCE_PATCH_V2 if v2 else S.SCHEMA_SOURCE_PATCH,
        "campaign_id": "ak-isa", "proposal_id": "akp-isa", "candidate_id": "akc-isa",
        "source_tree": "llama.cpp", "production_base_commit": "0" * 40,
        "instrument_commit": "1" * 40, "change_class": change_class,
        "declared_files": [PATH], "declared_symbols": {PATH: ["fattn_dispatch"]},
        "mechanism_id": mechanism, "patch_sha256": hashlib.sha256(PATCH).hexdigest(),
        "patch_encoding": "base64", "patch_base64": base64.b64encode(PATCH).decode()}
    if v2:
        row = {"name": "ggml/src/ggml-hip/co/fattn_hd128.co", "sha256": CO_SHA, "bytes": 8,
               "store_path": S.asm_artifact_store_path(CO_SHA),
               "provenance": {"retarget_tool_sha": S.retarget_tool_blob_sha(D._REPO_ROOT),
                              "src_co_sha": hashlib.sha256(b"src").hexdigest(),
                              "llvm_mc_ver": "LLVM 19"}}
        row.update(art_over)
        payload["asm_artifacts"] = [row]
    return payload


def _load(payload):
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "source-patch.json"
        path.write_text(json.dumps(payload))
        return D._load_planner_manifest(path)


class PlannerManifestValidation(unittest.TestCase):
    def test_v1_unchanged(self):
        m = _load(_payload(v2=False, change_class="arithmetic", mechanism="ak.m/v1"))
        self.assertEqual(m.asm_artifacts, ())

    def test_v1_isa_schedule_source_level_accepted(self):
        self.assertEqual(_load(_payload(v2=False)).change_class, "isa_schedule")

    def test_v2_accepted(self):
        m = _load(_payload(v2=True))
        self.assertEqual(m.asm_artifacts[0].sha256, CO_SHA)

    def test_inline_bytes_refused_retryably(self):
        for key in ("data_base64", "bytes_base64", "inline_hex"):
            payload = _payload(v2=True)
            payload["asm_artifacts"][0][key] = "AAAA"
            with self.assertRaises(D.PlannerOutputRefusal):
                _load(payload)

    def test_unknown_mechanism_refused(self):
        with self.assertRaises(D.PlannerOutputRefusal):
            _load(_payload(v2=True, mechanism="gfx90a.invented"))
        with self.assertRaises(D.PlannerOutputRefusal):
            _load(_payload(v2=False, mechanism="gfx90a.invented"))

    def test_asm_artifacts_need_isa_schedule_class(self):
        with self.assertRaises(D.PlannerOutputRefusal):
            _load(_payload(v2=True, change_class="arithmetic", mechanism="ak.m/v1"))

    def test_retarget_tool_sha_mismatch_refused(self):
        bad = {"retarget_tool_sha": "a" * 40, "src_co_sha": hashlib.sha256(b"s").hexdigest(),
               "llvm_mc_ver": "LLVM 19"}
        with self.assertRaises(D.PlannerOutputRefusal):
            _load(_payload(v2=True, provenance=bad))


class PlannerPrompt(unittest.TestCase):
    def test_prompt_states_isa_schedule_and_v2_rules(self):
        case = T.Tests("test_planner_prompt_has_exact_schemas_source_paths_and_structural_example")
        captured = {}
        package = case.source_package()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            wrapper = root / "codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
            planner = D.CodexPlanner(wrapper=wrapper, environment={"PATH": "/usr/bin"},
                                     reviewed_sources=package)
            assignment = D.AuthoringAssignment("ak-test", "akp-test", "akc-test", "0" * 40, "1" * 40)

            def actor(**kwargs):
                captured.update(json.loads(kwargs["prompt"]))
                return SimpleNamespace(returncode=1, stderr="stop")
            with patch.object(D.codex_container_actor, "runtime_identity", return_value=T.RUNTIME), \
                 patch.object(D.codex_container_actor, "run_actor", side_effect=actor):
                with self.assertRaises(D.DiscoveryControllerError):
                    planner.plan(context={"authoring_assignment": assignment.to_dict(),
                                          "planner_context": {"reviewed_source_package_sha256":
                                                              package.package_sha256}}, workspace=root)
        isa = captured["authoring_contract"]["isa_schedule_contract"]
        self.assertEqual(set(isa["mechanism_ids"]), set(schemas.GFX90A_ISA_SCHEDULE_MECHANISMS))
        self.assertEqual(len(isa["mechanism_ids"]), 6)
        self.assertEqual(isa["v2_schema"]["schema"], S.SCHEMA_SOURCE_PATCH_V2)
        self.assertIn("NEVER emit raw bytes", isa["v2_schema"]["store_rule"])
        self.assertIn("SAME op", isa["sub_tier_b_hand_asm"])
        # v1 contract keys untouched
        self.assertEqual(captured["authoring_contract"]["source_manifest_schema"]["constants"]["schema"],
                         S.SCHEMA_SOURCE_PATCH)


if __name__ == "__main__":
    unittest.main()
