"""AK-GFX90A-ASM-SURFACE slice 2: isa_schedule class, v2 asm_artifacts, staging."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from . import hypothesis_portfolio as P
from . import schemas
from . import source_candidate as S
from .evaluator import integrity
from .loop import source_build_preparation as B

PATH = "ggml/src/ggml-hip/fattn.cu"
PATCH = (
    b"diff --git a/ggml/src/ggml-hip/fattn.cu b/ggml/src/ggml-hip/fattn.cu\n"
    b"--- a/ggml/src/ggml-hip/fattn.cu\n+++ b/ggml/src/ggml-hip/fattn.cu\n"
    b"@@ -1,1 +1,1 @@ fattn_dispatch\n-old\n+new\n")
CO = b"\x7fELF fake retargeted code object \x00\x01\x02" * 7
CO_SHA = hashlib.sha256(CO).hexdigest()
PROV = {"retarget_tool_sha": hashlib.sha1(b"tool").hexdigest(),
        "src_co_sha": hashlib.sha256(b"gfx942 source co").hexdigest(), "llvm_mc_ver": "LLVM 19.0.0git"}


def _artifact(**over):
    row = {"name": "ggml/src/ggml-hip/co/fattn_hd128.co", "sha256": CO_SHA,
           "bytes": len(CO), "store_path": S.asm_artifact_store_path(CO_SHA),
           "provenance": dict(PROV)}
    row.update(over)
    return row


def _manifest(artifacts=(), **over):
    kwargs = dict(
        campaign_id="ak-isa", proposal_id="akp-isa", candidate_id="akc-isa",
        source_tree="llama.cpp", production_base_commit="0" * 40,
        instrument_commit="1" * 40, change_class="isa_schedule",
        declared_files=(PATH,), declared_symbols={PATH: ("fattn_dispatch",)},
        mechanism_id="gfx90a.iglp_opt1",
        patch_sha256=hashlib.sha256(PATCH).hexdigest(), patch_bytes=PATCH,
        asm_artifacts=tuple(S.AsmArtifact(**row) for row in artifacts))
    kwargs.update(over)
    return S.SourcePatchManifest(**kwargs)


class IsaScheduleClass(unittest.TestCase):
    def test_closed_sets_and_cheap_suite(self):
        self.assertIn("isa_schedule", schemas.CHANGE_CLASSES)
        self.assertIn("isa_schedule", P.SOURCE_MANIFEST_CHANGE_CLASSES)
        self.assertIn("gfx90a_isa_audit", schemas.CHANGE_CLASS_CHEAP_SUITE["isa_schedule"])
        self.assertEqual(set(schemas.CHANGE_CLASSES), set(schemas.CHANGE_CLASS_CHEAP_SUITE))
        self.assertEqual(len(schemas.GFX90A_ISA_SCHEDULE_MECHANISMS), 6)

    def test_isa_schedule_mechanism_ids_are_closed(self):
        facets = {"mechanism": "gfx90a.s_setprio", "ops": ["FLASH_ATTN_EXT"],
                  "files": [PATH], "symbols": ["fattn_dispatch"], "change_class": "isa_schedule"}
        P.mechanism_fingerprint(facets)
        with self.assertRaises(P.PortfolioError):
            P.mechanism_fingerprint(dict(facets, mechanism="gfx90a.invented"))


class ManifestVersions(unittest.TestCase):
    def test_v1_round_trip_digest_unchanged(self):
        m = _manifest(change_class="arithmetic", mechanism_id="ak.m/v1")
        raw = S.source_patch_manifest_bytes(m)
        payload = json.loads(raw)
        self.assertEqual(payload["schema"], S.SCHEMA_SOURCE_PATCH)
        self.assertNotIn("asm_artifacts", payload)
        # v1 carrier keys are exactly the pre-v2 set (digest formula untouched).
        self.assertEqual(set(payload), set(S._V1_FIELDS))
        again = S.manifest_from_payload(payload)
        self.assertEqual(S.source_patch_manifest_bytes(again), raw)
        self.assertEqual(again.patch_bundle_sha256, hashlib.sha256(raw).hexdigest())

    def test_v1_digest_matches_golden_pre_v2_encoding(self):
        m = _manifest(change_class="arithmetic", mechanism_id="ak.m/v1")
        import base64
        legacy = schemas.canonical_bytes({
            "schema": "epyc.autokernel.source-patch.v1", "campaign_id": m.campaign_id,
            "proposal_id": m.proposal_id, "candidate_id": m.candidate_id,
            "source_tree": "llama.cpp", "production_base_commit": m.production_base_commit,
            "instrument_commit": m.instrument_commit, "change_class": "arithmetic",
            "declared_files": [PATH], "declared_symbols": {PATH: ["fattn_dispatch"]},
            "mechanism_id": "ak.m/v1", "patch_sha256": m.patch_sha256,
            "patch_encoding": "base64", "patch_base64": base64.b64encode(PATCH).decode()})
        self.assertEqual(S.source_patch_manifest_bytes(m), legacy)

    def test_v2_round_trip(self):
        m = _manifest([_artifact()])
        raw = S.source_patch_manifest_bytes(m)
        payload = json.loads(raw)
        self.assertEqual(payload["schema"], S.SCHEMA_SOURCE_PATCH_V2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "m.json")
            path.write_bytes(raw)
            loaded = S.load_source_patch_manifest(path)
        self.assertEqual(S.source_patch_manifest_bytes(loaded), raw)
        self.assertEqual(loaded.asm_artifacts[0].sha256, CO_SHA)
        # build-prep carrier parser accepts v2 too
        self.assertEqual(B._manifest(payload).patch_bundle_sha256, m.patch_bundle_sha256)

    def test_v2_without_artifacts_and_extra_keys_refused(self):
        payload = json.loads(S.source_patch_manifest_bytes(_manifest([_artifact()])))
        with self.assertRaises(S.SourceCandidateError):
            S.manifest_from_payload(dict(payload, asm_artifacts=[]))
        v1 = {k: v for k, v in payload.items() if k != "asm_artifacts"}
        with self.assertRaises(S.SourceCandidateError):
            S.manifest_from_payload(v1)  # v2 schema missing field
        with self.assertRaises(S.SourceCandidateError):
            S.manifest_from_payload(dict(v1, schema=S.SCHEMA_SOURCE_PATCH,
                                         asm_artifacts=payload["asm_artifacts"]))

    def test_binary_patch_member_still_refused(self):
        patch = (b"diff --git a/ggml/src/ggml-hip/x.co b/ggml/src/ggml-hip/x.co\n"
                 b"Binary files a/ggml/src/ggml-hip/x.co and b/ggml/src/ggml-hip/x.co differ\n")
        with self.assertRaises(S.SourceCandidateError):
            _manifest([_artifact()], declared_files=("ggml/src/ggml-hip/x.co",),
                      declared_symbols={"ggml/src/ggml-hip/x.co": (S.FILE_SCOPE,)},
                      patch_bytes=patch, patch_sha256=hashlib.sha256(patch).hexdigest())

    def test_artifact_field_validation(self):
        for bad in (_artifact(name="../evil.co"), _artifact(name="src/llama.cpp"),
                    _artifact(store_path="../x"), _artifact(bytes=0),
                    _artifact(provenance=dict(PROV, retarget_tool_sha="xyz")),
                    _artifact(provenance={"src_co_sha": PROV["src_co_sha"]})):
            with self.subTest(bad=bad), self.assertRaises(S.SourceCandidateError):
                S.AsmArtifact(**bad)
        with self.assertRaises(S.SourceCandidateError):
            _manifest([_artifact(name=PATH)])  # shadowing a patched file


class StoreAndStaging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name, "store")
        (self.store / "sha256").mkdir(parents=True)
        self.tree = Path(self.tmp.name, "tree")
        self.tree.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, data=CO, sha=CO_SHA):
        (self.store / "sha256" / sha).write_bytes(data)

    def test_missing_store_entry_refused(self):
        with self.assertRaisesRegex(S.SourceCandidateError, "missing"):
            S.resolve_asm_artifacts(_manifest([_artifact()]), str(self.store))

    def test_sha_mismatch_refused(self):
        self.put(data=CO[:-1] + b"\xff")
        with self.assertRaisesRegex(S.SourceCandidateError, "sha256 mismatch"):
            S.resolve_asm_artifacts(_manifest([_artifact()]), str(self.store))

    def test_byte_count_mismatch_refused(self):
        self.put()
        with self.assertRaisesRegex(S.SourceCandidateError, "byte count"):
            S.resolve_asm_artifacts(_manifest([_artifact(bytes=len(CO) + 1)]), str(self.store))

    def test_store_symlink_escape_refused(self):
        outside = Path(self.tmp.name, "outside")
        outside.write_bytes(CO)
        (self.store / "sha256" / CO_SHA).symlink_to(outside)
        with self.assertRaisesRegex(S.SourceCandidateError, "escapes"):
            S.resolve_asm_artifacts(_manifest([_artifact()]), str(self.store))

    def test_staging_writes_exact_bytes_and_refuses_overwrite(self):
        self.put()
        m = _manifest([_artifact()])
        receipt = B.stage_asm_artifacts(m, source_root=str(self.tree), store_root=str(self.store))
        staged = self.tree / "ggml/src/ggml-hip/co/fattn_hd128.co"
        self.assertEqual(staged.read_bytes(), CO)
        self.assertEqual(receipt["staged"][0]["sha256"], CO_SHA)
        with self.assertRaises(B.PreparationBindingRefused):
            B.stage_asm_artifacts(m, source_root=str(self.tree), store_root=str(self.store))

    def test_staging_refuses_unverified_store(self):
        with self.assertRaises(B.PreparationBindingRefused):
            B.stage_asm_artifacts(_manifest([_artifact()]), source_root=str(self.tree),
                                  store_root=str(self.store))
        self.assertFalse((self.tree / "ggml").exists())

    def test_staging_refuses_symlinked_dir_escape(self):
        self.put()
        outside = Path(self.tmp.name, "outside_dir")
        outside.mkdir()
        (self.tree / "ggml/src").mkdir(parents=True)
        (self.tree / "ggml/src/ggml-hip").symlink_to(outside)
        with self.assertRaisesRegex(B.PreparationBindingRefused, "escapes"):
            B.stage_asm_artifacts(_manifest([_artifact()]), source_root=str(self.tree),
                                  store_root=str(self.store))

    def test_diff_bound_ignores_artifact_bytes(self):
        bare = _manifest()
        with_art = _manifest([_artifact(bytes=len(CO))])
        count = integrity.parse_unified_diff(bare.patch_text).total_changed
        self.assertEqual(count, integrity.parse_unified_diff(with_art.patch_text).total_changed)
        self.assertEqual(with_art.patch_bytes, PATCH)


if __name__ == "__main__":
    unittest.main()
