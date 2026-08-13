"""Adversarial no-inference tests for held-out provider-frame identity."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import least_commitment_heldout as H
from . import schemas
from .test_schemas import _proposal


def _parameter_proposal(provider: dict) -> dict:
    value = _proposal()
    value.update({
        "campaign_id": "ak-provider-frame-test",
        "proposal_id": "akp-provider-frame-test",
        "campaign_kind": "config",
        "change_class": "parameter",
        "provider_reference": copy.deepcopy(provider),
    })
    value["target"]["regimes"] = ["prefill"]
    value["provider_reference"]["target_backend"] = "llama_cpu"
    value["change"]["parameter_surface"] = {
        "candidate": {"ggml_iqk": "1"},
        "anchor": {"ggml_iqk": "0"},
    }
    return value


def _bundle(provider: dict, linkage_text: str) -> tuple[dict, Path]:
    root = Path(tempfile.mkdtemp(prefix="ak-provider-frame-bundle-"))
    linkage = root / "linkage.instrument.txt"
    linkage.write_text(linkage_text, encoding="utf-8")
    bound = copy.deepcopy(provider)
    bound["linkage_manifest_sha256"] = hashlib.sha256(
        linkage.read_bytes()).hexdigest()
    source = {
        "schema": "epyc.autokernel.runtime_source_label.v1",
        "measurement_binary_sha256": bound["artifact_sha256"],
        "measurement_instrument_commit": bound["source_commit"],
        "measurement_linkage_sha256": bound["linkage_manifest_sha256"],
        "measurement_toolchain_manifest_sha256": bound[
            "toolchain_manifest_sha256"],
    }
    source_sha = schemas.content_hash(source)
    (root / "runtime-source-label.json").write_text(
        json.dumps({**source, "source_sha256": source_sha}), encoding="utf-8")
    (root / "campaign_declaration.json").write_text(
        json.dumps({"source_sha256": source_sha}), encoding="utf-8")
    return bound, root


def _factors(provider: dict, calibration: Path) -> dict:
    return {
        "candidate_ref": "registered:ggml_iqk",
        "backend": "llama_cpu",
        "model_sha256": "a" * 64,
        "cpu_list": "0-95",
        "devices": [],
        "device_names": [],
        "device_index": 0,
        "n_gpu_layers": 99,
        "production_commit": "b" * 40,
        "measurement_commit": "c" * 40,
        "provider_reference": copy.deepcopy(provider),
        "calibration": {"evidence_ref": str(calibration)},
    }


class StableProviderFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.library_root = Path(tempfile.mkdtemp(prefix="ak-provider-dso-"))
        self.library = self.library_root / "libcandidate.so"
        self.library.write_bytes(b"same-provider-library\n")
        self.provider = _proposal()["provider_reference"]
        self.provider["target_backend"] = "llama_cpu"

    def _frame(self, provider: dict, bundle: Path) -> dict:
        proposal = _parameter_proposal(provider)
        return H.candidate_frame_from_factors(
            _factors(provider, bundle), proposal)

    def test_same_dso_receipt_with_different_aslr_addresses_is_same_frame(self):
        first, first_root = _bundle(
            self.provider,
            f"libcandidate.so => {self.library} (0x00007f0011111000)\n")
        second, second_root = _bundle(
            self.provider,
            f"libcandidate.so => {self.library} (0x0000745a82222000)\n")
        first_frame = self._frame(first, first_root)
        second_frame = self._frame(second, second_root)
        self.assertNotEqual(
            first["linkage_manifest_sha256"], second["linkage_manifest_sha256"])
        self.assertEqual(first_frame, second_frame)
        self.assertEqual(
            H.candidate_frame_id(first_frame), H.candidate_frame_id(second_frame))
        self.assertNotIn(
            "linkage_manifest_sha256", first_frame["provider_reference"])

    def test_different_dso_path_is_refused_by_frame_identity(self):
        alternate = self.library_root / "alternate" / "libcandidate.so"
        alternate.parent.mkdir()
        alternate.write_bytes(self.library.read_bytes())
        first, first_root = _bundle(
            self.provider,
            f"libcandidate.so => {self.library} (0x00007f0011111000)\n")
        second, second_root = _bundle(
            self.provider,
            f"libcandidate.so => {alternate} (0x0000745a82222000)\n")
        self.assertNotEqual(
            H.candidate_frame_id(self._frame(first, first_root)),
            H.candidate_frame_id(self._frame(second, second_root)))

    def test_different_dso_content_is_refused_by_frame_identity(self):
        first, first_root = _bundle(
            self.provider,
            f"libcandidate.so => {self.library} (0x00007f0011111000)\n")
        first_frame = self._frame(first, first_root)
        self.library.write_bytes(b"different-provider-library\n")
        second, second_root = _bundle(
            self.provider,
            f"libcandidate.so => {self.library} (0x0000745a82222000)\n")
        self.assertNotEqual(
            H.candidate_frame_id(first_frame),
            H.candidate_frame_id(self._frame(second, second_root)))

    def test_raw_linkage_receipt_mismatch_is_refused_before_normalization(self):
        provider, root = _bundle(
            self.provider,
            f"libcandidate.so => {self.library} (0x00007f0011111000)\n")
        provider["linkage_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(
                H.HeldoutProjectionError,
                "provider linkage_manifest_sha256 differs from its exact"):
            self._frame(provider, root)


if __name__ == "__main__":
    unittest.main()
