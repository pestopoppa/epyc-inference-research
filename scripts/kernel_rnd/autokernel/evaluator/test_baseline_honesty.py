from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from . import baseline_honesty as B


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def surface(**overrides):
    values = dict(
        workload="qwen25-coder-0.5b-prefill", backend="gfx90a",
        model_sha256="a" * 64, quant="Q4_K_M", operation="gemm",
        shape=(4096, 512, 896), dtype="f16", build_sha256="b" * 64,
        factors={"flash_attention": "on", "mmq_mfma": "off",
                 "rocwmma_fattn": "on"},
    )
    values.update(overrides)
    return B.SurfaceKey.create(**values)


def observation(provider, metric, *, measured_surface=None, metric_id="throughput_tps"):
    reference = {
        "schema": "epyc.autokernel.provider_reference.v1",
        "kind": "rocm_library", "source_mode": "source",
        "source_ref": f"https://github.com/ROCm/{provider}",
        "source_commit": "a" * 40, "artifact_sha256": digest("provider-source"),
        "license_check": "MIT, verified",
        "isolation_root": f"/mnt/raid0/llm/autokernel/providers/{provider}",
        "toolchain_manifest_sha256": digest("provider-toolchain"),
        "linkage_manifest_sha256": digest("provider-linkage"),
        "target_backend": "llama_gpu", "evidence_authority": "diagnostic_only",
    }
    return B.BaselineObservation(
        provider=provider, surface=measured_surface or surface(), metric=metric,
        metric_id=metric_id, evidence_ref=f"receipt.json#{provider}",
        provider_manifest=B.ProviderManifest(
            schema=B.PROVIDER_MANIFEST_SCHEMA, provider=provider,
            package_version="rocm-6.2.0", library_binary_sha256=digest("library"),
            reference=reference))


class TestExactSurfaceBaselineSelection(unittest.TestCase):
    def test_stronger_vendor_baseline_is_selected(self):
        target = surface()
        selected = B.select_strongest_prefill_baseline(target, (
            observation("rocblas", 100.0), observation("hipblaslt", 110.0)))
        self.assertEqual(selected.selected.provider, "hipblaslt")
        self.assertEqual(selected.to_dict()["compared_providers"],
                         ["hipblaslt", "rocblas"])

    def test_lower_is_better_is_supported_explicitly(self):
        selected = B.select_strongest_prefill_baseline(surface(), (
            observation("rocblas", 8.0), observation("hipblaslt", 7.0)),
            metric_direction="lower_better")
        self.assertEqual(selected.selected.provider, "hipblaslt")

    def test_missing_weaker_or_stronger_vendor_arm_is_refused(self):
        with self.assertRaisesRegex(ValueError, "requires one rocBLAS and one hipBLASLt"):
            B.select_strongest_prefill_baseline(
                surface(), (observation("rocblas", 100.0),))

    def test_duplicate_provider_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            B.select_strongest_prefill_baseline(surface(), (
                observation("rocblas", 100.0), observation("rocblas", 101.0),
                observation("hipblaslt", 110.0)))

    def test_metric_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "different metrics"):
            B.select_strongest_prefill_baseline(surface(), (
                observation("rocblas", 100.0),
                observation("hipblaslt", 110.0, metric_id="latency_ms")))

    def test_model_transfer_is_refused(self):
        selected = B.select_strongest_prefill_baseline(surface(), (
            observation("rocblas", 100.0), observation("hipblaslt", 110.0)))
        with self.assertRaisesRegex(ValueError, "differs"):
            B.require_candidate_surface(selected, surface(model_sha256="c" * 64))

    def test_quant_transfer_is_refused(self):
        selected = B.select_strongest_prefill_baseline(surface(), (
            observation("rocblas", 100.0), observation("hipblaslt", 110.0)))
        with self.assertRaisesRegex(ValueError, "differs"):
            B.require_candidate_surface(selected, surface(quant="Q8_0"))

    def test_shape_transfer_is_refused(self):
        selected = B.select_strongest_prefill_baseline(surface(), (
            observation("rocblas", 100.0), observation("hipblaslt", 110.0)))
        with self.assertRaisesRegex(ValueError, "differs"):
            B.require_candidate_surface(selected, surface(shape=(4096, 1, 896)))

    def test_implicit_or_auto_factor_is_refused(self):
        with self.assertRaisesRegex(ValueError, "implicit"):
            surface(factors={"flash_attention": "on"})
        with self.assertRaisesRegex(ValueError, "auto"):
            surface(factors={"flash_attention": "auto", "mmq_mfma": "off",
                             "rocwmma_fattn": "on"})

    def test_provider_label_cannot_disagree_with_manifest(self):
        item = observation("rocblas", 1.0)
        with self.assertRaisesRegex(ValueError, "differs"):
            B.BaselineObservation(
                provider="hipblaslt", surface=item.surface, metric=1.0,
                metric_id=item.metric_id, evidence_ref=item.evidence_ref,
                provider_manifest=item.provider_manifest)

    def test_opaque_baseline_is_admissible_but_cannot_claim_candidate_authority(self):
        item = observation("rocblas", 1.0)
        reference = dict(item.provider_manifest.reference)
        reference.update(source_mode="opaque_binary", source_commit=None,
                         evidence_authority="candidate_eligible")
        with self.assertRaisesRegex(ValueError, "opaque binaries are diagnostic_only"):
            B.ProviderManifest(
                schema=B.PROVIDER_MANIFEST_SCHEMA, provider="rocblas",
                package_version="rocm-6.2.0", library_binary_sha256=digest("library"),
                reference=reference)

    def test_provider_manifest_refuses_shared_rocm_prefix(self):
        item = observation("rocblas", 1.0)
        reference = dict(item.provider_manifest.reference)
        reference["isolation_root"] = "/opt/rocm"
        with self.assertRaisesRegex(ValueError, "prohibited shared prefix"):
            B.ProviderManifest(
                schema=B.PROVIDER_MANIFEST_SCHEMA, provider="rocblas",
                package_version="rocm-6.2.0", library_binary_sha256=digest("library"),
                reference=reference)

    def test_provider_manifest_refuses_symlink_into_shared_rocm(self):
        item = observation("rocblas", 1.0)
        with tempfile.TemporaryDirectory(prefix="ak-baseline-provider-") as root:
            link = Path(root, "rocm")
            link.symlink_to("/opt/rocm", target_is_directory=True)
            reference = dict(item.provider_manifest.reference)
            reference["isolation_root"] = str(link)
            with self.assertRaisesRegex(ValueError, "invalid provider isolation"):
                B.ProviderManifest(
                    schema=B.PROVIDER_MANIFEST_SCHEMA, provider="rocblas",
                    package_version="rocm-6.2.0",
                    library_binary_sha256=digest("library"), reference=reference)


if __name__ == "__main__":
    unittest.main()
