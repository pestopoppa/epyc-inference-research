from __future__ import annotations

import unittest

from . import baseline_honesty as B


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
    return B.BaselineObservation(
        provider=provider, surface=measured_surface or surface(), metric=metric,
        metric_id=metric_id, evidence_ref=f"receipt.json#{provider}")


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


if __name__ == "__main__":
    unittest.main()
