from __future__ import annotations

import math
import unittest

from .. import schemas
from . import physical_bounds as P


class PhysicalSpeedOfLightTest(unittest.TestCase):
    @staticmethod
    def envelope(**overrides):
        kwargs = dict(
            shape_id="mul_mat.m1024.n1.k4096.q4_k",
            delivered_unit="output_row",
            flops_per_unit=8_388_608,
            bytes_per_unit=2_097_152,
            peak_compute_flops_s=45.3e12,
            peak_memory_bytes_s=1.638e12,
            measurement_frame_sha256=P.measurement_frame_sha256(
                "test.op.v1", {"shape": "m1024.n1.k4096.q4_k"}),
            work_derivation_ref="shape-manifest:sha256:abc",
            hardware_peak_ref="mi210-datasheet:2026-08-10",
        )
        kwargs.update(overrides)
        return P.PhysicalEnvelope(**kwargs)

    def test_time_max_is_exactly_throughput_min(self):
        bound = self.envelope()
        self.assertEqual(bound.time_floor_s, max(
            bound.compute_time_floor_s, bound.memory_time_floor_s))
        self.assertTrue(math.isclose(
            bound.throughput_ceiling_units_s, 1.0 / bound.time_floor_s,
            rel_tol=1e-15))
        self.assertEqual(bound.throughput_ceiling_units_s, min(
            bound.compute_ceiling_units_s, bound.memory_ceiling_units_s))

    def test_memory_bound_and_compute_bound_shapes_pick_the_right_limit(self):
        self.assertEqual(self.envelope().limiting_resource, "memory")
        compute = self.envelope(bytes_per_unit=1, flops_per_unit=1e12)
        self.assertEqual(compute.limiting_resource, "compute")

    def test_a_sample_over_the_ceiling_fails_before_ranking(self):
        bound = self.envelope()
        result = bound.check_throughput((0.5 * bound.throughput_ceiling_units_s,
                                         1.01 * bound.throughput_ceiling_units_s))
        self.assertEqual(result.outcome, schemas.FAIL)
        self.assertIn("wrong work, wrong unit, or wrong timer", " ".join(result.reasons))

    def test_a_sample_at_the_ceiling_passes(self):
        bound = self.envelope()
        self.assertEqual(bound.check_throughput((bound.throughput_ceiling_units_s,)).outcome,
                         schemas.PASS)

    def test_empty_or_nonfinite_samples_never_pass(self):
        bound = self.envelope()
        self.assertEqual(bound.check_throughput(()).outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(bound.check_throughput((float("nan"),)).outcome,
                         schemas.COULD_NOT_CHECK)

    def test_work_and_peak_receipts_are_required(self):
        with self.assertRaisesRegex(P.PhysicalBoundError, "work_derivation_ref"):
            self.envelope(work_derivation_ref="")

    def test_candidate_cannot_select_an_unknown_bound_version(self):
        with self.assertRaisesRegex(P.PhysicalBoundError, "bound_id"):
            self.envelope(bound_id="candidate-bound/v99")

    def test_serialized_receipt_round_trips_without_trusting_derived_fields(self):
        original = self.envelope()
        restored = P.PhysicalEnvelope.from_mapping(original.to_dict())
        self.assertEqual(restored, original)

    def test_unknown_or_tampered_derived_fields_are_refused(self):
        payload = self.envelope().to_dict()
        payload["candidate_margin_percent"] = 10
        with self.assertRaisesRegex(P.PhysicalBoundError, "unknown fields"):
            P.PhysicalEnvelope.from_mapping(payload)
        payload = self.envelope().to_dict()
        payload["throughput_ceiling_units_s"] *= 2
        with self.assertRaisesRegex(P.PhysicalBoundError, "derived field"):
            P.PhysicalEnvelope.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
