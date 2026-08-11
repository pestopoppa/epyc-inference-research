#!/usr/bin/env python3
import copy
import unittest

from . import substrate as F


class SubstrateFactsTest(unittest.TestCase):
    def test_checked_in_facts_validate_and_keep_bases_separate(self):
        facts = F.load().document
        self.assertEqual(facts["numa_node"], 1)
        self.assertEqual(facts["facts"]["memory_bandwidth_gbps"]["measured"], 1433.3)
        self.assertEqual(facts["facts"]["memory_bandwidth_gbps"]["datasheet"], 1638.0)

    def test_measured_ridge_is_rederived_not_trusted(self):
        doc = copy.deepcopy(F.load().document)
        doc["derived"]["ridge_flop_per_byte_measured_basis"] = 99.0
        with self.assertRaisesRegex(F.SubstrateFactError, "does not rederive"):
            F.SubstrateFacts(doc)

    def test_every_measured_fact_needs_a_receipt(self):
        doc = copy.deepcopy(F.load().document)
        doc["facts"]["pcie_gbps"]["measured_receipt"] = ""
        with self.assertRaisesRegex(F.SubstrateFactError, "receipt"):
            F.SubstrateFacts(doc)


class PerQuantRooflineTest(unittest.TestCase):
    @staticmethod
    def observation(quantization="bf16", *, measured_tps=800.0):
        return F.QuantRooflineObservation(
            hardware="AMD Instinct MI210",
            quantization=quantization,
            workload_regime="batch1_single_sequence_decode",
            measured_tps=measured_tps,
            bytes_per_token=1_000_000_000,
            measured_bandwidth_gbps=1433.3,
            datasheet_bandwidth_gbps=1638.0,
            measurement_receipt="data/campaign/receipt.json",
        )

    def test_local_and_cross_vendor_bases_are_separate(self):
        observation = self.observation()
        cell, = F.compare_per_quant((observation,))
        self.assertAlmostEqual(observation.achievable_utilization, 800 / 1433.3)
        self.assertAlmostEqual(observation.spec_utilization, 800 / 1638.0)
        self.assertEqual(cell.local_headroom_basis, F.ACHIEVABLE_BASIS)
        self.assertEqual(cell.cross_vendor_basis, F.SPEC_BASIS)
        self.assertEqual(cell.target_utilization_spec_basis, 0.78)
        self.assertAlmostEqual(cell.target_tps_on_local_spec_roof, 0.78 * 1638)
        self.assertEqual(cell.role, F.ROOFLINE_ROLE)

    def test_q4_relative_speedup_is_not_fabricated_into_an_absolute_anchor(self):
        cell, = F.compare_per_quant((self.observation("q4_k"),))
        self.assertEqual(cell.anchor_status, "COULD_NOT_CHECK")
        self.assertIsNone(cell.target_utilization_spec_basis)
        self.assertIn("3.87x", cell.anchor_gap)
        self.assertIn("not GGUF Q4_K", cell.anchor_gap)

    def test_bf16_anchor_cannot_be_borrowed_by_q8(self):
        cell, = F.compare_per_quant((self.observation("q8_0"),))
        self.assertEqual(cell.anchor_status, "COULD_NOT_CHECK")
        self.assertIsNone(cell.cuda_anchor)

    def test_pooled_quant_is_refused(self):
        with self.assertRaisesRegex(F.SubstrateFactError, "never a pooled quant"):
            self.observation("mixed")

    def test_anchor_requires_exact_quant_and_workload(self):
        with self.assertRaisesRegex(F.SubstrateFactError, "quantization does not exactly match"):
            F.QuantRooflineComparison(
                observation=self.observation("q8_0"),
                cuda_anchor=F.BF16_CUDA_ANCHOR,
                anchor_gap=None,
            )

    def test_cross_vendor_mixed_basis_is_refused(self):
        with self.assertRaisesRegex(F.SubstrateFactError, "must be spec-basis"):
            F.QuantRooflineComparison(
                observation=self.observation(),
                cuda_anchor=F.BF16_CUDA_ANCHOR,
                anchor_gap=None,
                cross_vendor_basis=F.ACHIEVABLE_BASIS,
            )

    def test_observation_above_roof_exposes_bad_denominator(self):
        with self.assertRaisesRegex(F.SubstrateFactError, "exceeds the achievable"):
            self.observation(measured_tps=1500.0)


if __name__ == "__main__":
    unittest.main()
