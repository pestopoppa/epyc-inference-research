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


if __name__ == "__main__":
    unittest.main()
