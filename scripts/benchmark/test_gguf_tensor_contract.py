#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_tensor_contract as contract


class GgufTensorContractTests(unittest.TestCase):
    def test_layer_index(self) -> None:
        self.assertEqual(contract.layer_index("blk.78.nextn.hnorm.weight"), 78)
        self.assertEqual(contract.layer_index("model.blk.3.attn_q.weight"), 3)
        self.assertIsNone(contract.layer_index("output_norm.weight"))

    def test_layer_allowed(self) -> None:
        self.assertTrue(contract.layer_allowed(78, 78, 79))
        self.assertFalse(contract.layer_allowed(77, 78, 79))
        self.assertFalse(contract.layer_allowed(79, 78, 79))
        self.assertFalse(contract.layer_allowed(None, 78, 79))
        self.assertTrue(contract.layer_allowed(None, None, None))

    def test_matches_any_empty_means_all(self) -> None:
        self.assertTrue(contract.matches_any([], "blk.78.nextn.hnorm.weight"))

    def test_matches_any_regex(self) -> None:
        patterns = contract.compile_regexes([r"nextn", r"blk\.78\."])
        self.assertTrue(contract.matches_any(patterns, "blk.78.attn_norm.weight"))
        self.assertTrue(contract.matches_any(patterns, "blk.12.nextn.hnorm.weight"))
        self.assertFalse(contract.matches_any(patterns, "blk.12.attn_norm.weight"))


if __name__ == "__main__":
    unittest.main()
