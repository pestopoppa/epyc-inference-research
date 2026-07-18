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

    def test_unique_patterns_preserves_order(self) -> None:
        self.assertEqual(contract.unique_patterns(["a", "b", "a", "c"]), ["a", "b", "c"])

    def test_validate_glm_nextn_contract_passes(self) -> None:
        fake_contract = {
            "metadata": {
                "model.gguf": {
                    "general.architecture": "glm-dsa",
                    "glm-dsa.block_count": 79,
                    "glm-dsa.nextn_predict_layers": 1,
                    "glm-dsa.embedding_length": 6144,
                }
            },
            "tensors": [
                {"name": "blk.78.attn_norm.weight", "shape": [6144]},
                {"name": "blk.78.ffn_norm.weight", "shape": [6144]},
                {"name": "blk.78.indexer.proj.weight", "shape": [6144, 32]},
                {"name": "blk.78.nextn.eh_proj.weight", "shape": [12288, 6144]},
                {"name": "blk.78.nextn.enorm.weight", "shape": [6144]},
                {"name": "blk.78.nextn.hnorm.weight", "shape": [6144]},
                {"name": "blk.78.nextn.shared_head_norm.weight", "shape": [6144]},
            ],
        }

        result = contract.validate_glm_nextn_contract(fake_contract)

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["facts"]["tail_layers"], [78])
        self.assertEqual(result["facts"]["tail_group_counts"]["nextn"], 4)

    def test_validate_glm_nextn_contract_fails_closed(self) -> None:
        fake_contract = {
            "metadata": {
                "model.gguf": {
                    "general.architecture": "glm-dsa",
                    "glm-dsa.block_count": 79,
                    "glm-dsa.nextn_predict_layers": 1,
                    "glm-dsa.embedding_length": 6144,
                }
            },
            "tensors": [
                {"name": "blk.78.attn_norm.weight", "shape": [6144]},
                {"name": "blk.78.ffn_norm.weight", "shape": [6144]},
                {"name": "blk.78.indexer.proj.weight", "shape": [6144, 32]},
                {"name": "blk.78.nextn.enorm.weight", "shape": [6144]},
                {"name": "blk.78.nextn.hnorm.weight", "shape": [6144]},
            ],
        }

        result = contract.validate_glm_nextn_contract(fake_contract)

        self.assertFalse(result["passed"], result)
        self.assertIn("missing required NextN tensor: blk.78.nextn.eh_proj.weight", result["errors"])

    def test_validate_glm_nextn_contract_handles_list_metadata_values(self) -> None:
        fake_contract = {
            "metadata": {
                "model.gguf": {
                    "general.architecture": ["glm-dsa"],
                    "glm-dsa.block_count": [79],
                    "glm-dsa.nextn_predict_layers": [1],
                    "glm-dsa.embedding_length": [6144],
                }
            },
            "tensors": [
                {"name": "blk.78.attn_norm.weight", "shape": [6144]},
                {"name": "blk.78.ffn_norm.weight", "shape": [6144]},
                {"name": "blk.78.indexer.proj.weight", "shape": [6144, 32]},
                {"name": "blk.78.nextn.eh_proj.weight", "shape": [12288, 6144]},
                {"name": "blk.78.nextn.enorm.weight", "shape": [6144]},
                {"name": "blk.78.nextn.hnorm.weight", "shape": [6144]},
            ],
        }

        result = contract.validate_glm_nextn_contract(fake_contract)

        self.assertTrue(result["passed"], result)
        self.assertIn("optional NextN tensor absent: blk.78.nextn.shared_head_norm.weight", result["warnings"])


if __name__ == "__main__":
    unittest.main()
