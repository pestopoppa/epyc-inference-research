#!/usr/bin/env python3
import struct
import sys
import tempfile
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

    def test_align_offset(self) -> None:
        self.assertEqual(contract.align_offset(0, 32), 0)
        self.assertEqual(contract.align_offset(31, 32), 32)
        self.assertEqual(contract.align_offset(32, 32), 32)
        self.assertEqual(contract.align_offset(33, 32), 64)

    def test_q2_layout_contract_passes_standard_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "standard.gguf"
            write_minimal_q2_gguf(path, physical_span=36, include_next_tensor=True)

            result = contract.validate_q2_layout_contract([path])

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["files"][0]["q2_0_tensor_count"], 1)
        self.assertEqual(result["files"][0]["q2_0_mismatch_count"], 0)

    def test_q2_layout_contract_fails_short_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.gguf"
            write_minimal_q2_gguf(path, physical_span=34, include_next_tensor=True)

            result = contract.validate_q2_layout_contract([path])

        self.assertFalse(result["passed"], result)
        self.assertEqual(result["files"][0]["q2_0_mismatch_count"], 1)
        self.assertEqual(result["files"][0]["mismatches"][0]["span_delta"], -2)
        self.assertIn("Q2_0 tensor output.weight is 2 bytes short", result["errors"][0])


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_minimal_q2_gguf(path: Path, *, physical_span: int, include_next_tensor: bool) -> None:
    header = bytearray()
    header += struct.pack("<IIQQ", contract.GGUF_MAGIC, 3, 2 if include_next_tensor else 1, 1)
    header += pack_string("general.alignment")
    header += struct.pack("<I", 4)  # UINT32
    header += struct.pack("<I", 2)
    header += pack_string("output.weight")
    header += struct.pack("<IQQIQ", 2, 64, 2, 42, 0)  # 128 elems -> 2 q2_0 blocks -> 36 bytes.
    if include_next_tensor:
        header += pack_string("output_norm.weight")
        header += struct.pack("<IQIQ", 1, 1, 0, physical_span)  # F32 one scalar.
    data_start = contract.align_offset(len(header), 2)
    payload = bytearray(data_start - len(header))
    payload += b"\0" * physical_span
    if include_next_tensor:
        payload += b"\0" * 4
    path.write_bytes(bytes(header + payload))


if __name__ == "__main__":
    unittest.main()
