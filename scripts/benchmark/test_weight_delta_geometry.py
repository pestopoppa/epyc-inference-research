#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weight_delta_geometry as geometry


def q8_block(scale: float, values: list[int]) -> bytes:
    return np.float16(scale).tobytes() + np.asarray(values, dtype=np.int8).tobytes()


def write_q8(path: Path, blocks: dict[str, bytes]) -> None:
    header = bytearray(struct.pack("<IIQQ", geometry.GGUF_MAGIC, 3, len(blocks), 1))
    key = b"general.alignment"
    header += struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 32)
    offset = 0
    for name, payload in blocks.items():
        raw = name.encode()
        header += struct.pack("<Q", len(raw)) + raw
        header += struct.pack("<IQIQ", 1, len(payload) // geometry.Q8_BLOCK_BYTES * 32, 8, offset)
        offset += len(payload)
    data_start = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (data_start - len(header)) + b"".join(blocks.values()))


def write_gguf(path: Path, tensors: dict[str, tuple[int, int, bytes]]) -> None:
    """Like ``write_q8`` but each tensor supplies its own (type_id, element_count, payload)
    so tests can construct type/shape mismatches that ``write_q8`` cannot express."""
    header = bytearray(struct.pack("<IIQQ", geometry.GGUF_MAGIC, 3, len(tensors), 1))
    key = b"general.alignment"
    header += struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 32)
    offset = 0
    for name, (type_id, elements, payload) in tensors.items():
        raw = name.encode()
        header += struct.pack("<Q", len(raw)) + raw
        header += struct.pack("<IQIQ", 1, elements, type_id, offset)
        offset += len(payload)
    data_start = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (data_start - len(header)) + b"".join(p for _, _, p in tensors.values()))


class WeightDeltaGeometryTests(unittest.TestCase):
    def test_q8_dequantize(self) -> None:
        values = list(range(-16, 16))
        result = geometry.q8_dequantize(q8_block(0.5, values))
        np.testing.assert_allclose(result, np.asarray(values, dtype=np.float32) * 0.5, rtol=0, atol=0)

    def test_q8_dequantize_multiple_blocks_use_their_own_scale(self) -> None:
        """Regression for a broadcasting defect found during the 2026-08-12 execution:
        ``scales[:, None]`` on an already-(N,1) scales array silently built an (N,N,32)
        array instead of (N,32), which only manifests once a chunk holds more than one
        block (the pre-existing tests always used chunk_bytes=34, i.e. exactly one block,
        so it was never exercised). This checks two blocks with distinct scales."""
        block_a = list(range(-16, 16))
        block_b = list(range(16, -16, -1))
        raw = q8_block(0.5, block_a) + q8_block(2.0, block_b)
        result = geometry.q8_dequantize(raw)
        expected = np.concatenate([
            np.asarray(block_a, dtype=np.float32) * 0.5,
            np.asarray(block_b, dtype=np.float32) * 2.0,
        ])
        self.assertEqual(result.shape, (64,))
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)

    def test_execute_reports_known_geometry_and_zero_delta_control(self) -> None:
        values = list(range(-16, 16))
        base = q8_block(1.0, values)
        tc = q8_block(1.0, [value + 1 for value in values])
        ff = q8_block(1.0, [value + 2 for value in values])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_q8(root / "stock.gguf", {"blk.0.weight": base, "blk.1.weight": base})
            write_q8(root / "tc.gguf", {"blk.0.weight": tc, "blk.1.weight": base})
            write_q8(root / "ff.gguf", {"blk.0.weight": ff, "blk.1.weight": ff})
            result = geometry.execute(root / "stock.gguf", root / "tc.gguf", root / "ff.gguf", chunk_bytes=34)
        row = next(item for item in result["tensors"] if item["name"] == "blk.0.weight")
        self.assertAlmostEqual(row["r"], 2.0)
        self.assertAlmostEqual(row["cos"], 1.0)
        self.assertAlmostEqual(row["p"], 2.0)
        self.assertEqual(result["zero_tc_tensor_names"], ["blk.1.weight"])

    def test_plan_is_default_and_does_not_require_input_files(self) -> None:
        self.assertEqual(geometry.main(["--stock", "/missing/stock.gguf"]), 0)

    def test_excluded_tensors_are_listed_by_name_not_silently_dropped(self) -> None:
        """Regression for the defect found during the 2026-08-12 execution: a name absent
        from one file, a type mismatch across the trio, and a shape mismatch must all
        surface as named entries in ``exclusions``, not vanish into an opaque count."""
        base = q8_block(1.0, list(range(-16, 16)))
        f32_payload = np.arange(32, dtype=np.float32).tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_gguf(root / "stock.gguf", {
                "shared.weight": (8, 32, base),
                "type_mismatch.weight": (0, 32, f32_payload),
                "shape_mismatch.weight": (8, 32, base),
            })
            write_gguf(root / "tc.gguf", {
                "shared.weight": (8, 32, base),
                "type_mismatch.weight": (8, 32, base),
                "shape_mismatch.weight": (8, 32, base),
                "tc_only.weight": (8, 32, base),
            })
            write_gguf(root / "ff.gguf", {
                "shared.weight": (8, 32, base),
                "type_mismatch.weight": (8, 32, base),
                "shape_mismatch.weight": (8, 64, base + base),
            })
            result = geometry.execute(root / "stock.gguf", root / "tc.gguf", root / "ff.gguf", chunk_bytes=34)
        self.assertEqual([row["name"] for row in result["tensors"]], ["shared.weight"])
        not_shared = {row["name"]: row for row in result["exclusions"]["not_in_all_three"]}
        self.assertIn("tc_only.weight", not_shared)
        self.assertEqual(not_shared["tc_only.weight"], {"name": "tc_only.weight", "in_stock": False, "in_thinkingcap": True, "in_fable": False})
        type_mismatch_names = {row["name"] for row in result["exclusions"]["type_mismatch"]}
        self.assertEqual(type_mismatch_names, {"type_mismatch.weight"})
        shape_mismatch_names = {row["name"] for row in result["exclusions"]["shape_mismatch"]}
        self.assertEqual(shape_mismatch_names, {"shape_mismatch.weight"})
        self.assertEqual(result["skipped"], {"not_in_all_three": 1, "shape_mismatch": 1, "type_mismatch": 1, "uniform_non_q8_0": 0})

    def test_execute_writes_incremental_jsonl_per_tensor(self) -> None:
        values = list(range(-16, 16))
        base = q8_block(1.0, values)
        tc = q8_block(1.0, [value + 1 for value in values])
        ff = q8_block(1.0, [value + 2 for value in values])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_q8(root / "stock.gguf", {"blk.0.weight": base, "blk.1.weight": base})
            write_q8(root / "tc.gguf", {"blk.0.weight": tc, "blk.1.weight": base})
            write_q8(root / "ff.gguf", {"blk.0.weight": ff, "blk.1.weight": ff})
            jsonl_path = root / "results.jsonl"
            geometry.execute(root / "stock.gguf", root / "tc.gguf", root / "ff.gguf", chunk_bytes=34, jsonl_path=jsonl_path)
            lines = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        self.assertEqual({row["name"] for row in lines}, {"blk.0.weight", "blk.1.weight"})


if __name__ == "__main__":
    unittest.main()
