#!/usr/bin/env python3
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


class WeightDeltaGeometryTests(unittest.TestCase):
    def test_q8_dequantize(self) -> None:
        values = list(range(-16, 16))
        result = geometry.q8_dequantize(q8_block(0.5, values))
        np.testing.assert_allclose(result, np.asarray(values, dtype=np.float32) * 0.5, rtol=0, atol=0)

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


if __name__ == "__main__":
    unittest.main()
