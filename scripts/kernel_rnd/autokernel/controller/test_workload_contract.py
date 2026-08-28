"""The screening workload must dispatch the kernels production dispatches.

The defect: `Qwen2.5-Coder-0.5B-Q4_K_M.gguf` revalidated cleanly for a month while
being 132x Q5_0 and 12x Q4_K. n_embd=896 is not divisible by the 256-element K-quant
superblock, so llama.cpp fell back silently. The loop optimised `mul_mat_vec_q<Q5_0>`
-- a path production never dispatches -- and its flagship hypothesis was
`akh-v2-q5-type-specific-dequant`.

The filename said Q4_K_M. Only the tensor table said otherwise.
"""
import struct
import unittest
from pathlib import Path

from autokernel.controller import workload_contract as wc


def _write_gguf(path: Path, *, architecture: str, n_embd: int,
                tensors: dict[int, int]) -> Path:
    """A minimal but real GGUF header: metadata KVs then the tensor table."""
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", sum(tensors.values()))
    out += struct.pack("<Q", 2)

    def kv_string(key: str, value: str) -> bytes:
        blob = struct.pack("<Q", len(key)) + key.encode()
        blob += struct.pack("<I", 8)
        blob += struct.pack("<Q", len(value)) + value.encode()
        return blob

    def kv_uint32(key: str, value: int) -> bytes:
        blob = struct.pack("<Q", len(key)) + key.encode()
        blob += struct.pack("<I", 4) + struct.pack("<I", value)
        return blob

    out += kv_string("general.architecture", architecture)
    out += kv_uint32(f"{architecture}.embedding_length", n_embd)

    index = 0
    for type_id, count in tensors.items():
        for _ in range(count):
            name = f"blk.{index}.weight".encode()
            index += 1
            out += struct.pack("<Q", len(name)) + name
            out += struct.pack("<I", 1) + struct.pack("<Q", 4096)
            out += struct.pack("<I", type_id) + struct.pack("<Q", 0)

    path.write_bytes(bytes(out))
    return path


Q4_K, Q6_K, Q5_0, Q8_0, F32 = 12, 14, 6, 8, 0


class Census(unittest.TestCase):

    def test_it_reads_the_tensor_table_not_the_filename(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Named like a K-quant, built like a legacy quant. The real defect.
            path = _write_gguf(Path(tmp) / "Model-Q4_K_M.gguf", architecture="qwen2",
                               n_embd=896, tensors={Q5_0: 132, F32: 121, Q4_K: 12})
            census = wc.read_census(path)
            self.assertEqual(census.dominant_quant, "Q5_0")
            self.assertEqual(census.n_embd, 896)
            self.assertFalse(census.superblock_compatible)

    def test_float_tensors_never_count_as_the_dominant_quant(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "m.gguf", architecture="qwen3",
                               n_embd=1024, tensors={F32: 500, Q4_K: 10})
            self.assertEqual(wc.read_census(path).dominant_quant, "Q4_K")


class Gate(unittest.TestCase):

    def test_the_real_defect_is_refused(self):
        """896 / 256 = 3.5. This is the run that cost a month."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "Qwen2.5-Coder-0.5B-Q4_K_M.gguf",
                               architecture="qwen2", n_embd=896,
                               tensors={Q5_0: 132, F32: 121, Q8_0: 13, Q6_K: 12})
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.verify_workload(path)
            self.assertIn("not divisible by 256", str(caught.exception))

    def test_a_production_family_workload_is_accepted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "ok.gguf", architecture="qwen2",
                               n_embd=1536, tensors={Q4_K: 169, F32: 141, Q6_K: 29})
            census = wc.verify_workload(path)
            self.assertEqual(census.dominant_quant, "Q4_K")
            self.assertTrue(census.in_production_family)

    def test_a_divisible_model_dominated_by_a_legacy_quant_is_still_refused(self):
        """The two failure modes are independent.

        A hidden dim divisible by 256 does not make a Q5_0-dominated model a valid
        screening surface: it still exercises a kernel production never dispatches.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "legacy.gguf", architecture="qwen2",
                               n_embd=1536, tensors={Q5_0: 169, F32: 141})
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.verify_workload(path)
            self.assertIn("outside the production family", str(caught.exception))

    def test_an_iquant_workload_is_in_the_production_family(self):
        """Production serves 122B IQ2; the family is K-quants AND I-quants."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "iq.gguf", architecture="qwen3",
                               n_embd=4096, tensors={16: 200, F32: 50})
            self.assertEqual(wc.verify_workload(path).dominant_quant, "IQ2_XXS")

    def test_a_non_gguf_file_is_refused_clearly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not.gguf"
            path.write_bytes(b"this is not a gguf file at all")
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.read_census(path)
            self.assertIn("not a GGUF", str(caught.exception))


class TheRealModelsOnDisk(unittest.TestCase):
    """Guard rails against the actual files, skipped when they are absent."""

    OLD = Path("/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/"
               "Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
    NEW = Path("/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")

    def test_the_superseded_workload_is_refused(self):
        if not self.OLD.is_file():
            self.skipTest("superseded workload not present")
        with self.assertRaises(wc.WorkloadContractError):
            wc.verify_workload(self.OLD)

    def test_the_replacement_workload_passes_and_dispatches_q4_k(self):
        if not self.NEW.is_file():
            self.skipTest("replacement workload not present")
        census = wc.verify_workload(self.NEW)
        self.assertEqual(census.dominant_quant, "Q4_K")
        self.assertEqual(census.n_embd, 1536)
        self.assertGreaterEqual(census.tensor_types.get("Q4_K", 0), 100)


if __name__ == "__main__":
    unittest.main()
