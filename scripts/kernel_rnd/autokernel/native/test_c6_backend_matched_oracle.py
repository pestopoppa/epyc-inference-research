from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from . import c6_backend_matched_oracle as O

FIXTURE = json.loads((Path(__file__).with_name(
    "fixture_c6_native_residency_r2.json")).read_text())


def _f32(hexstr: str) -> list[float]:
    raw = bytes.fromhex(hexstr)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class TestBackendMatchedQ4KOracle(unittest.TestCase):
    """r2 captured a bit-exact production v10 ROCm0 q4_K MMVQ output that scored
    matched_ratio 0.625 against the CPU (Q8_K) reference."""

    def setUp(self):
        self.weights = bytes.fromhex(FIXTURE["weights_hex"])
        self.acts = _f32(FIXTURE["activations_f32le_hex"])
        self.cpu = _f32(FIXTURE["cpu_reference_f32le_hex"])
        self.gpu = _f32(FIXTURE["rocm0_candidate_f32le_hex"])
        self.ref = O.q4_k_mmvq_reference(self.weights, self.acts, 32, 256, 1)

    def test_emulation_reproduces_the_rocm_output(self):
        self.assertLess(max(abs(a - b) for a, b in zip(self.ref, self.gpu)), 1e-5)
        self.assertEqual(O.matched_ratio(self.ref, self.gpu, 0.01, 0.01), 1.0)

    def test_cpu_reference_reproduces_the_r2_false_fail(self):
        self.assertEqual(O.matched_ratio(self.cpu, self.gpu, 0.01, 0.01), 0.625)
        self.assertLess(O.nmse(self.cpu, self.gpu), 5e-4)

    def test_a_wrong_kernel_is_still_caught(self):
        # mutations: a uniform offset (a mis-scaled min term) and one sign flip
        bad = [g + 0.25 for g in self.gpu]
        self.assertLess(O.matched_ratio(self.ref, bad, 0.01, 0.01), 0.95)
        flipped = list(self.gpu)
        flipped[3] = -flipped[3]
        self.assertLess(O.matched_ratio(self.ref, flipped, 0.01, 0.01), 1.0)

    def test_quantized_sum_mode_differs_and_still_fails_cpu_gate(self):
        champ = O.q4_k_mmvq_reference(self.weights, self.acts, 32, 256, 1, min_sum="quantized")
        self.assertGreater(max(abs(a - b) for a, b in zip(champ, self.gpu)), 1e-3)
        self.assertLess(O.matched_ratio(self.cpu, champ, 0.01, 0.01), 0.95)
        self.assertLess(O.nmse(self.cpu, champ), O.nmse(self.cpu, self.gpu))
        with self.assertRaises(ValueError):
            O.q4_k_mmvq_reference(self.weights, self.acts, 32, 256, 1, min_sum="x")

    def test_shape_mismatch_refused(self):
        with self.assertRaises(ValueError):
            O.q4_k_mmvq_reference(self.weights, self.acts, 31, 256, 1)


if __name__ == "__main__":
    unittest.main()
