"""The C6 split policy refusal is typed and names the refusing legs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.kernel_rnd.autokernel.controller import gpu_source_evidence as E  # noqa: E402


def _rows(*ok):
    return [NS(correct=v) for v in ok]


class C6SplitPolicyRefusalTest(unittest.TestCase):
    def legs(self, **kw):
        base = dict(oracle_exit_code=0, numerical=_rows(True, True, True),
                    precision_equivalence=_rows(True), determinism_correct=True,
                    semantic_gating=False)
        base.update(kw)
        return E._c6_policy_failed_legs(**base)

    def test_all_pass_is_empty(self):
        self.assertEqual(self.legs(), ())

    def test_each_leg_is_named(self):
        self.assertEqual(self.legs(oracle_exit_code=1), ("oracle_exit",))
        self.assertEqual(self.legs(numerical=_rows(True, False)), ("numerics",))
        self.assertEqual(self.legs(precision_equivalence=_rows(False)),
                         ("precision_equivalence",))
        self.assertEqual(self.legs(determinism_correct=False), ("determinism",))
        self.assertEqual(self.legs(semantic_gating=True), ("semantic_judge",))
        self.assertEqual(self.legs(determinism_correct=False, semantic_gating=True),
                         ("determinism", "semantic_judge"))

    def test_typed_not_generic(self):
        exc = E.C6SplitPolicyRefusal(("determinism",))
        self.assertIsInstance(exc, E.EvidenceProducerError)
        self.assertTrue(type(exc).__name__.startswith("C6"))
        self.assertEqual(exc.failed_legs, ("determinism",))
        self.assertIn("determinism", str(exc))


    def test_carries_sealed_receipt(self):
        exc = E.C6SplitPolicyRefusal(("precision_equivalence",),
                                     receipt_path="/r", receipt_sha256="a" * 64)
        self.assertIsInstance(exc, E.SealedEvidenceRefusal)
        self.assertEqual(exc.receipt_path, "/r")


class QuantizedPrecisionBoundTest(unittest.TestCase):
    """r12: a correct q5_0 MMVQ kernel (NMSE 3.6e-5) had normalized RMS
    6.0e-3 against the unquantized-activation float64 reference."""

    def policy(self, dtype):
        H = E.hawkeye_measurement
        return H.PrecisionEquivalencePolicy(
            operator_id="MUL_MAT", template_id="cuda-vecdotq-v1",
            input_dtype=dtype, required_output_dtype="f32",
            required_accumulator_dtype="f32", reduce_dimension=256,
            structural_evidence_sha256="a" * 64)

    def verdict(self, dtype, rel):
        ref = [5.0 * (1 if i % 2 else -1) for i in range(32)]
        got = [v * (1 + rel) for v in ref]
        return E.hawkeye_measurement.evaluate_precision_equivalence(
            ref, got, policy=self.policy(dtype),
            observed_output_dtype="f32", observed_accumulator_dtype="f32")

    def test_f32_bound_refuses_q8_quantization_noise(self):
        self.assertFalse(self.verdict("float32", 1.2e-3).correct)

    def test_q8_block_bound_accepts_noise_and_refuses_wrong_kernel(self):
        self.assertTrue(self.verdict("q8_1", 1.2e-3).correct)
        self.assertEqual(self.policy("q8_1").input_dtype, "q8_block")
        self.assertFalse(self.verdict("q8_1", 0.5).correct)


if __name__ == "__main__":
    unittest.main()
