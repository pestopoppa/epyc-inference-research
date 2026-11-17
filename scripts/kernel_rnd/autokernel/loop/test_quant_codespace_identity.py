"""MUL_MAT-identity error classes, replayed from the real receipts of the first C12 row
akx31-qdec-c12-gpu-20261117 (fixtures/qdec_c12_r1/*.judge.json: the judge's first mismatches per leg,
with codes, reference and native float32 bits). That row ended QDEC_FAIL on every mul_mat leg because the
old judge required |native - 127*ref| <= 8 float32 ulps, which no q8-activation kernel provides."""
import json
import struct
import unittest
from pathlib import Path

import numpy as np

from autokernel.loop import quant_codespace as qc

FIX = Path(__file__).with_name("fixtures") / "qdec_c12_r1"


def _f32(h):
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _replay(path):
    d = json.loads(path.read_text())
    rows = d["first"]
    ref = np.array([_f32(r["ref"]) for r in rows], dtype=np.float32)
    got = np.array([_f32(r["got"]) for r in rows], dtype=np.float32)
    codes = [r["codes"] for r in rows]
    col = lambda k: np.array([c[k] for c in codes]) if all(k in c for c in codes) else None
    q8 = col("q") if d["type"] == "Q8_0" else None
    bad, cls = qc.classify_identity(ref, got, np.abs(ref.astype(np.float64)), d["onehot"],
                                    d=col("d"), dmin=col("dmin"), e=col("e"), q8=q8)
    return d, bad, cls


EXPECT = {   # (type, leg) -> the class every replayed first-mismatch falls in; None = must stay unexplained
    ("Q8_0", "mul_mat:CPU"): "int8_min_code",
    ("Q8_0", "mul_mat:ROCm0"): "ftz_subnormal_scale",
    ("Q4_K", "mul_mat:CPU"): "fp16_scale_rounding",
    ("Q4_K", "mul_mat:ROCm0"): "fp16_scale_rounding",
    ("Q6_K", "mul_mat:ROCm0"): "fp16_scale_rounding",
    ("IQ2_XXS", "mul_mat:ROCm0"): "ftz_subnormal_scale",
    ("IQ2_XS", "mul_mat:ROCm0"): "ftz_subnormal_scale",
    ("IQ2_S", "mul_mat:ROCm0"): "ftz_subnormal_scale",
    ("IQ3_XXS", "mul_mat:ROCm0"): "ftz_subnormal_scale",
    ("IQ3_S", "mul_mat:CPU"): "fp16_scale_rounding",
    ("IQ4_XS", "mul_mat:CPU"): "fp16_scale_rounding",
    ("IQ4_XS", "mul_mat:ROCm0"): "fp16_scale_rounding",
    ("MXFP4", "mul_mat:CPU"): "onehot_overflow",
    ("MXFP4", "mul_mat:ROCm0"): "ftz_subnormal_scale",
    # real residuals the error model must NOT absorb (kernel review, not a judge fix):
    ("IQ3_XXS", "mul_mat:CPU"): None,     # ref 1 -> 1.25 at d = 1.0 (25 %)
}


class IdentityClassesFromRealReceipts(unittest.TestCase):
    def test_every_fixture_classifies_as_expected(self):
        seen = set()
        for p in sorted(FIX.glob("*.mul_mat.*.judge.json")):
            raw = json.loads(p.read_text())
            if raw["verdict"] == "pass" or (raw["type"], raw["leg"]) not in EXPECT:
                continue
            d, bad, cls = _replay(p)
            key = (d["type"], d["leg"])
            seen.add(key)
            want = EXPECT[key]
            with self.subTest(key=key, classes=cls):
                if want is None:
                    self.assertTrue(bad.any(), "a large unexplained error must stay a failure")
                else:
                    self.assertFalse(bad.any())
                    self.assertEqual(cls[want], len(d["first"]))
        self.assertEqual(seen, set(EXPECT))

    def test_q6k_cpu_residual_is_partly_unexplained(self):
        d, bad, cls = _replay(FIX / "Q6_K.mul_mat.CPU.judge.json")
        # -16 -> -15.999 is fp16 rounding, -11 -> -10.96 (3.6e-3) is not: the residual stays a failure
        self.assertTrue(bad.any()); self.assertGreater(cls["fp16_scale_rounding"], 0)

    def test_exact_values_and_wrong_sign_are_not_benign(self):
        ref = np.array([2.0, -128.0, 1.0], dtype=np.float32)
        got = np.array([254.0, 128.0 * 127, 0.0], dtype=np.float32)    # exact; sign flip without -128 code; 0 at normal scale
        bad, cls = qc.classify_identity(ref, got, np.abs(ref), 127.0, d=np.array([0x3C00] * 3), q8=np.array([2, 5, 1]))
        self.assertEqual(bad.tolist(), [False, True, True]); self.assertEqual(cls["unexplained"], 2)


if __name__ == "__main__":
    unittest.main()
