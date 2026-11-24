"""MUL_MAT-identity error classes, replayed from the real receipts of the first C12 row
akx31-qdec-c12-gpu-20261117 (fixtures/qdec_c12_r1/*.judge.json: the judge's first mismatches per leg,
with codes, reference and native float32 bits). That row ended QDEC_FAIL on every mul_mat leg because the
old judge required |native - 127*ref| <= 8 float32 ulps, which no q8-activation kernel provides."""
import json
import struct
import unittest
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="numerics tests need numpy (declared project dependency)")
import numpy as np

from autokernel.loop import quant_codespace as qc

FIX = Path(__file__).with_name("fixtures") / "qdec_c12_r1"


def _f32(h):
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _term_magnitude(d, codes, ref):
    """Q4_K term magnitude |d*sc*q| + |dmin*m| from the receipt codes (what the judge passes); |ref| otherwise."""
    if d["type"] != "Q4_K":
        return np.abs(ref.astype(np.float64))
    f16 = lambda k: np.array([c[k] for c in codes], dtype=np.uint16).view(np.float16).astype(np.float64)
    i = lambda k: np.array([c[k] for c in codes], dtype=np.float64)
    return np.abs(f16("d") * i("sc") * i("q")) + np.abs(f16("dmin") * i("m"))


def _replay(path, gpu=False, term_magnitude=False):
    d = json.loads(path.read_text())
    rows = d["first"]
    ref = np.array([_f32(r["ref"]) for r in rows], dtype=np.float32)
    got = np.array([_f32(r["got"]) for r in rows], dtype=np.float32)
    codes = [r["codes"] for r in rows]
    col = lambda k: np.array([c[k] for c in codes]) if all(k in c for c in codes) else None
    q8 = col("q") if d["type"] == "Q8_0" else None
    mag = _term_magnitude(d, codes, ref) if term_magnitude else np.abs(ref.astype(np.float64))
    bad, cls = qc.classify_identity(ref, got, mag, d["onehot"],
                                    d=col("d"), dmin=col("dmin"), e=col("e"), q8=q8, gpu=gpu)
    return d, bad, cls


FIX3 = Path(__file__).with_name("fixtures") / "qdec_c12_r3"

EXPECT_R3 = {   # r3 (akx31-qdec-c12-gpu-20261117-r3) failing legs -> class of every replayed first mismatch
    ("Q4_K", "mul_mat:CPU"): "fp16_scale_rounding",      # cancellation: benign only against the term magnitude
    ("Q4_K", "mul_mat:ROCm0"): "fp16_scale_rounding",
    ("MXFP4", "mul_mat:ROCm0"): ("ftz_output_subnormal", "fp16_ftz_operand"),  # e small: subnormal operand, got 0
    ("IQ2_XXS", "mul_mat:ROCm0"): "fp16_intermediate_overflow",   # d = 0x7BFF -> NaN (out of domain)
    ("IQ2_XS", "mul_mat:ROCm0"): "fp16_intermediate_overflow",
    ("IQ2_S", "mul_mat:ROCm0"): "fp16_intermediate_overflow",
    ("IQ3_XXS", "mul_mat:ROCm0"): "fp16_intermediate_overflow",
    # REAL CPU defects (AKX-ALL-31.QDEC-CPU-IQ3Q6) must stay failures:
    ("IQ3_XXS", "mul_mat:CPU"): None,   # iqk Ny>=2 _mm256_add_epi32 byte carry: +1 grid unit (1 -> 1.25)
    ("Q6_K", "mul_mat:CPU"): None,      # iqk Q6_K -> Q8_0_R8 8-bit requantization at Ny >= 64
}


class R3JudgeAmendments(unittest.TestCase):
    def test_r3_fixtures_classify_as_expected(self):
        seen = set()
        for p in sorted(FIX3.glob("*.mul_mat.*.judge.json")):
            raw = json.loads(p.read_text())
            key = (raw["type"], raw["leg"])
            if raw["verdict"] == "pass" or key not in EXPECT_R3:
                continue
            d, bad, cls = _replay(p, gpu=not raw["leg"].endswith(":CPU"), term_magnitude=True)
            seen.add(key)
            want = EXPECT_R3[key]
            with self.subTest(key=key, classes=cls):
                if want is None:
                    self.assertTrue(bad.any(), "a real CPU defect must stay a failure")
                    self.assertGreater(cls["max_rel_err_unexplained"], qc.FP16_SCALE_REL)
                else:
                    self.assertFalse(bad.any())
                    want = (want,) if isinstance(want, str) else want
                    self.assertEqual(sum(cls[w] for w in want), len(d["first"]))
        self.assertEqual(seen, set(EXPECT_R3))

    def test_q4k_cancellation_needs_term_magnitude(self):
        # result-relative rounding (the r3 defect) leaves the Q4_K receipts unexplained; term-relative clears them
        for dev in ("CPU", "ROCm0"):
            p = FIX3 / f"Q4_K.mul_mat.{dev}.judge.json"
            _, bad_res, _ = _replay(p, gpu=dev != "CPU", term_magnitude=False)
            _, bad_term, _ = _replay(p, gpu=dev != "CPU", term_magnitude=True)
            self.assertTrue(bad_res.any()); self.assertFalse(bad_term.any())

    def test_iq3xxs_cpu_is_one_grid_unit(self):
        d, bad, cls = _replay(FIX3 / "IQ3_XXS.mul_mat.CPU.judge.json", term_magnitude=True)
        for r in d["first"]:
            dv = np.array([r["codes"]["d"]], dtype=np.uint16).view(np.float16)[0]
            unit = float(dv) * (2 * r["codes"]["s"] + 1) / 4
            self.assertAlmostEqual(r["got_value"] / d["onehot"] - r["ref_value"], unit, places=3)

    def test_overflow_class_is_gpu_only_and_out_of_domain(self):
        p = FIX3 / "IQ3_XXS.mul_mat.ROCm0.judge.json"
        _, bad_cpu, cls_cpu = _replay(p, gpu=False)
        self.assertTrue(bad_cpu.all()); self.assertEqual(cls_cpu["fp16_intermediate_overflow"], 0)
        self.assertNotIn("fp16_intermediate_overflow", qc.IDENTITY_CLASSES)
        self.assertIn("fp16_intermediate_overflow", qc.OUT_OF_DOMAIN_CLASSES)

    def test_output_subnormal_ftz_requires_subnormal_reference(self):
        tiny = np.finfo(np.float32).tiny
        ref = np.array([tiny / 2, tiny * 2], dtype=np.float32)
        got = np.zeros(2, dtype=np.float32)
        bad, cls = qc.classify_identity(ref, got, np.abs(ref), 127.0, e=np.array([1, 2]))
        self.assertEqual(bad.tolist(), [False, True]); self.assertEqual(cls["ftz_output_subnormal"], 1)

    def test_row_poisoning_and_fp16_ftz_operand(self):
        row = 4
        ref = np.array([1e6, 1.0, 2.0, 3.0, 1.0, 2.0, 1e-6, 4.0], dtype=np.float32)
        got = np.array([np.inf, np.nan, np.nan, np.nan, np.nan, 254.0, 0.0, 508.0], dtype=np.float32)
        bad, cls = qc.classify_identity(ref, got, np.abs(ref), 127.0, gpu=True, row=row)
        # row 0 holds an overflowed element: its NaNs are out of domain; row 1's NaN is not
        self.assertEqual(bad.tolist(), [False, False, False, False, True, False, False, False])
        self.assertEqual(cls["fp16_intermediate_overflow"], 4); self.assertEqual(cls["fp16_ftz_operand"], 1)
        bad_cpu, _ = qc.classify_identity(ref, got, np.abs(ref), 127.0, gpu=False, row=row)
        self.assertTrue(bad_cpu[:5].all() and bad_cpu[6])

    def test_judge_verdict_and_max_err(self):
        qt = qc.qtype("IQ3_XXS")
        # synthetic: one block of zeros decodes to grid index 0; feed back 127*ref exactly -> pass, max_err 0
        b = qt.empty(1)
        b[:, 0:2] = np.array([0x3C00], dtype=np.uint16).view(np.uint8)
        ref = qt.decode(b).ravel()
        res = qc.judge(qt, b.tobytes(), ref * 127.0, "mul_mat:CPU", "identity")
        self.assertEqual(res["verdict"], "pass"); self.assertEqual(res["max_err"]["abs_err"], 0.0)
        bad = ref * 127.0; bad[1] += 0.25 * 127.0
        res = qc.judge(qt, b.tobytes(), bad, "mul_mat:CPU", "identity")
        self.assertEqual(res["verdict"], "fail"); self.assertGreater(res["max_err"]["rel_err_unexplained"], 0.0)
        nan = ref * 127.0; nan[:] = np.nan
        b2 = b.copy(); b2[:, 0:2] = np.array([0x7BFF], dtype=np.uint16).view(np.uint8)
        res = qc.judge(qt, b2.tobytes(), nan, "mul_mat:ROCm0", "identity")
        self.assertEqual(res["verdict"], "out_of_domain"); self.assertEqual(res["out_of_domain"], 256)
        self.assertEqual(res["mismatches"], 0)


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

    def test_r1_fixtures_unchanged_under_r3_judge_gpu_flag(self):
        # the r1 expectations above hold with the leg's real gpu flag too (no ROCm0 r1 leg is out of domain)
        for p in sorted(FIX.glob("*.mul_mat.ROCm0.judge.json")):
            d, bad, cls = _replay(p, gpu=True)
            if (d["type"], d["leg"]) in EXPECT and d["verdict"] != "pass":
                self.assertEqual(cls["fp16_intermediate_overflow"], 0, p.name)

    def test_exact_values_and_wrong_sign_are_not_benign(self):
        ref = np.array([2.0, -128.0, 1.0], dtype=np.float32)
        got = np.array([254.0, 128.0 * 127, 0.0], dtype=np.float32)    # exact; sign flip without -128 code; 0 at normal scale
        bad, cls = qc.classify_identity(ref, got, np.abs(ref), 127.0, d=np.array([0x3C00] * 3), q8=np.array([2, 5, 1]))
        self.assertEqual(bad.tolist(), [False, True, True]); self.assertEqual(cls["unexplained"], 2)


def _q6k_mixed_block(sc_hi=127, sc_lo=1, d_bits=0x3C00, seed=0):
    """One Q6_K block whose every 32-block spans two sub-blocks (sc_hi, sc_lo); q random in 0..63."""
    qt = qc.qtype("Q6_K")
    rng = np.random.default_rng(seed)
    f = {"d": np.array([[d_bits]], dtype=np.int64), "q": rng.integers(0, 64, (1, 256)),
         "sc": np.array([[sc_hi, sc_lo] * 8], dtype=np.int64)}
    return qt, f, qt.pack(f)


def _sim_q8_0_requant(qt, f):
    """Mimic iqk_convert_q6_k_q8_0_r8 on one block: per 32-block, int8 codes under one fp16 scale M/127."""
    e = qt.elements(f)
    v = (e["sc"].astype(np.uint8).view(np.int8).astype(np.int64) * (e["q"] - 32)).reshape(8, 32).astype(np.float64)
    m = np.abs(v).max(axis=1, keepdims=True)
    d8 = (m / 127.0).astype(np.float16).astype(np.float64)
    code = np.rint(v / np.where(d8 > 0, d8, 1.0))
    return (code * d8 * float(np.array([f["d"][0, 0]], dtype=np.uint16).view(np.float16)[0])).ravel()


class Q6KRequantClass(unittest.TestCase):
    """AK-Q6K-QDEC-ARTEFACT: iqk requantises Q6_K to Q8_0_R8 (one fp16 scale per 32-block) at >= 64 activation columns."""

    def _judge(self, leg, got_delta):
        qt, f, blk = _q6k_mixed_block()
        ref = qt.decode(blk).ravel().astype(np.float64)
        got = (ref * 127.0 + got_delta).astype(np.float32)
        return qc.judge(qt, blk.tobytes(), got, leg, "identity"), qt, f, ref

    def test_bound_value_for_worst_case_block(self):
        qt, f, blk = _q6k_mixed_block(sc_hi=127, sc_lo=1)
        f["q"][:, :] = 0                       # |sc*(q-32)| = 127*32 = 4064 in the sc_hi half
        scale = qc.q6k_block32_scale(qt, f, qt.elements(f))
        self.assertTrue(np.all(scale == 4064.0))
        # 127 * 4064 * REQUANT_REL = 127 * (16 + 1) = 2159 (16 = half int8 step of 32, 1 = fp16 slack)
        self.assertAlmostEqual(127.0 * 4064.0 * qc.REQUANT_REL, 2159.0, places=6)

    def test_simulated_requant_is_inside_the_bound(self):
        worst = 0.0
        for seed in range(200):
            hi = int(np.random.default_rng(seed).integers(40, 128)); lo = int(np.random.default_rng(seed + 1).integers(1, 20))
            qt, f, blk = _q6k_mixed_block(sc_hi=hi, sc_lo=lo, seed=seed)
            ref = qt.decode(blk).ravel().astype(np.float64)
            got = _sim_q8_0_requant(qt, f) * 127.0
            bound = 127.0 * qc.q6k_block32_scale(qt, f, qt.elements(f)) * qc.REQUANT_REL
            err = np.abs(got - ref * 127.0)
            self.assertTrue(np.all(err <= bound), (seed, float((err - bound).max())))
            worst = max(worst, float((err / bound).max()))
        self.assertGreater(worst, 0.3)         # the model is not vacuous: the simulated loss really uses the budget

    def test_mixed_sc_block_at_the_bound_passes_and_beyond_stays_unexplained(self):
        qt, f, blk = _q6k_mixed_block()
        scale = qc.q6k_block32_scale(qt, f, qt.elements(f))
        bound = 127.0 * scale * qc.REQUANT_REL
        res, *_ = self._judge("mul_mat:CPU", bound * 0.999)   # every element just inside its own bound
        self.assertEqual(res["verdict"], "pass"); self.assertEqual(res["mismatches"], 0)
        self.assertGreater(res["classes"]["iqk_q6k_q8_0_requant"], 0)
        self.assertLessEqual(res["max_err"]["requant_err_over_bound"], 1.0)
        delta2 = bound * 0.999; delta2[37] = bound[37] * 1.01   # one element beyond its bound
        res2, *_ = self._judge("mul_mat:CPU", delta2)
        self.assertEqual(res2["verdict"], "fail"); self.assertEqual(res2["mismatches"], 1)
        self.assertEqual(res2["first"][0]["element"], 37)

    def test_class_is_offered_to_the_wide_cpu_leg_only(self):
        qt, f, blk = _q6k_mixed_block()
        scale = qc.q6k_block32_scale(qt, f, qt.elements(f))
        delta = 127.0 * scale * qc.REQUANT_REL * 0.5
        for leg in ("mul_mat_n32:CPU", "mul_mat_dq0:CPU", "mul_mat_iqk0:CPU", "mul_mat_tight:CPU"):
            res, *_ = self._judge(leg, delta)
            self.assertEqual(res["verdict"], "fail", leg); self.assertFalse(res["requant_class"])
            self.assertEqual(res["classes"]["iqk_q6k_q8_0_requant"], 0)
        res, *_ = self._judge("mul_mat:ROCm0", delta)            # GPU leg: never
        self.assertEqual(res["verdict"], "fail")
        for leg in ("mul_mat:CPU", "mul_mat_n32:CPU", "mul_mat_dq0:CPU"):   # an exact result passes under every leg
            res, *_ = self._judge(leg, np.zeros(256))
            self.assertEqual(res["verdict"], "pass", leg)

    def test_other_types_never_get_the_class(self):
        qt = qc.qtype("Q8_0")
        b = qt.empty(8); b[:, 0:2] = np.array([0x3C00], dtype=np.uint16).view(np.uint8)
        ref = qt.decode(b).ravel()
        res = qc.judge(qt, b.tobytes(), ref * 127.0 + 5.0, "mul_mat:CPU", "identity")
        self.assertFalse(res["requant_class"])

    def test_gross_error_stays_unexplained_on_the_wide_leg(self):
        qt, f, blk = _q6k_mixed_block()
        ref = qt.decode(blk).ravel().astype(np.float64)
        res, *_ = self._judge("mul_mat:CPU", -2.0 * ref * 127.0 + np.where(ref >= 0, 1e6, -1e6))
        self.assertEqual(res["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
