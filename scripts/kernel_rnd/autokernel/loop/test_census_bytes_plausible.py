"""VKRC-BYTES-PLAUSIBLE: census_extract.flag_implausible_bytes on the verify_roofline fixture (over-peak, off-footprint, plausible)."""
import copy
import json
import os
import unittest

from autokernel.loop.census_extract import extract_verify_roofline, flag_implausible_bytes

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "census", "verify_roofline.RESULT.json")


class BytesPlausible(unittest.TestCase):
    def setUp(self):
        self.res = json.load(open(FX))

    def flags(self, res):
        return {f["kernel_class"]: f["flag"] for f in flag_implausible_bytes(res)}

    def test_plausible_fixture_not_flagged(self):
        self.assertEqual(flag_implausible_bytes(self.res), [])

    def test_over_peak_row_flagged(self):
        res = copy.deepcopy(self.res)
        res["shapes"][0]["kernels"][2]["time_ms"] = 0.1   # gv_fattn: 315.9 MB / 16 launches in 6.25 us/launch = 3.2 TB/s
        f = self.flags(res)
        self.assertTrue(f["gv_fattn_q8kv"].startswith("over_peak"), f)
        self.assertEqual(len(f), 1)

    def test_bytes_3x_footprint_flagged(self):
        res = copy.deepcopy(self.res)
        k = res["shapes"][0]["kernels"][0]    # 64 launches, 189.5 MB each = the fused gate+up footprint (2 x 94.7 MB)
        k.update(weight_shape="17408x5120", fused_gate_up=True)   # 64 launches is ambiguous, so the row must carry its shape
        k["bytes"] *= 3
        k["time_ms"] *= 3                     # keep GB/s unchanged so only the footprint rule can fire
        f = self.flags(res)
        self.assertEqual(list(f), ["vk_dense_mmvq"])
        self.assertTrue(f["vk_dense_mmvq"].startswith("off_footprint >2x (3.00x"), f)

    def shaped(self, factor):
        res = copy.deepcopy(self.res)
        k = res["shapes"][0]["kernels"][0]
        k.update(weight_shape="17408x5120", fused_gate_up=True)
        k["bytes"] *= factor
        k["time_ms"] *= factor
        return res

    def test_2p5x_under_footprint_flagged(self):
        f = self.flags(self.shaped(0.4))
        self.assertEqual(list(f), ["vk_dense_mmvq"])
        self.assertIn("off_footprint >2x (0.40x", f["vk_dense_mmvq"])

    def test_1p5x_over_footprint_not_flagged(self):
        self.assertEqual(flag_implausible_bytes(self.shaped(1.5)), [])

    def test_unknown_shape_marked_unknown_not_dropped(self):
        rows = extract_verify_roofline(self.res)
        r64 = [o for o in rows if o["kernel_class"] == "vk_dense_mmvq" and o["calls_per_round"] == 64]
        self.assertEqual([o["footprint"] for o in r64], ["unknown"])   # 64 launches = ffn_down, ssm_out or fused gate+up: ambiguous
        self.assertEqual(r64[0]["bytes_flags"], [])

    def test_wired_extractor_rows_carry_flag_fields(self):
        res = copy.deepcopy(self.res)
        res["shapes"][0]["kernels"][2]["time_ms"] = 0.1
        rows = extract_verify_roofline(res)
        self.assertTrue(rows and all("bytes_flags" in o and "footprint" in o for o in rows))
        self.assertEqual([o["kernel_class"] for o in rows if o["bytes_flags"]], ["gv_fattn_q8kv"])
        self.assertEqual(len(rows), len(extract_verify_roofline(self.res)))   # flags never filter


if __name__ == "__main__":
    unittest.main()
