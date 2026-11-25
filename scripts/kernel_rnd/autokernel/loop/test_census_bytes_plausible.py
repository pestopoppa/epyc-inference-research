"""VKRC-BYTES-PLAUSIBLE: census_extract.flag_implausible_bytes on the verify_roofline fixture (over-peak, off-footprint, plausible)."""
import copy
import json
import os
import unittest

from autokernel.loop.census_extract import flag_implausible_bytes

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
        k["bytes"] *= 3
        k["time_ms"] *= 3                     # keep GB/s unchanged so only the footprint rule can fire
        f = self.flags(res)
        self.assertEqual(f, {"vk_dense_mmvq": "off_footprint >2x"})


if __name__ == "__main__":
    unittest.main()
