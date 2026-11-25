"""VKRC-BYTES-PLAUSIBLE: census_extract.flag_implausible_bytes on the verify_roofline fixture (over-peak, off-footprint, plausible)."""
import copy
import json
import os
import unittest

from autokernel.loop import census_update as cu
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
        k = res["shapes"][0]["kernels"][0]    # as produced: 64 launches, grid 2228224 x ncols_y 1 / 128 = 17408 rows, fused gate+up (2 x 94.7 MB)
        self.assertNotIn("weight_shape", k)  # no injected key: the shape comes from the grid
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

    def test_grid_rows_select_or_reject_shape(self):
        rows = [o for o in extract_verify_roofline(self.res) if o["kernel_class"] in ("vk_dense_mmvq", "vk_lm_head")]
        fp = {(o["calls_per_round"], o["shape"].rsplit(":g", 1)[-1].split("x")[0]): o["footprint"] for o in rows}
        self.assertEqual(fp[(64, "2228224")], 189399040)     # 17408 rows, fused: no injected key
        self.assertEqual({o["footprint"] for o in rows if o["kernel_class"] == "vk_lm_head"}, {248320 * 5120 * 34 // 32})
        self.assertEqual(fp[(80, "655360")], "unknown")      # 5120 rows, 64 ffn_down + 16 attn_out: ambiguous
        self.assertEqual(fp[(48, "655360")], 5120 * 6144 * 34 // 32)   # 5120 rows, 48 launches = ssm_out
        self.assertEqual(fp[(96, "6144")], "unknown")        # 48 rows: no known shape
        self.assertTrue(all(o["bytes_flags"] == [] for o in rows))

    def test_weight_shape_non_string_ignored(self):
        res = copy.deepcopy(self.res)
        res["shapes"][0]["kernels"][0]["weight_shape"] = 17408
        self.assertEqual(flag_implausible_bytes(res), [])

    def test_apply_prints_implausible_and_check_lines(self):
        res = copy.deepcopy(self.res)
        res["shapes"][0]["kernels"][2]["time_ms"] = 0.1
        obs = extract_verify_roofline(res)
        lines = cu.apply(cu.empty_census(), obs, "r", "s" * 16, "2026-11-25T00:00:00Z", "t")
        imp = [x for x in lines if "BYTES-IMPLAUSIBLE" in x]
        chk = [x for x in lines if "BYTES-CHECK" in x]
        self.assertEqual(len(imp), 1)
        self.assertIn("gv_fattn_q8kv", imp[0])
        unk = sum(o["footprint"] == "unknown" for o in obs)
        self.assertEqual(len(chk), 1)
        self.assertTrue(chk[0].endswith(f"BYTES-CHECK: {len(obs)} rows, 1 flagged, {unk} footprint-unknown (peak check only)"), chk)

    def test_no_check_line_without_verify_roofline_rows(self):
        lines = cu.apply(cu.empty_census(), [], "r", "s" * 16, "2026-11-25T00:00:00Z", "t")
        self.assertEqual([x for x in lines if "BYTES-" in x], [])

    def test_wired_extractor_rows_carry_flag_fields(self):
        res = copy.deepcopy(self.res)
        res["shapes"][0]["kernels"][2]["time_ms"] = 0.1
        rows = extract_verify_roofline(res)
        self.assertTrue(rows and all("bytes_flags" in o and "footprint" in o for o in rows))
        self.assertEqual([o["kernel_class"] for o in rows if o["bytes_flags"]], ["gv_fattn_q8kv"])
        self.assertEqual(len(rows), len(extract_verify_roofline(self.res)))   # flags never filter


if __name__ == "__main__":
    unittest.main()
