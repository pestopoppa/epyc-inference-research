"""AK-BNSIG P1: bottleneck_signature classifier, census signature field, golden no-change. No GPU."""
import json
import os
import unittest

from autokernel.loop import bottleneck_signature as bs
from autokernel.loop import census_update as cu
from autokernel.loop import gpu_counter_signals as gcs

FX = os.path.join(os.path.dirname(__file__), "fixtures", "bnsig")
RESULT = os.path.join(FX, "q8wt_diag_RESULT_20261125.json")   # trimmed from ak-c22-q8wt-op-diag-gpu-20261125/out/RESULT.json
GOLDEN = os.path.join(FX, "census_golden_no_signature.json")   # written by origin/main census_update before AK-BNSIG


def load():
    return json.load(open(RESULT))


class Classifier(unittest.TestCase):
    def test_v5i_issue_bound_on_todays_result(self):
        r = load()
        d = bs.from_result(r, "v5i", gbps=r["summary"]["v5i_n8_geomean_gbps"])
        self.assertEqual(d["schema"], "epyc.autokernel.bottleneck_signature.v1")
        self.assertEqual(d["class"], "issue_bound")
        self.assertEqual(d["lever"], "cut instructions per byte")
        self.assertEqual(d["refs"]["S16_id"], "S16")
        self.assertEqual(d["refs"]["row"], "ak-c22-q8wt-op-diag-gpu-20261125")
        self.assertAlmostEqual(d["axes"]["issue_busy"], 0.4645)
        self.assertAlmostEqual(d["axes"]["wait_inst"], 0.5065)
        self.assertEqual(d["counter_set_sha256"], gcs.signature_counter_set_sha256())
        # finding 1: with the direction taken from the references, BOTH over-subscription legs are past the
        # midpoint for V5I (the diag's inverted wait_any leg said "not over-subscribed")
        self.assertEqual(d["oversub_legs"], {"mem_stalled": True, "wait_any": True})
        self.assertIn("over_fetch", d["ruled_out"])
        # every firing class is reported; over_subscribed is marginal (mem_stalled band position 0.53 < 0.75), so the
        # primary is issue_bound
        self.assertEqual((d["primary"], d["classes"], d["marginal"]), ("issue_bound", ["over_subscribed", "issue_bound"],
                                                                       ["over_subscribed"]))
        self.assertAlmostEqual(d["band_position"]["mem_stalled"], 0.5274)
        self.assertIsNone(d["band_position"]["ea_req_per_kb"])      # references do not separate on it
        self.assertEqual(set(d["levers"]), {"over_subscribed", "issue_bound"})

    def test_issue_bound_needs_a_discriminator_the_stream_lacks(self):
        g = load()["summary"]["signatures"]
        b = bs.reference_bands(g["S16"], g["S64"])
        k = dict(g["S64"], issue_busy=0.45, mem_stalled=g["S16"]["mem_stalled"])   # wait_inst 0.41 like S64, stream VALU
        self.assertNotIn("issue_bound", bs.classify(k, g["S16"], b, 0.8, g["S64"])["classes"])
        k["valu_per_kb"] = 29.0
        self.assertIn("issue_bound", bs.classify(k, g["S16"], b, 0.8, g["S64"])["classes"])

    def test_over_subscribed_leans_on_mem_stalled_not_wait_any(self):
        g = load()["summary"]["signatures"]
        b = bs.reference_bands(g["S16"], g["S64"])
        k = dict(g["S16"], wait_any=g["S64"]["wait_any"])        # wait_any alone at S64: not over-subscribed
        self.assertNotIn("over_subscribed", bs.classify(k, g["S16"], b, 1.0, g["S64"])["classes"])
        k = dict(g["S16"], mem_stalled=g["S64"]["mem_stalled"])  # mem_stalled at S64, wait_any healthy: firm
        r = bs.classify(k, g["S16"], b, 1.0, g["S64"])
        self.assertEqual((r["primary"], r["marginal"]), ("over_subscribed", []))

    def test_wait_any_direction_from_refs(self):
        g = load()["summary"]["signatures"]
        b = bs.reference_bands(g["S16"], g["S64"])
        self.assertEqual(b["wait_any"]["direction"], -1)     # 0.9623 -> 0.5627: FALLS under over-subscription
        self.assertEqual(b["mem_stalled"]["direction"], 1)
        self.assertAlmostEqual(b["wait_any"]["mid"], 0.7625)
        # S64 passes its own test (the diag's `>= mid` leg made S64 fail it)
        self.assertTrue(bs.oversub_legs(g["S64"], b)["wait_any"])
        self.assertFalse(g["S64"]["wait_any"] >= b["wait_any"]["mid"])

    def test_reference_self_check_s64_is_oversub(self):
        g = load()["summary"]["signatures"]
        chk = bs.self_check(g["S16"], g["S64"])
        self.assertEqual((chk["ok"], chk["S16"], chk["S64"]), (True, "healthy_bw", "over_subscribed"))

    def test_incoherent_references(self):
        g = load()["summary"]["signatures"]
        swapped = bs.signature("v5i", g["v5i"], g["S64"], g["S16"])  # S16/S64 swapped
        self.assertEqual(swapped["class"], "reference_incoherent")
        flat = dict(g["S16"], mem_stalled=g["S16"]["mem_stalled"] + 0.01, wait_any=g["S16"]["wait_any"] - 0.01)
        d = bs.signature("v5i", g["v5i"], g["S16"], flat)
        self.assertEqual(d["class"], "reference_incoherent")
        self.assertIn("incoherent axes", d["rule"])

    def test_absent_counter_is_none_not_zero(self):
        c = {"FETCH_SIZE": 1024.0, "SQ_WAIT_ANY": 5.0}
        d = bs.derive(c, 1024 * 1024)
        self.assertIsNone(d["wait_any"])
        self.assertIsNone(d["mem_stalled"])
        g = load()["summary"]["signatures"]
        self.assertEqual(bs.signature("k", d, g["S16"], g["S64"])["class"], "harness")

    def test_memunit_zero_falls_back_to_ta(self):
        c = {"MemUnitBusy": 0.0, "MemUnitStalled": 0.0, "TA_BUSY_avr": 50.0, "TA_ADDR_STALLED_BY_TC_CYCLES_sum": 104 * 20.0,
             "C7.GRBM_GUI_ACTIVE": 100.0, "GRBM_GUI_ACTIVE": 1.0}
        d = bs.derive(c, 1024)
        self.assertEqual((d["mem_busy"], d["mem_stalled"]), (0.5, 0.2))
        self.assertEqual((d["sdk_mem_unit_busy"], d["sdk_mem_unit_stalled"]), (0.0, 0.0))

    def test_derive_matches_parse_counters(self):
        rows = [{"Dispatch_Id": str(i), "Kernel_Name": "k", "Counter_Name": n, "Counter_Value": v, "Grid_Size": "64"}
                for i in (1, 2) for n, v in (("SQ_WAIT_ANY", 30), ("SQ_WAVE_CYCLES", 100))]
        c, nd, _meta, fam, grids, _ps = bs.parse_counters(rows, lambda r: "main")
        self.assertEqual((nd, fam["main"], grids), (2, 2, [64]))
        self.assertAlmostEqual(bs.derive(c, 1024)["wait_any"], 0.3)

    def test_rejected_frac_futility(self):
        r = load()
        r["summary"]["rejected_frac"] = 0.4
        self.assertEqual(bs.from_result(r, "v5i")["class"], "harness")

    def test_healthy_and_not_in_flight(self):
        g = load()["summary"]["signatures"]
        s16 = g["S16"]
        self.assertEqual(bs.signature("x", s16, s16, g["S64"], gbps=1450, s16_gbps=1460.8)["class"], "healthy_bw")
        self.assertEqual(bs.signature("x", s16, s16, g["S64"], gbps=900, s16_gbps=1460.8)["class"], "not_in_flight")

    def test_signature_pass_inputs_drop_c3(self):
        self.assertNotIn("C3", gcs.SIGNATURE_PASSES)
        self.assertTrue(gcs.signature_pass_input("C7").startswith("pmc: TA_BUSY_avr"))


class CensusSignature(unittest.TestCase):
    def setUp(self):
        self.c = json.load(open(GOLDEN))
        self.k = "27b-q8_0|ffn_gate_up_v5i|17408x5120|n8"
        r = load()
        self.sig = bs.from_result(r, "v5i", gbps=r["summary"]["v5i_n8_geomean_gbps"])

    def test_golden_no_signature_byte_identical(self):
        def q8(us):
            return {"row": "fx-q8", "at": "2026-11-22T00:00:00Z", "op": {"cells_us": {
                "17408x5120:n8": {"V": us, "OFF": 104.8}, "6144x5120:n8": {"V": 35.59, "OFF": 39.7}}}}
        c = cu.empty_census()
        cu.apply(c, cu.extract(q8(90.41)), "fx-a", "a" * 64, "2026-11-22T00:00:00Z")
        cu.apply(c, cu.extract(q8(70.0)), "fx-b", "b" * 64, "2026-11-23T00:00:00Z")
        self.assertEqual(json.dumps(c, sort_keys=True, indent=1), open(GOLDEN).read())

    def test_apply_signature_warrant_and_rollup(self):
        lines = cu.apply_signature(self.c, {self.k: self.sig, "nope|x|1x1|n1": self.sig}, "ak-diag", "d" * 64,
                                   "2026-11-25T17:00:00Z", build="stream2:f4a48e00")
        s = self.c["records"][self.k]["signature"]
        self.assertEqual((s["class"], s["source_row"], s["result_sha256"], s["build"]),
                         ("issue_bound", "ak-diag", "d" * 64, "stream2:f4a48e00"))
        self.assertFalse(s["stale_vs_timing"])
        self.assertEqual((s["classes"], s["marginal"]), (["over_subscribed", "issue_bound"], ["over_subscribed"]))
        self.assertTrue(any("unknown record" in ln for ln in lines))
        self.assertNotIn("nope|x|1x1|n1", self.c["records"])
        surf = next(x for x in self.c["surfaces"] if self.k in x["census_rows"])
        self.assertEqual(surf["signature_classes"], {"issue_bound": 1})
        self.assertTrue(all("signature_classes" not in x for x in self.c["surfaces"] if self.k not in x["census_rows"]))

    def test_apply_signature_idempotent_on_sha(self):
        cu.apply_signature(self.c, {self.k: self.sig}, "ak-diag", "d" * 64, "t")
        self.assertEqual(cu.apply_signature(self.c, {self.k: dict(self.sig, **{"class": "over_fetch"})}, "ak-diag", "d" * 64, "t"), [])
        self.assertEqual(self.c["records"][self.k]["signature"]["class"], "issue_bound")
        self.assertNotIn("d" * 64, [a["sha256"] for a in self.c["applied"]])   # timing ledger untouched
        with self.assertRaises(ValueError):
            cu.apply_signature(self.c, {}, "r", "", "t")

    def test_signature_survives_timing_update_and_goes_stale(self):
        cu.apply_signature(self.c, {self.k: self.sig}, "ak-diag", "d" * 64, "t")
        obs = cu.extract({"row": "fx-q8", "at": "2026-11-26T00:00:00Z", "op": {"cells_us": {"17408x5120:n8": {"V": 50.0}}}})
        cu.apply(self.c, obs, "fx-c", "c" * 64, "2026-11-26T00:00:00Z")
        self.assertEqual(self.c["records"][self.k]["source_row"], "fx-c")
        s = self.c["records"][self.k]["signature"]
        self.assertEqual(s["class"], "issue_bound")
        self.assertTrue(s["stale_vs_timing"])


if __name__ == "__main__":
    unittest.main()
