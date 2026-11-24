"""census_families.py: family table coverage, weighted family ranking, per-path weights, unpriced knob arms, live census copy."""
import json
import os
import shutil
import tempfile
import unittest

from autokernel.loop import census_families as cf
from autokernel.loop import census_update as cu

LIVE = "/mnt/raid0/llm/tmp/ak-bw-census/mi210_bandwidth_census.json"


def S(path, op, saving, step=40.0):
    return {"path": path, "op": op, "saving_ms_per_step": saving, "step_ms": step}


FA27 = S("27b-df2-mi210", "gv_fattn + merge (verify FA)", 2.0)
FAA3 = S("a3b-mtp-mi210", "FA / gv (10 layers, f16 KV)", 1.5)
DENSE27 = S("27b-df2-mi210", "verify dense Q8 GEMV/GEMM (q8mf6, V5I)", 3.0)
MMID_A3 = S("a3b-mtp-mi210", "MUL_MAT_ID experts n=3", 1.0)


class Families(unittest.TestCase):
    def test_every_op_map_entry_has_a_family(self):
        self.assertEqual(cf.unmapped_op_map_entries(), [])
        for path, op in set(cu.OP_MAP.values()):
            self.assertNotEqual(cf.op_family(path, op), cf.UNFAMILIED, (path, op))

    def test_a_missing_family_is_detected(self):
        cu.OP_MAP[("zz", "zz_class")] = ("zz-path", "zz op")
        try:
            self.assertEqual(cf.unmapped_op_map_entries(), [("zz-path", "zz op")])
        finally:
            del cu.OP_MAP[("zz", "zz_class")]

    def test_family_sums_across_two_paths(self):
        r = {x["family"]: x for x in cf.rank([FA27, FAA3, DENSE27])}
        self.assertAlmostEqual(r["flash-attn-verify"]["score"], 3.5)
        self.assertEqual(r["flash-attn-verify"]["n_surfaces"], 2)
        self.assertEqual(cf.rank([FA27, FAA3, DENSE27])[0]["family"], "flash-attn-verify")  # 3.5 > dense 3.0

    def test_model_weights_reorder_the_rank(self):
        base = [x["family"] for x in cf.rank([FA27, FAA3, DENSE27])]
        w = [x["family"] for x in cf.rank([FA27, FAA3, DENSE27], {"a3b-q8_0": 0.0, "27b-q8_0": 1.0})]
        self.assertEqual(base[0], "flash-attn-verify")
        self.assertEqual(w[0], "dense-q8-gemv")  # fa drops to 2.0 < 3.0

    def test_weight_can_be_keyed_per_path_and_beats_the_model_key(self):
        sf = [FA27, FAA3, MMID_A3]
        # served-today vs next-GPU: A3B GPU counted at half weight, by path; the model key says 0 but the path key wins
        r = {x["family"]: x for x in cf.rank(sf, {"a3b-mtp-mi210": 0.5, "a3b-q8_0": 0.0})}
        self.assertAlmostEqual(r["flash-attn-verify"]["score"], 2.0 + 0.75)
        self.assertAlmostEqual(r["mmid-moe"]["score"], 0.5)
        self.assertEqual(cf.weight_of("a3b-mtp-mi210", {"a3b-q8_0": 0.3}), 0.3)  # model key alone reaches the path
        self.assertEqual(cf.weight_of("a3b-mtp-cpu", {"a3b-mtp-mi210": 0.0}), 1.0)  # other paths keep the default
        self.assertEqual(len(cf.rank(sf)), 2)  # A3B surfaces stay in the rank at default weight

    def test_unpriced_surface_and_unfamilied_op_are_visible(self):
        r = {x["family"]: x for x in cf.rank([S("27b-df2-mi210", "draft q8ip_permute", None), S("p", "novel op", 1.0)])}
        self.assertIsNone(r["act-quantize"]["score"])
        self.assertEqual(r["act-quantize"]["status"], "unpriced")
        self.assertEqual(r[cf.UNFAMILIED]["score"], 1.0)

    def test_unpriced_knob_arms(self):
        arms = [{"name": "np4"}, {"name": "ub512", "vs_control_pct": None},
                {"name": "fa-off", "path": "27b-df2-mi210", "vs_control_pct": 10.0}, {"name": "no-path", "vs_control_pct": 5.0}]
        r = {x["family"]: x for x in cf.rank([DENSE27], knob_arms=arms)}
        for n in ("np4", "ub512", "no-path"):  # no screened pct, or no step to price it against
            self.assertIsNone(r["serving-knob:" + n]["saving_ms_per_step"])
            self.assertEqual(r["serving-knob:" + n]["status"], "unpriced")
        self.assertAlmostEqual(r["serving-knob:fa-off"]["saving_ms_per_step"], 40 * (1 - 1 / 1.1), places=3)  # throughput +10 %
        self.assertEqual(cf.rank([DENSE27], knob_arms=arms)[-1]["status"], "unpriced")  # unpriced sort last

    def test_knob_pct_is_a_throughput_gain(self):
        r = {x["family"]: x for x in cf.rank([], knob_arms=[
            {"name": "up", "step_ms": 100.0, "vs_control_pct": 25.0},
            {"name": "down", "step_ms": 100.0, "vs_control_pct": -8.0}])}
        self.assertAlmostEqual(r["serving-knob:up"]["saving_ms_per_step"], 20.0)  # +25 % tok/s on 100 ms saves 20 ms, not 25
        self.assertEqual(r["serving-knob:down"]["saving_ms_per_step"], 0.0)  # negative clamps to 0 ...
        self.assertEqual(r["serving-knob:down"]["status"], "priced")  # ... and stays priced

    def test_live_census_copy(self):
        if not os.path.exists(LIVE):
            self.skipTest("live census absent")
        d = tempfile.mkdtemp()
        try:
            cp = os.path.join(d, "c.json")
            shutil.copy(LIVE, cp)
            surf = json.load(open(cp))["surfaces"]
        finally:
            shutil.rmtree(d)
        self.assertGreater(len(surf), 10)
        rows = cf.rank(surf)
        self.assertEqual([x["family"] for x in rows if x["family"] == cf.UNFAMILIED], [], "census op with no family")
        self.assertAlmostEqual(sum(x["score"] or 0 for x in rows), sum(s["saving_ms_per_step"] or 0 for s in surf), 2)
        self.assertEqual(sum(x["n_surfaces"] for x in rows), len(surf))
        self.assertTrue(any(x["family"] == "flash-attn-verify" and x["score"] for x in rows))
        self.assertTrue(any(m["path"] == "a3b-mtp-mi210" for x in rows for m in x["members"]))  # A3B stays ranked


if __name__ == "__main__":
    unittest.main()
