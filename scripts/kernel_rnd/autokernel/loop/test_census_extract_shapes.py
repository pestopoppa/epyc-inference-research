"""census_extract.py: GDN cells, DTREE rowcost / host-gap, rocprof trace by-class, CPU ubench tables (trimmed real RESULTs)."""
import json
import os
import tempfile
import unittest

from autokernel.loop import census_update as cu
from autokernel.loop.census_extract import extract, extract_cpu

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "census")


def load(name):
    return json.load(open(os.path.join(FX, name)))


def by(obs):
    return {(o["kernel_class"], o["shape"], o["n"]): o for o in obs}


class TimeOnly(unittest.TestCase):
    def check_time_only(self, obs):
        self.assertTrue(obs)
        for o in obs:
            self.assertIsNone(o["bytes_per_call"], o)  # never invent a byte count
            self.assertTrue(o["bytes_basis"].startswith("unknown"))
            self.assertGreater(o["us_per_call"], -1e9)

    def test_gdn_fused_arms_and_info_arm_skipped(self):
        obs = extract(load("gdn_fused.RESULT.json"))
        self.check_time_only(obs)
        self.assertEqual(len(obs), 6)  # 3 cells x (OFF, ON); us_onplain_info is an info column
        k = by(obs)
        self.assertEqual(k[("gdn_core:off", "sv128:ns1:nt12:K12:f32", 12)]["us_per_call"], 69.13)
        self.assertEqual(k[("gdn_core:on", "sv128:ns1:nt12:K12:f32", 12)]["us_per_call"], 44.7)
        self.assertNotIn("onplain_info", " ".join(o["kernel_class"] for o in obs))

    def test_gdn_dispatch_colwidth_verify_base(self):
        k = by(extract(load("gdn_dispatch.RESULT.json")))
        self.assertEqual(k[("gdn_core:r3", "sv128:ns4:nt12:K12:f32", 12)]["us_per_call"], 385.0)
        self.assertEqual(k[("gdn_core:disp", "sv128:ns4:nt12:K12:f32", 12)]["us_per_call"], 168.32)
        k = by(extract(load("gdn_colwidth.RESULT.json")))
        self.assertEqual(sorted(c for c, _, _ in k), ["gdn_core:c1", "gdn_core:c2", "gdn_core:c4"])
        k = by(extract(load("gdn_verify_base.RESULT.json")))
        self.assertEqual(k[("gdn_core:verify_base", "sv128:ns1:nt16:K16:bf16", 16)]["us_per_call"], 86.13)

    def test_dtree_rowcost_and_slope_match_program_md_lead(self):
        obs = extract(load("dtree_rowcost.RESULT.json"))
        self.check_time_only(obs)
        k = by(obs)
        self.assertEqual(k[("verify_ms:gdn", "T12", 12)]["us_per_call"], 4066.7)
        gdn, mm = k[("rowslope:gdn", "T8->12", 12)]["us_per_call"], k[("rowslope:mul_mat", "T8->12", 12)]["us_per_call"]
        self.assertAlmostEqual((gdn - mm) / 1e3, 0.254, places=3)  # documented: GDN T8->12 lead +0.254 ms/row
        self.assertEqual(k[("rowslope:launch_other", "T8->12", 12)]["us_per_call"], -28.2)  # negative slope kept as measured

    def test_host_gap_post_phase(self):
        k = by(extract(load("dtree_host_gap.RESULT.json")))
        self.assertEqual(k[("hostgap:post", "T12", 1)]["us_per_call"], 2688.5)  # documented: post T12 2.69 ms
        self.assertEqual(k[("hostgap:post", "L", 1)]["us_per_call"], 444.8)

    def test_trace_by_class_averages_reps_and_needs_a_model(self):
        res = load("trace_a3b_byclass.RESULT.json")
        obs = extract(res)
        k = {(o["kernel_class"], o["shape"]): o for o in obs}
        dense = k[("trace_verify:dense_mm", "OFF")]
        self.assertEqual(dense["reps"], 2)
        self.assertEqual(dense["model"], "a3b-q8_0")
        self.assertIn(("trace_draft:mmvq_ids", "OFF"), k)
        self.check_time_only(obs)
        res["row"] = "ak-c18-unknown-trace"  # no 27b/a3b tag in row/task: unattributable -> no record
        self.assertEqual([o for o in extract(res) if o["kernel_class"].startswith("trace_")], [])

    def test_getrows_ab_phase_times(self):
        k = by(extract(load("trace_getrows.RESULT.json")))
        self.assertEqual(k[("e2e_draft_ms", "OFF", 0)]["model"], "27b-q8_0")
        self.assertAlmostEqual(k[("e2e_draft_ms", "OFF", 0)]["us_per_call"], 2455.982, places=2)
        self.assertAlmostEqual(k[("e2e_draft_ms", "ON", 0)]["us_per_call"], 2013.421, places=2)

    def test_existing_bandwidth_trace_rows_unchanged(self):
        obs = extract(load("trace_27b.RESULT.json"))
        self.assertTrue([o for o in obs if o["bytes_per_call"]])


class Ubench(unittest.TestCase):
    def test_mmid_base_only_with_row_bytes_and_quiet_guard(self):
        obs = extract_cpu(load("ubench_mmid.RESULT.json"))
        self.assertEqual(sorted((o["kernel_class"], o["n"]) for o in obs),
                         [("MUL_MAT:shared_expert@ubench", 1), ("MUL_MAT_ID:exps_gate_up@ubench", 1), ("MUL_MAT_ID:exps_gate_up@ubench", 3)])
        o = next(o for o in obs if o["n"] == 1 and "gate_up" in o["kernel_class"])
        self.assertEqual((o["bytes_per_call"], o["us_per_call"], o["calls_per_round"]), (17825792, 175.41, 40))
        self.assertEqual((o["device"], o["config"], o["shape"]), ("cpu", "quarter-24t", "quarter-24t-ubench"))
        self.assertTrue(all(x.get("quiet_host") for x in obs))  # probe p50 0.74 %, drift 0.44 %
        self.assertAlmostEqual(cu.gbps(o["bytes_per_call"], o["us_per_call"]), 101.6, places=1)  # row table: 100.5 mean, 108.05 ref

    def test_shexp_noisy_host_is_not_quiet_and_down_arm_kept(self):
        obs = extract_cpu(load("ubench_shexp.RESULT.json"))
        self.assertEqual(sorted(o["kernel_class"] for o in obs), ["MUL_MAT:shared_expert@ubench", "MUL_MAT:shexp_down@ubench"])
        self.assertFalse(any(o.get("quiet_host") for o in obs))  # SMT probe p50 12.61 %, post-drift 11 %
        down = next(o for o in obs if "down" in o["kernel_class"])
        self.assertEqual((down["bytes_per_call"], down["us_per_call"], down["n"]), (1114112, 17.03, 2))

    def test_invalid_arm_dropped(self):
        res = load("ubench_mmid.RESULT.json")
        res["arm_invalid"] = ["shexp|base"]
        self.assertEqual([o for o in extract_cpu(res) if "shared_expert" in o["kernel_class"]], [])


class Apply(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.c = os.path.join(self.d, "c.json")

    def run_(self, name):
        return cu.run(os.path.join(FX, name), None, self.c, None, None, None)

    def test_time_only_stored_without_gbps_rank_or_seed_and_idempotent(self):
        rep = self.run_("gdn_fused.RESULT.json")
        self.assertEqual((rep["observations"], rep["journal_lines"]), (6, 1))
        c = json.load(open(self.c))
        self.assertEqual(c["records"], {})  # nothing enters the bandwidth ranking
        self.assertEqual(len(c["time_only"]), 6)
        r = next(iter(c["time_only"].values()))
        self.assertIsNone(r["bytes_per_call"])
        self.assertNotIn("gbps", r)
        self.assertEqual(self.run_("gdn_fused.RESULT.json")["journal_lines"], 0)  # same sha: no-op

    def test_noise_rule_and_history_on_time_only(self):
        self.run_("gdn_fused.RESULT.json")
        c = json.load(open(self.c))
        o = dict(next(iter(c["time_only"].values())))
        self.assertFalse(cu.apply_time_only(c, o, "r2", "s2", "t"))  # unchanged time: not a move
        self.assertTrue(cu.apply_time_only(c, dict(o, us_per_call=o["us_per_call"] * 2), "r2", "s2", "t"))
        self.assertEqual(len(c["time_only"][cu.rid(o)]["history"]), 2)

    def test_ubench_enters_cpu_ranking_with_its_own_rid(self):
        self.run_("ubench_mmid.RESULT.json")
        c = json.load(open(self.c))
        k = "cpu|a3b-q8_0-cpu|MUL_MAT_ID:exps_gate_up@ubench|quarter-24t-ubench|n1"
        self.assertIn(k, c["records"])
        self.assertTrue(c["records"][k]["quiet_host"])


class CpuConfigThreads(unittest.TestCase):
    """-t 96 (NUMA_FULL) maps to the full-host ceiling key (520 GB/s), not an unceilinged '96t'."""
    def test_thread_counts(self):
        from autokernel.loop.census_extract import cpu_config
        for t, want in ((24, "quarter-24t"), (48, "half-48t"), (72, "q0q2"), (96, "full-host"), (12, "12t")):
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "servers"))
                json.dump({"argv": ["llama-server", "-t", str(t)]}, open(os.path.join(d, "servers", "c-L1.argv.json"), "w"))
                self.assertEqual(cpu_config(os.path.join(d, "RESULT.json"), "c"), want)
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["full-host"], 520.0)
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["q0q2"], 387.0)


class VattnIpb(unittest.TestCase):
    """Verify-attention profiler census RESULT -> one gv_fattn_q8kv observation per (kv, nq), bytes from kv_bytes."""
    def test_shapes(self):
        from autokernel.loop.census_extract import extract
        res = {"row": "ak-rka-ipb-census-gpu-20261124-r2", "complete": True, "shapes": [
            {"kv": 8192, "nq": 1, "kv_bytes": 17825792.0, "dur_us": 118.07, "achieved_gbps": 151.0, "valu_util": 0.085, "counters": {"SQ_WAVES": 512.0}},
            {"kv": 8192, "nq": 8, "kv_bytes": 17825792.0, "dur_us": 0, "achieved_gbps": 0}]}
        o = [x for x in extract(res) if x["kernel_class"] == "gv_fattn_q8kv"]
        self.assertEqual(len(o), 1)
        self.assertEqual((o[0]["model"], o[0]["shape"], o[0]["n"], o[0]["bytes_per_call"]), ("27b-q8_0", "kv8192", 1, 17825792))
        self.assertEqual(extract(dict(res, complete=False)), [])
        self.assertIn(("27b-q8_0", "gv_fattn_q8kv"), cu.OP_MAP)


class VerifyRoofline(unittest.TestCase):
    """27B verify-kernel roofline census (trimmed real RESULT: ctx 8192 at width 1 and 8) -> one observation per
    (kernel class, ctx, width); bytes/us/calls are the row's own."""
    res = load("verify_roofline.RESULT.json")

    def obs(self):
        from autokernel.loop.census_extract import extract_verify_roofline
        return extract_verify_roofline(self.res)

    def test_per_class_ctx_width_with_the_rows_own_bytes_us_calls(self):
        o = self.obs()
        keys = [(x["kernel_class"], x["shape"], x["n"]) for x in o]
        self.assertEqual(len(keys), len(set(keys)))                       # rid-unique: no observation overwrites another
        self.assertEqual({x["n"] for x in o}, {1, 8})
        k = {(x["kernel_class"], x["shape"], x["n"]): x for x in o}
        lm = k[("vk_lm_head", "ctx8192", 1)]                              # the one full-vocab GEMV: 1352 MB, 1.065 ms, 1 call
        self.assertEqual(lm["calls_per_round"], 1)
        self.assertAlmostEqual(lm["bytes_per_call"], 1351.98e6, delta=0.1e6)
        self.assertAlmostEqual(lm["us_per_call"], 1065.0, delta=1.0)
        fa = k[("gv_fattn_q8kv", "ctx8192", 1)]                           # 315.9 MB / 16 calls, 1.9385 ms / 16 calls
        self.assertEqual(fa["calls_per_round"], 16)
        self.assertAlmostEqual(fa["us_per_call"], 1938.5 / 16, places=2)
        self.assertAlmostEqual(fa["bytes_per_call"], 315.9e6 / 16, delta=0.1e6)
        dense = [x for x in o if x["kernel_class"] == "vk_dense_mmvq" and x["n"] == 1]
        self.assertEqual(sum(x["calls_per_round"] for x in dense), 64 + 80 + 48 * 3 + 96 + 16 + 32)   # every mmvq launch but the lm_head
        self.assertTrue(any(x["kernel_class"] == "vk_dense_q8wv" and x["n"] == 8 for x in o))
        for x in o:                                                       # bytes known, never invented; GB/s derivable
            self.assertGreater(x["bytes_per_call"], 0)
            self.assertGreater(x["us_per_call"], 0)

    def test_tail_and_actquant_are_one_pooled_observation_per_shape(self):
        o = self.obs()
        for cls in ("vk_launch_tail", "vk_actquant"):
            rows = [x for x in o if x["kernel_class"] == cls]
            self.assertEqual(sorted((x["shape"], x["n"]) for x in rows), [("ctx8192", 1), ("ctx8192", 8)])
        tail1 = next(x for x in o if x["kernel_class"] == "vk_launch_tail" and x["n"] == 1)
        shape1 = next(s for s in self.res["shapes"] if s["w"] == 1)
        calls = sum(k["calls"] for k in shape1["kernels"] if k["short"] not in (
            "mul_mat_vec_q", "gv_fattn_kernel", "gv_merge_kernel", "gated_delta_net_cuda", "q8sk_mul_mat", "quantize_q8_1_1d", "quantize_q8_1"))
        self.assertEqual(tail1["calls_per_round"], calls)

    def test_gv_fattn_calls_come_from_one_constant_the_row_fills(self):
        from autokernel.loop.census_extract import GV_FATTN_CALLS_PER_STEP, extract
        fa = [x for x in self.obs() if x["kernel_class"] == "gv_fattn_q8kv"]
        self.assertTrue(fa)
        self.assertEqual({x["calls_per_round"] for x in fa}, {GV_FATTN_CALLS_PER_STEP})   # the row's own count == the constant
        res = {"row": "ak-rka-ipb-census-gpu-20261124-r2", "complete": True, "shapes": [
            {"kv": 8192, "nq": 1, "kv_bytes": 17825792.0, "dur_us": 118.07}]}
        rka = [x for x in extract(res) if x["kernel_class"] == "gv_fattn_q8kv"]
        self.assertEqual([x["calls_per_round"] for x in rka], [16])                        # was 1: one launch per full-attention layer

    def test_every_kernel_class_maps_to_an_op_with_a_family(self):
        from autokernel.loop import census_families as cf
        for cls in {x["kernel_class"] for x in self.obs()}:
            m = cu.OP_MAP.get(("27b-q8_0", cls))
            self.assertIsNotNone(m, cls)
            self.assertNotEqual(cf.op_family(*m), cf.UNFAMILIED, (cls, m))

    def test_unrelated_or_incomplete_results_yield_nothing(self):
        from autokernel.loop.census_extract import extract_verify_roofline
        self.assertEqual(extract_verify_roofline(dict(self.res, complete=False)), [])
        self.assertEqual(extract_verify_roofline(dict(self.res, row="ak-rka-ipb-census-gpu-20261124-r2")), [])
        self.assertEqual(extract_verify_roofline({"shapes": []}), [])

    def test_applies_to_a_census_copy_with_all_records_resolving(self):
        c = cu.empty_census()
        lines = cu.apply(c, self.obs(), self.res["row"], "a" * 64, "2026-11-24T23:17:39Z", "at")
        self.assertTrue(lines)
        self.assertEqual(len(c["records"]), len(self.obs()))
        self.assertTrue(all(cu.op_of(r) is not None for r in c["records"].values()))


if __name__ == "__main__":
    unittest.main()
