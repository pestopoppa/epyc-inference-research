"""census_update.py: idempotency, threshold flagging, seed linking (fixture RESULT.json files, no GPU)."""
import json
import os
import shutil
import tempfile
import unittest

from autokernel.loop import census_update as cu

def q8_result(us_n8: float, at="2026-11-22T00:00:00Z"):
    return {"row": "fx-q8", "at": at, "op": {"cells_us": {
        "17408x5120:n8": {"V": us_n8, "OFF": 104.8},
        "6144x5120:n8": {"V": 35.59, "OFF": 39.7},
        "6144x5120:n2": {"V": 30.0}}}}

def mmid_result(us):
    return {"row": "fx-mmid", "at": "2026-11-22T01:00:00Z", "op": {"table": {
        "gate_up:n4": {"us_on": us, "gbps_on": 848.6}}}}

class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.census = os.path.join(self.d, "c.json")
        self.prog = os.path.join(self.d, "program.md")
        self.doc = os.path.join(self.d, "doc.md")
        open(self.prog, "w").write("# program\n\n## Other\n\ntext\n")
        open(self.doc, "w").write("# doc\n")

    def res(self, name, obj):
        p = os.path.join(self.d, name)
        json.dump(obj, open(p, "w"))
        return p

    def run_(self, p, row=None):
        return cu.run(p, row, self.census, self.prog, self.doc, os.path.join(self.d, "stubs"))

class Idempotency(Base):
    def test_same_result_twice_changes_nothing(self):
        p = self.res("r.json", q8_result(90.41))
        r1 = self.run_(p)
        snap = [open(f).read() for f in (self.census, self.prog, self.doc)]
        r2 = self.run_(p)
        self.assertTrue(r1["changed"])
        self.assertFalse(r2["changed"])
        self.assertEqual(r2["journal_lines"], 0)
        self.assertEqual(snap, [open(f).read() for f in (self.census, self.prog, self.doc)])

    def test_within_noise_is_not_a_move(self):
        self.run_(self.res("a.json", q8_result(90.41)))
        r = self.run_(self.res("b.json", q8_result(91.0, at="x")))
        self.assertEqual(r["journal_lines"], 0)

    def test_beyond_noise_moves_and_journals_once(self):
        self.run_(self.res("a.json", q8_result(90.41)))
        r = self.run_(self.res("b.json", q8_result(75.0, at="y")))
        self.assertEqual(r["journal_lines"], 1)
        txt = open(self.prog).read()
        self.assertIn(cu.JOURNAL_HEAD, txt)
        self.assertIn("1047 GB/s", txt)
        self.assertIn("1263 GB/s", txt)
        self.assertLess(txt.index("## Other"), txt.index(cu.JOURNAL_HEAD))
        h = json.load(open(self.census))["records"]["27b-q8_0|ffn_gate_up_v5i|17408x5120|n8"]["history"]
        self.assertEqual([g for _, g in h], [1047.4, 1262.7])

class Threshold(Base):
    def test_below_floor_is_open_surface_and_stubbed(self):
        self.run_(self.res("a.json", q8_result(90.41)))
        c = json.load(open(self.census))
        gate = c["records"]["27b-q8_0|gdn_attn_gate_v5i|6144x5120|n8"]
        ffn = c["records"]["27b-q8_0|ffn_gate_up_v5i|17408x5120|n8"]
        self.assertEqual(gate["status"], "open_surface")
        self.assertEqual(ffn["status"], "ok")
        self.assertIn(gate, [c["records"][s["record"]] for s in c["seed_stubs"]])
        self.assertEqual(len(os.listdir(os.path.join(self.d, "stubs"))), 1)
        self.assertNotIn("n2", "".join(c["records"]))  # n=2 is not a served n

    def test_doc_block_regenerated_in_place(self):
        self.run_(self.res("a.json", q8_result(90.41)))
        self.run_(self.res("b.json", q8_result(75.0, at="z")))
        txt = open(self.doc).read()
        self.assertEqual(txt.count(cu.BEGIN), 1)
        self.assertIn("open_surface", txt)

class Seeds(Base):
    def test_linked_seed_suppresses_stub_and_is_marked_done(self):
        c = cu.empty_census()
        c["seeds"]["BW-MMID"] = {"match": {"model": "a3b-q8_0", "kernel_class": "mmid_gate_up"},
                                 "target_gbps": 1000, "status": "open"}
        json.dump(c, open(self.census, "w"))
        self.run_(self.res("a.json", mmid_result(40.2)))
        c = json.load(open(self.census))
        self.assertEqual(c["seed_stubs"], [])
        self.assertEqual(c["records"]["a3b-q8_0|mmid_gate_up|gate_up|n4"]["seed"], "BW-MMID")
        self.assertEqual(c["seeds"]["BW-MMID"]["status"], "open")
        r = self.run_(self.res("b.json", mmid_result(30.0)))  # 34.1 MB / 30 us = 1137 GB/s
        c = json.load(open(self.census))
        self.assertEqual(c["seeds"]["BW-MMID"]["status"], "done")
        self.assertIn("seed BW-MMID DONE", open(self.prog).read())
        self.assertEqual(r["journal_lines"], 2)

    def test_unknown_result_shape_yields_nothing(self):
        r = self.run_(self.res("a.json", {"verdict": "KEEP"}))
        self.assertEqual(r["observations"], 0)
        self.assertEqual(r["journal_lines"], 0)

def cpu_result(gate_up_ms=8.8, rid="a3b-cpu-gap-census-fx"):
    fam = [{"family": "MUL_MAT:gdn_proj", "share_pct": 25.691, "ms_per_timing_step": 10.4335},
           {"family": "MUL_MAT_ID:exps_gate_up", "share_pct": 18.682, "ms_per_timing_step": gate_up_ms},
           {"family": "FLASH_ATTN", "share_pct": 5.2, "ms_per_timing_step": 2.11}]
    return {"id": rid, "updated_utc": "2026-11-22T00:57:21Z", "configs": {
        "plain-greedy": {"mean_verify_rows": 1.0, "ranked_families": fam},
        "mtp-greedy": {"mean_verify_rows": 4.9, "ranked_families": fam}}}

class Cpu(Base):
    def cpu_res(self, name, obj):
        d = os.path.join(self.d, name)
        os.makedirs(os.path.join(d, "servers"))
        json.dump({"argv": ["llama-server", "-t", "24", "-tb", "24"]},
                  open(os.path.join(d, "servers", "plain-greedy-L1.argv.json"), "w"))
        p = os.path.join(d, "RESULT.json")
        json.dump(obj, open(p, "w"))
        return p

    def test_cpu_records_flag_below_75pct_of_measured_ceiling(self):
        self.run_(self.cpu_res("a", cpu_result()))
        c = json.load(open(self.census))
        gdn = c["records"]["cpu|a3b-q8_0-cpu|MUL_MAT:gdn_proj|quarter-24t|n1"]
        gu = c["records"]["cpu|a3b-q8_0-cpu|MUL_MAT_ID:exps_gate_up|quarter-24t|n1"]
        self.assertEqual(gdn["ceiling_gbps"], 112.9)
        self.assertEqual(gdn["status"], "ok")            # 102.5 GB/s = 91% of ceiling
        self.assertEqual(gu["status"], "open_surface")   # 81.0 GB/s = 72% of ceiling (< 0.75 x 112.9 = 84.7)
        self.assertNotIn("FLASH_ATTN", "".join(c["records"]))   # no static bytes -> no record
        self.assertEqual(sum("cpu|" in k for k in c["records"]), 2)  # MTP config skipped
        self.assertIn("### CPU (EPYC 9655", open(self.doc).read())

    def test_record_without_share_renders(self):
        c = cu.empty_census()
        c["records"]["cpu|m|k|quarter-24t|n1"] = {"device": "cpu", "model": "m", "kernel_class": "k", "shape": "quarter-24t",
            "config": "quarter-24t", "n": 1, "bytes_per_call": 5.7e9, "us_per_call": 94000.0, "calls_per_round": 1,
            "share_of_round": None, "source_row": "doc"}
        cu.rerank(c)
        self.assertIn("| doc |".replace("doc", "`doc`"), cu.table_md(c))

    def test_flag_frac_is_a_tunable_parameter(self):
        c = cu.empty_census()
        c["cpu"]["flag_frac"] = 0.70
        json.dump(c, open(self.census, "w"))
        self.run_(self.cpu_res("a", cpu_result()))
        c = json.load(open(self.census))
        self.assertEqual(c["records"]["cpu|a3b-q8_0-cpu|MUL_MAT_ID:exps_gate_up|quarter-24t|n1"]["status"], "ok")

    def test_cpu_seed_done_and_idempotent(self):
        c = cu.empty_census()
        c["seeds"]["CPU-A3B-MMID"] = {"device": "cpu", "match": {"model": "a3b-q8_0-cpu",
                                      "kernel_class": "MUL_MAT_ID:exps_gate_up"}, "target_gbps": 100, "status": "open"}
        json.dump(c, open(self.census, "w"))
        p = self.cpu_res("a", cpu_result())
        self.run_(p)
        self.assertEqual(json.load(open(self.census))["seed_stubs"], [])
        self.assertFalse(self.run_(p)["changed"])
        self.run_(self.cpu_res("b", cpu_result(gate_up_ms=6.5)))   # 109.7 GB/s >= 100
        self.assertEqual(json.load(open(self.census))["seeds"]["CPU-A3B-MMID"]["status"], "done")

if __name__ == "__main__":
    unittest.main()

class UnitRule(unittest.TestCase):
    def test_compute_bound_in_gbps_rejected(self):
        rec = {"bytes_per_call": 1_000_000, "flops_per_call": 5e9, "unit": "GB/s"}  # AI 5000 >> ridge
        self.assertTrue(cu.validate_unit(rec))
        self.assertEqual(cu.unit_for(5e9, 1e6), "TFLOP/s")

    def test_ok_cases(self):
        self.assertEqual(cu.validate_unit({"bytes_per_call": 1e6, "flops_per_call": 5e9, "unit": "TFLOP/s"}), [])
        self.assertEqual(cu.validate_unit({"bytes_per_call": 1e9, "flops_per_call": 2e9, "unit": "GB/s"}), [])
        self.assertEqual(cu.validate_unit({"bytes_per_call": 1e9}), [])  # old row, unit null
        self.assertTrue(cu.validate_unit({"unit": "GBps"}))

    def test_apply_rejects_and_stamps_unit(self):
        c = cu.empty_census()
        bad = {"model": "m", "kernel_class": "k", "shape": "s", "n": 8, "arm": "V", "bytes_per_call": 10**6,
               "us_per_call": 5.0, "calls_per_round": 1, "flops_per_call": 5e9, "unit": "GB/s"}
        with self.assertRaises(ValueError):
            cu.apply(c, [bad], "r", "x" * 64, "t")
        good = {k: v for k, v in bad.items() if k not in ("flops_per_call", "unit")}
        cu.apply(c, [good], "r", "y" * 64, "t")
        self.assertEqual(next(iter(c["records"].values()))["unit"], "GB/s")

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "census")

class CpuCeiling(unittest.TestCase):
    """AK-CPU-CEIL-MEASURED: quarter-24t ceiling is the measured 112.9 GB/s, not the derived 129.9."""

    def test_quarter_ceiling_is_measured_with_provenance(self):
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["quarter-24t"], 112.9)
        src = cu.CPU_DEFAULTS["ceiling_source"]["quarter-24t"]
        self.assertTrue(src.startswith("measured:"))
        self.assertIn("AK-CPU-QUARTER-READCEIL", src)
        self.assertIn("9e379493", src)
        self.assertEqual(cu.empty_census()["cpu"]["ceiling_gbps"]["quarter-24t"], 112.9)
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["half-48t"], 259.8)  # untouched

    def test_open_surface_threshold_is_75pct_of_112_9(self):
        c = cu.empty_census()
        for name, g in (("below", 84.6), ("above", 84.8)):
            c["records"][f"cpu|m|{name}|quarter-24t|n1"] = {
                "device": "cpu", "model": "m", "kernel_class": name, "shape": "quarter-24t", "config": "quarter-24t",
                "n": 1, "bytes_per_call": g * 1e6, "us_per_call": 1e3, "calls_per_round": 1, "source_row": "fx"}
        cu.rerank(c)
        self.assertEqual(c["records"]["cpu|m|below|quarter-24t|n1"]["status"], "open_surface")
        self.assertEqual(c["records"]["cpu|m|above|quarter-24t|n1"]["status"], "ok")

    def test_committed_census_is_reranked_against_measured_ceiling(self):
        c = json.load(open(cu.CENSUS))
        self.assertEqual(c["cpu"]["ceiling_gbps"]["quarter-24t"], 112.9)
        self.assertEqual(c["cpu"]["ceiling_source"]["quarter-24t"], cu.CPU_DEFAULTS["ceiling_source"]["quarter-24t"])
        q = [r for r in c["records"].values() if r.get("device") == "cpu" and r.get("config") == "quarter-24t"]
        self.assertTrue(q)
        self.assertTrue(all(r["ceiling_gbps"] == 112.9 for r in q))
        self.assertTrue(all(r["status"] == ("open_surface" if r["gbps"] < 0.75 * 112.9 else "ok") for r in q))


class RealResultFixtures(Base):
    """Trimmed copies of real RESULT.json files from /mnt/raid0/llm/tmp (families/summary kept, bulk dropped)."""

    def test_q38fn_extract_cpu_reads_model_key_bytes_and_depth(self):
        p = os.path.join(FX, "q38fn", "RESULT.json")
        obs = cu.extract_cpu(json.load(open(p)), p)
        self.assertTrue(obs)
        self.assertEqual({o["model"] for o in obs}, {"q38fn-ud-iq4xs-cpu"})
        self.assertEqual({o["shape"] for o in obs}, {"quarter-24t-d1k", "quarter-24t-d32k"})  # plain configs only
        g = [o for o in obs if o["kernel_class"] == "MUL_MAT:gdn_proj" and o["shape"].endswith("d1k")][0]
        self.assertEqual(g["bytes_per_call"], 2247229440)       # the row's own bytes_per_step, not the A3B table
        self.assertAlmostEqual(g["us_per_call"], 26899.2, places=1)
        self.assertEqual(rid_of(g), "cpu|q38fn-ud-iq4xs-cpu|MUL_MAT:gdn_proj|quarter-24t-d1k|n1")

    def test_q38fn_folds_and_ranks_against_measured_ceiling(self):
        p = os.path.join(FX, "q38fn", "RESULT.json")
        self.run_(p)
        r = json.load(open(self.census))["records"]["cpu|q38fn-ud-iq4xs-cpu|MUL_MAT:gdn_proj|quarter-24t-d1k|n1"]
        self.assertEqual(r["ceiling_gbps"], 112.9)
        self.assertEqual(r["frac_of_ceiling"], round(83.5 / 112.9, 3))
        self.assertEqual(r["status"], "open_surface")   # 83.5 GB/s < 0.75 x 112.9 = 84.7

    def test_a3b_r2_still_folds_with_static_bytes(self):
        p = os.path.join(FX, "a3b_r2", "RESULT.json")
        obs = cu.extract_cpu(json.load(open(p)), p)
        self.assertEqual({o["model"] for o in obs}, {"a3b-q8_0-cpu"})
        gu = [o for o in obs if o["kernel_class"] == "MUL_MAT_ID:exps_gate_up"][0]
        self.assertEqual(gu["bytes_per_call"], 713_000_000)
        self.assertEqual(gu["shape"], "quarter-24t")

    def test_unknown_model_id_yields_no_cpu_records(self):
        p = os.path.join(FX, "a3b_r2", "RESULT.json")
        res = json.load(open(p))
        res["id"] = "something-else-20261122"
        self.assertEqual(cu.extract_cpu(res, p), [])
        res["model_key"] = "q38fn-iq4xs-cpu"  # explicit model_key resolves it
        self.assertEqual(cu.extract_cpu(res, p), [])  # q38fn model but A3B families carry no bytes_per_step: no guess

    def test_trace_27b_tbo_result_is_extracted(self):
        res = json.load(open(os.path.join(FX, "trace_27b.RESULT.json")))
        obs = cu.extract(res)
        self.assertTrue(obs)
        kv = {o["n"]: o for o in obs if o["kernel_class"] == "attn_kv_v5i"}
        self.assertEqual(kv[1]["bytes_per_call"], 5570560)
        self.assertEqual(kv[1]["calls_per_round"], 32)
        self.assertAlmostEqual(cu.gbps(kv[1]["bytes_per_call"], kv[1]["us_per_call"]), 780.7, delta=0.2)
        lm = [o for o in obs if o["kernel_class"] == "lm_head_v5i" and o["n"] == 16][0]
        self.assertEqual(lm["bytes_per_call"], 1350860800)
        self.assertEqual({o["n"] for o in obs} - cu.SERVED_N, set())   # served n only

    def test_trace_a3b_dense_aggregate_is_extracted(self):
        res = json.load(open(os.path.join(FX, "trace_a3b.RESULT.json")))
        obs = {o["kernel_class"]: o for o in cu.extract(res)}
        self.assertEqual(set(obs), {"dense_verify_aggregate", "dense_verify_aggregate_v5i"})
        self.assertAlmostEqual(cu.gbps(obs["dense_verify_aggregate"]["bytes_per_call"],
                                       obs["dense_verify_aggregate"]["us_per_call"]), 404.9, delta=0.1)
        self.assertEqual(obs["dense_verify_aggregate_v5i"]["n"], 4)

    def test_trace_result_folds_idempotently(self):
        p = os.path.join(FX, "trace_27b.RESULT.json")
        r1 = self.run_(p)
        self.assertGreater(r1["observations"], 0)
        before = open(self.census).read()
        self.run_(p)
        self.assertEqual(open(self.census).read(), before)

    def _diag(self, quiet_l2=True, ms=28.0):
        unit = lambda q: {"guard": {"quiet": q}}
        pf = {"MUL_MAT:gdn_proj": {"plain-greedy-d1k-L1": {"gbps": 92.85, "ms_per_step": 24.2},
                                   "plain-greedy-d32k-L1": {"gbps": 80.0, "ms_per_step": ms},
                                   "plain-greedy-d32k-L2": {"gbps": 95.0, "ms_per_step": ms - 4.4}}}
        return {"id": "ak-q38fn-cpu-d32k-gemv-diag-fx", "status": "complete", "updated_utc": "2026-11-23T08:00:00Z",
                "evidence": {"gemv_gbps_d1k": 87.4, "gemv_gbps_d32k_mean": 84.9}, "per_family_by_unit": pf,
                "units": {"plain-greedy-d1k-L1": unit(True), "plain-greedy-d32k-L1": unit(True),
                          "plain-greedy-d32k-L2": unit(quiet_l2)}}

    def test_quiet_host_supersedes_contended_record_for_same_op_shape(self):
        self.run_(os.path.join(FX, "q38fn", "RESULT.json"), row="census")  # unguarded: d32k gdn_proj 36.7 GB/s
        k32, k1 = ("cpu|q38fn-ud-iq4xs-cpu|MUL_MAT:gdn_proj|quarter-24t-d%s|n1" % d for d in ("32k", "1k"))
        old = json.load(open(self.census))["records"]
        p = self.res("diag.json", self._diag())
        os.makedirs(os.path.join(self.d, "servers"))
        shutil.copy(os.path.join(FX, "q38fn", "servers", "plain-greedy-d32k-L1.argv.json"), os.path.join(self.d, "servers"))
        self.assertEqual([o["shape"] for o in cu.extract_cpu(self._diag(), p)], ["quarter-24t-d32k"])  # subject depth only
        self.run_(p, row="diag")
        recs = json.load(open(self.census))["records"]
        self.assertTrue(recs[k32]["quiet_host"])
        self.assertAlmostEqual(recs[k32]["us_per_call"], 25800.0, places=0)  # mean of the two replicate ms/step
        self.assertEqual(recs[k32]["superseded"][0]["validity"], "contention_invalid")
        self.assertEqual(recs[k32]["superseded"][0]["source_row"], old[k32]["source_row"])
        self.assertEqual(recs[k32]["superseded"][0]["superseded_by"]["row"], "diag")
        self.assertEqual(recs[k1], old[k1] | {"rank": recs[k1]["rank"]} | {x: recs[k1][x] for x in recs[k1] if x not in old[k1]})  # d1k untouched
        self.assertIn("supersedes contention_invalid", open(self.prog).read())
        c = json.load(open(self.census))  # a later unguarded observation of the same key never displaces the quiet one
        cu.apply(c, [dict(cu.extract_cpu(self._diag(), p)[0], quiet_host=False, us_per_call=99999.0)], "late", "f" * 64, "z")
        self.assertEqual(c["records"][k32]["source_row"], "diag")

    def test_non_quiet_replicate_is_dropped(self):
        p = self.res("diag.json", self._diag(quiet_l2=False))
        os.makedirs(os.path.join(self.d, "servers"))
        shutil.copy(os.path.join(FX, "q38fn", "servers", "plain-greedy-d32k-L1.argv.json"), os.path.join(self.d, "servers"))
        self.assertEqual(cu.extract_cpu(self._diag(quiet_l2=False), p)[0]["replicates"], 1)

    def _idx_diag(self, with_prof=True):
        """d32k diag RESULT with an indexer family; servers/<unit>.prof.json carries 12 bf16 weight nodes (0.4 ms) and 12 f32 score nodes (0.6 ms)."""
        res = self._diag()
        res["per_family_by_unit"] = {"MUL_MAT:indexer": {u: {"gbps": 39.3216e6 / (1000 * 1e3) / 1.0, "ms_per_step": 1.0}
                                                         for u in ("plain-greedy-d32k-L1", "plain-greedy-d32k-L2")}}
        p = self.res("diag.json", res)
        os.makedirs(os.path.join(self.d, "servers"))
        shutil.copy(os.path.join(FX, "q38fn", "servers", "plain-greedy-d32k-L1.argv.json"), os.path.join(self.d, "servers"))
        node = lambda t, ne, w: {"op": "MUL_MAT", "src0_type": t, "src0_ne": ne, "wall_us": w}
        nodes = [node("bf16", [2560, 512, 1], 400 / 12)] * 12 + [node("f32", [128, 64, 1], 600 / 12)] * 12 + [node("f32", [2560, 512, 1], 9e3)]
        for u in ("plain-greedy-d32k-L1", "plain-greedy-d32k-L2"):
            if with_prof:
                json.dump({"nodes": nodes}, open(os.path.join(self.d, "servers", u + ".prof.json"), "w"))
        return res, p

    def test_indexer_family_splits_into_weight_bandwidth_and_time_only_score(self):
        res, p = self._idx_diag()
        w, s = cu.extract_cpu(res, p)
        self.assertEqual((w["kernel_class"], s["kernel_class"]), ("MUL_MAT:indexer_weight", "MUL_MAT:indexer_score"))
        self.assertAlmostEqual(w["us_per_call"], 400.0, places=0)  # 40% weight share of 1.0 ms
        self.assertEqual(w["bytes_per_call"], 39321600)  # weight bytes keep the counted basis
        self.assertAlmostEqual(w["bytes_per_call"] / w["us_per_call"] / 1e3, 98.3, places=1)
        self.assertIsNone(s["bytes_per_call"])  # score: time-only, never a bandwidth record
        self.assertEqual((s["bound"], s["split_from"]), ("compute_latency", "MUL_MAT:indexer"))
        self.assertAlmostEqual(s["us_per_call"], 600.0, places=0)

    def test_indexer_without_prof_stays_lumped(self):
        res, p = self._idx_diag(with_prof=False)
        self.assertEqual([o["kernel_class"] for o in cu.extract_cpu(res, p)], ["MUL_MAT:indexer"])

    def test_classifier_split_supersedes_lumped_record_and_score_never_ranks(self):
        res, p = self._idx_diag()
        lump = {"device": "cpu", "model": "q38fn-ud-iq4xs-cpu", "kernel_class": "MUL_MAT:indexer", "shape": "quarter-24t-d32k", "n": 1,
                "bytes_per_call": 39321600, "us_per_call": 1206.5, "calls_per_round": 1, "config": "quarter-24t"}
        c = cu.empty_census()
        cu.apply(c, [lump], "census", "a" * 64, "t0")
        cu.apply(c, cu.extract_cpu(res, p), "diag", "b" * 64, "t1")
        recs = c["records"]
        self.assertNotIn(cu.rid(lump), recs)  # the lumped record leaves the ranked set
        wk = cu.rid(dict(lump, kernel_class="MUL_MAT:indexer_weight"))
        sup = recs[wk]["superseded"][0]
        self.assertEqual((sup["validity"], sup["source_row"], sup["superseded_by"]["row"]), ("classifier_split", "census", "diag"))
        self.assertEqual(sup["gbps"], 32.6)
        self.assertFalse([k for k in recs if "indexer_score" in k])  # score lives only under time_only
        self.assertTrue([k for k in c["time_only"] if "indexer_score" in k])


    def test_op_result_is_not_double_counted_by_trace_path(self):
        self.assertEqual(cu.extract_trace(q8_result(70.0)), [])


def rid_of(o):
    return cu.rid(o)
