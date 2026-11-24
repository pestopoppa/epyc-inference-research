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
        # per-record stubs stay a journal note; stub FILES are per surface, and this surface has no share (no parity) -> unranked
        self.assertEqual(os.listdir(os.path.join(self.d, "stubs")), [])
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
        self.assertEqual(gdn["ceiling_gbps"], 129.0)
        self.assertEqual(gdn["status"], "ok")            # 102.5 GB/s = 79% of ceiling
        self.assertEqual(gu["status"], "open_surface")   # 81.0 GB/s = 63% of ceiling (< 0.75 x 129 = 96.75)
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
        c["cpu"]["flag_frac"] = 0.60  # 0.60 x 129 = 77.4 < 81.0
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


class Parity(unittest.TestCase):
    def test_gap_times_share_ranks_and_flags_unworked(self):
        c = cu.empty_census()
        c["parity"] = {"refs": {"r": {"pct": 85, "basis": "b", "cite": "x"}}, "ops": [
            {"path": "p", "op": "a", "ours_gbps": 500, "roofline_gbps": 1000, "ref": "r", "share_of_step": 0.1,
             "covered_by": "", "refresh_c21": True, "source": "s", "served": True},
            {"path": "p", "op": "b", "ours_pct": 80, "roofline_gbps": 1000, "ref": "r", "share_of_step": 0.5,
             "covered_by": "BOX", "refresh_c21": False, "source": "s", "served": True}]}
        rows = cu.parity_rows(c)
        self.assertEqual([r["op"] for r in rows], ["a", "b"])  # 35*0.1=3.5 > 5*0.5=2.5
        self.assertAlmostEqual(rows[0]["score"], 3.5)
        md = "\n".join(cu.parity_md(c))
        self.assertIn("Top unworked gaps (served, no box or row): p a (3.5)", md)

    def test_null_ref_pct_has_no_gap_or_score(self):
        c = cu.empty_census()
        c["parity"] = {"refs": {"n": {"pct": None, "basis": "b", "cite": "x", "status": "no-published-measurement",
                                      "bracket_pct": {"e2e": 80}}},
                       "ops": [{"path": "p", "op": "a", "ours_pct": 35, "ref": "n", "share_of_step": 0.5,
                                "covered_by": "", "refresh_c21": False, "source": "s", "served": True}]}
        r = cu.parity_rows(c)[0]
        self.assertEqual((r["ref_pct"], r["gap"], r["score"]), (None, None, None))
        md = "\n".join(cu.parity_md(c))
        self.assertIn("no published measurement", md)
        self.assertIn("e2e 80%", md)

    def test_status_label_and_batch_table_render(self):
        c = cu.empty_census()
        c["parity"] = {"refs": {"m": {"pct": 88, "basis": "b", "cite": "x", "status": "measured"},
                                "o": {"pct": 85, "basis": "b", "cite": "x"}}, "ops": [],
                       "batch_dependence": {"note": "n", "source": "src",
                                            "rows": [{"kernel": "k4", "pct": "18-38", "batch": "1"}]}}
        c["parity"]["ops"] = [{"path": "p", "op": "a", "ours_pct": 10, "roofline_gbps": 1, "ref": "m", "share_of_step": 0.1,
                               "covered_by": "", "refresh_c21": False, "source": "s", "served": True}]
        md = "\n".join(cu.parity_md(c))
        self.assertIn("`m` 88% [measured]", md)
        self.assertIn("`o` 85% [recalled, not re-verified]", md)
        self.assertIn("| k4 | 18-38 | 1 |", md)

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
    """Operator 2026-11-24: one ceiling per device -- MI210 1500, CPU quarter-24t 129, CPU full host 520 GB/s."""

    def test_quarter_ceiling_is_operator_set_with_provenance(self):
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["quarter-24t"], 129.0)
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["full-host"], 520.0)
        self.assertEqual(cu.CEILINGS_GBPS, {"mi210": 1500.0, "cpu-quarter-24t": 129.0, "cpu-full-host": 520.0})
        src = cu.CPU_DEFAULTS["ceiling_source"]["quarter-24t"]
        self.assertTrue(src.startswith("operator 2026-11-24"))
        self.assertIn("9e379493", src)  # the superseded measurement stays cited
        self.assertEqual(cu.empty_census()["cpu"]["ceiling_gbps"]["quarter-24t"], 129.0)
        self.assertEqual(cu.CPU_DEFAULTS["ceiling_gbps"]["half-48t"], 259.8)  # untouched

    def test_open_surface_threshold_is_75pct_of_129(self):
        c = cu.empty_census()
        for name, g in (("below", 96.7), ("above", 96.8)):
            c["records"][f"cpu|m|{name}|quarter-24t|n1"] = {
                "device": "cpu", "model": "m", "kernel_class": name, "shape": "quarter-24t", "config": "quarter-24t",
                "n": 1, "bytes_per_call": g * 1e6, "us_per_call": 1e3, "calls_per_round": 1, "source_row": "fx"}
        cu.rerank(c)
        self.assertEqual(c["records"]["cpu|m|below|quarter-24t|n1"]["status"], "open_surface")
        self.assertEqual(c["records"]["cpu|m|above|quarter-24t|n1"]["status"], "ok")

    def test_committed_census_is_reranked_against_measured_ceiling(self):
        c = json.load(open(cu.CENSUS))
        self.assertEqual(c["cpu"]["ceiling_gbps"]["quarter-24t"], 129.0)
        self.assertEqual(c["ceiling_gbps"], 1500.0)
        self.assertEqual(c["cpu"]["ceiling_source"]["quarter-24t"], cu.CPU_DEFAULTS["ceiling_source"]["quarter-24t"])
        q = [r for r in c["records"].values() if r.get("device") == "cpu" and r.get("config") == "quarter-24t"]
        self.assertTrue(q)
        self.assertTrue(all(r["ceiling_gbps"] == 129.0 for r in q))
        self.assertTrue(all(r["status"] == ("open_surface" if r["gbps"] < 0.75 * 129.0 else "ok") for r in q))


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
        self.assertEqual(r["ceiling_gbps"], 129.0)
        self.assertEqual(r["frac_of_ceiling"], round(83.5 / 129.0, 3))
        self.assertEqual(r["status"], "open_surface")   # 83.5 GB/s < 0.75 x 129 = 96.75

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


class MergeStatic(Base):
    def committed(self):
        return {"parity": {"ops": [], "refs": {}, "note": "n"}, "time_only": {"x": {"us_per_call": 1.0}}, "newkey": {"a": 1},
                "records": {"R": {"model": "m", "kernel_class": "k", "shape": "s", "n": 1, "bytes_per_call": 1e6, "us_per_call": 1.0,
                                  "calls_per_round": 1, "source_row": "z", "measured_at": "t"}}, "applied": [{"row": "z", "sha256": "z"}],
                "seeds": {"S": {"status": "done", "match": {"model": "none"}}}}  # well-formed: run() now unions these in

    def live(self):
        c = cu.empty_census()
        c["records"] = {"L": {"gbps": 9}}
        c["applied"] = [{"row": "a", "sha256": "a"}]
        c["seeds"] = {"LS": {"status": "open"}}
        return c

    def test_adds_new_and_static_keys(self):
        live = self.live()
        ch = cu.merge_static(live, self.committed())
        self.assertEqual(ch, ["newkey", "parity", "time_only"])
        self.assertEqual(live["parity"], self.committed()["parity"])
        self.assertEqual(live["newkey"], {"a": 1})

    def test_records_applied_seeds_preserved(self):
        live, before = self.live(), self.live()
        cu.merge_static(live, self.committed())
        for k in ("records", "applied", "seeds", "cpu", "floor_gbps"):
            self.assertEqual(live[k], before[k])

    def test_idempotent(self):
        live = self.live()
        cu.merge_static(live, self.committed())
        snap = json.dumps(live, sort_keys=True)
        self.assertEqual(cu.merge_static(live, self.committed()), [])
        self.assertEqual(json.dumps(live, sort_keys=True), snap)

    def test_cli_flag_merges_file_and_is_idempotent(self):
        src = os.path.join(self.d, "src.json")
        json.dump(self.committed(), open(src, "w"))
        json.dump(self.live(), open(self.census, "w"))
        first = cu.run(None, None, self.census, None, None, None, merge_static_from=src)["merge_static"]
        self.assertTrue({"newkey", "parity", "time_only", "surfaces"} <= set(first))
        self.assertEqual(cu.run(None, None, self.census, None, None, None, merge_static_from=src)["merge_static"], [])
        self.assertIn("parity", json.load(open(self.census)))

    def test_merge_then_apply_keeps_parity(self):
        src = os.path.join(self.d, "src.json")
        json.dump(self.committed(), open(src, "w"))
        json.dump(cu.empty_census(), open(self.census, "w"))
        cu.run(self.res("r.json", q8_result(90.41)), None, self.census, self.prog, self.doc, None, merge_static_from=src)
        c = json.load(open(self.census))
        self.assertIn("parity", c)
        self.assertTrue(c["records"])


class Surfaces(unittest.TestCase):
    """One ranked list of bandwidth surfaces (operator 2026-11-24)."""

    def committed(self):
        c = json.load(open(cu.CENSUS))
        cu.rerank(c)
        cu.derive_surfaces(c)
        return c

    def test_every_record_maps_to_an_op(self):
        c = self.committed()
        self.assertEqual([k for k, r in c["records"].items() if r.get("op") is None], [])
        for r in c["records"].values():
            self.assertEqual(cu.op_of(r), (r["op_path"], r["op"]))

    def test_mapping_examples(self):
        v = ("27b-df2-mi210", "verify dense Q8 GEMV/GEMM (q8mf6, V5I)")
        for k in ("ffn_gate_up_v5i", "ffn_down_v5i", "gdn_attn_qkv_v5i", "gdn_attn_gate_v5i", "ssm_out+attn_output_v5i",
                  "attn_q_v5i"):
            self.assertEqual(cu.OP_MAP[("27b-q8_0", k)], v)
        for k in ("mmid_gate_up_trace", "mmid_down_trace"):
            self.assertEqual(cu.OP_MAP[("a3b-q8_0", k)], ("a3b-mtp-mi210", "MUL_MAT_ID experts n=3"))
        self.assertEqual(cu.OP_MAP[("a3b-q8_0-cpu", "MUL_MAT:gdn_proj")], ("a3b-mtp-cpu", "gdn_proj"))
        self.assertEqual(cu.OP_MAP[("q38fn-ud-iq4xs-cpu", "MUL_MAT:gdn_proj")], ("flash-next-cpu", "gdn_proj"))

    def test_saving_formula_and_ranking(self):
        c = self.committed()
        s = c["surfaces"]
        self.assertEqual(c["target_pct"], 88.0)
        for x in s:
            if x["saving_ms_per_step"] is not None:
                want = x["step_ms"] * x["share_of_step"] * max(0.0, 1 - x["ours_pct"] / 88.0)
                self.assertAlmostEqual(x["saving_ms_per_step"], want, places=2)
        sv = [x["saving_ms_per_step"] for x in s]
        known = [v for v in sv if v is not None]
        self.assertEqual(known, sorted(known, reverse=True))
        self.assertTrue(all(v is None for v in sv[len(known):]))  # nulls sink

    def test_sanity_anchors(self):
        by = {(x["path"], x["op"]): x for x in self.committed()["surfaces"]}
        for key, want in ((("a3b-mtp-mi210", "dense MMVQ (attn, GDN proj, shexp, lm_head)"), 7.6),
                          (("27b-df2-mi210", "verify dense Q8 GEMV/GEMM (q8mf6, V5I)"), 4.8),  # 6.5 was the 1638 nameplate
                          (("a3b-mtp-mi210", "router mul_mat_f F32 [2048,256]"), 4.5),
                          (("27b-df2-mi210", "gv_fattn + merge (verify FA)"), 2.8)):
            self.assertAlmostEqual(by[key]["saving_ms_per_step"], want, delta=0.15)

    def test_placeholders_and_null_refs(self):
        by = {(x["path"], x["op"]): x for x in self.committed()["surfaces"]}
        tail = by[("27b-df2-mi210", "launch-bound tail (eltwise/norm/copy/rope/conv)")]
        self.assertFalse(tail["ours_measured"])
        self.assertIsNone(tail["ours_pct"])
        self.assertEqual(tail["ours_pct_placeholder"], 35)
        self.assertIsNone(tail["saving_ms_per_step"])
        self.assertEqual(tail["status"], "needs_measurement")
        aq = by[("27b-df2-mi210", "actquant (q8mf_pre_i8 + q8sk_quantize)")]
        self.assertEqual(aq["measured_ms"], 2.1)
        self.assertIsNone(aq["ours_pct"])
        v = by[("27b-df2-mi210", "verify dense Q8 GEMV/GEMM (q8mf6, V5I)")]
        self.assertIsNone(v["ref_pct"])
        self.assertIsNone(v["gap_pp"])  # null ref -> null gap, never 0
        self.assertTrue(v["ours_measured"])
        self.assertEqual(len(v["census_rows"]), 31)

    def test_rollup_is_never_a_surface(self):
        c = self.committed()
        ops = {(x["path"], x["op"]) for x in c["surfaces"]}
        self.assertNotIn(("flash-next-cpu", "forward m1 dense (UD-IQ4_XS)"), ops)
        g = c["surface_groups"][0]
        self.assertTrue(g["unreconciled"])
        self.assertEqual(g["members"], sum(x["group"] == g["op"] for x in c["surfaces"]))
        gdn = next(x for x in c["surfaces"] if x["path"] == "flash-next-cpu" and x["op"] == "gdn_proj")
        self.assertIsNotNone(gdn["d32k_pct"])
        self.assertTrue(gdn["step_source"].startswith("derived"))

    def test_gpu_rows_carry_parity_share_and_ceilings_single(self):
        c = self.committed()
        g = [r for r in c["records"].values() if r.get("device", "gpu") == "gpu"]
        self.assertTrue(all(r.get("share_of_round") is not None for r in g))
        self.assertTrue(all(o["roofline_gbps"] in (1500.0, 129.0) for o in c["parity"]["ops"]))

    def test_keeps_old_keys(self):
        c = self.committed()
        for k in ("records", "parity", "seeds", "time_only", "surfaces", "target_pct"):
            self.assertIn(k, c)

    def test_every_extractor_class_maps(self):
        """Every kernel class an extractor can emit for a served model resolves (live census holds rows the committed one lacks)."""
        want = [("27b-q8_0", c) for c, _ in cu.Q8_SHAPE_CLASS.values()] + [("27b-q8_0", c + "_v5i") for c, _ in cu.Q8_SHAPE_CLASS.values()]
        want += [("27b-q8_0", c) for c, _ in cu.TRACE_Q8_CLASS.values()] + [("a3b-q8_0", c) for c, _ in cu.MMID_CLASS.values()]
        want += [("a3b-q8_0-cpu", f) for f in cu.A3B_CPU_FAMILY_BYTES] + [("a3b-q8_0-cpu", c) for c, _ in cu.UBENCH_CLASS.values()]
        self.assertEqual([w for w in want if w not in cu.OP_MAP], [])


class SurfaceWiring(Base):
    """Planner block, per-surface seeds, keep updates (operator 2026-11-24, AutoKernel seed format)."""

    TAIL = "27b-df2-mi210/launch-bound tail (eltwise/norm/copy/rope/conv)"

    def committed_live(self):
        shutil.copy(cu.CENSUS, self.census)
        return json.load(open(self.census))

    def surf(self, key):
        return next(s for s in json.load(open(self.census))["surfaces"] if cu.skey(s) == key)

    def test_mapped_surface_rederives_from_its_census_rows(self):
        self.run_(self.res("a.json", q8_result(90.41)))
        k = "27b-df2-mi210/verify dense Q8 GEMV/GEMM (q8mf6, V5I)"
        before = self.surf(k)["ours_pct"]
        self.run_(self.res("b.json", q8_result(75.0, at="y")))
        self.assertNotEqual(self.surf(k)["ours_pct"], before)
        self.assertNotIn("stale", self.surf(k))

    def test_keep_on_parity_only_surface_marks_stale_once_and_clears_on_parity_change(self):
        self.committed_live()
        res = {"verdict": "KEEP", "at": "2026-11-24T10:00:00Z", "target_surface": self.TAIL.upper()}
        r = self.run_(self.res("k.json", res), row="ak-tail-fuse-gpu-20261124")
        s = self.surf(self.TAIL)
        self.assertEqual((s["stale"], s["stale_since"], s["stale_row"]),
                         (True, "2026-11-24T10:00:00Z", "ak-tail-fuse-gpu-20261124"))
        self.assertIsNone(s["ours_pct"])  # never invented from a RESULT without bytes and time
        self.assertEqual(sum("marked STALE" in ln for ln in open(self.prog).read().splitlines()), 1)
        self.assertIn("marked STALE", open(self.prog).read())
        self.assertIn("STALE since 2026-11-24T10:00:00Z", open(self.prog).read())
        self.assertEqual(self.run_(self.res("k.json", res))["journal_lines"], 0)
        c = json.load(open(self.census))  # committed parity refreshed -> the flag clears
        next(o for o in c["parity"]["ops"] if cu.skey(o) == self.TAIL)["ours_gbps"] = 600.0
        cu.derive_surfaces(c)
        self.assertNotIn("stale", next(x for x in c["surfaces"] if cu.skey(x) == self.TAIL))
        self.assertEqual(c["surface_stale"], {})

    def test_non_keep_or_ambiguous_name_marks_nothing(self):
        self.committed_live()
        self.run_(self.res("d.json", {"verdict": "DROP(x)", "at": "t1", "target_surface": self.TAIL}))
        self.run_(self.res("e.json", {"verdict": "KEEP", "at": "t2", "target_surface": "lm_head"}))  # 3 paths -> ambiguous
        self.assertEqual(json.load(open(self.census)).get("surface_stale", {}), {})

    def test_bare_unique_op_matches(self):
        c = self.committed_live()
        cu.derive_surfaces(c)
        self.assertEqual([cu.skey(s) for s in cu.match_surfaces(c, {"target_surfaces": ["quantize_q8_1"]})],
                         ["a3b-mtp-mi210/quantize_q8_1"])

    def test_one_stub_per_seedable_surface_and_committed_seed_file_is_current(self):
        c = self.committed_live()
        cu.rerank(c)
        cu.derive_surfaces(c)
        ss = cu.seedable_surfaces(c)
        self.assertTrue(all(s["status"] == "open" and not s["covered_by"] and s["ours_measured"] for s in ss))
        keys = {cu.skey(s) for s in ss}
        self.assertIn("flash-next-cpu/gdn_proj", keys)
        self.assertNotIn("a3b-mtp-mi210/router mul_mat_f F32 [2048,256]", keys)  # AK-RF-A3B-ROUTER-FUSE
        self.assertNotIn("a3b-mtp-mi210/FA / gv (10 layers, f16 KV)", keys)  # AK-RF-A3B-GV-ATTN
        d = os.path.join(self.d, "stubs")
        self.assertEqual(cu.write_stubs(d, c), len(ss))
        self.assertEqual(cu.write_stubs(d, c), 0)
        self.assertEqual(sorted(os.listdir(d)), sorted(cu.surface_seed_id(s) + ".md" for s in ss))
        committed = os.path.join(os.path.dirname(cu.CENSUS), "..", "seeds", "hypotheses", "26-bw-surfaces-20261124.md")
        self.assertEqual(open(committed).read(), cu.surface_seeds_md(c, "2026-11-24"))
        for s in ss:
            self.assertIn(f"target_surface: {cu.skey(s)}", open(committed).read())

    def test_table_md_has_surfaces_section(self):
        c = self.committed_live()
        cu.derive_surfaces(c)
        t = cu.table_md(c)
        self.assertIn("### Bandwidth surfaces (ranked by saving ms/step)", t)
        self.assertIn("| Owner |", t)


class SpecTargetFallback(unittest.TestCase):
    def test_result_without_target_uses_spec(self):
        import tempfile
        from autokernel.loop import census_update as cu
        d = tempfile.mkdtemp()
        sp = os.path.join(d, "spec.json")
        json.dump({"target_surface": "27b-df2-mi210/launch-bound tail"}, open(sp, "w"))
        self.assertEqual(cu.with_spec_target({"verdict": "KEEP"}, sp)["target_surfaces"], ["27b-df2-mi210/launch-bound tail"])
        self.assertEqual(cu.with_spec_target({"verdict": "KEEP", "target_surface": "x/y"}, sp), {"verdict": "KEEP", "target_surface": "x/y"})
        self.assertEqual(cu.with_spec_target({"verdict": "KEEP"}, None), {"verdict": "KEEP"})


class PublishedFromFallback(unittest.TestCase):
    def test_manifest_without_target_follows_published_from(self):
        d = os.path.realpath(tempfile.mkdtemp())
        old = cu.SPEC_ROOT
        cu.SPEC_ROOT = d
        self.addCleanup(setattr, cu, "SPEC_ROOT", old)
        row = os.path.join(d, "spec.json")
        json.dump({"target_surface": "27b-df2-mi210/q4_0 KV attention (gv_fattn, q4_0 KV)"}, open(row, "w"))
        man = os.path.join(d, "ready.json")
        json.dump({"published_from": row}, open(man, "w"))
        self.assertEqual(cu.with_spec_target({"verdict": "KEEP"}, man)["target_surfaces"],
                         ["27b-df2-mi210/q4_0 KV attention (gv_fattn, q4_0 KV)"])
        json.dump({"published_from": "/etc/spec.json"}, open(man, "w"))
        self.assertEqual(cu.with_spec_target({"verdict": "KEEP"}, man), {"verdict": "KEEP"})


class ResultWarrant(Base):
    """CPU-shaped RESULTs carry their stamp under different fields; every record needs the RESULT's own stamp + sha."""

    def _cpu(self, **ts):
        res = {k: v for k, v in json.load(open(os.path.join(FX, "a3b_r2", "RESULT.json"))).items() if k not in cu.TS_FIELDS}
        return self.res("RESULT.json", dict(res, **ts))

    def _cpu_recs(self):
        return [r for r in json.load(open(self.census))["records"].values() if r.get("device") == "cpu"]

    def test_each_timestamp_field_stamps_the_records(self):
        for f in cu.TS_FIELDS:
            if os.path.exists(self.census):
                os.remove(self.census)
            self.run_(self._cpu(**{f: "2026-11-22T00:57:21Z"}), row="fx-cpu")
            recs = self._cpu_recs()
            self.assertTrue(recs, f)
            self.assertTrue(all(r["measured_at"] == "2026-11-22T00:57:21Z" and r["measured_at_basis"] == f for r in recs), f)

    def test_no_timestamp_is_missing_not_invented(self):
        p = self._cpu()
        self.run_(p, row="fx-cpu")
        for r in self._cpu_recs():
            self.assertEqual((r["measured_at"], r["measured_at_basis"]), ("", "missing"))  # not mtime, not now()

    def test_every_record_carries_the_result_files_sha(self):
        p = self._cpu(updated_utc="2026-11-22T00:57:21Z")
        self.run_(p, row="fx-cpu")
        recs = self._cpu_recs()
        self.assertTrue(recs and all(r["result_sha256"] == cu.sha256(p) for r in recs))
        with self.assertRaises(ValueError):
            cu.apply(json.load(open(self.census)), [], "fx", "", "2026-11-22")  # a null/empty sha is refused, never stored
