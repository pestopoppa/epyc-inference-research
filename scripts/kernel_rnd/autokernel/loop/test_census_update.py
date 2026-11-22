"""census_update.py: idempotency, threshold flagging, seed linking (fixture RESULT.json files, no GPU)."""
import json
import os
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


def cpu_result(gate_up_ms=7.5872, rid="a3b-cpu-gap-census-fx"):
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
        self.assertEqual(gdn["ceiling_gbps"], 129.9)
        self.assertEqual(gdn["status"], "ok")            # 102.5 GB/s = 79% of ceiling
        self.assertEqual(gu["status"], "open_surface")   # 94.0 GB/s = 72% of ceiling
        self.assertNotIn("FLASH_ATTN", "".join(c["records"]))   # no static bytes -> no record
        self.assertEqual(sum("cpu|" in k for k in c["records"]), 2)  # MTP config skipped
        self.assertIn("### CPU (EPYC 9655", open(self.doc).read())

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
