"""RSL-R1/R4/R7 wiring: census_update writes `coverage` and `family_rank`, the planner block and /loop carry them, nothing is invented."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from autokernel import loop as _  # noqa: F401
from autokernel.loop import census_coverage as cc
from autokernel.loop import census_update as cu
from autokernel.loop import loop

A3B, Q27 = "qwen36_35b_a3b_mtp_q8_local", "qwen38_27b_q8_local"
LIVE = "/mnt/raid0/llm/tmp/ak-bw-census/mi210_bandwidth_census.json"
FA27 = ("27b-df2-mi210", "gv_fattn + merge (verify FA)")
DENSE27 = ("27b-df2-mi210", "verify dense Q8 GEMV/GEMM (q8mf6, V5I)")


def registry():
    return {"server_mode": {
        "architect_critic": {"model_role": Q27, "device": "ROCm0", "slots": 4, "acceleration": {"draft_max": 7}},
        "frontdoor": {"model_role": A3B, "slots": 4, "acceleration": {"draft_max": 4}}}}


def surf(key, saving, step=40.0, share=0.25, dev="mi210"):
    return {"path": key[0], "op": key[1], "device": dev, "saving_ms_per_step": saving, "step_ms": step,
            "share_of_step": share, "status": "open", "covered_by": None, "ours_measured": True, "ours_pct": 50.0}


def census(recs=(("27b-df2-mi210", "x"), ("a3b-mtp-cpu", "y")), surfaces=()):
    return {"parity": {"ops": [{"path": "27b-df2-mi210", "op": "x", "served": True},
                               {"path": "a3b-mtp-mi210", "op": "ghost", "served": True}]},
            "records": {f"r{i}": {"op_path": p, "op": o} for i, (p, o) in enumerate(recs)}, "surfaces": list(surfaces)}


class Coverage(unittest.TestCase):
    def test_every_served_path_with_a_record_has_no_violation(self):
        c = census()
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        self.assertEqual(c["coverage"]["status"], "ok")
        self.assertEqual(c["coverage"]["paths"]["27b-q8_0|mi210"]["uncovered_ops"], [])
        # the planned a3b-mtp-mi210 path has no record: tracked as if served, so it is a violation (plus the ghost parity op)
        self.assertEqual(cu.coverage_violations(c), ["a3b-q8_0|mi210: served path has zero census records"])

    def test_violation_when_a_served_path_has_zero_records(self):
        c = census(recs=(("27b-df2-mi210", "x"),))  # a3b-mtp-cpu (served on CPU) has no census record at all
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        v = cu.coverage_violations(c)
        self.assertTrue(v)
        self.assertIn("a3b-q8_0-cpu|cpu: served path has zero census records", v)

    def test_violation_when_a_parity_op_claims_a_path_the_registry_lacks(self):
        c = census()
        c["parity"]["ops"].append({"path": "gone-path", "op": "z", "served": True})
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        self.assertTrue(any("gone-path/z" in v for v in cu.coverage_violations(c)))

    def test_unreadable_registry_fails_soft_and_is_a_violation(self):
        old = os.environ.get(cc.REGISTRY_ENV)
        os.environ[cc.REGISTRY_ENV] = "/nonexistent/registry.yaml"
        try:
            c = census()
            cu.annotate(c, hypotheses_dir=None)
        finally:
            os.environ.pop(cc.REGISTRY_ENV) if old is None else os.environ.__setitem__(cc.REGISTRY_ENV, old)
        self.assertEqual(c["coverage"]["status"], "registry-unreadable")
        self.assertIn("missing", c["coverage"]["error"])   # the cause is kept, not swallowed (defect 9)
        self.assertTrue(any("registry-unreadable" in v and c["coverage"]["error"] in v for v in c["coverage"]["violations"]))
        self.assertEqual(c["family_rank"], [])
        self.assertTrue(cu.coverage_violations(c))
        json.dumps(c)


class FamilyRank(unittest.TestCase):
    def _c(self):
        return census(surfaces=[surf(FA27, 2.0), surf(DENSE27, 3.0), surf(("a3b-mtp-mi210", "FA / gv (10 layers, f16 KV)"), 1.5),
                                surf(("flash-next-cpu", "lm_head"), None, dev="cpu")])

    def test_surfaces_tagged_and_family_rank_written(self):
        c = self._c()
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        self.assertEqual([s["family"] for s in c["surfaces"]],
                         ["flash-attn-verify", "dense-q8-gemv", "flash-attn-verify", "lm-head-gemv"])
        fr = {r["family"]: r for r in c["family_rank"]}
        self.assertEqual(fr["flash-attn-verify"]["score"], 3.5)  # 2.0 on 27B + 1.5 on A3B, one family across two paths
        self.assertIsNone(fr["lm-head-gemv"]["score"])           # unpriced stays null, ranked last
        self.assertEqual(c["family_rank"][-1]["family"], "lm-head-gemv")

    def test_model_weights_from_the_census_reorder_the_rank(self):
        c = self._c()
        c["model_weights"] = {"a3b-q8_0": 0.0, "27b-q8_0": 1.0}
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        self.assertEqual(c["family_rank"][0]["family"], "dense-q8-gemv")
        self.assertEqual({r["family"]: r["score"] for r in c["family_rank"]}["flash-attn-verify"], 2.0)

    def test_knob_arms_are_unpriced_without_a_screened_pct(self):
        c = self._c()
        c["knob_arms"] = [{"name": "np", "path": FA27[0]}]
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        k = next(r for r in c["family_rank"] if r["kind"] == "knob")
        self.assertEqual((k["status"], k["score"], k["saving_ms_per_step"]), ("unpriced", None, None))

    def test_hand_written_hypothesis_ranks_in_the_same_list_priced_from_census_rows(self):
        c = self._c()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "h-fuse.md").write_text(
                "# fuse\n<!-- journal-key: " + json.dumps({"op_family": "flash-attn-verify", "shape_class": {"n": [1, 8]}, "touches": [
                    {"path": FA27[0], "op": FA27[1], "fraction": 0.5}, {"path": "a3b-mtp-mi210", "family": "flash-attn-verify", "fraction": 1.0}]}) + " -->\n")
            (Path(d) / "h-plain.md").write_text("# no key, never ranked\n")
            cu.annotate(c, registry=registry(), hypotheses_dir=d)
        h = next(r for r in c["family_rank"] if r["kind"] == "hypothesis")
        self.assertEqual(h["family"], "hypothesis:h-fuse")
        self.assertEqual(h["saving_ms_per_step"], 10.0 * 0.5 + 10.0 * 1.0)  # share_ms = step 40 x share 0.25 = 10 per touched surface
        self.assertEqual(h["by_device"], {"mi210": h["saving_ms_per_step"]})
        self.assertEqual(len([r for r in c["family_rank"] if r["kind"] == "hypothesis"]), 1)

    def test_planner_block_carries_family_and_rank(self):
        c = self._c()
        cu.annotate(c, registry=registry(), hypotheses_dir=None)
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "journal.md"
            j.write_text("# header\n\n" + cu.JOURNAL_HEAD + "\n- `r0` (x): line\n")
            self.assertTrue(cu.write_surfaces_block(str(j), c))
            text = loop.program_text(j)
        self.assertIn("| Family |", text)
        self.assertIn("Family rank", text)
        self.assertIn("| 1 | flash-attn-verify | family | 3.50 |", text)
        self.assertIn("| lm-head-gemv | family | - |", text)


class RunWritesTheKeys(unittest.TestCase):
    def test_run_writes_coverage_and_family_rank_on_a_copy_of_the_live_census(self):
        if not os.path.exists(LIVE):
            self.skipTest("live census absent")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            c = json.load(open(LIVE))
            c.pop("coverage", None), c.pop("family_rank", None)
            json.dump(c, open(p, "w"))
            res = os.path.join(d, "r.json")
            json.dump({"row": "fx", "at": "2026-11-24T00:00:00Z", "op": {"cells_us": {}}}, open(res, "w"))
            cu.run(res, "fx", p, os.path.join(d, "j.md"), None, None)
            out = json.load(open(p))
            self.assertIn("coverage", out)
            self.assertIn("family_rank", out)
            self.assertTrue(all("family" in s for s in out["surfaces"]))
            self.assertEqual(out["coverage"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
