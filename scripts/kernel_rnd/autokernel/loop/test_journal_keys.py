"""journal_keys.py (RSL-R7): op_family + shape_class keys, cross-model suppression, per-device hypothesis pricing, unkeyed reporting."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from autokernel.loop import census_families as cf
from autokernel.loop import journal_keys as jk
from autokernel.loop import seed

NEG = {"akm-x": {"op_family": "flash-attn-verify", "device": "gpu", "shape_class": {"kv": [32768, 131072]},
                 "status": "measured_null"}}


def K(**kw):
    return {"op_family": "flash-attn-verify", **kw}


class Overlap(unittest.TestCase):
    def test_other_model_overlapping_range_is_suppressed(self):  # the model is not in the key
        self.assertEqual(jk.suppressed_by(K(shape_class={"kv": [65536, 200000]}, model="a3b"), NEG), "akm-x")

    def test_disjoint_range_other_family_or_device_is_not(self):
        self.assertIsNone(jk.suppressed_by(K(shape_class={"kv": [1024, 8192]}), NEG))
        self.assertIsNone(jk.suppressed_by({"op_family": "mmid-moe", "shape_class": {"kv": 40000}}, NEG))
        self.assertIsNone(jk.suppressed_by(K(device="cpu", shape_class={"kv": 40000}), NEG))

    def test_dim_missing_on_either_side_is_a_wildcard(self):
        self.assertEqual(jk.suppressed_by(K(), NEG), "akm-x")  # whole family overlaps any range
        self.assertEqual(jk.suppressed_by(K(shape_class={"n": 8}), NEG), "akm-x")

    def test_touching_endpoints_overlap_and_reopens_skips(self):
        self.assertEqual(jk.suppressed_by(K(shape_class={"kv": 131072}), NEG), "akm-x")
        self.assertIsNone(jk.suppressed_by(K(shape_class={"kv": 131072}, reopens=["akm-x"]), NEG))

    def test_unkeyed_and_malformed_are_never_suppressed_and_only_nulls_block(self):
        self.assertIsNone(jk.suppressed_by(None, NEG))
        self.assertIsNone(jk.suppressed_by(K(shape_class={"kv": [9, 1]}), NEG))
        self.assertIsNone(jk.suppressed_by(K(), {"akm-y": {**NEG["akm-x"], "status": "keep"}}))
        self.assertIsNone(jk.parse_key("<!-- journal-key: {not json} -->"))


class SeedInstall(unittest.TestCase):
    def _seeds(self, d: Path, hyp: dict) -> Path:
        s = d / "seeds"
        shutil.copytree(seed.SEEDS, s)
        for f in (s / "hypotheses").glob("*.md"):
            f.unlink()
        for name, text in hyp.items():
            (s / "hypotheses" / name).write_text(text, encoding="utf-8")
        return s

    def test_keyed_hypothesis_is_suppressed_unkeyed_installs_and_is_reported(self):
        key = lambda sc: f'<!-- journal-key: {json.dumps({"op_family": "flash-attn-verify", "shape_class": sc})} -->\n'
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s = self._seeds(d, {"a-blocked.md": key({"kv": 65536}) + "Falsifier: x", "b-clear.md": key({"kv": 1024}) + "Falsifier: x",
                                "c-legacy.md": "Falsifier: no key"})
            r = seed.install(d / "store", seeds=s)
            self.assertEqual(r["suppressed"], {"a-blocked.md": "akm-c13-favec-combine-tail-split"})
            self.assertEqual(r["inbox_files"], ["b-clear.md", "c-legacy.md"])
            self.assertEqual(r["unkeyed"]["hypotheses"], ["c-legacy.md"])
            self.assertFalse((d / "store" / "inbox" / "a-blocked.md").exists())
            self.assertEqual(seed.install(d / "store", seeds=s)["negatives_added"], 0)  # keys stay out of the digest

    def test_shipped_negatives_are_keyed_with_census_families_and_the_rest_reported_unkeyed(self):
        recs = json.loads((seed.SEEDS / "negatives.json").read_text(encoding="utf-8"))["records"]
        keyed = [r for r in recs if r.get("op_family")]
        self.assertGreaterEqual(len(keyed), 10)
        for r in keyed:
            self.assertIn(r["op_family"], set(cf._FAMILY.values()), r["mechanism_id"])
            self.assertEqual(jk.problems(r), [], r["mechanism_id"])
        with tempfile.TemporaryDirectory() as tmp:
            r = seed.install(Path(tmp))
            self.assertEqual(len(r["unkeyed"]["negatives"]), len(recs) - len(keyed))
            self.assertIn("akm-hist-graph-opt", r["unkeyed"]["negatives"])

    def test_shipped_hypotheses_all_install_today(self):  # none carries a key yet: no suppression, all reported unkeyed
        with tempfile.TemporaryDirectory() as tmp:
            r = seed.install(Path(tmp))
        self.assertEqual(r["suppressed"], {})
        self.assertEqual(sorted(r["unkeyed"]["hypotheses"]), sorted(r["inbox_files"]))

    def test_key_entry_keys_a_later_journal_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            jk.key_entry(tmp, "akm-new", K(shape_class={"kv": 5}))
            self.assertEqual(jk.unkeyed(tmp, ["akm-new", "akm-old"]), ["akm-old"])
            with self.assertRaises(ValueError):
                jk.key_entry(tmp, "akm-bad", {"shape_class": {"kv": [2, 1]}})


FA27 = {"path": "27b-df2-mi210", "op": "gv_fattn + merge (verify FA)", "device": "mi210", "step_ms": 45.7, "share_of_step": 0.2,
        "share_ms": 9.14, "saving_ms_per_step": 2.0}
FNX = {"path": "flash-next-cpu", "op": "exps_down", "device": "cpu-quarter-24t", "step_ms": 80.0, "share_of_step": 0.25,
       "share_ms": None, "saving_ms_per_step": 3.0}
GATE = {"path": "flash-next-cpu", "op": "exps_gate_up", "device": "cpu-quarter-24t", "step_ms": 80.0, "share_of_step": 0.25,
        "share_ms": 20.0, "saving_ms_per_step": 1.0}


class Pricing(unittest.TestCase):
    HYB = {"id": "hybrid-fnx", "touches": [{"path": "flash-next-cpu", "family": "mmid-moe", "fraction": 0.5},
                                           {"path": "27b-df2-mi210", "family": "flash-attn-verify", "fraction": -0.1}]}

    def test_priced_from_census_rows_on_every_touched_device(self):
        p = jk.price(self.HYB, [FA27, FNX, GATE], cf.op_family)
        self.assertEqual(p["by_device"], {"cpu-quarter-24t": round(0.5 * (20.0 + 80 * 0.25), 3), "mi210": -0.914})
        self.assertAlmostEqual(p["saving_ms_per_step"], 20.0 - 0.914)

    def test_unmatched_or_unpriced_touch_is_counted_not_guessed(self):
        p = jk.price({"touches": [{"path": "nowhere", "op": "x", "fraction": 1.0}]}, [FA27], cf.op_family)
        self.assertEqual((p["saving_ms_per_step"], p["n_unpriced"]), (None, 1))

    def test_hypothesis_enters_census_families_rank(self):
        rows = cf.rank([FA27, FNX, GATE], model_weights={"flash-next-cpu": 0.5}, hypotheses=[self.HYB])
        h = next(r for r in rows if r["kind"] == "hypothesis")
        self.assertEqual(h["family"], "hypothesis:hybrid-fnx")
        self.assertAlmostEqual(h["score"], 0.5 * 20.0 - 0.914)  # weights apply per touched path
        self.assertEqual(rows[0]["family"], "hypothesis:hybrid-fnx")  # outranks the 2.0 / 4.0 families
        self.assertEqual([r["family"] for r in cf.rank([FA27])], ["flash-attn-verify"])  # no hypotheses = unchanged

    def test_hypothesis_files_with_touches_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = '<!-- journal-key: {"op_family":"mmid-moe","touches":[{"path":"p","op":"o","fraction":0.1}]} -->\ntext'
            (Path(tmp) / "30-x.md").write_text(body, encoding="utf-8")
            (Path(tmp) / "31-unkeyed.md").write_text("text", encoding="utf-8")
            self.assertEqual([h["id"] for h in jk.hypothesis_entries(tmp)], ["30-x"])


if __name__ == "__main__":
    unittest.main()
