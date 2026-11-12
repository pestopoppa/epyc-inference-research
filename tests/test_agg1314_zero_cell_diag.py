"""Zero-cell diagnosis tests (AGG-1.3 r2 / AGG-1.4 r2). Stdlib only: no model, no network, no orchestrator needed.

The real-store test runs only when the saved r2 stores exist on this host.
"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg1314_zero_cell_diag as Z  # noqa: E402
import agg13 as G  # noqa: E402
import agg_phase0 as A0  # noqa: E402
import phase1_arms as X  # noqa: E402

R13 = Path("/mnt/raid0/llm/tmp/agg13-reingest-cpu-20261111-r2/out")
R14 = Path("/mnt/raid0/llm/tmp/agg14-backfill-cpu-20261111-r2/out")


def _db(rows):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE facts (doc_id, field, period, value, unit, extractor, page, quote)")
    c.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)", rows)
    return c


class Classes(unittest.TestCase):
    def test_period_label(self):  # AGG-1.3: value right, model labelled the FY2021 value FY2022
        d = Z.diagnose_cell({"doc": "D", "field": "revenue", "gold": 168088.0, "fy": "2021"},
                            [{"period": "FY2022", "value": 168088.0, "unit": "USD_millions"}])
        self.assertEqual(d["verdict"], "PERIOD_LABEL")

    def test_unit_scale(self):  # AGG-1.4: raw 2,467 emitted as unit "USD", reconcile stored 0.002467 USD_millions
        d = Z.diagnose_cell({"doc": "D", "field": "long_term_debt", "gold": 2467.0, "fy": "2022"},
                            [{"period": "2022-12-31", "value": 0.002467, "unit": "USD_millions"}])
        self.assertEqual(d["verdict"], "UNIT_SCALE")

    def test_hit_near_wrong_none(self):
        c = {"doc": "D", "field": "f", "gold": 100.0, "fy": "2022"}
        self.assertEqual(Z.diagnose_cell(c, [{"period": "2022", "value": 100.2, "unit": "USD_millions"}])["verdict"], "HIT")
        self.assertEqual(Z.diagnose_cell(c, [{"period": "2022", "value": 101.5, "unit": "USD_millions"}])["verdict"],
                         "NEAR_ROUNDED")
        self.assertEqual(Z.diagnose_cell(c, [{"period": "2022", "value": 138.0, "unit": "USD_millions"}])["verdict"],
                         "WRONG_VALUE")
        self.assertEqual(Z.diagnose_cell(c, [])["verdict"], "NO_FACT")

    def test_strict_equals_frozen_scorers(self):
        facts = [{"doc_id": "D", "field": "f", "period": p, "value": v, "unit": u}
                 for p, v, u in (("2021", 100.0, "USD_millions"), ("FY2022", 7.0, "USD_millions"),
                                 ("2022-12-31", 0.002, "USD_millions"), ("2022", 5000.0, "USD_thousands"))]
        cells = [("D", "f", 100.0, "2021"), ("D", "f", 7.0, "2021"), ("D", "f", 2000.0, "2022"), ("D", "f", 5.0, "2022")]
        frozen = G.m1_hits_fy(facts, cells, unit_scale=X.M1R_UNIT_SCALE)
        mine = [Z.diagnose_cell({"doc": d, "field": f, "gold": g, "fy": fy},
                                [x for x in facts if x["doc_id"] == d and x["field"] == f])["verdict"] == "HIT"
                for d, f, g, fy in cells]
        self.assertEqual(frozen, mine)


class Guards(unittest.TestCase):
    def test_fy_contract_flags_agg13_prompt_not_agg14(self):
        cells13 = [{"field": f, "fy": "2021"} for f in A0.METRICS]
        probs = Z.fy_contract_problems(A0.METRICS, cells13)
        self.assertEqual(len(probs), len(A0.METRICS))          # every L4 description says FY2022
        self.assertTrue(all("FY2022" in p and "FY2021" in p for p in probs))
        self.assertEqual(Z.fy_contract_problems(A0.METRICS, [{"field": f, "fy": "2022"} for f in A0.METRICS]), [])
        self.assertEqual(Z.fy_contract_problems({"x": "no year here"}, [{"field": "x", "fy": "2021"}]), [])

    def test_scale_signature(self):
        c = _db([("D", "ltd", "2022", 0.0025, "USD_millions", "l4_llm.v1", 1, "q"),
                 ("D", "ltd", "2022", 2467.0, "USD_millions", "l4_llm.v1", 1, "q"),
                 ("D", "ltd", "2022", 0.5, "USD_millions", "l3_regex.v1", 1, "q")])
        self.assertEqual(Z.scale_signature(c, "ltd"), {"ltd": {"n": 2, "abs_lt_1": 1}})

    def test_rescore_keeps_strict_separate(self):
        ds = [{"verdict": v} for v in ("HIT", "PERIOD_LABEL", "UNIT_SCALE", "NO_FACT")]
        r = Z.rescore(ds)
        self.assertEqual((r["strict"]["found"], r["diag_period_agnostic"]["found"],
                          r["diag_unit_normalised"]["found"], r["diag_both"]["found"]), (1, 2, 2, 3))


@unittest.skipUnless((R13 / "store.sqlite").exists() and (R14 / "store.sqlite").exists(), "saved r2 stores absent")
class SavedStores(unittest.TestCase):
    def test_agg13_r2(self):
        r = Z.run13(R13)
        self.assertEqual(r["changed"]["strict"]["found"], 0)               # reproduces the recorded zero
        self.assertEqual(r["changed"]["classes"]["PERIOD_LABEL"], 3)
        self.assertEqual(r["changed"]["classes"]["NO_FACT"], 19)
        self.assertEqual(r["g4_overall"]["diag_period_agnostic"]["found"], 168)

    def test_agg14_r2(self):
        r = Z.run14(R14)
        self.assertEqual(r["all_cells"]["strict"]["found"], 16)            # reproduces the recorded G2 0.64
        self.assertEqual(r["backfilled_docs"]["strict"]["found"], 0)
        self.assertEqual(r["backfilled_docs"]["classes"]["UNIT_SCALE"], 6)
        self.assertEqual(r["all_cells"]["diag_unit_normalised"]["found"], 22)
        self.assertEqual(r["scale_signature"]["long_term_debt"], {"n": 13, "abs_lt_1": 13})


if __name__ == "__main__":
    unittest.main()
