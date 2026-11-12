"""AGG-1.12 L3 period probe tests: no model, no server. The synthetic tests are stdlib; the store test needs the
orchestrator package (``AGG_ORCH_ROOT``, default the AGG-1.12 tree) and skips without it."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg_l3period as L  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/worktrees/agg-l3period-orch-20261112"))
HAVE_ORCH = (ORCH / "src" / "aggregation" / "facts_det.py").exists()
PROBE = D1 / "agg_l3period_probe.json"


class Pure(unittest.TestCase):
    def test_diff_counts_fixes_and_breaks(self):
        d = L.diff({"a": True, "b": False, "c": True, "d": False}, {"a": True, "b": True, "c": False, "d": False})
        self.assertEqual((d["n_fixed"], d["n_broken"], d["before"], d["after"]), (1, 1, 2, 2))
        self.assertEqual((d["fixed"], d["broken"]), (["b"], ["c"]))

    def test_adoption_needs_zero_breaks_under_every_reading_and_a_fix(self):
        clean = {"n_broken": 0, "n_fixed": 2}
        none = {"n_broken": 0, "n_fixed": 0}
        broke = {"n_broken": 1, "n_fixed": 3}
        self.assertTrue(L.adoption(clean, clean, clean)["adopted"])
        self.assertFalse(L.adoption(clean, broke, clean)["adopted"])
        self.assertFalse(L.adoption(clean, clean, broke)["adopted"])        # the third reading counts
        self.assertFalse(L.adoption(none, none, none)["adopted"])           # nothing fixed is not an adoption
        self.assertTrue(L.adoption(none, clean)["adopted"])

    def test_fiscal_years_come_from_the_manifest(self):
        corpus = {"fiscal_year": 2022, "docs": [{"doc_name": "A"}, {"doc_name": "B", "fiscal_year": 2021}]}
        self.assertEqual(L.fiscal_years(corpus), {"A": "2022", "B": "2021"})
        real = json.loads((D1 / "corpus.json").read_text())
        self.assertEqual(set(L.fiscal_years(real).values()), {"2022"})
        self.assertEqual(len(L.fiscal_years(real)), 28)

    def test_period_census(self):
        c = L.period_census([{"period": None}, {"period": "2022"}, {"period": "FY2022"}, {"period": "2021"}])
        self.assertEqual(c, {"l3_facts": 4, "period_none": 1, "period_2022": 2})

    def test_the_eleven_l3_class_cells_come_from_the_committed_triage(self):
        cells = L.l3_class_cells(json.loads((D1 / "agg111_triage.json").read_text()))
        self.assertEqual(len(cells), L.EXPECTED["l3_class"])
        self.assertEqual(sum(c["group"] == "c" for c in cells), 10)
        self.assertEqual([c["doc"] + "|" + c["field"] for c in cells if c["group"] == "b"], ["COCACOLA_2022_10K|long_term_debt"])

    def test_copy_store_copies_without_touching_the_source_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "s.sqlite", Path(td) / "d.sqlite"
            c = sqlite3.connect(src)
            c.execute("CREATE TABLE t(x)")
            c.execute("INSERT INTO t VALUES (1)")
            c.commit()
            c.close()
            before = L.sha256_file(src)
            L.copy_store(src, dst)
            self.assertEqual(sqlite3.connect(dst).execute("SELECT x FROM t").fetchone(), (1,))
            self.assertEqual(L.sha256_file(src), before)
            with self.assertRaises(FileExistsError):
                L.copy_store(src, dst)


class CommittedProbe(unittest.TestCase):
    @unittest.skipUnless(PROBE.exists(), "probe JSON not committed yet")
    def test_probe_records_the_declared_properties(self):
        p = json.loads(PROBE.read_text())
        self.assertTrue(p["post_hoc"] and p["in_sample"])
        self.assertEqual(p["model_calls"], 0)
        self.assertEqual(p["n_cells"], 211)
        self.assertTrue(p["source_store_unchanged"])
        self.assertTrue(p["variants"]["control_v1"]["reproduces_stored_l3_exactly"])
        self.assertEqual((p["variants"]["control_v1"]["raw"]["n_fixed"], p["variants"]["control_v1"]["raw"]["n_broken"]), (0, 0))
        v2 = p["variants"]["v2"]
        self.assertTrue(v2["adoption"]["adopted"])
        self.assertEqual((v2["raw"]["n_broken"], v2["canon_v1"]["n_broken"], v2["canon_v2_attrib"]["n_broken"]), (0, 0, 0))
        for name in ("v2_plus_date_header", "v2_plus_inline_header", "v2_plus_all_optional"):
            self.assertFalse(p["variants"][name]["adoption"]["adopted"], name)
        self.assertEqual(p["verdict"]["v2_default_adopted"], True)


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not available")
class WithOrchestrator(unittest.TestCase):
    def test_variant_table_names_and_the_default_components(self):
        L.import_orch(ORCH)
        v = L.variants()
        self.assertEqual(list(v)[:2], ["control_v1", "v2"])
        self.assertIsNone(v["control_v1"]["period_version"])
        from src.aggregation import facts_det  # noqa: PLC0415
        self.assertEqual(v["v2"]["components"], list(facts_det.L3_PERIOD_COMPONENTS))
        self.assertIn("v2_plus_inline_header", v)
        self.assertIn("v2_minus_row_select", v)

    def test_rerun_l3_replaces_only_the_l3_facts(self):
        L.import_orch(ORCH)
        from src.aggregation.store import CorpusStore  # noqa: PLC0415
        from src.models.document import DocumentPreprocessResult, Section, TableRef  # noqa: PLC0415
        st = CorpusStore()
        s = Section(id="s1", title="Balance", level=1, content="x", page_start=57, page_end=57)
        t = TableRef(id="t1", page=57, caption="(Dollars in millions)", section_id="s1",
                     rows=[["Total assets $137,100", "$138,552"]])
        st.ingest(DocumentPreprocessResult(original_path="/x.pdf", sections=[s], figures=[], total_pages=60, tables=[t]),
                  corpus_id="c", doc_id="D_2022_10K", entity="D")
        h = st.corpus_docs("c")[0]["doc_hash"]
        st.write_facts([{"fact_id": "l4", "doc_hash": h, "doc_id": "D_2022_10K", "field": "total_assets", "period": "2022",
                         "value": 1.0, "unit": "USD_millions", "extractor": "l4_llm.v1", "confidence": "high",
                         "audit_status": "unaudited"}])
        v = L.variants()
        L.rerun_l3(st, "c", {"D_2022_10K": "2022"}, v["control_v1"])
        got = st.conn.execute("SELECT value, period FROM facts WHERE extractor LIKE 'l3_table%'").fetchall()
        self.assertEqual(got, [(138552.0, None)])
        L.rerun_l3(st, "c", {"D_2022_10K": "2022"}, v["v2"])
        got = st.conn.execute("SELECT value, period FROM facts WHERE extractor LIKE 'l3_table%'").fetchall()
        self.assertEqual(got, [(137100.0, "2022")])
        self.assertEqual(st.conn.execute("SELECT COUNT(*) FROM facts WHERE extractor='l4_llm.v1'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
