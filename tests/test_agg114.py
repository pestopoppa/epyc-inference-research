"""AGG-1.14 tests (no model, no network, no process). The replay tests import the orchestrator package at the AGG-1.14 commit
(``AGG_ORCH_ROOT``) and skip without ``facts_typed.value_in_quote``; the JSON tests read the committed ``agg114_rescore.json``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg114_rescore as R  # noqa: E402
import phase0_arms as P  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/worktrees/agg114-orch"))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import facts_llm as fl, facts_typed as ft, pipeline as pl  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = hasattr(ft, "value_in_quote")
except Exception:  # pragma: no cover
    HAVE_ORCH = False

OUT = json.loads((D1 / "agg114_rescore.json").read_text())


class Renderer(unittest.TestCase):
    def test_render_reads_rows_by_op(self):
        self.assertEqual(R.render("argmax", [{"entity": "Amazon", "revenue": 5.0}]), "FINAL: Amazon")
        self.assertEqual(R.render("top3", [{"entity": "A"}, {"entity": "B"}, {"entity": "C"}, {"entity": "D"}]), "FINAL: A; B; C")
        self.assertEqual(R.render("sum", [{"SUM(x)": 99494.548}]), "FINAL: 99,494.548")
        self.assertEqual(R.render("count_gt", [{"COUNT(*)": 7}]), "FINAL: 7")
        self.assertEqual(R.render("sum", []), "FINAL: no data")

    def test_the_frozen_scorer_reads_the_rendered_final(self):
        self.assertTrue(P.score_agg({"op": "argmax", "gold": "Amazon"}, R.render("argmax", [{"entity": "Amazon"}])))
        self.assertTrue(P.score_agg({"op": "sum", "gold": 99494.548}, R.render("sum", [{"s": 99494.548}])))
        self.assertFalse(P.score_agg({"op": "sum", "gold": 99494.548}, R.render("sum", [{"s": 101707.848}])))

    def test_rows_key_is_order_and_type_stable(self):
        self.assertEqual(R.rows_key([{"b": 1, "a": 2}]), R.rows_key([{"a": 2, "b": 1}]))
        self.assertNotEqual(R.rows_key([{"a": 1}]), R.rows_key([{"a": 2}]))


@unittest.skipUnless(HAVE_ORCH, "orchestrator at the AGG-1.14 commit not importable")
class Replay(unittest.TestCase):
    CID = R.CORPUS_ID

    def _store(self, tmp: Path) -> tuple[Path, str]:
        secs = [Section(id="s1", title="Cash flows", level=1, page_start=1, page_end=1,
                        content="(Dollars in millions)\nDividends to shareholders (11,682)\n"
                                "Cash dividends paid were $4.45 per share in fiscal year 2022\n")]
        path = tmp / "s.sqlite"
        st = CorpusStore(str(path))
        st.ingest(DocumentPreprocessResult(original_path="/x/d.pdf", sections=secs, figures=[], total_pages=1, tables=[]),
                  corpus_id=self.CID, doc_id="D", entity="E")
        h = st.corpus_docs(self.CID)[0]["doc_hash"]
        st.close()
        return path, h

    def _rec(self, h: str) -> dict:
        def f(i, value, quote):
            return {"fact_id": f"f{i}", "doc_hash": h, "doc_id": "D", "entity": "E", "period": "FY2022", "field": "dividends_paid",
                    "value": value, "unit": "USD_millions", "quote": quote, "quote_start": 0, "quote_end": len(quote),
                    "chunk_id": "c#w0", "table_id": None, "page": 1, "line_start": 1, "line_end": 1, "confidence": "high",
                    "extractor": "l4_llm.v1", "audit_status": "unaudited", "declared_scale": "USD_millions",
                    "declared_scale_basis": "table_header"}
        return {"kind": "t_window", "final": True, "doc": "D", "chunk_id": "c#w0", "chosen_by": ["dividends_paid"],
                "facts": [f(1, 13895.3, "Cash dividends paid were $4.45 per share in fiscal year 2022"),
                          f(2, 11682.0, "Dividends to shareholders (11,682)")]}

    def test_baseline_replay_keeps_the_unquoted_row_and_the_rule_removes_it(self):
        spec = pl.pipeline_spec("agg_pipeline.v2")
        with tempfile.TemporaryDirectory() as d:
            path, h = self._store(Path(d))
            wins = R.window_records([self._rec(h)])
            self.assertEqual(sorted(wins), ["D"])
            for rule, want in ((False, {13895.3, 11682.0}), (True, {11682.0})):
                p = Path(d) / f"r{int(rule)}.sqlite"
                p.write_bytes(path.read_bytes())
                out = R.replay(p, wins, rule=rule, ft=ft, fl=fl, spec=spec)
                self.assertEqual({r[4] for r in R.l4_signature(out["store"])}, want, rule)
                self.assertEqual(len(out["dropped"]), 0 if not rule else 1)
                out["store"].close()

    def test_violation_report_counts_per_field_and_flags_stored_rows(self):
        with tempfile.TemporaryDirectory() as d:
            _p, h = self._store(Path(d))
            rep = R.violation_report(ft, R.window_records([self._rec(h)]), {"f1"})
        self.assertEqual(rep["rows_entering_stage"]["n_violating"], 1)
        self.assertEqual(rep["rows_entering_stage"]["by_field"], {"dividends_paid": 1})
        self.assertEqual(rep["rows_entering_stage"]["by_basis"], {"printed": 1, "violation": 1})
        self.assertEqual(rep["violating_rows_entering_stage"][0]["value"], 13895.3)
        self.assertTrue(rep["violating_rows_entering_stage"][0]["stored_in_agg112_store"])


class CommittedResult(unittest.TestCase):
    def test_baseline_replay_reproduces_the_saved_agg112_store(self):
        b = OUT["baseline_replay"]
        self.assertTrue(b["l4_facts_match_saved_store"] and b["raw_cells_equal_saved"] and b["canonical_cells_equal_saved"])
        self.assertEqual((b["raw"], b["canonical"], b["n_l4_facts"]), (208, 202, 345))

    def test_violating_rows_and_corrected_counts(self):
        v = OUT["violations"]["rows_entering_stage"]
        self.assertEqual((v["n_rows"], v["n_violating"]), (445, 21))
        self.assertEqual(sum(v["by_field"].values()), 21)
        self.assertEqual((OUT["violations"]["all_window_rows"]["n_rows"], OUT["violations"]["all_window_rows"]["n_violating"]), (693, 74))
        c = OUT["corrected"]
        self.assertEqual((c["raw"], c["canonical"]), (208, 203))
        self.assertEqual((c["n_raw_fixed"], c["n_raw_broken"], c["n_canonical_fixed"], c["n_canonical_broken"]), (23, 0, 26, 0))
        self.assertEqual(c["cells_lost_by_the_rule_vs_agg112"], [])
        self.assertTrue(OUT["no_previously_correct_cell_regresses"])

    def test_johnson_johnson_dividends_paid_is_correct_now(self):
        j = OUT["johnson_johnson_dividends_paid"]
        self.assertTrue(j["canonical_ok_now"] and j["canonical_ok_agg11"] and not j["canonical_ok_agg112"])
        self.assertEqual([r["value"] for r in j["canonical_rows_now"]], [11682.0])

    def test_p_large_rescore(self):
        pl_ = OUT["p_large"]["planners"]
        v1, v0 = pl_["agg110.v1"], pl_["agg11.v0"]
        self.assertEqual((v1["n"], v1["saved_correct"], v1["rescored_correct"]), (58, 39, 40))
        self.assertEqual((v0["n"], v0["saved_correct"], v0["rescored_correct"]), (58, 26, 27))
        self.assertEqual((v1["vs_agg11_rescored"]["only_new"], v1["vs_agg11_rescored"]["only_old"]), (23, 1))
        self.assertEqual((v0["vs_agg11_rescored"]["only_new"], v0["vs_agg11_rescored"]["only_old"]), (9, 0))
        self.assertEqual(v1["answer_changed_questions"], ["agg18"])
        self.assertEqual(v1["regressed_vs_agg11"], ["p1_01"])
        for v in (v0, v1):
            self.assertEqual(v["old_rows_not_reproducing_saved"], [])
            for r in v["rows"]:
                if r["changed"]:     # the renderer must agree with the saved score on the OLD rows of every re-scored question
                    self.assertTrue(r["mechanical_agrees_with_saved_on_old_rows"], r["qid"])


if __name__ == "__main__":
    unittest.main()
