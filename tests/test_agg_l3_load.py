"""AGG-0.2 L3 load job + L3-only M1 report tests (fake parser; no model, no PDF parsing, no network).

The job imports the orchestrator package, so run these with the orchestrator venv python and
``AGG_ORCH_ROOT=<orchestrator tree containing src/aggregation>``; without it the job tests skip.
``m1_l3_report`` is stdlib only and always runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "benchmarks" / "agg_phase0"
sys.path.insert(0, str(D))
import m1_l3_report as M  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/epyc-orchestrator"))
try:
    sys.path.insert(0, str(ORCH))
    from src.models.document import DocumentPreprocessResult, Section, TableRef  # noqa: E402
    import src.aggregation.registry  # noqa: E402,F401
    import l3_load as L  # noqa: E402

    HAVE_ORCH = True
except Exception:  # pragma: no cover - skip path
    HAVE_ORCH = False

SHEET_COLS = ["doc_name", "company", "metric", "fiscal_year", "value_usd_m", "status", "page", "quote",
              "reviewer", "note"]


def _fake_doc(revenue: str) -> "DocumentPreprocessResult":
    s = Section(id="s1", title="Income", level=1, content="Total revenues", page_start=60, page_end=60)
    t = TableRef(id="t1", page=60, caption="(in millions)", section_id="s1",
                 rows=[["", "2022", "2021"], ["Total revenues", f"${revenue}", "$24,000"],
                       ["Capital expenditures", "(1,200)", "900"]])
    return DocumentPreprocessResult(original_path="/x.pdf", sections=[s], figures=[], total_pages=10,
                                    tables=[t])


def _set_dir(tmp: Path, sheet_rows: list[dict] | None = None) -> Path:
    sd = tmp / "set"
    sd.mkdir()
    docs = []
    for i, name in enumerate(("B_DOC", "A_DOC", "C_DOC")):
        pdf = tmp / f"{name}.pdf"
        pdf.write_bytes(b"%PDF-fake-" + name.encode() * (i + 1))
        docs.append({"doc_name": name, "company": name.title(), "pdf": str(pdf),
                     "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "bytes": pdf.stat().st_size})
    (sd / "corpus.json").write_text(json.dumps({"docs": docs}))
    rows = sheet_rows if sheet_rows is not None else [
        {"doc_name": n, "company": n, "metric": m, "fiscal_year": "2022", "value_usd_m": "", "status": "todo"}
        for n in ("A_DOC", "B_DOC", "C_DOC") for m in ("revenue", "capex")]
    with (sd / "facts_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, SHEET_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in SHEET_COLS})
    return sd


def _args(out: Path, sd: Path, **kw) -> argparse.Namespace:
    return argparse.Namespace(out=out, orch_root=ORCH, set_dir=sd, only=kw.get("only"),
                              smoke=kw.get("smoke", False), orch_pin="o", research_pin="r")


@unittest.skipUnless(HAVE_ORCH, "needs the orchestrator venv python and AGG_ORCH_ROOT")
class LoadJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for k in ("FILLER_DEADLINE", "FILLER_STOPFILE", "FILLER_SMOKE"):
            os.environ.pop(k, None)
        self.calls: list[str] = []

    def parse(self, pdf: Path):
        self.calls.append(pdf.stem)
        # distinct content per doc: the store keys documents by content hash alone
        return _fake_doc({"A_DOC": "26,000", "B_DOC": "27,000", "C_DOC": "28,000"}[pdf.stem])

    def recs(self, out: Path) -> list[dict]:
        return [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]

    def test_full_run_records_early_and_resumes(self):
        sd, out = _set_dir(self.tmp), self.tmp / "out"
        self.assertEqual(L.run(_args(out, sd), self.parse), 0)
        self.assertEqual(self.calls, ["B_DOC", "A_DOC", "C_DOC"])  # smallest pdf first
        kinds = [r["kind"] for r in self.recs(out)]
        for k in ("meta", "probe", "parse", "store", "l3_doc", "doc_done", "corpus", "m1_l3", "done"):
            self.assertIn(k, kinds)
        self.assertEqual(kinds.count("doc_done"), 3)
        self.assertLess(kinds.index("doc_done"), kinds.index("corpus"))  # per doc, before corpus stage
        l3 = [r for r in self.recs(out) if r["kind"] == "l3_doc"][0]
        self.assertGreater(l3["n_facts"], 0)
        self.assertEqual((l3["llm_calls"], l3["prefill_tokens"]), (0, 0))
        res = json.loads((out / "result.json").read_text())
        self.assertTrue(res["complete"])
        self.assertEqual(res["model_calls"], 0)
        corpus = [r for r in self.recs(out) if r["kind"] == "corpus"][0]
        self.assertEqual(corpus["built_layers"], ["L1", "L3"])
        # resume: nothing re-parsed
        self.calls.clear()

        def boom(_):
            raise AssertionError("must not re-parse")

        self.assertEqual(L.run(_args(out, sd), boom), 0)
        self.assertEqual(self.calls, [])
        self.assertEqual([r["kind"] for r in self.recs(out)].count("skip"), 3)

    def test_stop_between_docs_is_partial_then_resumes_remaining_only(self):
        sd, out = _set_dir(self.tmp), self.tmp / "out"
        stop = self.tmp / "STOP"
        os.environ["FILLER_STOPFILE"] = str(stop)

        def parse_then_stop(pdf):
            r = self.parse(pdf)
            stop.write_text("")
            return r

        self.assertEqual(L.run(_args(out, sd), parse_then_stop), 75)
        self.assertEqual(self.calls, ["B_DOC"])
        self.assertFalse((out / "result.json").exists())
        stop.unlink()
        self.calls.clear()
        self.assertEqual(L.run(_args(out, sd), self.parse), 0)
        self.assertEqual(self.calls, ["A_DOC", "C_DOC"])

    def test_deadline_passed_parses_nothing(self):
        sd, out = _set_dir(self.tmp), self.tmp / "out"
        os.environ["FILLER_DEADLINE"] = "1"
        self.assertEqual(L.run(_args(out, sd), self.parse), 75)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.recs(out)[-1]["reason"], "deadline")

    def test_smoke_runs_smallest_doc_only(self):
        sd, out = _set_dir(self.tmp), self.tmp / "out"
        os.environ["FILLER_SMOKE"] = "1"
        self.assertEqual(L.run(_args(out, sd), self.parse), 0)
        self.assertEqual(self.calls, ["B_DOC"])
        self.assertFalse((out / "result.json").exists())
        self.assertEqual([r["kind"] for r in self.recs(out)].count("doc_done"), 1)

    def test_parse_failure_is_recorded_and_rc3(self):
        sd, out = _set_dir(self.tmp), self.tmp / "out"

        def bad(pdf):
            if pdf.stem == "A_DOC":
                raise RuntimeError("odl exploded")
            return _fake_doc("27,000")

        self.assertEqual(L.run(_args(out, sd), bad), 3)
        fails = [r for r in self.recs(out) if r["kind"] == "doc_fail"]
        self.assertEqual([f["doc"] for f in fails], ["A_DOC"])

    def test_pdf_sha_mismatch_fails_closed(self):
        sd, out = _set_dir(self.tmp), self.tmp / "out"
        (self.tmp / "A_DOC.pdf").write_bytes(b"tampered")
        self.assertEqual(L.run(_args(out, sd), self.parse), 2)
        self.assertEqual(self.calls, [])

    def test_m1_with_filled_gold_after_run(self):
        rows = [{"doc_name": "A_DOC", "metric": "revenue", "fiscal_year": "2022", "value_usd_m": "26000",
                 "status": "filled"},
                {"doc_name": "A_DOC", "metric": "capex", "fiscal_year": "2022", "value_usd_m": "1200",
                 "status": "filled"},
                {"doc_name": "B_DOC", "metric": "revenue", "fiscal_year": "2022", "value_usd_m": "99999",
                 "status": "filled"}]
        sd, out = _set_dir(self.tmp, rows), self.tmp / "out"
        self.assertEqual(L.run(_args(out, sd), self.parse), 0)
        rep = json.loads((out / "m1_l3_report.json").read_text())
        self.assertEqual((rep["gold_cells"], rep["found"]), (3, 2))
        self.assertAlmostEqual(rep["m1_l3_only"], 2 / 3)
        self.assertTrue(rep["l3_below_kill_min"])  # 0.667 < 0.70


class M1ReportTests(unittest.TestCase):
    def _store(self, tmp: Path, facts: list[tuple]) -> Path:
        import sqlite3

        p = tmp / "s.sqlite"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE facts (doc_id TEXT, field TEXT, period TEXT, value DOUBLE, unit TEXT)")
        c.executemany("INSERT INTO facts VALUES (?,?,?,?,?)", facts)
        c.commit()
        c.close()
        return p

    def _sheet(self, tmp: Path, rows: list[dict]) -> Path:
        p = tmp / "sheet.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, SHEET_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in SHEET_COLS})
        return p

    def test_no_gold_is_null_not_zero(self):
        tmp = Path(tempfile.mkdtemp())
        sheet = self._sheet(tmp, [{"doc_name": "A", "metric": "revenue", "fiscal_year": "2022",
                                   "status": "todo"}])
        rep = M.report(self._store(tmp, [("A", "revenue", "2022", 5.0, "USD_millions")]), sheet)
        self.assertIsNone(rep["m1_l3_only"])
        self.assertEqual(rep["m1_status"], "no_gold_cells_filled")
        self.assertEqual(rep["cells_with_any_l3_candidate"], 1)

    def test_unit_conversion_abs_and_tolerance(self):
        tmp = Path(tempfile.mkdtemp())
        sheet = self._sheet(tmp, [
            {"doc_name": "A", "metric": "revenue", "fiscal_year": "2022", "value_usd_m": "5363", "status": "filled"},
            {"doc_name": "A", "metric": "capex", "fiscal_year": "2022", "value_usd_m": "1200", "status": "filled"},
            {"doc_name": "A", "metric": "total_assets", "fiscal_year": "2022", "value_usd_m": "100", "status": "filled"},
            {"doc_name": "A", "metric": "net_income", "fiscal_year": "2022", "value_usd_m": "10", "status": "filled"}])
        store = self._store(tmp, [
            ("A", "revenue", "2022", 5363000.0, "USD_thousands"),   # converts to 5363
            ("A", "capex", "2022", -1203.0, "USD_millions"),         # abs, within 0.5%
            ("A", "total_assets", "2022", 120.0, "USD_millions"),    # outside tolerance
            ("A", "net_income_attributable", "2022", 10.0, "percent")])  # percent never a USD cell
        rep = M.report(store, sheet)
        self.assertEqual((rep["gold_cells"], rep["found"]), (4, 2))
        self.assertEqual(rep["per_metric"]["total_assets"], {"gold": 1, "found": 0})
        self.assertTrue(rep["l3_below_kill_min"])


if __name__ == "__main__":
    unittest.main()
