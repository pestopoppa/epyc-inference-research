"""AGG-1.4 tests (fake backend: no model, no network, no process, no PDF).

Stdlib tests always run. The store / planner / L4 tests import the orchestrator package and skip without it: run them with the
orchestrator venv python and ``AGG_ORCH_ROOT=<tree at the AGG-1.4 pin>`` (the tree must carry ``planner.backfill_plan``).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg14 as G  # noqa: E402
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_agg14 as R  # noqa: E402

DEFAULT_ORCH = "/mnt/raid0/llm/worktrees/agg13-14-orch-20261111"
ORCH = Path(os.environ.get("AGG_ORCH_ROOT", DEFAULT_ORCH))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import planner  # noqa: E402
    from src.aggregation.llm_backend import GATE_GBNF  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = hasattr(planner, "backfill_plan")
except Exception:  # pragma: no cover
    HAVE_ORCH = False


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class PinAndInputTests(unittest.TestCase):
    def test_pins_match(self):
        self.assertEqual(G.pin_problems(D1), [])
        self.assertEqual(sha(D1 / "PREDECLARED_AGG14.md"), G.PREDECLARED_SHA)
        self.assertEqual(sha(D1 / "PREDECLARED_AGG14_AMENDMENT1_20261111.md"), G.AMENDMENT1_SHA)
        self.assertEqual(sha(D1 / "PREDECLARED.md"), X.PREDECLARED_SHA)

    def test_gold_cells_28_with_25_filled_and_the_three_na(self):
        inp = G.load_inputs(D1)
        self.assertEqual(G.cells_problems(inp), [])
        self.assertEqual((len(inp["cells"]), len(G.gold_cells(inp))), (28, 25))
        self.assertEqual(G.na_docs(inp), sorted(G.EXPECTED_NA))
        x = json.loads((D1 / "agg14_xcheck.json").read_text())
        self.assertEqual((x["summary"]["cells"], x["summary"]["checked"], x["summary"]["fails"]), (28, 28, 0))
        # thousands filers converted to millions
        by = {r["doc_name"]: r for r in inp["cells"]}
        self.assertEqual(float(by["NETFLIX_2022_10K"]["value_usd_m"]), 14353.076)
        self.assertEqual(float(by["AMAZON_2022_10K"]["value_usd_m"]), 67150.0)   # reversed columns: FY2022 = 67,150

    def test_field_pick_is_the_documented_rule(self):
        pick = json.loads((D1 / "agg14_field_pick.json").read_text())
        self.assertEqual(pick["picked"], G.FIELD)
        elig = [c for c in pick["candidates"] if c["eligible"]]
        self.assertEqual(min(elig, key=lambda c: (c["l3_recall_proxy"], c["field"]))["field"], G.FIELD)
        self.assertTrue(all(c["applicable_docs"] >= 20 for c in elig))
        self.assertTrue(all(c["applicable_docs"] < 20 for c in pick["candidates"] if not c["eligible"]))
        self.assertFalse(set(c["field"] for c in pick["candidates"]) & set(A0.METRICS))   # no L4-extracted field
        self.assertEqual(sorted(set(pick["l4_fields_in_store"])), sorted(A0.METRICS))

    def test_gates_are_the_boxes(self):
        self.assertEqual((G.CALL_GATE, G.M1_GATE, G.PILOT_WALL_S), (100, 0.75, 600))
        text = (D1 / "PREDECLARED_AGG14.md").read_text()
        for needle in ("G1 backfill calls <= 100", "M1 >= 0.75", "<= 600 s", "REPL-retrieval alternative", "ESTIMATE"):
            self.assertIn(needle, text, needle)

    def test_field_description_context_restores_the_metric_table(self):
        before = dict(A0.METRICS)
        with G.field_description():
            self.assertEqual(A0.METRICS[G.FIELD], G.FIELD_DESC)
            self.assertEqual(len(A0.METRICS), 9)
        self.assertEqual(A0.METRICS, before)
        with self.assertRaises(RuntimeError):
            with G.field_description():
                raise RuntimeError("x")
        self.assertEqual(A0.METRICS, before)
        self.assertEqual(G.OLD_FIELDS, tuple(before))

    def test_pilot_eval_arithmetic(self):
        u = lambda c, w: {"llm_calls": c, "prefill_tokens": 7000, "wall_s": w}  # noqa: E731
        self.assertEqual(G.pilot_eval(u(8, 60.0), 12)["gates_tripped"], [])           # 96 <= 100
        self.assertEqual(G.pilot_eval(u(9, 60.0), 12)["gates_tripped"], ["G1"])       # 108 > 100
        self.assertEqual(G.pilot_eval(u(6, 601.0), 12)["budget_stops"], ["pilot_wall_over_600s", "projected_wall_over_2x_booked"])
        self.assertEqual(G.pilot_eval(u(6, 167.0), 12)["budget_stops"], ["projected_wall_over_2x_booked"])  # 2004 > 2000
        self.assertEqual(G.pilot_eval(u(6, 166.0), 12)["budget_stops"], [])                                # 1992

    def test_g1_eval_boundary(self):
        self.assertTrue(G.g1_eval({"llm_calls": 100, "prefill_tokens": 1, "wall_s": 1.0}, 12, 12)["ok"])
        self.assertFalse(G.g1_eval({"llm_calls": 101, "prefill_tokens": 1, "wall_s": 1.0}, 12, 12)["ok"])

    def test_m1_field_scores_every_filled_cell_and_the_backfilled_subset(self):
        cells = [("A", G.FIELD, 100.0), ("B", G.FIELD, 200.0), ("C", G.FIELD, 7.0)]
        facts = [{"doc_id": "A", "field": G.FIELD, "period": "FY2022", "value": 100.0, "unit": "USD_millions"},
                 {"doc_id": "C", "field": G.FIELD, "period": "2022", "value": 7000.0, "unit": "USD_thousands"}]
        m = G.m1_field(facts, cells, {"B", "C"})
        self.assertEqual((m["found"], m["gold"], m["backfilled_docs"]["gold"], m["backfilled_docs"]["found"]), (2, 3, 2, 1))


# --------------------------------------------------------------------------------------------- fixture
DOCS = ["AAA_2022_10K", "BBB_2022_10K", "CCC_2022_10K", "DDD_2022_10K"]
LTD = {"AAA_2022_10K": 1234.0, "BBB_2022_10K": 5678.0, "CCC_2022_10K": 910.0}   # DDD prints no long-term debt: NA
MISSING = ("CCC_2022_10K", "DDD_2022_10K")


class FakeBackend:
    """Revenue rows for ``Total revenues N``; long-term debt rows for ``Long-term debt N``; yes to every gate."""

    def __init__(self):
        self.calls = self.prefill_tokens = 0

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4
        if grammar == GATE_GBNF:
            return {"text": "yes", "prompt_tokens": len(prompt) // 4, "cache_n": 0}
        body = prompt.split("Excerpt:\n", 1)[-1]
        rows = []
        for fld, label in (("revenue", "Total revenues"), ("long_term_debt", "Long-term debt")):
            m = re.search(rf"{label} [\d,]+", body)
            if m and f"Field: {fld}" in prompt:
                rows.append({"period": "FY2022", "value": float(m.group(0).split()[-1].replace(",", "")),
                             "unit": "USD_millions", "quote": m.group(0), "confidence": "medium"})
        return {"text": json.dumps(rows), "prompt_tokens": len(prompt) // 4, "cache_n": 0}


def doc(name: str):
    ltd = f"Long-term debt {LTD[name]:,.0f}\n" if name in LTD else "Notes about nothing\n"
    secs = [Section(id="s1", title="Balance", level=1, content=f"Balance sheet FY2022\nTotal revenues 100\n{ltd}",
                    page_start=60, page_end=60)]
    return DocumentPreprocessResult(original_path=f"/{name}.pdf", sections=secs, figures=[], total_pages=100, tables=[])


def make_fixture(tmp: Path, *, drop_old_unit: bool = False) -> dict:
    base_dir = tmp / "agg11"
    base_dir.mkdir()
    store = CorpusStore(base_dir / "store.sqlite")
    for n in DOCS:
        r = store.ingest(doc(n), corpus_id=P.CORPUS_ID, doc_id=n, entity=n.split("_")[0])
    store.conn.execute("UPDATE documents SET n_tokens_est = 400000")      # 1.6M tokens: aggregation -> sql path
    store.conn.commit()
    hashes = {d["doc_id"]: d["doc_hash"] for d in store.corpus_docs(P.CORPUS_ID)}
    l3 = []
    for n in ("AAA_2022_10K", "BBB_2022_10K"):                             # L3 already covers 2 of the 4 documents
        l3.append({"fact_id": f"l3-{n}", "doc_hash": hashes[n], "doc_id": n, "entity": n.split("_")[0], "period": "2022",
                   "field": G.FIELD, "value": LTD[n], "unit": "USD_millions", "quote": f"Long-term debt {LTD[n]:,.0f}",
                   "quote_start": 0, "quote_end": 5, "chunk_id": f"{hashes[n][:16]}:00000", "table_id": None, "page": 60,
                   "line_start": 3, "line_end": 3, "confidence": "high", "extractor": "l3_textrow.v1",
                   "audit_status": "unaudited"})
    store.write_facts(l3)
    rec = P.Recorder(base_dir / "results.jsonl")
    units = P.windows(store, P.CORPUS_ID, P.CFG.window_chars)
    b = FakeBackend()
    for f in A0.METRICS:
        for n in DOCS:
            X.l4_unit(rec, store, b, units, f, n, lambda: None)
    if drop_old_unit:
        lines = [json.loads(l) for l in (base_dir / "results.jsonl").read_text().splitlines()]
        gone = next(r for r in lines if r.get("kind") == "l4_unit")
        (base_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines if r is not gone))
    store.close()
    corpus = {"docs": [{"doc_name": n, "company": n.split("_")[0]} for n in DOCS]}
    cells = [{"doc_name": n, "metric": G.FIELD, "fiscal_year": "2022", "verified": "true",
              "status": "filled" if n in LTD else "NA", "value_usd_m": str(LTD[n]) if n in LTD else ""} for n in DOCS]
    return {"inp": {"corpus": corpus, "cells": cells}, "base": base_dir / "store.sqlite", "results11": base_dir / "results.jsonl"}


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with planner.backfill_plan not importable")
class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.guard = lambda: None
        self.patches = [mock.patch.object(G, "EXPECTED_MISSING", MISSING), mock.patch.object(G, "PILOT_DOC", "CCC_2022_10K")]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def _run(self, fx, out, backend=None, *, rec=None, smoke=False, **kw):
        rec = rec or P.Recorder(out / "results.jsonl")
        backend = backend or FakeBackend()
        res = G.run_backfill(rec, out, fx["inp"], Path("/unused"), backend, base_store=fx["base"],
                             agg11_results=fx["results11"], guard=self.guard, smoke=smoke,
                             expected_missing=kw.pop("expected_missing", MISSING), pilot_doc="CCC_2022_10K", **kw)
        return rec, res, backend

    def test_receipt_trigger_backfill_recall_and_untouched_units(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out)
        self.assertEqual(res["rc"], 0, res)
        self.assertEqual((res["verdict"], res["gate_tripped"]), ("agg_1_4_done", []))
        # G4: the receipt
        r = next(x for x in rec.records if x.get("unit") == "receipt")
        self.assertEqual((r["receipt"]["path"], r["receipt"]["llm_called"], r["receipt"]["path_label"]), ("sql", False, "sql+backfill"))
        self.assertEqual(r["receipt"]["backfill"]["fields"], [G.FIELD])
        self.assertEqual(r["receipt"]["backfill"]["missing_docs"], list(MISSING))
        # only the missing documents' units for ONLY this field ran, pilot first
        fresh = [x for x in rec.records if x["kind"] == "l4_unit" and not x.get("reused")]
        self.assertEqual([(x["field"], x["doc"]) for x in fresh], [(G.FIELD, "CCC_2022_10K"), (G.FIELD, "DDD_2022_10K")])
        self.assertEqual(res["g1"]["calls"], b.calls)
        self.assertEqual(sum(1 for x in rec.records if x["kind"] == "l4_unit" and x.get("reused")), 32)   # 8 fields x 4 docs
        # G2: recall over the 3 filled cells (2 from L3, 1 from the backfill)
        self.assertEqual((res["g2"]["gold"], res["g2"]["found"], res["g2"]["l3_only"]["found"], res["g2"]["lift"]), (3, 3, 2, 1))
        self.assertEqual(res["g2"]["backfilled_docs"], {"gold": 1, "found": 1, "m1": 1.0})
        # G3: the earlier L4 facts are byte-identical, nothing re-ran
        self.assertTrue(res["g3"]["ok"], res["g3"])
        self.assertEqual((res["g3"]["previously_built_units"], res["g3"]["previously_built_rerun"], res["g3"]["fresh_calls_on_old_fields"]), (32, 0, 0))
        self.assertEqual(res["g3"]["counts_before"], res["g3"]["counts_after"])
        # the field is now answerable deterministically
        self.assertEqual(res["deterministic_check"]["table_argmax"], "BBB_2022_10K")
        self.assertEqual(res["deterministic_check"]["gold_argmax"], "BBB_2022_10K")
        # the metric table was left alone
        self.assertNotIn(G.FIELD, A0.METRICS)
        # resume: zero new calls
        b2 = FakeBackend()
        self.assertEqual(self._run(fx, out, b2)[1]["rc"], 0)
        self.assertEqual(b2.calls, 0)
        # the base store was only read
        self.assertEqual(CorpusStore(fx["base"]).conn.execute("SELECT COUNT(*) FROM facts WHERE field = ? AND extractor = 'l4_llm.v1'", [G.FIELD]).fetchone()[0], 0)

    def test_call_gate_trips_in_the_pilot_projection_and_blocks_everything_after(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        with mock.patch.object(G, "CALL_GATE", 1):                      # 1 call x 2 units = 2 > 1
            rec, res, b = self._run(fx, out)
        self.assertEqual((res["rc"], res["gate_tripped"], res["where"]), (G.RC_GATE, ["G1"], "pilot"))
        self.assertEqual(len([x for x in rec.records if x["kind"] == "l4_unit" and not x.get("reused")]), 1)

    def test_running_total_over_the_gate_stops_at_once(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        with mock.patch.object(G, "pilot_eval", lambda u, n: {"gates_tripped": [], "budget_stops": []}), \
                mock.patch.object(G, "CALL_GATE", 1):
            rec, res, b = self._run(fx, out)                           # the pilot passes (mocked), the 2nd unit crosses 1
        self.assertEqual((res["rc"], res["where"]), (G.RC_GATE, "running"))

    def test_pilot_wall_over_bound_is_a_budget_stop_not_a_gate(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        with mock.patch.object(G, "PILOT_WALL_S", -1.0):
            rec, res, b = self._run(fx, out)
        self.assertEqual((res["rc"], res["budget_stop"]), (G.RC_BUDGET, "pilot_wall_over_600s"))
        self.assertFalse((out / "result.cpu.json").exists())

    def test_wrong_backfill_set_trips_g4_before_any_compute(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, expected_missing=("CCC_2022_10K",))
        self.assertEqual((res["rc"], res["gate_tripped"], res["where"]), (G.RC_GATE, ["G4"], "receipt"))
        self.assertEqual(b.calls, 0)

    def test_a_missing_seeded_unit_means_not_every_earlier_unit_is_a_cache_hit(self):
        fx = make_fixture(self.tmp, drop_old_unit=True)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out)
        self.assertEqual(res["gate_tripped"], ["G3"])
        self.assertEqual(res["g3"]["previously_built_units"], 31)

    def test_snapshot_changes_when_an_earlier_l4_fact_changes(self):
        fx = make_fixture(self.tmp)
        s = CorpusStore(fx["base"])
        a = G.l4_snapshot(s.conn)
        self.assertGreater(a["n_l4_facts"], 0)
        s.conn.execute("UPDATE facts SET value = value + 1 WHERE extractor = 'l4_llm.v1' AND field = 'revenue'")
        self.assertNotEqual(G.l4_snapshot(s.conn)["sha256"], a["sha256"])

    def test_smoke_runs_one_real_unit_on_the_pilot_document(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec = P.Recorder(out / "results.jsonl")
        rec.emit("meta", phase="x")
        t0 = max(r["ts"] for r in rec.records)
        rec, res, b = self._run(fx, out, rec=rec, smoke=True)
        self.assertTrue(res["smoke"])
        new = [r for r in rec.records if r["kind"] == "l4_unit" and not r.get("reused") and r["ts"] > t0]
        self.assertEqual([(r["field"], r["doc"]) for r in new], [(G.FIELD, "CCC_2022_10K")])
        self.assertGreaterEqual(new[0]["llm_calls"], 1)
        rec2, res2, _ = self._run(fx, out, rec=P.Recorder(out / "results.jsonl"))
        self.assertEqual(res2["rc"], 0)
        units = [r["unit"] for r in rec2.records if r["kind"] == "l4_unit" and not r.get("reused")]
        self.assertEqual(len(units), len(set(units)))


@unittest.skipUnless(HAVE_ORCH and (G.DEFAULT_AGG11_JOB / "SOURCE.sha256").exists()
                     and (G.DEFAULT_AGG11_JOB / "out" / "store.sqlite").exists(), "AGG-1.1 job dir not on this host")
class CliDryRunTests(unittest.TestCase):
    def test_dry_run_on_real_inputs_is_clean_writes_nothing_and_leaves_the_agg11_store_alone(self):
        st = G.DEFAULT_AGG11_JOB / "out" / "store.sqlite"
        before = (st.stat().st_mtime_ns, st.stat().st_size, sha(st))
        out = Path(tempfile.mkdtemp()) / "must-not-exist"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--phase", "backfill", "--dry-run", "--orch-root", str(ORCH), "--out", str(out)])
        d = json.loads(buf.getvalue())
        self.assertEqual((rc, d["problems"]), (0, []))
        self.assertEqual((d["model_calls"], d["servers_started"], d["wrote_files"]), (0, 0, 0))
        self.assertFalse(out.exists())
        self.assertEqual(d["plan"]["units"], 12)
        self.assertEqual(d["plan"]["order"][0], "COSTCO_2022_10K")
        self.assertEqual(d["checks"]["backfill_set"]["docs"], sorted(G.EXPECTED_MISSING))
        self.assertEqual(before, (st.stat().st_mtime_ns, st.stat().st_size, sha(st)))


class SeedFromAgg11ReusedRecordTests(unittest.TestCase):
    """Regression: AGG-1.1 results already carry ``reused``/``reused_from`` on their L4 units; seed_units passes both
    explicitly, so copying them through raised ``TypeError: emit() got multiple values for keyword argument 'reused'``."""

    REAL_AGG11_RECORD = json.loads(
        '{"kind": "l4_unit", "ts": 1794409385.2753885, "unit": "l4:capex:AES_2022_10K", "field": "capex", "doc": "AES_2022_10K", "status": "ok", "ok": true, "n_candidates": 6, "n_facts": 1, "n_conflicts": 0, "receipts": [{"stage": "L4a_admit", "calls": 0, "prompt_tokens": 0, "cache_n": 0, "rejected": 0, "notes": []}, {"stage": "L4b_extract", "calls": 6, "prompt_tokens": 7332, "cache_n": 0, "rejected": 0, "notes": []}], "llm_calls": 6, "prefill_tokens": 7332, "wall_s": 34.122517108917236, "final": true, "reused": true, "reused_from": "/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2/out/results.jsonl"}'
    )

    def test_seed_from_record_that_already_has_reused(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "agg11.jsonl"
            src.write_text(json.dumps(self.REAL_AGG11_RECORD) + "\n")
            rec = P.Recorder(td / "out" / "results.jsonl")
            G.seed_units(rec, src)
            got = [r for r in rec.records if r["kind"] == "l4_unit"]
            self.assertEqual(len(got), 1)
            self.assertIs(got[0]["reused"], True)
            self.assertEqual(got[0]["reused_from"], str(src))
            self.assertEqual(got[0]["unit"], self.REAL_AGG11_RECORD["unit"])


if __name__ == "__main__":
    unittest.main()
