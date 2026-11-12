"""AGG-1.9 tests (fake backend: no model, no network, no process, no PDF).

Stdlib tests always run. The store / L4 tests import the orchestrator package and skip without it: run them with the
orchestrator venv python and ``AGG_ORCH_ROOT=<tree at the AGG-1.9 pin>`` (the tree must carry ``store.CANONICAL_RULE_V2``).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg19 as G  # noqa: E402
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_agg19 as R  # noqa: E402

DEFAULT_ORCH = "/mnt/raid0/llm/worktrees/agg19-110-orch"
ORCH = Path(os.environ.get("AGG_ORCH_ROOT", DEFAULT_ORCH))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import facts_llm  # noqa: E402,F401
    from src.aggregation.llm_backend import GATE_GBNF  # noqa: E402
    from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2, CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = True
except Exception:  # pragma: no cover
    HAVE_ORCH = False


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class PinAndInputTests(unittest.TestCase):
    def test_pins_match(self):
        self.assertEqual(G.pin_problems(D1), [])
        self.assertEqual(sha(D1 / "PREDECLARED_AGG19.md"), G.PREDECLARED_SHA)
        self.assertEqual(sha(D1 / G.AMENDMENT1_FILE), G.AMENDMENT1_SHA)
        self.assertEqual(sha(D1 / G.PROBE_FILE), G.PROBE_SHA)
        self.assertEqual(sha(D1 / "PREDECLARED.md"), X.PREDECLARED_SHA)

    def test_a_changed_pin_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        for f in list(G.FROZEN) + ["PREDECLARED_AGG19.md", G.AMENDMENT1_FILE, G.PROBE_FILE]:
            shutil.copy(D1 / f, tmp / f)
        self.assertEqual(G.pin_problems(tmp), [])
        (tmp / "agg19_cells.csv").write_text((tmp / "agg19_cells.csv").read_text() + "x")
        self.assertTrue(any("agg19_cells.csv" in p for p in G.pin_problems(tmp)))
        (tmp / G.AMENDMENT1_FILE).unlink()
        self.assertTrue(any(G.AMENDMENT1_FILE in p for p in G.pin_problems(tmp)))

    def test_the_26_cells_are_the_agg18_never_extracted_cells(self):
        inp = G.load_inputs(D1)
        self.assertEqual(G.cells_problems(inp), [])
        self.assertEqual(len(inp["cells"]), 26)
        miss = json.loads((D1 / "agg18_miss_table.json").read_text())
        want = {(b["doc"], b["field"]) for b in miss["bad_cells"]
                if not b["correct_candidate_in_facts"] and b["canonical_state"] != "spurious"}
        self.assertEqual({(c["doc"], c["field"]) for c in inp["cells"]}, want)
        self.assertEqual(len({G.unit_of(c) for c in inp["cells"]}), 26)
        self.assertEqual(len(inp["cells211"]), 211)
        self.assertEqual(len(inp["store_gap_qids"]), 14)
        self.assertEqual([G.unit_of(c) for c in G.pilot_cells(inp)],
                         ["l4:net_income:3M_2022_10K", "l4:capex:AMERICANEXPRESS_2022_10K", "l4:net_income:COCACOLA_2022_10K",
                          "l4:net_income:COSTCO_2022_10K", "l4:total_assets:MGMRESORTS_2022_10K"])

    def test_cell_golds_must_equal_the_verified_cells(self):
        inp = G.load_inputs(D1)
        bad = dict(inp, cells=[dict(inp["cells"][0], gold=inp["cells"][0]["gold"] + 1)] + inp["cells"][1:])
        self.assertTrue(any("is not the verified 211-cell value" in p for p in G.cells_problems(bad)))

    def test_gates_and_stops_are_the_boxes(self):
        self.assertEqual((G.M1_GATE, G.PILOT_MIN_FILLED, G.PILOT_WALL_S, G.PILOT_INDEX), (0.75, 2, 300, (0, 5, 10, 15, 20)))
        text = (D1 / "PREDECLARED_AGG19.md").read_text()
        for needle in ("5 cells", "300 s", "fewer than 2 of 5", "futility-only", "M1 over 211", "14 saved store-gap SQLs",
                       "no retained cell regresses", "ESTIMATE", "post-hoc"):
            self.assertIn(needle.lower(), text.lower(), needle)

    def test_pilot_eval_arithmetic(self):
        self.assertEqual(G.pilot_eval(2, 5, 200.0, 26)["gates_tripped"], [])
        self.assertEqual(G.pilot_eval(1, 5, 200.0, 26)["gates_tripped"], ["pilot_fills_lt_2_of_5"])
        self.assertEqual(G.pilot_eval(5, 5, 301.0, 26)["budget_stops"], ["pilot_wall_over_300s"])
        self.assertEqual(G.pilot_eval(5, 5, 300.0, 26)["budget_stops"], [])
        self.assertEqual(G.pilot_eval(5, 5, 270.0, 26)["budget_stops"], [])          # measured rate: 1,404 s <= 2,800 s
        self.assertEqual(G.pilot_eval(5, 5, 300.0, 70)["budget_stops"], ["projected_wall_over_2x_booked"])

    def test_sql_scorer_matches_agg18(self):
        sc = G.sql_scores
        self.assertTrue(sc([("Amazon",)], {"op": "argmax", "gold": "amazon"}))
        self.assertFalse(sc([("Apple",)], {"op": "argmax", "gold": "amazon"}))
        self.assertTrue(sc([("A", 1), ("B", 2), ("C", 3)], {"op": "top3", "gold": ["a", "b", "c"]}))
        self.assertTrue(sc([(7,)], {"op": "count_gt", "gold": 7}))
        self.assertFalse(sc([(7.001,)], {"op": "count_gt", "gold": 7}))
        self.assertTrue(sc([(1000.0,)], {"op": "sum", "gold": 1004.0}))              # 0.4 percent
        self.assertFalse(sc([(1000.0,)], {"op": "sum", "gold": 1006.0}))
        self.assertFalse(sc([], {"op": "sum", "gold": 1.0}))
        self.assertFalse(sc(None, {"op": "sum", "gold": 1.0}))

    def test_regressions_count_both_measures(self):
        before = {"canonical_ok_cells": ["a|x", "b|x"], "raw_hit_cells": ["a|x", "b|x", "c|x"]}
        self.assertTrue(G.regressions(before, {"a|x": True, "b|x": True}, {"a|x": True, "b|x": True, "c|x": True})["ok"])
        r = G.regressions(before, {"a|x": True, "b|x": False}, {"a|x": True, "b|x": True, "c|x": False})
        self.assertEqual((r["ok"], r["canonical_regressed"], r["raw_regressed"]), (False, ["b|x"], ["c|x"]))


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with canonical.v2 not importable")
class ProbeTests(unittest.TestCase):
    def test_probe_on_the_real_agg11_store_is_the_frozen_file(self):
        base = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/store.sqlite")
        ans = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/results.jsonl")
        if not base.exists() or not ans.exists():
            self.skipTest("AGG-1.1 store not present")
        live = G.probe_base(base, G.load_inputs(D1), ans, X.CORPUS_ID)
        self.assertEqual(json.loads(json.dumps(live, default=str)), json.loads((D1 / G.PROBE_FILE).read_text()))
        self.assertEqual((live["raw_hits"]["found"], live["control"]["n_ok"]), (185, 177))
        self.assertEqual(live["adopted_guards"], ["attributable_net_income"])
        v = live["variants"]
        self.assertEqual((len(v["attributable_net_income"]["fixed"]), len(v["attributable_net_income"]["broken"])), (4, 0))
        self.assertEqual((len(v["thousands_unit"]["fixed"]), len(v["thousands_unit"]["broken"])), (0, 0))
        self.assertEqual(live["saved_sql_on_agg11_store"]["n_correct"], 0)

    def test_adoption_rule_is_fix_at_least_one_and_break_none(self):
        rows = [{"corpus_id": "c", "fact_id": "a", "doc_hash": "h", "doc_id": "D", "entity": "Delta", "period": "2022",
                 "field": "net_income", "value": 50.0, "unit": "USD_millions", "quote": "Net income 50", "page": 1,
                 "confidence": "high", "extractor": "l3_textrow.v1", "audit_status": "unaudited"},
                {"corpus_id": "c", "fact_id": "b", "doc_hash": "h", "doc_id": "D", "entity": "Delta", "period": "2022",
                 "field": "net_income", "value": 40.0, "unit": "USD_millions",
                 "quote": "Net income attributable to Delta 40", "page": 2, "confidence": "high",
                 "extractor": "l4_llm.v1", "audit_status": "unaudited"}]
        t = G.variant_table(rows, [("D", "net_income", 40.0)])
        self.assertEqual(t["adopted_guards"], ["attributable_net_income"])
        t = G.variant_table(rows, [("D", "net_income", 50.0)])                       # the guard would break this cell
        self.assertEqual((t["adopted_guards"], t["variants"]["attributable_net_income"]["broken"]),
                         ([], ["D|net_income"]))


# --------------------------------------------------------------------------------------------- fixture
DOCS = ["AAA_2022_10K", "BBB_2022_10K", "CCC_2022_10K"]
REV = {"AAA_2022_10K": 100.0, "BBB_2022_10K": 200.0, "CCC_2022_10K": 300.0}
NI = {"AAA_2022_10K": 10.0, "BBB_2022_10K": 20.0, "CCC_2022_10K": 30.0}


class FakeBackend:
    """Revenue rows always; net income rows only when ``net_income`` is True (the AGG-1.1 run missed them)."""

    def __init__(self, net_income: bool = True):
        self.calls = self.prefill_tokens = 0
        self.net_income = net_income

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4
        if grammar == GATE_GBNF:
            return {"text": "yes", "prompt_tokens": len(prompt) // 4, "cache_n": 0}
        body = prompt.split("Excerpt:\n", 1)[-1]
        rows = []
        for fld, label in (("revenue", "Total revenues"), ("net_income", "Net income attributable to shareholders")):
            m = re.search(rf"{label} [\d,]+", body)
            if m and f"Field: {fld}" in prompt and (fld != "net_income" or self.net_income):
                rows.append({"period": "FY2022", "value": float(m.group(0).split()[-1].replace(",", "")),
                             "unit": "USD_millions", "quote": m.group(0), "confidence": "medium"})
        return {"text": json.dumps(rows), "prompt_tokens": len(prompt) // 4, "cache_n": 0}


def doc(name: str):
    secs = [Section(id="s1", title="Income", level=1, page_start=60, page_end=60,
                    content=f"Income statement FY2022\n(in millions)\nTotal revenues {REV[name]:,.0f}\n"
                            f"Net income attributable to shareholders {NI[name]:,.0f}\n")]
    return DocumentPreprocessResult(original_path=f"/{name}.pdf", sections=secs, figures=[], total_pages=100, tables=[])


def make_fixture(tmp: Path) -> dict:
    base_dir = tmp / "agg11"
    base_dir.mkdir()
    store = CorpusStore(base_dir / "store.sqlite")
    for n in DOCS:
        store.ingest(doc(n), corpus_id=P.CORPUS_ID, doc_id=n, entity=n.split("_")[0])
    store.conn.execute("UPDATE documents SET n_tokens_est = 400000")
    store.conn.commit()
    rec = P.Recorder(base_dir / "results.jsonl")
    units = P.windows(store, P.CORPUS_ID, P.CFG.window_chars)
    b = FakeBackend(net_income=False)
    for f in A0.METRICS:
        for n in DOCS:
            X.l4_unit(rec, store, b, units, f, n, lambda: None)
    store.refresh_canonical(P.CORPUS_ID)
    store.close()
    cells = [{"doc": n, "field": "net_income", "fy": "2022", "gold": NI[n], "state": "missing"} for n in DOCS]
    cells211 = [(n, "revenue", REV[n]) for n in DOCS] + [(n, "net_income", NI[n]) for n in DOCS]
    answers = base_dir / "answers.jsonl"
    answers.write_text(json.dumps({"kind": "answer", "arm": "P-large", "final": True, "qid": "q1", "correct": False,
                                   "sql": "SELECT SUM(value) FROM facts_canonical WHERE field = 'net_income' "
                                          "AND period = '2022'"}) + "\n")
    golds = {"q1": {"id": "q1", "op": "sum", "gold": 60.0}}
    inp = {"cells": sorted(cells, key=lambda c: (c["doc"], c["field"])), "cells211": cells211, "golds": golds,
           "store_gap_qids": ["q1"], "corpus": {"docs": [{"doc_name": n} for n in DOCS]}}
    return {"inp": inp, "base": base_dir / "store.sqlite", "results11": base_dir / "results.jsonl", "answers": answers}


PROBE = {"adopted_guards": ["attributable_net_income"]}


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with canonical.v2 not importable")
class FillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.guard = lambda: None
        self.patches = [mock.patch.object(G, "PILOT_INDEX", (0, 1))]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def _run(self, fx, out, backend=None, *, smoke=False, **kw):
        rec = P.Recorder(out / "results.jsonl")
        backend = backend or FakeBackend()
        res = G.run_fill(rec, out, fx["inp"], Path("/unused"), backend, base_store=fx["base"],
                         agg11_results=fx["results11"], agg11_answers=fx["answers"], guard=self.guard, probe=PROBE,
                         smoke=smoke, **kw)
        return rec, res, backend

    def test_fill_runs_only_the_targets_and_reports_m1_before_after_and_the_sql(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out)
        self.assertEqual(res["rc"], 0, res)
        self.assertEqual((res["verdict"], res["gate_tripped"]), ("agg_1_9_done", []))
        fresh = [x for x in rec.records if x["kind"] == "l4_unit" and not x.get("reused")]
        self.assertEqual([(x["field"], x["doc"]) for x in fresh], [("net_income", d) for d in DOCS])     # pilot (2) first
        seeded = [x for x in rec.records if x["kind"] == "l4_unit" and x.get("reused")]
        self.assertEqual(len(seeded), len(A0.METRICS) * 3 - 3)                                           # the 3 targets not seeded
        self.assertFalse({x["unit"] for x in seeded} & {f"l4:net_income:{d}" for d in DOCS})
        m = res["m1_detail"]
        self.assertEqual((m["before"]["found"], m["after"]["found"], m["before"]["n"]), (3, 6, 6))      # 3 revenue cells; +3 net income
        self.assertEqual(res["n_filled"], 3)
        self.assertTrue(all(c["filled"] for c in res["per_cell"]))
        self.assertTrue(res["regressions"]["ok"], res["regressions"])
        # the saved store-gap SQL now answers: SUM(net_income) = 60 over the three filled cells
        self.assertEqual((res["sql14"]["n"], res["sql14"]["n_correct"], res["sql14"]["agg11_correct"]), (1, 1, 0))
        self.assertEqual(res["canonical"]["rule"], CANONICAL_RULE_V2)
        self.assertEqual(res["canonical"]["guards"], ["attributable_net_income"])
        # the base store was only read
        base = CorpusStore(fx["base"])
        self.assertEqual(base.conn.execute("SELECT COUNT(*) FROM facts WHERE field = 'net_income'").fetchone()[0], 0)
        # resume: zero new calls
        b2 = FakeBackend()
        self.assertEqual(self._run(fx, out, b2)[1]["rc"], 0)
        self.assertEqual(b2.calls, 0)

    def test_pilot_that_fills_fewer_than_two_stops_on_futility(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, FakeBackend(net_income=False))                  # the model still cannot read them
        self.assertEqual((res["rc"], res["gate_tripped"], res["where"]), (G.RC_GATE, ["pilot_fills_lt_2_of_5"], "pilot"))
        self.assertEqual(res["verdict"], "phase1_gate_tripped")
        self.assertEqual(len([x for x in rec.records if x["kind"] == "l4_unit" and not x.get("reused")]), 2)   # nothing after the pilot

    def test_pilot_wall_over_bound_is_a_budget_stop_not_a_gate(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        with mock.patch.object(G, "PILOT_WALL_S", -1):
            rec, res, b = self._run(fx, out)
        self.assertEqual(res["rc"], G.RC_BUDGET)
        self.assertIn("pilot_wall_over_300s", res["budget_stop"])

    def test_regression_trips_the_gate_and_is_reported_cell_by_cell(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        bad = {"ok": False, "canonical_regressed": ["AAA_2022_10K|revenue"], "raw_regressed": [],
               "n_retained_canonical": 3, "n_retained_raw": 3}
        with mock.patch.object(G, "regressions", lambda *a, **k: bad):
            rec, res, b = self._run(fx, out)
        self.assertEqual((res["rc"], res["gate_tripped"], res["verdict"]),
                         (G.RC_GATE, ["retained_cell_regression"], "phase1_gate_tripped"))
        self.assertEqual(res["regressions"]["canonical_regressed"], ["AAA_2022_10K|revenue"])

    def test_smoke_runs_one_real_unit_and_stops(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, smoke=True)
        self.assertEqual((res["rc"], res.get("smoke")), (0, True))
        fresh = [x for x in rec.records if x["kind"] == "l4_unit" and not x.get("reused")]
        self.assertEqual(len(fresh), 1)
        self.assertTrue(fresh[0]["ok"] and fresh[0]["llm_calls"] >= 1 and fresh[0]["prefill_tokens"] > 0)

    def test_l4_reuse_failure_is_a_precondition_stop(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, reuse_ok=False)
        self.assertEqual((res["complete"], res["rc"]), (False, G.RC_PRECOND))
        self.assertEqual(b.calls, 0)


class CliDryRunTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ORCH, "orchestrator tree with canonical.v2 not importable")
    def test_dry_run_prints_the_plan_and_writes_nothing(self):
        tmp = Path(tempfile.mkdtemp())
        fx = make_fixture(tmp)
        out = tmp / "never"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--phase", "fill", "--dry-run", "--pins", "none", "--orch-root", str(ORCH),
                         "--base-store", str(fx["base"]), "--agg11-job", str(tmp / "nojob"), "--out", str(out)])
        self.assertEqual(rc, 0, buf.getvalue()[:2000])
        o = json.loads(buf.getvalue())
        self.assertEqual((o["dry_run"], o["model_calls"], o["servers_started"], o["wrote_files"]), (True, 0, 0, 0))
        self.assertEqual((o["plan"]["units"], len(o["plan"]["pilot"]), o["plan"]["l4_units_seeded_as_cache_hits"]), (26, 5, 198))
        self.assertFalse(out.exists())

    def test_missing_cells_file_is_a_preflight_failure(self):
        tmp = Path(tempfile.mkdtemp())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--phase", "fill", "--dry-run", "--set-dir", str(tmp)])
        self.assertEqual(rc, X.RC_PREFLIGHT)


if __name__ == "__main__":
    unittest.main()
