"""AGG-1.6 tests (fake backend: no model, no network, no process). Stdlib tests always run; the store / L4 tests import the
orchestrator package and skip without it (``AGG_ORCH_ROOT``, default the AGG-1.12 tree)."""

from __future__ import annotations

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
import agg16 as G  # noqa: E402
import phase0_arms as P  # noqa: E402
import run_agg16 as R  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/worktrees/agg-l3period-orch-20261112"))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import facts_llm  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = True
except Exception:  # pragma: no cover
    HAVE_ORCH = False


class PinsAndTargets(unittest.TestCase):
    def test_pins_match_the_committed_files(self):
        self.assertEqual(G.pin_problems(D1), [])

    def test_a_changed_predeclared_is_refused(self):
        self.assertTrue(G.pin_problems(D1, predeclared_sha="0" * 64))

    def test_the_38_targets_and_their_classes(self):
        t = G.load_targets(D1)
        self.assertEqual(G.targets_problems(t), [])
        self.assertEqual((len(t), sum(x["l4_class"] == "FORMAT" for x in t), sum(x["l4_class"] == "RETRIEVAL" for x in t)), (38, 21, 17))
        self.assertEqual(sum(x["primary"] == "FORMAT" for x in t), 14)
        self.assertEqual(sum(x["store"] == "agg13" for x in t), 10)
        self.assertEqual(len({x["key"] for x in t}), 38)
        # same (document, field) at two fiscal years are two cells
        self.assertEqual({x["fy"] for x in t if x["doc"] == "MICROSOFT_2022_10K" and x["field"] == "capex"}, {"2021", "2022"})

    def test_pilot_is_five_distinct_targets(self):
        t = G.load_targets(D1)
        p = G.pilot_targets(t)
        self.assertEqual((len(p), len({x["key"] for x in p})), (5, 5))
        self.assertTrue(all(x["l4_class"] == "FORMAT" for x in p))

    def test_target_problems_are_reported(self):
        t = G.load_targets(D1)
        self.assertTrue(G.targets_problems(t[:-1]))
        self.assertTrue(G.targets_problems(t + [dict(t[0])]))


class Design(unittest.TestCase):
    def test_constants_equal_the_predeclared_text(self):
        txt = (D1 / "PREDECLARED_AGG16.md").read_text()
        self.assertEqual((G.N_PREDICT_C1, G.N_PREDICT_T, G.T_MAX_ROWS, G.T_QUOTE_CHARS, G.K_FIELD), (512, 1024, 2, 100, 3))
        for needle in ("`n_predict` 512", "`n_predict` 1024", "AT MOST 2 rows", "100 characters", "field-specific top-3",
                       "exactly 5", "fewer than 2 of the 5", "at most 0.01", "at\nleast 10 of the 14", "4,200 s"):
            self.assertIn(needle, txt, needle)
        self.assertEqual((G.TRUNC_MAX, G.PRIMARY_FORMAT_MIN, G.PILOT_MIN_FILLED, G.PILOT_WALL_S), (0.01, 10, 2, 600))

    def test_t_grammar_is_bounded_and_has_every_slot(self):
        g = G.t_grammar()
        for s in G.T_SLOTS:
            self.assertIn(f'"\\"{s}\\":"', g)
        self.assertIn("qchar{1,100}", g)
        self.assertIn("pchar{1,20}", g)
        self.assertIn("ws ::= [ ]?", g)                               # no unbounded whitespace runs
        self.assertIn('rows ::= "[" ws ( row ( "," ws row )? )? ws "]"', g)     # at most 2 rows
        self.assertEqual(g.count("root ::="), 1)
        self.assertEqual(len(G.T_SLOTS), 9)

    VALIDATOR = Path("/mnt/raid0/llm/llama.cpp/build/bin/test-gbnf-validator")

    @unittest.skipUnless(VALIDATOR.exists(), "llama.cpp test-gbnf-validator not on this host")
    def test_t_grammar_parses_and_enforces_its_bounds_in_the_llama_cpp_grammar_engine(self):
        """The real grammar engine (a local tool, no server): a 2-row reply passes; 3 rows and a 101-character quote are refused."""
        import subprocess
        ok = {s: [] for s in G.T_SLOTS}
        row = {"period": "FY2021", "value": 20622, "unit": "USD_millions", "quote": "Additions (20,622", "confidence": "high"}
        cases = {"one.json": ({**ok, "capex": [row]}, True), "two.json": ({**ok, "capex": [row, row]}, True),
                 "three.json": ({**ok, "capex": [row] * 3}, False), "long.json": ({**ok, "capex": [dict(row, quote="x" * 101)]}, False),
                 "edge.json": ({**ok, "capex": [dict(row, quote="x" * 100)]}, True)}
        with tempfile.TemporaryDirectory() as td:
            gram = Path(td) / "t.gbnf"
            gram.write_text(G.t_grammar() + "\n")
            env = dict(os.environ, LD_LIBRARY_PATH=str(self.VALIDATOR.parent))
            for name, (obj, want) in cases.items():
                f = Path(td) / name
                f.write_text(json.dumps(obj))
                r = subprocess.run([str(self.VALIDATOR), str(gram), str(f)], capture_output=True, text=True, timeout=60, env=env)
                self.assertEqual("Input string is valid" in (r.stdout + r.stderr), want, (name, r.stdout[-300:], r.stderr[-300:]))

    def test_t_prompt_names_every_field_at_the_cells_fiscal_year(self):
        p = G.t_prompt("WINDOW TEXT", "2021")
        for s in G.T_SLOTS:
            self.assertIn(f"- {s}:", p)
        self.assertIn("FY2021", p)
        self.assertNotIn("FY2022", p)
        self.assertTrue(p.endswith("WINDOW TEXT"))
        self.assertIn("FY2022", G.t_prompt("w", "2022"))

    def test_parse_typed(self):
        ok = json.dumps({s: [] for s in G.T_SLOTS})
        self.assertEqual(set(G.parse_typed(ok)), set(G.T_SLOTS))
        self.assertIsNone(G.parse_typed("{"))
        self.assertIsNone(G.parse_typed(json.dumps({"capex": []})))                # a missing slot
        self.assertIsNone(G.parse_typed(json.dumps([1])))


class Scoring(unittest.TestCase):
    cell = {"doc": "D", "field": "capex", "fy": "2021", "gold": 20622.0, "key": "D|capex|2021"}

    def test_cell_hit_needs_fy_field_doc_and_value(self):
        f = {"doc_id": "D", "field": "capex", "period": "FY2021", "value": 20622.0, "unit": "USD_millions"}
        self.assertTrue(G.cell_hit([f], self.cell))
        self.assertFalse(G.cell_hit([dict(f, period="2022")], self.cell))
        self.assertFalse(G.cell_hit([dict(f, field="net_income")], self.cell))
        self.assertFalse(G.cell_hit([dict(f, doc_id="E")], self.cell))
        self.assertTrue(G.cell_hit([dict(f, value=-20622.0)], self.cell))              # outflow sign
        self.assertTrue(G.cell_hit([dict(f, value=20650.0)], self.cell))               # within 0.5 percent
        self.assertFalse(G.cell_hit([dict(f, value=21000.0)], self.cell))
        self.assertFalse(G.cell_hit([dict(f, value=20622.0, unit="USD_thousands")], self.cell))   # no leniency for a 1000x slip
        self.assertTrue(G.cell_hit([dict(f, value=20622000.0, unit="USD_thousands")], self.cell))

    def test_bad_replies_counts_a_cut_and_unparseable_reply_once(self):
        self.assertEqual(G._bad(10, 2, 2), 2)
        self.assertEqual(G._bad(10, 1, 3), 3)
        self.assertEqual(G._bad(2, 5, 0), 2)

    def _records(self, arm, hits, bad=0, ext=10):
        tg = G.load_targets(D1)
        recs = []
        for i, t in enumerate(tg):
            recs.append({"kind": "arm_cell", "arm": arm, "cell": t["key"], "final": True, "hit": hits(t, i), "calls": 6,
                         "gate_calls": 0, "extract_calls": ext, "prefill_tokens": 1000, "cache_n": 0, "decode_tokens": 5,
                         "bad_replies": bad, "cut": bad, "unparseable": bad, "wall_s": 10.0})
        return tg, recs

    def test_arm_summary_and_the_done_when_verdict(self):
        tg, rec = self._records("C1", lambda t, i: t["primary"] == "FORMAT", bad=0)
        s = G.arm_summary(rec, "C1", tg)
        self.assertEqual((s["n_units"], s["fill"]["filled"], s["fill_primary_format"]["filled"], s["fill_primary_format"]["cells"]),
                         (38, 14, 14, 14))
        self.assertEqual((s["calls_per_unit"], s["prefill_per_unit"], s["bad_rate"]), (6.0, 1000.0, 0.0))
        v = G.verdict({"C1": s})
        self.assertTrue(v["met"] and v["winner"] == "C1")
        # rate above 1 percent -> not met even with every FORMAT cell filled
        _, rec2 = self._records("C1", lambda t, i: t["primary"] == "FORMAT", bad=1, ext=10)
        self.assertFalse(G.verdict({"C1": G.arm_summary(rec2, "C1", tg)})["met"])
        # only 9 of the 14 primary FORMAT cells -> not met
        first9 = [t["key"] for t in tg if t["primary"] == "FORMAT"][:9]
        _, rec3 = self._records("C1", lambda t, i: t["key"] in first9)
        v3 = G.verdict({"C1": G.arm_summary(rec3, "C1", tg)})
        self.assertFalse(v3["met"])
        self.assertEqual(v3["primary_format_filled"], 9)

    def test_winner_is_the_arm_with_more_primary_format_fills_then_lower_rate(self):
        tg, c1 = self._records("C1", lambda t, i: t["primary"] == "FORMAT")
        _, t_ = self._records("T", lambda t, i: t["primary"] == "FORMAT", bad=1)
        v = G.verdict({"C1": G.arm_summary(c1, "C1", tg), "T": G.arm_summary(t_, "T", tg)})
        self.assertEqual(v["winner"], "C1")                       # tie on fills, C1 has the lower bad rate
        self.assertEqual(G.verdict({})["winner"], None)

    def test_pilot_eval_continue_futility_and_budget(self):
        tg = G.load_targets(D1)
        pilot = G.pilot_targets(tg)
        recs = [{"kind": "arm_cell", "arm": "C1", "cell": p["key"], "final": True, "hit": i < 3, "wall_s": 100.0, "calls": 7}
                for i, p in enumerate(pilot)]
        recs += [{"kind": "arm_cell", "arm": "T", "cell": p["key"], "final": True, "hit": i < 1, "wall_s": 50.0, "calls": 4}
                 for i, p in enumerate(pilot)]
        ev = G.pilot_eval(recs, pilot, booked_s=4200)
        self.assertEqual((ev["arms"]["C1"]["filled"], ev["arms"]["T"]["filled"]), (3, 1))
        self.assertEqual((ev["arms"]["C1"]["continue"], ev["arms"]["T"]["continue"], ev["futility"]), (True, False, False))
        self.assertIn("pilot_wall_over_600s", ev["budget_stops"])                  # 750 s
        self.assertEqual(ev["wall_s"], 750.0)
        none = [dict(r, hit=False) for r in recs]
        self.assertTrue(G.pilot_eval(none, pilot, booked_s=4200)["futility"])

    def test_c0_summary_reads_the_saved_receipts(self):
        t = [{"key": "D|capex|2022", "doc": "D", "field": "capex", "group": "c"}]
        rec = {"kind": "l4_unit", "unit": "l4:capex:D", "ok": True, "llm_calls": 7, "prefill_tokens": 9000,
               "receipts": [{"stage": "L4a_admit", "calls": 1, "notes": []},
                            {"stage": "L4b_extract", "calls": 6, "notes": ["unparseable:x", "unparseable:y"]}]}
        s = G.c0_summary(t, {"c": [rec]})
        self.assertEqual((s["calls"], s["extract_calls"], s["unparseable"], s["fill"]), (7, 6, 2, 0))
        self.assertAlmostEqual(s["bad_rate"], 2 / 6)
        self.assertEqual(G.c0_summary(t, {"c": []})["missing"], ["D|capex|2022"])

    def test_estimates_are_labelled_and_use_the_measured_rates(self):
        e = G.estimates({"t_calls": 201})
        self.assertEqual(e["label"], "ESTIMATE")
        self.assertEqual(e["c1_calls"], 38 * 6.0)
        self.assertAlmostEqual(e["t_s"], 201 * 8.2, places=1)


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not available")
class WithOrchestrator(unittest.TestCase):
    def test_the_stage_pin_matches_the_orchestrator_tree(self):
        self.assertEqual(G.stage_problems(ORCH), [])

    def test_typed_window_facts_equal_the_unchanged_extract(self):
        win = {"chunk_id": "h:00001#w0", "doc_hash": "h", "doc_id": "D", "entity": "D", "page_start": 5,
               "text": "(in millions)\nTotal assets $ 100 $ 90\nAdditions to property and equipment (20,622) (18,000)\n"}
        row = {"period": "2021", "value": 20622, "unit": "USD_millions", "quote": "Additions to property and equipment (20,622",
               "confidence": "high"}
        obj = {s: [] for s in G.T_SLOTS}
        obj["capex"] = [row, dict(row, quote="not in the window")]
        st = CorpusStore(":memory:")
        bf = facts_llm.LazyFieldBackfill(st, mock.Mock(), n_predict=1)
        got = G.typed_window_facts(win, obj, "2021", bf)
        r = facts_llm.StageReceipt("x")
        want = facts_llm.extract(win, "capex", G.field_desc("capex", "2021"), G._Replay(json.dumps([row])), n_predict=0,
                                 id_slot=None, receipt=r)
        self.assertEqual(len(got), 1)                                              # the quote check drops the invented row
        self.assertEqual({k: v for k, v in got[0].items() if k not in ("declared_scale", "declared_scale_basis")},
                         {k: v for k, v in want[0].items() if k not in ("declared_scale", "declared_scale_basis")})
        self.assertEqual(got[0]["declared_scale"], "USD_millions")

    def test_recording_backend_logs_stage_and_finish_reason(self):
        inner = mock.Mock()
        inner.complete.side_effect = [{"text": "yes", "prompt_tokens": 5, "cache_n": 0},
                                      {"text": "[]", "prompt_tokens": 100, "cache_n": 7, "decode_tokens": 2, "finish_reason": "length"}]
        rb = G.RecordingBackend(inner)
        rb.complete("p", grammar="g", n_predict=1, cache_prompt=True, id_slot=None)
        rb.complete("p", grammar="g", n_predict=512, cache_prompt=True, id_slot=None)
        cs = G.calls_stats(rb.log)
        self.assertEqual((cs["calls"], cs["gate_calls"], cs["extract_calls"], cs["cut"], cs["prefill_tokens"]), (2, 1, 1, 1, 105))


class FakeBackend:
    """Answers the three prompt shapes from the excerpt: the gate says yes, an L4b prompt returns the matching row, a typed prompt
    returns the typed object. ``cut_first`` makes the first extraction reply a cut (unparseable) one."""

    def __init__(self, cut_first: bool = False) -> None:
        self.n = 0
        self.cut_first = cut_first

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self.n += 1
        if n_predict == 1:
            return {"text": "yes", "prompt_tokens": 10, "cache_n": 0}
        if self.cut_first and self.n == 1:
            return {"text": '[{"period": "FY2022", "va', "prompt_tokens": 300, "cache_n": 0, "decode_tokens": n_predict,
                    "finish_reason": "length"}
        excerpt = prompt.split("Excerpt:\n", 1)[1]
        m = re.search(r"Total assets \$ ([\d,]+)", excerpt)
        rows = []
        if m:
            rows = [{"period": "FY2022", "value": float(m.group(1).replace(",", "")), "unit": "USD_millions",
                     "quote": m.group(0), "confidence": "high"}]
        if grammar.startswith("root ::= \"{\""):
            obj = {s: [] for s in G.T_SLOTS}
            obj["total_assets"] = rows
            text = json.dumps(obj)
        else:
            text = json.dumps(rows)
        return {"text": text, "prompt_tokens": 300, "cache_n": 0, "decode_tokens": 20, "finish_reason": "stop"}


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not available")
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.seed = self.dir / "seed.sqlite"
        st = CorpusStore(self.seed)
        for doc, n in (("DOCA_2022_10K", 100), ("DOCB_2022_10K", 200)):
            body = "\n".join(f"filler line {i} about nothing" for i in range(40))
            sec = Section(id="s1", title="Consolidated Balance Sheets", level=1, page_start=3, page_end=3,
                          content=f"(in millions)\n{body}\nTotal assets $ {n} $ {n - 10}\nTotal assets and liabilities table")
            st.ingest(DocumentPreprocessResult(original_path=f"/x/{doc}.pdf", sections=[sec], figures=[], total_pages=10),
                      corpus_id=P.CORPUS_ID, doc_id=doc, entity=doc)
        st.close()
        self.targets = [{"doc": d, "field": "total_assets", "fy": "2022", "gold": g, "group": "c", "store": "agg11",
                         "l4_class": c, "primary": c, "key": f"{d}|total_assets|2022"}
                        for d, g, c in (("DOCA_2022_10K", 100.0, "FORMAT"), ("DOCB_2022_10K", 200.0, "RETRIEVAL"))]
        self.inp = {"targets": self.targets, "corpus": {}, "set_dir": D1}
        self.pilot = (("DOCA_2022_10K", "total_assets", "2022"),)
        (self.dir / "c0.jsonl").write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, backend, out, *, smoke=False):
        rec = P.Recorder(out / "results.jsonl")
        with mock.patch.object(G, "PILOT_CELLS", self.pilot), mock.patch.object(G, "N_TARGETS", 2), \
                mock.patch.object(G, "PILOT_MIN_FILLED", 1), mock.patch.object(G, "universes", lambda sd: {"agg11": [], "agg13": []}):
            return rec, G.run_ab(rec, out, self.inp, backend, seeds={"agg11": self.seed, "agg13": self.seed},
                                 c0_results={"c": self.dir / "c0.jsonl"}, guard=lambda: None, booked_s=4200.0, smoke=smoke)

    def test_smoke_runs_one_c1_cell_and_nothing_else(self):
        out = self.dir / "o1"
        be = FakeBackend()
        rec, res = self.run_it(be, out, smoke=True)
        self.assertEqual(res["rc"], 0)
        cells = [r for r in rec.records if r["kind"] == "arm_cell"]
        self.assertEqual([(r["arm"], r["cell"]) for r in cells], [("C1", "DOCA_2022_10K|total_assets|2022")])
        self.assertTrue(cells[0]["ok"] and cells[0]["hit"] and cells[0]["calls"] >= 1 and cells[0]["prefill_tokens"] > 0)
        self.assertFalse((out / "result.cpu.json").exists())

    def test_full_run_files_a_verdict_and_resume_repeats_nothing(self):
        out = self.dir / "o2"
        be = FakeBackend()
        rec, res = self.run_it(be, out)
        self.assertTrue(res["complete"])
        self.assertIn(res["verdict"], ("agg_1_6_met", "agg_1_6_not_met"))
        kinds = [r["kind"] for r in rec.records]
        self.assertIn("pilot_done", [r.get("unit") for r in rec.records])
        arm = {a: {r["cell"] for r in rec.records if r["kind"] == "arm_cell" and r["arm"] == a} for a in ("C1", "T")}
        self.assertEqual(arm["C1"], {t["key"] for t in self.targets})
        self.assertEqual(arm["T"], {t["key"] for t in self.targets})
        s = res["summaries"]
        self.assertEqual((s["C1"]["fill"]["filled"], s["T"]["fill"]["filled"]), (2, 2))
        self.assertEqual(s["C1"]["bad_rate"], 0.0)
        self.assertEqual(res["non_target_effects"]["C1"]["n_broken"], 0)
        n1 = be.n
        rec2, res2 = self.run_it(be, out)
        self.assertEqual(be.n, n1)                                                 # resume: no finished unit is repeated
        self.assertEqual(res2["verdict"], res["verdict"])
        self.assertTrue((out / "result.cpu.json").exists())
        self.assertEqual(kinds.count("t_window"), len([r for r in rec2.records if r["kind"] == "t_window"]))

    def test_t_window_shared_by_two_cells_is_called_once(self):
        out = self.dir / "o3"
        extra = dict(self.targets[0], field="net_income", key="DOCA_2022_10K|net_income|2022")
        self.inp["targets"] = self.targets + [extra]
        be = FakeBackend()
        rec, _res = self.run_it(be, out)
        tw = [r for r in rec.records if r["kind"] == "t_window" and r["doc"] == "DOCA_2022_10K"]
        self.assertEqual(len({r["unit"] for r in tw}), len(tw))                      # each window once
        cells = {r["cell"]: r for r in rec.records if r["kind"] == "arm_cell" and r["arm"] == "T"}
        a, b = cells["DOCA_2022_10K|total_assets|2022"], cells["DOCA_2022_10K|net_income|2022"]
        self.assertEqual(a["fresh_windows"] + b["fresh_windows"], len(tw))           # charged to the first cell that needed it

    def test_a_cut_reply_is_counted_and_pilot_futility_stops_the_row(self):
        out = self.dir / "o4"

        class Never(FakeBackend):
            def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
                r = super().complete(prompt, grammar=grammar, n_predict=n_predict, cache_prompt=cache_prompt, id_slot=id_slot)
                if n_predict > 1:
                    r["text"] = "[]" if not grammar.startswith("root ::= \"{\"") else json.dumps({s: [] for s in G.T_SLOTS})
                return r
        rec, res = self.run_it(Never(), out)
        self.assertEqual((res["verdict"], res["rc"]), ("phase1_gate_tripped", G.RC_GATE))
        rec_c = [r for r in rec.records if r["kind"] == "arm_cell" and r["arm"] == "C1"]
        self.assertEqual(len(rec_c), 1)                                              # no non-pilot cell ran

    def test_c1_counts_a_cut_reply(self):
        out = self.dir / "o5"
        be = FakeBackend(cut_first=True)
        rec, _res = self.run_it(be, out, smoke=True)
        r = [x for x in rec.records if x["kind"] == "arm_cell"][0]
        self.assertEqual((r["cut"], r["unparseable"], r["bad_replies"]), (1, 1, 1))


class Cli(unittest.TestCase):
    def test_dry_run_prints_the_plan_and_writes_nothing(self):
        if not HAVE_ORCH or not R.DEFAULT_SEEDS["agg11"].exists():
            self.skipTest("seed stores or orchestrator not available on this host")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--phase", "plan", "--dry-run", "--pins", "none", "--orch-root", str(ORCH)])
        out = json.loads(buf.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual((out["model_calls"], out["servers_started"], out["wrote_files"]), (0, 0, 0))
        self.assertEqual(out["targets"]["n"], 38)
        self.assertEqual(out["plan"]["estimate"]["label"], "ESTIMATE")


if __name__ == "__main__":
    unittest.main()
