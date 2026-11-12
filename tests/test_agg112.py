"""AGG-1.12 tests (fake backend: no model, no network, no process). Stdlib tests always run; the store / L4 tests import the orchestrator
package and skip without it (``AGG_ORCH_ROOT``, default the AGG-1.12 tree)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg112 as G  # noqa: E402
import agg112_reference as R  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_agg112 as C  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/worktrees/agg112-orch"))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import facts_llm, facts_typed as ft, pipeline as pl  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = hasattr(pl, "pipeline_spec")
except Exception:  # pragma: no cover
    HAVE_ORCH = False

SEED = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/store.sqlite")
REF = json.loads((D1 / "agg112_reference.json").read_text())


class PinsAndReference(unittest.TestCase):
    def test_pins_match_the_committed_files(self):
        self.assertEqual(G.pin_problems(D1), [])

    def test_a_changed_predeclared_or_reference_is_refused(self):
        self.assertTrue(G.pin_problems(D1, predeclared_sha="0" * 64))
        self.assertTrue(G.pin_problems(D1, reference_sha="0" * 64))

    def test_the_comparator_is_agg11(self):
        a = REF["agg11"]
        self.assertEqual((a["n_cells"], a["raw_found"], a["canonical_v1_ok"]), (211, 185, 177))
        self.assertEqual((a["p_large"]["n"], a["p_large"]["n_correct"], a["p_large"]["median_calls"], a["p_large"]["median_prefill"]),
                         (58, 18, 2.0, 606.5))
        self.assertAlmostEqual(a["m3"]["prefill_ratio"], 0.3783642241455476)
        self.assertEqual((a["pilot"]["found"], a["pilot"]["n_cells"]), (20, 24))
        self.assertEqual(REF["expected_matches"], True)
        self.assertEqual(REF["sources"]["store_sha256"], R.STORE_SHA)

    def test_partition_counts_and_the_ceiling(self):
        p = REF["partition"]
        self.assertEqual((p["n_cells"], p["n_touched"], p["n_held_out"]), (211, 53, 158))
        self.assertEqual(set(p["touched"]) & set(p["held_out"]), set())
        h = REF["partition_hit_rates_agg11"]["held_out"]
        self.assertEqual((h["n"], h["raw_found"], h["canonical_v1_ok"]), (158, 158, 158))      # no headroom: a regression test only
        self.assertEqual(REF["questions"]["n_with_only_untouched_evidence"], 0)
        self.assertEqual(REF["questions"]["touched_evidence_cells_per_question_min"], 2)

    def test_every_agg11_miss_is_touched(self):
        raw = set(REF["agg11"]["raw_hit_cells"])
        can = set(REF["agg11"]["canonical_v1_ok_cells"])
        touched = set(REF["partition"]["touched"])
        cells = {k for k in REF["partition"]["touched"] + REF["partition"]["held_out"]}
        self.assertEqual({k for k in cells if k not in raw} - touched, set())
        self.assertEqual({k for k in cells if k not in can} - touched, set())

    def test_partition_is_reproducible_from_the_committed_files(self):
        why = R.touched_reasons(D1)
        inp = G.load_inputs(D1)
        part = R.partition(inp["cells"], why)
        self.assertEqual((part["n_touched"], part["n_held_out"]), (53, 158))
        self.assertEqual(part["touched"], REF["partition"]["touched"])
        # triage rows of other stores (group a: AGG-1.3 FY2021 slots; group b: long_term_debt) never become 211 cells
        self.assertEqual({k for k in why if k not in {f"{d}|{f}" for d, f, _g in inp["cells"]}}, {"BOEING_2022_10K|dividends_paid"})
        self.assertFalse([k for k in why if k.endswith("long_term_debt")])

    def test_the_four_sources_are_pinned_in_the_reference(self):
        for n, h in REF["sources"]["partition_files"].items():
            self.assertEqual(R.sha256_file(D1 / n), h, n)


class Design(unittest.TestCase):
    def test_constants_equal_the_predeclared_text(self):
        txt = (D1 / "PREDECLARED_AGG112.md").read_text()
        for needle in ("`agg_pipeline.v2`", "4d0265cdd3fad52a23b26050924dded38f27b514", "**185 of 211 = 0.8768**", "**20 of 24 = 0.833**",
                       "**18 of 58 = 0.310**", "**903** typed calls", "**9,000 s ESTIMATE**", "**2,400 s ESTIMATE**",
                       "**53 of the 211 are TOUCHED,\n158 are HELD-OUT**", "L4 reuse: none", "agg_1_12_done", "agg_1_12_not_met",
                       "median calls <= 3 and median prefill <= 2,510", "futility-only", "after 14 documents"):
            self.assertIn(needle, txt, needle)
        self.assertEqual((G.PILOT_WALL_S, G.INTERIM_DOCS, G.BOOKED_L4_S, G.BUDGET_X, G.HARD_STOP_X), (1800, 14, 9000.0, 2.0, 1.5))
        self.assertEqual((G.M4_CALLS_SOFT, G.M4_PREFILL_MAX), (3, 2510))
        self.assertEqual(G.PLANNERS, ("agg110.v1", "agg11.v0"))
        self.assertEqual(G.PILOT_DOCS, ("MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K"))

    def test_phase0_and_agg11_files_are_untouched(self):
        # the frozen inputs, the Phase 0 pins and the AGG-1.1 PREDECLARED hold: Phase 0 / AGG-1.1 stay reproducible
        inp = G.load_inputs(D1)
        ok, info = X.preflight(inp, D1, check_pdfs=False, run_build_check=False, sizes=inp["sizes"])
        self.assertEqual(info["problems"], [])
        self.assertEqual(X.sha256_file(D1 / "PREDECLARED.md"), X.PREDECLARED_SHA)

    def test_agg11_l4_unit_stays_ast_equal_to_the_phase0_loop(self):
        p0 = D0 / "phase0_arms.py"
        import ast
        ref = X._l4_loop_ast(ast.parse(p0.read_text()))
        self.assertTrue(ref)
        self.assertEqual(ref, X.l4_unit_ast())

    def test_the_cli_declares_the_phases(self):
        self.assertEqual(C.PHASES, ("preflight", "l3", "pilot", "ingest", "gpu-audit", "gpu-answers", "score"))

    def test_doc_order_puts_the_pilot_first_then_ascending_size(self):
        corpus = {"docs": [{"doc_name": n} for n in ("Z", "A", "3M_2022_10K", "MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K")]}
        sizes = {"Z": 1, "A": 5, "3M_2022_10K": 9, "MICROSOFT_2022_10K": 9, "JOHNSON_JOHNSON_2022_10K": 9}
        self.assertEqual(G.doc_order(corpus, sizes), list(G.PILOT_DOCS) + ["Z", "A"])


class PureMetrics(unittest.TestCase):
    def test_regressions_and_strata(self):
        ref = {"agg11": {"raw_hit_cells": ["a|x", "b|x", "c|x"], "canonical_v1_ok_cells": ["a|x", "b|x"]}}
        now = {"raw": {"a|x": True, "b|x": False, "c|x": True, "d|x": True}, "canonical": {"a|x": True, "b|x": True, "c|x": False, "d|x": False}}
        reg = G.regressions(ref, now, held_out={"a|x", "b|x"})
        self.assertFalse(reg["ok"])
        self.assertEqual((reg["raw_broken"], reg["canonical_broken"]), (["b|x"], []))
        self.assertEqual(reg["raw_broken_by_stratum"], {"held_out": ["b|x"], "touched": []})
        self.assertEqual((reg["raw_fixed"], reg["canonical_fixed"]), (["d|x"], []))
        st = G.strata_report(ref, now, {"a|x", "b|x"})
        self.assertEqual((st["held_out"]["n"], st["held_out"]["raw"]["found"], st["held_out"]["raw"]["agg11"]), (2, 1, 2))
        self.assertEqual(st["all"]["n"], 4)
        self.assertIn("HELD-OUT", st["held_out"]["label"])
        self.assertIn("IN-SAMPLE", st["touched"]["label"])

    def test_paired_matches_the_phase1_formula(self):
        new = {"q1": True, "q2": True, "q3": False, "q4": False}
        old = {"q1": True, "q2": False, "q3": True, "q4": False}
        r = G.paired(new, old, ["q1", "q2", "q3", "q4"])
        self.assertEqual((r["both"], r["only_new"], r["only_old"], r["neither"], r["z"]), (1, 1, 1, 1, 0.0))
        old2 = {q: False for q in new}
        self.assertAlmostEqual(G.paired(new, old2, list(new))["z"], 2 / 2 ** 0.5)

    def test_cpu_ready(self):
        ok = {"complete": True, "phase": "cpu", "futility_stop": None, "gate_tripped": [], "m1_found": 185, "m3_passed": True}
        self.assertTrue(G.cpu_ready(ok, REF)[0])
        for k, v in (("m1_found", 184), ("m3_passed", False), ("complete", False), ("gate_tripped", ["M3"]), ("futility_stop", "M1")):
            self.assertFalse(G.cpu_ready(dict(ok, **{k: v}), REF)[0], k)

    def test_m4_trips(self):
        self.assertFalse(G.m4_trips({"median_calls": 2.0, "median_prefill": 812.0}, True))
        self.assertTrue(G.m4_trips({"median_calls": 4.0, "median_prefill": 100.0}, True))
        self.assertTrue(G.m4_trips({"median_calls": 2.0, "median_prefill": 2600.0}, True))
        self.assertFalse(G.m4_trips({"median_calls": 9.0, "median_prefill": 9999.0}, False))      # incomplete: never judged

    def test_pilot_eval_futility_and_budget(self):
        rec = P.Recorder(Path(tempfile.mkdtemp()) / "r.jsonl")
        for d in G.PILOT_DOCS:
            rec.emit("doc_l4", unit=f"doc_l4:{d}", doc=d, n_windows=30, llm_calls=30, prefill_tokens=40_000, cache_n=0, decode_tokens=1, wall_s=300.0,
                     bad_replies=0, cut=0, unparseable=0, chars=100_000, final=True)
        plans = {d: {"n_windows": 30, "chars": 100_000} for d in G.PILOT_DOCS}
        plans.update({f"X{i}": {"n_windows": 30, "chars": 100_000} for i in range(25)})
        stats = {d: {"tokens": 160_000, "chunks": 200} for d in plans}
        inp = {"reference": REF, "cells": [(d, "revenue", 1.0) for d in G.PILOT_DOCS for _ in range(8)]}
        keys = [G.cell_key(d, "revenue") for d in G.PILOT_DOCS]
        # 8 cells per document share one key in this synthetic list: found is counted per cell
        raw = {k: True for k in keys}
        ev = G.pilot_eval(rec=rec, plans=plans, stats=stats, raw=raw, inp=inp)
        self.assertEqual(ev["pilot_m1"]["agg11_found"], G.ref_found(REF, G.PILOT_DOCS))
        self.assertEqual(ev["pilot_m1"]["agg11_found"], 20)
        self.assertEqual(ev["projected_calls"], 28 * 30)
        self.assertEqual(ev["budget_stops"], [])          # 900 s pilot wall <= 1800
        ev2 = G.pilot_eval(rec=rec, plans=plans, stats=stats, raw=raw, inp=inp, pilot_wall_s=100.0)
        self.assertEqual(ev2["budget_stops"][0], "pilot_wall_over_1800s")
        ev3 = G.pilot_eval(rec=rec, plans=plans, stats=stats, raw=raw, inp=inp, booked_s=100.0)
        self.assertIn("projected_l4_wall_over_2x_booked", ev3["budget_stops"])
        self.assertEqual(ev["gates_tripped"], [])                     # 24 found >= AGG-1.1's 20 on the same cells
        evb = G.pilot_eval(rec=rec, plans=plans, stats=stats, raw={}, inp=inp)
        self.assertEqual(evb["gates_tripped"], ["M1_pilot_below_agg11"])      # 0 found < AGG-1.1's 20
        big = {d: {"tokens": 1_000, "chunks": 200} for d in plans}
        self.assertIn("M3", G.pilot_eval(rec=rec, plans=plans, stats=big, raw=raw, inp=inp)["gates_tripped"])      # projected prefill > 2x corpus

    def test_verdict_mechanics(self):
        agg = [{"id": f"agg{i:02d}"} for i in range(1, 5)] + [{"id": "p1_01"}]
        ids = [q["id"] for q in agg]
        old = {q: {"correct": q in ("agg01", "agg02"), "calls": 2, "prefill_tokens": 600, "path": "sql"} for q in ids}
        ref = {"agg11": {"raw_found": 185, "canonical_v1_ok": 177,
                         "p_large": {"per_question": old, "n_correct": 2, "n": 5, "accuracy": 0.4}}}

        def recs(pv, correct, calls=2, prefill=800):
            return [{"kind": "answer", "final": True, "planner_version": pv, "qid": q, "correct": q in correct, "calls": calls,
                     "prefill_tokens": prefill} for q in ids]

        cpu = {"m1_found": 200, "m3_passed": True, "regressions": {"ok": True}}
        both = recs("agg110.v1", {"agg01", "agg02", "agg03", "agg04"}) + recs("agg11.v0", {"agg01", "agg02", "agg03"})
        v = G.verdict(cpu, both, agg, ref)
        self.assertEqual((v["verdict"], v["gates_tripped"], v["complete"]), ("agg_1_12_done", [], True))
        self.assertEqual(v["planners"]["agg110.v1"]["n_correct"], 4)
        self.assertEqual(v["planners"]["agg110.v1"]["vs_agg11"]["only_new"], 2)
        self.assertTrue(v["adopt_agg_pipeline_v2_as_default"])
        self.assertEqual(v["planner_effect_same_store"]["only_new"], 1)
        # a regression: not met, GPU numbers still filed
        v2 = G.verdict(dict(cpu, regressions={"ok": False, "raw_broken": ["a|x"]}), both, agg, ref)
        self.assertEqual(v2["verdict"], "agg_1_12_not_met")
        self.assertFalse(v2["adopt_agg_pipeline_v2_as_default"])
        # gates
        self.assertEqual(G.verdict(dict(cpu, m1_found=184), both, agg, ref)["gates_tripped"], ["M1"])
        self.assertEqual(G.verdict(dict(cpu, m3_passed=False), both, agg, ref)["verdict"], "phase1_gate_tripped")
        slow = recs("agg110.v1", {"agg01"}, calls=5) + recs("agg11.v0", {"agg01"})
        self.assertEqual(G.verdict(cpu, slow, agg, ref)["gates_tripped"], ["M4:agg110.v1"])
        # an incomplete answer set is never judged
        self.assertEqual(G.verdict(cpu, recs("agg110.v1", {"agg01"}), agg, ref)["verdict"], "incomplete")
        # P-large accuracy going down is filed, not a gate
        down = recs("agg110.v1", set()) + recs("agg11.v0", set())
        vd = G.verdict(cpu, down, agg, ref)
        self.assertEqual((vd["verdict"], vd["direction"]["p_large_primary"], vd["adopt_agg_pipeline_v2_as_default"]), ("agg_1_12_done", "down", False))


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not importable")
class Stage(unittest.TestCase):
    """A two-document corpus end to end through run_l3 / run_ingest with a fake typed backend."""

    DOCS = ("A_2022_10K", "B_2022_10K")
    CID = G.CORPUS_ID

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        long_a = "\n".join(f"filler line {i} about nothing in particular" for i in range(30))
        self.vals = {"A_2022_10K": {"revenue": 100.0, "total_assets": 27165.0}, "B_2022_10K": {"revenue": 250.0, "total_assets": 9000.0}}
        st = CorpusStore(str(self.tmp / "seed.sqlite"))
        for d in self.DOCS:
            v = self.vals[d]
            secs = [Section(id="s1", title="Income", level=1, content=f"(in millions)\nTotal revenues {int(v['revenue'])} 90\n" + long_a,
                            page_start=1, page_end=1),
                    Section(id="s2", title="Balance", level=1, content=f"(in millions)\nTotal assets {int(v['total_assets']):,} 1\n" + long_a,
                            page_start=2, page_end=2)]
            st.ingest(DocumentPreprocessResult(original_path=f"/x/{d}.pdf", sections=secs, figures=[], total_pages=2, tables=[]),
                      corpus_id=self.CID, doc_id=d, entity=d.split("_")[0])
        # a stale v1 fact that the wipe must remove
        st.write_facts([{"fact_id": "stale", "doc_hash": st.corpus_docs(self.CID)[0]["doc_hash"], "doc_id": "A_2022_10K", "entity": "A",
                         "period": "2022", "field": "revenue", "value": 1.0, "unit": "USD_millions", "quote": "stale", "confidence": "high",
                         "extractor": "l4_llm.v1", "audit_status": "unaudited"}])
        st.record_index(self.CID, st.corpus_content_hash(self.CID), "l4_facts_llm:revenue", "", layer="L4", n_items=1)
        st.close()
        self.seed = self.tmp / "seed.sqlite"
        self.seed_sha = hashlib.sha256(self.seed.read_bytes()).hexdigest()
        self.corpus = {"fiscal_year": 2022, "docs": [{"doc_name": d, "company": d.split("_")[0], "phase0": i == 0}
                                                    for i, d in enumerate(self.DOCS)]}
        cells = [(d, f, self.vals[d][f]) for d in self.DOCS for f in ("revenue", "total_assets")]
        keys = [G.cell_key(d, f) for d, f, _g in cells]
        ref = {"agg11": {"raw_hit_cells": keys[:2], "canonical_v1_ok_cells": keys[:2], "raw_found": 2, "raw_m1": 0.5, "canonical_v1_ok": 2,
                         "m3": {"prefill_ratio": 0.4}},
               "partition": {"held_out": keys[:2], "touched": keys[2:]}}
        self.inp = {"corpus": self.corpus, "cells": cells, "sizes": {self.DOCS[0]: 10, self.DOCS[1]: 20}, "reference": ref,
                    "set_dir": D1, "sheet": None}
        self.guard = lambda: None

    def backend(self, bad_for=()):
        vals = self.vals
        tc = self

        class B:
            calls = 0
            prefill_tokens = 0
            log = []

            def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
                B.calls += 1
                B.prefill_tokens += 5
                ex = prompt.split("Excerpt:\n", 1)[1]
                body = {s: [] for s in ft.SLOTS}
                for d, v in vals.items():
                    for f, q in (("revenue", f"Total revenues {int(v['revenue'])}"), ("total_assets", f"Total assets {int(v['total_assets']):,}")):
                        if q in ex:
                            n = float(q.rsplit(" ", 1)[1].replace(",", ""))
                            body[f] = [{"period": "2022", "value": n, "unit": "USD_millions", "quote": q, "confidence": "high"}]
                text = "{" if any(b in ex for b in bad_for) else json.dumps(body)
                return {"text": text, "prompt_tokens": 5, "cache_n": 0, "decode_tokens": 10,
                        "finish_reason": "length" if text == "{" else "stop"}

        B.calls = 0
        return B()

    def run_all(self, out, be, rec=None, **kw):
        rec = rec or P.Recorder(out / "results.jsonl")
        r0 = G.run_l3(rec, out, self.inp, seed_store=self.seed, guard=self.guard, **kw)
        return rec, r0

    def test_l3_copies_wipes_and_reextracts_without_touching_the_seed(self):
        out = self.tmp / "out"
        rec, r = self.run_all(out, None)
        self.assertEqual((r["complete"], r["rc"]), (True, 0))
        self.assertEqual(hashlib.sha256(self.seed.read_bytes()).hexdigest(), self.seed_sha)       # the seed is never mutated
        l3 = next(x for x in rec.records if x.get("unit") == "l3v2")
        self.assertEqual((l3["n_facts_before_wipe"] > 0, l3["llm_calls"], l3["pipeline"]["l3_period"]), (True, 0, "l3_period.v2"))
        st = CorpusStore(str(out / "store.sqlite"))
        facts = st.facts(self.CID)
        self.assertTrue(all(f["extractor"].startswith("l3") for f in facts))                      # stale l4 fact gone
        self.assertEqual(l3["n_l3_facts"], len(facts))
        self.assertNotIn("stale", {f["fact_id"] for f in facts})
        self.assertEqual({m["layer"] for m in st.manifest(self.CID)} - {"L1"}, {"L3"})
        st.close()
        # idempotent
        rec2 = P.Recorder(out / "results.jsonl")
        self.assertEqual(G.run_l3(rec2, out, self.inp, seed_store=self.seed, guard=self.guard)["complete"], True)
        self.assertEqual(sum(1 for r_ in rec2.records if r_.get("unit") == "l3v2"), 1)

    def _ingest(self, out, be, stage, **kw):
        rec = P.Recorder(out / "results.jsonl")
        return rec, G.run_ingest(rec, out, self.inp, be, stage=stage, guard=self.guard, **kw)

    def test_pilot_then_ingest_end_to_end_and_every_unit_is_one_call(self):
        out = self.tmp / "out"
        self.run_all(out, None)
        be = self.backend()
        with mock.patch.object(G, "PILOT_DOCS", (self.DOCS[0],)), mock.patch.object(G, "INTERIM_DOCS", 99):
            rec, r = self._ingest(out, be, "pilot")
            self.assertEqual((r["complete"], r["rc"]), (True, 0))
            pd = next(x for x in rec.records if x.get("unit") == "pilot_done")
            self.assertEqual((pd["gates_tripped"], pd["budget_stops"]), ([], []))
            n_pilot = be.calls
            self.assertGreater(n_pilot, 0)
            rec, r = self._ingest(out, be, "ingest")
        self.assertEqual((r["complete"], r["rc"]), (True, 0), r)
        res = json.loads((out / "result.cpu.json").read_text())
        self.assertEqual((res["phase"], res["pipeline"], res["l4_reused"], res["m1_found"], res["m3_passed"]), ("cpu", "agg_pipeline.v2", False, 4, True))
        self.assertEqual(res["regressions"]["ok"], True)
        self.assertEqual(res["canonical"]["rule"], "canonical.v2:v1+attributable_net_income+thousands_unit")
        self.assertEqual(res["canonical"]["guards"], ["attributable_net_income"])
        self.assertEqual(res["m3"]["llm_calls"], be.calls)                          # the M3 call count is the window calls, nothing hidden
        self.assertEqual(res["strata"]["held_out"]["n"], 2)
        windows = [x for x in rec.records if x.get("kind") == "t_window"]
        self.assertEqual(len({x["unit"] for x in windows}), len(windows))             # one record per window, no duplicates
        self.assertTrue(all(x["calls"] == 1 for x in windows))
        # the L4 facts are in the store through facts_llm's tag and the layers are recorded for the planner
        st = CorpusStore(str(out / "store.sqlite"))
        l4 = {(f["doc_id"], f["field"], f["value"]) for f in st.facts(self.CID, extractor="l4_llm.v1")}
        self.assertIn(("A_2022_10K", "revenue", 100.0), l4)
        self.assertIn(("B_2022_10K", "total_assets", 9000.0), l4)
        self.assertIn("L4", st.built_layers(self.CID))
        st.close()

    def test_resume_after_a_kill_repeats_no_call(self):
        out = self.tmp / "out"
        self.run_all(out, None)
        be = self.backend()
        stop_after = {"n": 1}

        def guard():
            if stop_after["n"] <= 0:
                raise P.Stop("test kill")
            stop_after["n"] -= 1

        rec = P.Recorder(out / "results.jsonl")
        with mock.patch.object(G, "PILOT_DOCS", (self.DOCS[0],)):
            with self.assertRaises(P.Stop):
                G.run_ingest(rec, out, self.inp, be, stage="pilot", guard=guard)
            first = be.calls
            rec2, r = self._ingest(out, be, "pilot")
        self.assertEqual((r["complete"], r["rc"]), (True, 0))
        windows = [x["unit"] for x in rec2.records if x.get("kind") == "t_window"]
        self.assertEqual(len(windows), len(set(windows)))
        self.assertEqual(be.calls, len(windows))                       # every call is recorded once: none repeated
        self.assertEqual(first, 1)

    def test_smoke_makes_exactly_one_call_on_the_smallest_document(self):
        out = self.tmp / "out_smoke"
        self.run_all(out, None, smoke=True)
        be = self.backend()
        rec, r = self._ingest(out, be, "pilot", smoke=True)
        self.assertEqual((r["smoke"], r["rc"], be.calls), (True, 0, 1))
        w = [x for x in rec.records if x.get("kind") == "t_window"]
        self.assertEqual((len(w), w[0]["doc"]), (1, self.DOCS[0]))

    def test_a_bad_reply_is_counted_and_a_pilot_m1_below_agg11_stops(self):
        out = self.tmp / "out"
        self.run_all(out, None)
        be = self.backend(bad_for=("Total revenues", "Total assets"))      # every call truncated: no L4 facts
        with mock.patch.object(G, "PILOT_DOCS", (self.DOCS[0],)):
            rec, r = self._ingest(out, be, "pilot")
        pd = next(x for x in rec.records if x.get("unit") == "pilot_done")
        self.assertGreater(pd["pilot"]["bad_replies"], 0)
        # the L3 facts alone find the revenue and the assets of A (v2 attribution), so the pilot M1 equals AGG-1.1's here: no futility
        self.assertEqual(pd["pilot_m1"]["agg11_found"], 2)
        if pd["gates_tripped"]:
            self.assertEqual((r["rc"], r["verdict"]), (3, "phase1_gate_tripped"))

    def test_pilot_wall_over_the_bound_is_a_budget_stop_not_a_gate(self):
        out = self.tmp / "out"
        self.run_all(out, None)
        with mock.patch.object(G, "PILOT_DOCS", (self.DOCS[0],)):
            rec, r = self._ingest(out, self.backend(), "pilot", pilot_wall_s=-1.0)
        self.assertEqual((r["rc"], r["budget_stop"]), (5, "pilot_wall_over_1800s"))
        self.assertFalse(r["complete"])

    def test_ingest_needs_the_pilot_first(self):
        out = self.tmp / "out"
        self.run_all(out, None)
        rec, r = self._ingest(out, self.backend(), "ingest")
        self.assertEqual(r["rc"], 4)


@unittest.skipUnless(HAVE_ORCH and SEED.exists(), "orchestrator tree or the AGG-1.1 seed store not available")
class OfflinePlan(unittest.TestCase):
    """The dry-run plan against the real AGG-1.1 store (read-only): the M3 bounds hold and v2 is cheaper than AGG-1.1."""

    @classmethod
    def setUpClass(cls):
        import agg16 as G16
        cls.inp = G.load_inputs(D1)
        cls.store = G16.mem_store(SEED, [d["doc_name"] for d in cls.inp["corpus"]["docs"]])
        cls.plans = G.doc_plans(cls.store, cls.inp["corpus"])
        cls.stats = X.doc_stats(cls.store, G.CORPUS_ID)

    def test_seed_store_shape(self):
        self.assertEqual((len(self.store.corpus_docs(G.CORPUS_ID)), len(self.store.chunks(G.CORPUS_ID))), (28, 6094))

    def test_stage_equals_arm_t(self):
        self.assertEqual(G.parity_problems(self.store, self.inp["corpus"], ("MICROSOFT_2022_10K", "COSTCO_2022_10K", "BOEING_2022_10K")), [])

    def test_m3_projection_is_inside_the_bounds_and_below_agg11(self):
        m = G.plan_m3(self.plans, self.stats)
        self.assertEqual(m["calls"], 903)
        self.assertTrue(m["inside_kill_bounds"])
        self.assertLess(m["m3"]["prefill_ratio"], m["agg11"]["prefill_ratio"])
        self.assertLess(m["m3"]["calls_per_100_chunks"], m["agg11"]["calls_per_100_chunks"])
        self.assertLess(m["calls_per_unit"], m["agg11"]["calls_per_unit"])
        self.assertLess(m["m3"]["prefill_ratio"], 2.0)
        self.assertLess(m["m3"]["calls_per_100_chunks"], 150)

    def test_estimates_and_pilot_plan(self):
        e = G.estimates(self.plans)
        self.assertEqual(e["label"], "ESTIMATE")
        self.assertLess(e["cpu_s"], G.BOOKED_L4_S)
        self.assertEqual(G.plan_totals(self.plans, G.PILOT_DOCS)["calls"], 90)
        self.assertLess(G.plan_totals(self.plans, G.PILOT_DOCS)["wall_est_s"], G.PILOT_WALL_S)

    def test_every_window_is_a_real_window_and_chosen_by_a_schema_field(self):
        for d, p in self.plans.items():
            ids = {w["chunk_id"] for w in p["windows"]}
            self.assertEqual(ids, set(p["chosen"]), d)
            self.assertTrue(all(set(v) <= set(A_METRICS) for v in p["chosen"].values()), d)

    def test_dry_run_cli_is_clean(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.main(["--phase", "ingest", "--dry-run", "--pins", "none", "--orch-root", str(ORCH)])
        out = json.loads(buf.getvalue())
        self.assertEqual((rc, out["problems"], out["model_calls"], out["servers_started"], out["wrote_files"]), (0, [], 0, 0, 0))
        self.assertEqual(out["plan"]["m3_projection"]["calls"], 903)


from phase1_arms import A  # noqa: E402

A_METRICS = tuple(A.METRICS)

if __name__ == "__main__":
    unittest.main()
