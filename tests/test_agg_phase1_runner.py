"""AGG-1.1 step B runner tests (fake backends and a fake document parser: no model, no network, no process, no PDF).

Stdlib tests always run. The store / L3 / L4 / answer tests import the orchestrator package and skip without it:
run them with the orchestrator venv python and ``AGG_ORCH_ROOT=<tree at the Phase 1 pin>``, e.g. the pinned copy
inside a Phase 0 job dir (``<job>/orch``). ``ORCHESTRATOR_FEATURE_AGG_LAYER=1`` is set here.
"""

from __future__ import annotations

import ast
import contextlib
import csv
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
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_phase1 as R  # noqa: E402

DEFAULT_ORCH = "/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2/orch"
ORCH = Path(os.environ.get("AGG_ORCH_ROOT", DEFAULT_ORCH))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation.llm_backend import GATE_GBNF, OutOfWindow  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = hasattr(CorpusStore, "refresh_canonical")
except Exception:  # pragma: no cover
    HAVE_ORCH = False

P0_JOB = Path("/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2")
LENIENT_M3 = {"m3_prefill_ratio_max": 1e9, "m3_calls_per_100_chunks_max": 1e9}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- pins
class FrozenPinTests(unittest.TestCase):
    def test_phase1_frozen_inputs_match_pins_and_predeclared(self):
        text = (D1 / "PREDECLARED.md").read_text()
        for name, want in X.FROZEN_P1.items():
            self.assertEqual(sha(D1 / name), want, name)
            self.assertIn(want, text, name)  # the pin in code is the pin in the predeclared table
        self.assertEqual(sha(D1 / "PREDECLARED.md"), X.PREDECLARED_SHA)
        self.assertTrue(X.PREDECLARED_SHA.startswith("2d4a5321"))
        self.assertEqual(json.loads((D1 / "corpus.json").read_text())["manifest_digest_sha256"], X.MANIFEST_DIGEST)

    def test_phase0_code_and_predeclared_untouched(self):
        for name, want in X.PHASE0_PINS.items():
            self.assertEqual(sha(D0 / name), want, name)  # scorer, KILL, L4 loop, L3 loader: unchanged
        self.assertEqual(X.PHASE0_PREDECLARED_SHA, "976a5fe87e3db33d6a5e4547ee4bb6cae324f44d107980342b62f7b6212b8bc9")

    def test_gate_constants_are_the_predeclared_ones(self):
        self.assertEqual(A0.KILL["m3_prefill_ratio_max"], 2.0)
        self.assertEqual(A0.KILL["m3_calls_per_100_chunks_max"], 150)
        self.assertEqual(A0.KILL["median_calls_max"], 8)
        self.assertEqual((X.M1_GATE, X.M1_PILOT_MIN, X.M4_MEDIAN_CALLS_SOFT, X.M4_MEDIAN_PREFILL_MAX),
                         (0.75, 0.60, 3, 2510))
        self.assertEqual((X.INTERIM_NEW_DOCS, X.PILOT_WALL_S, X.B1_LOOK_AT, X.B1_STOP_Z, X.CALLS_PER_UNIT_RESTIMATE),
                         (10, 1800, 30, 3.0, 12))
        text = (D1 / "PREDECLARED.md").read_text()
        for needle in ("M1 < 0.75", "pilot M1 < 0.60", "after 10 of the 18 new", "after B1 has answered 30",
                       ">= 3 SE", "stop and re-estimate above 12"):
            self.assertIn(needle, text, needle)

    def test_default_set_preflights_clean(self):
        inp = X.load_inputs(corpus=D1 / "corpus.json", golds=D1 / "golds.jsonl",
                            verification=D1 / "verification_record.json", facts_sheet=D1 / "facts_sheet.csv")
        sizes = X.load_sizes(D1 / "extracted_sizes.json")
        ok, info = X.preflight(inp, D1, check_pdfs=False, run_build_check=True, sizes=sizes)
        self.assertTrue(ok, info["problems"])
        self.assertEqual((info["n_agg_verified"], info["n_agg_excluded"]), (58, 0))
        self.assertEqual(info["m1_cells"], {"total": 211, "phase0_docs": 75, "new_docs": 136})
        self.assertEqual(info["build_check"]["rc"], 0)

    def test_tampered_golds_fail_preflight(self):
        tmp = Path(tempfile.mkdtemp())
        for f in list(X.FROZEN_P1) + ["PREDECLARED.md", "extracted_sizes.json", "build_phase1.py", "phase1_questions.py"]:
            shutil.copy(D1 / f, tmp / f)
        lines = (tmp / "golds.jsonl").read_text().splitlines()
        g = json.loads(lines[0])
        g["gold"] = "Apple"
        lines[0] = json.dumps(g)
        (tmp / "golds.jsonl").write_text("\n".join(lines) + "\n")
        inp = X.load_inputs(corpus=tmp / "corpus.json", golds=tmp / "golds.jsonl",
                            verification=tmp / "verification_record.json", facts_sheet=tmp / "facts_sheet.csv")
        ok, info = X.preflight(inp, tmp, check_pdfs=False, run_build_check=False)
        self.assertFalse(ok)
        self.assertTrue(any("golds.jsonl" in p for p in info["problems"]), info["problems"])


# --------------------------------------------------------------------------------------------- plan
class PlanTests(unittest.TestCase):
    def setUp(self):
        self.corpus = json.loads((D1 / "corpus.json").read_text())
        self.sizes = X.load_sizes(D1 / "extracted_sizes.json")

    def test_sizes_total_and_pilot_ranks(self):
        self.assertEqual(sum(self.sizes.values()), 17_898_545)  # PREDECLARED corpus size
        self.assertEqual(len(self.sizes), 28)
        ranks = X.size_ranks(self.corpus, self.sizes)
        self.assertEqual(len(ranks), 18)
        self.assertEqual(tuple(ranks[d] for d in X.PILOT_DOCS), X.PILOT_RANKS)

    def test_order_and_units(self):
        order = X.new_doc_order(self.corpus, self.sizes)
        self.assertEqual(order[:3], list(X.PILOT_DOCS))
        self.assertEqual(len(order), 18)
        rest = order[3:]
        self.assertEqual(rest, sorted(rest, key=lambda d: (self.sizes[d], d)))
        reuse = X.plan_units(self.corpus, self.sizes, True)
        self.assertEqual(len(reuse), 224)
        self.assertEqual({s: sum(p["stage"] == s for p in reuse) for s in ("reused", "pilot", "ingest")},
                         {"reused": 80, "pilot": 24, "ingest": 120})
        self.assertEqual(len({p["unit"] for p in reuse}), 224)
        fresh = X.plan_units(self.corpus, self.sizes, False)
        self.assertEqual({s: sum(p["stage"] == s for p in fresh) for s in ("reused", "pilot", "ingest")},
                         {"reused": 0, "pilot": 24, "ingest": 200})
        self.assertEqual([p["stage"] for p in fresh[:24]], ["pilot"] * 24)  # the pilot decides first

    def test_m1_cells_match_phase0_function_on_phase0_set(self):
        golds = [json.loads(l) for l in (D0 / "golds.jsonl").read_text().splitlines() if l.strip()]
        ver = json.loads((D0 / "verification_record.json").read_text())["questions"]
        run = [g for g in golds if ver.get(g["id"], {}).get("verified")]
        self.assertEqual(X.m1_cells(D0 / "facts_sheet.csv", run, A0.DOCS), P.m1_cells(D0, run))
        self.assertEqual(len(P.m1_cells(D0, run)), 75)


# --------------------------------------------------------------------------------------------- arithmetic
class GateArithmeticTests(unittest.TestCase):
    def test_m1_hits_no_unit_leniency(self):
        scale = {"USD_thousands": 1e-3, "USD_millions": 1.0}
        cells = [("A", "revenue", 1000.0), ("B", "revenue", 500.0), ("C", "revenue", 200.0)]
        facts = [{"doc_id": "A", "field": "revenue", "period": "FY2022", "value": 1000.0, "unit": "USD_millions"},
                 {"doc_id": "B", "field": "revenue", "period": "FY2022", "value": 500.0, "unit": "USD_millions"},
                 # a filing in thousands stored 1000x off (unit labelled millions): a miss
                 {"doc_id": "B", "field": "revenue", "period": "FY2021", "value": 500.0, "unit": "USD_millions"},
                 {"doc_id": "C", "field": "revenue", "period": "FY2022", "value": 200000.0, "unit": "USD_millions"}]
        self.assertEqual(X.m1_hits(facts, cells, unit_scale=scale), [True, True, False])
        facts[3]["unit"] = "USD_thousands"  # correctly labelled: found after scaling
        self.assertEqual(X.m1_hits(facts, cells, unit_scale=scale), [True, True, True])
        s = X.m1_summary(cells, [True, True, False], {"A"})
        self.assertEqual((s["phase0_docs"]["gold"], s["new_docs"]["gold"], s["new_docs"]["found"]), (1, 2, 1))

    @unittest.skipUnless(HAVE_ORCH, "orchestrator tree not importable")
    def test_m1_hits_equals_phase0_m1_found(self):
        from src.aggregation.facts_llm import UNIT_SCALE  # noqa: PLC0415
        cells = [("A", "revenue", 100.0), ("B", "capex", 7.0), ("C", "revenue", 9.0)]
        facts = [{"doc_id": "A", "field": "revenue", "period": "FY2022", "value": 100.3, "unit": "USD_millions"},
                 {"doc_id": "B", "field": "capex", "period": "2022", "value": -7000.0, "unit": "USD_thousands"},
                 {"doc_id": "C", "field": "revenue", "period": "FY2021", "value": 9.0, "unit": "USD_millions"}]
        self.assertEqual(sum(X.m1_hits(facts, cells, unit_scale=UNIT_SCALE)), P.m1_found(facts, cells))
        self.assertEqual(X.m1_hits(facts, cells), [True, True, False])

    def test_m3_bounds(self):
        ok = X.m3_values(1_000_000, 500, 1_000_000, 2_000)
        self.assertFalse(X.m3_trips(ok))
        self.assertTrue(X.m3_trips(X.m3_values(2_000_001, 500, 1_000_000, 2_000)))      # > 2.0
        self.assertTrue(X.m3_trips(X.m3_values(1_000, 3_001, 1_000_000, 2_000)))        # > 150 / 100 chunks
        self.assertFalse(X.m3_trips(X.m3_values(1_000, 3_000, 1_000_000, 2_000)))       # == 150: not over
        self.assertTrue(X.m3_trips(X.m3_values(1, 1, 0, 0)))                            # fail closed

    def _pilot(self, per_calls=6, per_prefill=7_000, per_wall=60.0, hits=None, booked=7500.0):
        corpus = json.loads((D1 / "corpus.json").read_text())
        sizes = X.load_sizes(D1 / "extracted_sizes.json")
        plan = X.plan_units(corpus, sizes, True)
        ok = {}
        for p in plan:
            if p["stage"] in ("reused", "pilot"):
                reused = p["stage"] == "reused"
                ok[p["unit"]] = {"unit": p["unit"], "llm_calls": 6 if reused else per_calls,
                                 "prefill_tokens": 7_667 if reused else per_prefill, "wall_s": 52.0 if reused else per_wall,
                                 "ok": True, "kind": "l4_unit", "reused": reused}
        stats = {d["doc_name"]: {"tokens": 70_000, "chunks": 90} for d in corpus["docs"]}
        hits = [True] * 24 if hits is None else hits
        return X.pilot_eval(ok=ok, plan=plan, stats=stats, pilot_hits=hits,
                            reused_tally=X.tally(r for r in ok.values() if r["reused"]),
                            total_tokens=28 * 70_000, total_chunks=28 * 90, booked_l4_s=booked, reuse_ok=True)

    def test_pilot_projection_and_gates(self):
        ev = self._pilot()
        self.assertEqual(ev["units_to_run"], 144)
        self.assertEqual(ev["gates_tripped"], [])
        self.assertAlmostEqual(ev["projected_l4_wall_s"], 60.0 * 144)
        self.assertAlmostEqual(ev["projected_m3"]["llm_calls"], 80 * 6 + 144 * 6)
        self.assertAlmostEqual(ev["rebook_cap_s"], 1.25 * ev["projected_l4_wall_s"])
        self.assertAlmostEqual(ev["hard_stop_s"], 1.5 * ev["projected_l4_wall_s"])
        self.assertEqual(self._pilot(per_prefill=70_000)["gates_tripped"], ["M3"])        # ratio > 2.0
        self.assertEqual(self._pilot(hits=[True] * 14 + [False] * 10)["gates_tripped"], ["M1"])  # 0.583 < 0.60
        self.assertEqual(self._pilot(hits=[True] * 15 + [False] * 9)["gates_tripped"], [])       # 0.625 >= 0.60
        self.assertEqual(self._pilot(per_wall=105.0)["budget_stops"], ["projected_ingest_wall_over_2x_booked"])
        self.assertEqual(self._pilot(per_wall=104.0)["budget_stops"], [])
        self.assertEqual(self._pilot(per_calls=13)["budget_stops"], ["calls_per_unit_over_12"])

    def _ans(self, arm, qid, correct, calls=2, prefill=500, final=True):
        return {"kind": "answer", "arm": arm, "qid": qid, "correct": correct, "calls": calls,
                "prefill_tokens": prefill, "final": final, "status": "ok", "wall_s": 7.0}

    def test_b1_look_stop_on_answer(self):
        ids = [f"q{i}" for i in range(40)]

        def recs(only_p, only_b, n=30):
            out = []
            for i, q in enumerate(ids):
                p_ok, b_ok = i < only_p, only_p <= i < only_p + only_b
                out += [self._ans("P-large", q, True if p_ok or (i >= only_p + only_b and i < 25) else False),
                        self._ans("B1", q, True if b_ok or (only_p + only_b <= i < 25) else None) if i < n else {}]
            return [r for r in out if r]
        look = X.b1_look(recs(9, 0), ids)
        self.assertEqual((look["only_p_large"], look["only_b1"], look["n"]), (9, 0, 30))
        self.assertAlmostEqual(look["z"], 3.0)
        self.assertTrue(look["stop"])                      # exactly 3 SE stops (>= 3)
        look8 = X.b1_look(recs(8, 0), ids)
        self.assertAlmostEqual(look8["z"], 8 / 8 ** 0.5)
        self.assertFalse(look8["stop"])
        self.assertIsNone(X.b1_look(recs(9, 0, n=29), ids))  # not yet 30 B1 answers
        self.assertFalse(X.b1_look(recs(9, 1), ids)["stop"])  # (9-1)/sqrt(10) = 2.53
        st = X.paired_stats({"a": True, "b": True}, {"a": True}, ["a", "b", "c"])
        self.assertEqual((st["both_right"], st["only_p_large"], st["neither"]), (1, 1, 1))

    def test_verdict_gates(self):
        agg = [{"id": f"agg{i:02d}", "continuity": True} for i in range(1, 21)] + \
              [{"id": f"p1_{i:02d}", "continuity": False} for i in range(1, 39)]
        cpu = [{"kind": "ingest", "prefill_tokens": 700_000, "llm_calls": 800, "corpus_tokens": 4_400_000,
                "n_chunks": 6_000}, {"kind": "m1", "m1": 0.9, "gold_facts": 211, "found": 190}]

        def own(calls=2, prefill=510, b1=True):
            out = [self._ans("P-large", q["id"], True, calls, prefill) for q in agg]
            if b1:
                out += [self._ans("B1", q["id"], q["id"].endswith("1"), 30, 80_000) for q in agg]
            return out
        v = X.verdict(cpu, own(), agg)
        self.assertEqual((v["verdict"], v["gates_tripped"], v["complete"]), ("agg_1_1_done", [], True))
        self.assertEqual(X.verdict(cpu, own(calls=4), agg)["gates_tripped"], ["M4"])        # median calls > 3
        self.assertEqual(X.verdict(cpu, own(prefill=2511), agg)["gates_tripped"], ["M4"])
        self.assertEqual(X.verdict(cpu, own(calls=9), agg)["gates_tripped"], ["M4"])        # > 8 kill
        self.assertEqual(X.verdict(cpu, own(prefill=2510, calls=3), agg)["gates_tripped"], [])
        bad = [dict(cpu[0], prefill_tokens=9_000_000), cpu[1]]
        self.assertEqual(X.verdict(bad, own(), agg)["gates_tripped"], ["M3"])
        low = [cpu[0], {"kind": "m1", "m1": 0.74, "gold_facts": 211, "found": 156}]
        v = X.verdict(low, own(), agg)
        self.assertEqual((v["gates_tripped"], v["verdict"]), (["M1"], "phase1_gate_tripped"))
        inc = X.verdict(cpu, own(b1=False), agg)  # B1 neither complete nor early-stopped
        self.assertEqual((inc["complete"], inc["verdict"]), (False, "incomplete"))
        stopped = own(b1=False) + [self._ans("B1", q["id"], None, 30, 1) for q in agg[:30]] + \
            [{"kind": "b1_early_stop"}]
        self.assertTrue(X.verdict(cpu, stopped, agg)["complete"])

    def test_cpu_ready(self):
        good = {"complete": True, "futility_stop": None, "gate_tripped": [], "m1": 0.75, "m3_passed": True}
        self.assertTrue(X.cpu_ready(good)[0])
        for k, v in (("m1", 0.7499), ("m3_passed", False), ("complete", False), ("futility_stop", "M1"),
                     ("gate_tripped", ["M3"])):
            self.assertFalse(X.cpu_ready(dict(good, **{k: v}))[0], k)
        self.assertFalse(X.cpu_ready({})[0])


# --------------------------------------------------------------------------------------------- L4 reuse
class ReuseCheckTests(unittest.TestCase):
    def _job(self, tmp: Path, *, facts="A", backend="B", arms: str | None = None) -> tuple[Path, Path]:
        job, orch = tmp / "job", tmp / "orch"
        (orch / "src/aggregation").mkdir(parents=True)
        (orch / "src/aggregation/facts_llm.py").write_text(facts)
        (orch / "src/aggregation/llm_backend.py").write_text(backend)
        res = job / "res/benchmarks/agg_phase0"
        res.mkdir(parents=True)
        shutil.copy(D0 / "phase0_arms.py", res / "phase0_arms.py")
        if arms is not None:
            (res / "phase0_arms.py").write_text(arms)
        lines = [f"{sha(orch / 'src/aggregation/facts_llm.py')}  orch/src/aggregation/facts_llm.py",
                 f"{sha(orch / 'src/aggregation/llm_backend.py')}  orch/src/aggregation/llm_backend.py",
                 f"{sha(res / 'phase0_arms.py')}  res/benchmarks/agg_phase0/phase0_arms.py"]
        (job / "SOURCE.sha256").write_text("\n".join(lines) + "\n")
        return job, orch

    def test_empty_diff_reuses(self):
        tmp = Path(tempfile.mkdtemp())
        job, orch = self._job(tmp)
        r = X.l4_reuse_check(job, orch)
        self.assertTrue(r["reuse_ok"], r)
        self.assertTrue(r["l4_loop_ast_equal"])

    def test_any_l4_stage_diff_disables_reuse(self):
        tmp = Path(tempfile.mkdtemp())
        job, orch = self._job(tmp)
        (orch / "src/aggregation/facts_llm.py").write_text("A changed")
        r = X.l4_reuse_check(job, orch)
        self.assertFalse(r["reuse_ok"])
        self.assertFalse(r["files"]["src/aggregation/facts_llm.py"]["equal"])
        tmp2 = Path(tempfile.mkdtemp())
        job2, orch2 = self._job(tmp2)
        (orch2 / "src/aggregation/llm_backend.py").write_text("B changed")
        self.assertFalse(X.l4_reuse_check(job2, orch2)["reuse_ok"])
        # the Phase 0 research loop differs from the file Phase 1 imports
        tmp3 = Path(tempfile.mkdtemp())
        src = (D0 / "phase0_arms.py").read_text().replace("cfg.k_per_doc", "(cfg.k_per_doc + 1)", 1)
        job3, orch3 = self._job(tmp3, arms=src)
        r3 = X.l4_reuse_check(job3, orch3)
        self.assertFalse(r3["reuse_ok"])
        self.assertFalse(r3["l4_loop_ast_equal"])
        self.assertFalse(r3["files"]["research:phase0_arms.py"]["equal"])

    def test_missing_phase0_pin_fails_closed(self):
        tmp = Path(tempfile.mkdtemp())
        self.assertFalse(X.l4_reuse_check(tmp / "nojob", tmp / "orch")["reuse_ok"])

    def test_l4_loop_ast_is_run_cpus_loop(self):
        ref = X._l4_loop_ast(ast.parse((D0 / "phase0_arms.py").read_text()))
        self.assertGreater(len(ref), 5)
        self.assertEqual(ref, X.l4_unit_ast())
        changed = (D0 / "phase0_arms.py").read_text().replace("hi=cfg.hi, lo=cfg.lo)", "hi=cfg.hi, lo=cfg.lo + 0.1)", 1)
        self.assertNotEqual(X._l4_loop_ast(ast.parse(changed)), X.l4_unit_ast())

    @unittest.skipUnless((P0_JOB / "SOURCE.sha256").exists() and HAVE_ORCH, "Phase 0 job dir not on this host")
    def test_real_phase0_job_pin_allows_reuse(self):
        r = X.l4_reuse_check(P0_JOB, ORCH)
        self.assertTrue(r["reuse_ok"], r)


# --------------------------------------------------------------------------------------------- stores
CO = {"CORNING_2022_10K": ("Corning", 14189.0), "PAYPAL_2022_10K": ("Paypal", 27518.0),
      "MICROSOFT_2022_10K": ("Microsoft", 198270.0), "JOHNSON_JOHNSON_2022_10K": ("Johnson & Johnson", 94943.0),
      "3M_2022_10K": ("3M", 34229.0), "COSTCO_2022_10K": ("Costco", 226954.0),
      "AMAZON_2022_10K": ("Amazon", 513983.0)}
P0_NAMES = ["CORNING_2022_10K", "PAYPAL_2022_10K"]
NEW_NAMES = ["MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K", "COSTCO_2022_10K", "AMAZON_2022_10K"]
SIZES = {"MICROSOFT_2022_10K": 50, "JOHNSON_JOHNSON_2022_10K": 51, "3M_2022_10K": 52, "AMAZON_2022_10K": 40,
         "COSTCO_2022_10K": 30}  # new docs ascending: COSTCO, AMAZON, MICROSOFT, J&J, 3M


class FakeBackend:
    """Revenue rows only for text stating ``Total revenues N``; SQL, SEARCH loop and plain answers."""

    def __init__(self, final="FINAL: Amazon", search_final=None):
        self.calls = self.prefill_tokens = 0
        self.final, self.search_final = final, search_final or final

    def _bill(self, prompt):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        if grammar == GATE_GBNF:
            self._bill(prompt)
            return {"text": "yes", "prompt_tokens": len(prompt) // 4, "cache_n": 0}
        return self.chat(prompt, n_predict=n_predict, grammar=grammar)

    def chat(self, prompt, *, n_predict, grammar=None, cache_prompt=True, id_slot=None, logprobs=None):
        self._bill(prompt)
        if grammar and grammar.startswith("root ::= \"["):
            m = re.search(r"Total revenues [\d,]+", prompt.split("Excerpt:\n", 1)[-1])
            rows = [] if not m or "Field: revenue" not in prompt else [{
                "period": "FY2022", "value": float(m.group(0).split()[-1].replace(",", "")),
                "unit": "USD_millions", "quote": m.group(0), "confidence": "medium"}]
            text = json.dumps(rows)
        elif grammar and grammar.startswith('root ::= "SELECT '):
            text = "SELECT entity, value FROM facts_canonical WHERE field = 'revenue' ORDER BY value DESC"
        elif grammar and "SEARCH" in grammar:
            text = "SEARCH: revenues" if "> SEARCH" not in prompt else self.search_final
        else:
            text = f"Revenue rows considered.\n{self.final}"
        return {"text": text, "prompt_tokens": len(prompt) // 4, "cache_n": 0, "finish_reason": "stop"}


def make_fixture(tmp: Path, *, wrong: tuple[str, ...] = ()) -> dict:
    """Corpus (2 Phase 0 + 5 new docs), sheet, golds, a base store holding the Phase 0 docs, dummy PDFs."""
    pdfs = tmp / "pdfs"
    pdfs.mkdir()
    docs = []
    for name, (co, rev) in CO.items():
        pdf = pdfs / f"{name}.pdf"
        pdf.write_bytes(f"%PDF fake {name}".encode())
        docs.append({"doc_name": name, "company": co, "phase0": name in P0_NAMES, "pdf": str(pdf),
                     "sha256": sha(pdf), "bytes": pdf.stat().st_size, "pages": 100})
    corpus = {"docs": docs, "manifest_digest_sha256": X.digest_docs(docs)}
    sheet = tmp / "facts_sheet.csv"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, ["doc_name", "metric", "value_usd_m", "status"])
        w.writeheader()
        for name, (_, rev) in CO.items():
            w.writerow({"doc_name": name, "metric": "revenue", "value_usd_m": "999999" if name in wrong else rev,
                        "status": "filled"})
    golds = [{"id": "agg01", "question": "Which company had the highest FY2022 revenue?", "expr": "revenue",
              "op": "argmax", "gold": "Amazon", "excluded_by_rule": [], "continuity": True, "verified": True},
             {"id": "agg02", "question": "Which company reported the highest FY2022 total revenue?", "expr": "revenue",
              "op": "argmax", "gold": "Amazon", "excluded_by_rule": [], "continuity": True, "verified": True},
             {"id": "p1_01", "question": "Which company had the second highest FY2022 revenue?", "expr": "revenue",
              "op": "argmax", "gold": "Amazon", "excluded_by_rule": [], "continuity": False, "verified": True}]
    inp = {"corpus": corpus, "golds": golds, "sheet": sheet,
           "verified": {g["id"]: {"verified": True} for g in golds}}
    base = tmp / "p0store"
    base.mkdir()
    store = CorpusStore(base / "store.sqlite")
    for name in P0_NAMES:
        store.ingest(parse(Path(f"/{name}.pdf")), corpus_id=P.CORPUS_ID, doc_id=name, entity=CO[name][0])
    return {"inp": inp, "corpus": corpus, "p0store": store, "p0path": base / "store.sqlite"}


def parse(pdf: Path):
    name = pdf.stem
    rev = f"{CO[name][1]:,.0f}"
    secs = [Section(id="s1", title="Income", level=1, content=f"Statements of Income\nTotal revenues {rev}\n",
                    page_start=60, page_end=60),
            Section(id="s2", title="Risk", level=1, content="Risk factors about weather " * 40,
                    page_start=10, page_end=11)]
    return DocumentPreprocessResult(original_path=str(pdf), sections=secs, figures=[], total_pages=100, tables=[])


def p0_job(fx, tmp: Path) -> tuple[Path, Path]:
    """A fake Phase 0 CPU job: L4 over the Phase 0 documents into the base store + its results.jsonl."""
    rec = P.Recorder(tmp / "p0job" / "results.jsonl")
    b = FakeBackend()
    units = P.windows(fx["p0store"], P.CORPUS_ID, P.CFG.window_chars)
    for f in A0.METRICS:
        for d in P0_NAMES:
            X.l4_unit(rec, fx["p0store"], b, units, f, d, lambda: None)
    l4 = [r for r in rec.records if r["kind"] == "l4_unit"]
    rec.emit("ingest", unit="ingest", prefill_tokens=sum(r["prefill_tokens"] for r in l4),
             corpus_tokens=fx["p0store"].corpus_tokens(P.CORPUS_ID), llm_calls=sum(r["llm_calls"] for r in l4),
             n_chunks=len(fx["p0store"].chunks(P.CORPUS_ID)))
    return tmp / "p0job" / "results.jsonl", fx["p0path"]


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not importable")
class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.guard = lambda: None
        self.patches = [mock.patch.object(X, "PILOT_DOCS", tuple(NEW_NAMES[:3])),
                        mock.patch.object(X, "PILOT_RANKS", (3, 4, 5)),
                        mock.patch.object(X, "INTERIM_NEW_DOCS", 4)]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def _l3(self, fx, out, *, reuse=True, rec=None, parse_fn=None, smoke=False):
        rec = rec or P.Recorder(out / "results.jsonl")
        p0res, p0store = p0_job(fx, self.tmp) if reuse else (None, None)
        if reuse and "reuse_seeded" not in rec.done:
            X.seed_reuse(rec, p0res)
            rec.emit("reuse_seeded", unit="reuse_seeded")
        if not reuse:
            p0store = fx["p0path"]
        res = X.run_l3(rec, out, fx["inp"], SIZES, ORCH, store_src=p0store, guard=self.guard,
                       parse_fn=parse_fn or parse, smoke=smoke)
        return rec, res

    def _run(self, fx, out, rec, stage, backend, **kw):
        store = P.open_store(out / "store.sqlite")
        try:
            return X.run_l4_stage(rec, out, store, backend, fx["inp"], SIZES, stage=stage, guard=self.guard,
                                  reuse_ok=kw.pop("reuse_ok", True), **kw)
        finally:
            store.close()

    # ------------------------------------------------------------------ L3
    def test_l3_loads_new_docs_resumable(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        calls = []
        rec, res = self._l3(fx, out, parse_fn=lambda p: (calls.append(p.stem), parse(p))[1])
        self.assertTrue(res["complete"])
        self.assertEqual(calls, ["MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K", "COSTCO_2022_10K",
                                 "AMAZON_2022_10K"])  # pilot documents first, then ascending size
        kinds = [r["kind"] for r in rec.records]
        for k in ("parse", "store", "l3_doc", "doc_done", "corpus", "m1_l3", "reuse_seeded"):
            self.assertIn(k, kinds)
        self.assertEqual(sum(1 for r in rec.records if r["kind"] == "doc_done"), 5)
        self.assertEqual(sum(r["llm_calls"] for r in rec.records if r["kind"] == "l3_doc"), 0)  # no model in L3
        store = P.open_store(out / "store.sqlite")
        self.assertEqual(len(store.corpus_docs(P.CORPUS_ID)), 7)
        store.close()
        calls.clear()
        rec2, res2 = self._l3(fx, out, rec=P.Recorder(out / "results.jsonl"), reuse=False,
                              parse_fn=lambda p: calls.append(p) or parse(p))
        self.assertEqual(calls, [])  # every document already done: zero parses
        self.assertTrue(res2["complete"])

    def test_l3_three_parse_failures_stop_one_is_recorded(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"

        def bad(names):
            def f(p):
                if p.stem in names:
                    raise RuntimeError("odl crashed")
                return parse(p)
            return f
        rec, res = self._l3(fx, out, parse_fn=bad({"MICROSOFT_2022_10K"}))
        self.assertTrue(res["complete"])  # fewer than 3: recorded doc_fail, run continues
        self.assertEqual([r["doc"] for r in rec.records if r["kind"] == "doc_fail"], ["MICROSOFT_2022_10K"])
        self.assertTrue(next(r for r in rec.records if r["kind"] == "doc_fail")["final"])
        self.assertEqual(res["failed"], ["MICROSOFT_2022_10K"])
        out2 = self.tmp / "out2"
        fx2 = make_fixture(Path(tempfile.mkdtemp()))
        rec2, res2 = self._l3(fx2, out2, parse_fn=bad({"MICROSOFT_2022_10K", "3M_2022_10K", "COSTCO_2022_10K"}))
        self.assertEqual((res2["infrastructure_stop"], res2["rc"]), ("l3_parse_futility", X.RC_L3))
        self.assertEqual(rec2.records[-1]["reason"], "futility_l3_parse")
        self.assertEqual(sum(1 for r in rec2.records if r["kind"] == "doc_done"), 1)  # AMAZON never reached

    def test_l3_failed_doc_counts_as_m1_miss_not_dropped(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res = self._l3(fx, out, parse_fn=lambda p: (_ for _ in ()).throw(RuntimeError("x"))
                            if p.stem == "AMAZON_2022_10K" else parse(p))
        self.assertEqual(res["failed"], ["AMAZON_2022_10K"])
        cells = X.m1_cells(fx["inp"]["sheet"], X.verified_agg(fx["inp"]), [d["doc_name"] for d in fx["corpus"]["docs"]])
        self.assertEqual(len(cells), 7)  # the failed document's cell is still a gold cell
        store = P.open_store(out / "store.sqlite")
        hits = X.m1_hits(store.facts(P.CORPUS_ID), cells)
        self.assertFalse(hits[[c[0] for c in cells].index("AMAZON_2022_10K")])
        store.close()

    # ------------------------------------------------------------------ pilot + ingest
    def test_pilot_then_ingest_reuses_phase0_units(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out)
        n_seed = sum(1 for r in rec.records if r["kind"] == "l4_unit" and r.get("reused"))
        self.assertEqual(n_seed, 16)  # 2 documents x 8 fields, cache hits
        b = FakeBackend()
        with mock.patch.dict(A0.KILL, LENIENT_M3):
            res = self._run(fx, out, rec, "pilot", b)
        self.assertEqual((res["complete"], res["rc"]), (True, X.RC_OK))
        pd = next(r for r in rec.records if r.get("unit") == "pilot_done")
        self.assertEqual(pd["n_pilot_units"], 24)
        self.assertEqual(pd["units_to_run"], 40)  # 5 new documents x 8 fields (Phase 0 units not re-run)
        self.assertEqual(pd["pilot_m1"]["gold"], 3)
        self.assertEqual(pd["gates_tripped"], [])
        self.assertTrue((out / "result.pilot.json").exists())
        calls_pilot = b.calls
        # ingest before the pilot completed is a precondition failure; after it, only the 16 remaining units run
        fx2 = make_fixture(Path(tempfile.mkdtemp()))
        rec_np = P.Recorder(self.tmp / "np" / "results.jsonl")
        (self.tmp / "np").mkdir(exist_ok=True)
        shutil.copy(out / "store.sqlite", self.tmp / "np" / "store.sqlite")
        self.assertEqual(self._run(fx, self.tmp / "np", rec_np, "ingest", FakeBackend())["rc"], X.RC_PRECOND)
        with mock.patch.dict(A0.KILL, LENIENT_M3):
            res = self._run(fx, out, rec, "ingest", b)
        self.assertEqual((res["complete"], res["rc"], res["m3_passed"], res["l4_reused"]), (True, X.RC_OK, True, True))
        fresh = [r for r in rec.records if r["kind"] == "l4_unit" and not r.get("reused")]
        self.assertEqual(len({r["unit"] for r in fresh}), 40)
        self.assertGreater(b.calls, calls_pilot)
        ing = next(r for r in rec.records if r["kind"] == "ingest")
        p0_ing = next(r for r in rec.records if r["kind"] == "p0_ingest")
        inc_calls = sum(r["llm_calls"] for r in fresh)
        self.assertEqual(ing["llm_calls"], p0_ing["llm_calls"] + inc_calls)       # gated = Phase 0 + incremental
        self.assertEqual(ing["incremental"]["llm_calls"], inc_calls)
        self.assertEqual(ing["reused_units"], 16)
        self.assertEqual(ing["corpus_tokens"], P.open_store(out / "store.sqlite").corpus_tokens(P.CORPUS_ID))
        m1 = next(r for r in rec.records if r["kind"] == "m1")
        self.assertEqual((m1["gold_facts"], m1["found"], m1["m1"]), (7, 7, 1.0))
        self.assertEqual((m1["phase0_docs"]["gold"], m1["new_docs"]["gold"]), (2, 5))
        done = json.loads((out / "result.cpu.json").read_text())
        self.assertTrue(done["complete"] and done["m3_passed"] and done["m1"] == 1.0 and not done["gate_tripped"])
        self.assertTrue(X.cpu_ready(done)[0])
        il = next(r for r in rec.records if r.get("unit") == "interim10")
        self.assertEqual((il["n_new_docs_done"], il["gates_tripped"]), (4, []))
        for line in (out / "results.jsonl").read_text().splitlines():
            json.loads(line)  # every line complete JSON, written as produced
        # resume: zero new calls
        b2 = FakeBackend()
        res2 = self._run(fx, out, P.Recorder(out / "results.jsonl"), "ingest", b2)
        self.assertEqual(b2.calls, 0)
        self.assertTrue(res2["complete"])

    def test_no_reuse_runs_every_unit_fresh(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out, reuse=False)
        self.assertEqual(sum(1 for r in rec.records if r["kind"] == "l4_unit"), 0)
        b = FakeBackend()
        with mock.patch.dict(A0.KILL, LENIENT_M3):
            self.assertEqual(self._run(fx, out, rec, "pilot", b, reuse_ok=False)["rc"], X.RC_OK)
            res = self._run(fx, out, rec, "ingest", b, reuse_ok=False)
        self.assertEqual((res["complete"], res["l4_reused"]), (True, False))
        self.assertEqual(len({r["unit"] for r in rec.records if r["kind"] == "l4_unit"}), 56)  # 7 docs x 8
        ing = next(r for r in rec.records if r["kind"] == "ingest")
        self.assertEqual((ing["reused_units"], ing["llm_calls"]), (0, b.calls))

    def test_real_kill_bounds_trip_m3_in_the_pilot(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out)
        res = self._run(fx, out, rec, "pilot", FakeBackend())  # tiny documents: prefill / corpus tokens is large
        self.assertEqual(res["rc"], X.RC_GATE)
        self.assertEqual(res["gate_tripped"], ["M3"])
        cpu = json.loads((out / "result.cpu.json").read_text())
        self.assertEqual((cpu["complete"], cpu["verdict"], cpu["futility_stop"]), (True, "phase1_gate_tripped", "M3"))
        self.assertFalse(X.cpu_ready(cpu)[0])  # the GPU window is never requested
        self.assertEqual(self._run(fx, out, rec, "ingest", FakeBackend())["rc"], X.RC_GATE)  # stays stopped

    def test_pilot_m1_futility(self):
        fx = make_fixture(self.tmp, wrong=tuple(NEW_NAMES[:3]))  # the pilot documents' gold cells never match
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out)
        with mock.patch.dict(A0.KILL, LENIENT_M3):
            res = self._run(fx, out, rec, "pilot", FakeBackend())
        self.assertEqual((res["rc"], res["gate_tripped"]), (X.RC_GATE, ["M1"]))
        self.assertEqual(next(r for r in rec.records if r.get("unit") == "pilot_done")["pilot_m1"]["found"], 0)

    def test_interim_look_stops_when_m1_gate_unreachable(self):
        # pilot documents fine; the two later documents wrong: after all 5 new documents found = 5 of 7 < 0.75 x 7
        fx = make_fixture(self.tmp, wrong=("COSTCO_2022_10K", "AMAZON_2022_10K"))
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out)
        with mock.patch.dict(A0.KILL, LENIENT_M3), mock.patch.object(X, "INTERIM_NEW_DOCS", 5):
            self.assertEqual(self._run(fx, out, rec, "pilot", FakeBackend())["rc"], X.RC_OK)
            res = self._run(fx, out, rec, "ingest", FakeBackend())
        self.assertEqual((res["rc"], res["gate_tripped"], res["where"]), (X.RC_GATE, ["M1"], "interim"))
        il = next(r for r in rec.records if r.get("unit") == "interim10")
        self.assertTrue(il["m1_unreachable"])
        self.assertEqual((il["found"], il["not_yet_evaluated"]), (5, 0))

    def test_interim_look_is_futility_only_never_an_early_go(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out)
        with mock.patch.dict(A0.KILL, LENIENT_M3), mock.patch.object(X, "INTERIM_NEW_DOCS", 3):
            self.assertEqual(self._run(fx, out, rec, "pilot", FakeBackend())["rc"], X.RC_OK)
            b = FakeBackend()
            res = self._run(fx, out, rec, "ingest", b)
        self.assertTrue(rec.done >= {"interim10"})
        self.assertEqual(res["rc"], X.RC_OK)
        self.assertTrue(json.loads((out / "result.cpu.json").read_text())["complete"])
        self.assertEqual(sum(1 for r in rec.records if r.get("unit") == "interim10"), 1)  # evaluated once

    def test_final_m1_below_gate_trips_and_blocks_gpu(self):
        # 5 of 7 cells right: 0.714 < 0.75, but the interim look (reachable) lets the ingest finish
        fx = make_fixture(self.tmp, wrong=("COSTCO_2022_10K", "AMAZON_2022_10K"))
        out = self.tmp / "out"
        rec, _ = self._l3(fx, out)
        with mock.patch.dict(A0.KILL, LENIENT_M3):
            self.assertEqual(self._run(fx, out, rec, "pilot", FakeBackend())["rc"], X.RC_OK)
            res = self._run(fx, out, rec, "ingest", FakeBackend())  # INTERIM_NEW_DOCS = 4: 2 wrong + 2 unseen
        self.assertEqual((res["rc"], res["gate_tripped"], res["where"]), (X.RC_GATE, ["M1"], "final"))
        cpu = json.loads((out / "result.cpu.json").read_text())
        self.assertAlmostEqual(cpu["m1"], 5 / 7)
        self.assertFalse(X.cpu_ready(cpu)[0])

    def test_smoke_runs_one_real_unit_on_the_smallest_new_document(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res = self._l3(fx, out, smoke=True)  # smallest new document only
        self.assertEqual([r["doc"] for r in rec.records if r["kind"] == "doc_done"], ["COSTCO_2022_10K"])
        t0 = max(r["ts"] for r in rec.records)
        b = FakeBackend()
        res = self._run(fx, out, rec, "pilot", b, smoke=True)
        self.assertTrue(res["smoke"])
        new = [r for r in rec.records if r["kind"] == "l4_unit" and r["ts"] >= t0 and not r.get("reused")]
        self.assertEqual([(r["field"], r["doc"]) for r in new], [("revenue", "COSTCO_2022_10K")])
        self.assertGreaterEqual(new[0]["llm_calls"], 1)
        self.assertFalse((out / "result.pilot.json").exists())

    # ------------------------------------------------------------------ GPU
    def _ingested(self, fx, out):
        rec, _ = self._l3(fx, out)
        with mock.patch.dict(A0.KILL, LENIENT_M3):
            self._run(fx, out, rec, "pilot", FakeBackend())
            self._run(fx, out, rec, "ingest", FakeBackend())
        store = P.open_store(out / "store.sqlite")
        store.conn.execute("UPDATE documents SET n_tokens_est = 200000")  # >= 1M tokens: aggregation -> sql
        store.conn.commit()
        return store

    def test_p_large_then_b1_with_early_stop_and_verdict(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        store = self._ingested(fx, out)
        gpu = P.Recorder(self.tmp / "gpu" / "results.jsonl")
        names = sorted({d["company"] for d in fx["corpus"]["docs"]})
        agg = X.verified_agg(fx["inp"])
        b = FakeBackend(final="FINAL: Amazon", search_final="FINAL: Corning")  # B1 wrong on agg01 and p1_01
        with mock.patch.object(X, "B1_LOOK_AT", 2), mock.patch.object(X, "B1_STOP_Z", 1.4):
            res = X.run_answers(gpu, store, b, agg=agg, names=names, guard=self.guard)
            self.assertTrue(res["complete"])
            self.assertTrue(res["b1_early_stop"])
            ans = {(r["arm"], r["qid"]): r for r in gpu.records if r["kind"] == "answer"}
            self.assertEqual({a for a, _ in ans}, {"P-large", "B1"})  # no P-small / B-today / B2 in AGG-1.1
            self.assertEqual(len([k for k in ans if k[0] == "P-large"]), 3)
            self.assertEqual([k[1] for k in ans if k[0] == "B1"], ["agg01", "agg02"])  # stopped after 2 of 3
            self.assertEqual(ans[("P-large", "agg01")]["path"], "sql")
            self.assertIs(ans[("P-large", "agg01")]["correct"], True)
            self.assertIs(ans[("B1", "agg01")]["correct"], False)
            look = next(r for r in gpu.records if r.get("unit") == "b1_look")
            self.assertTrue(look["stop"])
            cpu = [{"kind": "ingest", "prefill_tokens": 1, "llm_calls": 1, "corpus_tokens": 100, "n_chunks": 10},
                   {"kind": "m1", "m1": 0.9, "gold_facts": 7, "found": 6}]
            v = X.verdict(cpu, gpu.records, agg, store=store)
            self.assertTrue(v["complete"])
            self.assertTrue(v["M2"]["b1_early_stop"])
            self.assertEqual(v["M2"]["paired_on_b1_answered"]["n"], 2)
            self.assertIsNotNone(v["sql_latency"]["sql_exec_max_s"])
            # resume after the early stop: no new calls, B1 stays stopped
            b2 = FakeBackend()
            res2 = X.run_answers(P.Recorder(self.tmp / "gpu" / "results.jsonl"), store, b2, agg=agg, names=names,
                                 guard=self.guard)
            self.assertEqual(b2.calls, 0)
            self.assertTrue(res2["complete"])

    def test_b1_runs_to_the_end_without_a_stop(self):
        fx = make_fixture(self.tmp)
        store = self._ingested(fx, self.tmp / "out")
        gpu = P.Recorder(self.tmp / "gpu" / "results.jsonl")
        names = sorted({d["company"] for d in fx["corpus"]["docs"]})
        agg = X.verified_agg(fx["inp"])
        with mock.patch.object(X, "B1_LOOK_AT", 2):
            res = X.run_answers(gpu, store, FakeBackend(final="FINAL: Amazon"), agg=agg, names=names, guard=self.guard)
        self.assertTrue(res["complete"])
        self.assertFalse(res["b1_early_stop"])  # P-large == B1: z = 0, B1 answers all three
        self.assertEqual(sum(1 for r in gpu.records if r["kind"] == "answer" and r["arm"] == "B1"), 3)

    def test_p_large_branch_matches_phase0_run_answers(self):
        """The P-large path is the Phase 0 branch: same records for the same backend on the same store."""
        fx = make_fixture(self.tmp)
        store = self._ingested(fx, self.tmp / "out")
        names = sorted({d["company"] for d in fx["corpus"]["docs"]})
        agg = X.verified_agg(fx["inp"])
        mine = P.Recorder(self.tmp / "mine" / "results.jsonl")
        X.run_answers(mine, store, FakeBackend(), agg=agg, names=names, guard=self.guard)
        theirs = P.Recorder(self.tmp / "theirs" / "results.jsonl")
        P.run_answers(theirs, store, FakeBackend(), agg=agg, single=[], keys={}, names=names, guard=self.guard)

        def view(rec):
            return {(r["arm"], r["qid"]): (r["status"], r["correct"], r["calls"], r["prefill_tokens"], r.get("path"),
                                           r.get("sql")) for r in rec.records
                    if r["kind"] == "answer" and r["arm"] in ("P-large", "B1")}
        self.assertEqual(view(mine), view(theirs))


# --------------------------------------------------------------------------------------------- CLI
class CliTests(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(argv)
        return rc, buf.getvalue()

    @unittest.skipUnless(all(Path(d["pdf"]).exists() for d in json.loads((D1 / "corpus.json").read_text())["docs"])
                         and (P0_JOB / "SOURCE.sha256").exists() and HAVE_ORCH, "PDFs / Phase 0 job not on this host")
    def test_dry_run_plans_units_without_side_effects(self):
        before = set(os.listdir(D1))
        rc, text = self._run(["--phase", "ingest", "--dry-run", "--orch-root", str(ORCH), "--p0-job", str(P0_JOB)])
        self.assertEqual(rc, 0, text)
        out = json.loads(text)
        self.assertEqual((out["model_calls"], out["servers_started"], out["wrote_files"]), (0, 0, 0))
        self.assertTrue(out["l4_reuse"]["reuse_ok"])
        self.assertEqual(out["l4_reuse"]["p0_reused_units"]["llm_calls"], 483)
        self.assertEqual(out["l4_reuse"]["p0_reused_units"]["prefill_tokens"], 613361)
        ing = out["plan"]["ingest"]
        self.assertEqual((ing["units_reused"], ing["units_to_run_after_pilot"], ing["total_units"]), (80, 120, 224))
        self.assertEqual(out["plan"]["pilot"]["units"], 24)
        self.assertEqual(out["plan"]["pilot"]["docs"], list(X.PILOT_DOCS))
        self.assertEqual(out["preflight"]["m1_cells"]["total"], 211)
        self.assertEqual(out["predeclared_sha256"], X.PREDECLARED_SHA)
        self.assertEqual(set(os.listdir(D1)) - {"__pycache__"}, before - {"__pycache__"})

    def test_dry_run_gpu_plan_and_unready_precondition(self):
        rc, text = self._run(["--phase", "gpu-answers", "--dry-run"])
        self.assertEqual(rc, 0, text)
        out = json.loads(text)
        self.assertEqual((out["plan"]["p_large"]["n"], out["plan"]["b1"]["n"]), (58, 58))
        self.assertEqual(out["plan"]["p_large"]["order"][:2], ["agg01", "agg02"])
        self.assertEqual(out["plan"]["p_large"]["order"][20], "p1_01")  # continuity first, then new by id
        self.assertEqual(out["plan"]["arms_not_run"], ["P-small", "B-today", "B2"])
        self.assertFalse(out["precondition_cpu"]["ready"])
        self.assertEqual(out["model_calls"], 0)

    def test_dry_run_rejects_a_tampered_input(self):
        tmp = Path(tempfile.mkdtemp())
        for f in list(X.FROZEN_P1) + ["PREDECLARED.md", "extracted_sizes.json", "build_phase1.py", "phase1_questions.py"]:
            shutil.copy(D1 / f, tmp / f)
        q = json.loads((tmp / "aggregation_questions.json").read_text())
        q["questions"][0]["q"] += " (edited)"
        (tmp / "aggregation_questions.json").write_text(json.dumps(q, indent=1) + "\n")
        rc, text = self._run(["--phase", "gpu-answers", "--dry-run", "--set-dir", str(tmp)])
        self.assertEqual(rc, X.RC_PREFLIGHT)
        self.assertIn("aggregation_questions.json", text)

    def test_parameters_replace_the_hardcoded_phase0_set(self):
        rc, text = self._run(["--phase", "gpu-answers", "--dry-run", "--golds", str(D0 / "golds.jsonl"),
                              "--corpus", str(D0 / "corpus.json"), "--verification", str(D0 / "verification_record.json"),
                              "--facts-sheet", str(D0 / "facts_sheet.csv"), "--pins", "none"])
        # a 10-document corpus is accepted as input (parameters, no hardcoded 28); it just is not the Phase 1 set
        out = json.loads(text)
        self.assertEqual(out["preflight"]["n_docs"], 10)
        self.assertEqual(out["plan"]["p_large"]["n"], 20)

    def test_gpu_phase_fails_closed_without_a_ready_cpu_result(self):
        tmp = Path(tempfile.mkdtemp())
        rc, _ = self._run(["--phase", "gpu-answers", "--out", str(tmp / "o"), "--orch-root", str(ORCH)])
        self.assertEqual(rc, X.RC_PRECOND)
        last = json.loads((tmp / "o" / "results.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last["kind"], "precondition_failed")
        cpu = tmp / "cpu"
        cpu.mkdir()
        (cpu / "results.jsonl").write_text("")
        (cpu / "result.cpu.json").write_text(json.dumps({"complete": True, "m1": 0.7, "m3_passed": True}))
        rc, _ = self._run(["--phase", "gpu-audit", "--out", str(tmp / "o2"), "--orch-root", str(ORCH),
                           "--cpu-results", str(cpu / "results.jsonl")])
        self.assertEqual(rc, X.RC_PRECOND)
        self.assertIn("not ready", (tmp / "o2" / "results.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
