"""AGG-1.3 tests (fake backends and a fake document parser: no model, no network, no process, no PDF).

Stdlib tests always run. The store / L3 / L4 tests import the orchestrator package and skip without it: run them with the
orchestrator venv python and ``AGG_ORCH_ROOT=<tree at the AGG-1.3 pin>`` (the tree must carry ``superseded_facts``).
``ORCHESTRATOR_FEATURE_AGG_LAYER=1`` is set here.
"""

from __future__ import annotations

import contextlib
import copy
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
import agg13 as G  # noqa: E402
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_agg13 as R  # noqa: E402

DEFAULT_ORCH = "/mnt/raid0/llm/tmp/agg13-reingest-cpu-20261111-r3/orch"   # the amended stage (amendment 2); AGG_ORCH_ROOT overrides
ORCH = Path(os.environ.get("AGG_ORCH_ROOT", DEFAULT_ORCH))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation.llm_backend import GATE_GBNF  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = hasattr(CorpusStore, "superseded_facts")
except Exception:  # pragma: no cover
    HAVE_ORCH = False

LENIENT = {"llm_calls": 1e9, "prefill_tokens": 1e9, "plain_calls": 1e9, "plain_prefill": 1e9}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- pins and arithmetic
class PinTests(unittest.TestCase):
    def test_frozen_inputs_and_predeclared_match_pins(self):
        self.assertEqual(G.pin_problems(D1), [])
        self.assertEqual(sha(D1 / "PREDECLARED_AGG13.md"), G.PREDECLARED_SHA)
        self.assertEqual(sha(D1 / "PREDECLARED_AGG13_AMENDMENT1_20261111.md"), G.AMENDMENT1_SHA)
        self.assertIn("160.17", (D1 / "PREDECLARED_AGG13_AMENDMENT1_20261111.md").read_text())
        self.assertEqual(sha(D1 / G.AMENDMENT2_FILE), G.AMENDMENT2_SHA)
        self.assertEqual(G.AMENDMENT2_FILE, "PREDECLARED_AGG13_AMENDMENT2_20261112.md")
        self.assertIn(X.AMENDED_L4["src/aggregation/facts_llm.py"], (D1 / G.AMENDMENT2_FILE).read_text())
        for name, want in G.FROZEN.items():
            self.assertEqual(sha(D1 / name), want, name)

    def test_amendment2_pin_is_asserted_by_preflight(self):
        tmp = Path(tempfile.mkdtemp())
        for f in list(G.FROZEN) + ["PREDECLARED_AGG13.md", "PREDECLARED_AGG13_AMENDMENT1_20261111.md", G.AMENDMENT2_FILE]:
            shutil.copy(D1 / f, tmp / f)
        self.assertEqual(G.pin_problems(tmp), [])
        (tmp / G.AMENDMENT2_FILE).write_text((tmp / G.AMENDMENT2_FILE).read_text() + "x")
        self.assertTrue(any(G.AMENDMENT2_FILE in p for p in G.pin_problems(tmp)))
        (tmp / G.AMENDMENT2_FILE).unlink()
        self.assertTrue(any(G.AMENDMENT2_FILE in p for p in G.pin_problems(tmp)))

    def test_agg11_predeclared_untouched(self):
        self.assertEqual(sha(D1 / "PREDECLARED.md"), X.PREDECLARED_SHA)
        for name, want in X.FROZEN_P1.items():
            self.assertEqual(sha(D1 / name), want, name)

    def test_tampered_gold_fails_pin(self):
        tmp = Path(tempfile.mkdtemp())
        for f in list(G.FROZEN) + ["PREDECLARED_AGG13.md", "PREDECLARED_AGG13_AMENDMENT1_20261111.md"]:
            shutil.copy(D1 / f, tmp / f)
        (tmp / "agg13_cells.csv").write_text((tmp / "agg13_cells.csv").read_text().replace("168088", "168089"))
        self.assertTrue(any("agg13_cells.csv" in p for p in G.pin_problems(tmp)))

    def test_g3_bounds_are_three_28ths_plus_ten_percent_of_the_agg11_ingest(self):
        b = G.g3_bounds()
        self.assertAlmostEqual(b["llm_calls"], 1359 * 3 / 28 * 1.10)
        self.assertAlmostEqual(b["prefill_tokens"], 1_712_985 * 3 / 28 * 1.10)
        self.assertAlmostEqual(b["plain_calls"], 145.607, places=2)
        # the AGG-1.1 pilot cost of the same three documents sits under the bound (the box's expectation)
        self.assertLess(146, b["llm_calls"])
        self.assertLess(185_976, b["prefill_tokens"])
        self.assertEqual(sum(v["llm_calls"] for v in G.AGG11_PILOT.values()), 146)
        self.assertEqual(sum(v["prefill_tokens"] for v in G.AGG11_PILOT.values()), 185_976)

    def test_g3_eval_boundary(self):
        b = G.g3_bounds()
        self.assertTrue(G.g3_eval({"llm_calls": 160, "prefill_tokens": 201_887}, b)["ok"])
        self.assertEqual(G.g3_eval({"llm_calls": 161, "prefill_tokens": 1}, b)["over"], ["llm_calls"])
        self.assertEqual(G.g3_eval({"llm_calls": 1, "prefill_tokens": 201_888}, b)["over"], ["prefill_tokens"])

    def test_first_doc_budget_stop_is_2x_the_agg11_pilot_rate(self):
        slot = "MICROSOFT_2022_10K"
        mk = lambda c, p, w: [{"llm_calls": c, "prefill_tokens": p, "wall_s": w}]  # noqa: E731
        self.assertIsNone(G.first_doc_over(mk(96, 115_236, 832.4), slot))     # exactly 2x: not over
        self.assertEqual(G.first_doc_over(mk(97, 1, 1), slot)["over"], ["llm_calls"])
        self.assertEqual(G.first_doc_over(mk(1, 115_237, 1), slot)["over"], ["prefill_tokens"])
        self.assertEqual(G.first_doc_over(mk(1, 1, 832.5), slot)["over"], ["wall_s"])

    def test_g5_m4_unchanged(self):
        ans = lambda n, c, p: [{"final": True, "calls": c, "prefill_tokens": p} for _ in range(n)]  # noqa: E731
        self.assertTrue(G.g5_eval(ans(58, 2, 606), 58)["ok"])
        self.assertTrue(G.g5_eval(ans(58, 3, 2510), 58)["ok"])
        self.assertFalse(G.g5_eval(ans(58, 4, 606), 58)["ok"])
        self.assertFalse(G.g5_eval(ans(58, 2, 2511), 58)["ok"])
        g = G.g5_eval(ans(57, 2, 606), 58)
        self.assertEqual((g["complete"], g["ok"]), (False, False))

    def test_g1_eval_exactly_three_and_twenty_five(self):
        names = [f"D{i}" for i in range(25)] + list(G.REPLACED)
        man = [{"name": n, "status": "replaced" if n in G.REPLACED else "unchanged",
                "parse": "real" if n in G.REPLACED else "cached", "l3_facts": 5 if n in G.REPLACED else 0,
                "l3_ran": n in G.REPLACED,
                "l4_units_run": 8 if n in G.REPLACED else 0, "llm_calls": 40 if n in G.REPLACED else 0} for n in names]
        self.assertTrue(G.g1_eval(man, [])["ok"])
        bad = [dict(m) for m in man]
        bad[0]["parse"] = "real"                    # a retained document that was really re-parsed
        self.assertFalse(G.g1_eval(bad, [])["ok"])
        bad = [dict(m) for m in man]
        bad[-1]["l4_units_run"] = 7                 # a changed document not fully re-extracted
        self.assertFalse(G.g1_eval(bad, [])["ok"])
        self.assertFalse(G.g1_eval(man, ["X"])["ok"])


class CorpusAndCellsTests(unittest.TestCase):
    def setUp(self):
        self.inp = G.load_inputs(D1)

    def test_corpus_is_the_agg11_corpus_with_three_slots_swapped(self):
        self.assertEqual(G.corpus_problems(self.inp["corpus"], self.inp["corpus11"], check_pdfs=False), [])
        new = {d["doc_name"]: d for d in self.inp["corpus"]["docs"]}
        old = {d["doc_name"]: d for d in self.inp["corpus11"]["docs"]}
        self.assertEqual(sorted(new), sorted(old))
        for n in G.REPLACED:
            self.assertNotEqual(new[n]["sha256"], old[n]["sha256"])
            self.assertTrue(new[n]["file_name"].endswith("_2021_10K"))
            self.assertEqual(new[n]["replaces"]["sha256"], old[n]["sha256"])
        self.assertEqual(sum(1 for n in new if new[n] != old[n]), 3)

    def test_corpus_problems_catch_a_changed_retained_entry_and_a_wrong_replacement(self):
        c = json.loads(json.dumps(self.inp["corpus"]))
        c["docs"][0]["sha256"] = "0" * 64
        self.assertTrue(any("differs" in p or "digest" in p for p in G.corpus_problems(c, self.inp["corpus11"], check_pdfs=False)))
        c = json.loads(json.dumps(self.inp["corpus"]))
        d = next(x for x in c["docs"] if x["doc_name"] == G.REPLACED[0])
        d["file_name"] = "MICROSOFT_2023_10K"
        c["manifest_digest_sha256"] = X.digest_docs(c["docs"])
        self.assertTrue(any("FY2021" in p for p in G.corpus_problems(c, self.inp["corpus11"], check_pdfs=False)))

    def test_cells_187_retained_plus_24_new(self):
        self.assertEqual(G.cells_problems(self.inp), [])
        cells = G.m1_cells13(self.inp)
        self.assertEqual(len(cells), 211)
        self.assertEqual(sum(1 for c in cells if c[0] in G.REPLACED), 24)
        self.assertEqual({c[3] for c in cells if c[0] in G.REPLACED}, {"2021"})
        self.assertEqual({c[3] for c in cells if c[0] not in G.REPLACED}, {"2022"})
        self.assertEqual(len([c for c in cells if c[0] not in G.REPLACED]), 187)

    def test_m1_hits_use_the_cells_own_year_and_no_unit_leniency(self):
        cells = [("A", "revenue", 100.0, "2021"), ("B", "revenue", 50.0, "2022"), ("C", "revenue", 7.0, "2021")]
        facts = [{"doc_id": "A", "field": "revenue", "period": "FY2021", "value": 100.0, "unit": "USD_millions"},
                 {"doc_id": "A", "field": "revenue", "period": "FY2022", "value": 50.0, "unit": "USD_millions"},
                 {"doc_id": "B", "field": "revenue", "period": "FY2021", "value": 50.0, "unit": "USD_millions"},  # wrong year
                 {"doc_id": "C", "field": "revenue", "period": "2021", "value": 7000.0, "unit": "USD_millions"}]  # 1000x off
        self.assertEqual(G.m1_hits_fy(facts, cells), [True, False, False])
        facts[3]["unit"] = "USD_thousands"
        self.assertEqual(G.m1_hits_fy(facts, cells), [True, False, True])
        s = G.m1_summary13(cells, [True, False, True])
        self.assertEqual((s["overall"]["found"], s["overall"]["gold"]), (2, 3))

    def test_fy_contract_passes_on_the_real_inputs_and_fails_the_r2_configuration(self):
        """The AGG-1.3 r2 defect: FY2022 descriptions for FY2021 slots. The hard preflight check catches it."""
        self.assertEqual(G.fy_problems(self.inp), [])
        for d in G.replaced_docs(self.inp["corpus"]):
            self.assertEqual(X.doc_fiscal_year(d, self.inp["corpus11"]), "2021")
        for d in G.retained(self.inp["corpus"]):
            self.assertEqual(X.doc_fiscal_year(d, self.inp["corpus11"]), "2022")
        broken = copy.deepcopy(self.inp)
        for d in broken["corpus"]["docs"]:
            d.pop("fiscal_year", None)                       # the manifest no longer says the slots are FY2021
        probs = G.fy_problems(broken)
        self.assertEqual(len(probs), 3 * len(A0.METRICS))
        self.assertTrue(all("names FY2022 but the gold cells are FY2021" in p for p in probs), probs[:2])

    def test_runner_preflight_reports_the_fy_contract(self):
        broken = copy.deepcopy(G.load_inputs(D1))
        for d in broken["corpus"]["docs"]:
            d.pop("fiscal_year", None)
        a = R.build_parser().parse_args(["--phase", "preflight", "--pins", "none"])
        problems, _ = R.checks(a, broken, deep=False)
        self.assertTrue(any(p.startswith("fiscal-year contract:") for p in problems))

    def test_cells_problems_reject_a_missing_metric(self):
        inp = dict(self.inp)
        inp["cells13"] = [r for r in self.inp["cells13"] if not (r["doc_name"] == "3M_2022_10K" and r["metric"] == "capex")]
        self.assertTrue(any("need the 8 metrics" in p for p in G.cells_problems(inp)))


# --------------------------------------------------------------------------------------------- fixture
RET = ["CORNING_2022_10K", "PAYPAL_2022_10K"]
NAMES = RET + list(G.REPLACED)
CO = {"CORNING_2022_10K": "Corning", "PAYPAL_2022_10K": "Paypal", "MICROSOFT_2022_10K": "Microsoft",
      "JOHNSON_JOHNSON_2022_10K": "Johnson & Johnson", "3M_2022_10K": "3M"}
OLD_REV = {"CORNING_2022_10K": 14189.0, "PAYPAL_2022_10K": 27518.0, "MICROSOFT_2022_10K": 198270.0,
           "JOHNSON_JOHNSON_2022_10K": 94943.0, "3M_2022_10K": 34229.0}
NEW_REV = {"MICROSOFT_2022_10K": 168088.0, "JOHNSON_JOHNSON_2022_10K": 93775.0, "3M_2022_10K": 35355.0}
NEW_FILE = {"MICROSOFT_2022_10K": "MICROSOFT_2021_10K", "JOHNSON_JOHNSON_2022_10K": "JOHNSON_JOHNSON_2021_10K",
            "3M_2022_10K": "3M_2021_10K"}


class FakeBackend:
    """Revenue rows only for text stating ``Total revenues N`` (period from ``FY....`` in the text, default FY2022)."""

    def __init__(self):
        self.calls = self.prefill_tokens = 0

    def _bill(self, prompt):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        self._bill(prompt)
        if grammar == GATE_GBNF:
            return {"text": "yes", "prompt_tokens": len(prompt) // 4, "cache_n": 0}
        body = prompt.split("Excerpt:\n", 1)[-1]
        m = re.search(r"Total revenues [\d,]+", body)
        per = re.search(r"FY(\d{4})", body)
        rows = [] if not m or "Field: revenue" not in prompt else [{
            "period": f"FY{per.group(1) if per else '2022'}", "value": float(m.group(0).split()[-1].replace(",", "")),
            "unit": "USD_millions", "quote": m.group(0), "confidence": "medium"}]
        return {"text": json.dumps(rows), "prompt_tokens": len(prompt) // 4, "cache_n": 0}


class DescBackend(FakeBackend):
    """Behaves like the r2 model: the period it reports and whether it answers at all follow the fiscal year the FIELD
    DESCRIPTION asks for (``FY2021`` in the description -> only a FY2021 line in the excerpt can answer)."""

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        if grammar == GATE_GBNF:
            return super().complete(prompt, grammar=grammar, n_predict=n_predict, cache_prompt=cache_prompt, id_slot=id_slot)
        self._bill(prompt)
        head, body = prompt.split("Excerpt:\n", 1)
        want = re.search(r"FY(\d{4})", head.split("Field:", 1)[-1])
        m = re.search(r"Total revenues [\d,]+", body)
        has = re.search(r"FY(\d{4})", body)
        ok = m and "Field: revenue" in prompt and want and has and want.group(1) == has.group(1)
        rows = [{"period": f"FY{want.group(1)}", "value": float(m.group(0).split()[-1].replace(",", "")),
                 "unit": "USD_millions", "quote": m.group(0), "confidence": "medium"}] if ok else []
        return {"text": json.dumps(rows), "prompt_tokens": len(prompt) // 4, "cache_n": 0}


def result_for(name: str, rev: float, fy: str):
    secs = [Section(id="s1", title="Income", level=1, content=f"Statements of Income FY{fy}\nTotal revenues {rev:,.0f}\n",
                    page_start=60, page_end=60),
            Section(id="s2", title="Risk", level=1, content=f"Risk factors about weather {name} " * 40,
                    page_start=10, page_end=11)]
    return DocumentPreprocessResult(original_path=f"/{name}.pdf", sections=secs, figures=[], total_pages=100, tables=[])


def make_fixture(tmp: Path, *, tamper_retained_cache: str | None = None, drop_cache: str | None = None) -> dict:
    """5-document corpus: 2 retained + the 3 slots. A base store with the OLD documents + L3 + 40 L4 units, the AGG-1.1
    results file, a parse cache for the retained documents, new fake PDFs for the slots."""
    pdfs, cache = tmp / "pdfs", tmp / "cache11"
    pdfs.mkdir()
    cache.mkdir()
    old_docs, new_docs = [], []
    for n in NAMES:
        old = pdfs / f"{n}.pdf"
        old.write_bytes(f"%PDF old {n}".encode())
        o = {"doc_name": n, "company": CO[n], "phase0": n in RET, "pdf": str(old), "sha256": sha(old),
             "bytes": old.stat().st_size, "pages": 100}
        old_docs.append(o)
        if n in G.REPLACED:
            newp = pdfs / f"{NEW_FILE[n]}.pdf"
            newp.write_bytes(f"%PDF new {n}".encode())
            e = dict(o, pdf=str(newp), sha256=sha(newp), bytes=newp.stat().st_size, file_name=NEW_FILE[n],
                     fiscal_year=2021, replaces={"file_name": n, "sha256": o["sha256"], "bytes": o["bytes"], "pages": 100})
            new_docs.append(e)
        else:
            new_docs.append(dict(o))
    corpus11 = {"docs": old_docs, "manifest_digest_sha256": X.digest_docs(old_docs)}
    corpus13 = {"docs": new_docs, "manifest_digest_sha256": X.digest_docs(new_docs)}
    # base store (AGG-1.1 state) + its results
    base_dir = tmp / "agg11"
    base_dir.mkdir()
    store = CorpusStore(base_dir / "store.sqlite")
    from l3_load import doc_l3_facts  # noqa: PLC0415
    for d in old_docs:
        res = result_for(d["doc_name"], OLD_REV[d["doc_name"]], "2022")
        store.ingest(res, corpus_id=P.CORPUS_ID, doc_id=d["doc_name"], entity=d["company"])
        store.write_facts(doc_l3_facts(store, P.CORPUS_ID, d["doc_name"])[0])
        if d["doc_name"] in RET:
            res_c = result_for(d["doc_name"], OLD_REV[d["doc_name"]], "2022")
            blob = res_c.to_cache_dict()
            blob["_parse_s"] = 1.0
            if d["doc_name"] == tamper_retained_cache:
                blob = result_for(d["doc_name"], 1.0, "2022").to_cache_dict()
            if d["doc_name"] != drop_cache:
                (cache / G.cache_name(d)).write_text(json.dumps(blob))
    rec = P.Recorder(base_dir / "results.jsonl")
    units = P.windows(store, P.CORPUS_ID, P.CFG.window_chars)
    b = FakeBackend()
    for f in A0.METRICS:
        for d in old_docs:
            X.l4_unit(rec, store, b, units, f, d["doc_name"], lambda: None)
    l4 = [r for r in rec.records if r["kind"] == "l4_unit"]
    rec.emit("ingest", unit="ingest", prefill_tokens=sum(r["prefill_tokens"] for r in l4), llm_calls=sum(r["llm_calls"] for r in l4),
             corpus_tokens=store.corpus_tokens(P.CORPUS_ID), n_chunks=len(store.chunks(P.CORPUS_ID)))
    store.close()
    # gold: the retained FY2022 revenue cells + the new FY2021 cells
    sheet = tmp / "facts_sheet.csv"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, ["doc_name", "metric", "value_usd_m", "status"])
        w.writeheader()
        for n in RET:
            w.writerow({"doc_name": n, "metric": "revenue", "value_usd_m": OLD_REV[n], "status": "filled"})
    cells13 = [{"doc_name": n, "metric": "revenue", "fiscal_year": "2021", "value_usd_m": str(NEW_REV[n]),
                "status": "filled", "verified": "true"} for n in G.REPLACED]
    golds = [{"id": "agg01", "question": "Which company had the highest FY2022 revenue?", "expr": "revenue",
              "op": "argmax", "gold": "Paypal", "excluded_by_rule": [], "continuity": True, "verified": True,
              "evidence": [{"doc": n} for n in NAMES]}]
    inp = {"corpus": corpus13, "corpus11": corpus11, "golds": golds, "sheet": sheet, "cells13": cells13,
           "verified": {"agg01": {"verified": True}}}
    return {"inp": inp, "base": base_dir / "store.sqlite", "results11": base_dir / "results.jsonl", "cache11": cache}


def new_parse(pdf: Path):
    stem = pdf.stem
    slot = next(s for s, f in NEW_FILE.items() if f == stem)
    return result_for(slot, NEW_REV[slot], "2021")


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with superseded_facts not importable")
class ReingestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.guard = lambda: None
        self.patches = [mock.patch.object(G, "N_DOCS", 5)]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def _run(self, fx, out, backend=None, *, bounds=LENIENT, parse_fn=new_parse, rec=None, smoke=False, **kw):
        rec = rec or P.Recorder(out / "results.jsonl")
        backend = backend or FakeBackend()
        res = G.run_reingest(rec, out, fx["inp"], Path("/unused"), backend, base_store=fx["base"],
                             agg11_results=fx["results11"], cache_dirs=[fx["cache11"]], guard=self.guard,
                             parse_fn=parse_fn, bounds=bounds, smoke=smoke, **kw)
        return rec, res, backend

    def test_three_replaced_and_the_rest_skipped_by_hash(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        parsed = []
        rec, res, b = self._run(fx, out, parse_fn=lambda p: (parsed.append(p.stem), new_parse(p))[1])
        self.assertEqual(res["rc"], 0, res)
        self.assertEqual(sorted(parsed), sorted(NEW_FILE.values()))       # exactly the 3 replaced were really parsed
        man = json.loads((out / "reingest_manifest.json").read_text())
        self.assertEqual((man["n_reparsed_reextracted"], man["n_skipped_by_hash"]), (3, 2))
        st = {d["name"]: d for d in man["docs"]}
        for n in RET:
            self.assertEqual((st[n]["status"], st[n]["parse"], st[n]["l4_units_run"], st[n]["llm_calls"]), ("unchanged", "cached", 0, 0))
        for n in G.REPLACED:
            self.assertEqual((st[n]["status"], st[n]["parse"], st[n]["l4_units_run"]), ("replaced", "real", 8))
            self.assertTrue(st[n]["superseded"])
        self.assertTrue(res["g1"]["ok"] and res["g2"]["ok"] and res["g3"]["ok"] and res["g4"]["ok"], res)
        self.assertEqual(res["verdict"], "agg_1_3_cpu_pass")
        self.assertTrue(G.cpu_ready(res)[0])
        # cost: only the 24 new units spent calls; the 16 seeded units are cache hits
        recs = [r for r in rec.records if r["kind"] == "l4_unit"]
        self.assertEqual(sum(1 for r in recs if r.get("reused")), 16)
        self.assertEqual(sum(1 for r in recs if not r.get("reused")), 24)
        self.assertEqual(res["fresh"]["llm_calls"], b.calls)
        # M1: retained FY2022 + new FY2021 cells, all found
        self.assertEqual((res["g4"]["overall"]["gold"], res["g4"]["overall"]["found"]), (5, 5))
        self.assertEqual((res["m1_changed"]["gold"], res["m1_changed"]["found"]), (3, 3))
        self.assertIsNone(res["g4"]["retained_equals_agg11"])       # the fixture's AGG-1.1 results carry no per-doc M1
        # the store: old hashes superseded and gone, ledger complete
        s = CorpusStore(out / "store.sqlite")
        base = next(r for r in rec.records if r.get("unit") == "base_snapshot")
        for slot in G.REPLACED:
            old = base["old_hashes"][slot]
            self.assertEqual(s.conn.execute("SELECT COUNT(*) FROM facts WHERE doc_hash = ?", [old]).fetchone()[0], 0)
            self.assertEqual(len(s.superseded_facts(P.CORPUS_ID, doc_hash=old)), base["old_fact_counts"][slot])
        self.assertEqual(len(s.superseded_documents(P.CORPUS_ID)), 3)
        self.assertEqual(json.loads(s.conn.execute("SELECT value FROM meta WHERE key = 'agg13_reingest_manifest'").fetchone()[0])["n_skipped_by_hash"], 2)
        s.close()
        # the AGG-1.1 base store was only read
        self.assertEqual(CorpusStore(fx["base"]).superseded_documents(), [])
        # resume: zero new calls, same verdict
        b2 = FakeBackend()
        rec2, res2, _ = self._run(fx, out, b2, parse_fn=lambda p: (_ for _ in ()).throw(AssertionError("re-parsed")))
        self.assertEqual(b2.calls, 0)
        self.assertEqual(res2["rc"], 0)

    def test_changed_slots_are_extracted_with_their_own_fiscal_year(self):
        """The r2 defect, end to end: a model that follows the description's fiscal year finds the FY2021 cells only when
        the description names FY2021. Under the r2 configuration (FY2022 for every slot) the changed cells score 0."""
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, DescBackend())
        self.assertEqual(res["rc"], 0, res)
        self.assertEqual((res["m1_changed"]["gold"], res["m1_changed"]["found"]), (3, 3))
        fresh = [r for r in rec.records if r["kind"] == "l4_unit" and not r.get("reused")]
        self.assertEqual({r["fy"] for r in fresh}, {"2021"})
        # r2 configuration: the manifest year is ignored, FY2022 descriptions
        tmp2 = Path(tempfile.mkdtemp())
        fx2 = make_fixture(tmp2)
        with mock.patch.object(X, "doc_fiscal_year", lambda e, c=None: "2022"):
            res2 = G.run_reingest(P.Recorder(tmp2 / "out" / "results.jsonl"), tmp2 / "out", fx2["inp"], Path("/unused"),
                                  DescBackend(), base_store=fx2["base"], agg11_results=fx2["results11"],
                                  cache_dirs=[fx2["cache11"]], guard=self.guard, parse_fn=new_parse, bounds=LENIENT)
        self.assertEqual((res2["m1_changed"]["gold"], res2["m1_changed"]["found"]), (3, 0))

    def test_retained_document_without_a_cached_parse_trips_g1_before_any_compute(self):
        fx = make_fixture(self.tmp, drop_cache="PAYPAL_2022_10K")
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out)
        self.assertEqual((res["rc"], res["gate_tripped"]), (G.RC_GATE, ["G1"]))
        self.assertEqual(b.calls, 0)
        self.assertEqual([r for r in rec.records if r["kind"] == "l4_unit" and not r.get("reused")], [])
        self.assertFalse(G.cpu_ready(json.loads((out / "result.cpu.json").read_text()))[0])

    def test_a_retained_cache_that_hashes_differently_is_refused_and_never_replaces_the_document(self):
        fx = make_fixture(self.tmp, tamper_retained_cache="CORNING_2022_10K")
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out)
        self.assertEqual((res["rc"], res["gate_tripped"]), (G.RC_GATE, ["G1"]))
        s = CorpusStore(out / "store.sqlite")
        self.assertEqual(s.superseded_documents(), [])               # nothing was replaced
        self.assertGreater(s.conn.execute("SELECT COUNT(*) FROM facts WHERE doc_id = 'CORNING_2022_10K'").fetchone()[0], 0)
        self.assertEqual(b.calls, 0)

    def test_parse_failure_of_a_changed_document_trips_g1(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, parse_fn=lambda p: (_ for _ in ()).throw(RuntimeError("odl down")))
        self.assertEqual((res["rc"], res["gate_tripped"]), (G.RC_GATE, ["G1"]))
        self.assertEqual(self._run(fx, out)[1]["rc"], G.RC_GATE)    # stays stopped on resume

    def test_g3_trip_stops_at_the_first_unit_over_the_bound(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        tight = dict(LENIENT, llm_calls=5.0)
        rec, res, b = self._run(fx, out, bounds=tight)
        self.assertEqual((res["rc"], res["gate_tripped"], res["where"]), (G.RC_GATE, ["G3"], "running"))
        self.assertEqual(b.calls, 6)                                  # stopped right after the 6th call crossed 5
        self.assertFalse(G.cpu_ready(res)[0])

    def test_g3_trip_at_the_final_look(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, b = self._run(fx, out, bounds=dict(LENIENT, prefill_tokens=float(10)))
        self.assertEqual(res["gate_tripped"], ["G3"])

    def test_first_changed_document_over_twice_its_pilot_rate_is_a_budget_stop(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        with mock.patch.dict(G.AGG11_PILOT, {"MICROSOFT_2022_10K": {"llm_calls": 1, "prefill_tokens": 10 ** 9, "wall_s": 10 ** 9}}):
            rec, res, b = self._run(fx, out)
        self.assertEqual((res["rc"], res["budget_stop"]), (G.RC_BUDGET, "first_changed_doc_over_2x_pilot_rate"))
        self.assertEqual(res["over"], ["llm_calls"])
        self.assertFalse((out / "result.cpu.json").exists())          # not a gate: no verdict filed

    def test_smoke_runs_one_real_unit_on_the_first_changed_document(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec = P.Recorder(out / "results.jsonl")
        rec.emit("meta", phase="x")
        t0 = max(r["ts"] for r in rec.records)
        rec, res, b = self._run(fx, out, rec=rec, smoke=True)
        self.assertTrue(res["smoke"])
        new = [r for r in rec.records if r["kind"] == "l4_unit" and not r.get("reused") and r["ts"] > t0]
        self.assertEqual([(r["field"], r["doc"]) for r in new], [("revenue", "MICROSOFT_2022_10K")])
        self.assertGreaterEqual(new[0]["llm_calls"], 1)
        # the main run continues from the smoke (no unit repeated)
        b2 = FakeBackend()
        rec2, res2, _ = self._run(fx, out, b2, rec=P.Recorder(out / "results.jsonl"))
        self.assertEqual(res2["rc"], 0)
        units = [r["unit"] for r in rec2.records if r["kind"] == "l4_unit" and not r.get("reused")]
        self.assertEqual(len(units), len(set(units)))

    def test_resume_after_a_kill_between_the_store_commit_and_the_record_still_writes_l3_and_marks_replaced(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        # first entry: runs the whole job, then we erase the doc_ingest record + L3 facts of one slot to mimic the kill
        rec, res, _ = self._run(fx, out)
        slot = "3M_2022_10K"
        lines = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        keep = [r for r in lines if not (r.get("kind") in ("doc_ingest", "l3_doc") and r.get("doc") == slot)
                and r.get("kind") not in ("manifest", "gates_cpu")
                and not (r.get("kind") == "l4_unit" and r.get("doc") == slot)]
        (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
        s = CorpusStore(out / "store.sqlite")
        s.conn.execute("DELETE FROM facts WHERE doc_id = ? AND extractor LIKE 'l3%'", [slot])
        s.conn.execute("DELETE FROM facts WHERE doc_id = ? AND extractor = 'l4_llm.v1'", [slot])
        s.conn.commit()
        s.close()
        rec2, res2, _ = self._run(fx, out)
        rows = {r["doc"]: r for r in rec2.records if r["kind"] == "doc_ingest"}
        self.assertEqual((rows[slot]["status"], len(rows[slot]["superseded"])), ("replaced", 1))
        self.assertTrue(rows[slot]["l3_ran"])
        self.assertEqual(res2["rc"], 0, res2)

    def test_retained_hits_must_equal_the_agg11_hits(self):
        fx = make_fixture(self.tmp)
        lines = [json.loads(l) for l in fx["results11"].read_text().splitlines()]
        per = {n: {"gold": 1, "found": 1} for n in NAMES}
        lines.append({"kind": "m1", "ts": 1.0, "unit": "m1", "m1": 1.0, "per_doc": per})
        fx["results11"].write_text("".join(json.dumps(r) + "\n" for r in lines))
        out = self.tmp / "out"
        rec, res, _ = self._run(fx, out)
        self.assertTrue(res["g4"]["ok"] and res["g4"]["retained_equals_agg11"], res["g4"])
        # AGG-1.1 had one retained document found that is no longer found: G4 trips even though M1 is high
        fx2 = make_fixture(Path(tempfile.mkdtemp()))
        lines = [json.loads(l) for l in fx2["results11"].read_text().splitlines()]
        per = {n: {"gold": 1, "found": 1} for n in NAMES}
        per["CORNING_2022_10K"]["found"] = 2   # AGG-1.1 reported a different count for a retained document
        lines.append({"kind": "m1", "ts": 1.0, "unit": "m1", "m1": 1.0, "per_doc": per})
        fx2["results11"].write_text("".join(json.dumps(r) + "\n" for r in lines))
        rec2, res2, _ = self._run(fx2, Path(tempfile.mkdtemp()))
        self.assertEqual((res2["gate_tripped"], res2["g4"]["retained_equals_agg11"]), (["G4"], False))

    def test_old_facts_leaked_into_the_live_tables_are_caught_by_the_supersession_check(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, _ = self._run(fx, out)
        self.assertTrue(res["g2"]["ok"])
        s = CorpusStore(out / "store.sqlite")
        base = next(r for r in rec.records if r.get("unit") == "base_snapshot")
        new_hashes = {r["doc"]: r["doc_hash"] for r in rec.records if r["kind"] == "doc_ingest"}
        old = base["old_hashes"]["3M_2022_10K"]
        led = s.superseded_facts(P.CORPUS_ID, doc_hash=old)[0]
        leak = {k: led[k] for k in led if k != "superseded_by"}
        s.write_facts([leak])                                          # a superseded fact back in the live table
        chk = G.superseded_check(s, P.CORPUS_ID, base, new_hashes)
        self.assertFalse(chk["ok"])
        self.assertTrue(any("old hash still live" in p or "superseded fact_ids" in p for p in chk["problems"]))

    def test_sql_absence_flags_a_row_that_equals_a_superseded_fact(self):
        fx = make_fixture(self.tmp)
        out = self.tmp / "out"
        rec, res, _ = self._run(fx, out)
        s = CorpusStore(out / "store.sqlite")
        s.refresh_canonical(P.CORPUS_ID)
        good = "SELECT doc_id, field, period, value, quote FROM facts_canonical WHERE field = 'revenue'"
        self.assertTrue(G.sql_absence(s, P.CORPUS_ID, [good])["ok"])
        sup = s.superseded_facts(P.CORPUS_ID)[0]
        lit = lambda v: "NULL" if v is None else (repr(v) if not isinstance(v, str) else "'" + v.replace("'", "''") + "'")  # noqa: E731
        bad = (f"SELECT {lit(sup['doc_id'])} AS doc_id, {lit(sup['field'])} AS field, {lit(sup['period'])} AS period, "
               f"{lit(sup['value'])} AS value, {lit(sup['quote'])} AS quote")
        chk = G.sql_absence(s, P.CORPUS_ID, [bad])
        self.assertFalse(chk["ok"])
        self.assertEqual(len(chk["superseded_hits"]), 1)


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with superseded_facts not importable")
class VerdictTests(unittest.TestCase):
    def _ans(self, q, calls=2, prefill=600):
        return {"kind": "answer", "arm": "P-large", "qid": q, "calls": calls, "prefill_tokens": prefill, "final": True,
                "status": "ok", "correct": None, "wall_s": 6.0}

    def test_verdict_is_done_only_with_cpu_gates_and_m4(self):
        agg = [{"id": f"q{i}", "evidence": [{"doc": "3M_2022_10K"}]} for i in range(4)]
        cpu = {g: {"ok": True} for g in ("g1", "g2", "g3", "g4")}
        v = G.verdict13(cpu, [self._ans(f"q{i}") for i in range(4)], agg)
        self.assertEqual((v["verdict"], v["gates_tripped"], v["scored"], v["n_questions_touching_replaced"]),
                         ("agg_1_3_done", [], False, 4))
        v = G.verdict13(cpu, [self._ans(f"q{i}", calls=4) for i in range(4)], agg)
        self.assertEqual((v["verdict"], v["gates_tripped"]), ("phase1_gate_tripped", ["G5"]))
        self.assertEqual(G.verdict13(cpu, [self._ans("q0")], agg)["verdict"], "incomplete")
        bad = dict(cpu, g3={"ok": False})
        self.assertIn("G3", G.verdict13(bad, [self._ans(f"q{i}") for i in range(4)], agg)["gates_tripped"])


# --------------------------------------------------------------------------------------------- CLI
@unittest.skipUnless(HAVE_ORCH and Path(G.DEFAULT_AGG11_JOB / "SOURCE.sha256").exists()
                     and (G.DEFAULT_AGG11_JOB / "out" / "store.sqlite").exists(), "AGG-1.1 job dir not on this host")
class CliDryRunTests(unittest.TestCase):
    def test_dry_run_on_real_inputs_is_clean_and_writes_nothing(self):
        out = Path(tempfile.mkdtemp()) / "must-not-exist"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--phase", "reingest", "--dry-run", "--orch-root", str(ORCH), "--out", str(out)])
        d = json.loads(buf.getvalue())
        self.assertEqual((rc, d["problems"]), (0, []))
        self.assertEqual((d["model_calls"], d["servers_started"], d["wrote_files"]), (0, 0, 0))
        self.assertFalse(out.exists())
        self.assertEqual((d["plan"]["retained_skipped_by_hash_expected"], d["plan"]["changed_reparsed_reextracted_expected"]), (25, 3))
        self.assertEqual(d["checks"]["base_hash_check"]["retained_equal"], 25)
        self.assertTrue(d["checks"]["l4_reuse"]["reuse_ok"])
        self.assertEqual(d["checks"]["m1_cells"], {"total": 211, "retained": 187, "changed": 24})

    def test_dry_run_does_not_mutate_the_agg11_store(self):
        st = G.DEFAULT_AGG11_JOB / "out" / "store.sqlite"
        before = (st.stat().st_mtime_ns, st.stat().st_size, sha(st))
        with contextlib.redirect_stdout(io.StringIO()):
            R.main(["--phase", "reingest", "--dry-run", "--orch-root", str(ORCH)])
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
            G.seed_units(rec, src, {"docs": [{"doc_name": "AES_2022_10K"}]})
            got = [r for r in rec.records if r["kind"] == "l4_unit"]
            self.assertEqual(len(got), 1)
            self.assertIs(got[0]["reused"], True)
            self.assertEqual(got[0]["reused_from"], str(src))
            self.assertEqual(got[0]["unit"], self.REAL_AGG11_RECORD["unit"])


if __name__ == "__main__":
    unittest.main()
