"""AGG-1.13 step B runner tests (fake backends: no model, no network, no process). Stdlib tests always run; the store / ingest tests import the
orchestrator package and skip without it (``AGG_ORCH_ROOT``, default the AGG-1.13 pinned tree: 737bf0535 plus the amendment-1 ``fy_end`` commit 8ac97157)."""

from __future__ import annotations

import hashlib
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
import agg113 as H  # noqa: E402
import agg112 as G  # noqa: E402
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_agg113 as C  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/worktrees/agg113-orch"))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import facts_det, facts_typed as ft, pipeline as pl  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section, TableRef  # noqa: E402
    HAVE_ORCH = hasattr(pl, "pipeline_spec")
except Exception:  # pragma: no cover
    HAVE_ORCH = False

needs_orch = unittest.skipUnless(HAVE_ORCH, "orchestrator tree not available")
INP = H.load_inputs(H.D113)
JAN = H.STRATA["january_end_filers"]
FL = H.STRATA["unlabelled_subtotal_filer"][0]            # the corpus document names are spelled only in agg113/design113.py (rule 4 scan)
ULTA, WMT = H.PILOT_DOCS[1], H.PILOT_DOCS[0]
BBY = next(d for d in JAN if d not in (FL, ULTA, WMT))


class PinsAndInputs(unittest.TestCase):
    def test_pins_match_the_committed_files(self):
        self.assertEqual(H.pin_problems(H.D113), [])

    def test_the_amendment_sha_is_pinned_and_a_changed_amendment_is_refused(self):
        self.assertEqual(H.SHA256(H.D113 / H.AMENDMENT_FILE), H.AMENDMENT_SHA)
        self.assertTrue(H.pin_problems(H.D113, amendment_sha="0" * 64))
        self.assertEqual(H.SHA256(H.D113 / H.PREDECLARED_FILE), H.PREDECLARED_SHA)          # the parent is untouched
        self.assertEqual(H.ORCH_PIN_MIN, "8ac97157557ca45a4b86c8c9d2874c4b13977b6b")

    def test_a_changed_input_or_predeclared_is_refused(self):
        self.assertTrue(H.pin_problems(H.D113, predeclared_sha="0" * 64))
        self.assertTrue(H.pin_problems(H.D113, frozen={"golds.jsonl": "0" * 64}))

    def test_corpus_shape(self):
        self.assertEqual(H.corpus_problems(INP, check_pdfs=False), [])
        self.assertEqual((len(INP["corpus"]["docs"]), len(INP["cells"]), len(X.verified_agg(INP))), (10, 76, 30))
        self.assertEqual(sum(1 for c in INP["cells"] if c[0] in H.PILOT_DOCS), 22)

    def test_doc_order_pilot_first_then_ascending_pdf_size(self):
        order = H.doc_order(INP["corpus"])
        self.assertEqual(order[:3], list(H.PILOT_DOCS))
        sizes = {d["doc_name"]: d["bytes"] for d in INP["corpus"]["docs"]}
        rest = order[3:]
        self.assertEqual(rest, sorted(rest, key=lambda n: (sizes[n], n)))
        self.assertEqual(len(order), 10)

    def test_constants_equal_the_predeclared_text(self):
        text = (H.D113 / "PREDECLARED_AGG113.md").read_text()
        self.assertIn(H.PREDECLARED_SHA, (H.D113 / "PREDECLARED_AGG113.sha256").read_text())
        self.assertEqual((H.PILOT_WALL_S, H.INTERIM_DOCS, H.L3_FUTILITY_FAILS), (1800, 6, 2))
        self.assertIn("1,800 s", text)
        self.assertIn("after 6 of the 10 documents", text)
        self.assertEqual({k: v for k, v in H.ARMS.items()}, {"A": ("v1", "agg11.v0"), "B": ("v2", "agg11.v0"), "C": ("v2", "agg110.v1")})
        self.assertEqual(list(H.ARMS), ["A", "B", "C"])                         # the run order is fixed
        self.assertEqual((X.M4_MEDIAN_CALLS_SOFT, X.M4_MEDIAN_PREFILL_MAX), (3, 2510))

    def test_frozen_scorer_and_metrics_are_untouched(self):
        # the runner imports, never redefines, the scorer and the M1 reading
        self.assertIs(H.P.score_agg, P.score_agg)
        self.assertIs(H.X.m1_hits, X.m1_hits)
        for name in ("score_agg", "m1_hits"):
            self.assertFalse(hasattr(H, name))

    def test_strata_cover_declared_documents(self):
        names = {d["doc_name"] for d in INP["corpus"]["docs"]}
        for docs in H.STRATA.values():
            self.assertTrue(set(docs) <= names)
        self.assertEqual(set(H.january_end_docs(INP["corpus"])), set(H.STRATA["january_end_filers"]))


class PairedStatistic(unittest.TestCase):
    def test_exact_cases_from_the_predeclared_text(self):
        def run(b, c, n):
            keys = [f"k{i}" for i in range(n)]
            first = {k: True for k in keys[:b]}
            second = {k: True for k in keys[b:b + c]}
            return H.paired_113(first, second, keys)
        r = run(7, 0, 76)                       # "b = 7, c = 0: z 2.65"
        self.assertAlmostEqual(r["z"], 2.6458, places=3)
        self.assertTrue(r["meets_planned_z"])
        self.assertEqual(r["reading"], "v2_better")
        r = run(6, 0, 76)                       # "b = 6, c = 0: 2.45"
        self.assertAlmostEqual(r["z"], 2.449, places=3)
        self.assertFalse(r["meets_planned_z"])
        self.assertEqual(run(6, 1, 30)["reading"], "inconclusive")   # "b = 6, c = 1 (z 1.9) ... indistinguishable from noise"
        self.assertAlmostEqual(run(6, 1, 30)["z"], 1.89, places=2)
        self.assertEqual(run(4, 0, 30)["reading"], "v2_better")      # "the smallest effect that clears z = 2 is b = 4 with c = 0"
        self.assertEqual(run(0, 5, 30)["reading"], "v2_worse")

    def test_undefined_when_no_discordant_pairs(self):
        r = H.paired_113({"a": True}, {"a": True}, ["a", "b"])
        self.assertIsNone(r["z"])
        self.assertIsNone(r["p"])
        self.assertEqual(r["reading"], "inconclusive")
        self.assertIn("undefined", r["note"])

    def test_se_and_interval_and_exact_p(self):
        keys = [f"k{i}" for i in range(76)]
        first = {k: True for k in keys[:8]}
        r = H.paired_113(first, {}, keys)
        self.assertAlmostEqual(r["d"], 8 / 76)
        self.assertAlmostEqual(r["se"], (8 - 64 / 76) ** 0.5 / 76)
        self.assertAlmostEqual(r["p"], 2 * 0.5 ** 8)                         # exact two-sided binomial, b = 8, c = 0
        self.assertAlmostEqual(H.binom_two_sided(3, 3), 1.0)
        self.assertIsNone(H.binom_two_sided(0, 0))

    def test_cross_futility_is_strict(self):
        keys = ["a", "b", "c"]
        self.assertTrue(H.cross_futility({"a": True}, {"a": True, "b": True}, keys)["tripped"])
        self.assertFalse(H.cross_futility({"a": True, "b": True}, {"a": True, "b": True}, keys)["tripped"])
        self.assertFalse(H.cross_futility({"a": True, "b": True}, {"a": True}, keys)["tripped"])

    def test_cpu_ready(self):
        ok = {"complete": True, "phase": "cpu", "pipeline": "agg_pipeline.v2", "futility_stop": None, "gate_tripped": [], "m3_passed": True}
        self.assertTrue(H.cpu_ready(ok, "v2")[0])
        self.assertFalse(H.cpu_ready(ok, "v1")[0])                           # wrong pipeline for the slot
        self.assertFalse(H.cpu_ready({**ok, "futility_stop": "M3"}, "v2")[0])
        self.assertFalse(H.cpu_ready({**ok, "m3_passed": False}, "v2")[0])
        self.assertFalse(H.cpu_ready({**ok, "complete": False}, "v2")[0])


class ScoreMechanics(unittest.TestCase):
    def test_score_files_agg_1_13_with_every_reading(self):
        cells = INP["cells"]
        keys = H.cell_keys(cells)
        r1 = {k: True for k in keys[:60]}
        r2 = {k: True for k in keys[:70]}
        cu = {"complete": True, "m3_passed": True, "m3": {"prefill_ratio": 0.3}}
        agg = X.verified_agg(INP)
        recs = []
        for arm, n_ok in (("A", 8), ("B", 12), ("C", 20)):
            for i, q in enumerate(agg):
                recs.append({"kind": "answer", "final": True, "arm": arm, "qid": q["id"], "correct": i < n_ok, "calls": 2, "prefill_tokens": 500})
        out = H.score(cu, cu, {"raw": r1, "canonical": r1}, {"raw": r2, "canonical": r2}, recs, agg, cells,
                      checkpoints={"pilot": {"docs": list(H.PILOT_DOCS)}})
        self.assertEqual(out["verdict"], "agg_1_13_filed")
        self.assertEqual(out["gates_tripped"], [])
        self.assertEqual(out["m1"]["raw"]["paired_v2_vs_v1"]["b_only_first"], 10)
        self.assertEqual(out["m1"]["raw"]["paired_v2_vs_v1"]["c_only_second"], 0)
        self.assertEqual(out["p_large"]["total_C_vs_A"]["b_only_first"], 12)
        self.assertEqual(out["p_large"]["pipeline_B_vs_A"]["b_only_first"], 4)
        self.assertEqual(out["p_large"]["planner_C_vs_B"]["b_only_first"], 8)
        self.assertIsNone(out["adopt_recommendation"])                       # never computed from these data alone
        self.assertEqual(len(out["m1"]["regressions_v1_found_v2_not"]), 0)
        self.assertEqual(len(out["m1"]["fixes_v2_found_v1_not"]), 10)
        self.assertTrue(out["p_large"]["complete"])

    def test_futility_and_gate_labels(self):
        cells = INP["cells"]
        keys = H.cell_keys(cells)
        pil = H.cell_keys(cells, H.PILOT_DOCS)
        r1 = {k: True for k in pil}
        r2 = {k: True for k in pil[:-1]}                                      # v2 one below v1 on the pilot cells
        cu = {"complete": True, "m3_passed": True, "m3": {}}
        out = H.score(cu, {**cu, "m3_passed": False}, {"raw": r1, "canonical": r1}, {"raw": r2, "canonical": r2}, [], X.verified_agg(INP), cells,
                      checkpoints={"pilot": {"docs": list(H.PILOT_DOCS)}})
        self.assertIn("agg113_futility", out["also"])
        self.assertIn("phase1_gate_tripped", out["also"])
        self.assertIn("M3:v2", out["gates_tripped"])
        self.assertFalse(out["complete"])
        self.assertEqual(out["verdict"], "agg_1_13_filed")

    def test_m4_trip(self):
        agg = X.verified_agg(INP)
        recs = [{"kind": "answer", "final": True, "arm": "A", "qid": q["id"], "calls": 5, "prefill_tokens": 100} for q in agg]
        m = H.m4_arm(recs, "A", [q["id"] for q in agg])
        self.assertTrue(m["trips"] and m["complete"])
        m = H.m4_arm(recs[:3], "A", [q["id"] for q in agg])
        self.assertFalse(m["trips"])                                          # incomplete: never tripped on a partial set


@needs_orch
class FiscalYearContract(unittest.TestCase):
    """The preflight hard check. THE FINDING on the real PREDECLARED corpus: Best Buy and Walmart pass; Ulta (all 7 filled cells) and Foot Locker's
    three balance-sheet cells carry the CALENDAR year 2023 in their printed first-column header, so both L3 versions label the fiscal-2022 column
    '2023' and the frozen ``m1_hits`` (period must contain '2022') cannot see it. This test pins that fact; it fails the day the contract is amended."""

    def test_the_real_corpus_fails_the_contract_for_foot_locker_and_ulta_only(self):
        fc = H.fy_contract(INP["corpus"], INP["cells"])
        self.assertFalse(fc["ok"])
        self.assertEqual({p.split("/")[0] for p in fc["problems"]}, {FL, ULTA})
        ulta = [p for p in fc["problems"] if p.startswith(ULTA)]
        fl = [p for p in fc["problems"] if p.startswith(FL)]
        self.assertEqual(len(ulta), 14)                                        # 7 cells x 2 L3 versions
        self.assertEqual(len(fl), 6)                                           # 3 balance-sheet cells x 2 L3 versions
        self.assertTrue(all("balance_sheet" in p for p in fl))
        self.assertEqual(fc["n_cells_checked"], 31)
        for doc in (BBY, WMT):
            for cell in fc["by_doc"][doc]["cells"].values():
                self.assertTrue(cell["v1_legacy"]["m1_hits_sees_it"] and cell["v2"]["m1_hits_sees_it"])

    def test_a_fiscal_label_header_passes(self):
        corpus = {"docs": [{"doc_name": WMT, "fiscal_year_end": "2022-01-31"}]}
        cells = [(WMT, f, 1.0) for f in ("revenue", "total_assets", "capex")]
        fc = H.fy_contract(corpus, cells)
        self.assertTrue(fc["ok"], fc["problems"])
        self.assertEqual(fc["n_cells_checked"], 3)

    def test_a_calendar_date_header_of_the_next_year_fails_and_a_missing_fixture_fails(self):
        corpus = {"docs": [{"doc_name": ULTA, "fiscal_year_end": "2023-01-28"},
                           {"doc_name": "NEWCO_2023_10K", "fiscal_year_end": "2023-01-28"}]}
        cells = [(ULTA, "revenue", 1.0), ("NEWCO_2023_10K", "revenue", 1.0)]
        fc = H.fy_contract(corpus, cells)
        self.assertFalse(fc["ok"])
        self.assertTrue(any("no declared printed-header fixture" in p for p in fc["problems"]))
        self.assertTrue(any("'2023'" in p for p in fc["problems"]))

    def test_with_the_amendment_flag_the_real_corpus_passes_for_all_31_cells_in_both_versions(self):
        fc = H.fy_contract(INP["corpus"], INP["cells"], fy_end=True)
        self.assertTrue(fc["ok"], fc["problems"])
        self.assertEqual((fc["fy_end"], fc["n_cells_checked"]), (True, 31))
        for doc, rec in fc["by_doc"].items():
            for cell in rec["cells"].values():
                self.assertTrue(cell["v1_legacy"]["m1_hits_sees_it"] and cell["v2"]["m1_hits_sees_it"], (doc, cell))
        for doc in (ULTA, FL):                                                  # the cells the original contract lost now carry fiscal 2022
            self.assertTrue(all(c["v2"]["period"] == "2022" and c["v1_legacy"]["period"] == "2022" for c in fc["by_doc"][doc]["cells"].values()))

    def test_the_default_without_the_flag_is_still_the_original_failing_check(self):
        self.assertEqual(H.fy_contract(INP["corpus"], INP["cells"], fy_end=False)["problems"], H.fy_contract(INP["corpus"], INP["cells"])["problems"])

    def test_a_next_calendar_year_header_passes_only_through_the_rule_and_best_buy_is_not_shifted(self):
        corpus = {"docs": [{"doc_name": ULTA, "fiscal_year_end": "2023-01-28"}, {"doc_name": BBY, "fiscal_year_end": "2022-01-29"}]}
        cells = [(ULTA, "revenue", 1.0), (BBY, "revenue", 1.0)]
        self.assertFalse(H.fy_contract(corpus, cells)["ok"])
        fc = H.fy_contract(corpus, cells, fy_end=True)
        self.assertTrue(fc["ok"], fc["problems"])
        self.assertEqual(fc["by_doc"][BBY]["cells"]["revenue"]["v2"]["period"], "2022")

    def test_the_preflight_cli_exits_0_with_the_amendment_flag_and_asserts_its_sha(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.main(["--phase", "preflight", "--dry-run", "--fy-end", "--orch-root", str(ORCH), "--orch-pin", H.ORCH_PIN_MIN])
        out = json.loads(buf.getvalue())
        self.assertEqual((rc, out["problems"]), (0, []))
        self.assertEqual((out["fy_contract"]["ok"], out["fy_contract"]["n_cells_checked"], out["fy_end"]), (True, 31, True))
        self.assertEqual(out["amendment1_sha256"], H.AMENDMENT_SHA)

    def test_the_preflight_cli_exits_2_on_the_failed_contract(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.main(["--phase", "preflight", "--dry-run", "--orch-root", str(ORCH), "--orch-pin", H.ORCH_PIN_MIN])
        self.assertEqual(rc, H.RC_PREFLIGHT)
        out = json.loads(buf.getvalue())
        self.assertEqual((out["model_calls"], out["servers_started"], out["wrote_files"]), (0, 0, 0))
        self.assertFalse(out["ok"])
        self.assertTrue(all(p.startswith("fiscal-year contract") for p in out["problems"]), out["problems"][:3])


class ScopedRecTests(unittest.TestCase):
    def test_two_scopes_do_not_see_each_other(self):
        rec = P.Recorder(Path(tempfile.mkdtemp()) / "r.jsonl")
        a, b = H.ScopedRec(rec, "v1"), H.ScopedRec(rec, "v2")
        a.emit("audit", unit="audit:f1", status="ok")
        self.assertIn("audit:f1", a.done)
        self.assertNotIn("audit:f1", b.done)
        b.emit("audit_done", unit="audit_done")
        self.assertEqual([r["kind"] for r in a.records], ["audit"])
        self.assertEqual([r["kind"] for r in b.records], ["audit_done"])
        self.assertEqual(rec.records[0]["unit"], "v1:audit:f1")


@needs_orch
class Stage(unittest.TestCase):
    """A two-document corpus end to end: shared L3 (fake parse), then each arm through run_ingest with fake backends."""

    DOCS = ("A_2022_10K", "B_2022_10K")
    CID = H.CORPUS_ID

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.vals = {"A_2022_10K": {"revenue": 100.0, "total_assets": 27165.0}, "B_2022_10K": {"revenue": 250.0, "total_assets": 9000.0}}
        self.long = "\n".join(f"filler line {i} about nothing in particular" for i in range(30))
        self.corpus = {"fiscal_year": 2022, "pilot_docs": [self.DOCS[0]], "docs": []}
        for d in self.DOCS:
            pdf = self.tmp / f"{d}.pdf"
            pdf.write_bytes(d.encode() * 50)
            self.corpus["docs"].append({"doc_name": d, "company": d.split("_")[0], "pdf": str(pdf), "bytes": pdf.stat().st_size,
                                        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "fiscal_year_end": "2022-12-31"})
        cells = [(d, f, self.vals[d][f]) for d in self.DOCS for f in ("revenue", "total_assets")]
        self.inp = {"corpus": self.corpus, "cells": cells, "sheet": None, "set_dir": H.D113}
        self.guard = lambda: None
        self.patches = [mock.patch.object(H, "PILOT_DOCS", (self.DOCS[0],)), mock.patch.object(H, "INTERIM_DOCS", 99)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def parse(self, pdf: Path):
        d = pdf.stem
        v = self.vals[d]
        secs = [Section(id="s1", title="Income", level=1, content=f"(in millions)\nTotal revenues {int(v['revenue'])} 90\n" + self.long, page_start=1, page_end=1),
                Section(id="s2", title="Balance", level=1, content=f"(in millions)\nTotal assets {int(v['total_assets']):,} 1\n" + self.long, page_start=2, page_end=2)]
        return DocumentPreprocessResult(original_path=str(pdf), sections=secs, figures=[], total_pages=2, tables=[])

    def make_seed(self):
        out = self.tmp / "l3"
        rec = P.Recorder(out / "results.jsonl")
        r = H.run_l3(rec, out, self.inp, orch_root=ORCH, guard=self.guard, parse_fn=self.parse)
        return out, rec, r

    def be_v2(self, bad_for=()):
        vals = self.vals

        class B:
            calls = 0
            prefill_tokens = 0

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
                return {"text": json.dumps(body), "prompt_tokens": 5, "cache_n": 0, "decode_tokens": 10, "finish_reason": "stop"}

        B.calls = 0
        return B()

    def be_v1(self):
        class B:
            calls = 0
            prefill_tokens = 0

            def _bill(self, prompt):
                B.calls += 1
                B.prefill_tokens += len(prompt) // 4

            def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
                return self.chat(prompt, n_predict=n_predict, grammar=grammar)

            def chat(self, prompt, *, n_predict, grammar=None, cache_prompt=True, id_slot=None, logprobs=None):
                self._bill(prompt)
                if grammar and grammar.startswith('root ::= "[') or (grammar and "[" in grammar[:20]):
                    m = re.search(r"Total revenues [\d,]+", prompt.split("Excerpt:\n", 1)[-1])
                    rows = [] if not m or "Field: revenue" not in prompt else [{
                        "period": "FY2022", "value": float(m.group(0).split()[-1].replace(",", "")), "unit": "USD_millions", "quote": m.group(0),
                        "confidence": "medium"}]
                    return {"text": json.dumps(rows), "prompt_tokens": len(prompt) // 4, "cache_n": 0, "finish_reason": "stop"}
                return {"text": "yes", "prompt_tokens": len(prompt) // 4, "cache_n": 0, "finish_reason": "stop"}

        B.calls = 0
        return B()

    def ingest(self, out, be, pipeline, seed, **kw):
        rec = P.Recorder(out / "results.jsonl")
        r = H.run_ingest(rec, out, self.inp, be, pipeline=pipeline, guard=self.guard, seed_store=seed, **kw)
        return rec, r

    def test_l3_builds_a_fresh_store_with_no_model_and_records_the_fy_check(self):
        out, rec, r = self.make_seed()
        self.assertEqual((r["complete"], r["rc"], r["failed"]), (True, 0, []))
        self.assertEqual(r["model_calls"], 0)
        self.assertEqual(r["m1_l3_n"], 4)
        corpus_rec = next(x for x in rec.records if x.get("unit") == "corpus")
        self.assertEqual((corpus_rec["n_docs"], corpus_rec["llm_calls"], corpus_rec["prefill_tokens"]), (2, 0, 0))
        self.assertTrue((out / "store.sqlite").exists())

    def _ulta_like_parse(self, pdf: Path):
        hdr = ["January 28, 2023", "January 29, 2022"]
        secs = [Section(id="s1", title="Consolidated Balance Sheets", level=1, content="(in millions)\n" + "  ".join(hdr) + "\nTotal assets 27,165 1\n" + self.long,
                        page_start=1, page_end=1)]
        tb = TableRef(id="t1", page=1, caption="(in millions)", section_id="s1", rows=[[""] + hdr, ["Total assets", "27,165", "1"]])
        return DocumentPreprocessResult(original_path=str(pdf), sections=secs, figures=[], total_pages=2, tables=[tb])

    def _l3_periods(self, fy_end):
        out = self.tmp / f"l3fy{int(fy_end)}"
        rec = P.Recorder(out / "results.jsonl")
        r = H.run_l3(rec, out, self.inp, orch_root=ORCH, guard=self.guard, parse_fn=self._ulta_like_parse, fy_end=fy_end)
        self.assertEqual((r["complete"], r["rc"], r["fy_end"], r["amendment_sha256"]), (True, 0, fy_end, H.AMENDMENT_SHA))
        (out / "result.l3.json").write_text(json.dumps({k: v for k, v in r.items() if k != "rc"}))      # what run_agg113.main files next to the store
        st = P.open_store(out / "store.sqlite")
        try:
            rows = [(f["extractor"], f["period"], f["value"]) for f in st.facts(self.CID) if f["field"] == "total_assets"]
        finally:
            st.close()
        return out, rows

    def test_l3_fy_end_labels_both_paths_once_and_the_default_keeps_the_defect(self):
        _o, off = self._l3_periods(False)
        self.assertIn("2023", {p for _e, p, _v in off})
        _o, on = self._l3_periods(True)
        self.assertEqual({p for _e, p, _v in on} - {None}, {"2022", "2021"})              # no 2023 left, on every extractor the store has
        self.assertIn("l3_table.v1", {e for e, _p, _v in on})
        self.assertEqual(len(on), len(set(on)))                                             # no fact twice under two labels (both L3 paths of run_l3)
        self.assertEqual(sorted(p for e, p, v in on if e == "l3_table.v1" and v == 27165.0), ["2022"])

    def test_the_seed_must_carry_the_rule_and_an_arm_refuses_one_that_does_not(self):
        out, _on = self._l3_periods(True)
        self.assertIsNone(H.seed_fy_end_problem(out / "store.sqlite"))
        out0, _off = self._l3_periods(False)
        self.assertTrue(H.seed_fy_end_problem(out0 / "store.sqlite"))
        self.assertTrue(H.seed_fy_end_problem(self.tmp / "nowhere" / "store.sqlite"))
        rec = P.Recorder(self.tmp / "arm" / "results.jsonl")
        r = H.run_ingest(rec, self.tmp / "arm", self.inp, None, pipeline=H.V1, guard=self.guard, seed_store=out0 / "store.sqlite", fy_end=True)
        self.assertEqual(r["rc"], H.RC_PRECOND)
        self.assertEqual(rec.records[-1]["kind"], "precondition_failed")

    def test_v2_arm_l3_applies_the_same_rule_to_its_copy_and_leaves_the_seed_alone(self):
        out, _on = self._l3_periods(True)
        copy = self.tmp / "arm2.sqlite"
        P.copy_store(out / "store.sqlite", copy)
        st = P.open_store(copy)
        try:
            rec = P.Recorder(self.tmp / "arm2" / "results.jsonl")
            H.arm_l3(rec, st, self.inp, H.V2, fy_end=True)
            got = {f["period"] for f in st.facts(self.CID) if f["field"] == "total_assets"}
            self.assertTrue({"2022"} <= got and "2023" not in got, got)
            self.assertTrue(rec.records[-1]["fy_end"])
        finally:
            st.close()

    def test_l3_two_parse_failures_stop_with_rc_6_and_one_is_a_recorded_miss(self):
        out = self.tmp / "l3x"

        def boom(pdf):
            raise RuntimeError("jvm")
        rec = P.Recorder(out / "results.jsonl")
        r = H.run_l3(rec, out, self.inp, orch_root=ORCH, guard=self.guard, parse_fn=boom)
        self.assertEqual((r["rc"], r["infrastructure_stop"]), (H.RC_L3, "l3_parse_futility"))
        out2 = self.tmp / "l3y"
        calls = {"n": 0}

        def one_boom(pdf):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("jvm")
            return self.parse(pdf)
        rec = P.Recorder(out2 / "results.jsonl")
        r = H.run_l3(rec, out2, self.inp, orch_root=ORCH, guard=self.guard, parse_fn=one_boom)
        self.assertEqual((r["complete"], len(r["failed"])), (True, 1))

    def test_l3_smoke_parses_the_smallest_document_only_and_resumes_without_reparsing(self):
        out = self.tmp / "l3s"
        rec = P.Recorder(out / "results.jsonl")
        seen = []
        H.run_l3(rec, out, self.inp, orch_root=ORCH, guard=self.guard, parse_fn=lambda p: (seen.append(p.name), self.parse(p))[1], smoke=True)
        self.assertEqual(len(seen), 1)
        rec = P.Recorder(out / "results.jsonl")
        H.run_l3(rec, out, self.inp, orch_root=ORCH, guard=self.guard, parse_fn=lambda p: (seen.append(p.name), self.parse(p))[1])
        self.assertEqual(len(seen), 2)                                          # the smoke document is a doc_done: not parsed again

    def test_v2_arm_end_to_end_and_the_seed_is_never_mutated(self):
        seed_dir, _r, _x = self.make_seed()
        seed = seed_dir / "store.sqlite"
        sha = hashlib.sha256(seed.read_bytes()).hexdigest()
        out = self.tmp / "v2"
        be = self.be_v2()
        rec, r = self.ingest(out, be, "v2", seed)
        self.assertEqual((r["complete"], r["rc"]), (True, 0), r)
        self.assertEqual(hashlib.sha256(seed.read_bytes()).hexdigest(), sha)
        res = json.loads((out / "result.cpu.json").read_text())
        self.assertEqual((res["phase"], res["pipeline"], res["m1_found"], res["m1_n"], res["m3_passed"]), ("cpu", "agg_pipeline.v2", 4, 4, True))
        self.assertEqual(res["canonical"]["guards"], ["attributable_net_income"])
        self.assertEqual(res["m3"]["llm_calls"], be.calls)
        cells = json.loads((out / "cells.json").read_text())
        self.assertEqual(sum(cells["raw"].values()), 4)
        self.assertTrue((out / "checkpoint.pilot.json").exists())
        windows = [x for x in rec.records if x.get("kind") == "t_window"]
        self.assertEqual(len({x["unit"] for x in windows}), len(windows))
        self.assertTrue(H.cpu_ready(res, "v2")[0])

    def test_v1_arm_end_to_end_runs_the_amended_row_stage_one_unit_per_field_and_document(self):
        seed_dir, _r, _x = self.make_seed()
        out = self.tmp / "v1"
        be = self.be_v1()
        with mock.patch.object(X, "m3_trips", lambda m3: False):                   # a 4-chunk fixture is far over the calls-per-100-chunks bound
            rec, r = self.ingest(out, be, "v1", seed_dir / "store.sqlite")
        self.assertEqual((r["complete"], r["rc"]), (True, 0), r)
        units = [x for x in rec.records if x.get("kind") == "l4_unit"]
        self.assertEqual(len(units), 16)                                       # 2 documents x 8 fields
        self.assertEqual(len({x["unit"] for x in units}), 16)
        res = json.loads((out / "result.cpu.json").read_text())
        self.assertEqual((res["pipeline"], res["m3_passed"], res["m1_n"]), ("agg_pipeline.v1", True, 4))
        self.assertEqual(res["m3"]["llm_calls"], sum(x["llm_calls"] for x in units))
        self.assertTrue(any(x.get("unit") == "l3v1" for x in rec.records))      # v1 keeps the seed's legacy L3
        self.assertTrue(res["canonical"]["rule"].startswith("canonical.v1"))

    def test_a_projected_m3_over_a_kill_bound_stops_at_the_pilot(self):
        seed_dir, _r, _x = self.make_seed()
        out = self.tmp / "m3"
        rec, r = self.ingest(out, self.be_v1(), "v1", seed_dir / "store.sqlite")      # the tiny corpus projects over the bound
        self.assertEqual(r["rc"], H.RC_GATE)
        res = json.loads((out / "result.cpu.json").read_text())
        self.assertEqual((res["verdict"], res["m3_passed"], res["gate_tripped"]), ("phase1_gate_tripped", False, ["M3"]))

    def test_resume_repeats_no_call(self):
        seed_dir, _r, _x = self.make_seed()
        out = self.tmp / "v2r"
        be = self.be_v2()
        self.ingest(out, be, "v2", seed_dir / "store.sqlite")
        n = be.calls
        be2 = self.be_v2()
        rec, r = self.ingest(out, be2, "v2", seed_dir / "store.sqlite")
        self.assertEqual(be2.calls, 0)
        self.assertEqual((n > 0, r["complete"]), (True, True))

    def test_smoke_makes_exactly_one_real_call_on_the_smallest_document(self):
        seed_dir, _r, _x = self.make_seed()
        for pipe, be in (("v2", self.be_v2()), ("v1", self.be_v1())):
            out = self.tmp / f"smoke-{pipe}"
            rec, r = self.ingest(out, be, pipe, seed_dir / "store.sqlite", smoke=True)
            self.assertTrue(r.get("smoke"))
            self.assertEqual(be.calls >= 1, True)
            fresh = [x for x in rec.records if x.get("kind") in ("t_window", "l4_unit")]
            self.assertEqual(len(fresh), 1)

    def test_no_seed_is_a_precondition_not_a_crash(self):
        out = self.tmp / "noseed"
        rec, r = self.ingest(out, self.be_v2(), "v2", self.tmp / "missing.sqlite")
        self.assertEqual((r["complete"], r["rc"]), (False, H.RC_PRECOND))

    def test_a_pilot_wall_over_the_bound_is_a_budget_stop_not_a_gate(self):
        seed_dir, _r, _x = self.make_seed()
        out = self.tmp / "budget"
        be = self.be_v2()
        rec, r = self.ingest(out, be, "v2", seed_dir / "store.sqlite", pilot_wall_s=-1.0)
        self.assertEqual(r["rc"], H.RC_BUDGET)
        self.assertFalse((out / "result.cpu.json").exists())

    def test_v2_pilot_below_the_peer_v1_is_agg113_futility(self):
        seed_dir, _r, _x = self.make_seed()
        peer = self.tmp / "peer"
        peer.mkdir()
        keys = H.cell_keys(self.inp["cells"], (self.DOCS[0],))
        (peer / "checkpoint.pilot.json").write_text(json.dumps({"docs": [self.DOCS[0]], "hits": {k: True for k in keys}}))
        out = self.tmp / "fut"

        class Blank:                                                              # a v2 backend that finds nothing: v2 hits 0 < v1 hits 2
            calls = 0
            prefill_tokens = 0

            def complete(self, prompt, **kw):
                Blank.calls += 1
                return {"text": json.dumps({s: [] for s in ft.SLOTS}), "prompt_tokens": 5, "cache_n": 0, "decode_tokens": 1, "finish_reason": "stop"}
        rec, r = self.ingest(out, Blank(), "v2", seed_dir / "store.sqlite", peer_out=peer)
        self.assertEqual(r["rc"], H.RC_GATE)
        res = json.loads((out / "result.cpu.json").read_text())
        self.assertEqual((res["verdict"], res["futility_stop"]), ("agg113_futility", "M1_pilot_below_v1"))
        self.assertFalse(H.cpu_ready(res, "v2")[0])

    def test_audit_and_answers_run_per_store_and_per_arm(self):
        seed_dir, _r, _x = self.make_seed()
        outs = {}
        for pipe, be in (("v1", self.be_v1()), ("v2", self.be_v2())):
            outs[pipe] = self.tmp / pipe
            self.ingest(outs[pipe], be, pipe, seed_dir / "store.sqlite")
        stores = {t: P.open_store(outs[t] / "store.sqlite") for t in outs}
        rec = P.Recorder(self.tmp / "gpu" / "results.jsonl")
        try:
            class Aud:
                calls = 0
                prefill_tokens = 0

                def complete(self, prompt, **kw):
                    Aud.calls += 1
                    return {"text": "[]", "prompt_tokens": 5, "cache_n": 0, "finish_reason": "stop"}

                chat = complete
            r = H.run_audit(rec, stores, Aud(), guard=self.guard)
            self.assertTrue(r["complete"])
            self.assertTrue({"v1:audit_done", "v2:audit_done"} <= rec.done)
            r2 = H.run_audit(rec, stores, Aud(), guard=self.guard)           # resume: nothing redone
            self.assertTrue(r2["complete"])
        finally:
            for s in stores.values():
                s.close()


class CliDryRun(unittest.TestCase):
    def test_the_cli_declares_the_phases(self):
        self.assertEqual(C.PHASES, ("preflight", "l3", "ingest", "gpu-audit", "gpu-answers", "gpu", "score"))
        ns = C.build_parser().parse_args(["--phase", "ingest", "--pipeline", "v2", "--dry-run"])
        self.assertEqual((ns.phase, ns.pipeline, ns.dry_run), ("ingest", "v2", True))

    @needs_orch
    def test_dry_run_writes_nothing_and_starts_nothing(self):
        import io
        from contextlib import redirect_stdout
        tmp = Path(tempfile.mkdtemp())
        buf = io.StringIO()
        with redirect_stdout(buf):
            C.main(["--phase", "ingest", "--pipeline", "v1", "--dry-run", "--orch-root", str(ORCH), "--orch-pin", H.ORCH_PIN_MIN,
                    "--seed-store", str(tmp / "nope.sqlite")])
        out = json.loads(buf.getvalue())
        self.assertEqual((out["model_calls"], out["servers_started"], out["wrote_files"]), (0, 0, 0))
        self.assertEqual(out["plan"]["estimate"]["label"], "ESTIMATE")
        self.assertEqual(list(tmp.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
