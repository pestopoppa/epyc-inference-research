"""AGG-0.4 arms tests with fake backends (no model, no network, no process).

Scorer tests are stdlib. The store/arm tests import the orchestrator package: run with the
orchestrator venv python and ``AGG_ORCH_ROOT=<orchestrator tree with src/aggregation/llm_backend.py>``;
without it they skip. ``ORCHESTRATOR_FEATURE_AGG_LAYER=1`` is set here.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "benchmarks" / "agg_phase0"
sys.path.insert(0, str(D))
import phase0_arms as P  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/epyc-orchestrator"))
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation.llm_backend import GATE_GBNF, OutOfWindow  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402

    HAVE_ORCH = True
except Exception:  # pragma: no cover
    HAVE_ORCH = False

os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
DOCS = {"CORNING_2022_10K": ("Corning", "14,189"), "PAYPAL_2022_10K": ("Paypal", "27,518")}


class FakeBackend:
    """Rows only for text that states revenue; answers name Corning... unless told otherwise."""

    def __init__(self, final="FINAL: Paypal", overflow_on=None, never_final=False, truncate=False):
        self.never_final, self.truncate = never_final, truncate
        self.calls = self.prefill_tokens = 0
        self.prompts: list[str] = []
        self.final, self.overflow_on = final, overflow_on

    def _bill(self, prompt):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4
        self.prompts.append(prompt)
        if self.overflow_on and self.overflow_on in prompt:
            raise OutOfWindow("exceed_context_size_error")

    def complete(self, prompt, *, grammar, n_predict, cache_prompt, id_slot):
        if grammar == GATE_GBNF:
            self._bill(prompt)
            return {"text": "yes", "prompt_tokens": len(prompt) // 4, "cache_n": 0}
        return self.chat(prompt, n_predict=n_predict, grammar=grammar)

    def chat(self, prompt, *, n_predict, grammar=None, cache_prompt=True, id_slot=None, logprobs=None):
        self._bill(prompt)
        if grammar and grammar.startswith("root ::= \"[\""):
            m = re.search(r"Total revenues [\d,]+", prompt.split("Excerpt:\n", 1)[-1])
            rows = [] if not m or "Field: revenue" not in prompt else [{
                "period": "FY2022", "value": float(m.group(0).split()[-1].replace(",", "")),
                "unit": "USD_millions", "quote": m.group(0), "confidence": "medium"}]
            text = json.dumps(rows)
        elif grammar and grammar.startswith('root ::= "SELECT '):
            text = "SELECT entity, value FROM facts_canonical WHERE field = 'revenue' ORDER BY value DESC"
        elif grammar and "SEARCH" in grammar:
            if self.never_final:  # a runaway loop: alternates SEARCH / READ, never FINAL
                text = "SEARCH: revenues" if prompt.count("\n> ") % 2 == 0 else "READ: nope"
            else:
                text = "SEARCH: revenues" if "> SEARCH" not in prompt else self.final
        else:
            text = f"Revenue rows considered.\n{self.final}"
        return {"text": text, "prompt_tokens": len(prompt) // 4, "cache_n": 0,
                "finish_reason": "length" if self.truncate and not grammar else "stop"}


class ScorerTests(unittest.TestCase):
    def test_final_line(self):
        self.assertEqual(P.final_line("x\nFINAL: Verizon\n"), "Verizon")
        self.assertEqual(P.final_line("a\nlast"), "last")
        self.assertEqual(P.final_line(""), "")

    def test_agg_ops(self):
        g = lambda op, gold: {"op": op, "gold": gold}  # noqa: E731
        self.assertTrue(P.score_agg(g("argmax", "AES Corporation"), "FINAL: AES Corporation"))
        self.assertFalse(P.score_agg(g("argmax", "Verizon"), "FINAL: JPMorgan"))
        self.assertTrue(P.score_agg(g("top3", ["Verizon", "JPMorgan", "PepsiCo"]), "FINAL: Verizon, JPMorgan, PepsiCo"))
        self.assertFalse(P.score_agg(g("top3", ["Verizon", "JPMorgan", "PepsiCo"]), "FINAL: JPMorgan, Verizon, PepsiCo"))
        self.assertTrue(P.score_agg(g("sum", 538018.8), "FINAL: $538,019 million"))
        self.assertTrue(P.score_agg(g("sum", 538018.8), "FINAL: 538.0 billion"))  # 0.003% off
        self.assertFalse(P.score_agg(g("sum", 538018.8), "FINAL: 545,000"))      # 1.3% off
        self.assertTrue(P.score_agg(g("count_gt", 4), "FINAL: 4"))
        self.assertFalse(P.score_agg(g("count_gt", 4), "FINAL: 5"))
        self.assertIsNone(P.score_agg(g("count_gt", 4), "FINAL: none"))

    def test_single_keys(self):
        keys = json.loads((D / "single_fact_keys.json").read_text())["keys"]
        ok = {"financebench_id_00070": "FINAL: No, working capital was -$1,561 million.",
              "financebench_id_00005": "FINAL: Yes, $2.278 billion",
              "financebench_id_01346": "FINAL: It rose from 20% in FY2021 to 23% in FY2022",
              "financebench_id_00216": "FINAL: No, quick ratio 0.54",
              "financebench_id_10420": "FINAL: -1.42%",
              "financebench_id_06272": "FINAL: 0.80",
              "financebench_id_03718": "FINAL: 0.4%",
              "financebench_id_04481": "FINAL: 16.5%"}
        for qid, text in ok.items():
            self.assertTrue(P.score_single(keys[qid], text), qid)
        self.assertFalse(P.score_single(keys["financebench_id_00070"], "FINAL: Yes, 1561"))
        self.assertFalse(P.score_single(keys["financebench_id_10420"], "FINAL: 1.42%"))  # signed
        self.assertFalse(P.score_single(keys["financebench_id_04481"], "FINAL: 16.7%"))
        self.assertEqual(set(keys), {json.loads(l)["id"] for l in (D / "single_fact.jsonl").read_text().splitlines()})


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not importable")
class ScopeOnRealQuestionsTests(unittest.TestCase):
    def test_all_30_predeclared_questions_resolve_as_designed(self):
        from src.aggregation import planner  # noqa: PLC0415
        os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
        corpus = json.loads((D / "corpus.json").read_text())["docs"]
        docs = [{"doc_id": d["doc_name"], "entity": d["company"], "n_tokens_est": 150_000} for d in corpus]
        for line in (D / "single_fact.jsonl").read_text().splitlines():
            q = json.loads(line)
            doc, why = planner.resolve_scope(q["question"], docs)
            self.assertEqual((doc or {}).get("doc_id"), q["doc_name"], q["id"])
        for q in json.loads((D / "aggregation_questions.json").read_text())["questions"]:
            doc, why = planner.resolve_scope(q["q"] if "q" in q else q["question"], docs)
            self.assertIsNone(doc, q["id"])


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree not importable")
class ArmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = CorpusStore(self.tmp / "store.sqlite")
        for name, (co, rev) in DOCS.items():
            secs = [Section(id="s1", title="Income", level=1, content=f"Statements of Income\nTotal revenues {rev}\n",
                            page_start=60, page_end=60),
                    Section(id="s2", title="Risk", level=1, content="Risk factors about weather " * 40,
                            page_start=10, page_end=11)]
            self.store.ingest(DocumentPreprocessResult(original_path=f"/{name}.pdf", sections=secs, figures=[],
                                                       total_pages=100, tables=[]),
                              corpus_id=P.CORPUS_ID, doc_id=name, entity=co)
        self.set_dir = self.tmp / "set"
        self.set_dir.mkdir()
        with (self.set_dir / "facts_sheet.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["doc_name", "metric", "value_usd_m", "status"])
            w.writeheader()
            for name, (_, rev) in DOCS.items():
                w.writerow({"doc_name": name, "metric": "revenue", "value_usd_m": rev, "status": "filled"})
        self.agg = [{"id": "agg01", "question": "Which company had the highest FY2022 revenue?", "expr": "revenue",
                     "op": "argmax", "gold": "Paypal", "excluded_by_rule": []}]
        self.single = [{"id": "financebench_id_00080", "doc_name": "PAYPAL_2022_10K",
                        "question": "Does Paypal have positive working capital?"}]
        self.guard = lambda: None

    def test_cpu_phase_l4_m1_resumable(self):
        rec = P.Recorder(self.tmp / "cpu" / "results.jsonl")
        b = FakeBackend()
        res = P.run_cpu(rec, self.store, self.set_dir, b, agg=self.agg, single=self.single, guard=self.guard)
        self.assertTrue(res["complete"])
        self.assertEqual(res["m1"], 1.0)
        self.assertIsNone(res["futility_stop"])
        kinds = [r["kind"] for r in rec.records]
        self.assertLess(kinds.index("manifest"), kinds.index("l4_unit"))  # futility 1 before any LLM call
        man = [r for r in rec.records if r["kind"] == "manifest"][0]
        self.assertFalse(set(man["built_layers"]) & {"L2", "L3", "L4"})
        ing = [r for r in rec.records if r["kind"] == "ingest"][0]
        self.assertEqual(ing["llm_calls"], b.calls)
        self.assertGreater(ing["prefill_tokens"], 0)
        # resume: a second run re-uses every unit, zero new calls
        rec2 = P.Recorder(self.tmp / "cpu" / "results.jsonl")
        b2 = FakeBackend()
        P.run_cpu(rec2, self.store, self.set_dir, b2, agg=self.agg, single=self.single, guard=self.guard)
        self.assertEqual(b2.calls, 0)
        # every line on disk is complete JSON (written as produced)
        for line in (self.tmp / "cpu" / "results.jsonl").read_text().splitlines():
            json.loads(line)

    def test_m1_futility_stop(self):
        with (self.set_dir / "facts_sheet.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["doc_name", "metric", "value_usd_m", "status"])
            w.writeheader()
            for name in DOCS:
                w.writerow({"doc_name": name, "metric": "revenue", "value_usd_m": "999", "status": "filled"})
        rec = P.Recorder(self.tmp / "cpu2" / "results.jsonl")
        res = P.run_cpu(rec, self.store, self.set_dir, FakeBackend(), agg=self.agg, single=self.single,
                        guard=self.guard)
        self.assertEqual(res["futility_stop"], "M1")
        self.assertEqual(rec.records[-1]["kind"], "stop")

    def test_stop_raises_and_resumes(self):
        rec = P.Recorder(self.tmp / "s" / "results.jsonl")
        calls = {"n": 0}

        def guard():
            calls["n"] += 1
            if calls["n"] == 3:
                raise P.Stop("deadline")
        with self.assertRaises(P.Stop):
            P.run_cpu(rec, self.store, self.set_dir, FakeBackend(), agg=self.agg, single=self.single, guard=guard)
        n_done = len([r for r in rec.records if r["kind"] == "l4_unit"])
        self.assertEqual(n_done, 2)

    def test_audit_and_answers_and_verdict(self):
        cpu = P.Recorder(self.tmp / "cpu" / "results.jsonl")
        P.run_cpu(cpu, self.store, self.set_dir, FakeBackend(), agg=self.agg, single=self.single, guard=self.guard)
        gpu = P.Recorder(self.tmp / "gpu" / "results.jsonl")
        a = P.run_audit(gpu, self.store, FakeBackend(), guard=self.guard)
        self.assertTrue(a["complete"])
        aud = [r for r in gpu.records if r["kind"] == "audit"]
        self.assertTrue(aud and all(r["verdict"] == "audited_ok" for r in aud))  # medium -> always audited
        keys = {"financebench_id_00080": {"yesno": "yes", "numbers": [{"value": 12416, "decimals": 0}]}}
        res = P.run_answers(gpu, self.store, FakeBackend(), agg=self.agg, single=self.single, keys=keys,
                            names=["Corning", "Paypal"], guard=self.guard)
        self.assertTrue(res["complete"])
        ans = {(r["arm"], r["qid"]): r for r in gpu.records if r["kind"] == "answer"}
        self.assertEqual(len(ans), 2 + 2 + 2 + 1)  # P-small, B-today, P-large x2, B1 x2, B2 x1
        pl = ans[("P-large", "agg01")]
        self.assertEqual(pl["decided_by"], "size_rule")  # tiny test corpus: size rule, no planner LLM
        text, st, info = P.sql_answer(FakeBackend(), self.store, "highest revenue?", ["Corning", "Paypal"], P.CFG)
        self.assertEqual((st, info["sql_valid"], info["n_rows"]), ("ok", True, 2))
        self.assertIs(ans[("B2", "agg01")]["correct"], True)
        self.assertEqual(ans[("B2", "agg01")]["calls"], 3)  # 2 notes + 1 aggregate
        run_man = [r for r in gpu.records if r["kind"] == "manifest"]
        self.assertTrue(run_man and not set(run_man[0]["built_layers"]) & {"L2", "L3", "L4"})
        import run_phase0 as R  # noqa: PLC0415
        v = R.verdict(self.tmp / "cpu" / "results.jsonl", gpu, self.store)
        self.assertIn(v["verdict"], ("go_phase_1", "superseded_v2_not_viable"))
        self.assertIn("M2", v["kill_gate_tripped"])  # P-large == B1 here: no 25 pp margin

    def _gpu_answers(self, backend, name, single=None, agg=None, keys=None):
        cpu = P.Recorder(self.tmp / "cpu" / "results.jsonl")
        if not cpu.records:
            P.run_cpu(cpu, self.store, self.set_dir, FakeBackend(), agg=self.agg, single=self.single,
                      guard=self.guard)
        gpu = P.Recorder(self.tmp / name / "results.jsonl")
        P.run_answers(gpu, self.store, backend, agg=agg or self.agg, single=single or self.single,
                      keys=keys or {"financebench_id_00080": {"yesno": "yes"}},
                      names=["Corning", "Paypal"], guard=self.guard)
        return gpu, {(r["arm"], r["qid"]): r for r in gpu.records if r["kind"] == "answer"}

    def test_out_of_window_whole_doc_falls_back_to_loop_not_empty(self):
        """Defect 3: B-today recorded an empty answer with 0 calls when the filing exceeded the window."""
        gpu, ans = self._gpu_answers(FakeBackend(overflow_on="Filing (PAYPAL"), "g2")
        bt = ans[("B-today", "financebench_id_00080")]
        self.assertEqual(bt["status"], "ok")           # answered by the REPL search/read loop
        self.assertTrue(bt["whole_doc_refused"])
        self.assertGreaterEqual(bt["calls"], 1)
        self.assertTrue(bt["answer"])
        self.assertEqual(bt["call_cap"], P.CFG.single_doc_max_calls)
        self.assertEqual(ans[("B2", "agg01")]["status"], "out_of_window")  # B2 has no loop: recorded, not truncated

    def test_out_of_window_with_no_fallback_is_recorded_not_truncated(self):
        class Refuse(FakeBackend):
            def chat(self, prompt, **k):
                if "Filing (PAYPAL" in prompt or "Question:" in prompt and "SEARCH" in (k.get("grammar") or ""):
                    raise OutOfWindow("exceed_context_size_error")
                return super().chat(prompt, **k)
        gpu, ans = self._gpu_answers(Refuse(), "g3")
        bt = ans[("B-today", "financebench_id_00080")]
        self.assertEqual((bt["status"], bt["calls"], bt["correct"]), ("out_of_window", 0, None))

    def test_runaway_loop_is_capped_and_traced(self):
        """Defect 3: P-small ran 30 calls. The single-document loop stops at the harness cap, with its trace."""
        big = CorpusStore(self.tmp / "big.sqlite")  # a filing above the 200k whole-doc rule -> retrieval
        secs = [Section(id="s1", title="Income", level=1, content="Total revenues 1\n" * 20, page_start=1, page_end=1)]
        big.ingest(DocumentPreprocessResult(original_path="/x.pdf", sections=secs, figures=[], total_pages=9, tables=[]),
                   corpus_id=P.CORPUS_ID, doc_id="PAYPAL_2022_10K", entity="Paypal")
        big.conn.execute("UPDATE documents SET n_tokens_est = 400000")
        big.conn.commit()
        for arm_doc in ("small", "today"):
            b = FakeBackend(never_final=True)
            extra: dict = {}
            text, st = P.single_doc_answer(b, big, "PAYPAL_2022_10K", "q?", ["Paypal"], P.CFG,
                                           "retrieval" if arm_doc == "small" else "whole_doc",
                                           big.chunks(P.CORPUS_ID), extra)
            self.assertEqual((st, b.calls), ("budget_exhausted", P.CFG.single_doc_max_calls))
            self.assertEqual(len(extra["loop_actions"]), P.CFG.single_doc_max_calls)
        self.assertLess(P.CFG.single_doc_max_calls, P.CFG.b1_max_calls)
        self.assertEqual(P.CFG.b1_max_calls, 30)  # PREDECLARED: B1 <= 30, untouched

    def test_whole_doc_answer_records_truncation(self):
        gpu, ans = self._gpu_answers(FakeBackend(truncate=True), "g4")
        self.assertTrue(ans[("B-today", "financebench_id_00080")]["truncated"])
        self.assertGreaterEqual(P.CFG.answer_n_predict, 1024)  # 384 cut whole-doc answers before FINAL

    def test_sql_path_reads_canonical_one_row_per_entity(self):
        """Defect 1: duplicate L4 rows per (entity, field, period) no longer reach SUM / top-k."""
        dup = []
        for name, (co, rev) in DOCS.items():
            v = float(rev.replace(",", ""))
            for k, (period, val, ext) in enumerate([("FY2022", v, "l4_llm.v1"), ("FY2022", v, "l4_llm.v1"),
                                                    ("FY2022", v * 0.4, "l4_llm.v1"), ("2022", v, "l3_textrow.v1")]):
                dup.append({"fact_id": f"{name}{k}", "doc_hash": self.store.corpus_docs(P.CORPUS_ID)[0]["doc_hash"],
                            "doc_id": name, "entity": co, "period": period, "field": "revenue", "value": val,
                            "unit": "USD_millions", "quote": "q", "page": 60 + k, "confidence": "high",
                            "extractor": ext, "audit_status": "unaudited"})
        self.store.write_facts(dup)
        stats = self.store.refresh_canonical(P.CORPUS_ID)
        self.assertEqual(stats["n_canonical"], len(DOCS))
        self.assertGreater(stats["n_conflicts"], 0)
        text, st, info = P.sql_answer(FakeBackend(), self.store, "highest revenue?", ["Corning", "Paypal"], P.CFG)
        self.assertEqual((st, info["sql_valid"], info["n_rows"], info["sql_table"]), ("ok", True, 2, "facts_canonical"))
        total = self.store.conn.execute("SELECT SUM(value) FROM facts_canonical WHERE field='revenue' "
                                        "AND period='2022'").fetchone()[0]
        self.assertAlmostEqual(total, sum(float(r.replace(",", "")) for _, r in DOCS.values()))

    def test_p_large_single_fact_follows_size_rules_over_a_large_corpus(self):
        """Defect 2: over a >= 1M-token corpus every question took sql (0 rows). Now a question that names
        one filing is sized by that filing."""
        self.store.conn.execute("UPDATE documents SET n_tokens_est = 150000 WHERE doc_id = 'CORNING_2022_10K'")
        self.store.conn.execute("UPDATE documents SET n_tokens_est = 900000 WHERE doc_id = 'PAYPAL_2022_10K'")
        self.store.conn.commit()
        single = [{"id": "s_corn", "doc_name": "CORNING_2022_10K", "question": "Does Corning have positive working capital?"},
                  {"id": "s_corn2", "doc_name": "CORNING_2022_10K", "question": "What is the effective tax rate of Corning?"},
                  {"id": "s_pp", "doc_name": "PAYPAL_2022_10K", "question": "Does Paypal have positive working capital?"}]
        keys = {q["id"]: {"yesno": "yes"} for q in single}
        gpu, ans = self._gpu_answers(FakeBackend(), "g5", single=single, keys=keys)
        c1, c2, pp = (ans[("P-large", x)] for x in ("s_corn", "s_corn2", "s_pp"))
        self.assertEqual((c1["path"], c1["decided_by"], c1["scope_doc"], c1["calls"]),
                         ("whole_doc", "deterministic_default_stub", "CORNING_2022_10K", 1))  # <= 200k at Q = 1
        self.assertEqual((c2["path"], c2["planner_questions_so_far"]), ("repl_nav", 2))  # <= 200k, Q = 2
        self.assertEqual((pp["path"], pp["scope_doc"]), ("retrieval", "PAYPAL_2022_10K"))  # > 200k
        for r in (c1, c2, pp):
            self.assertNotEqual(r["path"], "sql")
            self.assertNotIn("sql", r)
        # an aggregation question over the same >= 1M corpus still goes to sql
        self.assertEqual(ans[("P-large", "agg01")]["path"], "sql")
        self.assertEqual(ans[("P-large", "agg01")]["sql_table"], "facts_canonical")
        can = [r for r in gpu.records if r["kind"] == "canonical"]
        self.assertEqual(len(can), 1)

    def test_cli_gpu_phase_fails_closed_without_cpu_results(self):
        import contextlib, io  # noqa: E401,PLC0415
        import run_phase0 as R  # noqa: PLC0415
        with contextlib.redirect_stdout(io.StringIO()):
            rc = R.main(["--set-dir", str(D), "--out", str(self.tmp / "cli"), "--phase", "gpu-audit",
                         "--orch-root", str(ORCH), "--store", str(self.tmp / "store.sqlite"),
                         "--url", "http://127.0.0.1:1"])
        self.assertIn(rc, (2, 4))  # 2 only if the pinned PDFs are absent on this host
        if rc == 4:
            last = json.loads((self.tmp / "cli" / "results.jsonl").read_text().splitlines()[-1])
            self.assertEqual(last["kind"], "precondition_failed")


if __name__ == "__main__":
    unittest.main()
