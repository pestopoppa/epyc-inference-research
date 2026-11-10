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

    def __init__(self, final="FINAL: Paypal", overflow_on=None):
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
            text = "SELECT entity, value FROM facts WHERE field = 'revenue' ORDER BY value DESC"
        elif grammar and "SEARCH" in grammar:
            text = "SEARCH: revenues" if "> SEARCH" not in prompt else self.final
        else:
            text = f"Revenue rows considered.\n{self.final}"
        return {"text": text, "prompt_tokens": len(prompt) // 4, "cache_n": 0}


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

    def test_out_of_window_recorded_not_truncated(self):
        cpu = P.Recorder(self.tmp / "cpu" / "results.jsonl")
        P.run_cpu(cpu, self.store, self.set_dir, FakeBackend(), agg=self.agg, single=self.single, guard=self.guard)
        gpu = P.Recorder(self.tmp / "g2" / "results.jsonl")
        P.run_answers(gpu, self.store, FakeBackend(overflow_on="Filing (PAYPAL"), agg=self.agg,
                      single=self.single, keys={"financebench_id_00080": {"yesno": "yes"}},
                      names=["Corning", "Paypal"], guard=self.guard)
        ans = {(r["arm"], r["qid"]): r for r in gpu.records if r["kind"] == "answer"}
        self.assertEqual(ans[("B-today", "financebench_id_00080")]["status"], "out_of_window")
        self.assertIsNone(ans[("B-today", "financebench_id_00080")]["correct"])
        self.assertEqual(ans[("B2", "agg01")]["status"], "out_of_window")

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
