"""AGG-1.10 tests (fake backend: no model, no network, no process, no PDF).

Stdlib tests always run. The store / planner / SQL-answer tests import the orchestrator package and skip without it: run them
with the orchestrator venv python and ``AGG_ORCH_ROOT=<tree at the agg110.v1 pin>`` (the tree must carry ``planner.sql_shape``).
"""

from __future__ import annotations

import hashlib
import json
import os
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
import agg110 as G  # noqa: E402
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import run_agg110 as R  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/epyc-orchestrator"))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation import planner  # noqa: E402
    from src.aggregation import sql_grammar as SG  # noqa: E402
    from src.aggregation.store import CorpusStore  # noqa: E402
    from src.models.document import DocumentPreprocessResult, Section  # noqa: E402
    HAVE_ORCH = hasattr(planner, "sql_shape") and hasattr(SG, "build_pivot_gbnf")
except Exception:  # pragma: no cover
    HAVE_ORCH = False


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class PinAndFrozenTests(unittest.TestCase):
    def test_pins_match(self):
        self.assertEqual(G.pin_problems(D1, frozen=G.FROZEN_ALL), [])
        self.assertEqual(G.phase0_pin_problems(D1), [])
        self.assertEqual(sha(D1 / "PREDECLARED_AGG110.md"), G.PREDECLARED_SHA)
        self.assertEqual(sha(D1 / "PREDECLARED.md"), X.PREDECLARED_SHA)

    def test_pin_check_fails_on_a_changed_predeclared(self):
        tmp = Path(tempfile.mkdtemp())
        for f in list(G.FROZEN_ALL) + [G.PREDECLARED_FILE]:
            shutil.copy(D1 / f, tmp / f)
        self.assertEqual(G.pin_problems(tmp, frozen=G.FROZEN_ALL), [])
        (tmp / G.PREDECLARED_FILE).write_text((tmp / G.PREDECLARED_FILE).read_text() + "x")
        self.assertTrue(any(G.PREDECLARED_FILE in p for p in G.pin_problems(tmp, frozen=G.FROZEN_ALL)))
        (tmp / "agg110_ceiling.json").write_text("{}")
        self.assertTrue(any("agg110_ceiling.json" in p for p in G.pin_problems(tmp, frozen=G.FROZEN_ALL)))

    def test_planner_versions_and_caps(self):
        self.assertEqual((G.V0, G.V1), ("agg11.v0", "agg110.v1"))
        self.assertEqual(G.SQL_N_PREDICT, {"agg11.v0": 160, "agg110.v1": 256})
        self.assertEqual(P.CFG.sql_n_predict, 160)            # the frozen Phase 0 value: v0 keeps it

    def test_the_26_and_the_frozen_ceiling_and_pilot(self):
        fz = G.load_frozen(D1)
        golds = {json.loads(l)["id"]: json.loads(l) for l in (D1 / "golds.jsonl").read_text().splitlines() if l.strip()}
        self.assertEqual(G.frozen_problems(fz, golds), [])
        rows = G.affected(fz["miss"])
        self.assertEqual(len(rows), 26)
        self.assertEqual(sum(r["category"] == "wrong_plan_derived_or_filtered" for r in rows), 17)
        self.assertEqual(sum(r["category"] == "sql_malformed_or_degenerate" for r in rows), 9)
        c = fz["ceiling"]
        self.assertEqual((c["n"], c["n_valid"], c["n_ceiling_true"]), (26, 26, 9))
        self.assertEqual(c["reference_on_gold_cells"]["n_correct"], 26)       # compile sanity: a correct plan on gold cells is right
        self.assertTrue(c["longest_within_60pct_of_cap"])
        pilot = fz["pilot"]["qids"]
        self.assertEqual(len(pilot), 6)
        cm = {r["qid"]: r for r in c["rows"]}
        self.assertTrue(all(cm[q]["ceiling"] for q in pilot))
        self.assertEqual(pilot, G.pilot_selection(c)["qids"])
        self.assertEqual(G.question_order([r["qid"] for r in rows], pilot)[:6], pilot)
        self.assertEqual(len(set(G.question_order([r["qid"] for r in rows], pilot))), 26)

    def test_reference_sql_shapes(self):
        corpus = {"docs": [{"doc_name": "JPMORGAN_2022_10K", "company": "JPMorgan"}]}
        g = {"op": "top3", "expr": "net_income/revenue", "where": None, "excluded_by_rule": []}
        self.assertEqual(G.compile_reference_sql(g, corpus),
                         "SELECT entity, net_income / revenue FROM facts_pivot WHERE period = '2022' ORDER BY net_income / "
                         "revenue DESC LIMIT 3")
        g = {"op": "count_gt", "expr": "dividends_paid/net_income", "threshold": 0.6, "excluded_by_rule": ["JPMORGAN_2022_10K"],
             "where": {"expr": "net_income", "cmp": ">", "t": 0}}
        self.assertEqual(G.compile_reference_sql(g, corpus),
                         "SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND net_income > 0 AND entity != 'JPMorgan' "
                         "AND dividends_paid / net_income > 0.6")
        g = {"op": "sum", "expr": "net_income", "where": {"expr": "revenue", "cmp": ">", "t": 90000}, "excluded_by_rule": []}
        self.assertEqual(G.compile_reference_sql(g, corpus),
                         "SELECT SUM(net_income) FROM facts_pivot WHERE period = '2022' AND revenue > 90000")
        self.assertEqual(G.rows_to_final({"op": "top3"}, [("A", 1), ("B", 2), ("C", 3)]), "FINAL: A, B, C")
        self.assertEqual(G.rows_to_final({"op": "count_gt"}, [(7.0,)]), "FINAL: 7")

    def test_ceiling_json_is_what_the_script_computes_from_the_frozen_inputs(self):
        # the stored ceiling's reference statements recompile from golds.jsonl + corpus.json (no store needed)
        c = json.loads((D1 / "agg110_ceiling.json").read_text())
        golds = {json.loads(l)["id"]: json.loads(l) for l in (D1 / "golds.jsonl").read_text().splitlines() if l.strip()}
        corpus = json.loads((D1 / "corpus.json").read_text())
        for r in c["rows"]:
            self.assertEqual(r["reference_sql"], G.compile_reference_sql(golds[r["qid"]], corpus), r["qid"])


class PureEvalTests(unittest.TestCase):
    def _ceiling(self):
        return {"n_ceiling_true": 1, "n": 2, "rows": [{"qid": "a", "category": "wrong_plan_derived_or_filtered", "ceiling": True},
                                                      {"qid": "b", "category": "sql_malformed_or_degenerate", "ceiling": False}]}

    def test_m4_and_bounds(self):
        ok = G.m4([{"calls": 2, "prefill_tokens": 700}, {"calls": 3, "prefill_tokens": 900}, {"calls": 2, "prefill_tokens": 650}])
        self.assertEqual((ok["median_calls"], ok["median_prefill"], ok["within_bounds"]), (2, 700, True))
        self.assertFalse(G.m4([{"calls": 4, "prefill_tokens": 100}])["within_bounds"])
        self.assertFalse(G.m4([{"calls": 2, "prefill_tokens": 2600}])["within_bounds"])
        self.assertEqual(G.m4([])["n"], 0)

    def test_pilot_futility_and_budget(self):
        pilot = ["a", "b"]
        ans = {"a": {"correct": False, "wall_s": 10}, "b": {"correct": None, "wall_s": 5}}
        ev = G.pilot_eval(ans, pilot)
        self.assertEqual((ev["fixed"], ev["gates_tripped"]), (0, ["futility:pilot_fixes_0_of_6"]))
        self.assertEqual(G.pilot_eval({"a": {"correct": False, "wall_s": 1}}, pilot)["gates_tripped"], [])   # pilot not finished
        ans["b"]["correct"] = True
        self.assertEqual(G.pilot_eval(ans, pilot)["gates_tripped"], [])
        ans["a"]["wall_s"] = 700
        self.assertEqual(G.pilot_eval(ans, pilot)["budget_stops"], ["pilot_wall_over_600s"])

    def test_verdict_splits_and_reports_planner_version(self):
        ans = {"a": {"qid": "a", "correct": True, "calls": 2, "prefill_tokens": 800, "planner_version": G.V1, "sql_reask": False,
                     "sql_valid": True, "finish_reason": "stop"},
               "b": {"qid": "b", "correct": False, "calls": 3, "prefill_tokens": 900, "planner_version": G.V1, "sql_reask": True,
                     "sql_valid": False, "finish_reason": "length"}}
        v = G.verdict110(ans, self._ceiling(), ["a", "b"])
        self.assertEqual(v["after"], {"correct": 1, "n": 2, "accuracy": 0.5})
        self.assertEqual(v["before"]["correct"], 0)
        self.assertEqual(v["by_ceiling"], {"ceiling_true": {"n": 1, "correct": 1}, "store_limited": {"n": 1, "correct": 0}})
        self.assertEqual((v["n_reask"], v["n_sql_invalid"], v["n_answer_truncated"], v["planner_versions"]), (1, 1, 1, [G.V1]))
        self.assertEqual((v["verdict"], v["complete"], v["gates_tripped"]), ("agg_1_10_done", True, []))
        partial = G.verdict110({"a": ans["a"]}, self._ceiling(), ["a", "b"])
        self.assertEqual((partial["verdict"], partial["complete"], partial["m4"]["n"]), ("incomplete", False, 1))
        ans["a"]["prefill_tokens"] = 5000
        ans["b"]["prefill_tokens"] = 5000
        self.assertEqual(G.verdict110(ans, self._ceiling(), ["a", "b"])["verdict"], "phase1_gate_tripped")

    def test_runner_defaults(self):
        a = R.build_parser().parse_args(["--phase", "answers"])
        self.assertEqual(a.planner_version, G.V1)
        self.assertEqual(R.PHASES, ("preflight", "answers", "score"))


def _doc(text):
    return DocumentPreprocessResult(original_path="/x/a.pdf", sections=[Section(id="s1", title="S", level=1, content=text,
                                    page_start=1, page_end=1)], figures=[], total_pages=1, tables=[])


class ScriptedBackend:
    """Returns the scripted SQL strings in order for grammar calls, a FINAL line for the answer call."""

    def __init__(self, sqls, finish="stop"):
        self.calls = self.prefill_tokens = 0
        self.sqls, self.finish, self.prompts, self.grammars = list(sqls), finish, [], []

    def chat(self, prompt, *, n_predict, grammar=None, cache_prompt=True, id_slot=None, logprobs=None):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4
        self.prompts.append(prompt)
        if grammar:
            self.grammars.append((grammar, n_predict))
            return {"text": self.sqls.pop(0), "finish_reason": self.finish}
        return {"text": "rows seen\nFINAL: Alpha", "finish_reason": "stop"}


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with the agg110.v1 plan not importable (set AGG_ORCH_ROOT)")
class SqlAnswerV1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = CorpusStore(self.tmp / "s.sqlite")
        self.old_corpus = P.CORPUS_ID
        P.CORPUS_ID = "c"
        facts = []
        for ent, fl in {"Alpha": {"revenue": 100.0, "capex": 10.0}, "Beta": {"revenue": 200.0, "capex": 5.0},
                        "Gamma": {"revenue": 50.0}}.items():
            r = self.store.ingest(_doc(f"{ent} filing\n"), corpus_id="c", doc_id=ent.upper() + "_2022_10K", entity=ent)
            for i, (f, v) in enumerate(fl.items()):
                facts.append({"fact_id": f"{ent}{i}", "doc_hash": r.doc_hash, "doc_id": ent.upper() + "_2022_10K", "entity": ent,
                              "period": "2022", "field": f, "value": v, "unit": "USD_millions", "quote": "q", "quote_start": 0,
                              "quote_end": 1, "chunk_id": "x", "table_id": None, "page": 1, "line_start": 1, "line_end": 1,
                              "confidence": "high", "extractor": "l3_textrow.v1", "audit_status": "unaudited"})
        self.store.write_facts(facts)
        self.store.refresh_canonical("c")
        self.names = ["Alpha", "Beta", "Gamma"]

    def tearDown(self):
        P.CORPUS_ID = self.old_corpus
        self.store.close()

    GOOD = "SELECT entity FROM facts_pivot WHERE period = '2022' ORDER BY capex / revenue ASC LIMIT 1"

    def test_valid_statement_one_sql_call_plus_answer_and_null_guard(self):
        b = ScriptedBackend([self.GOOD])
        text, st, info = G.sql_answer_v1(b, self.store, "Which company had the lowest capex share of revenue?", self.names, P.CFG,
                                         corpus_id="c")
        self.assertEqual((st, b.calls, info["sql_valid"], info["sql_reask"], info["planner_version"]), ("ok", 2, True, False, "agg110.v1"))
        self.assertEqual(info["n_rows"], 1)
        self.assertIn("capex IS NOT NULL", info["sql_executed"])            # Gamma (no capex) never ranks first
        self.assertIn("revenue IS NOT NULL", info["sql_executed"])
        self.assertEqual(b.grammars[0][1], 256)                              # SQL n_predict
        self.assertIn("facts_pivot", b.grammars[0][0])
        self.assertIn("SQL: " + info["sql_executed"], b.prompts[-1])
        self.assertTrue(text.endswith("FINAL: Alpha"))

    def test_entity_enumeration_is_rejected_and_reasked_once(self):
        enum = ("SELECT COUNT(*) FROM facts_pivot WHERE period = '2022' AND entity = 'Alpha' AND entity = 'Beta' "
                "AND entity = 'Gamma'")
        b = ScriptedBackend([enum, self.GOOD])
        _t, st, info = G.sql_answer_v1(b, self.store, "q", self.names, P.CFG, corpus_id="c")
        self.assertEqual((st, b.calls, info["sql_reask"], info["sql_valid"]), ("ok", 3, True, True))
        self.assertTrue(info["sql_first_reason"].startswith("entity_enumeration"))
        self.assertIn("rejected", b.prompts[1])

    def test_two_rejections_answer_from_no_rows_and_record_it(self):
        b = ScriptedBackend(["SELECT SUM(quote) FROM facts_pivot WHERE period = '2022'"] * 2)
        _t, st, info = G.sql_answer_v1(b, self.store, "q", self.names, P.CFG, corpus_id="c")
        self.assertEqual((st, b.calls, info["sql_valid"], info["n_rows"]), ("ok", 3, False, 0))

    def test_truncation_is_invalid_even_if_the_prefix_looks_fine(self):
        b = ScriptedBackend([self.GOOD, self.GOOD], finish="length")
        _t, _st, info = G.sql_answer_v1(b, self.store, "q", self.names, P.CFG, corpus_id="c")
        self.assertTrue(info.get("sql_truncated") and info["sql_valid"] is False)

    def test_store_file_is_not_changed_by_the_pivot_view(self):
        self.store.conn.commit()
        before = sha(self.tmp / "s.sqlite")
        G.sql_answer_v1(ScriptedBackend([self.GOOD]), self.store, "q", self.names, P.CFG, corpus_id="c")
        self.store.conn.commit()
        self.assertEqual(sha(self.tmp / "s.sqlite"), before)

    def test_version_dispatch_v0_is_the_unchanged_aggregate_path_and_records_version(self):
        sentinel = ("t", "ok", {"path": "sql"})
        with mock.patch.object(X, "p_large_answer", return_value=sentinel) as m:
            text, st, extra = G.p_large_answer_v(None, None, {"question": "q"}, scope=None, planner=None, docs_meta=None, built=[],
                                                 sql_present=True, names=[], cfg=P.CFG, corpus_chunks=[], corpus_bm25=None,
                                                 doc_chunks=lambda d: [])
        self.assertEqual((m.call_count, extra["planner_version"], extra["path"]), (1, "agg11.v0", "sql"))
        with self.assertRaises(ValueError):
            G.p_large_answer_v(None, None, {"question": "q"}, scope=None, planner=None, docs_meta=None, built=[], sql_present=True,
                               names=[], cfg=P.CFG, corpus_chunks=[], corpus_bm25=None, doc_chunks=lambda d: [],
                               planner_version="agg999")

    def test_version_dispatch_v1_replaces_only_the_sql_answer(self):
        class Dec:
            path, decided_by, receipt = "sql", "size_rule", {}

        class Pl:
            def scoped_facts(self, q, docs, **kw):
                return type("F", (), {"corpus_tokens": 4_000_000})(), "aggregation_cue"

            def plan(self, q, f):
                return Dec()

            def sql_shape(self, q):
                return {"shape": "pivot", "fields": [], "cues": []}

        with mock.patch.object(G, "sql_answer_v1", return_value=("FINAL: x", "ok", {"sql": "s"})) as m, \
                mock.patch.object(X, "p_large_answer") as old:
            text, st, extra = G.p_large_answer_v(None, None, {"question": "q"}, scope=(None, "aggregation_cue", 1), planner=Pl(),
                                                 docs_meta=[], built=[], sql_present=True, names=["A"], cfg=P.CFG,
                                                 corpus_chunks=[], corpus_bm25=None, doc_chunks=lambda d: [],
                                                 planner_version=G.V1, corpus_id="c")
        self.assertEqual((m.call_count, old.call_count, extra["planner_version"], extra["path"], extra["sql"]),
                         (1, 0, "agg110.v1", "sql", "s"))
        self.assertEqual(extra["sql_shape"]["shape"], "pivot")

    def test_v0_grammar_and_prompt_unchanged_by_the_new_code(self):
        # sql_answer (v0) still builds its facts_canonical grammar and 160-token cap
        b = ScriptedBackend(["SELECT entity FROM facts_canonical WHERE field = 'revenue' AND period = '2022' ORDER BY value DESC LIMIT 1"])
        _t, st, info = P.sql_answer(b, self.store, "Which company had the highest revenue?", self.names, P.CFG)
        self.assertEqual((st, info["sql_table"], b.grammars[0][1]), ("ok", "facts_canonical", 160))
        self.assertIn("table facts_canonical(entity, doc_id, period, field, value, unit, quote, page)", b.prompts[0])
        self.assertNotIn("planner_version", info)


@unittest.skipUnless(HAVE_ORCH, "orchestrator tree with the agg110.v1 plan not importable (set AGG_ORCH_ROOT)")
class RoutingTests(unittest.TestCase):
    def test_all_26_route_pivot_and_exposure_is_reported(self):
        inp = R.load(D1)
        agg = X.verified_agg(inp)
        ids = [r["qid"] for r in G.affected(inp["frozen"]["miss"])]
        rep = G.routing_report(agg, planner, {}, ids)
        self.assertEqual((rep["n"], rep["pivot_in_26"], rep["single_in_26"]), (58, 26, []))
        self.assertEqual(rep["pivot"], 30)


STORE11 = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/store.sqlite")


class OracleBackend:
    """Plays the model: the reference SQL of the question it is asked (from the frozen ceiling) and that row's scored answer."""

    def __init__(self, inp, wrong=False):
        self.calls = self.prefill_tokens = 0
        self.by_q = {g["question"]: g["id"] for g in inp["golds"]}
        self.cm = {r["qid"]: r for r in inp["frozen"]["ceiling"]["rows"]}
        self.wrong, self.cur = wrong, None

    def chat(self, prompt, *, n_predict, grammar=None, cache_prompt=True, id_slot=None, logprobs=None):
        self.calls += 1
        self.prefill_tokens += len(prompt) // 4
        if grammar:
            q = prompt.split("Question: ", 1)[1].rsplit("\nSQL:", 1)[0].split("\nThat statement was rejected")[0].strip()
            self.cur = self.cm[self.by_q[q]]
            return {"text": self.cur["reference_sql"], "finish_reason": "stop"}
        return {"text": "ok\n" + ("FINAL: nothing" if self.wrong else self.cur["answer"]), "finish_reason": "stop"}


@unittest.skipUnless(HAVE_ORCH and STORE11.exists(), "needs the orchestrator tree and the AGG-1.1 store")
class RunnerEndToEndTests(unittest.TestCase):
    """The runner's answers loop on a COPY of the real AGG-1.1 store with an oracle standing in for the model."""

    def _run(self, wrong):
        tmp = Path(tempfile.mkdtemp())
        inp = R.load(D1)
        X.configure(P.CORPUS_ID)
        rec = P.Recorder(tmp / "results.jsonl")
        before = sha(STORE11)
        res = R.run_answers(rec, tmp, inp, planner, OracleBackend(inp, wrong), store_src=STORE11, guard=lambda: None,
                            smoke=False, corpus_id=P.CORPUS_ID, planner_version=G.V1)
        self.assertEqual(sha(STORE11), before)                      # the source store is never mutated
        return tmp, rec, res

    def test_full_run_matches_the_ceiling_and_records_the_version(self):
        tmp, rec, res = self._run(wrong=False)
        self.assertEqual((res["complete"], res["verdict"], res["rc"]), (True, "agg_1_10_done", 0))
        v = res["result"]
        self.assertEqual((v["after"]["correct"], v["n_scored"], v["m4"]["median_calls"]), (9, 26, 2))
        self.assertEqual(v["planner_versions"], ["agg110.v1"])
        ans = [r for r in rec.records if r["kind"] == "answer"]
        self.assertEqual(len(ans), 26)
        self.assertTrue(all(r["planner_version"] == "agg110.v1" and r["sql_shape"]["shape"] == "pivot" and r["path"] == "sql"
                            for r in ans))
        self.assertEqual([r["qid"] for r in ans][:6], inp_pilot())
        self.assertIsNotNone(next(r for r in rec.records if r.get("unit") == "pilot_done"))

    def test_pilot_futility_stops_after_six(self):
        tmp, rec, res = self._run(wrong=True)
        self.assertEqual((res["verdict"], res["rc"], res["gate_tripped"]), ("phase1_gate_tripped", 3, ["futility:pilot_fixes_0_of_6"]))
        self.assertEqual(len([r for r in rec.records if r["kind"] == "answer"]), 6)

    def test_resume_skips_finished_units(self):
        tmp, rec, res = self._run(wrong=False)
        inp = R.load(D1)
        b = OracleBackend(inp)
        rec2 = P.Recorder(tmp / "results.jsonl")
        res2 = R.run_answers(rec2, tmp, inp, planner, b, store_src=STORE11, guard=lambda: None, smoke=False,
                             corpus_id=P.CORPUS_ID, planner_version=G.V1)
        self.assertEqual((b.calls, res2["verdict"]), (0, "agg_1_10_done"))


def inp_pilot():
    return json.loads((D1 / "agg110_pilot.json").read_text())["qids"]


if __name__ == "__main__":
    unittest.main()
