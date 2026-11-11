"""AGG-1.5 parity (M5) tests: no model, no network, no process, no PDF parsing.

Selection / scoring / verdict tests are stdlib. The store and arm tests import the orchestrator package: set
``AGG_ORCH_ROOT`` to a tree with ``src/aggregation/llm_backend.py`` (the AGG-0 pin) and run with a python that
imports it (the orchestrator venv); without it they skip. They need the Phase 0 store for the real-store run
(``AGG15_STORE``, default the CPU job's store) and skip when it is missing.
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
from unittest import mock

B = Path(__file__).resolve().parents[1] / "benchmarks"
D1, D0 = B / "agg_phase1", B / "agg_phase0"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import parity_m5 as M  # noqa: E402

ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/tmp/agg0-arms-gpu-20261110-r2/orch"))
STORE = Path(os.environ.get("AGG15_STORE", str(M.STORE_DEFAULT)))
os.environ["ORCHESTRATOR_FEATURE_AGG_LAYER"] = "1"
try:
    sys.path.insert(0, str(ORCH))
    from src.aggregation.llm_backend import OutOfWindow  # noqa: E402,F401
    from src.aggregation.registry import PassageHandle  # noqa: E402,F401  (the r3 builder; older pins lack it)
    HAVE_ORCH = True
except Exception:  # pragma: no cover
    HAVE_ORCH = False
HAVE_STORE = HAVE_ORCH and STORE.exists()


def ans(arm, qid, *, ok=True, calls=3, pre=1000, builds=0, bs=0.0, doc="D"):
    return {"kind": "answer", "arm": arm, "qid": qid, "doc": doc, "final": True, "status": "ok",
            "correct": ok, "calls": calls, "prefill_tokens": pre, "index_builds": builds, "index_build_s": bs,
            "wall_s": 10.0}


def records(n=20, p_kw=None, b_kw=None, built=()):
    r = []
    for i in range(n):
        q = f"p15_{i + 1:02d}"
        doc = f"D{i % 4}"
        r.append(ans("P-retrieval", q, doc=doc, builds=1 if i < 4 else 0, bs=0.1 if i < 4 else 0.0, **(p_kw or {})))
        r.append(ans("B-context.search", q, doc=doc, builds=1, bs=0.3, **(b_kw or {})))
        r.append({"kind": "manifest", "arm": "P-retrieval", "qid": q, "built_layers": ["L1", *built]})
    return r


class FrozenSetTests(unittest.TestCase):
    def setUp(self):
        self.q = json.loads((D1 / "parity_questions.json").read_text())
        self.v = json.loads((D1 / "parity_verification.json").read_text())

    def test_twenty_questions_four_mid_size_docs_all_verified(self):
        self.assertEqual(len(self.q["questions"]), 20)
        self.assertEqual([d["doc_name"] for d in self.q["docs"]], list(M.DOCS))
        self.assertEqual(sum(d["n_questions"] for d in self.q["docs"]), 20)
        for d in self.q["docs"]:
            self.assertTrue(M.MID_LO < d["n_tokens_est"] <= M.MID_HI, d)
        self.assertEqual(self.v["n_verified"], 20)
        self.assertEqual(len(set(q["id"] for q in self.q["questions"])), 20)
        for r in self.v["questions"].values():
            self.assertTrue(r["pdfium_found"] and r["pypdf_found"] and r["value_in_quote"] and r["verified"])

    def test_selection_is_reproducible_from_the_frozen_sheet(self):
        rows = list(csv.DictReader((D1 / "facts_sheet.csv").open(newline="")))
        cells = M.select_cells(rows)
        self.assertEqual([(c["doc_name"], c["metric"]) for c in cells],
                         [(q["doc_name"], q["metric"]) for q in self.q["questions"]])
        # the pilot (first 5) touches all four documents
        self.assertEqual(len({q["doc_name"] for q in self.q["questions"][:M.PILOT_N]}), 4)

    def test_hashes_pin_the_inputs(self):
        self.assertEqual(self.q["corpus_sha256"], M.sha_file(D1 / "corpus.json"))
        self.assertEqual(self.q["sheet_sha256"], M.sha_file(D1 / "facts_sheet.csv"))

    def test_label_and_keys(self):
        self.assertEqual(M.label_of("Total assets(a) $ 3,665,743"), "Total assets")
        self.assertEqual(M.label_of("Capital expenditures (4,551)"), "Capital expenditures")
        self.assertEqual(M.label_of("NET INCOME (LOSS) ATTRIBUTABLE TO THE AES CORPORATION $ (546)"),
                         "NET INCOME (LOSS) ATTRIBUTABLE TO THE AES CORPORATION")
        ni = next(q for q in self.q["questions"] if q["doc_name"].startswith("AES") and q["metric"] == "net_income")
        self.assertEqual(ni["gold_value_usd_m"], -546.0)
        self.assertEqual(ni["key"], {"numbers": [{"value": 546.0, "decimals": 0}]})  # unsigned: the magnitude
        import phase0_arms as P
        self.assertTrue(P.score_single(ni["key"], "FINAL: $546 million"))
        self.assertTrue(P.score_single(ni["key"], "FINAL: (546)"))
        self.assertFalse(P.score_single(ni["key"], "FINAL: 564"))

    def test_phase0_predeclared_untouched(self):
        self.assertEqual(M.sha_file(D0 / "PREDECLARED.md"),
                         "976a5fe87e3db33d6a5e4547ee4bb6cae324f44d107980342b62f7b6212b8bc9")

    def test_tolerances_are_the_phase0_m5_constants(self):
        self.assertEqual((M.ACC_TOL, M.CALLS_TOL, M.PREFILL_TOL), (1, 1, 2000))


class VerdictTests(unittest.TestCase):
    def test_parity_holds(self):
        s = M.score(records())
        self.assertEqual(s["verdict"], "parity_holds", s)
        self.assertEqual(s["saving"]["index_builds"], 16)
        self.assertAlmostEqual(s["saving"]["index_build_s"], 20 * 0.3 - 4 * 0.1)

    def test_accuracy_gate_is_within_one_question(self):
        r = records()
        for x in r:
            if x.get("arm") == "P-retrieval" and x["qid"] in ("p15_01", "p15_02"):
                x["correct"] = False
        s = M.score(r)  # P 18, B 20: diff 2 > 1
        self.assertEqual((s["verdict"], s["failing"]), ("phase1_gate_tripped", ["accuracy"]))
        r2 = [dict(x, correct=False) if x.get("arm") == "P-retrieval" and x["qid"] == "p15_01" else x for x in r]
        r2 = [dict(x, correct=True) if x.get("arm") == "P-retrieval" and x["qid"] == "p15_02" else x for x in r2]
        self.assertEqual(M.score(r2)["verdict"], "parity_holds")  # diff 1 is inside the tolerance

    def test_calls_and_prefill_are_per_question_maxima(self):
        r = records()
        r[0]["calls"] = 5
        r[0]["prefill_tokens"] = 3001
        s = M.score(r)
        self.assertEqual(s["failing"], ["calls", "prefill"])
        self.assertEqual((s["max_extra_calls"], s["max_extra_prefill"]), (2, 2001))
        r = records()
        r[0]["calls"], r[0]["prefill_tokens"] = 4, 3000  # exactly +1 call, +2000 tokens: inside
        self.assertEqual(M.score(r)["verdict"], "parity_holds")

    def test_l3_l4_in_manifest_trips(self):
        s = M.score(records(built=("L3",)))
        self.assertEqual((s["verdict"], s["failing"], s["indexes_built"]), ("phase1_gate_tripped", ["no_l3_l4"], ["L3"]))
        self.assertEqual(M.score([x for x in records() if x["kind"] != "manifest"])["failing"], ["no_l3_l4"])

    def test_rebuild_per_request_on_the_p_side_trips(self):
        r = records()
        for x in r:
            if x.get("arm") == "P-retrieval":
                x["index_builds"] = 1  # a build per question
        s = M.score(r)
        self.assertEqual(s["failing"], ["no_rebuild_per_request"])

    def test_incomplete(self):
        self.assertEqual(M.score(records(n=7))["verdict"], "incomplete")

    def test_pilot_futility_and_budget(self):
        ids = [f"p15_{i + 1:02d}" for i in range(5)]
        ok = M.pilot_check(records(), ids)
        self.assertEqual((ok["futility"], ok["over_budget"]), ([], False))
        r = records()
        for x in r:
            if x.get("arm") == "P-retrieval" and x["qid"] in ids[:3]:
                x["correct"] = False  # 2 vs 5: diff 3 > 2
        self.assertEqual(M.pilot_check(r, ids)["futility"], ["accuracy"])
        r = records()
        r[0]["prefill_tokens"] = 3100
        self.assertEqual(M.pilot_check(r, ids)["futility"], ["prefill"])
        r = records()
        r[1]["prefill_tokens"] = 5000  # B dearer: P is cheaper, not a parity failure
        self.assertEqual(M.pilot_check(r, ids)["futility"], [])
        r = records()
        for x in r:
            if x.get("kind") == "answer":
                x["wall_s"] = 100.0  # 10 answers x 100 s = 1000 s > 900 s
        self.assertTrue(M.pilot_check(r, ids)["over_budget"])


@unittest.skipUnless(HAVE_STORE, "needs the orchestrator tree and the Phase 0 store")
class ArmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.q = json.loads((D1 / "parity_questions.json").read_text())
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_arms(self, backend, **kw):
        rec = M.P0.Recorder(self.out / "results.jsonl")
        # the fake model's token bill is not a real prefill: the prefill gates are exercised in VerdictTests
        with mock.patch.object(M, "PILOT_FUTILE_PREFILL", 10**9), mock.patch.object(M, "PREFILL_TOL", 10**9):
            res = M.run_arms(rec, STORE, self.out / "parity.sqlite", backend, self.q, guard=lambda: None, **kw)
        return rec, res

    def test_every_question_plans_retrieval_on_one_mid_size_doc(self):
        rec, _ = self.run_arms(M.FakeBackend(), smoke=True)
        plans = [r for r in rec.records if r["kind"] == "plan"]
        self.assertEqual(len(plans), 20)
        for p in plans:
            self.assertTrue(p["ok"], p)
            self.assertEqual((p["path"], p["scope_reason"], p["decided_by"], p["planner_llm_called"]),
                             ("retrieval", "single_entity", "deterministic_default_stub", False))
            self.assertEqual(p["layers"], ["L0", "L1", "L2"])

    def test_full_fake_run_holds_parity_and_shows_the_saving(self):
        rec, res = self.run_arms(M.FakeBackend())
        v = res["verdict"]
        self.assertEqual(v["verdict"], "parity_holds", v)
        self.assertEqual(v["index_builds"]["P"], 4)       # one build per document
        self.assertEqual(v["index_builds"]["B"], 20)      # one per request
        self.assertEqual(v["saving"]["index_builds"], 16)
        self.assertEqual(v["indexes_built"], [])
        from src.aggregation.store import CorpusStore
        store = CorpusStore(self.out / "parity.sqlite")
        for d in M.DOCS:
            self.assertEqual({m["layer"] for m in store.manifest(M.CORPUS_PREFIX + d)}, {"L1"})
            self.assertEqual(len(store.facts(M.CORPUS_PREFIX + d)), 0)

    def test_resume_skips_finished_units(self):
        rec, _ = self.run_arms(M.FakeBackend())
        n = len(rec.records)
        b2 = M.FakeBackend()
        rec2, res = self.run_arms(b2)
        self.assertEqual(b2.calls, 0)
        self.assertEqual(res["verdict"]["verdict"], "parity_holds")
        self.assertEqual(sum(1 for r in rec2.records if r["kind"] == "verdict"), 1)
        self.assertGreaterEqual(len(rec2.records), n)

    def test_smoke_stops_after_the_first_question_both_arms(self):
        rec, res = self.run_arms(M.FakeBackend(), smoke=True)
        self.assertTrue(res["smoke"])
        a = [r for r in rec.records if r["kind"] == "answer"]
        self.assertEqual(sorted(r["arm"] for r in a), sorted(M.ARMS))
        self.assertEqual({r["qid"] for r in a}, {"p15_01"})

    def test_pilot_futility_stops_before_the_remaining_questions(self):
        gold = {q["question"]: q["key"]["numbers"][0]["value"] for q in self.q["questions"]}

        class Wrong(M.FakeBackend):  # B always right, P always wrong (the arm is set by the arm's open())
            arm = "B"

            def chat(self, prompt, **kw):
                r = super().chat(prompt, **kw)
                if r["text"].startswith("FINAL:"):
                    q = next(k for k in gold if k in prompt)
                    r["text"] = "FINAL: 1" if self.arm == "P" else f"FINAL: {gold[q]:.0f}"
                return r
        be = Wrong()
        p_open, b_open = M.PArm.open, M.BArm.open
        with mock.patch.object(M.PArm, "open", lambda a: (setattr(be, "arm", "P"), p_open(a))[1]), \
                mock.patch.object(M.BArm, "open", lambda a: (setattr(be, "arm", "B"), b_open(a))[1]):
            rec, res = self.run_arms(be)
        v = res["verdict"]
        self.assertEqual((v["verdict"], v["stage"]), ("phase1_gate_tripped", "pilot"))
        self.assertIn("accuracy", v["failing"])
        self.assertEqual(len([r for r in rec.records if r["kind"] == "answer"]), 10)  # 5 questions x 2 arms

    def test_forbidden_manifest_row_stops_the_run(self):
        from src.aggregation.store import CorpusStore
        s = M.build_parity_store(STORE, self.out / "parity.sqlite", M.DOCS)
        cid = M.CORPUS_PREFIX + M.DOCS[0]
        s.record_index(cid, s.corpus_content_hash(cid), "l3_facts_det", "", "L3", 1)
        rec, res = self.run_arms(M.FakeBackend())
        self.assertEqual(res.get("futility_stop"), "plan")
        self.assertEqual(res["forbidden_manifest_rows"], 1)
        self.assertEqual([r for r in rec.records if r["kind"] == "answer"], [])

    def test_p_arm_reuses_the_store_index_b_arm_rebuilds_per_request(self):
        from src.aggregation.registry import IndexRegistry
        from src.aggregation.store import CorpusStore
        s = M.build_parity_store(STORE, self.out / "parity.sqlite", M.DOCS)
        reg = IndexRegistry(s)
        first = M.PArm(s, reg, M.DOCS[0]); first.open()
        second = M.PArm(s, reg, M.DOCS[0]); second.open()
        self.assertEqual((first.builds, second.builds), (1, 0))
        self.assertIs(first.handle, second.handle)
        text = "\n\n".join(c["text"] or "" for c in s.chunks(M.CORPUS_PREFIX + M.DOCS[0]))
        b1, b2 = M.BArm(M.DOCS[0], text), M.BArm(M.DOCS[0], text)
        b1.search("total net revenue"); b2.search("total net revenue")
        self.assertEqual((b1.builds, b2.builds), (1, 1))
        ptr, _ = b1.search("total net revenue")[0]
        self.assertTrue(b1.read(ptr))

    def test_dry_run_cli(self):
        rc, out = M.dry_run(D1, STORE, None)
        self.assertEqual(rc, 0, out)
        self.assertTrue(out["fake_backend"])
        self.assertEqual((out["plans_ok"], out["plans"], out["paths"]), (20, 20, ["retrieval"]))


class ReadSpanTests(unittest.TestCase):
    """AMENDMENT 1 (read span, post-data) as carried into r3: one pointer parser, one 800-character cap; stdlib."""

    def test_span_constant_is_the_context_search_chunk(self):
        self.assertEqual(M.READ_SPAN_CHARS, 800)
        self.assertEqual(M.R1_READ_CHARS, 12_000)  # the amended-away value stays on record

    def test_parse_span_caps_the_read(self):
        self.assertEqual(M.parse_span("100-500"), (100, 500))
        self.assertEqual(M.parse_span(" 100-5000 "), (100, 900))
        self.assertEqual(M.parse_span("100-50"), (100, 100))
        self.assertIsNone(M.parse_span("c23f592eb1ea5048:00083"))
        self.assertIsNone(M.parse_span("nope"))

    def test_hit_text_shown_is_the_same_in_both_arms(self):
        self.assertEqual(M.snippet(7, "x" * 500), "line 7 | " + "x" * 200)

    def test_r2_artifacts_are_gone_from_the_runner(self):
        self.assertFalse(hasattr(M, "hit_window"))
        self.assertEqual((M.P_BUILDER, M.R2_P_BUILDER), ("l1_passage_bm25", "l1_bm25"))


class _Arm:
    """Minimal arm for the loop tests: pointers 'a' and 's-e' spans, read returns a long text."""
    def __init__(self):
        self.dup_reads, self.dup_searches, self.reads, self.searches = 0, 0, [], []

    def search(self, q):
        self.searches.append(q)
        return [("a", "first hit"), ("b", "second hit")]

    def read_key(self, ptr):
        return ptr.strip() or None

    def read(self, ptr):
        self.reads.append(ptr)
        return f"TEXT-OF-{ptr}"


class _Scripted:
    def __init__(self, acts):
        self.acts, self.calls, self.prefill_tokens, self.prompts = list(acts), 0, 0, []

    def chat(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return {"text": self.acts.pop(0)}


@unittest.skipUnless(HAVE_ORCH, "needs the orchestrator tree (OutOfWindow import)")
class RepeatedReadGuardTests(unittest.TestCase):
    def run_loop(self, acts, arm=None):
        arm = arm or _Arm()
        b = _Scripted(acts)
        trace = []
        out = M.retrieval_loop(b, "q?", arm, M.P0.CFG, trace)
        return arm, b, out

    def test_second_read_of_a_pointer_returns_the_note_not_the_text(self):
        arm, b, out = self.run_loop(["SEARCH: x", "READ: a", "READ: a", "READ: b", "FINAL: 1"])
        self.assertEqual(out, ("FINAL: 1", "ok"))
        self.assertEqual(arm.reads, ["a", "b"])  # the duplicate never reached the arm's read
        self.assertEqual(arm.dup_reads, 1)
        last = b.prompts[-1]
        self.assertEqual(last.count("TEXT-OF-a"), 1)
        self.assertIn(M.ALREADY_READ.format(ptr="a"), last)
        self.assertLess(len(M.ALREADY_READ.format(ptr="a")), 150)  # a short note

    def test_the_duplicate_still_costs_a_call(self):
        _, b, _ = self.run_loop(["READ: a", "READ: a", "FINAL: 1"])
        self.assertEqual(b.calls, 3)

    def test_unknown_pointer_is_not_marked_seen(self):
        class A(_Arm):
            def read_key(self, ptr):
                return None
            def read(self, ptr):
                return None
        arm, b, _ = self.run_loop(["READ: zz", "READ: zz", "FINAL: 1"], A())
        self.assertEqual(arm.dup_reads, 0)
        self.assertEqual(b.prompts[-1].count("(unknown pointer)"), 2)

    def test_guard_does_not_swallow_search_or_final(self):
        arm, b, out = self.run_loop(["SEARCH: x", "SEARCH: x", "FINAL: 7"])
        self.assertEqual(out, ("FINAL: 7", "ok"))
        self.assertEqual(arm.dup_reads, 0)


class RepeatedSearchGuardTests(unittest.TestCase):
    """AMENDMENT 2 (post-data): an identical SEARCH returns a note, in both arms, through the one shared loop."""

    def run_loop(self, acts, arm=None):
        arm = arm or _Arm()
        b = _Scripted(acts)
        return arm, b, M.retrieval_loop(b, "q?", arm, M.P0.CFG, [])

    @unittest.skipUnless(HAVE_ORCH, "needs the orchestrator tree (OutOfWindow import)")
    def test_identical_search_returns_the_note_not_the_hits(self):
        arm, b, out = self.run_loop(["SEARCH: Total Current Assets", "SEARCH: Total Current Assets", "READ: a", "FINAL: 1"])
        self.assertEqual(out, ("FINAL: 1", "ok"))
        self.assertEqual(arm.searches, ["Total Current Assets"])  # the duplicate never reached the arm
        self.assertEqual(arm.dup_searches, 1)
        self.assertIn(M.ALREADY_SEARCHED.format(q="Total Current Assets"), b.prompts[2])
        self.assertEqual(b.prompts[2].count("first hit"), 1)
        self.assertLess(len(M.ALREADY_SEARCHED.format(q="x")), 160)

    @unittest.skipUnless(HAVE_ORCH, "needs the orchestrator tree (OutOfWindow import)")
    def test_case_and_whitespace_variants_are_the_same_query_a_different_query_is_not(self):
        arm, b, _ = self.run_loop(["SEARCH: Total  Current Assets", "SEARCH: total current assets ", "SEARCH: current assets",
                                   "FINAL: 1"])
        self.assertEqual(arm.dup_searches, 1)
        self.assertEqual(arm.searches, ["Total  Current Assets", "current assets"])

    @unittest.skipUnless(HAVE_ORCH, "needs the orchestrator tree (OutOfWindow import)")
    def test_the_duplicate_still_costs_a_call_and_reads_are_independent(self):
        arm, b, _ = self.run_loop(["SEARCH: x", "SEARCH: x", "READ: a", "READ: a", "FINAL: 1"])
        self.assertEqual(b.calls, 5)
        self.assertEqual((arm.dup_searches, arm.dup_reads), (1, 1))

    @unittest.skipUnless(HAVE_ORCH, "needs the orchestrator tree (OutOfWindow import)")
    def test_an_empty_search_is_not_marked_asked(self):
        arm, _, out = self.run_loop(["SEARCH: ", "SEARCH: ", "FINAL: 2"])
        self.assertEqual(out, ("FINAL: 2", "ok"))
        self.assertEqual(arm.dup_searches, 0)


@unittest.skipUnless(HAVE_STORE, "needs the orchestrator tree and the Phase 0 store")
class AmendedArmTests(unittest.TestCase):
    """r3: both arms show the same hit text, read the same span, and are guarded identically (one shared loop)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from src.aggregation.registry import IndexRegistry
        self.store = M.build_parity_store(STORE, Path(self.tmp.name) / "p.sqlite", M.DOCS)
        self.reg = IndexRegistry(self.store)
        self.doc = M.DOCS[1]
        self.text = "\n\n".join(c["text"] or "" for c in self.store.chunks(M.CORPUS_PREFIX + self.doc))

    def tearDown(self):
        self.tmp.cleanup()

    def test_p_and_b_return_identical_pointers_and_hit_text(self):
        pa = M.PArm(self.store, self.reg, self.doc); pa.open()
        ba = M.BArm(self.doc, self.text)
        for query in ("total current assets", "Consolidated Balance Sheets", "net revenue", "income before income tax"):
            self.assertEqual(pa.search(query), ba.search(query), query)
            for ptr, _ in pa.search(query):
                self.assertEqual(pa.read(ptr), ba.read(ptr))
                self.assertLessEqual(len(pa.read(ptr)), M.READ_SPAN_CHARS)

    def test_p_pointer_provenance_is_a_stored_chunk(self):
        pa = M.PArm(self.store, self.reg, self.doc); pa.open()
        ptrs = [p for p, _ in pa.search("total current assets")]
        self.assertTrue(ptrs)
        ids = {c["chunk_id"] for c in self.store.chunks(M.CORPUS_PREFIX + self.doc)}
        self.assertTrue(all(pa.ptr_chunk[p] in ids for p in ptrs))

    def test_p_index_is_l1_only_and_built_once_across_arms(self):
        first = M.PArm(self.store, self.reg, self.doc); first.open()
        second = M.PArm(self.store, self.reg, self.doc); second.open()
        self.assertEqual((first.builds, second.builds), (1, 0))
        self.assertIs(first.handle, second.handle)
        self.assertEqual(self.store.built_layers(M.CORPUS_PREFIX + self.doc), {"L1"})
        self.assertEqual({m["builder"] for m in self.store.manifest(M.CORPUS_PREFIX + self.doc)}, {M.P_BUILDER})

    def test_b_read_is_at_most_800_chars_even_for_a_wide_pointer(self):
        ba = M.BArm(self.doc, self.text)
        ptr, _ = ba.search("total current assets")[0]
        s, e = map(int, ptr.split("-"))
        self.assertLessEqual(e - s, 800)
        self.assertLessEqual(len(ba.read(ptr)), 800)
        self.assertLessEqual(len(ba.read(f"{s}-{s + 5000}")), 800)
        self.assertEqual(ba.read_key(f"{s}-{s + 5000}"), (s, s + 800))
        self.assertIsNone(ba.read("not-a-pointer"))
        pa = M.PArm(self.store, self.reg, self.doc); pa.open()
        self.assertEqual(pa.read_key(f"{s}-{s + 5000}"), (s, s + 800))
        self.assertIsNone(pa.read("not-a-pointer"))
        self.assertIsNone(pa.read(f"{len(self.text) + 10}-{len(self.text) + 20}"))

    def test_full_loop_repeated_read_and_search_notes_in_both_arms(self):
        class Re(M.FakeBackend):  # SEARCH, SEARCH again, READ first, READ it again, then FINAL
            def chat(self, prompt, **kw):
                self.calls += 1
                acts = re.findall(r"\n> (SEARCH|READ|FINAL): ?", prompt)
                if not acts:
                    return {"text": "SEARCH: net revenue"}
                if acts == ["SEARCH"]:
                    return {"text": "SEARCH: Net  Revenue"}
                if acts == ["SEARCH", "SEARCH"]:
                    self.ptr = re.search(r"\n\[([^\]]+)\]", prompt).group(1)
                    return {"text": f"READ: {self.ptr}"}
                if acts == ["SEARCH", "SEARCH", "READ"]:
                    return {"text": f"READ: {self.ptr}"}
                return {"text": "FINAL: 1"}
        pa = M.PArm(self.store, self.reg, self.doc); pa.open()
        ba = M.BArm(self.doc, self.text)
        for arm in (pa, ba):
            out = M.retrieval_loop(Re(), "q?", arm, M.P0.CFG, [])
            self.assertEqual(out, ("FINAL: 1", "ok"))
            self.assertEqual((arm.dup_searches, arm.dup_reads), (1, 1))

    def test_answer_record_carries_dup_counters(self):
        q = json.loads((D1 / "parity_questions.json").read_text())
        rec = M.P0.Recorder(Path(self.tmp.name) / "r.jsonl")
        with mock.patch.object(M, "PILOT_FUTILE_PREFILL", 10**9), mock.patch.object(M, "PREFILL_TOL", 10**9):
            M.run_arms(rec, STORE, Path(self.tmp.name) / "p2.sqlite", M.FakeBackend(), q, guard=lambda: None, smoke=True)
        a = [r for r in rec.records if r["kind"] == "answer"]
        self.assertEqual(len(a), 2)
        self.assertTrue(all(r["dup_reads"] == 0 and r["dup_searches"] == 0 for r in a))


@unittest.skipUnless(HAVE_STORE, "needs the orchestrator tree and the Phase 0 store")
class OfflineReplayTests(unittest.TestCase):
    """AMENDMENT 2 diagnosis as tests: no model. The replay compares the r1/r2 P arm (stored chunks whole, head shown),
    B (context.search) and the r3 P arm over the same text, for the 20 label queries and the 45 SEARCH strings of the
    r1/r2 traces (committed in agg15_trace_queries_r1_r2.json)."""

    @classmethod
    def setUpClass(cls):
        import parity_replay as R
        cls.R = R
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = M.build_parity_store(STORE, Path(cls.tmp.name) / "p.sqlite", M.DOCS)
        cls.qdoc = json.loads((D1 / "parity_questions.json").read_text())
        cls.traces = json.loads((D1 / "agg15_trace_queries_r1_r2.json").read_text())
        cls.out = R.run_replay(cls.store, cls.qdoc, cls.traces)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_trace_file_is_the_two_runs(self):
        self.assertEqual(len(self.traces), 45)
        self.assertEqual({t["run"] for t in self.traces}, {"r1", "r2"})
        self.assertEqual({t["qid"] for t in self.traces}, {"p15_01", "p15_02", "p15_03", "p15_04", "p15_05"})

    def test_gold_reaches_the_top_8_on_p_wherever_it_does_on_b_all_20_questions(self):
        rows = self.out["label_queries"]["rows"]
        self.assertEqual(len(rows), 20)
        for r in rows:
            if r["B"]["in_top8"]:
                self.assertTrue(r["new_P"]["in_top8"], r)
            self.assertEqual(r["new_P"]["gold_rank"], r["B"]["gold_rank"], r)
        self.assertGreaterEqual(sum(r["B"]["in_top8"] for r in rows), 15)  # the property is not vacuous
        self.assertEqual(self.out["label_queries"]["summary"]["new_P_ranking_identical_to_B"], 20)

    def test_same_on_every_replayed_trace_query(self):
        rows = self.out["trace_queries"]["rows"]
        self.assertEqual(len(rows), 45)
        for r in rows:
            if r["B"]["in_top8"]:
                self.assertTrue(r["new_P"]["in_top8"], r)
            self.assertTrue(r["new_P_equals_B"], r)  # same pointers, same order, same spans
        self.assertEqual(self.out["trace_queries"]["summary"]["new_P_ranking_identical_to_B"], 45)

    def test_diagnosis_old_p_ranked_fine_but_showed_the_wrong_text(self):
        """The cause (see PREDECLARED_AGG15_AMENDMENT2_20261111.md): not ranking. The r1/r2 P arm found the gold chunk in the
        top 8 at least as often as B, but a hit was the HEAD of a ~3k-10k-character chunk, so the model saw the match in
        the 200 shown characters on 1 of 20 label queries (B: 5 of 20) and on 0 of 45 trace queries."""
        lab, tr = self.out["label_queries"]["summary"], self.out["trace_queries"]["summary"]
        for s in (lab, tr):
            self.assertGreaterEqual(s["old_P"]["gold_in_top8"], s["B"]["gold_in_top8"])
        self.assertLessEqual(lab["old_P"]["gold_quote_inside_shown_200"], 1)
        self.assertEqual(tr["old_P"]["gold_quote_inside_shown_200"], 0)
        self.assertGreater(lab["B"]["gold_quote_inside_shown_200"], lab["old_P"]["gold_quote_inside_shown_200"])
        self.assertGreater(lab["old_P"]["gold_hit_median_chars"], 3 * lab["B"]["gold_hit_median_chars"])

    def test_the_p15_02_loop_hit_is_unreadable_on_old_p_and_readable_on_new_p(self):
        rows = [r for r in self.out["trace_queries"]["rows"]
                if r["qid"] == "p15_02" and r["arm"] == "P-retrieval" and r["query"] == "Total Current Assets"]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["old_P"]["gold_rank"], 1)             # ranked first, yet ...
            self.assertFalse(r["old_P"]["quote_in_shown"])           # ... the quote is not in what the model saw
            self.assertGreater(r["old_P"]["quote_offset"], 5000)     # it sits 6,008 characters into a 9,789-character chunk
            self.assertTrue(r["new_P"]["quote_in_shown"] or r["new_P"]["quote_offset"] < 800)
            self.assertLess(r["new_P"]["hit_chars"], 801)


if __name__ == "__main__":
    unittest.main()
