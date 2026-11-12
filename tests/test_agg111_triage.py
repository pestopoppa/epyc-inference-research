"""AGG-1.11 step 1 triage tests. Stdlib only: no model, no server, no orchestrator package (the BM25 module is loaded by
file path). The real-store tests run only when the saved r3 / AGG-1.1 stores exist on this host; they reproduce the class
counts committed in ``benchmarks/agg_phase1/agg111_triage.json``.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D0, D1 = ROOT / "agg_phase0", ROOT / "agg_phase1"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg111_triage as T  # noqa: E402

HAVE_STORES = all((r["dir"] / "store.sqlite").exists() and (r["dir"] / "results.jsonl").exists() for r in T.RUNS.values())
HAVE_BM25 = (T.ORCH / "src" / "repl_environment" / "context_search.py").exists()


def _cell(**kw):
    base = {"gold_check": {"xcheck_ok": True}, "gold_windows": {"by_quote": ["x"], "by_number": ["x"]},
            "l3_wrong_fact": False, "l3_column_error": None, "l3_gold_fact": None, "diag_verdict": "NO_FACT",
            "gold_window_ids": ["d:1#w0"], "gold_best_rank": 2, "top": [{"chunk_id": "d:1#w0", "sim": 0.9}],
            "admission_status": {"d:1#w0": "admit_sim"}, "unparseable_windows": [], "reply_evidence": "receipt_notes_only",
            "gold_reply_tokens_min": None, "secondary": []}
    base.update(kw)
    return base


class Classification(unittest.TestCase):
    def test_retrieval_outside_top_k(self):
        c = _cell(gold_window_ids=["d:9#w0"], gold_best_rank=9)
        self.assertEqual(T.classify(c)[:2], ("RETRIEVAL", "gold_window_rank_9_outside_top_6"))

    def test_retrieval_rejected_by_admission(self):
        c = _cell(admission_status={"d:1#w0": "reject_sim"})
        self.assertEqual(T.classify(c)[0], "RETRIEVAL")

    def test_unknown_gate(self):
        c = _cell(admission_status={"d:1#w0": "gate_unknown"})
        self.assertEqual(T.classify(c)[0], "UNKNOWN")

    def test_format_needs_every_admitted_gold_window_unparseable(self):
        c = _cell(unparseable_windows=["d:1#w0"], reply_evidence="server_log_mapped")
        self.assertEqual(T.classify(c)[:2], ("FORMAT", "reply_cut_at_n_predict_256"))
        two = _cell(gold_window_ids=["d:1#w0", "d:2#w0"], top=[{"chunk_id": "d:1#w0", "sim": 0.9}, {"chunk_id": "d:2#w0", "sim": 0.8}],
                    admission_status={"d:1#w0": "admit_sim", "d:2#w0": "admit_sim"}, unparseable_windows=["d:1#w0"])
        self.assertEqual(T.classify(two)[0], "EXTRACTION")      # one gold window parsed and gave nothing

    def test_extraction_sub_reasons(self):
        self.assertEqual(T.classify(_cell(gold_reply_tokens_min=2))[:2], ("EXTRACTION", "reply_empty_array"))
        self.assertEqual(T.classify(_cell(diag_verdict="UNIT_SCALE"))[:2], ("EXTRACTION", "value_right_unit_label_wrong"))

    def test_l3_beats_l4_and_records_the_l4_class(self):
        c = _cell(l3_gold_fact={"period": None}, gold_window_ids=["d:9#w0"], gold_best_rank=9)
        cls, sub, sec = T.classify(c)
        self.assertEqual((cls, sub), ("L3", "gold_value_stored_without_fy_period"))
        self.assertIn("l4_also_retrieval", sec)

    def test_gold_and_canonical(self):
        self.assertEqual(T.classify(_cell(gold_check={"xcheck_ok": False}))[0], "GOLD")
        self.assertEqual(T.classify(_cell(gold_windows={"by_quote": [], "by_number": []}))[0], "GOLD")
        self.assertEqual(T.classify(_cell(diag_verdict="HIT"))[0], "CANONICAL")


class Admission(unittest.TestCase):
    top = [{"chunk_id": "a", "sim": 1.0}, {"chunk_id": "b", "sim": 0.5}, {"chunk_id": "c", "sim": 0.4}, {"chunk_id": "d", "sim": 0.1}]

    def test_all_gated_rejected(self):
        a = T.admission(self.top, {"calls": 2, "rejected": 3})       # 1 low + 2 gate-no
        self.assertEqual([a["status"][k] for k in "abcd"], ["admit_sim", "gate_no", "gate_no", "reject_sim"])
        self.assertTrue(a["consistent"])

    def test_all_gated_admitted(self):
        a = T.admission(self.top, {"calls": 2, "rejected": 1})
        self.assertEqual([a["status"][k] for k in "bc"], ["gate_yes", "gate_yes"])

    def test_mixture_is_unknown_and_wrong_call_count_is_flagged(self):
        a = T.admission(self.top, {"calls": 2, "rejected": 2})
        self.assertEqual([a["status"][k] for k in "bc"], ["gate_unknown", "gate_unknown"])
        self.assertFalse(T.admission(self.top, {"calls": 1, "rejected": 2})["consistent"])


class Helpers(unittest.TestCase):
    def test_number_regex_is_exact(self):
        rx = T.num_re(3652.0)
        self.assertTrue(rx.search("Additions (3,652) 4,000"))
        self.assertFalse(rx.search("13,652"))
        self.assertFalse(rx.search("3,652,000"))
        self.assertTrue(T.num_re(7432.817).search("7,432.817"))

    def test_first_printed_and_short(self):
        self.assertEqual(T.first_printed("Total assets $137,100 | $138,552"), 137100.0)
        self.assertEqual(T.first_printed("Cash dividends paid (8,983) (8,729)"), 8983.0)
        self.assertEqual(T.short("94871b823d067ea7:00101#w0"), "00101#w0")

    def test_log_parser_counts_only_decode_lines(self):
        import tempfile
        log = ("slot print_timing: id 1 | task 0 | prompt eval time = 1.0 ms / 645 tokens (x)\n"
               "slot print_timing: id 1 | task 0 |        eval time = 33547.81 ms /   256 tokens (131 ms per token)\n"
               "slot print_timing: id 0 | task 5 |        eval time =   313.22 ms /     2 tokens (156 ms per token)\n")
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(log)
        self.assertEqual(T.parse_log_tokens(Path(f.name)), [256, 2])

    def test_l3_helpers(self):
        facts = [{"extractor": "l3_table.v1", "value": 38116.0, "unit": "unknown", "period": "2024",
                  "quote": "Long-term debt $ 36,377 | $ 38,116", "page": 91}]
        self.assertIsNotNone(T.l3_column_error(facts, 36377.0))
        self.assertIsNone(T.l3_gold_fact(facts, 36377.0, "2022"))
        facts2 = [{"extractor": "l3_textrow.v1", "value": 27165.0, "unit": "USD_millions", "period": None,
                   "quote": "Total assets $ 27,165 $ 27,241", "page": 53}]
        self.assertIsNotNone(T.l3_gold_fact(facts2, 27165.0, "2022"))
        self.assertIsNone(T.l3_column_error(facts2, 27165.0))       # the value IS stored, so it is not a column error

    @unittest.skipUnless(HAVE_STORES, "saved stores not on this host")
    def test_stores_open_read_only(self):
        c = T.ro(T.RUNS["b"]["dir"] / "store.sqlite")
        with self.assertRaises(sqlite3.OperationalError):
            c.execute("CREATE TABLE zz (x)")
        c.close()


@unittest.skipUnless(HAVE_STORES and HAVE_BM25, "saved stores / orchestrator BM25 module not on this host")
class SavedStores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = T.build()

    def test_class_counts_per_group(self):
        g = {k: v["summary"]["by_class"] for k, v in self.rep["groups"].items()}
        self.assertEqual({k: v for k, v in g["a"].items() if v}, {"RETRIEVAL": 5, "FORMAT": 5})
        self.assertEqual({k: v for k, v in g["b"].items() if v}, {"RETRIEVAL": 2, "L3": 1})
        self.assertEqual({k: v for k, v in g["c"].items() if v}, {"RETRIEVAL": 6, "FORMAT": 9, "EXTRACTION": 1, "L3": 10})
        self.assertEqual({k: v for k, v in g["all"].items() if v}, {"RETRIEVAL": 13, "FORMAT": 14, "EXTRACTION": 1, "L3": 11})
        self.assertEqual([self.rep["groups"][k]["summary"]["n"] for k in "abc"], [10, 3, 26])

    def test_the_groups_are_the_boxes_cells(self):
        self.assertEqual(len([c for c in self.rep["cells"] if c["group"] == "a"]), 10)      # AGG-1.3 r3: 14/24 hit
        self.assertEqual({c["doc"] for c in self.rep["cells"] if c["group"] == "b"},
                         {"ACTIVISIONBLIZZARD_2022_10K", "BOEING_2022_10K", "COCACOLA_2022_10K"})
        miss = json.loads((D1 / "agg18_miss_table.json").read_text())
        want = {(b["doc"], b["field"]) for b in miss["bad_cells"]
                if not b["correct_candidate_in_facts"] and b["canonical_state"] != "spurious"}
        self.assertEqual({(c["doc"], c["field"]) for c in self.rep["cells"] if c["group"] == "c"}, want)

    def test_replay_is_validated_against_the_saved_runs(self):
        for g in "abc":
            v = self.rep["groups"][g]["replay_validation"]
            self.assertTrue(v["ok"], (g, v))
            self.assertGreater(v["units_checked"], 200)
            m = self.rep["groups"][g]["reply_mapping"]
            self.assertTrue(m["ok"], (g, m))

    def test_truncation_is_the_unparseable_reply(self):
        pl = self.rep["pilot19_reply_lengths"]
        self.assertEqual((pl["calls"], pl["unparseable_notes"], pl["replies_cut_at_256"], pl["replies_empty_array"]), (30, 7, 7, 22))
        for g in "abc":
            m = self.rep["groups"][g]["reply_mapping"]
            self.assertGreater(m["replies_cut_at_n_predict"], 0)

    def test_pilot_cells_explain_the_futility_stop(self):
        by = {(c["doc"], c["field"]): c["primary"] for c in self.rep["cells"]}
        pilot = [by[k] for k in T.PILOT19_CELLS]
        self.assertEqual(pilot, ["FORMAT", "RETRIEVAL", "FORMAT", "FORMAT", "RETRIEVAL"])     # none is a plain re-run fill

    def test_retrieval_probe(self):
        p = self.rep["retrieval_probe_211"]
        self.assertEqual((p["cells"], p["gold_in_no_window"], p["current_top6"], p["field_top6"], p["union_cur6_field3"]),
                         (211, 0, 173, 205, 206))
        self.assertEqual(p["lost_by_union"], [])

    def test_committed_json_matches_a_fresh_run(self):
        saved = json.loads((D1 / "agg111_triage.json").read_text())
        key = lambda r: {(c["group"], c["doc"], c["field"]): (c["primary"], c["sub"], c["l4_class"]) for c in r["cells"]}  # noqa: E731
        self.assertEqual(key(saved), key(self.rep))
        self.assertEqual({g: v["summary"]["by_class"] for g, v in saved["groups"].items()},
                         {g: v["summary"]["by_class"] for g, v in self.rep["groups"].items()})


if __name__ == "__main__":
    unittest.main()
