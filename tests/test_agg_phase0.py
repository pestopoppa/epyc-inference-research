"""AGG-0.1 set + scorer tests (stdlib unittest; no model, no network)."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "benchmarks" / "agg_phase0"
sys.path.insert(0, str(D))
import agg_phase0 as A  # noqa: E402
import run_phase0 as R  # noqa: E402


def _fill(sheet: Path, values: dict) -> None:
    with sheet.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        v = values.get((r["doc_name"], r["metric"]), 1000.0)
        if v is None:
            r["status"], r["value_usd_m"] = "na", ""
        else:
            r["status"], r["value_usd_m"] = "filled", str(v)
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, A.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


class SetTests(unittest.TestCase):
    def test_frozen_set_shape(self):
        self.assertEqual(len(A.DOCS), 10)
        self.assertEqual(len(set(A.DOCS)), 10)
        self.assertEqual(len(A.AGG_QUESTIONS), 20)
        self.assertEqual(len(A.SINGLE_FACT_IDS), 10)
        golds = [json.loads(l) for l in (D / "golds.jsonl").read_text().splitlines()]
        single = [json.loads(l) for l in (D / "single_fact.jsonl").read_text().splitlines()]
        self.assertTrue(all(g["verified"] is False for g in golds + single))
        ver = json.loads((D / "verification_record.json").read_text())["questions"]
        self.assertEqual(len(ver), 30)
        self.assertFalse(any(v["verified"] for v in ver.values()))

    def test_predeclared_matches_code(self):
        text = (D / "PREDECLARED.md").read_text()
        self.assertIn("M1 < 0.70", text)
        self.assertIn("> 2.0", text)
        self.assertIn("> 150", text)
        self.assertIn("< 25 pp", text)
        self.assertIn("> 8", text)
        self.assertEqual(A.KILL["m1_min"], 0.70)
        self.assertEqual(A.KILL["m2_agg_margin_pp_min"], 25.0)

    def test_gold_derivation_and_anchors(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for f in ("facts_sheet.csv", "single_fact.jsonl", "corpus.json"):
                shutil.copy(D / f, t / f)
            vals = {("JPMORGAN_2022_10K", "total_current_assets"): None,
                    ("JPMORGAN_2022_10K", "total_current_liabilities"): None,
                    ("PEPSICO_2022_10K", "revenue"): 86392.0,
                    ("AES_2022_10K", "net_income"): -546.0,
                    ("AMERICANWATERWORKS_2022_10K", "total_current_assets"): 1000.0,
                    ("AMERICANWATERWORKS_2022_10K", "total_current_liabilities"): 2561.0}
            _fill(t / "facts_sheet.csv", vals)
            golds = {g["id"]: g for g in A.derive_golds(t)}
            self.assertEqual(golds["agg01"]["gold"], "PepsiCo")
            self.assertEqual(golds["agg07"]["gold"], 1)
            self.assertEqual(golds["agg16"]["excluded_na"], ["JPMORGAN_2022_10K"])
            self.assertEqual(golds["agg17"]["gold"], "American Water Works")
            self.assertTrue(all(g["verified"] is False for g in golds.values()))
            anchors = {a["doc"] + a["expr"]: a for a in
                       json.loads((t / "anchor_checks.json").read_text())}
            self.assertIs(anchors["AMERICANWATERWORKS_2022_10Ktotal_current_assets-total_current_liabilities"]["agrees"], True)
            self.assertIs(anchors["AES_2022_10Knet_income"]["agrees"], True)
            self.assertIn("agg20", (t / "GOLD_REVIEW.md").read_text())

    def test_committed_golds_are_unverified_drafts(self):
        golds = [json.loads(l) for l in (D / "golds.jsonl").read_text().splitlines()]
        self.assertEqual(len(golds), 20)
        self.assertTrue(all(g["verified"] is False for g in golds))

    def test_preflight_fails_closed_today(self):
        ok, info = R.preflight(D)
        self.assertFalse(ok)
        self.assertIn("no verified aggregation questions", info["problems"])
        self.assertRegex(info["predeclared_sha256"], r"^[0-9a-f]{64}$")


def _ans(arm, qid, qclass, correct, calls=3, pre=10000):
    return {"kind": "answer", "arm": arm, "qid": qid, "qclass": qclass, "correct": correct,
            "calls": calls, "prefill_tokens": pre}


class ScoreTests(unittest.TestCase):
    def base(self):
        recs = [{"kind": "ingest", "prefill_tokens": 1_100_000, "corpus_tokens": 1_000_000,
                 "llm_calls": 250, "n_chunks": 250}, {"kind": "m1", "gold_facts": 100, "found": 80}]
        recs += [_ans("P-large", f"a{i}", "agg", i < 16) for i in range(20)]
        recs += [_ans("B1", f"a{i}", "agg", i < 8) for i in range(20)]
        recs += [_ans("P-small", f"s{i}", "single", True, 2, 150_500) for i in range(10)]
        recs += [_ans("B-today", f"s{i}", "single", True, 1, 150_000) for i in range(10)]
        recs += [{"kind": "manifest", "arm": "P-small", "qid": f"s{i}", "built_layers": ["L0"]}
                 for i in range(10)]
        return recs

    def test_go(self):
        s = A.score(self.base())
        self.assertEqual(s["kill_gate_tripped"], [])
        self.assertEqual(s["verdict"], "go_phase_1")
        self.assertAlmostEqual(s["M1"], 0.8)

    def test_each_gate_trips(self):
        r = self.base()
        r[1]["found"] = 60
        self.assertIn("M1", A.score(r)["kill_gate_tripped"])
        r = self.base()
        r[0]["llm_calls"] = 400
        self.assertIn("M3", A.score(r)["kill_gate_tripped"])
        r = [x if not (x.get("arm") == "B1") else dict(x, correct=True) for x in self.base()]
        self.assertIn("M2", A.score(r)["kill_gate_tripped"])
        r = [dict(x, calls=9) if x.get("arm") == "P-large" else x for x in self.base()]
        self.assertIn("M4", A.score(r)["kill_gate_tripped"])
        r = self.base() + [{"kind": "manifest", "arm": "P-small", "qid": "s0", "built_layers": ["L3"]}]
        s = A.score(r)
        self.assertIn("M5", s["kill_gate_tripped"])
        self.assertEqual(s["verdict"], "superseded_v2_not_viable")

    def test_empty_records_fail_closed(self):
        s = A.score([])
        self.assertEqual(sorted(s["kill_gate_tripped"]), ["M1", "M2", "M3", "M4", "M5"])


if __name__ == "__main__":
    unittest.main()
