"""AGG-1.1 Phase 1 frozen-set tests (stdlib unittest; no model, no network, no PDF parsing)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D1 = ROOT / "agg_phase1"
D0 = ROOT / "agg_phase0"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg_phase0 as A0  # noqa: E402
import build_phase1 as B  # noqa: E402
import phase1_questions as Q  # noqa: E402

PHASE0_PREDECLARED_SHA = "976a5fe87e3db33d6a5e4547ee4bb6cae324f44d107980342b62f7b6212b8bc9"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class FrozenSetTests(unittest.TestCase):
    def test_phase0_predeclared_untouched(self):
        self.assertEqual(sha(D0 / "PREDECLARED.md"), PHASE0_PREDECLARED_SHA)

    def test_corpus_manifest(self):
        c = json.loads((D1 / "corpus.json").read_text())
        self.assertEqual(c["n_docs"], 28)
        self.assertEqual(len(c["docs"]), 28)
        self.assertEqual(len({d["doc_name"] for d in c["docs"]}), 28)
        self.assertEqual(c["n_phase0_docs"], 10)
        self.assertEqual(c["manifest_digest_sha256"], B.digest_docs(c["docs"]))
        p0 = {d["doc_name"]: d for d in json.loads((D0 / "corpus.json").read_text())["docs"]}
        for d in c["docs"]:
            self.assertEqual(len(d["sha256"]), 64)
            self.assertGreater(d["bytes"], 0)
            self.assertGreater(d["pages"], 0)
            self.assertTrue(d["source_url"].startswith("https://github.com/patronus-ai/financebench/raw/main/pdfs/"))
            if d["phase0"]:
                self.assertEqual(d["sha256"], p0[d["doc_name"]]["sha256"])
        self.assertEqual({d["doc_name"] for d in c["docs"] if d["phase0"]}, set(p0))

    def test_question_set_shape(self):
        self.assertEqual(len(Q.CONTINUITY), 20)
        self.assertGreaterEqual(len(Q.NEW), 30)
        ids = [q["id"] for q in Q.ALL]
        self.assertEqual(len(ids), len(set(ids)))
        p0 = {q["id"]: q for q in A0.AGG_QUESTIONS}
        for q in Q.CONTINUITY:  # continuity = Phase 0 question set, only agg18 text/exclusions differ
            b = p0[q["id"]]
            self.assertEqual((q["expr"], q["op"], q.get("t")), (b["expr"], b["op"], b.get("t")))
            if q["id"] != "agg18":
                self.assertEqual(q["q"], b["q"])
        self.assertTrue(next(q for q in Q.CONTINUITY if q["id"] == "agg18")["text_changed_from_phase0"])
        scorer_ops = {"argmax", "argmin", "sum", "count_gt", "count_lt", "top3"}
        self.assertTrue(all(q["op"] in scorer_ops for q in Q.ALL))

    def test_sheet_shape_and_phase0_cells_unchanged(self):
        rows = list(csv.DictReader((D1 / "facts_sheet.csv").open(newline="")))
        self.assertEqual(len(rows), 224)
        self.assertEqual(sum(r["status"] == "filled" for r in rows), 211)
        self.assertEqual(sum(r["status"] == "na" for r in rows), 13)
        p0 = {(r["doc_name"], r["metric"]): r for r in csv.DictReader((D0 / "facts_sheet.csv").open(newline=""))}
        new = {(r["doc_name"], r["metric"]): r for r in rows}
        for k, r in p0.items():
            for col in ("value_usd_m", "status", "page", "quote", "verified"):
                self.assertEqual(new[k][col], r[col], k)
        for r in rows:
            if r["status"] == "filled":
                self.assertTrue(r["quote"] and r["page"].isdigit(), r)
            self.assertIn(r["verified"], ("true", "false"))

    def test_golds_recompute_and_all_verified(self):
        self.assertEqual(B.check(D1), 0)
        golds = [json.loads(l) for l in (D1 / "golds.jsonl").read_text().splitlines()]
        self.assertEqual(len(golds), len(Q.ALL))
        self.assertTrue(all(g["verified"] is True and g["gold"] is not None for g in golds))
        self.assertTrue(all(not g["margin"]["near_tie"] for g in golds))
        ver = json.loads((D1 / "verification_record.json").read_text())["questions"]
        self.assertEqual(sum(v["verified"] for v in ver.values()), 58)
        for g in golds:  # every gold cites doc, metric, value and PDF page for every contributing cell
            self.assertTrue(g["evidence"])
            for e in g["evidence"]:
                self.assertEqual(set(e), {"doc", "metric", "value", "page"})

    def test_golds_accepted_by_phase0_scorer(self):
        sys.path.insert(0, str(D0))
        import phase0_arms as P  # noqa: PLC0415
        for l in (D1 / "golds.jsonl").read_text().splitlines():
            g = json.loads(l)
            gold = g["gold"]
            if g["op"] == "top3":
                ans = "FINAL: " + ", ".join(gold)
            elif g["op"] in ("argmax", "argmin"):
                ans = f"FINAL: {gold}"
            elif g["op"] == "sum":
                ans = f"FINAL: {gold:,.1f} million"
            else:
                ans = f"FINAL: {gold}"
            self.assertIs(P.score_agg(g, ans), True, g["id"])

    def test_xcheck_clean(self):
        x = json.loads((D1 / "facts_sheet_xcheck.json").read_text())
        self.assertEqual(x["summary"]["fails"], 0)
        self.assertEqual(x["summary"]["cells"], 224)

    def test_predeclared_pins_frozen_inputs_and_gates(self):
        text = (D1 / "PREDECLARED.md").read_text()
        for f in ("corpus.json", "facts_sheet.csv", "facts_sheet_xcheck.json", "aggregation_questions.json",
                  "golds.jsonl", "verification_record.json"):
            self.assertIn(sha(D1 / f), text, f)
        self.assertIn(json.loads((D1 / "corpus.json").read_text())["manifest_digest_sha256"], text)
        self.assertIn(PHASE0_PREDECLARED_SHA, text)
        for needle in ("M1 < 0.75", "> 2.0", "> 150", "median P-large calls per question > 8", "2,510",
                       "futility-only", "post-data", "MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K",
                       "phase1_gate_tripped"):
            self.assertIn(needle, text, needle)
        self.assertEqual(A0.KILL["m3_prefill_ratio_max"], 2.0)
        self.assertEqual(A0.KILL["m3_calls_per_100_chunks_max"], 150)
        self.assertEqual(A0.KILL["median_calls_max"], 8)


if __name__ == "__main__":
    unittest.main()
