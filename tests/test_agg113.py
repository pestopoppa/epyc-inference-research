"""AGG-1.13 step A frozen-set tests (stdlib unittest; no model, no network, no PDF parsing)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
D1 = ROOT / "agg_phase1"
D13 = D1 / "agg113"
D0 = ROOT / "agg_phase0"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D13))
import agg_phase0 as A0  # noqa: E402
import agg113_questions as Q  # noqa: E402
import build_agg113 as B  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Agg113FrozenSetTests(unittest.TestCase):
    def test_predeclared_sidecar_matches(self):
        side = (D13 / "PREDECLARED_AGG113.sha256").read_text().split()
        self.assertEqual(side[0], sha(D13 / "PREDECLARED_AGG113.md"))
        self.assertEqual(side[1], "PREDECLARED_AGG113.md")

    def test_selection_rule_and_no_design_overlap(self):
        sel = json.loads((D13 / "selection.json").read_text())
        in28 = {d["doc_name"] for d in json.loads((D1 / "corpus.json").read_text())["docs"]}
        in13 = {d["doc_name"] for d in json.loads((D1 / "corpus_agg13.json").read_text())["docs"]}
        self.assertEqual(len(sel["selected"]), sel["n_eligible"])
        self.assertLess(sel["n_eligible"], sel["target_n_docs"])  # all eligible taken: the shortfall is declared
        self.assertFalse(set(sel["selected"]) & in28)
        self.assertFalse(set(sel["selected"]) & in13)
        by = {c["doc_name"]: c for c in sel["candidates"]}
        for d in sel["selected"]:
            self.assertTrue(by[d]["own_label_fiscal_2022"], d)
            self.assertFalse(by[d]["named_in_agg1x_design_files"], d)
        self.assertIn("FOOTLOCKER_2022_10K", sel["excluded_candidates"])  # fiscal 2021 by its own label
        self.assertEqual(sel["pilot_docs"], sorted(sel["selected"], key=B.salted)[:3])
        self.assertEqual(sel["pilot_docs"], sel["pilot_order_by_salted_hash"][:3])
        # a recomputation of the rule from the committed inputs gives the committed selection
        self.assertEqual(B.build_selection()["selected"], sel["selected"])

    def test_corpus_manifest(self):
        c = json.loads((D13 / "corpus.json").read_text())
        sel = json.loads((D13 / "selection.json").read_text())
        self.assertEqual(c["n_docs"], 10)
        self.assertEqual(len(c["docs"]), 10)
        self.assertEqual(sorted(d["doc_name"] for d in c["docs"]), sel["selected"])
        self.assertEqual(c["manifest_digest_sha256"], B.digest_docs(c["docs"]))
        self.assertEqual(c["selection_sha256"], sha(D13 / "selection.json"))
        self.assertEqual(c["fiscal_year"], 2022)
        for d in c["docs"]:
            self.assertEqual(len(d["sha256"]), 64)
            self.assertGreater(d["bytes"], 0)
            self.assertGreater(d["pages"], 0)
            self.assertTrue(d["source_url"].startswith("https://github.com/patronus-ai/financebench/raw/main/pdfs/"))

    def test_sheet_shape(self):
        rows = list(csv.DictReader((D13 / "facts_sheet.csv").open(newline="")))
        self.assertEqual(len(rows), 80)
        self.assertEqual(sum(r["status"] == "filled" for r in rows), 76)
        self.assertEqual(sum(r["status"] == "na" for r in rows), 4)
        self.assertEqual({r["metric"] for r in rows}, set(A0.METRICS))
        self.assertEqual({r["doc_name"] for r in rows}, {d["doc_name"] for d in json.loads((D13 / "corpus.json").read_text())["docs"]})
        for r in rows:
            self.assertTrue(r["quote"] and r["page"].isdigit(), r)  # every cell, NA included, has a page and a verbatim quote
            self.assertEqual(r["verified"], "true", r)  # an unverifiable cell would have been excluded, not carried
            if r["status"] == "filled":
                float(r["value_usd_m"])

    def test_golds_recompute_and_all_verified(self):
        self.assertEqual(B.check(D13), 0)
        golds = [json.loads(l) for l in (D13 / "golds.jsonl").read_text().splitlines()]
        qs = json.loads((D13 / "aggregation_questions.json").read_text())["questions"]
        self.assertEqual(len(golds), 30)
        self.assertEqual([g["id"] for g in golds], [q["id"] for q in qs])
        self.assertEqual([g["id"] for g in golds], [f"n13_{i:02d}" for i in range(1, 31)])
        self.assertTrue(all(g["verified"] is True and g["gold"] is not None for g in golds))
        self.assertTrue(all(not g["margin"]["near_tie"] and g["margin"]["min_rel_gap"] >= Q.MARGIN_MIN for g in golds if g["margin"]["min_rel_gap"] is not None))
        ver = json.loads((D13 / "verification_record.json").read_text())
        self.assertEqual(sum(v["verified"] for v in ver["questions"].values()), 30)
        self.assertEqual(ver["dropped_templates"], [])
        scorer_ops = {"argmax", "argmin", "sum", "count_gt", "count_lt", "top3"}
        self.assertTrue(all(g["op"] in scorer_ops for g in golds))
        fam = {}
        for g in golds:
            fam[g["family"]] = fam.get(g["family"], 0) + 1
        self.assertEqual(fam, Q.QUOTA)
        cells = set()
        for g in golds:
            self.assertTrue(g["evidence"])
            for e in g["evidence"]:
                self.assertEqual(set(e), {"doc", "metric", "value", "page"})
                cells.add((e["doc"], e["metric"]))
        filled = {(r["doc_name"], r["metric"]) for r in csv.DictReader((D13 / "facts_sheet.csv").open(newline="")) if r["status"] == "filled"}
        self.assertEqual(cells, filled)  # every filled cell feeds a question: the M1 denominator is all 76

    def test_golds_recompute_from_sheet_by_independent_arithmetic(self):
        sheet, company, docs = B.load_sheet(D13 / "facts_sheet.csv")
        g = {json.loads(l)["id"]: json.loads(l) for l in (D13 / "golds.jsonl").read_text().splitlines()}
        v = lambda d, m: float(sheet[(d, m)]["value_usd_m"])  # noqa: E731
        # n13_09: count of revenue above $30 billion, recomputed here without the library
        self.assertEqual(g["n13_09"]["gold"], sum(1 for x in docs if v(x, "revenue") > g["n13_09"]["threshold"]))
        top = sorted(docs, key=lambda x: -v(x, "revenue"))[:3]
        self.assertEqual(g["n13_10"]["gold"], [company[x] for x in top])
        self.assertEqual(g["n13_08"]["gold"], company[min(docs, key=lambda x: v(x, "net_income"))])
        self.assertEqual(g["n13_17"]["gold"], sum(1 for x in docs if v(x, "operating_cash_flow") - v(x, "capex") < 0))
        wc = sum(v(x, "total_current_assets") - v(x, "total_current_liabilities") for x in docs)
        self.assertGreater(wc, 0)

    def test_golds_accepted_by_phase0_scorer(self):
        import phase0_arms as P  # noqa: PLC0415
        for l in (D13 / "golds.jsonl").read_text().splitlines():
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
        x = json.loads((D13 / "facts_sheet_xcheck.json").read_text())
        self.assertEqual(x["summary"]["fails"], 0)
        self.assertEqual(x["summary"]["cells"], 80)
        self.assertEqual(x["summary"]["checked"], 80)
        self.assertTrue(all(c["ok"] for c in x["cells"] if "ok" in c))

    def test_predeclared_pins_frozen_inputs_and_rules(self):
        text = (D13 / "PREDECLARED_AGG113.md").read_text()
        for f in ("financebench_document_information.jsonl", "selection.json", "corpus.json", "facts_sheet.csv",
                  "facts_sheet_xcheck.json", "aggregation_questions.json", "golds.jsonl", "verification_record.json"):
            self.assertIn(sha(D13 / f), text, f)
        self.assertIn(sha(D1 / "verify_phase1_cells.py"), text)
        self.assertIn(json.loads((D13 / "corpus.json").read_text())["manifest_digest_sha256"], text)
        for needle in ("737bf0535d1abecba154a2a22fbba0d910821ed8", "agg_pipeline.v1", "agg_pipeline.v2", "agg11.v0", "agg110.v1",
                       "2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af",
                       "22479709afbe5110115f6b2d1a951e799b8bb8752fabc5c51681c4aa30f59c1c",
                       "b5dec670f649b50af48ec4971e7503018d84c0f831c6bb859613b53b2348c584",
                       "WALMART_2022_10K", "ULTABEAUTY_2023_10K", "BLOCK_2022_10K", "1,800 s", "futility-only", "POST-DATA",
                       "in-sample", "Nothing on these documents may be inspected beyond gold building",
                       "2,510", "150", "NOT reachable", "10 documents, not 15"):
            self.assertIn(needle, text, needle)
        sel = json.loads((D13 / "selection.json").read_text())
        for d in sel["selected"]:
            self.assertIn(d, text)
        self.assertEqual(A0.KILL["m3_prefill_ratio_max"], 2.0)
        self.assertEqual(A0.KILL["m3_calls_per_100_chunks_max"], 150)
        self.assertEqual(A0.KILL["median_calls_max"], 8)

    def test_agg11_agg112_files_untouched(self):
        self.assertEqual(sha(D1 / "PREDECLARED.md"), "2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af")
        self.assertEqual(sha(D1 / "PREDECLARED_AGG112.md"), "22479709afbe5110115f6b2d1a951e799b8bb8752fabc5c51681c4aa30f59c1c")

    def test_pick_threshold(self):
        t = Q.pick_threshold([1.0, 2.0, 8.0, 9.0])
        self.assertIn(t, Q.grid() + [int(x) for x in Q.grid() if float(x).is_integer()])
        self.assertTrue(2 < t < 8)  # splits 2/2 with the widest gap


if __name__ == "__main__":
    unittest.main()
