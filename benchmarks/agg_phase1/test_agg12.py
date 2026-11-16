"""Unit tests for run_agg12 (zero model, zero server): verdict rules, scorer, verifier, SQL mirror, grammar text. Run: python3 -I -m unittest test_agg12"""
import sqlite3
import unittest
from pathlib import Path

import run_agg12 as R

Q = [{"id": f"q{i}", "type": "singledoc_rolls", "question": "?", "gold": str(i)} for i in range(10)]


def recs(arm, correct_ids, *, status="ok", calls=2, prefill=500, missing=()):
    out = []
    for q in Q:
        if q["id"] in missing:
            continue
        ans = f"FINAL: {q['gold']}" if q["id"] in correct_ids else "FINAL: 999"
        out.append({"kind": "answer", "arm": arm, "qid": q["id"], "final": True, "status": status, "answer": ans,
                    "calls": calls, "prefill_tokens": prefill, "correct": True})   # 'correct' must be ignored
    return out


class T(unittest.TestCase):
    def v(self, p, b, **kw):
        ids = [q["id"] for q in Q]
        r = recs("P-large", set(ids[:p])) + recs("B1", set(ids[10 - b:]) if b else set()) + recs("RET", set())
        return R.verdict(r, Q, **kw)

    def test_score(self):
        self.assertTrue(R.score_count("5", "x FINAL: 5"))
        self.assertFalse(R.score_count("5", "FINAL: 6"))
        self.assertIsNone(R.score_count("5", "FINAL: many"))
        self.assertFalse(R.score_count("5", "FINAL: 5.5"))
        self.assertTrue(R.score_count("2", "FINAL: 2%"))

    def test_tie_within_one(self):
        self.assertEqual(self.v(4, 3)["verdict"], "answered:tied_within_1")

    def test_p_large_wins_needs_9_discordant_and_z3(self):
        r = self.v(9, 0)           # b=9, c=0, z=3
        self.assertEqual(r["verdict"], "answered:p_large_wins")
        self.assertAlmostEqual(r["m2"]["z"], 3.0)
        r = self.v(8, 0)           # b+c = 8 < 9: UNRESOLVED even though z = 2.83
        self.assertEqual(r["verdict"], "unresolved")

    def test_b1_wins(self):
        self.assertEqual(self.v(0, 9)["verdict"], "answered:b1_wins")

    def test_unresolved(self):
        self.assertEqual(self.v(6, 2)["verdict"], "unresolved")

    def test_incomplete_and_futility(self):
        ids = [q["id"] for q in Q]
        r = R.verdict(recs("P-large", set(ids), missing={"q9"}) + recs("B1", set()), Q)
        self.assertEqual(r["verdict"], "incomplete")
        self.assertEqual(self.v(5, 5, futility="wall_cap_1800s")["verdict"], "futility:wall_cap_1800s")

    def test_cap_and_unparseable_counted_wrong(self):
        ids = [q["id"] for q in Q]
        r = recs("P-large", set(ids)) + recs("B1", set(ids), status="budget_exhausted")
        out = R.verdict(r, Q)
        self.assertEqual(out["arms"]["B1"]["n_cap"], 10)
        self.assertEqual(out["arms"]["B1"]["n_correct"], 0)
        self.assertEqual(out["verdict"], "answered:p_large_wins")
        r = recs("P-large", set()) + recs("B1", set())
        for x in r:
            x["answer"] = "no idea"
        out = R.verdict(r, Q)
        self.assertEqual(out["arms"]["P-large"]["n_unparseable"], 10)

    def test_m1_null_and_m4(self):
        r = self.v(5, 5)
        self.assertIsNone(r["m1"])
        self.assertIn("gold-evidence", r["m1_reason"])
        self.assertTrue(r["m4"]["within_phase0_bounds"])
        ids = [q["id"] for q in Q]
        bad = R.verdict(recs("P-large", set(ids), calls=4) + recs("B1", set()), Q)
        self.assertFalse(bad["m4"]["within_phase0_bounds"])
        bad = R.verdict(recs("P-large", set(ids), prefill=2511) + recs("B1", set()), Q)
        self.assertFalse(bad["m4"]["within_phase0_bounds"])

    def test_verify_event(self):
        lines = {1: "Matt: go ahead", 2: "Liam: Uh. 18.", 3: "Marisha: I cast Stoneskin on Grog."}
        ch = {"Liam", "Marisha"}
        ok = lambda k, e: R.verify_event(k, e, lines, (1, 3), ch)[0]  # noqa: E731
        self.assertTrue(ok("roll", {"line": 2, "character": "Liam", "roll_type": "Stealth", "value": 18}))
        self.assertFalse(ok("roll", {"line": 2, "character": "Liam", "roll_type": "Stealth", "value": 8}))      # 8 is inside 18
        self.assertFalse(ok("roll", {"line": 9, "character": "Liam", "roll_type": "Stealth", "value": 18}))
        self.assertFalse(ok("roll", {"line": 2, "character": "Nobody", "roll_type": "Stealth", "value": 18}))
        self.assertTrue(ok("spell", {"line": 3, "character": "Marisha", "spell": "Stoneskin"}))
        self.assertFalse(ok("spell", {"line": 3, "character": "Marisha", "spell": "Fireball"}))

    def test_sql_mirror(self):
        c = sqlite3.connect(":memory:")
        R.write_events(c, [{"kind": "roll", "line": 1, "character": "Grog", "roll_type": "Nature", "value": 5, "natural": True, "block": "b"},
                           {"kind": "spell", "line": 2, "character": "Tiberius", "spell": "Stoneskin", "block": "b"}], {"Grog": "Travis", "Tiberius": "Orion"})
        cat = R.EventCatalog(c)
        good = ["SELECT COUNT(*) FROM events WHERE kind = 'roll' AND roll_type = 'Nature'",
                "SELECT COUNT(DISTINCT character) FROM events WHERE kind = 'spell' AND spell = 'Stoneskin'",
                "SELECT ROUND(100.0 * SUM(CASE WHEN value = 4 THEN 1 ELSE 0 END) / COUNT(*)) FROM events WHERE kind = 'roll'",
                "SELECT COUNT(*) FROM events WHERE kind = 'roll' AND value = 1 AND nat = 1",
                "SELECT character, COUNT(*) FROM events WHERE kind = 'spell' GROUP BY character ORDER BY COUNT(*) DESC LIMIT 3"]
        bad = ["SELECT COUNT(*) FROM events WHERE kind = 'roll'; DROP TABLE events", "SELECT * FROM events", "SELECT COUNT(*) FROM events WHERE kind = 'roll' AND spell = 'Fireball'",
               "SELECT COUNT(*) FROM oolong_events WHERE kind = 'roll'"]
        for s in good:
            self.assertTrue(R.sql_ok(s, cat)[0], s)
            R.create_events_view(c)
            c.execute(s).fetchall()
        for s in bad:
            self.assertFalse(R.sql_ok(s, cat)[0], s)
        self.assertEqual(c.execute("SELECT player FROM oolong_events WHERE kind='roll'").fetchone()[0], "Travis")

    def test_blocks_and_transcript(self):
        text = "Matt plays the character DM.\nLaura plays the character Vex'ahlia.\n\n[START OF EPISODE]\nMatt: hi\nLaura: 12.\n\n[END OF EPISODE]\n"
        tr = R.parse_transcript(text)
        self.assertEqual(tr["lines"], [(1, "Matt: hi"), (2, "Laura: 12.")])
        self.assertEqual(R.characters_of(tr["mapping"]), {"DM": "Matt", "Vexahlia": "Laura"})
        self.assertEqual(len(R.blocks([(i, "x" * 100) for i in range(1, 121)], 6000)), 3)

    def test_gbnf_text(self):
        g = R.extract_gbnf(["Grog", "DM"])
        self.assertIn('"Unknown"', g)
        self.assertTrue(g.startswith("root ::="))


if __name__ == "__main__":
    unittest.main()
