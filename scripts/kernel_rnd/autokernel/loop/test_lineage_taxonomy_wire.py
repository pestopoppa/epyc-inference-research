import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from autokernel.loop import lineage_taxonomy_wire as wire

COMMIT = "b" * 40
ROWS = [("a1", "sym", "fix the race overflow in the reduction", "kept", 0.04),
        ("a2", "other", "tune the tile size constant", "measured_null", 0.0),
        ("a3", "sym", "fix the race overflow in the reduction", "regression", -0.01),
        ("a4", "z", "zzz qqq", "rejected", None)]


class WireTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / "experiments.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE experiments(attempt_id,recorded_at,campaign_id,target_symbol,"
                    "statement,status,effect_fraction)")
        for i, (a, sym, st, status, eff) in enumerate(ROWS):
            con.execute("INSERT INTO experiments VALUES(?,?,?,?,?,?,?)",
                        (a, f"2026-11-0{i + 1}", "c", sym, st, status, eff))
        con.commit()
        con.close()

    def test_receipt_binds_digest_join_counts_and_unlabeled(self):
        r = wire.build([self.db], producer_commit=COMMIT, observed_at="2026-11-22T00:00:00+00:00")
        (s,) = r["stores"]
        self.assertEqual(s["n"], 4)
        self.assertEqual(len(s["sha256"]), 64)
        self.assertEqual(s["cycling"]["recycled"], 1)
        self.assertEqual(sum(s["label_counts"].values()), 4)
        self.assertEqual(s["unlabeled"], 1)
        self.assertEqual([j["attempt_id"] for j in s["join"]], ["a1", "a2", "a3", "a4"])
        self.assertTrue(s["join"][2]["recycled"])
        self.assertEqual(r["detector"]["version"], wire.DETECTOR_VERSION)
        (m,) = r["belief_measurements"]
        self.assertEqual(m["reps"], 4)
        self.assertEqual(m["attestation_sha256"], s["sha256"])

    def test_publish_is_immutable_and_store_untouched(self):
        before = self.db.read_bytes()
        kw = dict(producer_commit=COMMIT, observed_at="2026-11-22T00:00:00+00:00")
        p = wire.publish(self.root / "out", [self.db], **kw)
        self.assertEqual(json.loads(p.read_text())["schema"], wire.SCHEMA)
        self.assertEqual(wire.publish(self.root / "out", [self.db], **kw), p)
        self.assertEqual(self.db.read_bytes(), before)

    def test_bad_producer_commit_refused(self):
        with self.assertRaises(ValueError):
            wire.build([self.db], producer_commit="abc", observed_at="x")


if __name__ == "__main__":
    unittest.main()
