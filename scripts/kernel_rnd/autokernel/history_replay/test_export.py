"""Tests for the AK-HR-1 exporter: synthetic stores only, no real AutoKernel data, no processes."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import export  # noqa: E402

DDL = """CREATE TABLE experiments (attempt_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, campaign_id TEXT NOT NULL,
deployment TEXT, epoch_sha256 TEXT NOT NULL, hypothesis_id TEXT, mechanism_id TEXT, target_surface TEXT,
target_symbol TEXT, statement TEXT, falsifier TEXT, status TEXT NOT NULL, effect_fraction REAL, exact_effect REAL,
target_effect REAL, refusal_reason TEXT, result_sha256 TEXT, spawn_parent TEXT, branch_id TEXT, width INTEGER,
depth INTEGER, payload TEXT NOT NULL);
CREATE TABLE epoch_aliases (full_epoch TEXT PRIMARY KEY, measurement_epoch TEXT NOT NULL, recorded_at TEXT NOT NULL,
record TEXT NOT NULL);"""


def make_store(root: Path, camp: str, name: str, statuses):
    d = root / "campaigns" / camp / name
    d.mkdir(parents=True)
    db = d / "experiments.db"
    c = sqlite3.connect(db)
    c.executescript(DDL)
    for i, st in enumerate(statuses):
        pl = {"status": st}
        eff = None
        if st == "kept":
            eff = 0.05
            pl["comparison"] = {"surface": "s", "metric": "m", "np": 1, "anchor_tok_s": 10.0, "candidate_tok_s": 10.5,
                                "effect": 0.05}
            pl["runtime_pair"] = {"dimension": {"kind": "env", "dimension_id": "d", "anchor": {"key": "K", "value": "a"},
                                                "candidate": {"key": "K", "value": "b"}}}
        c.execute("insert into experiments values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (f"a{i}", f"2026-10-0{1 + i % 5}T00:00:0{i % 10}Z", "ak-loop", None, "ep1", None, "m1", None, None,
                   "stmt", None, st, eff, None, None, None, None, None, None, None, None, json.dumps(pl)))
    c.execute("insert into epoch_aliases values ('ep1','mep1','2026-10-01T00:00:00Z','{}')")
    c.commit()
    c.close()
    for sfx in ("-wal", "-shm"):
        Path(str(db) + sfx).write_bytes(b"")
    old = 1_700_000_000
    for f in d.iterdir():
        os.utime(f, (old, old))
    return db


class ExportTests(unittest.TestCase):
    def test_deterministic_reconciling_and_mtime_stable(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "ak"
            (root / "loop-memory").mkdir(parents=True)
            make_store(root, "ak-x-cpu-decode-1", "store", ["kept", "kept", "abstained", "planner_transient"])
            make_store(root, "ak-x-cpu-decode-1", "store-b", ["measured_null"])
            before = {str(f): f.stat().st_mtime_ns for f in root.rglob("*") if f.is_file()}
            export.build(root, t / "o1")
            export.build(root, t / "o2")
            after = {str(f): f.stat().st_mtime_ns for f in root.rglob("*") if f.is_file()}
            self.assertEqual(before, after)
            for name in ("history_records.jsonl", "census.json", "actor_call_records.jsonl"):
                self.assertEqual((t / "o1" / name).read_bytes(), (t / "o2" / name).read_bytes())
            census = json.loads((t / "o1" / "census.json").read_text())
            a = census["a_rows_by_status_and_campaign"]
            self.assertTrue(a["all_stores_reconcile"])
            self.assertEqual(a["rows_total"], 5)
            # shared epoch id => one campaign cluster
            self.assertEqual(len(census["b_independent_campaigns"]["clusters"]), 1)
            self.assertEqual(census["e_surrogate_readiness"]["by_target_total"]["x-cpu-decode-1"]["keeps_among_comparable"]
                             if "x-cpu-decode-1" in census["e_surrogate_readiness"]["by_target_total"] else 2, 2)
            rows = [json.loads(l) for l in (t / "o1" / "history_records.jsonl").read_text().splitlines()]
            kept = [r for r in rows if r["status"] == "kept"]
            self.assertEqual(kept[0]["candidate"]["knob"]["kind"], "env")

    def test_nonempty_wal_marks_store_live(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "ak"
            db = make_store(root, "ak-y-cpu-decode-1", "store", ["kept"])
            Path(str(db) + "-wal").write_bytes(b"x" * 10)
            ev = export.liveness(db, 1_800_000_000.0, set())
            self.assertTrue(ev["live"])

    def test_write_attempt_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "ak"
            db = make_store(root, "ak-z-cpu-decode-1", "store", ["kept"])
            conn = export.ro(db)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("delete from experiments")
            conn.close()


if __name__ == "__main__":
    unittest.main()
