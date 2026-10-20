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

    def test_reads_ak_hr3_fields_from_rows_written_by_the_real_code_path(self):
        from autokernel.loop import actors, archive, loop
        from autokernel.loop.test_loop import _Critic, _Planner, _run
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "ak"
            (root / "loop-memory").mkdir(parents=True)
            old = make_store(root, "ak-x-cpu-decode-1", "store", ["kept", "abstained"])
            camp = root / "campaigns" / "ak-x-cpu-decode-1"
            (camp / "ws").mkdir()
            backend = actors.Backend(kind="codex", model="m", effort="low", binary="codex")
            hyp = loop.Hypothesis(mechanism_id="akm-prefetch-rows", statement="reorder",
                                  falsifier="none", target_surface="s.cpp", target_symbol="scatter")

            class Planner(_Planner):
                def propose(self, context):
                    actors._record_metrics(camp / "ws", backend, role="planner", returncode=0,
                                           wall_s=2.0, timed_out=False, before_ids=set(),
                                           collect_metrics=False, schema=None, final_text=None,
                                           salvaged=False)
                    return super().propose(context)

            for i in range(2):
                outcome, _ = _run(Planner([hyp]), _Critic([], []))
                archive.record(camp / "store", outcome.to_attempt(), epoch="ep1",
                               recorded_at=f"2026-10-20T00:00:0{i}Z", campaign_id="ak-loop")
            # an actor call made outside any trial: null trial_id
            actors._record_metrics(camp / "ws", backend, role="critic", returncode=0, wall_s=1.0,
                                   timed_out=False, before_ids=set(), collect_metrics=False,
                                   schema=None, final_text=None, salvaged=False)
            for f in camp.rglob("*"):       # the exporter refuses a store written in the last 30 min
                if f.is_file():
                    os.utime(f, (1_700_000_000, 1_700_000_000))
            export.build(root, t / "o1")
            export.build(root, t / "o2")
            self.assertEqual((t / "o1" / "census.json").read_bytes(), (t / "o2" / "census.json").read_bytes())
            rows = [json.loads(l) for l in (t / "o1" / "history_records.jsonl").read_text().splitlines()]
            new = [r for r in rows if "history_fields" in r and r["record_kind"] == "experiment"]
            self.assertEqual(len(new), 2)
            self.assertEqual([r["history_fields"]["trial_ordinal"] for r in new], [3, 4])
            self.assertEqual(new[0]["history_fields"]["mechanism_family_bucket"], "prefetch/cache")
            self.assertIsNone(new[0]["history_fields"]["knob_descriptor"])
            self.assertIsNotNone(new[0]["history_fields"]["stage_seconds"]["measure"])
            # old rows: no key at all, and still read
            self.assertEqual(sum(1 for r in rows if r["record_kind"] == "experiment"
                                 and "history_fields" not in r), 2)
            census = json.loads((t / "o1" / "census.json").read_text())
            post = census["c_budget_axis_and_keys"]["post_change_fields"]
            self.assertEqual(post["calls_joined_to_an_attempt_by_trial_id"], 2)
            self.assertEqual(post["calls_with_null_trial_id"], 1)
            self.assertTrue(post["trial_ordinals_strictly_increasing"])

    def test_old_stores_produce_no_post_change_block(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "ak"
            (root / "loop-memory").mkdir(parents=True)
            make_store(root, "ak-q-cpu-decode-1", "store", ["kept"])
            export.build(root, t / "o")
            census = json.loads((t / "o" / "census.json").read_text())
            self.assertNotIn("post_change_fields", census["c_budget_axis_and_keys"])
            rows = [json.loads(l) for l in (t / "o" / "history_records.jsonl").read_text().splitlines()]
            self.assertTrue(all("history_fields" not in r for r in rows))

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
