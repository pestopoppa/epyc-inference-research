"""Unit tests for kernel_store.py — the MI210 kernel-R&D strategy store.

Covers three things added/fixed on 2026-07-17 (loops-and-dashboards-audit
P4·rank-9):

  1. The insert/duplicate counter regression. The old code decided per-row
     "inserted vs duplicate" from `sqlite3.Connection.total_changes`, which is
     CUMULATIVE over the connection's lifetime. After the first successful
     insert it was never 0 again, so every later row — dups included — was
     miscounted as an insert (n over-counted, dup stuck at 0).
     `test_dup_counter_*` reproduce that and pin the rowcount-based fix.

  2. `purge --git-sha SHA` — remove every row for a retracted kernel build.
  3. `rewind` — roll the store back along its append-order (`id`) timeline to
     just after a git_sha, or to an explicit boundary id.

Purely local SQLite — NO inference, NO server, NO model. Run standalone:

    python3 scripts/kernel_rnd/test_kernel_store.py

Or via pytest if available:

    python3 -m pytest scripts/kernel_rnd/test_kernel_store.py -v
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest

# Allow running directly OR via pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_store as ks


def _rec(label, ts, git_sha, model="mi210", status="OK",
         tbo="42/42 tests passed", coherence="byte-identical",
         single_v=100.0, delta=5.0):
    """Build one OBSERVATION JSONL record matching kernel_eval.sh's shape."""
    return {
        "label": label, "ts": ts, "git_sha": git_sha, "model": model,
        "status": status,
        "correctness": {"test_backend_ops": tbo, "coherence": coherence},
        "single_tps_baseline": 95.0, "single_tps_variant": single_v,
        "delta_pct": delta, "aggregate_tps_variant": None,
        "mechanism": {},
    }


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class _StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kernel_store_test_")
        self.db = os.path.join(self.tmp, "store.sqlite")
        self.jsonl = os.path.join(self.tmp, "records.jsonl")

    def tearDown(self):
        for p in (self.db, self.jsonl):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _count(self):
        c = sqlite3.connect(self.db)
        try:
            return c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        finally:
            c.close()

    def _shas(self):
        c = sqlite3.connect(self.db)
        try:
            return sorted(r[0] for r in c.execute("SELECT git_sha FROM runs"))
        finally:
            c.close()


class TestDupCounter(_StoreTest):
    def test_dup_counter_regression(self):
        """Reproduce the pre-fix miscount: after the first insert, duplicates
        were counted as inserts. Here row 1 is unique and rows 2-3 collide with
        it on UNIQUE(label,ts,git_sha); row 4 is unique. Correct tally is
        2 inserted / 2 duplicate — the buggy code reported 4 / 0."""
        recs = [
            _rec("A", "2026-07-17T00:00:00Z", "sha1"),   # insert
            _rec("A", "2026-07-17T00:00:00Z", "sha1"),   # dup of #1
            _rec("A", "2026-07-17T00:00:00Z", "sha1"),   # dup of #1
            _rec("B", "2026-07-17T00:01:00Z", "sha1"),   # insert
        ]
        _write_jsonl(self.jsonl, recs)
        n, dup, bad = ks.ingest(self.db, self.jsonl)
        self.assertEqual((n, dup, bad), (2, 2, 0))
        self.assertEqual(self._count(), 2)

    def test_reingest_is_all_duplicates(self):
        """Idempotence: ingesting the same file twice inserts nothing the 2nd
        time (every row is a duplicate) and the row count is unchanged. This
        also exercises the 'first row is already a dup' path that the old
        total_changes==0 check got wrong."""
        recs = [
            _rec("A", "2026-07-17T00:00:00Z", "sha1"),
            _rec("B", "2026-07-17T00:01:00Z", "sha1"),
            _rec("C", "2026-07-17T00:02:00Z", "sha2"),
        ]
        _write_jsonl(self.jsonl, recs)
        n1, dup1, _ = ks.ingest(self.db, self.jsonl)
        self.assertEqual((n1, dup1), (3, 0))
        n2, dup2, _ = ks.ingest(self.db, self.jsonl)
        self.assertEqual((n2, dup2), (0, 3))
        self.assertEqual(self._count(), 3)

    def test_unparseable_lines_counted(self):
        with open(self.jsonl, "w") as f:
            f.write(json.dumps(_rec("A", "t1", "sha1")) + "\n")
            f.write("{ not json\n")
            f.write("\n")  # blank line skipped, not counted as bad
            f.write(json.dumps(_rec("B", "t2", "sha1")) + "\n")
        n, dup, bad = ks.ingest(self.db, self.jsonl)
        self.assertEqual((n, dup, bad), (2, 0, 1))


class TestPurge(_StoreTest):
    def setUp(self):
        super().setUp()
        _write_jsonl(self.jsonl, [
            _rec("A", "t1", "sha1"),
            _rec("B", "t2", "sha1"),
            _rec("C", "t3", "sha2"),
            _rec("D", "t4", "sha3"),
        ])
        ks.ingest(self.db, self.jsonl)

    def test_purge_dry_run_removes_nothing(self):
        removed = ks.purge(self.db, "sha1", force=False)
        self.assertEqual(removed, 0)
        self.assertEqual(self._count(), 4)

    def test_purge_force_removes_only_that_sha(self):
        removed = ks.purge(self.db, "sha1", force=True)
        self.assertEqual(removed, 2)
        self.assertEqual(self._count(), 2)
        # sha1 rows gone; sha2 + sha3 rows survive untouched
        self.assertEqual(self._shas(), ["sha2", "sha3"])

    def test_purge_unknown_sha_is_noop(self):
        removed = ks.purge(self.db, "does-not-exist", force=True)
        self.assertEqual(removed, 0)
        self.assertEqual(self._count(), 4)


class TestRewind(_StoreTest):
    def setUp(self):
        super().setUp()
        # Ingested in append order: ids 1..5.
        _write_jsonl(self.jsonl, [
            _rec("A", "t1", "sha1"),   # id 1
            _rec("B", "t2", "sha1"),   # id 2
            _rec("C", "t3", "sha2"),   # id 3  <- rewind-to-sha1 boundary
            _rec("D", "t4", "sha2"),   # id 4
            _rec("E", "t5", "sha3"),   # id 5
        ])
        ks.ingest(self.db, self.jsonl)

    def test_rewind_by_git_sha_dry_run(self):
        removed = ks.rewind(self.db, git_sha="sha1", boundary_id=None, force=False)
        self.assertEqual(removed, 0)
        self.assertEqual(self._count(), 5)

    def test_rewind_by_git_sha_force(self):
        # Boundary = MAX(id) among sha1 rows = id 2; drops ids 3,4,5.
        removed = ks.rewind(self.db, git_sha="sha1", boundary_id=None, force=True)
        self.assertEqual(removed, 3)
        self.assertEqual(self._count(), 2)
        self.assertEqual(self._shas(), ["sha1", "sha1"])

    def test_rewind_restores_prior_state(self):
        """Snapshot the store, append a new sha, then rewind by boundary id and
        confirm the store is byte-for-byte the prior set of rows again."""
        before = self._snapshot()
        # append two more rows (ids 6,7)
        extra = os.path.join(self.tmp, "extra.jsonl")
        _write_jsonl(extra, [
            _rec("F", "t6", "sha4"),
            _rec("G", "t7", "sha4"),
        ])
        ks.ingest(self.db, extra)
        os.unlink(extra)
        self.assertEqual(self._count(), 7)
        # rewind to explicit boundary id 5 -> drop 6,7
        removed = ks.rewind(self.db, git_sha=None, boundary_id=5, force=True)
        self.assertEqual(removed, 2)
        self.assertEqual(self._snapshot(), before)

    def test_rewind_unknown_sha_errors(self):
        with self.assertRaises(SystemExit):
            ks.rewind(self.db, git_sha="nope", boundary_id=None, force=True)

    def _snapshot(self):
        c = sqlite3.connect(self.db)
        try:
            return c.execute(
                "SELECT id,label,ts,git_sha FROM runs ORDER BY id"
            ).fetchall()
        finally:
            c.close()


class TestReadOnlyGuard(_StoreTest):
    def test_dry_run_uses_readonly_connection(self):
        """The dry-run inspection path must not create or mutate the store."""
        # store does not exist yet -> read-only open should exit cleanly
        with self.assertRaises(SystemExit):
            ks.purge(self.db, "sha1", force=False)
        self.assertFalse(os.path.exists(self.db))


if __name__ == "__main__":
    unittest.main(verbosity=2)
