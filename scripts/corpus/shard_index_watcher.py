#!/usr/bin/env python3
"""Watch for v3 sharded corpus shard DBs and create indexes as they appear.

Runs alongside the build process. Uses WAL mode's concurrent access to
CREATE INDEX on shards as soon as merge produces them, making the corpus
queryable before the build's own Phase 3.

Usage:
    python scripts/corpus/shard_index_watcher.py [--path /mnt/raid0/llm/cache/corpus/v3_sharded]
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [watcher] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_PATH = "/mnt/raid0/llm/cache/corpus/v3_sharded"
POLL_INTERVAL = 15  # seconds between checks
INDEX_TIMEOUT = 300  # 5 min timeout per shard index attempt


def shard_has_index(shard_path: str) -> bool:
    """Check if a shard DB already has the ngrams index."""
    try:
        conn = sqlite3.connect(shard_path, timeout=5)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_ngrams_gram'"
        ).fetchall()
        conn.close()
        return len(indexes) > 0
    except Exception:
        return False


def shard_row_count(shard_path: str) -> int:
    """Get approximate row count (fast via sqlite_stat if available, else COUNT)."""
    try:
        conn = sqlite3.connect(shard_path, timeout=5)
        count = conn.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return -1


def try_create_index(shard_path: str) -> bool:
    """Attempt to create index on a shard. Returns True on success."""
    shard_name = os.path.basename(shard_path)
    rows = shard_row_count(shard_path)
    if rows == 0:
        log.info("%s: empty, skipping", shard_name)
        return False
    if rows < 0:
        log.info("%s: can't read, skipping", shard_name)
        return False

    log.info("%s: %d ngrams, creating index...", shard_name, rows)
    t0 = time.perf_counter()

    try:
        conn = sqlite3.connect(shard_path, timeout=INDEX_TIMEOUT)
        conn.execute("PRAGMA cache_size=-512000")
        conn.execute("PRAGMA mmap_size=4294967296")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram)")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")  # Non-blocking checkpoint
        conn.close()

        elapsed = time.perf_counter() - t0
        log.info("%s: indexed %d ngrams in %.1fs", shard_name, rows, elapsed)
        return True

    except sqlite3.OperationalError as e:
        elapsed = time.perf_counter() - t0
        if "locked" in str(e).lower():
            log.info("%s: database locked (build writing), will retry (%.1fs)", shard_name, elapsed)
        else:
            log.warning("%s: error: %s (%.1fs)", shard_name, e, elapsed)
        return False
    except Exception as e:
        log.warning("%s: unexpected error: %s", shard_name, e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Watch and index corpus shards")
    parser.add_argument("--path", default=DEFAULT_PATH, help="Corpus output directory")
    parser.add_argument("--shards", type=int, default=16, help="Expected number of shards")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL, help="Poll interval (seconds)")
    args = parser.parse_args()

    corpus_dir = args.path
    num_shards = args.shards
    indexed: set[int] = set()

    log.info("Watching %s for %d shards (poll every %ds)", corpus_dir, num_shards, args.poll)

    while len(indexed) < num_shards:
        for i in range(num_shards):
            if i in indexed:
                continue

            shard_path = os.path.join(corpus_dir, f"shard_{i:02d}.db")
            if not os.path.exists(shard_path):
                continue

            if shard_has_index(shard_path):
                log.info("shard_%02d.db: already indexed", i)
                indexed.add(i)
                continue

            if try_create_index(shard_path):
                indexed.add(i)

        if len(indexed) < num_shards:
            remaining = num_shards - len(indexed)
            log.info("Progress: %d/%d shards indexed, %d remaining. Sleeping %ds...",
                     len(indexed), num_shards, remaining, args.poll)
            time.sleep(args.poll)

    # Also checkpoint snippets.db
    snippets_path = os.path.join(corpus_dir, "snippets.db")
    if os.path.exists(snippets_path):
        try:
            conn = sqlite3.connect(snippets_path, timeout=30)
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.close()
            log.info("snippets.db: WAL checkpoint complete")
        except Exception as e:
            log.warning("snippets.db checkpoint failed: %s", e)

    log.info("All %d shards indexed. Corpus is fully queryable.", num_shards)


if __name__ == "__main__":
    main()
