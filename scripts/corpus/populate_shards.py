#!/usr/bin/env python3
"""Populate sharded corpus from temp DBs — parallel, streaming, always queryable.

Uses dual-cursor merge (no per-snippet ngram lookups — O(N) not O(N²)).
Processes multiple languages in parallel via separate processes.
50% sampling via --sample-rate to prune corpus while preserving diversity.

Usage:
    python scripts/corpus/populate_shards.py
    python scripts/corpus/populate_shards.py --languages python,c++,rust
    python scripts/corpus/populate_shards.py --sample-rate 0.5  # keep 50%
    python scripts/corpus/populate_shards.py --clean  # clear existing shard data first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_PATH = "/mnt/raid0/llm/cache/corpus/v3_sharded"
COMMIT_INTERVAL = 100_000  # Commit every N snippets
MMAP_SIZE = 8 * 1024 * 1024 * 1024  # 8GB mmap per temp DB

# Pre-allocated ID ranges — each language gets 1 billion IDs
LANG_ID_OFFSETS = {
    "python": 0,
    "javascript": 1_000_000_000,
    "typescript": 2_000_000_000,
    "rust": 3_000_000_000,
    "go": 4_000_000_000,
    "c++": 5_000_000_000,
}


def gram_to_shard(gram: str, num_shards: int) -> int:
    h = hashlib.md5(gram.encode()).digest()
    return int.from_bytes(h[:4], "little") % num_shards


def ensure_output_dbs(output_dir: str, num_shards: int):
    """Create snippets.db + shard DBs if they don't exist. Idempotent."""
    snippets_path = os.path.join(output_dir, "snippets.db")
    conn = sqlite3.connect(snippets_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            source TEXT DEFAULT '',
            hash TEXT NOT NULL,
            language TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
    """)
    conn.close()

    for i in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{i:02d}.db")
        conn = sqlite3.connect(shard_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ngrams (
                gram TEXT NOT NULL,
                snippet_id INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram);
        """)
        conn.close()


def clean_output_dbs(output_dir: str, num_shards: int):
    """Delete all data from output DBs (keep schema). For fresh repopulation."""
    log.info("Cleaning existing data from output DBs...")

    snippets_path = os.path.join(output_dir, "snippets.db")
    if os.path.exists(snippets_path):
        conn = sqlite3.connect(snippets_path, isolation_level=None)
        conn.execute("DELETE FROM snippets")
        conn.execute("VACUUM")
        conn.close()
        log.info("Cleaned snippets.db")

    def clean_shard(i):
        shard_path = os.path.join(output_dir, f"shard_{i:02d}.db")
        if os.path.exists(shard_path):
            conn = sqlite3.connect(shard_path, isolation_level=None)
            conn.execute("DELETE FROM ngrams")
            conn.execute("VACUUM")
            conn.close()
        return i

    with ThreadPoolExecutor(max_workers=min(num_shards, 16)) as pool:
        for i in pool.map(clean_shard, range(num_shards)):
            log.info("Cleaned shard_%02d.db", i)

    # Clear state
    state_file = os.path.join(output_dir, "populate_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
    log.info("Clean complete")


def flush_shard(shard_id: int, output_dir: str, ngrams: list[tuple[str, int]]):
    """Write a batch of ngrams to one shard DB. Used by thread pool."""
    shard_path = os.path.join(output_dir, f"shard_{shard_id:02d}.db")
    for attempt in range(10):
        try:
            conn = sqlite3.connect(shard_path, timeout=300)
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=-128000")
            conn.executemany(
                "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                ngrams,
            )
            conn.commit()
            conn.close()
            return len(ngrams)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < 9:
                time.sleep(1 + attempt)
                continue
            raise


def populate_language(
    language: str,
    output_dir: str,
    num_shards: int,
    sample_rate: float,
    state_file: str,
):
    """Stream one language's temp DB into output shards using dual-cursor merge.

    The key insight: both snippets and ngrams in temp DBs are ordered by
    snippet_id (inserted sequentially during download). So we scan both
    tables simultaneously — O(N) total, no per-snippet lookups.
    """
    temp_db = os.path.join(output_dir, "temp", f"{language}.db")
    if not os.path.exists(temp_db):
        log.warning("[%s] No temp DB found at %s", language, temp_db)
        return

    id_offset = LANG_ID_OFFSETS[language]

    # Load state for resume
    state = {}
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
    last_local_id = state.get(language, {}).get("last_id", -1)
    total_snippets = state.get(language, {}).get("snippets", 0)
    total_ngrams = state.get(language, {}).get("ngrams", 0)

    # Open source (read-only, aggressive caching)
    src = sqlite3.connect(temp_db, timeout=30)
    src.execute("PRAGMA query_only=ON")
    src.execute(f"PRAGMA mmap_size={MMAP_SIZE}")
    src.execute("PRAGMA cache_size=-1000000")  # 1GB page cache

    src_count = src.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
    remaining = src.execute(
        "SELECT COUNT(*) FROM snippets WHERE id > ?", (last_local_id,)
    ).fetchone()[0]

    if remaining == 0:
        log.info("[%s] Already fully populated (%d snippets)", language, total_snippets)
        src.close()
        return

    # After sampling
    expected = int(remaining * sample_rate)
    log.info(
        "[%s] %d remaining of %d total → sampling %.0f%% → ~%d snippets to process",
        language, remaining, src_count, sample_rate * 100, expected,
    )

    # Open output snippets DB
    snippets_conn = sqlite3.connect(
        os.path.join(output_dir, "snippets.db"), timeout=300,
    )
    snippets_conn.execute("PRAGMA synchronous=OFF")
    snippets_conn.execute("PRAGMA journal_mode=WAL")
    snippets_conn.execute("PRAGMA cache_size=-512000")  # 512MB
    snippets_conn.execute("PRAGMA temp_store=MEMORY")

    # --- Dual-cursor merge ---
    # Cursor 1: snippets ordered by id
    snippet_cursor = src.execute(
        "SELECT id, code, source, hash, language FROM snippets WHERE id > ? ORDER BY id",
        (last_local_id,),
    )

    # Cursor 2: ngrams ordered by snippet_id (already in insertion order)
    ngram_cursor = src.execute(
        "SELECT snippet_id, gram FROM ngrams WHERE snippet_id > ? ORDER BY rowid",
        (last_local_id,),
    )

    # Pre-fetch first ngram row
    current_ngram = next(ngram_cursor, None)

    t0 = time.perf_counter()
    snippet_batch = []
    shard_buffers: list[list[tuple[str, int]]] = [[] for _ in range(num_shards)]
    batch_snippets = 0
    batch_ngrams = 0
    skipped = 0

    # Thread pool for parallel shard writes
    shard_pool = ThreadPoolExecutor(max_workers=16)

    for local_id, code, source, h, lang in snippet_cursor:
        # Deterministic sampling: keep if hash of id falls within rate
        # Using modular arithmetic for speed (id % N < N*rate)
        keep = (local_id % 1000) < int(sample_rate * 1000)

        # Collect all ngrams for this snippet from cursor 2
        snippet_ngrams = []
        while current_ngram is not None and current_ngram[0] == local_id:
            snippet_ngrams.append(current_ngram[1])
            current_ngram = next(ngram_cursor, None)

        # Also skip past any ngrams for IDs we've passed (shouldn't happen
        # but be defensive about ordering)
        while current_ngram is not None and current_ngram[0] < local_id:
            current_ngram = next(ngram_cursor, None)

        if not keep:
            skipped += 1
            continue

        global_id = id_offset + local_id
        snippet_batch.append((global_id, code, source, h, lang))

        for gram in snippet_ngrams:
            shard_id = gram_to_shard(gram, num_shards)
            shard_buffers[shard_id].append((gram, global_id))
            batch_ngrams += 1

        batch_snippets += 1
        last_local_id = local_id

        # Flush batches
        if batch_snippets >= COMMIT_INTERVAL:
            # Write snippets
            snippets_conn.executemany(
                "INSERT OR IGNORE INTO snippets (id, code, source, hash, language) "
                "VALUES (?, ?, ?, ?, ?)",
                snippet_batch,
            )
            snippets_conn.commit()
            snippet_batch = []

            # Write ngrams to shards in parallel (16 threads, one per shard)
            futures = []
            for sid in range(num_shards):
                if shard_buffers[sid]:
                    futures.append(
                        shard_pool.submit(flush_shard, sid, output_dir, shard_buffers[sid])
                    )
                    shard_buffers[sid] = []
            for f in futures:
                f.result()  # Wait for all shard writes

            total_snippets += batch_snippets
            total_ngrams += batch_ngrams
            elapsed = time.perf_counter() - t0
            rate = total_snippets / elapsed if elapsed > 0 else 0

            log.info(
                "[%s] %d snippets, %d ngrams, %d skipped (%.0f snippets/s, %.1fs)",
                language, total_snippets, total_ngrams, skipped, rate, elapsed,
            )

            # Save state for resume
            state[language] = {
                "last_id": last_local_id,
                "snippets": total_snippets,
                "ngrams": total_ngrams,
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            batch_snippets = 0
            batch_ngrams = 0

    # Final flush
    if snippet_batch:
        snippets_conn.executemany(
            "INSERT OR IGNORE INTO snippets (id, code, source, hash, language) "
            "VALUES (?, ?, ?, ?, ?)",
            snippet_batch,
        )
        snippets_conn.commit()

        futures = []
        for sid in range(num_shards):
            if shard_buffers[sid]:
                futures.append(
                    shard_pool.submit(flush_shard, sid, output_dir, shard_buffers[sid])
                )
                shard_buffers[sid] = []
        for f in futures:
            f.result()

        total_snippets += batch_snippets
        total_ngrams += batch_ngrams

    shard_pool.shutdown()
    elapsed = time.perf_counter() - t0
    rate = total_snippets / elapsed if elapsed > 0 else 0
    log.info(
        "[%s] DONE: %d snippets, %d ngrams, %d skipped in %.1fs (%.0f snippets/s)",
        language, total_snippets, total_ngrams, skipped, elapsed, rate,
    )

    state[language] = {
        "last_id": last_local_id,
        "snippets": total_snippets,
        "ngrams": total_ngrams,
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    snippets_conn.close()
    src.close()

    # Delete temp DB to free disk space
    log.info("[%s] Deleting temp DB %s to reclaim space...", language, temp_db)
    os.remove(temp_db)
    log.info("[%s] Deleted temp DB", language)


def populate_language_worker(
    language: str, output_dir: str, num_shards: int, sample_rate: float, state_file: str,
):
    """Wrapper for multiprocessing — catches exceptions."""
    try:
        populate_language(language, output_dir, num_shards, sample_rate, state_file)
    except Exception:
        log.exception("[%s] FAILED", language)


def main():
    parser = argparse.ArgumentParser(description="Populate shards from temp DBs")
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--languages", default=None, help="Comma-separated (default: all available)")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Keep this fraction (0.5 = 50%%)")
    parser.add_argument("--clean", action="store_true", help="Clear existing shard data first")
    parser.add_argument("--parallel", action="store_true", help="Process languages in parallel")
    args = parser.parse_args()

    output_dir = args.path
    num_shards = args.shards
    state_file = os.path.join(output_dir, "populate_state.json")

    # Ensure output DBs exist
    ensure_output_dbs(output_dir, num_shards)

    # Optionally clean
    if args.clean:
        clean_output_dbs(output_dir, num_shards)

    # Determine languages
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",")]
    else:
        languages = [
            lang for lang in LANG_ID_OFFSETS
            if os.path.exists(os.path.join(output_dir, "temp", f"{lang}.db"))
        ]

    languages = [lang for lang in languages if lang in LANG_ID_OFFSETS]
    log.info("Languages to populate: %s (sample_rate=%.2f)", languages, args.sample_rate)

    if args.parallel and len(languages) > 1:
        # Process all languages simultaneously — they write to non-overlapping
        # ID ranges. WAL mode handles concurrent shard writes.
        procs = []
        for lang in languages:
            p = Process(
                target=populate_language_worker,
                args=(lang, output_dir, num_shards, args.sample_rate, state_file),
                name=lang,
            )
            p.start()
            procs.append((lang, p))
            log.info("Started process for %s (PID %d)", lang, p.pid)

        for lang, p in procs:
            p.join()
            log.info("%s process exited (code=%s)", lang, p.exitcode)
    else:
        for lang in languages:
            populate_language(lang, output_dir, num_shards, args.sample_rate, state_file)

    # Load final state
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {}

    # Write meta.json
    total_snippets = sum(s.get("snippets", 0) for s in state.values())
    total_ngrams = sum(s.get("ngrams", 0) for s in state.values())

    meta = {
        "version": 3,
        "format": "sharded_sqlite",
        "ngram_size": 4,
        "num_shards": num_shards,
        "num_snippets": total_snippets,
        "num_ngrams": total_ngrams,
        "sample_rate": args.sample_rate,
        "languages": languages,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lang_stats": state,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("=" * 60)
    log.info("Population complete: %d snippets, %d ngrams", total_snippets, total_ngrams)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
