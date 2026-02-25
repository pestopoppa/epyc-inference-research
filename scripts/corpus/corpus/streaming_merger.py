#!/usr/bin/env python3
"""Streaming merger: reads temp DBs while downloads are still running.

Incrementally merges snippets + ngrams from temp/{lang}.db into
snippets.db + shard_{00..15}.db as data arrives. Tracks progress
per language to avoid re-processing.

Runs alongside the download workers. Uses WAL concurrent reads.

Usage:
    python scripts/corpus/streaming_merger.py --path /mnt/raid0/llm/cache/corpus/v3_sharded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [merger] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_PATH = "/mnt/raid0/llm/cache/corpus/v3_sharded"
BATCH_SIZE = 10_000
POLL_INTERVAL = 10  # seconds between merge sweeps

# Pre-allocated ID ranges per language (1 billion each)
LANG_ID_OFFSETS = {
    "rust":       0,
    "c++":        1_000_000_000,
    "go":         2_000_000_000,
    "python":     3_000_000_000,
    "javascript": 4_000_000_000,
    "typescript": 5_000_000_000,
}


def gram_to_shard(gram: str, num_shards: int) -> int:
    h = hashlib.md5(gram.encode()).digest()
    return int.from_bytes(h[:4], "little") % num_shards


def open_output_dbs(
    output_dir: str, num_shards: int,
) -> tuple[sqlite3.Connection, list[sqlite3.Connection]]:
    """Open or create snippets.db + shard DBs."""
    snippets_path = os.path.join(output_dir, "snippets.db")
    snippets_conn = sqlite3.connect(snippets_path, timeout=300)
    snippets_conn.execute("PRAGMA journal_mode=WAL")
    snippets_conn.execute("PRAGMA cache_size=-512000")
    snippets_conn.execute("PRAGMA temp_store=MEMORY")
    # Tables already created by build — skip DDL to avoid needing write lock on open

    shard_conns = []
    for i in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{i:02d}.db")
        conn = sqlite3.connect(shard_path, timeout=300)
        # Read-only PRAGMAs don't need write lock
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")
        # Skip CREATE TABLE/INDEX on open — they already exist from the build.
        # We'll set synchronous=OFF on first write instead.
        shard_conns.append(conn)

    return snippets_conn, shard_conns


def merge_incremental(
    language: str,
    temp_db_path: str,
    snippets_conn: sqlite3.Connection,
    shard_conns: list[sqlite3.Connection],
    num_shards: int,
    last_local_id: int,
) -> tuple[int, int, int]:
    """Read new rows from temp DB and merge into output.

    Returns (new_last_local_id, snippets_added, ngrams_added).
    """
    id_offset = LANG_ID_OFFSETS.get(language, 0)

    try:
        src = sqlite3.connect(temp_db_path, timeout=5)
        src.execute("PRAGMA query_only=ON")
    except Exception as e:
        log.warning("[%s] Can't open temp DB: %s", language, e)
        return last_local_id, 0, 0

    # Get new snippets since last check
    try:
        new_snippets = src.execute(
            "SELECT id, code, source, hash, language FROM snippets WHERE id > ? ORDER BY id",
            (last_local_id,),
        ).fetchall()
    except Exception as e:
        log.warning("[%s] Can't read snippets: %s", language, e)
        src.close()
        return last_local_id, 0, 0

    if not new_snippets:
        src.close()
        return last_local_id, 0, 0

    # Insert snippets with global IDs
    snippet_batch = []
    local_ids = []
    for local_id, code, source, h, lang in new_snippets:
        global_id = id_offset + local_id
        snippet_batch.append((global_id, code, source, h, lang))
        local_ids.append(local_id)

        if len(snippet_batch) >= BATCH_SIZE:
            snippets_conn.executemany(
                "INSERT OR IGNORE INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
                snippet_batch,
            )
            snippet_batch = []

    if snippet_batch:
        snippets_conn.executemany(
            "INSERT OR IGNORE INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
            snippet_batch,
        )
    snippets_conn.commit()

    new_last_id = local_ids[-1]

    # Get ngrams: sequential scan filtered by ID range (no index needed)
    shard_buffers: list[list[tuple[str, int]]] = [[] for _ in range(num_shards)]
    ngram_count = 0
    flush_threshold = BATCH_SIZE * 5
    min_id = last_local_id + 1
    max_id = new_last_id

    try:
        cursor = src.execute(
            "SELECT gram, snippet_id FROM ngrams WHERE snippet_id >= ? AND snippet_id <= ?",
            (min_id, max_id),
        )
        for gram, local_sid in cursor:
            global_sid = id_offset + local_sid
            shard_id = gram_to_shard(gram, num_shards)
            shard_buffers[shard_id].append((gram, global_sid))
            ngram_count += 1

            if len(shard_buffers[shard_id]) >= flush_threshold:
                shard_conns[shard_id].executemany(
                    "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                    shard_buffers[shard_id],
                )
                shard_buffers[shard_id] = []
    except Exception as e:
        log.warning("[%s] Ngram read error: %s", language, e)

    # Flush remaining
    for sid in range(num_shards):
        if shard_buffers[sid]:
            shard_conns[sid].executemany(
                "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                shard_buffers[sid],
            )

    for conn in shard_conns:
        conn.commit()

    src.close()
    return new_last_id, len(new_snippets), ngram_count


def main():
    parser = argparse.ArgumentParser(description="Streaming merger for corpus shards")
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL)
    parser.add_argument("--once", action="store_true", help="Run one pass and exit")
    args = parser.parse_args()

    output_dir = args.path
    temp_dir = os.path.join(output_dir, "temp")
    num_shards = args.shards

    # Check what the build already merged (avoid double-processing)
    progress_file = os.path.join(output_dir, "build_progress.json")
    already_merged: set[str] = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        already_merged = set(progress.get("merged", {}).keys())
        if already_merged:
            log.info("Skipping languages already merged by build: %s", already_merged)

    # State: track last processed ID per language
    state_file = os.path.join(output_dir, "merger_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {}

    log.info("Opening output DBs (%d shards)", num_shards)
    snippets_conn, shard_conns = open_output_dbs(output_dir, num_shards)

    total_snippets = 0
    total_ngrams = 0

    while True:
        sweep_snippets = 0
        sweep_ngrams = 0
        active_langs = 0

        for lang in LANG_ID_OFFSETS:
            if lang in already_merged:
                continue
            temp_db = os.path.join(temp_dir, f"{lang}.db")
            if not os.path.exists(temp_db):
                continue

            active_langs += 1
            last_id = state.get(lang, -1)
            new_last_id, n_snip, n_ngram = merge_incremental(
                lang, temp_db, snippets_conn, shard_conns, num_shards, last_id,
            )

            if n_snip > 0:
                state[lang] = new_last_id
                total_snippets += n_snip
                total_ngrams += n_ngram
                sweep_snippets += n_snip
                sweep_ngrams += n_ngram
                log.info(
                    "[%s] +%d snippets, +%d ngrams (total: id %d→%d)",
                    lang, n_snip, n_ngram, last_id, new_last_id,
                )

        if sweep_snippets > 0:
            # Save state
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            log.info(
                "Sweep: +%d snippets, +%d ngrams. Cumulative: %d snippets, %d ngrams",
                sweep_snippets, sweep_ngrams, total_snippets, total_ngrams,
            )

        if args.once:
            break

        # Check if all downloads are done
        all_done = all(
            os.path.exists(os.path.join(temp_dir, f"{lang}.db.done"))
            for lang in LANG_ID_OFFSETS
            if lang not in already_merged and os.path.exists(os.path.join(temp_dir, f"{lang}.db"))
        )
        if all_done and sweep_snippets == 0:
            log.info("All downloads complete and fully merged. Exiting.")
            break

        if sweep_snippets == 0:
            log.info("No new data. %d temp DBs found. Sleeping %ds...", active_langs, args.poll)

        time.sleep(args.poll)

    # Final commit and checkpoint
    for conn in shard_conns:
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()
    snippets_conn.commit()
    snippets_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    snippets_conn.close()

    log.info("Done. Total: %d snippets, %d ngrams merged.", total_snippets, total_ngrams)


# ── Dedup monitor: cleans up entries added by the build's own merger ──────


# The build's merge uses sequential IDs starting after Rust (14,271,972).
# The streaming merger uses pre-allocated ranges (>= 1,000,000,000).
# Anything in between was added by the build and should be removed.
BUILD_ID_LOW = 14_271_972   # First ID the build would assign after Rust
BUILD_ID_HIGH = 1_000_000_000  # Below our pre-allocated ranges


def dedup_pass(output_dir: str, num_shards: int) -> int:
    """Delete entries added by the build's merger. Returns rows deleted."""
    snippets_path = os.path.join(output_dir, "snippets.db")
    if not os.path.exists(snippets_path):
        return 0

    total_deleted = 0

    # Delete from snippets.db
    try:
        conn = sqlite3.connect(snippets_path, timeout=10)
        count = conn.execute(
            "SELECT COUNT(*) FROM snippets WHERE id >= ? AND id < ?",
            (BUILD_ID_LOW, BUILD_ID_HIGH),
        ).fetchone()[0]
        if count > 0:
            conn.execute(
                "DELETE FROM snippets WHERE id >= ? AND id < ?",
                (BUILD_ID_LOW, BUILD_ID_HIGH),
            )
            conn.commit()
            total_deleted += count
            log.info("[dedup] Removed %d build-merge snippets from snippets.db", count)
        conn.close()
    except Exception as e:
        log.warning("[dedup] snippets.db: %s", e)

    # Delete from each shard
    for i in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{i:02d}.db")
        if not os.path.exists(shard_path):
            continue
        try:
            conn = sqlite3.connect(shard_path, timeout=10)
            count = conn.execute(
                "SELECT COUNT(*) FROM ngrams WHERE snippet_id >= ? AND snippet_id < ?",
                (BUILD_ID_LOW, BUILD_ID_HIGH),
            ).fetchone()[0]
            if count > 0:
                conn.execute(
                    "DELETE FROM ngrams WHERE snippet_id >= ? AND snippet_id < ?",
                    (BUILD_ID_LOW, BUILD_ID_HIGH),
                )
                conn.commit()
                total_deleted += count
                log.info("[dedup] Removed %d build-merge ngrams from shard_%02d.db", count, i)
            conn.close()
        except Exception as e:
            log.warning("[dedup] shard_%02d.db: %s", i, e)

    return total_deleted


def main_with_dedup():
    """Run streaming merger + periodic dedup in the same loop."""
    parser = argparse.ArgumentParser(description="Streaming merger + dedup for corpus shards")
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL)
    parser.add_argument("--once", action="store_true", help="Run one pass and exit")
    parser.add_argument("--dedup-only", action="store_true", help="Only run dedup, no merge")
    args = parser.parse_args()

    if args.dedup_only:
        deleted = dedup_pass(args.path, args.shards)
        log.info("Dedup-only: %d rows removed", deleted)
        return

    output_dir = args.path
    temp_dir = os.path.join(output_dir, "temp")
    num_shards = args.shards

    # Check what the build already merged
    progress_file = os.path.join(output_dir, "build_progress.json")
    already_merged: set[str] = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        already_merged = set(progress.get("merged", {}).keys())
        if already_merged:
            log.info("Skipping languages already merged by build: %s", already_merged)

    # State: track last processed ID per language
    state_file = os.path.join(output_dir, "merger_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {}

    log.info("Opening output DBs (%d shards)", num_shards)
    snippets_conn, shard_conns = open_output_dbs(output_dir, num_shards)

    total_snippets = 0
    total_ngrams = 0
    sweeps_since_dedup = 0

    while True:
        sweep_snippets = 0
        sweep_ngrams = 0
        active_langs = 0

        for lang in LANG_ID_OFFSETS:
            if lang in already_merged:
                continue
            temp_db = os.path.join(temp_dir, f"{lang}.db")
            if not os.path.exists(temp_db):
                continue

            active_langs += 1
            last_id = state.get(lang, -1)
            new_last_id, n_snip, n_ngram = merge_incremental(
                lang, temp_db, snippets_conn, shard_conns, num_shards, last_id,
            )

            if n_snip > 0:
                state[lang] = new_last_id
                total_snippets += n_snip
                total_ngrams += n_ngram
                sweep_snippets += n_snip
                sweep_ngrams += n_ngram
                log.info(
                    "[%s] +%d snippets, +%d ngrams (total: id %d→%d)",
                    lang, n_snip, n_ngram, last_id, new_last_id,
                )

        if sweep_snippets > 0:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            log.info(
                "Sweep: +%d snippets, +%d ngrams. Cumulative: %d snippets, %d ngrams",
                sweep_snippets, sweep_ngrams, total_snippets, total_ngrams,
            )

        # Run dedup every 4 sweeps to clean up build's merge entries
        sweeps_since_dedup += 1
        if sweeps_since_dedup >= 4:
            # Close our connections briefly so dedup can write
            for conn in shard_conns:
                conn.commit()
            snippets_conn.commit()
            deleted = dedup_pass(output_dir, num_shards)
            if deleted > 0:
                log.info("[dedup] Cleaned %d build-merge rows", deleted)
            sweeps_since_dedup = 0

        if args.once:
            break

        # Check if all downloads are done
        all_done = all(
            os.path.exists(os.path.join(temp_dir, f"{lang}.db.done"))
            for lang in LANG_ID_OFFSETS
            if lang not in already_merged and os.path.exists(os.path.join(temp_dir, f"{lang}.db"))
        )
        if all_done and sweep_snippets == 0:
            # Final dedup
            for conn in shard_conns:
                conn.commit()
            snippets_conn.commit()
            dedup_pass(output_dir, num_shards)
            log.info("All downloads complete and fully merged. Exiting.")
            break

        if sweep_snippets == 0:
            log.info("No new data. %d temp DBs active. Sleeping %ds...", active_langs, args.poll)

        time.sleep(args.poll)

    # Final commit and checkpoint
    for conn in shard_conns:
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()
    snippets_conn.commit()
    snippets_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    snippets_conn.close()

    log.info("Done. Total: %d snippets, %d ngrams merged.", total_snippets, total_ngrams)


if __name__ == "__main__":
    main_with_dedup()
