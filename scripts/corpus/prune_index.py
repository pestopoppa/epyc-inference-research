#!/usr/bin/env python3
"""Prune a corpus index to a target size by keeping highest-quality snippets.

Usage:
    # Keep top 5M snippets
    python scripts/corpus/prune_index.py --target-snippets 5000000

    # Keep top 50GB DB size
    python scripts/corpus/prune_index.py --target-gb 50

    # Preview without modifying
    python scripts/corpus/prune_index.py --target-snippets 5000000 --dry-run

Pruning strategy:
  1. Score each snippet by n-gram uniqueness (how many unique grams it contributes)
  2. Keep snippets that contribute the most unique n-grams (maximizes coverage)
  3. Maintain per-language quotas proportional to language distribution
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB = "/mnt/raid0/llm/cache/corpus/full_index/corpus.db"


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get current index statistics."""
    total_snippets = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
    total_ngrams = conn.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
    lang_counts = dict(
        conn.execute(
            "SELECT language, COUNT(*) FROM snippets GROUP BY language"
        ).fetchall()
    )
    return {
        "total_snippets": total_snippets,
        "total_ngrams": total_ngrams,
        "languages": lang_counts,
    }


def prune_by_count(
    conn: sqlite3.Connection,
    target_snippets: int,
    dry_run: bool = False,
) -> None:
    """Prune to target snippet count, keeping proportional language distribution."""
    stats = get_stats(conn)
    current = stats["total_snippets"]

    if current <= target_snippets:
        log.info("Already at %d snippets (target: %d). Nothing to prune.", current, target_snippets)
        return

    to_remove = current - target_snippets
    log.info("Pruning %d snippets (%d -> %d)", to_remove, current, target_snippets)

    # Calculate per-language quotas (proportional)
    lang_quotas = {}
    for lang, count in stats["languages"].items():
        ratio = count / current
        lang_quotas[lang] = int(target_snippets * ratio)

    # Adjust rounding errors
    total_quota = sum(lang_quotas.values())
    if total_quota < target_snippets:
        # Give remainder to largest language
        largest = max(lang_quotas, key=lang_quotas.get)
        lang_quotas[largest] += target_snippets - total_quota

    log.info("Language quotas: %s", lang_quotas)

    if dry_run:
        log.info("[DRY RUN] Would prune %d snippets", to_remove)
        return

    # For each language, keep the top-N snippets by n-gram contribution count
    # (snippets with more n-grams = more useful for lookup matching)
    keep_ids = set()
    for lang, quota in lang_quotas.items():
        log.info("Selecting top %d snippets for %s...", quota, lang)
        # Get snippets ranked by how many n-grams they contribute
        rows = conn.execute(
            """
            SELECT s.id
            FROM snippets s
            JOIN (
                SELECT snippet_id, COUNT(*) as gram_count
                FROM ngrams
                WHERE snippet_id IN (SELECT id FROM snippets WHERE language = ?)
                GROUP BY snippet_id
                ORDER BY gram_count DESC
                LIMIT ?
            ) ranked ON s.id = ranked.snippet_id
            WHERE s.language = ?
            """,
            (lang, quota, lang),
        ).fetchall()
        for r in rows:
            keep_ids.add(r[0])
        log.info("  Keeping %d/%d snippets for %s", len(rows), stats["languages"][lang], lang)

    # Delete snippets not in keep set
    log.info("Deleting pruned snippets and n-grams...")
    t0 = time.perf_counter()

    # Delete n-grams first (they reference snippets)
    conn.execute(
        f"DELETE FROM ngrams WHERE snippet_id NOT IN ({','.join('?' * len(keep_ids))})",
        list(keep_ids),
    ) if len(keep_ids) < 100000 else _batch_delete_ngrams(conn, keep_ids)

    # Delete snippets
    conn.execute(
        f"DELETE FROM snippets WHERE id NOT IN ({','.join('?' * len(keep_ids))})",
        list(keep_ids),
    ) if len(keep_ids) < 100000 else _batch_delete_snippets(conn, keep_ids)

    conn.commit()
    elapsed = time.perf_counter() - t0
    log.info("Deletion took %.1fs", elapsed)

    # VACUUM to reclaim space
    log.info("Running VACUUM to reclaim disk space...")
    t0 = time.perf_counter()
    conn.execute("VACUUM")
    elapsed = time.perf_counter() - t0
    log.info("VACUUM took %.1fs", elapsed)

    # Report final stats
    final = get_stats(conn)
    log.info("Final: %d snippets, %d n-grams", final["total_snippets"], final["total_ngrams"])


def _batch_delete_ngrams(conn: sqlite3.Connection, keep_ids: set[int]) -> None:
    """Delete n-grams for snippets not in keep set (batch approach for large sets)."""
    # Create temp table with keep IDs
    conn.execute("CREATE TEMP TABLE _keep_ids (id INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO _keep_ids VALUES (?)",
        [(i,) for i in keep_ids],
    )
    conn.execute(
        "DELETE FROM ngrams WHERE snippet_id NOT IN (SELECT id FROM _keep_ids)"
    )
    conn.execute("DROP TABLE _keep_ids")


def _batch_delete_snippets(conn: sqlite3.Connection, keep_ids: set[int]) -> None:
    """Delete snippets not in keep set (batch approach for large sets)."""
    conn.execute("CREATE TEMP TABLE _keep_ids2 (id INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO _keep_ids2 VALUES (?)",
        [(i,) for i in keep_ids],
    )
    conn.execute(
        "DELETE FROM snippets WHERE id NOT IN (SELECT id FROM _keep_ids2)"
    )
    conn.execute("DROP TABLE _keep_ids2")


def main():
    parser = argparse.ArgumentParser(description="Prune corpus index")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to corpus.db")
    parser.add_argument("--target-snippets", type=int, help="Target number of snippets to keep")
    parser.add_argument("--target-gb", type=float, help="Target DB size in GB (approximate)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying")
    args = parser.parse_args()

    if not args.target_snippets and not args.target_gb:
        parser.error("Specify --target-snippets or --target-gb")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    stats = get_stats(conn)
    db_size_gb = os.path.getsize(args.db) / 1024**3

    log.info("Current: %d snippets, %d n-grams, %.1f GB", stats["total_snippets"], stats["total_ngrams"], db_size_gb)
    for lang, count in stats["languages"].items():
        log.info("  %s: %d", lang, count)

    if args.target_gb and not args.target_snippets:
        # Estimate snippets per GB
        snippets_per_gb = stats["total_snippets"] / db_size_gb if db_size_gb > 0 else 200000
        args.target_snippets = int(args.target_gb * snippets_per_gb)
        log.info("Estimated target: %d snippets for %.1f GB", args.target_snippets, args.target_gb)

    prune_by_count(conn, args.target_snippets, args.dry_run)

    # Update metadata
    if not args.dry_run:
        final = get_stats(conn)
        meta = {
            "version": 2,
            "format": "sqlite",
            "ngram_size": 4,
            "num_snippets": final["total_snippets"],
            "num_ngrams": final["total_ngrams"],
            "languages": final["languages"],
            "pruned_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "db_size_gb": os.path.getsize(args.db) / 1024**3,
        }
        meta_path = os.path.join(os.path.dirname(args.db), "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    conn.close()


if __name__ == "__main__":
    main()
