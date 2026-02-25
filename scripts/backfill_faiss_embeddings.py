#!/usr/bin/env python3
"""Backfill FAISS embeddings for SQLite records missing embeddings.

This fixes the mismatch between SQLite memory count and FAISS embedding count
that occurs when old seeding scripts add memories without proper FAISS integration.

Usage:
    python scripts/backfill_faiss_embeddings.py [--servers URL1,URL2,...] [--batch-size N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sqlite3
import sys
from itertools import cycle
from pathlib import Path
from typing import Iterator, List, Set

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.episodic_store import EpisodicStore

DEFAULT_SERVERS = ["http://127.0.0.1:8090", "http://127.0.0.1:8091",
                   "http://127.0.0.1:8092", "http://127.0.0.1:8093"]


class ParallelEmbedder:
    """Embed texts across multiple llama-server instances."""

    def __init__(self, server_urls: List[str]):
        self.servers = server_urls
        self.clients = [httpx.Client(timeout=60.0) for _ in server_urls]
        self.server_cycle: Iterator = cycle(range(len(server_urls)))
        self._verify_servers()

    def _verify_servers(self):
        available = []
        for i, (url, client) in enumerate(zip(self.servers, self.clients)):
            try:
                resp = client.post(f"{url}/embedding", json={"content": "test"})
                if resp.status_code == 200:
                    available.append(i)
            except Exception:
                pass
        if not available:
            raise RuntimeError("No embedding servers available!")
        print(f"  {len(available)}/{len(self.servers)} embedding servers available")
        self.available_indices = available
        self.server_cycle = cycle(available)

    def embed_one(self, text: str) -> np.ndarray:
        idx = next(self.server_cycle)
        url = self.servers[idx]
        client = self.clients[idx]
        try:
            resp = client.post(f"{url}/embedding", json={"content": text})
            data = resp.json()
            emb = data[0]["embedding"][0]
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Embedding failed on {url}: {e}")

    def embed_batch(self, texts: List[str], max_workers: int = 4) -> List[np.ndarray]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.embed_one, t) for t in texts]
            return [f.result() for f in futures]

    def close(self):
        for client in self.clients:
            client.close()


def get_all_memory_ids(store: EpisodicStore) -> Set[str]:
    """Get all memory IDs from SQLite."""
    conn = sqlite3.connect(store.sqlite_path)
    cursor = conn.execute("SELECT id FROM memories")
    ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return ids


def get_faiss_memory_ids(store: EpisodicStore) -> Set[str]:
    """Get memory IDs that have FAISS embeddings."""
    # The FAISS store maintains an id_map
    if hasattr(store._embedding_store, 'id_map'):
        return set(store._embedding_store.id_map)
    return set()


def get_memory_context(store: EpisodicStore, memory_id: str) -> dict:
    """Get memory context (task description) for embedding."""
    conn = sqlite3.connect(store.sqlite_path)
    cursor = conn.execute(
        "SELECT context FROM memories WHERE id = ?",
        (memory_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        import json
        ctx = json.loads(row[0])
        # Try to get task description for embedding
        return ctx.get("task_description", ctx.get("task", str(ctx)))
    return memory_id  # Fallback to ID


def backfill_embeddings(store: EpisodicStore, embedder: ParallelEmbedder,
                        batch_size: int = 20) -> dict:
    """Backfill missing FAISS embeddings."""
    # Find missing IDs
    all_ids = get_all_memory_ids(store)
    faiss_ids = get_faiss_memory_ids(store)
    missing_ids = all_ids - faiss_ids

    print(f"SQLite memories: {len(all_ids)}")
    print(f"FAISS embeddings: {len(faiss_ids)}")
    print(f"Missing: {len(missing_ids)}")

    if not missing_ids:
        print("No backfill needed!")
        return {"backfilled": 0, "failed": 0}

    stats = {"backfilled": 0, "failed": 0}
    missing_list = list(missing_ids)

    conn = sqlite3.connect(store.sqlite_path)
    cursor = conn.cursor()

    for i in range(0, len(missing_list), batch_size):
        batch_ids = missing_list[i:i + batch_size]

        # Get contexts for embedding
        contexts = []
        for mid in batch_ids:
            ctx = get_memory_context(store, mid)
            contexts.append(ctx if isinstance(ctx, str) else str(ctx))

        try:
            # Generate embeddings in parallel
            embeddings = embedder.embed_batch(contexts, max_workers=len(embedder.available_indices))

            # Add to FAISS
            for mid, emb in zip(batch_ids, embeddings):
                try:
                    idx = store._embedding_store.add(mid, emb)
                    cursor.execute(
                        "UPDATE memories SET embedding_idx = ? WHERE id = ?",
                        (idx, mid),
                    )
                    stats["backfilled"] += 1
                except Exception as e:
                    print(f"  Failed to add {mid}: {e}")
                    stats["failed"] += 1
            conn.commit()

        except Exception as e:
            print(f"  Batch embedding failed: {e}")
            stats["failed"] += len(batch_ids)

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(missing_list):
            print(f"  Progress: {min(i + batch_size, len(missing_list))}/{len(missing_list)}")

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill FAISS embeddings for SQLite records")
    parser.add_argument("--servers", type=str, default=",".join(DEFAULT_SERVERS),
                       help="Comma-separated list of embedding server URLs")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for parallel embedding")
    args = parser.parse_args()

    print("=== Backfill FAISS Embeddings ===\n")

    # Initialize
    server_urls = args.servers.split(",")
    print(f"Initializing {len(server_urls)} embedding servers...")
    embedder = ParallelEmbedder(server_urls)

    print("Opening episodic store...")
    store = EpisodicStore()

    # Backfill
    print("\nBackfilling missing embeddings...")
    stats = backfill_embeddings(store, embedder, args.batch_size)

    # Flush and save
    print("\nFlushing to disk...")
    store.flush()
    if hasattr(store, '_embedding_store'):
        store._embedding_store.save()

    # Final stats
    final_sqlite = len(get_all_memory_ids(store))
    final_faiss = store._embedding_store.count

    print(f"\n=== Results ===")
    print(f"Backfilled: {stats['backfilled']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nFinal counts:")
    print(f"  SQLite: {final_sqlite}")
    print(f"  FAISS: {final_faiss}")

    if final_sqlite == final_faiss:
        print("  ✓ Counts match!")
    else:
        print(f"  ⚠ Mismatch: {final_sqlite - final_faiss} records without embeddings")

    embedder.close()


if __name__ == "__main__":
    main()
