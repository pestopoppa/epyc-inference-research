#!/usr/bin/env python3
"""
Migrate existing NumPy embeddings to FAISS index.

This script converts legacy NumPy mmap-based embedding storage to FAISS IndexFlatIP
while preserving the SQLite metadata unchanged.

Usage:
    python scripts/migrate_to_faiss.py [--db-path PATH] [--dry-run] [--batch-size N]

The migration is non-destructive: original embeddings.npy is preserved until
you explicitly delete it after validating the FAISS index.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Default path (on RAID array per CLAUDE.md requirements)
DEFAULT_DB_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/sessions")


def migrate_to_faiss(
    db_path: Path,
    batch_size: int = 10000,
    dry_run: bool = False,
) -> bool:
    """
    Migrate NumPy embeddings to FAISS index.

    Args:
        db_path: Directory containing episodic.db and embeddings.npy
        batch_size: Number of embeddings to process per batch
        dry_run: If True, only report what would be done

    Returns:
        True if migration successful, False otherwise
    """
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss-cpu not installed. Run: pip install faiss-cpu>=1.7.4")
        return False

    db_path = Path(db_path)

    # Locate files
    npy_path = db_path / "embeddings.npy"
    sqlite_path = db_path / "episodic.db"
    faiss_path = db_path / "embeddings.faiss"
    id_map_path = db_path / "id_map.npy"

    # Validate source files exist
    if not npy_path.exists():
        print(f"ERROR: NumPy embeddings not found at {npy_path}")
        return False

    if not sqlite_path.exists():
        print(f"ERROR: SQLite database not found at {sqlite_path}")
        return False

    # Check if FAISS index already exists
    if faiss_path.exists():
        print(f"WARNING: FAISS index already exists at {faiss_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return False

    # Load existing NumPy embeddings
    print(f"Loading NumPy embeddings from {npy_path}...")
    embeddings = np.load(npy_path, mmap_mode="r")
    n_total, dim = embeddings.shape
    print(f"  Found {n_total} embeddings of dimension {dim}")

    # Get memory IDs from SQLite (ordered by embedding_idx)
    print(f"Loading memory IDs from {sqlite_path}...")
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute(
        "SELECT id, embedding_idx FROM memories ORDER BY embedding_idx"
    ).fetchall()
    conn.close()

    if not rows:
        print("WARNING: No memories found in SQLite database")
        return False

    # Build id_map from SQLite
    # Note: embedding_idx may have gaps if entries were deleted
    max_idx = max(r[1] for r in rows)
    n_memories = len(rows)
    print(f"  Found {n_memories} memories (max embedding_idx: {max_idx})")

    # Validate indices are within bounds
    invalid_indices = [r for r in rows if r[1] >= n_total]
    if invalid_indices:
        print(f"ERROR: {len(invalid_indices)} memories reference invalid embedding indices")
        print(f"  First few: {invalid_indices[:5]}")
        return False

    if dry_run:
        print("\n=== DRY RUN - No changes made ===")
        print(f"Would create FAISS index at {faiss_path}")
        print(f"Would create id_map at {id_map_path}")
        print(f"  - {n_memories} embeddings to migrate")
        print(f"  - Estimated index size: ~{(n_memories * dim * 4) // (1024*1024)} MB")
        return True

    # Create FAISS index
    print(f"\nCreating FAISS IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)

    # Build id_map and add embeddings in batch order
    id_map = []
    embeddings_to_add = []

    for memory_id, embedding_idx in rows:
        id_map.append(memory_id)
        embedding = embeddings[embedding_idx].astype(np.float32)
        embeddings_to_add.append(embedding)

    # Process in batches
    print(f"Adding {n_memories} embeddings in batches of {batch_size}...")
    for i in range(0, n_memories, batch_size):
        batch_end = min(i + batch_size, n_memories)
        batch = np.array(embeddings_to_add[i:batch_end], dtype=np.float32)

        # L2 normalize for cosine similarity
        faiss.normalize_L2(batch)
        index.add(batch)

        pct = (batch_end / n_memories) * 100
        print(f"  Added {batch_end}/{n_memories} ({pct:.1f}%)")

    # Save FAISS index
    print(f"\nSaving FAISS index to {faiss_path}...")
    faiss.write_index(index, str(faiss_path))
    index_size = faiss_path.stat().st_size / (1024 * 1024)
    print(f"  Index size: {index_size:.2f} MB")

    # Save id_map
    print(f"Saving id_map to {id_map_path}...")
    np.save(id_map_path, np.array(id_map, dtype=object))

    # Validate
    print("\nValidating migration...")
    loaded_index = faiss.read_index(str(faiss_path))
    loaded_id_map = np.load(id_map_path, allow_pickle=True).tolist()

    if loaded_index.ntotal != n_memories:
        print(f"ERROR: Index count mismatch: {loaded_index.ntotal} vs {n_memories}")
        return False

    if len(loaded_id_map) != n_memories:
        print(f"ERROR: id_map count mismatch: {len(loaded_id_map)} vs {n_memories}")
        return False

    print(f"  Index entries: {loaded_index.ntotal}")
    print(f"  id_map entries: {len(loaded_id_map)}")
    print("\nMigration successful!")
    print(f"\nOriginal embeddings preserved at: {npy_path}")
    print("After validating FAISS search results, you can remove it with:")
    print(f"  rm {npy_path}")

    return True


def compare_backends(db_path: Path, n_queries: int = 10) -> None:
    """
    Compare search results between NumPy and FAISS backends.

    Args:
        db_path: Directory containing both backends' data
        n_queries: Number of random queries to compare
    """
    from orchestration.repl_memory import EpisodicStore

    print(f"\n=== Comparing backends with {n_queries} queries ===\n")

    # Initialize both backends
    try:
        numpy_store = EpisodicStore(db_path=db_path, use_faiss=False)
        faiss_store = EpisodicStore(db_path=db_path, use_faiss=True)
    except Exception as e:
        print(f"ERROR: Could not initialize stores: {e}")
        return

    print(f"NumPy backend: {numpy_store.count()} memories")
    print(f"FAISS backend: {faiss_store.count()} memories")

    # Generate random query embeddings
    rng = np.random.default_rng(42)
    dim = 1024  # BGE-large embedding dim

    overlaps = []
    for i in range(n_queries):
        query = rng.random(dim).astype(np.float32)

        # Search both backends
        numpy_results = numpy_store.retrieve_by_similarity(query, k=20)
        faiss_results = faiss_store.retrieve_by_similarity(query, k=20)

        # Compare top-20 IDs
        numpy_ids = {m.id for m in numpy_results}
        faiss_ids = {m.id for m in faiss_results}

        overlap = len(numpy_ids & faiss_ids) / max(len(numpy_ids), len(faiss_ids), 1)
        overlaps.append(overlap)

        print(f"Query {i+1}: {overlap*100:.1f}% overlap in top-20")

    avg_overlap = np.mean(overlaps)
    print(f"\nAverage overlap: {avg_overlap*100:.1f}%")

    if avg_overlap >= 0.95:
        print("PASS: >95% overlap - backends are equivalent")
    else:
        print("WARNING: <95% overlap - investigate differences")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NumPy embeddings to FAISS index"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to database directory (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for adding embeddings (default: 10000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without making changes",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare NumPy and FAISS search results after migration",
    )
    parser.add_argument(
        "--compare-queries",
        type=int,
        default=10,
        help="Number of queries to compare (default: 10)",
    )

    args = parser.parse_args()

    success = migrate_to_faiss(
        db_path=args.db_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if success and args.compare and not args.dry_run:
        compare_backends(args.db_path, n_queries=args.compare_queries)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
