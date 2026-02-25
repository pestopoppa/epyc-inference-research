#!/usr/bin/env python3
from __future__ import annotations

"""
Q-Scorer Runner: Periodic job for updating Q-values in episodic memory.

This script runs the Q-scorer to process completed tasks from progress logs
and update Q-values in the episodic store.

Usage:
    # Run once
    python3 scripts/q_scorer_runner.py

    # Run with verbose output
    python3 scripts/q_scorer_runner.py --verbose

    # Force scoring (ignore min interval)
    python3 scripts/q_scorer_runner.py --force

    # Dry run (no writes)
    python3 scripts/q_scorer_runner.py --dry-run

Cron example (run every 5 minutes):
    */5 * * * * /home/daniele/miniforge3/bin/python3 /mnt/raid0/llm/claude/scripts/q_scorer_runner.py >> /mnt/raid0/llm/claude/logs/q_scorer.log 2>&1

Systemd timer setup:
    See scripts/systemd/q-scorer.timer and q-scorer.service
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.progress_logger import ProgressLogger, ProgressReader
from orchestration.repl_memory.q_scorer import QScorer, ScoringConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run Q-scorer to update episodic memory Q-values"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force scoring (ignore min interval)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Dry run - don't write to database"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=50,
        help="Maximum tasks to process per run (default: 50)"
    )
    parser.add_argument(
        "--learning-rate", "-l",
        type=float,
        default=0.1,
        help="Learning rate for Q-value updates (default: 0.1)"
    )
    args = parser.parse_args()

    timestamp = datetime.utcnow().isoformat()

    if args.verbose:
        print(f"[{timestamp}] Q-Scorer starting...")

    # Initialize components
    try:
        store = EpisodicStore()
        embedder = TaskEmbedder()
        logger = ProgressLogger()
        reader = ProgressReader()

        config = ScoringConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            min_score_interval_seconds=0 if args.force else 300,
        )

        scorer = QScorer(
            store=store,
            embedder=embedder,
            logger=logger,
            reader=reader,
            config=config,
        )

        if args.verbose:
            print(f"[{timestamp}] Components initialized")

    except Exception as e:
        print(f"[{timestamp}] ERROR: Failed to initialize components: {e}")
        return 1

    # Check for unscored tasks first
    unscored = reader.get_unscored_tasks()
    if args.verbose:
        print(f"[{timestamp}] Found {len(unscored)} unscored tasks")

    if not unscored:
        print(f"[{timestamp}] No pending tasks to score")
        return 0

    # Run scoring
    if args.dry_run:
        print(f"[{timestamp}] DRY RUN: Would process {min(len(unscored), args.batch_size)} tasks")
        for task_id in unscored[:5]:
            trajectory = reader.get_task_trajectory(task_id)
            print(f"  - {task_id}: {len(trajectory)} events")
        if len(unscored) > 5:
            print(f"  ... and {len(unscored) - 5} more")
        return 0

    try:
        results = scorer.score_pending_tasks()

        if results.get("skipped"):
            if args.verbose:
                print(f"[{timestamp}] Skipped: {results.get('reason')}")
            return 0

        # Log results
        print(f"[{timestamp}] Q-Scorer completed:")
        print(f"  Tasks processed: {results.get('tasks_processed', 0)}")
        print(f"  Memories updated: {results.get('memories_updated', 0)}")
        print(f"  Memories created: {results.get('memories_created', 0)}")

        errors = results.get("errors", [])
        if errors:
            print(f"  Errors: {len(errors)}")
            for err in errors[:3]:
                print(f"    - {err.get('task_id')}: {err.get('error')}")

        # Flush logger
        logger.flush()

        return 0 if not errors else 1

    except Exception as e:
        print(f"[{timestamp}] ERROR: Scoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
