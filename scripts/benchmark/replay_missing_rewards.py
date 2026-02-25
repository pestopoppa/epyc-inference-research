#!/usr/bin/env python3
"""Replay missing reward injections from 3-way eval checkpoints.

Scans checkpoint JSONL files for questions where rewards_injected < len(rewards),
then re-posts all rewards for those questions to /chat/reward.

Since checkpoints don't record *which* rewards succeeded, all rewards for
affected questions are re-injected. Duplicates are harmless — they update
existing Q-values incrementally.

Usage:
    # Replay all missing rewards (requires orchestrator API on port 8000)
    python3 scripts/benchmark/replay_missing_rewards.py

    # Dry run — show what would be replayed
    python3 scripts/benchmark/replay_missing_rewards.py --dry-run

    # Replay from specific checkpoint file
    python3 scripts/benchmark/replay_missing_rewards.py --file benchmarks/results/eval/3way_20260204_222556.jsonl

    # Custom API URL and timeout
    python3 scripts/benchmark/replay_missing_rewards.py --url http://localhost:8000 --timeout 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "results" / "eval"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30  # seconds — generous to avoid the original bug


def find_entries(paths: list[Path], force_all: bool = False) -> list[dict]:
    """Return checkpoint entries to replay.

    Args:
        paths: Checkpoint files to scan.
        force_all: If True, return ALL entries with rewards (for DB rebuild).
                   If False, only return entries where rewards_injected < len(rewards).
    """
    entries = []
    for path in paths:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"{path.name}:{lineno} — invalid JSON, skipping")
                    continue
                rewards = entry.get("rewards", {})
                injected = entry.get("rewards_injected", 0)

                # Include entry if force_all OR if some rewards are missing
                if rewards and (force_all or injected < len(rewards)):
                    entry["_source_file"] = path.name
                    entry["_missing_count"] = len(rewards) - injected
                    entries.append(entry)
    return entries


def replay_rewards(
    entries: list[dict],
    url: str,
    timeout: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Re-inject rewards for entries with missing injections.

    Returns (total_attempted, total_succeeded).
    """
    if dry_run:
        total = sum(len(e.get("rewards", {})) for e in entries)
        for e in entries:
            rewards = e.get("rewards", {})
            logger.info(
                f"  [dry-run] {e['question_id']}: "
                f"would inject {len(rewards)} rewards "
                f"(had {e['rewards_injected']}/{len(rewards)})"
            )
        return total, 0

    import httpx

    client = httpx.Client(timeout=timeout)
    attempted = 0
    succeeded = 0

    for entry in entries:
        qid = entry["question_id"]
        suite = entry["suite"]
        prompt = entry.get("prompt", "")
        rewards = entry.get("rewards", {})
        metadata = entry.get("metadata", {})
        cost_metrics = metadata.get("cost_metrics", {})

        for action_key, reward in rewards.items():
            attempted += 1
            action_cost = cost_metrics.get(action_key, {})

            context = {
                "task_type": suite,
                "source": "3way_eval_replay",
                "question_id": qid,
                "action_type": "routing",
                "tools_helped": metadata.get("tools_helped", False),
                "tools_neutral": metadata.get("tools_neutral", False),
                "tools_hurt": metadata.get("tools_hurt", False),
                "tool_advantage": metadata.get("tool_advantage", 0),
                "elapsed_seconds": action_cost.get("elapsed_seconds", 0.0),
                "tokens_generated": action_cost.get("tokens_generated", 0),
                "predicted_tps": action_cost.get("predicted_tps", 0.0),
                "generation_ms": action_cost.get("generation_ms", 0.0),
                "tools_used": action_cost.get("tools_used", 0),
            }

            try:
                resp = client.post(
                    f"{url}/chat/reward",
                    json={
                        "task_description": prompt[:200],
                        "action": action_key,
                        "reward": reward,
                        "context": context,
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    succeeded += 1
                else:
                    logger.warning(
                        f"  {qid}/{action_key}: HTTP {resp.status_code}"
                    )
            except Exception as e:
                logger.warning(f"  {qid}/{action_key}: {e}")

        logger.info(f"  {qid}: replayed {len(rewards)} rewards")

    client.close()
    return attempted, succeeded


def main():
    parser = argparse.ArgumentParser(
        description="Replay missing reward injections from eval checkpoints."
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        action="append",
        default=None,
        help="Specific checkpoint file(s) to scan. Default: all 3way_*.jsonl in eval dir.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Orchestrator API URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be replayed without sending requests.",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Replay ALL rewards (for DB rebuild), not just missing ones.",
    )
    args = parser.parse_args()

    if args.file:
        paths = args.file
    else:
        if not EVAL_DIR.exists():
            logger.error(f"Eval directory not found: {EVAL_DIR}")
            sys.exit(1)
        paths = sorted(EVAL_DIR.glob("3way_*.jsonl"))
        if not paths:
            logger.info("No 3way checkpoint files found.")
            sys.exit(0)

    logger.info(f"Scanning {len(paths)} checkpoint file(s)...")
    entries = find_entries(paths, force_all=args.force_all)

    if not entries:
        if args.force_all:
            logger.info("No checkpoint entries with rewards found.")
        else:
            logger.info("No missing rewards found — all checkpoints fully injected.")
        sys.exit(0)

    total_rewards = sum(len(e.get("rewards", {})) for e in entries)
    if args.force_all:
        logger.info(
            f"Found {len(entries)} questions with {total_rewards} total rewards (--force-all)"
        )
    else:
        total_missing = sum(e["_missing_count"] for e in entries)
        logger.info(
            f"Found {len(entries)} questions with missing rewards "
            f"({total_missing} missed out of {total_rewards} total)"
        )

    attempted, succeeded = replay_rewards(entries, args.url, args.timeout, args.dry_run)

    if args.dry_run:
        logger.info(f"Dry run complete. Would attempt {attempted} reward injections.")
    else:
        logger.info(f"Replay complete: {succeeded}/{attempted} rewards injected.")
        if succeeded < attempted:
            logger.warning(f"{attempted - succeeded} rewards still failed — check API health.")
            sys.exit(1)


if __name__ == "__main__":
    main()
