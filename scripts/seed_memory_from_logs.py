#!/usr/bin/env python3
from __future__ import annotations

"""Seed REPL memory from existing progress logs.

Parses task_started + task_completed event pairs from progress logs
and creates memory entries with embeddings and Q-values.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore


def parse_logs(log_dir: Path, max_entries: int = 5000) -> list[dict]:
    """Parse progress logs and extract task pairs."""
    tasks = {}  # task_id -> {started: event, completed: event, routing: event}

    log_files = sorted(log_dir.glob("*.jsonl"))
    entries_parsed = 0

    for log_file in log_files:
        print(f"Parsing {log_file.name}...")
        with open(log_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                task_id = event.get("task_id")
                if not task_id:
                    continue

                event_type = event.get("event_type")
                if event_type == "task_started":
                    tasks[task_id] = {"started": event}
                elif event_type == "routing_decision" and task_id in tasks:
                    tasks[task_id]["routing"] = event
                elif event_type == "task_completed" and task_id in tasks:
                    tasks[task_id]["completed"] = event
                    entries_parsed += 1
                    if entries_parsed >= max_entries:
                        print(f"Reached max entries: {max_entries}")
                        return list(tasks.values())

    return list(tasks.values())


def seed_memory(store: EpisodicStore, embedder: TaskEmbedder,
                tasks: list[dict], batch_size: int = 100) -> int:
    """Seed memory with task data."""
    seeded = 0
    skipped = 0

    for i, task in enumerate(tasks):
        if "started" not in task or "completed" not in task:
            skipped += 1
            continue

        started = task["started"]
        completed = task["completed"]
        routing = task.get("routing", {})

        # Extract task_ir data
        data = started.get("data", {})
        task_type = data.get("task_type", "chat")
        objective = data.get("objective", "")

        if not objective or len(objective) < 5:
            skipped += 1
            continue

        # Build task_ir for embedding
        task_ir = {
            "task_type": task_type,
            "objective": objective[:200],
            "priority": data.get("priority", "interactive"),
        }

        # Determine outcome and Q-value
        outcome = completed.get("outcome", "success")
        if outcome == "success":
            initial_q = 1.0
        elif outcome == "failure":
            initial_q = 0.25
        else:
            initial_q = 0.5

        # Extract routing action
        routing_data = routing.get("data", {})
        routing_decision = routing_data.get("routing", ["frontdoor"])
        action = ",".join(routing_decision)

        # Generate embedding
        try:
            embedding = embedder.embed_task_ir(task_ir)
        except Exception as e:
            print(f"Embedding error: {e}")
            skipped += 1
            continue

        # Store memory
        try:
            store.store(
                embedding=embedding,
                action=action,
                action_type="routing",
                context=task_ir,
                outcome=outcome,
                initial_q=initial_q,
            )
            seeded += 1
        except Exception as e:
            print(f"Store error: {e}")
            skipped += 1
            continue

        if (i + 1) % batch_size == 0:
            print(f"Processed {i + 1} tasks, seeded {seeded}, skipped {skipped}")

    return seeded


def main():
    log_dir = Path("/mnt/raid0/llm/claude/logs/progress")
    max_entries = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print(f"Initializing embedder...")
    embedder = TaskEmbedder()
    print(f"Model available: {embedder.is_model_available}")

    print(f"Initializing store...")
    store = EpisodicStore()

    print(f"Parsing logs (max {max_entries} entries)...")
    tasks = parse_logs(log_dir, max_entries)
    print(f"Found {len(tasks)} task pairs")

    print(f"Seeding memory...")
    seeded = seed_memory(store, embedder, tasks)

    print(f"\n=== Results ===")
    print(f"Seeded: {seeded} memories")
    stats = store.get_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Avg Q-value: {stats['overall_avg_q']:.4f}")


if __name__ == "__main__":
    main()
