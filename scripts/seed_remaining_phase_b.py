#!/usr/bin/env python3
"""Run remaining Phase B seeding scripts with parallel embedding.

Runs:
- seed_memory_from_logs.py (up to 1000 memories)
- Additional diverse patterns

Uses 8 parallel embedding servers for speed.
"""

from __future__ import annotations

import concurrent.futures
import json
import random
import re
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Iterator, List

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.episodic_store import EpisodicStore

SERVERS = [f"http://127.0.0.1:{p}" for p in range(8090, 8098)]
PROGRESS_DIR = Path("/mnt/raid0/llm/claude/progress")


class ParallelEmbedder:
    """Fast parallel embedding across 8 servers."""

    def __init__(self, server_urls: List[str]):
        self.servers = server_urls
        self.clients = [httpx.Client(timeout=60.0) for _ in server_urls]
        self.available = self._check_servers()
        self.cycle: Iterator = cycle(self.available)
        print(f"  {len(self.available)}/{len(server_urls)} embedding servers ready")

    def _check_servers(self) -> List[int]:
        available = []
        for i, (url, client) in enumerate(zip(self.servers, self.clients)):
            try:
                resp = client.post(f"{url}/embedding", json={"content": "test"})
                if resp.status_code == 200:
                    available.append(i)
            except Exception:
                pass
        return available

    def embed_one(self, text: str) -> np.ndarray:
        idx = next(self.cycle)
        resp = self.clients[idx].post(
            f"{self.servers[idx]}/embedding",
            json={"content": text}
        )
        data = resp.json()
        return np.array(data[0]["embedding"][0], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.available)) as ex:
            return list(ex.map(self.embed_one, texts))

    def close(self):
        for c in self.clients:
            c.close()


def parse_progress_logs(max_entries: int = 500) -> List[dict]:
    """Parse progress logs for task patterns."""
    patterns = []

    # Find recent progress files
    for month_dir in sorted(PROGRESS_DIR.iterdir(), reverse=True)[:3]:
        if not month_dir.is_dir():
            continue
        for log_file in sorted(month_dir.glob("*.md"), reverse=True)[:10]:
            try:
                content = log_file.read_text()
                # Extract task patterns from logs
                # Look for lines with actions/completions
                for line in content.split("\n"):
                    if any(kw in line.lower() for kw in ["completed", "implemented", "fixed", "added", "created"]):
                        # Clean up the line
                        task = re.sub(r'^[#\-*\s]+', '', line).strip()
                        if 10 < len(task) < 200:
                            patterns.append({
                                "task": task,
                                "source": log_file.name,
                                "action_type": "exploration",
                                "q_value": 0.85 + random.uniform(-0.05, 0.1),
                            })
                            if len(patterns) >= max_entries:
                                return patterns
            except Exception:
                continue

    return patterns


def generate_more_routing_patterns() -> List[dict]:
    """Generate additional routing patterns."""
    patterns = []

    # Model-specific routing patterns
    model_tasks = {
        "frontdoor": [
            "What is {topic}?",
            "Explain {concept} briefly",
            "Quick help with {task}",
            "How do I {action}?",
        ],
        "coder_escalation": [
            "Write a {language} function to {action}",
            "Implement {feature} in {language}",
            "Debug this {language} code",
            "Refactor this function for {goal}",
            "Add {feature} to this class",
        ],
        "architect_general": [
            "Design a {system} architecture",
            "Plan the migration from {old} to {new}",
            "Review this system design",
            "Evaluate {approach} vs {alternative}",
        ],
        "ingest_long_context": [
            "Summarize these {count} documents",
            "Extract key insights from this {doc_type}",
            "Compare findings across {topic} papers",
        ],
    }

    vars_pool = {
        "topic": ["async programming", "microservices", "caching", "databases",
                 "testing", "CI/CD", "Docker", "Kubernetes", "APIs"],
        "concept": ["dependency injection", "event sourcing", "CQRS", "DDD",
                   "OAuth", "JWT", "rate limiting", "circuit breakers"],
        "task": ["debugging", "logging", "metrics", "deployment", "scaling"],
        "action": ["parse JSON", "validate input", "handle errors", "cache data",
                  "queue jobs", "send emails", "upload files"],
        "language": ["Python", "TypeScript", "Go", "Rust", "Java"],
        "feature": ["pagination", "search", "filtering", "sorting", "auth"],
        "goal": ["readability", "performance", "testability", "maintainability"],
        "system": ["e-commerce", "chat app", "monitoring", "ETL pipeline"],
        "old": ["monolith", "REST", "MySQL", "on-prem"],
        "new": ["microservices", "GraphQL", "PostgreSQL", "cloud"],
        "approach": ["event-driven", "sync", "batch", "streaming"],
        "alternative": ["request-response", "async", "real-time", "polling"],
        "count": ["5", "10", "20"],
        "doc_type": ["whitepaper", "specification", "codebase"],
    }

    for role, templates in model_tasks.items():
        for template in templates:
            # Expand variables
            expanded = [template]
            for var, values in vars_pool.items():
                if f"{{{var}}}" in template:
                    expanded = [t.replace(f"{{{var}}}", v)
                               for t in expanded for v in values[:4]]

            for task in expanded[:15]:  # Limit per template
                patterns.append({
                    "task": task,
                    "action": f"route_to('{role}')",
                    "action_type": "routing",
                    "q_value": 0.88 + random.uniform(-0.03, 0.05),
                    "context": {"role": role, "is_seed": True},
                })

    return patterns


def seed_patterns(patterns: List[dict], embedder: ParallelEmbedder,
                  store: EpisodicStore, batch_size: int = 40) -> dict:
    """Seed patterns with parallel embedding."""
    stats = {"seeded": 0, "failed": 0}

    for i in range(0, len(patterns), batch_size):
        batch = patterns[i:i + batch_size]
        tasks = [p["task"] for p in batch]

        try:
            embeddings = embedder.embed_batch(tasks)

            for p, emb in zip(batch, embeddings):
                try:
                    store.store(
                        embedding=emb,
                        action=p.get("action", "exploration"),
                        action_type=p.get("action_type", "exploration"),
                        context=p.get("context", {"task_description": p["task"], "is_seed": True}),
                        outcome="success",
                        initial_q=p.get("q_value", 0.85),
                    )
                    stats["seeded"] += 1
                except Exception as e:
                    stats["failed"] += 1
        except Exception as e:
            print(f"  Batch failed: {e}")
            stats["failed"] += len(batch)

        if (i + batch_size) % 200 == 0 or i + batch_size >= len(patterns):
            print(f"  Progress: {min(i + batch_size, len(patterns))}/{len(patterns)}")

    return stats


def main():
    print("=== Phase B Remaining: Parallel Seeding ===\n")

    print("Initializing 8 embedding servers...")
    embedder = ParallelEmbedder(SERVERS)

    print("Opening episodic store...")
    store = EpisodicStore()
    baseline = store.get_stats()
    print(f"Current memories: {baseline['total_memories']}")

    # Parse progress logs
    print("\nParsing progress logs...")
    log_patterns = parse_progress_logs(500)
    print(f"  Found {len(log_patterns)} patterns from logs")

    # Generate more routing patterns
    print("Generating additional routing patterns...")
    routing_patterns = generate_more_routing_patterns()
    print(f"  Generated {len(routing_patterns)} routing patterns")

    # Combine all patterns
    all_patterns = log_patterns + routing_patterns
    random.shuffle(all_patterns)
    print(f"\nTotal patterns to seed: {len(all_patterns)}")

    # Seed
    print("\nSeeding patterns...")
    stats = seed_patterns(all_patterns, embedder, store, batch_size=40)

    # Flush
    print("\nFlushing to disk...")
    store.flush()
    store._embedding_store.save()

    # Final stats
    final = store.get_stats()
    print(f"\n=== Results ===")
    print(f"Seeded: {stats['seeded']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total memories: {final['total_memories']}")
    print(f"FAISS embeddings: {store._embedding_store.count}")

    embedder.close()


if __name__ == "__main__":
    main()
