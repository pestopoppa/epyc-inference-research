#!/usr/bin/env python3
"""Seed success patterns - routing, tool selection, escalation memories.

Generates ~3,100 high-quality success memories from model_registry and tool_registry.
Uses parallel embedding across multiple servers for speed.

Usage:
    python scripts/seed_success_patterns.py [--servers PORT1,PORT2,...] [--dry-run]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import httpx
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.episodic_store import EpisodicStore

# Default embedding servers (start with: scripts/start_embed_servers.sh)
DEFAULT_SERVERS = ["http://127.0.0.1:8090", "http://127.0.0.1:8091",
                   "http://127.0.0.1:8092", "http://127.0.0.1:8093"]

REGISTRY_PATH = Path(__file__).parent.parent / "orchestration/model_registry.yaml"
TOOL_REGISTRY_PATH = Path(__file__).parent.parent / "orchestration/tool_registry.yaml"


@dataclass
class SuccessPattern:
    """A success pattern to seed into episodic memory."""
    task: str
    action: str  # Code/action that succeeds
    action_type: str  # routing, tool_use, escalation, decomposition
    q_value: float  # 0.85-0.95 for success
    context: Dict


def load_registries() -> Tuple[Dict, Dict]:
    """Load model and tool registries."""
    with open(REGISTRY_PATH) as f:
        model_reg = yaml.safe_load(f)
    with open(TOOL_REGISTRY_PATH) as f:
        tool_reg = yaml.safe_load(f)
    return model_reg, tool_reg


class ParallelEmbedder:
    """Embed texts across multiple llama-server instances."""

    def __init__(self, server_urls: List[str]):
        self.servers = server_urls
        self.clients = [httpx.Client(timeout=30.0) for _ in server_urls]
        self.server_cycle: Iterator = cycle(range(len(server_urls)))
        self._verify_servers()

    def _verify_servers(self):
        """Check which servers are available."""
        available = []
        for i, (url, client) in enumerate(zip(self.servers, self.clients)):
            try:
                resp = client.post(f"{url}/embedding",
                                   json={"content": "test"})
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
        """Embed a single text using round-robin server selection."""
        idx = next(self.server_cycle)
        url = self.servers[idx]
        client = self.clients[idx]

        try:
            resp = client.post(f"{url}/embedding", json={"content": text})
            data = resp.json()
            # Handle llama-server format: [{"index": 0, "embedding": [[...]]}]
            emb = data[0]["embedding"][0]
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Embedding failed on {url}: {e}")

    def embed_batch(self, texts: List[str], max_workers: int = 4) -> List[np.ndarray]:
        """Embed multiple texts in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.embed_one, t) for t in texts]
            return [f.result() for f in futures]

    def close(self):
        for client in self.clients:
            client.close()


def generate_routing_patterns(model_reg: Dict) -> List[SuccessPattern]:
    """Generate task→role routing success patterns."""
    patterns = []

    # Extract roles from server_mode
    server_mode = model_reg.get("server_mode", {})
    roles = {
        "frontdoor": {
            "tasks": ["quick question", "simple lookup", "intent classification",
                     "route this request", "hello world", "basic chat"],
            "q": 0.90,
        },
        "coder_escalation": {
            "tasks": ["implement {feature}", "write a function to {action}",
                     "refactor this code", "fix this bug", "add error handling",
                     "write unit tests for", "optimize this algorithm"],
            "vars": {
                "feature": ["authentication", "caching", "logging", "pagination",
                           "file upload", "data validation", "API endpoint"],
                "action": ["parse JSON", "validate email", "sort items",
                          "filter data", "transform CSV", "calculate hash"],
            },
            "q": 0.92,
        },
        "architect_general": {
            "tasks": ["design the architecture for {system}",
                     "plan the implementation of {feature}",
                     "review this system design", "evaluate trade-offs for",
                     "create a technical spec for"],
            "vars": {
                "system": ["microservices", "event-driven system", "API gateway",
                          "distributed cache", "message queue", "load balancer"],
                "feature": ["real-time notifications", "batch processing",
                           "data pipeline", "authentication system"],
            },
            "q": 0.88,
        },
        "architect_coding": {
            "tasks": ["debug this complex concurrency issue",
                     "optimize this critical algorithm",
                     "design a lock-free data structure",
                     "implement distributed consensus",
                     "fix this memory corruption bug"],
            "q": 0.85,
        },
        "ingest_long_context": {
            "tasks": ["summarize this {doc_type}",
                     "extract key points from these {count} documents",
                     "analyze this research paper", "synthesize these findings"],
            "vars": {
                "doc_type": ["whitepaper", "technical documentation", "codebase",
                            "API specification", "research paper"],
                "count": ["5", "10", "20", "50"],
            },
            "q": 0.90,
        },
        "worker_general": {
            "tasks": ["format this {content}", "rewrite this {target}",
                     "proofread this text", "translate to {language}"],
            "vars": {
                "content": ["code", "documentation", "JSON", "YAML", "markdown"],
                "target": ["function", "class", "module", "README"],
                "language": ["Python", "JavaScript", "TypeScript", "Go"],
            },
            "q": 0.88,
        },
    }

    for role, config in roles.items():
        tasks = config["tasks"]
        q = config["q"]
        vars_dict = config.get("vars", {})

        for task_template in tasks:
            # Expand template variables
            if "{" in task_template and vars_dict:
                for var_name, var_values in vars_dict.items():
                    if f"{{{var_name}}}" in task_template:
                        for val in var_values[:8]:  # Limit expansion
                            task = task_template.replace(f"{{{var_name}}}", val)
                            patterns.append(SuccessPattern(
                                task=task,
                                action=f"route_to('{role}')",
                                action_type="routing",
                                q_value=q + random.uniform(-0.03, 0.03),
                                context={
                                    "role": role,
                                    "routing_reason": f"Task matches {role} capabilities",
                                    "is_seed": True,
                                },
                            ))
            else:
                patterns.append(SuccessPattern(
                    task=task_template,
                    action=f"route_to('{role}')",
                    action_type="routing",
                    q_value=q + random.uniform(-0.03, 0.03),
                    context={
                        "role": role,
                        "routing_reason": f"Task matches {role} capabilities",
                        "is_seed": True,
                    },
                ))

    return patterns


def generate_tool_patterns(tool_reg: Dict) -> List[SuccessPattern]:
    """Generate objective→tool selection success patterns."""
    patterns = []

    tools = tool_reg.get("tools", {})

    # Task templates for each tool category
    task_templates = {
        "http_get": ["fetch data from {url}", "get the content at {url}",
                    "retrieve {resource} from API"],
        "http_post": ["send data to {endpoint}", "post {payload} to API",
                     "submit form to {url}"],
        "web_search": ["search for {query}", "find information about {topic}",
                      "look up {subject} online"],
        "json_query": ["extract {field} from JSON", "query the data for {pattern}",
                      "filter JSON by {criteria}"],
        "csv_to_json": ["convert this CSV to JSON", "parse CSV data",
                       "transform tabular data"],
        "sql_query": ["query the database for {data}", "run SQL to find {records}",
                     "aggregate {metrics} with SQL"],
        "python_eval": ["calculate {expression}", "evaluate {formula}",
                       "compute {value}"],
        "run_shell": ["run {command}", "execute shell command for {task}",
                     "use terminal to {action}"],
        "git_status": ["check git status", "see what files changed",
                      "view repository state"],
        "lint_python": ["lint this Python code", "check code style",
                       "find code issues"],
        "calculate": ["compute {math}", "calculate {expression}",
                     "evaluate {formula}"],
        "statistics": ["calculate statistics for {data}", "get mean/std of {values}",
                      "analyze distribution of {numbers}"],
    }

    vars_dict = {
        "url": ["https://api.example.com", "https://data.source.io"],
        "resource": ["user data", "config", "metrics"],
        "endpoint": ["/api/submit", "/webhook", "/data"],
        "payload": ["user profile", "form data", "configuration"],
        "query": ["best practices", "latest news", "documentation"],
        "topic": ["machine learning", "API design", "system architecture"],
        "subject": ["Python async", "database optimization", "caching strategies"],
        "field": ["name", "id", "timestamp", "status"],
        "pattern": ["users[*].email", "data.items[?status=='active']"],
        "criteria": ["status", "date range", "category"],
        "data": ["active users", "recent orders", "error logs"],
        "records": ["matching items", "aggregated results", "filtered entries"],
        "metrics": ["counts", "averages", "totals"],
        "expression": ["2**10", "sum(range(100))", "len(data)"],
        "formula": ["a*b + c", "sqrt(x**2 + y**2)", "factorial(n)"],
        "value": ["hash", "checksum", "total"],
        "command": ["ls -la", "git log", "df -h"],
        "task": ["list files", "check disk space", "view processes"],
        "action": ["find files", "compress data", "monitor system"],
        "math": ["sin(pi/4)", "log2(1024)", "e**2"],
        "numbers": ["response times", "error rates", "request counts"],
        "values": ["measurements", "samples", "observations"],
    }

    for tool_name, templates in task_templates.items():
        if tool_name not in tools:
            continue

        tool_info = tools[tool_name]
        for template in templates:
            # Expand variables
            expanded = [template]
            for var, values in vars_dict.items():
                if f"{{{var}}}" in template:
                    expanded = [t.replace(f"{{{var}}}", v)
                               for t in expanded for v in values[:3]]

            for task in expanded[:10]:  # Limit per tool
                patterns.append(SuccessPattern(
                    task=task,
                    action=f"use_tool('{tool_name}', ...)",
                    action_type="tool_use",
                    q_value=0.88 + random.uniform(-0.03, 0.05),
                    context={
                        "tool": tool_name,
                        "category": tool_info.get("category", "unknown"),
                        "is_seed": True,
                    },
                ))

    return patterns


def generate_escalation_patterns() -> List[SuccessPattern]:
    """Generate successful escalation patterns."""
    patterns = []

    escalation_scenarios = [
        {
            "task": "This code change affects {scope} - need architecture review",
            "from_role": "coder_escalation",
            "to_role": "architect_general",
            "reason": "Cross-cutting architectural concern",
            "vars": {"scope": ["multiple services", "the database schema",
                              "the API contract", "core authentication"]},
        },
        {
            "task": "Worker failed twice on {task_type} - escalating",
            "from_role": "worker_general",
            "to_role": "coder_escalation",
            "reason": "Repeated failure requires specialist",
            "vars": {"task_type": ["complex refactoring", "concurrency fix",
                                  "performance optimization", "security patch"]},
        },
        {
            "task": "This requires deep {domain} expertise",
            "from_role": "frontdoor",
            "to_role": "architect_coding",
            "reason": "Specialist domain knowledge needed",
            "vars": {"domain": ["systems programming", "distributed systems",
                               "compiler design", "cryptography"]},
        },
        {
            "task": "Need to process {size} context - routing to specialist",
            "from_role": "frontdoor",
            "to_role": "ingest_long_context",
            "reason": "Large context requires specialized model",
            "vars": {"size": ["50K tokens", "100K tokens", "multiple documents",
                             "entire codebase"]},
        },
    ]

    for scenario in escalation_scenarios:
        template = scenario["task"]
        vars_dict = scenario.get("vars", {})

        for var, values in vars_dict.items():
            for val in values:
                task = template.replace(f"{{{var}}}", val)
                patterns.append(SuccessPattern(
                    task=task,
                    action=f"escalate(from='{scenario['from_role']}', to='{scenario['to_role']}')",
                    action_type="escalation",
                    q_value=0.85 + random.uniform(-0.02, 0.05),
                    context={
                        "from_role": scenario["from_role"],
                        "to_role": scenario["to_role"],
                        "escalation_reason": scenario["reason"],
                        "is_seed": True,
                    },
                ))

    return patterns


def generate_decomposition_patterns() -> List[SuccessPattern]:
    """Generate multi-step decomposition success patterns."""
    patterns = []

    decomposition_templates = [
        {
            "task": "Implement {feature} with tests and documentation",
            "steps": ["1. Design API interface", "2. Implement core logic",
                     "3. Write unit tests", "4. Add integration tests",
                     "5. Write documentation"],
            "vars": {"feature": ["user authentication", "file upload",
                                "search functionality", "notification system",
                                "rate limiting", "data export"]},
        },
        {
            "task": "Migrate {system} to new architecture",
            "steps": ["1. Analyze current implementation", "2. Design migration plan",
                     "3. Create compatibility layer", "4. Migrate incrementally",
                     "5. Verify and clean up"],
            "vars": {"system": ["database", "API", "frontend", "cache layer"]},
        },
        {
            "task": "Debug and fix {issue}",
            "steps": ["1. Reproduce the issue", "2. Analyze stack trace",
                     "3. Identify root cause", "4. Implement fix",
                     "5. Add regression test"],
            "vars": {"issue": ["memory leak", "race condition", "deadlock",
                              "performance regression", "data corruption"]},
        },
        {
            "task": "Review and improve {aspect} of the codebase",
            "steps": ["1. Analyze current state", "2. Identify improvement areas",
                     "3. Prioritize changes", "4. Implement improvements",
                     "5. Validate results"],
            "vars": {"aspect": ["test coverage", "error handling", "logging",
                               "type safety", "documentation"]},
        },
    ]

    for template in decomposition_templates:
        task_template = template["task"]
        steps = template["steps"]
        vars_dict = template.get("vars", {})

        for var, values in vars_dict.items():
            for val in values:
                task = task_template.replace(f"{{{var}}}", val)
                patterns.append(SuccessPattern(
                    task=task,
                    action=f"decompose([{', '.join(repr(s) for s in steps)}])",
                    action_type="decomposition",
                    q_value=0.90 + random.uniform(-0.02, 0.03),
                    context={
                        "step_count": len(steps),
                        "steps": steps,
                        "is_seed": True,
                    },
                ))

    return patterns


def seed_patterns(patterns: List[SuccessPattern], embedder: ParallelEmbedder,
                  store: EpisodicStore, batch_size: int = 20) -> Dict:
    """Seed patterns into episodic memory with parallel embedding."""
    stats = {"seeded": 0, "failed": 0, "by_type": {}}

    # Process in batches for efficiency
    for i in range(0, len(patterns), batch_size):
        batch = patterns[i:i + batch_size]
        tasks = [p.task for p in batch]

        try:
            embeddings = embedder.embed_batch(tasks, max_workers=len(embedder.available_indices))

            for pattern, embedding in zip(batch, embeddings):
                try:
                    store.store(
                        embedding=embedding,
                        action=pattern.action,
                        action_type=pattern.action_type,
                        context=pattern.context,
                        outcome="success",
                        initial_q=pattern.q_value,
                    )
                    stats["seeded"] += 1
                    stats["by_type"][pattern.action_type] = stats["by_type"].get(
                        pattern.action_type, 0) + 1
                except Exception as e:
                    print(f"  Failed to store: {e}")
                    stats["failed"] += 1
        except Exception as e:
            print(f"  Batch embedding failed: {e}")
            stats["failed"] += len(batch)

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(patterns):
            print(f"  Progress: {min(i + batch_size, len(patterns))}/{len(patterns)}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Seed success patterns into episodic memory")
    parser.add_argument("--servers", type=str, default=",".join(DEFAULT_SERVERS),
                       help="Comma-separated list of embedding server URLs")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate patterns but don't seed")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for parallel embedding")
    args = parser.parse_args()

    print("=== Seed Success Patterns ===\n")

    # Load registries
    print("Loading registries...")
    model_reg, tool_reg = load_registries()

    # Generate patterns
    print("Generating patterns...")
    routing_patterns = generate_routing_patterns(model_reg)
    print(f"  Routing patterns: {len(routing_patterns)}")

    tool_patterns = generate_tool_patterns(tool_reg)
    print(f"  Tool patterns: {len(tool_patterns)}")

    escalation_patterns = generate_escalation_patterns()
    print(f"  Escalation patterns: {len(escalation_patterns)}")

    decomposition_patterns = generate_decomposition_patterns()
    print(f"  Decomposition patterns: {len(decomposition_patterns)}")

    all_patterns = routing_patterns + tool_patterns + escalation_patterns + decomposition_patterns
    random.shuffle(all_patterns)  # Shuffle for variety
    print(f"\nTotal patterns: {len(all_patterns)}")

    if args.dry_run:
        print("\n[Dry run - not seeding]")
        return

    # Initialize embedder and store
    server_urls = args.servers.split(",")
    print(f"\nInitializing {len(server_urls)} embedding servers...")
    embedder = ParallelEmbedder(server_urls)

    print("Initializing episodic store...")
    store = EpisodicStore()

    # Get baseline stats
    baseline_stats = store.get_stats()
    print(f"Current memories: {baseline_stats['total_memories']}")

    # Seed patterns
    print(f"\nSeeding {len(all_patterns)} patterns...")
    stats = seed_patterns(all_patterns, embedder, store, args.batch_size)

    # Flush and save
    print("\nFlushing to disk...")
    store.flush()
    if hasattr(store, '_embedding_store'):
        store._embedding_store.save()

    # Final stats
    final_stats = store.get_stats()
    print(f"\n=== Results ===")
    print(f"Seeded: {stats['seeded']}")
    print(f"Failed: {stats['failed']}")
    print(f"By type: {stats['by_type']}")
    print(f"\nTotal memories: {final_stats['total_memories']}")
    print(f"FAISS embeddings: {store._embedding_store.count if hasattr(store, '_embedding_store') else 'N/A'}")

    embedder.close()


if __name__ == "__main__":
    main()
