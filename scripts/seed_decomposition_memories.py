#!/usr/bin/env python3
from __future__ import annotations

"""Seed memories for hierarchical task decomposition patterns.

These memories teach the system how to break down complex tasks into
structured plans with appropriate routing for each subtask.

The action stored is a JSON plan structure, not a single role.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore

# Decomposition patterns: complex task -> structured plan
DECOMPOSITION_PATTERNS = {
    "feature_implementation": {
        "task_type": "code",
        "templates": [
            "Implement {feature} for {system}",
            "Add {feature} functionality to {component}",
            "Build the {feature} module from scratch",
            "Create a complete {feature} solution",
        ],
        "vars": {
            "feature": ["user authentication", "payment processing", "file upload",
                       "search functionality", "notification system", "rate limiting",
                       "caching layer", "API versioning", "logging infrastructure"],
            "system": ["the web app", "our API", "the mobile backend", "the microservice"],
            "component": ["user service", "order system", "content management", "admin panel"],
        },
        "plan": {
            "type": "hierarchical",
            "steps": [
                {"role": "architect_general", "task": "Design high-level architecture and interfaces"},
                {"role": "coder_escalation", "task": "Implement core functionality"},
                {"role": "worker_general", "task": "Write unit tests", "parallel": True},
                {"role": "worker_general", "task": "Write integration tests", "parallel": True},
                {"role": "coder_escalation", "task": "Review and refactor"},
            ],
        },
    },
    "bug_investigation": {
        "task_type": "debug",
        "templates": [
            "Debug the {issue} in {component}",
            "Investigate why {symptom} is happening",
            "Find and fix the root cause of {problem}",
            "Troubleshoot the {error_type} errors in production",
        ],
        "vars": {
            "issue": ["memory leak", "race condition", "performance degradation",
                     "data corruption", "authentication failure", "timeout issue"],
            "component": ["API gateway", "database layer", "cache service", "message queue"],
            "symptom": ["requests are timing out", "memory usage is growing",
                       "data is inconsistent", "users are getting 500 errors"],
            "problem": ["the intermittent failures", "the slow response times",
                       "the data sync issues", "the connection drops"],
            "error_type": ["NullPointer", "OutOfMemory", "Connection", "Timeout"],
        },
        "plan": {
            "type": "diagnostic",
            "steps": [
                {"role": "tool_logs", "task": "Gather relevant logs and metrics"},
                {"role": "coder_escalation", "task": "Analyze error patterns and stack traces"},
                {"role": "architect_coding", "task": "Identify potential root causes"},
                {"role": "coder_escalation", "task": "Implement and test fix"},
                {"role": "tool_monitoring", "task": "Verify fix in monitoring"},
            ],
        },
    },
    "system_design": {
        "task_type": "design",
        "templates": [
            "Design the architecture for {system}",
            "Create a system design for {requirement}",
            "Architect a solution for {challenge}",
            "Plan the technical approach for {project}",
        ],
        "vars": {
            "system": ["a real-time chat application", "an e-commerce platform",
                      "a content delivery network", "a recommendation engine",
                      "a distributed task queue", "a data pipeline"],
            "requirement": ["handling 1M concurrent users", "sub-100ms latency globally",
                          "99.99% uptime SLA", "GDPR compliance"],
            "challenge": ["scaling to 10x traffic", "migrating to microservices",
                         "implementing multi-region deployment", "reducing costs by 50%"],
            "project": ["the platform modernization", "the cloud migration",
                       "the new product launch", "the infrastructure overhaul"],
        },
        "plan": {
            "type": "design",
            "steps": [
                {"role": "architect_general", "task": "Define requirements and constraints"},
                {"role": "architect_general", "task": "Design high-level architecture"},
                {"role": "architect_coding", "task": "Design key component interfaces"},
                {"role": "tool_diagram", "task": "Create architecture diagrams"},
                {"role": "architect_general", "task": "Identify risks and trade-offs"},
                {"role": "formalizer", "task": "Create implementation roadmap"},
            ],
        },
    },
    "data_analysis": {
        "task_type": "analysis",
        "templates": [
            "Analyze {dataset} and provide insights",
            "Investigate the trends in {data_source}",
            "Find patterns in {data_type} data",
            "Create a report on {analysis_topic}",
        ],
        "vars": {
            "dataset": ["user behavior data", "sales transactions", "error logs",
                       "performance metrics", "customer feedback"],
            "data_source": ["the last quarter's data", "production logs",
                           "A/B test results", "user surveys"],
            "data_type": ["time series", "categorical", "geospatial", "clickstream"],
            "analysis_topic": ["conversion funnel", "user retention",
                              "system performance", "error rates"],
        },
        "plan": {
            "type": "analytical",
            "steps": [
                {"role": "tool_data", "task": "Load and validate data"},
                {"role": "worker_math", "task": "Calculate summary statistics"},
                {"role": "tool_script", "task": "Run statistical tests"},
                {"role": "coder_escalation", "task": "Generate visualizations"},
                {"role": "frontdoor", "task": "Synthesize findings into report"},
            ],
        },
    },
    "security_audit": {
        "task_type": "security",
        "templates": [
            "Perform a security audit of {target}",
            "Review the security posture of {system}",
            "Identify vulnerabilities in {component}",
            "Assess the security risks of {change}",
        ],
        "vars": {
            "target": ["the authentication system", "the API endpoints",
                      "the data storage layer", "the third-party integrations"],
            "system": ["our production environment", "the user-facing services",
                      "the admin interfaces", "the internal tools"],
            "component": ["the login flow", "the payment processing",
                         "the file upload", "the session management"],
            "change": ["the new deployment", "the library upgrade",
                      "the architecture change", "the new feature"],
        },
        "plan": {
            "type": "audit",
            "steps": [
                {"role": "architect_general", "task": "Define audit scope and criteria"},
                {"role": "tool_security", "task": "Run automated security scans"},
                {"role": "coder_escalation", "task": "Review code for vulnerabilities"},
                {"role": "architect_coding", "task": "Analyze authentication/authorization"},
                {"role": "frontdoor", "task": "Document findings and recommendations"},
            ],
        },
    },
    "documentation": {
        "task_type": "doc",
        "templates": [
            "Document the {subject} comprehensively",
            "Create API documentation for {api}",
            "Write a guide for {topic}",
            "Update the documentation for {change}",
        ],
        "vars": {
            "subject": ["authentication system", "deployment process",
                       "API endpoints", "data models"],
            "api": ["user management", "payment processing", "search", "notifications"],
            "topic": ["getting started", "best practices", "troubleshooting", "architecture"],
            "change": ["the new features", "the API changes", "the migration", "the refactor"],
        },
        "plan": {
            "type": "documentation",
            "steps": [
                {"role": "coder_escalation", "task": "Analyze code structure and interfaces"},
                {"role": "tool_code", "task": "Extract API signatures and types"},
                {"role": "frontdoor", "task": "Write clear explanations"},
                {"role": "worker_general", "task": "Create code examples"},
                {"role": "frontdoor", "task": "Review and finalize documentation"},
            ],
        },
    },
    "refactoring": {
        "task_type": "code",
        "templates": [
            "Refactor {component} to improve {quality}",
            "Modernize the {legacy} codebase",
            "Clean up and optimize {target}",
            "Restructure {code} for better {goal}",
        ],
        "vars": {
            "component": ["the data access layer", "the authentication module",
                         "the API handlers", "the utility functions"],
            "quality": ["maintainability", "performance", "testability", "readability"],
            "legacy": ["authentication", "database access", "API", "configuration"],
            "target": ["the core business logic", "the test suite", "the build system"],
            "code": ["the monolithic module", "the tightly coupled classes",
                    "the duplicated logic", "the complex conditionals"],
            "goal": ["extensibility", "separation of concerns", "dependency injection"],
        },
        "plan": {
            "type": "refactoring",
            "steps": [
                {"role": "architect_coding", "task": "Analyze current structure and identify issues"},
                {"role": "architect_coding", "task": "Design target architecture"},
                {"role": "worker_general", "task": "Add tests for existing behavior"},
                {"role": "coder_escalation", "task": "Implement refactoring in small steps"},
                {"role": "worker_general", "task": "Verify tests still pass"},
                {"role": "coder_escalation", "task": "Clean up and document changes"},
            ],
        },
    },
}


def generate_task_and_plan(pattern_name: str) -> tuple[dict, dict]:
    """Generate a task and its decomposition plan."""
    pattern = DECOMPOSITION_PATTERNS[pattern_name]

    template = random.choice(pattern["templates"])
    objective = template
    for var, options in pattern["vars"].items():
        placeholder = "{" + var + "}"
        if placeholder in objective:
            objective = objective.replace(placeholder, random.choice(options), 1)

    task_ir = {
        "task_type": pattern["task_type"],
        "objective": objective,
        "priority": random.choice(["interactive", "batch"]),
    }

    # Create the action as a structured plan
    plan = pattern["plan"].copy()
    plan["objective"] = objective

    return task_ir, plan


def seed_decomposition_memories(store: EpisodicStore, embedder: TaskEmbedder,
                                count_per_pattern: int = 20) -> int:
    """Seed memories for hierarchical decomposition."""
    seeded = 0

    for pattern_name in DECOMPOSITION_PATTERNS:
        print(f"\nSeeding {count_per_pattern} decomposition memories for: {pattern_name}")

        for i in range(count_per_pattern):
            task_ir, plan = generate_task_and_plan(pattern_name)

            # Most decompositions succeed
            if random.random() < 0.05:
                outcome = "failure"
                initial_q = 0.25
            else:
                outcome = "success"
                initial_q = 1.0

            # Action is the JSON-serialized plan
            action = json.dumps(plan)

            try:
                embedding = embedder.embed_task_ir(task_ir)
                store.store(
                    embedding=embedding,
                    action=action,
                    action_type="decomposition",  # New action type!
                    context=task_ir,
                    outcome=outcome,
                    initial_q=initial_q,
                )
                seeded += 1
            except Exception as e:
                print(f"  Error: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{count_per_pattern}")

    return seeded


def main():
    count_per_pattern = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    print("Initializing...")
    embedder = TaskEmbedder()
    store = EpisodicStore()

    print(f"Seeding {count_per_pattern} decomposition memories per pattern...")
    print(f"Patterns: {list(DECOMPOSITION_PATTERNS.keys())}")

    seeded = seed_decomposition_memories(store, embedder, count_per_pattern)

    print(f"\n=== Results ===")
    print(f"Seeded: {seeded} decomposition memories")
    stats = store.get_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"\nBy action type:")
    for action_type, info in stats['by_action_type'].items():
        print(f"  {action_type}: {info['count']} memories, avg Q={info['avg_q']:.4f}")


if __name__ == "__main__":
    main()
