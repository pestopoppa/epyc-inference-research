#!/usr/bin/env python3
from __future__ import annotations

"""Seed memories for FAILED strategies and common pitfalls.

These teach the system what NOT to do - critical for learned routing.
Low Q-values (0.0-0.3) signal these strategies should be avoided.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore

# Failure patterns: task -> wrong approach -> why it failed
FAILURE_PATTERNS = {
    # === WRONG ROLE SELECTION ===
    "worker_for_architecture": {
        "tasks": [
            "Design the authentication system architecture",
            "Create a scalable microservices design",
            "Plan the database schema for the new feature",
            "Architect the API gateway structure",
            "Design the event-driven messaging system",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Worker lacks context for architectural decisions - produced fragmented, inconsistent design",
        "correct_action": "architect_general",
        "q_value": 0.1,
    },
    "frontdoor_for_complex_code": {
        "tasks": [
            "Implement the distributed consensus algorithm",
            "Write the lock-free concurrent data structure",
            "Build the custom memory allocator",
            "Implement the B-tree with MVCC support",
            "Create the JIT compiler optimization pass",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Frontdoor is for routing/chat, not complex implementation - produced superficial stub code",
        "correct_action": "coder_escalation",
        "q_value": 0.05,
    },
    "coder_for_math_proof": {
        "tasks": [
            "Prove the algorithm's time complexity is O(n log n)",
            "Verify the cryptographic security properties",
            "Derive the optimal parameters mathematically",
            "Prove the invariant holds under all conditions",
            "Show the convergence guarantee for the optimizer",
        ],
        "wrong_action": "coder_escalation",
        "failure_reason": "Coder wrote code instead of mathematical proof - missed formal verification",
        "correct_action": "worker_math",
        "q_value": 0.15,
    },
    "small_model_for_long_context": {
        "tasks": [
            "Analyze this 50-page technical specification",
            "Summarize all findings from the 100-file codebase",
            "Review the entire conversation history and extract action items",
            "Process the full API documentation and create a migration guide",
            "Synthesize information from all 20 research papers",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Worker model context too small - truncated input, missed critical information",
        "correct_action": "ingest_long_context",
        "q_value": 0.1,
    },

    # === MISSING ESCALATION ===
    "no_escalation_on_failure": {
        "tasks": [
            "Fix the race condition (attempt 3, still failing)",
            "Debug the memory corruption (tried 5 approaches)",
            "Resolve the deadlock (workers stuck in loop)",
            "Fix the data inconsistency (multiple failed attempts)",
            "Solve the performance regression (no progress after 4 tries)",
        ],
        "wrong_action": "worker_general",  # Same tier, no escalation
        "failure_reason": "Kept retrying at same tier instead of escalating - wasted cycles, no progress",
        "correct_action": "architect_coding",  # Should escalate
        "q_value": 0.0,
    },
    "premature_escalation": {
        "tasks": [
            "Fix the typo in the error message",
            "Update the version number in package.json",
            "Add a log statement for debugging",
            "Rename the variable for clarity",
            "Fix the off-by-one error in the loop",
        ],
        "wrong_action": "architect_general",
        "failure_reason": "Escalated trivial task to architect - wasted expensive resources on simple fix",
        "correct_action": "worker_general",
        "q_value": 0.2,
    },

    # === WRONG DECOMPOSITION ===
    "no_decomposition_for_complex": {
        "tasks": [
            "Build a complete e-commerce platform",
            "Implement the entire CI/CD pipeline",
            "Create a full authentication system with OAuth, MFA, and SSO",
            "Build the real-time collaboration feature end-to-end",
            "Implement the complete monitoring and alerting stack",
        ],
        "wrong_action": "coder_escalation",  # Single agent, no decomposition
        "failure_reason": "Attempted monolithic implementation - overwhelming scope, incomplete result",
        "correct_action": "decomposition",  # Should decompose first
        "q_value": 0.1,
    },
    "over_decomposition_simple": {
        "tasks": [
            "Add null check before the function call",
            "Fix the CSS padding on the button",
            "Update the README with new installation step",
            "Add the missing import statement",
            "Change the default timeout value",
        ],
        "wrong_action": "decomposition",
        "failure_reason": "Over-engineered simple task with unnecessary planning overhead",
        "correct_action": "worker_general",
        "q_value": 0.25,
    },

    # === TOOL MISUSE ===
    "code_when_tool_needed": {
        "tasks": [
            "What's the current Bitcoin price?",
            "Find recent papers on transformer architectures",
            "Get the weather forecast for tomorrow",
            "Look up the API rate limits in the documentation",
            "Search for the error message in Stack Overflow",
        ],
        "wrong_action": "coder_escalation",
        "failure_reason": "Wrote code to fetch data instead of using search/API tool - slower, less accurate",
        "correct_action": "tool_websearch",
        "q_value": 0.15,
    },
    "tool_when_reasoning_needed": {
        "tasks": [
            "Should we use PostgreSQL or MongoDB for this use case?",
            "What's the best architecture pattern for our requirements?",
            "How should we handle backward compatibility?",
            "What are the security implications of this design?",
            "Which caching strategy fits our access patterns?",
        ],
        "wrong_action": "tool_websearch",
        "failure_reason": "Searched for generic advice instead of reasoning about specific context",
        "correct_action": "architect_general",
        "q_value": 0.2,
    },

    # === SPECIFICATION FAILURES ===
    "vague_to_coder_directly": {
        "tasks": [
            "Make the app faster",
            "Improve the user experience",
            "Fix the performance issues",
            "Make it more scalable",
            "Clean up the codebase",
        ],
        "wrong_action": "coder_escalation",
        "failure_reason": "Vague requirement sent directly to coder - random changes, no clear improvement",
        "correct_action": "formalizer",  # Should formalize first
        "q_value": 0.1,
    },
    "formal_spec_for_exploration": {
        "tasks": [
            "What does this codebase do?",
            "How does the authentication flow work?",
            "Explain the data model",
            "What are the main components?",
            "Give me an overview of the architecture",
        ],
        "wrong_action": "formalizer",
        "failure_reason": "Tried to formalize exploratory question - unnecessary rigidity, missed the point",
        "correct_action": "frontdoor",
        "q_value": 0.2,
    },

    # === PARALLELIZATION FAILURES ===
    "parallel_with_dependencies": {
        "tasks": [
            "Create the database schema, then write the migrations, then seed the data",
            "Build the API, then write tests that call it, then document the endpoints",
            "Design the interface, implement it, then integrate with existing code",
            "Parse the config, validate it, then apply the settings",
            "Fetch the data, transform it, then store the results",
        ],
        "wrong_action": "parallel_workers",
        "failure_reason": "Parallelized sequential tasks - race conditions, missing dependencies, failures",
        "correct_action": "sequential_decomposition",
        "q_value": 0.05,
    },
    "sequential_when_parallel_possible": {
        "tasks": [
            "Run linting, type checking, and unit tests",
            "Generate reports for sales, marketing, and support",
            "Validate user input, file format, and permissions",
            "Check API health for auth, payments, and notifications services",
            "Compress images in folder A, B, and C",
        ],
        "wrong_action": "sequential_workers",
        "failure_reason": "Ran independent tasks sequentially - unnecessary latency, poor resource utilization",
        "correct_action": "parallel_workers",
        "q_value": 0.3,
    },

    # === MODEL CAPABILITY MISMATCH ===
    "vision_task_to_text_model": {
        "tasks": [
            "Describe what's in this screenshot",
            "Extract text from the uploaded image",
            "Analyze the chart in the attached PNG",
            "Read the error message from the terminal screenshot",
            "Identify the UI elements in this mockup",
        ],
        "wrong_action": "coder_escalation",
        "failure_reason": "Text model cannot process images - hallucinated or refused",
        "correct_action": "worker_vision",
        "q_value": 0.0,
    },
    "text_model_for_code_generation": {
        "tasks": [
            "Generate Python code for the sorting algorithm",
            "Write the TypeScript interface definitions",
            "Create the SQL migration scripts",
            "Implement the React component",
            "Write the bash deployment script",
        ],
        "wrong_action": "frontdoor",  # General chat model
        "failure_reason": "General model produced syntactically broken code - missing imports, wrong patterns",
        "correct_action": "coder_escalation",  # Code-specialized model
        "q_value": 0.15,
    },

    # === CONTEXT/STATE FAILURES ===
    "stateless_for_multi_turn": {
        "tasks": [
            "Continue implementing the feature from our last session",
            "Apply the feedback I gave on the previous version",
            "Use the same approach we discussed earlier",
            "Build on top of what you already created",
            "Fix the issues I mentioned in my previous message",
        ],
        "wrong_action": "worker_general",  # Stateless worker
        "failure_reason": "Stateless worker lost context - started from scratch, ignored prior work",
        "correct_action": "coder_escalation",  # Stateful, maintains context
        "q_value": 0.1,
    },
    "ignored_conversation_context": {
        "tasks": [
            "As I said, use the factory pattern",
            "Remember, we're targeting Python 3.8",
            "Don't forget the constraints I mentioned",
            "Apply the same style as before",
            "Keep the changes minimal like I asked",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Ignored prior context - produced incompatible or unwanted output",
        "correct_action": "coder_escalation",
        "q_value": 0.15,
    },

    # === SECURITY/SAFETY FAILURES ===
    "unsafe_code_execution": {
        "tasks": [
            "Run this user-provided Python script",
            "Execute the shell command from the input field",
            "Eval the JavaScript expression the user typed",
            "Process the uploaded pickle file",
            "Run the SQL query from the request parameter",
        ],
        "wrong_action": "tool_script",  # Direct execution
        "failure_reason": "Executed untrusted code without sandboxing - security vulnerability",
        "correct_action": "restricted_executor",  # Sandboxed execution
        "q_value": 0.0,
    },
    "exposed_secrets": {
        "tasks": [
            "Show me the full config including API keys",
            "Log the authentication token for debugging",
            "Print the database connection string",
            "Display the environment variables",
            "Include the credentials in the error message",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Exposed sensitive credentials in output - security breach",
        "correct_action": "redacted_output",
        "q_value": 0.0,
    },
}


def seed_failure_memories(store: EpisodicStore, embedder: TaskEmbedder,
                          count_per_pattern: int = 5) -> int:
    """Seed failure memories with low Q-values."""
    seeded = 0

    for pattern_name, pattern in FAILURE_PATTERNS.items():
        print(f"\nSeeding failures for: {pattern_name}")

        tasks = pattern["tasks"]
        wrong_action = pattern["wrong_action"]
        q_value = pattern["q_value"]
        failure_reason = pattern["failure_reason"]
        correct_action = pattern["correct_action"]

        for i, task in enumerate(tasks[:count_per_pattern]):
            # Create task IR
            task_ir = {
                "task_type": "general",
                "objective": task,
                "priority": "interactive",
            }

            # Store the WRONG action with low Q-value
            context = {
                **task_ir,
                "failure_reason": failure_reason,
                "correct_action": correct_action,
            }

            try:
                embedding = embedder.embed_task_ir(task_ir)
                store.store(
                    embedding=embedding,
                    action=wrong_action,
                    action_type="routing",
                    context=context,
                    outcome="failure",
                    initial_q=q_value,
                )
                seeded += 1
            except Exception as e:
                print(f"  Error: {e}")
                continue

        print(f"  Seeded {min(count_per_pattern, len(tasks))} failures (Q={q_value})")

    return seeded


def main():
    count_per_pattern = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print("Initializing...")
    embedder = TaskEmbedder()
    store = EpisodicStore()

    print(f"Seeding {count_per_pattern} failure memories per pattern...")
    print(f"Patterns: {len(FAILURE_PATTERNS)}")

    seeded = seed_failure_memories(store, embedder, count_per_pattern)

    print(f"\n=== Results ===")
    print(f"Seeded: {seeded} failure memories")
    stats = store.get_stats()
    print(f"Total memories: {stats['total_memories']}")

    # Show Q-value distribution
    print(f"\nBy action type:")
    for action_type, info in stats['by_action_type'].items():
        print(f"  {action_type}: {info['count']} memories, avg Q={info['avg_q']:.4f}")


if __name__ == "__main__":
    main()
