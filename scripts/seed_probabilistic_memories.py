#!/usr/bin/env python3
from __future__ import annotations

"""Seed probabilistic memories - same strategy, variable outcomes.

Real strategies don't always succeed or always fail. These memories
capture the uncertainty: "this approach works ~70% of the time."
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore

# Probabilistic patterns: strategies with variable success rates
PROBABILISTIC_PATTERNS = {
    # === RESEARCH & DISCOVERY ===
    "web_search_effectiveness": {
        "tasks": [
            "Find recent research on {topic}",
            "Look up best practices for {topic}",
            "Search for examples of {topic}",
            "Find documentation about {topic}",
        ],
        "vars": {
            "topic": ["distributed consensus", "API design patterns", "ML model optimization",
                      "database indexing strategies", "caching architectures", "security best practices",
                      "performance tuning", "error handling patterns", "testing strategies",
                      "monitoring approaches", "deployment automation", "config management"],
        },
        "action": "tool_websearch",
        "success_rate": 0.7,  # Works 70% of time
        "success_q": 0.85,
        "failure_q": 0.25,
        "success_reason": "Found relevant, high-quality sources",
        "failure_reason": "Results were outdated, irrelevant, or low quality",
    },
    "codebase_exploration": {
        "tasks": [
            "Find where {feature} is implemented",
            "Locate the {feature} handling code",
            "Understand how {feature} works in this codebase",
            "Find all usages of {feature}",
        ],
        "vars": {
            "feature": ["authentication", "error handling", "logging", "caching",
                        "validation", "routing", "middleware", "database access",
                        "API endpoints", "event handling", "state management"],
        },
        "action": "coder_escalation",
        "success_rate": 0.75,
        "success_q": 0.9,
        "failure_q": 0.3,
        "success_reason": "Found the relevant code and understood the pattern",
        "failure_reason": "Codebase was poorly documented or pattern was scattered",
    },

    # === COMMUNICATION ===
    "clarification_request": {
        "tasks": [
            "Ask the user to clarify {ambiguity}",
            "Request more details about {ambiguity}",
            "Get clarification on {ambiguity}",
            "Ask what they mean by {ambiguity}",
        ],
        "vars": {
            "ambiguity": ["the requirements", "the expected behavior", "the priority",
                          "the scope", "the deadline", "the constraints", "the audience",
                          "the format", "the acceptance criteria", "the edge cases"],
        },
        "action": "ask_clarification",
        "success_rate": 0.6,  # Sometimes users appreciate it, sometimes annoyed
        "success_q": 0.8,
        "failure_q": 0.35,
        "success_reason": "User provided helpful clarification, better outcome",
        "failure_reason": "User was frustrated by the question, wanted quick action",
    },
    "propose_alternative": {
        "tasks": [
            "Suggest a different approach to {problem}",
            "Propose an alternative solution for {problem}",
            "Recommend reconsidering {problem}",
            "Offer a different perspective on {problem}",
        ],
        "vars": {
            "problem": ["the architecture", "the timeline", "the technology choice",
                        "the pricing strategy", "the feature scope", "the team structure",
                        "the process", "the vendor selection", "the implementation plan"],
        },
        "action": "architect_general",
        "success_rate": 0.55,
        "success_q": 0.85,
        "failure_q": 0.3,
        "success_reason": "Alternative was well-received, improved the outcome",
        "failure_reason": "User was committed to original approach, felt dismissed",
    },

    # === ESTIMATION & PLANNING ===
    "aggressive_estimate": {
        "tasks": [
            "Give optimistic timeline for {project}",
            "Provide best-case estimate for {project}",
            "Estimate minimum time for {project}",
            "Project fast completion for {project}",
        ],
        "vars": {
            "project": ["the feature", "the migration", "the refactor", "the integration",
                        "the launch", "the fix", "the prototype", "the MVP", "the release"],
        },
        "action": "optimistic_estimator",
        "success_rate": 0.3,  # Usually backfires
        "success_q": 0.7,
        "failure_q": 0.15,
        "success_reason": "Team delivered on aggressive timeline, impressed stakeholders",
        "failure_reason": "Missed deadline, damaged credibility, rushed poor quality",
    },
    "conservative_estimate": {
        "tasks": [
            "Give padded timeline for {project}",
            "Provide worst-case estimate for {project}",
            "Estimate with buffer for {project}",
            "Project safe completion for {project}",
        ],
        "vars": {
            "project": ["the feature", "the migration", "the refactor", "the integration",
                        "the launch", "the fix", "the prototype", "the MVP", "the release"],
        },
        "action": "conservative_estimator",
        "success_rate": 0.8,  # Usually safer
        "success_q": 0.75,
        "failure_q": 0.4,
        "success_reason": "Delivered ahead of schedule, built trust",
        "failure_reason": "Lost opportunity due to slow timeline, competitor moved faster",
    },

    # === CREATIVE WORK ===
    "first_draft_approach": {
        "tasks": [
            "Generate initial {content} quickly",
            "Create rough draft of {content}",
            "Produce quick version of {content}",
            "Write fast first pass at {content}",
        ],
        "vars": {
            "content": ["copy", "design", "proposal", "outline", "mockup",
                        "prototype", "script", "presentation", "report", "plan"],
        },
        "action": "worker_general",
        "success_rate": 0.5,
        "success_q": 0.7,
        "failure_q": 0.35,
        "success_reason": "First draft was good enough, saved iteration time",
        "failure_reason": "First draft was too rough, needed complete rewrite",
    },
    "perfectionist_approach": {
        "tasks": [
            "Carefully craft perfect {content}",
            "Create polished {content} on first try",
            "Produce publication-ready {content}",
            "Write final version of {content} immediately",
        ],
        "vars": {
            "content": ["copy", "design", "proposal", "outline", "mockup",
                        "prototype", "script", "presentation", "report", "plan"],
        },
        "action": "coder_escalation",
        "success_rate": 0.4,
        "success_q": 0.9,
        "failure_q": 0.25,
        "success_reason": "Perfect output impressed stakeholders, no revisions needed",
        "failure_reason": "Spent too long, requirements changed, effort wasted",
    },

    # === PROBLEM SOLVING ===
    "quick_fix_attempt": {
        "tasks": [
            "Try quick fix for {issue}",
            "Apply band-aid solution to {issue}",
            "Do minimal fix for {issue}",
            "Patch {issue} quickly",
        ],
        "vars": {
            "issue": ["the bug", "the performance problem", "the error", "the crash",
                      "the memory leak", "the timeout", "the race condition",
                      "the data corruption", "the integration failure"],
        },
        "action": "worker_general",
        "success_rate": 0.45,
        "success_q": 0.75,
        "failure_q": 0.2,
        "success_reason": "Quick fix resolved the issue efficiently",
        "failure_reason": "Quick fix caused new problems or didn't stick",
    },
    "deep_investigation": {
        "tasks": [
            "Do thorough root cause analysis of {issue}",
            "Investigate {issue} comprehensively",
            "Understand full scope of {issue}",
            "Debug {issue} systematically",
        ],
        "vars": {
            "issue": ["the bug", "the performance problem", "the error", "the crash",
                      "the memory leak", "the timeout", "the race condition",
                      "the data corruption", "the integration failure"],
        },
        "action": "coder_escalation",
        "success_rate": 0.75,
        "success_q": 0.9,
        "failure_q": 0.4,
        "success_reason": "Found and fixed root cause, prevented recurrence",
        "failure_reason": "Investigation took too long, issue was simpler than expected",
    },

    # === AUTOMATION ===
    "automate_immediately": {
        "tasks": [
            "Automate {task} right away",
            "Build automation for {task} now",
            "Create script to handle {task}",
            "Set up automated {task}",
        ],
        "vars": {
            "task": ["deployment", "testing", "reporting", "backups", "monitoring",
                     "data processing", "notifications", "cleanup", "provisioning"],
        },
        "action": "coder_escalation",
        "success_rate": 0.5,
        "success_q": 0.85,
        "failure_q": 0.25,
        "success_reason": "Automation saved significant time over manual process",
        "failure_reason": "Automation took longer to build than doing it manually",
    },
    "manual_first": {
        "tasks": [
            "Do {task} manually first to understand it",
            "Handle {task} by hand before automating",
            "Run {task} manually to validate process",
            "Execute {task} manually this time",
        ],
        "vars": {
            "task": ["deployment", "testing", "reporting", "backups", "monitoring",
                     "data processing", "notifications", "cleanup", "provisioning"],
        },
        "action": "worker_general",
        "success_rate": 0.65,
        "success_q": 0.7,
        "failure_q": 0.4,
        "success_reason": "Manual run revealed edge cases, better automation later",
        "failure_reason": "Manual process was error-prone, should have automated",
    },

    # === ESCALATION ===
    "early_escalation": {
        "tasks": [
            "Escalate {problem} immediately to senior",
            "Ask for help with {problem} early",
            "Raise {problem} to leadership now",
            "Get expert input on {problem} upfront",
        ],
        "vars": {
            "problem": ["the blocker", "the conflict", "the risk", "the decision",
                        "the technical challenge", "the resource constraint",
                        "the timeline concern", "the scope creep", "the dependency"],
        },
        "action": "architect_general",
        "success_rate": 0.55,
        "success_q": 0.8,
        "failure_q": 0.35,
        "success_reason": "Early escalation prevented bigger problem, good judgment",
        "failure_reason": "Escalated too early, should have tried to solve first",
    },
    "delayed_escalation": {
        "tasks": [
            "Try to solve {problem} before escalating",
            "Attempt own solution for {problem} first",
            "Work on {problem} independently first",
            "Exhaust options before escalating {problem}",
        ],
        "vars": {
            "problem": ["the blocker", "the conflict", "the risk", "the decision",
                        "the technical challenge", "the resource constraint",
                        "the timeline concern", "the scope creep", "the dependency"],
        },
        "action": "coder_escalation",
        "success_rate": 0.5,
        "success_q": 0.85,
        "failure_q": 0.2,
        "success_reason": "Solved independently, demonstrated capability",
        "failure_reason": "Delayed too long, problem grew worse, lost credibility",
    },

    # === SCOPE DECISIONS ===
    "expand_scope_proactively": {
        "tasks": [
            "Add related improvements while working on {task}",
            "Include adjacent fixes with {task}",
            "Expand {task} to cover related issues",
            "Do comprehensive update instead of just {task}",
        ],
        "vars": {
            "task": ["the bug fix", "the feature", "the refactor", "the update",
                     "the migration", "the optimization", "the cleanup"],
        },
        "action": "coder_escalation",
        "success_rate": 0.4,
        "success_q": 0.8,
        "failure_q": 0.2,
        "success_reason": "Comprehensive fix prevented future issues, efficient",
        "failure_reason": "Scope creep delayed delivery, introduced new bugs",
    },
    "minimal_scope": {
        "tasks": [
            "Do only exactly what was asked for {task}",
            "Keep {task} scope minimal",
            "Don't expand beyond {task} requirements",
            "Stick strictly to {task} specification",
        ],
        "vars": {
            "task": ["the bug fix", "the feature", "the refactor", "the update",
                     "the migration", "the optimization", "the cleanup"],
        },
        "action": "worker_general",
        "success_rate": 0.65,
        "success_q": 0.75,
        "failure_q": 0.4,
        "success_reason": "Delivered quickly, met expectations, no surprises",
        "failure_reason": "Missed obvious related issues, had to revisit later",
    },
}


def generate_task(pattern):
    """Generate a task from template."""
    template = random.choice(pattern["tasks"])
    task = template
    for var, options in pattern["vars"].items():
        placeholder = "{" + var + "}"
        if placeholder in task:
            task = task.replace(placeholder, random.choice(options), 1)
    return task


def seed_probabilistic_memories(store: EpisodicStore, embedder: TaskEmbedder,
                                 count_per_pattern: int = 30) -> int:
    """Seed memories with probabilistic outcomes."""
    seeded = 0

    for pattern_name, pattern in PROBABILISTIC_PATTERNS.items():
        print(f"Seeding: {pattern_name} (success rate: {pattern['success_rate']:.0%})")

        for _ in range(count_per_pattern):
            task = generate_task(pattern)
            task_ir = {
                "task_type": "general",
                "objective": task,
                "priority": "interactive",
            }

            # Probabilistic outcome based on success rate
            if random.random() < pattern["success_rate"]:
                outcome = "success"
                q_value = pattern["success_q"]
                reason = pattern["success_reason"]
            else:
                outcome = "failure"
                q_value = pattern["failure_q"]
                reason = pattern["failure_reason"]

            # Add some variance to Q-values
            q_value = max(0.0, min(1.0, q_value + random.gauss(0, 0.05)))

            context = {
                **task_ir,
                "outcome_reason": reason,
                "success_rate": pattern["success_rate"],
            }

            try:
                embedding = embedder.embed_task_ir(task_ir)
                store.store(
                    embedding=embedding,
                    action=pattern["action"],
                    action_type="routing",
                    context=context,
                    outcome=outcome,
                    initial_q=q_value,
                )
                seeded += 1
            except Exception as e:
                print(f"  Error: {e}")

    return seeded


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print("Initializing...")
    embedder = TaskEmbedder()
    store = EpisodicStore()

    print(f"\nSeeding probabilistic memories ({len(PROBABILISTIC_PATTERNS)} patterns, {count} each)...")
    seeded = seed_probabilistic_memories(store, embedder, count)

    print(f"\n=== Results ===")
    print(f"Seeded: {seeded} probabilistic memories")

    stats = store.get_stats()
    print(f"Total memories: {stats['total_memories']}")


if __name__ == "__main__":
    main()
