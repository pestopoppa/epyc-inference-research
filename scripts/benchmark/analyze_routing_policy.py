#!/usr/bin/env python3
"""Per-Suite Routing Policy Analysis.

Reads the MemRL episodic store, groups memories by suite/task_type,
and shows the top-ranked action by Q-value per group. Outputs a
"learned routing table" showing emergent per-suite policies.

Usage:
    python scripts/benchmark/analyze_routing_policy.py
    python scripts/benchmark/analyze_routing_policy.py --min-samples 5
    python scripts/benchmark/analyze_routing_policy.py --format csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_routing_policy(
    min_samples: int = 3,
    output_format: str = "table",
) -> dict:
    """Analyze learned routing policies from episodic store.

    Args:
        min_samples: Minimum samples per action to include.
        output_format: "table" or "csv" or "json".

    Returns:
        Policy dict: suite -> {best_action, q_value, n, all_actions}.
    """
    from orchestration.repl_memory.episodic_store import EpisodicStore

    store = EpisodicStore()

    # Get overall action Q-value summary
    action_summary = store.get_action_q_summary()

    if not action_summary:
        print("No memories in episodic store. Run seeding or learning loop first.")
        return {}

    # Print global action summary
    print(f"\n{'='*70}")
    print("GLOBAL ACTION Q-VALUE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Action':35s} {'N':>6s} {'Mean Q':>8s} {'Std Q':>8s}")
    print("-" * 60)
    for action, (n, mean_q, std_q) in sorted(
        action_summary.items(), key=lambda x: -x[1][1]
    ):
        if n >= min_samples:
            print(f"{action:35s} {n:6d} {mean_q:8.3f} {std_q:8.3f}")

    # Now analyze per-suite policies by reading memories with context
    # The context dict typically has task_type (suite name)
    import sqlite3
    conn = sqlite3.connect(store.sqlite_path)
    rows = conn.execute(
        "SELECT action, context, q_value FROM memories"
    ).fetchall()
    conn.close()

    # Group by suite (task_type in context)
    suite_actions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for action, context_str, q_value in rows:
        try:
            context = json.loads(context_str) if context_str else {}
        except (json.JSONDecodeError, TypeError):
            context = {}

        suite = context.get("task_type", "unknown")
        suite_actions[suite][action].append(q_value)

    if not suite_actions:
        print("\nNo suite-tagged memories found.")
        return {}

    # Build policy table
    policy: dict[str, dict] = {}
    for suite in sorted(suite_actions.keys()):
        actions = suite_actions[suite]
        best_action = None
        best_q = -float("inf")
        all_actions = {}

        for action, q_values in sorted(actions.items()):
            n = len(q_values)
            if n < min_samples:
                continue
            mean_q = sum(q_values) / n
            all_actions[action] = {"n": n, "mean_q": mean_q}
            if mean_q > best_q:
                best_q = mean_q
                best_action = action

        if best_action:
            policy[suite] = {
                "best_action": best_action,
                "q_value": best_q,
                "n": all_actions[best_action]["n"],
                "all_actions": all_actions,
            }

    # Print per-suite learned routing table
    print(f"\n{'='*70}")
    print("LEARNED ROUTING TABLE (Per-Suite Best Actions)")
    print(f"{'='*70}")

    if output_format == "csv":
        print("suite,best_action,q_value,n")
        for suite, info in sorted(policy.items()):
            print(f"{suite},{info['best_action']},{info['q_value']:.3f},{info['n']}")
    else:
        print(f"{'Suite':20s} | {'Best Action':30s} | {'Q-value':>8s} | {'N':>5s}")
        print("-" * 70)
        for suite, info in sorted(policy.items()):
            print(
                f"{suite:20s} | {info['best_action']:30s} | "
                f"{info['q_value']:8.3f} | {info['n']:5d}"
            )

    # Print per-suite detail (all competing actions)
    print(f"\n{'='*70}")
    print("PER-SUITE ACTION DETAILS")
    print(f"{'='*70}")
    for suite in sorted(suite_actions.keys()):
        actions = suite_actions[suite]
        entries = []
        for action, q_values in actions.items():
            n = len(q_values)
            mean_q = sum(q_values) / n
            entries.append((action, n, mean_q))

        entries.sort(key=lambda x: -x[2])  # Sort by Q-value desc

        print(f"\n  {suite}:")
        for action, n, mean_q in entries:
            marker = " ★" if n >= min_samples and entries[0][0] == action else ""
            below_threshold = " (below threshold)" if n < min_samples else ""
            print(f"    {action:35s}  n={n:4d}  Q={mean_q:.3f}{marker}{below_threshold}")

    # Specialist utilization summary
    specialist_roles = {"coder_escalation", "coder_escalation", "architect_general", "architect_coding"}
    specialist_wins = 0
    total_suites = len(policy)

    for suite, info in policy.items():
        action_role = info["best_action"].split(":")[0]
        if action_role in specialist_roles:
            specialist_wins += 1

    if total_suites > 0:
        print(f"\n{'='*70}")
        print("SPECIALIST UTILIZATION")
        print(f"{'='*70}")
        print(f"Suites with specialist as best action: {specialist_wins}/{total_suites}")
        print(f"Specialist utilization: {specialist_wins/total_suites*100:.0f}%")

    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Per-Suite Routing Policy Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-samples", type=int, default=3,
        help="Minimum samples per action to include (default: 3)",
    )
    parser.add_argument(
        "--format", choices=["table", "csv", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save JSON output to file",
    )

    args = parser.parse_args()

    policy = analyze_routing_policy(
        min_samples=args.min_samples,
        output_format=args.format,
    )

    if args.format == "json":
        print(json.dumps(policy, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(policy, f, indent=2)
        print(f"\nPolicy saved to: {args.output}")


if __name__ == "__main__":
    main()
