#!/usr/bin/env python3
"""Analyze web_research baseline metrics from seeding checkpoint JSONL.

Parses checkpoint files and reports:
  - % of questions triggering web_research
  - Pass rate with vs without web_research
  - Config breakdown of web_research usage
  - Telemetry aggregates (pages fetched, domains, timing)

Usage:
    python analyze_web_research_baseline.py [checkpoint_dir]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def analyze_checkpoints(checkpoint_dir: Path) -> None:
    files = sorted(checkpoint_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {checkpoint_dir}")
        return

    total_questions = 0
    wr_triggered = 0
    pass_with_wr = 0
    fail_with_wr = 0
    pass_without_wr = 0
    fail_without_wr = 0
    config_usage: Counter[str] = Counter()
    total_calls = 0
    total_pages = 0
    total_domains = 0

    for f in files:
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            metadata = entry.get("metadata", {})
            wr_baseline = metadata.get("web_research_baseline", {})
            if not wr_baseline:
                continue

            total_questions += 1
            triggered = wr_baseline.get("triggered", False)
            call_count = wr_baseline.get("call_count", 0)
            configs = wr_baseline.get("configs_using", [])

            # Determine overall pass/fail from rewards
            rewards = entry.get("rewards", {})
            any_passed = any(v > 0 for v in rewards.values()) if rewards else False

            if triggered:
                wr_triggered += 1
                total_calls += call_count
                for c in configs:
                    config_usage[c] += 1
                if any_passed:
                    pass_with_wr += 1
                else:
                    fail_with_wr += 1
            else:
                if any_passed:
                    pass_without_wr += 1
                else:
                    fail_without_wr += 1

            # Telemetry aggregates
            wr_telemetry = metadata.get("web_research_telemetry", {})
            for config_data in wr_telemetry.values():
                total_pages += config_data.get("total_pages_fetched", 0)
                total_domains += config_data.get("unique_domains", 0)

    if total_questions == 0:
        print("No questions with web_research_baseline metadata found.")
        return

    pct_triggered = 100.0 * wr_triggered / total_questions
    with_wr_total = pass_with_wr + fail_with_wr
    without_wr_total = pass_without_wr + fail_without_wr
    pass_rate_with = 100.0 * pass_with_wr / with_wr_total if with_wr_total else 0.0
    pass_rate_without = 100.0 * pass_without_wr / without_wr_total if without_wr_total else 0.0

    print(f"=== Web Research Baseline Analysis ===")
    print(f"Total questions analyzed: {total_questions}")
    print(f"Questions triggering web_research: {wr_triggered} ({pct_triggered:.1f}%)")
    print(f"Total web_research calls: {total_calls}")
    print()
    print(f"Pass rate WITH web_research:    {pass_with_wr}/{with_wr_total} ({pass_rate_with:.1f}%)")
    print(f"Pass rate WITHOUT web_research: {pass_without_wr}/{without_wr_total} ({pass_rate_without:.1f}%)")
    print()
    if total_calls > 0:
        print(f"Avg pages fetched per call: {total_pages / total_calls:.1f}")
        print(f"Avg unique domains per call: {total_domains / total_calls:.1f}")
    print()
    if config_usage:
        print("Config breakdown:")
        for config, count in config_usage.most_common():
            print(f"  {config}: {count} questions")


if __name__ == "__main__":
    checkpoint_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval"
    )
    analyze_checkpoints(checkpoint_dir)
