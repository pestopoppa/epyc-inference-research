#!/usr/bin/env python3
"""Utilities for summarizing memory viability results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_decision_markdown(
    *,
    stage: str,
    run_dir: Path,
    decision: str,
    rationale: str,
    round_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Memory Viability Decision")
    lines.append("")
    lines.append(f"- Run: `{run_dir}`")
    lines.append(f"- Stage: `{stage}`")
    lines.append(f"- Decision: **{decision}**")
    lines.append(f"- Rationale: {rationale}")
    lines.append("")
    lines.append("## Round Summary")
    lines.append("")
    lines.append("| Round | Variant | Provenance | Control | Questions | Accuracy | Baseline | Uplift (pp) |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(round_rows, key=lambda x: (x["round_index"], x["is_control"], -float(x["uplift_pp"]))):
        lines.append(
            "| {round_index} | `{variant_name}` | {provenance} | {is_control} | {questions} | {accuracy} | {baseline_accuracy} | {uplift_pp} |".format(
                round_index=r["round_index"],
                variant_name=r["variant_name"],
                provenance=r["provenance"],
                is_control="yes" if r["is_control"] else "no",
                questions=r["questions"],
                accuracy=r["accuracy"],
                baseline_accuracy=r["baseline_accuracy"],
                uplift_pp=r["uplift_pp"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def summarize_results(results_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((int(r["round_index"]), str(r["variant_name"])), []).append(r)

    summary: list[dict[str, Any]] = []
    for (round_index, variant_name), group in sorted(grouped.items()):
        n = len(group)
        acc = sum(1 for g in group if g.get("correct")) / n if n else 0.0
        summary.append(
            {
                "round_index": round_index,
                "variant_name": variant_name,
                "questions": n,
                "accuracy": acc,
            }
        )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize memory viability JSONL")
    p.add_argument("--results-jsonl", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    args = p.parse_args()

    summary = summarize_results(args.results_jsonl)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round_index", "variant_name", "questions", "accuracy"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
