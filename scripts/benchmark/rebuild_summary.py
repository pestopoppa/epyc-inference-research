#!/usr/bin/env python3
from __future__ import annotations

"""
Rebuild summary.csv from ALL result JSON files and review CSVs.

This script crawls through all benchmark results and rebuilds the master
benchmark table deterministically.

Philosophy:
- SEPARATE ROWS for configs that affect QUALITY:
  - Different base models
  - MoE expert reductions (moe2, moe4, moe6)
  - Different quantizations (Q4_K_M vs Q6_K_L)

- EXTRA COLUMNS for configs that affect SPEED ONLY (preserve quality):
  - Spec decode (draft model + K values)
  - Lookup/n-gram (n3, n4, n5)
  - Combinations thereof

Data sources:
- Result JSON files: benchmarks/results/runs/*/*.json
- Review CSV files: benchmarks/results/reviews/*.csv

Output:
- benchmarks/results/reviews/summary.csv
"""

import json
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

BASE_DIR = "/mnt/raid0/llm/epyc-inference-research/benchmarks/results"
RUNS_DIR = os.path.join(BASE_DIR, "runs")
REVIEWS_DIR = os.path.join(BASE_DIR, "reviews")
OUTPUT_FILE = os.path.join(REVIEWS_DIR, "summary.csv")

SUITES = ["thinking", "general", "math", "agentic", "coder", "instruction_precision", "long_context"]


def load_review_scores(review_name: str) -> dict | None:
    """Load claude_scores from review CSV if it exists."""
    review_file = os.path.join(REVIEWS_DIR, f"{review_name}.csv")

    if not os.path.exists(review_file):
        return None

    scores = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(review_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            suite = row.get("suite", "")
            if suite not in SUITES:
                continue

            try:
                claude_score = int(row.get("claude_score", 0))
                scores[suite]["total"] += 1
                if claude_score >= 2:
                    scores[suite]["correct"] += 1
            except (ValueError, TypeError):
                continue

    return dict(scores) if scores else None


def extract_model_name(model_path: str) -> str:
    """Extract clean model name from model_path."""
    if not model_path:
        return "unknown"

    filename = os.path.basename(model_path)

    if filename.endswith(".gguf"):
        filename = filename[:-5]

    # Remove multi-part suffix like -00001-of-00008
    filename = re.sub(r'-\d{5}-of-\d{5}$', '', filename)

    return filename


def extract_eval_speed(result_data: dict) -> float:
    """Extract average evaluation (generation) speed from result data."""
    summary = result_data.get("summary", {})
    if summary.get("avg_tokens_per_second"):
        return summary["avg_tokens_per_second"]

    speeds = []
    for suite_name, questions in result_data.get("results", {}).items():
        if not isinstance(questions, dict):
            continue
        for q_id, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue
            tps = q_data.get("tokens_per_second")
            if tps is not None and tps > 0:
                speeds.append(tps)

    return sum(speeds) / len(speeds) if speeds else 0.0


def get_run_timestamp(filepath: str) -> datetime:
    """Extract timestamp from run directory name."""
    run_dir = os.path.basename(os.path.dirname(filepath))
    try:
        return datetime.strptime(run_dir, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.min


def parse_config(config_name: str) -> dict:
    """
    Parse config name into components.

    Returns dict with:
    - type: baseline, moe, spec_draft, lookup, spec_lookup, unknown
    - moe: expert count (for moe configs)
    - spec_draft: draft model name (for spec configs)
    - k: K value (for spec configs)
    - lookup_n: n-gram size (for lookup configs)
    """
    if not config_name or config_name == "baseline":
        return {"type": "baseline", "moe": None, "spec_draft": None, "k": None, "lookup_n": None}

    # Check for MoE config
    moe_match = re.match(r'^moe(\d+)$', config_name)
    if moe_match:
        return {"type": "moe", "moe": int(moe_match.group(1)), "spec_draft": None, "k": None, "lookup_n": None}

    # Check for lookup config
    lookup_match = re.match(r'^lookup_n(\d+)$', config_name)
    if lookup_match:
        return {"type": "lookup", "moe": None, "spec_draft": None, "k": None, "lookup_n": int(lookup_match.group(1))}

    # Check for spec_draft config
    spec_match = re.match(r'^spec_draft_(.+)_k(\d+)$', config_name)
    if spec_match:
        return {"type": "spec_draft", "moe": None, "spec_draft": spec_match.group(1), "k": int(spec_match.group(2)), "lookup_n": None}

    # Check for spec_draft + lookup combination
    spec_lookup_match = re.match(r'^spec_draft_(.+)_k(\d+)_lookup_n(\d+)$', config_name)
    if spec_lookup_match:
        return {
            "type": "spec_lookup",
            "moe": None,
            "spec_draft": spec_lookup_match.group(1),
            "k": int(spec_lookup_match.group(2)),
            "lookup_n": int(spec_lookup_match.group(3))
        }

    # Check for lookup + spec_draft combination (alternate order)
    lookup_spec_match = re.match(r'^lookup_n(\d+)_spec_draft_(.+)_k(\d+)$', config_name)
    if lookup_spec_match:
        return {
            "type": "spec_lookup",
            "moe": None,
            "spec_draft": lookup_spec_match.group(2),
            "k": int(lookup_spec_match.group(3)),
            "lookup_n": int(lookup_spec_match.group(1))
        }

    # Check for MoE + spec_draft
    moe_spec_match = re.match(r'^moe(\d+)_spec_draft_(.+)_k(\d+)$', config_name)
    if moe_spec_match:
        return {
            "type": "moe_spec",
            "moe": int(moe_spec_match.group(1)),
            "spec_draft": moe_spec_match.group(2),
            "k": int(moe_spec_match.group(3)),
            "lookup_n": None
        }

    # Check for MoE + lookup
    moe_lookup_match = re.match(r'^moe(\d+)_lookup_n(\d+)$', config_name)
    if moe_lookup_match:
        return {
            "type": "moe_lookup",
            "moe": int(moe_lookup_match.group(1)),
            "spec_draft": None,
            "k": None,
            "lookup_n": int(moe_lookup_match.group(2))
        }

    return {"type": "unknown", "moe": None, "spec_draft": None, "k": None, "lookup_n": None, "raw": config_name}


def get_row_key(model_name: str, config: dict) -> tuple:
    """
    Determine the row key for grouping.

    Quality-affecting configs (MoE) get separate rows.
    Speed-only configs (spec_draft, lookup) attach to baseline.
    """
    config_type = config["type"]

    if config_type == "baseline":
        return (model_name, "baseline")
    elif config_type == "moe":
        return (model_name, f"moe{config['moe']}")
    elif config_type in ("spec_draft", "lookup", "spec_lookup"):
        # Speed optimizations attach to baseline
        return (model_name, "baseline")
    elif config_type == "moe_spec":
        # MoE + spec: attach spec to MoE row
        return (model_name, f"moe{config['moe']}")
    elif config_type == "moe_lookup":
        # MoE + lookup: attach lookup to MoE row
        return (model_name, f"moe{config['moe']}")
    else:
        return (model_name, "baseline")


def process_all_results() -> list[dict]:
    """Process all result files and return aggregated rows."""
    pattern = os.path.join(RUNS_DIR, "*", "*.json")
    result_files = sorted(glob.glob(pattern))

    print(f"Found {len(result_files)} result files")

    # First pass: collect all results
    all_results = []
    for filepath in result_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {filepath}: {e}")
            continue

        original_name = Path(filepath).stem
        model_path = data.get("model_path", "")
        model_name = extract_model_name(model_path)
        config_name = data.get("config_name", "baseline")
        config = parse_config(config_name)
        avg_tps = extract_eval_speed(data)
        timestamp = get_run_timestamp(filepath)

        # Load quality scores from Claude-as-Judge reviews ONLY
        # Algorithmic scores are deprecated and excluded from the master table
        review_scores = load_review_scores(original_name)

        suite_scores = {}
        if review_scores:
            for suite in SUITES:
                if suite in review_scores:
                    c = review_scores[suite]["correct"]
                    t = review_scores[suite]["total"]
                    suite_scores[suite] = {"correct": c, "total": t, "str": f"{c}/{t}"}
                else:
                    suite_scores[suite] = {"correct": 0, "total": 0, "str": "-"}
        else:
            # No Claude-as-Judge review - show as unscored
            for suite in SUITES:
                suite_scores[suite] = {"correct": 0, "total": 0, "str": "-"}

        total_correct = sum(s["correct"] for s in suite_scores.values())
        total_questions = sum(s["total"] for s in suite_scores.values())
        pct = (total_correct / total_questions * 100) if total_questions > 0 else 0

        row_key = get_row_key(model_name, config)

        all_results.append({
            "model_name": model_name,
            "config": config,
            "config_name": config_name,
            "row_key": row_key,
            "avg_tps": avg_tps,
            "timestamp": timestamp,
            "suite_scores": suite_scores,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "pct": pct,
            "filepath": filepath
        })

    # Second pass: group by row key
    model_groups = defaultdict(list)
    for r in all_results:
        model_groups[r["row_key"]].append(r)

    # Third pass: create final rows
    rows = []
    for (model_name, config_key), results_list in model_groups.items():
        # Separate by config type
        base_results = [r for r in results_list if r["config"]["type"] in ("baseline", "moe")]
        spec_results = [r for r in results_list if r["config"]["type"] in ("spec_draft", "moe_spec")]
        lookup_results = [r for r in results_list if r["config"]["type"] in ("lookup", "moe_lookup")]
        spec_lookup_results = [r for r in results_list if r["config"]["type"] == "spec_lookup"]

        # Pick best base result (latest timestamp, highest score)
        if base_results:
            base_results.sort(key=lambda r: (r["timestamp"], r["pct"]), reverse=True)
            best_base = base_results[0]
        else:
            # No base result - use first available as reference for quality scores
            all_sorted = sorted(results_list, key=lambda r: (r["timestamp"], r["pct"]), reverse=True)
            if all_sorted:
                best_base = all_sorted[0]
            else:
                continue

        # Aggregate spec decode results by draft model
        spec_by_draft = defaultdict(dict)
        for r in spec_results:
            draft = r["config"]["spec_draft"]
            k = r["config"]["k"]
            if draft and k:
                spec_by_draft[draft][k] = r["avg_tps"]

        # Build spec decode summary
        spec_summary = ""
        spec_best_tps = 0
        spec_draft_used = ""

        if spec_by_draft:
            # Find best draft model (highest max speed)
            best_draft = None
            best_max = 0
            for draft, k_speeds in spec_by_draft.items():
                max_speed = max(k_speeds.values()) if k_speeds else 0
                if max_speed > best_max:
                    best_max = max_speed
                    best_draft = draft

            if best_draft:
                spec_draft_used = best_draft
                k_speeds = spec_by_draft[best_draft]
                spec_best_tps = max(k_speeds.values()) if k_speeds else 0

                # Format: "k4:50.1,k8:52.3,k16:53.0,k24:52.8"
                k_parts = []
                for k in sorted(k_speeds.keys()):
                    k_parts.append(f"k{k}:{k_speeds[k]:.1f}")
                spec_summary = ",".join(k_parts)

        # Aggregate lookup results
        lookup_by_n = {}
        for r in lookup_results:
            n = r["config"]["lookup_n"]
            if n:
                lookup_by_n[n] = r["avg_tps"]

        # Build lookup summary
        lookup_summary = ""
        lookup_best_tps = 0

        if lookup_by_n:
            lookup_best_tps = max(lookup_by_n.values())
            # Format: "n3:3.8,n4:4.5,n5:4.5"
            n_parts = []
            for n in sorted(lookup_by_n.keys()):
                n_parts.append(f"n{n}:{lookup_by_n[n]:.1f}")
            lookup_summary = ",".join(n_parts)

        # Aggregate spec+lookup combination results
        spec_lookup_by_config = {}
        for r in spec_lookup_results:
            draft = r["config"]["spec_draft"]
            k = r["config"]["k"]
            n = r["config"]["lookup_n"]
            if draft and k and n:
                key = f"{draft}_k{k}_n{n}"
                spec_lookup_by_config[key] = r["avg_tps"]

        spec_lookup_summary = ""
        spec_lookup_best_tps = 0

        if spec_lookup_by_config:
            spec_lookup_best_tps = max(spec_lookup_by_config.values())
            # Format: "draft_k8_n3:55.2,draft_k16_n4:58.1"
            parts = [f"{k}:{v:.1f}" for k, v in sorted(spec_lookup_by_config.items())]
            spec_lookup_summary = ",".join(parts)

        # Determine row label
        if config_key == "baseline":
            row_label = model_name
        else:
            row_label = f"{model_name}_{config_key}"

        row = {
            "model": row_label,
            "thinking": best_base["suite_scores"]["thinking"]["str"],
            "general": best_base["suite_scores"]["general"]["str"],
            "math": best_base["suite_scores"]["math"]["str"],
            "agentic": best_base["suite_scores"]["agentic"]["str"],
            "coder": best_base["suite_scores"]["coder"]["str"],
            "instruction_precision": best_base["suite_scores"]["instruction_precision"]["str"],
            "long_context": best_base["suite_scores"]["long_context"]["str"],
            "total": f"{best_base['total_correct']}/{best_base['total_questions']}",
            "pct_str": f"{best_base['pct']:.0f}%",
            "avg_tps": round(best_base["avg_tps"], 1) if best_base["avg_tps"] else 0.0,
            "spec_draft": spec_draft_used,
            "spec_best_tps": round(spec_best_tps, 1) if spec_best_tps else "",
            "spec_k_results": spec_summary,
            "lookup_best_tps": round(lookup_best_tps, 1) if lookup_best_tps else "",
            "lookup_n_results": lookup_summary,
            "spec_lookup_best_tps": round(spec_lookup_best_tps, 1) if spec_lookup_best_tps else "",
            "spec_lookup_results": spec_lookup_summary,
        }
        rows.append(row)

    # Sort by model name
    rows.sort(key=lambda r: r["model"].lower())

    return rows


def main():
    print(f"Scanning results from: {RUNS_DIR}")
    print(f"Reviews directory: {REVIEWS_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    rows = process_all_results()

    print(f"Generated {len(rows)} unique model entries")

    # Write CSV
    fieldnames = [
        "model", "thinking", "general", "math", "agentic", "coder",
        "instruction_precision", "long_context", "total", "pct_str", "avg_tps",
        "spec_draft", "spec_best_tps", "spec_k_results",
        "lookup_best_tps", "lookup_n_results",
        "spec_lookup_best_tps", "spec_lookup_results"
    ]

    with open(OUTPUT_FILE, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {OUTPUT_FILE}")

    # Stats
    with_spec = sum(1 for r in rows if r["spec_draft"])
    with_lookup = sum(1 for r in rows if r["lookup_best_tps"])
    with_spec_lookup = sum(1 for r in rows if r["spec_lookup_best_tps"])
    moe_rows = sum(1 for r in rows if "_moe" in r["model"])

    print(f"\n=== Summary ===")
    print(f"Total rows: {len(rows)}")
    print(f"MoE variants: {moe_rows}")
    print(f"With spec decode: {with_spec}")
    print(f"With lookup: {with_lookup}")
    print(f"With spec+lookup: {with_spec_lookup}")

    # Speed sanity check
    print("\n=== Speed Sanity Check ===")
    suspicious = [r for r in rows if r["avg_tps"] > 200]
    if suspicious:
        print(f"WARNING: {len(suspicious)} entries with baseline speed > 200 t/s")
        for r in suspicious[:5]:
            print(f"  {r['model']}: {r['avg_tps']} t/s")
    else:
        print("All baseline speeds look reasonable (< 200 t/s)")

    speeds = [r["avg_tps"] for r in rows if r["avg_tps"] > 0]
    if speeds:
        print(f"\nBaseline speed range: {min(speeds):.1f} - {max(speeds):.1f} t/s")


if __name__ == "__main__":
    main()
