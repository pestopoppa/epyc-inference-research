#!/usr/bin/env python3
"""Summarizer quality evaluation — compare Tier 2 consolidation across model tiers.

Phase 2a of context-folding-progressive. Runs Tier 2 consolidation across
model tiers (1.5B, 7B, 32B) on real session logs, scores with Claude-as-Judge.

Usage:
    python eval_summarizer.py --dry-run --n-traces 5
    python eval_summarizer.py --traces-dir /mnt/raid0/llm/tmp/ --model-ports 8072,8071,8070
    python eval_summarizer.py --output results/summarizer_quality.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MODEL_TIERS = ["1.5B", "7B", "32B"]
DEFAULT_PORTS = [8072, 8071, 8070]  # worker_fast, worker_explore, coder_esc


def find_session_traces(traces_dir: Path, n_traces: int) -> list[Path]:
    """Find session log files, returning up to n_traces."""
    traces = sorted(traces_dir.glob("session_*.md"))
    # Exclude test/padding traces that don't contain real session data
    traces = [t for t in traces if not t.name.startswith("session_test_")]
    # Prefer longer traces (more content to summarize)
    traces.sort(key=lambda p: p.stat().st_size, reverse=True)
    return traces[:n_traces]


def evaluate_summarizer(
    trace: Path,
    model_port: int,
    model_tier: str,
    judge_port: int,
) -> dict:
    """Run Tier 2 consolidation and score quality for one trace + one model."""
    from eval_helpers import parse_session_trace, call_model, judge_quality, estimate_tokens

    try:
        turns = parse_session_trace(trace)
        if not turns:
            return _error_result(trace.name, model_port, model_tier, "no turns parsed")

        # Extract Tier 1 blocks (raw turn lines)
        tier1_text = "\n".join(t.raw_text for t in turns)
        tokens_before = estimate_tokens(tier1_text)

        # Call model to produce Tier 2 consolidation
        consolidated, usage = call_model(
            port=model_port,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Consolidate these session turn logs into a dense summary paragraph. "
                        "Preserve key decisions, code changes, errors, and outcomes. "
                        "Omit redundant nudges and repeated failures."
                    ),
                },
                {"role": "user", "content": tier1_text},
            ],
            timeout=180.0,
        )
        tokens_after = estimate_tokens(consolidated)

        # Extract probe: last non-nudge turn
        probe = ""
        for turn in reversed(turns):
            if turn.outcome != "nudge":
                probe = turn.output or turn.error or turn.first_line or ""
                break

        scores = judge_quality(
            original=tier1_text,
            summary=consolidated,
            probe=probe,
            port=judge_port,
        )

        compression_ratio = tokens_before / max(tokens_after, 1)
        return {
            "trace": trace.name,
            "model_port": model_port,
            "model_tier": model_tier,
            "faithfulness": scores["faithfulness"],
            "compression_ratio": round(compression_ratio, 2),
            "retention": scores["retention"],
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "status": "live",
        }
    except Exception as e:
        return _error_result(trace.name, model_port, model_tier, str(e))


def _error_result(trace_name: str, port: int, tier: str, error: str) -> dict:
    return {
        "trace": trace_name,
        "model_port": port,
        "model_tier": tier,
        "faithfulness": -1,
        "compression_ratio": 0.0,
        "retention": -1,
        "tokens_before": 0,
        "tokens_after": 0,
        "status": f"error: {error}",
    }


def _mock_result(trace_name: str, port: int, tier: str, tier_idx: int) -> dict:
    """Generate mock result for dry-run. Higher tiers produce better quality."""
    base_faith = 1.5 + tier_idx * 0.5
    base_retain = 1.2 + tier_idx * 0.6
    return {
        "trace": trace_name,
        "model_port": port,
        "model_tier": tier,
        "faithfulness": round(min(3.0, base_faith), 2),
        "compression_ratio": round(3.0 + tier_idx * 0.5, 2),
        "retention": round(min(3.0, base_retain), 2),
        "tokens_before": 2000,
        "tokens_after": int(2000 / (3.0 + tier_idx * 0.5)),
        "status": "dry_run",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summarizer quality across model tiers (CF Phase 2a)",
    )
    parser.add_argument(
        "--traces-dir", type=Path, default=Path("/mnt/raid0/llm/tmp"),
        help="Directory containing session trace files",
    )
    parser.add_argument(
        "--model-ports", type=str, default=",".join(str(p) for p in DEFAULT_PORTS),
        help="Comma-separated ports for model tiers (1.5B,7B,32B)",
    )
    parser.add_argument(
        "--judge-port", type=int, default=8082,
        help="Judge model port for quality scoring (default: 8082)",
    )
    parser.add_argument(
        "--n-traces", type=int, default=20,
        help="Number of traces to evaluate (default: 20)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: stdout as JSON)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate mock results without model servers",
    )
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.model_ports.split(",")]
    if len(ports) != len(MODEL_TIERS):
        print(
            f"Error: expected {len(MODEL_TIERS)} ports, got {len(ports)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        traces = [Path(f"synthetic_session_{i}.md") for i in range(args.n_traces)]
    else:
        traces = find_session_traces(args.traces_dir, args.n_traces)
        if not traces:
            print(f"No session traces found in {args.traces_dir}", file=sys.stderr)
            sys.exit(1)

    results = []
    for trace in traces:
        for tier_idx, (port, tier) in enumerate(zip(ports, MODEL_TIERS)):
            if args.dry_run:
                result = _mock_result(trace.name, port, tier, tier_idx)
            else:
                result = evaluate_summarizer(trace, port, tier, args.judge_port)
            results.append(result)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} results to {args.output}")
    else:
        for r in results:
            print(json.dumps(r))

    # Summary: per-tier averages
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}"
          f"{len(results)} evaluations across {len(MODEL_TIERS)} tiers")
    for tier in MODEL_TIERS:
        tier_results = [r for r in results if r["model_tier"] == tier]
        valid = [r for r in tier_results if r["faithfulness"] >= 0]
        if valid:
            avg_faith = sum(r["faithfulness"] for r in valid) / len(valid)
            avg_retain = sum(r["retention"] for r in valid) / len(valid)
            avg_ratio = sum(r["compression_ratio"] for r in valid) / len(valid)
            print(
                f"  {tier}: faithfulness={avg_faith:.2f}/3.0  "
                f"retention={avg_retain:.2f}/3.0  "
                f"compression={avg_ratio:.1f}x"
            )
        else:
            print(f"  {tier}: no valid results")


if __name__ == "__main__":
    main()
