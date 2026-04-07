#!/usr/bin/env python3
"""Compaction sweep evaluation — measure quality vs compression ratio.

Sweeps compression levels (L1-L5: 20%-95% reduction) on real session logs,
measuring information retention via probe tasks scored by Claude-as-Judge.

Requires model servers for live eval. Use --dry-run for offline validation.

Usage:
    python eval_compaction_sweep.py --dry-run
    python eval_compaction_sweep.py --traces-dir /mnt/raid0/llm/tmp/ --levels 1,3,5
    python eval_compaction_sweep.py --output results/compaction_sweep.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


COMPRESSION_LEVELS = {
    1: 0.20,  # 20% reduction (gentle)
    2: 0.40,  # 40% reduction (moderate)
    3: 0.60,  # 60% reduction (aggressive)
    4: 0.80,  # 80% reduction (extreme)
    5: 0.95,  # 95% reduction (maximum)
}


def find_session_traces(traces_dir: Path) -> list[Path]:
    """Find session log files in the traces directory."""
    patterns = ["session_*.md", "session_*.jsonl"]
    traces = []
    for pat in patterns:
        traces.extend(sorted(traces_dir.glob(pat)))
    return traces


def sweep_compaction_profiles(
    traces: list[Path],
    levels: list[int],
    *,
    dry_run: bool = False,
    model_port: int = 8071,
    judge_port: int = 8082,
) -> list[dict]:
    """Sweep compression levels across session traces.

    Returns list of result dicts with quality metrics per level.
    """
    results = []

    for trace in traces:
        for level in levels:
            ratio = COMPRESSION_LEVELS[level]

            if dry_run:
                # Mock results for offline validation
                result = {
                    "trace": str(trace.name),
                    "level": level,
                    "ratio": ratio,
                    "faithfulness": max(0.0, 3.0 - level * 0.4),
                    "retention": max(0.0, 3.0 - level * 0.5),
                    "compression_achieved": ratio,
                    "tokens_before": 5000,
                    "tokens_after": int(5000 * (1.0 - ratio)),
                    "status": "dry_run",
                }
            else:
                # Live eval path — requires model servers
                result = evaluate_compaction(
                    trace, level, ratio,
                    model_port=model_port, judge_port=judge_port,
                )

            results.append(result)

    return results


def evaluate_compaction(
    trace: Path,
    level: int,
    ratio: float,
    *,
    model_port: int = 8071,
    judge_port: int = 8082,
) -> dict:
    """Run live compaction evaluation on one trace at one level.

    Requires model servers to be running. Steps:
    1. Load session trace
    2. Compact at target ratio using model server
    3. Extract probe task from session
    4. Score with Claude-as-Judge
    """
    from eval_helpers import parse_session_trace, call_model, judge_quality, estimate_tokens

    try:
        turns = parse_session_trace(trace)
        original_text = trace.read_text(errors="replace")
        original_tokens = estimate_tokens(original_text)
        target_tokens = int(original_tokens * (1.0 - ratio))

        consolidated, usage = call_model(
            port=model_port,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Consolidate this session log to approximately {target_tokens} tokens. "
                        "Preserve key decisions, errors, and outcomes. "
                        "Remove redundant nudges and repeated content."
                    ),
                },
                {"role": "user", "content": original_text},
            ],
            timeout=180.0,
        )

        # Extract probe: last non-nudge turn content
        probe = ""
        for turn in reversed(turns):
            if turn.outcome != "nudge":
                probe = turn.output or turn.error or turn.first_line or ""
                break

        scores = judge_quality(
            original=original_text,
            summary=consolidated,
            probe=probe,
            port=judge_port,
        )

        consolidated_tokens = estimate_tokens(consolidated)
        return {
            "trace": str(trace.name),
            "level": level,
            "ratio": ratio,
            "faithfulness": scores["faithfulness"],
            "retention": scores["retention"],
            "compression_achieved": round(
                1.0 - (consolidated_tokens / max(original_tokens, 1)), 3
            ),
            "tokens_before": original_tokens,
            "tokens_after": consolidated_tokens,
            "status": "live",
        }
    except Exception as e:
        return {
            "trace": str(trace.name),
            "level": level,
            "ratio": ratio,
            "faithfulness": -1,
            "retention": -1,
            "compression_achieved": 0.0,
            "tokens_before": 0,
            "tokens_after": 0,
            "status": f"error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep compaction levels and measure quality retention",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("/mnt/raid0/llm/tmp"),
        help="Directory containing session trace files",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated compression levels to sweep (1-5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout as JSON)",
    )
    parser.add_argument(
        "--model-port",
        type=int,
        default=8071,
        help="Model server port for consolidation (default: 8071, worker_explore)",
    )
    parser.add_argument(
        "--judge-port",
        type=int,
        default=8082,
        help="Judge model port for quality scoring (default: 8082)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate mock results without model servers",
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    for lv in levels:
        if lv not in COMPRESSION_LEVELS:
            print(f"Error: invalid level {lv}. Must be 1-5.", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        # Use synthetic traces for dry-run
        traces = [Path(f"synthetic_session_{i}.md") for i in range(5)]
    else:
        traces = find_session_traces(args.traces_dir)
        if not traces:
            print(f"No session traces found in {args.traces_dir}", file=sys.stderr)
            sys.exit(1)

    results = sweep_compaction_profiles(
        traces, levels,
        dry_run=args.dry_run,
        model_port=args.model_port,
        judge_port=args.judge_port,
    )

    if args.output:
        # CSV output
        import csv
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} results to {args.output}")
    else:
        # JSON to stdout
        for r in results:
            print(json.dumps(r))

    # Summary
    if args.dry_run:
        print(f"\n[DRY RUN] {len(results)} mock evaluations across {len(levels)} levels")
        for lv in levels:
            lv_results = [r for r in results if r["level"] == lv]
            avg_faith = sum(r["faithfulness"] for r in lv_results) / len(lv_results)
            avg_retain = sum(r["retention"] for r in lv_results) / len(lv_results)
            print(
                f"  L{lv} ({COMPRESSION_LEVELS[lv]:.0%} reduction): "
                f"faithfulness={avg_faith:.2f}/3.0  retention={avg_retain:.2f}/3.0"
            )


if __name__ == "__main__":
    main()
