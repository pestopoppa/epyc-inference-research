#!/usr/bin/env python3
"""Fail-closed comparison of counterbalanced v8/v9 production-role runs.

The inputs are terminal summaries emitted by
``k35_stack_context_matrix_runner.py``. Multiple summaries per arm are pooled
so an A-B-B-A schedule can retain independent fresh-server blocks while the
comparison stays deterministic and reviewable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_RATIO = 0.98
FAIL_RATIO = 0.95


class ComparisonError(RuntimeError):
    """The supplied evidence cannot support a comparison."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read JSON {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def response_content(response: dict[str, Any]) -> str:
    if isinstance(response.get("content"), str):
        return response["content"]
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = first.get("text") if isinstance(first, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def median_mad(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ComparisonError("throughput samples must be finite and positive")
    median = statistics.median(values)
    return {
        "n": len(values),
        "samples": values,
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
    }


def load_arm(paths: list[Path], expected_binary: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected = str(expected_binary.resolve())
    for block, summary_path in enumerate(paths, start=1):
        summary = load_json(summary_path)
        if summary.get("status") != "ok":
            raise ComparisonError(f"{label} summary is not terminal-ok: {summary_path}")
        plan_path = summary_path.with_name("plan.json")
        plan = load_json(plan_path)
        actual_binary = str(Path(plan.get("binary", "")).resolve())
        if actual_binary != expected:
            raise ComparisonError(
                f"{label} binary mismatch in {plan_path}: {actual_binary} != {expected}"
            )
        results = summary.get("results")
        if not isinstance(results, list) or not results:
            raise ComparisonError(f"{label} has no results: {summary_path}")
        for result in results:
            if result.get("status") != "ok" or result.get("passed_min_completion") is not True:
                raise ComparisonError(f"{label} contains a failed result in {summary_path}")
            cleanup = result.get("cleanup") or {}
            if cleanup.get("completed") is not True or cleanup.get("dead") is not True:
                raise ComparisonError(f"{label} cleanup is incomplete in {summary_path}")
            response_path = resolve_artifact_path(str(result.get("response_path") or ""))
            request_path = resolve_artifact_path(str(result.get("request_path") or ""))
            prompt_hash_path = resolve_artifact_path(str(result.get("prompt_sha256_path") or ""))
            if not response_path.is_file() or not request_path.is_file() or not prompt_hash_path.is_file():
                raise ComparisonError(f"{label} result references a missing request artifact")
            response = load_json(response_path)
            record = dict(result)
            record.update(
                {
                    "arm": label,
                    "block": block,
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256(summary_path),
                    "response_sha256": sha256(response_path),
                    "request_sha256": sha256(request_path),
                    "prompt_sha256": prompt_hash_path.read_text(encoding="utf-8").strip(),
                    "content": response_content(response),
                }
            )
            records.append(record)
    return records


def rep_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record["scenario"]),
        int(record["nominal_context"]),
        int(record["block"]),
        int(record["rep"]),
    )


def shape_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["scenario"]), int(record["nominal_context"])


def compare(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    minimum_reps: int,
    gate_throughput: bool,
) -> dict[str, Any]:
    baseline_by_key = {rep_key(record): record for record in baseline}
    candidate_by_key = {rep_key(record): record for record in candidate}
    if len(baseline_by_key) != len(baseline) or len(candidate_by_key) != len(candidate):
        raise ComparisonError("duplicate scenario/context/block/rep key")
    if set(baseline_by_key) != set(candidate_by_key):
        missing_candidate = sorted(set(baseline_by_key) - set(candidate_by_key))
        missing_baseline = sorted(set(candidate_by_key) - set(baseline_by_key))
        raise ComparisonError(
            f"arm cardinality mismatch: missing_candidate={missing_candidate}, "
            f"missing_baseline={missing_baseline}"
        )

    parity_rows: list[dict[str, Any]] = []
    parity_pass = True
    for key in sorted(baseline_by_key):
        left, right = baseline_by_key[key], candidate_by_key[key]
        checks = {
            "prompt_sha256_equal": left["prompt_sha256"] == right["prompt_sha256"],
            "request_sha256_equal": left["request_sha256"] == right["request_sha256"],
            "content_exact": left["content"] == right["content"],
            "completion_tokens_equal": left.get("completion_tokens") == right.get("completion_tokens"),
            "token_ids_equal_when_present": (
                left.get("token_ids") == right.get("token_ids")
                if left.get("token_ids") is not None or right.get("token_ids") is not None
                else True
            ),
        }
        passed = all(checks.values())
        parity_pass = parity_pass and passed
        parity_rows.append(
            {
                "key": list(key),
                "status": "pass" if passed else "fail",
                "checks": checks,
                "baseline_response_sha256": left["response_sha256"],
                "candidate_response_sha256": right["response_sha256"],
                "content_sha256": hashlib.sha256(left["content"].encode()).hexdigest(),
            }
        )

    grouped_baseline: dict[tuple[str, int], list[float]] = defaultdict(list)
    grouped_candidate: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in baseline:
        grouped_baseline[shape_key(record)].append(float(record["decode_tps"]))
    for record in candidate:
        grouped_candidate[shape_key(record)].append(float(record["decode_tps"]))

    throughput_rows: list[dict[str, Any]] = []
    throughput_states: list[str] = []
    for key in sorted(grouped_baseline):
        left = median_mad(grouped_baseline[key])
        right = median_mad(grouped_candidate[key])
        if left["n"] < minimum_reps or right["n"] < minimum_reps:
            raise ComparisonError(f"{key} has fewer than {minimum_reps} samples per arm")
        ratio = right["median"] / left["median"]
        state = "pass" if ratio >= PASS_RATIO else ("gray" if ratio >= FAIL_RATIO else "fail")
        throughput_states.append(state)
        throughput_rows.append(
            {
                "scenario": key[0],
                "nominal_context": key[1],
                "baseline": left,
                "candidate": right,
                "candidate_over_baseline": ratio,
                "state": state,
            }
        )

    throughput_status = (
        "pass"
        if all(state == "pass" for state in throughput_states)
        else ("fail" if "fail" in throughput_states else "gray")
    )
    gate_status = (
        "pass"
        if parity_pass and (not gate_throughput or throughput_status == "pass")
        else "fail"
    )
    return {
        "schema": "epyc.kernel_v9.production_role_comparison.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": gate_status,
        "metric_direction": "higher_is_better",
        "thresholds": {"pass_ratio": PASS_RATIO, "gray_floor": FAIL_RATIO},
        "minimum_reps_per_arm": minimum_reps,
        "throughput_gates_decision": gate_throughput,
        "quality_transfer": {
            "status": "pass" if parity_pass else "fail",
            "basis": "exact deterministic response parity",
            "rows": parity_rows,
        },
        "throughput": {"status": throughput_status, "rows": throughput_rows},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, action="append", required=True)
    parser.add_argument("--candidate-summary", type=Path, action="append", required=True)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--minimum-reps", type=int, default=10)
    parser.add_argument("--observation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_reps <= 0:
        raise ComparisonError("--minimum-reps must be positive")
    baseline = load_arm(args.baseline_summary, args.baseline_binary, "baseline")
    candidate = load_arm(args.candidate_summary, args.candidate_binary, "candidate")
    result = compare(
        baseline,
        candidate,
        minimum_reps=args.minimum_reps,
        gate_throughput=not args.observation_only,
    )
    result["inputs"] = {
        "baseline_summaries": [str(path) for path in args.baseline_summary],
        "candidate_summaries": [str(path) for path in args.candidate_summary],
        "baseline_binary": str(args.baseline_binary.resolve()),
        "candidate_binary": str(args.candidate_binary.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
