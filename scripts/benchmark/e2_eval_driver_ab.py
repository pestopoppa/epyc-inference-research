#!/usr/bin/env python3
"""Build the E2 eval-driver A/B run plan for batched decode.

E2 compares the current EvalTower fan-out path against a single full
llama-server instance using continuous batching (`-np 8`). This script does
not hide host-health state: it records attestation, emits commands only when
decision-grade preconditions pass (or an explicit scout override is supplied),
and writes a manifest that can be committed with the resulting evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server_np_sweep import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QUESTION_POOL,
    collect_attestation,
    host_health_warnings,
    load_prompt_batch,
)


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_BATCH_MODEL_KEY = "qwen36_q8_0"
DEFAULT_BATCH_NP = 8
DEFAULT_CURRENT_CONCURRENCY = 3
DEFAULT_TRIAL_ID_BASE = 920000
DEFAULT_MIN_SPEEDUP = 1.05
DEFAULT_MAX_ERROR_RATE = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def shell_join(parts: list[str | Path | int]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))


def build_batch_command(args: argparse.Namespace, output_dir: Path) -> str:
    run_id = f"{args.run_id}-batch-np{args.batch_np}"
    command: list[str | Path | int] = [
        "uv",
        "run",
        "--extra",
        "benchmark",
        "python",
        "scripts/benchmark/server_np_sweep.py",
        "--run-id",
        run_id,
        "--output-root",
        output_dir / "serving",
        "--model-key",
        args.batch_model_key,
        "--np-levels",
        str(args.batch_np),
        "--prompt-limit",
        args.prompt_limit,
        "--prompt-seed",
        args.prompt_seed,
        "--tier",
        args.tier,
        "--n-predict",
        args.n_predict,
        "--port-base",
        args.port_base,
    ]
    if args.allow_host_health_warning:
        command.append("--allow-host-health-warning")
    if args.scout_skip_clean_check:
        command.append("--skip-clean-check")
    return f"cd {shlex.quote(str(args.research_root))}\n{shell_join(command)}"


def build_current_command(args: argparse.Namespace, output_dir: Path) -> str:
    calibration_id = args.calibration_id or f"{args.run_id}-current-quarters"
    out_jsonl = output_dir / "current_quarters.jsonl"
    env = {
        "AUTOPILOT_EVAL_CONCURRENCY": str(args.current_concurrency),
    }
    command: list[str | Path | int] = [
        "uv",
        "run",
        "python",
        "scripts/autopilot/core_v2_calibrate.py",
        "--calibration-id",
        calibration_id,
        "--out-jsonl",
        out_jsonl,
        "--n",
        args.prompt_limit,
        "--repeats",
        1,
        "--seed",
        args.prompt_seed,
        "--trial-id-base",
        args.trial_id_base,
        "--overwrite",
    ]
    return (
        f"cd {shlex.quote(str(args.orchestrator_root))}\n"
        f"{shell_env(env)} {shell_join(command)}"
    )


def command_block(command: str) -> str:
    return "(\n" + "\n".join(f"  {line}" for line in command.splitlines()) + "\n)"


def commented_block(command: str, reason: str) -> str:
    lines = [f"# blocked: {reason}"]
    for line in command_block(command).splitlines():
        lines.append(f"# {line}")
    return "\n".join(lines)


def write_commands(path: Path, arms: list[dict[str, Any]], runnable: bool, reason: str) -> None:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# E2 eval-driver A/B. Run arms sequentially in a clean/quiesced window.",
        "",
    ]
    for arm in arms:
        lines.append(f"# arm: {arm['name']}")
        if runnable:
            lines.append(command_block(str(arm["command"])))
        else:
            lines.append(commented_block(str(arm["command"]), reason))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], bool, str]:
    output_dir = args.output_root / args.run_id
    prompts = load_prompt_batch(
        args.question_pool,
        limit=args.prompt_limit,
        seed=args.prompt_seed,
        tier=args.tier,
        suites=set(),
    )
    attestation = collect_attestation()
    warnings = host_health_warnings(attestation)
    decision_grade = not warnings
    runnable = decision_grade or args.allow_host_health_warning
    blocked_reason = (
        "host-health preconditions failed; rerun after clean host-health or pass "
        "--allow-host-health-warning for explicitly non-decision-grade scout data"
    )

    arms = [
        {
            "name": f"batch_np{args.batch_np}_single_full_instance",
            "kind": "server_np_sweep",
            "metric": "wall_minutes_per_eval_proxy",
            "command": build_batch_command(args, output_dir),
            "primary_artifacts": [
                str(output_dir / "serving" / f"{args.run_id}-batch-np{args.batch_np}" / "summary.csv"),
                str(output_dir / "serving" / f"{args.run_id}-batch-np{args.batch_np}" / "recommendations.json"),
            ],
        },
        {
            "name": "current_three_concurrent_quarters",
            "kind": "core_v2_calibrate",
            "metric": "eval_wall_s",
            "command": build_current_command(args, output_dir),
            "primary_artifacts": [str(output_dir / "current_quarters.jsonl")],
        },
    ]

    manifest = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocol_id": "P-BENCH-3/E2",
        "status": "runnable" if runnable else "blocked",
        "decision_grade": decision_grade,
        "output_dir": str(output_dir),
        "comparison": {
            "purpose": "Price a single continuous-batching eval serving class against current EvalTower fan-out.",
            "metric": "wall_minutes_per_eval",
            "batch_arm": f"single full instance with -np {args.batch_np}",
            "current_arm": f"EvalTower fan-out with AUTOPILOT_EVAL_CONCURRENCY={args.current_concurrency}",
            "acceptance": "record keep-or-kill recommendation for an eval-batch instance set",
        },
        "prompt_batch": {
            "source": str(args.question_pool),
            "limit": args.prompt_limit,
            "seed": args.prompt_seed,
            "tier": args.tier,
            "qids": [prompt.qid for prompt in prompts],
            "selected_prompts_jsonl": str(output_dir / "selected_prompts.jsonl"),
        },
        "attestation": attestation,
        "host_health_warnings": warnings,
        "allow_host_health_warning": args.allow_host_health_warning,
        "scout_skip_clean_check": args.scout_skip_clean_check,
        "arms": arms,
        "notes": [
            "Run arms sequentially, not concurrently.",
            "The batch arm uses the E1 serving primitive as the single-instance continuous-batching arm.",
            "The current arm records EvalTower eval_wall_s for the live fan-out path.",
        ],
    }
    return manifest, arms, runnable, "" if runnable else blocked_reason


def write_outputs(args: argparse.Namespace) -> Path:
    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, arms, runnable, reason = build_manifest(args)

    selected_path = output_dir / "selected_prompts.jsonl"
    prompts = load_prompt_batch(
        args.question_pool,
        limit=args.prompt_limit,
        seed=args.prompt_seed,
        tier=args.tier,
        suites=set(),
    )
    with selected_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(asdict(prompt), sort_keys=True) + "\n")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_commands(output_dir / "commands.sh", arms, runnable, reason)
    return output_dir


def _float_value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value is not None and value != "" else default
    except (TypeError, ValueError):
        return default


def _int_value(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        return int(value) if value is not None and value != "" else default
    except (TypeError, ValueError):
        return default


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _batch_summary_path(manifest: dict[str, Any], run_dir: Path) -> Path:
    arms = manifest.get("arms")
    if isinstance(arms, list):
        for arm in arms:
            if not isinstance(arm, dict) or arm.get("kind") != "server_np_sweep":
                continue
            artifacts = arm.get("primary_artifacts")
            if isinstance(artifacts, list) and artifacts:
                return Path(str(artifacts[0]))
    run_id = str(manifest.get("run_id") or run_dir.name)
    batch_np = str((manifest.get("comparison") or {}).get("batch_np") or DEFAULT_BATCH_NP)
    return run_dir / "serving" / f"{run_id}-batch-np{batch_np}" / "summary.csv"


def _current_jsonl_path(manifest: dict[str, Any], run_dir: Path) -> Path:
    arms = manifest.get("arms")
    if isinstance(arms, list):
        for arm in arms:
            if not isinstance(arm, dict) or arm.get("kind") != "core_v2_calibrate":
                continue
            artifacts = arm.get("primary_artifacts")
            if isinstance(artifacts, list) and artifacts:
                return Path(str(artifacts[0]))
    return run_dir / "current_quarters.jsonl"


def _load_batch_row(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing batch summary: {path}"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, f"empty batch summary: {path}"
    successful = [row for row in rows if _int_value(row, "success_count") > 0]
    candidates = successful or rows
    return max(candidates, key=lambda row: _float_value(row, "tasks_per_hour")), None


def _load_current_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing current-arm JSONL: {path}"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                return [], f"invalid JSONL at {path}:{line_no}: {exc}"
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        return [], f"empty current-arm JSONL: {path}"
    return rows, None


def summarize_run(
    run_dir: Path,
    *,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> dict[str, Any]:
    """Summarize completed E2 arm artifacts into a keep/kill/hold recommendation."""
    manifest = _read_manifest(run_dir)
    batch_path = _batch_summary_path(manifest, run_dir)
    current_path = _current_jsonl_path(manifest, run_dir)
    missing: list[str] = []

    batch_row, error = _load_batch_row(batch_path)
    if error:
        missing.append(error)
    current_rows, error = _load_current_rows(current_path)
    if error:
        missing.append(error)

    decision_grade = bool(manifest.get("decision_grade", False))
    status = "incomplete" if missing else "hold"
    reasons: list[str] = list(missing)
    batch_metrics: dict[str, Any] = {"path": str(batch_path)}
    current_metrics: dict[str, Any] = {"path": str(current_path)}
    comparison: dict[str, Any] = {}

    if batch_row and current_rows:
        batch_wall_s = _float_value(batch_row, "wall_seconds")
        batch_error_rate = _float_value(batch_row, "error_rate", 1.0)
        current_wall_values = [_float_value(row, "eval_wall_s") for row in current_rows]
        current_wall_values = [value for value in current_wall_values if value > 0]
        current_question_values = [_int_value(row, "n_questions") for row in current_rows]
        current_question_values = [value for value in current_question_values if value > 0]
        current_wall_s = statistics.mean(current_wall_values) if current_wall_values else 0.0

        batch_metrics.update(
            {
                "model": batch_row.get("model"),
                "np": _int_value(batch_row, "np"),
                "success_count": _int_value(batch_row, "success_count"),
                "total_count": _int_value(batch_row, "total_count"),
                "error_rate": batch_error_rate,
                "wall_seconds": batch_wall_s,
                "wall_minutes_per_eval": batch_wall_s / 60.0 if batch_wall_s > 0 else None,
                "tasks_per_hour": _float_value(batch_row, "tasks_per_hour"),
                "p95_latency_ms": _float_value(batch_row, "p95_latency_ms"),
            }
        )
        current_metrics.update(
            {
                "rows": len(current_rows),
                "eval_concurrency": current_rows[-1].get("eval_concurrency"),
                "mean_eval_wall_s": current_wall_s,
                "wall_minutes_per_eval": current_wall_s / 60.0 if current_wall_s > 0 else None,
                "mean_n_questions": statistics.mean(current_question_values) if current_question_values else None,
            }
        )
        speedup = current_wall_s / batch_wall_s if batch_wall_s > 0 and current_wall_s > 0 else 0.0
        comparison.update(
            {
                "speedup_current_over_batch": speedup,
                "batch_wall_delta_s": batch_wall_s - current_wall_s,
                "min_speedup": min_speedup,
                "max_error_rate": max_error_rate,
            }
        )

        if not decision_grade:
            status = "scout_only"
            reasons.append("manifest is not decision-grade; do not use this summary for production keep/kill")
        elif batch_error_rate > max_error_rate:
            status = "kill_candidate"
            reasons.append(f"batch error_rate {batch_error_rate:.3f} exceeds {max_error_rate:.3f}")
        elif speedup >= min_speedup:
            status = "keep_candidate"
            reasons.append(f"batch arm is {speedup:.3f}x faster than current arm")
        else:
            status = "kill_candidate"
            reasons.append(f"batch arm speedup {speedup:.3f} is below required {min_speedup:.3f}")

    return {
        "run_dir": str(run_dir),
        "created_at": utc_now(),
        "protocol_id": "P-BENCH-3/E2",
        "status": status,
        "decision_grade": decision_grade and not missing,
        "recommendation": {
            "status": status,
            "reasons": reasons,
        },
        "batch_arm": batch_metrics,
        "current_arm": current_metrics,
        "comparison": comparison,
        "manifest": {
            "path": str(run_dir / "manifest.json"),
            "run_id": manifest.get("run_id"),
            "source_status": manifest.get("status"),
            "source_decision_grade": manifest.get("decision_grade"),
        },
    }


def write_summary(run_dir: Path, output_path: Path | None, *, min_speedup: float, max_error_rate: float) -> Path:
    summary = summarize_run(
        run_dir,
        min_speedup=min_speedup,
        max_error_rate=max_error_rate,
    )
    path = output_path or run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarize-run",
        type=Path,
        help="Summarize a completed E2 run directory instead of writing a new run plan.",
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument("--max-error-rate", type=float, default=DEFAULT_MAX_ERROR_RATE)
    parser.add_argument("--run-id", default=f"e2-eval-driver-ab-{utc_compact()}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--research-root", type=Path, default=RESEARCH_ROOT)
    parser.add_argument("--orchestrator-root", type=Path, default=DEFAULT_ORCHESTRATOR_ROOT)
    parser.add_argument("--question-pool", type=Path, default=DEFAULT_QUESTION_POOL)
    parser.add_argument("--prompt-limit", type=int, default=43)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--batch-model-key", default=DEFAULT_BATCH_MODEL_KEY)
    parser.add_argument("--batch-np", type=int, default=DEFAULT_BATCH_NP)
    parser.add_argument("--current-concurrency", type=int, default=DEFAULT_CURRENT_CONCURRENCY)
    parser.add_argument("--trial-id-base", type=int, default=DEFAULT_TRIAL_ID_BASE)
    parser.add_argument("--calibration-id")
    parser.add_argument("--port-base", type=int, default=18070)
    parser.add_argument(
        "--allow-host-health-warning",
        action="store_true",
        help="Emit runnable commands for explicitly non-decision-grade scout data.",
    )
    parser.add_argument(
        "--scout-skip-clean-check",
        action="store_true",
        help="Append --skip-clean-check to the batch arm. Use only for non-gating scout data.",
    )
    args = parser.parse_args(argv)
    if args.prompt_limit <= 0:
        parser.error("--prompt-limit must be positive")
    if args.n_predict <= 0:
        parser.error("--n-predict must be positive")
    if args.batch_np <= 0:
        parser.error("--batch-np must be positive")
    if args.current_concurrency <= 0:
        parser.error("--current-concurrency must be positive")
    if args.scout_skip_clean_check and not args.allow_host_health_warning:
        parser.error("--scout-skip-clean-check requires --allow-host-health-warning")
    if args.min_speedup <= 0:
        parser.error("--min-speedup must be positive")
    if not (0 <= args.max_error_rate <= 1):
        parser.error("--max-error-rate must be between 0 and 1")
    if args.summary_output and not args.summarize_run:
        parser.error("--summary-output requires --summarize-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_run:
        summary_path = write_summary(
            args.summarize_run,
            args.summary_output,
            min_speedup=args.min_speedup,
            max_error_rate=args.max_error_rate,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"wrote {summary_path}")
        print(f"status={summary['status']} decision_grade={summary['decision_grade']}")
        for reason in summary["recommendation"]["reasons"]:
            print(f"reason: {reason}")
        return 0

    output_dir = write_outputs(args)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"wrote {output_dir}")
    print(f"status={manifest['status']} decision_grade={manifest['decision_grade']}")
    if manifest["host_health_warnings"]:
        for warning in manifest["host_health_warnings"]:
            print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
