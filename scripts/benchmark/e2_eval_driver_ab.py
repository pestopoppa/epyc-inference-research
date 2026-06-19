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
import json
import shlex
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
