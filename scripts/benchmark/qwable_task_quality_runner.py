#!/usr/bin/env python3
"""Qwable IQ4/Q8 task-quality runner.

This is deliberately separate from qwable_reasoning_economics_runner.py:
the economics runner proves bounded loading and structured-output behavior,
while this runner scores a small deterministic task slice before any Qwable
role claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import qwable_reasoning_economics_runner as base


RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "qwable_reasoning_economics"
    / f"qwable_task_quality_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)

DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 8192
DEFAULT_MAX_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 7
DEFAULT_PORT_BASE = 18840
DEFAULT_REQUEST_TIMEOUT_S = 180
DEFAULT_STARTUP_TIMEOUT_S = 240


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    name: str
    model_path: Path
    device: str
    ngl: int
    role: str


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    scorer: str
    expected: Any
    max_tokens: int = DEFAULT_MAX_TOKENS


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        name="iq4_gpu",
        model_path=base.MODEL_IQ4_XS,
        device="ROCm0",
        ngl=99,
        role="mi210_iq4_task_quality",
    ),
    ArmSpec(
        name="q8_gpu",
        model_path=base.MODEL_Q8_0,
        device="ROCm0",
        ngl=99,
        role="mi210_q8_task_quality",
    ),
    ArmSpec(
        name="iq4_cpu",
        model_path=base.MODEL_IQ4_XS,
        device="none",
        ngl=0,
        role="cpu_iq4_task_quality",
    ),
    ArmSpec(
        name="q8_cpu",
        model_path=base.MODEL_Q8_0,
        device="none",
        ngl=0,
        role="cpu_q8_task_quality",
    ),
)

TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="arithmetic_sum_json",
        prompt=(
            "Compute 34 + 21. Return only minified JSON with exactly one key "
            '"answer" whose string value is the integer result.'
        ),
        scorer="json_exact",
        expected={"answer": "55"},
    ),
    TaskSpec(
        task_id="logic_transitive_json",
        prompt=(
            "All dax are mip. Every mip is vun. Is every dax a vun? Return "
            'only minified JSON with exactly one key "answer" whose string '
            'value is "yes" or "no".'
        ),
        scorer="json_exact",
        expected={"answer": "yes"},
    ),
    TaskSpec(
        task_id="time_addition_json",
        prompt=(
            "A job starts at 09:15 and lasts 80 minutes. Return only minified "
            'JSON with exactly one key "finish_24h" using HH:MM.'
        ),
        scorer="json_exact",
        expected={"finish_24h": "10:35"},
    ),
    TaskSpec(
        task_id="option_reasoning_letter",
        prompt=(
            "A benchmark gives the correct answer in the prompt. Does a model "
            "passing that benchmark prove independent reasoning? A) yes B) no. "
            "Return exactly one character: A or B."
        ),
        scorer="exact",
        expected="B",
        max_tokens=16,
    ),
    TaskSpec(
        task_id="lowercase_five_words",
        prompt=(
            "In exactly five lowercase words, describe why held-out tests matter. "
            "Return only the five words."
        ),
        scorer="five_lowercase_words",
        expected=5,
        max_tokens=48,
    ),
    TaskSpec(
        task_id="sorted_keys_json",
        prompt=(
            "Given label=qwable and status=ready, return only minified JSON "
            "with exactly keys label and status sorted alphabetically."
        ),
        scorer="json_exact",
        expected={"label": "qwable", "status": "ready"},
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwable IQ4/Q8 task-quality runner")
    parser.add_argument("--execute", action="store_true", help="Run selected arms after writing the plan")
    parser.add_argument(
        "--only",
        action="append",
        choices=[arm.name for arm in ARMS],
        help="Arm to execute. May be repeated. Defaults to GPU IQ4 and Q8.",
    )
    parser.add_argument(
        "--all-arms",
        action="store_true",
        help="Execute all CPU and GPU arms unless --only is provided",
    )
    parser.add_argument(
        "--allow-glm-download",
        action="store_true",
        help="Override GLM download guard in execute mode",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    return parser.parse_args(argv)


def selected_arm_indices(args: argparse.Namespace) -> list[int]:
    if args.only:
        wanted = set(args.only)
        return [index for index, arm in enumerate(ARMS) if arm.name in wanted]
    if args.all_arms:
        return list(range(len(ARMS)))
    return [0, 1]


def normalize_text(text: str) -> str:
    return text.strip().replace("\u00a0", " ")


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def parse_json_content(content: str) -> tuple[str, Any | None]:
    parsed = base.parse_content_json(content)
    return str(parsed["content_json_mode"]), parsed["content_json"]


def score_task(task: TaskSpec, content: str) -> dict[str, Any]:
    normalized = normalize_text(content)
    if task.scorer == "json_exact":
        json_mode, parsed = parse_json_content(normalized)
        expected = task.expected
        passed = parsed == expected
        return {
            "passed": passed,
            "scorer": task.scorer,
            "expected": expected,
            "observed": parsed,
            "json_mode": json_mode,
            "normalized": normalized,
        }
    if task.scorer == "exact":
        passed = normalized == str(task.expected)
        return {
            "passed": passed,
            "scorer": task.scorer,
            "expected": task.expected,
            "observed": normalized,
            "normalized": normalized,
        }
    if task.scorer == "five_lowercase_words":
        words = normalized.split()
        passed = (
            len(words) == int(task.expected)
            and all(re.fullmatch(r"[a-z]+", word) for word in words)
        )
        return {
            "passed": passed,
            "scorer": task.scorer,
            "expected_word_count": task.expected,
            "observed_words": words,
            "normalized": normalized,
        }
    raise ValueError(f"unknown scorer: {task.scorer}")


def launch_argv(arm: ArmSpec, port: int, args: argparse.Namespace) -> list[str]:
    return [
        str(base.SERVER_BIN),
        "-m",
        str(arm.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        arm.device,
        "-ngl",
        str(arm.ngl),
        "-t",
        str(args.threads),
        "-c",
        str(args.context),
        "-fa",
        "on",
        "-rea",
        "off",
    ]


def task_payload(task: TaskSpec, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "Answer only what the user asks for. Do not include reasoning or markdown.",
            },
            {"role": "user", "content": task.prompt},
        ],
        "max_tokens": min(task.max_tokens, args.max_tokens),
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 1,
        "seed": args.seed,
        "stream": False,
    }


def arm_port(args: argparse.Namespace, arm_index: int) -> int:
    return args.port_base + (arm_index * 10)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "qwable_task_quality_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "experimental_root": str(base.EXPERIMENTAL_ROOT),
        "server_bin": str(base.SERVER_BIN),
        "ld_library_path": str(base.SERVER_LIB_DIR),
        "selected_arms": [ARMS[index].name for index in selected_arm_indices(args)],
        "glm_guard": {
            "pattern": base.GLM_PATTERN,
            "active": base.glm_download_active(),
            "blocked_in_execute": True,
            "allow_override_flag": "--allow-glm-download",
        },
        "request": {
            "context": args.context,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "request_timeout_s": args.request_timeout,
            "startup_timeout_s": args.startup_timeout,
        },
        "arms": [
            {
                "name": arm.name,
                "role": arm.role,
                "model_path": str(arm.model_path),
                "device": arm.device,
                "ngl": arm.ngl,
                "port": arm_port(args, index),
            }
            for index, arm in enumerate(ARMS)
        ],
        "tasks": [dataclasses.asdict(task) for task in TASKS],
        "classification": (
            "small deterministic task-quality slice; not a role-quality gate "
            "or production-stack promotion decision by itself"
        ),
    }


def write_plan(output_dir: Path, plan: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "responses").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def run_arm(
    args: argparse.Namespace,
    output_dir: Path,
    arm_index: int,
    query: Callable[[int, dict[str, Any], int], tuple[dict[str, Any], str]] = base.query_chat,
) -> dict[str, Any]:
    for directory in (output_dir / "responses", output_dir / "results", output_dir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    arm = ARMS[arm_index]
    port = arm_port(args, arm_index)
    log_path = output_dir / "logs" / f"{arm.name}.server.log"
    proc = None
    task_records: list[dict[str, Any]] = []
    try:
        proc = base.launch_server(launch_argv(arm, port, args), log_path)
        base.wait_for_health(port, args.startup_timeout, pid=proc.pid)
        for task in TASKS:
            payload = task_payload(task, args)
            response, raw = query(port, payload, args.request_timeout)
            response_dir = output_dir / "responses" / arm.name
            response_dir.mkdir(parents=True, exist_ok=True)
            raw_path = response_dir / f"{task.task_id}.raw.json"
            raw_path.write_text(raw, encoding="utf-8")
            content = extract_content(response)
            score = score_task(task, content)
            timings = response.get("timings") if isinstance(response, dict) else None
            usage = response.get("usage") if isinstance(response, dict) else None
            task_records.append(
                {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "response_path": str(raw_path),
                    "content": content,
                    "score": score,
                    "timings": timings,
                    "usage": usage,
                }
            )
    finally:
        if proc is not None:
            try:
                base.terminate_server(proc)
            finally:
                log_handle = getattr(proc, "_qwable_log_handle", None)
                if log_handle is not None:
                    log_handle.close()

    passed = sum(1 for record in task_records if record["score"]["passed"])
    decode_rates = [
        float(record["timings"]["predicted_per_second"])
        for record in task_records
        if isinstance(record.get("timings"), dict)
        and isinstance(record["timings"].get("predicted_per_second"), (int, float))
    ]
    prompt_rates = [
        float(record["timings"]["prompt_per_second"])
        for record in task_records
        if isinstance(record.get("timings"), dict)
        and isinstance(record["timings"].get("prompt_per_second"), (int, float))
    ]
    arm_result = {
        "arm": arm.name,
        "role": arm.role,
        "model_path": str(arm.model_path),
        "device": arm.device,
        "ngl": arm.ngl,
        "port": port,
        "passed": passed,
        "total": len(task_records),
        "pass_rate": passed / len(task_records) if task_records else 0.0,
        "mean_decode_tps": sum(decode_rates) / len(decode_rates) if decode_rates else None,
        "mean_prompt_tps": sum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
        "tasks": task_records,
    }
    result_path = output_dir / "results" / f"{arm.name}.json"
    result_path.write_text(json.dumps(arm_result, indent=2, sort_keys=True), encoding="utf-8")
    return arm_result


def run_execute(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    results = [run_arm(args, output_dir, index) for index in selected_arm_indices(args)]
    summary = {
        "schema": "qwable_task_quality_execute.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute",
        "classification": (
            "small deterministic task-quality slice; compare IQ4_XS vs Q8_0, "
            "but do not promote a production role from this alone"
        ),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and not args.allow_glm_download and base.glm_download_active():
        print("FATAL: GLM-5.2 download is active; rerun with --allow-glm-download only if acceptable.", file=sys.stderr)
        return 75
    if not str(base.SERVER_BIN).startswith(str(base.EXPERIMENTAL_BIN_DIR)):
        raise RuntimeError(f"refusing non-experimental server binary: {base.SERVER_BIN}")

    plan = build_plan(args)
    write_plan(args.output_dir, plan)

    print("Qwable task-quality runner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"selected_arms: {', '.join(plan['selected_arms'])}")
    print(f"server_bin: {base.SERVER_BIN}")
    print(f"glm_active: {plan['glm_guard']['active']}")

    if not args.execute:
        print(f"Plan written to {args.output_dir / 'plan.json'}")
        return 0

    try:
        summary = run_execute(args, args.output_dir)
    except Exception as exc:
        print(f"Execute mode failed: {exc}", file=sys.stderr)
        return 1

    for result in summary["results"]:
        print(
            f"{result['arm']}: {result['passed']}/{result['total']} "
            f"mean_decode_tps={result['mean_decode_tps']}"
        )
    print(f"Summary written to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
