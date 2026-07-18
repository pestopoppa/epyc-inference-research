#!/usr/bin/env python3
"""Hy3 IQ1_M task-quality runner.

Hy3 admission already proved load/coherence and MTP/no-spec closure. This
runner answers the next question: whether the realistic no-spec CPU and
MI210-hybrid lanes are task-coherent enough to justify more architecture work.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import qwable_reasoning_economics_runner as base
import qwable_task_quality_runner as quality


RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = Path("/mnt/raid0/llm/models/hy3-angelslim/Hy3-IQ1_M-mtp.gguf")
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "hy3_task_quality"
    / f"hy3_task_quality_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)

DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 7
DEFAULT_PORT_BASE = 19240
DEFAULT_REQUEST_TIMEOUT_S = 300
DEFAULT_STARTUP_TIMEOUT_S = 900


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    name: str
    device: str
    ngl: int
    role: str
    resource_class: str
    hybrid: bool = False


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        name="hybrid_nospec",
        device="ROCm0",
        ngl=99,
        role="mi210_hybrid_cpu_experts_nospec",
        resource_class="MI210 + CPU experts",
        hybrid=True,
    ),
    ArmSpec(
        name="cpu_nospec",
        device="none",
        ngl=0,
        role="cpu_nospec",
        resource_class="CPU-only",
    ),
)

TASKS: tuple[quality.TaskSpec, ...] = (
    quality.TaskSpec(
        task_id="strict_json",
        prompt='Return exactly this minified JSON and nothing else: {"status":"ok","model":"hy3"}',
        scorer="json_exact",
        expected={"status": "ok", "model": "hy3"},
        max_tokens=64,
    ),
    quality.TaskSpec(
        task_id="simple_math",
        prompt="Answer with only the integer result: 37 + 58",
        scorer="exact",
        expected="95",
        max_tokens=32,
    ),
    quality.TaskSpec(
        task_id="lowercase_six_words",
        prompt=(
            "In exactly six lowercase words, describe why sparse routing helps. "
            "Return only the six words."
        ),
        scorer="five_lowercase_words",
        expected=6,
        max_tokens=64,
    ),
    quality.TaskSpec(
        task_id="routing_tradeoffs",
        prompt=(
            "Compare dense inference with sparse expert routing in under 140 words. "
            "Mention compute, memory bandwidth, load balancing, and routing overhead."
        ),
        scorer="contains_all_groups",
        expected={
            "groups": [
                ["compute", "computation"],
                ["memory bandwidth", "bandwidth"],
                ["load balancing", "load-balanced", "imbalance"],
                ["routing overhead", "router overhead", "routing cost"],
            ],
            "case_sensitive": False,
        },
        max_tokens=256,
    ),
    quality.TaskSpec(
        task_id="needle_code",
        prompt=(
            "Read the passage and answer with only the deployment code.\n\n"
            "Operators compared CPU-only and hybrid inference lanes. Several "
            "unrelated identifiers appear: HX-1000, ROUTE-77, and TEST-404. "
            "The deployment code to preserve is HY3-DELTA-9421. Later notes "
            "mention caches, ports, and throughput, but no other deployment code.\n\n"
            "Question: what is the deployment code?"
        ),
        scorer="exact",
        expected="HY3-DELTA-9421",
        max_tokens=48,
    ),
    quality.TaskSpec(
        task_id="binary_search_code",
        prompt=(
            "Write a Python binary_search(arr, target) function. Return code only. "
            "It must handle an empty array and return -1 when absent."
        ),
        scorer="contains_all",
        expected={
            "terms": ["def binary_search", "while", "return -1"],
            "case_sensitive": True,
        },
        max_tokens=384,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hy3 IQ1_M task-quality runner")
    parser.add_argument("--execute", action="store_true", help="Run selected arms after writing the plan")
    parser.add_argument(
        "--only",
        action="append",
        choices=[arm.name for arm in ARMS],
        help="Arm to execute. May be repeated. Defaults to hybrid then CPU no-spec.",
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
    parser.add_argument(
        "--allow-glm-download",
        action="store_true",
        help="Override GLM download guard in execute mode.",
    )
    return parser.parse_args(argv)


def selected_arm_indices(args: argparse.Namespace) -> list[int]:
    if args.only:
        wanted = set(args.only)
        return [index for index, arm in enumerate(ARMS) if arm.name in wanted]
    return list(range(len(ARMS)))


def arm_port(args: argparse.Namespace, index: int) -> int:
    return args.port_base + (index * 10)


def list_llama_server_pids() -> list[int]:
    probe = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in probe.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, args_text = parts
        argv0 = args_text.split(maxsplit=1)[0]
        if Path(argv0).name == "llama-server" and pid_text.isdigit():
            pids.append(int(pid_text))
    return sorted(pids)


def validate_experimental_server() -> Path:
    resolved = base.SERVER_BIN.resolve()
    production = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server").resolve()
    if resolved == production:
        raise RuntimeError("refusing production v6 llama-server binary")
    if base.EXPERIMENTAL_ROOT not in resolved.parents and resolved.parent != base.EXPERIMENTAL_BIN_DIR:
        raise RuntimeError(f"refusing non-experimental server binary: {resolved}")
    return resolved


def launch_argv(arm: ArmSpec, port: int, args: argparse.Namespace) -> list[str]:
    argv = [
        str(validate_experimental_server()),
        "-m",
        str(MODEL_PATH),
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
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    if arm.device == "none":
        argv.extend(["--device-draft", "none"])
    if arm.hybrid:
        argv.extend(["--cpu-moe", "--fit", "on"])
    return argv


def task_payload(task: quality.TaskSpec, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "Answer only what the user asks for. Do not include reasoning unless explicitly requested.",
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


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "hy3_task_quality_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "experimental_root": str(base.EXPERIMENTAL_ROOT),
        "server_bin": str(validate_experimental_server()),
        "model_path": str(MODEL_PATH),
        "selected_arms": [ARMS[index].name for index in selected_arm_indices(args)],
        "request": {
            "context": args.context,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "request_timeout_s": args.request_timeout,
            "startup_timeout_s": args.startup_timeout,
        },
        "glm_guard": {
            "pattern": base.GLM_PATTERN,
            "active": base.glm_download_active(),
            "blocked_in_execute": True,
            "allow_override_flag": "--allow-glm-download",
        },
        "preexisting_llama_server_pids": list_llama_server_pids(),
        "gpu_only_disposition": "not runnable on one MI210 for this 91.8GB artifact; meaningful GPU lane is --cpu-moe hybrid",
        "arms": [
            {
                "name": arm.name,
                "role": arm.role,
                "device": arm.device,
                "ngl": arm.ngl,
                "hybrid": arm.hybrid,
                "resource_class": arm.resource_class,
                "port": arm_port(args, index),
                "command": " ".join(launch_argv(arm, arm_port(args, index), args)),
            }
            for index, arm in enumerate(ARMS)
        ],
        "tasks": [dataclasses.asdict(task) for task in TASKS],
        "classification": (
            "deterministic task-quality and CPU-vs-hybrid architecture-fit slice; "
            "does not register Hy3 into production roles by itself"
        ),
    }


def write_plan(output_dir: Path, plan: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "responses", "results"):
        (output_dir / child).mkdir(exist_ok=True)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def extract_content(response: dict[str, Any]) -> str:
    return quality.extract_content(response)


def run_arm(
    args: argparse.Namespace,
    output_dir: Path,
    arm_index: int,
    query: Callable[[int, dict[str, Any], int], tuple[dict[str, Any], str]] = base.query_chat,
) -> dict[str, Any]:
    for child in ("logs", "responses", "results"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    arm = ARMS[arm_index]
    port = arm_port(args, arm_index)
    log_path = output_dir / "logs" / f"{arm.name}.server.log"
    proc: subprocess.Popen[str] | None = None
    task_records: list[dict[str, Any]] = []
    try:
        proc = base.launch_server(launch_argv(arm, port, args), log_path)
        base.wait_for_health(port, args.startup_timeout, pid=proc.pid)
        for task in TASKS:
            response, raw = query(port, task_payload(task, args), args.request_timeout)
            response_dir = output_dir / "responses" / arm.name
            response_dir.mkdir(parents=True, exist_ok=True)
            raw_path = response_dir / f"{task.task_id}.raw.json"
            raw_path.write_text(raw, encoding="utf-8")
            content = extract_content(response)
            task_records.append(
                {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "response_path": str(raw_path),
                    "content": content,
                    "score": quality.score_task(task, content),
                    "timings": response.get("timings") if isinstance(response, dict) else None,
                    "usage": response.get("usage") if isinstance(response, dict) else None,
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
    result = {
        "arm": arm.name,
        "role": arm.role,
        "resource_class": arm.resource_class,
        "model_path": str(MODEL_PATH),
        "device": arm.device,
        "ngl": arm.ngl,
        "hybrid": arm.hybrid,
        "port": port,
        "passed": passed,
        "total": len(task_records),
        "pass_rate": passed / len(task_records) if task_records else 0.0,
        "mean_decode_tps": sum(decode_rates) / len(decode_rates) if decode_rates else None,
        "mean_prompt_tps": sum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
        "tasks": task_records,
    }
    (output_dir / "results" / f"{arm.name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def verify_cleanup(allowed_pids: list[int]) -> dict[str, Any]:
    observed = list_llama_server_pids()
    extra = sorted(set(observed) - set(allowed_pids))
    return {
        "allowed_pids": allowed_pids,
        "observed_pids": observed,
        "extra_pids": extra,
        "passed": not extra,
    }


def run_execute(args: argparse.Namespace, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    allowed_pids = list(plan["preexisting_llama_server_pids"])
    results = [run_arm(args, output_dir, index) for index in selected_arm_indices(args)]
    cleanup = verify_cleanup(allowed_pids)
    summary = {
        "schema": "hy3_task_quality_execute.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute",
        "classification": plan["classification"],
        "gpu_only_disposition": plan["gpu_only_disposition"],
        "results": results,
        "quality_gate_passed": all(result["passed"] == result["total"] for result in results),
        "cleanup": cleanup,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not cleanup["passed"]:
        raise RuntimeError(f"cleanup failed; extra llama-server pids: {cleanup['extra_pids']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and not args.allow_glm_download and base.glm_download_active():
        print("FATAL: GLM-5.2 download is active; rerun with --allow-glm-download only if acceptable.", file=sys.stderr)
        return 75
    if not MODEL_PATH.exists():
        print(f"FATAL: missing Hy3 model artifact: {MODEL_PATH}", file=sys.stderr)
        return 75

    plan = build_plan(args)
    write_plan(args.output_dir, plan)
    print("Hy3 task-quality runner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"selected_arms: {', '.join(plan['selected_arms'])}")
    print(f"server_bin: {plan['server_bin']}")
    print(f"glm_active: {plan['glm_guard']['active']}")
    if not args.execute:
        print(f"Plan written to {args.output_dir / 'plan.json'}")
        return 0

    try:
        summary = run_execute(args, args.output_dir, plan)
    except Exception as exc:
        print(f"Execute mode failed: {exc}", file=sys.stderr)
        return 1

    for result in summary["results"]:
        print(
            f"{result['arm']}: {result['passed']}/{result['total']} "
            f"mean_decode_tps={result['mean_decode_tps']}"
        )
    print(f"cleanup_passed: {summary['cleanup']['passed']}")
    print(f"Summary written to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
