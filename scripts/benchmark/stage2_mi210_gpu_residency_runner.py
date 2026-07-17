#!/usr/bin/env python3
"""Dry-run-first Stage-2 MI210 frontdoor residency runner.

Stage 2 compares the Qwen3.6 frontdoor fully resident on ROCm0:

  - GPU target, speculation disabled
  - GPU target, native MTP
  - GPU target plus co-resident external Qwen3.5-0.8B drafter

The runner is intentionally sequential and single-owner. It reuses the Stage-1
fresh-server helpers so each arm gets a new port, log, raw response, result, and
cleanup verification.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage1_mi210_gpu_drafter_planner import (
    DEFAULT_CONTEXT,
    DEFAULT_DRAFT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_COMPLETION_RATIO,
    DEFAULT_PROMPT_PACK,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SPEC_P_SPLIT,
    DEFAULT_STARTUP_TIMEOUT_S,
    DEFAULT_TARGET_MODEL,
    DEFAULT_THREADS,
    DEFAULT_UBATCH,
    EXPERIMENTAL_BIN_DIR,
    EXPERIMENTAL_SERVER,
    PASS_SPEEDUP_THRESHOLD,
    canonical_json,
    collect_guard_state,
    collect_process_snapshot,
    launch_server,
    pick_ephemeral_port,
    query_chat,
    sha256_text,
    summarize_arm,
    terminate_server,
    validate_experimental_server,
    wait_for_health,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "specdec_frontdoor_alpha"
    / f"stage2_mi210_gpu_residency_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_NATIVE_DRAFT_MAX = 3
DEFAULT_EXTERNAL_DRAFT_MAX = 1


@dataclass(frozen=True)
class Stage2Arm:
    name: str
    purpose: str
    spec_type: str
    speculative: bool


ARMS = (
    Stage2Arm(
        name="gpu_no_spec",
        purpose="MI210-resident target baseline with speculation disabled",
        spec_type="none",
        speculative=False,
    ),
    Stage2Arm(
        name="gpu_native_mtp",
        purpose="MI210-resident target with native MTP enabled",
        spec_type="draft-mtp",
        speculative=True,
    ),
    Stage2Arm(
        name="gpu_external_drafter",
        purpose="MI210-resident target plus co-resident external Qwen3.5-0.8B drafter",
        spec_type="draft-tree",
        speculative=True,
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _message_text(response: dict[str, Any]) -> dict[str, str]:
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}

    def coerce(value: Any) -> str:
        if isinstance(value, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in value
            )
        return str(value or "")

    return {
        "content": coerce(message.get("content", "")),
        "reasoning_content": coerce(message.get("reasoning_content", "")),
    }


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _server_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return (result.stdout + result.stderr).strip()


def build_server_argv(args: argparse.Namespace, arm: Stage2Arm, port: int | str) -> list[str]:
    argv = [
        "env",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_BIN_DIR}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(args.binary),
        "-m",
        str(args.target_model),
        "-t",
        str(args.threads),
        "-np",
        "1",
        "-c",
        str(args.context),
        "-ub",
        str(args.ubatch),
        "-ngl",
        str(args.n_gpu_layers),
        "--device",
        "ROCm0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        "--slots",
        "--jinja",
        "--reasoning",
        "auto",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
    ]
    if arm.spec_type == "none":
        argv.extend(["--spec-type", "none"])
    elif arm.spec_type == "draft-mtp":
        argv.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(args.native_draft_max),
            ]
        )
    elif arm.spec_type == "draft-tree":
        argv.extend(
            [
                "-md",
                str(args.draft_model),
                "--spec-type",
                "draft-tree",
                "--spec-draft-n-max",
                str(args.external_draft_max),
                "--spec-draft-p-split",
                str(args.spec_p_split),
                "--spec-draft-device",
                "ROCm0",
                "--spec-draft-ngl",
                str(args.n_gpu_layers),
            ]
        )
    else:
        raise ValueError(f"unknown Stage-2 spec type: {arm.spec_type}")
    return argv


def render_commands(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated dry-run package. Execute with the Python runner for cleanup guarantees.",
        f'export LD_LIBRARY_PATH="{EXPERIMENTAL_BIN_DIR}"',
        "",
    ]
    for arm in plan["arms"]:
        lines.append(f"# arm: {arm['name']}")
        lines.append(f"# purpose: {arm['purpose']}")
        lines.append(shlex.join(arm["argv"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    ports: dict[str, int] = {}
    for arm in ARMS:
        port = pick_ephemeral_port() if args.execute else args.template_port + len(ports)
        while port in ports.values():
            port = pick_ephemeral_port()
        ports[arm.name] = port
    return {
        "schema": "stage2_mi210_gpu_residency_plan.v1",
        "created_at": _utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "binary": str(args.binary),
        "server_version": _server_version(args.binary),
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "max_tokens": args.max_tokens,
        "native_draft_max": args.native_draft_max,
        "external_draft_max": args.external_draft_max,
        "pass_speedup_gte": PASS_SPEEDUP_THRESHOLD,
        "arms": [
            {
                "name": arm.name,
                "purpose": arm.purpose,
                "spec_type": arm.spec_type,
                "port": ports[arm.name],
                "argv": build_server_argv(args, arm, ports[arm.name]),
            }
            for arm in ARMS
        ],
    }


def run_arm(
    *,
    arm: Stage2Arm,
    argv: list[str],
    port: int,
    output_dir: Path,
    prompts: list[str],
    max_tokens: int,
    request_timeout: int,
    startup_timeout: int,
) -> dict[str, Any]:
    log_path = output_dir / f"{arm.name}.server.log"
    raw_path = output_dir / f"{arm.name}.raw.json"
    result_path = output_dir / f"{arm.name}.result.json"
    records: list[dict[str, Any]] = []
    proc: subprocess.Popen[str] | None = None
    started_at = _utc_now()
    startup_elapsed_s = 0.0
    try:
        started = time.monotonic()
        proc = launch_server(argv, log_path)
        wait_for_health(port, startup_timeout, pid=proc.pid)
        startup_elapsed_s = time.monotonic() - started
        for index, prompt in enumerate(prompts, start=1):
            try:
                response, raw_response, request_duration_s = query_chat(
                    port=port,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    timeout_s=request_timeout,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
            message = _message_text(response)
            timings = response.get("timings", {})
            semantic_output = {
                "content": message["content"],
                "reasoning_content": message["reasoning_content"],
            }
            records.append(
                {
                    "status": "ok",
                    "prompt_index": index,
                    "prompt": prompt,
                    "response_sha256": sha256_text(canonical_json(response)),
                    "output_sha256": sha256_text(canonical_json(semantic_output)),
                    "content": message["content"],
                    "reasoning_content": message["reasoning_content"],
                    "usage": response.get("usage", {}),
                    "timings": timings,
                    "draft_n": _integer(timings, "draft_n"),
                    "draft_n_accepted": _integer(timings, "draft_n_accepted"),
                    "request_duration_s": request_duration_s,
                    "raw_response": raw_response,
                }
            )
    except Exception as exc:
        records.append(
            {
                "status": "error",
                "error": str(exc),
                "prompt_index": len(records) + 1,
            }
        )
    finally:
        cleanup_error = None
        if proc is not None:
            try:
                terminate_server(proc, port)
            except Exception as exc:
                cleanup_error = str(exc)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        summary = summarize_arm(records, log_text, speculative=arm.speculative)
        result = {
            "arm": arm.name,
            "purpose": arm.purpose,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "port": port,
            "server_pid": proc.pid if proc is not None else None,
            "server_argv": argv,
            "startup_elapsed_s": startup_elapsed_s,
            "server_log": str(log_path),
            "raw_response_path": str(raw_path),
            "summary": summary,
            "records": [
                {key: value for key, value in record.items() if key != "raw_response"}
                for record in records
            ],
        }
        if cleanup_error:
            result["cleanup_error"] = cleanup_error
            result["summary"]["status_ok"] = False
        _write_json(
            raw_path,
            {
                "arm": arm.name,
                "responses": [
                    {
                        "prompt_index": record.get("prompt_index"),
                        "raw_response": record.get("raw_response"),
                    }
                    for record in records
                    if "raw_response" in record
                ],
            },
        )
        _write_json(result_path, result)
        if cleanup_error:
            raise RuntimeError(cleanup_error)
        return result


def speedup(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    base = float(baseline.get(key) or 0.0)
    value = float(candidate.get(key) or 0.0)
    return value / base if base > 0 else 0.0


def usable_draft(summary: dict[str, Any]) -> bool:
    return bool(summary.get("status_ok")) and int(summary.get("draft_n") or 0) > 0 and int(summary.get("draft_n_accepted") or 0) > 0


def run_execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    guards = collect_guard_state()
    if not guards.quiet_host_ready:
        raise RuntimeError(f"quiet host blockers present: {guards.quiet_host_blockers}")
    if not args.target_model.exists():
        raise FileNotFoundError(f"target model not found: {args.target_model}")
    if not args.draft_model.exists():
        raise FileNotFoundError(f"draft model not found: {args.draft_model}")

    results: dict[str, Any] = {}
    for arm_plan, arm in zip(plan["arms"], ARMS):
        results[arm.name] = run_arm(
            arm=arm,
            argv=arm_plan["argv"],
            port=int(arm_plan["port"]),
            output_dir=args.output_dir,
            prompts=args.prompts,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
        )

    baseline = results["gpu_no_spec"]["summary"]
    comparisons: dict[str, Any] = {}
    min_completion_tokens = int(len(args.prompts) * args.max_tokens * args.min_completion_ratio)
    for name in ("gpu_native_mtp", "gpu_external_drafter"):
        summary = results[name]["summary"]
        comparisons[name] = {
            "decode_speedup_vs_no_spec": speedup(summary, baseline, "predicted_per_second"),
            "wall_speedup_vs_no_spec": speedup(summary, baseline, "wall_tokens_per_second"),
            "usable_draft": usable_draft(summary),
        }

    native = results["gpu_native_mtp"]["summary"]
    external = results["gpu_external_drafter"]["summary"]
    comparisons["external_vs_native"] = {
        "decode_speedup": speedup(external, native, "predicted_per_second"),
        "wall_speedup": speedup(external, native, "wall_tokens_per_second"),
    }

    enough_completion = all(
        int(results[arm.name]["summary"].get("completion_tokens") or 0) >= min_completion_tokens
        for arm in ARMS
    )
    all_ok = all(bool(results[arm.name]["summary"].get("status_ok")) for arm in ARMS)
    best_spec = max(
        ("gpu_native_mtp", "gpu_external_drafter"),
        key=lambda name: comparisons[name]["decode_speedup_vs_no_spec"],
    )
    pass_gate = (
        all_ok
        and enough_completion
        and comparisons[best_spec]["usable_draft"]
        and comparisons[best_spec]["decode_speedup_vs_no_spec"] >= PASS_SPEEDUP_THRESHOLD
    )
    summary = {
        "schema": "stage2_mi210_gpu_residency_result.v1",
        "created_at": _utc_now(),
        "mode": "execute",
        "verdict": "pass" if pass_gate else "fail",
        "decision_grade": pass_gate,
        "pass_speedup_gte": PASS_SPEEDUP_THRESHOLD,
        "min_completion_tokens_per_arm": min_completion_tokens,
        "enough_completion": enough_completion,
        "all_ok": all_ok,
        "best_spec_arm": best_spec,
        "comparisons": comparisons,
        "quiet_host": {
            "ready": guards.quiet_host_ready,
            "blockers": guards.quiet_host_blockers,
            "process_snapshot": collect_process_snapshot(),
        },
        "plan": plan,
        "arms": {name: result["summary"] for name, result in results.items()},
        "artifacts": {
            "plan": str(args.output_dir / "plan.json"),
            "commands": str(args.output_dir / "commands.sh"),
            **{
                f"{arm.name}_result": str(args.output_dir / f"{arm.name}.result.json")
                for arm in ARMS
            },
            **{
                f"{arm.name}_log": str(args.output_dir / f"{arm.name}.server.log")
                for arm in ARMS
            },
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 MI210 frontdoor residency runner")
    parser.add_argument("--execute", action="store_true", help="Launch sequential fresh-server Stage-2 A/B/C")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=EXPERIMENTAL_SERVER)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--native-draft-max", type=int, default=DEFAULT_NATIVE_DRAFT_MAX)
    parser.add_argument("--external-draft-max", type=int, default=DEFAULT_EXTERNAL_DRAFT_MAX)
    parser.add_argument("--spec-p-split", type=float, default=DEFAULT_SPEC_P_SPLIT)
    parser.add_argument("--template-port", type=int, default=19220)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-completion-ratio", type=float, default=DEFAULT_MIN_COMPLETION_RATIO)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    args = parser.parse_args(argv)
    args.binary = validate_experimental_server(args.binary)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPT_PACK)
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not (0 < args.min_completion_ratio <= 1):
        parser.error("--min-completion-ratio must be in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "plan.json", plan)
    (args.output_dir / "commands.sh").write_text(render_commands(plan), encoding="utf-8")

    print("Stage-2 MI210 GPU residency runner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"binary: {args.binary}")
    print(f"plan: {args.output_dir / 'plan.json'}")
    print(f"commands: {args.output_dir / 'commands.sh'}")
    if not args.execute:
        print("Dry run only. No inference was launched.")
        return 0

    summary = run_execute(args, plan)
    print(f"summary: {args.output_dir / 'summary.json'}")
    print(f"verdict: {summary['verdict']}")
    for name, comparison in summary["comparisons"].items():
        if name == "external_vs_native":
            print(
                f"{name}: decode {comparison['decode_speedup']:.3f}x, "
                f"wall {comparison['wall_speedup']:.3f}x"
            )
        else:
            print(
                f"{name}: decode {comparison['decode_speedup_vs_no_spec']:.3f}x, "
                f"wall {comparison['wall_speedup_vs_no_spec']:.3f}x, "
                f"usable_draft={comparison['usable_draft']}"
            )
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
