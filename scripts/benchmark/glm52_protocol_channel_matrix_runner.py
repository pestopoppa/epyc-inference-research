#!/usr/bin/env python3
"""GLM-5.2 protocol/channel matrix runner.

This is a narrow follow-up to glm52_dsa_probe_runner.py. It reuses the
validated GLM server launch, inventory, prompt-floor, and cleanup helpers, but
keeps one server alive long enough to test multiple request endpoints against
the same prompt band and runtime mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import glm52_dsa_probe_runner as base


SCHEMA = "glm52_protocol_channel_matrix.v1"
JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision"],
        "properties": {"decision": {"enum": ["allow"]}},
    },
    separators=(",", ":"),
)


@dataclass(frozen=True)
class PromptBand:
    name: str
    context_length: int
    min_prompt_tokens: int
    indexer_top_k: int
    prompt_context_guard_tokens: int


@dataclass(frozen=True)
class RuntimeMode:
    name: str
    task_line: str
    answer_instruction: str
    expected: str
    validator: str
    extra_args: tuple[str, ...]


PROMPT_BANDS: dict[str, PromptBand] = {
    "p2056_tk2048": PromptBand(
        name="p2056_tk2048",
        context_length=4096,
        min_prompt_tokens=2056,
        indexer_top_k=2048,
        prompt_context_guard_tokens=128,
    ),
    "p2168_tk4096": PromptBand(
        name="p2168_tk4096",
        context_length=4096,
        min_prompt_tokens=2168,
        indexer_top_k=4096,
        prompt_context_guard_tokens=128,
    ),
    "p3045_tk4096": PromptBand(
        name="p3045_tk4096",
        context_length=4096,
        min_prompt_tokens=3045,
        indexer_top_k=4096,
        prompt_context_guard_tokens=128,
    ),
    "p12000_tk16384": PromptBand(
        name="p12000_tk16384",
        context_length=16384,
        min_prompt_tokens=12000,
        indexer_top_k=16384,
        prompt_context_guard_tokens=512,
    ),
}

RUNTIME_MODES: dict[str, RuntimeMode] = {
    "free_auto": RuntimeMode(
        name="free_auto",
        task_line="Return exactly READY and nothing else.",
        answer_instruction="Do not explain. Do not use markdown.",
        expected="READY",
        validator="exact_ready",
        extra_args=("--reasoning-format", "deepseek", "--reasoning", "auto"),
    ),
    "free_reasoning_off": RuntimeMode(
        name="free_reasoning_off",
        task_line="Return exactly READY and nothing else.",
        answer_instruction="Do not explain. Do not use markdown.",
        expected="READY",
        validator="exact_ready",
        extra_args=(
            "--reasoning-format",
            "deepseek",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
        ),
    ),
    "json_reasoning_off": RuntimeMode(
        name="json_reasoning_off",
        task_line='Return exactly {"decision":"allow"} and nothing else.',
        answer_instruction="Do not explain. Do not use markdown.",
        expected='{"decision":"allow"}',
        validator="json_decision_allow",
        extra_args=(
            "--reasoning-format",
            "deepseek",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--json-schema",
            JSON_SCHEMA,
        ),
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_names(names: list[str], allowed: dict[str, Any], label: str) -> list[str]:
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    return names


def pgrep(pattern: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        rows.append({"pid": int(parts[0]), "command": parts[1] if len(parts) > 1 else ""})
    return rows


def extract_channels(response: dict[str, Any]) -> dict[str, str]:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    text = first.get("text") if isinstance(first, dict) else None
    raw_content = response.get("content")
    return {
        "content": content if isinstance(content, str) else "",
        "reasoning_content": reasoning if isinstance(reasoning, str) else "",
        "text": text if isinstance(text, str) else "",
        "raw_content": raw_content if isinstance(raw_content, str) else "",
        "combined": base._response_completion_text(response),
    }


def validate_response(mode: RuntimeMode, channels: dict[str, str]) -> dict[str, Any]:
    combined = channels["combined"].strip()
    content = channels["content"].strip()
    primary = content or combined
    if mode.validator == "exact_ready":
        return {
            "passed": primary == mode.expected,
            "validator": mode.validator,
            "primary_text": primary[:200],
            "combined_exact": combined == mode.expected,
            "combined_contains_expected": mode.expected in combined,
        }
    if mode.validator == "json_decision_allow":
        parsed: Any = None
        error: str | None = None
        try:
            parsed = json.loads(primary)
        except json.JSONDecodeError as exc:
            error = str(exc)
        return {
            "passed": parsed == {"decision": "allow"},
            "validator": mode.validator,
            "primary_text": primary[:200],
            "json_error": error,
            "parsed": parsed,
        }
    raise ValueError(f"unsupported validator: {mode.validator}")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    band_names = validate_names(split_csv(args.bands), PROMPT_BANDS, "band")
    mode_names = validate_names(split_csv(args.modes), RUNTIME_MODES, "mode")
    endpoints = validate_names(split_csv(args.endpoints), {name: name for name in base.REQUEST_ENDPOINTS}, "endpoint")
    binary = base.resolve_binary(args.binary)
    library_path = base.resolve_library_path(binary, args.library_path)
    inventory = base.collect_inventory(args.model_dir)
    primary_shard = Path(inventory["primary_shard"]) if inventory["primary_shard"] else args.model_dir

    cells: list[dict[str, Any]] = []
    cell_idx = 0
    for band_name in band_names:
        band = PROMPT_BANDS[band_name]
        for mode_name in mode_names:
            mode = RUNTIME_MODES[mode_name]
            log_file = args.output_dir / "logs" / f"{band.name}__{mode.name}.server.log"
            port = args.port_base + cell_idx
            server = base._server_spec(
                binary=binary,
                library_path=library_path,
                model_path=primary_shard,
                port=port,
                context_length=band.context_length,
                threads=args.threads,
                ubatch=args.ubatch,
                indexer_top_k=band.indexer_top_k,
                trace_logs=args.trace_logs,
                metrics=args.metrics,
                log_file=log_file if args.trace_logs else None,
                extra_args=list(mode.extra_args),
            )
            cells.append(
                {
                    "band": band.__dict__,
                    "mode": {
                        "name": mode.name,
                        "task_line": mode.task_line,
                        "answer_instruction": mode.answer_instruction,
                        "expected": mode.expected,
                        "validator": mode.validator,
                        "extra_args": list(mode.extra_args),
                    },
                    "server": server,
                    "request": {
                        "endpoints": endpoints,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "timeout_s": args.request_timeout,
                    },
                }
            )
            cell_idx += 1

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "binary": str(binary),
        "library_path": str(library_path),
        "model_dir": str(args.model_dir.resolve()),
        "model_path": str(primary_shard),
        "output_dir": str(args.output_dir),
        "execution_allowed": inventory["status"] == "ready",
        "refusal_reasons": inventory["refusal_reasons"],
        "inventory": inventory,
        "preexisting_processes": pgrep("llama-server|llama-cli|autopilot|glm52"),
        "cells": cells,
        "execution": None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def request_artifact_paths(output_dir: Path, band: str, mode: str, endpoint: str) -> dict[str, Path]:
    stem = f"{band}__{mode}__{endpoint}"
    artifact_dir = output_dir / "artifacts"
    return {
        "prompt": artifact_dir / f"{stem}.prompt.txt",
        "request": artifact_dir / f"{stem}.request.json",
        "response": artifact_dir / f"{stem}.response.json",
    }


def write_request_artifacts(
    output_dir: Path,
    band: str,
    mode: str,
    endpoint: str,
    prompt: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
    port: int,
) -> dict[str, str]:
    paths = request_artifact_paths(output_dir, band, mode, endpoint)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["prompt"].write_text(prompt, encoding="utf-8")
    write_json(
        paths["request"],
        {
            "endpoint": endpoint,
            "url": base.build_request_url(port, endpoint),
            "payload": request_payload,
        },
    )
    write_json(paths["response"], response)
    return {key: str(value) for key, value in paths.items()}


def call_endpoint(
    port: int,
    endpoint: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = base.build_request_payload(endpoint, prompt, max_tokens, temperature, seed)
    try:
        response = base.call_completion(port, prompt, max_tokens, temperature, seed, timeout_s, endpoint)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        response = {
            "request_error": base._request_error_payload(exc),
            "usage": {},
            "timings": {},
            "choices": [],
        }
    return payload, response


def run_cell(cell: dict[str, Any], output_dir: Path, request_timeout: int) -> dict[str, Any]:
    band_name = cell["band"]["name"]
    mode_name = cell["mode"]["name"]
    proc = base.launch_server(cell["server"]["server_command"])
    endpoint_results: list[dict[str, Any]] = []
    prompt_info: dict[str, Any] | None = None
    try:
        base.wait_for_health(cell["server"]["port"], timeout_s=240)
        prompt_info = base.build_prompt_with_token_floor(
            task_line=cell["mode"]["task_line"],
            context_length=cell["band"]["context_length"],
            min_prompt_tokens=cell["band"]["min_prompt_tokens"],
            max_completion_tokens=cell["request"]["max_tokens"],
            prompt_context_guard_tokens=cell["band"]["prompt_context_guard_tokens"],
            token_counter=lambda prompt: base.count_prompt_tokens(
                cell["server"]["port"],
                prompt,
                max(60, min(request_timeout, 600)),
            ),
            answer_instruction=cell["mode"]["answer_instruction"],
        )
        mode = RUNTIME_MODES[mode_name]
        for endpoint in cell["request"]["endpoints"]:
            payload, response = call_endpoint(
                cell["server"]["port"],
                endpoint,
                prompt_info["prompt"],
                cell["request"]["max_tokens"],
                cell["request"]["temperature"],
                cell["request"]["seed"],
                cell["request"]["timeout_s"],
            )
            channels = extract_channels(response)
            validation = validate_response(mode, channels)
            artifacts = write_request_artifacts(
                output_dir,
                band_name,
                mode_name,
                endpoint,
                prompt_info["prompt"],
                payload,
                response,
                cell["server"]["port"],
            )
            endpoint_results.append(
                {
                    "endpoint": endpoint,
                    "status": "ok" if validation["passed"] else "failed_validation",
                    "validation": validation,
                    "usage": response.get("usage", {}),
                    "timings": response.get("timings", {}),
                    "channels": {key: value[:300] for key, value in channels.items()},
                    "artifacts": artifacts,
                    "request_error": response.get("request_error"),
                }
            )
    finally:
        base.terminate_server(proc)

    assert prompt_info is not None
    server_log = base.summarize_server_log(cell["server"].get("log_file"))
    return {
        "band": band_name,
        "mode": mode_name,
        "status": "ok" if all(result["status"] == "ok" for result in endpoint_results) else "failed",
        "port": cell["server"]["port"],
        "context_length": cell["band"]["context_length"],
        "indexer_top_k": cell["band"]["indexer_top_k"],
        "prompt_token_count": prompt_info["prompt_token_count"],
        "prompt_token_min": prompt_info["prompt_token_min"],
        "prompt_char_count": prompt_info["prompt_char_count"],
        "server_log": server_log,
        "endpoints": endpoint_results,
    }


def run_execution(plan: dict[str, Any], request_timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    results = [run_cell(cell, Path(plan["output_dir"]), request_timeout) for cell in plan["cells"]]
    return {
        "status": "ok" if all(result["status"] == "ok" for result in results) else "failed",
        "elapsed_s": round(time.monotonic() - started, 3),
        "cells": results,
        "post_processes": pgrep("llama-server|llama-cli|autopilot|glm52"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM-5.2 protocol/channel matrix runner")
    parser.add_argument("--execute", action="store_true", help="Run the matrix. Default is dry-run only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.RESEARCH_ROOT / "data" / "glm52_protocol_channel_matrix" / utc_stamp(),
    )
    parser.add_argument("--model-dir", type=Path, default=base.MODEL_DIR)
    parser.add_argument("--binary", type=Path, default=base.DEFAULT_BINARY)
    parser.add_argument("--library-path", type=Path, default=None)
    parser.add_argument("--bands", default="p2056_tk2048,p2168_tk4096,p12000_tk16384")
    parser.add_argument("--modes", default="free_reasoning_off,json_reasoning_off")
    parser.add_argument(
        "--endpoints",
        default="chat",
        help="Comma-separated endpoints to probe. Use explicit v1_completions/completion arms only in a narrow run; they can be much slower for GLM.",
    )
    parser.add_argument("--threads", type=int, default=base.DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=base.DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--port-base", type=int, default=19420)
    parser.add_argument("--trace-logs", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    plan = build_plan(args)
    write_json(args.output_dir / "plan.json", plan)
    if not args.execute:
        print(f"dry-run wrote {args.output_dir / 'plan.json'}")
        return 0
    if not plan["execution_allowed"]:
        print("execution refused: " + "; ".join(plan["refusal_reasons"]), file=sys.stderr)
        return 2
    plan["execution"] = run_execution(plan, args.request_timeout)
    write_json(args.output_dir / "summary.json", plan)
    status = plan["execution"]["status"]
    print(f"execution {status}; wrote {args.output_dir / 'summary.json'}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
