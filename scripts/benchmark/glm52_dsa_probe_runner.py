#!/usr/bin/env python3
"""GLM-5.2 DSA probe runner.

Defaults to dry-run / preflight mode. The runner only stages commands unless
--execute is provided, and execution is refused until the GLM shard directory
is clean and all non-cache shards are present.

The plan is intentionally narrow:
  1. shard integrity inventory (size/path/count only; no hashing)
  2. short load/decode smoke
  3. long-context DSA probe
  4. KV-length scaling with a fixed indexer_top_k

The default binary points at the experimental v7 tree and the runtime
environment is sanitized with a pinned LD_LIBRARY_PATH. Production paths are
rejected by default.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent

MODEL_DIR = Path("/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M")
DEFAULT_BINARY = Path("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server")
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_LIBRARY_PATH = DEFAULT_BINARY.parent

SCHEMA = "glm52_dsa_probe_plan.v1"
REQUIRED_NON_CACHE_SHARDS = 6
DEFAULT_THREADS = 96
DEFAULT_UBATCH = 512
DEFAULT_MAX_TOKENS = 32
DEFAULT_THROUGHPUT_MAX_TOKENS = 768
DEFAULT_MIN_COMPLETION_TOKENS = 384
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
DEFAULT_INDEXER_TOP_K = 32
DEFAULT_REQUEST_TIMEOUT = 3600
DEFAULT_PROGRESS_POLL_INTERVAL = 30
REQUEST_ENDPOINTS = ("chat", "v1_completions", "completion")
REQUEST_ENDPOINT_PATHS = {
    "chat": "/v1/chat/completions",
    "v1_completions": "/v1/completions",
    "completion": "/completion",
}
DEFAULT_SHORT_CONTEXT = 4096
DEFAULT_LONG_CONTEXT = 32768
DEFAULT_KV_CONTEXTS = (4096, 8192, 16384, 32768)
DEFAULT_PORT_BASE = 19100
INDEXER_TOP_K_OVERRIDE_KEY = "glm-dsa.attention.indexer.top_k"

PROMPT_RESERVE_TOKENS = 192
PROMPT_CHARS_PER_TOKEN_HEURISTIC = 4.0
DEFAULT_MIN_PROMPT_TOKENS = 0
DEFAULT_PROMPT_CONTEXT_GUARD_TOKENS = 512
PROMPT_TOKEN_FLOOR_MAX_ATTEMPTS = 6
FILLER_TEXT = (
    "The GLM DSA probe keeps the context deterministic while the runner checks "
    "shard integrity, load behavior, and KV scaling under a fixed indexer "
    "configuration. "
)
LONG_OUTPUT_REPEAT_COUNT = 620
LONG_OUTPUT_TASK_LINE = (
    "First output READY. Then output the word tokenstream repeated exactly "
    f"{LONG_OUTPUT_REPEAT_COUNT} times, separated by single spaces. Do not use "
    "markdown. Do not explain. Do not stop early."
)
SHORT_ANSWER_INSTRUCTION = "Answer with a single short sentence."
LONG_OUTPUT_ANSWER_INSTRUCTION = (
    "Return the requested READY marker and tokenstream sequence only."
)


@dataclasses.dataclass(frozen=True)
class ShardRecord:
    path: str
    size: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _is_under_production_root(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(PRODUCTION_ROOT.resolve())
    except FileNotFoundError:
        return False


def resolve_binary(binary: Path) -> Path:
    resolved = binary.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"llama binary not found: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"llama binary is not executable: {resolved}")
    if _is_under_production_root(resolved):
        raise ValueError(f"refusing production llama binary: {resolved}")
    return resolved


def resolve_library_path(binary: Path, library_path: Path | None = None) -> Path:
    path = library_path if library_path is not None else binary.parent
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"library path not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"library path is not a directory: {resolved}")
    return resolved


def is_blocker_file(path: Path) -> bool:
    return path.name.endswith(".incomplete") or path.name.endswith(".lock")


def _best_manifest_match(model_dir: Path, shards: list[ShardRecord]) -> dict[str, Any]:
    tree_dir = model_dir / ".cache" / "huggingface" / "trees"
    best: dict[str, Any] = {
        "status": "missing",
        "path": None,
        "matched_shards": 0,
        "mismatches": [],
    }
    if not tree_dir.exists():
        return best

    shard_by_path = {record.path: record.size for record in shards}
    for tree_path in sorted(tree_dir.glob("*.json")):
        try:
            data = json.loads(tree_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            candidate = {
                "status": "unreadable",
                "path": str(tree_path),
                "matched_shards": 0,
                "mismatches": [str(exc)],
            }
            if best["status"] == "missing":
                best = candidate
            continue

        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict):
            continue

        matched = 0
        mismatches: list[str] = []
        for rel_path, local_size in shard_by_path.items():
            meta = files.get(rel_path)
            if not isinstance(meta, dict):
                mismatches.append(f"{rel_path}: missing from manifest")
                continue
            manifest_size = meta.get("size")
            lfs_size = meta.get("lfs_size", manifest_size)
            if local_size == manifest_size == lfs_size:
                matched += 1
            else:
                mismatches.append(
                    f"{rel_path}: local={local_size} manifest={manifest_size} lfs={lfs_size}"
                )

        candidate = {
            "status": "complete" if matched == len(shards) and not mismatches else "partial",
            "path": str(tree_path),
            "matched_shards": matched,
            "mismatches": mismatches,
        }
        if matched > int(best.get("matched_shards", 0)):
            best = candidate
        if candidate["status"] == "complete":
            return candidate

    return best


def collect_inventory(model_dir: Path) -> dict[str, Any]:
    resolved_dir = model_dir.expanduser().resolve()
    shards: list[ShardRecord] = []
    blocker_records: list[ShardRecord] = []
    stale_cache_markers: list[ShardRecord] = []

    if resolved_dir.exists():
        for entry in sorted(resolved_dir.rglob("*")):
            if not entry.is_file():
                continue
            rel = entry.relative_to(resolved_dir).as_posix()
            size = entry.stat().st_size
            if is_blocker_file(entry):
                blocker_records.append(ShardRecord(path=rel, size=size))
            if ".cache" not in entry.parts and entry.suffix == ".gguf":
                shards.append(ShardRecord(path=rel, size=size))

    shard_records = [dataclasses.asdict(record) for record in shards]
    manifest = _best_manifest_match(resolved_dir, shards)
    manifest_complete = (
        len(shards) == REQUIRED_NON_CACHE_SHARDS
        and manifest["status"] == "complete"
    )
    effective_blockers: list[ShardRecord] = []
    for record in blocker_records:
        is_stale_hf_incomplete = (
            manifest_complete
            and record.path.startswith(".cache/huggingface/download/")
            and record.path.endswith(".incomplete")
        )
        if is_stale_hf_incomplete:
            stale_cache_markers.append(record)
        else:
            effective_blockers.append(record)

    blocker_payload = [dataclasses.asdict(record) for record in effective_blockers]
    stale_cache_payload = [dataclasses.asdict(record) for record in stale_cache_markers]
    total_bytes = sum(record.size for record in shards)
    shard_count = len(shards)
    status = "ready" if shard_count == REQUIRED_NON_CACHE_SHARDS and not effective_blockers else "blocked"
    primary_shard = str((resolved_dir / shards[0].path).resolve()) if shards else None

    reasons: list[str] = []
    if not resolved_dir.exists():
        reasons.append(f"model directory missing: {resolved_dir}")
    if shard_count != REQUIRED_NON_CACHE_SHARDS:
        reasons.append(
            f"expected {REQUIRED_NON_CACHE_SHARDS} non-cache gguf shards, found {shard_count}"
        )
    if effective_blockers:
        reasons.append(f"found blocker files: {', '.join(record.path for record in effective_blockers)}")
    if manifest["status"] not in {"complete", "missing"}:
        reasons.append(f"manifest verification not complete: {manifest['status']}")

    return {
        "model_dir": str(resolved_dir),
        "status": status,
        "required_non_cache_shards": REQUIRED_NON_CACHE_SHARDS,
        "non_cache_shard_count": shard_count,
        "non_cache_shards": shard_records,
        "primary_shard": primary_shard,
        "blocker_files": blocker_payload,
        "stale_cache_marker_files": stale_cache_payload,
        "hf_tree_manifest": manifest,
        "total_shard_bytes": total_bytes,
        "refusal_reasons": reasons,
    }


def build_server_command(
    *,
    binary: Path,
    library_path: Path,
    model_path: Path,
    port: int,
    context_length: int,
    threads: int,
    ubatch: int,
    indexer_top_k: int,
    trace_logs: bool = False,
    log_file: Path | None = None,
    metrics: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={library_path}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(model_path),
        "--override-kv",
        f"{INDEXER_TOP_K_OVERRIDE_KEY}=int:{indexer_top_k}",
        "--device",
        "none",
        "-ngl",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(context_length),
        "-t",
        str(threads),
        "-ub",
        str(ubatch),
    ]
    if trace_logs:
        command.extend(["--log-verbosity", "4"])
        if log_file is not None:
            command.extend(["--log-file", str(log_file)])
    else:
        command.append("--log-disable")
    if metrics:
        command.append("--metrics")
    if extra_args:
        command.extend(extra_args)
    return command


def _target_prompt_chars(context_length: int) -> int:
    return int(max(0, (context_length - PROMPT_RESERVE_TOKENS) * PROMPT_CHARS_PER_TOKEN_HEURISTIC))


def _build_prompt_from_chars(
    task_line: str,
    target_chars: int,
    answer_instruction: str = SHORT_ANSWER_INSTRUCTION,
    needle_text: str | None = None,
    needle_depth: float = 0.5,
) -> str:
    needle_block = ""
    if needle_text:
        needle_block = f"\n\n--- NEEDLE RECORD ---\n{needle_text}\n--- END NEEDLE RECORD ---\n\n"
    filler_target_chars = max(0, target_chars - len(needle_block))
    repeats = max(1, filler_target_chars // len(FILLER_TEXT) + 1)
    filler = (FILLER_TEXT * repeats)[:filler_target_chars]
    if needle_block:
        depth = min(1.0, max(0.0, needle_depth))
        insert_at = int(len(filler) * depth)
        filler = f"{filler[:insert_at]}{needle_block}{filler[insert_at:]}"
    return (
        f"{filler}\n\n--- TASK ---\n"
        f"{task_line}\n"
        f"{answer_instruction}"
    )


def _build_prompt(
    task_line: str,
    context_length: int,
    answer_instruction: str = SHORT_ANSWER_INSTRUCTION,
    needle_text: str | None = None,
    needle_depth: float = 0.5,
) -> str:
    return _build_prompt_from_chars(
        task_line,
        _target_prompt_chars(context_length),
        answer_instruction=answer_instruction,
        needle_text=needle_text,
        needle_depth=needle_depth,
    )


def _prompt_spec(
    kind: str,
    context_length: int,
    task_line: str,
    min_prompt_tokens: int,
    prompt_context_guard_tokens: int,
    answer_instruction: str = SHORT_ANSWER_INSTRUCTION,
    needle_text: str | None = None,
    needle_depth: float = 0.5,
    expected_substring: str | None = None,
) -> dict[str, Any]:
    spec = {
        "kind": kind,
        "context_length": context_length,
        "task_line": task_line,
        "answer_instruction": answer_instruction,
        "chars_per_token_heuristic": PROMPT_CHARS_PER_TOKEN_HEURISTIC,
        "min_prompt_tokens": min_prompt_tokens,
        "prompt_context_guard_tokens": prompt_context_guard_tokens,
    }
    if needle_text:
        spec["needle_text"] = needle_text
        spec["needle_depth"] = needle_depth
    if expected_substring:
        spec["expected_substring"] = expected_substring
    return spec


def _request_spec(
    *,
    request_endpoint: str = "chat",
    max_tokens: int,
    seed: int,
    temperature: float,
    timeout_s: int,
    min_completion_tokens: int = 0,
    progress_poll_interval_s: int = 0,
    purpose: str = "coherence",
    stream: bool = False,
) -> dict[str, Any]:
    return {
        "request_endpoint": request_endpoint,
        "endpoint": REQUEST_ENDPOINT_PATHS[request_endpoint],
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": temperature,
        "timeout_s": timeout_s,
        "min_completion_tokens": min_completion_tokens,
        "progress_poll_interval_s": progress_poll_interval_s,
        "purpose": purpose,
        "stream": stream,
    }


def _server_spec(
    *,
    binary: Path,
    library_path: Path,
    model_path: Path,
    port: int,
    context_length: int,
    threads: int,
    ubatch: int,
    indexer_top_k: int,
    trace_logs: bool,
    metrics: bool,
    log_file: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    command = build_server_command(
        binary=binary,
        library_path=library_path,
        model_path=model_path,
        port=port,
        context_length=context_length,
        threads=threads,
        ubatch=ubatch,
        indexer_top_k=indexer_top_k,
        trace_logs=trace_logs,
        log_file=log_file,
        metrics=metrics,
        extra_args=extra_args,
    )
    return {
        "server_command": command,
        "server_command_shell": shlex.join(command),
        "port": port,
        "context_length": context_length,
        "threads": threads,
        "ubatch": ubatch,
        "indexer_top_k": indexer_top_k,
        "indexer_top_k_override": f"{INDEXER_TOP_K_OVERRIDE_KEY}=int:{indexer_top_k}",
        "trace_logs": trace_logs,
        "metrics": metrics,
        "log_file": str(log_file) if log_file is not None else None,
        "extra_args": extra_args or [],
    }


def selected_stage_names(args: argparse.Namespace) -> set[str] | None:
    if not args.only_stage:
        return None
    return set(args.only_stage)


def build_plan(args: argparse.Namespace, inventory: dict[str, Any], binary: Path, library_path: Path) -> dict[str, Any]:
    short_task = "Return READY after the short load/decode smoke."
    long_task = (
        args.long_task_line
        or (LONG_OUTPUT_TASK_LINE if args.long_output else "Return READY after the long-context DSA probe.")
    )
    long_answer_instruction = (
        args.long_answer_instruction
        or (LONG_OUTPUT_ANSWER_INSTRUCTION if args.long_output else SHORT_ANSWER_INSTRUCTION)
    )
    long_max_tokens = args.throughput_max_tokens if args.long_output else args.max_tokens
    long_min_completion_tokens = args.min_completion_tokens if args.long_output else 0
    long_progress_poll_interval = args.progress_poll_interval if args.long_output else 0
    long_metrics = args.metrics or args.long_output
    long_trace_logs = args.trace_logs or args.long_output
    scaling_task = "Return READY for the KV-length scaling checkpoint."

    primary_shard = Path(inventory["primary_shard"]) if inventory["primary_shard"] else args.model_dir
    log_dir = args.output.parent / "logs"

    stages: list[dict[str, Any]] = [
        {
            "name": "shard_integrity_inventory",
            "kind": "inventory",
            "status": inventory["status"],
            "hashing": "disabled",
            "summary": {
                "required_non_cache_shards": REQUIRED_NON_CACHE_SHARDS,
                "non_cache_shard_count": inventory["non_cache_shard_count"],
                "total_shard_bytes": inventory["total_shard_bytes"],
            },
            "blockers": inventory["blocker_files"],
            "stale_cache_markers": inventory.get("stale_cache_marker_files", []),
        },
        {
            "name": "short_load_decode_smoke",
            "kind": "load_decode_smoke",
            "status": "ready" if inventory["status"] == "ready" else "blocked",
            "prompt": _prompt_spec(
                "short_load_decode_smoke",
                args.short_context,
                short_task,
                args.min_prompt_tokens,
                args.prompt_context_guard_tokens,
            ),
            "server": _server_spec(
                binary=binary,
                library_path=library_path,
                model_path=primary_shard,
                port=args.port_base + 1,
                context_length=args.short_context,
                threads=args.threads,
                ubatch=args.ubatch,
                indexer_top_k=args.indexer_top_k,
                trace_logs=args.trace_logs,
                metrics=args.metrics,
                log_file=log_dir / "short_load_decode_smoke.server.log" if args.trace_logs else None,
                extra_args=args.server_extra_arg,
            ),
            "request": _request_spec(
                request_endpoint=args.request_endpoint,
                max_tokens=args.max_tokens,
                seed=args.seed,
                temperature=args.temperature,
                timeout_s=args.request_timeout,
            ),
        },
        {
            "name": "long_context_dsa_probe",
            "kind": "long_context_probe",
            "status": "ready" if inventory["status"] == "ready" else "blocked",
            "prompt": _prompt_spec(
                "long_context_dsa_probe",
                args.long_context,
                long_task,
                args.min_prompt_tokens,
                args.prompt_context_guard_tokens,
                answer_instruction=long_answer_instruction,
                needle_text=args.long_needle_text,
                needle_depth=args.long_needle_depth,
                expected_substring=args.long_expected_substring,
            ),
            "server": _server_spec(
                binary=binary,
                library_path=library_path,
                model_path=primary_shard,
                port=args.port_base + 2,
                context_length=args.long_context,
                threads=args.threads,
                ubatch=args.ubatch,
                indexer_top_k=args.indexer_top_k,
                trace_logs=long_trace_logs,
                metrics=long_metrics,
                log_file=log_dir / "long_context_dsa_probe.server.log" if long_trace_logs else None,
                extra_args=args.server_extra_arg,
            ),
            "request": _request_spec(
                request_endpoint=args.request_endpoint,
                max_tokens=long_max_tokens,
                seed=args.seed,
                temperature=args.temperature,
                timeout_s=args.request_timeout,
                min_completion_tokens=long_min_completion_tokens,
                progress_poll_interval_s=long_progress_poll_interval,
                purpose="coherence_plus_throughput" if args.long_output else "coherence",
                stream=args.long_output and args.request_endpoint == "chat",
            ),
        },
        {
            "name": "kv_length_scaling",
            "kind": "kv_length_scaling",
            "status": "ready" if inventory["status"] == "ready" else "blocked",
            "fixed_indexer_top_k": args.indexer_top_k,
            "series": [
                {
                    "context_length": context_length,
                    "prompt": _prompt_spec(
                        "kv_length_scaling",
                        context_length,
                        scaling_task,
                        args.min_prompt_tokens,
                        args.prompt_context_guard_tokens,
                    ),
                    "server": _server_spec(
                        binary=binary,
                        library_path=library_path,
                        model_path=primary_shard,
                        port=args.port_base + 10 + idx,
                        context_length=context_length,
                        threads=args.threads,
                        ubatch=args.ubatch,
                        indexer_top_k=args.indexer_top_k,
                        trace_logs=args.trace_logs,
                        metrics=args.metrics,
                        log_file=log_dir / f"kv_length_scaling_{context_length}.server.log" if args.trace_logs else None,
                        extra_args=args.server_extra_arg,
                    ),
                    "request": _request_spec(
                        request_endpoint=args.request_endpoint,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                        temperature=args.temperature,
                        timeout_s=args.request_timeout,
                    ),
                }
                for idx, context_length in enumerate(args.kv_contexts)
            ],
        },
    ]

    plan = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "model_dir": str(args.model_dir.resolve()),
        "model_path": str(primary_shard),
        "binary": str(binary),
        "library_path": str(library_path),
        "execution_allowed": inventory["status"] == "ready",
        "refusal_reasons": inventory["refusal_reasons"],
        "inventory": inventory,
        "selected_stages": sorted(selected_stage_names(args)) if selected_stage_names(args) else "all",
        "artifact_dir": str(args.output.parent / "artifacts"),
        "stages": stages,
        "execution": None,
    }
    return plan


def write_plan(output_path: Path, plan: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_canonical_json(plan) + "\n", encoding="utf-8")


def wait_for_health(port: int, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip().lower()
                if "ok" in body:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server on port {port} did not become healthy within {timeout_s}s")


def build_request_payload(
    request_endpoint: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    if request_endpoint == "chat":
        payload: dict[str, Any] = {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload
    if request_endpoint == "v1_completions":
        return {
            "model": "auto",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
    if request_endpoint == "completion":
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
    raise ValueError(f"unsupported request endpoint: {request_endpoint}")


def build_request_url(port: int, request_endpoint: str) -> str:
    return f"http://127.0.0.1:{port}{REQUEST_ENDPOINT_PATHS[request_endpoint]}"


def call_completion(
    port: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: int,
    request_endpoint: str = "chat",
) -> dict[str, Any]:
    payload = build_request_payload(
        request_endpoint,
        prompt,
        max_tokens,
        temperature,
        seed,
    )
    req = urllib.request.Request(
        build_request_url(port, request_endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _chat_stream_payload(prompt: str, max_tokens: int, temperature: float, seed: int) -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _delta_text(delta: dict[str, Any], key: str) -> str:
    value = delta.get(key)
    return value if isinstance(value, str) else ""


def call_completion_streaming(
    port: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = _chat_stream_payload(prompt, max_tokens, temperature, seed)
    req = urllib.request.Request(
        build_request_url(port, "chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    parse_errors: list[str] = []
    chunk_count = 0
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    started_at = datetime.now(timezone.utc)
    first_chunk_at: str | None = None
    completed_at: str | None = None

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                completed_at = datetime.now(timezone.utc).isoformat()
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"{exc}: {data[:160]}")
                continue

            chunk_count += 1
            if first_chunk_at is None:
                first_chunk_at = datetime.now(timezone.utc).isoformat()
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event.get("timings"), dict):
                timings = event["timings"]

            choices = event.get("choices") or []
            first = choices[0] if choices else {}
            if isinstance(first, dict) and first.get("finish_reason") is not None:
                finish_reason = first.get("finish_reason")
            delta = first.get("delta") if isinstance(first, dict) else {}
            if isinstance(delta, dict):
                content_delta = _delta_text(delta, "content")
                reasoning_delta = _delta_text(delta, "reasoning_content")
                if content_delta:
                    content_parts.append(content_delta)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)

            if progress_callback is not None:
                progress_callback(
                    {
                        "sampled_at": datetime.now(timezone.utc).isoformat(),
                        "status": "stream_chunk",
                        "chunk_count": chunk_count,
                        "content_char_count": sum(len(part) for part in content_parts),
                        "reasoning_char_count": sum(len(part) for part in reasoning_parts),
                    }
                )

    if completed_at is None:
        completed_at = datetime.now(timezone.utc).isoformat()

    return {
        "choices": [
            {
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts) or None,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "timings": timings,
        "streaming": {
            "enabled": True,
            "chunk_count": chunk_count,
            "started_at": started_at.isoformat(),
            "first_chunk_at": first_chunk_at,
            "completed_at": completed_at,
            "parse_errors_tail": parse_errors[-10:],
        },
    }


def count_prompt_tokens(port: int, prompt: str, timeout_s: int) -> int:
    payload = {
        "content": prompt,
        "add_special": True,
        "parse_special": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    response = json.loads(raw)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"/tokenize response missing tokens array: {response!r}")
    return len(tokens)


def launch_server(command: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )


def terminate_server(proc: subprocess.Popen[str]) -> None:
    pid = proc.pid
    if pid is None:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = None

    def send(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return

    if proc.poll() is None:
        send(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            proc.wait(timeout=20)

    ps = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,stat="],
        check=False,
        capture_output=True,
        text=True,
    )
    if ps.returncode == 0:
        fields = ps.stdout.strip().split()
        if len(fields) >= 2 and fields[0] == str(pid) and "Z" in fields[1]:
            return
    if ps.returncode == 0 and ps.stdout.strip().split()[0:1] == [str(pid)]:
        raise RuntimeError(f"server pid {pid} still visible after stop")


def _response_summary(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None
    content = _response_completion_text(response)
    return {
        "request_error": response.get("request_error"),
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "streaming": response.get("streaming", {}),
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
        "content_preview": content[:200],
        "reasoning_preview": reasoning_content[:200] if isinstance(reasoning_content, str) else None,
    }


def _response_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    parts: list[str] = []
    if isinstance(message, dict):
        reasoning_content = message.get("reasoning_content")
        content = message.get("content")
        if isinstance(reasoning_content, str):
            parts.append(reasoning_content)
        if isinstance(content, str):
            parts.append(content)
    text = first.get("text") if isinstance(first, dict) else None
    if isinstance(text, str):
        parts.append(text)
    raw_content = response.get("content")
    if isinstance(raw_content, str):
        parts.append(raw_content)
    return "".join(parts)


def _request_error_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, urllib.error.HTTPError):
        payload["http_code"] = exc.code
        payload["http_reason"] = exc.reason
        try:
            payload["body_preview"] = exc.read().decode("utf-8", errors="replace")[:1000]
        except Exception as body_exc:  # pragma: no cover - defensive only
            payload["body_read_error"] = f"{type(body_exc).__name__}: {body_exc}"
    return payload


def _stage_artifact_stem(stage: dict[str, Any]) -> str:
    return f"{stage['name']}_{stage['server']['context_length']}"


def write_stage_artifacts(
    artifact_dir: Path,
    stage: dict[str, Any],
    prompt: str,
    request_url: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = _stage_artifact_stem(stage)
    prompt_path = artifact_dir / f"{stem}_prompt.txt"
    request_path = artifact_dir / f"{stem}_request.json"
    response_path = artifact_dir / f"{stem}_response.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    request_path.write_text(
        _canonical_json(
            {
                "endpoint": stage["request"].get("endpoint"),
                "request_endpoint": stage["request"].get("request_endpoint", "chat"),
                "url": request_url,
                "payload": request_payload,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response_path.write_text(_canonical_json(response) + "\n", encoding="utf-8")
    return {
        "prompt": str(prompt_path),
        "request": str(request_path),
        "response": str(response_path),
    }


def build_prompt_with_token_floor(
    *,
    task_line: str,
    context_length: int,
    min_prompt_tokens: int,
    max_completion_tokens: int,
    prompt_context_guard_tokens: int,
    token_counter: Callable[[str], int],
    answer_instruction: str = SHORT_ANSWER_INSTRUCTION,
    needle_text: str | None = None,
    needle_depth: float = 0.5,
) -> dict[str, Any]:
    max_prompt_tokens = context_length - max_completion_tokens - prompt_context_guard_tokens
    if max_prompt_tokens <= 0:
        raise ValueError(
            "prompt token budget is non-positive: "
            f"context_length={context_length}, max_completion_tokens={max_completion_tokens}, "
            f"prompt_context_guard_tokens={prompt_context_guard_tokens}"
        )
    if min_prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"min_prompt_tokens={min_prompt_tokens} exceeds safe prompt budget "
            f"{max_prompt_tokens} for context_length={context_length}"
        )

    target_chars = _target_prompt_chars(context_length)
    prompt = _build_prompt_from_chars(
        task_line,
        target_chars,
        answer_instruction=answer_instruction,
        needle_text=needle_text,
        needle_depth=needle_depth,
    )
    token_count = token_counter(prompt)
    attempts = 1

    while min_prompt_tokens > 0 and token_count < min_prompt_tokens:
        if attempts >= PROMPT_TOKEN_FLOOR_MAX_ATTEMPTS:
            raise RuntimeError(
                f"could not reach min_prompt_tokens={min_prompt_tokens}; "
                f"last token_count={token_count}, target_chars={target_chars}"
            )
        observed_chars_per_token = max(1.0, len(prompt) / max(token_count, 1))
        deficit = min_prompt_tokens - token_count
        target_chars += int(deficit * observed_chars_per_token * 1.10) + len(FILLER_TEXT)
        prompt = _build_prompt_from_chars(
            task_line,
            target_chars,
            answer_instruction=answer_instruction,
            needle_text=needle_text,
            needle_depth=needle_depth,
        )
        token_count = token_counter(prompt)
        attempts += 1

    if token_count > max_prompt_tokens:
        raise ValueError(
            f"prompt token count {token_count} exceeds safe prompt budget "
            f"{max_prompt_tokens} for context_length={context_length}"
        )

    return {
        "prompt": prompt,
        "prompt_token_count": token_count,
        "prompt_char_count": len(prompt),
        "prompt_token_min": min_prompt_tokens,
        "prompt_token_max": max_prompt_tokens,
        "prompt_token_adjustment_attempts": attempts,
    }


def _summarize_metrics_text(text: str) -> dict[str, Any]:
    patterns = ("tokens", "prompt", "eval", "slot", "request", "latency", "processing")
    matches = [
        line
        for line in text.splitlines()
        if any(pattern in line.lower() for pattern in patterns)
    ]
    return {
        "bytes": len(text.encode("utf-8", errors="replace")),
        "line_count": text.count("\n") + (1 if text else 0),
        "matched_line_count": len(matches),
        "matched_lines_tail": matches[-40:],
    }


def fetch_metrics(port: int, timeout_s: int = 5) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return _summarize_metrics_text(text)


def collect_progress_metrics(
    *,
    port: int,
    interval_s: int,
    stop_event: threading.Event,
    samples: list[dict[str, Any]],
    max_samples: int = 80,
) -> None:
    while not stop_event.wait(interval_s):
        sample: dict[str, Any] = {"sampled_at": datetime.now(timezone.utc).isoformat()}
        try:
            sample.update({"status": "ok", **fetch_metrics(port)})
        except Exception as exc:
            sample.update({"status": "error", "error": str(exc)})
        samples.append(sample)
        if len(samples) > max_samples:
            del samples[: len(samples) - max_samples]


def summarize_server_log(log_file: str | None) -> dict[str, Any]:
    if not log_file:
        return {"status": "disabled", "path": None}
    path = Path(log_file)
    if not path.exists():
        return {"status": "missing", "path": str(path)}

    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        "lightning",
        "indexer",
        "fused_lid",
        "dsa",
        "unused tensor",
        "prompt eval time",
        "eval time",
        "graphs reused",
        "slot",
        "progress",
        "tokens",
        "n_layer",
        "top_k",
    )
    matches: list[str] = []
    prompt_eval_tokens: int | None = None
    prompt_eval_tps: float | None = None
    decode_tokens: int | None = None
    decode_tps: float | None = None
    max_decoded_checkpoint: int | None = None
    for line in text.splitlines():
        lower = line.lower()
        if any(pattern in lower for pattern in patterns):
            matches.append(line)
        decoded_match = re.search(r"n_decoded\s*=\s*(\d+)", line)
        if decoded_match:
            max_decoded_checkpoint = max(
                max_decoded_checkpoint or 0,
                int(decoded_match.group(1)),
            )
        prompt_match = re.search(
            r"prompt eval time\s*=.*?/\s*(\d+)\s+tokens.*?,\s*([0-9.]+)\s+tokens per second",
            line,
        )
        if prompt_match:
            prompt_eval_tokens = int(prompt_match.group(1))
            prompt_eval_tps = float(prompt_match.group(2))
        eval_match = re.search(
            r"\beval time\s*=.*?/\s*(\d+)\s+tokens.*?,\s*([0-9.]+)\s+tokens per second",
            line,
        )
        if eval_match and "prompt eval time" not in lower:
            decode_tokens = int(eval_match.group(1))
            decode_tps = float(eval_match.group(2))
    return {
        "status": "ok",
        "path": str(path),
        "bytes": path.stat().st_size,
        "line_count": text.count("\n") + (1 if text else 0),
        "matched_line_count": len(matches),
        "matched_lines_tail": matches[-80:],
        "prompt_eval_tokens": prompt_eval_tokens,
        "prompt_eval_tps": prompt_eval_tps,
        "decode_tokens": decode_tokens,
        "decode_tps": decode_tps,
        "max_decoded_checkpoint": max_decoded_checkpoint,
    }


def run_stage(stage: dict[str, Any], artifact_dir: Path | None = None) -> dict[str, Any]:
    server_command = stage["server"]["server_command"]
    log_file = stage["server"].get("log_file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.unlink(missing_ok=True)
    proc = launch_server(server_command)
    response: dict[str, Any] | None = None
    prompt_info: dict[str, Any] | None = None
    progress_samples: list[dict[str, Any]] = []
    stream_progress_samples: list[dict[str, Any]] = []
    stop_event = threading.Event()
    poll_thread: threading.Thread | None = None
    completion_tokens_from_tokenize: int | None = None
    request_error: dict[str, Any] | None = None
    request_endpoint = str(stage["request"].get("request_endpoint", "chat"))
    request_url = build_request_url(stage["server"]["port"], request_endpoint)
    request_payload: dict[str, Any] | None = None
    artifacts: dict[str, str] = {}
    try:
        wait_for_health(stage["server"]["port"])
        tokenize_timeout = max(60, min(stage["request"]["timeout_s"], 600))
        prompt_info = build_prompt_with_token_floor(
            task_line=stage["prompt"]["task_line"],
            context_length=stage["prompt"]["context_length"],
            min_prompt_tokens=int(stage["prompt"].get("min_prompt_tokens", 0)),
            max_completion_tokens=stage["request"]["max_tokens"],
            prompt_context_guard_tokens=int(stage["prompt"].get("prompt_context_guard_tokens", 0)),
            answer_instruction=str(stage["prompt"].get("answer_instruction", SHORT_ANSWER_INSTRUCTION)),
            needle_text=stage["prompt"].get("needle_text"),
            needle_depth=float(stage["prompt"].get("needle_depth", 0.5)),
            token_counter=lambda prompt: count_prompt_tokens(stage["server"]["port"], prompt, tokenize_timeout),
        )
        poll_interval = int(stage["request"].get("progress_poll_interval_s", 0) or 0)
        if stage["server"].get("metrics") and poll_interval > 0 and not stage["request"].get("stream"):
            poll_thread = threading.Thread(
                target=collect_progress_metrics,
                kwargs={
                    "port": stage["server"]["port"],
                    "interval_s": poll_interval,
                    "stop_event": stop_event,
                    "samples": progress_samples,
                },
                daemon=True,
            )
            poll_thread.start()
        try:
            if stage["request"].get("stream") and request_endpoint == "chat":
                request_payload = _chat_stream_payload(
                    prompt_info["prompt"],
                    stage["request"]["max_tokens"],
                    stage["request"]["temperature"],
                    stage["request"]["seed"],
                )
                response = call_completion_streaming(
                    stage["server"]["port"],
                    prompt_info["prompt"],
                    stage["request"]["max_tokens"],
                    stage["request"]["temperature"],
                    stage["request"]["seed"],
                    stage["request"]["timeout_s"],
                    progress_callback=lambda sample: stream_progress_samples.append(sample),
                )
            else:
                request_payload = build_request_payload(
                    request_endpoint,
                    prompt_info["prompt"],
                    stage["request"]["max_tokens"],
                    stage["request"]["temperature"],
                    stage["request"]["seed"],
                )
                response = call_completion(
                    stage["server"]["port"],
                    prompt_info["prompt"],
                    stage["request"]["max_tokens"],
                    stage["request"]["temperature"],
                    stage["request"]["seed"],
                    stage["request"]["timeout_s"],
                    request_endpoint,
                )
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if request_payload is None:
                request_payload = build_request_payload(
                    request_endpoint,
                    prompt_info["prompt"],
                    stage["request"]["max_tokens"],
                    stage["request"]["temperature"],
                    stage["request"]["seed"],
                    stream=bool(stage["request"].get("stream") and request_endpoint == "chat"),
                )
            request_error = _request_error_payload(exc)
            response = {
                "request_error": request_error,
                "usage": {},
                "timings": {},
                "choices": [],
            }
        if artifact_dir is not None:
            assert response is not None
            assert request_payload is not None
            artifacts = write_stage_artifacts(
                artifact_dir,
                stage,
                prompt_info["prompt"],
                request_url,
                request_payload,
                response,
            )
        min_completion_tokens = int(stage["request"].get("min_completion_tokens", 0) or 0)
        response_usage = response.get("usage", {}) if isinstance(response, dict) else {}
        response_completion_tokens = (
            response_usage.get("completion_tokens") if isinstance(response_usage, dict) else None
        )
        if min_completion_tokens > 0 and not isinstance(response_completion_tokens, int):
            completion_text = _response_completion_text(response)
            completion_tokens_from_tokenize = (
                count_prompt_tokens(stage["server"]["port"], completion_text, tokenize_timeout)
                if completion_text
                else 0
            )
        stop_event.set()
        if poll_thread is not None:
            poll_thread.join(timeout=10)
    finally:
        stop_event.set()
        if poll_thread is not None:
            poll_thread.join(timeout=10)
        terminate_server(proc)

    assert response is not None
    assert prompt_info is not None
    summary = _response_summary(response)
    server_log = summarize_server_log(stage["server"].get("log_file"))
    usage = summary.get("usage", {})
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    completion_token_count_source = "usage"
    if not isinstance(completion_tokens, int) and isinstance(server_log.get("decode_tokens"), int):
        completion_tokens = int(server_log["decode_tokens"])
        completion_token_count_source = "server_log_eval_time"
    if not isinstance(completion_tokens, int) and isinstance(server_log.get("max_decoded_checkpoint"), int):
        completion_tokens = int(server_log["max_decoded_checkpoint"])
        completion_token_count_source = "server_log_decoded_checkpoint"
    if not isinstance(completion_tokens, int) and completion_tokens_from_tokenize is not None:
        completion_tokens = completion_tokens_from_tokenize
        completion_token_count_source = "tokenize_completion_text"
    min_completion_tokens = int(stage["request"].get("min_completion_tokens", 0) or 0)
    completion_text = _response_completion_text(response)
    expected_substring = stage["prompt"].get("expected_substring")
    expected_substring_passed = (
        None if not expected_substring else str(expected_substring) in completion_text
    )
    completion_token_min_passed = (
        min_completion_tokens == 0
        or (isinstance(completion_tokens, int) and completion_tokens >= min_completion_tokens)
    )
    acceptance_passed = completion_token_min_passed and expected_substring_passed is not False
    status = "ok" if acceptance_passed else "failed_acceptance"
    if completion_token_min_passed is False:
        status = "failed_completion_floor"
    if request_error is not None:
        status = "failed_request"
    return {
        "name": stage["name"],
        "status": status,
        "port": stage["server"]["port"],
        "context_length": stage["server"]["context_length"],
        "prompt_kind": stage["prompt"]["kind"],
        "request_purpose": stage["request"].get("purpose", "coherence"),
        "prompt_char_count": prompt_info["prompt_char_count"],
        "prompt_token_count": prompt_info["prompt_token_count"],
        "prompt_token_min": prompt_info["prompt_token_min"],
        "prompt_token_max": prompt_info["prompt_token_max"],
        "prompt_token_adjustment_attempts": prompt_info["prompt_token_adjustment_attempts"],
        "completion_token_min": min_completion_tokens,
        "completion_token_count": completion_tokens,
        "completion_token_count_source": completion_token_count_source,
        "completion_token_min_passed": completion_token_min_passed,
        "expected_substring": expected_substring,
        "expected_substring_passed": expected_substring_passed,
        "progress_metrics_samples": progress_samples,
        "stream_progress_samples_tail": stream_progress_samples[-120:],
        "artifacts": artifacts,
        **summary,
        "server_log": server_log,
    }


def _stage_selected(stage_name: str, selected: set[str] | None) -> bool:
    return selected is None or stage_name in selected


def _execution_result_failed(result: dict[str, Any]) -> bool:
    status = result.get("status")
    if isinstance(status, str) and status.startswith("failed"):
        return True
    return any(
        _execution_result_failed(item)
        for item in result.get("series", [])
        if isinstance(item, dict)
    )


def run_execution(plan: dict[str, Any]) -> dict[str, Any]:
    selected_payload = plan.get("selected_stages")
    selected = None if selected_payload == "all" else set(selected_payload)
    artifact_dir = Path(plan["artifact_dir"]) if plan.get("artifact_dir") else None
    results: list[dict[str, Any]] = []
    for stage in plan["stages"]:
        if not _stage_selected(stage["name"], selected):
            results.append({"name": stage["name"], "status": "skipped", "reason": "not selected"})
            continue
        if stage["status"] != "ready":
            results.append({"name": stage["name"], "status": "skipped", "reason": "preflight blocked"})
            continue
        if stage["kind"] == "inventory":
            results.append({"name": stage["name"], "status": "checked"})
            continue
        if stage["kind"] == "kv_length_scaling":
            series_results = []
            for entry in stage["series"]:
                stage_payload = {
                    "name": stage["name"],
                    "prompt": entry["prompt"],
                    "server": entry["server"],
                    "request": entry["request"],
                }
                if artifact_dir is None:
                    series_results.append(run_stage(stage_payload))
                else:
                    series_results.append(run_stage(stage_payload, artifact_dir=artifact_dir))
            results.append({
                "name": stage["name"],
                "status": "ok",
                "fixed_indexer_top_k": stage["fixed_indexer_top_k"],
                "series": series_results,
            })
            continue
        if artifact_dir is None:
            results.append(run_stage(stage))
        else:
            results.append(run_stage(stage, artifact_dir=artifact_dir))
    return {
        "status": "failed" if any(_execution_result_failed(result) for result in results) else "ok",
        "stages": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM-5.2 DSA probe runner")
    parser.add_argument("--execute", action="store_true", help="Execute the staged probe after preflight")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH_ROOT / "data" / "glm52_dsa_probe" / _utc_timestamp() / "plan.json",
        help="Plan JSON output path",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="GLM shard directory")
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help="llama-server binary (experimental v7 default)",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        default=None,
        help="Pinned LD_LIBRARY_PATH directory (defaults to the binary directory)",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="llama-server threads")
    parser.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, help="llama-server ubatch")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Request max_tokens")
    parser.add_argument(
        "--long-output",
        action="store_true",
        help="Use one long-context request that combines READY coherence with enough generation for throughput evidence",
    )
    parser.add_argument(
        "--throughput-max-tokens",
        type=int,
        default=DEFAULT_THROUGHPUT_MAX_TOKENS,
        help="max_tokens for --long-output long-context throughput probes",
    )
    parser.add_argument(
        "--min-completion-tokens",
        type=int,
        default=DEFAULT_MIN_COMPLETION_TOKENS,
        help="Minimum completion tokens required for --long-output throughput evidence",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Request seed")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Request temperature")
    parser.add_argument(
        "--request-endpoint",
        choices=REQUEST_ENDPOINTS,
        default="chat",
        help="Completion request endpoint: chat=/v1/chat/completions, v1_completions=/v1/completions, completion=/completion",
    )
    parser.add_argument("--indexer-top-k", type=int, default=DEFAULT_INDEXER_TOP_K, help="Fixed DSA indexer_top_k")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP request timeout in seconds for long CPU-prefill probes",
    )
    parser.add_argument(
        "--trace-logs",
        action="store_true",
        help="Keep llama-server trace logs instead of passing --log-disable",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable llama-server --metrics in staged server commands",
    )
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Extra llama-server argument appended to each staged server command. Repeat once per argument.",
    )
    parser.add_argument(
        "--progress-poll-interval",
        type=int,
        default=DEFAULT_PROGRESS_POLL_INTERVAL,
        help="Seconds between /metrics samples during --long-output completion calls",
    )
    parser.add_argument(
        "--long-task-line",
        default=None,
        help="Override the long-context task line; used for explicit needle/coherence probes.",
    )
    parser.add_argument(
        "--long-answer-instruction",
        default=None,
        help="Override the long-context answer instruction.",
    )
    parser.add_argument(
        "--long-needle-text",
        default=None,
        help="Needle text to insert into the long-context filler.",
    )
    parser.add_argument(
        "--long-needle-depth",
        type=float,
        default=0.5,
        help="Approximate insertion depth for --long-needle-text, in [0.0, 1.0].",
    )
    parser.add_argument(
        "--long-expected-substring",
        default=None,
        help="Substring that must appear in the long-context completion for acceptance.",
    )
    parser.add_argument(
        "--only-stage",
        action="append",
        choices=[
            "shard_integrity_inventory",
            "short_load_decode_smoke",
            "long_context_dsa_probe",
            "kv_length_scaling",
        ],
        help="Execute only the named stage. May be repeated. Dry-run plans still include all stages.",
    )
    parser.add_argument(
        "--short-context",
        type=int,
        default=DEFAULT_SHORT_CONTEXT,
        help="Context length for the short load/decode smoke",
    )
    parser.add_argument(
        "--long-context",
        type=int,
        default=DEFAULT_LONG_CONTEXT,
        help="Context length for the long-context DSA probe",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=DEFAULT_MIN_PROMPT_TOKENS,
        help="Minimum actual prompt tokens verified through the live /tokenize endpoint before completion",
    )
    parser.add_argument(
        "--prompt-context-guard-tokens",
        type=int,
        default=DEFAULT_PROMPT_CONTEXT_GUARD_TOKENS,
        help="Prompt-token budget held back for chat template overhead and generated tokens",
    )
    parser.add_argument(
        "--kv-contexts",
        nargs="+",
        type=int,
        default=list(DEFAULT_KV_CONTEXTS),
        help="Context lengths for the KV-length scaling series",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=DEFAULT_PORT_BASE,
        help="Base port for staged server commands",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = resolve_binary(args.binary)
    library_path = resolve_library_path(binary, args.library_path)
    inventory = collect_inventory(args.model_dir)
    plan = build_plan(args, inventory, binary, library_path)
    write_plan(args.output, plan)

    if not args.execute:
        print(f"[glm52] mode: dry-run")
        print(f"[glm52] plan: {args.output}")
        if plan["execution_allowed"]:
            print("[glm52] preflight: ready for execute mode")
        else:
            print("[glm52] preflight: blocked")
            for reason in plan["refusal_reasons"]:
                print(f"[glm52]   - {reason}")
        return 0

    if not plan["execution_allowed"]:
        print("[glm52] refusing execute mode: preflight blocked", file=sys.stderr)
        for reason in plan["refusal_reasons"]:
            print(f"[glm52]   - {reason}", file=sys.stderr)
        return 3

    execution = run_execution(plan)
    plan["execution"] = execution
    write_plan(args.output, plan)
    print(f"[glm52] execute complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
