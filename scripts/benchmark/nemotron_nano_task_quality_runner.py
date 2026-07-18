#!/usr/bin/env python3
"""Bounded dry-run-first MI210 quality gate for Nemotron-Nano Q8.

This runner launches exactly one experimental llama-server instance on MI210,
queries the corrected chat endpoint, scores a small deterministic task slice,
and proves post-run cleanup against the preexisting llama-server PID set.

The pass/fail target here is task quality only. Any throughput numbers gathered
while the concurrent CPU-only GLM gate is active are treated as contaminated
observations and are not decision-making evidence.
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
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPERIMENTAL_BIN_DIR = EXPERIMENTAL_ROOT / "build-hip" / "bin"
SERVER_BIN = EXPERIMENTAL_BIN_DIR / "llama-server"
SERVER_LIB_DIR = EXPERIMENTAL_BIN_DIR
MODEL_PATH = Path(
    "/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/"
    "nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf"
)
GLM_PATTERN = "scripts/benchmark/glm52_dsa_probe_runner.py"

DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "nemotron_nano_task_quality"
    / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 8192
DEFAULT_PORT = 19122
DEFAULT_REQUEST_TIMEOUT_S = 240
DEFAULT_STARTUP_TIMEOUT_S = 240
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 7
DEFAULT_MAX_TOKENS = 160
DEFAULT_PROTOCOL = "deepseek"
PROTOCOL_SPECS = {
    "deepseek": {"reasoning_format": "deepseek", "reasoning": "off"},
    "none": {"reasoning_format": "none", "reasoning": "off"},
    "deepseek_legacy": {"reasoning_format": "deepseek-legacy", "reasoning": "off"},
}
SCORE_SOURCES = ("content", "reasoning_content", "content_or_reasoning")
STRICT_SYSTEM_PROMPT = "Answer only with the requested final answer. Do not emit reasoning."

SANITIZED_ENV = {
    "HOME": "/tmp",
    "LD_LIBRARY_PATH": str(SERVER_LIB_DIR),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    scorer: str
    expected: Any
    max_tokens: int


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="exact_ok",
        prompt="Return exactly: ok",
        scorer="exact",
        expected="ok",
        max_tokens=24,
    ),
    TaskSpec(
        task_id="strict_json",
        prompt=(
            'Return exactly this minified JSON and nothing else: '
            '{"status":"ok","model":"nemotron"}'
        ),
        scorer="json_exact",
        expected={"status": "ok", "model": "nemotron"},
        max_tokens=48,
    ),
    TaskSpec(
        task_id="simple_math",
        prompt="Answer with only the integer result: 37 + 58",
        scorer="exact",
        expected="95",
        max_tokens=24,
    ),
    TaskSpec(
        task_id="five_lowercase_words",
        prompt=(
            "In exactly five lowercase words, describe why held-out tests matter. "
            "Return only the five words."
        ),
        scorer="lowercase_word_count",
        expected=5,
        max_tokens=48,
    ),
    TaskSpec(
        task_id="long_context_needle",
        prompt=(
            "Read the full passage, then answer with only the project code.\n\n"
            "Status notes: alpha reports remain internal. "
            "Audit reviewers asked for a longer context check to confirm the chat "
            "endpoint still returns the answer in message.content without mixing in "
            "reasoning-only fields. "
            "The archive repeats the same operational warning for scanning depth. "
            "Line 1: ignore incidental numbers like 104, 205, or 999.\n"
            "Line 2: only one project code is authoritative.\n"
            "Line 3: the project code appears once and should be copied exactly.\n"
            "Line 4: filler text discusses ports, GPUs, and cache settings.\n"
            "Line 5: project summaries remain provisional pending the quality gate.\n"
            "Line 6: the canonical project code is NN-4242-DELTA.\n"
            "Line 7: later lines mention unrelated strings such as AA-1000 or test-draft.\n"
            "Line 8: answer with the code only.\n\n"
            "Question: what is the project code?"
        ),
        scorer="exact",
        expected="NN-4242-DELTA",
        max_tokens=48,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nemotron-Nano Q8 MI210 quality gate runner")
    parser.add_argument("--execute", action="store_true", help="Run the live gate after writing the plan")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument(
        "--ignore-task-token-caps",
        action="store_true",
        help="Use --max-tokens for every task instead of the conservative per-task caps.",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Omit the strict final-answer system message; useful for protocol-channel isolation.",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=sorted(PROTOCOL_SPECS),
        help="Protocol arm to run; repeat for multiple arms. Defaults to deepseek.",
    )
    parser.add_argument(
        "--protocol-matrix",
        action="store_true",
        help="Run all known Nemotron protocol arms sequentially.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validated_server_bin() -> Path:
    resolved = SERVER_BIN.resolve()
    production = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server").resolve()
    if resolved == production:
        raise RuntimeError("refusing production v6 llama-server binary")
    if EXPERIMENTAL_ROOT not in resolved.parents and resolved.parent != EXPERIMENTAL_BIN_DIR:
        raise RuntimeError(f"refusing non-experimental server binary: {resolved}")
    return resolved


def is_pid_alive(pid: int) -> bool:
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == str(pid)


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
        pid_text, args = parts
        argv0 = args.split(maxsplit=1)[0]
        if Path(argv0).name == "llama-server" and pid_text.isdigit():
            pids.append(int(pid_text))
    return sorted(pids)


def detect_glm_pids() -> list[int]:
    probe = subprocess.run(
        ["pgrep", "-af", GLM_PATTERN],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    if probe.returncode != 0:
        return pids
    for line in probe.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts and parts[0].isdigit():
            pids.append(int(parts[0]))
    return sorted(pids)


def detect_q8_kv_support() -> bool:
    probe = subprocess.run(
        [str(_validated_server_bin()), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=SANITIZED_ENV,
    )
    help_text = f"{probe.stdout}\n{probe.stderr}"
    return "-ctk" in help_text and "-ctv" in help_text


def pick_available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def selected_protocols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.protocol_matrix:
        return tuple(PROTOCOL_SPECS)
    if args.protocol:
        return tuple(dict.fromkeys(args.protocol))
    return (DEFAULT_PROTOCOL,)


def launch_argv(
    args: argparse.Namespace,
    port: int,
    q8_kv_supported: bool,
    protocol: str = DEFAULT_PROTOCOL,
) -> list[str]:
    spec = PROTOCOL_SPECS[protocol]
    argv = [
        "numactl",
        "--interleave=all",
        str(_validated_server_bin()),
        "-m",
        str(MODEL_PATH),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "ROCm0",
        "-ngl",
        "99",
        "-t",
        str(args.threads),
        "-c",
        str(args.context),
        "-fa",
        "on",
        "--metrics",
        "--temp",
        "0",
    ]
    reasoning_format = spec.get("reasoning_format")
    if reasoning_format:
        argv.extend(["--reasoning-format", reasoning_format])
    reasoning = spec.get("reasoning")
    if reasoning:
        argv.extend(["--reasoning", reasoning])
    if q8_kv_supported:
        argv.extend(["-ctk", "q8_0", "-ctv", "q8_0"])
    return argv


def launch_command_string(argv: list[str]) -> str:
    return " ".join(
        shlex.quote(part)
        for part in ["env", "-i", *[f"{key}={value}" for key, value in SANITIZED_ENV.items()], *argv]
    )


def task_payload(task: TaskSpec, args: argparse.Namespace) -> dict[str, Any]:
    max_tokens = args.max_tokens if args.ignore_task_token_caps else min(args.max_tokens, task.max_tokens)
    messages: list[dict[str, str]] = []
    if not args.no_system_prompt:
        messages.append(
            {
                "role": "system",
                "content": STRICT_SYSTEM_PROMPT,
            }
        )
    messages.append({"role": "user", "content": task.prompt})
    return {
        "model": "auto",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 1,
        "seed": args.seed,
        "stream": False,
    }


def parse_content_json(content: str) -> tuple[str, Any | None]:
    stripped = content.strip()
    if not stripped:
        return "empty", None
    try:
        return "strict", json.loads(stripped)
    except json.JSONDecodeError:
        pass
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            try:
                return "fenced", json.loads("\n".join(lines[1:-1]).strip())
            except json.JSONDecodeError:
                pass
    return "non_json", None


def normalize_text(text: str) -> str:
    return text.strip().replace("\u00a0", " ")


def extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def extract_message_field(response: dict[str, Any], field: str) -> str:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        return ""
    value = message.get(field)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in value
        )
    return ""


def score_sources(task: TaskSpec, content: str, reasoning_content: str | None) -> dict[str, dict[str, Any]]:
    reasoning_text = reasoning_content or ""
    source_values = {
        "content": content,
        "reasoning_content": reasoning_text,
        "content_or_reasoning": content if normalize_text(content) else reasoning_text,
    }
    return {source: score_task(task, value) for source, value in source_values.items()}


def score_task(task: TaskSpec, content: str) -> dict[str, Any]:
    normalized = normalize_text(content)
    if task.scorer == "exact":
        return {
            "passed": normalized == str(task.expected),
            "expected": task.expected,
            "observed": normalized,
        }
    if task.scorer == "json_exact":
        json_mode, parsed = parse_content_json(normalized)
        expected = task.expected
        return {
            "passed": parsed == expected,
            "expected": expected,
            "observed": parsed,
            "json_mode": json_mode,
        }
    if task.scorer == "lowercase_word_count":
        words = normalized.split()
        return {
            "passed": len(words) == int(task.expected) and all(re.fullmatch(r"[a-z]+", word) for word in words),
            "expected_word_count": task.expected,
            "observed_words": words,
        }
    raise ValueError(f"unknown scorer: {task.scorer}")


def wait_for_health(port: int, timeout_s: int, pid: int | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if pid is not None and not is_pid_alive(pid):
            raise RuntimeError(f"server pid {pid} exited before health check on port {port}")
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip().lower()
                if "ok" in body:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"server on port {port} did not become healthy within {timeout_s}s")


def query_chat(port: int, payload: dict[str, Any], timeout_s: int) -> tuple[dict[str, Any], str]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw), raw


def launch_server(argv: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=SANITIZED_ENV,
        )
    except Exception:
        log_handle.close()
        raise
    proc._nemotron_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


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
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)

    if proc.poll() is None:
        send(signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)

    if proc.poll() is None:
        raise RuntimeError(f"failed to terminate server pid {pid}")
    if is_pid_alive(pid):
        raise RuntimeError(f"server pid {pid} still appears alive after termination")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    glm_pids = detect_glm_pids()
    preexisting_pids = list_llama_server_pids()
    q8_kv_supported = detect_q8_kv_support()
    protocols = selected_protocols(args)
    servers: dict[str, dict[str, Any]] = {}
    for idx, protocol in enumerate(protocols):
        chosen_port = pick_available_port(args.port + idx)
        launch = launch_argv(args, chosen_port, q8_kv_supported, protocol)
        servers[protocol] = {
            "protocol": protocol,
            "reasoning_format": PROTOCOL_SPECS[protocol]["reasoning_format"],
            "reasoning": PROTOCOL_SPECS[protocol]["reasoning"],
            "port": chosen_port,
            "device": "ROCm0",
            "ngl": 99,
            "kv_cache": "q8_0/q8_0" if q8_kv_supported else "server_default",
            "q8_kv_supported": q8_kv_supported,
            "argv": launch,
            "command": launch_command_string(launch),
        }
    primary_server = servers[protocols[0]]
    return {
        "schema": "nemotron_nano_task_quality_plan.v1",
        "created_at": utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "classification": "quality-only protocol matrix; throughput is observational unless the host is otherwise quiet",
        "experimental_root": str(EXPERIMENTAL_ROOT),
        "server_bin": str(_validated_server_bin()),
        "model_path": str(MODEL_PATH),
        "request_endpoint": "/v1/chat/completions",
        "message_inspection": {
            "score_sources": list(SCORE_SOURCES),
            "primary_source": "content",
        },
        "prompt_policy": {
            "system_prompt": None if args.no_system_prompt else STRICT_SYSTEM_PROMPT,
            "no_system_prompt": args.no_system_prompt,
        },
        "server": primary_server,
        "servers": servers,
        "request": {
            "context": args.context,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "request_timeout_s": args.request_timeout,
            "startup_timeout_s": args.startup_timeout,
            "deterministic_sampling": {
                "temperature": args.temperature,
                "top_p": 1.0,
                "top_k": 1,
                "seed": args.seed,
            },
        },
        "task_slice": [dataclasses.asdict(task) for task in TASKS],
        "preexisting_server_pids": preexisting_pids,
        "concurrent_glm_probe_pids": glm_pids,
        "cleanup_expectation": {
            "allowed_pids_after_run": preexisting_pids,
            "note": "no new llama-server may remain after cleanup; concurrent GLM server PID is preserved if present",
        },
    }


def write_plan(output_dir: Path, plan: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "responses").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def verify_cleanup(allowed_pids: list[int]) -> dict[str, Any]:
    observed = list_llama_server_pids()
    extras = [pid for pid in observed if pid not in allowed_pids]
    missing = [pid for pid in allowed_pids if pid not in observed]
    return {
        "allowed_pids": allowed_pids,
        "observed_pids": observed,
        "extra_pids": extras,
        "missing_allowed_pids": missing,
        "passed": not extras,
    }


def run_execute(args: argparse.Namespace, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    for directory in (output_dir / "logs", output_dir / "responses", output_dir / "results"):
        directory.mkdir(parents=True, exist_ok=True)
    proc: subprocess.Popen[str] | None = None
    log_path = output_dir / "logs" / "nemotron_nano_q8_mi210.server.log"
    protocol_summaries: list[dict[str, Any]] = []
    cleanup_record: dict[str, Any] | None = None
    try:
        servers = plan.get("servers") or {plan.get("server", {}).get("protocol", DEFAULT_PROTOCOL): plan["server"]}
        for protocol, server in servers.items():
            records: list[dict[str, Any]] = []
            log_path = output_dir / "logs" / f"nemotron_nano_q8_mi210.{protocol}.server.log"
            proc = None
            try:
                proc = launch_server(server["argv"], log_path)
                wait_for_health(server["port"], args.startup_timeout, pid=proc.pid)
                response_dir = output_dir / "responses" / protocol
                response_dir.mkdir(parents=True, exist_ok=True)
                for task in TASKS:
                    payload = task_payload(task, args)
                    response, raw = query_chat(server["port"], payload, args.request_timeout)
                    raw_path = response_dir / f"{task.task_id}.raw.json"
                    raw_path.write_text(raw, encoding="utf-8")
                    content = extract_message_content(response)
                    reasoning_content = extract_message_field(response, "reasoning_content")
                    scores_by_source = score_sources(task, content, reasoning_content)
                    choices = response.get("choices") or []
                    first = choices[0] if choices else {}
                    record = {
                        "task_id": task.task_id,
                        "prompt": task.prompt,
                        "response_path": str(raw_path),
                        "content": content,
                        "score": scores_by_source["content"],
                        "scores_by_source": scores_by_source,
                        "usage": response.get("usage"),
                        "timings": response.get("timings"),
                        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
                        "reasoning_content_observed": bool(reasoning_content),
                        "reasoning_content_preview": reasoning_content[:160] if reasoning_content else None,
                    }
                    records.append(record)
            finally:
                if proc is not None:
                    try:
                        terminate_server(proc)
                    finally:
                        log_handle = getattr(proc, "_nemotron_log_handle", None)
                        if log_handle is not None:
                            log_handle.close()
            passed_by_source = {
                source: sum(1 for record in records if record["scores_by_source"][source]["passed"])
                for source in SCORE_SOURCES
            }
            protocol_summaries.append(
                {
                    "protocol": protocol,
                    "server": {
                        "port": server["port"],
                        "reasoning_format": server.get("reasoning_format"),
                        "reasoning": server.get("reasoning"),
                        "log_path": str(log_path),
                    },
                    "passed_by_source": passed_by_source,
                    "quality_gate_passed_by_source": {
                        source: passed == len(TASKS)
                        for source, passed in passed_by_source.items()
                    },
                    "results": records,
                }
            )
    finally:
        cleanup_record = verify_cleanup(plan["cleanup_expectation"]["allowed_pids_after_run"])
        if not cleanup_record["passed"]:
            raise RuntimeError(f"cleanup check failed: {cleanup_record}")

    primary_protocol = protocol_summaries[0] if protocol_summaries else {"results": [], "passed_by_source": {"content": 0}}
    primary_records = primary_protocol["results"]
    passed = primary_protocol["passed_by_source"]["content"]
    decode_rates = [
        float(record["timings"]["predicted_per_second"])
        for protocol in protocol_summaries
        for record in protocol["results"]
        if isinstance(record.get("timings"), dict)
        and isinstance(record["timings"].get("predicted_per_second"), (int, float))
    ]
    prompt_rates = [
        float(record["timings"]["prompt_per_second"])
        for protocol in protocol_summaries
        for record in protocol["results"]
        if isinstance(record.get("timings"), dict)
        and isinstance(record["timings"].get("prompt_per_second"), (int, float))
    ]
    throughput_contaminated = bool(plan["preexisting_server_pids"] or plan["concurrent_glm_probe_pids"])
    summary = {
        "schema": "nemotron_nano_task_quality_summary.v1",
        "created_at": utc_now(),
        "mode": "execute",
        "quality_gate_passed": passed == len(TASKS),
        "classification": plan["classification"],
        "model_path": str(MODEL_PATH),
        "server_pid": None,
        "server_port": plan["server"]["port"],
        "q8_kv_supported": plan["server"]["q8_kv_supported"],
        "prompt_policy": plan["prompt_policy"],
        "throughput_observation": {
            "contaminated": throughput_contaminated,
            "decision_use": "forbidden_contaminated" if throughput_contaminated else "observation_only_quality_gate",
            "mean_decode_tps": sum(decode_rates) / len(decode_rates) if decode_rates else None,
            "mean_prompt_tps": sum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
        },
        "passed": passed,
        "total": len(TASKS),
        "results": primary_records,
        "protocol_summaries": protocol_summaries,
        "best_protocol_source": max(
            (
                (protocol["passed_by_source"][source], protocol["protocol"], source)
                for protocol in protocol_summaries
                for source in SCORE_SOURCES
            ),
            default=(0, None, None),
        ),
        "cleanup": cleanup_record,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    write_plan(args.output_dir, plan)

    print("Nemotron-Nano Q8 MI210 quality gate")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"server_bin: {plan['server_bin']}")
    print(f"model_path: {plan['model_path']}")
    print(f"preexisting_server_pids: {plan['preexisting_server_pids']}")
    print(f"concurrent_glm_probe_pids: {plan['concurrent_glm_probe_pids']}")
    print(f"selected_port: {plan['server']['port']}")
    print(f"q8_kv_supported: {plan['server']['q8_kv_supported']}")

    if not args.execute:
        print(f"Plan written to {args.output_dir / 'plan.json'}")
        return 0

    try:
        summary = run_execute(args, args.output_dir, plan)
    except Exception as exc:
        print(f"Execute mode failed: {exc}", file=sys.stderr)
        return 1

    print(f"quality_gate_passed: {summary['quality_gate_passed']} ({summary['passed']}/{summary['total']})")
    print(f"cleanup_passed: {summary['cleanup']['passed']}")
    print(f"Summary written to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
