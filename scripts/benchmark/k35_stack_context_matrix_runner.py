#!/usr/bin/env python3
"""K35 optimized stack throughput-vs-context runner.

This is a dry-run-first harness for the v7 release artifact required by K35.
It measures one stack-role scenario at a time, records exact launch commands,
server/version/host state, raw responses, timing summaries, and cleanup proof.

The scenario list intentionally starts with only configurations that already
have enough prior evidence to call them "optimized" rather than baseline:

* frontdoor: MI210-resident Qwen3.6-35B no-spec, because Stage-2 showed it was
  faster than native MTP or external draft-tree on the measured workload.
* worker_general: CPU composed ngram-mod,draft-mtp with the live Gemma4
  assistant head, q8 KV, reasoning off, and production-shaped thread flags.
* architect_general: CPU native NEXTN/draft-mtp, same-file draft head, q4/f16 KV,
  and request-level enable_thinking=false so content is measured.
* ingest_long_context: CPU Qwen3-Next default-expert route with speculation
  disabled. Historical MoE4 registry entries are treated as stale until
  re-approved.

Add vision scenarios only after their fastest safe configs are settled and
documented.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPERIMENTAL_BIN_DIR = EXPERIMENTAL_ROOT / "build-hip" / "bin"
DEFAULT_BINARY = EXPERIMENTAL_BIN_DIR / "llama-server"
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "k35_stack_context_matrix"
    / f"k35_stack_context_matrix_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_CONTEXTS = (2048, 8192, 32768)
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT_S = 900
DEFAULT_STARTUP_TIMEOUT_S = 300
DEFAULT_MIN_COMPLETION_TOKENS = 128
DEFAULT_BASE_PORT = 19100

BLOCKER_BASENAMES = {"llama-server", "llama-cli", "llama-bench", "llama-mtmd-cli"}
AUTOPILOT_MARKERS = (
    "scripts/autopilot/autopilot.py start",
    "start_fable_authority_daemon.py",
    "autopilot_supervisor.py",
)


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    role: str
    description: str
    model: Path
    max_context: int
    threads: int
    ubatch: int
    device: str
    n_gpu_layers: int
    kv_k: str
    kv_v: str
    reasoning: str
    prior_evidence: str
    parallel: int = 1
    jinja: bool = True
    mlock: bool = False
    enable_thinking: bool | None = None
    draft_model: Path | None = None
    spec_type: str = "none"
    spec_draft_n_max: int | None = None
    spec_draft_threads: int | None = None
    spec_draft_device: str | None = None
    spec_draft_ngl: int | None = None
    no_mmap: bool = False
    override_kv: tuple[str, ...] = ()
    slot_save_path: Path | None = None
    extra_args: tuple[str, ...] = ()


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="frontdoor_gpu_resident_no_spec",
        role="frontdoor",
        description=(
            "MI210-resident Qwen3.6-35B frontdoor fastest validated Stage-2 arm; "
            "native MTP and external draft-tree were slower in Stage-2 evidence."
        ),
        model=Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"),
        max_context=32768,
        threads=96,
        ubatch=512,
        device="ROCm0",
        n_gpu_layers=99,
        kv_k="q8_0",
        kv_v="q8_0",
        reasoning="off",
        enable_thinking=False,
        prior_evidence=(
            "data/specdec_frontdoor_alpha/stage2_mi210_gpu_residency_20260717T0510Z/"
            "summary.json: gpu_no_spec 101.64 t/s vs native MTP 96.40 and external 36.06"
        ),
    ),
    Scenario(
        name="worker_general_cpu_composed_spec",
        role="worker_general",
        description=(
            "Production-shaped Gemma4 worker CPU lane with composed ngram-mod,draft-mtp, "
            "assistant v6 Q8 head, q8 KV, and reasoning off."
        ),
        model=Path("/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"),
        draft_model=Path("/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"),
        max_context=16384,
        threads=96,
        ubatch=512,
        device="none",
        n_gpu_layers=0,
        kv_k="q8_0",
        kv_v="q8_0",
        reasoning="off",
        spec_type="ngram-mod,draft-mtp",
        spec_draft_n_max=2,
        spec_draft_threads=16,
        spec_draft_device="none",
        spec_draft_ngl=0,
        no_mmap=True,
        prior_evidence=(
            "/mnt/raid0/llm/tmp/v7-worker-general-shortctx-recheck-20260717T112435Z/: "
            "short-context full-instance composed spec decoded at 76.02 and 116.96 t/s"
        ),
    ),
    Scenario(
        name="architect_general_cpu_native_mtp",
        role="architect_general",
        description=(
            "Production-shaped Qwen3.5-122B architect CPU lane with native NEXTN "
            "same-file draft-mtp, q4_0/f16 KV, jinja, mlock, and thinking disabled."
        ),
        model=Path(
            "/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/"
            "Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf"
        ),
        max_context=16384,
        threads=96,
        ubatch=8192,
        device="none",
        n_gpu_layers=0,
        kv_k="q4_0",
        kv_v="f16",
        reasoning="off",
        parallel=2,
        mlock=True,
        enable_thinking=False,
        spec_type="draft-mtp",
        spec_draft_n_max=4,
        slot_save_path=Path("/mnt/raid0/llm/cache/kv_slots/architect_general"),
        prior_evidence=(
            "/mnt/raid0/llm/tmp/v7-spec-server-ab-20260716T155208Z/summary.json: "
            "architect draft-mtp prod 19.34 t/s vs v7 19.30 t/s with matched acceptance"
        ),
    ),
    Scenario(
        name="ingest_long_context_cpu_default_experts",
        role="ingest_long_context",
        description=(
            "Qwen3-Next ingest CPU lane with default expert count and speculation "
            "disabled because SSM/recurrent state is unsafe for spec."
        ),
        model=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-Next-80B-A3B-Instruct-GGUF/"
            "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
        ),
        max_context=32768,
        threads=96,
        ubatch=8192,
        device="none",
        n_gpu_layers=0,
        kv_k="q4_0",
        kv_v="q4_0",
        reasoning="auto",
        mlock=True,
        spec_type="none",
        slot_save_path=Path("/mnt/raid0/llm/cache/kv_slots/ingest_long_context"),
        prior_evidence=(
            "Operator correction 2026-07-17: historical qwen3next.expert_used_count=int:4 "
            "registry policy is stale for K35; measure default-expert ingest with spec disabled."
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def scenario_by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


def selected_scenarios(names: list[str] | None) -> list[Scenario]:
    if not names:
        return list(SCENARIOS)
    return [scenario_by_name(name) for name in names]


def selected_contexts(values: list[int] | None) -> list[int]:
    if not values:
        return list(DEFAULT_CONTEXTS)
    return values


def prompt_for_context(nominal_context: int, max_tokens: int) -> str:
    # Keep the filler word tokenizer-friendly. Earlier alphanumeric markers
    # expanded 3x past the nominal context and turned a 2K smoke into 6K tokens.
    target_words = max(64, nominal_context - max_tokens - 1024)
    repeated = " ".join("benchmark" for _ in range(target_words))
    return (
        "You are serving a throughput benchmark. Preserve the instruction at the end.\n\n"
        f"Context block ({nominal_context} nominal tokens):\n{repeated}\n\n"
        f"Now write exactly {max_tokens} lowercase words. Use the word benchmark repeated, "
        "separated by spaces. Do not add bullets, numbering, or commentary."
    )


def pick_port(base: int) -> int:
    for port in range(base, base + 1000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in range {base}-{base + 999}")


def server_context(scenario: Scenario, nominal_context: int, max_tokens: int) -> int:
    per_slot_requested = max(2048, nominal_context + max_tokens + 1024)
    requested = per_slot_requested * scenario.parallel
    capped = min(scenario.max_context, requested)
    per_slot_capped = capped // scenario.parallel
    if per_slot_capped < max_tokens + 1024:
        raise ValueError(
            f"{scenario.name} context cap {scenario.max_context} is too small for {max_tokens} tokens"
        )
    return capped


def build_server_argv(
    scenario: Scenario,
    *,
    binary: Path,
    port: int,
    nominal_context: int,
    max_tokens: int,
) -> list[str]:
    ctx = server_context(scenario, nominal_context, max_tokens)
    argv = [
        "env",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_BIN_DIR}",
        "GGML_IQK=1",
        "ROCR_VISIBLE_DEVICES=0",
        "HIP_VISIBLE_DEVICES=0",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(scenario.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        str(scenario.parallel),
        "-c",
        str(ctx),
        "-t",
        str(scenario.threads),
        "-ub",
        str(scenario.ubatch),
        "--metrics",
        "--slots",
    ]
    if scenario.jinja:
        argv.append("--jinja")
    argv.extend(
        [
            "--reasoning",
            scenario.reasoning,
            "--device",
            scenario.device,
            "-ngl",
            str(scenario.n_gpu_layers),
            "-ctk",
            scenario.kv_k,
            "-ctv",
            scenario.kv_v,
            "-fa",
            "on",
            "--spec-type",
            scenario.spec_type,
        ]
    )
    if scenario.mlock:
        argv.append("--mlock")
    if scenario.no_mmap:
        argv.append("--no-mmap")
    if scenario.draft_model is not None:
        argv.extend(["-md", str(scenario.draft_model)])
    if scenario.spec_draft_n_max is not None:
        argv.extend(["--spec-draft-n-max", str(scenario.spec_draft_n_max)])
    if scenario.spec_draft_threads is not None:
        argv.extend(["--spec-draft-threads", str(scenario.spec_draft_threads)])
    if scenario.spec_draft_device is not None:
        argv.extend(["--spec-draft-device", scenario.spec_draft_device])
    if scenario.spec_draft_ngl is not None:
        argv.extend(["--spec-draft-ngl", str(scenario.spec_draft_ngl)])
    for override in scenario.override_kv:
        argv.extend(["--override-kv", override])
    if scenario.slot_save_path is not None:
        argv.extend(["--slot-save-path", str(scenario.slot_save_path)])
    argv.extend(scenario.extra_args)
    return argv


def run_capture(argv: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": argv, "ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "argv": argv,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def collect_process_blockers() -> list[dict[str, Any]]:
    proc = run_capture(["ps", "-eo", "pid=,comm=,args="], timeout=10)
    blockers: list[dict[str, Any]] = []
    if not proc["ok"]:
        return [{"error": proc["stderr"]}]
    current = os.getpid()
    for line in proc["stdout"].splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_text, comm, args = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current:
            continue
        argv0 = Path(args.split(maxsplit=1)[0]).name if args else comm
        if comm in BLOCKER_BASENAMES or argv0 in BLOCKER_BASENAMES:
            blockers.append({"pid": pid, "comm": comm, "args": args, "reason": "llama workload"})
        elif any(marker in args for marker in AUTOPILOT_MARKERS):
            blockers.append({"pid": pid, "comm": comm, "args": args, "reason": "autopilot"})
    return blockers


def collect_guard_state(binary: Path) -> dict[str, Any]:
    return {
        "captured_at": utc_now(),
        "binary": str(binary),
        "server_version": run_capture([str(binary), "--version"], timeout=30),
        "git": {
            "experimental_head": run_capture(
                ["git", "-C", str(EXPERIMENTAL_ROOT), "rev-parse", "--short", "HEAD"], timeout=10
            ),
            "experimental_status": run_capture(
                ["git", "-C", str(EXPERIMENTAL_ROOT), "status", "--short"], timeout=10
            ),
        },
        "devices": run_capture(
            [
                "env",
                f"LD_LIBRARY_PATH={EXPERIMENTAL_BIN_DIR}",
                "ROCR_VISIBLE_DEVICES=0",
                "HIP_VISIBLE_DEVICES=0",
                str(binary),
                "--list-devices",
            ],
            timeout=30,
        ),
        "process_blockers": collect_process_blockers(),
        "memory": run_capture(["free", "-h"], timeout=10),
        "numa": run_capture(["numactl", "--hardware"], timeout=20),
    }


def wait_for_health(port: int, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become healthy: {last_error}")


def query_chat(
    scenario: Scenario,
    port: int,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    body = {
        "model": "k35-local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 35,
        "stream": False,
        "cache_prompt": False,
    }
    if scenario.enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": scenario.enable_thinking}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def terminate(proc: subprocess.Popen[str], *, timeout_s: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": proc.pid, "terminated": False, "killed": False}
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
            result["killed"] = True
    result["returncode"] = proc.returncode
    ps = run_capture(["ps", "-p", str(proc.pid), "-o", "pid=,comm=,args="], timeout=10)
    result["ps_after"] = ps
    result["dead"] = str(proc.pid) not in ps.get("stdout", "")
    return result


def content_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def summarize_response(
    scenario: Scenario,
    nominal_context: int,
    max_tokens: int,
    response: dict[str, Any],
    elapsed_s: float,
    min_completion_tokens: int,
) -> dict[str, Any]:
    timings = response.get("timings") or {}
    usage = response.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    content = content_from_response(response)
    return {
        "scenario": scenario.name,
        "role": scenario.role,
        "nominal_context": nominal_context,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tps": timings.get("prompt_per_second"),
        "decode_tps": timings.get("predicted_per_second"),
        "elapsed_s": elapsed_s,
        "content_preview": content[:240],
        "passed_min_completion": completion_tokens >= min_completion_tokens,
        "min_completion_tokens": min_completion_tokens,
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "max_tokens": max_tokens,
    }


def render_commands(plan: dict[str, Any]) -> str:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for cell in plan["cells"]:
        lines.append(f"# {cell['scenario']} nominal_context={cell['nominal_context']}")
        lines.append(shlex.join(cell["server_argv"]))
        lines.append("")
    return "\n".join(lines)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = selected_scenarios(args.only)
    contexts = selected_contexts(args.context)
    cells: list[dict[str, Any]] = []
    port = args.port_base
    for scenario in scenarios:
        for nominal_context in contexts:
            if nominal_context > scenario.max_context // scenario.parallel:
                continue
            cell_port = port
            port += 1
            cells.append(
                {
                    "scenario": scenario.name,
                    "role": scenario.role,
                    "description": scenario.description,
                    "prior_evidence": scenario.prior_evidence,
                    "nominal_context": nominal_context,
                    "server_context": server_context(scenario, nominal_context, args.max_tokens),
                    "max_tokens": args.max_tokens,
                    "server_argv": build_server_argv(
                        scenario,
                        binary=args.binary,
                        port=cell_port,
                        nominal_context=nominal_context,
                        max_tokens=args.max_tokens,
                    ),
                    "port": cell_port,
                }
            )
    return {
        "schema": "epyc.k35_stack_context_matrix.plan.v1",
        "created_at": utc_now(),
        "execute": args.execute,
        "binary": str(args.binary),
        "contexts": contexts,
        "max_tokens": args.max_tokens,
        "cells": cells,
    }


def run_cell(cell: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    scenario = scenario_by_name(cell["scenario"])
    cell_dir = output_dir / "runs" / f"{scenario.name}_ctx{cell['nominal_context']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    server_log = cell_dir / "server.stderr"
    response_path = cell_dir / "response.json"
    result_path = cell_dir / "result.json"
    argv_path = cell_dir / "server_argv.json"
    write_json(argv_path, cell["server_argv"])
    with server_log.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            cell["server_argv"],
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
    started = time.monotonic()
    cleanup: dict[str, Any] | None = None
    try:
        wait_for_health(cell["port"], args.startup_timeout)
        prompt = prompt_for_context(cell["nominal_context"], args.max_tokens)
        request_started = time.monotonic()
        response = query_chat(
            scenario,
            cell["port"],
            prompt,
            max_tokens=args.max_tokens,
            timeout_s=args.request_timeout,
        )
        elapsed_s = time.monotonic() - request_started
        write_json(response_path, response)
        result = summarize_response(
            scenario,
            cell["nominal_context"],
            args.max_tokens,
            response,
            elapsed_s,
            args.min_completion_tokens,
        )
        result["status"] = "ok" if result["passed_min_completion"] else "short_completion"
    except Exception as exc:  # noqa: BLE001 - artifact should capture failure details
        result = {
            "scenario": scenario.name,
            "role": scenario.role,
            "nominal_context": cell["nominal_context"],
            "status": "error",
            "error": repr(exc),
        }
    finally:
        cleanup = terminate(proc)
    result["started_at_monotonic"] = started
    result["cleanup"] = cleanup
    result["server_log"] = str(server_log)
    result["response_path"] = str(response_path)
    write_json(result_path, result)
    return result


def execute_plan(plan: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    guard = collect_guard_state(args.binary)
    write_json(output_dir / "guard_state.json", guard)
    blockers = guard.get("process_blockers") or []
    if blockers and not args.allow_dirty_host:
        summary = {
            "schema": "epyc.k35_stack_context_matrix.summary.v1",
            "created_at": utc_now(),
            "status": "blocked",
            "reason": "process blockers present",
            "blockers": blockers,
            "results": [],
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    results = [run_cell(cell, args, output_dir) for cell in plan["cells"]]
    cleanup_guard = collect_process_blockers()
    summary = {
        "schema": "epyc.k35_stack_context_matrix.summary.v1",
        "created_at": utc_now(),
        "status": "ok" if all(result.get("status") == "ok" for result in results) else "partial",
        "results": results,
        "cleanup_process_blockers": cleanup_guard,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the selected matrix cells")
    parser.add_argument(
        "--only",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Scenario to include. May be repeated. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--context",
        action="append",
        type=int,
        help="Nominal context depth to include. May be repeated. Defaults to 2048/8192/32768.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-completion-tokens", type=int, default=DEFAULT_MIN_COMPLETION_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--port-base", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--allow-dirty-host", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    write_json(args.output_dir / "plan.json", plan)
    (args.output_dir / "commands.sh").write_text(render_commands(plan), encoding="utf-8")
    if not args.execute:
        print(f"dry-run plan written to {args.output_dir}")
        print(f"cells: {len(plan['cells'])}")
        return 0
    summary = execute_plan(plan, args, args.output_dir)
    print(canonical_json(summary))
    return 0 if summary.get("status") in {"ok", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
