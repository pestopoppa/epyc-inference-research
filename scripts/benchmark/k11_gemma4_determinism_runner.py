#!/usr/bin/env python3
"""K11 Gemma4 external-head determinism runner.

Defaults to a dry-run plan. Pass --execute to launch fresh llama-server
instances sequentially, one per run, against the experimental v7 HIP build.

The runner is intentionally narrow:
  - binary is pinned to /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server
  - LD_LIBRARY_PATH is pinned to that build directory
  - each run gets a fresh high ephemeral port
  - requests are chat-completions with temp=0 and seed=42
  - output hashes, draft counters, and server logs are captured per run
  - server PIDs are always terminated and verified dead
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
SERVER_BIN = Path("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server")
SERVER_LIB_DIR = SERVER_BIN.parent

DEFAULT_TARGET_MODEL = Path("/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf")
DEFAULT_DRAFT_MODEL = Path("/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf")
DEFAULT_PROMPT = (
    "Return a compact JSON object with keys status, model, and seed, "
    'using values "ok", "gemma4", and 42.'
)

DEFAULT_PORT_TIMEOUT_S = 180
DEFAULT_REQUEST_TIMEOUT_S = 120
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 64
DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 16384
DEFAULT_UBATCH = 512
DEFAULT_N_GPU_LAYERS = 99
DEFAULT_SPEC_DRAFT_N_MAX = 2
DEFAULT_REPEATS = 2
DEFAULT_SLOTS = 4
DEFAULT_REQUEST_SAMPLER_MODE = "current"
DEFAULT_DRAFT_BACKEND_SAMPLING = "default"
DEFAULT_SCHEMA_TASK = "none"
SCHEMA_TASK_WORD_ARRAY_200 = "word-array-200"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    label: str
    repeat_index: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K11 Gemma4 determinism runner")
    parser.add_argument("--execute", action="store_true", help="Launch servers and run the chat request")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESEARCH_ROOT / "data" / "k11_gemma4_determinism"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="Output directory for plan/results/logs",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_REPEATS, help="Fresh-server runs to execute")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic request seed")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Request temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send on each run")
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Stop string to send in the chat-completions payload; may be repeated",
    )
    parser.add_argument(
        "--spec-type",
        choices=("draft-mtp", "none"),
        default="draft-mtp",
        help="Speculative mode for the server under test",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
        help="Server parallel slots (-np). Defaults to the historical runner shape.",
    )
    parser.add_argument(
        "--expected-word",
        default=None,
        help="Optional exact repeated word to score in message.content",
    )
    parser.add_argument(
        "--expected-word-count",
        type=int,
        default=None,
        help="Optional exact repeated word count to score in message.content",
    )
    parser.add_argument(
        "--schema-task",
        choices=(DEFAULT_SCHEMA_TASK, SCHEMA_TASK_WORD_ARRAY_200),
        default=DEFAULT_SCHEMA_TASK,
        help=(
            "Optional request-level JSON schema task. word-array-200 constrains the response "
            "to a JSON object with 200 benchmark entries and done=END."
        ),
    )
    parser.add_argument(
        "--request-sampler-mode",
        choices=("current", "explicit-greedy", "cpu-top-k"),
        default=DEFAULT_REQUEST_SAMPLER_MODE,
        help=(
            "Request sampler payload mode. current keeps the historical top_k=1 shape; "
            "explicit-greedy uses samplers=[temperature], top_k=0, backend_sampling=false; "
            "cpu-top-k uses samplers=[top_k,temperature], top_k=1, backend_sampling=false."
        ),
    )
    parser.add_argument(
        "--draft-backend-sampling",
        choices=("default", "on", "off"),
        default=DEFAULT_DRAFT_BACKEND_SAMPLING,
        help="Server-side speculative draft backend sampling toggle for K11 diagnostics.",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        default=DEFAULT_TARGET_MODEL,
        help="Gemma4 target model path",
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        default=DEFAULT_DRAFT_MODEL,
        help="Gemma4 assistant draft model path",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="CPU threads for llama-server")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="Context size")
    parser.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, help="Micro-batch size")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=DEFAULT_N_GPU_LAYERS,
        help="Target model GPU layers",
    )
    parser.add_argument(
        "--spec-draft-n-max",
        type=int,
        default=DEFAULT_SPEC_DRAFT_N_MAX,
        help="Speculative draft length for the assistant head",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=DEFAULT_PORT_TIMEOUT_S,
        help="Server health timeout in seconds",
    )
    return parser.parse_args(argv)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pick_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def apply_request_sampler_mode(payload: dict[str, Any], mode: str) -> None:
    if mode == "current":
        return
    payload["min_p"] = 0.0
    payload["backend_sampling"] = False
    if mode == "explicit-greedy":
        payload["samplers"] = ["temperature"]
        payload["top_k"] = 0
    elif mode == "cpu-top-k":
        payload["samplers"] = ["top_k", "temperature"]
        payload["top_k"] = 1
    else:
        raise ValueError(f"unknown request sampler mode: {mode}")


def json_schema_for_schema_task(schema_task: str) -> dict[str, Any] | None:
    if schema_task == DEFAULT_SCHEMA_TASK:
        return None
    if schema_task == SCHEMA_TASK_WORD_ARRAY_200:
        return {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "minItems": 200,
                    "maxItems": 200,
                    "items": {"type": "string", "enum": ["benchmark"]},
                },
                "done": {"type": "string", "enum": ["END"]},
            },
            "required": ["words", "done"],
            "additionalProperties": False,
        }
    raise ValueError(f"unknown schema task: {schema_task}")


def query_chat(
    port: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: int,
    request_sampler_mode: str = DEFAULT_REQUEST_SAMPLER_MODE,
    stop: list[str] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 1,
        "seed": seed,
        "stream": False,
    }
    if stop:
        payload["stop"] = list(stop)
    if json_schema is not None:
        payload["json_schema"] = json_schema
    apply_request_sampler_mode(payload, request_sampler_mode)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw), raw


def build_server_argv(args: argparse.Namespace, port: int | str) -> list[str]:
    argv = [
        "env",
        f"LD_LIBRARY_PATH={SERVER_LIB_DIR}",
        "numactl",
        "--interleave=all",
        str(SERVER_BIN),
        "-m",
        str(args.target_model),
        "-np",
        str(args.slots),
        "--device",
        "ROCm0",
        "-ngl",
        str(args.n_gpu_layers),
        "-t",
        str(args.threads),
        "-ub",
        str(args.ubatch),
        "-c",
        str(args.context),
        "-fa",
        "on",
        "-rea",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if args.spec_type == "draft-mtp":
        argv.extend(
            [
                "-md",
                str(args.draft_model),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(args.spec_draft_n_max),
                "--device-draft",
                "ROCm0",
                "--spec-draft-ngl",
                str(args.n_gpu_layers),
            ]
        )
        if args.draft_backend_sampling == "on":
            argv.append("--spec-draft-backend-sampling")
        elif args.draft_backend_sampling == "off":
            argv.append("--no-spec-draft-backend-sampling")
    else:
        argv.extend(["--spec-type", "none"])
    return argv


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
        )
    except Exception:
        log_handle.close()
        raise
    proc._k11_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def is_pid_alive(pid: int) -> bool:
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == str(pid)


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
            pass

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

    log_handle = getattr(proc, "_k11_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    runs = [RunSpec(label=f"run_{index:02d}", repeat_index=index) for index in range(1, args.runs + 1)]
    return {
        "meta": {
            "mode": "dry_run",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "server_bin": str(SERVER_BIN),
            "ld_library_path": str(SERVER_LIB_DIR),
            "target_model": str(args.target_model),
            "draft_model": str(args.draft_model),
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt": args.prompt,
            "stop": args.stop,
            "spec_type": args.spec_type,
            "spec_draft_n_max": args.spec_draft_n_max,
            "slots": args.slots,
            "expected_word": args.expected_word,
            "expected_word_count": args.expected_word_count,
            "schema_task": args.schema_task,
            "json_schema": json_schema_for_schema_task(args.schema_task),
            "request_sampler_mode": args.request_sampler_mode,
            "draft_backend_sampling": args.draft_backend_sampling,
            "threads": args.threads,
            "context": args.context,
            "ubatch": args.ubatch,
            "n_gpu_layers": args.n_gpu_layers,
        },
        "runs": [
            {
                "label": run.label,
                "repeat_index": run.repeat_index,
        "server_argv": build_server_argv(args, "[ephemeral]"),
            }
            for run in runs
        ],
    }


def render_commands(args: argparse.Namespace, output_dir: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'RESULT_DIR="{output_dir}"',
        f'LLAMA_SERVER="{SERVER_BIN}"',
        f'export LD_LIBRARY_PATH="{SERVER_LIB_DIR}"',
        "",
    ]
    for index in range(1, args.runs + 1):
        port = 30000 + index
        argv = build_server_argv(args, port)
        lines.append(f"# run_{index:02d}")
        lines.append(" ".join(shlex.quote(part) for part in argv))
        lines.append("")
    return "\n".join(lines)


def score_word_task(content: str, expected_word: str | None, expected_count: int | None) -> dict[str, Any] | None:
    if expected_word is None and expected_count is None:
        return None
    words = content.split()
    bad_words = [word for word in words if expected_word is not None and word != expected_word]
    count_ok = expected_count is None or len(words) == expected_count
    word_ok = expected_word is None or not bad_words
    return {
        "expected_word": expected_word,
        "expected_word_count": expected_count,
        "observed_word_count": len(words),
        "bad_word_count": len(bad_words),
        "count_ok": count_ok,
        "word_ok": word_ok,
        "passed": count_ok and word_ok,
    }


def score_schema_task(content: str, schema_task: str) -> dict[str, Any] | None:
    if schema_task == DEFAULT_SCHEMA_TASK:
        return None
    if schema_task != SCHEMA_TASK_WORD_ARRAY_200:
        raise ValueError(f"unknown schema task: {schema_task}")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "schema_task": schema_task,
            "json_ok": False,
            "parse_error": str(exc),
            "passed": False,
        }
    words = parsed.get("words") if isinstance(parsed, dict) else None
    done = parsed.get("done") if isinstance(parsed, dict) else None
    words_ok = isinstance(words, list) and len(words) == 200 and all(word == "benchmark" for word in words)
    done_ok = done == "END"
    return {
        "schema_task": schema_task,
        "json_ok": True,
        "observed_word_count": len(words) if isinstance(words, list) else None,
        "bad_word_count": (
            sum(1 for word in words if word != "benchmark") if isinstance(words, list) else None
        ),
        "done": done,
        "words_ok": words_ok,
        "done_ok": done_ok,
        "passed": words_ok and done_ok,
    }


def run_execute(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    logs_dir = output_dir / "logs"
    responses_dir = output_dir / "responses"
    for directory in (runs_dir, logs_dir, responses_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    json_schema = json_schema_for_schema_task(args.schema_task)
    for index in range(1, args.runs + 1):
        label = f"run_{index:02d}"
        port = pick_ephemeral_port()
        log_path = logs_dir / f"{label}.server.log"
        result_path = runs_dir / f"{label}.json"
        raw_response_path = responses_dir / f"{label}.raw.json"
        argv = build_server_argv(args, port)
        proc: subprocess.Popen[str] | None = None
        run_record: dict[str, Any] = {
            "label": label,
            "repeat_index": index,
            "port": port,
            "server_argv": argv,
            "server_log": str(log_path),
            "request": {
                "seed": args.seed,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "prompt": args.prompt,
                "stop": args.stop,
                "schema_task": args.schema_task,
            },
        }
        try:
            proc = launch_server(argv, log_path)
            run_record["server_pid"] = proc.pid
            wait_for_health(port, args.startup_timeout, pid=proc.pid)
            response, raw_response = query_chat(
                port=port,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
                timeout_s=args.request_timeout,
                request_sampler_mode=args.request_sampler_mode,
                stop=args.stop,
                json_schema=json_schema,
            )
            raw_response_path.write_text(raw_response, encoding="utf-8")
            choices = response.get("choices", [])
            choice = choices[0] if choices else {}
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            reasoning_content = message.get("reasoning_content", "")
            if isinstance(reasoning_content, list):
                reasoning_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in reasoning_content
                )
            semantic_output = {
                "content": str(content or ""),
                "reasoning_content": str(reasoning_content or ""),
            }
            timings = response.get("timings", {})
            usage = response.get("usage", {})
            output_hash = sha256_text(canonical_json(semantic_output))
            response_hash = sha256_text(canonical_json(response))
            draft_n = int(timings.get("draft_n") or 0)
            draft_accepted = int(timings.get("draft_n_accepted") or 0)
            acceptance_rate = (draft_accepted / draft_n) if draft_n > 0 else 0.0
            task_eval = score_schema_task(semantic_output["content"], args.schema_task)
            if task_eval is None:
                task_eval = score_word_task(
                    semantic_output["content"],
                    args.expected_word,
                    args.expected_word_count,
                )
            run_record.update(
                {
                    "status": "ok",
                    "response_sha256": response_hash,
                    "output_sha256": output_hash,
                    "content": semantic_output["content"],
                    "reasoning_content": semantic_output["reasoning_content"],
                    "usage": usage,
                    "timings": timings,
                    "draft_n": draft_n,
                    "draft_n_accepted": draft_accepted,
                    "acceptance_rate": round(acceptance_rate, 6),
                    "task_eval": task_eval,
                    "response_path": str(raw_response_path),
                }
            )
            result_path.write_text(json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8")
            results.append(run_record)
            print(
                f"{label}: ok hash={output_hash[:12]} draft={draft_accepted}/{draft_n} "
                f"accept={acceptance_rate:.1%} pid={proc.pid}",
                flush=True,
            )
        except Exception as exc:
            run_record.update({"status": "error", "error": str(exc)})
            result_path.write_text(json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8")
            results.append(run_record)
            print(f"{label}: error {exc}", file=sys.stderr, flush=True)
        finally:
            if proc is not None:
                try:
                    terminate_server(proc)
                except Exception as exc:
                    run_record["cleanup_error"] = str(exc)
                    result_path.write_text(json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8")
                    raise

    hashes = [r.get("output_sha256") for r in results if r.get("status") == "ok"]
    unique_hashes = sorted({h for h in hashes if h})
    deterministic = len(unique_hashes) <= 1 and len(hashes) == args.runs and all(r.get("status") == "ok" for r in results)
    task_evals = [r.get("task_eval") for r in results if r.get("task_eval") is not None]
    task_passed = all(e.get("passed") for e in task_evals) if task_evals else None
    summary = {
        "mode": "execute",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_bin": str(SERVER_BIN),
        "ld_library_path": str(SERVER_LIB_DIR),
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "spec_type": args.spec_type,
        "slots": args.slots,
        "stop": args.stop,
        "expected_word": args.expected_word,
        "expected_word_count": args.expected_word_count,
        "schema_task": args.schema_task,
        "json_schema": json_schema,
        "request_sampler_mode": args.request_sampler_mode,
        "draft_backend_sampling": args.draft_backend_sampling,
        "runs": results,
        "unique_output_hashes": unique_hashes,
        "deterministic": deterministic,
        "task_passed": task_passed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if SERVER_BIN != Path("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server"):
        raise RuntimeError(f"unexpected server binary: {SERVER_BIN}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "commands.sh").write_text(render_commands(args, output_dir), encoding="utf-8")

    print(f"K11 Gemma4 determinism runner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {output_dir}")
    print(f"server_bin: {SERVER_BIN}")
    print(f"ld_library_path: {SERVER_LIB_DIR}")
    print(f"target_model: {args.target_model}")
    print(f"draft_model: {args.draft_model}")
    print(f"runs: {args.runs}")
    print(f"prompt: {args.prompt}")
    print(f"seed: {args.seed}")
    print(f"temperature: {args.temperature}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"stop: {args.stop}")
    print(f"spec_type: {args.spec_type}")
    print(f"spec_draft_n_max: {args.spec_draft_n_max}")
    print(f"slots: {args.slots}")
    print(f"request_sampler_mode: {args.request_sampler_mode}")
    print(f"draft_backend_sampling: {args.draft_backend_sampling}")
    print(f"schema_task: {args.schema_task}")
    if args.expected_word is not None or args.expected_word_count is not None:
        print(f"expected_word: {args.expected_word}")
        print(f"expected_word_count: {args.expected_word_count}")

    if not args.execute:
        print("Dry run only. No server will be launched.")
        print(f"Plan written to {output_dir / 'plan.json'}")
        print(f"Commands written to {output_dir / 'commands.sh'}")
        return 0

    summary = run_execute(args, output_dir)
    print(f"Summary written to {output_dir / 'summary.json'}")
    print(f"Unique output hashes: {len(summary['unique_output_hashes'])}")
    if not summary["deterministic"]:
        print("Determinism check failed: output hashes diverged or a run failed.", file=sys.stderr)
        return 1
    print("Determinism check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
