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
DEFAULT_TARGET_DEVICE = "ROCm0"
DEFAULT_DRAFT_DEVICE = "ROCm0"
DEFAULT_SPEC_DRAFT_N_MAX = 2
DEFAULT_REPEATS = 2
DEFAULT_SLOTS = 4
DEFAULT_REQUEST_SAMPLER_MODE = "current"
DEFAULT_DRAFT_BACKEND_SAMPLING = "default"
DEFAULT_SCHEMA_TASK = "none"
SCHEMA_TASK_WORD_ARRAY_200 = "word-array-200"
DEFAULT_TRACE_N_PROBS = 5
DEFAULT_TRACE_RESPONSE_FIELDS = [
    "choices",
    "usage",
    "timings",
    "tokens",
    "completion_probabilities",
    "__verbose/tokens",
    "__verbose/completion_probabilities",
]


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
        "--trace-token-divergence",
        action="store_true",
        help=(
            "Request token/probability metadata and write compact per-run traces for "
            "first-divergent-token diagnostics."
        ),
    )
    parser.add_argument(
        "--trace-n-probs",
        type=int,
        default=DEFAULT_TRACE_N_PROBS,
        help="Top token probabilities to request when --trace-token-divergence is set.",
    )
    parser.add_argument(
        "--trace-post-sampling-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When tracing, request post-sampling probabilities. Use "
            "--no-trace-post-sampling-probs for pre-sampling logprobs."
        ),
    )
    parser.add_argument(
        "--trace-response-field",
        action="append",
        dest="trace_response_fields",
        default=[],
        help=(
            "Response field to request when tracing; may be repeated. Defaults to "
            "compact chat/native token fields."
        ),
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
        "--target-device",
        default=DEFAULT_TARGET_DEVICE,
        help="Target model device argument for llama-server --device; use 'none' for CPU-only controls.",
    )
    parser.add_argument(
        "--draft-device",
        default=DEFAULT_DRAFT_DEVICE,
        help="Draft model device argument for --device-draft when --spec-type=draft-mtp.",
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
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra environment assignment for the llama-server subprocess; may be repeated. "
            "Recorded in plan/summary for graph/backend A/B diagnostics."
        ),
    )
    args = parser.parse_args(argv)
    if args.trace_n_probs < 0:
        parser.error("--trace-n-probs must be >= 0")
    for item in args.server_env:
        key, sep, _value = item.partition("=")
        if not sep or not key:
            parser.error("--server-env must be KEY=VALUE with a non-empty KEY")
    return args


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


def trace_response_fields_from_args(args: argparse.Namespace) -> list[str]:
    fields = getattr(args, "trace_response_fields", None)
    return list(fields) if fields else list(DEFAULT_TRACE_RESPONSE_FIELDS)


def apply_token_trace_options(
    payload: dict[str, Any],
    *,
    enabled: bool,
    n_probs: int = DEFAULT_TRACE_N_PROBS,
    post_sampling_probs: bool = True,
    response_fields: list[str] | None = None,
) -> None:
    if not enabled:
        return
    payload["return_tokens"] = True
    payload["n_probs"] = max(0, n_probs)
    payload["post_sampling_probs"] = bool(post_sampling_probs)
    payload["response_fields"] = list(response_fields or DEFAULT_TRACE_RESPONSE_FIELDS)
    payload["verbose"] = True


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
    *,
    trace_token_divergence: bool = False,
    trace_n_probs: int = DEFAULT_TRACE_N_PROBS,
    trace_post_sampling_probs: bool = True,
    trace_response_fields: list[str] | None = None,
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
    apply_token_trace_options(
        payload,
        enabled=trace_token_divergence,
        n_probs=trace_n_probs,
        post_sampling_probs=trace_post_sampling_probs,
        response_fields=trace_response_fields,
    )
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
        *args.server_env,
        "numactl",
        "--interleave=all",
        str(SERVER_BIN),
        "-m",
        str(args.target_model),
        "-np",
        str(args.slots),
        "--device",
        str(args.target_device),
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
                str(args.draft_device),
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
            "trace_token_divergence": args.trace_token_divergence,
            "trace_n_probs": args.trace_n_probs,
            "trace_post_sampling_probs": args.trace_post_sampling_probs,
            "trace_response_fields": trace_response_fields_from_args(args),
            "threads": args.threads,
            "context": args.context,
            "ubatch": args.ubatch,
            "n_gpu_layers": args.n_gpu_layers,
            "target_device": args.target_device,
            "draft_device": args.draft_device,
            "server_env": args.server_env,
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


def normalize_text_content(value: Any) -> str:
    if isinstance(value, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in value)
    return str(value or "")


def first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    return choice if isinstance(choice, dict) else {}


def semantic_output_from_response(response: dict[str, Any]) -> tuple[dict[str, str], str | None]:
    choice = first_choice(response)
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    return (
        {
            "content": normalize_text_content(message.get("content", "")),
            "reasoning_content": normalize_text_content(message.get("reasoning_content", "")),
        },
        choice.get("finish_reason") if isinstance(choice, dict) else None,
    )


def word_count(text: str) -> int:
    return len(text.split())


def compact_token_entry(entry: Any, *, include_top: bool = True) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {"value": entry}

    compact: dict[str, Any] = {}
    for key in ("id", "token", "bytes", "prob", "logprob"):
        if key in entry:
            compact[key] = entry[key]

    if include_top:
        for top_key in ("top_probs", "top_logprobs"):
            top_values = entry.get(top_key)
            if isinstance(top_values, list):
                compact[top_key] = [
                    compact_token_entry(top_entry, include_top=False)
                    for top_entry in top_values
                    if isinstance(top_entry, dict)
                ]
    return compact


def _token_ids_from_list(values: Any) -> list[int | str] | None:
    if not isinstance(values, list):
        return None
    token_ids: list[int | str] = []
    for value in values:
        if isinstance(value, int):
            token_ids.append(value)
        elif isinstance(value, str):
            token_ids.append(value)
        elif isinstance(value, dict) and "id" in value:
            token_ids.append(value["id"])
        else:
            return None
    return token_ids


def extract_token_ids(response: dict[str, Any]) -> tuple[list[int | str] | None, str | None]:
    top_level = _token_ids_from_list(response.get("tokens"))
    if top_level is not None:
        return top_level, "tokens"

    verbose = response.get("__verbose")
    if isinstance(verbose, dict):
        verbose_tokens = _token_ids_from_list(verbose.get("tokens"))
        if verbose_tokens is not None:
            return verbose_tokens, "__verbose.tokens"
    return None, None


def extract_probability_entries(response: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    choice = first_choice(response)
    logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
    if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
        return [entry for entry in logprobs["content"] if isinstance(entry, dict)], "choices[0].logprobs.content"

    completion_probabilities = response.get("completion_probabilities")
    if isinstance(completion_probabilities, list):
        return [entry for entry in completion_probabilities if isinstance(entry, dict)], "completion_probabilities"

    verbose = response.get("__verbose")
    if isinstance(verbose, dict) and isinstance(verbose.get("completion_probabilities"), list):
        return [
            entry for entry in verbose["completion_probabilities"] if isinstance(entry, dict)
        ], "__verbose.completion_probabilities"

    return [], None


def token_identity_from_entry(entry: dict[str, Any]) -> int | str | None:
    if "id" in entry:
        return entry["id"]
    if "token" in entry:
        return str(entry["token"])
    return None


def build_token_trace(
    *,
    label: str,
    response: dict[str, Any],
    content: str,
    finish_reason: str | None,
) -> dict[str, Any]:
    token_ids, token_source = extract_token_ids(response)
    probability_entries, probability_source = extract_probability_entries(response)
    compact_entries = [compact_token_entry(entry) for entry in probability_entries]

    sequence: list[int | str] = []
    sequence_source: str | None = None
    if token_ids is not None:
        sequence = list(token_ids)
        sequence_source = token_source
    elif probability_entries:
        sequence = [
            identity
            for identity in (token_identity_from_entry(entry) for entry in probability_entries)
            if identity is not None
        ]
        sequence_source = probability_source

    trace: dict[str, Any] = {
        "label": label,
        "finish_reason": finish_reason,
        "content_word_count": word_count(content),
        "token_count": len(sequence),
        "sequence_source": sequence_source,
        "probability_source": probability_source,
        "sequence": sequence,
    }
    if token_ids is not None:
        trace["token_ids"] = token_ids
    if compact_entries:
        trace["tokens"] = compact_entries
    return trace


def token_at(trace: dict[str, Any], index: int) -> dict[str, Any] | None:
    tokens = trace.get("tokens")
    if isinstance(tokens, list) and 0 <= index < len(tokens):
        token = tokens[index]
        return token if isinstance(token, dict) else {"value": token}
    sequence = trace.get("sequence")
    if isinstance(sequence, list) and 0 <= index < len(sequence):
        return {"value": sequence[index]}
    return None


def common_prefix_length(sequences: list[list[int | str]]) -> int | None:
    if not sequences:
        return None
    min_length = min(len(sequence) for sequence in sequences)
    for index in range(min_length):
        first = sequences[0][index]
        if any(sequence[index] != first for sequence in sequences[1:]):
            return index
    return min_length


def count_values(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value if value is not None else "missing")
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_token_divergence_summary(
    *,
    enabled: bool,
    traces: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    trace_by_label = {trace.get("label"): trace for trace in traces}
    ok_results = [result for result in results if result.get("status") == "ok"]
    ordered_traces = [
        trace_by_label.get(result.get("label"))
        for result in ok_results
        if isinstance(trace_by_label.get(result.get("label")), dict)
    ]
    sequences = [
        list(trace.get("sequence", []))
        for trace in ordered_traces
        if trace.get("sequence_source") is not None and isinstance(trace.get("sequence"), list)
    ]
    per_run = [
        {
            "label": trace.get("label"),
            "finish_reason": trace.get("finish_reason"),
            "content_word_count": trace.get("content_word_count"),
            "token_count": trace.get("token_count"),
            "sequence_source": trace.get("sequence_source"),
            "probability_source": trace.get("probability_source"),
        }
        for trace in ordered_traces
    ]
    if len(sequences) != len(ok_results) or not sequences:
        return {
            "enabled": True,
            "available": False,
            "ok_run_count": len(ok_results),
            "trace_run_count": len(ordered_traces),
            "sequence_run_count": len(sequences),
            "per_run": per_run,
        }

    prefix_len = common_prefix_length(sequences)
    assert prefix_len is not None
    all_identical = all(sequence == sequences[0] for sequence in sequences[1:])
    first_divergence = None
    if not all_identical:
        first_divergence = {
            "index": prefix_len,
            "by_run": [
                {
                    "label": trace.get("label"),
                    "finish_reason": trace.get("finish_reason"),
                    "token_count": trace.get("token_count"),
                    "token": token_at(trace, prefix_len),
                    "ended_before_index": prefix_len >= len(trace.get("sequence", [])),
                }
                for trace in ordered_traces
            ],
        }

    token_counts = [len(sequence) for sequence in sequences]
    return {
        "enabled": True,
        "available": True,
        "ok_run_count": len(ok_results),
        "trace_run_count": len(ordered_traces),
        "sequence_run_count": len(sequences),
        "common_prefix_length": prefix_len,
        "all_token_sequences_identical": all_identical,
        "min_token_count": min(token_counts),
        "max_token_count": max(token_counts),
        "first_divergence": first_divergence,
        "per_run": per_run,
    }


def run_execute(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    logs_dir = output_dir / "logs"
    responses_dir = output_dir / "responses"
    token_traces_dir = output_dir / "token_traces"
    directories = [runs_dir, logs_dir, responses_dir]
    if args.trace_token_divergence:
        directories.append(token_traces_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    token_traces: list[dict[str, Any]] = []
    json_schema = json_schema_for_schema_task(args.schema_task)
    trace_response_fields = trace_response_fields_from_args(args)
    for index in range(1, args.runs + 1):
        label = f"run_{index:02d}"
        port = pick_ephemeral_port()
        log_path = logs_dir / f"{label}.server.log"
        result_path = runs_dir / f"{label}.json"
        raw_response_path = responses_dir / f"{label}.raw.json"
        token_trace_path = token_traces_dir / f"{label}.tokens.json"
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
                "trace_token_divergence": args.trace_token_divergence,
                "trace_n_probs": args.trace_n_probs if args.trace_token_divergence else None,
                "trace_post_sampling_probs": (
                    args.trace_post_sampling_probs if args.trace_token_divergence else None
                ),
                "trace_response_fields": trace_response_fields if args.trace_token_divergence else None,
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
                trace_token_divergence=args.trace_token_divergence,
                trace_n_probs=args.trace_n_probs,
                trace_post_sampling_probs=args.trace_post_sampling_probs,
                trace_response_fields=trace_response_fields,
            )
            raw_response_path.write_text(raw_response, encoding="utf-8")
            semantic_output, finish_reason = semantic_output_from_response(response)
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
            content_word_count = word_count(semantic_output["content"])
            reasoning_word_count = word_count(semantic_output["reasoning_content"])
            token_trace_summary = None
            if args.trace_token_divergence:
                token_trace = build_token_trace(
                    label=label,
                    response=response,
                    content=semantic_output["content"],
                    finish_reason=finish_reason,
                )
                token_trace_path.write_text(
                    json.dumps(token_trace, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                token_traces.append(token_trace)
                token_trace_summary = {
                    "path": str(token_trace_path),
                    "token_count": token_trace.get("token_count"),
                    "sequence_source": token_trace.get("sequence_source"),
                    "probability_source": token_trace.get("probability_source"),
                }
            run_record.update(
                {
                    "status": "ok",
                    "finish_reason": finish_reason,
                    "response_sha256": response_hash,
                    "output_sha256": output_hash,
                    "content": semantic_output["content"],
                    "reasoning_content": semantic_output["reasoning_content"],
                    "content_word_count": content_word_count,
                    "reasoning_word_count": reasoning_word_count,
                    "usage": usage,
                    "timings": timings,
                    "draft_n": draft_n,
                    "draft_n_accepted": draft_accepted,
                    "acceptance_rate": round(acceptance_rate, 6),
                    "task_eval": task_eval,
                    "response_path": str(raw_response_path),
                    "token_trace": token_trace_summary,
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
    ok_results = [r for r in results if r.get("status") == "ok"]
    finish_reasons = count_values([r.get("finish_reason") for r in ok_results])
    content_word_counts = [r.get("content_word_count") for r in ok_results]
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
        "trace_token_divergence": args.trace_token_divergence,
        "trace_n_probs": args.trace_n_probs,
        "trace_post_sampling_probs": args.trace_post_sampling_probs,
        "trace_response_fields": trace_response_fields,
        "server_env": args.server_env,
        "runs": results,
        "unique_output_hashes": unique_hashes,
        "deterministic": deterministic,
        "task_passed": task_passed,
        "finish_reasons": finish_reasons,
        "content_word_counts": content_word_counts,
        "token_divergence": build_token_divergence_summary(
            enabled=args.trace_token_divergence,
            traces=token_traces,
            results=results,
        ),
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
    print(f"trace_token_divergence: {args.trace_token_divergence}")
    if args.trace_token_divergence:
        print(f"trace_n_probs: {args.trace_n_probs}")
        print(f"trace_post_sampling_probs: {args.trace_post_sampling_probs}")
        print(f"trace_response_fields: {trace_response_fields_from_args(args)}")
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
