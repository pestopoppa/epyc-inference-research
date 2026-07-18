#!/usr/bin/env python3
"""Qwable verifier/selector best-of-N runner.

Dry-run is the default. Execute mode launches two experimental-v7 llama-server
instances, generates N diverse beneficiary candidates, asks Qwable to select the
best candidate, scores the selected answer, and writes observation-grade
quality/economics artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
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
QUESTION_POOL = RESEARCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"

BENEFICIARY_MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")
QWABLE_IQ4_XS = Path("/mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf")
QWABLE_Q8_0 = Path("/mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf")

DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "qwable_reasoning_economics"
    / f"verifier_selector_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)

DEFAULT_SUITE = "cruxeval"
DEFAULT_TIER = "1"
DEFAULT_N_CANDIDATES = 5
DEFAULT_LIMIT = 3
DEFAULT_PORT_BASE = 18940
DEFAULT_CONTEXT = 16384
DEFAULT_THREADS = 6
DEFAULT_CANDIDATE_MAX_TOKENS = 1024
DEFAULT_VERIFIER_MAX_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT_S = 1200
DEFAULT_STARTUP_TIMEOUT_S = 900
DEFAULT_SEED = 42

SANITIZED_ENV = {
    "HOME": "/tmp",
    "LD_LIBRARY_PATH": str(SERVER_LIB_DIR),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
}

sys.path.insert(0, str(SCRIPT_DIR))
from debug_scorer import _extract_code_block, score_answer  # noqa: E402


@dataclasses.dataclass(frozen=True)
class ServerSpec:
    name: str
    model_path: Path
    port: int
    device: str
    ngl: int
    context: int
    threads: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwable verifier/selector best-of-N probes")
    parser.add_argument("--execute", action="store_true", help="Launch servers and run selected questions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--question-pool", type=Path, default=QUESTION_POOL)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--tier", default=DEFAULT_TIER, help="Tier to filter, or empty string for all tiers")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--beneficiary-model", type=Path, default=BENEFICIARY_MODEL)
    parser.add_argument("--verifier-model", type=Path, default=QWABLE_IQ4_XS)
    parser.add_argument("--beneficiary-device", default="ROCm0")
    parser.add_argument("--verifier-device", default="ROCm0")
    parser.add_argument("--beneficiary-ngl", type=int, default=99)
    parser.add_argument("--verifier-ngl", type=int, default=99)
    parser.add_argument("--candidate-max-tokens", type=int, default=DEFAULT_CANDIDATE_MAX_TOKENS)
    parser.add_argument("--verifier-max-tokens", type=int, default=DEFAULT_VERIFIER_MAX_TOKENS)
    parser.add_argument("--candidate-temperature", type=float, default=0.7)
    parser.add_argument("--verifier-temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-chars", type=int, default=1600)
    parser.add_argument("--candidate-answer-chars", type=int, default=320)
    parser.add_argument("--verifier-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verifier-solve-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--beneficiary-thinking", action="store_true")
    return parser.parse_args(argv)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_pid_alive(pid: int) -> bool:
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == str(pid)


def process_snapshot() -> str:
    probe = subprocess.run(
        [
            "pgrep",
            "-af",
            "llama-server|llama-cli|llama-bench|perf record|perf stat|autopilot",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout


def server_specs(args: argparse.Namespace) -> tuple[ServerSpec, ServerSpec]:
    return (
        ServerSpec(
            name="beneficiary",
            model_path=args.beneficiary_model,
            port=args.port_base,
            device=args.beneficiary_device,
            ngl=args.beneficiary_ngl,
            context=args.context,
            threads=args.threads,
        ),
        ServerSpec(
            name="verifier",
            model_path=args.verifier_model,
            port=args.port_base + 1,
            device=args.verifier_device,
            ngl=args.verifier_ngl,
            context=args.context,
            threads=args.threads,
        ),
    )


def launch_argv(spec: ServerSpec) -> list[str]:
    return [
        str(SERVER_BIN),
        "-m",
        str(spec.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--device",
        spec.device,
        "-ngl",
        str(spec.ngl),
        "-t",
        str(spec.threads),
        "-c",
        str(spec.context),
        "-fa",
        "on",
        "--no-webui",
    ]


def shell_command(argv: list[str]) -> str:
    parts = ["env", "-i", *[f"{k}={v}" for k, v in SANITIZED_ENV.items()], *argv]
    return " ".join(subprocess.list2cmdline([part]) for part in parts)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    specs = server_specs(args)
    end = args.start + args.limit
    return {
        "schema": "qwable_verifier_selector_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "evidence_grade": "observation",
        "experimental_root": str(EXPERIMENTAL_ROOT),
        "server_bin": str(SERVER_BIN),
        "ld_library_path": str(SERVER_LIB_DIR),
        "question_pool": str(args.question_pool),
        "suite": args.suite,
        "tier": args.tier,
        "question_slice": {"start": args.start, "end_exclusive": end, "limit": args.limit},
        "n_candidates": args.n_candidates,
        "request": {
            "seed": args.seed,
            "candidate_max_tokens": args.candidate_max_tokens,
            "verifier_max_tokens": args.verifier_max_tokens,
            "candidate_temperature": args.candidate_temperature,
            "verifier_temperature": args.verifier_temperature,
            "candidate_chars_shown_to_verifier": args.candidate_chars,
            "candidate_answer_chars_shown_to_verifier": args.candidate_answer_chars,
            "beneficiary_thinking": args.beneficiary_thinking,
            "verifier_thinking": args.verifier_thinking,
            "verifier_solve_first": args.verifier_solve_first,
            "request_timeout_s": args.request_timeout,
            "startup_timeout_s": args.startup_timeout,
        },
        "servers": {
            spec.name: {
                "model_path": str(spec.model_path),
                "port": spec.port,
                "device": spec.device,
                "ngl": spec.ngl,
                "context": spec.context,
                "threads": spec.threads,
                "launch": shell_command(launch_argv(spec)),
            }
            for spec in specs
        },
    }


def write_plan(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for spec in plan["servers"].values():
        commands.append(spec["launch"])
    commands.append("")
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "commands.sh").write_text("\n".join(commands), encoding="utf-8")


def load_questions(path: Path, suite: str, tier_text: str) -> list[dict[str, Any]]:
    tier = int(tier_text) if tier_text else None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != suite:
                continue
            if tier is not None and row.get("tier") != tier:
                continue
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("id", "")))
    return rows


def wait_for_health(spec: ServerSpec, timeout_s: int, pid: int) -> None:
    url = f"http://127.0.0.1:{spec.port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            raise RuntimeError(f"{spec.name} server pid {pid} exited before health check")
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace").lower()
            if "ok" in body:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{spec.name} server on port {spec.port} did not become healthy")


def launch_server(spec: ServerSpec, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            launch_argv(spec),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=SANITIZED_ENV,
        )
    except Exception:
        log_handle.close()
        raise
    proc._qwable_verifier_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def terminate_server(proc: subprocess.Popen[str]) -> None:
    pid = proc.pid
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
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
    if proc.poll() is None:
        send(signal.SIGKILL)
        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
    if proc.poll() is None or is_pid_alive(pid):
        raise RuntimeError(f"failed to terminate server pid {pid}")


def chat(port: int, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or message.get("reasoning_content") or "")


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


TYPING_PREFIX = (
    "from typing import List, Optional, Tuple, Dict, Set, Any\n"
    "from collections import defaultdict, deque, Counter\n"
    "import math, heapq, bisect, itertools, functools\n\n"
)


def run_code(answer: str, question: dict[str, Any]) -> bool:
    cfg = question.get("scoring_config") or {}
    code = _extract_code_block(answer, cfg.get("language", "python"))
    if not code:
        return False
    test_code = cfg.get("test_code", "")
    entry = cfg.get("entry_point", "")
    expected = question.get("expected", "")
    if "input()" in code or "sys.stdin" in code:
        return bool(score_answer(answer, expected, "code_execution", cfg))

    full = TYPING_PREFIX + code
    if test_code:
        full += "\n\n" + test_code
    elif entry and expected:
        full += f"\n\nassert {entry}() == {expected}"
    else:
        return bool(score_answer(answer, expected, "code_execution", cfg))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp", encoding="utf-8") as handle:
        handle.write(full)
        path = Path(handle.name)
    try:
        timeout = int(cfg.get("timeout", 10))
        result = subprocess.run(["python3", str(path)], capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        path.unlink(missing_ok=True)


def score_candidate(question: dict[str, Any], answer: str) -> bool:
    method = question.get("scoring_method", "substring")
    if method == "code_execution":
        return run_code(answer, question)
    return bool(score_answer(answer, question.get("expected", ""), method, question.get("scoring_config") or {}))


def extract_final_answer(text: str, question: dict[str, Any]) -> str:
    stripped = strip_think(text)
    cfg = question.get("scoring_config") or {}
    pattern = cfg.get("extract_pattern")
    patterns = [pattern] if isinstance(pattern, str) and pattern else []
    patterns.extend([r"<answer>(.*?)</answer>", r"FINAL\s*[:\-]\s*(.*)"])
    for candidate_pattern in patterns:
        try:
            matches = re.findall(candidate_pattern, stripped, re.IGNORECASE | re.DOTALL)
        except re.error:
            continue
        if matches:
            match = matches[-1]
            if isinstance(match, tuple):
                match = match[-1]
            answer = str(match).strip()
            if answer:
                return answer

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def candidate_verifier_excerpt(text: str, question: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    stripped = strip_think(text)
    answer = extract_final_answer(stripped, question)
    if answer:
        excerpt = f"Extracted final answer:\n{answer[: args.candidate_answer_chars]}"
        if stripped and stripped != answer:
            excerpt += f"\n\nBounded candidate context:\n{stripped[: args.candidate_chars]}"
        return answer, excerpt
    return "", stripped[: args.candidate_chars]


def candidate_payload(question: dict[str, Any], args: argparse.Namespace, index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "auto",
        "messages": [{"role": "user", "content": question["prompt"]}],
        "max_tokens": args.candidate_max_tokens,
        "temperature": args.candidate_temperature,
        "top_p": 0.95,
        "top_k": 20,
        "seed": args.seed + index,
        "stream": False,
        "cache_prompt": True,
        "reasoning_format": "none",
    }
    payload["chat_template_kwargs"] = {"enable_thinking": bool(args.beneficiary_thinking)}
    return payload


def verifier_prompt(question: dict[str, Any], candidates: list[dict[str, Any]], args: argparse.Namespace) -> str:
    parts = [f"## Question\n{question['prompt']}\n", "## Candidate answers"]
    for candidate in candidates:
        shown = str(candidate.get("verifier_excerpt", ""))
        parts.append(f"\n### Candidate {candidate['index']}\n{shown}")
    if args.verifier_solve_first:
        parts.append(
            "\n\nFirst solve the problem independently. Then judge only final-answer correctness. "
            "Prefer each candidate's extracted final answer; use the bounded context only when the extracted answer is absent or ambiguous. "
            "Ignore verbosity, markdown, and explanation length. On the last line output exactly:\n"
            f"FINAL: <index>\n(index 0 to {args.n_candidates - 1})."
        )
    else:
        parts.append(
            "\n\nDo not solve the problem from scratch and do not write reasoning. "
            "Compare only the extracted final answers unless they are absent or ambiguous. "
            "Output exactly one line and nothing else:\n"
            f"FINAL: <index>\n(index 0 to {args.n_candidates - 1})."
        )
    return "\n".join(parts)


def verifier_payload(question: dict[str, Any], candidates: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    system_prompt = (
        "You are a strict verifier. Another model produced several candidate answers. "
        "Select the candidate most likely to be correct."
    )
    if not args.verifier_thinking:
        system_prompt += " Do not explain. Return only the final line: FINAL: <index>."
    return {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": verifier_prompt(question, candidates, args)},
        ],
        "max_tokens": args.verifier_max_tokens,
        "temperature": args.verifier_temperature,
        "top_p": 0.95,
        "top_k": 20,
        "seed": args.seed,
        "stream": False,
        "cache_prompt": True,
        "reasoning_format": "none",
        "chat_template_kwargs": {"enable_thinking": bool(args.verifier_thinking)},
    }


INDEX_PATTERNS = [
    r"FINAL\s*[:\-]?\s*\[?\(?(\d+)",
    r"best\s+candidate\s*(?:is|:)?\s*\[?\(?(\d+)",
    r"<answer>\s*(\d+)\s*</answer>",
    r"\bindex\s*(?:is|:|=)?\s*\[?\(?(\d+)",
]


def parse_index(text: str, n_candidates: int) -> tuple[int, str]:
    for pattern in INDEX_PATTERNS:
        matches = re.findall(pattern, text or "", re.IGNORECASE)
        if matches:
            index = int(matches[-1])
            if 0 <= index < n_candidates:
                return index, "marker"
    for token in reversed(re.findall(r"\b(\d+)\b", text or "")):
        index = int(token)
        if 0 <= index < n_candidates:
            return index, "lastint"
    return 0, "fallback"


def usage(response: dict[str, Any]) -> dict[str, Any]:
    return response.get("usage") or {}


def run_question(
    question: dict[str, Any],
    args: argparse.Namespace,
    beneficiary: ServerSpec,
    verifier: ServerSpec,
    response_dir: Path,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for index in range(args.n_candidates):
        response = chat(beneficiary.port, candidate_payload(question, args, index), args.request_timeout)
        text = message_text(response)
        stripped = strip_think(text)
        extracted_answer, verifier_excerpt = candidate_verifier_excerpt(text, question, args)
        candidates.append(
            {
                "index": index,
                "seed": args.seed + index,
                "correct": score_candidate(question, text),
                "text_sha256": sha256_text(text),
                "extracted_answer": extracted_answer,
                "extracted_answer_sha256": sha256_text(extracted_answer),
                "verifier_excerpt": verifier_excerpt,
                "text_tail": stripped[-240:],
                "usage": usage(response),
                "finish_reason": ((response.get("choices") or [{}])[0].get("finish_reason")),
            }
        )
        (response_dir / f"{question['id']}.candidate_{index}.json").write_text(
            json.dumps(response, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    verifier_response = chat(verifier.port, verifier_payload(question, candidates, args), args.request_timeout)
    verifier_text = message_text(verifier_response)
    selected, parse_mode = parse_index(verifier_text, args.n_candidates)
    passes = [bool(candidate["correct"]) for candidate in candidates]
    (response_dir / f"{question['id']}.verifier.json").write_text(
        json.dumps(verifier_response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "qid": question["id"],
        "suite": question.get("suite"),
        "tier": question.get("tier"),
        "scoring_method": question.get("scoring_method"),
        "n_candidates": args.n_candidates,
        "pass_at_1": passes[0],
        "oracle_pass_at_n": any(passes),
        "n_passing": sum(int(value) for value in passes),
        "verifier_selected_index": selected,
        "verifier_parse": parse_mode,
        "verifier_pass": passes[selected],
        "has_passing": any(passes),
        "verifier_selected_passing": passes[selected] if any(passes) else None,
        "candidates": candidates,
        "verifier_usage": usage(verifier_response),
        "verifier_finish_reason": ((verifier_response.get("choices") or [{}])[0].get("finish_reason")),
        "verifier_text_sha256": sha256_text(verifier_text),
        "verifier_head": verifier_text[:240],
        "verifier_tail": verifier_text[-360:],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    pass1 = sum(int(bool(row.get("pass_at_1"))) for row in rows)
    verifier = sum(int(bool(row.get("verifier_pass"))) for row in rows)
    oracle = sum(int(bool(row.get("oracle_pass_at_n"))) for row in rows)
    has_passing = sum(int(bool(row.get("has_passing"))) for row in rows)
    selected_passing = sum(int(bool(row.get("verifier_selected_passing"))) for row in rows)
    gap = oracle - pass1
    return {
        "n": n,
        "pass_at_1": pass1,
        "verifier_selected": verifier,
        "oracle_pass_at_n": oracle,
        "selection_accuracy_numerator": selected_passing,
        "selection_accuracy_denominator": has_passing,
        "gap_recovered": ((verifier - pass1) / gap) if gap else None,
        "selection_accuracy": (selected_passing / has_passing) if has_passing else None,
    }


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    if not str(SERVER_BIN).startswith(str(EXPERIMENTAL_BIN_DIR)):
        raise RuntimeError(f"refusing non-experimental server binary: {SERVER_BIN}")
    for spec in server_specs(args):
        if not spec.model_path.exists():
            raise FileNotFoundError(spec.model_path)

    preflight = process_snapshot()
    (args.output_dir / "pre_pgrep.txt").write_text(preflight, encoding="utf-8")
    questions = load_questions(args.question_pool, args.suite, args.tier)
    selected_questions = questions[args.start : args.start + args.limit]
    if not selected_questions:
        raise RuntimeError(f"no questions selected for suite={args.suite} tier={args.tier!r}")

    response_dir = args.output_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    beneficiary, verifier = server_specs(args)
    procs: list[subprocess.Popen[str]] = []
    rows: list[dict[str, Any]] = []
    try:
        for spec in (beneficiary, verifier):
            proc = launch_server(spec, args.output_dir / "logs" / f"{spec.name}.server.log")
            procs.append(proc)
            wait_for_health(spec, args.startup_timeout, proc.pid)

        with results_path.open("w", encoding="utf-8") as handle:
            for question in selected_questions:
                started = time.monotonic()
                try:
                    row = run_question(question, args, beneficiary, verifier, response_dir)
                    row["status"] = "ok"
                except Exception as exc:
                    row = {"qid": question.get("id"), "status": "error", "error": str(exc)}
                row["wall_seconds"] = round(time.monotonic() - started, 3)
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    finally:
        for proc in reversed(procs):
            try:
                terminate_server(proc)
            finally:
                log_handle = getattr(proc, "_qwable_verifier_log_handle", None)
                if log_handle is not None:
                    log_handle.close()
        (args.output_dir / "post_pgrep.txt").write_text(process_snapshot(), encoding="utf-8")

    summary = {
        "schema": "qwable_verifier_selector_execute.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(args.output_dir / "plan.json"),
        "results_path": str(results_path),
        "question_ids": [row.get("qid") for row in rows],
        "metrics": summarize_rows([row for row in rows if row.get("status") == "ok"]),
        "plan": plan,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    write_plan(args, plan)
    print("Qwable verifier/selector runner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"server_bin: {SERVER_BIN}")
    print(f"beneficiary_model: {args.beneficiary_model}")
    print(f"verifier_model: {args.verifier_model}")
    print(f"slice: {args.suite} tier={args.tier!r} start={args.start} limit={args.limit}")
    print(f"n_candidates: {args.n_candidates}")
    if not args.execute:
        print(f"Plan written to {args.output_dir / 'plan.json'}")
        print(f"Commands written to {args.output_dir / 'commands.sh'}")
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    sys.exit(main())
