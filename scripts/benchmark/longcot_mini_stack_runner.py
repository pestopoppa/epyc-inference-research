#!/usr/bin/env python3
"""Run LongCoT-Mini against already-resident optimized stack ports.

This runner is intentionally narrow for RE-4/K-LCM-1. It does not start or stop
model servers. Instead it verifies provided llama-server ports and fans out a
single role across those ports, preserving the existing ``run_benchmark.py``
result JSON shape so ``score_longcot_run.py`` remains the deterministic scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from suites import Question, load_suite  # noqa: E402

SUITE_NAME = "longcot_mini"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_role_ports(items: Iterable[str]) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"role ports must be role=port[,port...], got {item!r}")
        role, raw_ports = item.split("=", 1)
        role = role.strip()
        ports = [int(p.strip()) for p in raw_ports.split(",") if p.strip()]
        if not role or not ports:
            raise ValueError(f"invalid role port mapping {item!r}")
        roles[role] = ports
    return roles


def _port_open(port: int, host: str = "127.0.0.1", timeout_s: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _http_json(url: str, payload: dict[str, Any] | None, timeout_s: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _health_ok(port: int, host: str = "127.0.0.1") -> bool:
    if not _port_open(port, host=host):
        return False
    try:
        data = _http_json(f"http://{host}:{port}/health", None, timeout_s=2)
    except Exception:
        return False
    return data.get("status") == "ok"


def _completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    content = data.get("content")
    return content if isinstance(content, str) else ""


def _run_question(
    *,
    host: str,
    port: int,
    role: str,
    question: Question,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    endpoint: str,
    disable_thinking: bool,
) -> tuple[str, dict[str, Any]]:
    start = time.time()
    url = (
        f"http://{host}:{port}/v1/chat/completions"
        if endpoint == "chat"
        else f"http://{host}:{port}/completion"
    )
    if endpoint == "chat":
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": question.prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        payload = {
            "prompt": question.prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
            "cache_prompt": False,
        }

    row: dict[str, Any] = {
        "question_id": question.id,
        "prompt": question.prompt,
        "expected": question.expected,
        "port": port,
        "role": role,
        "endpoint": endpoint,
    }
    try:
        data = _http_json(url, payload, timeout_s=timeout_s)
        elapsed_s = time.time() - start
        text = _completion_text(data)
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        completion_tokens = (
            usage.get("completion_tokens")
            or timings.get("predicted_n")
            or len(text.split())
        )
        prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
        tps = timings.get("predicted_per_second")
        if not tps and completion_tokens and elapsed_s > 0:
            tps = float(completion_tokens) / elapsed_s
        row.update(
            {
                "response": text,
                "success": True,
                "elapsed_s": elapsed_s,
                "tokens_per_second": tps,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_usage": usage,
            }
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        row.update(
            {
                "response": "",
                "success": False,
                "elapsed_s": time.time() - start,
                "tokens_per_second": None,
                "error": str(exc),
            }
        )
    return question.id, row


def _write_role_result(
    path: Path,
    *,
    run_id: str,
    role: str,
    rows: dict[str, dict[str, Any]],
    model_path: str | None = None,
    started_at: str | None = None,
) -> None:
    tps = [
        float(row["tokens_per_second"])
        for row in rows.values()
        if row.get("tokens_per_second") is not None
    ]
    payload = {
        "model_role": role,
        "model_path": model_path,
        "config_name": "baseline",
        "run_id": run_id,
        "timestamp": started_at or _utcnow_iso(),
        "summary": {
            "avg_tokens_per_second": (sum(tps) / len(tps)) if tps else None,
            "avg_algorithmic_score": None,
            "questions_tested": len(rows),
            "questions_passed": 0,
        },
        "results": {SUITE_NAME: dict(sorted(rows.items()))},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_role(
    *,
    role: str,
    ports: list[int],
    questions: list[Question],
    output_path: Path,
    run_id: str,
    host: str = "127.0.0.1",
    endpoint: str = "chat",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    timeout_s: int = 900,
    disable_thinking: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    unhealthy = [p for p in ports if not _health_ok(p, host=host)]
    if unhealthy:
        raise RuntimeError(f"{role}: unhealthy stack ports: {unhealthy}")

    started_at = _utcnow_iso()
    rows: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        rows.update(existing.get("results", {}).get(SUITE_NAME, {}))

    pending = [q for q in questions if q.id not in rows]
    with cf.ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures: dict[cf.Future[tuple[str, dict[str, Any]]], Question] = {}
        for idx, question in enumerate(pending):
            port = ports[idx % len(ports)]
            futures[
                pool.submit(
                    _run_question,
                    host=host,
                    port=port,
                    role=role,
                    question=question,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    endpoint=endpoint,
                    disable_thinking=disable_thinking,
                )
            ] = question
        for fut in cf.as_completed(futures):
            qid, row = fut.result()
            rows[qid] = row
            _write_role_result(output_path, run_id=run_id, role=role, rows=rows, started_at=started_at)

    _write_role_result(output_path, run_id=run_id, role=role, rows=rows, started_at=started_at)
    failures = [qid for qid, row in rows.items() if not row.get("success", True)]
    return {
        "role": role,
        "ports": ports,
        "output_path": str(output_path),
        "questions": len(questions),
        "rows": len(rows),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role-ports", action="append", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results/runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint", choices=("chat", "completion"), default="chat")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-thinking", action="store_true")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)

    suite = load_suite(SUITE_NAME)
    if suite is None:
        raise SystemExit(f"missing suite {SUITE_NAME}")
    questions = list(suite.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    role_ports = _parse_role_ports(args.role_ports)

    run_root = Path(args.output_dir) / args.run_id
    summary = {
        "schema_version": "longcot_mini_stack_runner.v1",
        "run_id": args.run_id,
        "suite": SUITE_NAME,
        "started_at": _utcnow_iso(),
        "endpoint": args.endpoint,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "roles": [],
    }
    exit_code = 0
    for role, ports in role_ports.items():
        output_path = run_root / f"{role}_baseline.json"
        role_summary = run_role(
            role=role,
            ports=ports,
            questions=questions,
            output_path=output_path,
            run_id=args.run_id,
            host=args.host,
            endpoint=args.endpoint,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
            disable_thinking=not args.allow_thinking,
            resume=not args.no_resume,
        )
        summary["roles"].append(role_summary)
        if role_summary["failures"]:
            exit_code = 2
    summary["ended_at"] = _utcnow_iso()
    summary_path = Path(args.summary_out) if args.summary_out else run_root / "stack_runner_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
