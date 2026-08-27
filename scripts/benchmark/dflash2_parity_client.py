#!/usr/bin/env python3
"""Exact-token greedy capture client for the DF2-6 parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "epyc.df2.greedy_parity_capture.v1"


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=canonical(payload), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("completion response is not one JSON object")
    return value


def run(args) -> None:
    out = Path(args.output)
    if out.exists() or Path(args.per_question_out).exists():
        raise RuntimeError("parity output already exists")
    questions = json.loads(Path(args.questions_in).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 12:
        raise RuntimeError("parity requires exactly 12 pinned questions")
    rows = []
    started = time.monotonic()
    for index, question in enumerate(questions):
        payload = {
            "prompt": question["prompt"], "n_predict": 256,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
            "typical_p": 1.0, "repeat_penalty": 1.0, "seed": 42,
            "cache_prompt": False, "return_tokens": True, "stream": False,
            "samplers": ["top_k", "temperature"],
        }
        request_sha = hashlib.sha256(canonical(payload)).hexdigest()
        began = time.monotonic()
        response = post(f"http://{args.host}:{args.port}/completion", payload, args.timeout)
        elapsed = time.monotonic() - began
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(value) is not int for value in tokens):
            raise RuntimeError(f"question {index}: missing exact integer token sequence")
        content = response.get("content")
        timings = response.get("timings")
        if not isinstance(content, str) or not isinstance(timings, dict) or not isinstance(
                timings.get("predicted_ms"), (int, float)):
            raise RuntimeError(f"question {index}: malformed completion response")
        row = {
            "schema": SCHEMA, "arm": args.arm, "index": index, "id": question["id"],
            "suite": question["suite"],
            "prompt_fingerprint": hashlib.sha256(question["prompt"].encode()).hexdigest(),
            "request": payload, "request_sha256": request_sha, "tokens": tokens,
            "tokens_sha256": hashlib.sha256(canonical(tokens)).hexdigest(),
            "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "stop_type": response.get("stop_type"), "truncated": response.get("truncated"),
            "predicted_n": timings.get("predicted_n"), "predicted_ms": timings["predicted_ms"],
            "predicted_per_second": timings.get("predicted_per_second"), "wall_s": elapsed,
        }
        rows.append(row)
        with Path(args.per_question_out).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(Path(args.live_status_out), {
            "schema": SCHEMA, "arm": args.arm, "completed_draws": len(rows),
            "expected_draws": 12, "complete": len(rows) == 12,
            "request_error_rows": 0, "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    predicted_ms = sum(float(row["predicted_ms"]) for row in rows)
    n_tokens = sum(len(row["tokens"]) for row in rows)
    aggregate = n_tokens / (predicted_ms / 1000.0)
    atomic_json(out, {
        "meta": {"schema": SCHEMA, "arm": args.arm, "kernel": args.kernel,
                 "binary": args.binary, "models": args.models,
                 "questions_pinned": args.questions_in, "temperature": 0.0,
                 "seed": 42, "max_tokens": 256, "endpoint": "completion",
                 "return_tokens": True},
        "suites": [{"suite": "olympiadbench_hard", "n": 12,
                    "throughput": {"concurrency": 1, "wall_s": time.monotonic() - started,
                                   "completion_tokens": n_tokens,
                                   "aggregate_decode_tok_s": round(aggregate, 1),
                                   "aggregate_total_tok_s": round(aggregate, 1)}}],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-question-out", required=True)
    parser.add_argument("--live-status-out", required=True)
    parser.add_argument("--questions-in", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    run(parser.parse_args())
