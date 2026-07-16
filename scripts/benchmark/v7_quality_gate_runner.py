#!/usr/bin/env python3
"""V7 quality-gate runner: evaluate MMLU-Pro + GPQA-Diamond on a kernel.

Samples questions from registered suites, queries a llama-server instance,
scores multiple-choice answers, and writes per-suite accuracy JSON ready for
v7_quality_gate_compare.py.

Usage:
    v7_quality_gate_runner.py --port 18072 --output results.json \
        --suites mmlu_pro gpqa --n 200 --seed 42 --endpoint chat

Output JSON shape:
    {
      "meta": {"kernel": "v7-experimental", "binary": "...", "models": "...", "timestamp": "..."},
      "suites": [
        {"suite": "mmlu_pro", "accuracy": 0.82, "n": 200, "correct": 164,
         "per_tier": {"1": {"accuracy": 0.85, "n": 50, "correct": 42}, ...}},
        {"suite": "gpqa", "accuracy": 0.63, "n": 100, "correct": 63, ...}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def wait_for_server(url: str, timeout: int = 120) -> None:
    """Wait for llama-server /health to return ok."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode().strip().lower()
                if "ok" in body:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server at {url} did not become healthy within {timeout}s")


def query_server(
    url: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 42,
    endpoint: str = "chat",
) -> str:
    """Query llama-server and return the response text."""
    if endpoint == "chat":
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        request_path = "/v1/chat/completions"
    elif endpoint == "completion":
        payload = {
            "model": "",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "top_k": 1,
            "logprobs": 0,
        }
        request_path = "/v1/completions"
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")

    req = urllib.request.Request(
        f"{url}{request_path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            choices = result.get("choices", [])
            if not choices:
                return ""
            choice = choices[0]
            if endpoint == "chat":
                message = choice.get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content or "")
            return choice.get("text", "")
    except Exception as e:
        print(f"  [runner] query failed: {e}", file=sys.stderr)
        return ""


def extract_letter_answer(response: str) -> str:
    """Extract a single letter (A-J) from the model's response."""
    stripped = response.strip()

    # Prefer explicit answer markers over arbitrary standalone letters.
    match = re.search(
        r'\b(?:answer|option|choice|letter)\s*(?:is|:|=|\.|-)?\s*\(?([A-Ja-j])\)?',
        stripped,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Accept terse responses like "C" or "C.".
    match = re.fullmatch(r'\(?([A-Ja-j])\)?[.)]?', stripped)
    if match:
        return match.group(1).upper()

    # Fall back only when there is exactly one candidate letter in the response.
    matches = re.findall(r'\b([A-Ja-j])\b', stripped)
    if len(matches) == 1:
        return matches[0].upper()
    return ""


def run_suite(
    suite_name: str,
    url: str,
    n: int,
    seed: int,
    stratify: bool = False,
    max_tokens: int = 64,
    endpoint: str = "chat",
) -> dict:
    """Run eval on a single suite and return per-suite results."""
    # Import the adapter
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dataset_adapters import get_adapter
    except ImportError:
        # Try orchestrator path
        sys.path.insert(
            0,
            str(Path(__file__).resolve().parent.parent / "benchmark"),
        )
        from dataset_adapter_modules.registry import get_adapter

    adapter = get_adapter(suite_name)
    if adapter is None:
        return {"suite": suite_name, "accuracy": 0, "n": 0, "correct": 0,
                "error": f"no adapter for suite {suite_name}"}

    # Sample questions
    questions = adapter.sample(n=n, seed=seed, stratify=stratify)
    if not questions:
        return {"suite": suite_name, "accuracy": 0, "n": 0, "correct": 0,
                "error": "no questions sampled"}

    correct = 0
    total = 0
    errors = 0
    per_tier: dict[str, dict] = {}

    for i, q in enumerate(questions):
        expected = q.get("expected", "").upper().strip()
        if not expected:
            continue

        tier = str(q.get("tier", 2))
        if tier not in per_tier:
            per_tier[tier] = {"correct": 0, "n": 0}

        per_tier[tier]["n"] += 1
        total += 1

        response = query_server(
            url,
            q["prompt"],
            max_tokens=max_tokens,
            seed=seed,
            endpoint=endpoint,
        )
        if not response:
            errors += 1
            continue

        answer = extract_letter_answer(response)
        if answer == expected:
            correct += 1
            per_tier[tier]["correct"] += 1

        if (i + 1) % 25 == 0:
            print(f"  [runner] {suite_name}: {i+1}/{len(questions)} "
                  f"({correct}/{i+1} correct so far)", file=sys.stderr)

    accuracy = correct / total if total > 0 else 0.0

    tier_results = {}
    for t, data in sorted(per_tier.items()):
        tier_results[t] = {
            "accuracy": data["correct"] / data["n"] if data["n"] > 0 else 0,
            "n": data["n"],
            "correct": data["correct"],
        }

    return {
        "suite": suite_name,
        "accuracy": accuracy,
        "n": total,
        "correct": correct,
        "errors": errors,
        "per_tier": tier_results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V7 quality-gate runner")
    p.add_argument("--port", type=int, default=18072,
                   help="llama-server port (default: 18072)")
    p.add_argument("--host", default="localhost",
                   help="llama-server host (default: localhost)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path")
    p.add_argument("--suites", nargs="+", default=["mmlu_pro", "gpqa"],
                   help="Suites to evaluate (default: mmlu_pro gpqa)")
    p.add_argument("--n", type=int, default=200,
                   help="Questions per suite (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    p.add_argument("--stratify", action="store_true",
                   help="Use stratified sampling (equal per tier)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Max tokens for model response (default: 64)")
    p.add_argument("--endpoint", choices=["chat", "completion"], default="chat",
                   help="llama-server API endpoint mode (default: chat)")
    p.add_argument("--kernel", default="v7-candidate",
                   help="Kernel label for output metadata")
    p.add_argument("--binary", default="",
                   help="Binary path for output metadata")
    p.add_argument("--models", default="",
                   help="Model path(s) for output metadata")
    p.add_argument("--timeout", type=int, default=120,
                   help="Server health check timeout (seconds)")
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"[runner] Waiting for server at {url}...", file=sys.stderr)
    wait_for_server(url, timeout=args.timeout)
    print(f"[runner] Server healthy", file=sys.stderr)

    # Determine binary path
    binary = args.binary or os.environ.get("LLAMA_BINARY", "")

    suites_results = []
    start = time.monotonic()

    for suite in args.suites:
        print(f"\n[runner] Evaluating {suite} (n={args.n}, seed={args.seed})...",
              file=sys.stderr)
        result = run_suite(
            suite, url, args.n, args.seed,
            stratify=args.stratify,
            max_tokens=args.max_tokens,
            endpoint=args.endpoint,
        )
        suites_results.append(result)
        acc = result.get("accuracy", 0)
        n = result.get("n", 0)
        print(f"[runner] {suite}: {acc:.1%} ({result.get('correct',0)}/{n})",
              file=sys.stderr)

    elapsed = time.monotonic() - start

    output = {
        "meta": {
            "kernel": args.kernel,
            "binary": binary,
            "models": args.models or "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "n_per_suite": args.n,
            "seed": args.seed,
            "stratify": args.stratify,
            "endpoint": args.endpoint,
        },
        "suites": suites_results,
    }

    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n[runner] Results written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
