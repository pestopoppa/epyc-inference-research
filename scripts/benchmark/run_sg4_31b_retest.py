#!/usr/bin/env python3
"""
Targeted retest of SuperGemma4-31b questions that scored 0 due to degenerate repetition/empty responses.

Tests only the 7 specific questions that scored 0 on the baseline benchmark.
Failure modes: repetition collapse (gemma#622), prompt echoing, think loops.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# --- Config ---
MODEL_PATH = "/mnt/raid0/llm/models/SuperGemma4-31b-abliterated.Q4_K_M.gguf"
SERVER_BIN = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
PORT = 8092
THREADS = 96
CONTEXT = 8192
MAX_TOKENS = 2048
TEMPERATURE = 0.6
REPEAT_PENALTY = 1.05  # gemma4 repetition collapse workaround (google-deepmind/gemma#622)
TIMEOUT = 240  # seconds per question (gemma4 is slow ~9 tps, 2048 tokens @ 9 tps = ~227s max)

PROMPTS_DIR = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/v1")

# The 7 questions that scored 0 on SG4-31b baseline
FAILING_QUESTIONS = [
    # Degenerate repetition (4)
    ("instruction_precision", "t1_q3_structured_format"),
    ("instruction_precision", "t1_q4_multiple_constraints"),
    ("instruction_precision", "t2_q1_resist_elaboration"),
    ("math", "t3_q2_combinatorics"),
    # Prompt echoing / empty (3)
    ("math", "t1_q2_system_equations"),
    ("coder", "t2_q2_debug_complex"),
    ("math", "t3_q1_analysis"),
]


def load_question(suite_name: str, question_id: str) -> dict:
    """Load a specific question from the suite YAML."""
    suite_path = PROMPTS_DIR / f"{suite_name}.yaml"
    with open(suite_path) as f:
        suite_data = yaml.safe_load(f)

    prompts = suite_data.get("prompts", {})
    if question_id in prompts:
        q = prompts[question_id]
        q["id"] = question_id
        return q
    raise ValueError(f"Question {question_id} not found in {suite_name}.yaml")


def start_server():
    """Start llama-server for SuperGemma4-31b."""
    cmd = [
        "numactl", "--interleave=all",
        SERVER_BIN,
        "-m", MODEL_PATH,
        "-t", str(THREADS),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-c", str(CONTEXT),
        "--parallel", "1",
        "-ub", "8192",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "--jinja",
    ]
    print(f"Starting server: {' '.join(cmd[:6])}...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_server(timeout=300):
    """Wait for server health endpoint (300s default for large model load)."""
    url = f"http://127.0.0.1:{PORT}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    elapsed = time.time() - start
                    print(f"Server ready in {elapsed:.0f}s", flush=True)
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def detect_degenerate_repetition(text: str) -> bool:
    """Detect degenerate repetition: same 10-char substring repeated 3+ times."""
    if len(text) < 30:
        return False
    for i in range(len(text) - 30):
        substr = text[i:i + 10]
        if substr.strip() == "":
            continue
        count = text.count(substr)
        if count >= 3:
            # Verify it's not just a common word — check that occurrences are spread
            # Find positions of all occurrences
            positions = []
            start = 0
            while True:
                pos = text.find(substr, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            # If 3+ occurrences within a 500-char window, it's degenerate
            for j in range(len(positions) - 2):
                if positions[j + 2] - positions[j] < 500:
                    return True
    return False


def run_question(suite_name: str, question_id: str, prompt: str) -> dict:
    """Send a question to the server and return result."""
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "repeat_penalty": REPEAT_PENALTY,
        "stream": False,
    }

    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=(30, TIMEOUT))
        elapsed = time.time() - start

        if r.status_code != 200:
            return {
                "suite": suite_name,
                "question_id": question_id,
                "status": "error",
                "error": f"HTTP {r.status_code}",
                "elapsed": elapsed,
            }

        data = r.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")

        usage = data.get("usage", {})
        timings = data.get("timings", {})
        tps = timings.get("predicted_per_second", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Client-side TPS fallback
        if not tps and completion_tokens > 0 and elapsed > 0:
            tps = completion_tokens / elapsed

        # Detect degenerate responses
        is_empty = len(content.strip()) < 5
        is_think_loop = "</think>" in content and content.count("</think>") > 2
        is_repetition = detect_degenerate_repetition(content)

        if is_think_loop:
            status = "think_loop"
        elif is_empty:
            status = "empty"
        elif is_repetition:
            status = "repetition"
        else:
            status = "ok"

        return {
            "suite": suite_name,
            "question_id": question_id,
            "status": status,
            "content_length": len(content),
            "tokens_per_second": round(tps, 1),
            "completion_tokens": completion_tokens,
            "elapsed": round(elapsed, 1),
            "content_preview": content[:200] if content else "(empty)",
        }

    except requests.Timeout:
        return {
            "suite": suite_name,
            "question_id": question_id,
            "status": "timeout",
            "elapsed": TIMEOUT,
        }
    except Exception as e:
        return {
            "suite": suite_name,
            "question_id": question_id,
            "status": "error",
            "error": str(e),
            "elapsed": time.time() - start,
        }


def main():
    # Check if server is already running, otherwise start one
    proc = None
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
        if r.status_code == 200:
            print(f"Reusing existing server on port {PORT}.", flush=True)
    except (requests.ConnectionError, requests.Timeout):
        proc = start_server()
        print("Waiting for server to be ready...", flush=True)
        if not wait_for_server():
            print("ERROR: Server failed to start", flush=True)
            proc.kill()
            sys.exit(1)

    try:
        print(f"Server ready on port {PORT}. Running {len(FAILING_QUESTIONS)} questions.", flush=True)
        print(f"  Model: SuperGemma4-31b-abliterated Q4_K_M", flush=True)
        print(f"  repeat_penalty={REPEAT_PENALTY}, max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}", flush=True)
        print(flush=True)
        print(f"{'Suite':<25} {'Question':<35} {'Status':<12} {'TPS':>6} {'Tokens':>7} {'Time':>6}", flush=True)
        print("-" * 95, flush=True)

        results = []
        ok_count = 0
        for i, (suite_name, question_id) in enumerate(FAILING_QUESTIONS, 1):
            q = load_question(suite_name, question_id)
            prompt = q["prompt"]

            result = run_question(suite_name, question_id, prompt)
            results.append(result)

            status = result["status"]
            tps = result.get("tokens_per_second", 0)
            tokens = result.get("completion_tokens", 0)
            elapsed = result.get("elapsed", 0)

            status_display = "PASS" if status == "ok" else status.upper()
            if status == "ok":
                ok_count += 1

            print(
                f"{suite_name:<25} {question_id:<35} {status_display:<12} {tps:>5.1f} {tokens:>7} {elapsed:>5.1f}s",
                flush=True,
            )

        # Summary
        print("\n" + "=" * 95, flush=True)
        print(f"Results: {ok_count}/{len(FAILING_QUESTIONS)} PASS  "
              f"({len(FAILING_QUESTIONS) - ok_count} still failing)", flush=True)

        # Save results
        output_path = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/sg4_31b_retest_gemma4_fixes.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "model": "SuperGemma4-31b-abliterated-Q4_K_M",
                "model_path": MODEL_PATH,
                "binary": SERVER_BIN,
                "port": PORT,
                "params": {
                    "repeat_penalty": REPEAT_PENALTY,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "kv_cache": "q8_0",
                    "jinja": True,
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "summary": f"{ok_count}/{len(FAILING_QUESTIONS)} pass",
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}", flush=True)

        # Print content previews for any non-ok
        failures = [r for r in results if r["status"] != "ok"]
        if failures:
            print(f"\n--- Failures ({len(failures)}) ---", flush=True)
            for r in failures:
                print(f"\n  {r['suite']}/{r['question_id']}: {r['status']}", flush=True)
                if "content_preview" in r:
                    print(f"  Preview: {r['content_preview'][:100]}", flush=True)

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("\nServer stopped.", flush=True)
        else:
            print("\n(Server left running.)", flush=True)


if __name__ == "__main__":
    main()
