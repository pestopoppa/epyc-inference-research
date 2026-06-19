#!/usr/bin/env python3
"""
Targeted retest of Qwen3.6 questions that scored 0 due to think-loops/empty responses.

Tests only the 16 specific questions that failed on our fork but passed on upstream.
Uses the patched production binary (cherry-picked 56666fa60 + Gemma4 fixes).
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# --- Config ---
MODEL_PATH = "/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf"
SERVER_BIN = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
PORT = 8091
THREADS = 96
CONTEXT = 8192
MAX_TOKENS = 2048  # 512 * 4x multiplier for thinking model (thinking disabled, so budget is adequate)
TEMPERATURE = 0.6
TIMEOUT = 120  # seconds per question (2048 tokens @ ~26 t/s = ~79s max)

PROMPTS_DIR = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/v1")

# The 16 questions that scored 0 on our fork
FAILING_QUESTIONS = [
    # Think tag loops (9)
    ("coder", "t1_q1_algorithm"),
    ("coder", "t1_q2_refactor"),
    ("coder", "t3_q1_concurrent_correctness"),
    ("coder", "t3_q3_algorithmic_hardness"),
    ("general", "t2_q3_schedule"),
    ("math", "t1_q2_system_equations"),
    ("math", "t3_q1_analysis"),
    ("math", "t3_q3_probability_theory"),
    ("thinking", "t3_q2_causal_inference"),
    # Empty responses (7)
    ("instruction_precision", "t1_q2_word_limit"),
    ("instruction_precision", "t3_q1_self_referential"),
    ("instruction_precision", "t3_q2_cascading_constraints"),
    ("instruction_precision", "t3_q3_meta_instruction"),
    ("math", "t3_q2_combinatorics"),
    ("tool_compliance", "t2_q3_search_in_file"),
    ("tool_compliance", "t3_q1_ocr_document"),
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
    """Start llama-server with our patched binary."""
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
        "--reasoning", "auto",
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


def run_question(suite_name: str, question_id: str, prompt: str) -> dict:
    """Send a question to the server and return result."""
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},  # Match upstream benchmark conditions
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
        reasoning = msg.get("reasoning_content", "")

        usage = data.get("usage", {})
        timings = data.get("timings", {})
        tps = timings.get("predicted_per_second", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Client-side TPS fallback
        if not tps and completion_tokens > 0 and elapsed > 0:
            tps = completion_tokens / elapsed

        # Detect degenerate responses
        # With --reasoning auto, real content is in 'content' and thinking in 'reasoning_content'
        # "Empty" means no content AND no reasoning (truly failed)
        # "Budget exhausted" means reasoning exists but no content (model ran out of tokens mid-think)
        is_think_loop = "</think>" in content and content.count("</think>") > 2
        has_reasoning = len(reasoning) > 50 if reasoning else False
        is_empty = len(content.strip()) < 5 and not has_reasoning
        is_budget_exhausted = len(content.strip()) < 5 and has_reasoning

        if is_think_loop:
            status = "think_loop"
        elif is_empty:
            status = "empty"
        elif is_budget_exhausted:
            status = "budget_exhausted"
        else:
            status = "ok"

        return {
            "suite": suite_name,
            "question_id": question_id,
            "status": status,
            "content_length": len(content),
            "reasoning_length": len(reasoning) if reasoning else 0,
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
        print(f"Server ready on port {PORT}. Running {len(FAILING_QUESTIONS)} questions (max_tokens={MAX_TOKENS}).\n", flush=True)
        print(f"{'Suite':<25} {'Question':<30} {'Status':<12} {'TPS':>6} {'Tokens':>7} {'Time':>6}", flush=True)
        print("-" * 90, flush=True)

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
                f"{suite_name:<25} {question_id:<30} {status_display:<12} {tps:>5.1f} {tokens:>7} {elapsed:>5.1f}s",
                flush=True,
            )

        # Summary
        print("\n" + "=" * 90, flush=True)
        print(f"Results: {ok_count}/{len(FAILING_QUESTIONS)} PASS  "
              f"({len(FAILING_QUESTIONS) - ok_count} still failing)", flush=True)

        # Save results
        output_path = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/qwen36_q8_0_retest_fork_fix.json")
        with open(output_path, "w") as f:
            json.dump({
                "model": "Qwen3.6-35B-A3B-Q8_0",
                "binary": SERVER_BIN,
                "cherry_picks": ["56666fa60", "ddf03c6d9", "d7ff074c8", "3fc65063d"],
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
