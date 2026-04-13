#!/usr/bin/env python3
"""SEAL multi-role A/B test — baseline vs control vector across all orchestrator roles.

Tests each role-model with and without its SEAL concise reasoning cvector.
Measures: token reduction %, answer correctness, generation speed.

Usage:
    python seal_multi_role_test.py
"""

import subprocess, os, time, requests, json, sys

ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = "/mnt/raid0/llm/llama.cpp/build/bin:/usr/lib/llvm-20/lib"
SERVER = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

ROLES = [
    {
        "name": "worker",
        "model": "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "cvector": "/mnt/raid0/llm/models/qwen3-coder-30b-seal-concise.gguf",
        "threads": 48,
    },
    {
        "name": "coder",
        "model": "/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf",
        "cvector": "/mnt/raid0/llm/models/qwen2.5-coder-32b-seal-concise.gguf",
        "threads": 96,
    },
    # frontdoor (Qwen3.5 SSM): BLOCKED — cvector-generator can't load SSM models
    # reap (246B): cvector training takes very long, test separately if needed
]

PROBLEMS = [
    ("What is the sum of the first 10 prime numbers?", "129"),
    ("Solve for x: 3x^2 - 12x + 9 = 0", ["1", "3"]),
    ("How many ways can you arrange the letters in MISSISSIPPI?", "34650"),
    ("What is the remainder when 2^100 is divided by 7?", "4"),
    ("Find the area of a triangle with sides 5, 12, and 13.", "30"),
    ("Write a Python function to check if a string is a palindrome.", "def"),
    ("What is the derivative of x^3 + 2x^2 - 5x + 7?", "3x"),
]


def check_answer(text, expected):
    t = text.lower().replace(",", "")
    if isinstance(expected, list):
        return all(e in t for e in expected)
    return expected.lower() in t


def generate(port, prompt, max_tokens=400):
    resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "m",
        "messages": [
            {"role": "system", "content": "You are a math and coding expert. Solve problems accurately."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }, timeout=300)
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return content, tokens


def start_server(model, port, threads, cvec=None):
    args = [SERVER, "-m", model, "-t", str(threads), "-c", "2048",
            "--port", str(port), "--host", "127.0.0.1", "--no-warmup", "-np", "1"]
    if cvec and os.path.exists(cvec):
        args += ["--control-vector-scaled", f"{cvec}:0.5"]
    proc = subprocess.Popen(args, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(25)
    return proc


def test_role(role):
    name = role["name"]
    model = role["model"]
    cvec = role["cvector"]
    threads = role["threads"]

    if not os.path.exists(cvec):
        return {"name": name, "status": "SKIPPED", "reason": f"cvector not found: {cvec}"}

    print(f"\n{'='*60}")
    print(f"Testing role: {name}")
    print(f"  Model: {os.path.basename(model)}")
    print(f"  CVec:  {os.path.basename(cvec)}")

    results = {"name": name, "problems": []}

    # Baseline
    print(f"  Starting baseline server...")
    proc = start_server(model, 9095, threads)
    try:
        requests.get("http://127.0.0.1:9095/health", timeout=5)
    except Exception:
        proc.terminate()
        return {"name": name, "status": "FAILED", "reason": "baseline server didn't start"}

    base_correct = 0
    base_tokens = 0
    for question, expected in PROBLEMS:
        text, tok = generate(9095, question)
        ok = check_answer(text, expected)
        base_correct += int(ok)
        base_tokens += tok
        results["problems"].append({"q": question[:50], "base_tok": tok, "base_ok": ok})

    proc.terminate()
    proc.wait()
    time.sleep(3)

    # With cvector
    print(f"  Starting cvector server (scale=0.5)...")
    proc = start_server(model, 9095, threads, cvec)
    try:
        requests.get("http://127.0.0.1:9095/health", timeout=5)
    except Exception:
        proc.terminate()
        return {"name": name, "status": "FAILED", "reason": "cvector server didn't start"}

    cv_correct = 0
    cv_tokens = 0
    for i, (question, expected) in enumerate(PROBLEMS):
        text, tok = generate(9095, question)
        ok = check_answer(text, expected)
        cv_correct += int(ok)
        cv_tokens += tok
        results["problems"][i]["cv_tok"] = tok
        results["problems"][i]["cv_ok"] = ok

    proc.terminate()
    proc.wait()
    time.sleep(3)

    delta = (cv_tokens - base_tokens) / max(base_tokens, 1) * 100
    results["base_correct"] = base_correct
    results["cv_correct"] = cv_correct
    results["base_tokens"] = base_tokens
    results["cv_tokens"] = cv_tokens
    results["delta_pct"] = delta
    results["status"] = "OK"

    # Print summary for this role
    print(f"\n  {'Problem':<42} {'Base':>6} {'CV':>6} {'B✓':>4} {'C✓':>4}")
    print(f"  {'-'*65}")
    for p in results["problems"]:
        b_s = "PASS" if p["base_ok"] else "FAIL"
        c_s = "PASS" if p["cv_ok"] else "FAIL"
        print(f"  {p['q']:<42} {p['base_tok']:>6} {p['cv_tok']:>6} {b_s:>4} {c_s:>4}")
    print(f"  {'-'*65}")
    print(f"  TOTAL: base={base_tokens} tok ({base_correct}/{len(PROBLEMS)}), cv={cv_tokens} tok ({cv_correct}/{len(PROBLEMS)}), Δ={delta:+.1f}%")
    regression = cv_correct < base_correct
    print(f"  Regression: {'YES — accuracy dropped!' if regression else 'NO'}")

    return results


if __name__ == "__main__":
    print("SEAL Multi-Role Regression Test")
    print("=" * 60)

    all_results = []
    for role in ROLES:
        result = test_role(role)
        all_results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print(f"{'Role':<15} {'Base Acc':>10} {'CV Acc':>10} {'Tokens Δ':>10} {'Regression':>12}")
    print("-" * 60)
    for r in all_results:
        if r.get("status") == "OK":
            n = len(PROBLEMS)
            print(f"{r['name']:<15} {r['base_correct']}/{n:>8} {r['cv_correct']}/{n:>8} {r['delta_pct']:>+9.1f}% {'YES!' if r['cv_correct'] < r['base_correct'] else 'NO':>12}")
        else:
            print(f"{r['name']:<15} {r.get('status','?'):>10} — {r.get('reason','')}")

    out = "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/runs/seal-multi-role.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")
