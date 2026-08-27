#!/usr/bin/env python3
"""Memento S2 MATH-500 delta: base vs base+LoRA accuracy on a seeded subset.

Runs ONE llama-server process per arm (CPU), asks each question via
/v1/chat/completions (greedy), extracts the final answer (\\boxed{...} else last
line), normalizes, and exact-matches against the dataset reference. Writes a
JSONL per question + a summary line.

Usage:
  python3 memento_math500_eval.py --questions math_500.jsonl --n 50 --seed 4242 \
      --model /mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf \
      [--lora /mnt/raid0/llm/epyc-inference-research/output/memento/memento-final/memento-lora.gguf] \
      --port 8457 --out results_base.jsonl
"""

import argparse
import json
import random
import re
import subprocess
import time
import urllib.request

BASE_URL = "http://127.0.0.1:{port}/v1/chat/completions"


def extract_answer(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return normalize(m.group(1))
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return normalize(lines[-1]) if lines else ""


def normalize(s: str) -> str:
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace("$", "").strip()
    s = re.sub(r"\s+", "", s)
    return s


def ask(port: int, problem: str, max_tokens: int = 512) -> str:
    body = json.dumps({
        "messages": [{"role": "user", "content": f"Solve the following problem. Show your reasoning, then state the final answer on its own line or in a \\boxed{{}}.\n\n{problem}"}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Qwen3 thinking mode leaves content empty until the think block closes;
        # for answer evaluation we disable it (verified 2026-08-27).
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE_URL.format(port=port), data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--port", type=int, default=8457)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in open(args.questions)]
    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.n, len(rows)))

    cmd = ["/mnt/raid0/llm/llama.cpp/build/bin/llama-server",
           "-m", args.model, "-c", "4096", "--temp", "0",
           "--port", str(args.port), "--host", "127.0.0.1"]
    if args.lora:
        cmd += ["-l", args.lora]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        # wait for readiness
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health",
                                       timeout=3)
                break
            except Exception:
                time.sleep(2)
        correct = 0
        with open(args.out, "w") as fh:
            for i, q in enumerate(picked):
                text = ask(args.port, q["data"])
                got = extract_answer(text)
                ref = normalize(q["answer"])
                ok = got == ref
                correct += int(ok)
                fh.write(json.dumps({
                    "i": i, "unique_id": q.get("unique_id", i),
                    "correct": ok, "got": got, "ref": ref,
                }) + "\n")
                fh.flush()
                print(f"[{i+1}/{len(picked)}] {'OK ' if ok else 'X  '} "
                      f"got={got[:40]!r} ref={ref[:40]!r}", flush=True)
        print(f"ACCURACY: {correct}/{len(picked)} = {correct/len(picked):.3f}")
    finally:
        srv.terminate()
        for _ in range(30):
            if srv.poll() is not None:
                break
            time.sleep(1)
        if srv.poll() is None:
            srv.kill()


if __name__ == "__main__":
    main()
