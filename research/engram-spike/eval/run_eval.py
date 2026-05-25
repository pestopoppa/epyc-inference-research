#!/usr/bin/env python3
"""Head-to-head sentinel eval for the engram-conditional-memory Phase 5
quality gate.

Hits an OpenAI-compatible /v1/chat/completions endpoint per question. Scores
deterministically using the scoring_method declared on each sentinel question.
Saves per-question results to JSON for audit.

Usage:
    run_eval.py --url http://127.0.0.1:8500/v1/chat/completions \\
                --model longcat \\
                --questions /mnt/raid0/llm/epyc-orchestrator/scripts/autopilot/sentinel_questions.yaml \\
                --out longcat-results.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
import yaml


def score_substring(answer: str, expected: str, config: dict) -> bool:
    case_sensitive = config.get("case_sensitive", False)
    if case_sensitive:
        return expected in answer
    return expected.lower() in answer.lower()


def score_exact_match(answer: str, expected: str, config: dict) -> bool:
    case_sensitive = config.get("case_sensitive", False)
    a = answer.strip()
    e = expected.strip()
    if not case_sensitive:
        a, e = a.lower(), e.lower()
    return a == e


def score_multiple_choice(answer: str, expected: str, config: dict) -> bool:
    expected_letter = expected.strip().upper()
    if len(expected_letter) != 1 or not expected_letter.isalpha():
        return False
    pattern = rf"(?:^|[^A-Za-z]){re.escape(expected_letter)}(?:[^A-Za-z]|$)"
    return bool(re.search(pattern, answer))


def score_f1(answer: str, expected: str, config: dict) -> bool:
    threshold = config.get("threshold", 0.5)
    a_toks = set(re.findall(r"\w+", answer.lower()))
    e_toks = set(re.findall(r"\w+", expected.lower()))
    if not e_toks or not a_toks:
        return False
    overlap = len(a_toks & e_toks)
    if overlap == 0:
        return False
    p = overlap / len(a_toks)
    r = overlap / len(e_toks)
    f1 = 2 * p * r / (p + r)
    return f1 >= threshold


SCORERS = {
    "substring": score_substring,
    "exact_match": score_exact_match,
    "multiple_choice": score_multiple_choice,
    "f1": score_f1,
}


def call_model(url, prompt, max_tokens=512, timeout=240.0):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return ("", elapsed, f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        answer = data["choices"][0]["message"]["content"]
        return (answer, elapsed, None)
    except Exception as e:
        elapsed = time.time() - t0
        return ("", elapsed, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text())
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"[{args.model}] loaded {len(questions)} questions, posting to {args.url}", flush=True)

    results = []
    by_suite_c, by_suite_t = {}, {}
    overall_c = overall_t = 0

    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        suite = q.get("suite", "unknown")
        prompt = q.get("prompt", "")
        expected = q.get("expected", "")
        method = q.get("scoring_method", "substring")
        config = q.get("scoring_config", {}) or {}

        answer, elapsed, error = call_model(args.url, prompt, args.max_tokens, args.timeout)

        if error:
            correct = False
        else:
            scorer = SCORERS.get(method, score_substring)
            try:
                correct = scorer(answer, expected, config)
            except Exception as e:
                correct = False
                error = f"score_err: {e}"

        results.append({
            "id": qid, "suite": suite, "method": method, "expected": expected,
            "answer": answer, "correct": correct, "elapsed_s": round(elapsed, 2),
            "error": error,
        })

        by_suite_c.setdefault(suite, 0)
        by_suite_t.setdefault(suite, 0)
        by_suite_t[suite] += 1
        if correct:
            by_suite_c[suite] += 1
            overall_c += 1
        overall_t += 1

        mark = "✓" if correct else "✗"
        status = f"err={error[:60]}" if error else "ok"
        print(f"  [{i+1:>2}/{len(questions)}] {mark} {qid:<40} ({elapsed:5.1f}s) {status}", flush=True)

    summary = {
        "model": args.model, "url": args.url, "total": overall_t, "correct": overall_c,
        "accuracy": overall_c / max(1, overall_t),
        "by_suite": {s: {"correct": by_suite_c[s], "total": by_suite_t[s], "accuracy": by_suite_c[s] / by_suite_t[s]} for s in sorted(by_suite_t)},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(flush=True)
    print(f"=== [{args.model}] SUMMARY ===", flush=True)
    print(f"  Overall: {overall_c}/{overall_t} = {summary['accuracy']:.1%}", flush=True)
    print(f"  By suite:", flush=True)
    for s, stats in summary["by_suite"].items():
        print(f"    {s:<24} {stats['correct']:>2}/{stats['total']:<2} = {stats['accuracy']:.1%}", flush=True)
    print(f"  Saved: {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
