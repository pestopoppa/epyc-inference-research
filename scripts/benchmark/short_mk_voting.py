#!/usr/bin/env python3
from __future__ import annotations

"""Run short-m@k majority-vote evaluations against a llama-server."""

import argparse
import concurrent.futures
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_POOL_PATH = PROJECT_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "benchmarks" / "short_mk_voting"

sys.path.insert(0, str(BENCHMARK_DIR))

from debug_scorer import score_answer
from question_pool import load_pool, sample_from_pool


@dataclass(frozen=True)
class Completion:
    index: int
    text: str
    completion_tokens: int
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class Vote:
    answer: str
    vote_key: str
    count: int
    completion_index: int
    completion_tokens: int
    correct: bool


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _normalize_vote_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().rstrip(".")).strip()


def _extract_by_pattern(answer: str, pattern: str) -> str | None:
    try:
        match = re.search(pattern, answer, flags=re.DOTALL)
    except re.error:
        return None
    if not match:
        return None
    return match.group(1 if match.groups() else 0).strip()


def _extract_multiple_choice(answer: str) -> str:
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-H])\)?(?![a-zA-Z])",
        r"^\s*\(?([A-H])\)?\s*$",
        r"\*\*([A-H])\*\*",
        r"\b([A-H])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return str(matches[-1]).upper()
    match = re.match(r"\s*\(?([A-H])\)?\s*[.:\-\n]", answer)
    if match:
        return match.group(1).upper()
    return ""


def extract_vote_key(answer: str, question: dict[str, Any]) -> str:
    clean = strip_think_blocks(answer)
    method = question.get("scoring_method", "exact_match")
    config = question.get("scoring_config") or {}

    if method == "multiple_choice":
        return _extract_multiple_choice(clean)

    if method == "exact_match":
        patterns = [
            config.get("extract_pattern", r"<answer>(.*?)</answer>"),
            r"####[ \t]*\n?(\S+)",
            r"\\boxed\{([^{}]+)\}",
        ]
        for pattern in patterns:
            extracted = _extract_by_pattern(clean, pattern)
            if extracted:
                return _normalize_vote_key(extracted)

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    return _normalize_vote_key(lines[-1] if lines else clean)


def majority_vote(question: dict[str, Any], completions: list[Completion]) -> Vote:
    successful = [item for item in completions if item.error is None and item.text.strip()]
    if not successful:
        raise ValueError("no successful completions to vote on")

    keyed: dict[str, list[Completion]] = defaultdict(list)
    for completion in successful:
        key = extract_vote_key(completion.text, question)
        if key:
            keyed[key].append(completion)

    if not keyed:
        keyed[_normalize_vote_key(strip_think_blocks(successful[0].text))].append(successful[0])

    counts = Counter({key: len(items) for key, items in keyed.items()})
    best_count = max(counts.values())
    candidate_keys = [key for key, count in counts.items() if count == best_count]

    def tie_key(key: str) -> tuple[int, int, int]:
        best_completion = min(keyed[key], key=lambda item: (item.completion_tokens, len(item.text), item.index))
        return (best_completion.completion_tokens, len(best_completion.text), best_completion.index)

    winner_key = min(candidate_keys, key=tie_key)
    winner_completion = min(keyed[winner_key], key=lambda item: (item.completion_tokens, len(item.text), item.index))
    correct = score_answer(
        winner_completion.text,
        expected=question.get("expected", ""),
        scoring_method=question.get("scoring_method", "exact_match"),
        scoring_config=question.get("scoring_config"),
    )
    return Vote(
        answer=strip_think_blocks(winner_completion.text),
        vote_key=winner_key,
        count=counts[winner_key],
        completion_index=winner_completion.index,
        completion_tokens=winner_completion.completion_tokens,
        correct=correct,
    )


def build_prompt(question: dict[str, Any]) -> str:
    context = str(question.get("context") or "").strip()
    prompt = str(question.get("prompt") or "").strip()
    if context:
        return f"{context}\n\n{prompt}"
    return prompt


def build_questions(pool_path: Path, suites: list[str], sample_per_suite: int, seed: int) -> list[dict[str, Any]]:
    pool = load_pool(pool_path)
    questions = sample_from_pool(pool, suites=suites, sample_per_suite=sample_per_suite, seed=seed)
    if not questions:
        raise ValueError(f"no questions sampled from {pool_path}")
    return questions


def generate_response(
    prompt: str,
    *,
    host: str,
    port: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    index: int,
) -> Completion:
    import httpx

    started = time.monotonic()
    try:
        response = httpx.post(
            f"http://{host}:{port}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        text = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        if reasoning:
            text = f"<think>\n{reasoning}\n</think>\n{text}"
        usage = data.get("usage", {})
        return Completion(
            index=index,
            text=text,
            completion_tokens=int(usage.get("completion_tokens", max(1, len(text) // 4))),
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return Completion(
            index=index,
            text="",
            completion_tokens=0,
            elapsed_seconds=time.monotonic() - started,
            error=str(exc),
        )


def run_question(question: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = build_prompt(question)
    completions: list[Completion] = []

    if args.parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.k) as executor:
            futures = [
                executor.submit(
                    generate_response,
                    prompt,
                    host=args.host,
                    port=args.model_port,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    index=index,
                )
                for index in range(args.k)
            ]
            for future in concurrent.futures.as_completed(futures):
                completion = future.result()
                completions.append(completion)
                if len([item for item in completions if item.error is None]) >= args.m:
                    for pending in futures:
                        pending.cancel()
                    break
    else:
        for index in range(args.k):
            completion = generate_response(
                prompt,
                host=args.host,
                port=args.model_port,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                index=index,
            )
            completions.append(completion)
            if len([item for item in completions if item.error is None]) >= args.m:
                break

    vote = majority_vote(question, completions)
    return {
        "id": question.get("id"),
        "suite": question.get("suite"),
        "expected": question.get("expected"),
        "scoring_method": question.get("scoring_method", "exact_match"),
        "vote": asdict(vote),
        "completions": [asdict(item) for item in completions],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_suite: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_total = 0
    for result in results:
        suite = str(result.get("suite") or "unknown")
        correct = bool(result["vote"]["correct"])
        by_suite[suite]["total"] += 1
        by_suite[suite]["correct"] += int(correct)
        correct_total += int(correct)

    suite_summary = {
        suite: {
            "correct": values["correct"],
            "total": values["total"],
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
        }
        for suite, values in sorted(by_suite.items())
    }
    total = len(results)
    return {
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "by_suite": suite_summary,
    }


def default_output_path(role: str | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    role_part = role or "role_unknown"
    return DEFAULT_OUTPUT_ROOT / f"{stamp}-{role_part}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["gpqa", "math"])
    parser.add_argument("--sample-per-suite", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--role")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model-port", type=int, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parallel", action="store_true")
    mode.add_argument("--sequential", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.m < 1 or args.m > args.k:
        parser.error("--m must be between 1 and --k")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = build_questions(args.pool, args.suites, args.sample_per_suite, args.seed)
    output_path = args.output or default_output_path(args.role)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "host": args.host,
        "model_port": args.model_port,
        "config": {
            "suites": args.suites,
            "sample_per_suite": args.sample_per_suite,
            "seed": args.seed,
            "k": args.k,
            "m": args.m,
            "mode": "parallel" if args.parallel else "sequential",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "pool": str(args.pool),
        "question_count": len(questions),
    }

    if args.dry_run:
        payload["status"] = "dry_run"
        payload["sampled"] = [{"id": item.get("id"), "suite": item.get("suite")} for item in questions]
    else:
        results = [run_question(question, args) for question in questions]
        payload["status"] = "complete"
        payload["summary"] = summarize(results)
        payload["results"] = results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "question_count")}, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
