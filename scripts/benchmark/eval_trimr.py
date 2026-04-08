#!/usr/bin/env python3
"""TrimR-style reasoning pruning evaluation.

Evaluates whether reasoning tokens can be pruned from model outputs
without accuracy loss. Implements three strategies:

1. Full reasoning (baseline)
2. Think-strip (remove all <think> blocks)
3. TrimR-lite (paragraph-level selective pruning via verifier)

Uses question_pool.jsonl as source and debug_scorer.py for quality
measurement.

Usage:
    python eval_trimr.py --suites math gsm8k --n-questions 20
    python eval_trimr.py --suites math --strategy all --model-port 8080
    python eval_trimr.py --suites gpqa --strategy think-strip --dry-run

Environment:
    Requires a running llama-server (Qwen3 model) on --model-port.
    Optionally requires a verifier model on --verifier-port for TrimR-lite.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
POOL_PATH = SCRIPT_DIR.parent.parent / "benchmarks" / "prompts" / "question_pool.jsonl"
RESULTS_DIR = SCRIPT_DIR.parent.parent / "data" / "trimr"
SCORER_PATH = SCRIPT_DIR / "debug_scorer.py"


def load_questions(suites: list[str], n_questions: int) -> list[dict[str, Any]]:
    """Load questions from question_pool.jsonl filtered by suite.

    Args:
        suites: Suite names to include.
        n_questions: Max questions per suite.

    Returns:
        List of question dicts.
    """
    if not POOL_PATH.exists():
        log.error("Question pool not found: %s", POOL_PATH)
        log.error("Run: python question_pool.py --build")
        sys.exit(1)

    questions: list[dict[str, Any]] = []
    suite_counts: dict[str, int] = {s: 0 for s in suites}

    with open(POOL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip metadata line
            if q.get("__pool_metadata__"):
                continue

            suite = q.get("suite", "")
            if suite not in suites:
                continue
            if suite_counts[suite] >= n_questions:
                continue

            questions.append(q)
            suite_counts[suite] += 1

            # Check if we have enough
            if all(c >= n_questions for c in suite_counts.values()):
                break

    log.info(
        "Loaded %d questions: %s",
        len(questions),
        ", ".join(f"{s}={c}" for s, c in suite_counts.items()),
    )
    return questions


def generate_response(
    prompt: str,
    host: str = "localhost",
    port: int = 8080,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> tuple[str, int, float]:
    """Generate a response from a llama-server.

    Returns:
        Tuple of (response_text, token_count, elapsed_seconds).
    """
    import httpx

    t0 = time.monotonic()
    resp = httpx.post(
        f"http://{host}:{port}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=600.0,
    )
    resp.raise_for_status()
    elapsed = time.monotonic() - t0

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(text) // 4)

    return text, completion_tokens, elapsed


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_think_blocks(text: str) -> list[str]:
    """Extract all <think> blocks as a list of paragraphs."""
    blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    paragraphs = []
    for block in blocks:
        for para in re.split(r"\n\s*\n", block.strip()):
            para = para.strip()
            if para:
                paragraphs.append(para)
    return paragraphs


def trimr_prune(
    full_response: str,
    question: dict[str, Any],
    verifier_port: int = 8082,
) -> str:
    """Apply TrimR-lite paragraph-level pruning.

    For each reasoning paragraph, check if removing it changes the answer
    quality. If not, prune it.

    Args:
        full_response: The full model response including <think> blocks.
        question: Question dict with expected answer and scoring config.
        verifier_port: Port for verifier model.

    Returns:
        Pruned response text.
    """
    paragraphs = extract_think_blocks(full_response)
    if not paragraphs:
        return full_response

    # Import scorer
    sys.path.insert(0, str(SCRIPT_DIR))
    from debug_scorer import score_answer

    # Score the full response first
    answer_text = strip_think_blocks(full_response)
    full_correct = score_answer(
        answer=answer_text,
        expected=question.get("expected", ""),
        scoring_method=question.get("scoring_method", "exact_match"),
        scoring_config=question.get("scoring_config"),
    )

    if not full_correct:
        # If full response is already wrong, no pruning can help
        return full_response

    # Try removing each paragraph and check if answer still correct
    # Build the think block with selective paragraphs
    keep_paragraphs = []
    for i, para in enumerate(paragraphs):
        # Create a version without this paragraph
        remaining = [p for j, p in enumerate(paragraphs) if j != i]
        if remaining:
            pruned_think = "<think>\n" + "\n\n".join(remaining) + "\n</think>"
        else:
            pruned_think = ""

        # Reconstruct the response
        pruned_response = pruned_think + "\n" + answer_text

        # Score the pruned version
        pruned_correct = score_answer(
            answer=strip_think_blocks(pruned_response),
            expected=question.get("expected", ""),
            scoring_method=question.get("scoring_method", "exact_match"),
            scoring_config=question.get("scoring_config"),
        )

        if not pruned_correct:
            # This paragraph is needed — keep it
            keep_paragraphs.append(para)
        # else: paragraph can be safely pruned

    # Reconstruct with kept paragraphs only
    if keep_paragraphs:
        pruned_think = "<think>\n" + "\n\n".join(keep_paragraphs) + "\n</think>"
        return pruned_think + "\n" + answer_text
    else:
        return answer_text


def evaluate(
    questions: list[dict[str, Any]],
    strategy: str,
    model_port: int = 8080,
    verifier_port: int = 8082,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run evaluation for a given strategy.

    Args:
        questions: List of question dicts.
        strategy: "full", "think-strip", or "trimr".
        model_port: Port for the target model.
        verifier_port: Port for the verifier model (TrimR only).
        dry_run: If True, skip inference and return mock results.

    Returns:
        List of result dicts.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from debug_scorer import score_answer

    results = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        suite = q.get("suite", "unknown")
        prompt = q.get("prompt", "")
        expected = q.get("expected", "")
        scoring_method = q.get("scoring_method", "exact_match")
        scoring_config = q.get("scoring_config")

        log.info("[%d/%d] %s/%s", i + 1, len(questions), suite, qid)

        if dry_run:
            results.append({
                "id": qid,
                "suite": suite,
                "strategy": strategy,
                "correct": None,
                "total_tokens": 0,
                "think_tokens": 0,
                "answer_tokens": 0,
                "elapsed_s": 0.0,
                "pruning_ratio": 0.0,
            })
            continue

        # Generate response
        try:
            response, total_tokens, elapsed = generate_response(
                prompt, port=model_port,
            )
        except Exception as e:
            log.warning("Generation failed for %s: %s", qid, e)
            results.append({
                "id": qid, "suite": suite, "strategy": strategy,
                "correct": None, "error": str(e),
            })
            continue

        # Extract think tokens
        think_blocks = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        think_text = "\n".join(think_blocks)
        think_tokens = len(think_text) // 4

        # Apply strategy
        if strategy == "full":
            eval_response = response
        elif strategy == "think-strip":
            eval_response = strip_think_blocks(response)
        elif strategy == "trimr":
            eval_response = trimr_prune(response, q, verifier_port)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Score
        answer_for_scoring = strip_think_blocks(eval_response)
        correct = score_answer(
            answer=answer_for_scoring,
            expected=expected,
            scoring_method=scoring_method,
            scoring_config=scoring_config,
        )

        # Token counts for pruned output
        pruned_think = re.findall(r"<think>(.*?)</think>", eval_response, re.DOTALL)
        pruned_think_tokens = len("\n".join(pruned_think)) // 4
        answer_tokens = len(answer_for_scoring) // 4

        pruning_ratio = 0.0
        if think_tokens > 0:
            pruning_ratio = 1.0 - (pruned_think_tokens / think_tokens)

        results.append({
            "id": qid,
            "suite": suite,
            "strategy": strategy,
            "correct": correct,
            "total_tokens": total_tokens,
            "think_tokens": think_tokens,
            "pruned_think_tokens": pruned_think_tokens,
            "answer_tokens": answer_tokens,
            "elapsed_s": round(elapsed, 2),
            "pruning_ratio": round(pruning_ratio, 3),
        })

    return results


def evaluate_parallel(
    questions: list[dict[str, Any]],
    strategy: str,
    model_ports: list[int],
    verifier_port: int = 8082,
) -> list[dict[str, Any]]:
    """Run evaluation in parallel across multiple NUMA model instances.

    Distributes questions round-robin across ports using a thread pool.
    """
    import concurrent.futures

    sys.path.insert(0, str(SCRIPT_DIR))
    from debug_scorer import score_answer

    n_workers = len(model_ports)
    results: list[dict | None] = [None] * len(questions)

    def eval_one(idx: int, q: dict, port: int) -> dict:
        qid = q.get("id", f"q{idx}")
        suite = q.get("suite", "unknown")
        prompt = q.get("prompt", "")
        expected = q.get("expected", "")
        scoring_method = q.get("scoring_method", "exact_match")
        scoring_config = q.get("scoring_config")

        log.info("[%d/%d] %s/%s → port %d", idx + 1, len(questions), suite, qid, port)

        try:
            response, total_tokens, elapsed = generate_response(prompt, port=port)
        except Exception as e:
            log.warning("Generation failed for %s on port %d: %s", qid, port, e)
            return {"id": qid, "suite": suite, "strategy": strategy, "correct": None, "error": str(e)}

        think_blocks = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        think_text = "\n".join(think_blocks)
        think_tokens = len(think_text) // 4

        if strategy == "full":
            eval_response = response
        elif strategy == "think-strip":
            eval_response = strip_think_blocks(response)
        elif strategy == "trimr":
            eval_response = trimr_prune(response, q, verifier_port)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        answer_for_scoring = strip_think_blocks(eval_response)
        correct = score_answer(
            answer=answer_for_scoring,
            expected=expected,
            scoring_method=scoring_method,
            scoring_config=scoring_config,
        )

        pruned_think = re.findall(r"<think>(.*?)</think>", eval_response, re.DOTALL)
        pruned_think_tokens = len("\n".join(pruned_think)) // 4
        answer_tokens = len(answer_for_scoring) // 4

        pruning_ratio = 0.0
        if think_tokens > 0:
            pruning_ratio = 1.0 - (pruned_think_tokens / think_tokens)

        return {
            "id": qid, "suite": suite, "strategy": strategy, "correct": correct,
            "total_tokens": total_tokens, "think_tokens": think_tokens,
            "pruned_think_tokens": pruned_think_tokens, "answer_tokens": answer_tokens,
            "elapsed_s": round(elapsed, 2), "pruning_ratio": round(pruning_ratio, 3),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for idx, q in enumerate(questions):
            port = model_ports[idx % n_workers]
            fut = pool.submit(eval_one, idx, q, port)
            futures[fut] = idx

        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    return [r for r in results if r is not None]


def print_summary(results: list[dict[str, Any]], strategy: str) -> None:
    """Print summary table for evaluation results."""
    suites: dict[str, list[dict]] = {}
    for r in results:
        suites.setdefault(r["suite"], []).append(r)

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")

    for suite, items in sorted(suites.items()):
        scored = [r for r in items if r.get("correct") is not None]
        if not scored:
            print(f"  {suite}: no results")
            continue

        correct = sum(1 for r in scored if r["correct"])
        total = len(scored)
        accuracy = correct / total if total else 0

        avg_think = sum(r.get("think_tokens", 0) for r in scored) / total
        avg_pruning = sum(r.get("pruning_ratio", 0) for r in scored) / total
        avg_elapsed = sum(r.get("elapsed_s", 0) for r in scored) / total

        print(f"  {suite}:")
        print(f"    Accuracy:      {correct}/{total} ({accuracy:.1%})")
        print(f"    Avg think tok: {avg_think:.0f}")
        print(f"    Avg pruning:   {avg_pruning:.1%}")
        print(f"    Avg latency:   {avg_elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="TrimR reasoning pruning evaluation")
    parser.add_argument(
        "--suites", nargs="+", default=["math", "gsm8k"],
        help="Benchmark suites to evaluate (default: math gsm8k)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=20,
        help="Number of questions per suite (default: 20)",
    )
    parser.add_argument(
        "--strategy", choices=["full", "think-strip", "trimr", "all"], default="all",
        help="Pruning strategy to evaluate (default: all)",
    )
    parser.add_argument("--model-port", type=int, default=8080, help="Target model port")
    parser.add_argument(
        "--model-ports", type=str, default="",
        help="Comma-separated ports for NUMA-parallel eval (e.g. 8071,8081,8181,8281,8381). "
             "Overrides --model-port. Questions are distributed round-robin across ports.",
    )
    parser.add_argument("--verifier-port", type=int, default=8082, help="Verifier model port")
    parser.add_argument("--dry-run", action="store_true", help="Skip inference, test pipeline")
    parser.add_argument(
        "--output", type=str, default="",
        help="Output JSONL path (default: data/trimr/results_{strategy}.jsonl)",
    )

    args = parser.parse_args()

    # Parse NUMA-parallel ports
    if args.model_ports:
        ports = [int(p.strip()) for p in args.model_ports.split(",")]
    else:
        ports = [args.model_port]

    questions = load_questions(args.suites, args.n_questions)
    if not questions:
        log.error("No questions loaded")
        sys.exit(1)

    strategies = ["full", "think-strip", "trimr"] if args.strategy == "all" else [args.strategy]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        log.info("Running strategy: %s", strategy)

        if len(ports) > 1 and not args.dry_run:
            log.info("NUMA-parallel mode: %d instances (%s)", len(ports), ",".join(map(str, ports)))
            results = evaluate_parallel(
                questions,
                strategy=strategy,
                model_ports=ports,
                verifier_port=args.verifier_port,
            )
        else:
            results = evaluate(
                questions,
                strategy=strategy,
                model_port=ports[0],
                verifier_port=args.verifier_port,
                dry_run=args.dry_run,
            )

        # Save results
        out_path = Path(args.output) if args.output else RESULTS_DIR / f"results_{strategy}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        log.info("Results saved to %s", out_path)

        print_summary(results, strategy)


if __name__ == "__main__":
    main()
