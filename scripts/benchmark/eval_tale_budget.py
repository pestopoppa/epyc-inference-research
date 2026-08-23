#!/usr/bin/env python3
"""TALE dynamic budget estimation evaluation.

Compares three brevity strategies on existing eval suites:

1. Baseline — no brevity constraint
2. Static word limits — Action 12 format templates (50w math, 60w general)
3. TALE self-estimated budget — model estimates its own word budget, then
   generates with "Answer in under {beta} words" constraint

Measures accuracy, token count, and OAA/PTI per condition.

Reference: TALE (arXiv:2412.18547) — "use less than {beta} tokens" gives
+3.1pp on GSM8K while cutting 76% of tokens.

Usage:
    python eval_tale_budget.py --suites math general --n-questions 20
    python eval_tale_budget.py --suites math --model-port 8080 --dry-run
"""

# Scoring: delegates to orchestrator B7 debug_scorer (2026-08-23) — no pre-B7 semantics

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
POOL_PATH = SCRIPT_DIR.parent.parent / "benchmarks" / "prompts" / "question_pool.jsonl"
RESULTS_DIR = SCRIPT_DIR.parent.parent / "data" / "tale_budget"

# Static word limits from Action 12 (per suite type)
STATIC_LIMITS = {
    "math": "Show essential steps only, under 50 words. Final answer on last line.",
    "gsm8k": "Show essential steps only, under 50 words. Final answer on last line.",
    "general": "Answer in under 60 words, key point only. No preamble.",
    "hotpotqa": "Answer in under 60 words, key point only. No preamble.",
    "gpqa": "Letter + ONE sentence justification.",
    "agentic": "Answer in under 60 words, key point only. No preamble.",
    "coder": "Output code only.",
}

TALE_PREPASS_PROMPT = (
    "Estimate how many words you need to answer this question correctly.\n"
    "Reply with ONLY a number.\n\n"
    "Question: {question}\n\n"
    "Words needed:"
)

TALE_CONSTRAINT_TEMPLATE = "Answer in under {beta} words.\n\n"


@dataclass
class TrialResult:
    question_id: str
    suite: str
    condition: str  # "baseline", "static", "tale"
    prompt: str
    response: str
    correct: bool | None = None
    total_tokens: int = 0
    elapsed_s: float = 0.0
    tale_budget: int | None = None  # Only for TALE condition


def load_questions(suites: list[str], n_questions: int) -> list[dict[str, Any]]:
    """Load questions from question_pool.jsonl filtered by suite."""
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
            if q.get("__pool_metadata__"):
                continue
            suite = q.get("suite", "")
            if suite not in suites:
                continue
            if suite_counts[suite] >= n_questions:
                continue
            questions.append(q)
            suite_counts[suite] += 1
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
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> tuple[str, int, float]:
    """Generate a response from a llama-server.

    Returns (response_text, token_count, elapsed_seconds).
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
        timeout=1200.0,
    )
    resp.raise_for_status()
    elapsed = time.monotonic() - t0

    data = resp.json()
    text = data["choices"][0]["message"].get("content", "")
    reasoning = data["choices"][0]["message"].get("reasoning_content", "")
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(text) // 4)

    if reasoning:
        text = f"<think>\n{reasoning}\n</think>\n{text}"

    return text, completion_tokens, elapsed


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def score_question(answer: str, question: dict[str, Any]) -> bool:
    """Score an answer against expected. Uses debug_scorer if available."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from debug_scorer import score_answer

    return score_answer(
        answer=strip_think_blocks(answer),
        expected=question.get("expected", ""),
        scoring_method=question.get("scoring_method", "exact_match"),
        scoring_config=question.get("scoring_config"),
    )


def estimate_tale_budget(
    question_text: str,
    host: str = "localhost",
    port: int = 8080,
) -> int:
    """Run TALE pre-pass: ask model to estimate word budget.

    Returns estimated word count, clamped to [10, 500].
    """
    prompt = TALE_PREPASS_PROMPT.format(question=question_text)
    text, _, _ = generate_response(prompt, host, port, max_tokens=32, temperature=0.0)

    # Extract first number from response
    text_clean = strip_think_blocks(text)
    match = re.search(r"\d+", text_clean)
    if match:
        budget = int(match.group())
        return max(10, min(budget, 500))
    log.warning("TALE pre-pass returned no number: %r, defaulting to 60", text_clean)
    return 60


def run_trial(
    question: dict[str, Any],
    condition: str,
    host: str,
    port: int,
    tale_budget: int | None = None,
) -> TrialResult:
    """Run a single trial (one question, one condition)."""
    q_text = question.get("prompt", question.get("question", ""))
    suite = question.get("suite", "unknown")
    q_id = question.get("id", question.get("question_id", "?"))

    if condition == "baseline":
        prompt = q_text
    elif condition == "static":
        limit = STATIC_LIMITS.get(suite, STATIC_LIMITS["general"])
        prompt = f"{limit}\n\n{q_text}"
    elif condition == "tale":
        constraint = TALE_CONSTRAINT_TEMPLATE.format(beta=tale_budget)
        prompt = f"{constraint}{q_text}"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    text, tokens, elapsed = generate_response(prompt, host, port)
    correct = score_question(text, question)

    return TrialResult(
        question_id=str(q_id),
        suite=suite,
        condition=condition,
        prompt=prompt,
        response=text,
        correct=correct,
        total_tokens=tokens,
        elapsed_s=round(elapsed, 2),
        tale_budget=tale_budget,
    )


def run_evaluation(
    questions: list[dict[str, Any]],
    conditions: list[str],
    host: str,
    port: int,
    dry_run: bool = False,
) -> list[TrialResult]:
    """Run all trials across questions and conditions."""
    results: list[TrialResult] = []

    for i, q in enumerate(questions):
        q_text = q.get("prompt", q.get("question", ""))
        q_id = q.get("id", q.get("question_id", "?"))
        suite = q.get("suite", "unknown")
        log.info("[%d/%d] %s q=%s", i + 1, len(questions), suite, q_id)

        if dry_run:
            for cond in conditions:
                log.info("  DRY-RUN %s: would send %d-char prompt", cond, len(q_text))
            continue

        # Estimate TALE budget once per question (shared across TALE trials)
        tale_budget = None
        if "tale" in conditions:
            tale_budget = estimate_tale_budget(q_text, host, port)
            log.info("  TALE budget estimate: %d words", tale_budget)

        for cond in conditions:
            result = run_trial(q, cond, host, port, tale_budget)
            results.append(result)
            status = "correct" if result.correct else "wrong"
            log.info(
                "  %s: %s, %d tokens, %.1fs",
                cond, status, result.total_tokens, result.elapsed_s,
            )

    return results


def save_results(results: list[TrialResult], output_path: Path) -> None:
    """Save results as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    log.info("Saved %d results to %s", len(results), output_path)


def print_summary(results: list[TrialResult]) -> None:
    """Print per-condition summary with OAA/PTI."""
    from eval_metrics import compute_batch_oaa

    conditions = sorted(set(r.condition for r in results))
    print("\n" + "=" * 70)
    print("TALE Budget Evaluation Summary")
    print("=" * 70)

    for cond in conditions:
        cond_results = [asdict(r) for r in results if r.condition == cond]
        metrics = compute_batch_oaa(cond_results, alpha=0.5)
        n = len(cond_results)
        correct = sum(1 for r in cond_results if r.get("correct"))
        print(f"\n  {cond.upper()} ({n} questions, {correct} correct):")
        print(f"    Accuracy:   {metrics['accuracy']:.1%}")
        print(f"    OAA (a=.5): {metrics['oaa']:.4f}")
        print(f"    PTI:        {metrics['pti']:.6f}")
        print(f"    Avg tokens: {metrics['avg_tokens']:.0f}")
        print(f"    Ref tokens: {metrics['reference_tokens']}")

    # Per-suite breakdown
    suites = sorted(set(r.suite for r in results))
    if len(suites) > 1:
        print(f"\n{'─' * 70}")
        print("Per-suite breakdown:")
        for suite in suites:
            print(f"\n  [{suite}]")
            for cond in conditions:
                subset = [
                    asdict(r)
                    for r in results
                    if r.condition == cond and r.suite == suite
                ]
                if not subset:
                    continue
                metrics = compute_batch_oaa(subset, alpha=0.5)
                print(
                    f"    {cond:10s}: acc={metrics['accuracy']:.0%} "
                    f"oaa={metrics['oaa']:.3f} "
                    f"avg_tok={metrics['avg_tokens']:.0f}"
                )

    # TALE budget distribution
    tale_results = [r for r in results if r.condition == "tale" and r.tale_budget]
    if tale_results:
        budgets = [r.tale_budget for r in tale_results]
        print(f"\n{'─' * 70}")
        print(f"TALE budget distribution: min={min(budgets)}, max={max(budgets)}, "
              f"median={sorted(budgets)[len(budgets)//2]}, "
              f"mean={sum(budgets)/len(budgets):.0f}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="TALE dynamic budget estimation evaluation",
    )
    parser.add_argument(
        "--suites", nargs="+", default=["math", "general"],
        help="Suite names to evaluate (default: math general)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=20,
        help="Max questions per suite (default: 20)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["baseline", "static", "tale"],
        choices=["baseline", "static", "tale"],
        help="Conditions to run (default: all three)",
    )
    parser.add_argument(
        "--model-host", default="localhost",
        help="Model server host (default: localhost)",
    )
    parser.add_argument(
        "--model-port", type=int, default=8080,
        help="Model server port (default: 8080)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: data/tale_budget/YYYYMMDD_HHMMSS.jsonl)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load questions and show prompts without sending to model",
    )
    args = parser.parse_args()

    questions = load_questions(args.suites, args.n_questions)
    if not questions:
        log.error("No questions loaded. Check suite names and question_pool.jsonl.")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — %d questions, conditions: %s", len(questions), args.conditions)
        run_evaluation(questions, args.conditions, args.model_host, args.model_port, dry_run=True)

        # Show sample prompts for each condition
        sample = questions[0]
        q_text = sample.get("prompt", sample.get("question", ""))
        suite = sample.get("suite", "unknown")
        print(f"\n--- Sample prompts for suite={suite} ---\n")
        print(f"BASELINE:\n{q_text[:200]}...\n")
        limit = STATIC_LIMITS.get(suite, STATIC_LIMITS["general"])
        print(f"STATIC:\n{limit}\n\n{q_text[:200]}...\n")
        print(f"TALE (assuming budget=45):\n{TALE_CONSTRAINT_TEMPLATE.format(beta=45)}{q_text[:200]}...\n")
        return

    results = run_evaluation(
        questions, args.conditions, args.model_host, args.model_port,
    )

    if not results:
        log.warning("No results generated.")
        return

    # Save
    if args.output:
        output_path = args.output
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"{ts}.jsonl"
    save_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
