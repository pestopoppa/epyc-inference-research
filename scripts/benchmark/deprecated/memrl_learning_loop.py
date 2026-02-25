#!/usr/bin/env python3
"""MemRL Iterative Learning Loop.

Runs benchmark prompts through the orchestrator, scores answers deterministically,
and feeds rewards back to the MemRL Q-scorer for continuous improvement.

Usage:
    # Quick test (1 iteration, 5 samples per suite)
    python scripts/benchmark/memrl_learning_loop.py --iterations 1 --sample-size 5

    # Full learning run (5 iterations, 10 samples per suite)
    python scripts/benchmark/memrl_learning_loop.py --iterations 5 --sample-size 10

    # Specific suites only
    python scripts/benchmark/memrl_learning_loop.py --iterations 3 --suites thinking math

    # Dry run (score only, no reward injection)
    python scripts/benchmark/memrl_learning_loop.py --dry-run --iterations 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default suites for learning loop
DEFAULT_SUITES = [
    "thinking", "general", "math", "agentic",
    "coder", "instruction_precision", "vl",
]

# Orchestrator API
DEFAULT_ORCHESTRATOR_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120


@dataclass
class IterationResult:
    """Result of one learning loop iteration."""

    iteration: int
    timestamp: str
    total_questions: int
    correct: int
    incorrect: int
    errors: int
    accuracy: float
    rewards_injected: int
    mode_distribution: dict[str, int] = field(default_factory=dict)
    suite_scores: dict[str, dict[str, int]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class QuestionResult:
    """Result of a single question."""

    suite: str
    question_id: str
    prompt: str
    answer: str
    expected: str
    passed: bool
    reward: float
    mode: str
    elapsed_seconds: float
    error: str | None = None
    routed_to: str = "frontdoor"


def load_debug_prompts(
    suites: list[str],
    sample_per_suite: int,
    seed: int,
    partition: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Load and sample debug benchmark prompts (deterministic scoring).

    Two modes:

    1. **Sampling** (partition=None): Randomly samples `sample_per_suite`
       questions per suite.
    2. **Partition** (partition=(chunk_index, total_chunks)): Shuffles all
       questions per suite with `seed`, splits into non-overlapping chunks,
       returns only the chunk_index-th chunk. Zero redundancy across
       iterations.

    Args:
        suites: List of suite names to load.
        sample_per_suite: Number of questions to sample per suite.
            Ignored when partition is set.
        seed: Random seed for reproducibility.
        partition: Optional (chunk_index, total_chunks) for non-overlapping
            partitioning. chunk_index is 0-based.

    Returns:
        List of prompt dicts with suite, id, prompt, expected, scoring info.
    """
    import yaml

    DEBUG_PROMPTS_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "prompts" / "debug"

    rng = random.Random(seed)
    all_prompts = []

    for suite_name in suites:
        yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
        if not yaml_path.exists():
            logger.warning(f"No debug suite for '{suite_name}' at {yaml_path}")
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        questions = data.get("questions", [])
        if not questions:
            logger.warning(f"Empty debug suite: {suite_name}")
            continue

        if partition is not None:
            chunk_idx, total_chunks = partition
            # Shuffle deterministically with base seed, then select one chunk
            rng_part = random.Random(seed)
            shuffled = list(questions)
            rng_part.shuffle(shuffled)
            chunk_size = len(shuffled) // total_chunks
            remainder = len(shuffled) % total_chunks
            # First `remainder` chunks get +1 question
            start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            end = start + chunk_size + (1 if chunk_idx < remainder else 0)
            questions = shuffled[start:end]
            logger.info(f"  {suite_name}: partition {chunk_idx}/{total_chunks} → "
                        f"{len(questions)}/{len(shuffled)} questions")
        else:
            if len(questions) > sample_per_suite:
                questions = rng.sample(questions, sample_per_suite)

        for q in questions:
            all_prompts.append({
                "suite": suite_name,
                "id": q["id"],
                "prompt": q["prompt"].strip(),
                "expected": q.get("expected", ""),
                "scoring_method": q.get("scoring_method", "exact_match"),
                "scoring_config": q.get("scoring_config", {}),
                "image_path": q.get("image_path", ""),
            })

    return all_prompts


def call_orchestrator(
    prompt: str,
    url: str = DEFAULT_ORCHESTRATOR_URL,
    timeout: int = DEFAULT_TIMEOUT,
    image_path: str = "",
    client: "httpx.Client | None" = None,
) -> dict[str, Any]:
    """Call the orchestrator API.

    Args:
        prompt: The question to send.
        url: Orchestrator API URL.
        timeout: Request timeout in seconds.
        image_path: Optional path to image file for VL questions.
        client: Optional persistent httpx.Client for connection reuse.

    Returns:
        Response dict with answer, routing_strategy, routed_to, etc.
    """
    import httpx

    payload: dict[str, Any] = {
        "prompt": prompt,
        "real_mode": True,
    }
    if image_path:
        payload["image_path"] = image_path

    try:
        if client is not None:
            response = client.post(f"{url}/chat", json=payload)
        else:
            response = httpx.post(
                f"{url}/chat",
                json=payload,
                timeout=timeout,
            )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": "", "error": str(e), "routing_strategy": "error"}


def score_answer_deterministic(
    answer: str,
    expected: str,
    scoring_method: str = "exact_match",
    scoring_config: dict[str, Any] | None = None,
) -> bool:
    """Score an answer deterministically using the debug scorer.

    Args:
        answer: Model's answer.
        expected: Expected answer.
        scoring_method: Scoring method name (e.g. multiple_choice, exact_match).
        scoring_config: Optional config dict for the scorer.

    Returns:
        True if answer is correct.
    """
    from benchmark.debug_scorer import score_answer

    return score_answer(answer, expected, scoring_method, scoring_config or {})


def run_iteration(
    iteration: int,
    prompts: list[dict[str, Any]],
    url: str = DEFAULT_ORCHESTRATOR_URL,
    dry_run: bool = False,
) -> IterationResult:
    """Run one learning loop iteration.

    Args:
        iteration: Iteration number (1-indexed).
        prompts: List of prompt dicts.
        url: Orchestrator URL.
        dry_run: If True, don't inject rewards.

    Returns:
        IterationResult with scores and statistics.
    """
    start_time = time.perf_counter()
    timestamp = datetime.utcnow().isoformat()

    results: list[QuestionResult] = []
    mode_dist: dict[str, int] = {"direct": 0, "react": 0, "repl": 0, "unknown": 0}
    suite_scores: dict[str, dict[str, int]] = {}

    # Persistent HTTP client for connection reuse across iteration
    import httpx as _httpx
    _client = _httpx.Client(timeout=DEFAULT_TIMEOUT)

    for i, prompt_info in enumerate(prompts):
        suite = prompt_info["suite"]
        qid = prompt_info["id"]
        prompt = prompt_info["prompt"]
        expected = prompt_info["expected"]
        scoring_method = prompt_info.get("scoring_method", "exact_match")
        scoring_config = prompt_info.get("scoring_config", {})
        image_path = prompt_info.get("image_path", "")

        logger.info(
            f"  [{i+1}/{len(prompts)}] {suite}/{qid}"
            + (" [VL]" if image_path else "")
        )

        q_start = time.perf_counter()
        response = call_orchestrator(prompt, url, image_path=image_path, client=_client)
        q_elapsed = time.perf_counter() - q_start

        answer = response.get("answer", "")
        error = response.get("error")
        mode = response.get("routing_strategy", "unknown")
        routed_to = response.get("routed_to", "frontdoor")

        # Normalize mode
        if mode not in ("direct", "react", "repl"):
            if mode in ("rules", "learned", "classified"):
                mode = "direct"  # Pre-react routing strategies map to direct
            else:
                mode = "unknown"

        mode_dist[mode] = mode_dist.get(mode, 0) + 1

        if error:
            passed = False
            reward = -0.5
        else:
            passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)
            reward = 1.0 if passed else -0.5

        # Track suite scores
        if suite not in suite_scores:
            suite_scores[suite] = {"correct": 0, "total": 0}
        suite_scores[suite]["total"] += 1
        if passed:
            suite_scores[suite]["correct"] += 1

        results.append(QuestionResult(
            suite=suite,
            question_id=qid,
            prompt=prompt[:200],
            answer=answer[:500] if answer else "",
            expected=expected[:200],
            passed=passed,
            reward=reward,
            mode=mode,
            elapsed_seconds=q_elapsed,
            error=error,
            routed_to=routed_to,
        ))

    # Inject rewards into MemRL
    rewards_injected = 0
    if not dry_run:
        rewards_injected = _inject_rewards(results, url)

        # Active Q-scorer: score pending tasks and log Q-value distribution
        try:
            from src.api.state import get_state as _get_state
            _state = _get_state()
            if _state.q_scorer and _state.q_scorer_enabled:
                _state.q_scorer.score_pending_tasks()
                if _state.episodic_store and hasattr(_state.episodic_store, 'get_action_q_summary'):
                    summary = _state.episodic_store.get_action_q_summary()
                    if summary:
                        logger.info("  Q-value distribution:")
                        for action, (n, mean_q, std_q) in sorted(
                            summary.items(), key=lambda x: -x[1][1]
                        ):
                            logger.info(
                                f"    Q[{action}]: n={n}, mean={mean_q:.3f}, std={std_q:.3f}"
                            )
        except Exception as e:
            logger.debug(f"Q-scorer update skipped: {e}")

    _client.close()

    elapsed = time.perf_counter() - start_time
    correct = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    return IterationResult(
        iteration=iteration,
        timestamp=timestamp,
        total_questions=total,
        correct=correct,
        incorrect=total - correct - errors,
        errors=errors,
        accuracy=accuracy,
        rewards_injected=rewards_injected,
        mode_distribution=mode_dist,
        suite_scores=suite_scores,
        elapsed_seconds=elapsed,
    )


def _inject_rewards(
    results: list[QuestionResult],
    url: str,
) -> int:
    """Inject rewards into MemRL via the Q-scorer.

    Uses the store_external_reward API endpoint or direct Python API.

    Args:
        results: List of scored question results.
        url: Orchestrator URL (for API-based injection).

    Returns:
        Number of rewards successfully injected.
    """
    injected = 0

    # Try direct Python API first (faster, no HTTP overhead)
    try:
        from src.api.services.memrl import store_external_reward
        from src.api.state import get_state
        state = get_state()

        for r in results:
            # Use actual routed_to from orchestrator instead of hardcoded frontdoor
            action = f"{r.routed_to}:{r.mode}" if r.mode != "unknown" else f"{r.routed_to}:direct"
            context = {
                "task_type": r.suite,
                "source": "learning_loop",
                "question_id": r.question_id,
            }
            success = store_external_reward(
                state=state,
                task_description=r.prompt[:200],
                action=action,
                reward=r.reward,
                context=context,
            )
            if success:
                injected += 1
        return injected
    except Exception as e:
        logger.warning(f"Direct API injection failed: {e}")

    # Fallback: HTTP API injection
    try:
        import httpx

        for r in results:
            action = f"{r.routed_to}:{r.mode}" if r.mode != "unknown" else f"{r.routed_to}:direct"
            try:
                resp = httpx.post(
                    f"{url}/memrl/reward",
                    json={
                        "task_description": r.prompt[:200],
                        "action": action,
                        "reward": r.reward,
                        "context": {
                            "task_type": r.suite,
                            "source": "learning_loop",
                            "question_id": r.question_id,
                        },
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    injected += 1
            except Exception as e:
                continue
    except ImportError:
        logger.error("httpx not available for HTTP injection")

    return injected


def main():
    parser = argparse.ArgumentParser(
        description="MemRL Iterative Learning Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of learning iterations (default: 5)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Questions per suite per iteration (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--suites", nargs="+", default=DEFAULT_SUITES,
        help=f"Suites to run (default: {' '.join(DEFAULT_SUITES)})",
    )
    parser.add_argument(
        "--url", default=DEFAULT_ORCHESTRATOR_URL,
        help=f"Orchestrator URL (default: {DEFAULT_ORCHESTRATOR_URL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score only, don't inject rewards into MemRL",
    )
    parser.add_argument(
        "--regression-check", action="store_true",
        help="Enable regression gates: halt on 3 consecutive accuracy drops, "
        "per-suite parity check, and latency guard",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results (default: auto-generated)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MemRL Learning Loop")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"  Sample size: {args.sample_size}/suite")
    logger.info(f"  Suites: {args.suites}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("=" * 60)

    all_results: list[IterationResult] = []

    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n{'='*40}")
        if args.iterations > 1:
            logger.info(f"Iteration {iteration}/{args.iterations} "
                        f"(partition {iteration-1}/{args.iterations})")
        else:
            logger.info(f"Iteration {iteration}/{args.iterations}")
        logger.info(f"{'='*40}")

        prompts = load_debug_prompts(
            suites=args.suites,
            sample_per_suite=args.sample_size,
            seed=args.seed,
            partition=(iteration - 1, args.iterations) if args.iterations > 1 else None,
        )

        if not prompts:
            logger.error("No prompts loaded, aborting")
            sys.exit(1)

        logger.info(f"Loaded {len(prompts)} questions")

        result = run_iteration(
            iteration=iteration,
            prompts=prompts,
            url=args.url,
            dry_run=args.dry_run,
        )
        all_results.append(result)

        # Print iteration summary
        logger.info(f"\nIteration {iteration} Summary:")
        logger.info(f"  Accuracy: {result.accuracy:.1%} ({result.correct}/{result.total_questions})")
        logger.info(f"  Errors: {result.errors}")
        logger.info(f"  Rewards injected: {result.rewards_injected}")
        logger.info(f"  Mode distribution: {result.mode_distribution}")
        logger.info(f"  Elapsed: {result.elapsed_seconds:.1f}s")

        for suite, scores in sorted(result.suite_scores.items()):
            pct = scores["correct"] / scores["total"] * 100 if scores["total"] > 0 else 0
            logger.info(f"    {suite}: {scores['correct']}/{scores['total']} ({pct:.0f}%)")

        # Regression gates
        if args.regression_check and len(all_results) >= 2:
            regression_halt = False

            # Gate 1: Accuracy monotonicity — halt on 3 consecutive drops
            if len(all_results) >= 3:
                recent = [r.accuracy for r in all_results[-3:]]
                drops = sum(
                    1 for i in range(1, len(recent))
                    if recent[i] < recent[i-1] - 0.05
                )
                if drops >= 2:  # 3 consecutive results = 2 drop transitions
                    logger.error(
                        f"REGRESSION GATE 1 FAILED: 3 consecutive accuracy drops "
                        f"({' → '.join(f'{a:.1%}' for a in recent)})"
                    )
                    regression_halt = True

            # Gate 2: Per-suite parity — each suite >= baseline - 1 point
            if len(all_results) >= 2:
                baseline_scores = all_results[0].suite_scores
                current_scores = result.suite_scores
                for suite_name, baseline_s in baseline_scores.items():
                    if suite_name in current_scores:
                        baseline_correct = baseline_s["correct"]
                        current_correct = current_scores[suite_name]["correct"]
                        if current_correct < baseline_correct - 1:
                            logger.warning(
                                f"REGRESSION GATE 2 WARNING: {suite_name} dropped "
                                f"from {baseline_correct} to {current_correct} "
                                f"(below parity threshold)"
                            )

            # Gate 3: Latency — p50 latency with routing <= 3x baseline
            if len(all_results) >= 2:
                baseline_time = all_results[0].elapsed_seconds
                current_time = result.elapsed_seconds
                if baseline_time > 0 and current_time > 3 * baseline_time:
                    logger.warning(
                        f"REGRESSION GATE 3 WARNING: Latency {current_time:.1f}s > "
                        f"3x baseline {baseline_time:.1f}s"
                    )

            if regression_halt:
                logger.error("Halting learning loop due to regression gate failure")
                break

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Learning Loop Complete")
    logger.info(f"{'='*60}")

    accuracies = [r.accuracy for r in all_results]
    logger.info(f"  Accuracy trend: {' → '.join(f'{a:.1%}' for a in accuracies)}")

    if len(accuracies) >= 2:
        delta = accuracies[-1] - accuracies[0]
        logger.info(f"  Accuracy change: {delta:+.1%}")

    total_rewards = sum(r.rewards_injected for r in all_results)
    logger.info(f"  Total rewards injected: {total_rewards}")

    # Save results
    output_path = args.output
    if output_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(
            PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
            / f"memrl_learning_{ts}.json"
        )

    output_data = {
        "config": {
            "iterations": args.iterations,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "suites": args.suites,
            "dry_run": args.dry_run,
        },
        "results": [asdict(r) for r in all_results],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
