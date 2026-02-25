#!/usr/bin/env python3
from __future__ import annotations

"""
Temperature Optimizer for LLM Inference

Uses binary search to find the optimal temperature that maximizes quality score.
Temperature does NOT affect inference speed, only output quality.

This module is shared with the orchestrator project.
"""

from dataclasses import dataclass
from typing import Callable, Optional

try:
    from .executor import Executor, Config
    from .output_parser import parse_output
    from .scorer import score_response, ScoreResult
    from .registry import load_registry
except ImportError:
    from executor import Executor, Config
    from output_parser import parse_output
    from scorer import score_response, ScoreResult
    from registry import load_registry


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml."""
    try:
        reg = load_registry()
        if reg and reg._raw:
            timeouts = reg._raw.get("runtime_defaults", {}).get("timeouts", {})
            cat_data = timeouts.get(category, {})
            return cat_data.get(key, timeouts.get("default", fallback))
    except Exception as e:
        pass
    return fallback


_TEMP_OPT_TIMEOUT = _read_registry_timeout("scripts", "temperature_opt", 180)


@dataclass
class TemperatureResult:
    """Result of temperature optimization."""

    optimal_temp: float
    best_score: int
    iterations: int
    confidence_low: float
    confidence_high: float
    all_results: list[tuple[float, int]]  # (temp, score) pairs


def optimize_temperature(
    model_path: str,
    config: Config,
    suite: str,
    question_id: str,
    prompt: str,
    executor: Optional[Executor] = None,
    min_temp: float = 0.0,
    max_temp: float = 1.0,
    precision: float = 0.05,
    max_iterations: int = 10,
    timeout: int = _TEMP_OPT_TIMEOUT,
) -> TemperatureResult:
    """Find optimal temperature using binary search on quality score.

    Args:
        model_path: Path to the model file.
        config: The benchmark configuration.
        suite: Suite name for scoring.
        question_id: Question ID for scoring.
        prompt: The prompt text.
        executor: Optional Executor instance.
        min_temp: Minimum temperature to search.
        max_temp: Maximum temperature to search.
        precision: Stop when range is smaller than this.
        max_iterations: Maximum search iterations.
        timeout: Inference timeout in seconds.

    Returns:
        TemperatureResult with optimal temperature and metadata.
    """
    exec_instance = executor or Executor()
    all_results: list[tuple[float, int]] = []

    def test_temp(temp: float) -> int:
        """Run inference at a temperature and return quality score."""
        result = exec_instance.run_inference(
            model_path=model_path,
            config=config,
            prompt=prompt,
            temperature=temp,
            timeout=timeout,
        )

        if not result.success:
            return 0

        parsed = parse_output(result.raw_output)
        score_result = score_response(suite, question_id, parsed.response)
        return score_result.score

    low = min_temp
    high = max_temp
    best_temp = (low + high) / 2
    best_score = 0
    iteration = 0

    # Initial sampling at endpoints and midpoint
    for temp in [low, (low + high) / 2, high]:
        score = test_temp(temp)
        all_results.append((temp, score))
        if score > best_score:
            best_score = score
            best_temp = temp

    # Binary search
    while (high - low) > precision and iteration < max_iterations:
        iteration += 1

        mid = (low + high) / 2
        q1 = (low + mid) / 2
        q3 = (mid + high) / 2

        # Test quartile points
        score_q1 = test_temp(q1)
        score_q3 = test_temp(q3)

        all_results.append((q1, score_q1))
        all_results.append((q3, score_q3))

        # Update best
        for temp, score in [(q1, score_q1), (q3, score_q3)]:
            if score > best_score:
                best_score = score
                best_temp = temp

        # Narrow the search range based on scores
        if score_q1 >= score_q3:
            high = mid
        else:
            low = mid

        # Early exit if we found max score
        if best_score == 3:
            break

    return TemperatureResult(
        optimal_temp=best_temp,
        best_score=best_score,
        iterations=iteration,
        confidence_low=low,
        confidence_high=high,
        all_results=sorted(all_results, key=lambda x: x[0]),
    )


def quick_temperature_sweep(
    model_path: str,
    config: Config,
    suite: str,
    question_id: str,
    prompt: str,
    executor: Optional[Executor] = None,
    temps: Optional[list[float]] = None,
    timeout: int = _TEMP_OPT_TIMEOUT,
) -> TemperatureResult:
    """Quick temperature sweep over a fixed grid.

    Args:
        model_path: Path to the model file.
        config: The benchmark configuration.
        suite: Suite name for scoring.
        question_id: Question ID for scoring.
        prompt: The prompt text.
        executor: Optional Executor instance.
        temps: List of temperatures to test. Default: [0.0, 0.3, 0.6, 0.9]
        timeout: Inference timeout in seconds.

    Returns:
        TemperatureResult with best temperature from the sweep.
    """
    exec_instance = executor or Executor()
    temps = temps or [0.0, 0.3, 0.6, 0.9]

    all_results: list[tuple[float, int]] = []
    best_temp = temps[0]
    best_score = 0

    for temp in temps:
        result = exec_instance.run_inference(
            model_path=model_path,
            config=config,
            prompt=prompt,
            temperature=temp,
            timeout=timeout,
        )

        if result.success:
            parsed = parse_output(result.raw_output)
            score_result = score_response(suite, question_id, parsed.response)
            score = score_result.score
        else:
            score = 0

        all_results.append((temp, score))

        if score > best_score:
            best_score = score
            best_temp = temp

        # Early exit on max score
        if best_score == 3:
            break

    return TemperatureResult(
        optimal_temp=best_temp,
        best_score=best_score,
        iterations=len(all_results),
        confidence_low=min(temps),
        confidence_high=max(temps),
        all_results=all_results,
    )


if __name__ == "__main__":
    print("=== Temperature Optimizer Test ===\n")
    print("This module requires actual model inference to test.")
    print("Use optimize_temperature() or quick_temperature_sweep() from the benchmark script.")
    print()
    print("Example usage:")
    print("  from lib.temperature_optimizer import optimize_temperature")
    print("  result = optimize_temperature(model_path, config, 'thinking', 't1_q1_logic', prompt)")
    print("  print(f'Optimal temp: {result.optimal_temp}, Score: {result.best_score}')")
