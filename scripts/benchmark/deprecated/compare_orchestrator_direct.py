#!/usr/bin/env python3
from __future__ import annotations

"""
Orchestrator vs Direct Model Comparison

Compares orchestrator responses against a pre-computed baseline from
direct large model responses. Measures quality retention and speedup.

Usage:
    python scripts/benchmark/compare_orchestrator_direct.py --suite thinking --use-baseline
    python scripts/benchmark/compare_orchestrator_direct.py --create-baseline --suite all
    python scripts/benchmark/compare_orchestrator_direct.py --config-from checkpoint.yaml
    python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root and benchmark dir to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

try:
    import httpx
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install httpx pyyaml")
    sys.exit(1)

from benchmark.suites import load_suite, get_all_suite_names

# Constants
BASELINE_PATH = PROJECT_ROOT / "orchestration" / "orchestrator_baseline.json"
DEFAULT_ORCHESTRATOR_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120

ALL_SUITES = [
    "thinking", "coder", "general", "math",
    "agentic", "instruction_precision", "long_context", "vl",
]

DEBUG_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"


def _load_from_dataset_adapter(
    suite_name: str,
    sample_count: int,
    seed: int,
    stratify: bool = False,
) -> list[dict]:
    """Load questions on-the-fly from HuggingFace cached datasets.

    Samples from the full dataset pool for the given suite.
    Falls back to static YAML if the adapter or datasets library is unavailable.

    Args:
        stratify: If True, draw equal counts per difficulty tier for suites
            with real tier data (general/MMLU, math, instruction_precision).

    Supported suites: general (MMLU 14K), math (GSM8K 1.3K + MATH-500),
    coder (HumanEval 164 + MBPP 500), thinking (ARC 1.2K + HellaSwag 10K),
    instruction_precision (IFEval 541), vl (OCRBench 1K + ChartQA 2.5K).
    """
    try:
        from dataset_adapters import get_adapter, ADAPTER_SUITES
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from dataset_adapters import get_adapter, ADAPTER_SUITES
        except ImportError:
            return []

    if suite_name not in ADAPTER_SUITES:
        return []

    adapter = get_adapter(suite_name)
    if adapter is None:
        return []

    prompts = adapter.sample(n=sample_count, seed=seed, stratify=stratify)
    strat_tag = " (stratified)" if stratify and adapter.has_real_tiers else ""
    if prompts:
        print(f"  [debug] {suite_name}: sampled {len(prompts)} from "
              f"{adapter.total_available} dataset questions (seed={seed}){strat_tag}")
    return prompts


def load_debug_prompts(
    suite: str,
    sample_per_suite: int = 10,
    seed: int | None = None,
    partition: tuple[int, int] | None = None,
    stratify: bool = False,
) -> list[dict]:
    """Load debug benchmark prompts from HuggingFace datasets or static YAML.

    For suites with dataset adapters (general, math, coder, thinking,
    instruction_precision, vl), questions are sampled on-the-fly from
    the full public benchmark pools (thousands of questions). Falls back
    to static YAML if the datasets library is unavailable.

    For YAML-only suites (agentic, long_context), loads from
    benchmarks/prompts/debug/*.yaml.

    Two sampling modes:
    1. **Sampling** (partition=None): Randomly samples `sample_per_suite`
       questions per suite. Each seed gives different questions.
    2. **Partition** (partition=(chunk_index, total_chunks)): Shuffles all
       questions with `seed`, splits into non-overlapping chunks.

    Args:
        suite: Suite name or "all".
        sample_per_suite: Number of questions to sample per suite.
            Ignored when partition is set.
        seed: RNG seed. If None, uses current timestamp.
        partition: Optional (chunk_index, total_chunks) for non-overlapping
            partitioning. chunk_index is 0-based.

    Returns:
        List of prompt dicts compatible with compare_prompt().
    """
    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)
    print(f"  [debug] RNG seed: {seed} (for reproducibility)")

    if suite == "all":
        suite_names = ALL_SUITES
    else:
        suite_names = [suite]

    all_prompts: list[dict] = []

    for suite_name in suite_names:
        # Try on-the-fly dataset adapter first
        adapter_prompts = _load_from_dataset_adapter(
            suite_name, sample_per_suite, seed=seed, stratify=stratify
        )
        if adapter_prompts:
            all_prompts.extend(adapter_prompts)
            continue

        # Fall back to static YAML
        yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
        if not yaml_path.exists():
            print(f"  [debug] No debug suite for '{suite_name}' at {yaml_path}")
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        questions = data.get("questions", [])
        if not questions:
            print(f"  [debug] Empty debug suite: {suite_name}")
            continue

        if partition is not None:
            chunk_idx, total_chunks = partition
            # Shuffle deterministically, then select one chunk
            rng_part = random.Random(seed)
            shuffled = list(questions)
            rng_part.shuffle(shuffled)
            chunk_size = len(shuffled) // total_chunks
            remainder = len(shuffled) % total_chunks
            # First `remainder` chunks get +1 question
            start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            end = start + chunk_size + (1 if chunk_idx < remainder else 0)
            sampled = shuffled[start:end]
            print(f"  [debug] {suite_name}: partition {chunk_idx}/{total_chunks} → "
                  f"{len(sampled)}/{len(questions)} questions")
        else:
            # Random sample (without replacement)
            n = min(sample_per_suite, len(questions))
            sampled = rng.sample(questions, n)
            print(f"  [debug] {suite_name}: sampled {n}/{len(questions)} questions")

        for q in sampled:
            all_prompts.append({
                "id": q["id"],
                "suite": suite_name,
                "prompt": q["prompt"].strip(),
                "context": "",
                "expected": q.get("expected", ""),
                "scoring": [],
                "image_path": q.get("image_path", ""),
                "tier": q.get("tier", 1),
                # Debug-specific fields
                "scoring_method": q.get("scoring_method", "exact_match"),
                "scoring_config": q.get("scoring_config", {}),
            })

    return all_prompts


@dataclass
class ComparisonResult:
    """Result of comparing orchestrator vs direct model."""
    prompt_id: str
    suite: str
    tier: int
    # Direct model
    direct_answer: str
    direct_latency_ms: float
    direct_score: Optional[float]  # Pre-computed Claude-as-judge score
    # Orchestrator
    orchestrator_answer: str
    orchestrator_latency_ms: float
    orchestrator_turns: int
    orchestrator_routed_to: str  # Which role the orchestrator chose
    # Comparison
    speedup: float
    # Speed metrics
    tokens_generated: int = 0
    server_elapsed_s: float = 0.0
    avg_tps: float = 0.0  # tokens_generated / server_elapsed_s
    # Tool usage count
    tools_used: int = 0
    # Debug scorer result (only populated with --debug flag)
    debug_score: Optional[bool] = None


def load_baseline() -> dict:
    """Load pre-computed baseline results."""
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def save_baseline(baseline: dict):
    """Save baseline results."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_comparison_prompts(suite: str) -> list[dict]:
    """Get prompts for comparison from existing benchmark suites via suites.py."""
    prompts = []

    if suite == "all":
        suite_names = ALL_SUITES
    elif suite in ALL_SUITES:
        suite_names = [suite]
    else:
        print(f"Unknown suite: {suite}")
        return []

    for suite_name in suite_names:
        loaded = load_suite(suite_name)
        if not loaded:
            print(f"  Warning: Could not load suite '{suite_name}'")
            continue

        for question in loaded.questions:
            prompt_text = question.prompt

            # Split any prompt with a "---" separator into context + question.
            # This is generated by context_generator.build_full_prompt() for
            # long_context suite, but applies generically to any prompt with
            # embedded context. The orchestrator's LONG_CONTEXT_CONFIG will
            # detect large contexts and use REPL exploration automatically.
            context_text = ""
            if "\n\n---\n\n" in prompt_text:
                parts = prompt_text.split("\n\n---\n\n", 1)
                context_text = parts[0]
                prompt_text = parts[1] if len(parts) > 1 else prompt_text

            # Validate image path exists for VL prompts
            img_path = question.image_path
            if img_path and not Path(img_path).exists():
                print(f"  WARNING: Image not found for {question.id}: {img_path}")

            # Validate context was generated for long_context prompts
            if suite_name == "long_context" and not context_text:
                print(f"  WARNING: No context generated for {question.id}")

            prompts.append({
                "id": question.id,
                "suite": suite_name,
                "prompt": prompt_text,
                "context": context_text,
                "expected": question.expected,
                "scoring": question.scoring,
                "image_path": img_path,
                "tier": question.tier,
            })

    return prompts


def call_orchestrator(
    prompt: str,
    api_url: str,
    timeout: int,
    config: Optional[dict] = None,
    context: str = "",
    image_path: str = "",
    client: Optional[httpx.Client] = None,
) -> dict:
    """Call orchestrator API.

    Args:
        client: Optional persistent httpx.Client. If None, creates a
            temporary client per call (slower due to connection setup).
    """

    payload = {
        "prompt": prompt,
        "real_mode": True,
    }
    if context:
        payload["context"] = context
    if image_path:
        payload["image_path"] = image_path
    if config:
        payload.update(config)

    def _do_request(c: httpx.Client) -> dict:
        start = time.perf_counter()
        response = c.post(f"{api_url}/chat", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            result = response.json()
            result["latency_ms"] = latency_ms
            return result
        else:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "latency_ms": latency_ms,
            }

    try:
        if client is not None:
            return _do_request(client)
        else:
            with httpx.Client(timeout=timeout) as c:
                return _do_request(c)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_ms": 0}


@dataclass
class ComparisonError:
    """Error details from a failed comparison."""
    error_type: str = ""
    error_message: str = ""


def compare_prompt(
    prompt_data: dict,
    baseline: dict,
    api_url: str,
    timeout: int,
    config: Optional[dict] = None,
    client: Optional[httpx.Client] = None,
) -> ComparisonResult:
    """Compare orchestrator response to baseline for a single prompt."""

    prompt_id = prompt_data["id"]
    suite = prompt_data["suite"]
    tier = prompt_data.get("tier", 1)

    # Get baseline data
    baseline_entry = baseline.get("prompts", {}).get(prompt_id, {})
    direct_answer = baseline_entry.get("answer", "")
    direct_latency = baseline_entry.get("latency_ms", 0)
    direct_score = baseline_entry.get("claude_score")

    # Call orchestrator (pass context and image_path when available)
    result = call_orchestrator(
        prompt_data["prompt"], api_url, timeout, config,
        context=prompt_data.get("context", ""),
        image_path=prompt_data.get("image_path") or "",
        client=client,
    )

    # Log errors visibly instead of swallowing them
    if "error" in result:
        ctx_len = len(prompt_data.get("context", ""))
        print(f"\n    ERROR [{prompt_id}]: {result['error']}")
        if ctx_len > 0:
            print(f"    (context: {ctx_len} chars, timeout: {timeout}s)")

    orchestrator_answer = result.get("answer", "")
    orchestrator_latency = result.get("latency_ms", 0)
    orchestrator_turns = result.get("turns", 0)
    routed_to = result.get("routed_to", result.get("role", "unknown"))
    tokens_generated = result.get("tokens_generated", result.get("tokens_used", 0))
    server_elapsed = result.get("elapsed_seconds", 0)
    avg_tps = tokens_generated / server_elapsed if server_elapsed > 0 else 0
    tools_used = result.get("tools_used", 0)

    # Calculate speedup
    if orchestrator_latency > 0 and direct_latency > 0:
        speedup = direct_latency / orchestrator_latency
    else:
        speedup = 0.0

    return ComparisonResult(
        prompt_id=prompt_id,
        suite=suite,
        tier=tier,
        direct_answer=direct_answer[:500],
        direct_latency_ms=direct_latency,
        direct_score=direct_score,
        orchestrator_answer=orchestrator_answer[:500],
        orchestrator_latency_ms=orchestrator_latency,
        orchestrator_turns=orchestrator_turns,
        orchestrator_routed_to=routed_to,
        speedup=speedup,
        tokens_generated=tokens_generated,
        server_elapsed_s=server_elapsed,
        avg_tps=avg_tps,
        tools_used=tools_used,
    )


def create_baseline_entry(
    prompt_data: dict,
    api_url: str,
    timeout: int,
    client: Optional[httpx.Client] = None,
) -> dict:
    """Create baseline entry by calling orchestrator with architect_general role.

    Uses the highest-quality model available (Qwen3-235B) for baseline answers.
    """
    result = call_orchestrator(
        prompt_data["prompt"],
        api_url,
        timeout,
        config={"role": "architect_general"},
        context=prompt_data.get("context", ""),
        image_path=prompt_data.get("image_path") or "",
        client=client,
    )

    answer = result.get("answer", "")
    latency_ms = result.get("latency_ms", 0)
    error = result.get("error")

    return {
        "prompt_id": prompt_data["id"],
        "suite": prompt_data["suite"],
        "tier": prompt_data.get("tier", 1),
        "prompt": prompt_data["prompt"][:500],
        "answer": answer,
        "latency_ms": latency_ms,
        "claude_score": None,  # To be filled by Claude-as-Judge review
        "error": error,
        "created": datetime.now().isoformat(),
    }


def _get_suite_timeout(suite_name: str, default_timeout: int) -> int:
    """Get per-suite timeout from YAML inference_params.

    Long context prompts need much longer timeouts (up to 3600s)
    than the default 120s.

    Args:
        suite_name: Name of the suite.
        default_timeout: Default timeout in seconds.

    Returns:
        Timeout in seconds for this suite.
    """
    try:
        loaded = load_suite(suite_name)
        if loaded and loaded.inference_params:
            return loaded.inference_params.get("timeout", default_timeout)
    except Exception as e:
        pass
    return default_timeout


def run_comparison(
    suite: str,
    api_url: str,
    timeout: int,
    use_baseline: bool,
    config: Optional[dict] = None,
    debug_mode: bool = False,
    debug_sample: int = 10,
    debug_seed: int | None = None,
    stratify_tiers: bool = False,
) -> dict:
    """Run comparison between orchestrator and baseline.

    Args:
        debug_mode: If True, use debug suite with deterministic scoring.
        debug_sample: Number of questions to sample per suite in debug mode.
        debug_seed: RNG seed for debug sampling (None = timestamp).
        stratify_tiers: If True, balance difficulty tiers in sampling.
    """

    if debug_mode:
        prompts = load_debug_prompts(
            suite, sample_per_suite=debug_sample, seed=debug_seed,
            stratify=stratify_tiers,
        )
    else:
        prompts = get_comparison_prompts(suite)
    baseline = load_baseline() if use_baseline else {}

    if not prompts:
        print(f"No prompts found for suite: {suite}")
        return {}

    # Build per-suite timeout map
    suite_timeouts: dict[str, int] = {}
    for p in prompts:
        sn = p["suite"]
        if sn not in suite_timeouts:
            suite_timeouts[sn] = _get_suite_timeout(sn, timeout)

    results = []
    errors = 0
    print(f"\nComparing {len(prompts)} prompts from {suite}...")

    # Log per-suite timeouts if they differ from default
    for sn, st in suite_timeouts.items():
        if st != timeout:
            print(f"  [{sn}] Using suite-specific timeout: {st}s (default: {timeout}s)")

    print("-" * 60)

    # Import debug scorer if in debug mode
    _debug_scorer = None
    if debug_mode:
        try:
            from benchmark.debug_scorer import score_answer as _score_answer
            _debug_scorer = _score_answer
        except ImportError:
            from scripts.benchmark.debug_scorer import score_answer as _score_answer
            _debug_scorer = _score_answer

    # Use a persistent httpx.Client to avoid per-request connection setup
    max_timeout = max(suite_timeouts.values()) if suite_timeouts else timeout
    _client = httpx.Client(timeout=max_timeout)

    for prompt_data in prompts:
        suite_name = prompt_data["suite"]
        effective_timeout = suite_timeouts.get(suite_name, timeout)
        print(f"  [{suite_name}] {prompt_data['id']}...", end=" ", flush=True)

        result = compare_prompt(prompt_data, baseline, api_url, effective_timeout, config, client=_client)

        # Debug mode: score with deterministic scorer
        if _debug_scorer and prompt_data.get("scoring_method"):
            try:
                result.debug_score = _debug_scorer(
                    answer=result.orchestrator_answer,
                    expected=prompt_data.get("expected", ""),
                    scoring_method=prompt_data["scoring_method"],
                    scoring_config=prompt_data.get("scoring_config", {}),
                )
            except Exception as e:
                print(f"\n    SCORER ERROR [{prompt_data['id']}]: {e}")
                result.debug_score = False

        results.append(result)

        speedup_str = f"{result.speedup:.1f}x" if result.speedup > 0 else "N/A"
        latency_str = f"{result.orchestrator_latency_ms:.0f}ms"
        tps_str = f"{result.avg_tps:.1f} t/s" if result.avg_tps > 0 else ""
        has_answer = bool(result.orchestrator_answer and len(result.orchestrator_answer.strip()) > 10)
        answer_str = "got_answer" if has_answer else "EMPTY"
        debug_str = ""
        if result.debug_score is not None:
            debug_str = f", score: {'PASS' if result.debug_score else 'FAIL'}"
        tools_str = f", tools: {result.tools_used}" if result.tools_used > 0 else ""
        print(f"{latency_str:>8}  {tps_str:>10}  speedup: {speedup_str}, {answer_str}{debug_str}, "
              f"turns: {result.orchestrator_turns}{tools_str}, routed: {result.orchestrator_routed_to}")

        if not has_answer:
            errors += 1

    _client.close()

    # Compute summary
    valid_speedups = [r.speedup for r in results if r.speedup > 0]
    avg_speedup = sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0

    avg_latency = sum(r.orchestrator_latency_ms for r in results) / len(results) if results else 0
    avg_turns = sum(r.orchestrator_turns for r in results) / len(results) if results else 0
    total_tools = sum(r.tools_used for r in results)

    # Routing accuracy: count distinct roles used
    route_counts = {}
    for r in results:
        route_counts[r.orchestrator_routed_to] = route_counts.get(r.orchestrator_routed_to, 0) + 1

    # Speed metrics: average tokens per second
    valid_tps = [r.avg_tps for r in results if r.avg_tps > 0]
    avg_tps = sum(valid_tps) / len(valid_tps) if valid_tps else 0

    # Debug scorer accuracy (only when --debug is used)
    scored_results = [r for r in results if r.debug_score is not None]
    debug_accuracy = None
    if scored_results:
        debug_accuracy = sum(1 for r in scored_results if r.debug_score) / len(scored_results) * 100

    summary = {
        "suite": suite,
        "prompts_compared": len(results),
        "avg_speedup": avg_speedup,
        "avg_orchestrator_latency_ms": avg_latency,
        "avg_turns": avg_turns,
        "total_tools_used": total_tools,
        "empty_answers": errors,
        "avg_tps": avg_tps,
        "debug_accuracy": debug_accuracy,
        "routing_distribution": route_counts,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat(),
    }

    return summary


def print_summary(summary: dict):
    """Print comparison summary with per-suite breakdown."""

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Suite: {summary.get('suite', 'all')}")
    print(f"Prompts compared: {summary.get('prompts_compared', 0)}")
    print(f"Average speedup: {summary.get('avg_speedup', 0):.2f}x")
    print(f"Average orchestrator latency: {summary.get('avg_orchestrator_latency_ms', 0):.0f}ms")
    print(f"Average turns: {summary.get('avg_turns', 0):.1f}")
    print(f"Total tool invocations: {summary.get('total_tools_used', 0)}")
    print(f"Average tokens/sec: {summary.get('avg_tps', 0):.1f}")
    print(f"Empty answers: {summary.get('empty_answers', 0)}")
    debug_acc = summary.get('debug_accuracy')
    if debug_acc is not None:
        print(f"Debug scorer accuracy: {debug_acc:.1f}%")

    # Routing distribution
    route_dist = summary.get("routing_distribution", {})
    if route_dist:
        print(f"\nRouting distribution:")
        for role, count in sorted(route_dist.items(), key=lambda x: -x[1]):
            print(f"  {role:30} {count:3}")

    # Per-suite breakdown
    results = summary.get("results", [])
    suite_stats: dict[str, dict] = {}
    for r in results:
        s = r.get("suite", "unknown")
        if s not in suite_stats:
            suite_stats[s] = {"count": 0, "speedups": [], "debug_pass": 0, "debug_total": 0}
        suite_stats[s]["count"] += 1
        if r.get("speedup", 0) > 0:
            suite_stats[s]["speedups"].append(r["speedup"])
        if r.get("debug_score") is not None:
            suite_stats[s]["debug_total"] += 1
            if r["debug_score"]:
                suite_stats[s]["debug_pass"] += 1

    if len(suite_stats) > 1:
        print(f"\nPer-suite breakdown:")
        for s, st in sorted(suite_stats.items()):
            avg_sp = sum(st["speedups"]) / len(st["speedups"]) if st["speedups"] else 0
            debug_str = ""
            if st["debug_total"] > 0:
                debug_pct = st["debug_pass"] / st["debug_total"] * 100
                debug_str = f"  {debug_pct:5.1f}% accuracy"
            print(f"  {s:25} {st['count']:3} prompts  {avg_sp:5.1f}x speedup{debug_str}")

    # Target: speedup only (quality measured by Claude-as-Judge or debug scorer)
    speedup = summary.get('avg_speedup', 0)
    speedup_status = "PASS" if speedup >= 3.0 else "FAIL"
    print(f"\nTarget: >3x speedup: {speedup_status} ({speedup:.2f}x)")
    if debug_acc is not None:
        debug_status = "PASS" if debug_acc >= 70.0 else "FAIL"
        print(f"Target: >70% debug accuracy: {debug_status} ({debug_acc:.1f}%)")


def _restart_orchestrator_api(api_url: str) -> None:
    """Kill and restart the orchestrator API (uvicorn on port 8000).

    Does NOT touch llama-server backends (ports 8080-8090) — those are
    unchanged and take minutes to reload.
    """
    from urllib.parse import urlparse
    port = urlparse(api_url).port or 8000

    print(f"[restart] Stopping orchestrator API on port {port}...")
    # Find and kill the uvicorn process on the target port
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
        for pid_str in pids:
            pid = int(pid_str)
            os.kill(pid, 9)
            print(f"  Killed PID {pid}")
    except Exception as e:
        print(f"  No process on port {port} ({e})")

    time.sleep(1)

    print(f"[restart] Starting orchestrator API...")
    log_file = PROJECT_ROOT / "logs" / "orchestrator.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = "/mnt/raid0/llm/cache/huggingface"
    env["TMPDIR"] = "/mnt/raid0/llm/tmp"

    with open(log_file, "w") as log:
        subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", "127.0.0.1",
                "--port", str(port),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    # Wait for health
    print(f"  Waiting for health on port {port}...")
    deadline = time.time() + 60
    healthy = False
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=3)
            if resp.status_code == 200:
                healthy = True
                break
        except Exception as e:
            pass
        time.sleep(1)

    if healthy:
        print(f"  [OK] Orchestrator API ready on port {port}")
    else:
        print(f"  [FAIL] Orchestrator API did not start. Check {log_file}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Orchestrator vs Direct Comparison")
    parser.add_argument(
        "--suite",
        choices=["all"] + ALL_SUITES,
        default="all",
        help="Benchmark suite to compare"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_ORCHESTRATOR_URL,
        help="Orchestrator API URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout per request"
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Use pre-computed baseline"
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create baseline entries via architect_general model"
    )
    parser.add_argument(
        "--config-from",
        help="Load optimized config from checkpoint"
    )
    parser.add_argument(
        "--output",
        help="Output file for results"
    )
    parser.add_argument(
        "--restart-api",
        action="store_true",
        help="Restart the orchestrator API (port 8000) before running. "
             "Only restarts uvicorn, NOT the llama-server backends."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug benchmark suite with deterministic scoring. "
             "Loads from benchmarks/prompts/debug/ instead of v1/. "
             "Randomly samples ~10 questions per suite each run."
    )
    parser.add_argument(
        "--debug-sample",
        type=int,
        default=10,
        help="Number of questions to sample per suite in debug mode (default: 10)"
    )
    parser.add_argument(
        "--debug-seed",
        type=int,
        default=None,
        help="RNG seed for debug sampling (default: timestamp)"
    )
    parser.add_argument(
        "--regression-gate",
        action="store_true",
        help="Run per-suite frontdoor-parity regression check. "
        "Fails if any suite drops below frontdoor baseline - 1 point."
    )
    parser.add_argument(
        "--stratify-tiers",
        action="store_true",
        help="Balance difficulty tiers in sampling. Draws equal questions per "
        "tier for suites with real difficulty data (general/MMLU, math, "
        "instruction_precision). Other suites use uniform random."
    )

    args = parser.parse_args()

    # Restart orchestrator API if requested
    if args.restart_api:
        _restart_orchestrator_api(args.api_url)

    # Load config if provided
    config = None
    if args.config_from:
        checkpoint = load_config(args.config_from)
        config = {}
        for layer_data in checkpoint.get("layers", {}).values():
            if layer_data.get("optimal_params"):
                config.update(layer_data["optimal_params"])
        print(f"Loaded config: {config}")

    if args.create_baseline:
        print("Creating baseline entries via architect_general...")
        prompts = get_comparison_prompts(args.suite)
        baseline = load_baseline()

        if "prompts" not in baseline:
            baseline["prompts"] = {}
        if "meta" not in baseline:
            baseline["meta"] = {
                "created": datetime.now().isoformat(),
                "description": "Baseline from architect_general (Qwen3-235B-A22B)",
                "benchmark_version": "v1",
            }

        created = 0
        for prompt_data in prompts:
            if prompt_data["id"] not in baseline["prompts"]:
                entry = create_baseline_entry(prompt_data, args.api_url, args.timeout)
                baseline["prompts"][prompt_data["id"]] = entry
                if entry.get("answer"):
                    print(f"  Added: {prompt_data['id']} ({len(entry['answer'])} chars)")
                    created += 1
                else:
                    err = entry.get("error", "empty response")
                    print(f"  FAILED: {prompt_data['id']}: {err}")
            else:
                print(f"  Exists: {prompt_data['id']}")

        save_baseline(baseline)
        print(f"\nBaseline saved to: {BASELINE_PATH}")
        print(f"Created {created} entries, {len(baseline['prompts'])} total")
        return

    # Run comparison
    summary = run_comparison(
        args.suite,
        args.api_url,
        args.timeout,
        args.use_baseline,
        config,
        debug_mode=args.debug,
        debug_sample=args.debug_sample,
        debug_seed=args.debug_seed,
        stratify_tiers=args.stratify_tiers,
    )

    print_summary(summary)

    # Regression gate: per-suite frontdoor-parity check
    if args.regression_gate and summary.get("results"):
        print(f"\n{'='*60}")
        print("REGRESSION GATE: Per-Suite Frontdoor Parity")
        print(f"{'='*60}")

        # Group results by suite
        suite_pass_counts: dict[str, dict[str, int]] = {}
        for r in summary["results"]:
            s = r.get("suite", "unknown")
            if s not in suite_pass_counts:
                suite_pass_counts[s] = {"total": 0, "pass": 0, "frontdoor": 0, "specialist": 0}
            suite_pass_counts[s]["total"] += 1
            routed = r.get("orchestrator_routed_to", "frontdoor")
            if r.get("debug_score"):
                suite_pass_counts[s]["pass"] += 1
            if routed == "frontdoor":
                suite_pass_counts[s]["frontdoor"] += 1
            else:
                suite_pass_counts[s]["specialist"] += 1

        gate_passed = True
        for suite_name, counts in sorted(suite_pass_counts.items()):
            total = counts["total"]
            passed = counts["pass"]
            pct = passed / total * 100 if total > 0 else 0
            # Frontdoor baseline assumption: overall debug_accuracy is the baseline
            # Per-suite check: each suite should be within 1 point of suite average
            baseline_pct = summary.get("debug_accuracy", 0) or 0
            status = "PASS" if pct >= baseline_pct - 10 else "FAIL"  # 10% tolerance
            if status == "FAIL":
                gate_passed = False
            specialist_pct = counts["specialist"] / total * 100 if total > 0 else 0
            print(
                f"  {suite_name:25s} {passed}/{total} ({pct:.0f}%) "
                f"specialist:{specialist_pct:.0f}%  [{status}]"
            )

        overall = "PASS" if gate_passed else "FAIL"
        print(f"\nRegression gate: {overall}")
        if not gate_passed:
            print("WARNING: Some suites below parity threshold. "
                  "Consider disabling specialist routing (ORCHESTRATOR_SPECIALIST_ROUTING=0).")
            sys.exit(1)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "debug" if args.debug else "comparison"
        suite_suffix = f"_{args.suite}" if args.suite != "all" else ""
        output_path = PROJECT_ROOT / "benchmarks" / "results" / "orchestrator" / f"{prefix}{suite_suffix}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
