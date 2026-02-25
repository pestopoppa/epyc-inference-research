#!/usr/bin/env python3
"""Resumable per-prompt orchestrator evaluation with MemRL reward injection.

Replaces compare_orchestrator_direct.py's batch-first design with a
checkpoint-per-prompt architecture:

1. Sample unseen questions from HF datasets (or static YAML fallback)
2. POST /chat for each prompt (orchestrator routes freely)
3. Score deterministically via debug_scorer
4. POST /chat/reward to inject reward into MemRL
5. Append result to JSONL checkpoint (Ctrl+C safe)

Usage:
    # Evaluate 10 questions per suite, auto-resume
    python scripts/benchmark/orchestrator_eval.py --suite all

    # Resume a specific session
    python scripts/benchmark/orchestrator_eval.py --resume eval_20260201_143022

    # Continuous mode (run until Ctrl+C)
    python scripts/benchmark/orchestrator_eval.py --suite all --continuous

    # Show stats from past sessions
    python scripts/benchmark/orchestrator_eval.py --stats

    # Skip MemRL reward injection
    python scripts/benchmark/orchestrator_eval.py --suite all --no-reward
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Add project root and benchmark dir to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

try:
    import httpx
except ImportError:
    print("Missing dependency: httpx")
    print("Run: pip install httpx")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120
DEFAULT_SAMPLE = 10
EVAL_DIR = PROJECT_ROOT / "benchmarks" / "results" / "eval"
SEEN_FILE = EVAL_DIR / "seen_questions.jsonl"
DEBUG_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"

ALL_SUITES = [
    "thinking", "coder", "general", "math",
    "agentic", "instruction_precision", "long_context", "vl",
]

# Per-suite timeout overrides (seconds)
SUITE_TIMEOUTS: dict[str, int] = {
    "long_context": 600,
    "coder": 180,
    "vl": 180,
}

# Graceful shutdown flag
_shutdown = False


def _handle_sigint(sig, frame):
    global _shutdown
    if _shutdown:
        # Second Ctrl+C = hard exit
        sys.exit(1)
    _shutdown = True
    print("\n[SIGINT] Finishing current prompt, then stopping...")


signal.signal(signal.SIGINT, _handle_sigint)


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Per-prompt evaluation result. One line in the JSONL checkpoint."""

    prompt_id: str
    suite: str
    tier: int
    dataset_source: str
    prompt_hash: str
    timestamp: str

    # Orchestrator response
    orchestrator_answer: str
    routed_to: str
    mode: str
    latency_ms: float
    tokens_generated: int
    tps: float
    turns: int
    tools_used: int
    task_id: str

    # Scoring
    scoring_method: str
    expected: str
    correct: bool

    # Reward
    reward_injected: bool

    # Fields with defaults (must come last in dataclass)
    error: str = ""
    tools_called: list[str] = field(default_factory=list)
    role_history: list[str] = field(default_factory=list)


# ── Checkpoint management ─────────────────────────────────────────────


def _ensure_eval_dir():
    """Create eval directory if needed."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(session_id: str) -> list[EvalResult]:
    """Load completed results from a session's JSONL checkpoint."""
    path = EVAL_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(EvalResult(**data))
            except (json.JSONDecodeError, TypeError):
                continue
    return results


def append_checkpoint(session_id: str, result: EvalResult):
    """Append one result to the session's JSONL file (atomic-ish)."""
    _ensure_eval_dir()
    path = EVAL_DIR / f"{session_id}.jsonl"
    line = json.dumps(asdict(result), ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_seen_questions() -> set[str]:
    """Load all prompt_ids ever evaluated across all sessions."""
    seen: set[str] = set()
    if not EVAL_DIR.exists():
        return seen

    # Read from all JSONL checkpoint files
    for path in EVAL_DIR.glob("*.jsonl"):
        if path.name == "seen_questions.jsonl":
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    pid = data.get("prompt_id", "")
                    if pid:
                        seen.add(pid)
                except json.JSONDecodeError:
                    continue

    # Also read the dedicated seen file (backward compat)
    if SEEN_FILE.exists():
        with open(SEEN_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    pid = data.get("prompt_id", "")
                    if pid:
                        seen.add(pid)
                except json.JSONDecodeError:
                    continue

    return seen


def record_seen(prompt_id: str, suite: str, session_id: str):
    """Append to the global seen questions file."""
    _ensure_eval_dir()
    entry = {
        "prompt_id": prompt_id,
        "suite": suite,
        "session": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(SEEN_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Question sampling ─────────────────────────────────────────────────


def _load_from_dataset_adapter(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Sample questions from HF dataset adapters."""
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

    prompts = adapter.sample(n=sample_count, seed=seed)
    if prompts:
        print(f"  [{suite_name}] Sampled {len(prompts)} from "
              f"{adapter.total_available} dataset questions (seed={seed})")
    return prompts


def _load_from_yaml(suite_name: str, sample_count: int, seed: int) -> list[dict]:
    """Fall back to static YAML debug prompts."""
    if yaml is None:
        return []
    yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
    if not yaml_path.exists():
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    questions = data.get("questions", [])
    if not questions:
        return []

    import random
    rng = random.Random(seed)
    n = min(sample_count, len(questions))
    sampled = rng.sample(questions, n)
    print(f"  [{suite_name}] Sampled {n}/{len(questions)} from YAML (seed={seed})")

    result = []
    for q in sampled:
        result.append({
            "id": q["id"],
            "suite": suite_name,
            "prompt": q["prompt"].strip(),
            "context": "",
            "expected": q.get("expected", ""),
            "image_path": q.get("image_path", ""),
            "tier": q.get("tier", 1),
            "scoring_method": q.get("scoring_method", "exact_match"),
            "scoring_config": q.get("scoring_config", {}),
            "dataset_source": "yaml",
        })
    return result


def sample_unseen_questions(
    suite: str,
    sample_per_suite: int,
    seen: set[str],
    seed: int,
) -> list[dict]:
    """Sample questions not in the seen set.

    Oversamples by 3x to compensate for dedup filtering.
    Falls back to YAML if no dataset adapter available.
    """
    suite_names = ALL_SUITES if suite == "all" else [suite]
    all_prompts: list[dict] = []

    for suite_name in suite_names:
        # Oversample to have enough after dedup
        oversample = sample_per_suite * 3

        prompts = _load_from_dataset_adapter(suite_name, oversample, seed)
        if not prompts:
            prompts = _load_from_yaml(suite_name, oversample, seed)

        # Filter out seen questions
        fresh = [p for p in prompts if p["id"] not in seen]
        if len(fresh) < len(prompts):
            filtered = len(prompts) - len(fresh)
            print(f"  [{suite_name}] Filtered {filtered} previously seen questions")

        # Take only what we need
        all_prompts.extend(fresh[:sample_per_suite])

    return all_prompts


# ── Orchestrator interaction ──────────────────────────────────────────


def call_orchestrator(
    prompt: str,
    api_url: str,
    timeout: int,
    client: httpx.Client,
    context: str = "",
    image_path: str = "",
) -> dict:
    """POST /chat and return the response dict."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "real_mode": True,
    }
    if context:
        payload["context"] = context
    if image_path:
        payload["image_path"] = image_path

    try:
        start = time.perf_counter()
        response = client.post(
            f"{api_url}/chat",
            json=payload,
            timeout=timeout,
        )
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
    except httpx.TimeoutException:
        return {"error": f"Timeout after {timeout}s", "latency_ms": timeout * 1000}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_ms": 0}


def inject_reward(
    api_url: str,
    client: httpx.Client,
    task_description: str,
    action: str,
    reward: float,
    context: dict | None = None,
) -> bool:
    """POST /chat/reward to inject external reward into MemRL."""
    try:
        resp = client.post(
            f"{api_url}/chat/reward",
            json={
                "task_description": task_description,
                "action": action,
                "reward": reward,
                "context": context,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("success", False)
        return False
    except Exception as e:
        return False


# ── Evaluation loop ───────────────────────────────────────────────────


def _prompt_hash(text: str) -> str:
    """Short hash of prompt text for dedup verification."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def evaluate_prompt(
    prompt_data: dict,
    api_url: str,
    client: httpx.Client,
    session_id: str,
    inject_rewards: bool = True,
) -> EvalResult:
    """Evaluate a single prompt: call orchestrator, score, inject reward."""
    prompt_id = prompt_data["id"]
    suite = prompt_data["suite"]
    tier = prompt_data.get("tier", 1)
    scoring_method = prompt_data.get("scoring_method", "exact_match")
    expected = prompt_data.get("expected", "")
    prompt_text = prompt_data["prompt"]
    dataset_source = prompt_data.get("dataset_source", "hf")
    timeout = SUITE_TIMEOUTS.get(suite, DEFAULT_TIMEOUT)

    # 1. Call orchestrator
    result = call_orchestrator(
        prompt=prompt_text,
        api_url=api_url,
        timeout=timeout,
        client=client,
        context=prompt_data.get("context", ""),
        image_path=prompt_data.get("image_path", ""),
    )

    error = result.get("error", "")
    answer = result.get("answer", "")
    routed_to = result.get("routed_to", result.get("role", "unknown"))
    mode = result.get("mode", "unknown")
    latency_ms = result.get("latency_ms", 0)
    tokens_generated = result.get("tokens_generated", result.get("tokens_used", 0))
    # Use predicted_tps from llama.cpp (generation-only, excludes prompt eval)
    # Fall back to wall-clock calculation if not available
    predicted_tps = result.get("predicted_tps", 0.0)
    if predicted_tps > 0:
        tps = predicted_tps
    else:
        server_elapsed = result.get("elapsed_seconds", 0)
        tps = tokens_generated / server_elapsed if server_elapsed > 0 else 0.0
    turns = result.get("turns", 0)
    tools_used = result.get("tools_used", 0)
    tools_called = result.get("tools_called", [])
    role_history = result.get("role_history", [])
    # ChatResponse doesn't expose task_id; use error_code as a proxy for failure
    task_id = result.get("error_detail", "") if result.get("error_code") else ""

    # 2. Score with debug_scorer
    correct = False
    if answer and scoring_method:
        try:
            from benchmark.debug_scorer import score_answer
            correct = score_answer(
                answer=answer,
                expected=expected,
                scoring_method=scoring_method,
                scoring_config=prompt_data.get("scoring_config", {}),
            )
        except Exception as e:
            error = error or f"scorer: {e}"

    # 3. Inject reward into MemRL
    reward_injected = False
    if inject_rewards and not error:
        reward_value = 1.0 if correct else -1.0
        task_desc = f"{suite}/{prompt_id}: {prompt_text[:100]}"
        action_str = f"{routed_to}:{mode}"
        reward_injected = inject_reward(
            api_url=api_url,
            client=client,
            task_description=task_desc,
            action=action_str,
            reward=reward_value,
            context={"suite": suite, "tier": tier, "scoring_method": scoring_method},
        )

    return EvalResult(
        prompt_id=prompt_id,
        suite=suite,
        tier=tier,
        dataset_source=dataset_source,
        prompt_hash=_prompt_hash(prompt_text),
        timestamp=datetime.now(timezone.utc).isoformat(),
        orchestrator_answer=answer[:500],
        routed_to=routed_to,
        mode=mode,
        latency_ms=latency_ms,
        tokens_generated=tokens_generated,
        tps=tps,
        turns=turns,
        tools_used=tools_used,
        task_id=task_id,
        scoring_method=scoring_method,
        expected=expected[:200],
        correct=correct,
        reward_injected=reward_injected,
        error=error,
        tools_called=tools_called,
        role_history=role_history,
    )


def run_eval(
    suite: str,
    sample: int,
    seed: int | None,
    api_url: str,
    session_id: str,
    inject_rewards: bool = True,
) -> list[EvalResult]:
    """Run one evaluation batch: sample, evaluate, checkpoint."""
    if seed is None:
        seed = int(time.time())

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Suite: {suite}  Sample: {sample}/suite  Seed: {seed}")
    print(f"API: {api_url}  Rewards: {'on' if inject_rewards else 'off'}")
    print(f"{'='*60}\n")

    # Load existing checkpoint + seen set
    completed = load_checkpoint(session_id)
    completed_ids = {r.prompt_id for r in completed}
    seen = load_seen_questions()
    print(f"Checkpoint: {len(completed)} completed, {len(seen)} total seen")

    # Sample unseen questions
    questions = sample_unseen_questions(suite, sample, seen, seed)
    # Also skip questions already in this session's checkpoint
    questions = [q for q in questions if q["id"] not in completed_ids]

    if not questions:
        print("No unseen questions available. Try a different seed or suite.")
        return completed

    print(f"Evaluating {len(questions)} questions...\n")
    print(f"{'ID':40s} {'Suite':12s} {'ms':>7s} {'t/s':>7s} {'Route':>15s} {'Mode':>8s} {'Score':>6s}")
    print("-" * 100)

    max_timeout = max(SUITE_TIMEOUTS.get(q["suite"], DEFAULT_TIMEOUT) for q in questions)
    client = httpx.Client(timeout=max_timeout)
    new_results: list[EvalResult] = []

    try:
        for i, q in enumerate(questions):
            if _shutdown:
                print(f"\n[Stopped after {i} prompts]")
                break

            result = evaluate_prompt(q, api_url, client, session_id, inject_rewards)

            # Checkpoint immediately
            append_checkpoint(session_id, result)
            record_seen(result.prompt_id, result.suite, session_id)
            new_results.append(result)

            # Live output — main line
            score_str = "PASS" if result.correct else "FAIL"
            if result.error:
                score_str = "ERR"
            tps_str = f"{result.tps:.1f}" if result.tps > 0 else "-"
            print(
                f"{result.prompt_id:40s} {result.suite:12s} "
                f"{result.latency_ms:7.0f} {tps_str:>7s} "
                f"{result.routed_to:>15s} {result.mode:>8s} "
                f"{score_str:>6s}"
            )
            # Detail line — role chain and tools (only when interesting)
            details = []
            if len(result.role_history) > 1:
                details.append("chain: " + " → ".join(result.role_history))
            if result.tools_called:
                unique_tools = list(dict.fromkeys(result.tools_called))  # dedup, preserve order
                details.append(f"tools({result.tools_used}): " + ", ".join(unique_tools))
            if details:
                print(f"{'':40s}   {'  '.join(details)}")
    finally:
        client.close()

    # Summary
    all_results = completed + new_results
    _print_summary(all_results, session_id)
    return all_results


# ── Summary / stats ───────────────────────────────────────────────────


def _print_summary(results: list[EvalResult], session_id: str):
    """Print summary for a single session."""
    if not results:
        print("\nNo results.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.error)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY: {session_id}")
    print(f"{'='*60}")
    print(f"Total: {total}  Correct: {correct}  Errors: {errors}  "
          f"Accuracy: {accuracy:.1f}%")

    # Per-suite breakdown
    suite_stats: dict[str, dict[str, int]] = {}
    for r in results:
        s = suite_stats.setdefault(r.suite, {"total": 0, "correct": 0, "errors": 0})
        s["total"] += 1
        if r.correct:
            s["correct"] += 1
        if r.error:
            s["errors"] += 1

    if len(suite_stats) > 1:
        print(f"\nPer-suite:")
        for name, st in sorted(suite_stats.items()):
            pct = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
            print(f"  {name:25s} {st['correct']}/{st['total']} ({pct:.0f}%)")

    # Routing distribution
    route_counts: dict[str, int] = {}
    for r in results:
        route_counts[r.routed_to] = route_counts.get(r.routed_to, 0) + 1
    if route_counts:
        print(f"\nRouting:")
        for role, count in sorted(route_counts.items(), key=lambda x: -x[1]):
            print(f"  {role:25s} {count:3d}")

    # Mode distribution
    mode_counts: dict[str, int] = {}
    for r in results:
        mode_counts[r.mode] = mode_counts.get(r.mode, 0) + 1
    if mode_counts:
        print(f"\nExecution mode:")
        for mode_name, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            print(f"  {mode_name:25s} {count:3d}")

    # Tool usage
    tool_results = [r for r in results if r.tools_used > 0]
    if tool_results:
        all_tools: dict[str, int] = {}
        for r in tool_results:
            for t in r.tools_called:
                all_tools[t] = all_tools.get(t, 0) + 1
        print(f"\nTool usage: {len(tool_results)}/{total} prompts used tools")
        if all_tools:
            for tool_name, count in sorted(all_tools.items(), key=lambda x: -x[1])[:10]:
                print(f"  {tool_name:25s} {count:3d}")

    # Speed
    valid_tps = [r.tps for r in results if r.tps > 0]
    if valid_tps:
        avg_tps = sum(valid_tps) / len(valid_tps)
        print(f"\nAvg speed: {avg_tps:.1f} t/s (generation-only)")

    # Reward injection
    rewarded = sum(1 for r in results if r.reward_injected)
    print(f"Rewards injected: {rewarded}/{total}")


def print_stats():
    """Aggregate stats across all sessions."""
    if not EVAL_DIR.exists():
        print("No evaluation data found.")
        return

    sessions: dict[str, list[EvalResult]] = {}
    for path in sorted(EVAL_DIR.glob("eval_*.jsonl")):
        sid = path.stem
        results = load_checkpoint(sid)
        if results:
            sessions[sid] = results

    if not sessions:
        print("No evaluation sessions found.")
        return

    print(f"\n{'='*60}")
    print("ALL SESSIONS")
    print(f"{'='*60}")

    total_all = 0
    correct_all = 0
    suite_agg: dict[str, dict[str, int]] = {}

    for sid, results in sessions.items():
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        pct = correct / total * 100 if total > 0 else 0
        ts = results[0].timestamp[:10] if results else "?"
        print(f"  {sid:40s} {correct}/{total} ({pct:.0f}%)  {ts}")
        total_all += total
        correct_all += correct

        for r in results:
            s = suite_agg.setdefault(r.suite, {"total": 0, "correct": 0})
            s["total"] += 1
            if r.correct:
                s["correct"] += 1

    overall_pct = correct_all / total_all * 100 if total_all > 0 else 0
    print(f"\nOverall: {correct_all}/{total_all} ({overall_pct:.1f}%)")
    print(f"Sessions: {len(sessions)}")

    seen = load_seen_questions()
    print(f"Unique questions seen: {len(seen)}")

    if len(suite_agg) > 1:
        print(f"\nPer-suite aggregate:")
        for name, st in sorted(suite_agg.items()):
            pct = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
            print(f"  {name:25s} {st['correct']}/{st['total']} ({pct:.0f}%)")

    # Learning curve: accuracy by session order
    if len(sessions) >= 3:
        print(f"\nLearning curve (accuracy by session):")
        for i, (sid, results) in enumerate(sessions.items()):
            total = len(results)
            correct = sum(1 for r in results if r.correct)
            pct = correct / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 5)
            print(f"  {i+1:2d}. {pct:5.1f}% {bar}")


# ── Restart helper ────────────────────────────────────────────────────


def restart_orchestrator_api(api_url: str) -> None:
    """Kill and restart the orchestrator API (uvicorn on port 8000)."""
    port = urlparse(api_url).port or 8000

    print(f"[restart] Stopping orchestrator API on port {port}...")
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
    # Feature flags: enable production capabilities
    env["ORCHESTRATOR_MEMRL"] = "1"
    env["ORCHESTRATOR_TOOLS"] = "1"
    env["ORCHESTRATOR_SCRIPTS"] = "1"
    # NOTE: Do NOT set ORCHESTRATOR_REPL here — it collides with
    # OrchestratorSettings.repl (REPLSettings model) in config.py.
    # The repl feature flag defaults to True in features.py already.
    env["ORCHESTRATOR_CACHING"] = "1"
    env["ORCHESTRATOR_MOCK_MODE"] = "0"
    env["ORCHESTRATOR_GENERATION_MONITOR"] = "1"
    env["ORCHESTRATOR_REACT_MODE"] = "1"

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

    print(f"  Waiting for health on port {port}...")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=3)
            if resp.status_code == 200:
                print(f"  [OK] Orchestrator API ready")
                return
        except Exception as e:
            pass
        time.sleep(1)

    print(f"  [FAIL] Orchestrator API did not start. Check {log_file}")
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Resumable orchestrator evaluation with MemRL reward injection",
    )
    parser.add_argument(
        "--suite",
        choices=["all"] + ALL_SUITES,
        default="all",
        help="Benchmark suite(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE,
        help=f"Questions per suite (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default: timestamp)",
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL,
        help=f"Orchestrator API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume a specific session ID (e.g. eval_20260201_143022)",
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run batches continuously until Ctrl+C",
    )
    parser.add_argument(
        "--continuous-interval", type=int, default=60,
        help="Seconds between continuous batches (default: 60)",
    )
    parser.add_argument(
        "--no-reward", action="store_true",
        help="Skip MemRL reward injection",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show aggregate stats from all sessions",
    )
    parser.add_argument(
        "--restart-api", action="store_true",
        help="Restart orchestrator API before running",
    )

    args = parser.parse_args()

    # Stats mode
    if args.stats:
        print_stats()
        return

    # Restart API if requested
    if args.restart_api:
        restart_orchestrator_api(args.api_url)

    # Session ID
    if args.resume:
        session_id = args.resume
    else:
        session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    inject_rewards = not args.no_reward

    if args.continuous:
        batch = 0
        while not _shutdown:
            batch += 1
            # Use incrementing seed for each batch
            batch_seed = (args.seed or int(time.time())) + batch
            print(f"\n[Continuous batch {batch}]")
            run_eval(
                suite=args.suite,
                sample=args.sample,
                seed=batch_seed,
                api_url=args.api_url,
                session_id=session_id,
                inject_rewards=inject_rewards,
            )
            if _shutdown:
                break
            print(f"\n[Sleeping {args.continuous_interval}s before next batch...]")
            for _ in range(args.continuous_interval):
                if _shutdown:
                    break
                time.sleep(1)
    else:
        run_eval(
            suite=args.suite,
            sample=args.sample,
            seed=args.seed,
            api_url=args.api_url,
            session_id=session_id,
            inject_rewards=inject_rewards,
        )


if __name__ == "__main__":
    main()
