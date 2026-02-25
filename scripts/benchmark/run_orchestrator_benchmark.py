#!/usr/bin/env python3
"""
Orchestrator Benchmark Runner (Phases 1-4)

Deterministic script for validating the full production orchestrator stack.
Packages YOLO handoff Phases 1-4 into a single manually-runnable tool.

Phases:
  1. Smoke Test       — health-check all 11 components, validate speeds
  2. Orchestrator     — end-to-end benchmark through frontdoor (all 8 suites)
  3. Optuna (opt-in)  — hyperparameter optimization (--optimize flag)
  4. Prompt Lookup    — validate lookup acceptance rates across workloads

Usage:
    ./run_orchestrator_benchmark.py                        # Phases 1,2,4
    ./run_orchestrator_benchmark.py --optimize             # Phases 1,2,3,4
    ./run_orchestrator_benchmark.py --phase 1              # Smoke test only
    ./run_orchestrator_benchmark.py --phase 2 4            # Specific phases
    ./run_orchestrator_benchmark.py --start-stack          # Launch stack first
    ./run_orchestrator_benchmark.py --stop-after           # Tear down after
    ./run_orchestrator_benchmark.py --suite thinking coder # Phase 2 filter
    ./run_orchestrator_benchmark.py --dry-run              # Preview only

Exit codes: 0=all pass, 1=phase failure, 2=stack not ready
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    print("Missing dependency: httpx")
    print("Run: pip install httpx")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None  # Will use fallback defaults

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent  # /mnt/raid0/llm/claude


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml without project imports."""
    if yaml is None:
        return fallback
    registry_path = PROJECT_ROOT / "orchestration" / "model_registry.yaml"
    try:
        with registry_path.open() as f:
            data = yaml.safe_load(f)
        timeouts = data.get("runtime_defaults", {}).get("timeouts", {})
        cat_data = timeouts.get(category, {})
        return cat_data.get(key, timeouts.get("default", fallback))
    except Exception:
        return fallback
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
LOCK_FILE = "/mnt/raid0/llm/tmp/orchestrator_benchmark.lock"

COMPARE_SCRIPT = str(PROJECT_ROOT / "scripts" / "benchmark" / "compare_orchestrator_direct.py")
OPTUNA_SCRIPT = str(PROJECT_ROOT / "scripts" / "benchmark" / "optuna_orchestrator.py")
STACK_SCRIPT = str(PROJECT_ROOT / "scripts" / "server" / "orchestrator_stack.py")

SPEED_TOLERANCE = 0.20  # 20%

ALL_SUITES = [
    "thinking", "coder", "general", "math",
    "agentic", "instruction_precision", "long_context", "vl",
]

# Server topology — mirrors CLAUDE.md and orchestrator_stack.py
#
# expected_tps = peak sustained speed on representative workloads (from RESULTS.md)
# smoke_min_tps = minimum acceptable on a smoke test prompt (shorter, less acceleration)
#
# Spec+lookup servers (8081, 8082) show much lower speed on short prompts because:
#   - Prompt lookup needs n-gram patterns in the prompt to match against
#   - Speculative decoding has warmup overhead disproportionate to short generation
# The smoke_min_tps is set to ~40-50% of peak for accelerated servers.
# MoE servers (8084) also show lower speed on short prompts (prefill overhead).
SERVERS = [
    {"port": 8080, "role": "frontdoor",          "expected_tps": 18.0,  "smoke_min_tps": 14.0,  "type": "completion"},
    {"port": 8081, "role": "coder_escalation",    "expected_tps": 39.0,  "smoke_min_tps": 15.0,  "type": "completion", "accel": "spec+lookup"},
    {"port": 8082, "role": "worker_explore",      "expected_tps": 44.0,  "smoke_min_tps": 15.0,  "type": "completion", "accel": "spec+lookup"},
    {"port": 8083, "role": "architect_general",   "expected_tps": 6.75,  "smoke_min_tps": 5.0,   "type": "completion"},
    {"port": 8084, "role": "architect_coding",    "expected_tps": 10.3,  "smoke_min_tps": 4.0,   "type": "completion", "accel": "moe3"},
    {"port": 8085, "role": "ingest_long_context", "expected_tps": 6.3,   "smoke_min_tps": 4.5,   "type": "completion"},
    {"port": 8086, "role": "worker_vision",       "expected_tps": 15.0,  "smoke_min_tps": 10.0,  "type": "vl"},
    {"port": 8087, "role": "vision_escalation",   "expected_tps": 10.0,  "smoke_min_tps": 6.0,   "type": "vl"},
    {"port": 8090, "role": "embedder",            "expected_tps": None,  "smoke_min_tps": None,  "type": "embedding"},
]

SERVICES = [
    {"port": 9001, "role": "document_formalizer", "type": "health"},
    {"port": 8000, "role": "orchestrator",        "type": "health"},
]

# Phase 1 test prompts
# Short prompt for non-accelerated servers
SMOKE_PROMPT_SHORT = "Write a Python function that checks if a number is prime. Include type hints."

# Longer prompt with code context for spec+lookup servers — gives lookup n-grams to
# match against and gives speculative decoding enough generation to amortize warmup.
SMOKE_PROMPT_CODE = """You are reviewing the following codebase. Explain what each function does and suggest improvements:

```python
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> Any:
        self.access_count += 1
        return self.value

class LRUCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            return None
        return entry.access()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self._store) >= self.max_size:
            self._evict_oldest()
        expires = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self._store[key] = CacheEntry(value=value, expires_at=expires)

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        total = len(self._store)
        expired = sum(1 for e in self._store.values() if e.is_expired())
        return {"total": total, "expired": expired, "active": total - expired}
```

Review and improvements:
"""

# Phase 4 workload prompts — prompt lookup needs LONG prompts with repeating
# n-gram patterns to build a useful cache. Short prompts (< 2K chars) show
# regression because cache overhead > benefit. Use 3K+ char prompts.
LOOKUP_WORKLOADS = {
    "code_editing": {
        "desc": "Code editing (high n-gram overlap, ~3K char context)",
        "prompt": SMOKE_PROMPT_CODE,  # ~2.5K chars of Python code with repeating patterns
    },
    "summarization": {
        "desc": "Summarization (moderate n-gram overlap, ~4K char context)",
        "prompt": (
            "Summarize the key technical decisions in the following architecture document:\n\n"
            "## System Architecture Overview\n\n"
            "### Tier A: Front Door\n"
            "The inference server uses a hierarchical orchestration model with three tiers. "
            "Tier A serves as the front door and handles intent classification and task routing. "
            "It runs a lightweight MoE model (Qwen3-Coder-30B-A3B) with expert reduction for "
            "fast response times (~18 t/s). The front door model classifies incoming requests "
            "into categories: code generation, long-context ingestion, architectural reasoning, "
            "math/logic, vision tasks, and general queries. Based on classification, the front "
            "door emits a TaskIR JSON document specifying the task type, priority, required "
            "agents, execution plan, verification gates, and escalation policy.\n\n"
            "### Tier B: Specialists\n"
            "Tier B contains specialist models for code generation, long-context ingestion, "
            "and architectural reasoning. Each specialist is optimized differently:\n"
            "- The coder (Qwen2.5-Coder-32B) uses speculative decoding with a small draft "
            "model (Qwen2.5-Coder-0.5B) for 11x speedup, achieving 33 t/s on code generation.\n"
            "- The ingestion model (Qwen3-Next-80B-A3B) uses SSM architecture which is "
            "incompatible with speculation and prompt lookup. It handles 131K token contexts.\n"
            "- The architect model (Qwen3-235B-A22B) uses expert reduction (4 experts instead "
            "of 22) for balanced speed and quality, achieving 6.75 t/s.\n"
            "- The coding architect (Qwen3-Coder-480B-A35B) is the ultimate escalation target "
            "for code tasks, using 3-expert reduction for 10.3 t/s.\n\n"
            "### Tier C: Workers\n"
            "Tier C workers handle file-level implementation tasks in parallel, using smaller "
            "7B-8B models that are cheap to run concurrently. Workers include:\n"
            "- worker_explore (Qwen2.5-7B with spec decode) at 44 t/s\n"
            "- worker_vision (Qwen2.5-VL-7B) for image analysis at 15 t/s\n"
            "- worker_math (Qwen2.5-Math-7B) for mathematical reasoning\n\n"
            "### Routing and Memory\n"
            "The system routes tasks based on a combination of learned Q-values from "
            "reinforcement learning and semantic similarity matching against an episodic "
            "memory store backed by FAISS. The HybridRouter uses 70% Q-value weighting and "
            "30% semantic similarity. Task embeddings are computed by the Qwen2.5-Coder-0.5B "
            "model running as an embedding server on port 8090.\n\n"
            "### Escalation Protocol\n"
            "Escalation follows a strict protocol:\n"
            "1. First failure returns to the producing agent with gate feedback\n"
            "2. Second failure escalates one tier (worker→coder, coder→architect)\n"
            "3. Third failure at terminal tier triggers REPL exploration fallback\n"
            "All artifacts must pass schema validation, shell linting, format checking, "
            "markdown linting, and unit tests before acceptance. The gate pipeline is run "
            "via `make gates` or `just gates`.\n\n"
            "### Acceleration Methods\n"
            "Three production acceleration tracks are used:\n"
            "- Track 1: External draft speculative decoding (5.9-11x speedup)\n"
            "- Track 2: MoE expert reduction (+21-48% throughput)\n"
            "- Track 8: Prompt lookup n-gram caching (8.6-12.7x on summarization)\n\n"
            "Summary of key technical decisions:\n"
        ),
    },
    "novel_generation": {
        "desc": "Novel generation (low n-gram overlap — control case)",
        "prompt": (
            "Write an original technical blog post introduction about the surprising challenges "
            "of optimizing large language model inference on high-core-count server CPUs. "
            "Cover at least three non-obvious bottlenecks. Be specific and technical."
        ),
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServerCheckResult:
    port: int
    role: str
    healthy: bool
    actual_tps: Optional[float] = None
    expected_tps: Optional[float] = None
    within_tolerance: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class PhaseResult:
    phase: int
    name: str
    passed: bool
    details: dict = field(default_factory=dict)
    duration_s: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkRun:
    run_id: str
    timestamp: str
    phases: list
    args: dict


# ---------------------------------------------------------------------------
# Lock management
# ---------------------------------------------------------------------------

def acquire_lock() -> Optional[int]:
    """Acquire exclusive lock. Returns fd or None."""
    try:
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except (OSError, BlockingIOError):
        return None


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        os.unlink(LOCK_FILE)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_subprocess(
    cmd: list[str],
    dry_run: bool = False,
    timeout: int = _read_registry_timeout("scripts", "orchestrator_phase", 3600),
    label: str = "",
) -> tuple[int, str, str]:
    """Run subprocess, return (exit_code, stdout, stderr)."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY] {label or cmd_str}")
        return 0, "", ""

    print(f"  [RUN] {label or cmd_str}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0 and result.stderr:
            # Print last 5 lines of stderr for diagnostics
            err_lines = result.stderr.strip().split("\n")[-5:]
            for line in err_lines:
                print(f"        {line}")
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {label or cmd_str} (>{timeout}s)")
        return -1, "", "timeout"
    except Exception as e:
        print(f"  [ERROR] {e}")
        return -1, "", str(e)


def check_health(port: int, timeout: int = _read_registry_timeout("health", "quick_check", 10)) -> bool:
    """Check /health endpoint. Returns True if HTTP 200."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"http://localhost:{port}/health")
            return resp.status_code == 200
    except Exception:
        return False


def print_phase_header(phase: int, name: str):
    print(f"\n{'=' * 60}")
    print(f"  PHASE {phase}: {name}")
    print(f"{'=' * 60}")


def result_icon(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# ---------------------------------------------------------------------------
# Phase 1: Smoke Test
# ---------------------------------------------------------------------------

def _send_completion_test(port: int, timeout: int, prompt: str = "", n_predict: int = 128) -> dict:
    """POST /completion, return timings dict."""
    payload = {
        "prompt": prompt or SMOKE_PROMPT_SHORT,
        "n_predict": n_predict,
        "temperature": 0.6,
        "cache_prompt": False,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"http://localhost:{port}/completion", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                timings = data.get("timings", {})
                return {
                    "tps": timings.get("predicted_per_second"),
                    "prompt_tokens": timings.get("prompt_n"),
                    "predicted_tokens": timings.get("predicted_n"),
                }
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def _send_vl_test(port: int, timeout: int) -> dict:
    """POST /v1/chat/completions (text-only smoke test for VL)."""
    payload = {
        "messages": [{"role": "user", "content": "Describe what capabilities you have."}],
        "max_tokens": 64,
        "temperature": 0.6,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            start = time.perf_counter()
            resp = client.post(
                f"http://localhost:{port}/v1/chat/completions", json=payload
            )
            elapsed = time.perf_counter() - start
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens else None
                return {"tps": tps, "completion_tokens": completion_tokens}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "request failed"}


def _send_embedding_test(port: int, timeout: int) -> dict:
    """POST /embedding, verify response contains embedding array.

    llama-server /embedding returns either:
      - A list of floats directly: [0.1, 0.2, ...]
      - A dict: {"embedding": [0.1, 0.2, ...]}
    Handle both formats.
    """
    payload = {"content": "test embedding for smoke check"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"http://localhost:{port}/embedding", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                # Handle both formats
                if isinstance(data, list):
                    embedding = data
                elif isinstance(data, dict):
                    embedding = data.get("embedding", [])
                else:
                    return {"error": f"unexpected response type: {type(data).__name__}"}
                if isinstance(embedding, list) and len(embedding) > 0:
                    return {"embedding_dim": len(embedding)}
                return {"error": "empty embedding vector"}
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def phase1_smoke_test(args) -> PhaseResult:
    """Health-check all servers, send test prompts, validate speed."""
    start = time.monotonic()
    print_phase_header(1, "Smoke Test")
    results: list[ServerCheckResult] = []

    # Check model servers
    for server in SERVERS:
        port = server["port"]
        role = server["role"]
        expected = server["expected_tps"]
        smoke_min = server["smoke_min_tps"]
        stype = server["type"]
        accel = server.get("accel", "")

        # Health check
        if args.dry_run:
            print(f"  [DRY] Health check port {port} ({role})")
            results.append(ServerCheckResult(port=port, role=role, healthy=True))
            continue

        healthy = check_health(port, timeout=10)
        if not healthy:
            print(f"  [FAIL] {role:30} port {port} — not responding")
            results.append(ServerCheckResult(
                port=port, role=role, healthy=False, error="health check failed",
            ))
            continue

        # Send test prompt — use longer code prompt for spec+lookup servers
        if stype == "completion":
            if accel and "lookup" in accel:
                # Spec+lookup servers need a longer prompt with n-gram patterns
                test_result = _send_completion_test(port, args.timeout, SMOKE_PROMPT_CODE, 256)
            else:
                test_result = _send_completion_test(port, args.timeout, SMOKE_PROMPT_SHORT, 128)
        elif stype == "vl":
            test_result = _send_vl_test(port, args.timeout)
        elif stype == "embedding":
            test_result = _send_embedding_test(port, args.timeout)
        else:
            test_result = {}

        actual_tps = test_result.get("tps")
        error = test_result.get("error")

        # Speed check uses smoke_min_tps (not peak expected_tps)
        within = None
        if actual_tps is not None and smoke_min is not None:
            within = actual_tps >= smoke_min

        tps_str = f"{actual_tps:.1f} t/s" if actual_tps else "N/A"
        exp_str = f"(peak {expected}, min {smoke_min})" if expected else ""
        status = result_icon(within is not False and error is None)
        detail = error or ""
        if within is False:
            detail = f"slow: {tps_str} < {smoke_min}"

        print(f"  [{status}] {role:30} port {port:5}  {tps_str:>12} {exp_str} {detail}")

        results.append(ServerCheckResult(
            port=port, role=role, healthy=True,
            actual_tps=actual_tps, expected_tps=expected,
            within_tolerance=within, error=error,
        ))

    # Check services (health-only)
    for service in SERVICES:
        port = service["port"]
        role = service["role"]

        if args.dry_run:
            print(f"  [DRY] Health check port {port} ({role})")
            results.append(ServerCheckResult(port=port, role=role, healthy=True))
            continue

        healthy = check_health(port, timeout=10)
        status = result_icon(healthy)
        print(f"  [{status}] {role:30} port {port:5}  (health-only)")
        results.append(ServerCheckResult(
            port=port, role=role, healthy=healthy,
            error=None if healthy else "health check failed",
        ))

    all_healthy = all(r.healthy for r in results)
    all_fast = all(r.within_tolerance is not False for r in results)
    passed = all_healthy and all_fast

    return PhaseResult(
        phase=1,
        name="Smoke Test",
        passed=passed or args.dry_run,
        details={"servers": [asdict(r) for r in results]},
        duration_s=time.monotonic() - start,
        error=None if passed else "One or more servers failed smoke test",
    )


# ---------------------------------------------------------------------------
# Telemetry: Parse orchestrator progress logs
# ---------------------------------------------------------------------------

PROGRESS_LOG_DIR = PROJECT_ROOT / "logs" / "progress"


def _parse_progress_telemetry(since_iso: str) -> dict:
    """Parse orchestrator progress logs for entries after `since_iso` timestamp.

    Returns aggregated telemetry:
      - role_usage: counter of roles routed to
      - routing_strategy: counter of rules vs learned
      - escalation_chains: list of (from_tier, to_tier, reason)
      - task_outcomes: counter of success/failure
      - avg_latency_s: average task latency
      - total_tasks: total tasks started
    """
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = PROGRESS_LOG_DIR / f"{today}.jsonl"

    if not log_file.exists():
        return {"error": f"no log file for {today}"}

    role_usage: dict[str, int] = {}
    strategy_usage: dict[str, int] = {}
    escalations: list[dict] = []
    outcomes: dict[str, int] = {"success": 0, "failure": 0}
    latencies: list[float] = []
    total_tasks = 0
    # Tool usage from exploration_completed events
    tool_usage: dict[str, int] = {}
    exploration_count = 0
    exploration_tokens = 0
    # Formalizer usage
    formalizer_invocations = 0
    formalizer_pages = 0
    formalizer_elapsed = 0.0
    formalizer_errors = 0

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = entry.get("timestamp", "")
            if ts < since_iso:
                continue

            event = entry.get("event_type")

            if event == "task_started":
                total_tasks += 1

            elif event == "routing_decision":
                data = entry.get("data", {})
                for role in data.get("routing", []):
                    role_usage[role] = role_usage.get(role, 0) + 1
                strategy = data.get("strategy", "unknown")
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

            elif event == "escalation_triggered":
                data = entry.get("data", {})
                escalations.append({
                    "task_id": entry.get("task_id"),
                    "from": data.get("from_tier"),
                    "to": data.get("to_tier"),
                    "reason": data.get("reason", ""),
                })

            elif event == "task_completed":
                outcome = entry.get("outcome", "unknown")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                # Extract latency from outcome_details: "Real inference: N turns, X.XXXs"
                details = entry.get("outcome_details", "")
                if details and "s," in details or details.endswith("s"):
                    m = re.search(r"(\d+\.\d+)s", details)
                    if m:
                        latencies.append(float(m.group(1)))

            elif event == "task_failed":
                outcomes["failure"] = outcomes.get("failure", 0) + 1

            elif event == "exploration_completed":
                exploration_count += 1
                data = entry.get("data", {})
                exploration_tokens += data.get("tokens_spent", 0)
                for fn, cnt in data.get("function_counts", {}).items():
                    tool_usage[fn] = tool_usage.get(fn, 0) + cnt

            elif event == "formalizer_invoked":
                formalizer_invocations += 1
                data = entry.get("data", {})
                formalizer_pages += data.get("pages", 0)
                formalizer_elapsed += data.get("elapsed_sec", 0)
                if entry.get("outcome") == "failure":
                    formalizer_errors += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_tasks": total_tasks,
        "role_usage": dict(sorted(role_usage.items(), key=lambda x: -x[1])),
        "routing_strategy": strategy_usage,
        "escalations": escalations,
        "escalation_count": len(escalations),
        "outcomes": outcomes,
        "avg_latency_s": round(avg_latency, 2),
        "latency_p50_s": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
        "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0.0,
        "tool_usage": dict(sorted(tool_usage.items(), key=lambda x: -x[1])),
        "exploration_count": exploration_count,
        "exploration_tokens": exploration_tokens,
        "formalizer": {
            "invocations": formalizer_invocations,
            "total_pages": formalizer_pages,
            "total_elapsed_sec": round(formalizer_elapsed, 2),
            "errors": formalizer_errors,
        },
    }


def _print_telemetry(telemetry: dict):
    """Print telemetry summary."""
    if "error" in telemetry:
        print(f"\n  Telemetry: {telemetry['error']}")
        return

    total = telemetry["total_tasks"]
    if total == 0:
        print("\n  Telemetry: no tasks found in progress log")
        return

    print(f"\n  {'─' * 50}")
    print(f"  Telemetry ({total} tasks from progress log)")
    print(f"  {'─' * 50}")

    # Role usage
    print(f"  Role usage:")
    for role, count in telemetry["role_usage"].items():
        pct = count / total * 100
        print(f"    {role:30} {count:4} ({pct:.0f}%)")

    # Routing strategy
    print(f"  Routing strategy:")
    for strategy, count in telemetry["routing_strategy"].items():
        print(f"    {strategy:30} {count:4}")

    # Outcomes
    print(f"  Outcomes:")
    for outcome, count in telemetry["outcomes"].items():
        print(f"    {outcome:30} {count:4}")

    # Latency
    print(f"  Latency: avg={telemetry['avg_latency_s']}s  "
          f"p50={telemetry['latency_p50_s']}s  "
          f"p95={telemetry['latency_p95_s']}s")

    # Tool usage (from exploration events)
    tool_usage = telemetry.get("tool_usage", {})
    expl_count = telemetry.get("exploration_count", 0)
    expl_tokens = telemetry.get("exploration_tokens", 0)
    if tool_usage:
        print(f"  Tool usage ({expl_count} explorations, {expl_tokens} tokens):")
        for fn, count in tool_usage.items():
            print(f"    {fn:30} {count:4}")
    elif expl_count > 0:
        print(f"  Explorations: {expl_count} ({expl_tokens} tokens, no tool breakdown)")

    # Formalizer usage
    form = telemetry.get("formalizer", {})
    form_invocations = form.get("invocations", 0)
    if form_invocations > 0:
        form_pages = form.get("total_pages", 0)
        form_elapsed = form.get("total_elapsed_sec", 0)
        form_errors = form.get("errors", 0)
        err_str = f"  ({form_errors} errors)" if form_errors else ""
        print(f"  Formalizer: {form_invocations} calls, "
              f"{form_pages} pages, {form_elapsed:.1f}s total{err_str}")

    # Escalations
    esc_count = telemetry["escalation_count"]
    if esc_count > 0:
        print(f"  Escalations: {esc_count}")
        # Show escalation flow summary
        esc_flows: dict[str, int] = {}
        for esc in telemetry["escalations"]:
            flow = f"{esc['from']} → {esc['to']}"
            esc_flows[flow] = esc_flows.get(flow, 0) + 1
        for flow, count in sorted(esc_flows.items(), key=lambda x: -x[1]):
            print(f"    {flow:40} {count:4}")
    else:
        print(f"  Escalations: 0")

    print(f"  {'─' * 50}")


# ---------------------------------------------------------------------------
# Phase 2: Orchestrator Benchmarking
# ---------------------------------------------------------------------------

def phase2_orchestrator_benchmark(args) -> PhaseResult:
    """Run compare_orchestrator_direct.py for all requested suites."""
    start = time.monotonic()
    telemetry_start = datetime.now().isoformat()  # For progress log parsing
    print_phase_header(2, "Orchestrator Benchmarking")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suites = ALL_SUITES if "all" in args.suite else list(args.suite)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Create baseline if missing
    baseline_path = PROJECT_ROOT / "orchestration" / "orchestrator_baseline.json"
    if not baseline_path.exists() or args.dry_run:
        print("\n  Creating baseline via architect_general...")
        cmd = [
            sys.executable, COMPARE_SCRIPT,
            "--create-baseline", "--suite", "all",
            "--api-url", args.api_url,
            "--timeout", str(args.timeout),
        ]
        exit_code, stdout, stderr = run_subprocess(
            cmd, args.dry_run, timeout=7200,
            label="compare_orchestrator_direct.py --create-baseline --suite all",
        )
        if exit_code != 0 and not args.dry_run:
            return PhaseResult(
                phase=2, name="Orchestrator Benchmark",
                passed=False,
                details={"error": "baseline creation failed", "stderr": stderr[-500:]},
                duration_s=time.monotonic() - start,
                error="Baseline creation failed",
            )

    # Step 2: Run comparison per suite
    print(f"\n  Running {len(suites)} suites through orchestrator frontdoor...\n")
    suite_results = {}

    for suite in suites:
        output_file = str(RESULTS_DIR / f"comparison_{suite}_{timestamp}.json")
        cmd = [
            sys.executable, COMPARE_SCRIPT,
            "--suite", suite,
            "--use-baseline",
            "--api-url", args.api_url,
            "--timeout", str(args.timeout),
            "--output", output_file,
        ]
        exit_code, stdout, stderr = run_subprocess(
            cmd, args.dry_run, timeout=3600,
            label=f"compare_orchestrator_direct.py --suite {suite}",
        )
        suite_results[suite] = {
            "exit_code": exit_code,
            "output_file": output_file,
        }

        # Try to extract and display summary from output file
        if not args.dry_run and exit_code == 0 and Path(output_file).exists():
            try:
                with open(output_file) as f:
                    data = json.load(f)
                suite_results[suite]["quality_pass_rate"] = data.get("quality_pass_rate")
                suite_results[suite]["avg_speedup"] = data.get("avg_speedup")
                suite_results[suite]["avg_tps"] = data.get("avg_tps", 0)
                suite_results[suite]["avg_latency_ms"] = data.get("avg_orchestrator_latency_ms", 0)
                suite_results[suite]["prompts_compared"] = data.get("prompts_compared", 0)
                # Print per-suite mini-summary
                n = data.get("prompts_compared", 0)
                q = data.get("quality_pass_rate", 0)
                tps = data.get("avg_tps", 0)
                lat = data.get("avg_orchestrator_latency_ms", 0)
                q_icon = "✓" if q >= 90 else "✗"
                tps_str = f"{tps:.1f} t/s" if tps > 0 else "—"
                print(f"    {suite:25} {n:2} prompts  {q_icon} {q:5.1f}% quality  "
                      f"{lat:7.0f}ms avg  {tps_str}")
            except (json.JSONDecodeError, KeyError):
                pass
        elif not args.dry_run and exit_code != 0:
            print(f"    {suite:25} [FAILED] exit code {exit_code}")

    all_passed = all(r["exit_code"] == 0 for r in suite_results.values())

    # Print aggregate summary across all suites
    if not args.dry_run:
        total_prompts = sum(r.get("prompts_compared", 0) for r in suite_results.values())
        all_quality = [r["quality_pass_rate"] for r in suite_results.values() if "quality_pass_rate" in r and r["quality_pass_rate"] is not None]
        all_tps = [r["avg_tps"] for r in suite_results.values() if r.get("avg_tps", 0) > 0]
        all_latency = [r["avg_latency_ms"] for r in suite_results.values() if r.get("avg_latency_ms", 0) > 0]
        avg_q = sum(all_quality) / len(all_quality) if all_quality else 0
        avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0
        avg_lat = sum(all_latency) / len(all_latency) if all_latency else 0
        phase_elapsed = time.monotonic() - start
        print(f"\n  {'─' * 55}")
        print(f"  Phase 2 totals: {total_prompts} prompts across {len(suites)} suites in {phase_elapsed:.0f}s")
        if all_quality:
            q_icon = "✓" if avg_q >= 90 else "✗"
            print(f"    Quality: {q_icon} {avg_q:.1f}% avg")
        if all_tps:
            print(f"    Speed:   {avg_tps:.1f} t/s avg")
        if all_latency:
            print(f"    Latency: {avg_lat:.0f}ms avg")
        print(f"  {'─' * 55}")

    # Step 3: Collect telemetry from progress logs
    telemetry = {}
    if not args.dry_run:
        print("\n  Collecting telemetry from progress logs...")
        telemetry = _parse_progress_telemetry(telemetry_start)
        _print_telemetry(telemetry)
    else:
        print("\n  [DRY] Collect telemetry from progress logs")

    return PhaseResult(
        phase=2,
        name="Orchestrator Benchmark",
        passed=all_passed or args.dry_run,
        details={"suites": suite_results, "timestamp": timestamp, "telemetry": telemetry},
        duration_s=time.monotonic() - start,
        error=None if all_passed else "Some suites failed",
    )


# ---------------------------------------------------------------------------
# Phase 3: Optuna Optimization
# ---------------------------------------------------------------------------

def phase3_optuna_optimization(args) -> PhaseResult:
    """Run Optuna optimization across routing, escalation, learning layers."""
    start = time.monotonic()
    print_phase_header(3, "Optuna Optimization")

    layers = [
        ("routing", 50),
        ("escalation", 25),
        ("learning", 25),
    ]

    layer_results = {}
    for layer_name, trials in layers:
        cmd = [
            sys.executable, OPTUNA_SCRIPT,
            "--layer", layer_name,
            "--trials", str(trials),
            "--api-url", args.api_url,
            "--timeout", str(args.timeout),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        exit_code, stdout, stderr = run_subprocess(
            cmd, args.dry_run, timeout=7200,
            label=f"optuna_orchestrator.py --layer {layer_name} --trials {trials}",
        )
        layer_results[layer_name] = {"exit_code": exit_code, "trials": trials}

    # Verification pass with optimized checkpoint
    checkpoint_path = PROJECT_ROOT / "orchestration" / "optimization_checkpoint.yaml"
    if checkpoint_path.exists() and not args.dry_run:
        print("\n  Verification pass with optimized config...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd = [
            sys.executable, COMPARE_SCRIPT,
            "--suite", "all",
            "--use-baseline",
            "--config-from", str(checkpoint_path),
            "--api-url", args.api_url,
            "--timeout", str(args.timeout),
            "--output", str(RESULTS_DIR / f"optimized_comparison_{timestamp}.json"),
        ]
        exit_code, stdout, stderr = run_subprocess(
            cmd, args.dry_run, timeout=7200,
            label="compare_orchestrator_direct.py --config-from checkpoint (verification)",
        )
        layer_results["verification"] = {"exit_code": exit_code}
    elif args.dry_run:
        print("  [DRY] Verification pass with optimized config")
        layer_results["verification"] = {"exit_code": 0}

    all_passed = all(r["exit_code"] == 0 for r in layer_results.values())

    return PhaseResult(
        phase=3,
        name="Optuna Optimization",
        passed=all_passed or args.dry_run,
        details={"layers": layer_results},
        duration_s=time.monotonic() - start,
        error=None if all_passed else "Some layers failed optimization",
    )


# ---------------------------------------------------------------------------
# Phase 4: Prompt Lookup Validation
# ---------------------------------------------------------------------------

def _send_lookup_test(port: int, prompt: str, lookup: bool, timeout: int) -> dict:
    """Send completion to port with lookup flag, return detailed timings."""
    payload = {
        "prompt": prompt,
        "n_predict": 256,
        "temperature": 0.6,
        "cache_prompt": False,
        "stream": False,
        "lookup": lookup,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"http://localhost:{port}/completion", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                timings = data.get("timings", {})
                return {
                    "tps": timings.get("predicted_per_second"),
                    "prompt_tokens": timings.get("prompt_n"),
                    "predicted_tokens": timings.get("predicted_n"),
                    "acceptance_rate": timings.get("draft_acceptance_rate"),
                    "draft_n": timings.get("draft_n"),
                    "draft_accepted": timings.get("draft_n_accepted"),
                }
    except Exception as e:
        return {"error": str(e)}
    return {"error": "request failed"}


def phase4_prompt_lookup_validation(args) -> PhaseResult:
    """Validate prompt lookup on port 8081 (spec+lookup server)."""
    start = time.monotonic()
    print_phase_header(4, "Prompt Lookup Validation")

    port = 8081  # coder_escalation with spec+lookup
    workload_results = {}

    print(f"  Testing on port {port} (coder_escalation: spec K=24 + lookup)")
    print(f"  Comparing lookup=true vs lookup=false (same server, same model)\n")

    for name, workload in LOOKUP_WORKLOADS.items():
        print(f"  {workload['desc']}:")

        if args.dry_run:
            print(f"    [DRY] POST /completion lookup=true")
            print(f"    [DRY] POST /completion lookup=false")
            workload_results[name] = {"dry_run": True}
            continue

        # With lookup
        lookup_on = _send_lookup_test(port, workload["prompt"], True, args.timeout)
        # Without lookup
        lookup_off = _send_lookup_test(port, workload["prompt"], False, args.timeout)

        on_tps = lookup_on.get("tps")
        off_tps = lookup_off.get("tps")
        acceptance = lookup_on.get("acceptance_rate")

        speedup = on_tps / off_tps if on_tps and off_tps and off_tps > 0 else None

        on_str = f"{on_tps:.1f} t/s" if on_tps else "ERR"
        off_str = f"{off_tps:.1f} t/s" if off_tps else "ERR"
        acc_str = f"{acceptance:.1%}" if acceptance is not None else "N/A"
        sp_str = f"{speedup:.2f}x" if speedup else "N/A"

        print(f"    lookup=true:  {on_str:>12}  acceptance: {acc_str}")
        print(f"    lookup=false: {off_str:>12}")
        print(f"    speedup:      {sp_str:>12}")

        workload_results[name] = {
            "lookup_tps": on_tps,
            "baseline_tps": off_tps,
            "acceptance_rate": acceptance,
            "speedup": speedup,
            "lookup_error": lookup_on.get("error"),
            "baseline_error": lookup_off.get("error"),
        }

    return PhaseResult(
        phase=4,
        name="Prompt Lookup Validation",
        passed=True,  # Informational — always passes
        details={"workloads": workload_results, "port": port},
        duration_s=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# Stack lifecycle
# ---------------------------------------------------------------------------

def restart_api(api_url: str = "http://localhost:8000", dry_run: bool = False) -> bool:
    """Restart only the orchestrator API (uvicorn), not llama-server backends.

    The Python code changes (chat.py, prompt_builders.py, etc.) run inside
    uvicorn. The llama-server model backends (ports 8080-8090) are unchanged
    and take minutes to reload — don't touch them.
    """
    from urllib.parse import urlparse
    port = urlparse(api_url).port or 8000

    if dry_run:
        print(f"  [DRY] Restart orchestrator API on port {port}")
        return True

    print(f"\n  Restarting orchestrator API on port {port}...")

    # Kill existing uvicorn on the port
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
        for pid_str in pids:
            pid = int(pid_str)
            os.kill(pid, 9)
            print(f"    Killed PID {pid}")
    except Exception as e:
        print(f"    No process on port {port} ({e})")

    time.sleep(1)

    # Start uvicorn
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
    print(f"    Waiting for health on port {port}...")
    deadline = time.time() + 60
    while time.time() < deadline:
        if check_health(port, timeout=3):
            print(f"    [OK] Orchestrator API ready")
            return True
        time.sleep(1)

    print(f"    [FAIL] Orchestrator API did not start. Check {log_file}")
    return False


def start_stack(dry_run: bool = False) -> bool:
    """Start orchestrator stack."""
    print("\nStarting orchestrator stack...")
    cmd = [sys.executable, STACK_SCRIPT, "start"]
    exit_code, _, _ = run_subprocess(
        cmd, dry_run, timeout=600, label="orchestrator_stack.py start"
    )
    return exit_code == 0 or dry_run


def stop_stack(dry_run: bool = False) -> bool:
    """Stop orchestrator stack."""
    print("\nStopping orchestrator stack...")
    cmd = [sys.executable, STACK_SCRIPT, "stop", "--all"]
    exit_code, _, _ = run_subprocess(
        cmd, dry_run, timeout=120, label="orchestrator_stack.py stop --all"
    )
    return exit_code == 0 or dry_run


def wait_for_stack(
    timeout: int = _read_registry_timeout("server", "stack_startup", 300),
    dry_run: bool = False,
) -> bool:
    """Wait for critical ports to become healthy."""
    if dry_run:
        print("  [DRY] Wait for stack health")
        return True

    critical_ports = [p["port"] for p in SERVERS] + [8000]
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        all_up = all(check_health(p, timeout=5) for p in critical_ports)
        if all_up:
            print(f"  All {len(critical_ports)} ports healthy")
            return True
        elapsed = int(time.monotonic() - start)
        print(f"  Waiting for stack... ({elapsed}s / {timeout}s)")
        time.sleep(10)

    # Report which ports are down
    for port in critical_ports:
        if not check_health(port, timeout=3):
            print(f"  [DOWN] port {port}")
    return False


# ---------------------------------------------------------------------------
# Results & summary
# ---------------------------------------------------------------------------

def save_results(run: BenchmarkRun) -> Path:
    """Save results JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"run_{run.run_id}.json"

    data = {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "args": run.args,
        "phases": [asdict(p) for p in run.phases],
        "summary": {
            "total_phases": len(run.phases),
            "passed": sum(1 for p in run.phases if p.passed),
            "failed": sum(1 for p in run.phases if not p.passed),
            "total_duration_s": sum(p.duration_s for p in run.phases),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def print_summary(phases: list[PhaseResult]):
    """Print final summary table."""
    print(f"\n{'=' * 60}")
    print("  ORCHESTRATOR BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Phase':<8} {'Name':<32} {'Status':<8} {'Duration':>10}")
    print(f"  {'-' * 58}")

    for p in phases:
        status = result_icon(p.passed)
        dur = f"{p.duration_s:.1f}s"
        print(f"  {p.phase:<8} {p.name:<32} {status:<8} {dur:>10}")

    total = sum(p.duration_s for p in phases)
    all_pass = all(p.passed for p in phases)
    print(f"  {'-' * 58}")
    print(f"  {'':8} {'TOTAL':<32} {result_icon(all_pass):<8} {total:.1f}s")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrator Benchmark Runner (Phases 1-4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           Run phases 1,2,4 (default)
  %(prog)s --optimize                Include Phase 3 (Optuna)
  %(prog)s --phase 1                 Smoke test only
  %(prog)s --phase 2 4              Specific phases
  %(prog)s --start-stack             Launch full stack first
  %(prog)s --restart-api             Restart uvicorn only (fast)
  %(prog)s --stop-after              Tear down after
  %(prog)s --suite thinking coder    Phase 2 filter
  %(prog)s --dry-run                 Preview commands
        """,
    )

    parser.add_argument(
        "--phase", nargs="+", type=int, choices=[1, 2, 3, 4],
        default=None,
        help="Run specific phases (default: 1 2 4; phase 3 requires --optimize)",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Enable Phase 3: Optuna hyperparameter optimization",
    )
    parser.add_argument(
        "--suite", nargs="+",
        choices=["all"] + ALL_SUITES,
        default=["all"],
        help="Suites for Phase 2 comparison (default: all)",
    )
    parser.add_argument(
        "--start-stack", action="store_true",
        help="Start orchestrator stack before benchmarking",
    )
    parser.add_argument(
        "--restart-api", action="store_true",
        help="Restart only the orchestrator API (uvicorn, port 8000) before benchmarking. "
             "Does NOT restart llama-server backends (ports 8080-8090).",
    )
    parser.add_argument(
        "--stop-after", action="store_true",
        help="Stop orchestrator stack after benchmarking",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="Orchestrator API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-request timeout in seconds (default: 120)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Determine phases
    if args.phase is not None:
        phases_to_run = sorted(set(args.phase))
    elif args.optimize:
        phases_to_run = [1, 2, 3, 4]
    else:
        phases_to_run = [1, 2, 4]

    if 3 in phases_to_run and not args.optimize:
        print("ERROR: Phase 3 (Optuna) requires --optimize flag")
        sys.exit(1)

    print(f"Orchestrator Benchmark — phases: {phases_to_run}")
    if args.dry_run:
        print("DRY RUN — no commands will be executed\n")

    # Acquire lock
    lock_fd = None
    if not args.dry_run:
        lock_fd = acquire_lock()
        if lock_fd is None:
            print(f"ERROR: Another orchestrator benchmark is running.")
            print(f"       Lock file: {LOCK_FILE}")
            sys.exit(1)

    try:
        # Optional: start stack
        if args.start_stack:
            if not start_stack(args.dry_run):
                print("FATAL: Stack failed to start")
                sys.exit(2)
            print("Waiting for stack to become healthy...")
            if not wait_for_stack(timeout=300, dry_run=args.dry_run):
                print("FATAL: Stack did not become healthy within 5 minutes")
                sys.exit(2)

        # Optional: restart API only (uvicorn, not model backends)
        if args.restart_api:
            if not restart_api(args.api_url, args.dry_run):
                print("FATAL: Orchestrator API failed to restart")
                sys.exit(2)

        # Quick readiness check (unless dry-run or Phase 1 will do it anyway)
        if not args.dry_run and 1 not in phases_to_run:
            if not check_health(8000, timeout=5):
                print("ERROR: Orchestrator API (port 8000) not responding.")
                print("       Start it with: python3 scripts/server/orchestrator_stack.py start")
                print("       Or add --start-stack flag")
                sys.exit(2)

        # Execute phases
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase_results: list[PhaseResult] = []

        phase_map = {
            1: ("Smoke Test", phase1_smoke_test),
            2: ("Orchestrator Benchmark", phase2_orchestrator_benchmark),
            3: ("Optuna Optimization", phase3_optuna_optimization),
            4: ("Prompt Lookup Validation", phase4_prompt_lookup_validation),
        }

        for phase_num in phases_to_run:
            name, func = phase_map[phase_num]

            try:
                result = func(args)
                phase_results.append(result)

                # Phase 1 failure = stack not ready, abort
                if phase_num == 1 and not result.passed and not args.dry_run:
                    print("\nFATAL: Smoke test failed — stack not ready for benchmarking.")
                    print_summary(phase_results)
                    sys.exit(2)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                phase_results.append(PhaseResult(
                    phase=phase_num, name=name, passed=False,
                    error="interrupted by user",
                ))
                break
            except Exception as e:
                print(f"\n  [ERROR] Phase {phase_num} exception: {e}")
                phase_results.append(PhaseResult(
                    phase=phase_num, name=name, passed=False,
                    error=str(e),
                ))

        # Summary
        print_summary(phase_results)

        # Save results
        run = BenchmarkRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            phases=phase_results,
            args={k: v for k, v in vars(args).items() if k != "func"},
        )
        output_path = save_results(run)
        print(f"\nResults saved to: {output_path}")

        # Exit code
        if all(p.passed for p in phase_results):
            sys.exit(0)
        else:
            sys.exit(1)

    finally:
        if args.stop_after:
            stop_stack(args.dry_run)
        if lock_fd is not None:
            release_lock(lock_fd)


if __name__ == "__main__":
    main()
