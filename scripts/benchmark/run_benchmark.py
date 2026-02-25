#!/usr/bin/env python3
from __future__ import annotations

"""
Unified Benchmark Runner

Single entry point for all benchmarks with clean nested loops:

    for role in registry.roles:
        for config in get_configs(role.architecture):
            for suite in get_suites(role):
                for question in suite.questions:
                    run_and_save(role, config, suite, question)

Usage:
    ./run_benchmark.py                    # Run all (skips existing by default)
    ./run_benchmark.py --force            # Force re-run (don't skip)
    ./run_benchmark.py --model coder_escalation  # Run specific model
    ./run_benchmark.py --suite thinking   # Run specific suite
    ./run_benchmark.py --vision-only      # Only VL models (with mmproj)
    ./run_benchmark.py --dry-run          # Show what would run
    ./run_benchmark.py --process-queue    # Process queued models
"""

import argparse
import fcntl
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Add parent directory for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.registry import ModelRegistry, load_registry
from lib.executor import Executor, Config, ServerManager
from lib.output_parser import parse_output
# NOTE: Algorithmic scoring is deprecated. Quality evaluation is done via Claude-as-Judge only.
# See benchmarks/results/reviews/ for Claude-as-Judge scores.

from suites import load_suite, get_suites_for_role, get_inference_params, get_all_suite_names
from results import (
    ResultsManager,
    QuestionResult,
    result_exists,
    result_exists_for_model,
    copy_result_from_role,
)


# Lock file for single instance
LOCK_FILE = "/mnt/raid0/llm/tmp/benchmark.lock"
QUEUE_FILE = "/mnt/raid0/llm/tmp/benchmark_queue.txt"

# Speed test prompt for configs that only need speed measurement (quality inherited from baseline)
SPEED_TEST_PROMPT = """Write a Python function that calculates the Fibonacci sequence up to n terms. Include a docstring explaining the function and type hints for all parameters and return value. Then write a brief example showing how to use the function."""

# Longer speed test prompt for lookup configs (lookup needs substantial input for n-gram matching)
# This prompt includes repetitive patterns (code, technical docs) that give lookup n-grams to match.
LOOKUP_SPEED_TEST_PROMPT = """You are reviewing the following codebase. Summarize what each function does:

```python
# Module: cache_manager.py
# Provides caching utilities for the application

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    \"\"\"Represents a single cache entry with metadata.\"\"\"
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        \"\"\"Check if this cache entry has expired.\"\"\"
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> T:
        \"\"\"Mark entry as accessed and return value.\"\"\"
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value


class CacheManager:
    \"\"\"Thread-safe cache manager with TTL and eviction support.\"\"\"

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _generate_key(self, *args, **kwargs) -> str:
        \"\"\"Generate a unique cache key from arguments.\"\"\"
        data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        \"\"\"Get a value from cache by key.\"\"\"
        entry = self._cache.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return None
        if entry.is_expired():
            self._evict(key)
            self._stats["misses"] += 1
            return None
        self._stats["hits"] += 1
        return entry.access()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        \"\"\"Set a value in cache with optional TTL.\"\"\"
        if len(self._cache) >= self._max_size:
            self._evict_lru()
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def _evict(self, key: str) -> None:
        \"\"\"Evict a specific key from cache.\"\"\"
        if key in self._cache:
            del self._cache[key]
            self._stats["evictions"] += 1

    def _evict_lru(self) -> None:
        \"\"\"Evict the least recently used entry.\"\"\"
        if not self._cache:
            return
        lru_key = min(self._cache.keys(),
                      key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at)
        self._evict(lru_key)

    def clear(self) -> None:
        \"\"\"Clear all cache entries.\"\"\"
        self._cache.clear()

    @property
    def stats(self) -> Dict[str, int]:
        \"\"\"Get cache statistics.\"\"\"
        return dict(self._stats)


def cached(ttl: int = 3600, cache: Optional[CacheManager] = None):
    \"\"\"Decorator to cache function results.\"\"\"
    _cache = cache or CacheManager()
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = _cache._generate_key(func.__name__, *args, **kwargs)
            result = _cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                _cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator
```

Provide a summary of:
1. The main classes and their purposes
2. Key methods and what they do
3. Any design patterns used"""

# Reference TPS for timeout multiplier calculation (20 t/s = 1.0x multiplier)
REFERENCE_TPS = 20.0
# Minimum timeout multiplier (even fast models don't get shorter timeouts)
MIN_TIMEOUT_MULTIPLIER = 1.0
# Default multiplier when speed test fails
DEFAULT_TIMEOUT_MULTIPLIER = 2.0

# Inference defaults
_DEFAULT_MAX_TOKENS = 256
_LOOKUP_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.6
_SERVER_STARTUP_TIMEOUT_BASE = 600

# Timeout scaling: timeout = max(base, size_gb * multiplier + buffer)
_TIMEOUT_SIZE_MULTIPLIER = 3
_TIMEOUT_SIZE_BUFFER = 120

# Log noise prefixes to skip when extracting errors from stderr
_LOG_PREFIXES = ('build:', 'main:', 'llama_model_loader:', 'print_info:', 'load_')
# Keywords indicating a real error line
_ERROR_KEYWORDS = ('error:', 'error ', 'failed', 'fatal', 'abort', 'segfault', 'exception')


def _compute_timeout(size_gb: float, base: int = 180) -> int:
    """Compute dynamic timeout based on model size in GB."""
    return max(base, int(size_gb * _TIMEOUT_SIZE_MULTIPLIER) + _TIMEOUT_SIZE_BUFFER)


def _extract_error_hint(stderr: str, max_chars: int = 80) -> str:
    """Extract meaningful error from stderr, filtering log noise."""
    for line in reversed(stderr.split('\n')):
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(p) for p in _LOG_PREFIXES):
            continue
        if any(x in line.lower() for x in _ERROR_KEYWORDS):
            return line[:max_chars]
    return ""


def acquire_lock() -> Optional[int]:
    """Acquire exclusive lock for single-instance execution.

    Returns:
        File descriptor if lock acquired, None if another instance is running.
    """
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except (OSError, BlockingIOError):
        return None


def release_lock(fd: int) -> None:
    """Release the lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        os.unlink(LOCK_FILE)
    except OSError:
        pass


def print_progress(
    role: str,
    config: str,
    suite: str,
    question: str,
    status: str,
    tokens_per_second: Optional[float] = None,
    score: Optional[int] = None,
) -> None:
    """Print progress line."""
    tps_str = f"{tokens_per_second:.1f} t/s" if tokens_per_second else "---"
    score_str = f"{score}/3" if score is not None else "---"
    print(f"[{status:8}] {role:25} {config:15} {suite:20} {question:15} {tps_str:>10} {score_str:>5}")


def count_pending_tests(
    run_id: str,
    role: str,
    configs: list,
    suite_names: list[str],
    force: bool = False,
) -> tuple[int, int]:
    """Count tests that need to run vs total tests.

    This is a "preflight check" to avoid loading models when all tests are complete.

    Returns:
        (pending_count, total_count)
    """
    pending = 0
    total = 0

    for config in configs:
        if config.speed_test_only:
            total += 1
            if force or not result_exists(run_id, role, config.name):
                pending += 1
        else:
            for suite_name in suite_names:
                suite = load_suite(suite_name)
                if not suite:
                    continue
                # Skip lookup configs for non-long_context suites
                if config.config_type in ("lookup", "moe_lookup") and suite_name != "long_context":
                    continue
                for question in suite.questions:
                    total += 1
                    if force or not result_exists(run_id, role, config.name, suite_name, question.id):
                        pending += 1

    return pending, total


def build_work_items(
    registry: ModelRegistry,
    executor: Executor,
    model_filter: Optional[str] = None,
    suite_filter: Optional[str] = None,
) -> list[dict]:
    """Build flat list of work items from nested loops.

    Returns list of dicts with: role, model_path, architecture, config, suite_name, suite, question, params
    """
    work_items = []

    roles = registry.get_all_roles(include_deprecated=False)
    if model_filter:
        roles = [r for r in roles if r == model_filter]

    for role in roles:
        model_path = registry.get_model_path(role)
        if not model_path or not os.path.exists(model_path):
            continue

        architecture = registry.get_architecture(role)
        configs = executor.get_configs_for_architecture(architecture, role, registry)

        for config in configs:
            suite_names = get_suites_for_role(role, registry)
            if suite_filter:
                suite_names = [s for s in suite_names if s == suite_filter]

            for suite_name in suite_names:
                suite = load_suite(suite_name)
                if not suite:
                    continue

                params = get_inference_params(suite)

                for question in suite.questions:
                    work_items.append({
                        "role": role,
                        "model_path": model_path,
                        "architecture": architecture,
                        "config": config,
                        "suite_name": suite_name,
                        "suite": suite,
                        "question": question,
                        "params": params,
                    })

    return work_items


class _ServerState:
    """Mutable state for the benchmark server lifecycle."""

    __slots__ = ("server", "model_path", "experts", "draft_path")

    def __init__(self) -> None:
        self.server: Optional[ServerManager] = None
        self.model_path: Optional[str] = None
        self.experts: Optional[int] = None
        self.draft_path: Optional[str] = None

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None


def _ensure_server(
    ss: _ServerState,
    model_path: str,
    config,
    role: str,
    size_gb: float,
    registry: ModelRegistry,
    no_mmap: bool,
    mmproj_path: Optional[str],
    is_new_model: bool,
) -> None:
    """Start or restart the llama-server to match the requirements of *config*.

    Handles three scenarios:
    1. New model — stop old server, start fresh
    2. Different MoE expert count — restart with new override
    3. Different draft model — restart with new draft
    """
    # --- 1. New model ---
    if is_new_model:
        if ss.server and ss.model_path != model_path:
            print(f"    [SERVER] Stopping server for previous model", flush=True)
            ss.stop()

        if ss.server is None:
            print(f"    [SERVER] Starting llama-server (model will stay in RAM)...", flush=True)
            ss.server = ServerManager(port=8080)
            ss.server.start(model_path, moe_override=None, registry=registry,
                            no_mmap=no_mmap, role=role, mmproj_path=mmproj_path)
            timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
            if not ss.server.wait_ready(timeout=timeout):
                print(f"    [SERVER] Failed to start, falling back to subprocess mode", flush=True)
                ss.server = None
            else:
                ss.model_path = model_path
                ss.experts = None
                ss.draft_path = None
                print(f"    [SERVER] Ready, model loaded in RAM (default experts)", flush=True)
        return

    if ss.server is None or not ss.server.is_running():
        return

    # --- 2. MoE expert count change ---
    if config.config_type == "moe":
        required_experts = config.moe_experts
    elif config.config_type == "baseline":
        required_experts = None
    else:
        required_experts = ss.experts

    if required_experts != ss.experts:
        if required_experts is None:
            print(f"      [SERVER] Restarting for baseline (default experts)...", flush=True)
            moe_override = None
        else:
            moe_key = registry.get_moe_override_key(role) or "qwen3moe.expert_used_count"
            moe_override = f"{moe_key}=int:{required_experts}"
            print(f"      [SERVER] Restarting for {config.name} ({required_experts} experts)...", flush=True)

        ss.stop()
        ss.server = ServerManager(port=8080)
        ss.server.start(model_path, moe_override=moe_override, registry=registry,
                        no_mmap=no_mmap, role=role, mmproj_path=mmproj_path)
        timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
        if not ss.server.wait_ready(timeout=timeout):
            print(f"      [SERVER] Failed to restart, falling back to subprocess", flush=True)
            ss.server = None
            ss.experts = None
        else:
            ss.experts = required_experts
            ss.draft_path = None
            print(f"      [SERVER] Ready", flush=True)

    if ss.server is None or not ss.server.is_running():
        return

    # --- 3. Draft model change (spec decode) ---
    if config.config_type in ("spec", "moe_spec"):
        required_draft = config.draft_model_path
        if required_draft != ss.draft_path:
            if config.config_type == "moe_spec":
                moe_key = registry.get_moe_override_key(role) or "qwen3moe.expert_used_count"
                moe_override = f"{moe_key}=int:{config.moe_experts}"
            else:
                moe_override = None

            draft_name = Path(required_draft).stem if required_draft else "unknown"
            print(f"      [SERVER] Restarting with draft {draft_name}...", flush=True)
            ss.stop()
            ss.server = ServerManager(port=8080)
            ss.server.start(
                model_path, moe_override=moe_override, registry=registry,
                no_mmap=no_mmap, role=role,
                draft_model_path=required_draft,
                draft_max=config.spec_k,
                mmproj_path=mmproj_path,
            )
            timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
            if not ss.server.wait_ready(timeout=timeout):
                print(f"      [SERVER] Failed to restart with draft, falling back to subprocess", flush=True)
                ss.server = None
                ss.draft_path = None
            else:
                ss.draft_path = required_draft
                if config.config_type == "moe_spec":
                    ss.experts = config.moe_experts
                print(f"      [SERVER] Ready with draft", flush=True)


def _run_speed_test(
    executor: Executor,
    results_manager: ResultsManager,
    config,
    model_path: str,
    size_gb: float,
    mmproj_path: Optional[str],
    role: str,
    run_id: str,
    stats: dict,
) -> None:
    """Execute a speed-only benchmark for *config* (no quality questions)."""
    is_lookup = config.config_type in ("lookup", "moe_lookup")
    speed_prompt = LOOKUP_SPEED_TEST_PROMPT if is_lookup else SPEED_TEST_PROMPT
    speed_max_tokens = _LOOKUP_MAX_TOKENS if is_lookup else _DEFAULT_MAX_TOKENS
    speed_timeout = _compute_timeout(size_gb, base=300 if is_lookup else 180)

    result = executor.run_inference(
        model_path=model_path,
        config=config,
        prompt=speed_prompt,
        max_tokens=speed_max_tokens,
        temperature=_DEFAULT_TEMPERATURE,
        timeout=speed_timeout,
        mmproj_path=mmproj_path,
        role=role,
    )

    if result.timed_out:
        stats["errors"] += 1
        print(f"    [TIMEOUT] {role}/{config.name} (speed test)")
        return

    if not result.success:
        stats["errors"] += 1
        err_hint = (_extract_error_hint(result.stderr, max_chars=60) if result.stderr else "") or f"exit={result.exit_code}"
        print(f"    [ERROR] {role}/{config.name} (speed test): {err_hint}")
        return

    parsed = parse_output(result.raw_output)

    results_manager.add_speed_result(
        run_id=run_id,
        model_role=role,
        config_name=config.name,
        model_path=model_path,
        tokens_per_second=parsed.tokens_per_second or 0,
        inherits_quality_from=config.inherits_quality_from or "baseline",
        acceptance_rate=parsed.acceptance_rate,
    )

    tps = parsed.tokens_per_second
    tps_str = f"{tps:.1f}t/s" if tps else "---"
    acc_str = f"acc={parsed.acceptance_rate:.1%}" if parsed.acceptance_rate else ""
    print(f"      ⚡ {config.name}: {tps_str} {acc_str} (speed only, quality from {config.inherits_quality_from})", flush=True)
    stats["passed"] += 1


def _run_quality_question(
    executor: Executor,
    results_manager: ResultsManager,
    ss: _ServerState,
    config,
    model_path: str,
    mmproj_path: Optional[str],
    role: str,
    run_id: str,
    suite_name: str,
    question,
    params: dict,
    stats: dict,
    force: bool,
) -> None:
    """Execute a single quality benchmark question and store the result."""
    exists = result_exists(run_id, role, config.name, suite_name, question.id)
    if not force and exists:
        stats["skipped"] += 1
        return

    if not force:
        existing_role = result_exists_for_model(
            run_id, model_path, config.name, suite_name, question.id
        )
        if existing_role and existing_role != role:
            copied = copy_result_from_role(
                run_id=run_id,
                from_role=existing_role,
                to_role=role,
                config_name=config.name,
                suite=suite_name,
                question_id=question.id,
                model_path=model_path,
            )
            if copied:
                stats["skipped"] += 1
                print(f"    [COPY] {role}/{config.name}/{question.id} <- {existing_role}")
                return

    try:
        use_server = (
            ss.server is not None
            and ss.server.is_running()
            and config.config_type in ("baseline", "moe", "spec", "moe_spec")
        )

        if use_server:
            spec_k = config.spec_k if config.config_type in ("spec", "moe_spec") else None
            result = ss.server.run_inference(
                prompt=question.prompt,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                timeout=params["timeout"],
                speculative_n_max=spec_k,
                image_path=question.image_path,
            )
        else:
            result = executor.run_inference(
                model_path=model_path,
                config=config,
                prompt=question.prompt,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                timeout=params["timeout"],
                mmproj_path=mmproj_path,
                image_path=question.image_path,
                context_size=question.context_tokens,
                role=role,
            )

        if result.timed_out:
            stats["errors"] += 1
            parsed = parse_output(result.raw_output)
            if parsed.response and len(parsed.response.strip()) > 50:
                char_count = len(parsed.response)
                print(f"    [TIMEOUT] {role}/{config.name}/{question.id}: partial output saved ({char_count} chars)")
            else:
                print(f"    [TIMEOUT] {role}/{config.name}/{question.id}: no usable output")
                return

        elif not result.success:
            stats["errors"] += 1
            err_hint = (_extract_error_hint(result.stderr) if result.stderr else "")
            if not err_hint and result.raw_output:
                first_line = result.raw_output.split('\n')[0][:80]
                if not first_line.startswith('build:'):
                    err_hint = first_line
            err_hint = err_hint or f"exit={result.exit_code}"
            print(f"    [ERROR] {role}/{config.name}/{question.id}: {err_hint}")
            return

        else:
            parsed = parse_output(result.raw_output)

        qresult = QuestionResult(
            question_id=question.id,
            prompt=question.prompt,
            response=parsed.response,
            tokens_per_second=parsed.tokens_per_second,
            prompt_tokens=parsed.prompt_tokens,
            completion_tokens=parsed.completion_tokens,
            total_time_ms=parsed.total_time_ms,
            algorithmic_score=None,
            score_reason=None,
            acceptance_rate=parsed.acceptance_rate,
        )

        results_manager.add_question_result(
            run_id=run_id,
            model_role=role,
            config_name=config.name,
            model_path=model_path,
            suite=suite_name,
            question_result=qresult,
        )

        tps = parsed.tokens_per_second
        tps_str = f"{tps:.1f}t/s" if tps else "---"
        print(f"      {config.name}/{suite_name}/{question.id}: {tps_str}", flush=True)
        stats["passed"] += 1

    except Exception as e:
        stats["errors"] += 1
        print(f"    [ERROR] {role}/{config.name}/{question.id}: {e}")


def run_benchmark(
    registry: ModelRegistry,
    executor: Executor,
    results_manager: ResultsManager,
    run_id: str,
    model_filter: Optional[str] = None,
    suite_filter: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    server_mode: bool = False,
    no_mmap: bool = False,
    skip_long_context: bool = False,
    vision_only: bool = False,
) -> dict:
    """Run the benchmark with nested progress bars.

    Outer bar: models
    Inner bar: configs × suites × questions for current model
    """
    stats = {"total": 0, "skipped": 0, "passed": 0, "errors": 0}  # "passed" = completed

    # Get models to test
    roles = registry.get_all_roles(include_deprecated=False)
    if model_filter:
        roles = [r for r in roles if r == model_filter]
    if vision_only:
        # Filter to only VL models (those with mmproj_path configured)
        roles = [r for r in roles if registry.get_mmproj_path(r) is not None]

    # Group roles by model path to avoid duplicate display
    model_to_roles: dict[str, list[str]] = {}
    role_sizes: dict[str, float] = {}

    for role in roles:
        model_path = registry.get_model_path(role)
        if model_path and os.path.exists(model_path):
            if model_path not in model_to_roles:
                model_to_roles[model_path] = []
            model_to_roles[model_path].append(role)

            # Get model size from registry or file
            config = registry.get_role_config(role)
            size_gb = config.get("model", {}).get("size_gb", 0) if config else 0
            if size_gb == 0:
                size_gb = os.path.getsize(model_path) / (1024**3)
            role_sizes[role] = size_gb

    # Build list of (model_path, size, roles) sorted by size (largest first)
    models_sorted = []
    for model_path, role_list in model_to_roles.items():
        size_gb = role_sizes[role_list[0]]  # All roles for same model have same size
        models_sorted.append((model_path, size_gb, role_list))
    models_sorted.sort(key=lambda x: x[1], reverse=False)

    # Flatten back to role list but preserve model grouping order
    valid_roles = []
    for model_path, size_gb, role_list in models_sorted:
        for role in role_list:
            valid_roles.append((role, size_gb))

    print(f"\nBenchmark: {run_id} | {len(models_sorted)} models, {len(valid_roles)} roles (smallest first)")

    printed_models: set[str] = set()
    model_tps: dict[str, float] = {}
    ss = _ServerState()

    # Outer progress bar: roles
    role_iter = tqdm(valid_roles, desc="Roles") if TQDM_AVAILABLE else valid_roles

    try:
      for role, size_gb in role_iter:
        model_path = registry.get_model_path(role)
        mmproj_path = registry.get_mmproj_path(role)  # VL models have mmproj
        arch = registry.get_architecture(role)
        configs = executor.get_configs_for_architecture(arch, role, registry)

        suite_names = get_suites_for_role(role, registry)
        if suite_filter:
            suite_names = [s for s in suite_names if s == suite_filter]
        if skip_long_context:
            suite_names = [s for s in suite_names if s != "long_context"]

        # LONG_CONTEXT OPTIMIZATION: If spec decode is available, use it for quality tests
        # (faster than baseline, produces identical output since same target model)
        spec_configs = [c for c in configs if c.config_type == "spec"]
        has_long_context = "long_context" in suite_names
        long_context_spec_config = spec_configs[0].name if (spec_configs and has_long_context) else None

        # PREFLIGHT CHECK: Skip model entirely if all tests are complete
        pending_tests, total_tests = count_pending_tests(run_id, role, configs, suite_names, force)
        if pending_tests == 0 and not dry_run:
            print(f"  [{role}] All {total_tests} tests complete - skipping", flush=True)
            stats["skipped"] += total_tests
            continue  # Skip to next role WITHOUT loading model

        # Run baseline speed test for each model (once per model, not per role)
        if model_path not in model_tps:
            if dry_run:
                # In dry run, use registry baseline_tps or default
                reg_tps = registry.get_baseline_tps(role)
                model_tps[model_path] = reg_tps if reg_tps else REFERENCE_TPS
            else:
                # Find the baseline config to measure actual speed
                baseline_config = next((c for c in configs if c.name == "baseline"), None)
                if baseline_config:
                    try:
                        speed_result = executor.run_inference(
                            model_path=model_path,
                            config=baseline_config,
                            prompt=SPEED_TEST_PROMPT,
                            max_tokens=_DEFAULT_MAX_TOKENS,
                            temperature=_DEFAULT_TEMPERATURE,
                            timeout=_compute_timeout(size_gb),
                            mmproj_path=mmproj_path,  # VL models need mmproj even for text
                            role=role,  # For paged attention on 70B+ models
                        )
                        if speed_result.success and not speed_result.timed_out:
                            parsed = parse_output(speed_result.raw_output)
                            if parsed.tokens_per_second and parsed.tokens_per_second > 0:
                                model_tps[model_path] = parsed.tokens_per_second
                            else:
                                model_tps[model_path] = REFERENCE_TPS
                        else:
                            model_tps[model_path] = REFERENCE_TPS
                    except Exception as e:
                        model_tps[model_path] = REFERENCE_TPS
                else:
                    # No baseline config, use registry or default
                    reg_tps = registry.get_baseline_tps(role)
                    model_tps[model_path] = reg_tps if reg_tps else REFERENCE_TPS

        # Calculate timeout multiplier based on measured speed
        measured_tps = model_tps.get(model_path, REFERENCE_TPS)
        timeout_multiplier = max(MIN_TIMEOUT_MULTIPLIER, REFERENCE_TPS / measured_tps)

        # Preload suites and count questions (with timeout multiplier applied)
        suites_data = {}
        for sname in suite_names:
            suite = load_suite(sname)
            if suite:
                suites_data[sname] = {"suite": suite, "params": get_inference_params(suite, timeout_multiplier)}

        total_questions = sum(len(suites_data[s]["suite"].questions) for s in suites_data)
        inner_total = len(configs) * total_questions

        # Print model header once
        is_new_model = model_path not in printed_models
        if is_new_model:
            printed_models.add(model_path)
            model_name = Path(model_path).stem if model_path else role
            tps_str = f"{measured_tps:.1f} t/s"
            mult_str = f"{timeout_multiplier:.1f}x" if timeout_multiplier > 1.0 else "1x"
            print(f"\n  {model_name} ({size_gb:.1f}GB) @ {tps_str} → timeout {mult_str}", flush=True)
            print(f"    roles: {', '.join(model_to_roles[model_path])}", flush=True)

            if server_mode and not dry_run:
                _ensure_server(ss, model_path, configs[0] if configs else None, role,
                               size_gb, registry, no_mmap, mmproj_path, is_new_model=True)

        print(f"    [{role}] {pending_tests}/{inner_total} tests pending ({len(configs)} configs × {len(suite_names)} suites)", flush=True)

        for config in configs:
            # Server management per config
            if server_mode and ss.server is not None and ss.server.is_running():
                _ensure_server(ss, model_path, config, role, size_gb,
                               registry, no_mmap, mmproj_path, is_new_model=False)

            # Speed-test-only configs
            if config.speed_test_only:
                stats["total"] += 1
                if dry_run:
                    print(f"      [SPEED] {config.name} (inherits quality from {config.inherits_quality_from})", flush=True)
                    continue
                if not force and result_exists(run_id, role, config.name):
                    stats["skipped"] += 1
                    continue
                try:
                    _run_speed_test(executor, results_manager, config, model_path,
                                    size_gb, mmproj_path, role, run_id, stats)
                except Exception as e:
                    stats["errors"] += 1
                    print(f"    [ERROR] {role}/{config.name}: {e}")
                if config.name != long_context_spec_config:
                    continue

            # Quality benchmark loop
            for suite_name, sdata in suites_data.items():
                if config.config_type in ("lookup", "moe_lookup") and suite_name != "long_context":
                    continue
                if suite_name == "long_context" and config.name == "baseline" and long_context_spec_config:
                    continue
                if suite_name != "long_context" and config.name == long_context_spec_config:
                    continue

                suite = sdata["suite"]
                params = sdata["params"]

                for question in suite.questions:
                    stats["total"] += 1
                    if dry_run:
                        continue
                    _run_quality_question(
                        executor, results_manager, ss, config, model_path,
                        mmproj_path, role, run_id, suite_name, question, params,
                        stats, force,
                    )

    finally:
        if ss.server is not None:
            print(f"\n  [SERVER] Stopping server...", flush=True)
            ss.stop()

    print(f"\nDone: {stats['passed']} completed, {stats['skipped']} skipped, {stats['errors']} errors")
    return stats


def process_queue(
    registry: ModelRegistry,
    executor: Executor,
    results_manager: ResultsManager,
) -> None:
    """Process queued models from the queue file."""
    if not os.path.exists(QUEUE_FILE):
        print("No benchmark queue found.")
        return

    with open(QUEUE_FILE) as f:
        queued_models = [line.strip() for line in f if line.strip()]

    if not queued_models:
        print("Benchmark queue is empty.")
        return

    print(f"Processing {len(queued_models)} queued models...")

    for model in queued_models:
        run_id = results_manager.generate_run_id()
        run_benchmark(
            registry=registry,
            executor=executor,
            results_manager=results_manager,
            run_id=run_id,
            model_filter=model,
        )

    # Clear queue
    os.unlink(QUEUE_FILE)
    print("Queue processed and cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./run_benchmark.py                     # Run all benchmarks
  ./run_benchmark.py --force             # Force re-run
  ./run_benchmark.py --model coder_escalation --suite thinking
  ./run_benchmark.py --vision-only       # Only VL models (with mmproj)
  ./run_benchmark.py --vision-only --suite vl  # VL models, VL suite only
  ./run_benchmark.py --dry-run           # Preview what would run
  ./run_benchmark.py --process-queue     # Run queued models
        """,
    )
    parser.add_argument("--model", "-m", help="Only run this model role")
    parser.add_argument("--suite", "-s", help="Only run this suite")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-run (don't skip existing)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would run without executing")
    parser.add_argument("--process-queue", action="store_true", help="Process queued models")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume the latest run (skip completed, retry errors)")
    parser.add_argument("--server-mode", action="store_true", help="Keep model in RAM via llama-server (faster for large models)")
    parser.add_argument("--no-mmap", action="store_true", help="Use bulk read instead of mmap (may be faster for cold loads)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-suites", action="store_true", help="List available suites")
    parser.add_argument("--skip-long-context", action="store_true", help="Skip long_context suite (saves time on quick runs)")
    parser.add_argument("--vision-only", action="store_true", help="Only benchmark vision-language models (models with mmproj_path)")

    args = parser.parse_args()

    # Initialize components
    registry = load_registry()
    executor = Executor(registry)
    results_manager = ResultsManager()

    # List modes
    if args.list_models:
        print("Available models:")
        for role in registry.get_all_roles():
            tier = registry.get_tier(role)
            arch = registry.get_architecture(role)
            path = registry.get_model_path(role)
            exists = "✓" if path and os.path.exists(path) else "✗"
            print(f"  [{tier}] {role:30} {arch:15} {exists}")
        return

    if args.list_suites:
        print("Available suites:")
        for name in get_all_suite_names():
            suite = load_suite(name)
            if suite:
                print(f"  {name:25} ({len(suite.questions)} questions) - {suite.description[:50]}...")
        return

    # Process queue mode
    if args.process_queue:
        lock_fd = acquire_lock()
        if lock_fd is None:
            print("ERROR: Another benchmark is already running.")
            sys.exit(1)
        try:
            process_queue(registry, executor, results_manager)
        finally:
            release_lock(lock_fd)
        return

    # Acquire lock for benchmark run
    if not args.dry_run:
        lock_fd = acquire_lock()
        if lock_fd is None:
            print("ERROR: Another benchmark is already running.")
            print("       Kill the existing process or wait for it to complete.")
            sys.exit(1)
    else:
        lock_fd = None

    try:
        if args.resume:
            run_id = results_manager.get_latest_run()
            if not run_id:
                print("ERROR: No previous run to resume. Start a new run without --resume.")
                sys.exit(1)
            print(f"Resuming run: {run_id}")
        else:
            run_id = results_manager.generate_run_id()

        run_benchmark(
            registry=registry,
            executor=executor,
            results_manager=results_manager,
            run_id=run_id,
            model_filter=args.model,
            suite_filter=args.suite,
            force=args.force,
            dry_run=args.dry_run,
            server_mode=args.server_mode,
            no_mmap=args.no_mmap,
            skip_long_context=args.skip_long_context,
            vision_only=args.vision_only,
        )
    finally:
        if lock_fd is not None:
            release_lock(lock_fd)


if __name__ == "__main__":
    main()
