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
import subprocess
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

from suites import PROMPTS_DIR, load_suite, get_suites_for_role, get_inference_params, get_all_suite_names
from results import (
    ResultsManager,
    QuestionResult,
    result_exists,
    result_exists_for_model,
    copy_result_from_role,
    get_slowest_questions,
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
_DEFAULT_MAX_TOKENS = 512
_LOOKUP_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.6
_SERVER_STARTUP_TIMEOUT_BASE = 600

# Timeout scaling: timeout = max(base, size_gb * multiplier + buffer)
_TIMEOUT_SIZE_MULTIPLIER = 10
_TIMEOUT_SIZE_BUFFER = 300

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
    speed_questions: int = 0,
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
            n_speed = max(1, speed_questions) if speed_questions > 0 else 1
            total += n_speed
            if force or not result_exists(run_id, role, config.name):
                pending += n_speed
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

    __slots__ = ("server", "model_path", "experts", "draft_path", "lookup",
                 "lookup_ngram", "_model_flags")

    def __init__(self) -> None:
        self.server: Optional[ServerManager] = None
        self.model_path: Optional[str] = None
        self.experts: Optional[int] = None
        self.draft_path: Optional[str] = None
        self.lookup: bool = False
        self.lookup_ngram: Optional[int] = None  # ngram-simple n-size baked into the live server
        self._model_flags: dict = {}

    def _start_server(self, model_path, moe_override=None, registry=None,
                      no_mmap=False, role=None, mmproj_path=None,
                      draft_model_path=None, draft_max=None, lookup=False,
                      spec_type=None, spec_ngram_size_n=None,
                      runtime_requirements=None, draft_p_min=None,
                      threads_draft=None, ubatch_override=None):
        """Start server using cached model config flags."""
        # _model_flags may contain runtime_requirements / draft_model_path / draft_p_min /
        # threads_draft / ubatch_override from cache; explicit params take precedence.
        flags = dict(self._model_flags or {})
        cached_runtime = flags.pop("runtime_requirements", None)
        cached_draft = flags.pop("draft_model_path", None)
        cached_p_min = flags.pop("draft_p_min", None)
        cached_threads_draft = flags.pop("threads_draft", None)
        cached_ubatch = flags.pop("ubatch_override", None)
        runtime_requirements = runtime_requirements if runtime_requirements is not None else cached_runtime
        draft_model_path = draft_model_path if draft_model_path is not None else cached_draft
        draft_p_min = draft_p_min if draft_p_min is not None else cached_p_min
        threads_draft = threads_draft if threads_draft is not None else cached_threads_draft
        ubatch_override = ubatch_override if ubatch_override is not None else cached_ubatch
        return self.server.start(model_path, moe_override=moe_override,
                                 registry=registry, no_mmap=no_mmap, role=role,
                                 mmproj_path=mmproj_path,
                                 draft_model_path=draft_model_path,
                                 draft_max=draft_max, lookup=lookup,
                                 spec_type=spec_type,
                                 spec_ngram_size_n=spec_ngram_size_n,
                                 runtime_requirements=runtime_requirements,
                                 draft_p_min=draft_p_min,
                                 threads_draft=threads_draft,
                                 ubatch_override=ubatch_override,
                                 **flags)

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
    with_lookup: bool = False,
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
            # Check if this model uses ngram-simple instead of --lookup
            # 2026-05-07: also honor explicit spec_type on speculative_decoding accel
            # blocks (e.g. gemma4 MTP). Without this, MTP-equipped models silently
            # fell back to baseline mode at run_benchmark.py launch — see
            # progress/2026-05/2026-05-07.md.
            accel = registry.get_acceleration(role) if with_lookup else {}
            _accel_for_spec = registry.get_acceleration(role) or {}
            if _accel_for_spec.get("type") == "ngram_lookup":
                use_spec_type = _accel_for_spec.get("spec_type")
            elif _accel_for_spec.get("type") == "speculative_decoding" and _accel_for_spec.get("spec_type"):
                # MTP / Eagle / etc. — explicit spec_type means baseline launches
                # should activate the model's NATIVE speculative path.
                use_spec_type = _accel_for_spec.get("spec_type")
            else:
                use_spec_type = None
            use_lookup = with_lookup and not use_spec_type
            accel_draft_max = _accel_for_spec.get("draft_max") if use_spec_type else None
            # Check if model requires chat template (e.g. gemma4)
            role_config = registry.get_role_config(role) if registry else None
            model_cfg = role_config.get("model", {}) if role_config else {}
            # Resolve draft model path from accel.draft_role (when spec_type is set).
            # Without this, executor.start() never sees the -md flag and spec decode
            # fails to engage even when the binary supports it.
            draft_model_path_resolved = None
            if use_spec_type and _accel_for_spec.get("draft_role") and registry:
                _draft_role = _accel_for_spec["draft_role"]
                _draft_cfg = registry.get_role_config(_draft_role) or {}
                _draft_model_cfg = _draft_cfg.get("model", {})
                _draft_path = _draft_model_cfg.get("path")
                if _draft_path:
                    # Resolve relative paths via model_base_path (same logic as main model)
                    if not _draft_path.startswith("/"):
                        _base = registry.config.get("runtime_defaults", {}).get(
                            "model_base_path", "/mnt/raid0/llm/lmstudio/models"
                        )
                        _draft_path = os.path.join(_base, _draft_path)
                    draft_model_path_resolved = _draft_path
            # Per-role runtime_requirements (override binary path + LD_LIBRARY_PATH)
            # for models that need a non-default llama.cpp build (e.g. ik_llama.cpp
            # PR #1744 for gemma4 MTP).
            runtime_reqs = (role_config or {}).get("runtime_requirements") if role_config else None
            # draft_p_min (ik_llama.cpp): 0.0 lets drafter produce full draft_max chains
            # greedily (best for MTP); 0.8 default rejects low-confidence draft tokens
            # early (drafter drops out at 1.8 tokens/call instead of full draft_max=3).
            # Read explicit accel.draft_p_min when set; None = use server default.
            draft_p_min_resolved = _accel_for_spec.get("draft_p_min") if use_spec_type else None
            # threads_draft: dedicate N cores to the drafter (default = same as --threads,
            # which causes drafter to monopolize all 96 cores during its forward).
            threads_draft_resolved = _accel_for_spec.get("threads_draft") if use_spec_type else None
            # ubatch override: per-accel preferred ubatch (e.g., gemma4 MTP wants 512).
            ubatch_override_resolved = _accel_for_spec.get("ubatch") if use_spec_type else None
            use_chat = model_cfg.get("use_chat_api", False)
            chat_tmpl = model_cfg.get("chat_template")  # Override template (e.g., "chatml" for Qwen3.6)
            no_jinja = model_cfg.get("no_jinja", False)  # --no-jinja for legacy template path (avoids PEG parser crash on <think>)
            # KV cache type overrides (e.g. Qwen3.6 needs bf16/q8_0, not default f16)
            kv_cfg = model_cfg.get("kv_cache", {})
            ctk = kv_cfg.get("type_k")
            ctv = kv_cfg.get("type_v")
            # Reasoning mode (off/on/auto — "off" disables think blocks for M2.7, Qwen3.6)
            reasoning = model_cfg.get("reasoning")
            reasoning_budget = model_cfg.get("reasoning_budget")  # Force </think> after N tokens

            accel_str = f" +{use_spec_type}" if use_spec_type else (" +lookup" if use_lookup else "")
            chat_str = f" +{chat_tmpl}" if chat_tmpl else (" +chat" if use_chat else "")
            kv_str = f" kv={ctk}/{ctv}" if ctk else ""
            reason_str = f" reason={reasoning}" if reasoning is not None and reasoning is not False else ""
            noj_str = " --no-jinja" if no_jinja else ""
            print(f"    [SERVER] Starting llama-server{accel_str}{chat_str}{kv_str}{reason_str}{noj_str} (model will stay in RAM)...", flush=True)
            # Performance-critical env vars shared across models (validated April 29 canonical OMP stack)
            env_vars = model_cfg.get("env_vars", {}) or {}
            ss._model_flags = {
                "use_chat_api": use_chat,
                "chat_template": chat_tmpl,
                "no_jinja": no_jinja,
                "cache_type_k": ctk,
                "cache_type_v": ctv,
                "reasoning": reasoning,
                "reasoning_budget": reasoning_budget,
                "env_vars": env_vars,
                "runtime_requirements": runtime_reqs,
                "draft_model_path": draft_model_path_resolved,
                "draft_p_min": draft_p_min_resolved,
                "threads_draft": threads_draft_resolved,
                "ubatch_override": ubatch_override_resolved,
            }
            ss.server = ServerManager(port=8080)
            ss._start_server(model_path, registry=registry,
                           no_mmap=no_mmap, role=role, mmproj_path=mmproj_path,
                           lookup=use_lookup, spec_type=use_spec_type,
                           draft_max=accel_draft_max,
                           draft_model_path=draft_model_path_resolved,
                           runtime_requirements=runtime_reqs,
                           draft_p_min=draft_p_min_resolved,
                           threads_draft=threads_draft_resolved,
                           ubatch_override=ubatch_override_resolved)
            timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
            if not ss.server.wait_ready(timeout=timeout):
                print(f"    [SERVER] Failed to start, falling back to subprocess mode", flush=True)
                ss.server = None
            else:
                ss.model_path = model_path
                ss.experts = None
                ss.draft_path = None
                ss.lookup = with_lookup
                print(f"    [SERVER] Ready, model loaded in RAM{accel_str}", flush=True)
        return

    if ss.server is not None and not ss.server.is_running():
        # Server crashed (e.g. GGML_ASSERT from spec+long_context) — restart it
        print(f"      [SERVER] Crashed, restarting...", flush=True)
        moe_override = None
        if ss.experts is not None:
            moe_key = registry.get_moe_override_key(role) or "qwen3moe.expert_used_count"
            moe_override = f"{moe_key}=int:{ss.experts}"
        ss.stop()
        accel = registry.get_acceleration(role) if with_lookup else {}
        use_spec_type = accel.get("spec_type") if accel.get("type") == "ngram_lookup" else None
        use_lookup = with_lookup and not use_spec_type
        accel_draft_max = accel.get("draft_max") if use_spec_type else None
        ss.server = ServerManager(port=8080)
        ss._start_server(model_path, moe_override=moe_override, registry=registry,
                       no_mmap=no_mmap, role=role, mmproj_path=mmproj_path,
                       lookup=use_lookup, spec_type=use_spec_type,
                       draft_max=accel_draft_max)
        timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
        if not ss.server.wait_ready(timeout=timeout):
            print(f"      [SERVER] Failed to restart after crash, falling back to subprocess", flush=True)
            ss.server = None
            ss.experts = None
            return
        else:
            ss.draft_path = None
            ss.lookup = with_lookup
            print(f"      [SERVER] Recovered", flush=True)

    if ss.server is None:
        return

    # --- 2. MoE expert count change ---
    if config.config_type in ("moe", "moe_lookup", "moe_spec", "moe_spec_lookup"):
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
        ss._start_server(model_path, moe_override=moe_override, registry=registry,
                       no_mmap=no_mmap, role=role, mmproj_path=mmproj_path)
        timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
        if not ss.server.wait_ready(timeout=timeout):
            print(f"      [SERVER] Failed to restart, falling back to subprocess", flush=True)
            ss.server = None
            ss.experts = None
        else:
            ss.experts = required_experts
            ss.draft_path = None
            ss.lookup = False
            print(f"      [SERVER] Ready", flush=True)

    if ss.server is None or not ss.server.is_running():
        return

    # --- 3. Draft model and/or lookup change ---
    # Strategy: minimize server restarts by reusing servers with superset capabilities.
    # A server with draft+lookup can serve spec, lookup, AND spec_lookup configs:
    #   - spec configs: send speculative.n_max=K (draft runs, lookup loaded but idle)
    #   - lookup configs: omit speculative.n_max (lookup runs, draft loaded but idle)
    #   - spec_lookup configs: send speculative.n_max=K (both active)
    # So we only restart when draft model or MoE experts change, and we always enable
    # --lookup if the config needs it OR we already have it (never downgrade).
    if config.config_type in ("spec", "moe_spec", "moe_spec_lookup", "spec_lookup"):
        required_draft = config.draft_model_path
        # Enable lookup if this config needs it, OR keep it if already on
        required_lookup = ss.lookup or config.config_type in ("moe_spec_lookup", "spec_lookup")
        # Surface registry-declared spec_type (e.g. "mtp" for Gemma 4 external-assistant drafters).
        accel = registry.get_acceleration(role)
        required_spec_type = accel.get("spec_type") if accel.get("type") == "speculative_decoding" else None
        needs_restart = (required_draft != ss.draft_path) or (required_lookup and not ss.lookup)

        if needs_restart:
            if config.config_type in ("moe_spec", "moe_spec_lookup"):
                moe_key = registry.get_moe_override_key(role) or "qwen3moe.expert_used_count"
                moe_override = f"{moe_key}=int:{config.moe_experts}"
            else:
                moe_override = None

            draft_name = Path(required_draft).stem if required_draft else "unknown"
            lookup_str = "+lookup" if required_lookup else ""
            spec_str = f"+{required_spec_type}" if required_spec_type else ""
            print(f"      [SERVER] Restarting with draft {draft_name}{lookup_str}{spec_str}...", flush=True)
            ss.stop()
            ss.server = ServerManager(port=8080)
            ss._start_server(
                model_path, moe_override=moe_override, registry=registry,
                no_mmap=no_mmap, role=role,
                draft_model_path=required_draft,
                draft_max=config.spec_k,
                mmproj_path=mmproj_path,
                lookup=required_lookup,
                spec_type=required_spec_type,
                # spec+lookup compounds: forward ngram size if config carries it
                # (Config.spec_lookup defaults lookup_ngram=4 when unset).
                spec_ngram_size_n=getattr(config, "lookup_ngram", None) if required_lookup else None,
            )
            timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
            if not ss.server.wait_ready(timeout=timeout):
                print(f"      [SERVER] Failed to restart with draft, falling back to subprocess", flush=True)
                ss.server = None
                ss.draft_path = None
                ss.lookup = False
            else:
                ss.draft_path = required_draft
                ss.lookup = required_lookup
                if config.config_type in ("moe_spec", "moe_spec_lookup"):
                    ss.experts = config.moe_experts
                print(f"      [SERVER] Ready with draft{lookup_str}", flush=True)

    elif config.config_type in ("lookup", "moe_lookup"):
        # Lookup-only: reuse current server if it already has --lookup (even with draft loaded).
        # A draft+lookup server works fine for lookup-only: just omit speculative.n_max.
        needs_restart = not ss.lookup

        if config.config_type == "moe_lookup":
            required_experts = config.moe_experts
            if required_experts != ss.experts:
                needs_restart = True

        if needs_restart:
            moe_override = None
            if config.config_type == "moe_lookup":
                moe_key = registry.get_moe_override_key(role) or "qwen3moe.expert_used_count"
                moe_override = f"{moe_key}=int:{config.moe_experts}"
            print(f"      [SERVER] Restarting with ngram-simple lookup for {config.name}...", flush=True)
            ss.stop()
            ss.server = ServerManager(port=8080)
            ss._start_server(
                model_path, moe_override=moe_override, registry=registry,
                no_mmap=no_mmap, role=role, mmproj_path=mmproj_path,
                lookup=True,
                # ngram size baked into server startup (upstream replaced our per-request
                # legacy --lookup flag). config.lookup_ngram=0 is the sweep sentinel; the
                # sweep itself restarts the server per ngram in _sweep_lookup_ngram.
                spec_ngram_size_n=getattr(config, "lookup_ngram", None) or None,
            )
            timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
            if not ss.server.wait_ready(timeout=timeout):
                print(f"      [SERVER] Failed to restart with lookup, skipping", flush=True)
                ss.server = None
                ss.lookup = False
            else:
                ss.lookup = True
                ss.draft_path = None
                if config.config_type == "moe_lookup":
                    ss.experts = config.moe_experts
                print(f"      [SERVER] Ready with --lookup", flush=True)


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
    ss=None,
) -> None:
    """Execute a speed-only benchmark for *config* (no quality questions).

    Prefers the live server (`ss.server`) when available — the subprocess
    fallback (`executor.run_inference`) requires `llama-speculative` /
    `llama-lookup` binaries that aren't built on production hosts (see
    preflight warning: "Missing subprocess binaries"). For spec / lookup
    configs the subprocess path therefore fails with exit=1 even when the
    server is running fine. Mirrors the pattern in `_sweep_lookup_ngram`.
    """
    is_lookup = config.config_type in ("lookup", "moe_lookup")
    speed_prompt = LOOKUP_SPEED_TEST_PROMPT if is_lookup else SPEED_TEST_PROMPT
    speed_max_tokens = _LOOKUP_MAX_TOKENS if is_lookup else _DEFAULT_MAX_TOKENS
    speed_timeout = _compute_timeout(size_gb, base=300 if is_lookup else 180)

    spec_k = getattr(config, "spec_k", None)
    if ss is not None and ss.server is not None and ss.server.is_running():
        result = ss.server.run_inference(
            prompt=speed_prompt,
            max_tokens=speed_max_tokens,
            temperature=_DEFAULT_TEMPERATURE,
            timeout=speed_timeout,
            speculative_n_max=spec_k,
        )
    else:
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
        # ServerManager.InferenceResult has no `stderr`/`exit_code` attrs;
        # subprocess InferenceResult has both. Use getattr to stay agnostic.
        stderr = getattr(result, "stderr", None)
        exit_code = getattr(result, "exit_code", None)
        err_hint = (_extract_error_hint(stderr, max_chars=60) if stderr else "")
        if not err_hint:
            err_hint = f"exit={exit_code}" if exit_code is not None else "no_output"
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


def _sweep_lookup_ngram(
    executor: Executor,
    results_manager: ResultsManager,
    ss,
    config_template,
    model_path: str,
    size_gb: float,
    mmproj_path: Optional[str],
    role: str,
    run_id: str,
    stats: dict,
    force: bool,
    registry,
    speed_questions: int,
    suite_names: list,
    baseline_run: Optional[str],
) -> None:
    """Binary peak search for optimal lookup ngram value.

    Divide-and-conquer from n=128 down. At each step, test the midpoint
    of the current range and narrow toward the peak. Converges in ~7 steps
    covering the full [2, 128] range.
    """
    is_moe = config_template.config_type == "moe_lookup"
    quality_ref = config_template.inherits_quality_from or "baseline"

    def _make_config(n: int):
        if is_moe:
            cfg = Config.compound_moe_lookup(
                config_template.moe_experts, config_template.moe_override_key, n
            )
        else:
            cfg = Config.lookup(n)
        cfg.speed_test_only = True
        cfg.inherits_quality_from = quality_ref
        return cfg

    def _test_ngram(n: int) -> Optional[float]:
        """Run speed test for ngram=n, return TPS or None on failure."""
        cfg = _make_config(n)
        stats["total"] += 1

        is_lookup = True
        speed_prompt = LOOKUP_SPEED_TEST_PROMPT
        speed_max_tokens = _LOOKUP_MAX_TOKENS
        speed_timeout = _compute_timeout(size_gb, base=300)

        # Upstream --spec-type ngram-simple sets ngram size at server startup
        # (legacy --lookup let us vary per-request — that flag is gone). Restart
        # the server with the new ngram size so the sweep actually exercises
        # different n values. If we already have a server with this ngram, reuse.
        moe_override = (
            f"{config_template.moe_override_key}=int:{config_template.moe_experts}"
            if is_moe else None
        )
        current_ngram = getattr(ss, "lookup_ngram", None)
        if (
            ss.server is None
            or not ss.server.is_running()
            or not ss.lookup
            or current_ngram != n
        ):
            if ss.server is not None:
                ss.stop()
            ss.server = ServerManager(port=8080)
            ss._start_server(
                model_path, moe_override=moe_override, registry=registry,
                no_mmap=False, role=role, mmproj_path=mmproj_path,
                lookup=True, spec_ngram_size_n=n,
            )
            timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
            if not ss.server.wait_ready(timeout=timeout):
                stats["errors"] += 1
                print(f"      ⚡ {cfg.name}: FAILED (server start)", flush=True)
                ss.server = None
                ss.lookup = False
                return None
            ss.lookup = True
            ss.lookup_ngram = n
            if is_moe:
                ss.experts = config_template.moe_experts

        result = ss.server.run_inference(
            prompt=speed_prompt,
            max_tokens=speed_max_tokens,
            temperature=_DEFAULT_TEMPERATURE,
            timeout=speed_timeout,
        )

        if result.timed_out or not result.success:
            stats["errors"] += 1
            print(f"      ⚡ {cfg.name}: FAILED", flush=True)
            return None

        parsed = parse_output(result.raw_output)
        tps = parsed.tokens_per_second or 0

        # Store result
        results_manager.add_speed_result(
            run_id=run_id,
            model_role=role,
            config_name=cfg.name,
            model_path=model_path,
            tokens_per_second=tps,
            inherits_quality_from=quality_ref,
            acceptance_rate=parsed.acceptance_rate,
        )

        print(f"      ⚡ {cfg.name}: {tps:.1f}t/s", flush=True)
        stats["passed"] += 1
        return tps if tps > 0 else None

    # Binary peak search over [lo, hi]
    lo, hi = 2, 128
    tested = {}
    print(f"      [LOOKUP SWEEP] Binary search for optimal ngram in [{lo}, {hi}]", flush=True)

    # Test boundaries first
    tps_lo = _test_ngram(lo)
    tps_hi = _test_ngram(hi)
    if tps_lo is not None:
        tested[lo] = tps_lo
    if tps_hi is not None:
        tested[hi] = tps_hi

    # Binary search: narrow toward peak
    while hi - lo > 2:
        mid = (lo + hi) // 2
        tps_mid = _test_ngram(mid)
        if tps_mid is not None:
            tested[mid] = tps_mid

        # Decide which half contains the peak
        tps_l = tested.get(lo, 0)
        tps_h = tested.get(hi, 0)

        if tps_l >= tps_h:
            hi = mid  # Peak is in lower half
        else:
            lo = mid  # Peak is in upper half

    # Final: test remaining midpoint if range is small
    if hi - lo == 2:
        mid = lo + 1
        tps_mid = _test_ngram(mid)
        if tps_mid is not None:
            tested[mid] = tps_mid

    # Report
    if tested:
        best_n = max(tested, key=tested.get)
        best_tps = tested[best_n]
        sorted_results = sorted(tested.items())
        curve = " | ".join(f"n={n}:{tps:.1f}" for n, tps in sorted_results)
        print(f"      [LOOKUP SWEEP] Best: ngram={best_n} @ {best_tps:.1f}t/s  [{curve}]", flush=True)
    else:
        print(f"      [LOOKUP SWEEP] No valid results", flush=True)


def _run_speed_question(
    executor: Executor,
    results_manager: ResultsManager,
    ss: _ServerState,
    config,
    model_path: str,
    size_gb: float,
    mmproj_path: Optional[str],
    role: str,
    run_id: str,
    suite_name: str,
    question_id: str,
    prompt: str,
    stats: dict,
    force: bool,
    registry: Optional["ModelRegistry"] = None,
) -> None:
    """Execute a speed benchmark on a specific question and store per-question result."""
    # Skip if already exists
    if not force and result_exists(run_id, role, config.name, suite_name, question_id):
        stats["skipped"] += 1
        return

    # Determine temperature (apply per-suite overrides if configured)
    temperature = _DEFAULT_TEMPERATURE
    if registry:
        override = registry.get_temperature_override(role, suite_name)
        if override is not None:
            temperature = override

    # Apply thinking disable trick if configured for this suite
    effective_prompt = prompt
    if registry:
        think_trick = registry.get_thinking_disable_trick(role, suite_name)
        if think_trick:
            effective_prompt = effective_prompt + think_trick

    speed_max_tokens = _LOOKUP_MAX_TOKENS if "lookup" in config.config_type else _DEFAULT_MAX_TOKENS
    speed_timeout = _compute_timeout(size_gb, base=300 if "lookup" in config.config_type else 180)

    try:
        use_server = (
            ss.server is not None
            and ss.server.is_running()
            and config.config_type in ("baseline", "moe", "spec", "moe_spec", "lookup", "moe_lookup", "moe_spec_lookup", "spec_lookup")
        )

        # spec_lookup and moe_spec_lookup require server mode (--lookup is our custom
        # llama.cpp flag, not available in llama-speculative binary)
        if not use_server and config.config_type in ("moe_spec_lookup", "spec_lookup"):
            stats["errors"] += 1
            print(f"    [SKIP] {role}/{config.name}/{suite_name}/{question_id}: {config.config_type} requires server mode", flush=True)
            return

        if use_server:
            spec_k = config.spec_k if config.config_type in ("spec", "moe_spec", "moe_spec_lookup", "spec_lookup") else None
            result = ss.server.run_inference(
                prompt=effective_prompt,
                max_tokens=speed_max_tokens,
                temperature=temperature,
                timeout=speed_timeout,
                speculative_n_max=spec_k,
            )
        else:
            result = executor.run_inference(
                model_path=model_path,
                config=config,
                prompt=effective_prompt,
                max_tokens=speed_max_tokens,
                temperature=temperature,
                timeout=speed_timeout,
                mmproj_path=mmproj_path,
                role=role,
            )

        if result.timed_out:
            stats["errors"] += 1
            print(f"    [TIMEOUT] {role}/{config.name}/{suite_name}/{question_id}")
            return

        if not result.success:
            stats["errors"] += 1
            err_hint = (_extract_error_hint(result.stderr, max_chars=60) if result.stderr else "") or f"exit={result.exit_code}"
            print(f"    [ERROR] {role}/{config.name}/{suite_name}/{question_id}: {err_hint}")
            return

        parsed = parse_output(result.raw_output)

        qresult = QuestionResult(
            question_id=question_id,
            prompt=prompt,
            response=parsed.response,
            tokens_per_second=parsed.tokens_per_second,
            prompt_tokens=parsed.prompt_tokens,
            completion_tokens=parsed.completion_tokens,
            total_time_ms=parsed.total_time_ms,
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
        acc_str = f"acc={parsed.acceptance_rate:.1%}" if parsed.acceptance_rate else ""
        print(f"      ⚡ {config.name}/{suite_name}/{question_id}: {tps_str} {acc_str}", flush=True)
        stats["passed"] += 1

    except Exception as e:
        stats["errors"] += 1
        print(f"    [ERROR] {role}/{config.name}/{suite_name}/{question_id}: {e}")


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
    registry: Optional["ModelRegistry"] = None,
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

    # Apply per-suite temperature override if configured
    effective_params = params
    if registry:
        temp_override = registry.get_temperature_override(role, suite_name)
        if temp_override is not None:
            effective_params = dict(params)
            effective_params["temperature"] = temp_override

    # Apply thinking disable trick if configured for this suite
    effective_prompt = question.prompt
    if registry:
        think_trick = registry.get_thinking_disable_trick(role, suite_name)
        if think_trick:
            effective_prompt = effective_prompt + think_trick

    # Read model-specific sampling params (e.g., repeat_penalty for Gemma4/M2.7)
    model_repeat_penalty = None
    model_disable_thinking = False
    max_tokens_mult = 1
    if registry:
        role_config = registry.get_role_config(role)
        if role_config:
            model_cfg = role_config.get("model", {})
            model_repeat_penalty = model_cfg.get("sampling", {}).get("repeat_penalty")
            model_disable_thinking = model_cfg.get("disable_thinking", False)
            max_tokens_mult = model_cfg.get("max_tokens_multiplier", 1)

    # Apply max_tokens multiplier for thinking models (need budget for reasoning + answer).
    # The timeout MUST scale alongside — generating Nx more tokens at the same t/s takes Nx
    # more wallclock. Without this coupling, thinking models (minimax-m27, qwen3.6 with
    # reasoning, ring-mini-linear, etc.) would hit the per-question timeout before consuming
    # the larger token budget and produce truncated/empty responses despite plenty of t/s.
    if max_tokens_mult > 1:
        effective_params = dict(effective_params)
        effective_params["max_tokens"] = int(effective_params["max_tokens"] * max_tokens_mult)
        effective_params["timeout"] = int(effective_params["timeout"] * max_tokens_mult)

    try:
        use_server = (
            ss.server is not None
            and ss.server.is_running()
            and config.config_type in ("baseline", "moe", "spec", "moe_spec", "moe_spec_lookup", "spec_lookup", "lookup", "moe_lookup")
        )

        if use_server:
            spec_k = config.spec_k if config.config_type in ("spec", "moe_spec", "moe_spec_lookup", "spec_lookup") else None
            result = ss.server.run_inference(
                prompt=effective_prompt,
                max_tokens=effective_params["max_tokens"],
                temperature=effective_params["temperature"],
                timeout=effective_params["timeout"],
                speculative_n_max=spec_k,
                image_path=question.image_path,
                repeat_penalty=model_repeat_penalty,
                disable_thinking=model_disable_thinking,
            )
        else:
            result = executor.run_inference(
                model_path=model_path,
                config=config,
                prompt=effective_prompt,
                max_tokens=effective_params["max_tokens"],
                temperature=effective_params["temperature"],
                timeout=effective_params["timeout"],
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
    speed_questions: int = 0,
    baseline_run: Optional[str] = None,
    with_lookup: bool = False,
    skip_moe_reduction: bool = False,
    skip_speed_tests: bool = False,
    all_suites: bool = False,
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
        if skip_moe_reduction:
            baseline_experts = registry.get_baseline_experts(role)
            configs = [c for c in configs if c.moe_experts is None or c.moe_experts == baseline_experts]
        if skip_speed_tests:
            # Drop every speed-only config (spec_*, lookup_*, moe<X>_spec_*, moe<X>_lookup_*).
            # Quality runs only need baseline + moe<X> configs that produce real per-question
            # answers. Cuts the per-MoE-model bench from ~30 min to ~5 min when only quality
            # data is wanted.
            configs = [c for c in configs if not getattr(c, "speed_test_only", False)]

        if all_suites:
            suite_names = [s for s in get_all_suite_names() if load_suite(s) and load_suite(s).questions]
            if not mmproj_path:
                suite_names = [s for s in suite_names if s != "vl"]
        else:
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
        pending_tests, total_tests = count_pending_tests(run_id, role, configs, suite_names, force, speed_questions)
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

        # Calculate timeout multiplier based on measured speed.
        # Optional per-model override `timeout_multiplier_min` (registry) acts as a floor —
        # use it for thinking models whose response sizes vary wildly (e.g. minimax-m27 with
        # interleaved-thinking can produce 8-16K-token responses on hard questions even at
        # 10-15 t/s, requiring >20 min per question).
        measured_tps = model_tps.get(model_path, REFERENCE_TPS)
        speed_multiplier = max(MIN_TIMEOUT_MULTIPLIER, REFERENCE_TPS / measured_tps)
        per_model_floor = MIN_TIMEOUT_MULTIPLIER
        if registry:
            role_cfg = registry.get_role_config(role)
            if role_cfg:
                per_model_floor = float(
                    role_cfg.get("model", {}).get("timeout_multiplier_min", MIN_TIMEOUT_MULTIPLIER)
                )
        timeout_multiplier = max(speed_multiplier, per_model_floor)

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
                               size_gb, registry, no_mmap, mmproj_path, is_new_model=True,
                               with_lookup=with_lookup)

        print(f"    [{role}] {pending_tests}/{inner_total} tests pending ({len(configs)} configs × {len(suite_names)} suites)", flush=True)

        for config in configs:
            # Preflight: skip configs where all tests are already complete
            if not force and not dry_run:
                if config.speed_test_only:
                    # Check if any speed questions still need running
                    _source = baseline_run or run_id
                    _qsrc = config.inherits_quality_from or "baseline"
                    _fetch_n = speed_questions * 10 if len(suite_names) < 7 else speed_questions
                    _slowest = get_slowest_questions(_source, role, _qsrc, _fetch_n) if speed_questions > 0 else []
                    if not _slowest and _source != run_id and speed_questions > 0:
                        _slowest = get_slowest_questions(run_id, role, _qsrc, _fetch_n)
                    if _slowest:
                        _slowest = [sq for sq in _slowest if sq["suite"] in suite_names][:speed_questions]
                    if speed_questions > 0 and _slowest:
                        all_done = all(
                            result_exists(run_id, role, config.name, sq["suite"], sq["question_id"])
                            for sq in _slowest
                        )
                    else:
                        all_done = result_exists(run_id, role, config.name)
                    if all_done:
                        if speed_questions > 0 and _slowest:
                            stats["skipped"] += len(_slowest)
                            stats["total"] += len(_slowest)
                        else:
                            stats["skipped"] += 1
                            stats["total"] += 1
                        continue
                else:
                    # Quality config: check if all suite/question combos exist
                    all_done = True
                    for sname, sdata in suites_data.items():
                        if config.config_type in ("lookup", "moe_lookup") and sname != "long_context":
                            continue
                        if sname == "long_context" and config.name == "baseline" and long_context_spec_config:
                            continue
                        if sname != "long_context" and config.name == long_context_spec_config:
                            continue
                        for question in sdata["suite"].questions:
                            if not result_exists(run_id, role, config.name, sname, question.id):
                                all_done = False
                                break
                        if not all_done:
                            break
                    if all_done:
                        continue

            # Server management per config
            if server_mode and not dry_run:
                if ss.server is None or (ss.server is not None and not ss.server.is_running()):
                    # Server was lost (crash, failed restart, or process died) — restart fresh baseline
                    reason = "crashed" if ss.server is not None else "stopped"
                    print(f"      [SERVER] Re-starting (was {reason})...", flush=True)
                    if ss.server is not None:
                        ss.stop()
                    ss.server = ServerManager(port=8080)
                    ss._start_server(model_path, registry=registry,
                                    no_mmap=no_mmap, role=role, mmproj_path=mmproj_path)
                    timeout = _compute_timeout(size_gb, base=_SERVER_STARTUP_TIMEOUT_BASE)
                    if not ss.server.wait_ready(timeout=timeout):
                        print(f"      [SERVER] Failed to re-start", flush=True)
                        ss.server = None
                    else:
                        ss.model_path = model_path
                        ss.experts = None
                        ss.draft_path = None
                        ss.lookup = False
                        print(f"      [SERVER] Recovered", flush=True)
                if ss.server is not None and ss.server.is_running():
                    _ensure_server(ss, model_path, config, role, size_gb,
                                   registry, no_mmap, mmproj_path, is_new_model=False,
                                   with_lookup=with_lookup)

            # Speed-test-only configs
            if config.speed_test_only:
                # Lookup sweep sentinel: ngram=0 triggers binary peak search
                if config.config_type in ("lookup", "moe_lookup") and config.lookup_ngram == 0:
                    if dry_run:
                        print(f"      [SPEED] {config.config_type} sweep (binary search n=2..128)", flush=True)
                    else:
                        _sweep_lookup_ngram(
                            executor, results_manager, ss, config, model_path,
                            size_gb, mmproj_path, role, run_id, stats, force,
                            registry, speed_questions, suite_names, baseline_run,
                        )
                    continue

                if speed_questions > 0:
                    # Run on N slowest baseline questions instead of fixed prompt
                    source_run = baseline_run or run_id
                    quality_source = config.inherits_quality_from or "baseline"
                    # Fetch extra candidates and filter to allowed suites (e.g. --skip-long-context)
                    fetch_n = speed_questions * 10 if len(suite_names) < 7 else speed_questions
                    slowest = get_slowest_questions(source_run, role, quality_source, fetch_n)
                    if not slowest and source_run != run_id:
                        slowest = get_slowest_questions(run_id, role, quality_source, fetch_n)
                    if slowest:
                        slowest = [sq for sq in slowest if sq["suite"] in suite_names][:speed_questions]
                    if not slowest:
                        # Fallback to fixed prompt
                        print(f"      [WARN] No baseline results for {role}/{quality_source}, falling back to fixed prompt", flush=True)
                        stats["total"] += 1
                        if dry_run:
                            print(f"      [SPEED] {config.name} (fixed prompt, inherits quality from {config.inherits_quality_from})", flush=True)
                            continue
                        if not force and result_exists(run_id, role, config.name):
                            stats["skipped"] += 1
                            continue
                        try:
                            _run_speed_test(executor, results_manager, config, model_path,
                                            size_gb, mmproj_path, role, run_id, stats,
                                            ss=ss)
                        except Exception as e:
                            stats["errors"] += 1
                            print(f"    [ERROR] {role}/{config.name}: {e}")
                    else:
                        for sq in slowest:
                            stats["total"] += 1
                            if dry_run:
                                print(f"      [SPEED] {config.name} on {sq['suite']}/{sq['question_id']} ({sq['tokens_per_second']:.1f}t/s baseline)", flush=True)
                                continue
                            _run_speed_question(
                                executor, results_manager, ss, config, model_path,
                                size_gb, mmproj_path, role, run_id,
                                sq["suite"], sq["question_id"], sq["prompt"],
                                stats, force, registry,
                            )
                else:
                    # Original fixed-prompt speed test
                    stats["total"] += 1
                    if dry_run:
                        print(f"      [SPEED] {config.name} (inherits quality from {config.inherits_quality_from})", flush=True)
                        continue
                    if not force and result_exists(run_id, role, config.name):
                        stats["skipped"] += 1
                        continue
                    try:
                        _run_speed_test(executor, results_manager, config, model_path,
                                        size_gb, mmproj_path, role, run_id, stats,
                                        ss=ss)
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
                        stats, force, registry,
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
    parser.add_argument("--resume", "-r", action="store_true", help="(default behavior, kept for compatibility)")
    parser.add_argument("--new-run", action="store_true", help="Force a new run directory instead of resuming the latest")
    parser.add_argument("--server-mode", action="store_true", help="Keep model in RAM via llama-server (faster for large models)")
    parser.add_argument("--no-mmap", action="store_true", help="Use bulk read instead of mmap (may be faster for cold loads)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-suites", action="store_true", help="List available suites")
    parser.add_argument("--skip-long-context", action="store_true", help="Skip long_context suite (saves time on quick runs)")
    parser.add_argument("--skip-moe-reduction", action="store_true", help="Skip MoE expert reduction configs; only run baseline + spec/lookup/spec+lookup at full expert count")
    parser.add_argument("--skip-speed-tests", action="store_true", help="Skip all speed-only configs (spec, lookup, moe+spec, moe+lookup, etc.). Quality benchmarks only — useful for tool_compliance / agentic-only runs.")
    parser.add_argument("--vision-only", action="store_true", help="Only benchmark vision-language models (models with mmproj_path)")
    parser.add_argument("--speed-questions", type=int, default=0, help="Run speed configs on N slowest baseline questions (0=fixed prompt)")
    parser.add_argument("--baseline-run", type=str, default=None, help="Pull slowest questions from this run ID (for models with existing baselines)")
    parser.add_argument("--with-lookup", action="store_true", help="Enable --lookup on server for all configs (accelerates baseline quality runs)")
    parser.add_argument("--all-suites", action="store_true", help="Run all quality suites regardless of role mapping (VL excluded for non-VL models)")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Skip the canonical-recipe preflight gate (uptime / libomp / launcher wrapping / tripwire / "
            "freq-under-load). Use only when you've just run the preflight manually and want to start "
            "another sweep within minutes. The preflight catches the multi-day-uptime CPU freq throttle "
            "and AOCC libomp issues that cost ~50%% throughput silently."
        ),
    )

    args = parser.parse_args()

    # Canonical-recipe preflight gate. Run before any heavy work (and before any
    # subprocess that would invalidate freq-under-load measurement). Skipped if
    # the user is in a non-execution mode (list/dry-run) or has explicitly opted
    # out via --skip-preflight.
    _is_dry_or_list = (
        args.dry_run
        or args.list_models
        or args.list_suites
    )
    if not args.skip_preflight and not _is_dry_or_list:
        preflight_path = Path(__file__).resolve().parent.parent / "preflight_canonical.py"
        if preflight_path.exists():
            print(f"\n[run_benchmark] Running canonical preflight: {preflight_path}\n")
            rc = subprocess.run([sys.executable, str(preflight_path)]).returncode
            if rc != 0:
                print(
                    "\n[run_benchmark] Preflight FAILED — refusing to proceed. "
                    "Address the fix above (typically: reboot for freq throttle, or "
                    "rebuild for libomp / launcher drift), then re-run. To bypass "
                    "the gate (NOT recommended), pass --skip-preflight.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print()  # blank line between preflight and benchmark output
        else:
            print(
                f"[run_benchmark] WARNING: preflight script not found at {preflight_path}; "
                "proceeding without canonical-recipe gate.",
                file=sys.stderr,
            )

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
        try:
            from dataset_adapters import ADAPTER_SUITES
        except ImportError:
            adapter_suites: set[str] = set()
        else:
            adapter_suites = set(ADAPTER_SUITES)
        prompts_dir = Path(PROMPTS_DIR)
        print("Available suites:")
        for name in get_all_suite_names(include_adapters=True):
            if name in adapter_suites and not (prompts_dir / f"{name}.yaml").exists():
                print(f"  {name:25} (adapter-backed; load with --suite {name})")
                continue
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
        if args.baseline_run:
            # Use baseline run as run_id so skip logic sees existing results
            run_id = args.baseline_run
            print(f"Continuing run: {run_id} (from --baseline-run)")
        elif args.new_run:
            run_id = results_manager.generate_run_id()
            print(f"New run: {run_id}")
        else:
            # Default: resume latest run (all results accumulate in one place)
            run_id = results_manager.get_latest_run()
            if not run_id:
                run_id = results_manager.generate_run_id()
                print(f"No previous run found, starting new: {run_id}")
            else:
                print(f"Resuming run: {run_id}")

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
            speed_questions=args.speed_questions,
            baseline_run=args.baseline_run,
            with_lookup=args.with_lookup,
            skip_moe_reduction=args.skip_moe_reduction,
            skip_speed_tests=args.skip_speed_tests,
            all_suites=args.all_suites,
        )
    finally:
        if lock_fd is not None:
            release_lock(lock_fd)


if __name__ == "__main__":
    main()
