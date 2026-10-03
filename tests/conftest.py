#!/usr/bin/env python3
"""Pytest configuration and fixtures for the orchestrator test suite.

Memory Safety:
    This file includes a memory guard that warns/exits if available RAM is below
    a safe threshold. This prevents crashes from running tests with pytest-xdist
    parallel workers that each load models.

    The 192-thread EPYC system with 1.13TB RAM can still crash if pytest spawns
    too many workers that each initialize the API (which loads the TaskEmbedder
    model). The lazy loading in src/api.py prevents this for mock mode tests,
    but this guard provides an additional safety net.

    The memory check is SKIPPED in CI environments (detected via CI or
    ORCHESTRATOR_MOCK_MODE env vars) since CI runners have limited RAM but
    only run mock-mode tests that don't load models.

Usage:
    pytest tests/              # Normal test run with memory check
    pytest tests/ -n 4         # Safe parallel execution (max 4 workers)
    pytest tests/ -n auto      # DANGEROUS on this machine - avoid

See also:
    - CLAUDE.md for memory constraints
    - research/ESCALATION_FLOW.md for memory pool architecture
"""

import json
import os
import warnings
from unittest.mock import MagicMock

import pytest

from src.config import reset_config
from src.api.state import AppState


# Memory threshold in GB (fail if less available)
MEMORY_THRESHOLD_GB = 100

# Check if running in CI environment (GitHub Actions, etc.)
IS_CI = os.environ.get("CI") == "true" or os.environ.get("ORCHESTRATOR_MOCK_MODE") == "true"

# Maximum safe parallel workers for this machine
# -n 8 gives 4x speedup (67s → 17s), -n 4 gives 3x (67s → 23s)
MAX_SAFE_WORKERS = 8


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-server",
        action="store_true",
        default=False,
        help="Run tests that require a live llama-server instance",
    )
    parser.addoption(
        "--run-ocr-server",
        action="store_true",
        default=False,
        help="Run tests that require a live OCR server",
    )
    parser.addoption(
        "--run-live-models",
        action="store_true",
        default=False,
        help="Run tests against live models (requires orchestrator)",
    )


def pytest_configure(config):
    """Check memory and register custom markers before running tests."""
    # Register custom markers
    config.addinivalue_line("markers", "heavy: marks tests that load models (may need more memory)")
    config.addinivalue_line("markers", "real_mode: marks tests that require real inference servers")
    config.addinivalue_line(
        "markers", "requires_server: marks tests that require a live llama-server"
    )
    config.addinivalue_line(
        "markers", "requires_live_models: marks tests that require live orchestrator models"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")

    # Check available memory (skip in CI - mock mode tests don't load models)
    if IS_CI:
        pass  # Skip memory check in CI environments
    else:
        try:
            import psutil

            free_gb = psutil.virtual_memory().available / (1024**3)

            if free_gb < MEMORY_THRESHOLD_GB:
                # Hard fail if memory is critically low
                pytest.exit(
                    f"DANGER: Only {free_gb:.0f}GB free RAM. "
                    f"Need {MEMORY_THRESHOLD_GB}GB+ for safe testing.\n"
                    "This machine can crash if tests load too many models.\n"
                    "Free up memory or wait for other processes to complete.",
                    returncode=1,
                )
            elif free_gb < MEMORY_THRESHOLD_GB * 2:
                # Warn if memory is getting low
                warnings.warn(
                    f"Low memory: {free_gb:.0f}GB free. Tests may be slow. Consider freeing memory.",
                    UserWarning,
                )
        except ImportError:
            warnings.warn(
                "psutil not installed - cannot check memory. Install with: pip install psutil",
                UserWarning,
            )


@pytest.fixture(scope="session", autouse=True)
def _pin_runtime_feature_flags(tmp_path_factory):
    """Pin the runtime feature-flag file so the suite is REPRODUCIBLE.

    `src.features.runtime_flags_path()` reads
    `orchestration/runtime_flags.json` — a gitignored file the RUNNING API
    rewrites at will (`set_by: api:127.0.0.1`). So unit results depended on
    whatever the live orchestrator last toggled, and could change *mid-run*:
    during a 2026-08-02 attribution pass the `memrl` flag flipped while the
    suite was executing.

    The cost of that was not hypothetical. Comparing a clean-worktree baseline
    against the working tree produced 18 apparent regressions; 17 of them were
    this file, because the worktree never had the live flags. `structured_tool_
    output: true` alone rewrites tool returns into a `ToolOutput` wrapper and
    fails four `test_tool_registry` cases that pass with flags neutral. Any
    attribution done without this pin is measuring operator toggles, not code.

    Pinned to an EMPTY flag set, so every test sees each feature at its declared
    default rather than at whatever production happens to be running. A test
    that specifically exercises flag loading should monkeypatch
    `ORCHESTRATOR_RUNTIME_FLAGS_PATH` to its own fixture; that still works,
    because this only sets the ambient default.
    """
    from src.features import RUNTIME_FLAGS_ENV

    previous = os.environ.get(RUNTIME_FLAGS_ENV)
    neutral = tmp_path_factory.mktemp("runtime_flags") / "runtime_flags.json"
    neutral.write_text(json.dumps({"flags": {}, "version": 1}), encoding="utf-8")
    os.environ[RUNTIME_FLAGS_ENV] = str(neutral)
    try:
        yield neutral
    finally:
        if previous is None:
            os.environ.pop(RUNTIME_FLAGS_ENV, None)
        else:
            os.environ[RUNTIME_FLAGS_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_kb_rag_query_length_log():
    """Keep the suite out of the live KB-RAG query-length telemetry (H2).

    `kb_rag.query()` appends to `data/kb_rag/telemetry/query_lengths.jsonl` by
    default; a test run from the shared clone would otherwise mix synthetic
    queries into the production observation. Tests that exercise the log
    monkeypatch the env var to their own tmp path.
    """
    from src.retrieval.kb_rag_query_telemetry import LOG_ENV

    previous = os.environ.get(LOG_ENV)
    os.environ[LOG_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LOG_ENV, None)
        else:
            os.environ[LOG_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_serving_calls_log():
    """Keep the suite out of the live per-call serving log (``logs/serving_calls``).

    Every ``LlamaServerBackend`` call writes one record; a test run from the shared
    clone would otherwise mix mocked calls into the production observation. Tests
    that exercise the log monkeypatch the env var to their own tmp path.
    """
    from src.backends.serving_calls import LOG_ENV

    previous = os.environ.get(LOG_ENV)
    os.environ[LOG_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LOG_ENV, None)
        else:
            os.environ[LOG_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_live_gpu_window():
    """Keep the suite off the live MI210 window file (parked roles).

    A real ``holder=autokernel`` window on this host would otherwise refuse every
    test call to a parked role. Tests that exercise parking point the env var at
    their own tmp file.
    """
    from src.runtime.gpu_window import PATH_ENV

    previous = os.environ.get(PATH_ENV)
    os.environ[PATH_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PATH_ENV, None)
        else:
            os.environ[PATH_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_live_context_limit_reads():
    """Keep the suite off the live stack's ``GET /props`` (context limits).

    ``ContextLimitResolver`` reads each role's per-request n_ctx from the live
    llama-server; unit tests must resolve from the registry (or from an injected
    resolver) instead of whatever happens to be serving on this host.
    """
    from src.backends.context_limits import LIVE_ENV

    previous = os.environ.get(LIVE_ENV)
    os.environ[LIVE_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LIVE_ENV, None)
        else:
            os.environ[LIVE_ENV] = previous


@pytest.fixture(autouse=True)
def _reset_context_limit_resolver():
    """Drop the process-wide context-limit cache between tests."""
    from src.backends.context_limits import set_context_limit_resolver

    set_context_limit_resolver(None)
    yield
    set_context_limit_resolver(None)


@pytest.fixture(autouse=True)
def _hermetic_long_prefill_lease(tmp_path_factory):
    """Keep tests off the HOST-WIDE long-prefill lease (KVU-15a).

    The lease is an flock on ``{tmp_dir}/kv_prefill_lease.{host}_{port}.lock`` —
    by default ``/mnt/raid0/llm/tmp``, where the LIVE API workers take it. A test
    that admits a long request on ``localhost:8083`` would otherwise hold the
    production lease (and stall real long prefills) for as long as the test
    holds it. Each test gets its own directory, so a lease a test never releases
    cannot leak into the next one either. Only this module's resolver is
    patched, never ``ORCHESTRATOR_TMP_DIR`` (see tests/unit/conftest.py for why
    a global tmp-dir override is the wrong seam).

    Uses a PRIVATE ``MonkeyPatch``, not the ``monkeypatch`` fixture: requesting
    that fixture from an autouse fixture declared before
    ``_reset_config_between_tests`` sets it up earlier and so tears it down
    LATER, i.e. a test's ``monkeypatch.setattr("src.config.get_config", ...)``
    would still be in place when ``reset_config()`` calls
    ``get_config.cache_clear()`` (AttributeError at teardown).
    """
    from src.runtime import long_prefill_lease

    lease_dir = tmp_path_factory.mktemp("hermetic_prefill_lease")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(long_prefill_lease, "lease_dir", lambda: lease_dir)
        yield lease_dir


@pytest.fixture(autouse=True)
def _reset_prefix_history():
    """Fresh KVU-15c prefix history per test: a prefix one test served must not
    credit another's request (its host-wide file already lives in the per-test
    lease dir above)."""
    from src.scheduling.prefix_history import set_prefix_history

    set_prefix_history(None)
    yield
    set_prefix_history(None)


@pytest.fixture(autouse=True)
def _reset_config_between_tests():
    """Ensure config cache is clean between tests.

    Prevents env var leaks from one test affecting another.
    Uses reset_config() which clears the lru_cache on get_config().
    """
    reset_config()
    yield
    reset_config()


def pytest_collection_modifyitems(config, items):
    """Add markers and warnings for parallel execution."""
    # Check if pytest-xdist is being used with too many workers
    try:
        num_workers = config.option.numprocesses
        if num_workers is not None:
            if num_workers == "auto":
                warnings.warn(
                    f"DANGER: pytest -n auto on 192-thread machine will spawn ~192 workers!\n"
                    f"Each worker may load models, causing memory exhaustion.\n"
                    f"Use: pytest -n {MAX_SAFE_WORKERS} instead.",
                    UserWarning,
                )
            elif isinstance(num_workers, int) and num_workers > MAX_SAFE_WORKERS:
                warnings.warn(
                    f"High parallelism: {num_workers} workers requested.\n"
                    f"Recommended max: {MAX_SAFE_WORKERS}. May cause memory issues.",
                    UserWarning,
                )
    except (AttributeError, TypeError):
        # pytest-xdist not installed or not using -n flag
        pass

    if not config.getoption("--run-live-models"):
        skip_live_models = pytest.mark.skip(reason="Need --run-live-models to run live model tests")
        for item in items:
            if "requires_live_models" in item.keywords:
                item.add_marker(skip_live_models)


@pytest.fixture
def mock_registry():
    """Shared registry mock fixture for chat pipeline tests."""
    registry = MagicMock()
    registry.routing_hints = {}
    return registry


@pytest.fixture
def mock_app_state(mock_registry):
    """Shared AppState mock fixture with common attributes used by route tests."""
    state = MagicMock(spec=AppState)
    state.progress_logger = MagicMock()
    state.hybrid_router = None
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.registry = mock_registry
    state.health_tracker = MagicMock()
    state.admission = MagicMock()
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def mock_llm_primitives():
    """Shared LLM primitives mock fixture with common telemetry fields."""
    primitives = MagicMock()
    primitives._backends = True
    primitives.total_tokens_generated = 100
    primitives.total_prompt_eval_ms = 50
    primitives.total_generation_ms = 200
    primitives._last_predicted_tps = 25.0
    primitives.total_http_overhead_ms = 10
    primitives.get_cache_stats.return_value = {"hits": 5, "misses": 2}
    primitives.llm_call.return_value = "Test response"
    return primitives
