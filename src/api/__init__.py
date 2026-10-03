"""FastAPI application factory for the orchestrator API.

This module provides the main FastAPI application with all routes,
middleware, and lifespan management.

Usage:
    # Development server
    uvicorn src.api:app --reload --port 8000

    # Production
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.features import features
from src.api.state import get_state, AppState as AppState
from src.api.routes import create_api_router
from src.api.services.memrl import (
    load_optional_imports,
    background_cleanup,
    ensure_memrl_initialized,
    shutdown_scoring,
    get_progress_logger_class,
    get_tool_registry_class,
    get_script_registry_class,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan.

    Initialization order:
    1. Load feature flags from environment
    2. Load optional module imports based on features
    3. Load registry (YAML parsing only)
    4. Initialize core components (LLM primitives, gate runner)
    5. Initialize optional components based on features
    6. Eagerly initialize MemRL (Q-scorer, TaskEmbedder, HybridRouter)
    7. Start background tasks if MemRL enabled
    """
    state = get_state()
    f = features()

    from src.runtime.config_attestation import publish_config_attestation

    try:
        publish_config_attestation(f)
    except Exception as exc:
        logger.error("Could not publish API worker config attestation: %s", exc)

    # Validate feature dependencies at startup
    validation_errors = f.validate()
    if validation_errors:
        for err in validation_errors:
            logger.error(f"Feature validation error: {err}")

    # Load feature flags and optional imports
    load_optional_imports()

    logger.info(f"Starting orchestrator with features: {f.enabled_features()}")

    # Load registry for role-based generation defaults (YAML parsing only, no models)
    try:
        from src.registry_loader import RegistryLoader

        state.registry = RegistryLoader(validate_paths=False)
    except Exception as e:
        logger.info(f"Registry file not found or invalid, using defaults: {e}")
        state.registry = None

    # Core components (always initialized)
    from src.llm_primitives import LLMPrimitives
    from src.gate_runner import GateRunner

    ProgressLogger = get_progress_logger_class()

    state.llm_primitives = LLMPrimitives(
        mock_mode=f.mock_mode,
        registry=state.registry,
        health_tracker=state.health_tracker,
        admission_controller=state.admission,
    )
    state.progress_logger = ProgressLogger() if ProgressLogger else None
    state.gate_runner = GateRunner(progress_logger=state.progress_logger)

    # Pre-create cpu_region_lock files for any role whose backend was wired
    # as ConcurrencyAwareBackend, so the dashboard's per-region-locks panel
    # shows the role as `wired` before the first dispatch arrives. Without
    # this, freshly-restarted roles with zero traffic look "not wired" even
    # though the lock-acquisition code path is fully in place.
    try:
        from src.backends.concurrency_aware import ConcurrencyAwareBackend
        from src.runtime.cpu_region_lock import region_lock_path, sweep_stale_region_lock_payloads
        from src.runtime.instance_topology import get_instance_regions, ATOMIC_REGIONS

        _topology = get_instance_regions()
        _role_regions: dict[str, set[str]] = {}
        for (_role, _idx), _regs in _topology.items():
            _role_regions.setdefault(_role, set()).update(_regs)

        _warmed = 0
        _stale_payloads_cleared = sweep_stale_region_lock_payloads()
        for _role, _backend in (state.llm_primitives._backends or {}).items():
            if not isinstance(_backend, ConcurrencyAwareBackend):
                continue
            for _region in _role_regions.get(_role, set(ATOMIC_REGIONS)):
                _path = region_lock_path(_role, _region)
                _path.parent.mkdir(parents=True, exist_ok=True)
                _path.touch(exist_ok=True)
                _warmed += 1
        if _warmed:
            logger.info("cpu_region_lock warm-up: pre-created %d lock files", _warmed)
        if _stale_payloads_cleared:
            logger.info(
                "cpu_region_lock startup sweep: cleared %d stale diagnostic payload(s)",
                _stale_payloads_cleared,
            )
    except Exception as _exc:
        logger.debug("cpu_region_lock warm-up skipped: %s", _exc)

    # Tool registry (feature-gated)
    ToolRegistry = get_tool_registry_class()
    if f.tools and ToolRegistry:
        try:
            from src.tool_registry import load_from_yaml as load_tools_from_yaml

            state.tool_registry = ToolRegistry()

            # Load tools from YAML registry (22 tools in orchestration/tool_registry.yaml)
            from src.config import get_config as _get_config

            _paths = _get_config().paths
            loaded = load_tools_from_yaml(
                state.tool_registry,
                str(_paths.tool_registry_path),
            )
            logger.info(f"Loaded {loaded} tools from YAML registry")

            # Load role permissions from model registry
            state.tool_registry.load_permissions_from_registry(
                str(_paths.registry_path),
            )

            # Register built-in tools (programmatic tools)
            # These may overlap with YAML tools - duplicates are skipped
            try:
                from src.builtin_tools import register_builtin_tools

                register_builtin_tools(state.tool_registry)
            except ImportError:
                logger.debug("No builtin_tools module found")
            except ValueError as e:
                # Tool already registered from YAML - this is fine
                logger.debug(f"Some builtin tools skipped (already in YAML): {e}")
        except Exception as e:
            logger.info(f"Tool registry not available: {e}")
            state.tool_registry = None
    else:
        state.tool_registry = None

    # Script registry (feature-gated, requires tools)
    ScriptRegistry = get_script_registry_class()
    if f.scripts and ScriptRegistry and state.tool_registry:
        try:
            state.script_registry = ScriptRegistry()
            state.script_registry.load_from_directory(
                str(_paths.script_registry_dir),
            )
        except Exception as e:
            logger.info(f"Script registry not available: {e}")
            state.script_registry = None
    else:
        state.script_registry = None

    # Preflight check for knowledge tools (warn only, don't block startup)
    if f.tools:
        from src.api.routes.health import _check_knowledge_tools

        knowledge_status = _check_knowledge_tools()
        if not knowledge_status["available"]:
            unavailable = [
                name
                for name, info in knowledge_status["tools"].items()
                if not info["available"]
            ]
            logger.warning(
                f"Knowledge tools preflight: {len(unavailable)} tools unavailable "
                f"({', '.join(unavailable)}). Install with: pip install -e '.[knowledge]'"
            )
        else:
            logger.info(
                f"Knowledge tools preflight: all {len(knowledge_status['tools'])} tools available"
            )

    # Eagerly initialize MemRL components at startup (loads TaskEmbedder model).
    # This avoids a ~10-30s block on the first real_mode request.
    if f.memrl:
        ensure_memrl_initialized(state)

        # Seed classification exemplars from YAML if episodic store is empty.
        # This ensures ClassificationRetriever has exemplars for Q-value voting
        # without requiring a manual seed_memory() call.
        try:
            from orchestration.repl_memory.seed_loader import seed_memory
            seed_stats = seed_memory(force=False)
            if seed_stats.get("seeded", 0) > 0:
                logger.info("Auto-seeded %d classification exemplars", seed_stats["seeded"])
        except Exception as e:
            logger.debug("Exemplar auto-seeding skipped: %s", e)

    # Background Q-scoring task (only if MemRL feature enabled).
    # Only run in ONE worker to avoid 6× parallel log scans that burn
    # 100% CPU on all workers.  Use a file lock to elect one worker.
    import os as _os
    try:
        from src.config import get_config as _gc
        _bg_lock_path = str(_gc().paths.log_dir / ".bg_cleanup.lock")
    except Exception:
        _bg_lock_path = "logs/.bg_cleanup.lock"
    _is_primary_worker = False
    if f.memrl:
        try:
            import fcntl as _fcntl
            _bg_lock_fh = open(_bg_lock_path, "w")
            _fcntl.flock(_bg_lock_fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            _bg_lock_fh.write(str(_os.getpid()))
            _bg_lock_fh.flush()
            _is_primary_worker = True
            logger.info("Background cleanup: this worker (pid=%d) elected as primary", _os.getpid())
        except (OSError, BlockingIOError):
            logger.info("Background cleanup: skipping (another worker is primary)")
    if f.memrl and _is_primary_worker:
        state._q_scorer_task = asyncio.create_task(background_cleanup(state))
    else:
        state._q_scorer_task = None

    yield

    from src.runtime.config_attestation import remove_config_attestation

    try:
        remove_config_attestation()
    except Exception as exc:
        logger.warning("Could not remove API worker config attestation: %s", exc)

    # Shutdown - cancel background task first
    if state._q_scorer_task:
        state._q_scorer_task.cancel()
        try:
            await state._q_scorer_task
        except asyncio.CancelledError:
            pass

    # Shutdown Q-scorer thread pool
    shutdown_scoring()

    # Flush progress logger before cleanup
    if state.progress_logger:
        state.progress_logger.flush()

    # Shutdown vision batch processor if initialized
    if state.vision_batch_processor is not None:
        state.vision_batch_processor.shutdown()

    # Clear state
    state.llm_primitives = None
    state.gate_runner = None
    state.progress_logger = None
    state.q_scorer = None
    state.episodic_store = None
    state.hybrid_router = None
    state.tool_registry = None
    state.script_registry = None
    state.registry = None
    state.vision_batch_processor = None
    state.health_tracker.reset()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    # TestClient startup can deadlock in some test environments when lifespan
    # initialization performs recursive config bootstrap. For pytest suites we
    # disable lifespan and rely on route-level lazy initialization.
    lifespan_ctx = None if os.getenv("PYTEST_CURRENT_TEST") else lifespan

    app = FastAPI(
        title="Orchestrator API",
        description="HTTP interface for the hierarchical orchestration system",
        version="0.1.0",
        lifespan=lifespan_ctx,
    )

    # Middleware (order matters: last added = first executed)
    from src.config import get_config as _get_config

    _api_cfg = _get_config().api

    # CORS — explicit origins required when credentials are enabled
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_api_cfg.cors_origins,
        allow_credentials=_api_cfg.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting — per-IP token bucket
    from src.api.rate_limit import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        rpm=_api_cfg.rate_limit_rpm,
        burst=_api_cfg.rate_limit_burst,
    )

    # RTG-47 Phase 1a — scoped CORS for the dashboard data plane so epyc-root hub
    # pages (:8100) can fetch /dashboard/api/* + /dashboard/events/* directly from
    # the browser. Added LAST, i.e. outermost: the global CORSMiddleware above
    # answers a preflight from a non-allowlisted origin with 400, so an inner
    # placement would never see the hub's OPTIONS. Every non-dashboard path is
    # passed through untouched — the /v1 and session surfaces stay same-origin.
    from src.api.dashboard_cors import DashboardCORSMiddleware

    app.add_middleware(DashboardCORSMiddleware)

    # Include all routes
    router = create_api_router()
    app.include_router(router)

    # 2026-05-24 Phase B (cross-role-bw-aware-routing): map ContentionDenied
    # to 503 + Retry-After so foreground callers see clean backpressure
    # instead of a generic 500. Background callers (autopilot seeder) catch
    # the 503 and skip/defer the probe.
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from src.scheduling.contention_gate import ContentionDenied

    def _log_contention_denied(request: Request, exc: ContentionDenied) -> None:
        state = get_state()
        progress_logger = getattr(state, "progress_logger", None)
        if not progress_logger:
            return
        try:
            from orchestration.repl_memory.progress_logger import EventType, ProgressEntry

            task_id = (
                request.headers.get("x-task-id")
                or request.query_params.get("task_id")
                or f"contention-denied:{request.url.path}"
            )
            progress_logger.log(
                ProgressEntry(
                    event_type=EventType.ROUTING_FALLBACK,
                    task_id=task_id,
                    data={
                        "kind": "contention_denied",
                        "path": request.url.path,
                        "method": request.method,
                        "detail": str(exc)[:500],
                        "retry_after_s": 5,
                    },
                    outcome="failure",
                    outcome_details=str(exc)[:500],
                )
            )
            flush = getattr(progress_logger, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            logger.debug("ContentionDenied progress telemetry failed", exc_info=True)

    @app.exception_handler(ContentionDenied)
    async def _contention_denied_handler(request: Request, exc: ContentionDenied):
        # Retry-After: short for foreground (next active decode should clear
        # within ~5 s), longer when explicitly background. We can't fully
        # introspect the request priority from here without re-parsing, so
        # use a conservative middle-ground default.
        _log_contention_denied(request, exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "contention_denied",
                "detail": str(exc),
                "retry_after_s": 5,
                "error_code": 503,
                "error_detail": str(exc),
                "failure_provenance": exc.provenance(),
            },
            headers={"Retry-After": "5"},
        )

    from src.exceptions import RoleParkedError

    @app.exception_handler(RoleParkedError)
    async def _role_parked_handler(request: Request, exc: RoleParkedError):
        # The role's GPU server is lent to AutoKernel (src/runtime/gpu_window.py):
        # an explicit, immediate 503 instead of connection-refused or a timeout.
        logger.warning("Role parked for %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content=exc.to_dict(),
            headers={"Retry-After": str(exc.retry_after_s)},
        )

    from src.exceptions import ContextOverflowError

    @app.exception_handler(ContextOverflowError)
    async def _context_overflow_handler(request: Request, exc: ContextOverflowError):
        # A request that cannot fit the role's per-request KV context is the
        # CALLER's to shrink (413); a shared pool that stayed exhausted through
        # the bounded retries is back-pressure (503 + Retry-After). Either way
        # an explicit typed error — never a truncated or empty answer.
        status = 503 if exc.retryable else 413
        headers = {"Retry-After": "10"} if exc.retryable else None
        logger.warning(
            "Context overflow for %s %s: kind=%s role=%s n_prompt=%s n_ctx=%s",
            request.method, request.url.path, exc.kind, exc.role,
            exc.n_prompt_tokens, exc.n_ctx,
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": "context_overflow",
                "detail": str(exc),
                "error_code": status,
                "error_detail": str(exc),
                "context_overflow": exc.to_dict(),
            },
            headers=headers,
        )

    @app.exception_handler(ResponseValidationError)
    async def _response_validation_error_handler(request: Request, exc: ResponseValidationError):
        logger.exception(
            "Response validation failed for %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "response_validation_error", "detail": "Internal server error"},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "Unhandled API exception for %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": "Internal server error"},
        )

    return app


# Create the default application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
