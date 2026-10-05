"""Main LLMPrimitives class that combines all mixins."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import threading
import time
from dataclasses import replace
from typing import Any

from .backend import BackendMixin
from .config import LLMPrimitivesConfig
from .cost_tracking import CostTrackingMixin
from .inference import InferenceMixin
from .mock import MockMixin
from .persona import PersonaMixin
from .stats import StatsMixin
from .tokens import TokensMixin
from .teleport import TeleportDecision, TeleportInputs, TeleportPolicy, decide_teleport
from .types import CallLogEntry, LLMResult
from src.exceptions import AdmissionDenied, AdmissionDeniedText
from src.roles import Role
from src.workload_model import infer_workload_class


class LLMPrimitives(
    BackendMixin,
    CostTrackingMixin,
    TokensMixin,
    PersonaMixin,
    MockMixin,
    InferenceMixin,
    StatsMixin,
):
    """LLM primitives for sub-LM spawning.

    Provides llm_call() for single calls and llm_batch() for parallel calls.
    Can operate in mock mode for testing, with CachingBackend (RadixAttention),
    or with a legacy ModelServer.
    """

    # Task type to worker role mapping (for pool routing)
    WORKER_TASK_ROUTING = {
        "explore": str(Role.WORKER_GENERAL),
        "summarize": str(Role.WORKER_GENERAL),
        "understand": str(Role.WORKER_GENERAL),
        # coding bursts map to worker_coder semantic role (fast worker backend)
        "code": "worker_coder",
        "coder": "worker_coder",
        "code_impl": "worker_coder",
        "refactor": "worker_coder",
        "test_gen": "worker_coder",
        "fast": "worker_coder",
        "boilerplate": "worker_coder",
        "transform": "worker_coder",
    }

    _DEFAULT_DEPTH_ROLE_OVERRIDES = {1: str(Role.WORKER_GENERAL)}
    _DEFAULT_DEPTH_OVERRIDE_MAX_DEPTH = 3

    def __init__(
        self,
        model_server: Any | None = None,
        mock_mode: bool = True,
        config: LLMPrimitivesConfig | None = None,
        mock_responses: dict[str, str] | None = None,
        server_urls: dict[str, str] | None = None,
        num_slots: int = 2,
        registry: Any | None = None,
        worker_pool: Any | None = None,
        use_worker_pool: bool = False,
        health_tracker: Any | None = None,
        admission_controller: Any | None = None,
        gpu_lease_manager: Any | None = None,
        teleport_policy: TeleportPolicy | None = None,
        server_urls_source: str = "config",
    ):
        """Initialize LLM primitives.

        Args:
            model_server: Optional ModelServer instance for legacy inference.
            mock_mode: If True, return mock responses instead of real inference.
            config: Optional configuration for output caps, parallelism, etc.
            mock_responses: Optional dict mapping prompts to mock responses.
            server_urls: Dict mapping role names to llama-server URLs.
                        If provided and not mock_mode, uses CachingBackend.
            num_slots: Number of slots per server for prefix caching.
            registry: Optional RegistryLoader for role-based generation defaults.
            worker_pool: Optional WorkerPoolManager for worker role routing.
            use_worker_pool: If True and worker_pool provided, route worker calls through pool.
            health_tracker: Optional BackendHealthTracker for circuit breaker integration.
            admission_controller: Optional AdmissionController for per-backend queue limiting.
            gpu_lease_manager: Optional process-local lease manager for AXA-2 decisions.
            teleport_policy: Optional default-off AXA-2 teleport policy.
        """
        self.model_server = model_server
        self.mock_mode = mock_mode
        self.health_tracker = health_tracker
        self.admission_controller = admission_controller
        self.gpu_lease_manager = gpu_lease_manager
        self.teleport_policy = teleport_policy if teleport_policy is not None else TeleportPolicy()
        self.config = config if config is not None else LLMPrimitivesConfig()
        self.mock_responses = mock_responses if mock_responses is not None else {}
        self.server_urls = server_urls
        # WP-12: "config" (default) marks server_urls as config-derived — the
        # fleet layer (when flagged on) may serve those roles from fleet truth.
        # "request" marks caller-supplied overrides as authoritative; the fleet
        # layer then leaves every role on the legacy per-role build.
        self.server_urls_source = server_urls_source
        self.num_slots = num_slots
        self.registry = registry
        self.worker_pool = worker_pool
        self.use_worker_pool = use_worker_pool and worker_pool is not None

        # CachingBackend instances per role (RadixAttention)
        self._backends: dict[str, Any] = {}

        # Initialize backends if server URLs provided and not mock mode
        if not mock_mode and server_urls:
            self._init_caching_backends(server_urls, num_slots)

        # Accurate token counting via llama-server /tokenize (C2)
        self._tokenizer = None
        try:
            from src.features import features as _get_features

            if _get_features().accurate_token_counting and not mock_mode and server_urls:
                from .tokenizer import LlamaTokenizer

                # Use first available server URL for tokenization
                first_url = next(iter(server_urls.values()), None)
                if first_url:
                    self._tokenizer = LlamaTokenizer(base_url=first_url)
        except Exception:
            self._tokenizer = None

        # Call log for debugging and testing
        self.call_log: list[CallLogEntry] = []

        # Stats
        self.total_calls = 0
        self.total_batch_calls = 0
        self.total_tokens_generated = 0
        # Sum of server-REPORTED prompt tokens (never estimated; see inference.py).
        self.total_prompt_tokens_reported = 0
        # Clean timing accumulators from llama.cpp timings
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self._last_predicted_tps = 0.0  # Most recent call's clean t/s

        # Recursion depth tracking
        self._recursion_depth = 0
        self._max_recursion_depth_reached = 0

        # HTTP overhead tracking (server-side overhead not captured in inference timings)
        self.total_http_overhead_ms = 0.0

        # Per-request cache_prompt override (None = backend default)
        self.cache_prompt: bool | None = None
        # Per-request cancellation/deadline hooks (set by API layer).
        self._request_cancel_check = None
        self._request_deadline_s = None
        self._request_task_id = None
        self._request_priority = "interactive"
        self._request_workload_class = "interactive"
        # Request-local context to avoid cross-request overwrite on shared primitives.
        self._request_cancel_check_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar(
            "llm_primitives_request_cancel_check",
            default=None,
        )
        self._request_deadline_s_ctx: contextvars.ContextVar[float | None] = contextvars.ContextVar(
            "llm_primitives_request_deadline_s",
            default=None,
        )
        self._request_task_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "llm_primitives_request_task_id",
            default=None,
        )
        self._request_session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "llm_primitives_request_session_id",
            default=None,
        )
        self._request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "llm_primitives_request_id",
            default=None,
        )
        self._request_trial_id_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar(
            "llm_primitives_request_trial_id",
            default=None,
        )
        self._request_batch_id_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar(
            "llm_primitives_request_batch_id",
            default=None,
        )
        self._max_queue_wait_ms_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            "llm_primitives_max_queue_wait_ms",
            default=None,
        )
        self._request_priority_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "llm_primitives_request_priority",
            default=None,
        )
        self._request_workload_class_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "llm_primitives_request_workload_class",
            default=None,
        )
        self._batch_placement_mode_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "llm_primitives_batch_placement_mode",
            default=None,
        )
        # HS-4 P0.1: structured chat payload bound for one chat_completion_call.
        self._chat_payload_ctx: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("llm_primitives_chat_payload", default=None)
        )
        # TD-21.21 concurrency fix (coordinator review of 56e33d49): `_last_inference_meta`
        # below is a PLAIN instance attribute, reassigned by every `llm_call`/`chat_completion_call`
        # against this (often SHARED -- see `_init_primitives` reuse of `state._real_primitives`)
        # instance, so a concurrent request's call can overwrite it between one call finishing and
        # its own caller reading it back -- there is no way to tell whose data is in it. This
        # ContextVar is the per-call-safe counterpart, set at the SAME points `_last_inference_meta`
        # is (see `src/llm_primitives/inference.py`): `asyncio.to_thread`/`asyncio.Task` each copy
        # the calling context once at creation (`contextvars.copy_context()`), so two concurrent
        # calls against the SAME primitives instance mutate two independent copies and can never
        # observe each other's value here, unlike the plain attribute. Read via
        # `get_last_inference_meta()`. Not a replacement for `_last_inference_meta` -- existing
        # consumers of the plain attribute are unchanged; this is additive.
        self._last_inference_meta_ctx: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("llm_primitives_last_inference_meta", default=None)
        )
        # HS-4 P0.2: typed /v1 request keys stamped onto the inference trace.
        # Instance-level (not a contextvar) because /v1 builds one primitives
        # object per request and its streaming body runs in a different context
        # from the endpoint function.
        self._request_trace_keys: dict[str, Any] = {}
        self._budget_diagnostics: dict[str, Any] = {
            "deadline_present": False,
            "budget_applied": False,
            "deadline_remaining_ms_start": None,
            "deadline_remaining_ms_end": None,
            "timeout_clamp_events": 0,
            "depth_override_enabled": False,
            "depth_override_events": 0,
            "depth_override_roles": [],
            "depth_override_skip_events": 0,
            "depth_override_skip_reasons": [],
        }
        self._depth_model_overrides_enabled = False
        self._depth_role_overrides: dict[int, str] = {}
        self._depth_override_max_depth = self._DEFAULT_DEPTH_OVERRIDE_MAX_DEPTH
        try:
            from src.features import features as _get_features

            self._depth_model_overrides_enabled = bool(_get_features().depth_model_overrides)
        except Exception:
            self._depth_model_overrides_enabled = False
        if self._depth_model_overrides_enabled:
            self._depth_role_overrides = self._load_depth_role_overrides()
            self._depth_override_max_depth = self._load_depth_override_max_depth()

        # Per-query cost tracking
        self._current_query = None
        self._completed_queries: list = []

        # Concurrency policy (small workers only)
        from src.concurrency import get_role_max_concurrency

        self._role_limits = {
            role: get_role_max_concurrency(role) for role in (server_urls or {}).keys()
        }
        self._role_semaphores: dict[str, threading.Semaphore] = {}

    def evaluate_teleport_decision(
        self,
        inputs: TeleportInputs,
        *,
        lease_owner: str | None = None,
    ) -> TeleportDecision:
        """Evaluate AXA-2 teleport eligibility without performing cutover.

        The disabled default returns before consulting the optional lease
        manager. An enabled policy treats a lease held by another owner as an
        unavailable GPU, but this helper never acquires or releases a lease.
        """
        if not self.teleport_policy.enabled:
            return decide_teleport(self.teleport_policy, inputs)

        effective_inputs = inputs
        if self.gpu_lease_manager is not None:
            lease_status = self.gpu_lease_manager.status()
            if lease_status.acquired and lease_status.owner != lease_owner:
                effective_inputs = replace(
                    inputs,
                    gpu_available=False,
                    metadata={
                        **inputs.metadata,
                        "gpu_lease_owner": lease_status.owner,
                    },
                )
        return decide_teleport(self.teleport_policy, effective_inputs)

    def _get_role_limit(self, role: str) -> int:
        """Return per-role concurrency limit (defaults to 1)."""
        from src.concurrency import get_role_max_concurrency

        return self._role_limits.get(role) or get_role_max_concurrency(role)

    def get_request_cancel_check(self):
        """Get request-local cancellation callback if present."""
        value = self._request_cancel_check_ctx.get()
        if value is not None:
            return value
        return self._request_cancel_check

    def get_request_deadline_s(self) -> float | None:
        """Get request-local deadline (perf_counter seconds) if present."""
        value = self._request_deadline_s_ctx.get()
        if value is not None:
            return value
        return self._request_deadline_s

    def get_request_task_id(self) -> str | None:
        """Get request-local task id for telemetry attribution."""
        value = self._request_task_id_ctx.get()
        if value is not None:
            return value
        return self._request_task_id

    def get_request_id(self) -> str | None:
        """Get caller-supplied request id for telemetry attribution."""
        value = self._request_id_ctx.get()
        return str(value) if value not in (None, "") else None

    def get_request_trial_id(self):
        """Get caller-supplied autopilot/benchmark trial id, if present."""
        return self._request_trial_id_ctx.get()

    def get_request_batch_id(self):
        """Get caller-supplied batch/concurrency id, if present."""
        return self._request_batch_id_ctx.get()

    def get_request_session_id(self) -> str | None:
        """Get caller-supplied conversation/session id, if present."""
        value = self._request_session_id_ctx.get()
        return str(value) if value not in (None, "") else None

    def get_request_priority(self) -> str:
        """Get request-local priority used by admission control."""
        value = self._request_priority_ctx.get()
        if value:
            return str(value)
        return str(self._request_priority or "interactive")

    def get_request_workload_class(self) -> str:
        """Get request-local workload class used for attribution."""
        value = self._request_workload_class_ctx.get()
        if value:
            return str(value)
        return str(self._request_workload_class or "interactive")

    def get_batch_placement_mode(self) -> str:
        """Get validated request-local burst placement intent."""
        from src.scheduling.placement_policy import coerce_batch_placement_mode

        return coerce_batch_placement_mode(self._batch_placement_mode_ctx.get()).value

    def get_last_inference_meta(self) -> dict[str, Any] | None:
        """Per-call-safe counterpart of the plain `_last_inference_meta` attribute (TD-21.21).

        Returns the metadata dict (role, completion_reason, tokens, timings, ...) of the most
        recent `llm_call`/`chat_completion_call` made from THIS context (this asyncio Task or
        `asyncio.to_thread` offload and anything called synchronously from it) -- never a
        concurrent call's data, even when this primitives instance is shared across requests.
        `None` when no call has been made yet in this context, which callers that need the
        finish reason should treat as unknown (fail-closed), never as a stale value.
        """
        return self._last_inference_meta_ctx.get()

    def get_max_queue_wait_ms(self) -> int | None:
        """Get request-local cross-role contention gate budget, if set.

        Returns None when the request did not specify one — the gate will
        then pick a class-appropriate default (5 s foreground / 90 s background).
        """
        value = self._max_queue_wait_ms_ctx.get()
        if value is not None:
            return int(value)
        return None

    @contextlib.contextmanager
    def request_context(
        self,
        *,
        cancel_check=None,
        deadline_s: float | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        trial_id=None,
        batch_id=None,
        priority: str = "interactive",
        workload_class: str | None = None,
        batch_placement_mode: str | None = None,
        max_queue_wait_ms: int | None = None,
    ):
        """Bind cancellation/deadline metadata to the current request context.

        `max_queue_wait_ms` is the cross-role contention gate budget — how
        long the gate may queue a request before rejecting (per the
        cross-role-bw-aware-routing handoff Phase B). None lets the gate
        pick the class-appropriate default.
        """
        normalized_priority = (
            "background"
            if str(priority or "interactive").strip().lower() == "background"
            else "interactive"
        )
        normalized_workload_class = infer_workload_class(
            explicit=workload_class,
            priority=normalized_priority,
            batch_id=str(batch_id) if batch_id is not None else None,
        )
        start_remaining_ms = None
        if deadline_s is not None:
            start_remaining_ms = max(0.0, (deadline_s - time.perf_counter()) * 1000.0)
        self._budget_diagnostics = {
            "deadline_present": deadline_s is not None,
            "budget_applied": False,
            "deadline_remaining_ms_start": round(start_remaining_ms, 1)
            if start_remaining_ms is not None
            else None,
            "deadline_remaining_ms_end": None,
            "timeout_clamp_events": 0,
            "depth_override_enabled": self._depth_model_overrides_enabled,
            "depth_override_events": 0,
            "depth_override_roles": [],
            "depth_override_skip_events": 0,
            "depth_override_skip_reasons": [],
            "request_priority": normalized_priority,
            "session_id": session_id,
            "request_id": request_id,
            "trial_id": trial_id,
            "batch_id": batch_id,
            "workload_class": normalized_workload_class,
            "batch_placement_mode": self.get_batch_placement_mode()
            if batch_placement_mode is None
            else batch_placement_mode,
        }
        token_cancel = self._request_cancel_check_ctx.set(cancel_check)
        token_deadline = self._request_deadline_s_ctx.set(deadline_s)
        token_task = self._request_task_id_ctx.set(task_id)
        token_session = self._request_session_id_ctx.set(session_id)
        token_request = self._request_id_ctx.set(request_id)
        token_trial = self._request_trial_id_ctx.set(trial_id)
        token_batch = self._request_batch_id_ctx.set(batch_id)
        token_priority = self._request_priority_ctx.set(normalized_priority)
        token_workload_class = self._request_workload_class_ctx.set(normalized_workload_class)
        token_batch_placement = self._batch_placement_mode_ctx.set(batch_placement_mode)
        token_queue_wait = self._max_queue_wait_ms_ctx.set(max_queue_wait_ms)
        try:
            yield
        finally:
            end_remaining_ms = None
            if deadline_s is not None:
                end_remaining_ms = max(0.0, (deadline_s - time.perf_counter()) * 1000.0)
            self._budget_diagnostics["deadline_remaining_ms_end"] = (
                round(end_remaining_ms, 1) if end_remaining_ms is not None else None
            )
            self._request_cancel_check_ctx.reset(token_cancel)
            self._request_deadline_s_ctx.reset(token_deadline)
            self._request_task_id_ctx.reset(token_task)
            self._request_session_id_ctx.reset(token_session)
            self._request_id_ctx.reset(token_request)
            self._request_trial_id_ctx.reset(token_trial)
            self._request_batch_id_ctx.reset(token_batch)
            self._request_priority_ctx.reset(token_priority)
            self._request_workload_class_ctx.reset(token_workload_class)
            self._batch_placement_mode_ctx.reset(token_batch_placement)
            self._max_queue_wait_ms_ctx.reset(token_queue_wait)

    def _remaining_deadline_s(self) -> float | None:
        """Return remaining request deadline in seconds, if any."""
        deadline_s = self.get_request_deadline_s()
        if deadline_s is None:
            return None
        return max(0.0, deadline_s - time.perf_counter())

    def _bind_current_context(self, fn, *args, **kwargs):
        """Bind current contextvars to a callable for thread/executor execution."""
        ctx = contextvars.copy_context()

        def _runner():
            return ctx.run(fn, *args, **kwargs)

        return _runner

    def _load_depth_role_overrides(self) -> dict[int, str]:
        """Load depth->role overrides from config (env fallback) with sane defaults."""
        raw = ""
        try:
            from src.config import get_config

            raw = str(get_config().llm.depth_role_overrides or "").strip()
        except Exception:
            raw = ""
        if not raw:
            raw = os.environ.get("ORCHESTRATOR_LLM_DEPTH_ROLE_OVERRIDES", "").strip()
        if not raw:
            return dict(self._DEFAULT_DEPTH_ROLE_OVERRIDES)

        parsed: dict[int, str] = {}
        # JSON object format: {"1":"worker_general","2":"worker_math"}
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        depth = int(key)
                        role = str(value).strip()
                        if depth >= 1 and role:
                            parsed[depth] = role
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed = {}
        else:
            # CSV format: "1:worker_general,2:worker_math"
            for item in raw.split(","):
                item = item.strip()
                if not item or ":" not in item:
                    continue
                d_raw, r_raw = item.split(":", 1)
                try:
                    depth = int(d_raw.strip())
                except ValueError:
                    continue
                role = r_raw.strip()
                if depth >= 1 and role:
                    parsed[depth] = role

        if not parsed:
            return dict(self._DEFAULT_DEPTH_ROLE_OVERRIDES)
        return parsed

    def _load_depth_override_max_depth(self) -> int:
        """Load max depth eligible for override routing."""
        raw: Any = None
        try:
            from src.config import get_config

            raw = get_config().llm.depth_override_max_depth
        except Exception:
            raw = None
        if raw is None:
            raw = os.environ.get("ORCHESTRATOR_LLM_DEPTH_OVERRIDE_MAX_DEPTH", "").strip()
        try:
            max_depth = int(raw)
        except (TypeError, ValueError):
            max_depth = self._DEFAULT_DEPTH_OVERRIDE_MAX_DEPTH
        return max(1, max_depth)

    def _record_depth_override_skip(self, reason: str) -> None:
        self._budget_diagnostics["depth_override_skip_events"] += 1
        reasons = self._budget_diagnostics.get("depth_override_skip_reasons")
        if isinstance(reasons, list) and reason not in reasons:
            reasons.append(reason)

    def _resolve_depth_override_role(self, role: str) -> str:
        """Resolve role override for nested llm_call depth when feature is enabled."""
        if not self._depth_model_overrides_enabled:
            return role
        # llm_call increments _recursion_depth before dispatch.
        # depth=0 means the caller level (no override).
        depth = max(0, self._recursion_depth - 1)
        if depth <= 0:
            return role
        if depth > self._depth_override_max_depth:
            self._record_depth_override_skip("over_max_depth")
            return role

        # Exact depth override wins. Otherwise use depth-1 baseline if configured.
        override_role = self._depth_role_overrides.get(depth) or self._depth_role_overrides.get(1)
        if not override_role:
            self._record_depth_override_skip("no_override_for_depth")
            return role
        # Depth overrides should remain on the worker tier for predictable cost/latency.
        if not override_role.startswith("worker_"):
            self._record_depth_override_skip("non_worker_target")
            return role
        # Keep behavior safe: if override role backend is unavailable, preserve requested role.
        if self.server_urls:
            if override_role not in self.server_urls:
                self._record_depth_override_skip("target_backend_unavailable")
                return role
        elif self._backends:
            # Tests and lightweight setups may inject _backends directly without server_urls.
            # Avoid remapping to a role that has no backend binding.
            if override_role not in self._backends:
                self._record_depth_override_skip("target_backend_unavailable")
                return role
        self._budget_diagnostics["depth_override_events"] += 1
        roles = self._budget_diagnostics.get("depth_override_roles")
        if isinstance(roles, list):
            role_edge = f"{role}->{override_role}"
            if role_edge not in roles:
                roles.append(role_edge)
        return override_role

    def _clamp_timeout_to_request_budget(self, timeout_s: int | float) -> int:
        """Bound a per-call timeout to the request deadline budget.

        The effective llama timeout is::

            min(remaining_budget, max(role_base_timeout, remaining_budget))
              == remaining_budget

        i.e. a call may use the whole remaining request budget, and is never
        capped below it. This is byte-IDENTICAL to the historic
        ``min(role_base, remaining)`` clamp for every budget where
        ``remaining <= role_base`` — which is exactly all interactive traffic,
        whose request deadline STARTS at the role SLA (``resolve_timeout``) and
        only decreases, so ``remaining`` never exceeds the base. It differs ONLY
        when ``remaining > role_base``, which happens solely for an eval_batch
        request that extended its deadline past the SLA (2026-07-21 EV-11c):
        there the historic clamp shrank the call back to the 60s worker SLA,
        producing a doomed >60s MATH-tail call that 504'd and tripped the
        circuit breaker. Extending to ``remaining`` fixes that without touching
        any interactive path.
        """
        timeout_f = max(1.0, float(timeout_s))
        remaining_s = self._remaining_deadline_s()
        if remaining_s is None:
            return int(timeout_f)
        if remaining_s > timeout_f:
            # Deadline budget EXCEEDS the role base: use the full remaining
            # budget rather than capping DOWN to the (interactive) role SLA.
            # This is NOT a clamp (we are not shrinking), so no budget_applied /
            # timeout_clamp_events diagnostic is recorded — matching the historic
            # path, which also recorded nothing when min() returned the base.
            return int(remaining_s)
        clamped_f = max(1.0, min(timeout_f, remaining_s))
        if clamped_f < timeout_f:
            self._budget_diagnostics["budget_applied"] = True
            self._budget_diagnostics["timeout_clamp_events"] += 1
        return int(clamped_f)

    def get_budget_diagnostics(self) -> dict[str, Any]:
        """Return request-budget telemetry for current response diagnostics."""
        return dict(self._budget_diagnostics)

    @contextlib.contextmanager
    def _acquire_role(self, role: str):
        """Acquire per-role concurrency gate."""
        limit = self._get_role_limit(role)
        if limit <= 1:
            sem = self._role_semaphores.setdefault(role, threading.Semaphore(1))
        else:
            sem = self._role_semaphores.setdefault(role, threading.Semaphore(limit))
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def llm_call(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        n_tokens: int | None = None,
        skip_suffix: bool = False,
        stop_sequences: list[str] | None = None,
        persona: str | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        n_probs: int | None = None,
        post_sampling_probs: bool = False,
    ) -> str:
        """Call a sub-LM with optional context slice.

        Args:
            prompt: The instruction/prompt for the sub-LM.
            context_slice: Optional context to include (appended to prompt).
            role: Role determining which model to use (e.g., "worker", "coder").
            n_tokens: Max tokens to generate. If None, uses role default from registry.
            skip_suffix: If True, skip the registry system_prompt_suffix for this role.
                Used in direct-answer mode where any suffix (like "Elaborate on
                specialist outputs") degrades instruction precision quality.
            stop_sequences: Optional list of stop sequences to halt generation early.
            persona: Optional persona name (e.g., "security_auditor"). When set and
                the personas feature is enabled, injects the persona's system prompt
                before the user prompt.
            json_schema: Optional JSON schema to constrain output structure.
            grammar: Optional GBNF grammar for constrained generation.
            temperature: Optional explicit decode temperature override.
            seed: Optional explicit deterministic decode seed.
            top_p: Optional explicit nucleus sampling override.
            top_k: Optional explicit top-k sampling override.
            n_probs: Optional llama.cpp top-k token probability capture.
            post_sampling_probs: When True and n_probs is set, the captured
                top-k comes from AFTER the full sampler chain (grammar mask
                included) instead of the default pre-sampling raw logits —
                see typed_decisions/native.py for why grammar-constrained
                callers need this (TD-1d.2).

        Returns:
            Sub-LM response (capped at output_cap chars).

        Raises:
            RecursionError: If max recursion depth exceeded.
        """
        # Check recursion depth
        if self._recursion_depth >= self.config.max_recursion_depth:
            raise RecursionError(
                f"Maximum recursion depth ({self.config.max_recursion_depth}) exceeded. "
                f"Sub-LM calls cannot be nested more than {self.config.max_recursion_depth} levels deep."
            )

        self._recursion_depth += 1
        self._max_recursion_depth_reached = max(
            self._max_recursion_depth_reached, self._recursion_depth
        )

        try:
            sampling_kwargs: dict[str, float | int | bool] = {}
            if temperature is not None:
                sampling_kwargs["temperature"] = temperature
            if seed is not None:
                sampling_kwargs["seed"] = seed
            if top_p is not None:
                sampling_kwargs["top_p"] = top_p
            if top_k is not None:
                sampling_kwargs["top_k"] = top_k
            if n_probs is not None:
                sampling_kwargs["n_probs"] = n_probs
            if post_sampling_probs:
                sampling_kwargs["post_sampling_probs"] = True
            return self._llm_call_impl(
                prompt, context_slice, role, n_tokens, skip_suffix, stop_sequences,
                persona, json_schema, grammar, **sampling_kwargs,
            )
        finally:
            self._recursion_depth -= 1

    def get_request_chat_payload(self) -> dict[str, Any] | None:
        """HS-4 P0.1: structured payload bound by ``chat_completion_call``."""
        return self._chat_payload_ctx.get()

    def set_request_trace_keys(self, keys: dict[str, Any] | None) -> None:
        """HS-4 P0.2: bind typed request keys (x_session_id, ...) for tracing."""
        self._request_trace_keys = {k: v for k, v in (keys or {}).items() if v is not None}

    def get_request_trace_keys(self) -> dict[str, Any]:
        return dict(getattr(self, "_request_trace_keys", None) or {})

    def chat_completion_call(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        n_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """HS-4 P0.1: one structured chat-completions call (client tool mode).

        Unlike ``llm_call`` the caller's messages, tools and tool_choice reach
        llama-server's ``/v1/chat/completions`` verbatim, and the model's tool
        calls come back to the caller instead of being executed here.

        The call runs through the same ``_real_call`` seam as ``llm_call``:
        contention gate, per-role semaphore, admission control, region locks,
        circuit breaker, same-tier model fallback and the inference tap all
        apply. It deliberately does NOT apply the prompt-shaping steps of
        ``llm_call`` (role system-prompt suffix, persona, RAG injection) —
        the client owns the conversation — nor the content cache.

        Returns ``{"content", "tool_calls", "finish_reason", "usage"}``;
        ``usage`` holds the backend's own ``prompt_tokens`` /
        ``completion_tokens`` / ``cached_tokens`` (None when the server did
        not report one — never re-tokenized or estimated). Backend
        failures RAISE (no in-band ``[ERROR: ...]`` string). The
        ``output_cap`` truncation of ``_llm_call_impl`` is not applied here;
        ``n_tokens`` (the request's ``max_tokens``) bounds the output.
        """
        if self.mock_mode:
            raise RuntimeError("client tool mode requires a real backend (mock_mode is on)")
        if n_tokens is None and self.registry:
            n_tokens = self.registry.get_role_defaults(role)[0]
        if n_tokens is None:
            n_tokens = -1

        payload: dict[str, Any] = {"messages": [dict(m) for m in messages]}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        # The text form is what the prefix-cache slot router hashes and the tap
        # records as the prompt; the backend sends the structured payload.
        trace_prompt = json.dumps(payload, sort_keys=True)

        sampling_kwargs: dict[str, float | int] = {}
        if temperature is not None:
            sampling_kwargs["temperature"] = temperature
        if seed is not None:
            sampling_kwargs["seed"] = seed
        if top_p is not None:
            sampling_kwargs["top_p"] = top_p
        if top_k is not None:
            sampling_kwargs["top_k"] = top_k

        start_time = time.perf_counter()
        self.total_calls += 1
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="chat_completion",
            prompt=trace_prompt[:2000],
            role=role,
        )
        self._last_inference_meta = {}
        token = self._chat_payload_ctx.set(payload)
        try:
            role_for_call = self._resolve_depth_override_role(role)
            content = self._real_call(trace_prompt, role_for_call, n_tokens, None, **sampling_kwargs)
        except Exception as exc:
            log_entry.error = str(exc)
            raise
        finally:
            self._chat_payload_ctx.reset(token)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

        # TD-21.33: `self._real_call` above and this read are in the SAME synchronous method
        # call (no thread/task boundary crossed) -- the per-call-safe getter applies cleanly,
        # and `self` is always a real `LLMPrimitives`, so no duck-typed fallback is needed here.
        meta = self.get_last_inference_meta() or {}
        tool_calls = [tc for tc in (meta.get("tool_calls") or []) if isinstance(tc, dict)]
        log_entry.result = (content or json.dumps(tool_calls))[:500]

        def _count(value: Any) -> int | None:
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        return {
            "content": content or "",
            "tool_calls": tool_calls,
            "finish_reason": str(meta.get("completion_reason") or "stop"),
            "usage": {
                "prompt_tokens": _count(meta.get("prompt_tokens")),
                "completion_tokens": _count(meta.get("tokens")),
                "cached_tokens": _count(meta.get("cached_prompt_tokens")),
            },
        }

    def _llm_call_impl(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        n_tokens: int | None = None,
        skip_suffix: bool = False,
        stop_sequences: list[str] | None = None,
        persona: str | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        n_probs: int | None = None,
        post_sampling_probs: bool = False,
    ) -> str:
        """Internal implementation of llm_call (after recursion check)."""
        start_time = time.perf_counter()
        self.total_calls += 1

        # Get role defaults from registry if available
        system_prompt_suffix = None
        if self.registry and not skip_suffix:
            default_n, _default_temp, system_prompt_suffix = self.registry.get_role_defaults(role)
            if n_tokens is None:
                n_tokens = default_n
        elif self.registry:
            # Still get default n_tokens even when skipping suffix
            default_n, _default_temp, _ = self.registry.get_role_defaults(role)
            if n_tokens is None:
                n_tokens = default_n
        if n_tokens is None:
            n_tokens = -1  # Unlimited — timeout is the real limit

        # Inject persona system prompt if specified and feature enabled
        if persona and not skip_suffix:
            prompt = self._apply_persona_prefix(prompt, persona)

        # Apply system prompt suffix if configured for this role
        if system_prompt_suffix:
            prompt = f"{prompt}\n\n{system_prompt_suffix}"

        # RAG quality injection for eligible worker roles
        if not skip_suffix and self._should_rag(role):
            full_prompt = self._apply_rag_context(prompt, context_slice)
        elif context_slice:
            full_prompt = f"{prompt}\n\nContext:\n{context_slice}"
        else:
            full_prompt = prompt

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="call",
            prompt=prompt,
            context_slice=context_slice[:500] if context_slice else None,
            role=role,
            persona=persona,
        )

        # Snapshot to guard against double-counting model output below: the real
        # backend path (_real_call -> inference.py) already adds its EXACT completion
        # token count to total_tokens_generated.
        _tokens_before = self.total_tokens_generated
        try:
            if self.mock_mode:
                result = self._mock_call(full_prompt, role)
            else:
                role_for_call = self._resolve_depth_override_role(role)
                sampling_kwargs: dict[str, float | int | bool] = {}
                if temperature is not None:
                    sampling_kwargs["temperature"] = temperature
                if seed is not None:
                    sampling_kwargs["seed"] = seed
                if top_p is not None:
                    sampling_kwargs["top_p"] = top_p
                if top_k is not None:
                    sampling_kwargs["top_k"] = top_k
                if n_probs is not None:
                    sampling_kwargs["n_probs"] = n_probs
                if post_sampling_probs:
                    sampling_kwargs["post_sampling_probs"] = True
                result = self._real_call(
                    full_prompt, role_for_call, n_tokens, stop_sequences,
                    json_schema=json_schema, grammar=grammar,
                    **sampling_kwargs,
                )

            # Cap output
            if len(result) > self.config.output_cap:
                result = (
                    result[: self.config.output_cap]
                    + f"\n[... truncated at {self.config.output_cap} chars]"
                )

            log_entry.result = result[:500]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            completion_tokens = self._estimate_completion_tokens(result)
            # Count model-decode tokens exactly ONCE. The real backend path already
            # added its exact completion count (inference.py); re-adding the
            # char-estimate here was double-counting and ~2x-inflating the speed
            # objective. Only fall back to the estimate when the backend did NOT count
            # (mock mode / content-cache hit, which return before the backend add).
            # INVARIANT: total_tokens_generated == MODEL-decoded tokens, once each.
            # Tool output is never counted here (it returns as next-turn input);
            # effective tool-output throughput lives only in eval_log_format.py (logs).
            if self.total_tokens_generated == _tokens_before:
                self.total_tokens_generated += completion_tokens
            if self._current_query is not None:
                prompt_tokens = self._estimate_prompt_tokens(full_prompt)
                self._current_query.prompt_tokens += prompt_tokens
                self._current_query.completion_tokens += completion_tokens
                self._current_query.total_tokens += prompt_tokens + completion_tokens
                self._current_query.calls_made += 1
                self._current_query.elapsed_seconds += log_entry.elapsed_seconds

            return result

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            if isinstance(e, AdmissionDenied):
                return AdmissionDeniedText(e)
            return f"[ERROR: {e}]"

    def _should_rag(self, role: str) -> bool:
        """Check if RAG quality injection should be applied for this role."""
        try:
            from src.services.corpus_retrieval import CorpusRetriever
            retriever = CorpusRetriever.get_instance()
            # Check RAG config on retriever; if not set, try registry
            rag_enabled = retriever.config.rag_enabled
            rag_roles = retriever.config.rag_roles
            if not rag_enabled and self.registry:
                corpus_cfg = self.registry.get_corpus_config()
                rag_enabled = corpus_cfg.get("rag_enabled", False)
                rag_roles = corpus_cfg.get("rag_roles")
                if rag_enabled:
                    # Propagate to singleton for future calls
                    retriever.config.rag_enabled = True
                    retriever.config.rag_roles = rag_roles
            if not rag_enabled:
                return False
            if rag_roles and role not in rag_roles:
                return False
            return True
        except Exception:
            return False

    def _apply_rag_context(self, prompt: str, context_slice: str) -> str:
        """Inject corpus RAG context into worker prompt."""
        from src.services.corpus_retrieval import CorpusRetriever, extract_code_query
        retriever = CorpusRetriever.get_instance()
        query = extract_code_query(prompt)
        snippets = retriever.retrieve_for_rag(query)
        if context_slice:
            task = f"{prompt}\n\nContext:\n{context_slice}"
        else:
            task = prompt
        if not snippets:
            return task
        return retriever.format_for_rag(snippets, task)

    def llm_batch(
        self,
        prompts: list[str],
        role: str = "worker",
        persona: str | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        n_tokens: int | None = None,
    ) -> list[str]:
        """Call multiple sub-LMs in parallel.

        Args:
            prompts: List of prompts to send to sub-LMs.
            role: Role determining which model to use.
            persona: Optional persona name for system prompt injection.
            json_schema: Optional JSON schema constraining EVERY prompt in
                this batch (TD-21.22a). One schema per call, not per-prompt —
                matches how ``combined_ops._batch_llm_query`` builds a batch
                (one schema shared by the whole list of prompts); a batch
                that genuinely needs per-prompt schemas should call
                ``llm_call`` per prompt instead. Threaded down to the same
                ``InferenceRequest.json_schema`` field ``llm_call`` uses.
                Default None reproduces the pre-TD-21.22a call exactly.
            grammar: Optional GBNF grammar, applied to the whole batch the
                same way as json_schema.
            n_tokens: Optional max-tokens cap applied to EVERY prompt in this
                batch (same per-batch semantics as json_schema/grammar above;
                ``llm_call`` has always had this parameter — two production
                callers, `src/api/routes/chat_summarization.py` and
                `src/repl_environment/context.py`, already passed it to
                ``llm_batch`` under the assumption it existed, which raised
                ``TypeError`` against the real class (MagicMock-based tests
                never caught it) and silently degraded to a
                slower/less-parallel fallback path. Threaded down to the same
                ``InferenceRequest.n_tokens``/``max_tokens`` fields
                ``llm_call`` uses. Default None (unlimited, timeout-bounded)
                reproduces the pre-existing call shape.

        Returns:
            List of responses in the same order as prompts.
        """
        start_time = time.perf_counter()
        self.total_batch_calls += 1

        # Apply persona prefix to all prompts if specified
        if persona:
            prompts = [self._apply_persona_prefix(p, persona) for p in prompts]

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="batch",
            prompts=prompts[:5] if len(prompts) <= 5 else prompts[:5] + ["..."],
            role=role,
            persona=persona,
        )

        _tokens_before = self.total_tokens_generated  # double-count guard (see call())
        try:
            if self.mock_mode:
                results = self._mock_batch(
                    prompts, role, json_schema=json_schema, grammar=grammar, n_tokens=n_tokens,
                )
            else:
                results = self._real_batch(
                    prompts, role, json_schema=json_schema, grammar=grammar, n_tokens=n_tokens,
                )

            # Cap each output
            capped_results = []
            for result in results:
                if len(result) > self.config.output_cap:
                    result = (
                        result[: self.config.output_cap]
                        + f"\n[... truncated at {self.config.output_cap} chars]"
                    )
                capped_results.append(result)

            log_entry.result = [r[:200] for r in capped_results[:3]]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            total_completion_tokens = sum(
                self._estimate_completion_tokens(r) for r in capped_results
            )
            # Count once: skip the char-estimate if the backend already counted (see call()).
            if self.total_tokens_generated == _tokens_before:
                self.total_tokens_generated += total_completion_tokens
            if self._current_query is not None:
                total_prompt_tokens = sum(self._estimate_prompt_tokens(p) for p in prompts)
                self._current_query.prompt_tokens += total_prompt_tokens
                self._current_query.completion_tokens += total_completion_tokens
                self._current_query.total_tokens += total_prompt_tokens + total_completion_tokens
                self._current_query.batch_calls_made += 1
                self._current_query.elapsed_seconds += log_entry.elapsed_seconds

            return capped_results

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return [f"[ERROR: {e}]" for _ in prompts]

    async def llm_batch_async(
        self,
        prompts: list[str],
        role: str = "worker",
        persona: str | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        n_tokens: int | None = None,
    ) -> list[str]:
        """Call multiple sub-LMs in parallel using asyncio.

        This is the async version of llm_batch() for use in async contexts.
        Uses asyncio.gather for parallel execution.

        Args:
            prompts: List of prompts to send to sub-LMs.
            role: Role determining which model to use.
            persona: Optional persona name for system prompt injection.
            json_schema: Optional JSON schema constraining every prompt in
                this batch (TD-21.22a; see ``llm_batch`` for the per-batch
                vs per-prompt rationale). Default None reproduces the
                pre-TD-21.22a call exactly.
            grammar: Optional GBNF grammar for the whole batch.
            n_tokens: Optional max-tokens cap for the whole batch (see
                ``llm_batch``'s docstring for why this exists). Default None
                reproduces the pre-existing call shape.

        Returns:
            List of responses in the same order as prompts.
        """
        start_time = time.perf_counter()
        self.total_batch_calls += 1

        # Apply persona prefix to all prompts if specified
        if persona:
            prompts = [self._apply_persona_prefix(p, persona) for p in prompts]

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="batch_async",
            prompts=prompts[:5] if len(prompts) <= 5 else prompts[:5] + ["..."],
            role=role,
            persona=persona,
        )

        # Only forward the constraint kwargs when set, so a caller that omits
        # them reaches _real_call with the exact pre-TD-21.22a argument list.
        _extra: dict[str, Any] = {}
        if json_schema is not None:
            _extra["json_schema"] = json_schema
        if grammar is not None:
            _extra["grammar"] = grammar
        if n_tokens is not None:
            _extra["n_tokens"] = n_tokens

        _tokens_before = self.total_tokens_generated  # double-count guard (see call())
        try:
            if self.mock_mode:
                # Mock mode: simulate async calls
                results = self._mock_batch(
                    prompts, role, json_schema=json_schema, grammar=grammar, n_tokens=n_tokens,
                )
            else:
                role_limit = self._get_role_limit(role)
                if role_limit <= 1:
                    results = []
                    for prompt in prompts:
                        results.append(
                            await asyncio.to_thread(self._real_call, prompt, role, **_extra)
                        )
                else:
                    # Real mode: run calls in parallel using asyncio
                    loop = asyncio.get_event_loop()
                    tasks = [
                        loop.run_in_executor(
                            None,
                            self._bind_current_context(
                                self._real_call, prompt, role, **_extra
                            ),
                        )
                        for prompt in prompts
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Convert exceptions to error strings
                    results = [str(r) if isinstance(r, Exception) else r for r in results]

            # Cap each output
            capped_results = []
            for result in results:
                if len(result) > self.config.output_cap:
                    result = (
                        result[: self.config.output_cap]
                        + f"\n[... truncated at {self.config.output_cap} chars]"
                    )
                capped_results.append(result)

            log_entry.result = [r[:200] for r in capped_results[:3]]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            total_completion_tokens = sum(
                self._estimate_completion_tokens(r) for r in capped_results
            )
            # Count once: skip the char-estimate if the backend already counted (see call()).
            if self.total_tokens_generated == _tokens_before:
                self.total_tokens_generated += total_completion_tokens
            if self._current_query is not None:
                total_prompt_tokens = sum(self._estimate_prompt_tokens(p) for p in prompts)
                self._current_query.prompt_tokens += total_prompt_tokens
                self._current_query.completion_tokens += total_completion_tokens
                self._current_query.total_tokens += total_prompt_tokens + total_completion_tokens
                self._current_query.batch_calls_made += 1
                self._current_query.elapsed_seconds += log_entry.elapsed_seconds

            return capped_results

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return [f"[ERROR: {e}]" for _ in prompts]

    def llm_call_monitored(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        monitor: Any = None,
        expected_length: int | None = None,
    ) -> LLMResult:
        """Call a sub-LM with generation monitoring for early abort.

        This method integrates with GenerationMonitor to detect likely
        failures during generation and abort early to save compute.

        Args:
            prompt: The instruction/prompt for the sub-LM.
            context_slice: Optional context to include.
            role: Role determining which model to use.
            monitor: GenerationMonitor instance (required).
            expected_length: Expected output length (for runaway detection).

        Returns:
            LLMResult with text and abort information.

        Raises:
            ValueError: If monitor is None.
        """
        if monitor is None:
            raise ValueError("monitor required for llm_call_monitored")

        start_time = time.perf_counter()
        self.total_calls += 1

        # Build full prompt
        if context_slice:
            full_prompt = f"{prompt}\n\nContext:\n{context_slice}"
        else:
            full_prompt = prompt

        # Set expected length if provided
        if expected_length:
            monitor.expected_length = expected_length

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="call_monitored",
            prompt=prompt,
            context_slice=context_slice[:500] if context_slice else None,
            role=role,
        )

        try:
            if self.mock_mode:
                result = self._mock_call_monitored(full_prompt, role, monitor)
            else:
                with self._acquire_role(role):
                    result = self._real_call_monitored(full_prompt, role, monitor)

            log_entry.result = result.text[:500] if result.text else None
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            result.elapsed_seconds = log_entry.elapsed_seconds
            return result

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return LLMResult(
                text=f"[ERROR: {e}]",
                aborted=False,
                elapsed_seconds=log_entry.elapsed_seconds,
            )
