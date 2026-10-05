"""Real inference methods using backends."""

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.backends import serving_calls
from src.exceptions import AdmissionDenied, ContextOverflowError
from src.scheduling import gate_observation

from .types import LLMResult

log = logging.getLogger(__name__)


def _prefix_index_kwargs(request: Any) -> dict[str, Any]:
    """``{"prefix_key": ...}`` for the KV pool gate when ORCHESTRATOR_PREFIX_INDEX
    is on (RTG-58 P2), else ``{}`` so the gate call is unchanged. Never raises."""
    try:
        from src.inference import prefix_index

        if prefix_index.enabled():
            return {"prefix_key": prefix_index.key_text_for_request(request)}
    except Exception:
        log.debug("prefix index: key failed", exc_info=True)
    return {}


def _frontdoor_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_FRONTDOOR_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_frontdoor_role(role: str) -> bool:
    norm = (role or "").strip().lower()
    return norm == "frontdoor" or norm.endswith(".frontdoor")


def _request_is_grammar_constrained(request: Any) -> bool:
    """Whether *request* already bounds output shape via json_schema/grammar.

    The streaming repetition guard (`_detect_streaming_repetition`) exists to
    catch degenerate free-form loops (the same paragraph repeated forever).
    A schema- or grammar-constrained response is a different shape: TD-1's
    per-question JSON answers are legitimately near-identical objects
    (``{"noul": false, "probabilities": {...}, "confidence": 1.0}`` repeated
    once per question), and the grammar/schema already bounds the output —
    there is no unbounded-loop risk for the guard to protect against. See
    TD-1d.2 (handoffs/active/typed-decision-plane.md): the guard aborted a
    24-question schema-constrained JSON response after 300 chunks, producing
    a truncated, unparseable object with zero decisions recovered.
    """
    return bool(getattr(request, "json_schema", None)) or bool(getattr(request, "grammar", None))


def _detect_streaming_repetition(text: str, min_block: int = 60, min_repeats: int = 3) -> bool:
    """Detect paragraph-level repetition in streaming text.

    Scans the most recent output for any line-aligned block of >= *min_block*
    chars that appears *min_repeats* or more times.  Designed to catch
    degenerate generation loops where the model repeats the same paragraph.

    Only called periodically (every ~50 chunks) to avoid overhead.
    """
    if len(text) < min_block * min_repeats:
        return False
    # Work on a reasonable tail to limit cost.
    tail = text[-4000:] if len(text) > 4000 else text
    # Split into lines and check line-aligned blocks.
    lines = tail.split("\n")
    if len(lines) < 6:
        return False
    # Slide a window of 3 consecutive lines over the tail.  If the same
    # 3-line block appears min_repeats times, it's a repetition loop.
    for i in range(len(lines) - 2):
        block = "\n".join(lines[i : i + 3])
        if len(block) < min_block or not block.strip():
            continue
        if tail.count(block) >= min_repeats:
            return True
    return False


def _primary_url(url_str: str) -> str:
    """Extract the first concrete HTTP URL from a config URL list."""
    if not url_str:
        return ""
    url = url_str.split(",")[0].strip()
    if url.startswith("full:"):
        url = url[len("full:") :]
    return url


def _extract_port(url: str) -> int | None:
    """Extract port number from a backend URL like 'http://localhost:8080'."""
    if not url:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(_primary_url(url))
        return parsed.port
    except Exception:
        return None


def _raise_if_context_overflow(result: Any, role: str, backend_url: str) -> None:
    """Turn a backend result that carries a context overflow into the typed error."""
    overflow = getattr(result, "context_overflow", None)
    if not isinstance(overflow, dict) or not overflow:
        return
    from src.backends.context_overflow import context_overflow_error_from_info

    raise context_overflow_error_from_info(overflow, role=role, backend_url=backend_url)


# Generation budget assumed for a request that sets no max_tokens (the
# chat-completions path uses the same default).
DEFAULT_GENERATION_BUDGET_TOKENS = 4096
# Headroom kept below n_ctx when clamping max_tokens (template/special tokens
# the character estimate cannot see).
MAX_TOKENS_CLAMP_MARGIN = 64
# Below this, clamping would leave no useful answer; leave the request as is
# and let the server be the judge (400 → reroute / typed error).
MIN_CLAMPED_MAX_TOKENS = 256


def _clamp_max_tokens(prompt: str, n_tokens: int, limit: Any) -> tuple[int, dict[str, Any] | None]:
    """Clamp ``n_tokens`` so prompt + generation fits the role's per-request n_ctx.

    llama-server never checks max_tokens: with context shift off it stops at
    n_ctx - 1 and marks the result truncated (server-context.cpp:1929-1936).
    vLLM refuses such a request, TGI's max_total_tokens budgets it; we clamp and
    annotate. Unbounded (<= 0) budgets are left alone.
    """
    if limit is None or not n_tokens or n_tokens <= 0:
        return n_tokens, None
    from src.backends.context_limits import estimate_tokens_conservative

    prompt_est = estimate_tokens_conservative(prompt)
    room = int(limit.per_request_n_ctx) - prompt_est - MAX_TOKENS_CLAMP_MARGIN
    if prompt_est + n_tokens < int(limit.per_request_n_ctx) - MAX_TOKENS_CLAMP_MARGIN:
        return n_tokens, None
    if room < MIN_CLAMPED_MAX_TOKENS:
        return n_tokens, None
    return room, {
        "from": int(n_tokens),
        "to": int(room),
        "n_ctx": int(limit.per_request_n_ctx),
        "prompt_tokens_est": int(prompt_est),
        "source": getattr(limit, "source", ""),
    }


def _request_cap_refusal(prompt: str, limit: Any) -> int | None:
    """Estimated prompt tokens when they reach a BINDING model cap, else None.

    Only when the cap is below the server's own slot n_ctx (the server would
    accept the request); otherwise the server decides as before. Uses the rough
    4 chars/token estimate so an estimate error never refuses a request that
    fits — the cap protects quality (n_ctx_train), it is not a hard VRAM bound.
    """
    if limit is None or not getattr(limit, "cap_binding", False):
        return None
    from src.backends.context_limits import estimate_tokens

    prompt_est = estimate_tokens(prompt)
    return prompt_est if prompt_est >= int(limit.per_request_n_ctx) else None


def _prefill_allowance(server_urls: str | None, prompt: str) -> dict[str, Any] | None:
    """UFH14-B1 F1: the seconds a long prompt's SILENT prefill needs on top of the
    role timeout (``src/backends/serving_params.py``: the server's measured prefill
    rate from serving-call records, per-request context from ContextLimitResolver).
    None for a short prompt or when nothing can be derived — the timeout is then
    exactly today's. Never raises."""
    if not server_urls:
        return None
    try:
        from src.backends.context_limits import estimate_tokens_conservative
        from src.backends.serving_params import get_serving_params_resolver

        return get_serving_params_resolver().prefill_allowance(
            server_urls, estimate_tokens_conservative(prompt)
        )
    except Exception:
        log.debug("prefill allowance unavailable", exc_info=True)
        return None


def _derived_prefill_timeout_enabled() -> bool:
    try:
        from src.features import features as _get_features

        return bool(_get_features().derived_prefill_timeout)
    except Exception:
        return False


def _note_prefill_budget(
    allowance: dict[str, Any],
    timeout_s: Any,
    *,
    prefill_tokens_est: Any = None,
    raised: bool = False,
) -> None:
    """Stamp the allowance and the post-clamp timeout on the serving record
    (``serving_params.budget_block``: ``at_risk`` vs the padded allowance, ``doomed``
    when the timeout cannot cover even one prefill of the uncached tokens — the call
    is abandoned mid-prefill while the server keeps prefilling it, D1). Never raises."""
    try:
        from src.backends.serving_params import budget_block

        block = budget_block(
            allowance, timeout_s, prefill_tokens_est=prefill_tokens_est, raised=raised
        )
        serving_calls.annotate_staged(serving_params=block)
        if block["doomed"]:
            log.warning(
                "doomed prefill: ~%s uncached prompt tokens need ~%ss of silent prefill on %s "
                "but the call's timeout is %ss; the server will keep prefilling after it",
                block.get("expected_prefill_tokens"), block.get("expected_prefill_s"),
                allowance.get("url"), block["timeout_s"],
            )
    except Exception:
        pass


def _sampling_cache_key(
    *,
    temperature: float | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    sampling = {
        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "top_k": top_k,
    }
    active = {key: value for key, value in sampling.items() if value is not None}
    if not active:
        return ""
    return json.dumps(active, sort_keys=True, separators=(",", ":"))


def _role_config_for_backend(registry: Any | None, role: str):
    """Use registry generation defaults instead of a synthetic greedy role."""
    if registry is not None:
        try:
            return registry.get_role(role)
        except Exception:
            pass

    from src.registry_loader import (
        AccelerationConfig,
        MemoryConfig,
        ModelConfig,
        PerformanceMetrics,
        RoleConfig,
    )

    return RoleConfig(
        name=role,
        tier="C",
        description=f"Dynamic role for {role}",
        model=ModelConfig(
            name="dynamic-model",
            path="",  # Backend already knows the model.
            quant="Q4_K_M",
            size_gb=0.0,
        ),
        acceleration=AccelerationConfig(type="baseline", temperature=None),
        performance=PerformanceMetrics(),
        memory=MemoryConfig(residency="warm"),
    )


def _per_region_locks_enabled() -> bool:
    return os.environ.get("ORCHESTRATOR_PER_REGION_LOCKS", "0").strip() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _backend_manages_region_locks(backend: Any) -> bool:
    """True for backends that acquire cpu_region_lock internally."""
    return hasattr(backend, "_dispatch") and hasattr(backend, "_tap_dispatch_metadata")


def _direct_native_batch_lock_spec(
    owner: Any,
    role: str,
    backend_url: str,
    *,
    request: Any | None = None,
) -> tuple[str, int, int] | None:
    """Return the certified shared-lock spec for one direct CPU backend.

    Direct auxiliary lanes do not have a ``ConcurrencyAwareBackend`` dispatcher,
    but their external region lock still owns placement and native-slot capacity.
    This helper is shared by the coarse-gate bypass and the actual lock acquire so
    the two decisions cannot drift apart.
    """
    if not backend_url or not _per_region_locks_enabled():
        return None
    try:
        from src.llm_primitives.backend import _certified_native_batch_width
        from src.runtime.cpu_region_lock import _cross_role_mutex_enabled
        from src.runtime.instance_topology import topology_instance_for_port

        resolved = topology_instance_for_port(_extract_port(backend_url) or 0)
        lock_role, lock_idx = resolved or (role, 0)
        if request is None:
            workload_class = owner.get_request_workload_class()
            batch_id = owner.get_request_batch_id()
            placement_mode = owner.get_batch_placement_mode()
        else:
            workload_class = getattr(request, "workload_class", "")
            batch_id = getattr(request, "batch_id", "")
            placement_mode = getattr(request, "batch_placement_mode", "auto")
        native_width = _certified_native_batch_width(
            owner,
            topology_role=lock_role,
            full_url=backend_url,
            logical_role=role,
        )
        if (
            native_width > 1
            and _cross_role_mutex_enabled()
            and str(workload_class or "") == "eval_batch"
            and bool(str(batch_id or "").strip())
            and (
                str(placement_mode or "auto") != "mixed_role_split"
                or lock_role == "eval_batch_frontdoor"
            )
            and lock_idx == 0
        ):
            return lock_role, lock_idx, native_width
    except Exception:
        return None
    return None


def _shape_for_regions(regions: frozenset[str] | set[str] | list[str]) -> str:
    rs = set(regions or [])
    if rs == {"q0", "q1", "q2", "q3"}:
        return "full"
    if rs == {"q0", "q1"}:
        return "half0"
    if rs == {"q2", "q3"}:
        return "half1"
    if len(rs) == 1:
        return next(iter(rs))
    return "+".join(sorted(rs)) if rs else "unknown"


def _direct_region_lock_metadata(
    role: str,
    backend_url: str,
    port: int | None,
    *,
    topology_role: str | None = None,
    topology_idx: int = 0,
) -> dict[str, Any]:
    """Tap metadata for a direct backend's resolved physical topology instance."""
    try:
        from src.runtime.instance_topology import get_instance_regions

        instance_regions = get_instance_regions()
    except Exception:
        return {}
    lock_role = topology_role or role
    key = (lock_role, topology_idx)
    if key not in instance_regions:
        return {}
    regions = instance_regions.get(key, frozenset())
    return {
        "topology_role": lock_role,
        "lock_role": lock_role,
        "instance_idx": topology_idx,
        "concurrency_idx": -1,
        "instance_shape": _shape_for_regions(regions),
        "instance_regions": sorted(regions),
        "port": port,
        "backend_url": backend_url,
    }


def _metadata_from_request_or_env(
    request: Any,
    attr_names: tuple[str, ...],
    env_names: tuple[str, ...],
) -> Any:
    for name in attr_names:
        value = getattr(request, name, None)
        if value not in (None, ""):
            return value
    extra = getattr(request, "extra", None)
    if isinstance(extra, dict):
        for name in attr_names:
            value = extra.get(name)
            if value not in (None, ""):
                return value
    for name in env_names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


def _build_tap_metadata(
    request: Any,
    *,
    role: str,
    backend_url: str,
    port: int | None,
    task_id: str | None,
    parent_request_id: str | None = None,
    context_trial_id: Any = None,
    context_batch_id: Any = None,
) -> dict[str, Any]:
    parent_request_id = parent_request_id or _metadata_from_request_or_env(
        request,
        ("parent_request_id", "request_id"),
        ("ORCHESTRATOR_REQUEST_ID", "AUTOPILOT_REQUEST_ID"),
    )
    request_id = _metadata_from_request_or_env(
        request,
        ("inference_request_id",),
        ("ORCHESTRATOR_INFERENCE_REQUEST_ID", "AUTOPILOT_INFERENCE_REQUEST_ID"),
    )
    if not request_id:
        base = parent_request_id or task_id
        request_id = f"{base}:{uuid.uuid4().hex[:8]}" if base else None
    trial_id = _metadata_from_request_or_env(
        request,
        ("trial_id", "trial", "current_trial"),
        ("ORCHESTRATOR_TRIAL_ID", "AUTOPILOT_TRIAL_ID", "GEPA_TRIAL_ID"),
    )
    if trial_id in (None, ""):
        trial_id = context_trial_id
    batch_id = _metadata_from_request_or_env(
        request,
        ("batch_id", "concurrency_batch_id", "eval_batch_id"),
        ("ORCHESTRATOR_BATCH_ID", "AUTOPILOT_BATCH_ID", "GEPA_BATCH_ID"),
    )
    if batch_id in (None, ""):
        batch_id = context_batch_id
    return {
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "task_id": task_id or getattr(request, "task_id", None),
        "trial_id": trial_id,
        "batch_id": batch_id,
        "batch_placement_mode": getattr(request, "batch_placement_mode", "auto"),
        "role": role,
        "backend_url": backend_url,
        "port": port,
    }


def _request_chat_payload(owner: Any) -> dict[str, Any] | None:
    """HS-4 P0.1: structured /v1 client-tool-mode payload bound to this call."""
    getter = getattr(owner, "get_request_chat_payload", None)
    payload = getter() if callable(getter) else None
    return payload if isinstance(payload, dict) else None


def _request_trace_keys(owner: Any) -> dict[str, Any]:
    """HS-4 P0.2: typed /v1 request keys to stamp onto the inference trace."""
    getter = getattr(owner, "get_request_trace_keys", None)
    keys = getter() if callable(getter) else None
    return dict(keys) if isinstance(keys, dict) else {}


class InferenceMixin:
    """Mixin for real inference methods."""

    def _stage_serving_caller(
        self, role: str, request: Any, backend_url: str | None, port: int | None
    ) -> None:
        """Stage this call's caller identity for the per-call serving record.

        Called BEFORE the inference / region lock is taken, so the recorded
        ``queue.pre_dispatch_wait_ms`` is the time this call spent queued. Never
        raises (``src/backends/serving_calls.py``).
        """
        try:
            meta = _build_tap_metadata(
                request,
                role=role,
                backend_url=backend_url or "",
                port=port,
                task_id=self.get_request_task_id(),
                parent_request_id=(
                    self.get_request_id() if hasattr(self, "get_request_id") else None
                ),
                context_trial_id=(
                    self.get_request_trial_id() if hasattr(self, "get_request_trial_id") else None
                ),
                context_batch_id=(
                    self.get_request_batch_id() if hasattr(self, "get_request_batch_id") else None
                ),
            )
            getter = getattr(self, "get_request_session_id", None)
            meta["session_id"] = getter() if callable(getter) else None
            getter = getattr(self, "get_request_workload_class", None)
            meta["workload_class"] = getter() if callable(getter) else None
            getter = getattr(self, "get_request_priority", None)
            meta["priority"] = getter() if callable(getter) else None
            meta["trace_keys"] = _request_trace_keys(self) or None
            meta["client"] = os.environ.get("ORCHESTRATOR_CLIENT_ID") or None
            serving_calls.stage_caller(**meta)
        except Exception:
            pass

    def _refuse_if_role_parked(self, role: str) -> None:
        """Raise ``RoleParkedError`` (and request a preempt) for a parked role."""
        from src.runtime import gpu_window

        backend_url = (
            _primary_url(self.server_urls.get(role, "")) if getattr(self, "server_urls", None) else ""
        )
        request_id = None
        for getter_name in ("get_request_id", "get_request_task_id"):
            getter = getattr(self, getter_name, None)
            if callable(getter):
                try:
                    request_id = getter() or None
                except Exception:
                    request_id = None
                if request_id:
                    break
        gpu_window.refuse_if_parked(
            role,
            _extract_port(backend_url) if backend_url else None,
            request_id=request_id,
            base_url=backend_url or None,
            caller={"source": "primitives", "role": role, "request_id": request_id},
        )

    def _set_last_inference_meta(self, meta: dict[str, Any]) -> None:
        """Record this call's metadata on BOTH channels (TD-21.21 coordinator fix).

        `self._last_inference_meta` is the long-standing plain attribute every existing
        consumer (`graph/helpers.py`, `typed_decisions/*`, `chat_pipeline/telemetry.py`,
        `chat_delegation.py`, ...) already reads -- kept unchanged so none of them regress.
        `self._last_inference_meta_ctx` is the per-call-safe `contextvars.ContextVar`
        counterpart (`src/llm_primitives/primitives.py`): a concurrent call against this same
        (often SHARED) primitives instance runs in its OWN copied context
        (`asyncio.to_thread`/`asyncio.Task` each `contextvars.copy_context()` at creation), so
        it can only ever mutate its OWN copy of this var, never this one. Every site that used
        to write `self._last_inference_meta = {...}` directly now goes through this one method
        so a future call site cannot add a 6th assignment that forgets the ContextVar half.
        """
        self._last_inference_meta = meta
        self._last_inference_meta_ctx.set(meta)

    def _real_call(
        self,
        prompt: str,
        role: str,
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        n_probs: int | None = None,
        post_sampling_probs: bool = False,
    ) -> str:
        """Make a real inference call via CachingBackend or legacy ModelServer.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.
            n_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences to halt generation.
            json_schema: Optional JSON schema to constrain output structure.
            grammar: Optional GBNF grammar for constrained generation.
            temperature: Optional explicit decode temperature override.
            seed: Optional explicit deterministic decode seed.
            top_p: Optional explicit nucleus sampling override.
            top_k: Optional explicit top-k sampling override.
            n_probs: Optional llama.cpp top-k token probability capture.
            post_sampling_probs: When True, the n_probs top-k is captured
                AFTER the full sampler chain (grammar mask included).

        Returns:
            Model response.

        Raises:
            RuntimeError: If no backend configured for this role.
        """
        # Parked role (GPU lent to AutoKernel, src/runtime/gpu_window.py): refuse
        # FIRST — before the contention gate, role semaphore and region locks — so
        # the caller gets an explicit role_parked error in microseconds, not a
        # queue wait followed by connection-refused. llm_call turns the raise into
        # the in-band ``[ERROR: role_parked: ...]`` sentinel (a /chat infra failure,
        # 503 via _annotate_error). Not parked = one cached stat.
        self._refuse_if_role_parked(role)

        # Cross-role contention gate (Phase B of cross-role-bw-aware-routing).
        # MUST run BEFORE _acquire_role — the per-role semaphore would hide
        # an admitted request from the active-decode snapshot during its wait.
        # ContentionDenied surfaces as 503 + Retry-After at the chat route.
        from src.scheduling.contention_gate import (
            _DEFAULT_BACKGROUND_WAIT_MS,
            _DEFAULT_FOREGROUND_WAIT_MS,
            ContentionDenied,
            get_gate,
        )
        from src.scheduling.contention import TrafficClass, shape_aware_contention_enabled

        priority = (
            self.get_request_priority() if hasattr(self, "get_request_priority") else "interactive"
        )
        traffic_class = (
            TrafficClass.BACKGROUND
            if str(priority).lower() == "background"
            else TrafficClass.FOREGROUND_INTERACTIVE
        )
        max_wait_ms = (
            self.get_max_queue_wait_ms() if hasattr(self, "get_max_queue_wait_ms") else None
        )
        workload_class = (
            self.get_request_workload_class()
            if hasattr(self, "get_request_workload_class")
            else "interactive"
        )
        wait_budget_ms = max_wait_ms
        if wait_budget_ms is None:
            wait_budget_ms = (
                _DEFAULT_BACKGROUND_WAIT_MS
                if traffic_class == TrafficClass.BACKGROUND
                else _DEFAULT_FOREGROUND_WAIT_MS
            )

        backend = self._backends.get(role) if hasattr(self, "_backends") else None
        backend_url = _primary_url(self.server_urls.get(role, "")) if self.server_urls else ""
        direct_native_lock = (
            _direct_native_batch_lock_spec(self, role, backend_url)
            if backend is not None and not _backend_manages_region_locks(backend)
            else None
        )
        defer_to_dispatch = (
            backend is not None
            and shape_aware_contention_enabled()
            and (_backend_manages_region_locks(backend) or direct_native_lock is not None)
        )
        if not defer_to_dispatch:
            gate = get_gate()
            decision = gate.admit(role, traffic_class, max_wait_ms)
            # BRIDGE RESIDUAL 1 — observational only, after the verdict.
            gate_observation.record(
                admitted=decision.admitted,
                decision=getattr(decision.decision, "value", str(decision.decision)),
                waited_s=decision.waited_s,
                blocking_roles=list(decision.blocking_roles or []),
                reason=decision.reason,
                role=role,
            )
            if not decision.admitted:
                raise ContentionDenied(
                    f"contention gate denied role={role} class={traffic_class.value}: {decision.reason}",
                    role=role,
                    workload_class=str(workload_class or "interactive"),
                    wait_budget_ms=wait_budget_ms,
                    failure_class="admission_denied",
                    code="contention_gate_timeout",
                )

        acquire = getattr(self, "_acquire_role", None)
        if acquire:
            with acquire(role):
                return self._real_call_impl(
                    prompt,
                    role,
                    n_tokens,
                    stop_sequences,
                    json_schema=json_schema,
                    grammar=grammar,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p,
                    top_k=top_k,
                    n_probs=n_probs,
                    post_sampling_probs=post_sampling_probs,
                )
        return self._real_call_impl(
            prompt,
            role,
            n_tokens,
            stop_sequences,
            json_schema=json_schema,
            grammar=grammar,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            n_probs=n_probs,
            post_sampling_probs=post_sampling_probs,
        )

    def _real_call_impl(
        self,
        prompt: str,
        role: str,
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        n_probs: int | None = None,
        post_sampling_probs: bool = False,
    ) -> str:
        """Internal real call implementation (no concurrency gating)."""
        # Content-addressable cache check
        from src.features import features as _get_features

        cache = getattr(self, "_content_cache", None)
        cache_key = None
        if (
            cache is not None
            and _get_features().content_cache
            and stop_sequences is None
            and n_probs is None
            and _request_chat_payload(self) is None
        ):
            from src.llm_cache import ContentAddressableCache

            sampling_key = _sampling_cache_key(
                temperature=temperature, seed=seed, top_p=top_p, top_k=top_k
            )
            cache_key = ContentAddressableCache.make_key(
                prompt,
                role,
                n_tokens,
                model_hash=getattr(self, "_model_hash", ""),
                sampling_key=sampling_key,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        result = None

        def _call_on(target_role: str) -> str:
            return self._real_call_single(
                prompt,
                target_role,
                n_tokens,
                stop_sequences,
                json_schema=json_schema,
                grammar=grammar,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                top_k=top_k,
                n_probs=n_probs,
                post_sampling_probs=post_sampling_probs,
            )

        try:
            result = _call_on(role)
        except ContextOverflowError as overflow:
            # MUST precede `except RuntimeError` (ContextOverflowError is one):
            # same-tier model fallback is the wrong remedy for a request that
            # does not fit, and it would mask the typed error.
            from src.backends.context_limits import context_overflow_roles

            from .context_recovery import recover_context_overflow

            server_urls = getattr(self, "server_urls", None) or {}
            backends = getattr(self, "_backends", None) or {}
            # Only reroute to roles this primitives instance can actually call.
            reroute_candidates = [
                r for r in context_overflow_roles()
                if r in backends or getattr(self, "model_server", None) is not None
            ]
            result = recover_context_overflow(
                overflow,
                prompt=prompt,
                role=role,
                call=_call_on,
                reroute_candidates=reroute_candidates,
                urls_for_role=(lambda r: server_urls.get(r)) if server_urls else None,
                deadline_s=self.get_request_deadline_s()
                if hasattr(self, "get_request_deadline_s") else None,
                cancel_check=self.get_request_cancel_check()
                if hasattr(self, "get_request_cancel_check") else None,
            )
        except RuntimeError as primary_error:
            # Model fallback: try same-tier alternatives on infrastructure failure
            if not _get_features().model_fallback:
                raise

            from src.roles import get_fallback_roles

            fallback_roles = get_fallback_roles(role)
            if not fallback_roles:
                raise

            # Classify the failure
            reason = "unknown"
            if self.health_tracker:
                reason = self.health_tracker.classify_failure(primary_error)

            log = logging.getLogger(__name__)
            for fallback_role in fallback_roles:
                fb_role_str = str(fallback_role)
                log.warning(
                    "Model fallback: %s → %s (reason: %s)",
                    role,
                    fb_role_str,
                    reason,
                )
                try:
                    result = self._real_call_single(
                        prompt,
                        fb_role_str,
                        n_tokens,
                        stop_sequences,
                        json_schema=json_schema,
                        grammar=grammar,
                        temperature=temperature,
                        seed=seed,
                        top_p=top_p,
                        top_k=top_k,
                        n_probs=n_probs,
                        post_sampling_probs=post_sampling_probs,
                    )
                    break
                except RuntimeError:
                    continue
            else:
                # All fallbacks failed
                raise

        # Store in content cache on success
        if result is not None and cache_key is not None and cache is not None:
            cache.put(cache_key, result, metadata={"role": role})

        return result

    def _real_call_single(
        self,
        prompt: str,
        role: str,
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        n_probs: int | None = None,
        post_sampling_probs: bool = False,
    ) -> str:
        """Execute a single inference call against one role's backend."""
        # Try CachingBackend first (RadixAttention)
        backend = self._backends.get(role)
        if backend is not None:
            return self._call_caching_backend(
                backend,
                prompt,
                role,
                n_tokens,
                stop_sequences,
                json_schema=json_schema,
                grammar=grammar,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                top_k=top_k,
                n_probs=n_probs,
                post_sampling_probs=post_sampling_probs,
            )

        # Fall back to legacy ModelServer
        if self.model_server is None:
            raise RuntimeError(
                f"No backend configured for role '{role}'. Provide server_urls or model_server."
            )
        if _request_chat_payload(self) is not None:
            raise RuntimeError(
                f"Client tool mode needs a llama-server chat-completions backend; "
                f"role '{role}' is served by the legacy ModelServer path."
            )

        from src.model_server import InferenceRequest
        from src.config import get_config

        role_timeout = (
            get_config().timeouts.role_timeouts_dict().get(role, self.config.call_timeout)
        )
        role_timeout = self._clamp_timeout_to_request_budget(role_timeout)

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=role_timeout,
            stop_sequences=stop_sequences,
            cache_prompt=self.cache_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            json_schema=json_schema,
            grammar=grammar,
            n_probs=n_probs,
            post_sampling_probs=post_sampling_probs,
        )
        req_started = time.perf_counter()
        from src.inference_lock import inference_lock

        _ms_port = _extract_port(
            (self.server_urls or {}).get(role, "") if hasattr(self, "server_urls") else ""
        )
        self._stage_serving_caller(
            role,
            request,
            (self.server_urls or {}).get(role, "") if hasattr(self, "server_urls") else "",
            _ms_port,
        )
        try:
            with inference_lock(
                role,
                cancel_check=self.get_request_cancel_check(),
                deadline_s=self.get_request_deadline_s(),
                request_tag=self.get_request_task_id(),
                port=_ms_port,
            ):
                request.timeout = self._clamp_timeout_to_request_budget(request.timeout)
                result = self.model_server.infer(role, request)
        except Exception as exc:
            serving_calls.abandon_staged(exc)
            req_elapsed_ms = (time.perf_counter() - req_started) * 1000
            self._set_last_inference_meta({
                "role": role,
                "transport": "model_server",
                "elapsed_ms": req_elapsed_ms,
                "completion_reason": "exception",
                "error": str(exc),
            })
            if "lock timeout" in str(exc).lower() or "cancelled" in str(exc).lower():
                log.warning(
                    "Inference aborted before model call (role=%s, transport=model_server, elapsed_ms=%.1f): %s",
                    role,
                    req_elapsed_ms,
                    exc,
                )
            raise

        serving_calls.clear_staged()
        req_elapsed_ms = (time.perf_counter() - req_started) * 1000
        _raise_if_context_overflow(result, role, (self.server_urls or {}).get(role, "") if hasattr(self, "server_urls") else "")
        self._set_last_inference_meta({
            "role": role,
            "transport": "model_server",
            "elapsed_ms": req_elapsed_ms,
            "first_token_ms": getattr(result, "first_token_ms", 0.0),
            "stream_chunks": getattr(result, "stream_chunks", 0),
            "completion_reason": getattr(result, "completion_reason", "") or "unknown",
            "tokens": result.tokens_generated,
            "prompt_ms": result.prompt_eval_ms,
            "gen_ms": result.generation_ms,
            "overhead_ms": result.http_overhead_ms,
            "completion_probabilities": list(getattr(result, "completion_probabilities", []) or []),
        })
        if _is_frontdoor_role(role) and _frontdoor_trace_enabled():
            log.warning(
                "Frontdoor inference telemetry: transport=model_server elapsed_ms=%.1f "
                "first_token_ms=%.1f chunks=%d completion_reason=%s "
                "tokens=%d prompt_ms=%.1f gen_ms=%.1f overhead_ms=%.1f",
                req_elapsed_ms,
                getattr(result, "first_token_ms", 0.0),
                getattr(result, "stream_chunks", 0),
                getattr(result, "completion_reason", "") or "unknown",
                result.tokens_generated,
                result.prompt_eval_ms,
                result.generation_ms,
                result.http_overhead_ms,
            )

        self.total_tokens_generated += result.tokens_generated
        self.total_prompt_eval_ms += result.prompt_eval_ms
        self.total_generation_ms += result.generation_ms
        if result.predicted_per_second > 0:
            self._last_predicted_tps = result.predicted_per_second
        return result.output

    def _call_caching_backend(
        self,
        backend: Any,
        prompt: str,
        role: str,
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        n_probs: int | None = None,
        post_sampling_probs: bool = False,
    ) -> str:
        """Call a CachingBackend with RadixAttention prefix caching.

        Args:
            backend: CachingBackend instance.
            prompt: The full prompt.
            role: The role name.
            n_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences to halt generation.
            json_schema: Optional JSON schema to constrain output structure.
            grammar: Optional GBNF grammar for constrained generation.
            post_sampling_probs: When True, the n_probs top-k is captured
                AFTER the full sampler chain (grammar mask included) instead
                of the default pre-sampling raw logits (TD-1d.2).

        Returns:
            Model response.
        """
        from src.model_server import InferenceRequest

        role_config = _role_config_for_backend(getattr(self, "registry", None), role)

        from src.config import get_config

        role_timeout = (
            get_config().timeouts.role_timeouts_dict().get(role, self.config.call_timeout)
        )
        role_timeout = self._clamp_timeout_to_request_budget(role_timeout)

        chat_payload = _request_chat_payload(self)
        # RI-23: per-call thinking-off override (chat_completions_roles.thinking_off();
        # None unless thinking_roles_chat_lane is on and the caller asked).
        from src.chat_completions_roles import current_chat_template_kwargs_override

        ctk_override = current_chat_template_kwargs_override()
        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=role_timeout,
            stop_sequences=stop_sequences,
            cache_prompt=self.cache_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            json_schema=json_schema,
            grammar=grammar,
            n_probs=n_probs,
            post_sampling_probs=post_sampling_probs,
            # HS-4 P0.1: only passed when set, so the default request is
            # constructed exactly as before.
            **({"chat_payload": chat_payload} if chat_payload is not None else {}),
            **({"chat_template_kwargs": ctk_override} if ctk_override is not None else {}),
        )
        # Dynamic attrs consumed by ConcurrencyAwareBackend's dispatch-time
        # placement-aware contention gate. The local dataclass has no slots.
        request.request_priority = self.get_request_priority()
        request.workload_class = self.get_request_workload_class()
        request.batch_placement_mode = self.get_batch_placement_mode()
        request.batch_id = self.get_request_batch_id()
        request.max_queue_wait_ms = self.get_max_queue_wait_ms()
        request.session_id = self.get_request_session_id()
        request.task_id = self.get_request_task_id()
        request.parent_request_id = self.get_request_id()
        request.inference_request_id = (
            f"{request.parent_request_id or request.task_id}:{uuid.uuid4().hex[:8]}"
            if (request.parent_request_id or request.task_id)
            else uuid.uuid4().hex
        )
        # Clamp max_tokens to the role's real per-request context before
        # dispatch, so a prompt that fits but whose budget does not comes back
        # as an explicit context_limit, not a silent finish_reason=length.
        max_tokens_clamp = None
        try:
            from src.backends.context_limits import get_context_limit_resolver

            _clamp_limit = get_context_limit_resolver().limit_for_role(
                role, (self.server_urls or {}).get(role) if self.server_urls else None
            )
            clamped, max_tokens_clamp = _clamp_max_tokens(prompt, n_tokens, _clamp_limit)
        except Exception:
            _clamp_limit = None
            clamped, max_tokens_clamp = n_tokens, None
        # Per-request cap: when the model's trained context (config ctx_max), not
        # the server, sets the limit (unified -c above n_ctx_train), the server
        # would accept the prompt, so refuse it here — never dispatched, typed
        # like the server's own 400 so context recovery reroutes/compacts it.
        _cap_refusal = _request_cap_refusal(prompt, _clamp_limit)
        if _cap_refusal is not None:
            raise ContextOverflowError(
                f"context overflow (per-request cap) on role {role}: ~{_cap_refusal} prompt "
                f"tokens exceed the model's per-request limit {_clamp_limit.per_request_n_ctx} "
                f"(server slot n_ctx {_clamp_limit.slot_n_ctx}); never dispatched",
                kind=ContextOverflowError.REQUEST_TOO_LARGE,
                role=role,
                backend_url=getattr(_clamp_limit, "url", "") or "",
                n_prompt_tokens=_cap_refusal,
                n_ctx=_clamp_limit.per_request_n_ctx,
                source="request_cap",
            )
        if max_tokens_clamp is not None:
            log.warning(
                "max_tokens clamped for role=%s: %d -> %d (per-request n_ctx %d, ~%d prompt tokens, %s)",
                role, max_tokens_clamp["from"], max_tokens_clamp["to"], max_tokens_clamp["n_ctx"],
                max_tokens_clamp["prompt_tokens_est"], max_tokens_clamp["source"],
            )
            n_tokens = clamped
            request.n_tokens = clamped
            if hasattr(request, "max_tokens"):
                request.max_tokens = clamped
        wants_probabilities = n_probs is not None and int(n_probs) > 0
        # Probability capture and structured chat payloads (tool calls arrive
        # whole) both need the batch response, never the text stream.
        batch_only = wants_probabilities or getattr(request, "chat_payload", None) is not None
        req_started = time.perf_counter()

        # Admission control: reject early if backend queue is full
        # Use primary URL for admission/health tracking (round-robin handled by backend layer)
        backend_url = _primary_url(self.server_urls.get(role, "")) if self.server_urls else ""
        admission = getattr(self, "admission_controller", None)
        admitted = False
        cancel_check = self.get_request_cancel_check()
        deadline_s = self.get_request_deadline_s()
        request_priority = self.get_request_priority()

        # UFH14-B1 F1 (D1): the role timeout is a DECODE budget; a long prompt's silent
        # prefill is added on top, derived per server (never a constant), when the
        # derived_prefill_timeout flag is on. Covers every fleet instance the call may
        # land on. NOTE: with a request deadline, _clamp_timeout_to_request_budget at
        # dispatch returns the REMAINING budget whatever the input, so this only
        # changes calls WITHOUT a deadline; deadlined calls (interactive and extended
        # eval/task_root alike) are byte-identical and get only the record block.
        prefill_allowance = _prefill_allowance(
            self.server_urls.get(role, "") if self.server_urls else "", prompt
        )
        prefill_timeout_raised = False
        if (
            prefill_allowance
            and prefill_allowance.get("allowance_s")
            and _derived_prefill_timeout_enabled()
        ):
            request.timeout = int(request.timeout or self.config.call_timeout) + int(
                prefill_allowance["allowance_s"]
            )
            prefill_timeout_raised = deadline_s is None
        # HSF-3: the native backend-admission boundary. This is not frontend
        # arrival, and is recorded only on records produced after this hook.
        enqueue_ts_epoch = time.time()
        if backend_url and admission:
            # Bounded wait at admission gate to smooth burst contention while
            # honoring request cancellation/deadlines.
            wait_budget_s = 2.0 if deadline_s is None else None
            if not admission.acquire(
                backend_url,
                priority=request_priority,
                wait=True,
                timeout_s=wait_budget_s,
                deadline_s=deadline_s,
                cancel_check=cancel_check,
            ):
                raise AdmissionDenied(f"[ERROR: admission] Backend queue full for {backend_url}")
            admitted = True

        # Shared (unified) KV pool — the PRIMARY overflow defence: never
        # oversubscribe it. Request slots are not the binding limit there,
        # tokens are, and llama-server fails EVERY in-flight request when the
        # pool runs dry. A request whose reservation (prompt + generation
        # budget) cannot fit alongside the in-flight ones is QUEUED (FCFS,
        # bounded by its own deadline/cancellation) and never dispatched until
        # it fits. Server-side exhaustion retry is only the fallback.
        # Single-URL roles only (a fleet dispatches below, to an endpoint this
        # layer does not know yet). Inert unless the server is known unified.
        pool_admission = None
        pool_ticket = None
        pool_success = False
        if backend_url and "," not in (self.server_urls.get(role, "") or ""):
            try:
                from src.backends.context_limits import get_context_limit_resolver

                pool_limit = get_context_limit_resolver().limit_for_url(backend_url)
            except Exception:
                pool_limit = None
            if pool_limit is not None and pool_limit.shared_pool:
                from src.scheduling.kv_pool_admission import get_shared_pool_admission

                from src.backends.context_limits import estimate_tokens_conservative
                from src.scheduling.kv_pool_admission import KVPoolQueueFull

                pool_admission = get_shared_pool_admission()
                pool_prompt_tokens = estimate_tokens_conservative(prompt)
                pool_new_tokens = n_tokens if n_tokens and n_tokens > 0 else DEFAULT_GENERATION_BUDGET_TOKENS
                pool_tokens_needed = pool_admission.reservation_tokens(
                    backend_url, pool_prompt_tokens, pool_new_tokens
                )
                try:
                    pool_ticket = pool_admission.acquire(
                        backend_url,
                        pool_prompt_tokens,
                        pool_limit.pool_tokens,
                        max_new_tokens=pool_new_tokens,
                        deadline_s=deadline_s,
                        cancel_check=cancel_check,
                        # KVU-15a: lets admission size the long-prefill rule on
                        # NEW tokens (an idle slot's cached prefix is credited).
                        # A chat payload is templated server-side, so its text
                        # is not what the slot caches: whole-prompt sizing.
                        prompt_text=(
                            prompt if getattr(request, "chat_payload", None) is None else None
                        ),
                        # KVU-15c: the primary credit source — the prefix this
                        # server already served (orchestrator-side history),
                        # fingerprinted exactly as the serving record does, so a
                        # chat payload is covered too.
                        prefix_ladder=serving_calls.prefix_ladder(request),
                        # RTG-58 P2: the prefix-index key, only with the flag on.
                        **_prefix_index_kwargs(request),
                    )
                except KVPoolQueueFull as queue_full:
                    if admitted and admission:
                        admission.release(backend_url)
                    raise ContextOverflowError(
                        f"context overflow (shared KV pool admission queue full) on role {role} "
                        f"({backend_url}): {queue_full.queued} requests already queued "
                        f"(limit {queue_full.limit}); retry later — never dispatched",
                        kind=ContextOverflowError.POOL_EXHAUSTED,
                        role=role,
                        backend_url=backend_url,
                        n_ctx=pool_limit.per_request_n_ctx,
                        source="admission",
                    ) from queue_full
                if pool_ticket is None:
                    if admitted and admission:
                        admission.release(backend_url)
                    raise ContextOverflowError(
                        f"context overflow (shared KV pool busy) on role {role} ({backend_url}): "
                        f"queued {pool_tokens_needed} tokens behind "
                        f"{pool_admission.in_flight_tokens(backend_url)} reserved "
                        f"(pool {pool_limit.pool_tokens}"
                        + (
                            ", waiting for the in-flight long prefill"
                            if pool_admission.is_long_prefill(pool_prompt_tokens)
                            and pool_admission.long_prefill_lease_busy(backend_url)
                            else ""
                        )
                        + ") and the request's wait budget ended before it fit; "
                        "it was never dispatched",
                        kind=ContextOverflowError.POOL_EXHAUSTED,
                        role=role,
                        backend_url=backend_url,
                        n_ctx=pool_limit.per_request_n_ctx,
                        source="admission",
                    )

        # WP-12 fleet layer: a fleet-shared backend records health per
        # DISPATCHED endpoint itself (one circuit per fleet port). The
        # legacy primary-URL proxy bookkeeping below is skipped for it —
        # fast-fail instead asks whether ANY fleet endpoint is available
        # (fleet circuit open == all endpoints open). Legacy backends
        # (attribute absent) keep the existing behavior byte-identical.
        fleet_managed = bool(getattr(backend, "fleet_health_managed", False))
        from src.runtime.live_telemetry import (
            bind_lifecycle_context,
            emit_lifecycle_transition,
            reset_lifecycle_context,
        )

        lifecycle_identity = {
            "request_id": request.inference_request_id,
            "task_id": request.task_id,
            "batch_id": request.batch_id,
            "role": role,
            "model": getattr(getattr(role_config, "model", None), "name", None),
            "port": _extract_port(backend_url),
        }
        lifecycle_token = bind_lifecycle_context(**lifecycle_identity)

        try:
            # Circuit breaker: fast-fail if backend is known to be down
            if backend_url and self.health_tracker:
                if fleet_managed:
                    if not backend.any_endpoint_available():
                        raise RuntimeError(f"Backend unavailable (circuit open): {backend_url}")
                elif not self.health_tracker.is_available(backend_url):
                    raise RuntimeError(f"Backend unavailable (circuit open): {backend_url}")

            from src.inference_lock import inference_lock

            can_stream = False
            _cb_port = _extract_port(backend_url)
            # Phase 3 of per-region-lock migration (2026-05-22): while the
            # feature flag is on, ConcurrencyAwareBackend takes precise locks
            # internally for multi-instance pools. Direct single-instance
            # CachingBackend roles (architect_general — the full-CPU :8074
            # instance since the 2026-09-27 swap —, architect_critic, worker_vision) still
            # need an external idx=0 region lock; otherwise they can stream
            # without appearing in /proc/locks or the contention gate snapshot.
            _per_region_on = _per_region_locks_enabled()
            direct_region_metadata: dict[str, Any] = {}
            if backend_url and _per_region_on and not _backend_manages_region_locks(backend):
                from src.runtime.cpu_region_lock import cpu_region_lock_for_instance
                from src.runtime.instance_topology import topology_instance_for_port

                resolved_topology = topology_instance_for_port(_cb_port or 0)
                lock_role, lock_idx = resolved_topology or (role, 0)
                direct_region_metadata = _direct_region_lock_metadata(
                    role,
                    backend_url,
                    _cb_port,
                    topology_role=lock_role,
                    topology_idx=lock_idx,
                )
                lock_kwargs: dict[str, Any] = {
                    "cancel_check": self.get_request_cancel_check(),
                    "deadline_s": self.get_request_deadline_s(),
                    "request_tag": self.get_request_task_id(),
                }
                direct_native_lock = _direct_native_batch_lock_spec(
                    self,
                    role,
                    backend_url,
                    request=request,
                )
                if direct_native_lock is not None:
                    _native_lock_role, _native_lock_idx, native_width = direct_native_lock
                    lock_kwargs.update(
                        shared=True,
                        capacity=native_width,
                        request_tag=str(getattr(request, "batch_id", "") or ""),
                    )
                lock_ctx = cpu_region_lock_for_instance(lock_role, lock_idx, **lock_kwargs)
            else:
                lock_ctx = (
                    inference_lock(
                        role,
                        cancel_check=self.get_request_cancel_check(),
                        deadline_s=self.get_request_deadline_s(),
                        request_tag=self.get_request_task_id(),
                        port=_cb_port,
                    )
                    if (backend_url and not _per_region_on)
                    else contextlib.nullcontext()
                )

            self._stage_serving_caller(role, request, backend_url, _cb_port)
            serving_calls.annotate_staged(enqueue_ts_epoch=enqueue_ts_epoch)
            if pool_admission is not None and pool_ticket is not None:
                # KVU-15c: the admission decision (credit, its source) rides
                # the serving record as ``kv_admission``, so KVU-15b can compare
                # the credited prefix with the server's ``cache_n``.
                serving_calls.annotate_staged(
                    kv_admission=pool_admission.admission_record(pool_ticket)
                )
            try:
                with lock_ctx:
                    request.timeout = self._clamp_timeout_to_request_budget(request.timeout)
                    if prefill_allowance:
                        _kv_info = (
                            pool_admission.admission_record(pool_ticket)
                            if pool_admission is not None and pool_ticket is not None
                            else None
                        ) or {}
                        _note_prefill_budget(
                            prefill_allowance,
                            request.timeout,
                            prefill_tokens_est=(
                                _kv_info.get("prefill_tokens_est")
                                if _kv_info.get("cache_credit_source")
                                else None
                            ),
                            raised=prefill_timeout_raised,
                        )
                    from src.inference_tap import (
                        is_active as _tap_active,
                        should_stream_role as _tap_should_stream_role,
                        tap_section,
                    )

                    tap_enabled = _tap_active() and bool(backend_url)
                    first_output_emitted = False

                    def _emit_first_output(content: str = "") -> None:
                        nonlocal first_output_emitted
                        if first_output_emitted:
                            return
                        first_output_emitted = True
                        if pool_admission is not None:
                            # Prefill is over: hand on the long-prefill lease.
                            pool_admission.prefill_done(backend_url, pool_ticket)
                        emit_lifecycle_transition(
                            "first_output",
                            details={"content_chars": len(content)},
                        )

                    if not _backend_manages_region_locks(backend):
                        emit_lifecycle_transition(
                            "backend_dispatched",
                            details={
                                "backend_url": backend_url,
                                "topology_role": direct_region_metadata.get("topology_role", role),
                                "instance_idx": direct_region_metadata.get("instance_idx", 0),
                                "instance_shape": direct_region_metadata.get(
                                    "instance_shape", "unknown"
                                ),
                                "batch_placement_mode": getattr(
                                    request, "batch_placement_mode", "auto"
                                ),
                            },
                        )
                    can_stream = (
                        tap_enabled
                        and hasattr(backend, "infer_stream_text")
                        and _tap_should_stream_role(role)
                        and not batch_only
                    )
                    if tap_enabled:
                        tap_metadata = _build_tap_metadata(
                            request,
                            role=role,
                            backend_url=backend_url,
                            port=_cb_port,
                            task_id=self.get_request_task_id(),
                            parent_request_id=(
                                self.get_request_id() if hasattr(self, "get_request_id") else None
                            ),
                            context_trial_id=(
                                self.get_request_trial_id()
                                if hasattr(self, "get_request_trial_id")
                                else None
                            ),
                            context_batch_id=(
                                self.get_request_batch_id()
                                if hasattr(self, "get_request_batch_id")
                                else None
                            ),
                        )
                        tap_metadata.update(direct_region_metadata)
                        trace_keys = _request_trace_keys(self)
                        if trace_keys:
                            tap_metadata["request_keys"] = trace_keys
                        with tap_section(role, prompt, metadata=tap_metadata) as tap:
                            if can_stream:
                                # Early-stop: if _early_stop_check is set, wrap the
                                # tap callback to also check accumulated output and
                                # raise StopIteration to abort streaming.
                                _stop_check = getattr(self, "_early_stop_check", None)
                                # Always accumulate for repetition detection,
                                # even when no _early_stop_check is set.
                                _acc: list[str] = []
                                _chunk_count = 0
                                _REP_CHECK_INTERVAL = 50  # check every ~50 chunks
                                # Schema/grammar-constrained requests bound
                                # their own output shape; structural repeats
                                # (one near-identical object per question)
                                # are expected, not a degenerate loop. See
                                # _request_is_grammar_constrained docstring.
                                _rep_guard_active = not _request_is_grammar_constrained(request)

                                _cancel = self.get_request_cancel_check()

                                def _on_chunk_guarded(content: str) -> None:
                                    nonlocal _chunk_count
                                    _emit_first_output(content)
                                    tap.write_chunk(content)
                                    _acc.append(content)
                                    _chunk_count += 1
                                    # Client disconnect → abort streaming to release lock sooner.
                                    if _cancel is not None and _cancel():
                                        raise StopIteration
                                    # Caller-provided early-stop (FINAL, TOON, etc.)
                                    if _stop_check is not None and _stop_check("".join(_acc)):
                                        raise StopIteration
                                    # Repetition guard: check periodically
                                    if _rep_guard_active and _chunk_count % _REP_CHECK_INTERVAL == 0:
                                        if _detect_streaming_repetition("".join(_acc)):
                                            log.warning(
                                                "Repetition loop detected after %d chunks, "
                                                "aborting generation",
                                                _chunk_count,
                                            )
                                            raise StopIteration

                                result = backend.infer_stream_text(
                                    role_config, request, on_chunk=_on_chunk_guarded
                                )
                            else:
                                # Prefer streaming for cancellation support
                                if not batch_only and hasattr(
                                    backend, "infer_stream_text"
                                ):
                                    _cancel_tap = self.get_request_cancel_check()

                                    def _on_chunk_tap(content: str) -> None:
                                        _emit_first_output(content)
                                        if _cancel_tap is not None and _cancel_tap():
                                            raise StopIteration

                                    result = backend.infer_stream_text(
                                        role_config, request, on_chunk=_on_chunk_tap
                                    )
                                else:
                                    result = backend.infer(role_config, request)
                                _emit_first_output(result.output)
                                tap.write_response(result.output)
                                if getattr(request, "chat_payload", None) is not None:
                                    tap.set_metadata(
                                        client_tool_calls=[
                                            (tc.get("function") or {}).get("name")
                                            for tc in (getattr(result, "tool_calls", None) or [])
                                        ]
                                    )
                            _emit_first_output(result.output)
                            tap.write_timings(
                                result.tokens_generated,
                                result.prompt_eval_ms,
                                result.generation_ms,
                                result.predicted_per_second,
                                # 5th positional = terminal prompt-token count
                                # as the server reported it (None when it did
                                # not); the tap records it WITH provenance and
                                # never estimates. Positional so duck-typed
                                # tap stubs (`write_timings(*a)`) keep working.
                                getattr(result, "prompt_tokens", None),
                            )
                    else:
                        # Use streaming even without tap — each chunk is a
                        # cancellation checkpoint, preventing indefinite lock
                        # hold when the httpx batch read hangs.
                        if not batch_only and hasattr(backend, "infer_stream_text"):
                            _cancel_nt = self.get_request_cancel_check()

                            def _cancel_only(content: str) -> None:
                                _emit_first_output(content)
                                if _cancel_nt is not None and _cancel_nt():
                                    raise StopIteration

                            result = backend.infer_stream_text(
                                role_config, request, on_chunk=_cancel_only
                            )
                        else:
                            result = backend.infer(role_config, request)
                        _emit_first_output(result.output)
            except Exception as exc:
                serving_calls.abandon_staged(exc)
                req_elapsed_ms = (time.perf_counter() - req_started) * 1000
                transport = "stream" if can_stream else "batch"
                self._set_last_inference_meta({
                    "role": role,
                    "transport": transport,
                    "elapsed_ms": req_elapsed_ms,
                    "completion_reason": "exception",
                    "error": str(exc),
                })
                if "lock timeout" in str(exc).lower() or "cancelled" in str(exc).lower():
                    log.warning(
                        "Inference aborted before backend response (role=%s, transport=%s, elapsed_ms=%.1f): %s",
                        role,
                        transport,
                        req_elapsed_ms,
                        exc,
                    )
                raise

            serving_calls.clear_staged()
            req_elapsed_ms = (time.perf_counter() - req_started) * 1000
            transport = "stream" if can_stream else "batch"
            # Keep a LOCAL handle on this call's dict: the plain attribute is shared by
            # concurrent calls on this instance (asyncio.to_thread), so the follow-up
            # writes below must not re-read it (TD-21.33c).
            call_meta: dict[str, Any] = {
                "role": role,
                "transport": transport,
                "elapsed_ms": req_elapsed_ms,
                "first_token_ms": getattr(result, "first_token_ms", 0.0),
                "stream_chunks": getattr(result, "stream_chunks", 0),
                "completion_reason": getattr(result, "completion_reason", "") or "unknown",
                "tokens": result.tokens_generated,
                "prompt_ms": result.prompt_eval_ms,
                "gen_ms": result.generation_ms,
                "overhead_ms": result.http_overhead_ms,
                "completion_probabilities": list(
                    getattr(result, "completion_probabilities", []) or []
                ),
            }
            self._set_last_inference_meta(call_meta)
            _reasoning = getattr(result, "reasoning_content", None)
            if isinstance(_reasoning, str) and _reasoning:
                from src.chat_completions_roles import thinking_roles_chat_lane_enabled

                if thinking_roles_chat_lane_enabled():
                    # RI-23: the server-split reasoning, for /v1 reasoning_content and
                    # telemetry. Key absent unless the flag is on (flag-off meta unchanged).
                    call_meta["reasoning_content"] = _reasoning
            if getattr(request, "chat_payload", None) is not None:
                call_meta["tool_calls"] = list(
                    getattr(result, "tool_calls", None) or []
                )
                # HS-4 P0.4: the server's own usage numbers for the /v1 client
                # response (None = not reported; never estimated).
                call_meta["prompt_tokens"] = getattr(result, "prompt_tokens", None)
                call_meta["cached_prompt_tokens"] = getattr(
                    result, "cached_prompt_tokens", None
                )
            if _is_frontdoor_role(role) and _frontdoor_trace_enabled():
                log.warning(
                    "Frontdoor inference telemetry: transport=%s elapsed_ms=%.1f "
                    "first_token_ms=%.1f chunks=%d completion_reason=%s "
                    "tokens=%d prompt_ms=%.1f gen_ms=%.1f overhead_ms=%.1f",
                    transport,
                    req_elapsed_ms,
                    getattr(result, "first_token_ms", 0.0),
                    getattr(result, "stream_chunks", 0),
                    getattr(result, "completion_reason", "") or "unknown",
                    result.tokens_generated,
                    result.prompt_eval_ms,
                    result.generation_ms,
                    result.http_overhead_ms,
                )

            # A context overflow is a property of the REQUEST (or of concurrent
            # load), not a sick backend: raise the typed error BEFORE the
            # circuit-breaker bookkeeping so it never counts toward opening
            # the circuit on a healthy server.
            _overflow = getattr(result, "context_overflow", None)
            if (
                isinstance(_overflow, dict)
                and _overflow.get("kind") == ContextOverflowError.POOL_EXHAUSTED
                and backend_url
            ):
                from src.scheduling.kv_pool_admission import get_shared_pool_admission

                get_shared_pool_admission().report_pool_exhausted(backend_url)
            _raise_if_context_overflow(result, role, backend_url)
            if max_tokens_clamp is not None:
                call_meta["max_tokens_clamped"] = dict(max_tokens_clamp)
                reason = str(getattr(result, "completion_reason", "") or "")
                if reason in {"length", "limit", "context_limit"} or (
                    getattr(result, "tokens_generated", 0) >= max_tokens_clamp["to"]
                ):
                    result.completion_reason = "context_limit"
                    call_meta["completion_reason"] = "context_limit"
            pool_success = True

            # Record success/failure for circuit breaker.
            # Partial results (read_timeout with salvaged output) count as
            # degraded — not a full success for health tracking, but not a
            # hard failure that should trip the circuit breaker either.
            if backend_url and self.health_tracker and not fleet_managed:
                if result.success and not result.partial:
                    self.health_tracker.record_success(backend_url)
                elif not result.success and not result.partial:
                    self.health_tracker.record_failure(backend_url)
                # partial results: skip health tracking (neither success nor failure)
            # fleet_managed: the fleet backend recorded per-endpoint health
            # for the endpoint it actually dispatched to.

            if not result.success and not result.partial:
                raise RuntimeError(f"Inference failed: {result.error_message}")

            # SINGLE SOURCE OF TRUTH for model-decode token accounting. This is the
            # backend's EXACT completion count; the primitives layer must NOT re-add a
            # char-estimate on top of it (it guards against that via a snapshot). The
            # Pareto/HV speed objective is derived from this count, so it must equal
            # MODEL-decoded tokens counted exactly once — tool output is never included
            # here (effective tool-output throughput is log-only, eval_log_format.py).
            self.total_tokens_generated += result.tokens_generated
            self.total_prompt_eval_ms += result.prompt_eval_ms
            self.total_generation_ms += result.generation_ms
            self.total_http_overhead_ms += result.http_overhead_ms
            # Server-reported prompt tokens (same provenance as the tap's
            # server_terminal count) — summed so /v1 can report measured usage
            # instead of a chars/4 estimate. Calls that reported none add 0.
            _reported_prompt = getattr(result, "prompt_tokens", None)
            if isinstance(_reported_prompt, int) and _reported_prompt > 0:
                self.total_prompt_tokens_reported = (
                    getattr(self, "total_prompt_tokens_reported", 0) + _reported_prompt
                )
            if result.predicted_per_second > 0:
                self._last_predicted_tps = result.predicted_per_second
            return result.output
        finally:
            reset_lifecycle_context(lifecycle_token)
            if pool_admission is not None:
                pool_admission.release(backend_url, pool_ticket, success=pool_success)
            if admitted and admission:
                admission.release(backend_url)

    def _real_batch(
        self,
        prompts: list[str],
        role: str,
        json_schema: dict | None = None,
        grammar: str | None = None,
        n_tokens: int | None = None,
    ) -> list[str]:
        """Make real inference calls in parallel.

        Args:
            prompts: List of prompts.
            role: The role determining which model to use.
            json_schema: Optional JSON schema constraining every prompt in
                this batch (TD-21.22a: per-batch, not per-prompt — mirrors
                how ``combined_ops._batch_llm_query`` builds one batch under
                one schema). Omitted (None) reproduces the pre-TD-21.22a
                call shape byte-for-byte.
            grammar: Optional GBNF grammar, applied to the whole batch the
                same way as json_schema.
            n_tokens: Optional max-tokens cap applied to every prompt in this
                batch (batch/n_tokens follow-up to TD-21.22a). Omitted (None)
                reproduces the pre-existing call shape.

        Returns:
            List of model responses in order.
        """
        # Use worker pool for worker roles if configured
        if self.use_worker_pool and role.startswith("worker"):
            return self._worker_pool_batch(
                prompts, role, json_schema=json_schema, grammar=grammar, n_tokens=n_tokens,
            )

        # Check if we have a backend for this role
        backend = self._backends.get(role)
        if backend is None and self.model_server is None:
            raise RuntimeError(
                f"No backend configured for role '{role}'. Provide server_urls or model_server."
            )

        role_limit = getattr(self, "_get_role_limit", lambda _r: self.config.batch_parallelism)(
            role
        )
        # Only forward the constraint kwargs when set, so a caller that omits
        # them (every caller before TD-21.22a) reaches _real_call with the
        # exact same argument list as before.
        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["json_schema"] = json_schema
        if grammar is not None:
            extra["grammar"] = grammar
        if n_tokens is not None:
            extra["n_tokens"] = n_tokens

        if role_limit <= 1:
            results = []
            for prompt in prompts:
                try:
                    results.append(self._real_call(prompt, role, **extra))
                except Exception as e:
                    results.append(f"[ERROR: {e}]")
            return results

        results: list[str | None] = [None] * len(prompts)
        max_workers = min(self.config.batch_parallelism, role_limit)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i, prompt in enumerate(prompts):
                ctx = contextvars.copy_context()
                future = executor.submit(ctx.run, self._real_call, prompt, role, **extra)
                future_to_idx[future] = i

            # Collect results in order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR: {e}]"

        return [r if r is not None else "" for r in results]

    def _worker_pool_batch(
        self,
        prompts: list[str],
        role: str,
        json_schema: dict | None = None,
        grammar: str | None = None,
        n_tokens: int | None = None,
    ) -> list[str]:
        """Execute batch using the heterogeneous worker pool.

        Routes to appropriate worker type based on role.

        Args:
            prompts: List of prompts.
            role: Worker role (determines task type routing).
            json_schema: Optional JSON schema constraining the whole batch
                (forwarded to ``WorkerPoolManager.batch``'s ``/completion``
                payload). Omitted reproduces the prior call byte-for-byte.
            grammar: Optional GBNF grammar for the whole batch.
            n_tokens: Optional max-tokens cap for the whole batch. Forwarded
                to ``WorkerPoolManager.batch`` as ``max_tokens`` (its own
                parameter name); the ``_fallback_batch``/``_real_call`` side
                keeps the ``n_tokens`` name throughout. Omitted reproduces
                the prior call byte-for-byte.

        Returns:
            List of model responses in order.
        """
        # Determine task type from role.
        # Canonical worker aliases collapse to their live task families
        # (e.g. worker_general -> explore, worker_coder -> coder, etc.).
        task_type = "explore"  # default
        if "_" in role:
            suffix = role.split("_", 1)[1]
            task_type = self.WORKER_TASK_ROUTING.get(suffix, suffix)

        # extra: kwargs for the _fallback_batch/_real_call side (n_tokens).
        # worker_pool_extra: kwargs for WorkerPoolManager.batch, which names
        # its token cap max_tokens, not n_tokens.
        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["json_schema"] = json_schema
        if grammar is not None:
            extra["grammar"] = grammar
        if n_tokens is not None:
            extra["n_tokens"] = n_tokens

        worker_pool_extra: dict[str, Any] = {}
        if json_schema is not None:
            worker_pool_extra["json_schema"] = json_schema
        if grammar is not None:
            worker_pool_extra["grammar"] = grammar
        if n_tokens is not None:
            worker_pool_extra["max_tokens"] = n_tokens

        try:
            # Run async batch in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            timeout_s = self._remaining_deadline_s()
            batch_coro = self.worker_pool.batch(prompts, task_type=task_type, **worker_pool_extra)
            if timeout_s is not None:
                timeout_s = max(1.0, timeout_s)
                batch_coro = asyncio.wait_for(batch_coro, timeout=timeout_s)
                self._budget_diagnostics["budget_applied"] = True
                self._budget_diagnostics["timeout_clamp_events"] += 1
            if loop is not None and loop.is_running():
                # If we're already in an async context, create a new task
                import nest_asyncio

                nest_asyncio.apply()
                results = loop.run_until_complete(batch_coro)
            else:
                results = asyncio.run(batch_coro)
            return results
        except Exception as e:
            # Fall back to standard batch if worker pool fails
            logging.getLogger(__name__).warning(
                f"Worker pool batch failed, falling back to standard: {e}"
            )
            return self._fallback_batch(prompts, role, **extra)

    def _fallback_batch(
        self,
        prompts: list[str],
        role: str,
        json_schema: dict | None = None,
        grammar: str | None = None,
        n_tokens: int | None = None,
    ) -> list[str]:
        """Fallback batch implementation using ThreadPoolExecutor.

        Used when worker pool is unavailable or fails.
        """
        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["json_schema"] = json_schema
        if grammar is not None:
            extra["grammar"] = grammar
        if n_tokens is not None:
            extra["n_tokens"] = n_tokens

        results: list[str | None] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.config.batch_parallelism) as executor:
            future_to_idx = {
                executor.submit(
                    contextvars.copy_context().run, self._real_call, prompt, role, **extra
                ): i
                for i, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR: {e}]"

        return [r if r is not None else "" for r in results]

    def _real_call_monitored(
        self,
        prompt: str,
        role: str,
        monitor: Any,
    ) -> LLMResult:
        """Make real inference call with monitoring.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.
            monitor: GenerationMonitor instance.

        Returns:
            LLMResult with text and abort status.

        Raises:
            RuntimeError: If model server not configured.
        """
        if self.model_server is None:
            raise RuntimeError("ModelServer not configured for real inference")

        # Reset monitor for this generation
        monitor.reset()

        # Use the model server's streaming infer method
        from src.model_server import InferenceRequest

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            timeout=self.config.call_timeout,
            stream=True,  # Enable streaming for per-token monitoring
        )
        request.timeout = self._clamp_timeout_to_request_budget(request.timeout)

        output_tokens = []
        from src.inference_lock import inference_lock

        _mon_port = _extract_port(
            (self.server_urls or {}).get(role, "") if hasattr(self, "server_urls") else ""
        )
        with inference_lock(
            role,
            cancel_check=self.get_request_cancel_check(),
            deadline_s=self.get_request_deadline_s(),
            request_tag=self.get_request_task_id(),
            port=_mon_port,
        ):
            request.timeout = self._clamp_timeout_to_request_budget(request.timeout)
            for token_id, logits in self.model_server.infer_stream(role, request):
                output_tokens.append(token_id)

                # Update monitor with real logits
                monitor.update(token_id, logits)

                # Check if we should abort
                should_abort, abort_reason = monitor.should_abort()
                if should_abort:
                    health = monitor.get_health()
                    # Decode partial output
                    partial_text = self.model_server.decode_tokens(output_tokens)
                    return LLMResult(
                        text=partial_text,
                        aborted=True,
                        abort_reason=abort_reason.value,
                        tokens_generated=len(output_tokens),
                        tokens_saved=0,  # Unknown for real inference
                        failure_probability=health.estimated_failure_prob,
                    )

        # Completed without abort
        health = monitor.get_health()
        full_text = self.model_server.decode_tokens(output_tokens)
        self.total_tokens_generated += len(output_tokens)

        return LLMResult(
            text=full_text,
            aborted=False,
            tokens_generated=len(output_tokens),
            failure_probability=health.estimated_failure_prob,
        )
