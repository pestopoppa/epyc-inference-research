#!/usr/bin/env python3
"""LlamaServerBackend - Persistent HTTP server mode for llama.cpp inference.

This backend connects to a running llama-server instance instead of spawning
per-inference subprocesses. Enables:
- KV cache reuse across requests (prefix caching)
- Lower latency (no model load per inference)
- Multi-slot parallel processing

Usage:
    from src.backends.llama_server import LlamaServerBackend

    backend = LlamaServerBackend(base_url="http://localhost:8080")
    result = backend.infer(role_config, request)

See research/radix_attention_handoff.md for implementation plan.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx

from src.backends.context_overflow import (
    ContextOverflowInfo,
    classify_error_body,
    classify_error_payload,
)
from src.model_server import InferenceRequest, InferenceResult, ModelBackend
from src.registry_loader import RoleConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def _slot_token_suffix(response: Any, key: str) -> str:
    """`` (N tokens)`` from a /slots save|restore body, or '' if unreadable.

    The count is the only thing in that response that distinguishes a real
    save/restore from a zero-token one, and this method previously discarded the
    body entirely. Deliberately total: a body that will not parse (or a test
    double that is not real JSON) yields an empty suffix and NEVER changes the
    caller's success verdict -- surfacing the number must not be able to fail a
    request that actually succeeded.
    """
    try:
        value = response.json().get(key)
        return f" ({int(value)} tokens)" if isinstance(value, (int, float)) else ""
    except Exception:
        return ""


# Logit probe output file (append-only JSONL)
_LOGIT_PROBE_PATH = str(_REPO_ROOT / "data/logit_probe.jsonl")

# Fixed RNG seed for reproducible decode. Sampling is deterministic for a given
# (prompt, params, seed); callers may override per-request via request.seed.
_DETERMINISTIC_SAMPLING_SEED = 42


def _empty_generation_failure_after_s() -> float:
    """Seconds after which an empty model response is treated as infrastructure failure."""
    try:
        return max(0.0, float(os.environ.get("LLAMA_EMPTY_GENERATION_FAILURE_AFTER_S", "30")))
    except (TypeError, ValueError):
        return 30.0


def _is_empty_long_generation(output: str, elapsed_s: float) -> bool:
    threshold = _empty_generation_failure_after_s()
    return threshold > 0 and not (output or "").strip() and elapsed_s >= threshold


def _write_logit_probe(prompt: str, first_token_probs: dict) -> None:
    """Append first-token log-probabilities to JSONL for routing classifier P1.5.

    Args:
        prompt: The input prompt (hashed, not stored raw).
        first_token_probs: First token probability data from llama.cpp completion_probabilities.
    """
    import hashlib
    import json
    from pathlib import Path

    try:
        probs = first_token_probs.get("probs", [])
        if not probs:
            return

        entry = {
            "timestamp": time.time(),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_len": len(prompt),
            "first_token": first_token_probs.get("content", ""),
            "top_k_probs": [
                {"tok": p.get("tok_str", ""), "prob": p.get("prob", 0.0)}
                for p in probs[:64]
            ],
        }

        path = Path(_LOGIT_PROBE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception:
        logger.debug("logit_probe write failed", exc_info=True)


def _chat_payload(request: Any) -> dict[str, Any] | None:
    """HS-4 P0.1: the structured /v1 client-tool-mode payload, if any."""
    payload = getattr(request, "chat_payload", None)
    return payload if isinstance(payload, dict) else None


def _thinking_chat_lane(role_config: Any, request: Any) -> bool:
    """RI-23: per-request chat-lane admission of a thinking-on role (flag-gated, live).

    False whenever ``thinking_roles_chat_lane`` is off, so flag-off routing is unchanged.
    """
    try:
        from src.chat_completions_roles import thinking_chat_lane_role

        return thinking_chat_lane_role(
            getattr(request, "role", None) or getattr(role_config, "name", None)
        )
    except Exception:
        return False


def _chat_template_kwargs(role_config: Any, request: Any) -> dict[str, Any] | None:
    """The chat_template_kwargs for one /v1/chat/completions request.

    J12: an explicit ``request.extra`` value wins; otherwise the role's registry default.
    RI-23: a per-call override (``request.chat_template_kwargs``, set by
    ``chat_completions_roles.thinking_off()``) is merged OVER that — a new dict, the
    cached registry dict is never mutated. No override = exactly the pre-RI-23 value.
    """
    ctk = request.extra.get("chat_template_kwargs") if getattr(request, "extra", None) else None
    if not ctk:
        try:
            from src.registry.registry_loader import chat_template_kwargs_for_role
            ctk = chat_template_kwargs_for_role(getattr(request, "role", None) or role_config.name)
        except Exception:
            ctk = None
    override = getattr(request, "chat_template_kwargs", None)
    if isinstance(override, dict) and override:
        ctk = {**(ctk or {}), **override}
    return ctk


def _response_text(response: Any) -> str:
    try:
        text = response.text
    except Exception:
        return ""
    return text.strip() if isinstance(text, str) else ""


def _overflow_from_error_response(response: Any) -> ContextOverflowInfo | None:
    """Classify an HTTP >= 400 llama-server response as a context overflow.

    Streaming responses have an unread body; read it first (bounded: the server
    sends a small JSON error object, server-context.cpp:4362-4365).
    """
    try:
        status = int(response.status_code)
    except Exception:
        return None
    if status < 400:
        return None
    try:
        response.read()
    except Exception:
        pass
    return classify_error_body(_response_text(response), status)


def _context_overflow_result(
    role_config: Any,
    info: ContextOverflowInfo,
    elapsed: float,
    *,
    partial_output: str = "",
) -> InferenceResult:
    """A failed InferenceResult that carries the overflow for the inference layer.

    ``output`` stays empty on purpose: a request the server aborted mid-way has
    its slot cleared (server-context.cpp:3776-3782), so partial text is not an
    answer and must never be returned as one. Its length is kept in the message.
    """
    logger.warning(
        "llama-server context overflow role=%s kind=%s status=%s n_prompt=%s n_ctx=%s "
        "partial_chars=%d: %s",
        getattr(role_config, "name", "?"), info.kind, info.http_status,
        info.n_prompt_tokens, info.n_ctx, len(partial_output), info.message,
    )
    return InferenceResult(
        role=getattr(role_config, "name", ""),
        output="",
        tokens_generated=0,
        generation_speed=0.0,
        elapsed_time=elapsed,
        success=False,
        error_message=f"context overflow ({info.kind}): {info.message}",
        partial=False,
        degraded=True,
        failure_stage="context",
        failure_reason="context_overflow",
        completion_reason="context_overflow",
        prompt_tokens=info.n_prompt_tokens,
        context_overflow=info.to_dict(),
    )


def _server_cfg():
    from src.config import get_config

    return get_config().server


@dataclass
class ServerConfig:
    """Configuration for a llama-server instance."""

    base_url: str = field(default_factory=lambda: _server_cfg().default_url)
    timeout: int = field(default_factory=lambda: _server_cfg().timeout)
    num_slots: int = field(default_factory=lambda: _server_cfg().num_slots)
    connect_timeout: int = field(default_factory=lambda: _server_cfg().connect_timeout)
    retry_count: int = field(default_factory=lambda: _server_cfg().retry_count)
    retry_backoff: float = field(default_factory=lambda: _server_cfg().retry_backoff)
    # 2026-05-23: when True, this backend routes through /v1/chat/completions
    # instead of /completion. Used for models whose GGUF chat_template emits
    # multi-channel output (e.g. gemma-4-26B-A4B-it's <|channel>thought
    # markers) that /completion can't apply server-side. The OpenAI-style
    # endpoint with --jinja handles templating + response parsing properly.
    # See backend.py:_init_caching_backends for env-driven role selection.
    use_chat_completions: bool = False


@dataclass
class SlotInfo:
    """Information about a server slot."""

    slot_id: int
    state: str  # "idle", "processing"
    prompt_tokens: int = 0
    cache_tokens: int = 0  # Tokens served from cache
    last_prompt_hash: str = ""
    last_access: float = 0.0


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_prompt_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    @property
    def token_savings_rate(self) -> float:
        """Percentage of prompt tokens served from cache."""
        if self.total_prompt_tokens == 0:
            return 0.0
        return (self.cached_prompt_tokens / self.total_prompt_tokens) * 100


class LlamaServerError(Exception):
    """Error communicating with llama-server."""

    pass


class LlamaServerBackend(ModelBackend):
    """Backend using persistent llama-server HTTP API.

    This backend connects to a running llama-server instance and uses its
    HTTP API for inference. Enables KV cache reuse for prefix caching.

    Key features:
    - cache_prompt=True enables automatic prefix caching
    - id_slot parameter enables sticky slot routing for cache hits
    - Slot state management for optimal cache utilization

    Attributes:
        config: Server configuration.
        session: HTTP session with retry logic.
        slots: Current slot information.
        cache_stats: Cache performance statistics.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        base_url: str = "http://localhost:8080",
    ):
        """Initialize the llama-server backend.

        Args:
            config: Server configuration. If None, creates default config.
            base_url: Base URL for the server (used if config is None).
        """
        if config is not None:
            self.config = config
        else:
            self.config = ServerConfig(base_url=base_url)

        # Create httpx client with connection pooling for ~6x latency reduction
        # Connection pool keeps persistent connections to reduce per-request overhead
        self.client = httpx.Client(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.timeout,
                write=self.config.timeout,
                pool=self.config.timeout,
            ),
            limits=httpx.Limits(
                max_connections=20,  # Total connections in pool
                max_keepalive_connections=10,
                keepalive_expiry=5.0,  # Short expiry as safety net
            ),
            transport=httpx.HTTPTransport(retries=self.config.retry_count),
        )

        # Slot tracking
        self.slots: dict[int, SlotInfo] = {}

        # Cache statistics
        self.cache_stats = CacheStats()

        # Server health
        self._healthy = False
        self._last_health_check = 0.0

    def load(self, role_config: RoleConfig) -> int:
        """Verify server is running and healthy.

        In server mode, models are pre-loaded. This method validates connectivity.

        Args:
            role_config: Role configuration (used for logging).

        Returns:
            0 (server mode doesn't track PIDs).

        Raises:
            LlamaServerError: If server is not reachable.
        """
        if not self.health_check(0):
            raise LlamaServerError(
                f"Cannot reach llama-server at {self.config.base_url}. "
                "Start the server with: llama-server -m MODEL.gguf --host 0.0.0.0 --port 8080"
            )

        logger.info(f"Connected to llama-server for role {role_config.name}")
        return 0

    def unload(self, pid: int) -> bool:
        """No-op in server mode (model stays loaded).

        Args:
            pid: Ignored in server mode.

        Returns:
            True (always succeeds).
        """
        return True

    def health_check(self, pid: int) -> bool:
        """Check if the llama-server is healthy.

        Args:
            pid: Ignored in server mode.

        Returns:
            True if server is reachable and healthy.
        """
        # Rate limit health checks
        now = time.time()
        if now - self._last_health_check < 1.0 and self._healthy:
            return self._healthy

        try:
            response = self.client.get(
                "/health",
                timeout=self.config.connect_timeout,
            )
            self._healthy = response.status_code == 200
            self._last_health_check = now
            return self._healthy
        except httpx.RequestError:
            self._healthy = False
            return False

    def infer(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference via llama-server HTTP API.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.

        Returns:
            InferenceResult with output and metrics.
        """
        start_time = time.time()
        self.cache_stats.total_requests += 1

        # 2026-05-23: when use_chat_completions is set on this backend's
        # ServerConfig, route through /v1/chat/completions (server-side
        # jinja template applied by --jinja flag). Otherwise legacy
        # /completion path.
        # HS-4 P0.1: a structured chat payload (the /v1 client-executed tool
        # mode) can only be expressed on /v1/chat/completions, whatever this
        # backend's default endpoint is.
        # RI-23: thinking-on roles join per request while thinking_roles_chat_lane is on.
        if (
            self.config.use_chat_completions
            or _chat_payload(request) is not None
            or _thinking_chat_lane(role_config, request)
        ):
            return self._infer_chat_completions(role_config, request, start_time)

        # Build request payload
        payload = self._build_payload(role_config, request)

        try:
            http_start = time.perf_counter()
            _overall = request.timeout or self.config.timeout
            _batch_timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                # Non-streaming: zero bytes arrive until generation completes,
                # so the read timeout MUST cover the whole request budget — a
                # 120s cap killed every >120s generation (EV-BASELINE-E7:
                # 11/12 timeouts at 119-120s on reasoning/long-context suites)
                # while the eval budget was 420s. Interactive stays safe: its
                # OVERALL budget is short (60s SLA), and read <= overall.
                read=_overall,
                write=_overall,
                pool=30,
            )
            _prompt_len = len(request.prompt or "")
            logger.warning(
                "llama POST /completion start role=%s prompt_chars=%d n_predict=%d timeout=%.0f",
                role_config.name, _prompt_len, request.n_tokens, _overall,
            )
            response = self.client.post(
                "/completion",
                json=payload,
                timeout=_batch_timeout,
            )
            http_elapsed_ms = (time.perf_counter() - http_start) * 1000
            logger.warning(
                "llama POST /completion done role=%s elapsed_ms=%.0f status=%d",
                role_config.name, http_elapsed_ms, response.status_code,
            )
            overflow = _overflow_from_error_response(response)
            if overflow is not None:
                return _context_overflow_result(role_config, overflow, time.time() - start_time)
            response.raise_for_status()
            result_data = response.json()

            elapsed = time.time() - start_time

            # Extract metrics
            output = result_data.get("content", "")
            tokens_generated = result_data.get("tokens_predicted", 0)
            prompt_tokens = result_data.get("tokens_evaluated", 0)
            completion_reason = str(
                result_data.get("stop_type")
                or result_data.get("finish_reason")
                or ("stop" if result_data.get("stop") else "")
            )
            if result_data.get("truncated") is True:
                # server-context.cpp:1929-1936: with context shift disabled the
                # server stops generation when the slot's n_ctx is full and
                # marks the result truncated. Never let that pass as a clean
                # stop — the answer was cut by the context limit.
                logger.warning(
                    "llama /completion truncated by context limit role=%s "
                    "tokens_evaluated=%s tokens_predicted=%s",
                    role_config.name, result_data.get("tokens_evaluated"),
                    result_data.get("tokens_predicted"),
                )
                completion_reason = "context_limit"

            # Extract clean timing data from llama.cpp timings object
            timings = result_data.get("timings", {})
            prompt_eval_ms = timings.get("prompt_ms", 0.0)
            generation_ms = timings.get("predicted_ms", 0.0)
            predicted_per_second = timings.get("predicted_per_second", 0.0)

            # Cache-hit count: llama.cpp's `timings.cache_n` is the true
            # number of prompt tokens reused from the KV cache (server-task
            # result_timings.cache_n = n_prompt_tokens_cache). The legacy
            # `tokens_cached` response field is the TOTAL tokens held in the
            # slot after the request (slot.prompt.n_tokens()), which is
            # non-zero for essentially every non-empty prompt — counting it
            # as a hit inflated cache_hits on nearly every request (RTE-Prefix
            # measurement defect, fixed 2026-08-13).
            cached_tokens = timings.get("cache_n", 0)

            # Compute server-side overhead (HTTP round-trip minus reported inference)
            inference_ms = prompt_eval_ms + generation_ms
            http_overhead_ms = max(0.0, http_elapsed_ms - inference_ms)

            # Log overhead when significant (> 5s)
            if http_overhead_ms > 5000:
                logger.warning(
                    f"Server overhead: {http_overhead_ms:.0f}ms "
                    f"(HTTP={http_elapsed_ms:.0f}ms, inference={inference_ms:.0f}ms) "
                    f"for {role_config.name}"
                )

            # Update cache stats
            self.cache_stats.total_prompt_tokens += prompt_tokens
            self.cache_stats.cached_prompt_tokens += cached_tokens
            if cached_tokens > 0:
                self.cache_stats.cache_hits += 1
            else:
                self.cache_stats.cache_misses += 1

            # Use clean predicted_per_second if available, else fall back to elapsed
            speed = (
                predicted_per_second
                if predicted_per_second > 0
                else (tokens_generated / elapsed if elapsed > 0 else 0.0)
            )

            # Log cache performance
            if cached_tokens > 0:
                logger.debug(
                    f"Cache hit: {cached_tokens}/{prompt_tokens} tokens "
                    f"({100 * cached_tokens / prompt_tokens:.1f}%) from cache"
                )

            # Extract speculative decoding acceptance telemetry
            n_drafted = timings.get("draft_n", 0)
            n_accepted = timings.get("draft_n_accepted", 0)
            accept_rate = (n_accepted / n_drafted) if n_drafted > 0 else 0.0
            if n_drafted > 0:
                logger.info(
                    "Spec accept: %d/%d (%.1f%%) for %s",
                    n_accepted, n_drafted, accept_rate * 100, role_config.name,
                )

            # Logit probe: capture first-token probabilities for routing classifier
            from src.features import features
            if features().logit_probe and role_config.name == "frontdoor":
                completion_probs = result_data.get("completion_probabilities", [])
                if completion_probs:
                    _write_logit_probe(
                        prompt=request.prompt or "",
                        first_token_probs=completion_probs[0] if completion_probs else {},
                    )

            empty_generation = _is_empty_long_generation(output, elapsed)
            if empty_generation:
                logger.warning(
                    "Empty llama response after %.1fs for %s "
                    "(tokens=%s completion_reason=%s)",
                    elapsed,
                    role_config.name,
                    tokens_generated,
                    completion_reason or "unknown",
                )

            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens_generated,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=not empty_generation,
                error_message=(
                    f"Empty generation after {elapsed:.1f}s"
                    if empty_generation else None
                ),
                partial=False,
                degraded=empty_generation,
                failure_stage="generation" if empty_generation else "",
                failure_reason="empty_generation" if empty_generation else "",
                prompt_eval_ms=prompt_eval_ms,
                generation_ms=generation_ms,
                predicted_per_second=predicted_per_second,
                http_overhead_ms=http_overhead_ms,
                n_tokens_drafted=n_drafted,
                n_tokens_accepted=n_accepted,
                acceptance_rate=accept_rate,
                completion_reason=(
                    "empty_generation" if empty_generation else completion_reason
                ),
                completion_probabilities=list(result_data.get("completion_probabilities") or []),
                prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
            )

        except httpx.TimeoutException:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=request.timeout or self.config.timeout,
                success=False,
                error_message=f"Request timed out after {request.timeout}s",
                failure_stage="request",
                failure_reason="timeout",
                completion_reason="timeout",
            )

        except httpx.HTTPStatusError as e:
            # A 4xx/5xx from the backend (e.g. worker_vision:8086 returning
            # HTTP 400 to a misrouted text /completion) is raised by
            # response.raise_for_status(). HTTPStatusError is a SIBLING of
            # RequestError, not a subclass, so without this clause it escaped
            # infer() uncaught and surfaced to the caller as a raw in-band
            # "[ERROR: ...]" answer string that the eval then mis-scored.
            # Convert it to a structured degraded result, matching the
            # /v1/chat/completions path (failure_stage/reason: transport/http_status).
            elapsed = time.time() - start_time
            overflow = _overflow_from_error_response(e.response)
            if overflow is not None:
                return _context_overflow_result(role_config, overflow, elapsed)
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=elapsed,
                success=False,
                error_message=f"llama-server HTTP {e.response.status_code}",
                failure_stage="transport",
                failure_reason="http_status",
                completion_reason="http_error",
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=elapsed,
                success=False,
                error_message=f"Server request failed: {e}",
                failure_stage="transport",
                failure_reason="request_error",
                completion_reason="request_error",
            )

    def _infer_chat_completions(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
        start_time: float,
    ) -> InferenceResult:
        """Run inference via the OpenAI-compatible /v1/chat/completions endpoint.

        Used for models whose chat template needs server-side jinja
        application (gemma-4-26B-A4B-it's multi-channel format, etc.).
        Builds a single-user-turn messages list from request.prompt and
        lets llama-server's --jinja flag handle templating + response
        parsing.

        Caller's prompt is treated as PLAIN user content — if the caller
        pre-templated, the inner-template markers will appear as literal
        user content and likely confuse the model. The chat.py path
        knows to skip pre-templating when this backend mode is in use
        (see _per_region_locks_enabled + chat_completion_roles env).
        """
        # Strip any prior chat-template wrapping the orchestrator might
        # have applied — we want the model to see only the user's actual
        # question. This is defensive; chat.py should already be passing
        # raw prompts for chat_completion roles.
        user_content = request.prompt or ""
        for marker in ("<|im_start|>user\n", "<|im_start|>user "):
            if user_content.startswith(marker):
                user_content = user_content[len(marker):]
                # also strip the closing markers
                for closer in ("<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>"):
                    idx = user_content.rfind(closer)
                    if idx > 0:
                        user_content = user_content[:idx]
                break
        if user_content.startswith("<start_of_turn>user\n"):
            user_content = user_content[len("<start_of_turn>user\n"):]
            for closer in ("<end_of_turn>\n<start_of_turn>model\n", "<end_of_turn>"):
                idx = user_content.rfind(closer)
                if idx > 0:
                    user_content = user_content[:idx]
        user_content = user_content.strip()

        chat_payload = _chat_payload(request)
        if chat_payload is not None:
            # HS-4 P0.1 client-executed tool mode: forward the caller's
            # structured history (assistant tool_calls + tool results) and its
            # tool definitions verbatim; llama-server --jinja templates them
            # and parses the model's tool calls back out.
            messages = [dict(m) for m in chat_payload.get("messages") or []]
            user_content = json.dumps(messages, sort_keys=True)
        else:
            messages = [{"role": "user", "content": user_content}]

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": request.n_tokens if request.n_tokens > 0 else 4096,
            "stream": False,
        }
        if chat_payload is not None:
            if chat_payload.get("tools") is not None:
                payload["tools"] = chat_payload["tools"]
            if chat_payload.get("tool_choice") is not None:
                payload["tool_choice"] = chat_payload["tool_choice"]
        self._apply_deterministic_sampling(payload, role_config, request)
        # Chat-path Option A (2026-07-22): llama's OpenAI-compat endpoint
        # IGNORES the native n_probs param the sampling applier just injected —
        # translate to logprobs/top_logprobs, else every chat_completions role
        # (frontdoor Qwen, gemma4 worker fleet) returns empty
        # completion_probabilities and calibration silently falls back to the
        # binary proxy (EV-4b/EV-11c confidence-void root cause).
        _n_probs = payload.pop("n_probs", None)
        if not _n_probs:
            _req_probs = getattr(request, "n_probs", None)
            _n_probs = int(_req_probs) if _req_probs else None
        if _n_probs and _n_probs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = max(1, min(int(_n_probs), 20))
            # TD-1d.2: post_sampling_probs=false (server default) reports the
            # PRE-sampling raw-vocab top-k (tools/server/server-context.cpp
            # populate_token_probs, post_sampling=false branch reads
            # get_token_probabilities BEFORE the grammar mask is applied),
            # so a grammar-narrowed position's declared candidates are often
            # absent from it even though the grammar-forced sampled token IS
            # one of them. post_sampling_probs=true switches to
            # common_sampler_get_candidates(..., post_sampling=true) — the
            # candidates AFTER the full sampler chain (grammar included) —
            # guaranteeing every still-legal candidate is what populates the
            # top-k. Same server-side field name on both endpoints; the
            # v9 /completion payload sets it below in _build_payload.
            if getattr(request, "post_sampling_probs", False):
                payload["post_sampling_probs"] = True
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        self._apply_schema_constraint(payload, request)

        # J12: per-role chat-template kwargs (e.g. enable_thinking=False for
        # Qwen3.6 frontdoor / Qwen3.5 architect — load-bearing per
        # feedback_qwen3x_enable_thinking_false). Explicit request override wins;
        # otherwise fall back to the role's registry default. Only meaningful on
        # this /v1/chat/completions path (the GGUF jinja template applies the kwarg).
        ctk = _chat_template_kwargs(role_config, request)
        if ctk:
            payload["chat_template_kwargs"] = ctk

        try:
            http_start = time.perf_counter()
            _overall = request.timeout or self.config.timeout
            _batch_timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                # Non-streaming: zero bytes arrive until generation completes,
                # so the read timeout MUST cover the whole request budget — a
                # 120s cap killed every >120s generation (EV-BASELINE-E7:
                # 11/12 timeouts at 119-120s on reasoning/long-context suites)
                # while the eval budget was 420s. Interactive stays safe: its
                # OVERALL budget is short (60s SLA), and read <= overall.
                read=_overall,
                write=_overall,
                pool=30,
            )
            logger.warning(
                "llama POST /v1/chat/completions start role=%s prompt_chars=%d max_tokens=%d timeout=%.0f",
                role_config.name, len(user_content), payload["max_tokens"], _overall,
            )
            response = self.client.post(
                "/v1/chat/completions",
                json=payload,
                timeout=_batch_timeout,
            )
            http_elapsed_ms = (time.perf_counter() - http_start) * 1000
            logger.warning(
                "llama POST /v1/chat/completions done role=%s elapsed_ms=%.0f status=%d",
                role_config.name, http_elapsed_ms, response.status_code,
            )
            overflow = _overflow_from_error_response(response)
            if overflow is not None:
                return _context_overflow_result(role_config, overflow, time.time() - start_time)
            response.raise_for_status()
            data = response.json()

            elapsed = time.time() - start_time

            choices = data.get("choices", [])
            output = ""
            completion_reason = "stop"
            chat_logprob_rows: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            reasoning_content: str | None = None
            if choices:
                msg = choices[0].get("message", {})
                output = msg.get("content", "") or ""
                # RI-23: llama-server's reasoning parser splits thinking out of content.
                _rc = msg.get("reasoning_content")
                if isinstance(_rc, str) and _rc:
                    reasoning_content = _rc
                raw_tool_calls = msg.get("tool_calls") if chat_payload is not None else None
                if isinstance(raw_tool_calls, list):
                    tool_calls = [tc for tc in raw_tool_calls if isinstance(tc, dict)]
                completion_reason = str(choices[0].get("finish_reason") or "stop")
                # OpenAI-shape logprobs.content rows carry a top-level
                # "logprob" per token — the confidence extractor
                # (_completion_probabilities_confidence) consumes them as-is.
                _lp = choices[0].get("logprobs")
                if isinstance(_lp, dict) and isinstance(_lp.get("content"), list):
                    chat_logprob_rows = [
                        row for row in _lp["content"] if isinstance(row, dict)
                    ]

            usage = data.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            tokens_generated = int(usage.get("completion_tokens", 0))
            # KV-reuse count: usage.prompt_tokens_details.cached_tokens (the
            # server's n_prompt_tokens_cache), else timings.cache_n — the same
            # quantity on both fields; None when the server reported neither.
            _details = usage.get("prompt_tokens_details")
            _cached_raw = _details.get("cached_tokens") if isinstance(_details, dict) else None
            if _cached_raw is None:
                _cached_raw = (data.get("timings") or {}).get("cache_n")
            cached_prompt_tokens = (
                int(_cached_raw)
                if isinstance(_cached_raw, (int, float)) and not isinstance(_cached_raw, bool)
                else None
            )
            if completion_reason == "length" and 0 < tokens_generated < int(payload["max_tokens"]):
                # The OpenAI shim has no `truncated` field, but a length stop
                # BELOW max_tokens can only be the slot's n_ctx running out
                # (ctx shift off → stop at n_ctx - 1, server-context.cpp:1929-1936).
                # Measured: split KV -np 2 -c 4096 → 271 + 1777 = 2048 = n_ctx/np.
                logger.warning(
                    "chat_completions truncated by context limit role=%s prompt=%d completion=%d max_tokens=%s",
                    role_config.name, prompt_tokens, tokens_generated, payload["max_tokens"],
                )
                completion_reason = "context_limit"

            # llama-server's OpenAI shim doesn't always emit timings; estimate
            timings = data.get("timings", {}) or {}
            prompt_eval_ms = float(timings.get("prompt_ms", 0.0))
            generation_ms = float(timings.get("predicted_ms", 0.0))
            predicted_per_second = float(timings.get("predicted_per_second", 0.0))
            inference_ms = prompt_eval_ms + generation_ms
            http_overhead_ms = max(0.0, http_elapsed_ms - inference_ms)

            self.cache_stats.total_prompt_tokens += prompt_tokens
            speed = (
                predicted_per_second if predicted_per_second > 0
                else (tokens_generated / elapsed if elapsed > 0 else 0.0)
            )

            # A tool-call-only turn legitimately has empty content.
            empty_generation = not tool_calls and _is_empty_long_generation(output, elapsed)
            if empty_generation:
                logger.warning(
                    "Empty chat_completions response after %.1fs for %s "
                    "(tokens=%s completion_reason=%s)",
                    elapsed,
                    role_config.name,
                    tokens_generated,
                    completion_reason or "unknown",
                )

            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens_generated,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=not empty_generation,
                error_message=(
                    f"Empty generation after {elapsed:.1f}s"
                    if empty_generation else None
                ),
                partial=False,
                degraded=empty_generation,
                failure_stage="generation" if empty_generation else "",
                failure_reason="empty_generation" if empty_generation else "",
                prompt_eval_ms=prompt_eval_ms,
                generation_ms=generation_ms,
                predicted_per_second=predicted_per_second,
                http_overhead_ms=http_overhead_ms,
                completion_reason=(
                    "empty_generation" if empty_generation else completion_reason
                ),
                completion_probabilities=chat_logprob_rows,
                tool_calls=tool_calls,
                prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
                cached_prompt_tokens=cached_prompt_tokens,
                reasoning_content=reasoning_content,
            )
        except httpx.HTTPStatusError as e:
            elapsed = time.time() - start_time
            overflow = _overflow_from_error_response(e.response)
            if overflow is not None:
                return _context_overflow_result(role_config, overflow, elapsed)
            return InferenceResult(
                role=role_config.name, output="", tokens_generated=0,
                generation_speed=0.0, elapsed_time=elapsed, success=False,
                # Keep the server's reason (e.g. llama-server without --jinja
                # rejecting `tools`), not just the bare status code.
                error_message=(
                    f"chat_completions HTTP {e.response.status_code}"
                    + (f": {_response_text(e.response)[:200]}" if _response_text(e.response) else "")
                ),
                failure_stage="transport", failure_reason="http_status",
                completion_reason="http_error",
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name, output="", tokens_generated=0,
                generation_speed=0.0, elapsed_time=elapsed, success=False,
                error_message=f"chat_completions failed: {e}",
                failure_stage="transport", failure_reason="request_error",
                completion_reason="request_error",
            )

    def infer_stream(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> Iterator[tuple[int, list[float] | None]]:
        """Stream inference tokens for monitoring support.

        Yields token IDs and optional logits for each generated token.
        Enables integration with GenerationMonitor for early abort.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters (must have stream=True).

        Yields:
            Tuple of (token_id, logits) for each token.
            logits may be None if not requested.
        """
        payload = self._build_payload(role_config, request)
        payload["stream"] = True

        try:
            _overall = request.timeout or self.config.timeout
            _stream_timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                # Non-streaming: zero bytes arrive until generation completes,
                # so the read timeout MUST cover the whole request budget — a
                # 120s cap killed every >120s generation (EV-BASELINE-E7:
                # 11/12 timeouts at 119-120s on reasoning/long-context suites)
                # while the eval budget was 420s. Interactive stays safe: its
                # OVERALL budget is short (60s SLA), and read <= overall.
                read=_overall,
                write=_overall,
                pool=30,
            )
            with self.client.stream(
                "POST",
                "/completion",
                json=payload,
                timeout=_stream_timeout,
            ) as response:
                response.raise_for_status()

                # Parse streaming response (SSE format)
                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = line if isinstance(line, str) else line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue

                    import json

                    try:
                        data = json.loads(line_str[6:])
                    except json.JSONDecodeError:
                        continue

                    # Extract token info
                    if "tokens" in data:
                        for token_id in data["tokens"]:
                            yield (token_id, None)
                    elif "content" in data:
                        # Single token mode - no direct ID available
                        # This is a limitation - ideally server returns token IDs
                        yield (0, None)

                    # Check for completion
                    if data.get("stop", False):
                        break

        except httpx.RequestError as e:
            logger.error(f"Stream error: {e}")
            return

    def infer_stream_text(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
        on_chunk: Any | None = None,
    ) -> InferenceResult:
        """Stream inference and return text result (with optional per-chunk callback).

        Like ``infer()`` but uses SSE streaming so that an ``on_chunk``
        callback can observe tokens as they arrive (e.g. for the
        inference tap).  The return value is identical to ``infer()``.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.
            on_chunk: Optional callback ``(str) -> None`` called for each
                text chunk received from the server.

        Returns:
            InferenceResult with output and metrics (same shape as batch).
        """
        import json as _json

        # HS-4 P0.1: tool calls only arrive whole on the batch response, so a
        # structured chat payload is never streamed (the primitives layer
        # already forces batch; this is the backstop).
        if _chat_payload(request) is not None:
            return self.infer(role_config, request)

        start_time = time.time()
        self.cache_stats.total_requests += 1

        # 2026-05-23: chat-completions streaming path. The orchestrator's
        # inference tap consumes per-token text chunks via on_chunk; the
        # OpenAI streaming format emits the same kind of incremental
        # content deltas, so we can route through /v1/chat/completions
        # without changing the on_chunk contract.
        if self.config.use_chat_completions or _thinking_chat_lane(role_config, request):
            return self._infer_stream_text_chat_completions(
                role_config, request, on_chunk, start_time,
            )

        payload = self._build_payload(role_config, request)
        payload["stream"] = True

        try:
            http_start = time.perf_counter()

            _overall_timeout = request.timeout or self.config.timeout
            # Read timeout MUST cover the whole request budget. This is the 5th
            # sibling of the four /completion + /v1/chat/completions read caps
            # lifted in c12484fb; it was missed there. A min(_overall, 120) cap
            # killed every >120s generation on /completion-streaming roles
            # (e.g. worker_vision, worker_explore) while the eval budget was
            # 420s — the same EV-BASELINE-E7 119-120s timeout signature. Under
            # 4-wide shared-bandwidth eval fan-out the server can withhold the
            # first SSE byte past 120s (slot busy / prompt eval), so a 120s read
            # cap fires even though the request legitimately has a longer budget.
            # Interactive stays safe: its OVERALL budget is the short role SLA,
            # and read <= overall by construction.
            _read_timeout = _overall_timeout
            _stream_timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                read=_read_timeout,
                write=_overall_timeout,
                pool=_overall_timeout,
            )
            chunks: list[str] = []  # outside with-block so ReadTimeout handler can access
            _prompt_len = len(request.prompt or "")
            logger.info(
                "llama STREAM /completion start role=%s prompt_chars=%d n_predict=%d timeout=%.0f id_slot=%s",
                role_config.name, _prompt_len, request.n_tokens, _overall_timeout,
                payload.get("id_slot", "NONE"),
            )

            with self.client.stream(
                "POST",
                "/completion",
                json=payload,
                timeout=_stream_timeout,
            ) as response:
                # The FIRST error of a stream is sent as a plain (non-SSE)
                # error response with its own status (server-context.cpp:4506-4519).
                overflow = _overflow_from_error_response(response)
                if overflow is not None:
                    return _context_overflow_result(role_config, overflow, time.time() - start_time)
                response.raise_for_status()

                timings: dict[str, Any] = {}
                tokens_generated = 0
                prompt_tokens = 0
                cached_tokens = 0
                first_token_ms = 0.0
                stream_chunks = 0
                completion_reason = ""

                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line if isinstance(line, str) else line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue

                    try:
                        data = _json.loads(line_str[6:])
                    except _json.JSONDecodeError:
                        continue

                    if isinstance(data, dict) and "error" in data:
                        # A LATER error arrives in-band as `data: {"error": ...}`
                        # (server-context.cpp:4540-4550) — e.g. every in-flight
                        # request when a unified KV pool runs dry. Ignoring it
                        # used to return the partial text (or "") as a success.
                        overflow = classify_error_payload(data)
                        if overflow is not None:
                            return _context_overflow_result(
                                role_config, overflow, time.time() - start_time,
                                partial_output="".join(chunks),
                            )
                        err = data.get("error")
                        err_msg = err.get("message") if isinstance(err, dict) else str(err)
                        return InferenceResult(
                            role=role_config.name,
                            output="",
                            tokens_generated=0,
                            generation_speed=0.0,
                            elapsed_time=time.time() - start_time,
                            success=False,
                            error_message=f"llama-server stream error: {err_msg}",
                            degraded=True,
                            failure_stage="stream",
                            failure_reason="server_error",
                            completion_reason="server_error",
                        )

                    content = data.get("content", "")
                    if content:
                        if first_token_ms <= 0.0:
                            first_token_ms = (time.perf_counter() - http_start) * 1000
                        stream_chunks += 1
                        chunks.append(content)
                    early_stopped = False
                    if content and on_chunk is not None:
                        try:
                            on_chunk(content)
                        except StopIteration:
                            early_stopped = True
                    if early_stopped:
                        tokens_generated = len(chunks)  # approximate
                        _es_elapsed = (time.perf_counter() - http_start) * 1000
                        timings = {
                            "predicted_ms": _es_elapsed,
                            "predicted_per_second": (
                                tokens_generated / (_es_elapsed / 1000)
                                if _es_elapsed > 0
                                else 0.0
                            ),
                        }
                        break

                    if data.get("stop", False):
                        timings = data.get("timings", {})
                        tokens_generated = data.get("tokens_predicted", 0)
                        prompt_tokens = data.get("tokens_evaluated", 0)
                        # cache_n = true KV-reuse hit count (see non-streaming
                        # path for the tokens_cached-vs-cache_n defect history).
                        cached_tokens = timings.get("cache_n", 0)
                        completion_reason = str(
                            data.get("stop_type")
                            or data.get("finish_reason")
                            or "stop"
                        )
                        if data.get("truncated") is True:
                            logger.warning(
                                "llama STREAM /completion truncated by context limit role=%s "
                                "tokens_evaluated=%s tokens_predicted=%s",
                                role_config.name, prompt_tokens, tokens_generated,
                            )
                            completion_reason = "context_limit"
                        break

            http_elapsed_ms = (time.perf_counter() - http_start) * 1000
            elapsed = time.time() - start_time

            prompt_eval_ms = timings.get("prompt_ms", 0.0)
            generation_ms = timings.get("predicted_ms", 0.0)
            predicted_per_second = timings.get("predicted_per_second", 0.0)

            # TIMINGS-zero fallback: llama-server occasionally emits a stop
            # event with `tokens_predicted=0` and an empty `timings` dict —
            # most commonly for thinking-mode responses that exhaust their
            # budget inside a `<think>` block, slot-kill / OOM aborts, and
            # certain spec-decode edge cases. The tap's per-chunk writer has
            # already streamed real content to disk by then, so reporting
            # "TIMINGS: 0 tokens in 0.00s" is misleading. Fall back to chunk
            # count + http elapsed when content exists but the server's
            # numbers are zero — preserves operator signal in the tap.
            if stream_chunks > 0:
                if tokens_generated == 0:
                    tokens_generated = stream_chunks
                if generation_ms == 0.0:
                    generation_ms = http_elapsed_ms
                if predicted_per_second == 0.0 and generation_ms > 0.0:
                    predicted_per_second = tokens_generated / (generation_ms / 1000.0)

            inference_ms = prompt_eval_ms + generation_ms
            http_overhead_ms = max(0.0, http_elapsed_ms - inference_ms)

            # Update cache stats
            self.cache_stats.total_prompt_tokens += prompt_tokens
            self.cache_stats.cached_prompt_tokens += cached_tokens
            if cached_tokens > 0:
                self.cache_stats.cache_hits += 1
            else:
                self.cache_stats.cache_misses += 1

            speed = (
                predicted_per_second
                if predicted_per_second > 0
                else (tokens_generated / elapsed if elapsed > 0 else 0.0)
            )

            # Extract speculative decoding acceptance telemetry (streaming)
            n_drafted = timings.get("draft_n", 0)
            n_accepted = timings.get("draft_n_accepted", 0)
            accept_rate = (n_accepted / n_drafted) if n_drafted > 0 else 0.0
            if n_drafted > 0:
                logger.info(
                    "Spec accept (stream): %d/%d (%.1f%%) for %s",
                    n_accepted, n_drafted, accept_rate * 100, role_config.name,
                )

            output = "".join(chunks)
            empty_generation = _is_empty_long_generation(output, elapsed)
            if empty_generation:
                logger.warning(
                    "Empty llama stream after %.1fs for %s "
                    "(tokens=%s completion_reason=%s)",
                    elapsed,
                    role_config.name,
                    tokens_generated,
                    completion_reason or "unknown",
                )

            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens_generated,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=not empty_generation,
                error_message=(
                    f"Empty generation after {elapsed:.1f}s"
                    if empty_generation else None
                ),
                partial=False,
                degraded=empty_generation,
                failure_stage="generation" if empty_generation else "",
                failure_reason="empty_generation" if empty_generation else "",
                prompt_eval_ms=prompt_eval_ms,
                generation_ms=generation_ms,
                predicted_per_second=predicted_per_second,
                http_overhead_ms=http_overhead_ms,
                n_tokens_drafted=n_drafted,
                n_tokens_accepted=n_accepted,
                acceptance_rate=accept_rate,
                first_token_ms=first_token_ms,
                stream_chunks=stream_chunks,
                completion_reason=(
                    "empty_generation" if empty_generation else completion_reason
                ),
                prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
            )

        except httpx.HTTPStatusError as e:
            # Was uncaught here (HTTPStatusError is not a RequestError), so a
            # non-overflow 4xx/5xx escaped infer_stream_text as a raw exception.
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=elapsed,
                success=False,
                error_message=(
                    f"llama-server HTTP {e.response.status_code}"
                    + (f": {_response_text(e.response)[:200]}" if _response_text(e.response) else "")
                ),
                failure_stage="transport",
                failure_reason="http_status",
                completion_reason="http_error",
            )

        except httpx.ReadTimeout:
            # Read timeout during streaming — server stopped sending SSE
            # events (slot finished without [DONE], or prompt eval exceeded
            # the per-read timeout).  Return partial content if available.
            _elapsed = time.time() - start_time
            _partial = "".join(chunks) if chunks else ""
            if _partial:
                logger.warning(
                    "Stream read timeout after %.1fs with %d chunks for %s; "
                    "returning partial content",
                    _elapsed, len(chunks), role_config.name,
                )
                # Populate the timing fields the tap writer reads so the
                # TIMINGS line reflects what actually happened during the
                # partial stream instead of falling back to dataclass zeros.
                _partial_tps = (len(chunks) / _elapsed) if _elapsed > 0 else 0.0
                return InferenceResult(
                    role=role_config.name,
                    output=_partial,
                    tokens_generated=len(chunks),
                    generation_speed=_partial_tps,
                    elapsed_time=_elapsed,
                    success=False,
                    partial=True,
                    degraded=True,
                    failure_stage="stream_read",
                    failure_reason="read_timeout",
                    completion_reason="read_timeout_partial",
                    generation_ms=_elapsed * 1000.0,
                    predicted_per_second=_partial_tps,
                )
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=_elapsed,
                success=False,
                error_message=f"Stream read timed out after {_read_timeout}s with no content",
                failure_stage="stream_read",
                failure_reason="timeout",
                completion_reason="timeout",
            )

        except httpx.TimeoutException:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=request.timeout or self.config.timeout,
                success=False,
                error_message=f"Request timed out after {request.timeout}s",
                failure_stage="request",
                failure_reason="timeout",
                completion_reason="timeout",
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=elapsed,
                success=False,
                error_message=f"Server request failed: {e}",
                failure_stage="transport",
                failure_reason="request_error",
                completion_reason="request_error",
            )

    def _apply_deterministic_sampling(
        self,
        payload: dict[str, Any],
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> None:
        """Pin sampling identically across /completion and /v1/chat/completions.

        Temperature precedence: explicit request override -> role acceleration
        override -> role generation_defaults (the registry's per-role intent,
        e.g. 0.1-0.3) -> greedy fallback. A fixed seed makes decode
        reproducible for a given (prompt, params); callers may override via
        request.seed. top_k/top_p/repeat_penalty are pinned to identical values
        on both endpoints unless the request explicitly overrides top_k/top_p.
        """
        temp = request.temperature
        if temp is None:
            temp = role_config.acceleration.temperature
        if temp is None and role_config.generation_defaults is not None:
            temp = role_config.generation_defaults.temperature
        if temp is None:
            temp = 0.0
        if temp is not None and temp >= 0:
            payload["temperature"] = temp
        payload["top_k"] = request.top_k if request.top_k is not None else 40
        payload["top_p"] = request.top_p if request.top_p is not None else 0.95
        payload["repeat_penalty"] = 1.1
        seed = getattr(request, "seed", None)
        payload["seed"] = seed if isinstance(seed, int) else _DETERMINISTIC_SAMPLING_SEED

    def _apply_schema_constraint(
        self,
        payload: dict[str, Any],
        request: InferenceRequest,
    ) -> None:
        """Forward json_schema/grammar onto the /v1/chat/completions wire (TD-21.0).

        Before this method existed, ``_infer_chat_completions`` and
        ``_infer_stream_text_chat_completions`` built their payloads with no
        schema field at all, so every /v1-lane role (frontdoor, worker*,
        toolrunner, architect_critic) silently ignored a caller's
        ``json_schema``/``grammar`` even though ``_build_payload`` (the native
        /completion lane, below) has forwarded both since HS-4. Confirmed
        against the frozen production tree
        (tools/server/server-common.cpp:939-953,1146-1154 +
        tools/server/server-schema.cpp:258-288): llama-server's OAI-compat
        chat endpoint accepts a JSON-schema constraint only via the OpenAI
        ``response_format`` envelope (a raw top-level ``json_schema`` key is
        also read, but is redundant with response_format and not needed
        here); a raw top-level ``grammar`` string is copied through
        unchanged to the same GBNF-constrained sampling path used by
        /completion.

        Mirrors ``_build_payload``: json_schema and grammar are forwarded
        independently, with no client-side precedence logic — if a caller
        sets both, both land on the wire and the server decides. Two
        relevant server behaviors observed in the frozen tree: (1) if a
        request carries a raw top-level ``json_schema`` key *and* a raw
        top-level ``grammar`` key, ``oaicompat_chat_params_parse`` throws
        "Cannot use both json_schema and grammar" (server-common.cpp:935-937)
        — not triggered here since we always wrap json_schema in
        response_format, never as a raw top-level key; (2) with
        ``response_format`` + ``grammar`` both present, that specific guard
        does not fire, and downstream chat-format handlers resolve the two
        inconsistently across model families (some let the schema-derived
        grammar win, e.g. common/chat.cpp:2844-2847; others pass the raw
        grammar straight through regardless of schema, e.g.
        common/chat.cpp:2667) — precedence in that dual-set case is
        model-format-dependent on the server side, not something this
        client layer can normalize.
        """
        if request.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.json_schema},
            }
        if request.grammar:
            payload["grammar"] = request.grammar

    def _build_payload(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        """Build the JSON payload for the completion endpoint.

        Args:
            role_config: Role configuration with acceleration settings.
            request: Inference request parameters.

        Returns:
            Dictionary payload for POST request.
        """
        # Honor per-request cache_prompt override; default to True
        cache_prompt = request.cache_prompt if request.cache_prompt is not None else True
        payload: dict[str, Any] = {
            "prompt": request.prompt or "",
            "n_predict": request.n_tokens,
            "cache_prompt": cache_prompt,
        }

        # Deterministic sampling (temperature honors generation_defaults; pinned
        # top_k/top_p/repeat_penalty + fixed seed, shared with the chat path).
        self._apply_deterministic_sampling(payload, role_config, request)

        # Forward stop sequences to llama-server
        stop_seqs = getattr(request, "stop_sequences", None)
        if stop_seqs:
            payload["stop"] = stop_seqs

        # Grammar-constrained generation (llama-server native support)
        if request.json_schema:
            payload["json_schema"] = request.json_schema
        if request.grammar:
            payload["grammar"] = request.grammar

        if request.n_probs is not None and int(request.n_probs) > 0:
            payload["n_probs"] = min(128, int(request.n_probs))
            # TD-1d.2: see the matching comment in _infer_chat_completions —
            # same server-side field, same pre- vs post-grammar-mask tradeoff,
            # on the /completion lane.
            if getattr(request, "post_sampling_probs", False):
                payload["post_sampling_probs"] = True

        # Prefix cache slot routing (id_slot=-1 means auto-assign)
        if request.slot_id is not None:
            payload["id_slot"] = request.slot_id

        # Logit probe: request top-k token probabilities for routing classifier
        from src.features import features
        if "n_probs" not in payload and features().logit_probe and role_config.name == "frontdoor":
            payload["n_probs"] = 64

        return payload

    def get_slots(self) -> list[SlotInfo]:
        """Get current slot information from the server.

        Returns:
            List of SlotInfo for each server slot.
        """
        try:
            response = self.client.get(
                "/slots",
                timeout=self.config.connect_timeout,
            )
            response.raise_for_status()
            slots_data = response.json()

            result = []
            for slot in slots_data:
                # Field names are from llama.cpp production-consolidated-v9
                # (server-context.cpp server_slot::to_json): n_prompt_tokens =
                # total prompt tokens, n_prompt_tokens_cache = tokens reused
                # from KV cache (the hit count), n_prompt_tokens_processed =
                # tokens actually evaluated. The legacy n_past/n_cache names
                # stopped existing upstream and always read 0 here (measurement
                # defect, fixed 2026-08-13 under RTE-Prefix).
                info = SlotInfo(
                    slot_id=slot.get("id", 0),
                    state=slot.get("state", "unknown"),
                    prompt_tokens=slot.get("n_prompt_tokens", 0),
                    cache_tokens=slot.get("n_prompt_tokens_cache", 0),
                    last_access=time.time(),
                )
                result.append(info)
                self.slots[info.slot_id] = info

            return result

        except httpx.RequestError as e:
            logger.warning(f"Failed to get slot info: {e}")
            return []

    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            CacheStats with hit rates and token savings.
        """
        return self.cache_stats

    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        self.cache_stats = CacheStats()

    def save_slot(self, slot_id: int, filename: str) -> bool:
        """Save a slot's KV cache state to disk.

        Args:
            slot_id: Slot to save.
            filename: BARE filename, never a path. The server resolves it under its
                own ``--slot-save-path``; it rejects any name containing a path
                separator via ``fs_validate_filename(name, allow_subdirs=false)``.

        Returns:
            True if save succeeded.
        """
        try:
            response = self.client.post(
                f"/slots/{slot_id}?action=save",
                json={"filename": filename},
                timeout=self.config.timeout,
            )
            # Catch httpx.HTTPError, not RequestError: raise_for_status() raises
            # HTTPStatusError, which is NOT a RequestError subclass, so a 4xx used to
            # propagate out of this method instead of returning False as documented.
            response.raise_for_status()
            logger.info(f"Saved slot {slot_id} to {filename}{_slot_token_suffix(response, 'n_saved')}")
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to save slot {slot_id} as {filename!r}: {e}")
            return False

    def restore_slot(self, slot_id: int, filename: str) -> bool:
        """Restore a slot's KV cache state from disk.

        Args:
            slot_id: Slot to restore.
            filename: BARE filename, never a path -- same contract as ``save_slot``.

        Returns:
            True if the request SUCCEEDED, which is not the same as "KV came
            back". llama-server answers a zero-token restore with HTTP 200, so
            the boolean proves transport only; ``n_restored`` in the body is the
            token count and it is logged, not returned. Any caller about to do
            something DESTRUCTIVE on the strength of this answer must read the
            count instead -- that exact confusion let a zero-token restore erase
            a source slot and record it as verified
            (see ``concurrency_aware._migrate_kv``, fixed 2026-08-22).
        """
        try:
            response = self.client.post(
                f"/slots/{slot_id}?action=restore",
                json={"filename": filename},
                timeout=self.config.timeout,
            )
            # See save_slot: HTTPStatusError is not a RequestError, so a 4xx used to
            # propagate rather than returning False.
            response.raise_for_status()
            logger.info(
                f"Restored slot {slot_id} from {filename}"
                f"{_slot_token_suffix(response, 'n_restored')}"
            )
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to restore slot {slot_id} from {filename!r}: {e}")
            return False

    def _infer_stream_text_chat_completions(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
        on_chunk,
        start_time: float,
    ) -> InferenceResult:
        """Streaming variant of /v1/chat/completions for use with the inference tap.

        Parses OpenAI-style SSE chunks (`data: {"choices":[{"delta":{"content":"..."}}]}`)
        and forwards each delta to on_chunk before assembling the final
        InferenceResult. Treats the request.prompt as PLAIN user content
        (same un-templating as _infer_chat_completions).
        """
        import json as _json

        user_content = request.prompt or ""
        # Same un-template logic as the non-streaming variant
        for marker in ("<|im_start|>user\n", "<|im_start|>user "):
            if user_content.startswith(marker):
                user_content = user_content[len(marker):]
                for closer in ("<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>"):
                    idx = user_content.rfind(closer)
                    if idx > 0:
                        user_content = user_content[:idx]
                break
        if user_content.startswith("<start_of_turn>user\n"):
            user_content = user_content[len("<start_of_turn>user\n"):]
            for closer in ("<end_of_turn>\n<start_of_turn>model\n", "<end_of_turn>"):
                idx = user_content.rfind(closer)
                if idx > 0:
                    user_content = user_content[:idx]
        user_content = user_content.strip()

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": request.n_tokens if request.n_tokens > 0 else 4096,
            "stream": True,
        }
        self._apply_deterministic_sampling(payload, role_config, request)
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        self._apply_schema_constraint(payload, request)

        ctk = _chat_template_kwargs(role_config, request)
        if ctk:
            payload["chat_template_kwargs"] = ctk

        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        completion_reason = "stop"

        try:
            _overall = request.timeout or self.config.timeout
            _read_timeout = _overall  # non-streaming: read must cover the full budget (see above)
            _stream_timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                read=_read_timeout, write=_overall, pool=_overall,
            )
            logger.info(
                "llama STREAM /v1/chat/completions start role=%s prompt_chars=%d max_tokens=%d",
                role_config.name, len(user_content), payload["max_tokens"],
            )
            with self.client.stream(
                "POST", "/v1/chat/completions",
                json=payload, timeout=_stream_timeout,
            ) as response:
                overflow = _overflow_from_error_response(response)
                if overflow is not None:
                    return _context_overflow_result(role_config, overflow, time.time() - start_time)
                response.raise_for_status()
                try:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                evt = _json.loads(data)
                                if isinstance(evt, dict) and "error" in evt:
                                    # In-band stream error (server-context.cpp:4540-4550).
                                    overflow = classify_error_payload(evt)
                                    if overflow is not None:
                                        return _context_overflow_result(
                                            role_config, overflow, time.time() - start_time,
                                            partial_output="".join(chunks),
                                        )
                                    err = evt.get("error")
                                    err_msg = err.get("message") if isinstance(err, dict) else str(err)
                                    return InferenceResult(
                                        role=role_config.name, output="", tokens_generated=0,
                                        generation_speed=0.0,
                                        elapsed_time=time.time() - start_time, success=False,
                                        error_message=f"chat_completions stream error: {err_msg}",
                                        degraded=True, failure_stage="stream",
                                        failure_reason="server_error",
                                        completion_reason="server_error",
                                    )
                                choices = evt.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta") or {}
                                    # RI-23: reasoning deltas are collected, never fed to
                                    # on_chunk (FINAL/early-stop checks see content only).
                                    _rc = delta.get("reasoning_content")
                                    if isinstance(_rc, str) and _rc:
                                        reasoning_chunks.append(_rc)
                                    content = delta.get("content")
                                    if content:
                                        chunks.append(content)
                                        if on_chunk is not None:
                                            try:
                                                on_chunk(content)
                                            except StopIteration:
                                                # Caller-controlled early stop
                                                logger.debug(
                                                    "chat_completions stream aborted by on_chunk"
                                                )
                                                break
                                    fr = choices[0].get("finish_reason")
                                    if fr:
                                        completion_reason = str(fr)
                            except _json.JSONDecodeError:
                                continue
                except httpx.ReadTimeout:
                    logger.warning(
                        "chat_completions stream read timeout (role=%s, accumulated %d chunks)",
                        role_config.name, len(chunks),
                    )
                    completion_reason = "read_timeout"

            output = "".join(chunks)
            elapsed = time.time() - start_time
            tokens_generated = len(output) // 4 + len(chunks)  # rough estimate from chunks
            if reasoning_chunks:
                # RI-23: reasoning tokens are model-decoded too (Pareto speed objective).
                _reasoning = "".join(reasoning_chunks)
                tokens_generated += len(_reasoning) // 4 + len(reasoning_chunks)
            if request.n_tokens > 0:
                # The streaming OpenAI shim does not send final usage/timings.
                # llama-server still enforces max_tokens, so keep telemetry from
                # overstating capped generations on chunk-heavy text.
                tokens_generated = min(tokens_generated, request.n_tokens)
            speed = tokens_generated / elapsed if elapsed > 0 else 0.0
            empty_generation = (
                completion_reason != "read_timeout"
                and _is_empty_long_generation(output, elapsed)
            )
            if empty_generation:
                logger.warning(
                    "Empty chat_completions stream after %.1fs for %s "
                    "(completion_reason=%s)",
                    elapsed,
                    role_config.name,
                    completion_reason or "unknown",
                )
            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens_generated,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=(
                    bool(output)
                    or (completion_reason != "read_timeout" and not empty_generation)
                ),
                error_message=(
                    f"Empty generation after {elapsed:.1f}s"
                    if empty_generation else None
                ),
                partial=False,
                degraded=empty_generation,
                failure_stage="generation" if empty_generation else "",
                failure_reason="empty_generation" if empty_generation else "",
                prompt_eval_ms=0.0,
                generation_ms=elapsed * 1000,
                predicted_per_second=speed,
                http_overhead_ms=0.0,
                completion_reason=(
                    "empty_generation" if empty_generation else completion_reason
                ),
                reasoning_content="".join(reasoning_chunks) or None,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name,
                output="".join(chunks),
                tokens_generated=len(chunks),
                generation_speed=0.0, elapsed_time=elapsed, success=False,
                error_message=f"chat_completions stream failed: {e}",
                failure_stage="transport", failure_reason="request_error",
                completion_reason="request_error",
            )

    def close(self) -> None:
        """Close the HTTP client and release connections.

        Call this when done with the backend to properly clean up resources.
        """
        self.client.close()
