"""OpenAI-compatible passthrough for one named role (UFH14-B6).

Harness experiments (workspace-89's codex / C95-style harnesses) need the raw
llama-server OpenAI surface — ``/v1/chat/completions`` and ``/v1/responses``
with true streaming, tool calls and per-request llama.cpp ``timings`` on the
wire — but they should still queue behind the orchestrator's admission gate
instead of hitting a model server directly. ``/v1/chat/completions`` on :8000
is not that surface: it rewrites prompts, replays finished answers one
character per chunk and drops ``timings``
(``/mnt/raid0/llm/tmp/kv-gate-8083-ec/BYPASS.md`` §3).

Route design — an explicit path, not a ``model``-name convention::

    POST /v1/passthrough/{role}/chat/completions   -> {server}/v1/chat/completions
    POST /v1/passthrough/{role}/responses          -> {server}/v1/responses
    GET  /v1/passthrough/{role}/models             -> {server}/v1/models

* A harness changes ONE setting, its OpenAI ``base_url``
  (``http://127.0.0.1:8000/v1/passthrough/architect_critic``); the body it
  sends is the body the server receives.
* The existing routes keep every byte of their behaviour: the ``model`` field
  already selects a role on ``/v1/chat/completions`` (REPL / direct / client
  modes) and its pydantic model refuses fields a passthrough must carry
  (``response_format``, ``stop``, penalties). A model-name convention would
  have had to fork that route; a separate path cannot regress it.
* The role is in the URL, so the ``model`` field stays the client's own and is
  forwarded untouched.

What the passthrough DOES (the gate, nothing else):

* resolves role -> server URL through the config ``server_urls`` map the
  primitives layer uses (ingress / compatibility aliases normalised the same
  way as ``x_force_role``); multi-endpoint fleets are refused (the token gate
  is single-URL only);
* takes the same per-backend request semaphore, the same token-aware shared KV
  pool reservation (``SharedKVPoolAdmission``: FCFS, queue bound, one long
  prefill per server) and the same CPU region / inference lock as
  ``_call_caching_backend`` (``src/llm_primitives/inference.py``). The
  long-prefill lease is the HOST-WIDE one (KVU-15a,
  ``src/runtime/long_prefill_lease.py``: an flock per server), so a
  passthrough call and an orchestrator call in another uvicorn worker exclude
  each other; the request's text estimate is passed as ``prompt_text`` so the
  long-prefill rule is sized on NEW tokens (see ``gate``);
* refuses a prompt at or above a BINDING per-request cap before dispatch
  (``context_limits``: the model's trained context below the server's slot
  n_ctx) with the typed ``request_too_large`` error (413);
* hands the long-prefill lease on at the first upstream chunk that is not a
  prompt-progress event, exactly as the streamed primitives path does, and
  returns the pool reservation when the upstream response ends;
* writes one ``serving_call.v1`` record per call
  (``src/backends/serving_calls.py``) with ``caller.source = "passthrough"``,
  carrying the server's ``timings`` parsed off the wire and the UFH14-B4
  prefix-cache fields: ``request.prefix_fp`` (fingerprinted exactly as the
  client-tool-mode chat lane fingerprints ``tools`` + ``messages``, so
  ``scripts/analysis/prefix_cache_report.py`` sees passthrough calls' missed
  reuse), ``request.slot_id`` / ``slot_id_sent`` (a client ``id_slot`` is
  forwarded untouched), ``passthrough.cached_prompt_tokens`` (``timings.cache_n``,
  else ``usage.prompt_tokens_details.cached_tokens``) and ``notes.server_slot``
  when the server reports an ``id_slot`` (v10's OAI chat/responses bodies do
  not; only ``/completion`` does).

What it does NOT do: no routing, compaction, prompt rewriting, role-default
sampling, ``max_tokens`` clamping or response rewriting. The request body is
forwarded as the client's raw bytes and the upstream bytes are streamed back
as they arrive (one worker thread per call reads the upstream response and
feeds an asyncio queue, so the event loop never blocks and region locks are
taken and released on one thread). It is the system under test.

``timings``: llama-server v10 already puts ``timings`` in the non-streamed
chat body, the final streamed chat chunk and the streamed ``/v1/responses``
``response.completed`` event, so they reach the client verbatim. The one
gap is a NON-streamed ``/v1/responses`` call: v10's
``to_json_oaicompat_resp()`` (tools/server/server-task.cpp:558) emits no
``timings`` at all, so neither the client nor the serving record gets them —
stream ``/v1/responses`` calls when timings matter (codex streams).

Safety: localhost only, the same rule as ``POST /config``. Kill switch
``ORCHESTRATOR_PASSTHROUGH=0``; optional role allow-list
``ORCHESTRATOR_PASSTHROUGH_ROLES`` (comma-separated canonical role names).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.api.dependencies import dep_app_state
from src.api.state import AppState
from src.backends import serving_calls
from src.exceptions import ContextOverflowError

logger = logging.getLogger(__name__)

router = APIRouter()

PASSTHROUGH_ENV = "ORCHESTRATOR_PASSTHROUGH"
PASSTHROUGH_ROLES_ENV = "ORCHESTRATOR_PASSTHROUGH_ROLES"
# Longest silence tolerated between two upstream chunks. A streamed call is
# silent for its whole prefill (two concurrent 46k cold prefills on :8083 took
# 395 s and 665 s), so this is generous; it matches the pool wait default.
READ_TIMEOUT_ENV = "ORCHESTRATOR_PASSTHROUGH_READ_TIMEOUT_S"
DEFAULT_READ_TIMEOUT_S = 1800.0
CONNECT_TIMEOUT_S = 10.0

LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")

# Passthrough endpoint -> llama-server path.
UPSTREAM_PATHS = {
    "chat/completions": "/v1/chat/completions",
    "responses": "/v1/responses",
}

# Structural JSON keys whose string values the chat template never renders as
# text; everything else in messages / input counts toward the prompt estimate.
_STRUCTURAL_KEYS = frozenset({
    "type", "role", "id", "call_id", "tool_call_id", "status", "detail", "format",
})

# llama-server's pool-exhaustion message (server-context.cpp, see ContextOverflowError).
_POOL_EXHAUSTED_MARKER = b"Context size has been exceeded"


# ── request inspection ───────────────────────────────────────────────────────


def _text_leaves(value: Any, out: list[str]) -> None:
    """Collect the string leaves a chat template would render (prompt estimate)."""
    if isinstance(value, str):
        if not value.startswith("data:"):  # inline image / audio payloads
            out.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in _STRUCTURAL_KEYS:
                continue
            _text_leaves(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _text_leaves(item, out)


def prompt_text(body: dict[str, Any]) -> str:
    """The text a request puts in front of the model, for token ESTIMATES only.

    Chat: every message's rendered strings (content parts, tool-call arguments,
    reasoning). Responses: ``instructions`` + ``input`` items. Both: the tool
    schemas, which the template renders as JSON. Never sent anywhere.
    """
    parts: list[str] = []
    for key in ("messages", "instructions", "input", "prompt"):
        if key in body:
            _text_leaves(body[key], parts)
    tools = body.get("tools")
    if tools:
        try:
            parts.append(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            pass
    return "\n".join(parts)


def requested_new_tokens(body: dict[str, Any]) -> int:
    """The client's generation budget (chat or responses spelling), or the default."""
    from src.llm_primitives.inference import DEFAULT_GENERATION_BUDGET_TOKENS

    for key in ("max_completion_tokens", "max_tokens", "max_output_tokens", "n_predict"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return DEFAULT_GENERATION_BUDGET_TOKENS


def _enabled() -> bool:
    return os.environ.get(PASSTHROUGH_ENV, "1").strip().lower() not in {
        "0", "false", "off", "no", "disabled",
    }


def _allowed_roles() -> set[str] | None:
    raw = os.environ.get(PASSTHROUGH_ROLES_ENV, "").strip()
    if not raw:
        return None
    return {r.strip() for r in raw.split(",") if r.strip()}


def _require_local(http_request: Request) -> str:
    client_ip = http_request.client.host if http_request.client else "unknown"
    if client_ip not in LOCAL_HOSTS:
        logger.warning("Rejected passthrough request from non-localhost: %s", client_ip)
        raise HTTPException(status_code=403, detail="Passthrough is only allowed from localhost")
    return client_ip


def _server_urls() -> dict[str, str]:
    """The role -> server URL map the primitives layer dispatches with."""
    from src.config import get_config

    return get_config().server_urls.as_dict()


def resolve_role_url(role: str) -> tuple[str, str]:
    """``(canonical role, server base URL)`` or an HTTP error."""
    from src.api.routes.openai_compat import normalize_override_role
    from src.roles import Role

    canonical = normalize_override_role(role)
    role_name = canonical.value if isinstance(canonical, Role) else str(canonical)
    allowed = _allowed_roles()
    if allowed is not None and role_name not in allowed and role not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"role {role_name!r} is not in {PASSTHROUGH_ROLES_ENV}",
        )
    try:
        urls = _server_urls()
    except Exception as exc:  # pragma: no cover - degraded config
        raise HTTPException(status_code=503, detail=f"server_urls unavailable: {exc}") from exc
    url = str(urls.get(role_name) or "").strip()
    if not url:
        raise HTTPException(
            status_code=404,
            detail=f"role {role!r} has no server URL; see GET /v1/models for servable roles",
        )
    if "," in url:
        raise HTTPException(
            status_code=422,
            detail=(
                f"role {role_name!r} is served by a multi-endpoint fleet ({url}); the "
                "passthrough targets single-endpoint roles only (the KV pool gate is per URL)"
            ),
        )
    return role_name, url.rstrip("/")


def _extract_port_of(url: str) -> int | None:
    from urllib.parse import urlparse

    try:
        return urlparse(url).port
    except ValueError:
        return None


# ── per-call context ─────────────────────────────────────────────────────────


@dataclass
class _Call:
    """Everything one passthrough call carries between the route and its worker."""

    role: str
    base_url: str
    endpoint: str
    raw_body: bytes
    stream: bool
    prompt_tokens_est: int
    prompt_tokens_rough: int
    new_tokens: int
    client_host: str
    request_id: str
    client_id: str | None
    role_config: Any = None
    # The prompt-estimate text (``prompt_text(body)``), handed to the KV pool
    # gate for the KVU-15a cached-prefix credit. Never sent anywhere.
    prompt_text: str | None = field(default=None, repr=False)
    # UFH14-B4 serving-record fields: wire-prompt fingerprints, the client's own
    # ``id_slot`` (forwarded untouched) and the slot the server reports.
    prefix_fp: dict[str, Any] | None = None
    slot_id: int | None = None
    server_slot: int | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    response: Any = None  # live httpx.Response, for a best-effort close on cancel
    # outcome
    t_staged: float = field(default_factory=time.time)
    t_dispatch: float | None = None
    t_first_chunk: float | None = None
    http_status: int | None = None
    bytes_out: int = 0
    events: int = 0
    timings: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    upstream_error: str | None = None
    pool_exhausted: bool = False
    client_disconnected: bool = False
    gate: dict[str, Any] = field(default_factory=dict)

    @property
    def upstream_url(self) -> str:
        return self.base_url + UPSTREAM_PATHS[self.endpoint]


class _SSEScanner:
    """Read-only scan of the upstream bytes: first output, timings, usage, errors.

    Never alters a byte; the scan only feeds the gate (long-prefill lease
    hand-off) and the serving record.
    """

    def __init__(self, call: _Call, on_first_output) -> None:
        self._call = call
        self._buf = b""
        self._on_first_output = on_first_output
        self._first_seen = False

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while True:
            # SSE events end with a blank line; accept \n\n and \r\n\r\n.
            idx = self._buf.find(b"\n\n")
            idx_crlf = self._buf.find(b"\r\n\r\n")
            if idx_crlf != -1 and (idx == -1 or idx_crlf < idx):
                event, self._buf = self._buf[:idx_crlf], self._buf[idx_crlf + 4:]
            elif idx != -1:
                event, self._buf = self._buf[:idx], self._buf[idx + 2:]
            else:
                return
            self._event(event)

    def _event(self, event: bytes) -> None:
        data_lines = [
            line[5:].strip() for line in event.splitlines() if line.startswith(b"data:")
        ]
        if not data_lines:
            return
        data = b"\n".join(data_lines)
        self._call.events += 1
        if data == b"[DONE]":
            return
        if not self._first_seen and b"prompt_progress" not in data:
            self._first_seen = True
            self._on_first_output()
        if _POOL_EXHAUSTED_MARKER in data:
            self._call.pool_exhausted = True
        if (b'"timings"' in data or b'"usage"' in data or b'"error"' in data
                or b'"id_slot"' in data):
            try:
                obj = json.loads(data)
            except ValueError:
                return
            absorb_payload(self._call, obj)


def absorb_payload(call: _Call, obj: Any) -> None:
    """Pick ``timings`` / ``usage`` / ``error`` out of one upstream JSON payload."""
    if not isinstance(obj, dict):
        return
    timings = obj.get("timings")
    if isinstance(timings, dict) and ("prompt_n" in timings or "predicted_n" in timings):
        call.timings = timings
    id_slot = obj.get("id_slot")
    if isinstance(id_slot, int) and not isinstance(id_slot, bool) and id_slot >= 0:
        call.server_slot = id_slot
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        resp = obj.get("response")  # /v1/responses response.completed event
        usage = resp.get("usage") if isinstance(resp, dict) else None
    if isinstance(usage, dict) and usage:
        call.usage = usage
    error = obj.get("error")
    if error:
        call.upstream_error = (
            str(error.get("message") or error) if isinstance(error, dict) else str(error)
        )[:300]


# ── the gate ─────────────────────────────────────────────────────────────────


class _Refused(Exception):
    """The gate refused the call before dispatch: an HTTP response to return."""

    def __init__(
        self,
        status: int,
        error: str,
        detail: str,
        retry_after: int | None = None,
        refusal: dict[str, Any] | None = None,
    ):
        super().__init__(detail)
        self.status = status
        self.error = error
        self.detail = detail
        self.retry_after = retry_after
        # UFH14-B6a: the structured reason. ``gate`` is the refusing gate's name
        # (== ``error``), so readers of the serving record never parse ``detail``.
        self.refusal: dict[str, Any] = {"gate": error, "http_status": status}
        if retry_after:
            self.refusal["retry_after_s"] = retry_after
        self.refusal.update(refusal or {})

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status,
            content={
                "error": self.error,
                "type": self.error,
                "detail": self.detail,
                "error_code": self.status,
                "error_detail": self.detail,
                "retry_after_s": self.retry_after,
                "refusal": dict(self.refusal),
            },
            headers={"Retry-After": str(self.retry_after)} if self.retry_after else None,
        )

    @classmethod
    def from_parked(cls, exc: Any) -> "_Refused":
        """A ``RoleParkedError`` as a passthrough refusal (same 503 body)."""
        return cls(503, "role_parked", str(exc), retry_after=exc.retry_after_s,
                   refusal=exc.refusal)


def _context_limit(call: _Call):
    try:
        from src.backends.context_limits import get_context_limit_resolver

        return get_context_limit_resolver().limit_for_url(call.base_url)
    except Exception:
        logger.debug("passthrough: context limit lookup failed for %s", call.base_url,
                     exc_info=True)
        return None


def check_request_cap(call: _Call, limit: Any) -> None:
    """Refuse a prompt at a BINDING per-request cap (the server would accept it).

    Same rule and estimate as ``_request_cap_refusal`` in the primitives layer:
    only when the model cap is below the server's slot n_ctx, with the rough
    4 chars/token estimate. Otherwise the server decides and its own 400 is
    forwarded verbatim.
    """
    if limit is None or not getattr(limit, "cap_binding", False):
        return
    if call.prompt_tokens_rough >= int(limit.per_request_n_ctx):
        raise ContextOverflowError(
            f"context overflow (per-request cap) on role {call.role}: "
            f"~{call.prompt_tokens_rough} prompt tokens exceed the model's per-request "
            f"limit {limit.per_request_n_ctx} (server slot n_ctx {limit.slot_n_ctx}); "
            "never dispatched",
            kind=ContextOverflowError.REQUEST_TOO_LARGE,
            role=call.role,
            backend_url=call.base_url,
            n_prompt_tokens=call.prompt_tokens_rough,
            n_ctx=int(limit.per_request_n_ctx),
            source="request_cap",
        )


def _region_lock(call: _Call):
    """The CPU region / inference lock ``_call_caching_backend`` takes for a
    direct single-endpoint role (no-op for a role with no CPU regions, e.g. a
    GPU-resident server)."""
    from src.llm_primitives.inference import _extract_port, _per_region_locks_enabled

    port = _extract_port(call.base_url)
    cancel_check = call.cancel.is_set
    tag = f"passthrough-{call.request_id}"
    if _per_region_locks_enabled():
        from src.runtime.cpu_region_lock import cpu_region_lock_for_instance
        from src.runtime.instance_topology import topology_instance_for_port

        lock_role, lock_idx = topology_instance_for_port(port or 0) or (call.role, 0)
        return cpu_region_lock_for_instance(
            lock_role, lock_idx, cancel_check=cancel_check, request_tag=tag,
        )
    from src.inference_lock import inference_lock

    return inference_lock(call.role, cancel_check=cancel_check, request_tag=tag, port=port)


@contextlib.contextmanager
def gate(call: _Call, state: AppState):
    """Admission for one call: request semaphore -> KV pool -> region lock.

    Yields a ``first_output()`` callback that hands the long-prefill lease on.
    Raises ``_Refused`` / ``ContextOverflowError`` before dispatch; releases
    everything (reporting pool exhaustion) on exit.
    """
    from src.scheduling.kv_pool_admission import KVPoolQueueFull, get_shared_pool_admission

    url = call.base_url
    t0 = time.perf_counter()
    health = getattr(state, "health_tracker", None)
    if health is not None and not health.is_available(url):
        raise _Refused(503, "backend_unavailable",
                       f"Backend unavailable (circuit open): {url}", retry_after=10)

    admission = getattr(state, "admission", None)
    admitted = False
    if admission is not None:
        # Same bounded wait as _call_caching_backend for a call with no deadline.
        if not admission.acquire(url, priority="interactive", wait=True, timeout_s=2.0,
                                 cancel_check=call.cancel.is_set):
            raise _Refused(503, "admission_queue_full",
                           f"[ERROR: admission] Backend queue full for {url}", retry_after=5)
        admitted = True

    pool = None
    ticket = None
    try:
        limit = _context_limit(call)
        if limit is not None and limit.shared_pool:
            pool = get_shared_pool_admission()
            try:
                # KVU-15a: ``prompt_text`` lets admission size the long-prefill
                # rule on NEW tokens. The passthrough body is a chat/responses
                # payload the server templates itself, so this text is not the
                # slot's cached text; the credit (``common_prefix_chars``) only
                # counts characters that match an idle slot's prompt right after
                # its template head, i.e. at most the first message's content —
                # an UNDER-estimate of the cached prefix, never an over-credit of
                # rendered text the server would have to prefill. No match (tools
                # rendered first, a dated template head, no /slots) means
                # whole-prompt sizing, exactly as before.
                ticket = pool.acquire(
                    url, call.prompt_tokens_est, limit.pool_tokens,
                    max_new_tokens=call.new_tokens, cancel_check=call.cancel.is_set,
                    prompt_text=call.prompt_text,
                )
            except KVPoolQueueFull as queue_full:
                raise ContextOverflowError(
                    f"context overflow (shared KV pool admission queue full) on role "
                    f"{call.role} ({url}): {queue_full.queued} requests already queued "
                    f"(limit {queue_full.limit}); retry later — never dispatched",
                    kind=ContextOverflowError.POOL_EXHAUSTED, role=call.role,
                    backend_url=url, n_ctx=limit.per_request_n_ctx, source="admission",
                ) from queue_full
            if ticket is None:
                if call.cancel.is_set():
                    raise _Refused(499, "client_closed", "client disconnected while queued")
                raise ContextOverflowError(
                    f"context overflow (shared KV pool busy) on role {call.role} ({url}): "
                    f"the request's wait budget ended before its "
                    f"{pool.reservation_tokens(url, call.prompt_tokens_est, call.new_tokens)} "
                    f"tokens fit (pool {limit.pool_tokens}); it was never dispatched",
                    kind=ContextOverflowError.POOL_EXHAUSTED, role=call.role,
                    backend_url=url, n_ctx=limit.per_request_n_ctx, source="admission",
                )
            # ``long_prefill`` = this call took the (host-wide) long-prefill
            # lease. With the KVU-15a cached-prefix credit a whole-prompt-long
            # request can be admitted as short; ``cache_credited`` marks that.
            took_lease = pool.long_prefill_holder(url) == ticket
            call.gate.update(
                pool_ticket=True,
                pool_tokens=limit.pool_tokens,
                long_prefill=took_lease,
                cache_credited=(
                    not took_lease and pool.is_long_prefill(call.prompt_tokens_est)
                ),
                lease_cross_process=pool.cross_process_lease(),
            )
        try:
            lock_ctx = _region_lock(call)
            lock_ctx.__enter__()
        except Exception as exc:
            if call.cancel.is_set():
                raise _Refused(499, "client_closed", "client disconnected while queued") from exc
            raise _Refused(503, "lock_unavailable",
                           f"inference/region lock not acquired for {call.role}: {exc}",
                           retry_after=5) from exc
        call.gate["queue_wait_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        def first_output() -> None:
            if pool is not None:
                pool.prefill_done(url, ticket)

        lock_error: BaseException | None = None
        try:
            yield first_output
        except BaseException as exc:
            lock_error = exc
            raise
        finally:
            with contextlib.suppress(Exception):
                if lock_error is None:
                    lock_ctx.__exit__(None, None, None)
                else:
                    lock_ctx.__exit__(type(lock_error), lock_error, lock_error.__traceback__)
    finally:
        if pool is not None and ticket is not None:
            if call.pool_exhausted:
                pool.report_pool_exhausted(url)
            pool.release(url, ticket, success=call.t_dispatch is not None
                         and not call.pool_exhausted and call.upstream_error is None)
        if admitted:
            admission.release(url)


# ── worker thread ────────────────────────────────────────────────────────────


def _read_timeout() -> float:
    try:
        return float(os.environ.get(READ_TIMEOUT_ENV, DEFAULT_READ_TIMEOUT_S))
    except ValueError:
        return DEFAULT_READ_TIMEOUT_S


def run_upstream(call: _Call, state: AppState, emit) -> None:
    """Gate, dispatch and stream one call; every outcome reaches ``emit``.

    ``emit(kind, payload)``: ``("start", (status, headers))`` once, then
    ``("data", bytes)`` per upstream read, then ``("end", None)``; or a single
    ``("refused", exc)`` when the gate or the connection fails before any byte.
    Runs on its own thread so region locks are taken and released on one thread.
    """
    error: BaseException | None = None
    started = False
    health = getattr(state, "health_tracker", None)
    try:
        with gate(call, state) as first_output:
            scanner = _SSEScanner(call, first_output)
            timeout = httpx.Timeout(CONNECT_TIMEOUT_S, read=_read_timeout() or None)
            call.t_dispatch = time.time()
            with httpx.Client(timeout=timeout) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                    "Accept": "text/event-stream" if call.stream else "application/json",
                }
                with client.stream("POST", call.upstream_url, content=call.raw_body,
                                   headers=headers) as resp:
                    call.response = resp
                    call.http_status = resp.status_code
                    passed = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() in ("content-type",)
                    }
                    emit("start", (resp.status_code, passed))
                    started = True
                    is_sse = "text/event-stream" in resp.headers.get("content-type", "")
                    body_buf = bytearray()
                    for chunk in resp.iter_raw():
                        if not chunk:
                            continue
                        if call.t_first_chunk is None:
                            call.t_first_chunk = time.time()
                        call.bytes_out += len(chunk)
                        if is_sse:
                            scanner.feed(chunk)
                        elif len(body_buf) < 64 * 1024 * 1024:
                            body_buf.extend(chunk)
                        emit("data", chunk)
                        if call.cancel.is_set():
                            call.client_disconnected = True
                            break
                    if call.cancel.is_set():
                        call.client_disconnected = True
                    if not is_sse and body_buf:
                        if _POOL_EXHAUSTED_MARKER in body_buf:
                            call.pool_exhausted = True
                        with contextlib.suppress(ValueError):
                            absorb_payload(call, json.loads(bytes(body_buf)))
                        if resp.status_code >= 400 and call.upstream_error is None:
                            call.upstream_error = bytes(body_buf[:300]).decode("utf-8", "replace")
        if health is not None:
            if call.http_status is not None and call.http_status < 500:
                health.record_success(call.base_url)
            elif call.http_status is not None:
                health.record_failure(call.base_url)
    except BaseException as exc:  # reported to the route / client, then recorded
        error = exc
        if call.cancel.is_set():
            call.client_disconnected = True
        if (health is not None and call.t_dispatch is not None
                and isinstance(exc, httpx.TransportError) and not call.client_disconnected):
            health.record_failure(call.base_url)
    finally:
        call.response = None
        write_serving_record(call, error)
        if error is not None and not started:
            emit("refused", error)
        else:
            emit("end", error)


def abort_upstream(resp: Any) -> None:
    """Unblock a worker parked in an upstream read, from another thread.

    ``Response.close()`` from a second thread does not interrupt a blocking
    ``recv``; shutting the socket down does, and it is also what tells
    llama-server the client is gone so it stops generating for this slot.
    """
    if resp is None:
        return
    try:
        stream = resp.extensions.get("network_stream")
        sock = stream.get_extra_info("socket") if stream is not None else None
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            return
    except Exception:
        pass
    with contextlib.suppress(Exception):
        resp.close()


def outcome(call: _Call, error: BaseException | None) -> str:
    if isinstance(error, ContextOverflowError) or call.pool_exhausted:
        return "context_overflow"
    if call.client_disconnected:
        return "cancelled"
    if isinstance(error, _Refused):
        return "refused"
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if error is not None:
        return "exception"
    if call.upstream_error is not None or (call.http_status or 0) >= 400:
        return "failed"
    return "ok"


def cached_prompt_tokens(call: _Call) -> int | None:
    """Prompt tokens the server reused for the call (UFH14-B4): ``timings.cache_n``
    (what ``serving_calls`` falls back to on the primitives lanes), else the OAI
    ``usage.prompt_tokens_details.cached_tokens``; None when neither was seen."""
    cache_n = (call.timings or {}).get("cache_n")
    if isinstance(cache_n, int) and not isinstance(cache_n, bool):
        return cache_n
    details = (call.usage or {}).get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool):
            return cached
    return None


def wire_prefix_fingerprints(endpoint: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """UFH14-B4 ``prefix_fp`` for a passthrough body. Chat bodies are
    fingerprinted through ``serving_calls.prefix_fingerprints`` as a client-tool
    ``chat_payload`` (``tools`` then ``messages``), so a passthrough call and an
    orchestrator client-tool call with the same payload agree. A /v1/responses
    body puts ``instructions`` + ``input`` where the messages go."""
    from types import SimpleNamespace

    if endpoint == "responses":
        messages: Any = {"instructions": body.get("instructions"), "input": body.get("input")}
    else:
        messages = body.get("messages")
    return serving_calls.prefix_fingerprints(
        SimpleNamespace(chat_payload={"tools": body.get("tools"), "messages": messages})
    )


def write_serving_record(call: _Call, error: BaseException | None) -> None:
    """One ``serving_call.v1`` record, ``caller.source = "passthrough"``. Never raises."""
    try:
        now = time.time()
        dispatched = call.t_dispatch is not None
        notes: dict[str, Any] = {"endpoint": UPSTREAM_PATHS[call.endpoint], "stream": call.stream}
        if call.timings:
            notes["timings"] = {
                k: call.timings[k] for k in serving_calls.TIMING_KEYS if k in call.timings
            }
        if call.usage:
            notes["usage"] = call.usage
        if call.t_first_chunk is not None and dispatched:
            notes["first_chunk_ms"] = round((call.t_first_chunk - call.t_dispatch) * 1000.0, 3)
        if call.server_slot is not None:
            notes["server_slot"] = call.server_slot  # same key as serving_calls.note_server_slot
        record = serving_calls.build_record(
            method="passthrough",
            role_config=call.role_config,
            request=None,
            base_url=call.base_url,
            ts_start=call.t_dispatch if dispatched else call.t_staged,
            ts_end=now,
            exc=error if not isinstance(error, _Refused) else None,
            notes=notes,
            dispatched=dispatched,
        )
        record["role"] = call.role
        record["request_role"] = call.role
        # UFH14-B6b: the same ``port`` / ``backend_url`` the primitives lanes
        # stage, so per-port analyses do not drop passthrough calls.
        record["caller"] = {
            "source": "passthrough",
            "role": call.role,
            "request_id": call.request_id,
            "client": call.client_id,
            "client_host": call.client_host,
            "backend_url": call.base_url,
            "port": _extract_port_of(call.base_url),
        }
        record["queue"] = {
            "pre_dispatch_wait_ms": round(
                max(0.0, ((call.t_dispatch if dispatched else now) - call.t_staged) * 1000.0), 3
            )
        }
        record["outcome"] = outcome(call, error)
        # UFH14-B4: the same ``request`` block shape the primitives lanes write,
        # so prefix_cache_report.py can fingerprint-match passthrough calls.
        record["request"] = {
            "n_tokens": call.new_tokens,
            "prompt_chars": len(call.prompt_text or ""),
            "timeout_s": None,
            "slot_id": call.slot_id,
            "slot_id_sent": call.slot_id is not None,  # raw body forwarded verbatim
            "chat_payload": True,
            "prefix_fp": call.prefix_fp,
        }
        record["passthrough"] = {
            "endpoint": call.endpoint,
            "stream": call.stream,
            "http_status": call.http_status,
            "bytes": call.bytes_out,
            "sse_events": call.events,
            "prompt_tokens_est": call.prompt_tokens_est,
            "new_tokens_budget": call.new_tokens,
            "upstream_error": call.upstream_error,
            "client_disconnected": call.client_disconnected,
            "cached_prompt_tokens": cached_prompt_tokens(call),
            **call.gate,
        }
        if isinstance(error, _Refused):
            record["error"] = {"type": error.error, "message": error.detail[:300]}
            record["refusal"] = dict(error.refusal)
        elif isinstance(error, ContextOverflowError):
            record.setdefault("error", {})["kind"] = error.kind
            if not dispatched:
                record["refusal"] = {
                    "gate": "context_overflow", "kind": error.kind, "source": error.source,
                }
        serving_calls.write_record(record)
    except Exception:
        logger.debug("passthrough: serving record failed", exc_info=True)


# ── routes ───────────────────────────────────────────────────────────────────


async def _passthrough(endpoint: str, role: str, http_request: Request, state: AppState):
    if not _enabled():
        raise HTTPException(status_code=404, detail="passthrough is disabled")
    client_host = _require_local(http_request)
    role_name, base_url = resolve_role_url(role)
    raw_body = await http_request.body()
    try:
        body = json.loads(raw_body or b"null")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"request body is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    from src.backends.context_limits import estimate_tokens, estimate_tokens_conservative
    from src.llm_primitives.inference import _role_config_for_backend

    text = prompt_text(body)
    call = _Call(
        role=role_name,
        base_url=base_url,
        endpoint=endpoint,
        raw_body=raw_body,
        stream=bool(body.get("stream")),
        prompt_tokens_est=estimate_tokens_conservative(text),
        prompt_tokens_rough=estimate_tokens(text),
        new_tokens=requested_new_tokens(body),
        client_host=client_host,
        request_id=http_request.headers.get("x-request-id") or uuid.uuid4().hex,
        client_id=http_request.headers.get("x-client-id"),
        role_config=_role_config_for_backend(getattr(state, "registry", None), role_name),
        prompt_text=text,
        prefix_fp=wire_prefix_fingerprints(endpoint, body),
        slot_id=(
            body.get("id_slot")
            if isinstance(body.get("id_slot"), int) and not isinstance(body.get("id_slot"), bool)
            else None
        ),
    )

    # Parked role (GPU lent to AutoKernel, src/runtime/gpu_window.py): an explicit
    # 503 role_parked before any gate, plus a preempt request so this real
    # request starts the drain. Recorded below with refusal.gate = role_parked.
    from src.exceptions import RoleParkedError
    from src.runtime import gpu_window

    try:
        gpu_window.refuse_if_parked(
            role_name, _extract_port_of(base_url), request_id=call.request_id, record=False,
        )
    except RoleParkedError as parked:
        refused = _Refused.from_parked(parked)
        write_serving_record(call, refused)
        return refused.response()

    try:
        check_request_cap(call, _context_limit(call))
    except ContextOverflowError as exc:
        write_serving_record(call, exc)
        raise

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(kind: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    worker = threading.Thread(
        target=run_upstream, args=(call, state, emit),
        name=f"passthrough-{role_name}-{call.request_id[:8]}", daemon=True,
    )
    worker.start()

    # Wait for the gate (the client may disconnect while queued).
    while True:
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=0.5)
            break
        except asyncio.TimeoutError:
            if await http_request.is_disconnected():
                call.cancel.set()
    if kind == "refused":
        if isinstance(payload, ContextOverflowError):
            raise payload
        if isinstance(payload, _Refused):
            return payload.response()
        if isinstance(payload, httpx.TransportError):
            return _Refused(502, "upstream_unreachable",
                            f"{call.upstream_url}: {payload}").response()
        return _Refused(502, "passthrough_error", f"{type(payload).__name__}: {payload}").response()

    status, headers = payload
    media_type = headers.get("content-type") or headers.get("Content-Type")
    out_headers = {
        "X-Orchestrator-Passthrough": role_name,
        "X-Orchestrator-Request-Id": call.request_id,
    }
    if media_type and "text/event-stream" in media_type:
        out_headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def body_iter() -> AsyncIterator[bytes]:
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "data":
                    yield payload
                else:  # "end": upstream finished (or failed mid-stream)
                    return
        finally:
            # Client gone or stream over: stop the worker reading upstream.
            if worker.is_alive():
                call.cancel.set()
                abort_upstream(call.response)

    return StreamingResponse(body_iter(), status_code=status, media_type=media_type,
                             headers=out_headers)


@router.post("/passthrough/{role}/chat/completions", response_model=None)
async def passthrough_chat_completions(
    role: str, http_request: Request, state: AppState = Depends(dep_app_state),
):
    """Raw llama-server ``/v1/chat/completions`` for ``role``, through the gate."""
    return await _passthrough("chat/completions", role, http_request, state)


@router.post("/passthrough/{role}/responses", response_model=None)
async def passthrough_responses(
    role: str, http_request: Request, state: AppState = Depends(dep_app_state),
):
    """Raw llama-server ``/v1/responses`` for ``role``, through the gate."""
    return await _passthrough("responses", role, http_request, state)


@router.get("/passthrough/{role}/models", response_model=None)
async def passthrough_models(role: str, http_request: Request):
    """The role's server ``/v1/models`` (no inference, so no admission)."""
    if not _enabled():
        raise HTTPException(status_code=404, detail="passthrough is disabled")
    _require_local(http_request)
    role_name, base_url = resolve_role_url(role)
    # A parked server is stopped: say so (no preempt — a listing is not a request).
    from src.exceptions import RoleParkedError
    from src.runtime import gpu_window

    try:
        gpu_window.refuse_if_parked(role_name, _extract_port_of(base_url),
                                    preempt=False, record=False)
    except RoleParkedError as parked:
        return _Refused.from_parked(parked).response()
    try:
        async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_S) as client:
            resp = await client.get(base_url + "/v1/models")
    except httpx.HTTPError as exc:
        return _Refused(502, "upstream_unreachable", f"{base_url}/v1/models: {exc}").response()
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))
