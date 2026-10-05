"""OpenAI-compatible endpoints for the orchestrator API.

These endpoints allow tools like Aider, LM Studio, and other OpenAI-compatible
clients to use our orchestrator backend for inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.dependencies import dep_app_state
from src.api.models import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIModelInfo,
    OpenAIModelsResponse,
    OpenAIUsage,
)
from src.api.routes.chat_pipeline.routing_decision import normalize_ingress_role
from src.api.routes.v1_escalation import (
    STAGE_DIRECT,
    STAGE_REPL,
    V1EscalationPlan,
    escalate_answer,
    plan_v1_escalation,
    record_escalation,
)
from src.api.routes.v1_subagent_link import (
    SubagentLink,
    get_registry,
    log_subagent_link,
    resolve_subagent_link,
)
from src.api.state import AppState
from src.autopilot_core.measurement_guards import inband_error_text
from src.prompt_builders import (
    build_root_lm_prompt,
    extract_code_from_response,
    auto_wrap_final,
    rescue_bare_name_final,
)
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_primary_port,
    stack_prior_serving,
)
from src.repl_environment import REPLEnvironment
from src.exceptions import AdmissionDenied, AdmissionDeniedText, ContextOverflowError
from src.scheduling.contention_gate import ContentionDenied
from src.roles import Role

logger = logging.getLogger(__name__)

router = APIRouter()

# B2 context compression on /v1 is FAIL-OPEN by policy: any compressor failure
# falls back to the unfolded history so a request never dies on a telemetry-grade
# optimisation. Fail-open conceals its own corruption, so the fallback must be
# VISIBLE: it is counted here (same in-process convention as
# ``TOOL_CALL_JSON_REPAIR_COUNTS`` in ``src/prompt_builders/code_utils.py`` --
# there is no prometheus_client in src/) and logged with the exception type as a
# structured ``context_compression_fallback exc_type=...`` WARNING. Changing the
# fail-open itself is a policy decision nobody has taken; this only makes it seen.
_compression_fallback_lock = threading.Lock()
CONTEXT_COMPRESSION_FALLBACK_COUNTS: dict[str, int] = {"total": 0}


def _record_compression_fallback(exc: BaseException, message_count: int) -> None:
    exc_type = type(exc).__name__
    with _compression_fallback_lock:
        CONTEXT_COMPRESSION_FALLBACK_COUNTS["total"] += 1
        CONTEXT_COMPRESSION_FALLBACK_COUNTS[exc_type] = (
            CONTEXT_COMPRESSION_FALLBACK_COUNTS.get(exc_type, 0) + 1
        )
    logger.warning(
        "context_compression_fallback exc_type=%s messages=%d detail=%s "
        "(fail-open: serving unfolded history)",
        exc_type, message_count, exc,
    )


def _compressed_history_dicts(history_messages: list[OpenAIMessage]) -> list[dict[str, Any]]:
    """Return history dicts, B2-compressed when the flag is on and the history is long.

    Fail-open on compressor error (see ``_record_compression_fallback``): the
    unfolded history is returned, and the fallback is counted and logged.
    """
    from src.features import features as _feat

    unfolded = [_history_message_dict(m) for m in history_messages]
    if not (_feat().context_compression and len(history_messages) > 8):
        return unfolded
    try:
        from src.context_compression import ContextCompressor

        # Fresh copy for the compressor: a compressor that mutates its input
        # before raising must not corrupt the unfolded history we fall back to.
        result = ContextCompressor().compress([_history_message_dict(m) for m in history_messages])
        if result.tool_outputs_summarized > 0 or result.tool_pairs_fixed > 0:
            logger.info(
                "B2 context compression: %d outputs summarized, %d pairs fixed",
                result.tool_outputs_summarized, result.tool_pairs_fixed,
            )
        return result.messages
    except Exception as exc:  # fail-open by policy; made visible, not swallowed
        _record_compression_fallback(exc, len(history_messages))
        return unfolded


def _repl_memrl_kwargs(state: AppState) -> dict[str, Any]:
    """MemRL components for a /v1 REPL, matching the /chat REPL sites.

    Without these a /v1 REPL had no retriever or router, so ``recall()`` fell
    into a broken legacy fallback for every /v1 client (the /chat paths in
    ``chat.py`` and ``chat_pipeline/`` always pass both).
    ``ensure_memrl_initialized`` is idempotent and returns False when the
    ``memrl`` flag is off; then both values are None and the REPL tools
    report their explicit fallback/unavailable results.
    """
    from src.api.services.memrl import ensure_memrl_initialized

    ensure_memrl_initialized(state)
    hybrid_router = state.hybrid_router
    return {
        "retriever": hybrid_router.retriever if hybrid_router is not None else None,
        "hybrid_router": hybrid_router,
    }


def _raise_admission_denied_text(value: str) -> None:
    # Preserve the typed cause before parsing, stripping or executing model text.
    # A model string with the same spelling cannot manufacture backpressure.
    if isinstance(value, AdmissionDeniedText):
        raise value.error


def _sse_error_event(
    *,
    chat_id: str,
    created: int,
    model: str,
    message: str,
    error_type: str,
    status_code: int,
) -> str:
    """Terminal SSE event for a backend failure (HS-OD-2).

    A stream cannot retract its 200 — headers are on the wire before the
    generator runs — so the only honest signal left is the event body. Emitting
    the failure as an ``error`` object rather than as assistant ``content`` is
    what lets a client tell "the model said this" from "the backend broke":
    previously both arrived as content and the stream still closed with
    ``finish_reason: "stop"``, so every downstream harness scored an outage as a
    low-quality generation.

    Mirrors the app-level envelope in ``src/api/__init__.py`` (``error`` /
    ``detail``) and the OpenAI streaming-error convention (``error.message`` /
    ``error.type``), so both kinds of client can key off it.
    """
    if status_code == 503:
        message = f"503 service unavailable: {message}"
    return "data: " + json.dumps(
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code,
            },
            "detail": message,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        }
    ) + "\n\n"


@dataclass(frozen=True)
class _OpenAIContentParts:
    text: str
    image_base64: str | None = None


def _parse_image_data_url(url: str) -> str:
    header, sep, payload = url.partition(",")
    header_l = header.lower()
    if sep != "," or not header_l.startswith("data:image/") or ";base64" not in header_l:
        raise ValueError(
            "OpenAI image_url content must use a data:image/...;base64 URL"
        )
    payload = payload.strip()
    try:
        b64decode(payload, validate=True)
    except (Base64Error, ValueError) as exc:
        raise ValueError("OpenAI image_url content contains invalid base64") from exc
    return payload


def _extract_openai_content(content: str | list | None, *, parse_images: bool) -> _OpenAIContentParts:
    """Extract text and, when requested, one data-URL image from OpenAI content."""
    if content is None:
        return _OpenAIContentParts(text="")
    if isinstance(content, str):
        return _OpenAIContentParts(text=content)
    if not isinstance(content, list):
        return _OpenAIContentParts(text="")

    text_parts: list[str] = []
    image_base64: str | None = None
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text", "")
            if isinstance(text, str):
                text_parts.append(text)
            continue
        if part_type != "image_url" or not parse_images:
            continue

        raw_image_url = part.get("image_url")
        if isinstance(raw_image_url, dict):
            url = raw_image_url.get("url")
        else:
            url = raw_image_url
        if not isinstance(url, str) or not url:
            raise ValueError("OpenAI image_url content must include image_url.url")
        if image_base64 is not None:
            raise ValueError("Only one OpenAI image_url part is supported per request")
        image_base64 = _parse_image_data_url(url)

    return _OpenAIContentParts(text=" ".join(text_parts), image_base64=image_base64)


def _extract_text(content: str | list | None) -> str:
    """Extract text from OpenAI content field (string or multipart array)."""
    return _extract_openai_content(content, parse_images=False).text


def _history_message_dict(message: OpenAIMessage) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": _extract_text(message.content) or "",
    }
    if message.tool_calls:
        data["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    if message.name:
        data["name"] = message.name
    return data


def _tool_function(tool: dict[str, Any]) -> dict[str, Any] | None:
    if tool.get("type") == "function":
        func = tool.get("function")
        return func if isinstance(func, dict) else None
    if "name" in tool:
        return tool
    return None


def _tool_choice_name(tool_choice: str | dict[str, Any] | None) -> str | None:
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    func = tool_choice.get("function")
    if isinstance(func, dict) and isinstance(func.get("name"), str):
        return func["name"]
    if isinstance(tool_choice.get("name"), str):
        return tool_choice["name"]
    return None


def _format_tool_call(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    func = func if isinstance(func, dict) else {}
    name = func.get("name") or tool_call.get("name") or "unknown_tool"
    args = func.get("arguments")
    if isinstance(args, (dict, list)):
        args_text = json.dumps(args, sort_keys=True)
    elif isinstance(args, str) and args:
        args_text = args
    else:
        args_text = "{}"
    call_id = tool_call.get("id")
    prefix = f"{call_id}: " if call_id else ""
    return f"{prefix}{name}({args_text})"


def _format_native_tools_for_repl(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> str | None:
    if not tools:
        return None
    choice = _tool_choice_name(tool_choice)
    if choice == "none":
        return None

    lines = [
        "OpenAI native tools were supplied by the caller.",
        "Use the existing REPL bridge to execute function tools as Python code:",
        '  result = CALL("tool_name", arg=value)',
        "Do not invent tool results; call the tool before FINAL when the answer depends on it.",
    ]
    if choice and choice not in {"auto", "none"}:
        lines.append(f"Tool choice policy: {choice}.")
    lines.append("Available function tools:")

    added = 0
    for tool in tools:
        func = _tool_function(tool)
        if not func:
            continue
        name = func.get("name")
        if not isinstance(name, str) or not name:
            continue
        desc = func.get("description")
        params = func.get("parameters")
        suffix = f" - {desc}" if isinstance(desc, str) and desc else ""
        lines.append(f"- {name}{suffix}")
        if isinstance(params, dict) and params:
            lines.append(f"  parameters: {json.dumps(params, sort_keys=True)}")
        added += 1

    if added == 0:
        return None
    return "\n".join(lines)


def _context_parts_from_history(
    history_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> list[str]:
    context_parts: list[str] = []
    for msg in history_messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        tool_calls = msg.get("tool_calls")
        if role == "tool":
            label = msg.get("name") or msg.get("tool_call_id") or "tool"
            if content:
                context_parts.append(f"Tool result {label}: {content}")
            continue
        role_label = role.capitalize()
        if content:
            context_parts.append(f"{role_label}: {content}")
        if isinstance(tool_calls, list) and tool_calls:
            calls = "; ".join(
                _format_tool_call(tc) for tc in tool_calls if isinstance(tc, dict)
            )
            if calls:
                context_parts.append(f"{role_label} tool_calls: {calls}")

    native_tools = _format_native_tools_for_repl(tools, tool_choice)
    if native_tools:
        context_parts.append(native_tools)
    return context_parts


def _executed_tool_metadata(repl: Any | None) -> dict[str, Any]:
    """Return request-local internal REPL tool telemetry for OpenAI metadata."""
    if repl is None:
        return {"tools_used": 0, "tools_called": []}

    invocations = list(getattr(repl, "_invoked_tools", None) or [])
    tools_called: list[str] = []
    for invocation in invocations:
        name = getattr(invocation, "tool_name", None) or getattr(invocation, "name", None)
        if isinstance(name, str) and name:
            tools_called.append(name)

    try:
        repl_count = int(getattr(repl, "_tool_invocations", 0) or 0)
    except (TypeError, ValueError):
        repl_count = 0
    return {
        "tools_used": max(repl_count, len(tools_called)),
        "tools_called": tools_called,
    }


def _apply_openai_tool_contract_metadata(
    meta: dict[str, Any],
    *,
    request_tools: list[dict[str, Any]] | None,
    repl: Any | None,
) -> dict[str, Any]:
    tool_meta = _executed_tool_metadata(repl)
    if request_tools is not None:
        meta["native_tool_contract"] = "internal_repl_execution"
        meta["response_tool_calls"] = "not_emitted"
    if request_tools is not None or tool_meta["tools_used"]:
        meta.update(tool_meta)
    return meta


# ── HS-4 P0.2: typed request keys ────────────────────────────────────────────
_REQUEST_KEY_FIELDS = (
    "x_session_id", "x_client_class", "x_user_id", "x_memory", "x_tool_mode", "x_escalation"
)


def _request_keys(request: OpenAIChatRequest) -> dict[str, str]:
    """The typed HS-4 keys the caller actually sent (validated by the model).

    A role override the caller sent (x_force_role, x_force_model, x_orchestrator_role)
    is echoed too, verbatim and only when set, so the inference tap records the pin a
    call carried. Without it a harness verify cannot tell a pinned call from an
    unpinned one (HS-19a S4-no-force-pin read request_keys that never held the key).
    A request without an override keeps exactly the keys it had before.
    """
    keys = {
        name: value
        for name in _REQUEST_KEY_FIELDS
        if (value := getattr(request, name, None)) is not None
    }
    for name in _ROLE_OVERRIDE_FIELDS:
        value = getattr(request, name, None)
        if value:
            keys[name] = value
    return keys


def _apply_request_key_metadata(meta: dict[str, Any], request_keys: dict[str, Any]) -> dict[str, Any]:
    """Echo the typed keys. Absent keys leave ``meta`` untouched (golden-pinned)."""
    if request_keys:
        meta["request_keys"] = dict(request_keys)
        if request_keys.get("x_memory") == "on":
            # Recorded, not acted on, until HS-4 P2 (same class as x_max_escalation).
            meta["memory_injection"] = "not_implemented"
    return meta


_OPENCODE_USER_AGENT_MARKER = "opencode"


def _session_guard_trigger(request: OpenAIChatRequest, user_agent: str) -> str | None:
    """Why this request must carry x_session_id, or None if it need not.

    Only agentic-shell requests are guarded: OpenCode (identified by its
    user-agent) and anything using the client-executed tool mode. Other /v1
    clients (Aider, eval harnesses, SDK scripts) are never affected.
    """
    if request.x_tool_mode == "client":
        return "x_tool_mode=client"
    if _OPENCODE_USER_AGENT_MARKER in user_agent.lower():
        return "an OpenCode user-agent"
    return None


def _enforce_client_session_guard(
    request: OpenAIChatRequest,
    http_request: Request,
    subagent_link: SubagentLink | None = None,
) -> None:
    """HS-4 P0.2 guard (flag ``v1_client_session_guard``).

    OpenCode only LOGS a plugin that fails to load, and ``OPENCODE_PURE``
    skips plugins entirely; the session-stamping plugin is what sends
    ``x_session_id``. Refusing here turns a silently missing plugin into a
    visible 422 instead of an unkeyed session.

    HS-16 (only when ``subagent_link`` is given, i.e. flag ``v1_subagent_link``
    is on): a header-resolved session id satisfies the guard for every trigger
    EXCEPT an OpenCode user-agent. A header gives identity, not
    ``x_tool_mode``, so a plugin-less OpenCode turn keeps its 422 rather than
    silently falling into REPL-bridge mode.
    """
    if request.x_session_id is not None:
        return
    from src.features import features as _features

    if not getattr(_features(), "v1_client_session_guard", False):
        return
    user_agent = http_request.headers.get("user-agent", "")
    trigger = _session_guard_trigger(request, user_agent)
    if trigger is None:
        return
    if (
        subagent_link is not None
        and subagent_link.session_id is not None
        and _OPENCODE_USER_AGENT_MARKER not in user_agent.lower()
    ):
        return
    raise HTTPException(
        status_code=422,
        detail=(
            f"x_session_id is required for requests with {trigger} "
            "(is the epyc-orchestrator session plugin loaded?). "
            "Disable with ORCHESTRATOR_V1_CLIENT_SESSION_GUARD=0."
        ),
    )


def _resolve_subagent_link(
    request: OpenAIChatRequest, http_request: Request, *, observe: bool = True
) -> SubagentLink | None:
    """HS-19a stage 1 (flag ``v1_subagent_link``): resolve the parent link.

    Returns None with the flag off, so the route is byte-identical to before.
    Raises 422 for malformed or spoofed ids (see ``v1_subagent_link``).
    """
    from src.features import features as _features

    if not getattr(_features(), "v1_subagent_link", False):
        return None
    return resolve_subagent_link(
        body_session_id=request.x_session_id,
        body_parent_session_id=request.x_parent_session_id,
        body_agent_name=request.x_agent_name,
        headers=http_request.headers,
        observe=observe,
    )


# ── HS-4 P0.1: client-executed tool mode ─────────────────────────────────────
_CLIENT_FINISH_REASONS = frozenset({"stop", "length", "content_filter"})


def _client_mode_messages(messages: list[OpenAIMessage]) -> list[dict[str, Any]]:
    """Structured history for the backend: roles, tool_calls and tool results kept.

    Multipart text is flattened to a string; image parts are refused (the
    client-mode backend path is text-only for now).
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            if any(
                isinstance(part, dict) and part.get("type") not in (None, "text")
                for part in content
            ):
                raise ValueError(
                    "x_tool_mode='client' supports text content only; "
                    "remove image parts or use the default tool mode"
                )
            content = _extract_text(content)
        data: dict[str, Any] = {"role": message.role, "content": content}
        if message.tool_calls:
            data["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            data["tool_call_id"] = message.tool_call_id
        if message.name:
            data["name"] = message.name
        out.append(data)
    return out


def _normalise_client_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI response shape: id, type=function, function{name, arguments:str}.

    A backend tool call without a function name is a backend failure, not a
    droppable item: skipping it could turn a tool turn into an empty "stop".
    It raises, which the route maps to 502 / a terminal SSE error event.
    """
    normalised: list[dict[str, Any]] = []
    for position, call in enumerate(tool_calls):
        func = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = func.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"backend returned tool call #{position} without a function name: "
                f"{json.dumps(call, default=str)[:200]}"
            )
        args = func.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        elif not isinstance(args, str):
            args = "{}"
        call_id = call.get("id")
        normalised.append(
            {
                "id": call_id if isinstance(call_id, str) and call_id else f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return normalised


def _run_client_tool_completion(
    primitives: Any,
    request: OpenAIChatRequest,
    messages: list[dict[str, Any]],
    *,
    role: str | Role,
    sampling_kwargs: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str, OpenAIUsage]:
    """One backend chat-completions call.

    Returns (content, tool_calls, finish_reason, usage); ``usage`` carries the
    backend's own token counts (see ``_client_usage``).

    Routing: ``role`` is the SAME resolved role the default mode would use
    (x_force_role > x_force_model > x_orchestrator_role > model alias). No REPL and
    no escalation HERE: with flag ``v1_escalation`` the route applies /chat's
    post-answer hooks to the returned answer (``v1_escalation.escalate_answer``).

    Output size: ``llm_call``'s ``output_cap`` (8192-char truncation) does NOT
    apply in client mode; ``max_tokens`` is the only bound.
    """
    result = primitives.chat_completion_call(
        messages,
        role=role,
        tools=request.tools,
        tool_choice=request.tool_choice,
        n_tokens=request.max_tokens,
        **sampling_kwargs,
    )
    tool_calls = _normalise_client_tool_calls(list(result.get("tool_calls") or []))
    content = str(result.get("content") or "")
    if tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = str(result.get("finish_reason") or "stop")
        if finish_reason not in _CLIENT_FINISH_REASONS:
            finish_reason = "stop"
    return content, tool_calls, finish_reason, _client_usage(result.get("usage"))


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _client_usage(backend_usage: Any) -> OpenAIUsage:
    """HS-4 P0.4: OpenAI ``usage`` from the backend's OWN counts.

    ``prompt_tokens`` / ``completion_tokens`` / ``cached_tokens`` are what
    llama-server reported for this call (the same numbers the inference tap
    records as ``server_terminal``). A count the server did not report is 0 —
    never re-tokenized or estimated, so a consumer gating on ``>= 1`` refuses
    an unmeasured turn instead of accepting a guess.
    """
    usage = backend_usage if isinstance(backend_usage, dict) else {}
    prompt_tokens = _token_count(usage.get("prompt_tokens"))
    completion_tokens = _token_count(usage.get("completion_tokens"))
    cached_tokens = _token_count(usage.get("cached_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        logger.warning(
            "client tool mode: backend reported no %s token count; usage reports 0",
            "prompt" if prompt_tokens is None else "completion",
        )
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    return OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=(
            {"cached_tokens": cached_tokens} if cached_tokens is not None else None
        ),
    )


def _default_usage(
    prompt: str, total_tokens: int, response_text: str, primitives: Any
) -> OpenAIUsage:
    """Usage for the default (REPL / direct) path.

    ``completion_tokens`` is the backend's decode count summed over every call
    of the request (``total_tokens_generated``). ``prompt_tokens`` is the
    server-reported prompt-token sum when the backends reported one, else the
    long-standing chars/4 estimate of the last user message (mock mode, a
    backend that reports no usage).
    """
    measured = _token_count(getattr(primitives, "total_prompt_tokens_reported", None))
    prompt_tokens = measured if measured else len(prompt) // 4
    completion_tokens = total_tokens or len(response_text) // 4
    return OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _stream_include_usage(request: OpenAIChatRequest) -> bool:
    options = request.stream_options
    return isinstance(options, dict) and options.get("include_usage") is True


def _apply_client_tool_contract_metadata(
    meta: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    meta["native_tool_contract"] = "client_execution"
    meta["response_tool_calls"] = "emitted" if tool_calls else "none"
    meta["tool_calls_emitted"] = [call["function"]["name"] for call in tool_calls]
    return meta


def _combined_prompt_with_context(prompt: str, context: str | None) -> str:
    if context:
        return f"{context}\n\nUser: {prompt}"
    return prompt


def _direct_call_prompt(prompt: str, role: str | Role, registry: Any) -> str:
    """The x_disable_repl direct call's prompt: /chat's direct-stage contract.

    A role whose backend speaks ``/completion`` gets NO server-side chat
    template, so the orchestrator must wrap the turn itself (as /chat's
    direct stage does, chat.py); a ``/v1/chat/completions`` role is templated
    by llama-server --jinja and must stay bare. The caller passes
    ``skip_suffix=True`` with this prompt: the registry's
    ``system_prompt_suffix`` appended AFTER the user's text reads as a
    continuation (a raw completion echoes it) or as a second user request.
    Before this, the direct call sent the bare question + suffix to
    ``/completion`` for every thinking-on role (the :8083 27B's
    architect_critic / coder_escalation / ingest_long_context).
    """
    from src.chat_completions_roles import chat_completions_roles

    role_name = _role_name(role)
    if role_name in chat_completions_roles():
        return prompt
    from src.api.routes.chat_utils import apply_chat_template_for_role

    return apply_chat_template_for_role(role_name, prompt, registry=registry)


_LEADING_THINK_RE = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.DOTALL)


def _split_leading_reasoning(text: str) -> tuple[str | None, str]:
    """Split a leading, CLOSED ``<think>...</think>`` block off a direct answer.

    The ``/completion`` lane does no reasoning parsing (llama-server fills
    ``reasoning_content`` only on /v1/chat/completions), so a thinking-on
    role's reasoning arrives inline. Returns ``(reasoning, answer)``; an
    unclosed or non-leading block is left in the answer untouched.
    """
    match = _LEADING_THINK_RE.match(text or "")
    if match is None:
        return None, text
    return match.group(1).strip(), text[match.end():]


def _last_call_reasoning(primitives: Any) -> str | None:
    """RI-23: the server-split ``reasoning_content`` of the primitives' last call, or None.

    Present only when ``thinking_roles_chat_lane`` is on and the call ran on the
    /v1/chat/completions lane (the primitives layer records it in the inference meta only
    then), so flag-off responses are unchanged. Read immediately after the call, in the
    same context, before anything else can make another call.
    """
    getter = getattr(primitives, "get_last_inference_meta", None)
    if not callable(getter):
        return None
    try:
        meta = getter()
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    reasoning = meta.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) and reasoning.strip() else None


def _sampling_kwargs(request: OpenAIChatRequest) -> dict[str, Any]:
    """Return only caller-explicit sampling controls for downstream inference."""
    explicit_fields = getattr(request, "model_fields_set", set())
    kwargs: dict[str, Any] = {}
    if "temperature" in explicit_fields:
        kwargs["temperature"] = request.temperature
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k
    return kwargs


def _unsupported_vision_sampling_field(request: OpenAIChatRequest) -> str | None:
    """Return the first explicit sampling control the vision path cannot honor."""
    explicit_fields = getattr(request, "model_fields_set", set())
    for field in ("temperature", "top_p", "top_k", "seed", "max_tokens"):
        if field not in explicit_fields:
            continue
        # Explicit null on optional controls has no effect and is not forwarded.
        if getattr(request, field) is None:
            continue
        return field
    return None


def _sampling_metadata(sampling_kwargs: dict[str, Any]) -> dict[str, Any]:
    if not sampling_kwargs:
        return {}
    return {"sampling": dict(sorted(sampling_kwargs.items()))}


def _role_name(role: str | Role) -> str:
    return role.value if isinstance(role, Role) else str(role)


def _plan_escalation(
    request: OpenAIChatRequest,
    role: object,
    *,
    flag_on: bool,
    image_input: bool,
    request_keys: dict[str, Any],
) -> V1EscalationPlan | None:
    """TE-1: None unless flag ``v1_escalation`` is on or x_escalation was sent."""
    plan = plan_v1_escalation(
        flag_on=flag_on,
        requested=request.x_escalation,
        role=role,
        role_override=any(getattr(request, name, None) for name in _ROLE_OVERRIDE_FIELDS),
        image_input=image_input,
    )
    if plan is not None:
        plan.base_trace_keys = dict(request_keys)
    return plan


def _escalated_client_usage(
    usage: OpenAIUsage | None, plan: V1EscalationPlan | None
) -> OpenAIUsage | None:
    """Client-mode usage plus the escalation calls' own server counts (TE-1).

    The default path already sums every call of the request; client mode reports
    the one backend call, so an escalated answer adds its calls here.
    """
    if usage is None or plan is None or not plan.fired:
        return usage
    extra_prompt, extra_completion = plan.usage_delta()
    prompt_tokens = usage.prompt_tokens + extra_prompt
    completion_tokens = usage.completion_tokens + extra_completion
    return OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=usage.prompt_tokens_details,
    )


def _escalation_stage(
    *,
    client_mode: bool,
    client_tool_calls: list[dict[str, Any]],
    image_input: bool,
    disable_repl: bool,
    repl_final_answered: bool,
) -> str | None:
    """Which /chat post-answer stage a finished /v1 answer maps to (TE-1).

    Client mode and x_disable_repl are one direct completion -> /chat's direct
    stage; a FINAL answer of the REPL bridge -> /chat's REPL stage. A tool-call
    turn, image input and an unfinished REPL loop are not answers: no stage.
    """
    if client_mode:
        return None if client_tool_calls else STAGE_DIRECT
    if image_input:
        return None
    if disable_repl:
        return STAGE_DIRECT
    return STAGE_REPL if repl_final_answered else None


async def _escalate_v1_answer(
    plan: V1EscalationPlan | None,
    stage: str | None,
    *,
    answer: str,
    question: str,
    direct_prompt: str,
    primitives: Any,
    state: AppState,
    chat_id: str,
) -> str:
    """Run /chat's post-answer hooks off the event loop; no-op when not enabled."""
    if plan is None or not plan.enabled or stage is None or primitives is None:
        return answer
    return await asyncio.to_thread(
        escalate_answer,
        plan,
        stage=stage,
        answer=answer,
        question=question,
        direct_prompt=direct_prompt,
        primitives=primitives,
        state=state,
        task_id=chat_id,
    )


async def _run_openai_vision_completion(
    *,
    prompt: str,
    context: str | None,
    image_base64: str,
    role: str | Role,
    primitives: Any,
    state: AppState,
    task_id: str,
) -> str:
    from src.api.models import ChatRequest
    from src.api.routes.chat_vision import _handle_vision_request

    role_id = _role_name(role)
    force_server = role_id if role_id in {"worker_vision", "vision_escalation"} else None
    vision_prompt = _combined_prompt_with_context(prompt or "Describe the image.", context)
    vision_request = ChatRequest(
        prompt=vision_prompt,
        mock_mode=False,
        real_mode=True,
        role=role_id,
        image_base64=image_base64,
    )
    return await _handle_vision_request(
        vision_request,
        primitives,
        state,
        task_id=task_id,
        force_server=force_server,
    )


COMPATIBILITY_MODEL_ALIASES = ("orchestrator", "architect", "worker")

# `model` values an OpenAI client sends that mean "let the orchestrator route":
# they resolve to the frontdoor. This is the `model`-field path only (HS-OD-7
# leaves it alone); the explicit override fields take ROLES, not model names.
FRONTDOOR_MODEL_ALIASES = ("orchestrator", "gpt-4", "gpt-3.5-turbo", "claude-3")

# HS-OD-7: the compatibility aliases /v1/models advertises are legal override
# values, so they must resolve to the role they stand for. `worker` shares
# `worker_general`'s server URL byte-for-byte; the other two resolved to NO
# backend before this map existed (they died at `server_urls.get(role, "")`).
_COMPATIBILITY_ALIAS_ROLES: dict[str, Role] = {
    "orchestrator": Role.FRONTDOOR,
    "architect": Role.ARCHITECT_GENERAL,
    "worker": Role.WORKER_GENERAL,
}

# The explicit role-override fields, highest precedence first (HS-OD-3).
# `x_force_model` is the deprecated alias of `x_force_role`; the request model
# already refused the two-different-values case with a 422.
_ROLE_OVERRIDE_FIELDS = ("x_force_role", "x_force_model", "x_orchestrator_role")


def _canonical_role_name(role: str) -> str:
    canonical = normalize_ingress_role(role)
    if isinstance(canonical, Role):
        return canonical.value
    return str(canonical)


def _servable_role_names() -> set[str]:
    """Role names a request may be routed to.

    The union of what ``/v1/models`` lists (live stack truth, canonicalised)
    and the role->server map the backend lookup itself consults
    (``server_urls.get(role, "")`` in ``llm_primitives/inference.py``). A
    value outside this set would have died at that lookup; a value inside it
    resolves exactly as before.
    """
    from src.config import get_config

    names = set(available_roles())
    try:
        names.update(get_config().server_urls.as_dict())
    except Exception as exc:  # degraded config: /v1/models truth still applies
        logger.debug("Could not load server_urls for role validation: %s", exc)
    return names


def normalize_override_role(value: str) -> object:
    """Normalise an explicit role-override value (never the ``model`` field).

    Ingress aliases (``worker_coder`` -> ``worker_general``) and legacy Role
    aliases (``coder`` -> ``coder_escalation``) go through
    ``normalize_ingress_role`` exactly as before; the ``/v1/models``
    compatibility aliases resolve to the role they advertise.
    """
    role = normalize_ingress_role(value)
    if isinstance(role, str):
        return _COMPATIBILITY_ALIAS_ROLES.get(role, role)
    return role


def _resolve_role_override(request: OpenAIChatRequest) -> object | None:
    """Return the normalised, validated role override, or None if none was set.

    Precedence: x_force_role > x_force_model (deprecated alias) >
    x_orchestrator_role. Every override is validated AFTER normalisation
    against ``_servable_role_names()``; an unknown value is refused with a 422
    naming the field and pointing at ``/v1/models`` (HS-OD-7) instead of
    reaching the backend lookup as a silent miss.
    """
    if request.x_force_model:
        logger.warning(
            "x_force_model is deprecated on /v1/chat/completions; send the role as "
            "x_force_role instead (HS-OD-3). value=%r",
            request.x_force_model,
        )
    for field in _ROLE_OVERRIDE_FIELDS:
        value = getattr(request, field)
        if not value:
            continue
        role = normalize_override_role(value)
        role_name = role.value if isinstance(role, Role) else str(role)
        if role_name not in _servable_role_names():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{field}={value!r} does not name a servable role; legal values are "
                    "the ids listed by GET /v1/models (HS-OD-7)"
                ),
            )
        return role
    return None


def _degraded_available_roles() -> list[str]:
    """Return degraded concrete roles when generated stack priors are absent.

    Concrete live roles intentionally do not fall back to stack_manifest
    constants here; those are launch inputs, not the /v1/models truth source.
    ``available_roles()`` still exposes compatibility aliases in degraded mode.
    """
    return []


def _ordered_live_role_ids(records: dict[str, dict]) -> list[str]:
    return [
        role
        for role, _record in sorted(
            records.items(),
            key=lambda item: (
                0 if item[0] == "frontdoor" else 1,
                stack_prior_primary_port(stack_prior_serving(item[1])) or 1_000_000,
                item[0],
            ),
        )
    ]


def _live_stack_role_ids() -> list[str]:
    """Read deployed role IDs from the generated stack-priors contract."""
    try:
        records = live_stack_role_records()
    except Exception as exc:
        logger.debug("Could not load stack priors for OpenAI models list: %s", exc)
        return []

    return _ordered_live_role_ids(records)


def available_roles() -> list[str]:
    """Return OpenAI-compatible model IDs from live stack truth plus aliases."""
    role_ids = [_canonical_role_name(role) for role in (_live_stack_role_ids() or _degraded_available_roles())]
    return list(dict.fromkeys([*COMPATIBILITY_MODEL_ALIASES, *role_ids]))


def _model_info(role: str) -> OpenAIModelInfo:
    """Model entry carrying the role's live per-request context length.

    Clients (opencode) size compaction from this instead of a hand-edited
    config limit that silently diverges from the server's -c/-np/--kv-unified.
    """
    context_length = None
    try:
        from src.backends.context_limits import get_context_limit_resolver

        limit = get_context_limit_resolver().limit_for_role(role)
        if limit is not None:
            context_length = int(limit.per_request_n_ctx)
    except Exception:
        logger.debug("context_length lookup failed for %s", role, exc_info=True)
    return OpenAIModelInfo(id=role, context_length=context_length, max_model_len=context_length)


@router.get("/models", response_model=OpenAIModelsResponse, response_model_exclude_none=True)
async def list_models() -> OpenAIModelsResponse:
    """List available models (roles) in OpenAI format, with context_length."""
    roles = available_roles()
    infos = await asyncio.to_thread(lambda: [_model_info(role) for role in roles])
    return OpenAIModelsResponse(data=infos)


def _execute_repl_turn(repl: REPLEnvironment, code: str) -> Any:
    """Execute one /v1 REPL turn, rescuing an unquoted one-word ``FINAL(OK)``.

    This loop never feeds ``last_error`` back into the next root prompt, so a
    ``FINAL(OK)`` NameError replayed identically every turn and the request
    returned ``""`` with ``finish_reason=stop`` (bare "OK"/"yes"/"Done" replies
    were dropped while "hello" — emitted quoted — survived). See
    :func:`rescue_bare_name_final` for the exact, narrow trigger.
    """
    result = repl.execute(code)
    if not result.is_final:
        rescued = rescue_bare_name_final(code, result.error)
        if rescued is not None:
            logger.info("Bare-name FINAL rescue: %r -> %r", code.strip(), rescued)
            result = repl.execute(rescued)
    return result


@router.post("/chat/completions", response_model=None)
async def openai_chat_completions(
    request: OpenAIChatRequest,
    http_request: Request,
    state: AppState = Depends(dep_app_state),
):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming modes.
    The 'model' field maps to orchestrator roles:
    - orchestrator: Auto-routing via frontdoor
    - frontdoor: Direct to frontdoor
    - coder: Direct to coder specialist
    - etc.

    For Aider integration:
    - Configure ~/.aider.conf.yml with openai-api-base: http://localhost:8000/v1
    - Aider will use this endpoint for all LLM calls
    """

    # HS-16 lifecycle control packet. It is intentionally handled before prompt
    # parsing, role resolution, scheduling, or inference. The feature flag keeps
    # the release behavior opt-in alongside the observed session registry.
    if request.x_session_final:
        # Resolve identity and precedence without touching the live tree: a final
        # packet may refer to an unknown or already-expired session.
        subagent_link = _resolve_subagent_link(request, http_request, observe=False)
        if subagent_link is not None:
            if subagent_link.session_id is None:
                raise HTTPException(status_code=422, detail="x_session_final requires a session id")
            get_registry().end(
                subagent_link.session_id,
                event=request.x_session_end_event or "final",
            )
        return OpenAIChatResponse(
            model=request.model,
            choices=[OpenAIChoice(message=OpenAIMessage(role="assistant", content=""), finish_reason="stop")],
            usage=OpenAIUsage(),
        )

    # Extract the last user message as the prompt
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    try:
        prompt_parts = _extract_openai_content(user_messages[-1].content, parse_images=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    prompt = prompt_parts.text

    if (
        request.x_disable_repl
        and request.x_tool_mode != "client"
        and _format_native_tools_for_repl(request.tools, request.tool_choice)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "tools cannot be used with x_disable_repl=true because no tool executor is "
                "available"
            ),
        )

    if prompt_parts.image_base64:
        unsupported_sampling_field = _unsupported_vision_sampling_field(request)
        if unsupported_sampling_field is not None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{unsupported_sampling_field}' is not supported for image (vision) "
                    "requests"
                ),
            )

    # Build conversation context from message history
    # B2: Apply context compression on structured messages before flattening.
    # Runs once here, so BOTH the streaming and non-streaming branches below
    # share the same (visible) fail-open fallback.
    history_messages = list(request.messages[:-1])
    history_messages_dicts = _compressed_history_dicts(history_messages)

    context_parts = _context_parts_from_history(
        history_messages_dicts,
        request.tools,
        request.tool_choice,
    )
    context = "\n\n".join(context_parts) if context_parts else None

    # Map model to role — x_force_role > x_force_model (deprecated alias) >
    # x_orchestrator_role > model field. Overrides are validated (422) in
    # _resolve_role_override; the `model` field keeps its own alias handling.
    role = _resolve_role_override(request)
    if role is None:
        if request.model in FRONTDOOR_MODEL_ALIASES:
            role = Role.FRONTDOOR
        else:
            role = normalize_ingress_role(request.model)

    # Parked role (GPU lent to AutoKernel, src/runtime/gpu_window.py): an explicit
    # 503 role_parked (app-level handler) before any work, and a preempt request
    # so this real request starts the drain. Not parked = one cached stat.
    from src.runtime import gpu_window

    gpu_window.refuse_if_parked(
        _role_name(role),
        request_id=http_request.headers.get("x-request-id"),
        caller={"source": "v1_chat_completions", "role": _role_name(role)},
    )

    # Escalation cap and REPL disable flags — pass through to metadata
    max_escalation = request.x_max_escalation
    disable_repl = request.x_disable_repl
    sampling_kwargs = _sampling_kwargs(request)

    # HS-4 P0.2 typed keys; HS-4 P0.1 client-executed tool mode. Routing above
    # is shared: client mode changes WHO executes tools, never which role runs.
    request_keys: dict[str, Any] = _request_keys(request)
    client_mode = request.x_tool_mode == "client"
    # HS-19a stage 1: record-only parent link (None while the flag is off).
    subagent_link = _resolve_subagent_link(request, http_request)
    if subagent_link is not None:
        request_keys = subagent_link.request_keys(request_keys)
    _enforce_client_session_guard(request, http_request, subagent_link)
    client_messages: list[dict[str, Any]] = []
    if client_mode:
        if prompt_parts.image_base64:
            raise HTTPException(
                status_code=400,
                detail="x_tool_mode='client' does not support image input yet",
            )
        try:
            client_messages = _client_mode_messages(request.messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    if subagent_link is not None:
        log_subagent_link(
            getattr(state, "progress_logger", None),
            subagent_link,
            chat_id=chat_id,
            user_id=request.x_user_id,
        )

    # Determine if we should use real inference
    # Real mode requires: registry loaded AND mock_mode disabled via env
    from src.features import features
    from src.config import get_config

    f = features()
    use_real_mode = (
        state.registry is not None and not f.mock_mode  # Respect mock_mode feature flag
    )
    # TE-1: None (nothing changes) unless flag v1_escalation is on or x_escalation was sent.
    escalation_plan = _plan_escalation(
        request,
        role,
        flag_on=bool(getattr(f, "v1_escalation", False)),
        image_input=bool(prompt_parts.image_base64),
        request_keys=request_keys,
    )

    # Build real primitives with server_urls (matching /chat endpoint pattern)
    primitives = None
    if use_real_mode:
        try:
            from src.llm_primitives import LLMPrimitives

            server_urls = get_config().server_urls.as_dict()
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
                registry=state.registry,
                health_tracker=state.health_tracker,
                admission_controller=getattr(state, "admission", None),
            )
        except Exception as e:
            logger.warning("Failed to create LLMPrimitives: %s", e)
            primitives = None
        if primitives is not None and request_keys:
            primitives.set_request_trace_keys(request_keys)

    if request.stream:
        # Streaming mode with real orchestration
        async def generate_stream() -> AsyncGenerator[str, None]:
            start_time = time.perf_counter()
            total_tokens = 0
            response_text = ""
            response_reasoning: str | None = None
            finish_reason = "stop"
            client_tool_calls: list[dict[str, Any]] = []
            client_usage: OpenAIUsage | None = None

            if not use_real_mode:
                # Mock mode fallback
                mock_response = f"[MOCK] Processed via {role}: {prompt[:100]}..."
                for i, char in enumerate(mock_response):
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": char}
                                if i > 0
                                else {"role": "assistant", "content": char},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                response_text = mock_response
            else:
                # Real orchestration with streaming
                repl_for_metadata: REPLEnvironment | None = None
                if primitives is None:
                    # Mirrors the 503 the non-streaming path raises for the same
                    # condition. This previously streamed the message as assistant
                    # content and then closed with finish_reason "stop", so a
                    # misconfigured server was indistinguishable from a model that
                    # had answered "LLM primitives not initialized".
                    yield _sse_error_event(
                        chat_id=chat_id,
                        created=created,
                        model=request.model,
                        message="LLM primitives not initialized — check server_urls config",
                        error_type="primitives_unavailable",
                        status_code=503,
                    )
                    yield "data: [DONE]\n\n"
                    return

                if primitives:
                    # Build combined context
                    combined_context = _combined_prompt_with_context(prompt, context)
                    repl_final_answered = False

                    if client_mode:
                        # HS-4 P0.1: the backend call is buffered (tool calls
                        # arrive whole); content and tool-call deltas are then
                        # replayed in OpenAI chunk format below.
                        try:
                            response_text, client_tool_calls, finish_reason, client_usage = (
                                _run_client_tool_completion(
                                    primitives, request, client_messages,
                                    role=role, sampling_kwargs=sampling_kwargs,
                                )
                            )
                            _raise_admission_denied_text(response_text)
                        except (AdmissionDenied, ContentionDenied) as e:
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=str(e), error_type=("admission_denied" if isinstance(e, AdmissionDenied)
                                                        else "contention_denied"),
                                status_code=503,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        except ContextOverflowError as e:
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=str(e), error_type="context_overflow",
                                status_code=503 if e.retryable else 413,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as e:
                            logger.exception(
                                "Streaming client-tool call failed for role %s (chat %s)",
                                role, chat_id,
                            )
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=f"Backend failed: {e}",
                                error_type="backend_error", status_code=502,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        total_tokens = primitives.total_tokens_generated
                    elif prompt_parts.image_base64:
                        try:
                            response_text = await _run_openai_vision_completion(
                                prompt=prompt,
                                context=context,
                                image_base64=prompt_parts.image_base64,
                                role=role,
                                primitives=primitives,
                                state=state,
                                task_id=chat_id,
                            )
                            _raise_admission_denied_text(response_text)
                        except (AdmissionDenied, ContentionDenied) as e:
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=str(e), error_type=("admission_denied" if isinstance(e, AdmissionDenied)
                                                        else "contention_denied"),
                                status_code=503,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        except ContextOverflowError as e:
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=str(e), error_type="context_overflow",
                                status_code=503 if e.retryable else 413,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as e:
                            logger.exception(
                                "Streaming vision request failed for role %s (chat %s)",
                                role, chat_id,
                            )
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=f"Vision request failed: {e}",
                                error_type="backend_error", status_code=502,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        total_tokens = primitives.total_tokens_generated
                    elif disable_repl:
                        # Direct LLM call — no REPL, no code execution.
                        # /chat's direct-stage prompt contract (_direct_call_prompt).
                        try:
                            response_text = primitives.llm_call(
                                _direct_call_prompt(
                                    combined_context, role,
                                    getattr(state, "registry", None),
                                ),
                                role=role,
                                n_tokens=request.max_tokens,
                                skip_suffix=True,
                                **sampling_kwargs,
                            )
                            _raise_admission_denied_text(response_text)
                        except (AdmissionDenied, ContentionDenied) as e:
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=str(e), error_type=("admission_denied" if isinstance(e, AdmissionDenied)
                                                        else "contention_denied"),
                                status_code=503,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        except ContextOverflowError as e:
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=str(e), error_type="context_overflow",
                                status_code=503 if e.retryable else 413,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as e:
                            logger.exception(
                                "Streaming direct call failed for role %s (chat %s)",
                                role, chat_id,
                            )
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=f"Direct call failed: {e}",
                                error_type="backend_error", status_code=502,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        # In-band guard: llm_call returns "[ERROR: ...]" rather
                        # than raising on backend failure. Emit the terminal
                        # error event instead of streaming it as content.
                        inband_error = inband_error_text(response_text)
                        if inband_error is not None:
                            logger.warning(
                                "Streaming backend in-band failure for role %s (chat %s): %s",
                                role, chat_id, inband_error,
                            )
                            yield _sse_error_event(
                                chat_id=chat_id, created=created, model=request.model,
                                message=f"Backend failed: {inband_error}",
                                error_type="backend_error", status_code=502,
                            )
                            yield "data: [DONE]\n\n"
                            return
                        response_reasoning, response_text = _split_leading_reasoning(
                            response_text
                        )
                        if response_reasoning is None:
                            # RI-23: chat lane — the server already split it off.
                            response_reasoning = _last_call_reasoning(primitives)
                        total_tokens = primitives.total_tokens_generated
                    else:
                        # Create REPL environment
                        repl = REPLEnvironment(
                            context=combined_context,
                            llm_primitives=primitives,
                            tool_registry=state.tool_registry,
                            script_registry=state.script_registry,
                            role=role,
                            **_repl_memrl_kwargs(state),
                        )
                        repl_for_metadata = repl

                        # Run orchestration loop (simplified for streaming)
                        max_turns = request.max_tokens // 500 if request.max_tokens else 3
                        max_turns = min(max(max_turns, 1), 5)

                        for turn in range(max_turns):
                            repl_state = repl.get_state()
                            root_prompt = build_root_lm_prompt(
                                state=repl_state,
                                original_prompt=prompt,
                                last_output="",
                                last_error="",
                                turn=turn,
                            )

                            try:
                                code = primitives.llm_call(
                                    root_prompt,
                                    role=role,
                                    n_tokens=1024,
                                    **sampling_kwargs,
                                )
                                _raise_admission_denied_text(code)
                                turn_reasoning = _last_call_reasoning(primitives)
                                # In-band guard: "[ERROR: ...]" at start-of-answer
                                # is a backend failure, not a generation — do not
                                # extract/auto-wrap/execute it as the answer.
                                inband_error = inband_error_text(code)
                                if inband_error is not None:
                                    logger.warning(
                                        "Streaming backend in-band failure for role %s (chat %s): %s",
                                        role, chat_id, inband_error,
                                    )
                                    yield _sse_error_event(
                                        chat_id=chat_id, created=created, model=request.model,
                                        message=f"Backend failed: {inband_error}",
                                        error_type="backend_error", status_code=502,
                                    )
                                    yield "data: [DONE]\n\n"
                                    return
                                code = extract_code_from_response(code)
                                code = auto_wrap_final(code)
                            except (AdmissionDenied, ContentionDenied) as e:
                                yield _sse_error_event(
                                    chat_id=chat_id, created=created, model=request.model,
                                    message=str(e), error_type=("admission_denied" if isinstance(e, AdmissionDenied)
                                                        else "contention_denied"),
                                    status_code=503,
                                )
                                yield "data: [DONE]\n\n"
                                return
                            except ContextOverflowError as e:
                                yield _sse_error_event(
                                    chat_id=chat_id, created=created, model=request.model,
                                    message=str(e), error_type="context_overflow",
                                    status_code=503 if e.retryable else 413,
                                )
                                yield "data: [DONE]\n\n"
                                return
                            except Exception as e:
                                # Was: code = FINAL("Error during generation: ...").
                                # That fed the backend failure back through the REPL
                                # as though the model had ANSWERED with it, so it
                                # left as ordinary assistant content.
                                logger.exception(
                                    "Streaming generation failed for role %s (chat %s)",
                                    role, chat_id,
                                )
                                yield _sse_error_event(
                                    chat_id=chat_id, created=created, model=request.model,
                                    message=f"Error during generation: {e}",
                                    error_type="backend_error", status_code=502,
                                )
                                yield "data: [DONE]\n\n"
                                return

                            # Execute in REPL
                            result = _execute_repl_turn(repl, code)

                            if result.is_final:
                                response_text = result.final_answer or ""
                                # RI-23: the reasoning of the turn that answered.
                                response_reasoning = turn_reasoning
                                repl_final_answered = True
                                break
                            elif result.output:
                                response_text = result.output
                        else:
                            # Max turns reached
                            response_text = response_text or f"[Completed {max_turns} turns]"

                        total_tokens = primitives.total_tokens_generated

                    # TE-1: /chat's post-answer escalation hooks (flag v1_escalation).
                    if escalation_plan is not None and escalation_plan.enabled:
                        escalated_text = await _escalate_v1_answer(
                            escalation_plan,
                            _escalation_stage(
                                client_mode=client_mode,
                                client_tool_calls=client_tool_calls,
                                image_input=bool(prompt_parts.image_base64),
                                disable_repl=disable_repl,
                                repl_final_answered=repl_final_answered,
                            ),
                            answer=response_text,
                            question=prompt,
                            direct_prompt=combined_context,
                            primitives=primitives,
                            state=state,
                            chat_id=chat_id,
                        )
                        if escalated_text != response_text:
                            response_text = escalated_text
                            response_reasoning = None  # belonged to the replaced answer
                            finish_reason = "stop"
                        total_tokens = primitives.total_tokens_generated
                        client_usage = _escalated_client_usage(client_usage, escalation_plan)

                    first_chunk = True
                    if response_reasoning:
                        # The direct call's split-off <think> block, as one
                        # reasoning_content delta ahead of the content deltas.
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "reasoning_content": response_reasoning,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        first_chunk = False
                        if request.x_show_routing:
                            chunk["x_role"] = role
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Stream the response character by character (OpenAI format)
                    for char in response_text:
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": char}
                                    if first_chunk
                                    else {"content": char},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        first_chunk = False
                        if request.x_show_routing:
                            chunk["x_role"] = role
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # HS-4 P0.1: one delta per tool call, complete arguments.
                    for tc_index, tool_call in enumerate(client_tool_calls):
                        delta: dict[str, Any] = {
                            "tool_calls": [{"index": tc_index, **tool_call}],
                        }
                        if first_chunk:
                            delta = {"role": "assistant", "content": None, **delta}
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                        }
                        first_chunk = False
                        if request.x_show_routing:
                            chunk["x_role"] = role
                        yield f"data: {json.dumps(chunk)}\n\n"

            escalation_receipt = record_escalation(
                escalation_plan, chat_id=chat_id, request_keys=request_keys, primitives=primitives,
            )
            # Final chunk with finish_reason
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if request.x_show_routing:
                meta = {
                    "role": role,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "tokens": total_tokens,
                }
                if max_escalation:
                    meta["max_escalation"] = max_escalation
                if disable_repl:
                    meta["repl_disabled"] = True
                meta.update(_sampling_metadata(sampling_kwargs))
                if client_mode and use_real_mode:
                    _apply_client_tool_contract_metadata(meta, client_tool_calls)
                else:
                    _apply_openai_tool_contract_metadata(
                        meta,
                        request_tools=request.tools,
                        repl=locals().get("repl_for_metadata"),
                    )
                if escalation_receipt is not None:
                    meta["escalation"] = escalation_receipt
                _apply_request_key_metadata(meta, request_keys)
                final_chunk["x_orchestrator_metadata"] = meta
            yield f"data: {json.dumps(final_chunk)}\n\n"
            # stream_options.include_usage (OpenAI spec; @ai-sdk/openai-compatible
            # sends it by default from OpenCode): one extra chunk, empty choices,
            # carrying usage — after the finish_reason chunk, before [DONE].
            if _stream_include_usage(request):
                stream_usage = (
                    client_usage
                    if client_usage is not None
                    else _default_usage(prompt, total_tokens, response_text, primitives)
                )
                usage_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [],
                    "usage": stream_usage.model_dump(),
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        stream = generate_stream()
        try:
            first = await anext(stream, None)
        except BaseException:
            await stream.aclose()
            raise
        if first is not None and first.startswith("data: "):
            event = json.loads(first[6:])
            error = event.get("error", {})
            if error.get("code") == 503 and error.get("type") in {
                "admission_denied", "contention_denied", "context_overflow"
            }:
                await stream.aclose()
                return JSONResponse(
                    status_code=503,
                    content={"error": error, "detail": error["message"]},
                    headers={"Retry-After": "5", "retry-after-ms": "5000"},
                )

        async def replay_stream():
            try:
                if first is not None:
                    yield first
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()

        return StreamingResponse(
            replay_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming mode with real orchestration
        start_time = time.perf_counter()
        total_tokens = 0
        finish_reason = "stop"
        client_tool_calls: list[dict[str, Any]] = []
        client_usage: OpenAIUsage | None = None
        response_reasoning: str | None = None

        if not use_real_mode:
            # Mock mode fallback
            response_text = f"[MOCK] Processed via {role}: {prompt[:100]}..."
            repl_for_metadata = None
        else:
            # Real orchestration
            repl_for_metadata = None
            try:
                if primitives is None:
                    raise HTTPException(
                        status_code=503,
                        detail="LLM primitives not initialized — check server_urls config",
                    )

                combined_context = _combined_prompt_with_context(prompt, context)
                repl_final_answered = False

                if client_mode:
                    response_text, client_tool_calls, finish_reason, client_usage = (
                        _run_client_tool_completion(
                            primitives, request, client_messages,
                            role=role, sampling_kwargs=sampling_kwargs,
                        )
                    )
                elif prompt_parts.image_base64:
                    response_text = await _run_openai_vision_completion(
                        prompt=prompt,
                        context=context,
                        image_base64=prompt_parts.image_base64,
                        role=role,
                        primitives=primitives,
                        state=state,
                        task_id=chat_id,
                    )
                elif disable_repl:
                    # Direct LLM call — no REPL, no code execution.
                    # /chat's direct-stage prompt contract (_direct_call_prompt).
                    response_text = primitives.llm_call(
                        _direct_call_prompt(
                            combined_context, role, getattr(state, "registry", None),
                        ),
                        role=role,
                        n_tokens=request.max_tokens,
                        skip_suffix=True,
                        **sampling_kwargs,
                    )
                    # llm_call does not raise on backend failure — it returns an
                    # in-band "[ERROR: ...]" at start-of-answer (LLMPrimitives
                    # fail-open contract). Without this, that string reached the
                    # client as assistant content with HTTP 200 (HS-OD-2).
                    _raise_admission_denied_text(response_text)
                    inband_error = inband_error_text(response_text)
                    if inband_error is not None:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Backend failed: {inband_error}",
                        )
                    response_reasoning, response_text = _split_leading_reasoning(
                        response_text
                    )
                    if response_reasoning is None:
                        # RI-23: chat lane — the server already split it off.
                        response_reasoning = _last_call_reasoning(primitives)
                else:
                    repl = REPLEnvironment(
                        context=combined_context,
                        llm_primitives=primitives,
                        tool_registry=state.tool_registry,
                        script_registry=state.script_registry,
                        role=role,
                        **_repl_memrl_kwargs(state),
                    )
                    repl_for_metadata = repl

                    max_turns = request.max_tokens // 500 if request.max_tokens else 3
                    max_turns = min(max(max_turns, 1), 5)

                    response_text = ""
                    for turn in range(max_turns):
                        repl_state = repl.get_state()
                        root_prompt = build_root_lm_prompt(
                            state=repl_state,
                            original_prompt=prompt,
                            last_output="",
                            last_error="",
                            turn=turn,
                        )

                        code = primitives.llm_call(
                            root_prompt,
                            role=role,
                            n_tokens=1024,
                            **sampling_kwargs,
                        )
                        _raise_admission_denied_text(code)
                        turn_reasoning = _last_call_reasoning(primitives)
                        # Same in-band guard as the direct path: an "[ERROR: ...]"
                        # generation is a backend failure, not code to auto-wrap
                        # and execute as the model's final answer.
                        inband_error = inband_error_text(code)
                        if inband_error is not None:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Backend failed: {inband_error}",
                            )
                        code = extract_code_from_response(code)
                        code = auto_wrap_final(code)

                        result = _execute_repl_turn(repl, code)

                        if result.is_final:
                            response_text = result.final_answer or ""
                            # RI-23: the reasoning of the turn that answered.
                            response_reasoning = turn_reasoning
                            repl_final_answered = True
                            break
                        elif result.output:
                            response_text = result.output

                # TE-1: /chat's post-answer escalation hooks (flag v1_escalation).
                if escalation_plan is not None and escalation_plan.enabled:
                    escalated_text = await _escalate_v1_answer(
                        escalation_plan,
                        _escalation_stage(
                            client_mode=client_mode,
                            client_tool_calls=client_tool_calls,
                            image_input=bool(prompt_parts.image_base64),
                            disable_repl=disable_repl,
                            repl_final_answered=repl_final_answered,
                        ),
                        answer=response_text,
                        question=prompt,
                        direct_prompt=combined_context,
                        primitives=primitives,
                        state=state,
                        chat_id=chat_id,
                    )
                    if escalated_text != response_text:
                        response_text = escalated_text
                        response_reasoning = None  # belonged to the replaced answer
                        finish_reason = "stop"
                    client_usage = _escalated_client_usage(client_usage, escalation_plan)

                total_tokens = primitives.total_tokens_generated

            except HTTPException:
                # Already carries its own status — including the 503 raised a few
                # lines above for uninitialised primitives, which the old blanket
                # `except Exception` swallowed into a 200.
                raise
            except AdmissionDenied as e:
                raise HTTPException(
                    status_code=503, detail=str(e),
                    headers={"Retry-After": "5", "retry-after-ms": "5000"},
                ) from e
            except ContextOverflowError:
                # Dedicated app-level handler: 413 (too large for the role) or
                # 503 + Retry-After (shared KV pool stayed exhausted).
                raise
            except ContentionDenied:
                # Has a dedicated app-level handler (503 + Retry-After +
                # failure_provenance). Swallowing it here turned a documented
                # back-pressure signal into a model answer, so callers retried
                # nothing and the denial never showed up in error metrics.
                raise
            except Exception as e:
                # HS-OD-2: a backend failure is an upstream failure, not a
                # completion. 502 rather than 500 — this route is a gateway in
                # front of the llama.cpp fleet, and the fault is the upstream's.
                logger.exception("Backend failed for role %s (chat %s)", role, chat_id)
                raise HTTPException(
                    status_code=502, detail=f"Backend failed: {e}"
                ) from e

        elapsed = time.perf_counter() - start_time
        escalation_receipt = record_escalation(
            escalation_plan, chat_id=chat_id, request_keys=request_keys, primitives=primitives,
        )

        if client_tool_calls:
            response_message = OpenAIMessage(
                role="assistant",
                content=response_text or None,
                tool_calls=client_tool_calls,
            )
        else:
            response_message = OpenAIMessage(
                role="assistant",
                content=response_text,
                reasoning_content=response_reasoning,
            )

        if request.x_show_routing:
            response_meta = {
                "role": role,
                "elapsed_seconds": elapsed,
                **({"max_escalation": max_escalation} if max_escalation else {}),
                **({"repl_disabled": True} if disable_repl else {}),
                **_sampling_metadata(sampling_kwargs),
            }
            if client_mode and use_real_mode:
                _apply_client_tool_contract_metadata(response_meta, client_tool_calls)
            else:
                _apply_openai_tool_contract_metadata(
                    response_meta,
                    request_tools=request.tools,
                    repl=repl_for_metadata,
                )
            if escalation_receipt is not None:
                response_meta["escalation"] = escalation_receipt
            _apply_request_key_metadata(response_meta, request_keys)
        else:
            response_meta = None

        return OpenAIChatResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=response_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=(
                client_usage
                if client_usage is not None
                else _default_usage(prompt, total_tokens, response_text, primitives)
            ),
            x_orchestrator_metadata=response_meta,
        )


@router.get("/models/{model_id}", response_model=OpenAIModelInfo, response_model_exclude_none=True)
async def get_model(model_id: str) -> OpenAIModelInfo:
    """Get info for a specific model."""
    if model_id not in available_roles():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return await asyncio.to_thread(_model_info, model_id)
