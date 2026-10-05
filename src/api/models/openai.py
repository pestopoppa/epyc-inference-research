"""OpenAI-compatible models for the orchestrator API."""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer, model_validator


class OpenAIMessage(BaseModel):
    """OpenAI message format."""

    role: str = Field(..., description="Role: system, user, assistant, tool")
    content: str | list | None = Field(
        default=None,
        description="Message content: string or multipart content array "
        '(e.g. [{"type": "text", "text": "..."}, {"type": "image_url", ...}])',
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Assistant tool calls in OpenAI chat-completions format",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Tool-call id for role=tool result messages",
    )
    name: str | None = Field(
        default=None,
        description="Optional participant or tool name",
    )
    reasoning_content: str | None = Field(
        default=None,
        description="Assistant reasoning split off the answer (response only; "
        "omitted when there is none)",
    )

    @model_serializer(mode="wrap")
    def _omit_absent_reasoning(self, handler) -> dict[str, Any]:
        data = handler(self)
        if isinstance(data, dict) and data.get("reasoning_content") is None:
            data.pop("reasoning_content", None)
        return data

    @model_validator(mode="after")
    def _require_content_or_tool_call(self) -> "OpenAIMessage":
        if self.content is None and not self.tool_calls:
            raise ValueError("content is required unless assistant tool_calls are present")
        return self


# HS-OD-1: standard OpenAI body fields this API does not honour must be REFUSED
# when honouring them would have changed the output — never silently dropped.
# Pydantic's default extra='ignore' was discarding response_format without error,
# so any JSON-mode client got prose with a 200 and no diagnostic. Value-sensitive
# on purpose: an explicit no-op (n=1, penalty 0.0, response_format {"type":"text"},
# empty stop list) is accepted so SDK clients that spell out defaults keep
# working; only a request whose semantics we would silently change is refused.
# Fields with no output effect (user, metadata) stay ignored. stream_options is
# a typed field: include_usage adds the final usage chunk to a stream.
_UNHONOURED_SEMANTIC_FIELDS: dict = {
    "response_format": (
        lambda v: v is not None and not (isinstance(v, dict) and v.get("type") == "text"),
        "JSON mode is not implemented on this seam; remove response_format "
        'or send {"type": "text"}',
    ),
    "n": (lambda v: v is not None and v != 1, "only n=1 is supported"),
    "stop": (lambda v: bool(v), "stop sequences are not forwarded to the backend"),
    "logprobs": (lambda v: bool(v), "logprobs are not returned"),
    "top_logprobs": (lambda v: v is not None, "logprobs are not returned"),
    "logit_bias": (lambda v: bool(v), "logit_bias is not forwarded to the backend"),
    "presence_penalty": (
        lambda v: v not in (None, 0, 0.0),
        "sampling penalties are not forwarded to the backend",
    ),
    "frequency_penalty": (
        lambda v: v not in (None, 0, 0.0),
        "sampling penalties are not forwarded to the backend",
    ),
    "functions": (
        lambda v: bool(v),
        "legacy function calling is not supported; use tools",
    ),
    "function_call": (
        lambda v: v is not None,
        "legacy function calling is not supported; use tool_choice",
    ),
}


# HS-4 P0.2: opaque client identifiers — printable, no whitespace, bounded.
_REQUEST_KEY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@+/=-]*$"
_CLIENT_TOOL_CHOICE_STRINGS = frozenset({"auto", "none", "required"})


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="orchestrator", description="Model/role to use")
    messages: list[OpenAIMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Decode temperature. Forwarded to the backend ONLY when sent explicitly; "
        "the schema default 0.0 is NOT forwarded, so an omitted temperature uses the "
        "backend's per-role default. Explicit sampling controls are rejected on image "
        "(vision) requests because that path cannot honor them.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling override. Forwarded when set; explicit non-null values "
        "are rejected on image (vision) requests.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Orchestrator extension: llama.cpp top-k sampling override. Forwarded when "
        "set; explicit non-null values are rejected on image (vision) requests.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional deterministic decode seed. Forwarded when set; explicit non-null "
        "values are rejected on image (vision) requests.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Generation cap ONLY with x_tool_mode='client' or x_disable_repl=true. In the "
        "default REPL mode it is NOT the token budget: each REPL turn generates up to a "
        "fixed 1024 tokens and max_tokens only sets the turn count (max_tokens // 500, "
        "clamped to 1..5). Explicit values are rejected on image (vision) requests. "
        "max_completion_tokens is "
        "accepted as an alias (422 if both are sent).",
    )
    stream: bool = Field(default=False, description="Enable streaming")
    stream_options: dict[str, Any] | None = Field(
        default=None,
        description="OpenAI stream options. {'include_usage': true} appends one final chunk "
        "(empty choices) carrying `usage` before [DONE]. Client tool mode reports the "
        "backend's own prompt/completion/cached token counts. Other keys are ignored; "
        "ignored when stream is false (non-stream responses always carry usage).",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI native tool definitions. Forwarded to the backend verbatim ONLY with "
        "x_tool_mode='client'. In the default REPL mode they are rendered into the prompt as "
        "CALL() instructions for the orchestrator REPL and are never returned as tool_calls "
        "(metadata native_tool_contract='internal_repl_execution'). With x_disable_repl=true "
        "they are still rendered as prompt text but there is no REPL to execute them.",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="OpenAI tool choice policy, e.g. 'auto', 'none', 'required', or function choice. "
        "Validated (422) and forwarded to the backend ONLY with x_tool_mode='client'. In the "
        "default REPL mode 'none' suppresses the rendered tool block and any other value is "
        "rendered as advisory prompt text, not enforced.",
    )
    # Extension fields — orchestrator routing overrides
    x_orchestrator_role: str | None = Field(
        default=None,
        description="Force specific orchestrator role. Values: any role from /v1/models (e.g. "
        "'architect_general', 'worker_math'). Honoured as the backend role on the text and "
        "client-tool paths. Validated after normalisation (HS-OD-7): a value that does not "
        "name a servable role is refused with a 422 naming this field and pointing at "
        "/v1/models, instead of reaching the backend lookup. Precedence: x_force_role > "
        "x_force_model (deprecated alias) > x_orchestrator_role > model alias. On image "
        "(vision) requests only 'worker_vision'/'vision_escalation' constrain the server; any "
        "other role is ignored by the vision path.",
    )
    x_max_escalation: str | None = Field(
        default=None,
        description="Requested escalation-tier cap. Values: 'A' (frontdoor only), 'B1' (coder), "
        "'B2' (architect), 'C' (worker). METADATA ONLY on /v1 today: the value is recorded "
        "in routing metadata and NOT enforced -- role/override resolution applies no "
        "escalation cap. Escalation itself is governed by x_escalation (flag v1_escalation), "
        "never by this field. Enforcement is HS-4 P4 work; until then this field does not "
        "prevent anything.",
    )
    x_force_role: str | None = Field(
        default=None,
        description="Highest-precedence ROLE override (x_force_role > x_force_model (deprecated "
        "alias) > x_orchestrator_role > model alias). The ROLE is the contract on this seam "
        "(operator decision 2026-09-17): legal values are exactly what GET /v1/models lists, "
        "plus the ingress aliases the orchestrator already normalises. Validated after "
        "normalisation; a value that does not name a servable role is refused with a 422 "
        "naming this field (HS-OD-7). Replaces x_force_model (HS-OD-3).",
    )
    x_force_model: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="DEPRECATED alias of x_force_role -- identical behaviour and precedence "
        "(x_force_role > x_force_model > x_orchestrator_role > model alias), kept so existing "
        "callers keep working (HS-OD-3); setting it logs a warning naming the replacement. "
        "Despite the name it does NOT select a model by registry name: the "
        "value is a ROLE label, normalised and validated exactly like x_force_role, so a "
        "registry model name (e.g. 'architect_qwen2_5_72b') is not resolved and is refused "
        "with a 422. Sending both x_force_role and x_force_model with different values is "
        "refused with a 422 naming both fields. Send a role from /v1/models via x_force_role.",
    )
    x_disable_repl: bool = Field(
        default=False,
        description="Skip REPL code execution -- direct response only. With x_tool_mode='client', "
        "the client/backend remains the tool executor. Otherwise, a request that would render "
        "tool instructions is rejected (422) because no executor would be available.",
    )
    x_show_routing: bool = Field(default=False, description="Include routing metadata")
    # HS-4 P0.2 — typed session/arm keys. Each value is validated (422 on a bad
    # one), echoed into x_orchestrator_metadata["request_keys"] and stamped onto
    # the inference-tap trace. Absent keys change nothing.
    x_session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_REQUEST_KEY_ID_PATTERN,
        description="Client conversation/session id (HS-4). Validated (422 on a bad value). "
        "RECORDED ONLY: stamped onto the inference-tap trace and echoed in "
        "x_orchestrator_metadata.request_keys when x_show_routing=true. No store is keyed on "
        "it on /v1 today (HS-4 P1/P3 are future work). Its ABSENCE is refused with a 422 for "
        "x_tool_mode='client' or an OpenCode user-agent when the v1_client_session_guard "
        "flag is on.",
    )
    x_client_class: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=_REQUEST_KEY_ID_PATTERN,
        description="Self-reported client class tag for serving-call attribution. Validated and "
        "recorded only in request metadata and native serving-call records; it does not verify "
        "client identity or change routing. Omission remains unknown, with no class inferred "
        "from workload_class or process configuration.",
    )
    x_user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_REQUEST_KEY_ID_PATTERN,
        description="Client user id (HS-4). Validated (422 on a bad value). RECORDED ONLY: "
        "stamped onto the inference-tap trace and echoed in x_orchestrator_metadata."
        "request_keys when x_show_routing=true. No user profile exists on /v1 today (HS-4 P2 "
        "is future work); the value changes nothing about the response.",
    )
    x_memory: Literal["on", "off"] | None = Field(
        default=None,
        description="Memory-injection arm (HS-4). Recorded only until HS-4 P2 ships: "
        "nothing is injected on /v1 today, so 'on' is reported as "
        "memory_injection='not_implemented' in the metadata (visible only with "
        "x_show_routing=true).",
    )
    x_tool_mode: Literal["repl", "client"] | None = Field(
        default=None,
        description="Tool execution mode (HS-4 P0.1). 'repl' (default when absent): client "
        "tools are rendered into the prompt as orchestrator REPL CALL() instructions and "
        "tool_calls are never returned (with x_disable_repl=true, requests that render tool "
        "instructions are refused with 422). "
        "'client': tools, tool_choice and tool history are forwarded to the backend and "
        "tool_calls are returned for the client to execute; tool_choice is validated (422), "
        "image input is refused (400), and x_session_id may be required (422, "
        "v1_client_session_guard flag). Neither mode escalates unless the v1_escalation flag "
        "is on (see x_escalation).",
    )
    # TE-1 (UFH-13) — per-request escalation switch, read only with flag v1_escalation.
    x_escalation: Literal["auto", "off", "architect_general"] | None = Field(
        default=None,
        description="Escalation switch (TE-1; flag v1_escalation, default off). With the flag "
        "ON, a frontdoor answer passes the SAME post-answer hooks /chat applies. Client tool "
        "mode and x_disable_repl (a turn with no tool calls) get /chat's direct-stage chain: "
        "quality escalation (generation_monitor flag) then the MemRL review gate "
        "(architect_general verdict, worker_general revision on WRONG). The default REPL "
        "mode gets /chat's REPL-stage hook, the review gate, on a FINAL answer. OPT-IN: "
        "absent means no escalation and no receipt, flag on or off (so enabling the flag "
        "changes no unkeyed traffic). 'auto' keeps /chat's targets (quality escalation "
        "-> coder_escalation). 'architect_general' keeps the triggers but pins every "
        "consultant call to architect_general. 'off' serves the answer exactly as the "
        "flag-off route does, with a disabled receipt. Never applied to a role-overridden request (x_force_role / "
        "x_force_model / x_orchestrator_role), a non-frontdoor role or image input. With the "
        "flag OFF the value is validated and recorded (request_keys, metadata "
        "escalation.disabled_reason='flag_off') and nothing escalates.",
    )
    # HS-19a stage 1 — harness subagent tree. Typed Any on purpose: they are read
    # ONLY when the v1_subagent_link flag is on, and then validated (422) by
    # src/api/routes/v1_subagent_link.py with the x_session_id rules. With the
    # flag off they are ignored exactly as unknown keys were before, so no live
    # request changes (not even a type error).
    x_parent_session_id: Any = Field(
        default=None,
        description="Parent session id (string) of a harness subagent request (HS-19a). Read "
        "ONLY when the v1_subagent_link flag is on: then validated like x_session_id (422 on "
        "a bad, self-referencing or cycle-closing value), preferred over the "
        "x-parent-session-id header, and RECORDED ONLY (inference-tap request_keys and the "
        "session log). It never changes model selection. Ignored while the flag is off.",
    )
    # HS-16: control-only lifecycle request emitted by the OpenCode event hook.
    # It is acknowledged before prompt parsing and never enters inference.
    x_session_final: bool = Field(
        default=False, description="Release observed session state without inference"
    )
    x_session_end_event: Literal["final", "idle", "deleted"] | None = Field(
        default=None, description="Client lifecycle event associated with x_session_final"
    )
    x_agent_name: Any = Field(
        default=None,
        description="Harness agent name (string), e.g. OpenCode's 'general' sub-agent "
        "(HS-19a). Read ONLY when the v1_subagent_link flag is on: then validated (422 on a "
        "bad value) and RECORDED ONLY. It never changes model selection. Ignored while the "
        "flag is off.",
    )

    @model_validator(mode="after")
    def _refuse_conflicting_force_role_alias(self) -> "OpenAIChatRequest":
        # HS-OD-3: x_force_model is a deprecated alias of x_force_role. Same
        # discipline as max_tokens/max_completion_tokens -- a double supply with
        # DIFFERENT values is refused rather than silently resolved by precedence.
        if (
            self.x_force_role is not None
            and self.x_force_model is not None
            and self.x_force_role != self.x_force_model
        ):
            raise ValueError(
                f"x_force_role={self.x_force_role!r} and its deprecated alias "
                f"x_force_model={self.x_force_model!r} were both supplied with different "
                "values; send exactly one (prefer x_force_role)"
            )
        return self

    @model_validator(mode="after")
    def _validate_client_tool_choice(self) -> "OpenAIChatRequest":
        # Only client mode is strict: the default REPL bridge keeps today's
        # permissive handling byte-for-byte (HS-4 P0.1(b)).
        if self.x_tool_mode != "client" or self.tool_choice is None:
            return self
        choice = self.tool_choice
        if isinstance(choice, str):
            if choice not in _CLIENT_TOOL_CHOICE_STRINGS:
                raise ValueError(
                    f"tool_choice {choice!r} is not one of "
                    f"{sorted(_CLIENT_TOOL_CHOICE_STRINGS)} (x_tool_mode='client')"
                )
            if choice == "required" and not self.tools:
                raise ValueError("tool_choice 'required' needs a non-empty tools list")
            return self
        func = choice.get("function") if choice.get("type") == "function" else None
        name = func.get("name") if isinstance(func, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(
                "tool_choice object must be "
                '{"type": "function", "function": {"name": ...}} (x_tool_mode=\'client\')'
            )
        declared = {
            (t.get("function") or {}).get("name")
            for t in (self.tools or [])
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
        }
        if name not in declared:
            raise ValueError(f"tool_choice names undeclared tool {name!r}")
        return self

    @model_validator(mode="before")
    @classmethod
    def _alias_max_completion_tokens(cls, data):
        # OpenAI deprecated max_tokens in favour of max_completion_tokens; SDK
        # clients send either. Runs mode="before" because after validation the
        # default max_tokens=1024 is indistinguishable from an explicit one.
        if isinstance(data, dict) and data.get("max_completion_tokens") is not None:
            if data.get("max_tokens") is not None:
                raise ValueError(
                    "max_tokens and max_completion_tokens were both supplied; "
                    "send exactly one"
                )
            data = dict(data)
            data["max_tokens"] = data.pop("max_completion_tokens")
        return data

    @model_validator(mode="before")
    @classmethod
    def _refuse_unhonoured_semantic_fields(cls, data):
        if isinstance(data, dict):
            for field, (would_change_output, reason) in _UNHONOURED_SEMANTIC_FIELDS.items():
                if field in data and would_change_output(data[field]):
                    raise ValueError(
                        f"'{field}' is not honoured by this API and is not a "
                        f"no-op in this request: {reason}. Refusing rather than "
                        "silently dropping it (HS-OD-1)."
                    )
        return data


class OpenAIChoice(BaseModel):
    """OpenAI choice object."""

    index: int = 0
    message: OpenAIMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class OpenAIUsage(BaseModel):
    """OpenAI usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # {"cached_tokens": N} — present only when the backend reported a KV-cache
    # reuse count (client tool mode); omitted from the JSON otherwise.
    prompt_tokens_details: dict[str, int] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_details(self, handler) -> dict[str, Any]:
        data = handler(self)
        if isinstance(data, dict) and data.get("prompt_tokens_details") is None:
            data.pop("prompt_tokens_details", None)
        return data


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orchestrator"
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    # Extension fields
    x_orchestrator_metadata: dict[str, Any] | None = None


class OpenAIModelInfo(BaseModel):
    """OpenAI model info."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "orchestrator"
    # The role's REAL per-request context (prompt + generation), read from the
    # serving llama-server's /props (registry fallback). `max_model_len` is the
    # vLLM spelling of the same number. Omitted when unknown.
    context_length: int | None = None
    max_model_len: int | None = None


class OpenAIModelsResponse(BaseModel):
    """OpenAI models list response."""

    object: str = "list"
    data: list[OpenAIModelInfo]
