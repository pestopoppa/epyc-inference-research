"""Request models for the orchestrator API."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Role strings that legitimately appear on the wire but are pipeline-stage
# markers rather than serving roles. They must not be rewritten to "" (that
# would change routing), but they are also not expected to resolve to a Role.
_NON_ROLE_SENTINELS = frozenset({"mock", "plan", "stream_init", "proactive_delegation"})


def _normalize_role_field(value: str | None, field_name: str) -> str | None:
    """Coerce a client-supplied role string to a canonical role name.

    ``role`` and ``force_role`` were bare ``str`` fields with no validation, so
    any client string flowed through the pipeline into the TASK_COMPLETED
    telemetry as ``producer_role``. A rescore of 117,074 historical completions
    (2026-07-21) found 157 rows with non-role values, including uppercase
    variants from 3-way seeding (``SELF``, ``WORKER``) and one row whose role
    field contained a prompt fragment ("Provide full Python snippet with
    variable name.").

    That is not cosmetic. ``compute_reward`` looks the role up in
    ``baseline_tps_by_role``; an unresolvable role misses every cost dimension
    and the task scores the full base reward. An unvalidated client string could
    therefore suppress the entire cost/speed penalty — a latent reward-hacking
    surface, and the same silent-miss shape as the ``role``/``producer_role``
    bug it was found alongside.

    Behaviour: resolve via ``Role.from_string`` (which already handles legacy
    aliases), accept a case-insensitive match, pass known non-role sentinels
    through untouched, and downgrade anything else to "" (auto-route) with a
    warning rather than letting it reach telemetry.
    """
    if value is None or value == "":
        return value

    from src.roles import Role

    if Role.from_string(value) is not None:
        return value
    if value in _NON_ROLE_SENTINELS:
        return value

    lowered = value.strip().lower()
    if Role.from_string(lowered) is not None:
        logger.warning(
            "Normalized non-canonical %s=%r to %r", field_name, value, lowered
        )
        return lowered

    logger.warning(
        "Rejected unrecognized %s=%r (%d chars); falling back to auto-route. "
        "Unresolvable roles silently disable the reward cost penalty.",
        field_name,
        value[:80],
        len(value),
    )
    return ""


class ScoutTargetSpec(BaseModel):
    """INF-78 OAB-8: one scout target — a profile hotspot and/or a candidate source file."""

    symbol: str | None = Field(default=None, max_length=512,
                               description="Profile symbol (perf/nsys name); normalised for a definition grep")
    file: str | None = Field(default=None, max_length=1024,
                             description="Candidate source file, relative to task_root or absolute inside the scope")
    share: float | None = Field(default=None, ge=0.0, le=1.0,
                                description="Profile share as a fraction (sampled_period_fraction)")
    label: str | None = Field(default=None, max_length=200)
    dso: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _needs_symbol_or_file(self):
        if not (self.symbol or self.file):
            raise ValueError("a scout target needs a symbol or a file")
        return self


class ScoutsSpec(BaseModel):
    """INF-78 OAB-8: orchestrator-run read-only scouts before the planner turn.

    Absent or ``enabled=false``: nothing runs and the request is unchanged."""

    enabled: bool = Field(default=False)
    targets: list[ScoutTargetSpec] = Field(default_factory=list, max_length=16)
    max: int = Field(default=4, ge=1, le=8,
                     description="Most scouts to run; the live /slots cap may run fewer")
    role: str | None = Field(default=None, max_length=64,
                             description="Role whose server the scouts use; default: force_role, "
                                         "else the routed role")
    max_turns: int = Field(default=8, ge=1, le=12)
    summary_tokens: int = Field(default=1500, ge=128, le=4096)
    budget_s: float = Field(default=240.0, ge=5.0, le=900.0,
                            description="Wall budget for the whole scout stage")
    reserve_slots: int = Field(default=1, ge=1, le=8,
                               description="Server slots always left free (>=1)")
    enable_thinking: bool = Field(default=False)

    @field_validator("role")
    @classmethod
    def _normalize_scout_role(cls, value: str | None) -> str | None:
        return _normalize_role_field(value, "scouts.role") or None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    prompt: str = Field(..., description="The user prompt to process")
    context: str = Field(default="", description="Optional context to include")
    mock_mode: bool = Field(
        default=True, description="Use mock responses instead of real inference"
    )
    real_mode: bool = Field(
        default=False, description="Enable real inference with RadixAttention caching"
    )
    # INF-78 OAB-1 / R3: 50 stays the ceiling for ordinary traffic; a task_root-scoped agentic
    # caller (the AutoKernel planner/author, ~60 steps) may declare up to MAX_TURNS_SCOPED.
    # Enforced in _validate_task_scope below.
    MAX_TURNS_UNSCOPED: ClassVar[int] = 50
    MAX_TURNS_SCOPED: ClassVar[int] = 100
    max_turns: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum orchestration turns (<=50; <=100 only when task_root is set)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=32768,
        description=(
            "Optional response-token cap for direct chat execution. When set, "
            "stage defaults such as MCQ/code budgets are clamped to this value."
        ),
    )
    n_probs: int | None = Field(
        default=None,
        ge=0,
        le=128,
        description=(
            "Optional llama.cpp top-k token probability capture for calibration "
            "instrumentation. Omitted by normal chat traffic."
        ),
    )
    role: str = Field(
        default="", description="Initial role to use (empty = auto-route via _classify_and_route)"
    )
    force_role: str | None = Field(
        default=None,
        description="Force routing to a specific role, bypassing all routing logic. "
        "Used by comparative seeding to test specialist quality.",
    )

    @field_validator("role", "force_role")
    @classmethod
    def _validate_role_fields(cls, value, info):
        """Reject free text in role fields — see _normalize_role_field."""
        return _normalize_role_field(value, info.field_name)
    force_mode: str | None = Field(
        default=None,
        description="Force execution mode ('direct', 'react', 'repl', 'delegated', or 'edit'), "
        "bypassing _select_mode heuristics. 'delegated' enables architect delegation "
        "where the architect formulates investigation briefs for faster specialists. "
        "'edit' runs a one-shot transactional file edit (flag-gated ORCHESTRATOR_EDIT_TRANSACTION=1 "
        "+ scoped ORCHESTRATOR_EDIT_ROOT; bypasses the multi-turn REPL loop for routine file edits).",
    )
    allow_delegation: bool | None = Field(
        default=None,
        description="Override delegation capability. True=allow delegation to workers, "
        "False=disable delegation (model handles alone). None=use feature flag default. "
        "Used by 3-way seeding to isolate SELF vs ARCHITECT behavior.",
    )
    server_urls: dict[str, str] | None = Field(
        default=None,
        description="Server URLs for real mode (e.g., {'frontdoor': 'http://localhost:8080'})",
    )
    # Vision support — when set, routes to VL workers (8086/8087)
    image_path: str | None = Field(
        default=None, description="Path to image file for vision tasks (routes to VL worker)"
    )
    image_base64: str | None = Field(
        default=None, description="Base64-encoded image data for vision tasks"
    )
    files: list[str] | None = Field(
        default=None,
        description="List of file paths for multi-file vision/document tasks (archives auto-extracted)",
    )
    # Per-request cache control
    cache_prompt: bool | None = Field(
        default=None,
        description="Override cache_prompt for this request (None=backend default True). "
        "Set to False for benchmark seeding where prefix caching adds overhead.",
    )
    # Extended thinking support (Claude Code parity)
    thinking_budget: int = Field(
        default=0,
        ge=0,
        le=32000,
        description="Token budget for internal reasoning (0=disabled, max=32000)",
    )
    permission_mode: str = Field(
        default="normal", description="Permission mode: 'normal', 'auto-accept', or 'plan'"
    )
    timeout_s: int | None = Field(
        default=None,
        ge=1,
        # 600->1800->3600: giant (~90K-token) eval prompts can spend nearly
        # 30 minutes in half-instance prefill. Policy (which workload may
        # extend how far) is governed by resolve_timeout's eval_batch scoping,
        # not this shape-validation ceiling; non-eval traffic remains clamped
        # to its role SLA.
        le=3600,
        description="Optional per-request server-side timeout budget in seconds. "
        "When set, orchestration deadlines and lock waits are bounded to this value.",
    )
    client_deadline_unix_s: float | None = Field(
        default=None,
        description="Optional client wall-clock deadline in Unix seconds. "
        "When set, server execution budget is additionally clamped to this deadline.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for cross-request REPL globals restore.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional caller request id for tracing. The live inference tap "
        "records this as parent_request_id and still creates one unique id per model call.",
    )
    trial_id: int | str | None = Field(
        default=None,
        description="Optional autopilot/benchmark trial id for inference tap attribution.",
    )
    batch_id: int | str | None = Field(
        default=None,
        description="Optional concurrency/eval batch id for inference tap attribution.",
    )
    request_priority: str = Field(
        default="interactive",
        description="Admission priority: 'interactive' (default) or 'background'. "
        "Interactive requests are prioritized at backend admission gates.",
    )
    workload_class: Literal["interactive", "eval_batch", "campaign"] | None = Field(
        default=None,
        description="Optional workload traffic class for attribution: "
        "'interactive', 'eval_batch', or 'campaign'. When unset, the server "
        "infers it from existing request metadata without changing admission priority.",
    )
    batch_placement_mode: Literal[
        "auto", "homogeneous_native_batch", "mixed_role_split"
    ] | None = Field(
        default=None,
        description="Optional burst placement intent. Homogeneous cohorts may share "
        "one full server's certified native slots; mixed routed pipelines start on "
        "sub-full instances so different CPU roles can occupy disjoint regions.",
    )
    x_orchestrator_prompt_root: str | None = Field(
        default=None,
        description=(
            "Internal AutoPilot/GEPA override for resolving prompt files from "
            "a scratch prompt tree. Accepted only from configured scratch roots."
        ),
    )
    routing_preferences: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional DAR-4b routing scalarization weights. Keys 'perf'/'performance' "
            "and 'cost' are normalized onto the performance-cost simplex. "
            "Absent means the existing retrieval score is preserved."
        ),
    )
    max_queue_wait_ms: int | None = Field(
        default=None,
        description="Maximum time the cross-role contention gate may queue "
        "this request before rejecting it (HTTP 503 + Retry-After). When unset, "
        "the gate uses 5s for interactive and 90s for background. Foreground "
        "callers with tight SLO budgets should set this explicitly.",
    )
    migration_budget_ms: int | None = Field(
        default=None,
        description="Phase E (cross-role-bw-aware-routing): maximum acceptable "
        "KV-save+restore latency for migrating an existing session from full "
        "to quarter on concurrent arrival. Short interactive turns should set "
        "this low (e.g. 200) to skip migration and queue/cold-start instead. "
        "Long conversations + background probes can amortize a longer budget. "
        "Honored by the per-region-locks session-handover migration transaction "
        "when one is attempted; see "
        "ConcurrencyAwareBackend.kv_migration_status() for runtime state.",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Additional stop sequences to halt generation. "
        "Merged with any pipeline-default stop sequences (e.g. QWEN_STOP). "
        "Used by benchmark seeding to stop after answer tags.",
    )
    # EVL-42 1c-fix (d): DEPRECATED, accepted-but-ignored. Nothing on the /chat
    # path consumes these (audit 2026-07-24: tool use is the bespoke REPL
    # TOOL()/CALL()/FINAL() protocol, not native function calling). They are kept
    # on the wire so published clients keep validating (extra='ignore' would drop
    # them silently anyway); native tool schemas are honoured ONLY by
    # /v1/chat/completions (OpenAIChatRequest.tools). Routes call
    # ignored_tool_fields() to log when a caller relies on them.
    DEPRECATED_TOOL_FIELDS: ClassVar[tuple[str, ...]] = ("tools", "tool_choice")

    tools: list[dict] | None = Field(
        default=None,
        deprecated="DEPRECATED (EVL-42 1c-fix d): accepted but NOT consumed by /chat. "
        "Native tool schemas are only honoured by /v1/chat/completions.",
        description="DEPRECATED and ignored on /chat: OpenAI-compatible tool schemas are "
        "NOT exposed to the REPL here. Use /v1/chat/completions (OpenAIChatRequest.tools) "
        "for native function tools.",
    )
    tool_choice: str | dict | None = Field(
        default=None,
        deprecated="DEPRECATED (EVL-42 1c-fix d): accepted but NOT consumed by /chat. "
        "Native tool_choice is only honoured by /v1/chat/completions.",
        description="DEPRECATED and ignored on /chat: tool choice policy is NOT applied "
        "here. Use /v1/chat/completions for native tool_choice.",
    )

    def ignored_tool_fields(self) -> tuple[str, ...]:
        """Names of deprecated tool fields the caller explicitly set (EVL-42 1c-fix d).

        Reads ``model_fields_set`` so it never triggers the field-level
        DeprecationWarning; routes log a warning when this is non-empty so a
        client relying on /chat tool schemas is visible instead of silently ignored.
        """
        return tuple(f for f in self.DEPRECATED_TOOL_FIELDS if f in self.model_fields_set)
    eval_fence: bool | None = Field(
        default=None,
        description=(
            "AP-54 eval knowledge fence. Sent by the AutoPilot EvalTower on every "
            "eval rollout. True: file/REPL/shell tools refuse the compiled wiki, "
            "knowledge roots and eval gold files, and the response echoes the "
            "paths the tools touched. False: nothing is refused but touched paths "
            "are still echoed (AP-54b control arm). Absent: production behaviour, "
            "unchanged. Older API builds ignore the field (extra='ignore')."
        ),
    )
    # ── INF-78 OAB-1: per-request task scope ─────────────────────────────────
    task_root: str | None = Field(
        default=None,
        description=(
            "INF-78 OAB-1. Absolute path of an existing directory strictly below the llm "
            "root or /tmp (never containing the orchestrator project root) that scopes THIS "
            "request: model file tools resolve relative paths under it, reads are confined to "
            "it plus read_roots, run_shell runs in it, code_search greps it. Per request "
            "(ContextVar), never process-wide; overrides ORCHESTRATOR_EDIT_ROOT. Absent: "
            "production behaviour, unchanged."
        ),
    )
    edit_mode: Literal["none", "direct"] = Field(
        default="none",
        description=(
            "INF-78 OAB-1. 'none' (default): the request cannot write any file. 'direct': "
            "file_write_safe writes inside task_root immediately (no approval queue, no .bak "
            "files); every other write surface stays refused. 'direct' requires task_root."
        ),
    )
    read_roots: list[str] | None = Field(
        default=None,
        description=(
            "INF-78 OAB-1. Extra existing directories (absolute, strictly below the llm root "
            "or /tmp) the request may READ in addition to task_root. Never writable. "
            "Requires task_root."
        ),
    )
    quiescent_after: bool = Field(
        default=False,
        description=(
            "INF-78 OAB-3 (R2). True: the orchestrator starts NO fire-and-forget work for "
            "this request (MemRL q-scoring, typed-decision shadow, KV "
            "migration), and the idle-time scoring loop stays quiet for a window after the "
            "reply, so nothing orchestrator-owned accrues CPU after /chat returns. The "
            "response echoes what was suppressed in `quiescence`."
        ),
    )
    # ── INF-78 OAB-7: the context bundle as a REPL variable ─────────────────────
    context_bundle: dict | None = Field(
        default=None,
        description=(
            "INF-78 OAB-7. Structured context the REPL exposes as the variable `context` "
            "instead of inlining it in the prompt (the RLM pattern): "
            "{schema?: 'epyc.orchestrator.context_bundle.v1', sections: [{name, text, "
            "kind?: 'text'|'json', inline?: bool, description?: str}, ...], manifest?: {...}}. "
            "The root prompt gets an index of the sections (names, sizes); the model pulls "
            "with context.index()/get()/grep()/json()/[name] (and search() with "
            "context_search), pulls land in REPL variables, "
            "and only what it prints reaches the root prompt, capped per turn at "
            "context_print_cap_bytes. Requires force_mode='repl'. The response echoes the "
            "pull accounting in `context_pulls`. Absent: production behaviour, unchanged."
        ),
    )
    context_print_cap_bytes: int = Field(
        default=4096,
        ge=256,
        le=65536,
        description="INF-78 OAB-7. Per-turn cap (UTF-8 bytes) on printed REPL output while a "
        "context_bundle is attached. Requires context_bundle when set.",
    )
    context_pull_budget_bytes: int | None = Field(
        default=None,
        ge=1,
        description="INF-78 OAB-7 / OAB-12. Optional cap on the bytes the model may pull from "
        "the context_bundle over the whole call (a pull past it raises in the REPL). "
        "Requires context_bundle.",
    )
    context_search: bool = Field(
        default=False,
        description="UFH-12. Expose context.search(query, k=8, section=None) on the bundle: "
        "hybrid lexical (BM25) + dense (the orchestrator's embedding pool, when the "
        "repl_embedding_pool flag is on) retrieval fused with RRF, returning POINTERS "
        "{section, start, end, line, score, via, mode} and never text; the model reads a hit "
        "through context.get, so pull accounting is unchanged. The root prompt lists it and "
        "context_pulls gains a `search` block. Requires context_bundle. Default off: the "
        "bundle's prompt and API are byte-identical to before.",
    )
    scouts: ScoutsSpec | None = Field(
        default=None,
        description=(
            "INF-78 OAB-8. When enabled, the orchestrator runs one read-only scout per target "
            "(profile hotspot / candidate file) concurrently BEFORE the planner turn, capped by "
            "the target server's free /slots minus reserve_slots, and prepends their labelled, "
            "sized summaries to the prompt. Requires task_root (scouts read only inside the "
            "task scope). The response echoes provenance in `scouts`. Absent: unchanged."
        ),
    )
    output_schema: dict | None = Field(
        default=None,
        description="Optional JSON Schema for the agent's FINAL() value. "
        "When set AND features().final_schema_validation is True, the agent receives "
        "the schema in its initial prompt and must call FINAL(json.dumps(value)). "
        "Validation failure injects a retry-with-error message into the next turn.",
    )

    @model_validator(mode="after")
    def _validate_context_bundle(self):
        """INF-78 OAB-7: a bundle is validated at the door (422), and only the REPL reads it."""
        if self.context_bundle is None:
            if "context_print_cap_bytes" in self.model_fields_set:
                raise ValueError("context_print_cap_bytes requires context_bundle")
            if self.context_pull_budget_bytes is not None:
                raise ValueError("context_pull_budget_bytes requires context_bundle")
            if self.context_search:
                raise ValueError("context_search requires context_bundle")
            return self
        if self.force_mode != "repl":
            # Every other path (direct, react, delegated, edit, the proactive and
            # cheap-first stages) would drop the bundle without a word.
            raise ValueError("context_bundle requires force_mode='repl'")
        from src.repl_environment.context_bundle import ContextBundle

        ContextBundle.from_payload(
            self.context_bundle,
            print_cap_bytes=self.context_print_cap_bytes,
            pull_budget_bytes=self.context_pull_budget_bytes,
        )
        return self

    @model_validator(mode="after")
    def _validate_task_scope(self):
        """INF-78 OAB-1: task_root / edit_mode / read_roots / max_turns coherence."""
        if self.task_root is None:
            if self.scouts is not None and self.scouts.enabled:
                raise ValueError("scouts require task_root (scouts read only inside the task scope)")
            if self.edit_mode != "none":
                raise ValueError("edit_mode='direct' requires task_root")
            if self.read_roots:
                raise ValueError("read_roots requires task_root")
            if self.max_turns > self.MAX_TURNS_UNSCOPED:
                raise ValueError(
                    f"max_turns > {self.MAX_TURNS_UNSCOPED} requires task_root "
                    f"(scoped ceiling {self.MAX_TURNS_SCOPED})"
                )
            return self
        from src.repl_environment.task_root import validate_scope_dir

        self.task_root = validate_scope_dir(self.task_root)
        if self.read_roots:
            self.read_roots = [
                validate_scope_dir(r, field="read_roots", writable_root=False)
                for r in self.read_roots
            ]
        return self


class RewardRequest(BaseModel):
    """Request model for injecting external rewards into MemRL."""

    task_description: str = Field(..., description="Description of the task that was scored")
    action: str = Field(..., description="Action taken, e.g. 'frontdoor:direct'")
    reward: float = Field(..., ge=-1.0, le=1.0, description="Reward value (-1.0 to 1.0)")
    context: dict | None = Field(
        default=None, description="Optional metadata (suite, tier, scoring_method)"
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Precomputed embedding for task_description (avoids re-embedding)",
    )


class GateRequest(BaseModel):
    """Request model for running gates."""

    gate_names: list[str] | None = Field(
        default=None, description="Specific gates to run (None = all)"
    )
    stop_on_first_failure: bool = Field(
        default=True, description="Stop after first required gate fails"
    )
    required_only: bool = Field(default=False, description="Only run required gates")
