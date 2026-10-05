#!/usr/bin/env python3
"""Feature flag system for optional orchestration modules.

This module defines all optional features that can be enabled/disabled
independently of core orchestration functionality. Each feature has:
- A clear description of what it does
- Dependencies (what other features/modules it requires)
- Environment variable to enable/disable

Usage:
    from src.features import Features, get_features

    # Get feature flags (reads from environment or config)
    features = get_features()

    # Check if a feature is enabled
    if features.memrl:
        from orchestration.repl_memory import TaskEmbedder
        # ... use MemRL components

    # Create features with explicit flags
    features = Features(memrl=False, tools=True)

Environment Variables:
    ORCHESTRATOR_MEMRL=1         Enable MemRL (learned routing, Q-scoring)
    ORCHESTRATOR_TOOLS=1         Enable tool registry (REPL tools)
    ORCHESTRATOR_SCRIPTS=1       Enable script registry (prepared scripts)
    ORCHESTRATOR_STREAMING=1     Enable SSE streaming endpoints
    ORCHESTRATOR_OPENAI_COMPAT=1 Enable OpenAI-compatible API
    ORCHESTRATOR_REPL=1          Enable REPL execution environment
    ORCHESTRATOR_DEFERRED_TOOL_RESULTS=1  Disable mixin tool-output wrapping

Design Principles:
    1. Core orchestration works with ALL features disabled
    2. Features are opt-in by default in tests, opt-out in production
    3. Each feature can be toggled independently
    4. Dependencies are documented and checked at initialization

Adding New Features:
    1. Add field to Features dataclass with description
    2. Add environment variable check in get_features()
    3. Add dependency documentation if needed
    4. Guard feature code with if features.your_feature:
    5. Add tests for both enabled/disabled states
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.env_parsing import env_bool

# Environment variable prefixes for feature flags.
#
# ORCHESTRATOR_<FLAG> is the legacy spelling and remains supported for
# compatibility. ORCHESTRATOR_FEATURE_<FLAG> is preferred for stack-managed
# API workers because top-level ORCHESTRATOR_* names can collide with
# structured Pydantic settings such as OrchestratorSettings.repl.
ENV_PREFIX = "ORCHESTRATOR_"
FEATURE_ENV_PREFIX = "ORCHESTRATOR_FEATURE_"
RUNTIME_FLAGS_ENV = "ORCHESTRATOR_RUNTIME_FLAGS_PATH"
RUNTIME_FLAGS_TTL_S = 1.0
# A recovery lifetime for experiment enables, not a measurement threshold.
EXPERIMENT_FLAG_TTL_S = 6 * 60 * 60
logger = logging.getLogger(__name__)


# ── Declarative Feature Registry ──────────────────────────────────────────
# Single source of truth for feature metadata. Drives summary(), get_features()
# defaults, and env parsing. The Features dataclass fields must stay in sync
# (validated by test_features_registry_consistency).


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative specification for a single feature flag."""
    name: str
    default_test: bool
    default_prod: bool
    env_var: str  # env var suffix (without ORCHESTRATOR_ prefix)
    description: str = ""
    dependencies: tuple[str, ...] = ()


_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # Phase 4: MemRL
    FeatureSpec("memrl", False, True, "MEMRL", "Memory-based Reinforcement Learning"),
    # Tool and Script Registries
    FeatureSpec("tools", False, True, "TOOLS", "Tool Registry for REPL"),
    FeatureSpec("scripts", False, True, "SCRIPTS", "Script Registry", ("tools",)),
    # API Features
    FeatureSpec("streaming", False, True, "STREAMING", "SSE streaming endpoints"),
    FeatureSpec("openai_compat", False, True, "OPENAI_COMPAT", "OpenAI-compatible API"),
    # Core Features
    FeatureSpec("repl", True, True, "REPL", "REPL execution environment"),
    FeatureSpec("caching", False, True, "CACHING", "Response caching with prefix routing"),
    # Phase 2: Structured Delimiters
    FeatureSpec("structured_delimiters", True, True, "STRUCTURED_DELIMITERS", "Wrap tool outputs with delimiters"),
    FeatureSpec("react_mode", False, True, "REACT_MODE", "ReAct-style tool loop"),
    FeatureSpec("output_formalizer", False, True, "OUTPUT_FORMALIZER", "Format constraint enforcement"),
    FeatureSpec("parallel_tools", True, True, "PARALLEL_TOOLS", "Parallel read-only tool dispatch"),
    FeatureSpec("deferred_tool_results", False, False, "DEFERRED_TOOL_RESULTS", "Deferred tool result wrapping"),
    FeatureSpec("escalation_compression", False, True, "ESCALATION_COMPRESSION", "LLMLingua-2 BERT for large prompts"),
    FeatureSpec("script_interception", False, False, "SCRIPT_INTERCEPTION", "Resolve trivial queries locally"),
    # Security Features
    FeatureSpec("credential_redaction", True, True, "CREDENTIAL_REDACTION", "Scan for leaked credentials"),
    FeatureSpec("cascading_tool_policy", False, True, "CASCADING_TOOL_POLICY", "Layered tool permission chain"),
    FeatureSpec("restricted_python", False, False, "RESTRICTED_PYTHON", "RestrictedPython for REPL"),
    # Phase 3: Specialist routing
    FeatureSpec("specialist_routing", False, True, "SPECIALIST_ROUTING", "Q-value specialist routing", ("memrl",)),
    FeatureSpec("graph_router", False, False, "GRAPH_ROUTER", "GNN-based parallel routing", ("specialist_routing",)),
    FeatureSpec("ingest_triviality_guard", False, False, "INGEST_TRIVIALITY_GUARD", "Demote trivially-easy short prompts off ingest_long_context", ("specialist_routing",)),
    FeatureSpec("plan_review", False, True, "PLAN_REVIEW", "Architect plan review", ("memrl",)),
    FeatureSpec("architect_delegation", False, True, "ARCHITECT_DELEGATION", "Architect delegation", ("memrl",)),
    FeatureSpec("parallel_execution", False, True, "PARALLEL_EXECUTION", "Wave-based step execution", ("architect_delegation",)),
    FeatureSpec("personas", False, False, "PERSONAS", "Persona registry", ("memrl",)),
    FeatureSpec("staged_rewards", False, False, "STAGED_REWARDS", "PARL-inspired annealing", ("memrl",)),
    # MemRL Distillation
    FeatureSpec("routing_classifier", False, False, "ROUTING_CLASSIFIER", "ColBERT-Zero routing classifier"),
    FeatureSpec("skillbank", False, False, "SKILLBANK", "SkillRL experience distillation", ("memrl",)),
    # Phase 4: Input formalizer
    FeatureSpec("input_formalizer", False, True, "INPUT_FORMALIZER", "MathSmith-8B formal spec extraction"),
    # Unified streaming
    FeatureSpec("unified_streaming", False, True, "UNIFIED_STREAMING", "Route streaming through pipeline stages"),
    # Semantic classifiers
    FeatureSpec("semantic_classifiers", True, True, "SEMANTIC_CLASSIFIERS", "Config-driven classifiers"),
    # Generation Monitoring
    FeatureSpec("generation_monitor", False, True, "GENERATION_MONITOR", "Early failure detection"),
    # OpenClaw/Lobster concepts
    FeatureSpec("side_effect_tracking", False, True, "SIDE_EFFECT_TRACKING", "Tool side effect declarations"),
    FeatureSpec("structured_tool_output", False, True, "STRUCTURED_TOOL_OUTPUT", "ToolOutput envelope"),
    FeatureSpec("model_fallback", False, True, "MODEL_FALLBACK", "Same-tier alternatives on circuit-open"),
    FeatureSpec("content_cache", False, False, "CONTENT_CACHE", "SHA-256 keyed response cache"),
    FeatureSpec("session_compaction", False, True, "SESSION_COMPACTION", "Summarize old context"),
    FeatureSpec("session_log", False, True, "SESSION_LOG", "Append-only processing journal"),
    FeatureSpec("session_scratchpad", False, True, "SESSION_SCRATCHPAD", "Model-extracted semantic insights"),
    FeatureSpec("depth_model_overrides", False, True, "DEPTH_MODEL_OVERRIDES", "Map nested depth to cheaper roles"),
    FeatureSpec("resume_tokens", False, True, "RESUME_TOKENS", "Base64url continuation tokens"),
    FeatureSpec("approval_gates", False, True, "APPROVAL_GATES", "Human approval at escalation boundaries", ("resume_tokens", "side_effect_tracking")),
    FeatureSpec("binding_routing", False, False, "BINDING_ROUTING", "Priority-ordered routing overrides"),
    # Budget controls
    FeatureSpec("worker_call_budget", False, True, "WORKER_CALL_BUDGET", "Cap total REPL executions per task"),
    FeatureSpec("task_token_budget", False, True, "TASK_TOKEN_BUDGET", "Cap cumulative tokens per task"),
    FeatureSpec("final_schema_validation", False, False, "FINAL_SCHEMA_VALIDATION", "Validate FINAL() value against caller-supplied JSON Schema; retry-with-error on failure"),
    # Context-Folding
    FeatureSpec("two_level_condensation", False, False, "TWO_LEVEL_CONDENSATION", "CF Phase 1: granular + deep consolidation"),
    FeatureSpec("segment_cache_dedup", False, False, "SEGMENT_CACHE_DEDUP", "CF Phase 1+: hash-based dedup"),
    FeatureSpec("helpfulness_scoring", False, False, "HELPFULNESS_SCORING", "CF Phase 2c: heuristic helpfulness scoring"),
    FeatureSpec("process_reward_telemetry", False, False, "PROCESS_REWARD_TELEMETRY", "CF Phase 3a: process reward telemetry"),
    FeatureSpec("role_aware_compaction", False, False, "ROLE_AWARE_COMPACTION", "CF Phase 3b: role-aware profiles"),
    # Context window management
    FeatureSpec("accurate_token_counting", False, False, "ACCURATE_TOKEN_COUNTING", "Use llama-server /tokenize"),
    FeatureSpec("tool_result_clearing", False, True, "TOOL_RESULT_CLEARING", "Clear stale tool output blocks"),
    # Reasoning length alarm
    FeatureSpec("reasoning_length_alarm", False, True, "REASONING_LENGTH_ALARM", "Retry verbose reasoning"),
    # Tool output compression
    FeatureSpec("tool_output_compression", False, False, "TOOL_OUTPUT_COMPRESSION", "Compress verbose output"),
    # Output spill
    FeatureSpec("output_spill_to_file", False, True, "OUTPUT_SPILL_TO_FILE", "Spill truncated output to file"),
    # Pipeline monitoring
    FeatureSpec("model_grading", False, False, "MODEL_GRADING", "Post-hoc model-graded evals"),
    # P-BENCH-3 A7: eval-batch serving-class switch. Metadata/shadow support may
    # land separately, but execution routing remains off until this flag is set.
    FeatureSpec("eval_batch_serving", False, False, "EVAL_BATCH_SERVING", "Route eval batches through an evidence-gated batch-serving class"),
    # HSD
    FeatureSpec("self_speculation", False, False, "SELF_SPECULATION", "Self-speculation with layer-exit draft"),
    FeatureSpec("hierarchical_speculation", False, False, "HIERARCHICAL_SPECULATION", "Hierarchical intermediate verification"),
    # LangGraph pre-migration
    FeatureSpec("state_history_snapshots", False, False, "STATE_HISTORY_SNAPSHOTS", "Full TaskState snapshots each turn"),
    FeatureSpec("generalized_interrupts", False, False, "GENERALIZED_INTERRUPTS", "Pluggable interrupt conditions", ("approval_gates", "resume_tokens")),
    # LangGraph migration
    FeatureSpec("langgraph_bridge", False, False, "LANGGRAPH_BRIDGE", "LangGraph Phase 1: hybrid bridge"),
    FeatureSpec("langgraph_ingest", False, False, "LANGGRAPH_INGEST", "LangGraph Phase 3: IngestNode"),
    FeatureSpec("langgraph_architect", False, False, "LANGGRAPH_ARCHITECT", "LangGraph Phase 3: ArchitectNode"),
    FeatureSpec("langgraph_worker", False, False, "LANGGRAPH_WORKER", "LangGraph Phase 3: WorkerNode"),
    FeatureSpec("langgraph_frontdoor", False, False, "LANGGRAPH_FRONTDOOR", "LangGraph Phase 3: FrontdoorNode"),
    FeatureSpec("langgraph_coder", False, False, "LANGGRAPH_CODER", "LangGraph Phase 3: CoderNode"),
    FeatureSpec("langgraph_coder_escalation", False, False, "LANGGRAPH_CODER_ESCALATION", "LangGraph Phase 3: CoderEscalationNode"),
    # Conversation Management
    FeatureSpec("injection_scanning", False, True, "INJECTION_SCANNING", "B7: prompt injection scanning"),
    FeatureSpec("context_compression", False, False, "CONTEXT_COMPRESSION", "B2: protected-zone compression"),
    FeatureSpec("user_modeling", False, False, "USER_MODELING", "B1: cross-session user preferences", ("injection_scanning",)),
    FeatureSpec("session_token_budget", False, False, "SESSION_TOKEN_BUDGET", "B5: per-session token budget"),
    # Claude Code Local
    FeatureSpec("claude_code_mcp_chat", False, False, "CLAUDE_CODE_MCP_CHAT", "CC Local: MCP chat delegation"),
    # HS-4 P0.2: refuse (422) an agentic-shell /v1 request that lacks x_session_id
    # (OpenCode user-agent or x_tool_mode=client) — a failed session plugin is only
    # logged by OpenCode, so without this it degrades silently. Other clients unaffected.
    FeatureSpec("v1_client_session_guard", True, True, "V1_CLIENT_SESSION_GUARD", "HS-4: require x_session_id from OpenCode/client-tool-mode /v1 requests"),
    # HS-19a stage 1 ("Linked"): record the harness subagent tree on /v1 — the
    # x-parent-session-id header OpenCode already sends (plus HS-16's session-header
    # fallback and the typed x_parent_session_id / x_agent_name body keys) into the
    # inference-tap request_keys and the progress (session) log. RECORD ONLY: model
    # selection is unchanged. Default OFF in test and prod; off = byte-identical.
    FeatureSpec("v1_subagent_link", False, False, "V1_SUBAGENT_LINK", "HS-19a stage 1: record harness subagent parent/child links on /v1 (tap + session log); never changes model selection"),
    # TE-1 (UFH-13) / HS-4 P4 subset: let a /v1 frontdoor answer take /chat's own
    # post-answer escalation hooks (quality escalation to coder_escalation and the
    # MemRL review gate to architect_general), OPT-IN per request via
    # x_escalation=auto|architect_general (absent or off = no escalation), with an
    # escalation receipt in the tap. Default OFF in test and prod. Flag off, or flag
    # on with the key absent, = byte-identical (golden-pinned in
    # test_v1_escalation_off_golden.py), so enabling it changes no unkeyed traffic.
    FeatureSpec("v1_escalation", False, False, "V1_ESCALATION", "TE-1: /v1 frontdoor answers may escalate through /chat's quality-escalation and review-gate hooks, opt-in per request (x_escalation=auto|architect_general; absent/off = none); default off"),
    # RI-23 / OP-69 (a), 2026-09-29: thinking-on roles (live stack priors: --jinja AND
    # acceleration.enable_thinking true — today the :8083 27B's architect_critic /
    # coder_escalation / ingest_long_context) move from /completion to the
    # /v1/chat/completions lane, so their GGUF chat template and registry
    # chat_template_kwargs (enable_thinking, reasoning_effort) take effect and the server
    # splits reasoning into reasoning_content. Read LIVE per request by both the backend
    # router and the orchestrator-side template skip, so the two cannot disagree across a
    # runtime flip. Default OFF in test and prod; flag off = byte-identical (RI-23b A/B).
    FeatureSpec("thinking_roles_chat_lane", False, False, "THINKING_ROLES_CHAT_LANE", "RI-23: thinking-on roles (jinja + enable_thinking) route through /v1/chat/completions so chat_template_kwargs apply and reasoning_content is split and surfaced; short structured calls (review verdict, plan review/plan JSON) run thinking-off per call; default off pending the RI-23b A/B"),
    # Web research reranking
    FeatureSpec("web_research_rerank", False, False, "WEB_RESEARCH_RERANK", "ColBERT snippet reranking in web_research pipeline"),
    # Routing telemetry
    FeatureSpec("logit_probe", False, False, "LOGIT_PROBE", "First-token logprob capture for learned routing P1.5"),
    # NIB2-45 MindDR deep research mode (three-agent pipeline, feature-flag-gated at pipeline entry)
    FeatureSpec("deep_research_mode", False, False, "DEEP_RESEARCH_MODE", "NIB2-45 MindDR Phase 1: three-agent research pipeline"),
    # intake-607 harness cluster (default-off; shadow/observe only)
    FeatureSpec("ure_uncertainty_shadow_log", False, False, "URE_UNCERTAINTY_SHADOW_LOG", "URE-1: shadow-log routing decision-uncertainty (J10); changes no routing behavior"),
    FeatureSpec("batch_edit_mode", False, False, "BATCH_EDIT_MODE", "BEP-1: coder emits one structured patch set after reasoning instead of interleaved REPL (J8)"),
    FeatureSpec("interleaved_edit_rider", False, False, "INTERLEAVED_EDIT_RIDER", "BEP-2 baseline: per-turn rider instructing coder/architect to apply interleaved file edits (vs answering in prose); experiment-only, default-off"),
    FeatureSpec("dcp_pre_assembly", False, False, "DCP_PRE_ASSEMBLY", "DCP-4 (J7): advisory delegation context pre-assembly — seed the specialist with a budget-bounded code bundle; reactive discovery stays on; default-off"),
    FeatureSpec("dcp_for_consult", False, False, "DCP_FOR_CONSULT", "Internal consults may reuse DCP pre-assembled context; requires dcp_pre_assembly", ("dcp_pre_assembly",)),
    FeatureSpec("review_before_commit_consult", False, False, "REVIEW_BEFORE_COMMIT_CONSULT", "P2 internal interaction: architect_general review_before_commit consult at the edit-transaction seam"),
    FeatureSpec("review_before_commit_targeted_gate", False, False, "REVIEW_BEFORE_COMMIT_TARGETED_GATE", "J17: restrict review_before_commit consults to high-risk edit shapes"),
    # RTE-Prefix (repl-turn-efficiency): render the fixed sections (task,
    # instruction) BEFORE the per-turn-mutating sections (state, context,
    # reference_code) in RootLMPrompt.to_string(), so successive REPL turns
    # share a longer prompt prefix and llama-server KV cache reuse pays off.
    # Default-off: behavioral change to the prompt, needs A/B validation first.
    FeatureSpec("prefix_stable_order", False, False, "PREFIX_STABLE_ORDER", "RTE-Prefix: put fixed prompt sections before per-turn-mutating ones so REPL turns share a longer cache prefix"),
    # UFH14-B1 F1 (D1): add the derived silent-prefill allowance (serving_params:
    # measured prefill rate of THIS server x THIS prompt) to the HTTP timeout of
    # long-prompt calls — primitives calls with no request deadline, and the
    # passthrough read timeout (raised, never lowered). A deadlined call is
    # clamped back to its deadline regardless. The serving_params record block
    # (allowance, at_risk, doomed) is written whether this is on or off.
    # Default-off in BOTH: it lengthens timeouts (weaker hang detection) and the
    # rate has no live samples yet — flip after a doomed-count window.
    FeatureSpec("derived_prefill_timeout", False, False, "DERIVED_PREFILL_TIMEOUT", "UFH14-B1 F1: add the measured silent-prefill allowance to long-prompt call timeouts (no-deadline primitives calls; passthrough read timeout raised, never lowered)"),
    # UFH14-B1 F2 (D4): wall-time forced-answer turn in the REPL graph. Once the
    # elapsed share of the request budget reaches ORCHESTRATOR_REPL_ANSWER_FORCE_FRAC
    # (default 0.65, F12), the turn prompt demands FINAL(best answer) now and
    # that turn skips session compaction. Default-off in BOTH: prompt change.
    # UFH14-B1: session compaction rewrites TaskState.context, which no turn
    # prompt renders; its worker_general index call therefore reached no model.
    # Off = deterministic index, no LLM call (the bug fix). On = the old LLM
    # index with its input capped to the worker's window, for a future renderer.
    FeatureSpec("session_compaction_llm_index", False, False, "SESSION_COMPACTION_LLM_INDEX", "UFH14-B1: LLM-written compaction index (worker_general, input capped to its window); off = deterministic index, since TaskState.context is not rendered into turn prompts"),
    FeatureSpec("repl_answer_force", False, False, "REPL_ANSWER_FORCE", "UFH14-B1 F2: wall-time forced-answer turn (FINAL now at a fraction of the request budget; no compaction on that turn)"),
    # intake-614/615 DAR-6 scaffolding (default-off in BOTH test and prod; no production routing until DAR-6.5 A/B clears)
    FeatureSpec("swarm_fanout", False, False, "SWARM_FANOUT", "DAR-6.1: fan high-injection-risk prompts to N>=2 concurrent serves + BT-aggregate (J14). Scaffolding only — default-off until the DAR-6.5 injection-suite A/B clears (handoffs/active/decision-aware-routing.md § DAR-6.5)."),
    # P21.A test-time compute: DeepConf offline confidence-filtered self-consistency (intake-603)
    FeatureSpec("deepconf", False, False, "DEEPCONF", "DeepConf offline confidence-filtered self-consistency"),
    # RD-5 Reviewer decision plane (shadow/enforce split; intake-849 P2). Both default-off
    # in BOTH test and prod. review_decision_shadow has NO dependencies on purpose — it is
    # the DECOUPLED shadow-emission path: plan_review requires memrl (see validate()), but
    # the decision-plane shadow machinery must be able to flow WITHOUT flipping plan_review
    # (and therefore without requiring memrl). Always-on trace emission (TM-3) is independent
    # of both flags; these gate only the extra shadow processing / (blocked) enforcement.
    FeatureSpec("review_decision_shadow", False, False, "REVIEW_DECISION_SHADOW", "RD-5: emit + trace review-plane decisions (verifier precedence, reject-admissibility, warn-only downgrade) WITHOUT acting. Decoupled from plan_review→memrl (no dependencies)."),
    FeatureSpec("review_decision_enforce", False, False, "REVIEW_DECISION_ENFORCE", "RD-5: ACT on review-plane decisions. Default OFF and BLOCKED on the H-LB LB-6 latency/sampling budget gate — do not enable until that gate passes."),
    # gpu-serving-tie-in-program P0-7: role-agnostic GPU-resident shadow lane
    # (MI210). Default-off in BOTH test and prod. Gates only the lane's
    # scaffolding surfaces (np_ceiling policy loader + preflight tooling); the
    # lane serves NO production traffic until the P3-3 three-gates sign-off
    # (shadow-only invariant, program decision D3).
    FeatureSpec("gpu_shadow_lane", False, False, "GPU_SHADOW_LANE", "GPU-resident shadow serving lane (MI210): tenant-as-data scaffolding, shadow-only"),
    # Typed decision plane (TD-1, intake-1473): one-pass choice/score/noul
    # decisions over the llm_call seam. Default-off in BOTH test and prod;
    # TD-1 ships the core only and is NOT wired into any live route — TD-5
    # performs the wiring under this flag.
    FeatureSpec("typed_decisions", False, False, "TYPED_DECISIONS", "TD-1: one-pass typed decision plane (choice/score/noul)"),
    # TD-5: observability-only shadow of the incumbent routing decision. The
    # flag alone is not enough — a log path must also be configured via
    # ORCHESTRATOR_TYPED_DECISIONS_SHADOW_LOG, otherwise submit_shadow no-ops.
    FeatureSpec("typed_decisions_shadow", False, False, "TYPED_DECISIONS_SHADOW", "TD-5: log one typed-decision shadow call beside the incumbent routing decision; never changes routing"),
    # TD-4: closed-set typed selection of tool-call arguments, wired at the
    # REPL tool dispatch chokepoint (src/repl_environment/context.py). The
    # typed arm is fail-open: any decline keeps the model-provided arguments.
    FeatureSpec("typed_decisions_tool_args", False, False, "TYPED_DECISIONS_TOOL_ARGS", "TD-4: closed-set typed selection of tool-call arguments; fail-open to the model-provided arguments"),
    # UFH-12 REPL-EMB-1.1/1.4: placement-aware pooled embedding client with the
    # neighbour in-flight cap (src/embedding_pool/). Default OFF in BOTH test and
    # prod; with it off the factories return None and no consumer is migrated.
    FeatureSpec("repl_embedding_pool", False, False, "REPL_EMBEDDING_POOL", "UFH-12 REPL-EMB-1.1/1.4: pooled embedding client with placement-aware scheduling and the busy-frontdoor neighbour cap"),
    # Debug/Development
    FeatureSpec("mock_mode", True, False, "MOCK_MODE", "Mock mode for safety"),
)

# Indexed for fast lookup
_REGISTRY_BY_NAME: dict[str, FeatureSpec] = {s.name: s for s in _FEATURE_REGISTRY}


def runtime_flags_path() -> Path:
    """Shared runtime flag override file used by all API workers."""
    override = os.environ.get(RUNTIME_FLAGS_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "orchestration" / "runtime_flags.json"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_runtime_expiry(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        return None
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _validate_runtime_flag_ttl(ttl_s: Any) -> float:
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)):
        raise ValueError("ttl_s must be a finite number of seconds")
    try:
        ttl = float(ttl_s)
    except OverflowError as exc:
        raise ValueError("ttl_s must be a finite number of seconds") from exc
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("ttl_s must be a finite positive number of seconds")
    return ttl


def _runtime_records(
    path: Path | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    path = path or runtime_flags_path()
    as_of = as_of or _utc_now()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("flags", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for name, record in raw.items():
        if name not in _REGISTRY_BY_NAME:
            continue
        if isinstance(record, dict):
            value = _coerce_bool(record.get("value"))
            if value is None:
                continue
            normalized = {
                "value": value,
                "set_by": str(record.get("set_by") or "unknown"),
                "ts": str(record.get("ts") or ""),
            }
            if "expires_at" in record:
                expiry = _parse_runtime_expiry(record.get("expires_at"))
                if expiry is None:
                    logger.warning("Ignoring runtime flag %s with invalid or timezone-naive expiry", name)
                    continue
                if expiry <= as_of:
                    logger.warning("Ignoring expired runtime flag override %s; using environment/registry baseline", name)
                    continue
                normalized["expires_at"] = expiry.isoformat(timespec="seconds")
            records[name] = normalized
        else:
            value = _coerce_bool(record)
            if value is not None:
                records[name] = {"value": value, "set_by": "legacy", "ts": ""}
    return records


def runtime_flag_overrides(
    path: Path | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, bool]:
    """Return valid runtime overrides from the shared flag file."""
    return {
        name: bool(record["value"])
        for name, record in _runtime_records(path, as_of=as_of).items()
    }


def write_runtime_flag_overrides(
    overrides: dict[str, bool],
    *,
    set_by: str = "unknown",
    ttl_s: int | float | None = None,
    expires_at: datetime | str | None = None,
    path: Path | None = None,
) -> Path:
    """Atomically write runtime overrides; expiry metadata applies only to enables."""
    if ttl_s is not None and expires_at is not None:
        raise ValueError("specify ttl_s or expires_at, not both")
    path = path or runtime_flags_path()
    records = _runtime_records(path)
    now = _utc_now()
    ts = now.isoformat(timespec="seconds")
    expiry = None
    if ttl_s is not None:
        ttl = _validate_runtime_flag_ttl(ttl_s)
        try:
            expiry = now + timedelta(seconds=ttl)
        except OverflowError as exc:
            raise ValueError("ttl_s exceeds the supported datetime range") from exc
    elif expires_at is not None:
        expiry = _parse_runtime_expiry(expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at)
        if expiry is None or expiry <= now:
            raise ValueError("expires_at must be a future timezone-aware ISO-8601 timestamp")
    for name, value in overrides.items():
        if name not in _REGISTRY_BY_NAME:
            continue
        record = {
            "value": bool(value),
            "set_by": set_by,
            "ts": ts,
        }
        record_expiry = expiry
        if value and record_expiry is None and name == "repl_embedding_pool":
            try:
                record_expiry = now + timedelta(seconds=EXPERIMENT_FLAG_TTL_S)
            except OverflowError as exc:
                raise ValueError("default runtime flag expiry exceeds the supported datetime range") from exc
        if value and record_expiry is not None:
            record["expires_at"] = record_expiry.isoformat(timespec="seconds")
        records[name] = record
    payload = {
        "version": 1,
        "updated_at": ts,
        "flags": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


@dataclass
class Features:
    """Feature flags for optional orchestration modules.

    All features default to False for test isolation. Production code should
    use get_features() which reads from environment variables.

    Attributes:
        memrl: Memory-based Reinforcement Learning (Phase 4)
            - TaskEmbedder: BGE-large embeddings for task similarity (1024-dim)
            - QScorer: Q-value scoring for escalation decisions
            - HybridRouter: Learned + rule-based routing
            - EpisodicStore: SQLite storage for task memories
            Dependencies: numpy, sqlite3, sentence-transformers (optional)

        tools: Tool Registry for REPL
            - TOOL() function in REPL environment
            - Role-based permission checking
            - Built-in tools (lint, test, search)
            Dependencies: None (tools are pure Python)

        scripts: Script Registry for prepared scripts
            - SCRIPT() function in REPL
            - Semantic search for script matching
            Dependencies: tools feature (scripts can invoke tools)

        streaming: SSE streaming for chat responses
            - /chat/stream endpoint
            - Server-sent events for incremental output
            Dependencies: None

        openai_compat: OpenAI-compatible API endpoints
            - /v1/chat/completions
            - /v1/models
            Dependencies: None

        repl: REPL execution environment
            - Sandboxed Python execution
            - Context-as-variable pattern
            - peek(), grep(), FINAL() built-ins
            Dependencies: None

        caching: Response caching with prefix routing
            - CachingBackend for LLM responses
            - Prefix-based routing to workers
            Dependencies: None

        restricted_python: Use RestrictedPython for REPL sandboxing
            - More battle-tested security model
            - compile_restricted for safer compilation
            - Built-in guards against attribute access exploits
            Dependencies: RestrictedPython>=7.0
    """

    # Phase 4: MemRL (Memory-based Reinforcement Learning)
    memrl: bool = False

    # Tool and Script Registries
    tools: bool = True  # Enable TOOL() in REPL
    scripts: bool = False  # Scripts require script_registry.yaml to exist

    # API Features
    streaming: bool = False
    openai_compat: bool = False

    # Core Features (usually enabled)
    repl: bool = True
    caching: bool = True

    # Phase 2: Structured Delimiters for tool output isolation
    structured_delimiters: bool = True  # Wrap tool outputs with <<<TOOL_OUTPUT>>> delimiters

    # Phase 2: ReAct-style tool loop (direct mode with tool access)
    react_mode: bool = False  # Enable ReAct tool loop for direct-mode prompts

    # Phase 2: Output formalizer (format constraint enforcement)
    output_formalizer: bool = False  # Post-process answers to satisfy format constraints

    # Parallel read-only tool dispatch (ThreadPoolExecutor for independent REPL tools)
    parallel_tools: bool = True  # Dispatch independent read-only tools in parallel

    # Deferred tool result wrapping (keep mixin tool outputs out of prompt-by-default)
    deferred_tool_results: bool = False

    # Escalation prompt compression (LLMLingua-2 BERT for large prompts)
    escalation_compression: bool = False  # Compress prompts on architect escalation

    # Pre-routing optimization
    script_interception: bool = False  # Resolve trivial queries locally without LLM call

    # Security Features
    credential_redaction: bool = True  # Scan tool/REPL output for leaked credentials
    cascading_tool_policy: bool = True  # Layered tool permission chain (Global→Role→Task)
    restricted_python: bool = False  # Use RestrictedPython for REPL (requires library)

    # Phase 3: Specialist routing (MemRL-driven intelligent orchestration)
    specialist_routing: bool = False  # Enable specialist routing (coder, architect) via Q-values

    # Phase 3+: GraphRouter (GNN-based parallel routing signal for cold-start optimization)
    graph_router: bool = False  # Enable bipartite GAT routing predictor

    # Ingest-triviality guard: keep trivially-easy short prompts off the 80B
    # accuracy/long-context specialist (opt-in; requires specialist_routing)
    ingest_triviality_guard: bool = False  # Demote easy short prompts off ingest_long_context

    # Phase 3: Architect plan review (pre-execution plan vetting)
    plan_review: bool = False  # Enable architect review of frontdoor plans before execution

    # Phase 5: Architect delegation (investigate via specialist tools)
    architect_delegation: bool = False  # Architect delegates tool work to faster specialists

    # Phase 7: Parallel step execution (wave-based dependency ordering)
    parallel_execution: bool = False  # Enable wave-based step execution in ProactiveDelegator

    # Phase 8: Persona registry (dynamic prompt specialization)
    personas: bool = False  # Enable persona-based system prompt overlays

    # Phase 9: Staged reward shaping (PARL-inspired explore→exploit annealing)
    staged_rewards: bool = False  # Anneal exploration bonus in Q-value updates

    # MemRL Distillation: offline-trained routing classifier (ColBERT-Zero inspired)
    routing_classifier: bool = False  # Fast MLP routing before FAISS retrieval

    # SkillBank: Experience distillation into structured skills (SkillRL §3.1)
    skillbank: bool = False  # Enable SkillBank skill retrieval + prompt injection

    # Phase 4: Input formalizer (extract formal specs before specialist execution)
    input_formalizer: bool = False  # Preprocess complex prompts via MathSmith-8B

    # Unified streaming: route streaming endpoint through pipeline stages
    unified_streaming: bool = False  # Use stream_adapter.py instead of inline generator

    # Phase 10: Semantic classifiers (externalized keyword matching + MemRL routing)
    semantic_classifiers: bool = True  # Use config-driven classifiers from classifier_config.yaml

    # Generation Monitoring (Phase 6)
    generation_monitor: bool = True  # Enable early failure detection (post-hoc quality check)

    # OpenClaw/Lobster concept integration
    side_effect_tracking: bool = False  # Declare tool side effects for safety reasoning
    structured_tool_output: bool = False  # ToolOutput envelope (human + machine modes)
    model_fallback: bool = False  # Try same-tier alternatives on circuit-open
    content_cache: bool = False  # SHA-256 keyed response cache for LLM calls
    session_compaction: bool = False  # Summarize old context on long conversations
    session_log: bool = False  # Append-only processing journal across REPL turns
    session_scratchpad: bool = False  # Model-extracted semantic insights from session log
    depth_model_overrides: bool = False  # Map nested llm_call depth to cheaper roles
    resume_tokens: bool = False  # Base64url continuation tokens for crash recovery
    approval_gates: bool = False  # Human approval at escalation boundaries
    binding_routing: bool = False  # Priority-ordered routing overrides

    # LangGraph pre-migration: full state snapshots + generalized interrupts
    state_history_snapshots: bool = False  # Full TaskState snapshots each turn
    generalized_interrupts: bool = False  # Pluggable interrupt conditions before REPL

    # LangGraph migration Phase 1: hybrid bridge (run_task dispatches to LG backend)
    langgraph_bridge: bool = False  # Route orchestration through LangGraph instead of pydantic_graph

    # LangGraph migration Phase 3: per-node migration flags
    langgraph_ingest: bool = False             # Migrate IngestNode to LangGraph backend
    langgraph_architect: bool = False          # Migrate ArchitectNode to LangGraph backend
    langgraph_worker: bool = False             # Migrate WorkerNode to LangGraph backend
    langgraph_frontdoor: bool = False          # Migrate FrontdoorNode to LangGraph backend
    langgraph_coder: bool = False              # Migrate CoderNode to LangGraph backend
    langgraph_coder_escalation: bool = False   # Migrate CoderEscalationNode to LangGraph backend

    # Budget controls (Fast-RLM)
    worker_call_budget: bool = False  # Cap total REPL executions per task
    task_token_budget: bool = False   # Cap cumulative tokens across all turns
    final_schema_validation: bool = False  # Validate FINAL() value against caller JSON Schema

    # Pipeline monitoring: model-graded subjective evals
    model_grading: bool = False  # Post-hoc model-graded evals via the live worker-general path

    # P-BENCH-3 A7: opt-in eval-batch serving-class switch.
    eval_batch_serving: bool = False

    # HSD: Hierarchical Self-Speculation
    self_speculation: bool = False  # Self-speculation with layer-exit draft
    hierarchical_speculation: bool = False  # Hierarchical intermediate verification

    # Two-level condensation (Context-Folding Phase 1)
    two_level_condensation: bool = False  # Granular + deep consolidation instead of per-2-turn re-summarization

    # Context-Folding Phase 1+: segment hash dedup cache
    segment_cache_dedup: bool = False  # Hash-based dedup to skip LLM consolidation on repeated blocks

    # Context-Folding Phase 2c: heuristic helpfulness scoring
    helpfulness_scoring: bool = False  # Score segments by recency/overlap/outcome for compaction priority

    # Context-Folding Phase 3a: process reward telemetry
    process_reward_telemetry: bool = False  # Log token_budget_ratio, on_scope, tool_success per turn

    # Context-Folding Phase 3b: role-aware compaction profiles
    role_aware_compaction: bool = False  # Per-role compaction aggressiveness (architect conservative, worker aggressive)

    # Context window management (C2/C3/C1)
    accurate_token_counting: bool = False  # Use llama-server /tokenize for exact token counts
    tool_result_clearing: bool = False  # Clear stale <<<TOOL_OUTPUT>>> blocks from last_output

    # Reasoning length alarm (short-m@k Action 9): retry with conciseness nudge
    reasoning_length_alarm: bool = False  # Cancel + retry when <think> exceeds 1.5× band budget

    # Tool output compression (Phase 2 native): compress verbose output before prompt injection
    tool_output_compression: bool = False  # Compress pytest/git/build output to preserve actionable info only

    # CMV-style output spill (Action 11): write truncated output/error to temp file with peek() pointer
    output_spill_to_file: bool = False  # Spill long REPL output/error to file + retrieval pointer

    # Conversation Management (B-series cherry-picks from Hermes/OpenGauss)
    injection_scanning: bool = False  # B7: Prompt injection scanning on loaded context
    context_compression: bool = False  # B2: Protected-zone context compression
    user_modeling: bool = False  # B1: Cross-session user preference modeling
    session_token_budget: bool = False  # B5: Per-session token budget with compact/stop signals

    # Claude Code Local Integration (CC Local)
    claude_code_mcp_chat: bool = False  # MCP tools for delegating chat to running orchestrator
    v1_client_session_guard: bool = False  # HS-4: 422 on OpenCode/client-mode /v1 requests without x_session_id
    v1_subagent_link: bool = False  # HS-19a stage 1: record /v1 subagent parent links (tap + session log); default off
    v1_escalation: bool = False  # TE-1: opt-in /v1 frontdoor escalation via /chat's post-answer hooks (explicit x_escalation); default off
    thinking_roles_chat_lane: bool = False  # RI-23: thinking-on roles on /v1/chat/completions (kwargs + reasoning_content); default off

    # Web research reranking (ColBERT snippet pre-fetch filtering)
    web_research_rerank: bool = False  # Rerank DDG snippets via ColBERT before page fetch

    # Routing telemetry: capture first-token logprobs for learned routing P1.5
    logit_probe: bool = False  # Log top-64 first-token logprobs from frontdoor to JSONL

    # NIB2-45 MindDR: feature-flag-gated three-agent research pipeline (Planning → DeepSearch → Report).
    # When on, research-like queries (per classifier.is_research_like) route through the MindDR subgraph
    # instead of the direct flow. Phase 1 is prompt-level only; Phase 2 training is GPU-gated.
    deep_research_mode: bool = False

    # intake-607 harness cluster (default-off; shadow/observe only)
    ure_uncertainty_shadow_log: bool = False  # URE-1 (J10): shadow-log routing uncertainty; no behavior change
    batch_edit_mode: bool = False  # BEP-1 (J8): coder emits one structured patch set after reasoning
    interleaved_edit_rider: bool = False  # BEP-2 baseline: per-turn interleaved-edit rider (experiment-only)
    dcp_pre_assembly: bool = False  # DCP-4 (J7): advisory delegation context pre-assembly (seed bundle)
    dcp_for_consult: bool = False  # P2 consult may reuse DCP seed context; requires dcp_pre_assembly
    review_before_commit_consult: bool = False  # P2 edit-transaction consult seam; default-off
    review_before_commit_targeted_gate: bool = False  # J17 targeted high-risk edit-shape consult gate

    # intake-614/615 DAR-6 (default-off; scaffolding only — no production routing until DAR-6.5 A/B clears)
    swarm_fanout: bool = False  # DAR-6.1: fan high-injection-risk prompts to N≥2 concurrent serves + BT-aggregate (J14)
    # P21.A test-time compute: DeepConf offline confidence-filtered self-consistency (intake-603).
    # When on, the caller filters low-confidence reasoning traces (bottom-10% group confidence
    # from per-token top-k logprobs) and takes a confidence-weighted majority vote. Default OFF;
    # build + scoring logic in src/test_time/deepconf.py. Wiring onto N parallel llama-server
    # completions (payload n_probs=K) is P21.A3, gated on the A2 live-server sanity check.
    deepconf: bool = False

    # RD-5 Reviewer decision plane (shadow/enforce split).
    # review_decision_shadow: emit + trace review decisions (verifier precedence,
    #   reject-admissibility, warn-only downgrade) but NEVER act. Deliberately has
    #   NO memrl dependency — this is the decoupled shadow-emission path so shadow
    #   data flows without flipping plan_review (which requires memrl).
    # review_decision_enforce: act on review decisions. Default OFF and BLOCKED on
    #   the H-LB LB-6 latency/sampling budget gate — do not enable until it passes.
    review_decision_shadow: bool = False
    review_decision_enforce: bool = False

    # GPU shadow lane (gpu-serving-tie-in-program P0-7): default OFF everywhere.
    # Gates the np_ceiling policy loader + preflight scaffolding only; no
    # production routing or launch path reads this flag (D3 shadow-only).
    gpu_shadow_lane: bool = False

    # Typed decision plane (TD-1, intake-1473): default OFF everywhere. TD-1
    # ships the core (types/confidence/schema/runner) only; no route reads
    # this flag yet — TD-5 wires the plane in under it.
    typed_decisions: bool = False

    # TD-5: observability-only shadow of the incumbent routing decision.
    # Default OFF; also requires ORCHESTRATOR_TYPED_DECISIONS_SHADOW_LOG to be
    # set — with no configured sink the shadow no-ops even when on.
    typed_decisions_shadow: bool = False

    # TD-4: closed-set typed selection of tool-call arguments at the REPL tool
    # dispatch chokepoint. Default OFF in both test and prod; fail-open — any
    # decline keeps the model-provided arguments.
    typed_decisions_tool_args: bool = False

    # UFH-12 REPL-EMB-1.1/1.4: pooled embedding client (src/embedding_pool/).
    # Default OFF in both test and prod; get_pooled_embedder()/get_sync_embedder()
    # return None while off, so no live path changes until a consumer migrates.
    repl_embedding_pool: bool = False

    # RTE-Prefix (repl-turn-efficiency): render fixed prompt sections (task,
    # instruction) BEFORE per-turn-mutating ones (state, context,
    # reference_code) in RootLMPrompt.to_string(), so successive REPL turns
    # share a longer llama-server KV-cache prefix. Default OFF — prompt
    # behavioral change, pending A/B validation.
    prefix_stable_order: bool = False

    # UFH14-B1 F1: derived silent-prefill allowance on long-prompt call timeouts.
    derived_prefill_timeout: bool = False
    # UFH14-B1: LLM-written session-compaction index (off = deterministic).
    session_compaction_llm_index: bool = False
    # UFH14-B1 F2: wall-time forced-answer turn in the REPL graph.
    repl_answer_force: bool = False

    # Debug/Development
    mock_mode: bool = True  # Default to mock mode for safety

    def validate(self) -> list[str]:
        """Validate feature dependencies.

        Returns:
            List of validation errors (empty if all valid).
        """
        errors = []

        # Scripts require tools
        if self.scripts and not self.tools:
            errors.append("scripts feature requires tools feature")

        # MemRL-dependent features
        if self.specialist_routing and not self.memrl:
            errors.append("specialist_routing feature requires memrl feature")
        if self.graph_router and not self.specialist_routing:
            errors.append("graph_router feature requires specialist_routing feature")
        if self.plan_review and not self.memrl:
            errors.append("plan_review feature requires memrl feature")
        if self.architect_delegation and not self.memrl:
            errors.append("architect_delegation feature requires memrl feature")
        if self.parallel_execution and not self.architect_delegation:
            errors.append("parallel_execution feature requires architect_delegation feature")
        if self.personas and not self.memrl:
            errors.append("personas feature requires memrl feature")
        if self.staged_rewards and not self.memrl:
            errors.append("staged_rewards feature requires memrl feature")
        if self.skillbank and not self.memrl:
            errors.append("skillbank feature requires memrl feature")

        # Approval gates require resume tokens and side effect tracking
        if self.approval_gates and not self.resume_tokens:
            errors.append("approval_gates feature requires resume_tokens feature")
        if self.approval_gates and not self.side_effect_tracking:
            errors.append("approval_gates feature requires side_effect_tracking feature")

        # Generalized interrupts depend on approval gates and resume tokens
        if self.generalized_interrupts and not self.approval_gates:
            errors.append("generalized_interrupts requires approval_gates")
        if self.generalized_interrupts and not self.resume_tokens:
            errors.append("generalized_interrupts requires resume_tokens")

        # User modeling requires injection scanning for write safety
        if self.user_modeling and not self.injection_scanning:
            errors.append("user_modeling feature requires injection_scanning feature")
        if self.dcp_for_consult and not self.dcp_pre_assembly:
            errors.append("dcp_for_consult feature requires dcp_pre_assembly feature")

        # RestrictedPython requires the library
        if self.restricted_python:
            try:
                import RestrictedPython  # noqa: F401
            except ImportError:
                errors.append(
                    "restricted_python feature requires RestrictedPython library: "
                    "pip install RestrictedPython>=7.0"
                )

        return errors

    def summary(self) -> dict[str, bool]:
        """Get summary of all feature flags (derived from registry).

        Returns:
            Dictionary of feature name -> enabled status.
        """
        return {spec.name: getattr(self, spec.name) for spec in _FEATURE_REGISTRY}

    def enabled_features(self) -> list[str]:
        """Get list of enabled feature names.

        Returns:
            List of enabled feature names.
        """
        return [name for name, enabled in self.summary().items() if enabled]


def _feature_flag_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from environment variable.

    Truthy values: 1, true, yes, on (case-insensitive)
    Falsy values: 0, false, no, off (case-insensitive)

    Args:
        name: Environment variable name (without prefix).
        default: Default value if not set.

    Returns:
        Boolean value.
    """
    key = f"{ENV_PREFIX}{name.upper()}"
    return env_bool(key, default)


def _compute_feature_flags(
    *,
    production: bool = False,
    override: dict[str, bool] | None = None,
    runtime_at: datetime | None = None,
) -> tuple[dict[str, bool], dict[str, str]]:
    defaults = {
        spec.name: (spec.default_prod if production else spec.default_test)
        for spec in _FEATURE_REGISTRY
    }
    flags: dict[str, bool] = {}
    sources: dict[str, str] = {}
    for spec in _FEATURE_REGISTRY:
        env_key = f"{ENV_PREFIX}{spec.env_var}"
        feature_env_key = f"{FEATURE_ENV_PREFIX}{spec.env_var}"
        if env_key in os.environ:
            value = _feature_flag_bool(spec.env_var, defaults[spec.name])
            source = env_key
        elif feature_env_key in os.environ:
            value = env_bool(feature_env_key, defaults[spec.name])
            source = feature_env_key
        else:
            value = defaults[spec.name]
            source = "default_prod" if production else "default_test"
        flags[spec.name] = value
        sources[spec.name] = source

    known_names = set(defaults)
    runtime_path = runtime_flags_path()
    for name, value in runtime_flag_overrides(runtime_path, as_of=runtime_at).items():
        if name not in known_names:
            continue
        flags[name] = value
        sources[name] = f"runtime_file:{runtime_path}"

    if override:
        for name, value in override.items():
            if name not in known_names:
                continue
            flags[name] = bool(value)
            sources[name] = "override"

    return flags, sources


def get_features(
    *,
    production: bool = False,
    override: dict[str, bool] | None = None,
    _runtime_at: datetime | None = None,
) -> Features:
    """Get feature flags from environment variables.

    In production mode (production=True), most features default to enabled.
    In test mode (production=False), most features default to disabled.

    Defaults, env-var names, and flag inventory are driven by ``_FEATURE_REGISTRY``
    — the single source of truth. Adding a new flag only requires a new
    ``FeatureSpec`` entry (plus the matching dataclass field on ``Features``).

    Args:
        production: If True, use production defaults (most features on).
        override: Explicit overrides for specific features.

    Returns:
        Features instance with flags set.

    Example:
        # Read from environment
        features = get_features()

        # Production defaults
        features = get_features(production=True)

        # Test with specific features
        features = get_features(override={"memrl": True, "tools": False})
    """
    flags, _sources = _compute_feature_flags(
        production=production, override=override, runtime_at=_runtime_at
    )
    return Features(**flags)


def feature_sources(*, production: bool = False) -> dict[str, str]:
    """Return the source for each effective feature value."""
    _flags, sources = _compute_feature_flags(production=production)
    return sources


# Singleton for global access (lazy-loaded, thread-safe)
_features: Features | None = None
_features_lock = threading.Lock()
_features_runtime_path: Path | None = None
_features_runtime_mtime: float | None = None
_features_runtime_last_check = 0.0
_features_runtime_next_expiry: float | None = None


def _runtime_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _runtime_next_expiry(path: Path, *, loaded_at: datetime) -> float | None:
    # Inspect the raw records so an expiry that crosses between feature loading
    # and this calculation still schedules an immediate cache refresh.
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("flags", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return None
    expiries = [
        _parse_runtime_expiry(record["expires_at"])
        for record in raw.values()
        if isinstance(record, dict) and "expires_at" in record
    ]
    valid = [expiry for expiry in expiries if expiry is not None and expiry > loaded_at]
    if not valid:
        return None
    seconds_left = min((expiry - _utc_now()).total_seconds() for expiry in valid)
    return time.monotonic() + max(0.0, seconds_left)


def features() -> Features:
    """Get the global Features instance (lazy-loaded from environment).

    Thread-safe via double-checked locking (matches PromptCompressor,
    WorkerPoolManager patterns).

    For most code, use this function:
        from src.features import features
        if features().memrl:
            ...

    Returns:
        Global Features instance.
    """
    global _features, _features_runtime_last_check, _features_runtime_mtime, _features_runtime_path
    global _features_runtime_next_expiry
    now = time.monotonic()
    path = runtime_flags_path()
    with _features_lock:
        if _features is None:
            loaded_at = _utc_now()
            _features = get_features(_runtime_at=loaded_at)
            _features_runtime_path = path
            _features_runtime_mtime = _runtime_mtime(path)
            _features_runtime_next_expiry = _runtime_next_expiry(path, loaded_at=loaded_at)
            _features_runtime_last_check = now
            return _features

        if now - _features_runtime_last_check >= RUNTIME_FLAGS_TTL_S:
            mtime = _runtime_mtime(path)
            expired = _features_runtime_next_expiry is not None and now >= _features_runtime_next_expiry
            if path != _features_runtime_path or mtime != _features_runtime_mtime or expired:
                loaded_at = _utc_now()
                _features = get_features(_runtime_at=loaded_at)
                _features_runtime_path = path
                _features_runtime_mtime = mtime
                _features_runtime_next_expiry = _runtime_next_expiry(path, loaded_at=loaded_at)
            _features_runtime_last_check = now
        return _features


def reset_features() -> None:
    """Reset the global Features instance (useful for tests).

    Call this to re-read feature flags from environment.
    """
    global _features, _features_runtime_last_check, _features_runtime_mtime, _features_runtime_path
    global _features_runtime_next_expiry
    with _features_lock:
        _features = None
        _features_runtime_path = None
        _features_runtime_mtime = None
        _features_runtime_last_check = 0.0
        _features_runtime_next_expiry = None


def set_features(new_features: Features) -> None:
    """Set the global Features instance (useful for tests).

    Args:
        new_features: Features instance to use globally.
    """
    global _features, _features_runtime_last_check, _features_runtime_mtime, _features_runtime_path
    global _features_runtime_next_expiry
    with _features_lock:
        loaded_at = _utc_now()
        _features = new_features
        _features_runtime_path = runtime_flags_path()
        _features_runtime_mtime = _runtime_mtime(_features_runtime_path)
        _features_runtime_next_expiry = _runtime_next_expiry(
            _features_runtime_path, loaded_at=loaded_at
        )
        _features_runtime_last_check = time.monotonic()
