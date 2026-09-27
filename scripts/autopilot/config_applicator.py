"""Config applicator: route parameter changes to correct application method.

- Hot-swap (no restart): POST /config for feature flags + runtime config
- API restart: uvicorn reload for code-level changes
- Model server restart: orchestrator_stack.py (expensive, avoid)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from hashlib import sha256
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:  # H4: cross-process single-writer lock on autopilot_state.json
    from scripts.autopilot.state_lock import (
        PAUSE_OWNER_FIELD,
        claim_pause_lease,
        release_pause_lease,
        state_write_lock,
    )
except ImportError:  # pragma: no cover - path-dependent import
    from state_lock import (
        PAUSE_OWNER_FIELD,
        claim_pause_lease,
        release_pause_lease,
        state_write_lock,
    )

#: Lease owner stamped on ``autopilot_state.json`` while this applicator holds the
#: dispatch pause around a role/API restart.
DISPATCH_PAUSE_OWNER = "config_applicator"

log = logging.getLogger("autopilot.config")

ORCHESTRATOR_URL = "http://localhost:8000"
ORCH_ROOT = Path(__file__).resolve().parents[1].parent
DEFAULT_STACK_PRIORS_PATH = ORCH_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_AUTOPILOT_STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
DEFAULT_MODEL_REGISTRY_PATH = ORCH_ROOT / "orchestration" / "model_registry.yaml"

# Parameters that can be hot-swapped via POST /config (feature flags)
HOT_SWAP_FEATURES = {
    "memrl", "tools", "scripts", "streaming", "openai_compat", "repl",
    "caching", "specialist_routing", "architect_delegation", "session_log",
    "generation_monitor", "think_harder", "graph_router", "skillbank",
    "routing_classifier", "staged_rewards", "session_compaction",
    "try_cheap_first", "long_context", "web_search", "web_research",
    "cascading_tool_policy", "factual_risk",
    # Tier 3 promotions 2026-05-20: opt-in via StructuralLab's flag-mutation
    # pool. All three were previously default-off and required hand-rolled
    # experiments per the rlm-orchestrator-roadmap.md R6 candidate matrix.
    "structured_tool_output",  # Lobster ToolOutput envelope on tool invocations
    "content_cache",            # SHA-256 keyed LLM-response cache
    "model_fallback",           # Same-tier alternatives on circuit-open
}

# Parameters applied via env vars (require API restart)
ENV_PARAMS = {
    "memrl_retrieval": {
        "q_weight": "ORCHESTRATOR_MEMRL_RETRIEVAL_Q_WEIGHT",
        "min_similarity": "ORCHESTRATOR_MEMRL_RETRIEVAL_MIN_SIMILARITY",
        "min_q_value": "ORCHESTRATOR_MEMRL_RETRIEVAL_MIN_Q_VALUE",
        "confidence_threshold": "ORCHESTRATOR_MEMRL_RETRIEVAL_CONFIDENCE_THRESHOLD",
        "semantic_k": "ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K",
        "prior_strength": "ORCHESTRATOR_MEMRL_RETRIEVAL_PRIOR_STRENGTH",
    },
    "think_harder": {
        "min_expected_roi": "ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI",
        "token_budget_min": "ORCHESTRATOR_THINK_HARDER_TOKEN_BUDGET_MIN",
        "token_budget_max": "ORCHESTRATOR_THINK_HARDER_TOKEN_BUDGET_MAX",
        "cot_roi_threshold": "ORCHESTRATOR_THINK_HARDER_COT_ROI_THRESHOLD",
    },
    "monitor": {
        "entropy_threshold": "ORCHESTRATOR_MONITOR_ENTROPY_THRESHOLD",
        "repetition_threshold": "ORCHESTRATOR_MONITOR_REPETITION_THRESHOLD",
        "entropy_spike_threshold": "ORCHESTRATOR_MONITOR_ENTROPY_SPIKE_THRESHOLD",
    },
    "chat": {
        "try_cheap_first_q_threshold": "ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD",
        # Back-compat for historical rows/blacklists emitted before the live
        # Phase-B/C gate name was corrected.
        "try_cheap_first_quality_threshold": "ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD",
        "long_context_threshold_chars": "ORCHESTRATOR_CHAT_LONG_CONTEXT_THRESHOLD_CHARS",
        "summarization_threshold_tokens": "ORCHESTRATOR_CHAT_SUMMARIZATION_THRESHOLD_TOKENS",
        "review_low_q_threshold": "ORCHESTRATOR_CHAT_REVIEW_LOW_Q_THRESHOLD",
        "review_skip_q_threshold": "ORCHESTRATOR_CHAT_REVIEW_SKIP_Q_THRESHOLD",
    },
    "escalation": {
        "max_retries": "ORCHESTRATOR_ESCALATION_MAX_RETRIES",
        "max_escalations": "ORCHESTRATOR_ESCALATION_MAX_ESCALATIONS",
    },
    "repl": {
        "turn_token_cap": "ORCHESTRATOR_REPL_TURN_N_TOKENS",
        # Tier 1 wire-in 2026-05-20:
        "frontdoor_non_tool_token_cap": "ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS",
        "worker_call_budget_cap": "ORCHESTRATOR_WORKER_CALL_BUDGET_CAP",
        "task_token_budget_cap": "ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP",
    },
}


# ── Reviewer control-plane governance knobs (AP-1 / RD-11, class 1) ────────────
#
# The "class-1 tuning surface" declared by the reviewer decision plane
# (reviewer-decision-plane.md RD-11) and consumed by the autopilot (H8 AP-1).
# These are env-backed delegation-config knobs, so classify_params routes them
# through the env_restart surface (an API reload picks up the new environment).
#
# COORDINATE-BY-CONVENTION: a parallel effort (W2c) declares the SAME knob set in
# the guarded numeric-surface manifest (`orchestration/review_plane_knobs.yaml`).
# This module owns only the apply_params PLUMBING that maps the manifest's bare
# knob names -> the concrete env/config write. `load_review_plane_knob_manifest`
# reads that manifest if it exists and overlays its bounds/defaults; when the file
# is absent the built-in `REVIEW_PLANE_KNOB_SPECS` below are the source of truth.
#
# restart-cost annotation: today every knob is env-backed => "api_restart" (a
# cheap orchestrator-only reload, NOT a model-server restart). The decision plane
# intends these to become no-restart runtime flags once RD-5's decoupled
# activation path lands; at that point flip the affected specs' restart_cost to
# "none" and their apply_surface to "hot_swap". The annotation is honest now and
# is what the species budget reads to price a review-policy trial.
ENV_PARAMS["delegation"] = {
    "review_trigger_complexity_threshold": (
        "ORCHESTRATOR_DELEGATION_REVIEW_TRIGGER_COMPLEXITY_THRESHOLD"
    ),
    "max_review_iterations": "ORCHESTRATOR_DELEGATION_MAX_REVIEW_ITERATIONS",
    "reminder_cadence": "ORCHESTRATOR_DELEGATION_REMINDER_CADENCE",
    "per_subtask_review_enabled": "ORCHESTRATOR_DELEGATION_PER_SUBTASK_REVIEW_ENABLED",
    "review_majority_k": "ORCHESTRATOR_DELEGATION_REVIEW_MAJORITY_K",
    "request_evidence_round_budget": "ORCHESTRATOR_DELEGATION_REQUEST_EVIDENCE_ROUND_BUDGET",
    "review_token_multiplier": "ORCHESTRATOR_DELEGATION_REVIEW_TOKEN_MULTIPLIER",
}

_REVIEW_PLANE_SECTION = "delegation"
_AXA3_POLICY_SECTION = "placement_policy"
_AP3_ROLE_RESTART_SECTION = "role_restart"
_AP3_ROLE_RESTART_ENABLE_ENV = "AUTOPILOT_AP3_ROLE_RESTART_ENABLE"
DEFAULT_REVIEW_PLANE_KNOB_MANIFEST = (
    ORCH_ROOT / "orchestration" / "review_plane_knobs.yaml"
)
DEFAULT_AXA3_POLICY_KNOB_MANIFEST = DEFAULT_REVIEW_PLANE_KNOB_MANIFEST
DEFAULT_AP3_ROLE_RESTART_KNOB_MANIFEST = DEFAULT_REVIEW_PLANE_KNOB_MANIFEST

ENV_PARAMS[_AXA3_POLICY_SECTION] = {
    "teleport_enabled": "ORCHESTRATOR_PLACEMENT_POLICY_TELEPORT_ENABLED",
    "long_running_trigger_tokens": (
        "ORCHESTRATOR_PLACEMENT_POLICY_LONG_RUNNING_TRIGGER_TOKENS"
    ),
    "rate_window_tokens": "ORCHESTRATOR_PLACEMENT_POLICY_RATE_WINDOW_TOKENS",
    "min_resident_remaining_tokens": (
        "ORCHESTRATOR_PLACEMENT_POLICY_MIN_RESIDENT_REMAINING_TOKENS"
    ),
    "min_speedup": "ORCHESTRATOR_PLACEMENT_POLICY_MIN_SPEEDUP",
    "lease_interactive_weight": "ORCHESTRATOR_PLACEMENT_POLICY_LEASE_INTERACTIVE_WEIGHT",
    "lease_batch_weight": "ORCHESTRATOR_PLACEMENT_POLICY_LEASE_BATCH_WEIGHT",
    "lease_eval_weight": "ORCHESTRATOR_PLACEMENT_POLICY_LEASE_EVAL_WEIGHT",
}


@dataclass(frozen=True)
class ReviewPlaneKnob:
    """One class-1 governance knob: bounds + restart-cost + apply target.

    ``apply_key`` is the dotted key ``classify_params`` understands (e.g.
    ``delegation.max_review_iterations``); ``name`` is the bare manifest/action
    name the controller emits. ``validate`` returns an error string or ``None``.
    """

    name: str
    kind: str  # "float" | "int" | "bool"
    default: Any
    lo: float | None = None
    hi: float | None = None
    restart_cost: str = "api_restart"  # "none" | "api_restart" | "role_restart"
    apply_surface: str = "env_restart"  # classify_params bucket the value lands in
    doc: str = ""
    section: str = _REVIEW_PLANE_SECTION

    @property
    def apply_key(self) -> str:
        return f"{self.section}.{self.name}"

    def coerce(self, value: Any) -> Any:
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if self.kind == "int":
            return int(value)
        return float(value)

    def validate(self, value: Any) -> str | None:
        try:
            coerced = self.coerce(value)
        except (TypeError, ValueError):
            return f"{self.name}: value {value!r} is not a valid {self.kind}"
        if self.kind == "bool":
            return None
        if self.lo is not None and coerced < self.lo:
            return f"{self.name}: {coerced} < min {self.lo}"
        if self.hi is not None and coerced > self.hi:
            return f"{self.name}: {coerced} > max {self.hi}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "lo": self.lo,
            "hi": self.hi,
            "restart_cost": self.restart_cost,
            "apply_surface": self.apply_surface,
            "apply_key": self.apply_key,
            "section": self.section,
            "doc": self.doc,
        }


_KV_PROFILE_VALUES: dict[str, dict[str, str]] = {
    "q8_0_q8_0": {"k": "q8_0", "v": "q8_0"},
    "f16_f16": {"k": "f16", "v": "f16"},
}


@dataclass(frozen=True)
class RoleRestartKnob:
    """One AP-3 role-restart launch knob with registry-override mapping."""

    name: str
    kind: str  # "float" | "int" | "enum"
    default: Any
    role: str
    registry_path: str | None = None
    lo: float | None = None
    hi: float | None = None
    allowed_values: tuple[str, ...] = ()
    restart_cost: str = "role_restart"
    doc: str = ""
    section: str = _AP3_ROLE_RESTART_SECTION

    @property
    def apply_key(self) -> str:
        return f"{self.section}.{self.name}"

    def coerce(self, value: Any) -> Any:
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        if self.kind == "enum":
            return str(value).strip()
        raise ValueError(f"unknown knob kind {self.kind!r}")

    def validate(self, value: Any) -> str | None:
        try:
            coerced = self.coerce(value)
        except (TypeError, ValueError):
            return f"{self.name}: value {value!r} is not a valid {self.kind}"
        if self.kind == "enum":
            if coerced not in self.allowed_values:
                allowed = ", ".join(self.allowed_values)
                return f"{self.name}: {coerced!r} not in allowed values [{allowed}]"
            return None
        if self.lo is not None and coerced < self.lo:
            return f"{self.name}: {coerced} < min {self.lo}"
        if self.hi is not None and coerced > self.hi:
            return f"{self.name}: {coerced} > max {self.hi}"
        return None

    def registry_overrides(self, value: Any) -> dict[str, Any]:
        coerced = self.coerce(value)
        if self.name.endswith("_kv_profile"):
            profile = _KV_PROFILE_VALUES.get(str(coerced))
            if profile is None:
                raise ValueError(f"{self.name}: unknown KV profile {coerced!r}")
            if not self.registry_path:
                raise ValueError(f"{self.name}: registry_path is required")
            return {self.registry_path: dict(profile)}
        if not self.registry_path:
            raise ValueError(f"{self.name}: registry_path is required")
        return {self.registry_path: coerced}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "lo": self.lo,
            "hi": self.hi,
            "role": self.role,
            "registry_path": self.registry_path,
            "allowed_values": list(self.allowed_values),
            "restart_cost": self.restart_cost,
            "apply_key": self.apply_key,
            "section": self.section,
            "doc": self.doc,
        }


# Built-in specs. Bounds/defaults/kinds are aligned to the W2c numeric-surface
# manifest (orchestration/review_plane_knobs.yaml v1); all defaults are the
# behavior-preserving values (NOT the LB-2 `proposed` targets). ``restart_cost``
# here is the REALIZED cost of THIS module's apply surface (env_restart =>
# "api_restart"); the manifest DECLARES these as "none"/runtime (RD-5's future
# no-restart surface). load_review_plane_knob_manifest() surfaces the declared view.
REVIEW_PLANE_KNOB_SPECS: dict[str, ReviewPlaneKnob] = {
    knob.name: knob
    for knob in (
        ReviewPlaneKnob(
            "review_trigger_complexity_threshold", "int", 0, 0, 3,
            doc="TaskComplexity rank (TRIVIAL=0..COMPLEX=3) at/above which a "
            "subtask is reviewed; below routes to final-aggregate review (RD-10).",
        ),
        ReviewPlaneKnob(
            "max_review_iterations", "int", 3, 1, 5,
            doc="Max review->revise cycles per subtask before giving up (RD-9).",
        ),
        ReviewPlaneKnob(
            "reminder_cadence", "int", 0, 0, 50,
            doc="Re-inject the plan reminder every N steps/tool-calls (0=off; "
            "RD-9, preferred over re-review).",
        ),
        ReviewPlaneKnob(
            "per_subtask_review_enabled", "bool", True,
            doc="Review every subtask output (True) vs single final-aggregate "
            "review (False) (RD-10). Default True preserves current behavior.",
        ),
        ReviewPlaneKnob(
            "review_majority_k", "int", 1, 1, 5,
            doc="Grade each candidate k times and take the majority near band "
            "edges (RD-2; judge flakiness 2-9%).",
        ),
        ReviewPlaneKnob(
            "request_evidence_round_budget", "int", 0, 0, 3,
            doc="How many request_evidence -> verifier rounds are allowed before "
            "falling back (RD-3/RD-4).",
        ),
        ReviewPlaneKnob(
            "review_token_multiplier", "float", 2.0, 1.0, 2.0,
            doc="Scales the reviewer's token budget relative to the base "
            "grading budget (H-LB latency lever).",
        ),
    )
}

# dotted apply_key -> spec, for validating already-qualified params.
_REVIEW_PLANE_BY_APPLY_KEY: dict[str, ReviewPlaneKnob] = {
    knob.apply_key: knob for knob in REVIEW_PLANE_KNOB_SPECS.values()
}


AXA3_POLICY_KNOB_SPECS: dict[str, ReviewPlaneKnob] = {
    knob.name: knob
    for knob in (
        ReviewPlaneKnob(
            "teleport_enabled", "bool", False,
            doc="Default-off AXA-3 operator gate. Registered for explicit staging "
            "only; NumericSwarm does not sweep this enable flag.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "long_running_trigger_tokens", "int", 128, 0, 4096,
            doc="Generated-token count before a stream can be considered long-running "
            "enough for v1 re-prefill cutover.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "rate_window_tokens", "int", 64, 16, 1024,
            doc="Decode-rate observation window, in generated tokens, used by the "
            "future long-running stream detector.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "min_resident_remaining_tokens", "int", 150, 64, 1024,
            doc="Resident-lane positive-savings break-even threshold for estimated "
            "remaining tokens. Cold-load break-even stays operator-gated.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "min_speedup", "float", 1.05, 1.0, 4.0,
            doc="Minimum GPU/CPU decode-rate ratio before teleport can be considered.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "lease_interactive_weight", "float", 1.0, 0.0, 4.0,
            doc="MI210 lease priority weight for interactive workload mix.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "lease_batch_weight", "float", 0.25, 0.0, 4.0,
            doc="MI210 lease priority weight for batch/offline workload mix.",
            section=_AXA3_POLICY_SECTION,
        ),
        ReviewPlaneKnob(
            "lease_eval_weight", "float", 0.1, 0.0, 4.0,
            doc="MI210 lease priority weight for eval/benchmark workload mix.",
            section=_AXA3_POLICY_SECTION,
        ),
    )
}

_AXA3_POLICY_BY_APPLY_KEY: dict[str, ReviewPlaneKnob] = {
    knob.apply_key: knob for knob in AXA3_POLICY_KNOB_SPECS.values()
}


AP3_ROLE_RESTART_KNOB_SPECS: dict[str, RoleRestartKnob] = {
    knob.name: knob
    for knob in (
        RoleRestartKnob(
            "frontdoor_spec_type", "enum", "draft-mtp",
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.spec_type",
            allowed_values=("draft-mtp", "ngram-mod,draft-mtp"),
            doc="Frontdoor composed-spec launch type. ngram composition is gated by task-level quality evidence.",
        ),
        RoleRestartKnob(
            "frontdoor_draft_max", "int", 4,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.draft_max",
            lo=1,
            hi=8,
            doc="Frontdoor --spec-draft-n-max for NEXTN/draft-mtp launches.",
        ),
        RoleRestartKnob(
            "frontdoor_draft_min", "int", 0,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.draft_min",
            lo=0,
            hi=8,
            doc="Frontdoor --spec-draft-n-min for NEXTN/draft-mtp launches.",
        ),
        RoleRestartKnob(
            "frontdoor_draft_p_min", "float", 0.0,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.draft_p_min",
            lo=0.0,
            hi=0.5,
            doc="Frontdoor --draft-p-min for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "frontdoor_draft_p_split", "float", 0.0,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.draft_p_split",
            lo=0.0,
            hi=1.0,
            doc="Frontdoor --draft-p-split for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "frontdoor_ngram_mod_n_min", "int", 48,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.ngram_mod_n_min",
            lo=0,
            hi=1024,
            doc="Frontdoor --spec-ngram-mod-n-min for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "frontdoor_ngram_mod_n_max", "int", 64,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.ngram_mod_n_max",
            lo=0,
            hi=1024,
            doc="Frontdoor --spec-ngram-mod-n-max for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "frontdoor_ngram_mod_n_match", "int", 24,
            role="frontdoor",
            registry_path="server_mode.frontdoor.acceleration.ngram_mod_n_match",
            lo=1,
            hi=1024,
            doc="Frontdoor --spec-ngram-mod-n-match for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "frontdoor_kv_profile", "enum", "q8_0_q8_0",
            role="frontdoor",
            registry_path="server_mode.frontdoor.kv_quant",
            allowed_values=tuple(_KV_PROFILE_VALUES),
            doc="Frontdoor launch KV profile. Mixed K/V profiles stay out of AP-3 until lane-specific evidence clears.",
        ),
        RoleRestartKnob(
            "worker_spec_type", "enum", "draft-mtp",
            role="worker_general",
            registry_path="server_mode.worker.acceleration.spec_type",
            allowed_values=("draft-mtp", "ngram-mod,draft-mtp"),
            doc="Worker composed-spec launch type; broad worker default remains draft-mtp until ngram quality clears.",
        ),
        RoleRestartKnob(
            "worker_draft_max", "int", 2,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.draft_max",
            lo=1,
            hi=8,
            doc="Worker --spec-draft-n-max for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "worker_draft_min", "int", 0,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.draft_min",
            lo=0,
            hi=8,
            doc="Worker --spec-draft-n-min for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "worker_draft_p_min", "float", 0.0,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.draft_p_min",
            lo=0.0,
            hi=0.5,
            doc="Worker --draft-p-min for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "worker_draft_p_split", "float", 0.0,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.draft_p_split",
            lo=0.0,
            hi=1.0,
            doc="Worker --draft-p-split for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "worker_threads_draft", "int", 16,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.threads_draft",
            lo=1,
            hi=96,
            doc="Worker --threads-draft for launch-time spec decode.",
        ),
        RoleRestartKnob(
            "worker_ngram_mod_n_min", "int", 48,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.ngram_mod_n_min",
            lo=0,
            hi=1024,
            doc="Worker --spec-ngram-mod-n-min for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "worker_ngram_mod_n_max", "int", 64,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.ngram_mod_n_max",
            lo=0,
            hi=1024,
            doc="Worker --spec-ngram-mod-n-max for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "worker_ngram_mod_n_match", "int", 24,
            role="worker_general",
            registry_path="server_mode.worker.acceleration.ngram_mod_n_match",
            lo=1,
            hi=1024,
            doc="Worker --spec-ngram-mod-n-match for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "worker_kv_profile", "enum", "q8_0_q8_0",
            role="worker_general",
            registry_path="server_mode.worker.kv_quant",
            allowed_values=tuple(_KV_PROFILE_VALUES),
            doc="Worker launch KV profile. Mixed K/V profiles stay gated outside AP-3.",
        ),
        # 2026-09-27 ARCHITECT SWAP (operator-decided): the architect_* launch knobs
        # carry the MI210 27B's spec-decode geometry and q8_0/q8_0 KV profile, so
        # they follow that PROCESS to architect_critic (:8083). Left on
        # architect_general they would write the 27B's defaults into Flash-Next's
        # server_mode row. Knob NAMES are unchanged (strategy rows reference them).
        RoleRestartKnob(
            "architect_spec_type", "enum", "draft-mtp",
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.spec_type",
            allowed_values=("draft-mtp",),
            doc=(
                "Architect spec launch type. 2026-08-01: 'ngram-mod,draft-mtp' REMOVED from "
                "the allowed set. The composed recipe was falsified on 2026-07-31 — 35B "
                "long-context coding 0/12, gemma produced ZERO drafts on 9/12 (byte-identical "
                "arms), 80B ngram-alone acceptance 1.6-4.7% because SSM state cannot fork. "
                "Leaving it here let autopilot write the falsified recipe straight back into "
                "the registry block the cutover had just corrected, so 'do not reintroduce "
                "ngram-mod' was not enforceable by registry edits alone. A single-value enum "
                "is retained rather than deleting the knob so existing strategy rows that "
                "reference it still resolve."
            ),
        ),
        RoleRestartKnob(
            "architect_draft_max", "int", 4,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.draft_max",
            lo=1,
            hi=8,
            doc="Architect --spec-draft-n-max for NEXTN/draft-mtp launches.",
        ),
        RoleRestartKnob(
            "architect_draft_min", "int", 0,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.draft_min",
            lo=0,
            hi=8,
            doc="Architect --spec-draft-n-min for NEXTN/draft-mtp launches.",
        ),
        RoleRestartKnob(
            "architect_draft_p_min", "float", 0.0,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.draft_p_min",
            lo=0.0,
            hi=0.5,
            doc="Architect --draft-p-min for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "architect_draft_p_split", "float", 0.0,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.draft_p_split",
            lo=0.0,
            hi=1.0,
            doc="Architect --draft-p-split for MTP/composed-spec launches.",
        ),
        RoleRestartKnob(
            "architect_ngram_mod_n_min", "int", 48,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.ngram_mod_n_min",
            lo=0,
            hi=1024,
            doc="Architect --spec-ngram-mod-n-min for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "architect_ngram_mod_n_max", "int", 64,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.ngram_mod_n_max",
            lo=0,
            hi=1024,
            doc="Architect --spec-ngram-mod-n-max for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "architect_ngram_mod_n_match", "int", 24,
            role="architect_critic",
            registry_path="server_mode.architect_critic.acceleration.ngram_mod_n_match",
            lo=1,
            hi=1024,
            doc="Architect --spec-ngram-mod-n-match for ngram-mod composed speculation.",
        ),
        RoleRestartKnob(
            "architect_kv_profile", "enum", "q8_0_q8_0",
            role="architect_critic",
            registry_path="server_mode.architect_critic.kv_quant",
            allowed_values=tuple(_KV_PROFILE_VALUES),
            doc="Architect launch KV profile; mixed K/V profiles remain lane-gated.",
        ),
    )
}

_AP3_ROLE_RESTART_BY_APPLY_KEY: dict[str, RoleRestartKnob] = {
    knob.apply_key: knob for knob in AP3_ROLE_RESTART_KNOB_SPECS.values()
}


def load_review_plane_knob_manifest(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the W2c numeric-surface manifest, overlaying built-in knob specs.

    Graceful missing-file behavior: when the manifest does not exist (the common
    case while W2c is still authoring it) the built-in ``REVIEW_PLANE_KNOB_SPECS``
    are returned verbatim. When it exists, per-knob ``lo``/``hi``/``default``/
    ``restart_cost`` overrides are overlaid onto the built-ins. Unknown manifest
    knobs are surfaced under their name so the caller can see drift, but never
    crash the loader. The manifest is READ-ONLY here — W2c owns the file.

    Accepts the W2c list convention (``{"knobs": [{"name": "review_plane.X",
    "low": .., "high": .., "param_type": ..}]}`` — ParamSpec field names, the
    ``review_plane.`` name prefix stripped to the bare knob name), as well as the
    simpler ``{"knobs": {name: {...}}}`` / bare ``{name: {...}}`` mapping forms.
    """
    resolved = path or DEFAULT_REVIEW_PLANE_KNOB_MANIFEST
    merged: dict[str, dict[str, Any]] = {
        name: knob.to_dict() for name, knob in REVIEW_PLANE_KNOB_SPECS.items()
    }
    try:
        if not resolved.exists():
            return merged
        raw = _load_yaml_mapping(resolved)
    except Exception as exc:  # noqa: BLE001 — never let a bad manifest break apply
        log.warning("review_plane knob manifest unreadable (%s); using built-ins", exc)
        return merged

    entries = raw.get("knobs", raw) if isinstance(raw, dict) else raw
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(entries, dict):
        items = [(str(k), v) for k, v in entries.items() if isinstance(v, dict)]
    elif isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name"):
                items.append((str(entry["name"]), entry))

    for raw_name, override in items:
        name = raw_name.split(".", 1)[1] if raw_name.startswith("review_plane.") else raw_name
        base = merged.get(name, {"name": name, "manifest_only": True})
        for spec_field, value in _normalize_manifest_knob_fields(override).items():
            base[spec_field] = value
        merged[name] = base
    return merged


def load_axa3_policy_knob_manifest(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load AXA-3 class-3 GPU placement/teleport policy knob declarations.

    The manifest lives beside the AP-1 review-plane manifest because both are
    guarded control-plane numeric surfaces. This loader reads the dedicated
    ``placement_policy_knobs`` list first, and also tolerates entries under the
    historical ``knobs`` list when their name starts with ``placement_policy.``.
    """
    resolved = path or DEFAULT_AXA3_POLICY_KNOB_MANIFEST
    merged: dict[str, dict[str, Any]] = {
        name: knob.to_dict() for name, knob in AXA3_POLICY_KNOB_SPECS.items()
    }
    try:
        if not resolved.exists():
            return merged
        raw = _load_yaml_mapping(resolved)
    except Exception as exc:  # noqa: BLE001 — never let a bad manifest break apply
        log.warning("AXA-3 policy knob manifest unreadable (%s); using built-ins", exc)
        return merged

    entries: list[Any] = []
    explicit = raw.get("placement_policy_knobs") if isinstance(raw, dict) else None
    if isinstance(explicit, list):
        entries.extend(explicit)
    legacy = raw.get("knobs") if isinstance(raw, dict) else None
    if isinstance(legacy, list):
        entries.extend(
            item
            for item in legacy
            if isinstance(item, dict)
            and str(item.get("name", "")).startswith("placement_policy.")
        )

    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        raw_name = str(entry["name"])
        name = (
            raw_name.split(".", 1)[1]
            if raw_name.startswith("placement_policy.")
            else raw_name
        )
        base = merged.get(name, {"name": name, "manifest_only": True})
        for spec_field, value in _normalize_manifest_knob_fields(entry).items():
            base[spec_field] = value
        merged[name] = base
    return merged


def load_ap3_role_restart_knob_manifest(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load AP-3 role-restart launch knob declarations."""
    resolved = path or DEFAULT_AP3_ROLE_RESTART_KNOB_MANIFEST
    merged: dict[str, dict[str, Any]] = {
        name: knob.to_dict() for name, knob in AP3_ROLE_RESTART_KNOB_SPECS.items()
    }
    try:
        if not resolved.exists():
            return merged
        raw = _load_yaml_mapping(resolved)
    except Exception as exc:  # noqa: BLE001 — never let a bad manifest break apply
        log.warning("AP-3 role-restart knob manifest unreadable (%s); using built-ins", exc)
        return merged

    entries: list[Any] = []
    explicit = raw.get("role_restart_knobs") if isinstance(raw, dict) else None
    if isinstance(explicit, list):
        entries.extend(explicit)
    legacy = raw.get("knobs") if isinstance(raw, dict) else None
    if isinstance(legacy, list):
        entries.extend(
            item
            for item in legacy
            if isinstance(item, dict)
            and str(item.get("name", "")).startswith("role_restart.")
        )

    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        raw_name = str(entry["name"])
        name = (
            raw_name.split(".", 1)[1]
            if raw_name.startswith("role_restart.")
            else raw_name
        )
        base = merged.get(name, {"name": name, "manifest_only": True})
        for spec_field, value in _normalize_manifest_knob_fields(entry).items():
            base[spec_field] = value
        if "allowed_values" in entry and isinstance(entry["allowed_values"], list):
            base["allowed_values"] = [str(value) for value in entry["allowed_values"]]
        if "role" in entry:
            base["role"] = str(entry["role"])
        if "registry_path" in entry:
            base["registry_path"] = str(entry["registry_path"])
        merged[name] = base
    return merged


# W2c ParamSpec field aliases -> ReviewPlaneKnob field names.
_MANIFEST_FIELD_ALIASES: dict[str, str] = {
    "low": "lo",
    "high": "hi",
    "lo": "lo",
    "hi": "hi",
    "param_type": "kind",
    "dtype": "kind",
    "kind": "kind",
    "default": "default",
    "restart_cost": "restart_cost",
    "registry_path": "registry_path",
    "role": "role",
    "doc": "doc",
    "semantics": "doc",
}


def _normalize_manifest_knob_fields(override: dict[str, Any]) -> dict[str, Any]:
    """Map a manifest knob entry's fields onto ReviewPlaneKnob's field names."""
    out: dict[str, Any] = {}
    for key, value in override.items():
        target = _MANIFEST_FIELD_ALIASES.get(key)
        # Prefer param_type over dtype; don't let a later doc-source clobber doc.
        if target is None or (target in out and key in {"dtype", "semantics"}):
            continue
        out[target] = value
    return out


def normalize_review_plane_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate bare review-plane knob names to their dotted apply keys.

    Additive + regression-safe: if ``params`` contains no bare review-plane knob
    name the input dict is returned unchanged (same object), so every existing
    ``apply_params`` caller is unaffected.
    """
    if not any(key in REVIEW_PLANE_KNOB_SPECS for key in params):
        return params
    out: dict[str, Any] = {}
    for key, value in params.items():
        spec = REVIEW_PLANE_KNOB_SPECS.get(key)
        out[spec.apply_key if spec is not None else key] = value
    return out


def normalize_axa3_policy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate bare AXA-3 policy knob names to dotted apply keys."""
    if not any(key in AXA3_POLICY_KNOB_SPECS for key in params):
        return params
    out: dict[str, Any] = {}
    for key, value in params.items():
        spec = AXA3_POLICY_KNOB_SPECS.get(key)
        out[spec.apply_key if spec is not None else key] = value
    return out


def normalize_ap3_role_restart_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate bare AP-3 role-restart knob names to dotted apply keys."""
    if not any(key in AP3_ROLE_RESTART_KNOB_SPECS for key in params):
        return params
    out: dict[str, Any] = {}
    for key, value in params.items():
        spec = AP3_ROLE_RESTART_KNOB_SPECS.get(key)
        out[spec.apply_key if spec is not None else key] = value
    return out


def validate_review_plane_params(params: dict[str, Any]) -> str | None:
    """Bounds-check any review-plane knobs in ``params`` (bare or dotted).

    Returns a ``"; "``-joined error string, or ``None`` when every review-plane
    knob is in range (or none are present). Non-review keys are ignored so the
    validator is a no-op for legacy callers.
    """
    errors: list[str] = []
    for key, value in params.items():
        spec = REVIEW_PLANE_KNOB_SPECS.get(key) or _REVIEW_PLANE_BY_APPLY_KEY.get(key)
        if spec is None:
            continue
        err = spec.validate(value)
        if err:
            errors.append(err)
    return "; ".join(errors) if errors else None


def validate_axa3_policy_params(params: dict[str, Any]) -> str | None:
    """Bounds-check AXA-3 GPU placement/teleport policy knobs."""
    errors: list[str] = []
    for key, value in params.items():
        spec = AXA3_POLICY_KNOB_SPECS.get(key) or _AXA3_POLICY_BY_APPLY_KEY.get(key)
        if spec is None:
            continue
        err = spec.validate(value)
        if err:
            errors.append(err)
            continue
        if spec.name == "teleport_enabled" and spec.coerce(value) is True:
            # Keep the registered gate default-off. NumericSwarm never sweeps this
            # bool, and explicit true requires a separate operator/debug env.
            import os

            gate = os.environ.get("AUTOPILOT_AXA3_TELEPORT_ENABLE", "").strip().lower()
            if gate not in {"1", "true", "yes", "on"}:
                errors.append(
                    "teleport_enabled: true requires AUTOPILOT_AXA3_TELEPORT_ENABLE=1"
                )
    return "; ".join(errors) if errors else None


def validate_ap3_role_restart_params(params: dict[str, Any]) -> str | None:
    """Bounds-check AP-3 role-restart launch knobs."""
    errors: list[str] = []
    for key, value in params.items():
        spec = AP3_ROLE_RESTART_KNOB_SPECS.get(key) or _AP3_ROLE_RESTART_BY_APPLY_KEY.get(key)
        if spec is None:
            continue
        err = spec.validate(value)
        if err:
            errors.append(err)
    return "; ".join(errors) if errors else None


# Tier 2 (2026-05-20): KV compression knobs applied at runtime via
# kv_compress.compress_slot() — NOT env-restart, NOT POST /config.
# Keys here are the NumericSwarm param names (e.g. "kv.keep_ratio"); values
# are the kwarg names compress_slot() expects.
KV_COMPACT_PARAMS = {
    "keep_ratio": "keep_ratio",
    "keep_first": "keep_first",
    "n_future": "n_future",
}

# 2026-07-16 resume-precondition (epyc-root handoffs/active/
# autopilot-continuous-optimization.md): kv_compaction trials 1422/1427
# hard-failed on HTTP 500 {"message": "Expected Attention compression failed"}
# from ingest_long_context. On the frozen v8 server the ONLY source of that
# message is tools/server/server-context.cpp SERVER_TASK_TYPE_SLOT_COMPACT when
# llama_kv_compress_evict() returns -1, and evict returns -1 ONLY when
# llama_kv_compress_score() returns empty — i.e. a scoring PRECONDITION failed
# (empty slot: seq pos_max < 0; no attention KV cache resolvable for the model;
# or no attention layers), never a mid-compaction fault. That is "nothing to
# compact on this slot" — a capability/state condition of the target role, not
# a failure of the trial's kv.* params — so it must be classified per-role as a
# skip, not a hard apply error that fails the trial and feeds the thrash loop.
KV_COMPACT_UNCOMPACTABLE_MARKER = "Expected Attention compression failed"


@dataclass
class ApplyResult:
    """Typed result for one parameter-application surface."""

    status: str = "ok"
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == "error" or bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.payload)
        result["status"] = "error" if self.failed else self.status
        if self.errors:
            result["errors"] = list(self.errors)
        return result

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ApplyResult":
        status = str(payload.get("status", "ok"))
        errors = [str(error) for error in payload.get("errors", [])]
        if status == "error" and payload.get("error"):
            errors.append(str(payload["error"]))
        elif status == "error" and not errors:
            errors.append(str(payload.get("error") or "unknown error"))
        for role, role_result in payload.get("per_role", {}).items():
            if isinstance(role_result, dict) and (
                role_result.get("error") or role_result.get("success") is False
            ):
                errors.append(f"{role}: {role_result.get('error') or 'not applied'}")
        return cls(status=status, payload=payload, errors=errors)


RoleSmokeCheck = Callable[
    [str, list[str]], ApplyResult | dict[str, Any] | bool | None
]


class HotSwapApplicator:
    """Apply feature-flag changes through the live config endpoint."""

    def __init__(self, url: str = ORCHESTRATOR_URL) -> None:
        self.url = url

    def apply(self, params: dict[str, Any]) -> ApplyResult:
        try:
            resp = httpx.post(
                f"{self.url}/config",
                json=params,
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            log.info("Hot-swap applied: %s → %s", params, payload.get("status"))
            return ApplyResult.from_payload(payload)
        except Exception as exc:
            log.error("Hot-swap failed: %s", exc)
            return ApplyResult(
                status="error",
                payload={"status": "error", "error": str(exc)},
                errors=[str(exc)],
            )


PARAM_PREIMAGE_SCHEMA = "autopilot-param-preimage.v1"


def capture_reversible_param_preimage(
    params: dict[str, Any],
    *,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Capture the exact live values needed to undo a staged numeric trial.

    Multi-tier validation spans several evaluations.  A rejected candidate must
    therefore restore the runtime configuration that preceded its T1 screen,
    not merely restore AutoPilot's file checkpoint.  Only surfaces with an
    exact, independently attestable inverse are admitted here.
    """
    params = normalize_review_plane_params(params)
    params = normalize_axa3_policy_params(params)
    params = normalize_ap3_role_restart_params(params)
    classified = classify_params(params)
    unsupported = {
        name: sorted(values)
        for name, values in classified.items()
        if name in {"role_restart", "kv_compact", "unknown"} and values
    }
    if unsupported:
        return {
            "schema_version": PARAM_PREIMAGE_SCHEMA,
            "status": "unsupported",
            "reversible": False,
            "unsupported": unsupported,
        }

    try:
        response = httpx.get(f"{url}/config/attest", timeout=10)
        response.raise_for_status()
        attestation = response.json()
        pid = int(attestation["pid"])
        flags = attestation.get("flags") or {}
        if not isinstance(flags, dict):
            raise ValueError("/config/attest flags is not an object")
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        live_env = dict(
            pair.split("=", 1)
            for pair in raw.decode("utf-8", errors="strict").split("\0")
            if "=" in pair
        )
    except Exception as exc:
        return {
            "schema_version": PARAM_PREIMAGE_SCHEMA,
            "status": "error",
            "reversible": False,
            "error": f"live config preimage unavailable: {type(exc).__name__}: {exc}",
        }

    hot_swap: dict[str, bool] = {}
    for name in classified["hot_swap"]:
        if name not in flags:
            return {
                "schema_version": PARAM_PREIMAGE_SCHEMA,
                "status": "error",
                "reversible": False,
                "error": f"live flag preimage missing {name!r}",
            }
        hot_swap[name] = bool(flags[name])

    env: dict[str, dict[str, Any]] = {}
    for dotted_name in classified["env_restart"]:
        section, param = dotted_name.split(".", 1)
        env_name = ENV_PARAMS[section][param]
        env[env_name] = {
            "present": env_name in live_env,
            "value": live_env.get(env_name),
        }

    return {
        "schema_version": PARAM_PREIMAGE_SCHEMA,
        "status": "ok",
        "reversible": True,
        "source_pid": pid,
        "params": sorted(params),
        "hot_swap": hot_swap,
        "env": env,
    }


def restore_reversible_param_preimage(
    preimage: dict[str, Any],
    *,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Restore and attest a preimage produced by the capture helper."""
    if (
        not isinstance(preimage, dict)
        or preimage.get("schema_version") != PARAM_PREIMAGE_SCHEMA
        or preimage.get("status") != "ok"
        or preimage.get("reversible") is not True
    ):
        return {"status": "error", "error": "invalid or non-reversible parameter preimage"}

    hot_swap = preimage.get("hot_swap") or {}
    env_records = preimage.get("env") or {}
    if not isinstance(hot_swap, dict) or not isinstance(env_records, dict):
        return {"status": "error", "error": "malformed parameter preimage"}

    hot_result: dict[str, Any] | None = None
    if hot_swap:
        hot_result = apply_hot_swap(hot_swap, url=url)
        if ApplyResult.from_payload(hot_result).failed:
            return {
                "status": "error",
                "error": "hot-swap preimage restore failed",
                "hot_swap_result": hot_result,
            }

    env_overrides: dict[str, str] = {}
    env_unset: list[str] = []
    for env_name, record in env_records.items():
        if not isinstance(record, dict) or not isinstance(record.get("present"), bool):
            return {"status": "error", "error": f"malformed env preimage for {env_name}"}
        if record["present"]:
            value = record.get("value")
            if not isinstance(value, str):
                return {"status": "error", "error": f"missing env value for {env_name}"}
            env_overrides[str(env_name)] = value
        else:
            env_unset.append(str(env_name))

    env_result: dict[str, Any] | None = None
    if env_overrides or env_unset:
        env_result = _reload_api_via_stack(
            env_overrides=env_overrides,
            env_unset=env_unset,
            url=url,
        )
        if env_result.get("status") != "ok":
            return {
                "status": "error",
                "error": "environment preimage restore failed",
                "env_result": env_result,
                "hot_swap_result": hot_result,
            }

    # Verify both surfaces against the replacement API worker rather than
    # trusting the reload command's exit code.
    check = capture_reversible_param_preimage(
        {
            **{name: value for name, value in hot_swap.items()},
            **{
                dotted_name: 0
                for dotted_name in preimage.get("params", [])
                if isinstance(dotted_name, str)
                and "." in dotted_name
                and dotted_name.split(".", 1)[0] in ENV_PARAMS
            },
        },
        url=url,
    )
    if check.get("status") != "ok":
        return {
            "status": "error",
            "error": "restored configuration could not be attested",
            "attestation": check,
        }
    if check.get("hot_swap") != hot_swap or check.get("env") != env_records:
        return {
            "status": "error",
            "error": "restored configuration does not match preimage",
            "expected": {"hot_swap": hot_swap, "env": env_records},
            "actual": {"hot_swap": check.get("hot_swap"), "env": check.get("env")},
        }
    return {
        "status": "ok",
        "hot_swap_result": hot_result,
        "env_result": env_result,
        "attestation": check,
    }


def _live_api_env(keys: "list[str] | tuple[str, ...]") -> dict[str, str | None] | None:
    """Exec-time env of the live API parent process, for the given keys.

    Read from /proc/<pid>/environ — a process env cannot change after exec, so
    this is exactly what a restart would (or would not) change. Returns None
    when the live process can't be found/read; callers must treat None as
    "unknown" and restart (fail-safe toward applying, never toward skipping).
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", r"uvicorn src\.api:app"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p for p in result.stdout.split() if p.strip().isdigit()]
        if not pids:
            return None
        raw = Path(f"/proc/{pids[0]}/environ").read_bytes()
        env = dict(
            pair.split("=", 1)
            for pair in raw.decode("utf-8", errors="ignore").split("\0")
            if "=" in pair
        )
        return {k: env.get(k) for k in keys}
    except Exception:
        return None


class EnvRestartApplicator:
    """Apply env-backed params by staging env vars and optionally reloading API."""

    def __init__(self, url: str = ORCHESTRATOR_URL, restart: bool = True) -> None:
        self.url = url
        self.restart = restart

    def env_changes_for(self, params: dict[str, Any]) -> dict[str, str]:
        env_changes: dict[str, str] = {}
        for key, value in params.items():
            section, param = key.split(".", 1)
            if section in ENV_PARAMS and param in ENV_PARAMS[section]:
                env_var = ENV_PARAMS[section][param]
                env_changes[env_var] = str(value)
        return env_changes

    def apply(self, params: dict[str, Any]) -> ApplyResult:
        validation_error = _validate_chat_review_threshold_order(params)
        if validation_error:
            return ApplyResult(
                status="error",
                payload={"status": "error", "error": validation_error},
                errors=[validation_error],
            )

        env_changes = self.env_changes_for(params)

        if not env_changes:
            return ApplyResult(status="no_changes", payload={"status": "no_changes"})

        log.info("Env params to apply: %s", env_changes)

        if self.restart:
            # No-op guard: env-carrying trials restart the API twice (apply +
            # boundary revert), and reverting to a baseline the process already
            # runs — or re-applying an env identical to what's live — buys
            # nothing while tearing down every SSE/in-flight request (~212
            # restarts in one daemon log). Skip only on a POSITIVE match of
            # every target key against the live process env; any uncertainty
            # (process not found, unreadable environ) restarts as before.
            # `api_restart` rides in the payload → journal, as an eval
            # covariate (fresh-vs-warm API is a timing regime signal).
            live = _live_api_env(list(env_changes.keys()))
            if live is not None and all(
                live.get(k) == v for k, v in env_changes.items()
            ):
                log.info(
                    "Env already live on API (%s); skipping no-op restart",
                    env_changes,
                )
                return ApplyResult(
                    status="skipped_noop",
                    payload={
                        "status": "skipped_noop",
                        "api_restart": "skipped_noop",
                        "env_changes": env_changes,
                    },
                )
            result = ApplyResult.from_payload(
                restart_api(env_overrides=env_changes, url=self.url)
            )
            result.payload.setdefault("api_restart", "performed")
            return result
        return ApplyResult(
            status="staged",
            payload={"status": "staged", "env_changes": env_changes},
        )


def _current_chat_review_thresholds() -> tuple[float, float]:
    try:
        from src.config import get_config

        cfg = get_config().chat
        return float(cfg.review_low_q_threshold), float(cfg.review_skip_q_threshold)
    except Exception:
        return 0.6, 0.6


def _validate_chat_review_threshold_order(params: dict[str, Any]) -> str | None:
    low_key = "chat.review_low_q_threshold"
    skip_key = "chat.review_skip_q_threshold"
    if low_key not in params and skip_key not in params:
        return None
    try:
        low, skip = _current_chat_review_thresholds()
        if low_key in params:
            low = float(params[low_key])
        if skip_key in params:
            skip = float(params[skip_key])
    except (TypeError, ValueError) as exc:
        return f"chat review thresholds must be numeric: {exc}"
    if low > skip:
        return (
            "chat review thresholds require "
            "chat.review_low_q_threshold <= chat.review_skip_q_threshold; "
            f"got {low:.6g} > {skip:.6g}"
        )
    return None


class KvCompactionApplicator:
    """Apply KV-compaction params to llama-server slots."""

    def __init__(self, roles: list[str] | None = None) -> None:
        self.roles = roles

    def apply(self, params: dict[str, Any]) -> ApplyResult:
        try:
            from scripts.autopilot.kv_compress import (
                compress_slot,
                production_ports,
            )
        except ImportError as exc:
            log.error("kv_compress module not available: %s", exc)
            return ApplyResult(
                status="error",
                payload={"status": "error", "error": str(exc)},
                errors=[str(exc)],
            )

        kwargs: dict[str, Any] = {}
        for key, value in params.items():
            if not key.startswith("kv."):
                continue
            short = key.split(".", 1)[1]
            if short in KV_COMPACT_PARAMS:
                kwargs[KV_COMPACT_PARAMS[short]] = value

        if not kwargs:
            return ApplyResult(status="no_changes", payload={"status": "no_changes"})

        default_ports = production_ports()
        role_ports = production_ports(include_aliases=True)
        target_roles = self.roles or list(default_ports.keys())
        payload: dict[str, Any] = {"per_role": {}}
        errors: list[str] = []
        for role in target_roles:
            port = role_ports.get(role)
            if port is None:
                payload["per_role"][role] = {"status": "skipped", "reason": "no port mapping"}
                continue
            res = compress_slot(port=port, slot_id=0, **kwargs)
            if not res.success and KV_COMPACT_UNCOMPACTABLE_MARKER in (res.error or ""):
                # Slot not compactable (empty KV cache / unsupported memory
                # layout) — see KV_COMPACT_UNCOMPACTABLE_MARKER. Skip the role
                # instead of hard-failing the whole trial apply.
                log.warning(
                    "KV compaction skipped for role %s (port %d): slot not "
                    "compactable (empty KV cache or unsupported memory layout): %s",
                    role, port, res.error,
                )
                payload["per_role"][role] = {
                    "status": "skipped",
                    "reason": "slot not compactable (empty or unsupported)",
                    "detail": res.error,
                }
                continue
            role_result = {
                "success": res.success,
                "n_evicted": res.n_evicted,
                "elapsed_ms": res.elapsed_ms,
                "error": res.error,
            }
            payload["per_role"][role] = role_result
            if role_result.get("error") or role_result.get("success") is False:
                errors.append(f"{role}: {role_result.get('error') or 'not applied'}")
        applied_any = any(
            isinstance(role_result, dict) and role_result.get("success")
            for role_result in payload["per_role"].values()
        )
        if errors:
            status = "error"
        elif not applied_any:
            # Every target role was skipped (uncompactable slot / no port
            # mapping): no live config changed, so the trial must journal as a
            # benign no-change skip — NOT evaluate a baseline masquerading as
            # the trial params (actions._numeric_apply_no_changes handles this).
            status = "no_changes"
        else:
            status = "ok"
        return ApplyResult(status=status, payload=payload, errors=errors)


def classify_params(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Classify parameters by application method.

    Returns {
        "hot_swap": {...},
        "env_restart": {...},
        "role_restart": {...},
        "kv_compact": {...},
        "unknown": {...},
    }.
    """
    result: dict[str, dict[str, Any]] = {
        "hot_swap": {},
        "env_restart": {},
        "role_restart": {},
        "kv_compact": {},
        "unknown": {},
    }

    for key, value in params.items():
        if key in HOT_SWAP_FEATURES:
            result["hot_swap"][key] = value
        elif "." in key:
            section, param = key.split(".", 1)
            if section == "kv" and param in KV_COMPACT_PARAMS:
                result["kv_compact"][key] = value
            elif key in _AP3_ROLE_RESTART_BY_APPLY_KEY:
                result["role_restart"][key] = value
            elif section in ENV_PARAMS and param in ENV_PARAMS[section]:
                result["env_restart"][key] = value
            else:
                result["unknown"][key] = value
        else:
            result["unknown"][key] = value

    return result


def apply_hot_swap(
    params: dict[str, Any], url: str = ORCHESTRATOR_URL
) -> dict[str, Any]:
    """Apply feature flag changes via POST /config."""
    return HotSwapApplicator(url=url).apply(params).to_dict()


def apply_env_params(
    params: dict[str, Any],
    restart: bool = True,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Apply environment-variable-based params and optionally restart API.

    params: dict like {"memrl_retrieval.q_weight": 0.75}
    """
    return EnvRestartApplicator(url=url, restart=restart).apply(params).to_dict()


def _ap3_role_restart_gate_enabled() -> bool:
    import os

    return os.environ.get(_AP3_ROLE_RESTART_ENABLE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ap3_role_restart_groups(params: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    groups: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for key, value in params.items():
        spec = AP3_ROLE_RESTART_KNOB_SPECS.get(key) or _AP3_ROLE_RESTART_BY_APPLY_KEY.get(key)
        if spec is None:
            errors.append(f"{key}: unknown AP-3 role-restart knob")
            continue
        err = spec.validate(value)
        if err:
            errors.append(err)
            continue
        try:
            overrides = spec.registry_overrides(value)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        role_group = groups.setdefault(spec.role, {})
        role_group.update(overrides)
    return groups, errors


def apply_role_restart_params(
    params: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
    journal: Any | None = None,
    smoke_check: RoleSmokeCheck | None = None,
    require_smoke_check: bool = True,
    pause_dispatch: bool = True,
    trial_id: int | None = None,
    actor: str = "config_applicator.apply_role_restart_params",
) -> dict[str, Any]:
    """Apply AP-3 launch-param changes via guarded sequential role reloads."""
    if not params:
        return {"status": "no_changes", "per_role": {}}

    params = normalize_ap3_role_restart_params(params)
    groups, errors = _ap3_role_restart_groups(params)
    if errors:
        return {"status": "error", "errors": errors, "per_role": {}}
    if not groups:
        return {"status": "no_changes", "per_role": {}}
    if not _ap3_role_restart_gate_enabled():
        return {
            "status": "error",
            "errors": [
                f"AP-3 role restart requires {_AP3_ROLE_RESTART_ENABLE_ENV}=1"
            ],
            "per_role": {},
            "registry_override_keys": sorted(
                key for overrides in groups.values() for key in overrides
            ),
        }

    per_role: dict[str, Any] = {}
    result_errors: list[str] = []
    for role in sorted(groups):
        overrides = groups[role]
        prewarm = _ap3_prewarm_role_targets(
            role,
            registry_path=registry_path,
        )
        if prewarm.get("status") != "ok":
            role_result = {
                "status": "error",
                "role": role,
                "error": "AP-3 NUMA prewarm failed before registry mutation",
                "ap3_prewarm": prewarm,
            }
            _attach_restart_boundary_event(
                role_result,
                journal=journal,
                role=role,
                affected_roles=prewarm.get("affected_roles") or [role],
                env_keys=[],
                registry_override_keys=sorted(overrides),
                trial_id=trial_id,
                reason="AP-3 NUMA prewarm failed before role restart",
                evidence={"ap3_prewarm": prewarm},
                actor=actor,
                command="ap3-numa-prewarm",
            )
            per_role[role] = role_result
            result_errors.append(f"{role}: {role_result['error']}")
            break

        strict_smoke = _ap3_strict_smoke_with_readiness(
            smoke_check=smoke_check,
            prewarm=prewarm,
        )
        role_result = restart_role(
            role=role,
            registry_overrides=overrides,
            registry_path=registry_path,
            journal=journal,
            smoke_check=strict_smoke,
            require_smoke_check=require_smoke_check,
            pause_dispatch=pause_dispatch,
            trial_id=trial_id,
            boundary_reason="AP-3 launch role-restart trial",
            boundary_evidence={"ap3_prewarm": prewarm},
            actor=actor,
        )
        role_result["ap3_prewarm"] = prewarm
        per_role[role] = role_result
        if role_result.get("status") == "error" or role_result.get("error"):
            result_errors.append(f"{role}: {role_result.get('error') or 'role restart failed'}")
            break

    result: dict[str, Any] = {
        "status": "error" if result_errors else "ok",
        "per_role": per_role,
        "registry_override_keys": sorted(
            key for overrides in groups.values() for key in overrides
        ),
    }
    if result_errors:
        result["errors"] = result_errors
    return result


def _ap3_prewarm_role_targets(
    role: str,
    *,
    registry_path: Path,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, Any]:
    """Resolve and warm the exact AP-3 reload files before registry mutation.

    This deliberately lives in the AP-3 wrapper rather than ``restart_role``:
    a manual generic restart must retain its existing behavior.  Stack priors
    provide the live role/port scope; launch argv provides model and draft
    paths.  Resolution is strict because an unreadable target must never turn
    into a cold, unmeasured role restart.
    """
    started = datetime.now(timezone.utc).isoformat()
    evidence: dict[str, Any] = {
        "policy": "ap3-numa-prewarm-v1",
        "started_at": started,
        "evidence_ref": f"ap3-numa-prewarm:{role}:{started}",
        "role": role,
        "targets": [],
    }
    try:
        # RegistryLoader, NOT scripts.lib.registry.load_registry. `build_server_command`
        # consumes role_config by nested ATTRIBUTE access (`role_config.model.full_path`,
        # `.acceleration`, `.name` — orchestrator_stack.py:1353-1355), which is the
        # RoleConfig dataclass RegistryLoader.get_role() returns. ModelRegistry, what
        # load_registry() returns, has no `get_role` at all and its `get_role_config()`
        # returns a plain dict — so the obvious swap trades an AttributeError on
        # `registry.get_role` for an AttributeError on `dict.model`, one frame deeper.
        from src.registry.registry_loader import RegistryLoader
        from scripts.server.orchestrator_stack import build_server_command
        from scripts.server import stack_prewarm
        from src.registry.stack_priors import live_stack_role_records, stack_prior_launch_entries

        records = live_stack_role_records(stack_priors_path)
        target = records.get(role)
        if target is None:
            raise RuntimeError(f"live stack-priors record missing for role {role!r}")
        affinity = _launch_affinity_keys(target, launch_entries=stack_prior_launch_entries)
        scoped = {
            name: record for name, record in records.items()
            if affinity & _launch_affinity_keys(record, launch_entries=stack_prior_launch_entries)
        }
        if not scoped:
            raise RuntimeError(f"no live launch scope resolved for role {role!r}")
        registry = RegistryLoader(registry_path)
        by_inode: dict[tuple[int, int], dict[str, Any]] = {}
        for scoped_role, record in sorted(scoped.items()):
            for entry in stack_prior_launch_entries(record):
                port = entry.get("port")
                primary = entry.get("primary_role")
                if not isinstance(port, int) or not isinstance(primary, str) or not primary:
                    raise RuntimeError(f"invalid live launch entry for {scoped_role!r}: {entry!r}")
                role_config = registry.get_role(primary)
                cmd = build_server_command(
                    role_config, port,
                    worker_pool_mode=entry.get("mode") == "worker_pool",
                    worker_type=entry.get("worker_type"),
                    vision_mode=entry.get("mode") == "vision",
                    vision_type=entry.get("vision_type"),
                    numa_instance=int(entry.get("numa_instance", 0)),
                )
                for raw_path in stack_prewarm._extract_paths_from_cmd(cmd):
                    path = Path(raw_path).resolve(strict=True)
                    if not path.is_file() or not os.access(path, os.R_OK):
                        raise RuntimeError(f"unreadable model/draft path: {path}")
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino)
                    target_entry = by_inode.setdefault(key, {
                        "path": path, "dev": stat.st_dev, "inode": stat.st_ino,
                        "size_bytes": stat.st_size, "roles": set(), "ports": set(),
                    })
                    target_entry["roles"].add(scoped_role)
                    target_entry["ports"].add(port)
        if not by_inode:
            raise RuntimeError(f"no model/draft paths resolved for role {role!r}")
        evidence["affected_roles"] = sorted(
            {name for target_entry in by_inode.values() for name in target_entry["roles"]}
        )
        evidence["affected_ports"] = sorted(
            {port for target_entry in by_inode.values() for port in target_entry["ports"]}
        )
        for target_entry in sorted(by_inode.values(), key=lambda item: str(item["path"])):
            item_started = datetime.now(timezone.utc).isoformat()
            ok, elapsed_s, message = stack_prewarm.prewarm_file(target_entry["path"])
            item = {
                "path": str(target_entry["path"]), "dev": target_entry["dev"],
                "inode": target_entry["inode"], "size_bytes": target_entry["size_bytes"],
                "roles": sorted(target_entry["roles"]), "ports": sorted(target_entry["ports"]),
                "started_at": item_started, "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": elapsed_s, "result": "ok" if ok else "error", "detail": message,
            }
            evidence["targets"].append(item)
            if not ok:
                evidence.update(status="error", error=f"prewarm failed for {item['path']}: {message}")
                return evidence
    except Exception as exc:
        evidence.update(status="error", error=str(exc))
        return evidence
    evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
    evidence["status"] = "ok"
    return evidence


def _ap3_strict_smoke_with_readiness(
    *,
    smoke_check: RoleSmokeCheck | None,
    prewarm: dict[str, Any],
) -> RoleSmokeCheck:
    """Compose AP-3 smoke with a verified one-token, untimed readiness probe."""
    def check(role: str, affected_roles: list[str]) -> dict[str, Any]:
        base = _run_role_smoke_check(role=role, affected_roles=affected_roles, smoke_check=smoke_check)
        if base.failed:
            result = base.to_dict()
            result["base_smoke"] = base.to_dict()
            return result
        probes: list[dict[str, Any]] = []
        for port in sorted({port for target in prewarm["targets"] for port in target["ports"]}):
            started_at = datetime.now(timezone.utc)
            response: Any | None = None
            probe: dict[str, Any] = {
                "port": port,
                "prompt_sha256": sha256(b"ready").hexdigest(),
                "started_at": started_at.isoformat(),
                "timing_excluded": True,
                "http_status": None,
                "response_sha256": None,
                "token_count": None,
            }
            try:
                response = httpx.post(
                    f"http://127.0.0.1:{port}/completion",
                    json={"prompt": "ready", "n_predict": 1, "temperature": 0}, timeout=30,
                )
                response.raise_for_status()
                raw_body = response.content
                probe["http_status"] = response.status_code
                probe["response_sha256"] = sha256(raw_body).hexdigest()
                payload = response.json()
                token_count = payload.get("tokens_predicted")
                if not isinstance(token_count, int) or isinstance(token_count, bool):
                    raise ValueError("completion response missing integer tokens_predicted")
                probe["token_count"] = token_count
                if token_count < 1:
                    raise ValueError("completion response generated zero tokens")
                probe["status"] = "ok"
            except Exception as exc:
                probe["status"] = "error"
                probe["error_type"] = type(exc).__name__
                probe["http_status"] = probe["http_status"] or getattr(
                    response, "status_code", None
                )
                probes.append(_finish_ap3_readiness_probe(probe, started_at))
                return {
                    "status": "error",
                    "error": f"AP-3 readiness failed on port {port}: {type(exc).__name__}",
                    "base_smoke": base.to_dict(),
                    "readiness": probes,
                }
            probes.append(_finish_ap3_readiness_probe(probe, started_at))
        return {
            "status": "ok",
            "base_smoke": base.to_dict(),
            "readiness": probes,
            "prewarm_evidence_ref": prewarm["evidence_ref"],
        }
    return check


def _finish_ap3_readiness_probe(
    probe: dict[str, Any], started_at: datetime
) -> dict[str, Any]:
    """Record bounded readiness evidence without retaining request/response text."""
    finished_at = datetime.now(timezone.utc)
    probe["finished_at"] = finished_at.isoformat()
    probe["elapsed_s"] = (finished_at - started_at).total_seconds()
    return probe


def resolve_restart_affected_roles(
    role: str,
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> list[str]:
    """Return live roles that share launch ownership with a restart target."""
    role = role.strip()
    if not role:
        return []

    try:
        from src.registry.stack_priors import (
            live_stack_role_records,
            stack_prior_launch_entries,
        )
    except Exception as exc:
        log.warning("Could not load stack-priors helpers: %s", exc)
        return [role]

    records = live_stack_role_records(stack_priors_path)
    target_record = records.get(role)
    if target_record is None:
        return [role]

    target_keys = _launch_affinity_keys(
        target_record,
        launch_entries=stack_prior_launch_entries,
    )
    if not target_keys:
        return [role]

    affected = [
        candidate_role
        for candidate_role, record in records.items()
        if target_keys
        & _launch_affinity_keys(record, launch_entries=stack_prior_launch_entries)
    ]
    return _ordered_roles(role, affected)


def _launch_affinity_keys(
    record: dict[str, Any],
    *,
    launch_entries: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for entry in launch_entries(record):
        port = entry.get("port")
        if isinstance(port, int):
            keys.add(("port", str(port)))
        elif isinstance(port, str) and port.isdigit():
            keys.add(("port", port))

        primary_role = entry.get("primary_role")
        if isinstance(primary_role, str) and primary_role:
            keys.add(("primary_role", primary_role))
    return keys


def _ordered_roles(primary_role: str, roles: list[str]) -> list[str]:
    unique = sorted({role for role in roles if role})
    if primary_role in unique:
        return [primary_role, *(role for role in unique if role != primary_role)]
    return [primary_role, *unique]


def _run_role_smoke_check(
    *,
    role: str,
    affected_roles: list[str],
    smoke_check: RoleSmokeCheck | None,
) -> ApplyResult:
    """Normalize an optional role smoke check into an ApplyResult."""
    payload_base = {"role": role, "affected_roles": list(affected_roles)}
    if smoke_check is None:
        return ApplyResult(
            status="skipped",
            payload={**payload_base, "status": "skipped", "reason": "no_smoke_check"},
        )

    try:
        raw = smoke_check(role, list(affected_roles))
    except Exception as exc:
        return ApplyResult(
            status="error",
            payload={**payload_base, "status": "error", "error": str(exc)},
            errors=[str(exc)],
        )

    if isinstance(raw, ApplyResult):
        return raw
    if raw is None or raw is True:
        return ApplyResult(status="ok", payload={**payload_base, "status": "ok"})
    if raw is False:
        return ApplyResult(
            status="error",
            payload={**payload_base, "status": "error", "error": "smoke check failed"},
            errors=["smoke check failed"],
        )
    if isinstance(raw, dict):
        payload = {**payload_base, **raw}
        if payload.get("success") is False or payload.get("error"):
            payload["status"] = "error"
        return ApplyResult.from_payload(payload)
    return ApplyResult(
        status="error",
        payload={
            **payload_base,
            "status": "error",
            "error": f"unsupported smoke check result: {type(raw).__name__}",
        },
        errors=[f"unsupported smoke check result: {type(raw).__name__}"],
    )


def _resolve_autopilot_state_path(path: Path | None = None) -> Path:
    """Resolve the AutoPilot state path without importing the heavy loop module."""
    if path is not None:
        return path
    import os

    env_override = os.environ.get("AUTOPILOT_STATE")
    return Path(env_override) if env_override else DEFAULT_AUTOPILOT_STATE_PATH


def _pause_autopilot_dispatch(
    *,
    state_path: Path | None = None,
    grace_s: float = 11.0,
) -> dict[str, Any]:
    """Set AutoPilot paused=True so the loop stops dispatching new trials.

    Stamps a pause LEASE (``pause_owner``/``pause_token``) whenever this call is
    the one that transitions paused False->True, so the matching restore can tell
    "still my pause" from "the operator paused during my apply window". Nothing is
    stamped when AutoPilot was already paused — that pause belongs to someone else
    and the restore leaves it alone (``already_paused``) exactly as before.
    """
    import json
    import os

    resolved = _resolve_autopilot_state_path(state_path)
    result: dict[str, Any] = {
        "status": "ok",
        "state_path": str(resolved),
        "paused_pre": None,
        "paused_set": False,
        "grace_s": grace_s,
        "pause_owner": None,
        "pause_token": None,
    }
    try:
        # H4: set paused=True under a BRIEF locked read-modify-write, then RELEASE
        # the lock BEFORE the grace sleep — never hold the state lock across a sleep.
        token: str | None = None
        with state_write_lock(resolved):
            with open(resolved, encoding="utf-8") as handle:
                state = json.load(handle)
            paused_pre = bool(state.get("paused", False))
            if paused_pre:
                # Someone else's pause. Do NOT take the lease off them.
                state["paused"] = True
            else:
                token = claim_pause_lease(state, DISPATCH_PAUSE_OWNER)
            tmp = resolved.with_suffix(resolved.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
            os.replace(tmp, resolved)
        result["paused_pre"] = paused_pre
        result["paused_set"] = True
        if token is not None:
            result["pause_owner"] = DISPATCH_PAUSE_OWNER
            result["pause_token"] = token
        if grace_s > 0:
            time.sleep(grace_s)
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
    return result


def _restore_autopilot_dispatch_pause(pause_result: dict[str, Any]) -> dict[str, Any]:
    """Restore paused=False only while this applicator still holds the pause lease.

    THE INTERLOCK. The apply window (stack reload + health + smoke + rollback) is
    tens of seconds to minutes long; an operator ``autopilot.py pause`` can land
    inside it. Before the lease existed, that operator pause was indistinguishable
    from this applicator's own (both just ``"paused": true``) and this function
    resumed AutoPilot anyway — honouring the pause, then silently undoing it.

    Now the resume is refused unless our token is still on disk, and the refusal is
    REPORTED (``status="superseded"``, plus the observed owner) rather than being a
    quiet no-op: the result rides into the apply payload and the journalled restart
    boundary event, so the collision is legible after the fact.
    """
    import json
    import os

    state_path = Path(str(pause_result.get("state_path") or DEFAULT_AUTOPILOT_STATE_PATH))
    result: dict[str, Any] = {
        "status": "skipped",
        "state_path": str(state_path),
        "restored": False,
    }
    if pause_result.get("status") != "ok" or not pause_result.get("paused_set"):
        result["reason"] = "pause_not_set"
        return result
    if pause_result.get("paused_pre") is not False:
        result["reason"] = "already_paused"
        return result
    token = pause_result.get("pause_token")
    try:
        # H4: restore paused=False under a BRIEF locked read-modify-write.
        with state_write_lock(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("paused") is not True:
                result.update({"status": "skipped", "reason": "pause_already_cleared"})
            elif release_pause_lease(state, token):
                tmp = state_path.with_suffix(state_path.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(state, handle, indent=2)
                    handle.write("\n")
                os.replace(tmp, state_path)
                result.update({"status": "ok", "restored": True})
            else:
                observed_owner = state.get(PAUSE_OWNER_FIELD)
                result.update(
                    {
                        "status": "superseded",
                        "reason": "pause_superseded",
                        "expected_token": token,
                        "observed_owner": observed_owner,
                        "observed_token": state.get("pause_token"),
                    }
                )
                log.warning(
                    "Refusing to resume AutoPilot: the dispatch pause was superseded "
                    "during the config apply (now held by %s). Leaving paused=True; "
                    "an operator resume is required.",
                    observed_owner or "an unknown owner",
                )
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
    return result


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _write_yaml_mapping_atomic(path: Path, data: dict[str, Any]) -> None:
    import os

    import yaml

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _resolve_dotted_parent(
    data: dict[str, Any],
    dotted_path: str,
) -> tuple[dict[str, Any], str]:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("registry override path is empty")

    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(f"registry override parent is not a mapping: {dotted_path}")
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            raise ValueError(f"registry override parent missing: {dotted_path}")
        cursor = next_value
    if not isinstance(cursor, dict):
        raise ValueError(f"registry override parent is not a mapping: {dotted_path}")
    return cursor, parts[-1]


def _apply_registry_overrides(
    *,
    registry_path: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply dotted YAML overrides and return the exact rollback record."""
    import copy

    if not overrides:
        return {
            "status": "skipped",
            "registry_path": str(registry_path),
            "override_keys": [],
            "rollback_record": [],
        }

    data = _load_yaml_mapping(registry_path)
    rollback_record: list[dict[str, Any]] = []
    for dotted_path, value in overrides.items():
        parent, leaf = _resolve_dotted_parent(data, dotted_path)
        rollback_record.append(
            {
                "path": dotted_path,
                "existed": leaf in parent,
                "value": copy.deepcopy(parent.get(leaf)),
            }
        )
        parent[leaf] = value

    _write_yaml_mapping_atomic(registry_path, data)
    return {
        "status": "ok",
        "registry_path": str(registry_path),
        "override_keys": sorted(overrides),
        "rollback_record": rollback_record,
    }


def _restore_registry_overrides(
    *,
    registry_path: Path,
    rollback_record: list[dict[str, Any]],
) -> dict[str, Any]:
    """Restore a registry file from an override rollback record."""
    if not rollback_record:
        return {
            "status": "skipped",
            "registry_path": str(registry_path),
            "restored_keys": [],
        }

    data = _load_yaml_mapping(registry_path)
    restored_keys: list[str] = []
    for record in rollback_record:
        dotted_path = str(record.get("path") or "")
        parent, leaf = _resolve_dotted_parent(data, dotted_path)
        if record.get("existed"):
            parent[leaf] = record.get("value")
        else:
            parent.pop(leaf, None)
        restored_keys.append(dotted_path)

    _write_yaml_mapping_atomic(registry_path, data)
    return {
        "status": "ok",
        "registry_path": str(registry_path),
        "restored_keys": sorted(restored_keys),
    }


def restart_api(
    env_overrides: dict[str, str] | None = None,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Restart the API server (uvicorn reload).

    Env-backed tuning must relaunch the API from a process that already has
    the new environment. A SIGHUP cannot mutate another process environment,
    so env overrides always go through orchestrator_stack.py reload.
    """
    import os
    import signal

    log.info("Restarting API server...")

    if env_overrides:
        return _reload_api_via_stack(env_overrides=env_overrides, url=url)

    # Try to find and signal the uvicorn process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn.*orchestrator"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            pid = int(result.stdout.strip().split("\n")[0])
            os.kill(pid, signal.SIGHUP)
            log.info("Sent SIGHUP to uvicorn pid %d", pid)
            # Wait for restart
            time.sleep(3)
            if health_check(url):
                return {"status": "ok", "method": "sighup", "pid": pid}
    except Exception as e:
        log.warning("SIGHUP restart failed: %s", e)

    return _reload_api_via_stack(env_overrides=None, url=url)


def restart_role(
    role: str,
    env_overrides: dict[str, str] | None = None,
    registry_overrides: dict[str, Any] | None = None,
    url: str = ORCHESTRATOR_URL,
    journal: Any | None = None,
    affected_roles: list[str] | None = None,
    trial_id: int | None = None,
    boundary_reason: str = "intentional role restart",
    boundary_evidence: dict[str, Any] | None = None,
    actor: str = "config_applicator.restart_role",
    smoke_check: RoleSmokeCheck | None = None,
    require_smoke_check: bool = False,
    require_explicit_affected_roles: bool = False,
    pause_dispatch: bool = False,
    autopilot_state_path: Path | None = None,
    dispatch_pause_grace_s: float = 11.0,
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    """Reload one stack role with rollback to the prior environment on failure.

    This is a dormant W3 applicator primitive. Capability promotion remains
    gated elsewhere; callers must still journal restart boundaries before use.
    """
    import os

    dispatch_pause: dict[str, Any] | None = None
    registry_apply: dict[str, Any] | None = None
    registry_rollback_record: list[dict[str, Any]] = []
    try:
        role = role.strip()
        if not role:
            return {"status": "error", "error": "role is required", "method": "stack_reload"}

        if affected_roles is None:
            if require_explicit_affected_roles:
                return {
                    "status": "error",
                    "method": "stack_reload",
                    "role": role,
                    "error": "affected_roles required for strict role restart",
                    "reason": "affected_roles_required",
                }
            resolved_affected_roles = resolve_restart_affected_roles(role)
        else:
            resolved_affected_roles = _ordered_roles(role, list(affected_roles))

        if require_smoke_check and smoke_check is None:
            return {
                "status": "error",
                "method": "stack_reload",
                "role": role,
                "affected_roles": list(resolved_affected_roles),
                "error": "smoke_check required for strict role restart",
                "reason": "smoke_check_required",
            }

        if pause_dispatch:
            dispatch_pause = _pause_autopilot_dispatch(
                state_path=autopilot_state_path,
                grace_s=dispatch_pause_grace_s,
            )
            if dispatch_pause.get("status") != "ok":
                return {
                    "status": "error",
                    "method": "stack_reload",
                    "role": role,
                    "error": "failed to pause autopilot dispatch",
                    "dispatch_pause": dispatch_pause,
                }

        env_overrides = dict(env_overrides or {})
        registry_overrides = dict(registry_overrides or {})
        prior_env = {key: os.environ.get(key) for key in env_overrides}

        if registry_overrides:
            try:
                registry_apply = _apply_registry_overrides(
                    registry_path=registry_path,
                    overrides=registry_overrides,
                )
                registry_rollback_record = list(registry_apply["rollback_record"])
            except Exception as exc:
                return {
                    "status": "error",
                    "method": "stack_reload",
                    "role": role,
                    "error": f"failed to apply registry_overrides: {exc}",
                    "registry_override_keys": sorted(registry_overrides),
                    "registry_path": str(registry_path),
                }

        first = _reload_role_via_stack(
            role=role,
            env_overrides=env_overrides,
            env_unset=[],
        )
        first["role"] = role
        first["affected_roles"] = list(resolved_affected_roles)
        if registry_apply is not None:
            first["registry_overrides"] = {
                "status": registry_apply["status"],
                "registry_path": registry_apply["registry_path"],
                "override_keys": registry_apply["override_keys"],
            }
        if dispatch_pause is not None:
            first["dispatch_pause"] = dispatch_pause
        if first.get("status") == "ok":
            if role == "orchestrator":
                health = health_check(url)
                if health:
                    smoke = _run_role_smoke_check(
                        role=role,
                        affected_roles=resolved_affected_roles,
                        smoke_check=smoke_check,
                    )
                    first["smoke_check"] = smoke.to_dict()
                    if smoke.failed:
                        first.update(smoke.to_dict())
                    else:
                        _attach_restart_boundary_event(
                            first,
                            journal=journal,
                            role=role,
                            affected_roles=resolved_affected_roles,
                            env_keys=sorted(env_overrides),
                            registry_override_keys=sorted(registry_overrides),
                            trial_id=trial_id,
                            reason=boundary_reason,
                            evidence=_restart_boundary_evidence(
                                boundary_evidence, smoke.to_dict()
                            ),
                            actor=actor,
                        )
                        return first
                if first.get("status") == "ok":
                    first.update(
                        {
                            "status": "error",
                            "error": health.failure_reason,
                            "detail": health.failure_detail,
                        }
                    )
            else:
                smoke = _run_role_smoke_check(
                    role=role,
                    affected_roles=resolved_affected_roles,
                    smoke_check=smoke_check,
                )
                first["smoke_check"] = smoke.to_dict()
                if smoke.failed:
                    first.update(smoke.to_dict())
                else:
                    _attach_restart_boundary_event(
                        first,
                        journal=journal,
                        role=role,
                        affected_roles=resolved_affected_roles,
                        env_keys=sorted(env_overrides),
                        registry_override_keys=sorted(registry_overrides),
                        trial_id=trial_id,
                        reason=boundary_reason,
                        evidence=_restart_boundary_evidence(
                            boundary_evidence, smoke.to_dict()
                        ),
                        actor=actor,
                    )
                    return first

        rollback_overrides = {
            key: value for key, value in prior_env.items() if value is not None
        }
        rollback_unset = [key for key, value in prior_env.items() if value is None]
        registry_rollback = None
        if registry_rollback_record:
            try:
                registry_rollback = _restore_registry_overrides(
                    registry_path=registry_path,
                    rollback_record=registry_rollback_record,
                )
            except Exception as exc:
                registry_rollback = {
                    "status": "error",
                    "registry_path": str(registry_path),
                    "error": str(exc),
                }
        rollback = _reload_role_via_stack(
            role=role,
            env_overrides=rollback_overrides,
            env_unset=rollback_unset,
        )
        first["rollback"] = {
            "attempted": True,
            "status": rollback.get("status", "error"),
            "env_keys": sorted(prior_env),
        }
        if registry_rollback is not None:
            first["rollback"]["registry"] = registry_rollback
        _attach_restart_boundary_event(
            first,
            journal=journal,
            role=role,
            affected_roles=resolved_affected_roles,
            env_keys=sorted(env_overrides),
            registry_override_keys=sorted(registry_overrides),
            trial_id=trial_id,
            reason=boundary_reason,
            evidence=_restart_boundary_evidence(
                boundary_evidence, first.get("smoke_check")
            ),
            actor=actor,
        )
        return first
    finally:
        if dispatch_pause is not None:
            dispatch_pause["restore"] = _restore_autopilot_dispatch_pause(dispatch_pause)


def _restart_boundary_evidence(
    evidence: dict[str, Any] | None,
    smoke_check: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Embed AP-3 readiness evidence without changing generic restart events."""
    if evidence is None:
        return None
    result = dict(evidence)
    if smoke_check is not None:
        result["smoke_check"] = smoke_check
    return result


def _attach_restart_boundary_event(
    result: dict[str, Any],
    *,
    journal: Any | None,
    role: str,
    affected_roles: list[str] | None,
    env_keys: list[str],
    registry_override_keys: list[str],
    trial_id: int | None,
    reason: str,
    evidence: dict[str, Any] | None = None,
    command: str | None = None,
    actor: str,
) -> None:
    """Attach an append-only restart-boundary event when a journal is provided."""
    if journal is None:
        return
    append = getattr(journal, "append_role_restart_boundary_event", None)
    if append is None:
        result["restart_boundary_error"] = "journal lacks append_role_restart_boundary_event"
        return
    try:
        payload = {
            "role": role,
            "affected_roles": list(affected_roles or [role]),
            "env_keys": env_keys,
            "registry_override_keys": registry_override_keys,
            "status": str(result.get("status", "")),
            "rollback_status": str((result.get("rollback") or {}).get("status", "")),
            "reason": reason,
            "actor": actor,
            "boundary_trial_id": trial_id,
            "command": command or f"orchestrator_stack.py reload {role}",
        }
        if evidence is not None:
            payload["evidence"] = evidence
        result["restart_boundary_event"] = append(**payload)
    except Exception as exc:
        result["restart_boundary_error"] = str(exc)


def _reload_role_via_stack(
    *,
    role: str,
    env_overrides: dict[str, str] | None,
    env_unset: list[str],
) -> dict[str, Any]:
    """Run orchestrator_stack.py reload for a role with an explicit environment."""
    import os

    stack_script = ORCH_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    if not stack_script.exists():
        return {"status": "error", "error": f"Stack script not found: {stack_script}"}

    env = os.environ.copy()
    for key in env_unset:
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)

    try:
        subprocess.run(
            [_stack_reload_python(), str(stack_script), "reload", role],
            cwd=str(ORCH_ROOT),
            env=env,
            timeout=180,
            check=True,
        )
    except Exception as exc:
        log.error("Stack role reload failed for %s: %s", role, exc)
        return {
            "status": "error",
            "error": str(exc),
            "method": "stack_reload",
            "env_keys": sorted((env_overrides or {}).keys()),
            "env_unset": sorted(env_unset),
        }
    return {
        "status": "ok",
        "method": "stack_reload",
        "env_keys": sorted((env_overrides or {}).keys()),
        "env_unset": sorted(env_unset),
    }


def _reload_api_via_stack(
    env_overrides: dict[str, str] | None,
    url: str,
    env_unset: list[str] | None = None,
) -> dict[str, Any]:
    """Reload the orchestrator through the stack manager."""
    import os

    stack_script = ORCH_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    if not stack_script.exists():
        return {"status": "error", "error": f"Stack script not found: {stack_script}"}

    env = os.environ.copy()
    for key in env_unset or []:
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)

    try:
        subprocess.run(
            [_stack_reload_python(), str(stack_script), "reload", "orchestrator"],
            cwd=str(ORCH_ROOT),
            env=env,
            timeout=90,
            check=True,
        )
        time.sleep(3)
        health = health_check(url)
        if health:
            return {
                "status": "ok",
                "method": "stack_reload",
                "env_keys": sorted((env_overrides or {}).keys()),
                "env_unset": sorted(env_unset or []),
            }
        return {
            "status": "error",
            "method": "stack_reload",
            "error": health.failure_reason,
            "detail": health.failure_detail,
            "env_keys": sorted((env_overrides or {}).keys()),
            "env_unset": sorted(env_unset or []),
        }
    except Exception as e:
        log.error("Stack reload failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "method": "stack_reload",
            "env_unset": sorted(env_unset or []),
        }


def _stack_reload_python() -> str:
    """Return a stable Python executable for orchestrator_stack reloads.

    AutoPilot is often launched as ``.venv/bin/python``. If another session
    recreates the venv while AutoPilot is running, ``sys.executable`` can become
    a stale symlink even though the current process remains alive. Conversely,
    resolving a healthy venv symlink to its base interpreter drops site-packages
    and makes reloads fail on imports such as ``yaml``. Prefer an explicit
    override, then a live repo venv entrypoint, and only then base interpreters.
    """
    import os

    path_candidates = [
        os.environ.get("AUTOPILOT_STACK_RELOAD_PYTHON"),
        ORCH_ROOT / ".venv" / "bin" / "python",
        ORCH_ROOT / ".venv" / "bin" / "python3",
        sys.executable,
    ]
    for candidate in path_candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if _stack_reload_python_usable(path):
            return str(path)

    base_candidates = [
        getattr(sys, "_base_executable", None),
        "/home/node/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12",
        "/home/node/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12",
        "/usr/bin/python3",
    ]
    for candidate in base_candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if _stack_reload_python_usable(resolved):
            return str(resolved)
    return sys.executable


def _stack_reload_python_usable(path: Path) -> bool:
    """Return true when ``path`` can import stack reload dependencies."""
    if not path.exists():
        return False
    try:
        proc = subprocess.Popen(
            [str(path), "-c", "import yaml"],
            cwd=str(ORCH_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(timeout=10)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class HealthCheckResult:
    """Typed health check result — truthy when healthy, carries diagnostics when not."""

    __slots__ = ("ok", "failure_reason", "failure_detail")

    def __init__(self, ok: bool, failure_reason: str = "", failure_detail: str = ""):
        self.ok = ok
        self.failure_reason = failure_reason
        self.failure_detail = failure_detail

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        if self.ok:
            return "HealthCheckResult(ok=True)"
        return f"HealthCheckResult(ok=False, failure_reason={self.failure_reason!r})"


def health_check(url: str = ORCHESTRATOR_URL, retries: int = 5) -> HealthCheckResult:
    """Verify API is healthy after restart.

    Returns a HealthCheckResult that is truthy when healthy. Callers using
    ``if health_check(...)`` or ``if not health_check(...)`` continue to
    work unchanged; callers that want diagnostics can read
    ``.failure_reason`` and ``.failure_detail``.
    """
    last_reason, last_detail = "max_retries_exceeded", ""
    for i in range(retries):
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return HealthCheckResult(ok=True)
            last_reason = "http_status"
            last_detail = f"status={resp.status_code}"
        except httpx.TimeoutException as exc:
            last_reason = "timeout"
            last_detail = str(exc)
        except httpx.ConnectError as exc:
            last_reason = "connection_refused"
            last_detail = str(exc)
        except (httpx.HTTPError, OSError) as exc:
            last_reason = type(exc).__name__
            last_detail = str(exc)
        time.sleep(1)
    return HealthCheckResult(ok=False, failure_reason=last_reason, failure_detail=last_detail)


def apply_kv_compact(
    params: dict[str, Any],
    roles: list[str] | None = None,
) -> dict[str, Any]:
    """Apply KV-compression trial params via kv_compress.compress_slot().

    params: NumericSwarm trial values like {"kv.keep_ratio": 0.5, "kv.keep_first": 4, "kv.n_future": 128}.
    roles: subset of roles to compact (default = physical primary roles from
           stack priors). Each role is compacted in turn; only idle slots are
           touched inside the underlying compress_slot endpoint via the server.
    """
    return KvCompactionApplicator(roles=roles).apply(params).to_dict()


def apply_params(
    params: dict[str, Any],
    url: str = ORCHESTRATOR_URL,
    dry_run: bool = False,
    kv_roles: list[str] | None = None,
) -> dict[str, Any]:
    """Apply parameters using the appropriate method.

    Returns summary of what was applied.
    """
    # AP-1/AP-2/AP-3: translate bare control-plane knob names to their dotted apply
    # keys and bounds-check them. Both are no-ops (identical behavior) when no
    # matching knob is present, so every legacy caller is unaffected.
    params = normalize_review_plane_params(params)
    params = normalize_axa3_policy_params(params)
    params = normalize_ap3_role_restart_params(params)
    review_plane_error = validate_review_plane_params(params)
    axa3_policy_error = validate_axa3_policy_params(params)
    ap3_role_restart_error = validate_ap3_role_restart_params(params)

    classified = classify_params(params)
    results: dict[str, Any] = {"classified": classified, "status": "ok"}
    errors: list[str] = []

    if review_plane_error or axa3_policy_error or ap3_role_restart_error:
        # Reject out-of-bounds knobs before touching the live surface: an invalid
        # control-plane value must never trigger an API reload.
        error_payload = []
        if review_plane_error:
            error_payload.append(f"review_plane: {review_plane_error}")
        if axa3_policy_error:
            error_payload.append(f"placement_policy: {axa3_policy_error}")
        if ap3_role_restart_error:
            error_payload.append(f"role_restart: {ap3_role_restart_error}")
        results["status"] = "error"
        results["errors"] = error_payload
        return results

    if dry_run:
        results["dry_run"] = True
        return results

    # Hot-swap first (instant)
    if classified["hot_swap"]:
        hot_swap_result = ApplyResult.from_payload(
            apply_hot_swap(classified["hot_swap"], url=url)
        )
        results["hot_swap_result"] = hot_swap_result.to_dict()
        errors.extend(f"hot_swap: {error}" for error in hot_swap_result.errors)

    # Env params (may require restart)
    if classified["env_restart"]:
        env_result = ApplyResult.from_payload(
            apply_env_params(classified["env_restart"], url=url)
        )
        results["env_result"] = env_result.to_dict()
        errors.extend(f"env_restart: {error}" for error in env_result.errors)

    # AP-3 role restart (expensive): registry overrides + sequential stack reloads.
    if classified["role_restart"]:
        role_restart_result = ApplyResult.from_payload(
            apply_role_restart_params(classified["role_restart"])
        )
        results["role_restart_result"] = role_restart_result.to_dict()
        errors.extend(
            f"role_restart: {error}" for error in role_restart_result.errors
        )

    # KV compaction (runtime POST to llama-server /slots/{id}?action=compact)
    if classified["kv_compact"]:
        kv_result = ApplyResult.from_payload(
            apply_kv_compact(classified["kv_compact"], roles=kv_roles)
        )
        results["kv_compact_result"] = kv_result.to_dict()
        errors.extend(f"kv_compact:{error}" for error in kv_result.errors)

    if classified["unknown"]:
        log.warning("Unknown params (not applied): %s", list(classified["unknown"].keys()))
        results["unknown_params"] = list(classified["unknown"].keys())
        errors.append(f"unknown_params: {', '.join(results['unknown_params'])}")

    if errors:
        results["status"] = "error"
        results["errors"] = errors

    return results
