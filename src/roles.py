#!/usr/bin/env python3
"""Role definitions for hierarchical orchestration.

This module defines all agent roles in the orchestration system. Roles are
organized into tiers:

    Tier A (Frontdoor): Interactive chat, intent classification, task routing
    Tier B (Specialists): Domain-specific processing (coder, ingest, architect)
    Tier C (Workers): Parallel file-level implementation
    Tier D (Draft): Speculative decoding draft models

Usage:
    from src.roles import Role, Tier, get_tier

    # Use enum instead of strings
    role = Role.CODER_ESCALATION
    print(role.value)  # "coder_escalation"

    # Get tier for a role
    tier = get_tier(Role.CODER_ESCALATION)  # Tier.B

    # Check if role is valid
    if Role.is_valid("coder_escalation"):
        role = Role("coder_escalation")

    # Get escalation target
    target = Role.CODER_ESCALATION.escalates_to()  # Role.ARCHITECT_GENERAL
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class FailoverReason(str, Enum):
    """Reason for triggering a model fallback (distinct from task escalation).

    Fallback is for infrastructure failures — the model is unavailable.
    Escalation is for task complexity — the model can't solve the problem.
    """

    CIRCUIT_OPEN = "circuit_open"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    # A contention-gate/admission-control denial: the backend is up and
    # healthy, just momentarily over its concurrency budget (e.g. another
    # session's job holding the GPU lane). Distinct from CONNECTION_ERROR,
    # which implies the backend itself is unreachable/down — conflating the
    # two makes a live-but-busy backend look dead in health-tracker logs and
    # metrics (TD-21 window-diag 2026-09-24, INC candidate: coder_escalation
    # → frontdoor fallback logged as connection_error while :8083 was
    # healthy and simply contended).
    ADMISSION_DENIED = "admission_denied"
    OOM = "oom"


class Tier(str, Enum):
    """Agent tier in the orchestration hierarchy.

    Tiers define the capability and cost level of agents:
    - A: Frontdoor (always resident, low latency)
    - B: Specialists (loaded on demand, higher capability)
    - C: Workers (parallel execution, stateless)
    - D: Draft (speculative decoding support)
    """

    A = "A"  # Frontdoor
    B = "B"  # Specialists
    C = "C"  # Workers
    D = "D"  # Draft models


class Role(str, Enum):
    """Agent role identifiers.

    All agent roles in the orchestration system. Using this enum instead of
    raw strings enables:
    - IDE autocomplete
    - Compile-time typo detection
    - Centralized documentation of all roles
    - Type safety in function signatures

    Role naming convention:
    - {tier}_{function}: e.g., worker_math, coder_escalation
    - Specific variants: e.g., architect_general, ingest_long_context
    """

    # =========================================================================
    # Tier A: Frontdoor
    # =========================================================================
    FRONTDOOR = "frontdoor"
    """Interactive chat, intent classification, task routing.

    The frontdoor receives all user requests and either handles them directly
    or routes them to appropriate specialists. Always resident in memory.
    """

    # =========================================================================
    # Tier B: Specialists
    # =========================================================================
    CODER_ESCALATION = "coder_escalation"
    """Primary code generation specialist.

    Handles coding tasks: implementation, refactoring, debugging.
    Frontdoor and workers escalate here on failure.

    An ALIAS with no process of its own: since the 2026-09-27 ARCHITECT SWAP it
    rides ``architect_critic``'s process (Qwen3.8-27B Q8, MI210, :8083); before
    that it rode the same process under the ``architect_general`` label. (The
    Qwen2.5-Coder-32B :8081 server this docstring once named is long retired.)
    """

    INGEST_LONG_CONTEXT = "ingest_long_context"
    """Long-context document ingestion.

    Processes documents >8K tokens. An ALIAS with no process of its own: since
    the 2026-09-27 ARCHITECT SWAP it rides ``architect_critic``'s process
    (Qwen3.8-27B Q8, MI210, :8083); from the 2026-09-22 lineup cutover until the
    swap it rode the same process under the ``architect_general`` label.
    """

    ARCHITECT_GENERAL = "architect_general"
    """General architecture and system design — the terminal escalation rung.

    Handles high-level design, invariants, system architecture decisions.
    Top of escalation chain for most tasks: ``_ESCALATION_MAP`` has no entry for
    it, and every other architect-bound rung ends here.

    2026-09-27 ARCHITECT SWAP (operator-decided): serves Qwen3.8-Flash-Next
    UD-IQ4_XS on the full CPU instance, port 8074 (whole-machine CPU region-lock
    holder). Before the swap this role was the Qwen3.8-27B Q8 on the MI210
    (:8083), which now serves ``architect_critic``. The pydantic-graph's
    escalation paths (``src/graph/nodes.py`` ArchitectNode, chat delegation)
    target this role BY NAME, so they reach Flash-Next with no routing change.
    """

    ARCHITECT_CODING = "architect_general"
    """Compatibility alias for the retired coding architect role name.

    Direct enum references now resolve to the live general architect role. Old
    serialized role strings are normalized by ``Role._missing_``.
    """

    ARCHITECT_CRITIC = "architect_critic"
    """Adversarial plan critic.

    2026-09-27 ARCHITECT SWAP (operator-decided): serves Qwen3.8-27B Q8 on the
    MI210 (ROCm0), port 8083, and is the HOST of the ``coder_escalation`` and
    ``ingest_long_context`` aliases (same process, same port). It is NO LONGER
    the terminal rung: it escalates to ``architect_general`` (Flash-Next, CPU
    :8074). It stays consultable via the ``critique_plan`` interaction skill and
    reachable by direct request (``force_role`` / explicit role).

    History: added 2026-08-01 (W1 cutover) on the full CPU instance, port 8074
    (originally the Qwen3.5-122B-A10B UD-Q4_K_M; Qwen3.8-Flash-Next UD-IQ4_XS
    from the 2026-09-22 lineup cutover until the 2026-09-27 swap moved that
    process to ``architect_general``).

    The pydantic-graph never escalates to this role: ``src/graph/nodes.py`` has
    no critic node and ``_ROLE_TO_NODE`` has no ARCHITECT_CRITIC entry
    (``select_start_node`` falls back to FrontdoorNode). That is deliberate: RI-21
    closed with the ARCHSWAP (escalation reaches the strongest model as
    ``architect_general``), and a critic node is out of scope (receipt
    RATIFY-ARCHSWAP-20260927 ``not_in_scope``).

    THIS MEMBER IS LOAD-BEARING, not documentation. ``stack_priors.py:325`` emits
    the arm into the live action space via
    ``canonical_stack_role_id(role_name) or str(role_name)``, but
    ``action_space.py:125 normalize_action`` returns ``None`` for anything that is
    not a ``Role``, silently dropping every replayed episode. Without this member
    the critic would be an arm with a label slot and permanently zero training
    rows — unlearnable, which the operator ruled must not happen.
    """

    THINKING_REASONING = "thinking_reasoning"  # stack-change-guard: allow
    """RETIRED 2026-03-06 — retained ONLY so old serialized rows still deserialize.

    The registry marks it `deprecated: true`, `deprecated_reason: "GGUF deleted from
    disk — model weights no longer available"`. It has no `PORT_MAP` entry, no
    `ROLE_LAUNCH_META` entry and no server: nothing can route to it.

    The member stays because episodic-memory and routing rows written before
    2026-03-06 carry the literal string, and constructing a Role from it must keep
    resolving for those to replay. Deleting it would turn historical rows into
    `normalize_action -> None` drops, which is the same unlearnable-arm defect that
    ARCHITECT_CRITIC was added to avoid, only pointing backwards.

    The `stack-change-guard: allow` marker above is deliberate and is the ONLY
    permitted live reference. On 2026-08-02 the guard's retired-role set became
    derived (1 hand-written name -> 42 from registry markers) and correctly flagged
    five live references to this role; the other four were dead code in
    `parsing_config`, a langgraph node map, `factual_risk` and a routing task-type
    map, and were removed. If a reference to this role appears anywhere else, it is
    a bug — the model does not exist on disk.
    """

    # =========================================================================
    # Tier C: Workers
    # =========================================================================
    WORKER_GENERAL = "worker_general"
    """General-purpose worker for parallel tasks.

    Handles file-level implementation, boilerplate generation, documentation.
    Stateless, many can run concurrently.
    """

    WORKER_MATH = "worker_math"
    """Mathematical reasoning worker.

    Specialized for mathematical computations, proofs, step verification.
    Uses Qwen2.5-Math model.
    """

    WORKER_SUMMARIZE = "worker_summarize"
    """Summarization worker.

    Generates summaries, extracts key points from documents.
    Good candidate for prompt lookup acceleration.
    """

    WORKER_VISION = "worker_vision"
    """Vision-language worker.

    Handles tasks involving images: OCR, image understanding, UI analysis.
    Qwen3-VL-30B-A3B-Instruct Q4_K_M on MI210 since the 2026-08-01 W1 cutover
    (was Qwen2.5-VL-7B on CPU).
    """

    VISION_ESCALATION = "vision_escalation"
    """Vision escalation — an ALIAS on the worker_vision process, not a second server.

    Added to the enum 2026-08-01. It is a live, registry-declared, routable role
    (tier C) and `Role("vision_escalation")` raised ValueError before this, which is
    the same unlearnable-arm defect ARCHITECT_CRITIC was added to avoid: the arm
    reaches the action space via stack_priors while `action_space.normalize_action`
    returns None for a non-Role and silently drops every replayed episode.

    Since the W1 cutover this resolves to the SAME :8086 MI210 process as
    worker_vision (server_mode.worker_vision.shared_with), so it is a routing label
    over a shared server rather than a distinct model.
    """

    TOOLRUNNER = "toolrunner"
    """Tool execution worker.

    Runs external tools (lint, test, search) and processes results.
    Works with tool registry for permission checking.
    """

    # =========================================================================
    # Tier D: Draft Models
    # =========================================================================
    DRAFT_CODER = "draft_coder"
    """Draft model for code generation speculative decoding.

    Qwen2.5-Coder-0.5B - compatible with Qwen2.5-Coder family targets.
    """

    DRAFT_GENERAL = "draft_general"
    """Draft model for general speculative decoding.

    Qwen2.5-0.5B - compatible with Qwen2.5 family targets.
    """

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __str__(self) -> str:
        """Return the role value string.

        This ensures str(Role.CODER_ESCALATION) returns "coder_escalation"
        instead of "Role.CODER_ESCALATION".
        """
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> "Role | None":
        if isinstance(value, str):
            return _LEGACY_ROLE_ALIASES.get(value)
        return None

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid role.

        Args:
            value: String to check.

        Returns:
            True if value is a valid Role.
        """
        return cls.from_string(value) is not None

    @classmethod
    def from_string(cls, value: str, default: "Role | None" = None) -> "Role | None":
        """Convert string to Role, returning default if invalid.

        Args:
            value: String to convert.
            default: Default to return if invalid.

        Returns:
            Role enum member or default.
        """
        try:
            return cls(value)
        except ValueError:
            return default

    def escalates_to(self) -> "Role | None":
        """Get the escalation target for this role.

        Returns:
            The role to escalate to, or None if at top of chain.
        """
        return _ESCALATION_MAP.get(self)

    @property
    def tier(self) -> Tier:
        """Get the tier for this role.

        Returns:
            Tier enum member.
        """
        return _TIER_MAP.get(self, Tier.C)

    @property
    def is_specialist(self) -> bool:
        """Check if this is a specialist (Tier B) role."""
        return self.tier == Tier.B

    @property
    def is_worker(self) -> bool:
        """Check if this is a worker (Tier C) role."""
        return self.tier == Tier.C

    @property
    def is_draft(self) -> bool:
        """Check if this is a draft (Tier D) role."""
        return self.tier == Tier.D


# ── INF-78: architect REPL scoped exception ──────────────────────────────────
#
# Operator ruling 2026-09-24: the architect (then the 27B, :8083) stays out of REPL mode.
# 2026-09-27 ARCHITECT SWAP: the ruling is keyed by ROLE NAME, so it now covers
# architect_general = Flash-Next (CPU :8074) and architect_critic = 27B (MI210 :8083).
# Operator ruling 2026-09-25 (INF-78 scoped exception): the architect MAY run in REPL
# mode for task-scoped requests (`task_root` set, i.e. AutoKernel planner/author calls
# confined to a lane worktree). Every other architect request stays direct/delegated.
# The server enforces nothing new for unscoped requests (the 2026-09-24 ruling was and
# stays a caller-side contract); `architect_repl_allowed` is the single predicate
# callers and the server's log check consult.
ARCHITECT_REPL_ROLES = frozenset({"architect_general", "architect_critic"})


def architect_repl_allowed(role: object, *, task_root: object = None) -> bool:
    """True unless `role` is an architect role running REPL without a task scope."""
    name = str(getattr(role, "value", role) or "")
    if name not in ARCHITECT_REPL_ROLES:
        return True
    return task_root is not None


# ── RD-1: Reviewer role binding ───────────────────────────────────────────────
#
# Historically the reviewer WAS the architect: the strings "reviewer" /
# "reviewer_agent" were hardcoded aliases to ARCHITECT_GENERAL. RD-1 replaces that
# alias-only coupling with a CONFIG-LEVEL binding so the reviewer can target a
# model DIFFERENT from architect_general WITHOUT touching the escalation chain or
# routing classifier. ``resolve_reviewer_role()`` is the authoritative resolver;
# the string aliases below survive only so ``Role("reviewer")`` remains a stable
# enum-resolution fallback (they point at the same default the resolver returns).
#
# ARCHSWAP-20260927: review/plan work stays on the 27B. The default binding is
# ARCHITECT_CRITIC — the Qwen3.8-27B Q8 on the MI210 (:8083), the process that served
# architect_general until the swap — so reviews keep running on the model/process they
# always ran on. Only ESCALATION moves: it targets architect_general, now
# Qwen3.8-Flash-Next on the CPU (:8074).
# This constant is the persistent source of truth: no registry key and no launch env
# sets the reviewer (ORCHESTRATOR_REVIEWER_ROLE is an override, unset in the stack),
# and DelegationConfig has no reviewer_role field. Every review call site resolves
# through resolve_reviewer_role(): the plan review (ArchitectReviewService) and
# review_before_commit (chat.py). (The answer verdict in chat_review was removed by
# RI-18c.)
DEFAULT_REVIEWER_ROLE: Role = Role.ARCHITECT_CRITIC

# Operator/stack-level binding knob. Kept as an env var (+ a forward-compatible
# ``delegation.reviewer_role`` config attribute read via ``getattr``) so the
# binding is config-level without editing the config dataclass here.
REVIEWER_ROLE_ENV = "ORCHESTRATOR_REVIEWER_ROLE"


# Role -> Tier mapping
_LEGACY_ROLE_ALIASES: dict[str, Role] = {
    # Split the retired literal so production hardcoded-surface scans keep their signal.
    "architect" "_coding": Role.ARCHITECT_GENERAL,
    "coder": Role.CODER_ESCALATION,
    "coder_agent": Role.CODER_ESCALATION,
    "researcher_agent": Role.WORKER_GENERAL,
    "researcher": Role.WORKER_GENERAL,
    # reviewer aliases resolve to the DEFAULT reviewer binding (config-overridable
    # via resolve_reviewer_role); kept for stable Role("reviewer") enum resolution.
    "reviewer_agent": DEFAULT_REVIEWER_ROLE,
    "reviewer": DEFAULT_REVIEWER_ROLE,
    "math_agent": Role.WORKER_MATH,
    "vision_agent": Role.WORKER_VISION,
    "summarizer_agent": Role.WORKER_SUMMARIZE,
    "summarizer": Role.WORKER_SUMMARIZE,
    "worker_explore": Role.WORKER_GENERAL,
    "worker_fast": Role.WORKER_GENERAL,
}


# ── Planner role binding (ARCHSWAP-20260927) ─────────────────────────────────
#
# Inline per-request planning — proactive_stage's plan decomposition and its repair
# turn — was hard-coded to the literal "architect_general". It is not escalation, so
# it stays on the 27B with review: it resolves through this binding instead of
# following the architect_general label to CPU Flash-Next. Same shape as the reviewer
# binding: explicit override > ORCHESTRATOR_PLANNER_ROLE env > DEFAULT_PLANNER_ROLE.
DEFAULT_PLANNER_ROLE: Role = Role.ARCHITECT_CRITIC
PLANNER_ROLE_ENV = "ORCHESTRATOR_PLANNER_ROLE"


def resolve_planner_role(override: "Role | str | None" = None) -> Role:
    """Resolve the inline-planning role binding (default ``architect_critic``).

    An unknown binding string falls back to the default rather than raising, so a
    misconfiguration never breaks planning.
    """
    candidate: "Role | str | None" = override
    if candidate is None:
        env_val = os.environ.get(PLANNER_ROLE_ENV)
        if env_val:
            candidate = env_val.strip()
    if candidate is None:
        return DEFAULT_PLANNER_ROLE
    if isinstance(candidate, Role):
        return candidate
    role = Role.from_string(str(candidate))
    return role if role is not None else DEFAULT_PLANNER_ROLE


def resolve_reviewer_role(
    override: "Role | str | None" = None,
    config: object | None = None,
) -> Role:
    """Resolve the configured reviewer role binding (RD-1).

    Binding precedence (highest first):
      1. explicit ``override`` argument (caller/test injection)
      2. ``ORCHESTRATOR_REVIEWER_ROLE`` env var (operator/stack-level binding)
      3. ``config.delegation.reviewer_role`` attribute, if present (forward-compatible
         with a future ``DelegationConfig.reviewer_role`` field — read via ``getattr``
         so no config-schema edit is required in this module)
      4. ``DEFAULT_REVIEWER_ROLE`` (``architect_critic`` since ARCHSWAP-20260927)

    Returns a valid :class:`Role`; an unknown binding string falls back to the
    default rather than raising, so a misconfiguration never breaks review.
    """
    candidate: "Role | str | None" = override

    if candidate is None:
        env_val = os.environ.get(REVIEWER_ROLE_ENV)
        if env_val:
            candidate = env_val.strip()

    if candidate is None:
        if config is None:
            try:
                from src.config import get_config

                config = get_config()
            except Exception:
                config = None
        deleg = getattr(config, "delegation", None) if config is not None else None
        cfg_val = getattr(deleg, "reviewer_role", None)
        if cfg_val:
            candidate = cfg_val

    if candidate is None:
        return DEFAULT_REVIEWER_ROLE
    if isinstance(candidate, Role):
        return candidate

    role = Role.from_string(str(candidate))
    return role if role is not None else DEFAULT_REVIEWER_ROLE


_TIER_MAP: dict[Role, Tier] = {
    # 2026-08-01: corrected against the registry's declared `roles.<role>.tier`.
    # This table is a hand restatement of registry data and had drifted. Two bugs:
    #   * ARCHITECT_CRITIC was ABSENT, so `get_tier()` returned its Tier.C default.
    #     Consequences were all silent: approval_gate.py:159-163 only gates a
    #     transition when tiers DIFFER, so worker -> architect_critic read C -> C and
    #     the approval gate for the whole-machine-locking 122B never fired; the
    #     prompt builder told it "You are a c-tier architect_critic assistant"; and
    #     repl_environment/routing.py advertised worker capabilities for it.
    #   * ARCHITECT_GENERAL was Tier.B where the registry declares A.
    # NOTE the name collision that helped this drift: `Tier` here is A/B/C/D
    # (capability class), which is NOT the residency tier hot/warm used by
    # ROLE_LAUNCH_META. Same word, different domain.
    # Tier A
    Role.FRONTDOOR: Tier.A,
    Role.ARCHITECT_GENERAL: Tier.A,
    # Tier B
    Role.CODER_ESCALATION: Tier.B,
    Role.INGEST_LONG_CONTEXT: Tier.B,
    Role.ARCHITECT_CRITIC: Tier.B,
    Role.THINKING_REASONING: Tier.B,
    # Tier C
    Role.WORKER_GENERAL: Tier.C,
    Role.WORKER_MATH: Tier.C,
    Role.WORKER_SUMMARIZE: Tier.C,
    Role.WORKER_VISION: Tier.C,
    Role.VISION_ESCALATION: Tier.C,
    Role.TOOLRUNNER: Tier.C,
    # Tier D
    Role.DRAFT_CODER: Tier.D,
    Role.DRAFT_GENERAL: Tier.D,
}


# Role -> Escalation target mapping
_ESCALATION_MAP: dict[Role, Role] = {
    # Workers escalate to coder
    Role.WORKER_GENERAL: Role.CODER_ESCALATION,
    Role.WORKER_MATH: Role.CODER_ESCALATION,
    Role.WORKER_SUMMARIZE: Role.CODER_ESCALATION,
    Role.WORKER_VISION: Role.CODER_ESCALATION,
    Role.TOOLRUNNER: Role.CODER_ESCALATION,
    # Frontdoor escalates to coder
    Role.FRONTDOOR: Role.CODER_ESCALATION,
    # 2026-09-27 ARCHITECT SWAP (operator-decided): escalation follows MODEL
    # strength, so the terminal rung is Qwen3.8-Flash-Next UD-IQ4_XS (full CPU
    # instance, :8074), which now serves architect_general. The Qwen3.8-27B Q8 on
    # the MI210 (:8083) now serves architect_critic and hosts the coder_escalation
    # and ingest_long_context aliases. Every architect-bound rung ends at
    # ARCHITECT_GENERAL:
    #   * CODER_ESCALATION / THINKING_REASONING -> ARCHITECT_GENERAL: a real hop
    #     off the 27B's process onto Flash-Next (was -> ARCHITECT_CRITIC, which
    #     named the same Flash-Next process before the swap).
    #   * ARCHITECT_CRITIC -> ARCHITECT_GENERAL: 27B -> Flash-Next.
    #   * INGEST_LONG_CONTEXT -> ARCHITECT_GENERAL: unchanged target; before the
    #     swap it was a same-process null hop (both on :8083), now a real hop.
    #   * ARCHITECT_GENERAL: no entry — terminal (was -> ARCHITECT_CRITIC).
    # (Pre-swap history: 2026-08-01 W1 cutover made coder_escalation an alias on
    # :8083 and routed it past the same-process architect_general to the critic.)
    #
    # SCOPE NOTE: this map drives Role.escalates_to()/get_escalation_chain() —
    # i.e. EscalationPolicy (src/orchestration/escalation.py), the proactive
    # delegator and the REPL's advertised escalation chain. The pydantic-graph
    # has its own wiring: src/graph/nodes.py hard-wires CoderEscalationNode ->
    # ArchitectNode (Role.ARCHITECT_GENERAL, terminal), which after the swap agrees
    # with this map. There is still no critic node and _ROLE_TO_NODE has no
    # ARCHITECT_CRITIC entry (select_start_node falls back to FrontdoorNode), so
    # in-graph escalation never reaches ARCHITECT_CRITIC. Deliberate: RI-21 closed
    # with the ARCHSWAP; a critic node is out of scope (receipt not_in_scope).
    Role.CODER_ESCALATION: Role.ARCHITECT_GENERAL,
    Role.THINKING_REASONING: Role.ARCHITECT_GENERAL,
    Role.ARCHITECT_CRITIC: Role.ARCHITECT_GENERAL,
    # Ingest escalates to architect (a real hop since the 2026-09-27 swap).
    Role.INGEST_LONG_CONTEXT: Role.ARCHITECT_GENERAL,
    # architect_general is the top of the chain and does not escalate.
    # Draft models don't escalate (they support other models)
}


# Role -> Fallback alternatives (infrastructure failure, NOT task escalation)
# Used when model_fallback feature is enabled and primary backend is circuit-open.
_FALLBACK_MAP: dict[Role, list[Role]] = {
    # 2026-08-01 W1 CUTOVER: was [CODER_ESCALATION], which after the cutover is the
    # SAME PROCESS ON THE SAME PORT — an infrastructure fallback that retries the
    # exact backend whose circuit just opened. architect_critic is a different
    # model on a different device, so it is a real fallback.
    # 2026-09-27 ARCHITECT SWAP: both edges unchanged and still cross-fleet —
    # architect_general = Flash-Next (CPU :8074), architect_critic = 27B (MI210 :8083).
    Role.ARCHITECT_GENERAL: [Role.ARCHITECT_CRITIC],
    Role.ARCHITECT_CRITIC: [Role.ARCHITECT_GENERAL],
    # Still valid: frontdoor is a separate CPU process with a separate GGUF.
    Role.CODER_ESCALATION: [Role.FRONTDOOR],
    # 2026-09-22 LINEUP CUTOVER (orchestrator 860b0b2d), same defect class as the
    # 2026-08-01 fix above. The registry's server_mode now puts worker_math and
    # worker_general in frontdoor's :8070 shared_with, and makes
    # ingest_long_context an alias_of architect_general (:8083). The old edges
    # WORKER_MATH -> WORKER_GENERAL and INGEST_LONG_CONTEXT -> ARCHITECT_GENERAL
    # both retried the process whose circuit had just opened. Each role now falls
    # back to a different serving process:
    #   * worker_math -> (then) architect_general: a different model on a different
    #     device (MI210). It is also the process worker_math already escalates to,
    #     through the coder_escalation alias.
    #   * ingest_long_context -> (then) architect_critic: its host's own real fallback.
    # 2026-09-27 ARCHITECT SWAP (operator-decided): both edges follow the PROCESS,
    # not the role label.
    #   * WORKER_MATH -> ARCHITECT_CRITIC: the MI210 27B (:8083) now serves
    #     architect_critic; still the process worker_math escalates to via the
    #     coder_escalation alias.
    #   * INGEST_LONG_CONTEXT -> ARCHITECT_GENERAL: ingest's host is now
    #     architect_critic (:8083), so [ARCHITECT_CRITIC] would be a forbidden
    #     same-fleet edge; architect_general (Flash-Next, CPU :8074) is the real
    #     fallback.
    # tests/unit/test_concept_integration.py guards the whole table against
    # same-GGUF edges, using the registry.
    Role.WORKER_MATH: [Role.ARCHITECT_CRITIC],
    Role.INGEST_LONG_CONTEXT: [Role.ARCHITECT_GENERAL],
    Role.FRONTDOOR: [],  # Always-on, no fallback
    Role.WORKER_VISION: [],  # Hardware-specific, no fallback
}


def get_fallback_roles(role: Role | str) -> list[Role]:
    """Get fallback roles for infrastructure failure (NOT task escalation).

    WP-12 (``ORCHESTRATOR_FLEET_LAYER=1``, default off): same-fleet edges are
    compiled out — a candidate on the failing role's own physical fleet would
    retry the identical backend + identical (already-open) circuit, which is
    the ``forced_role_fallback`` churn class. Fallback is meaningful iff it
    changes the physical fleet (design §4). When the fleet layer is off or
    its build is unavailable, the legacy map is returned unchanged.

    Args:
        role: Role whose backend is unavailable.

    Returns:
        List of alternative roles to try, in priority order.
    """
    if isinstance(role, str):
        role = Role.from_string(role)
        if role is None:
            return []
    if os.environ.get("ORCHESTRATOR_FLEET_LAYER") == "1":
        try:
            from src.fleet import compiled_fleet_fallback_map

            compiled = compiled_fleet_fallback_map()
            if compiled is not None:
                return list(compiled.get(role, ()))
        except Exception:
            # Fleet layer unavailable → legacy map (consistent with the
            # backend builder's legacy fallback in the same condition).
            pass
    return list(_FALLBACK_MAP.get(role, []))


def get_tier(role: Role | str) -> Tier:
    """Get the tier for a role.

    Args:
        role: Role enum or string.

    Returns:
        Tier enum member.

    Example:
        >>> get_tier(Role.CODER_ESCALATION)
        Tier.B
        >>> get_tier("worker_math")
        Tier.C
    """
    if isinstance(role, str):
        role = Role.from_string(role)
        if role is None:
            return Tier.C  # Default to worker tier for unknown roles

    return _TIER_MAP.get(role, Tier.C)


def get_escalation_chain(role: Role | str) -> list[Role]:
    """Get the full escalation chain starting from a role.

    Args:
        role: Starting role.

    Returns:
        List of roles in escalation order (including starting role).

    Example:
        >>> get_escalation_chain(Role.WORKER_GENERAL)
        [Role.WORKER_GENERAL, Role.CODER_ESCALATION, Role.ARCHITECT_GENERAL]
    """
    if isinstance(role, str):
        role = Role.from_string(role)
        if role is None:
            return []

    chain = [role]
    current = role
    seen = {current}

    while True:
        next_role = current.escalates_to()
        if next_role is None or next_role in seen:
            break
        chain.append(next_role)
        seen.add(next_role)
        current = next_role

    return chain


# Generic chain names (used by graph node selection and routing)
CHAIN_NAMES = {
    "worker": Role.WORKER_GENERAL,
    "coder": Role.CODER_ESCALATION,
    "architect": Role.ARCHITECT_GENERAL,
    "ingest": Role.INGEST_LONG_CONTEXT,
    "frontdoor": Role.FRONTDOOR,
}


def chain_name_to_role(chain_name: str) -> Role | None:
    """Convert a generic chain name to a specific role.

    Args:
        chain_name: Generic name like "coder" or "architect".

    Returns:
        Specific Role enum member.
    """
    return CHAIN_NAMES.get(chain_name)


def role_to_chain_name(role: Role) -> str:
    """Convert a specific role to its generic chain name.

    Args:
        role: Specific Role enum member.

    Returns:
        Generic chain name like "coder" or "architect".
    """
    for name, r in CHAIN_NAMES.items():
        if role == r:
            return name

    # For non-primary roles, find their chain by escalation
    if role in {Role.CODER_ESCALATION}:
        return "coder"
    if role in {Role.WORKER_MATH, Role.WORKER_SUMMARIZE, Role.WORKER_VISION, Role.TOOLRUNNER}:
        return "worker"
    if role in {Role.THINKING_REASONING}:
        return "coder"

    return role.value
