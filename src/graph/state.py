"""Graph state, dependencies, and result types for orchestration graph.

TaskState is the mutable state threaded through the graph execution.
TaskDeps holds immutable references to infrastructure (LLM, REPL, MemRL).
TaskResult is the final output produced at graph termination.
GraphConfig controls retry/escalation limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

from src.escalation import ErrorCategory
from src.roles import Role

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives
    from src.repl_environment import REPLEnvironment
    from src.api.protocols import FailureGraphProtocol


def _think_cfg():
    from src.config import get_config

    return get_config().think_harder


def _make_task_manager():
    from orchestration.tools.task_management import TaskManager

    return TaskManager()


# ---------------------------------------------------------------------------
# GraphConfig — controls retry/escalation limits
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """Configuration for the orchestration graph.

    Defaults are intentionally sourced from the centralized config module
    so that environment variable overrides work transparently.
    """

    max_retries: int = 2
    max_escalations: int = 2
    max_turns: int = 15
    optional_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({"typecheck", "integration", "shellcheck"})
    )
    no_escalate_categories: frozenset[ErrorCategory] = field(
        default_factory=lambda: frozenset({ErrorCategory.FORMAT})
    )
    confidence_threshold: float = 0.7

    @classmethod
    def from_config(cls) -> GraphConfig:
        """Create from centralized config (src.config)."""
        from src.config import get_config

        esc = get_config().escalation
        return cls(
            max_retries=esc.max_retries,
            max_escalations=esc.max_escalations,
            optional_gates=esc.optional_gates,
        )


# ---------------------------------------------------------------------------
# TaskState — mutable state tracked across graph execution
# ---------------------------------------------------------------------------


@dataclass
class TaskState:
    """Mutable state that travels through the graph.

    Updated by each node's ``run()`` method. Persisted via snapshots
    when using SQLiteStatePersistence.
    """

    task_id: str = ""
    prompt: str = ""
    context: str = ""
    current_role: Role | str = Role.FRONTDOOR
    consecutive_failures: int = 0
    consecutive_nudges: int = 0
    escalation_count: int = 0
    role_history: list[str] = field(default_factory=list)
    escalation_prompt: str = ""
    last_error: str = ""
    last_output: str = ""
    last_code: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    task_ir: dict[str, Any] = field(default_factory=dict)
    task_type: str = "chat"
    turns: int = 0
    max_turns: int = 15
    gathered_files: list[str] = field(default_factory=list)
    last_failure_id: str | None = None
    anti_pattern_warning: str = ""
    task_manager: Any = field(default_factory=_make_task_manager)

    # Delegation tracking
    delegation_events: list[dict] = field(default_factory=list)

    # Session compaction tracking
    compaction_count: int = 0
    compaction_tokens_saved: int = 0
    context_file_paths: list[str] = field(default_factory=list)
    last_compaction_turn: int = 0

    # UFH14-B1 F2: wall-time forced-answer turn (perf_counter time; None = off)
    answer_force_at_s: float | None = None
    answer_forced_turns: int = 0

    # Session log tracking (append-only processing journal)
    session_log_path: str = ""
    session_log_records: list[Any] = field(default_factory=list)
    session_summary_cache: str = ""
    session_summary_turn: int = 0

    # Session scratchpad (model-extracted semantic insights)
    scratchpad_entries: list[Any] = field(default_factory=list)

    # Two-level condensation (CF Phase 1)
    consolidated_segments: list[Any] = field(default_factory=list)
    pending_granular_blocks: list[str] = field(default_factory=list)
    pending_granular_start_turn: int = 0

    # Segment cache for hash-based dedup (CF Phase 1+)
    segment_cache: Any = None  # SegmentCache instance, lazily initialized

    # Compaction quality monitor (CF Phase 3b)
    compaction_quality_monitor: Any = None  # CompactionQualityMonitor instance

    # Budget controls (Fast-RLM)
    repl_executions: int = 0       # Total REPL execute() calls across all turns
    aggregate_tokens: int = 0      # Cumulative completion tokens across all turns

    # REPL loop-guard: consecutive no-progress turns (no file write / no FINAL). MUST be a
    # DECLARED field, not an ad-hoc `state._x` attribute — the graph snapshots/restores state
    # between turns from declared fields only, so an ad-hoc attr resets every turn and the
    # loop-guard intervention never fires (root cause of the BEP-2 multi-file non-firing, 2026-05-27).
    repl_noprogress_count: int = 0

    # Resume token for crash recovery (Phase 3B)
    resume_token: str = ""

    # Pending approval for halt-and-resume (Phase 4A)
    pending_approval: Any = None

    # Think-harder escalation (same model, boosted config before model swap)
    think_harder_config: dict | None = None
    think_harder_attempted: bool = False
    think_harder_succeeded: bool | None = None
    # Per-role think-harder ROI stats for regulation.
    think_harder_roi_by_role: dict[str, dict[str, float]] = field(default_factory=dict)
    think_harder_min_expected_roi: float = field(
        default_factory=lambda: _think_cfg().min_expected_roi
    )
    think_harder_min_samples: int = field(default_factory=lambda: _think_cfg().min_samples)
    think_harder_cooldown_turns: int = field(default_factory=lambda: _think_cfg().cooldown_turns)
    think_harder_ema_alpha: float = field(default_factory=lambda: _think_cfg().ema_alpha)
    think_harder_min_marginal_utility: float = field(
        default_factory=lambda: _think_cfg().min_marginal_utility
    )

    # Tool requirement (from routing classification)
    tool_required: bool = False
    tool_hint: str | None = None

    # Difficulty-adaptive signal (from routing classifier, used for token budget modulation)
    difficulty_band: str = ""

    # Grammar enforcement tracking
    grammar_enforced: bool = False

    # Cache affinity bonus applied during routing
    cache_affinity_bonus: float = 0.0

    # Tool output compression metrics (populated in helpers.py when compression is active)
    compression_metrics: dict[str, Any] = field(default_factory=dict)

    # Global workspace state (shared blackboard across delegation/escalation turns).
    workspace_state: dict[str, Any] = field(
        default_factory=lambda: {
            "version": 1,
            "broadcast_version": 0,
            "selection_policy": "priority_then_recency",
            "objective": "",
            "constraints": [],
            "invariants": [],
            "proposals": [],
            "commitments": [],
            "open_questions": [],
            "resolved_questions": [],
            "decisions": [],
            "broadcast_log": [],
            "updated_at": "",
        }
    )

    def record_role(self, role: Role | str) -> None:
        """Append a role to history and update current_role."""
        self.current_role = role
        self.role_history.append(str(role))


# ---------------------------------------------------------------------------
# TaskResult — final output produced at graph termination
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Result produced when the graph reaches an End node."""

    answer: str
    success: bool
    role_history: list[str] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)
    tools_used: int = 0
    turns: int = 0
    delegation_events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MemRLSuggestor Protocol — pluggable MemRL query interface
# ---------------------------------------------------------------------------


@runtime_checkable
class MemRLSuggestor(Protocol):
    """Protocol for querying learned escalation policy."""

    def query_escalation(
        self,
        role: str,
        error_category: str,
        error_message: str,
        failure_count: int,
    ) -> MemRLSuggestion | None: ...


@dataclass
class MemRLSuggestion:
    """Advisory suggestion from learned escalation policy."""

    action: str  # "retry", "escalate", "fail"
    target_role: str | None = None
    confidence: float = 0.0
    similar_cases: int = 0
    memory_id: str | None = None


# ---------------------------------------------------------------------------
# HypothesisGraphProtocol — for typing deps
# ---------------------------------------------------------------------------


@runtime_checkable
class HypothesisGraphProtocol(Protocol):
    """Interface for hypothesis confidence tracking."""

    def add_evidence(self, hypothesis_id: str, outcome: str, source: str) -> float: ...
    def get_confidence(self, action: str, task_type: str) -> float: ...


# ---------------------------------------------------------------------------
# TaskDeps — immutable dependency container
# ---------------------------------------------------------------------------


@dataclass
class TaskDeps:
    """Immutable references injected into the graph at run time.

    These are not modified by graph execution — they provide the
    infrastructure that nodes need to do their work.
    """

    primitives: LLMPrimitives | None = None
    repl: REPLEnvironment | None = None
    failure_graph: FailureGraphProtocol | None = None
    hypothesis_graph: HypothesisGraphProtocol | None = None
    memrl_suggestor: MemRLSuggestor | None = None
    config: GraphConfig = field(default_factory=GraphConfig)
    progress_logger: Any = None  # ProgressLoggerProtocol
    session_store: Any = None  # SQLiteSessionStore
    # D-f3: set only when the turn holds the session lease, so graph snapshot
    # writes are session-scoped and fenced; otherwise snapshots are run-scoped.
    session_id: str | None = None
    session_fencing_token: int | None = None
    approval_callback: Any = None  # ApprovalCallback protocol (Phase 4A)
    interrupt_conditions: list[Any] = field(default_factory=list)  # InterruptCondition protocol instances
