"""LangGraph state definition — TypedDict equivalent of TaskState.

Maps the 50+ field pydantic_graph TaskState dataclass into a LangGraph-compatible
TypedDict with appropriate reducer annotations. Fields use three reducer strategies:

- ``replace`` (default): Last writer wins — used for scalars and overwritten-each-turn fields
- ``append`` (operator.add): List concatenation — used for append-only lists
- ``custom merge``: Deep merge with conflict resolution — used for dicts with complex semantics

Config constants (think_harder_min_*, etc.) are moved to LangGraphConfig to keep
checkpointed state lean.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState  # noqa: F401 — re-export for convenience
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Custom reducers for complex dict fields
# ---------------------------------------------------------------------------


def _merge_artifacts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Deep merge artifacts — latest turn wins on key conflict."""
    merged = {**left, **right}
    return merged


def _merge_think_harder_roi(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Merge per-role think-harder ROI stats — latest values per role win."""
    merged = {**left}
    for role, stats in right.items():
        if role in merged:
            merged[role] = {**merged[role], **stats}
        else:
            merged[role] = stats
    return merged


def _merge_workspace_state(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Version-aware workspace merge.

    - Higher ``version`` wins on conflict
    - ``broadcast_log`` appends
    - Lists (``proposals``, ``commitments``, etc.) use union-merge
    """
    if right.get("version", 0) < left.get("version", 0):
        return left

    merged = {**left, **right}

    # Append broadcast_log
    left_log = left.get("broadcast_log", [])
    right_log = right.get("broadcast_log", [])
    if right_log and left_log:
        merged["broadcast_log"] = left_log + [
            entry for entry in right_log if entry not in left_log
        ]

    # Union-merge list fields
    for key in ("proposals", "commitments", "open_questions", "resolved_questions", "decisions"):
        left_list = left.get(key, [])
        right_list = right.get(key, [])
        if left_list or right_list:
            # Use id-based dedup if items have 'id', otherwise simple union
            seen_ids = set()
            merged_list = []
            for item in left_list + right_list:
                item_id = item.get("id") if isinstance(item, dict) else id(item)
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    merged_list.append(item)
            merged[key] = merged_list

    return merged


# ---------------------------------------------------------------------------
# LangGraphConfig — config constants extracted from TaskState
# ---------------------------------------------------------------------------


class LangGraphConfig(TypedDict, total=False):
    """Non-checkpointed config passed via ``config["configurable"]``.

    Contains:
    - TaskDeps (primitives, repl, failure_graph, etc.)
    - Config constants that were previously on TaskState
    """

    # TaskDeps fields (not serialized into checkpoints)
    deps: Any  # TaskDeps instance

    # Config constants moved out of state
    think_harder_min_expected_roi: float
    think_harder_min_samples: int
    think_harder_cooldown_turns: int
    think_harder_ema_alpha: float
    think_harder_min_marginal_utility: float


# ---------------------------------------------------------------------------
# OrchestratorState — LangGraph TypedDict equivalent of TaskState
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict, total=False):
    """LangGraph state for the orchestration graph.

    Field-by-field mapping from TaskState with reducer annotations.
    Fields marked ``total=False`` are optional (may not be present in initial state).
    """

    # --- Identity & input (set once) ---
    task_id: str
    prompt: str
    context: str
    task_type: str
    task_ir: dict[str, Any]

    # --- Current execution state (replace semantics) ---
    current_role: str  # Role enum serialized as string
    consecutive_failures: int
    consecutive_nudges: int
    escalation_count: int
    escalation_prompt: str
    last_error: str
    last_output: str
    last_code: str
    turns: int
    max_turns: int
    last_failure_id: str
    anti_pattern_warning: str

    # --- Append-only lists ---
    role_history: Annotated[list[str], operator.add]
    gathered_files: Annotated[list[str], operator.add]
    delegation_events: Annotated[list[dict], operator.add]
    context_file_paths: Annotated[list[str], operator.add]
    session_log_records: Annotated[list[Any], operator.add]
    scratchpad_entries: Annotated[list[Any], operator.add]

    # --- Custom merge dicts ---
    artifacts: Annotated[dict[str, Any], _merge_artifacts]
    think_harder_roi_by_role: Annotated[
        dict[str, dict[str, float]], _merge_think_harder_roi
    ]
    workspace_state: Annotated[dict[str, Any], _merge_workspace_state]

    # --- Session compaction tracking ---
    compaction_count: int
    compaction_tokens_saved: int
    last_compaction_turn: int
    # UFH14-B1 F2: wall-time forced-answer turn
    answer_force_at_s: float | None
    answer_forced_turns: int
    session_log_path: str
    session_summary_cache: str
    session_summary_turn: int

    # --- Two-level condensation (CF Phase 1) ---
    consolidated_segments: Annotated[list[Any], operator.add]
    pending_granular_blocks: Annotated[list[str], operator.add]
    pending_granular_start_turn: int

    # --- Budget controls ---
    repl_executions: int
    aggregate_tokens: int

    # --- Resume / approval ---
    resume_token: str

    # --- Think-harder state ---
    think_harder_config: dict | None
    think_harder_attempted: bool
    think_harder_succeeded: bool | None

    # --- Routing classification ---
    tool_required: bool
    tool_hint: str | None
    difficulty_band: str
    grammar_enforced: bool
    cache_affinity_bonus: float

    # --- Terminal result (set by _handle_end) ---
    _result: dict[str, Any]

    # --- Graph control (internal) ---
    # Which node to execute next — used by conditional edges
    next_node: str


# ---------------------------------------------------------------------------
# Conversion helpers — TaskState <-> OrchestratorState
# ---------------------------------------------------------------------------

# Fields to skip when converting TaskState -> OrchestratorState
_SKIP_TO_LG = frozenset({
    "task_manager", "pending_approval",
    "segment_cache", "compaction_quality_monitor",
})

# Config constants that moved to LangGraphConfig
_CONFIG_FIELDS = frozenset({
    "think_harder_min_expected_roi",
    "think_harder_min_samples",
    "think_harder_cooldown_turns",
    "think_harder_ema_alpha",
    "think_harder_min_marginal_utility",
})


# Fields with operator.add reducers — nodes must return deltas, not full lists
APPEND_FIELDS = frozenset({
    "role_history", "gathered_files", "delegation_events",
    "context_file_paths", "session_log_records", "scratchpad_entries",
    "consolidated_segments", "pending_granular_blocks",
})


def snapshot_append_lengths(state_dict: dict[str, Any]) -> dict[str, int]:
    """Capture current lengths of all append-reducer fields.

    Call at node entry to record baseline lengths. Used by
    ``state_update_delta()`` to compute deltas.
    """
    return {
        field: len(state_dict.get(field, []))
        for field in APPEND_FIELDS
    }


def state_update_delta(
    update: dict[str, Any],
    snapshot_lengths: dict[str, int],
) -> dict[str, Any]:
    """Trim append-reducer fields in a state update dict to deltas only.

    Given a full state dict and the snapshot of list lengths at node entry,
    slices each append-reducer field to contain only new elements added
    during this node's execution. This prevents LangGraph's operator.add
    reducer from duplicating existing elements.
    """
    for field, orig_len in snapshot_lengths.items():
        if field in update and isinstance(update[field], list):
            update[field] = update[field][orig_len:]
    return update


def task_state_to_lg(state) -> dict[str, Any]:
    """Convert a TaskState dataclass to an OrchestratorState dict.

    Args:
        state: TaskState instance.

    Returns:
        Dict compatible with OrchestratorState TypedDict.
    """
    import dataclasses
    from enum import Enum

    result: dict[str, Any] = {}
    for f in dataclasses.fields(state):
        if f.name in _SKIP_TO_LG or f.name in _CONFIG_FIELDS:
            continue
        val = getattr(state, f.name)
        if isinstance(val, Enum):
            val = str(val)
        result[f.name] = val
    return result


def lg_to_task_state(lg_state: dict[str, Any], state) -> None:
    """Update a TaskState dataclass from an OrchestratorState dict.

    Mutates ``state`` in place. Skips fields not present in ``lg_state``
    and fields that are config constants or non-serializable.

    Args:
        lg_state: OrchestratorState dict.
        state: TaskState instance to update.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(state)}
    for key, val in lg_state.items():
        if key in _SKIP_TO_LG or key in _CONFIG_FIELDS:
            continue
        if key == "next_node":
            continue  # Internal graph control field
        if key in field_names:
            setattr(state, key, val)
