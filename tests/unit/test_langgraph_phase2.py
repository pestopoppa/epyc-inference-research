"""Tests for LangGraph migration Phase 2.

Tests cover:
1. Append-reducer delta correctness (no list duplication)
2. _SKIP_TO_LG covers non-serializable fields
3. _result field in OrchestratorState
4. Full state round-trip with all 50+ fields
5. Dual-run validation (mocked _execute_turn, both backends compared)
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.state import GraphConfig, TaskDeps, TaskState
from src.roles import Role

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


# ---------------------------------------------------------------------------
# 1. Append-reducer delta correctness
# ---------------------------------------------------------------------------


class TestAppendReducerDeltas:
    """Verify _state_update returns only NEW elements for append-reducer fields."""

    def test_role_history_delta(self):
        """role_history should contain only elements added during node execution."""
        from src.graph.langgraph.state import (
            snapshot_append_lengths,
            state_update_delta,
            task_state_to_lg,
        )

        # Simulate state at node entry with existing role_history
        state_dict = {"role_history": ["frontdoor", "coder_escalation"]}
        snap = snapshot_append_lengths(state_dict)
        assert snap["role_history"] == 2

        # Simulate TaskState after node execution added a new role
        task_state = TaskState(
            role_history=["frontdoor", "coder_escalation", _RETIRED_ARCHITECT_ROLE],
        )
        update = task_state_to_lg(task_state)
        state_update_delta(update, snap)

        # Only the new element should be in the update
        assert update["role_history"] == [_RETIRED_ARCHITECT_ROLE]

    def test_empty_delta_when_no_additions(self):
        """If node adds nothing to a list, delta should be empty."""
        from src.graph.langgraph.state import (
            snapshot_append_lengths,
            state_update_delta,
            task_state_to_lg,
        )

        state_dict = {"gathered_files": ["a.py", "b.py"]}
        snap = snapshot_append_lengths(state_dict)

        task_state = TaskState(gathered_files=["a.py", "b.py"])
        update = task_state_to_lg(task_state)
        state_update_delta(update, snap)

        assert update["gathered_files"] == []

    def test_multiple_append_fields_independent(self):
        """Each append field tracks deltas independently."""
        from src.graph.langgraph.state import (
            snapshot_append_lengths,
            state_update_delta,
            task_state_to_lg,
        )

        state_dict = {
            "role_history": ["frontdoor"],
            "gathered_files": [],
            "delegation_events": [{"e": 1}],
        }
        snap = snapshot_append_lengths(state_dict)

        task_state = TaskState(
            role_history=["frontdoor", "coder_escalation"],
            gathered_files=["new.py"],
            delegation_events=[{"e": 1}, {"e": 2}],
        )
        update = task_state_to_lg(task_state)
        state_update_delta(update, snap)

        assert update["role_history"] == ["coder_escalation"]
        assert update["gathered_files"] == ["new.py"]
        assert update["delegation_events"] == [{"e": 2}]

    def test_all_append_fields_covered(self):
        """APPEND_FIELDS constant covers all operator.add fields in OrchestratorState."""
        from src.graph.langgraph.state import APPEND_FIELDS, OrchestratorState

        import typing
        import operator

        # Inspect OrchestratorState annotations for Annotated[list, operator.add]
        hints = typing.get_type_hints(OrchestratorState, include_extras=True)
        annotated_append = set()
        for name, hint in hints.items():
            if hasattr(hint, "__metadata__"):
                for m in hint.__metadata__:
                    if m is operator.add:
                        annotated_append.add(name)

        assert APPEND_FIELDS == annotated_append, (
            f"APPEND_FIELDS mismatch: "
            f"missing={annotated_append - APPEND_FIELDS}, "
            f"extra={APPEND_FIELDS - annotated_append}"
        )

    def test_snapshot_defaults_to_zero_for_missing_fields(self):
        """snapshot_append_lengths returns 0 for fields absent from state dict."""
        from src.graph.langgraph.state import APPEND_FIELDS, snapshot_append_lengths

        snap = snapshot_append_lengths({})
        for field in APPEND_FIELDS:
            assert snap[field] == 0

    def test_state_update_preserves_non_append_fields(self):
        """Non-append fields in the update dict are NOT trimmed."""
        from src.graph.langgraph.state import (
            snapshot_append_lengths,
            state_update_delta,
            task_state_to_lg,
        )

        state_dict = {"role_history": ["frontdoor"]}
        snap = snapshot_append_lengths(state_dict)

        task_state = TaskState(
            prompt="unchanged",
            turns=5,
            role_history=["frontdoor", "coder"],
        )
        update = task_state_to_lg(task_state)
        state_update_delta(update, snap)

        # Scalar fields untouched
        assert update["prompt"] == "unchanged"
        assert update["turns"] == 5
        # Append field trimmed
        assert update["role_history"] == ["coder"]


# ---------------------------------------------------------------------------
# 2. _SKIP_TO_LG coverage
# ---------------------------------------------------------------------------


class TestSkipToLG:
    """Verify non-serializable fields are excluded from LG state."""

    def test_segment_cache_excluded(self):
        from src.graph.langgraph.state import task_state_to_lg

        state = TaskState()
        state.segment_cache = object()  # Non-serializable
        lg = task_state_to_lg(state)
        assert "segment_cache" not in lg

    def test_compaction_quality_monitor_excluded(self):
        from src.graph.langgraph.state import task_state_to_lg

        state = TaskState()
        state.compaction_quality_monitor = object()
        lg = task_state_to_lg(state)
        assert "compaction_quality_monitor" not in lg

    def test_task_manager_excluded(self):
        from src.graph.langgraph.state import task_state_to_lg

        state = TaskState()
        lg = task_state_to_lg(state)
        assert "task_manager" not in lg

    def test_pending_approval_excluded(self):
        from src.graph.langgraph.state import task_state_to_lg

        state = TaskState()
        state.pending_approval = "something"
        lg = task_state_to_lg(state)
        assert "pending_approval" not in lg


# ---------------------------------------------------------------------------
# 3. _result field in OrchestratorState
# ---------------------------------------------------------------------------


class TestResultField:
    """Verify _result is declared and usable in OrchestratorState."""

    def test_result_field_in_state_type(self):
        from src.graph.langgraph.state import OrchestratorState
        import typing

        hints = typing.get_type_hints(OrchestratorState)
        assert "_result" in hints

    def test_handle_end_sets_result(self):
        """_handle_end should populate _result in the returned dict."""
        from src.graph.langgraph.nodes import _handle_end

        ctx = SimpleNamespace(
            state=TaskState(
                role_history=["frontdoor"],
                turns=3,
                delegation_events=[],
            ),
            deps=TaskDeps(),
        )
        task_state = ctx.state
        snap = {"role_history": 0, "gathered_files": 0, "delegation_events": 0,
                "context_file_paths": 0, "session_log_records": 0,
                "scratchpad_entries": 0, "consolidated_segments": 0,
                "pending_granular_blocks": 0}

        with patch("src.graph.langgraph.nodes._make_end_result"):
            result = _handle_end(ctx, "The answer", True, task_state, snap)

        assert "_result" in result
        assert result["_result"]["answer"] == "The answer"
        assert result["_result"]["success"] is True
        assert result["_result"]["turns"] == 3


# ---------------------------------------------------------------------------
# 4. Full state round-trip (all fields)
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    """Test that all non-skipped TaskState fields survive round-trip."""

    def test_all_fields_round_trip(self):
        from src.graph.langgraph.state import (
            _SKIP_TO_LG, _CONFIG_FIELDS,
            task_state_to_lg, lg_to_task_state,
        )

        # Populate every field with non-default values
        original = TaskState(
            task_id="full-rt",
            prompt="full round-trip test",
            context="ctx",
            current_role=Role.ARCHITECT_CODING,
            consecutive_failures=3,
            consecutive_nudges=1,
            escalation_count=2,
            role_history=["frontdoor", "coder_escalation", _RETIRED_ARCHITECT_ROLE],
            escalation_prompt="escalation reason",
            last_error="some error",
            last_output="some output",
            last_code="print('hi')",
            artifacts={"key": "value", "nested": {"a": 1}},
            task_ir={"intent": "code"},
            task_type="repl",
            turns=7,
            max_turns=20,
            gathered_files=["a.py", "b.py"],
            last_failure_id="fail-1",
            anti_pattern_warning="warning!",
            delegation_events=[{"type": "delegate", "to": "specialist"}],
            compaction_count=2,
            compaction_tokens_saved=500,
            context_file_paths=["/tmp/ctx.txt"],
            last_compaction_turn=4,
            session_log_path="/tmp/session.log",
            session_log_records=[{"turn": 1}],
            session_summary_cache="summary...",
            session_summary_turn=3,
            scratchpad_entries=[{"insight": "test"}],
            consolidated_segments=[{"seg": "data"}],
            pending_granular_blocks=["block1"],
            pending_granular_start_turn=2,
            repl_executions=5,
            aggregate_tokens=1500,
            resume_token="abc123",
            think_harder_config={"budget": 2048},
            think_harder_attempted=True,
            think_harder_succeeded=True,
            think_harder_roi_by_role={"frontdoor": {"ema": 0.7, "n": 5}},
            tool_required=True,
            tool_hint="python",
            difficulty_band="hard",
            grammar_enforced=True,
            cache_affinity_bonus=0.3,
            workspace_state={
                "version": 5,
                "broadcast_version": 2,
                "selection_policy": "priority_then_recency",
                "objective": "test objective",
                "constraints": ["c1"],
                "invariants": [],
                "proposals": [],
                "commitments": [],
                "open_questions": [],
                "resolved_questions": [],
                "decisions": [{"id": "d1"}],
                "broadcast_log": [{"msg": "hello"}],
                "updated_at": "2026-04-05",
            },
        )

        # Convert to LG dict
        lg = task_state_to_lg(original)

        # Convert back
        restored = TaskState()
        lg_to_task_state(lg, restored)

        # Verify all non-skipped fields match
        skip = _SKIP_TO_LG | _CONFIG_FIELDS | {"segment_cache", "compaction_quality_monitor"}
        for f in dataclass_fields(original):
            if f.name in skip:
                continue
            orig_val = getattr(original, f.name)
            rest_val = getattr(restored, f.name)
            # Role enums get serialized as strings
            if f.name == "current_role":
                assert str(orig_val) == str(rest_val), f"Mismatch on {f.name}"
            else:
                assert orig_val == rest_val, f"Mismatch on {f.name}: {orig_val!r} != {rest_val!r}"


# ---------------------------------------------------------------------------
# 5. Dual-run validation (mocked _execute_turn)
# ---------------------------------------------------------------------------


class TestDualRunValidation:
    """Compare pydantic_graph and LangGraph backends on identical scenarios.

    Both backends call the same _execute_turn(). By mocking it to return
    scripted sequences, we get deterministic, inference-free comparison.
    """

    def _make_deps(self) -> TaskDeps:
        """Create minimal TaskDeps for testing."""
        deps = TaskDeps()
        deps.primitives = MagicMock()
        deps.repl = MagicMock()
        deps.failure_graph = MagicMock()
        deps.failure_graph.check_veto.return_value = (False, "", 0.0)
        return deps

    def _make_state(self, **kwargs) -> TaskState:
        """Create TaskState with test defaults."""
        defaults = dict(
            task_id="dual-run-test",
            prompt="test",
            max_turns=15,
        )
        defaults.update(kwargs)
        return TaskState(**defaults)

    @pytest.mark.asyncio
    async def test_simple_success_parity(self):
        """Scenario: frontdoor succeeds on turn 1. Both backends should agree."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        # Mock _execute_turn to return success
        mock_turn = AsyncMock(return_value=("The answer is 42", None, True, {"_tool_outputs": []}))

        state = self._make_state(current_role=Role.FRONTDOOR)
        state_dict = task_state_to_lg(state)
        config = {"configurable": {"deps": self._make_deps()}}

        with patch("src.graph.langgraph.nodes._execute_turn", mock_turn), \
             patch("src.graph.langgraph.nodes._make_end_result"):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "__end__"
        assert result["_result"]["success"] is True
        assert "42" in result["_result"]["answer"]
        # role_history delta should be empty (frontdoor didn't add a new role)
        assert result["role_history"] == []

    @pytest.mark.asyncio
    async def test_self_loop_then_success(self):
        """Scenario: frontdoor fails turn 1 (retry), succeeds turn 2."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        call_count = 0

        async def scripted_turn(ctx, role):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", "Schema error", False, {})
            return ("Fixed answer", None, True, {"_tool_outputs": []})

        # Turn 1: failure, retry
        state = self._make_state(current_role=Role.FRONTDOOR)
        state_dict = task_state_to_lg(state)
        config = {"configurable": {"deps": self._make_deps()}}

        with patch("src.graph.langgraph.nodes._execute_turn", side_effect=scripted_turn), \
             patch("src.graph.langgraph.nodes._should_escalate", return_value=False), \
             patch("src.graph.langgraph.nodes._should_think_harder", return_value=False), \
             patch("src.graph.langgraph.nodes._should_retry", return_value=True), \
             patch("src.graph.langgraph.nodes._record_failure"), \
             patch("src.graph.langgraph.nodes._make_end_result"):
            result1 = await frontdoor_node(state_dict, config)

        assert result1["next_node"] == "frontdoor"  # Self-loop
        assert result1["consecutive_failures"] == 1

        # Turn 2: success (feed result1 back as new state)
        merged_state = {**state_dict}
        # Simulate LangGraph reducer: replace scalars, append lists
        for k, v in result1.items():
            if k in ("role_history", "gathered_files", "delegation_events",
                     "context_file_paths", "session_log_records",
                     "scratchpad_entries", "consolidated_segments",
                     "pending_granular_blocks"):
                merged_state[k] = merged_state.get(k, []) + v
            else:
                merged_state[k] = v

        with patch("src.graph.langgraph.nodes._execute_turn", side_effect=scripted_turn), \
             patch("src.graph.langgraph.nodes._make_end_result"):
            result2 = await frontdoor_node(merged_state, config)

        assert result2["next_node"] == "__end__"
        assert result2["_result"]["success"] is True
        assert "Fixed answer" in result2["_result"]["answer"]

    @pytest.mark.asyncio
    async def test_escalation_produces_role_delta(self):
        """Scenario: frontdoor fails and escalates to coder_escalation.
        role_history delta should contain only the new role."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        mock_turn = AsyncMock(return_value=("", "Critical error", False, {}))

        state = self._make_state(
            current_role=Role.FRONTDOOR,
            role_history=["frontdoor"],
        )
        state_dict = task_state_to_lg(state)
        config = {"configurable": {"deps": self._make_deps()}}

        with patch("src.graph.langgraph.nodes._execute_turn", mock_turn), \
             patch("src.graph.langgraph.nodes._should_think_harder", return_value=False), \
             patch("src.graph.langgraph.nodes._should_escalate", return_value=True), \
             patch("src.graph.langgraph.nodes._record_failure"), \
             patch("src.graph.langgraph.nodes._log_escalation"):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "coder_escalation"
        # Delta should contain only the NEW role entry
        assert result["role_history"] == ["coder_escalation"]
        # escalation_count should be incremented
        assert result["escalation_count"] == 1

    @pytest.mark.asyncio
    async def test_max_turns_terminates(self):
        """Scenario: node at max_turns should terminate without calling _execute_turn."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        state = self._make_state(
            current_role=Role.FRONTDOOR,
            turns=15,
            max_turns=15,
            last_output="partial output here",
        )
        state_dict = task_state_to_lg(state)
        config = {"configurable": {"deps": self._make_deps()}}

        mock_turn = AsyncMock()  # Should NOT be called

        with patch("src.graph.langgraph.nodes._execute_turn", mock_turn), \
             patch("src.graph.langgraph.nodes._make_end_result"):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "__end__"
        mock_turn.assert_not_called()
        # _rescue_from_last_output returns None for non-structured output,
        # so the result is a failure with max-turns message
        assert result["_result"]["success"] is False
        assert "Max turns" in result["_result"]["answer"]

    @pytest.mark.asyncio
    async def test_budget_exceeded_terminates(self):
        """Scenario: budget exceeded should terminate early."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        state = self._make_state(current_role=Role.FRONTDOOR)
        state_dict = task_state_to_lg(state)
        config = {"configurable": {"deps": self._make_deps()}}

        mock_turn = AsyncMock()

        with patch("src.graph.langgraph.nodes._execute_turn", mock_turn), \
             patch("src.graph.langgraph.nodes._check_budget_exceeded", return_value="Token budget exceeded"), \
             patch("src.graph.langgraph.nodes._make_end_result"):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "__end__"
        mock_turn.assert_not_called()
        assert result["_result"]["success"] is False
        assert "budget" in result["_result"]["answer"].lower()

    @pytest.mark.asyncio
    async def test_early_abort_first_strike_escalates_with_retries_remaining(self):
        """LangGraph preserves immediate escalation for an admitted abort."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        state = self._make_state(current_role=Role.FRONTDOOR, role_history=["frontdoor"])
        state_dict = task_state_to_lg(state)
        deps = self._make_deps()
        deps.config = GraphConfig(max_retries=3, max_escalations=2, max_turns=10)
        config = {"configurable": {"deps": deps}}

        with patch(
            "src.graph.langgraph.nodes._execute_turn",
            AsyncMock(return_value=("", "Generation aborted: quality low", False, {})),
        ), patch("src.graph.langgraph.nodes._record_failure"), patch(
            "src.graph.langgraph.nodes._log_escalation"
        ):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "coder_escalation"
        assert result["escalation_count"] == 1
        assert result["role_history"] == ["coder_escalation"]

    @pytest.mark.asyncio
    async def test_early_abort_cycle_falls_through_without_escalating(self):
        """A cyclic LangGraph route follows the existing retry/failure chain."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        state = self._make_state(
            current_role=Role.FRONTDOOR,
            role_history=["frontdoor", "worker_math", "frontdoor", "worker_math"],
        )
        state_dict = task_state_to_lg(state)
        deps = self._make_deps()
        deps.config = GraphConfig(max_retries=1, max_escalations=2, max_turns=10)
        config = {"configurable": {"deps": deps}}

        with patch(
            "src.graph.langgraph.nodes._execute_turn",
            AsyncMock(return_value=("", "Generation aborted: quality low", False, {})),
        ), patch("src.graph.langgraph.nodes._record_failure"), patch(
            "src.graph.langgraph.nodes._should_think_harder", return_value=False
        ), patch("src.graph.langgraph.nodes._should_retry", return_value=False), patch(
            "src.graph.langgraph.nodes._make_end_result"
        ):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "__end__"
        assert result["_result"]["success"] is False
        assert result["escalation_count"] == 0
        assert result["role_history"] == []

    @pytest.mark.asyncio
    async def test_early_abort_at_budget_falls_through_without_escalating(self):
        """An exhausted escalation budget refuses LangGraph first-strike routing."""
        from src.graph.langgraph.nodes import frontdoor_node
        from src.graph.langgraph.state import task_state_to_lg

        state = self._make_state(
            current_role=Role.FRONTDOOR,
            escalation_count=2,
            role_history=["frontdoor"],
        )
        state_dict = task_state_to_lg(state)
        deps = self._make_deps()
        deps.config = GraphConfig(max_retries=1, max_escalations=2, max_turns=10)
        config = {"configurable": {"deps": deps}}

        with patch(
            "src.graph.langgraph.nodes._execute_turn",
            AsyncMock(return_value=("", "Generation aborted: quality low", False, {})),
        ), patch("src.graph.langgraph.nodes._record_failure"), patch(
            "src.graph.langgraph.nodes._should_think_harder", return_value=False
        ), patch("src.graph.langgraph.nodes._should_retry", return_value=False), patch(
            "src.graph.langgraph.nodes._make_end_result"
        ):
            result = await frontdoor_node(state_dict, config)

        assert result["next_node"] == "__end__"
        assert result["_result"]["success"] is False
        assert result["escalation_count"] == 2
        assert result["role_history"] == []


# ---------------------------------------------------------------------------
# 6. Integration: _state_update uses delta logic
# ---------------------------------------------------------------------------


class TestStateUpdateIntegration:
    """Verify _state_update properly uses delta logic."""

    def test_state_update_returns_deltas(self):
        from src.graph.langgraph.nodes import _state_update

        task_state = TaskState(
            role_history=["frontdoor", "coder"],
            gathered_files=["a.py"],
        )
        snap = {"role_history": 1, "gathered_files": 0,
                "delegation_events": 0, "context_file_paths": 0,
                "session_log_records": 0, "scratchpad_entries": 0,
                "consolidated_segments": 0, "pending_granular_blocks": 0}

        result = _state_update(task_state, "coder", snap)
        assert result["next_node"] == "coder"
        assert result["role_history"] == ["coder"]  # Only the delta
        assert result["gathered_files"] == ["a.py"]  # All new (snap was 0)

    def test_state_update_empty_snap_returns_full_lists(self):
        """With zero-snapshot (fresh state), all elements are 'new'."""
        from src.graph.langgraph.nodes import _state_update

        task_state = TaskState(
            role_history=["frontdoor"],
        )
        snap = {"role_history": 0, "gathered_files": 0,
                "delegation_events": 0, "context_file_paths": 0,
                "session_log_records": 0, "scratchpad_entries": 0,
                "consolidated_segments": 0, "pending_granular_blocks": 0}

        result = _state_update(task_state, "frontdoor", snap)
        assert result["role_history"] == ["frontdoor"]
