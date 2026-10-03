"""Tests for context window management: tool result clearing (C3) and compaction (C1)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

from src.graph.state import TaskState, TaskDeps, GraphConfig
from src.graph.helpers import (
    _clear_stale_tool_outputs,
    _maybe_compact_context,
    _resolve_compaction_prompt,
)


# ---------------------------------------------------------------------------
# C3: Tool Result Clearing
# ---------------------------------------------------------------------------

def _make_tool_block(content: str) -> str:
    return f"<<<TOOL_OUTPUT>>>{content}<<<END_TOOL_OUTPUT>>>"


def _make_state(**kwargs) -> TaskState:
    return TaskState(**kwargs)


class TestClearStaleToolOutputs:
    """Tests for _clear_stale_tool_outputs (C3)."""

    @patch("src.graph.helpers._get_features")
    def _enable_flag(self, mock_features):
        """Helper context — prefer direct patching in each test."""
        pass

    def test_no_clearing_when_flag_disabled(self):
        """Does nothing when tool_result_clearing=False."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=False))
        try:
            state = _make_state(
                last_output=_make_tool_block("old1") + _make_tool_block("old2"),
                context="x" * 20000,
            )
            freed = _clear_stale_tool_outputs(state)
            assert freed == 0
            assert "<<<TOOL_OUTPUT>>>" in state.last_output
        finally:
            reset_features()

    def test_clears_old_blocks_keeps_recent(self):
        """Old blocks replaced, recent blocks preserved."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            blocks = [_make_tool_block(f"content_{i}") for i in range(5)]
            state = _make_state(
                last_output="prefix\n" + "\n".join(blocks) + "\nsuffix",
                context="x" * 20000,
            )
            freed = _clear_stale_tool_outputs(state, keep_recent=2)
            assert freed > 0
            # Last 2 blocks should be preserved
            assert "content_3" in state.last_output
            assert "content_4" in state.last_output
            # First 3 should be cleared
            assert "content_0" not in state.last_output
            assert "content_1" not in state.last_output
            assert "content_2" not in state.last_output
            assert state.last_output.count("[Tool result cleared]") == 3
        finally:
            reset_features()

    def test_no_clearing_when_few_blocks(self):
        """Does not clear when blocks <= keep_recent."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            blocks = [_make_tool_block(f"content_{i}") for i in range(2)]
            state = _make_state(
                last_output="\n".join(blocks),
                context="x" * 20000,
            )
            freed = _clear_stale_tool_outputs(state, keep_recent=2)
            assert freed == 0
            assert "content_0" in state.last_output
            assert "content_1" in state.last_output
        finally:
            reset_features()

    def test_no_clearing_when_context_small(self):
        """Does not fire when context is below threshold."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            blocks = [_make_tool_block(f"content_{i}") for i in range(5)]
            state = _make_state(
                last_output="\n".join(blocks),
                context="x" * 100,  # Small context
            )
            freed = _clear_stale_tool_outputs(state)
            assert freed == 0
        finally:
            reset_features()

    def test_clears_all_when_keep_recent_zero(self):
        """keep_recent=0 clears all blocks."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            blocks = [_make_tool_block(f"content_{i}") for i in range(3)]
            state = _make_state(
                last_output="\n".join(blocks),
                context="x" * 20000,
            )
            freed = _clear_stale_tool_outputs(state, keep_recent=0)
            assert freed > 0
            assert "<<<TOOL_OUTPUT>>>" not in state.last_output
            assert state.last_output.count("[Tool result cleared]") == 3
        finally:
            reset_features()

    def test_empty_last_output(self):
        """Gracefully handles empty last_output."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            state = _make_state(last_output="", context="x" * 20000)
            freed = _clear_stale_tool_outputs(state)
            assert freed == 0
        finally:
            reset_features()

    def test_preserves_surrounding_text(self):
        """Text before/after/between tool blocks is preserved."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            state = _make_state(
                last_output=(
                    "before\n"
                    + _make_tool_block("old")
                    + "\nmiddle\n"
                    + _make_tool_block("new")
                    + "\nafter"
                ),
                context="x" * 20000,
            )
            _clear_stale_tool_outputs(state, keep_recent=1)
            assert "before" in state.last_output
            assert "middle" in state.last_output
            assert "after" in state.last_output
            assert "new" in state.last_output
            assert "old" not in state.last_output
        finally:
            reset_features()

    def test_max_context_tokens_gate(self):
        """Uses token-based gating when max_context_tokens is set."""
        from src.features import Features, set_features, reset_features

        set_features(Features(tool_result_clearing=True))
        try:
            blocks = [_make_tool_block(f"content_{i}") for i in range(5)]
            # Context is 100 chars ≈ 25 tokens; max_context=1000; 25/1000=0.025 < 0.4
            state = _make_state(
                last_output="\n".join(blocks),
                context="x" * 100,
            )
            freed = _clear_stale_tool_outputs(
                state, max_context_tokens=1000, context_ratio_trigger=0.4
            )
            assert freed == 0

            # Context is 10000 chars ≈ 2500 tokens; 2500/4000=0.625 > 0.4
            state.context = "x" * 10000
            freed = _clear_stale_tool_outputs(
                state, max_context_tokens=4000, context_ratio_trigger=0.4
            )
            assert freed > 0
        finally:
            reset_features()


# ---------------------------------------------------------------------------
# C1: Enhanced Conversation Compactor
# ---------------------------------------------------------------------------


class TestResolveCompactionPrompt:
    """Test compaction prompt loading."""

    def test_loads_from_file(self):
        prompt = _resolve_compaction_prompt()
        assert "structured index" in prompt.lower()
        assert "line" in prompt.lower()  # Should mention line coordinates

    def test_fallback_when_file_missing(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            prompt = _resolve_compaction_prompt()
            assert "structured index" in prompt.lower()


def _make_ctx(state, primitives=None):
    """Build a mock GraphRunContext for _maybe_compact_context."""
    if primitives is not None:
        # Prevent _get_model_max_context from traversing MagicMock attributes
        # (int(MagicMock().n_ctx) returns 1, making trigger threshold 0)
        primitives.registry = None
    deps = TaskDeps(primitives=primitives, config=GraphConfig())
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = deps
    return ctx


class TestMaybeCompactContext:
    """Tests for _maybe_compact_context (C1 enhanced)."""

    def test_no_compaction_when_flag_disabled(self):
        """Does nothing when session_compaction=False."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=False))
        try:
            state = _make_state(
                turns=10,
                context="x" * 50000,
                task_id="test_task",
            )
            primitives = MagicMock()
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)
            ctx = _make_ctx(state, primitives)
            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))
            assert state.compaction_count == 0
        finally:
            reset_features()

    def test_no_compaction_when_turns_low(self):
        """Does not trigger when turns <= 5."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=3,
                context="x" * 50000,
                task_id="test_task",
            )
            primitives = MagicMock()
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)
            ctx = _make_ctx(state, primitives)
            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))
            assert state.compaction_count == 0
        finally:
            reset_features()

    def test_compaction_turn_guard_is_configurable(self):
        """Compaction can run earlier when session_compaction_min_turns is lowered."""
        from src.features import Features, set_features, reset_features
        from src.config import ChatPipelineConfig

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=3,  # below default guard (5)
                context="x" * 50000,
                task_id="test_min_turns_override",
            )
            primitives = MagicMock()
            primitives.llm_call.return_value = "- Index entry"
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)
            ctx = _make_ctx(state, primitives)

            mock_cfg = ChatPipelineConfig(session_compaction_min_turns=1)
            with patch("src.config.get_config") as mock_get_config:
                mock_get_config.return_value.chat = mock_cfg
                asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            assert state.compaction_count == 1

            for p in state.context_file_paths:
                if os.path.exists(p):
                    os.unlink(p)
        finally:
            reset_features()

    def test_no_compaction_when_context_small(self):
        """Does not trigger when context is small."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=10,
                context="x" * 100,
                task_id="test_task",
            )
            primitives = MagicMock()
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)
            ctx = _make_ctx(state, primitives)
            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))
            assert state.compaction_count == 0
        finally:
            reset_features()

    def test_compaction_writes_file_and_replaces_context(self):
        """Full compaction: writes file, generates index, replaces context."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            original_context = "Important discussion about architecture\n" * 1000
            state = _make_state(
                turns=10,
                context=original_context,
                task_id="test_compact",
            )
            primitives = MagicMock()
            primitives.llm_call.return_value = "- **Architecture** (lines 1-50)\n  - Files: src/main.py"
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)

            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            assert state.compaction_count == 1
            assert state.compaction_tokens_saved > 0
            assert len(state.context_file_paths) == 1

            # Context should contain index and read_file pointer
            assert "[Context Index" in state.context
            assert "read_file(" in state.context
            assert "[Recent Context]" in state.context

            # File should exist with original content
            ctx_path = state.context_file_paths[0]
            assert os.path.exists(ctx_path)
            with open(ctx_path) as f:
                saved = f.read()
            assert saved == original_context

            # LLM was called with the canonical worker_general role
            primitives.llm_call.assert_called_once()
            call_kwargs = primitives.llm_call.call_args
            assert call_kwargs[1]["role"] == "worker_general"

            # Clean up
            os.unlink(ctx_path)
        finally:
            reset_features()

    def test_compaction_keeps_recent_context_verbatim(self):
        """Recent ~20% of context is preserved verbatim."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            # Create context with identifiable recent content
            old_part = "OLD_CONTENT\n" * 500
            recent_part = "RECENT_UNIQUE_MARKER\n" * 100
            state = _make_state(
                turns=10,
                context=old_part + recent_part,
                task_id="test_recent",
            )
            primitives = MagicMock()
            primitives.llm_call.return_value = "- Index entry"
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)
            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            assert "RECENT_UNIQUE_MARKER" in state.context

            # Clean up
            for p in state.context_file_paths:
                if os.path.exists(p):
                    os.unlink(p)
        finally:
            reset_features()

    def test_compaction_survives_llm_failure(self):
        """Compaction still proceeds with fallback index if LLM index call fails."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=10,
                context="x" * 50000,
                task_id="test_fail",
            )
            primitives = MagicMock()
            primitives.llm_call.side_effect = RuntimeError("LLM unavailable")
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)
            # Should not raise
            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))
            assert state.compaction_count == 1
            assert "[Fallback Index]" in state.context
            assert "read_file(" in state.context

            for p in state.context_file_paths:
                if os.path.exists(p):
                    os.unlink(p)
        finally:
            reset_features()

    def test_configurable_keep_recent_ratio(self):
        """Configurable keep_recent_ratio controls how much context is kept verbatim."""
        from src.features import Features, set_features, reset_features
        from src.config import ChatPipelineConfig

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            original_context = "A" * 10000 + "B" * 40000  # 50K total
            state = _make_state(
                turns=10,
                context=original_context,
                task_id="test_ratio",
            )
            primitives = MagicMock()
            primitives.llm_call.return_value = "- Index entry"
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)

            # Patch config to use 10% ratio instead of default 20%
            mock_cfg = ChatPipelineConfig(session_compaction_keep_recent_ratio=0.10)
            with patch("src.config.get_config") as mock_get_config:
                mock_get_config.return_value.chat = mock_cfg
                asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            assert state.compaction_count == 1
            # With 10% ratio on 50K context, keep_chars = max(3000, 5000) = 5000
            # So last 5000 chars should be in context (all B's)
            assert "BBBBB" in state.context

            # Clean up
            for p in state.context_file_paths:
                if os.path.exists(p):
                    os.unlink(p)
        finally:
            reset_features()

    def test_recompaction_interval_triggers(self):
        """Recompaction interval triggers compaction based on turns since last."""
        from src.features import Features, set_features, reset_features
        from src.config import ChatPipelineConfig

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            # Context is small (below normal threshold) but recompaction should trigger
            state = _make_state(
                turns=20,
                context="x" * 8000,  # Below 12K char threshold
                task_id="test_recompact",
                compaction_count=1,  # Already compacted once
                last_compaction_turn=5,  # Last compacted at turn 5
            )
            primitives = MagicMock()
            primitives.llm_call.return_value = "- Index entry"
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)

            # Set recompaction interval to 10 turns — turns since last = 15 >= 10
            mock_cfg = ChatPipelineConfig(session_compaction_recompaction_interval=10)
            with patch("src.config.get_config") as mock_get_config:
                mock_get_config.return_value.chat = mock_cfg
                asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            assert state.compaction_count == 2
            assert state.last_compaction_turn == 20

            # Clean up
            for p in state.context_file_paths:
                if os.path.exists(p):
                    os.unlink(p)
        finally:
            reset_features()

    def test_recompaction_interval_no_trigger_when_recent(self):
        """Recompaction does not trigger if not enough turns have passed."""
        from src.features import Features, set_features, reset_features
        from src.config import ChatPipelineConfig

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=12,
                context="x" * 8000,  # Below normal threshold
                task_id="test_no_recompact",
                compaction_count=1,
                last_compaction_turn=10,  # Only 2 turns ago
            )
            primitives = MagicMock()
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)

            mock_cfg = ChatPipelineConfig(session_compaction_recompaction_interval=10)
            with patch("src.config.get_config") as mock_get_config:
                mock_get_config.return_value.chat = mock_cfg
                asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            # Should NOT have compacted (only 2 turns since last, need 10)
            assert state.compaction_count == 1
        finally:
            reset_features()

    def test_recompaction_disabled_when_zero(self):
        """Recompaction interval of 0 means disabled."""
        from src.features import Features, set_features, reset_features
        from src.config import ChatPipelineConfig

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=100,
                context="x" * 8000,  # Below normal threshold
                task_id="test_no_interval",
                compaction_count=1,
                last_compaction_turn=5,  # 95 turns ago
            )
            primitives = MagicMock()
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)

            # interval=0 means disabled
            mock_cfg = ChatPipelineConfig(session_compaction_recompaction_interval=0)
            with patch("src.config.get_config") as mock_get_config:
                mock_get_config.return_value.chat = mock_cfg
                asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            # Should NOT have compacted (interval disabled)
            assert state.compaction_count == 1
        finally:
            reset_features()

    def test_last_compaction_turn_set_after_compaction(self):
        """last_compaction_turn is set to current turn after compaction."""
        from src.features import Features, set_features, reset_features

        set_features(Features(session_compaction=True, session_compaction_llm_index=True))
        try:
            state = _make_state(
                turns=15,
                context="x" * 50000,
                task_id="test_turn_track",
            )
            primitives = MagicMock()
            primitives.llm_call.return_value = "- Index"
            primitives._count_tokens = MagicMock(side_effect=lambda t: len(t) // 4)

            ctx = _make_ctx(state, primitives)
            asyncio.new_event_loop().run_until_complete(_maybe_compact_context(ctx))

            assert state.compaction_count == 1
            assert state.last_compaction_turn == 15

            # Clean up
            for p in state.context_file_paths:
                if os.path.exists(p):
                    os.unlink(p)
        finally:
            reset_features()
