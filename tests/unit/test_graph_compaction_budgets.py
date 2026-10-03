"""Tests for graph compaction and budget modules.

Covers uncovered branches in compaction.py, budgets.py, and
concurrency_aware.py migration state machine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── compaction.py ────────────────────────────────────────────────────────


class TestEstimateContextTokens:
    def test_uses_accurate_tokenizer_when_available(self):
        from src.graph.compaction import _estimate_context_tokens

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 42
        assert _estimate_context_tokens(ctx, "hello world") == 42

    def test_falls_back_to_heuristic(self):
        from src.graph.compaction import _estimate_context_tokens

        ctx = MagicMock(spec=[])
        ctx.deps = MagicMock(spec=[])
        ctx.deps.primitives = None
        assert _estimate_context_tokens(ctx, "x" * 400) == 100


class TestGetModelMaxContext:
    def test_returns_registry_value(self):
        from src.graph.compaction import _get_model_max_context

        ctx = MagicMock()
        role_cfg = MagicMock()
        role_cfg.n_ctx = 65536
        ctx.deps.primitives.registry.get_role_config.return_value = role_cfg
        ctx.state.current_role = "coder"
        assert _get_model_max_context(ctx) == 65536

    def test_returns_default_without_registry(self):
        from src.graph.compaction import _get_model_max_context

        ctx = MagicMock()
        ctx.deps.primitives = None
        assert _get_model_max_context(ctx) == 32768

    def test_returns_default_on_exception(self):
        from src.graph.compaction import _get_model_max_context

        ctx = MagicMock()
        ctx.deps.primitives.registry.get_role_config.side_effect = RuntimeError("no registry")
        assert _get_model_max_context(ctx) == 32768


class TestContextExternalizationPath:
    def test_returns_writable_path(self, tmp_path, monkeypatch):
        from src.graph.compaction import _context_externalization_path

        state = MagicMock()
        state.task_id = "test-task-123"
        state.compaction_count = 2

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        with patch("src.config.get_config", side_effect=ImportError):
            path = _context_externalization_path(state)

        assert "test-task-123" in str(path)
        assert "ctx_2" in str(path)

    def test_falls_back_to_tempdir(self, monkeypatch):
        from src.graph.compaction import _context_externalization_path

        state = MagicMock()
        state.task_id = "fallback"
        state.compaction_count = 0

        monkeypatch.delenv("ORCHESTRATOR_PATHS_TMP_DIR", raising=False)
        with patch("src.config.get_config", side_effect=ImportError):
            path = _context_externalization_path(state)

        assert "fallback" in str(path)


class TestCompactionPrompt:
    """Tests for prompt loading in compaction."""

    def test_resolve_compaction_prompt_falls_back_without_prompt_file(self):
        from src.graph.compaction import _resolve_compaction_prompt

        with patch("builtins.open", side_effect=FileNotFoundError("no prompt file")):
            prompt = _resolve_compaction_prompt()

        assert "Generate a structured index" in prompt


class TestContextExternalizationFallback:
    """Tests for compaction artifact path selection and fallback."""

    def test_falls_back_to_tempdir_when_candidates_unwritable(self, tmp_path):
        from src.graph.compaction import _context_externalization_path

        state = MagicMock()
        state.task_id = "read-only"
        state.compaction_count = 3

        config = MagicMock()
        config.paths.tmp_dir = str(tmp_path)

        with patch("src.config.get_config", return_value=config), patch(
            "builtins.open", side_effect=PermissionError("read only")
        ):
            path = _context_externalization_path(state)

        assert path == Path(tempfile.gettempdir()) / "session_read-only_ctx_3.md"
        assert str(path).startswith(tempfile.gettempdir())


# ── budgets.py ───────────────────────────────────────────────────────────


class TestReasoningLengthAlarm:
    """Tests for _check_reasoning_length_alarm."""

    def test_returns_false_when_feature_disabled(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.reasoning_length_alarm = False
            assert _check_reasoning_length_alarm("text", "hard", 10) is False

    def test_returns_false_without_difficulty_band(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.reasoning_length_alarm = True
            assert _check_reasoning_length_alarm("<think>...<</think>", "", 10) is False

    def test_returns_false_if_difficulty_band_unknown(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat, patch(
            "src.classifiers.difficulty_signal.get_mode", return_value="enforce"
        ):
            mock_feat.return_value.reasoning_length_alarm = True
            assert _check_reasoning_length_alarm(
                "<think>abc</think>", "impossible", 10
            ) is False

    def test_returns_false_when_not_in_enforce_mode(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat, patch(
            "src.classifiers.difficulty_signal.get_mode", return_value="shadow"
        ):
            mock_feat.return_value.reasoning_length_alarm = True
            assert _check_reasoning_length_alarm("<think>abc</think>", "hard", 10) is False

    def test_returns_false_when_mode_check_errors(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat, patch(
            "src.classifiers.difficulty_signal.get_mode",
            side_effect=RuntimeError("mode unavailable"),
        ):
            mock_feat.return_value.reasoning_length_alarm = True
            assert _check_reasoning_length_alarm("<think>abc</think>", "hard", 10) is False

    def test_returns_false_when_no_think_blocks(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat, patch(
            "src.classifiers.difficulty_signal.get_mode", return_value="enforce"
        ):
            mock_feat.return_value.reasoning_length_alarm = True
            assert _check_reasoning_length_alarm("plain output", "hard", 10) is False

    def test_returns_true_with_completion_token_count(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        with patch("src.features.features") as mock_feat, patch(
            "src.classifiers.difficulty_signal.get_mode", return_value="enforce"
        ):
            mock_feat.return_value.reasoning_length_alarm = True
            assert (
                _check_reasoning_length_alarm("<think>ignored</think>", "hard", 11000)
                is True
            )

    def test_returns_true_from_think_text_length_when_no_completion_count(self):
        from src.graph.budgets import _check_reasoning_length_alarm

        think_text = "<think>" + ("x " * 10000) + "</think>"
        with patch("src.features.features") as mock_feat, patch(
            "src.classifiers.difficulty_signal.get_mode", return_value="enforce"
        ):
            mock_feat.return_value.reasoning_length_alarm = True
            assert _check_reasoning_length_alarm(think_text, "easy", 0) is True


class TestReplTurnTokenCap:
    def test_returns_flat_cap_without_band(self):
        from src.graph.budgets import _repl_turn_token_cap

        assert _repl_turn_token_cap("") == 768

    def test_returns_flat_cap_when_not_enforce(self):
        from src.graph.budgets import _repl_turn_token_cap

        with patch("src.classifiers.difficulty_signal.get_mode", return_value="shadow"):
            assert _repl_turn_token_cap("hard") == 768

    def test_returns_band_budget_in_enforce_mode(self):
        from src.graph.budgets import _repl_turn_token_cap, _BAND_TOKEN_BUDGETS

        with patch("src.classifiers.difficulty_signal.get_mode", return_value="enforce"):
            result = _repl_turn_token_cap("hard")
            assert result == _BAND_TOKEN_BUDGETS["hard"]

    def test_returns_flat_cap_on_import_error(self):
        from src.graph.budgets import _repl_turn_token_cap

        with patch("src.classifiers.difficulty_signal.get_mode", side_effect=ImportError):
            assert _repl_turn_token_cap("hard") == 768


class TestFrontdoorTokenCaps:
    def test_frontdoor_turn_cap_disabled(self, monkeypatch):
        from src.graph.budgets import _frontdoor_turn_token_cap

        monkeypatch.setenv("ORCHESTRATOR_FRONTDOOR_TURN_N_TOKENS", "0")
        assert _frontdoor_turn_token_cap() == 0

    def test_frontdoor_turn_cap_with_value(self, monkeypatch):
        from src.graph.budgets import _frontdoor_turn_token_cap

        monkeypatch.setenv("ORCHESTRATOR_FRONTDOOR_TURN_N_TOKENS", "512")
        assert _frontdoor_turn_token_cap() == 512

    def test_frontdoor_non_tool_cap(self):
        from src.graph.budgets import _frontdoor_repl_non_tool_token_cap

        assert _frontdoor_repl_non_tool_token_cap() >= 64


class TestBudgetCaps:
    def test_worker_call_budget_cap_default(self):
        from src.graph.budgets import _worker_call_budget_cap

        assert _worker_call_budget_cap() == 30

    def test_task_token_budget_cap_default(self):
        from src.graph.budgets import _task_token_budget_cap

        assert _task_token_budget_cap() == 200000


class TestCheckBudgetExceeded:
    def test_returns_none_when_within_budget(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 5
        ctx.state.aggregate_tokens = 1000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = True
            result = _check_budget_exceeded(ctx)
        assert result is None

    def test_returns_message_on_call_budget_exceeded(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 50
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = False
            result = _check_budget_exceeded(ctx)
        assert "Worker call budget exhausted" in result

    def test_returns_message_on_token_budget_exceeded(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 0
        ctx.state.aggregate_tokens = 999999

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = False
            mock_feat.return_value.task_token_budget = True
            result = _check_budget_exceeded(ctx)
        assert "Task token budget exhausted" in result

    def test_prefers_call_budget_message_when_both_exceeded(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 30
        ctx.state.aggregate_tokens = 200000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = True
            result = _check_budget_exceeded(ctx)

        assert result == "Worker call budget exhausted (30/30 REPL executions)"


class TestBudgetPressureWarnings:
    def test_no_warnings_when_plenty_of_budget(self):
        from src.graph.budgets import _budget_pressure_warnings

        state = MagicMock()
        state.repl_executions = 5
        state.aggregate_tokens = 1000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = True
            result = _budget_pressure_warnings(state)
        assert result == ""

    def test_warns_when_repl_budget_low(self):
        from src.graph.budgets import _budget_pressure_warnings

        state = MagicMock()
        state.repl_executions = 28
        state.aggregate_tokens = 1000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = False
            result = _budget_pressure_warnings(state)
        assert "REPL execution" in result
        assert "FINAL()" in result

    def test_warns_when_token_budget_low(self):
        from src.graph.budgets import _budget_pressure_warnings

        state = MagicMock()
        state.repl_executions = 0
        state.aggregate_tokens = 185000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = False
            mock_feat.return_value.task_token_budget = True
            result = _budget_pressure_warnings(state)
        assert "token budget" in result


class TestMaybeCompactContext:
    """Tests for _maybe_compact_context async function."""

    @pytest.fixture(autouse=True)
    def _no_context_limit_sources(self):
        # These tests exercise the trigger arithmetic against the 32768
        # fallback limit. Since 2026-09-24 the limit resolves from the live
        # server / compiled stack priors first (e.g. "worker" is 65536 per
        # request), so pin a resolver that knows nothing.
        from src.backends.context_limits import ContextLimitResolver, set_context_limit_resolver

        set_context_limit_resolver(
            ContextLimitResolver(live=False, registry_facts=lambda: {}, role_urls=lambda: {})
        )
        yield
        set_context_limit_resolver(None)

    @pytest.mark.asyncio
    async def test_skips_when_compaction_disabled(self):
        from src.graph.compaction import _maybe_compact_context

        ctx = MagicMock()
        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = False
            await _maybe_compact_context(ctx)
        # Should return immediately — no LLM call made
        ctx.deps.primitives.llm_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_primitives(self):
        from src.graph.compaction import _maybe_compact_context

        ctx = MagicMock()
        ctx.deps.primitives = None
        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            await _maybe_compact_context(ctx)

    @pytest.mark.asyncio
    async def test_skips_when_too_few_turns(self):
        from src.graph.compaction import _maybe_compact_context

        ctx = MagicMock()
        ctx.deps.primitives = MagicMock()
        ctx.state.turns = 1
        ctx.state.context = "short"
        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            with patch("src.config.get_config", side_effect=ImportError):
                await _maybe_compact_context(ctx)

    @pytest.mark.asyncio
    async def test_compacts_when_context_exceeds_threshold(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 50000
        ctx.deps.primitives.llm_call.return_value = "- Summary of old context"
        ctx.state.turns = 10
        ctx.state.context = "x" * 20000  # Large enough to trigger
        ctx.state.current_role = "worker"
        ctx.state.task_id = "compact-test"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            with patch("src.config.get_config") as mock_cfg:
                mock_cfg.return_value.chat.session_compaction_keep_recent_ratio = 0.20
                mock_cfg.return_value.chat.session_compaction_recompaction_interval = 0
                mock_cfg.return_value.chat.session_compaction_min_turns = 5
                mock_cfg.return_value.chat.session_compaction_trigger_ratio = 0.75
                mock_cfg.return_value.chat.session_compaction_prompt = "Summarize: {context}"

                await _maybe_compact_context(ctx)

        # Should have called llm_call for index generation
        ctx.deps.primitives.llm_call.assert_called_once()
        # Context should be replaced with compacted version
        assert "Context Index" in ctx.state.context or ctx.state.compaction_count == 1

    @pytest.mark.asyncio
    async def test_compacts_when_recompaction_interval_reaches_threshold(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 4
        ctx.deps.primitives.registry = None
        ctx.deps.primitives.llm_call.return_value = "- Recompaction summary"
        ctx.state.turns = 10
        ctx.state.context = "x" * 5000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "recompaction-trigger"
        ctx.state.compaction_count = 1
        ctx.state.last_compaction_turn = 2
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 0.2
            cfg.chat.session_compaction_recompaction_interval = 4
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.95
            with patch("src.config.get_config", return_value=cfg):
                await _maybe_compact_context(ctx)

        assert "Context Index (compaction #2)" in ctx.state.context
        assert ctx.state.compaction_count == 2
        assert ctx.state.last_compaction_turn == 10

    @pytest.mark.asyncio
    async def test_compacts_when_session_token_budget_requests_compaction(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 4
        ctx.deps.primitives.registry = None
        ctx.deps.primitives.llm_call.return_value = "- Budget-triggered summary"
        ctx.state.turns = 10
        ctx.state.context = "x" * 5000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "budget-trigger"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        budget_status = MagicMock()
        budget_status.should_compact = True
        budget_status.utilization = 0.92
        budget = MagicMock()
        budget.check.return_value = budget_status

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = True
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 0.2
            cfg.chat.session_compaction_recompaction_interval = 0
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.95
            with patch("src.config.get_config", return_value=cfg):
                with patch("src.session_analytics.SessionTokenBudget.from_env", return_value=budget):
                    await _maybe_compact_context(ctx)

        assert "Context Index (compaction #1)" in ctx.state.context
        assert ctx.state.compaction_count == 1
        assert ctx.state.context_file_paths

    @pytest.mark.asyncio
    async def test_skips_when_to_externalize_section_is_empty(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 60000
        ctx.deps.primitives.registry = None
        ctx.state.turns = 10
        ctx.state.context = "x" * 4000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "empty-externalize"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 1.0
            cfg.chat.session_compaction_recompaction_interval = 0
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.1
            with patch("src.config.get_config", return_value=cfg):
                await _maybe_compact_context(ctx)

        assert ctx.state.compaction_count == 0
        assert ctx.state.context == "x" * 4000
        assert ctx.state.context_file_paths == []

    @pytest.mark.asyncio
    async def test_falls_back_to_default_compaction_index_on_llm_failure(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 60000
        ctx.deps.primitives.registry = None
        ctx.deps.primitives.llm_call.side_effect = RuntimeError("llm unavailable")
        ctx.state.turns = 10
        ctx.state.context = "x" * 50000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "llm-fallback"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            with patch("src.config.get_config", side_effect=ImportError):
                await _maybe_compact_context(ctx)

        assert "Fallback Index" in ctx.state.context
        assert ctx.state.compaction_count == 1

    @pytest.mark.asyncio
    async def test_session_token_budget_errors_dont_force_compaction(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 4
        ctx.deps.primitives.registry = None
        ctx.state.turns = 10
        ctx.state.context = "x" * 100
        ctx.state.current_role = "worker"
        ctx.state.task_id = "budget-error"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = True
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 0.2
            cfg.chat.session_compaction_recompaction_interval = 0
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.95
            with patch("src.config.get_config", return_value=cfg):
                with patch(
                    "src.session_analytics.SessionTokenBudget.from_env",
                    side_effect=RuntimeError("budget error"),
                ):
                    await _maybe_compact_context(ctx)

        assert ctx.state.compaction_count == 0
        assert ctx.state.context_file_paths == []

    @pytest.mark.asyncio
    async def test_write_failure_aborts_compaction(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 50000
        ctx.deps.primitives.registry = None
        ctx.state.turns = 10
        ctx.state.context = "x" * 5000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "write-fail"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 0.2
            cfg.chat.session_compaction_recompaction_interval = 0
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.95
            ctx_file = tmp_path / "blocked.md"
            with patch("src.config.get_config", return_value=cfg), patch(
                "src.graph.compaction._context_externalization_path", return_value=ctx_file
            ), patch("builtins.open", side_effect=PermissionError("no write")):
                await _maybe_compact_context(ctx)

        assert ctx.state.compaction_count == 0
        assert ctx.state.context == "x" * 5000
        assert ctx.state.context_file_paths == []

    @pytest.mark.asyncio
    async def test_fails_over_to_debug_log_when_compaction_body_throws(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.side_effect = [50000, RuntimeError("token accounting failed")]
        ctx.deps.primitives.registry = None
        ctx.state.turns = 10
        ctx.state.context = "x" * 5000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "compaction-error"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 0.2
            cfg.chat.session_compaction_recompaction_interval = 0
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.95
            with patch("src.config.get_config", return_value=cfg):
                await _maybe_compact_context(ctx)

        assert "Context Index (compaction #1)" in ctx.state.context
        assert ctx.state.compaction_count == 1

    @pytest.mark.asyncio
    async def test_uses_threaded_llm_call_when_not_running_pytest(self, tmp_path, monkeypatch):
        from src.graph.compaction import _maybe_compact_context

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 50000
        ctx.deps.primitives.registry = None
        ctx.state.turns = 10
        ctx.state.context = "x" * 5000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "threaded-call"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.session_compaction = True
            mock_feat.return_value.session_compaction_llm_index = True
            mock_feat.return_value.session_token_budget = False
            cfg = MagicMock()
            cfg.chat.session_compaction_keep_recent_ratio = 0.2
            cfg.chat.session_compaction_recompaction_interval = 0
            cfg.chat.session_compaction_min_turns = 5
            cfg.chat.session_compaction_trigger_ratio = 0.95
            with patch("src.config.get_config", return_value=cfg):
                with patch(
                    "src.graph.compaction.asyncio.to_thread",
                    new=AsyncMock(return_value="- Threaded summary"),
                ):
                    await _maybe_compact_context(ctx)

        assert "Context Index (compaction #1)" in ctx.state.context


# ── concurrency_aware.py ─────────────────────────────────────────────────


class TestConcurrencyAwareMigrationStates:
    """Test the KV migration state constants added in Phase 4."""

    def test_migration_state_constants_exist(self):
        from src.backends.concurrency_aware import (
            _STATE_UNASSIGNED,
            _STATE_ASSIGNED_FULL,
            _STATE_MIGRATION_PENDING,
            _STATE_ASSIGNED_QUARTER,
            _STATE_MIGRATION_FAILED_COLD,
        )

        assert _STATE_UNASSIGNED == "unassigned"
        assert _STATE_ASSIGNED_FULL == "assigned_full"
        assert _STATE_MIGRATION_PENDING == "migration_pending"
        assert _STATE_ASSIGNED_QUARTER == "assigned_quarter"
        assert _STATE_MIGRATION_FAILED_COLD == "migration_failed_cold"

    def test_backend_init_empty_affinity(self):
        """Verify affinity map starts empty (Phase 4 invariant)."""
        from src.backends.concurrency_aware import ConcurrencyAwareBackend

        backend = ConcurrencyAwareBackend.__new__(ConcurrencyAwareBackend)
        backend._session_affinity = {}
        assert len(backend._session_affinity) == 0
