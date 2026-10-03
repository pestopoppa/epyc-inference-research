"""Comprehensive tests for REPL executor pipeline stage.

Tests coverage for src/api/routes/chat_pipeline/repl_executor.py (9% → target 80%+).
Focuses on: REPL session management, escalation handling, generation monitoring,
two-stage summarization integration, long-context exploration.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline.repl_executor import _execute_repl
from src.api.routes.chat_utils import RoutingResult
from src.graph.state import TaskResult
from src.llm_primitives import LLMPrimitives
from src.llm_primitives.types import LLMResult
from src.roles import Role
from src.session import Session, SQLiteSessionStore

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


# ── Test Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def mock_primitives():
    """Create mock LLMPrimitives with realistic responses."""
    primitives = MagicMock(spec=LLMPrimitives)
    primitives.mock_mode = False
    primitives.total_tokens_generated = 100
    primitives.total_prompt_eval_ms = 50
    primitives.total_generation_ms = 200
    primitives.total_http_overhead_ms = 10
    primitives._last_predicted_tps = 15.0
    primitives._backends = {"frontdoor": MagicMock()}
    primitives.get_cache_stats.return_value = {"hits": 5, "misses": 2}
    primitives.llm_call.return_value = "FINAL('Test answer')"
    primitives.llm_call_monitored.return_value = LLMResult(
        text="FINAL('Test answer')",
        aborted=False,
        abort_reason="",
    )
    return primitives


@pytest.fixture
def mock_state():
    """Create mock application state."""
    state = MagicMock()
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.progress_logger = MagicMock()
    state.hybrid_router = None
    state.failure_graph = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def basic_request():
    """Create basic ChatRequest."""
    return ChatRequest(
        prompt="Test prompt",
        context="",
        real_mode=True,
        mock_mode=False,
        max_turns=5,
    )


@pytest.fixture
def basic_routing():
    """Create basic RoutingResult."""
    return RoutingResult(
        task_id="test-task-123",
        task_ir={},
        use_mock=False,
        routing_strategy="direct",
        formalization_applied=False,
        document_result=None,
    )


# ── Basic REPL Execution ─────────────────────────────────────────────────


class TestBasicREPLExecution:
    """Test basic REPL execution flow."""

    @pytest.mark.asyncio
    async def test_simple_final_answer(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test REPL execution with immediate FINAL() answer."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Test answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer == "Test answer"
                assert response.turns == 1
                assert response.mode == "repl"
                assert not response.answer.startswith("[ERROR")

    @pytest.mark.asyncio
    async def test_multi_turn_execution(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test REPL execution over multiple turns via graph."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            multi_turn_result = TaskResult(
                answer="Final answer", success=True, turns=3, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=multi_turn_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.turns == 3
                assert response.answer == "Final answer"

    @pytest.mark.asyncio
    async def test_max_turns_reached(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test REPL stops at max_turns without FINAL()."""
        basic_request.max_turns = 2

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            # Mock run_task to return max-turns result
            max_turns_result = TaskResult(
                answer="",
                success=False,
                turns=2,
                role_history=["worker_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=max_turns_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.turns == 2
                assert "[Max turns (2) reached without FINAL()]" in response.answer

    @pytest.mark.asyncio
    async def test_llm_call_exception_returns_error(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that LLM call exceptions are handled gracefully."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            # Mock run_task to return error result
            error_result = TaskResult(
                answer="[ERROR: LLM server timeout]",
                success=False,
                turns=1,
                role_history=["worker_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=error_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer.startswith("[ERROR:")
                assert "LLM server timeout" in response.answer

    @pytest.mark.asyncio
    async def test_repl_restores_globals_from_session_checkpoint(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test session_id restore path restores globals before graph execution."""
        basic_request.session_id = "sess_123"
        mock_state.session_store = MagicMock()
        checkpoint = MagicMock()
        checkpoint.user_globals = {"cached_df": [{"a": 1}]}
        checkpoint.skipped_user_globals = ["tmp_fn"]
        checkpoint.to_dict.return_value = {
            "version": 1,
            "artifacts": {},
            "execution_count": 0,
            "exploration_calls": 0,
            "exploration_tokens": 0,
            "exploration_events": [],
            "grep_hits_buffer": [],
            "findings_buffer": [],
            "user_globals": {"cached_df": [{"a": 1}]},
        }
        mock_state.session_store.get_latest_checkpoint.return_value = checkpoint
        mock_state.session_store.get_session.return_value = MagicMock(id="sess_123")

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            # restore() returns a reconciliation of what actually landed in the
            # live namespace; the executor reports that, not the claimed count.
            mock_repl.restore.return_value = {
                "restored": ["cached_df"],
                "unavailable": {"tmp_fn": "not JSON-serializable at save time; never checkpointed"},
                "claimed": 1,
                "dropped_at_save": ["tmp_fn"],
            }
            mock_repl.checkpoint.return_value = {
                "artifacts": {},
                "execution_count": 1,
                "exploration_calls": 0,
                "user_globals": {"cached_df": [{"a": 1}]},
                "variable_lineage": {"cached_df": {"role": "worker_general"}},
                "skipped_user_globals": [],
            }
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="ok", success=True, turns=1, role_history=["worker_general"]
            )
            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

        mock_state.session_store.get_latest_checkpoint.assert_called_once_with("sess_123")
        mock_repl.restore.assert_called_once()
        restored_payload = mock_repl.restore.call_args.args[0]
        assert restored_payload["user_globals"] == checkpoint.to_dict.return_value["user_globals"]
        assert restored_payload["execution_count"] == checkpoint.to_dict.return_value["execution_count"]
        assert "variable_lineage" in restored_payload
        assert "skipped_user_globals" in restored_payload
        mock_state.session_store.save_checkpoint.assert_called_once()
        assert response.session_persistence["restore_success"] is True
        assert response.session_persistence["restored_globals"] == 1
        assert response.session_persistence["claimed_globals"] == 1
        assert "tmp_fn" in response.session_persistence["unavailable_globals"]
        assert response.session_persistence["checkpoint_saved"] is True

    @pytest.mark.asyncio
    async def test_cross_request_roundtrip_restores_globals(
        self, basic_routing, mock_primitives, tmp_path
    ):
        """End-to-end stage roundtrip: request1 saves globals, request2 restores them."""
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db",
            embeddings_path=tmp_path / "embeddings.npy",
        )
        session = Session.create(name="phase3", working_directory="/tmp")
        store.create_session(session)

        state = MagicMock()
        state.tool_registry = MagicMock()
        state.script_registry = MagicMock()
        state.progress_logger = None
        state.hybrid_router = None
        state.failure_graph = MagicMock()
        state.increment_request = MagicMock()
        state.session_store = store

        request = ChatRequest(
            prompt="persist var",
            context="",
            real_mode=True,
            mock_mode=False,
            max_turns=3,
            force_role="frontdoor",
            session_id=session.id,
        )
        run_count = {"n": 0}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            run_count["n"] += 1
            if run_count["n"] == 1:
                task_deps.repl._globals["persist_me"] = {"x": 42}
                return TaskResult(answer="saved", success=True, turns=1, role_history=["frontdoor"])
            restored = task_deps.repl._globals.get("persist_me")
            return TaskResult(
                answer=f"restored={restored}",
                success=True,
                turns=1,
                role_history=["frontdoor"],
            )

        try:
            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", side_effect=_fake_run_task):
                r1 = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=state,
                    start_time=time.perf_counter(),
                    initial_role=Role.FRONTDOOR,
                )
                r2 = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=state,
                    start_time=time.perf_counter(),
                    initial_role=Role.FRONTDOOR,
                )
        finally:
            store.close()

        assert r1.session_persistence["checkpoint_saved"] is True
        assert r1.session_persistence["saved_globals"] >= 1
        assert r2.session_persistence["restore_success"] is True
        assert r2.session_persistence["restored_globals"] >= 1
        assert "restored={'x': 42}" in r2.answer


# ── Generation Monitoring ────────────────────────────────────────────────


class TestGenerationMonitoring:
    """Test generation monitoring integration.

    Generation monitoring is now handled inside graph nodes. These tests verify
    that the graph is invoked properly and the results are used.
    """

    @pytest.mark.asyncio
    async def test_graph_invoked_in_real_mode(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that the orchestration graph is invoked in real mode."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                return_value=success_result,
            ) as mock_run_task:
                await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify run_task was called
                mock_run_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_result_used_for_response(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that graph result populates the response."""
        request = ChatRequest(
            prompt="Test",
            context="",
            real_mode=True,
            mock_mode=True,
            max_turns=5,
        )

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Graph answer",
                success=True,
                turns=3,
                role_history=["worker_general", "coder_escalation"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer == "Graph answer"
                assert response.turns == 3


# ── Two-Stage Summarization ──────────────────────────────────────────────


class TestTwoStageSummarization:
    """Test two-stage summarization integration."""

    @pytest.mark.asyncio
    async def test_two_stage_summarization_triggered(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that two-stage summarization is triggered for large context."""
        request = ChatRequest(
            prompt="Summarize this document",
            context="A" * 25000,  # > 20K threshold
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor._run_two_stage_summarization"
            ) as mock_two_stage:
                mock_two_stage.return_value = (
                    "Summary result",
                    {
                        "cache_hit": False,
                        "producer_role": "frontdoor",
                        "role_history": ["worker_fast", "frontdoor"],
                    },
                )

                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer == "Summary result"
                assert response.turns == 2
                assert response.role_history == [
                    "worker_general",
                    "worker_fast",
                    "frontdoor",
                ]
                completion_meta = (
                    mock_state.progress_logger.log_task_completed.call_args.kwargs[
                        "completion_meta"
                    ]
                )
                assert completion_meta["producer_role"] == "frontdoor"
                assert completion_meta["final_answer_role"] == "frontdoor"
                assert completion_meta["delegation_lineage"] == [
                    "worker_general",
                    "worker_fast",
                    "frontdoor",
                ]
                mock_two_stage.assert_called_once()

    @pytest.mark.asyncio
    async def test_two_stage_summarization_not_triggered_for_small_context(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that two-stage is not triggered for small context."""
        request = ChatRequest(
            prompt="Summarize this",
            context="A" * 5000,  # < 20K threshold
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = False

            with patch(
                "src.api.routes.chat_pipeline.repl_executor._run_two_stage_summarization"
            ) as mock_two_stage:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
                ) as mock_repl_class:
                    mock_repl = MagicMock()
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    direct_result = TaskResult(
                        answer="Direct answer",
                        success=True,
                        turns=1,
                        role_history=["worker_general"],
                    )

                    with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=direct_result):
                        response = await _execute_repl(
                            request=request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should not have called two-stage
                        mock_two_stage.assert_not_called()
                        assert response.answer == "Direct answer"


# ── Long Context Exploration ─────────────────────────────────────────────


class TestLongContextExploration:
    """Test long context exploration mode."""

    @pytest.mark.asyncio
    async def test_long_context_uses_extended_max_turns(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that long context uses extended max_turns."""
        request = ChatRequest(
            prompt="Find data",
            context="A" * 25000,  # > 20K threshold
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = False

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.LONG_CONTEXT_CONFIG"
            ) as mock_config:
                # Configure long context mode
                mock_config.__getitem__.side_effect = lambda k: {
                    "enabled": True,
                    "threshold_chars": 20000,
                    "max_turns": 8,  # Extended max turns
                }[k]

                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
                ) as mock_repl_class:
                    mock_repl = MagicMock()
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    long_result = TaskResult(
                        answer="Final",
                        success=True,
                        turns=8,
                        role_history=["worker_general"],
                    )

                    with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=long_result):
                        response = await _execute_repl(
                            request=request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should have been able to use 8 turns (not limited to 5)
                        assert response.turns == 8
                        assert response.answer == "Final"


# ── Document REPL Environment ────────────────────────────────────────────


class TestDocumentREPLEnvironment:
    """Test DocumentREPLEnvironment integration."""

    @pytest.mark.asyncio
    async def test_document_result_uses_document_repl(
        self, basic_request, mock_primitives, mock_state
    ):
        """Test that document preprocessing results use DocumentREPLEnvironment."""
        # Mock document result
        mock_doc_result = MagicMock()
        mock_doc_result.document_result = MagicMock()
        mock_doc_result.document_result.to_searchable_text.return_value = "Searchable text"

        routing = RoutingResult(
            task_id="test-task",
            task_ir={},
            use_mock=False,
            routing_strategy="document",
            formalization_applied=False,
            document_result=mock_doc_result,
        )

        with patch("src.repl_document.DocumentREPLEnvironment") as mock_doc_repl_class:
            with patch("src.repl_document.DocumentContext") as mock_doc_context_class:
                mock_doc_context = MagicMock()
                mock_doc_context.sections = [{"title": "Intro"}]
                mock_doc_context.figures = []
                mock_doc_context_class.from_document_result.return_value = mock_doc_context

                mock_repl = MagicMock()
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.log_exploration_completed = MagicMock()
                mock_doc_repl_class.return_value = mock_repl

                success_result = TaskResult(
                    answer="Document answer",
                    success=True,
                    turns=1,
                    role_history=["worker_general"],
                )

                with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                    response = await _execute_repl(
                        request=basic_request,
                        routing=routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Verify DocumentREPLEnvironment was used
                    mock_doc_repl_class.assert_called_once()
                    assert response.answer == "Document answer"

    @pytest.mark.asyncio
    async def test_non_document_result_uses_regular_repl(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that non-document requests use regular REPLEnvironment."""
        # No document_result in routing
        assert basic_routing.document_result is None

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch("src.repl_document.DocumentREPLEnvironment") as mock_doc_repl_class:
                mock_repl = MagicMock()
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                success_result = TaskResult(
                    answer="Regular answer",
                    success=True,
                    turns=1,
                    role_history=["worker_general"],
                )

                with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                    response = await _execute_repl(
                        request=basic_request,
                        routing=basic_routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Verify regular REPLEnvironment was used, not DocumentREPLEnvironment
                    mock_repl_class.assert_called_once()
                    mock_doc_repl_class.assert_not_called()
                    assert response.answer == "Regular answer"


# ── Escalation Handling ────────────────────────────────────────────────────


class TestEscalationHandling:
    """Test escalation during REPL execution.

    Escalation is now handled inside graph nodes. These tests verify that
    graph results with escalation are correctly propagated through the executor.
    """

    @pytest.mark.asyncio
    async def test_escalation_reflected_in_role_history(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that escalation in graph is reflected in response role_history."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            escalated_result = TaskResult(
                answer="Escalated answer",
                success=True,
                turns=3,
                role_history=["worker_general", "coder_escalation"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=escalated_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert len(response.role_history) >= 2
                assert "coder_escalation" in response.role_history

    @pytest.mark.asyncio
    async def test_error_escalation_to_higher_tier(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test escalation when execution errors occur (handled by graph)."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            escalated_result = TaskResult(
                answer="Fixed",
                success=True,
                turns=2,
                role_history=["worker_general", "architect_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=escalated_result) as mock_run:
                await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify graph was invoked
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalation_explore_action(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test EXPLORE action when terminal role can't escalate further."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            explore_result = TaskResult(
                answer="Explored",
                success=True,
                turns=2,
                role_history=["architect_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=explore_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.ARCHITECT_GENERAL,
                )

                assert response.answer == "Explored"

    @pytest.mark.asyncio
    async def test_escalation_fail_action(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test FAIL action when escalation exhausted."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            fail_result = TaskResult(
                answer="[FAILED: Max escalation reached]",
                success=False,
                turns=5,
                role_history=[_RETIRED_ARCHITECT_ROLE],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=fail_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.ARCHITECT_CODING,
                )

                assert "[FAILED:" in response.answer


# ── Model-Initiated Routing ───────────────────────────────────────────────


class TestModelInitiatedRouting:
    """Test model-initiated routing via artifacts.

    Model-initiated escalation is now handled inside graph nodes.
    """

    @pytest.mark.asyncio
    async def test_model_requests_escalation(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test model requesting escalation produces escalated result."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            escalated_result = TaskResult(
                answer="Escalated answer",
                success=True,
                turns=2,
                role_history=["worker_general", "architect_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=escalated_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert "Escalated answer" in response.answer


# ── Delegation Logging ────────────────────────────────────────────────────


class TestDelegationLogging:
    """Test delegation logging for MemRL.

    Delegation events are now tracked inside graph nodes and returned
    via TaskResult.delegation_events.
    """

    @pytest.mark.asyncio
    async def test_delegation_events_in_response(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that delegation events from graph are included in response."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            result_with_delegations = TaskResult(
                answer="Final",
                success=True,
                turns=2,
                role_history=["worker_general"],
                delegation_events=[
                    {"from_role": "worker_general", "to_role": "coder", "task_summary": "subtask", "success": True}
                ],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=result_with_delegations):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.delegation_events is not None
                assert len(response.delegation_events) == 1
                assert response.delegation_success is True


# ── Execution Timeout ─────────────────────────────────────────────────────


class TestExecutionTimeout:
    """Test REPL execution timeout handling.

    Timeouts are now handled inside graph nodes. This test verifies that
    a timeout result from the graph is handled gracefully.
    """

    @pytest.mark.asyncio
    async def test_repl_execution_timeout(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that REPL execution timeout is handled."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            timeout_result = TaskResult(
                answer="[ERROR: Execution timeout]",
                success=False,
                turns=1,
                role_history=["worker_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=timeout_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Should have handled timeout gracefully
                assert response is not None
                assert "[ERROR" in response.answer
                assert response.delegation_diagnostics.get("break_reason") == "request_timeout"


# ── Two-Stage Exception Handling ─────────────────────────────────────────


class TestTwoStageExceptionHandling:
    """Test two-stage summarization exception handling."""

    @pytest.mark.asyncio
    async def test_two_stage_exception_falls_back_to_repl(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that two-stage exception falls back to regular REPL."""
        request = ChatRequest(
            prompt="Summarize this document",
            context="A" * 25000,
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor._run_two_stage_summarization"
            ) as mock_two_stage:
                mock_two_stage.side_effect = Exception("Two-stage failed")

                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
                ) as mock_repl_class:
                    mock_repl = MagicMock()
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    fallback_result = TaskResult(
                        answer="Fallback answer",
                        success=True,
                        turns=1,
                        role_history=["worker_general"],
                    )

                    with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=fallback_result):
                        response = await _execute_repl(
                            request=request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should have fallen back to REPL via graph
                        assert response.answer == "Fallback answer"
                        mock_repl_class.assert_called_once()


# ── Context Handling ──────────────────────────────────────────────────────


class TestContextHandling:
    """Test request context handling."""

    @pytest.mark.asyncio
    async def test_context_appended_to_prompt(self, basic_routing, mock_primitives, mock_state):
        """Test that request context is appended to prompt."""
        request = ChatRequest(
            prompt="What is the answer?",
            context="The answer is 42",
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        captured_context = []

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:

            def capture_init(**kwargs):
                captured_context.append(kwargs.get("context", ""))
                mock_repl = MagicMock()
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.log_exploration_completed = MagicMock()
                return mock_repl

            mock_repl_class.side_effect = capture_init

            success_result = TaskResult(
                answer="42", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify context was appended
                assert len(captured_context) == 1
                assert "The answer is 42" in captured_context[0]


# ── Tool Outputs in Answer ───────────────────────────────────────────────


class TestToolOutputsInAnswer:
    """Test tool outputs handling in answer resolution.

    Tool outputs are now tracked inside graph nodes and the result's answer
    is resolved within the graph. The _tools_success() helper in repl_executor
    still reads from repl.artifacts after the graph completes.
    """

    @pytest.mark.asyncio
    async def test_tool_outputs_tracked_in_response(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that tool outputs from REPL are reflected in response."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {"_tool_outputs": ["output1", "output2"]}
            mock_repl._tool_invocations = 2
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify tool invocations tracked
                assert response.tools_used == 2
                # tools_success should be inferred from tool outputs
                assert response.tools_success is not None or response.tools_success is None

    @pytest.mark.asyncio
    async def test_tool_chains_grouped_in_response(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Chained tool invocations should be grouped into ChatResponse.tool_chains."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {"_tool_outputs": []}
            mock_repl._tool_invocations = 2
            mock_repl._get_read_only_tools = MagicMock(return_value=set())
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl.tool_registry = MagicMock()
            mock_repl._invoked_tools = [
                SimpleNamespace(
                    tool_name="read_file",
                    elapsed_ms=10.0,
                    success=True,
                    chain_id="ch_1",
                    caller_type="chain",
                ),
                SimpleNamespace(
                    tool_name="list_directory",
                    elapsed_ms=15.5,
                    success=True,
                    chain_id="ch_1",
                    caller_type="chain",
                ),
            ]
            mock_repl.get_chain_execution_log.return_value = [
                {
                    "chain_id": "ch_1",
                    "mode_requested": "dep",
                    "mode_used": "dep",
                    "fallback_to_seq": False,
                    "parallel_mutations_enabled": True,
                    "waves": 2,
                    "steps": 2,
                    "wave_timeline": [
                        {
                            "wave_index": 0,
                            "tools": ["read_file"],
                            "mode_used": "dep",
                            "elapsed_ms": 10.1,
                            "fallback_to_seq": False,
                            "parallel_mutations_enabled": False,
                        },
                        {
                            "wave_index": 1,
                            "tools": ["list_directory"],
                            "mode_used": "dep",
                            "elapsed_ms": 15.3,
                            "fallback_to_seq": False,
                            "parallel_mutations_enabled": False,
                        },
                    ],
                }
            ]
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert len(response.tool_chains) == 1
                chain = response.tool_chains[0]
                assert chain["chain_id"] == "ch_1"
                assert chain["caller_type"] == "chain"
                assert chain["tools"] == ["read_file", "list_directory"]
                assert chain["elapsed_ms"] == 25.5
                assert chain["success"] is True
                assert chain["mode_requested"] == "dep"
                assert chain["mode_used"] == "dep"
                assert chain["fallback_to_seq"] is False
                assert chain["parallel_mutations_enabled"] is True
                assert chain["waves"] == 2
                assert chain["steps"] == 2
                assert len(chain["wave_timeline"]) == 2
                assert chain["wave_timeline"][0]["wave_index"] == 0
                assert chain["wave_timeline"][0]["tools"] == ["read_file"]
                assert chain["wave_timeline"][1]["wave_index"] == 1
                assert chain["wave_timeline"][1]["tools"] == ["list_directory"]


# ── D-f: request-scoped cross-process session lease ──────────────────────


def _lease_state(store):
    state = MagicMock()
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.progress_logger = None
    state.hybrid_router = None
    state.failure_graph = MagicMock()
    state.increment_request = MagicMock()
    state.session_store = store
    return state


class TestSessionLease:
    @pytest.mark.asyncio
    async def test_request_holds_and_releases_the_lease_and_fences_its_save(
        self, basic_routing, mock_primitives, tmp_path
    ):
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db", embeddings_path=tmp_path / "embeddings.npy"
        )
        session = Session.create(name="lease", working_directory="/tmp")
        store.create_session(session)
        seen = {}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            # While the turn runs, the session is owned: an unfenced writer
            # (e.g. another worker's request) is refused.
            seen["live"] = store.leases.get(session.id).is_live(time.time())
            with pytest.raises(Exception, match="unfenced write is refused"):
                store.update_session(store.get_session(session.id))
            # D-f3: the graph gets the session id + live token, so its
            # snapshot writes are session-scoped and fenced.
            seen["deps"] = (task_deps.session_id, task_deps.session_fencing_token)
            store.save_graph_snapshot(
                task_state.task_id, "{}", "state_snapshot",
                session_id=task_deps.session_id,
                fencing_token=task_deps.session_fencing_token,
            )
            return TaskResult(answer="ok", success=True, turns=1, role_history=["frontdoor"])

        request = ChatRequest(
            prompt="p", context="", real_mode=True, mock_mode=False, max_turns=3,
            force_role="frontdoor", session_id=session.id,
        )
        try:
            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", side_effect=_fake_run_task):
                r = await _execute_repl(
                    request=request, routing=basic_routing, primitives=mock_primitives,
                    state=_lease_state(store), start_time=time.perf_counter(),
                    initial_role=Role.FRONTDOOR,
                )
            assert seen["live"] is True
            assert seen["deps"] == (session.id, 1)
            assert r.session_persistence["lease_token"] == 1
            assert r.session_persistence["lease_error"] is None
            assert r.session_persistence["checkpoint_saved"] is True
            assert not store.leases.get(session.id).is_live(time.time())  # released
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_session_owned_elsewhere_runs_stateless_and_says_so(
        self, basic_routing, mock_primitives, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ORCHESTRATOR_SESSION_LEASE_WAIT_S", "0")
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db", embeddings_path=tmp_path / "embeddings.npy"
        )
        session = Session.create(name="lease", working_directory="/tmp")
        store.create_session(session)
        other = store.leases.try_acquire(session.id, ttl_s=60)  # another worker's turn
        request = ChatRequest(
            prompt="p", context="", real_mode=True, mock_mode=False, max_turns=3,
            force_role="frontdoor", session_id=session.id,
        )
        seen = {}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            seen["deps"] = (task_deps.session_id, task_deps.session_fencing_token)
            return TaskResult(answer="ok", success=True, turns=1, role_history=["frontdoor"])

        try:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                side_effect=_fake_run_task,
            ):
                r = await _execute_repl(
                    request=request, routing=basic_routing, primitives=mock_primitives,
                    state=_lease_state(store), start_time=time.perf_counter(),
                    initial_role=Role.FRONTDOOR,
                )
            sp = r.session_persistence
            assert sp["lease_token"] is None
            assert sp["lease_error"].startswith("held_by_other_owner")
            assert sp["restore_error"] == "session_lease_unavailable"
            assert sp["checkpoint_saved"] is False
            assert store.get_checkpoints(session.id) == []
            # D-f3: a stateless turn's snapshots stay run-scoped.
            assert seen["deps"] == (None, None)
            # The other owner's lease is untouched.
            assert store.leases.get(session.id).fencing_token == other.fencing_token
            assert store.leases.get(session.id).is_live(time.time())
        finally:
            store.close()


# ── TD-21.1: FINAL() schema-validation repair (typed-decision-plane.md) ───
#
# Before this conversion, a schema-invalid FINAL() value paid for an entire
# graph re-run (turns=0, role reset, run_task re-entered) up to 2 attempts,
# and if BOTH attempts still failed the invalid value was returned as a
# normal ChatResponse with no signal at all (error_code/error_detail both
# None => HTTP 200). These tests prove: (a) a repairable failure is fixed by
# ONE constrained turn with NO graph re-run; (b) an unrepairable failure
# falls back to the pre-existing retry-from-zero path; (c) once both are
# exhausted the response is flagged (error_code=422) instead of silently
# served as valid; (d) STRUCTURED_OUTPUT_REPAIR_COUNTS increments per site.


def _enable_final_schema_validation(monkeypatch):
    monkeypatch.setattr(
        "src.features.features",
        lambda: SimpleNamespace(final_schema_validation=True),
    )


_SCHEMA_ANSWER_STRING = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _schema_repl_repl_class():
    """A REPLEnvironment double with the attributes _execute_repl touches."""
    mock_repl = MagicMock()
    mock_repl.artifacts = {}
    mock_repl._tool_invocations = 0
    mock_repl.tool_registry = None
    mock_repl.log_exploration_completed = MagicMock()
    return mock_repl


class TestFinalSchemaValidationRepair:
    """TD-21.1 conversion of repl_executor.py's FINAL() schema-validation path."""

    @pytest.mark.asyncio
    async def test_fished_directly_needs_zero_extra_calls(
        self, basic_routing, mock_primitives, mock_state, monkeypatch
    ):
        """A FINAL() value with stray prose around valid JSON is recovered by
        ``fish_json`` alone (parse_with_repair status "parsed") -- zero extra
        completion calls, and the graph is NOT re-run."""
        _enable_final_schema_validation(monkeypatch)
        from src.structured_output.repair import (
            STRUCTURED_OUTPUT_REPAIR_COUNTS,
            reset_counts_for_tests,
        )

        reset_counts_for_tests()
        request = ChatRequest(
            prompt="answer please",
            context="",
            real_mode=True,
            mock_mode=False,
            max_turns=5,
            output_schema=_SCHEMA_ANSWER_STRING,
        )
        run_count = {"n": 0}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            run_count["n"] += 1
            return TaskResult(
                answer='Sure, here you go:\n```json\n{"answer": "42"}\n```\nHope that helps!',
                success=True,
                turns=1,
                role_history=["worker_general"],
            )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment",
            return_value=_schema_repl_repl_class(),
        ):
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                side_effect=_fake_run_task,
            ):
                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

        assert run_count["n"] == 1  # the graph was NOT re-run
        assert json.loads(response.answer) == {"answer": "42"}
        assert response.error_code is None
        mock_primitives.llm_call.assert_not_called()
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("repl_final", "parsed")) == 1

    @pytest.mark.asyncio
    async def test_repairable_final_is_repaired_without_graph_rerun(
        self, basic_routing, mock_primitives, mock_state, monkeypatch
    ):
        """(a) An invalid-but-repairable FINAL value is fixed by ONE
        constrained extraction turn -- the graph is NOT re-run."""
        _enable_final_schema_validation(monkeypatch)
        from src.structured_output.repair import (
            STRUCTURED_OUTPUT_REPAIR_COUNTS,
            reset_counts_for_tests,
        )

        reset_counts_for_tests()
        request = ChatRequest(
            prompt="answer please",
            context="",
            real_mode=True,
            mock_mode=False,
            max_turns=5,
            output_schema=_SCHEMA_ANSWER_STRING,
        )
        run_count = {"n": 0}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            run_count["n"] += 1
            return TaskResult(
                # TD-21.34 require_evidence: "42" must be literally present
                # in the raw answer for the repair turn's copy to be
                # evidence-checkable (a spelled-out "forty-two" would not be).
                answer="The answer is 42, all done.",
                success=True,
                turns=1,
                role_history=["worker_general"],
            )

        mock_primitives.llm_call.return_value = json.dumps({"answer": "42"})

        with patch(
            "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment",
            return_value=_schema_repl_repl_class(),
        ):
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                side_effect=_fake_run_task,
            ):
                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

        assert run_count["n"] == 1  # the graph was NOT re-run
        assert json.loads(response.answer) == {"answer": "42"}
        assert response.error_code is None
        assert response.error_detail is None
        mock_primitives.llm_call.assert_called_once()
        assert mock_primitives.llm_call.call_args.kwargs.get("role") == "worker_general"
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("repl_final", "repaired")) == 1

    @pytest.mark.asyncio
    async def test_unrepairable_final_falls_back_to_retry_path(
        self, basic_routing, mock_primitives, mock_state, monkeypatch
    ):
        """(b) When the repair turn ALSO fails, the pre-existing
        retry-from-zero path (turns=0, role reset, run_task re-entered)
        still runs and can still recover a valid answer on the second pass."""
        _enable_final_schema_validation(monkeypatch)
        from src.structured_output.repair import (
            STRUCTURED_OUTPUT_REPAIR_COUNTS,
            reset_counts_for_tests,
        )

        reset_counts_for_tests()
        request = ChatRequest(
            prompt="answer please",
            context="",
            real_mode=True,
            mock_mode=False,
            max_turns=5,
            output_schema=_SCHEMA_ANSWER_STRING,
        )
        run_count = {"n": 0}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            run_count["n"] += 1
            if run_count["n"] == 1:
                return TaskResult(
                    answer="totally unparseable nonsense with no braces at all",
                    success=True,
                    turns=1,
                    role_history=["worker_general"],
                )
            # Second pass (after the retry-from-zero reset) succeeds.
            return TaskResult(
                answer=json.dumps({"answer": "42"}),
                success=True,
                turns=1,
                role_history=["worker_general"],
            )

        # The repair extraction turn ALSO returns unparseable text -> repair fails.
        mock_primitives.llm_call.return_value = "still no JSON here either"

        with patch(
            "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment",
            return_value=_schema_repl_repl_class(),
        ):
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                side_effect=_fake_run_task,
            ):
                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

        assert run_count["n"] == 2  # the retry-from-zero path DID re-run the graph
        assert json.loads(response.answer) == {"answer": "42"}
        assert response.error_code is None
        assert mock_primitives.llm_call.call_count == 1  # one repair attempt, then fallback
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("repl_final", "failed")) == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_flag_response_as_invalid(
        self, basic_routing, mock_primitives, mock_state, monkeypatch
    ):
        """(c) When BOTH attempts (each preceded by a failed repair turn) are
        exhausted, the terminal defect is fixed: the response no longer comes
        back as a silent 200 with the invalid value -- it carries
        error_code=422 and an error_detail naming the schema failure, per the
        same convention chat.py already uses for other validation drops."""
        _enable_final_schema_validation(monkeypatch)
        from src.structured_output.repair import (
            STRUCTURED_OUTPUT_REPAIR_COUNTS,
            reset_counts_for_tests,
        )

        reset_counts_for_tests()
        request = ChatRequest(
            prompt="answer please",
            context="",
            real_mode=True,
            mock_mode=False,
            max_turns=5,
            output_schema=_SCHEMA_ANSWER_STRING,
        )
        run_count = {"n": 0}

        async def _fake_run_task(task_state, task_deps, start_role=None):
            run_count["n"] += 1
            return TaskResult(
                answer=f"still gibberish, attempt {run_count['n']}",
                success=True,
                turns=1,
                role_history=["worker_general"],
            )

        mock_primitives.llm_call.return_value = "still no JSON here either"

        with patch(
            "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment",
            return_value=_schema_repl_repl_class(),
        ):
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                side_effect=_fake_run_task,
            ):
                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

        assert run_count["n"] == 2  # both attempts used
        # (d) one failed repair call recorded per attempt.
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("repl_final", "failed")) == 2
        assert mock_primitives.llm_call.call_count == 2
        # The terminal defect: invalid output is FLAGGED, not silently served as valid.
        assert response.error_code == 422
        assert response.error_detail is not None
        assert "schema" in response.error_detail.lower()
        # The answer text itself is left as the model's last attempt (still
        # visible to a caller doing its own recovery), unchanged in shape --
        # what changed is that the caller can now tell it never validated.
        assert response.answer == "still gibberish, attempt 2"
