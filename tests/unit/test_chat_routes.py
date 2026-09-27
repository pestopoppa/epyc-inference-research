"""Characterization tests for src/api/routes/chat.py.

Tests the chat route handlers and helper functions:
- _try_cheap_first() edge cases (disabled, forced, vision, delegated, empty, short)
- /chat endpoint routing via TestClient with mock mode
- /chat/reward endpoint
- _handle_chat dispatcher behavior
- /chat/stream endpoint returns StreamingResponse
"""

from __future__ import annotations

import time
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from src.api.dependencies import dep_app_state
from src.api.models import ChatRequest, ChatResponse, RewardRequest
from src.api.routes.chat import (
    _handle_chat,
    _try_cheap_first,
    chat,
    chat_stream,
    inject_reward,
    router,
)
from src.api.routes.chat_utils import RoutingResult
from src.api.state import AppState
from src.prompt_builders.resolver import current_prompt_dir, resolve_prompt
from src.proactive_delegation.types import PlanReviewResult


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_state():
    """Create a mock AppState with required attributes."""
    state = MagicMock(spec=AppState)
    state.progress_logger = None
    state.hybrid_router = None
    state.tool_registry = None
    state.script_registry = None
    state.registry = None
    state.health_tracker = MagicMock()
    state.admission = MagicMock()
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def mock_primitives():
    """Create a mock LLMPrimitives."""
    primitives = MagicMock()
    primitives.llm_call = MagicMock(
        return_value="This is a sufficiently long mock answer for testing purposes."
    )
    return primitives


@pytest.fixture
def base_routing():
    """Create a baseline RoutingResult for cheap-first tests."""
    return RoutingResult(
        task_id="test-abc123",
        task_ir={"task_type": "chat", "objective": "test"},
        use_mock=False,
        routing_decision=["frontdoor"],
        routing_strategy="rules",
        skill_ids=[],
    )


@pytest.fixture
def base_request():
    """Create a baseline ChatRequest."""
    return ChatRequest(
        prompt="What is the meaning of life?",
        mock_mode=False,
        real_mode=True,
    )


@pytest.fixture
def test_app(mock_state):
    """Create a FastAPI test app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[dep_app_state] = lambda: mock_state
    return app


@pytest.fixture
def client(test_app):
    """Create a synchronous test client."""
    with TestClient(test_app) as test_client:
        yield test_client


# ── _try_cheap_first edge cases ─────────────────────────────────────────────


class TestTryCheapFirstEdgeCases:
    """Tests for the _try_cheap_first speculative pre-filter."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When try_cheap_first_enabled=False, returns None immediately."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = False
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_forced_mode(self, base_routing, mock_primitives, mock_state):
        """When request has force_mode set, returns None."""
        request = ChatRequest(
            prompt="test prompt",
            mock_mode=False,
            real_mode=True,
            force_mode="direct",
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_forced_role(self, base_routing, mock_primitives, mock_state):
        """When request has force_role set, returns None."""
        request = ChatRequest(
            prompt="test prompt",
            mock_mode=False,
            real_mode=True,
            force_role="coder",
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_vision_role(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When initial_role is a vision role, returns None (already cheap)."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "worker_vision",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("initial_role", ["worker_general", "worker_explore"])
    async def test_returns_none_for_live_worker_role(
        self, base_request, base_routing, mock_primitives, mock_state, initial_role
    ):
        """When initial_role is a cheap worker role, returns None."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                initial_role,
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_long_context_capacity_route(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """Long-context routing must never speculate with the cheap worker."""
        base_routing.routing_decision = ["ingest_long_context"]
        base_routing.routing_strategy = "long_context_guard"
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "ingest_long_context",
                "direct",
            )

        assert result is None
        mock_primitives.llm_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_for_delegated_mode(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When execution_mode is 'delegated', returns None."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "delegated",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns empty string, returns None."""
        mock_primitives.llm_call.return_value = ""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_short_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns answer shorter than 20 chars, returns None."""
        mock_primitives.llm_call.return_value = "Too short"
        mock_state.progress_logger = MagicMock()
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None
            entry = mock_state.progress_logger.log.call_args.args[0]
            assert entry.event_type.value == "routing_fallback"
            assert entry.data["kind"] == "try_cheap_first"
            assert entry.data["cheap_first_attempted"] is True
            assert entry.data["cheap_first_passed"] is False
            assert entry.data["reason"] == "empty_or_short_answer"

    @pytest.mark.asyncio
    async def test_returns_none_on_error_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns an error-like answer, returns None."""
        mock_primitives.llm_call.return_value = "[ERROR] Something went wrong with inference"
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_response_on_good_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns a good answer, returns ChatResponse."""
        mock_primitives.llm_call.return_value = (
            "The meaning of life is a philosophical question that has been debated for centuries."
        )
        mock_state.progress_logger = MagicMock()
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            with patch(
                "src.api.routes.chat_review._detect_output_quality_issue",
                return_value=None,
            ):
                result = await _try_cheap_first(
                    base_request,
                    base_routing,
                    mock_primitives,
                    mock_state,
                    time.perf_counter(),
                    "frontdoor",
                    "direct",
                )
                assert result is not None
                assert isinstance(result, ChatResponse)
                assert result.cheap_first_attempted is True
                assert result.cheap_first_passed is True
                assert result.routed_to == "worker_general"
                assert "cheap_first" in result.routing_strategy
                entry = mock_state.progress_logger.log.call_args.args[0]
                assert entry.event_type.value == "routing_fallback"
                assert entry.data["kind"] == "try_cheap_first"
                assert entry.data["cheap_first_attempted"] is True
                assert entry.data["cheap_first_passed"] is True
                assert entry.data["reason"] == "passed"

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_exception(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM call raises exception, returns None gracefully."""
        mock_primitives.llm_call.side_effect = ConnectionError("Backend down")
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_quality_issue(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When quality detector flags an issue, returns None."""
        mock_primitives.llm_call.return_value = (
            "Here is a long enough answer that should pass length checks easily."
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_general"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            with patch(
                "src.api.routes.chat_review._detect_output_quality_issue",
                return_value="repetitive output",
            ):
                result = await _try_cheap_first(
                    base_request,
                    base_routing,
                    mock_primitives,
                    mock_state,
                    time.perf_counter(),
                    "frontdoor",
                    "direct",
                )
                assert result is None


# ── /chat endpoint tests ────────────────────────────────────────────────────


class TestChatEndpoint:
    """Tests for POST /chat via TestClient."""

    @pytest.mark.asyncio
    async def test_mock_mode_returns_200(self, mock_state):
        """Mock mode chat request returns 200 with mock answer."""
        mock_response = ChatResponse(
            answer="[MOCK] Processed prompt: Hello...",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
            real_mode=False,
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="Hello", mock_mode=True), _FakeRequest(), mock_state
            )
            assert response is mock_response
            data = response.model_dump()
            assert data["mock_mode"] is True
            assert "[MOCK]" in data["answer"]

    @pytest.mark.asyncio
    async def test_chat_returns_200_for_successful_request(self, mock_state):
        """Chat endpoint returns 200 for a successful non-error response."""
        mock_response = ChatResponse(
            answer="A real answer to the user question.",
            turns=2,
            elapsed_seconds=1.5,
            mock_mode=False,
            real_mode=True,
            routed_to="frontdoor",
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="Explain recursion", mock_mode=False, real_mode=True),
                _FakeRequest(),
                mock_state,
            )
            data = response.model_dump()
            assert data["answer"] == "A real answer to the user question."
            assert data["turns"] == 2

    @pytest.mark.asyncio
    async def test_chat_returns_error_status_on_error_code(self, mock_state):
        """When _handle_chat returns error_code, HTTP status matches."""
        mock_response = ChatResponse(
            answer="Backend timeout",
            turns=1,
            elapsed_seconds=60.0,
            mock_mode=False,
            real_mode=True,
            error_code=504,
            error_detail="Request timed out",
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="Slow query", mock_mode=False, real_mode=True),
                _FakeRequest(),
                mock_state,
            )
            assert response.status_code == 504
            data = json.loads(response.body.decode("utf-8"))
            assert data["error_code"] == 504

    @pytest.mark.asyncio
    async def test_chat_503_includes_retry_after(self, mock_state):
        """When error_code=503, response includes Retry-After header."""
        mock_response = ChatResponse(
            answer="Service unavailable",
            turns=0,
            elapsed_seconds=0.0,
            mock_mode=False,
            real_mode=True,
            error_code=503,
            error_detail="Backend down",
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="test", mock_mode=False, real_mode=True),
                _FakeRequest(),
                mock_state,
            )
            assert response.status_code == 503
            assert response.headers.get("retry-after") == "30"

    @pytest.mark.asyncio
    async def test_chat_increments_and_decrements_active(self, mock_state):
        """Chat endpoint calls increment_active/decrement_active around handling."""
        mock_response = ChatResponse(
            answer="ok",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            await chat(ChatRequest(prompt="test"), _FakeRequest(), mock_state)
            mock_state.increment_active.assert_called_once()
            mock_state.decrement_active.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_prompt_root_override_is_request_scoped(
        self, tmp_path, monkeypatch, mock_state
    ):
        """Internal GEPA prompt roots redirect prompt resolution only during handling."""
        scratch = tmp_path / "gepa" / "candidate"
        scratch.mkdir(parents=True)
        (scratch / "unit_prompt.md").write_text("scratch prompt")
        monkeypatch.setenv("ORCHESTRATOR_PROMPT_ROOT_OVERRIDE_BASE", str(tmp_path / "gepa"))

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            assert current_prompt_dir() == scratch.resolve()
            assert resolve_prompt("unit_prompt", "fallback") == "scratch prompt"
            return ChatResponse(answer="ok", turns=1, elapsed_seconds=0.01, mock_mode=True)

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="test", x_orchestrator_prompt_root=str(scratch)),
                _FakeRequest(),
                mock_state,
            )
        assert response.answer == "ok"
        assert current_prompt_dir() != scratch.resolve()


# ── /chat/reward endpoint tests ─────────────────────────────────────────────


class TestRewardEndpoint:
    """Tests for POST /chat/reward."""

    @pytest.mark.asyncio
    async def test_inject_reward_returns_success(self, mock_state):
        """inject_reward endpoint returns success dict."""

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("src.api.routes.chat.store_external_reward", return_value=True) as mock_store,
            patch("src.api.routes.chat.asyncio.to_thread", new=fake_to_thread),
        ):
            response = await inject_reward(
                RewardRequest(
                    task_description="Solve fizzbuzz",
                    action="coder:direct",
                    reward=0.85,
                ),
                mock_state,
            )
            assert response["success"] is True
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_reward_returns_failure(self, mock_state):
        """inject_reward returns success=False when store fails."""

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("src.api.routes.chat.store_external_reward", return_value=False),
            patch("src.api.routes.chat.asyncio.to_thread", new=fake_to_thread),
        ):
            response = await inject_reward(
                RewardRequest(
                    task_description="Hard problem",
                    action="architect:delegated",
                    reward=-0.5,
                ),
                mock_state,
            )
            assert response["success"] is False

    @pytest.mark.asyncio
    async def test_inject_reward_with_embedding(self, mock_state):
        """inject_reward passes precomputed embedding to store."""
        embedding = [0.1, 0.2, 0.3]

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("src.api.routes.chat.store_external_reward", return_value=True) as mock_store,
            patch("src.api.routes.chat.asyncio.to_thread", new=fake_to_thread),
        ):
            response = await inject_reward(
                RewardRequest(
                    task_description="Test task",
                    action="frontdoor:direct",
                    reward=1.0,
                    embedding=embedding,
                ),
                mock_state,
            )
            assert response["success"] is True
            # Verify embedding was passed through
            call_args = mock_store.call_args
            assert call_args[0][5] == embedding or call_args[1].get("embedding") == embedding


# ── _handle_chat dispatcher tests ───────────────────────────────────────────


class TestHandleChat:
    """Tests for _handle_chat dispatcher logic."""

    @pytest.mark.asyncio
    async def test_script_interception_returns_before_routing_when_enabled(self, mock_state):
        request = ChatRequest(
            prompt="How many y-intercepts does the graph of the parabola x = y^2 - 4y - 1 have?",
            mock_mode=False,
            real_mode=True,
        )

        with (
            patch(
                "src.api.routes.chat.features",
                return_value=SimpleNamespace(script_interception=True),
            ),
            patch("src.api.routes.chat._route_request") as mock_route,
        ):
            result = await _handle_chat(request, mock_state)

        mock_route.assert_not_called()
        mock_state.increment_request.assert_called_once_with(mock_mode=False, turns=0)
        assert result.answer == "2"
        assert result.turns == 0
        assert result.tokens_used == 0
        assert result.real_mode is False
        assert result.routed_to == "local_script"
        assert result.routing_strategy == "script_interception:parabola_y_intercepts"
        assert result.mode == "script_interception"

    @pytest.mark.asyncio
    async def test_dispatches_to_execute_mock(self, mock_state):
        """_handle_chat returns mock response when use_mock is True."""
        request = ChatRequest(prompt="Hello", mock_mode=True, real_mode=False)
        mock_routing = RoutingResult(
            task_id="test-123",
            task_ir={"task_type": "chat", "objective": "Hello"},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        mock_response = ChatResponse(
            answer="[MOCK] Hello",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )
        with (
            patch("src.api.routes.chat._route_request", return_value=mock_routing) as mock_route,
            patch("src.api.routes.chat._preprocess"),
            patch("src.api.routes.chat._execute_mock", return_value=mock_response) as mock_exec,
        ):
            result = await _handle_chat(request, mock_state)
            mock_route.assert_called_once_with(request, mock_state)
            mock_exec.assert_called_once()
            assert result.mock_mode is True
            assert result.answer == "[MOCK] Hello"

    @pytest.mark.asyncio
    async def test_calls_route_request_first(self, mock_state):
        """_handle_chat calls _route_request as the first stage."""
        request = ChatRequest(prompt="Test", mock_mode=True)
        mock_routing = RoutingResult(
            task_id="test-456",
            task_ir={"task_type": "chat", "objective": "Test"},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        mock_response = ChatResponse(
            answer="[MOCK] Test",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )
        with (
            patch("src.api.routes.chat._route_request", return_value=mock_routing) as mock_route,
            patch("src.api.routes.chat._preprocess"),
            patch("src.api.routes.chat._execute_mock", return_value=mock_response),
        ):
            await _handle_chat(request, mock_state)
            mock_route.assert_called_once_with(request, mock_state)

    @pytest.mark.asyncio
    async def test_calls_preprocess_before_execution(self, mock_state):
        """_handle_chat calls _preprocess after routing."""
        request = ChatRequest(prompt="Preprocess test", mock_mode=True)
        mock_routing = RoutingResult(
            task_id="test-789",
            task_ir={"task_type": "chat", "objective": "Preprocess test"},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        mock_response = ChatResponse(
            answer="[MOCK] Preprocessed",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )
        with (
            patch("src.api.routes.chat._route_request", return_value=mock_routing),
            patch("src.api.routes.chat._preprocess") as mock_pre,
            patch("src.api.routes.chat._execute_mock", return_value=mock_response),
        ):
            await _handle_chat(request, mock_state)
            mock_pre.assert_called_once_with(request, mock_state, mock_routing)


# ── /chat/stream endpoint tests ─────────────────────────────────────────────


class TestChatStreamEndpoint:
    """Tests for POST /chat/stream."""

    @pytest.mark.asyncio
    async def test_stream_returns_streaming_response(self, mock_state):
        """chat_stream endpoint returns a streaming response (200)."""
        # Patch features to use legacy streaming (not unified)
        with patch("src.api.routes.chat.features") as mock_features:
            mock_features.return_value.unified_streaming = False
            mock_state.progress_logger = None

            response = await chat_stream(
                ChatRequest(prompt="Stream test", mock_mode=True, real_mode=False),
                mock_state,
            )
            assert isinstance(response, StreamingResponse)
            assert "text/event-stream" in (response.media_type or "")

    @pytest.mark.asyncio
    async def test_stream_mock_mode_contains_done(self, mock_state):
        """Mock mode stream contains [DONE] sentinel."""
        with patch("src.api.routes.chat.features") as mock_features:
            mock_features.return_value.unified_streaming = False
            mock_state.progress_logger = None

            response = await chat_stream(
                ChatRequest(prompt="Hello stream", mock_mode=True, real_mode=False),
                mock_state,
            )
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunks.append(chunk.decode("utf-8", errors="replace"))
                else:
                    chunks.append(str(chunk))
            body = "".join(chunks)
            assert "[DONE]" in body

    @pytest.mark.asyncio
    async def test_stream_rejects_task_scope_and_quiescence(self, mock_state, tmp_path):
        """INF-78 review fix F2: /chat/stream installs no task scope and no quiescence
        carrier, so a request carrying any of task_root/read_roots/quiescent_after/edit_mode
        must be refused (422) rather than silently run unscoped."""
        from fastapi import HTTPException

        with patch("src.api.routes.chat.features") as mock_features:
            mock_features.return_value.unified_streaming = False
            mock_state.progress_logger = None

            for kwargs in (
                {"task_root": str(tmp_path)},
                {"task_root": str(tmp_path), "read_roots": [str(tmp_path)]},
                {"task_root": str(tmp_path), "edit_mode": "direct"},
                {"quiescent_after": True},
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await chat_stream(
                        ChatRequest(prompt="scoped stream", mock_mode=True,
                                    real_mode=False, **kwargs),
                        mock_state,
                    )
                assert exc_info.value.status_code == 422
                assert "task scope" in exc_info.value.detail


class TestPlanReviewDrop:
    """Plan-review drop discards scaffolding and continues normal execution."""

    @pytest.mark.asyncio
    async def test_drop_falls_back_to_default_route_execution(
        self, mock_state, mock_primitives, base_routing
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        mock_primitives.request_context = MagicMock(return_value=nullcontext())
        mock_primitives.total_tokens_generated = 49
        mock_primitives.total_generation_ms = 4005.0

        review = PlanReviewResult(
            decision="drop",
            score=0.9,
            feedback="Plan incomplete; missing problem logic and solution steps.",
            patches=[],
        )
        req = ChatRequest(
            prompt="Implement Aho-Corasick.",
            mock_mode=False,
            real_mode=True,
        )
        executed = ChatResponse(
            answer="implemented",
            turns=1,
            elapsed_seconds=0.1,
            mock_mode=False,
            real_mode=True,
            routed_to="frontdoor",
            mode="direct",
        )

        with (
            patch("src.api.routes.chat._route_request", return_value=base_routing),
            patch("src.api.routes.chat._preprocess", return_value=None),
            patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
            patch("src.api.routes.chat._plan_review_gate", return_value=review),
            patch(
                "src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)
            ) as mock_vision,
            patch(
                "src.api.routes.chat._execute_vision_multimodal", new=AsyncMock(return_value=None)
            ) as mock_vision_mm,
            patch(
                "src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)
            ) as mock_proactive,
            patch(
                "src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)
            ) as mock_cheap,
            patch("src.api.routes.chat._select_mode", return_value="direct"),
            patch("src.api.routes.chat._execute_direct", return_value=executed) as mock_direct,
        ):
            result = await _handle_chat(req, mock_state)

        assert result.mode == "direct"
        assert result.error_code is None
        assert result.answer == "implemented"
        assert result.routed_to == "frontdoor"
        mock_vision.assert_awaited_once()
        mock_vision_mm.assert_not_called()
        mock_proactive.assert_awaited_once()
        mock_cheap.assert_awaited_once()
        mock_direct.assert_called_once()


class TestEditModeFailClosed:
    """force_mode='edit' must FAIL CLOSED (HTTP 412) — not silently fall through to the REPL —
    when the edit-transaction preconditions (ORCHESTRATOR_EDIT_TRANSACTION + scoped
    ORCHESTRATOR_EDIT_ROOT) are unmet. Regression for the 2026-05-27 review finding #3; the
    end-to-end 412 was live-probed, this locks it in CI."""

    @pytest.mark.asyncio
    async def test_force_mode_edit_412_when_flag_and_root_missing(
        self, mock_state, mock_primitives, base_routing, monkeypatch
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        from fastapi import HTTPException

        # preconditions deliberately unmet (prod default): no flag, no scoped edit-root.
        monkeypatch.delenv("ORCHESTRATOR_EDIT_TRANSACTION", raising=False)
        monkeypatch.delenv("ORCHESTRATOR_EDIT_ROOT", raising=False)
        # request_context must not suppress the raised HTTPException (a bare MagicMock __exit__
        # returns truthy and would swallow it).
        mock_primitives.request_context = MagicMock(return_value=nullcontext())

        req = ChatRequest(
            prompt="add a helper to utils.py",
            mock_mode=False,
            real_mode=True,
            force_mode="edit",
            force_role="coder_escalation",
        )

        # Mock the pre-dispatch pipeline stages so _handle_chat reaches the 8b2 edit branch;
        # the 412 fires there BEFORE any model call.
        with (
            patch("src.api.routes.chat._route_request", return_value=base_routing),
            patch("src.api.routes.chat._preprocess", return_value=None),
            patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
            patch("src.api.routes.chat._plan_review_gate", return_value=None),
            patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
            patch(
                "src.api.routes.chat._execute_vision_multimodal", new=AsyncMock(return_value=None)
            ),
            patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
            patch("src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _handle_chat(req, mock_state)

        assert exc_info.value.status_code == 412
        detail = str(exc_info.value.detail)
        assert "ORCHESTRATOR_EDIT_TRANSACTION" in detail
        assert "ORCHESTRATOR_EDIT_ROOT" in detail

    @pytest.mark.asyncio
    async def test_force_mode_edit_applies_transaction_when_flag_and_root_set(
        self, mock_state, mock_primitives, base_routing, monkeypatch, tmp_path
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        from src.api.models import ChatResponse

        edit_root = tmp_path / "task-root"
        edit_root.mkdir()
        target_file = edit_root / "calc.py"
        target_file.write_text("VALUE = 1\n")

        monkeypatch.setenv("ORCHESTRATOR_EDIT_TRANSACTION", "1")
        monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(edit_root))
        mock_primitives.request_context = MagicMock(return_value=nullcontext())

        req = ChatRequest(
            prompt="update calc.py",
            mock_mode=False,
            real_mode=True,
            force_mode="edit",
            force_role="coder_escalation",
        )

        def _fake_execute_direct(request, routing, primitives, state, start_time, initial_role):
            assert "Current file contents:" in request.prompt
            return ChatResponse(
                answer="<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>",
                turns=1,
                elapsed_seconds=0.01,
                mock_mode=False,
                real_mode=True,
                routed_to=str(initial_role),
                role_history=[str(initial_role)],
                routing_strategy=routing.routing_strategy,
                mode="edit",
            )

        with (
            patch("src.api.routes.chat._route_request", return_value=base_routing),
            patch("src.api.routes.chat._preprocess", return_value=None),
            patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
            patch("src.api.routes.chat._plan_review_gate", return_value=None),
            patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
            patch(
                "src.api.routes.chat._execute_vision_multimodal", new=AsyncMock(return_value=None)
            ),
            patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
            patch("src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)),
            patch(
                "src.api.routes.chat._execute_direct", side_effect=_fake_execute_direct
            ) as mock_execute_direct,
        ):
            result = await _handle_chat(req, mock_state)

        assert mock_execute_direct.call_count == 1
        assert result.mode == "edit"
        assert result.answer.startswith("edit transaction applied: 1 write(s), 0 delete(s)")
        assert target_file.read_text() == "VALUE = 2"

    @pytest.mark.asyncio
    async def test_force_mode_edit_runs_review_consult_when_feature_enabled(
        self, mock_state, mock_primitives, base_routing, monkeypatch, tmp_path
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        from src.api.models import ChatResponse
        from src.features import reset_features

        edit_root = tmp_path / "task-root"
        edit_root.mkdir()
        target_file = edit_root / "calc.py"
        target_file.write_text("VALUE = 1\n")

        monkeypatch.setenv("ORCHESTRATOR_EDIT_TRANSACTION", "1")
        monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(edit_root))
        monkeypatch.setenv("ORCHESTRATOR_FEATURE_REVIEW_BEFORE_COMMIT_CONSULT", "1")
        monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(tmp_path / "runtime_flags.json"))
        reset_features()
        mock_primitives.request_context = MagicMock(return_value=nullcontext())

        req = ChatRequest(
            prompt="update calc.py",
            mock_mode=False,
            real_mode=True,
            force_mode="edit",
            force_role="coder_escalation",
        )
        base_routing.routing_decision = ["coder_escalation"]
        drafts = [
            "<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>",
            "<<<FILE: calc.py>>>\nVALUE = 3\n<<<END>>>",
        ]

        def _fake_execute_direct(request, routing, primitives, state, start_time, initial_role):
            assert "Current file contents:" in request.prompt
            return ChatResponse(
                answer=drafts.pop(0),
                turns=1,
                elapsed_seconds=0.01,
                mock_mode=False,
                real_mode=True,
                routed_to=str(initial_role),
                role_history=[str(initial_role)],
                routing_strategy=routing.routing_strategy,
                mode="edit",
            )

        def _fake_consult(**kwargs):
            assert kwargs["consultant_role"] == "architect_critic"  # ARCHSWAP-20260927: review follows the reviewer binding
            assert kwargs["requester_role"] == "coder_escalation"
            assert kwargs["skill"] == "review_before_commit"
            assert "VALUE = 2" in kwargs["context"]
            return {
                "risks": ["wrong requested value"],
                "blocking_issues": ["final value must be 3"],
                "confidence": 0.9,
                "recommended_delta": "write VALUE = 3",
            }, {"schema_hash": "schema123"}

        try:
            with (
                patch("src.api.routes.chat._route_request", return_value=base_routing),
                patch("src.api.routes.chat._preprocess", return_value=None),
                patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
                patch("src.api.routes.chat._plan_review_gate", return_value=None),
                patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
                patch(
                    "src.api.routes.chat._execute_vision_multimodal",
                    new=AsyncMock(return_value=None),
                ),
                patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
                patch("src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)),
                patch(
                    "src.api.routes.chat._execute_direct", side_effect=_fake_execute_direct
                ) as mock_execute_direct,
                patch(
                    "src.orchestration.consultation.consult", side_effect=_fake_consult
                ) as mock_consult,
            ):
                result = await _handle_chat(req, mock_state)
        finally:
            reset_features()

        assert mock_execute_direct.call_count == 2
        assert mock_consult.call_count == 1
        assert result.mode == "edit"
        assert result.answer.startswith("edit transaction applied: 1 write(s), 0 delete(s)")
        assert target_file.read_text() == "VALUE = 3"
        events = result.delegation_diagnostics["edit_transaction_consult_events"]
        assert events[0]["success"] is True
        assert events[0]["rerun_requested"] is True
        assert events[0]["schema_hash"] == "schema123"

    @pytest.mark.asyncio
    async def test_force_mode_edit_targeted_gate_skips_plain_edit(
        self, mock_state, mock_primitives, base_routing, monkeypatch, tmp_path
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        from src.api.models import ChatResponse
        from src.features import reset_features

        edit_root = tmp_path / "task-root"
        edit_root.mkdir()
        target_file = edit_root / "calc.py"
        target_file.write_text("VALUE = 1\n")

        monkeypatch.setenv("ORCHESTRATOR_EDIT_TRANSACTION", "1")
        monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(edit_root))
        monkeypatch.setenv("ORCHESTRATOR_FEATURE_REVIEW_BEFORE_COMMIT_CONSULT", "1")
        monkeypatch.setenv("ORCHESTRATOR_FEATURE_REVIEW_BEFORE_COMMIT_TARGETED_GATE", "1")
        monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(tmp_path / "runtime_flags.json"))
        reset_features()
        mock_primitives.request_context = MagicMock(return_value=nullcontext())

        req = ChatRequest(
            prompt="update calc.py",
            mock_mode=False,
            real_mode=True,
            force_mode="edit",
            force_role="coder_escalation",
        )
        base_routing.routing_decision = ["coder_escalation"]

        def _fake_execute_direct(request, routing, primitives, state, start_time, initial_role):
            assert "Current file contents:" in request.prompt
            return ChatResponse(
                answer="<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>",
                turns=1,
                elapsed_seconds=0.01,
                mock_mode=False,
                real_mode=True,
                routed_to=str(initial_role),
                role_history=[str(initial_role)],
                routing_strategy=routing.routing_strategy,
                mode="edit",
            )

        try:
            with (
                patch("src.api.routes.chat._route_request", return_value=base_routing),
                patch("src.api.routes.chat._preprocess", return_value=None),
                patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
                patch("src.api.routes.chat._plan_review_gate", return_value=None),
                patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
                patch(
                    "src.api.routes.chat._execute_vision_multimodal",
                    new=AsyncMock(return_value=None),
                ),
                patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
                patch("src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)),
                patch(
                    "src.api.routes.chat._execute_direct", side_effect=_fake_execute_direct
                ) as mock_execute_direct,
                patch("src.orchestration.consultation.consult") as mock_consult,
            ):
                result = await _handle_chat(req, mock_state)
        finally:
            reset_features()

        assert mock_execute_direct.call_count == 1
        assert mock_consult.call_count == 0
        assert result.mode == "edit"
        assert target_file.read_text() == "VALUE = 2"
        events = result.delegation_diagnostics["edit_transaction_consult_events"]
        assert events[0]["skipped"] is True
        assert events[0]["reason"] == "targeted_gate_skip"


class TestHandleChatXmasCheapFirst:
    """X-MAS enforce should not suppress cheap-first for the cheap role itself."""

    @pytest.mark.asyncio
    async def test_xmas_enforce_to_cheap_role_still_allows_cheap_first(
        self, mock_state, mock_primitives
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        mock_primitives.request_context = MagicMock(return_value=nullcontext())
        xmas_meta = {
            "mode": "enforce",
            "applied": True,
            "suggested_role": "worker_general",
            "apply_reason": "evidence_quality_lift",
        }
        routing = RoutingResult(
            task_id="xmas-cheap",
            task_ir={"task_type": "chat", "objective": "solve"},
            use_mock=False,
            routing_decision=["worker_general"],
            routing_strategy="xmas_enforce:learned",
            xmas_meta=xmas_meta,
        )
        cheap_response = ChatResponse(
            answer="<answer>42</answer>",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=False,
            real_mode=True,
            routed_to="worker_general",
            role_history=["worker_general"],
            routing_strategy="cheap_first:xmas_enforce:learned",
            mode="direct",
        )
        cfg = SimpleNamespace(
            chat=SimpleNamespace(
                try_cheap_first_role="worker_general",
                try_cheap_first_phase="A",
            )
        )

        with (
            patch("src.api.routes.chat.features") as mock_features,
            patch("src.api.routes.chat.get_config", return_value=cfg),
            patch("src.api.routes.chat._route_request", return_value=routing),
            patch("src.api.routes.chat._preprocess", return_value=None),
            patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
            patch("src.api.routes.chat._plan_review_gate", return_value=None),
            patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
            patch(
                "src.api.routes.chat._execute_vision_multimodal", new=AsyncMock(return_value=None)
            ),
            patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
            patch("src.api.routes.chat._select_mode", return_value="direct"),
            patch(
                "src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=cheap_response)
            ) as mock_cheap,
        ):
            mock_features.return_value.script_interception = False
            result = await _handle_chat(
                ChatRequest(
                    prompt="Solve this arithmetic problem.", mock_mode=False, real_mode=True
                ),
                mock_state,
            )

        assert mock_cheap.await_count == 1
        assert result.routing_strategy == "cheap_first:xmas_enforce:learned"
        assert result.xmas_meta is xmas_meta

    @pytest.mark.asyncio
    async def test_xmas_enforce_to_non_cheap_role_keeps_cheap_first_bypass(
        self, mock_state, mock_primitives
    ):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock

        mock_primitives.request_context = MagicMock(return_value=nullcontext())
        xmas_meta = {
            "mode": "enforce",
            "applied": True,
            "suggested_role": "architect_general",
            "apply_reason": "evidence_quality_lift",
        }
        routing = RoutingResult(
            task_id="xmas-architect",
            task_ir={"task_type": "chat", "objective": "verify"},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="xmas_enforce:learned",
            xmas_meta=xmas_meta,
        )
        direct_response = ChatResponse(
            answer="<answer>valid</answer>",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=False,
            real_mode=True,
            routed_to="architect_general",
            role_history=["architect_general"],
            routing_strategy="xmas_enforce:learned",
            mode="direct",
        )
        cfg = SimpleNamespace(
            chat=SimpleNamespace(
                try_cheap_first_role="worker_general",
                try_cheap_first_phase="A",
            )
        )

        with (
            patch("src.api.routes.chat.features") as mock_features,
            patch("src.api.routes.chat.get_config", return_value=cfg),
            patch("src.api.routes.chat._route_request", return_value=routing),
            patch("src.api.routes.chat._preprocess", return_value=None),
            patch("src.api.routes.chat._init_primitives", return_value=mock_primitives),
            patch("src.api.routes.chat._plan_review_gate", return_value=None),
            patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
            patch(
                "src.api.routes.chat._execute_vision_multimodal", new=AsyncMock(return_value=None)
            ),
            patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
            patch("src.api.routes.chat._select_mode", return_value="direct"),
            patch(
                "src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)
            ) as mock_cheap,
            patch("src.api.routes.chat._execute_direct", return_value=direct_response),
        ):
            mock_features.return_value.script_interception = False
            result = await _handle_chat(
                ChatRequest(prompt="Verify this reasoning.", mock_mode=False, real_mode=True),
                mock_state,
            )

        assert mock_cheap.await_count == 0
        assert result.routing_strategy == "xmas_enforce:learned"
        assert result.xmas_meta is xmas_meta
