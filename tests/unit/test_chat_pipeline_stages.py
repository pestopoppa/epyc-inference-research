"""Comprehensive tests for chat pipeline stages.

Tests for src/api/routes/chat_pipeline/stages.py covering:
- Stage 4: Mock mode (_execute_mock)
- Stage 6: Vision preprocessing (_execute_vision)
- Stage 7: Delegated mode (_execute_delegated)
- Stage 7.5: Proactive delegation (_execute_proactive)
- Stage 8: ReAct mode (_execute_react)
- Stage 9: Direct mode (_execute_direct)
- Error annotation (_annotate_error)
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_pipeline.stages import (
    _annotate_error,
    _execute_mock,
    _execute_react,
)
from src.api.routes.chat_pipeline.vision_stage import _execute_vision
from src.api.routes.chat_pipeline.delegation_stage import _execute_delegated
from src.api.routes.chat_pipeline.proactive_stage import _execute_proactive, _parse_plan_steps
from src.api.routes.chat_pipeline.direct_stage import (
    _execute_direct,
    _restore_stripped_answer_stop,
)
from src.api.routes.chat_utils import RoutingResult

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_state(mock_app_state):
    """Reuse shared app-state fixture under the local name expected by tests."""
    return mock_app_state


@pytest.fixture
def mock_primitives(mock_llm_primitives):
    """Reuse shared primitives fixture under the local name expected by tests."""
    return mock_llm_primitives


@pytest.fixture
def basic_request():
    """Create a basic ChatRequest for testing."""
    return ChatRequest(prompt="Test prompt", real_mode=True)


@pytest.fixture
def basic_routing():
    """Create a basic RoutingResult for testing."""
    return RoutingResult(
        task_id="test-task-001",
        task_ir={"task_type": "chat"},
        use_mock=False,
        routing_decision=["frontdoor"],
        routing_strategy="deterministic",
    )


# ─────────────────────────────────────────────────────────────────────────────
# _parse_plan_steps (pure logic)
# ─────────────────────────────────────────────────────────────────────────────


class TestParsePlanSteps:
    """Tests for _parse_plan_steps JSON parsing."""

    def test_valid_json_array(self):
        """Valid JSON array with required fields parses correctly."""
        raw = '[{"id": "s1", "action": "do something"}]'
        result = _parse_plan_steps(raw)
        assert len(result) == 1
        assert result[0]["id"] == "s1"
        assert result[0]["action"] == "do something"

    def test_multiple_steps(self):
        """Multiple steps parse correctly."""
        raw = """[
            {"id": "s1", "action": "first step"},
            {"id": "s2", "action": "second step", "depends_on": ["s1"]}
        ]"""
        result = _parse_plan_steps(raw)
        assert len(result) == 2
        assert result[0]["id"] == "s1"
        assert result[1]["depends_on"] == ["s1"]

    def test_defaults_applied(self):
        """Default values for actor, depends_on, outputs applied."""
        raw = '[{"id": "s1", "action": "test"}]'
        result = _parse_plan_steps(raw)
        assert result[0]["actor"] == "worker"
        assert result[0]["depends_on"] == []
        assert result[0]["outputs"] == []

    def test_markdown_fence_json(self):
        """JSON wrapped in markdown code fence."""
        raw = '```json\n[{"id": "s1", "action": "test"}]\n```'
        result = _parse_plan_steps(raw)
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    def test_markdown_fence_no_language(self):
        """JSON wrapped in plain markdown code fence."""
        raw = '```\n[{"id": "s1", "action": "test"}]\n```'
        result = _parse_plan_steps(raw)
        assert len(result) == 1

    def test_trailing_comma_fixed(self):
        """Trailing commas before ] are fixed."""
        raw = '[{"id": "s1", "action": "test"},]'
        result = _parse_plan_steps(raw)
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        """Invalid JSON returns empty list."""
        raw = "not valid json at all"
        result = _parse_plan_steps(raw)
        assert result == []

    def test_non_array_returns_empty(self):
        """JSON object (not array) returns empty list."""
        raw = '{"id": "s1", "action": "test"}'
        result = _parse_plan_steps(raw)
        assert result == []

    def test_missing_id_skipped(self):
        """Steps missing 'id' field are skipped."""
        raw = '[{"action": "no id"}, {"id": "s1", "action": "has id"}]'
        result = _parse_plan_steps(raw)
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    def test_missing_action_skipped(self):
        """Steps missing 'action' field are skipped."""
        raw = '[{"id": "s1"}, {"id": "s2", "action": "has action"}]'
        result = _parse_plan_steps(raw)
        assert len(result) == 1
        assert result[0]["id"] == "s2"

    def test_non_dict_items_skipped(self):
        """Non-dict items in array are skipped."""
        raw = '["string", {"id": "s1", "action": "test"}, 123]'
        result = _parse_plan_steps(raw)
        assert len(result) == 1

    def test_whitespace_handling(self):
        """Leading/trailing whitespace handled."""
        raw = '  \n\n [{"id": "s1", "action": "test"}]  \n  '
        result = _parse_plan_steps(raw)
        assert len(result) == 1

    def test_empty_array(self):
        """Empty array returns empty list."""
        raw = "[]"
        result = _parse_plan_steps(raw)
        assert result == []

    def test_preserves_custom_fields(self):
        """Custom fields in steps are preserved."""
        raw = '[{"id": "s1", "action": "test", "custom": "value", "actor": "coder"}]'
        result = _parse_plan_steps(raw)
        assert result[0]["custom"] == "value"
        assert result[0]["actor"] == "coder"


# ─────────────────────────────────────────────────────────────────────────────
# _annotate_error (pure logic)
# ─────────────────────────────────────────────────────────────────────────────


class TestAnnotateError:
    """Tests for _annotate_error response annotation."""

    def test_no_answer_unchanged(self):
        """Empty answer leaves response unchanged."""
        response = ChatResponse(
            answer="",
            turns=1,
            tokens_used=0,
            elapsed_seconds=0.1,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code is None
        assert result.error_detail is None

    def test_normal_answer_unchanged(self):
        """Normal answer without error patterns unchanged."""
        response = ChatResponse(
            answer="This is a valid response.",
            turns=1,
            tokens_used=100,
            elapsed_seconds=0.5,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code is None
        assert result.error_detail is None

    def test_timeout_error_504(self):
        """Timeout errors set error_code 504."""
        response = ChatResponse(
            answer="[ERROR: Request timed out after 30s]",
            turns=1,
            tokens_used=0,
            elapsed_seconds=30.0,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code == 504
        assert "timed out" in result.error_detail.lower()

    def test_timeout_uppercase_504(self):
        """TIMEOUT keyword also triggers 504."""
        response = ChatResponse(
            answer="[ERROR: TIMEOUT exceeded]",
            turns=1,
            tokens_used=0,
            elapsed_seconds=30.0,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code == 504

    def test_backend_error_502(self):
        """Backend errors set error_code 502."""
        response = ChatResponse(
            answer="[ERROR: Backend connection failed]",
            turns=1,
            tokens_used=0,
            elapsed_seconds=0.5,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code == 502
        assert "Backend" in result.error_detail

    def test_failed_error_502(self):
        """Generic failed errors set error_code 502."""
        response = ChatResponse(
            answer="[ERROR: LLM call failed]",
            turns=1,
            tokens_used=0,
            elapsed_seconds=0.5,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code == 502

    def test_generic_error_500(self):
        """Generic [ERROR: ...] sets error_code 500."""
        response = ChatResponse(
            answer="[ERROR: Something went wrong]",
            turns=1,
            tokens_used=0,
            elapsed_seconds=0.5,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code == 500
        assert result.error_detail == "[ERROR: Something went wrong]"

    def test_failed_prefix_500(self):
        """[FAILED: ...] prefix sets error_code 500."""
        response = ChatResponse(
            answer="[FAILED: Task could not complete]",
            turns=1,
            tokens_used=0,
            elapsed_seconds=1.0,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code == 500

    def test_error_in_middle_not_detected(self):
        """Error patterns not at start are not detected."""
        response = ChatResponse(
            answer="Here is your answer. [ERROR: This is not an error]",
            turns=1,
            tokens_used=50,
            elapsed_seconds=0.5,
            mock_mode=False,
            real_mode=True,
        )
        result = _annotate_error(response)
        assert result.error_code is None


# ─────────────────────────────────────────────────────────────────────────────
# _execute_mock (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteMock:
    """Tests for _execute_mock stage."""

    def test_basic_mock_response(self, mock_state):
        """Basic mock mode returns simulated response."""
        request = ChatRequest(prompt="Hello world", real_mode=False)
        routing = RoutingResult(
            task_id="mock-001",
            task_ir={},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        start_time = time.perf_counter()

        result = _execute_mock(request, routing, mock_state, start_time)

        assert result.mock_mode is True
        assert result.real_mode is False
        assert "[MOCK]" in result.answer
        assert "Hello world" in result.answer
        assert result.turns == 1
        assert result.tokens_used == 0
        assert result.mode == "mock"

    def test_mock_with_context(self, mock_state):
        """Mock mode includes context length in answer."""
        request = ChatRequest(
            prompt="Question?",
            context="A" * 500,
            real_mode=False,
        )
        routing = RoutingResult(
            task_id="mock-002",
            task_ir={},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        start_time = time.perf_counter()

        result = _execute_mock(request, routing, mock_state, start_time)

        assert "500 chars of context" in result.answer

    def test_mock_increments_state(self, mock_state):
        """Mock mode increments request counter."""
        request = ChatRequest(prompt="Test", real_mode=False)
        routing = RoutingResult(
            task_id="mock-003",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        start_time = time.perf_counter()

        _execute_mock(request, routing, mock_state, start_time)

        mock_state.increment_request.assert_called_once_with(mock_mode=True, turns=1)

    def test_mock_logs_completion(self, mock_state):
        """Mock mode logs task completion."""
        request = ChatRequest(prompt="Test", real_mode=False)
        routing = RoutingResult(
            task_id="mock-004",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.stages.score_completed_task"):
            _execute_mock(request, routing, mock_state, start_time)

        mock_state.progress_logger.log_task_completed.assert_called_once()

    def test_mock_no_progress_logger(self):
        """Mock mode handles missing progress_logger."""
        state = MagicMock()
        state.progress_logger = None
        state.increment_request = MagicMock()

        request = ChatRequest(prompt="Test", real_mode=False)
        routing = RoutingResult(
            task_id="mock-005",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        start_time = time.perf_counter()

        result = _execute_mock(request, routing, state, start_time)

        assert result.mock_mode is True

    def test_mock_truncates_long_prompt(self, mock_state):
        """Mock mode truncates prompt display to 100 chars."""
        long_prompt = "X" * 200
        request = ChatRequest(prompt=long_prompt, real_mode=False)
        routing = RoutingResult(
            task_id="mock-006",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        start_time = time.perf_counter()

        result = _execute_mock(request, routing, mock_state, start_time)

        # Should only include first 100 chars plus "..."
        assert len(result.answer) < len(long_prompt) + 50


# ─────────────────────────────────────────────────────────────────────────────
# _execute_vision (Stage 6)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteVision:
    """Tests for _execute_vision stage."""

    @pytest.mark.asyncio
    async def test_returns_none_for_non_real_mode(self, mock_primitives, mock_state):
        """Returns None for non-real-mode requests."""
        request = ChatRequest(
            prompt="Describe this image",
            image_path="/path/to/image.png",
            real_mode=False,
        )
        routing = RoutingResult(
            task_id="vision-001",
            task_ir={},
            use_mock=False,
            routing_decision=["worker_vision"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        result = await _execute_vision(request, routing, mock_primitives, mock_state, start_time)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_no_vision_input(self, mock_primitives, mock_state):
        """Returns None for requests without vision input."""
        request = ChatRequest(prompt="No image here", real_mode=True)
        routing = RoutingResult(
            task_id="vision-002",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        result = await _execute_vision(request, routing, mock_primitives, mock_state, start_time)

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_image_path_preprocessing(self, mock_primitives, mock_state):
        """Successful preprocessing with image_path returns None (fall through)."""
        request = ChatRequest(
            prompt="Describe this",
            image_path="/path/to/test.png",
            real_mode=True,
        )
        routing = RoutingResult(
            task_id="vision-003",
            task_ir={},
            use_mock=False,
            routing_decision=["worker_vision"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.document_result = MagicMock()
        mock_result.document_result.sections = ["section1"]
        mock_result.document_result.figures = []

        with patch("src.services.document_preprocessor.DocumentPreprocessor") as mock_prep:
            mock_prep.return_value.preprocess_file = AsyncMock(return_value=mock_result)
            result = await _execute_vision(
                request, routing, mock_primitives, mock_state, start_time
            )

        assert result is None
        assert routing.document_result is not None

    @pytest.mark.asyncio
    async def test_successful_base64_preprocessing(self, mock_primitives, mock_state):
        """Successful preprocessing with base64 image returns None (fall through)."""
        request = ChatRequest(
            prompt="Describe this",
            image_base64="aGVsbG8gd29ybGQ=",
            real_mode=True,
        )
        routing = RoutingResult(
            task_id="vision-004",
            task_ir={},
            use_mock=False,
            routing_decision=["worker_vision"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.document_result = MagicMock()
        mock_result.document_result.sections = []
        mock_result.document_result.figures = ["fig1"]

        with patch("src.services.document_preprocessor.DocumentPreprocessor") as mock_prep:
            mock_prep.return_value.preprocess = AsyncMock(return_value=mock_result)
            result = await _execute_vision(
                request, routing, mock_primitives, mock_state, start_time
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_files_preprocessing(self, mock_primitives, mock_state):
        """Successful preprocessing with files list returns None (fall through)."""
        request = ChatRequest(
            prompt="Analyze these",
            files=["/path/a.pdf", "/path/b.pdf"],
            real_mode=True,
        )
        routing = RoutingResult(
            task_id="vision-005",
            task_ir={},
            use_mock=False,
            routing_decision=["worker_vision"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.document_result = MagicMock()
        mock_result.document_result.sections = []
        mock_result.document_result.figures = []

        with patch("src.services.document_preprocessor.DocumentPreprocessor") as mock_prep:
            mock_prep.return_value.preprocess = AsyncMock(return_value=mock_result)
            result = await _execute_vision(
                request, routing, mock_primitives, mock_state, start_time
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_preprocessing_failure_adds_context_note(self, mock_primitives, mock_state):
        """Failed preprocessing injects context note and returns None."""
        request = ChatRequest(
            prompt="Describe this",
            image_path="/path/to/bad.png",
            real_mode=True,
        )
        routing = RoutingResult(
            task_id="vision-006",
            task_ir={},
            use_mock=False,
            routing_decision=["worker_vision"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "OCR failed"
        mock_result.document_result = None

        with patch("src.services.document_preprocessor.DocumentPreprocessor") as mock_prep:
            mock_prep.return_value.preprocess_file = AsyncMock(return_value=mock_result)
            result = await _execute_vision(
                request, routing, mock_primitives, mock_state, start_time
            )

        assert result is None
        assert "[IMAGE:" in request.context
        assert "Document pipeline failed" in request.context

    @pytest.mark.asyncio
    async def test_preprocessing_exception_adds_context_note(self, mock_primitives, mock_state):
        """Exception during preprocessing injects context note and returns None."""
        request = ChatRequest(
            prompt="Describe this",
            image_path="/path/to/crash.png",
            real_mode=True,
        )
        routing = RoutingResult(
            task_id="vision-007",
            task_ir={},
            use_mock=False,
            routing_decision=["worker_vision"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.services.document_preprocessor.DocumentPreprocessor") as mock_prep:
            mock_prep.return_value.preprocess_file = AsyncMock(side_effect=RuntimeError("Crash"))
            result = await _execute_vision(
                request, routing, mock_primitives, mock_state, start_time
            )

        assert result is None
        assert "Document pipeline failed" in request.context


# ─────────────────────────────────────────────────────────────────────────────
# _execute_delegated (Stage 7)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteDelegated:
    """Tests for _execute_delegated stage."""

    def test_returns_none_for_non_real_mode(self, mock_primitives, mock_state):
        """Returns None for non-real-mode requests."""
        request = ChatRequest(prompt="Test", real_mode=False)
        routing = RoutingResult(
            task_id="deleg-001",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        result = _execute_delegated(
            request,
            routing,
            mock_primitives,
            mock_state,
            start_time,
            initial_role="architect_general",
            execution_mode="direct",
        )

        assert result is None

    def test_returns_none_for_non_architect_without_flag(self, mock_primitives, mock_state):
        """Returns None for non-architect role when not in delegated mode."""
        request = ChatRequest(prompt="Test", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-002",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = False
            result = _execute_delegated(
                request,
                routing,
                mock_primitives,
                mock_state,
                start_time,
                initial_role="frontdoor",
                execution_mode="direct",
            )

        assert result is None

    def test_architect_delegation_success(self, mock_primitives, mock_state):
        """Successful architect delegation returns ChatResponse."""
        request = ChatRequest(prompt="Design a system", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-003",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        delegation_stats = {
            "loops": 2,
            "phases": [
                {"phase": "A", "loop": 1, "ms": 100},
                {"phase": "B", "loop": 1, "ms": 200, "delegate_to": "coder"},
            ],
            "tools_used": 1,
            "tools_called": ["tool1"],
        }

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = True
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer"
            ) as mock_deleg:
                mock_deleg.return_value = ("Delegated answer", delegation_stats)
                with patch("src.api.routes.chat_pipeline.delegation_stage.score_completed_task"):
                    result = _execute_delegated(
                        request,
                        routing,
                        mock_primitives,
                        mock_state,
                        start_time,
                        initial_role="architect_general",
                        execution_mode="direct",
                    )

        assert result is not None
        assert result.answer == "Delegated answer"
        assert result.mode == "delegated"
        assert result.turns == 3  # 1 + loops
        assert "coder" in result.role_history

    def test_delegation_exception_returns_none(self, mock_primitives, mock_state):
        """Exception during delegation returns None (fall through)."""
        request = ChatRequest(prompt="Crash", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-004",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = True
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer"
            ) as mock_deleg:
                mock_deleg.side_effect = RuntimeError("Delegation crash")
                result = _execute_delegated(
                    request,
                    routing,
                    mock_primitives,
                    mock_state,
                    start_time,
                    initial_role="architect_general",
                    execution_mode="direct",
                )

        assert result is None

    def test_delegation_lock_timeout_returns_structured_error(self, mock_primitives, mock_state):
        """Lock-timeout style delegation errors return explicit delegated diagnostics."""
        request = ChatRequest(prompt="Timeout", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-004b",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = True
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer"
            ) as mock_deleg:
                mock_deleg.side_effect = RuntimeError(
                    f"Inference lock timeout (role={_RETIRED_ARCHITECT_ROLE}, mode=exclusive)"
                )
                result = _execute_delegated(
                    request,
                    routing,
                    mock_primitives,
                    mock_state,
                    start_time,
                    initial_role="architect_general",
                    execution_mode="direct",
                )

        assert result is not None
        assert result.mode == "delegated"
        assert result.answer.startswith("[ERROR: Delegated inference timed out/cancelled")
        assert result.delegation_diagnostics.get("break_reason") == "pre_delegation_lock_timeout"
        assert result.delegation_success is False

    def test_empty_answer_returns_none(self, mock_primitives, mock_state):
        """Empty answer from delegation returns None (fall through)."""
        request = ChatRequest(prompt="Empty", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-005",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = True
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer"
            ) as mock_deleg:
                mock_deleg.return_value = ("", {"loops": 0, "phases": []})
                result = _execute_delegated(
                    request,
                    routing,
                    mock_primitives,
                    mock_state,
                    start_time,
                    initial_role="architect_general",
                    execution_mode="direct",
                )

        assert result is None

    def test_error_answer_logs_failed_completion(self, mock_primitives, mock_state):
        """Delegated error answers must not be logged as successful completions."""
        request = ChatRequest(prompt="Error", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-error",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = True
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer"
            ) as mock_deleg:
                mock_deleg.return_value = (
                    "[ERROR: Delegation exhausted]",
                    {"loops": 0, "phases": [], "delegation_events": []},
                )
                with patch("src.api.routes.chat_pipeline.delegation_stage.score_completed_task"):
                    result = _execute_delegated(
                        request,
                        routing,
                        mock_primitives,
                        mock_state,
                        start_time,
                        initial_role="architect_general",
                        execution_mode="direct",
                    )

        assert result is not None
        assert result.answer == "[ERROR: Delegation exhausted]"
        call = mock_state.progress_logger.log_task_completed.call_args
        assert call.kwargs["success"] is False

    def test_delegated_completion_logs_cache_counters(self, mock_primitives, mock_state):
        """Delegated completions expose cache hit/miss counters for bake telemetry."""
        request = ChatRequest(prompt="Use delegation", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-cache-counters",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        stats = {
            "loops": 1,
            "phases": [{"loop": 0, "phase": "B", "ms": 12, "delegate_to": "worker_general"}],
            "delegation_events": [],
            "delegation_cache_lookups": 3,
            "delegation_cache_hits": 1,
            "delegation_cache_misses": 2,
        }
        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = True
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer",
                return_value=("Delegated answer", stats),
            ):
                with patch("src.api.routes.chat_pipeline.delegation_stage.score_completed_task"):
                    result = _execute_delegated(
                        request,
                        routing,
                        mock_primitives,
                        mock_state,
                        start_time,
                        initial_role="architect_general",
                        execution_mode="direct",
                    )

        assert result is not None
        call = mock_state.progress_logger.log_task_completed.call_args
        meta = call.kwargs["completion_meta"]
        assert meta["delegation_cache_lookups"] == 3
        assert meta["delegation_cache_hits"] == 1
        assert meta["delegation_cache_misses"] == 2
        assert meta["delegation_cache_hit_rate"] == pytest.approx(1 / 3, abs=0.0001)
        assert result.delegation_diagnostics["delegation_cache_lookups"] == 3
        assert result.delegation_diagnostics["delegation_cache_hits"] == 1
        assert result.delegation_diagnostics["delegation_cache_misses"] == 2

    def test_forced_delegated_mode(self, mock_primitives, mock_state):
        """execution_mode='delegated' forces delegation for non-architect."""
        request = ChatRequest(prompt="Force delegated", real_mode=True)
        routing = RoutingResult(
            task_id="deleg-006",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.delegation_stage.features") as mock_features:
            mock_features.return_value.architect_delegation = False  # Should still work
            with patch(
                "src.api.routes.chat_pipeline.delegation_stage._architect_delegated_answer"
            ) as mock_deleg:
                mock_deleg.return_value = ("Forced delegation", {"loops": 0, "phases": []})
                with patch("src.api.routes.chat_pipeline.delegation_stage.score_completed_task"):
                    result = _execute_delegated(
                        request,
                        routing,
                        mock_primitives,
                        mock_state,
                        start_time,
                        initial_role="frontdoor",
                        execution_mode="delegated",
                    )

        assert result is not None
        assert result.mode == "delegated"


# ─────────────────────────────────────────────────────────────────────────────
# _execute_proactive (Stage 7.5)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteProactive:
    """Tests for _execute_proactive stage."""

    @pytest.mark.asyncio
    async def test_returns_none_when_feature_disabled(self, mock_primitives, mock_state):
        """Returns None when parallel_execution feature disabled."""
        request = ChatRequest(prompt="Complex task", real_mode=True)
        routing = RoutingResult(
            task_id="proactive-001",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.proactive_stage.features") as mock_features:
            mock_features.return_value.parallel_execution = False
            result = await _execute_proactive(
                request, routing, mock_primitives, mock_state, start_time
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_non_complex_task(self, mock_primitives, mock_state):
        """Returns None for non-COMPLEX classified tasks."""
        request = ChatRequest(prompt="Simple question", real_mode=True)
        routing = RoutingResult(
            task_id="proactive-002",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.proactive_stage.features") as mock_features:
            mock_features.return_value.parallel_execution = True
            with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
                from src.proactive_delegation import TaskComplexity

                mock_classify.return_value = (TaskComplexity.SIMPLE, {})
                result = await _execute_proactive(
                    request, routing, mock_primitives, mock_state, start_time
                )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_architect_role(self, mock_primitives, mock_state):
        """Returns None when architect already selected (avoids double-entry)."""
        request = ChatRequest(prompt="Complex task", real_mode=True)
        routing = RoutingResult(
            task_id="proactive-003",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.proactive_stage.features") as mock_features:
            mock_features.return_value.parallel_execution = True
            with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
                from src.proactive_delegation import TaskComplexity

                mock_classify.return_value = (TaskComplexity.COMPLEX, {})
                result = await _execute_proactive(
                    request, routing, mock_primitives, mock_state, start_time
                )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_plan_too_short(self, mock_primitives, mock_state):
        """Returns None when plan has fewer than 2 steps."""
        request = ChatRequest(prompt="Implement a complex feature", real_mode=True)
        routing = RoutingResult(
            task_id="proactive-004",
            task_ir={"task_type": "code"},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.proactive_stage.features") as mock_features:
            mock_features.return_value.parallel_execution = True
            with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
                from src.proactive_delegation import TaskComplexity

                mock_classify.return_value = (TaskComplexity.COMPLEX, {})
                mock_primitives.llm_call.return_value = '[{"id": "s1", "action": "only one step"}]'
                result = await _execute_proactive(
                    request, routing, mock_primitives, mock_state, start_time
                )

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_proactive_delegation(self, mock_primitives, mock_state):
        """Successful proactive delegation returns ChatResponse."""
        request = ChatRequest(prompt="Build a complex system", real_mode=True)
        routing = RoutingResult(
            task_id="proactive-005",
            task_ir={"task_type": "code"},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_deleg_result = MagicMock()
        mock_deleg_result.aggregated_output = "Proactive result"
        mock_deleg_result.all_approved = True
        mock_deleg_result.subtask_results = [MagicMock(), MagicMock()]
        mock_deleg_result.roles_used = ["coder", "architect"]

        with patch("src.api.routes.chat_pipeline.proactive_stage.features") as mock_features:
            mock_features.return_value.parallel_execution = True
            with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
                from src.proactive_delegation import TaskComplexity

                mock_classify.return_value = (TaskComplexity.COMPLEX, {})
                mock_primitives.llm_call.return_value = """[
                    {"id": "s1", "action": "first step"},
                    {"id": "s2", "action": "second step", "depends_on": ["s1"]}
                ]"""
                with patch("src.proactive_delegation.ProactiveDelegator") as mock_pd:
                    mock_pd.return_value.delegate = AsyncMock(return_value=mock_deleg_result)
                    with patch("src.api.routes.chat_pipeline.proactive_stage.score_completed_task"):
                        result = await _execute_proactive(
                            request, routing, mock_primitives, mock_state, start_time
                        )

        assert result is not None
        assert result.answer == "Proactive result"
        assert result.mode == "proactive"
        assert result.turns == 3  # 1 + 2 subtasks

    @pytest.mark.asyncio
    async def test_proactive_delegation_exception(self, mock_primitives, mock_state):
        """Exception during delegation execution returns None."""
        request = ChatRequest(prompt="Build a complex system", real_mode=True)
        routing = RoutingResult(
            task_id="proactive-006",
            task_ir={"task_type": "code"},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.proactive_stage.features") as mock_features:
            mock_features.return_value.parallel_execution = True
            with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
                from src.proactive_delegation import TaskComplexity

                mock_classify.return_value = (TaskComplexity.COMPLEX, {})
                mock_primitives.llm_call.return_value = """[
                    {"id": "s1", "action": "first"},
                    {"id": "s2", "action": "second"}
                ]"""
                with patch("src.proactive_delegation.ProactiveDelegator") as mock_pd:
                    mock_pd.return_value.delegate = AsyncMock(side_effect=RuntimeError("Crash"))
                    result = await _execute_proactive(
                        request, routing, mock_primitives, mock_state, start_time
                    )

        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _execute_react (Stage 8)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteReact:
    """Tests for _execute_react stage."""

    def test_returns_none_for_non_real_mode(self, mock_primitives, mock_state):
        """Returns None for non-real-mode requests."""
        request = ChatRequest(prompt="Test", real_mode=False)
        routing = RoutingResult(
            task_id="react-001",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        result = _execute_react(
            request,
            routing,
            mock_primitives,
            mock_state,
            start_time,
            initial_role="frontdoor",
        )

        assert result is None

    def test_successful_react_mode(self, mock_primitives, mock_state):
        """Successful ReAct mode returns ChatResponse."""
        request = ChatRequest(prompt="Use tools to solve", real_mode=True)
        routing = RoutingResult(
            task_id="react-002",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_repl = MagicMock()
        mock_repl.get_prompt.return_value = "prompt"
        mock_result = MagicMock()
        mock_result.is_final = True
        mock_result.final_answer = "ReAct result"
        mock_result.output = ""
        mock_result.error = None
        mock_repl.execute.return_value = mock_result
        mock_repl._tool_invocations = 2
        mock_repl.tool_registry = None
        with patch("src.api.routes.chat_pipeline.stages.REPLEnvironment", return_value=mock_repl):
            with patch("src.api.routes.chat_pipeline.stages._truncate_looped_answer") as mock_trunc:
                mock_trunc.return_value = "ReAct result"
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    with patch("src.api.routes.chat_pipeline.stages.score_completed_task"):
                        result = _execute_react(
                            request,
                            routing,
                            mock_primitives,
                            mock_state,
                            start_time,
                            initial_role="frontdoor",
                        )

        assert result is not None
        assert result.answer == "ReAct result"
        assert result.mode == "react"
        assert result.tools_used == 2

    def test_react_exception_returns_none(self, mock_primitives, mock_state):
        """Exception during ReAct returns None (fall through)."""
        request = ChatRequest(prompt="Crash", real_mode=True)
        routing = RoutingResult(
            task_id="react-003",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        with patch("src.api.routes.chat_pipeline.stages.REPLEnvironment", side_effect=RuntimeError("REPL crash")):
            result = _execute_react(
                request,
                routing,
                mock_primitives,
                mock_state,
                start_time,
                initial_role="frontdoor",
            )

        assert result is None

    def test_empty_react_answer_returns_none(self, mock_primitives, mock_state):
        """Empty answer from ReAct returns None (fall through)."""
        request = ChatRequest(prompt="Empty", real_mode=True)
        routing = RoutingResult(
            task_id="react-004",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_repl = MagicMock()
        mock_repl.get_prompt.return_value = "prompt"
        mock_repl.execute.return_value = {}
        mock_repl.get_state.return_value = ""
        mock_repl._tool_invocations = 0
        mock_repl.tool_registry = None
        with patch("src.api.routes.chat_pipeline.stages.REPLEnvironment", return_value=mock_repl):
            result = _execute_react(
                request,
                routing,
                mock_primitives,
                mock_state,
                start_time,
                initial_role="frontdoor",
            )

        assert result is None

    def test_react_quality_escalation(self, mock_primitives, mock_state):
        """ReAct escalates on quality issues when generation_monitor enabled."""
        request = ChatRequest(prompt="Quality test", real_mode=True)
        routing = RoutingResult(
            task_id="react-005",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_repl = MagicMock()
        mock_repl.get_prompt.return_value = "prompt"
        mock_result = MagicMock()
        mock_result.is_final = True
        mock_result.final_answer = "Low quality answer"
        mock_result.output = ""
        mock_result.error = None
        mock_repl.execute.return_value = mock_result
        mock_repl._tool_invocations = 1
        mock_repl.tool_registry = None
        with patch("src.api.routes.chat_pipeline.stages.REPLEnvironment", return_value=mock_repl):
            with patch("src.api.routes.chat_pipeline.stages._truncate_looped_answer") as mock_trunc:
                mock_trunc.return_value = "Low quality answer"
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = True
                    with patch(
                        "src.api.routes.chat_pipeline.stages._detect_output_quality_issue"
                    ) as mock_detect:
                        mock_detect.return_value = "repetitive"
                        mock_primitives.llm_call.return_value = "Escalated answer"
                        with patch("src.api.routes.chat_pipeline.stages.score_completed_task"):
                            result = _execute_react(
                                request,
                                routing,
                                mock_primitives,
                                mock_state,
                                start_time,
                                initial_role="frontdoor",
                            )

        assert result is not None
        assert result.answer == "Escalated answer"


# ─────────────────────────────────────────────────────────────────────────────
# _execute_direct (Stage 9)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteDirect:
    """Tests for _execute_direct stage."""

    def test_restore_stripped_answer_stop_appends_missing_close_tag(self):
        """Direct stop sequence strips </answer>; restore it for API scoring."""
        assert (
            _restore_stripped_answer_stop("<answer>East Lancashire Railway")
            == "<answer>East Lancashire Railway</answer>"
        )

    def test_restore_stripped_answer_stop_leaves_other_answers_unchanged(self):
        assert (
            _restore_stripped_answer_stop("East Lancashire Railway")
            == "East Lancashire Railway"
        )
        assert (
            _restore_stripped_answer_stop("<answer>East Lancashire Railway</answer>")
            == "<answer>East Lancashire Railway</answer>"
        )

    def test_basic_direct_call(self, mock_primitives, mock_state):
        """Basic direct LLM call returns ChatResponse."""
        request = ChatRequest(prompt="Direct question", real_mode=True, n_probs=7)
        routing = RoutingResult(
            task_id="direct-001",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = "Direct answer"
        mock_primitives._last_inference_meta = {
            "completion_probabilities": [
                {"content": "D", "probs": [{"tok_str": "D", "prob": 0.8}]}
            ]
        }

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Direct answer"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                        result = _execute_direct(
                            request,
                            routing,
                            mock_primitives,
                            mock_state,
                            start_time,
                            initial_role="frontdoor",
                        )

        assert result is not None
        assert result.answer == "Direct answer"
        assert result.mode == "direct"
        assert result.completion_probabilities[0]["content"] == "D"
        mock_primitives.llm_call.assert_called_once()
        assert mock_primitives.llm_call.call_args.kwargs["n_probs"] == 7
        call = mock_state.progress_logger.log_task_completed.call_args
        assert call.kwargs["completion_meta"]["tokens_generated"] == 100
        assert call.kwargs["completion_meta"]["prompt_eval_ms"] == 50.0
        assert call.kwargs["completion_meta"]["generation_ms"] == 200.0
        assert call.kwargs["completion_meta"]["http_overhead_ms"] == 10.0

    def test_direct_call_clamps_default_token_budget(self, mock_primitives, mock_state):
        """Request max_tokens caps direct-stage generation budgets."""
        request = ChatRequest(
            prompt="Direct question",
            real_mode=True,
            max_tokens=1024,
        )
        routing = RoutingResult(
            task_id="direct-token-cap",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = "Direct answer"

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Direct answer"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                        result = _execute_direct(
                            request,
                            routing,
                            mock_primitives,
                            mock_state,
                            start_time,
                            initial_role="frontdoor",
                        )

        assert result.answer == "Direct answer"
        assert mock_primitives.llm_call.call_args.kwargs["n_tokens"] == 1024

    def test_direct_call_forwards_output_schema_as_json_schema(self, mock_primitives, mock_state):
        """Forced direct probes can use backend JSON-schema constrained output."""
        schema = {"type": "object", "required": ["decision"]}
        request = ChatRequest(
            prompt="Review this patch.",
            real_mode=True,
            force_role="frontdoor",
            force_mode="direct",
            max_tokens=256,
            output_schema=schema,
        )
        routing = RoutingResult(
            task_id="direct-json-schema",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="forced",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = '{"decision":"approve"}'

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = '{"decision":"approve"}'
            with patch("src.api.routes.chat_pipeline.direct_stage._quality_escalate") as mock_quality:
                mock_quality.side_effect = lambda answer, _prompt, _prims, role, **_kw: (answer, role)
                with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                    result = _execute_direct(
                        request,
                        routing,
                        mock_primitives,
                        mock_state,
                        start_time,
                        initial_role="frontdoor",
                    )

        assert result.answer == '{"decision":"approve"}'
        assert mock_primitives.llm_call.call_args.kwargs["json_schema"] == schema
        assert mock_primitives.llm_call.call_args.kwargs["n_tokens"] == 256

    def test_forced_direct_call_skips_output_formalizer(self, mock_primitives, mock_state):
        """Forced eval traffic measures the selected role, not a worker rewrite."""
        request = ChatRequest(
            prompt="Return exactly two bullet points.",
            real_mode=True,
            force_role="frontdoor",
            max_tokens=1024,
        )
        routing = RoutingResult(
            task_id="direct-force-no-formalizer",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="forced",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = "Direct answer"

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Direct answer"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_should:
                mock_should.return_value = (True, "exactly two bullet points")
                with patch("src.api.routes.chat_pipeline.direct_stage._formalize_output") as mock_fmt:
                    with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                        mock_features.return_value.generation_monitor = False
                        with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                            result = _execute_direct(
                                request,
                                routing,
                                mock_primitives,
                                mock_state,
                                start_time,
                                initial_role="frontdoor",
                            )

        assert result.answer == "Direct answer"
        mock_should.assert_not_called()
        mock_fmt.assert_not_called()
        assert mock_primitives.llm_call.call_count == 1

    def test_direct_with_context(self, mock_primitives, mock_state):
        """Direct call prepends context to prompt."""
        request = ChatRequest(
            prompt="Question",
            context="Background info",
            real_mode=True,
        )
        routing = RoutingResult(
            task_id="direct-002",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = "Answer"

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Answer"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                        _execute_direct(
                            request,
                            routing,
                            mock_primitives,
                            mock_state,
                            start_time,
                            initial_role="frontdoor",
                        )

        # Check that llm_call was called with context prepended
        call_args = mock_primitives.llm_call.call_args
        assert "Background info" in call_args[0][0]
        assert "Question" in call_args[0][0]

    def test_direct_retry_on_failure(self, mock_primitives, mock_state):
        """Direct call retries once on failure."""
        request = ChatRequest(prompt="Retry test", real_mode=True)
        routing = RoutingResult(
            task_id="direct-003",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        # First call fails, second succeeds
        mock_primitives.llm_call.side_effect = [
            RuntimeError("First fail"),
            "Retry success",
        ]

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Retry success"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                        result = _execute_direct(
                            request,
                            routing,
                            mock_primitives,
                            mock_state,
                            start_time,
                            initial_role="frontdoor",
                        )

        assert result.answer == "Retry success"
        assert mock_primitives.llm_call.call_count == 2

    def test_direct_retry_forwards_n_probs(self, mock_primitives, mock_state):
        """TD audit X5: the retry call must forward n_probs like the primary call.

        Before this fix, the retry branch (direct_stage.py, second llm_call
        in _execute_direct's except block) omitted n_probs entirely, so
        probability telemetry silently vanished on any retried call
        (confidence fell back to the binary proxy) even though the caller
        asked for it.
        """
        request = ChatRequest(prompt="Retry test", real_mode=True, n_probs=5)
        routing = RoutingResult(
            task_id="direct-005",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.side_effect = [
            RuntimeError("First fail"),
            "Retry success",
        ]

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Retry success"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                        result = _execute_direct(
                            request,
                            routing,
                            mock_primitives,
                            mock_state,
                            start_time,
                            initial_role="frontdoor",
                        )

        assert result.answer == "Retry success"
        assert mock_primitives.llm_call.call_count == 2
        retry_call = mock_primitives.llm_call.call_args_list[1]
        assert retry_call.kwargs["n_probs"] == 5

    def test_direct_both_calls_fail(self, mock_primitives, mock_state):
        """Direct call returns error message when both calls fail."""
        request = ChatRequest(prompt="Double fail", real_mode=True)
        routing = RoutingResult(
            task_id="direct-004",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.side_effect = [
            RuntimeError("First fail"),
            RuntimeError("Second fail"),
        ]

        with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
            result = _execute_direct(
                request,
                routing,
                mock_primitives,
                mock_state,
                start_time,
                initial_role="frontdoor",
            )

        assert result.answer.startswith("[ERROR:")
        assert "retry" in result.answer.lower()

    def test_direct_formalization_applied(self, mock_primitives, mock_state):
        """Direct call applies formalization when needed."""
        request = ChatRequest(prompt="Format this as JSON", real_mode=True)
        routing = RoutingResult(
            task_id="direct-005",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = "Raw answer"

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Raw answer"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (True, {"format": "json"})
                with patch("src.api.routes.chat_pipeline.direct_stage._formalize_output") as mock_formal:
                    mock_formal.return_value = '{"formatted": true}'
                    with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                        mock_features.return_value.generation_monitor = False
                        with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                            result = _execute_direct(
                                request,
                                routing,
                                mock_primitives,
                                mock_state,
                                start_time,
                                initial_role="frontdoor",
                            )

        assert result.answer == '{"formatted": true}'

    def test_direct_quality_escalation(self, mock_primitives, mock_state):
        """Direct call escalates on quality issues when generation_monitor enabled."""
        request = ChatRequest(prompt="Quality test", real_mode=True)
        routing = RoutingResult(
            task_id="direct-006",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.side_effect = [
            "Low quality answer",
            "Escalated answer",
        ]

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.side_effect = lambda x, _: x
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = True
                    with patch(
                        "src.api.routes.chat_pipeline.stages._detect_output_quality_issue"
                    ) as mock_detect:
                        mock_detect.return_value = "repetitive"
                        with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                            result = _execute_direct(
                                request,
                                routing,
                                mock_primitives,
                                mock_state,
                                start_time,
                                initial_role="frontdoor",
                            )

        assert result.answer == "Escalated answer"

    def test_direct_empty_answer_logs_failed_completion(self, mock_primitives, mock_state):
        """Empty direct outputs must not be scored as successful completions."""
        request = ChatRequest(prompt="Empty answer", real_mode=True)
        routing = RoutingResult(
            task_id="direct-empty",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()
        mock_primitives.llm_call.return_value = "  \n"

        with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
            mock_fmt.return_value = (False, None)
            with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                mock_features.return_value.generation_monitor = False
                with patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"):
                    result = _execute_direct(
                        request,
                        routing,
                        mock_primitives,
                        mock_state,
                        start_time,
                        initial_role="frontdoor",
                    )

        assert result.answer == ""
        call = mock_state.progress_logger.log_task_completed.call_args
        assert call.kwargs["success"] is False
        assert call.kwargs["completion_meta"]["answer_chars"] == 0

    def test_direct_no_progress_logger(self, mock_primitives):
        """Direct call handles missing progress_logger."""
        state = MagicMock()
        state.progress_logger = None
        state.increment_request = MagicMock()

        request = ChatRequest(prompt="No logger", real_mode=True)
        routing = RoutingResult(
            task_id="direct-008",
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        start_time = time.perf_counter()

        mock_primitives.llm_call.return_value = "Answer"

        with patch("src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer") as mock_trunc:
            mock_trunc.return_value = "Answer"
            with patch("src.api.routes.chat_pipeline.direct_stage._should_formalize") as mock_fmt:
                mock_fmt.return_value = (False, None)
                with patch("src.api.routes.chat_pipeline.stages.features") as mock_features:
                    mock_features.return_value.generation_monitor = False
                    result = _execute_direct(
                        request,
                        routing,
                        mock_primitives,
                        state,
                        start_time,
                        initial_role="frontdoor",
                    )

        assert result.answer == "Answer"
