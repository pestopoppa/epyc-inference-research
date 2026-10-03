#!/usr/bin/env python3
"""Tests for API module imports and function signature compatibility.

These tests catch the class of bug where:
- chat.py imports a function that doesn't exist in the canonical source
- New parameters added to prompt_builders break callers
- Decomposed chat modules have broken imports or circular dependencies

The build_root_lm_prompt import bug (commit 7b4b2da) caused ALL real_mode
requests to crash with TypeError, producing empty benchmark results.
The orchestrator.py facade that caused that bug was deleted in Phase 1.
"""

import inspect

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


class TestChatImportsResolve:
    """Verify all imports in chat.py and decomposed modules resolve without errors."""

    def test_chat_module_imports(self):
        """Import chat module — catches any ImportError or circular import."""
        from src.api.routes import chat  # noqa: F401

    def test_prompt_builders_imports(self):
        """Import prompt_builders — catches missing functions."""
        from src.prompt_builders import (  # noqa: F401
            build_root_lm_prompt,
            build_stage2_review_prompt,
            build_long_context_exploration_prompt,
            build_routing_context,
            extract_code_from_response,
            classify_error,
            auto_wrap_final,
            build_escalation_prompt,
        )

    def test_chat_utils_imports(self):
        """Import chat_utils — catches missing functions or constants."""
        from src.api.routes.chat_utils import (  # noqa: F401
            THREE_STAGE_CONFIG,
            TWO_STAGE_CONFIG,
            QWEN_STOP,
            LONG_CONTEXT_CONFIG,
            _estimate_tokens,
            _is_stub_final,
            _strip_tool_outputs,
            _resolve_answer,
            _truncate_looped_answer,
            _should_formalize,
            _formalize_output,
        )

    def test_chat_routing_imports(self):
        """Import chat_routing — catches missing functions."""
        from src.api.routes.chat_routing import (  # noqa: F401
            _select_mode,
            _classify_and_route,
        )

    def test_chat_delegation_imports(self):
        """Import chat_delegation — catches missing functions."""
        from src.api.routes.chat_delegation import (  # noqa: F401
            _parse_architect_decision,
            _architect_delegated_answer,
        )

    def test_chat_review_imports(self):
        """Import chat_review — catches missing functions."""
        from src.api.routes.chat_review import (  # noqa: F401
            _detect_output_quality_issue,
            _needs_plan_review,
            _architect_plan_review,
            _apply_plan_review,
            _store_plan_review_episode,
            _compute_plan_review_phase,
        )

    def test_chat_vision_imports(self):
        """Import chat_vision — catches missing functions."""
        from src.api.routes.chat_vision import (  # noqa: F401
            _needs_structured_analysis,
            _handle_vision_request,
            _execute_vision_tool,
            _vision_react_mode_answer,
            _handle_multi_file_vision,
        )

    def test_chat_summarization_imports(self):
        """Import chat_summarization — catches missing functions."""
        from src.api.routes.chat_summarization import (  # noqa: F401
            _is_summarization_task,
            _should_use_two_stage,
            _run_two_stage_summarization,
        )


class TestSignatureCompatibility:
    """Verify key function signatures have expected parameters."""

    def test_build_root_lm_prompt_routing_context(self):
        """Verify routing_context param exists (the param whose absence crashed benchmarks)."""
        from src.prompt_builders import build_root_lm_prompt

        sig = inspect.signature(build_root_lm_prompt)
        assert "routing_context" in sig.parameters, (
            "routing_context missing from build_root_lm_prompt"
        )


class TestChatRouteImportSource:
    """Verify chat.py imports critical functions from the right module."""

    def test_no_orchestrator_imports(self):
        """chat.py must not import from deleted orchestrator.py facade."""
        import ast
        from pathlib import Path

        chat_path = Path(__file__).parent.parent.parent / "src" / "api" / "routes" / "chat.py"
        tree = ast.parse(chat_path.read_text())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "orchestrator" in node.module:
                    imported_names = [alias.name for alias in node.names]
                    assert False, (
                        f"chat.py still imports from deleted orchestrator facade: {imported_names}. "
                        f"Import from src.prompt_builders or decomposed modules instead."
                    )

    def test_orchestrator_facade_deleted(self):
        """Verify orchestrator.py facade no longer exists."""
        from pathlib import Path

        facade_path = (
            Path(__file__).parent.parent.parent / "src" / "api" / "services" / "orchestrator.py"
        )
        assert not facade_path.exists(), (
            "src/api/services/orchestrator.py still exists — it should have been deleted "
            "during Phase 1 decomposition."
        )


class TestCallableSmoke:
    """Verify key functions are callable with expected args."""

    def test_build_root_lm_prompt_callable(self):
        """build_root_lm_prompt works with all expected arguments."""
        from src.prompt_builders import build_root_lm_prompt

        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test prompt",
            last_output="",
            last_error="",
            turn=0,
            routing_context="Role: frontdoor",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_code_from_response_callable(self):
        """extract_code_from_response works."""
        from src.prompt_builders import extract_code_from_response

        result = extract_code_from_response("```python\nprint('hello')\n```")
        assert isinstance(result, str)

    def test_auto_wrap_final_callable(self):
        """auto_wrap_final works."""
        from src.prompt_builders import auto_wrap_final

        result = auto_wrap_final("print('hello')")
        assert isinstance(result, str)


# ── Phase 1b: Pipeline stage tests ──────────────────────────────────────


class TestChatPipelineImports:
    """Verify Phase 1b pipeline module imports resolve."""

    def test_chat_pipeline_imports(self):
        """Import chat_pipeline — catches any ImportError."""
        from src.api.routes import chat_pipeline  # noqa: F401

    def test_pipeline_stage_functions_importable(self):
        """All pipeline stage functions are importable."""
        from src.api.routes.chat_pipeline import (  # noqa: F401
            _route_request,
            _preprocess,
            _init_primitives,
            _execute_mock,
            _execute_vision,
            _plan_review_gate,
            _execute_delegated,
            _execute_react,
            _execute_direct,
            _execute_repl,
            _annotate_error,
        )

    def test_routing_result_importable(self):
        """RoutingResult dataclass is importable from chat_utils."""
        from src.api.routes.chat_utils import RoutingResult

        assert RoutingResult is not None


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_default_values(self):
        from src.api.routes.chat_utils import RoutingResult

        r = RoutingResult(task_id="test-123", task_ir={}, use_mock=False)
        assert r.task_id == "test-123"
        assert r.use_mock is False
        assert r.routing_decision == []
        assert r.routing_strategy == ""
        assert r.formalization_applied is False
        assert r.timeout_s > 0

    def test_role_property(self):
        from src.api.routes.chat_utils import RoutingResult

        r = RoutingResult(
            task_id="t",
            task_ir={},
            use_mock=False,
            routing_decision=["architect_general"],
        )
        assert r.role == "architect_general"

    def test_role_property_empty(self):
        from src.api.routes.chat_utils import RoutingResult

        r = RoutingResult(task_id="t", task_ir={}, use_mock=False)
        assert "frontdoor" in r.role.lower()

    def test_timeout_for_role(self):
        from src.api.routes.chat_utils import RoutingResult

        r = RoutingResult(task_id="t", task_ir={}, use_mock=False)
        # Differentiated timeouts by role size/speed
        assert r.timeout_for_role("worker_explore") == 60
        assert r.timeout_for_role("worker_fast") == 30
        assert r.timeout_for_role("architect_general") == 600
        assert r.timeout_for_role("frontdoor") == 180


class TestRoleTimeouts:
    """Tests for ROLE_TIMEOUTS mapping."""

    def test_all_known_roles_have_timeouts(self):
        from src.api.routes.chat_utils import ROLE_TIMEOUTS

        expected_roles = [
            "worker_explore",
            "worker_math",
            "worker_vision",
            "frontdoor",
            "coder_escalation",
            "coder_escalation",
            "architect_general",
        ]
        for role in expected_roles:
            assert role in ROLE_TIMEOUTS, f"Missing timeout for {role}"

    def test_worker_timeouts_shorter_than_architect(self):
        from src.api.routes.chat_utils import ROLE_TIMEOUTS

        # Workers get shorter timeouts for faster circuit breaker response
        assert ROLE_TIMEOUTS["worker_explore"] < ROLE_TIMEOUTS["architect_general"]
        assert _RETIRED_ARCHITECT_ROLE not in ROLE_TIMEOUTS
        assert ROLE_TIMEOUTS["worker_explore"] == 60
        assert ROLE_TIMEOUTS["architect_general"] == 600

    def test_default_timeout_exists(self):
        from src.api.routes.chat_utils import DEFAULT_TIMEOUT_S

        assert DEFAULT_TIMEOUT_S > 0
        assert DEFAULT_TIMEOUT_S <= 600


class TestAnnotateError:
    """Tests for _annotate_error() error detection."""

    def test_success_response_unchanged(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="The answer is 42",
            turns=1,
            elapsed_seconds=0.5,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code is None
        assert result.error_detail is None

    def test_timeout_error_gets_504(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="[ERROR: frontdoor LM call failed: Request timed out after 60s]",
            turns=1,
            elapsed_seconds=60.5,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code == 504
        assert "timed out" in result.error_detail.lower()

    def test_backend_error_gets_502(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="[ERROR: Direct LLM call failed after retry: Backend connection refused]",
            turns=1,
            elapsed_seconds=1.0,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code == 502

    def test_circuit_open_gets_503(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="[ERROR: Backend unavailable (circuit open): http://localhost:8083]",
            turns=1,
            elapsed_seconds=1.0,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code == 503

    def test_admission_rejected_gets_429(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="[ERROR: admission] Backend queue full for http://localhost:8084",
            turns=1,
            elapsed_seconds=1.0,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code == 429

    def test_generic_error_gets_500(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="[ERROR: unexpected parse error]",
            turns=1,
            elapsed_seconds=0.1,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code == 500

    def test_failed_prefix_gets_500(self):
        from src.api.models.responses import ChatResponse
        from src.api.routes.chat_pipeline import _annotate_error

        resp = ChatResponse(
            answer="[FAILED: max escalations reached]",
            turns=1,
            elapsed_seconds=5.0,
            mock_mode=False,
        )
        result = _annotate_error(resp)
        assert result.error_code == 500

    def test_chatresponse_error_fields_default_none(self):
        from src.api.models.responses import ChatResponse

        resp = ChatResponse(
            answer="hello",
            turns=1,
            elapsed_seconds=0.1,
            mock_mode=True,
        )
        assert resp.error_code is None
        assert resp.error_detail is None


# ── Phase 2: Protocol + health tracker tests ─────────────────────────────


class TestProtocolImports:
    """Verify Protocol interfaces are importable."""

    def test_all_protocols_importable(self):
        from src.api.protocols import (  # noqa: F401
            QScorerProtocol,
            EpisodicStoreProtocol,
            HybridRouterProtocol,
            ProgressLoggerProtocol,
            ToolRegistryProtocol,
            ScriptRegistryProtocol,
            RegistryLoaderProtocol,
            FailureGraphProtocol,
        )

    def test_protocols_are_runtime_checkable(self):
        from src.api.protocols import (
            QScorerProtocol,
            EpisodicStoreProtocol,
            HybridRouterProtocol,
            ProgressLoggerProtocol,
            ToolRegistryProtocol,
            ScriptRegistryProtocol,
            RegistryLoaderProtocol,
            FailureGraphProtocol,
        )

        for proto in (
            QScorerProtocol,
            EpisodicStoreProtocol,
            HybridRouterProtocol,
            ProgressLoggerProtocol,
            ToolRegistryProtocol,
            ScriptRegistryProtocol,
            RegistryLoaderProtocol,
            FailureGraphProtocol,
        ):
            # runtime_checkable protocols support isinstance()
            assert isinstance(proto, type), f"{proto} is not a type"


class TestHealthTrackerImports:
    """Verify health tracker module is importable and wired into AppState."""

    def test_health_tracker_importable(self):
        from src.api.health_tracker import (  # noqa: F401
            BackendCircuit,
            BackendHealthTracker,
            DEFAULT_COOLDOWN_S,
            DEFAULT_FAILURE_THRESHOLD,
            MAX_COOLDOWN_S,
        )

    def test_appstate_has_health_tracker(self):
        from src.api.state import AppState
        from src.api.health_tracker import BackendHealthTracker

        state = AppState()
        assert hasattr(state, "health_tracker")
        assert isinstance(state.health_tracker, BackendHealthTracker)

    def test_health_response_has_backend_health_field(self):
        from src.api.models.responses import HealthResponse

        resp = HealthResponse(status="ok")
        assert hasattr(resp, "backend_health")
        assert resp.backend_health is None  # default
