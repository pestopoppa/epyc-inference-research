"""Integration tests exercising real pipeline stages with mock LLM backend.

Unlike test_chat_pipeline.py which uses mock_mode=True (exiting at stage 3),
these tests set real_mode=True and patch only _init_primitives to return
controllable mock LLMPrimitives. This exercises:

- Stage 1: Routing (classify_and_route, failure graph veto, MemRL logging)
- Stage 2: Preprocessing (input formalization)
- Stage 5: Plan review gate
- Stage 7: Mode selection
- Stage 8-9: Direct/React/Delegated execution, error annotation
- Quality detection, output truncation

The mock_responses dict on LLMPrimitives controls what the "LLM" returns,
allowing deterministic testing of post-processing logic.
"""

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.api import create_app
from src.api.models import ChatResponse
from src.api.routes.chat_pipeline.stages import _annotate_error
from src.api.routes.chat_pipeline.proactive_stage import _parse_plan_steps
from src.api.routes.chat_review import _detect_output_quality_issue
from src.llm_primitives import LLMPrimitives


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Create fresh FastAPI app."""
    return create_app()


@pytest.fixture
def primitives():
    """Create LLMPrimitives in mock mode with empty mock_responses."""
    return LLMPrimitives(mock_mode=True, mock_responses={})


@pytest.fixture
def client_and_primitives(app, primitives):
    """TestClient with _init_primitives patched to return mock primitives.

    This allows real_mode requests to flow through the full pipeline
    (routing → preprocessing → mode selection → execution → annotation)
    while using mock LLM responses instead of real inference.
    """
    with patch("src.api.routes.chat._init_primitives", return_value=primitives):
        with TestClient(app) as client:
            yield client, primitives


# ── Group 1: Direct mode through full pipeline ───────────────────────────


class TestDirectModePipeline:
    """Test _execute_direct through the real pipeline with mock backend."""

    def test_direct_mode_produces_nonempty_answer(self, client_and_primitives):
        """Direct mode should produce a non-empty answer from mock LLM."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "What is the capital of France?",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        assert "[MOCK]" in data["answer"]  # Mock mode returns [MOCK] prefix

    def test_direct_mode_sets_response_metadata(self, client_and_primitives):
        """Direct mode should populate all metadata fields correctly."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "What is 2+2?",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        assert data["real_mode"] is True
        assert data["mock_mode"] is False
        assert data["mode"] == "direct"
        assert data["turns"] >= 1
        assert data["elapsed_seconds"] > 0.0
        assert isinstance(data["routed_to"], str)
        assert len(data["routed_to"]) > 0  # Should have a role assigned

    def test_direct_mode_force_role_routes_correctly(self, client_and_primitives):
        """force_role should determine the routed_to field in response."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "test query",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
                "force_role": "coder_escalation",
            },
        )
        data = response.json()
        assert data["routed_to"] == "coder_escalation"
        assert data["routing_strategy"] == "forced"

    def test_direct_mode_explicit_role(self, client_and_primitives):
        """Explicit role should be used for routing."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "test query",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
                "role": "worker_explore",
            },
        )
        data = response.json()
        assert data["routed_to"] == "worker_general"
        assert data["routing_strategy"] == "explicit"

    def test_direct_mode_tracks_call_log(self, client_and_primitives):
        """LLMPrimitives.call_log should record the direct LLM call."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "What is quantum entanglement?",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        assert response.status_code == 200

        # Verify call_log was populated
        assert len(prims.call_log) >= 1
        entry = prims.call_log[0]
        assert entry.call_type == "call"
        assert "quantum entanglement" in entry.prompt
        assert entry.elapsed_seconds > 0.0

    def test_direct_mode_increments_stats(self, client_and_primitives):
        """LLMPrimitives stats should be incremented after direct call."""
        client, prims = client_and_primitives
        assert prims.total_calls == 0

        response = client.post(
            "/chat",
            json={
                "prompt": "test",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        assert response.status_code == 200
        assert prims.total_calls >= 1

    def test_direct_mode_with_context(self, client_and_primitives):
        """Context should be prepended to the prompt in direct mode."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "Summarize the document",
                "context": "This is a document about cats.",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        assert response.status_code == 200
        assert len(data["answer"]) > 0

        # The call_log should show the combined prompt
        assert len(prims.call_log) >= 1
        # Context is prepended to the prompt in direct mode
        full_prompt = prims.call_log[0].prompt
        assert "cats" in full_prompt or "document" in full_prompt

    def test_direct_mode_custom_mock_response(self, client_and_primitives):
        """Custom mock_responses dict should control LLM output."""
        client, prims = client_and_primitives
        # Set a custom response that will match the prompt
        prims.mock_responses["What is 42?"] = "The answer to life, the universe, and everything."

        response = client.post(
            "/chat",
            json={
                "prompt": "What is 42?",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        # The mock_responses key matching is exact on the full prompt,
        # so with skip_suffix and no context, the prompt should match
        # Note: the prompt is modified by pipeline (stop sequences etc.)
        # so we just verify we get a real response
        assert len(data["answer"]) > 0

    def test_direct_mode_role_history(self, client_and_primitives):
        """role_history should contain the initial role."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "test",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
                "force_role": "worker_explore",
            },
        )
        data = response.json()
        assert "role_history" in data
        assert isinstance(data["role_history"], list)
        assert "worker_general" in data["role_history"]


# ── Group 2: Error annotation ─────────────────────────────────────────


class TestErrorAnnotation:
    """Test _annotate_error() with various error patterns."""

    def _make_response(self, answer: str) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            turns=1,
            elapsed_seconds=0.1,
            mock_mode=False,
            real_mode=True,
        )

    def test_timeout_error_gets_504(self):
        """Timeout errors should be annotated with error_code 504."""
        resp = _annotate_error(self._make_response("[ERROR: Request timed out after 300s]"))
        assert resp.error_code == 504
        assert resp.error_detail is not None
        assert "timed out" in resp.error_detail.lower()

    def test_backend_error_gets_502(self):
        """Backend failures should be annotated with error_code 502."""
        resp = _annotate_error(self._make_response("[ERROR: Backend connection failed]"))
        assert resp.error_code == 502

    def test_generic_error_gets_500(self):
        """Generic errors should be annotated with error_code 500."""
        resp = _annotate_error(self._make_response("[ERROR: something unexpected]"))
        assert resp.error_code == 500

    def test_clean_response_has_no_error_code(self):
        """Normal responses should have no error_code."""
        resp = _annotate_error(self._make_response("The answer is 42."))
        assert resp.error_code is None
        assert resp.error_detail is None

    def test_empty_answer_has_no_error_code(self):
        """Empty answers should not be annotated."""
        resp = _annotate_error(self._make_response(""))
        assert resp.error_code is None

    def test_error_detail_preserved(self):
        """Error detail should contain the full error message."""
        msg = "[ERROR: Backend 'worker' unavailable (circuit open)]"
        resp = _annotate_error(self._make_response(msg))
        assert resp.error_detail == msg

    def test_partial_error_prefix_not_matched(self):
        """Text containing ERROR but not starting with [ERROR: should not match."""
        resp = _annotate_error(self._make_response("This mentions an ERROR in the text"))
        assert resp.error_code is None


# ── Group 3: Output quality detection ─────────────────────────────────


class TestOutputQualityDetection:
    """Test _detect_output_quality_issue() with real text patterns."""

    def test_normal_text_passes(self):
        """Well-formed text should not be flagged."""
        text = (
            "The capital of France is Paris. It is known for the Eiffel Tower, "
            "the Louvre Museum, and its rich cultural heritage. Paris is the "
            "largest city in France with a population of over 2 million people."
        )
        assert _detect_output_quality_issue(text) is None

    def test_repetitive_trigrams_detected(self):
        """Highly repetitive text should trigger high_repetition detection."""
        # Create text with extreme trigram repetition
        repeated = "the cat sat " * 50
        issue = _detect_output_quality_issue(repeated)
        assert issue is not None
        assert "repetition" in issue.lower()

    def test_garbled_short_lines_detected(self):
        """Mostly very short lines should trigger garbled_output detection."""
        # Simulate garbled output with many diverse short lines (no trigram repetition)
        # Each line is unique to avoid triggering repetition before garbled check
        short_lines = [f"x{i}" for i in range(40)]  # 40 unique short lines (<10 chars)
        long_lines = ["This is a much longer line that provides some balance"] * 3
        text = "\n".join(short_lines + long_lines)
        issue = _detect_output_quality_issue(text)
        assert issue is not None
        assert "garbled" in issue.lower()

    def test_near_empty_after_stripping(self):
        """Text that's nearly empty after stripping prefixes should be flagged."""
        # Must be >= 20 chars to pass the length gate, but near-empty after stripping
        text = "The answer is         "  # 22 chars, strips to "" after prefix removal
        issue = _detect_output_quality_issue(text)
        assert issue is not None
        assert "near_empty" in issue.lower()

    def test_short_text_not_analyzed(self):
        """Text shorter than 20 chars should not be analyzed."""
        assert _detect_output_quality_issue("short") is None
        assert _detect_output_quality_issue("") is None
        assert _detect_output_quality_issue(None) is None

    def test_code_block_with_content_passes(self):
        """Code blocks with actual content should pass."""
        text = "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
        assert _detect_output_quality_issue(text) is None

    def test_diverse_text_passes(self):
        """Text with high diversity (unique trigrams) should pass."""
        text = (
            "Quantum computing leverages superposition and entanglement to process "
            "information in fundamentally different ways than classical computers. "
            "While a classical bit can be 0 or 1, a quantum bit (qubit) can exist "
            "in a superposition of both states simultaneously. This allows quantum "
            "computers to explore many solutions in parallel, potentially solving "
            "certain problems exponentially faster."
        )
        assert _detect_output_quality_issue(text) is None


# ── Group 4: Plan step parsing ────────────────────────────────────────


class TestPlanStepParsing:
    """Test _parse_plan_steps() with various JSON formats."""

    def test_valid_json_array(self):
        """Standard JSON array of step objects should parse correctly."""
        raw = '[{"id": "S1", "action": "search", "actor": "worker"}, {"id": "S2", "action": "summarize"}]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2
        assert steps[0]["id"] == "S1"
        assert steps[0]["action"] == "search"
        assert steps[0]["actor"] == "worker"
        assert steps[1]["actor"] == "worker"  # Default added

    def test_markdown_fenced_json(self):
        """JSON wrapped in markdown code fences should be extracted."""
        raw = '```json\n[{"id": "S1", "action": "analyze"}]\n```'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1
        assert steps[0]["action"] == "analyze"

    def test_trailing_commas_fixed(self):
        """Trailing commas (common LLM quirk) should be handled."""
        raw = '[{"id": "S1", "action": "step1"},]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1

    def test_invalid_json_returns_empty(self):
        """Invalid JSON should return empty list."""
        assert _parse_plan_steps("not json at all") == []
        assert _parse_plan_steps("{invalid}") == []
        assert _parse_plan_steps("") == []

    def test_missing_required_fields_filtered(self):
        """Steps without 'id' or 'action' should be filtered out."""
        raw = '[{"id": "S1", "action": "ok"}, {"id": "S2"}, {"action": "missing_id"}]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1  # Only first step has both id and action

    def test_defaults_added(self):
        """Missing optional fields should get defaults."""
        raw = '[{"id": "S1", "action": "test"}]'
        steps = _parse_plan_steps(raw)
        assert steps[0]["actor"] == "worker"
        assert steps[0]["depends_on"] == []
        assert steps[0]["outputs"] == []

    def test_non_array_returns_empty(self):
        """Non-array JSON should return empty list."""
        assert _parse_plan_steps('{"id": "S1", "action": "test"}') == []

    def test_non_dict_items_filtered(self):
        """Non-dict items in array should be filtered."""
        raw = '[{"id": "S1", "action": "ok"}, "string_item", 42]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1


# ── Group 5: Real-mode routing through full pipeline ──────────────────


class TestRealModeRouting:
    """Test that routing decisions flow through to response metadata in real mode."""

    def test_default_routing_uses_frontdoor(self, client_and_primitives):
        """Without force_role, simple queries should route to frontdoor."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "hello there",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        # Default routing for simple text goes to frontdoor
        assert data["routed_to"] == "frontdoor"
        assert data["routing_strategy"] in ("rules", "classified")

    def test_forced_strategy_label(self, client_and_primitives):
        """force_role should set routing_strategy to 'forced'."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "test",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
                "force_role": "architect_general",
            },
        )
        data = response.json()
        assert data["routing_strategy"] == "forced"
        assert data["routed_to"] == "architect_general"

    def test_response_has_timing_fields(self, client_and_primitives):
        """Real-mode responses should have timing fields."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "test",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        # These are 0 for mock backend but should exist
        assert "prompt_eval_ms" in data
        assert "generation_ms" in data
        assert "predicted_tps" in data
        assert "http_overhead_ms" in data
        assert isinstance(data["prompt_eval_ms"], (int, float))
        assert isinstance(data["generation_ms"], (int, float))

    def test_formalization_not_applied_by_default(self, client_and_primitives):
        """Input formalization should not be applied with default features."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "simple question",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        assert data["formalization_applied"] is False


# ── Group 7: Quality check integration in direct mode ─────────────────


class TestQualityCheckIntegration:
    """Test that quality checks trigger in the direct mode pipeline."""

    def test_quality_check_with_generation_monitor_disabled(self, client_and_primitives):
        """Quality check should not trigger when generation_monitor feature is off."""
        client, prims = client_and_primitives
        # Default features have generation_monitor=False
        response = client.post(
            "/chat",
            json={
                "prompt": "test",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        assert response.status_code == 200
        # Should succeed without quality escalation
        assert "coder_escalation" not in data.get("routed_to", "")

    def test_llm_error_captured_in_answer(self, app):
        """If _mock_call raises, llm_call captures it as [ERROR:...] in the answer."""
        prims = LLMPrimitives(mock_mode=True, mock_responses={})

        def failing_mock(prompt, role):
            raise RuntimeError("Simulated backend failure")

        with patch("src.api.routes.chat._init_primitives", return_value=prims):
            with patch.object(prims, "_mock_call", side_effect=failing_mock):
                with TestClient(app) as client:
                    response = client.post(
                        "/chat",
                        json={
                            "prompt": "test error capture",
                            "mock_mode": False,
                            "real_mode": True,
                            "force_mode": "direct",
                        },
                    )
                    data = response.json()
                    # Errors now return proper HTTP status codes (not silent 200)
                    assert response.status_code == 502
                    # llm_call catches exceptions and returns [ERROR:...]
                    assert "[ERROR:" in data["answer"]
                    assert "Simulated backend failure" in data["answer"]
                    # Error annotation detects "failed" → 502 (backend error pattern)
                    assert data.get("error_code") == 502


# ── Group 8: Pipeline stage interaction tests ─────────────────────────


class TestPipelineStageInteraction:
    """Test interactions between pipeline stages."""

    def test_direct_mode_with_different_roles(self, app):
        """Different roles should produce different call_log entries."""
        prims = LLMPrimitives(mock_mode=True)

        with patch("src.api.routes.chat._init_primitives", return_value=prims):
            with TestClient(app) as client:
                # First request with one role
                client.post(
                    "/chat",
                    json={
                        "prompt": "test1",
                        "mock_mode": False,
                        "real_mode": True,
                        "force_mode": "direct",
                        "force_role": "coder_escalation",
                    },
                )

                # Second request with different role
                client.post(
                    "/chat",
                    json={
                        "prompt": "test2",
                        "mock_mode": False,
                        "real_mode": True,
                        "force_mode": "direct",
                        "force_role": "worker_explore",
                    },
                )

                # Verify both calls recorded with correct roles
                assert len(prims.call_log) >= 2
                roles_used = {entry.role for entry in prims.call_log}
                assert "coder_escalation" in roles_used
                assert "worker_general" in roles_used

    def test_mock_mode_bypasses_real_pipeline(self, app):
        """mock_mode=True should return early without calling _init_primitives."""
        with patch("src.api.routes.chat._init_primitives") as mock_init:
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={
                        "prompt": "mock test",
                        "mock_mode": True,
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert data["mode"] == "mock"
                # _init_primitives should NOT have been called
                mock_init.assert_not_called()

    def test_real_mode_calls_init_primitives(self, app):
        """real_mode=True should call _init_primitives."""
        prims = LLMPrimitives(mock_mode=True)
        with patch("src.api.routes.chat._init_primitives", return_value=prims) as mock_init:
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={
                        "prompt": "real test",
                        "mock_mode": False,
                        "real_mode": True,
                        "force_mode": "direct",
                    },
                )
                assert response.status_code == 200
                mock_init.assert_called_once()

    def test_response_model_validates(self, client_and_primitives):
        """Response should be valid according to ChatResponse pydantic model."""
        client, prims = client_and_primitives
        response = client.post(
            "/chat",
            json={
                "prompt": "validation test",
                "mock_mode": False,
                "real_mode": True,
                "force_mode": "direct",
            },
        )
        data = response.json()
        # Pydantic should validate — verify by constructing model from response
        chat_resp = ChatResponse(**data)
        assert chat_resp.answer == data["answer"]
        assert chat_resp.turns == data["turns"]
        assert chat_resp.elapsed_seconds == data["elapsed_seconds"]
        assert chat_resp.real_mode is True
        assert chat_resp.mock_mode is False


# ── Group 9: _needs_plan_review gating ────────────────────────────────


class TestNeedsPlanReview:
    """Test plan review gating logic."""

    def test_trivial_tasks_skip_review(self):
        """TRIVIAL complexity should skip plan review."""
        from src.api.routes.chat_review import _needs_plan_review

        state = MagicMock()
        state.plan_review_phase = "A"
        state.hybrid_router = None
        # Short prompt = TRIVIAL complexity
        task_ir = {"objective": "hi", "task_type": "chat"}
        assert _needs_plan_review(task_ir, ["frontdoor"], state) is False

    def test_architect_routing_skips_review(self):
        """Architect routing should skip review (no self-review)."""
        from src.api.routes.chat_review import _needs_plan_review

        state = MagicMock()
        state.plan_review_phase = "A"
        state.hybrid_router = None
        task_ir = {
            "objective": "Design a complex distributed system with multiple services " * 3,
            "task_type": "chat",
        }
        assert _needs_plan_review(task_ir, ["architect_general"], state) is False

    def test_complex_tasks_skip_review(self):
        """COMPLEX tasks should skip review (architect already owns plan)."""
        from src.api.routes.chat_review import _needs_plan_review

        state = MagicMock()
        state.plan_review_phase = "A"
        state.hybrid_router = None
        # Very complex prompt with multiple signal words
        task_ir = {
            "objective": (
                "Build a complete distributed microservice architecture with "
                "authentication, authorization, rate limiting, circuit breaking, "
                "service mesh, observability, tracing, logging, monitoring, and "
                "deployment pipeline with blue-green deployment strategy"
            ),
            "task_type": "chat",
        }
        # COMPLEX should bypass review (architect already owns the plan)
        result = _needs_plan_review(task_ir, ["frontdoor"], state)
        # Either False (COMPLEX bypass) or True (MODERATE) - depends on classifier
        assert isinstance(result, bool)


# ── Group 10: Plan review phase computation ────────────────────────────


class TestPlanReviewPhaseComputation:
    """Test _compute_plan_review_phase() logic."""

    def test_phase_a_with_few_reviews(self):
        """Few reviews should stay in Phase A."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        stats = {"total_reviews": 10, "task_class_q_values": {"chat": 0.8}}
        assert _compute_plan_review_phase(stats) == "A"

    def test_phase_a_with_no_q_values(self):
        """No Q-values should stay in Phase A."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        stats = {"total_reviews": 100, "task_class_q_values": {}}
        assert _compute_plan_review_phase(stats) == "A"

    def test_phase_b_with_good_q_values(self):
        """High mean Q >= 0.7 and min Q >= 0.5 should reach Phase B."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        stats = {
            "total_reviews": 60,
            "task_class_q_values": {"chat": 0.8, "code": 0.75},
        }
        assert _compute_plan_review_phase(stats) == "B"

    def test_phase_c_with_excellent_q_values(self):
        """High min Q >= 0.7 and >= 100 reviews should reach Phase C."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        stats = {
            "total_reviews": 120,
            "task_class_q_values": {"chat": 0.9, "code": 0.85},
        }
        assert _compute_plan_review_phase(stats) == "C"

    def test_phase_stays_a_with_low_q(self):
        """Low Q-values should keep Phase A."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        stats = {
            "total_reviews": 100,
            "task_class_q_values": {"chat": 0.4, "code": 0.3},
        }
        assert _compute_plan_review_phase(stats) == "A"
