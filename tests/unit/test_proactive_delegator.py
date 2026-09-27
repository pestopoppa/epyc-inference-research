"""Comprehensive tests for proactive delegation.

Tests coverage for src/proactive_delegation/delegator.py (13% coverage)
and src/proactive_delegation/review_service.py (34% coverage).
"""

import json
from unittest.mock import Mock, patch

import pytest

from src.proactive_delegation.delegator import ProactiveDelegator
from src.roles import Role
from src.proactive_delegation.review_service import (
    AggregationService,
    ArchitectReviewService,
)
from src.proactive_delegation.types import (
    ComplexitySignals,
    ReviewDecision,
    SubtaskResult,
    TaskComplexity,
)


class TestProactiveDelegatorInit:
    """Test ProactiveDelegator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        registry = Mock()
        primitives = Mock()

        delegator = ProactiveDelegator(registry, primitives)

        assert delegator.registry is registry
        assert delegator.primitives is primitives
        assert delegator.review_service is not None
        assert delegator.aggregation_service is not None
        assert delegator.skip_complexity_check is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        registry = Mock()
        primitives = Mock()
        progress_logger = Mock()
        hybrid_router = Mock()

        delegator = ProactiveDelegator(
            registry,
            primitives,
            progress_logger=progress_logger,
            hybrid_router=hybrid_router,
            max_iterations=5,
            skip_complexity_check=True,
        )

        assert delegator.progress_logger is progress_logger
        assert delegator.hybrid_router is hybrid_router
        assert delegator.iteration_context.max_iterations == 5
        assert delegator.skip_complexity_check is True


class TestComplexityRouting:
    """Test complexity-based routing logic."""

    def test_route_by_complexity_trivial(self):
        """Test routing for trivial tasks."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        complexity, action, signals, confidence = delegator.route_by_complexity("What is 2+2?")

        assert complexity == TaskComplexity.TRIVIAL
        assert action == "direct"
        assert confidence == 1.0

    def test_route_by_complexity_simple(self):
        """Test routing for simple tasks."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        complexity, action, signals, confidence = delegator.route_by_complexity("Print hello world")

        # Should be simple or moderate depending on heuristics
        assert action in ["direct", "repl", "specialist"]

    def test_route_by_complexity_complex(self):
        """Test routing for complex tasks."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        complexity, action, signals, confidence = delegator.route_by_complexity(
            "Implement a distributed transaction system with ACID guarantees"
        )

        # Should route to architect for complex tasks
        assert action in ["specialist", "architect"]

    def test_route_by_complexity_with_skip(self):
        """Test routing when complexity check is skipped."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives, skip_complexity_check=True)

        complexity, action, signals, confidence = delegator.route_by_complexity("Any task")

        assert complexity == TaskComplexity.COMPLEX
        assert action == "architect"

    def test_route_by_complexity_with_hybrid_router(self):
        """Test routing with MemRL hybrid router."""
        registry = Mock()
        primitives = Mock()
        hybrid_router = Mock()
        hybrid_router.route.return_value = (["architect_general"], "learned")

        delegator = ProactiveDelegator(registry, primitives, hybrid_router=hybrid_router)

        task_ir = {"objective": "Complex task", "task_type": "code"}
        complexity, action, signals, confidence = delegator.route_by_complexity(
            "Complex task", task_ir=task_ir
        )

        # Learned routing should upgrade to COMPLEX
        assert complexity == TaskComplexity.COMPLEX
        assert action == "architect"
        assert confidence == 0.8  # Learned confidence


class TestTargetRoleSelection:
    """Test target role selection."""

    def test_get_target_role_with_thinking_signal(self):
        """Test role selection with deep-reasoning escalation."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        signals = ComplexitySignals(thinking_requested=True)
        role = delegator.get_target_role("specialist", signals)

        assert role == "architect_general"

    def test_thinking_trigger_routes_to_live_architect_role(self):
        """Test /think-style prompts avoid the retired dedicated-thinking role."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        _complexity, action, signals, _confidence = delegator.route_by_complexity(
            "/think analyze the failure mode carefully"
        )
        role = delegator.get_target_role(action, signals)

        assert signals.thinking_requested is True
        assert role == "architect_general"

    def test_get_target_role_direct(self):
        """Test role for direct action."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        signals = ComplexitySignals()
        role = delegator.get_target_role("direct", signals)

        assert role == "frontdoor"

    def test_get_target_role_specialist(self):
        """Test role for specialist action."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        signals = ComplexitySignals()
        role = delegator.get_target_role("specialist", signals)

        assert role == "coder_escalation"

    def test_get_target_role_architect(self):
        """Test role for architect action."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        signals = ComplexitySignals()
        role = delegator.get_target_role("architect", signals)

        assert role == "architect_general"


class TestDelegateWorkflow:
    """Test main delegate() workflow."""

    @pytest.mark.asyncio
    async def test_delegate_no_steps(self):
        """Test delegation with empty plan."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        task_ir = {
            "task_id": "test123",
            "objective": "Test objective",
            "plan": {"steps": []},
        }

        result = await delegator.delegate(task_ir)

        assert result.task_id == "test123"
        assert result.aggregated_output == "[ERROR: No subtasks in plan]"

    @pytest.mark.asyncio
    async def test_delegate_sequential_execution(self):
        """Test sequential subtask execution."""
        registry = Mock()
        primitives = Mock()
        primitives.llm_call.return_value = "Task completed"

        delegator = ProactiveDelegator(registry, primitives)

        # Mock review service to always approve
        delegator.review_service.review = Mock(
            return_value=Mock(
                decision=ReviewDecision.APPROVE,
                approved_output="Task completed",
                feedback="",
            )
        )

        task_ir = {
            "task_id": "test123",
            "objective": "Multi-step task",
            "plan": {
                "steps": [
                    {"id": "S1", "actor": "coder", "action": "Write code"},
                    {"id": "S2", "actor": "worker", "action": "Write tests"},
                ]
            },
        }

        with patch("src.features.features") as mock_features:
            mock_features.return_value.parallel_execution = False

            result = await delegator.delegate(task_ir)

            assert len(result.subtask_results) == 2
            assert result.all_approved is True

    @pytest.mark.asyncio
    async def test_delegate_canonicalizes_worker_explore_actor(self):
        """Test worker_explore delegates through the live worker_general path."""
        registry = Mock()
        primitives = Mock()
        primitives.llm_call.return_value = "Task completed"

        delegator = ProactiveDelegator(registry, primitives)

        delegator.review_service.review = Mock(
            return_value=Mock(
                decision=ReviewDecision.APPROVE,
                approved_output="Task completed",
                feedback="",
            )
        )

        task_ir = {
            "task_id": "test123",
            "objective": "Single step task",
            "plan": {
                "steps": [
                    {"id": "S1", "actor": "worker_explore", "action": "Write tests"},
                ]
            },
        }

        with patch("src.features.features") as mock_features:
            mock_features.return_value.parallel_execution = False

            await delegator.delegate(task_ir)

        assert primitives.llm_call.call_args.kwargs["role"] == "worker_general"

    @pytest.mark.asyncio
    async def test_delegate_canonicalizes_worker_chain_actor(self):
        """Test generic worker chain names resolve through the live worker role."""
        registry = Mock()
        primitives = Mock()
        primitives.llm_call.return_value = "Task completed"

        delegator = ProactiveDelegator(registry, primitives)

        delegator.review_service.review = Mock(
            return_value=Mock(
                decision=ReviewDecision.APPROVE,
                approved_output="Task completed",
                feedback="",
            )
        )

        task_ir = {
            "task_id": "test123",
            "objective": "Single step task",
            "plan": {
                "steps": [
                    {"id": "S1", "actor": "worker", "action": "Write tests"},
                ]
            },
        }

        with patch("src.features.features") as mock_features:
            mock_features.return_value.parallel_execution = False

            await delegator.delegate(task_ir)

        assert primitives.llm_call.call_args.kwargs["role"] == "worker_general"

    @pytest.mark.asyncio
    async def test_delegate_with_progress_logger(self):
        """Test delegation with progress logging."""
        registry = Mock()
        primitives = Mock()
        primitives.llm_call.return_value = "Task completed"

        progress_logger = Mock()
        delegator = ProactiveDelegator(registry, primitives, progress_logger=progress_logger)

        # Mock review service
        delegator.review_service.review = Mock(
            return_value=Mock(
                decision=ReviewDecision.APPROVE,
                approved_output="Task completed",
                feedback="",
            )
        )

        task_ir = {
            "task_id": "test123",
            "objective": "Test task",
            "plan": {"steps": [{"id": "S1", "actor": "coder", "action": "Do work"}]},
        }

        with patch("src.features.features") as mock_features:
            mock_features.return_value.parallel_execution = False

            await delegator.delegate(task_ir)

            progress_logger.log_task_started.assert_called_once()
            progress_logger.log_task_completed.assert_called_once()


class TestSpecialistPromptBuilding:
    """Test specialist prompt construction."""

    def test_build_specialist_prompt_complete(self):
        """Test prompt building with all fields."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        task_ir = {"objective": "Overall goal"}
        step = {
            "action": "Implement feature X",
            "inputs": ["requirements.txt", "design.md"],
            "outputs": ["feature.py", "tests.py"],
        }

        prompt = delegator._build_specialist_prompt(task_ir, step)

        assert "Overall goal" in prompt
        assert "Implement feature X" in prompt
        assert "requirements.txt" in prompt
        assert "feature.py" in prompt

    def test_build_specialist_prompt_minimal(self):
        """Test prompt building with minimal fields."""
        registry = Mock()
        primitives = Mock()
        delegator = ProactiveDelegator(registry, primitives)

        task_ir = {"objective": "Simple goal"}
        step = {"action": "Do task"}

        prompt = delegator._build_specialist_prompt(task_ir, step)

        assert "Simple goal" in prompt
        assert "Do task" in prompt


class TestRoleEscalation:
    """Test role escalation logic.

    2026-08-01: the delegator's private escalation map was deleted; `_escalate_role`
    now delegates to the canonical `Role.escalates_to()` in src/roles.py. These tests
    assert AGREEMENT with the canonical ladder rather than restating a fifth copy of
    it, plus the two edge behaviours the private map used to own.
    """

    @staticmethod
    def _delegator() -> ProactiveDelegator:
        return ProactiveDelegator(Mock(), Mock())

    def test_escalate_role_worker_to_coder(self):
        """Test worker escalation."""
        assert self._delegator()._escalate_role("worker_general") == "coder_escalation"

    def test_escalate_role_agrees_with_canonical_ladder(self):
        """Every role in the canonical ladder escalates exactly as src/roles.py says."""
        delegator = self._delegator()
        for role in Role:
            target = role.escalates_to()
            if target is None:
                continue
            assert delegator._escalate_role(role.value) == target.value

    def test_escalate_role_coder_no_longer_null_hops_to_its_own_host(self):
        """coder_escalation is an ALIAS on the MI210 27B's :8083 process — the
        architect_general host until the 2026-09-27 ARCHITECT SWAP, architect_critic
        since.

        The deleted private map sent it to its own host — the same GGUF in the same
        process, so the "escalation" could only reroll the sampling seed. After the
        swap it escalates to architect_general (Flash-Next, CPU :8074), a real hop.
        """
        escalated = self._delegator()._escalate_role("coder_escalation")

        assert escalated == Role.CODER_ESCALATION.escalates_to().value
        assert escalated != "architect_critic"
        assert escalated == "architect_general"

    def test_escalate_role_architect_critic_is_not_a_self_loop(self):
        """architect_critic (the 27B since the 2026-09-27 swap) escalates to
        architect_general (Flash-Next), never to itself."""
        escalated = self._delegator()._escalate_role("architect_critic")

        assert escalated != "architect_critic"
        assert escalated == Role.ARCHITECT_CRITIC.escalates_to().value == "architect_general"

    def test_escalate_role_terminal_rung_returns_itself(self):
        """A role with no escalation target stays put — never a downgrade.

        The terminal architect rung is architect_general (Flash-Next) since the
        2026-09-27 ARCHITECT SWAP (it was architect_critic before)."""
        terminal = [role for role in Role if role.escalates_to() is None]
        assert Role.ARCHITECT_GENERAL in terminal
        assert Role.ARCHITECT_CRITIC not in terminal
        delegator = self._delegator()
        assert delegator._escalate_role(Role.ARCHITECT_GENERAL.value) == "architect_general"

    def test_escalate_role_unknown_string_keeps_historical_fallback(self):
        """Behaviour preserved for anything outside the ladder."""
        assert self._delegator()._escalate_role("not_a_role") == "architect_general"


class TestAggregationService:
    """Test aggregation service strategies."""

    def test_aggregate_empty_list(self):
        """Test aggregating empty result list."""
        service = AggregationService()
        result = service.aggregate([])
        assert result == ""

    def test_aggregate_concatenate(self):
        """Test concatenate strategy."""
        service = AggregationService()
        results = [
            SubtaskResult(
                subtask_id="S1",
                role="coder",
                output="Code output",
                success=True,
            ),
            SubtaskResult(
                subtask_id="S2",
                role="worker",
                output="Test output",
                success=True,
            ),
        ]

        aggregated = service.aggregate(results, strategy="concatenate")

        assert "S1" in aggregated
        assert "S2" in aggregated
        assert "Code output" in aggregated
        assert "Test output" in aggregated

    def test_aggregate_merge_code(self):
        """Test merge_code strategy."""
        service = AggregationService()
        results = [
            SubtaskResult(
                subtask_id="S1",
                role="coder",
                output="import os\n\ndef func1():\n    pass",
                success=True,
            ),
            SubtaskResult(
                subtask_id="S2",
                role="coder",
                output="import sys\n\ndef func2():\n    pass",
                success=True,
            ),
        ]

        aggregated = service.aggregate(results, strategy="merge_code")

        # Imports should be deduplicated and at top
        assert "import os" in aggregated
        assert "import sys" in aggregated
        assert "def func1" in aggregated
        assert "def func2" in aggregated

    def test_aggregate_structured(self):
        """Test structured JSON strategy."""
        service = AggregationService()
        results = [
            SubtaskResult(
                subtask_id="S1",
                role="coder",
                output="Output 1",
                success=True,
            ),
            SubtaskResult(
                subtask_id="S2",
                role="worker",
                output="Output 2",
                success=False,
                error="Failed",
            ),
        ]

        aggregated = service.aggregate(results, strategy="structured")

        # Should be valid JSON
        parsed = json.loads(aggregated)
        assert len(parsed["results"]) == 2
        assert parsed["summary"]["total"] == 2
        assert parsed["summary"]["successful"] == 1
        assert parsed["summary"]["failed"] == 1


class TestArchitectReviewService:
    """Test architect review service."""

    def test_review_approve(self):
        """Test review that approves output."""
        primitives = Mock()
        primitives.llm_call.return_value = '{"d": "approve", "s": 0.9, "f": "Good"}'

        service = ArchitectReviewService(primitives)

        spec = {"objective": "Build feature"}
        subtask = {"id": "S1", "action": "Write code"}
        output = "def feature(): pass"

        review = service.review(spec, subtask, output)

        assert review.decision == ReviewDecision.APPROVE
        assert review.score == 0.9
        assert review.approved_output == output

    def test_review_request_changes(self):
        """Test review requesting changes."""
        primitives = Mock()
        primitives.llm_call.return_value = '{"d": "changes", "s": 0.5, "f": "Add tests"}'

        service = ArchitectReviewService(primitives)

        spec = {"objective": "Build feature"}
        subtask = {"id": "S1", "action": "Write code"}
        output = "def feature(): pass"

        review = service.review(spec, subtask, output)

        assert review.decision == ReviewDecision.REQUEST_CHANGES
        assert review.feedback == "Add tests"

    def test_review_parse_error(self):
        """Test review with parse error."""
        primitives = Mock()
        primitives.llm_call.return_value = "Invalid JSON"

        service = ArchitectReviewService(primitives)

        spec = {"objective": "Build feature"}
        subtask = {"id": "S1", "action": "Write code"}
        output = "def feature(): pass"

        review = service.review(spec, subtask, output)

        # TD-21.6: a terminal (repair-exhausted -- the stubbed model returns the
        # same unparseable text on the repair turn too) parse failure withholds
        # as REQUEST_EVIDENCE -- never a default verdict indistinguishable from a
        # genuine reviewer decision (the old behavior this test used to lock in).
        assert review.decision == ReviewDecision.REQUEST_EVIDENCE
        assert review.feedback.startswith("parse_failure:")

    def test_review_quick_mode(self):
        """Test quick review mode."""
        primitives = Mock()
        primitives.llm_call.return_value = '{"d": "approve", "s": 0.8}'

        service = ArchitectReviewService(primitives)

        spec = {"objective": "Build feature"}
        subtask = {"id": "S1", "action": "Write code"}
        output = "def feature(): pass"

        review = service.review(spec, subtask, output, quick_mode=True)

        assert review.decision == ReviewDecision.APPROVE


class TestArchitectReviewServiceHelpers:
    """Test architect review service helper methods."""

    def test_parse_review_response_clean_json(self):
        """Test parsing clean JSON response."""
        primitives = Mock()
        service = ArchitectReviewService(primitives)

        response = '{"d": "approve", "s": 0.9}'
        parsed = service._parse_review_response(response)

        assert parsed["d"] == "approve"
        assert parsed["s"] == 0.9

    def test_parse_review_response_markdown_json(self):
        """Test parsing JSON in markdown code blocks."""
        primitives = Mock()
        service = ArchitectReviewService(primitives)

        response = '```json\n{"d": "approve", "s": 0.9}\n```'
        parsed = service._parse_review_response(response)

        assert parsed["d"] == "approve"

    def test_parse_review_response_with_text(self):
        """Test parsing JSON embedded in text."""
        primitives = Mock()
        service = ArchitectReviewService(primitives)

        response = 'Here is the review: {"d": "approve", "s": 0.9} Done.'
        parsed = service._parse_review_response(response)

        assert parsed["d"] == "approve"
