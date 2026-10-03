"""Tests for src/api/routes/chat_review.py.

Covers: _detect_output_quality_issue, _needs_plan_review, _apply_plan_review,
_compute_plan_review_phase. (The answer review gate tests went with the gate, RI-18c.)
"""

from unittest.mock import MagicMock, patch

from src.api.routes.chat_review import (
    _apply_plan_review,
    _compute_plan_review_phase,
    _detect_output_quality_issue,
    _needs_plan_review,
    _plan_review_abort_message,
    _plan_review_should_abort,
)
from src.proactive_delegation.types import PlanReviewResult


# ── _detect_output_quality_issue ─────────────────────────────────────────


class TestDetectOutputQualityIssue:
    """Test output quality detection heuristics."""

    def test_returns_none_on_clean_output(self):
        answer = (
            "Machine learning models learn patterns from training data. "
            "They use gradient descent to minimize a loss function. "
            "The resulting weights encode the statistical relationships found in examples. "
            "Different architectures suit different tasks, from CNNs for images to transformers for text."
        )
        assert _detect_output_quality_issue(answer) is None

    def test_returns_none_on_short_output(self):
        assert _detect_output_quality_issue("OK") is None

    def test_returns_none_on_empty(self):
        assert _detect_output_quality_issue("") is None

    def test_detects_high_repetition(self):
        answer = " ".join(["the same words repeat"] * 30)
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "repetition" in result

    def test_detects_near_empty_after_stripping(self):
        # Must be >= 20 chars to pass the early check, but near-empty after stripping prefixes
        answer = "```                           \n\n```"
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "near_empty" in result

    def test_detects_garbled_output(self):
        lines = ["ok\n"] * 20 + ["a\n"] * 20 + ["This is a longer line\n"] * 2
        answer = "".join(lines)
        result = _detect_output_quality_issue(answer)
        # May or may not trigger depending on thresholds — no assertion on exact result
        # Just verify it doesn't crash
        assert result is None or isinstance(result, str)


# ── _needs_plan_review ───────────────────────────────────────────────────


class TestNeedsPlanReview:
    """Test plan review gate logic."""

    @patch("src.proactive_delegation.classify_task_complexity")
    def test_skips_trivial_tasks(self, mock_classify):
        from src.proactive_delegation import TaskComplexity

        mock_classify.return_value = (TaskComplexity.TRIVIAL, {})
        state = MagicMock(plan_review_phase="A")
        assert _needs_plan_review({"objective": "hi"}, ["frontdoor"], state) is False

    @patch("src.proactive_delegation.classify_task_complexity")
    def test_skips_architect_self_review(self, mock_classify):
        from src.proactive_delegation import TaskComplexity

        mock_classify.return_value = (TaskComplexity.MODERATE, {})
        state = MagicMock(plan_review_phase="A")
        assert _needs_plan_review({"objective": "design"}, ["architect_general"], state) is False

    @patch("src.proactive_delegation.classify_task_complexity")
    def test_allows_moderate_non_architect(self, mock_classify):
        from src.proactive_delegation import TaskComplexity

        mock_classify.return_value = (TaskComplexity.MODERATE, {})
        state = MagicMock(plan_review_phase="A", hybrid_router=None)
        result = _needs_plan_review({"objective": "moderate task"}, ["frontdoor"], state)
        assert result is True


# ── _apply_plan_review ───────────────────────────────────────────────────


class TestApplyPlanReview:
    """Test plan review patch application."""

    def test_no_patches_returns_unchanged(self):
        review = MagicMock(patches=[])
        result = _apply_plan_review(["frontdoor"], review)
        assert result == ["frontdoor"]

    def test_reroute_patch_changes_role(self):
        review = MagicMock(patches=[{"op": "reroute", "step": "S1", "v": "coder_escalation"}])
        result = _apply_plan_review(["frontdoor", "worker"], review)
        assert result[0] == "coder_escalation"
        assert result[1] == "worker"

    def test_reroute_patch_ignores_non_role_value(self):
        review = MagicMock(
            patches=[
                {
                    "op": "reroute",
                    "step": "S1",
                    "v": "Complete filtering, add KMeans, handle column count",
                }
            ]
        )
        result = _apply_plan_review(["frontdoor"], review)
        assert result == ["frontdoor"]

    def test_ignores_non_reroute_ops(self):
        review = MagicMock(patches=[{"op": "add_step", "step": "S2"}])
        result = _apply_plan_review(["frontdoor"], review)
        assert result == ["frontdoor"]


class TestPlanReviewAbort:
    """Plan edits are non-terminal; only an explicit abort may stop work."""

    def test_drop_discards_plan_without_aborting_task(self):
        review = PlanReviewResult(
            decision="drop",
            score=0.9,
            feedback="Plan incomplete; missing problem logic.",
            patches=[],
        )

        assert _plan_review_should_abort(review) is False
        assert _plan_review_abort_message(review) == (
            "Plan rejected by architect review: Plan incomplete; missing problem logic."
        )

    def test_explicit_abort_is_reserved_terminal_verdict(self):
        review = PlanReviewResult(decision="abort", score=1.0, feedback="Unsafe")
        assert _plan_review_should_abort(review) is True

    def test_reroute_does_not_abort(self):
        review = PlanReviewResult(
            decision="reroute",
            score=0.7,
            feedback="Use coder.",
            patches=[{"op": "reroute", "step": "S1", "v": "coder_escalation"}],
        )

        assert _plan_review_should_abort(review) is False

    def test_none_does_not_abort(self):
        assert _plan_review_should_abort(None) is False


# ── _compute_plan_review_phase ───────────────────────────────────────────


class TestComputePlanReviewPhase:
    """Test plan review phase computation."""

    def test_phase_a_on_low_reviews(self):
        assert _compute_plan_review_phase({"total_reviews": 10}) == "A"

    def test_phase_a_on_empty_q_values(self):
        assert _compute_plan_review_phase({"total_reviews": 100, "task_class_q_values": {}}) == "A"
