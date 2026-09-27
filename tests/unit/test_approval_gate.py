#!/usr/bin/env python3
"""Tests for the approval gate module."""

from unittest.mock import MagicMock

import pytest
import yaml

from src.features import Features, set_features, reset_features
from src.graph.approval_gate import (
    ApprovalDecision,
    AutoApproveCallback,
    HaltReason,
    HaltState,
    should_halt,
    request_approval_for_escalation,
    check_interrupt_conditions,
    request_approval_for_interrupt,
)


@pytest.fixture(autouse=True)
def _reset_features():
    """Reset features after each test."""
    yield
    reset_features()


class TestHaltState:
    """Tests for HaltState dataclass."""

    def test_creation(self):
        halt = HaltState(
            reason=HaltReason.ESCALATION,
            from_role="coder_escalation",
            to_role="architect_general",
            description="Retries exhausted",
        )
        assert halt.reason == HaltReason.ESCALATION
        assert halt.from_role == "coder_escalation"
        assert halt.to_role == "architect_general"

    def test_destructive_tool_halt(self):
        halt = HaltState(
            reason=HaltReason.DESTRUCTIVE_TOOL,
            side_effects=["modifies_files", "system_state"],
        )
        assert halt.reason == HaltReason.DESTRUCTIVE_TOOL
        assert "modifies_files" in halt.side_effects


class TestAutoApproveCallback:
    """Tests for default auto-approve callback."""

    def test_always_approves(self):
        cb = AutoApproveCallback()
        halt = HaltState(reason=HaltReason.ESCALATION)
        assert cb.request_approval(halt) == ApprovalDecision.APPROVE

    def test_approves_destructive(self):
        cb = AutoApproveCallback()
        halt = HaltState(reason=HaltReason.DESTRUCTIVE_TOOL)
        assert cb.request_approval(halt) == ApprovalDecision.APPROVE


def _assert_same_tier(from_role: str, to_role: str) -> None:
    """Precondition for every HIGH_COST test: the pair must NOT cross a tier.

    `should_halt()` returns ESCALATION for a tier crossing before it ever
    consults the high-cost set, so a cross-tier pair makes a HIGH_COST
    assertion untestable rather than false. Asserting the precondition
    explicitly means a future tier reassignment fails here, naming the reason,
    instead of the HIGH_COST tests quietly reporting on a different code path.
    """
    from src.roles import get_tier

    assert get_tier(from_role) == get_tier(to_role), (
        f"HIGH_COST test precondition broken: {from_role} is "
        f"{get_tier(from_role)} but {to_role} is {get_tier(to_role)}; "
        "should_halt() will short-circuit to ESCALATION and never reach the "
        "high-cost check. Pick a same-tier pair."
    )


class TestShouldHalt:
    """Tests for should_halt() function."""

    def test_disabled_returns_none(self):
        set_features(Features(approval_gates=False))
        assert should_halt("coder_escalation", "architect_general") is None

    def test_tier_crossing_triggers_escalation(self):
        set_features(Features(approval_gates=True))
        result = should_halt("worker_general", "coder_escalation")
        assert result == HaltReason.ESCALATION

    def test_same_tier_no_halt(self):
        set_features(Features(approval_gates=True))
        # Worker to worker — same tier
        result = should_halt("worker_general", "worker_math")
        assert result is None

    def test_architect_triggers_high_cost(self):
        set_features(Features(approval_gates=True))
        # should_halt() checks the tier boundary FIRST and returns ESCALATION,
        # so the HIGH_COST path is only reachable for a SAME-TIER pair. This
        # used to read coder_escalation -> architect_general and was commented
        # "both Tier B"; on 2026-08-01 ARCHITECT_GENERAL was corrected to Tier.A
        # against the registry's declared tier, and these tests silently started
        # measuring ESCALATION instead of the path they name.
        # architect_critic, not architect_general: on 2026-07-31 the 122B
        # vacated architect_general for the new architect_critic role, leaving
        # architect_general serving the 27B. High cost is derived from model
        # mem_gb, so the expensive role moved with the weights — the live priors
        # now list architect_critic alone. Asserting against architect_general
        # would be asserting that a 27B model is expensive.
        # 2026-09-27 ARCHITECT SWAP (operator-decided): the big CPU model
        # (Flash-Next, :8074) now serves architect_general (Tier A) and the 27B
        # serves architect_critic, so the expensive role is architect_general
        # again. The same-tier pair that reaches HIGH_COST is frontdoor (A) ->
        # architect_general (A); coder_escalation -> architect_critic is now a
        # same-process, not-expensive pair.
        _assert_same_tier("frontdoor", "architect_general")
        result = should_halt("frontdoor", "architect_general")
        assert result == HaltReason.HIGH_COST
        _assert_same_tier("coder_escalation", "architect_critic")
        assert should_halt("coder_escalation", "architect_critic") is None

    def test_generated_stack_priors_drive_high_cost_roles(self, tmp_path):
        set_features(Features(approval_gates=True))
        stack_priors = tmp_path / "stack_priors.yaml"
        stack_priors.write_text(
            yaml.safe_dump(
                {
                    "roles": {
                        "ingest_long_context": {
                            "deployment_status": "live_stack",
                            "model": {"mem_gb": 10.0},
                        },
                        "coder_escalation": {
                            "deployment_status": "live_stack",
                            "model": {"mem_gb": 70.0},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        # Same-tier pair so the high-cost check is actually reached; the point
        # of this test is that mem_gb in the GENERATED priors decides, not that
        # a tier is crossed.
        _assert_same_tier("ingest_long_context", "coder_escalation")
        assert (
            should_halt(
                "ingest_long_context",
                "coder_escalation",
                stack_priors_path=stack_priors,
            )
            == HaltReason.HIGH_COST
        )
        assert (
            should_halt(
                "coder_escalation",
                "ingest_long_context",
                stack_priors_path=stack_priors,
            )
            is None
        )

    def test_missing_stack_priors_use_degraded_high_cost_fallback(self, tmp_path):
        set_features(Features(approval_gates=True))
        missing_stack_priors = tmp_path / "missing.yaml"

        _assert_same_tier("frontdoor", "architect_general")
        result = should_halt(
            "frontdoor",
            "architect_general",
            stack_priors_path=missing_stack_priors,
        )

        assert result == HaltReason.HIGH_COST

    def test_valid_stack_priors_without_high_cost_roles_do_not_use_fallback(self, tmp_path):
        set_features(Features(approval_gates=True))
        stack_priors = tmp_path / "stack_priors.yaml"
        stack_priors.write_text(
            yaml.safe_dump(
                {
                    "roles": {
                        "architect_general": {
                            "deployment_status": "live_stack",
                            "model": {"mem_gb": 10.0},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        _assert_same_tier("frontdoor", "architect_general")
        result = should_halt(
            "frontdoor",
            "architect_general",
            stack_priors_path=stack_priors,
        )

        assert result is None


class TestRequestApproval:
    """Tests for request_approval_for_escalation()."""

    def test_no_callback_auto_approves(self):
        set_features(Features(approval_gates=False))
        ctx = MagicMock()
        ctx.deps.approval_callback = None
        decision = request_approval_for_escalation(
            ctx, "coder_escalation", "architect_general", "test"
        )
        assert decision == ApprovalDecision.APPROVE

    def test_callback_called_on_halt(self):
        set_features(Features(approval_gates=True, resume_tokens=False))
        callback = MagicMock()
        callback.request_approval.return_value = ApprovalDecision.REJECT

        ctx = MagicMock()
        ctx.deps.approval_callback = callback
        ctx.state.pending_approval = None

        decision = request_approval_for_escalation(
            ctx, "worker_general", "coder_escalation", "test reason"
        )
        assert decision == ApprovalDecision.REJECT
        callback.request_approval.assert_called_once()

        # Verify halt state was passed
        halt = callback.request_approval.call_args[0][0]
        assert halt.reason == HaltReason.ESCALATION
        assert halt.from_role == "worker_general"
        assert halt.to_role == "coder_escalation"


class TestInterruptCondition:
    """Tests for generalized interrupt conditions (LangGraph pre-migration)."""

    def test_interrupt_condition_halt_reason(self):
        assert HaltReason.INTERRUPT_CONDITION == "interrupt_condition"

    def test_check_interrupt_conditions_none(self):
        """Empty conditions list returns None."""
        assert check_interrupt_conditions([], {}, {}) is None

    def test_check_interrupt_conditions_triggers(self):
        cond = MagicMock()
        cond.should_interrupt.return_value = "budget exceeded"
        result = check_interrupt_conditions([cond], {}, {})
        assert result == "budget exceeded"

    def test_check_interrupt_conditions_skips_none(self):
        cond = MagicMock()
        cond.should_interrupt.return_value = None
        assert check_interrupt_conditions([cond], {}, {}) is None

    def test_check_interrupt_conditions_handles_exception(self):
        cond = MagicMock()
        cond.should_interrupt.side_effect = RuntimeError("boom")
        assert check_interrupt_conditions([cond], {}, {}) is None

    def test_request_approval_for_interrupt_no_callback(self):
        ctx = MagicMock()
        ctx.deps.approval_callback = None
        decision = request_approval_for_interrupt(ctx, "test interrupt")
        assert decision == ApprovalDecision.APPROVE

    def test_request_approval_for_interrupt_reject(self):
        set_features(Features(approval_gates=True, resume_tokens=False))
        callback = MagicMock()
        callback.request_approval.return_value = ApprovalDecision.REJECT

        ctx = MagicMock()
        ctx.deps.approval_callback = callback
        ctx.state.pending_approval = None

        decision = request_approval_for_interrupt(ctx, "budget exceeded")
        assert decision == ApprovalDecision.REJECT
        callback.request_approval.assert_called_once()

        halt = callback.request_approval.call_args[0][0]
        assert halt.reason == HaltReason.INTERRUPT_CONDITION
        assert halt.description == "budget exceeded"
