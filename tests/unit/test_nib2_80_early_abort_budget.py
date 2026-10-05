"""NIB2-80: the EARLY_ABORT escalation path must respect cfg.max_escalations.

Before the fix, `FrontdoorNode` (and its siblings WorkerNode/CoderNode/
CoderEscalationNode/IngestNode) escalated unconditionally on
ErrorCategory.EARLY_ABORT without calling `_should_escalate`, so a run that
kept early-aborting could blow through `max_escalations` — the budget it was
nominally charged against was never actually read.

Reading chosen (see handoff NIB2-80 + agent report): immediate escalation on
early-abort is deliberate (it deliberately skips the retry precondition that
`_should_escalate` enforces for other categories — routing early-abort through
`_should_escalate` unmodified would block first-strike escalation until
`consecutive_failures >= max_retries`, which is a behaviour change beyond
closing the budget hole). The minimal, behaviour-preserving fix is to gate the
early-abort branch on `state.escalation_count < cfg.max_escalations` and, when
the budget is exhausted, fall through to the exact same
think-harder / `_should_escalate` / `_should_retry` / fail chain that every
sibling site already uses when its own gate refuses.
"""

from __future__ import annotations

import pytest

from src.graph.nodes import FrontdoorNode, WorkerNode
from src.graph.state import GraphConfig
from src.graph.graph import orchestration_graph
from src.roles import Role

from tests.unit.test_graph_nodes import MockREPLResult, make_deps, make_state


class TestEarlyAbortBudget:
    @pytest.mark.asyncio
    async def test_early_abort_escalates_when_under_budget(self):
        """escalation_count < max_escalations: early-abort still escalates immediately
        (no retry precondition), and the count is incremented."""
        state = make_state(current_role=Role.FRONTDOOR, escalation_count=0)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="Generation aborted: quality low"),
                MockREPLResult(output="fixed by escalation", is_final=True),
            ],
            llm_responses=[
                "x = broken_code()",  # error turn — no FINAL to avoid rescue
                "FINAL('fixed by escalation')",
            ],
            # First-strike escalation remains immediate even when ordinary
            # retries would still be available.
            config=GraphConfig(max_retries=3, max_escalations=2, max_turns=10),
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)

        assert isinstance(result.output.answer, str)
        assert state.escalation_count == 1
        assert str(Role.CODER_ESCALATION) in state.role_history

    @pytest.mark.asyncio
    async def test_early_abort_cycle_is_denied_without_charging_escalation(self):
        """An early abort cannot bypass the shared role-cycle guard."""
        state = make_state(
            current_role=Role.FRONTDOOR,
            role_history=["frontdoor", "worker_math", "frontdoor", "worker_math"],
        )
        deps = make_deps(
            repl_results=[MockREPLResult(error="Generation aborted: quality low")],
            llm_responses=["x = broken_code()"],
            config=GraphConfig(max_retries=1, max_escalations=2, max_turns=10),
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)

        assert result.output.success is False
        assert state.escalation_count == 0
        assert str(Role.CODER_ESCALATION) not in state.role_history

    @pytest.mark.asyncio
    async def test_early_abort_blocked_at_budget_falls_through_to_normal_failure_path(self):
        """escalation_count == max_escalations: early-abort must NOT escalate.
        It must land at the same destination sibling sites use when
        `_should_escalate` refuses (retry-if-available, else FAILED end) —
        never CoderEscalationNode, and the count must not be bumped further."""
        state = make_state(current_role=Role.FRONTDOOR, escalation_count=2)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="Generation aborted: quality low"),
            ],
            llm_responses=[
                "x = broken_code()",
            ],
            # max_retries=1 so think-harder's "consecutive_failures == max_retries-1"
            # trigger (== 0) can't fire on the first failure and mask the fall-through.
            config=GraphConfig(max_retries=1, max_escalations=2, max_turns=10),
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)

        assert result.output.success is False
        assert "FAILED" in result.output.answer
        # Budget must not have been charged again, and no escalation to coder_escalation.
        assert state.escalation_count == 2
        assert str(Role.CODER_ESCALATION) not in state.role_history

    @pytest.mark.asyncio
    async def test_sibling_non_early_abort_path_unaffected(self):
        """Regression guard: a non-EARLY_ABORT error (CODE category) on WorkerNode still
        retries then escalates through `_should_escalate` exactly as before the fix."""
        state = make_state(current_role=Role.WORKER_MATH)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="SyntaxError in code"),
                MockREPLResult(error="SyntaxError in code"),
                MockREPLResult(output="fixed", is_final=True),
            ],
            llm_responses=[
                "x = broken_code()",
                "x = broken_code()",
                "FINAL('fixed')",
            ],
            config=GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
        )
        result = await orchestration_graph.run(WorkerNode(), state=state, deps=deps)

        assert isinstance(result.output.answer, str)
        assert state.escalation_count == 1
        assert str(Role.CODER_ESCALATION) in state.role_history
