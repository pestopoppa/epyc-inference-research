#!/usr/bin/env python3
"""Tests for architect delegation (investigate via specialist tools).

Tests the TOON/JSON parser, multi-loop delegation, feature flag gating,
and specialist document passthrough.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Parser Tests ──────────────────────────────────────────────────────────


class TestParseArchitectDecision:
    """Tests for _parse_architect_decision()."""

    def test_toon_direct(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("D|The answer is 42")
        assert result["mode"] == "direct"
        assert result["answer"] == "The answer is 42"
        assert result["brief"] == ""

    def test_toon_investigate(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision(
            "I|brief:Check src/api.py for error handling|to:coder_escalation"
        )
        assert result["mode"] == "investigate"
        assert "Check src/api.py" in result["brief"]
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "react"

    def test_toon_investigate_repl_mode(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision(
            "I|brief:Draft implementation doc|to:coder_escalation|mode:repl"
        )
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "repl"

    def test_toon_investigate_worker_coder_role_normalized(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision(
            "I|brief:split into parallel file tasks|to:worker_coder|mode:repl"
        )
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "repl"

    def test_json_direct(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        payload = json.dumps({"mode": "direct", "answer": "The answer is 42"})
        result = _parse_architect_decision(payload)
        assert result["mode"] == "direct"
        assert result["answer"] == "The answer is 42"

    def test_json_investigate(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        payload = json.dumps(
            {
                "mode": "investigate",
                "brief": "Search for X",
                "to": "worker_explore",
            }
        )
        result = _parse_architect_decision(payload)
        assert result["mode"] == "investigate"
        assert result["brief"] == "Search for X"
        assert result["delegate_to"] == "worker_general"

    def test_markdown_wrapped_json(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        wrapped = '```json\n{"mode": "direct", "answer": "hello"}\n```'
        result = _parse_architect_decision(wrapped)
        assert result["mode"] == "direct"
        assert result["answer"] == "hello"

    def test_fallback_bare_text(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("Just a plain text answer with no format")
        assert result["mode"] == "direct"
        assert result["answer"] == "Just a plain text answer with no format"

    def test_invalid_role_clamped(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("I|brief:Check something|to:nonexistent_role")
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"

    def test_invalid_mode_clamped(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("I|brief:Check something|to:coder_escalation|mode:invalid")
        assert result["delegate_mode"] == "react"


# ── Delegation Flow Tests ─────────────────────────────────────────────────


class TestArchitectDelegatedAnswer:
    """Tests for _architect_delegated_answer()."""

    def _mock_primitives(self, responses):
        """Create mock LLMPrimitives that returns responses in sequence."""
        primitives = MagicMock()
        primitives.llm_call = MagicMock(side_effect=responses)
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 100
        primitives.get_cache_stats.return_value = None
        return primitives

    def _mock_state(self):
        """Create mock AppState."""
        state = MagicMock()
        state.tool_registry = None
        return state

    def test_direct_answer_no_loops(self):
        """Architect answers directly — no delegation loops."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = self._mock_primitives(["D|The answer is 42"])
        state = self._mock_state()

        answer, stats = _architect_delegated_answer(
            question="What is 6*7?",
            context="",
            primitives=primitives,
            state=state,
        )

        assert answer == "The answer is 42"
        assert stats["loops"] == 0
        assert primitives.llm_call.call_count == 1

    def test_pre_delegation_architect_lock_timeout_sets_break_reason(self):
        """Architect Phase-A lock timeout should set explicit pre-delegation break reason."""
        from src.api.routes.chat_delegation import _architect_delegated_answer
        from src.exceptions import InferenceError

        primitives = MagicMock()
        primitives.llm_call.side_effect = InferenceError(
            "Inference lock timeout (request deadline exceeded)"
        )
        primitives.total_tokens_generated = 0
        state = self._mock_state()

        answer, stats = _architect_delegated_answer(
            question="Investigate this issue",
            context="",
            primitives=primitives,
            state=state,
        )

        assert answer.startswith("[ERROR: Architect delegation failed]")
        assert stats.get("break_reason") == "pre_delegation_lock_timeout"
        assert stats.get("loops") == 0

    def test_pre_delegation_error_string_sets_break_reason(self):
        """Phase-A [ERROR: ...] string from llm_call should not be treated as D| direct answer."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives.llm_call.return_value = (
            "[ERROR: Inference lock timeout (request deadline exceeded) "
            "(role=architect_general, mode=exclusive)]"
        )
        primitives.total_tokens_generated = 0
        state = self._mock_state()

        answer, stats = _architect_delegated_answer(
            question="Investigate this issue",
            context="",
            primitives=primitives,
            state=state,
        )

        assert answer.startswith("[ERROR: Architect delegation failed]")
        assert stats.get("break_reason") == "pre_delegation_lock_timeout"
        assert stats.get("loops") == 0

    def test_one_investigation_loop(self):
        """Architect investigates once, then synthesizes."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        state = self._mock_state()

        # Mock _run_architect_decision to return investigate then direct answer
        decisions = [
            ("I|brief:Check the file src/api.py|to:coder_escalation", 1, 0),
            ("D|Based on the investigation, the answer is X", 1, 0),
        ]
        decision_iter = iter(decisions)

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            side_effect=lambda *a, **kw: next(decision_iter),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(
                "File contents: error handling at line 42",
                2,
                ["code_search"],
                [],
                False,
                False,
                {},
                [],
            ),
        ):
            answer, stats = _architect_delegated_answer(
                question="Where is error handling?",
                context="",
                primitives=primitives,
                state=state,
            )

        assert "answer is X" in answer
        assert stats["loops"] == 1
        assert len(stats["phases"]) >= 2

    def test_delegation_diagnostics_are_byte_equal_for_identical_inputs(self):
        """P1 lifecycle refactor gate: identical inputs produce stable diagnostics."""
        from src.api.routes.chat_delegation import _architect_delegated_answer
        from src.api.routes.chat_pipeline.delegation_stage import _delegation_diagnostics

        def run_once() -> str:
            primitives = MagicMock()
            primitives._backends = {"test": True}
            primitives.total_tokens_generated = 0
            primitives.get_request_priority.return_value = "interactive"
            primitives.get_max_queue_wait_ms.return_value = None
            primitives.get_migration_budget_ms.return_value = None
            primitives.get_request_cancel_check.return_value = None
            primitives.get_request_task_id.return_value = "task-123"
            state = self._mock_state()
            cache = MagicMock()
            cache.make_key.return_value = "stable-cache-key"
            cache.get.return_value = None

            with patch(
                "src.api.routes.chat_delegation._run_architect_decision",
                return_value=(
                    "I|brief:inspect deterministic path|to:coder_escalation|mode:repl",
                    1,
                    0,
                ),
            ), patch(
                "src.api.routes.chat_delegation._run_specialist_loop",
                return_value=(
                    "stable specialist report",
                    0,
                    [],
                    [],
                    False,
                    True,
                    {
                        "transport": "chat",
                        "completion_reason": "stop",
                        "prompt_ms": 10.0,
                        "gen_ms": 20.0,
                        "first_token_ms": 3.0,
                        "stream_chunks": 4,
                        "tokens": 12,
                        "llm_elapsed_ms": 33.0,
                    },
                    [],
                ),
            ), patch(
                "src.delegation_cache.get_delegation_cache",
                return_value=cache,
            ):
                _answer, stats = _architect_delegated_answer(
                    question="q",
                    context="ctx",
                    primitives=primitives,
                    state=state,
                    max_loops=3,
                    force_response_on_cap=True,
                )

            diagnostics = _delegation_diagnostics(
                "architect_general",
                ["architect_general", "coder_escalation"],
                stats["delegation_events"],
                stats,
            )
            return json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))

        assert run_once() == run_once()

    def test_multi_loop_investigation(self):
        """Architect requests two investigations before answering."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        responses = [
            # Loop 0 Phase A: first investigation request
            "I|brief:Check file A|to:coder_escalation",
            # Loop 1 Phase A: second investigation request
            "I|brief:Check file B|to:worker_explore",
            # Loop 2 Phase A: final answer
            "D|After investigating A and B, the answer is Z",
        ]

        primitives = self._mock_primitives(responses)
        state = self._mock_state()

        call_count = [0]
        reports = ["Report from A", "Report from B"]
        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            def make_repl(*a, **kw):
                mock_repl = MagicMock()
                mock_repl.get_prompt.return_value = "prompt"
                idx = min(call_count[0], len(reports) - 1)
                mock_repl.execute.return_value = {"final": reports[idx]}
                mock_repl._tool_invocations = 1
                mock_repl.tool_registry = None
                call_count[0] += 1
                return mock_repl
            mock_repl_cls.side_effect = make_repl

            primitives.llm_call.side_effect = [
                responses[0],  # investigate A
                "code",  # REPL code for A
                responses[1],  # investigate B
                "code",  # REPL code for B
                responses[2],  # final answer
            ]

            answer, stats = _architect_delegated_answer(
                question="Compare A and B",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
            )

        assert "answer is Z" in answer
        assert stats["loops"] == 2

    def test_max_loops_cap_forces_response(self):
        """When architect keeps investigating past max_loops, force synthesis."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        responses = [
            # Loop 0: investigate
            "I|brief:Check X|to:coder_escalation",
            # Loop 1: investigate again
            "I|brief:Check Y|to:coder_escalation",
            # Loop 2: investigate AGAIN (would exceed max_loops=2)
            "I|brief:Check Z|to:coder_escalation",
            # Forced synthesis after cap
            "Forced answer from all reports",
        ]

        primitives = self._mock_primitives(responses)
        state = self._mock_state()

        call_count = [0]
        reports = ["Report X", "Report Y", "Report Z"]
        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            def make_repl(*a, **kw):
                mock_repl = MagicMock()
                mock_repl.get_prompt.return_value = "prompt"
                idx = min(call_count[0], len(reports) - 1)
                mock_repl.execute.return_value = {"final": reports[idx]}
                mock_repl._tool_invocations = 1
                mock_repl.tool_registry = None
                call_count[0] += 1
                return mock_repl
            mock_repl_cls.side_effect = make_repl

            primitives.llm_call.side_effect = [
                responses[0],  # investigate X
                "code",  # REPL code
                responses[1],  # investigate Y
                "code",  # REPL code
                responses[2],  # investigate Z (exceeds cap)
                "code",  # REPL code
                responses[3],  # forced synthesis
            ]

            answer, stats = _architect_delegated_answer(
                question="Complex question",
                context="",
                primitives=primitives,
                state=state,
                max_loops=2,
                force_response_on_cap=True,
            )

        assert answer  # Should have a forced answer
        assert stats["loops"] == 2

    def test_specialist_document_passthrough(self):
        """Specialist drafts document via rescued report → document is final answer."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        specialist_doc = "def hello():\n    return 'world'"

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        state = self._mock_state()

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            return_value=("I|brief:Draft the implementation|to:coder_escalation|mode:repl", 1, 0),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(specialist_doc, 0, [], [], False, True, {}, []),
        ):
            answer, stats = _architect_delegated_answer(
                question="Write a hello function",
                context="",
                primitives=primitives,
                state=state,
            )

        # The specialist document should be the final answer (via report_rescued)
        assert answer == specialist_doc
        assert stats.get("break_reason") == "specialist_report"

    def test_delegate_role_validation(self):
        """Invalid delegate role gets clamped to coder_escalation."""
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("I|brief:Do something|to:invalid_role_xyz")
        assert result["delegate_to"] == "coder_escalation"


# ── Feature Flag Gating ──────────────────────────────────────────────────


class TestFeatureFlagGating:
    """Tests for architect_delegation feature flag."""

    def test_feature_flag_exists(self):
        """Feature flag is defined in Features dataclass."""
        from src.features import Features

        f = Features()
        assert hasattr(f, "architect_delegation")
        assert f.architect_delegation is False

    def test_feature_flag_in_summary(self):
        """Feature flag appears in summary dict."""
        from src.features import Features

        f = Features(architect_delegation=True)
        summary = f.summary()
        assert "architect_delegation" in summary
        assert summary["architect_delegation"] is True

    def test_env_var_enables_flag(self):
        """ORCHESTRATOR_ARCHITECT_DELEGATION=1 enables the flag."""
        import os
        from src.features import get_features

        os.environ["ORCHESTRATOR_ARCHITECT_DELEGATION"] = "1"
        try:
            f = get_features()
            assert f.architect_delegation is True
        finally:
            del os.environ["ORCHESTRATOR_ARCHITECT_DELEGATION"]


class TestDelegationTokenCaps:
    """Tests for delegation token budgeting guards."""

    def test_specialist_turn_has_default_token_cap(self):
        from src.api.routes.chat_delegation import _run_specialist_loop

        primitives = MagicMock()
        primitives.llm_call = MagicMock(return_value="FINAL('ok')")

        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_result = MagicMock()
            mock_result.is_final = True
            mock_result.final_answer = "ok"
            mock_result.output = ""
            mock_result.error = None
            mock_repl.execute.return_value = mock_result
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            _run_specialist_loop(
                question="q",
                context="",
                brief="b",
                delegate_to="coder_escalation",
                delegate_mode="repl",
                primitives=primitives,
                tool_registry=None,
            )

        kwargs = primitives.llm_call.call_args.kwargs
        assert kwargs.get("n_tokens") == 224

    def test_forced_synthesis_has_token_cap(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        # Loop 0: investigate; cap reached immediately (max_loops=0) -> forced synthesis
        primitives.llm_call = MagicMock(side_effect=[
            "I|brief:check|to:coder_escalation",
            "forced synthesis answer",
        ])

        state = MagicMock()
        state.tool_registry = None

        answer, _stats = _architect_delegated_answer(
            question="q",
            context="",
            primitives=primitives,
            state=state,
            max_loops=0,
            force_response_on_cap=True,
        )
        assert answer
        assert primitives.llm_call.call_args_list[-1].kwargs.get("n_tokens") == 128

    def test_timeout_break_skips_forced_synthesis_call(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="should_not_be_called")

        state = MagicMock()
        state.tool_registry = None

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            return_value=("I|brief:investigate|to:coder_escalation", 1, 0),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(
                "[Delegation timeout after 2 turn(s), 45.0s]",
                0,
                [],
                [],
                True,
                False,
                {},
                [],
            ),
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=2,
                force_response_on_cap=True,
            )

        assert stats.get("break_reason") == "specialist_timeout"
        assert isinstance(answer, str) and answer
        # No additional architect synthesis call should be made after timeout.
        assert primitives.llm_call.call_count == 0

    def test_specialist_code_report_rescue_avoids_extra_turns(self):
        from src.api.routes.chat_delegation import _run_specialist_loop

        raw = """
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = 0.0

    def refill(self, now):
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now

    def consume(self, n=1):
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

class RateLimiter:
    def __init__(self):
        self.buckets = {}
""".strip()
        primitives = MagicMock()
        primitives.llm_call = MagicMock(return_value=raw)

        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_result = MagicMock()
            mock_result.is_final = False
            mock_result.final_answer = ""
            mock_result.output = ""
            mock_result.error = ""
            mock_repl.execute.return_value = mock_result
            mock_repl.get_state.return_value = ""
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            with patch("src.prompt_builders.extract_code_from_response", return_value=raw), patch(
                "src.prompt_builders.auto_wrap_final", return_value=raw
            ):
                (
                    report,
                    _tools,
                    _called,
                    _timings,
                    timed_out,
                    report_rescued,
                    _infer,
                    _repl_errors,
                ) = _run_specialist_loop(
                    question="q",
                    context="",
                    brief="b",
                    delegate_to="coder_escalation",
                    delegate_mode="react",
                    primitives=primitives,
                    tool_registry=None,
                )

        assert timed_out is False
        assert report_rescued is True
        assert "class TokenBucket" in report
        # Single specialist generation turn only.
        assert primitives.llm_call.call_count == 1

    def test_specialist_final_answer_marks_report_rescued(self):
        from src.api.routes.chat_delegation import _run_specialist_loop

        primitives = MagicMock()
        primitives.llm_call = MagicMock(return_value="FINAL('10482163')")

        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_result = MagicMock()
            mock_result.is_final = True
            mock_result.final_answer = "10482163"
            mock_result.output = ""
            mock_result.error = None
            mock_repl.execute.return_value = mock_result
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            (
                report,
                _tools,
                _called,
                _timings,
                timed_out,
                report_rescued,
                _infer,
                _repl_errors,
            ) = _run_specialist_loop(
                question="q",
                context="",
                brief="b",
                delegate_to="coder_escalation",
                delegate_mode="repl",
                primitives=primitives,
                tool_registry=None,
            )

        assert timed_out is False
        assert report == "10482163"
        assert report_rescued is True

    def test_specialist_timeout_exception_sets_timed_out_flag(self):
        from src.api.routes.chat_delegation import _run_specialist_loop

        primitives = MagicMock()
        primitives.llm_call = MagicMock(side_effect=TimeoutError("Request timed out after 82s"))

        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            (
                report,
                _tools,
                _called,
                _timings,
                timed_out,
                report_rescued,
                _infer,
                _repl_errors,
            ) = _run_specialist_loop(
                question="q",
                context="",
                brief="b",
                delegate_to="coder_escalation",
                delegate_mode="repl",
                primitives=primitives,
                tool_registry=None,
            )

        assert timed_out is True
        assert report.startswith("[Delegation failed:")
        assert report_rescued is False

    def test_architect_returns_rescued_specialist_report_directly(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="should_not_be_called")
        state = MagicMock()
        state.tool_registry = None

        decisions = iter([
            ("I|brief:investigate|to:coder_escalation", 1, 0),
            ("D|should_not_be_called", 1, 0),
        ])
        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "test-key"
        mock_cache.get.return_value = None

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            side_effect=lambda *a, **kw: next(decisions),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=("specialist report body", 0, [], [], False, True, {}, []),
        ), patch(
            "src.delegation_cache.get_delegation_cache",
            return_value=mock_cache,
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
                force_response_on_cap=True,
            )

        assert answer == "specialist report body"
        assert stats.get("break_reason") == "specialist_report"
        assert primitives.llm_call.call_count == 0

    def test_rescued_long_specialist_report_keeps_full_answer(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="compact summary")
        state = MagicMock()
        state.tool_registry = None

        full_report = "FULL SPECIALIST ANSWER\n" + ("detail line\n" * 400)
        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "long-rescue-key"
        mock_cache.get.return_value = None

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            return_value=("I|brief:investigate|to:coder_escalation", 1, 0),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(full_report, 0, [], [], False, True, {}, []),
        ), patch(
            "src.delegation_cache.get_delegation_cache",
            return_value=mock_cache,
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
                force_response_on_cap=True,
            )

        assert answer == full_report
        assert stats["specialist_output"] == full_report
        assert stats["report_handles"]
        assert stats.get("break_reason") == "specialist_report"
        cache_args = mock_cache.put.call_args.args
        assert cache_args[1] == full_report
        assert "[REPORT_HANDLE id=" not in answer

    def test_approved_non_rescued_long_report_returns_full_report(self, monkeypatch, tmp_path):
        """C5/DCP-13: D|Approved hands the user the full report, not handle+summary text."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REPORT_DIR", str(tmp_path))
        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="compact summary")
        state = MagicMock()
        state.tool_registry = None

        full_report = "FULL SPECIALIST DOCUMENT\n" + ("design detail line\n" * 400)
        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "approved-key"
        mock_cache.get.return_value = None
        arch_prompts: list[str] = []
        decisions = iter([
            ("I|brief:draft the document|to:coder_escalation", 1, 0),
            ("D|Approved", 1, 0),
        ])

        def _decision(prompt_text, *_a, **_kw):
            arch_prompts.append(prompt_text)
            return next(decisions)

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            side_effect=_decision,
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(full_report, 0, [], [], False, False, {}, []),
        ), patch(
            "src.delegation_cache.get_delegation_cache",
            return_value=mock_cache,
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
                force_response_on_cap=True,
            )

        assert answer == full_report
        assert "[REPORT_HANDLE" not in answer
        assert stats["specialist_output"] == full_report
        assert stats["report_handles"]
        # The architect loop prompt still carries the compact handle+summary text.
        assert "[REPORT_HANDLE id=" in arch_prompts[1]
        assert "fetch_report" not in arch_prompts[1]
        assert "compact summary" in arch_prompts[1]
        cache_kwargs = mock_cache.put.call_args.kwargs
        assert cache_kwargs["full_report"] == full_report
        assert cache_kwargs["report_handle"] == stats["report_handles"][0]

    def test_non_rescued_cache_hit_keeps_full_user_report_and_compact_loop_text(self):
        from types import SimpleNamespace

        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        state = MagicMock()
        state.tool_registry = None
        full_report = "FULL CACHED REPORT\n" + ("details\n" * 300)
        loop_report = "[REPORT_HANDLE id=report-123 chars=2400 sha16=abc]\nSummary:\nshort form"
        handle = {"id": "report-123", "chars": "2400", "sha16": "abc"}
        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "cached-key"
        mock_cache.get.return_value = SimpleNamespace(
            report=loop_report,
            full_report=full_report,
            report_handle=handle,
            age_seconds=1.0,
        )
        arch_prompts: list[str] = []
        decisions = iter([
            ("I|brief:investigate|to:coder_escalation", 1, 0),
            ("D|Approved", 1, 0),
        ])

        def _decision(prompt_text, *_a, **_kw):
            arch_prompts.append(prompt_text)
            return next(decisions)

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            side_effect=_decision,
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
        ) as run_specialist, patch(
            "src.delegation_cache.get_delegation_cache",
            return_value=mock_cache,
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
                force_response_on_cap=True,
            )

        assert answer == full_report
        assert stats["specialist_output"] == full_report
        assert stats["delegation_cache_hits"] == 1
        assert stats["report_handles"] == [handle]
        assert "short form" in arch_prompts[1]
        assert full_report not in arch_prompts[1]
        run_specialist.assert_not_called()

    def test_legacy_compact_cache_entry_is_not_used_as_full_report(self):
        from types import SimpleNamespace

        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="summary")
        state = MagicMock()
        state.tool_registry = None
        full_report = "NEW FULL SPECIALIST REPORT"
        legacy_entry = SimpleNamespace(
            report="legacy compact summary",
            report_handle=None,
            age_seconds=1.0,
        )
        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "legacy-key"
        mock_cache.get.return_value = None
        arch_prompts: list[str] = []
        decisions = iter([
            ("I|brief:investigate|to:coder_escalation", 1, 0),
            ("D|Approved", 1, 0),
        ])

        def _decision(prompt_text, *_a, **_kw):
            arch_prompts.append(prompt_text)
            return next(decisions)

        def _get_cache_entry(*_args, **_kwargs):
            return legacy_entry

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            side_effect=_decision,
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(full_report, 0, [], [], False, False, {}, []),
        ) as run_specialist, patch(
            "src.delegation_cache.get_delegation_cache",
            return_value=mock_cache,
        ):
            mock_cache.get.side_effect = _get_cache_entry
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
                force_response_on_cap=True,
            )

        assert answer == full_report
        assert stats["delegation_cache_misses"] == 1
        assert stats.get("delegation_cache_hits", 0) == 0
        assert stats["specialist_output"] == full_report
        assert "legacy compact summary" not in arch_prompts[1]
        run_specialist.assert_called_once()

    def test_timeout_skip_synthesis_non_rescued_long_report_returns_full_report(
        self, monkeypatch, tmp_path
    ):
        """C5/DCP-13: skip_synthesis_on_timeout returns the full report, not handle text."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REPORT_DIR", str(tmp_path))
        monkeypatch.setenv("ORCHESTRATOR_DELEGATION_SKIP_SYNTHESIS_ON_TIMEOUT", "1")
        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="compact summary")
        state = MagicMock()
        state.tool_registry = None

        full_report = "PARTIAL SPECIALIST FINDINGS\n" + ("evidence line\n" * 400)
        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "timeout-key"
        mock_cache.get.return_value = None

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            return_value=("I|brief:investigate|to:coder_escalation", 1, 0),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=(full_report, 0, [], [], True, False, {}, []),
        ), patch(
            "src.delegation_cache.get_delegation_cache",
            return_value=mock_cache,
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=2,
                force_response_on_cap=True,
            )

        assert stats.get("break_reason") == "specialist_timeout"
        assert answer == full_report
        assert "[REPORT_HANDLE" not in answer
        assert stats["report_handles"]
        # Only the worker_summarize compression call ran; no forced architect synthesis.
        roles = [c.kwargs.get("role") for c in primitives.llm_call.call_args_list]
        assert roles == ["worker_summarize"]


# ── Prompt Builder Tests ─────────────────────────────────────────────────


class TestArchitectPromptBuilders:
    """Tests for architect prompt template functions."""

    def test_investigate_prompt_has_decision_instructions(self):
        from src.prompt_builders import build_architect_investigate_prompt

        prompt = build_architect_investigate_prompt("What is X?")
        assert "D|" in prompt
        assert "I|" in prompt
        assert "brief:" in prompt
        assert "Question:" in prompt
        assert "open questions" in prompt
        assert "Do NOT propose an implementation" in prompt

    def test_investigate_prompt_includes_context(self):
        from src.prompt_builders import build_architect_investigate_prompt

        prompt = build_architect_investigate_prompt("What is X?", context="some ctx")
        assert "some ctx" in prompt
        assert "not a solution plan" in prompt

    def test_synthesis_prompt_includes_report(self):
        from src.prompt_builders import build_architect_synthesis_prompt

        prompt = build_architect_synthesis_prompt(
            "What is X?",
            "Report: found Y",
            loop_num=1,
            max_loops=3,
        )
        assert "Report: found Y" in prompt
        assert "Question:" in prompt

    def test_synthesis_prompt_no_reinvestigate_on_good_report(self):
        """Non-failed specialist report suppresses I| re-delegation option."""
        from src.prompt_builders import build_architect_synthesis_prompt

        prompt = build_architect_synthesis_prompt(
            "Q",
            "R",
            loop_num=2,
            max_loops=3,
        )
        assert "Do NOT" in prompt
        assert "I|brief" not in prompt

    def test_synthesis_prompt_reinvestigate_on_failed_report(self):
        """Failed specialist report allows re-delegation with loop count."""
        from src.prompt_builders import build_architect_synthesis_prompt

        prompt = build_architect_synthesis_prompt(
            "Q",
            "[ERROR: specialist crashed]",
            loop_num=1,
            max_loops=3,
        )
        assert "I|brief" in prompt
        assert "1" in prompt and "3" in prompt

    def test_synthesis_prompt_does_not_advertise_fetch_report(self):
        """C4/DCP-13: the synthesis decision breaks on turn 0, so fetch_report is unreachable."""
        from src.prompt_builders import build_architect_synthesis_prompt
        from src.prompt_builders.review import _ARCHITECT_SYNTHESIS_FALLBACK

        prompt = build_architect_synthesis_prompt(
            "Q",
            "[REPORT_HANDLE id=coder_escalation-1-abc chars=9000 sha16=abc]\nSummary:\n- x",
            loop_num=1,
            max_loops=3,
        )
        assert "fetch_report" not in prompt
        assert "fetch_report" not in _ARCHITECT_SYNTHESIS_FALLBACK


# ── Whitelist Tests ──────────────────────────────────────────────────────


class TestFileToolsWhitelist:
    """Tests that file tools are in REACT_TOOL_WHITELIST."""

    def test_read_file_in_whitelist(self):
        from src.prompt_builders import REACT_TOOL_WHITELIST

        assert "read_file" in REACT_TOOL_WHITELIST

    def test_list_directory_in_whitelist(self):
        from src.prompt_builders import REACT_TOOL_WHITELIST

        assert "list_directory" in REACT_TOOL_WHITELIST

    def test_original_tools_still_present(self):
        from src.prompt_builders import REACT_TOOL_WHITELIST

        assert "web_search" in REACT_TOOL_WHITELIST
        assert "calculate" in REACT_TOOL_WHITELIST
        assert "python_eval" in REACT_TOOL_WHITELIST


# ── Seeding Script Tests ─────────────────────────────────────────────────
