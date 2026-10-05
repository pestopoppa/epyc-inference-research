"""Unit tests for graph helper utilities."""

import json
from types import SimpleNamespace

from src.escalation import ErrorCategory
from src.graph.helpers import (
    _add_evidence,
    _classify_error,
    _detect_role_cycle,
    _looks_like_prompt_echo,
    _loop_guard_classification_view,
    _record_mitigation,
    _select_and_broadcast_workspace_delta,
    _should_attempt_prose_rescue,
    _workspace_prompt_block,
    _update_workspace_from_turn,
)
from src.graph.state import TaskState


class TestClassifyError:
    def test_classifies_timeout(self):
        assert _classify_error("Request timed out after 30s") == ErrorCategory.TIMEOUT

    def test_classifies_schema(self):
        assert _classify_error("JSON schema validation failed") == ErrorCategory.SCHEMA

    def test_classifies_code(self):
        assert _classify_error("TypeError: unsupported operand type") == ErrorCategory.CODE

    def test_classifies_unknown(self):
        assert _classify_error("completely novel failure") == ErrorCategory.UNKNOWN


class TestRoleCycleDetection:
    def test_detects_period_2_cycle(self):
        assert _detect_role_cycle(["frontdoor", "coder", "frontdoor", "coder"]) is True

    def test_detects_period_3_cycle(self):
        assert _detect_role_cycle(
            ["frontdoor", "coder", "architect", "frontdoor", "coder", "architect"]
        ) is True

    def test_no_cycle_for_short_history(self):
        assert _detect_role_cycle(["frontdoor", "coder", "architect"]) is False


class TestWorkspaceSelection:
    def test_select_and_broadcast_prioritizes_open_questions(self):
        ws = {
            "broadcast_version": 0,
            "proposals": [
                {"kind": "commitment", "owner": "coder", "text": "do X", "priority": "normal"},
                {"kind": "open_question", "owner": "coder", "text": "need API key", "priority": "high"},
                {"kind": "commitment", "owner": "architect", "text": "do Y", "priority": "high"},
            ],
            "open_questions": [],
            "commitments": [],
            "resolved_questions": [],
            "broadcast_log": [],
        }

        _select_and_broadcast_workspace_delta(ws)

        assert ws["broadcast_version"] == 1
        assert len(ws["broadcast_log"]) == 1
        assert any(item["text"] == "need API key" for item in ws["open_questions"])
        assert any(item["text"] == "do Y" for item in ws["commitments"])

    def test_update_workspace_from_turn_caps_proposals(self):
        state = TaskState(
            task_id="t1",
            prompt="Investigate issue",
            turns=1,
        )
        ws = state.workspace_state
        ws["proposals"] = [
            {"kind": "commitment", "owner": "worker", "text": f"old-{i}", "priority": "normal"}
            for i in range(12)
        ]

        _update_workspace_from_turn(
            state=state,
            role="coder",
            output="new proposal",
            error=None,
        )

        assert len(ws["proposals"]) == 12
        assert ws["proposals"][-1]["text"] == "new proposal"
        assert ws["updated_at"] != ""

    def test_workspace_prompt_includes_task_progress_and_warning(self):
        state = TaskState(task_id="t1", prompt="Do work")
        state.task_manager.create(subject="Step one", description="desc")
        state.anti_pattern_warning = "Avoid repeated timeout retries."

        prompt_block = _workspace_prompt_block(state)

        assert "task_progress" in prompt_block
        assert "Step one" in prompt_block
        assert "Avoid repeated timeout retries." in prompt_block


class TestFailureAndHypothesisHooks:
    def test_record_mitigation_uses_failure_id_and_action(self):
        captured = {}

        class _FakeFailureGraph:
            def record_mitigation(self, failure_id, action, worked):
                captured["failure_id"] = failure_id
                captured["action"] = action
                captured["worked"] = worked

        ctx = SimpleNamespace(
            state=SimpleNamespace(last_failure_id="f-123"),
            deps=SimpleNamespace(failure_graph=_FakeFailureGraph()),
        )

        _record_mitigation(ctx, "frontdoor", "coder_escalation")

        assert captured["failure_id"] == "f-123"
        assert captured["action"] == "escalate:frontdoor->coder_escalation"
        assert captured["worked"] is True

    def test_add_evidence_uses_outcome_and_source(self):
        captured = {}

        class _FakeHypothesisGraph:
            def add_evidence(self, hypothesis_id, outcome, source):
                captured["hypothesis_id"] = hypothesis_id
                captured["outcome"] = outcome
                captured["source"] = source
                return 0.75

        ctx = SimpleNamespace(
            state=SimpleNamespace(task_id="task-1", current_role="frontdoor", turns=3),
            deps=SimpleNamespace(hypothesis_graph=_FakeHypothesisGraph()),
        )

        _add_evidence(ctx, "success", 0.5)

        assert captured["hypothesis_id"] == "task-1"
        assert captured["outcome"] == "success"
        assert captured["source"] == "frontdoor:turn_3"


class TestProseRescueGuards:
    def test_prompt_echo_detected(self):
        text = "Answer with the letter only. Question: 2+2=? Options: A)3 B)4"
        assert _looks_like_prompt_echo(text) is True

    def test_short_answer_like_output_allowed(self):
        assert _should_attempt_prose_rescue("The answer is B", "") is True

    def test_prompt_echo_not_allowed(self):
        raw = "Answer with the letter only. Question: 2+2=? The answer is B"
        assert _should_attempt_prose_rescue(raw, "") is False

    def test_code_fence_not_allowed(self):
        raw = "```python\nprint('x')\n```"
        assert _should_attempt_prose_rescue(raw, "") is False


class TestLoopGuardClassificationView:
    def _extract_refusal_code(self, raw_arguments: str) -> str:
        from src.prompt_builders.code_utils import extract_code_from_response

        response = json.dumps(
            [{
                "id": "call_refused",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": raw_arguments,
                },
            }]
        )
        return extract_code_from_response(response)

    def test_refusal_echo_markers_do_not_count_but_remain_in_original_code(self):
        for marker in ("FINAL(", "file_write_safe("):
            raw_arguments = f"{{query: {marker}private-diagnostic"
            code = self._extract_refusal_code(raw_arguments)

            assert marker in code
            assert "private-diagnostic" in code
            classification = _loop_guard_classification_view(code)
            assert marker not in classification
            assert "[ERROR:" in code  # Raw refusal remains available for execution/display.

    def test_mixed_valid_and_refused_calls_keep_real_call_progress(self):
        from src.prompt_builders.code_utils import extract_code_from_response

        response = json.dumps([
            {
                "id": "call_valid",
                "function": {
                    "name": "file_write_safe",
                    "arguments": json.dumps({"path": "answer.txt", "content": "ready"}),
                },
            },
            {
                "id": "call_refused",
                "function": {"name": "web_search", "arguments": "{query: FINAL(private"},
            },
        ])
        code = extract_code_from_response(response)
        classification = _loop_guard_classification_view(code)

        assert 'result_0 = CALL("file_write_safe", path="answer.txt", content="ready")' in classification
        assert "file_write_safe" in classification
        assert ("file_write_safe" in classification) or ("FINAL(" in classification)
        assert 'result_1 = "[ERROR:' in code
        assert "FINAL(" in code
        assert "FINAL(" not in classification

    def test_unicode_line_separator_in_prior_literal_does_not_shift_refusal_span(self):
        prefix = 'status = "before\u0085after"\n'
        code = prefix + self._extract_refusal_code("{query: FINAL(private")

        classification = _loop_guard_classification_view(code)

        assert classification.startswith(prefix)
        assert "FINAL(" not in classification

    def test_real_progress_markers_are_unchanged(self):
        for code in ("FINAL('answer')", "file_write_safe('answer.py', 'content')"):
            assert _loop_guard_classification_view(code) == code

    def test_malformed_python_falls_back_to_original_classification_text(self):
        code = "result = '''[ERROR: tool call 'x' NOT executed: Raw: FINAL("
        assert _loop_guard_classification_view(code) == code
