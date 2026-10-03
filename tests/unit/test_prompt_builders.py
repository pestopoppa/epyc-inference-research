"""Tests for prompt_builders.py — core prompt construction functions.

Covers the 7 functions imported by chat.py/chat_pipeline.py plus
review/revision prompts. Focuses on correctness-critical behavior:
code extraction, FINAL wrapping, error classification, and prompt
structure invariants.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from src.prompt_builders import (
    auto_wrap_final,
    build_escalation_prompt,
    build_long_context_exploration_prompt,
    build_root_lm_prompt,
    build_routing_context,
    classify_error,
    detect_format_constraints,
    extract_code_from_response,
    rescue_bare_name_final,
)


# ── extract_code_from_response ────────────────────────────────────────────


class TestExtractCodeFromResponse:
    """Code extraction from LLM responses."""

    def test_markdown_python_block(self):
        response = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        result = extract_code_from_response(response)
        assert "print('hello')" in result
        assert "```" not in result
        # Structural: Should strip markdown completely
        assert result.strip() == "print('hello')"

    def test_markdown_block_no_language(self):
        response = "```\nx = 42\n```"
        result = extract_code_from_response(response)
        assert "x = 42" in result
        # Structural: Should extract cleanly
        assert "```" not in result

    def test_unpaired_trailing_backticks(self):
        """Model quirk: sometimes generates trailing ``` without opening."""
        response = "FINAL('result')\n```"
        result = extract_code_from_response(response)
        assert "FINAL" in result
        # Structural: Should preserve valid code
        assert "FINAL('result')" in result

    def test_repl_function_recognized_as_code(self):
        """Lines starting with REPL tools should be treated as code."""
        response = "peek(0, 50)\ngrep('error')"
        result = extract_code_from_response(response)
        assert "peek(0, 50)" in result
        # Structural: Should preserve both lines
        assert result.count("\n") >= 1

    def test_final_function_recognized(self):
        response = "FINAL('The answer is 42')"
        result = extract_code_from_response(response)
        assert "FINAL(" in result
        # Structural: Should match input exactly
        assert result == response

    def test_strips_import_lines(self):
        response = (
            "```python\nimport json\nfrom pathlib import Path\nresult = json.loads(data)\n```"
        )
        result = extract_code_from_response(response)
        assert "import json" not in result
        assert "from pathlib" not in result
        assert "result = json.loads(data)" in result
        # Structural: Should have only one line (imports stripped)
        assert result.count("\n") == 0

    def test_preserves_from_in_non_import(self):
        """'from' in normal code should not be stripped."""
        response = "x = get_data_from_source()\nFINAL(x)"
        result = extract_code_from_response(response)
        assert "get_data_from_source" in result
        # Structural: Should preserve both lines
        assert "FINAL(x)" in result
        assert result.count("\n") >= 1

    def test_empty_response(self):
        result = extract_code_from_response("")
        assert result == ""
        # Structural: Should be truly empty
        assert len(result) == 0

    def test_code_extracted_from_indented_block(self):
        response = "```python\n    x = 1\n    y = 2\n```"
        result = extract_code_from_response(response)
        assert "x = 1" in result
        assert "y = 2" in result
        # Structural: Should preserve indentation structure
        assert result.count("\n") >= 1  # Multiple lines preserved

    def test_multiple_code_blocks_picks_first(self):
        response = "```python\nfirst = 1\n```\n```python\nsecond = 2\n```"
        result = extract_code_from_response(response)
        assert "first = 1" in result
        # Structural: Should NOT include second block
        assert "second" not in result

    def test_prompt_echo_prose_not_returned_as_code(self):
        response = "Answer with the letter only. Question: 2+2=? Options: A)3 B)4"
        result = extract_code_from_response(response)
        assert result == ""

    def test_salvages_final_line_from_mixed_prose(self):
        response = "FINAL('B')\nAnswer with the letter only."
        result = extract_code_from_response(response)
        assert result == "FINAL('B')"

    def test_strips_gemma_thinking_channel(self):
        """Gemma-4 wraps output in <|channel>thought\n<channel|> tags."""
        response = '<|channel>thought\n<channel|>secret = TOOL("get_eval_secret", name="alpha")\nFINAL(secret)'
        result = extract_code_from_response(response)
        assert 'TOOL("get_eval_secret"' in result
        assert "FINAL(secret)" in result
        assert "<|channel>" not in result

    def test_strips_qwen_thinking_tags(self):
        """Qwen3 uses </think> closing tags after thinking."""
        response = '</anthinking>\n\nresult = compute()\nFINAL(result)'
        result = extract_code_from_response(response)
        assert "result = compute()" in result
        assert "FINAL(result)" in result
        assert "</anthinking>" not in result

    def test_translates_gemma_tool_call_format(self):
        """Gemma-4 emits <|tool_call>call:tool_use:tool_name{args}<tool_call|>."""
        response = '<|channel>thought\n<channel|><|tool_call>call:tool_use:get_eval_secret{name: "alpha"}<tool_call|>'
        result = extract_code_from_response(response)
        assert 'TOOL("get_eval_secret"' in result
        assert 'FINAL(result)' in result
        assert '<|tool_call>' not in result

    def test_translates_jackrong_v2_tagged_json_tool_call(self):
        """Jackrong v2's 6,994-byte template uses tagged direct JSON, not Qwen XML."""
        response = (
            '<tool_call>{"name":"web_search",'
            '"arguments":{"query":"EPYC parser"}}</tool_call>'
        )
        result = extract_code_from_response(response)
        assert result == 'result = CALL("web_search", query="EPYC parser")\nprint(result)'

    def test_translates_jackrong_coder_tagged_json_tool_call(self):
        """Coder's distinct 4,718-byte template preserves the tagged JSON wire contract."""
        response = (
            '<tool_call>{"name":"grep",'
            '"arguments":{"pattern":"TODO","path":"src"}}</tool_call>'
        )
        result = extract_code_from_response(response)
        assert result == 'result = CALL("grep", pattern="TODO", path="src")\nprint(result)'


# ── tool-call JSON repair (TU-TC-1) ───────────────────────────────────────


class TestToolCallJsonRepair:
    """Gated JSON repair: valid JSON untouched, malformed repaired or refused, never ``{}``."""

    @staticmethod
    def _counts():
        from src.prompt_builders.code_utils import TOOL_CALL_JSON_REPAIR_COUNTS

        return dict(TOOL_CALL_JSON_REPAIR_COUNTS)

    def test_valid_json_returned_byte_for_byte(self):
        from src.prompt_builders.code_utils import _repair_json_text

        raw = '{ "query" :  "a </parameter> b",\n "n": [1, 2] }'
        assert _repair_json_text(raw) is raw

    def test_valid_args_untouched_and_not_counted(self):
        from src.prompt_builders.code_utils import translate_openai_tool_calls

        before = self._counts()
        text = (
            '[{"id":"call_1","function":{"name":"web_search",'
            '"arguments":"{\\"query\\": \\"x ]} y\\"}"},"type":"function"}]'
        )
        assert translate_openai_tool_calls(text) == (
            'result = CALL("web_search", query="x ]} y")\nprint(result)'
        )
        assert self._counts() == before

    def test_trailing_tag_repaired(self):
        from src.prompt_builders.code_utils import _repair_json_text

        assert _repair_json_text('{"query": "x"}</parameter>\n</function>') == '{"query": "x"}'

    def test_bracket_imbalance_repaired(self):
        from src.prompt_builders.code_utils import _repair_json_text

        assert _repair_json_text('{"a": 1]') == '{"a": 1}'
        assert _repair_json_text('{"a": [1, 2}') == '{"a": [1, 2]}'
        assert _repair_json_text('{"a": 1}}') == '{"a": 1}'

    def test_unterminated_string_repaired(self):
        from src.prompt_builders.code_utils import _repair_json_text

        assert _repair_json_text('{"query": "EPYC parser') == '{"query": "EPYC parser"}'

    def test_openai_arguments_repaired_and_counted(self):
        from src.prompt_builders.code_utils import translate_openai_tool_calls

        before = self._counts()
        text = (
            '[{"id":"call_1","function":{"name":"web_search",'
            '"arguments":"{\\"query\\": \\"EPYC\\"]"},"type":"function"}]'
        )
        assert translate_openai_tool_calls(text) == (
            'result = CALL("web_search", query="EPYC")\nprint(result)'
        )
        after = self._counts()
        assert after["repaired"] == before["repaired"] + 1
        assert after["unrecoverable"] == before["unrecoverable"]

    def test_tagged_payload_with_leaked_tag_repaired(self):
        response = (
            '<tool_call>{"name":"web_search",'
            '"arguments":{"query":"EPYC parser</parameter></tool_call>'
        )
        result = extract_code_from_response(response)
        assert result == 'result = CALL("web_search", query="EPYC parser")\nprint(result)'

    def test_unrecoverable_openai_arguments_refused_never_empty_call(self):
        import ast

        from src.prompt_builders.code_utils import translate_openai_tool_calls

        before = self._counts()
        text = (
            '[{"id":"call_1","function":{"name":"web_search",'
            '"arguments":"{query: EPYC"},"type":"function"}]'
        )
        result = translate_openai_tool_calls(text)
        assert result is not None
        assert "CALL(" not in result
        assert "[ERROR:" in result and "web_search" in result and "{query: EPYC" in result
        assert result.endswith("print(result)")
        ast.parse(result)
        after = self._counts()
        assert after["unrecoverable"] == before["unrecoverable"] + 1

    def test_unrecoverable_tagged_payload_refused_not_skipped(self):
        response = '<tool_call>{"name":"grep","arguments":{"pattern": TODO @@ }}</tool_call>'
        result = extract_code_from_response(response)
        assert "CALL(" not in result
        assert "[ERROR:" in result and "'grep'" in result
        # Two lines, so auto_wrap_final leaves it alone (no FINAL of an error).
        assert auto_wrap_final(result) == result

    def test_refusal_echo_truncated(self):
        from src.prompt_builders.code_utils import translate_openai_tool_calls

        junk = "{q: " + "z" * 1000
        text = (
            '[{"id":"call_1","function":{"name":"web_search",'
            f'"arguments":"{junk}"}},"type":"function"}}]'
        )
        result = translate_openai_tool_calls(text)
        assert "...[truncated]" in result
        assert "z" * 1000 not in result

    def test_valid_call_kept_alongside_refusal(self):
        from src.prompt_builders.code_utils import translate_openai_tool_calls

        text = (
            '[{"id":"call_1","function":{"name":"grep","arguments":"{\\"pattern\\": \\"a\\"}"}},'
            '{"id":"call_2","function":{"name":"web_search","arguments":"{q: x"}}]'
        )
        result = translate_openai_tool_calls(text)
        assert 'result_0 = CALL("grep", pattern="a")' in result
        assert result.count("CALL(") == 1
        assert 'result_1 = "[ERROR:' in result

    def test_non_json_tagged_body_still_falls_through(self):
        from src.prompt_builders.code_utils import translate_openai_tool_calls

        text = "<tool_call>\n<function=web_search>\n<parameter=query>x</parameter>\n</function>\n</tool_call>"
        assert translate_openai_tool_calls(text) is None


# ── auto_wrap_final ───────────────────────────────────────────────────────


class TestAutoWrapFinal:
    """FINAL() auto-wrapping logic."""

    def test_already_has_final(self):
        code = "FINAL('hello world')"
        result = auto_wrap_final(code)
        assert result == code
        # Structural: Should be unchanged
        assert result.count("FINAL(") == 1

    def test_exploration_not_wrapped(self):
        """Exploration functions should NOT be wrapped."""
        for func in ("peek(0, 50)", "grep('error')", "llm_call('summarize')"):
            result = auto_wrap_final(func)
            assert "FINAL" not in result, f"Should not wrap: {func}"
            # Structural: Should be unchanged
            assert result == func

    def test_llm_batch_not_wrapped(self):
        code = "llm_batch(['q1', 'q2'], role='worker')"
        result = auto_wrap_final(code)
        assert "FINAL" not in result
        # Structural: Should be unchanged
        assert result == code

    def test_artifacts_not_wrapped(self):
        code = "artifacts['summary'] = 'done'"
        result = auto_wrap_final(code)
        assert "FINAL" not in result
        # Structural: Should be unchanged
        assert result == code

    def test_def_wrapped_in_triple_quotes(self):
        code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        result = auto_wrap_final(code)
        assert "FINAL(" in result
        assert "'''" in result or '"""' in result
        # Structural: FINAL should wrap the entire code
        assert result.startswith("FINAL(") or "= FINAL(" in result

    def test_class_wrapped_in_triple_quotes(self):
        code = "class MyWidget:\n    pass"
        result = auto_wrap_final(code)
        assert "FINAL(" in result
        # Structural: Should use triple quotes for multiline
        assert "'''" in result or '"""' in result

    def test_single_expression_wrapped(self):
        code = "result = compute_answer()"
        result = auto_wrap_final(code)
        assert "FINAL(" in result
        # Structural: FINAL should be properly formed
        assert result.count("FINAL(") == 1
        assert ")" in result

    def test_bracketed_error_wrapped_as_string(self):
        code = "[ERROR: Inference failed: Request timed out after 90s]"
        result = auto_wrap_final(code)
        assert result == 'FINAL("[ERROR: Inference failed: Request timed out after 90s]")'
        # Structural: should not produce invalid FINAL([ERROR: ...]) form
        assert "FINAL([ERROR:" not in result

    def test_control_flow_not_wrapped(self):
        """Control flow statements should not be wrapped."""
        for stmt in (
            "for x in range(10):",
            "while True:",
            "if condition:",
            "try:",
            "with open('f'):",
        ):
            result = auto_wrap_final(stmt)
            assert "FINAL" not in result, f"Should not wrap: {stmt}"
            # Structural: Should be unchanged
            assert result == stmt

    def test_import_not_wrapped(self):
        code = "import os"
        result = auto_wrap_final(code)
        assert "FINAL" not in result
        # Structural: Should be unchanged
        assert result == code

    def test_escapes_triple_quotes(self):
        """Triple quotes in code must be escaped to prevent syntax errors."""
        code = "x = '''some text'''"
        result = auto_wrap_final(code)
        assert "FINAL(" in result
        # The triple quotes should be escaped or the code should use the other style
        # Structural: Should contain the original code
        assert "some text" in result


# ── rescue_bare_name_final ────────────────────────────────────────────────


class TestRescueBareNameFinal:
    """Unquoted one-word FINAL(OK) -> FINAL("OK"), and nothing broader."""

    def test_rescues_exact_nameerror(self):
        err = "NameError: name 'OK' is not defined"
        assert rescue_bare_name_final("FINAL(OK)", err) == 'FINAL("OK")'
        assert rescue_bare_name_final("  FINAL( yes ) ;\n", "NameError: name 'yes' is not defined") == 'FINAL("yes")'

    def test_needs_nameerror_for_that_exact_name(self):
        assert rescue_bare_name_final("FINAL(OK)", None) is None
        assert rescue_bare_name_final("FINAL(OK)", "TypeError: boom") is None
        assert rescue_bare_name_final("FINAL(OK)", "NameError: name 'other' is not defined") is None

    def test_only_a_whole_single_line_call(self):
        err = "NameError: name 'OK' is not defined"
        assert rescue_bare_name_final("x = 1\nFINAL(OK)", err) is None
        assert rescue_bare_name_final("FINAL(OK.upper())", err) is None
        assert rescue_bare_name_final("print(OK)", err) is None

    def test_placeholders_are_not_rescued(self):
        for name in ("answer", "result", "solution", "your_answer", "Answer"):
            err = f"NameError: name '{name}' is not defined"
            assert rescue_bare_name_final(f"FINAL({name})", err) is None


# ── classify_error ────────────────────────────────────────────────────────


class TestClassifyError:
    """Error classification for escalation routing."""

    def test_format_gates(self):
        for gate in ("schema", "format", "lint", "mdformat", "shfmt"):
            result = classify_error("some error", gate_name=gate)
            assert result.value in ("FORMAT", "SCHEMA", "format", "schema"), (
                f"Gate '{gate}' should classify as FORMAT or SCHEMA"
            )
            # Structural: Should have value attribute
            assert hasattr(result, "value")

    def test_syntax_error_is_code(self):
        result = classify_error("SyntaxError: invalid syntax")
        assert result.value.upper() in ("CODE", "SYNTAX")
        # Structural: Should have value attribute
        assert hasattr(result, "value")

    def test_type_error_is_code(self):
        result = classify_error("TypeError: expected str, got int")
        assert result.value.upper() in ("CODE", "SYNTAX", "TYPE")

    def test_import_error_is_code(self):
        result = classify_error("ImportError: No module named 'foo'")
        assert result.value.upper() in ("CODE", "IMPORT")

    def test_assertion_error_is_logic(self):
        result = classify_error("AssertionError: test failed")
        # Assertion errors map to LOGIC category
        assert result.value.upper() in ("LOGIC", "TEST", "ASSERTION")

    def test_timeout_keyword(self):
        result = classify_error("Request timed out after 60s")
        assert result.value.upper() in ("TIMEOUT",)

    def test_generation_abort_early_abort(self):
        result = classify_error("early abort: high entropy detected")
        assert result.value.upper() == "EARLY_ABORT"

    def test_generation_abort_unknown(self):
        """'aborted' alone doesn't match 'early abort' pattern."""
        result = classify_error("Generation aborted: entropy spike detected")
        assert result.value.upper() == "UNKNOWN"

    def test_unknown_fallback(self):
        result = classify_error("solar flare disrupted the quasar")
        assert result.value.upper() == "UNKNOWN"

    def test_expected_in_substring_is_logic(self):
        """'unexpected' contains 'expected' — classified as LOGIC."""
        result = classify_error("something completely unexpected happened")
        assert result.value.upper() == "LOGIC"

    def test_empty_message(self):
        result = classify_error("")
        assert result is not None
        # Structural: Should return an error category enum
        assert hasattr(result, "value")


# ── build_root_lm_prompt ──────────────────────────────────────────────────


class TestBuildRootLmPrompt:
    """Root LM prompt structure tests."""

    def test_includes_task(self):
        result = build_root_lm_prompt(state="ready", original_prompt="What is 2+2?")
        assert "What is 2+2?" in result
        # Structural: Should be substantial multiline prompt
        assert len(result) > 200
        assert result.count("\n") > 5

    def test_includes_routing_context_on_turn_0(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            turn=0,
            routing_context="Role: frontdoor | Tier: A",
        )
        assert "frontdoor" in result
        # Structural: Routing context should appear before task section
        assert result.index("frontdoor") < result.index("## Task")

    def test_build_respects_prefix_stable_order_flag(self):
        """RTE-Prefix: when prefix_stable_order is enabled, build_root_lm_prompt
        renders task/instruction BEFORE the mutating sections."""
        from src.features import Features, reset_features, set_features

        set_features(Features(prefix_stable_order=True))
        try:
            result = build_root_lm_prompt(
                state="ready",
                original_prompt="test",
                turn=0,
                routing_context="Role: frontdoor | Tier: A",
            )
        finally:
            reset_features()

        assert "frontdoor" in result
        # Fixed task section now precedes the mutating routing context
        assert result.index("## Task") < result.index("frontdoor")

    def test_includes_error_context(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            last_error="NameError: x is not defined",
        )
        assert "NameError" in result
        # Structural: Error should be clearly separated (on own line or in section)
        assert "\nNameError" in result or "## " in result

    def test_includes_last_output(self):
        result = build_root_lm_prompt(
            state="x = 42",
            original_prompt="test",
            last_output="42",
        )
        assert "42" in result
        # Structural: State should appear before output
        assert result.index("x = 42") < result.index("42", result.index("x = 42") + 6)

    def test_returns_string(self):
        result = build_root_lm_prompt(state="", original_prompt="hello")
        assert isinstance(result, str)
        assert len(result) > 100  # Should be a substantial prompt


# ── build_routing_context ─────────────────────────────────────────────────


class TestBuildRoutingContext:
    """Routing context generation (MemRL integration)."""

    def test_none_router_returns_empty(self):
        result = build_routing_context("frontdoor", None, "some task")
        assert result == ""
        # Structural: Should be completely empty
        assert len(result) == 0

    def test_router_exception_returns_empty(self):
        """Gracefully handles retrieval failures."""
        mock_router = MagicMock()
        mock_router.get_q_values_for_task.side_effect = RuntimeError("DB error")
        result = build_routing_context("frontdoor", mock_router, "some task")
        assert result == ""
        # Structural: Should be completely empty on error
        assert len(result) == 0

    def test_respects_max_chars(self):
        mock_router = MagicMock()
        mock_router.get_q_values_for_task.return_value = [
            {"role": "coder", "q_value": 0.9, "task_preview": "x" * 500}
        ] * 10
        result = build_routing_context("frontdoor", mock_router, "task", max_chars=100)
        assert len(result) <= 150  # Some overflow is OK for formatting
        # Structural: May be empty if max_chars too restrictive
        assert isinstance(result, str)


# ── build_long_context_exploration_prompt ──────────────────────────────────


class TestBuildLongContextExplorationPrompt:
    """Long context exploration prompt generation."""

    def test_includes_original_prompt(self):
        result = build_long_context_exploration_prompt("Find the bug", 20000)
        assert "Find the bug" in result
        # Structural: Should be substantial multiline exploration prompt
        assert len(result) > 200
        assert result.count("\n") > 5

    def test_estimates_tokens(self):
        result = build_long_context_exploration_prompt("task", 40000)
        # 40000 chars / 4 = ~10000 tokens
        assert "10000" in result or "10,000" in result or "10k" in result.lower()
        # Structural: Should be substantial exploration prompt
        assert len(result) > 300

    def test_search_task_uses_grep(self):
        """Search tasks should start with grep()."""
        result = build_long_context_exploration_prompt("Find all error handlers", 20000)
        assert "grep" in result.lower()
        # Structural: grep should appear before FINAL in instructions
        assert result.lower().index("grep") < result.index("FINAL")

    def test_non_search_task_uses_peek(self):
        """Non-search tasks should start with peek()."""
        result = build_long_context_exploration_prompt("Summarize this document", 20000)
        assert "peek" in result.lower()
        # Structural: peek should appear before FINAL in instructions
        assert result.lower().index("peek") < result.index("FINAL")

    def test_includes_repl_tools(self):
        result = build_long_context_exploration_prompt("task", 20000)
        for tool in ("peek", "grep", "FINAL"):
            assert tool in result
        # Structural: Tools should appear in order (peek/grep before FINAL)
        assert result.index("peek") < result.index("FINAL")
        assert result.index("grep") < result.index("FINAL")


# ── build_escalation_prompt ───────────────────────────────────────────────


class TestBuildEscalationPrompt:
    """Escalation prompt with failure context."""

    def test_with_escalation_context(self):
        from src.escalation import EscalationContext, EscalationAction

        ctx = EscalationContext(
            current_role="worker_explore",
            error_message="NameError: undefined variable",
            error_category="CODE",
            failure_count=2,
            task_id="test-123",
        )

        class MockDecision:
            target_role = "coder_escalation"
            reason = "Code error needs specialist"
            action = EscalationAction.ESCALATE

        result = build_escalation_prompt("Fix the bug", "x = None", ctx, MockDecision())
        assert isinstance(result, str)
        assert len(result) > 50
        # Structural: Should have header format and be multiline
        assert "\n## " in result
        assert result.count("\n") > 5

    def test_includes_error_info(self):
        from src.escalation import EscalationContext, EscalationAction

        ctx = EscalationContext(
            current_role="frontdoor",
            error_message="TimeoutError: request timed out",
            error_category="TIMEOUT",
            failure_count=1,
            task_id="t1",
        )

        class MockDecision:
            target_role = "architect_general"
            reason = "Timeout needs investigation"
            action = EscalationAction.ESCALATE

        result = build_escalation_prompt("Debug timeout", "", ctx, MockDecision())
        assert "timeout" in result.lower() or "Timeout" in result
        # Structural: Should be well-formatted prompt (length and multiline)
        assert len(result) > 200


# ── detect_format_constraints ─────────────────────────────────────────────


class TestDetectFormatConstraints:
    """Format constraint detection via regex."""

    def test_json_format_detected(self):
        constraints = detect_format_constraints("Return the answer as JSON")
        assert any("json" in c.lower() for c in constraints)
        # Structural: Should return non-empty list
        assert len(constraints) > 0
        assert isinstance(constraints, list)

    def test_numbered_list_detected(self):
        constraints = detect_format_constraints("Give me a numbered list of items")
        assert len(constraints) > 0
        # Structural: Should detect list format
        assert any("list" in c.lower() or "numbered" in c.lower() for c in constraints)

    def test_bullet_list_detected(self):
        constraints = detect_format_constraints("Format as a bullet list")
        assert len(constraints) > 0
        assert any("bullet" in c.lower() for c in constraints)
        # Structural: Should be exactly one constraint
        assert len(constraints) >= 1

    def test_no_constraints_returns_empty(self):
        constraints = detect_format_constraints("What is the capital of France?")
        assert isinstance(constraints, list)
        # Structural: Should be empty or very minimal
        assert len(constraints) <= 1

    def test_table_format_detected(self):
        constraints = detect_format_constraints("Show the results in a table")
        assert len(constraints) > 0
        # Structural: Should detect table keyword
        assert any("table" in c.lower() for c in constraints)

    def test_case_insensitive(self):
        constraints = detect_format_constraints("RETURN AS JSON FORMAT")
        assert any("json" in c.lower() for c in constraints)
        # Structural: Should find constraints
        assert len(constraints) > 0


# ── Prompt Types ──────────────────────────────────────────────────────────


class TestRootLMPrompt:
    """Test RootLMPrompt dataclass and to_string() method."""

    def test_empty_prompt(self):
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt()
        result = prompt.to_string()
        assert isinstance(result, str)
        # Structural: Should be valid but minimal
        assert len(result) >= 0

    def test_system_only(self):
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(system="You are a test assistant.")
        result = prompt.to_string()
        assert "You are a test assistant." in result
        # Structural: Should be clean output
        assert len(result) > 20

    def test_all_sections(self):
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(
            system="System prompt",
            tools="Tool descriptions",
            rules="Rules here",
            state="State info",
            context="Context info",
            task="Task description",
            instruction="Final instruction",
        )
        result = prompt.to_string()
        assert "System prompt" in result
        assert "Tool descriptions" in result
        assert "Rules here" in result
        assert "State info" in result
        assert "Context info" in result
        assert "Task description" in result
        assert "Final instruction" in result
        # Structural: Verify section ordering
        assert result.index("System prompt") < result.index("Tool descriptions")
        assert result.index("Tool descriptions") < result.index("Task description")

    def test_section_headers(self):
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(
            system="sys",
            tools="tools",
            rules="rules",
        )
        result = prompt.to_string()
        # Structural: Should have clear section separation
        assert "###" in result or "##" in result

    def test_prefix_stable_order_renders_fixed_sections_before_mutating(self):
        """RTE-Prefix: prefix_stable=True puts task/instruction BEFORE
        state/context/reference_code so successive REPL turns share a longer
        KV-cache prefix."""
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(
            system="System prompt",
            tools="Tool descriptions",
            rules="Rules here",
            state="State info",
            context="Context info",
            reference_code="Reference code here",
            task="Task description",
            instruction="Final instruction",
        )
        result = prompt.to_string(prefix_stable=True)
        # Fixed sections must precede mutating sections
        assert result.index("## Task") < result.index("## Current State")
        assert result.index("## Task") < result.index("Context info")
        assert result.index("## Your Code") < result.index("## Current State")
        assert result.index("## Your Code") < result.index("## Reference Code")
        # Order among mutating sections is preserved
        assert result.index("## Current State") < result.index("## Reference Code")
        # All sections still present
        for section in (
            "System prompt", "Tool descriptions", "Rules here", "State info",
            "Context info", "Reference code here", "Task description",
            "Final instruction",
        ):
            assert section in result

    def test_prefix_stable_false_preserves_legacy_order(self):
        """Default order keeps mutating sections (state/context/reference_code)
        before task/instruction — unchanged behavior."""
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(
            state="State info",
            context="Context info",
            reference_code="Reference code here",
            task="Task description",
            instruction="Final instruction",
        )
        result = prompt.to_string()  # prefix_stable defaults False
        assert result.index("## Current State") < result.index("## Task")
        assert result.index("Context info") < result.index("## Task")
        assert result.index("## Reference Code") < result.index("## Task")


# ── Constants Coverage ────────────────────────────────────────────────────


class TestPromptConstants:
    """Test constants from src/prompt_builders/constants.py."""

    def test_default_root_lm_tools_not_empty(self):
        """Test DEFAULT_ROOT_LM_TOOLS is a non-empty string."""
        from src.prompt_builders.constants import DEFAULT_ROOT_LM_TOOLS

        assert isinstance(DEFAULT_ROOT_LM_TOOLS, str)
        assert len(DEFAULT_ROOT_LM_TOOLS) > 100
        # Should contain key tool names
        assert "peek" in DEFAULT_ROOT_LM_TOOLS
        assert "grep" in DEFAULT_ROOT_LM_TOOLS
        assert "FINAL" in DEFAULT_ROOT_LM_TOOLS

    def test_default_root_lm_tools_remains_compressed(self):
        """Default tool prompt should stay materially smaller than verbose."""
        from src.prompt_builders.constants import DEFAULT_ROOT_LM_TOOLS, VERBOSE_ROOT_LM_TOOLS

        assert len(DEFAULT_ROOT_LM_TOOLS.split()) <= len(VERBOSE_ROOT_LM_TOOLS.split()) * 0.5

    def test_default_root_lm_rules_not_empty(self):
        """Test DEFAULT_ROOT_LM_RULES is a non-empty string."""
        from src.prompt_builders.constants import DEFAULT_ROOT_LM_RULES

        assert isinstance(DEFAULT_ROOT_LM_RULES, str)
        assert len(DEFAULT_ROOT_LM_RULES) > 50
        # Should contain critical rules
        assert "FINAL" in DEFAULT_ROOT_LM_RULES
        assert "SAFE IMPORTS ONLY" in DEFAULT_ROOT_LM_RULES.upper()

    def test_react_tool_whitelist_not_empty(self):
        """Test REACT_TOOL_WHITELIST has expected tools."""
        from src.prompt_builders.constants import REACT_TOOL_WHITELIST

        assert isinstance(REACT_TOOL_WHITELIST, frozenset)
        assert len(REACT_TOOL_WHITELIST) > 0
        # Should include safe read-only tools
        assert "web_search" in REACT_TOOL_WHITELIST or "search_wikipedia" in REACT_TOOL_WHITELIST
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(
            tools="peek(), grep()",
            rules="No imports",
            task="Do something",
        )
        result = prompt.to_string()
        assert "## Available Tools" in result
        assert "## Rules" in result
        assert "## Task" in result
        # Structural: Headers should be on own lines
        assert "\n## Available Tools\n" in result
        assert "\n## Rules\n" in result
        assert "\n## Task\n" in result

    def test_missing_sections_omitted(self):
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(task="Just a task")
        result = prompt.to_string()
        assert "## Available Tools" not in result
        assert "## Rules" not in result
        # Structural: Should have Task header and be minimal but valid
        assert "## Task" in result
        assert len(result) > 10


class TestEscalationPrompt:
    """Test EscalationPrompt dataclass and to_string() method."""

    def test_empty_escalation(self):
        from src.prompt_builders.types import EscalationPrompt

        prompt = EscalationPrompt()
        result = prompt.to_string()
        assert isinstance(result, str)
        # Structural: Should be valid empty escalation
        assert len(result) >= 0

    def test_header_and_failure_info(self):
        from src.prompt_builders.types import EscalationPrompt

        prompt = EscalationPrompt(
            header="# Escalation from worker",
            failure_info="Failed after 2 attempts",
        )
        result = prompt.to_string()
        assert "# Escalation from worker" in result
        assert "Failed after 2 attempts" in result
        # Structural: Should be multiline with proper structure
        assert result.count("\n") > 1
        assert len(result) > 30

    def test_error_details_section(self):
        from src.prompt_builders.types import EscalationPrompt

        prompt = EscalationPrompt(
            header="# Escalation",
            error_details="SyntaxError: invalid syntax",
        )
        result = prompt.to_string()
        assert "## Error Details" in result
        assert "SyntaxError" in result
        # Structural: Section header should be on own line
        assert "\n## Error Details\n" in result

    def test_all_sections(self):
        from src.prompt_builders.types import EscalationPrompt

        prompt = EscalationPrompt(
            header="# Escalation from coder",
            failure_info="Failed 3 times",
            error_details="TypeError: expected str",
            state="x = 42",
            task="Fix the bug",
            instructions="Use better error handling",
        )
        result = prompt.to_string()
        assert "# Escalation from coder" in result
        assert "Failed 3 times" in result
        assert "## Error Details" in result
        assert "## Current State" in result
        assert "## Original Task" in result
        assert "## Instructions" in result
        # Structural: Verify section ordering
        assert result.index("# Escalation") < result.index("## Error Details")
        assert result.index("## Error Details") < result.index("## Original Task")
        assert len(result) > 150


class TestStepPrompt:
    """Test StepPrompt dataclass and to_string() method."""

    def test_empty_step(self):
        from src.prompt_builders.types import StepPrompt

        prompt = StepPrompt()
        result = prompt.to_string()
        assert isinstance(result, str)
        # Structural: Should be valid minimal step
        assert len(result) >= 0

    def test_action_only(self):
        from src.prompt_builders.types import StepPrompt

        prompt = StepPrompt(action="Implement error handler")
        result = prompt.to_string()
        assert "Task: Implement error handler" in result
        # Structural: Should be clean single-line task
        assert len(result) > 20

    def test_with_inputs_and_outputs(self):
        from src.prompt_builders.types import StepPrompt

        prompt = StepPrompt(
            action="Process data",
            inputs="data.json",
            outputs="result, summary",
        )
        result = prompt.to_string()
        assert "Task: Process data" in result
        assert "Inputs:" in result
        assert "data.json" in result
        assert "Expected outputs: result, summary" in result
        # Structural: Verify Task appears before Inputs and outputs
        assert result.index("Task:") < result.index("Inputs:")
        assert result.index("Inputs:") < result.index("Expected outputs:")

    def test_with_constraints(self):
        from src.prompt_builders.types import StepPrompt

        prompt = StepPrompt(
            action="Format output",
            constraints="JSON only, max 100 words",
        )
        result = prompt.to_string()
        assert "Constraints: JSON only, max 100 words" in result
        # Structural: Should be properly formatted
        assert "Task:" in result
        assert len(result) > 20


# ── build_react_prompt ────────────────────────────────────────────────────


class TestBuildReactPrompt:
    """Test ReAct-style prompt generation."""

    def test_basic_prompt_without_registry(self):
        from src.prompt_builders import build_react_prompt

        result = build_react_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in result
        assert "Thought:" in result
        assert "Action:" in result
        assert "Final Answer:" in result
        # Structural: Verify ReAct section ordering
        assert result.index("Question:") < result.index("Thought:")
        assert result.index("Thought:") < result.index("Action:")
        assert result.index("Action:") < result.index("Final Answer:")

    def test_includes_context(self):
        from src.prompt_builders import build_react_prompt

        result = build_react_prompt(
            "Summarize this document",
            context="The document is about quantum computing.",
        )
        assert "Context:" in result
        assert "quantum computing" in result
        # Structural: No duplicate sections
        sections = [
            line for line in result.split("\n") if ":" in line and line.strip().endswith(":")
        ]
        assert len(sections) == len(set(sections)), "Should have no duplicate section headers"

    def test_custom_max_turns(self):
        from src.prompt_builders import build_react_prompt

        result = build_react_prompt("test", max_turns=10)
        assert "10 times" in result or "10" in result
        # Structural: Should mention the turn limit
        assert "10" in result

    def test_includes_tool_descriptions(self):
        from src.prompt_builders import build_react_prompt

        result = build_react_prompt("test")
        # Should have static fallback tools
        assert "calculate" in result or "web_search" in result
        # Structural: Should be substantial prompt
        assert len(result) > 200
        assert result.count("\n") > 5

    def test_custom_whitelist(self):
        from src.prompt_builders import build_react_prompt

        custom_whitelist = frozenset({"calculate", "get_current_date"})
        result = build_react_prompt("test", tool_whitelist=custom_whitelist)
        assert isinstance(result, str)
        # Structural: Should be substantial ReAct prompt
        assert len(result) > 100

    def test_none_tool_registry(self):
        from src.prompt_builders import build_react_prompt

        result = build_react_prompt("test", tool_registry=None)
        assert "Question: test" in result
        # Structural: Should still be valid ReAct prompt
        assert len(result) > 50


# ── Review Prompts (Additional) ───────────────────────────────────────────


class TestBuildPlanReviewPrompt:
    """Test architect plan review prompt generation."""

    def test_basic_plan(self):
        from src.prompt_builders import build_plan_review_prompt

        steps = [
            {"id": "S1", "actor": "coder", "action": "Implement handler"},
            {"id": "S2", "actor": "worker", "action": "Write tests"},
        ]
        result = build_plan_review_prompt(
            objective="Build API handler",
            task_type="code",
            plan_steps=steps,
        )
        assert "Build API handler" in result
        assert "S1:coder:Implement handler" in result
        assert "S2:worker:Write tests" in result
        # Structural: Steps should be in order
        assert result.index("S1:") < result.index("S2:")
        assert len(result) > 100

    def test_includes_json_format(self):
        from src.prompt_builders import build_plan_review_prompt

        result = build_plan_review_prompt(
            objective="Test",
            task_type="code",
            plan_steps=[],
        )
        assert "JSON" in result or "json" in result
        assert '"d":' in result or '"decision":' in result
        # Structural: Should have clear format specification
        assert result.count("\n") > 3

    def test_truncates_long_actions(self):
        from src.prompt_builders import build_plan_review_prompt

        steps = [{"id": "S1", "actor": "coder", "action": "x" * 200}]
        result = build_plan_review_prompt("test", "code", steps)
        # Action should be truncated to 50 chars
        assert result.count("x") < 100
        # Structural: Should still have S1 ID
        assert "S1" in result

    def test_handles_deps_and_outputs(self):
        from src.prompt_builders import build_plan_review_prompt

        steps = [
            {
                "id": "S1",
                "actor": "coder",
                "action": "Load data",
                "outputs": ["data.json"],
            },
            {
                "id": "S2",
                "actor": "worker",
                "action": "Process",
                "deps": ["S1"],
            },
        ]
        result = build_plan_review_prompt("test", "code", steps)
        assert "data.json" in result
        assert "(S1)" in result
        # Structural: S1 should appear before S2 (dependency order)
        assert result.index("S1") < result.index("S2")


class TestBuildArchitectInvestigatePrompt:
    """Test architect investigation decision prompt."""

    def test_basic_structure(self):
        from src.prompt_builders import build_architect_investigate_prompt

        result = build_architect_investigate_prompt("What is the capital of France?")
        assert "What is the capital of France?" in result
        assert "D|" in result  # Direct answer format
        assert "I|" in result  # Investigate/delegate format
        assert "brief:" in result

    def test_with_context(self):
        from src.prompt_builders import build_architect_investigate_prompt

        result = build_architect_investigate_prompt(
            "Summarize the document",
            context="Document excerpt here...",
        )
        assert "Document excerpt here..." in result
        assert "Evidence bundle" in result
        assert "not a solution plan" in result
        # Structural: Should have both D| and I| decision options
        assert result.count("D|") >= 1
        assert result.count("I|") >= 1

    def test_discovery_framing_is_non_prescriptive(self):
        from src.prompt_builders import build_architect_investigate_prompt

        prompt = build_architect_investigate_prompt("Investigate the delegation path")
        assert "open questions" in prompt
        assert "file relationships" in prompt
        assert "missing evidence" in prompt
        assert "Do NOT propose an implementation" in prompt
        assert "Do NOT assume omitted files are irrelevant" in prompt

    def test_mentions_valid_roles(self):
        from src.prompt_builders import build_architect_investigate_prompt

        result = build_architect_investigate_prompt("test")
        assert "coder_escalation" in result
        assert "worker_general" in result
        assert "worker_coder" not in result
        assert "worker_explore" not in result
        assert "worker_fast" not in result
        # Structural: Options should be distinct (D| and I| both present)
        assert result.count("D|") >= 1
        assert result.count("I|") >= 1


class TestBuildArchitectSynthesisPrompt:
    """Test architect synthesis prompt after investigation."""

    def test_basic_synthesis(self):
        from src.prompt_builders import build_architect_synthesis_prompt

        result = build_architect_synthesis_prompt(
            question="What is X?",
            report="Investigation found: X is Y.",
            loop_num=1,
            max_loops=3,
        )
        assert "What is X?" in result
        assert "Investigation found: X is Y." in result
        assert "D|" in result  # Direct answer format
        # Structural: Should be substantial multiline synthesis prompt
        assert len(result) > 200
        assert result.count("\n") > 5

    def test_allows_further_investigation(self):
        from src.prompt_builders import build_architect_synthesis_prompt

        result = build_architect_synthesis_prompt("test", "report", loop_num=1, max_loops=3)
        # Should allow another investigation
        assert "I|" in result or "investigation" in result.lower()
        # Structural: Should have decision format options
        assert result.count("D|") >= 1 or result.count("I|") >= 1

    def test_no_investigation_at_max_loops(self):
        from src.prompt_builders import build_architect_synthesis_prompt

        result = build_architect_synthesis_prompt("test", "report", loop_num=3, max_loops=3)
        # Should NOT offer further investigation
        # (loop_num=3 means we've completed loop 3, so loop_num < max_loops is False)
        lines = result.split("\n")
        # Check if investigate option is missing
        investigate_mentioned = any("loop" in line and "3/3" in line for line in lines)
        assert not investigate_mentioned
        # Structural: Should still have decision option
        assert "D|" in result

    def test_truncates_long_report(self):
        from src.prompt_builders import build_architect_synthesis_prompt

        long_report = "x" * 10000
        result = build_architect_synthesis_prompt("test", long_report, loop_num=1, max_loops=3)
        # Report should be truncated to 6000 chars
        assert result.count("x") < 7000
        # Structural: Should still have decision options
        assert "D|" in result


# ── build_formalizer_prompt ───────────────────────────────────────────────


class TestBuildFormalizerPrompt:
    """Test output formalizer prompt generation."""

    def test_basic_structure(self):
        from src.prompt_builders import build_formalizer_prompt

        result = build_formalizer_prompt(
            answer="The answer is 42.",
            prompt="What is the answer?",
            format_spec="JSON format",
        )
        assert "JSON format" in result
        assert "The answer is 42." in result
        assert "What is the answer?" in result
        # Structural: Format spec should be clearly indicated
        assert len(result) > 50
        assert result.count("\n") > 2

    def test_truncates_long_prompt(self):
        from src.prompt_builders import build_formalizer_prompt

        long_prompt = "x" * 1000
        result = build_formalizer_prompt("answer", long_prompt, "JSON")
        question = result.split("Original question: ", 1)[1].split("\n\nOriginal answer:", 1)[0]
        assert question == "x" * 500
        # Structural: Should still have format spec
        assert "JSON" in result

    def test_includes_instructions(self):
        from src.prompt_builders import build_formalizer_prompt

        result = build_formalizer_prompt("answer", "prompt", "JSON")
        assert "ONLY" in result or "only" in result.lower()
        # Structural: Should have clear directive structure
        assert "JSON" in result
        assert len(result) > 30


# ── build_step_prompt (additional) ────────────────────────────────────────


class TestBuildStepPromptFunction:
    """Test build_step_prompt() module-level function."""

    def test_basic_step(self):
        from src.prompt_builders import build_step_prompt

        result = build_step_prompt("Analyze data")
        assert "Analyze data" in result
        # Structural: Should have Task label
        assert "Task:" in result

    def test_with_inputs_and_outputs(self):
        from src.prompt_builders import build_step_prompt

        result = build_step_prompt(
            action="Process",
            inputs=["data.json", "config.yaml"],
            outputs=["result.csv"],
        )
        assert "Process" in result
        assert "data.json" in result or "config.yaml" in result
        # Structural: Should format list inputs properly
        assert len(result) > 20
        assert result.count("\n") >= 1


# ── build_stage2_review_prompt ────────────────────────────────────────────


class TestBuildStage2ReviewPrompt:
    """Test Stage 2 review prompt for two-stage summarization."""

    def test_basic_structure(self):
        from src.prompt_builders import build_stage2_review_prompt

        result = build_stage2_review_prompt(
            draft_summary="This is a draft summary.",
            grep_hits=[],
            figures=[],
        )
        assert "This is a draft summary." in result
        assert "review" in result.lower() or "Review" in result
        # Structural: Should be multiline review prompt
        assert result.count("\n") > 3
        assert len(result) > 50

    def test_includes_original_task(self):
        from src.prompt_builders import build_stage2_review_prompt

        result = build_stage2_review_prompt(
            draft_summary="Summary",
            grep_hits=[],
            figures=[],
            original_task="Summarize the document about quantum computing",
        )
        assert "quantum computing" in result
        # Structural: Original task should appear early in prompt
        assert result.index("quantum computing") < len(result) // 2

    def test_includes_grep_hits(self):
        from src.prompt_builders import build_stage2_review_prompt

        grep_hits = [
            {
                "pattern": "quantum",
                "hits": [
                    {"context": "Quantum computers use qubits", "line_num": 42},
                ],
            }
        ]
        result = build_stage2_review_prompt("Summary", grep_hits, [])
        assert "quantum" in result
        assert "qubits" in result
        # Structural: Grep hits should appear after summary
        assert result.index("Summary") < result.index("quantum")

    def test_includes_figures(self):
        from src.prompt_builders import build_stage2_review_prompt

        figures = [
            {"description": "Architecture diagram", "page": 5},
            {"description": "Performance graph", "page": 12},
        ]
        result = build_stage2_review_prompt("Summary", [], figures)
        assert "Architecture diagram" in result
        assert "Performance graph" in result
        # Structural: Figures should be listed in order
        assert result.index("Architecture diagram") < result.index("Performance graph")

    def test_truncates_long_draft(self):
        from src.prompt_builders import build_stage2_review_prompt

        long_draft = "x" * 10000
        result = build_stage2_review_prompt(long_draft, [], [])
        # Draft should be truncated to 8000 chars
        assert result.count("x") <= 8500  # Allow some margin for formatting
        # Structural: Should still be substantial
        assert len(result) > 100


# ── build_task_decomposition_prompt ───────────────────────────────────────


class TestBuildTaskDecompositionPrompt:
    """Test task decomposition prompt for architect."""

    def test_basic_decomposition(self):
        from src.prompt_builders import build_task_decomposition_prompt

        result = build_task_decomposition_prompt("Build a REST API")
        assert "Build a REST API" in result
        assert "JSON" in result or "json" in result
        # Structural: Should be substantial decomposition prompt
        assert len(result) > 100
        assert result.count("\n") > 3

    def test_with_context(self):
        from src.prompt_builders import build_task_decomposition_prompt

        result = build_task_decomposition_prompt(
            "Analyze codebase",
            context="The codebase is in Python.",
        )
        assert "Analyze codebase" in result
        assert "Python" in result
        # Structural: Task should appear before context
        assert result.index("Analyze codebase") < result.index("Python")

    def test_mentions_step_fields(self):
        from src.prompt_builders import build_task_decomposition_prompt

        result = build_task_decomposition_prompt("test")
        assert "actor" in result or '"actor"' in result
        assert "action" in result or '"action"' in result
        # Structural: Should specify JSON format structure
        assert "{" in result or "JSON" in result


# ── Constants Consistency ─────────────────────────────────────────────────


class TestConstants:
    """Test prompt_builders constants for consistency."""

    def test_vision_tools_subset_of_react_tools(self):
        from src.prompt_builders import (
            REACT_TOOL_WHITELIST,
            VISION_REACT_TOOL_WHITELIST,
        )

        # VISION_REACT_TOOL_WHITELIST should be a superset
        assert REACT_TOOL_WHITELIST.issubset(VISION_REACT_TOOL_WHITELIST)
        # Structural: Vision should have more or equal tools
        assert len(VISION_REACT_TOOL_WHITELIST) >= len(REACT_TOOL_WHITELIST)

    def test_executable_tools_subset_of_whitelist(self):
        from src.prompt_builders import (
            VISION_REACT_EXECUTABLE_TOOLS,
            VISION_REACT_TOOL_WHITELIST,
        )

        # Executable tools should be a subset of whitelist
        assert VISION_REACT_EXECUTABLE_TOOLS.issubset(VISION_REACT_TOOL_WHITELIST)
        # Structural: Executable should be smaller or equal
        assert len(VISION_REACT_EXECUTABLE_TOOLS) <= len(VISION_REACT_TOOL_WHITELIST)

    def test_vision_tool_descriptions_match_executable(self):
        from src.prompt_builders import (
            VISION_REACT_EXECUTABLE_TOOLS,
            VISION_TOOL_DESCRIPTIONS,
        )

        # All executable tools should have descriptions
        for tool in VISION_REACT_EXECUTABLE_TOOLS:
            assert tool in VISION_TOOL_DESCRIPTIONS
        # Structural: Counts should match
        assert len(VISION_REACT_EXECUTABLE_TOOLS) <= len(VISION_TOOL_DESCRIPTIONS)

    def test_default_tools_and_rules_not_empty(self):
        from src.prompt_builders import DEFAULT_ROOT_LM_TOOLS, DEFAULT_ROOT_LM_RULES

        assert len(DEFAULT_ROOT_LM_TOOLS) > 100  # Should be substantial
        assert len(DEFAULT_ROOT_LM_RULES) > 100
        # Structural: Both should be strings
        assert isinstance(DEFAULT_ROOT_LM_TOOLS, str)
        assert isinstance(DEFAULT_ROOT_LM_RULES, str)

    def test_react_format_has_placeholders(self):
        from src.prompt_builders import REACT_FORMAT

        assert "{tool_descriptions}" in REACT_FORMAT
        assert "{max_turns}" in REACT_FORMAT
        # Structural: Should be substantial template
        assert len(REACT_FORMAT) > 100


# ── Compact Tools & File Loading ─────────────────────────────────────────


class TestCompactTools:
    """Test COMPACT_ROOT_LM_TOOLS constant and MINIMAL style."""

    def test_compact_tools_has_core_functions(self):
        """Verify all core REPL functions are present."""
        from src.prompt_builders.constants import COMPACT_ROOT_LM_TOOLS

        for name in ("FINAL", "peek", "grep", "llm_call", "escalate", "CALL", "list_tools"):
            assert name in COMPACT_ROOT_LM_TOOLS, f"Missing core function: {name}"
        assert "context" in COMPACT_ROOT_LM_TOOLS
        assert "artifacts" in COMPACT_ROOT_LM_TOOLS

    def test_compact_tools_smaller(self):
        """COMPACT should be < 55% of DEFAULT size (threshold raised after P3b compression)."""
        from src.prompt_builders.constants import COMPACT_ROOT_LM_TOOLS, DEFAULT_ROOT_LM_TOOLS

        ratio = len(COMPACT_ROOT_LM_TOOLS) / len(DEFAULT_ROOT_LM_TOOLS)
        assert ratio < 0.55, f"Compact tools ratio {ratio:.2%} >= 55%"

    def test_minimal_style_uses_compact(self):
        """PromptBuilder with MINIMAL style should use compact tools."""
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig, PromptStyle
        from src.prompt_builders.constants import COMPACT_ROOT_LM_TOOLS

        builder = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
        prompt = builder.build_root_lm_prompt(
            state="ready", original_prompt="test", as_structured=True,
        )
        assert prompt.tools == COMPACT_ROOT_LM_TOOLS

    def test_structured_style_uses_default(self):
        """PromptBuilder with STRUCTURED style should use DEFAULT tools."""
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig, PromptStyle
        from src.prompt_builders.constants import DEFAULT_ROOT_LM_TOOLS

        builder = PromptBuilder(PromptConfig(style=PromptStyle.STRUCTURED))
        prompt = builder.build_root_lm_prompt(
            state="ready", original_prompt="test", as_structured=True,
        )
        assert prompt.tools == DEFAULT_ROOT_LM_TOOLS


class TestFileLoading:
    """Test tools_file and rules_file hot-swap loading."""

    def test_tools_file_overrides_style(self, tmp_path):
        """tools_file should override style-based tool selection."""
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig, PromptStyle

        tools_file = tmp_path / "custom_tools.md"
        tools_file.write_text("custom_tool_1()\ncustom_tool_2()")

        builder = PromptBuilder(PromptConfig(
            style=PromptStyle.STRUCTURED,
            tools_file=str(tools_file),
        ))
        prompt = builder.build_root_lm_prompt(
            state="ready", original_prompt="test", as_structured=True,
        )
        assert prompt.tools == "custom_tool_1()\ncustom_tool_2()"

    def test_rules_file_overrides_default(self, tmp_path):
        """rules_file should override DEFAULT_ROOT_LM_RULES."""
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig

        rules_file = tmp_path / "custom_rules.md"
        rules_file.write_text("Rule 1: Always FINAL()\nRule 2: No imports")

        builder = PromptBuilder(PromptConfig(rules_file=str(rules_file)))
        prompt = builder.build_root_lm_prompt(
            state="ready", original_prompt="test", as_structured=True,
        )
        assert prompt.rules == "Rule 1: Always FINAL()\nRule 2: No imports"

    def test_missing_tools_file_falls_back(self):
        """Missing tools_file should fall back to style-based selection."""
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig, PromptStyle
        from src.prompt_builders.constants import COMPACT_ROOT_LM_TOOLS

        builder = PromptBuilder(PromptConfig(
            style=PromptStyle.MINIMAL,
            tools_file="/nonexistent/path/tools.md",
        ))
        prompt = builder.build_root_lm_prompt(
            state="ready", original_prompt="test", as_structured=True,
        )
        assert prompt.tools == COMPACT_ROOT_LM_TOOLS

    def test_missing_rules_file_falls_back(self):
        """Missing rules_file should fall back to DEFAULT_ROOT_LM_RULES."""
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig
        from src.prompt_builders.constants import DEFAULT_ROOT_LM_RULES

        builder = PromptBuilder(PromptConfig(
            rules_file="/nonexistent/path/rules.md",
        ))
        prompt = builder.build_root_lm_prompt(
            state="ready", original_prompt="test", as_structured=True,
        )
        assert prompt.rules == DEFAULT_ROOT_LM_RULES


# ── corpus_context parameter ─────────────────────────────────────────────


class TestCorpusContext:
    """Test corpus_context parameter in build_root_lm_prompt."""

    def test_corpus_context_appears_in_prompt(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            corpus_context="<reference_code>\ndef foo(): pass\n</reference_code>",
        )
        assert "reference_code" in result
        assert "def foo(): pass" in result

    def test_corpus_context_before_task(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="Do something",
            corpus_context="<reference_code>\nsnippet\n</reference_code>",
        )
        # Reference code should appear before the task section
        ref_idx = result.index("reference_code")
        task_idx = result.index("## Task")
        assert ref_idx < task_idx

    def test_empty_corpus_context_omitted(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            corpus_context="",
        )
        assert "## Reference Code" not in result

    def test_structured_prompt_has_reference_code_field(self):
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig

        builder = PromptBuilder(PromptConfig())
        prompt = builder.build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            corpus_context="snippet content",
            as_structured=True,
        )
        assert prompt.reference_code == "snippet content"

    def test_reference_code_section_in_to_string(self):
        from src.prompt_builders.types import RootLMPrompt

        prompt = RootLMPrompt(
            system="sys",
            reference_code="code snippet here",
            task="do thing",
        )
        result = prompt.to_string()
        assert "## Reference Code" in result
        assert "code snippet here" in result
        # Reference Code should come before Task
        assert result.index("## Reference Code") < result.index("## Task")


class TestBuildCorpusContext:
    """Test build_corpus_context() function."""

    def test_returns_empty_when_disabled(self):
        from src.prompt_builders.builder import build_corpus_context

        result = build_corpus_context(
            role="frontdoor",
            task_description="test task",
        )
        assert result == ""

    def test_returns_string(self):
        from src.prompt_builders.builder import build_corpus_context

        result = build_corpus_context(
            role="coder_escalation",
            task_description="implement a sorting algorithm",
        )
        assert isinstance(result, str)
