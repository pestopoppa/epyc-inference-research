#!/usr/bin/env python3
"""Unit tests for the REPL environment."""

from src.repl_environment import (
    REPLConfig,
    REPLEnvironment,
    REPLSecurityError,
)


class TestREPLBasicExecution:
    """Test basic code execution in the REPL."""

    def test_simple_print(self):
        """Test that print() output is captured."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute('print("hello")')

        assert result.output.strip() == "hello"
        assert not result.is_final
        assert result.error is None

    def test_context_available(self):
        """Test that context variable is accessible."""
        repl = REPLEnvironment(context="This is the full context")
        result = repl.execute("print(len(context))")

        assert "24" in result.output  # len("This is the full context")
        assert not result.is_final

    def test_context_content(self):
        """Test that context contains the expected content."""
        repl = REPLEnvironment(context="Hello World")
        result = repl.execute("print(context)")

        assert "Hello World" in result.output

    def test_artifacts_available(self):
        """Test that artifacts dict is accessible."""
        repl = REPLEnvironment(context="test", artifacts={"key": "value"})
        result = repl.execute("print(artifacts['key'])")

        assert "value" in result.output

    def test_artifacts_can_be_modified(self):
        """Test that artifacts can be written to."""
        repl = REPLEnvironment(context="test")
        repl.execute("artifacts['result'] = 42")
        result = repl.execute("print(artifacts['result'])")

        assert "42" in result.output
        assert repl.artifacts["result"] == 42

    def test_multiline_code(self):
        """Test execution of multiline code."""
        repl = REPLEnvironment(context="test")
        code = """
x = 5
y = 10
print(x + y)
"""
        result = repl.execute(code)

        assert "15" in result.output

    def test_syntax_error_captured(self):
        """Test that syntax errors are captured."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(")

        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error_captured(self):
        """Test that runtime errors are captured."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("x = 1 / 0")

        assert result.error is not None
        assert "ZeroDivisionError" in result.error


class TestPeekFunction:
    """Test the peek() function."""

    def test_peek_default(self):
        """Test peek with default n=500."""
        context = "A" * 1000
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(len(peek()))")

        assert "500" in result.output

    def test_peek_custom_n(self):
        """Test peek with custom n."""
        context = "Hello World"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(peek(5))")

        assert "Hello" in result.output

    def test_peek_larger_than_context(self):
        """Test peek when n > len(context)."""
        context = "Short"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(peek(100))")

        assert "Short" in result.output


class TestGrepFunction:
    """Test the grep() function."""

    def test_grep_finds_matches(self):
        """Test that grep finds matching lines."""
        context = """Line 1: foo
Line 2: bar
Line 3: foo bar
Line 4: baz"""
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(grep('foo'))")

        assert "Line 1" in result.output
        assert "Line 3" in result.output
        assert "Line 2" not in result.output

    def test_grep_case_insensitive(self):
        """Test that grep is case insensitive."""
        context = "FOO\nfoo\nFoO"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(len(grep('foo')))")

        assert "3" in result.output

    def test_grep_regex_pattern(self):
        """Test grep with regex pattern."""
        context = "test123\ntest456\nhello"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(grep(r'test\\d+'))")

        assert "test123" in result.output
        assert "test456" in result.output
        assert "hello" not in result.output

    def test_grep_invalid_regex(self):
        """Test grep with invalid regex returns error message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(grep('[invalid'))")

        assert "REGEX ERROR" in result.output

    def test_grep_truncation(self):
        """Test that grep results are truncated."""
        # Create context with many matching lines
        context = "\n".join([f"match line {i}" for i in range(200)])
        config = REPLConfig(max_grep_results=50)
        repl = REPLEnvironment(context=context, config=config)
        result = repl.execute("print(grep('match'))")

        assert "truncated" in result.output


class TestFinalFunction:
    """Test the FINAL() and FINAL_VAR() functions."""

    def test_final_signals_completion(self):
        """Test that FINAL() signals completion."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("FINAL('The answer is 42')")

        assert result.is_final
        assert result.final_answer == "The answer is 42"

    def test_final_with_computation(self):
        """Test FINAL with computed value."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("FINAL(str(2 + 2))")

        assert result.is_final
        assert result.final_answer == "4"

    def test_final_accepts_result_keyword(self):
        """FINAL(result=...) should normalize to the final answer."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('FINAL(result="computed answer")')

        assert result.is_final
        assert result.final_answer == "computed answer"

    def test_final_accepts_secret_keyword(self):
        """FINAL(secret=...) should normalize to the final answer."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('FINAL(secret="opaque-secret")')

        assert result.is_final
        assert result.final_answer == "opaque-secret"

    def test_final_rejects_unknown_keywords(self):
        """Unknown FINAL keyword arguments should produce a clear REPL error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('FINAL(result="ok", extra="bad")')

        assert not result.is_final
        assert result.error is not None
        assert "unsupported keyword" in result.error

    def test_final_var_returns_artifact(self):
        """Test that FINAL_VAR returns artifact content."""
        repl = REPLEnvironment(context="test")
        repl.execute("artifacts['result'] = 'computed value'")
        result = repl.execute("FINAL_VAR('result')")

        assert result.is_final
        assert result.final_answer == "computed value"

    def test_final_var_missing_key(self):
        """Test FINAL_VAR with missing key raises error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("FINAL_VAR('nonexistent')")

        assert not result.is_final
        assert result.error is not None
        assert "KeyError" in result.error

    def test_output_before_final_captured(self):
        """Test that output before FINAL is captured."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
print("before final")
FINAL("done")
""")

        assert result.is_final
        assert "before final" in result.output
        assert result.final_answer == "done"


class TestStructuredFinalExtraction:
    """Regression tests for FINAL extraction in structured mode."""

    def test_structured_final_triple_double_quotes(self):
        repl = REPLEnvironment(context="test", structured_mode=True)
        result = repl.execute('FINAL("""line1\\nline2""")')
        assert result.is_final is True
        assert result.final_answer == "line1\\nline2"

    def test_structured_final_single_quotes(self):
        repl = REPLEnvironment(context="test", structured_mode=True)
        result = repl.execute("FINAL('answer')")
        assert result.is_final is True
        assert result.final_answer == "answer"


class TestSecuritySandbox:
    """Test security restrictions in the sandbox."""

    def test_import_os_blocked(self):
        """Test that import os is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("import os")

        assert result.error is not None
        assert "not allowed" in result.error.lower() or "Dangerous" in result.error

    def test_import_subprocess_blocked(self):
        """Test that import subprocess is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("import subprocess")

        assert result.error is not None

    def test_from_os_import_blocked(self):
        """Test that from os import is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("from os import system")

        assert result.error is not None

    def test_dunder_import_blocked(self):
        """Test that __import__ is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("__import__('os')")

        assert result.error is not None

    def test_eval_blocked(self):
        """Test that eval() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("eval('1+1')")

        assert result.error is not None

    def test_exec_blocked(self):
        """Test that exec() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("exec('x=1')")

        assert result.error is not None

    def test_open_blocked(self):
        """Test that open() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("open('/etc/passwd')")

        assert result.error is not None

    def test_getattr_blocked(self):
        """Test that getattr() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("getattr(object, '__class__')")

        assert result.error is not None

    def test_dunder_class_blocked(self):
        """Test that __class__ access is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("''.__class__")

        assert result.error is not None

    def test_dunder_subclasses_blocked(self):
        """Test that __subclasses__ is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("object.__subclasses__()")

        assert result.error is not None

    def test_globals_blocked(self):
        """Test that globals() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("globals()")

        assert result.error is not None

    def test_builtins_is_safe_dict(self):
        """Test that __builtins__ is a safe dict with restricted functions."""
        repl = REPLEnvironment(context="test")

        # __builtins__ is accessible but is a restricted dict (not full module)
        result = repl.execute("print(type(__builtins__))")
        assert result.error is None
        assert "dict" in result.output

        # Dangerous functions should NOT be in __builtins__
        result = repl.execute("print('exec' in __builtins__)")
        assert result.error is None
        assert "False" in result.output

        result = repl.execute("print('eval' in __builtins__)")
        assert result.error is None
        assert "False" in result.output

        result = repl.execute("print('open' in __builtins__)")
        assert result.error is None
        assert "False" in result.output


class TestOutputCapping:
    """Test output capping functionality."""

    def test_output_capped(self, tmp_path):
        """Test that large output is spilled to file with summary."""
        config = REPLConfig(output_cap=100, spill_dir=str(tmp_path))
        repl = REPLEnvironment(context="test", config=config)
        result = repl.execute("print('x' * 200)")

        # Output should be a spill summary, not raw output
        assert "chars" in result.output
        assert "peek(" in result.output
        spill = tmp_path / repl._session_id / "turn_1.txt"
        assert spill.is_file()
        assert spill.parent.resolve().is_relative_to(tmp_path.resolve())
        assert spill.read_text().strip() == "x" * 200

    def test_normal_output_not_capped(self):
        """Test that normal output is not capped."""
        config = REPLConfig(output_cap=1000)
        repl = REPLEnvironment(context="test", config=config)
        result = repl.execute("print('hello')")

        assert result.output.strip() == "hello"
        assert "truncated" not in result.output


class TestREPLState:
    """Test REPL state management."""

    def test_get_state(self):
        """Test get_state() returns summary."""
        repl = REPLEnvironment(context="Hello World")
        repl.execute("artifacts['key'] = 'value'")
        state = repl.get_state()

        assert "context" in state
        assert "11 chars" in state  # len("Hello World")
        assert "key" in state

    def test_reset_clears_artifacts(self):
        """Test that reset() clears artifacts."""
        repl = REPLEnvironment(context="test")
        repl.execute("artifacts['key'] = 'value'")
        repl.reset()

        assert len(repl.artifacts) == 0

    def test_reset_keeps_context(self):
        """Test that reset() keeps context."""
        repl = REPLEnvironment(context="important context")
        repl.reset()

        assert repl.context == "important context"

    def test_state_persists_across_executions(self):
        """Test that state persists across execute() calls."""
        repl = REPLEnvironment(context="test")
        repl.execute("x = 42")
        result = repl.execute("print(x)")

        assert "42" in result.output


class TestAllowedBuiltins:
    """Test that allowed builtins work correctly."""

    def test_len_works(self):
        """Test that len() works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(len([1,2,3]))")
        assert "3" in result.output

    def test_range_works(self):
        """Test that range() works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(list(range(3)))")
        assert "[0, 1, 2]" in result.output

    def test_sorted_works(self):
        """Test that sorted() works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(sorted([3,1,2]))")
        assert "[1, 2, 3]" in result.output

    def test_dict_comprehension_works(self):
        """Test that dict comprehension works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print({i: i*2 for i in range(3)})")
        assert "0: 0" in result.output

    def test_list_comprehension_works(self):
        """Test that list comprehension works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print([x*2 for x in range(3)])")
        assert "[0, 2, 4]" in result.output

    def test_try_except_works(self):
        """Test that try/except works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
try:
    x = 1/0
except ZeroDivisionError:
    print("caught")
""")
        assert "caught" in result.output
        assert result.error is None


class TestElapsedTime:
    """Test elapsed time tracking."""

    def test_elapsed_time_recorded(self):
        """Test that elapsed time is recorded."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("x = sum(range(1000))")

        assert result.elapsed_seconds >= 0


class TestConfig:
    """Test REPLConfig options."""

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = REPLConfig(timeout_seconds=60)
        repl = REPLEnvironment(context="test", config=config)

        assert repl.config.timeout_seconds == 60

    def test_custom_output_cap(self):
        """Test custom output cap configuration."""
        config = REPLConfig(output_cap=500)
        repl = REPLEnvironment(context="test", config=config)

        assert repl.config.output_cap == 500

    def test_custom_max_grep_results(self):
        """Test custom max grep results configuration."""
        config = REPLConfig(max_grep_results=10)
        repl = REPLEnvironment(context="test", config=config)

        assert repl.config.max_grep_results == 10


# ============================================================
# NEW TEST CLASSES FOR UNTESTED SUBSYSTEMS
# ============================================================


class TestREPLTypes:
    """Test exception classes and type utilities."""

    def test_repl_error_instantiation(self):
        """Test REPLError can be instantiated and raised."""
        from src.repl_environment import REPLError

        error = REPLError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_repl_timeout_inheritance(self):
        """Test REPLTimeout inherits from REPLError."""
        from src.repl_environment import REPLTimeout, REPLError

        timeout = REPLTimeout("timed out")
        assert str(timeout) == "timed out"
        assert isinstance(timeout, REPLError)
        assert isinstance(timeout, Exception)

    def test_repl_security_error_inheritance(self):
        """Test REPLSecurityError inherits from REPLError."""
        from src.repl_environment import REPLError

        error = REPLSecurityError("dangerous code")
        assert str(error) == "dangerous code"
        assert isinstance(error, REPLError)
        assert isinstance(error, Exception)

    def test_final_signal_exception(self):
        """Test FinalSignal exception with answer attribute."""
        from src.repl_environment import FinalSignal

        signal = FinalSignal("the answer is 42")
        assert signal.answer == "the answer is 42"
        assert str(signal) == "the answer is 42"
        assert isinstance(signal, Exception)

    def test_final_signal_preserves_type(self):
        """Test FinalSignal converts answer to string."""
        from src.repl_environment import FinalSignal

        signal = FinalSignal("test")
        assert isinstance(signal.answer, str)

    def test_wrap_tool_output_function(self):
        """Test wrap_tool_output adds delimiters."""
        from src.repl_environment import wrap_tool_output, TOOL_OUTPUT_START, TOOL_OUTPUT_END

        output = wrap_tool_output("some tool output")
        assert output.startswith(TOOL_OUTPUT_START)
        assert output.endswith(TOOL_OUTPUT_END)
        assert "some tool output" in output

    def test_wrap_tool_output_constants(self):
        """Test tool output delimiter constants exist."""
        from src.repl_environment import TOOL_OUTPUT_START, TOOL_OUTPUT_END

        assert isinstance(TOOL_OUTPUT_START, str)
        assert isinstance(TOOL_OUTPUT_END, str)
        assert len(TOOL_OUTPUT_START) > 0
        assert len(TOOL_OUTPUT_END) > 0


class TestASTSecurityVisitor:
    """Test the AST security visitor."""

    def test_visitor_initialization(self):
        """Test ASTSecurityVisitor initializes with empty violations."""
        from src.repl_environment import ASTSecurityVisitor

        visitor = ASTSecurityVisitor()
        assert visitor.violations == []

    def test_visit_import_forbidden(self):
        """Test visitor detects forbidden module imports."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "import os"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "os" in str(visitor.violations)

    def test_visit_import_subprocess(self):
        """Test visitor detects subprocess import."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "import subprocess"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "subprocess" in str(visitor.violations)

    def test_visit_from_import_forbidden(self):
        """Test visitor detects from...import of forbidden modules."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "from os import path"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "os" in str(visitor.violations)

    def test_visit_call_eval(self):
        """Test visitor detects eval() calls."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "eval('1+1')"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "eval" in str(visitor.violations)

    def test_visit_call_exec(self):
        """Test visitor detects exec() calls."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "exec('x=1')"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "exec" in str(visitor.violations)

    def test_visit_call_dunder_import(self):
        """Test visitor detects __import__() calls."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "__import__('os')"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "__import__" in str(visitor.violations)

    def test_visit_attribute_dunder_class(self):
        """Test visitor detects __class__ attribute access."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "x.__class__"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "__class__" in str(visitor.violations)

    def test_visit_subscript_dunder_string(self):
        """Test visitor detects string-based dunder subscript access."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = "obj['__class__']"
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) > 0
        assert "__class__" in str(visitor.violations)

    def test_safe_code_no_violations(self):
        """Test safe code produces no violations."""
        import ast
        from src.repl_environment import ASTSecurityVisitor

        code = """
x = [1, 2, 3]
y = sum(x)
print(y)
"""
        tree = ast.parse(code)
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        assert visitor.violations == []


class TestContextMixin:
    """Test context management methods."""

    def test_context_len(self):
        """Test _context_len returns character count."""
        repl = REPLEnvironment(context="Hello World")
        result = repl.execute("print(context_len())")

        assert "11" in result.output

    def test_context_len_empty(self):
        """Test _context_len with empty context."""
        repl = REPLEnvironment(context="")
        result = repl.execute("print(context_len())")

        assert "0" in result.output

    def test_chunk_context_basic(self):
        """Test _chunk_context splits context into chunks."""
        context = "A" * 1000
        repl = REPLEnvironment(context=context)
        result = repl.execute("chunks = chunk_context(4); print(len(chunks))")

        assert "4" in result.output
        assert result.error is None

    def test_chunk_context_returns_metadata(self):
        """Test chunk_context returns chunks with metadata."""
        context = "Hello World"
        repl = REPLEnvironment(context=context)
        result = repl.execute("""
chunks = chunk_context(2)
print(chunks[0]['index'])
print('text' in chunks[0])
print('char_count' in chunks[0])
""")

        assert "0" in result.output
        assert "True" in result.output

    def test_chunk_context_tracks_exploration(self):
        """Test chunk_context increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("chunk_context(2)")

        assert repl._exploration_calls > initial_calls


class TestRoutingMixin:
    """Test routing and delegation methods."""

    def test_my_role_returns_role_info(self):
        """Test my_role() returns current role information."""
        repl = REPLEnvironment(context="test")
        # Set role explicitly
        repl.role = "worker_explore"
        result = repl.execute("output = my_role(); print('TOOL_OUTPUT' in output)")

        # my_role() should return output wrapped in tool delimiters
        assert "True" in result.output or result.error is None

    def test_escalate_sets_artifacts(self):
        """Test escalate() sets escalation artifacts."""
        repl = REPLEnvironment(context="test")
        repl.execute("escalate('need help', 'architect_general')")

        assert repl.artifacts.get("_escalation_requested") is True
        assert "need help" in repl.artifacts.get("_escalation_reason", "")

    def test_escalate_returns_message(self):
        """Test escalate() returns acknowledgment message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(escalate('complex task'))")

        assert "ESCALATION REQUESTED" in result.output
        assert "complex task" in result.output


class TestExplorationLog:
    """Test exploration event logging."""

    def test_exploration_event_creation(self):
        """Test ExplorationEvent dataclass creation."""
        from src.repl_environment.types import ExplorationEvent

        event = ExplorationEvent(
            function="peek",
            args={"n": 100},
            result_size=500,
            timestamp=123.456,
            token_estimate=125,
        )

        assert event.function == "peek"
        assert event.args == {"n": 100}
        assert event.result_size == 500
        assert event.token_estimate == 125

    def test_exploration_log_initialization(self):
        """Test ExplorationLog initializes with empty events."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        assert log.events == []
        assert log.total_exploration_tokens == 0

    def test_exploration_log_add_event_string(self):
        """Test add_event with string result."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("peek", {"n": 100}, "A" * 100)

        assert len(log.events) == 1
        assert log.events[0].function == "peek"
        assert log.events[0].result_size == 100
        assert log.total_exploration_tokens == 25  # 100 / 4

    def test_exploration_log_add_event_list(self):
        """Test add_event with list result."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("grep", {"pattern": "test"}, ["line1", "line2", "line3"])

        assert len(log.events) == 1
        assert log.events[0].result_size == 3
        # Token estimate for lists is 0 (based on list length, not character count)
        # Only strings get token estimates
        assert log.events[0].token_estimate == 0

    def test_exploration_log_strategy_summary(self):
        """Test get_strategy_summary returns function counts."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("peek", {}, "A" * 100)
        log.add_event("grep", {}, "B" * 100)
        log.add_event("peek", {}, "C" * 100)

        summary = log.get_strategy_summary()
        assert summary["total_events"] == 3
        assert summary["function_counts"]["peek"] == 2
        assert summary["function_counts"]["grep"] == 1

    def test_exploration_log_token_efficiency(self):
        """Test get_token_efficiency calculates ratio."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("peek", {}, "A" * 400)  # 100 tokens

        efficiency = log.get_token_efficiency(result_tokens=200)
        assert efficiency["exploration_tokens"] == 100
        assert efficiency["result_tokens"] == 200
        assert efficiency["efficiency_ratio"] == 2.0

    def test_exploration_log_classify_strategy_delegated(self):
        """Test strategy classification identifies delegated."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("llm_call", {}, "response")

        summary = log.get_strategy_summary()
        assert summary["strategy_type"] == "delegated"

    def test_exploration_log_classify_strategy_search(self):
        """Test strategy classification identifies search."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("grep", {}, "result1")
        log.add_event("grep", {}, "result2")
        log.add_event("peek", {}, "preview")

        summary = log.get_strategy_summary()
        assert summary["strategy_type"] == "search"

    def test_exploration_log_classify_strategy_scan(self):
        """Test strategy classification identifies scan."""
        from src.repl_environment.types import ExplorationLog

        log = ExplorationLog()
        log.add_event("peek", {}, "preview1")
        log.add_event("peek", {}, "preview2")

        summary = log.get_strategy_summary()
        assert summary["strategy_type"] == "scan"


# ============================================================
# CHARACTERIZATION TESTS — REPLEnvironment public API contract
# ============================================================


class TestConstructorDefaults:
    """Characterize constructor defaults and attribute initialization."""

    def test_constructor_sets_context(self):
        """Constructor stores context string verbatim."""
        repl = REPLEnvironment(context="test context")
        assert repl.context == "test context"

    def test_constructor_default_config(self):
        """Constructor creates default REPLConfig when none provided."""
        repl = REPLEnvironment(context="test context")
        assert isinstance(repl.config, REPLConfig)
        # Should have the default output_cap
        assert repl.config.output_cap == 8192

    def test_constructor_default_role_is_worker_general(self):
        """Constructor defaults role to 'worker_general'."""
        repl = REPLEnvironment(context="test context")
        assert repl.role == "worker_general"

    def test_constructor_custom_role(self):
        """Constructor accepts custom role string."""
        repl = REPLEnvironment(context="test context", role="coder_escalation")
        assert repl.role == "coder_escalation"

    def test_constructor_default_artifacts_empty_dict(self):
        """Constructor defaults artifacts to empty dict when None."""
        repl = REPLEnvironment(context="test context")
        assert repl.artifacts == {}
        assert isinstance(repl.artifacts, dict)

    def test_constructor_initial_execution_count_zero(self):
        """Constructor initializes execution count to zero."""
        repl = REPLEnvironment(context="test context")
        assert repl._execution_count == 0

    def test_constructor_initial_final_answer_none(self):
        """Constructor initializes final answer to None."""
        repl = REPLEnvironment(context="test context")
        assert repl._final_answer is None


class TestSafeImportModules:
    """Characterize the SAFE_IMPORT_MODULES whitelist."""

    def test_contains_math(self):
        """SAFE_IMPORT_MODULES includes 'math'."""
        assert "math" in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_contains_json(self):
        """SAFE_IMPORT_MODULES includes 'json'."""
        assert "json" in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_contains_re(self):
        """SAFE_IMPORT_MODULES includes 're'."""
        assert "re" in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_contains_collections(self):
        """SAFE_IMPORT_MODULES includes 'collections'."""
        assert "collections" in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_contains_itertools(self):
        """SAFE_IMPORT_MODULES includes 'itertools'."""
        assert "itertools" in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_contains_datetime(self):
        """SAFE_IMPORT_MODULES includes 'datetime'."""
        assert "datetime" in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_does_not_contain_os(self):
        """SAFE_IMPORT_MODULES must NOT include 'os'."""
        assert "os" not in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_does_not_contain_subprocess(self):
        """SAFE_IMPORT_MODULES must NOT include 'subprocess'."""
        assert "subprocess" not in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_does_not_contain_shutil(self):
        """SAFE_IMPORT_MODULES must NOT include 'shutil'."""
        assert "shutil" not in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_does_not_contain_socket(self):
        """SAFE_IMPORT_MODULES must NOT include 'socket'."""
        assert "socket" not in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_does_not_contain_sys(self):
        """SAFE_IMPORT_MODULES must NOT include 'sys'."""
        assert "sys" not in REPLEnvironment.SAFE_IMPORT_MODULES

    def test_is_frozenset(self):
        """SAFE_IMPORT_MODULES is a frozenset (immutable)."""
        assert isinstance(REPLEnvironment.SAFE_IMPORT_MODULES, frozenset)


class TestBuildGlobals:
    """Characterize _build_globals() namespace construction."""

    def test_globals_includes_context(self):
        """_build_globals puts context into the namespace."""
        repl = REPLEnvironment(context="my context string")
        g = repl._build_globals()
        assert g["context"] == "my context string"

    def test_globals_includes_artifacts(self):
        """_build_globals puts artifacts dict into the namespace."""
        artifacts = {"key": "value"}
        repl = REPLEnvironment(context="test", artifacts=artifacts)
        g = repl._build_globals()
        assert g["artifacts"] is artifacts

    def test_globals_includes_safe_builtins_len(self):
        """_build_globals includes len in builtins."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        builtins = g["__builtins__"]
        assert "len" in builtins
        assert builtins["len"] is __builtins__["len"] if isinstance(__builtins__, dict) else builtins["len"] is len

    def test_globals_includes_safe_builtins_range(self):
        """_build_globals includes range in builtins."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "range" in g["__builtins__"]

    def test_globals_includes_safe_builtins_sorted(self):
        """_build_globals includes sorted in builtins."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "sorted" in g["__builtins__"]

    def test_globals_includes_safe_builtins_print(self):
        """_build_globals includes print in builtins."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "print" in g["__builtins__"]

    def test_globals_includes_FINAL_function(self):
        """_build_globals includes FINAL callable."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "FINAL" in g
        assert callable(g["FINAL"])

    def test_globals_includes_peek_function(self):
        """_build_globals includes peek callable."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "peek" in g
        assert callable(g["peek"])

    def test_globals_includes_grep_function(self):
        """_build_globals includes grep callable."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "grep" in g
        assert callable(g["grep"])

    def test_globals_includes_FINAL_VAR_function(self):
        """_build_globals includes FINAL_VAR callable."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        assert "FINAL_VAR" in g
        assert callable(g["FINAL_VAR"])

    def test_globals_includes_preloaded_math(self):
        """_build_globals pre-loads the math module."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        import math
        assert g["math"] is math

    def test_globals_includes_preloaded_json(self):
        """_build_globals pre-loads the json module."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        import json
        assert g["json"] is json

    def test_globals_excludes_dangerous_builtins(self):
        """_build_globals does NOT include exec/eval/open in builtins."""
        repl = REPLEnvironment(context="test")
        g = repl._build_globals()
        builtins = g["__builtins__"]
        assert "exec" not in builtins
        assert "eval" not in builtins
        assert "open" not in builtins


class TestExecuteCharacterization:
    """Characterize execute() behavior for the public API contract."""

    def test_execute_simple_math(self):
        """execute() runs simple arithmetic: x = 1 + 1."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("x = 1 + 1\nprint(x)")
        assert "2" in result.output
        assert result.error is None
        assert not result.is_final

    def test_execute_final_sets_answer(self):
        """execute() with FINAL('answer') sets is_final and final_answer."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("FINAL('answer')")
        assert result.is_final is True
        assert result.final_answer == "answer"

    def test_execute_blocks_import_os(self):
        """execute() blocks 'import os' with a security error."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("import os")
        assert result.error is not None
        # The error comes from AST validation (Dangerous operation) or ImportError
        assert "not allowed" in result.error.lower() or "Dangerous" in result.error

    def test_execute_blocks_import_subprocess(self):
        """execute() blocks 'import subprocess' with a security error."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("import subprocess")
        assert result.error is not None

    def test_execute_allows_import_math(self):
        """execute() allows 'import math' and math functions work."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("import math\nprint(math.sqrt(16))")
        assert result.error is None
        assert "4.0" in result.output

    def test_execute_allows_import_json(self):
        """execute() allows 'import json' and json functions work."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute('import json\nprint(json.dumps({"a": 1}))')
        assert result.error is None
        assert '"a"' in result.output

    def test_execute_allows_import_re(self):
        """execute() allows 'import re' and regex functions work."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("import re\nprint(re.findall(r'\\d+', 'abc123def456'))")
        assert result.error is None
        assert "123" in result.output

    def test_execute_allows_import_collections(self):
        """execute() allows 'import collections'."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("import collections\nc = collections.Counter('aabbc')\nprint(c['a'])")
        assert result.error is None
        assert "2" in result.output

    def test_execute_increments_execution_count(self):
        """execute() increments the internal execution counter."""
        repl = REPLEnvironment(context="test context")
        assert repl._execution_count == 0
        repl.execute("x = 1")
        assert repl._execution_count == 1
        repl.execute("y = 2")
        assert repl._execution_count == 2

    def test_execute_returns_execution_result(self):
        """execute() returns an ExecutionResult dataclass."""
        from src.repl_environment.types import ExecutionResult
        repl = REPLEnvironment(context="test context")
        result = repl.execute("x = 1")
        assert isinstance(result, ExecutionResult)

    def test_execute_elapsed_seconds_positive(self):
        """execute() records a positive elapsed_seconds."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute("x = sum(range(1000))")
        assert result.elapsed_seconds >= 0


class TestExecuteTimeout:
    """Characterize timeout behavior of execute()."""

    def test_timeout_is_configurable(self):
        """REPLConfig.timeout_seconds controls execution timeout."""
        config = REPLConfig(timeout_seconds=30)
        repl = REPLEnvironment(context="test", config=config)
        assert repl.config.timeout_seconds == 30

    def test_short_code_completes_within_timeout(self):
        """Fast code completes without timeout error."""
        config = REPLConfig(timeout_seconds=5)
        repl = REPLEnvironment(context="test", config=config)
        result = repl.execute("x = 1 + 1")
        assert result.error is None

    def test_alarm_timeout_marks_terminal_and_refuses_later_execute(self, monkeypatch):
        import signal

        handlers = {}

        def fake_signal(signum, handler):
            old_handler = handlers.get(signum)
            handlers[signum] = handler
            return old_handler

        def fake_alarm(seconds):
            if seconds:
                handlers[signal.SIGALRM](signal.SIGALRM, None)
            return 0

        monkeypatch.setattr("src.repl_environment.environment.signal.signal", fake_signal)
        monkeypatch.setattr("src.repl_environment.environment.signal.alarm", fake_alarm)
        repl = REPLEnvironment(context="test", config=REPLConfig(timeout_seconds=1))

        result = repl.execute("x = 1")

        assert result.error == "REPLTimeout: Execution timed out after 1s"
        assert repl.timed_out is True
        assert repl.execute("print('late')").error == result.error
        repl.reset()
        assert repl.execute("print('after reset')").error == result.error

    def test_structured_alarm_timeout_is_sticky(self, monkeypatch):
        import signal

        handlers = {}

        def fake_signal(signum, handler):
            old_handler = handlers.get(signum)
            handlers[signum] = handler
            return old_handler

        def fake_alarm(seconds):
            if seconds:
                handlers[signal.SIGALRM](signal.SIGALRM, None)
            return 0

        monkeypatch.setattr("src.repl_environment.environment.signal.signal", fake_signal)
        monkeypatch.setattr("src.repl_environment.environment.signal.alarm", fake_alarm)
        repl = REPLEnvironment(
            context="test", config=REPLConfig(timeout_seconds=1, structured_mode=True)
        )

        result = repl.execute("print('hello')")

        assert result.error == "REPLTimeout: Execution timed out after 1s"
        assert repl.timed_out is True
        assert repl.execute("FINAL('late')").error == result.error

    def test_restricted_timeout_result_marks_terminal_without_executor_import(self):
        from types import SimpleNamespace

        from src.repl_environment.environment import _RestrictedREPLEnvironment
        from src.repl_environment.types import ExecutionResult

        repl = object.__new__(_RestrictedREPLEnvironment)
        repl.config = REPLConfig(timeout_seconds=3)
        repl._timeout_terminal = False
        repl._restricted_executor = SimpleNamespace(
            execute=lambda code: ExecutionResult(
                output="", is_final=False, error="Execution timed out after 3s"
            )
        )

        result = repl.execute("ignored")

        assert result.error == "REPLTimeout: Execution timed out after 3s"
        assert repl.timed_out is True
        assert repl.execute("ignored again").error == result.error


class TestMixinContractAssertions:
    """Characterize that mixin contract assertions pass on construction."""

    def test_mixin_contracts_pass_minimal_args(self):
        """REPLEnvironment(context='test') satisfies all mixin contracts."""
        # If this raises AssertionError, mixin contracts are broken
        repl = REPLEnvironment(context="test context")
        # Verify key attributes exist (same list as __init__ required_attrs)
        assert hasattr(repl, "config")
        assert hasattr(repl, "context")
        assert hasattr(repl, "artifacts")
        assert hasattr(repl, "role")
        assert hasattr(repl, "task_id")
        assert hasattr(repl, "_exploration_calls")
        assert hasattr(repl, "_execution_count")
        assert hasattr(repl, "_tool_invocations")
        assert hasattr(repl, "_exploration_log")
        assert hasattr(repl, "_grep_hits_buffer")
        assert hasattr(repl, "_findings_buffer")
        assert hasattr(repl, "_research_context")
        assert hasattr(repl, "_last_research_node")
        assert hasattr(repl, "_final_answer")
        assert hasattr(repl, "_globals")
        assert hasattr(repl, "llm_primitives")
        assert hasattr(repl, "tool_registry")
        assert hasattr(repl, "script_registry")
        assert hasattr(repl, "progress_logger")
        assert hasattr(repl, "_retriever")
        assert hasattr(repl, "_hybrid_router")
        assert hasattr(repl, "_validate_file_path")
        assert hasattr(repl, "_build_globals")

    def test_mixin_contracts_pass_with_all_optional_none(self):
        """Construction with explicit None for all optional args satisfies contracts."""
        repl = REPLEnvironment(
            context="test",
            artifacts=None,
            config=None,
            llm_primitives=None,
            tool_registry=None,
            script_registry=None,
            role=None,
            progress_logger=None,
            task_id=None,
            retriever=None,
            hybrid_router=None,
            structured_mode=False,
        )
        assert repl.context == "test"
        assert repl.artifacts == {}
        assert repl.role == "worker_general"


class TestAllowedFilePaths:
    """Characterize ALLOWED_FILE_PATHS security boundary."""

    def test_allowed_file_paths_is_list(self):
        """ALLOWED_FILE_PATHS is a list of strings."""
        assert isinstance(REPLEnvironment.ALLOWED_FILE_PATHS, list)
        for path in REPLEnvironment.ALLOWED_FILE_PATHS:
            assert isinstance(path, str)

    def test_allowed_file_paths_includes_raid(self):
        """ALLOWED_FILE_PATHS includes configured LLM root prefix."""
        from src.config import get_config

        llm_root = str(get_config().paths.llm_root)
        llm_prefix = llm_root if llm_root.endswith("/") else f"{llm_root}/"
        assert any(p.startswith(llm_prefix) for p in REPLEnvironment.ALLOWED_FILE_PATHS)

    def test_allowed_file_paths_includes_tmp(self):
        """ALLOWED_FILE_PATHS includes /tmp/ prefix."""
        assert any("/tmp/" in p for p in REPLEnvironment.ALLOWED_FILE_PATHS)

    def test_validate_file_path_allowed(self):
        """_validate_file_path accepts paths under allowed prefixes."""
        from src.config import get_config

        repl = REPLEnvironment(context="test")
        llm_root = str(get_config().paths.llm_root)
        probe_path = f"{llm_root.rstrip('/')}/test.txt"
        is_valid, error = repl._validate_file_path(probe_path)
        assert is_valid is True
        assert error is None

    def test_validate_file_path_forbidden(self):
        """_validate_file_path rejects paths outside allowed prefixes."""
        repl = REPLEnvironment(context="test")
        is_valid, error = repl._validate_file_path("/etc/passwd")
        assert is_valid is False
        assert error is not None
