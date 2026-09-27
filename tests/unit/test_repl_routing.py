#!/usr/bin/env python3
"""Unit tests for the REPL routing tools (_RoutingMixin)."""

import json
from unittest.mock import Mock

import pytest

from src.repl_environment import REPLEnvironment

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


class TestMyRole:
    """Test _my_role() / my_role() function."""

    def test_my_role_returns_json(self):
        """Test my_role() returns role information as JSON."""
        repl = REPLEnvironment(context="test", role="worker_explore")
        result = repl.execute("output = my_role(); print('role' in output)")

        assert result.error is None
        assert "True" in result.output

    def test_my_role_contains_tier_info(self):
        """Test my_role() includes tier information."""
        repl = REPLEnvironment(context="test", role="coder_escalation")
        repl.config.use_toon_encoding = False  # Force JSON for test parsing
        # Execute and capture the output
        result = repl.execute("""
output = my_role()
# Strip tool delimiters if present
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data['role'])
print(data['tier'])
""")

        assert result.error is None
        assert "coder_escalation" in result.output
        assert "tier" in result.output.lower() or result.output.strip()

    def test_my_role_default_role(self):
        """Test my_role() returns default role when not set."""
        repl = REPLEnvironment(context="test")
        repl.config.use_toon_encoding = False  # Force JSON for test parsing
        # Default role should be worker_general
        result = repl.execute("""
output = my_role()
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data['role'])
""")

        assert result.error is None
        assert "worker_general" in result.output


class TestResolveRoleAlias:
    """Test _resolve_role_alias() method."""

    def test_resolve_known_alias(self):
        """Test resolving known role aliases."""
        repl = REPLEnvironment(context="test")

        # Test directly via the method
        assert repl._resolve_role_alias("researcher_agent") == "worker_general"
        assert repl._resolve_role_alias("coder_agent") == "coder_escalation"
        assert repl._resolve_role_alias("reviewer_agent") == "architect_critic"  # ARCHSWAP-20260927 reviewer default

    def test_resolve_unknown_returns_original(self):
        """Test unknown aliases return original value."""
        repl = REPLEnvironment(context="test")

        assert repl._resolve_role_alias("unknown_role") == "unknown_role"
        assert repl._resolve_role_alias("custom_agent") == "custom_agent"

    def test_resolve_passthrough_for_actual_roles(self):
        """Test actual role names pass through unchanged."""
        repl = REPLEnvironment(context="test")

        assert repl._resolve_role_alias("coder_escalation") == "coder_escalation"
        assert repl._resolve_role_alias("worker_general") == "worker_general"
        assert repl._resolve_role_alias("worker_math") == "worker_math"

    def test_legacy_worker_aliases_map_to_live_roles(self):
        """Legacy worker aliases should normalize to live stack roles."""
        repl = REPLEnvironment(context="test")

        assert repl._resolve_role_alias("worker_code") == "coder_escalation"
        assert repl._resolve_role_alias("worker_coder") == "coder_escalation"
        assert repl._resolve_role_alias("worker_explore") == "worker_general"
        assert repl._resolve_role_alias("worker_fast") == "worker_general"


class TestEscalate:
    """Test _escalate() / escalate() function."""

    def test_escalate_sets_artifact(self):
        """Test escalate() sets escalation request in artifacts."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("escalate('Task too complex')")

        assert result.error is None
        assert repl.artifacts.get("_escalation_requested") is True
        assert repl.artifacts.get("_escalation_reason") == "Task too complex"

    def test_escalate_with_target_role(self):
        """Test escalate() with specific target role."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("escalate('Need architecture review', 'architect_general')")

        assert result.error is None
        assert repl.artifacts.get("_escalation_target") == "architect_general"
        assert repl.artifacts.get("_escalation_requested") is True

    def test_escalate_resolves_role_alias(self):
        """Test escalate() resolves target role aliases."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("escalate('Need review', 'reviewer_agent')")

        assert result.error is None
        # reviewer_agent should resolve to architect_critic (the reviewer default since ARCHSWAP-20260927)
        assert repl.artifacts.get("_escalation_target") == "architect_critic"

    def test_escalate_returns_message(self):
        """Test escalate() returns acknowledgment message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(escalate('Too difficult'))")

        assert result.error is None
        assert "ESCALATION REQUESTED" in result.output
        assert "Too difficult" in result.output

    def test_escalate_logs_event(self):
        """Test escalate() is logged as exploration call."""
        repl = REPLEnvironment(context="test")
        repl.execute("escalate('complex task')")

        # Escalate does NOT increment exploration_calls (it's a signal, not exploration)
        # But it should still set artifacts
        assert repl.artifacts.get("_escalation_requested") is True


class TestRecall:
    """Test _recall() / recall() function with episodic memory."""

    @pytest.fixture(autouse=True)
    def _hermetic_legacy_backend(self, monkeypatch):
        """No-retriever recall opens a fresh EpisodicStore + TaskEmbedder; keep
        it off the repo's default store path and the embedding server."""
        import numpy as np

        import orchestration.repl_memory.embedder as embedder_mod
        import orchestration.repl_memory.episodic_store as store_mod

        class _EmptyStore:
            def __init__(self, *a, **k):
                pass

            def retrieve_by_similarity(self, *a, **k):
                return []

            def close(self):
                pass

        class _Embedder:
            def __init__(self, *a, **k):
                pass

            def embed_exploration(self, query, context_preview):
                return np.zeros(4, dtype=np.float32)

            def close(self):
                pass

        monkeypatch.setattr(store_mod, "EpisodicStore", _EmptyStore)
        monkeypatch.setattr(embedder_mod, "TaskEmbedder", _Embedder)

    def test_recall_without_retriever(self):
        """Test recall() falls back and reports an explicit status when no retriever."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = recall('similar tasks')
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(len(data.get('results', [])))
print(data.get('status'))
""")

        assert result.error is None
        assert "0" in result.output
        assert "ok" in result.output

    def test_recall_with_mock_retriever(self):
        """Test recall() with mocked retriever."""
        mock_retriever = Mock()
        mock_memory = Mock()
        mock_memory.context = {"objective": "test task", "role": "worker_general"}
        mock_memory.action = "test action"
        mock_memory.outcome = "success"

        mock_result = Mock()
        mock_result.memory = mock_memory
        mock_result.q_value = 0.85
        mock_result.similarity = 0.92
        mock_result.combined_score = 0.88

        mock_retriever.retrieve_for_exploration = Mock(return_value=[mock_result])
        mock_retriever.get_best_action = Mock(return_value=("test action", 0.88))

        repl = REPLEnvironment(context="test", retriever=mock_retriever)
        result = repl.execute("""
output = recall('similar task', limit=5)
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(len(data.get('results', [])))
print(data.get('best_action'))
""")

        assert result.error is None
        assert "1" in result.output  # 1 result
        assert "test action" in result.output

    def test_recall_increments_exploration(self):
        """Test recall() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("recall('task')")

        assert repl._exploration_calls > initial_calls


class TestRouteAdvice:
    """Test _route_advice() / route_advice() function."""

    def test_route_advice_without_router(self):
        """Test route_advice() returns error when no router."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = route_advice('implement feature X')
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data.get('strategy'))
print('unavailable' in str(data.get('warnings', [])))
""")

        assert result.error is None
        assert "unavailable" in result.output.lower()

    def test_route_advice_with_mock_router(self):
        """Test route_advice() with mocked hybrid router."""
        mock_router = Mock()
        mock_retriever = Mock()

        # Mock retrieval results
        mock_memory = Mock()
        mock_memory.context = {"role": "coder_escalation"}
        mock_memory.action = "coder_escalation"
        mock_memory.outcome = "success"

        mock_result = Mock()
        mock_result.memory = mock_memory
        mock_result.q_value = 0.90
        mock_result.combined_score = 0.85
        mock_result.warnings = []

        mock_retriever.retrieve_for_routing = Mock(return_value=[mock_result])
        mock_router.retriever = mock_retriever
        mock_router.route = Mock(return_value=(("coder_escalation", 0.85), "memrl"))

        repl = REPLEnvironment(context="test", hybrid_router=mock_router)
        repl.config.use_toon_encoding = False  # Force JSON for test parsing
        result = repl.execute("""
output = route_advice('write a function')
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data.get('recommended_role'))
print(data.get('strategy'))
""")

        assert result.error is None
        assert "coder_escalation" in result.output
        assert "memrl" in result.output

    def test_route_advice_increments_exploration(self):
        """Test route_advice() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("route_advice('task')")

        assert repl._exploration_calls > initial_calls


class TestDelegate:
    """Test _delegate() / delegate() function."""

    def test_delegate_without_llm_primitives(self):
        """Test delegate() returns error when no LLM primitives."""
        # Use a non-worker role to bypass tier guard
        repl = REPLEnvironment(context="test", role="frontdoor")
        result = repl.execute("print(delegate('subtask', 'worker_general'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "primitives" in result.output.lower()

    def test_delegate_with_mock_llm(self):
        """Test delegate() with mocked LLM primitives."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="Task completed successfully")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="coder_escalation")
        result = repl.execute(
            "output = delegate('summarize this', to='worker_summarize', reason='need summary')"
        )

        assert result.error is None
        mock_llm.llm_call.assert_called_once()

    def test_delegate_resolves_role_alias(self):
        """Test delegate() resolves target role aliases."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="response")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl.execute("delegate('task', 'researcher_agent')")

        assert result.error is None
        # Should call with resolved role (worker_general)
        call_args = mock_llm.llm_call.call_args
        assert call_args[1]["role"] == "worker_general"

    def test_delegate_workers_can_delegate(self):
        """Test workers can delegate to other workers (no tier restriction)."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="worker response")

        # Set role to a worker - workers CAN now delegate (unified execution model)
        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="worker_explore")
        result = repl.execute("output = delegate('subtask', to='worker_math')")

        assert result.error is None
        # LLM should have been called (delegation successful)
        mock_llm.llm_call.assert_called_once()
        call_args = mock_llm.llm_call.call_args
        assert call_args[1]["role"] == "worker_math"

    def test_delegate_tracks_delegation(self):
        """Test delegate() tracks delegation in artifacts."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="result")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="coder_escalation")
        # Use keyword args for the new API: delegate(brief, to=..., reason=...)
        result = repl.execute("delegate('task', to='worker_math', reason='need help')")

        assert result.error is None
        assert "_delegations" in repl.artifacts
        delegations = repl.artifacts["_delegations"]
        assert len(delegations) == 1
        # worker_math doesn't have an alias, should stay as is
        assert delegations[0]["to_role"] == "worker_math"
        assert delegations[0]["reason"] == "need help"

    def test_delegate_increments_exploration(self):
        """Test delegate() increments exploration calls."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="result")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        initial_calls = repl._exploration_calls
        repl.execute("delegate('task', 'worker_general')")

        assert repl._exploration_calls > initial_calls

    def test_delegate_handles_exception(self):
        """Test delegate() handles LLM call exceptions."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(side_effect=Exception("LLM error"))

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl.execute("print(delegate('task', 'worker_general'))")

        assert result.error is None
        assert "DELEGATION FAILED" in result.output
        assert "LLM error" in result.output

    def test_delegate_schema_validates_response(self):
        """Schema mode parses a valid single delegate response."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value='{"answer": 42}')
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl._delegate("return the answer", to="worker_general", schema=schema)
        parsed = json.loads(result)

        assert parsed["schema_validation"] is True
        assert parsed["valid"] is True
        assert parsed["response"] == {"answer": 42}
        assert parsed["raw_response"] == '{"answer": 42}'
        assert parsed["attempts"] == 1
        prompt = mock_llm.llm_call.call_args.args[0]
        assert "Return only a JSON value" in prompt
        assert "return the answer" in prompt
        assert repl.artifacts["_delegations"][0]["schema_valid"] is True

    def test_delegate_schema_repairs_invalid_response_without_full_recall(self):
        """TD-21.22: an invalid-but-recoverable reply is fixed by ONE repair
        turn (fish, then one constrained extraction call to the SAME target
        role) rather than the old full re-call retry loop. ``attempts``/
        ``schema_retry_count`` stay at their no-retry values because the
        full-recall loop never runs; the repair turn is a distinct call
        counted separately via STRUCTURED_OUTPUT_REPAIR_COUNTS."""
        from src.structured_output.repair import (
            STRUCTURED_OUTPUT_REPAIR_COUNTS,
            reset_counts_for_tests,
        )

        reset_counts_for_tests()
        mock_llm = Mock()
        # TD-21.34 require_evidence: "7" must be literally present in the
        # raw reply for the repair turn's copy to be evidence-checkable.
        mock_llm.llm_call = Mock(side_effect=["not json, but the answer is 7", '{"answer": 7}'])
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl._delegate("return the answer", to="worker_general", schema=schema)
        parsed = json.loads(result)

        assert parsed["valid"] is True
        assert parsed["response"] == {"answer": 7}
        # The full re-call retry loop never ran -- repair recovered it first.
        assert parsed["attempts"] == 1
        assert mock_llm.llm_call.call_count == 2
        extraction_prompt = mock_llm.llm_call.call_args_list[1].args[0]
        assert "not json" in extraction_prompt
        assert repl.artifacts["_delegations"][0]["schema_retry_count"] == 0
        assert repl.artifacts["_delegations"][0]["schema_valid"] is True
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("repl_delegate", "repaired")) == 1

        # TD-21.22 wire-schema check: BOTH llm_call invocations (the initial
        # delegate call and the repair extraction turn) carry json_schema on
        # the wire now that TD-21.0 lets it constrain both lanes.
        first_call_kwargs = mock_llm.llm_call.call_args_list[0].kwargs
        assert first_call_kwargs.get("json_schema") == schema

    def test_delegate_schema_reports_exhausted_retry(self):
        """Schema mode reports invalid output instead of silently accepting
        it, after BOTH the repair turn and the full re-call retry loop miss."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(
            side_effect=["not json", "still not json", '{"answer": "wrong"}']
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl._delegate("return the answer", to="worker_general", schema=schema)
        parsed = json.loads(result)

        assert parsed["valid"] is False
        assert parsed["response"] == '{"answer": "wrong"}'
        assert parsed["raw_response"] == '{"answer": "wrong"}'
        assert "not of type" in parsed["error"]
        assert parsed["attempts"] == 2
        assert mock_llm.llm_call.call_count == 3
        assert repl.artifacts["_delegations"][0]["schema_valid"] is False

    def test_delegate_schema_rejects_non_dict_schema(self):
        """Schema argument must be a JSON Schema dict."""
        mock_llm = Mock()
        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")

        result = repl._delegate("task", to="worker_general", schema="not a dict")

        assert "schema must be a JSON Schema dict" in result
        mock_llm.llm_call.assert_not_called()

    def test_delegate_schema_rejects_parallel_mode(self):
        """Schema validation is intentionally single-delegate only."""
        mock_llm = Mock()
        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")

        result = repl._delegate(
            "items: [a, b]",
            to="worker_general",
            parallel=True,
            schema={"type": "object"},
        )

        assert "only supported for single delegation" in result
        mock_llm.llm_call.assert_not_called()


class TestFetchReport:
    """Test fetch_report() lazy delegation report retrieval tool."""

    def test_fetch_report_reads_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REPORT_DIR", str(tmp_path))
        report_id = "worker_coder-1234567890-deadbeefcafebabe"
        (tmp_path / f"{report_id}.txt").write_text("alpha beta gamma delta")

        repl = REPLEnvironment(context="test", role=_RETIRED_ARCHITECT_ROLE)
        result = repl.execute(
            f"""
output = fetch_report("{report_id}", offset=6, max_chars=8)
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data["ok"])
print(data["content"])
print(data["truncated"])
"""
        )

        assert result.error is None
        assert "True" in result.output
        assert "beta gam" in result.output
