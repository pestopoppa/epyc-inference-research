"""Combined REPL operations to reduce turn count.

Mixin providing batched multi-tool patterns based on autopilot log analysis:
- batch_web_search: N sequential web searches in 1 turn
- search_and_verify: web_search + wikipedia verification in 1 turn
- peek_grep: file read + regex search in 1 turn
- batch_llm_query: N parallel sub-LM calls with structured output in 1 turn

Feature-gated by REPL_COMBINED_OPS env var (default off).
"""

from __future__ import annotations

import json
import logging
import os
import re

from src.graph.helpers import _validate_final_answer
from src.structured_output.repair import parse_with_repair

logger = logging.getLogger(__name__)

_FEATURE_FLAG_VALUES = {"1", "true", "on"}
_MAX_QUERIES = 5
_MAX_SCHEMA_RETRIES = 2


def _feature_enabled() -> bool:
    """Check whether combined ops are enabled via env var."""
    return os.environ.get("REPL_COMBINED_OPS", "").strip().lower() in _FEATURE_FLAG_VALUES


def _render_child_schema_preamble(schema: dict) -> str:
    """Prompt prefix for schema-constrained child LLM calls."""
    return (
        "Return only a JSON value matching this JSON Schema. Do not wrap it in markdown "
        "or add prose.\n```json\n"
        + json.dumps(schema, indent=2)
        + "\n```"
    )


def _render_child_schema_retry_prompt(
    original_prompt: str,
    schema: dict,
    err: str,
    rejected: str,
) -> str:
    rejected_trunc = rejected[:500]
    if len(rejected) > 500:
        rejected_trunc += f"...[truncated {len(rejected) - 500} chars]"
    return (
        f"{_render_child_schema_preamble(schema)}\n\n"
        "Your previous response failed schema validation.\n"
        f"Validation error: {err}\n"
        f"Rejected response: {rejected_trunc}\n\n"
        f"Original task:\n{original_prompt}"
    )


def _batch_repair_completer(llm_primitives, role: str, persona: str | None):
    """A ``parse_with_repair`` ``CompleteFn`` over ``llm_batch`` (one item).

    TD-21.22a: ``llm_batch`` now takes a ``json_schema`` parameter
    (threaded through ``_real_batch``/``_mock_batch``/``_worker_pool_batch``
    down to the same ``InferenceRequest.json_schema`` field ``llm_call``
    uses), so the repair/extraction turn gets the wire constraint too --
    the text preamble below stays as the belt-and-suspenders prompt framing
    ``_delegate_single`` (routing.py) also keeps alongside the wire schema.
    """

    def complete(messages, schema):
        rendered = "\n\n".join(
            f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages
        )
        prompt = f"{_render_child_schema_preamble(schema)}\n\n{rendered}"
        results = llm_primitives.llm_batch(
            [prompt], role=role, persona=persona, json_schema=schema,
        )
        return results[0] if results else ""

    return complete


class _CombinedOpsMixin:
    """Mixin providing combined REPL operations to reduce turn count.

    Feature flag: REPL_COMBINED_OPS env var (1/true/on to enable, default off).

    These operations batch common multi-tool patterns into single calls,
    based on autopilot log analysis showing repeated sequential web_search
    calls as the dominant REPL pattern (94.8% of tool calls).

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        tool_registry: Any | None — ToolRegistry for TOOL() invocations
        role: str — current agent role
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation
    """

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web for a query and return results.

        Args:
            query: Search query string.
            max_results: Maximum number of results (default 5).

        Returns:
            Search results as formatted text.
        """
        if self.tool_registry is None:
            return "[ERROR: No tool registry configured — cannot invoke web_search]"
        try:
            result = self.tool_registry.invoke(
                "web_search",
                self.role,
                caller_type="direct",
                query=query,
                max_results=max_results,
            )
            import json as _json
            if isinstance(result, (dict, list)):
                return _json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as exc:
            return f"[ERROR: web_search failed: {exc}]"

    def _batch_web_search(self, queries: list[str], max_results: int = 3) -> str:
        """Run multiple web searches and return consolidated results.

        Addresses the dominant pattern: repeated web_search -> web_search calls
        (5727 bigrams in autopilot logs). Reduces N sequential turns to 1 turn.

        Args:
            queries: List of search queries (max 5).
            max_results: Results per query (default 3).

        Returns:
            Consolidated results in JSON or TOON format.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        if not queries:
            return "[ERROR: No queries provided]"

        # Cap queries to prevent abuse
        if len(queries) > _MAX_QUERIES:
            queries = queries[:_MAX_QUERIES]
            capped = True
        else:
            capped = False

        if self.tool_registry is None:
            return "[ERROR: No tool registry configured — cannot invoke web_search]"

        results = {}
        for query in queries:
            try:
                result = self.tool_registry.invoke(
                    "web_search",
                    self.role,
                    caller_type="direct",
                    query=query,
                    max_results=max_results,
                )
                results[query] = result
            except Exception as e:
                results[query] = f"[ERROR: {type(e).__name__}: {e}]"

        self._exploration_log.add_event(
            "batch_web_search",
            {"queries": queries, "max_results": max_results, "capped": capped},
            results,
        )

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            lines = [f"=== batch_web_search ({len(queries)} queries) ==="]
            if capped:
                lines.append(f"[WARNING: Capped to {_MAX_QUERIES} queries]")
            for query, result in results.items():
                lines.append(f'--- "{query}" ---')
                if isinstance(result, (list, dict)):
                    lines.append(json.dumps(result, indent=2, default=str))
                else:
                    lines.append(str(result))
            return "\n".join(lines)
        else:
            output = {
                "operation": "batch_web_search",
                "query_count": len(queries),
                "capped": capped,
                "results": {q: r for q, r in results.items()},
            }
            return json.dumps(output, indent=2, default=str)

    def _search_and_verify(self, query: str, max_results: int = 3) -> str:
        """Web search followed by Wikipedia verification in one turn.

        Addresses: web_search -> search_wikipedia pattern (171 bigrams in logs).

        Args:
            query: Search query.
            max_results: Web results to return (default 3).

        Returns:
            Combined web + wikipedia results.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        if self.tool_registry is None:
            return "[ERROR: No tool registry configured — cannot invoke search tools]"

        web_result = None
        wiki_result = None

        # Step 1: web search
        try:
            web_result = self.tool_registry.invoke(
                "web_search",
                self.role,
                caller_type="direct",
                query=query,
                max_results=max_results,
            )
        except Exception as e:
            web_result = f"[ERROR: {type(e).__name__}: {e}]"

        # Step 2: wikipedia verification
        try:
            wiki_result = self.tool_registry.invoke(
                "search_wikipedia",
                self.role,
                caller_type="direct",
                query=query,
            )
        except Exception as e:
            wiki_result = f"[ERROR: {type(e).__name__}: {e}]"

        self._exploration_log.add_event(
            "search_and_verify",
            {"query": query, "max_results": max_results},
            {"web": web_result, "wiki": wiki_result},
        )

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            lines = [
                f'=== search_and_verify: "{query}" ===',
                "## Web Results",
            ]
            if isinstance(web_result, (list, dict)):
                lines.append(json.dumps(web_result, indent=2, default=str))
            else:
                lines.append(str(web_result))
            lines.append("## Wikipedia")
            if isinstance(wiki_result, (list, dict)):
                lines.append(json.dumps(wiki_result, indent=2, default=str))
            else:
                lines.append(str(wiki_result))
            return "\n".join(lines)
        else:
            output = {
                "operation": "search_and_verify",
                "query": query,
                "web_results": web_result,
                "wikipedia": wiki_result,
            }
            return json.dumps(output, indent=2, default=str)

    def _peek_grep(self, path: str, pattern: str, context_lines: int = 3) -> str:
        """Read a file and grep it in a single call.

        Preemptive: file exploration tools aren't used yet in autopilot,
        but this is the most natural combined file operation for when they are.

        Args:
            path: File path to read.
            pattern: Regex pattern to search.
            context_lines: Lines of context around matches (default 3).

        Returns:
            Matched lines with surrounding context.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            from src.repl_environment.exploration_core import grep_file
            from src.repl_environment.file_exploration import _explicit_exploration_root

            root, resolved = _explicit_exploration_root(self, path)
            result = grep_file(
                root, resolved, pattern, context_lines=max(0, context_lines),
                max_hits=None, max_line_chars=None, max_context_chars=None,
                max_bytes=None, ignore_case=False, splitlines=True,
                allow_empty_pattern=True, compile_after_read=True,
            )
        except FileNotFoundError:
            return f"[ERROR: File not found: {path}]"
        except IsADirectoryError:
            return f"[ERROR: Path is a directory: {path}]"
        except Exception as e:
            message = str(e)
            if message.startswith("invalid regex pattern: "):
                return f"[ERROR: Invalid regex pattern: {message[len('invalid regex pattern: '):]}]"
            return f"[ERROR: {type(e).__name__}: {e}]"

        matches = []
        for item in result.matches:
            start = max(0, item.line_num - 1 - max(0, context_lines))
            context_block = []
            for j, line in enumerate(item.context_lines, start=start):
                marker = ">>>" if j + 1 == item.line_num else "   "
                context_block.append(f"{marker} {j + 1:4d} | {line}")
            matches.append("\n".join(context_block))

        self._exploration_log.add_event(
            "peek_grep",
            {"path": path, "pattern": pattern, "context_lines": context_lines},
            {"match_count": len(matches)},
        )

        if not matches:
            return f'[No matches for pattern "{pattern}" in {path}]'

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            header = f'=== peek_grep: "{pattern}" in {path} ({len(matches)} matches) ==='
            return header + "\n" + "\n---\n".join(matches)
        else:
            output = {
                "operation": "peek_grep",
                "path": path,
                "pattern": pattern,
                "match_count": len(matches),
                "matches": matches,
            }
            return json.dumps(output, indent=2, default=str)

    def _batch_llm_query(
        self,
        prompts: list[str],
        role: str = "worker",
        persona: str | None = None,
        schema: dict | None = None,
        max_retries: int = 1,
    ) -> str:
        """Run multiple sub-LM queries in parallel with structured output.

        Wraps llm_primitives.llm_batch() with TOON/JSON formatting,
        input capping, and exploration logging. Useful for multi-sub-query
        tasks common in agentic/coder suites.

        Args:
            prompts: List of prompts to send to sub-LMs (max 5).
            role: Role determining which model to use (default "worker").
            persona: Optional persona name for system prompt injection.
            schema: Optional JSON Schema dict. When provided, child responses
                are parsed and validated before being returned.
            max_retries: Per-result schema retry count, capped to 2.

        Returns:
            Structured results in JSON or TOON format.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        if not prompts:
            return "[ERROR: No prompts provided]"

        if self.llm_primitives is None:
            return "[ERROR: No LLM primitives configured — cannot run batch queries]"

        if schema is not None and not isinstance(schema, dict):
            return "[ERROR: schema must be a JSON Schema dict]"

        # Cap prompts to prevent abuse
        if len(prompts) > _MAX_QUERIES:
            prompts = prompts[:_MAX_QUERIES]
            capped = True
        else:
            capped = False

        retry_budget = max(0, min(max_retries, _MAX_SCHEMA_RETRIES))
        query_prompts = (
            [f"{_render_child_schema_preamble(schema)}\n\n{prompt}" for prompt in prompts]
            if schema is not None
            else prompts
        )
        # TD-21.22a: the schema now also reaches the wire (json_schema=),
        # not just the prose preamble baked into query_prompts above -- one
        # schema constrains every prompt in THIS batch (per-batch, matching
        # how this call already builds one shared schema over the whole
        # prompt list; a caller needing per-prompt schemas would need one
        # llm_batch call per schema, or llm_call per prompt).
        raw_results = self.llm_primitives.llm_batch(
            query_prompts, role=role, persona=persona, json_schema=schema,
        )
        result_entries = []
        retry_count = 0
        validation_failures = 0

        for prompt, raw_result in zip(prompts, raw_results):
            if schema is None:
                result_entries.append({"prompt": prompt[:100], "response": raw_result})
                continue

            attempts = 1
            current_raw = raw_result
            ok, err, parsed = _validate_final_answer(current_raw, schema)

            if not ok:
                # TD-21.22 residual: one repair turn (fish, then at most one
                # constrained extraction call back to the same role) before
                # paying for a full re-call, multiplied across the batch.
                repair = parse_with_repair(
                    current_raw,
                    schema=schema,
                    complete=_batch_repair_completer(self.llm_primitives, role, persona),
                    site="repl_combined_ops",
                    # TD-21.34: same caller-schema fabrication risk as
                    # `repl_final`/`repl_delegate`.
                    require_evidence=True,
                )
                if repair.status in ("parsed", "repaired"):
                    parsed = repair.value
                    current_raw = json.dumps(parsed)
                    ok, err = True, None
                else:
                    err = repair.reason or err

            while not ok and attempts <= retry_budget:
                retry_count += 1
                attempts += 1
                retry_prompt = _render_child_schema_retry_prompt(
                    prompt,
                    schema,
                    err or "unknown validation error",
                    current_raw,
                )
                retry_result = self.llm_primitives.llm_batch(
                    [retry_prompt], role=role, persona=persona, json_schema=schema,
                )
                current_raw = retry_result[0] if retry_result else ""
                ok, err, parsed = _validate_final_answer(current_raw, schema)

            if not ok:
                validation_failures += 1

            result_entries.append(
                {
                    "prompt": prompt[:100],
                    "response": parsed if ok else current_raw,
                    "raw_response": current_raw,
                    "valid": ok,
                    "error": None if ok else err,
                    "attempts": attempts,
                }
            )

        self._exploration_log.add_event(
            "batch_llm_query",
            {
                "prompts": [p[:100] for p in prompts],
                "role": role,
                "persona": persona,
                "capped": capped,
                "schema": schema is not None,
            },
            {
                "response_count": len(raw_results),
                "response_lengths": [len(r) for r in raw_results],
                "retry_count": retry_count,
                "validation_failures": validation_failures,
            },
        )

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            lines = [f"=== batch_llm_query ({len(prompts)} prompts, role={role}) ==="]
            if capped:
                lines.append(f"[WARNING: Capped to {_MAX_QUERIES} prompts]")
            for i, entry in enumerate(result_entries):
                prompt = entry["prompt"]
                prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
                lines.append(f'--- prompt {i + 1}: "{prompt_preview}" ---')
                if schema is not None and not entry["valid"]:
                    lines.append(f"[SCHEMA INVALID: {entry['error']}]")
                response = entry["response"]
                lines.append(response if isinstance(response, str) else json.dumps(response, default=str))
            return "\n".join(lines)
        else:
            output = {
                "operation": "batch_llm_query",
                "prompt_count": len(prompts),
                "role": role,
                "persona": persona,
                "capped": capped,
                "schema_validation": schema is not None,
                "results": result_entries,
            }
            return json.dumps(output, indent=2, default=str)

    def _workspace_scan(
        self,
        query: str | None = None,
        limit: int = 20,
    ) -> str:
        """Single-turn metadata-first workspace exploration.

        Returns frecency-ranked file list from access history. If a query
        is provided, re-ranks results by filename relevance (substring match).

        This is the frecency-only fallback — full sub_lm summarization
        is blocked on AP-26 RLM integration testing.

        Args:
            query: Optional search query for filename filtering/re-ranking.
            limit: Maximum files to return (default 20).

        Returns:
            Structured file list in JSON or TOON format.
        """
        # INF-78 OAB-1: unavailable in a task_root-scoped request (no-op otherwise).
        from src.repl_environment.task_root import scope_refusal as _scope_refusal

        _scope_denial = _scope_refusal('workspace_scan')
        if _scope_denial is not None:
            return f"[ERROR: {_scope_denial}]"
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        # Get frecency-ranked files
        try:
            frecency_store = getattr(self, "_frecency_store", None)
            if frecency_store is None:
                from src.repl_environment.file_recency import FrecencyStore
                frecency_store = FrecencyStore()
                self._frecency_store = frecency_store

            top_files = frecency_store.top_files(limit=limit * 2 if query else limit)
        except Exception as e:
            return f"[ERROR: FrecencyStore unavailable: {e}]"

        if not top_files:
            return "[No file access history available. Use list_dir or peek to build history.]"

        # Re-rank by query relevance if provided
        if query:
            query_lower = query.lower()
            scored = []
            for path, frecency in top_files:
                path_lower = path.lower()
                if query_lower in path_lower:
                    relevance = 2.0
                elif any(query_lower in comp for comp in path_lower.split("/")):
                    relevance = 1.0
                else:
                    relevance = 0.0
                scored.append((path, frecency, relevance))
            scored.sort(key=lambda x: (x[2], x[1]), reverse=True)
            top_files = [(p, f) for p, f, _ in scored[:limit]]

        self._exploration_log.add_event(
            "workspace_scan",
            {"query": query, "limit": limit},
            {"file_count": len(top_files)},
        )

        if getattr(self.config, "use_toon_encoding", False):
            header = f"=== workspace_scan ({len(top_files)} files"
            if query:
                header += f', query="{query}"'
            header += ") ==="
            lines = [header]
            for i, (path, score) in enumerate(top_files):
                lines.append(f"  {i + 1:2d}. [{score:.2f}] {path}")
            return "\n".join(lines)
        else:
            output = {
                "operation": "workspace_scan",
                "query": query,
                "file_count": len(top_files),
                "files": [
                    {"path": path, "frecency_score": round(score, 3)}
                    for path, score in top_files
                ],
            }
            return json.dumps(output, indent=2)
