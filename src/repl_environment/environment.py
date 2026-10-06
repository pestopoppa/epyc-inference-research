"""Core REPLEnvironment class, factory function, and restricted variant.

This module assembles all mixins into the final REPLEnvironment class and
provides the create_repl_environment() factory function.
"""

from __future__ import annotations

import ast
import io
import logging
import os
import signal
import threading
import uuid
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

from src.repl_environment.types import (
    ExplorationLog,
    ExecutionResult,
    FinalSignal,
    REPLConfig,
    REPLSecurityError,
    REPLTimeout,
    wrap_tool_output,
)
from src.repl_environment.parallel_dispatch import (
    _ParallelCall,
    _eval_ast_arg,
    _extract_parallel_calls,
    extract_tool_calls,
    execute_parallel_calls,
)
from src.repl_environment.security import ASTSecurityVisitor
from src.repl_environment.unicode_sanitizer import sanitize_code_unicode
from src.repl_environment.combined_ops import _CombinedOpsMixin
from src.repl_environment.suggestions import _SuggestionsMixin
from src.repl_environment.file_tools import _FileToolsMixin
from src.repl_environment.document_tools import _DocumentToolsMixin
from src.repl_environment.routing import _RoutingMixin
from src.repl_environment.code_search import _CodeSearchMixin
from src.repl_environment.procedure_tools import _ProcedureToolsMixin
from src.repl_environment.context import _ContextMixin
from src.repl_environment.state import _StateMixin
from src.research_context import ResearchContext

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger


class FileToolsContract(Protocol):
    """Contract for filesystem-oriented mixin methods used by REPLEnvironment."""

    def _list_dir(self, path: str = ".") -> str: ...
    def _file_info(self, path: str) -> str: ...
    def _peek(self, n: int = 500, file_path: str | None = None, offset: int = 0) -> str: ...
    def _grep(self, pattern: str) -> str: ...


class DocumentToolsContract(Protocol):
    """Contract for document processing mixin methods."""

    def _ocr_document(self, path: str) -> str: ...
    def _extract_figure(self, path: str, page: int, index: int) -> str: ...


class RoutingContract(Protocol):
    """Contract for routing/delegation-related mixin methods."""

    def _my_role(self) -> str: ...
    def _route_advice(self, query: str) -> str: ...
    def _delegate(
        self,
        prompt: str,
        to_role: str = "worker_general",
        parallel: bool = False,
        reason: str = "",
        persona: str = "",
        schema: dict[str, Any] | None = None,
        max_retries: int = 1,
    ) -> str | list[str]: ...
    def _escalate(self, reason: str) -> str: ...


class ProcedureToolsContract(Protocol):
    """Contract for script/procedure mixin methods."""

    def _script(self, script_id: str, **kwargs: Any) -> Any: ...
    def _find_scripts(self, query: str) -> list[dict[str, Any]]: ...


class StateContract(Protocol):
    """Contract for state-tracking mixin methods."""

    def get_state(self) -> dict[str, Any]: ...
    def clear_state(self) -> None: ...


def _get_allowed_file_paths() -> list[str]:
    """Get allowed file path prefixes from config with fallback."""
    try:
        from src.config import get_config

        cfg = get_config()
        llm_root = str(cfg.paths.llm_root)
        # Ensure trailing slash for prefix matching
        return [
            llm_root if llm_root.endswith("/") else f"{llm_root}/",
            "/tmp/",
        ]
    except Exception:
        llm_root = os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm")
        return [f"{llm_root}/" if not llm_root.endswith("/") else llm_root, "/tmp/"]


class REPLEnvironment(
    _SuggestionsMixin,
    _CombinedOpsMixin,
    _FileToolsMixin,
    _DocumentToolsMixin,
    _RoutingMixin,
    _CodeSearchMixin,
    _ProcedureToolsMixin,
    _ContextMixin,
    _StateMixin,
):
    """Sandboxed Python REPL with context-as-variable pattern.

    The REPL provides:
    - `context`: The full input as a string (never sent to LLM)
    - `artifacts`: Dict for storing intermediate results
    - `peek(n)`: Return first n characters of context
    - `grep(pattern)`: Regex search in context
    - `FINAL(answer)`: Signal completion with final answer
    - `FINAL_VAR(var_name)`: Signal completion, return variable contents

    When tool_registry is provided:
    - `TOOL(name, **kwargs)`: Invoke a registered tool
    - `list_tools()`: List available tools for current role

    When script_registry is provided:
    - `SCRIPT(id, **kwargs)`: Invoke a prepared script by ID
    - `find_scripts(query)`: Find scripts matching a description
    """

    # Allowed file paths for file operations (security)
    ALLOWED_FILE_PATHS = _get_allowed_file_paths()

    def __init__(
        self,
        context: str,
        artifacts: dict[str, Any] | None = None,
        config: REPLConfig | None = None,
        llm_primitives: Any | None = None,  # LLMPrimitives instance
        tool_registry: Any | None = None,  # ToolRegistry instance
        script_registry: Any | None = None,  # ScriptRegistry instance
        role: str | None = None,  # Role for permission checking
        progress_logger: ProgressLogger | None = None,  # For exploration logging
        task_id: str | None = None,  # Task ID for progress logging
        # MemRL components for routing-aware REPL
        retriever: Any | None = None,  # TwoPhaseRetriever for recall/route_advice
        hybrid_router: Any | None = None,  # HybridRouter for route_advice
        # Structured mode for React-style execution
        structured_mode: bool = False,
        tool_context: dict[str, Any] | None = None,
    ):
        """Initialize the REPL environment.

        Args:
            context: The full input context (stored as variable, not in LLM prompt).
            artifacts: Optional dict of pre-existing artifacts from previous turns.
            config: Optional configuration for timeouts, output caps, etc.
            llm_primitives: Optional LLMPrimitives instance for llm_call/llm_batch.
            tool_registry: Optional ToolRegistry for TOOL() invocations.
            script_registry: Optional ScriptRegistry for SCRIPT() invocations.
            role: Role name for permission checking (e.g., "frontdoor", "coder_escalation").
            progress_logger: Optional ProgressLogger for logging exploration events.
            task_id: Optional task ID for associating exploration logs with tasks.
            retriever: Optional TwoPhaseRetriever for recall/route_advice.
            hybrid_router: Optional HybridRouter for route_advice.
            structured_mode: If True, enforce one-tool-per-turn React-style execution.
            tool_context: Optional context dict for cascading tool policy (e.g. no_web).
        """
        self.context = context
        self.artifacts = artifacts if artifacts is not None else {}
        self.config = config if config is not None else REPLConfig()
        self.llm_primitives = llm_primitives
        self.progress_logger = progress_logger
        self.task_id = task_id or str(uuid.uuid4())
        self.tool_registry = tool_registry
        self.script_registry = script_registry
        self.role = role or "worker_general"  # Default to restricted role
        self.tool_context = tool_context or {}

        # A timed-out REPL request may leave its worker thread running. Keep
        # this environment terminal so its late mutations cannot be executed
        # again or checkpointed as a completed turn.
        self._timeout_terminal = False

        # MemRL components for routing-aware functions (recall, route_advice)
        self._retriever = retriever
        self._hybrid_router = hybrid_router

        # Structured mode: React-style one-tool-per-turn execution
        # Can be set via config or constructor parameter (constructor takes precedence)
        self._structured_mode = structured_mode or self.config.structured_mode
        self._tool_chain_mode = os.environ.get("ORCHESTRATOR_TOOL_CHAIN_MODE", "seq").strip().lower()
        if self._tool_chain_mode not in {"legacy", "seq", "dep"}:
            self._tool_chain_mode = "seq"
        self._tool_chain_parallel_mutations = (
            os.environ.get("ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # Execution state
        self._final_answer: str | None = None
        self._execution_count = 0
        self._turn_counter = 0
        self._session_id = task_id or uuid.uuid4().hex[:8]
        self._last_spill_summary: str | None = None  # Rolling summary across spills
        self._tool_invocations = 0  # Count of TOOL()/CALL() invocations
        # Request-local invocation records (SimpleNamespace with tool_name,
        # elapsed_ms, success, chain_id, caller_type, result) captured at the
        # single _invoke_tool chokepoint (context.py), in call order. This is the
        # ONLY source the API uses for per-request tool telemetry — the shared
        # ToolRegistry invocation log is a process-global bounded diagnostic ring,
        # so reading it leaks other concurrent requests' tools into this one. Per-REPL (per request), so it is inherently request-scoped.
        self._invoked_tools: list = []
        self._active_tool_chain_id: str | None = None
        self._active_tool_chain_index: int = 0
        self._active_tool_chain_meta: dict[str, Any] | None = None
        self._chain_execution_log: list[dict[str, Any]] = []
        self._task_manager: Any = None
        self._task_type: str = "chat"

        # Exploration tracking (for forced exploration validation)
        self._exploration_calls = 0  # Count of peek/grep/llm_call calls
        self._exploration_log = ExplorationLog()  # Detailed exploration log

        # Grep hits buffer for two-stage summarization pipeline
        self._grep_hits_buffer: list[dict[str, Any]] = []

        # Key findings buffer for session persistence
        self._findings_buffer: list[dict[str, Any]] = []

        # Research context tracker for tool result lineage
        self._research_context = ResearchContext()
        self._last_research_node: str | None = None

        # Thread-safe lock for parallel tool dispatch (protects state mutations
        # in _exploration_calls, _exploration_log, _grep_hits_buffer, artifacts)
        self._state_lock = threading.Lock()

        # Feature flags (lazy import to avoid circular deps)
        from src.features import features as _get_features
        self._features = _get_features()
        self._deferred_tool_results = bool(self._features.deferred_tool_results)

        # INF-78 OAB-7: a request-carried context bundle (attach_context_bundle). None ==
        # the historical string `context`, unchanged.
        self._context_bundle = None
        # The bundle's print cap and preview size, COPIED at attach time so code that
        # reaches (and mutates) the bundle object cannot lift the cap.
        self._bundle_print_cap_bytes: int | None = None
        self._bundle_preview_chars: int | None = None

        # Build restricted globals
        self._globals = self._build_globals()
        # Snapshot built-in/injected globals so get_state() can highlight
        # user-defined variables created by model code in later turns.
        self._builtin_global_keys = frozenset(self._globals.keys())

        # Verify mixin contracts are satisfied (runtime validation)
        # This ensures all required attributes exist before mixin methods are called
        required_attrs = [
            # Core state
            "config",
            "context",
            "artifacts",
            "role",
            "task_id",
            # Counters
            "_exploration_calls",
            "_execution_count",
            "_tool_invocations",
            "_invoked_tools",
            # Logs and buffers
            "_exploration_log",
            "_grep_hits_buffer",
            "_findings_buffer",
            # Research context
            "_research_context",
            "_last_research_node",
            # State
            "_final_answer",
            "_globals",
            # Optional dependencies
            "llm_primitives",
            "tool_registry",
            "script_registry",
            "progress_logger",
            "_retriever",
            "_hybrid_router",
            # Methods
            "_validate_file_path",
            "_build_globals",
        ]
        for attr in required_attrs:
            assert hasattr(self, attr), (
                f"Mixin contract violated: missing required attribute '{attr}'"
            )

    @property
    def timed_out(self) -> bool:
        """Whether execution timed out and this environment is terminal."""
        return self._timeout_terminal

    @property
    def timeout_error(self) -> str:
        """Canonical user-facing error for this environment's timeout."""
        return f"REPLTimeout: Execution timed out after {self.config.timeout_seconds}s"

    def mark_timed_out(self) -> str:
        """Make timeout terminal for this environment and return its error."""
        self._timeout_terminal = True
        return self.timeout_error

    def _timeout_result(self, start_time: float) -> ExecutionResult:
        import time

        return ExecutionResult(
            output="",
            is_final=False,
            error=self.timeout_error,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    # Modules safe for import inside the REPL sandbox.  These are
    # pure-computation or data-structure libraries — no filesystem,
    # network, or process-spawning capabilities.
    SAFE_IMPORT_MODULES: frozenset[str] = frozenset({
        # Standard library — math & data
        "math", "cmath", "decimal", "fractions", "statistics",
        "random", "numbers",
        # Standard library — data structures & iteration
        "collections", "itertools", "functools", "operator",
        "heapq", "bisect", "array", "queue",
        # Standard library — string & text
        "re", "string", "textwrap", "unicodedata",
        # Standard library — time (read-only)
        "time", "datetime", "calendar",
        # Standard library — misc safe
        "copy", "enum", "dataclasses", "typing",
        "json", "csv", "io", "struct", "base64",
        "hashlib", "hmac", "secrets",
        "pprint", "reprlib",
        # Scientific computing (if installed)
        "numpy", "scipy", "sympy",
    })
    # Read-only REPL-callable helpers (safe for parallel dispatch).
    # This covers non-registry built-ins like peek/grep plus utility calls.
    _READ_ONLY_REPL_TOOLS: frozenset[str] = frozenset({
        "peek", "grep", "list_dir", "file_info", "list_tools",
        "recall", "list_findings", "registry_lookup", "my_role",
        "route_advice", "list_procedures", "get_procedure_status",
        "context_len", "find_scripts", "benchmark_compare", "fetch_report",
        "code_search", "doc_search",
    })
    _CHAINABLE_REPL_TOOLS: frozenset[str] = frozenset({
        "run_shell",
        "run_python_code",
        "file_write_safe",
        "log_append",
        "benchmark_run",
    })
    _PARALLEL_MUTATION_REPL_TOOLS: frozenset[str] = frozenset({
        "run_shell",
        "file_write_safe",
        "log_append",
        "benchmark_run",
    })

    def _build_globals(self) -> dict[str, Any]:
        """Build the restricted globals dict for exec()."""
        import builtins

        # Build safe builtins from the builtins module
        safe_builtins = {}
        for name in self.config.allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Safe import: allows whitelisted modules only
        _safe_modules = self.SAFE_IMPORT_MODULES

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            top = name.split(".")[0]
            if top not in _safe_modules:
                raise ImportError(
                    f"Module '{name}' is not allowed in the REPL sandbox. "
                    f"Allowed: {', '.join(sorted(_safe_modules))}"
                )
            return builtins.__import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = _safe_import

        # Pre-load commonly used safe modules so _strip_import_lines()
        # doesn't cause NameErrors when it removes import statements.
        import json as _json
        import math as _math
        import re as _re
        import collections as _collections
        import itertools as _itertools
        import functools as _functools
        import statistics as _statistics
        import datetime as _datetime
        import fractions as _fractions
        import decimal as _decimal
        import copy as _copy

        # Optional scientific modules (may not be installed)
        try:
            import numpy as _numpy
        except ImportError:
            _numpy = None
        try:
            import scipy as _scipy
        except ImportError:
            _scipy = None

        globals_dict = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",  # Needed for class definitions in exec()
            # Context variables. With a context bundle attached, `context` is its counted
            # view -- also after a checkpoint restore or reset() rebuilds these globals.
            "context": (self.context if getattr(self, "_context_bundle", None) is None
                        else self._bundle_view()),
            "artifacts": self.artifacts,
            # Safe modules (pre-loaded to match _strip_import_lines behavior)
            "json": _json,
            "math": _math,
            "re": _re,
            "collections": _collections,
            "itertools": _itertools,
            "functools": _functools,
            "statistics": _statistics,
            "datetime": _datetime,
            "fractions": _fractions,
            "decimal": _decimal,
            "copy": _copy,
            **({"numpy": _numpy} if _numpy else {}),
            **({"scipy": _scipy} if _scipy else {}),
            # Built-in functions
            "peek": self._peek,
            "grep": self._grep,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "STUCK": self._stuck,
            # Curation: mark a variable as worth keeping, with a note (D-d)
            "remember": self.remember,
            # Document processing tools
            "ocr_document": self._ocr_document,
            "analyze_figure": self._analyze_figure,
            "extract_figure": self._extract_figure,
            # File system tools
            "list_dir": self._list_dir,
            "file_info": self._file_info,
            # Archive tools
            "archive_open": self._archive_open,
            "archive_extract": self._archive_extract,
            "archive_file": self._archive_file,
            "archive_search": self._archive_search,
            # Web tools
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            # Combined operations (feature-gated by REPL_COMBINED_OPS env var)
            "batch_web_search": self._batch_web_search,
            "search_and_verify": self._search_and_verify,
            "peek_grep": self._peek_grep,
            "batch_llm_query": self._batch_llm_query,
            "workspace_scan": self._workspace_scan,
            # Memory tools
            "recall": self._recall,
            # Code & document retrieval (NextPLAID multi-vector search)
            "code_search": self._code_search,
            "doc_search": self._doc_search,
            # Session persistence tools
            "mark_finding": self._mark_finding,
            "list_findings": self._list_findings,
            # Routing & self-assessment tools
            "escalate": self._escalate,
            "my_role": self._my_role,
            "route_advice": self._route_advice,
            "delegate": self._delegate,
            "fetch_report": self._fetch_report,
            # Shell tools (sandboxed)
            "run_shell": self._run_shell,
            "run_python_code": self._run_python_code,
            # Self-management procedure tools
            "run_procedure": self._run_procedure,
            "list_procedures": self._list_procedures,
            "get_procedure_status": self._get_procedure_status,
            "checkpoint_create": self._checkpoint_create,
            "checkpoint_restore": self._checkpoint_restore,
            "registry_lookup": self._registry_lookup,
            "registry_update": self._registry_update,
            "benchmark_run": self._benchmark_run,
            "benchmark_compare": self._benchmark_compare,
            "gate_run": self._gate_run,
            "log_append": self._log_append,
            "file_write_safe": self._file_write_safe,
            # Long context exploration tools
            "chunk_context": self._chunk_context,
            "summarize_chunks": self._summarize_chunks,
            "context_len": self._context_len,
        }

        # Add LLM primitives if available (wrapped for exploration tracking)
        if self.llm_primitives is not None:
            globals_dict["llm_call"] = self._tracked_llm_call
            globals_dict["llm_batch"] = self._tracked_llm_batch

        # Add tool registry functions if available
        if self.tool_registry is not None:
            globals_dict["TOOL"] = self._invoke_tool
            globals_dict["CALL"] = self._call_tool
            globals_dict["list_tools"] = self._list_tools

        # Add script registry functions if available
        if self.script_registry is not None:
            globals_dict["SCRIPT"] = self._invoke_script
            globals_dict["find_scripts"] = self._find_scripts

        return globals_dict

    def _validate_file_path(self, path: str) -> tuple[bool, str | None]:
        """Validate that a file path is allowed.

        Args:
            path: Path to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        import os

        # BEP harness (Phase 1, #1/#2): when ORCHESTRATOR_EDIT_ROOT is active, a RELATIVE model
        # path resolves under the scratch task-root (not the process cwd), and the task-root is an
        # allowed prefix — so the model inspects/edits the same scratch repo. Default-off:
        # resolve_task_path == os.path.realpath and the allowed list is unchanged → exact parity.
        from src.repl_environment.task_root import (
            get_task_root,
            resolve_task_path,
            task_root_active,
        )

        resolved = resolve_task_path(path)
        from src.repl_environment.task_root import request_scope

        scope = request_scope()
        if scope is not None:
            # INF-78 OAB-1: a task_root-scoped request reads ONLY its root plus the explicit
            # read_roots (writes are gated separately in _file_write_safe).
            denial = scope.read_denial(resolved)
            if denial is not None:
                return False, denial
            from src.repl_environment.knowledge_fence import check_path

            fence_denial = check_path(resolved)
            if fence_denial is not None:
                return False, fence_denial
            return True, None
        if task_root_active():
            # ISOLATION (BEP/DCP A/B): the model may touch ONLY the scratch task-root. The prior
            # code APPENDED the scratch root to the global allowed set (llm_root + /tmp), but the
            # scratch lives under /tmp, so the global /tmp prefix let every outside path (/tmp/x,
            # ../x, the orchestrator tree) validate — a real isolation leak (operator-found
            # 2026-05-27). Restrict to the scratch root ONLY; realpath both sides so a symlinked
            # /tmp (or trailing-slash differences) still match.
            tr = os.path.realpath(str(get_task_root()))
            allowed_prefixes = [tr if tr.endswith("/") else f"{tr}/"]
        else:
            allowed_prefixes = list(self.ALLOWED_FILE_PATHS)
        for allowed in allowed_prefixes:
            if resolved.startswith(allowed):
                # AP-54 eval knowledge fence: no-op unless the request carried
                # `eval_fence` (then it records the path, and denies when armed).
                from src.repl_environment.knowledge_fence import check_path

                fence_denial = check_path(path)
                if fence_denial is not None:
                    return False, fence_denial
                return True, None
        return False, f"Path not in allowed locations: {allowed_prefixes}"

    def _maybe_wrap_tool_output(self, output: str) -> str:
        """Return wrapped tool output in legacy mode, raw output in deferred mode."""
        if self._deferred_tool_results:
            return output
        self.artifacts.setdefault("_tool_outputs", []).append(output)
        return wrap_tool_output(output)

    def _get_read_only_tools(self) -> set[str]:
        """Return the unified read-only tool set for REPL dispatch/telemetry."""
        tools = set(self._READ_ONLY_REPL_TOOLS)
        if self.tool_registry and hasattr(self.tool_registry, "get_read_only_tools"):
            try:
                tools.update(self.tool_registry.get_read_only_tools())
            except Exception:
                pass
        return tools

    def _validate_code(self, code: str) -> None:
        """Validate code for dangerous patterns using AST analysis.

        Args:
            code: Python code to validate.

        Raises:
            REPLSecurityError: If dangerous patterns detected or syntax is invalid.
        """
        # First, try to parse the code to get an AST
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            # Let the actual execution handle syntax errors for better messages
            return

        # Run AST-based security analysis
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        if visitor.violations:
            violation = visitor.violations[0]
            # Provide actionable hints for common cases
            hints = ""
            if "input()" in violation:
                hints = (
                    " Hint: input() is not available in the REPL. "
                    "For competitive programming solutions that read from stdin, "
                    "wrap your solution in a string and use "
                    'CALL("run_python_code", code=your_code_string, stdin_data=test_input) '
                    "to test it, then FINAL(your_code_string) to submit."
                )

            elif "exec()" in violation or "eval()" in violation:
                hints = (
                    " Hint: use run_python_code(code_string, stdin_data) "
                    "to test code, or write code directly (it is executed "
                    "automatically). Do NOT use exec()/eval()."
                )
            elif "open()" in violation:
                hints = (
                    " Hint: use peek(path) to read files or "
                    "file_write_safe(path, content) to write."
                )
            elif "import " in violation:
                hints = (
                    " Hint: for competitive programming solutions that need sys/os, "
                    "wrap your solution in a string and use "
                    'CALL("run_python_code", code=your_code_string, stdin_data=test_input) '
                    "to test it, then FINAL(your_code_string) to submit."
                )
            raise REPLSecurityError(
                f"Dangerous operation not allowed: {violation}.{hints}"
            )

    def get_chain_execution_log(self) -> list[dict[str, Any]]:
        """Return structured chain execution diagnostics for this REPL session."""
        return [dict(item) for item in self._chain_execution_log]

    def _finalize_active_chain_meta(self) -> None:
        """Persist and clear active chain metadata for the current structured turn."""
        if self._active_tool_chain_meta is not None:
            self._chain_execution_log.append(dict(self._active_tool_chain_meta))
        self._active_tool_chain_id = None
        self._active_tool_chain_index = 0
        self._active_tool_chain_meta = None

    # ------------------------------------------------------------------
    # INF-78 OAB-7: the context bundle as the REPL variable `context`
    # ------------------------------------------------------------------
    def attach_context_bundle(self, bundle: Any) -> None:
        """Bind a request's ``ContextBundle`` to the REPL name ``context``.

        ``self.context`` (the root prompt text) stays a string for the internals that
        read it; the MODEL's ``context`` becomes the counted bundle view, and the
        no-``file_path`` forms of ``peek``/``grep``/``context_len``/``chunk_context``
        read the bundle (counted) instead of the root prompt the model already has.
        From here on every turn's printed output is capped at the bundle's
        ``print_cap_bytes`` -- the only route from a pulled value into the root prompt.
        The cap is copied here, into this environment, and applied from the copy.
        """
        from src.repl_environment.context_bundle import CAP_MARKER_MAX_CHARS

        if hasattr(self, "_restricted_executor"):
            # The RestrictedPython executor keeps its own globals (the string context);
            # a bundle bound here would be silently absent there. /chat never builds it.
            raise ValueError("context bundles are not supported under RestrictedPython execution")
        cap = int(bundle.print_cap_bytes)
        self._bundle_print_cap_bytes = cap
        self._bundle_preview_chars = cap + CAP_MARKER_MAX_CHARS
        self._context_bundle = bundle
        self._globals["context"] = self._bundle_view()

    def _bundle_view(self) -> Any:
        from src.repl_environment.context_bundle import repl_view

        return repl_view(self._context_bundle)

    @property
    def context_bundle(self) -> Any:
        """The attached ``ContextBundle``, or None."""
        return self._context_bundle

    @property
    def bundle_output_preview_chars(self) -> int | None:
        """Output-preview size the graph must allow so a capped turn is not re-truncated
        (None without a bundle: the graph keeps its own preview)."""
        return None if self._context_bundle is None else self._bundle_preview_chars

    def _bundle_begin_turn(self) -> None:
        if self._context_bundle is not None:
            self._context_bundle.begin_turn(self._execution_count)

    def _bundle_cap_output(self, output: str) -> str:
        """Cap one turn's printed output at the bundle's per-turn byte cap (no-op
        without a bundle). Runs BEFORE ``_spill_output``, so a capped turn never spills
        or asks the worker for a summary."""
        if self._context_bundle is None:
            return output
        from src.repl_environment.context_bundle import cap_printed_output

        shown, raw_bytes, shown_bytes, capped = cap_printed_output(
            output, self._bundle_print_cap_bytes)
        try:
            self._context_bundle.record_printed(raw_bytes, shown_bytes, capped)
        except Exception as exc:  # noqa: BLE001 -- accounting never un-caps a turn
            _log.warning("context bundle: print accounting failed: %s", exc)
        return shown

    def _spill_output(self, output: str) -> str:
        """Write full output to file when it exceeds output_cap, return summary.

        Uses a rolling summary pattern: each spill passes the previous summary
        + new tail to the worker, so the summary accumulates knowledge across
        turns (like a compressed scratchpad). The model sees an always-up-to-date
        summary without re-reading earlier spill files.

        Fallback: static head/tail for unit tests or worker failure.

        Args:
            output: Raw execution output string.

        Returns:
            Original output if within cap, or summary string with spill path.
        """
        if len(output) <= self.config.output_cap:
            return output

        self._turn_counter += 1

        # 1. Spill full output to file
        spill_dir = Path(self.config.spill_dir) / self._session_id
        spill_dir.mkdir(parents=True, exist_ok=True)
        spill_path = spill_dir / f"turn_{self._turn_counter}.txt"
        spill_path.write_text(output)

        lines = output.splitlines()
        n_lines = len(lines)
        header = f"[Output: {len(output)} chars, {n_lines} lines → {spill_path}]"
        # Signatures are peek(n, file_path=None) / grep(pattern, file_path=None): pass the
        # path by keyword, or peek/grep treat it as n / the regex and never read the file.
        footer = (
            f'Use peek(2000, file_path="{spill_path}") or '
            f'grep("<pattern>", file_path="{spill_path}") to inspect.'
        )

        # 2. Try worker summary. role="worker" resolves to worker_general, an alias
        #    on frontdoor's :8070 fleet (Qwen3.6-35B-A3B MTP Q8, CPU) since the
        #    2026-09-22 lineup cutover; it shares that server with frontdoor traffic.
        if self.llm_primitives is not None:
            try:
                # Rolling summary: previous summary + new tail
                # The worker updates the running summary with new information,
                # so the model always has accumulated context.
                ctx_budget = 4000
                tail_budget = ctx_budget
                if self._last_spill_summary:
                    # Reserve space for previous summary context
                    prev_ctx = f"Previous summary:\n{self._last_spill_summary}\n\nNew output (tail):\n"
                    tail_budget = ctx_budget - len(prev_ctx)
                    ctx = prev_ctx + output[-tail_budget:]
                    prompt = (
                        "Update the previous summary with information from this new output. "
                        "Preserve key findings from before, add new results/errors/values. "
                        "Drop details that are superseded by newer output. Max 10 lines."
                    )
                else:
                    # First spill: just summarize the tail
                    ctx = output[-tail_budget:]
                    prompt = (
                        "Summarize this program output concisely. "
                        "Include: key results, error indicators, final values, "
                        "pass/fail counts if present. Max 10 lines."
                    )

                summary = self.llm_primitives.llm_call(
                    prompt=prompt,
                    context_slice=ctx,
                    role="worker",
                    n_tokens=512,
                    skip_suffix=True,
                )
                self._last_spill_summary = summary
                return f"{header}\n{summary}\n{footer}"
            except Exception:
                pass  # Fall through to static summary

        # 3. Fallback: static head/tail
        head = "\n".join(lines[:15])
        tail = "\n".join(lines[-5:]) if n_lines > 20 else ""
        parts = [header, head]
        if tail:
            parts.append(f"...\n[lines {n_lines - 4}–{n_lines}]")
            parts.append(tail)
        parts.append(footer)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Tool-hint injection for failed code that looks like a tool call
    # ------------------------------------------------------------------
    def _tool_hint_if_relevant(self, code: str, exc: Exception | None = None) -> str:
        """Return a tool-usage hint string if *code* looks like a failed tool call.

        Returns empty string when no hint is appropriate.
        """
        import re as _re

        # Quick patterns that suggest the model tried to call a tool
        _TOOL_PATTERNS = (
            _re.compile(r'"(?:function|name)"\s*:\s*"(\w+)"'),       # JSON tool_call
            _re.compile(r'\b(web_search|web_research|calculator|python|bash)\s*\('),  # direct name()
            _re.compile(r'tool_call', _re.IGNORECASE),
            _re.compile(r'"type"\s*:\s*"function"'),
        )

        if not any(p.search(code) for p in _TOOL_PATTERNS):
            return ""

        # Build hint
        parts: list[str] = [
            "\n--- Tool Usage Hint ---",
            "It looks like you tried to call a tool but the syntax was incorrect.",
            "Use the CALL() function to invoke tools:",
            "  result = CALL('tool_name', arg1=value1, arg2=value2)",
        ]

        # List available tools if registry exists
        try:
            tools = self._list_tools()
            if tools:
                names = [t if isinstance(t, str) else getattr(t, "name", str(t)) for t in tools]
                parts.append(f"Available tools: {', '.join(names)}")
        except Exception:
            pass

        parts.append("Run list_tools() for full details.")
        parts.append("--- End Hint ---")
        return "\n".join(parts)

    def _execute_structured(self, code: str, start_time: float) -> ExecutionResult:
        """Execute in structured (React-style) mode: one tool call per turn.

        This mode enforces the Thought/Action/Observation pattern:
        - Parses code for a single tool function call
        - If FINAL() found, treats as completion
        - If multiple tools, returns error requesting one at a time
        - Otherwise executes single tool and returns observation

        Args:
            code: Python code to execute (should be single tool call).
            start_time: Performance timer start for elapsed_seconds.

        Returns:
            ExecutionResult with observation output or final answer.
        """
        import re
        import time

        if self.timed_out:
            return self._timeout_result(start_time)

        code = sanitize_code_unicode(code)

        # Validate code for dangerous patterns first
        try:
            self._validate_code(code)
        except REPLSecurityError as e:
            return ExecutionResult(
                output="",
                is_final=False,
                error=str(e),
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Check for FINAL() call - treat as completion.
        # Try triple-quoted strings first (models wrap multi-line code in
        # FINAL('''...''') or FINAL(\"\"\"...\"\"\") — auto_wrap_final
        # also generates this form), then fall back to single-quoted.
        final_match = (
            re.search(r"""\bFINAL\s*\(\s*'{3}(.+?)'{3}\s*\)""", code, re.DOTALL)
            or re.search(r'''\bFINAL\s*\(\s*"{3}(.+?)"{3}\s*\)''', code, re.DOTALL)
            or re.search(r'\bFINAL\s*\(\s*["\'](.+?)["\']\s*\)', code, re.DOTALL)
        )
        final_var_match = re.search(r'\bFINAL_VAR\s*\(\s*["\'](\w+)["\']\s*\)', code)

        if final_match:
            answer = next((g for g in final_match.groups() if g is not None), "")
            return ExecutionResult(
                output="",
                is_final=True,
                final_answer=answer,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        if final_var_match:
            var_name = final_var_match.group(1)
            answer = str(self._globals.get(var_name, f"[Variable '{var_name}' not found]"))
            return ExecutionResult(
                output="",
                is_final=True,
                final_answer=answer,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # List of known tool functions in the REPL environment
        tool_functions = {
            "peek",
            "grep",
            "llm_call",
            "llm_batch",
            "TOOL",
            "CALL",
            "list_tools",
            "SCRIPT",
            "find_scripts",
            "list_dir",
            "file_info",
            "ocr_document",
            "analyze_figure",
            "extract_figure",
            "archive_open",
            "archive_extract",
            "archive_file",
            "archive_search",
            "web_fetch",
            "recall",
            "mark_finding",
            "list_findings",
            "escalate",
            "my_role",
            "route_advice",
            "delegate",
            "fetch_report",
            "run_shell",
            "run_python_code",
            "run_procedure",
            "list_procedures",
            "get_procedure_status",
            "checkpoint_create",
            "checkpoint_restore",
            "registry_lookup",
            "registry_update",
            "benchmark_run",
            "benchmark_compare",
            "gate_run",
            "log_append",
            "file_write_safe",
            "chunk_context",
            "summarize_chunks",
            "context_len",
        }

        # Count tool function calls using AST, so comments/strings do not
        # trigger false positives.
        call_sites = extract_tool_calls(code, tool_functions)
        tool_calls = [site.func_name for site in call_sites]

        if len(tool_calls) > 1:
            self._active_tool_chain_id = uuid.uuid4().hex[:12]
            self._active_tool_chain_index = 0
            self._active_tool_chain_meta = {
                "chain_id": self._active_tool_chain_id,
                "mode_requested": self._tool_chain_mode,
                "mode_used": "seq",
                "fallback_to_seq": False,
                "parallel_mutations_enabled": self._tool_chain_parallel_mutations,
                "waves": 0,
                "steps": len(tool_calls),
                "wave_timeline": [],
            }
            # Check if ALL tool calls are read-only — if so, allow parallel execution
            # Read-only tools (grep, peek, list_dir, etc.) are safe to run concurrently
            read_only_tools = self._get_read_only_tools()
            tool_names = list(tool_calls)
            all_read_only = all(name in read_only_tools for name in tool_names)
            non_read_only = [name for name in tool_names if name not in read_only_tools]
            chainable_tools = set()
            if self.tool_registry and hasattr(self.tool_registry, "get_chainable_tools"):
                try:
                    chainable_tools = set(self.tool_registry.get_chainable_tools())
                except Exception:
                    chainable_tools = set()
            chainable_tools.update(self._CHAINABLE_REPL_TOOLS)
            blocked_routing_tools = {"delegate", "escalate"}
            non_chainable: set[str] = set()
            for name in non_read_only:
                if name in blocked_routing_tools:
                    non_chainable.add(name)
                    continue
                if name in {"TOOL", "CALL"}:
                    # Check wrapped tool eligibility below.
                    continue
                if name not in chainable_tools:
                    non_chainable.add(name)

            # TOOL/CALL wrapping is allowed only if the referenced registry
            # tool is statically known and chain-enabled.
            if self.tool_registry:
                try:
                    parsed = ast.parse(code)
                except SyntaxError:
                    parsed = None
                if parsed is not None:
                    for node in ast.walk(parsed):
                        if not isinstance(node, ast.Call):
                            continue
                        if not isinstance(node.func, ast.Name):
                            continue
                        if node.func.id not in {"TOOL", "CALL"}:
                            continue
                        if not node.args:
                            non_chainable.add(f"{node.func.id}(dynamic)")
                            continue
                        first_arg = node.args[0]
                        if not (
                            isinstance(first_arg, ast.Constant)
                            and isinstance(first_arg.value, str)
                        ):
                            non_chainable.add(f"{node.func.id}(dynamic)")
                            continue
                        wrapped_tool = first_arg.value
                        if wrapped_tool not in chainable_tools:
                            non_chainable.add(f"{node.func.id}({wrapped_tool})")

            non_chainable_sorted = sorted(non_chainable)

            if not all_read_only and non_chainable_sorted:
                self._finalize_active_chain_meta()
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=(
                        "Structured mode: multi-tool chaining requires chain opt-in for non-read-only tools. "
                        f"Blocked tools: {', '.join(non_chainable_sorted)}. "
                        "Use one tool per turn, or mark tools with allowed_callers=['chain']."
                    ),
                    elapsed_seconds=time.perf_counter() - start_time,
                )
            # All read-only: attempt parallel dispatch via AST extraction
            if all_read_only and self._features.parallel_tools:
                parallel_calls = _extract_parallel_calls(
                    code, self._globals, read_only_tools
                )
                if parallel_calls is not None and len(parallel_calls) > 1:
                    self._active_tool_chain_meta = {
                        "chain_id": self._active_tool_chain_id,
                        "mode_requested": self._tool_chain_mode,
                        "mode_used": "parallel_read_only",
                        "fallback_to_seq": False,
                        "waves": 1,
                        "steps": len(parallel_calls),
                        "wave_timeline": [
                            {
                                "wave_index": 0,
                                "tools": [c.func_name for c in parallel_calls],
                                "mode_used": "parallel_read_only",
                                "elapsed_ms": None,
                                "fallback_to_seq": False,
                                "parallel_mutations_enabled": False,
                            }
                        ],
                    }
                    results = execute_parallel_calls(
                        parallel_calls, self._globals, self._state_lock
                    )
                    # Inject results into globals for subsequent code
                    for var_name, value in results.items():
                        if var_name and not var_name.startswith("_result_"):
                            self._globals[var_name] = value
                    # Format combined observation
                    parts = []
                    for c in parallel_calls:
                        key = c.target_var or f"_result_{c.index}"
                        val = results.get(key, "")
                        label = c.target_var or c.func_name
                        parts.append(f"[{label}]: {val}")
                    observation = self._spill_output(
                        "Observation:\n" + "\n---\n".join(parts)
                    )
                    self._finalize_active_chain_meta()
                    return ExecutionResult(
                        output=observation,
                        is_final=False,
                        elapsed_seconds=time.perf_counter() - start_time,
                    )
            if self._tool_chain_mode == "dep":
                dep_result = self._execute_dependency_aware_chain(
                    code=code,
                    start_time=start_time,
                    read_only_tools=read_only_tools,
                    tool_functions=tool_functions,
                )
                if dep_result is not None:
                    self._finalize_active_chain_meta()
                    return dep_result
                if self._active_tool_chain_meta is not None:
                    self._active_tool_chain_meta["fallback_to_seq"] = True
                    self._active_tool_chain_meta["mode_used"] = "seq"
            # Fall through to sequential exec()

        # Execute the code (single tool or simple expression)
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_handler = None
        _alarm_set = False
        try:
            if hasattr(signal, "SIGALRM"):
                try:

                    def timeout_handler(signum, frame):
                        raise REPLTimeout(
                            f"Execution timed out after {self.config.timeout_seconds}s"
                        )

                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.config.timeout_seconds)
                    _alarm_set = True
                except ValueError:
                    pass

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self._globals)

                output = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Redact credentials before they enter model context
                from src.repl_environment.redaction import redact_if_enabled

                output = redact_if_enabled(output)

                # Format as observation for React-style loop
                if output:
                    observation = f"Observation: {output.strip()}"
                else:
                    observation = "Observation: [No output]"

                # Spill large output to file with summary
                observation = self._spill_output(observation)

                # S3a: Append contextual suggestions (feature-gated)
                _tool_name = tool_calls[0] if tool_calls else ""
                observation = self._maybe_append_suggestions(_tool_name, observation)

                return ExecutionResult(
                    output=observation,
                    is_final=False,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except FinalSignal as e:
                return ExecutionResult(
                    output=stdout_capture.getvalue(),
                    is_final=True,
                    final_answer=e.answer,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except SyntaxError as e:
                hint = self._tool_hint_if_relevant(code, e)
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=f"SyntaxError: {e}{hint}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except Exception as e:
                if isinstance(e, REPLTimeout):
                    self.mark_timed_out()
                    return self._timeout_result(start_time)
                hint = self._tool_hint_if_relevant(code, e)
                return ExecutionResult(
                    output=self._bundle_cap_output(stdout_capture.getvalue()),
                    is_final=False,
                    error=f"{type(e).__name__}: {e}{hint}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

        finally:
            self._finalize_active_chain_meta()
            if _alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _execute_dependency_aware_chain(
        self,
        *,
        code: str,
        start_time: float,
        read_only_tools: set[str],
        tool_functions: set[str],
    ) -> ExecutionResult | None:
        """Execute top-level tool chains via dependency-wave scheduling.

        Returns None to signal fallback to sequential exec() when analysis
        cannot safely classify the chain.
        """
        import time
        from collections import defaultdict, deque

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        steps: list[dict[str, Any]] = []
        produced_by: dict[str, int] = {}
        for idx, stmt in enumerate(tree.body):
            target_var: str | None = None
            call_node: ast.Call | None = None
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    return None
                if not isinstance(stmt.value, ast.Call) or not isinstance(stmt.value.func, ast.Name):
                    return None
                target_var = stmt.targets[0].id
                call_node = stmt.value
            elif isinstance(stmt, ast.Expr):
                if not isinstance(stmt.value, ast.Call) or not isinstance(stmt.value.func, ast.Name):
                    return None
                call_node = stmt.value
            else:
                return None

            func_name = call_node.func.id
            if func_name not in tool_functions:
                return None

            name_refs: set[str] = set()
            for arg_node in list(call_node.args) + [kw.value for kw in call_node.keywords if kw.arg is not None]:
                for n in ast.walk(arg_node):
                    if isinstance(n, ast.Name):
                        name_refs.add(n.id)

            deps = {produced_by[n] for n in name_refs if n in produced_by}
            is_read_only = func_name in read_only_tools
            parallel_mutation_candidate = (
                self._tool_chain_parallel_mutations
                and func_name in self._PARALLEL_MUTATION_REPL_TOOLS
            )

            steps.append(
                {
                    "index": idx,
                    "func_name": func_name,
                    "target_var": target_var,
                    "call_node": call_node,
                    "deps": deps,
                    "is_read_only": is_read_only,
                    "parallel_mutation_candidate": parallel_mutation_candidate,
                }
            )
            if target_var:
                produced_by[target_var] = idx

        if len(steps) < 2:
            return None

        # Build dependency waves (Kahn topological scheduling).
        indegree: dict[int, int] = {s["index"]: len(s["deps"]) for s in steps}
        dependents: dict[int, list[int]] = defaultdict(list)
        for s in steps:
            for dep in s["deps"]:
                dependents[dep].append(s["index"])

        ready = deque(sorted(i for i, d in indegree.items() if d == 0))
        waves: list[list[int]] = []
        scheduled = 0
        while ready:
            current_wave = list(ready)
            ready = deque()
            waves.append(current_wave)
            scheduled += len(current_wave)
            for node in current_wave:
                for dep in dependents.get(node, []):
                    indegree[dep] -= 1
                    if indegree[dep] == 0:
                        ready.append(dep)
            ready = deque(sorted(ready))

        if scheduled != len(steps):
            return None

        step_by_idx = {s["index"]: s for s in steps}
        obs_parts: list[str] = []

        def _eval_call_args(call_node: ast.Call) -> tuple[list[Any], dict[str, Any]] | None:
            args: list[Any] = []
            kwargs: dict[str, Any] = {}
            for arg_node in call_node.args:
                val = _eval_ast_arg(arg_node, self._globals)
                if val is None and not (isinstance(arg_node, ast.Constant) and arg_node.value is None):
                    return None
                args.append(val)
            for kw in call_node.keywords:
                if kw.arg is None:
                    return None
                val = _eval_ast_arg(kw.value, self._globals)
                if val is None and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                    return None
                kwargs[kw.arg] = val
            return args, kwargs

        def _record_result(step: dict[str, Any], val: Any) -> None:
            key = step["target_var"] or step["func_name"]
            if step["target_var"]:
                self._globals[step["target_var"]] = val
            obs_parts.append(f"[{key}]: {val}")

        for wave in waves:
            wave_started = time.perf_counter()
            wave_steps = [step_by_idx[i] for i in sorted(wave)]
            parallel_calls: list[tuple[dict[str, Any], _ParallelCall]] = []
            serial_steps: list[tuple[dict[str, Any], list[Any], dict[str, Any]]] = []

            for s in wave_steps:
                evaluated = _eval_call_args(s["call_node"])
                if evaluated is None:
                    return None
                args, kwargs = evaluated

                should_parallel = bool(s["is_read_only"] or s["parallel_mutation_candidate"])
                if should_parallel:
                    parallel_calls.append(
                        (
                            s,
                            _ParallelCall(
                                func_name=s["func_name"],
                                args=args,
                                kwargs=kwargs,
                                target_var=s["target_var"],
                                index=s["index"],
                            ),
                        )
                    )
                else:
                    serial_steps.append((s, args, kwargs))

            if parallel_calls:
                only_calls = [pc for _, pc in parallel_calls]
                results = execute_parallel_calls(only_calls, self._globals, self._state_lock)
                for s, pc in parallel_calls:
                    key = pc.target_var or f"_result_{pc.index}"
                    val = results.get(key, "")
                    if isinstance(val, str) and val.startswith("[ERROR:"):
                        return ExecutionResult(
                            output="",
                            is_final=False,
                            error=val.removeprefix("[ERROR:").rstrip("]"),
                            elapsed_seconds=time.perf_counter() - start_time,
                        )
                    _record_result(s, val)

            for s, args, kwargs in serial_steps:
                try:
                    val = self._globals[s["func_name"]](*args, **kwargs)
                except Exception as e:
                    return ExecutionResult(
                        output="",
                        is_final=False,
                        error=f"{type(e).__name__}: {e}",
                        elapsed_seconds=time.perf_counter() - start_time,
                    )
                _record_result(s, val)

            wave_elapsed_ms = (time.perf_counter() - wave_started) * 1000.0
            if self._active_tool_chain_meta is not None:
                self._active_tool_chain_meta.setdefault("wave_timeline", []).append(
                    {
                        "wave_index": len(self._active_tool_chain_meta.get("wave_timeline", [])),
                        "tools": [str(s["func_name"]) for s in wave_steps],
                        "mode_used": "dep",
                        "elapsed_ms": round(wave_elapsed_ms, 1),
                        "fallback_to_seq": False,
                        "parallel_mutations_enabled": bool(
                            self._tool_chain_parallel_mutations
                            and any(s["parallel_mutation_candidate"] for s in wave_steps)
                        ),
                    }
                )

        if self._active_tool_chain_meta is not None:
            self._active_tool_chain_meta["mode_used"] = "dep"
            self._active_tool_chain_meta["fallback_to_seq"] = False
            self._active_tool_chain_meta["waves"] = len(waves)
        observation = self._spill_output("Observation:\n" + "\n---\n".join(obs_parts))
        return ExecutionResult(
            output=observation,
            is_final=False,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the sandboxed environment.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        import time

        if self.timed_out:
            return self._timeout_result(time.perf_counter())

        self._execution_count += 1
        start_time = time.perf_counter()
        self._bundle_begin_turn()

        # Structured mode: React-style one-tool-per-turn execution
        if self._structured_mode:
            result = self._execute_structured(code, start_time)
            if self._context_bundle is not None:
                result.output = self._bundle_cap_output(result.output or "")
                if result.error:
                    # exception text reaches the root prompt too: same private cap copy
                    result.error = self._bundle_cap_output(result.error)
            return result

        # Sanitize Unicode characters that models copy from question text
        code = sanitize_code_unicode(code)

        # Validate code for dangerous patterns
        try:
            self._validate_code(code)
        except REPLSecurityError as e:
            self._record_code_log(code, ok=False)
            return ExecutionResult(
                output="",
                is_final=False,
                error=str(e),
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Record the step for the resume code log (D-c1 measurement input).
        # Recording only — the log is NOT injected into any prompt today.
        self._record_code_log(code, ok=True)

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise REPLTimeout(f"Execution timed out after {self.config.timeout_seconds}s")

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_handler = None
        _alarm_set = False
        try:
            # Set timeout (Unix only, main thread only)
            if hasattr(signal, "SIGALRM"):
                try:
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.config.timeout_seconds)
                    _alarm_set = True
                except ValueError:
                    pass  # Not main thread — skip signal-based timeout

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self._globals)

                output = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Redact credentials before they enter model context
                from src.repl_environment.redaction import redact_if_enabled

                output = redact_if_enabled(output)

                # INF-78 OAB-7: per-turn print cap while a context bundle is attached.
                output = self._bundle_cap_output(output)

                # Spill large output to file with summary
                output = self._spill_output(output)

                return ExecutionResult(
                    output=output,
                    is_final=False,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except FinalSignal as e:
                output = self._bundle_cap_output(stdout_capture.getvalue())
                return ExecutionResult(
                    output=output,
                    is_final=True,
                    final_answer=e.answer,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except SyntaxError as e:
                hint = self._tool_hint_if_relevant(code, e)
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=f"SyntaxError: {e}{hint}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except Exception as e:
                if isinstance(e, REPLTimeout):
                    self.mark_timed_out()
                    return self._timeout_result(start_time)
                hint = self._tool_hint_if_relevant(code, e)
                return ExecutionResult(
                    # INF-78 OAB-7: the print cap holds on the error path too -- for the
                    # printed output AND the exception text (`raise ValueError(context[...])`
                    # would otherwise carry a pulled value into the root prompt uncapped).
                    output=self._bundle_cap_output(stdout_capture.getvalue()),
                    is_final=False,
                    error=self._bundle_cap_output(f"{type(e).__name__}: {e}{hint}"),
                    elapsed_seconds=time.perf_counter() - start_time,
                )

        finally:
            # Clear timeout
            if _alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


def create_repl_environment(
    context: str,
    artifacts: dict[str, Any] | None = None,
    config: REPLConfig | None = None,
    llm_primitives: Any | None = None,
    tool_registry: Any | None = None,
    script_registry: Any | None = None,
    role: str | None = None,
    use_restricted_python: bool | None = None,
    progress_logger: ProgressLogger | None = None,
    task_id: str | None = None,
    structured_mode: bool = False,
) -> REPLEnvironment:
    """Factory function to create REPL environment with optional RestrictedPython.

    Args:
        context: The full input context.
        artifacts: Optional pre-existing artifacts.
        config: Optional REPL configuration.
        llm_primitives: Optional LLMPrimitives for llm_call/llm_batch.
        tool_registry: Optional ToolRegistry for TOOL() invocations.
        script_registry: Optional ScriptRegistry for SCRIPT() invocations.
        role: Role name for permission checking.
        use_restricted_python: Override feature flag (None = use flag).
        progress_logger: Optional ProgressLogger for exploration logging.
        task_id: Optional task ID for associating exploration logs.
        structured_mode: If True, enforce one-tool-per-turn React-style execution.

    Returns:
        REPLEnvironment instance (may use RestrictedExecutor internally).
    """
    # Check feature flag if not explicitly set
    if use_restricted_python is None:
        from src.features import features

        use_restricted_python = features().restricted_python

    # Try to use RestrictedPython if requested
    if use_restricted_python:
        try:
            from src.restricted_executor import is_available, RestrictedExecutor  # noqa: F401

            if is_available():
                # Create a wrapped environment that uses RestrictedExecutor
                return _RestrictedREPLEnvironment(
                    context=context,
                    artifacts=artifacts,
                    config=config,
                    llm_primitives=llm_primitives,
                    tool_registry=tool_registry,
                    script_registry=script_registry,
                    role=role,
                    progress_logger=progress_logger,
                    task_id=task_id,
                    structured_mode=structured_mode,
                )
        except ImportError:
            pass  # Fall back to standard REPL

    # Standard REPL environment
    return REPLEnvironment(
        context=context,
        artifacts=artifacts,
        config=config,
        llm_primitives=llm_primitives,
        tool_registry=tool_registry,
        script_registry=script_registry,
        role=role,
        progress_logger=progress_logger,
        task_id=task_id,
        structured_mode=structured_mode,
    )


class _RestrictedREPLEnvironment(REPLEnvironment):
    """REPL environment that uses RestrictedPython for execution.

    This is a wrapper around the standard REPLEnvironment that delegates
    execution to RestrictedExecutor for stronger security guarantees.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with RestrictedExecutor."""
        super().__init__(*args, **kwargs)

        # Import here to avoid issues if not available
        from src.restricted_executor import RestrictedExecutor

        # Create restricted executor with same config
        self._restricted_executor = RestrictedExecutor(
            context=self.context,
            artifacts=self.artifacts,
            timeout_seconds=self.config.timeout_seconds,
            output_cap=self.config.output_cap,
            max_grep_results=self.config.max_grep_results,
            llm_primitives=self.llm_primitives,
        )

    def execute(self, code: str) -> ExecutionResult:
        """Execute using RestrictedPython.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        import time

        if self.timed_out:
            return self._timeout_result(time.perf_counter())

        # Use the restricted executor
        result = self._restricted_executor.execute(code)

        expected_timeout = f"Execution timed out after {self.config.timeout_seconds}s"
        if result.error == expected_timeout:
            self.mark_timed_out()
            result.error = self.timeout_error

        # Redact credentials from output
        output = result.output
        if output:
            from src.repl_environment.redaction import redact_if_enabled

            output = redact_if_enabled(output)

        # Convert to our ExecutionResult type
        return ExecutionResult(
            output=output,
            is_final=result.is_final,
            final_answer=result.final_answer,
            error=result.error,
            elapsed_seconds=result.elapsed_seconds,
        )
