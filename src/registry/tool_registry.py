#!/usr/bin/env python3
"""Tool Registry for orchestrator models.

This module provides a registry for tools that orchestrator models can invoke.
Tools are registered with standardized interfaces and role-based permissions.

Design principles:
- MCP-first architecture (can integrate with MCP servers)
- Role-based access control (Tier A/B have web access, Tier C doesn't)
- Tool invocations are logged and validated

Usage:
    from src.tool_registry import ToolRegistry, ToolPermissions

    registry = ToolRegistry()
    registry.load_from_yaml("orchestration/tool_registry.yaml")

    # Check if role can use a tool
    if registry.can_use_tool("frontdoor", "fetch_docs"):
        result = registry.invoke("fetch_docs", url="https://docs.python.org")
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

#: Hard cap on the registry's diagnostic invocation ring.
#:
#: The registry is ONE process-global object (``src/api/__init__.py`` builds a
#: single ``state.tool_registry``), so this log is shared by every concurrently
#: served request. Before 2026-09-14 it was an unbounded ``list`` that nothing
#: ever cleared: it grew for the lifetime of the uvicorn process, and per-request
#: readers picked up other requests' calls (INV-LOG defect, RTG-02).
#:
#: Per-request telemetry now comes from ``REPLEnvironment._invoked_tools``, which
#: is request-scoped by construction. What remains here is a *diagnostic* ring
#: for tests and interactive inspection, so it is bounded: oldest entries are
#: evicted once the cap is reached, which makes the memory footprint O(cap)
#: regardless of process uptime. Override with
#: ``ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX`` (<= 0 disables logging entirely).
INVOCATION_LOG_MAX_DEFAULT = 1000


def _invocation_log_max() -> int:
    """Resolve the invocation-ring cap, tolerating a malformed override."""
    raw = os.environ.get("ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX")
    if raw is None or raw.strip() == "":
        return INVOCATION_LOG_MAX_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring malformed ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX=%r; using %d",
            raw,
            INVOCATION_LOG_MAX_DEFAULT,
        )
        return INVOCATION_LOG_MAX_DEFAULT
    return max(0, value)


class ToolCategory(str, Enum):
    """Categories of tools available to models."""

    WEB = "web"
    FILE = "file"
    CODE = "code"
    DATA = "data"
    SYSTEM = "system"
    MATH = "math"
    LLM = "llm"
    SPECIALIZED = "specialized"


class SideEffect(str, Enum):
    """Declared side effects for tools (Lobster pattern).

    Enables the graph to reason about tool safety without executing.
    Only active when features().side_effect_tracking is True.
    """

    LOCAL_EXEC = "local_exec"
    CALLS_LLM = "calls_llm"
    MODIFIES_FILES = "modifies_files"
    NETWORK_ACCESS = "network_access"
    SYSTEM_STATE = "system_state"
    READ_ONLY = "read_only"


@dataclass
class ToolPermissions:
    """Permissions for a specific role's tool access."""

    web_access: bool = False
    allowed_categories: list[ToolCategory] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)

    def can_use_tool(self, tool: "Tool") -> bool:
        """Check if this permission set allows using a tool.

        Args:
            tool: The tool to check access for.

        Returns:
            True if the tool can be used under these permissions.
        """
        # Check forbidden list first
        if tool.name in self.forbidden_tools:
            return False

        # Check explicit allow list
        if self.allowed_tools and tool.name in self.allowed_tools:
            return True

        # Check category
        if tool.category in self.allowed_categories:
            # Web tools require web_access
            if tool.category == ToolCategory.WEB and not self.web_access:
                return False
            return True

        return False


@dataclass
class Tool:
    """A registered tool that models can invoke."""

    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, dict[str, Any]]  # name -> {type, description, required}
    handler: Callable[..., Any] | None = None
    mcp_server: str | None = None  # MCP server identifier if MCP-backed
    code_hash: str | None = None  # SHA256 of handler code for integrity
    side_effects: list[str] = field(default_factory=list)  # SideEffect values
    destructive: bool = False  # Whether tool modifies state irreversibly
    allowed_callers: list[str] = field(default_factory=lambda: ["direct"])

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        """Validate arguments against parameter schema.

        Args:
            args: Arguments to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check required parameters
        for param_name, param_spec in self.parameters.items():
            if param_spec.get("required", False) and param_name not in args:
                errors.append(f"Missing required parameter: {param_name}")

        # Check types (basic validation)
        for arg_name, arg_value in args.items():
            if arg_name not in self.parameters:
                errors.append(f"Unknown parameter: {arg_name}")
                continue

            expected_type = self.parameters[arg_name].get("type", "string")
            if expected_type == "string" and not isinstance(arg_value, str):
                errors.append(
                    f"Parameter {arg_name} must be string, got {type(arg_value).__name__}"
                )
            elif expected_type == "integer" and not isinstance(arg_value, int):
                errors.append(
                    f"Parameter {arg_name} must be integer, got {type(arg_value).__name__}"
                )
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                errors.append(
                    f"Parameter {arg_name} must be boolean, got {type(arg_value).__name__}"
                )
            elif expected_type == "array" and not isinstance(arg_value, list):
                errors.append(f"Parameter {arg_name} must be array, got {type(arg_value).__name__}")

        return errors


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""

    tool_name: str
    args: dict[str, Any]
    role: str
    success: bool
    result: Any
    error: str | None = None
    elapsed_ms: float = 0.0
    caller_type: str = "direct"
    chain_id: str | None = None
    chain_index: int = 0


@dataclass
class ToolOutput:
    """Structured envelope for tool results (Lobster pattern).

    Provides dual output modes: human-readable and machine-parseable
    from the same invocation. Foundation for halt-and-resume protocol.
    Only used when features().structured_tool_output is True.
    """

    protocol_version: int = 1
    ok: bool = True
    status: str = "success"  # "success" | "error" | "pending_approval"
    output: Any = None
    side_effects_declared: list[str] = field(default_factory=list)
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_human(self) -> str:
        """Human-readable representation."""
        if not self.ok:
            return f"[ERROR] {self.output}"
        if self.status == "pending_approval":
            effects = ", ".join(self.side_effects_declared) or "none"
            return f"[PENDING APPROVAL] Side effects: {effects}"
        text = str(self.output) if self.output is not None else ""
        if text:
            from src.repl_environment.redaction import redact_if_enabled

            text = redact_if_enabled(text)
        return text

    def to_machine(self) -> dict[str, Any]:
        """Machine-parseable representation."""
        output = self.output
        if isinstance(output, str) and output:
            from src.repl_environment.redaction import redact_if_enabled

            output = redact_if_enabled(output)
        return {
            "protocol_version": self.protocol_version,
            "ok": self.ok,
            "status": self.status,
            "output": output,
            "side_effects_declared": self.side_effects_declared,
            "requires_approval": self.requires_approval,
            "metadata": self.metadata,
        }


class ToolRegistry:
    """Registry of tools available to orchestrator models.

    Manages tool registration, permission checking, and invocation.
    Supports both native Python handlers and MCP server backends.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}
        self._permissions: dict[str, ToolPermissions] = {}
        # Bounded diagnostic ring, NOT a per-request telemetry source. See
        # INVOCATION_LOG_MAX_DEFAULT: this registry is process-global, so anything
        # read out of here is a mix of every in-flight request's calls. Request
        # telemetry lives on REPLEnvironment._invoked_tools.
        self._invocation_log: deque[ToolInvocation] = deque(
            maxlen=_invocation_log_max() or 1,
        )
        # deque.append is thread-safe on CPython, but iterating a deque while
        # another thread mutates it is not a stable snapshot operation. Keep
        # append, snapshot, and clear under one lock; tool dispatch stays outside.
        self._invocation_log_lock = Lock()
        self._invocation_log_enabled: bool = _invocation_log_max() > 0
        self._mcp_configs: dict[str, Any] | None = None
        # Cascading tool policy (used when features().cascading_tool_policy is True)
        self._global_policies: list = []
        self._role_policies: dict[str, list] = {}

    def register_tool(self, tool: Tool, update: bool = False) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool to register.
            update: If True, update existing tool instead of raising.

        Raises:
            ValueError: If tool with same name already exists and update=False.
        """
        if tool.name in self._tools:
            if update:
                logger.debug(f"Updating existing tool: {tool.name}")
            else:
                raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")

    def register_handler(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        parameters: dict[str, dict[str, Any]],
        side_effects: list[str] | None = None,
        destructive: bool = False,
        allowed_callers: list[str] | None = None,
        update: bool = True,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a function as a tool handler.

        Args:
            name: Tool name.
            description: Tool description.
            category: Tool category.
            parameters: Parameter definitions.
            update: If True, update existing tool (default). Useful when
                programmatic handlers should override YAML-loaded stubs.

        Usage:
            @registry.register_handler(
                name="fetch_docs",
                description="Fetch documentation from URL",
                category=ToolCategory.WEB,
                parameters={"url": {"type": "string", "required": True}}
            )
            def fetch_docs(url: str) -> str:
                ...
        """

        def decorator(func: Callable) -> Callable:
            # Compute code hash for integrity
            import inspect

            source = inspect.getsource(func)
            code_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

            tool = Tool(
                name=name,
                description=description,
                category=category,
                parameters=parameters,
                handler=func,
                code_hash=code_hash,
                side_effects=list(side_effects or []),
                destructive=destructive,
                allowed_callers=list(allowed_callers or ["direct"]),
            )
            self.register_tool(tool, update=update)
            return func

        return decorator

    def set_role_permissions(self, role: str, permissions: ToolPermissions) -> None:
        """Set permissions for a role.

        Args:
            role: Role name (e.g., "frontdoor", "coder_escalation", "worker_general").
            permissions: Permission configuration for this role.
        """
        self._permissions[role] = permissions
        logger.debug(f"Set permissions for role: {role}")

    def load_permissions_from_registry(self, registry_path: str | Path) -> None:
        """Load role permissions from model_registry.yaml.

        Args:
            registry_path: Path to model_registry.yaml.
        """
        path = Path(registry_path)
        if not path.exists():
            logger.warning(f"Registry file not found: {path}")
            return

        with open(path) as f:
            data = yaml.safe_load(f)

        roles = data.get("roles", {})
        for role_name, role_config in roles.items():
            perms = role_config.get("tool_permissions", {})
            if perms:
                permissions = ToolPermissions(
                    web_access=perms.get("web_access", False),
                    allowed_categories=[
                        ToolCategory(c) for c in perms.get("allowed_categories", [])
                    ],
                    allowed_tools=perms.get("allowed_tools", []),
                    forbidden_tools=perms.get("forbidden_tools", []),
                )
                self.set_role_permissions(role_name, permissions)

    def add_global_policy(self, layer: Any) -> None:
        """Add a global policy layer (applies to all roles).

        Args:
            layer: PolicyLayer to add to the global chain.
        """
        self._global_policies.append(layer)

    def add_role_policy(self, role: str, layer: Any) -> None:
        """Add a role-specific policy layer.

        Args:
            role: Role name.
            layer: PolicyLayer to add.
        """
        if role not in self._role_policies:
            self._role_policies[role] = []
        self._role_policies[role].append(layer)

    def can_use_tool(
        self,
        role: str,
        tool_name: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if a role can use a specific tool.

        When features().cascading_tool_policy is enabled, resolves access
        through the policy chain: global → role → context layers.
        Otherwise falls back to legacy ToolPermissions.can_use_tool().

        Args:
            role: Role name.
            tool_name: Tool name.
            context: Optional task-level constraints (e.g. {"read_only": True}).

        Returns:
            True if the role can use the tool.
        """
        if tool_name not in self._tools:
            return False

        from src.features import features as _get_features

        if _get_features().cascading_tool_policy:
            return self._can_use_tool_cascading(role, tool_name, context)

        # Legacy path
        if role not in self._permissions:
            logger.warning(f"Unknown role: {role}, denying access")
            return False

        return self._permissions[role].can_use_tool(self._tools[tool_name])

    def _can_use_tool_cascading(
        self,
        role: str,
        tool_name: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Resolve tool access through cascading policy chain."""
        from src.tool_policy import (
            PolicyLayer,
            permissions_to_policy,
            resolve_policy_chain,
        )

        chain: list[PolicyLayer] = []

        # Layer 1: Global policies
        chain.extend(self._global_policies)

        # Layer 2: Role policies (from explicit policy layers OR adapted from ToolPermissions)
        if role in self._role_policies:
            chain.extend(self._role_policies[role])
        elif role in self._permissions:
            # Adapt legacy ToolPermissions to a PolicyLayer
            chain.append(
                permissions_to_policy(f"role:{role}", self._permissions[role], self._tools)
            )

        # Layer 3: Task-level constraints from context
        if context:
            if context.get("read_only"):
                chain.append(PolicyLayer(name="task:read_only", deny=frozenset({"group:write"})))
            if context.get("no_web"):
                chain.append(PolicyLayer(name="task:no_web", deny=frozenset({"group:web"})))

        all_tools = frozenset(self._tools.keys())
        allowed = resolve_policy_chain(chain, all_tools)
        return tool_name in allowed

    def invoke(
        self,
        tool_name: str,
        role: str,
        caller_type: str = "direct",
        chain_id: str | None = None,
        chain_index: int = 0,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a tool with the given arguments.

        Args:
            tool_name: Name of the tool to invoke.
            role: Role making the invocation (for permission check).
            context: Optional task-level constraints for cascading policy.
            **kwargs: Tool arguments.

        Returns:
            Tool result.

        Raises:
            PermissionError: If role cannot use this tool.
            ValueError: If tool doesn't exist or args are invalid.
            RuntimeError: If tool execution fails.
        """
        import time

        start = time.perf_counter()

        # Check tool exists
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]

        # Check permissions
        if not self.can_use_tool(role, tool_name, context=context):
            raise PermissionError(f"Role '{role}' cannot use tool '{tool_name}'")

        # Validate arguments
        errors = tool.validate_args(kwargs)
        if errors:
            raise ValueError(f"Invalid arguments: {'; '.join(errors)}")

        # Check if structured output is enabled
        from src.features import features as _get_features

        use_structured = _get_features().structured_tool_output

        # AP-54 eval knowledge fence. A no-op unless the request carried
        # `eval_fence`; production chat never installs a carrier. A denial is an
        # ordinary failed tool result, never an exception, so the rollout goes on.
        from src.repl_environment import knowledge_fence as _fence

        _fence_denial = _fence.check_tool_call(tool_name, kwargs)
        _denial_key = "eval_fence"
        if _fence_denial is None:
            # INF-78 OAB-1: a task_root-scoped /chat request runs only read-only tools, with
            # path arguments confined to task_root + read_roots. No-op without a scope.
            from src.repl_environment.task_root import check_registry_tool_scope

            _fence_denial = check_registry_tool_scope(tool_name, kwargs, tool.side_effects)
            _denial_key = "task_scope"
        if _fence_denial is not None:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_invocation(
                ToolInvocation(
                    tool_name=tool_name,
                    args=kwargs,
                    role=role,
                    success=False,
                    result=None,
                    error=_fence_denial,
                    elapsed_ms=elapsed,
                    caller_type=caller_type,
                    chain_id=chain_id,
                    chain_index=chain_index,
                )
            )
            if use_structured:
                return ToolOutput(
                    ok=False,
                    status="error",
                    output=_fence_denial,
                    side_effects_declared=tool.side_effects,
                    metadata={"elapsed_ms": elapsed, _denial_key: "denied"},
                )
            return {"success": False, "error": _fence_denial, _denial_key: "denied"}

        # Check approval requirement for destructive tools
        if use_structured and tool.destructive and _get_features().side_effect_tracking:
            return ToolOutput(
                ok=True,
                status="pending_approval",
                output=None,
                side_effects_declared=tool.side_effects,
                requires_approval=True,
                metadata={"tool_name": tool_name, "args": kwargs},
            )

        # Execute
        try:
            if tool.handler is not None:
                result = tool.handler(**kwargs)
            elif tool.mcp_server is not None:
                result = self._invoke_mcp(tool.mcp_server, tool_name, kwargs)
            else:
                raise RuntimeError(f"Tool '{tool_name}' has no handler")

            # Redact credentials from string results
            if isinstance(result, str):
                from src.repl_environment.redaction import redact_if_enabled

                result = redact_if_enabled(result)

            elapsed = (time.perf_counter() - start) * 1000

            # Log invocation
            self._record_invocation(
                ToolInvocation(
                    tool_name=tool_name,
                    args=kwargs,
                    role=role,
                    success=True,
                    result=result,
                    elapsed_ms=elapsed,
                    caller_type=caller_type,
                    chain_id=chain_id,
                    chain_index=chain_index,
                )
            )

            if use_structured:
                return ToolOutput(
                    ok=True,
                    status="success",
                    output=result,
                    side_effects_declared=tool.side_effects,
                    metadata={"elapsed_ms": elapsed},
                )
            return result

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000

            self._record_invocation(
                ToolInvocation(
                    tool_name=tool_name,
                    args=kwargs,
                    role=role,
                    success=False,
                    result=None,
                    error=str(e),
                    elapsed_ms=elapsed,
                    caller_type=caller_type,
                    chain_id=chain_id,
                    chain_index=chain_index,
                )
            )

            if use_structured:
                return ToolOutput(
                    ok=False,
                    status="error",
                    output=str(e),
                    side_effects_declared=tool.side_effects,
                    metadata={"elapsed_ms": elapsed},
                )
            raise RuntimeError(f"Tool execution failed: {e}") from e

    def _invoke_mcp(
        self,
        server: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> Any:
        """Invoke a tool via MCP server.

        Args:
            server: MCP server identifier.
            tool_name: Tool name on the server.
            args: Tool arguments.

        Returns:
            MCP tool result as text string.

        Raises:
            RuntimeError: If server is unknown or tool call fails.
        """
        from src.mcp_client import call_mcp_tool, load_server_configs

        if self._mcp_configs is None:
            # 2026-08-01: was `Path(__file__).parent.parent`, which from
            # src/registry/ resolves to `src/` — `src/orchestration/mcp_servers.yaml`
            # has never existed, so every MCP tool call through this path failed on
            # a missing file rather than on anything real. Same defect as
            # script_registry.py:435 and the registry_loader defaults.
            config_path = Path(__file__).resolve().parents[2] / "orchestration" / "mcp_servers.yaml"
            self._mcp_configs = load_server_configs(config_path)

        if server not in self._mcp_configs:
            raise RuntimeError(
                f"Unknown MCP server: {server}. Configure in orchestration/mcp_servers.yaml"
            )

        return call_mcp_tool(self._mcp_configs[server], tool_name, args)

    def list_tools(
        self, role: str | None = None, context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """List available tools, optionally filtered by role permissions.

        Args:
            role: Optional role to filter by permissions.
            context: Optional task-level constraints for cascading policy.

        Returns:
            List of tool info dicts.
        """
        result = []
        for tool in self._tools.values():
            if role is None or self.can_use_tool(role, tool.name, context=context):
                info: dict[str, Any] = {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "parameters": tool.parameters,
                    "mcp_backed": tool.mcp_server is not None,
                }
                if tool.side_effects:
                    info["side_effects"] = tool.side_effects
                if tool.destructive:
                    info["destructive"] = True
                result.append(info)
        return result

    def generate_gbnf_grammar(self, role: str | None = None) -> str:
        """Generate GBNF grammar for valid tool call syntax.

        Constrains model output to valid TOOL() calls for the given role's
        permitted tools. Used with llama-server's --grammar for structured mode.

        Args:
            role: Optional role to filter available tools.

        Returns:
            GBNF grammar string.
        """
        tools = self.list_tools(role)
        tool_names = [t["name"] for t in tools]

        if not tool_names:
            # Fallback: allow any string if no tools
            return 'root ::= [^\\x00]+'

        # Build tool-name alternation
        name_alts = " | ".join(f'"{name}"' for name in tool_names)

        grammar = f"""\
root ::= thought action
thought ::= "# " [^\\n]+ "\\n"
action ::= tool-call | final-call | code-line
tool-call ::= "TOOL(\\"" tool-name "\\"" args ")\\n"
tool-name ::= {name_alts}
args ::= ("," ws arg)*
arg ::= [a-zA-Z_]+ "=" value
value ::= quoted-string | number | "True" | "False" | "None"
quoted-string ::= "\\"" [^\\"]* "\\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
final-call ::= "FINAL(" value ")\\n"
code-line ::= [^\\n]+ "\\n"
ws ::= " "*
"""
        return grammar

    def get_read_only_tools(self) -> set[str]:
        """Return set of tool names with READ_ONLY side effect.

        Used by parallel tool execution to determine which tools
        can safely run concurrently.
        """
        return {
            tool.name
            for tool in self._tools.values()
            if SideEffect.READ_ONLY in tool.side_effects
        }

    def get_chainable_tools(self) -> set[str]:
        """Return set of tool names that opt in to chained execution."""
        return {
            tool.name
            for tool in self._tools.values()
            if "chain" in tool.allowed_callers
        }

    def _record_invocation(self, invocation: ToolInvocation) -> None:
        """Append one record to the bounded diagnostic ring.

        A no-op when the ring is disabled (``ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX``
        <= 0). Never raises: telemetry must not be the reason a tool call fails.
        """
        if not self._invocation_log_enabled:
            return
        with self._invocation_log_lock:
            self._invocation_log.append(invocation)

    def get_invocation_log(self) -> list[ToolInvocation]:
        """Get a snapshot of the bounded diagnostic invocation ring.

        NOT a per-request telemetry source. This registry is process-global (one
        ``state.tool_registry`` per API process), so under concurrency the ring
        interleaves every in-flight request's calls and retains at most
        ``ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX`` of them. Durable records and
        per-request tool telemetry MUST read ``REPLEnvironment._invoked_tools``,
        which is request-scoped by construction (see
        ``src/repl_environment/context.py::_invoke_tool``).

        Legitimate uses: tests, interactive debugging, and process-level
        diagnostics that deliberately want the whole-process view.
        """
        with self._invocation_log_lock:
            return list(self._invocation_log)

    def clear_invocation_log(self) -> None:
        """Clear the diagnostic invocation ring.

        A process-wide reset intended for tests and interactive diagnostics. It is
        deliberately NOT a per-request mechanism: clearing a shared log to scope
        one request's telemetry is racy under concurrency (another request's calls
        land between the clear and the read). Use ``_invoked_tools`` for that.
        """
        with self._invocation_log_lock:
            self._invocation_log.clear()


# Default global registry
_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get or create the default tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def load_from_yaml(
    registry: ToolRegistry,
    yaml_path: str | Path,
) -> int:
    """Load tools from a YAML registry file.

    This bridges the YAML-based tool definitions in orchestration/tool_registry.yaml
    with the programmatic ToolRegistry. Each tool's implementation is dynamically
    imported and wrapped as a handler.

    Args:
        registry: ToolRegistry to load tools into.
        yaml_path: Path to tool_registry.yaml.

    Returns:
        Number of tools successfully loaded.

    Example:
        registry = ToolRegistry()
        loaded = load_from_yaml(registry, "orchestration/tool_registry.yaml")
        print(f"Loaded {loaded} tools")
    """
    import importlib

    path = Path(yaml_path)
    if not path.exists():
        logger.warning(f"Tool registry YAML not found: {path}")
        return 0

    with open(path) as f:
        data = yaml.safe_load(f)

    tools_data = data.get("tools", {})
    loaded_count = 0

    for tool_name, tool_spec in tools_data.items():
        try:
            # Parse category
            category_str = tool_spec.get("category", "specialized")
            try:
                category = ToolCategory(category_str)
            except ValueError:
                category = ToolCategory.SPECIALIZED

            # Parse parameters
            params = {}
            for param_name, param_spec in tool_spec.get("parameters", {}).items():
                params[param_name] = {
                    "type": param_spec.get("type", "string"),
                    "required": param_spec.get("required", False),
                    "description": param_spec.get("description", ""),
                }
                if "default" in param_spec:
                    params[param_name]["default"] = param_spec["default"]

            # Load handler function dynamically
            impl = tool_spec.get("implementation", {})
            handler = None

            if impl.get("type") == "python":
                module_name = impl.get("module")
                func_name = impl.get("function")

                if module_name and func_name:
                    try:
                        module = importlib.import_module(module_name)
                        handler = getattr(module, func_name)
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Could not load handler for tool '{tool_name}': {e}")

                        # Create a stub handler that returns an error
                        def make_stub(name: str, error: str):
                            def stub(**kwargs):
                                return {"error": f"Tool '{name}' handler not available: {error}"}

                            return stub

                        handler = make_stub(tool_name, str(e))

            # Parse side effects
            raw_effects = tool_spec.get("side_effects", [])
            side_effects = []
            for eff in raw_effects:
                try:
                    SideEffect(eff)  # validate
                    side_effects.append(eff)
                except ValueError:
                    logger.warning(f"Unknown side effect '{eff}' for tool '{tool_name}'")

            allowed_callers = tool_spec.get("allowed_callers", ["direct"])
            if not isinstance(allowed_callers, list):
                logger.warning(
                    "Invalid allowed_callers for tool '%s' (expected list), defaulting to ['direct']",
                    tool_name,
                )
                allowed_callers = ["direct"]

            # Create Tool object
            tool = Tool(
                name=tool_name,
                description=tool_spec.get("description", ""),
                category=category,
                parameters=params,
                handler=handler,
                mcp_server=tool_spec.get("mcp_server"),
                side_effects=side_effects,
                destructive=tool_spec.get("destructive", False),
                allowed_callers=allowed_callers,
            )

            registry.register_tool(tool)
            loaded_count += 1
            logger.debug(f"Loaded tool from YAML: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to load tool '{tool_name}': {e}")
            continue

    logger.info(f"Loaded {loaded_count}/{len(tools_data)} tools from {path}")
    return loaded_count
