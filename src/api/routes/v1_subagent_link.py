"""HS-19a stage 1 ("Linked") — record the harness subagent tree on ``/v1``.

A harness subagent (OpenCode's ``task`` tool) is just a concurrent ``/v1``
request in a CHILD session. OpenCode stamps the child's own id (``X-Session-Id``
header, and ``x_session_id`` via the plugin) and the parent's id
(``x-parent-session-id`` header, ``session/llm/request.ts:187-201`` at tag
``v1.18.31``). This module turns that into a recorded parent→child edge.

Stage 1 RECORDS ONLY. It never changes model selection, admission or the
response body: the orchestrator owns model selection for harness subagents
(operator, 2026-09-27) and stage 1 leaves scheduling exactly as it is.

Everything here runs only when the ``v1_subagent_link`` feature flag is on.
With it off, the route never calls into this module, so request keys,
inference-tap metadata and the progress log are byte-identical to before.

Resolution (HS-16's header fallback; body always wins):

* session id:  body ``x_session_id`` → ``x-dynamo-session-id`` → ``X-Session-Id``
* parent id:   body ``x_parent_session_id`` → ``x-dynamo-parent-session-id``
  → ``x-parent-session-id``

Refused with 422 (same class as the P0.2 typed-key validation):

* a malformed id or agent name (header or body);
* a parent link with no resolvable child session id;
* a session that names itself as its parent;
* a parent link that would close a cycle in the observed tree;
* a session re-linked to a different parent than the one already recorded.

Depth is derived from the tree this process has observed: a child of a known
node is ``depth(parent) + 1`` (basis ``observed``); a child whose parent was
never seen is recorded at depth 1 (basis ``parent_unseen``, a lower bound).
Entries expire after an idle TTL (HS-16: a predeclared constant until HSF-3
measures the harness-class p99).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Same shape as the P0.2 typed keys (src/api/models/openai.py).
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+/=-]*$")
_ID_MAX_LEN = 128
_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AGENT_NAME_MAX_LEN = 64

# Header precedence, highest first (HS-16). Starlette headers are case-insensitive.
SESSION_HEADERS = ("x-dynamo-session-id", "x-session-id")
PARENT_HEADERS = ("x-dynamo-parent-session-id", "x-parent-session-id")

# HS-16 idle-retention TTL: a predeclared constant until HSF-3 measures the
# harness-class p99 session idle gap.
SUBAGENT_LINK_IDLE_TTL_S = 3600.0
SUBAGENT_LINK_MAX_ENTRIES = 10_000
_MAX_CHAIN_WALK = 64

SESSION_LOG_KIND = "harness_subagent_link"


@dataclass(frozen=True)
class SubagentLink:
    """The resolved identity of one ``/v1`` request (flag on)."""

    session_id: str | None
    session_id_source: str | None  # "body" | "header:<name>" | None
    session_id_mismatch: bool
    parent_session_id: str | None
    parent_session_id_source: str | None
    parent_session_id_mismatch: bool
    agent_name: str | None
    depth: int | None = None
    depth_basis: str | None = None  # "observed" | "parent_unseen"
    newly_linked: bool = False

    @property
    def is_child(self) -> bool:
        return self.parent_session_id is not None

    def request_keys(self, base: Mapping[str, Any]) -> dict[str, Any]:
        """``base`` plus the link keys. Root requests whose identity came from
        the body get ``base`` back unchanged (golden-pinned)."""
        keys: dict[str, Any] = dict(base)
        if self.session_id is not None and self.session_id_source != "body":
            keys["x_session_id"] = self.session_id
            keys["session_id_source"] = self.session_id_source
        if self.session_id_mismatch:
            keys["session_id_mismatch"] = True
        if self.agent_name is not None:
            keys["x_agent_name"] = self.agent_name
        if self.is_child:
            keys["parent_session_id"] = self.parent_session_id
            keys["parent_session_id_source"] = self.parent_session_id_source
            if self.parent_session_id_mismatch:
                keys["parent_session_id_mismatch"] = True
            keys["subagent_depth"] = self.depth
            keys["subagent_depth_basis"] = self.depth_basis
        return keys


class _Node:
    __slots__ = ("parent", "depth", "basis", "last_seen")

    def __init__(self, parent: str | None, depth: int, last_seen: float) -> None:
        self.parent = parent
        self.depth = depth
        self.basis: str | None = None
        self.last_seen = last_seen


class SubagentTreeRegistry:
    """Bounded, TTL-expiring in-process record of the observed session tree."""

    def __init__(
        self,
        *,
        ttl_s: float = SUBAGENT_LINK_IDLE_TTL_S,
        max_entries: int = SUBAGENT_LINK_MAX_ENTRIES,
        clock=time.monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._nodes: OrderedDict[str, _Node] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    def _expire(self, now: float) -> None:
        # OrderedDict is kept in last-seen order, so expiry stops at the first live node.
        while self._nodes:
            sid, node = next(iter(self._nodes.items()))
            if now - node.last_seen <= self._ttl_s:
                break
            del self._nodes[sid]
        while len(self._nodes) > self._max_entries:
            self._nodes.popitem(last=False)

    def _touch(self, sid: str, node: _Node, now: float) -> None:
        node.last_seen = now
        self._nodes[sid] = node
        self._nodes.move_to_end(sid)
        while len(self._nodes) > self._max_entries:
            self._nodes.popitem(last=False)

    def observe(self, session_id: str, parent_session_id: str | None) -> tuple[int, str | None, bool]:
        """Record one request. Returns ``(depth, depth_basis, newly_linked)``.

        Raises ``ValueError`` for a cycle or a conflicting re-link.
        """
        now = self._clock()
        with self._lock:
            self._expire(now)
            existing = self._nodes.get(session_id)
            if parent_session_id is None:
                if existing is None:
                    self._touch(session_id, _Node(None, 0, now), now)
                    return 0, None, False
                self._touch(session_id, existing, now)
                return existing.depth, None, False

            if existing is not None and existing.parent is not None:
                if existing.parent != parent_session_id:
                    raise ValueError(
                        f"session {session_id!r} is already linked to parent "
                        f"{existing.parent!r}; refusing a re-link to {parent_session_id!r}"
                    )
                self._touch(session_id, existing, now)
                return existing.depth, existing.basis, False

            # New edge (or a known root gaining its parent). Refuse a cycle.
            cursor: str | None = parent_session_id
            for _ in range(_MAX_CHAIN_WALK):
                if cursor is None:
                    break
                if cursor == session_id:
                    raise ValueError(
                        f"parent link {session_id!r} -> {parent_session_id!r} would close a cycle"
                    )
                node = self._nodes.get(cursor)
                cursor = node.parent if node is not None else None

            parent = self._nodes.get(parent_session_id)
            if parent is not None:
                depth, basis = parent.depth + 1, "observed"
                self._touch(parent_session_id, parent, now)
            else:
                depth, basis = 1, "parent_unseen"
            node = existing or _Node(None, 0, now)
            node.parent = parent_session_id
            node.depth = depth
            node.basis = basis
            self._touch(session_id, node, now)
            return depth, basis, True


_registry = SubagentTreeRegistry()


def get_registry() -> SubagentTreeRegistry:
    return _registry


def reset_registry() -> None:
    """Test hook: forget every observed session."""
    global _registry
    _registry = SubagentTreeRegistry()


def _refuse(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=f"{detail} (HS-19a v1_subagent_link)")


def _valid_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _ID_MAX_LEN or not _ID_RE.match(value):
        raise _refuse(
            f"{where} must be 1-{_ID_MAX_LEN} characters matching {_ID_RE.pattern}"
        )
    return value


def _first_header(headers: Mapping[str, str], names: tuple[str, ...]) -> tuple[str | None, str | None]:
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None:
            return _valid_id(value.strip(), f"header {name}"), f"header:{name}"
    return None, None


def _pick(
    body_value: str | None,
    headers: Mapping[str, str],
    names: tuple[str, ...],
    body_name: str,
) -> tuple[str | None, str | None, bool]:
    """(value, source, mismatch). The body wins; a differing header is flagged."""
    header_value, header_source = _first_header(headers, names)
    if body_value is not None:
        body_value = _valid_id(body_value, body_name)
        mismatch = header_value is not None and header_value != body_value
        if mismatch:
            logger.warning(
                "HS-19a: %s=%r disagrees with %s=%r; the body wins",
                body_name,
                body_value,
                header_source,
                header_value,
            )
        return body_value, "body", mismatch
    return header_value, header_source, False


def resolve_subagent_link(
    *,
    body_session_id: str | None,
    body_parent_session_id: str | None,
    body_agent_name: str | None,
    headers: Mapping[str, str],
    registry: SubagentTreeRegistry | None = None,
) -> SubagentLink:
    """Resolve and record one request's place in the subagent tree (flag on).

    Raises ``HTTPException(422)`` for malformed or spoofed values.
    """
    session_id, session_source, session_mismatch = _pick(
        body_session_id, headers, SESSION_HEADERS, "x_session_id"
    )
    parent_id, parent_source, parent_mismatch = _pick(
        body_parent_session_id, headers, PARENT_HEADERS, "x_parent_session_id"
    )
    agent_name: str | None = None
    if body_agent_name is not None:
        if (
            not isinstance(body_agent_name, str)
            or not body_agent_name
            or len(body_agent_name) > _AGENT_NAME_MAX_LEN
            or not _AGENT_NAME_RE.match(body_agent_name)
        ):
            raise _refuse(
                f"x_agent_name must be 1-{_AGENT_NAME_MAX_LEN} characters matching "
                f"{_AGENT_NAME_RE.pattern}"
            )
        agent_name = body_agent_name

    if parent_id is not None:
        if session_id is None:
            raise _refuse(
                "a parent session link needs the child's own session id "
                "(x_session_id, x-dynamo-session-id or X-Session-Id)"
            )
        if parent_id == session_id:
            raise _refuse(f"session {session_id!r} names itself as its parent")

    depth: int | None = None
    basis: str | None = None
    newly_linked = False
    if session_id is not None:
        reg = registry if registry is not None else get_registry()
        try:
            depth, basis, newly_linked = reg.observe(session_id, parent_id)
        except ValueError as exc:
            raise _refuse(str(exc)) from exc

    return SubagentLink(
        session_id=session_id,
        session_id_source=session_source,
        session_id_mismatch=session_mismatch,
        parent_session_id=parent_id,
        parent_session_id_source=parent_source,
        parent_session_id_mismatch=parent_mismatch,
        agent_name=agent_name,
        depth=depth if parent_id is not None else None,
        depth_basis=basis if parent_id is not None else None,
        newly_linked=newly_linked,
    )


def log_subagent_link(
    progress_logger: Any,
    link: SubagentLink,
    *,
    chat_id: str,
    user_id: str | None,
) -> bool:
    """Append one session-log row for a NEWLY linked child. Fail-silent.

    The row goes to the orchestrator's progress JSONL as a ``session_created``
    event with ``data.kind == "harness_subagent_link"``, so the parent→child
    tree can be rebuilt from the log alone (stage-2 training/eval data).
    """
    if progress_logger is None or not link.is_child or not link.newly_linked:
        return False
    try:
        from orchestration.repl_memory.progress_logger import EventType, ProgressEntry

        data: dict[str, Any] = {
            "kind": SESSION_LOG_KIND,
            "session_id": link.session_id,
            "parent_session_id": link.parent_session_id,
            "parent_session_id_source": link.parent_session_id_source,
            "subagent_depth": link.depth,
            "subagent_depth_basis": link.depth_basis,
            "name": link.agent_name,
            "project": None,
            "user_id": user_id,
        }
        if link.session_id_source != "body":
            data["session_id_source"] = link.session_id_source
        entry = ProgressEntry(event_type=EventType.SESSION_CREATED, task_id=chat_id, data=data)
        # A lineage row is a durable record read back by other processes (the
        # HS-19a acceptance S6, stage-2 tree rebuilds): write it through now
        # instead of leaving it in this worker's batch buffer (buffer_size=10,
        # one buffer per uvicorn worker) until later traffic or shutdown.
        log_durable = getattr(progress_logger, "log_durable", None)
        if callable(log_durable):
            log_durable(entry)
        else:
            progress_logger.log(entry)
            flush = getattr(progress_logger, "flush", None)
            if callable(flush):
                flush()
        return True
    except Exception:
        logger.debug("HS-19a subagent-link session log failed", exc_info=True)
        return False
