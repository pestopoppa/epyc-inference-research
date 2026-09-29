"""Single source of truth for which orchestrator roles route through ``/v1/chat/completions``
(server-side jinja templating + thinking-off) vs ``/completion``.

The backend router (`src/llm_primitives/backend.py`) and the chat route's orchestrator-side
template-SKIP logic (`src/api/routes/chat.py`) MUST agree on this set: if a role routes to
chat-completions (server applies the GGUF jinja template) but the chat route still applies an
orchestrator-side template, the role is DOUBLE-templated; if the inverse, it is un-templated.
The live default is derived from generated stack priors so stack swaps do not require code edits.
If the priors artifact is missing or malformed, we fall back to a narrow degraded set instead of
the full historical literal table.

Read live from ``ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES`` so an A/B can flip it across restarts.

RI-23 (OP-69 (a), 2026-09-29): with the ``thinking_roles_chat_lane`` feature flag ON, the
THINKING-ON roles (``--jinja`` AND ``acceleration.enable_thinking is True`` in the live stack
priors) join the chat lane too, so their chat template and ``chat_template_kwargs``
(``enable_thinking``, ``reasoning_effort``) take effect and llama-server splits the reasoning
into ``reasoning_content``. Those roles are NOT baked into a backend's ``ServerConfig`` at
startup (``static_chat_completions_roles``); the backend admits them per request
(``thinking_chat_lane_role``) and the template skip reads ``chat_completions_roles`` — both
read the flag live, so a runtime flip can never leave one side templating what the other
already templated. Flag off: both functions return exactly what they returned before.

Callers whose output must not think (an 80-token review verdict, a 128-token plan-review
JSON) wrap the call in ``thinking_off()``: a per-call ``chat_template_kwargs`` override
(``enable_thinking: false``) carried on the inference request. A no-op with the flag off.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
from pathlib import Path
from typing import Any, Iterator

from src.registry.stack_priors import live_stack_role_records
from src.roles import Role

try:
    from scripts.server.stack_manifest import HOT_SERVERS, ROLE_LAUNCH_META, WARM_SERVERS
except Exception:  # pragma: no cover - catastrophic import fallback
    HOT_SERVERS = ()
    WARM_SERVERS = ()
    ROLE_LAUNCH_META = {}

ENV_VAR = "ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES"


def _live_chat_completions_roles() -> set[str]:
    """Return live roles that the generated priors mark as chat-completions users."""
    try:
        records = live_stack_role_records()
    except Exception:
        return set()

    roles: set[str] = set()
    for role, record in records.items():
        launch = record.get("serving", {}).get("launch", {})
        runtime = launch.get("runtime", {})
        flags = runtime.get("flags", {})
        acceleration = record.get("acceleration", {})
        if (
            isinstance(flags, dict)
            and flags.get("jinja") is True
            and isinstance(acceleration, dict)
            and acceleration.get("enable_thinking") is False
        ):
            roles.add(role)
    return roles


def _launch_primary_for_role(role: str) -> tuple[str | None, dict[str, Any]]:
    """Return the primary launch role and metadata for a raw or canonical role."""
    canonical = Role.from_string(role)
    candidates = [role]
    if canonical is not None and str(canonical) not in candidates:
        candidates.append(str(canonical))

    for candidate in candidates:
        meta = ROLE_LAUNCH_META.get(candidate)
        if isinstance(meta, dict):
            return candidate, meta

    for primary, meta in ROLE_LAUNCH_META.items():
        if not isinstance(meta, dict):
            continue
        shared = meta.get("shared_with_first_n")
        if not isinstance(shared, list):
            continue
        if any(candidate in shared for candidate in candidates):
            return str(primary), meta

    return None, {}


def _degraded_chat_completions_roles() -> set[str]:
    """Derive the narrow fallback set from launch-manifest roles and order."""
    roles: set[str] = set()
    for server in tuple(HOT_SERVERS) + tuple(WARM_SERVERS):
        if not isinstance(server, dict):
            continue
        for role in server.get("roles") or ():
            if not isinstance(role, str):
                continue

            primary, meta = _launch_primary_for_role(role)
            mode = meta.get("mode")
            canonical = Role.from_string(role)
            role_value = str(canonical) if canonical is not None else role

            if mode == "default" and primary == str(Role.FRONTDOOR):
                roles.add(role_value)
            elif mode == "worker_pool" and meta.get("worker_type") == "explore":
                roles.add(role_value)

    return roles


def static_chat_completions_roles() -> set[str]:
    """Roles baked onto a /v1/chat/completions backend at startup (pre-RI-23 set).

    Excludes the flag-admitted thinking roles on purpose: those are admitted per request
    (``thinking_chat_lane_role``) so a live flag flip moves the backend and the template
    skip together.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is not None:
        return {r.strip() for r in raw.split(",") if r.strip()}

    live_roles = _live_chat_completions_roles()
    return live_roles or _degraded_chat_completions_roles()


def chat_completions_roles() -> set[str]:
    """The set of roles that route through /v1/chat/completions (server-side jinja). Read live.

    The template-skip SoT: the static set plus, when ``thinking_roles_chat_lane`` is on, the
    thinking-on roles. Flag off: identical to ``static_chat_completions_roles()``.
    """
    roles = static_chat_completions_roles()
    if thinking_roles_chat_lane_enabled():
        roles |= thinking_chat_lane_roles()
    return roles


# ── RI-23: thinking-on roles on the chat lane ───────────────────────────────

THINKING_OFF_KWARGS: dict[str, Any] = {"enable_thinking": False}

_thinking_roles_cache: tuple[tuple[str, float | None], frozenset[str]] | None = None


def thinking_roles_chat_lane_enabled() -> bool:
    """Live read of the ``thinking_roles_chat_lane`` feature flag (fail-closed: off)."""
    try:
        from src.features import features

        return bool(getattr(features(), "thinking_roles_chat_lane", False))
    except Exception:
        return False


def _priors_cache_key() -> tuple[str, float | None]:
    try:
        from src.registry.stack_priors import DEFAULT_OUTPUT

        path = Path(DEFAULT_OUTPUT)
        try:
            return str(path), path.stat().st_mtime
        except OSError:
            return str(path), None
    except Exception:
        return "", None


def _derive_thinking_chat_lane_roles() -> frozenset[str]:
    try:
        records = live_stack_role_records()
    except Exception:
        return frozenset()

    roles: set[str] = set()
    for role, record in records.items():
        launch = record.get("serving", {}).get("launch", {})
        runtime = launch.get("runtime", {}) if isinstance(launch, dict) else {}
        flags = runtime.get("flags", {}) if isinstance(runtime, dict) else {}
        acceleration = record.get("acceleration", {})
        if (
            isinstance(flags, dict)
            and flags.get("jinja") is True
            and isinstance(acceleration, dict)
            and acceleration.get("enable_thinking") is True
        ):
            roles.add(role)
    return frozenset(roles)


def thinking_chat_lane_roles() -> set[str]:
    """Live thinking-on roles: ``--jinja`` AND ``acceleration.enable_thinking is True``.

    Independent of the flag (callers gate). A role without ``--jinja`` (worker_vision:
    Qwen3-VL Instruct, no thinking toggle) never qualifies — chat_template_kwargs would be
    inert on it. Cached on the priors artifact's mtime: this runs per request.
    """
    global _thinking_roles_cache
    key = _priors_cache_key()
    cached = _thinking_roles_cache
    if cached is not None and cached[0] == key and key[1] is not None:
        return set(cached[1])
    roles = _derive_thinking_chat_lane_roles()
    _thinking_roles_cache = (key, roles)
    return set(roles)


def thinking_chat_lane_role(role: str | None) -> bool:
    """Per-request backend admission: flag on AND ``role`` is a thinking-on role."""
    if not role or not thinking_roles_chat_lane_enabled():
        return False
    roles = thinking_chat_lane_roles()
    if role in roles:
        return True
    canonical = Role.from_string(role)
    return canonical is not None and str(canonical) in roles


_THINKING_OVERRIDE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "chat_template_kwargs_override", default=None
)


@contextlib.contextmanager
def thinking_off() -> Iterator[None]:
    """Run the enclosed inference call(s) with ``enable_thinking: false`` (chat lane only).

    For callers whose output is a short structured artifact (the 80-token review verdict,
    a 128-token plan-review JSON) that thinking would consume. A no-op while
    ``thinking_roles_chat_lane`` is off, so flag-off requests are byte-identical. The
    override rides a ContextVar into the request built by the primitives layer; code that
    hops threads must enter it inside the hop (``asyncio.to_thread`` copies the context, so
    entering it around the ``await`` also works).
    """
    if not thinking_roles_chat_lane_enabled():
        yield
        return
    token = _THINKING_OVERRIDE.set(dict(THINKING_OFF_KWARGS))
    try:
        yield
    finally:
        _THINKING_OVERRIDE.reset(token)


def current_chat_template_kwargs_override() -> dict[str, Any] | None:
    """The per-call ``chat_template_kwargs`` override bound by ``thinking_off()``, or None."""
    value = _THINKING_OVERRIDE.get()
    return dict(value) if isinstance(value, dict) else None
